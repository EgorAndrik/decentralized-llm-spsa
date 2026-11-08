"""Measured task--model outcome tables from public routing benchmarks.

The simulated ``ReplayOracle`` draws every outcome from a hand-set linear
model.  This module replaces it with outcomes that real LLMs actually produced
on real prompts, taken from two public routing benchmarks:

* SPROUT (Somerstep et al., CARROT, 2025): 13 models, judge scores against a
  golden answer, measured input and output token counts.  Prices are the
  table the benchmark authors publish with CARROT.
* RouterBench (Hu et al., 2024): 11 models, graded correctness and the dollar
  cost the benchmark authors recorded for each call.

Both are reduced to the same two tables.  ``items`` holds one row per prompt
with the covariates a controller may observe before execution (domain, prompt
length, code flag, difficulty level where the source provides one) and a
grouped train/validation/test split.  ``outcomes`` holds one row per
(prompt, model) potential outcome.  Nothing in ``outcomes`` is visible to a
controller except for the pair it actually dispatched.

What is measured and what is not is recorded per column in ``PROVENANCE`` and
copied into every run manifest.  Latency is the one quantity neither benchmark
records; it is modelled in ``src.table_oracle`` from the measured token counts.
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import logging
import os
import pickle
import re
from pathlib import Path

import numpy as np
import pandas as pd


logger = logging.getLogger(__name__)

EXTERNAL = Path("data/external")
PROCESSED = EXTERNAL / "processed"

SPROUT_REPO = "CARROT-LLM-Routing/SPROUT"
SPROUT_REVISION = "d54e3cda2189ca1ed13a877a44ed34736c4b6920"
SPROUT_FILES = ("data/validation-00000-of-00001.parquet",
                "data/test-00000-of-00001.parquet")
ROUTERBENCH_REPO = "withmartian/routerbench"
ROUTERBENCH_REVISION = "784021482c3f320c6619ed4b3bb3b41a21424fcb"
ROUTERBENCH_FILE = "routerbench_0shot.pkl"
# SHA-256 of that file at ROUTERBENCH_REVISION (the Hub's LFS object id).
ROUTERBENCH_SHA256 = "ba4f77f19517610a707c374e99322d7750c30fc4ae7ff5527888595a1e65d36d"

# USD per million tokens, [input, output].  Copied verbatim from
# carrot/constants.py (SPROUT_COSTS) at github.com/somerstep/CARROT@3e6acff, the
# price table the SPROUT authors evaluate against.  Using their table rather
# than today's prices keeps cost figures comparable with the published CARROT
# results.
SPROUT_PRICES_PER_MTOK = {
    "aws-claude-3-5-sonnet-v1": (3.0, 15.0),
    "aws-titan-text-premier-v1": (0.8, 3.2),
    "openai-gpt-4o": (2.5, 10.0),
    "openai-gpt-4o-mini": (0.15, 0.6),
    "wxai-granite-3-2b-instruct-8k-max-tokens": (0.1, 0.1),
    "wxai-granite-3-8b-instruct-8k-max-tokens": (0.2, 0.2),
    "wxai-llama-3-1-70b-instruct": (0.9, 0.9),
    "wxai-llama-3-1-8b-instruct": (0.2, 0.2),
    "wxai-llama-3-2-1b-instruct": (0.06, 0.06),
    "wxai-llama-3-2-3b-instruct": (0.06, 0.06),
    "wxai-llama-3-3-70b-instruct": (0.9, 0.9),
    "wxai-mixtral-8x7b-instruct-v01": (0.6, 0.6),
    "wxai-llama-3-405b-instruct": (3.5, 3.5),
}

# Six routing domains per benchmark.  The controller architecture has six
# domain coordinates, so each benchmark's source datasets are grouped into six
# task families by what they ask of a model, never by outcome.
SPROUT_DOMAINS = ("knowledge", "math", "science_reasoning", "instruction",
                  "rag_general", "rag_financial_legal")
ROUTERBENCH_DOMAINS = ("knowledge", "commonsense", "science_reasoning", "math",
                       "code", "open_generation")
LLMROUTERBENCH_DOMAINS = ("knowledge", "math", "code", "software_engineering",
                          "factual_qa", "agentic_reasoning")

# LLMRouterBench (Li et al., 2026), performance--cost setting: frontier models
# called through their APIs, with the tokens and dollar cost of every call.
# The published archive is bench-release.tar.gz at HF NPULH/LLMRouterBench.
LLMROUTERBENCH_REPO = "NPULH/LLMRouterBench"
LLMROUTERBENCH_REVISION = "0e5af1b84bf73437a01a1849c0f1d2468baa93fc"
LLMROUTERBENCH_DATASETS = {
    # directory: domain
    "mmlupro/test_3000": "knowledge",
    "gpqa/test": "knowledge",
    "aime/hybrid": "math",
    "livemathbench/test": "math",
    "livecodebench/test": "code",
    "swe-bench/verified": "software_engineering",
    "simpleqa/test": "factual_qa",
    "hle/test": "factual_qa",
    "tau2/test": "agentic_reasoning",
    "arc-agi/v1": "agentic_reasoning",
}
# The 13 flagship models minus two with a missing dataset: gpt-5-chat has no
# tau2 run and qwen3-235b-a22b-thinking-2507 no SWE-bench run.  Keeping both
# datasets, which are the closest to software-engineering workloads, costs
# those two models.
LLMROUTERBENCH_MODELS = (
    "claude-sonnet-4", "deepseek-r1-0528", "deepseek-v3-0324",
    "deepseek-v3.1-terminus", "gemini-2.5-flash", "gemini-2.5-pro", "glm-4.6",
    "gpt-5", "intern-s1", "kimi-k2-0905", "qwen3-235b-a22b-2507",
)
# OpenRouter's commercial auto-router, run by the benchmark authors on the same
# prompts.  It is a reference system, not a pool member.
LLMROUTERBENCH_REFERENCE = "openrouter"

PROVENANCE = {
    "sprout": {
        "quality": "measured: judge correctness score against golden answer (SPROUT)",
        "success": "derived: quality >= 0.5",
        "input_tokens": "measured (SPROUT)",
        "output_tokens": "measured (SPROUT)",
        "cost_usd": "derived: measured tokens x CARROT SPROUT_COSTS price table",
        "latency": "modelled: TTFT + output_tokens / decode rate (src.table_oracle)",
    },
    "routerbench": {
        "quality": "measured: graded correctness (RouterBench)",
        "success": "derived: quality >= 0.5",
        "input_tokens": "estimated: prompt characters / 4",
        "output_tokens": "estimated: response characters / 4",
        "cost_usd": "measured: total_cost recorded by RouterBench",
        "latency": "modelled: TTFT + output_tokens / decode rate (src.table_oracle)",
    },
    "llmrouterbench": {
        "quality": "measured: benchmark score (execution tests for code/SWE-bench, exact match, "
                   "task success for tau2) (LLMRouterBench)",
        "success": "derived: quality >= 0.5",
        "input_tokens": "measured (LLMRouterBench)",
        "output_tokens": "measured, including reasoning tokens (LLMRouterBench)",
        "cost_usd": "measured: API cost recorded by LLMRouterBench",
        "latency": "modelled: TTFT + output_tokens / decode rate (src.table_oracle)",
        "reference_router": "measured: OpenRouter auto-router on the same prompts (LLMRouterBench)",
    },
}

CODE_PATTERN = re.compile(r"```|\bdef |\bclass |#include|;\s*$|\breturn\b|=>", re.MULTILINE)


def _split_of(item_id: str) -> str:
    """Grouped 60/20/20 split keyed by the prompt identity alone."""
    digest = int(hashlib.sha256(f"split|{item_id}".encode()).hexdigest()[:8], 16)
    fraction = digest / 0xFFFFFFFF
    if fraction < 0.60:
        return "train"
    if fraction < 0.80:
        return "validation"
    return "test"


def _sprout_domain(dataset: str) -> str:
    if dataset.startswith("TIGER-Lab/MMLU-Pro"):
        return "knowledge"
    if dataset.startswith("lighteval/MATH"):
        return "math"
    if dataset.startswith(("Idavidrein/gpqa", "TAUR-Lab/MuSR")):
        return "science_reasoning"
    if dataset.startswith("openhermes"):
        return "instruction"
    if dataset.startswith("rungalileo/ragbench/") and dataset.rsplit("/", 1)[-1] in {"finqa", "tatqa", "cuad"}:
        return "rag_financial_legal"
    if dataset.startswith("rungalileo/ragbench/"):
        return "rag_general"
    raise ValueError(f"unmapped SPROUT dataset: {dataset}")


def _routerbench_domain(eval_name: str) -> str:
    if eval_name.startswith("mmlu-"):
        return "knowledge"
    if eval_name in {"hellaswag", "winogrande"}:
        return "commonsense"
    if eval_name == "arc-challenge":
        return "science_reasoning"
    if eval_name in {"grade-school-math", "mtbench-math", "chinese-remainder-theorem"}:
        return "math"
    if eval_name == "mbpp":
        return "code"
    return "open_generation"


def _level(value: str) -> float:
    """Source-declared difficulty in [0, 1], or NaN when the source has none."""
    match = re.search(r"Level (\d)", str(value or ""))
    return (int(match.group(1)) - 1) / 4.0 if match else float("nan")


def build_sprout(root: Path = EXTERNAL) -> tuple[pd.DataFrame, pd.DataFrame]:
    frames = [pd.read_parquet(root / "sprout" / name) for name in SPROUT_FILES]
    raw = pd.concat(frames, ignore_index=True).drop_duplicates("key")
    models = list(SPROUT_PRICES_PER_MTOK)
    items = pd.DataFrame({
        "item_id": "sprout:" + raw["key"].astype(str),
        "source_dataset": raw["dataset"].astype(str),
        "domain_name": raw["dataset"].map(_sprout_domain),
        "level": raw["dataset_level"].map(_level),
        "prompt_chars": raw["prompt"].str.len().astype(int),
        "has_code": raw["prompt"].map(lambda text: bool(CODE_PATTERN.search(str(text)))),
        "prompt": raw["prompt"].astype(str),
    })
    rows = []
    for model in models:
        cell = pd.json_normalize(raw[model].tolist())
        price_in, price_out = SPROUT_PRICES_PER_MTOK[model]
        rows.append(pd.DataFrame({
            "item_id": items["item_id"].values,
            "model": model,
            "quality": cell["score"].astype(float).clip(0.0, 1.0).values,
            "input_tokens": cell["num_input_tokens"].astype(float).values,
            "output_tokens": cell["num_output_tokens"].astype(float).values,
            "cost_usd": (cell["num_input_tokens"] * price_in
                         + cell["num_output_tokens"] * price_out).astype(float).values / 1e6,
        }))
    return _finish(items, pd.concat(rows, ignore_index=True), SPROUT_DOMAINS)


class _RestrictedUnpickler(pickle.Unpickler):
    """Unpickler that resolves only an exact list of pandas/numpy globals.

    The RouterBench table is a pickle downloaded from a third-party repository;
    plain ``pd.read_pickle`` would execute whatever code such a file carries.
    An earlier version allowed whole module prefixes, and a payload calling
    ``DataFrame.to_csv`` could then write a file.  Only the listed names are
    resolved now and dotted attribute paths are refused.  The primary control
    is the SHA-256 check in ``read_trusted_dataframe``; this list is a second
    layer, not a sandbox.
    """

    _NUMPY = ("numpy", "numpy.core.multiarray", "numpy._core.multiarray",
              "numpy.core.numeric", "numpy._core.numeric")
    _NAMES = frozenset(
        {(m, n) for m in _NUMPY for n in ("dtype", "ndarray", "_reconstruct", "scalar", "_frombuffer")}
        | {
            ("pandas", "DataFrame"), ("pandas", "Series"), ("pandas", "Index"),
            ("pandas", "MultiIndex"), ("pandas", "RangeIndex"), ("pandas", "StringDtype"),
            ("pandas", "Timestamp"), ("pandas", "Timedelta"),
            ("pandas.core.frame", "DataFrame"), ("pandas.core.series", "Series"),
            ("pandas.core.indexes.base", "Index"), ("pandas.core.indexes.base", "_new_Index"),
            ("pandas.core.indexes.range", "RangeIndex"), ("pandas.core.indexes.multi", "MultiIndex"),
            ("pandas.core.indexes.numeric", "Int64Index"), ("pandas.core.indexes.numeric", "Float64Index"),
            ("pandas.core.indexes.datetimes", "DatetimeIndex"),
            ("pandas.core.indexes.datetimes", "_new_DatetimeIndex"),
            ("pandas.core.internals.managers", "BlockManager"),
            ("pandas.core.internals.managers", "SingleBlockManager"),
            ("pandas._libs.internals", "_unpickle_block"), ("pandas._libs.internals", "BlockPlacement"),
            ("pandas._libs.arrays", "__pyx_unpickle_NDArrayBacked"),
            ("pandas.arrays", "DatetimeArray"), ("pandas.arrays", "ArrowStringArray"),
            ("pandas._libs.tslibs.timestamps", "_unpickle_timestamp"),
            ("pandas._libs.tslibs.timedeltas", "_timedelta_unpickle"),
            ("pyarrow.lib", "_restore_array"), ("pyarrow.lib", "py_buffer"),
            ("pyarrow.lib", "type_for_alias"),
            ("builtins", "slice"), ("builtins", "range"), ("builtins", "set"),
            ("builtins", "frozenset"), ("builtins", "complex"), ("builtins", "bytearray"),
            ("collections", "OrderedDict"), ("_codecs", "encode"),
            ("datetime", "datetime"), ("datetime", "date"), ("datetime", "timedelta"),
        }
    )

    def find_class(self, module, name):
        if "." not in name and (module, name) in self._NAMES:
            return super().find_class(module, name)
        raise pickle.UnpicklingError(f"blocked global {module}.{name} in pickle")


def read_trusted_dataframe(path: Path, sha256: str | None = None) -> pd.DataFrame:
    """Load a DataFrame pickle without executing arbitrary code.

    With ``sha256`` the file must hash to exactly that value, so only the
    published file is ever unpickled; the bytes that were hashed are the bytes
    that are loaded.  Set ROUTERBENCH_UNSAFE_PICKLE=1 to skip every check and
    use ``pd.read_pickle`` on a file you have verified yourself.
    """
    if os.environ.get("ROUTERBENCH_UNSAFE_PICKLE") == "1":
        logger.warning("Loading %s with unrestricted pickle (ROUTERBENCH_UNSAFE_PICKLE=1)", path)
        return pd.read_pickle(path)
    with open(path, "rb") as handle:
        data = handle.read()
    if sha256 is not None:
        digest = hashlib.sha256(data).hexdigest()
        if digest != sha256.lower():
            raise ValueError(
                f"{path}: SHA-256 {digest} does not match the pinned {sha256}; refusing to unpickle. "
                "Re-download the file from the pinned revision."
            )
    frame = _RestrictedUnpickler(io.BytesIO(data)).load()
    if not isinstance(frame, pd.DataFrame):
        raise ValueError(f"{path} does not contain a DataFrame")
    return frame


def build_routerbench(root: Path = EXTERNAL) -> tuple[pd.DataFrame, pd.DataFrame]:
    raw = read_trusted_dataframe(root / "routerbench" / ROUTERBENCH_FILE, ROUTERBENCH_SHA256)
    raw = raw.drop_duplicates("sample_id")
    models = [c for c in raw.columns
              if "|" not in c and c not in {"sample_id", "prompt", "eval_name", "oracle_model_to_route_to"}]
    prompt_text = raw["prompt"].astype(str)
    items = pd.DataFrame({
        "item_id": "routerbench:" + raw["sample_id"].astype(str),
        "source_dataset": raw["eval_name"].astype(str),
        "domain_name": raw["eval_name"].map(_routerbench_domain),
        "level": float("nan"),
        "prompt_chars": prompt_text.str.len().astype(int),
        "has_code": prompt_text.map(lambda text: bool(CODE_PATTERN.search(text))),
        "prompt": prompt_text,
    })
    rows = []
    for model in models:
        response = raw[f"{model}|model_response"].astype(str)
        rows.append(pd.DataFrame({
            "item_id": items["item_id"].values,
            "model": model,
            "quality": pd.to_numeric(raw[model], errors="coerce").clip(0.0, 1.0).values,
            "input_tokens": (prompt_text.str.len() / 4.0).values,
            "output_tokens": (response.str.len() / 4.0).values,
            "cost_usd": pd.to_numeric(raw[f"{model}|total_cost"], errors="coerce").values,
        }))
    return _finish(items, pd.concat(rows, ignore_index=True), ROUTERBENCH_DOMAINS)


def _read_run(directory: Path) -> list[dict]:
    files = sorted(directory.glob("*.json"))
    if len(files) != 1:
        raise ValueError(f"expected one result file in {directory}, found {len(files)}")
    return json.loads(files[0].read_text(encoding="utf-8"))["records"]


def _number(value) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


def build_llmrouterbench(root: Path = EXTERNAL):
    base = root / "llmrouterbench" / "bench-release"
    items, rows, reference = [], [], []
    for directory, domain in LLMROUTERBENCH_DATASETS.items():
        per_model = {m: _read_run(base / directory / m) for m in LLMROUTERBENCH_MODELS}
        first = per_model[LLMROUTERBENCH_MODELS[0]]
        for record in first:
            text = str(record.get("origin_query") or record.get("prompt") or "")
            items.append({
                "item_id": f"llmrouterbench:{directory}:{record['index']}",
                "source_dataset": directory,
                "domain_name": domain,
                "level": float("nan"),
                "prompt_chars": len(text),
                "has_code": bool(CODE_PATTERN.search(text)),
                "prompt": text,
            })
        for model, records in per_model.items():
            for record in records:
                rows.append({
                    "item_id": f"llmrouterbench:{directory}:{record['index']}",
                    "model": model,
                    "quality": min(max(_number(record["score"]), 0.0), 1.0),
                    "input_tokens": _number(record["prompt_tokens"]),
                    "output_tokens": _number(record["completion_tokens"]),
                    "cost_usd": _number(record["cost"]),
                })
        ref_dir = base / directory / LLMROUTERBENCH_REFERENCE
        if ref_dir.exists():
            for record in _read_run(ref_dir):
                extra = record.get("extra_fields")
                if isinstance(extra, str):
                    match = re.search(r"'actual_model':\s*'([^']+)'", extra)
                    chosen = match.group(1) if match else ""
                else:
                    chosen = (extra or {}).get("actual_model", "")
                reference.append({
                    "item_id": f"llmrouterbench:{directory}:{record['index']}",
                    "router": "openrouter_auto",
                    "quality": min(max(_number(record["score"]), 0.0), 1.0),
                    "cost_usd": _number(record["cost"]),
                    "output_tokens": _number(record["completion_tokens"]),
                    "chosen_model": chosen,
                })
    items, outcomes = _finish(pd.DataFrame(items), pd.DataFrame(rows), LLMROUTERBENCH_DOMAINS)
    reference = pd.DataFrame(reference)
    reference = reference[reference["item_id"].isin(items["item_id"])].reset_index(drop=True)
    return items, outcomes, reference


def _complete_outcomes(outcomes: pd.DataFrame) -> pd.DataFrame:
    # A prompt with a missing outcome for any model cannot be replayed as a
    # full counterfactual row, so it is dropped rather than imputed.
    metrics = ["quality", "input_tokens", "output_tokens", "cost_usd"]
    values = outcomes[metrics].to_numpy(dtype=float, na_value=np.nan)
    complete = outcomes[np.isfinite(values).all(axis=1) & (values >= 0).all(axis=1)]
    counts = complete.groupby("item_id")["model"].nunique()
    keep = counts[counts == outcomes["model"].nunique()].index
    return complete[complete["item_id"].isin(keep)].reset_index(drop=True)


def _finish(items: pd.DataFrame, outcomes: pd.DataFrame, domains: tuple[str, ...]):
    outcomes = _complete_outcomes(outcomes)
    items = items[items["item_id"].isin(outcomes["item_id"])].reset_index(drop=True)
    items["domain"] = items["domain_name"].map({name: i for i, name in enumerate(domains)}).astype(int)
    items["split"] = items["item_id"].map(_split_of)
    outcomes["success"] = outcomes["quality"] >= 0.5
    return items, outcomes


BUILDERS = {"sprout": build_sprout, "routerbench": build_routerbench,
            "llmrouterbench": build_llmrouterbench}
DOMAINS = {"sprout": SPROUT_DOMAINS, "routerbench": ROUTERBENCH_DOMAINS,
           "llmrouterbench": LLMROUTERBENCH_DOMAINS}
REVISIONS = {
    "sprout": {"repo": SPROUT_REPO, "revision": SPROUT_REVISION, "files": list(SPROUT_FILES),
               "prices": "github.com/somerstep/CARROT@3e6acff carrot/constants.py SPROUT_COSTS"},
    "routerbench": {"repo": ROUTERBENCH_REPO, "revision": ROUTERBENCH_REVISION,
                    "files": [ROUTERBENCH_FILE]},
    "llmrouterbench": {"repo": LLMROUTERBENCH_REPO, "revision": LLMROUTERBENCH_REVISION,
                       "files": ["bench-release.tar.gz"],
                       "datasets": LLMROUTERBENCH_DATASETS,
                       "models": list(LLMROUTERBENCH_MODELS)},
}


def download(name: str, root: Path = EXTERNAL) -> None:
    from huggingface_hub import hf_hub_download
    spec = REVISIONS[name]
    for filename in spec["files"]:
        path = hf_hub_download(spec["repo"], filename, repo_type="dataset",
                               revision=spec["revision"], local_dir=root / name)
        if filename.endswith(".tar.gz"):
            import tarfile
            with tarfile.open(path) as archive:
                archive.extractall(root / name, filter="data")


def load(name: str, root: Path = EXTERNAL) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Processed tables, built on first use and cached as parquet."""
    out = root / "processed"
    items_path, outcomes_path = out / f"{name}_items.parquet", out / f"{name}_outcomes.parquet"
    if items_path.exists() and outcomes_path.exists():
        outcomes = _complete_outcomes(pd.read_parquet(outcomes_path))
        items = pd.read_parquet(items_path)
        return items[items["item_id"].isin(outcomes["item_id"])].reset_index(drop=True), outcomes
    built = BUILDERS[name](root)
    items, outcomes = built[0], built[1]
    out.mkdir(parents=True, exist_ok=True)
    if len(built) > 2:
        built[2].to_parquet(out / f"{name}_reference.parquet", index=False)
    items.to_parquet(items_path, index=False)
    outcomes.to_parquet(outcomes_path, index=False)
    audit(name, items, outcomes, out / f"{name}_audit.json")
    return items, outcomes


def load_reference(name: str, root: Path = EXTERNAL) -> pd.DataFrame | None:
    """Outcomes of an external reference router on the same prompts, if any."""
    path = root / "processed" / f"{name}_reference.parquet"
    if not path.exists():
        load(name, root)
    return pd.read_parquet(path) if path.exists() else None


def audit(name: str, items: pd.DataFrame, outcomes: pd.DataFrame, path: Path | None = None) -> dict:
    """Descriptive statistics a reader needs to judge the table.

    ``oracle_gap`` is the headroom any router has: the mean per-prompt best
    quality minus the best single model.  ``domain_rank_changes`` counts how
    often the best model differs between domains, which is the premise of
    domain-specialised controllers.
    """
    merged = outcomes.merge(items[["item_id", "domain_name", "split"]], on="item_id")
    per_model = merged.groupby("model").agg(
        quality=("quality", "mean"), cost_usd=("cost_usd", "mean"),
        output_tokens=("output_tokens", "median"))
    by_domain = merged.pivot_table(index="domain_name", columns="model", values="quality", aggfunc="mean")
    oracle = merged.groupby("item_id")["quality"].max().mean()
    best_by_domain = by_domain.idxmax(axis=1).to_dict()
    report = {
        "benchmark": name,
        "source": REVISIONS[name],
        "provenance": PROVENANCE[name],
        "items": int(len(items)),
        "models": int(outcomes["model"].nunique()),
        "items_per_split": items["split"].value_counts().to_dict(),
        "items_per_domain": items["domain_name"].value_counts().to_dict(),
        "best_single_model": str(per_model["quality"].idxmax()),
        "best_single_quality": float(per_model["quality"].max()),
        "oracle_quality": float(oracle),
        "oracle_gap": float(oracle - per_model["quality"].max()),
        "best_model_by_domain": best_by_domain,
        "distinct_domain_winners": int(len(set(best_by_domain.values()))),
        "per_model": per_model.round(6).to_dict(orient="index"),
        "quality_by_domain": by_domain.round(4).to_dict(orient="index"),
    }
    if path is not None:
        path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("benchmark", choices=sorted(BUILDERS))
    parser.add_argument("--download", action="store_true")
    parser.add_argument("--rebuild", action="store_true")
    args = parser.parse_args()
    if args.download:
        download(args.benchmark)
    if args.rebuild:
        for suffix in ("items", "outcomes"):
            (PROCESSED / f"{args.benchmark}_{suffix}.parquet").unlink(missing_ok=True)
    items, outcomes = load(args.benchmark)
    report = audit(args.benchmark, items, outcomes, PROCESSED / f"{args.benchmark}_audit.json")
    print(json.dumps({k: v for k, v in report.items() if k not in {"per_model", "quality_by_domain"}}, indent=2))


if __name__ == "__main__":
    main()
