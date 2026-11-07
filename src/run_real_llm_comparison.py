"""
Real-LLM validation: all 7 SPSA variants on a shared execution cache.

Architecture
------------
Phase 1 — Pre-collect (one pass, all agents):
    For every task send it to all LLM agents; store outputs, latencies,
    and token counts in a PromptCachingLLMClient.

Phase 2 — Replay (all variants, zero extra API calls):
    Each variant runs with the same caching client.  When the orchestrator
    asks agent k to process task t, the client returns the pre-recorded
    response instantly.  The first variant warms the judge cache, so
    subsequent variants pay zero judge API calls too.

API call budget (per seed, 200 tasks, 5 agents)
    Pre-collect : N_tasks × N_agents              = 200 × 5 = 1000 calls
    Variant 1   : 200 agent hits + 200 judge miss =           200 calls
    Variants 2–7: all hits                        =             0 calls
    Total                                                     1200 calls

Usage
-----
    # verify keys / models first
    python -m src.run_real_llm_comparison --dry-run

    # full run — Groq + OpenRouter, 200 tasks, 3 seeds
    python -m src.run_real_llm_comparison --config llm_config_openrouter \\
        --tasks 200 --seeds 11,42,123 --output-dir real_llm_outputs_or

    # original Groq+OpenAI config
    python -m src.run_real_llm_comparison --tasks 200 --output-dir real_llm_outputs_v5
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# Windows cp1251 terminals can't print Unicode arrows/subscripts; force UTF-8.
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

try:
    from .main import ExactOrchestrator, SimulationVariant, PROFILES
    from .llm_api import LLMAPIClient, BudgetExceeded, import_config_module
    from .run_spsa_comparison import (
        SPSA_VARIANTS, generate_tasks, BEST_PARAMS, _build_variant,
    )
    from . import spsa_variants as _spsa_mod
except ImportError:
    import sys as _sys
    _sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from main import ExactOrchestrator, SimulationVariant, PROFILES
    from llm_api import LLMAPIClient, BudgetExceeded, import_config_module
    from run_spsa_comparison import (
        SPSA_VARIANTS, generate_tasks, BEST_PARAMS, _build_variant,
    )
    import spsa_variants as _spsa_mod

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Hard-coded best hyperparams from simulation tuning (Table 1 in paper).
# Avoids re-running grid search on live API calls.
# ---------------------------------------------------------------------------
PAPER_PARAMS: Dict[str, dict] = {
    "aspsa":   dict(alpha=0.10, beta=0.30, beta_nes_max=0.30),
    "spsa":    dict(alpha=0.70, beta=0.05, beta_nes_max=1.0),
    "kw":      dict(alpha=1.00, beta=0.30, beta_nes_max=1.0),
    "zo_pgd":  dict(alpha=0.50, beta=0.10, beta_nes_max=1.0),
    "sp_gt":   dict(alpha=0.10, beta=0.30, beta_nes_max=1.0),
    "zo_gt":   dict(alpha=0.10, beta=0.30, beta_nes_max=1.0),
    "pd_2pt":  dict(alpha=0.05, beta=0.10, beta_nes_max=1.0),
}

COLORS = {
    "aspsa":  "#d62728",
    "spsa":   "#1f77b4",
    "kw":     "#2ca02c",
    "zo_pgd": "#ff7f0e",
    "sp_gt":  "#9467bd",
    "zo_gt":  "#8c564b",
    "pd_2pt": "#17becf",
}
LABELS = {
    "aspsa":  "A-SPSA",
    "spsa":   "SPSA",
    "kw":     "KW",
    "zo_pgd": "ZO-PGD",
    "sp_gt":  "SP-GT",
    "zo_gt":  "ZO-GT",
    "pd_2pt": "PD-2pt",
}


# ---------------------------------------------------------------------------
# Rate-limited raw client
# ---------------------------------------------------------------------------

class _RateLimitedClient:
    """
    Enforces a minimum gap between consecutive API calls and retries on 429
    with exponential back-off (so transient free-tier throttling does not
    corrupt the cache with empty fallback results).
    """

    def __init__(
        self,
        client: LLMAPIClient,
        min_interval_s: float = 2.5,
        max_retries: int = 4,
        retry_base_s: float = 25.0,
    ):
        self._client = client
        self._min_interval = min_interval_s
        self._last = 0.0
        self._max_retries = max_retries
        self._retry_base = retry_base_s

    def call(self, model_name: str, prompt: str, task_type: str) -> dict:
        gap = time.perf_counter() - self._last
        if gap < self._min_interval:
            time.sleep(self._min_interval - gap)

        wait = self._retry_base
        for attempt in range(self._max_retries + 1):
            try:
                result = self._client.call(model_name, prompt, task_type)
                self._last = time.perf_counter()
                return result
            except Exception as exc:
                is_rate_limit = "429" in str(exc) or "Too Many Requests" in str(exc)
                if not is_rate_limit or attempt == self._max_retries:
                    raise
                logger.warning(
                    "Rate-limited (attempt %d/%d) — waiting %.0f s ...",
                    attempt + 1, self._max_retries, wait,
                )
                time.sleep(wait)
                wait = min(wait * 2, 300.0)  # cap at 5 min

        raise RuntimeError("unreachable")


# ---------------------------------------------------------------------------
# Prompt-caching LLM client — the core of the shared-execution architecture
# ---------------------------------------------------------------------------

class PromptCachingLLMClient:
    """
    Wraps a real LLM client with a (model_name, prompt) → result cache.

    Phase 1 — pre_collect():
        Calls all agents for all tasks and warms the cache.

    Phase 2 — call() (used by ExactOrchestrator):
        Returns cached results instantly; falls back to live API only on
        genuine cache misses (should not happen after pre_collect).

    Because the judge's prompt embeds the model output, judge calls are also
    automatically cached after the first variant run warms them.
    """

    def __init__(self, raw_client: _RateLimitedClient):
        self._raw = raw_client
        self._cache: Dict[Tuple[str, str], dict] = {}
        self.hits = 0
        self.misses = 0

    def _key(self, model_name: str, prompt: str) -> Tuple[str, str]:
        # The whole prompt. The key used to be cut to 400 characters, so prompts
        # that shared a prefix (a judge prompt differs from another only in the
        # model output at its end) received each other's cached answers.
        # Caches saved with the old truncated keys still hit for prompts up to
        # 400 characters and miss for longer ones, which are requested again.
        return (model_name, prompt)

    def call(self, model_name: str, prompt: str, task_type: str) -> dict:
        key = self._key(model_name, prompt)
        if key in self._cache:
            self.hits += 1
            return dict(self._cache[key])
        # Live call (cache miss)
        self.misses += 1
        try:
            result = self._raw.call(model_name, prompt, task_type)
        except BudgetExceeded:
            # A spending cap must stop the run, not be absorbed as one more fallback.
            raise
        except Exception as exc:
            logger.warning("API call failed (%s) — using fallback result.", exc)
            result = {"output": "", "success": False, "latency": 5.0}
            result["_api_fallback"] = True
        # Never cache fallback results — a stale failure would corrupt all 7 variants.
        if not result.get("_api_fallback"):
            self._cache[key] = result
        return dict(result)

    def set_latency_profiles(self, profiles: dict, rng: "np.random.RandomState") -> None:
        """Approach A: inject simulated latency profiles into cached results."""
        self._latency_profiles = profiles
        self._latency_rng = rng

    def _inject_latency(self, model_name: str, result: dict) -> dict:
        """Replace API latency with a sample from the model's simulated profile."""
        profiles = getattr(self, "_latency_profiles", {})
        if model_name in profiles and not result.get("_api_fallback"):
            mean, std = profiles[model_name]
            result = dict(result)
            result["latency"] = float(max(0.1, self._latency_rng.normal(mean, std)))
        return result

    def pre_collect(self, tasks: list, model_names: List[str], max_fallback_rate: float = 0.05) -> None:
        """
        Phase 1: call every (task, agent) pair once to warm the cache.
        Total API calls = len(tasks) × len(model_names).
        Tolerates isolated 400/content-filter errors (up to max_fallback_rate);
        aborts only if a systematic failure (rate limit exhaustion) is detected.
        """
        total = len(tasks) * len(model_names)
        done = 0
        fallbacks = 0
        for task in tasks:
            prompt = task.text or (
                f"{task.type.name}: complexity={task.h_Ti:.2f}, urgency={task.urgency:.2f}"
            )
            for model_name in model_names:
                key = self._key(model_name, prompt)
                if key not in self._cache:
                    result = self.call(model_name, prompt, task.type.name)
                    if result.get("_api_fallback"):
                        fallbacks += 1
                        fallback_rate = fallbacks / max(1, done + 1)
                        # Tolerate isolated content-filter / transient 400 errors.
                        # Abort only if fallback rate exceeds threshold (systematic failure).
                        if fallback_rate > max_fallback_rate:
                            raise RuntimeError(
                                f"Model '{model_name}' fallback rate {fallback_rate:.1%} "
                                f"exceeds {max_fallback_rate:.0%} on call {done+1}/{total}. "
                                "Likely rate limit or broken endpoint — aborting."
                            )
                        logger.warning(
                            "Skipping fallback for model '%s' call %d/%d "
                            "(total fallbacks: %d, rate: %.1f%%)",
                            model_name, done + 1, total, fallbacks, fallback_rate * 100,
                        )
                    else:
                        # Approach A: overwrite latency with simulated profile value
                        if key in self._cache:
                            self._cache[key] = self._inject_latency(model_name, self._cache[key])
                done += 1
                if done % 10 == 0 or done == total:
                    logger.info(
                        "  Pre-collect %d/%d  (cache size=%d)", done, total, len(self._cache)
                    )

    def cache_stats(self) -> dict:
        return {
            "cache_size": len(self._cache),
            "hits": self.hits,
            "misses": self.misses,
            "hit_rate": self.hits / max(1, self.hits + self.misses),
        }

    def save(self, path: Path) -> None:
        path.write_text(
            json.dumps(
                {f"{k[0]}|||{k[1]}": v for k, v in self._cache.items()},
                indent=2, default=str,
            ),
            encoding="utf-8",
        )

    @classmethod
    def load(cls, path: Path, raw_client: _RateLimitedClient) -> "PromptCachingLLMClient":
        obj = cls(raw_client)
        data = json.loads(path.read_text(encoding="utf-8"))
        for compound_key, result in data.items():
            model, prompt = compound_key.split("|||", 1)
            obj._cache[(model, prompt)] = result
        return obj


# ---------------------------------------------------------------------------
# API config + client factory
# ---------------------------------------------------------------------------

def _load_api_config(config_name: str = "llm_config") -> Tuple[dict, dict, dict, str, dict, dict]:
    cfg = import_config_module(config_name, __package__)
    logger.info("Loaded API config from module: %s", config_name)
    return (
        getattr(cfg, "LLM_ENDPOINTS", {}),
        getattr(cfg, "LLM_API_KEYS", {}),
        getattr(cfg, "LLM_MODEL_ALIASES", {}),
        getattr(cfg, "JUDGE_MODEL_NAME", "qwen2.5-72b-instruct"),
        getattr(cfg, "LATENCY_PROFILES", {}),
        getattr(cfg, "MAX_TOKENS_PER_MODEL", {}),
    )


def _build_clients(
    min_interval_s: float = 2.5,
    config_name: str = "llm_config",
    max_calls: Optional[int] = None,
) -> Tuple[PromptCachingLLMClient, str, dict]:
    endpoints, api_keys, aliases, judge_model, latency_profiles, max_tokens = _load_api_config(config_name)
    raw_base = LLMAPIClient(
        endpoints, api_keys, model_aliases=aliases,
        max_tokens_per_model=max_tokens, timeout=90,
        max_calls=max_calls,
    )
    raw = _RateLimitedClient(raw_base, min_interval_s=min_interval_s)
    return PromptCachingLLMClient(raw), judge_model, latency_profiles


# ---------------------------------------------------------------------------
# Dry-run
# ---------------------------------------------------------------------------

def dry_run(caching_client: PromptCachingLLMClient, judge_model: str) -> None:
    """Test judge + all 5 agent models so rate-limit issues are caught before a full run."""
    from .main import PROFILES  # noqa: F401 (already imported at top, just for clarity)
    agent_models = [p.model_name for p in PROFILES]
    all_models = [judge_model] + [m for m in agent_models if m != judge_model]

    print(f"Dry-run: testing {len(all_models)} models ...")
    any_failed = False
    for model in all_models:
        result = caching_client.call(model, "Reply with the single word OK.", "TEST")
        ok = "FAILED" if result.get("_api_fallback") else "OK"
        out = result.get("output", "")[:50]
        lat = result.get("latency", 0)
        tag = "(judge)" if model == judge_model else "(agent)"
        print(f"  {model} {tag}: {ok}  out={out!r}  lat={lat:.2f}s")
        if result.get("_api_fallback"):
            any_failed = True
    print()
    if any_failed:
        print("WARNING: one or more models returned fallback — check rate limits before running.")


# ---------------------------------------------------------------------------
# F2 one-sided hinge wait loss
# ---------------------------------------------------------------------------

_F2_DELTA = 0.1  # dead-band margin, matches spsa_variants._wait_loss and system_model.tex

def _f2_loss(records: List[dict]) -> float:
    vals = [
        max(0.0, float(r["true_wait"]) + _F2_DELTA - float(r["predicted_wait"])) ** 2
        for r in records
        if r.get("true_wait") is not None and r.get("predicted_wait") is not None
    ]
    return float(np.mean(vals)) if vals else float("nan")


# ---------------------------------------------------------------------------
# Single variant run (uses caching_client — may be pure replay)
# ---------------------------------------------------------------------------

def _run_one_variant(
    variant_name: str,
    tasks: list,
    caching_client: PromptCachingLLMClient,
    judge_model: str,
    seed: int,
    num_ctrl: int,
    num_agents: int,
    collect_trace: bool = False,
    spsa_interval: int = 5,
) -> Tuple[dict, list, list]:
    p = PAPER_PARAMS[variant_name]
    variant = SimulationVariant(
        name=variant_name,
        routing_mode="adaptive",
        use_spsa=True,
        use_judge=True,
        spsa_variant=variant_name,
        spsa_alpha=p["alpha"],
        spsa_beta=p["beta"],
        spsa_beta_nes_max=p["beta_nes_max"],
        spsa_interval=spsa_interval,
    )
    np.random.seed(seed)

    if collect_trace:
        _spsa_mod.enable_trace(True)
        _spsa_mod.clear_trace()
    else:
        _spsa_mod.enable_trace(False)

    logger.info(
        "    Hyperparams: alpha=%.3f  beta=%.3f  beta_nes_max=%.3f",
        p["alpha"], p["beta"], p["beta_nes_max"],
    )

    orch = ExactOrchestrator(
        num_ctrl=num_ctrl,
        num_agents=num_agents,
        llm_api_client=caching_client,
        judge_model_name=judge_model,
        variant=variant,
        classifier_seed=seed,
    )
    orch.run(list(tasks))
    summary = orch.collect_summary()
    records = list(orch.task_records)
    trace   = _spsa_mod.get_trace() if collect_trace else []

    for r in records:
        r["variant"] = variant_name
        r["seed"] = seed
        r["api_fallback"] = int(r.get("api_fallback") or r.get("_api_fallback") or 0)

    summary.update(
        variant=variant_name,
        seed=seed,
        f2_wait_loss=_f2_loss(records),
        api_fallback_rate=sum(r.get("api_fallback", 0) for r in records) / max(1, len(records)),
    )
    return summary, records, trace


# ---------------------------------------------------------------------------
# Aggregate helpers
# ---------------------------------------------------------------------------

METRIC_KEYS = [
    "routing_objective", "deadline_hit_rate", "f2_wait_loss",
    "success_rate", "mean_latency", "total_cost", "api_fallback_rate",
]


def _aggregate(summaries: List[dict]) -> dict:
    agg = {"variant": summaries[0]["variant"]}
    for k in METRIC_KEYS:
        vals = [s[k] for s in summaries if k in s and s[k] is not None]
        if vals:
            agg[f"{k}_mean"] = float(np.mean(vals))
            agg[f"{k}_std"]  = float(np.std(vals, ddof=1) if len(vals) > 1 else 0.0)
    return agg


# ---------------------------------------------------------------------------
# Figures — publication quality
# ---------------------------------------------------------------------------

_FONT = {"family": "DejaVu Sans", "size": 10}
_DPI  = 200


def _rolling_curves(
    records: List[dict], key: str, window: int
) -> tuple:
    """Return (mean_curve, std_curve) averaged over seeds."""
    by_seed: Dict[int, list] = {}
    for r in records:
        val = r.get(key)
        by_seed.setdefault(r["seed"], []).append(float(val) if val is not None else 0.0)
    if not by_seed:
        return np.array([]), np.array([])
    curves = [
        pd.Series(vals).rolling(window, min_periods=1).mean().values
        for vals in by_seed.values()
    ]
    max_len = max(len(c) for c in curves)
    mat = np.full((len(curves), max_len), np.nan)
    for i, c in enumerate(curves):
        mat[i, : len(c)] = c
    return np.nanmean(mat, axis=0), np.nanstd(mat, axis=0)


def _plot_bar(agg_rows: List[dict], out_path: Path) -> None:
    """Figure 1: grouped bar chart of summary metrics with error bars."""
    metrics = [
        ("routing_objective", "Routing objective ↑"),
        ("deadline_hit_rate", "Deadline hit rate ↑"),
        ("f2_wait_loss",      "F₂ wait loss ↓"),
    ]
    plt.rc("font", **_FONT)
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))

    for ax, (key, label) in zip(axes, metrics):
        names  = [LABELS.get(r["variant"], r["variant"]) for r in agg_rows]
        means  = [r.get(f"{key}_mean", 0.0) for r in agg_rows]
        stds   = [r.get(f"{key}_std",  0.0) for r in agg_rows]
        colors = [COLORS.get(r["variant"], "#888888") for r in agg_rows]
        lws    = [2.5 if r["variant"] == "aspsa" else 0.8 for r in agg_rows]
        edges  = ["#111111" if r["variant"] == "aspsa" else c
                  for r, c in zip(agg_rows, colors)]

        ax.bar(names, means, yerr=stds, color=colors,
               edgecolor=edges, linewidth=lws, capsize=4,
               error_kw={"elinewidth": 1.2, "alpha": 0.8})
        ax.set_title(label, fontsize=10, pad=6)
        ax.tick_params(axis="x", labelrotation=35, labelsize=8)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.grid(axis="y", linewidth=0.4, alpha=0.5)

    fig.suptitle("Real-LLM Validation — Algorithm Comparison",
                 fontsize=11, fontweight="bold", y=1.01)
    fig.tight_layout()
    fig.savefig(out_path, dpi=_DPI, bbox_inches="tight")
    plt.close(fig)
    logger.info("Figure 1 (bar chart) → %s", out_path)


def _plot_convergence(
    all_records: Dict[str, List[dict]], out_path: Path, window: int = 15
) -> None:
    """Figure 2: routing-objective convergence curves with ±1σ shading."""
    plt.rc("font", **_FONT)
    fig, ax = plt.subplots(figsize=(9, 4))

    # draw baselines first (thin), then A-SPSA on top (thick)
    order = sorted(all_records, key=lambda v: 0 if v == "aspsa" else 1)
    for vname in order:
        records = all_records[vname]
        mean, std = _rolling_curves(records, "routing_objective", window)
        if len(mean) == 0:
            # fall back to success if routing_objective missing
            mean, std = _rolling_curves(records, "success", window)
        if len(mean) == 0:
            continue
        xs = np.arange(len(mean))
        lw = 2.5 if vname == "aspsa" else 1.1
        color = COLORS.get(vname, "#888888")
        ax.plot(xs, mean, label=LABELS.get(vname, vname), color=color, lw=lw)
        ax.fill_between(xs, mean - std, mean + std, color=color, alpha=0.10)

    ax.set_xlabel("Task index", labelpad=4)
    ax.set_ylabel(f"Rolling routing objective (window={window})", labelpad=4)
    ax.set_title("Convergence of Routing Objective (Real-LLM Validation)", pad=8)
    ax.legend(ncol=2, fontsize=8, framealpha=0.85)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(linewidth=0.35, alpha=0.45)
    fig.tight_layout()
    fig.savefig(out_path, dpi=_DPI, bbox_inches="tight")
    plt.close(fig)
    logger.info("Figure 2 (convergence) → %s", out_path)


def _plot_multimetric(
    all_records: Dict[str, List[dict]], out_path: Path, window: int = 15
) -> None:
    """Figure 3: 2×2 panel — routing obj, DHR, F2 loss, success rate."""
    plt.rc("font", **_FONT)
    panels = [
        ("routing_objective", "Routing objective ↑"),
        ("deadline_hit",      "Deadline hit rate ↑"),
        ("f2_loss_approx",    "F₂ wait loss proxy ↓"),
        ("success",           "Success rate ↑"),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(13, 8), sharex=False)
    axes_flat = axes.flatten()

    order = sorted(all_records, key=lambda v: 0 if v == "aspsa" else 1)
    for ax, (key, title) in zip(axes_flat, panels):
        for vname in order:
            records = all_records[vname]
            # compute f2 proxy from true_wait / predicted_wait if needed
            if key == "f2_loss_approx":
                by_seed: Dict[int, list] = {}
                for r in records:
                    tw = r.get("true_wait")
                    pw = r.get("predicted_wait")
                    if tw is not None and pw is not None:
                        val = max(0.0, float(tw) + 0.1 - float(pw)) ** 2
                    else:
                        val = 0.0
                    by_seed.setdefault(r["seed"], []).append(val)
                if not by_seed:
                    continue
                curves = [
                    pd.Series(v).rolling(window, min_periods=1).mean().values
                    for v in by_seed.values()
                ]
                max_len = max(len(c) for c in curves)
                mat = np.full((len(curves), max_len), np.nan)
                for i, c in enumerate(curves):
                    mat[i, :len(c)] = c
                mean, std = np.nanmean(mat, 0), np.nanstd(mat, 0)
            else:
                mean, std = _rolling_curves(records, key, window)
                if len(mean) == 0:
                    continue

            xs = np.arange(len(mean))
            lw = 2.2 if vname == "aspsa" else 1.0
            color = COLORS.get(vname, "#888888")
            ax.plot(xs, mean, label=LABELS.get(vname, vname), color=color, lw=lw)
            ax.fill_between(xs, mean - std, mean + std, color=color, alpha=0.09)

        ax.set_title(title, fontsize=9, pad=5)
        ax.set_xlabel("Task index", fontsize=8)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.grid(linewidth=0.3, alpha=0.4)

    # single shared legend
    handles, lbls = axes_flat[0].get_legend_handles_labels()
    fig.legend(handles, lbls, loc="lower center", ncol=4, fontsize=8,
               framealpha=0.9, bbox_to_anchor=(0.5, -0.02))
    fig.suptitle("Real-LLM Validation — Multi-Metric Convergence",
                 fontsize=11, fontweight="bold")
    fig.tight_layout(rect=[0, 0.05, 1, 1])
    fig.savefig(out_path, dpi=_DPI, bbox_inches="tight")
    plt.close(fig)
    logger.info("Figure 3 (multi-metric) → %s", out_path)


def _plot_spsa_traces(root: Path, variants: List[str], seeds: List[int]) -> None:
    """Figure 4: adaptive step α_k and curvature κ_k for A-SPSA vs fixed steps."""
    plt.rc("font", **_FONT)

    # Collect trace CSVs
    traces: Dict[str, pd.DataFrame] = {}
    for vname in variants:
        frames = []
        for seed in seeds:
            p = root / f"spsa_trace_{vname}_seed{seed}.csv"
            if p.exists():
                df = pd.read_csv(p)
                df["seed"] = seed
                frames.append(df)
        if frames:
            traces[vname] = pd.concat(frames, ignore_index=True)

    if not traces:
        logger.info("No SPSA trace files found — skipping Figure 4.")
        return

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    # Left: α_k per variant
    ax = axes[0]
    for vname, df in traces.items():
        by_seed = df.groupby("seed")["alpha_k"].apply(list)
        curves = [pd.Series(v).rolling(5, min_periods=1).mean().values for v in by_seed]
        max_len = max(len(c) for c in curves)
        mat = np.full((len(curves), max_len), np.nan)
        for i, c in enumerate(curves):
            mat[i, :len(c)] = c
        mean = np.nanmean(mat, 0)
        lw = 2.5 if vname == "aspsa" else 1.0
        ax.plot(mean, label=LABELS.get(vname, vname),
                color=COLORS.get(vname, "#888888"), lw=lw)
    ax.set_xlabel("SPSA step")
    ax.set_ylabel("Step size α_k")
    ax.set_title("Adaptive Step Size α_k per Algorithm")
    ax.legend(ncol=2, fontsize=7)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(linewidth=0.35, alpha=0.45)

    # Right: κ_k (curvature) — A-SPSA only
    ax = axes[1]
    aspsa_variants = [v for v in traces if "aspsa" in v]
    for vname in aspsa_variants:
        df = traces[vname]
        kappa_col = df["kappa_k"].replace(float("nan"), np.nan)
        if kappa_col.notna().sum() == 0:
            continue
        by_seed = df.groupby("seed")["kappa_k"].apply(list)
        curves = [pd.Series(v).rolling(5, min_periods=1).mean().values for v in by_seed]
        max_len = max(len(c) for c in curves)
        mat = np.full((len(curves), max_len), np.nan)
        for i, c in enumerate(curves):
            mat[i, :len(c)] = c
        mean = np.nanmean(mat, 0)
        lw = 2.5 if vname == "aspsa" else 1.4
        label = LABELS.get(vname, vname)
        ax.plot(mean, label=label, color=COLORS.get(vname, "#d62728"), lw=lw,
                linestyle="-" if vname == "aspsa" else "--")
    ax.set_xlabel("SPSA step")
    ax.set_ylabel("Curvature estimate κ_k")
    ax.set_title("A-SPSA Adaptive Curvature κ_k (Eq. 11)")
    ax.legend(fontsize=8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(linewidth=0.35, alpha=0.45)

    fig.suptitle("SPSA Parameter Trajectories over Optimisation Steps",
                 fontsize=11, fontweight="bold")
    fig.tight_layout()
    out_path = root / "fig_spsa_traces.png"
    fig.savefig(out_path, dpi=_DPI, bbox_inches="tight")
    plt.close(fig)
    logger.info("Figure 4 (SPSA traces) → %s", out_path)


# ---------------------------------------------------------------------------
# Paper-ready console table
# ---------------------------------------------------------------------------

def _print_table(agg_rows: List[dict]) -> None:
    W = 78
    print("\n" + "=" * W)
    print("Real-LLM Validation Results  (mean ± std over seeds)")
    print("=" * W)
    print(f"{'Algorithm':10s}  {'Routing obj ↑':>16s}  {'DHR ↑':>12s}  "
          f"{'F₂ loss ↓':>12s}  {'Fallback':>8s}")
    print("-" * W)
    for row in agg_rows:
        name = LABELS.get(row["variant"], row["variant"])
        ro  = f"{row.get('routing_objective_mean',0):.3f}±{row.get('routing_objective_std',0):.3f}"
        dhr = f"{row.get('deadline_hit_rate_mean',0):.3f}±{row.get('deadline_hit_rate_std',0):.3f}"
        f2  = f"{row.get('f2_wait_loss_mean',0):.2f}±{row.get('f2_wait_loss_std',0):.2f}"
        fb  = f"{row.get('api_fallback_rate_mean',0)*100:.1f}%"
        print(f"{name:10s}  {ro:>16s}  {dhr:>12s}  {f2:>12s}  {fb:>8s}")
    print("=" * W + "\n")


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def run_real_llm_comparison(
    num_tasks: int,
    seeds: List[int],
    variants: List[str],
    output_dir: str,
    do_dry_run: bool,
    rate_interval_s: float,
    num_ctrl: int,
    num_agents: int,
    cache_file: Optional[str],
    config_name: str = "llm_config",
    log_file: Optional[str] = None,
    spsa_interval: int = 5,
    max_api_calls: Optional[int] = None,
) -> None:
    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)

    handlers: List[logging.Handler] = [
        logging.StreamHandler(sys.stdout),
    ]
    if log_file:
        handlers.append(logging.FileHandler(log_file, encoding="utf-8"))
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-7s  %(message)s",
        datefmt="%H:%M:%S",
        handlers=handlers,
        force=True,
    )
    if log_file:
        logging.getLogger("src.spsa_variants").setLevel(logging.DEBUG)
        logging.getLogger("spsa_variants").setLevel(logging.DEBUG)

    logger.info("Config module  : %s", config_name)
    logger.info("Tasks          : %d", num_tasks)
    logger.info("Seeds          : %s", seeds)
    logger.info("Variants       : %s", variants)
    logger.info("SPSA interval  : %d (effective steps per 200 tasks: ~%d)",
                spsa_interval, max(0, (num_tasks - 15) // max(1, spsa_interval)))
    logger.info("Output dir     : %s", root.resolve())

    caching_client, judge_model, latency_profiles = _build_clients(
        min_interval_s=rate_interval_s, config_name=config_name,
        max_calls=max_api_calls,
    )

    # Approach A: inject simulated latency profiles after real API calls
    if latency_profiles:
        # Use a fixed seed so latency samples are reproducible; per-seed RNG
        # is set again inside the seed loop below.
        caching_client.set_latency_profiles(latency_profiles, np.random.RandomState(0))

    # Optionally restore a previously saved cache (resume interrupted run)
    if cache_file and Path(cache_file).exists():
        logger.info("Loading existing cache from %s ...", cache_file)
        caching_client = PromptCachingLLMClient.load(
            Path(cache_file), caching_client._raw
        )
        if latency_profiles:
            caching_client.set_latency_profiles(latency_profiles, np.random.RandomState(0))

    if do_dry_run:
        dry_run(caching_client, judge_model)
        return

    agent_model_names = [p.model_name for p in PROFILES[:num_agents]]
    all_summaries: Dict[str, List[dict]] = {v: [] for v in variants}
    all_records:   Dict[str, List[dict]] = {v: [] for v in variants}

    for seed_idx, seed in enumerate(seeds):
        logger.info("=" * 60)
        logger.info("Seed %d (%d/%d)", seed, seed_idx + 1, len(seeds))
        logger.info("=" * 60)

        tasks = generate_tasks(num_tasks, seed=seed, scenario="balanced")

        # Update latency RNG to be seed-specific (reproducible per seed)
        if latency_profiles:
            caching_client.set_latency_profiles(latency_profiles, np.random.RandomState(seed))

        # ------------------------------------------------------------------
        # Phase 1: Pre-collect all agent responses (one API call per agent
        # per task; judge calls happen automatically in Phase 2 and are also
        # cached for reuse across variants).
        # ------------------------------------------------------------------
        logger.info(
            "Phase 1 — pre-collecting %d tasks × %d agents = %d API calls ...",
            len(tasks), len(agent_model_names), len(tasks) * len(agent_model_names),
        )
        caching_client.pre_collect(tasks, agent_model_names)

        # Save cache checkpoint after pre-collection
        cache_path = root / f"execution_cache_seed{seed}.json"
        caching_client.save(cache_path)
        logger.info("Cache saved → %s", cache_path)

        stats = caching_client.cache_stats()
        logger.info(
            "Cache after pre-collect: size=%d  hits=%d  misses=%d  hit_rate=%.1f%%",
            stats["cache_size"], stats["hits"], stats["misses"], stats["hit_rate"] * 100,
        )

        # ------------------------------------------------------------------
        # Phase 2: Run all variants — mostly cache hits, no new API calls
        # (except judge calls for variant 1, also cached for variants 2-7).
        # ------------------------------------------------------------------
        logger.info("Phase 2 — replaying %d variants …", len(variants))
        for v_idx, variant_name in enumerate(variants):
            hits_before   = caching_client.hits
            misses_before = caching_client.misses

            logger.info("  [%d/%d] %s  seed=%d", v_idx + 1, len(variants),
                        variant_name.upper(), seed)
            t0 = time.perf_counter()
            summary, records, trace = _run_one_variant(
                variant_name, tasks, caching_client, judge_model,
                seed=seed, num_ctrl=num_ctrl, num_agents=num_agents,
                collect_trace=True,
                spsa_interval=spsa_interval,
            )
            elapsed = time.perf_counter() - t0

            new_hits   = caching_client.hits   - hits_before
            new_misses = caching_client.misses  - misses_before
            logger.info(
                "  Done %.1fs  routing=%.3f  DHR=%.3f  F2=%.2f  "
                "cache hits=%d misses=%d  trace_steps=%d",
                elapsed,
                summary.get("routing_objective", 0),
                summary.get("deadline_hit_rate", 0),
                summary.get("f2_wait_loss", float("nan")),
                new_hits, new_misses, len(trace),
            )

            all_summaries[variant_name].append(summary)
            all_records[variant_name].extend(records)

            rec_path = root / f"records_{variant_name}_seed{seed}.csv"
            pd.DataFrame(records).to_csv(rec_path, index=False)

            if trace:
                trace_path = root / f"spsa_trace_{variant_name}_seed{seed}.csv"
                pd.DataFrame(trace).to_csv(trace_path, index=False)
                logger.info("  SPSA trace → %s", trace_path)

    # ------------------------------------------------------------------
    # Aggregate + outputs
    # ------------------------------------------------------------------
    agg_rows = [_aggregate(all_summaries[v]) for v in variants]

    summary_path = root / "summary_real_llm.csv"
    pd.DataFrame(agg_rows).to_csv(summary_path, index=False)
    logger.info("Summary → %s", summary_path)

    raw_path = root / "summaries_per_seed.json"
    raw_path.write_text(
        json.dumps({v: all_summaries[v] for v in variants}, indent=2, default=str),
        encoding="utf-8",
    )

    _plot_bar(agg_rows, root / "fig_comparison_bar.png")
    _plot_convergence(all_records, root / "fig_convergence_routing.png")
    _plot_multimetric(all_records, root / "fig_convergence_multi.png")
    _plot_spsa_traces(root, variants, seeds)
    _print_table(agg_rows)

    final_stats = caching_client.cache_stats()
    logger.info(
        "Total cache stats: hits=%d  misses=%d  hit_rate=%.1f%%",
        final_stats["hits"], final_stats["misses"], final_stats["hit_rate"] * 100,
    )
    print(f"Outputs written to: {root.resolve()}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Real-LLM validation with shared execution cache.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--tasks",    type=int,  default=200,
                   help="Number of tasks per seed (200 recommended for SPSA convergence).")
    p.add_argument("--seeds",    type=str,  default="11,42,123")
    p.add_argument("--variants", type=str,  default=",".join(SPSA_VARIANTS))
    p.add_argument("--output-dir", type=str, default="real_llm_outputs")
    p.add_argument("--config",   type=str,  default="llm_config",
                   help="Config module name: 'llm_config' (Groq+OpenAI) or "
                        "'llm_config_openrouter' (Groq+OpenRouter free).")
    p.add_argument("--max-api-calls", type=int, default=None,
                   help="Hard cap on paid API requests; the run stops with BudgetExceeded "
                        "when it is reached (default: LLM_MAX_CALLS, else no cap).")
    p.add_argument("--dry-run",  action="store_true",
                   help="Send one test call per model and exit.")
    p.add_argument("--rate-interval", type=float, default=3.0,
                   help="Min seconds between consecutive real API calls.")
    p.add_argument("--controllers", type=int, default=3)
    p.add_argument("--agents",      type=int, default=5)
    p.add_argument("--cache-file",  type=str, default=None,
                   help="Path to a previously saved execution_cache_*.json to resume.")
    p.add_argument("--log-file",      type=str, default=None,
                   help="Write full logs (incl. DEBUG SPSA params) to this file.")
    p.add_argument("--spsa-interval", type=int, default=5,
                   help="Call SPSA every N tasks (default 5 → 38 steps/200 tasks; "
                        "use 1 for ~185 steps/200 tasks).")
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    run_real_llm_comparison(
        num_tasks       = args.tasks,
        seeds           = [int(s) for s in args.seeds.split(",")],
        variants        = args.variants.split(","),
        output_dir      = args.output_dir,
        do_dry_run      = args.dry_run,
        rate_interval_s = args.rate_interval,
        num_ctrl        = args.controllers,
        num_agents      = args.agents,
        cache_file      = args.cache_file,
        config_name     = args.config,
        log_file        = args.log_file,
        spsa_interval   = args.spsa_interval,
        max_api_calls   = args.max_api_calls,
    )
