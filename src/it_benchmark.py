"""Scientifically controlled IT-domain benchmark construction and audit.

The primary routing benchmark uses one common Stack Overflow source for every
domain.  This removes the fatal ``dataset source == class label`` confound in
the earlier one-source-per-domain draft.  Domain labels come only from explicit
Stack Overflow tags; keyword inference is not used as ground truth.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
import hashlib
import html
import json
import math
from pathlib import Path
import random
import re
from statistics import median
from typing import Any, Iterable, Mapping, Sequence

from .it_dataset_loader import ITSkill, TaskKind, infer_task_kind


STACKPULSE_ID = "Omarrran/StackPulse_778K_QnA_Code_dataset"
STACKPULSE_CONFIG = "high_quality"
STACKPULSE_REVISION = "18c9ddf"  # short form, recorded in the committed manifest
# Full commit used for the download; the short form above is only its label.
STACKPULSE_COMMIT = "18c9ddf9a10d3a2c5cf433db7dcd36d7573fe44c"

# Tags are deliberately conservative.  Ambiguous tags such as
# ``authentication`` and generic ``python`` are not domain labels.
DOMAIN_TAGS: dict[ITSkill, frozenset[str]] = {
    ITSkill.FRONTEND: frozenset({
        "html", "css", "javascript", "typescript", "reactjs", "vue.js",
        "angular", "jquery", "frontend", "sass", "webpack",
    }),
    ITSkill.BACKEND_API: frozenset({
        "node.js", "django", "flask", "fastapi", "spring", "spring-boot",
        "asp.net", "asp.net-core", "express", "rest", "api", "graphql",
    }),
    ITSkill.DATABASES: frozenset({
        "mysql", "sql-server", "postgresql", "mongodb", "database", "sql",
        "sqlite", "oracle", "redis", "elasticsearch", "apache-spark",
        "airflow", "apache-kafka",
    }),
    ITSkill.ML_LLM: frozenset({
        "machine-learning", "deep-learning", "tensorflow", "pytorch",
        "scikit-learn", "nlp", "keras", "neural-network",
        "huggingface-transformers", "large-language-model",
    }),
    ITSkill.DEVOPS_CLOUD: frozenset({
        "docker", "kubernetes", "terraform", "ansible", "jenkins",
        "github-actions", "gitlab-ci", "amazon-web-services", "azure",
        "google-cloud-platform", "devops", "deployment", "helm",
    }),
    ITSkill.SECURITY: frozenset({
        "security", "cryptography", "encryption", "xss", "csrf",
        "sql-injection", "jwt", "ssl", "tls", "oauth", "oauth-2.0",
    }),
}


@dataclass(frozen=True)
class RoutingBenchmarkRecord:
    id: str
    prompt: str
    primary_skill: ITSkill
    skill_vector: tuple[float, ...]
    task_kind: TaskKind
    source: str
    source_revision: str
    source_url: str
    source_license: str
    tags: tuple[str, ...]
    prompt_chars: int
    has_code: bool
    difficulty_bucket: str
    quality_score: int
    view_count: int
    reference_answer: str = ""


@dataclass(frozen=True)
class AuditThresholds:
    min_unique_fraction: float = 0.99
    max_class_count_delta: int = 0
    max_median_length_ratio: float = 3.0
    max_task_kind_total_variation: float = 0.55
    min_tag_purity: float = 1.0


def parse_tags(value: Any) -> tuple[str, ...]:
    if isinstance(value, (list, tuple)):
        parts = value
    else:
        parts = str(value or "").split("|")
    return tuple(sorted({str(tag).strip().lower() for tag in parts if str(tag).strip()}))


def domain_hits(tags: Sequence[str]) -> tuple[ITSkill, ...]:
    tag_set = set(tags)
    return tuple(skill for skill in ITSkill if tag_set & DOMAIN_TAGS[skill])


def _plain_text(value: Any) -> str:
    text = html.unescape(str(value or ""))
    text = re.sub(r"<pre[^>]*><code[^>]*>", "\n[CODE]\n", text, flags=re.I)
    text = re.sub(r"</code></pre>", "\n[/CODE]\n", text, flags=re.I)
    text = re.sub(r"<[^>]+>", " ", text)
    return " ".join(text.split())


def _difficulty(row: Mapping[str, Any]) -> str:
    score = int(row.get("score") or 0)
    views = int(row.get("view_count") or 0)
    code_blocks = int(row.get("code_block_cnt") or 0)
    body_len = int(row.get("body_len") or 0)
    signal = math.log1p(max(0, views)) + 0.35 * code_blocks + 0.001 * body_len
    if score >= 20 or signal >= 10.0:
        return "hard"
    if score >= 8 or signal >= 7.0:
        return "medium"
    return "easy"


def normalize_stackoverflow_row(row: Mapping[str, Any], primary: ITSkill) -> RoutingBenchmarkRecord | None:
    tags = parse_tags(row.get("tags"))
    hits = domain_hits(tags)
    if primary not in hits:
        return None
    title = _plain_text(row.get("title"))
    body = _plain_text(row.get("question_body") or row.get("body_text"))
    prompt = f"{title}\n\n{body}".strip()
    if len(prompt) < 40:
        return None
    qid = str(row.get("id") or "").strip()
    if not qid:
        qid = hashlib.sha256(prompt.encode("utf-8")).hexdigest()[:20]
    skill_vector = tuple(1.0 if skill in hits else 0.0 for skill in ITSkill)
    return RoutingBenchmarkRecord(
        id=f"stackoverflow:{qid}:{primary.value}",
        prompt=prompt,
        primary_skill=primary,
        skill_vector=skill_vector,
        task_kind=infer_task_kind(prompt, row),
        source=STACKPULSE_ID,
        source_revision=STACKPULSE_REVISION,
        source_url=f"https://stackoverflow.com/questions/{qid}",
        # Stack Overflow content remains CC BY-SA irrespective of the wrapper
        # dataset's Apache metadata.  Text is not redistributed by default.
        source_license="CC-BY-SA (Stack Overflow upstream)",
        tags=tags,
        prompt_chars=len(prompt),
        has_code=bool(row.get("has_code")),
        difficulty_bucket=_difficulty(row),
        quality_score=int(row.get("score") or 0),
        view_count=int(row.get("view_count") or 0),
    )


def _stratum(record: RoutingBenchmarkRecord) -> tuple[bool, str, str]:
    length = "short" if record.prompt_chars < 400 else "medium" if record.prompt_chars < 900 else "long"
    return record.has_code, length, record.difficulty_bucket


def build_balanced_benchmark(
    rows: Iterable[Mapping[str, Any]],
    per_skill: int = 200,
    seed: int = 42,
) -> list[RoutingBenchmarkRecord]:
    """Build equal domain classes while spreading code/length/difficulty strata."""
    if per_skill <= 0:
        raise ValueError("per_skill must be positive")
    candidates: dict[ITSkill, list[RoutingBenchmarkRecord]] = defaultdict(list)
    seen_ids: dict[ITSkill, set[str]] = defaultdict(set)
    for row in rows:
        tags = parse_tags(row.get("tags"))
        for skill in domain_hits(tags):
            item = normalize_stackoverflow_row(row, skill)
            if item is not None and item.id not in seen_ids[skill]:
                seen_ids[skill].add(item.id)
                candidates[skill].append(item)

    rng = random.Random(seed)
    result: list[RoutingBenchmarkRecord] = []
    selected_questions: set[str] = set()

    # Select rare domains first. A multi-label Stack Overflow question may be a
    # candidate for several domains, but it must occur only once in the final
    # benchmark; otherwise train/test leakage and inflated sample size follow.
    skill_order = sorted(ITSkill, key=lambda skill: (len(candidates[skill]), skill.value))
    for skill in skill_order:
        pool = candidates[skill]
        if len(pool) < per_skill:
            raise RuntimeError(f"{skill.value}: only {len(pool)} valid tagged tasks; {per_skill} required")
        strata: dict[tuple[bool, str, str], list[RoutingBenchmarkRecord]] = defaultdict(list)
        for item in pool:
            if item.source_url not in selected_questions:
                strata[_stratum(item)].append(item)
        for values in strata.values():
            rng.shuffle(values)
        selected: list[RoutingBenchmarkRecord] = []
        # Round-robin prevents the largest single stratum from dominating.
        keys = sorted(strata, key=str)
        while len(selected) < per_skill:
            progressed = False
            for key in keys:
                while strata[key] and strata[key][-1].source_url in selected_questions:
                    strata[key].pop()
                if strata[key] and len(selected) < per_skill:
                    item = strata[key].pop()
                    selected.append(item)
                    selected_questions.add(item.source_url)
                    progressed = True
            if not progressed:
                break
        if len(selected) < per_skill:
            raise RuntimeError(
                f"{skill.value}: only {len(selected)} globally unique tasks remain; "
                f"{per_skill} required"
            )
        result.extend(selected)
    rng.shuffle(result)
    return result


def load_routing_benchmark(per_skill: int = 200, seed: int = 42, cache_dir: str | Path | None = None) -> list[RoutingBenchmarkRecord]:
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise RuntimeError("Install dependencies from requirements.txt") from exc
    kwargs: dict[str, Any] = {"split": "train", "revision": STACKPULSE_COMMIT}
    if cache_dir is not None:
        kwargs["cache_dir"] = str(cache_dir)
    rows = load_dataset(STACKPULSE_ID, STACKPULSE_CONFIG, **kwargs)
    records = build_balanced_benchmark(rows, per_skill=per_skill, seed=seed)
    audit = audit_benchmark(records)
    if not audit["passed"]:
        raise RuntimeError("IT benchmark failed quality gates: " + "; ".join(audit["failures"]))
    return records


def _distribution(values: Sequence[str]) -> dict[str, float]:
    counts = Counter(values)
    total = max(1, len(values))
    return {key: count / total for key, count in counts.items()}


def _total_variation(left: Mapping[str, float], right: Mapping[str, float]) -> float:
    keys = set(left) | set(right)
    return 0.5 * sum(abs(left.get(key, 0.0) - right.get(key, 0.0)) for key in keys)


def audit_benchmark(records: Sequence[RoutingBenchmarkRecord], thresholds: AuditThresholds | None = None) -> dict[str, Any]:
    thresholds = thresholds or AuditThresholds()
    by_skill: dict[ITSkill, list[RoutingBenchmarkRecord]] = defaultdict(list)
    for item in records:
        by_skill[item.primary_skill].append(item)
    counts = {skill.value: len(by_skill[skill]) for skill in ITSkill}
    canonical = [" ".join(item.prompt.lower().split()) for item in records]
    unique_fraction = len(set(canonical)) / max(1, len(canonical))
    medians = {skill.value: median([x.prompt_chars for x in by_skill[skill]]) if by_skill[skill] else 0 for skill in ITSkill}
    positive_medians = [value for value in medians.values() if value > 0]
    length_ratio = max(positive_medians) / min(positive_medians) if positive_medians else float("inf")
    overall_kind = _distribution([x.task_kind.value for x in records])
    kind_tv = {
        skill.value: _total_variation(_distribution([x.task_kind.value for x in by_skill[skill]]), overall_kind)
        for skill in ITSkill
    }
    tag_purity = sum(
        item.primary_skill in domain_hits(item.tags) for item in records
    ) / max(1, len(records))
    failures: list[str] = []
    if unique_fraction < thresholds.min_unique_fraction:
        failures.append(f"unique_fraction={unique_fraction:.3f}")
    if counts and max(counts.values()) - min(counts.values()) > thresholds.max_class_count_delta:
        failures.append(f"class_counts={counts}")
    if length_ratio > thresholds.max_median_length_ratio:
        failures.append(f"median_length_ratio={length_ratio:.3f}")
    if kind_tv and max(kind_tv.values()) > thresholds.max_task_kind_total_variation:
        failures.append(f"task_kind_TV={kind_tv}")
    if tag_purity < thresholds.min_tag_purity:
        failures.append(f"tag_purity={tag_purity:.3f}")
    return {
        "passed": not failures,
        "failures": failures,
        "records": len(records),
        "class_counts": counts,
        "unique_fraction": unique_fraction,
        "median_prompt_chars": medians,
        "median_length_ratio": length_ratio,
        "task_kind_distribution": overall_kind,
        "task_kind_total_variation": kind_tv,
        "tag_purity": tag_purity,
        "source_count": len({item.source for item in records}),
        "source_class_confounding": "controlled: one common source for every class",
        "reference_answers_available": sum(bool(item.reference_answer) for item in records),
    }


def write_benchmark(records: Sequence[RoutingBenchmarkRecord], output: str | Path, include_text: bool = False) -> None:
    """Write metadata by default; text export is opt-in due to CC attribution."""
    path = Path(output)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for item in records:
            payload = asdict(item)
            payload["primary_skill"] = item.primary_skill.value
            payload["task_kind"] = item.task_kind.value
            if not include_text:
                payload.pop("prompt")
                payload.pop("reference_answer")
            handle.write(json.dumps(payload, ensure_ascii=False) + "\n")

