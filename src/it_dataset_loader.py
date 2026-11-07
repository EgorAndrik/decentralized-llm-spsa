"""Heterogeneous external-validity dataset loader.

Do not pool these sources as the primary routing benchmark: source formatting is
confounded with domain. Use :mod:`src.it_benchmark` for the controlled paper
workload. Keyword-derived overlaps here are weak metadata, not ground truth.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import hashlib
import json
from pathlib import Path
import random
import re
from typing import Any, Iterable, Mapping, Sequence


class ITSkill(str, Enum):
    FRONTEND = "frontend"
    BACKEND_API = "backend_api"
    DATABASES = "databases_data_engineering"
    ML_LLM = "ml_llm"
    DEVOPS_CLOUD = "devops_cloud"
    SECURITY = "security"


class TaskKind(str, Enum):
    CODE_GENERATION = "code_generation"
    DEBUGGING = "debugging"
    QUESTION_ANSWERING = "question_answering"
    DESIGN = "design"
    CODE_EDITING = "code_editing"


@dataclass(frozen=True)
class DatasetSource:
    skill: ITSkill
    dataset_id: str
    split: str
    license: str
    homepage: str
    revision: str | None = None
    data_file: str | None = None


@dataclass(frozen=True)
class ITTaskRecord:
    id: str
    prompt: str
    reference_answer: str
    primary_skill: ITSkill
    skill_vector: tuple[float, ...]
    task_kind: TaskKind
    source: str
    source_license: str
    metadata: Mapping[str, Any]

    def skill_scores(self) -> dict[str, float]:
        return {skill.value: self.skill_vector[i] for i, skill in enumerate(ITSkill)}


DATASET_SOURCES: dict[ITSkill, DatasetSource] = {
    # Retained for reproducibility of the rejected draft only: 14 unique
    # prompts were observed, so this source is not scientifically usable.
    ITSkill.FRONTEND: DatasetSource(ITSkill.FRONTEND, "xlelords/vulcan", "train", "MIT", "https://huggingface.co/datasets/xlelords/vulcan", revision="19d2a482c78554e59feca0ce2b50fb7835e90c6f"),
    ITSkill.BACKEND_API: DatasetSource(ITSkill.BACKEND_API, "glaiveai/glaive-function-calling-v2", "train", "Apache-2.0", "https://huggingface.co/datasets/glaiveai/glaive-function-calling-v2", revision="e7f4b6456019f5d8bcb991ef0dd67d8ff23221ac"),
    ITSkill.DATABASES: DatasetSource(ITSkill.DATABASES, "b-mc2/sql-create-context", "train", "CC-BY-4.0", "https://huggingface.co/datasets/b-mc2/sql-create-context", revision="9d80a6a118b838d9defc3798d659a54a2ac2ff37"),
    ITSkill.ML_LLM: DatasetSource(ITSkill.ML_LLM, "xlangai/DS-1000", "test", "CC-BY-SA-4.0", "https://huggingface.co/datasets/xlangai/DS-1000", revision="4416080ac5cb80bdf7576aefb8f9a0b4d5426a44"),
    ITSkill.DEVOPS_CLOUD: DatasetSource(ITSkill.DEVOPS_CLOUD, "Skilln/devops-qa-dataset", "train", "CC-BY-SA-4.0", "https://huggingface.co/datasets/Skilln/devops-qa-dataset", data_file="hf://datasets/Skilln/devops-qa-dataset@ef888f1d0c8122ee51ed13bd0f90b5f7ef05c64b/devops_dataset.parquet"),
    ITSkill.SECURITY: DatasetSource(ITSkill.SECURITY, "Trendyol/Trendyol-Cybersecurity-Instruction-Tuning-Dataset", "train", "Apache-2.0", "https://huggingface.co/datasets/Trendyol/Trendyol-Cybersecurity-Instruction-Tuning-Dataset", revision="357544e7576607d88eaeac9b0adb07e9fd8bb2bb"),
}


_KEYWORDS = {
    ITSkill.FRONTEND: ("html", "css", "javascript", "typescript", "react", "vue", "angular", "tailwind", "browser", "dom", "responsive", "frontend", "webpage"),
    ITSkill.BACKEND_API: ("api", "backend", "rest", "graphql", "websocket", "endpoint", "server", "microservice", "authentication", "fastapi", "django", "express"),
    ITSkill.DATABASES: ("sql", "database", "table", "schema", "query", "postgres", "mysql", "warehouse", "etl", "pipeline", "pandas", "spark"),
    ITSkill.ML_LLM: ("machine learning", "deep learning", "neural", "pytorch", "tensorflow", "sklearn", "model", "training", "inference", "embedding", "llm", "nlp"),
    ITSkill.DEVOPS_CLOUD: ("docker", "kubernetes", "terraform", "ansible", "cloud", "aws", "azure", "gcp", "ci/cd", "github actions", "deployment", "helm", "devops"),
    ITSkill.SECURITY: ("security", "vulnerability", "cve", "xss", "csrf", "injection", "threat", "incident", "malware", "encryption", "secure", "firewall", "oauth"),
}


def _text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, (dict, list, tuple)):
        return json.dumps(value, ensure_ascii=False)
    return str(value).strip()


def _first(record: Mapping[str, Any], names: Sequence[str]) -> str:
    for name in names:
        value = _text(record.get(name))
        if value:
            return value
    return ""


def _chat_text(value: Any, roles: set[str]) -> str:
    if isinstance(value, list):
        return "\n".join(
            _text(row.get("content")) for row in value
            if isinstance(row, Mapping) and str(row.get("role", "")).lower() in roles
        ).strip()
    text = _text(value)
    labels = "USER|HUMAN" if roles & {"user", "human"} else "ASSISTANT"
    matches = re.findall(
        rf"(?:^|\n)(?:{labels}):\s*(.*?)(?=\n(?:USER|HUMAN|ASSISTANT|FUNCTION|SYSTEM):|\Z)",
        text, flags=re.IGNORECASE | re.DOTALL,
    )
    return "\n".join(part.strip() for part in matches if part.strip())


def _extract_prompt(record: Mapping[str, Any]) -> str:
    direct = _first(record, ("instruction", "prompt", "question", "query", "problem", "input_text", "user"))
    context = _first(record, ("input", "context", "code_context", "schema"))
    if direct:
        return f"{direct}\n\nContext:\n{context}" if context else direct
    for field in ("messages", "conversation", "conversations", "chat"):
        if field in record:
            return _chat_text(record[field], {"user", "human"}) or _text(record[field])
    return ""


def _extract_answer(record: Mapping[str, Any]) -> str:
    direct = _first(record, ("output", "answer", "response", "solution", "reference_answer", "reference_code", "code", "assistant"))
    if direct:
        return direct
    for field in ("messages", "conversation", "conversations", "chat"):
        if field in record:
            return _chat_text(record[field], {"assistant"})
    return ""


def infer_task_kind(prompt: str, record: Mapping[str, Any] | None = None) -> TaskKind:
    text = f"{prompt} {_text((record or {}).get('category'))}".lower()
    if any(x in text for x in ("debug", "fix", "bug", "error", "vulnerab")):
        return TaskKind.DEBUGGING
    if any(x in text for x in ("edit", "refactor", "modify", "update existing")):
        return TaskKind.CODE_EDITING
    if any(x in text for x in ("design", "architect", "schema", "strategy", "plan")):
        return TaskKind.DESIGN
    if any(x in text for x in ("write code", "implement", "create", "build", "generate", "sql")):
        return TaskKind.CODE_GENERATION
    return TaskKind.QUESTION_ANSWERING


def infer_skill_vector(prompt: str, primary: ITSkill) -> tuple[float, ...]:
    lowered = prompt.lower()
    scores = {
        skill: min(0.85, 0.15 * sum(keyword in lowered for keyword in keywords))
        for skill, keywords in _KEYWORDS.items()
    }
    scores[primary] = 1.0
    return tuple(float(scores[skill]) for skill in ITSkill)


def normalize_record(record: Mapping[str, Any], source: DatasetSource, row_index: int) -> ITTaskRecord | None:
    prompt = _extract_prompt(record).strip()
    answer = _extract_answer(record).strip()
    if len(prompt) < 20 or not answer:
        return None
    digest = hashlib.sha256(f"{source.dataset_id}\0{prompt}".encode()).hexdigest()[:20]
    return ITTaskRecord(
        id=f"{source.skill.value}:{digest}", prompt=prompt, reference_answer=answer,
        primary_skill=source.skill, skill_vector=infer_skill_vector(prompt, source.skill),
        task_kind=infer_task_kind(prompt, record), source=source.dataset_id,
        source_license=source.license,
        metadata={"row_index": row_index, "source_split": source.split},
    )


def normalize_records(records: Iterable[Mapping[str, Any]], source: DatasetSource, limit: int | None = None) -> list[ITTaskRecord]:
    result: list[ITTaskRecord] = []
    seen: set[str] = set()
    for index, record in enumerate(records):
        item = normalize_record(record, source, index)
        if item is None:
            continue
        key = " ".join(item.prompt.lower().split())
        if key in seen:
            continue
        seen.add(key)
        result.append(item)
        if limit is not None and len(result) >= limit:
            break
    return result


def load_it_dataset(per_skill: int = 1_000, seed: int = 42, skills: Sequence[ITSkill] | None = None, cache_dir: str | Path | None = None) -> list[ITTaskRecord]:
    """Stream, normalize, deduplicate and balance the configured sources."""
    if per_skill <= 0:
        raise ValueError("per_skill must be positive")
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise RuntimeError("Install dependencies from requirements.txt to load IT datasets") from exc
    selected = tuple(skills or tuple(ITSkill))
    rng = random.Random(seed)
    result: list[ITTaskRecord] = []
    for skill in selected:
        if skill not in DATASET_SOURCES:
            raise ValueError(f"No configured source for {skill}")
        source = DATASET_SOURCES[skill]
        kwargs: dict[str, Any] = {"split": source.split, "streaming": True}
        if cache_dir is not None:
            kwargs["cache_dir"] = str(cache_dir)
        if source.revision:
            kwargs["revision"] = source.revision
        if source.data_file:
            kwargs["data_files"] = source.data_file
            stream = load_dataset("parquet", **kwargs)
        else:
            stream = load_dataset(source.dataset_id, **kwargs)
        candidates = normalize_records(stream, source, max(per_skill * 3, per_skill + 100))
        if len(candidates) < per_skill:
            raise RuntimeError(f"{source.dataset_id}: {len(candidates)} valid unique records; {per_skill} required")
        rng.shuffle(candidates)
        result.extend(candidates[:per_skill])
    rng.shuffle(result)
    return result


def write_jsonl(records: Sequence[ITTaskRecord], output_path: str | Path) -> None:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps({
                "id": record.id, "prompt": record.prompt,
                "reference_answer": record.reference_answer,
                "primary_skill": record.primary_skill.value,
                "skill_scores": record.skill_scores(), "task_kind": record.task_kind.value,
                "source": record.source, "source_license": record.source_license,
                "metadata": dict(record.metadata),
            }, ensure_ascii=False) + "\n")


