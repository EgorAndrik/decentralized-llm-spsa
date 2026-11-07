"""
Real dataset loaders for task generation.

MBPP and TriviaQA are loaded from local files in data/.
XSum, FLORES-200, and BFCL are downloaded via HuggingFace datasets on first
use and cached to data/.hf_cache/ as compact JSON files.

Falls back to a single template string per type if a dataset cannot be loaded,
so experiments degrade gracefully rather than crashing.
"""

from __future__ import annotations

import json
import logging
import os
import re
from pathlib import Path
from typing import Dict, List

logger = logging.getLogger(__name__)

_DATA_ROOT = Path(__file__).parent.parent / "data"
_HF_CACHE = _DATA_ROOT / ".hf_cache"

# In-process memory cache: populated on first call per type.
_POOL_CACHE: Dict[str, List[str]] = {}

# Single-string fallbacks used when a dataset cannot be loaded.
_FALLBACKS: Dict[str, str] = {
    "PROGRAMMING":   "Build ML pipeline code with feature engineering and model training.",
    "QA":            "Answer factual question and explain reasoning with references.",
    "SUMMARIZATION": "Summarize the report into concise bullet points.",
    "TRANSLATION":   "Translate customer message from Russian to English.",
    "TOOL_USE":      "Call CRM API tool to create ticket and update status.",
}


# ---------------------------------------------------------------------------
# Local-file loaders
# ---------------------------------------------------------------------------

def _load_mbpp() -> List[str]:
    """Load MBPP programming challenge descriptions (local JSONL)."""
    path = _DATA_ROOT / "mbpp.jsonl"
    texts: List[str] = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                texts.append(json.loads(line)["text"])
    logger.info("MBPP: loaded %d tasks", len(texts))
    return texts


def _load_triviaqa() -> List[str]:
    """Stream TriviaQA questions from the unfiltered dev set (298 MB).

    Uses regex scanning in 128 KB chunks to avoid loading the full JSON into
    RAM.  Extracted questions are cached to .hf_cache/triviaqa_questions.json
    on first run so subsequent calls are instant.
    """
    cache = _HF_CACHE / "triviaqa_questions.json"
    if cache.exists():
        questions: List[str] = json.loads(cache.read_text(encoding="utf-8"))
        logger.info("TriviaQA: loaded %d questions from cache", len(questions))
        return questions

    path = _DATA_ROOT / "triviaqa-unfiltered" / "unfiltered-web-dev.json"
    # TriviaQA questions are plain English; no escaped quotes expected.
    pattern = re.compile(r'"Question"\s*:\s*"([^"]+)"')
    questions = []
    overlap = ""
    chunk_size = 1 << 17  # 128 KB
    with open(path, encoding="utf-8", buffering=1 << 20) as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            buf = overlap + chunk
            for m in pattern.finditer(buf):
                questions.append(m.group(1))
            # Keep last 200 chars to handle matches that span chunk boundaries.
            overlap = buf[-200:]

    _HF_CACHE.mkdir(parents=True, exist_ok=True)
    cache.write_text(json.dumps(questions), encoding="utf-8")
    logger.info("TriviaQA: extracted and cached %d questions", len(questions))
    return questions


# Hugging Face revisions the corpora are downloaded at.  Without a pin a
# dataset can change upstream and silently change the experiment; the commit
# hashes below are the heads observed on 2026-11-05.  If the paper's numbers
# were produced from earlier snapshots, replace them with those hashes.
HF_REVISIONS = {
    "EdinburghNLP/xsum": "7d4d486c2f8ef850b1a11aead99b894ff3dd7da9",
    "Helsinki-NLP/opus-100": "805090dc28bf78897da9641cdf08b61287580df9",
    "NousResearch/hermes-function-calling-v1": "dae3e1d28cfbcf4b915c04ea1e072030529b4bda",
}

# The revision is part of each cache file name, so changing a pin above makes
# the old cache invisible instead of silently serving data from another snapshot.
XSUM_CACHE = f"xsum_documents_{HF_REVISIONS['EdinburghNLP/xsum'][:8]}.json"
OPUS100_CACHE = f"opus100_sentences_{HF_REVISIONS['Helsinki-NLP/opus-100'][:8]}.json"
HERMES_CACHE = f"hermes_fc_queries_{HF_REVISIONS['NousResearch/hermes-function-calling-v1'][:8]}.json"

# Caches written before the pins existed.  They hold the snapshot the earlier
# runs used, so they take precedence over downloading the pinned revision.
LEGACY_CACHES = {
    XSUM_CACHE: "xsum_documents.json",
    OPUS100_CACHE: "opus100_sentences.json",
    HERMES_CACHE: "hermes_fc_queries.json",
}


def _read_cache(name: str, cache_name: str):
    """Texts from the revision-keyed cache, else from the pre-pin cache, else None."""
    for file_name, legacy in ((cache_name, False), (LEGACY_CACHES[cache_name], True)):
        cache = _HF_CACHE / file_name
        if cache.exists():
            texts: List[str] = json.loads(cache.read_text(encoding="utf-8"))
            if legacy:
                logger.info("%s: loaded %d tasks from pre-pin cache %s (revision unknown)",
                            name, len(texts), file_name)
            else:
                logger.info("%s: loaded %d tasks from cache", name, len(texts))
            return texts
    return None


def _require_download_allowed(name: str) -> None:
    """Tests and offline runs set LLM_SPSA_OFFLINE=1: no silent network access."""
    if os.environ.get("LLM_SPSA_OFFLINE") == "1":
        raise RuntimeError(f"{name}: download disabled (LLM_SPSA_OFFLINE=1) and no cache present")


# ---------------------------------------------------------------------------
# HuggingFace loaders (download once, cache as JSON)
# ---------------------------------------------------------------------------

def _load_xsum() -> List[str]:
    """Download XSum BBC articles as summarization task prompts."""
    cached = _read_cache("XSum", XSUM_CACHE)
    if cached is not None:
        return cached
    cache = _HF_CACHE / XSUM_CACHE

    _require_download_allowed("XSum")
    from datasets import load_dataset  # type: ignore[import]

    ds = load_dataset("EdinburghNLP/xsum", split="test", trust_remote_code=False,
                      revision=HF_REVISIONS["EdinburghNLP/xsum"])
    texts = [
        f"Summarize the following article:\n{row['document'][:500]}"
        for row in ds
    ]
    _HF_CACHE.mkdir(parents=True, exist_ok=True)
    cache.write_text(json.dumps(texts), encoding="utf-8")
    logger.info("XSum: downloaded and cached %d tasks", len(texts))
    return texts


def _load_opus100() -> List[str]:
    """Download OPUS-100 English→French translation prompts (Helsinki-NLP)."""
    cached = _read_cache("OPUS-100", OPUS100_CACHE)
    if cached is not None:
        return cached
    cache = _HF_CACHE / OPUS100_CACHE

    _require_download_allowed("OPUS-100")
    from datasets import load_dataset  # type: ignore[import]

    ds = load_dataset(
        "Helsinki-NLP/opus-100", "en-fr",
        split="test", trust_remote_code=False,
        revision=HF_REVISIONS["Helsinki-NLP/opus-100"],
    )
    texts = [
        f"Translate from English to French:\n{row['translation']['en']}"
        for row in ds
    ]
    _HF_CACHE.mkdir(parents=True, exist_ok=True)
    cache.write_text(json.dumps(texts), encoding="utf-8")
    logger.info("OPUS-100: downloaded and cached %d tasks", len(texts))
    return texts


def _load_hermes_fc() -> List[str]:
    """Download Hermes function-calling v1 user instructions as tool-use tasks.

    Extracts the first human turn from each conversation, which contains a
    natural-language request that requires API / tool invocation.
    """
    cached = _read_cache("Hermes FC", HERMES_CACHE)
    if cached is not None:
        return cached
    cache = _HF_CACHE / HERMES_CACHE

    _require_download_allowed("Hermes FC")
    from datasets import load_dataset  # type: ignore[import]

    ds = load_dataset(
        "NousResearch/hermes-function-calling-v1",
        split="train", trust_remote_code=False,
        revision=HF_REVISIONS["NousResearch/hermes-function-calling-v1"],
    )
    texts: List[str] = []
    for row in ds:
        for turn in row.get("conversations", []):
            if turn.get("from") in ("human", "user"):
                content = turn.get("value", "").strip()
                if content:
                    texts.append(content[:500])
                break  # only the first user turn per example

    _HF_CACHE.mkdir(parents=True, exist_ok=True)
    cache.write_text(json.dumps(texts), encoding="utf-8")
    logger.info("Hermes FC: downloaded and cached %d tasks", len(texts))
    return texts


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

_LOADERS = {
    "PROGRAMMING":   _load_mbpp,
    "QA":            _load_triviaqa,
    "SUMMARIZATION": _load_xsum,
    "TRANSLATION":   _load_opus100,
    "TOOL_USE":      _load_hermes_fc,
}


def get_task_pool(type_name: str) -> List[str]:
    """Return a non-empty list of real task prompt strings for *type_name*.

    ``type_name`` must be one of: PROGRAMMING, QA, SUMMARIZATION,
    TRANSLATION, TOOL_USE (i.e. ``TaskType.name``).

    Results are cached in memory after the first call per type.  If the
    underlying dataset cannot be loaded a single fallback template is returned
    so that experiments do not crash.
    """
    if type_name in _POOL_CACHE:
        return _POOL_CACHE[type_name]

    loader = _LOADERS.get(type_name)
    if loader is None:
        raise ValueError(f"Unknown task type: {type_name!r}")

    try:
        pool = loader()
        if not pool:
            raise ValueError("Empty pool returned")
    except Exception as exc:
        logger.warning(
            "Failed to load real data for %s (%s). Using fallback template.",
            type_name, exc,
        )
        pool = [_FALLBACKS[type_name]]

    _POOL_CACHE[type_name] = pool
    return pool
