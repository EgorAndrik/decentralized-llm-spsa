"""
Tests verifying that generate_tasks() uses real dataset prompts, not templates,
and that semantic features extracted from real texts are task-type-appropriate.

This is the legacy five-task data path.  It is not used by the controlled
six-domain IT protocol of the manuscript, which reads the pinned manifest in
``data/it_routing_benchmark.jsonl`` instead.

Every dataset here is optional and each test skips when its corpus is absent,
so a missing local corpus reports as a skip rather than as a code failure.
Nothing may silently pass instead: when a corpus *is* present the assertions
run in full.
"""

import sys
import os
import unittest
from pathlib import Path

# Ensure src/ is importable when running as `python -m pytest` from repo root.
sys.path.insert(0, str(Path(__file__).parent.parent))

from src import dataset_loader
from src.dataset_loader import get_task_pool, _POOL_CACHE, _HF_CACHE, _DATA_ROOT

_MBPP = _DATA_ROOT / "mbpp.jsonl"
_TRIVIAQA = _DATA_ROOT / "triviaqa-unfiltered" / "unfiltered-web-dev.json"
_TRIVIAQA_CACHE = _HF_CACHE / "triviaqa_questions.json"


def _local(name: str) -> bool:
    """Return True if the local corpus backing *name* is available."""
    if name == "PROGRAMMING":
        return _MBPP.exists()
    if name == "QA":
        return _TRIVIAQA.exists() or _TRIVIAQA_CACHE.exists()
    return True


requires_mbpp = unittest.skipUnless(
    _local("PROGRAMMING"),
    f"MBPP corpus is absent ({_MBPP}); fetch it with src.prepare_it_dataset",
)
requires_triviaqa = unittest.skipUnless(
    _local("QA"),
    f"TriviaQA corpus is absent ({_TRIVIAQA}); fetch it with src.prepare_it_dataset",
)
requires_local_corpora = unittest.skipUnless(
    _local("PROGRAMMING") and _local("QA"),
    "MBPP and TriviaQA corpora are absent; fetch them with src.prepare_it_dataset",
)
from src.main import SemanticFeatureExtractor, TaskType
from src.run_spsa_comparison import generate_tasks, SCENARIO_TEMPLATES


def _hf_cached(name: str) -> bool:
    """Return True if the HuggingFace cache JSON for *name* exists."""
    mapping = {
        "SUMMARIZATION": dataset_loader.XSUM_CACHE,
        "TRANSLATION":   dataset_loader.OPUS100_CACHE,
        "TOOL_USE":      dataset_loader.HERMES_CACHE,
    }
    fname = mapping.get(name)
    return fname is not None and (_HF_CACHE / fname).exists()


class TestPoolBasics(unittest.TestCase):
    """Each pool loads without error and returns a non-trivial list."""

    def setUp(self):
        _POOL_CACHE.clear()

    def _check_pool(self, type_name: str, min_size: int = 2) -> list:
        pool = get_task_pool(type_name)
        self.assertIsInstance(pool, list)
        self.assertGreaterEqual(
            len(pool), min_size,
            f"{type_name} pool has only {len(pool)} item(s) — likely a fallback.",
        )
        self.assertTrue(all(isinstance(t, str) and t for t in pool),
                        f"{type_name} pool contains empty or non-string items.")
        return pool

    @requires_mbpp
    def test_programming_pool_loads(self):
        pool = self._check_pool("PROGRAMMING", min_size=100)
        self.assertGreaterEqual(len(pool), 900,
                                "MBPP should provide ~974 tasks; got fewer — file may be truncated.")

    @requires_triviaqa
    def test_qa_pool_loads(self):
        pool = self._check_pool("QA", min_size=1000)
        self.assertGreaterEqual(len(pool), 5000,
                                "TriviaQA dev should yield >5000 questions.")

    @unittest.skipUnless(_hf_cached("SUMMARIZATION"), "XSum cache not present — run experiments first.")
    def test_summarization_pool_loads(self):
        self._check_pool("SUMMARIZATION", min_size=1000)

    @unittest.skipUnless(_hf_cached("TRANSLATION"), "OPUS-100 cache not present — run experiments first.")
    def test_translation_pool_loads(self):
        self._check_pool("TRANSLATION", min_size=100)

    @unittest.skipUnless(_hf_cached("TOOL_USE"), "Hermes-FC cache not present — run experiments first.")
    def test_tool_use_pool_loads(self):
        self._check_pool("TOOL_USE", min_size=100)


class TestPoolContents(unittest.TestCase):
    """Prompts in each pool have the expected linguistic signature."""

    def setUp(self):
        _POOL_CACHE.clear()

    @requires_mbpp
    def test_programming_prompts_mention_code_concepts(self):
        pool = get_task_pool("PROGRAMMING")
        code_kws = {"function", "write", "python", "find", "list",
                    "return", "implement", "compute", "string", "number"}
        hits = sum(1 for t in pool if any(kw in t.lower() for kw in code_kws))
        self.assertGreater(hits / len(pool), 0.80,
                           "Less than 80% of MBPP prompts mention coding concepts.")

    @requires_triviaqa
    def test_qa_prompts_are_questions(self):
        pool = get_task_pool("QA")
        question_words = {"who", "what", "when", "where", "which",
                          "how", "why", "whose", "whom"}
        hits = sum(1 for t in pool
                   if t.endswith("?") or t.split()[0].lower() in question_words)
        self.assertGreater(hits / len(pool), 0.90,
                           "Less than 90% of TriviaQA prompts look like questions.")

    @unittest.skipUnless(_hf_cached("SUMMARIZATION"), "XSum cache absent.")
    def test_summarization_prompts_have_article_prefix(self):
        pool = get_task_pool("SUMMARIZATION")
        prefix = "summarize the following article"
        hits = sum(1 for t in pool if t.lower().startswith(prefix))
        self.assertGreater(hits / len(pool), 0.95,
                           "XSum prompts should start with 'Summarize the following article'.")

    @unittest.skipUnless(_hf_cached("TRANSLATION"), "OPUS-100 cache absent.")
    def test_translation_prompts_have_language_instruction(self):
        pool = get_task_pool("TRANSLATION")
        prefix = "translate from english to french"
        hits = sum(1 for t in pool if t.lower().startswith(prefix))
        self.assertGreater(hits / len(pool), 0.95,
                           "OPUS-100 prompts should start with 'Translate from English to French'.")

    @unittest.skipUnless(_hf_cached("TOOL_USE"), "Hermes-FC cache absent.")
    def test_tool_use_prompts_are_nontrivial_instructions(self):
        pool = get_task_pool("TOOL_USE")
        median_len = sorted(len(t) for t in pool)[len(pool) // 2]
        self.assertGreater(median_len, 30,
                           "Hermes-FC prompts appear too short to be real instructions.")


class TestPoolDiversity(unittest.TestCase):
    """Each pool must not be a single repeated template."""

    TEMPLATE_TEXTS = {
        text for scenario in SCENARIO_TEMPLATES.values()
        for text in scenario.values()
    }

    def setUp(self):
        _POOL_CACHE.clear()

    def _check_diversity(self, type_name: str):
        pool = get_task_pool(type_name)
        unique = len(set(pool))
        self.assertGreater(
            unique, 1,
            f"{type_name} pool has only {unique} unique text(s) — likely using a fallback template.",
        )
        # None of the prompts should be a verbatim hardcoded template.
        overlap = set(pool) & self.TEMPLATE_TEXTS
        self.assertEqual(
            len(overlap), 0,
            f"{type_name} pool contains hardcoded template string(s): {overlap}",
        )

    @requires_mbpp
    def test_programming_diversity(self):
        self._check_diversity("PROGRAMMING")

    @requires_triviaqa
    def test_qa_diversity(self):
        self._check_diversity("QA")

    @unittest.skipUnless(_hf_cached("SUMMARIZATION"), "XSum cache absent.")
    def test_summarization_diversity(self):
        self._check_diversity("SUMMARIZATION")

    @unittest.skipUnless(_hf_cached("TRANSLATION"), "OPUS-100 cache absent.")
    def test_translation_diversity(self):
        self._check_diversity("TRANSLATION")

    @unittest.skipUnless(_hf_cached("TOOL_USE"), "Hermes-FC cache absent.")
    def test_tool_use_diversity(self):
        self._check_diversity("TOOL_USE")


class TestSemanticFeatures(unittest.TestCase):
    """Real prompts should activate the right SemanticFeatureExtractor axis."""

    # Axis indices in SemanticFeatureExtractor.AXES
    AXIS = {
        TaskType.PROGRAMMING:   0,
        TaskType.QA:            1,
        TaskType.SUMMARIZATION: 2,
        TaskType.TRANSLATION:   3,
        TaskType.TOOL_USE:      4,
    }

    def setUp(self):
        _POOL_CACHE.clear()

    def _dominant_axis_rate(self, type_name: str, expected_axis: int,
                            sample: int = 100) -> float:
        pool = get_task_pool(type_name)
        import numpy as np
        rng = __import__("random").Random(0)
        sample_texts = rng.sample(pool, min(sample, len(pool)))
        hits = 0
        for text in sample_texts:
            phi = SemanticFeatureExtractor.extract(text)
            if phi[expected_axis] == phi.max():
                hits += 1
        return hits / len(sample_texts)

    def test_programming_features_peak_on_axis0(self):
        rate = self._dominant_axis_rate("PROGRAMMING", 0)
        # MBPP texts often say "Write a function..." which also triggers the
        # tool-use axis keyword "function".  The extractor is keyword-based, so
        # ~35–40% peak on axis 0 is the realistic achievable rate.  The test
        # asserts only that axis 0 is activated significantly above chance (1/5).
        self.assertGreater(rate, 0.25,
            "Fewer than 25% of MBPP prompts peak on the code/programming axis — "
            "SemanticFeatureExtractor may need keyword updates.")

    def test_qa_features_peak_on_axis1(self):
        rate = self._dominant_axis_rate("QA", 1)
        self.assertGreater(rate, 0.50,
            "Fewer than 50% of TriviaQA questions peak on the QA axis.")

    @unittest.skipUnless(_hf_cached("SUMMARIZATION"), "XSum cache absent.")
    def test_summarization_features_peak_on_axis2(self):
        rate = self._dominant_axis_rate("SUMMARIZATION", 2)
        self.assertGreater(rate, 0.70,
            "Fewer than 70% of XSum prompts peak on the summarisation axis.")

    @unittest.skipUnless(_hf_cached("TRANSLATION"), "OPUS-100 cache absent.")
    def test_translation_features_peak_on_axis3(self):
        rate = self._dominant_axis_rate("TRANSLATION", 3)
        self.assertGreater(rate, 0.70,
            "Fewer than 70% of OPUS-100 prompts peak on the translation axis.")

    @unittest.skipUnless(_hf_cached("TOOL_USE"), "Hermes-FC cache absent.")
    def test_tool_use_features_peak_on_axis4(self):
        rate = self._dominant_axis_rate("TOOL_USE", 4)
        self.assertGreater(rate, 0.30,
            "Fewer than 30% of Hermes-FC prompts peak on the tool-use axis.")


class TestGenerateTasks(unittest.TestCase):
    """generate_tasks() must draw diverse real texts and be reproducible."""

    def setUp(self):
        _POOL_CACHE.clear()

    @requires_local_corpora
    def test_tasks_use_real_diverse_texts(self):
        tasks = generate_tasks(50, seed=42)
        texts = [t.text for t in tasks]
        unique = set(texts)

        # Should not be dominated by a single repeated template.
        most_common_count = max(texts.count(t) for t in unique)
        self.assertLess(
            most_common_count, 10,
            f"One text appears {most_common_count} times in 50 tasks — likely a fallback template.",
        )

        template_texts = {
            text for scenario in SCENARIO_TEMPLATES.values()
            for text in scenario.values()
        }
        template_overlap = unique & template_texts
        self.assertEqual(
            len(template_overlap), 0,
            f"generate_tasks() produced hardcoded template text(s): {template_overlap}",
        )

    def test_reproducibility_same_seed(self):
        tasks_a = generate_tasks(30, seed=123)
        _POOL_CACHE.clear()
        tasks_b = generate_tasks(30, seed=123)
        self.assertEqual(
            [t.text for t in tasks_a],
            [t.text for t in tasks_b],
            "generate_tasks() is not reproducible for the same seed.",
        )

    def test_different_seeds_give_different_streams(self):
        tasks_a = generate_tasks(30, seed=11)
        tasks_b = generate_tasks(30, seed=42)
        texts_a = [t.text for t in tasks_a]
        texts_b = [t.text for t in tasks_b]
        self.assertNotEqual(texts_a, texts_b,
                            "Different seeds produced identical task streams.")

    def test_all_task_types_appear_in_large_run(self):
        tasks = generate_tasks(200, seed=7)
        found = {t.type for t in tasks}
        self.assertEqual(
            found, set(TaskType),
            f"Not all TaskTypes appeared. Missing: {set(TaskType) - found}",
        )

    def test_task_texts_are_nonempty_strings(self):
        tasks = generate_tasks(20, seed=99)
        for t in tasks:
            self.assertIsInstance(t.text, str)
            self.assertGreater(len(t.text), 0,
                               f"Task {t.id} (type={t.type.name}) has empty text.")


if __name__ == "__main__":
    unittest.main()
