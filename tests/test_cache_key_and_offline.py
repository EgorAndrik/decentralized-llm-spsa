import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from src import dataset_loader
from src.run_real_llm_comparison import PromptCachingLLMClient


class CountingRaw:
    def __init__(self):
        self.calls = 0

    def call(self, model_name, prompt, task_type):
        self.calls += 1
        return {"output": prompt[-20:], "success": True, "latency": 0.1}


class TestFullPromptCacheKey(unittest.TestCase):
    def test_prompts_sharing_a_long_prefix_do_not_share_an_answer(self):
        prefix = "x" * 500
        client = PromptCachingLLMClient(CountingRaw())
        a = client.call("m", prefix + " ending A", "t")
        b = client.call("m", prefix + " ending B", "t")
        self.assertNotEqual(a["output"], b["output"])
        self.assertEqual(client._raw.calls, 2)
        client.call("m", prefix + " ending A", "t")
        self.assertEqual(client._raw.calls, 2)  # the identical prompt is served from the cache

    def test_cache_round_trips_through_json(self):
        raw = CountingRaw()
        client = PromptCachingLLMClient(raw)
        long_prompt = "y" * 600
        client.call("m", long_prompt, "t")
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "cache.json"
            client.save(path)
            restored = PromptCachingLLMClient.load(path, raw)
        restored.call("m", long_prompt, "t")
        self.assertEqual(raw.calls, 1)


class TestOfflineAndRevisionKeyedCache(unittest.TestCase):
    def test_tests_run_offline(self):
        self.assertEqual(os.environ.get("LLM_SPSA_OFFLINE"), "1")

    def test_download_is_refused_offline_and_pool_falls_back(self):
        with tempfile.TemporaryDirectory() as tmp, \
                mock.patch.object(dataset_loader, "_HF_CACHE", Path(tmp)), \
                mock.patch.dict(dataset_loader._POOL_CACHE, {}, clear=True):
            with self.assertRaisesRegex(RuntimeError, "download disabled"):
                dataset_loader._load_xsum()
            pool = dataset_loader.get_task_pool("SUMMARIZATION")
            self.assertEqual(pool, [dataset_loader._FALLBACKS["SUMMARIZATION"]])
            self.assertFalse(any(Path(tmp).iterdir()))  # nothing was written

    def test_cache_names_contain_the_pinned_revision(self):
        for name, repo in ((dataset_loader.XSUM_CACHE, "EdinburghNLP/xsum"),
                           (dataset_loader.OPUS100_CACHE, "Helsinki-NLP/opus-100"),
                           (dataset_loader.HERMES_CACHE, "NousResearch/hermes-function-calling-v1")):
            self.assertIn(dataset_loader.HF_REVISIONS[repo][:8], name)

    def test_cache_for_another_revision_is_not_used(self):
        with tempfile.TemporaryDirectory() as tmp, mock.patch.object(dataset_loader, "_HF_CACHE", Path(tmp)):
            (Path(tmp) / "xsum_documents.json").write_text(json.dumps(["stale text"]))  # the old, unkeyed name
            with self.assertRaises(RuntimeError):
                dataset_loader._load_xsum()


if __name__ == "__main__":
    unittest.main()
