import re
import unittest

from src import dataset_loader, it_benchmark, it_dataset_loader, prompt_embeddings, real_benchmarks

SHA = re.compile(r"^[0-9a-f]{40}$")


class TestPinnedRevisions(unittest.TestCase):
    def test_corpus_revisions_are_commit_hashes(self):
        for dataset, revision in dataset_loader.HF_REVISIONS.items():
            self.assertRegex(revision, SHA, dataset)

    def test_it_sources_are_pinned(self):
        for skill, source in it_dataset_loader.DATASET_SOURCES.items():
            pinned = source.revision or (source.data_file and re.search(r"@([0-9a-f]{40})/", source.data_file).group(1))
            self.assertRegex(pinned or "", SHA, skill)

    def test_benchmark_revisions_are_full_hashes(self):
        # a 7-character prefix is a weak pin: it only has to stay unambiguous
        for revision in (real_benchmarks.SPROUT_REVISION, real_benchmarks.ROUTERBENCH_REVISION,
                         real_benchmarks.LLMROUTERBENCH_REVISION, it_benchmark.STACKPULSE_COMMIT):
            self.assertRegex(revision, SHA)
        self.assertTrue(it_benchmark.STACKPULSE_COMMIT.startswith(it_benchmark.STACKPULSE_REVISION))

    def test_encoder_is_pinned(self):
        self.assertRegex(prompt_embeddings.ENCODER_REVISION, SHA)


if __name__ == "__main__":
    unittest.main()
