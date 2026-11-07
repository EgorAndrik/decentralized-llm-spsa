"""Test-wide settings: the suite must not use the network or depend on files earlier runs created.

Before this file existed the tests silently downloaded XSum, OPUS-100 and
Hermes into data/.hf_cache, so a clean checkout (CI) skipped 12 more tests
than a developer's second run, and the downloads were not tied to the pinned
revisions.
"""
import os

os.environ.setdefault("LLM_SPSA_OFFLINE", "1")
os.environ.setdefault("HF_DATASETS_OFFLINE", "1")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
