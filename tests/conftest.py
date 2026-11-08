"""Test-wide settings: the suite runs offline except for the configured PostgreSQL test database.

Before this file existed the tests silently downloaded XSum, OPUS-100 and
Hermes into data/.hf_cache, so a clean checkout (CI) skipped 12 more tests
than a developer's second run, and the downloads were not tied to the pinned
revisions.
"""
import os
from uuid import uuid4

import psycopg
from psycopg import sql
from psycopg.conninfo import conninfo_to_dict, make_conninfo
import pytest

os.environ.setdefault("LLM_SPSA_OFFLINE", "1")
os.environ.setdefault("HF_DATASETS_OFFLINE", "1")
os.environ.setdefault("HF_HUB_OFFLINE", "1")


@pytest.fixture
def knowledge_dsn():
    dsn = os.environ.get("TEST_ROUTING_DATABASE_URL")
    if not dsn:
        pytest.skip("Set TEST_ROUTING_DATABASE_URL to run the PostgreSQL integration tests")
    schema_name = f"routing_test_{uuid4().hex}"
    schema = sql.Identifier(schema_name)
    options = conninfo_to_dict(dsn).get("options", "")
    with psycopg.connect(dsn, autocommit=True, connect_timeout=5) as connection:
        connection.execute(sql.SQL("CREATE SCHEMA {}").format(schema))
        try:
            yield make_conninfo(dsn, options=f"{options} -csearch_path={schema_name}".strip())
        finally:
            connection.execute(sql.SQL("DROP SCHEMA {} CASCADE").format(schema))
