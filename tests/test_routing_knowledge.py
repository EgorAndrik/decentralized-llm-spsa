from contextlib import closing

import pytest

from src.routing_knowledge import RoutingKnowledge


def test_local_experience_publishes_once_per_round_and_survives_reopen(knowledge_dsn):
    scope = {"mode": "api", "agents": ["a"], "controllers": 3}
    with closing(RoutingKnowledge(knowledge_dsn, scope, round_size=2, mvp_rounds=2)) as knowledge:
        assert knowledge.record(0, "QA", 0, success=False, quality=0.1, service_seconds=3.0, prompt_digest="a") is None
        assert knowledge.estimate(0, "QA", 0, 0.5, 2.0) == pytest.approx((0.3, 2.25))
        assert knowledge.estimate(2, "QA", 0, 0.5, 2.0) == (0.5, 2.0)

    with closing(RoutingKnowledge(knowledge_dsn, scope)) as knowledge:
        assert knowledge.record(1, "QA", 0, success=True, quality=0.7, service_seconds=1.0) == 1
        assert knowledge.estimate(2, "QA", 0, 0.5, 2.0) == pytest.approx((0.368, 2.0))
        assert knowledge.estimate(0, "QA", 0, 0.5, 2.0) == pytest.approx((0.3, 2.25))
        assert not knowledge.ready
        assert knowledge.record(0, "QA", 0, success=True, quality=0.9, service_seconds=2.0) is None
        assert knowledge.estimate(2, "QA", 0, 0.5, 2.0) == pytest.approx((0.368, 2.0))
        assert knowledge.record(2, "QA", 0, success=True, quality=0.9, service_seconds=2.0, prompt_digest="b") == 2
        assert knowledge.estimate(3, "QA", 0, 0.5, 2.0) == pytest.approx((4.1 * 6 / 49, 2.0))
        assert knowledge.observations == 4
        assert knowledge.ready

    with pytest.raises(ValueError, match="different mode or model pool"):
        RoutingKnowledge(knowledge_dsn, {**scope, "mode": "simulation"})


def test_failed_calls_survive_restart_and_do_not_advance_round(knowledge_dsn):
    scope = {"mode": "api", "agents": ["a", "b"], "controllers": 1}
    with closing(RoutingKnowledge(knowledge_dsn, scope, round_size=1, mvp_rounds=1)) as knowledge:
        knowledge.record(0, "QA", 0, success=True, quality=0.8, service_seconds=2, prompt_digest="same")
        knowledge.record(0, "QA", 1, success=True, quality=0.8, service_seconds=2, prompt_digest="same")
        knowledge.record_failure(0, "QA", 0)
        assert not knowledge.ready

    with closing(RoutingKnowledge(knowledge_dsn, scope)) as knowledge:
        assert knowledge.observations == 2 and knowledge.failures == 1
        assert knowledge.estimate(0, "QA", 0, 0.5, 2)[0] < knowledge.estimate(0, "QA", 1, 0.5, 2)[0]
        knowledge.record(0, "QA", 1, success=True, quality=0.8, service_seconds=2, prompt_digest="different")
        assert knowledge.ready


def test_readiness_changes_only_when_coverage_is_published(knowledge_dsn):
    scope = {"mode": "api", "agents": ["a"], "controllers": 1}
    with closing(RoutingKnowledge(knowledge_dsn, scope, round_size=2, mvp_rounds=1)) as knowledge:
        for digest in ("a", "a", "b"):
            knowledge.record(0, "QA", 0, success=True, quality=0.8, service_seconds=1, prompt_digest=digest)
        assert knowledge.coverage()[2] == 2
        assert not knowledge.ready
        knowledge.record(0, "QA", 0, success=True, quality=0.8, service_seconds=1, prompt_digest="b")
        assert knowledge.ready
        knowledge.record(0, "QA", 0, success=True, quality=0.8, service_seconds=1, prompt_digest="a")
        assert knowledge.ready


def test_concurrent_writers_publish_one_complete_round(knowledge_dsn):
    from concurrent.futures import ThreadPoolExecutor
    from threading import Barrier

    scope = {"mode": "api", "agents": ["a", "b"], "controllers": 2}
    ready_to_write = Barrier(2)

    def record(index):
        with closing(RoutingKnowledge(knowledge_dsn, scope, round_size=2, mvp_rounds=1)) as knowledge:
            ready_to_write.wait(timeout=5)
            return knowledge.record(
                index, "QA", index, success=True, quality=0.8, service_seconds=1, prompt_digest=str(index),
            )

    with ThreadPoolExecutor(max_workers=2) as executor:
        results = list(executor.map(record, range(2)))
    assert results.count(None) == results.count(1) == 1
    with closing(RoutingKnowledge(knowledge_dsn, scope)) as knowledge:
        assert knowledge.observations == 2
        assert knowledge.ready
        assert knowledge.estimate(2, "QA", 1, 0.5, 2) == pytest.approx((0.575, 1.75))


def test_import_preserves_snapshot_and_unpublished_experience(knowledge_dsn, tmp_path):
    import json
    import sqlite3

    from src.migrate_routing_knowledge import migrate

    source = tmp_path / "legacy.sqlite3"
    scope = {"mode": "api", "agents": ["a"], "controllers": 1}
    with closing(sqlite3.connect(source)) as db, db:
        db.executescript("""
            CREATE TABLE config (id INTEGER PRIMARY KEY, value TEXT NOT NULL);
            CREATE TABLE experience (
                controller_id INTEGER, task_type TEXT, agent_id INTEGER, observations INTEGER,
                successes INTEGER, quality_sum REAL, service_seconds REAL
            );
            INSERT INTO experience VALUES (0, 'QA', 0, 3, 3, 2.5, 5.0), (-1, 'QA', 0, 2, 2, 1.6, 4.0);
            CREATE TABLE execution_failures (controller_id INTEGER, task_type TEXT, agent_id INTEGER, failures INTEGER);
            INSERT INTO execution_failures VALUES (0, 'QA', 0, 2), (-1, 'QA', 0, 1);
            CREATE TABLE evaluated_prompts (digest TEXT PRIMARY KEY);
            INSERT INTO evaluated_prompts VALUES ('a'), ('b');
            CREATE TABLE published_readiness (id INTEGER PRIMARY KEY, ready INTEGER);
            INSERT INTO published_readiness VALUES (1, 1);
        """)
        config = {"version": 1, "scope": scope, "round_size": 2, "mvp_rounds": 1}
        db.execute("INSERT INTO config VALUES (1, ?)", (json.dumps(config),))

    migrate(source, knowledge_dsn)
    with pytest.raises(ValueError, match="already contains routing data"):
        migrate(source, knowledge_dsn)
    with closing(RoutingKnowledge(knowledge_dsn, scope)) as knowledge:
        assert knowledge.observations == 3
        assert knowledge.completed_rounds == 1
        assert knowledge.failures == 2
        assert knowledge.coverage() == ({0}, {0}, 2)
        assert knowledge.ready
        assert knowledge.estimate(1, "QA", 0, 0.5, 2) == pytest.approx((0.62 * 5 / 6, 2.0))
        assert knowledge.record(0, "QA", 0, success=True, quality=0.8, service_seconds=2) == 2
