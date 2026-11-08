from contextlib import closing

import pytest

from src.routing_knowledge import RoutingKnowledge


def test_local_experience_publishes_once_per_round_and_survives_reopen(tmp_path):
    path = tmp_path / "knowledge.sqlite3"
    scope = {"mode": "api", "agents": ["a"], "controllers": 3}
    with closing(RoutingKnowledge(path, scope, round_size=2, mvp_rounds=2)) as knowledge:
        assert knowledge.record(0, "QA", 0, success=False, quality=0.1, service_seconds=3.0, prompt_digest="a") is None
        assert knowledge.estimate(0, "QA", 0, 0.5, 2.0) == pytest.approx((0.3, 2.25))
        assert knowledge.estimate(2, "QA", 0, 0.5, 2.0) == (0.5, 2.0)

    with closing(RoutingKnowledge(path, scope)) as knowledge:
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
        RoutingKnowledge(path, {**scope, "mode": "simulation"})


def test_failed_calls_survive_restart_and_do_not_advance_round(tmp_path):
    path = tmp_path / "knowledge.sqlite3"
    scope = {"mode": "api", "agents": ["a", "b"], "controllers": 1}
    with closing(RoutingKnowledge(path, scope, round_size=1, mvp_rounds=1)) as knowledge:
        knowledge.record(0, "QA", 0, success=True, quality=0.8, service_seconds=2, prompt_digest="same")
        knowledge.record(0, "QA", 1, success=True, quality=0.8, service_seconds=2, prompt_digest="same")
        knowledge.record_failure(0, "QA", 0)
        assert not knowledge.ready

    with closing(RoutingKnowledge(path, scope)) as knowledge:
        assert knowledge.observations == 2 and knowledge.failures == 1
        assert knowledge.estimate(0, "QA", 0, 0.5, 2)[0] < knowledge.estimate(0, "QA", 1, 0.5, 2)[0]
        knowledge.record(0, "QA", 1, success=True, quality=0.8, service_seconds=2, prompt_digest="different")
        assert knowledge.ready


def test_readiness_changes_only_when_coverage_is_published(tmp_path):
    scope = {"mode": "api", "agents": ["a"], "controllers": 1}
    with closing(RoutingKnowledge(tmp_path / "knowledge.sqlite3", scope, round_size=2, mvp_rounds=1)) as knowledge:
        for digest in ("a", "a", "b"):
            knowledge.record(0, "QA", 0, success=True, quality=0.8, service_seconds=1, prompt_digest=digest)
        assert knowledge.coverage()[2] == 2
        assert not knowledge.ready
        knowledge.record(0, "QA", 0, success=True, quality=0.8, service_seconds=1, prompt_digest="b")
        assert knowledge.ready
        knowledge.record(0, "QA", 0, success=True, quality=0.8, service_seconds=1, prompt_digest="a")
        assert knowledge.ready
