from contextlib import closing

import pytest

from src.routing_knowledge import RoutingKnowledge


def test_local_experience_publishes_once_per_round_and_survives_reopen(tmp_path):
    path = tmp_path / "knowledge.sqlite3"
    scope = {"mode": "api", "agents": ["a"], "controllers": 3}
    with closing(RoutingKnowledge(path, scope, round_size=2, mvp_rounds=2)) as knowledge:
        assert knowledge.record(0, "QA", 0, success=False, quality=0.1, service_seconds=3.0) is None
        assert knowledge.estimate(0, "QA", 0, 0.5, 2.0) == pytest.approx((0.4, 2.25))
        assert knowledge.estimate(2, "QA", 0, 0.5, 2.0) == (0.5, 2.0)

    with closing(RoutingKnowledge(path, scope)) as knowledge:
        assert knowledge.record(1, "QA", 0, success=True, quality=0.7, service_seconds=1.0) == 1
        assert knowledge.estimate(2, "QA", 0, 0.5, 2.0) == pytest.approx((0.46, 2.0))
        assert knowledge.estimate(0, "QA", 0, 0.5, 2.0) == pytest.approx((0.4, 2.25))
        assert not knowledge.ready
        assert knowledge.record(0, "QA", 0, success=True, quality=0.9, service_seconds=2.0) is None
        assert knowledge.estimate(2, "QA", 0, 0.5, 2.0) == pytest.approx((0.46, 2.0))
        assert knowledge.record(1, "QA", 0, success=True, quality=0.9, service_seconds=2.0) == 2
        assert knowledge.estimate(2, "QA", 0, 0.5, 2.0) == pytest.approx((4.1 / 7, 2.0))
        assert knowledge.observations == 4
        assert knowledge.ready

    with pytest.raises(ValueError, match="different mode or model pool"):
        RoutingKnowledge(path, {**scope, "mode": "simulation"})
