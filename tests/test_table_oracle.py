"""Measured-outcome replay: the table is replayed as recorded and nothing leaks."""
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src import real_benchmarks
from src.paper_experiment import ControlledReplayExperiment, ExperimentConfig

SPROUT_RAW = Path("data/external/sprout") / real_benchmarks.SPROUT_FILES[0]
AZURE = Path("data/external/azure/AzureLLMInferenceTrace_code.csv")
requires_sprout = pytest.mark.skipif(
    not SPROUT_RAW.exists() or not AZURE.exists(),
    reason="SPROUT or Azure trace absent; fetch with python -m src.real_benchmarks sprout --download",
)


def test_split_is_a_pure_function_of_the_prompt_identity():
    assert real_benchmarks._split_of("sprout:abc") == real_benchmarks._split_of("sprout:abc")
    splits = [real_benchmarks._split_of(f"x:{i}") for i in range(5000)]
    shares = {name: splits.count(name) / len(splits) for name in ("train", "validation", "test")}
    assert abs(shares["train"] - 0.6) < 0.03 and abs(shares["test"] - 0.2) < 0.03


def test_replay_drops_whole_prompts_with_invalid_input_tokens_in_builds_and_caches(tmp_path):
    items = pd.DataFrame({"item_id": ["ok", "nan", "inf"], "domain_name": ["knowledge"] * 3})
    outcomes = pd.DataFrame([
        {
            "item_id": item_id, "model": model, "quality": 1.0, "cost_usd": 0.01,
            "input_tokens": tokens if model == "a" else 10, "output_tokens": 20,
        }
        for item_id, tokens in [("ok", 10), ("nan", np.nan), ("inf", np.inf)]
        for model in ("a", "b")
    ])
    built_items, built_outcomes = real_benchmarks._finish(items, outcomes, real_benchmarks.SPROUT_DOMAINS)
    assert built_items["item_id"].tolist() == ["ok"]
    assert built_outcomes["model"].tolist() == ["a", "b"]

    cache = tmp_path / "processed"
    cache.mkdir()
    items.to_parquet(cache / "sprout_items.parquet", index=False)
    outcomes.to_parquet(cache / "sprout_outcomes.parquet", index=False)
    cached_items, cached_outcomes = real_benchmarks.load("sprout", tmp_path)
    assert cached_items["item_id"].tolist() == ["ok"]
    assert cached_outcomes["item_id"].tolist() == ["ok", "ok"]


def test_domain_maps_cover_six_families_and_reject_unknown_sources():
    assert len(real_benchmarks.SPROUT_DOMAINS) == len(real_benchmarks.ROUTERBENCH_DOMAINS) == 6
    assert real_benchmarks._sprout_domain("rungalileo/ragbench/finqa") == "rag_financial_legal"
    assert real_benchmarks._sprout_domain("lighteval/MATH/all/test") == "math"
    assert real_benchmarks._routerbench_domain("mmlu-anatomy") == "knowledge"
    with pytest.raises(ValueError):
        real_benchmarks._sprout_domain("unknown/dataset")


@pytest.fixture(scope="module")
def sprout():
    from src.table_oracle import OutcomeTable, TableConfig, TableOracle, make_table_agents
    table = OutcomeTable("sprout", TableConfig(), seed=3)
    agents = make_table_agents(table)
    return table, agents, TableOracle(table, agents)


@requires_sprout
def test_oracle_returns_the_recorded_outcome(sprout):
    from src.table_oracle import load_table_tasks
    table, agents, oracle = sprout
    _, outcomes = real_benchmarks.load("sprout")
    task = load_table_tasks(table, agents, seed=3, rho=0.5, limit=5)[0]
    for agent in agents[:4]:
        recorded = outcomes[(outcomes["item_id"] == task.source_id)
                            & (outcomes["model"] == table.models[agent.agent_id])].iloc[0]
        got = oracle.outcome(task, agent.agent_id)
        assert got.quality == pytest.approx(recorded["quality"])
        assert got.cost == pytest.approx(recorded["cost_usd"])
        assert got.success == bool(recorded["success"])


@requires_sprout
def test_profiles_and_hardness_use_the_train_split_only(sprout):
    table, _, _ = sprout
    capability, _, _ = table.train_profiles()
    # Rewriting every test outcome must not move any pre-execution covariate.
    saved = table.quality
    test_rows = [table.row(i) for i in table.items_for("test")["item_id"]]
    table.quality = saved.copy()
    table.quality[test_rows] = 0.0
    try:
        assert np.allclose(table.train_profiles()[0], capability)
        assert table._group_difficulty() == table.hardness
    finally:
        table.quality = saved


@requires_sprout
def test_offered_load_is_relative_to_the_streams_own_mix(sprout):
    from src.table_oracle import load_table_tasks
    table, agents, _ = sprout
    low = load_table_tasks(table, agents, seed=3, rho=0.25, limit=400)
    high = load_table_tasks(table, agents, seed=3, rho=1.00, limit=400)
    # Same trace window, four times the rate.
    assert low[-1].arrival == pytest.approx(4.0 * high[-1].arrival)
    assert all(t.deadline > t.arrival for t in low)


@requires_sprout
def test_measured_replay_runs_through_the_unchanged_architecture(sprout):
    from src.table_oracle import load_table_tasks
    table, agents, oracle = sprout
    tasks = load_table_tasks(table, agents, seed=3, rho=0.5, limit=120)
    config = ExperimentConfig(controllers=3, agents=len(agents), rho=0.5, seed=3,
                              architecture="global_exact", optimizer="aspsa")
    summary = ControlledReplayExperiment(tasks, config, agents=agents, oracle=oracle).run()
    assert summary["tasks"] == 120
    assert summary["rho_effective"] == pytest.approx(0.5)
    assert summary["random_quality"] <= summary["oracle_quality"]
    assert summary["oracle_cost_usd"] <= summary["best_single_cost_usd"] * 10
    assert 0.0 <= summary["deadline_hit_rate"] <= 1.0


def test_frontier_metrics_follow_the_routerbench_definition():
    import pandas as pd
    from src.run_real_benchmark import frontier_metrics
    points = pd.DataFrame({"mean_cost": [1.0, 2.0, 3.0], "mean_quality": [0.5, 0.7, 0.6]})
    result = frontier_metrics(points, lo=0.0, hi=4.0, best_q=0.7, best_c=2.5)
    # Non-decreasing hull: 0 on [0,1), 0.5 on [1,2), 0.7 on [2,4].
    assert result["aiq"] == pytest.approx((0.5 * 1 + 0.7 * 2) / 4, abs=0.01)
    assert result["quality_at_best_cost"] == pytest.approx(0.7)
    assert result["cost_at_best_quality"] == pytest.approx(2.0)


EMBEDDINGS = Path("data/external/processed/sprout_emb_all-MiniLM-L6-v2.parquet")
requires_embeddings = pytest.mark.skipif(
    not EMBEDDINGS.exists(), reason="prompt embeddings absent; build with python -m src.prompt_embeddings sprout")


@requires_sprout
@requires_embeddings
def test_prompt_features_respect_the_norm_bound_and_leave_one_out():
    from src.prompt_embeddings import PromptFeatures
    features = PromptFeatures("sprout", dim=8)
    assert features.reduced.shape[1] == 8
    assert np.linalg.norm(features.reduced, axis=1).max() <= 1.0 + 1e-9
    items, outcomes = real_benchmarks.load("sprout")
    train_id = items.loc[items["split"] == "train", "item_id"].iloc[0]
    row = features.index[train_id]
    similarity = features.vectors @ features.vectors[row]
    train = (items["split"] == "train").to_numpy()
    similarity[~train] = -np.inf
    similarity[row] = -np.inf
    nearest = np.argsort(-similarity)[:10]
    wide = outcomes.pivot(index="item_id", columns="model", values="quality").loc[features.item_ids]
    expected = wide.to_numpy()[nearest].mean(axis=0)
    assert np.allclose(features.knn(train_id), expected)


@requires_sprout
@requires_embeddings
def test_embedding_reaches_the_predictors_and_keeps_the_feature_bound(sprout):
    from src.prompt_embeddings import PromptFeatures
    from src.table_oracle import TableOracle, load_table_tasks
    table, agents, _ = sprout
    prompts = PromptFeatures("sprout", dim=4)
    tasks = load_table_tasks(table, agents, seed=3, rho=0.5, limit=60, prompt_features=prompts)
    assert all(len(t.embedding) == 4 for t in tasks)
    config = ExperimentConfig(controllers=3, agents=len(agents), rho=0.5, seed=3,
                              architecture="global_exact", optimizer="local_rls",
                              embedding_dim=4)
    experiment = ControlledReplayExperiment(tasks, config, agents=agents,
                                            oracle=TableOracle(table, agents, prompts))
    assert experiment.controllers[0].feature_dim == 6 + 5 + 4
    x = experiment._prediction_features(tasks[0], 0.0)
    assert np.linalg.norm(x) <= 1.0 + 1e-9
    summary = experiment.run()
    assert summary["tasks"] == 60
    with pytest.raises(ValueError):
        ControlledReplayExperiment(tasks, replace_dim(config, 0), agents=agents,
                                   oracle=TableOracle(table, agents, prompts))


def replace_dim(config, dim):
    from dataclasses import replace
    return replace(config, embedding_dim=dim)


@requires_sprout
def test_slots_of_one_model_share_one_predictor(sprout):
    """Replicas are capacity, not separate predictors (regression)."""
    from src.paper_experiment import PaperController
    from src.paper_losses import LossObservation
    from src.table_oracle import make_table_agents
    table, _, _ = sprout
    agents = make_table_agents(table, slots_per_model=3)
    models = len(table.models)
    config = ExperimentConfig(controllers=1, agents=len(agents), optimizer="local_rls")
    controller = PaperController(0, agents, config)
    assert controller.rows == models
    assert controller.theta_quality.shape[0] == models
    x = np.zeros(controller.feature_dim)
    x[0], x[6] = 0.4, 0.2
    before = controller.predict_quality(models, x)
    controller.observe(LossObservation(x, 0, 1.0, 0.0, 1.0, 0.5, 0.2))
    # Feedback from slot 0 moves the prediction of slot M, the same model.
    assert controller.predict_quality(models, x) != pytest.approx(before)
    assert controller.predict_quality(models, x) == pytest.approx(controller.predict_quality(0, x))
