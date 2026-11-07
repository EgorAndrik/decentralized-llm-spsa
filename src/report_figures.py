"""Figures and figure data for REAL_BENCHMARK_REPORT.md.

Every figure is computed from the artefacts of the measured-outcome runs
(``real_benchmark_outputs/...``) or from re-simulations of the weights and
configurations those runs selected; nothing is re-tuned here.  Intermediate
tables are cached in ``figures/data`` so a figure can be redrawn without
re-simulating.

    python -m src.report_figures            # everything
    python -m src.report_figures fig2 fig3  # selected figures
"""
from __future__ import annotations

import os

for _name in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_name, "1")

import argparse
from concurrent.futures import ProcessPoolExecutor
from dataclasses import replace
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from . import table_oracle
from .assignment_buffer import BufferPolicy, GlobalAssignmentBuffer
from .paper_experiment import ControlledReplayExperiment, ExperimentConfig
from .paper_optimizers import OptimizerConfig
from .policy_real import _world, family, site_domains
from .policy_search import DEFAULTS, WEIGHT_NAMES

OUT = Path("figures")
DATA = OUT / "data"
RUNS = Path("real_benchmark_outputs")
# Which policy-layer runs the policy figures show.  "5": the manuscript's five
# buffer weights; "6": deadline urgency added (policy6_*), so that
# earliest-deadline-first is a member of the family the optimizers search.
POLICY_FAMILY = "6"
POLICY_RUNS = {
    "5": dict(trace="policy_real_trace", ablation="policy_real_ablation",
              tuning=("policy_real_tuned", "policy_real_tuned_wide"),
              budget=(("low_noise", "policy_real_trace", 0), ("low_noise", "policy_real_tuned", 4),
                      ("low_noise", "policy_real_tuned_wide", 7), ("noisy", "policy_real_trace", 0),
                      ("noisy", "policy_real_tuned_noisy", 7)),
              rules="policy_real_rules", cache=""),
    "6": dict(trace="policy6_trace", ablation="policy6_ablation", tuning=("policy6_tuned",),
              budget=(("low_noise", "policy6_trace", 0), ("low_noise", "policy6_tuned", 7),
                      ("noisy", "policy6_trace", 0), ("noisy", "policy6_tuned_noisy", 7)),
              rules="policy6_trace", cache="_w6"),
}[POLICY_FAMILY]
POLICY = RUNS / POLICY_RUNS["trace"]
POLICY_FALLBACK = Path("real_benchmark_outputs/policy_real")
CACHE = POLICY_RUNS["cache"]
SYSTEM = Path("real_benchmark_outputs/llmrouterbench_v3")

# One colour per entity in every figure (validated categorical palette, light).
COLOR = {
    "aspsa": "#2a78d6", "spsa": "#eb6834", "zo_gt": "#1baf7a", "zo_pgd": "#eda100",
    "knn_router": "#e87ba4", "static_domain": "#4a3aa7", "local_rls": "#1baf7a",
    "hand-set defaults": "#8a8a85", "quality_greedy": "#b5b4ae", "cost_greedy": "#b5b4ae",
    "random": "#b5b4ae", "shortest_queue": "#b5b4ae", "oracle": "#2b2b29",
    "mlp_router": "#008300", "aspsa_mlp": "#5598e7",
}
LABEL = {
    "aspsa": "A-SPSA", "spsa": "SPSA", "zo_gt": "ZO-GT", "zo_pgd": "ZO-PGD",
    "hand-set defaults": "hand-set weights", "knn_router": "kNN router (RouterBench)",
    "static_domain": "best model per domain", "quality_greedy": "best model when free",
    "cost_greedy": "cheapest model", "random": "random", "shortest_queue": "shortest queue",
    "oracle": "per-prompt oracle", "local_rls": "our system (RLS only)",
    "mlp_router": "MLP router (RouterBench)", "aspsa_mlp": "our system, MLP prior",
}
MARKER = {"aspsa": "o", "spsa": "s", "zo_gt": "^", "zo_pgd": "D", "hand-set defaults": "x"}
METHODS = ("aspsa", "spsa", "zo_gt", "zo_pgd")

plt.rcParams.update({
    "figure.dpi": 150, "savefig.dpi": 200, "font.size": 9, "axes.titlesize": 10,
    "axes.labelsize": 9, "legend.fontsize": 8, "axes.spines.top": False,
    "axes.spines.right": False, "axes.grid": True, "grid.color": "#e6e6e3",
    "grid.linewidth": 0.6, "axes.edgecolor": "#8a8a85", "axes.axisbelow": True,
    "lines.linewidth": 2.0, "figure.facecolor": "#fcfcfb", "axes.facecolor": "#fcfcfb",
})


def _ci(values: np.ndarray, axis: int = 0) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    n = np.sum(np.isfinite(values), axis=axis)
    return 1.96 * np.nanstd(values, axis=axis, ddof=1) / np.sqrt(np.maximum(n, 1))


def _policy_frame() -> pd.DataFrame:
    path = POLICY / "policy_real_raw.csv"
    if not path.exists():
        path = POLICY_FALLBACK / "policy_real_raw.csv"
    return pd.read_csv(path)


def _policy_cfg() -> dict:
    manifest = POLICY / "experiment_manifest.json"
    if not manifest.exists():
        manifest = POLICY_FALLBACK / "experiment_manifest.json"
    return json.loads(manifest.read_text(encoding="utf-8"))["config"]


# ------------------------------------------------------------ simulations
def _simulate_records(weights: np.ndarray, cfg: dict, site: int, seed: int, rho: float,
                      order: str = "weighted") -> dict:
    """One test-stream simulation; returns per-task records and the summary."""
    table, agents, prompts = _world(cfg["benchmark"], cfg["slots"], cfg["embedding_dim"])
    tasks = table_oracle.load_table_tasks(
        table, agents, seed, rho, limit=cfg["tasks_per_site"], split="test",
        prompt_features=prompts, domains=site_domains(site, cfg["sites"]))
    config = ExperimentConfig(
        controllers=1, agents=len(agents), rho=rho, sync_interval=5, seed=seed,
        architecture="global_exact", optimizer="local_rls", embedding_dim=cfg["embedding_dim"],
        max_processing_units=100.0, max_waiting_units=200.0, profile_prior=True,
        rls_prior_strength=100.0, pair_quality_feature=True)
    experiment = ControlledReplayExperiment(
        tasks, config, agents=agents, oracle=table_oracle.TableOracle(table, agents, prompts))
    experiment.buffer = GlobalAssignmentBuffer(BufferPolicy(
        matching="exact", order=order, **dict(zip(family(cfg)[0], (float(v) for v in weights)))))
    summary = experiment.run()
    records = pd.DataFrame(experiment.task_records)[
        ["agent_id", "arrival", "assigned_at", "finish", "deadline_hit", "success", "quality_real"]]
    return {"summary": summary, "records": records, "agents": len(agents)}


def _policy_run(job: dict) -> list[dict]:
    """All sites of one (method, optimizer seed, test seed, rho) evaluation."""
    cfg, weights = job["cfg"], job["weights"]
    out = []
    for site in range(cfg["sites"]):
        result = _simulate_records(np.asarray(weights[site]), cfg, site, job["seed"], job["rho"],
                                   job.get("order", "weighted"))
        records = result["records"].assign(site=site)
        out.append({"method": job["method"], "optimizer_seed": job["optimizer_seed"],
                    "seed": job["seed"], "rho": job["rho"], "site": site,
                    "agents": result["agents"], "records": records.to_dict("list"),
                    **{k: result["summary"][k] for k in ("deadline_hit_rate", "goodput",
                                                         "mean_quality", "p95_wait")}})
    return out


def _policy_jobs(noise: str, rhos, test_seeds) -> list[dict]:
    frame = _policy_frame()
    cfg = _policy_cfg()
    jobs = []
    rows = frame[(frame["noise"] == noise) | (frame["method"] == "hand-set defaults")]
    for _, row in rows.iterrows():
        weights = json.loads(row["weights"])
        for rho in rhos:
            for seed in test_seeds:
                jobs.append({"cfg": cfg, "weights": weights, "method": row["method"],
                             "optimizer_seed": int(row["optimizer_seed"]), "seed": seed, "rho": rho})
    return jobs


def _run_jobs(jobs, workers: int) -> list[dict]:
    if workers <= 1:
        return [r for job in jobs for r in _policy_run(job)]
    with ProcessPoolExecutor(max_workers=workers) as pool:
        return [r for batch in pool.map(_policy_run, jobs, chunksize=1) for r in batch]


def policy_operational_data(workers: int) -> pd.DataFrame:
    """Test-stream simulations with the low-noise weights, at the tuning load."""
    # Parquet, not pickle: unpickling a cache file would execute code from it.
    path = DATA / f"policy_operational{CACHE}.parquet"
    if path.exists():
        return pd.read_parquet(path)
    cfg = _policy_cfg()
    rows = _run_jobs(_policy_jobs("low_noise", [cfg["rho"]], cfg["test_seeds"]), workers)
    frame = pd.DataFrame(rows)
    DATA.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(path, index=False)
    return frame


def load_sweep_data(workers: int) -> pd.DataFrame:
    path = DATA / f"policy_load_sweep{CACHE}.csv"
    if path.exists():
        return pd.read_csv(path)
    rows = _run_jobs(_policy_jobs("low_noise", (1.0, 1.5, 2.0, 2.5, 3.0), (11, 23)), workers)
    frame = pd.DataFrame(rows).drop(columns=["records"])
    DATA.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False)
    return frame


RULES = ("edf", "llf", "sjf")
RULE_STYLE = {"edf": ("EDF", (0, (4, 2))), "llf": ("least slack first", (0, (1, 2))),
              "sjf": ("shortest job first", (0, (6, 2, 1, 2)))}


def load_sweep_rules_data(workers: int) -> pd.DataFrame:
    """The classic rules at the loads of the sweep, on the sweep's test streams."""
    path = DATA / "policy_load_sweep_rules.csv"
    if path.exists():
        return pd.read_csv(path)
    cfg = _policy_cfg()
    weights = [family(cfg)[1].tolist()] * cfg["sites"]
    jobs = [{"cfg": cfg, "weights": weights, "method": rule, "optimizer_seed": -1, "seed": seed,
             "rho": rho, "order": rule}
            for rule in RULES for rho in (1.0, 1.5, 2.0, 2.5, 3.0) for seed in (11, 23)]
    frame = pd.DataFrame(_run_jobs(jobs, workers)).drop(columns=["records"])
    DATA.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False)
    return frame


def _prediction_run(job: dict) -> list[dict]:
    """Main-system run keeping per-round Phi values, with the selected settings.

    Uses exactly the per-block steps, RLS prior and mu - eta of the system run
    it illustrates (``SYSTEM_PREDICTION``), so the figure and the tables agree.
    """
    from . import run_real_benchmark as R
    regime = job["regime"]
    slots, rho, latency_weight = R.REGIMES[regime]
    run_job = dict(
        benchmark="llmrouterbench", tasks=1000, per_domain=None, controllers=3, embedding_dim=8,
        controller_assignment="load",
        table_cfg=dict(time_unit_seconds=30.0, arrival_process="azure_code", slo_scale=3.0),
        split="test", seed=job["seed"], price_weight=-0.08, step_size=job.get("step", 0.04),
        max_processing_units=100.0, max_waiting_units=200.0, rls_prior_strength=job["prior"],
        pair_quality_feature=True, diffusion_mixing=0.0, specialization=0.5,
        regime=regime, slots=slots, rho=rho, latency_weight=latency_weight, system=job["system"])
    if job.get("block_steps"):
        run_job["block_steps"] = job["block_steps"]
    rows = []
    original = ControlledReplayExperiment.run

    def run_and_keep(self):
        summary = original(self)
        rows.extend({"system": job["system"], "seed": job["seed"], "regime": regime,
                     **{k: r[k] for k in ("round", "phi_p", "phi_w", "phi_q", "data_p", "data_w",
                                          "data_q", "disagreement_p", "disagreement_q")}}
                    for r in self.round_records)
        return summary

    ControlledReplayExperiment.run = run_and_keep
    try:
        R.run_one(run_job)
    finally:
        ControlledReplayExperiment.run = original
    return rows


SYSTEM_PREDICTION = Path("real_benchmark_outputs/llmrouterbench_v4_blocks")
SYSTEM_MLP = Path("real_benchmark_outputs/llmrouterbench_v6_mlp")   # MLP router and the MLP-prior system


EQUAL_STEP = Path("real_benchmark_outputs/llmrouterbench_v5_equal_step")
FIG1B_STEP = 0.01


def prediction_rounds_data(workers: int, regime: str = "capacity") -> pd.DataFrame:
    """Per-round Phi of the equal-step run (every method, every block, h = 0.01).

    Same configuration as ``llmrouterbench_v5_equal_step``: the figure shows the
    rounds behind the test-seed numbers reported for that run.
    """
    path = DATA / f"prediction_rounds_{regime}_h{FIG1B_STEP:g}.csv"
    if path.exists():
        return pd.read_csv(path)
    manifest = json.loads((EQUAL_STEP / "experiment_manifest.json").read_text(encoding="utf-8"))
    prior = manifest["selected_rls_prior_strength"][regime]
    jobs = []
    for system in ("aspsa", "spsa", "zo_gt", "local_rls"):
        extra = ({} if system == "local_rls"
                 else {"block_steps": {b: FIG1B_STEP for b in ("processing", "waiting", "quality")}})
        jobs += [{"system": system, "seed": seed, "prior": prior, "regime": regime, **extra}
                 for seed in range(1, 11)]
    if workers <= 1:
        rows = [r for job in jobs for r in _prediction_run(job)]
    else:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            rows = [r for batch in pool.map(_prediction_run, jobs, chunksize=1) for r in batch]
    frame = pd.DataFrame(rows)
    DATA.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False)
    return frame


def _system_runs(regime: str) -> pd.DataFrame:
    """Baseline systems from llmrouterbench_v3, learned systems from v4.

    v4 re-ran the learned systems with per-block steps and mu - eta = mu/2; a
    later directory replaces every (regime, system) it contains.
    """
    from .run_real_benchmark import merge_runs
    dirs = [d for d in (SYSTEM, SYSTEM_PREDICTION, SYSTEM_MLP) if (d / f"runs_{regime}.csv").exists()]
    runs, _ = merge_runs(dirs)
    return runs[runs["regime"] == regime]


# ------------------------------------------------------------ figures
def _save(fig, name: str) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / f"{name}.png", bbox_inches="tight")
    plt.close(fig)
    print("wrote", OUT / f"{name}.png")


def fig1_policy_dynamics(_workers: int) -> None:
    """Objective the optimizer minimises, per round, both oracle conditions."""
    frame = _policy_frame()
    if "trace" not in frame.columns:
        print("fig1: traces not available yet (policy_real_trace still running)")
        return
    fig, axes = plt.subplots(1, 2, figsize=(9.0, 3.4), sharey=True)
    for ax, noise, title in zip(axes, ("noisy", "low_noise"),
                                ("noisy oracle (1 stream per call)", "low-noise oracle (4 streams per call)")):
        for method in METHODS:
            part = frame[(frame["noise"] == noise) & (frame["method"] == method)]
            traces = np.array([json.loads(t) for t in part["trace"]], dtype=float)
            # Smooth over a 3-round window: each round is only two probe points.
            kernel = np.ones(3) / 3.0
            smooth = np.array([np.convolve(np.pad(t, 1, mode="edge"), kernel, "valid") for t in traces])
            rounds = np.arange(1, smooth.shape[1] + 1)
            mean, ci = smooth.mean(axis=0), _ci(smooth)
            ax.plot(rounds, mean, color=COLOR[method], label=LABEL[method],
                    marker=MARKER[method], markevery=5, markersize=5)
            ax.fill_between(rounds, mean - ci, mean + ci, color=COLOR[method], alpha=0.15, linewidth=0)
        ax.set_title(title)
        ax.set_xlabel("synchronisation round")
    axes[0].set_ylabel("deadline-miss rate on own traffic\n(objective, train streams)")
    axes[1].legend(loc="upper right", frameon=False)
    fig.suptitle("Policy layer: objective minimised by each method (mean ± 95% CI, 9 optimizer seeds)",
                 y=1.02)
    _save(fig, "fig1_policy_objective_dynamics")


REGIME_TITLE = {"api": "API regime (32 slots per model)", "capacity": "capacity regime (4 slots per model)",
                "stress": "stress regime (1 slot per model)"}


def fig1b_prediction_functionals(workers: int) -> None:
    """The processing-time functional Phi_p per synchronisation round, three regimes."""
    fig, axes = plt.subplots(1, 3, figsize=(12.0, 3.4))
    for ax, regime in zip(axes, REGIME_TITLE):
        frame = prediction_rounds_data(workers, regime)
        for system in ("local_rls", "spsa", "zo_gt", "aspsa"):
            part = frame[frame["system"] == system].pivot(index="seed", columns="round", values="phi_p")
            mean, ci = part.mean(axis=0), _ci(part.to_numpy())
            colour = COLOR[system] if system != "local_rls" else "#8a8a85"
            label = LABEL[system] if system != "local_rls" else "RLS only, no ZO layer"
            ax.plot(part.columns, mean, "--" if system == "local_rls" else "-", color=colour,
                    label=label, linewidth=1.6)
            ax.fill_between(part.columns, mean - ci, mean + ci, alpha=0.12, linewidth=0, color=colour)
        ax.set_title(REGIME_TITLE[regime])
        ax.set_xlabel("synchronisation round")
        ax.set_ylim(bottom=0)
    axes[0].set_ylabel("network functional Φ_p\n(processing time, rolling window)")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=len(labels), frameon=False,
               bbox_to_anchor=(0.5, -0.02))
    fig.suptitle(f"Prediction layer: Φ_p per synchronisation round at the equal step h = {FIG1B_STEP:g} "
                 f"(mean ± 95% CI, 10 seeds)", y=1.03)
    _save(fig, "fig1b_prediction_functionals")


def _epoch_utilisation(records: dict, agents: int, epochs: int = 10) -> tuple[np.ndarray, np.ndarray]:
    """Per-epoch mean utilisation across agents and its across-agent std."""
    r = pd.DataFrame(records)
    # Epochs span the arrival window, the period in which load is applied;
    # after the last arrival the system only drains its queue.
    start, end = float(r["arrival"].min()), float(r["arrival"].max())
    edges = np.linspace(start, end, epochs + 1)
    busy = np.zeros((agents, epochs))
    for agent, a0, a1 in zip(r["agent_id"], r["assigned_at"], r["finish"]):
        overlap = np.clip(np.minimum(a1, edges[1:]) - np.maximum(a0, edges[:-1]), 0.0, None)
        busy[int(agent)] += overlap
    utilisation = busy / np.diff(edges)[None, :]
    return utilisation.mean(axis=0), utilisation.std(axis=0)


def _epoch_deadline_rate(records: dict, epochs: int = 10) -> np.ndarray:
    """Deadline-hit rate of the tasks that arrived in each epoch."""
    r = pd.DataFrame(records)
    start, end = float(r["arrival"].min()), float(r["arrival"].max())
    edges = np.linspace(start, end, epochs + 1)
    index = np.clip(np.searchsorted(edges, r["arrival"].to_numpy(), side="right") - 1, 0, epochs - 1)
    hits = np.bincount(index, weights=r["deadline_hit"].to_numpy(float), minlength=epochs)
    counts = np.bincount(index, minlength=epochs)
    return np.where(counts > 0, hits / np.maximum(counts, 1), np.nan)


def fig2_utilisation(workers: int) -> None:
    """Load per epoch (mean and across-agent spread) and deadline keeping under it."""
    frame = policy_operational_data(workers)
    order = list(METHODS) + ["hand-set defaults"]
    fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.5))
    rows = []
    for method in order:
        part = frame[frame["method"] == method]
        pairs = [_epoch_utilisation(rec, int(n)) for rec, n in zip(part["records"], part["agents"])]
        means = np.array([p[0] for p in pairs])
        spreads = np.array([p[1] for p in pairs])
        dhr = np.array([_epoch_deadline_rate(rec) for rec in part["records"]])
        epochs = np.arange(1, means.shape[1] + 1)
        mean_util, mean_spread = means.mean(axis=0), spreads.mean(axis=0)
        axes[0].plot(epochs, mean_util, color=COLOR[method], label=LABEL[method],
                     marker=MARKER[method], markersize=5)
        if method in ("aspsa", "hand-set defaults"):
            axes[0].fill_between(epochs, mean_util - mean_spread, np.minimum(mean_util + mean_spread, 1.0),
                                 color=COLOR[method], alpha=0.12, linewidth=0)
        m, c = np.nanmean(dhr, axis=0), _ci(dhr)
        axes[1].plot(epochs, m, color=COLOR[method], label=LABEL[method],
                     marker=MARKER[method], markersize=5)
        axes[1].fill_between(epochs, m - c, m + c, color=COLOR[method], alpha=0.12, linewidth=0)
        rows += [{"method": method, "epoch": int(e), "mean_utilisation": float(u),
                  "utilisation_std_across_agents": float(sd), "deadline_hit_rate": float(d)}
                 for e, u, sd, d in zip(epochs, mean_util, mean_spread, m)]
    axes[0].set_title("mean agent utilisation per epoch\n(band: ± std across agents, A-SPSA and hand-set)")
    axes[0].set_ylabel("busy fraction")
    axes[0].set_ylim(0, 1.05)
    axes[1].set_title("deadline-hit rate of tasks arriving in the epoch\n(band: 95% CI over runs)")
    axes[1].set_ylabel("deadline-hit rate")
    for ax in axes:
        ax.set_xlabel("epoch (tenth of the arrival window)")
    axes[1].legend(frameon=False, loc="lower right")
    fig.suptitle("Load handling at ρ = 2.0 with the tuned priority weights (low-noise oracle)", y=1.04)
    _save(fig, "fig2_utilisation_by_epoch")
    pd.DataFrame(rows).to_csv(DATA / f"fig2_epochs{CACHE}.csv", index=False)


def fig3_cumulative_deadline(workers: int) -> None:
    frame = policy_operational_data(workers)
    order = list(METHODS) + ["hand-set defaults"]
    fig, ax = plt.subplots(figsize=(6.2, 3.6))
    grid = np.linspace(0.0, 1.0, 101)
    finals = {}
    for method in order:
        part = frame[frame["method"] == method]
        curves = []
        for (oseed, seed), group in part.groupby(["optimizer_seed", "seed"]):
            r = pd.concat([pd.DataFrame(rec) for rec in group["records"]], ignore_index=True)
            r = r.sort_values("finish")
            t = (r["finish"] - r["arrival"].min()) / (r["finish"].max() - r["arrival"].min())
            met = np.cumsum(r["deadline_hit"].to_numpy())
            curves.append(np.interp(grid, t.to_numpy(), met, left=0.0))
        curves = np.array(curves)
        finals[method] = curves[:, -1]
        mean = curves.mean(axis=0)
        ax.plot(grid, mean, color=COLOR[method], label=LABEL[method],
                marker=MARKER[method], markevery=20, markersize=5)
        ax.fill_between(grid, mean - _ci(curves), mean + _ci(curves), color=COLOR[method],
                        alpha=0.12, linewidth=0)
    ax.set_xlabel("elapsed share of the test run")
    ax.set_ylabel("tasks completed within deadline (cumulative)")
    ax.set_title("Cumulative on-time completions, 5 sites × 5 test streams, ρ = 2.0")
    ax.legend(frameon=False, loc="upper left")
    _save(fig, "fig3_cumulative_deadline_hits")
    pd.DataFrame([{"method": m, "mean_final_on_time": float(np.mean(v)), "runs": len(v)}
                  for m, v in finals.items()]).to_csv(DATA / f"fig3_final_counts{CACHE}.csv", index=False)


def fig4_load_sweep(workers: int) -> None:
    frame = load_sweep_data(workers)
    rules = load_sweep_rules_data(workers)
    order = list(METHODS) + ["hand-set defaults"]
    fig, axes = plt.subplots(1, 2, figsize=(9.0, 3.4))
    for ax, key, title in zip(axes, ("deadline_hit_rate", "goodput"),
                              ("deadline-hit rate", "goodput (correct and on time, per time unit)")):
        for method in order:
            part = (frame[frame["method"] == method]
                    .groupby(["rho", "optimizer_seed", "seed"])[key].mean().reset_index())
            by_rho = part.groupby("rho")[key]
            mean = by_rho.mean()
            ci = by_rho.apply(lambda v: _ci(v.to_numpy()))
            ax.plot(mean.index, mean.values, color=COLOR[method], label=LABEL[method],
                    marker=MARKER[method], markersize=5)
            ax.fill_between(mean.index, mean - ci, mean + ci, color=COLOR[method], alpha=0.12, linewidth=0)
        # The classic rules need no tuning: one deterministic value per load.
        for rule, (label, style) in RULE_STYLE.items():
            mean = rules[rules["method"] == rule].groupby(["rho", "seed"])[key].mean().groupby("rho").mean()
            ax.plot(mean.index, mean.values, color="#52524e", linestyle=style, linewidth=1.4, label=label)
        ax.set_title(title)
        ax.set_xlabel("offered load ρ")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=4, frameon=False, bbox_to_anchor=(0.5, -0.02))
    fig.suptitle("Robustness to load: weights tuned at ρ = 2.0, evaluated at other loads", y=1.02)
    _save(fig, "fig4_load_sweep")


def fig5_api_frontier(_workers: int) -> None:
    runs = _system_runs("api")
    fig, ax = plt.subplots(figsize=(6.4, 4.0))
    for system, style in (("aspsa", "-"), ("aspsa_mlp", "--"), ("knn_router", "-"), ("mlp_router", "-")):
        if system not in set(runs["system"]):
            continue
        part = runs[runs["system"] == system].groupby("price_weight")[["mean_cost", "mean_quality"]].mean()
        part = part.sort_values("mean_cost")
        label = "our system (A-SPSA)" if system == "aspsa" else LABEL[system]
        ax.plot(part["mean_cost"] * 1000, part["mean_quality"], color=COLOR[system], label=label,
                marker="o", markersize=6, linestyle=style, mfc="none" if style == "--" else COLOR[system])
    points = {
        "static_domain": runs[runs.system == "static_domain"][["mean_cost", "mean_quality"]].mean(),
        "quality_greedy": runs[runs.system == "quality_greedy"][["mean_cost", "mean_quality"]].mean(),
        "random": runs[runs.system == "random"][["mean_cost", "mean_quality"]].mean(),
        "cost_greedy": runs[runs.system == "cost_greedy"][["mean_cost", "mean_quality"]].mean(),
    }
    for name, p in points.items():
        ax.scatter(p["mean_cost"] * 1000, p["mean_quality"], color=COLOR[name], s=40, zorder=3,
                   edgecolor="#fcfcfb", linewidth=1.5)
        ax.annotate(LABEL[name], (p["mean_cost"] * 1000, p["mean_quality"]), textcoords="offset points",
                    xytext=(6, -3), fontsize=7.5, color="#52524e")
    refs = {
        "OpenRouter auto-router (commercial)": (runs["reference_router_cost_usd"].mean(),
                                                runs["reference_router_quality"].mean(), "P"),
        "GPT-5 alone (no deadlines)": (runs["best_single_cost_usd"].mean(), runs["best_single_quality"].mean(), "*"),
        "per-prompt oracle": (runs["oracle_cost_usd"].mean(), runs["oracle_quality"].mean(), "D"),
    }
    for name, (c, q, m) in refs.items():
        ax.scatter(c * 1000, q, marker=m, s=70, color="#2b2b29", zorder=4)
        ax.annotate(name, (c * 1000, q), textcoords="offset points", xytext=(7, 2), fontsize=7.5,
                    color="#2b2b29")
    ax.set_xlabel("mean cost per request, USD × 10⁻³")
    ax.set_ylabel("mean answer quality (measured)")
    ax.set_title("API regime: cost–quality frontier on LLMRouterBench (10 seeds × 1000 prompts)")
    ax.legend(frameon=False, loc="lower right")
    _save(fig, "fig5_api_cost_quality_frontier")


def fig6_architecture(_workers: int) -> None:
    order = ["aspsa", "aspsa_mlp", "knn_router", "mlp_router", "static_domain", "quality_greedy", "cost_greedy",
             "random", "oracle"]
    order = [s for s in order if s in set(_system_runs("capacity")["system"])]
    metrics = (("deadline_hit_rate", "deadline-hit rate (SLO attainment)"),
               ("goodput_per_s", "goodput, correct & on-time tasks / s"),
               ("p95_latency_s", "p95 end-to-end latency, s"))
    fig, axes = plt.subplots(2, 3, figsize=(10.5, 5.4))
    for row, regime in enumerate(("capacity", "stress")):
        runs = _system_runs(regime)
        runs = runs[np.isclose(runs["price_weight"], -0.08)]
        for col, (key, title) in enumerate(metrics):
            ax = axes[row, col]
            means = [runs[runs.system == s][key].mean() for s in order]
            cis = [_ci(runs[runs.system == s][key].to_numpy()) for s in order]
            colors = [COLOR[s] for s in order]
            ax.barh(range(len(order)), means, xerr=cis, color=colors, height=0.62,
                    error_kw=dict(ecolor="#52524e", lw=1, capsize=2))
            ax.set_yticks(range(len(order)))
            ax.set_yticklabels(["our system (A-SPSA)" if s == "aspsa" else LABEL[s] for s in order]
                               if col == 0 else [])
            ax.invert_yaxis()
            ax.set_title(title if row == 0 else "")
            ax.grid(axis="y", visible=False)
        axes[row, 0].set_ylabel({"capacity": "capacity regime\n4 slots/model, ρ = 0.85",
                                 "stress": "stress regime\n1 slot/model, ρ = 0.85"}[regime])
    fig.suptitle("Architecture under load (mean ± 95% CI, 10 paired seeds)", y=1.01)
    fig.tight_layout()
    _save(fig, "fig6_architecture_under_load")


def fig7_noise_and_seeds(_workers: int) -> None:
    frame = _policy_frame()
    fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.4))
    ax = axes[0]
    for i, method in enumerate(METHODS):
        for j, noise in enumerate(("noisy", "low_noise")):
            v = frame[(frame.method == method) & (frame.noise == noise)]["deadline_hit_rate"].to_numpy()
            x = i + (j - 0.5) * 0.3
            ax.errorbar(x, v.mean(), yerr=_ci(v), fmt=MARKER[method], color=COLOR[method],
                        markersize=7, mfc=COLOR[method] if noise == "low_noise" else "#fcfcfb",
                        capsize=3, lw=1.2)
        ax.annotate("", xy=(i + 0.15, frame[(frame.method == method) & (frame.noise == "low_noise")]
                            ["deadline_hit_rate"].mean()),
                    xytext=(i - 0.15, frame[(frame.method == method) & (frame.noise == "noisy")]
                            ["deadline_hit_rate"].mean()),
                    arrowprops=dict(arrowstyle="->", color="#8a8a85", lw=1))
    hand = frame[frame.method == "hand-set defaults"]["deadline_hit_rate"].mean()
    ax.axhline(hand, color="#8a8a85", ls="--", lw=1)
    ax.text(-0.45, hand, "hand-set weights", va="bottom", ha="left", fontsize=7.5, color="#52524e")
    if (RULES_DIR / "policy_real_raw.csv").exists():
        rules = pd.read_csv(RULES_DIR / "policy_real_raw.csv")
        edf = float(rules[rules["method"] == "edf"]["deadline_hit_rate"].mean())
        ax.axhline(edf, color="#52524e", linestyle=RULE_STYLE["edf"][1], lw=1)
        ax.text(1.5, edf, "EDF (no tuning)", va="bottom", ha="center", fontsize=7.5, color="#52524e")
    ax.set_xticks(range(len(METHODS)))
    ax.set_xticklabels([LABEL[m] for m in METHODS])
    ax.set_ylabel("held-out deadline-hit rate")
    ax.set_title("noisy (hollow) → low-noise (filled) oracle")
    ax = axes[1]
    low = frame[frame.noise == "low_noise"].pivot(index="optimizer_seed", columns="method",
                                                   values="deadline_hit_rate")
    best_other = low[["spsa", "zo_gt", "zo_pgd"]].max(axis=1)
    for seed in low.index:
        ax.plot([0, 1], [best_other[seed], low.loc[seed, "aspsa"]], color="#b5b4ae", lw=1)
    ax.scatter(np.zeros(len(low)), best_other, color="#8a8a85", s=30, zorder=3, label="best of SPSA/ZO-GT/ZO-PGD")
    ax.scatter(np.ones(len(low)), low["aspsa"], color=COLOR["aspsa"], s=30, zorder=3, label="A-SPSA")
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["best competitor\n(same seed)", "A-SPSA"])
    ax.set_xlim(-0.4, 1.4)
    ax.set_ylabel("held-out deadline-hit rate")
    ax.set_title(f"per optimizer seed, low-noise oracle ({int((low['aspsa'] > best_other).sum())}/{len(low)} seeds)")
    ax.legend(frameon=False, loc="upper left")
    _save(fig, "fig7_noise_gain_and_paired_seeds")


TUNING_FILES = tuple(RUNS / name / "step_tuning.csv" for name in POLICY_RUNS["tuning"])


def fig8_step_robustness(_workers: int) -> None:
    """Validation deadline-hit rate against the step size, every method, one grid."""
    frames = [pd.read_csv(p) for p in TUNING_FILES if p.exists()]
    if not frames:
        print("fig8: no tuning data yet")
        return
    tuning = pd.concat(frames, ignore_index=True)
    fig, ax = plt.subplots(figsize=(6.4, 3.8))
    for method in METHODS:
        part = tuning[tuning["method"] == method].groupby("step_size")["deadline_hit_rate"]
        mean = part.mean()
        spread = part.apply(lambda v: v.std(ddof=1) if len(v) > 1 else 0.0)
        ax.plot(mean.index, mean.values, color=COLOR[method], marker=MARKER[method], markersize=6,
                label=LABEL[method])
        ax.fill_between(mean.index, mean - spread, mean + spread, color=COLOR[method], alpha=0.12,
                        linewidth=0)
        best = mean.idxmax()
        ax.scatter([best], [mean[best]], s=110, facecolor="none", edgecolor=COLOR[method], linewidth=1.8,
                   zorder=4)
    ax.set_xscale("log")
    ax.set_xlabel("nominal step size h (log scale)")
    ax.set_ylabel("validation deadline-hit rate")
    ax.set_title("Validation deadline-hit rate vs step size, low-noise oracle (circle: best step)")
    ax.legend(frameon=False, loc="lower right")
    _save(fig, "fig8_step_robustness")


ABLATION = RUNS / POLICY_RUNS["ablation"] / "policy_real_raw.csv"
ABLATION_LABEL = {"aspsa": "A-SPSA (full)", "aspsa_convex": "A-SPSA, normalised probe",
                  "aspsa_no_consensus": "A-SPSA, no consensus", "aspsa_no_momentum": "A-SPSA, no acceleration",
                  "spsa": "SPSA", "zo_gt": "ZO-GT", "zo_pgd": "ZO-PGD", "hand-set defaults": "hand-set weights"}


def fig9_ablation(_workers: int) -> None:
    """Held-out deadline-hit rate of the A-SPSA variants against the baselines."""
    if not ABLATION.exists():
        print("fig9: no ablation data yet")
        return
    frame = pd.concat([_policy_frame(), pd.read_csv(ABLATION)], ignore_index=True)
    order = ["aspsa", "aspsa_convex", "aspsa_no_consensus", "aspsa_no_momentum", "spsa", "zo_gt", "zo_pgd"]
    colour = {"aspsa": COLOR["aspsa"], "aspsa_convex": "#5598e7", "aspsa_no_consensus": "#86b6ef",
              "aspsa_no_momentum": "#b7d3f6", "spsa": COLOR["spsa"], "zo_gt": COLOR["zo_gt"],
              "zo_pgd": COLOR["zo_pgd"]}
    fig, axes = plt.subplots(1, 2, figsize=(10.0, 3.6), sharex=True)
    means_all = frame[frame["method"].isin(order)].groupby(["noise", "method"])["deadline_hit_rate"].mean()
    lo, hi = float(means_all.min()), float(means_all.max())
    for ax, noise, title in zip(axes, ("low_noise", "noisy"), ("low-noise oracle", "noisy oracle")):
        part = frame[frame["noise"] == noise]
        means = [part[part["method"] == m]["deadline_hit_rate"].mean() for m in order]
        cis = [_ci(part[part["method"] == m]["deadline_hit_rate"].to_numpy()) for m in order]
        # Dots with intervals, not bars: the axis does not start at zero, and
        # a truncated bar would exaggerate the differences.
        for y, (m, mean, ci) in enumerate(zip(order, means, cis)):
            ax.errorbar(mean, y, xerr=ci, fmt="o", color=colour[m], markersize=8, capsize=3,
                        elinewidth=1.4, markeredgecolor="#fcfcfb", markeredgewidth=1.2)
            ax.annotate(f"{mean:.3f}", (mean + ci, y), textcoords="offset points", xytext=(6, -3),
                        fontsize=7.5, color="#52524e")
        hand = frame[frame["method"] == "hand-set defaults"]["deadline_hit_rate"].mean()
        ax.axvline(hand, color="#8a8a85", ls="--", lw=1)
        if (RULES_DIR / "policy_real_raw.csv").exists():
            rules = pd.read_csv(RULES_DIR / "policy_real_raw.csv")
            edf = float(rules[rules["method"] == "edf"]["deadline_hit_rate"].mean())
            ax.axvline(edf, color="#52524e", linestyle=RULE_STYLE["edf"][1], lw=1)
        ax.set_yticks(range(len(order)))
        ax.set_yticklabels([ABLATION_LABEL[m] for m in order] if ax is axes[0] else [])
        ax.set_title(title)
        ax.set_xlabel("held-out deadline-hit rate")
        ax.set_xlim(min(lo, hand) - 0.015, hi + 0.02)
        ax.set_ylim(len(order) - 0.5, -0.5)
    fig.suptitle("Which part of A-SPSA matters (9 optimizer seeds, mean ± 95% CI; grey dashed: hand-set, "
                 "dark dashed: EDF)", y=1.02)
    _save(fig, "fig9_aspsa_ablation")


THREE = Path("real_benchmark_outputs/policy_real_three/three_blocks_raw.csv")


def fig10_three_functionals(_workers: int) -> None:
    """The manuscript's three functionals on the scheduler: dynamics and held-out values."""
    if not THREE.exists():
        print("fig10: no three-block data yet")
        return
    frame = pd.read_csv(THREE)
    blocks = (("processing", "mean service time (block p)"), ("waiting", "p95 waiting time (block w)"),
              ("quality", "1 − mean quality (block q)"))
    fig, axes = plt.subplots(1, 3, figsize=(12.0, 3.4))
    for ax, (block, title) in zip(axes, blocks):
        for method in METHODS:
            part = frame[frame["method"] == method]
            traces = np.array([json.loads(t)[block] for t in part["traces"]], dtype=float)
            kernel = np.ones(3) / 3.0
            smooth = np.array([np.convolve(np.pad(t, 1, mode="edge"), kernel, "valid") for t in traces])
            rounds = np.arange(1, smooth.shape[1] + 1)
            mean, ci = smooth.mean(axis=0), _ci(smooth)
            ax.plot(rounds, mean, color=COLOR[method], label=LABEL[method], marker=MARKER[method],
                    markevery=5, markersize=5)
            ax.fill_between(rounds, mean - ci, mean + ci, color=COLOR[method], alpha=0.12, linewidth=0)
        ax.set_title(title)
        ax.set_xlabel("synchronisation round")
    axes[0].set_ylabel("block objective on train streams\n(lower is better)")
    axes[2].legend(frameon=False, loc="upper right")
    fig.suptitle("Scheduler policy with the manuscript's three functionals (mean ± 95% CI, 9 seeds)", y=1.03)
    _save(fig, "fig10_three_functionals")


def fig11_learned_weights(_workers: int) -> None:
    """Priority weights each method ends with (low-noise oracle, mean over sites and seeds)."""
    frame = _policy_frame()
    names = ["λ_H hotness", "λ_W wait pressure", "λ_D deadline gap", "λ_R budget overrun", "α_H hotness slope",
             "λ_E deadline urgency"][:len(family(_policy_cfg())[0])]
    order = list(METHODS) + ["hand-set defaults"]
    rows = []
    for method in order:
        part = frame[(frame["method"] == method) & (frame["noise"].isin(["low_noise", "reference"]))]
        weights = np.array([json.loads(w) for w in part["weights"]], dtype=float)   # runs x sites x weights
        per_run = weights.mean(axis=1)
        rows.append((method, per_run.mean(axis=0),
                     _ci(per_run) if len(per_run) > 1 else np.zeros(len(names))))
    fig, axes = plt.subplots(1, len(names), figsize=(2.5 * len(names), 3.0))
    for k, ax in enumerate(axes):
        means = [r[1][k] for r in rows]
        cis = [r[2][k] for r in rows]
        ax.barh(range(len(order)), means, xerr=cis, color=[COLOR[m] for m in order], height=0.62,
                error_kw=dict(ecolor="#52524e", lw=1, capsize=2))
        ax.set_yticks(range(len(order)))
        ax.set_yticklabels([LABEL[m] for m in order] if k == 0 else [])
        ax.invert_yaxis()
        ax.set_title(names[k])
        ax.grid(axis="y", visible=False)
    fig.suptitle("Buffer priority weights found by each method (low-noise oracle; mean over 5 sites, "
                 "95% CI over 9 seeds)", y=1.05)
    _save(fig, "fig11_learned_weights")
    pd.DataFrame([{"method": m, **{n: float(v[i]) for i, n in enumerate(names)}} for m, v, _ in rows]).to_csv(
        DATA / f"fig11_weights{CACHE}.csv", index=False)


# Steps within an order of the worst-case bound h_p <= 1/M_p = 0.0025; at
# h = 0.05 the accelerated recursion loses consensus between controllers.
FIG12_STEPS = (0.001, 0.01)


def fig12_equal_step(_workers: int) -> None:
    """Out-of-sample F1 of each method at one common step h, all three regimes."""
    if not (EQUAL_STEP / "runs_api.csv").exists():
        print("fig12: no equal-step runs yet")
        return
    methods = ("aspsa", "spsa", "zo_gt")
    fig, axes = plt.subplots(1, 3, figsize=(12.0, 3.5))
    for ax, (regime, title) in zip(axes, REGIME_TITLE.items()):
        runs = pd.read_csv(EQUAL_STEP / f"runs_{regime}.csv")
        steps = list(FIG12_STEPS)
        for k, method in enumerate(methods):
            for i, step in enumerate(steps):
                values = runs[(runs["system"] == method) & np.isclose(runs["common_step"], step)]["f1"].to_numpy()
                ax.errorbar(i + (k - 1) * 0.22, values.mean(), yerr=_ci(values), fmt=MARKER[method],
                            color=COLOR[method], markersize=7, capsize=3, elinewidth=1.4,
                            markeredgecolor="#fcfcfb", markeredgewidth=1.0,
                            label=LABEL[method] if i == 0 else None)
        # RLS alone has no step: one reference level per regime.
        baseline = pd.read_csv(SYSTEM_PREDICTION / f"runs_{regime}.csv")
        rls = baseline[(baseline["system"] == "local_rls") & np.isclose(baseline["price_weight"], -0.08)]
        ax.axhline(rls["f1"].mean(), color="#8a8a85", ls="--", lw=1.2, label="RLS only, no ZO layer")
        ax.set_xticks(range(len(steps)))
        ax.set_xticklabels([f"h = {s:g}" for s in steps])
        ax.set_xlim(-0.6, len(steps) - 0.4)
        ax.grid(axis="x", visible=False)
        ax.set_title(title)
        ax.set_xlabel("common step of every method and block")
    axes[0].set_ylabel("out-of-sample F1\n(processing-time loss, lower is better)")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=len(labels), frameon=False,
               bbox_to_anchor=(0.5, -0.02))
    fig.suptitle("Prediction layer at equal step: F1 on the test streams (mean ± 95% CI, 10 paired seeds)",
                 y=1.03)
    _save(fig, "fig12_equal_step_f1")


RUN_CALLS = 300      # 30 rounds x 2 oracle calls x 5 sites
TUNING_SEEDS = 2     # optimizer seeds per grid point of the step search
# (oracle condition, run directory, grid points searched before the test run)
BUDGET_SOURCES = tuple((noise, RUNS / name, grid) for noise, name, grid in POLICY_RUNS["budget"])
RULES_DIR = RUNS / POLICY_RUNS["rules"]


def tuning_budget_data() -> pd.DataFrame:
    """Held-out deadline-hit rate against the oracle calls spent, step search included.

    A run costs 300 calls; selecting the step over a grid of g points on two
    optimizer seeds adds 2 * g runs.  The validation evaluations of the grid
    are not counted, which favours the methods that need the search.
    """
    rows = []
    for noise, directory, grid in BUDGET_SOURCES:
        raw = pd.read_csv(directory / "policy_real_raw.csv")
        for method, part in raw[(raw["noise"] == noise) & raw["method"].isin(METHODS)].groupby("method"):
            values = part["deadline_hit_rate"].to_numpy()
            rows.append({"noise": noise, "method": method, "grid_points": grid,
                         "oracle_calls": RUN_CALLS * (1 + TUNING_SEEDS * grid),
                         "step": float(part["step_size"].iloc[0]), "dhr": float(values.mean()),
                         "ci": float(_ci(values)), "min": float(values.min()), "max": float(values.max())})
    ablation = pd.read_csv(ABLATION)
    for noise, part in ablation[ablation["method"] == "aspsa_no_momentum"].groupby("noise"):
        values = part["deadline_hit_rate"].to_numpy()
        rows.append({"noise": noise, "method": "spsa_constant", "grid_points": 0, "oracle_calls": RUN_CALLS,
                     "step": 0.2, "dhr": float(values.mean()), "ci": float(_ci(values)),
                     "min": float(values.min()), "max": float(values.max())})
    frame = pd.DataFrame(rows)
    DATA.mkdir(parents=True, exist_ok=True)
    frame.to_csv(DATA / "fig13_tuning_budget.csv", index=False)
    return frame


def fig13_tuning_budget(_workers: int) -> None:
    """What each method reaches for a budget of oracle calls, step search included."""
    frame = tuning_budget_data()
    rules = pd.read_csv(RULES_DIR / "policy_real_raw.csv") if (RULES_DIR / "policy_real_raw.csv").exists() else None
    policy = _policy_frame()
    hand = float(policy[policy["method"] == "hand-set defaults"]["deadline_hit_rate"].mean())
    budgets = sorted(frame["oracle_calls"].unique())
    fig, axes = plt.subplots(1, 2, figsize=(10.0, 3.8), sharey=True)
    for ax, noise, title in zip(axes, ("low_noise", "noisy"), ("low-noise oracle", "noisy oracle")):
        part = frame[frame["noise"] == noise]
        for method in METHODS:
            points = part[part["method"] == method].sort_values("oracle_calls")
            ax.errorbar(points["oracle_calls"], points["dhr"], yerr=points["ci"], color=COLOR[method],
                        marker=MARKER[method], markersize=6, capsize=3, linewidth=1.6, label=LABEL[method])
        constant = part[part["method"] == "spsa_constant"]
        ax.errorbar(constant["oracle_calls"], constant["dhr"], yerr=constant["ci"], fmt="s", mfc="none",
                    color=COLOR["spsa"], markersize=8, capsize=3, label="SPSA, constant step h = 0.2")
        ax.axhline(hand, color="#8a8a85", linestyle="--", linewidth=1, label="hand-set weights")
        if rules is not None:
            # SJF sits far below every tuned policy; the reference that matters is EDF.
            for rule in ("edf", "llf"):
                label, style = RULE_STYLE[rule]
                value = rules[rules["method"] == rule]["deadline_hit_rate"].mean()
                ax.axhline(value, color="#52524e", linestyle=style, linewidth=1.2, label=label)
        ax.set_ylim(bottom=hand - 0.008)
        ax.set_xscale("log")
        ax.minorticks_off()
        ax.set_xticks(budgets)
        ax.set_xticklabels([f"{b:,}".replace(",", " ") for b in budgets])
        ax.set_xlim(budgets[0] / 1.4, budgets[-1] * 1.4)
        ax.set_xlabel("oracle calls spent, step search included")
        ax.set_title(title)
    axes[0].set_ylabel("held-out deadline-hit rate")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=5, frameon=False, bbox_to_anchor=(0.5, -0.02))
    fig.suptitle("Deadline-hit rate reached for a budget of oracle calls (9 optimizer seeds, mean ± 95% CI)",
                 y=1.02)
    _save(fig, "fig13_tuning_budget")


FIGURES = {
    "fig11": fig11_learned_weights,
    "fig1": fig1_policy_dynamics, "fig1b": fig1b_prediction_functionals,
    "fig2": fig2_utilisation, "fig3": fig3_cumulative_deadline, "fig4": fig4_load_sweep,
    "fig5": fig5_api_frontier, "fig6": fig6_architecture, "fig7": fig7_noise_and_seeds,
    "fig8": fig8_step_robustness, "fig9": fig9_ablation, "fig10": fig10_three_functionals,
    "fig12": fig12_equal_step, "fig13": fig13_tuning_budget,
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("figures", nargs="*", default=list(FIGURES))
    parser.add_argument("--workers", type=int, default=1)
    args = parser.parse_args()
    for name in args.figures:
        FIGURES[name](args.workers)


if __name__ == "__main__":
    main()
