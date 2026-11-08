"""Replay of measured LLM outcomes through the controlled experiment.

``TableOracle`` has the interface of ``paper_experiment.ReplayOracle`` but
returns what a real model produced on a real prompt (``src.real_benchmarks``)
instead of drawing from the hand-set linear generator.  Everything a controller
may use before execution -- the agent profiles, task hardness and the SLO
deadline -- is derived from the train split or from the prompt alone, so no
outcome of a test prompt leaks into a decision about it.

Latency is the one modelled quantity.  Neither benchmark records wall-clock
time, so service time is ``TTFT + input_tokens / prefill_rate + output_tokens / decode_rate`` with the
measured output tokens and a per-model profile, times a lognormal provider
jitter that is fixed per (seed, prompt, model).  The profiles are assumptions
until they are replaced by measurements (``LATENCY_PROFILES_SOURCE``); every
run manifest records which profile was used.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from .paper_experiment import (
    AgentSpec, PotentialOutcome, ReplayTask, _stable_rng, length_score,
)
from . import real_benchmarks


# (TTFT seconds, decode tokens per second).  Order-of-magnitude values for
# hosted inference of each model class; they fix relative speed, which is what
# routing reacts to.  Replace with measured values via --latency-profiles.
LATENCY_PROFILES_SOURCE = "assumed order-of-magnitude hosted-inference profiles; not measured"
DEFAULT_LATENCY_PROFILES: dict[str, tuple[float, float]] = {
    # SPROUT pool
    "aws-claude-3-5-sonnet-v1": (0.9, 60.0),
    "aws-titan-text-premier-v1": (0.6, 60.0),
    "openai-gpt-4o": (0.5, 80.0),
    "openai-gpt-4o-mini": (0.4, 90.0),
    "wxai-granite-3-2b-instruct-8k-max-tokens": (0.25, 180.0),
    "wxai-granite-3-8b-instruct-8k-max-tokens": (0.3, 120.0),
    "wxai-llama-3-1-70b-instruct": (0.5, 45.0),
    "wxai-llama-3-1-8b-instruct": (0.3, 140.0),
    "wxai-llama-3-2-1b-instruct": (0.2, 220.0),
    "wxai-llama-3-2-3b-instruct": (0.25, 180.0),
    "wxai-llama-3-3-70b-instruct": (0.5, 45.0),
    "wxai-llama-3-405b-instruct": (0.9, 25.0),
    "wxai-mixtral-8x7b-instruct-v01": (0.4, 80.0),
    # RouterBench pool
    "WizardLM/WizardLM-13B-V1.2": (0.3, 90.0),
    "claude-instant-v1": (0.5, 90.0),
    "claude-v1": (0.9, 40.0),
    "claude-v2": (1.0, 40.0),
    "gpt-3.5-turbo-1106": (0.4, 90.0),
    "gpt-4-1106-preview": (0.8, 30.0),
    "meta/code-llama-instruct-34b-chat": (0.4, 60.0),
    "meta/llama-2-70b-chat": (0.5, 40.0),
    "mistralai/mistral-7b-chat": (0.25, 140.0),
    "mistralai/mixtral-8x7b-chat": (0.4, 80.0),
    "zero-one-ai/Yi-34B-Chat": (0.4, 60.0),
    # LLMRouterBench pool.  Output tokens include reasoning tokens, so the
    # long service times of reasoning models come from the measured tokens,
    # not from these rates.
    "claude-sonnet-4": (1.2, 60.0),
    "deepseek-r1-0528": (1.5, 35.0),
    "deepseek-v3-0324": (1.0, 35.0),
    "deepseek-v3.1-terminus": (1.0, 40.0),
    "gemini-2.5-flash": (0.5, 200.0),
    "gemini-2.5-pro": (1.5, 110.0),
    "glm-4.6": (0.8, 50.0),
    "gpt-5": (1.5, 70.0),
    "intern-s1": (0.8, 50.0),
    "kimi-k2-0905": (0.8, 45.0),
    "qwen3-235b-a22b-2507": (0.7, 60.0),
}
# Prompt processing is two orders of magnitude faster than decoding, but the
# agentic and repository-level prompts run to tens of thousands of tokens, so
# prefill is not negligible there.  One pool-wide rate.
PREFILL_TOKENS_PER_SECOND = 4000.0
LATENCY_JITTER_SIGMA = 0.25

AZURE_TRACES = {
    "azure_code": Path("data/external/azure/AzureLLMInferenceTrace_code.csv"),
    "azure_conv": Path("data/external/azure/AzureLLMInferenceTrace_conv.csv"),
}


@dataclass(frozen=True)
class TableConfig:
    benchmark: str = "sprout"
    # Seconds per simulator time unit.  The controller priors, clipping boxes
    # and loss scales were designed for service times of order one, so real
    # seconds are expressed in units of this many seconds.  Reports convert
    # back to seconds.
    time_unit_seconds: float = 5.0
    arrival_process: str = "azure_code"   # poisson | azure_code | azure_conv
    # Deadline = arrival + slo_scale * the prompt's isolated latency, taken as
    # the median over the pool.  This is the SLO-scale convention of the LLM
    # serving literature; it is exogenous to every policy.
    slo_scale: float = 3.0
    latency_profiles: tuple[tuple[str, float, float], ...] | None = None


class OutcomeTable:
    """Measured outcomes for one benchmark, indexed for O(1) replay."""

    def __init__(self, benchmark: str, cfg: TableConfig, seed: int):
        items, outcomes = real_benchmarks.load(benchmark)
        if items.empty or outcomes.empty:
            raise ValueError(f"{benchmark} has no complete prompt/model outcomes to replay")
        self.benchmark = benchmark
        self.cfg = cfg
        self.domains = real_benchmarks.DOMAINS[benchmark]
        self.items = items.set_index("item_id")
        self.models = sorted(outcomes["model"].unique())
        profiles = dict(DEFAULT_LATENCY_PROFILES)
        if cfg.latency_profiles:
            profiles.update({name: (ttft, rate) for name, ttft, rate in cfg.latency_profiles})
        missing = [m for m in self.models if m not in profiles]
        if missing:
            raise ValueError(f"no latency profile for {missing}")
        self.profiles = {m: profiles[m] for m in self.models}
        wide = outcomes.pivot(index="item_id", columns="model")
        self.item_index = {item: i for i, item in enumerate(wide.index)}
        self.quality = wide["quality"][self.models].to_numpy(float)
        self.success = wide["success"][self.models].to_numpy(bool)
        self.cost = wide["cost_usd"][self.models].to_numpy(float)
        self.output_tokens = wide["output_tokens"][self.models].to_numpy(float)
        self.input_tokens = wide["input_tokens"][self.models].to_numpy(float)
        self.service = self._service_seconds(wide.index, seed) / cfg.time_unit_seconds
        self.hardness = self._group_difficulty()

    def _service_seconds(self, item_ids, seed: int) -> np.ndarray:
        ttft = np.array([self.profiles[m][0] for m in self.models])
        rate = np.array([self.profiles[m][1] for m in self.models])
        base = (ttft[None, :] + self.input_tokens / PREFILL_TOKENS_PER_SECOND
                + self.output_tokens / rate[None, :])
        jitter = np.empty_like(base)
        for row, item in enumerate(item_ids):
            for col, model in enumerate(self.models):
                rng = _stable_rng(seed, item, model, "latency")
                jitter[row, col] = rng.lognormal(-0.5 * LATENCY_JITTER_SIGMA**2, LATENCY_JITTER_SIGMA)
        return base * jitter

    def _group_difficulty(self) -> dict[str, float]:
        """Hardness of a prompt family, estimated from the train split only.

        One minus the pool-mean quality of the (source dataset, level) group,
        min-max scaled to [0, 1].  It is prior knowledge about the kind of task
        -- MATH level 5 is harder than level 1 -- not about the prompt itself,
        so a test prompt's own outcomes never enter its covariate.
        """
        items = self.items.reset_index()
        items["group"] = items["source_dataset"] + "|" + items["level"].fillna(-1).astype(str)
        rows = np.array([self.item_index[i] for i in items["item_id"]])
        items["pool_quality"] = self.quality[rows].mean(axis=1)
        train = items[items["split"] == "train"]
        by_group = 1.0 - train.groupby("group")["pool_quality"].mean()
        fallback = 1.0 - train.groupby("domain_name")["pool_quality"].mean()
        raw = items.apply(
            lambda r: by_group.get(r["group"], fallback.get(r["domain_name"], by_group.mean())), axis=1)
        low, high = float(by_group.min()), float(by_group.max())
        scaled = ((raw - low) / max(high - low, 1e-9)).clip(0.0, 1.0)
        return dict(zip(items["item_id"], scaled.astype(float)))

    def row(self, item_id: str) -> int:
        return self.item_index[item_id]

    def items_for(self, split: str) -> pd.DataFrame:
        frame = self.items.reset_index()
        return frame if split == "all" else frame[frame["split"] == split]

    def train_profiles(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Per-(model, domain) mean quality and service, and relative price.

        These are what an operator knows about the pool before deployment:
        offline evaluation on the train split, as every published router uses.
        """
        train = self.items_for("train")
        rows = np.array([self.item_index[i] for i in train["item_id"]])
        domains = train["domain"].to_numpy(int)
        capability = np.zeros((len(self.models), len(self.domains)))
        service = np.zeros_like(capability)
        for d in range(len(self.domains)):
            mask = rows[domains == d]
            capability[:, d] = self.quality[mask].mean(axis=0)
            service[:, d] = self.service[mask].mean(axis=0)
        price = self.cost[rows].mean(axis=0)
        return capability, service, price / price.max()


def make_table_agents(table: OutcomeTable, slots_per_model: int = 1) -> list[AgentSpec]:
    """One agent per serving slot; slot ``k`` runs model ``k mod M``.

    A slot is one request a model can serve at a time.  ``slots_per_model`` is
    instances times per-instance concurrency, the capacity model of
    capacity-constrained routing (Markovic-Voronov et al., CAIS 2026).  One slot
    is a small self-hosted pool; many slots at low load approximate a hosted
    API, whose concurrency the router does not bound.
    """
    capability, service, price = table.train_profiles()
    count = len(table.models) * int(slots_per_model)
    agents = []
    for agent_id in range(count):
        m = agent_id % len(table.models)
        replica = agent_id // len(table.models)
        name = table.models[m] + (f"#{replica}" if replica else "")
        agents.append(AgentSpec(agent_id, name, capability[m].copy(), service[m].copy(),
                                float(price[m]), model_id=m))
    return agents


class TableOracle:
    """Drop-in for ``ReplayOracle`` backed by measured outcomes."""

    PRIORS = ("knn", "mlp")

    def __init__(self, table: OutcomeTable, agents: list[AgentSpec], prompt_features=None,
                 prior: str = "knn"):
        self.table = table
        self.agents = agents
        self.model_of = [a.agent_id % len(table.models) for a in agents]
        self.prompt_features = prompt_features
        if prompt_features is not None and list(prompt_features.models) != list(table.models):
            raise ValueError("prompt features and outcome table list different models")
        if prior not in self.PRIORS:
            raise ValueError(f"prior must be one of {self.PRIORS}")
        # Which offline router's estimate the controllers receive as their
        # task--agent quality coordinate.  The kNN and MLP router baselines
        # always read their own estimate, whatever this is.
        self.prior = prior
        self._reference = real_benchmarks.load_reference(table.benchmark)

    def prior_quality(self, task: ReplayTask, agent_id: int) -> float:
        """The controllers' offline estimate of the agent's quality on this prompt."""
        if self.prior == "mlp":
            return self.router_quality(task, agent_id)
        return self.knn_quality(task, agent_id)

    def knn_quality(self, task: ReplayTask, agent_id: int) -> float:
        """kNN estimate of the agent's quality on this prompt from train prompts."""
        if self.prompt_features is None:
            raise ValueError("the knn_router baseline needs prompt features")
        return float(self.prompt_features.knn(task.source_id)[self.model_of[agent_id]])

    def router_quality(self, task: ReplayTask, agent_id: int) -> float:
        """The trained MLP router's estimate of the agent's quality on this prompt."""
        if self.prompt_features is None:
            raise ValueError("the mlp_router baseline needs prompt features")
        return float(self.prompt_features.mlp(task.source_id)[self.model_of[agent_id]])

    def reference_summary(self, tasks: list[ReplayTask]) -> dict[str, float]:
        """An external router's measured quality and cost on this stream.

        LLMRouterBench ran OpenRouter's commercial auto-router on the same
        prompts.  It is an API router with no queue, so it is comparable with
        the unlimited-capacity single-model references, not with latency.
        """
        if self._reference is None:
            return {}
        ref = self._reference.set_index("item_id")
        ids = [t.source_id for t in tasks if t.source_id in ref.index]
        if not ids:
            return {}
        rows = ref.loc[ids]
        return {
            "reference_router_quality": float(rows["quality"].mean()),
            "reference_router_cost_usd": float(rows["cost_usd"].mean()),
            "reference_router_coverage": float(len(ids) / len(tasks)),
        }

    def outcome(self, task: ReplayTask, agent_id: int) -> PotentialOutcome:
        r, m = self.table.row(task.source_id), self.model_of[agent_id]
        return PotentialOutcome(
            float(self.table.service[r, m]), float(self.table.quality[r, m]),
            bool(self.table.success[r, m]), float(self.table.cost[r, m]),
        )

    def potential(self, tasks: list[ReplayTask]) -> dict[str, np.ndarray]:
        """Full counterfactual rows for a task list, one column per agent."""
        rows = np.array([self.table.row(t.source_id) for t in tasks])
        cols = np.array(self.model_of)
        return {
            "quality": self.table.quality[np.ix_(rows, cols)],
            "cost": self.table.cost[np.ix_(rows, cols)],
            "service": self.table.service[np.ix_(rows, cols)],
            "success": self.table.success[np.ix_(rows, cols)],
        }


def _arrival_gaps(process: str, count: int, rng: np.random.Generator) -> np.ndarray:
    """Unit-mean inter-arrival gaps.

    For an Azure trace a random contiguous window is taken and its gaps are
    divided by their mean, so burstiness and simultaneous arrivals survive
    while the offered load is set by ``rho`` alone.
    """
    if process == "poisson":
        return rng.exponential(1.0, count)
    path = AZURE_TRACES.get(process)
    if path is None:
        raise ValueError(f"unknown arrival process: {process}")
    stamps = pd.to_datetime(pd.read_csv(path, usecols=["TIMESTAMP"])["TIMESTAMP"])
    gaps = np.diff(stamps.to_numpy().astype("datetime64[ns]").astype(np.int64)) / 1e9
    gaps = np.clip(gaps, 0.0, None)
    if count > len(gaps):
        raise ValueError(f"{process} has {len(gaps)} gaps, fewer than {count} tasks")
    start = int(rng.integers(0, len(gaps) - count + 1))
    window = gaps[start:start + count]
    return window / max(float(window.mean()), 1e-12)


def load_table_tasks(
    table: OutcomeTable,
    agents: list[AgentSpec],
    seed: int,
    rho: float,
    limit: int | None = None,
    split: str = "test",
    per_domain: int | None = None,
    prompt_features=None,
    domains: set[int] | None = None,
    load_reference: str = "pool",
) -> list[ReplayTask]:
    """Task stream over measured prompts.

    ``per_domain`` draws a balanced sample, as the IT benchmark does; without
    it the natural domain mix of the benchmark is kept, which is the realistic
    stream.  Offered load ``rho`` is relative to the capacity of this pool, so
    ``rho_effective == rho`` here.
    """
    frame = table.items_for(split)
    if domains is not None:
        # A site's own traffic: offered load then applies to that stream.
        frame = frame[frame["domain"].isin(sorted(domains))]
    rng = np.random.default_rng(seed)
    if per_domain is not None:
        frame = (frame.groupby("domain", group_keys=False)
                 .apply(lambda g: g.sample(min(per_domain, len(g)), random_state=seed)))
    frame = frame.sample(frac=1.0, random_state=seed).reset_index(drop=True)
    if limit is not None:
        frame = frame.head(limit)
    # Capacity is weighted by the stream's own domain mix.  An unweighted mean
    # over domains understates service time whenever long-output families
    # dominate the stream -- instruction prompts are 44% of SPROUT -- and the
    # nominal rho then overloads the pool.
    mix = np.bincount(frame["domain"].to_numpy(int), minlength=len(table.domains)).astype(float)
    mix /= max(mix.sum(), 1.0)
    if load_reference == "pool":
        capacity = float(sum(1.0 / float(a.mean_service @ mix) for a in agents))
    elif load_reference == "slowest_model":
        # Hosted-API emulation: rho is the utilisation the slowest model would
        # reach if it received all traffic, so no routing decision is ever
        # constrained by capacity and the comparison is about routing alone.
        slots = len(agents) / len(table.models)
        capacity = float(slots / max(float(a.mean_service @ mix) for a in agents))
    else:
        raise ValueError(f"unknown load reference: {load_reference}")
    gaps = _arrival_gaps(table.cfg.arrival_process, len(frame), rng) / max(rho * capacity, 1e-9)
    arrivals = np.cumsum(gaps)
    model_cols = sorted({a.agent_id % len(table.models) for a in agents})
    tasks = []
    for index, row in frame.iterrows():
        r = table.row(row["item_id"])
        isolated = float(np.median(table.service[r, model_cols]))
        skill = [0.0] * len(table.domains)
        skill[int(row["domain"])] = 1.0
        hardness = table.hardness[row["item_id"]]
        tasks.append(ReplayTask(
            int(index), str(row["item_id"]), int(row["domain"]), tuple(skill),
            float(hardness), float(hardness), int(row["prompt_chars"]), bool(row["has_code"]),
            float(arrivals[index]), float(arrivals[index] + table.cfg.slo_scale * isolated),
            length_score(int(row["prompt_chars"])), 0, str(row["domain_name"]),
            prompt_features.embedding(row["item_id"]) if prompt_features is not None else (),
        ))
    return tasks
