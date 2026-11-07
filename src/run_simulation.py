import argparse
from pathlib import Path

import numpy as np
import pandas as pd

try:
    from .main import ExactOrchestrator, SimulationVariant
    from .llm_api import LLMAPIClient, import_config_module
except ImportError:
    from main import ExactOrchestrator, SimulationVariant
    from llm_api import LLMAPIClient, import_config_module


def load_api_config():
    cfg = import_config_module("llm_config", __package__)
    endpoints = getattr(cfg, "LLM_ENDPOINTS", {})
    api_keys = getattr(cfg, "LLM_API_KEYS", {})
    aliases = getattr(cfg, "LLM_MODEL_ALIASES", {})
    judge_model = getattr(cfg, "JUDGE_MODEL_NAME", "qwen2.5-72b-instruct")
    return endpoints, api_keys, aliases, judge_model


def build_orchestrator(mode: str, num_ctrl: int, num_agents: int, output_dir: Path, seed: int = 42):
    plot_path = output_dir / f"{mode}_simulation.png"

    if mode == "test":
        return ExactOrchestrator(
            num_ctrl=num_ctrl,
            num_agents=num_agents,
            llm_api_client=None,
            variant=SimulationVariant(name="test", use_judge=False),
            plot_output_path=str(plot_path),
            classifier_seed=seed,
        )

    if mode == "api":
        endpoints, api_keys, aliases, judge_model_name = load_api_config()
        llm_client = LLMAPIClient(endpoints, api_keys, model_aliases=aliases)
        return ExactOrchestrator(
            num_ctrl=num_ctrl,
            num_agents=num_agents,
            llm_api_client=llm_client,
            judge_model_name=judge_model_name,
            variant=SimulationVariant(name="api", use_judge=True),
            plot_output_path=str(plot_path),
            classifier_seed=seed,
        )

    raise ValueError(f"Unsupported mode: {mode}")


def run_simulation(
    mode: str,
    num_tasks: int,
    seed: int,
    num_ctrl: int,
    num_agents: int,
    output_dir: str,
):
    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)

    np.random.seed(seed)
    orchestrator = build_orchestrator(
        mode=mode,
        num_ctrl=num_ctrl,
        num_agents=num_agents,
        output_dir=root,
        seed=seed,
    )
    tasks = orchestrator.generate_realistic_stream(num_tasks)
    orchestrator.run(tasks)

    summary = orchestrator.collect_summary()
    pd.DataFrame(orchestrator.task_records).to_csv(root / f"{mode}_task_records.csv", index=False)
    pd.DataFrame([summary]).to_csv(root / f"{mode}_summary.csv", index=False)
    return summary


def parse_args():
    parser = argparse.ArgumentParser(description="Run simulation in test or api mode.")
    parser.add_argument(
        "--mode",
        choices=["test", "api"],
        default="test",
        help="test = численная имитация без API, api = реальные LLM endpoint.",
    )
    parser.add_argument("--tasks", type=int, default=500, help="Number of tasks to simulate.")
    parser.add_argument("--seed", type=int, default=42, help="Global random seed.")
    parser.add_argument("--controllers", type=int, default=3, help="Number of controllers.")
    parser.add_argument("--agents", type=int, default=5, help="Number of agents/models.")
    parser.add_argument(
        "--output-dir",
        type=str,
        default="single_run_outputs",
        help="Directory for PNG and CSV artifacts.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    summary = run_simulation(
        mode=args.mode,
        num_tasks=args.tasks,
        seed=args.seed,
        num_ctrl=args.controllers,
        num_agents=args.agents,
        output_dir=args.output_dir,
    )
    print("\n=== SINGLE RUN SUMMARY ===")
    for key, value in summary.items():
        print(f"{key}: {value}")
