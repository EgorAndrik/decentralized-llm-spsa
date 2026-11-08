"""CLI runner for the global-buffer architecture."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from .buffered_orchestrator import BufferedOrchestrator
from .llm_api import BudgetExceeded, LLMAPIClient, api_key_from_env, import_config_module
from .main import PROFILES, JudgeEvaluationError, SemanticFeatureExtractor, SimulationVariant, Task


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the global buffer in simulation or API mode.")
    parser.add_argument("--mode", choices=("simulation", "api"), default="simulation")
    parser.add_argument("--config", default="llm_config", help="API config module (llm_config*).")
    parser.add_argument("--prompt", help="Run one user prompt in API mode instead of the generated workload.")
    parser.add_argument("--tasks", type=int, help="Generated task count (simulation: 200, API: 1 by default).")
    parser.add_argument("--controllers", type=int, default=3)
    parser.add_argument("--agents", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch-size", type=int, default=5)
    parser.add_argument("--tick-duration", type=float, default=1.0)
    parser.add_argument("--output-dir", type=Path, default=Path("buffered_outputs"))
    return parser


def load_api_client(config_name: str, agent_count: int) -> tuple[LLMAPIClient, str]:
    cfg = import_config_module(config_name, __package__)
    endpoints = getattr(cfg, "LLM_ENDPOINTS", {})
    keys = getattr(cfg, "LLM_API_KEYS", {})
    judge = getattr(cfg, "JUDGE_MODEL_NAME", "qwen2.5-72b-instruct")
    models = [profile.model_name for profile in PROFILES[:agent_count]]
    for name in models + [judge]:
        endpoint = endpoints.get(name)
        if not endpoint:
            raise ValueError(f"Missing endpoint for {name!r} in LLM_ENDPOINTS.")
        if not api_key_from_env(name, keys.get(name), endpoint):
            raise ValueError(f"Missing API key for {name!r}. Set the provider or model key in the environment.")
    return LLMAPIClient(
        endpoints, keys, model_aliases=getattr(cfg, "LLM_MODEL_ALIASES", {}),
        max_tokens_per_model=getattr(cfg, "MAX_TOKENS_PER_MODEL", {}),
    ), judge


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    if not 1 <= args.agents <= len(PROFILES):
        parser.error(f"--agents must be between 1 and {len(PROFILES)}")
    if args.prompt is not None and (args.mode != "api" or not args.prompt.strip()):
        parser.error("--prompt requires API mode and a non-empty prompt")
    if args.tasks is not None and args.tasks < 1:
        parser.error("--tasks must be positive")

    api_options = {}
    if args.mode == "api":
        try:
            client, judge_model = load_api_client(args.config, args.agents)
        except (ImportError, ValueError) as exc:
            parser.error(f"API configuration: {exc}")
        api_options = dict(llm_api_client=client, judge_model_name=judge_model, strict_judge=True)

    np.random.seed(args.seed)
    variant = SimulationVariant(
        name="global_buffer_aspsa_api" if args.mode == "api" else "global_buffer_aspsa",
        routing_mode="adaptive",
        controller_mode="load_balancing",
        spsa_variant="aspsa",
        spsa_interval=5,
        spsa_warmup_tasks=15,
    )
    orchestrator = BufferedOrchestrator(
        num_ctrl=args.controllers,
        num_agents=args.agents,
        variant=variant,
        simulation_tick=args.tick_duration if args.mode == "simulation" else None,
        classifier_seed=args.seed,
        **api_options,
    )
    if args.prompt is None:
        task_count = args.tasks if args.tasks is not None else (1 if args.mode == "api" else 200)
        tasks = orchestrator.generate_batched_stream(
            task_count,
            batch_size=args.batch_size,
            tick_duration=args.tick_duration,
        )
    else:
        tasks = [Task(
            id=0,
            type=orchestrator.classifier.predict_from_text(args.prompt),
            t_arrival=0.0,
            h_Ti=3.0,
            phi_i=SemanticFeatureExtractor.extract(args.prompt),
            urgency=0.5,
            text=args.prompt,
        )]
    try:
        summary = orchestrator.run(tasks)
    except (BudgetExceeded, JudgeEvaluationError) as exc:
        parser.exit(1, f"{exc}\n")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(orchestrator.task_records).to_csv(
        args.output_dir / "task_records.csv", index=False
    )
    pd.DataFrame(orchestrator.event_records).to_csv(
        args.output_dir / "event_records.csv", index=False
    )
    pd.DataFrame(orchestrator.feedback_records).to_csv(
        args.output_dir / "feedback_records.csv", index=False
    )
    pd.DataFrame([summary]).to_csv(args.output_dir / "summary.csv", index=False)
    if args.prompt is None:
        print(pd.Series(summary).to_string())
    else:
        result = orchestrator.task_records[0]
        if result["execution_failed"]:
            parser.exit(1, "The selected agent's API call failed. See the log for details.\n")
        print(result["model_output"])


if __name__ == "__main__":
    main()
