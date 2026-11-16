"""CLI runner for the global-buffer architecture."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import time
from contextlib import ExitStack
from pathlib import Path

import numpy as np
import pandas as pd
import psycopg
from dotenv import load_dotenv

from .buffered_orchestrator import BufferedOrchestrator
from .llm_api import BudgetExceeded, LLMAPIClient, api_key_from_env, import_config_module
from .main import AgentProfile, PROFILES, JudgeEvaluationError, SemanticFeatureExtractor, SimulationVariant, Task, TaskType
from .routing_knowledge import RoutingKnowledge
from .tavily_search import TavilySearchAgent


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the global buffer in simulation or API mode.")
    parser.add_argument("--mode", choices=("simulation", "api"), default="simulation")
    parser.add_argument("--config", help="Use a Python API config module (llm_config*) instead of LLM_MODELS.")
    parser.add_argument("--prompt", help="Run one user prompt in API mode instead of the generated workload.")
    parser.add_argument("--web-search", action="store_true", help="Route the prompt to Tavily (requires TAVILY_API_KEY).")
    parser.add_argument("--knowledge-db", action="store_true", help="Persist routing experience using ROUTING_DATABASE_URL (PostgreSQL).")
    parser.add_argument("--round-size", type=int, help="Evaluated tasks per knowledge round (default: 5).")
    parser.add_argument("--mvp-rounds", type=int, help="Completed rounds before serving prompts (default: 3).")
    parser.add_argument("--warmup-file", type=Path, help="UTF-8 file with one warm-up prompt per non-empty line.")
    parser.add_argument("--tasks", type=int, help="Generated task count (simulation: 200, API: 1 by default).")
    parser.add_argument("--controllers", type=int, help="Number of controllers (default: ROUTING_CONTROLLERS or 3).")
    parser.add_argument("--agents", type=int, help="Number of agents (default: all LLM_MODELS, or 5 with built-in profiles).")
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
        token_prices=getattr(cfg, "LLM_TOKEN_PRICES", {}),
    ), judge


def load_env_api_client(models: list[str]) -> tuple[LLMAPIClient, str]:
    endpoint = os.environ.get("LLM_BASE_URL", "").strip()
    judge = os.environ.get("LLM_JUDGE_MODEL", "").strip()
    if not endpoint or not judge:
        raise ValueError("Set LLM_BASE_URL and LLM_JUDGE_MODEL when using LLM_MODELS.")

    endpoints = dict.fromkeys([*models, judge], endpoint)
    shared_key = os.environ.get("API_KEY") or os.environ.get("LLM_PROVIDER_API_KEY")
    for name in endpoints:
        if not api_key_from_env(name, None, endpoint, shared_key=shared_key):
            raise ValueError(f"Missing API key for {name!r}. Set API_KEY or a model-specific key.")
    try:
        prices = json.loads(os.environ.get("LLM_TOKEN_PRICES", "{}"))
        max_tokens = int(os.environ.get("LLM_MAX_TOKENS", "512"))
        timeout = int(os.environ.get("LLM_TIMEOUT_SECONDS", "60"))
    except ValueError:
        raise ValueError("Expected JSON in LLM_TOKEN_PRICES and integers for LLM_MAX_TOKENS and LLM_TIMEOUT_SECONDS.") from None
    if not isinstance(prices, dict):
        raise ValueError("LLM_TOKEN_PRICES must be a JSON object keyed by model ID.")
    if max_tokens < 1 or timeout < 1:
        raise ValueError("LLM_MAX_TOKENS and LLM_TIMEOUT_SECONDS must be positive.")
    return LLMAPIClient(
        endpoints, {}, token_prices=prices, max_tokens=max_tokens, timeout=timeout, shared_key=shared_key,
    ), judge


def write_reports(output_dir: Path, orchestrator: BufferedOrchestrator | None, summary: dict, mode: str) -> None:
    summary_frame = pd.DataFrame([summary])
    if summary["run_status"] == "running":
        summary_frame.to_csv(output_dir / "summary.csv", index=False)
    tasks = pd.DataFrame(orchestrator.task_records if orchestrator is not None else [])
    events = pd.DataFrame(orchestrator.event_records if orchestrator is not None else [])
    feedback = pd.DataFrame(orchestrator.feedback_records if orchestrator is not None else [])
    responses_path = output_dir / "responses.jsonl"
    if mode == "api":
        # Provider-controlled labels are still CSV cells. Keep raw labels and answers in JSONL.
        for column in ("model_name", "judge_model_name"):
            if column in tasks:
                tasks[column] = tasks[column].map(
                    lambda value: "'" + value
                    if isinstance(value, str) and value.lstrip().startswith(("=", "+", "-", "@"))
                    else value
                )
        events = events.rename(columns={"time": "scheduler_time"})
        with responses_path.open("w", encoding="utf-8") as file:
            if orchestrator is not None:
                for record in orchestrator.response_records:
                    file.write(json.dumps(record, ensure_ascii=False) + "\n")
    else:
        responses_path.unlink(missing_ok=True)
    tasks.to_csv(output_dir / "task_records.csv", index=False)
    events.to_csv(output_dir / "event_records.csv", index=False)
    feedback.to_csv(output_dir / "feedback_records.csv", index=False)
    if summary["run_status"] != "running":
        summary_frame.to_csv(output_dir / "summary.csv", index=False)


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    if args.prompt is not None and (args.mode != "api" or not args.prompt.strip()):
        parser.error("--prompt requires API mode and a non-empty prompt")
    if args.web_search and (args.mode != "api" or args.prompt is None):
        parser.error("--web-search requires --mode api and --prompt")
    if args.tasks is not None and args.tasks < 1:
        parser.error("--tasks must be positive")
    if args.batch_size < 1 or not math.isfinite(args.tick_duration) or args.tick_duration <= 0:
        parser.error("--batch-size and --tick-duration must be positive and finite")
    if not args.knowledge_db and (args.round_size is not None or args.mvp_rounds is not None):
        parser.error("--round-size and --mvp-rounds require --knowledge-db")
    load_dotenv(Path.cwd() / ".env", override=False)
    if args.controllers is None:
        try:
            args.controllers = int(os.environ.get("ROUTING_CONTROLLERS", "3"))
        except ValueError:
            parser.error("ROUTING_CONTROLLERS must be a positive integer")
    if args.controllers < 1:
        parser.error("--controllers and ROUTING_CONTROLLERS must be positive")
    env_models = os.environ.get("LLM_MODELS") if args.mode == "api" and args.config is None else None
    profiles = PROFILES
    if env_models is not None:
        models = [model.strip() for model in env_models.split(",")]
        if not all(models) or len(models) != len(set(models)):
            parser.error("LLM_MODELS must be a comma-separated list of distinct, non-empty model IDs.")
        profiles = [
            AgentProfile(i, model, model, np.full(len(TaskType), 0.5), np.ones(len(TaskType)),
                         np.full(len(TaskType), 1.0 / len(TaskType)))
            for i, model in enumerate(models)
        ]
    if args.agents is None:
        args.agents = len(profiles) if env_models is not None else 5
    if not 1 <= args.agents <= len(profiles):
        parser.error(f"--agents must be between 1 and {len(profiles)}")
    if env_models is not None:
        profiles = profiles[:args.agents]
    knowledge_dsn = os.environ.get("ROUTING_DATABASE_URL", "").strip() if args.knowledge_db else ""
    if args.knowledge_db and not knowledge_dsn:
        parser.error("Set ROUTING_DATABASE_URL to use --knowledge-db")

    warmup_prompts = []
    if args.warmup_file is not None:
        if not args.knowledge_db or args.prompt is not None:
            parser.error("--warmup-file requires --knowledge-db and cannot be combined with --prompt")
        try:
            warmup_prompts = [
                line.strip() for line in args.warmup_file.read_text(encoding="utf-8").splitlines() if line.strip()
            ]
        except (OSError, UnicodeError) as exc:
            parser.error(f"Warm-up file: {exc}")
        if not warmup_prompts:
            parser.error("Warm-up file must contain at least one prompt")
        warmup_prompts = warmup_prompts[:args.tasks]
    with ExitStack() as resources:
        run_configured(args, parser, resources, warmup_prompts, knowledge_dsn, profiles, env_models is not None)


def run_configured(
    args: argparse.Namespace, parser: argparse.ArgumentParser, resources: ExitStack, warmup_prompts: list[str], knowledge_dsn: str,
    profiles: list[AgentProfile], use_env_models: bool,
) -> None:
    tavily_key = os.environ.get("TAVILY_API_KEY", "").strip() if args.mode == "api" else ""
    if args.web_search and not tavily_key:
        parser.error("Set TAVILY_API_KEY to use --web-search")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_reports(args.output_dir, None, {"mode": args.mode, "run_status": "running"}, args.mode)
    api_options = {}
    agent_names = [profile.model_name for profile in profiles[:args.agents]]
    if args.mode == "api":
        try:
            if use_env_models:
                client, judge_model = load_env_api_client(agent_names)
            else:
                client, judge_model = load_api_client(args.config or "llm_config", args.agents)
        except (ImportError, ValueError) as exc:
            write_reports(args.output_dir, None, {"mode": args.mode, "run_status": "failed"}, args.mode)
            parser.error(f"API configuration: {exc}")
        api_options = dict(llm_api_client=client, judge_model_name=judge_model, strict_judge=True)
        if tavily_key:
            api_options["search_agent"] = TavilySearchAgent(tavily_key, reserve_call=client.reserve_call)

    knowledge = None
    if knowledge_dsn:
        scope = {"mode": args.mode, "controllers": args.controllers, "agents": agent_names}
        # The optional search worker reserves the slot after these LLMs. Its
        # WEB_SEARCH experience can safely coexist in an existing database.
        if args.mode == "api":
            # Bind experience to the configured provider and model, without storing URLs or keys.
            scope["api_models"] = {
                name: {
                    "model": client.model_aliases.get(name, name),
                    "endpoint_hash": hashlib.sha256(client.endpoints[name].rstrip("/").encode()).hexdigest(),
                }
                for name in agent_names + [judge_model]
            }
            scope["judge"] = judge_model
        try:
            knowledge = RoutingKnowledge(
                knowledge_dsn, scope, round_size=args.round_size, mvp_rounds=args.mvp_rounds,
            )
            resources.callback(knowledge.close)
            if args.prompt is not None and not knowledge.ready:
                agents, controllers, prompts = knowledge.coverage()
                raise ValueError(
                    f"MVP warm-up incomplete: {knowledge.completed_rounds}/{knowledge.mvp_rounds} rounds; "
                    f"{len(agents)}/{args.agents} agents, {len(controllers)}/{args.controllers} controllers, "
                    f"{prompts}/2 distinct prompts. Coverage is checked at the end of a round."
                )
        except psycopg.Error as exc:
            write_reports(args.output_dir, None, {"mode": args.mode, "run_status": "failed"}, args.mode)
            parser.error(f"Knowledge database: PostgreSQL setup failed ({type(exc).__name__}). Check the connection and permissions.")
        except ValueError as exc:
            write_reports(args.output_dir, None, {"mode": args.mode, "run_status": "failed"}, args.mode)
            parser.error(f"Knowledge database: {exc}")

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
        knowledge=knowledge,
        agent_profiles=profiles,
        **api_options,
    )
    if args.prompt is None and not warmup_prompts:
        task_count = args.tasks if args.tasks is not None else (1 if args.mode == "api" else 200)
        tasks = orchestrator.generate_batched_stream(
            task_count,
            batch_size=args.batch_size,
            tick_duration=args.tick_duration,
        )
    else:
        prompts = [args.prompt] if args.prompt is not None else warmup_prompts
        tasks = [Task(
            id=index,
            type=TaskType.QA if args.web_search else orchestrator.classifier.predict_from_text(prompt),
            t_arrival=(index // args.batch_size) * args.tick_duration,
            h_Ti=3.0,
            phi_i=SemanticFeatureExtractor.extract(prompt),
            urgency=0.5,
            text=prompt,
            requires_web_search=args.web_search,
        ) for index, prompt in enumerate(prompts)]
    error = None
    exit_code = 1
    status = "completed"
    started = time.perf_counter()
    try:
        summary = orchestrator.run(tasks)
    except psycopg.Error as exc:
        summary = orchestrator.collect_summary(include_knowledge=False)
        error = f"Knowledge database: PostgreSQL operation failed ({type(exc).__name__})."
        status = "failed"
    except (BudgetExceeded, JudgeEvaluationError) as exc:
        summary = orchestrator.collect_summary()
        error = str(exc)
        status = "failed"
    except KeyboardInterrupt:
        summary = orchestrator.collect_summary()
        error = "Run interrupted. Collected results were saved."
        exit_code = 130
        status = "interrupted"
    if args.mode == "api":
        summary["wall_run_seconds"] = time.perf_counter() - started
        failed_calls = int(summary["api_failed_calls"])
        unsuccessful = sum(not row["success"] for row in orchestrator.task_records)
        if unsuccessful and error is None:
            status = "failed" if unsuccessful == len(tasks) else "partial"
            error = (
                f"{unsuccessful} task(s) had no accepted result ({failed_calls} agent API failure(s)). "
                "See the run reports for details."
            )
    summary.update(mode=args.mode, run_status=status)
    write_reports(args.output_dir, orchestrator, summary, args.mode)
    if error is not None:
        parser.exit(exit_code, f"{error}\n")
    if args.prompt is None:
        print(pd.Series(summary).to_string())
    else:
        print(orchestrator.response_records[0]["model_output"])


if __name__ == "__main__":
    main()
