"""Event-driven implementation of the global-buffer architecture."""

from __future__ import annotations

import hashlib
import heapq
import itertools
import logging
import math
import time
from dataclasses import dataclass, field, replace
from typing import Dict, List, Optional, Sequence

import numpy as np

from .assignment_buffer import (
    Assignment,
    CandidateProposal,
    GlobalAssignmentBuffer,
    TaskStatus,
)
from .main import (
    MODEL_TOKEN_PRICE,
    PROFILES,
    AgentProfile,
    ControllerState,
    ExactOrchestrator,
    SimulationVariant,
    Task,
    TaskType,
    exact_load_balancing,
    communication_graph,
    mse,
    rank_agents,
)
from .llm_api import BudgetExceeded
from .routing_knowledge import RoutingKnowledge
from .spsa_variants import call_variant
from .tavily_search import TavilySearchAgent


logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class BufferedPayload:
    task: Task
    routed_task: Task
    predicted_time: float
    predicted_wait: float
    wall_arrived: Optional[float] = None


@dataclass(order=True)
class CompletionEvent:
    finish_time: float
    sequence: int
    assignment: Assignment = field(compare=False)
    result: dict = field(compare=False)


class BufferedOrchestrator(ExactOrchestrator):
    """Orchestrator in which controllers score and the buffer dispatches.

    Processing is event-driven.  An event is a task arrival or an agent
    completion.  Agents never receive a second task while busy.
    """

    def __init__(
        self,
        num_ctrl: int = 3,
        num_agents: int = 5,
        llm_api_client=None,
        judge_model_name: str = "qwen2.5-72b-instruct",
        variant: Optional[SimulationVariant] = None,
        routing_learning_rate: float = 0.05,
        simulation_tick: Optional[float] = None,
        classifier_seed: int = 42,
        strict_judge: bool = False,
        knowledge: RoutingKnowledge | None = None,
        search_agent: TavilySearchAgent | None = None,
    ) -> None:
        variant = variant or SimulationVariant(
            name="global_buffer_aspsa",
            routing_mode="adaptive",
            controller_mode="load_balancing",
            use_spsa=True,
            use_learning=True,
            use_judge=True,
            spsa_variant="aspsa",
            spsa_interval=5,
            spsa_warmup_tasks=15,
        )
        self.search_agent = search_agent
        self.search_agent_id = None
        profiles = PROFILES
        if search_agent is not None:
            if llm_api_client is None:
                raise ValueError("Tavily search requires API mode.")
            self.search_agent_id = num_agents
            profiles = [*PROFILES[:num_agents], AgentProfile(
                id=num_agents,
                name="Tavily Search",
                model_name=search_agent.name,
                P_success=np.full(len(TaskType), 0.5),
                tau_avg=np.ones(len(TaskType)),
                psi_k=np.array([0.0, 1.0, 0.0, 0.0, 0.0]),
            )]
        super().__init__(
            num_ctrl=num_ctrl,
            num_agents=num_agents + int(search_agent is not None),
            llm_api_client=llm_api_client,
            judge_model_name=judge_model_name,
            variant=variant,
            classifier_seed=classifier_seed,
            agent_profiles=profiles,
        )
        self.buffer = GlobalAssignmentBuffer()
        self.routing_learning_rate = float(routing_learning_rate)
        if simulation_tick is not None and simulation_tick <= 0.0:
            raise ValueError("simulation_tick must be positive or None")
        self.simulation_tick = simulation_tick
        self.strict_judge = strict_judge
        self.knowledge = knowledge
        self.completion_events: List[CompletionEvent] = []
        self.event_records: List[dict] = []
        self.feedback_records: List[dict] = []
        self.response_records: List[dict] = []
        self._api_costs: List[Optional[float]] = []
        self._api_clock_start: Optional[float] = None
        self.controller_assignment_counts = {ctrl.id: 0 for ctrl in self.controllers}
        self._completion_sequence = itertools.count()
        self._completed_updates = 0
        self._spsa_round = 0

    def _controller_load(self, controller_id: int) -> float:
        load = 0.0
        for buffered in self.buffer.tasks.values():
            if (
                buffered.controller_id == controller_id
                and buffered.status in (TaskStatus.PENDING, TaskStatus.ASSIGNED)
            ):
                task = buffered.payload.task
                load += 1.0 + 0.5 * float(task.urgency)
        return load

    def _select_controller_by_load_balancing(self, task: Task, seen_controllers: frozenset[int] = frozenset()) -> int:
        """LVP-style assignment of one task to one responsible controller."""
        if len(self.controllers) == 1:
            return 0
        states = [
            ControllerState(
                q_jt=self._controller_load(ctrl.id),
                p_j=max(0.25, self._controller_reliability(ctrl)),
                z_jt=1.0 + 0.5 * float(task.urgency),
            )
            for ctrl in self.controllers
        ]
        graph = communication_graph(len(self.buffer.tasks), len(self.controllers))
        exact_load_balancing(states, graph)
        return min(
            range(len(states)),
            key=lambda idx: (
                idx in seen_controllers,
                0.70 * states[idx].q_jt + 0.30 * states[idx].q_jt_next,
                self.controller_assignment_counts[idx],
                idx,
            ),
        )

    def submit_task(self, task: Task) -> int:
        """Classify, load-balance, and score the workers that can execute the task."""
        if task.requires_web_search and self.search_agent is None:
            raise ValueError("Web search requires TAVILY_API_KEY in API mode.")
        eligible = [
            agent for agent in self.agents
            if (agent.id == self.search_agent_id) == task.requires_web_search
        ]
        wall_arrived = None
        if self.llm_api_client is not None:
            now = time.perf_counter()
            if self._api_clock_start is None:
                self._api_clock_start = now
            wall_arrived = now - self._api_clock_start
        self.time = float(task.t_arrival)
        predicted_type = TaskType.QA if task.requires_web_search else self.classifier.predict(task)
        self.metrics["type_cls_acc"].append(
            1.0 if predicted_type == task.type else 0.0
        )
        routed_task = replace(task, type=predicted_type)
        task_kind = "WEB_SEARCH" if task.requires_web_search else predicted_type.name
        warmup = self.knowledge is not None and not task.requires_web_search and not self.knowledge.ready
        seen_agents, seen_controllers, _ = self.knowledge.coverage() if warmup else (set(), set(), 0)
        controller_id = self._select_controller_by_load_balancing(routed_task, frozenset(seen_controllers))
        controller = self.controllers[controller_id]
        self.controller_assignment_counts[controller_id] += 1

        raw_p = controller.predict_time(routed_task.type.value, routed_task.phi_i)
        raw_w = controller.predict_wait(routed_task.type.value, routed_task.phi_i)
        p_hat = self._apply_controller_calibration(
            controller, raw_p, controller.time_bias, controller.time_conf, 0.5, 10.0
        )
        w_hat = self._apply_controller_calibration(
            controller, raw_w, controller.wait_bias, controller.wait_conf, 0.3, 5.0
        )

        api_input_prices = None
        if self.llm_api_client is not None:
            prices = getattr(self.llm_api_client, "token_prices", {})
            aliases = getattr(self.llm_api_client, "model_aliases", {})
            api_input_prices = {}
            for agent in eligible:
                name = self.agent_profiles[agent.id].model_name
                rate = prices.get(aliases.get(name, name))
                if rate is not None:
                    api_input_prices[name] = rate[0]
            if len(api_input_prices) != len(eligible):
                # Avoid favouring an unknown tariff over a known one.
                api_input_prices.clear()
        candidates = rank_agents(
            controller=controller,
            task=routed_task,
            agents=eligible,
            time=self.time,
            dynamic_success=self.judge.dynamic_success,
            predicted_time=p_hat,
            predicted_wait=w_hat,
            top_k=len(eligible),
            variant=self.variant,
            agent_profiles=self.agent_profiles,
            api_input_prices=api_input_prices,
        )
        # The buffer architecture requires every eligible pair. The adaptive
        # ranker may hard-filter low-reputation agents for urgent tasks, so add
        # any missing pair using the same feature computation without filtering.
        present = {row["agent_id"] for row in candidates}
        if len(present) != len(eligible):
            unfiltered = rank_agents(
                controller=controller,
                task=routed_task,
                agents=eligible,
                time=self.time,
                dynamic_success=self.judge.dynamic_success,
                predicted_time=p_hat,
                predicted_wait=w_hat,
                top_k=len(eligible),
                variant=replace(self.variant, routing_mode="best_quality_static"),
                agent_profiles=self.agent_profiles,
                api_input_prices=api_input_prices,
            )
            candidates.extend(row for row in unfiltered if row["agent_id"] not in present)

        proposals = []
        for candidate in candidates:
            features = np.asarray(candidate["routing_features"], dtype=float).copy()
            predicted_processing = float(candidate["theor_time"])
            predicted_quality = float(candidate["predicted_quality"])
            if self.knowledge is not None:
                predicted_quality, predicted_processing = self.knowledge.estimate(
                    controller_id, task_kind, int(candidate["agent_id"]),
                    predicted_quality, predicted_processing,
                )
                features[2] = predicted_quality
                features[3] = np.exp(-predicted_processing / max(p_hat + 0.5, 0.5))
            # Availability and queue state are owned by the buffer.  Urgency is
            # represented by the raw task urgency; dynamic hotness is added later.
            features[1] = 0.0
            features[4] = 0.0
            features[7] = float(task.urgency)
            if task.requires_web_search:
                # Tavily usage is measured in credits, not in LLM tokens.
                features[5] = 1.0
            base_score = float(np.dot(controller.routing_weights, features))
            if warmup and candidate["agent_id"] not in seen_agents:
                base_score += 1.0  # Try uncovered executors before declaring the pool ready.
            proposals.append(
                CandidateProposal(
                    agent_id=int(candidate["agent_id"]),
                    base_score=base_score,
                    predicted_processing=predicted_processing,
                    predicted_quality=predicted_quality,
                    relevance=float(candidate["relevance"]),
                    cost=0.0 if task.requires_web_search else float(candidate["task_price"]),
                    feature_vector=tuple(float(x) for x in features),
                    predicted_wait_budget=float(w_hat),
                )
            )

        budget = max(float(p_hat + w_hat), 1e-6)
        deadline = float(task.t_arrival + budget)
        self.buffer.submit(
            task_id=task.id,
            controller_id=controller_id,
            arrival_time=task.t_arrival,
            deadline=deadline,
            time_budget=budget,
            proposals=proposals,
            payload=BufferedPayload(task, routed_task, float(p_hat), float(w_hat), wall_arrived),
        )
        self.event_records.append(
            {
                "event": "buffered",
                "time": self.time,
                "task_id": task.id,
                "controller_id": controller_id,
                "candidate_count": len(proposals),
            }
        )
        return controller_id

    def _agent_availability(self) -> Dict[int, float]:
        return {agent.id: float(agent.available_at) for agent in self.agents}

    def dispatch_ready(self) -> List[Assignment]:
        assignments = self.buffer.dispatch(self.time, self._agent_availability())
        for assignment in assignments:
            self._start_assignment(assignment)
        return assignments

    def _start_assignment(self, assignment: Assignment) -> None:
        payload: BufferedPayload = assignment.payload
        task = payload.task
        agent = self.agents[assignment.agent_id]
        self.metrics["assignments"] += 1

        result = self._execute_assignment(assignment, payload)
        if self.knowledge is not None and not result.get("result_available", True):
            task_kind = "WEB_SEARCH" if task.requires_web_search else payload.routed_task.type.name
            self.knowledge.record_failure(assignment.controller_id, task_kind, agent.id)
        if self.llm_api_client is not None and not result["execution_failed"]:
            self.response_records.append({
                "task_id": task.id,
                "agent_id": agent.id,
                "model_name": result["model_name"],
                "model_output": result["model_output"],
            })
        self._enqueue_agent_task(
            agent,
            task,
            start_time=float(result["actual_start"]),
            finish_time=float(result["finish_time"]),
            service_time=float(result["service_time"]),
        )
        self._refresh_agent_runtime(agent, self.time)
        heapq.heappush(
            self.completion_events,
            CompletionEvent(
                finish_time=float(result["finish_time"]),
                sequence=next(self._completion_sequence),
                assignment=assignment,
                result=result,
            ),
        )
        self.event_records.append(
            {
                "event": "assigned",
                "time": self.time,
                "task_id": task.id,
                "controller_id": assignment.controller_id,
                "agent_id": assignment.agent_id,
                "priority": assignment.priority,
                "slack": assignment.slack_coefficient,
            }
        )

    def _execute_assignment(self, assignment: Assignment, payload: BufferedPayload) -> dict:
        task = payload.task
        agent = self.agents[assignment.agent_id]
        profile = self.agent_profiles[agent.id]
        if self.llm_api_client is None:
            result = self._simulate_no_api_execution(
                task,
                agent,
                profile,
                p_hat=assignment.predicted_processing,
                w_hat=payload.predicted_wait,
                dispatch_time=self.time,
                request_time=task.t_arrival,
            )
            result["cost"] = MODEL_TOKEN_PRICE.get(profile.model_name, 0.5) * (
                (result["input_tokens"] + result["output_tokens"]) / 1000.0
            )
            self._quantize_synthetic_duration(result)
            return result

        prompt = task.text or (
            f"{task.type.name}: complexity={task.h_Ti:.2f}, urgency={task.urgency:.2f}"
        )
        started = time.perf_counter()
        requested_model = getattr(self.llm_api_client, "model_aliases", {}).get(
            profile.model_name, profile.model_name
        )
        try:
            if agent.id == self.search_agent_id:
                api_result = self.search_agent.search(task.text)
            else:
                api_result = self.llm_api_client.call(profile.model_name, prompt, task.type.name)
        except BudgetExceeded:
            raise  # лимит расходов должен остановить прогон, а не стать «сбоем исполнения»
        except Exception as exc:
            logger.warning("API call to %s failed (%s: %s); task marked execution_failed",
                           requested_model, type(exc).__name__, exc)
            api_result = {"success": False}
        finished = time.perf_counter()
        success = bool(api_result.get("success", True))
        output = str(api_result.get("output") or "")
        self.api_stats["calls"] += 1
        self.api_stats["failed"] += int(not success)
        cost = api_result.get("cost_usd")
        self._api_costs.append(cost)
        service_time = float(api_result.get("latency") or (finished - started))
        actual_wait = max(0.0, self.time - task.t_arrival)
        return {
            "success": success,
            "execution_failed": not success,
            "service_time": service_time,
            "actual_wait": actual_wait,
            "actual_start": self.time,
            "finish_time": self.time + service_time,
            "response_latency": actual_wait + service_time,
            "input_tokens": api_result.get("input_tokens"),
            "output_tokens": api_result.get("output_tokens"),
            "model_name": api_result.get("model_name", requested_model),
            "model_output": output,
            "result_available": success and bool(output.strip()) and bool(api_result.get("result_available", True)),
            "semantic_match": self._cosine_similarity(task.phi_i, profile.psi_k),
            "synthetic_quality": None,
            "cost": cost,
            "search_credits": api_result.get("search_credits"),
            "wall_worker_started": started - self._api_clock_start,
            "wall_worker_finished": finished - self._api_clock_start,
        }

    def _quantize_synthetic_duration(self, result: dict) -> None:
        """Optionally express numerical experiments in discrete time ticks."""
        if self.simulation_tick is None:
            return
        ticks = max(1, math.ceil(float(result["service_time"]) / self.simulation_tick))
        service_time = ticks * self.simulation_tick
        result["service_time"] = service_time
        result["finish_time"] = float(result["actual_start"]) + service_time
        result["response_latency"] = float(result["actual_wait"]) + service_time
        result["service_ticks"] = ticks

    def _finish_event(self, event: CompletionEvent) -> None:
        assignment = event.assignment
        payload: BufferedPayload = assignment.payload
        task = payload.task
        result = event.result
        controller = self.controllers[assignment.controller_id]
        agent = self.agents[assignment.agent_id]
        profile = self.agent_profiles[agent.id]
        task_kind = "WEB_SEARCH" if task.requires_web_search else payload.routed_task.type.name
        buffered = self.buffer.tasks[task.id]
        selected_candidate = buffered.proposals[agent.id].proposal

        execution_failed = bool(result.get("execution_failed", False))
        result_available = bool(result.get("result_available", not execution_failed))
        api_mode = self.llm_api_client is not None
        judge_usage = None
        judge_attempted = False
        judge_latency = None
        previous_evaluations = self.judge.judge_stats["llm_ok"]
        if self.strict_judge and not result_available:
            # A failed call or empty search leaves no answer for the judge.
            verdict = {"Q": 0.0, "success": False}
        else:
            judge_started = time.perf_counter() if api_mode else None
            previous_failures = self.judge.judge_stats["api_failed"]
            try:
                verdict = self.judge.evaluate(
                    agent.id,
                    task,
                    base_success=1.0 if result["success"] else 0.0,
                    latency=float(result["response_latency"]),
                    model_output=str(result["model_output"]),
                    synthetic_quality=result.get("synthetic_quality") if self.variant.use_judge else None,
                    allow_heuristic_fallback=not self.strict_judge,
                )
            finally:
                judge_usage = self.judge.last_api_usage
                judge_attempted = judge_usage is not None or self.judge.judge_stats["api_failed"] > previous_failures
                if api_mode and judge_attempted:
                    judge_latency = time.perf_counter() - judge_started
                    self._api_costs.append((judge_usage or {}).get("cost_usd"))
        quality = float(verdict["Q"])
        success = bool(verdict["success"]) and result_available
        if self.knowledge is not None and result_available and (
            not api_mode or self.judge.judge_stats["llm_ok"] > previous_evaluations
        ):
            text = " ".join(task.text.casefold().split())
            digest = hashlib.sha256(text.encode("utf-8")).hexdigest() if text else None
            round_number = self.knowledge.record(
                controller.id, task_kind, agent.id, success=success, quality=quality,
                service_seconds=float(result["service_time"]), prompt_digest=digest,
            )
            if round_number is not None:
                self.event_records.append({
                    "event": "knowledge_round", "time": event.finish_time, "round": round_number,
                })
        latency = float(result["response_latency"])
        actual_wait = float(result["actual_wait"])
        deadline_hit = float((success if api_mode else not execution_failed) and event.finish_time <= buffered.deadline)
        cost = result["cost"]
        api_record = {}
        timing = {
            "buffered_at": float(task.t_arrival),
            "assigned_at": float(assignment.dispatch_time),
            "deadline": float(buffered.deadline),
            "actual_start": float(result["actual_start"]),
            "finish_time": float(event.finish_time),
            "service_time": float(result["service_time"]),
        }
        if api_mode:
            wall_finished = time.perf_counter() - self._api_clock_start
            latency = wall_finished - payload.wall_arrived
            actual_wait = result["wall_worker_started"] - payload.wall_arrived
            wall_deadline = payload.wall_arrived + buffered.time_budget
            judge_usage = judge_usage or {}
            judge_cost = judge_usage.get("cost_usd")
            if judge_attempted:
                cost = cost + judge_cost if cost is not None and judge_cost is not None else None
            api_record = {
                "execution_failed": execution_failed,
                "result_available": result_available,
                "judge_verdict": verdict.get("judge_verdict"),
                "input_tokens": result["input_tokens"],
                "output_tokens": result["output_tokens"],
                "worker_cost_usd": result["cost"],
                "worker_finished_at": result["wall_worker_finished"],
                "judge_model_name": judge_usage.get("model_name"),
                "judge_input_tokens": judge_usage.get("input_tokens"),
                "judge_output_tokens": judge_usage.get("output_tokens"),
                "judge_cost_usd": judge_cost,
                "judge_latency": judge_latency,
                **{f"scheduler_{key}": value for key, value in timing.items()},
                "scheduler_latency": result["response_latency"],
                "scheduler_deadline_hit": deadline_hit,
            }
            if task.requires_web_search:
                api_record["tavily_credits"] = result["search_credits"]
            timing = {
                "buffered_at": payload.wall_arrived,
                "assigned_at": result["wall_worker_started"],
                "deadline": wall_deadline,
                "actual_start": result["wall_worker_started"],
                "finish_time": wall_finished,
                "service_time": result["wall_worker_finished"] - result["wall_worker_started"],
            }
            deadline_hit = float(success and wall_finished <= wall_deadline)
        utility = self._compute_observed_utility(
            q_true=quality,
            success=success,
            deadline_hit=deadline_hit,
            latency=latency,
            true_wait=actual_wait,
            cost=cost if cost is not None else 0.0,
            semantic_match=float(result["semantic_match"]),
            urgency=float(task.urgency),
            predicted_wait=payload.predicted_wait,
            predicted_budget=buffered.time_budget,
        )

        # Feedback is deliberately local: only the controller that published
        # this task's candidate pairs is updated.
        if self.variant.use_learning:
            controller.update_agent_rating(
                agent.id,
                payload.routed_task.type.value,
                judge_quality=quality,
                confidence=float(verdict.get("confidence", 0.5)),
            )
            controller.rls_quality_update(
                payload.routed_task.type.value,
                task.phi_i,
                profile.psi_k,
                payload.predicted_time,
                payload.predicted_wait,
                quality,
            )
            controller.rls_update(
                payload.routed_task.type.value,
                float(result["service_time"]),
                actual_wait,
                task.phi_i,
            )
            controller.adapt_routing_weights(
                np.asarray(assignment.feature_vector, dtype=float),
                observed_utility=utility,
                learning_rate=self.routing_learning_rate,
            )
            self._update_controller_meta(
                controller=controller,
                routing_features=np.asarray(assignment.feature_vector, dtype=float),
                utility=utility,
                raw_p_hat=payload.predicted_time,
                raw_w_hat=payload.predicted_wait,
                raw_q_hat=selected_candidate.predicted_quality,
                service_time=float(result["service_time"]),
                true_wait=actual_wait,
                true_q=quality,
                urgency=float(task.urgency),
                deadline_hit=deadline_hit,
                predicted_wait=payload.predicted_wait,
                predicted_budget=buffered.time_budget,
                latency=latency,
            )

        self.buffer.complete(task.id, event.finish_time, success=success)
        self._refresh_agent_runtime(agent, event.finish_time)
        self.metrics["success_rate"].append(1.0 if success else 0.0)
        self.metrics["judge_quality"].append(quality)
        self.metrics["latency"].append(latency)
        self.metrics["deadline_hits"].append(deadline_hit)
        if cost is not None:
            self.metrics["costs"].append(cost)
        self.metrics["weighted_success"].append(
            (1.0 if success else 0.0) * (1.0 + task.urgency)
        )
        self.metrics["routing_objective"].append(utility)
        self.metrics["Q_true"].append(quality)
        self.metrics["Q_pred"].append(selected_candidate.predicted_quality)
        self.metrics["Q_mse"].append(
            (quality - selected_candidate.predicted_quality) ** 2
        )
        self.metrics["benchmark_dynamic_mean"].append(
            float(np.mean(self.judge.dynamic_success))
        )
        self.metrics["mse_time"].append(
            mse(controller.theta_time, controller.obs_history, 0)
        )
        self.metrics["mse_wait"].append(
            mse(controller.theta_wait, controller.obs_history, 1)
        )
        self.controller_Q_sums[controller.id] += quality
        self.controller_Q_counts[controller.id] += 1
        self.controller_Q_mse_sums[controller.id] += (
            quality - selected_candidate.predicted_quality
        ) ** 2
        self.feedback_records.append(
            {
                "task_id": task.id,
                "controller_id": controller.id,
                "agent_id": agent.id,
                "quality": quality,
                "success": int(success),
                "utility": utility,
            }
        )
        self._record_task(
            variant=self.variant.name,
            parent_id=task.id,
            local_id=0,
            task_type=task_kind if task.requires_web_search else task.type.name,
            classified_type=task_kind,
            urgency=float(task.urgency),
            controller_id=controller.id,
            assigned=1,
            agent_id=agent.id,
            model_name=result.get("model_name", profile.model_name),
            predicted_time=float(assignment.predicted_processing),
            predicted_wait=float(payload.predicted_wait),
            latency=latency,
            true_wait=actual_wait,
            cost=cost,
            q_hat=float(selected_candidate.predicted_quality),
            q_true=quality,
            success=int(success),
            deadline_hit=int(deadline_hit),
            routing_score=float(assignment.base_score),
            buffer_priority=float(assignment.priority),
            slack_coefficient=float(assignment.slack_coefficient),
            routing_utility=utility,
            **timing,
            **api_record,
        )
        self.event_records.append(
            {
                "event": "completed" if success else "failed",
                "time": event.finish_time,
                "task_id": task.id,
                "controller_id": controller.id,
                "agent_id": agent.id,
            }
        )
        self._completed_updates += 1
        self._maybe_synchronize()

    def _maybe_synchronize(self) -> None:
        interval = max(1, int(self.variant.spsa_interval))
        warmup = max(interval * 2, int(self.variant.spsa_warmup_tasks))
        if (
            not self.variant.use_spsa
            or self._completed_updates < warmup
            or self._completed_updates % interval != 0
        ):
            return
        self._spsa_round += 1
        call_variant(
            self.variant.spsa_variant,
            self.controllers,
            self._spsa_round,
            alpha=self.variant.spsa_alpha,
            beta=self.variant.spsa_beta,
            beta_nes_max=self.variant.spsa_beta_nes_max,
        )
        self._log_theta(self._completed_updates, self._spsa_round)
        self.event_records.append(
            {
                "event": "aspsa_sync",
                "time": self.time,
                "round": self._spsa_round,
            }
        )

    def run(self, tasks: Sequence[Task]):
        """Run until all arrivals, buffered tasks, and executions are terminal."""
        ordered = sorted(tasks, key=lambda task: (task.t_arrival, task.id))
        next_task = 0
        eps = 1e-10

        while next_task < len(ordered) or self.completion_events or any(
            True for _ in self.buffer.pending_tasks()
        ):
            next_arrival = (
                float(ordered[next_task].t_arrival)
                if next_task < len(ordered)
                else float("inf")
            )
            next_completion = (
                float(self.completion_events[0].finish_time)
                if self.completion_events
                else float("inf")
            )
            next_time = min(next_arrival, next_completion)
            if not np.isfinite(next_time):
                raise RuntimeError("pending tasks cannot make progress")
            self.time = next_time

            while (
                self.completion_events
                and self.completion_events[0].finish_time <= self.time + eps
            ):
                event = heapq.heappop(self.completion_events)
                self._finish_event(event)

            while (
                next_task < len(ordered)
                and ordered[next_task].t_arrival <= self.time + eps
            ):
                self.submit_task(ordered[next_task])
                next_task += 1

            self.dispatch_ready()
            self.metrics["queue_len"].append(
                float(sum(1 for _ in self.buffer.pending_tasks()))
            )

        self.time = max(
            (record.get("scheduler_finish_time", record["finish_time"]) for record in self.task_records),
            default=self.time,
        )
        return self.collect_summary()

    def collect_summary(self, *, include_knowledge: bool = True):
        summary = super().collect_summary()
        if self.search_agent is not None:
            summary.update(
                tavily_calls=self.search_agent.calls_made,
                tavily_known_credits=self.search_agent.known_credits,
            )
        if include_knowledge and self.knowledge is not None:
            observations = self.knowledge.observations
            rounds = observations // self.knowledge.round_size
            agents, controllers, prompts = self.knowledge.coverage()
            summary.update(
                knowledge_observations=observations,
                knowledge_rounds=rounds,
                knowledge_failures=self.knowledge.failures,
                knowledge_agents_covered=len(agents),
                knowledge_controllers_covered=len(controllers),
                knowledge_distinct_prompts=prompts,
                mvp_required_rounds=self.knowledge.mvp_rounds,
                mvp_ready=self.knowledge.ready,
            )
        if self.llm_api_client is not None:
            unknown = self._api_costs.count(None)
            known_cost = sum(cost for cost in self._api_costs if cost is not None)
            attempts = self.api_stats["calls"]
            success_count = sum(self.metrics["success_rate"])
            summary.update(
                total_cost=None if unknown else known_cost,
                mean_cost_per_task=known_cost / attempts if not unknown and attempts else None,
                cost_per_success=known_cost / success_count if not unknown and success_count else None,
                known_api_cost_usd=known_cost,
                unpriced_api_calls=unknown,
                scheduler_deadline_hit_rate=self._safe_mean(
                    [row["scheduler_deadline_hit"] for row in self.task_records]
                ),
            )
        return summary

    def generate_batched_stream(
        self,
        task_count: int,
        batch_size: int = 5,
        tick_duration: float = 1.0,
    ) -> List[Task]:
        """Generate numerical workloads with a fixed number of arrivals per tick."""
        if batch_size <= 0 or tick_duration <= 0.0:
            raise ValueError("batch_size and tick_duration must be positive")
        generated = self.generate_realistic_stream(task_count)
        return [
            replace(task, t_arrival=(index // batch_size) * tick_duration)
            for index, task in enumerate(generated)
        ]
