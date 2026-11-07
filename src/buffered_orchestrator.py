"""Event-driven implementation of the global-buffer architecture."""

from __future__ import annotations

import heapq
import itertools
import logging
import math
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
    ControllerState,
    ExactOrchestrator,
    SimulationVariant,
    Task,
    exact_load_balancing,
    communication_graph,
    mse,
    rank_agents,
)
from .llm_api import BudgetExceeded
from .spsa_variants import call_variant


logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class BufferedPayload:
    task: Task
    routed_task: Task
    predicted_time: float
    predicted_wait: float


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
        super().__init__(
            num_ctrl=num_ctrl,
            num_agents=num_agents,
            llm_api_client=llm_api_client,
            judge_model_name=judge_model_name,
            variant=variant,
            classifier_seed=classifier_seed,
        )
        self.buffer = GlobalAssignmentBuffer()
        self.routing_learning_rate = float(routing_learning_rate)
        if simulation_tick is not None and simulation_tick <= 0.0:
            raise ValueError("simulation_tick must be positive or None")
        self.simulation_tick = simulation_tick
        self.completion_events: List[CompletionEvent] = []
        self.event_records: List[dict] = []
        self.feedback_records: List[dict] = []
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

    def _select_controller_by_load_balancing(self, task: Task) -> int:
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
                0.70 * states[idx].q_jt + 0.30 * states[idx].q_jt_next,
                self.controller_assignment_counts[idx],
                idx,
            ),
        )

    def submit_task(self, task: Task) -> int:
        """Classify, load-balance, score every task--agent pair, and buffer it."""
        self.time = float(task.t_arrival)
        predicted_type = self.classifier.predict(task)
        self.metrics["type_cls_acc"].append(
            1.0 if predicted_type == task.type else 0.0
        )
        routed_task = replace(task, type=predicted_type)
        controller_id = self._select_controller_by_load_balancing(routed_task)
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

        candidates = rank_agents(
            controller=controller,
            task=routed_task,
            agents=self.agents,
            time=self.time,
            dynamic_success=self.judge.dynamic_success,
            predicted_time=p_hat,
            predicted_wait=w_hat,
            top_k=len(self.agents),
            variant=self.variant,
        )
        # The buffer architecture requires every pair.  The legacy adaptive
        # ranker may hard-filter low-reputation agents for urgent tasks, so add
        # any missing pair using the same feature computation without filtering.
        present = {row["agent_id"] for row in candidates}
        if len(present) != len(self.agents):
            unfiltered = rank_agents(
                controller=controller,
                task=routed_task,
                agents=self.agents,
                time=self.time,
                dynamic_success=self.judge.dynamic_success,
                predicted_time=p_hat,
                predicted_wait=w_hat,
                top_k=len(self.agents),
                variant=replace(self.variant, routing_mode="best_quality_static"),
            )
            candidates.extend(row for row in unfiltered if row["agent_id"] not in present)

        proposals = []
        for candidate in candidates:
            features = np.asarray(candidate["routing_features"], dtype=float).copy()
            # Availability and queue state are owned by the buffer.  Urgency is
            # represented by the raw task urgency; dynamic hotness is added later.
            features[1] = 0.0
            features[4] = 0.0
            features[7] = float(task.urgency)
            base_score = float(np.dot(controller.routing_weights, features))
            proposals.append(
                CandidateProposal(
                    agent_id=int(candidate["agent_id"]),
                    base_score=base_score,
                    predicted_processing=float(candidate["theor_time"]),
                    predicted_quality=float(candidate["predicted_quality"]),
                    relevance=float(candidate["relevance"]),
                    cost=float(candidate["task_price"]),
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
            payload=BufferedPayload(task, routed_task, float(p_hat), float(w_hat)),
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
        profile = PROFILES[agent.id]
        self.metrics["assignments"] += 1

        result = self._execute_assignment(assignment, payload)
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
        profile = PROFILES[agent.id]
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
        try:
            api_result = self.llm_api_client.call(profile.model_name, prompt, task.type.name)
            self.api_stats["calls"] += 1
            service_time = float(api_result.get("latency") or assignment.predicted_processing)
            input_tokens = self._estimate_task_tokens(task)
            output = str(api_result.get("output", ""))
            output_tokens = max(32, int(len(output) / 4)) if output else 32
            actual_wait = max(0.0, self.time - task.t_arrival)
            return {
                "success": bool(api_result.get("success", True)),
                "service_time": service_time,
                "actual_wait": actual_wait,
                "agent_wait": 0.0,
                "controller_delay": actual_wait,
                "actual_start": self.time,
                "finish_time": self.time + service_time,
                "response_latency": actual_wait + service_time,
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "semantic_match": self._cosine_similarity(task.phi_i, profile.psi_k),
                "model_output": output,
                "synthetic_quality": None,
                "cost": MODEL_TOKEN_PRICE.get(profile.model_name, 0.5)
                * ((input_tokens + output_tokens) / 1000.0),
                "execution_failed": not bool(api_result.get("success", True)),
            }
        except BudgetExceeded:
            raise  # лимит расходов должен остановить прогон, а не стать «сбоем исполнения»
        except Exception as exc:
            self.api_stats["calls"] += 1
            self.api_stats["failed"] += 1
            logger.warning("LLM call to %s failed (%s: %s); task marked execution_failed",
                           profile.model_name, type(exc).__name__, exc)
            service_time = max(0.01, float(assignment.predicted_processing))
            input_tokens = self._estimate_task_tokens(task)
            actual_wait = max(0.0, self.time - task.t_arrival)
            return {
                "success": False,
                "execution_failed": True,
                "service_time": service_time,
                "actual_wait": actual_wait,
                "agent_wait": 0.0,
                "controller_delay": actual_wait,
                "actual_start": self.time,
                "finish_time": self.time + service_time,
                "response_latency": actual_wait + service_time,
                "input_tokens": input_tokens,
                "output_tokens": 0,
                "semantic_match": self._cosine_similarity(task.phi_i, profile.psi_k),
                "model_output": "",
                "synthetic_quality": None,
                "cost": MODEL_TOKEN_PRICE.get(profile.model_name, 0.5)
                * (input_tokens / 1000.0),
                "api_fallback": 1,
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
        profile = PROFILES[agent.id]
        buffered = self.buffer.tasks[task.id]
        selected_candidate = buffered.proposals[agent.id].proposal

        verdict = self.judge.evaluate(
            agent.id,
            task,
            base_success=1.0 if result["success"] else 0.0,
            latency=float(result["response_latency"]),
            model_output=str(result["model_output"]),
            synthetic_quality=result.get("synthetic_quality")
            if self.variant.use_judge
            else None,
        )
        quality = float(verdict["Q"])
        success = bool(verdict["success"]) and not bool(result.get("execution_failed", False))
        deadline_hit = 1.0 if event.finish_time <= buffered.deadline else 0.0
        utility = self._compute_observed_utility(
            q_true=quality,
            success=success,
            deadline_hit=deadline_hit,
            latency=float(result["response_latency"]),
            true_wait=float(result["actual_wait"]),
            cost=float(result["cost"]),
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
                float(result["actual_wait"]),
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
                true_wait=float(result["actual_wait"]),
                true_q=quality,
                urgency=float(task.urgency),
                deadline_hit=deadline_hit,
                predicted_wait=payload.predicted_wait,
                predicted_budget=buffered.time_budget,
                latency=float(result["response_latency"]),
            )

        self.buffer.complete(task.id, event.finish_time, success=success)
        self._refresh_agent_runtime(agent, event.finish_time)
        self.metrics["success_rate"].append(1.0 if success else 0.0)
        self.metrics["judge_quality"].append(quality)
        self.metrics["latency"].append(float(result["response_latency"]))
        self.metrics["deadline_hits"].append(deadline_hit)
        self.metrics["costs"].append(float(result["cost"]))
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
            task_type=task.type.name,
            classified_type=payload.routed_task.type.name,
            urgency=float(task.urgency),
            controller_id=controller.id,
            assigned=1,
            agent_id=agent.id,
            model_name=profile.model_name,
            buffered_at=float(task.t_arrival),
            assigned_at=float(assignment.dispatch_time),
            predicted_time=float(assignment.predicted_processing),
            predicted_wait=float(payload.predicted_wait),
            deadline=float(buffered.deadline),
            latency=float(result["response_latency"]),
            true_wait=float(result["actual_wait"]),
            actual_start=float(result["actual_start"]),
            finish_time=float(event.finish_time),
            service_time=float(result["service_time"]),
            cost=float(result["cost"]),
            q_hat=float(selected_candidate.predicted_quality),
            q_true=quality,
            success=int(success),
            deadline_hit=int(deadline_hit),
            routing_score=float(assignment.base_score),
            buffer_priority=float(assignment.priority),
            slack_coefficient=float(assignment.slack_coefficient),
            routing_utility=utility,
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

        self.time = max((record.get("finish_time", 0.0) for record in self.task_records), default=self.time)
        return self.collect_summary()

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
