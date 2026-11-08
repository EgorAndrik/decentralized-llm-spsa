import unittest
from types import SimpleNamespace

import numpy as np

from src.assignment_buffer import CandidateStatus, TaskStatus
from src.buffered_orchestrator import BufferedOrchestrator
from src.main import SimulationVariant, Task, TaskType


def make_task(task_id, arrival=0.0, urgency=0.5, task_type=TaskType.PROGRAMMING):
    features = {
        TaskType.PROGRAMMING: np.array([0.9, 0.3, 0.2, 0.2, 0.2]),
        TaskType.QA: np.array([0.3, 0.9, 0.4, 0.2, 0.2]),
    }.get(task_type, np.ones(5) / 5.0)
    return Task(
        id=task_id,
        type=task_type,
        t_arrival=arrival,
        h_Ti=2.0,
        phi_i=features,
        urgency=urgency,
        text="Build a Python utility." if task_type == TaskType.PROGRAMMING else "Answer the question.",
    )


def variant(**overrides):
    values = dict(
        name="buffer_test",
        routing_mode="adaptive",
        controller_mode="load_balancing",
        use_spsa=False,
        use_learning=True,
        use_judge=True,
        spsa_variant="aspsa",
    )
    values.update(overrides)
    return SimulationVariant(**values)


class TestBufferedOrchestrator(unittest.TestCase):
    def setUp(self):
        np.random.seed(7)

    def test_every_controller_task_publishes_every_agent_pair(self):
        orch = BufferedOrchestrator(num_ctrl=2, num_agents=4, variant=variant())
        controller_id = orch.submit_task(make_task(1))

        buffered = orch.buffer.tasks[1]
        self.assertEqual(buffered.controller_id, controller_id)
        self.assertEqual(set(buffered.proposals), {0, 1, 2, 3})

    def test_api_proposals_use_configured_tariffs_instead_of_legacy_profiles(self):
        client = SimpleNamespace(
            model_aliases={"llama3-8b-instruct": "actual-a", "deepseek-coder-v2": "actual-b"},
            token_prices={"actual-a": (4.0, 9.0), "actual-b": (1.0, 7.0)},
        )
        orch = BufferedOrchestrator(num_ctrl=1, num_agents=2, llm_api_client=client)
        orch.submit_task(make_task(1))
        proposals = orch.buffer.tasks[1].proposals
        self.assertAlmostEqual(proposals[0].proposal.cost, 16 * 4.0 / 1_000_000)
        self.assertAlmostEqual(proposals[1].proposal.cost, 16 / 1_000_000)

    def test_agent_executes_tasks_sequentially(self):
        orch = BufferedOrchestrator(num_ctrl=2, num_agents=1, variant=variant())
        tasks = [make_task(i, arrival=0.0, urgency=0.2 + i * 0.1) for i in range(3)]

        orch.run(tasks)

        records = sorted(orch.task_records, key=lambda row: row["actual_start"])
        self.assertEqual(len(records), 3)
        for previous, current in zip(records, records[1:]):
            self.assertGreaterEqual(current["actual_start"], previous["finish_time"])
        self.assertEqual(len({row["parent_id"] for row in records}), 3)

    def test_assignment_invalidates_but_retains_alternatives(self):
        orch = BufferedOrchestrator(num_ctrl=1, num_agents=3, variant=variant())
        orch.run([make_task(10)])

        buffered = orch.buffer.tasks[10]
        self.assertIn(buffered.status, (TaskStatus.COMPLETED, TaskStatus.FAILED))
        statuses = [candidate.status for candidate in buffered.proposals.values()]
        self.assertEqual(statuses.count(CandidateStatus.ASSIGNED), 1)
        self.assertEqual(statuses.count(CandidateStatus.INVALIDATED), 2)

    def test_feedback_returns_only_to_responsible_controller(self):
        orch = BufferedOrchestrator(num_ctrl=2, num_agents=2, variant=variant())
        before = [controller.routing_weights.copy() for controller in orch.controllers]
        owner = orch.submit_task(make_task(1))
        orch.dispatch_ready()
        while orch.completion_events:
            event = orch.completion_events.pop(0)
            orch.time = event.finish_time
            orch._finish_event(event)

        for controller in orch.controllers:
            if controller.id == owner:
                self.assertFalse(np.allclose(controller.routing_weights, before[owner]))
            else:
                self.assertTrue(np.allclose(controller.routing_weights, before[controller.id]))
        self.assertEqual(orch.feedback_records[0]["controller_id"], owner)

    def test_load_balancing_uses_more_than_one_controller_for_batch(self):
        orch = BufferedOrchestrator(num_ctrl=3, num_agents=2, variant=variant())
        for task_id in range(6):
            orch.submit_task(make_task(task_id, arrival=0.0))

        owners = {task.controller_id for task in orch.buffer.tasks.values()}
        self.assertGreater(len(owners), 1)

    def test_aspsa_runs_after_feedback_warmup(self):
        orch = BufferedOrchestrator(
            num_ctrl=1,
            num_agents=2,
            variant=variant(
                use_spsa=True,
                spsa_interval=2,
                spsa_warmup_tasks=4,
            ),
        )
        orch.run([make_task(i, arrival=float(i) * 0.1) for i in range(6)])

        sync_events = [row for row in orch.event_records if row["event"] == "aspsa_sync"]
        self.assertGreaterEqual(len(sync_events), 1)
        self.assertTrue(all(len(controller.quality_obs_history) > 0 for controller in orch.controllers))

    def test_discrete_batches_and_service_ticks(self):
        orch = BufferedOrchestrator(
            num_ctrl=1,
            num_agents=2,
            variant=variant(),
            simulation_tick=1.0,
        )
        tasks = orch.generate_batched_stream(6, batch_size=3, tick_duration=1.0)

        self.assertEqual([task.t_arrival for task in tasks], [0, 0, 0, 1, 1, 1])
        orch.run(tasks)
        self.assertTrue(
            all(float(row["service_time"]).is_integer() for row in orch.task_records)
        )

    def test_api_error_is_terminal_failure_without_retry(self):
        class FailingClient:
            def call(self, *args, **kwargs):
                raise RuntimeError("endpoint unavailable")

        orch = BufferedOrchestrator(
            num_ctrl=1,
            num_agents=1,
            variant=variant(),
            llm_api_client=FailingClient(),
        )
        orch.run([make_task(99)])

        self.assertEqual(orch.buffer.tasks[99].status, TaskStatus.FAILED)
        self.assertEqual(len(orch.task_records), 1)
        self.assertEqual(orch.task_records[0]["success"], 0)


if __name__ == "__main__":
    unittest.main()
