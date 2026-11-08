import unittest
import numpy as np
from src.main import (
    AdaptiveController,
    AgentState,
    ExactOrchestrator,
    SubTask,
    SimulationVariant,
    Task,
    TaskType,
    rank_agents,
    spsa_consensus,
)

class TestOrchestrator(unittest.TestCase):
    def test_tool_use_has_its_own_time_feature(self):
        controller = AdaptiveController(0)
        features = np.zeros(5)
        before = controller.predict_time(TaskType.TOOL_USE.value, features)
        other = controller.predict_time(TaskType.QA.value, features)
        controller.rls_update(TaskType.TOOL_USE.value, 5.0, 3.0, features)
        self.assertGreater(controller.predict_time(TaskType.TOOL_USE.value, features), before)
        self.assertEqual(controller.predict_time(TaskType.QA.value, features), other)
    def test_generate_stream(self):
        orch = ExactOrchestrator(num_ctrl=2, num_agents=3)
        tasks = orch.generate_realistic_stream(10)
        self.assertEqual(len(tasks), 10)
        self.assertTrue(all(hasattr(t, 'type') for t in tasks))

    def test_run_simulation(self):
        orch = ExactOrchestrator(num_ctrl=2, num_agents=3)
        tasks = orch.generate_realistic_stream(20)
        orch.run(tasks)
        self.assertGreaterEqual(orch.metrics['assignments'], 0)
        self.assertTrue(len(orch.metrics['success_rate']) > 0)

    def test_queues_are_released_after_run(self):
        orch = ExactOrchestrator(num_ctrl=2, num_agents=3)
        tasks = orch.generate_realistic_stream(15)
        orch.run(tasks)
        self.assertTrue(all(len(ctrl.task_queue) == 0 for ctrl in orch.controllers))
        self.assertTrue(all(len(orch.controller_dispatch_tasks[ctrl.id]) == 0 for ctrl in orch.controllers))
        self.assertTrue(all(len(agent.queue_pred_times) == 0 for agent in orch.agents))
        self.assertTrue(all((agent.current_task is None) for agent in orch.agents))

    def test_if_best_busy_choose_next_available_agent(self):
        orch = ExactOrchestrator(num_ctrl=1, num_agents=2)
        busy_task = Task(999, TaskType.PROGRAMMING, 0.0, 2.0, np.array([0.9, 0.3, 0.2, 0.2, 0.2]), 0.2, "busy")
        orch.agents[1].queue.append(busy_task)
        orch.agents[1].queue_start_times.append(0.0)
        orch.agents[1].queue_finish_times.append(100.0)
        orch.agents[1].queue_pred_times.append(100.0)
        orch.agents[1].available_at = 100.0
        orch._refresh_agent_runtime(orch.agents[1], 1.0)

        task = Task(
            1,
            TaskType.PROGRAMMING,
            1.0,
            1.5,
            np.array([0.9, 0.3, 0.2, 0.2, 0.2]),
            0.1,
            "Build simple code utility.",
        )
        orch.run([task])
        assigned = [r for r in orch.task_records if r["assigned"] == 1]
        self.assertTrue(len(assigned) > 0)
        self.assertNotEqual(assigned[0]["agent_id"], 1)

    def test_if_top3_unavailable_choose_best_remaining_model(self):
        orch = ExactOrchestrator(num_ctrl=1, num_agents=4)
        busy_task = Task(999, TaskType.PROGRAMMING, 0.0, 2.0, np.array([0.9, 0.3, 0.2, 0.2, 0.2]), 0.2, "busy")
        for agent_id in [0, 1, 2]:
            orch.agents[agent_id].queue.append(busy_task)
            orch.agents[agent_id].queue_start_times.append(0.0)
            orch.agents[agent_id].queue_finish_times.append(100.0 + agent_id)
            orch.agents[agent_id].queue_pred_times.append(100.0 + agent_id)
            orch.agents[agent_id].available_at = 100.0 + agent_id
            orch._refresh_agent_runtime(orch.agents[agent_id], 1.0)

        task = Task(
            1,
            TaskType.PROGRAMMING,
            1.0,
            1.5,
            np.array([0.9, 0.3, 0.2, 0.2, 0.2]),
            0.1,
            "Build simple code utility.",
        )
        orch.run([task])
        assigned = [r for r in orch.task_records if r["assigned"] == 1]
        self.assertTrue(len(assigned) > 0)
        self.assertEqual(assigned[0]["agent_id"], 3)

    def test_guaranteed_fallback_assigns_when_adaptive_filters_all_models(self):
        orch = ExactOrchestrator(num_ctrl=1, num_agents=2)
        orch.judge.dynamic_success[:, TaskType.PROGRAMMING.value] = 0.15
        task = Task(
            1,
            TaskType.PROGRAMMING,
            0.0,
            2.0,
            np.array([0.9, 0.3, 0.2, 0.2, 0.2]),
            0.95,
            "Urgent production programming task.",
        )
        orch.run([task])
        assigned = [r for r in orch.task_records if r["assigned"] == 1]
        self.assertTrue(len(assigned) > 0)

    def test_controller_selection_prefers_lower_predicted_finish(self):
        orch = ExactOrchestrator(num_ctrl=2, num_agents=2)
        orch.time = 0.0
        orch.controllers[0].theta_time = np.array([5.5, 5.5, 5.5, 5.5, 5.5], dtype=float)
        orch.controllers[0].theta_wait = np.array([2.5, 2.5, 2.5, 2.5, 2.5], dtype=float)
        orch.controllers[1].theta_time = np.array([1.2, 1.2, 1.2, 1.2, 1.2], dtype=float)
        orch.controllers[1].theta_wait = np.array([0.4, 0.4, 0.4, 0.4, 0.4], dtype=float)
        subtask = SubTask(
            parent_id=1,
            local_id=0,
            type=TaskType.PROGRAMMING,
            h_Ti=2.0,
            phi_i=np.array([0.9, 0.3, 0.2, 0.2, 0.2]),
            urgency=0.6,
            text="Build simple code utility.",
        )
        selected_ctrl = orch._select_controller(subtask)
        self.assertEqual(selected_ctrl, 1)

    def test_busy_controller_dispatch_is_penalized(self):
        orch = ExactOrchestrator(num_ctrl=2, num_agents=2)
        orch.time = 0.0
        busy = Task(999, TaskType.PROGRAMMING, 0.0, 2.0, np.array([0.9, 0.3, 0.2, 0.2, 0.2]), 0.4, "busy controller")
        orch.controller_dispatch_tasks[0].append(busy)
        orch.controller_dispatch_finish_times[0].append(25.0)
        subtask = SubTask(
            parent_id=1,
            local_id=0,
            type=TaskType.PROGRAMMING,
            h_Ti=2.0,
            phi_i=np.array([0.9, 0.3, 0.2, 0.2, 0.2]),
            urgency=0.5,
            text="Build simple code utility.",
        )
        snapshot = orch._build_coordination_snapshot(subtask)
        entry0 = next(entry for entry in snapshot["controllers"] if entry["controller_id"] == 0)
        entry1 = next(entry for entry in snapshot["controllers"] if entry["controller_id"] == 1)
        self.assertGreater(entry0["dispatch_wait"], 0.0)
        self.assertEqual(entry1["dispatch_wait"], 0.0)
        self.assertGreater(entry0["predicted_budget"], entry1["predicted_budget"])
        self.assertEqual(orch._select_controller(subtask, coordination_snapshot=snapshot), 1)

    def test_coordination_snapshot_prioritizes_best_global_candidate(self):
        orch = ExactOrchestrator(num_ctrl=2, num_agents=2)
        orch.time = 0.0
        orch.controllers[0].theta_time = np.array([5.5, 5.5, 5.5, 5.5, 5.5], dtype=float)
        orch.controllers[0].theta_wait = np.array([2.0, 2.0, 2.0, 2.0, 2.0], dtype=float)
        orch.controllers[1].theta_time = np.array([1.0, 1.0, 1.0, 1.0, 1.0], dtype=float)
        orch.controllers[1].theta_wait = np.array([0.3, 0.3, 0.3, 0.3, 0.3], dtype=float)
        subtask = SubTask(
            parent_id=2,
            local_id=0,
            type=TaskType.TOOL_USE,
            h_Ti=2.0,
            phi_i=np.array([0.3, 0.4, 0.5, 0.4, 0.95]),
            urgency=0.9,
            text="Urgent tool-use task.",
        )
        snapshot = orch._build_coordination_snapshot(subtask)
        self.assertTrue(len(snapshot["sketch_candidates"]) > 0)
        self.assertEqual(snapshot["sketch_candidates"][0]["controller_id"], 1)

    def test_numeric_mode_tracks_wait_and_service_time_separately(self):
        orch = ExactOrchestrator(num_ctrl=1, num_agents=1)
        first = Task(1, TaskType.PROGRAMMING, 0.0, 2.0, np.array([0.9, 0.3, 0.2, 0.2, 0.2]), 0.1, "Build utility")
        second = Task(2, TaskType.PROGRAMMING, 0.1, 2.0, np.array([0.9, 0.3, 0.2, 0.2, 0.2]), 0.8, "Refactor production pipeline")
        orch.run([first, second])

        assigned = [r for r in orch.task_records if r["assigned"] == 1]
        self.assertGreaterEqual(len(assigned), 2)
        self.assertTrue(all(r["latency"] >= r["service_time"] for r in assigned))
        self.assertTrue(all(r["finish_time"] >= r["actual_start"] for r in assigned))
        self.assertTrue(all("semantic_match" in r for r in assigned))
        self.assertTrue(any(r["parent_id"] == 2 and r["true_wait"] > 0.0 for r in assigned))
        self.assertTrue(all("routing_score" in r for r in assigned))
        self.assertTrue(all("routing_utility" in r for r in assigned))

    def test_controller_delay_is_included_in_latency(self):
        orch = ExactOrchestrator(num_ctrl=1, num_agents=1)
        busy = Task(999, TaskType.PROGRAMMING, 0.0, 2.0, np.array([0.9, 0.3, 0.2, 0.2, 0.2]), 0.4, "busy controller")
        orch.controller_dispatch_tasks[0].append(busy)
        orch.controller_dispatch_finish_times[0].append(5.0)
        task = Task(1, TaskType.PROGRAMMING, 0.0, 2.0, np.array([0.9, 0.3, 0.2, 0.2, 0.2]), 0.7, "Controller bottleneck test")
        orch.run([task])
        assigned = [r for r in orch.task_records if r["assigned"] == 1 and r["parent_id"] == 1]
        self.assertGreaterEqual(len(assigned), 1)
        record = assigned[0]
        self.assertGreater(record["controller_wait"], 0.0)
        self.assertGreater(record["latency"], record["service_time"])
        self.assertGreater(record["true_wait"], 0.0)

    def test_spsa_updates_theta_but_not_routing_weights(self):
        orch = ExactOrchestrator(num_ctrl=2, num_agents=2)
        for idx, ctrl in enumerate(orch.controllers):
            ctrl.theta_time = np.array([2.0, 2.1, 2.2, 2.3, 2.4], dtype=float) + idx * 0.05
            ctrl.theta_wait = np.array([1.2, 1.3, 1.4, 1.5, 1.6], dtype=float) + idx * 0.03
            ctrl.routing_weights = np.array([0.30, 0.15, 0.15, 0.10, 0.08, 0.10, 0.07, 0.05], dtype=float)
            for j in range(14):
                features = np.array([0.8, 0.3, 0.2, 0.2, 0.2], dtype=float)
                ctrl.obs_history.append(type("Obs", (), {
                    "task_type": TaskType.PROGRAMMING.value,
                    "true_time": 2.0 + 0.05 * j + 0.02 * idx,
                    "true_wait": 2.5 + 0.10 * j + 0.05 * idx,
                    "features": features,
                })())

        theta_before = [(ctrl.theta_time.copy(), ctrl.theta_wait.copy()) for ctrl in orch.controllers]
        weights_before = [ctrl.routing_weights.copy() for ctrl in orch.controllers]

        spsa_consensus(orch.controllers, step=1)

        for ctrl, (theta_time_before, theta_wait_before) in zip(orch.controllers, theta_before):
            self.assertFalse(np.allclose(ctrl.theta_time, theta_time_before))
            self.assertFalse(np.allclose(ctrl.theta_wait, theta_wait_before))
        self.assertTrue(all(np.allclose(ctrl.routing_weights, before) for ctrl, before in zip(orch.controllers, weights_before)))

    def test_predicted_budget_changes_adaptive_ranking(self):
        controller = AdaptiveController(0)
        controller.routing_weights = np.array([0.05, 0.05, 0.05, 0.40, 0.05, 0.01, 0.05, 0.34], dtype=float)
        variant = SimulationVariant(name="adaptive_test", routing_mode="adaptive", cost_weight=0.0)
        agents = [
            AgentState(0, np.array([0.82, 0.88, 0.68, 0.58, 0.50]), []),
            AgentState(1, np.array([0.92, 0.40, 0.30, 0.20, 0.10]), []),
        ]
        agents[0].available_at = 0.0
        agents[1].available_at = 1.4
        task = Task(
            1,
            TaskType.PROGRAMMING,
            0.0,
            2.5,
            np.array([0.92, 0.32, 0.22, 0.18, 0.12]),
            0.95,
            "Urgent production coding task with strict deadline.",
        )
        dynamic_success = np.full((5, 5), 0.75, dtype=float)

        relaxed = rank_agents(
            controller=controller,
            task=task,
            agents=agents,
            time=0.0,
            dynamic_success=dynamic_success,
            predicted_time=5.5,
            predicted_wait=3.5,
            top_k=2,
            variant=variant,
        )
        tight = rank_agents(
            controller=controller,
            task=task,
            agents=agents,
            time=0.0,
            dynamic_success=dynamic_success,
            predicted_time=1.2,
            predicted_wait=0.4,
            top_k=2,
            variant=variant,
        )

        self.assertEqual(relaxed[0]["agent_id"], 1)
        self.assertEqual(tight[0]["agent_id"], 0)

if __name__ == "__main__":
    unittest.main()
