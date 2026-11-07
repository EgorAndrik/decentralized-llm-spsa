import unittest

import numpy as np

from src.llm_api import BudgetExceeded
from src.main import ExactOrchestrator, SimulationVariant, Task, TaskType
from src.buffered_orchestrator import BufferedOrchestrator


class FailingClient:
    def __init__(self, exc):
        self.exc = exc
        self.calls = 0

    def call(self, model_name, prompt, task_type):
        self.calls += 1
        raise self.exc


class WorkingClient:
    def call(self, model_name, prompt, task_type):
        return {"output": "an answer", "success": True, "latency": 0.5}


def tasks(n=6):
    rng = np.random.RandomState(3)
    return [Task(i, TaskType.PROGRAMMING, float(i), 1.0, rng.rand(5), 0.5, f"Write function {i}.")
            for i in range(n)]


def variant():
    return SimulationVariant(name="api", use_judge=True)


class TestSimulationAfterApiFailure(unittest.TestCase):
    def run_orch(self, client):
        np.random.seed(5)
        orch = ExactOrchestrator(num_ctrl=2, num_agents=3, llm_api_client=client, variant=variant())
        orch.run(tasks())
        return orch, orch.collect_summary()

    def test_failed_calls_are_counted_and_logged_not_silent(self):
        client = FailingClient(RuntimeError("provider down"))
        with self.assertLogs("src.main", level="WARNING") as logs:
            orch, summary = self.run_orch(client)
        self.assertGreater(client.calls, 0)
        self.assertEqual(summary["api_simulated_after_failure"], float(client.calls))
        self.assertEqual(summary["api_simulated_rate"], 1.0)
        self.assertTrue(any("SIMULATED" in line for line in logs.output))

    def test_working_client_reports_zero_simulated(self):
        orch, summary = self.run_orch(WorkingClient())
        self.assertGreater(summary["api_calls"], 0)
        self.assertEqual(summary["api_simulated_after_failure"], 0.0)
        self.assertEqual(summary["api_simulated_rate"], 0.0)

    def test_no_client_reports_no_api_calls(self):
        np.random.seed(5)
        orch = ExactOrchestrator(num_ctrl=2, num_agents=3, variant=SimulationVariant(name="t", use_judge=False))
        orch.run(tasks())
        summary = orch.collect_summary()
        self.assertEqual(summary["api_calls"], 0.0)
        self.assertEqual(summary["api_simulated_rate"], 0.0)

    def test_budget_exceeded_stops_the_run(self):
        with self.assertRaises(BudgetExceeded):
            self.run_orch(FailingClient(BudgetExceeded("cap")))


class TestBufferedApiFailure(unittest.TestCase):
    def test_failure_is_counted_and_budget_propagates(self):
        np.random.seed(5)
        client = FailingClient(RuntimeError("provider down"))
        orch = BufferedOrchestrator(num_ctrl=2, num_agents=3, llm_api_client=client, variant=variant())
        with self.assertLogs("src.buffered_orchestrator", level="WARNING"):
            orch.run(tasks(4))
        self.assertGreater(orch.api_stats["failed"], 0)
        self.assertEqual(orch.collect_summary()["api_failed_calls"], float(orch.api_stats["failed"]))

        np.random.seed(5)
        orch = BufferedOrchestrator(num_ctrl=2, num_agents=3, llm_api_client=FailingClient(BudgetExceeded("cap")),
                                    variant=variant())
        with self.assertRaises(BudgetExceeded):
            orch.run(tasks(4))


if __name__ == "__main__":
    unittest.main()
