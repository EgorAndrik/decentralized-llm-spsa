import csv
import json
import sys
import time

import httpx
import pytest

from src import llm_api, llm_config_example, run_buffered_simulation


def read_rows(path):
    with path.open() as file:
        return list(csv.DictReader(file))


def test_default_mode_does_not_load_api_config(tmp_path, monkeypatch):
    def unexpected_api_config(*args):
        pytest.fail("Simulation should not load API configuration")

    monkeypatch.setattr(run_buffered_simulation, "import_config_module", unexpected_api_config)
    monkeypatch.setattr(sys, "argv", ["runner", "--tasks", "2", "--output-dir", str(tmp_path)])

    run_buffered_simulation.main()

    summary, = read_rows(tmp_path / "summary.csv")
    assert float(summary["api_calls"]) == 0.0
    assert not (tmp_path / "responses.jsonl").exists()


def configure_api_run(tmp_path, monkeypatch, args, *, judge_output=None, worker_fails=False):
    calls = []

    def respond(request):
        assert str(request.url) == llm_config_example.BASE_URL + "/chat/completions"
        body = json.loads(request.content)
        prompt = body["messages"][-1]["content"]
        is_judge = prompt.startswith("You are an LLM judge.")
        calls.append((body["model"], prompt, is_judge))
        if worker_fails and not is_judge:
            return httpx.Response(503, json={"error": {"message": "unavailable"}})
        if is_judge:
            time.sleep(0.01)
        content = judge_output if is_judge else "=1+1"
        return httpx.Response(200, json={
            "id": "completion", "object": "chat.completion", "created": 0, "model": body["model"],
            "choices": [{
                "index": 0, "finish_reason": "stop", "message": {"role": "assistant", "content": content},
            }],
            "usage": {"prompt_tokens": 100, "completion_tokens": 200, "total_tokens": 300},
        })

    def http_client(**kwargs):
        return httpx.Client(transport=httpx.MockTransport(respond), **kwargs)

    monkeypatch.setenv("GROQ_API_KEY", "test-key")
    monkeypatch.setenv("LLM_MAX_CALLS", "10")
    monkeypatch.setattr(run_buffered_simulation, "import_config_module", lambda *args: llm_config_example)
    monkeypatch.setattr(llm_api, "HTTPClient", http_client)
    monkeypatch.setattr(sys, "argv", [
        "runner", "--mode", "api", "--agents", "2", "--output-dir", str(tmp_path), *args,
    ])
    return calls


def test_api_prompt_returns_answer_and_exports_usage(tmp_path, monkeypatch, capsys):
    prompt = "Explain optimistic locking."
    calls = configure_api_run(
        tmp_path, monkeypatch, ["--prompt", prompt], judge_output='{"quality_score": 0.95, "verdict": "pass"}'
    )

    run_buffered_simulation.main()

    worker, judge = calls
    assert worker[1] == prompt
    assert judge[2] and "=1+1" in judge[1]
    assert capsys.readouterr().out.strip() == "=1+1"
    task, = read_rows(tmp_path / "task_records.csv")
    assert "model_output" not in task
    response = json.loads((tmp_path / "responses.jsonl").read_text())
    assert response["model_output"] == "=1+1"
    assert response["task_id"] == int(task["parent_id"])
    assert task["model_name"] == worker[0]
    assert task["judge_model_name"] == judge[0]
    assert int(task["input_tokens"]) == int(task["judge_input_tokens"]) == 100
    assert int(task["output_tokens"]) == int(task["judge_output_tokens"]) == 200
    assert float(task["judge_latency"]) >= 0.01
    assert float(task["latency"]) >= float(task["judge_latency"]) + float(task["service_time"])
    worker_rates = llm_config_example.LLM_TOKEN_PRICES[worker[0]]
    judge_rates = llm_config_example.LLM_TOKEN_PRICES[judge[0]]
    expected_cost = sum(100 * rates[0] + 200 * rates[1] for rates in (worker_rates, judge_rates)) / 1e6
    summary, = read_rows(tmp_path / "summary.csv")
    assert float(summary["total_cost"]) == pytest.approx(expected_cost)


def test_api_batch_reports_sequential_calls_separately_from_scheduler(tmp_path, monkeypatch):
    configure_api_run(tmp_path, monkeypatch, ["--tasks", "2"], judge_output='{"quality_score": 0.95}')

    run_buffered_simulation.main()

    first, second = sorted(read_rows(tmp_path / "task_records.csv"), key=lambda row: float(row["actual_start"]))
    assert first["scheduler_actual_start"] == second["scheduler_actual_start"] == "0.0"
    assert float(second["actual_start"]) >= float(first["worker_finished_at"])


def test_judge_failure_replaces_stale_reports_and_keeps_paid_answer(tmp_path, monkeypatch, capsys):
    (tmp_path / "summary.csv").write_text("mode,run_status\nsimulation,completed\n")
    (tmp_path / "task_records.csv").write_text("old_record\n1\n")
    configure_api_run(tmp_path, monkeypatch, ["--prompt", "Explain SQL."], judge_output="invalid")

    with pytest.raises(SystemExit) as error:
        run_buffered_simulation.main()

    assert error.value.code == 1
    assert "API judge failed (parse_failed)" in capsys.readouterr().err
    summary, = read_rows(tmp_path / "summary.csv")
    assert summary["mode"] == "api"
    assert summary["run_status"] == "failed"
    assert float(summary["total_cost"]) > 0
    assert not read_rows(tmp_path / "task_records.csv")
    assert json.loads((tmp_path / "responses.jsonl").read_text())["model_output"] == "=1+1"


def test_failed_worker_calls_miss_deadlines_and_return_error(tmp_path, monkeypatch):
    calls = configure_api_run(tmp_path, monkeypatch, ["--tasks", "2"], worker_fails=True)

    with pytest.raises(SystemExit) as error:
        run_buffered_simulation.main()

    assert error.value.code == 1
    assert len(calls) == 2 and not any(call[2] for call in calls)
    summary, = read_rows(tmp_path / "summary.csv")
    assert summary["run_status"] == "failed"
    assert float(summary["api_failed_calls"]) == 2.0
    assert float(summary["deadline_hit_rate"]) == 0.0
    assert float(summary["scheduler_deadline_hit_rate"]) == 0.0
    assert int(summary["unpriced_api_calls"]) == 2
    assert summary["total_cost"] == ""
    assert not (tmp_path / "responses.jsonl").read_text()
