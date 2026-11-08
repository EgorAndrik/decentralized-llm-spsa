import csv
import json
import sys
import time

import httpx
import pytest

from src import llm_api, llm_config_example, run_buffered_simulation, tavily_search


def read_rows(path):
    with path.open() as file:
        return list(csv.DictReader(file))


def test_default_mode_does_not_load_api_config(tmp_path, monkeypatch):
    def unexpected_api_config(*args):
        pytest.fail("Simulation should not load API configuration")

    monkeypatch.setattr(run_buffered_simulation, "import_config_module", unexpected_api_config)
    monkeypatch.setenv("TAVILY_API_KEY", "unused-key")
    monkeypatch.setattr(tavily_search, "HTTPClient", unexpected_api_config)
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
    monkeypatch.setenv("TAVILY_API_KEY", "unused-key")

    def unexpected_search(**kwargs):
        pytest.fail("Ordinary prompts should not contact Tavily")

    monkeypatch.setattr(tavily_search, "HTTPClient", unexpected_search)

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


def test_rejected_answer_is_saved_but_not_returned_as_success(tmp_path, monkeypatch, capsys):
    configure_api_run(
        tmp_path, monkeypatch, ["--prompt", "Explain SQL."],
        judge_output='{"quality_score": 0.95, "verdict": "fail"}',
    )
    with pytest.raises(SystemExit) as error:
        run_buffered_simulation.main()
    assert error.value.code == 1
    assert not capsys.readouterr().out
    row, = read_rows(tmp_path / "task_records.csv")
    assert row["success"] == row["deadline_hit"] == "0"
    assert row["judge_verdict"] == "fail"
    assert row["execution_failed"] == "False"
    assert read_rows(tmp_path / "summary.csv")[0]["run_status"] == "failed"


def test_interrupt_clears_previous_run_responses(tmp_path, monkeypatch):
    (tmp_path / "responses.jsonl").write_text('{"model_output": "old answer"}\n')
    configure_api_run(tmp_path, monkeypatch, ["--prompt", "Explain SQL."])

    def interrupt(**kwargs):
        raise KeyboardInterrupt

    monkeypatch.setattr(llm_api, "HTTPClient", interrupt)
    with pytest.raises(SystemExit) as error:
        run_buffered_simulation.main()
    assert error.value.code == 130
    assert read_rows(tmp_path / "summary.csv")[0]["run_status"] == "interrupted"
    assert not (tmp_path / "responses.jsonl").read_text()


def test_failed_worker_calls_miss_deadlines_and_return_error(tmp_path, monkeypatch, knowledge_dsn):
    monkeypatch.setenv("ROUTING_DATABASE_URL", knowledge_dsn)
    calls = configure_api_run(
        tmp_path, monkeypatch, ["--tasks", "2", "--knowledge-db"], worker_fails=True,
    )

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
    assert summary["knowledge_observations"] == "0"
    assert summary["knowledge_failures"] == "2"
    assert not (tmp_path / "responses.jsonl").read_text()


def test_mvp_prompt_requires_completed_rounds_from_persistent_knowledge(tmp_path, monkeypatch, capsys, knowledge_dsn):
    monkeypatch.setenv("ROUTING_DATABASE_URL", knowledge_dsn)
    db_args = ["--knowledge-db", "--controllers", "2"]
    warmup_file = tmp_path / "prompts.txt"
    warmup_file.write_text("Explain SQL joins.\n")
    warmup_args = [*db_args, "--warmup-file", str(warmup_file)]
    verdict = '{"quality_score": 0.95}'
    configure_api_run(
        tmp_path, monkeypatch, [*warmup_args, "--round-size", "2", "--mvp-rounds", "1"],
        judge_output=verdict,
    )
    run_buffered_simulation.main()
    assert read_rows(tmp_path / "summary.csv")[0]["mvp_ready"] == "False"

    calls = configure_api_run(
        tmp_path, monkeypatch, [*db_args, "--prompt", "Explain SQL."], judge_output=verdict,
    )
    with pytest.raises(SystemExit) as error:
        run_buffered_simulation.main()
    assert error.value.code == 2
    assert "0/1 rounds" in capsys.readouterr().err
    assert not calls

    warmup_file.write_text("Translate to Spanish: hello.\n")
    configure_api_run(tmp_path, monkeypatch, warmup_args, judge_output=verdict)
    run_buffered_simulation.main()
    summary, = read_rows(tmp_path / "summary.csv")
    assert summary["mvp_ready"] == "True"
    assert summary["knowledge_rounds"] == "1"
    assert any(row["event"] == "knowledge_round" for row in read_rows(tmp_path / "event_records.csv"))
    capsys.readouterr()

    calls = configure_api_run(
        tmp_path, monkeypatch, [*db_args, "--prompt", "Explain SQL."], judge_output=verdict,
    )
    run_buffered_simulation.main()
    assert capsys.readouterr().out.strip() == "=1+1"
    assert len(calls) == 2


def test_web_search_requires_key_before_contacting_providers(tmp_path, monkeypatch, capsys):
    calls = configure_api_run(tmp_path, monkeypatch, ["--web-search", "--prompt", "Find Python docs."])
    monkeypatch.delenv("TAVILY_API_KEY", raising=False)

    with pytest.raises(SystemExit) as error:
        run_buffered_simulation.main()

    assert error.value.code == 2
    assert "TAVILY_API_KEY" in capsys.readouterr().err
    assert not calls


@pytest.mark.parametrize("max_calls", [1, 2])
def test_tavily_routes_search_via_buffer_and_shares_api_budget(tmp_path, monkeypatch, capsys, max_calls, knowledge_dsn):
    prompt = "Find the official Python TaskGroup docs."
    monkeypatch.setenv("ROUTING_DATABASE_URL", knowledge_dsn)
    db_args = ["--knowledge-db"]
    verdict = '{"quality_score": 0.95, "verdict": "pass"}'
    monkeypatch.delenv("TAVILY_API_KEY", raising=False)
    configure_api_run(
        tmp_path, monkeypatch, [*db_args, "--tasks", "5", "--round-size", "5", "--mvp-rounds", "1"],
        judge_output=verdict,
    )
    run_buffered_simulation.main()
    capsys.readouterr()
    llm_calls = configure_api_run(
        tmp_path, monkeypatch, [*db_args, "--web-search", "--prompt", prompt], judge_output=verdict,
    )
    monkeypatch.setenv("TAVILY_API_KEY", "search-key")
    monkeypatch.setenv("LLM_MAX_CALLS", str(max_calls))
    search_calls = []
    source_url = "https://docs.python.org/3/library/asyncio-task.html#task-groups"

    def respond(request):
        search_calls.append(request)
        assert str(request.url) == "https://api.tavily.com/search"
        assert request.headers["Authorization"] == "Bearer search-key"
        payload = json.loads(request.content)
        assert payload["query"] == prompt and payload["include_usage"]
        return httpx.Response(200, json={
            "answer": "TaskGroup waits for all of its tasks.",
            "results": [
                {"title": "Python docs", "url": source_url, "content": "An asynchronous context manager."},
                {"title": "Invalid URL", "url": "javascript:alert(1)"},
            ],
            "usage": {"credits": 1},
        })

    monkeypatch.setattr(
        tavily_search, "HTTPClient", lambda **kwargs: httpx.Client(transport=httpx.MockTransport(respond), **kwargs),
    )
    if max_calls == 1:
        with pytest.raises(SystemExit) as error:
            run_buffered_simulation.main()
        assert error.value.code == 1
        assert "budget" in capsys.readouterr().err
        assert not llm_calls
    else:
        run_buffered_simulation.main()
        assert source_url in capsys.readouterr().out
        judge_call, = llm_calls
        assert judge_call[2] and source_url in judge_call[1]
        task, = read_rows(tmp_path / "task_records.csv")
        assert task["agent_id"] == "2"
        assert task["model_name"] == "tavily-search"
        assert task["task_type"] == "WEB_SEARCH"
        assert task["tavily_credits"] == "1"
        assert any(row["event"] == "assigned" for row in read_rows(tmp_path / "event_records.csv"))

    assert len(search_calls) == 1
    summary, = read_rows(tmp_path / "summary.csv")
    assert summary["tavily_calls"] == summary["tavily_known_credits"] == "1"
    assert summary["knowledge_observations"] == str(4 + max_calls)
    response = json.loads((tmp_path / "responses.jsonl").read_text())
    assert response["model_name"] == "tavily-search"
    assert source_url in response["model_output"]
    assert "javascript:" not in response["model_output"]


def test_tavily_redirect_is_a_failed_assignment_and_does_not_call_judge(tmp_path, monkeypatch):
    llm_calls = configure_api_run(tmp_path, monkeypatch, ["--web-search", "--prompt", "Find Python docs."])
    monkeypatch.setenv("TAVILY_API_KEY", "search-key")
    search_urls = []

    def respond(request):
        search_urls.append(str(request.url))
        return httpx.Response(302, headers={"Location": "https://unexpected.example/"})

    monkeypatch.setattr(
        tavily_search, "HTTPClient", lambda **kwargs: httpx.Client(transport=httpx.MockTransport(respond), **kwargs),
    )
    with pytest.raises(SystemExit) as error:
        run_buffered_simulation.main()

    assert error.value.code == 1
    assert search_urls == ["https://api.tavily.com/search"]
    assert not llm_calls
    task, = read_rows(tmp_path / "task_records.csv")
    assert task["execution_failed"] == "True"
    assert task["deadline_hit"] == "0"


def test_tavily_no_sources_does_not_count_as_api_error_or_accepted_answer(tmp_path, monkeypatch):
    llm_calls = configure_api_run(tmp_path, monkeypatch, ["--web-search", "--prompt", "Find Python docs."])
    monkeypatch.setenv("TAVILY_API_KEY", "search-key")
    monkeypatch.setattr(tavily_search, "HTTPClient", lambda **kwargs: httpx.Client(
        transport=httpx.MockTransport(lambda req: httpx.Response(200, json={"results": [], "usage": {"credits": 1}})),
        **kwargs,
    ))
    with pytest.raises(SystemExit) as error:
        run_buffered_simulation.main()
    assert error.value.code == 1
    assert not llm_calls
    task, = read_rows(tmp_path / "task_records.csv")
    assert task["execution_failed"] == task["result_available"] == "False"
    assert task["success"] == task["deadline_hit"] == "0"
