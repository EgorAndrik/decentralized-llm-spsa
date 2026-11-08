import csv
import sys

import pytest

from src import llm_config_example, run_buffered_simulation


def test_default_mode_does_not_load_api_config(tmp_path, monkeypatch):
    def unexpected_api_config(*args):
        pytest.fail("Simulation should not load API configuration")

    monkeypatch.setattr(run_buffered_simulation, "import_config_module", unexpected_api_config)
    monkeypatch.setattr(sys, "argv", ["runner", "--tasks", "2", "--output-dir", str(tmp_path)])

    run_buffered_simulation.main()

    with (tmp_path / "summary.csv").open() as file:
        assert float(next(csv.DictReader(file))["api_calls"]) == 0.0


@pytest.mark.parametrize("judge_output", ['{"quality_score": 0.95, "verdict": "pass"}', "invalid"])
def test_api_mode_routes_prompt_and_calls_judge(tmp_path, monkeypatch, capsys, judge_output):
    calls = []
    prompt = "Explain optimistic locking."
    answer = "Check the row version before updating."

    class StubClient:
        def __init__(self, *args, **kwargs):
            pass

        def call(self, model_name, text, task_type):
            calls.append((model_name, text, task_type))
            output = judge_output if task_type == "JUDGE" else answer
            return {"output": output, "success": True, "latency": 0.3}

    monkeypatch.setenv("GROQ_API_KEY", "test-key")
    monkeypatch.setattr(run_buffered_simulation, "import_config_module", lambda *args: llm_config_example)
    monkeypatch.setattr(run_buffered_simulation, "LLMAPIClient", StubClient)
    monkeypatch.setattr(sys, "argv", [
        "runner", "--mode", "api", "--agents", "2", "--prompt", prompt,
        "--output-dir", str(tmp_path),
    ])

    if judge_output == "invalid":
        with pytest.raises(SystemExit) as error:
            run_buffered_simulation.main()
        assert error.value.code == 1
        assert "API judge failed (parse_failed)" in capsys.readouterr().err
        assert not (tmp_path / "summary.csv").exists()
    else:
        run_buffered_simulation.main()
        assert capsys.readouterr().out.strip() == answer
        with (tmp_path / "task_records.csv").open() as file:
            assert next(csv.DictReader(file))["model_output"] == answer

    worker_call, judge_call = calls
    assert worker_call[0] in ("llama3-8b-instruct", "deepseek-coder-v2")
    assert worker_call[1] == prompt
    assert judge_call[0] == llm_config_example.JUDGE_MODEL_NAME
    assert answer in judge_call[1]
    assert judge_call[2] == "JUDGE"
