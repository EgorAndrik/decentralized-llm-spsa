# Global-Buffer Adaptive Multi-LLM Orchestration with A-SPSA

This repository is an independent fork that implements an event-driven global
assignment buffer. Controllers evaluate task--agent pairs but never dispatch
tasks themselves. The global buffer owns the final one-task/one-agent matching
decision, refreshes time and availability terms, and dispatches work only to
free agents.

See [ARCHITECTURE.md](ARCHITECTURE.md) for the state machine, equations, and
component responsibilities, and [PAPER_EXPERIMENTS.md](PAPER_EXPERIMENTS.md)
for the experiment protocol.

*Anonymous submission*

## Abstract

Adaptive routing in multi-LLM systems is difficult to evaluate because policies can observe different feedback. This repository implements a replay protocol that fixes task streams and potential task-agent outcomes, an event-driven global assignment buffer, and distributed zeroth-order optimization of the three manuscript losses. The controlled optimizer benchmark uses exogenous balanced routing so every method receives identical feedback, applies one shared consensus mechanism and one shared step grid to every method, and reports prediction accuracy and controller disagreement as separate endpoints rather than only their weighted sum. Architecture experiments are reported separately using operational quality, deadline, waiting, throughput, cost, and fairness metrics. Real-LLM validation has not yet been run, and all agent capabilities are simulated.

## Repository Structure

```
src/                Python simulation and plotting code
  paper_experiment.py     controlled six-domain replay experiment
  paper_losses.py         manuscript losses F1/F2/F3
  paper_optimizers.py     the seven compared zeroth-order methods
  run_paper_experiments.py  paper protocol runner (tuning, search, ablation)
  it_benchmark.py         pinned Stack Overflow six-domain benchmark builder
  audit_it_benchmark.py   benchmark construction audit and quality gates
  assignment_buffer.py    global task--agent candidate buffer and matching
  buffered_orchestrator.py event-driven orchestration and delayed feedback
  run_buffered_simulation.py CLI for the global-buffer architecture
  main.py                 legacy system model, RLS, A-SPSA, judge module
  spsa_variants.py        legacy optimizer implementations (see caveat below)
  dataset_loader.py       legacy MBPP / TriviaQA / XSum / OPUS-100 loader
  llm_api.py              OpenAI-compatible API client
  llm_config.py           [NOT COMMITTED] API keys and model aliases
  llm_config_example.py   template for llm_config.py
tests/              pytest unit tests
data/               pinned routing manifest (versioned) and caches (ignored)
requirements.txt    Python dependencies
```

The manuscript LaTeX source is distributed as `Sc_Rep_26_A_SPSA_LLM.zip`. There
is no `paper/` directory in this fork.

### Two optimizer code paths

`src/paper_optimizers.py` is the implementation used by the manuscript
protocol. `src/spsa_variants.py` is the legacy ApPLIED-era implementation kept
for the older `run_spsa_comparison.py` and `run_real_llm_comparison.py`
entry points. The two differ in substance: the legacy module optimizes plain
MSE and a one-sided hinge rather than F1/F2/F3, draws perturbations from the
global NumPy RNG, and keeps its own copy of the parameters so RLS updates
between synchronization rounds are ignored. **Do not mix numbers from the two
paths.** Only `src.run_paper_experiments` produces paper evidence.

## Setup

```bash
pip install -r requirements.txt
```

### Connecting a provider (real-LLM mode only)

Agents and the judge share a LangChain client for APIs that implement OpenAI
Chat Completions. Configure the base URL (without `/chat/completions`), the
provider's model IDs and API credentials. No provider-specific Python SDK is
needed to add another compatible endpoint.

From the repository root, copy `.env.example` to `.env` and fill in
`API_KEY`. Set `LLM_MODELS` to a comma-separated list of executor
model IDs and `LLM_JUDGE_MODEL` to the judge's ID. All listed executors
enter the candidate pool by default. To use only the first N, pass
`--agents N`. Each executor starts with the same routing priors.
Set `ROUTING_CONTROLLERS` in `.env` to choose the controller count in
either mode (default: 3). The `--controllers` flag takes precedence.
Each task's assigned controller scores the eligible executors from the shared model pool.
Exported environment variables take precedence over `.env`, which Git ignores.

```bash
cp .env.example .env
# Fill in API_KEY in .env, then run:
python -m src.run_buffered_simulation --mode api \
       --output-dir buffered_outputs/smoke-sql \
       --prompt "In SQL, a column has values 10, NULL, and 20. What do COUNT(*) and COUNT(column) return?"
```

Expect `COUNT(*) = 3` and `COUNT(column) = 2`. Another check is to ask
`What does Python's sorted([3, 1, 3, 2]) return?` and expect `[1, 2, 3, 3]`.
Use a different `--output-dir` for each case to retain its reports.
The template caps each run at two outgoing requests, one for the worker
and one for the judge. Increase `LLM_MAX_CALLS` for multi-task runs.

In `.env`, all workers and the judge use `LLM_BASE_URL` and `API_KEY`.
For another OpenAI-compatible provider, change the endpoint, key and model IDs.
Add its trusted HTTPS hostname to
`LLM_ALLOWED_HOSTS` if needed. For per-model credentials, use
`LLM_API_KEY_<MODEL_ID>` with the model ID uppercased and punctuation
replaced with underscores, such as `LLM_API_KEY_OPENAI_GPT_OSS_20B`.
The template also shows optional timeout, token limit and pricing settings.

For setups with multiple endpoints, use a Python config via
`--config llm_config`. When `LLM_MODELS` is absent, the existing
`src/llm_config.py` is the default.

## Reproducing Results

### Controlled paper experiments

The protocol, losses, fairness constraints, and statistics are documented in
[PAPER_EXPERIMENTS.md](PAPER_EXPERIMENTS.md); the audited outcome is in
[PAPER_RESULTS_AUDIT.md](PAPER_RESULTS_AUDIT.md).

```bash
# fast protocol check (diagnostics only, marked NOT FOR PAPER)
python -m src.run_paper_experiments --profile pilot \
       --output-dir paper_experiment_outputs_corrected_pilot

# paper-scale simulated run
python -m src.run_paper_experiments --profile full \
       --output-dir paper_experiment_outputs_corrected_full
```

Every output directory carries `experiment_manifest.json` recording the commit
(`code_revision`), the selected optimizer parameters, the theory check, and the
protocol. Artifacts produced before the fairness corrections are not comparable
with these; see the audit for what changed and why.

### Rebuilding the benchmark

`data/it_routing_benchmark.jsonl` is versioned. To rebuild and re-audit it:

```bash
python -m src.audit_it_benchmark --per-skill 200 --seed 42 \
       --output data/it_routing_benchmark.jsonl \
       --audit-output data/it_routing_benchmark.audit.json
```

### Global-buffer architecture demo

```bash
python -m src.run_buffered_simulation --mode simulation --tasks 200 --controllers 3 --agents 5 \
       --seed 42 --output-dir buffered_outputs
```

Simulation is the default and requires no credentials. Use `--mode api` with
the provider config above to run the same scheduler with real model calls for
both agents and the judge. API mode defaults to one generated task. To send
your own task and print the chosen model's answer:

```bash
LLM_MAX_CALLS=10 python -m src.run_buffered_simulation --mode api --agents 2 \
       --prompt "Explain optimistic locking in two sentences."
```

Both modes write task, event, feedback, and summary tables. In API mode,
model answers go into `responses.jsonl`, keyed by task ID. Failed calls,
empty results and answers rejected by the judge give a nonzero exit code.
In API mode they also miss the deadline metric, and `--prompt` only prints
an accepted answer. The summary records `run_status` (`running`, `completed`,
`partial`, `failed` or `interrupted`). Reports are cleared at the start of a
new run. On Ctrl+C the runner saves collected results and exits with code 130.

API latency measures elapsed time from task submission to evaluation,
including the judge. The scheduler still uses a virtual clock and sends
API requests sequentially. Virtual times are labelled `scheduler_*` in
the API tables.

API cost estimates include the agent and judge. They use the token counts
returned by the provider and `LLM_TOKEN_PRICES` (USD per million input
and output tokens, keyed by provider model ID). When any call lacks
usage or a price, the total is left blank and the summary reports the
known subtotal and number of unpriced calls.
Before dispatch, the router uses the configured input rates for its cost
estimate. It omits the price comparison when a candidate's rate is unknown.

#### Optional Tavily web search

Set `TAVILY_API_KEY` in `.env` or in the environment to add a separate search worker in
API mode. Tasks tagged with `--web-search` use that worker. The controller
and global buffer still handle assignment, and the configured judge scores
the answer and source excerpts. The search worker is separate from the
LLM agents counted by `--agents`.

With the API configuration above in place:

```bash
LLM_MAX_CALLS=2 python -m src.run_buffered_simulation --mode api \
       --output-dir buffered_outputs/smoke-search \
       --web-search --prompt "Find the official Python documentation for asyncio.TaskGroup."
```

Check that the answer cites an official `docs.python.org` page and explains
that the context manager waits for its tasks before exiting.

Ordinary prompts and simulation send no requests to Tavily. A search
request without a key exits with a configuration error. Search calls share
the run's API budget with models and the judge. The report records Tavily
credits separately from LLM tokens and estimated USD costs. If you use a
knowledge database, search feedback is recorded as `WEB_SEARCH` and can
use the same database as your existing LLM pool.

See the [Tavily Search API documentation](https://docs.tavily.com/documentation/api-reference/endpoint/search)
for the request and credit usage fields.

#### Routing experience and MVP warm-up

Set `ROUTING_DATABASE_URL` to the connection string for a PostgreSQL database
and pass `--knowledge-db` to keep routing experience across runs. The database
must already exist and the role must be able to create the `routing_*` tables
in its schema. Runs without `--knowledge-db` do not connect to PostgreSQL.
Each controller records the count, outcome, judge score and execution time
by task type and agent. Every `--round-size` evaluated tasks publish a shared
snapshot across controllers. The router uses local estimates when available
and the shared snapshot for pairs it has not seen. API failures without a
judge score do not advance the round. Failure counts are saved separately
and reduce the affected agent's estimate. The database stores a fingerprint
for coverage checks. It stores no prompt text or answers.

For example, collect three rounds of five tasks using `prompts.txt` with
one distinct representative prompt per line:

```bash
export ROUTING_DATABASE_URL='postgresql://user:password@localhost/routing'
LLM_MAX_CALLS=30 python -m src.run_buffered_simulation --mode api --agents 2 \
       --knowledge-db \
       --round-size 5 --mvp-rounds 3 --warmup-file prompts.txt --tasks 15
```

After the summary shows `mvp_ready=True`, use the same database to route a
user prompt:

```bash
python -m src.run_buffered_simulation --mode api --agents 2 \
       --knowledge-db \
       --prompt "Explain optimistic locking in two sentences."
```

MVP readiness is published at a round boundary and also requires at least
two distinct evaluated prompts and an evaluation for every configured LLM
agent and controller. Search feedback
does not cover an LLM slot. The database keeps incomplete rounds, so another
batch can finish the warm-up. Use a separate database or PostgreSQL schema
for simulation or a different model pool.

To move existing routing experience from the old SQLite file, point
`ROUTING_DATABASE_URL` at an empty PostgreSQL destination and run:

```bash
python -m src.migrate_routing_knowledge buffered_outputs/routing_knowledge.sqlite3
```

The import preserves the pool configuration, published snapshot, failures
and warm-up progress present in the file.
Knowledge rounds publish routing statistics and are separate from the
existing A-SPSA parameter synchronization rounds.

### Legacy comparison scripts

These use the legacy loss and optimizer path described above and are retained
for continuity, not as paper evidence:

```bash
python -m src.run_spsa_comparison --tasks 500 --seeds 1,2,3,4,5 \
       --output-dir spsa_comparison
python -m src.run_real_llm_comparison --tasks 100 --seeds 42,43 \
       --output-dir real_llm_outputs
```

## Tests

```bash
python -m pytest -q
```

Set `TEST_ROUTING_DATABASE_URL` to a PostgreSQL test database to also run
storage and CLI integration tests. The test role must be able to create and
drop temporary schemas. CI provisions its own PostgreSQL instance.

`tests/test_protocol_fairness.py` pins the properties that make the optimizer
comparison a comparison of optimizers: one consensus operator, one perturbation
scale, one step grid, manuscript-faithful losses, an identifiable feature
matrix, and an honest report of the query budget and the statistical resolution
floor.

Tests that need an optional corpus skip with a reason when it is absent. They
never pass silently: when the corpus is present the assertions run in full.

## License

MIT
