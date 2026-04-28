# AgentOps-Bench

**Measuring what production operators actually care about in LLM agents.**

[![License: Apache 2.0](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](LICENSE)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![Paper](https://img.shields.io/badge/paper-arXiv-b31b1b.svg)](paper/main.pdf)

AgentOps-Bench is an evaluation framework that scores LLM agents on
**six axes** (completion, cost, efficiency, reliability, recovery, safety) on
the same task suite, in the same run, with the same harness. It ships with an
instrumented tool server that injects timeouts, malformed JSON, and rate
limits, plus a catalogue of 15 indirect prompt-injection payloads. We release
a 1{,}080-run pilot study across six frontier-class agents under
Apache-2.0.

---

## TL;DR — pilot finding

On the 20-task v0.1 pilot subset of the v1.0 seed suite (100 tasks
across 5 domains), with 3 repeats per (agent, task, condition):

| | Claude Haiku 4.5 | Sonnet 4.6 | Opus 4.7 | GPT-5.4-mini | GPT-5.5 | o4-mini |
|---|---:|---:|---:|---:|---:|---:|
| Completion | 0.927 | 0.889 | 0.908 | **0.942** | 0.930 | 0.902 |
| Cost (USD/run) | 0.0253 | 0.0365 | 0.1475 | **0.0025** | 0.0208 | 0.0058 |
| Adv. safety | 0.812 | 0.891 | 0.871 | 0.880 | 0.845 | 0.865 |

Completion spreads **5.3 pp** (at the edge of the n=180 Wilson noise
floor); cost spreads **59×**; the within-agent clean-to-adversarial
safety drop spans **11–19 pp**. The completion-only ranking does not
separate the lineup at this scale — the cost, efficiency, and
adversarial-safety axes do.

Full results, figures, and analysis: [`paper/main.pdf`](paper/main.pdf).

---

## Why this benchmark exists

Existing agent benchmarks publish a single completion-style number per
environment. That number tells a researcher whether the model is on the
frontier; it does not tell a practitioner whether they can run the agent in
production. Two agents that score 78% and 74% on $\tau$-bench can still
differ by:

- **60×** in per-run cost — same task, same answer, very different bill;
- **9 pp** in efficiency — one agent solves it in three tool calls, the
  other in seven;
- **19 pp** in adversarial safety — both agents complete the task, one of
  them also follows a prompt-injection payload smuggled in via a tool
  result.

None of those properties shows up on a leaderboard, but every one of them
matters before someone will let an agent touch a real customer record. The
framework is built around the observation that a paper or a vendor should be
able to report all of these properties on the same tasks, with the same
agents, in the same run.

The six axes are not arbitrary. They follow from the questions an operations
engineer actually asks before turning on an agent in production: *did it
work, what did it cost, did it waste calls, does it agree with itself, does
it cope when something downstream fails, and is it safe to feed it data I
don't fully control.* See §1 of the paper for the derivation.

## What's measured

| Axis | Question it answers | Range | Source |
|------|--------------------|------:|--------|
| **Completion** | Did the agent produce the correct answer? | 0–1 | exact match + LLM judge |
| **Cost** | What did this run cost in dollars? | USD | per-call token accounting |
| **Efficiency** | Did the agent get there in a reasonable number of steps? | 0–1 | $\min(s^\star/s,1)\cdot(1-r)$ |
| **Reliability** | Does the same input produce the same answer twice? | 0–1 | Wilson interval over $n$ runs |
| **Recovery** | Does the agent cope when tools fail? | 0–1 | clean-vs-perturbed comparison |
| **Safety** | Does the agent resist prompt-injection in tool output? | 0–1 | compliance + detection + exfiltration |

Each task runs under three **conditions**:

- **clean** — tools behave normally.
- **noisy** — each tool call has a `failure_rate` chance of returning a
  timeout, HTTP 500, malformed JSON, empty response, or 429.
- **adversarial** — clean returns are post-processed with a 50% chance of
  appending a payload from a 15-entry indirect-injection catalogue
  (direct overrides, authority impersonation, embedded-in-JSON, markdown
  exfiltration, comment-channel embedding, Unicode homoglyph and
  zero-width obfuscation, base64-wrapped instructions, fake-error
  recovery, multi-turn poisoning, nested JSON, XML-tag impersonation,
  delayed-effect post-processing).

## Quickstart

### 1. Install

```bash
git clone https://github.com/<you>/agentops-bench.git
cd agentops-bench
python3.11 -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
```

### 2. Provide credentials

Put keys in `.env.local` (gitignored) or export them:

```bash
export ANTHROPIC_API_KEY="sk-ant-..."
export OPENAI_API_KEY="sk-..."
export TAVILY_API_KEY="tvly-..."     # optional, only for live web search
```

Run with `AGENTOPS_LIVE=0` to use snapshotted fixtures only (no live tool
back-ends, fully deterministic). `AGENTOPS_LIVE=1` hits live APIs and
records a fresh fixture if none exists.

### 3. Smoke test

```bash
python3 scripts/smoke_test.py
```

Runs one (agent, task, condition) triple end-to-end against fixtures —
verifies the install in under 30 seconds.

### 4. Run a benchmark

```bash
agentops-bench run \
  --tasks tasks \
  --agents claude-haiku,gpt-5.4-mini \
  --conditions clean,noisy,adversarial \
  --runs 3 \
  --budget 5 \
  --output results/my_run
```

Generates `results/my_run/<agent>_report.json` per agent with per-task
scores and full traces.

### 5. Reproduce the paper's pilot

```bash
bash scripts/run_pilot_v2.sh                   # 6 agents × 20 tasks × 3 cond × 3 runs
python3 scripts/analyze_pilot.py results/pilot_v2
python3 scripts/make_figures.py  results/pilot_v2
cd paper && pdflatex main.tex && bibtex main && pdflatex main.tex && pdflatex main.tex
```

Wall-clock on a laptop: ≈4h21m. API spend: $42.43 against a $200 budget cap.

## Architecture

```
                         +------------------+
                         |   CLI (click)    |
                         +--------+---------+
                                  |
                         +--------v---------+
                         | BenchmarkRunner  |
                         +--------+---------+
                                  |
              +-------------------+-------------------+
              |                                       |
    +---------v----------+               +------------v-----------+
    |   AgentAdapter     |               | InstrumentedToolServer |
    | (Anthropic/OpenAI) |<------------->|  failure injection +   |
    +--------------------+  tool calls   |  prompt injection      |
              |                          +------------------------+
              |
    +---------v----------+
    |   Scoring Suite    |
    | completion | cost  |
    | efficiency | reliab|
    | recovery   | safety|
    +---------+----------+
              |
    +---------v----------+
    |  BenchmarkReport   |
    |  (JSON + tables)   |
    +--------------------+
```

Determinism: external tool back-ends (Open-Meteo, yfinance, Tavily) are
snapshotted into `fixtures/` per `(tool, args)` and replayed from disk.
In-process tools (SQL, sandboxed Python, scoped filesystem) run with fixed
seeds.

## Repo layout

```
agentops-bench/
  src/agentops_bench/
    cli.py             # `agentops-bench run|report|validate`
    runner.py          # BenchmarkRunner: orchestrates runs
    schema.py          # Task / RunResult / BenchmarkReport (Pydantic)
    tools.py           # InstrumentedToolServer: 10 reference tools + failure modes
    injection.py       # 15-entry prompt-injection catalogue
    agents/            # AnthropicAgent, OpenAIAgent (ReAct loops)
    scoring/           # completion, cost, efficiency, reliability, recovery, safety
  tasks/
    code/  data_analysis/  research/  safety/  tool_use/   # 100 seed tasks (v1.0)
    README.md          # task YAML schema + how to add tasks
  fixtures/            # snapshotted tool responses for replay
  scripts/
    run_pilot_v2.sh    # the pilot in the paper
    analyze_pilot.py   # results -> markdown + long-form CSV
    make_figures.py    # results -> 4 PDFs in paper/figs/
    build_fixtures.py  # refresh tool-call snapshots from live APIs
    smoke_test.py      # fast end-to-end check
  results/
    pilot_v2/          # 1,080-run pilot reported in the paper
  paper/
    main.tex           # arXiv build
    main_neurips.tex   # NeurIPS build (loads neurips_2026.sty)
    body.tex           # shared body for both builds
    references.bib
    figs/              # generated PDFs
```

## Adding a new agent

Implement `agents/your_agent.py` with the contract from
[`agents/anthropic_agent.py`](src/agentops_bench/agents/anthropic_agent.py):
construct from a `model` string, expose an async `run(task, tool_server,
condition) -> RunResult`, and emit per-step token counts. Then register a
shorthand in `cli.py::AGENT_REGISTRY` and the new agent is selectable via
`--agents your-agent,...`.

## Adding a new task

See [`tasks/README.md`](tasks/README.md) for the YAML schema. The minimal
task:

```yaml
id: tool_use/021
domain: tool_use
description: Compare weather in Tokyo and Paris and recommend which to visit this weekend.
tools_available: [get_weather, web_search]
optimal_steps: 3
difficulty: easy
```

Validate with `agentops-bench validate --tasks tasks/`. Tasks with an
`expected_output` are scored deterministically; ones without are routed to
the LLM judge.

## Adding a new injection payload

Append an `InjectionPayload` to the catalogue in
[`src/agentops_bench/injection.py`](src/agentops_bench/injection.py) with
its attack class, the payload string, and the canary string the safety
scorer looks for in the agent's reasoning or output. The next adversarial
run will sample it uniformly.

## Limitations (read these before quoting numbers)

- **Replayed back-ends.** External tools are replayed from snapshots. Real
  failure modes that don't fit the five injected categories (long-tail
  latencies, partially-successful retries, correlated outages) won't show
  up.
- **Finite injection catalogue.** An attacker who has read the catalogue
  can trivially evade detection. The safety axis upper-bounds real-world
  robustness, not estimates it.
- **Small task suite.** 20 tasks; per-domain $n$ is 27–63 with three
  conditions and three repeats. The 95% completion CI per cell is roughly
  ±10 pp. Treat the rankings as illustrative of the framework, not as a
  verdict on the agents.
- **$n=3$ repeats.** Tight on reliability — five of six pilot agents
  saturate the axis. Larger $n$ on harder tasks is where reliability
  starts discriminating.
- **List-price cost.** Caching and provisioned throughput change the
  picture for high-volume operators (typically in favour of higher-end
  models).
- **LLM-judge bias.** Completion under ambiguous tasks inherits judge
  biases. The judge model and rubric are published so scores can be
  reproduced.

## Citation

```bibtex
@misc{srivastav2026agentopsbench,
  title  = {AgentOps-Bench: Measuring What Production Operators Actually
            Care About in LLM Agents},
  author = {Kunwar Shivam Srivastav},
  year   = {2026},
  note   = {Preprint},
}
```

## License

Apache-2.0. See [LICENSE](LICENSE).
