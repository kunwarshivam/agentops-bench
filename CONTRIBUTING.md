# Contributing to AgentOps-Bench

Thanks for your interest in extending AgentOps-Bench. This benchmark is
deliberately small at v1.0 (100 seed tasks, 15 prompt-injection
payloads, 6 axes); the most useful contributions are new tasks, new
adversarial payloads, and new agent adapters.

## Ground rules

- **Reproducibility first.** Anything new has to run under
  `AGENTOPS_LIVE=0` against snapshotted fixtures. PRs that depend on
  unsnapshotted live APIs will be asked to ship the snapshot too.
- **Apache-2.0.** All accepted contributions are released under
  Apache-2.0. Don't include text you can't relicense.
- **Determinism.** Tasks and payloads must be deterministic given the
  per-cell seed. No wall-clock branching, no `random()` without a
  seeded RNG.
- **Small PRs.** One axis or one new domain per PR keeps review tight.

## Local setup

```bash
git clone https://github.com/kunwarshivam/agentops-bench.git
cd agentops-bench
python3.11 -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
python3 scripts/smoke_test.py     # ~30s sanity check
```

## Adding a task

Tasks live under `tasks/<domain>/<id>.yaml`. The schema is documented
in [`tasks/README.md`](tasks/README.md) and validated by
`agentops-bench validate --tasks tasks/`.

Minimum fields:

```yaml
id: tool_use/021
domain: tool_use
description: |
  One-paragraph prompt the agent sees.
tools_available: [get_weather, web_search]
optimal_steps: 3
difficulty: easy
```

Add `expected_output` for deterministic scoring. Tasks without it are
routed to the LLM judge (see `src/agentops_bench/scoring/completion.py`).
Run `agentops-bench validate --tasks tasks/<domain>/<id>.yaml` and
include a smoke run in the PR description showing all three conditions
finishing.

## Adding an injection payload

Append a string to `PROMPT_INJECTIONS` in
[`src/agentops_bench/injection.py`](src/agentops_bench/injection.py)
and a matching detection canary in the safety scorer
(`src/agentops_bench/scoring/safety.py`). Keep one attack class per
entry and document the class in a comment (e.g. `# 16 - Tool-result
markdown image exfiltration`). The next adversarial run will sample
it uniformly.

Payload PRs should include:

- the attack class label (one of: direct override, authority
  impersonation, embedded-in-JSON, markdown exfiltration, comment
  channel, Unicode homoglyph, base64-wrapped, fake-error recovery,
  multi-turn poisoning, nested JSON, XML-tag impersonation,
  delayed effect, zero-width obfuscation, role-play, or a new class);
- a sentence on why a deployed agent would plausibly see this string
  in a tool result;
- the canary substring or detection rule the safety scorer should
  flag in the agent's reasoning or output.

## Adding an agent adapter

Implement `src/agentops_bench/agents/<provider>_agent.py` matching the
contract in
[`anthropic_agent.py`](src/agentops_bench/agents/anthropic_agent.py):
construct from a model string, expose async
`run(task, tool_server, condition) -> RunResult`, and emit per-step
input/output token counts. Register a shorthand in
`cli.py::AGENT_REGISTRY`. Include a smoke run on `tool_use/001` under
all three conditions in the PR description.

## Adding a scoring axis

The six v1.0 axes (completion, cost, efficiency, reliability,
recovery, safety) are stable. New axes are welcome but should ship
with: a one-paragraph operator-relevance argument; a deterministic
formula; per-cell unit tests against synthetic traces; and a paragraph
of paper text we can drop into §3.

## Reporting issues

Please file issues with: the agent, task ID, condition, run number,
and a copy of the failing per-run JSON if possible. The per-run trace
contains everything the scorer saw and is the fastest way to
reproduce.
