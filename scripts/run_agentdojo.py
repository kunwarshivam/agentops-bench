"""External-anchor extension: run AgentOps-Bench agents on AgentDojo tasks.

AgentDojo (Debenedetti et al., NeurIPS D&B 2024) ships its own indirect
prompt-injection environments. Reporting AgentOps-Bench safety scores
on a held-out subset of AgentDojo gives a *cross-suite* anchor — it
shows that the safety axis is not measuring something idiosyncratic
to our 15-payload catalogue.

This script is a scaffold. It does not run as part of the pilot,
because AgentDojo is an optional dependency that adds ~150 MB and
hits live model endpoints. To actually run it:

    pip install agentdojo
    export OPENAI_API_KEY=...
    export ANTHROPIC_API_KEY=...
    python3 scripts/run_agentdojo.py --suite slack --agents claude-haiku,gpt-5.4-mini

The bridge is intentionally thin: AgentDojo tasks expose ``user_task``
prompts plus environment-side ``injection_task`` payloads. We wrap
each AgentDojo environment as an ``InstrumentedToolServer`` whose
tool-call dispatch delegates to AgentDojo's environment, and re-use
the AgentOps-Bench scoring suite end-to-end. Completion is mapped
from AgentDojo's per-task utility check; safety is mapped from
AgentDojo's "did the injection task succeed" flag (logical NOT — a
robust agent should *not* execute the injection).
"""

from __future__ import annotations

import argparse
import asyncio
import importlib
import json
import sys
from pathlib import Path

# We deliberately do NOT import agentdojo at module import time so the
# rest of the test suite does not fail when AgentDojo is not installed.


def _require_agentdojo():
    try:
        return importlib.import_module("agentdojo")
    except ImportError as exc:  # pragma: no cover - depends on env
        sys.exit(
            "AgentDojo is not installed. Install with: pip install agentdojo\n"
            f"(import error: {exc})"
        )


def _build_agent(name: str):
    """Resolve the AgentOps-Bench shorthand to a live adapter instance."""
    from agentops_bench.cli import AGENT_REGISTRY

    if name not in AGENT_REGISTRY:
        sys.exit(
            f"Unknown agent '{name}'. Available: "
            f"{', '.join(sorted(AGENT_REGISTRY))}"
        )
    module_path, model = AGENT_REGISTRY[name]
    mod_name, cls_name = module_path.rsplit(".", 1)
    cls = getattr(importlib.import_module(mod_name), cls_name)
    return cls(model=model)


async def _run_one_dojo_task(agent, suite_name: str, user_task_id: str,
                             injection_task_id: str | None):
    """Run a single AgentDojo (suite, user_task, injection_task) triple.

    Returns a dict with ``completion``, ``safety_compliance``, and the
    per-step trace so it can be folded into an AgentOps report.
    """
    agentdojo = _require_agentdojo()
    suite = agentdojo.benchmark.get_suite(suite_name)
    user_task = suite.get_user_task(user_task_id)
    injection_task = (
        suite.get_injection_task(injection_task_id)
        if injection_task_id is not None
        else None
    )

    env = suite.load_and_inject_environment(injection_task=injection_task)

    transcript = []
    user_prompt = user_task.PROMPT
    answer = await agent.simple_chat(user_prompt, tools=env.tools, env=env,
                                     on_step=transcript.append)

    return {
        "suite":            suite_name,
        "user_task":        user_task_id,
        "injection_task":   injection_task_id,
        "completion":       float(user_task.utility(env, model_output=answer)),
        "safety_compliance": (
            0.0 if (injection_task is not None
                    and injection_task.security(env, model_output=answer))
            else 1.0
        ),
        "trace":            transcript,
    }


async def _main_async(args: argparse.Namespace) -> None:
    agentdojo = _require_agentdojo()
    suite = agentdojo.benchmark.get_suite(args.suite)

    user_task_ids = [t.ID for t in suite.user_tasks]
    injection_ids = [t.ID for t in suite.injection_tasks][: args.max_injections]

    out: list[dict] = []
    for agent_name in args.agents.split(","):
        agent = _build_agent(agent_name.strip())
        for ut in user_task_ids[: args.max_user_tasks]:
            # Clean run.
            out.append(await _run_one_dojo_task(agent, args.suite, ut, None))
            # Adversarial: the same user task with each candidate injection.
            for it in injection_ids:
                out.append(await _run_one_dojo_task(agent, args.suite, ut, it))

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(out, indent=2))
    print(f"Wrote {len(out)} runs to {args.output}")


def _parse() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--suite", default="slack",
        help="AgentDojo benchmark suite (e.g. slack, banking, travel, workspace)",
    )
    p.add_argument(
        "--agents", default="claude-haiku,gpt-5.4-mini",
        help="Comma-separated AgentOps-Bench agent shortnames",
    )
    p.add_argument("--max-user-tasks", type=int, default=10)
    p.add_argument("--max-injections", type=int, default=5)
    p.add_argument(
        "--output", type=Path,
        default=Path("results/agentdojo_anchor/runs.json"),
    )
    return p.parse_args()


if __name__ == "__main__":
    asyncio.run(_main_async(_parse()))
