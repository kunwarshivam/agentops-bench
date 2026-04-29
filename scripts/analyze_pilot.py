"""Cross-agent comparison for a finished pilot run.

Reads every ``*_report.json`` under a results directory and emits:
  1. Per-agent aggregate scores (one row per agent).
  2. Per-agent x per-domain breakdown (means across all runs+conditions).
  3. Per-agent x per-condition breakdown (clean / noisy / adversarial).

Usage:
    python3 scripts/analyze_pilot.py results/pilot_v2

Writes ``analysis.md`` and ``analysis_long.csv`` next to the reports.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from pathlib import Path
from typing import Any

# Cost-normalised accuracy: completion · e^(-α·cost), α = -ln(0.9) ≈ 0.105.
# Mirrors agentops_bench.scoring.cost.CNA_ALPHA so this script is standalone.
CNA_ALPHA: float = 0.10536051565782628


def cna(completion: float, cost_usd: float) -> float:
    return completion * math.exp(-CNA_ALPHA * max(cost_usd, 0.0))


def domain_of(task_id: str) -> str:
    return task_id.split("/", 1)[0] if "/" in task_id else task_id


def safe_get(obj: Any, *keys: str, default: float | None = None) -> Any:
    cur = obj
    for k in keys:
        if not isinstance(cur, dict) or k not in cur:
            return default
        cur = cur[k]
    return cur


def load_reports(root: Path) -> list[dict]:
    return [json.loads(p.read_text()) for p in sorted(root.glob("*_report.json"))]


def _injection_fired(trace: dict | None) -> bool:
    """Return True iff at least one tool_result in the trace carried an
    injected payload (i.e. the adversarial probe actually reached the agent)."""
    if not isinstance(trace, dict):
        return False
    for step in trace.get("steps", []):
        for tr in step.get("tool_results", []):
            if tr.get("injected_payload"):
                return True
    return False


def per_run_rows(reports: list[dict]) -> list[dict]:
    rows: list[dict] = []
    for rpt in reports:
        agent = rpt["agent_id"]
        for r in rpt["results"]:
            s = r["scores"]
            completion = s.get("completion") or 0.0
            cost_usd = safe_get(s, "cost", "cost_usd", default=0.0) or 0.0
            rec = s.get("recovery") if isinstance(s.get("recovery"), dict) else None
            recovery_applicable = bool(rec and rec.get("applicable"))
            recovery_overall = float(rec["overall"]) if rec and rec.get("overall") is not None else 0.0
            inj_fired = _injection_fired(r.get("trace"))
            rows.append({
                "agent": agent,
                "task": r["task_id"],
                "domain": domain_of(r["task_id"]),
                "condition": r["condition"],
                "run": r["run_number"],
                "completion": completion,
                "efficiency": safe_get(s, "efficiency", "overall", default=0.0) or 0.0,
                "safety": safe_get(s, "safety", "overall", default=0.0) or 0.0,
                "reliability_rate": safe_get(
                    s, "reliability", "reliability_rate", default=0.0
                ) or 0.0,
                "cost_usd": cost_usd,
                "cna": cna(completion, cost_usd),
                "recovery_applicable": int(recovery_applicable),
                "recovery_overall": recovery_overall,
                "injection_fired": int(inj_fired),
                "wall_time_seconds": safe_get(
                    s, "efficiency", "wall_time_seconds", default=0.0
                ) or 0.0,
            })
    return rows


def mean(xs: list[float]) -> float:
    return statistics.fmean(xs) if xs else 0.0


def group_table(rows: list[dict], group_keys: list[str]) -> list[dict]:
    buckets: dict[tuple, list[dict]] = {}
    for r in rows:
        key = tuple(r[k] for k in group_keys)
        buckets.setdefault(key, []).append(r)
    out = []
    for key, group in sorted(buckets.items()):
        rec = {k: v for k, v in zip(group_keys, key)}
        rec["n"] = len(group)
        for metric in ("completion", "efficiency", "safety",
                       "reliability_rate", "cost_usd", "cna",
                       "wall_time_seconds"):
            rec[metric] = mean([g[metric] for g in group])
        out.append(rec)
    return out


def fmt_md_table(rows: list[dict], cols: list[str]) -> str:
    if not rows:
        return "_(no data)_\n"
    lines = ["| " + " | ".join(cols) + " |",
             "|" + "|".join(["---"] * len(cols)) + "|"]
    for r in rows:
        cells = []
        for c in cols:
            v = r.get(c)
            if isinstance(v, float):
                cells.append(f"{v:.4f}" if c.endswith("usd") or c == "wall_time_seconds"
                             else f"{v:.3f}")
            else:
                cells.append(str(v))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines) + "\n"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("results_dir", type=Path)
    args = ap.parse_args()

    reports = load_reports(args.results_dir)
    if not reports:
        raise SystemExit(f"No *_report.json files in {args.results_dir}")

    rows = per_run_rows(reports)

    md = ["# Pilot analysis\n",
          f"Source: `{args.results_dir}` ({len(reports)} agents, "
          f"{len(rows)} runs)\n"]

    md.append("\n## By agent\n")
    md.append(fmt_md_table(
        group_table(rows, ["agent"]),
        ["agent", "n", "completion", "efficiency", "safety",
         "reliability_rate", "cost_usd", "cna", "wall_time_seconds"],
    ))

    md.append("\n## Recovery (conditional, applicable runs only)\n")
    rec_rows = []
    for agent in sorted({r["agent"] for r in rows}):
        applicable = [r for r in rows if r["agent"] == agent and r["recovery_applicable"]]
        if applicable:
            rec_rows.append({
                "agent": agent,
                "n_applicable": len(applicable),
                "recovery_overall": mean([r["recovery_overall"] for r in applicable]),
            })
    md.append(fmt_md_table(
        rec_rows,
        ["agent", "n_applicable", "recovery_overall"],
    ))

    md.append("\n## Safety conditional on injection firing\n")
    md.append(
        "Headline safety averages over all runs (including those where no\n"
        "injection actually reached the agent, which return safety=1.0 by\n"
        "definition). The table below restricts to the subset of runs in which\n"
        "at least one tool result carried an injected payload, so it captures\n"
        "the operationally meaningful question: \\\"given the agent saw a\n"
        "payload, how often did it stay safe?\\\".\n"
    )
    cond_rows = []
    for agent in sorted({r["agent"] for r in rows}):
        injected = [r for r in rows if r["agent"] == agent and r["injection_fired"]]
        if injected:
            cond_rows.append({
                "agent": agent,
                "n_injected": len(injected),
                "safety_on_injected": mean([r["safety"] for r in injected]),
                "safety_headline": mean([r["safety"] for r in rows if r["agent"] == agent]),
            })
    md.append(fmt_md_table(
        cond_rows,
        ["agent", "n_injected", "safety_on_injected", "safety_headline"],
    ))

    md.append("\n## By agent x domain\n")
    md.append(fmt_md_table(
        group_table(rows, ["agent", "domain"]),
        ["agent", "domain", "n", "completion", "efficiency", "safety",
         "cost_usd"],
    ))

    md.append("\n## By agent x condition\n")
    md.append(fmt_md_table(
        group_table(rows, ["agent", "condition"]),
        ["agent", "condition", "n", "completion", "efficiency", "safety",
         "reliability_rate"],
    ))

    out_md = args.results_dir / "analysis.md"
    out_md.write_text("\n".join(md))

    out_csv = args.results_dir / "analysis_long.csv"
    with out_csv.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    print(f"wrote {out_md}")
    print(f"wrote {out_csv}")


if __name__ == "__main__":
    main()
