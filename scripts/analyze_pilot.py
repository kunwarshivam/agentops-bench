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
import statistics
from pathlib import Path
from typing import Any


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


def per_run_rows(reports: list[dict]) -> list[dict]:
    rows: list[dict] = []
    for rpt in reports:
        agent = rpt["agent_id"]
        for r in rpt["results"]:
            s = r["scores"]
            rows.append({
                "agent": agent,
                "task": r["task_id"],
                "domain": domain_of(r["task_id"]),
                "condition": r["condition"],
                "run": r["run_number"],
                "completion": s.get("completion") or 0.0,
                "efficiency": safe_get(s, "efficiency", "overall", default=0.0) or 0.0,
                "safety": safe_get(s, "safety", "overall", default=0.0) or 0.0,
                "reliability_rate": safe_get(
                    s, "reliability", "reliability_rate", default=0.0
                ) or 0.0,
                "cost_usd": safe_get(s, "cost", "cost_usd", default=0.0) or 0.0,
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
                       "reliability_rate", "cost_usd",
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
         "reliability_rate", "cost_usd", "wall_time_seconds"],
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
