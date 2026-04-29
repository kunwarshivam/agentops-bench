"""Merge the two-part Opus 4.7 v1.0 pilot run.

The Opus pilot ran in two pieces because the first $50 budget cap stopped
the run partway through. Part 1 (results/pilot_v1_seeded/opus_4_7/) holds
50 fully-completed tasks plus 2 partial cells from research/011. Part 2
(results/pilot_v1_seeded/opus_4_7_part2/) ran a 50-task subset of the
remaining tasks (research/011-020 + safety/* + tool_use/*).

This script:
  1. Loads both per-agent reports.
  2. Drops cells from part 1 that belong to tasks not fully completed
     there (so the partial research/011 cells are discarded; part 2's
     research/011 cells are the ones we keep).
  3. Concatenates the kept cells, recomputes aggregate_scores, and
     overwrites the part 1 report file with the unified report.
  4. Renames part 2's report file so combine_reports.py's glob doesn't
     double-count it.

Run after part 2 finishes:
    python3 scripts/merge_opus_parts.py
"""
from __future__ import annotations

import json
import statistics
from collections import defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1] / "results" / "pilot_v1_seeded"
PART1 = ROOT / "opus_4_7" / "anthropic_claude-opus-4-7_report.json"
PART2 = ROOT / "opus_4_7_part2" / "anthropic_claude-opus-4-7_report.json"


def safe_get(obj, *keys, default=None):
    cur = obj
    for k in keys:
        if not isinstance(cur, dict) or k not in cur:
            return default
        cur = cur[k]
    return cur


def filter_fully_done(results):
    by_task = defaultdict(list)
    for r in results:
        by_task[r["task_id"]].append(r)
    kept = []
    dropped = []
    for tid, rs in by_task.items():
        if len(rs) == 9:
            kept.extend(rs)
        else:
            dropped.append((tid, len(rs)))
    return kept, dropped


def recompute_aggregate(results):
    completion = [r["scores"].get("completion") or 0.0 for r in results]
    efficiency = [
        safe_get(r["scores"], "efficiency", "overall", default=0.0) or 0.0
        for r in results
    ]
    safety = [
        safe_get(r["scores"], "safety", "overall", default=0.0) or 0.0
        for r in results
    ]
    cost_usd = [
        safe_get(r["scores"], "cost", "cost_usd", default=0.0) or 0.0
        for r in results
    ]
    return {
        "mean_completion": statistics.fmean(completion),
        "mean_efficiency": statistics.fmean(efficiency),
        "mean_safety": statistics.fmean(safety),
        "total_cost_usd": sum(cost_usd),
        "n_cells": len(results),
        "n_tasks": len({r["task_id"] for r in results}),
    }


def main() -> None:
    if not PART1.exists():
        raise SystemExit(f"missing part 1: {PART1}")
    if not PART2.exists():
        raise SystemExit(f"missing part 2: {PART2}")

    part1 = json.loads(PART1.read_text())
    part2 = json.loads(PART2.read_text())

    p1_kept, p1_dropped = filter_fully_done(part1["results"])
    p2_kept, p2_dropped = filter_fully_done(part2["results"])

    print(f"part 1: {len(part1['results'])} cells -> kept {len(p1_kept)} "
          f"(fully-done tasks), dropped {len(p1_dropped)} partial tasks: "
          f"{p1_dropped}")
    print(f"part 2: {len(part2['results'])} cells -> kept {len(p2_kept)} "
          f"(fully-done tasks), dropped {len(p2_dropped)} partial tasks: "
          f"{p2_dropped}")

    p1_task_ids = {r["task_id"] for r in p1_kept}
    p2_task_ids = {r["task_id"] for r in p2_kept}
    overlap = p1_task_ids & p2_task_ids
    if overlap:
        print(f"WARNING: overlap between parts: {sorted(overlap)} — "
              "deduplicating, preferring part 2")
        p1_kept = [r for r in p1_kept if r["task_id"] not in overlap]

    merged = p1_kept + p2_kept
    n_unique_tasks = len({r["task_id"] for r in merged})
    print(f"merged: {len(merged)} cells across {n_unique_tasks} unique tasks")

    agg = recompute_aggregate(merged)
    print("aggregate:", json.dumps(agg, indent=2))

    out = dict(part1)
    out["results"] = merged
    out["aggregate_scores"] = agg
    out["total_tasks"] = n_unique_tasks
    out["_merge_provenance"] = {
        "part1_source": str(PART1.relative_to(ROOT.parent.parent)),
        "part2_source": str(PART2.relative_to(ROOT.parent.parent)),
        "part1_kept": len(p1_kept),
        "part2_kept": len(p2_kept),
    }

    PART1.write_text(json.dumps(out, indent=2))
    print(f"wrote unified report -> {PART1}")

    hidden = PART2.with_name("_partial_provenance.json")
    if PART2.exists():
        PART2.rename(hidden)
    print(f"renamed part2 report so combine_reports.py skips it: "
          f"{PART2.name} -> {hidden.name}")


if __name__ == "__main__":
    main()
