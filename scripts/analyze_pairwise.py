"""Pairwise pairwise-comparison analysis with Holm-Bonferroni correction.

For every pair of agents, compute the difference in mean completion (and
adversarial safety) on the shared set of (task, condition, run) cells.
We treat each (task, condition, run) triple as a paired observation:
agent A's completion on cell X minus agent B's completion on cell X. The
test statistic is the Wilcoxon signed-rank statistic over those paired
deltas (a non-parametric paired test that does not assume normality of
the difference distribution).

After all C(6,2)=15 pairwise tests, we apply Holm-Bonferroni to control
family-wise error rate at 0.05.

Usage:
    python3 scripts/analyze_pairwise.py results/pilot_v2
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
from itertools import combinations
from pathlib import Path

SHORT = {
    "anthropic/claude-haiku-4-5-20251001": "haiku-4-5",
    "anthropic/claude-sonnet-4-6":         "sonnet-4-6",
    "anthropic/claude-opus-4-7":           "opus-4-7",
    "openai/gpt-5.4-mini":                 "gpt-5.4-mini",
    "openai/gpt-5.5":                      "gpt-5.5",
    "openai/o4-mini":                      "o4-mini",
}


def _injection_fired(trace: dict | None) -> bool:
    if not isinstance(trace, dict):
        return False
    for step in trace.get("steps", []):
        for tr in step.get("tool_results", []):
            if tr.get("injected_payload"):
                return True
    return False


def load_per_run(root: Path) -> list[dict]:
    """Return one record per (agent, task, condition, run_number).

    The ``injection_fired`` flag records whether any tool result in the trace
    actually carried an injected payload, so safety-axis pairwise tests can
    restrict to cells where both agents in the pair saw a payload (different
    agents take different tool-call paths and therefore different injection
    counts; comparing safety on cells where one side never saw the attack is
    not a paired test of injection resistance).
    """
    rows: list[dict] = []
    for rpt in sorted(root.glob("*_report.json")):
        d = json.loads(rpt.read_text())
        agent = d["agent_id"]
        for r in d["results"]:
            scores = r["scores"]
            rows.append({
                "agent": agent,
                "key": (r["task_id"], r["condition"], r["run_number"]),
                "completion": scores.get("completion") or 0.0,
                "safety":     (scores.get("safety") or {}).get("overall", 1.0),
                "condition":  r["condition"],
                "injection_fired": _injection_fired(r.get("trace")),
            })
    return rows


def wilcoxon_signed_rank_p(deltas: list[float]) -> float:
    """Two-sided Wilcoxon signed-rank test, normal approximation.

    Drops zero-deltas (zero-variance pairs add no information). For
    n_nonzero >= 20 the normal approximation is well-behaved; smaller
    samples we still use the same approximation rather than enumerating
    permutations — adequate for the >=120-pair tests we run here.
    """
    nonzero = [d for d in deltas if d != 0.0]
    n = len(nonzero)
    if n == 0:
        return 1.0
    abs_deltas = [abs(d) for d in nonzero]
    order = sorted(range(n), key=lambda i: abs_deltas[i])
    ranks = [0.0] * n
    i = 0
    while i < n:
        j = i
        while j + 1 < n and abs_deltas[order[j + 1]] == abs_deltas[order[i]]:
            j += 1
        avg_rank = (i + j) / 2.0 + 1.0  # 1-indexed midpoint
        for k in range(i, j + 1):
            ranks[order[k]] = avg_rank
        i = j + 1
    w_plus = sum(r for r, d in zip(ranks, nonzero) if d > 0)
    w_minus = sum(r for r, d in zip(ranks, nonzero) if d < 0)
    w = min(w_plus, w_minus)
    mean = n * (n + 1) / 4.0
    var  = n * (n + 1) * (2 * n + 1) / 24.0
    if var == 0:
        return 1.0
    z = (w - mean) / math.sqrt(var)
    p = 2.0 * 0.5 * math.erfc(abs(z) / math.sqrt(2))
    return min(1.0, p)


def pair_test(rows: list[dict], a1: str, a2: str, metric: str,
              condition: str | None = None,
              require_injection: bool = False) -> dict:
    """Paired test on cells shared between ``a1`` and ``a2``.

    When ``require_injection`` is True, the pair is restricted to cells where
    *both* agents recorded an injected payload in the trace; this is the
    appropriate filter for the adversarial-safety axis, where unfired
    injections trivially score 1.0 and would dilute the comparison.
    """
    by_a1 = {r["key"]: r for r in rows
             if r["agent"] == a1
             and (condition is None or r["condition"] == condition)
             and (not require_injection or r["injection_fired"])}
    by_a2 = {r["key"]: r for r in rows
             if r["agent"] == a2
             and (condition is None or r["condition"] == condition)
             and (not require_injection or r["injection_fired"])}
    keys  = sorted(set(by_a1) & set(by_a2))
    deltas = [by_a1[k][metric] - by_a2[k][metric] for k in keys]
    if not deltas:
        return {"n": 0, "delta": 0.0, "p": 1.0}
    return {
        "n":     len(deltas),
        "delta": statistics.fmean(deltas),
        "p":     wilcoxon_signed_rank_p(deltas),
    }


def holm_bonferroni(p_values: list[float], alpha: float = 0.05) -> list[bool]:
    """Return a list of booleans: True = reject H0 for that comparison."""
    n = len(p_values)
    indexed = sorted(enumerate(p_values), key=lambda kv: kv[1])
    rejects = [False] * n
    for rank, (orig_idx, p) in enumerate(indexed):
        threshold = alpha / (n - rank)
        if p <= threshold:
            rejects[orig_idx] = True
        else:
            break
    return rejects


def report(rows: list[dict], metric: str, condition: str | None, label: str,
           require_injection: bool = False):
    agents = sorted({r["agent"] for r in rows})
    pairs = list(combinations(agents, 2))
    results = []
    for a1, a2 in pairs:
        results.append((a1, a2, pair_test(rows, a1, a2, metric, condition,
                                          require_injection=require_injection)))
    p_values = [r[2]["p"] for r in results]
    rejects  = holm_bonferroni(p_values, alpha=0.05)

    print(f"\n## {label}")
    cond_str = f"condition={condition}" if condition else "all conditions"
    if require_injection:
        cond_str += ", injection actually fired in both traces"
    print(f"\nMetric: {metric}, {cond_str}.\n")
    print("| Agent A | Agent B | n pairs | mean delta | raw p | Holm-corrected reject H0 |")
    print("|---|---|---:|---:|---:|---:|")
    for (a1, a2, res), reject in zip(results, rejects):
        print(f"| {SHORT.get(a1, a1)} | {SHORT.get(a2, a2)} | "
              f"{res['n']} | {res['delta']:+.4f} | {res['p']:.4f} | "
              f"{'**yes**' if reject else 'no'} |")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("results_dir", type=Path)
    args = ap.parse_args()

    rows = load_per_run(args.results_dir)
    if not rows:
        raise SystemExit(f"no result data under {args.results_dir}")

    report(rows, "completion", None, "Pairwise completion (all conditions)")
    report(rows, "safety", "adversarial",
           "Pairwise adversarial safety (injection-fired cells only)",
           require_injection=True)


if __name__ == "__main__":
    main()
