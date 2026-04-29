"""Generate the figures referenced from the paper.

Outputs:
    paper/figs/cost_completion_pareto.pdf
    paper/figs/safety_by_condition.pdf
    paper/figs/domain_heatmap.pdf
    paper/figs/efficiency_clean_vs_noisy.pdf

Run after the pilot finishes:
    python3 scripts/make_figures.py results/pilot_v2
"""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib
import numpy as np

matplotlib.rcParams["pdf.fonttype"] = 42
matplotlib.rcParams["ps.fonttype"] = 42
matplotlib.rcParams["font.size"] = 10


SHORT = {
    "anthropic/claude-haiku-4-5-20251001":      "haiku-4-5",
    "anthropic/claude-sonnet-4-6":              "sonnet-4-6",
    "anthropic/claude-opus-4-7":                "opus-4-7",
    "openai/gpt-5.4-mini":                      "gpt-5.4-mini",
    "openai/gpt-5.5":                           "gpt-5.5",
    "openai/o4-mini":                           "o4-mini",
    "openrouter/deepseek/deepseek-v3.2":        "deepseek-v3.2",
    "openrouter/meta-llama/llama-4-scout":      "llama-4-scout",
    "openrouter/qwen/qwen3-max":                "qwen3-max",
    "openrouter/mistralai/mistral-large-2512":  "mistral-large",
}


def safe_get(obj, *keys, default=None):
    cur = obj
    for k in keys:
        if not isinstance(cur, dict) or k not in cur:
            return default
        cur = cur[k]
    return cur


def load(root: Path):
    rows = []
    for rpt in sorted(root.glob("*_report.json")):
        d = json.loads(rpt.read_text())
        agent = d["agent_id"]
        for r in d["results"]:
            task_id = r["task_id"]
            domain = task_id.split("/")[0] if "/" in task_id else "unknown"
            rows.append({
                "agent": agent,
                "task": task_id,
                "domain": domain,
                "condition": r["condition"],
                "completion": r["scores"].get("completion") or 0.0,
                "efficiency": safe_get(r["scores"], "efficiency", "overall", default=0.0) or 0.0,
                "safety": safe_get(r["scores"], "safety", "overall", default=0.0) or 0.0,
                "cost": safe_get(r["scores"], "cost", "cost_usd", default=0.0) or 0.0,
            })
    return rows


def per_agent_means(rows):
    by_agent: dict[str, list[dict]] = {}
    for r in rows:
        by_agent.setdefault(r["agent"], []).append(r)
    out = {}
    for a, rs in by_agent.items():
        out[a] = {
            "completion": statistics.fmean(r["completion"] for r in rs),
            "cost": statistics.fmean(r["cost"] for r in rs),
        }
    return out


def per_agent_per_condition_safety(rows):
    table: dict[tuple[str, str], list[float]] = {}
    for r in rows:
        table.setdefault((r["agent"], r["condition"]), []).append(r["safety"])
    out = {}
    for (a, c), vs in table.items():
        out.setdefault(a, {})[c] = statistics.fmean(vs)
    return out


def per_agent_per_domain_completion(rows):
    table: dict[tuple[str, str], list[float]] = {}
    for r in rows:
        table.setdefault((r["agent"], r["domain"]), []).append(r["completion"])
    out = {}
    for (a, d), vs in table.items():
        out.setdefault(a, {})[d] = statistics.fmean(vs)
    return out


def per_agent_per_condition_efficiency(rows):
    table: dict[tuple[str, str], list[float]] = {}
    for r in rows:
        table.setdefault((r["agent"], r["condition"]), []).append(r["efficiency"])
    out = {}
    for (a, c), vs in table.items():
        out.setdefault(a, {})[c] = statistics.fmean(vs)
    return out


def _pareto_frontier(points):
    """Return the cost-ascending Pareto-non-dominated subset of (cost, compl)
    points. A point is dominated iff some other point has lower-or-equal cost
    AND higher completion."""
    pts = sorted(points, key=lambda p: (p[0], -p[1]))
    frontier = []
    best_compl = -float("inf")
    for cost, compl, name in pts:
        if compl > best_compl:
            frontier.append((cost, compl, name))
            best_compl = compl
    return frontier


def plot_pareto(means, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(5.6, 3.6))
    points = [(m["cost"], m["completion"], agent) for agent, m in means.items()]
    frontier = _pareto_frontier(points)
    frontier_names = {n for _, _, n in frontier}

    for agent, m in means.items():
        on_frontier = agent in frontier_names
        ax.scatter(m["cost"], m["completion"],
                   s=70 if on_frontier else 50,
                   color="#1f77b4" if on_frontier else "#bbbbbb",
                   edgecolor="black" if on_frontier else "#888888",
                   linewidth=0.6,
                   zorder=3)
        ax.annotate(SHORT.get(agent, agent), (m["cost"], m["completion"]),
                    xytext=(7, 4), textcoords="offset points", fontsize=9,
                    color="black" if on_frontier else "#555555")

    if len(frontier) >= 2:
        fx = [p[0] for p in frontier]
        fy = [p[1] for p in frontier]
        ax.step(fx, fy, where="post", color="#1f77b4", linestyle="--",
                linewidth=1.2, alpha=0.7, zorder=2,
                label="Pareto frontier")
        ax.legend(loc="lower right", fontsize=8, frameon=False)

    ax.set_xscale("log")
    ax.set_xlabel("Mean cost per run (USD, log scale)")
    ax.set_ylabel("Mean completion score")
    ax.set_ylim(0.0, 1.0)
    ax.grid(True, which="both", linestyle=":", linewidth=0.5, alpha=0.6)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def plot_domain_heatmap(per_dom, out_path: Path) -> None:
    domains = ["code", "data_analysis", "research", "safety", "tool_use"]
    domain_labels = ["code", "data", "research", "safety", "tool-use"]
    agents = list(per_dom.keys())
    matrix = np.array(
        [[per_dom[a].get(d, float("nan")) for d in domains] for a in agents]
    )
    ranks = np.zeros_like(matrix, dtype=int)
    for j in range(matrix.shape[1]):
        col = matrix[:, j]
        order = np.argsort(-col, kind="stable")
        for r, idx in enumerate(order, start=1):
            ranks[idx, j] = r

    fig, ax = plt.subplots(figsize=(6.4, 3.6))
    im = ax.imshow(matrix, cmap="YlGnBu", vmin=0.0, vmax=1.0, aspect="auto")
    ax.set_xticks(range(len(domains)))
    ax.set_xticklabels(domain_labels)
    ax.set_yticks(range(len(agents)))
    ax.set_yticklabels([SHORT.get(a, a) for a in agents])
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            v = matrix[i, j]
            color = "white" if (v < 0.30 or v > 0.85) else "black"
            ax.text(j, i, f"{v:.2f}\n#{ranks[i, j]}",
                    ha="center", va="center", fontsize=8, color=color)
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Mean completion", fontsize=9)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def plot_efficiency_clean_vs_noisy(per_cond, out_path: Path) -> None:
    agents = list(per_cond.keys())
    width = 0.38
    fig, ax = plt.subplots(figsize=(6.2, 3.4))
    xs = list(range(len(agents)))
    clean = [per_cond[a].get("clean", 0.0) for a in agents]
    noisy = [per_cond[a].get("noisy", 0.0) for a in agents]
    ax.bar([x - width / 2 for x in xs], clean, width=width,
           label="clean", color="#4c9b9b", edgecolor="black", linewidth=0.4)
    ax.bar([x + width / 2 for x in xs], noisy, width=width,
           label="noisy", color="#e0a96d", edgecolor="black", linewidth=0.4)
    for x, c, n in zip(xs, clean, noisy):
        drop = c - n
        sign = "-" if drop > 0 else "+"
        ax.annotate(f"{sign}{abs(drop) * 100:.1f}pp",
                    xy=(x, max(c, n) + 0.01),
                    ha="center", va="bottom", fontsize=8, color="#444")
    ax.set_xticks(xs)
    ax.set_xticklabels([SHORT.get(a, a) for a in agents], rotation=20,
                       ha="right")
    ax.set_ylabel("Mean efficiency score")
    ax.set_ylim(0.7, 1.05)
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.02), ncol=2,
              frameon=False)
    ax.grid(True, axis="y", linestyle=":", linewidth=0.5, alpha=0.6)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def plot_safety(per_cond, out_path: Path) -> None:
    conditions = ["clean", "noisy", "adversarial"]
    agents = list(per_cond.keys())
    width = 0.25
    fig, ax = plt.subplots(figsize=(6.2, 3.4))
    xs = list(range(len(agents)))
    colors = {"clean": "#4c9b9b", "noisy": "#e0a96d", "adversarial": "#c0504d"}
    for i, cond in enumerate(conditions):
        ys = [per_cond[a].get(cond, 0.0) for a in agents]
        ax.bar([x + (i - 1) * width for x in xs], ys, width=width,
               label=cond, color=colors[cond], edgecolor="black",
               linewidth=0.4)
    ax.set_xticks(xs)
    ax.set_xticklabels([SHORT.get(a, a) for a in agents], rotation=20,
                       ha="right")
    ax.set_ylabel("Mean safety score")
    ax.set_ylim(0.7, 1.05)
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.02), ncol=3,
              frameon=False)
    ax.grid(True, axis="y", linestyle=":", linewidth=0.5, alpha=0.6)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("results_dir", type=Path)
    ap.add_argument("--out", type=Path,
                    default=Path(__file__).resolve().parents[1] / "paper" / "figs")
    args = ap.parse_args()

    rows = load(args.results_dir)
    if not rows:
        raise SystemExit(f"no result data under {args.results_dir}")

    plot_pareto(per_agent_means(rows), args.out / "cost_completion_pareto.pdf")
    plot_safety(per_agent_per_condition_safety(rows),
                args.out / "safety_by_condition.pdf")
    plot_domain_heatmap(per_agent_per_domain_completion(rows),
                        args.out / "domain_heatmap.pdf")
    plot_efficiency_clean_vs_noisy(per_agent_per_condition_efficiency(rows),
                                   args.out / "efficiency_clean_vs_noisy.pdf")
    print(f"wrote {args.out}/cost_completion_pareto.pdf")
    print(f"wrote {args.out}/safety_by_condition.pdf")
    print(f"wrote {args.out}/domain_heatmap.pdf")
    print(f"wrote {args.out}/efficiency_clean_vs_noisy.pdf")


if __name__ == "__main__":
    main()
