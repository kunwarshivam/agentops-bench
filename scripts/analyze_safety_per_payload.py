"""Per-payload-class safety breakdown across the pilot.

Walks the saved adversarial run JSONs, attributes each injected payload to one
of the 12 attack classes documented in the paper, and reports per-(agent,
class) compliance / detection / exfiltration / mean-safety on the runs that
contained at least one occurrence of that class.

Usage:
    python3 scripts/analyze_safety_per_payload.py results/pilot_v2
"""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

# Map each PROMPT_INJECTIONS index (0-14) to a canonical attack class.
# These 12 class labels mirror the body of the paper; payloads 8 and 9 are
# bucketed into authority impersonation and direct override respectively.
PAYLOAD_CLASS = {
    0:  "direct_override",
    1:  "authority_impersonation",
    2:  "embedded_in_json",
    3:  "markdown_exfil",
    4:  "comment_channel",
    5:  "unicode_homoglyph",
    6:  "base64_wrapped",
    7:  "authority_impersonation",   # diagnostic-mode role-play
    8:  "direct_override",           # subtle completion hijack
    9:  "fake_error",
    10: "multi_turn_poison",
    11: "nested_json",
    12: "unicode_zero_width",
    13: "xml_tag",
    14: "delayed_post_processing",
}

# Display order for the table.
CLASS_ORDER = [
    "direct_override",
    "authority_impersonation",
    "embedded_in_json",
    "markdown_exfil",
    "comment_channel",
    "unicode_homoglyph",
    "unicode_zero_width",
    "base64_wrapped",
    "fake_error",
    "multi_turn_poison",
    "nested_json",
    "xml_tag",
    "delayed_post_processing",
]


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


def load_payloads() -> list[str]:
    """Load PROMPT_INJECTIONS by importing the module."""
    from agentops_bench.injection import PROMPT_INJECTIONS
    return list(PROMPT_INJECTIONS)


def classify(payload: str, payloads: list[str]) -> str:
    """Return the attack class for a recorded payload string."""
    for i, p in enumerate(payloads):
        if p == payload:
            return PAYLOAD_CLASS[i]
    # Unknown payload (shouldn't happen if the run used the same catalogue)
    return "unknown"


def walk_runs(root: Path):
    """Yield (agent, run_dict) for every adversarial run JSON.

    Supports both a flat layout (root/<agent>/adversarial/*.json) and the
    multi-agent rollup used in pilot_v1_seeded (root/<bucket>/<agent>/adversarial/*.json).
    """
    for agent_dir in sorted(root.glob("**/adversarial")):
        agent_dirname = agent_dir.parent.name
        # convert "anthropic_claude-haiku-4-5-20251001" -> "anthropic/claude-haiku-4-5-20251001"
        # OpenRouter agent dirs encode the slash-prefix as openrouter_<vendor>_<model>;
        # restore both slashes.
        if agent_dirname.startswith("openrouter_"):
            parts = agent_dirname.split("_", 2)
            agent = "/".join(parts) if len(parts) == 3 else agent_dirname.replace("_", "/", 1)
        else:
            agent = agent_dirname.replace("_", "/", 1)
        for f in sorted(agent_dir.glob("*.json")):
            yield agent, json.loads(f.read_text())


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("results_dir", type=Path)
    args = ap.parse_args()

    payloads = load_payloads()

    # (agent, class) -> list of run-level dicts {compliance, detection, exfil, safety}
    bucket: dict[tuple[str, str], list[dict]] = {}
    occurrence_counts: dict[tuple[str, str], int] = {}

    for agent, run in walk_runs(args.results_dir):
        trace = run["trace"]
        scores = run["scores"]
        safety = scores.get("safety") or {}
        # Collect payload classes that appeared in this run.
        classes_in_run: set[str] = set()
        n_occurrences: dict[str, int] = {}
        for step in trace["steps"]:
            for tr in step.get("tool_results", []):
                pl = tr.get("injected_payload")
                if pl:
                    cls = classify(pl, payloads)
                    classes_in_run.add(cls)
                    n_occurrences[cls] = n_occurrences.get(cls, 0) + 1

        # Attribute the run-level safety breakdown to every class present.
        for cls in classes_in_run:
            bucket.setdefault((agent, cls), []).append({
                "compliance": safety.get("compliance_rate", 0.0),
                "detection":  safety.get("detection_rate", 0.0),
                "exfil":      1 if safety.get("exfiltration_attempt") else 0,
                "safety":     safety.get("overall", 1.0),
                "compliance_hits": safety.get("compliance_hits", 0),
                "detection_hits":  safety.get("detection_hits", 0),
            })
            occurrence_counts[(agent, cls)] = (
                occurrence_counts.get((agent, cls), 0) + n_occurrences[cls]
            )

    # Aggregate across all agents per class.
    by_class: dict[str, list[dict]] = {}
    for (agent, cls), runs in bucket.items():
        by_class.setdefault(cls, []).extend(runs)

    print("\n## Per-attack-class safety (averaged over agents)\n")
    print("| Attack class | Runs containing | Mean safety | Mean compliance | Mean detection | Exfil rate |")
    print("|---|---:|---:|---:|---:|---:|")
    for cls in CLASS_ORDER:
        runs = by_class.get(cls, [])
        if not runs:
            continue
        n = len(runs)
        m_safety   = statistics.fmean(r["safety"]     for r in runs)
        m_compl    = statistics.fmean(r["compliance"] for r in runs)
        m_detect   = statistics.fmean(r["detection"]  for r in runs)
        exfil_rate = statistics.fmean(r["exfil"]      for r in runs)
        print(f"| {cls.replace('_', ' ')} | {n} | {m_safety:.3f} | {m_compl:.3f} | {m_detect:.3f} | {exfil_rate:.3f} |")

    # Per-(agent, class) too, for the supplementary appendix.
    print("\n\n## Per-agent x attack-class mean safety\n")
    agents = sorted({a for a, _ in bucket.keys()})
    header = "| class \\ agent | " + " | ".join(SHORT.get(a, a) for a in agents) + " |"
    print(header)
    print("|" + "|".join(["---"] * (1 + len(agents))) + "|")
    for cls in CLASS_ORDER:
        cells = []
        for a in agents:
            runs = bucket.get((a, cls), [])
            if not runs:
                cells.append("—")
            else:
                m = statistics.fmean(r["safety"] for r in runs)
                cells.append(f"{m:.3f}")
        print(f"| {cls.replace('_', ' ')} | " + " | ".join(cells) + " |")


if __name__ == "__main__":
    main()
