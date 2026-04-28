"""Assemble the `_combined/` rollup directory consumed by analyze_pilot.py
and make_figures.py.

After scripts/run_pilot_v1_seeded.sh finishes, each per-agent output dir
under results/pilot_v1_seeded/<slug>/ contains a per-agent
<provider>_<model>_report.json plus the per-run trace JSONs in a nested
<provider>_<model>/ subdirectory. This script collects those reports
into results/pilot_v1_seeded/_combined/ so the downstream analysis
scripts can ingest them as a single rollup.

Usage:
    python3 scripts/combine_reports.py results/pilot_v1_seeded
"""
from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("results_dir", type=Path,
                    help="Parent directory containing per-agent output dirs")
    args = ap.parse_args()

    root: Path = args.results_dir
    if not root.is_dir():
        sys.exit(f"{root}: not a directory")

    combined = root / "_combined"
    combined.mkdir(exist_ok=True)

    copied = 0
    for report in root.glob("*/*_report.json"):
        if report.parent.name == "_combined":
            continue
        target = combined / report.name
        shutil.copy2(report, target)
        copied += 1
        print(f"copied {report.relative_to(root)} -> _combined/{report.name}")

    if copied == 0:
        sys.exit(f"no per-agent *_report.json files found under {root}/*/")

    print(f"\ncombined {copied} per-agent report(s) into {combined}")
    print("next:")
    print(f"  python3 scripts/analyze_pilot.py    {combined}")
    print(f"  python3 scripts/make_figures.py     {combined}")


if __name__ == "__main__":
    main()
