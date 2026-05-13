"""One-command reproduction of the data pipeline.

Runs steps 01--07 in order, producing the tables under output/tables/ and the
figure under output/figures/. Paper compilation (pdflatex + bibtex on
paper/FinalProject_Cheng.tex and paper/Slides_Cheng.tex) is left as a separate
manual step.

Usage:
    python run_all.py             # full pipeline (WRDS pull + analysis)
    python run_all.py --skip-data # skip step 01, start from 02_clean_merge
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CODE = ROOT / "code"

PIPELINE = [
    ("01  Pull WRDS data",       CODE / "01_pull_wrds.py"),
    ("02  Clean & merge",        CODE / "02_clean_merge.py"),
    ("03  Compute returns",      CODE / "03_compute_returns.py"),
    ("04  Construct variables",  CODE / "04_construct_vars.py"),
    ("05  Descriptive stats",    CODE / "05_descriptive.py"),
    ("06  Basu regressions",     CODE / "06_basu_regression.py"),
    ("07  Make figures",         CODE / "07_make_figures.py"),
]


def run(cmd: list[str], description: str, cwd: Path | None = None) -> None:
    print(f"\n{'='*60}")
    print(f"  {description}")
    print(f"  {' '.join(cmd)}")
    print(f"{'='*60}")
    result = subprocess.run(cmd, cwd=cwd)
    if result.returncode != 0:
        print(f"\n*** FAILED: {description} (exit code {result.returncode}) ***")
        sys.exit(result.returncode)


def main() -> None:
    parser = argparse.ArgumentParser(description="Reproduce data + analysis.")
    parser.add_argument("--skip-data", action="store_true",
                        help="Skip step 01 (WRDS pull); use existing data/raw/")
    args = parser.parse_args()

    py = sys.executable

    for description, script in PIPELINE:
        if args.skip_data and script.name == "01_pull_wrds.py":
            print(f"\n  [SKIPPED] {description}")
            continue
        run([py, str(script)], description)

    print(f"\n{'='*60}")
    print("  Pipeline complete. Tables in output/tables/, figure in output/figures/.")
    print("  To build the paper, compile paper/FinalProject_Cheng.tex and paper/Slides_Cheng.tex manually.")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
