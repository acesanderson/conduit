# /// script
# requires-python = ">=3.11"
# dependencies = ["pandas>=2.0", "openpyxl>=3.1"]
# ///
"""
Bundle every eval result CSV in the repo into a single xlsx for ad-hoc analysis.
One sheet per source CSV (raw, no transformations), plus a README sheet.

Usage:
    uv run python evals/export_results_to_xlsx.py
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parent.parent
OUT = REPO / "evals" / "all_eval_results_export.xlsx"

SOURCES = [
    {
        "sheet": "run2_strategy_eval",
        "path": REPO / "jobs" / "run2_results.csv",
        "purpose": "Run 2: strategy comparison (RollingRefine, MapDedupeReduce, HierarchicalTree, Recursive, OneShot) x models. Canonical fresh data — written by jobs/run2.py via Cronicle event emothl43a01.",
    },
    {
        "sheet": "run1_baseline",
        "path": REPO / "evals" / "results.csv",
        "purpose": "Run 1: RecursiveSummarizer baseline across models. Older schema (no `strategy` column — implicitly RecursiveSummarizer).",
    },
    {
        "sheet": "hybrid_eval",
        "path": REPO / "evals" / "results_hybrid.csv",
        "purpose": "Hybrid eval with separate oneshot_model and map_model columns. Schema differs from run1/run2.",
    },
    {
        "sheet": "ecw_cogito_14b",
        "path": REPO / "jobs" / "ecw_cogito_14b_results.csv",
        "purpose": "Effective context window eval for cogito:14b. NOTE: cogito is not one of the three target router models — keep for cross-reference only.",
    },
    {
        "sheet": "ecw_gpt-oss",
        "path": REPO / "jobs" / "ecw_gpt-oss_latest_results.csv",
        "purpose": "Effective context window eval for gpt-oss:latest. PARTIAL — only 7 rows present locally; ECW sweep (Cronicle empgbar020l) may have written more data on alphablue.",
    },
]


def main() -> None:
    readme_rows = []
    sheets_written = []

    with pd.ExcelWriter(OUT, engine="openpyxl") as writer:
        for src in SOURCES:
            path = src["path"]
            if not path.exists():
                readme_rows.append({
                    "sheet": src["sheet"],
                    "source_path": str(path.relative_to(REPO)),
                    "rows": 0,
                    "columns": "",
                    "status": "MISSING",
                    "purpose": src["purpose"],
                })
                continue

            df = pd.read_csv(path)
            df.to_excel(writer, sheet_name=src["sheet"], index=False)
            sheets_written.append(src["sheet"])

            readme_rows.append({
                "sheet": src["sheet"],
                "source_path": str(path.relative_to(REPO)),
                "rows": len(df),
                "columns": ", ".join(df.columns),
                "status": "OK",
                "purpose": src["purpose"],
            })

        readme_rows.append({
            "sheet": "(gap)",
            "source_path": "—",
            "rows": 0,
            "columns": "",
            "status": "MISSING",
            "purpose": "ECW eval CSVs for qwen3.6:latest and gemma4:latest are NOT present locally. The Cronicle 'ECW sweep (multi-model)' event (empgbar020l) on alphablue may have populated them — check there or Postgres `evals` DB.",
        })
        readme_rows.append({
            "sheet": "(gap)",
            "source_path": "—",
            "rows": 0,
            "columns": "",
            "status": "INFO",
            "purpose": "STRATEGY.md says runs are persisted to Postgres `evals` DB via ConduitDatasetAsync. CSV files may lag the DB — Postgres is canonical if the two disagree.",
        })

        readme_df = pd.DataFrame(readme_rows)
        readme_df.to_excel(writer, sheet_name="README", index=False)

        # Move README to the front
        wb = writer.book
        wb.move_sheet("README", offset=-len(sheets_written))

    print(f"Wrote {OUT}")
    print(f"Sheets: {['README'] + sheets_written}")


if __name__ == "__main__":
    main()
