#!/usr/bin/env python3
"""Appendix descriptive table: AI-role share, benefit prevalence, experience and
education requirement distributions over the full sample.

Reproducible replacement for the ad-hoc ``descriptive table 1.png``.

Two-stage design, like scripts/plot_benefit_cooccurrence.py:

1. Compute stage (needs the full labeled parquet): aggregates shares/counts to
   ``results/tables_2026/descriptive/appendix_descriptive_table_stats.csv``.
2. Render stage (runs anywhere): writes ``appendix_descriptive_table.tex`` from the CSV.

Run with no arguments: renders from the CSV if it exists, otherwise computes
it first. Use ``--recompute`` to force the compute stage.
"""

import argparse
import sys
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from package_files.benefits_defns import benefits4, benefits_labels_map  # noqa: E402

try:
    from package_files.config_utils import get_processed_dir
except ImportError:
    def get_processed_dir():
        return REPO_ROOT / "data" / "processed"

TABLES_DIR = REPO_ROOT / "results" / "tables_2026" / "descriptive"
STATS_CSV = TABLES_DIR / "appendix_descriptive_table_stats.csv"
OUTPUT_TEX = TABLES_DIR / "appendix_descriptive_table.tex"


def compute_aggregates():
    """Aggregate the full labeled data into the appendix descriptive table stats CSV."""
    data_path = get_processed_dir() / "labeled_v2.parquet"
    print(f"Loading {data_path} ...")
    df = pd.read_parquet(
        data_path,
        columns=["AI ROLE", "EXPERIENCE_BUCKET", "MIN_EDULEVELS_NAME", "YEAR"] + benefits4,
    )
    n = len(df)
    rows = [{
        "panel": "meta", "item": "sample",
        "pct": 100.0, "count": n,
        "detail": f"{int(df['YEAR'].min())}-{int(df['YEAR'].max())}",
    }, {
        "panel": "ai_role", "item": "AI Role",
        "pct": df["AI ROLE"].mean() * 100, "count": int(df["AI ROLE"].sum()),
        "detail": "",
    }]
    for benefit in benefits4:
        share = df[benefit].astype(bool).mean()
        rows.append({
            "panel": "benefits", "item": benefits_labels_map[benefit],
            "pct": share * 100, "count": int(df[benefit].astype(bool).sum()),
            "detail": benefit,
        })
    for panel, col in [("experience", "EXPERIENCE_BUCKET"), ("education", "MIN_EDULEVELS_NAME")]:
        counts = df[col].value_counts(dropna=False)
        for item, count in counts.items():
            rows.append({
                "panel": panel, "item": str(item),
                "pct": count / n * 100, "count": int(count),
                "detail": "",
            })

    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(STATS_CSV, index=False)
    print(f"Saved: {STATS_CSV}")


def fmt_pct(x):
    s = f"{x:.1f}"
    return (s.lstrip("0") if s.startswith("0.") else s) + r"\%"


def render_table():
    stats = pd.read_csv(STATS_CSV)
    meta = stats[stats["panel"] == "meta"].iloc[0]
    ai = stats[stats["panel"] == "ai_role"].iloc[0]

    def panel_rows(panel, sort=True):
        sub = stats[stats["panel"] == panel]
        if sort:
            sub = sub.sort_values("pct", ascending=False)
        return "\n".join(
            f"{row['item']} & {fmt_pct(row['pct'])} \\\\" for _, row in sub.iterrows()
        )

    tex = "\n".join([
        r"\begin{table}[htbp]",
        r"\centering",
        r"\caption{Descriptive statistics}",
        r"\label{tab:appendix-descriptive}",
        r"\begin{tabular}{lr}",
        r"\toprule",
        r"Variable & Percentage \\",
        r"\midrule",
        f"AI Role & {fmt_pct(ai['pct'])} \\\\",
        f"\\quad ({int(ai['count']):,} postings) & \\\\",
        r"\midrule",
        r"\multicolumn{2}{l}{\textbf{Benefits (\% offering each benefit)}} \\",
        panel_rows("benefits"),
        r"\midrule",
        r"\multicolumn{2}{l}{\textbf{Experience Requirements}} \\",
        panel_rows("experience"),
        r"\midrule",
        r"\multicolumn{2}{l}{\textbf{Education Requirements}} \\",
        panel_rows("education"),
        r"\bottomrule",
        r"\multicolumn{2}{l}{\footnotesize\emph{Note:} Sample includes "
        f"{int(meta['count']):,} job postings from {meta['detail']}.}} \\\\",
        r"\end{tabular}",
        r"\end{table}",
    ])
    OUTPUT_TEX.write_text(tex)
    print(f"Saved: {OUTPUT_TEX}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--recompute", action="store_true",
        help="Recompute the stats CSV from the labeled parquet.",
    )
    args = parser.parse_args()

    if args.recompute or not STATS_CSV.exists():
        compute_aggregates()
    render_table()


if __name__ == "__main__":
    main()
