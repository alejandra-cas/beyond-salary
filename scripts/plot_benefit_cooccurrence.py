#!/usr/bin/env python3
"""Benefit co-occurrence figure: jobs by number of benefits + correlation inset.

Replaces the ad-hoc ``results/figures/benefit cooccurrence.png`` from an
uncommitted notebook session with a reproducible pipeline.

Two-stage design, like scripts/generate_combined_figures.py:

1. Compute stage (needs the full labeled parquet, i.e. Matthew's machine):
   aggregates the per-posting benefit counts and the correlation matrix of
   the six keyword benefits plus a salary-posted indicator, and saves them to
   ``results/tables_2026/descriptive/benefit_cooccurrence_{hist,corr}.csv``.
2. Plot stage (runs anywhere): draws the figure from those CSVs.

Run with no arguments: plots from the CSVs if they exist, otherwise computes
them first. Use ``--recompute`` to force the compute stage.
"""

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
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
HIST_CSV = TABLES_DIR / "benefit_cooccurrence_hist.csv"
CORR_CSV = TABLES_DIR / "benefit_cooccurrence_corr.csv"
OUTPUT_PNG = REPO_ROOT / "results" / "figures_2026" / "descriptive" / "benefit_cooccurrence.png"

SALARY_LABEL = "Salary Posted"


def compute_aggregates():
    """Aggregate the full labeled data into the two small CSVs."""
    data_path = get_processed_dir() / "labeled_v2.parquet"
    print(f"Loading {data_path} ...")
    df = pd.read_parquet(data_path, columns=benefits4 + ["SALARY"])

    flags = df[benefits4].astype(bool)
    n_benefits = flags.sum(axis=1)
    hist = (
        n_benefits.value_counts().sort_index()
        .rename_axis("n_benefits").reset_index(name="n_jobs")
    )

    corr_input = flags.copy()
    corr_input[SALARY_LABEL] = df["SALARY"].notna()
    corr = corr_input.corr()

    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    hist.to_csv(HIST_CSV, index=False)
    corr.to_csv(CORR_CSV)
    print(f"Saved: {HIST_CSV}")
    print(f"Saved: {CORR_CSV}")


def display_label(key):
    return benefits_labels_map.get(key, key)


def plot_figure():
    hist = pd.read_csv(HIST_CSV)
    corr = pd.read_csv(CORR_CSV, index_col=0)

    fig, ax = plt.subplots(figsize=(11, 8))
    ax.bar(hist["n_benefits"], hist["n_jobs"], color="skyblue")
    ax.set_xlabel("Number of Benefits Offered", fontsize=16)
    ax.set_ylabel("Number of Jobs", fontsize=16)
    ax.set_xticks(hist["n_benefits"])
    ax.tick_params(labelsize=14)

    # Correlation heatmap inset (upper right)
    inset = ax.inset_axes([0.33, 0.35, 0.62, 0.62])
    values = corr.values
    inset.imshow(values, cmap="Blues", vmin=0, vmax=1)
    labels = [display_label(k) for k in corr.columns]
    inset.set_xticks(range(len(labels)))
    inset.set_yticks(range(len(labels)))
    inset.set_xticklabels(labels, rotation=45, ha="right", fontsize=9)
    inset.set_yticklabels(labels, fontsize=9)
    for i in range(values.shape[0]):
        for j in range(values.shape[1]):
            inset.text(
                j, i, f"{values[i, j]:.2g}",
                ha="center", va="center", fontsize=8,
                color="white" if values[i, j] > 0.5 else "black",
            )

    OUTPUT_PNG.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(OUTPUT_PNG, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved: {OUTPUT_PNG}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--recompute", action="store_true",
        help="Recompute the aggregate CSVs from the labeled parquet.",
    )
    args = parser.parse_args()

    if args.recompute or not (HIST_CSV.exists() and CORR_CSV.exists()):
        compute_aggregates()
    plot_figure()


if __name__ == "__main__":
    main()
