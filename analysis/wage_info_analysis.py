#!/usr/bin/env python3
"""Plot wage-information prevalence by benefit type and AI-role status."""

from pathlib import Path
import sys

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

sys.path.append(str(Path(__file__).parent.parent / "src"))
from package_files.benefits_defns import benefits4, benefit_colors, benefits_labels_map
from package_files.config_utils import get_processed_dir, get_repo_root


REPO_ROOT = get_repo_root()
PROCESSED_DIR = get_processed_dir()
DATA_PATH = PROCESSED_DIR / "labeled_v2.parquet"
FIGURES_DIR = REPO_ROOT / "results" / "figures_2026" / "descriptive" / "wage_info"
TABLES_DIR = REPO_ROOT / "results" / "tables_2026" / "descriptive"


def benefit_slug(benefit):
    """Return a filename-safe benefit name."""
    return benefit.lower().replace(" ", "_")


def load_data():
    """Load processed postings and add the wage-information indicator."""
    data = pd.read_parquet(DATA_PATH)
    required = ["YEAR", "AI ROLE", "SALARY"] + benefits4
    missing = [column for column in required if column not in data.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    data = data[required].copy()
    data["HAS_WAGE_INFO"] = data["SALARY"].notna()
    data["AI ROLE"] = data["AI ROLE"].astype(bool)
    data["YEAR"] = pd.to_numeric(data["YEAR"], errors="coerce")
    data = data.dropna(subset=["YEAR"]).copy()
    data["YEAR"] = data["YEAR"].astype(int)
    return data


def summarize_wage_info(data):
    """Summarize wage-information prevalence by year, AI role, and benefit presence."""
    rows = []
    for benefit in benefits4:
        grouped = (
            data.groupby(["YEAR", "AI ROLE", benefit], dropna=False, observed=True)
            .agg(
                wage_info_share=("HAS_WAGE_INFO", "mean"),
                postings=("HAS_WAGE_INFO", "size"),
                postings_with_wage_info=("HAS_WAGE_INFO", "sum"),
            )
            .reset_index()
        )
        grouped["benefit"] = benefit
        grouped["benefit_label"] = benefits_labels_map[benefit]
        grouped = grouped.rename(columns={benefit: "has_benefit"})
        rows.append(grouped)

    summary = pd.concat(rows, ignore_index=True)
    summary["ai_role_label"] = summary["AI ROLE"].map({True: "AI Role", False: "Non-AI Role"})
    summary["has_benefit_label"] = summary["has_benefit"].astype(bool).map(
        {True: "Has Benefit", False: "No Benefit"}
    )
    summary["postings_with_wage_info"] = summary["postings_with_wage_info"].astype(int)
    return summary


def plot_combined_wage_info(summary):
    """Create a 2x3 wage-information prevalence figure across keyword benefits."""
    fig, axes = plt.subplots(nrows=2, ncols=3, figsize=(18, 10), sharey=True)
    axes = axes.flatten()

    for index, benefit in enumerate(benefits4):
        ax = axes[index]
        subset = summary[summary["benefit"] == benefit].sort_values("YEAR").copy()
        sns.lineplot(
            data=subset,
            x="YEAR",
            y="wage_info_share",
            hue="ai_role_label",
            style="has_benefit_label",
            errorbar=None,
            palette={"AI Role": benefit_colors[benefit], "Non-AI Role": "gray"},
            hue_order=["AI Role", "Non-AI Role"],
            style_order=["Has Benefit", "No Benefit"],
            ax=ax,
        )
        ax.set_title(benefits_labels_map[benefit])
        ax.set_xlabel(None)
        ax.set_xticks(sorted(subset["YEAR"].unique()))
        ax.tick_params(axis="x", rotation=45)
        ax.set_ylim(0, 1)
        ax.grid(alpha=0.2)
        if index in [0, 3]:
            ax.set_ylabel("Share of Jobs with Wage Information")
        else:
            ax.set_ylabel(None)

        if index == 0:
            ax.legend(title=None, loc="upper left", fontsize=9)
        else:
            ax.legend().remove()

    plt.tight_layout()
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    output = FIGURES_DIR / "figure2b_wage_info_by_benefit_role.png"
    plt.savefig(output, bbox_inches="tight", dpi=300)
    plt.close()
    print(f"Saved: {output}")


def plot_individual_wage_info(summary):
    """Save one wage-information prevalence figure per keyword benefit."""
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    for benefit in benefits4:
        subset = summary[summary["benefit"] == benefit].sort_values("YEAR").copy()
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.lineplot(
            data=subset,
            x="YEAR",
            y="wage_info_share",
            hue="ai_role_label",
            style="has_benefit_label",
            errorbar=None,
            palette={"AI Role": benefit_colors[benefit], "Non-AI Role": "gray"},
            hue_order=["AI Role", "Non-AI Role"],
            style_order=["Has Benefit", "No Benefit"],
            ax=ax,
        )
        ax.set_title(benefits_labels_map[benefit])
        ax.set_xlabel(None)
        ax.set_ylabel("Share of Jobs with Wage Information")
        ax.set_xticks(sorted(subset["YEAR"].unique()))
        ax.tick_params(axis="x", rotation=45)
        ax.set_ylim(0, 1)
        ax.grid(alpha=0.2)
        ax.legend(title=None, loc="upper left", fontsize=9)
        plt.tight_layout()
        output = FIGURES_DIR / f"wage_info_{benefit_slug(benefit)}.png"
        plt.savefig(output, bbox_inches="tight", dpi=300)
        plt.close()
        print(f"Saved: {output}")


def main():
    data = load_data()
    print(f"Loaded {len(data):,} postings")
    summary = summarize_wage_info(data)
    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    output = TABLES_DIR / "wage_info_by_benefit_role.csv"
    summary.to_csv(output, index=False)
    print(f"Saved: {output}")
    plot_combined_wage_info(summary)
    plot_individual_wage_info(summary)


if __name__ == "__main__":
    main()
