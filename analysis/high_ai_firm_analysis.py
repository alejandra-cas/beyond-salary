#!/usr/bin/env python3
"""
High-AI Firm Analysis (Step 5)

Identifies firms with a high share of AI postings (top quartile by % AI ROLE)
and examines whether perk prevalence is uniformly high across all their postings
(AI and non-AI) or if there is meaningful within-firm variation.

This addresses the concern that large tech firms drive the results simply by
offering perks to everyone, not by specifically rewarding AI roles.

Outputs:
- results/figures_2026/high_ai_firms/within_firm_perk_diff.png
- results/figures_2026/high_ai_firms/perk_prevalence_by_firm_tier.png
- results/tables_2026/high_ai_firm/high_ai_firm_summary.csv
- results/tables_2026/high_ai_firm/within_firm_perk_diff.csv
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).parent.parent / "src"))
from package_files.benefits_defns import (
    benefits4,
    benefits4_labels,
    benefits_labels_map,
    benefit_colors,
    firm,
)
from package_files.config_utils import get_processed_dir, get_repo_root

plt.rcParams.update({"font.size": 14})

REPO_ROOT = get_repo_root()
PROCESSED_DIR = get_processed_dir()
OUTPUT_FIGS = REPO_ROOT / "results" / "figures_2026" / "high_ai_firms"
OUTPUT_TABLES = REPO_ROOT / "results" / "tables_2026" / "high_ai_firm"


def load_data():
    """Load the configured analysis dataset."""
    print("Loading analysis dataset...")

    data_path = PROCESSED_DIR / "labeled_v2.parquet"
    if not data_path.exists():
        print(f"Warning: {data_path} not found. Please update the path.")
        return None

    df = pd.read_parquet(data_path)
    print(f"Loaded {len(df):,} rows from {data_path}")
    return df


def compute_firm_stats(df):
    """Compute per-firm AI share and perk prevalence for AI and non-AI postings."""
    # Only keep firms with non-null COMPANY identifier
    df = df[df[firm].notna()].copy()
    df[firm] = df[firm].astype(str)

    # Firm-level summary: total postings, AI share
    firm_totals = df.groupby(firm).agg(
        total_postings=("AI ROLE", "count"),
        ai_share=("AI ROLE", "mean"),
    )

    # Keep firms with at least 30 postings so shares are stable
    firm_totals = firm_totals[firm_totals["total_postings"] >= 30]
    df = df[df[firm].isin(firm_totals.index)]

    print(f"Firms with >=30 postings: {len(firm_totals):,}")
    print(f"Postings in qualifying firms: {len(df):,}")

    return df, firm_totals


def assign_firm_tiers(firm_totals):
    """Assign firms to AI-share tiers.

    Many firms have 0% AI share, so equal-frequency quartiles produce degenerate
    bin edges. Instead we use meaningful fixed thresholds:
      - Zero-AI:  0% AI share
      - Low-AI:   0% < share <= 25th pct of firms with any AI
      - Mid-AI:   25–75th pct
      - High-AI:  top 25th pct (roughly top quartile among firms that post AI roles)
    """
    ai_firms = firm_totals[firm_totals["ai_share"] > 0]["ai_share"]
    p25 = ai_firms.quantile(0.25)
    p75 = ai_firms.quantile(0.75)

    def _tier(s):
        if s == 0:
            return "Zero-AI (0%)"
        elif s <= p25:
            return f"Low-AI (0–{p25*100:.0f}%]"
        elif s <= p75:
            return f"Mid-AI ({p25*100:.0f}–{p75*100:.0f}%]"
        else:
            return f"High-AI (>{p75*100:.0f}%)"

    firm_totals["ai_share_quartile"] = firm_totals["ai_share"].apply(_tier)

    print(f"\nAI-share tier cutpoints (among firms with any AI postings):")
    print(f"  p25 = {p25*100:.1f}%,  p75 = {p75*100:.1f}%")
    print("\nFirms per tier:")
    print(firm_totals["ai_share_quartile"].value_counts().to_string())
    return firm_totals, p25, p75


def within_firm_perk_diff(df, firm_totals):
    """
    For each firm, compute perk prevalence among AI and non-AI postings separately.
    Returns a long dataframe with within-firm differences.
    """
    df = df.merge(firm_totals[["ai_share_quartile"]], left_on=firm, right_index=True)

    rows = []
    for ben in benefits4:
        grp = (
            df.groupby([firm, "AI ROLE"])[ben]
            .mean()
            .unstack("AI ROLE")
            .rename(columns={False: "non_ai_perk_rate", True: "ai_perk_rate"})
        )
        grp = grp.dropna(subset=["non_ai_perk_rate", "ai_perk_rate"])
        grp["within_firm_diff"] = grp["ai_perk_rate"] - grp["non_ai_perk_rate"]
        grp["benefit"] = ben
        grp = grp.merge(
            firm_totals[["ai_share", "ai_share_quartile", "total_postings"]],
            left_index=True,
            right_index=True,
        )
        rows.append(grp.reset_index())

    result = pd.concat(rows, ignore_index=True)
    return result


def plot_within_firm_diff_by_quartile(diff_df):
    """
    Box plot: within-firm AI vs non-AI perk difference, split by firm AI-share quartile.
    One panel per benefit.
    """
    fig, axes = plt.subplots(2, 3, figsize=(16, 10), sharey=False)
    axes = axes.flatten()

    for i, ben in enumerate(benefits4):
        ax = axes[i]
        sub = diff_df[diff_df["benefit"] == ben].copy()

        sns.boxplot(
            data=sub,
            x="ai_share_quartile",
            y="within_firm_diff",
            ax=ax,
            color=benefit_colors[ben],
            width=0.5,
            fliersize=2,
        )
        ax.axhline(0, color="black", linewidth=0.8, linestyle="--")
        ax.set_title(benefits_labels_map[ben], fontsize=13)
        ax.set_xlabel("Firm AI-share quartile")
        ax.set_ylabel("AI − Non-AI perk rate (within firm)")
        ax.tick_params(axis="x", labelsize=9)

    fig.suptitle(
        "Within-Firm Perk Difference (AI vs Non-AI Roles) by Firm AI-Share Quartile",
        fontsize=14,
        y=1.01,
    )
    plt.tight_layout()
    plt.savefig(OUTPUT_FIGS / "within_firm_perk_diff_by_quartile.png", bbox_inches="tight")
    plt.close()
    print("Saved: within_firm_perk_diff_by_quartile.png")


def plot_perk_prevalence_by_firm_tier(df, firm_totals):
    """
    Bar chart: average perk rate for AI and non-AI postings, separately for
    high-AI firms and zero-AI firms.
    """
    df2 = df.merge(firm_totals[["ai_share_quartile"]], left_on=firm, right_index=True)

    all_tiers = firm_totals["ai_share_quartile"].unique()
    zero_tier = next(t for t in all_tiers if t.startswith("Zero"))
    high_tier = next(t for t in all_tiers if t.startswith("High"))
    tiers = [zero_tier, high_tier]
    tier_labels = [f"Zero-AI firms\n({zero_tier})", f"High-AI firms\n({high_tier})"]
    role_labels = {False: "Non-AI roles", True: "AI roles"}

    fig, axes = plt.subplots(1, 2, figsize=(16, 6), sharey=True)

    for ax, tier, tier_label in zip(axes, tiers, tier_labels):
        sub = df2[df2["ai_share_quartile"] == tier]
        rates = sub.groupby("AI ROLE")[benefits4].mean() * 100
        rates = rates.reindex([True, False])

        x = np.arange(len(benefits4))
        width = 0.35
        for j, (role, row) in enumerate(rates.iterrows()):
            colors = [benefit_colors[b] for b in benefits4]
            alpha = 1.0 if role else 0.5
            ax.bar(
                x + j * width,
                row[benefits4],
                width,
                color=colors,
                alpha=alpha,
                label=role_labels[role],
            )

        ax.set_title(tier_label, fontsize=13)
        ax.set_xticks(x + width / 2)
        ax.set_xticklabels(benefits4_labels, rotation=40, ha="right", fontsize=10)
        ax.set_ylabel("% jobs with benefit")
        ax.set_ylim(0, 60)

        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor="gray", alpha=1.0, label="AI roles"),
            Patch(facecolor="gray", alpha=0.5, label="Non-AI roles"),
        ]
        ax.legend(handles=legend_elements)

    fig.suptitle(
        "Perk Prevalence for AI vs Non-AI Roles: Low-AI vs High-AI Firms",
        fontsize=14,
    )
    plt.tight_layout()
    plt.savefig(OUTPUT_FIGS / "perk_prevalence_by_firm_tier.png", bbox_inches="tight")
    plt.close()
    print("Saved: perk_prevalence_by_firm_tier.png")


def print_summary_stats(diff_df, firm_totals):
    """Print and save summary statistics."""
    print("\n" + "=" * 60)
    print("SUMMARY: Mean within-firm AI−Non-AI perk difference by quartile")
    print("=" * 60)

    summary = (
        diff_df.groupby(["benefit", "ai_share_quartile"])["within_firm_diff"]
        .agg(["mean", "median", "count"])
        .reset_index()
    )
    summary["mean"] = (summary["mean"] * 100).round(2)
    summary["median"] = (summary["median"] * 100).round(2)
    summary.columns = ["benefit", "quartile", "mean_diff_pp", "median_diff_pp", "n_firms"]
    print(summary.to_string(index=False))

    summary.to_csv(OUTPUT_TABLES / "within_firm_perk_diff.csv", index=False)
    print(f"\nSaved: {OUTPUT_TABLES / 'within_firm_perk_diff.csv'}")

    # High-level firm tier summary
    tier_summary = (
        firm_totals.groupby("ai_share_quartile")
        .agg(n_firms=("total_postings", "count"), median_ai_share=("ai_share", "median"))
        .reset_index()
    )
    tier_summary["median_ai_share"] = (tier_summary["median_ai_share"] * 100).round(1)
    print("\nFirm tier summary:")
    print(tier_summary.to_string(index=False))
    tier_summary.to_csv(OUTPUT_TABLES / "high_ai_firm_summary.csv", index=False)
    print(f"Saved: {OUTPUT_TABLES / 'high_ai_firm_summary.csv'}")


def main():
    os.makedirs(OUTPUT_FIGS, exist_ok=True)
    os.makedirs(OUTPUT_TABLES, exist_ok=True)

    print("Loading data...")
    df = load_data()
    if df is None:
        print("Error: Could not load data. Please check the data path.")
        return

    print("\nComputing firm statistics...")
    df, firm_totals = compute_firm_stats(df)

    print("\nAssigning firm AI-share tiers...")
    firm_totals, p25, p75 = assign_firm_tiers(firm_totals)

    print("\nComputing within-firm perk differences...")
    diff_df = within_firm_perk_diff(df, firm_totals)

    print("\nGenerating plots...")
    plot_within_firm_diff_by_quartile(diff_df)
    plot_perk_prevalence_by_firm_tier(df, firm_totals)

    print("\nSummary statistics...")
    print_summary_stats(diff_df, firm_totals)

    print("\nDone.")


if __name__ == "__main__":
    main()
