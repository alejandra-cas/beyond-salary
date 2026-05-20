#!/usr/bin/env python3
"""
AI Classification Threshold Robustness (Step 6)

Tests whether the main results hold under more conservative AI role definitions:
  - 1+ AI skills  (baseline: current "AI ROLE" flag)
  - 2+ AI skills  (AI_ROLE_2PLUS)
  - 3+ AI skills  (AI_ROLE_3PLUS)

For each threshold, reruns the main model spec (P1 M2: Year + Industry + Education
+ Experience FE) for all 6 benefits and extracts the AI ROLE coefficient.

Outputs:
- results/tables_2026/ai_threshold_robustness.csv   — coefficients table
- results/figures_2026/robustness/ai_threshold_robustness.png  — coefficient plot
"""

import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent / "src"))
from package_files.logit_model import run_logit_model
from package_files.benefits_defns import (
    benefits4,
    benefits4_labels,
    benefit_colors,
    industry,
    education,
    year,
    experience,
)

plt.rcParams.update({"font.size": 13})

OUTPUT_FIGS = Path("results/figures_2026/robustness")
OUTPUT_TABLES = Path("results/tables_2026")

THRESHOLDS = {
    "1+ AI skills": ("AI ROLE", 1),
    "2+ AI skills": ("AI_ROLE_2PLUS", 2),
    "3+ AI skills": ("AI_ROLE_3PLUS", 3),
}


def load_data():
    _base = Path(__file__).parent.parent / "data" / "processed"
    df = pd.read_parquet(_base / "labeled_v1.parquet")
    print(f"Loaded {len(df):,} rows")

    # Verify AI_SKILL_COUNT is present
    if "AI_SKILL_COUNT" not in df.columns:
        raise ValueError(
            "AI_SKILL_COUNT column not found. Re-run scripts/prepare_data.py "
            "and scripts/label_benefits.py to regenerate labeled_v1.parquet."
        )

    # Create alternative threshold flags
    df["AI_ROLE_2PLUS"] = df["AI_SKILL_COUNT"] >= 2
    df["AI_ROLE_3PLUS"] = df["AI_SKILL_COUNT"] >= 3

    # Consolidate rare industries (mirrors regression_models.py)
    industry_counts = df[industry].value_counts()
    small_industries = industry_counts[industry_counts < 30].index
    df[industry] = df[industry].replace(small_industries, "Other")

    print("\nAI threshold sample sizes:")
    for label, (col, _) in THRESHOLDS.items():
        n = df[col].sum()
        pct = n / len(df) * 100
        print(f"  {label:20s}: {n:,} AI postings ({pct:.1f}%)")

    return df


def run_threshold_models(df):
    """Run the P1 M2 spec for each threshold × benefit combination."""
    results = []

    for threshold_label, (predictor_col, _) in THRESHOLDS.items():
        print(f"\n{'='*60}")
        print(f"Threshold: {threshold_label}  (predictor: {predictor_col})")
        print(f"{'='*60}")

        for benefit, benefit_label in zip(benefits4, benefits4_labels):
            print(f"\n  {benefit_label}")
            model = run_logit_model(
                df.copy(),
                dependent=benefit,
                predictor=predictor_col,
                cat_controls=[year, industry, education, experience],
                ref_category={
                    education: "No Education Listed",
                    experience: "None Listed",
                },
                get_vif=False,
            )

            if model == "Error" or isinstance(model, str):
                results.append({
                    "threshold": threshold_label,
                    "benefit": benefit,
                    "benefit_label": benefit_label,
                    "coef": np.nan,
                    "se": np.nan,
                    "pvalue": np.nan,
                    "nobs": np.nan,
                    "converged": False,
                })
            else:
                results.append({
                    "threshold": threshold_label,
                    "benefit": benefit,
                    "benefit_label": benefit_label,
                    "coef": model.params.get(predictor_col, np.nan),
                    "se": model.bse.get(predictor_col, np.nan),
                    "pvalue": model.pvalues.get(predictor_col, np.nan),
                    "nobs": int(model.nobs),
                    "converged": bool(model.converged),
                })

    return pd.DataFrame(results)


def save_table(results_df):
    os.makedirs(OUTPUT_TABLES, exist_ok=True)

    # Add significance stars
    def stars(p):
        if pd.isna(p):
            return ""
        if p < 0.01:
            return "***"
        if p < 0.05:
            return "**"
        if p < 0.1:
            return "*"
        return ""

    results_df["stars"] = results_df["pvalue"].apply(stars)
    results_df["coef_str"] = results_df.apply(
        lambda r: f"{r['coef']:.3f}{r['stars']}" if not pd.isna(r["coef"]) else "—",
        axis=1,
    )
    results_df["se_str"] = results_df["se"].apply(
        lambda s: f"({s:.3f})" if not pd.isna(s) else ""
    )

    out = OUTPUT_TABLES / "ai_threshold_robustness.csv"
    results_df.to_csv(out, index=False)
    print(f"\nSaved: {out}")

    # Pretty-print wide table
    wide = results_df.pivot_table(
        index="benefit_label",
        columns="threshold",
        values="coef_str",
        aggfunc="first",
    )[list(THRESHOLDS.keys())]
    print("\nAI ROLE coefficient by threshold (P1 M2 spec):")
    print(wide.to_string())

    return results_df


def plot_results(results_df):
    os.makedirs(OUTPUT_FIGS, exist_ok=True)

    results_df["lower"] = results_df["coef"] - 1.96 * results_df["se"]
    results_df["upper"] = results_df["coef"] + 1.96 * results_df["se"]

    threshold_labels = list(THRESHOLDS.keys())
    markers = ["o", "s", "^"]
    threshold_colors = ["#333333", "#888888", "#bbbbbb"]

    fig, axes = plt.subplots(2, 3, figsize=(16, 9), sharey=False)
    axes = axes.flatten()

    for i, (benefit, benefit_label) in enumerate(zip(benefits4, benefits4_labels)):
        ax = axes[i]
        sub = results_df[results_df["benefit"] == benefit].set_index("threshold")

        for j, (thresh, marker, color) in enumerate(
            zip(threshold_labels, markers, threshold_colors)
        ):
            if thresh not in sub.index:
                continue
            row = sub.loc[thresh]
            if pd.isna(row["coef"]):
                continue
            ax.errorbar(
                j,
                row["coef"],
                yerr=[[row["coef"] - row["lower"]], [row["upper"] - row["coef"]]],
                fmt=marker,
                color=benefit_colors[benefit],
                alpha=0.4 + 0.3 * j,
                markersize=9,
                capsize=4,
                label=thresh,
            )

        ax.axhline(0, color="black", linewidth=0.7, linestyle="--")
        ax.set_title(benefit_label, fontsize=12)
        ax.set_xticks(range(len(threshold_labels)))
        ax.set_xticklabels(threshold_labels, fontsize=9, rotation=15, ha="right")
        ax.set_ylabel("Log-odds (AI ROLE)", fontsize=10)
        ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.2f"))

    fig.suptitle(
        "AI Role Coefficient by Classification Threshold (P1 M2 spec, 95% CI)",
        fontsize=14,
        y=1.01,
    )
    plt.tight_layout()

    out = OUTPUT_FIGS / "ai_threshold_robustness.png"
    plt.savefig(out, bbox_inches="tight", dpi=150)
    plt.close()
    print(f"Saved: {out}")


def main():
    print("Loading data...")
    df = load_data()

    print("\nRunning models across thresholds...")
    results_df = run_threshold_models(df)

    print("\nSaving table...")
    results_df = save_table(results_df)

    print("\nGenerating plot...")
    plot_results(results_df)

    print("\nDone.")


if __name__ == "__main__":
    main()
