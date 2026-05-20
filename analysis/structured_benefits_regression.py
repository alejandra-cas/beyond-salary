#!/usr/bin/env python3
"""
Regression models for structured-field benefit categories (Step 9).

Runs the P1 M2 spec (Year + Industry + Education + Experience FE) for
the new structured-field benefits: S_FLEX_WORK, S_PROF_DEV,
S_HEALTH_WELLNESS, S_REMOTE.

Outputs:
- results/tables_2026/structured_benefits_regression.csv
- results/figures_2026/regression/structured_benefits_ai_role_coef.png
"""

import sys
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent / "src"))
from package_files.logit_model import run_logit_model
from package_files.benefits_defns import (
    industry,
    education,
    year,
    experience,
)

OUTPUT_TABLES = Path("results/tables_2026")
OUTPUT_FIGS = Path("results/figures_2026/regression")

STRUCTURED_BENEFITS = {
    "S_FLEX_WORK": "Flexible Work Schedules",
    "S_PROF_DEV": "Professional Development",
    "S_HEALTH_WELLNESS": "Health & Wellness Programs",
    "S_REMOTE": "Remote Work (Structured)",
}


def load_data():
    _base = Path(__file__).parent.parent / "data" / "processed"
    df = pd.read_parquet(_base / "labeled_v1.parquet")
    print(f"Loaded {len(df):,} rows")

    # Consolidate rare industries
    ind_counts = df[industry].value_counts()
    small = ind_counts[ind_counts < 30].index
    df[industry] = df[industry].replace(small, "Other")

    return df


def run_models(df):
    results = []

    for benefit, label in STRUCTURED_BENEFITS.items():
        print(f"\n{'='*60}")
        print(f"{label} ({benefit})")
        print(f"{'='*60}")

        model = run_logit_model(
            df.copy(),
            dependent=benefit,
            predictor="AI ROLE",
            cat_controls=[year, industry, education, experience],
            ref_category={
                education: "No Education Listed",
                experience: "None Listed",
            },
            get_vif=False,
        )

        if model == "Error" or isinstance(model, str):
            results.append({
                "benefit": benefit,
                "benefit_label": label,
                "coef": np.nan,
                "se": np.nan,
                "pvalue": np.nan,
                "nobs": np.nan,
                "converged": False,
            })
        else:
            coef = model.params.get("AI ROLE", np.nan)
            se = model.bse.get("AI ROLE", np.nan)
            pval = model.pvalues.get("AI ROLE", np.nan)
            results.append({
                "benefit": benefit,
                "benefit_label": label,
                "coef": coef,
                "se": se,
                "pvalue": pval,
                "nobs": int(model.nobs),
                "converged": bool(model.converged),
                "pseudo_r2": model.prsquared,
            })

    return pd.DataFrame(results)


def save_results(results_df):
    os.makedirs(OUTPUT_TABLES, exist_ok=True)

    def stars(p):
        if pd.isna(p): return ""
        if p < 0.01: return "***"
        if p < 0.05: return "**"
        if p < 0.1: return "*"
        return ""

    results_df["stars"] = results_df["pvalue"].apply(stars)
    results_df["coef_str"] = results_df.apply(
        lambda r: f"{r['coef']:.3f}{r['stars']}" if not pd.isna(r["coef"]) else "—",
        axis=1,
    )

    out = OUTPUT_TABLES / "structured_benefits_regression.csv"
    results_df.to_csv(out, index=False)
    print(f"\nSaved: {out}")

    print("\nAI ROLE coefficient (P1 M2 spec) for structured-field benefits:")
    print(results_df[["benefit_label", "coef_str", "se", "pvalue", "nobs", "pseudo_r2"]].to_string(index=False))


def plot_results(results_df):
    os.makedirs(OUTPUT_FIGS, exist_ok=True)

    plot_df = results_df.copy()
    plot_df = plot_df.dropna(subset=["coef", "se"])

    if plot_df.empty:
        print("No converged coefficients to plot.")
        return

    fig, ax = plt.subplots(figsize=(12, 6))
    x = np.arange(len(plot_df))

    ax.errorbar(
        x,
        plot_df["coef"].values,
        yerr=1.96 * plot_df["se"].values,
        fmt="o",
        color="#466eb4",
        ecolor="#466eb4",
        capsize=4,
        markersize=8,
    )
    ax.axhline(0, color="grey", linewidth=0.8)
    ax.set_xticks(x, labels=plot_df["benefit_label"].tolist())
    ax.set_xticklabels(plot_df["benefit_label"].tolist(), rotation=45, fontsize=14, ha="right")
    ax.tick_params(axis="y", labelsize=14)
    ax.set_xlabel(None)
    ax.set_ylabel("Log-Odds Coefficient (with 95% CI)", fontsize=16)
    plt.tight_layout()

    out = OUTPUT_FIGS / "structured_benefits_ai_role_coef.png"
    plt.savefig(out, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out}")


def main():
    print("Loading data...")
    df = load_data()

    print("\nRunning logit models...")
    results_df = run_models(df)

    print("\nSaving results...")
    save_results(results_df)
    plot_results(results_df)

    print("\nDone.")


if __name__ == "__main__":
    main()
