#!/usr/bin/env python3
"""
Industry-Year Coefficient Analysis

For each industry-year cell, estimates:
  A) OLS wage model:  log_salary ~ AI_ROLE + controls  → extracts AI wage premium beta
  B) Logit perk model: perk ~ AI_ROLE + controls       → extracts AI perk premium beta

Then produces scatterplots comparing wage betas (x) against perk betas (y) across
industry-year cells to test whether perks complement wages in attracting AI talent.
"""

import sys
from pathlib import Path

import pandas as pd
import numpy as np
import statsmodels.api as sm
import matplotlib.pyplot as plt

# Add src directory to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from package_files.benefits_defns import benefits_labels_map

# ─── Configuration ──────────────────────────────────────────────────────────────

industry = "NAICS_2022_2_NAME"
year_col = "YEAR"
education = "MIN_EDULEVELS_NAME"
experience = "EXPERIENCE_BUCKET"

benefits = ["EDU_ASSISTANCE", "PAID LEAVE", "HEALTH_WELLBEING", "PARENTAL_LEAVE", "CULTURE", "REMOTE_KW"]

MIN_CELL_OBS = 50       # Minimum observations per industry-year cell
MIN_POSITIVE = 10       # Minimum positive outcomes for logit to be estimable
OUTLIER_BETA_CAP = 5    # Drop perk betas with |beta| > this (quasi-separation artifacts)

RESULTS_DIR = Path(__file__).parent.parent / "results" / "figures_2026"
TABLES_DIR = Path(__file__).parent.parent / "results" / "tables_2026"


# ─── Data Loading ───────────────────────────────────────────────────────────────

def load_data():
    """Load and prepare the job-level dataset."""
    base = Path(__file__).parent.parent / "data" / "processed"
    data = pd.read_parquet(base / "labeled_v1.parquet")

    # Cast AI ROLE to int (stored as bool in parquet)
    data["AI ROLE"] = data["AI ROLE"].astype(int)

    # Collapse small industries
    industry_counts = data[industry].value_counts()
    small = industry_counts[industry_counts < 30].index
    data[industry] = data[industry].replace(small, "Other")

    return data


# ─── Step A: Wage Betas ─────────────────────────────────────────────────────────

def estimate_wage_betas(data):
    """
    For each industry-year cell, run OLS: LOG_SALARY ~ AI_ROLE + education + experience.
    Returns DataFrame with columns: industry, year, wage_airole_beta, wage_se, wage_pvalue, n_obs.
    """
    # Subset to rows with valid salary
    df = data.dropna(subset=["LOG_SALARY", "AI ROLE", education, experience]).copy()

    results = []
    for (ind, yr), group in df.groupby([industry, year_col]):
        if len(group) < MIN_CELL_OBS:
            continue

        try:
            y = group["LOG_SALARY"]
            X = group[["AI ROLE"]].copy()

            # Add education dummies
            edu_dummies = pd.get_dummies(group[education], prefix="edu", drop_first=True, dtype=int)
            X = pd.concat([X, edu_dummies], axis=1)

            # Add experience dummies
            exp_dummies = pd.get_dummies(group[experience], prefix="exp", drop_first=True, dtype=int)
            X = pd.concat([X, exp_dummies], axis=1)

            X = sm.add_constant(X)
            model = sm.OLS(y, X).fit()

            results.append({
                "industry": ind,
                "year": yr,
                "wage_airole_beta": model.params["AI ROLE"],
                "wage_se": model.bse["AI ROLE"],
                "wage_pvalue": model.pvalues["AI ROLE"],
                "n_obs": int(model.nobs),
            })
        except Exception:
            continue

    return pd.DataFrame(results)


# ─── Step B: Perk Betas ─────────────────────────────────────────────────────────

def estimate_perk_betas(data):
    """
    For each industry-year cell and each benefit, run logit: perk ~ AI_ROLE + education + experience.
    Returns DataFrame with columns: industry, year, benefit, perk_airole_beta, perk_se, perk_pvalue, n_obs.
    """
    df = data.dropna(subset=["AI ROLE", education, experience]).copy()

    results = []
    for (ind, yr), group in df.groupby([industry, year_col]):
        if len(group) < MIN_CELL_OBS:
            continue

        # Pre-compute control dummies once per cell
        edu_dummies = pd.get_dummies(group[education], prefix="edu", drop_first=True, dtype=int)
        exp_dummies = pd.get_dummies(group[experience], prefix="exp", drop_first=True, dtype=int)
        X_base = pd.concat([group[["AI ROLE"]].reset_index(drop=True),
                            edu_dummies.reset_index(drop=True),
                            exp_dummies.reset_index(drop=True)], axis=1)
        X_base = sm.add_constant(X_base)

        for benefit in benefits:
            y = group[benefit].reset_index(drop=True)
            valid = y.notna()
            y_valid = y[valid]
            X_valid = X_base[valid]

            # Check estimability
            if len(y_valid) < MIN_CELL_OBS:
                continue
            if y_valid.sum() < MIN_POSITIVE or (len(y_valid) - y_valid.sum()) < MIN_POSITIVE:
                continue

            try:
                model = sm.Logit(y_valid, X_valid).fit(disp=0, maxiter=100)
                if not model.mle_retvals["converged"]:
                    continue

                results.append({
                    "industry": ind,
                    "year": yr,
                    "benefit": benefit,
                    "perk_airole_beta": model.params["AI ROLE"],
                    "perk_se": model.bse["AI ROLE"],
                    "perk_pvalue": model.pvalues["AI ROLE"],
                    "n_obs": int(model.nobs),
                })
            except Exception:
                continue

    return pd.DataFrame(results)


# ─── Unconditional Models (AI ROLE only) ─────────────────────────────────────────

def estimate_wage_betas_unconditional(data):
    """OLS: LOG_SALARY ~ AI_ROLE (no controls) per industry-year cell."""
    df = data.dropna(subset=["LOG_SALARY", "AI ROLE"]).copy()

    results = []
    for (ind, yr), group in df.groupby([industry, year_col]):
        if len(group) < MIN_CELL_OBS:
            continue
        try:
            y = group["LOG_SALARY"]
            X = sm.add_constant(group[["AI ROLE"]])
            model = sm.OLS(y, X).fit()
            results.append({
                "industry": ind,
                "year": yr,
                "wage_airole_beta": model.params["AI ROLE"],
                "wage_se": model.bse["AI ROLE"],
                "wage_pvalue": model.pvalues["AI ROLE"],
                "n_obs": int(model.nobs),
            })
        except Exception:
            continue

    return pd.DataFrame(results)


def estimate_perk_betas_unconditional(data):
    """Logit: perk ~ AI_ROLE (no controls) per industry-year cell."""
    df = data.dropna(subset=["AI ROLE"]).copy()

    results = []
    for (ind, yr), group in df.groupby([industry, year_col]):
        if len(group) < MIN_CELL_OBS:
            continue

        X_base = sm.add_constant(group[["AI ROLE"]].reset_index(drop=True))

        for benefit in benefits:
            y = group[benefit].reset_index(drop=True)
            valid = y.notna()
            y_valid = y[valid]
            X_valid = X_base[valid]

            if len(y_valid) < MIN_CELL_OBS:
                continue
            if y_valid.sum() < MIN_POSITIVE or (len(y_valid) - y_valid.sum()) < MIN_POSITIVE:
                continue

            try:
                model = sm.Logit(y_valid, X_valid).fit(disp=0, maxiter=100)
                if not model.mle_retvals["converged"]:
                    continue
                results.append({
                    "industry": ind,
                    "year": yr,
                    "benefit": benefit,
                    "perk_airole_beta": model.params["AI ROLE"],
                    "perk_se": model.bse["AI ROLE"],
                    "perk_pvalue": model.pvalues["AI ROLE"],
                    "n_obs": int(model.nobs),
                })
            except Exception:
                continue

    return pd.DataFrame(results)


# ─── Outlier Filtering ──────────────────────────────────────────────────────────

def filter_outlier_betas(perk_df, cap=OUTLIER_BETA_CAP):
    """Drop perk beta estimates with implausibly large magnitudes."""
    if perk_df.empty:
        return perk_df
    before = len(perk_df)
    filtered = perk_df[perk_df["perk_airole_beta"].abs() <= cap].copy()
    dropped = before - len(filtered)
    if dropped:
        print(f"  Filtered {dropped} outlier perk betas (|beta| > {cap})")
    return filtered


# ─── Step C: Scatterplots ───────────────────────────────────────────────────────

def plot_wage_vs_perk_airole_betas(wage_df, perk_df, label="controlled", output_dir=None):
    """
    For each perk, scatterplot AI wage beta (x) against AI perk beta (y)
    across industry-year cells. Label distinguishes model variants in filenames/titles.
    """
    if output_dir is None:
        output_dir = RESULTS_DIR
    output_dir.mkdir(parents=True, exist_ok=True)

    if wage_df.empty or perk_df.empty:
        print("No coefficients to plot.")
        return None

    perk_filtered = filter_outlier_betas(perk_df)
    merged = perk_filtered.merge(wage_df, on=["industry", "year"], suffixes=("_perk", "_wage"))

    title_suffix = "(with controls)" if label == "controlled" else "(unconditional)"

    _, axes = plt.subplots(2, 3, figsize=(16, 10))
    axes = axes.flatten()

    for i, benefit in enumerate(benefits):
        ax = axes[i]
        subset = merged[merged["benefit"] == benefit]

        if subset.empty:
            ax.set_title(benefits_labels_map.get(benefit, benefit))
            ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes)
            continue

        ax.scatter(subset["wage_airole_beta"], subset["perk_airole_beta"], alpha=0.6, s=40, edgecolors="k", linewidths=0.3)

        # Fit and plot OLS trend line
        if len(subset) >= 5:
            z = np.polyfit(subset["wage_airole_beta"], subset["perk_airole_beta"], 1)
            p = np.poly1d(z)
            x_range = np.linspace(subset["wage_airole_beta"].min(), subset["wage_airole_beta"].max(), 50)
            ax.plot(x_range, p(x_range), "r--", linewidth=1.5, alpha=0.7)

            corr = subset["wage_airole_beta"].corr(subset["perk_airole_beta"])
            ax.annotate(f"r = {corr:.2f}  n = {len(subset)}", xy=(0.05, 0.92), xycoords="axes fraction", fontsize=10)

        ax.set_xlabel("AI Wage Premium (log points)")
        ax.set_ylabel("AI Perk Premium (log-odds)")
        ax.set_title(benefits_labels_map.get(benefit, benefit))
        ax.axhline(0, color="gray", linewidth=0.5, linestyle="--")
        ax.axvline(0, color="gray", linewidth=0.5, linestyle="--")

    plt.suptitle(f"AI Wage Premium vs. AI Perk Premium {title_suffix}\n(by Industry-Year Cell)", fontsize=14, y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.95])

    out_path = output_dir / f"wage_vs_perk_airole_betas_{label}.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out_path}")
    return out_path


def save_coefficient_table(wage_df, perk_df, label="controlled", output_dir=None):
    """Save the estimated coefficients as a CSV for reference."""
    if output_dir is None:
        output_dir = TABLES_DIR
    output_dir.mkdir(parents=True, exist_ok=True)

    # Wage betas
    wage_out = output_dir / f"industry_year_wage_airole_betas_{label}.csv"
    wage_df.to_csv(wage_out, index=False)
    print(f"Saved: {wage_out}")

    # Perk betas
    perk_out = output_dir / f"industry_year_perk_airole_betas_{label}.csv"
    perk_df.to_csv(perk_out, index=False)
    print(f"Saved: {perk_out}")

    # Summary: correlation between wage and perk betas per benefit (outlier-filtered)
    if wage_df.empty or perk_df.empty:
        print("No coefficients to summarize.")
        return
    perk_filtered = filter_outlier_betas(perk_df)
    merged = perk_filtered.merge(wage_df, on=["industry", "year"])
    summary_rows = []
    for benefit in benefits:
        sub = merged[merged["benefit"] == benefit]
        if len(sub) >= 3:
            corr = sub["wage_airole_beta"].corr(sub["perk_airole_beta"])
            summary_rows.append({
                "benefit": benefits_labels_map.get(benefit, benefit),
                "n_cells": len(sub),
                "correlation": round(corr, 3),
                "mean_wage_airole_beta": round(sub["wage_airole_beta"].mean(), 4),
                "mean_perk_airole_beta": round(sub["perk_airole_beta"].mean(), 4),
            })

    summary_df = pd.DataFrame(summary_rows)
    summary_out = output_dir / f"wage_perk_airole_beta_correlation_{label}.csv"
    summary_df.to_csv(summary_out, index=False)
    print(f"Saved: {summary_out}")
    print("\n", summary_df.to_string(index=False))


# ─── Main ───────────────────────────────────────────────────────────────────────

def main():
    print("=" * 80)
    print("INDUSTRY-YEAR COEFFICIENT ANALYSIS")
    print("=" * 80)

    print("\nLoading data...")
    data = load_data()
    print(f"Dataset: {data.shape[0]:,} rows, {data[industry].nunique()} industries, "
          f"years {data[year_col].min()}-{data[year_col].max()}")

    # ── Controlled models (with education + experience) ──
    print("\n" + "=" * 60)
    print("MODEL 1: CONTROLLED (AI ROLE + education + experience)")
    print("=" * 60)

    print("\n--- Wage betas (OLS) ---")
    wage_ctrl = estimate_wage_betas(data)
    print(f"Estimated {len(wage_ctrl)} industry-year wage betas")

    print("\n--- Perk betas (Logit) ---")
    perk_ctrl = estimate_perk_betas(data)
    print(f"Estimated {len(perk_ctrl)} industry-year-perk betas")

    print("\n--- Outputs ---")
    save_coefficient_table(wage_ctrl, perk_ctrl, label="controlled")
    plot_wage_vs_perk_airole_betas(wage_ctrl, perk_ctrl, label="controlled")

    # ── Unconditional models (AI ROLE only) ──
    print("\n" + "=" * 60)
    print("MODEL 2: UNCONDITIONAL (AI ROLE only)")
    print("=" * 60)

    print("\n--- Wage betas (OLS) ---")
    wage_uncond = estimate_wage_betas_unconditional(data)
    print(f"Estimated {len(wage_uncond)} industry-year wage betas")

    print("\n--- Perk betas (Logit) ---")
    perk_uncond = estimate_perk_betas_unconditional(data)
    print(f"Estimated {len(perk_uncond)} industry-year-perk betas")

    print("\n--- Outputs ---")
    save_coefficient_table(wage_uncond, perk_uncond, label="unconditional")
    plot_wage_vs_perk_airole_betas(wage_uncond, perk_uncond, label="unconditional")

    print("\nDone.")


if __name__ == "__main__":
    main()
