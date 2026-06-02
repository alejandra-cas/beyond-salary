#!/usr/bin/env python3
"""Estimate wage-perk interaction models for AI and non-AI postings."""

from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

sys.path.append(str(Path(__file__).parent.parent / "src"))
from package_files.benefits_defns import benefits4, benefits_labels_map


DATA_PATH = Path(__file__).parent.parent / "data" / "processed" / "labeled_v1.parquet"
TABLES_DIR = Path(__file__).parent.parent / "results" / "tables_2026"
FIGURES_DIR = Path(__file__).parent.parent / "results" / "figures_2026" / "wage_perk_interactions"

MIN_AI_WAGE_POSTINGS = 10
MIN_INTERACTION_CELL_POSTINGS = 3

SAMPLE_FILTERS = {
    "Full sample": lambda df: pd.Series(True, index=df.index),
    "Small firms": lambda df: df["firm_size_bucket"] == "Small (<=2)",
    "Medium firms": lambda df: df["firm_size_bucket"] == "Medium (3-9)",
    "Large firms": lambda df: df["firm_size_bucket"] == "Large (>=10)",
    "S&P 500 firms": lambda df: df["sp500"],
}


def load_data():
    """Load the wage sample and normalize model column names."""
    data = pd.read_parquet(DATA_PATH)
    required = [
        "LOG_SALARY",
        "AI ROLE",
        "MIN_EDULEVELS_NAME",
        "EXPERIENCE_BUCKET",
        "NAICS_2022_3_DIGIT",
        "STATE_NAME",
        "YEAR",
        "FIRM_SIZE_BUCKET",
        "SP500",
    ] + benefits4
    missing = [column for column in required if column not in data.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    rename = {
        "LOG_SALARY": "log_salary",
        "AI ROLE": "ai_role",
        "MIN_EDULEVELS_NAME": "education",
        "EXPERIENCE_BUCKET": "experience",
        "NAICS_2022_3_DIGIT": "naics3",
        "STATE_NAME": "state",
        "YEAR": "year",
        "FIRM_SIZE_BUCKET": "firm_size_bucket",
        "SP500": "sp500",
    }
    data = data.rename(columns=rename)
    data = data.replace([np.inf, -np.inf], np.nan)
    data = data.dropna(
        subset=["log_salary", "ai_role", "education", "experience", "naics3", "state", "year"]
    ).copy()
    data["ai_role"] = data["ai_role"].astype(int)
    data["sp500"] = data["sp500"].astype(bool)
    return data


def status_row(sample, period, perk, data, status, detail=""):
    """Create a result row for skipped or failed models."""
    return {
        "sample": sample,
        "period": period,
        "perk": perk,
        "perk_label": benefits_labels_map[perk],
        "status": status,
        "detail": detail,
        "nobs": len(data),
        "ai_wage_postings": int(data["ai_role"].sum()) if "ai_role" in data else 0,
        "beta1_ai_role": np.nan,
        "beta1_se": np.nan,
        "beta1_pvalue": np.nan,
        "beta2_perk": np.nan,
        "beta2_se": np.nan,
        "beta2_pvalue": np.nan,
        "beta3_interaction": np.nan,
        "beta3_se": np.nan,
        "beta3_pvalue": np.nan,
        "beta3_lower_ci": np.nan,
        "beta3_upper_ci": np.nan,
        "r_squared": np.nan,
    }


def fit_interaction_model(data, sample, period, perk, include_year_fe):
    """Fit one log-wage interaction model and return a tidy result row."""
    model_data = data.dropna(subset=[perk]).copy()
    model_data["perk"] = model_data[perk].astype(int)

    if sample == "S&P 500 firms" and model_data.empty:
        return status_row(sample, period, perk, model_data, "skipped_no_sp500_matches")
    if int(model_data["ai_role"].sum()) < MIN_AI_WAGE_POSTINGS:
        return status_row(sample, period, perk, model_data, "skipped_sparse_ai_wage_postings")

    cell_counts = model_data.groupby(["ai_role", "perk"]).size()
    expected_cells = pd.MultiIndex.from_product([[0, 1], [0, 1]])
    if (cell_counts.reindex(expected_cells, fill_value=0) < MIN_INTERACTION_CELL_POSTINGS).any():
        return status_row(sample, period, perk, model_data, "skipped_sparse_interaction_cells")

    fixed_effects = [
        "C(education)",
        "C(experience)",
        "C(naics3)",
        "C(state)",
    ]
    if include_year_fe:
        fixed_effects.append("C(year)")
    formula = "log_salary ~ ai_role + perk + ai_role:perk + " + " + ".join(fixed_effects)

    try:
        model = smf.ols(formula, data=model_data).fit(cov_type="HC1")
        beta3 = model.params["ai_role:perk"]
        beta3_se = model.bse["ai_role:perk"]
        return {
            "sample": sample,
            "period": period,
            "perk": perk,
            "perk_label": benefits_labels_map[perk],
            "status": "ok",
            "detail": "",
            "nobs": int(model.nobs),
            "ai_wage_postings": int(model_data["ai_role"].sum()),
            "beta1_ai_role": model.params["ai_role"],
            "beta1_se": model.bse["ai_role"],
            "beta1_pvalue": model.pvalues["ai_role"],
            "beta2_perk": model.params["perk"],
            "beta2_se": model.bse["perk"],
            "beta2_pvalue": model.pvalues["perk"],
            "beta3_interaction": beta3,
            "beta3_se": beta3_se,
            "beta3_pvalue": model.pvalues["ai_role:perk"],
            "beta3_lower_ci": beta3 - 1.96 * beta3_se,
            "beta3_upper_ci": beta3 + 1.96 * beta3_se,
            "r_squared": model.rsquared,
        }
    except Exception as error:
        return status_row(sample, period, perk, model_data, "error", str(error))


def run_models(data):
    """Run pooled and calendar-year interaction models for every sample and perk."""
    rows = []
    for sample, sample_filter in SAMPLE_FILTERS.items():
        sample_data = data[sample_filter(data)].copy()
        for perk in benefits4:
            rows.append(fit_interaction_model(sample_data, sample, "Pooled 2018-2025", perk, True))
        for year in sorted(data["year"].unique()):
            year_data = sample_data[sample_data["year"] == year]
            for perk in benefits4:
                rows.append(fit_interaction_model(year_data, sample, str(year), perk, False))
    return pd.DataFrame(rows)


def plot_pooled_results(results):
    """Plot pooled beta3 interaction coefficients by sample and perk."""
    pooled = results[(results["period"] == "Pooled 2018-2025") & (results["status"] == "ok")].copy()
    fig, axes = plt.subplots(2, 3, figsize=(16, 9), sharex=False)
    axes = axes.flatten()
    sample_order = list(SAMPLE_FILTERS)

    for ax, perk in zip(axes, benefits4):
        subset = pooled[pooled["perk"] == perk].set_index("sample").reindex(sample_order).dropna(subset=["beta3_interaction"])
        positions = np.arange(len(subset))
        ax.errorbar(
            subset["beta3_interaction"],
            positions,
            xerr=[
                subset["beta3_interaction"] - subset["beta3_lower_ci"],
                subset["beta3_upper_ci"] - subset["beta3_interaction"],
            ],
            fmt="o",
            capsize=4,
        )
        ax.axvline(0, color="gray", linewidth=0.8, linestyle="--")
        ax.set_yticks(positions)
        ax.set_yticklabels(subset.index)
        ax.set_title(benefits_labels_map[perk])
        ax.set_xlabel("AI Role × Perk Coefficient (log points)")
        ax.grid(alpha=0.2)

    plt.tight_layout()
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    output = FIGURES_DIR / "wage_perk_interaction_beta3_pooled.png"
    plt.savefig(output, bbox_inches="tight", dpi=300)
    plt.close()
    print(f"Saved: {output}")


def plot_yearly_results(results):
    """Plot yearly beta3 interaction coefficients for estimable models."""
    yearly = results[(results["period"] != "Pooled 2018-2025") & (results["status"] == "ok")].copy()
    yearly["year"] = yearly["period"].astype(int)
    fig, axes = plt.subplots(2, 3, figsize=(16, 9), sharex=True)
    axes = axes.flatten()

    for ax, perk in zip(axes, benefits4):
        subset = yearly[yearly["perk"] == perk]
        for sample in SAMPLE_FILTERS:
            line_data = subset[subset["sample"] == sample].sort_values("year")
            if line_data.empty:
                continue
            ax.plot(line_data["year"], line_data["beta3_interaction"], marker="o", label=sample)
        ax.axhline(0, color="gray", linewidth=0.8, linestyle="--")
        ax.set_title(benefits_labels_map[perk])
        ax.set_ylabel("AI Role × Perk Coefficient")
        ax.grid(alpha=0.2)

    axes[-1].legend(loc="upper left", bbox_to_anchor=(1.02, 1))
    plt.tight_layout()
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    output = FIGURES_DIR / "wage_perk_interaction_beta3_yearly.png"
    plt.savefig(output, bbox_inches="tight", dpi=300)
    plt.close()
    print(f"Saved: {output}")


def main():
    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    data = load_data()
    print(f"Wage sample rows: {len(data):,}")
    results = run_models(data)
    output = TABLES_DIR / "wage_perk_interaction_results.csv"
    results.to_csv(output, index=False)
    print(f"Saved: {output}")
    print(results.groupby(["period", "status"]).size().to_string())
    plot_pooled_results(results)
    plot_yearly_results(results)


if __name__ == "__main__":
    main()
