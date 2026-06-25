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
from package_files.config_utils import get_processed_dir, get_repo_root


REPO_ROOT = get_repo_root()
PROCESSED_DIR = get_processed_dir()
DATA_PATH = PROCESSED_DIR / "labeled_v2.parquet"
TABLES_DIR = REPO_ROOT / "results" / "tables_2026" / "wage_perk_interactions"
FIGURES_DIR = REPO_ROOT / "results" / "figures_2026" / "wage_perk_interactions"

MIN_AI_WAGE_POSTINGS = 10
MIN_INTERACTION_CELL_POSTINGS = 3
POOLED_PERIOD = "Pooled 2018-2025"
PERIOD_COMPARISON_FILTERS = {
    "Through 2022": lambda df: df["year"] <= 2022,
    "Post-2022": lambda df: df["year"] > 2022,
}

SAMPLE_FILTERS = {
    "Full sample": lambda df: pd.Series(True, index=df.index),
    "SMEs": lambda df: (~df["sp500"]) & (df["firm_posting_count"] < 10),
    "Large firms": lambda df: (~df["sp500"]) & (df["firm_posting_count"] >= 10),
    "S&P 500 firms": lambda df: df["sp500"],
}
SAMPLE_COLORS = {
    "Full sample": "#9467bd",
    "SMEs": "#1f77b4",
    "Large firms": "#2ca02c",
    "S&P 500 firms": "#d62728",
}
SAMPLE_MARKERS = {
    "Full sample": "D",
    "SMEs": "^",
    "Large firms": "s",
    "S&P 500 firms": "o",
}
FIRM_TYPE_SAMPLES = ["SMEs", "Large firms", "S&P 500 firms"]
GENAI_CUTOFF_YEAR = 2022.875


def add_genai_year_line(ax):
    """Mark the public GenAI inflection point on yearly plots."""
    ax.axvline(GENAI_CUTOFF_YEAR, color="black", linewidth=1.0, linestyle=":", alpha=0.75)


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
        "FIRM_POSTING_COUNT",
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
        "FIRM_POSTING_COUNT": "firm_posting_count",
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
    data["firm_posting_count"] = pd.to_numeric(data["firm_posting_count"], errors="coerce")
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
        "beta_perk_for_ai_role": np.nan,
        "beta_perk_for_ai_role_se": np.nan,
        "beta_perk_for_ai_role_lower_ci": np.nan,
        "beta_perk_for_ai_role_upper_ci": np.nan,
        "beta3_interaction": np.nan,
        "beta3_se": np.nan,
        "beta3_pvalue": np.nan,
        "beta3_lower_ci": np.nan,
        "beta3_upper_ci": np.nan,
        "beta_ai_role_with_perk": np.nan,
        "beta_ai_role_with_perk_se": np.nan,
        "beta_ai_role_with_perk_lower_ci": np.nan,
        "beta_ai_role_with_perk_upper_ci": np.nan,
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
        beta1 = model.params["ai_role"]
        beta1_se = model.bse["ai_role"]
        beta2 = model.params["perk"]
        beta2_se = model.bse["perk"]
        beta3 = model.params["ai_role:perk"]
        beta3_se = model.bse["ai_role:perk"]
        cov_params = model.cov_params()
        beta1_beta3_cov = cov_params.loc["ai_role", "ai_role:perk"]
        beta2_beta3_cov = cov_params.loc["perk", "ai_role:perk"]
        beta_ai_with_perk = beta1 + beta3
        beta_ai_with_perk_se = np.sqrt(beta1_se**2 + beta3_se**2 + 2 * beta1_beta3_cov)
        beta_perk_for_ai_role = beta2 + beta3
        beta_perk_for_ai_role_se = np.sqrt(beta2_se**2 + beta3_se**2 + 2 * beta2_beta3_cov)
        return {
            "sample": sample,
            "period": period,
            "perk": perk,
            "perk_label": benefits_labels_map[perk],
            "status": "ok",
            "detail": "",
            "nobs": int(model.nobs),
            "ai_wage_postings": int(model_data["ai_role"].sum()),
            "beta1_ai_role": beta1,
            "beta1_se": beta1_se,
            "beta1_pvalue": model.pvalues["ai_role"],
            "beta2_perk": beta2,
            "beta2_se": beta2_se,
            "beta2_pvalue": model.pvalues["perk"],
            "beta_perk_for_ai_role": beta_perk_for_ai_role,
            "beta_perk_for_ai_role_se": beta_perk_for_ai_role_se,
            "beta_perk_for_ai_role_lower_ci": beta_perk_for_ai_role - 1.96 * beta_perk_for_ai_role_se,
            "beta_perk_for_ai_role_upper_ci": beta_perk_for_ai_role + 1.96 * beta_perk_for_ai_role_se,
            "beta3_interaction": beta3,
            "beta3_se": beta3_se,
            "beta3_pvalue": model.pvalues["ai_role:perk"],
            "beta3_lower_ci": beta3 - 1.96 * beta3_se,
            "beta3_upper_ci": beta3 + 1.96 * beta3_se,
            "beta_ai_role_with_perk": beta_ai_with_perk,
            "beta_ai_role_with_perk_se": beta_ai_with_perk_se,
            "beta_ai_role_with_perk_lower_ci": beta_ai_with_perk - 1.96 * beta_ai_with_perk_se,
            "beta_ai_role_with_perk_upper_ci": beta_ai_with_perk + 1.96 * beta_ai_with_perk_se,
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
            rows.append(fit_interaction_model(sample_data, sample, POOLED_PERIOD, perk, True))
        for period, period_filter in PERIOD_COMPARISON_FILTERS.items():
            period_data = sample_data[period_filter(sample_data)]
            for perk in benefits4:
                rows.append(fit_interaction_model(period_data, sample, period, perk, True))
        for year in sorted(data["year"].unique()):
            year_data = sample_data[sample_data["year"] == year]
            for perk in benefits4:
                rows.append(fit_interaction_model(year_data, sample, str(year), perk, False))
    return pd.DataFrame(rows)


def plot_pooled_results(results):
    """Plot pooled beta3 interaction coefficients by sample and perk."""
    pooled = results[(results["period"] == POOLED_PERIOD) & (results["status"] == "ok")].copy()
    fig, axes = plt.subplots(2, 3, figsize=(16, 9), sharex=False)
    axes = axes.flatten()
    sample_order = FIRM_TYPE_SAMPLES

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
    yearly = results[
        results["period"].astype(str).str.match(r"^\d{4}$") & (results["status"] == "ok")
    ].copy()
    yearly["year"] = yearly["period"].astype(int)
    fig, axes = plt.subplots(2, 3, figsize=(16, 9), sharex=True)
    axes = axes.flatten()

    for ax, perk in zip(axes, benefits4):
        subset = yearly[yearly["perk"] == perk]
        for sample in FIRM_TYPE_SAMPLES:
            line_data = subset[subset["sample"] == sample].sort_values("year")
            if line_data.empty:
                continue
            color = SAMPLE_COLORS[sample]
            ax.plot(
                line_data["year"],
                line_data["beta3_interaction"],
                marker=SAMPLE_MARKERS[sample],
                markersize=4,
                linewidth=1.6,
                label=sample,
                color=color,
            )
        ax.axhline(0, color="gray", linewidth=0.8, linestyle="--")
        add_genai_year_line(ax)
        ax.set_ylim(-0.2, 0.2)
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


def plot_ai_role_premium_pooled(results):
    """Plot pooled AI-role wage coefficients from the interaction models."""
    pooled = results[(results["period"] == POOLED_PERIOD) & (results["status"] == "ok")].copy()
    fig, axes = plt.subplots(2, 3, figsize=(16, 9), sharex=False)
    axes = axes.flatten()
    sample_order = FIRM_TYPE_SAMPLES

    for ax, perk in zip(axes, benefits4):
        subset = (
            pooled[pooled["perk"] == perk]
            .set_index("sample")
            .reindex(sample_order)
            .dropna(subset=["beta1_ai_role"])
        )
        positions = np.arange(len(subset))
        ax.errorbar(
            subset["beta1_ai_role"],
            positions,
            xerr=[
                1.96 * subset["beta1_se"],
                1.96 * subset["beta1_se"],
            ],
            fmt="o",
            capsize=4,
            color="#0072B2",
        )
        ax.axvline(0, color="gray", linewidth=0.8, linestyle="--")
        ax.set_yticks(positions)
        ax.set_yticklabels(subset.index)
        ax.set_title(benefits_labels_map[perk])
        ax.set_xlabel("AI-Role Wage Coefficient (log points)")
        ax.grid(alpha=0.2)

    fig.suptitle(
        "Pooled AI-Role Wage Premium by Firm Category\n"
        "Interaction model coefficient: AI role, evaluated when perk = 0",
        fontsize=14,
    )
    plt.tight_layout()
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    output = FIGURES_DIR / "ai_role_wage_premium_beta1_pooled_by_firm_category.png"
    plt.savefig(output, bbox_inches="tight", dpi=300)
    plt.close()
    print(f"Saved: {output}")


def plot_ai_role_premium_yearly(results):
    """Plot yearly AI-role wage coefficients from the interaction models."""
    yearly = results[
        results["period"].astype(str).str.match(r"^\d{4}$") & (results["status"] == "ok")
    ].copy()
    yearly["year"] = yearly["period"].astype(int)
    fig, axes = plt.subplots(2, 3, figsize=(16, 9), sharex=True)
    axes = axes.flatten()
    sample_order = FIRM_TYPE_SAMPLES

    for ax, perk in zip(axes, benefits4):
        subset = yearly[yearly["perk"] == perk]
        for sample in sample_order:
            line = subset[subset["sample"] == sample].sort_values("year")
            if line.empty:
                continue
            color = SAMPLE_COLORS[sample]
            lower = line["beta1_ai_role"] - 1.96 * line["beta1_se"]
            upper = line["beta1_ai_role"] + 1.96 * line["beta1_se"]
            ax.fill_between(line["year"], lower, upper, alpha=0.10, color=color)
            ax.plot(
                line["year"],
                line["beta1_ai_role"],
                marker=SAMPLE_MARKERS[sample],
                label=sample,
                color=color,
                linewidth=1.6,
            )
        ax.axhline(0, color="gray", linewidth=0.8, linestyle="--")
        add_genai_year_line(ax)
        ax.set_title(benefits_labels_map[perk])
        ax.set_ylabel("AI-Role Wage Coefficient")
        ax.grid(alpha=0.2)

    axes[-1].legend(loc="upper left", bbox_to_anchor=(1.02, 1))
    fig.suptitle(
        "Yearly AI-Role Wage Premium by Firm Category\n"
        "Interaction model coefficient: AI role, evaluated when perk = 0",
        fontsize=14,
    )
    plt.tight_layout()
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    output = FIGURES_DIR / "ai_role_wage_premium_beta1_yearly_by_firm_category.png"
    plt.savefig(output, bbox_inches="tight", dpi=300)
    plt.close()
    print(f"Saved: {output}")


def plot_perk_wage_coefficient_pooled(results):
    """Plot pooled perk wage coefficients from the interaction models."""
    pooled = results[(results["period"] == POOLED_PERIOD) & (results["status"] == "ok")].copy()
    fig, axes = plt.subplots(2, 3, figsize=(16, 9), sharex=False)
    axes = axes.flatten()
    sample_order = FIRM_TYPE_SAMPLES

    for ax, perk in zip(axes, benefits4):
        subset = (
            pooled[pooled["perk"] == perk]
            .set_index("sample")
            .reindex(sample_order)
            .dropna(subset=["beta2_perk"])
        )
        positions = np.arange(len(subset))
        ax.errorbar(
            subset["beta2_perk"],
            positions,
            xerr=[
                1.96 * subset["beta2_se"],
                1.96 * subset["beta2_se"],
            ],
            fmt="o",
            capsize=4,
            color="#009E73",
        )
        ax.axvline(0, color="gray", linewidth=0.8, linestyle="--")
        ax.set_yticks(positions)
        ax.set_yticklabels(subset.index)
        ax.set_title(benefits_labels_map[perk])
        ax.set_xlabel("Perk Wage Coefficient (log points)")
        ax.grid(alpha=0.2)

    fig.suptitle(
        "Pooled Perk Wage Coefficient by Firm Category\n"
        "Interaction model coefficient: perk, evaluated when AI role = 0",
        fontsize=14,
    )
    plt.tight_layout()
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    output = FIGURES_DIR / "perk_wage_coefficient_beta2_pooled_by_firm_category.png"
    plt.savefig(output, bbox_inches="tight", dpi=300)
    plt.close()
    print(f"Saved: {output}")


def plot_perk_wage_coefficient_yearly(results):
    """Plot yearly perk wage coefficients from the interaction models."""
    yearly = results[
        results["period"].astype(str).str.match(r"^\d{4}$") & (results["status"] == "ok")
    ].copy()
    yearly["year"] = yearly["period"].astype(int)
    fig, axes = plt.subplots(2, 3, figsize=(16, 9), sharex=True)
    axes = axes.flatten()
    sample_order = FIRM_TYPE_SAMPLES

    for ax, perk in zip(axes, benefits4):
        subset = yearly[yearly["perk"] == perk]
        for sample in sample_order:
            line = subset[subset["sample"] == sample].sort_values("year")
            if line.empty:
                continue
            color = SAMPLE_COLORS[sample]
            lower = line["beta2_perk"] - 1.96 * line["beta2_se"]
            upper = line["beta2_perk"] + 1.96 * line["beta2_se"]
            ax.fill_between(line["year"], lower, upper, alpha=0.10, color=color)
            ax.plot(
                line["year"],
                line["beta2_perk"],
                marker=SAMPLE_MARKERS[sample],
                label=sample,
                color=color,
                linewidth=1.6,
            )
        ax.axhline(0, color="gray", linewidth=0.8, linestyle="--")
        add_genai_year_line(ax)
        ax.set_title(benefits_labels_map[perk])
        ax.set_ylabel("Perk Wage Coefficient")
        ax.grid(alpha=0.2)

    axes[-1].legend(loc="upper left", bbox_to_anchor=(1.02, 1))
    fig.suptitle(
        "Yearly Perk Wage Coefficient by Firm Category\n"
        "Interaction model coefficient: perk, evaluated when AI role = 0",
        fontsize=14,
    )
    plt.tight_layout()
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    output = FIGURES_DIR / "perk_wage_coefficient_beta2_yearly_by_firm_category.png"
    plt.savefig(output, bbox_inches="tight", dpi=300)
    plt.close()
    print(f"Saved: {output}")


def plot_heatmap(results, period=POOLED_PERIOD, output_name="wage_perk_interaction_beta3_heatmap.png"):
    """Plot a heatmap of beta3 coefficients for one pooled period."""
    pooled = results[
        (results["period"] == period) & (results["status"] == "ok")
    ].copy()

    perk_order = list(benefits4)
    perk_labels = [benefits_labels_map[p] for p in perk_order]
    sample_order = FIRM_TYPE_SAMPLES
    sample_labels = ["SMEs", "Large", "S&P 500"]

    matrix = np.full((len(perk_order), len(sample_order)), np.nan)
    pval_matrix = np.full_like(matrix, np.nan)
    for i, perk in enumerate(perk_order):
        for j, sample in enumerate(sample_order):
            row = pooled[(pooled["perk"] == perk) & (pooled["sample"] == sample)]
            if not row.empty:
                matrix[i, j] = row["beta3_interaction"].values[0]
                pval_matrix[i, j] = row["beta3_pvalue"].values[0]

    if np.isnan(matrix).all():
        print(f"No estimable heatmap results for {period}.")
        return

    vmax = np.nanmax(np.abs(matrix)) * 1.05
    if not np.isfinite(vmax) or vmax == 0:
        vmax = 0.01

    fig, ax = plt.subplots(figsize=(8, 5.5))
    # Diverging scale centered on zero: negative estimates are red, zero is
    # white, and positive estimates are blue.
    im = ax.imshow(matrix, cmap="RdBu", aspect="auto", vmin=-vmax, vmax=vmax)

    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            val = matrix[i, j]
            pval = pval_matrix[i, j]
            if np.isnan(val):
                continue
            stars = ""
            if pval < 0.01:
                stars = "***"
            elif pval < 0.05:
                stars = "**"
            elif pval < 0.1:
                stars = "*"
            label = f"{val:+.3f}{stars}"
            color = "white" if abs(val) > vmax * 0.55 else "black"
            ax.text(j, i, label, ha="center", va="center", fontsize=9, color=color)

    ax.set_xticks(range(len(sample_labels)))
    ax.set_xticklabels(sample_labels, fontsize=10)
    ax.set_xlabel("Firm Type", fontsize=11)
    ax.set_yticks(range(len(perk_labels)))
    ax.set_yticklabels(perk_labels, fontsize=10)
    ax.set_ylabel("Perk", fontsize=11)

    cbar = fig.colorbar(im, ax=ax, shrink=0.85, pad=0.02)
    cbar.set_label(r"$\beta_3$: AI Role × Perk Interaction (log points)", fontsize=10)

    ax.set_title(
        rf"Interaction Coefficient $\beta_3$ by Perk and Firm Type: {period}"
        "\nPositive = complementarity, Negative = substitution",
        fontsize=12,
    )

    plt.tight_layout()
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    output = FIGURES_DIR / output_name
    plt.savefig(output, bbox_inches="tight", dpi=300)
    plt.close()
    print(f"Saved: {output}")


def plot_period_comparison_heatmaps(results):
    """Plot pre- and post-2022 beta3 heatmaps in one shared figure."""
    periods = ["Through 2022", "Post-2022"]
    perk_order = list(benefits4)
    perk_labels = [benefits_labels_map[p] for p in perk_order]
    sample_order = FIRM_TYPE_SAMPLES
    sample_labels = ["SMEs", "Large", "S&P 500"]
    matrices = []
    pval_matrices = []

    for period in periods:
        subset = results[
            (results["period"] == period) & (results["status"] == "ok")
        ]
        matrix = np.full((len(perk_order), len(sample_order)), np.nan)
        pval_matrix = np.full_like(matrix, np.nan)
        for i, perk in enumerate(perk_order):
            for j, sample in enumerate(sample_order):
                row = subset[
                    (subset["perk"] == perk) & (subset["sample"] == sample)
                ]
                if not row.empty:
                    matrix[i, j] = row["beta3_interaction"].iloc[0]
                    pval_matrix[i, j] = row["beta3_pvalue"].iloc[0]
        matrices.append(matrix)
        pval_matrices.append(pval_matrix)

    vmax = max(np.nanmax(np.abs(matrix)) for matrix in matrices) * 1.05
    if not np.isfinite(vmax) or vmax == 0:
        vmax = 0.01

    fig, axes = plt.subplots(
        1, 2, figsize=(13, 6), sharey=True, constrained_layout=True
    )
    for ax, period, matrix, pval_matrix in zip(
        axes, periods, matrices, pval_matrices
    ):
        im = ax.imshow(
            matrix, cmap="RdBu", aspect="auto", vmin=-vmax, vmax=vmax
        )
        for i in range(matrix.shape[0]):
            for j in range(matrix.shape[1]):
                val = matrix[i, j]
                pval = pval_matrix[i, j]
                if np.isnan(val):
                    continue
                stars = ""
                if pval < 0.01:
                    stars = "***"
                elif pval < 0.05:
                    stars = "**"
                elif pval < 0.1:
                    stars = "*"
                color = "white" if abs(val) > vmax * 0.55 else "black"
                ax.text(
                    j,
                    i,
                    f"{val:+.3f}{stars}",
                    ha="center",
                    va="center",
                    fontsize=9,
                    color=color,
                )
        ax.set_xticks(range(len(sample_labels)))
        ax.set_xticklabels(sample_labels)
        ax.set_title(period, fontsize=12)

    axes[0].set_yticks(range(len(perk_labels)))
    axes[0].set_yticklabels(perk_labels)
    axes[0].set_ylabel("Perk", fontsize=11)
    fig.supxlabel("Firm Type", fontsize=11)
    fig.suptitle(
        r"AI Role $\times$ Perk Interaction Coefficient ($\beta_3$)"
        "\nPositive = complementarity, Negative = substitution",
        fontsize=13,
    )
    cbar = fig.colorbar(im, ax=axes, shrink=0.85, pad=0.02)
    cbar.set_label(r"$\beta_3$ (log points)", fontsize=10)

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    output = FIGURES_DIR / "wage_perk_interaction_beta3_heatmap_period_comparison.png"
    plt.savefig(output, bbox_inches="tight", dpi=300)
    plt.close()
    print(f"Saved: {output}")


def plot_yearly_single_perk(results, perk="REMOTE_KW"):
    """Plot a single-perk time series of beta3 by firm type with confidence ribbons."""
    yearly = results[
        results["period"].astype(str).str.match(r"^\d{4}$")
        & (results["status"] == "ok")
        & (results["perk"] == perk)
    ].copy()
    yearly["year"] = yearly["period"].astype(int)

    sample_order = FIRM_TYPE_SAMPLES
    fig, ax = plt.subplots(figsize=(9, 5.5))

    for sample in sample_order:
        line = yearly[yearly["sample"] == sample].sort_values("year")
        if line.empty:
            continue
        color = SAMPLE_COLORS[sample]
        ax.fill_between(
            line["year"],
            line["beta3_lower_ci"],
            line["beta3_upper_ci"],
            alpha=0.12,
            color=color,
        )
        ax.plot(
            line["year"],
            line["beta3_interaction"],
            marker=SAMPLE_MARKERS[sample],
            markersize=5,
            label=sample.replace(" firms", ""),
            color=color,
            linewidth=1.8,
        )

    ax.axhline(0, color="gray", linewidth=0.8, linestyle="--")
    add_genai_year_line(ax)
    ax.set_xlabel("Year", fontsize=11)
    ax.set_ylabel(r"$\beta_3$ Interaction Coefficient", fontsize=11)
    perk_label = benefits_labels_map[perk]
    ax.set_title(
        rf"Development of $\beta_3$ (AI × {perk_label} Interaction)"
        "\nPositive = Complementarity, Negative = Substitution",
        fontsize=12,
    )
    ax.legend(title="Firm Type", fontsize=9, title_fontsize=10)
    ax.grid(alpha=0.2)

    plt.tight_layout()
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    slug = perk.lower().replace(" ", "_")
    output = FIGURES_DIR / f"wage_perk_interaction_beta3_timeseries_{slug}.png"
    plt.savefig(output, bbox_inches="tight", dpi=300)
    plt.close()
    print(f"Saved: {output}")


def plot_all(results):
    """Generate all plots from a results DataFrame."""
    plot_ai_role_premium_pooled(results)
    plot_ai_role_premium_yearly(results)
    plot_perk_wage_coefficient_pooled(results)
    plot_perk_wage_coefficient_yearly(results)
    plot_pooled_results(results)
    plot_yearly_results(results)
    plot_heatmap(results)
    plot_period_comparison_heatmaps(results)
    plot_yearly_single_perk(results, perk="REMOTE_KW")


def main():
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--plot-only",
        action="store_true",
        help="Skip model fitting; regenerate plots from existing results CSV.",
    )
    args = parser.parse_args()

    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    output = TABLES_DIR / "wage_perk_interaction_results.csv"

    if args.plot_only:
        if not output.exists():
            raise FileNotFoundError(f"Results CSV not found: {output}")
        results = pd.read_csv(output)
        print(f"Loaded {len(results)} rows from {output}")
        plot_all(results)
        return

    data = load_data()
    print(f"Wage sample rows: {len(data):,}")
    results = run_models(data)
    results.to_csv(output, index=False)
    print(f"Saved: {output}")
    print(results.groupby(["period", "status"]).size().to_string())
    plot_all(results)


if __name__ == "__main__":
    main()
