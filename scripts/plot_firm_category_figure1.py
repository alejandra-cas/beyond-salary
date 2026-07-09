#!/usr/bin/env python3
"""
Plot Figure 1 by firm category using saved descriptive tables.

This script avoids re-estimating regressions or reprocessing posting-level
data by reading the saved quarterly demand and wage-beta CSVs.

Usage:
    uv run python scripts/plot_firm_category_figure1.py
"""

import os
from pathlib import Path
import sys

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.append(str(Path(__file__).parent.parent / "src"))
from package_files.config_utils import get_processed_dir, get_repo_root

REPO_ROOT = get_repo_root()
TABLES_DIR = REPO_ROOT / "results" / "tables_2026" / "descriptive"
FIG_DIR = REPO_ROOT / "results" / "figures_2026" / "descriptive" / "firm_categories"

FIRM_CATEGORY = "FIRM_CATEGORY"
FIRM_CATEGORY_ORDER = ["SMEs", "Large firms", "S&P 500 firms"]
FIRM_CATEGORY_COLORS = {
    "SMEs": "#1f77b4",
    "Large firms": "#2ca02c",
    "S&P 500 firms": "#d62728",
}
FIRM_CATEGORY_MARKERS = {
    "SMEs": "^",
    "Large firms": "s",
    "S&P 500 firms": "o",
}
GENAI_CUTOFF = pd.Period("2022Q4", freq="Q")
FIGURE1_DEMAND_INSET_BOUNDS = [0.04, 0.52, 0.34, 0.38]
FIGURE1_WAGE_INSET_BOUNDS = [0.62, 0.08, 0.34, 0.38]


def add_firm_category(df):
    """Add mutually exclusive firm-size categories."""
    sp500 = df.get("SP500", False)
    if not isinstance(sp500, pd.Series):
        sp500 = pd.Series(False, index=df.index)
    sp500 = sp500.fillna(False).astype(bool)

    if "FIRM_POSTING_COUNT" in df.columns:
        fpc = pd.to_numeric(df["FIRM_POSTING_COUNT"], errors="coerce")
    else:
        fpc = pd.Series(np.nan, index=df.index)

    df[FIRM_CATEGORY] = pd.NA
    df.loc[fpc < 50, FIRM_CATEGORY] = "SMEs"
    df.loc[fpc >= 50, FIRM_CATEGORY] = "Large firms"
    df.loc[sp500, FIRM_CATEGORY] = "S&P 500 firms"
    df[FIRM_CATEGORY] = pd.Categorical(
        df[FIRM_CATEGORY], categories=FIRM_CATEGORY_ORDER, ordered=True
    )
    return df


def add_chatgpt_quarter_line(ax, period_index, add_label=False):
    """Mark ChatGPT's November 2022 release with a prominent dotted line."""
    if GENAI_CUTOFF not in period_index:
        return
    cutoff_x = list(period_index).index(GENAI_CUTOFF) + 2 / 3
    ax.axvline(
        cutoff_x,
        color="black",
        linewidth=2.0,
        linestyle=":",
        alpha=0.9,
        zorder=5,
    )
    if add_label:
        ax.text(
            cutoff_x + 0.15,
            0.34,
            "ChatGPT",
            transform=ax.get_xaxis_transform(),
            rotation=90,
            ha="left",
            va="bottom",
            fontsize=11,
            fontweight="bold",
            color="black",
        )


def set_tight_symmetric_ylim(ax, values, lower_values=None, upper_values=None, pad=0.08, floor=0.02):
    pieces = [pd.Series(values).dropna()]
    if lower_values is not None:
        pieces.append(pd.Series(lower_values).dropna())
    if upper_values is not None:
        pieces.append(pd.Series(upper_values).dropna())
    combined = pd.concat(pieces, ignore_index=True)
    if combined.empty:
        return
    limit = max(float(combined.abs().max()) * (1 + pad), floor)
    ax.set_ylim(-limit, limit)


def set_zero_based_ylim(ax, values, upper_values=None, pad=0.10, floor=0.02, min_upper=None):
    pieces = [pd.Series(values).dropna()]
    if upper_values is not None:
        pieces.append(pd.Series(upper_values).dropna())
    combined = pd.concat(pieces, ignore_index=True)
    if combined.empty:
        return
    upper = max(float(combined.max()) * (1 + pad), floor)
    if min_upper is not None:
        upper = max(upper, min_upper)
    ax.set_ylim(0, upper)


def add_demand_inset(ax, demand_wide):
    """Add the Figure 1-style before/after demand inset from saved quarters."""
    period_means = {
        "Before": demand_wide[demand_wide.index.year <= 2022].mean(),
        "After": demand_wide[demand_wide.index.year > 2022].mean(),
    }
    inset = ax.inset_axes(FIGURE1_DEMAND_INSET_BOUNDS)
    periods = ["Before", "After"]
    x = np.arange(len(periods))
    bar_width = 0.22

    for i, category in enumerate(FIRM_CATEGORY_ORDER):
        offset = (i - 1) * bar_width
        inset.bar(
            x + offset,
            [period_means[p][category] for p in periods],
            width=bar_width,
            color=FIRM_CATEGORY_COLORS[category],
            alpha=0.85,
            label=category,
        )

    inset.set_xticks(x)
    inset.set_xticklabels(periods, fontsize=9)
    inset.set_ylabel("% AI", fontsize=8)
    inset.tick_params(axis="both", labelsize=8)
    inset.legend(fontsize=6.5, loc="upper left", framealpha=0.7)
    inset.grid(axis="y", alpha=0.18)
    inset.text(
        0.98,
        0.97,
        "AI Vacancy Share\nby Firm Type",
        transform=inset.transAxes,
        fontsize=8,
        ha="right",
        va="top",
        fontstyle="italic",
        color="#333333",
    )


def summarize_wage_betas_by_period(wage_betas):
    """Inverse-variance pool saved quarterly coefficients before and after 2022."""
    data = wage_betas[wage_betas["status"] == "ok"].copy()
    data["period"] = np.where(data["QUARTER"].dt.year <= 2022, "Before", "After")
    rows = []
    for (category, period), group in data.groupby([FIRM_CATEGORY, "period"]):
        valid = group[group["ai_role_se"] > 0].copy()
        if valid.empty:
            continue
        weights = 1 / valid["ai_role_se"].pow(2)
        estimate = np.average(valid["ai_role_beta"], weights=weights)
        standard_error = np.sqrt(1 / weights.sum())
        rows.append(
            {
                FIRM_CATEGORY: category,
                "period": period,
                "ai_role_beta": estimate,
                "lower_ci": estimate - 1.96 * standard_error,
                "upper_ci": estimate + 1.96 * standard_error,
            }
        )
    return pd.DataFrame(rows)


def add_wage_inset(ax, wage_betas):
    """Add the Figure 1-style before/after wage-premium inset."""
    period_betas = summarize_wage_betas_by_period(wage_betas)
    if period_betas.empty:
        return
    inset = ax.inset_axes(FIGURE1_WAGE_INSET_BOUNDS)
    periods = ["Before", "After"]
    x = np.arange(len(periods))
    bar_width = 0.22

    for i, category in enumerate(FIRM_CATEGORY_ORDER):
        category_data = period_betas[
            period_betas[FIRM_CATEGORY] == category
        ].set_index("period")
        values = [category_data.loc[p, "ai_role_beta"] for p in periods]
        yerr_lo = [
            category_data.loc[p, "ai_role_beta"] - category_data.loc[p, "lower_ci"]
            for p in periods
        ]
        yerr_hi = [
            category_data.loc[p, "upper_ci"] - category_data.loc[p, "ai_role_beta"]
            for p in periods
        ]
        offset = (i - 1) * bar_width
        inset.bar(
            x + offset,
            values,
            yerr=[yerr_lo, yerr_hi],
            width=bar_width,
            color=FIRM_CATEGORY_COLORS[category],
            alpha=0.85,
            capsize=2,
            label=category,
        )

    inset.axhline(0, color="gray", linewidth=0.7, linestyle="--")
    inset.set_xticks(x)
    inset.set_xticklabels(periods, fontsize=9)
    inset.set_ylabel("Coef.", fontsize=8)
    inset.tick_params(axis="both", labelsize=8)
    inset.legend(fontsize=6.5, loc="upper left", framealpha=0.7)
    inset.grid(axis="y", alpha=0.18)
    inset.text(
        0.98,
        0.97,
        "Wage Premium\nby Firm Type",
        transform=inset.transAxes,
        fontsize=8,
        ha="right",
        va="top",
        fontstyle="italic",
        color="#333333",
    )


def load_demand_wide():
    """Load cached demand by firm category, or compute it from minimal parquet columns."""
    demand_path = TABLES_DIR / "quarterly_ai_demand_by_firm_category.csv"
    if demand_path.exists():
        demand_wide = pd.read_csv(demand_path)
        demand_wide["QUARTER"] = demand_wide["QUARTER"].apply(
            lambda x: pd.Period(x, freq="Q")
        )
        print(f"Loaded demand table: {demand_path}")
    else:
        processed_path = get_processed_dir() / "labeled_v2.parquet"
        print(f"Demand table not found; computing from {processed_path}")
        columns = ["POSTED", "AI ROLE", "SP500", "FIRM_POSTING_COUNT"]
        try:
            demand_df = pd.read_parquet(processed_path, columns=columns)
        except OSError:
            import polars as pl

            demand_df = pl.scan_parquet(processed_path).select(columns).collect().to_pandas()
        demand_df["POSTED"] = pd.to_datetime(demand_df["POSTED"])
        demand_df["QUARTER"] = demand_df["POSTED"].dt.to_period("Q")
        demand_df = add_firm_category(demand_df).dropna(subset=[FIRM_CATEGORY])
        demand_wide = (
            demand_df.groupby(["QUARTER", FIRM_CATEGORY], observed=False)["AI ROLE"]
            .mean()
            .mul(100)
            .reset_index(name="ai_role_share")
            .pivot(index="QUARTER", columns=FIRM_CATEGORY, values="ai_role_share")
            .reset_index()
        )
        TABLES_DIR.mkdir(parents=True, exist_ok=True)
        demand_wide.assign(QUARTER=demand_wide["QUARTER"].astype(str)).to_csv(
            demand_path,
            index=False,
        )
        print(f"Saved demand table: {demand_path}")

    return (
        demand_wide.set_index("QUARTER")
        .sort_index()
        .reindex(columns=FIRM_CATEGORY_ORDER)
    )


def main():
    # --- Load pre-computed descriptive tables ---
    demand_wide = load_demand_wide()

    # --- Load pre-computed wage betas ---
    betas_path = TABLES_DIR / "quarterly_ai_wage_betas_by_firm_category.csv"
    if not betas_path.exists():
        raise FileNotFoundError(f"Wage betas CSV not found: {betas_path}")
    wage_betas = pd.read_csv(betas_path)
    wage_betas["QUARTER"] = wage_betas["QUARTER"].apply(lambda x: pd.Period(x, freq="Q"))
    print(f"Loaded {len(wage_betas)} wage beta rows from {betas_path}")

    # --- Plot ---
    fig, (demand_ax, wage_ax) = plt.subplots(nrows=2, ncols=1, figsize=(11, 9), sharex=True)
    quarter_positions = np.arange(len(demand_wide.index))
    quarter_position_map = {q: i for i, q in enumerate(demand_wide.index)}

    for category in FIRM_CATEGORY_ORDER:
        if category not in demand_wide:
            continue
        demand_ax.plot(
            quarter_positions,
            demand_wide[category],
            marker=FIRM_CATEGORY_MARKERS[category],
            markersize=3,
            linewidth=1.6,
            label=category,
            color=FIRM_CATEGORY_COLORS[category],
        )

    demand_ax.set_ylabel("% AI Roles", fontsize=15)
    demand_ax.set_title("Demand for AI Skills by Firm Category", fontsize=15)
    add_chatgpt_quarter_line(demand_ax, demand_wide.index, add_label=True)
    set_zero_based_ylim(demand_ax, demand_wide.stack())
    demand_ax.legend(title=None, fontsize=11, loc="lower right")
    demand_ax.grid(alpha=0.25)
    add_demand_inset(demand_ax, demand_wide)

    plotted = wage_betas[wage_betas["status"] == "ok"].copy()
    plotted["x"] = plotted["QUARTER"].map(quarter_position_map)
    for category in FIRM_CATEGORY_ORDER:
        line = plotted[plotted[FIRM_CATEGORY] == category].sort_values("QUARTER")
        if line.empty:
            continue
        wage_ax.plot(
            line["x"],
            line["ai_role_beta"],
            marker=FIRM_CATEGORY_MARKERS[category],
            markersize=4,
            linewidth=1.5,
            label=category,
            color=FIRM_CATEGORY_COLORS[category],
        )

    wage_ax.axhline(0, color="gray", linewidth=0.8, linestyle="--")
    add_chatgpt_quarter_line(wage_ax, demand_wide.index)
    post_ci = plotted[plotted["QUARTER"] >= pd.Period("2022Q1", freq="Q")]
    set_tight_symmetric_ylim(
        wage_ax,
        plotted["ai_role_beta"],
        post_ci["lower_ci"],
        post_ci["upper_ci"],
    )
    wage_ax.set_ylim(bottom=-0.1)
    wage_ax.set_ylabel("AI-Role Wage Coefficient\n(log points)", fontsize=13)
    wage_ax.set_title("Adjusted AI-Skills Wage Premium by Firm Category", fontsize=15)
    wage_ax.grid(alpha=0.25)
    wage_ax.legend(title=None, fontsize=11, loc="upper left")
    add_wage_inset(wage_ax, wage_betas)

    tick_positions = list(range(0, len(demand_wide.index), 4))
    wage_ax.set_xticks(tick_positions)
    wage_ax.set_xticklabels([str(demand_wide.index[i]) for i in tick_positions], rotation=45)
    wage_ax.set_xlabel("Quarter", fontsize=13)

    plt.tight_layout()
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    output_path = FIG_DIR / "figure1_ai_demand_wage_beta_by_firm_category.png"
    plt.savefig(output_path, bbox_inches="tight", dpi=300)
    plt.close()
    print(f"Saved: {output_path}")


if __name__ == "__main__":
    main()
