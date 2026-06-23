#!/usr/bin/env python3
"""
Plot Figure 1 by firm category using saved wage betas and raw demand data.

This script avoids re-estimating regressions by reading the quarterly wage
betas CSV produced by the full-sample run.  The demand panel (% AI roles)
is computed on-the-fly from the parquet since it's a simple groupby.

Usage:
    uv run python scripts/plot_firm_category_figure1.py
"""

from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.append(str(Path(__file__).parent.parent / "src"))
from package_files.config_utils import get_processed_dir, get_repo_root

REPO_ROOT = get_repo_root()
PROCESSED_DIR = get_processed_dir()
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
    df.loc[fpc < 10, FIRM_CATEGORY] = "SMEs"
    df.loc[fpc >= 10, FIRM_CATEGORY] = "Large firms"
    df.loc[sp500, FIRM_CATEGORY] = "S&P 500 firms"
    df[FIRM_CATEGORY] = pd.Categorical(
        df[FIRM_CATEGORY], categories=FIRM_CATEGORY_ORDER, ordered=True
    )
    return df


def add_genai_quarter_line(ax, period_index, label="Nov. 2022"):
    if GENAI_CUTOFF not in period_index:
        return
    cutoff_x = list(period_index).index(GENAI_CUTOFF) + 2 / 3
    ax.axvline(cutoff_x, color="black", linewidth=1.0, linestyle=":", alpha=0.75, label=label)


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


def main():
    # --- Load demand data from parquet (fast groupby, no models) ---
    parquet_path = PROCESSED_DIR / "labeled_v2.parquet"
    print(f"Loading data from {parquet_path} ...")
    cols = ["POSTED", "AI ROLE", "SP500", "FIRM_POSTING_COUNT"]
    df = pd.read_parquet(parquet_path, columns=cols)
    df["POSTED"] = pd.to_datetime(df["POSTED"])
    df["QUARTER"] = df["POSTED"].dt.to_period("Q")
    df = add_firm_category(df)

    category_df = df.dropna(subset=[FIRM_CATEGORY])
    demand = (
        category_df.groupby(["QUARTER", FIRM_CATEGORY], observed=False)["AI ROLE"]
        .mean()
        .mul(100)
        .reset_index(name="ai_role_share")
    )
    demand_wide = (
        demand.pivot(index="QUARTER", columns=FIRM_CATEGORY, values="ai_role_share")
        .sort_index()
        .reindex(columns=FIRM_CATEGORY_ORDER)
    )

    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    demand_path = TABLES_DIR / "quarterly_ai_demand_by_firm_category.csv"
    demand_wide.to_csv(demand_path)
    print(f"Saved demand table: {demand_path}")

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
    add_genai_quarter_line(demand_ax, demand_wide.index)
    demand_ax.legend(title=None, fontsize=11)
    demand_ax.grid(alpha=0.25)

    plotted = wage_betas[wage_betas["status"] == "ok"].copy()
    plotted["x"] = plotted["QUARTER"].map(quarter_position_map)
    for category in FIRM_CATEGORY_ORDER:
        line = plotted[plotted[FIRM_CATEGORY] == category].sort_values("QUARTER")
        if line.empty:
            continue
        post_2021 = line["QUARTER"] >= pd.Period("2022Q1", freq="Q")
        wage_ax.fill_between(
            line.loc[post_2021, "x"],
            line.loc[post_2021, "lower_ci"],
            line.loc[post_2021, "upper_ci"],
            color=FIRM_CATEGORY_COLORS[category],
            alpha=0.12,
            linewidth=0,
        )
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
    add_genai_quarter_line(wage_ax, demand_wide.index)
    post_ci = plotted[plotted["QUARTER"] >= pd.Period("2022Q1", freq="Q")]
    set_tight_symmetric_ylim(
        wage_ax,
        plotted["ai_role_beta"],
        post_ci["lower_ci"],
        post_ci["upper_ci"],
    )
    wage_ax.set_ylabel("AI-Role Wage Coefficient\n(log points)", fontsize=13)
    wage_ax.set_title("Adjusted AI-Skills Wage Premium by Firm Category", fontsize=15)
    wage_ax.grid(alpha=0.25)
    wage_ax.legend(title=None, fontsize=11)

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
