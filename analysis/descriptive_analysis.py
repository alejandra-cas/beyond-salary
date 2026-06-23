#!/usr/bin/env python3
"""
This script generates 5 visualizations from the descriptive statistics analysis:
1. % AI roles over time plot (overall and industry average)
2. Difference in percent jobs with benefit, AI-non-AI
3. Percent of jobs with each keyword benefit for AI and non-AI roles
4. Benefits over time (benefit_over_time_{benefit}.png)
5. Percent by occupation (percent_by_occupation_{benefit}.png)
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import statsmodels.api as sm
import sys
from pathlib import Path
from matplotlib.colors import to_rgba

sys.path.append(str(Path(__file__).parent.parent / "src"))

from package_files.config_utils import get_processed_dir, get_repo_root

# Benefit definitions and colors
benefits4 = [
    "EDU_ASSISTANCE",
    "PAID LEAVE",
    "HEALTH_WELLBEING",
    "PARENTAL_LEAVE",
    "CULTURE",
    "REMOTE_KW",
]

benefits_labels_map = {
    "EDU_ASSISTANCE": "Tuition Assistance",
    "PAID LEAVE": "Paid Leave",
    "HEALTH_WELLBEING": "Health and Wellbeing",
    "PARENTAL_LEAVE": "Parental Leave",
    "CULTURE": "Workplace Culture",
    "wfh_wham": "Remote Work",
    "REMOTE_KW": "Remote Work",
}
benefits4_labels = [
    "Tuition Assistance",
    "Paid Leave",
    "Health and Wellbeing",
    "Parental Leave",
    "Workplace Culture",
    "Remote Work",
]

benefit_colors = {
    "EDU_ASSISTANCE": "#41afaa",
    "PAID LEAVE": "#466eb4",
    "HEALTH_WELLBEING": "#e6a532",
    "PARENTAL_LEAVE": "#00a0e1",
    "CULTURE": "#d7642c",
    "wfh_wham": "#af4b91",
    "REMOTE_KW": "#af4b91",
}

benefit_colors_2 = {
    "EDU_ASSISTANCE": ["#41afaa", "#b3dfdd"],
    "PAID LEAVE": ["#466eb4", "#b5c5e1"],
    "HEALTH_WELLBEING": ["#e6a532", "#f5dbad"],
    "PARENTAL_LEAVE": ["#00a0e1", "#99d9f3"],
    "CULTURE": ["#d7642c", "#efc1ab"],
    "wfh_wham": ["#af4b91", "#dfb7d3"],
    "REMOTE_KW": ["#af4b91", "#dfb7d3"],
}

# Field mappings
industry = "NAICS_2022_2_NAME"
occupation = "SOC_MAJOR_GROUP"

# Set plot font size
plt.rcParams.update({"font.size": 14})

REPO_ROOT = get_repo_root()
PROCESSED_DIR = get_processed_dir()

FIG_ROOT = REPO_ROOT / "results" / "figures_2026" / "descriptive"
BENEFITS_OVER_TIME_DIR = FIG_ROOT / "benefits_over_time"
BENEFITS_BY_ROLE_DIR = FIG_ROOT / "pct_jobs_by_benefit_and_role_type"
BY_OCCUPATION_DIR = FIG_ROOT / "by_occupation"
FIRM_CATEGORY_DIR = FIG_ROOT / "firm_categories"
TABLES_DIR = REPO_ROOT / "results" / "tables_2026"

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
GENAI_CUTOFF_YEAR = 2022.875


def format_benefit_label(benefit_key, label):
    """Add benefit-source tag to labels used in figures."""
    if benefit_key.startswith("S_"):
        return f"{label} (Structured)"
    return label


def benefit_slug(benefit_key):
    """Filename-safe benefit key."""
    return benefit_key.lower().replace(" ", "_")


def add_genai_quarter_line(ax, period_index, label="Nov. 2022"):
    """Mark the public GenAI inflection point on quarter-indexed plots."""
    if GENAI_CUTOFF not in period_index:
        return None
    cutoff_x = list(period_index).index(GENAI_CUTOFF) + 2 / 3
    return ax.axvline(
        cutoff_x,
        color="black",
        linewidth=1.0,
        linestyle=":",
        alpha=0.75,
        label=label,
    )


def add_genai_year_line(ax, label="Nov. 2022"):
    """Mark the public GenAI inflection point on calendar-year plots."""
    return ax.axvline(
        GENAI_CUTOFF_YEAR,
        color="black",
        linewidth=1.0,
        linestyle=":",
        alpha=0.75,
        label=label,
    )


def set_tight_symmetric_ylim(ax, values, lower_values=None, upper_values=None, pad=0.08, floor=0.02):
    """Use a tighter symmetric y-axis around zero for coefficient plots."""
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


def load_data():
    """Load the main dataset."""
    usdf = pd.read_parquet(PROCESSED_DIR / "labeled_v2.parquet")
    print("Data loaded")
    usdf["POSTED"] = pd.to_datetime(usdf["POSTED"])
    usdf["QUARTER"] = usdf["POSTED"].dt.to_period("Q")
    usdf = add_firm_category(usdf)
    return usdf


def create_figures_directory():
    """Create directories for saving figures if they don't exist."""
    directories = [
        FIG_ROOT,
        BENEFITS_OVER_TIME_DIR,
        BENEFITS_BY_ROLE_DIR,
        BY_OCCUPATION_DIR,
        FIRM_CATEGORY_DIR,
        TABLES_DIR,
    ]
    for directory in directories:
        os.makedirs(directory, exist_ok=True)


def add_firm_category(usdf):
    """Add mutually exclusive SME, large, and S&P 500 firm categories."""
    usdf = usdf.copy()
    sp500 = usdf.get("SP500", False)
    if not isinstance(sp500, pd.Series):
        sp500 = pd.Series(False, index=usdf.index)
    sp500 = sp500.fillna(False).astype(bool)

    if "FIRM_POSTING_COUNT" in usdf.columns:
        firm_posting_count = pd.to_numeric(usdf["FIRM_POSTING_COUNT"], errors="coerce")
    else:
        firm_posting_count = pd.Series(np.nan, index=usdf.index)
    if firm_posting_count.isna().all() and "FIRM_SIZE_BUCKET" in usdf.columns:
        old_bucket = usdf["FIRM_SIZE_BUCKET"].astype("string")
        firm_posting_count = pd.Series(np.nan, index=usdf.index)
        firm_posting_count.loc[old_bucket.isin(["Small (<=2)", "Medium (3-9)", "SMEs (<10)"])] = 1
        firm_posting_count.loc[old_bucket.isin(["Large (>=10)", "Large firms (>=10)"])] = 10

    usdf[FIRM_CATEGORY] = pd.NA
    usdf.loc[firm_posting_count < 10, FIRM_CATEGORY] = "SMEs"
    usdf.loc[firm_posting_count >= 10, FIRM_CATEGORY] = "Large firms"
    usdf.loc[sp500, FIRM_CATEGORY] = "S&P 500 firms"
    usdf[FIRM_CATEGORY] = pd.Categorical(
        usdf[FIRM_CATEGORY],
        categories=FIRM_CATEGORY_ORDER,
        ordered=True,
    )
    return usdf


def estimate_quarterly_ai_wage_betas(usdf, min_ai_wage_postings=10):
    """Estimate quarterly AI log-wage coefficients with standard controls."""
    controls = [
        "MIN_EDULEVELS_NAME",
        "EXPERIENCE_BUCKET",
        "NAICS_2022_3_DIGIT",
        "STATE_NAME",
    ]
    required = ["QUARTER", "LOG_SALARY", "AI ROLE"] + controls
    wage_df = usdf.replace([np.inf, -np.inf], np.nan).dropna(subset=required).copy()

    results = []
    for quarter, group in wage_df.groupby("QUARTER", observed=True):
        ai_wage_postings = int(group["AI ROLE"].sum())
        row = {
            "QUARTER": quarter,
            "wage_postings": len(group),
            "ai_wage_postings": ai_wage_postings,
            "ai_role_beta": np.nan,
            "ai_role_se": np.nan,
            "lower_ci": np.nan,
            "upper_ci": np.nan,
            "status": "suppressed_sparse_ai_wage_postings",
        }
        if ai_wage_postings < min_ai_wage_postings:
            results.append(row)
            continue

        X = group[["AI ROLE"]].astype(float)
        for control in controls:
            dummies = pd.get_dummies(group[control], prefix=control, drop_first=True, dtype=float)
            X = pd.concat([X, dummies], axis=1)
        X = sm.add_constant(X, has_constant="add")

        try:
            model = sm.OLS(group["LOG_SALARY"].astype(float), X.astype(float)).fit(cov_type="HC1")
            beta = model.params["AI ROLE"]
            se = model.bse["AI ROLE"]
            row.update({
                "ai_role_beta": beta,
                "ai_role_se": se,
                "lower_ci": beta - 1.96 * se,
                "upper_ci": beta + 1.96 * se,
                "status": "ok",
            })
        except Exception as error:
            row["status"] = f"error: {error}"
        results.append(row)

    return pd.DataFrame(results).sort_values("QUARTER")


def estimate_period_ai_wage_betas(usdf, min_ai_wage_postings=10):
    """Estimate adjusted AI wage premiums before and after the GenAI inflection."""
    controls = [
        "MIN_EDULEVELS_NAME",
        "EXPERIENCE_BUCKET",
        "NAICS_2022_3_DIGIT",
        "STATE_NAME",
        "YEAR",
    ]
    required = ["YEAR", "LOG_SALARY", "AI ROLE"] + controls
    wage_df = usdf.replace([np.inf, -np.inf], np.nan).dropna(subset=required).copy()
    wage_df["GENAI_PERIOD"] = np.where(wage_df["YEAR"] <= 2022, "Before GenAI", "After GenAI")

    rows = []
    for period, group in wage_df.groupby("GENAI_PERIOD", sort=False):
        ai_wage_postings = int(group["AI ROLE"].sum())
        row = {
            "period": period,
            "wage_postings": len(group),
            "ai_wage_postings": ai_wage_postings,
            "ai_role_beta": np.nan,
            "ai_role_se": np.nan,
            "lower_ci": np.nan,
            "upper_ci": np.nan,
            "status": "suppressed_sparse_ai_wage_postings",
        }
        if ai_wage_postings < min_ai_wage_postings:
            rows.append(row)
            continue

        X = group[["AI ROLE"]].astype(float)
        for control in controls:
            dummies = pd.get_dummies(group[control], prefix=control, drop_first=True, dtype=float)
            X = pd.concat([X, dummies], axis=1)
        X = sm.add_constant(X, has_constant="add")

        try:
            model = sm.OLS(group["LOG_SALARY"].astype(float), X.astype(float)).fit(cov_type="HC1")
            beta = model.params["AI ROLE"]
            se = model.bse["AI ROLE"]
            row.update({
                "ai_role_beta": beta,
                "ai_role_se": se,
                "lower_ci": beta - 1.96 * se,
                "upper_ci": beta + 1.96 * se,
                "status": "ok",
            })
        except Exception as error:
            row["status"] = f"error: {error}"
        rows.append(row)

    result = pd.DataFrame(rows)
    period_order = ["Before GenAI", "After GenAI"]
    result["period"] = pd.Categorical(result["period"], period_order, ordered=True)
    return result.sort_values("period")


def add_genai_period_inset(ax, period_betas):
    """Add a compact before/after GenAI wage premium bar chart inside a wage plot."""
    ok = period_betas[period_betas["status"] == "ok"].copy()
    if ok.empty:
        return
    inset = ax.inset_axes([0.58, 0.10, 0.38, 0.34])
    x = np.arange(len(ok))
    inset.bar(
        x,
        ok["ai_role_beta"],
        yerr=[
            ok["ai_role_beta"] - ok["lower_ci"],
            ok["upper_ci"] - ok["ai_role_beta"],
        ],
        color=["#7f7f7f", "#0072B2"][: len(ok)],
        alpha=0.85,
        capsize=3,
        width=0.58,
    )
    inset.axhline(0, color="gray", linewidth=0.7, linestyle="--")
    inset.set_xticks(x)
    inset.set_xticklabels(["Before", "After"][: len(ok)], fontsize=8)
    inset.set_title("Before/After GenAI", fontsize=9)
    inset.tick_params(axis="y", labelsize=8)
    inset.grid(axis="y", alpha=0.18)


def generate_ai_roles_over_time_plot(usdf):
    """
    Generate Figure 1: quarterly AI demand and adjusted AI log-wage coefficients.
    """
    print("Generating Figure 1: AI demand and adjusted AI wage coefficients...")

    # Calculate overall percentages
    percent_data = (
        usdf.groupby("QUARTER")["AI ROLE"].mean().reset_index(name="% AI Roles")
    )
    percent_data["% AI Roles"] = percent_data["% AI Roles"] * 100

    # Calculate industry averages
    industries = usdf[industry].unique()
    ind_dfs = []
    for ind in industries:
        industry_df = usdf[usdf[industry] == ind]
        industry_pcts = (
            industry_df.groupby("QUARTER")["AI ROLE"]
            .mean()
            .reset_index(name="% AI Roles")
        )
        ind_dfs.append(industry_pcts)

    ind_df_all = pd.concat(ind_dfs, axis=0)
    avg_ind_year = ind_df_all.groupby("QUARTER")["% AI Roles"].mean().reset_index()
    avg_ind_year.rename(columns={"% AI Roles": "Industry Average"}, inplace=True)
    avg_ind_year["Industry Average"] = avg_ind_year["Industry Average"] * 100

    # Merge data
    pct_df_all = percent_data.merge(avg_ind_year, on="QUARTER")
    pct_df_all.set_index("QUARTER", inplace=True)
    # pct_df_all.drop(
    #     pct_df_all.index[-1], inplace=True
    # )  # Remove last incomplete quarter

    wage_betas = estimate_quarterly_ai_wage_betas(usdf)
    tables_dir = Path("results/tables_2026")
    tables_dir.mkdir(parents=True, exist_ok=True)
    wage_betas.assign(QUARTER=wage_betas["QUARTER"].astype(str)).to_csv(
        tables_dir / "quarterly_ai_wage_betas.csv",
        index=False,
    )
    period_betas = estimate_period_ai_wage_betas(usdf)
    period_betas.to_csv(tables_dir / "genai_period_ai_wage_betas.csv", index=False)

    # Create plot
    fig, (demand_ax, wage_ax) = plt.subplots(
        nrows=2,
        ncols=1,
        figsize=(11, 9),
        sharex=True,
    )
    quarter_positions = np.arange(len(pct_df_all.index))
    demand_ax.plot(
        quarter_positions,
        pct_df_all["% AI Roles"],
        linewidth=1.8,
        label="Overall",
    )
    demand_ax.plot(
        quarter_positions,
        pct_df_all["Industry Average"],
        linewidth=1.8,
        linestyle="--",
        label="Industry Average",
    )

    # Set industry average line to dashed with same color
    demand_ax.legend(
        title=None,
        labels=["Overall", "Industry Average"],
        loc="upper left",
        fontsize=16,
    )
    demand_ax.set_xlabel(None)
    demand_ax.set_ylabel("% AI Roles", fontsize=15)
    demand_ax.set_title("Demand for AI Skills", fontsize=15)
    add_genai_quarter_line(demand_ax, pct_df_all.index)
    demand_ax.grid(alpha=0.25)

    plotted = wage_betas[wage_betas["status"] == "ok"].copy()
    quarter_position_map = {quarter: idx for idx, quarter in enumerate(pct_df_all.index)}
    plotted["x"] = plotted["QUARTER"].map(quarter_position_map)
    post_2021 = plotted["QUARTER"] >= pd.Period("2022Q1", freq="Q")
    wage_ax.fill_between(
        plotted.loc[post_2021, "x"],
        plotted.loc[post_2021, "lower_ci"],
        plotted.loc[post_2021, "upper_ci"],
        color="#0072B2",
        alpha=0.14,
        linewidth=0,
    )
    wage_ax.plot(
        plotted["x"],
        plotted["ai_role_beta"],
        marker="o",
        color="#0072B2",
        linewidth=1.7,
        markersize=4,
    )
    wage_ax.axhline(0, color="gray", linewidth=0.8, linestyle="--")
    add_genai_quarter_line(wage_ax, pct_df_all.index)
    set_tight_symmetric_ylim(
        wage_ax,
        plotted["ai_role_beta"],
        plotted.loc[post_2021, "lower_ci"],
        plotted.loc[post_2021, "upper_ci"],
    )
    add_genai_period_inset(wage_ax, period_betas)
    wage_ax.set_ylabel("AI-Role Wage Coefficient\n(log points)", fontsize=13)
    wage_ax.set_title("Adjusted AI-Skills Wage Premium", fontsize=15)
    wage_ax.grid(alpha=0.25)

    tick_positions = list(range(0, len(pct_df_all.index), 4))
    wage_ax.set_xticks(tick_positions)
    wage_ax.set_xticklabels([str(pct_df_all.index[idx]) for idx in tick_positions], rotation=45)
    wage_ax.set_xlabel("Quarter", fontsize=13)

    plt.tight_layout()
    output_path = FIG_ROOT / "figure1_ai_demand_wage_beta.png"
    plt.savefig(output_path, bbox_inches="tight", dpi=300)
    plt.close()
    print(f"Saved: {output_path}")


def save_firm_category_descriptive_stats(usdf):
    """Save posting and AI-vacancy counts for each firm category."""
    print("Saving firm-category descriptive statistics...")
    stats = (
        usdf.dropna(subset=[FIRM_CATEGORY])
        .groupby(FIRM_CATEGORY, observed=False)
        .agg(
            postings=("AI ROLE", "size"),
            ai_vacancies=("AI ROLE", "sum"),
            firms=("COMPANY", "nunique"),
        )
        .reindex(FIRM_CATEGORY_ORDER)
        .reset_index()
    )
    stats["non_ai_vacancies"] = stats["postings"] - stats["ai_vacancies"]
    stats["ai_share"] = stats["ai_vacancies"] / stats["postings"]
    for column in ["postings", "ai_vacancies", "non_ai_vacancies", "firms"]:
        stats[column] = stats[column].fillna(0).astype(int)
    stats["ai_share"] = stats["ai_share"].fillna(0)
    # Add wage sample counts
    controls = ["MIN_EDULEVELS_NAME", "EXPERIENCE_BUCKET", "NAICS_2022_3_DIGIT", "STATE_NAME"]
    wage_required = ["LOG_SALARY", "AI ROLE"] + controls
    wage_df = usdf.replace([np.inf, -np.inf], np.nan).dropna(subset=wage_required)
    wage_stats = (
        wage_df.dropna(subset=[FIRM_CATEGORY])
        .groupby(FIRM_CATEGORY, observed=False)
        .agg(
            wage_postings=("AI ROLE", "size"),
            ai_wage_postings=("AI ROLE", "sum"),
        )
        .reindex(FIRM_CATEGORY_ORDER)
        .reset_index()
    )
    for column in ["wage_postings", "ai_wage_postings"]:
        wage_stats[column] = wage_stats[column].fillna(0).astype(int)
    stats = stats.merge(wage_stats, on=FIRM_CATEGORY)

    output = TABLES_DIR / "firm_category_descriptive_stats.csv"
    stats.to_csv(output, index=False)
    print(f"Saved: {output}")


def plot_firm_category_sample_sizes(usdf):
    """Visualize sample composition by firm category."""
    print("Generating firm-category sample size visualization...")
    stats = pd.read_csv(TABLES_DIR / "firm_category_descriptive_stats.csv")

    fig, axes = plt.subplots(1, 4, figsize=(18, 5))
    x = np.arange(len(stats))
    colors = [FIRM_CATEGORY_COLORS[c] for c in stats[FIRM_CATEGORY]]

    # Panel 1: Total postings
    bars = axes[0].bar(x, stats["postings"], color=colors, alpha=0.85)
    for bar, val in zip(bars, stats["postings"]):
        axes[0].text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                     f"{val:,}", ha="center", va="bottom", fontsize=9)
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(stats[FIRM_CATEGORY], fontsize=10)
    axes[0].set_ylabel("Count", fontsize=12)
    axes[0].set_title("Total Postings", fontsize=13)
    axes[0].grid(axis="y", alpha=0.2)

    # Panel 2: Unique firms
    bars = axes[1].bar(x, stats["firms"], color=colors, alpha=0.85)
    for bar, val in zip(bars, stats["firms"]):
        axes[1].text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                     f"{val:,}", ha="center", va="bottom", fontsize=9)
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(stats[FIRM_CATEGORY], fontsize=10)
    axes[1].set_title("Unique Firms", fontsize=13)
    axes[1].grid(axis="y", alpha=0.2)

    # Panel 3: AI share (%)
    bars = axes[2].bar(x, stats["ai_share"] * 100, color=colors, alpha=0.85)
    for bar, val in zip(bars, stats["ai_share"] * 100):
        axes[2].text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                     f"{val:.1f}%", ha="center", va="bottom", fontsize=9)
    axes[2].set_xticks(x)
    axes[2].set_xticklabels(stats[FIRM_CATEGORY], fontsize=10)
    axes[2].set_ylabel("% AI Roles", fontsize=12)
    axes[2].set_title("AI Share of Postings", fontsize=13)
    axes[2].grid(axis="y", alpha=0.2)

    # Panel 4: AI vacancies (total and wage sample)
    width = 0.35
    bars1 = axes[3].bar(x - width / 2, stats["ai_vacancies"], width,
                        label="All AI postings", color=colors, alpha=0.85)
    bars2 = axes[3].bar(x + width / 2, stats["ai_wage_postings"], width,
                        label="AI with salary", color=colors, alpha=0.45)
    for bar, val in zip(bars1, stats["ai_vacancies"]):
        axes[3].text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                     f"{val:,}", ha="center", va="bottom", fontsize=9)
    for bar, val in zip(bars2, stats["ai_wage_postings"]):
        axes[3].text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                     f"{val:,}", ha="center", va="bottom", fontsize=9)
    axes[3].set_xticks(x)
    axes[3].set_xticklabels(stats[FIRM_CATEGORY], fontsize=10)
    axes[3].set_title("AI Vacancies", fontsize=13)
    axes[3].legend(fontsize=9)
    axes[3].grid(axis="y", alpha=0.2)

    plt.suptitle("Sample Composition by Firm Category", fontsize=15, y=1.02)
    plt.tight_layout()
    output = FIRM_CATEGORY_DIR / "firm_category_sample_sizes.png"
    plt.savefig(output, bbox_inches="tight", dpi=300)
    plt.close()
    print(f"Saved: {output}")


def estimate_pooled_ai_wage_premium_by_firm_category(usdf, min_ai_wage_postings=10):
    """Estimate pooled AI wage premium and base pay level by firm category."""
    controls = [
        "MIN_EDULEVELS_NAME",
        "EXPERIENCE_BUCKET",
        "NAICS_2022_3_DIGIT",
        "STATE_NAME",
    ]
    required = ["LOG_SALARY", "AI ROLE", "YEAR", FIRM_CATEGORY] + controls
    wage_df = (
        usdf.replace([np.inf, -np.inf], np.nan)
        .dropna(subset=required)
        .copy()
    )

    rows = []
    for category in FIRM_CATEGORY_ORDER:
        subset = wage_df[wage_df[FIRM_CATEGORY] == category]
        ai_count = int(subset["AI ROLE"].sum())
        row = {
            "firm_category": category,
            "wage_postings": len(subset),
            "ai_wage_postings": ai_count,
            "mean_log_salary": subset["LOG_SALARY"].mean(),
            "mean_log_salary_ai": subset.loc[subset["AI ROLE"], "LOG_SALARY"].mean(),
            "mean_log_salary_non_ai": subset.loc[~subset["AI ROLE"], "LOG_SALARY"].mean(),
            "raw_gap": (
                subset.loc[subset["AI ROLE"], "LOG_SALARY"].mean()
                - subset.loc[~subset["AI ROLE"], "LOG_SALARY"].mean()
            ),
            "intercept": np.nan,
            "ai_role_beta": np.nan,
            "ai_role_se": np.nan,
            "lower_ci": np.nan,
            "upper_ci": np.nan,
            "r_squared": np.nan,
            "status": "suppressed",
        }
        if ai_count < min_ai_wage_postings:
            rows.append(row)
            continue

        X = subset[["AI ROLE"]].astype(float)
        for control in controls:
            dummies = pd.get_dummies(subset[control], prefix=control, drop_first=True, dtype=float)
            X = pd.concat([X, dummies], axis=1)
        year_dummies = pd.get_dummies(subset["YEAR"], prefix="YEAR", drop_first=True, dtype=float)
        X = pd.concat([X, year_dummies], axis=1)
        X = sm.add_constant(X, has_constant="add")

        try:
            model = sm.OLS(subset["LOG_SALARY"].astype(float), X.astype(float)).fit(cov_type="HC1")
            beta = model.params["AI ROLE"]
            se = model.bse["AI ROLE"]
            row.update({
                "intercept": model.params["const"],
                "ai_role_beta": beta,
                "ai_role_se": se,
                "lower_ci": beta - 1.96 * se,
                "upper_ci": beta + 1.96 * se,
                "r_squared": model.rsquared,
                "status": "ok",
            })
        except Exception as error:
            row["status"] = f"error: {error}"
        rows.append(row)

    result = pd.DataFrame(rows)
    output = TABLES_DIR / "pooled_ai_wage_premium_by_firm_category.csv"
    result.to_csv(output, index=False)
    print(f"Saved: {output}")
    return result


def plot_ai_wage_premium_by_firm_category(premium_df):
    """Bar chart comparing base pay and AI wage premium across firm categories."""
    ok = premium_df[premium_df["status"] == "ok"].copy()
    if ok.empty:
        print("No estimable firm categories for wage premium plot.")
        return

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5.5))

    # Left panel: mean log salary by role type
    x = np.arange(len(ok))
    width = 0.3
    ax1.bar(
        x - width / 2,
        ok["mean_log_salary_non_ai"],
        width,
        label="Non-AI Role",
        color="#7f7f7f",
        alpha=0.6,
    )
    ax1.bar(
        x + width / 2,
        ok["mean_log_salary_ai"],
        width,
        label="AI Role",
        color="#0072B2",
    )
    ax1.set_xticks(x)
    ax1.set_xticklabels(ok["firm_category"], fontsize=12)
    ax1.set_ylabel("Mean Log Salary", fontsize=13)
    ax1.set_title("Average Pay by Firm Category", fontsize=14)
    ax1.legend(fontsize=11)
    ax1.grid(axis="y", alpha=0.2)

    # Right panel: adjusted AI wage coefficient
    ax2.barh(
        x,
        ok["ai_role_beta"],
        xerr=[ok["ai_role_beta"] - ok["lower_ci"], ok["upper_ci"] - ok["ai_role_beta"]],
        color=[FIRM_CATEGORY_COLORS[c] for c in ok["firm_category"]],
        capsize=4,
        height=0.5,
    )
    ax2.axvline(0, color="gray", linewidth=0.8, linestyle="--")
    ax2.set_yticks(x)
    ax2.set_yticklabels(ok["firm_category"], fontsize=12)
    ax2.set_xlabel("AI-Role Wage Coefficient\n(log points, 95% CI)", fontsize=12)
    ax2.set_title("Adjusted AI Wage Premium", fontsize=14)
    ax2.grid(axis="x", alpha=0.2)

    plt.tight_layout()
    output = FIRM_CATEGORY_DIR / "ai_wage_premium_by_firm_category.png"
    plt.savefig(output, bbox_inches="tight", dpi=300)
    plt.close()
    print(f"Saved: {output}")


def _fit_one_wage_regression(subset, category, period_label, controls, min_ai_wage_postings):
    """Fit a single OLS wage regression and return a result row."""
    ai_count = int(subset["AI ROLE"].sum())
    row = {
        "firm_category": category,
        "period": period_label,
        "wage_postings": len(subset),
        "ai_wage_postings": ai_count,
        "mean_log_salary_ai": subset.loc[subset["AI ROLE"], "LOG_SALARY"].mean(),
        "mean_log_salary_non_ai": subset.loc[~subset["AI ROLE"], "LOG_SALARY"].mean(),
        "ai_role_beta": np.nan,
        "ai_role_se": np.nan,
        "lower_ci": np.nan,
        "upper_ci": np.nan,
        "status": "suppressed",
    }
    if ai_count < min_ai_wage_postings:
        return row

    X = subset[["AI ROLE"]].astype(float)
    for control in controls:
        dummies = pd.get_dummies(subset[control], prefix=control, drop_first=True, dtype=float)
        X = pd.concat([X, dummies], axis=1)
    X = sm.add_constant(X, has_constant="add")
    try:
        model = sm.OLS(subset["LOG_SALARY"].astype(float), X.astype(float)).fit(cov_type="HC1")
        beta = model.params["AI ROLE"]
        se = model.bse["AI ROLE"]
        row.update({
            "ai_role_beta": beta,
            "ai_role_se": se,
            "lower_ci": beta - 1.96 * se,
            "upper_ci": beta + 1.96 * se,
            "status": "ok",
        })
    except Exception as error:
        row["status"] = f"error: {error}"
    return row


def estimate_annual_ai_wage_premium_by_firm_category(usdf, min_ai_wage_postings=10):
    """Estimate annual AI wage premium by firm category; fall back to period groups."""
    controls = [
        "MIN_EDULEVELS_NAME",
        "EXPERIENCE_BUCKET",
        "NAICS_2022_3_DIGIT",
        "STATE_NAME",
    ]
    required = ["LOG_SALARY", "AI ROLE", "YEAR", FIRM_CATEGORY] + controls
    wage_df = (
        usdf.replace([np.inf, -np.inf], np.nan)
        .dropna(subset=required)
        .copy()
    )

    # Try annual first
    annual_rows = []
    for category in FIRM_CATEGORY_ORDER:
        cat_df = wage_df[wage_df[FIRM_CATEGORY] == category]
        for year in sorted(cat_df["YEAR"].unique()):
            year_df = cat_df[cat_df["YEAR"] == year]
            annual_rows.append(
                _fit_one_wage_regression(year_df, category, str(int(year)), controls, min_ai_wage_postings)
            )
    annual_df = pd.DataFrame(annual_rows)

    # Check if annual works or if we need period groups
    ok_counts = annual_df[annual_df["status"] == "ok"].groupby("firm_category").size()
    needs_grouping = any(ok_counts.get(c, 0) < 3 for c in FIRM_CATEGORY_ORDER)

    if needs_grouping:
        print("  Annual too sparse for some categories — adding period-group estimates...")
        period_map = {}
        for year in wage_df["YEAR"].unique():
            if year <= 2022:
                period_map[year] = "2018–2022"
            else:
                period_map[year] = "2023–2025"
        wage_df["PERIOD_GROUP"] = wage_df["YEAR"].map(period_map)
        group_rows = []
        for category in FIRM_CATEGORY_ORDER:
            cat_df = wage_df[wage_df[FIRM_CATEGORY] == category]
            for period, period_df in cat_df.groupby("PERIOD_GROUP"):
                group_rows.append(
                    _fit_one_wage_regression(period_df, category, period, controls, min_ai_wage_postings)
                )
        group_df = pd.DataFrame(group_rows)
        combined = pd.concat([annual_df, group_df], ignore_index=True)
    else:
        combined = annual_df

    output = TABLES_DIR / "annual_ai_wage_premium_by_firm_category.csv"
    combined.to_csv(output, index=False)
    print(f"Saved: {output}")
    return combined


def plot_annual_ai_wage_premium_by_firm_category(annual_df):
    """Plot AI wage premium over time by firm category (separate figures)."""
    ok = annual_df[annual_df["status"] == "ok"].copy()
    if ok.empty:
        print("No estimable results for annual wage premium plot.")
        return

    # Separate annual vs period-group rows
    annual_rows = ok[ok["period"].str.match(r"^\d{4}$")]
    group_rows = ok[~ok["period"].str.match(r"^\d{4}$")]

    if not annual_rows.empty:
        annual_rows = annual_rows.copy()
        annual_rows["year"] = annual_rows["period"].astype(int)
        fig, ax = plt.subplots(figsize=(10, 6))
        for category in FIRM_CATEGORY_ORDER:
            line = annual_rows[annual_rows["firm_category"] == category].sort_values("year")
            if line.empty:
                continue
            color = FIRM_CATEGORY_COLORS[category]
            ax.fill_between(
                line["year"],
                line["lower_ci"],
                line["upper_ci"],
                alpha=0.12,
                color=color,
            )
            ax.plot(
                line["year"],
                line["ai_role_beta"],
                marker=FIRM_CATEGORY_MARKERS[category],
                markersize=5,
                label=category,
                color=color,
                linewidth=1.8,
            )
        ax.axhline(0, color="gray", linewidth=0.8, linestyle="--")
        add_genai_year_line(ax)
        ax.set_xlabel("Year", fontsize=12)
        ax.set_ylabel("AI-Role Wage Coefficient\n(log points, 95% CI)", fontsize=12)
        ax.set_title("Annual AI Wage Premium by Firm Category", fontsize=14)
        ax.legend(fontsize=10)
        ax.grid(alpha=0.2)
        plt.tight_layout()
        output = FIRM_CATEGORY_DIR / "ai_wage_premium_by_firm_category_annual.png"
        plt.savefig(output, bbox_inches="tight", dpi=300)
        plt.close()
        print(f"Saved: {output}")

    if not group_rows.empty:
        period_order = sorted(group_rows["period"].unique())
        x = np.arange(len(period_order))
        n_cats = len(FIRM_CATEGORY_ORDER)
        width = 0.25
        fig, ax = plt.subplots(figsize=(10, 6))
        for i, category in enumerate(FIRM_CATEGORY_ORDER):
            cat_data = group_rows[group_rows["firm_category"] == category].set_index("period").reindex(period_order)
            vals = cat_data["ai_role_beta"].values
            errs = [
                cat_data["ai_role_beta"].values - cat_data["lower_ci"].values,
                cat_data["upper_ci"].values - cat_data["ai_role_beta"].values,
            ]
            ax.bar(
                x + (i - (n_cats - 1) / 2) * width,
                vals,
                width,
                yerr=errs,
                label=category,
                color=FIRM_CATEGORY_COLORS[category],
                capsize=3,
                alpha=0.85,
            )
        ax.axhline(0, color="gray", linewidth=0.8, linestyle="--")
        ax.set_xticks(x)
        ax.set_xticklabels(period_order, fontsize=12)
        ax.set_ylabel("AI-Role Wage Coefficient\n(log points, 95% CI)", fontsize=12)
        ax.set_title("AI Wage Premium: Pre vs Post GenAI", fontsize=14)
        ax.legend(fontsize=10)
        ax.grid(axis="y", alpha=0.2)
        plt.tight_layout()
        output = FIRM_CATEGORY_DIR / "ai_wage_premium_by_firm_category_pre_post_genai.png"
        plt.savefig(output, bbox_inches="tight", dpi=300)
        plt.close()
        print(f"Saved: {output}")


def estimate_quarterly_ai_wage_betas_by_firm_category(usdf, min_ai_wage_postings=10):
    """Estimate quarterly AI log-wage coefficients separately by firm category."""
    rows = []
    for category in FIRM_CATEGORY_ORDER:
        subset = usdf[usdf[FIRM_CATEGORY] == category].copy()
        estimates = estimate_quarterly_ai_wage_betas(subset, min_ai_wage_postings)
        estimates[FIRM_CATEGORY] = category
        rows.append(estimates)
    return pd.concat(rows, ignore_index=True)


def generate_ai_roles_over_time_by_firm_category_plot(usdf):
    """Replicate Figure 1 separately for SME, large, and S&P 500 firms."""
    print("Generating firm-category AI demand and wage premium plot...")
    category_df = usdf.dropna(subset=[FIRM_CATEGORY]).copy()
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

    wage_betas = estimate_quarterly_ai_wage_betas_by_firm_category(category_df)
    wage_betas.assign(QUARTER=wage_betas["QUARTER"].astype(str)).to_csv(
        TABLES_DIR / "quarterly_ai_wage_betas_by_firm_category.csv",
        index=False,
    )

    fig, (demand_ax, wage_ax) = plt.subplots(
        nrows=2,
        ncols=1,
        figsize=(11, 9),
        sharex=True,
    )
    quarter_positions = np.arange(len(demand_wide.index))
    quarter_position_map = {quarter: idx for idx, quarter in enumerate(demand_wide.index)}

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
    wage_ax.set_xticklabels([str(demand_wide.index[idx]) for idx in tick_positions], rotation=45)
    wage_ax.set_xlabel("Quarter", fontsize=13)

    plt.tight_layout()
    output_path = FIRM_CATEGORY_DIR / "figure1_ai_demand_wage_beta_by_firm_category.png"
    plt.savefig(output_path, bbox_inches="tight", dpi=300)
    plt.close()
    print(f"Saved: {output_path}")


def count_benefits(data, benefit, period):
    """Helper function to count benefits by period and AI role."""
    grouped_data = (
        data.groupby([period, "AI ROLE", benefit]).size().reset_index(name="count")
    )

    # Calculate total jobs per period for each AI role category
    total_monthly_jobs = (
        grouped_data.groupby([period, "AI ROLE"])["count"]
        .sum()
        .reset_index(name="total_count")
    )

    # Filter for benefit = 1
    benefit_jobs = grouped_data[grouped_data[benefit] == 1]
    benefit_monthly_jobs = (
        benefit_jobs.groupby([period, "AI ROLE"])["count"]
        .sum()
        .reset_index(name="benefit_count")
    )

    # Merge and calculate percentage
    merged_data = pd.merge(
        benefit_monthly_jobs, total_monthly_jobs, on=[period, "AI ROLE"]
    )
    merged_data["percentage"] = (
        merged_data["benefit_count"] / merged_data["total_count"]
    ) * 100

    return merged_data


def generate_benefit_differences_plot(usdf):
    """
    Generate Figure 2: Difference in percent jobs with benefit, AI-non-AI
    """
    print("Generating Figure 2: Difference in percent jobs with benefit, AI-non-AI...")

    all_quarters = usdf["QUARTER"].unique()
    pct_diff_df = pd.DataFrame(all_quarters).set_index(0)

    for benefit in benefits4:
        merged_data = count_benefits(usdf, benefit, "QUARTER")
        pivot_data = merged_data.pivot(
            index="QUARTER", columns="AI ROLE", values="percentage"
        ).fillna(0)
        pivot_data.columns = ["Other", "AI Role"]
        pivot_data["pct_diff"] = pivot_data["AI Role"] - pivot_data["Other"]
        pivot_data = pivot_data.reindex(pct_diff_df.index).fillna(0)
        pct_diff_df[benefit] = pivot_data["pct_diff"]

    pct_diff_df.sort_index(inplace=True)

    # Convert index to PeriodIndex for chronological ordering
    # pct_diff_df.index = pd.PeriodIndex(pct_diff_df.index, freq="Q")

    # Create plot
    fig, ax = plt.subplots(figsize=(10, 6))
    quarter_positions = np.arange(len(pct_diff_df.index))
    for benefit in benefits4:
        ax.plot(
            quarter_positions,
            pct_diff_df[benefit],
            label=format_benefit_label(benefit, benefits_labels_map[benefit]),
            color=benefit_colors[benefit],
            linewidth=1.6,
        )

    # Set x-ticks to show quarters at regular intervals
    # ax.set_xticks(range(0, len(pct_diff_df), 4))
    # ax.set_xticklabels(pct_diff_df.index[::4].strftime("%Y-Q%q"), rotation=45)

    ax.set_xlabel(None)
    ax.set_ylabel("Difference in Percent Jobs with Benefit, AI - Non-AI")
    add_genai_quarter_line(ax, pct_diff_df.index)
    tick_positions = list(range(0, len(pct_diff_df.index), 4))
    ax.set_xticks(tick_positions)
    ax.set_xticklabels([str(pct_diff_df.index[idx]) for idx in tick_positions], rotation=45)
    ax.legend()

    plt.tight_layout()
    plt.savefig(BENEFITS_OVER_TIME_DIR / "benefit_diffs_time_keyword.png")
    # plt.show()    


def generate_benefits_by_role_plot(usdf):
    """
    Generate Figure 3: Percent of jobs with each of 6 benefits for AI and non-AI roles
    """
    print("Generating Figure 3: Percent of jobs with each benefit by role type...")

    # Calculate percentages for each benefit by AI ROLE
    percentages = usdf.groupby("AI ROLE")[benefits4].mean() * 100
    # Put True row before False
    percentages = percentages.reindex([True, False])

    x = np.arange(len(benefits4))
    width = 0.25

    fig, ax = plt.subplots(layout="constrained")
    for i, (role, row) in enumerate(percentages.iterrows()):
        color = [benefit_colors[benefit] for benefit in benefits4]
        alpha = 0.6 if role == False else 1.0
        ax.bar(
            x + i * width,
            row[benefits4],
            width,
            label=f"AI Role: {role}",
            color=color,
            alpha=alpha,
        )

    ax.set_ylabel("Percent of Jobs")
    ax.set_xticks(
        x + width,
        [format_benefit_label(benefit, benefits_labels_map[benefit]) for benefit in benefits4],
        rotation=45,
        ha="right",
    )
    ax.set_ylim(0, 40)

    # Custom legend
    from matplotlib.patches import Patch

    legend_elements = [
        Patch(facecolor="gray", alpha=1.0, label="AI Role"),
        Patch(facecolor="gray", alpha=0.5, label="Non-AI Role"),
    ]
    ax.legend(handles=legend_elements, title=None)

    plt.savefig(BENEFITS_BY_ROLE_DIR / "figure2a_benefits_ai_role_keyword.png", bbox_inches="tight")
    # plt.show()


def plot_benefits_over_time(merged_data, benefit, period, colors_dict=None, labels_dict=None):
    """Helper function to plot benefits over time."""
    if colors_dict is None:
        colors_dict = benefit_colors_2
    if labels_dict is None:
        labels_dict = benefits_labels_map

    pivot_data = merged_data.pivot(
        index=period, columns="AI ROLE", values="percentage"
    ).fillna(0)
    pivot_data.columns = ["No", "Yes"]
    pivot_data = pivot_data[
        ["Yes"] + [col for col in pivot_data.columns if col != "Yes"]
    ]

    fig, ax = plt.subplots(figsize=(10, 6))
    quarter_positions = np.arange(len(pivot_data.index))
    for column, color in zip(pivot_data.columns, colors_dict[benefit]):
        ax.plot(
            quarter_positions,
            pivot_data[column],
            label=column,
            color=color,
            linewidth=1.7,
        )
    ax.set_xlabel(None)
    ax.set_ylabel("Percent Jobs with Benefit")
    add_genai_quarter_line(ax, pivot_data.index)
    tick_positions = list(range(0, len(pivot_data.index), 4))
    ax.set_xticks(tick_positions)
    ax.set_xticklabels([str(pivot_data.index[idx]) for idx in tick_positions], rotation=45)
    ax.legend(title="AI Role", labels=["Yes", "No"])
    ax.set_title(format_benefit_label(benefit, labels_dict[benefit]))
    ax.set_ylim(0, 50)
    plt.savefig(BENEFITS_OVER_TIME_DIR / f"benefit_over_time_{benefit_slug(benefit)}.png")
    plt.close()


def generate_benefits_over_time_plots(usdf):
    """
    Generate Figure 4: Benefits over time (benefit_over_time_{benefit}.png)
    """
    print("Generating Figure 4: Benefits over time plots...")

    for benefit in benefits4:
        print(f"  Generating {benefit} over time plot...")
        merged_data = count_benefits(usdf, benefit, "QUARTER")
        plot_benefits_over_time(merged_data, benefit, "QUARTER")


def get_benefit_by_occupation(usdf_select, benefit):
    """Helper function to get benefit percentages by occupation."""
    occ_ai_benefit_group = (
        usdf_select.groupby([occupation, "AI ROLE", benefit])
        .size()
        .reset_index(name="benefit_count")
    )
    occ_ai_group = (
        usdf_select.groupby([occupation, "AI ROLE"]).size().reset_index(name="count")
    )
    occ_ai_benefit_group = occ_ai_benefit_group.merge(
        occ_ai_group, on=[occupation, "AI ROLE"], how="left"
    )
    occ_ai_benefit_group[f"Percent with {benefit}"] = (
        occ_ai_benefit_group["benefit_count"] / occ_ai_benefit_group["count"]
    )
    occ_ai_benefit_percent = occ_ai_benefit_group[
        occ_ai_benefit_group[benefit] == 1.0
    ].copy()

    occ_ai_benefit_percent.rename(columns={occupation: "Occupation"}, inplace=True)
    occ_ai_benefit_percent["AI ROLE"] = occ_ai_benefit_percent["AI ROLE"].map(
        {True: "Yes", False: "No"}
    )
    occ_ai_benefit_percent["Occupation"] = occ_ai_benefit_percent[
        "Occupation"
    ].str.replace(" Occupations", "")

    return occ_ai_benefit_percent


def plot_benefit_occ_percents(df, benefit):
    """Plot benefit percentages by occupation."""
    sorted_data = df[df["AI ROLE"] == "No"]
    sorted_data = sorted_data.sort_values(
        by=["AI ROLE", f"Percent with {benefit}"], ascending=[False, True]
    )
    df = df.sort_values(
        by=["AI ROLE", f"Percent with {benefit}"], ascending=[False, True]
    )

    palette = benefit_colors_2[benefit]

    fig, ax = plt.subplots(figsize=(12, 6))
    sns.barplot(
        x=f"Percent with {benefit}",
        y="Occupation",
        hue="AI ROLE",
        data=df,
        ax=ax,
        palette=palette,
        dodge=True,
        order=sorted_data["Occupation"],
    )

    ax.set_xlabel(
        f"Percent Jobs with {format_benefit_label(benefit, benefits_labels_map[benefit])}",
        fontsize=14,
    )
    ax.set_ylabel("Occupation", fontsize=14)
    ax.set_title(None)

    plt.xticks(rotation=45)
    plt.legend(title="AI ROLE", bbox_to_anchor=(1.02, 1), loc="upper left")
    plt.subplots_adjust(bottom=0.2, left=0.1, right=0.9, top=0.9)
    plt.tight_layout()
    plt.savefig(BY_OCCUPATION_DIR / f"percent_by_occupation_{benefit_slug(benefit)}.png")
    # plt.show()


def generate_benefits_by_occupation_plots(usdf):
    """
    Generate Figure 5: Percent by occupation (percent_by_occupation_{benefit}.png)
    """
    print("Generating Figure 5: Benefits by occupation plots...")

    # Select major occupations
    occupations_select = [
        "Architecture and Engineering Occupations",
        "Arts, Design, Entertainment, Sports, and Media Occupations",
        "Business and Financial Operations Occupations",
        "Community and Social Service Occupations",
        "Computer and Mathematical Occupations",
        "Educational Instruction and Library Occupations",
        "Healthcare Practitioners and Technical Occupations",
        "Legal Occupations",
        "Life, Physical, and Social Science Occupations",
        "Management Occupations",
        "Office and Administrative Support Occupations",
        "Personal Care and Service Occupations",
        "Production Occupations",
        "Sales and Related Occupations",
        "Transportation and Material Moving Occupations",
    ]

    usdf_select = usdf[usdf[occupation].isin(occupations_select)]

    benefit_percent_dfs = []
    for benefit in benefits4:
        print(f"  Generating {benefit} by occupation plot...")
        occ_ai_benefit_percent = get_benefit_by_occupation(usdf_select, benefit)
        benefit_percent_dfs.append(occ_ai_benefit_percent)
        plot_benefit_occ_percents(occ_ai_benefit_percent, benefit)


# --- Combined figures: all benefits (keyword + structured) ---

# Import full benefit set from definitions
import sys
sys.path.append(str(Path(__file__).parent.parent / "src"))
from package_files.benefits_defns import (
    benefits6,
    benefits6_labels,
    benefit_colors as all_colors,
    benefit_colors_2 as all_colors_2,
)


def generate_combined_benefits_by_role_plot(usdf):
    """Percent of jobs with each benefit for AI and non-AI roles — all benefits."""
    print("Generating combined benefits by role plot...")

    percentages = usdf.groupby("AI ROLE")[benefits6].mean() * 100
    percentages = percentages.reindex([True, False])

    x = np.arange(len(benefits6))
    width = 0.3

    fig, ax = plt.subplots(figsize=(14, 6), layout="constrained")
    for i, (role, row) in enumerate(percentages.iterrows()):
        color = [all_colors[b] for b in benefits6]
        alpha = 0.5 if role == False else 1.0
        ax.bar(
            x + i * width,
            row[benefits6],
            width,
            color=color,
            alpha=alpha,
        )

    ax.set_ylabel("Percent of Jobs", fontsize=14)
    ax.set_xticks(x + width / 2)
    ax.set_xticklabels(
        [format_benefit_label(benefit, label) for benefit, label in zip(benefits6, benefits6_labels)],
        rotation=45,
        ha="right",
        fontsize=11,
    )
    ax.set_ylim(0, 40)

    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor="gray", alpha=1.0, label="AI Role"),
        Patch(facecolor="gray", alpha=0.5, label="Non-AI Role"),
    ]
    ax.legend(handles=legend_elements, title=None, fontsize=12)

    plt.savefig(BENEFITS_BY_ROLE_DIR / "benefits_all_ai_role.png", bbox_inches="tight", dpi=150)
    plt.close()
    print("Saved: benefits_all_ai_role.png")


def generate_benefits_by_role_by_firm_category_plot(usdf):
    """Replicate keyword benefit prevalence by AI-role status within each firm category."""
    print("Generating benefits by role and firm category plot...")
    category_df = usdf.dropna(subset=[FIRM_CATEGORY]).copy()
    role_labels = {True: "AI Role", False: "Non-AI Role"}

    fig, axes = plt.subplots(
        nrows=1,
        ncols=len(FIRM_CATEGORY_ORDER),
        figsize=(18, 6),
        sharey=True,
        layout="constrained",
    )
    x = np.arange(len(benefits4))
    width = 0.34

    for ax, category in zip(axes, FIRM_CATEGORY_ORDER):
        subset = category_df[category_df[FIRM_CATEGORY] == category]
        percentages = subset.groupby("AI ROLE")[benefits4].mean().mul(100).reindex([True, False])
        for i, role in enumerate([True, False]):
            if role not in percentages.index:
                continue
            ax.bar(
                x + (i - 0.5) * width,
                percentages.loc[role, benefits4],
                width,
                color=[benefit_colors[benefit] for benefit in benefits4],
                alpha=1.0 if role else 0.45,
                label=role_labels[role],
            )
        ax.set_title(category, fontsize=13)
        ax.set_xticks(x)
        ax.set_xticklabels(
            [
                format_benefit_label(benefit, label)
                for benefit, label in zip(benefits4, benefits4_labels)
            ],
            rotation=45,
            ha="right",
            fontsize=9,
        )
        ax.set_ylim(0, 45)
        ax.grid(axis="y", alpha=0.2)

    axes[0].set_ylabel("Percent of Jobs", fontsize=12)

    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor="gray", alpha=1.0, label="AI Role"),
        Patch(facecolor="gray", alpha=0.45, label="Non-AI Role"),
    ]
    axes[-1].legend(handles=legend_elements, title=None, fontsize=10, loc="upper right")

    output = FIRM_CATEGORY_DIR / "benefits_ai_role_keyword_by_firm_category.png"
    plt.savefig(output, bbox_inches="tight", dpi=300)
    plt.close()
    print(f"Saved: {output}")


def generate_benefit_gap_by_firm_category_plot(usdf):
    """Bar chart of AI minus non-AI perk prevalence, grouped by firm category."""
    print("Generating benefit gap by firm category plot...")
    category_df = usdf.dropna(subset=[FIRM_CATEGORY]).copy()

    gaps = []
    for category in FIRM_CATEGORY_ORDER:
        subset = category_df[category_df[FIRM_CATEGORY] == category]
        pcts = subset.groupby("AI ROLE")[benefits4].mean().mul(100)
        if True in pcts.index and False in pcts.index:
            diff = pcts.loc[True] - pcts.loc[False]
        else:
            diff = pd.Series(0.0, index=benefits4)
        diff.name = category
        gaps.append(diff)
    gap_df = pd.DataFrame(gaps)  # rows = categories, cols = benefits

    x = np.arange(len(benefits4))
    n_cats = len(FIRM_CATEGORY_ORDER)
    width = 0.25

    fig, ax = plt.subplots(figsize=(12, 6))
    for i, category in enumerate(FIRM_CATEGORY_ORDER):
        ax.bar(
            x + (i - (n_cats - 1) / 2) * width,
            gap_df.loc[category],
            width,
            label=category,
            color=FIRM_CATEGORY_COLORS[category],
            alpha=0.85,
        )

    ax.axhline(0, color="gray", linewidth=0.8, linestyle="--")
    ax.set_ylabel("Difference in % Jobs with Benefit\n(AI Role − Non-AI Role)", fontsize=13)
    ax.set_title("AI Perk Premium by Firm Category", fontsize=15)
    ax.set_xticks(x)
    ax.set_xticklabels(
        [format_benefit_label(b, benefits_labels_map[b]) for b in benefits4],
        rotation=30,
        ha="right",
        fontsize=12,
    )
    ax.legend(title=None, fontsize=11)
    ax.grid(axis="y", alpha=0.2)

    plt.tight_layout()
    output = FIRM_CATEGORY_DIR / "benefit_gap_by_firm_category.png"
    plt.savefig(output, bbox_inches="tight", dpi=300)
    plt.close()
    print(f"Saved: {output}")


def generate_combined_benefit_differences_plot(usdf):
    """Difference in percent jobs with benefit (AI − non-AI) over time — all benefits."""
    print("Generating combined benefit differences over time plot...")

    all_quarters = usdf["QUARTER"].unique()
    pct_diff_df = pd.DataFrame(all_quarters).set_index(0)

    for benefit in benefits6:
        merged_data = count_benefits(usdf, benefit, "QUARTER")
        pivot_data = merged_data.pivot(
            index="QUARTER", columns="AI ROLE", values="percentage"
        ).fillna(0)
        pivot_data.columns = ["Other", "AI Role"]
        pivot_data["pct_diff"] = pivot_data["AI Role"] - pivot_data["Other"]
        pivot_data = pivot_data.reindex(pct_diff_df.index).fillna(0)
        pct_diff_df[benefit] = pivot_data["pct_diff"]

    pct_diff_df.sort_index(inplace=True)

    fig, ax = plt.subplots(figsize=(12, 7))
    quarter_positions = np.arange(len(pct_diff_df.index))
    for benefit, label in zip(benefits6, benefits6_labels):
        ax.plot(
            quarter_positions,
            pct_diff_df[benefit],
            label=format_benefit_label(benefit, label),
            color=all_colors[benefit],
            linewidth=1.5,
        )

    ax.set_xlabel(None)
    ax.set_ylabel("Difference in Percent Jobs with Benefit, AI - Non-AI", fontsize=12)
    add_genai_quarter_line(ax, pct_diff_df.index)
    tick_positions = list(range(0, len(pct_diff_df.index), 4))
    ax.set_xticks(tick_positions)
    ax.set_xticklabels([str(pct_diff_df.index[idx]) for idx in tick_positions], rotation=45)
    ax.legend(fontsize=9, loc="upper left")

    plt.tight_layout()
    plt.savefig(BENEFITS_OVER_TIME_DIR / "benefit_diffs_time_all.png", bbox_inches="tight", dpi=150)
    plt.close()
    print("Saved: benefit_diffs_time_all.png")


def generate_combined_benefits_over_time_plots(usdf):
    """Benefits over time — individual plots for structured benefits."""
    print("Generating over-time plots for structured benefits...")

    from package_files.benefits_defns import benefits_labels_map as all_labels_map
    struct_benefits = ["S_FLEX_WORK", "S_PROF_DEV", "S_HEALTH_WELLNESS", "S_REMOTE"]
    for benefit in struct_benefits:
        print(f"  Generating {benefit} over time plot...")
        merged_data = count_benefits(usdf, benefit, "QUARTER")
        plot_benefits_over_time(merged_data, benefit, "QUARTER",
                                colors_dict=all_colors_2, labels_dict=all_labels_map)


def main(include_structured=False):
    """Main function to generate all figures."""
    print("Loading data...")
    usdf = load_data()

    print("Creating figure directories...")
    create_figures_directory()

    print("Generating key figures...")

    # Figure 1: % AI roles over time
    generate_ai_roles_over_time_plot(usdf)
    save_firm_category_descriptive_stats(usdf)
    plot_firm_category_sample_sizes(usdf)
    premium_df = estimate_pooled_ai_wage_premium_by_firm_category(usdf)
    plot_ai_wage_premium_by_firm_category(premium_df)
    annual_df = estimate_annual_ai_wage_premium_by_firm_category(usdf)
    plot_annual_ai_wage_premium_by_firm_category(annual_df)
    generate_ai_roles_over_time_by_firm_category_plot(usdf)

    # Figure 2: Difference in percent jobs with benefit
    generate_benefit_differences_plot(usdf)

    # Figure 3: Percent of jobs with benefits by role type
    generate_benefits_by_role_plot(usdf)

    # Figure 4: Benefits over time
    generate_benefits_over_time_plots(usdf)

    # Figure 5: Benefits by occupation
    generate_benefits_by_occupation_plots(usdf)

    generate_benefits_by_role_by_firm_category_plot(usdf)
    generate_benefit_gap_by_firm_category_plot(usdf)

    if include_structured:
        print("\nGenerating combined figures (keyword + structured benefits)...")
        generate_combined_benefits_by_role_plot(usdf)
        generate_combined_benefit_differences_plot(usdf)
        generate_combined_benefits_over_time_plots(usdf)

    print("All figures generated successfully!")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Descriptive analysis figures")
    parser.add_argument(
        "--include-structured",
        action="store_true",
        default=False,
        help="Also generate combined figures with structured-field benefits",
    )
    args = parser.parse_args()
    main(include_structured=args.include_structured)
