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


def format_benefit_label(benefit_key, label):
    """Add benefit-source tag to labels used in figures."""
    if benefit_key.startswith("S_"):
        return f"{label} (Structured)"
    return label


def benefit_slug(benefit_key):
    """Filename-safe benefit key."""
    return benefit_key.lower().replace(" ", "_")


def load_data():
    """Load the main dataset."""
    usdf = pd.read_parquet(PROCESSED_DIR / "labeled_v2.parquet")
    print("Data loaded")
    usdf["POSTED"] = pd.to_datetime(usdf["POSTED"])
    usdf["QUARTER"] = usdf["POSTED"].dt.to_period("Q")
    return usdf


def create_figures_directory():
    """Create directories for saving figures if they don't exist."""
    directories = [FIG_ROOT, BENEFITS_OVER_TIME_DIR, BENEFITS_BY_ROLE_DIR, BY_OCCUPATION_DIR]
    for directory in directories:
        os.makedirs(directory, exist_ok=True)


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
    demand_ax.grid(alpha=0.25)

    plotted = wage_betas[wage_betas["status"] == "ok"].copy()
    quarter_position_map = {quarter: idx for idx, quarter in enumerate(pct_df_all.index)}
    plotted["x"] = plotted["QUARTER"].map(quarter_position_map)
    wage_ax.errorbar(
        plotted["x"],
        plotted["ai_role_beta"],
        yerr=[
            plotted["ai_role_beta"] - plotted["lower_ci"],
            plotted["upper_ci"] - plotted["ai_role_beta"],
        ],
        fmt="o-",
        color="#0072B2",
        capsize=3,
        linewidth=1.5,
        markersize=4,
    )
    wage_ax.axhline(0, color="gray", linewidth=0.8, linestyle="--")
    wage_ax.set_ylabel("AI-Role Wage Coefficient\n(log points, 95% CI)", fontsize=13)
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
    pct_diff_df.plot(
        ax=ax,
        color=list(benefit_colors.values())[:4]
        + [benefit_colors["CULTURE"], benefit_colors["REMOTE_KW"]],
    )

    # Set x-ticks to show quarters at regular intervals
    # ax.set_xticks(range(0, len(pct_diff_df), 4))
    # ax.set_xticklabels(pct_diff_df.index[::4].strftime("%Y-Q%q"), rotation=45)

    ax.set_xlabel(None)
    ax.set_ylabel("Difference in Percent Jobs with Benefit, AI - Non-AI")
    ax.legend(
        [format_benefit_label(benefit, benefits_labels_map[benefit]) for benefit in benefits4]
    )

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

    plt.savefig(BENEFITS_BY_ROLE_DIR / "benefits_ai_role_keyword.png", bbox_inches="tight")
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
    pivot_data.plot(ax=ax, color=colors_dict[benefit])
    ax.set_xlabel(None)
    ax.set_ylabel("Percent Jobs with Benefit")
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
    color_list = [all_colors[b] for b in benefits6]
    pct_diff_df.plot(ax=ax, color=color_list)

    ax.set_xlabel(None)
    ax.set_ylabel("Difference in Percent Jobs with Benefit, AI - Non-AI", fontsize=12)
    ax.legend(
        [format_benefit_label(benefit, label) for benefit, label in zip(benefits6, benefits6_labels)],
        fontsize=9,
        loc="upper left",
    )

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

    # Figure 2: Difference in percent jobs with benefit
    generate_benefit_differences_plot(usdf)

    # Figure 3: Percent of jobs with benefits by role type
    generate_benefits_by_role_plot(usdf)

    # Figure 4: Benefits over time
    generate_benefits_over_time_plots(usdf)

    # Figure 5: Benefits by occupation
    generate_benefits_by_occupation_plots(usdf)

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
