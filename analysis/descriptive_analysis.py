#!/usr/bin/env python3
"""
This script generates 5 visualizations from the descriptive statistics analysis:
1. % AI roles over time plot (overall and industry average)
2. Difference in percent jobs with benefit, AI-non-AI
3. Percent of jobs with each of 6 benefits for ai and non-ai roles
4. Benefits over time (over_time_color_{benefit}.png)
5. Percent by occupation (percent_by_occupation_colors_2024_{benefit}.png)
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from pathlib import Path
from matplotlib.colors import to_rgba

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


def load_data():
    """Load the main dataset."""
    _base = Path(__file__).parent.parent / "data" / "processed"
    usdf = pd.read_parquet(_base / 'labeled_v1.parquet')
    print("Data loaded")
    usdf["POSTED"] = pd.to_datetime(usdf["POSTED"])
    usdf["QUARTER"] = usdf["POSTED"].dt.to_period("Q")
    usdf = usdf[usdf["QUARTER"] != "2024Q3"]  # Remove incomplete quarter
    return usdf


def create_figures_directory():
    """Create directories for saving figures if they don't exist."""
    directories = [
        "results/figures_2026",
        "results/figures_2026/benefits_over_time",
        "results/figures_2026/pct_jobs_by_benefit_and_role_type",
    ]
    for directory in directories:
        os.makedirs(directory, exist_ok=True)


def generate_ai_roles_over_time_plot(usdf):
    """
    Generate Figure 1: % AI roles over time plot (overall and industry average)
    """
    print("Generating Figure 1: % AI roles over time plot...")

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

    # Create plot
    fig, ax = plt.subplots(figsize=(10, 6))
    pct_df_all.plot(ax=ax, marker=None, fontsize=16)

    # Set industry average line to dashed with same color
    ai_roles_line = ax.lines[pct_df_all.columns.get_loc("% AI Roles")]
    industry_avg_line = ax.lines[pct_df_all.columns.get_loc("Industry Average")]
    industry_avg_line.set_linestyle("--")
    industry_avg_line.set_color(ai_roles_line.get_color())

    # Formatting
    ax.legend(
        title=None,
        labels=["Overall", "Industry Average"],
        loc="upper left",
        fontsize=16,
    )
    plt.xlabel(None)
    plt.ylabel("% AI Roles", fontsize=18)
    plt.savefig(
        "results/figures_2026/pct_ai_roles_overall_industry.png", bbox_inches="tight"
    )
    # plt.show()


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
    ax.legend(benefits4_labels)

    plt.tight_layout()
    plt.savefig("results/figures_2026/benefits_over_time/benefit_diffs_time.png")
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
    ax.set_xticks(x + width, benefits4_labels, rotation=45, ha="right")
    ax.set_ylim(0, 40)

    # Custom legend
    from matplotlib.patches import Patch

    legend_elements = [
        Patch(facecolor="gray", alpha=1.0, label="AI Role"),
        Patch(facecolor="gray", alpha=0.5, label="Non-AI Role"),
    ]
    ax.legend(handles=legend_elements, title=None)

    plt.savefig(
        "results/figures_2026/pct_jobs_by_benefit_and_role_type/benefits_ai_role_colors_10m_2024.png",
        bbox_inches="tight",
    )
    # plt.show()


def plot_benefits_over_time(merged_data, benefit, period):
    """Helper function to plot benefits over time."""
    pivot_data = merged_data.pivot(
        index=period, columns="AI ROLE", values="percentage"
    ).fillna(0)
    pivot_data.columns = ["No", "Yes"]
    pivot_data = pivot_data[
        ["Yes"] + [col for col in pivot_data.columns if col != "Yes"]
    ]

    fig, ax = plt.subplots(figsize=(10, 6))
    pivot_data.plot(ax=ax, color=benefit_colors_2[benefit])
    ax.set_xlabel(None)
    ax.set_ylabel("Percent Jobs with Benefit")
    ax.legend(title="AI Role", labels=["Yes", "No"])
    ax.set_title(benefits_labels_map[benefit])
    ax.set_ylim(0, 50)
    plt.savefig(f"results/figures_2026/benefits_over_time/over_time_color_{benefit}.png")
    # plt.show()


def generate_benefits_over_time_plots(usdf):
    """
    Generate Figure 4: Benefits over time (over_time_color_{benefit}.png)
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

    ax.set_xlabel(f"Percent Jobs with {benefits_labels_map[benefit]}", fontsize=14)
    ax.set_ylabel("Occupation", fontsize=14)
    ax.set_title(None)

    plt.xticks(rotation=45)
    plt.legend(title="AI ROLE", bbox_to_anchor=(1.02, 1), loc="upper left")
    plt.subplots_adjust(bottom=0.2, left=0.1, right=0.9, top=0.9)
    plt.tight_layout()
    plt.savefig(f"results/figures_2026/percent_by_occupation_colors_2024_{benefit}.png")
    # plt.show()


def generate_benefits_by_occupation_plots(usdf):
    """
    Generate Figure 5: Percent by occupation (percent_by_occupation_colors_2024_{benefit}.png)
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


def main():
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

    print("All figures generated successfully!")


if __name__ == "__main__":
    main()
