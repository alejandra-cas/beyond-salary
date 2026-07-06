#!/usr/bin/env python3
"""Generate combined paper figures 3 and 4 from saved results CSVs.

Both figures are assembled purely from results tables committed in the repo,
so this script needs no access to the raw labeled data:

- Figure 3: logit AI-role coefficients across model specifications (left, 3a)
  and yearly Model 1 AI-role coefficients (right, 3b).
- Figure 4: median salary by benefit and role type over time (left, 4a) and
  yearly full-sample AI x perk wage interaction coefficients (right, 4b).

Figure 4a requires ``results/tables_2026/salary/salary_by_benefit_stats.csv``,
which ``analysis/salary_analysis.py`` exports on a full-data run.
"""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

REPO_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from package_files.benefits_defns import (  # noqa: E402
    benefit_colors,
    benefits4,
    benefits4_labels,
    benefits_labels_map,
)

TABLES_DIR = REPO_ROOT / "results" / "tables_2026"
OUTPUT_DIR = REPO_ROOT / "results" / "figures_2026" / "paper"

MODEL_PANELS_CSV = TABLES_DIR / "regression" / "model_panels_ai_role_long.csv"
YEARLY_MODEL1_CSV = TABLES_DIR / "regression" / "yearly_model1_ai_role_coefficients.csv"
SALARY_STATS_CSV = TABLES_DIR / "salary" / "salary_by_benefit_stats.csv"
WAGE_PERK_CSV = TABLES_DIR / "wage_perk_interactions" / "wage_perk_interaction_results.csv"

# Okabe-Ito palette used by regression_models.py
colors = ['#E69F00', '#56B4E9', '#009E73', '#CC79A7', '#0072B2', '#D55E00', '#009E73']
GENAI_CUTOFF_YEAR = 2022.875

# Figure 4b y-axis is capped so a single wide CI band does not blow up the scale.
FIG4B_YMAX = 0.2


def add_panel_label(ax, label):
    ax.text(-0.06, 1.04, label, transform=ax.transAxes,
            fontsize=16, fontweight="bold", va="bottom", ha="right")


def plot_model_coefficients(ax, results):
    """Panel 3a: AI-role log-odds coefficients across model specifications."""
    results = results.copy()
    results["lower_ci"] = results["ai_role_coef"] - 1.96 * results["ai_role_se"]
    results["upper_ci"] = results["ai_role_coef"] + 1.96 * results["ai_role_se"]

    model_labels = list(results["model_label"].drop_duplicates())
    markers = ['o', 's', '^', 'D', 'P', 'X', 'v']
    benefit_ticks, benefit_tick_labels = [], []
    handles, labels = [], []
    current_pos = 0

    for benefit in benefits4:
        benefit_data = results[results["benefit"] == benefit]
        positions = []
        for i, model in enumerate(model_labels):
            row = benefit_data[benefit_data["model_label"] == model]
            if row.empty or row["ai_role_coef"].isna().all():
                continue
            row = row.iloc[0]
            pos = current_pos + i * 0.2
            handle = ax.errorbar(
                pos, row["ai_role_coef"],
                yerr=[[row["ai_role_coef"] - row["lower_ci"]],
                      [row["upper_ci"] - row["ai_role_coef"]]],
                fmt=markers[i % len(markers)], color=colors[i % len(colors)],
            )
            positions.append(pos)
            if model not in labels:
                handles.append(handle)
                labels.append(model)
        if positions:
            benefit_ticks.append((min(positions) + max(positions)) / 2)
            benefit_tick_labels.append(benefits4_labels[benefits4.index(benefit)])
        current_pos += len(model_labels) + 1

    ax.set_xticks(benefit_ticks)
    ax.set_xticklabels(benefit_tick_labels, rotation=45, fontsize=15, ha="right")
    ax.tick_params(axis="y", labelsize=14)
    ax.set_ylabel("AI-Role Log-Odds Coefficient (95% CI)", fontsize=16)
    ax.axhline(0, color="grey", linewidth=0.8)
    ax.legend(handles, labels, title="Model", loc="upper left", fontsize=12, title_fontsize=12)
    ax.grid(alpha=0.2)


def plot_yearly_model1(ax, results):
    """Panel 3b: yearly Model 1 AI-role log-odds coefficients by benefit."""
    plot_df = results[results["status"] == "ok"].copy()
    for i, benefit in enumerate(benefits4):
        line = plot_df[plot_df["benefit"] == benefit].sort_values("period")
        if line.empty:
            continue
        color = colors[i % len(colors)]
        ax.fill_between(line["period"], line["lower_ci"], line["upper_ci"],
                        color=color, alpha=0.12, linewidth=0)
        ax.plot(line["period"], line["ai_role_coef"], marker="o", markersize=4,
                linewidth=1.7, color=color,
                label=benefits4_labels[benefits4.index(benefit)])

    ax.axhline(0, color="gray", linewidth=0.8, linestyle="--")
    ax.axvline(GENAI_CUTOFF_YEAR, color="black", linewidth=1.0, linestyle=":", alpha=0.75)
    ax.text(GENAI_CUTOFF_YEAR + 0.03, ax.get_ylim()[1], "Nov. 2022",
            ha="left", va="top", fontsize=9, color="black")
    ax.set_xlabel("Year", fontsize=16)
    ax.set_ylabel("AI-Role Log-Odds Coefficient", fontsize=16)
    ax.set_xticks(sorted(plot_df["period"].unique()))
    ax.tick_params(labelsize=14)
    ax.legend(title=None, fontsize=12, ncols=2)
    ax.grid(alpha=0.2)


def generate_figure3():
    if not (MODEL_PANELS_CSV.exists() and YEARLY_MODEL1_CSV.exists()):
        print("Skipping Figure 3: regression results CSVs not found.")
        return
    panels = pd.read_csv(MODEL_PANELS_CSV)
    panels = panels[(panels["panel"] == "Main") & (panels["model_status"] == "ok")]
    yearly = pd.read_csv(YEARLY_MODEL1_CSV)

    fig, (ax_a, ax_b) = plt.subplots(
        1, 2, figsize=(18, 8.5), gridspec_kw={"width_ratios": [1.6, 1]}
    )
    plot_model_coefficients(ax_a, panels)
    plot_yearly_model1(ax_b, yearly)
    add_panel_label(ax_a, "(a)")
    add_panel_label(ax_b, "(b)")

    plt.tight_layout()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output = OUTPUT_DIR / "figure3_combined.png"
    plt.savefig(output, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved: {output}")


def plot_salary_panel(ax, benefit_data, benefit, show_legend):
    """One subpanel of 4a: median salary lines for a single benefit."""
    for ai_role in [True, False]:
        for benefit_present in [True, False]:
            subset = benefit_data[
                (benefit_data["AI ROLE"] == ai_role)
                & (benefit_data["Benefit_Present"] == benefit_present)
            ].sort_values("YEAR")
            if subset.empty:
                continue
            color = benefit_colors[benefit] if ai_role else "gray"
            alpha = 1.0 if benefit_present else 0.6
            linestyle = "-" if benefit_present else "--"
            label = (f"{'AI Role' if ai_role else 'Non-AI Role'}, "
                     f"{'With Benefit' if benefit_present else 'Without Benefit'}")
            ax.plot(subset["YEAR"], subset["Median_Salary"], color=color, alpha=alpha,
                    linestyle=linestyle, marker="o", markersize=3, label=label)
            ax.fill_between(subset["YEAR"], subset["CI_Lower"], subset["CI_Upper"],
                            color=color, alpha=0.2)

    ax.set_title(benefits_labels_map[benefit], fontsize=12, fontweight="bold")
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f"${x/1000:.0f}K"))
    years = sorted(benefit_data["YEAR"].unique())
    if years:
        ax.set_xticks(years)
        ax.set_xlim(min(years) - 0.5, max(years) + 0.5)
    ax.tick_params(labelsize=9)
    ax.tick_params(axis="x", rotation=45)
    if show_legend:
        ax.legend(fontsize=8, loc="upper left")
    ax.grid(True, alpha=0.3)


def plot_wage_perk_yearly(ax, results):
    """Panel 4b: full-sample yearly AI x perk interaction coefficients."""
    yearly = results[
        results["period"].astype(str).str.match(r"^\d{4}$")
        & (results["status"] == "ok")
        & (results["sample"] == "Full sample")
    ].copy()
    yearly["year"] = yearly["period"].astype(int)

    for perk in benefits4:
        line = yearly[yearly["perk"] == perk].sort_values("year")
        if line.empty:
            continue
        color = benefit_colors.get(perk)
        ax.fill_between(line["year"], line["beta3_lower_ci"], line["beta3_upper_ci"],
                        alpha=0.10, color=color, linewidth=0)
        ax.plot(line["year"], line["beta3_interaction"], marker="o", markersize=4.5,
                linewidth=1.9, color=color, label=benefits_labels_map[perk])

    ax.axhline(0, color="gray", linewidth=0.8, linestyle="--")
    ax.axvline(GENAI_CUTOFF_YEAR, color="black", linewidth=1.0, linestyle=":", alpha=0.75)
    ax.set_ylim(top=FIG4B_YMAX)
    ax.text(GENAI_CUTOFF_YEAR + 0.03, ax.get_ylim()[1], "Nov. 2022",
            ha="left", va="top", fontsize=9, color="black")
    ax.set_xlabel("Year", fontsize=15)
    ax.set_ylabel(r"AI Role $\times$ Benefit Interaction Coefficient", fontsize=15)
    ax.tick_params(labelsize=13)
    ax.legend(title="Benefit", fontsize=11, title_fontsize=12, loc="lower left")
    ax.grid(alpha=0.2)


def generate_figure4():
    if not WAGE_PERK_CSV.exists():
        print("Skipping Figure 4: wage-perk interaction results CSV not found.")
        return
    if not SALARY_STATS_CSV.exists():
        print(
            f"Skipping Figure 4: {SALARY_STATS_CSV} not found.\n"
            "Run analysis/salary_analysis.py on the full data first to export it."
        )
        return
    salary_stats = pd.read_csv(SALARY_STATS_CSV)
    wage_perk = pd.read_csv(WAGE_PERK_CSV)

    fig = plt.figure(figsize=(24, 12.5))
    subfig_a, subfig_b = fig.subfigures(1, 2, width_ratios=[1.6, 1])

    axes = subfig_a.subplots(2, 3)
    for i, benefit in enumerate(benefits4):
        ax = axes.flatten()[i]
        plot_salary_panel(ax, salary_stats[salary_stats["Benefit"] == benefit],
                          benefit, show_legend=(i == 0))
        if i % 3 == 0:
            ax.set_ylabel("Median Annual Salary (USD)", fontsize=11)
    add_panel_label(axes[0][0], "(a)")

    ax_b = subfig_b.subplots(1, 1)
    plot_wage_perk_yearly(ax_b, wage_perk)
    add_panel_label(ax_b, "(b)")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output = OUTPUT_DIR / "figure4_combined.png"
    plt.savefig(output, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved: {output}")


def main():
    generate_figure3()
    generate_figure4()


if __name__ == "__main__":
    main()
