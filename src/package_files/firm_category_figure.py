"""Shared helpers for the mixed-frequency firm-category wage panel."""

import numpy as np
import pandas as pd


ANNUAL_WAGE_END_YEAR = 2020
QUARTERLY_WAGE_START_YEAR = 2021


def draw_quarterly_wage_series(
    ax,
    quarterly_betas,
    quarter_position_map,
    categories,
    colors,
    markers,
    confidence_bands_from="2022Q1",
):
    """Draw the original all-quarterly wage series."""
    quarterly = quarterly_betas.copy()
    if not isinstance(quarterly["QUARTER"].dtype, pd.PeriodDtype):
        quarterly["QUARTER"] = quarterly["QUARTER"].map(
            lambda value: pd.Period(value, freq="Q")
        )
    if "status" in quarterly.columns:
        quarterly = quarterly[quarterly["status"] == "ok"].copy()
    quarterly["x"] = quarterly["QUARTER"].map(quarter_position_map)
    quarterly = quarterly.dropna(subset=["x"])
    band_start = pd.Period(confidence_bands_from, freq="Q")

    for category in categories:
        line = quarterly[quarterly["FIRM_CATEGORY"] == category].sort_values(
            "QUARTER"
        )
        if line.empty:
            continue
        band = line[line["QUARTER"] >= band_start]
        ax.fill_between(
            band["x"],
            band["lower_ci"],
            band["upper_ci"],
            color=colors[category],
            alpha=0.12,
            linewidth=0,
        )
        ax.plot(
            line["x"],
            line["ai_role_beta"],
            marker=markers[category],
            markersize=4,
            linewidth=1.5,
            color=colors[category],
            label=category,
        )
    return quarterly


def prepare_mixed_wage_series(annual_betas, quarterly_betas, quarter_position_map):
    """Return annual pre-2021 and quarterly post-2020 estimates on one x-axis."""
    annual = annual_betas.copy()
    if "FIRM_CATEGORY" not in annual.columns and "firm_category" in annual.columns:
        annual = annual.rename(columns={"firm_category": "FIRM_CATEGORY"})
    if "YEAR" not in annual.columns:
        period = annual["period"].astype(str)
        annual = annual[period.str.fullmatch(r"\d{4}", na=False)].copy()
        annual["YEAR"] = pd.to_numeric(annual["period"], errors="coerce")
    if "status" in annual.columns:
        annual = annual[annual["status"] == "ok"].copy()
    annual = annual[annual["YEAR"].le(ANNUAL_WAGE_END_YEAR)].copy()
    annual["x"] = annual["YEAR"].map(
        lambda year: quarter_position_map.get(
            pd.Period(f"{int(year)}Q2", freq="Q"), np.nan
        )
        + 0.5
    )
    annual = annual.dropna(subset=["x"])

    quarterly = quarterly_betas.copy()
    if not isinstance(quarterly["QUARTER"].dtype, pd.PeriodDtype):
        quarterly["QUARTER"] = quarterly["QUARTER"].map(
            lambda value: pd.Period(value, freq="Q")
        )
    if "status" in quarterly.columns:
        quarterly = quarterly[quarterly["status"] == "ok"].copy()
    quarterly = quarterly[
        quarterly["QUARTER"].map(lambda period: period.year)
        >= QUARTERLY_WAGE_START_YEAR
    ].copy()
    quarterly["x"] = quarterly["QUARTER"].map(quarter_position_map)
    quarterly = quarterly.dropna(subset=["x"])
    return annual, quarterly


def draw_mixed_wage_series(
    ax,
    annual_betas,
    quarterly_betas,
    quarter_position_map,
    categories,
    colors,
    markers,
):
    """Draw annual and quarterly estimates as visibly separate line segments."""
    annual, quarterly = prepare_mixed_wage_series(
        annual_betas,
        quarterly_betas,
        quarter_position_map,
    )

    for category in categories:
        color = colors[category]
        marker = markers[category]
        annual_line = annual[annual["FIRM_CATEGORY"] == category].sort_values("YEAR")
        quarterly_line = quarterly[
            quarterly["FIRM_CATEGORY"] == category
        ].sort_values("QUARTER")

        if not annual_line.empty:
            lower_error = annual_line["ai_role_beta"] - annual_line["lower_ci"]
            upper_error = annual_line["upper_ci"] - annual_line["ai_role_beta"]
            ax.errorbar(
                annual_line["x"],
                annual_line["ai_role_beta"],
                yerr=[lower_error, upper_error],
                marker=marker,
                markersize=5,
                linewidth=1.5,
                linestyle="--",
                capsize=2.5,
                color=color,
                alpha=0.9,
                label=category if quarterly_line.empty else None,
            )

        if not quarterly_line.empty:
            ax.fill_between(
                quarterly_line["x"],
                quarterly_line["lower_ci"],
                quarterly_line["upper_ci"],
                color=color,
                alpha=0.08,
                linewidth=0,
            )
            ax.plot(
                quarterly_line["x"],
                quarterly_line["ai_role_beta"],
                marker=marker,
                markersize=4,
                linewidth=1.5,
                linestyle="-",
                color=color,
                label=category,
            )

    transition = quarter_position_map.get(
        pd.Period(f"{QUARTERLY_WAGE_START_YEAR}Q1", freq="Q")
    )
    if transition is not None:
        transition -= 0.5
        ax.axvline(
            transition,
            color="#777777",
            linewidth=1.0,
            linestyle="--",
            alpha=0.65,
            zorder=0,
        )
        ax.text(
            transition - 0.25,
            0.975,
            f"Annual estimates, through {ANNUAL_WAGE_END_YEAR}",
            transform=ax.get_xaxis_transform(),
            ha="right",
            va="top",
            fontsize=8.5,
            color="#555555",
        )
        ax.text(
            transition + 0.25,
            0.975,
            f"Quarterly estimates, {QUARTERLY_WAGE_START_YEAR} onward",
            transform=ax.get_xaxis_transform(),
            ha="left",
            va="top",
            fontsize=8.5,
            color="#555555",
        )

    return annual, quarterly


def set_zero_inclusive_wage_ylim(ax, annual, quarterly, lower_pad=0.03, upper_pad=0.08):
    """Use a compact coefficient scale that still shows the zero reference line."""
    plotted = pd.concat([annual, quarterly], ignore_index=True)
    values = pd.concat(
        [
            plotted["ai_role_beta"],
            plotted["lower_ci"],
            plotted["upper_ci"],
            pd.Series([0.0]),
        ],
        ignore_index=True,
    ).dropna()
    if values.empty:
        return
    lower = min(0.0, float(values.min()))
    upper = max(0.0, float(values.max()))
    span = max(upper - lower, 0.02)
    ax.set_ylim(lower - lower_pad * span, upper + upper_pad * span)


def set_quarterly_wage_ylim(ax, quarterly, confidence_bands_from="2022Q1"):
    """Reproduce the original quarterly coefficient scale."""
    band_start = pd.Period(confidence_bands_from, freq="Q")
    band = quarterly[quarterly["QUARTER"] >= band_start]
    values = pd.concat(
        [quarterly["ai_role_beta"], band["lower_ci"], band["upper_ci"]],
        ignore_index=True,
    ).dropna()
    if values.empty:
        return
    limit = max(float(values.abs().max()) * 1.08, 0.02)
    ax.set_ylim(-limit, limit)
    ax.set_ylim(bottom=-0.1)
