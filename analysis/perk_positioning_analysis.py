#!/usr/bin/env python3
"""
Perk Positioning Analysis (Step 8)

Tests whether AI postings place perks more or less prominently than non-AI
postings, conditional on mentioning the perk at all.

Prominence score (0-100): 100 = top of posting, 0 = bottom of posting.
Analysis is restricted to postings that mention the benefit (position is NaN
otherwise).

Model: OLS — POSITION ~ AI_ROLE + Year FE + Industry FE + Education FE + Experience FE

Outputs:
- results/tables_2026/perk_positioning_results.csv
- results/figures_2026/perk_positioning_ai_coef.png
"""

import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import statsmodels.api as sm
from pathlib import Path

REPO_ROOT_PATH = Path(__file__).resolve().parent.parent
sys.path.append(str(REPO_ROOT_PATH))
sys.path.append(str(REPO_ROOT_PATH / "src"))
from package_files.benefits_defns import (
    benefits4,
    benefits4_labels,
    benefit_colors,
    industry,
    year,
)
from package_files.config_utils import get_processed_dir, get_repo_root
from scripts.label_benefits import (
    check_benefit_position,
    culture,
    edu_assistance,
    health_wellbeing,
    leave,
    parental_leave,
    tuition_to_exclude,
    wellbeing_to_exclude,
)
from scripts.label_benefits_remote import remote_keywords, remote_to_exclude

plt.rcParams.update({"font.size": 13})

REPO_ROOT = get_repo_root()
PROCESSED_DIR = get_processed_dir()
OUTPUT_FIGS = REPO_ROOT / "results" / "figures_2026"
OUTPUT_TABLES = REPO_ROOT / "results" / "tables_2026"

POSITION_COLS = {
    "EDU_ASSISTANCE":    "EDU_ASSISTANCE_POSITION",
    "PAID LEAVE":        "PAID_LEAVE_POSITION",
    "HEALTH_WELLBEING":  "HEALTH_WELLBEING_POSITION",
    "PARENTAL_LEAVE":    "PARENTAL_LEAVE_POSITION",
    "CULTURE":           "CULTURE_POSITION",
    "REMOTE_KW":         "REMOTE_KW_POSITION",
}

POSITION_KEYWORDS = {
    "EDU_ASSISTANCE": (edu_assistance, tuition_to_exclude),
    "PAID LEAVE": (leave, None),
    "HEALTH_WELLBEING": (health_wellbeing, wellbeing_to_exclude),
    "PARENTAL_LEAVE": (parental_leave, None),
    "CULTURE": (culture, None),
    "REMOTE_KW": (remote_keywords, remote_to_exclude),
}


def ensure_position_columns(df):
    """Add missing prominence columns from BODY using the existing keyword rules."""
    missing = [col for col in POSITION_COLS.values() if col not in df.columns]
    if not missing:
        return df

    if "BODY" not in df.columns:
        raise KeyError(
            "Missing perk prominence columns and BODY is unavailable, so positions "
            f"cannot be computed. Missing columns: {', '.join(missing)}"
        )

    print(
        "Computing missing perk prominence columns from BODY: "
        + ", ".join(missing)
    )

    for benefit, pos_col in POSITION_COLS.items():
        if pos_col in df.columns:
            continue

        keywords, exclusions = POSITION_KEYWORDS[benefit]
        if benefit in df.columns:
            mask = df[benefit].fillna(False).astype(bool)
        else:
            mask = pd.Series(True, index=df.index)
            print(f"  Warning: {benefit} flag missing; scanning all postings")

        df[pos_col] = np.nan
        if not mask.any():
            print(f"  {pos_col}: no matching {benefit} rows")
            continue

        positions = check_benefit_position(df.loc[mask, "BODY"], keywords, exclusions)
        df.loc[mask, pos_col] = pd.Series(
            positions.astype("float64"),
            index=df.index[mask],
        )

        valid = df[pos_col].notna().sum()
        print(f"  {pos_col}: {valid:,} matches")

    return df


def load_data():
    """Load and prepare the configured analysis dataset."""
    print("Loading analysis dataset...")

    data_path = PROCESSED_DIR / "labeled_v2.parquet"
    if not data_path.exists():
        print(f"Warning: {data_path} not found. Please update the path.")
        return None

    df = pd.read_parquet(data_path)
    print(f"Loaded {len(df):,} rows from {data_path}")

    # Consolidate rare industries
    ind_counts = df[industry].value_counts()
    small = ind_counts[ind_counts < 30].index
    df[industry] = df[industry].replace(small, "Other")

    df = ensure_position_columns(df)

    return df


def run_ols(df, position_col, predictor="AI ROLE"):
    """OLS: position ~ AI_ROLE + Year + Industry FE.
    Only rows where the benefit was mentioned (position is not NaN).
    """
    sub = df[df[position_col].notna()].copy()

    # Drop 2018 for consistency with regression_models.py
    sub = sub[sub[year] != 2018]

    controls = [year, industry]
    ref = {}

    X = sub[[predictor]].astype(float)

    for c in controls:
        sub[c] = sub[c].astype("category")
        if c in ref and ref[c] in sub[c].cat.categories:
            cats = [ref[c]] + [x for x in sub[c].cat.categories if x != ref[c]]
            sub[c] = sub[c].cat.reorder_categories(cats, ordered=True)
        dummies = pd.get_dummies(sub[c], drop_first=True)
        X = pd.concat([X, dummies], axis=1)

    X = X.apply(pd.to_numeric, errors="coerce")
    y = sub[position_col].astype(float)

    combined = pd.concat([X, y], axis=1).dropna()
    X = combined.drop(columns=[position_col])
    y = combined[position_col].astype("float64")
    X.columns = X.columns.astype(str)
    X = sm.add_constant(X.astype("float64"))

    model = sm.OLS(y, X).fit()
    return model, len(y)


def print_descriptive(df):
    print("\nMean prominence score by AI ROLE (postings that mention the benefit):")
    rows = []
    for ben, pos_col in POSITION_COLS.items():
        label = benefits4_labels[benefits4.index(ben)]
        grp = df[df[pos_col].notna()].groupby("AI ROLE")[pos_col].agg(["mean", "count"])
        rows.append({
            "Benefit": label,
            "Non-AI mean": grp.loc[False, "mean"] if False in grp.index else np.nan,
            "Non-AI n": grp.loc[False, "count"] if False in grp.index else 0,
            "AI mean": grp.loc[True, "mean"] if True in grp.index else np.nan,
            "AI n": grp.loc[True, "count"] if True in grp.index else 0,
        })
    desc = pd.DataFrame(rows)
    desc["Diff (AI - Non-AI)"] = (desc["AI mean"] - desc["Non-AI mean"]).round(2)
    desc["Non-AI mean"] = desc["Non-AI mean"].round(2)
    desc["AI mean"] = desc["AI mean"].round(2)
    print(desc.to_string(index=False))
    return desc


def run_all_models(df):
    results = []
    for ben in benefits4:
        pos_col = POSITION_COLS[ben]
        label = benefits4_labels[benefits4.index(ben)]
        print(f"\n  {label} ({pos_col})")

        model, n = run_ols(df, pos_col)

        coef = model.params.get("AI ROLE", np.nan)
        se   = model.bse.get("AI ROLE", np.nan)
        pval = model.pvalues.get("AI ROLE", np.nan)

        def stars(p):
            if pd.isna(p): return ""
            if p < 0.01: return "***"
            if p < 0.05: return "**"
            if p < 0.1:  return "*"
            return ""

        print(f"    AI ROLE coef: {coef:.3f}{stars(pval)}  SE: {se:.3f}  p: {pval:.3f}  n: {n:,}  R²: {model.rsquared:.3f}")

        results.append({
            "benefit": ben,
            "benefit_label": label,
            "coef": coef,
            "se": se,
            "pvalue": pval,
            "stars": stars(pval),
            "n": n,
            "r2": model.rsquared,
        })

    return pd.DataFrame(results)


def save_table(results_df, desc_df):
    os.makedirs(OUTPUT_TABLES, exist_ok=True)

    results_df["coef_str"] = results_df.apply(
        lambda r: f"{r['coef']:.3f}{r['stars']}" if not pd.isna(r["coef"]) else "—",
        axis=1,
    )
    results_df["se_str"] = results_df["se"].apply(
        lambda s: f"({s:.3f})" if not pd.isna(s) else ""
    )

    out = OUTPUT_TABLES / "perk_positioning_results.csv"
    results_df.to_csv(out, index=False)
    print(f"\nSaved: {out}")

    print("\nOLS AI ROLE coefficient on perk prominence (conditional on mention):")
    print(results_df[["benefit_label", "coef_str", "se_str", "n", "r2"]].to_string(index=False))


def plot_results(results_df):
    os.makedirs(OUTPUT_FIGS, exist_ok=True)

    results_df = results_df.copy()
    results_df["lower"] = results_df["coef"] - 1.96 * results_df["se"]
    results_df["upper"] = results_df["coef"] + 1.96 * results_df["se"]

    fig, ax = plt.subplots(figsize=(10, 5))

    x = np.arange(len(benefits4))
    colors = [benefit_colors[b] for b in benefits4]

    for i, (_, row) in enumerate(results_df.iterrows()):
        ax.errorbar(
            i, row["coef"],
            yerr=[[row["coef"] - row["lower"]], [row["upper"] - row["coef"]]],
            fmt="o", color=colors[i], markersize=9, capsize=5,
            label=row["benefit_label"],
        )
        # significance annotation
        if row["stars"]:
            ax.annotate(
                row["stars"],
                xy=(i, row["upper"]),
                xytext=(0, 4), textcoords="offset points",
                ha="center", fontsize=11, color=colors[i],
            )

    ax.axhline(0, color="black", linewidth=0.8, linestyle="--")
    ax.set_xticks(x)
    ax.set_xticklabels(results_df["benefit_label"], rotation=25, ha="right", fontsize=11)
    ax.set_ylabel("OLS coefficient on prominence score\n(AI ROLE, conditional on mention, 95% CI)", fontsize=11)
    ax.set_title("Are perks placed more or less prominently in AI job postings?", fontsize=13)
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.1f"))

    plt.tight_layout()
    out = OUTPUT_FIGS / "perk_positioning_ai_coef.png"
    plt.savefig(out, bbox_inches="tight", dpi=150)
    plt.close()
    print(f"Saved: {out}")


def main():
    print("Loading data...")
    df = load_data()
    if df is None:
        print("Error: Could not load data. Please check the data path.")
        return

    print("\nDescriptive statistics:")
    desc_df = print_descriptive(df)

    print("\nRunning OLS models...")
    results_df = run_all_models(df)

    print("\nSaving outputs...")
    save_table(results_df, desc_df)
    plot_results(results_df)

    print("\nDone.")


if __name__ == "__main__":
    main()
