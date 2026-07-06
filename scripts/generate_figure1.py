#!/usr/bin/env python3
"""Calculate and generate Figure 1 (both aggregate and firm-category variants).

This is the single, memory-safe entry point for Figure 1 on the full sample. It
streams the posts CSV instead of loading it, so it runs without the full
descriptive pipeline (and without the full body file). It reproduces the same
calculations as analysis/descriptive_analysis.py:
  - quarterly AI demand (overall, industry-average, and by firm category)
  - quarterly adjusted AI log-wage coefficients (aggregate and by firm category),
    OLS of LOG_SALARY on AI ROLE + edu + experience + 3-digit NAICS + state (HC1),
    suppressed when a quarter has < 10 AI wage postings.

Dependencies (produced once by scripts/label_ai_roles_full.py):
  data/processed/ai_skill_counts_by_id.parquet   (AI-role posting IDs)

Outputs:
  results/tables_2026/descriptive/quarterly_ai_demand_by_firm_category.csv
  results/tables_2026/descriptive/quarterly_ai_wage_betas_by_firm_category.csv
  results/tables_2026/descriptive/quarterly_ai_wage_betas.csv
  results/figures_2026/descriptive/figure1_ai_demand_wage_beta.png            (aggregate)
  results/figures_2026/descriptive/firm_categories/
      figure1_ai_demand_wage_beta_by_firm_category.png                        (by firm category)

Usage:
  uv run python scripts/generate_figure1.py
  uv run python scripts/generate_figure1.py --posts-csv data/OII_US_10M_POSTS_MAY26_SUBSAMPLE.csv
"""
from __future__ import annotations

import argparse
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyarrow.csv as pacsv
import statsmodels.api as sm

REPO = Path(__file__).resolve().parent.parent
TABLES = REPO / "results" / "tables_2026" / "descriptive"
FIG_ROOT = REPO / "results" / "figures_2026" / "descriptive"

FIRM_CATEGORY_ORDER = ["SMEs", "Large firms", "S&P 500 firms"]
SME_CUTOFF = 50  # FIRM_POSTING_COUNT < 50 => SME, else Large (matches descriptive_analysis)
CONTROLS = ["MIN_EDULEVELS_NAME", "EXPERIENCE_BUCKET", "NAICS_2022_3_DIGIT", "STATE_NAME"]
EXP_BINS = [-2, -1, 0, 2, 5, 10, 20, 100]
EXP_LABELS = ["Missing", "0 years", "1-2 years", "3-5 years", "6-10 years", "11-20 years", "21+ years"]
GENAI_CUTOFF = pd.Period("2022Q4", freq="Q")
MIN_AI_WAGE = 10

READ = pacsv.ReadOptions(use_threads=True, block_size=128 << 20)
PARSE = pacsv.ParseOptions(newlines_in_values=True)


# --------------------------------------------------------------------------- #
# Calculation
# --------------------------------------------------------------------------- #
def company_posting_counts(posts_csv: Path) -> pd.Series:
    """Pass 1: FIRM_POSTING_COUNT = postings per classified COMPANY (full sample)."""
    counts: defaultdict = defaultdict(int)
    reader = pacsv.open_csv(posts_csv, read_options=READ, parse_options=PARSE,
                            convert_options=pacsv.ConvertOptions(include_columns=["COMPANY"]))
    for batch in reader:
        comp = pd.to_numeric(batch.column(0).to_pandas(), errors="coerce")
        comp = comp[comp.notna() & (comp != 0)].astype("int64")
        for cid, n in comp.value_counts().items():
            counts[cid] += int(n)
    return pd.Series(counts, dtype="int64")


def categorize(comp: pd.Series, firm_count: pd.Series, sp_ids: set) -> pd.Series:
    """Mutually exclusive firm category: S&P 500 -> SME (<50) / Large (>=50)."""
    n = comp.map(firm_count)
    cat = pd.Series(pd.NA, index=comp.index, dtype="object")
    valid = comp.notna() & (comp != 0) & n.notna()
    cat[valid & (n < SME_CUTOFF)] = "SMEs"
    cat[valid & (n >= SME_CUTOFF)] = "Large firms"
    cat[comp.isin(sp_ids)] = "S&P 500 firms"
    return cat


def collect(posts_csv: Path, ai_ids: set, firm_count: pd.Series, sp_ids: set) -> dict:
    """Pass 2: demand counters (overall/industry/by-category) + wage-valid rows."""
    overall = defaultdict(lambda: [0, 0])          # quarter -> [postings, ai]
    industry = defaultdict(lambda: [0, 0])         # (quarter, naics2) -> [postings, ai]
    cat_demand = defaultdict(lambda: [0, 0])       # (quarter, category) -> [postings, ai]
    wage_parts = []

    reader = pacsv.open_csv(posts_csv, read_options=READ, parse_options=PARSE,
        convert_options=pacsv.ConvertOptions(include_columns=[
            "ID", "COMPANY", "MIN_EDULEVELS_NAME", "MIN_YEARS_EXPERIENCE",
            "SALARY", "STATE_NAME", "NAICS_2022_6", "YEAR", "QUARTER"]))
    for batch in reader:
        df = batch.to_pandas()
        yr = pd.to_numeric(df["YEAR"], errors="coerce")
        q = pd.to_numeric(df["QUARTER"], errors="coerce")
        keep = yr.notna() & q.notna()
        df = df[keep]
        df["QSTR"] = (yr[keep].astype(int).astype(str) + "Q" + q[keep].astype(int).astype(str)).values
        df["AI"] = df["ID"].isin(ai_ids)
        naics = pd.to_numeric(df["NAICS_2022_6"], errors="coerce").astype("Int64").astype(str)
        df["NAICS2"] = naics.str[:2].values
        df["NAICS_2022_3_DIGIT"] = naics.str[:3].values
        comp = pd.to_numeric(df["COMPANY"], errors="coerce")
        df["FIRM_CATEGORY"] = categorize(comp, firm_count, sp_ids).values

        # overall demand (all postings)
        for qs, sub in df.groupby("QSTR", observed=True):
            overall[qs][0] += len(sub); overall[qs][1] += int(sub["AI"].sum())
        # industry-average demand (2-digit NAICS, drop missing)
        ind = df[df["NAICS2"] != "<N"]
        for (qs, n2), s in ind.groupby(["QSTR", "NAICS2"], observed=True):
            industry[(qs, n2)][0] += len(s); industry[(qs, n2)][1] += int(s["AI"].sum())
        # by-category demand (company-attributed)
        cat = df[df["FIRM_CATEGORY"].notna()]
        for (qs, c), s in cat.groupby(["QSTR", "FIRM_CATEGORY"], observed=True):
            cat_demand[(qs, c)][0] += len(s); cat_demand[(qs, c)][1] += int(s["AI"].sum())

        # wage-valid rows (LOG_SALARY/edu/state non-null; experience/naics never null)
        sal = pd.to_numeric(df["SALARY"], errors="coerce")
        wok = sal.notna() & (sal > 0) & df["MIN_EDULEVELS_NAME"].notna() & df["STATE_NAME"].notna()
        w = df[wok].copy()
        if not w.empty:
            w["LOG_SALARY"] = np.log(sal[wok].to_numpy())
            eb = pd.cut(pd.to_numeric(w["MIN_YEARS_EXPERIENCE"], errors="coerce"),
                        bins=EXP_BINS, labels=EXP_LABELS, right=True).astype(str)
            w["EXPERIENCE_BUCKET"] = eb.replace("nan", "None Listed")
            wage_parts.append(w[["QSTR", "FIRM_CATEGORY", "LOG_SALARY", "AI"] + CONTROLS])

    return {"overall": overall, "industry": industry, "cat_demand": cat_demand,
            "wage": pd.concat(wage_parts, ignore_index=True)}


def estimate_quarter(group: pd.DataFrame) -> dict:
    """OLS of LOG_SALARY on AI ROLE + control dummies (HC1) for one quarter group."""
    ai = int(group["AI"].sum())
    row = dict(wage_postings=len(group), ai_wage_postings=ai, ai_role_beta=np.nan,
               ai_role_se=np.nan, lower_ci=np.nan, upper_ci=np.nan,
               status="suppressed_sparse_ai_wage_postings")
    if ai < MIN_AI_WAGE:
        return row
    X = group[["AI"]].astype(float).rename(columns={"AI": "AI ROLE"})
    for c in CONTROLS:
        X = pd.concat([X, pd.get_dummies(group[c], prefix=c, drop_first=True, dtype=float)], axis=1)
    X = sm.add_constant(X, has_constant="add")
    try:
        m = sm.OLS(group["LOG_SALARY"].astype(float), X.astype(float)).fit(cov_type="HC1")
        b, se = m.params["AI ROLE"], m.bse["AI ROLE"]
        row.update(ai_role_beta=b, ai_role_se=se, lower_ci=b - 1.96 * se,
                   upper_ci=b + 1.96 * se, status="ok")
    except Exception as e:  # noqa: BLE001 - record any singular-matrix/fit error
        row["status"] = f"error: {e}"
    return row


def quarterly_betas(wage: pd.DataFrame) -> pd.DataFrame:
    """Aggregate quarterly betas over all wage-valid rows."""
    rows = []
    for q, grp in wage.groupby("P", observed=True):
        r = estimate_quarter(grp); r["QUARTER"] = str(q); rows.append(r)
    return pd.DataFrame(rows).sort_values("QUARTER")


def quarterly_betas_by_category(wage: pd.DataFrame) -> pd.DataFrame:
    """Quarterly betas estimated separately within each firm category."""
    rows = []
    for cat in FIRM_CATEGORY_ORDER:
        sub = wage[wage["FIRM_CATEGORY"] == cat]
        for q, grp in sub.groupby("P", observed=True):
            r = estimate_quarter(grp); r["QUARTER"] = str(q); r["FIRM_CATEGORY"] = cat
            rows.append(r)
    return pd.DataFrame(rows).sort_values(["FIRM_CATEGORY", "QUARTER"])


def demand_frame(counter: dict) -> pd.DataFrame:
    """quarter -> % AI from a {quarter: [postings, ai]} counter."""
    df = pd.DataFrame([(k, 100 * v[1] / v[0]) for k, v in counter.items()],
                      columns=["QUARTER", "share"])
    df["P"] = pd.PeriodIndex(df["QUARTER"], freq="Q")
    return df.sort_values("P").reset_index(drop=True)


def industry_average_frame(counter: dict) -> pd.DataFrame:
    """quarter -> mean per-industry % AI from a {(quarter, naics2): [postings, ai]} counter."""
    df = pd.DataFrame([(k[0], 100 * v[1] / v[0]) for k, v in counter.items()],
                      columns=["QUARTER", "share"])
    ia = df.groupby("QUARTER")["share"].mean().reset_index()
    ia["P"] = pd.PeriodIndex(ia["QUARTER"], freq="Q")
    return ia.sort_values("P").reset_index(drop=True)


# --------------------------------------------------------------------------- #
# Aggregate figure (no firm-size split)
# --------------------------------------------------------------------------- #
def chatgpt_line(ax, pos, add_label=False):
    if GENAI_CUTOFF not in pos:
        return
    cutoff_x = pos[GENAI_CUTOFF] + 2 / 3
    ax.axvline(cutoff_x, color="black", lw=2.0, ls=":", alpha=0.9, zorder=5)
    if add_label:
        ax.text(cutoff_x + 0.15, 0.34, "ChatGPT", transform=ax.get_xaxis_transform(),
                rotation=90, ha="left", va="bottom", fontsize=11, fontweight="bold", color="black")


def inside_title(ax, text):
    ax.text(0.5, 0.96, text, transform=ax.transAxes, ha="center", va="top",
            fontsize=15, fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor="none", alpha=0.7))


def plot_aggregate(overall: pd.DataFrame, industry_avg: pd.DataFrame, betas: pd.DataFrame):
    dem = overall.rename(columns={"share": "Overall"}).merge(
        industry_avg.rename(columns={"share": "Industry Average"})[["P", "Industry Average"]], on="P")
    dem[["QUARTER", "Overall", "Industry Average"]].to_csv(
        TABLES / "quarterly_ai_demand.csv", index=False)
    periods = dem["P"].tolist()
    pos = {p: i for i, p in enumerate(periods)}
    x = np.arange(len(periods))

    fig, (dax, wax) = plt.subplots(2, 1, figsize=(11, 9), sharex=True)
    dax.plot(x, dem["Overall"], lw=1.8, label="Overall")
    dax.plot(x, dem["Industry Average"], lw=1.8, ls="--", label="Industry Average")
    dax.legend(loc="upper left", fontsize=14)
    dax.set_ylabel("% AI Roles", fontsize=15)
    inside_title(dax, "Demand for AI Skills")
    chatgpt_line(dax, pos, add_label=True); dax.grid(alpha=0.25)

    b = betas[betas["status"] == "ok"].copy()
    b["P"] = pd.PeriodIndex(b["QUARTER"], freq="Q")
    b["x"] = b["P"].map(pos)
    post = b["P"] >= pd.Period("2022Q1", freq="Q")
    wax.fill_between(b.loc[post, "x"], b.loc[post, "lower_ci"], b.loc[post, "upper_ci"],
                     color="#0072B2", alpha=0.14, lw=0)
    wax.plot(b["x"], b["ai_role_beta"], marker="o", color="#0072B2", lw=1.7, markersize=4)
    wax.axhline(0, color="gray", lw=0.8, ls="--")
    chatgpt_line(wax, pos)
    wax.set_ylabel("AI-Role Wage Coefficient\n(log points)", fontsize=13)
    inside_title(wax, "Adjusted AI-Skills Wage Premium")
    wax.grid(alpha=0.25)

    ticks = list(range(0, len(periods), 4))
    wax.set_xticks(ticks); wax.set_xticklabels([str(periods[i]) for i in ticks], rotation=45)
    wax.set_xlabel("Quarter", fontsize=13)
    plt.tight_layout()
    out = FIG_ROOT / "figure1_ai_demand_wage_beta.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, bbox_inches="tight", dpi=300)
    plt.close(fig)
    print(f"Saved aggregate figure: {out}")


# --------------------------------------------------------------------------- #
def load_sp500_ids(snapshot: Path) -> set:
    snap = pd.read_csv(snapshot)
    return set(pd.to_numeric(snap["COMPANY"], errors="coerce").dropna().astype("int64"))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--posts-csv", type=Path, default=REPO / "data" / "OII_US_10M_POSTS_MAY26.csv")
    ap.add_argument("--snapshot", type=Path, default=REPO / "data" / "sp500_snapshot.csv")
    ap.add_argument("--ai-counts", type=Path,
                    default=REPO / "data" / "processed" / "ai_skill_counts_by_id.parquet")
    args = ap.parse_args()

    if not args.ai_counts.exists():
        sys.exit(f"AI labels not found: {args.ai_counts}\n"
                 "Run scripts/label_ai_roles_full.py first.")

    ai_ids = set(pd.read_parquet(args.ai_counts)["ID"].tolist())
    sp_ids = load_sp500_ids(args.snapshot)
    print(f"AI-role postings: {len(ai_ids):,} | S&P 500 firms: {len(sp_ids):,}")

    print("Pass 1: firm posting counts...")
    firm_count = company_posting_counts(args.posts_csv)
    print(f"  classified firms: {len(firm_count):,}")

    print("Pass 2: demand + wage rows...")
    data = collect(args.posts_csv, ai_ids, firm_count, sp_ids)
    wage = data["wage"]
    wage["P"] = pd.PeriodIndex(wage["QSTR"], freq="Q")
    print(f"  wage-valid rows: {len(wage):,}")

    TABLES.mkdir(parents=True, exist_ok=True)

    # --- by-category demand + betas CSVs (consumed by plot_firm_category_figure1) ---
    cat_rows = [(k[0], k[1], 100 * v[1] / v[0]) for k, v in data["cat_demand"].items()]
    cat_wide = (pd.DataFrame(cat_rows, columns=["QUARTER", "FIRM_CATEGORY", "share"])
                .pivot(index="QUARTER", columns="FIRM_CATEGORY", values="share")
                .reindex(columns=FIRM_CATEGORY_ORDER))
    cat_wide.index = pd.PeriodIndex(cat_wide.index, freq="Q")
    cat_wide = cat_wide.sort_index()
    cat_wide.index = cat_wide.index.astype(str)
    cat_wide.to_csv(TABLES / "quarterly_ai_demand_by_firm_category.csv")

    quarterly_betas_by_category(wage).to_csv(
        TABLES / "quarterly_ai_wage_betas_by_firm_category.csv", index=False)

    # --- aggregate betas CSV + aggregate figure ---
    agg_betas = quarterly_betas(wage)
    agg_betas.to_csv(TABLES / "quarterly_ai_wage_betas.csv", index=False)
    plot_aggregate(demand_frame(data["overall"]), industry_average_frame(data["industry"]), agg_betas)

    # --- by-category figure: reuse the existing plot script (reads the CSVs above) ---
    sys.path.append(str(REPO / "scripts"))
    import plot_firm_category_figure1 as pf1
    pf1.main()


if __name__ == "__main__":
    main()
