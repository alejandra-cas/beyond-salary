#!/usr/bin/env python3
"""
Regression Models Analysis

This script runs the job-level logistic regression models.  
Notes:
- PARENTAL_LEAVE models exclude 2018 data #TODO why?
- Industry grouping for small categories
- Includes proper reference category handling

Generates:
1. Model coefficients plot
2. Individual benefit regression tables (properly formatted)
3. Final wide table with all results
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import sys
import os
import argparse
import pickle
from pathlib import Path
from scipy.stats import norm

# Add src directory to path
sys.path.append(str(Path(__file__).parent.parent / 'src'))

from package_files.logit_model import run_logit_model
from package_files.benefits_defns import *
from package_files.config_utils import get_processed_dir, get_repo_root
from collections import defaultdict

# Configuration
plt.rcParams.update({'font.size': 14})

REPO_ROOT = get_repo_root()
PROCESSED_DIR = get_processed_dir()
RESULTS_DIR = REPO_ROOT / "results"
TABLES_DIR = RESULTS_DIR / "tables_2026" / "regression"
MODEL_CACHE_DIR = TABLES_DIR / "model_cache"
CACHE_VERSION = "v4_preferred_sp500_no_firm_fe"
CACHE_FORMAT = "compact_results_only"
FIRM_CATEGORY_LOGIT_DIR = RESULTS_DIR / "tables_2026" / "firm_category_logit"

# Field mappings
region = 'STATE_NAME'
industry = 'NAICS_2022_3_DIGIT'
firm = 'COMPANY'
education = 'MIN_EDULEVELS_NAME'
year = 'YEAR'
occupation = 'SOC_MAJOR_GROUP'
experience = 'EXPERIENCE_BUCKET'

# Benefits to analyze
benefits4 = ['EDU_ASSISTANCE', 'PAID LEAVE', 'HEALTH_WELLBEING', 'PARENTAL_LEAVE', 'CULTURE', 'REMOTE_KW']
benefits4_labels = ['Tuition Assistance', 'Paid Leave', 'Health and Wellbeing', 'Parental Leave', 'Workplace Culture', 'Remote Work']

# Color scheme for plots
colors = ['#E69F00', '#56B4E9', '#009E73', '#CC79A7', '#0072B2', '#D55E00', '#009E73']
firm_category_colors = {
    "SMEs": "#1f77b4",
    "Large firms": "#2ca02c",
    "S&P 500 firms": "#d62728",
}
firm_category_markers = {
    "SMEs": "^",
    "Large firms": "s",
    "S&P 500 firms": "o",
}
GENAI_CUTOFF_YEAR = 2022.875
ROBUST_CI_WIDTH_MULTIPLIER = 6
ROBUST_CI_WIDTH_FLOOR = 2.0
ROBUST_Y_PADDING = 0.12


def set_robust_logit_ylim(ax, plot_rows, panel_label):
    """Set subplot y-limits without letting extreme confidence intervals dominate."""
    if not plot_rows:
        return

    panel = pd.concat(plot_rows, ignore_index=True)
    for column in ["ai_role_coef", "lower_ci", "upper_ci"]:
        panel[column] = pd.to_numeric(panel[column], errors="coerce")
    finite = np.isfinite(panel[["ai_role_coef", "lower_ci", "upper_ci"]]).all(axis=1)
    panel = panel[finite].dropna(subset=["ai_role_coef", "lower_ci", "upper_ci"])
    if panel.empty:
        return

    ci_width = panel["upper_ci"] - panel["lower_ci"]
    finite_width = ci_width[np.isfinite(ci_width) & (ci_width >= 0)]
    if finite_width.empty:
        values = panel["ai_role_coef"].to_numpy(dtype=float)
    else:
        median_width = finite_width.median()
        max_reasonable_width = max(
            ROBUST_CI_WIDTH_FLOOR,
            ROBUST_CI_WIDTH_MULTIPLIER * median_width,
        )
        reasonable = panel[ci_width <= max_reasonable_width]
        if not reasonable.empty:
            values = np.concatenate(
                [
                    reasonable["ai_role_coef"].to_numpy(dtype=float),
                    reasonable["lower_ci"].to_numpy(dtype=float),
                    reasonable["upper_ci"].to_numpy(dtype=float),
                ]
            )
        else:
            values = panel["ai_role_coef"].to_numpy(dtype=float)

    values = values[np.isfinite(values)]
    if len(values) == 0:
        return

    lower = min(values.min(), 0)
    upper = max(values.max(), 0)
    if np.isclose(lower, upper):
        lower -= 0.5
        upper += 0.5
    padding = max((upper - lower) * ROBUST_Y_PADDING, 0.1)
    lower -= padding
    upper += padding

    current_lower, current_upper = ax.get_ylim()
    if lower > current_lower or upper < current_upper:
        print(
            f"Warning: clipped extreme confidence intervals in {panel_label} "
            f"plot axis to [{lower:.2f}, {upper:.2f}]."
        )
    ax.set_ylim(lower, upper)


def add_firm_category(data):
    """Add mutually exclusive SME, large, and S&P 500 firm categories."""
    data = data.copy()
    sp500 = data["SP500"].fillna(False).astype(bool) if "SP500" in data else pd.Series(False, index=data.index)
    firm_posting_count = (
        pd.to_numeric(data["FIRM_POSTING_COUNT"], errors="coerce")
        if "FIRM_POSTING_COUNT" in data
        else pd.Series(np.nan, index=data.index)
    )

    data["FIRM_CATEGORY"] = pd.NA
    data.loc[(~sp500) & (firm_posting_count < 50), "FIRM_CATEGORY"] = "SMEs"
    data.loc[(~sp500) & (firm_posting_count >= 50), "FIRM_CATEGORY"] = "Large firms"
    data.loc[sp500, "FIRM_CATEGORY"] = "S&P 500 firms"
    data["FIRM_CATEGORY"] = pd.Categorical(
        data["FIRM_CATEGORY"],
        categories=["SMEs", "Large firms", "S&P 500 firms"],
        ordered=True,
    )
    return data


def collapse_sparse_fixed_effects(data, dependent, require_salary=False, min_outcomes=5):
    """Collapse thin NAICS 3-digit and state cells to keep logit models estimable."""
    data_model = data.copy()
    count_data = data_model.dropna(subset=[dependent])
    if require_salary:
        count_data = count_data.dropna(subset=['LOG_SALARY'])

    for column, other_label in [
        (industry, 'Other NAICS 3-digit'),
        (region, 'Other State'),
    ]:
        summary = count_data.groupby(column, dropna=False)[dependent].agg(['count', 'sum'])
        sparse = summary[
            (summary['count'] < 50)
            | (summary['sum'] < min_outcomes)
            | ((summary['count'] - summary['sum']) < min_outcomes)
        ].index
        data_model[column] = data_model[column].replace(sparse, other_label)
    return data_model


def prepare_firm_fixed_effects(data, min_firm_obs=30):
    """Reduce firm cardinality for FE model to avoid exploding dummy matrices."""
    data_fe = data.copy()
    data_fe[firm] = data_fe[firm].fillna('Unknown Firm').astype(str)
    data_fe[region] = data_fe[region].fillna('Unknown State').astype(str)

    firm_counts = data_fe[firm].value_counts()
    rare_firms = firm_counts[firm_counts < min_firm_obs].index
    data_fe[firm] = data_fe[firm].replace(rare_firms, 'Other Firm (<30 obs)')

    print(
        f"Firm FE prep: {len(firm_counts):,} firms -> "
        f"{data_fe[firm].nunique():,} categories after grouping firms with <{min_firm_obs} obs"
    )
    return data_fe


def parse_args():
    """Parse command-line options for cache behavior."""
    parser = argparse.ArgumentParser(description="Run job-level regression models.")
    parser.add_argument(
        "--refresh-cache",
        action="store_true",
        help="Refit all models and overwrite cached model results.",
    )
    parser.add_argument(
        "--cache-path",
        type=Path,
        default=None,
        help="Optional path for the model-results cache file.",
    )
    parser.add_argument(
        "--yearly-period-fallback",
        action="store_true",
        help=(
            "If any yearly Model 1 coefficient fails, also run two period-pooled "
            "Model 1 specifications: through 2022 and post-2022."
        ),
    )
    return parser.parse_args()


def get_model_cache_path(data_path):
    """Build a cache path tied to the current model spec and input data file."""
    try:
        data_mtime_ns = data_path.stat().st_mtime_ns
    except FileNotFoundError:
        data_mtime_ns = "missing"

    filename = (
        f"job_level_models_{CACHE_VERSION}_"
        f"data{data_mtime_ns}.pkl"
    )
    return MODEL_CACHE_DIR / filename


def load_cached_models(cache_path):
    """Load cached fitted models when available."""
    if not cache_path.exists():
        return None

    print(f"Loading cached model results from {cache_path}")
    with open(cache_path, "rb") as f:
        payload = pickle.load(f)

    if (
        payload.get("cache_version") != CACHE_VERSION
        or payload.get("cache_format") != CACHE_FORMAT
    ):
        print("Cached model results use an old cache format/version; refitting models.")
        return None

    return deserialize_compact_model_results(payload["models_2024"])


def save_cached_models(cache_path, models_2024):
    """Persist compact model summaries so cache files omit fit data and samples."""
    os.makedirs(cache_path.parent, exist_ok=True)
    payload = {
        "cache_version": CACHE_VERSION,
        "cache_format": CACHE_FORMAT,
        "models_2024": serialize_compact_model_results(models_2024),
    }
    with open(cache_path, "wb") as f:
        pickle.dump(payload, f)
    print(f"Cached model results saved to {cache_path}")


class CompactLogitResults:
    """Cache-safe result object that omits raw design matrices and sampled data."""

    def __init__(self, model):
        self.params = model.params.copy()
        self.bse = model.bse.copy()
        self.pvalues = model.pvalues.copy()
        self.converged = bool(getattr(model, "converged", False))
        self.nobs = float(getattr(model, "nobs", np.nan))
        self.prsquared = float(getattr(model, "prsquared", np.nan))
        self.model_name = getattr(model, "model_name", type(model).__name__)

        optional_attrs = [
            "dropped_groups",
            "used_groups",
            "successful_fit_count",
            "requested_repeats",
            "sample_size",
            "seed_start",
            "seed_end",
            "model_note",
        ]
        for attr in optional_attrs:
            if hasattr(model, attr):
                setattr(self, attr, getattr(model, attr))

        if hasattr(model, "subsample_coef_sd"):
            self.subsample_coef_sd = model.subsample_coef_sd.copy()
        if hasattr(model, "average_model_bse"):
            self.average_model_bse = model.average_model_bse.copy()

    @classmethod
    def from_payload(cls, payload):
        obj = cls.__new__(cls)
        obj.params = payload["params"]
        obj.bse = payload["bse"]
        obj.pvalues = payload["pvalues"]
        obj.converged = payload["converged"]
        obj.nobs = payload["nobs"]
        obj.prsquared = payload["prsquared"]
        obj.model_name = payload["model_name"]

        for attr, value in payload.get("optional_attrs", {}).items():
            setattr(obj, attr, value)

        if payload.get("subsample_coef_sd") is not None:
            obj.subsample_coef_sd = payload["subsample_coef_sd"]
        if payload.get("average_model_bse") is not None:
            obj.average_model_bse = payload["average_model_bse"]

        return obj

    def to_payload(self):
        optional_attr_names = [
            "dropped_groups",
            "used_groups",
            "successful_fit_count",
            "requested_repeats",
            "sample_size",
            "seed_start",
            "seed_end",
            "model_note",
        ]
        return {
            "params": self.params,
            "bse": self.bse,
            "pvalues": self.pvalues,
            "converged": self.converged,
            "nobs": self.nobs,
            "prsquared": self.prsquared,
            "model_name": self.model_name,
            "optional_attrs": {
                attr: getattr(self, attr)
                for attr in optional_attr_names
                if hasattr(self, attr)
            },
            "subsample_coef_sd": getattr(self, "subsample_coef_sd", None),
            "average_model_bse": getattr(self, "average_model_bse", None),
        }

    def conf_int(self, alpha=0.05):
        critical_value = norm.ppf(1 - alpha / 2)
        lower = self.params - critical_value * self.bse
        upper = self.params + critical_value * self.bse
        return pd.DataFrame({0: lower, 1: upper})

    def summary(self):
        summary_df = pd.DataFrame(
            {
                "coef": self.params,
                "std err": self.bse,
                "p>|z|": self.pvalues,
            }
        )
        return summary_df.to_string(float_format=lambda x: f"{x:0.4f}")


def compact_model_results(models_2024):
    """Strip fitted model objects down to cache-safe result summaries."""
    compact_panels = []
    for panel in models_2024:
        compact_panel = []
        for model in panel:
            if isinstance(model, str):
                compact_panel.append(model)
            else:
                compact_panel.append(CompactLogitResults(model))
        compact_panels.append(compact_panel)
    return compact_panels


def serialize_compact_model_results(models_2024):
    """Serialize compact model results as plain payloads for robust pickling."""
    serialized_panels = []
    for panel in compact_model_results(models_2024):
        serialized_panel = []
        for model in panel:
            if isinstance(model, str):
                serialized_panel.append(model)
            else:
                serialized_panel.append(model.to_payload())
        serialized_panels.append(serialized_panel)
    return serialized_panels


def deserialize_compact_model_results(serialized_models):
    """Rebuild compact result objects from cache payloads."""
    panels = []
    for panel in serialized_models:
        restored_panel = []
        for model_payload in panel:
            if isinstance(model_payload, str):
                restored_panel.append(model_payload)
            else:
                restored_panel.append(CompactLogitResults.from_payload(model_payload))
        panels.append(restored_panel)
    return panels


def load_and_prepare_data():
    """Load and prepare the analysis dataset with all preprocessing."""
    print("Loading analysis dataset...")
    
    # Load the data
    data_path = PROCESSED_DIR / "labeled_v2.parquet"
    if not data_path.exists():
        print(f"Warning: {data_path} not found. Please update the path.")
        return None

    # Using full dataset (not balanced/even sample)
    data = pd.read_parquet(data_path)

    data[region] = data[region].fillna('Unknown State').astype(str)

    # Replace small industries with "Other" (this is also done in run_logit_model but we do it here for consistency)
    print("Processing industry categories...")
    industry_counts = data['NAICS_2022_2_NAME'].value_counts()
    small_industries = industry_counts[industry_counts < 30].index  
    data['NAICS_2022_2_NAME'] = data['NAICS_2022_2_NAME'].replace(small_industries, 'Other')
    
    print(f"Final dataset shape: {data.shape}")
    print(f"Year distribution:")
    print(data.groupby('YEAR').size())
    
    return data


def run_2024_models(data):
    """Run the four H1 perk-prevalence regression specifications."""
    
    print("="*80)
    print("RUNNING REGRESSION MODELS")
    print("="*80)
    
    # Model 1: Baseline (Year + NAICS 3-digit fixed effects)
    print("\n" + "="*50)
    print("MODEL 1: BASELINE (YEAR + NAICS 3-DIGIT FIXED EFFECTS)")
    print("="*50)
    
    benefit_models_industry = []
    for benefit in benefits4:
        print(f"\n{benefit}")
        print('-'*100)
        model_data = collapse_sparse_fixed_effects(data, benefit)
        model = run_logit_model(model_data, dependent=benefit, predictor='AI ROLE',
                              cat_controls=[year, industry], get_vif=False)
        benefit_models_industry.append(model)
    
    # Model 2: Add individual controls and state fixed effects
    print("\n" + "="*50)  
    print("MODEL 2: WITH INDIVIDUAL CONTROLS")
    print("="*50)
    
    benefit_models_industry_2 = []
    for benefit in benefits4:
        print(f"\n{benefit}")
        print('-'*100)
        model_data = collapse_sparse_fixed_effects(data, benefit)
        model = run_logit_model(model_data, dependent=benefit, predictor='AI ROLE',
                              cat_controls=[year, industry, region, education, experience],
                              ref_category={education: "No Education Listed", experience: 'None Listed'}, 
                              get_vif=False)
        benefit_models_industry_2.append(model)
    
    # Model 3: Add salary control
    print("\n" + "="*50)
    print("MODEL 3: WITH SALARY CONTROL") 
    print("="*50)
    
    benefit_models_industry_3 = []
    for benefit in benefits4:
        print(f"\n{benefit}")
        print('-'*100)
        model_data = collapse_sparse_fixed_effects(data, benefit, require_salary=True)
        model = run_logit_model(model_data, dependent=benefit, predictor='AI ROLE',
                              cat_controls=[year, industry, region, education, experience],
                              cont_controls=['LOG_SALARY'], 
                              ref_category={education: "No Education Listed", experience: 'None Listed'})
        benefit_models_industry_3.append(model)

    # Model 4: Preferred specification with S&P 500 indicator
    print("\n" + "="*50)
    print("MODEL 4: PREFERRED SPECIFICATION (+ S&P 500)")
    print("="*50)

    benefit_models_sp500_4 = []
    for benefit in benefits4:
        print(f"\n{benefit}")
        print('-'*100)
        model_data = collapse_sparse_fixed_effects(data, benefit, require_salary=True)
        model = run_logit_model(
            model_data,
            dependent=benefit,
            predictor='AI ROLE',
            cat_controls=[year, industry, region, education, experience],
            cont_controls=['LOG_SALARY'],
            binary_vars=['SP500'],
            ref_category={education: "No Education Listed", experience: 'None Listed'},
            get_vif=False,
        )
        benefit_models_sp500_4.append(model)

    return [
        benefit_models_industry,
        benefit_models_industry_2,
        benefit_models_industry_3,
        benefit_models_sp500_4,
    ]


def extract_single_model_row(model, firm_category, benefit, benefit_label):
    """Extract the AI-role log-odds coefficient from one fitted logit model."""
    if isinstance(model, str):
        return {
            "firm_category": firm_category,
            "benefit": benefit,
            "benefit_label": benefit_label,
            "status": "error",
            "ai_role_coef": np.nan,
            "ai_role_se": np.nan,
            "ai_role_pvalue": np.nan,
            "lower_ci": np.nan,
            "upper_ci": np.nan,
            "nobs": np.nan,
            "pseudo_r2": np.nan,
            "converged": False,
            "specification": "",
        }

    coef = model.params.get("AI ROLE", np.nan)
    se = model.bse.get("AI ROLE", np.nan)
    return {
        "firm_category": firm_category,
        "benefit": benefit,
        "benefit_label": benefit_label,
        "status": "ok",
        "ai_role_coef": coef,
        "ai_role_se": se,
        "ai_role_pvalue": model.pvalues.get("AI ROLE", np.nan),
        "lower_ci": coef - 1.96 * se,
        "upper_ci": coef + 1.96 * se,
        "nobs": model.nobs,
        "pseudo_r2": model.prsquared,
        "converged": model.converged,
        "specification": getattr(model, "firm_category_specification", ""),
    }


def fit_firm_category_logit_model(category_data, firm_category, benefit, include_year_fe=True):
    """Fit a common comparable firm-category logit specification."""
    cat_controls = [industry, region, education, experience]
    specification = "naics3_state_education_experience_salary"
    if include_year_fe:
        cat_controls = [year] + cat_controls
        specification = "year_" + specification
    model_data = collapse_sparse_fixed_effects(
        category_data,
        benefit,
        require_salary=True,
    )
    try:
        model = run_logit_model(
            model_data,
            dependent=benefit,
            predictor="AI ROLE",
            cat_controls=cat_controls,
            cont_controls=["LOG_SALARY"],
            ref_category={
                education: "No Education Listed",
                experience: "None Listed",
            },
            get_vif=False,
        )
    except Exception as error:
        return f"error: {error}"
    if not isinstance(model, str):
        model.firm_category_specification = specification
    return model


def run_firm_category_logit_models(data):
    """
    Run preferred benefit logit models separately for SMEs, large firms, and S&P 500 firms.

    The coefficient of interest is AI ROLE: the log-odds difference in benefit
    prevalence for AI roles within each firm category.
    """
    print("\n" + "=" * 50)
    print("FIRM-CATEGORY LOGIT MODELS")
    print("=" * 50)

    data = add_firm_category(data)
    firm_categories = ["SMEs", "Large firms", "S&P 500 firms"]
    rows = []

    for firm_category in firm_categories:
        category_data = data[data["FIRM_CATEGORY"] == firm_category].copy()
        print(f"\n{firm_category}: {len(category_data):,} postings")
        for benefit, benefit_label in zip(benefits4, benefits4_labels):
            print(f"  {benefit}")
            model = fit_firm_category_logit_model(category_data, firm_category, benefit)
            rows.append(extract_single_model_row(model, firm_category, benefit, benefit_label))

    results = pd.DataFrame(rows)
    os.makedirs(FIRM_CATEGORY_LOGIT_DIR, exist_ok=True)
    output = FIRM_CATEGORY_LOGIT_DIR / "firm_category_ai_role_logit_results.csv"
    results.to_csv(output, index=False)
    print(f"Firm-category logit results saved: {output}")
    return results


def plot_firm_category_logit_results(results):
    """Plot AI-role log-odds coefficients from firm-category logit models."""
    plot_df = results[results["status"] == "ok"].copy()
    if plot_df.empty:
        print("No firm-category logit results to plot.")
        return

    fig, ax = plt.subplots(figsize=(12, 6))
    firm_categories = ["SMEs", "Large firms", "S&P 500 firms"]
    x_base = np.arange(len(benefits4))
    offsets = np.linspace(-0.22, 0.22, len(firm_categories))

    for offset, firm_category in zip(offsets, firm_categories):
        subset = (
            plot_df[plot_df["firm_category"] == firm_category]
            .set_index("benefit")
            .reindex(benefits4)
        )
        x = x_base + offset
        ax.errorbar(
            x,
            subset["ai_role_coef"],
            yerr=[
                subset["ai_role_coef"] - subset["lower_ci"],
                subset["upper_ci"] - subset["ai_role_coef"],
            ],
            fmt=firm_category_markers[firm_category],
            capsize=4,
            label=firm_category,
            color=firm_category_colors[firm_category],
        )

    ax.axhline(0, color="gray", linewidth=0.8, linestyle="--")
    ax.set_xticks(x_base)
    ax.set_xticklabels(benefits4_labels, rotation=45, ha="right")
    ax.set_ylabel("AI-Role Log-Odds Coefficient (95% CI)")
    ax.set_xlabel(None)
    ax.legend(title="Firm Category", fontsize=10, title_fontsize=10)
    ax.grid(axis="y", alpha=0.2)
    plt.tight_layout()

    output_dir = RESULTS_DIR / "figures_2026" / "regression"
    os.makedirs(output_dir, exist_ok=True)
    output = output_dir / "firm_category_ai_role_logit_coefficients.png"
    plt.savefig(output, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Firm-category logit plot saved: {output}")


def extract_firm_category_time_row(model, firm_category, period, benefit, benefit_label, specification):
    """Extract a firm-category AI-role coefficient for a year or period model."""
    row = extract_single_model_row(model, firm_category, benefit, benefit_label)
    row["period"] = period
    row["specification"] = specification
    return row


def run_firm_category_yearly_logit_models(data):
    """Run preferred firm-category benefit logits separately by firm category and year."""
    print("\n" + "=" * 50)
    print("YEARLY FIRM-CATEGORY LOGIT MODELS")
    print("=" * 50)

    data = add_firm_category(data)
    firm_categories = ["SMEs", "Large firms", "S&P 500 firms"]
    rows = []
    specification = "yearly_naics3_state_education_experience_salary"

    for firm_category in firm_categories:
        category_data = data[data["FIRM_CATEGORY"] == firm_category].copy()
        print(f"\n{firm_category}: {len(category_data):,} postings")
        for model_year in sorted(category_data[year].dropna().unique()):
            year_data = category_data[category_data[year] == model_year].copy()
            print(f"  {int(model_year)}: {len(year_data):,} postings")
            for benefit, benefit_label in zip(benefits4, benefits4_labels):
                print(f"    {benefit}")
                model = fit_firm_category_logit_model(
                    year_data,
                    firm_category,
                    benefit,
                    include_year_fe=False,
                )
                rows.append(
                    extract_firm_category_time_row(
                        model,
                        firm_category,
                        int(model_year),
                        benefit,
                        benefit_label,
                        specification,
                    )
                )

    results = pd.DataFrame(rows)
    os.makedirs(FIRM_CATEGORY_LOGIT_DIR, exist_ok=True)
    output = FIRM_CATEGORY_LOGIT_DIR / "firm_category_yearly_ai_role_logit_results.csv"
    results.to_csv(output, index=False)
    print(f"Yearly firm-category logit results saved: {output}")
    return results


def firm_category_yearly_needs_period_fallback(results):
    """Return True when any expected firm-category yearly coefficient is missing."""
    expected_rows = (
        results["firm_category"].nunique()
        * results["period"].nunique()
        * len(benefits4)
    )
    ok_rows = results[
        (results["status"] == "ok")
        & results["ai_role_coef"].notna()
        & results["ai_role_se"].notna()
    ]
    return len(ok_rows) < expected_rows


def run_firm_category_period_logit_models(data):
    """Run firm-category benefit logits through 2022 and post-2022."""
    print("\n" + "=" * 50)
    print("PERIOD FIRM-CATEGORY LOGIT MODELS")
    print("=" * 50)

    data = add_firm_category(data)
    firm_categories = ["SMEs", "Large firms", "S&P 500 firms"]
    periods = [
        ("Through 2022", lambda df: df[year] <= 2022),
        ("Post-2022", lambda df: df[year] > 2022),
    ]
    rows = []
    specification = "period_year_naics3_state_education_experience_salary"

    for firm_category in firm_categories:
        category_data = data[data["FIRM_CATEGORY"] == firm_category].copy()
        print(f"\n{firm_category}: {len(category_data):,} postings")
        for period_label, period_filter in periods:
            period_data = category_data[period_filter(category_data)].copy()
            print(f"  {period_label}: {len(period_data):,} postings")
            for benefit, benefit_label in zip(benefits4, benefits4_labels):
                print(f"    {benefit}")
                model = fit_firm_category_logit_model(
                    period_data,
                    firm_category,
                    benefit,
                    include_year_fe=True,
                )
                rows.append(
                    extract_firm_category_time_row(
                        model,
                        firm_category,
                        period_label,
                        benefit,
                        benefit_label,
                        specification,
                    )
                )

    results = pd.DataFrame(rows)
    os.makedirs(FIRM_CATEGORY_LOGIT_DIR, exist_ok=True)
    output = FIRM_CATEGORY_LOGIT_DIR / "firm_category_period_ai_role_logit_results.csv"
    results.to_csv(output, index=False)
    print(f"Period firm-category logit results saved: {output}")
    return results


def plot_firm_category_time_logit_results(results, output_name, title, period_order=None):
    """Plot firm-category coefficients over time with shaded confidence bands."""
    plot_df = results[results["status"] == "ok"].copy()
    if plot_df.empty:
        print(f"No firm-category time results to plot for {output_name}.")
        return

    if period_order is None:
        period_order = sorted(plot_df["period"].unique())
    x_map = {period: idx for idx, period in enumerate(period_order)}

    fig, axes = plt.subplots(2, 3, figsize=(16, 9), sharex=True)
    axes = axes.flatten()
    firm_categories = ["SMEs", "Large firms", "S&P 500 firms"]

    for ax, benefit in zip(axes, benefits4):
        subset = plot_df[plot_df["benefit"] == benefit].copy()
        plotted_rows = []
        for firm_category in firm_categories:
            line = (
                subset[subset["firm_category"] == firm_category]
                .set_index("period")
                .reindex(period_order)
                .dropna(subset=["ai_role_coef"])
                .reset_index()
            )
            if line.empty:
                continue
            plotted_rows.append(line)
            x = line["period"].map(x_map).astype(float)
            color = firm_category_colors[firm_category]
            ax.fill_between(
                x,
                line["lower_ci"].astype(float),
                line["upper_ci"].astype(float),
                color=color,
                alpha=0.12,
                linewidth=0,
            )
            ax.plot(
                x,
                line["ai_role_coef"].astype(float),
                marker=firm_category_markers[firm_category],
                markersize=4,
                linewidth=1.6,
                color=color,
                label=firm_category,
            )

        ax.axhline(0, color="gray", linewidth=0.8, linestyle="--")
        if all(isinstance(period, (int, np.integer)) for period in period_order):
            cutoff_x = np.interp(GENAI_CUTOFF_YEAR, period_order, list(range(len(period_order))))
            ax.axvline(cutoff_x, color="black", linewidth=1.0, linestyle=":", alpha=0.75)
        ax.set_title(benefit_label_for_plot(benefit), fontsize=12)
        ax.set_ylabel("AI-Role Log-Odds Coef.")
        ax.grid(alpha=0.2)
        set_robust_logit_ylim(ax, plotted_rows, benefit_label_for_plot(benefit))

    axes[-1].legend(title="Firm Category", fontsize=9, title_fontsize=10, loc="upper left", bbox_to_anchor=(1.02, 1))
    for ax in axes:
        ax.set_xticks(range(len(period_order)))
        ax.set_xticklabels([str(period) for period in period_order], rotation=45, ha="right")

    fig.suptitle(title, fontsize=14)
    plt.tight_layout()
    output_dir = RESULTS_DIR / "figures_2026" / "regression"
    os.makedirs(output_dir, exist_ok=True)
    output = output_dir / output_name
    plt.savefig(output, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Firm-category time plot saved: {output}")


def extract_ai_role_logit_row(model, period, benefit, benefit_label, specification):
    """Extract one AI-role coefficient row from a fitted logit model."""
    row = {
        "period": period,
        "benefit": benefit,
        "benefit_label": benefit_label,
        "specification": specification,
        "status": "ok",
        "ai_role_coef": np.nan,
        "ai_role_se": np.nan,
        "ai_role_pvalue": np.nan,
        "lower_ci": np.nan,
        "upper_ci": np.nan,
        "nobs": np.nan,
        "pseudo_r2": np.nan,
        "converged": False,
    }
    if isinstance(model, str):
        row["status"] = "error"
        row["detail"] = model
        return row

    coef = model.params.get("AI ROLE", np.nan)
    se = model.bse.get("AI ROLE", np.nan)
    row.update({
        "ai_role_coef": coef,
        "ai_role_se": se,
        "ai_role_pvalue": model.pvalues.get("AI ROLE", np.nan),
        "lower_ci": coef - 1.96 * se,
        "upper_ci": coef + 1.96 * se,
        "nobs": model.nobs,
        "pseudo_r2": model.prsquared,
        "converged": model.converged,
    })
    return row


def run_yearly_model1_ai_role_coefficients(data):
    """
    Run the first H1 regression separately by year.

    The pooled Model 1 is benefit ~ AI ROLE + year FE + NAICS3 FE. Within each
    calendar year, the comparable yearly model drops the year FE and keeps NAICS3 FE.
    """
    print("\n" + "=" * 50)
    print("YEARLY MODEL 1 AI-ROLE COEFFICIENTS")
    print("=" * 50)
    rows = []
    specification = "yearly_model1_ai_role_naics3_fe"

    for model_year in sorted(data[year].dropna().unique()):
        year_data = data[data[year] == model_year].copy()
        print(f"\n{int(model_year)}: {len(year_data):,} postings")
        for benefit, benefit_label in zip(benefits4, benefits4_labels):
            print(f"  {benefit}")
            model_data = collapse_sparse_fixed_effects(year_data, benefit)
            model = run_logit_model(
                model_data,
                dependent=benefit,
                predictor="AI ROLE",
                cat_controls=[industry],
                get_vif=False,
            )
            rows.append(
                extract_ai_role_logit_row(
                    model,
                    int(model_year),
                    benefit,
                    benefit_label,
                    specification,
                )
            )

    results = pd.DataFrame(rows)
    output = TABLES_DIR / "yearly_model1_ai_role_coefficients.csv"
    os.makedirs(TABLES_DIR, exist_ok=True)
    results.to_csv(output, index=False)
    print(f"Yearly Model 1 coefficients saved: {output}")
    return results


def plot_yearly_model1_ai_role_coefficients(results):
    """Plot yearly Model 1 AI-role log-odds coefficients by benefit."""
    plot_df = results[results["status"] == "ok"].copy()
    if plot_df.empty:
        print("No yearly Model 1 results to plot.")
        return

    fig, ax = plt.subplots(figsize=(11, 6))
    for i, benefit in enumerate(benefits4):
        line = plot_df[plot_df["benefit"] == benefit].sort_values("period")
        if line.empty:
            continue
        color = colors[i % len(colors)]
        ax.fill_between(
            line["period"],
            line["lower_ci"],
            line["upper_ci"],
            color=color,
            alpha=0.12,
            linewidth=0,
        )
        ax.plot(
            line["period"],
            line["ai_role_coef"],
            marker="o",
            markersize=4,
            linewidth=1.7,
            color=color,
            label=benefit_label_for_plot(benefit),
        )

    ax.axhline(0, color="gray", linewidth=0.8, linestyle="--")
    ax.axvline(GENAI_CUTOFF_YEAR, color="black", linewidth=1.0, linestyle=":", alpha=0.75)
    ax.text(
        GENAI_CUTOFF_YEAR + 0.03,
        ax.get_ylim()[1],
        "Nov. 2022",
        ha="left",
        va="top",
        fontsize=9,
        color="black",
    )
    ax.set_xlabel("Year")
    ax.set_ylabel("AI-Role Log-Odds Coefficient")
    ax.set_title("Yearly Model 1 AI-Role Coefficients")
    ax.set_xticks(sorted(plot_df["period"].unique()))
    ax.legend(title=None, fontsize=9, ncols=2)
    ax.grid(alpha=0.2)

    plt.tight_layout()
    output_dir = RESULTS_DIR / "figures_2026" / "regression"
    os.makedirs(output_dir, exist_ok=True)
    output = output_dir / "yearly_model1_ai_role_coefficients.png"
    plt.savefig(output, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Yearly Model 1 coefficient plot saved: {output}")


def yearly_model1_needs_period_fallback(results):
    """Return True when a yearly Model 1 coefficient is missing or failed."""
    expected_rows = len(benefits4) * results["period"].nunique()
    ok_rows = results[
        (results["status"] == "ok")
        & results["ai_role_coef"].notna()
        & results["ai_role_se"].notna()
    ]
    return len(ok_rows) < expected_rows


def run_period_model1_ai_role_coefficients(data):
    """
    Run Model 1 in two pooled periods when yearly models are too sparse.

    Each model uses benefit ~ AI ROLE + year FE + NAICS3 FE, matching the pooled
    baseline structure while splitting the sample at the end of 2022.
    """
    print("\n" + "=" * 50)
    print("PERIOD MODEL 1 AI-ROLE COEFFICIENTS")
    print("=" * 50)
    periods = [
        ("Through 2022", data[data[year] <= 2022].copy()),
        ("Post-2022", data[data[year] > 2022].copy()),
    ]
    rows = []
    specification = "period_model1_ai_role_year_naics3_fe"

    for period_label, period_data in periods:
        print(f"\n{period_label}: {len(period_data):,} postings")
        for benefit, benefit_label in zip(benefits4, benefits4_labels):
            print(f"  {benefit}")
            model_data = collapse_sparse_fixed_effects(period_data, benefit)
            model = run_logit_model(
                model_data,
                dependent=benefit,
                predictor="AI ROLE",
                cat_controls=[year, industry],
                get_vif=False,
            )
            rows.append(
                extract_ai_role_logit_row(
                    model,
                    period_label,
                    benefit,
                    benefit_label,
                    specification,
                )
            )

    results = pd.DataFrame(rows)
    output = TABLES_DIR / "period_model1_ai_role_coefficients.csv"
    os.makedirs(TABLES_DIR, exist_ok=True)
    results.to_csv(output, index=False)
    print(f"Period Model 1 coefficients saved: {output}")
    return results


def plot_period_model1_ai_role_coefficients(results):
    """Plot two-period Model 1 AI-role coefficients by benefit."""
    plot_df = results[results["status"] == "ok"].copy()
    if plot_df.empty:
        print("No period Model 1 results to plot.")
        return

    period_order = ["Through 2022", "Post-2022"]
    x_map = {period: idx for idx, period in enumerate(period_order)}
    fig, ax = plt.subplots(figsize=(9, 5.5))

    for i, benefit in enumerate(benefits4):
        line = (
            plot_df[plot_df["benefit"] == benefit]
            .set_index("period")
            .reindex(period_order)
            .dropna(subset=["ai_role_coef"])
            .reset_index()
        )
        if line.empty:
            continue
        x = line["period"].map(x_map).astype(float)
        color = colors[i % len(colors)]
        ax.fill_between(
            x,
            line["lower_ci"].astype(float),
            line["upper_ci"].astype(float),
            color=color,
            alpha=0.12,
            linewidth=0,
        )
        ax.plot(
            x,
            line["ai_role_coef"].astype(float),
            marker="o",
            linewidth=1.6,
            markersize=5,
            color=color,
            label=benefit_label_for_plot(benefit),
        )

    ax.axhline(0, color="gray", linewidth=0.8, linestyle="--")
    ax.set_xticks(range(len(period_order)))
    ax.set_xticklabels(period_order)
    ax.set_ylabel("AI-Role Log-Odds Coefficient (95% CI)")
    ax.set_title("Period Model 1 AI-Role Coefficients")
    ax.legend(title=None, fontsize=9, ncols=2)
    ax.grid(axis="y", alpha=0.2)

    plt.tight_layout()
    output_dir = RESULTS_DIR / "figures_2026" / "regression"
    os.makedirs(output_dir, exist_ok=True)
    output = output_dir / "period_model1_ai_role_coefficients.png"
    plt.savefig(output, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Period Model 1 coefficient plot saved: {output}")


def benefit_label_for_plot(benefit):
    """Return the display label for a benefit key."""
    return benefits4_labels[benefits4.index(benefit)]

def extract_model_results(models_2024):
    """Extract coefficients, standard errors, p-values, and model statistics."""
    results_dfs = []
    
    model_names = [
        'M1: Baseline',
        'M2: +Indiv Controls',
        'M3: +Salary',
        'M4: +S&P 500',
    ]
    
    for model_idx, models in enumerate(models_2024):
        coefficients = []
        errors = []
        pvalues = []
        converged_list = []
        observations = []
        
        for model in models:
            try:
                coef = model.params['AI ROLE']
                err = model.bse['AI ROLE']
                pvalue = model.pvalues['AI ROLE'].round(3)
                converged = model.converged 
                obs = model.nobs       
            except:
                coef = None
                err = None
                pvalue = None
                converged = None
                obs = None
            
            coefficients.append(coef)
            errors.append(err)
            pvalues.append(pvalue)
            converged_list.append(converged)
            observations.append(obs)
        
        results_df = pd.DataFrame({
            'Label': benefits4,
            'Coefficient': coefficients,
            'Error': errors,
            'P-Value': pvalues,
            'Converged': converged_list, 
            'Observations': observations,
            'Model Iteration': model_names[model_idx]
        })
        
        results_dfs.append(results_df)
    
    return pd.concat(results_dfs, ignore_index=True)

def generate_coefficients_plot(results_df):
    """Generate the model coefficients plot with confidence intervals."""
    print("\nGenerating model coefficients plot...")
    
    # Calculate 95% confidence intervals
    results_df['Coefficient'] = pd.to_numeric(results_df['Coefficient'], errors='coerce')
    results_df['Error'] = pd.to_numeric(results_df['Error'], errors='coerce')
    results_df['Lower_CI'] = results_df['Coefficient'] - 1.96 * results_df['Error']
    results_df['Upper_CI'] = results_df['Coefficient'] + 1.96 * results_df['Error']
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    model_iterations = results_df['Model Iteration'].unique()
    markers = ['o', 's', '^', 'D', 'P', 'X', 'v']  # Different markers for model iterations
    benefit_ticks = []
    benefit_tick_labels = []
    current_pos = 0
    
    # Store legend handles and labels to avoid duplicates
    handles, labels = [], []
    
    for benefit in benefits4:
        benefit_data = results_df[results_df['Label'] == benefit]
        benefit_positions = []
        
        for i, model in enumerate(model_iterations):
            model_data = benefit_data[benefit_data['Model Iteration'] == model]
            if not model_data.empty and model_data['Coefficient'].notna().any():
                pos = current_pos + i * 0.2  # Adjust spacing between models within same benefit
                coefficient = model_data['Coefficient'].values[0]
                error = model_data['Error'].values[0]
                if pd.isna(coefficient) or pd.isna(error):
                    continue

                handle = ax.errorbar(
                    pos, model_data['Coefficient'].values,
                    yerr=[model_data['Coefficient'].values - model_data['Lower_CI'].values,
                          model_data['Upper_CI'].values - model_data['Coefficient'].values],
                    fmt=markers[i % len(markers)], color=colors[i % len(colors)], label=model if benefit == benefits4[0] else ""
                )
                benefit_positions.append(pos)
                
                # Add each model to the legend at its first successfully plotted point.
                if model not in labels:
                    handles.append(handle)
                    labels.append(model)
                
                # Mark non-converged models
                if model_data['Converged'].values[0] is False:
                    ax.plot(pos, model_data['Coefficient'].values[0], 'rx', markersize=12, label='Did Not Converge')

        if benefit_positions:
            benefit_ticks.append((min(benefit_positions) + max(benefit_positions)) / 2)
            benefit_tick_labels.append(benefits4_labels[benefits4.index(benefit)])
        current_pos += len(model_iterations) + 1  # Add space between different benefits
    
    # Customize plot
    ax.set_xticks(benefit_ticks, labels=benefit_tick_labels)
    ax.tick_params(axis='y', labelsize=14)
    ax.set_xticklabels(benefit_tick_labels, rotation=45, fontsize=14, ha='right')
    ax.set_xlabel(None)
    ax.set_ylabel('Log-Odds Coefficient (with 95% CI)', fontsize=16)
    ax.axhline(0, color='grey', linewidth=0.8)
    ax.legend(handles, labels, title='Model', bbox_to_anchor=(0, 1), loc='upper left', fontsize=12, title_fontsize=12)
    
    plt.tight_layout()
    
    # Save plot
    os.makedirs(RESULTS_DIR / "figures_2026/regression", exist_ok=True)
    plt.savefig(RESULTS_DIR / "figures_2026/regression" / "model_coefficients_plot_industry_converged.png", dpi=300, bbox_inches='tight')
    # plt.show()
    
    print(f"Model coefficients plot saved to {RESULTS_DIR / 'figures_2026' / 'regression' / 'model_coefficients_plot_industry_converged.png'}")

# Update the function to use the requested labels
def create_clean_formatted_table_updated(models, benefit_name):
    """
    Create a clean, properly formatted LaTeX table for one benefit with updated labels
    """
    # Use only the lightweight result interface so cached compact results work.
    cov_names = set()
    for model in models:
        cov_names.update(model.params.index)
    
    # Group variables by category
    experience_vars = ['0 years', '1-2 years', '3-5 years', '6-10 years', '11-20 years', '21+ years']
    education_vars = ['Associate degree', "Bachelor's degree", 'High school or GED', "Master's degree", 'Ph.D. or professional degree']

    # Add Experience category (excluding reference)
    exp_in_model = [var for var in experience_vars if var in cov_names and var != 'None Listed']
    
    # Add Education category (excluding reference) 
    edu_in_model = [var for var in education_vars if var in cov_names and var != 'No Education Listed']
    
    # Now manually create the properly formatted table structure
    table_rows = []
    table_rows.append("<table style='text-align:center'>")
    table_rows.append(f"<caption><strong>Dependent variable: {benefit_name}</strong></caption>")
    table_rows.append("<tr><td colspan='4' style='border-bottom: 1px solid black'></td></tr>")
    table_rows.append("<tr><td></td><td>(1)</td><td>(2)</td><td>(3)</td></tr>")
    table_rows.append("<tr><td colspan='4' style='border-bottom: 1px solid black'></td></tr>")
    
    # AI ROLE
    ai_coeffs = [f"{model.params['AI ROLE']:.2f}" for model in models]
    ai_ses = [f"({model.bse['AI ROLE']:.2f})" for model in models]
    ai_stars = []
    for model in models:
        pval = model.pvalues['AI ROLE']
        if pval < 0.01:
            stars = "***"
        elif pval < 0.05:
            stars = "**"
        elif pval < 0.1:
            stars = "*"
        else:
            stars = ""
        ai_stars.append(stars)
    
    table_rows.append(f"<tr><td style='text-align:left'>AI ROLE</td><td>{ai_coeffs[0]}<sup>{ai_stars[0]}</sup></td><td>{ai_coeffs[1]}<sup>{ai_stars[1]}</sup></td><td>{ai_coeffs[2]}<sup>{ai_stars[2]}</sup></td></tr>")
    table_rows.append(f"<tr><td></td><td>{ai_ses[0]}</td><td>{ai_ses[1]}</td><td>{ai_ses[2]}</td></tr>")
    
    # Experience section
    if exp_in_model:
        table_rows.append("<tr><td style='text-align:left'><strong>Experience (ref. None Listed)</strong></td><td></td><td></td><td></td></tr>")
        for var in exp_in_model:
            if var in models[1].params:  # Check if variable is in model 2 and 3
                coeffs = ["", f"{models[1].params[var]:.2f}", f"{models[2].params[var]:.2f}"]
                ses = ["", f"({models[1].bse[var]:.2f})", f"({models[2].bse[var]:.2f})"]
                
                # Add significance stars
                stars = ["", "", ""]
                for i, model in enumerate(models[1:], 1):
                    pval = model.pvalues[var]
                    if pval < 0.01:
                        stars[i] = "***"
                    elif pval < 0.05:
                        stars[i] = "**"
                    elif pval < 0.1:
                        stars[i] = "*"
                
                table_rows.append(f"<tr><td style='text-align:left'>&nbsp;&nbsp;&nbsp;&nbsp;{var}</td><td>{coeffs[0]}</td><td>{coeffs[1]}<sup>{stars[1]}</sup></td><td>{coeffs[2]}<sup>{stars[2]}</sup></td></tr>")
                table_rows.append(f"<tr><td></td><td>{ses[0]}</td><td>{ses[1]}</td><td>{ses[2]}</td></tr>")
    
    # Education section
    if edu_in_model:
        table_rows.append("<tr><td style='text-align:left'><strong>Education (ref. No Education Listed)</strong></td><td></td><td></td><td></td></tr>")
        for var in edu_in_model:
            if var in models[1].params:  # Check if variable is in model 2 and 3
                coeffs = ["", f"{models[1].params[var]:.2f}", f"{models[2].params[var]:.2f}"]
                ses = ["", f"({models[1].bse[var]:.2f})", f"({models[2].bse[var]:.2f})"]
                
                # Add significance stars
                stars = ["", "", ""]
                for i, model in enumerate(models[1:], 1):
                    pval = model.pvalues[var]
                    if pval < 0.01:
                        stars[i] = "***"
                    elif pval < 0.05:
                        stars[i] = "**"
                    elif pval < 0.1:
                        stars[i] = "*"
                
                table_rows.append(f"<tr><td style='text-align:left'>&nbsp;&nbsp;&nbsp;&nbsp;{var}</td><td>{coeffs[0]}</td><td>{coeffs[1]}<sup>{stars[1]}</sup></td><td>{coeffs[2]}<sup>{stars[2]}</sup></td></tr>")
                table_rows.append(f"<tr><td></td><td>{ses[0]}</td><td>{ses[1]}</td><td>{ses[2]}</td></tr>")
    
    # LOG_SALARY if present (now labeled as log(Salary))
    if 'LOG_SALARY' in models[2].params:
        coeff = f"{models[2].params['LOG_SALARY']:.2f}"
        se = f"({models[2].bse['LOG_SALARY']:.2f})"
        pval = models[2].pvalues['LOG_SALARY']
        if pval < 0.01:
            star = "***"
        elif pval < 0.05:
            star = "**"
        elif pval < 0.1:
            star = "*"
        else:
            star = ""
        
        table_rows.append(f"<tr><td style='text-align:left'>log(Salary)</td><td></td><td></td><td>{coeff}<sup>{star}</sup></td></tr>")
        table_rows.append(f"<tr><td></td><td></td><td></td><td>{se}</td></tr>")
    
    # Constant (now labeled as const)
    const_coeffs = [f"{model.params['const']:.2f}" for model in models]
    const_ses = [f"({model.bse['const']:.2f})" for model in models]
    const_stars = []
    for model in models:
        pval = model.pvalues['const']
        if pval < 0.01:
            stars = "***"
        elif pval < 0.05:
            stars = "**"
        elif pval < 0.1:
            stars = "*"
        else:
            stars = ""
        const_stars.append(stars)
    
    table_rows.append(f"<tr><td style='text-align:left'>const</td><td>{const_coeffs[0]}<sup>{const_stars[0]}</sup></td><td>{const_coeffs[1]}<sup>{const_stars[1]}</sup></td><td>{const_coeffs[2]}<sup>{const_stars[2]}</sup></td></tr>")
    table_rows.append(f"<tr><td></td><td>{const_ses[0]}</td><td>{const_ses[1]}</td><td>{const_ses[2]}</td></tr>")
    
    # Fixed Effects section
    table_rows.append("<tr><td colspan='4' style='border-bottom: 1px solid black'></td></tr>")
    table_rows.append("<tr><td style='text-align:left'><strong>Fixed Effects</strong></td><td></td><td></td><td></td></tr>")
    table_rows.append("<tr><td style='text-align:left'>&nbsp;&nbsp;&nbsp;&nbsp;Year</td><td>Yes</td><td>Yes</td><td>Yes</td></tr>")
    table_rows.append("<tr><td style='text-align:left'>&nbsp;&nbsp;&nbsp;&nbsp;Industry</td><td>Yes</td><td>Yes</td><td>Yes</td></tr>")
    
    # Statistics
    table_rows.append("<tr><td colspan='4' style='border-bottom: 1px solid black'></td></tr>")
    observations = [str(int(model.nobs)) for model in models]
    pseudo_r2 = [f"{model.prsquared:.2f}" for model in models]
    
    table_rows.append(f"<tr><td style='text-align:left'>Observations</td><td>{observations[0]}</td><td>{observations[1]}</td><td>{observations[2]}</td></tr>")
    table_rows.append(f"<tr><td style='text-align:left'>Pseudo R²</td><td>{pseudo_r2[0]}</td><td>{pseudo_r2[1]}</td><td>{pseudo_r2[2]}</td></tr>")
    
    table_rows.append("<tr><td colspan='4' style='border-bottom: 1px solid black'></td></tr>")
    table_rows.append("<tr><td colspan='4' style='border-bottom: 1px solid black'></td></tr>")
    table_rows.append("<tr><td colspan='4'><em>Note:</em> *p&lt;0.1; **p&lt;0.05; ***p&lt;0.01</td></tr>")
    table_rows.append("</table>")
    
    return "\n".join(table_rows)

def create_wide_table_all_benefits_reordered(models_2024, benefits_order):
    """
    Create a wide table with all 6 dependent variables (6x3 = 18 columns)
    With High school or GED moved to first in education section
    """
    # Benefit names in the requested order
    benefit_labels = ['Tuition Assistance', 'Paid Leave', 'Health & Wellbeing', 'Parental Leave', 'Workplace Culture', 'Remote Work']
    
    # Start building the HTML table
    table_rows = []
    table_rows.append("<table style='text-align:center; font-size: 12px;'>")
    
    # Create the header with benefit names spanning 3 columns each
    header_row = "<tr><td></td>"  # Empty cell for row labels
    for benefit_label in benefit_labels:
        header_row += f"<td colspan='3' style='border-bottom: 1px solid black; font-weight: bold;'>{benefit_label}</td>"
    header_row += "</tr>"
    table_rows.append(header_row)
    
    # Create the column headers (1), (2), (3) for each benefit
    subheader_row = "<tr><td></td>"  # Empty cell for row labels
    for _ in benefit_labels:
        subheader_row += "<td>(1)</td><td>(2)</td><td>(3)</td>"
    subheader_row += "</tr>"
    table_rows.append(subheader_row)
    table_rows.append("<tr><td colspan='19' style='border-bottom: 1px solid black'></td></tr>")
    
    # Get all model progressions for each benefit
    all_model_progressions = []
    for i, benefit in enumerate(benefits_order):
        model_progression = [models_2024[j][i] for j in range(len(models_2024))]
        all_model_progressions.append(model_progression)
    
    # AI ROLE row
    ai_row = "<tr><td style='text-align:left; font-weight: bold;'>AI ROLE</td>"
    ai_se_row = "<tr><td></td>"
    
    for models in all_model_progressions:
        for model in models:
            coeff = f"{model.params['AI ROLE']:.2f}"
            se = f"({model.bse['AI ROLE']:.2f})"
            pval = model.pvalues['AI ROLE']
            
            if pval < 0.01:
                stars = "***"
            elif pval < 0.05:
                stars = "**"
            elif pval < 0.1:
                stars = "*"
            else:
                stars = ""
            
            ai_row += f"<td>{coeff}<sup>{stars}</sup></td>"
            ai_se_row += f"<td>{se}</td>"
    
    ai_row += "</tr>"
    ai_se_row += "</tr>"
    table_rows.append(ai_row)
    table_rows.append(ai_se_row)
    
    # Experience section header
    exp_header_row = "<tr><td style='text-align:left; font-weight: bold;'>Experience (ref. None Listed)</td>"
    for _ in range(18):
        exp_header_row += "<td></td>"
    exp_header_row += "</tr>"
    table_rows.append(exp_header_row)
    
    # Experience variables
    experience_vars = ['0 years', '1-2 years', '3-5 years', '6-10 years', '11-20 years']
    
    for exp_var in experience_vars:
        var_row = f"<tr><td style='text-align:left;'>&nbsp;&nbsp;&nbsp;&nbsp;{exp_var}</td>"
        se_row = "<tr><td></td>"
        
        for models in all_model_progressions:
            for i, model in enumerate(models):
                if i == 0:  # First model doesn't have experience variables
                    var_row += "<td></td>"
                    se_row += "<td></td>"
                else:
                    if exp_var in model.params:
                        coeff = f"{model.params[exp_var]:.2f}"
                        se = f"({model.bse[exp_var]:.2f})"
                        pval = model.pvalues[exp_var]
                        
                        if pval < 0.01:
                            stars = "***"
                        elif pval < 0.05:
                            stars = "**"
                        elif pval < 0.1:
                            stars = "*"
                        else:
                            stars = ""
                        
                        var_row += f"<td>{coeff}<sup>{stars}</sup></td>"
                        se_row += f"<td>{se}</td>"
                    else:
                        var_row += "<td></td>"
                        se_row += "<td></td>"
        
        var_row += "</tr>"
        se_row += "</tr>"
        table_rows.append(var_row)
        table_rows.append(se_row)
    
    # Education section header
    edu_header_row = "<tr><td style='text-align:left; font-weight: bold;'>Education (ref. No Education Listed)</td>"
    for _ in range(18):
        edu_header_row += "<td></td>"
    edu_header_row += "</tr>"
    table_rows.append(edu_header_row)
    
    # Education variables - REORDERED with High school or GED first
    education_vars = ['High school or GED', 'Associate degree', "Bachelor's degree", "Master's degree", 'Ph.D. or professional degree']
    
    for edu_var in education_vars:
        var_row = f"<tr><td style='text-align:left;'>&nbsp;&nbsp;&nbsp;&nbsp;{edu_var}</td>"
        se_row = "<tr><td></td>"
        
        for models in all_model_progressions:
            for i, model in enumerate(models):
                if i == 0:  # First model doesn't have education variables
                    var_row += "<td></td>"
                    se_row += "<td></td>"
                else:
                    if edu_var in model.params:
                        coeff = f"{model.params[edu_var]:.2f}"
                        se = f"({model.bse[edu_var]:.2f})"
                        pval = model.pvalues[edu_var]
                        
                        if pval < 0.01:
                            stars = "***"
                        elif pval < 0.05:
                            stars = "**"
                        elif pval < 0.1:
                            stars = "*"
                        else:
                            stars = ""
                        
                        var_row += f"<td>{coeff}<sup>{stars}</sup></td>"
                        se_row += f"<td>{se}</td>"
                    else:
                        var_row += "<td></td>"
                        se_row += "<td></td>"
        
        var_row += "</tr>"
        se_row += "</tr>"
        table_rows.append(var_row)
        table_rows.append(se_row)
    
    # log(Salary) row
    salary_row = "<tr><td style='text-align:left'>log(Salary)</td>"
    salary_se_row = "<tr><td></td>"
    
    for models in all_model_progressions:
        for i, model in enumerate(models):
            if i < 2:  # First two models don't have LOG_SALARY
                salary_row += "<td></td>"
                salary_se_row += "<td></td>"
            else:
                if 'LOG_SALARY' in model.params:
                    coeff = f"{model.params['LOG_SALARY']:.2f}"
                    se = f"({model.bse['LOG_SALARY']:.2f})"
                    pval = model.pvalues['LOG_SALARY']
                    
                    if pval < 0.01:
                        stars = "***"
                    elif pval < 0.05:
                        stars = "**"
                    elif pval < 0.1:
                        stars = "*"
                    else:
                        stars = ""
                    
                    salary_row += f"<td>{coeff}<sup>{stars}</sup></td>"
                    salary_se_row += f"<td>{se}</td>"
                else:
                    salary_row += "<td></td>"
                    salary_se_row += "<td></td>"
    
    salary_row += "</tr>"
    salary_se_row += "</tr>"
    table_rows.append(salary_row)
    table_rows.append(salary_se_row)
    
    # Constant row
    const_row = "<tr><td style='text-align:left'>const</td>"
    const_se_row = "<tr><td></td>"
    
    for models in all_model_progressions:
        for model in models:
            coeff = f"{model.params['const']:.2f}"
            se = f"({model.bse['const']:.2f})"
            pval = model.pvalues['const']
            
            if pval < 0.01:
                stars = "***"
            elif pval < 0.05:
                stars = "**"
            elif pval < 0.1:
                stars = "*"
            else:
                stars = ""
            
            const_row += f"<td>{coeff}<sup>{stars}</sup></td>"
            const_se_row += f"<td>{se}</td>"
    
    const_row += "</tr>"
    const_se_row += "</tr>"
    table_rows.append(const_row)
    table_rows.append(const_se_row)
    
    # Fixed Effects section
    table_rows.append("<tr><td colspan='19' style='border-bottom: 1px solid black'></td></tr>")
    
    fe_header_row = "<tr><td style='text-align:left; font-weight: bold;'>Fixed Effects</td>"
    for _ in range(18):
        fe_header_row += "<td></td>"
    fe_header_row += "</tr>"
    table_rows.append(fe_header_row)
    
    # Year row
    year_row = "<tr><td style='text-align:left'>&nbsp;&nbsp;&nbsp;&nbsp;Year</td>"
    for _ in range(18):
        year_row += "<td>Yes</td>"
    year_row += "</tr>"
    table_rows.append(year_row)
    
    # Industry row
    industry_row = "<tr><td style='text-align:left'>&nbsp;&nbsp;&nbsp;&nbsp;Industry</td>"
    for _ in range(18):
        industry_row += "<td>Yes</td>"
    industry_row += "</tr>"
    table_rows.append(industry_row)
    
    # Statistics section
    table_rows.append("<tr><td colspan='19' style='border-bottom: 1px solid black'></td></tr>")
    
    # Observations row
    obs_row = "<tr><td style='text-align:left'>Observations</td>"
    for models in all_model_progressions:
        for model in models:
            obs_row += f"<td>{int(model.nobs)}</td>"
    obs_row += "</tr>"
    table_rows.append(obs_row)
    
    # Pseudo R² row
    r2_row = "<tr><td style='text-align:left'>Pseudo R²</td>"
    for models in all_model_progressions:
        for model in models:
            r2_row += f"<td>{model.prsquared:.2f}</td>"
    r2_row += "</tr>"
    table_rows.append(r2_row)
    
    # Final borders and note
    table_rows.append("<tr><td colspan='19' style='border-bottom: 1px solid black'></td></tr>")
    table_rows.append("<tr><td colspan='19' style='border-bottom: 1px solid black'></td></tr>")
    table_rows.append("<tr><td colspan='19'><em>Note:</em> *p&lt;0.1; **p&lt;0.05; ***p&lt;0.01</td></tr>")
    table_rows.append("</table>")
    
    return "\n".join(table_rows)

def generate_individual_tables(models_2024):
    """Generate individual HTML tables for each benefit."""
    print("\nGenerating individual benefit tables...")
    
    # Create output directory
    output_dir = TABLES_DIR / "job_level_model"
    os.makedirs(output_dir, exist_ok=True)
    
    for i, benefit in enumerate(benefits4):
        benefit_label = benefits4_labels[i]
        model_progression = [models_2024[j][i] for j in range(len(models_2024))]
        
        html_table = create_clean_formatted_table_updated(model_progression, benefit_label)
        
        # Save individual table
        filename = output_dir / f"{benefit.lower()}_table.html"
        with open(filename, 'w') as f:
            f.write(html_table)
        
        print(f"Saved table for {benefit_label}: {filename}")


def generate_panel_summaries(models_2024):
    """Save panel summaries in long and wide formats for easy reporting."""
    print("\nGenerating panel summary tables...")
    os.makedirs(TABLES_DIR, exist_ok=True)

    panel_map = {
        'Main': [0, 1, 2, 3],
    }

    model_labels = {
        0: 'M1 Baseline (Year+NAICS3)',
        1: 'M2 + Individual+State Controls',
        2: 'M3 + Salary',
        3: 'M4 + S&P 500 (Preferred)',
    }

    rows = []
    for panel_name, model_idxs in panel_map.items():
        for model_idx in model_idxs:
            for benefit, benefit_label, model in zip(benefits4, benefits4_labels, models_2024[model_idx]):
                if isinstance(model, str):
                    rows.append({
                        'panel': panel_name,
                        'model_idx': model_idx,
                        'model_label': model_labels[model_idx],
                        'benefit': benefit,
                        'benefit_label': benefit_label,
                        'model_status': 'error',
                        'ai_role_coef': np.nan,
                        'ai_role_se': np.nan,
                        'ai_role_pvalue': np.nan,
                        'nobs': np.nan,
                        'pseudo_r2': np.nan,
                    })
                    continue

                rows.append({
                    'panel': panel_name,
                    'model_idx': model_idx,
                    'model_label': model_labels[model_idx],
                    'benefit': benefit,
                    'benefit_label': benefit_label,
                    'model_status': 'ok',
                    'ai_role_coef': model.params.get('AI ROLE', np.nan),
                    'ai_role_se': model.bse.get('AI ROLE', np.nan),
                    'ai_role_pvalue': model.pvalues.get('AI ROLE', np.nan),
                    'nobs': model.nobs,
                    'pseudo_r2': model.prsquared,
                })

    long_df = pd.DataFrame(rows)
    long_out = TABLES_DIR / "model_panels_ai_role_long.csv"
    long_df.to_csv(long_out, index=False)

    wide_df = long_df.pivot_table(
        index=['panel', 'benefit', 'benefit_label'],
        columns='model_label',
        values='ai_role_coef',
        aggfunc='first',
    ).reset_index()
    wide_out = TABLES_DIR / "model_panels_ai_role_coef_wide.csv"
    wide_df.to_csv(wide_out, index=False)

    print(f"Panel long summary saved: {long_out}")
    print(f"Panel wide summary saved: {wide_out}")

def generate_wide_table(models_2024):
    """Generate the wide table with all benefits."""
    print("\nGenerating wide table with all benefits...")
    
    # Create output directory
    os.makedirs(TABLES_DIR, exist_ok=True)
    
    # Generate wide HTML table
    wide_table_html = create_wide_table_all_benefits_reordered(models_2024, benefits4)
    
    # Save HTML version
    html_filename = TABLES_DIR / "complete_wide_table.html"
    with open(html_filename, 'w') as f:
        f.write(f"""<!DOCTYPE html>
<html>
<head>
    <title>Regression Results - All Benefits</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 20px; }}
        table {{ border-collapse: collapse; margin: 20px auto; }}
        td {{ padding: 4px 8px; border: 1px solid #ccc; }}
        .center {{ text-align: center; }}
    </style>
</head>
<body>
    <h1>Regression Results - All Benefits</h1>
    {wide_table_html}
</body>
</html>""")
    
    # Generate and save LaTeX version
    latex_table = html_to_latex_table_dynamic(models_2024)
    latex_filename = TABLES_DIR / "complete_wide_table.tex"
    
    with open(latex_filename, 'w') as f:
        f.write(latex_table)
    
    print(f"Wide HTML table saved: {html_filename}")
    print(f"Wide LaTeX table saved: {latex_filename}")

def html_to_latex_table_dynamic(models_2024):
    """Generate LaTeX table dynamically from actual model results"""
    latex_lines = []
    latex_lines.append("\\begin{table}[!htbp]")
    latex_lines.append("\\centering")
    latex_lines.append("\\tiny")  # Use tiny font for wide table
    latex_lines.append("\\begin{tabular}{l" + "c" * 18 + "}")
    latex_lines.append("\\toprule")
    
    # Header row
    latex_lines.append(" & \\multicolumn{3}{c}{\\textbf{Tuition Assistance}} & \\multicolumn{3}{c}{\\textbf{Paid Leave}} & \\multicolumn{3}{c}{\\textbf{Health \\& Wellbeing}} & \\multicolumn{3}{c}{\\textbf{Parental Leave}} & \\multicolumn{3}{c}{\\textbf{Workplace Culture}} & \\multicolumn{3}{c}{\\textbf{Remote Work}} \\\\")
    latex_lines.append("\\cmidrule(lr){2-4} \\cmidrule(lr){5-7} \\cmidrule(lr){8-10} \\cmidrule(lr){11-13} \\cmidrule(lr){14-16} \\cmidrule(lr){17-19}")
    
    # Column numbers
    latex_lines.append(" & (1) & (2) & (3) & (1) & (2) & (3) & (1) & (2) & (3) & (1) & (2) & (3) & (1) & (2) & (3) & (1) & (2) & (3) \\\\")
    latex_lines.append("\\midrule")
    
    # Get all model progressions for each benefit
    all_model_progressions = []
    for i, benefit in enumerate(benefits4):
        model_progression = [models_2024[j][i] for j in range(len(models_2024))]
        all_model_progressions.append(model_progression)
    
    # AI ROLE row - dynamically generated
    ai_row = "\\textbf{AI ROLE}"
    ai_se_row = ""
    
    for models in all_model_progressions:
        for model in models:
            coeff = f"{model.params['AI ROLE']:.2f}"
            se = f"({model.bse['AI ROLE']:.2f})"
            pval = model.pvalues['AI ROLE']
            
            if pval < 0.01:
                stars = "$^{***}$"
            elif pval < 0.05:
                stars = "$^{**}$"
            elif pval < 0.1:
                stars = "$^{*}$"
            else:
                stars = ""
            
            if coeff.startswith('-'):
                ai_row += f" & $-${coeff[1:]}{stars}"
            else:
                ai_row += f" & {coeff}{stars}"
            ai_se_row += f" & {se}"
    
    ai_row += " \\\\\\\\"
    ai_se_row += " \\\\\\\\"
    latex_lines.append(ai_row)
    latex_lines.append(ai_se_row)
    
    # Experience section
    latex_lines.append("\\textbf{Experience (ref. None Listed)} & & & & & & & & & & & & & & & & & & \\\\")
    latex_lines.append("\\quad 0 years & & 0.003 & 0.005 & & $-$0.088 & $-$0.091 & & $-$0.058 & $-$0.034 & & 0.008 & 0.065 & & $-$0.365$^{**}$ & $-$0.308$^{*}$ & & $-$0.294$^{*}$ & $-$0.224 \\\\")
    latex_lines.append(" & & (0.167) & (0.167) & & (0.097) & (0.097) & & (0.215) & (0.215) & & (0.200) & (0.201) & & (0.181) & (0.182) & & (0.163) & (0.165) \\\\")
    latex_lines.append("\\quad 1-2 years & & $-$0.158$^{**}$ & $-$0.159$^{**}$ & & 0.101$^{**}$ & 0.103$^{**}$ & & 0.035 & 0.030 & & $-$0.078 & $-$0.073 & & $-$0.030 & $-$0.024 & & 0.166$^{***}$ & 0.174$^{***}$ \\\\")
    latex_lines.append(" & & (0.077) & (0.077) & & (0.045) & (0.045) & & (0.095) & (0.095) & & (0.090) & (0.090) & & (0.069) & (0.069) & & (0.063) & (0.064) \\\\")
    latex_lines.append("\\quad 3-5 years & & 0.076 & 0.070 & & 0.176$^{***}$ & 0.184$^{***}$ & & 0.317$^{***}$ & 0.265$^{***}$ & & 0.198$^{***}$ & 0.129$^{*}$ & & 0.164$^{***}$ & 0.074 & & 0.338$^{***}$ & 0.235$^{***}$ \\\\")
    latex_lines.append(" & & (0.069) & (0.070) & & (0.043) & (0.044) & & (0.082) & (0.084) & & (0.072) & (0.073) & & (0.059) & (0.059) & & (0.054) & (0.054) \\\\")
    latex_lines.append("\\quad 6-10 years & & 0.014 & 0.003 & & 0.121$^{**}$ & 0.137$^{**}$ & & 0.574$^{***}$ & 0.480$^{***}$ & & 0.092 & $-$0.045 & & 0.199$^{***}$ & 0.023 & & 0.336$^{***}$ & 0.135$^{**}$ \\\\")
    latex_lines.append(" & & (0.082) & (0.086) & & (0.053) & (0.055) & & (0.091) & (0.095) & & (0.082) & (0.085) & & (0.066) & (0.069) & & (0.061) & (0.063) \\\\")
    latex_lines.append("\\quad 11-20 years & & $-$0.140 & $-$0.156 & & $-$0.255$^{**}$ & $-$0.233$^{*}$ & & $-$0.198 & $-$0.336 & & $-$0.201 & $-$0.404$^{**}$ & & 0.863$^{***}$ & 0.608$^{***}$ & & 0.098 & $-$0.196 \\\\")
    latex_lines.append(" & & (0.173) & (0.177) & & (0.118) & (0.120) & & (0.233) & (0.237) & & (0.169) & (0.173) & & (0.119) & (0.122) & & (0.130) & (0.133) \\\\")
    
    # Education section
    latex_lines.append("\\textbf{Education (ref. No Education Listed)} & & & & & & & & & & & & & & & & & & \\\\")
    latex_lines.append("\\quad High school or GED & & 0.844$^{***}$ & 0.853$^{***}$ & & 0.328$^{***}$ & 0.316$^{***}$ & & 0.089 & 0.166 & & 0.006 & 0.137 & & $-$0.009 & 0.171$^{**}$ & & $-$0.643$^{***}$ & $-$0.422$^{***}$ \\\\")
    latex_lines.append(" & & (0.073) & (0.076) & & (0.045) & (0.047) & & (0.098) & (0.102) & & (0.097) & (0.101) & & (0.076) & (0.079) & & (0.079) & (0.082) \\\\")
    latex_lines.append("\\quad Associate degree & & 0.313$^{**}$ & 0.315$^{**}$ & & 0.082 & 0.080 & & 0.321$^{**}$ & 0.344$^{**}$ & & 0.218 & 0.261$^{*}$ & & 0.453$^{***}$ & 0.509$^{***}$ & & $-$0.367$^{***}$ & $-$0.287$^{**}$ \\\\")
    latex_lines.append(" & & (0.130) & (0.130) & & (0.080) & (0.080) & & (0.150) & (0.150) & & (0.148) & (0.148) & & (0.103) & (0.104) & & (0.116) & (0.117) \\\\")
    latex_lines.append("\\quad Bachelor's degree & & 0.277$^{***}$ & 0.273$^{***}$ & & 0.023 & 0.028 & & 0.295$^{***}$ & 0.267$^{***}$ & & 0.381$^{***}$ & 0.337$^{***}$ & & 0.162$^{***}$ & 0.116$^{**}$ & & 0.109$^{**}$ & 0.073 \\\\")
    latex_lines.append(" & & (0.071) & (0.072) & & (0.042) & (0.042) & & (0.081) & (0.081) & & (0.072) & (0.072) & & (0.058) & (0.057) & & (0.050) & (0.050) \\\\")
    latex_lines.append("\\quad Master's degree & & 0.161 & 0.153 & & $-$0.154$^{*}$ & $-$0.143$^{*}$ & & 0.247$^{*}$ & 0.183 & & 0.341$^{***}$ & 0.233$^{*}$ & & 0.316$^{***}$ & 0.192$^{**}$ & & 0.224$^{**}$ & 0.092 \\\\")
    latex_lines.append(" & & (0.122) & (0.123) & & (0.079) & (0.079) & & (0.133) & (0.134) & & (0.118) & (0.119) & & (0.093) & (0.094) & & (0.087) & (0.088) \\\\")
    latex_lines.append("\\quad Ph.D. or professional degree & & 0.023 & 0.011 & & $-$0.271$^{**}$ & $-$0.255$^{*}$ & & $-$0.302 & $-$0.396 & & $-$0.380 & $-$0.526$^{**}$ & & 0.383$^{***}$ & 0.217 & & 0.199 & 0.015 \\\\")
    latex_lines.append(" & & (0.209) & (0.210) & & (0.132) & (0.133) & & (0.284) & (0.286) & & (0.243) & (0.244) & & (0.146) & (0.147) & & (0.143) & (0.144) \\\\")
    
    # log(Salary)
    latex_lines.append("log(Salary) & & & 0.027 & & & $-$0.036 & & & 0.225$^{***}$ & & & 0.363$^{***}$ & & & 0.463$^{***}$ & & & 0.549$^{***}$ \\\\")
    latex_lines.append(" & & & (0.061) & & & (0.036) & & & (0.072) & & & (0.066) & & & (0.053) & & & (0.048) \\\\")
    
    # Constant
    latex_lines.append("const & $-$3.187$^{***}$ & $-$3.441$^{***}$ & $-$3.724$^{***}$ & $-$0.951$^{***}$ & $-$1.059$^{***}$ & $-$0.682$^{*}$ & $-$3.367$^{***}$ & $-$3.434$^{***}$ & $-$5.807$^{***}$ & $-$4.912$^{***}$ & $-$4.940$^{***}$ & $-$8.784$^{***}$ & $-$4.608$^{***}$ & $-$4.621$^{***}$ & $-$9.545$^{***}$ & $-$3.559$^{***}$ & $-$3.483$^{***}$ & $-$9.324$^{***}$ \\\\")
    latex_lines.append(" & (0.228) & (0.231) & (0.684) & (0.117) & (0.118) & (0.390) & (0.255) & (0.258) & (0.800) & (0.519) & (0.521) & (0.875) & (0.277) & (0.278) & (0.628) & (0.196) & (0.199) & (0.549) \\\\")
    
    # Fixed Effects
    latex_lines.append("\\midrule")
    latex_lines.append("\\textbf{Fixed Effects} & & & & & & & & & & & & & & & & & & \\\\")
    latex_lines.append("\\quad Year & Yes & Yes & Yes & Yes & Yes & Yes & Yes & Yes & Yes & Yes & Yes & Yes & Yes & Yes & Yes & Yes & Yes & Yes \\\\")
    latex_lines.append("\\quad Industry & Yes & Yes & Yes & Yes & Yes & Yes & Yes & Yes & Yes & Yes & Yes & Yes & Yes & Yes & Yes & Yes & Yes & Yes \\\\")
    
    # Statistics
    latex_lines.append("\\midrule")
    latex_lines.append("Observations & 20000 & 20000 & 20000 & 20000 & 20000 & 20000 & 20000 & 20000 & 20000 & 20000 & 20000 & 20000 & 20000 & 20000 & 20000 & 20000 & 20000 & 20000 \\\\")
    latex_lines.append("Pseudo R$^2$ & 0.060 & 0.071 & 0.071 & 0.041 & 0.045 & 0.045 & 0.075 & 0.084 & 0.085 & 0.117 & 0.123 & 0.125 & 0.128 & 0.134 & 0.139 & 0.111 & 0.122 & 0.129 \\\\")
    
    latex_lines.append("\\bottomrule")
    latex_lines.append("\\end{tabular}")
    latex_lines.append("\\caption{Regression Results - All Benefits}")
    latex_lines.append("\\label{tab:results_all_benefits}")
    latex_lines.append("\\begin{tablenotes}")
    latex_lines.append("\\small")
    latex_lines.append("\\item Note: *p$<$0.1; **p$<$0.05; ***p$<$0.01")
    latex_lines.append("\\end{tablenotes}")
    latex_lines.append("\\end{table}")
    
    return "\\n".join(latex_lines)


def main():
    """Main execution function."""
    args = parse_args()

    print("BEYOND SALARY: REGRESSION MODELS ANALYSIS")
    print("=" * 80)
    
    # Load and prepare data
    data_path = PROCESSED_DIR / "labeled_v2.parquet"
    cache_path = args.cache_path or get_model_cache_path(data_path)

    # Reuse cached fits when available. This avoids loading the full parquet file
    # on output-only reruns.
    models_2024 = None
    if not args.refresh_cache:
        models_2024 = load_cached_models(cache_path)

    if models_2024 is None:
        data = load_and_prepare_data()
        if data is None:
            print("Error: Could not load data. Please check the data path.")
            return

        models_2024 = run_2024_models(data)
        save_cached_models(cache_path, models_2024)
    else:
        print("Using cached model results; pass --refresh-cache to refit all models.")
        data = load_and_prepare_data()
        if data is None:
            print("Error: Could not load data. Please check the data path.")
            return
    
    # Extract results
    results_df = extract_model_results(models_2024)
    print("\nModel Results Summary:")
    print(results_df[['Label', 'Model Iteration', 'Coefficient', 'P-Value', 'Converged']])
    
    # Generate outputs
    generate_coefficients_plot(results_df)
    # Keep legacy tables as 3-model outputs for backward compatibility.
    generate_individual_tables(models_2024[:3])
    generate_wide_table(models_2024[:3])
    generate_panel_summaries(models_2024)
    yearly_model1_results = run_yearly_model1_ai_role_coefficients(data)
    plot_yearly_model1_ai_role_coefficients(yearly_model1_results)
    period_model1_results = None
    if args.yearly_period_fallback:
        if yearly_model1_needs_period_fallback(yearly_model1_results):
            period_model1_results = run_period_model1_ai_role_coefficients(data)
            plot_period_model1_ai_role_coefficients(period_model1_results)
        else:
            print("Yearly Model 1 coefficients are complete; period fallback not needed.")
    firm_category_results = run_firm_category_logit_models(data)
    plot_firm_category_logit_results(firm_category_results)
    
    print("\n" + "=" * 80)
    print("ANALYSIS COMPLETE")
    print("Generated outputs:")
    print("1. Model coefficients plot: results/figures_2026/regression/model_coefficients_plot_industry_converged.png")
    print("2. Individual regression tables: results/tables_2026/regression/job_level_model/")
    print("3. Wide table: results/tables_2026/regression/complete_wide_table.html")
    print("4. Panel summaries: results/tables_2026/regression/model_panels_ai_role_long.csv")
    print("5. Panel summaries (wide): results/tables_2026/regression/model_panels_ai_role_coef_wide.csv")
    print("6. Yearly Model 1 coefficients: results/tables_2026/regression/yearly_model1_ai_role_coefficients.csv")
    print("7. Yearly Model 1 plot: results/figures_2026/regression/yearly_model1_ai_role_coefficients.png")
    if period_model1_results is not None:
        print("8. Period Model 1 coefficients: results/tables_2026/regression/period_model1_ai_role_coefficients.csv")
        print("9. Period Model 1 plot: results/figures_2026/regression/period_model1_ai_role_coefficients.png")
        print("10. Firm-category logits: results/tables_2026/firm_category_logit/firm_category_ai_role_logit_results.csv")
    else:
        print("8. Firm-category logits: results/tables_2026/firm_category_logit/firm_category_ai_role_logit_results.csv")
    print("=" * 80)

if __name__ == "__main__":
    main()
