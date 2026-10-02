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
from datetime import datetime, timezone
from pathlib import Path
from scipy.stats import norm

# Add src directory to path
sys.path.append(str(Path(__file__).parent.parent / 'src'))

from package_files.logit_model import run_logit_model
from package_files.benefits_defns import *
from package_files.config_utils import get_processed_dir, get_repo_root
from package_files.firm_size import FIRM_SIZE_CUTOFF
from package_files.llm_firm_sample import read_firm_labels, select_llm_firms
from package_files.latex_model_summary import render_latex_model_summary
from collections import defaultdict

# Configuration
plt.rcParams.update({'font.size': 14})

REPO_ROOT = get_repo_root()
PROCESSED_DIR = get_processed_dir()
RESULTS_DIR = REPO_ROOT / "results"
TABLES_DIR = RESULTS_DIR / "tables_2026" / "regression"
MODEL_CACHE_DIR = TABLES_DIR / "model_cache"
CACHE_VERSION = "v6_sp500_soc_major_group"
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
benefits4_labels = ['Tuition Assistance', 'Paid Leave', 'Health and Wellbeing', 'Parental Leave', 'Inclusive Workplace', 'Remote Work']

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


def benefit_output_slug(benefit):
    """Filename-safe benefit key for generated outputs."""
    if benefit == "CULTURE":
        return "inclusive_workplace"
    return benefit.lower().replace(" ", "_")
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
    data.loc[(~sp500) & (firm_posting_count < FIRM_SIZE_CUTOFF), "FIRM_CATEGORY"] = "SMEs"
    data.loc[(~sp500) & (firm_posting_count >= FIRM_SIZE_CUTOFF), "FIRM_CATEGORY"] = "Large firms"
    data.loc[sp500 & firm_posting_count.notna(), "FIRM_CATEGORY"] = "S&P 500 firms"
    if "LLM_FIRM_LABEL" in data:
        data["FIRM_CATEGORY"] = data["LLM_FIRM_LABEL"].map({"sme": "SMEs", "large": "Large firms"}).where(firm_posting_count.notna())
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


def add_llm_firm_arguments(parser):
    parser.add_argument('--llm-only', action='store_true',
                        help='Restrict to explicitly LLM-labeled firms and use their SME/large labels.')
    parser.add_argument('--llm-labels', type=Path,
                        default=RESULTS_DIR / 'tables_2026/validation/firm_size_2018_web_validation_20261001/firm_size_labeled_validation_sample.csv',
                        help='CSV containing COMPANY and llm_label; unknown/failed labels are excluded.')
    parser.add_argument('--firm-output-dir', type=Path,
                        help='Root for separate firm-model tables and figures; LLM runs default to results/firm_llm_labeled.')


def prepare_firm_run(args):
    output_root = args.firm_output_dir or (RESULTS_DIR / 'firm_llm_labeled' if args.llm_only else None)
    tables_dir = output_root / 'tables' if output_root else FIRM_CATEGORY_LOGIT_DIR
    figures_dir = output_root / 'figures' if output_root else RESULTS_DIR / 'figures_2026/regression'
    labels = read_firm_labels(args.llm_labels) if args.llm_only else None
    data = load_and_prepare_data(company_ids=labels.index.tolist()) if labels is not None else load_and_prepare_data()
    if data is None:
        raise RuntimeError('Could not load analysis data.')
    if labels is not None:
        data = select_llm_firms(data, labels)
        tables_dir.mkdir(parents=True, exist_ok=True)
        audit = data.groupby(['COMPANY', 'LLM_FIRM_LABEL'], observed=True).size().rename('POSTINGS').reset_index()
        audit['LLM_LABELS_SOURCE'] = str(args.llm_labels.resolve())
        audit.to_csv(tables_dir / 'llm_firm_sample_audit.csv', index=False)
        print(f'LLM-labeled sample: {len(audit):,} firms; {len(data):,} postings; categories use LLM labels.')
    return data, tables_dir, figures_dir


def parse_args():
    """Parse command-line options for cache behavior."""
    parser = argparse.ArgumentParser(description="Run job-level regression models.")
    parser.add_argument(
        "--m5-only", action="store_true",
        help="Run only M5 (M4 + 2-digit SOC FE), saving all outputs in a new directory.",
    )
    parser.add_argument(
        "--plot-only", action="store_true",
        help="Plot M1–M5 from saved summary CSVs without loading data or fitting models.",
    )
    parser.add_argument(
        "--latex-only", action="store_true",
        help="Regenerate only the five-model LaTeX summary from saved caches, preserving its numbered-column format.",
    )
    parser.add_argument(
        "--latex-output", type=Path, default=TABLES_DIR / 'complete_wide_table.tex',
        help="Destination for --latex-only (default: results/tables_2026/regression/complete_wide_table.tex).",
    )
    parser.add_argument(
        "--results-csv", type=Path, default=TABLES_DIR / 'model_panels_ai_role_long.csv',
        help="Saved M1–M4 (or M1–M5) panel summary CSV for --plot-only.",
    )
    parser.add_argument(
        "--m5-results", type=Path,
        help="M5 run directory or summary CSV for --plot-only; defaults to the latest saved M5 run when needed.",
    )
    parser.add_argument(
        "--plot-output", type=Path,
        default=RESULTS_DIR / 'figures_2026' / 'regression' / 'model_coefficients_plot_m1_m5.png',
        help="Destination PNG for --plot-only; defaults to a separate five-model plot.",
    )
    parser.add_argument(
        "--output-dir", type=Path,
        help="New output directory for --m5-only. Must not already exist; defaults to results/m5/<UTC timestamp>.",
    )
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
    parser.add_argument('--firm-only', action='store_true', help='Run only pooled firm-category benefit models and their coefficient plot.')
    add_llm_firm_arguments(parser)
    args = parser.parse_args()
    if sum([args.m5_only, args.plot_only, args.latex_only, args.firm_only]) > 1:
        parser.error("--m5-only, --plot-only, --latex-only, and --firm-only are mutually exclusive")
    if args.plot_only and (args.refresh_cache or args.cache_path is not None or args.yearly_period_fallback):
        parser.error("--plot-only cannot be combined with model-fitting or cache options")
    if args.latex_only and (args.refresh_cache or args.yearly_period_fallback):
        parser.error("--latex-only cannot be combined with model-fitting options")
    if args.m5_results is not None and not (args.plot_only or args.latex_only):
        parser.error("--m5-results requires --plot-only or --latex-only")
    if args.output_dir is not None and not args.m5_only:
        parser.error("--output-dir requires --m5-only")
    if args.m5_only and (args.cache_path is not None or args.yearly_period_fallback):
        parser.error("--m5-only cannot be combined with --cache-path or --yearly-period-fallback")
    if (args.llm_only or args.firm_output_dir is not None) and not args.firm_only:
        parser.error('--llm-only and --firm-output-dir require --firm-only')
    if args.firm_only and (args.refresh_cache or args.cache_path is not None or args.yearly_period_fallback):
        parser.error('--firm-only cannot be combined with main-model cache or fitting options')
    return args


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


def load_cached_models(cache_path, expected_panels=5):
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

    if len(payload["models_2024"]) != expected_panels:
        print("Cached results contain a different model selection; refitting models.")
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


def load_and_prepare_data(company_ids=None):
    """Load and prepare the analysis dataset with all preprocessing."""
    print("Loading analysis dataset...")
    
    # Load the data
    data_path = PROCESSED_DIR / "labeled_v2.parquet"
    if not data_path.exists():
        print(f"Warning: {data_path} not found. Please update the path.")
        return None

    # Using full dataset (not balanced/even sample)
    if company_ids is None:
        data = pd.read_parquet(data_path)
    else:
        import pyarrow as pa
        import pyarrow.parquet as pq
        company_type = pq.read_schema(data_path).field('COMPANY').type
        if pa.types.is_dictionary(company_type):
            company_type = company_type.value_type
        ids = [int(key) for key in company_ids] if pa.types.is_integer(company_type) else (
            [float(key) for key in company_ids] if pa.types.is_floating(company_type) else company_ids
        )
        # Filter employers while reading and omit posting bodies for this run.
        columns = list(dict.fromkeys([firm, year, region, industry, 'NAICS_2022_2_NAME', education, experience,
                                      'LOG_SALARY', 'AI ROLE', 'FIRM_POSTING_COUNT'] + benefits4))
        columns += [name for name in ('SP500', 'FIRM_SIZE_BASE_YEAR') if name in pq.read_schema(data_path).names]
        data = pd.read_parquet(data_path, columns=columns, filters=[('COMPANY', 'in', ids)])

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
    """Run the five H1 perk-prevalence regression specifications."""
    
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
        run_m5_models(data),
    ]


def prepare_occupation_fixed_effects(data):
    """Use the existing SOC major groups, retaining missing occupations."""
    if occupation not in data.columns:
        raise ValueError(f"M5 requires {occupation} for 2-digit SOC fixed effects.")
    data = data.copy()
    data[occupation] = data[occupation].astype('string').str.strip().replace('', pd.NA).fillna('Unknown SOC')
    return data


def run_m5_models(data):
    """Fit M4 plus 2-digit SOC fixed effects for each of the six benefits."""
    data = prepare_occupation_fixed_effects(data)
    print("\nMODEL 5: M4 + 2-DIGIT SOC FIXED EFFECTS")
    models = []
    for benefit in benefits4:
        model_data = collapse_sparse_fixed_effects(data, benefit, require_salary=True)
        models.append(run_logit_model(
            model_data, dependent=benefit, predictor='AI ROLE',
            cat_controls=[year, industry, region, education, experience, occupation],
            cont_controls=['LOG_SALARY'], binary_vars=['SP500'],
            ref_category={education: 'No Education Listed', experience: 'None Listed'},
            get_vif=False,
        ))
    return models


def run_m5_only(args):
    """Save a standalone M5 run without writing to any existing output path."""
    output_dir = args.output_dir or (
        RESULTS_DIR / 'm5' / datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    )
    # Exclusive creation protects previous results, including previous M5 runs.
    output_dir.mkdir(parents=True, exist_ok=False)
    data = load_and_prepare_data()
    if data is None:
        raise FileNotFoundError(f"Analysis dataset not found: {PROCESSED_DIR / 'labeled_v2.parquet'}")
    models = run_m5_models(data)
    save_cached_models(output_dir / 'model_cache' / 'm5.pkl', [models])
    summary = extract_model_results([models], model_names=['M5: M4 + 2-digit SOC FE'])
    summary['SOC 2-digit FE'] = True
    summary['Model Status'] = ['error' if isinstance(model, str) else 'ok' for model in models]
    summary['Pseudo R2'] = [np.nan if isinstance(model, str) else model.prsquared for model in models]
    summary['P-Value'] = [np.nan if isinstance(model, str) else model.pvalues.get('AI ROLE', np.nan) for model in models]
    tables_dir = output_dir / 'tables'
    tables_dir.mkdir()
    summary.to_csv(tables_dir / 'm5_ai_role_summary.csv', index=False)
    summary.to_html(tables_dir / 'm5_ai_role_summary.html', index=False)

    coefficient_tables = []
    for benefit, model in zip(benefits4, models):
        if isinstance(model, str):
            # Keep failures visible in the summary and allow other benefits to finish.
            continue
        ci = model.conf_int()
        coefficients = pd.DataFrame({
            'term': model.params.index, 'coefficient': model.params.to_numpy(),
            'std_error': model.bse.to_numpy(), 'p_value': model.pvalues.to_numpy(),
            'lower_ci': ci.iloc[:, 0].to_numpy(), 'upper_ci': ci.iloc[:, 1].to_numpy(),
        })
        coefficients.insert(0, 'benefit', benefit)
        coefficients['nobs'] = model.nobs
        coefficients['converged'] = model.converged
        coefficients['pseudo_r2'] = model.prsquared
        coefficients['model'] = 'M5: M4 + 2-digit SOC FE'
        coefficients.to_html(tables_dir / f'{benefit_output_slug(benefit)}_m5.html', index=False)
        coefficient_tables.append(coefficients)
    if coefficient_tables:
        pd.concat(coefficient_tables, ignore_index=True).to_csv(
            tables_dir / 'm5_full_coefficients.csv', index=False,
        )
    generate_coefficients_plot(summary, output_path=output_dir / 'figures' / 'm5_ai_role_coefficients.png')
    print(f"M5 outputs saved to {output_dir.resolve()}")


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


def run_firm_category_logit_models(data, output_dir=None):
    """
    Run preferred benefit logit models separately for SMEs, large firms, and S&P 500 firms.

    The coefficient of interest is AI ROLE: the log-odds difference in benefit
    prevalence for AI roles within each firm category.
    """
    print("\n" + "=" * 50)
    print("FIRM-CATEGORY LOGIT MODELS")
    print("=" * 50)

    output_dir = output_dir or FIRM_CATEGORY_LOGIT_DIR
    data = add_firm_category(data)
    firm_categories = ["SMEs", "Large firms", "S&P 500 firms"]
    rows = []

    for firm_category in firm_categories:
        category_data = data[data["FIRM_CATEGORY"] == firm_category].copy()
        print(f"\n{firm_category}: {len(category_data):,} postings")
        if category_data.empty:
            continue
        for benefit, benefit_label in zip(benefits4, benefits4_labels):
            print(f"  {benefit}")
            model = fit_firm_category_logit_model(category_data, firm_category, benefit)
            rows.append(extract_single_model_row(model, firm_category, benefit, benefit_label))

    results = pd.DataFrame(rows)
    os.makedirs(output_dir, exist_ok=True)
    output = output_dir / "firm_category_ai_role_logit_results.csv"
    results.to_csv(output, index=False)
    print(f"Firm-category logit results saved: {output}")
    return results


def plot_firm_category_logit_results(results, output_dir=None, title=None):
    """Plot AI-role log-odds coefficients from firm-category logit models."""
    plot_df = results[results["status"] == "ok"].copy()
    if plot_df.empty:
        print("No firm-category logit results to plot.")
        return

    fig, ax = plt.subplots(figsize=(12, 6))
    firm_categories = [category for category in ["SMEs", "Large firms", "S&P 500 firms"]
                       if category in plot_df["firm_category"].unique()]
    x_base = np.arange(len(benefits4))
    offsets = np.linspace(-0.22, 0.22, len(firm_categories))

    for offset, firm_category in zip(offsets, firm_categories):
        subset = (
            plot_df[plot_df["firm_category"] == firm_category]
            .set_index("benefit")
            .reindex(benefits4)
        )
        if subset["ai_role_coef"].isna().all():
            continue
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
    if title:
        ax.set_title(title)
    plt.tight_layout()

    output_dir = output_dir or RESULTS_DIR / "figures_2026" / "regression"
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


def run_firm_category_yearly_logit_models(data, output_dir=None):
    """Run preferred firm-category benefit logits separately by firm category and year."""
    print("\n" + "=" * 50)
    print("YEARLY FIRM-CATEGORY LOGIT MODELS")
    print("=" * 50)

    output_dir = output_dir or FIRM_CATEGORY_LOGIT_DIR
    data = add_firm_category(data)
    firm_categories = ["SMEs", "Large firms", "S&P 500 firms"]
    rows = []
    specification = "yearly_naics3_state_education_experience_salary"

    for firm_category in firm_categories:
        category_data = data[data["FIRM_CATEGORY"] == firm_category].copy()
        print(f"\n{firm_category}: {len(category_data):,} postings")
        if category_data.empty:
            continue
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
    os.makedirs(output_dir, exist_ok=True)
    output = output_dir / "firm_category_yearly_ai_role_logit_results.csv"
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


def run_firm_category_period_logit_models(data, output_dir=None):
    """Run firm-category benefit logits through 2022 and post-2022."""
    print("\n" + "=" * 50)
    print("PERIOD FIRM-CATEGORY LOGIT MODELS")
    print("=" * 50)

    output_dir = output_dir or FIRM_CATEGORY_LOGIT_DIR
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
        if category_data.empty:
            continue
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
    os.makedirs(output_dir, exist_ok=True)
    output = output_dir / "firm_category_period_ai_role_logit_results.csv"
    results.to_csv(output, index=False)
    print(f"Period firm-category logit results saved: {output}")
    return results


def plot_firm_category_time_logit_results(results, output_name, title, period_order=None, output_dir=None):
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
    output_dir = output_dir or RESULTS_DIR / "figures_2026" / "regression"
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

def extract_model_results(models_2024, model_names=None):
    """Extract coefficients, standard errors, p-values, and model statistics."""
    results_dfs = []
    
    model_names = model_names or [
        'M1: Baseline',
        'M2: +Indiv Controls',
        'M3: +Salary',
        'M4: +S&P 500',
        'M5: M4 + 2-digit SOC FE',
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

def load_five_model_plot_results(results_csv, m5_results=None):
    """Combine saved main-model and standalone M5 coefficients for plotting."""
    panels = pd.read_csv(results_csv)
    panels = panels.loc[panels['panel'].eq('Main') & panels['model_idx'].isin(range(5))].copy()
    panels = panels.rename(columns={
        'benefit': 'Label', 'ai_role_coef': 'Coefficient', 'ai_role_se': 'Error',
        'nobs': 'Observations',
    })
    panels['Converged'] = panels['model_status'].eq('ok')
    if m5_results is not None or not panels['model_idx'].eq(4).any():
        if m5_results is None:
            candidates = sorted((RESULTS_DIR / 'm5').glob('*/tables/m5_ai_role_summary.csv'))
            if not candidates:
                raise FileNotFoundError('No saved M5 summary found. Run --m5-only first or provide --m5-results.')
            m5_results = candidates[-1]
        elif m5_results.is_dir():
            m5_results = m5_results / 'tables' / 'm5_ai_role_summary.csv'
        m5 = pd.read_csv(m5_results)
        m5['model_idx'] = 4
        panels = pd.concat([panels.loc[panels['model_idx'].ne(4)], m5], ignore_index=True)
        print(f"Using M5 summary: {m5_results}")
    expected = pd.MultiIndex.from_product([range(5), benefits4], names=['model_idx', 'Label'])
    actual = pd.MultiIndex.from_frame(panels[['model_idx', 'Label']])
    if actual.has_duplicates or len(expected.difference(actual)):
        raise ValueError('Plot summaries must contain one row per model (M1–M5) and benefit, including failed fits.')
    panels = panels.set_index(['model_idx', 'Label']).reindex(expected).reset_index()
    model_names = ['M1: Baseline', 'M2: +Indiv Controls', 'M3: +Salary', 'M4: +S&P 500', 'M5: M4 + 2-digit SOC FE']
    panels['Model Iteration'] = panels['model_idx'].map(dict(enumerate(model_names)))
    return panels


def generate_coefficients_plot(results_df, output_path=None):
    """Generate the model coefficients plot with confidence intervals."""
    print("\nGenerating model coefficients plot...")
    results_df = results_df.copy()
    results_df['Model Iteration'] = results_df['Model Iteration'].replace({
        'M1: Baseline': 'M1: Baseline (Year+NAICS3)',
        'M2: +Indiv Controls': 'M2: M1 + Individ. + State',
        'M3: +Salary': 'M3: M2 + Salary',
        'M4: +S&P 500': 'M4: M3 + S&P 500',
        'M5: M4 + 2-digit SOC FE': 'M5: M4 + SOC2',
    })
    
    # Calculate 95% confidence intervals
    results_df['Coefficient'] = pd.to_numeric(results_df['Coefficient'], errors='coerce')
    results_df['Error'] = pd.to_numeric(results_df['Error'], errors='coerce')
    results_df['Lower_CI'] = results_df['Coefficient'] - 1.96 * results_df['Error']
    results_df['Upper_CI'] = results_df['Coefficient'] + 1.96 * results_df['Error']
    
    fig, ax = plt.subplots(figsize=(9, 6))
    
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
                    pos, coefficient, yerr=1.96 * error,
                    fmt=markers[i % len(markers)], color=colors[i % len(colors)], label=model if benefit == benefits4[0] else ""
                )
                benefit_positions.append(pos)
                
                # Add each model to the legend at its first successfully plotted point.
                if model not in labels:
                    handles.append(handle)
                    labels.append(model)
                
                # Mark non-converged models
                if pd.notna(model_data['Converged'].values[0]) and not bool(model_data['Converged'].values[0]):
                    ax.plot(pos, model_data['Coefficient'].values[0], 'rx', markersize=12, label='Did Not Converge')

        if benefit_positions:
            benefit_ticks.append((min(benefit_positions) + max(benefit_positions)) / 2)
            benefit_tick_labels.append(benefits4_labels[benefits4.index(benefit)])
        current_pos += len(model_iterations) * 0.2 + 0.6
    
    # Customize plot
    ax.set_xticks(benefit_ticks, labels=benefit_tick_labels)
    ax.tick_params(axis='y', labelsize=14)
    ax.set_xticklabels(benefit_tick_labels, rotation=45, fontsize=14, ha='right')
    ax.set_xlabel(None)
    ax.set_ylabel('Log-Odds Coefficient (with 95% CI)', fontsize=16)
    ax.axhline(0, color='grey', linewidth=0.8)
    ax.legend(handles, labels, title='Model', loc='upper left', fontsize=11, title_fontsize=12)
    
    plt.tight_layout()
    
    # Save plot
    output_path = output_path or RESULTS_DIR / "figures_2026/regression" / "model_coefficients_plot_industry_converged.png"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    # plt.show()
    
    print(f"Model coefficients plot saved to {output_path}")

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
    benefit_labels = ['Tuition Assistance', 'Paid Leave', 'Health & Wellbeing', 'Parental Leave', 'Inclusive Workplace', 'Remote Work']
    
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
        filename = output_dir / f"{benefit_output_slug(benefit)}_table.html"
        with open(filename, 'w') as f:
            f.write(html_table)
        
        print(f"Saved table for {benefit_label}: {filename}")


def generate_panel_summaries(models_2024):
    """Save panel summaries in long and wide formats for easy reporting."""
    print("\nGenerating panel summary tables...")
    os.makedirs(TABLES_DIR, exist_ok=True)

    panel_map = {
        'Main': [0, 1, 2, 3, 4],
    }

    model_labels = {
        0: 'M1 Baseline (Year+NAICS3)',
        1: 'M2 + Individual+State Controls',
        2: 'M3 + Salary',
        3: 'M4 + S&P 500 (Preferred)',
        4: 'M5 M4 + 2-digit SOC FE',
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

def generate_wide_table(models_2024, latex_models=None):
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
    latex_table = html_to_latex_table_dynamic(latex_models if latex_models is not None else models_2024)
    latex_filename = TABLES_DIR / "complete_wide_table.tex"
    
    with open(latex_filename, 'w') as f:
        f.write(latex_table)
    
    print(f"Wide HTML table saved: {html_filename}")
    print(f"Wide LaTeX table saved: {latex_filename}")

def html_to_latex_table_dynamic(models_2024):
    """Render all LaTeX coefficients and statistics from fitted results."""
    return render_latex_model_summary(models_2024, benefits4_labels)


def load_latex_summary_models(args):
    """Read existing full-coefficient caches without refitting or altering them."""
    cache_path = args.cache_path
    if cache_path is None:
        candidates = list(MODEL_CACHE_DIR.glob('job_level_models_*.pkl'))
        if not candidates:
            raise FileNotFoundError('No main-model cache found. Supply --cache-path with saved fits.')
        cache_path = max(candidates, key=lambda path: path.stat().st_mtime_ns)
    with cache_path.open('rb') as stream:
        payload = pickle.load(stream)
    if payload.get('cache_format') != CACHE_FORMAT or payload.get('cache_version') not in {
        CACHE_VERSION, 'v4_preferred_sp500_no_firm_fe',
    }:
        raise ValueError('Unsupported model cache; LaTeX regeneration will not refit models.')
    models = deserialize_compact_model_results(payload['models_2024'])
    if len(models) not in (4, 5) or any(len(panel) != len(benefits4) for panel in models):
        raise ValueError('Main-model cache must contain four or five models for each benefit.')
    # Validate against the summary so the full coefficient tables match the plotted fits.
    summaries = pd.read_csv(args.results_csv)
    summaries = summaries[summaries['panel'].eq('Main')]
    for index, panel in enumerate(models[:4]):
        for benefit, model in zip(benefits4, panel):
            rows = summaries[summaries['model_idx'].eq(index) & summaries['benefit'].eq(benefit)]
            if len(rows) != 1:
                raise ValueError(f'Missing or duplicate saved summary for M{index + 1}, {benefit}.')
            row = rows.iloc[0]
            matches = row['model_status'] != 'ok' if isinstance(model, str) else (
                np.isclose(model.params.get('AI ROLE', np.nan), row['ai_role_coef'], equal_nan=True)
                and np.isclose(model.nobs, row['nobs'])
            )
            if not matches:
                raise ValueError(f'Cache and saved summary disagree for M{index + 1}, {benefit}. Select matching --cache-path and --results-csv.')
    print(f'Using main-model cache: {cache_path}')
    if len(models) == 4 or args.m5_results is not None:
        m5_run = args.m5_results
        if m5_run is None:
            candidates = sorted((RESULTS_DIR / 'm5').glob('*/model_cache/m5.pkl'))
            if not candidates:
                raise FileNotFoundError('No M5 cache found. Run --m5-only first or supply --m5-results.')
            m5_cache = candidates[-1]
        elif m5_run.is_dir():
            m5_cache = m5_run / 'model_cache' / 'm5.pkl'
        elif m5_run.suffix == '.csv':
            m5_cache = m5_run.parent.parent / 'model_cache' / 'm5.pkl'
        else:
            m5_cache = m5_run
        m5 = load_cached_models(m5_cache, expected_panels=1)
        if m5 is None or len(m5[0]) != len(benefits4):
            raise ValueError('A current M5 cache containing all six benefits is required.')
        models = models[:4] + m5
    for benefit, m4_model, m5_model in zip(benefits4, models[3], models[4]):
        if not isinstance(m4_model, str) and not isinstance(m5_model, str) and m4_model.nobs != m5_model.nobs:
            raise ValueError(f'M4/M5 sample sizes differ for {benefit}; select matching caches.')
    return models


def main():
    """Main execution function."""
    args = parse_args()
    if args.firm_only:
        data, tables_dir, figures_dir = prepare_firm_run(args)
        results = run_firm_category_logit_models(data, output_dir=tables_dir)
        plot_firm_category_logit_results(results, output_dir=figures_dir,
                                        title='LLM-labeled firms' if args.llm_only else None)
        return
    if args.m5_only:
        run_m5_only(args)
        return
    if args.plot_only:
        results = load_five_model_plot_results(args.results_csv, args.m5_results)
        generate_coefficients_plot(results, output_path=args.plot_output)
        return
    if args.latex_only:
        models = load_latex_summary_models(args)
        args.latex_output.parent.mkdir(parents=True, exist_ok=True)
        args.latex_output.write_text(html_to_latex_table_dynamic(models))
        print(f'LaTeX summary saved: {args.latex_output}')
        return

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
    generate_wide_table(models_2024[:3], latex_models=models_2024)
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
