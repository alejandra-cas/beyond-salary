# Beyond Salary: AI Jobs and Non-Monetary Benefits

This repository contains the reproducible code and analysis for the paper Beyond pay: AI skills reward more job benefits.


## Repository Structure

```
├── analysis/           # Main analysis scripts
│   ├── descriptive_analysis.py              # Descriptive statistics and data exploration
│   ├── new_data_exploration.ipynb           # Exploratory analysis notebook for MAY26 subsample
│   ├── occupation_year_balanced_sample_analysis.py  # Occupation-year regression models
│   ├── regression_models.py                 # Job-level regression models
│   ├── salary_analysis.py                   # Salary premium analysis
│   ├── scatterplot_analysis.py              # Correlation and scatterplot analysis
│   ├── industry_year_coefficient_analysis.py # Industry-year wage vs perk coefficient comparison
│   ├── high_ai_firm_analysis.py             # Within-firm perk premium by firm AI-share tier
│   ├── ai_threshold_robustness.py           # Robustness check: 1+/2+/3+ AI skills thresholds
│   ├── keyword_vs_structured_benefits.ipynb  # Keyword label validation against structured fields
│   └── wage_info.ipynb                      # Wage distribution analysis notebook
├── scripts/            # Data preparation and processing scripts
│   ├── export_samples.py                    # Export balanced samples for analysis
│   ├── label_benefits.py                    # Benefit labeling and classification
│   ├── label_benefits_remote.py             # Remote work benefit labeling
│   ├── occ_year_analysis.py                 # Occupation-year and industry-year aggregation
│   └── prepare_data.py                      # Data merging, cleaning, and AI role classification
├── src/                # Supporting Python modules
│   └── package_files/  # Core utility functions and model definitions
│       ├── __init__.py                      # Package initialization
│       ├── benefits_defns.py                # Benefit category definitions and mappings
│       └── logit_model.py                   # Logistic regression utilities
├── data/               # Input data files and processed datasets (not included in repo)
│   ├── OII_US_10M_POSTS_MAY26_SUBSAMPLE.csv     # Job postings data
│   ├── OII_US_10M_SKILLS_MAY26_SUBSAMPLE.csv    # Skills data (for AI role classification)
│   ├── OII_US_10M_BODY_MAY26_SUBSAMPLE.csv      # Job posting body text
│   ├── ID_CNTRY_ALL_WHAM.csv                    # Remote work classification data (LLM)
│   └── processed/                               # Pipeline outputs
│       ├── data_v1.parquet                      # Cleaned data with AI roles and experience
│       ├── labeled_v1.parquet                   # Data with keyword + structured benefit labels
│       ├── occ_year_analysis_raw.parquet        # Occupation-year aggregated analysis data
│       └── ind_year_analysis_raw.parquet        # Industry-year aggregated analysis data
├── results/            # Generated outputs
│   ├── figures/                              # Original figures (prior runs)
│   ├── figures_2026/                         # Updated figures (MAY26 dataset)
│   │   ├── descriptive/                     # Descriptive figures
│   │   ├── regression/                      # Regression coefficient plots
│   │   ├── salary/                          # Salary analysis figures
│   │   ├── industry_year/                   # Industry-year wage vs perk coefficient figures
│   │   ├── robustness/                      # Robustness check figures
│   │   ├── perk_positioning/                # Perk prominence-position figures
│   │   ├── high_ai_firms/                   # Within-firm perk premium plots
│   │   └── scatterplots/                    # Correlation analyses
│   └── tables_2026/    # Updated regression tables and robustness outputs
│       ├── job_level_model_2026/            # Individual benefit regression tables
│       ├── occ_year_models/                 # Occupation-year tables
│       ├── within_firm_perk_diff.csv        # Within-firm AI−Non-AI perk gap by firm tier
│       ├── high_ai_firm_summary.csv         # Firm counts and median AI share per tier
│       ├── ai_threshold_robustness.csv      # AI ROLE coefficients across 1+/2+/3+ thresholds
│       ├── perk_positioning_results.csv     # OLS coefficients: AI ROLE on perk prominence score
│       └── structured_benefits_regression.csv  # Structured benefit logit results (P1 M2)
├── config.example.yaml # Template for local data path config (tracked)
├── config.yaml         # Local data path config (gitignored)
└── pyproject.toml      # Project dependencies (uv)
```

## Setup Instructions

1. **Install Dependencies**
   ```bash
   uv sync
   ```

2. **Configure Data Paths**

   Copy the example config and update it for your environment:
   ```bash
   cp config.example.yaml config.yaml
   ```

   Edit `config.yaml` to point to your data directory:
   ```yaml
   data:
     raw_dir: "/path/to/your/data"       # absolute or relative to repo root
     posts_csv: "OII_US_10M_POSTS.csv"   # your posts filename
     skills_csv: "OII_US_10M_SKILLS.csv" # your skills filename
     body_csv: "OII_US_10M_BODY.csv"     # your body text filename
     wham_csv: "ID_CNTRY_ALL_WHAM.csv"
     processed_dir: "data/processed"      # where pipeline outputs go
   ```

   `config.yaml` is gitignored, so each collaborator maintains their own paths without conflicts. If no `config.yaml` is found, the pipeline falls back to default paths under `data/`.

3. **Data Requirements**
   - Posts CSV - Job postings dataset
   - Skills CSV - Skills dataset (used for AI role classification)
   - Body CSV - Job posting body text
   - `ID_CNTRY_ALL_WHAM.csv` - Remote work classification data (LLM)

4. **Run Pipeline**
   ```bash
   # Data preparation
   python scripts/prepare_data.py
   python scripts/label_benefits.py
   python scripts/label_benefits_remote.py
   python scripts/occ_year_analysis.py
   python scripts/export_samples.py

   # Core analyses
   python analysis/descriptive_analysis.py
   python analysis/descriptive_analysis.py --include-structured
   python analysis/regression_models.py
   python analysis/occupation_year_balanced_sample_analysis.py
   python analysis/salary_analysis.py
   python analysis/scatterplot_analysis.py
   ```

## Key Analysis Components

### Data Preparation (`scripts/`)
- **prepare_data.py**: Loads posts, skills, and body CSVs; classifies AI roles using the SKILL_SUBCATEGORY_NAME field; adds experience buckets, log salary, year, and WHAM remote work classification. Outputs `data/processed/data_v1.parquet`.
- **label_benefits.py**: Processes and labels workplace benefits from job posting text. Also computes `{BENEFIT}_POSITION` columns (0–100 prominence score, 100 = top of posting) for each benefit.
- **label_benefits_remote.py**: Labels remote work benefits using keyword matching; merges `REMOTE_KW` and `REMOTE_KW_POSITION` directly into `labeled_v1.parquet`
- **occ_year_analysis.py**: Aggregates data at occupation-year and industry-year levels. Computes AI demand share, salary premiums (mean and median), benefit prevalence by AI/non-AI role, and benefit differences. Uses a `run_analysis()` function that accepts any grouping column (SOC major group, NAICS 2-digit industry, or county). Outputs `occ_year_analysis_raw.parquet` and `ind_year_analysis_raw.parquet`.
- **export_samples.py**: Creates balanced samples for regression analysis

### Core Analysis (`analysis/`)
- **descriptive_analysis.py**: Generates descriptive statistics and exploratory data analysis
  - Default run outputs keyword-benefit figures.
  - Add `--include-structured` to also output combined keyword+structured figures.
- **regression_models.py**: Runs job-level logit models for each benefit with `AI ROLE` as the key predictor using two model panels.
   - **Panel 1 (Industry-based progression)**
      1. **P1 M1 (Baseline):** Year FE + Industry FE
      2. **P1 M2 (+ Individual Controls):** Year FE + Industry FE + Education FE + Experience FE
      3. **P1 M3 (+ Salary):** P1 M2 + Log Salary
      4. **P1 M4 (+ Firm/State FE):** Year FE + Firm FE + State FE + Education FE + Experience FE + Log Salary
   - **Panel 2 (Firm/State baseline progression)**
      1. **P2 M1 (Baseline):** Year FE + Firm FE + State FE
      2. **P2 M2 (+ Individual Controls):** Year FE + Firm FE + State FE + Education FE + Experience FE
      3. **P2 M3 (+ Salary):** intentionally omitted to avoid repeating the full specification already fit in Panel 1
- **occupation_year_balanced_sample_analysis.py**: OLS regressions on benefit differences at occupation-year level
- **salary_analysis.py**: Analyzes salary premiums for AI vs non-AI roles
- **scatterplot_analysis.py**: Creates correlation plots and scatter analyses at occupation-year level
- **industry_year_coefficient_analysis.py**: Estimates AI wage (OLS) and perk (logit) premiums per industry-year cell, then scatterplots wage betas vs perk betas to test complementarity. Runs both controlled (education + experience) and unconditional (AI ROLE only) variants with outlier filtering.
- **high_ai_firm_analysis.py**: Robustness check addressing the concern that large tech firms drive results by offering perks to everyone. Assigns firms to AI-share tiers and computes within-firm perk gaps between AI and non-AI postings. Tuition assistance, paid leave, and parental leave show genuine within-firm AI premiums; remote work and culture appear more firm-wide.
- **ai_threshold_robustness.py**: Robustness check on AI role classification threshold. Re-runs the main P1 M2 spec with 1+, 2+, and 3+ AI skill requirements. Workplace culture strengthens with stricter thresholds; parental leave and remote work are robust at 1+ and 2+ but lose significance at 3+ (power issue — only 226 postings).
- **perk_positioning_analysis.py**: OLS regression of perk prominence score on AI ROLE, conditional on the benefit being mentioned. Health & Wellbeing and Paid Leave appear significantly *lower* in AI postings, suggesting perks are part of compensation packages rather than recruiting bait.
- **keyword_vs_structured_benefits.ipynb**: Validates keyword-based benefit labels against the structured `BENEFIT_NAME` / `BENEFIT_SUBCATEGORY_NAME` / `BENEFIT_CATEGORIES_NAME` fields. Reports precision, recall, and F1 per benefit, with disagreement inspection.
- **structured_benefits_regression.py**: Runs P1 M2 logit models for structured categories (`S_FLEX_WORK`, `S_PROF_DEV`, `S_HEALTH_WELLNESS`, `S_REMOTE`) and outputs a coefficient table + plot. Current takeaway: no notable positive structured-benefit premium for AI postings; coefficients are lower/near-zero for most structured benefits, except remote work which is positive.
- **new_data_exploration.ipynb**: Exploratory analysis notebook for the MAY26 subsample data
- **wage_info.ipynb**: Wage distribution analysis

### Supporting Modules (`src/package_files/`)
- **benefits_defns.py**: Benefit category definitions, labels, color mappings, and standard variable names
- **logit_model.py**: Logistic regression utilities for model fitting

## Generated Outputs

### Regression Tables
- **Individual benefit tables**: `results/tables_2026/job_level_model_2026/{benefit}_table.html`
- **Combined wide table**: `results/tables_2026/complete_wide_table_2026_corrected.html`
- **LaTeX versions**: `results/tables_2026/complete_wide_table_2026.tex`
- **Occupation-year differences**: `results/tables/occ_year_models/difference_regression_table_combined.html`

### Key Figures
- **Model coefficients plot**: `results/figures_2026/regression/model_coefficients_plot_industry_converged.png`
- **Structured-benefit coefficients plot**: `results/figures_2026/regression/structured_benefits_ai_role_coef.png`
- **AI roles over time**: `results/figures_2026/descriptive/pct_ai_roles_overall_industry.png`
- **Keyword benefit differences**: `results/figures_2026/descriptive/benefits_over_time/benefit_diffs_time_keyword.png`
- **Combined keyword + structured differences**: `results/figures_2026/descriptive/benefits_over_time/benefit_diffs_time_all.png`
- **Individual keyword benefit trends**: `results/figures_2026/descriptive/benefits_over_time/benefit_over_time_{benefit}.png` (6 files)
- **Occupation analysis**: `results/figures_2026/descriptive/by_occupation/percent_by_occupation_{benefit}.png` (6 files)
- **Salary analysis**: `results/figures_2026/salary/salary_by_benefit_combined.png`
- **Industry-year wage vs perk figures**: `results/figures_2026/industry_year/`
- **Threshold robustness figure**: `results/figures_2026/robustness/ai_threshold_robustness.png`
- **Perk positioning figure**: `results/figures_2026/perk_positioning/perk_positioning_ai_coef.png`
- **Scatterplots**: `results/figures_2026/scatterplots/` (multiple correlation analyses)


## Citation

  If you use this code or data in your research, please cite:

  Castaneda, Bone & Stephany. "Beyond pay: AI skills reward more job benefits." Working Paper, 2025.

## License
