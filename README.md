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
│       ├── labeled_v2.parquet                   # Data with benefit labels (incl. REMOTE_KW)
│       ├── occ_year_analysis_raw.parquet        # Occupation-year aggregated analysis data
│       └── ind_year_analysis_raw.parquet        # Industry-year aggregated analysis data
├── results/            # Generated outputs
│   ├── figures/                              # Original figures (prior runs)
│   ├── figures_2026/                         # Updated figures (MAY26 dataset)
│   │   ├── benefits_over_time/              # Time trend plots
│   │   ├── pct_jobs_by_benefit_and_role_type/  # Benefit prevalence plots
│   │   └── scatterplots/                    # Correlation analyses
│   └── tables_2026/    # Updated regression tables (HTML and LaTeX formats)
│       ├── job_level_model_2026/            # Individual benefit regression tables
│       └── occ_year_models/                 # Occupation-year tables
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
   python analysis/regression_models.py
   python analysis/occupation_year_balanced_sample_analysis.py
   python analysis/salary_analysis.py
   python analysis/scatterplot_analysis.py
   ```

## Key Analysis Components

### Data Preparation (`scripts/`)
- **prepare_data.py**: Loads posts, skills, and body CSVs; classifies AI roles using the SKILL_SUBCATEGORY_NAME field; adds experience buckets, log salary, year, and WHAM remote work classification. Outputs `data/processed/data_v1.parquet`.
- **label_benefits.py**: Processes and labels workplace benefits from job posting text
- **label_benefits_remote.py**: Labels remote work benefits using keyword matching; merges `REMOTE_KW` directly into `labeled_v2.parquet`
- **occ_year_analysis.py**: Aggregates data at occupation-year and industry-year levels. Computes AI demand share, salary premiums (mean and median), benefit prevalence by AI/non-AI role, and benefit differences. Uses a `run_analysis()` function that accepts any grouping column (SOC major group, NAICS 2-digit industry, or county). Outputs `occ_year_analysis_raw.parquet` and `ind_year_analysis_raw.parquet`.
- **export_samples.py**: Creates balanced samples for regression analysis

### Core Analysis (`analysis/`)
- **descriptive_analysis.py**: Generates descriptive statistics and exploratory data analysis
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
- **keyword_vs_structured_benefits.ipynb**: Validates keyword-based benefit labels against the structured `BENEFIT_NAME` / `BENEFIT_SUBCATEGORY_NAME` / `BENEFIT_CATEGORIES_NAME` fields. Reports precision, recall, and F1 per benefit, with disagreement inspection.
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
- **Model coefficients plot**: `results/figures_2026/model_coefficients_plot_industry_converged.png`
- **AI roles over time**: `results/figures_2026/pct_ai_roles_overall_industry.png`
- **Benefit differences**: `results/figures_2026/benefits_over_time/benefit_diffs_time.png`
- **Individual benefit trends**: `results/figures_2026/benefits_over_time/over_time_color_{benefit}.png` (6 files)
- **Occupation analysis**: `results/figures_2026/percent_by_occupation_colors_2024_{benefit}.png` (6 files)
- **Salary analysis**: `results/figures_2026/salary_by_benefit_combined.png`
- **Scatterplots**: `results/figures_2026/scatterplots/` (multiple correlation analyses)


## Citation

  If you use this code or data in your research, please cite:

  Castaneda, Bone & Stephany. "Beyond pay: AI skills reward more job benefits." Working Paper, 2025.

## License
