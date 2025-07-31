# Beyond Salary: AI Jobs and Benefit Analysis

This repository contains the reproducible code and analysis for the paper Beyond pay: AI skills reward more job benefits.

Alejandra Castaneda
University of Oxford

## Repository Structure

```
├── analysis/           # Main analysis scripts
│   ├── descriptive_analysis.py              # Descriptive statistics and data exploration
│   ├── join_remote_kw.ipynb                 # Jupyter notebook for remote work keyword analysis
│   ├── occupation_year_balanced_sample_analysis.py  # Occupation-year models
│   ├── regression_models_2024.py            # Job-level regression models
│   ├── salary_analysis.py                   # Salary premium analysis
│   └── scatterplot_analysis.py              # Correlation and scatterplot analysis
├── scripts/            # Data preparation and processing scripts
│   ├── export_samples_2024.py               # Export balanced samples for analysis
│   ├── label_benefits.py                    # Benefit labeling and classification
│   ├── label_benefits_remote.py             # Remote work benefit labeling
│   ├── prepare_data.py                      # Initial data merging and cleaning
│   └── prepare_data_2.py                    # Secondary processing and AI classification
├── src/                # Supporting Python modules
│   └── package_files/  # Core utility functions and model definitions
│       ├── __init__.py                      # Package initialization
│       ├── benefits_defns.py                # Benefit category definitions and mappings
│       └── logit_model.py                   # Logistic regression utilities
├── data/               # Input data files and processed datasets (not included in repo)
│   ├── ai_skill_ids.pkl                     # AI skill identifiers (included)
│   ├── balanced_sample_diffs.csv            # Pre-calculated balanced sample differences
│   ├── keywords_list.pkl                    # Benefit keywords for classification
│   ├── occ_year_analysis_2024_raw.csv       # Occupation-year level analysis data
│   ├── us_10m_nointernship_2018_2024_benefits.parquet.gzip  # Main processed dataset
│   ├── small_samples/                       # Balanced samples for regression analysis
│   │   ├── 2024_salary_sample.parquet.gzip
│   │   └── 2024_nosalary_sample.parquet.gzip
│   └── [additional processed data files]
├── results/            # Generated outputs
│   ├── figures/        
│   │   ├── benefits_over_time/              # Time trend plots
│   │   ├── pct_jobs_by_benefit_and_role_type/  # Benefit prevalence plots
│   │   └── scatterplots/                    # Correlation analyses
│   └── tables/         # Regression tables (HTML and LaTeX formats)
│       ├── job_level_model_2025/            # Individual benefit regression tables
│       └── occ_year_models/                  # Occupation-year tables
└── requirements/       # Dependencies and setup
    └── requirements.txt
```

## Setup Instructions

1. **Install Dependencies**
   ```bash
   pip install -r requirements/requirements.txt
   ```

2. **Data Requirements**
   - `US_10M_SAMP_2018_2024.csv` - Original job postings dataset
   - `us_10m_nointernship_2018_2024_benefits.parquet.gzip` - Processed data with benefits
   - `ai_skill_ids.pkl` - AI skill identifiers
   - `ID_CNTRY_ALL_WHAM.csv` - Remote work classification data (LLM)
   - `SALARIES.csv` - Salary data for sample

3. **Run Analysis**
   ```bash
   # Data preparation (run in order)
   python scripts/prepare_data.py
   python scripts/prepare_data_2.py
   python scripts/label_benefits.py
   python scripts/label_benefits_remote.py
   python scripts/export_samples_2024.py
   
   # Core analyses (all working and tested)
   python analysis/descriptive_analysis.py
   python analysis/regression_models_2024.py
   python analysis/occupation_year_balanced_sample_analysis.py
   python analysis/salary_analysis.py
   python analysis/scatterplot_analysis.py
   ```

## Key Analysis Components

### Data Preparation (`scripts/`)
- **prepare_data.py**: Merges job posting data with AI skills labels and remote work classification
- **prepare_data_2.py**: Adds experience buckets, completes AI skills classification, and processes salary data
- **label_benefits.py**: Processes and labels workplace benefits from job postings
- **label_benefits_remote.py**: Specialized remote work benefit labeling and processing
- **join_remote_kw.ipynb**: Join remote keyword labels to data
- **export_samples_2024.py**: Creates balanced samples for regression analysis

### Core Analysis (`analysis/`)
- **descriptive_analysis.py**: Generates descriptive statistics and exploratory data analysis
- **regression_models_2024.py**: Runs logistic regression models for benefit analysis with three specifications:
  1. Baseline (Year + Industry fixed effects)
  2. Individual Controls (+ Education + Experience)
  3. With Salary Control (+ Log Salary)
- **occupation_year_balanced_sample_analysis.py**: Benefit differences analysis at occupation-year level
- **salary_analysis.py**: Analyzes salary premiums for AI vs non-AI roles
- **scatterplot_analysis.py**: Creates correlation plots and scatter analyses at occupation-year level
- **join_remote_kw.ipynb**: Jupyter notebook for remote work keyword processing and evaluation

### Supporting Modules (`src/package_files/`)
- **benefits_defns.py**: Benefit category definitions, labels, and mappings
- **logit_model.py**: Logistic regression utilities for model fitting

## Generated Outputs

### Regression Tables
- **Individual benefit tables**: `results/tables/job_level_model_2025/{benefit}_table.html`
- **Combined wide table**: `results/tables/complete_wide_table_2024_corrected.html`
- **LaTeX versions**: `results/tables/complete_wide_table_2024.tex`
- **Occupation-year differences**: `results/tables/occ_year_models/difference_regression_table_combined.html`

### Key Figures
- **Model coefficients plot**: `results/figures/model_coefficients_plot_industry_converged.png`
- **AI roles over time**: `results/figures/pct_ai_roles_overall_industry.png`
- **Benefit differences**: `results/figures/benefits_over_time/benefit_diffs_time.png`
- **Individual benefit trends**: `results/figures/benefits_over_time/over_time_color_{benefit}.png` (6 files)
- **Occupation analysis**: `results/figures/percent_by_occupation_colors_2024_{benefit}.png` (6 files)
- **Salary analysis**: `results/figures/salary_by_benefit_combined.png`
- **Scatterplots**: `results/figures/scatterplots/` (multiple correlation analyses)


## Citation

  If you use this code or data in your research, please cite:

  Castaneda, Stephany & Bone. "Beyond pay: AI skills reward more job benefits." Working Paper, 2025.

## License

