# Beyond Salary: AI Jobs and Non-Monetary Benefits

This repository contains the reproducible code and analysis for the paper Beyond pay: AI skills reward more job benefits.


## Repository Structure

```
├── analysis/           # Main analysis scripts
│   ├── ai_threshold_robustness.py             # Robustness check: 1+/2+/3+ AI skills thresholds
│   ├── descriptive_analysis.py                # Descriptive statistics and data exploration
│   ├── firm_category_yearly_regression.py     # Firm-category logit models by year and period
│   ├── high_ai_firm_analysis.py               # Within-firm perk premium by firm AI-share tier
│   ├── industry_year_coefficient_analysis.py  # Industry-year wage vs perk coefficient comparison
│   ├── occupation_year_prevalence_analysis.py # Occupation-year perk prevalence models
│   ├── perk_positioning_analysis.py           # Perk prominence-position analysis
│   ├── regression_models.py                   # Job-level regression models
│   ├── salary_analysis.py                     # Salary premium analysis
│   ├── scatterplot_analysis.py                # Correlation and scatterplot analysis
│   ├── structured_benefits_regression.py      # Structured benefit regression models
│   ├── wage_perk_interaction_analysis.py      # H2 wage-perk interaction models
│   ├── join_remote_kw.ipynb                   # Remote keyword joining exploration
│   ├── keyword_vs_structured_benefits.ipynb   # Keyword label validation against structured fields
│   ├── new_data_exploration.ipynb             # Exploratory analysis notebook for MAY26 subsample
│   ├── regression_failure_investigation.ipynb # Regression diagnostics notebook
│   └── wage_info.ipynb                        # Wage distribution analysis notebook
├── scripts/            # Data preparation and processing scripts
│   ├── build_sp500_snapshot.py              # Build fixed S&P 500 company-ID snapshot
│   ├── export_samples.py                    # Export balanced samples for analysis
│   ├── label_benefits.py                    # Benefit labeling and classification
│   ├── label_benefits_remote.py             # Remote work benefit labeling
│   ├── occ_year_analysis.py                 # Occupation-year and industry-year aggregation
│   ├── prepare_data.py                      # Data merging, cleaning, and AI role classification
│   ├── relabel_sp500.py                     # Refresh SP500 labels in existing processed files
│   └── validate_firm_size_cutoffs.py        # LLM-audited firm-size cutoff validation
├── src/                # Supporting Python modules
│   └── package_files/  # Core utility functions and model definitions
│       ├── __init__.py                      # Package initialization
│       ├── benefits_defns.py                # Benefit category definitions and mappings
│       ├── config_utils.py                  # Shared config/path helpers
│       └── logit_model.py                   # Logistic regression utilities
├── data/               # Input data files and processed datasets (not included in repo)
│   ├── OII_US_10M_POSTS_MAY26_SUBSAMPLE.csv     # Job postings data
│   ├── OII_US_10M_SKILLS_MAY26_SUBSAMPLE.csv    # Skills data (for AI role classification)
│   ├── OII_US_10M_BODY_MAY26_SUBSAMPLE.csv      # Job posting body text
│   ├── ID_CNTRY_ALL_WHAM.csv                    # Remote work classification data (LLM)
│   ├── sp500_snapshot.csv                       # Matched S&P 500 company-ID snapshot
│   ├── sp500_snapshot_audit.csv                 # Snapshot matching audit
│   └── processed/                               # Pipeline outputs
│       ├── data_v1.parquet                      # Cleaned data with AI roles and experience
│       ├── labeled_v1.parquet                   # Data with non-remote benefit labels
│       ├── labeled_v2.parquet                   # Data with benefit labels (incl. REMOTE_KW)
│       ├── occ_year_analysis_raw.parquet        # Occupation-year aggregated analysis data
│       ├── ind_year_analysis_raw.parquet        # Industry-year aggregated analysis data
│       ├── sp500_match_audit.csv                # Processed-data S&P 500 match audit
│       └── sp500_relabel_audit.csv              # Standalone S&P 500 relabel audit
├── results/            # Generated outputs
│   ├── figures/                              # Original figures (prior runs)
│   ├── figures_2026/                         # Updated figures (MAY26 dataset)
│   │   ├── benefits_over_time/              # Benefit prevalence over time
│   │   ├── descriptive/                     # Descriptive figures
│   │   ├── high_ai_firms/                   # Within-firm perk premium plots
│   │   ├── pct_jobs_by_benefit_and_role_type/ # Benefit prevalence by AI role
│   │   ├── salary/                          # Salary analysis figures
│   │   └── scatterplots/                    # Correlation analyses
│   ├── tables/                              # Original tables (prior runs)
│   └── tables_2026/    # Updated regression tables and robustness outputs
│       ├── job_level_model/                 # Individual benefit regression tables
│       ├── job_level_model_2026/            # Individual benefit regression tables
│       ├── robustness/                      # Robustness check outputs
│       ├── within_firm_perk_diff.csv        # Within-firm AI−Non-AI perk gap by firm tier
│       ├── high_ai_firm_summary.csv         # Firm counts and median AI share per tier
│       ├── ai_threshold_robustness.csv      # AI ROLE coefficients across 1+/2+/3+ thresholds
│       ├── perk_positioning_results.csv     # OLS coefficients: AI ROLE on perk prominence score
│       ├── model_panels_ai_role_coef_wide.csv # Job-level model summary, wide format
│       └── model_panels_ai_role_long.csv    # Job-level model summary, long format
├── config.example.yaml # Template for local data path config (tracked)
├── config.yaml         # Local data path config (gitignored)
├── pyproject.toml      # Project dependencies (uv)
└── uv.lock             # Locked uv dependency versions
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
     sp500_csv: "sp500_snapshot.csv"     # optional fixed snapshot keyed by COMPANY
     processed_dir: "data/processed"      # where pipeline outputs go
   ```

   `config.yaml` is gitignored, so each collaborator maintains their own paths without conflicts. If no `config.yaml` is found, the pipeline falls back to default paths under `data/`.

3. **Data Requirements**
   - Posts CSV - Job postings dataset
   - Skills CSV - Skills dataset (used for AI role classification)
   - Body CSV - Job posting body text
   - `ID_CNTRY_ALL_WHAM.csv` - Remote work classification data (LLM)
   - Optional S&P 500 snapshot CSV - fixed constituent snapshot with a `COMPANY` column matching the posting data. Add `COMPANY_NAME` and `SNAPSHOT_DATE` for auditing when available.

   To convert a scraped constituent table into the required posting-system `COMPANY` snapshot, run:
   ```bash
   uv run python scripts/build_sp500_snapshot.py \
     --input /path/to/sp500_constituents.txt \
     --posts-csv data/OII_US_10M_POSTS_MAY26_SUBSAMPLE.csv \
     --output data/sp500_snapshot.csv \
     --audit-output data/sp500_snapshot_audit.csv \
     --snapshot-date YYYY-MM-DD
   ```
   Review `data/sp500_snapshot_audit.csv` before running the pipeline. The builder matches conservative normalized company names only and leaves uncertain names unmatched or ambiguous.

4. **Run Pipeline**
   ```bash
   # Data preparation
   uv run python scripts/prepare_data.py
   uv run python scripts/label_benefits.py
   uv run python scripts/label_benefits_remote.py
   # Optional: refresh SP500 labels only after updating data/sp500_snapshot.csv
   uv run python scripts/relabel_sp500.py
   uv run python scripts/occ_year_analysis.py
   uv run python scripts/export_samples.py

   # Optional: validate posting-count SME/large-firm cutoffs
   uv run python scripts/validate_firm_size_cutoffs.py --cutoffs 5:100:5
   # Add OPENAI_API_KEY=... to .env before running the LLM step
   uv run python scripts/validate_firm_size_cutoffs.py --cutoffs 5:100:5 --llm

   # Core analyses
   uv run python analysis/descriptive_analysis.py
   uv run python analysis/descriptive_analysis.py --include-structured
   uv run python analysis/regression_models.py
   uv run python analysis/firm_category_yearly_regression.py
   uv run python analysis/wage_perk_interaction_analysis.py
   uv run python analysis/occupation_year_prevalence_analysis.py
   uv run python analysis/salary_analysis.py
   uv run python analysis/scatterplot_analysis.py
   ```

## Key Analysis Components

Figure 1 extends the AI-skills wage-premium analysis in Bone, Ehlinger, and Stephany (2024), *Skills or Degree? The Rise of Skill-Based Hiring for AI and Green Jobs*. The cited UK analysis explains log asking wages using AI-skill indicators, education, experience, and fixed effects. This US extension plots quarterly `AI ROLE` coefficients directly in log points and intentionally omits occupation fixed effects.

### Data Preparation (`scripts/`)
- **prepare_data.py**: Loads posts, skills, and body CSVs; classifies AI roles using the SKILL_SUBCATEGORY_NAME field; adds experience buckets, log salary, year, NAICS 3-digit industry, posting-volume firm-size buckets, optional S&P 500 snapshot membership, and WHAM remote work classification. Outputs `data/processed/data_v1.parquet`.
- **label_benefits.py**: Processes and labels workplace benefits from job posting text. Also computes `{BENEFIT}_POSITION` columns (0–100 prominence score, 100 = top of posting) for each benefit.
- **label_benefits_remote.py**: Labels remote work benefits using keyword matching; merges `REMOTE_KW` and `REMOTE_KW_POSITION` directly into `labeled_v1.parquet`
- **occ_year_analysis.py**: Aggregates data at occupation-year and industry-year levels. Computes AI demand share, salary premiums (mean and median), benefit prevalence by AI/non-AI role, and benefit differences. Uses a `run_analysis()` function that accepts any grouping column (SOC major group, NAICS 2-digit industry, or county). Outputs `occ_year_analysis_raw.parquet` and `ind_year_analysis_raw.parquet`.
- **export_samples.py**: Creates balanced samples for regression analysis
- **validate_firm_size_cutoffs.py**: Iterates posting-count cutoffs (default `5, 10, ..., 100`), excludes S&P 500 firms when `SP500` is available, draws stratified random firm samples at each threshold, optionally labels sampled firms with the OpenAI Responses API as true SMEs or large firms, and writes false-positive, false-negative, SME sensitivity, large-firm sensitivity, precision, accuracy, and recall-plot outputs to `results/tables_2026/firm_size_cutoff_validation/`.

### Core Analysis (`analysis/`)
- **descriptive_analysis.py**: Generates descriptive statistics and exploratory data analysis. Figure 1 combines quarterly AI demand with quarterly adjusted `AI ROLE` log-wage coefficients and 95% confidence intervals. Quarterly wage models control for education, experience, NAICS 3-digit industry, and state fixed effects; quarters with fewer than 10 AI wage postings are suppressed. Also generates firm-category analyses: sample composition, AI demand over time, pooled and annual/period-group AI wage premiums, benefit gap (AI minus non-AI perk prevalence), and benefit prevalence by role type — all broken out by SMEs, Large firms, and S&P 500 firms.
  - Default run outputs keyword-benefit figures and all firm-category figures.
  - Add `--include-structured` to also output combined keyword+structured figures.
- **regression_models.py**: Runs H1 job-level logit models for each benefit with `AI ROLE` as the key predictor.
   1. **M1 (Baseline):** Year FE + NAICS 3-digit FE
   2. **M2 (+ Individual/State Controls):** M1 + State FE + Education FE + Experience FE
   3. **M3 (+ Salary):** M2 + Log Salary
   4. **M4 (+ S&P 500):** M3 + S&P 500 indicator. This is the preferred specification.
- **firm_category_yearly_regression.py**: Runs the preferred firm-category H1 logit specification separately by firm category and calendar year, then also runs the same firm-category split for `Through 2022` (`YEAR <= 2022`) and `Post-2022` (`YEAR >= 2023`) period groups. Yearly models omit year fixed effects; period-group models include year fixed effects. The default run produces both yearly and period-group outputs. Use `--yearly-only` or `--period-only` for narrower reruns. Sparse or singular cells are recorded as errors rather than changing controls or reference categories.
- **wage_perk_interaction_analysis.py**: Runs H2 log-wage models for every perk: `LOG_SALARY ~ AI_ROLE + PERK + AI_ROLE:PERK + Education FE + Experience FE + NAICS 3-digit FE + State FE + Year FE`. Models are estimated for the full wage sample, SMEs, Large firms, and S&P 500 firm subsamples. Calendar-year models omit year FE. The `AI_ROLE:PERK` coefficient captures complementarity when positive and substitution when negative. Firm-split charts show only SMEs, Large firms, and S&P 500 firms. Outputs include pooled and yearly AI-role wage premium (β₁) by firm category, β₃ heatmaps across all perks and firm types for pooled/through-2022/post-2022 models, and per-perk time-series plots with 95% CI ribbons.
- **occupation_year_prevalence_analysis.py**: OLS regressions on AI benefit prevalence at occupation-year level
- **salary_analysis.py**: Analyzes salary premiums for AI vs non-AI roles
- **scatterplot_analysis.py**: Creates correlation plots and scatter analyses at occupation-year level
- **industry_year_coefficient_analysis.py**: Supplementary prior analysis estimating AI wage and perk coefficients per industry-year cell. It is no longer the primary H2 strategy.
- **high_ai_firm_analysis.py**: Robustness check addressing the concern that large tech firms drive results by offering perks to everyone. Assigns firms to AI-share tiers and computes within-firm perk gaps between AI and non-AI postings. Tuition assistance, paid leave, and parental leave show genuine within-firm AI premiums; remote work and culture appear more firm-wide.
- **ai_threshold_robustness.py**: Robustness check on AI role classification threshold. Re-runs the main P1 M2 spec with 1+, 2+, and 3+ AI skill requirements. Workplace culture strengthens with stricter thresholds; parental leave and remote work are robust at 1+ and 2+ but lose significance at 3+ (power issue — only 226 postings).
- **perk_positioning_analysis.py**: Existing supplementary analysis of perk prominence. This analysis is currently on hold.
- **keyword_vs_structured_benefits.ipynb**: Validates keyword-based benefit labels against the structured `BENEFIT_NAME` / `BENEFIT_SUBCATEGORY_NAME` / `BENEFIT_CATEGORIES_NAME` fields. Reports precision, recall, and F1 per benefit, with disagreement inspection.
- **structured_benefits_regression.py**: Runs P1 M2 logit models for structured categories (`S_FLEX_WORK`, `S_PROF_DEV`, `S_HEALTH_WELLNESS`, `S_REMOTE`) and outputs a coefficient table + plot. Current takeaway: no notable positive structured-benefit premium for AI postings; coefficients are lower/near-zero for most structured benefits, except remote work which is positive.
- **new_data_exploration.ipynb**: Exploratory analysis notebook for the MAY26 subsample data
- **wage_info.ipynb**: Wage distribution analysis

### Supporting Modules (`src/package_files/`)
- **benefits_defns.py**: Benefit category definitions, labels, color mappings, and standard variable names
- **logit_model.py**: Logistic regression utilities for model fitting

## Methodology Decisions

### Sample and missing-value treatment
- The expanded US sample contains 99,860 postings from January 2018 through December 2025.
- AI roles are postings requiring at least one AI/ML skill.
- Missing experience requirements are retained as `None Listed`.
- Postings without listed education requirements are retained as `No Education Listed`, the education reference category.
- H1 models without salary controls retain the full available sample. H1 salary-controlled models drop postings without `LOG_SALARY`, leaving 35,710 postings for most perks.
- Existing pipeline behavior excludes 2018 from parental-leave models, leaving 34,090 observations in the salary-controlled H1 parental-leave specification.
- H2 starts from postings with wage information and drops rows missing state identifiers or any required model field. Education and experience are already retained in their explicit no-requirement categories. The resulting H2 wage sample contains 35,245 postings; the difference from the 35,710 H1 wage sample is primarily 465 postings without state identifiers.

### Sparse fixed-effect categories
- H1 uses NAICS 3-digit industry fixed effects and state fixed effects. All 99,860 postings have a derived NAICS 3-digit code, covering 97 categories.
- To avoid quasi-separation, singular matrices, and unstable coefficients in the perk logit models, sparse categories are grouped into `Other NAICS 3-digit` or `Other State`.
- Grouping is calculated separately for each perk and model sample. Salary-controlled models calculate sparse categories using only postings with wage information.
- A category is grouped when it has fewer than 50 relevant postings, fewer than 5 postings with the perk, or fewer than 5 postings without the perk.
- Alternative thresholds such as 30, 50, and 100 observations are suitable robustness checks.
- H1 is fitted with a binomial GLM solver. This preserves the logit specification while converging reliably with the fixed-effect design.

### Firm-size categories
- Firm size is a posting-volume proxy. `FIRM_POSTING_COUNT` is calculated from each firm's total observations in the complete labeled dataset.
- `COMPANY == 0` represents `Unclassified` employers. These 12,855 postings are retained in full-sample analyses but are excluded from firm-category subsamples so they are not treated as one large employer.
- Three mutually exclusive firm categories are used:
  - **SMEs**: firms with fewer than 10 postings
  - **Large firms**: firms with 10 or more postings (excluding S&P 500)
  - **S&P 500 firms**: firms matched to the S&P 500 snapshot (override Large)

| Firm category | Postings | AI vacancies | Firms | AI share |
| --- | ---: | ---: | ---: | ---: |
| SMEs | 49,776 | 619 | 33,700 | 1.2% |
| Large firms | 25,569 | 262 | 971 | 1.0% |
| S&P 500 firms | 11,660 | 359 | 409 | 3.1% |

### S&P 500 snapshot
- S&P 500 membership uses one fixed constituent snapshot.
- The snapshot builder uses conservative normalized-name matching and writes an audit so unmatched or ambiguous companies remain visible.
- The current snapshot matches 409 constituent companies and identifies 11,660 postings.
- Reviewed aliases, including `Alphabet -> Google` and `Capital One Financial -> Capital One`, are encoded in `scripts/build_sp500_snapshot.py`.

### Figure 1 and H2 thresholds
- Figure 1 estimates separate quarterly AI wage models and plots the `AI ROLE` coefficient directly in log points with 95% confidence intervals. It suppresses quarters with fewer than 10 AI postings containing wage information, leaving 14 plotted quarters.
- H2 pooled models cover 2018–2025 and include year fixed effects. Calendar-year H2 models omit year fixed effects.
- H2 period heatmaps use `Through 2022` (`YEAR <= 2022`) and `Post-2022` (`YEAR >= 2023`) model splits.
- H2 skips and records a model when its subsample has fewer than 10 AI wage postings or any `AI ROLE x PERK` cell contains fewer than 3 postings.
- Firm category (SMEs, Large firms, S&P 500 firms) defines H2 subsamples only; they are not included as H2 controls.
- A negative `AI ROLE x PERK` coefficient means the AI wage premium is smaller when the perk is present. This is consistent with substitution between monetary and non-monetary compensation, but it is not a causal estimate and does not imply that AI jobs with the perk pay less in absolute terms.

## Generated Outputs

### Regression Tables
- **Individual benefit tables**: `results/tables_2026/job_level_model_2026/{benefit}_table.html`
- **Combined wide table**: `results/tables_2026/complete_wide_table_2026_corrected.html`
- **LaTeX versions**: `results/tables_2026/complete_wide_table_2026.tex`
- **Occupation-year differences**: `results/tables/occ_year_models/difference_regression_table_combined.html`

### Key Figures
- **Model coefficients plot**: `results/figures_2026/regression/model_coefficients_plot_industry_converged.png`
- **Firm-category yearly H1 logit plot**: `results/figures_2026/regression/firm_category_yearly_ai_role_logit_coefficients.png`
- **Firm-category through-2022/post-2022 H1 logit plot**: `results/figures_2026/regression/firm_category_period_ai_role_logit_coefficients.png`
- **Structured-benefit coefficients plot**: `results/figures_2026/regression/structured_benefits_ai_role_coef.png`
- **Figure 1 — AI demand and wage coefficients**: `results/figures_2026/descriptive/figure1_ai_demand_wage_beta.png`
- **Figure 1 by firm category**: `results/figures_2026/descriptive/firm_categories/figure1_ai_demand_wage_beta_by_firm_category.png`
- **Sample composition by firm category**: `results/figures_2026/descriptive/firm_categories/firm_category_sample_sizes.png`
- **AI wage premium by firm category (pooled)**: `results/figures_2026/descriptive/firm_categories/ai_wage_premium_by_firm_category.png`
- **AI wage premium by firm category (annual)**: `results/figures_2026/descriptive/firm_categories/ai_wage_premium_by_firm_category_annual.png`
- **AI wage premium pre vs post GenAI**: `results/figures_2026/descriptive/firm_categories/ai_wage_premium_by_firm_category_pre_post_genai.png`
- **AI perk premium by firm category**: `results/figures_2026/descriptive/firm_categories/benefit_gap_by_firm_category.png`
- **Benefits by role and firm category**: `results/figures_2026/descriptive/firm_categories/benefits_all_ai_role_by_firm_category.png`
- **Keyword benefit differences**: `results/figures_2026/descriptive/benefits_over_time/benefit_diffs_time_keyword.png`
- **Combined keyword + structured differences**: `results/figures_2026/descriptive/benefits_over_time/benefit_diffs_time_all.png`
- **Individual keyword benefit trends**: `results/figures_2026/descriptive/benefits_over_time/benefit_over_time_{benefit}.png` (6 files)
- **Occupation analysis**: `results/figures_2026/descriptive/by_occupation/percent_by_occupation_{benefit}.png` (6 files)
- **Salary analysis**: `results/figures_2026/salary/salary_by_benefit_combined.png`
- **Wage-perk interaction figures**: `results/figures_2026/wage_perk_interactions/`
  - Pooled β₁ AI-role wage premium by firm category: `ai_role_wage_premium_beta1_pooled_by_firm_category.png`
  - Yearly β₁ AI-role wage premium by firm category: `ai_role_wage_premium_beta1_yearly_by_firm_category.png`
  - Pooled β₃ dot plots: `wage_perk_interaction_beta3_pooled.png`
  - Yearly β₃ time series (all perks): `wage_perk_interaction_beta3_yearly.png`
  - **β₃ heatmap (perk × firm type)**: `wage_perk_interaction_beta3_heatmap.png`
  - **β₃ heatmaps through 2022 and post-2022**: `wage_perk_interaction_beta3_heatmap_through_2022.png`, `wage_perk_interaction_beta3_heatmap_post_2022.png`
  - **β₃ single-perk time series with CI ribbons**: `wage_perk_interaction_beta3_timeseries_{perk}.png`
- **Industry-year wage vs perk figures**: `results/figures_2026/industry_year/`
- **Threshold robustness figure**: `results/figures_2026/robustness/ai_threshold_robustness.png`
- **Perk positioning figure**: `results/figures_2026/perk_positioning/perk_positioning_ai_coef.png`
- **Scatterplots**: `results/figures_2026/scatterplots/` (multiple correlation analyses)


## Citation

  If you use this code or data in your research, please cite:

  Castaneda, Bone & Stephany. "Beyond pay: AI skills reward more job benefits." Working Paper, 2025.

## License
