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
│   ├── wage_info_analysis.py                  # Wage-information prevalence by benefit and AI role
│   ├── wage_perk_interaction_analysis.py      # H2 wage-perk interaction models
│   ├── join_remote_kw.ipynb                   # Remote keyword joining exploration
│   ├── keyword_vs_structured_benefits.ipynb   # Keyword label validation against structured fields
│   ├── new_data_exploration.ipynb             # Exploratory analysis notebook for MAY26 subsample
│   ├── regression_failure_investigation.ipynb # Regression diagnostics notebook
│   └── wage_info.ipynb                        # Wage distribution analysis notebook
├── scripts/            # Data preparation and processing scripts
│   ├── build_sp500_snapshot.py              # Build fixed S&P 500 company-ID snapshot
│   ├── export_samples.py                    # Export balanced samples for analysis
│   ├── full_data_pipeline.py                # Shared streaming/checkpoint utilities
│   ├── build_full_labeled_v2.py             # Assemble full labeled data without body text
│   ├── label_ai_roles_full.py               # Stream AI-skill counts from full skills data
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
│   └── tables_2026/    # Updated tables, grouped by analysis into subfolders
│       ├── descriptive/                     # Descriptive stats and AI wage betas (incl. by firm category)
│       ├── regression/                      # Job-level models, wide table, panel summaries
│       │   └── job_level_model/             # Individual benefit regression tables
│       ├── firm_category_logit/             # Firm-category AI-role logit results
│       ├── structured_benefits/             # Structured-benefit regression outputs
│       ├── wage_perk_interactions/          # Wage×perk interaction results
│       ├── industry_year/                   # Industry-year wage/perk AI-role betas
│       ├── high_ai_firm/                    # Within-firm perk gap and tier summaries
│       ├── positioning/                     # OLS: AI ROLE on perk prominence score
│       ├── robustness/                      # AI-threshold robustness check outputs
│       └── validation/                      # Firm-size cutoff validation artifacts
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
   uv run python analysis/descriptive_analysis.py --use-existing-betas
   uv run python analysis/regression_models.py
   uv run python analysis/firm_category_yearly_regression.py
   uv run python analysis/wage_perk_interaction_analysis.py
   uv run python analysis/occupation_year_prevalence_analysis.py
   uv run python analysis/salary_analysis.py
   uv run python analysis/scatterplot_analysis.py
   ```

5. **Optional full-data preparation**

   The original commands above remain unchanged. For inputs that are too large
   for the original in-memory preparation path, first write compact ID-keyed
   label files and then assemble a separate full-data analysis parquet:

   ```bash
   uv run python scripts/label_ai_roles_full.py
   uv run python scripts/label_benefits.py --stream --workers 6
   uv run python scripts/label_benefits_remote.py --stream --workers 6
   uv run python scripts/build_full_labeled_v2.py
   ```

   These commands read paths from `config.yaml` and default to
   `data/processed/full/labeled_v2.parquet`, leaving the existing
   `data/processed/labeled_v2.parquet` untouched. Labeling uses bounded-memory
   batches, writes atomic checkpoint parts, and resumes only when the input,
   code, and processing settings still match. Existing outputs require an
   explicit `--overwrite` flag.

   The builder checks unique posting IDs, label coverage, completed-output
   metadata, and one-to-one joins. It uses the same 2018 posting-count firm-size
   classification and cutoff as `prepare_data.py`, and retains benefit-position,
   structured-benefit, S&P 500, and configured WHAM fields.

   The historical remote keyword list contains a concatenated entry. Published
   labels remain the default; pass `--correct-remote-keywords` to
   `label_benefits_remote.py --stream` only when intentionally regenerating and
   comparing labels and downstream results.

## Key Analysis Components

Figure 1 extends the AI-skills wage-premium analysis in Bone, Ehlinger, and Stephany (2024), *Skills or Degree? The Rise of Skill-Based Hiring for AI and Green Jobs*. The cited UK analysis explains log asking wages using AI-skill indicators, education, experience, and fixed effects. This US extension plots quarterly `AI ROLE` coefficients directly in log points and intentionally omits occupation fixed effects.

### Data Preparation (`scripts/`)
  - **prepare_data.py**: Loads posts, skills, and body CSVs; classifies AI roles using the SKILL_SUBCATEGORY_NAME field; adds experience buckets, log salary, year, NAICS 3-digit industry, first-year posting-volume firm-size buckets, optional S&P 500 snapshot membership, and WHAM remote work classification. Outputs `data/processed/data_v1.parquet`.
- **label_benefits.py**: Processes and labels workplace benefits from job posting text. Also computes `{BENEFIT}_POSITION` columns (0–100 prominence score, 100 = top of posting) for each benefit.
- **label_benefits_remote.py**: Labels remote work benefits using keyword matching; merges `REMOTE_KW` and `REMOTE_KW_POSITION` directly into `labeled_v1.parquet`
- **full_data_pipeline.py**: Provides optional bounded-memory labeling, validated checkpoints, atomic output replacement, and shared full-data configuration.
- **label_ai_roles_full.py**: Streams the skills input and saves compact AI-skill counts keyed by posting ID.
- **build_full_labeled_v2.py**: Joins compact labels to structured postings, applies the current preparation rules, validates row preservation and coverage, and writes a separate audited full-data parquet.
- **occ_year_analysis.py**: Aggregates data at occupation-year and industry-year levels. Computes AI demand share, salary premiums (mean and median), benefit prevalence by AI/non-AI role, and benefit differences. Uses a `run_analysis()` function that accepts any grouping column (SOC major group, NAICS 2-digit industry, or county). Outputs `occ_year_analysis_raw.parquet` and `ind_year_analysis_raw.parquet`.
- **export_samples.py**: Creates balanced samples for regression analysis
- **validate_firm_size_cutoffs.py**: Iterates posting-count cutoffs (default `5, 10, ..., 100`), excludes S&P 500 firms when `SP500` is available, draws one random sample of firms observed in 2018 for all thresholds, optionally obtains SME/large judgments through the OpenAI Responses API, and writes false-positive, false-negative, SME sensitivity, large-firm sensitivity, precision, accuracy, and recall-plot outputs to `results/tables_2026/validation/firm_size_cutoff_validation/`.

  LLM reviews use `gpt-6-luna` and require web search by default. Each review saves supporting URLs, an evidence explanation, its evidence period, and the number of completed search calls. Reviews without search or from another model are not reused. Use `--no-web-search` to reproduce the earlier review method. For the 2018 definition, compare fine-grained cutoffs with `--cutoffs 1:100:1`; use `--output-dir` and `--plot-output` to preserve earlier results.

### Core Analysis (`analysis/`)
- **descriptive_analysis.py**: Generates descriptive statistics and exploratory data analysis. Figure 1 combines quarterly AI demand with quarterly adjusted `AI ROLE` log-wage coefficients and 95% confidence intervals. Quarterly wage models control for education, experience, NAICS 3-digit industry, and state fixed effects; quarters with fewer than 10 AI wage postings are suppressed. Also generates firm-category analyses: sample composition, AI demand over time, pooled and annual/period-group AI wage premiums, benefit gap (AI minus non-AI perk prevalence), and benefit prevalence by role type — all broken out by SMEs, Large firms, and S&P 500 firms.
  - Default run outputs keyword-benefit figures and all firm-category figures, re-estimating the quarterly wage betas and overwriting `results/tables_2026/descriptive/quarterly_ai_wage_betas.csv`.
  - Add `--include-structured` to also output combined keyword+structured figures.
  - Add `--use-existing-betas` to plot Figure 1 from the saved `quarterly_ai_wage_betas.csv` instead of re-estimating (leaves the table untouched).
- **regression_models.py**: Runs H1 job-level logit models for each benefit with `AI ROLE` as the key predictor.
   1. **M1 (Baseline):** Year FE + NAICS 3-digit FE
   2. **M2 (+ Individual/State Controls):** M1 + State FE + Education FE + Experience FE
   3. **M3 (+ Salary):** M2 + Log Salary
   4. **M4 (+ S&P 500):** M3 + S&P 500 indicator. This is the preferred specification.
   5. **M5 (+ Occupation):** M4 + 2-digit SOC occupation fixed effects using the existing `occupation = 'SOC_MAJOR_GROUP'` field. Missing occupations are retained as `Unknown SOC`. M5 uses the same salary sample, sparse industry/state grouping, and parental-leave year exclusion as M4.

  To run **only M5** for the six keyword benefits, without overwriting any existing results:
  ```bash
  uv run python analysis/regression_models.py --m5-only
  ```
  Each invocation creates a new `results/m5/<UTC timestamp>/` directory containing an AI-role summary (CSV and HTML), full coefficients including fixed effects (CSV), individual benefit coefficient tables (HTML), a coefficient plot (PNG), and a compact model cache. Failed models are recorded in the summary. This mode skips M1–M4, yearly models, and firm-category models.

  To choose the destination explicitly:
  ```bash
  uv run python analysis/regression_models.py --m5-only --output-dir results/m5/soc_occupation_run
  ```
  The destination must not already exist; an existing directory is rejected before fitting or writing. A normal run estimates M1–M5 and includes M5 in the coefficient plot, panel-summary CSVs, and combined LaTeX summary; the legacy HTML tables retain M1–M3. The LaTeX summary keeps grouped benefit headings, numbered model columns, booktabs rules, and the original caption/notes format.

  To update only the LaTeX model summary from saved full-coefficient caches:
  ```bash
  uv run python analysis/regression_models.py --latex-only
  ```
  This saves `results/tables_2026/regression/complete_wide_table.tex` with M1–M5, without refitting or changing HTML tables, figures, or CSVs. It uses the latest main-model cache plus the latest standalone M5 run when needed. Use `--cache-path <main-cache.pkl>` and `--m5-results <M5-run-directory>` to select fits explicitly, or `--latex-output <path>` for another destination.

  To plot **all five models across the six benefits** from saved coefficients, without refitting:
  ```bash
  uv run python analysis/regression_models.py --plot-only
  ```
  This reads the main panel-summary CSV and, if it contains only M1–M4, adds the latest standalone M5 summary. It saves `results/figures_2026/regression/model_coefficients_plot_m1_m5.png`, preserving the original coefficient figure and all tables. Use `--m5-results results/m5/<run-directory>` to select a particular M5 run, `--results-csv <path>` for a different main-model summary, or `--plot-output <path>` to choose the output PNG. The figure displays log-odds coefficients with 95% confidence intervals.
- **firm_category_yearly_regression.py**: Runs the preferred firm-category H1 logit specification separately by firm category and calendar year, then also runs the same firm-category split for `Through 2022` (`YEAR <= 2022`) and `Post-2022` (`YEAR >= 2023`) period groups. Yearly models omit year fixed effects; period-group models include year fixed effects. The default run produces both yearly and period-group outputs. Use `--yearly-only` or `--period-only` for narrower reruns. Sparse or singular cells are recorded as errors rather than changing controls or reference categories.

  To estimate pooled benefit coefficients using only firms explicitly labeled SME/large by the LLM:
  ```bash
  uv run python analysis/regression_models.py --firm-only --llm-only
  ```
  This uses the saved `gpt-6-luna` web-search reviews by default and assigns SME/large categories from their LLM labels. Unknown/failed reviews, unmatched employers, and firms absent in 2018 are excluded. All available posting years are retained for eligible firms. The run reads only model columns for labeled employers and saves a sample audit, coefficient CSV, and plot separately under `results/firm_llm_labeled/`. No new LLM calls are made. Use `--llm-labels <CSV>` to choose another label file or `--firm-output-dir <directory>` for another output root.

  For yearly and pre/post-2022 models on the same labeled-firm sample:
  ```bash
  uv run python analysis/firm_category_yearly_regression.py --llm-only
  ```
  Add `--yearly-only`, `--period-only`, or `--pooled-only` to select one model group. Runs without `--llm-only` use the standard 2018 posting-count categories.

- **wage_perk_interaction_analysis.py**: Runs H2 log-wage models for every perk: `LOG_SALARY ~ AI_ROLE + PERK + AI_ROLE:PERK + Education FE + Experience FE + NAICS 3-digit FE + State FE + Year FE`. Models are estimated for the full wage sample, SMEs, Large firms, and S&P 500 firm subsamples. Calendar-year models omit year FE. The `AI_ROLE:PERK` coefficient captures complementarity when positive and substitution when negative. Firm-split charts show only SMEs, Large firms, and S&P 500 firms. Outputs include pooled and yearly AI-role wage premium (β₁) by firm category, β₃ heatmaps across all perks and firm types for pooled/through-2022/post-2022 models, and per-perk time-series plots with 95% CI ribbons. Use `--plot-only` to regenerate all plots from the existing results CSV without re-running models.
- **occupation_year_prevalence_analysis.py**: OLS regressions on AI benefit prevalence at occupation-year level
- **salary_analysis.py**: Analyzes salary premiums for AI vs non-AI roles
- **scatterplot_analysis.py**: Creates correlation plots and scatter analyses at occupation-year level
- **industry_year_coefficient_analysis.py**: Supplementary prior analysis estimating AI wage and perk coefficients per industry-year cell. It is no longer the primary H2 strategy.
- **high_ai_firm_analysis.py**: Robustness check addressing the concern that large tech firms drive results by offering perks to everyone. Assigns firms to AI-share tiers and computes within-firm perk gaps between AI and non-AI postings. Tuition assistance, paid leave, and parental leave show genuine within-firm AI premiums; remote work and inclusive workplace appear more firm-wide.
- **ai_threshold_robustness.py**: Robustness check on AI role classification threshold. Re-runs the main P1 M2 spec with 1+, 2+, and 3+ AI skill requirements. Inclusive workplace strengthens with stricter thresholds; parental leave and remote work are robust at 1+ and 2+ but lose significance at 3+ (power issue — only 226 postings).
- **perk_positioning_analysis.py**: Existing supplementary analysis of perk prominence. This analysis is currently on hold.
- **keyword_vs_structured_benefits.ipynb**: Validates keyword-based benefit labels against the structured `BENEFIT_NAME` / `BENEFIT_SUBCATEGORY_NAME` / `BENEFIT_CATEGORIES_NAME` fields. Reports precision, recall, and F1 per benefit, with disagreement inspection.
- **structured_benefits_regression.py**: Runs P1 M2 logit models for structured categories (`S_FLEX_WORK`, `S_PROF_DEV`, `S_HEALTH_WELLNESS`, `S_REMOTE`) and outputs a coefficient table + plot. Current takeaway: no notable positive structured-benefit premium for AI postings; coefficients are lower/near-zero for most structured benefits, except remote work which is positive.
- **new_data_exploration.ipynb**: Exploratory analysis notebook for the MAY26 subsample data
- **wage_info_analysis.py**: Plots wage-information prevalence by benefit type and AI-role status, producing per-benefit bar charts showing the share of postings with salary data
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
- Firm size is a posting-volume proxy frozen at the **2018 in the complete posting dataset**. `FIRM_POSTING_COUNT` counts only that year's postings for each firm and is carried unchanged into all later years. `FIRM_SIZE_BASE_YEAR` records the baseline year. Baseline counts are calculated before filtering by salary, benefit, occupation, or firm category.
- Firms absent from the baseline year have missing posting counts and unknown size; their postings remain in full-sample analyses but are excluded from all firm analyses, including the S&P 500 subsample.
- `COMPANY == 0` represents `Unclassified` employers. These 12,855 postings are retained in full-sample analyses but are excluded from firm-category subsamples so they are not treated as one large employer.
- Three mutually exclusive firm categories are used:
  - **SMEs**: firms with 1–5 postings in 2018
  - **Large firms**: firms with 6 or more postings in 2018 (excluding S&P 500)
  - **S&P 500 firms**: firms observed in 2018 and matched to the S&P 500 snapshot (override SME/large classification)

The shared cutoff is `FIRM_SIZE_CUTOFF = 6` in `src/package_files/firm_size.py`, used by data preparation, relabeling, descriptive plots, and H1/H2 firm-category models. It maximizes macro F1 weighted by 2018 postings in the `gpt-6-luna` validation with web search.

To refresh existing processed datasets without repeating benefit labeling:
```bash
uv run python scripts/relabel_firm_size.py --dry-run
uv run python scripts/relabel_firm_size.py
```
The script calculates one baseline mapping from the complete `labeled_v2.parquet` (or `data_v1.parquet` if unavailable), applies it to all three processed Parquets, and saves `firm_size_relabel_audit.csv`. It writes in batches and replaces each original only after a successful write. Use `--source <complete-posting-dataset.parquet>` to select the baseline source, or `--processed-dir <path>` for a different destination directory. Re-estimate firm-category analyses after refreshing; saved figures, model results, and cutoff-validation metrics still describe the previous definition until rerun.

The following sample composition is from the previous firm-size definition and needs recalculation using baseline counts:

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

## Paper Figure Mapping

Paper figure numbers map to repo outputs as follows (note: "Figure 1" in script/file names is an internal name and corresponds to paper Figure 5):

| Paper figure | Content | Output file(s) (`results/figures_2026/`) | Generating script |
|---|---|---|---|
| Figure 2 | Benefit prevalence by role type (left); AI vs non-AI benefit gap over time (right) | `descriptive/pct_jobs_by_benefit_and_role_type/figure2a_benefits_ai_role_keyword.png`; `descriptive/benefits_over_time/benefit_diffs_time_keyword.png` | `analysis/descriptive_analysis.py` |
| Figure 3 | Odds ratios for benefit inclusion in AI vs non-AI roles (3a); yearly Model 1 AI-role coefficients (3b) | `regression/model_coefficients_plot_industry_converged.png`; `regression/yearly_model1_ai_role_coefficients.png` | `analysis/regression_models.py` |
| Figure 4 | Median salary by benefit and role type over time (left, 4a); yearly AI×benefit wage interaction coefficients, full sample (right, 4b) | `salary/salary_by_benefit_combined.png`; `wage_perk_interactions/wage_perk_interaction_beta3_yearly_full_sample_all_perks.png` | `analysis/salary_analysis.py`; `analysis/wage_perk_interaction_analysis.py` |
| Figure 5 | AI demand (upper) and AI wage premium (lower) over time, with before/after-2022 firm-category insets | `descriptive/figure1_ai_demand_wage_beta.png` | `analysis/descriptive_analysis.py` |
| Figure 6 | Benefit-provision logit coefficients for AI vacancies by firm size | `regression/firm_category_ai_role_logit_coefficients.png` | `analysis/firm_category_yearly_regression.py` |
| Figure 7 | AI×benefit wage interaction coefficients by firm size, before vs after GenAI | `wage_perk_interactions/wage_perk_interaction_beta3_heatmap_period_comparison.png` (and `..._through_2022.png` / `..._post_2022.png`) | `analysis/wage_perk_interaction_analysis.py` |

The benefit previously labeled "Workplace Culture" is now labeled **"Inclusive Workplace"** (`CULTURE` internally).

Additional reproducibility tooling:
- `scripts/generate_combined_figures.py --figure 3 --output-dir results/figures_2026_paper` — regenerates `results/figures_2026_paper/figure3_combined.png` with M1–M5 in panel (a) and the saved yearly Model 1 coefficients in panel (b). Reads saved summaries without refitting; if M5 is absent from the main summary, it uses the latest standalone M5 run. Use `--m5-results <M5-run-directory>` to select a particular run. `--figure 3` updates only Figure 3.
- `scripts/plot_benefit_cooccurrence.py` — jobs by number of benefits offered + benefit correlation inset. Computes small aggregate CSVs (`results/tables_2026/descriptive/benefit_cooccurrence_{hist,corr}.csv`) from the labeled parquet on first run, then plots from the CSVs (re-runnable anywhere).
- `analysis/wage_perk_interaction_analysis.py` — every fitting run also exports each fitted model's complete coefficient vector (incl. fixed effects, with SEs, p-values, CIs) to `results/tables_2026/wage_perk_interactions/wage_perk_interaction_full_params.csv` for building regression tables without refitting (disable with `--no-full-params`).

## Generated Outputs

### Regression Tables
- **Individual benefit tables**: `results/tables_2026/regression/job_level_model/{benefit}_table.html`
- **Combined wide table**: `results/tables_2026/regression/complete_wide_table.html`
- **LaTeX versions**: `results/tables_2026/regression/complete_wide_table.tex`
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
