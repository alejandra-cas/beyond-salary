# Beyond Salary: AI Jobs and Benefit Analysis

This repository contains the reproducible code and analysis for the academic paper on AI job postings and workplace benefits.

## Repository Structure

```
├── analysis/           # Jupyter notebooks and Python scripts for analysis
│   ├── generate_key_figures.py    # Generate main figures for paper
├── scripts/            # Data preparation scripts
│   ├── prepare_data.py           # Initial data merging and cleaning
│   ├── prepare_data_2.py         # Secondary processing and AI classification
├── src/               # Supporting Python modules
│   └── package_files/ # Core utility functions and definitions
├── results/           # Output files
│   ├── figures/       # Generated visualizations
│   └── tables/        # Analysis tables and results
├── data/              # Input data files (not included in repo)
└── requirements/      # Dependencies and setup
    └── requirements.txt
```

## Setup Instructions

1. **Install Dependencies**
   ```bash
   pip install -r requirements/requirements.txt
   ```

2. **Data Requirements**
   Place the following data files in the `data/` directory:
   - `US_10M_SAMP_2018_2024.csv` - Main job postings dataset
   - `us_10m_nointernship_2018_2024_benefits.parquet.gzip` - Processed data with benefits
   - `ai_skill_ids.pkl` - AI skill identifiers
   - `ID_CNTRY_ALL_WHAM.csv` - Remote work classification data
   - `SALARIES.csv` - Additional salary information

3. **Run Analysis**
   ```bash
   # Data preparation (run in order)
   python scripts/prepare_data.py
   python scripts/prepare_data_2.py
   
   # Generate key figures
   python analysis/generate_key_figures.py
   ```

## Key Analysis Components

### Data Preparation (`scripts/`)
- **prepare_data.py**: Merges job posting data with AI skills labels and remote work classification
- **prepare_data_2.py**: Adds experience buckets, completes AI skills classification, and processes salary data

### Figure Generation (`analysis/`)
- **generate_key_figures.py**: Creates the 5 main visualizations used in the paper:
  1. % AI roles over time (overall and industry average)
  2. Difference in benefit prevalence between AI and non-AI roles
  3. Benefit prevalence by role type (AI vs non-AI)
  4. Individual benefit trends over time
  5. Benefit prevalence by occupation

### Supporting Modules (`src/package_files/`)
- **benefits_defns.py**: Benefit category definitions and mappings
- **logit_model.py**: Statistical modeling utilities

## Generated Outputs

The analysis produces the following key figures:
- `results/figures/pct_ai_roles_overall_industry.png`
- `results/figures/benefits_over_time/benefit_diffs_time.png`
- `results/figures/pct_jobs_by_benefit_and_role_type/benefits_ai_role_colors_10m_2024.png`
- `results/figures/benefits_over_time/over_time_color_{benefit}.png` (6 files)
- `results/figures/percent_by_occupation_colors_2024_{benefit}.png` (6 files)

## Data Sources

This analysis uses job posting data spanning 2018-2024, focusing on:
- AI skill requirements identification
- Workplace benefit classification
- Remote work capability assessment
- Industry and occupation categorization

## Citation

[Add your paper citation here]

## License

[Add license information]