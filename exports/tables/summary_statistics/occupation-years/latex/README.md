# LaTeX Tables for Occupation-Year Analysis

This directory contains LaTeX table files generated from the occupation-year analysis of non-monetary benefits for AI skills.

## Research Focus
- Unit of analysis: Occupation-Year (105 observations: 15 occupations × 7 years, 2018-2024)
- Research question: Do non-monetary benefits differ for AI skills, and does demand influence benefit offerings?

## Files Converted (20 total):
- ai_demand_distribution.tex - AI Demand Distribution Across Occupation-Years\n- ai_demand_summary.tex - AI Demand Summary Statistics\n- benefit_prevalence_rate_distributions.tex - Benefit Prevalence Rate Distributions Across Occupation-Years\n- benefit_prevalence_rates.tex - Benefit Prevalence Rates by Population (Occupation-Year Level)\n- benefit_salary_correlations.tex - Correlations between Benefit Prevalence and Salary Levels (2024)\n- benefits_by_salary_quartile.tex - Benefit Prevalence Rates by AI Salary Quartiles (2024)\n- benefits_by_top_ai_occupations_2024.tex - Benefit Prevalence in Top AI Occupations (2024)\n- benefits_summary.tex - Benefits Prevalence Summary Statistics\n- bottom_occupation_years_ai_demand.tex - Bottom 10 Occupation-Years by AI Demand\n- dataset_overview.tex - Dataset Overview and Summary Statistics\n- occupation_characteristics_2024.tex - Occupation Characteristics (2024)\n- occupation_year_correlations_matrix.tex - Occupation-Year Correlations Matrix\n- occupation_year_sample_characteristics.tex - Occupation-Year Sample Characteristics\n- salary_by_benefit_availability.tex - Median Salaries by Benefit Availability Quartiles (2024)\n- salary_summary.tex - Salary Summary Statistics by Population\n- temporal_variation_analysis.tex - Temporal and Cross-Sectional Variation Analysis\n- top_occupation_years_ai_demand.tex - Top 10 Occupation-Years by AI Demand\n- top_occupations_ai_demand_2024.tex - Top Occupations by AI Demand (2024)\n- yearly_benefit_trends.tex - Year-over-Year Benefit Prevalence Trends (2018-2024)\n- yearly_trends.tex - Year-over-Year Aggregate Trends (2018-2024)\n
## Master Document:
- comprehensive_occupation_year_analysis.tex - Complete document organized for research paper

## Key Tables for Paper:
### Essential (Core Analysis):
- benefit_prevalence_rates.tex - Prevalence by AI vs non-AI populations ⭐
- benefit_salary_correlations.tex - Demand-benefit relationships ⭐  
- benefits_by_salary_quartile.tex - Benefits by skill demand proxy ⭐
- yearly_benefit_trends.tex - Temporal trends in AI benefits ⭐

### Supporting Evidence:
- salary_by_benefit_availability.tex - Compensation analysis
- benefits_by_top_ai_occupations.tex - High-demand occupation patterns
- benefit_prevalence_rate_distributions.tex - Occupation-year level variation

### Context/Background:
- occupation_year_sample_characteristics.tex - Sample description
- ai_demand_distribution.tex - AI demand landscape
- temporal_variation_analysis.tex - Within/between variation

## Data Features:
- All numeric values rounded to 2 decimal places
- Occupation-year level analysis (distinct from job-level)
- 2018-2024 time series
- Focus on non-monetary benefits for AI skills

## Usage:
1. Include individual .tex files: \input{filename.tex}
2. Compile full document: pdflatex comprehensive_occupation_year_analysis.tex

## Required LaTeX packages:
- booktabs, longtable, geometry, rotating, array, tabularx

## Compilation:
```bash
cd exports/tables/summary_statistics/occupation-years/latex/
pdflatex comprehensive_occupation_year_analysis.tex
pdflatex comprehensive_occupation_year_analysis.tex  # Run twice for TOC
```
