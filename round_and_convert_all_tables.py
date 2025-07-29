import pandas as pd
import numpy as np
import os
import glob

# Paths
base_path = 'exports/tables/summary_statistics/occupation-years/'
latex_path = base_path + 'latex/'

# Create latex directory
os.makedirs(latex_path, exist_ok=True)

print("Rounding all existing tables to 2 decimal places and converting all tables to LaTeX...")

def round_numeric_columns(df):
    """Round all numeric columns to 2 decimal places"""
    df_rounded = df.copy()
    
    for col in df_rounded.columns:
        if df_rounded[col].dtype in ['float64', 'int64']:
            df_rounded[col] = df_rounded[col].round(2)
        elif df_rounded[col].dtype == 'object':
            # Try to extract and round numbers in string columns
            try:
                # Check if column contains percentage values
                if df_rounded[col].astype(str).str.contains('%').any():
                    # Extract numbers and round them
                    df_rounded[col] = df_rounded[col].astype(str).str.replace('%', '').astype(float).round(2).astype(str) + '%'
                # Check if column contains dollar values  
                elif df_rounded[col].astype(str).str.contains('\\$').any():
                    # Extract numbers and round them
                    numeric_vals = df_rounded[col].astype(str).str.replace('\\$', '').str.replace(',', '').astype(float).round(2)
                    df_rounded[col] = '$' + numeric_vals.astype(str)
                # Check if column contains pure numbers as strings
                elif df_rounded[col].astype(str).str.match(r'^-?\d+\.?\d*$').any():
                    df_rounded[col] = df_rounded[col].astype(float).round(2).astype(str)
            except:
                # If conversion fails, leave as is
                pass
    
    return df_rounded

def create_latex_table(df, filename, caption, label):
    """Convert DataFrame to LaTeX table with formatting"""
    
    # Clean up column names for LaTeX
    df_clean = df.copy()
    
    # Replace problematic characters in column names
    df_clean.columns = [col.replace('&', '\\&').replace('%', '\\%').replace('_', '\\_') for col in df_clean.columns]
    
    # Replace problematic characters in data
    for col in df_clean.select_dtypes(include=['object']).columns:
        df_clean[col] = df_clean[col].astype(str).str.replace('&', '\\&')
        df_clean[col] = df_clean[col].str.replace('%', '\\%')
        df_clean[col] = df_clean[col].str.replace('_', '\\_')
    
    # Generate LaTeX table
    latex_table = df_clean.to_latex(
        index=False,
        escape=False,  # Since we already escaped special characters
        column_format='l' + 'c' * (len(df_clean.columns) - 1),
        caption=caption,
        label=label,
        position='htbp'
    )
    
    # Add some formatting improvements
    latex_table = latex_table.replace('\\toprule', '\\hline\\hline')
    latex_table = latex_table.replace('\\midrule', '\\hline')
    latex_table = latex_table.replace('\\bottomrule', '\\hline\\hline')
    
    return latex_table

# Get all CSV files
csv_files = glob.glob(base_path + '*.csv')
csv_files = [os.path.basename(f) for f in csv_files]

print(f"Found {len(csv_files)} CSV files to process:")
for f in sorted(csv_files):
    print(f"  - {f}")

# Caption and label mappings - including new tables
captions_labels = {
    'dataset_overview.csv': ("Dataset Overview and Summary Statistics", "tab:dataset_overview"),
    'ai_demand_summary.csv': ("AI Demand Summary Statistics", "tab:ai_demand_summary"),
    'salary_summary.csv': ("Salary Summary Statistics by Population", "tab:salary_summary"),
    'benefits_summary.csv': ("Benefits Prevalence Summary Statistics", "tab:benefits_summary"),
    'benefit_prevalence_rates.csv': ("Benefit Prevalence Rates by Population (Occupation-Year Level)", "tab:benefit_prevalence_rates"),
    'salary_by_benefit_availability.csv': ("Median Salaries by Benefit Availability Quartiles (2024)", "tab:salary_by_benefit"),
    'benefits_by_salary_quartile.csv': ("Benefit Prevalence Rates by AI Salary Quartiles (2024)", "tab:benefits_by_salary"),
    'yearly_benefit_trends.csv': ("Year-over-Year Benefit Prevalence Trends (2018-2024)", "tab:yearly_benefit_trends"),
    'benefit_salary_correlations.csv': ("Correlations between Benefit Prevalence and Salary Levels (2024)", "tab:benefit_salary_corr"),
    'top_occupations_ai_demand_2024.csv': ("Top Occupations by AI Demand (2024)", "tab:top_ai_occupations"),
    'yearly_trends.csv': ("Year-over-Year Aggregate Trends (2018-2024)", "tab:yearly_trends"),
    'occupation_characteristics_2024.csv': ("Occupation Characteristics (2024)", "tab:occupation_chars_2024"),
    'benefits_by_top_ai_occupations_2024.csv': ("Benefit Prevalence in Top AI Occupations (2024)", "tab:benefits_top_ai_occs"),
    # New tables
    'occupation_year_sample_characteristics.csv': ("Occupation-Year Sample Characteristics", "tab:sample_characteristics"),
    'benefit_prevalence_rate_distributions.csv': ("Benefit Prevalence Rate Distributions Across Occupation-Years", "tab:prevalence_distributions"),
    'ai_demand_distribution.csv': ("AI Demand Distribution Across Occupation-Years", "tab:ai_demand_distribution"),
    'temporal_variation_analysis.csv': ("Temporal and Cross-Sectional Variation Analysis", "tab:temporal_variation"),
    'occupation_year_correlations_matrix.csv': ("Occupation-Year Correlations Matrix", "tab:correlations_matrix"),
    'top_occupation_years_ai_demand.csv': ("Top 10 Occupation-Years by AI Demand", "tab:top_occ_years"),
    'bottom_occupation_years_ai_demand.csv': ("Bottom 10 Occupation-Years by AI Demand", "tab:bottom_occ_years")
}

# Process each CSV file
converted_files = []
for csv_file in sorted(csv_files):
    print(f"\nProcessing {csv_file}...")
    
    try:
        df = pd.read_csv(base_path + csv_file)
        
        # Round numeric values to 2 decimal places
        df_rounded = round_numeric_columns(df)
        
        # Save the rounded CSV
        df_rounded.to_csv(base_path + csv_file, index=False)
        print(f"  ✓ Rounded values to 2 decimal places")
        
        # Get caption and label
        if csv_file in captions_labels:
            caption, label = captions_labels[csv_file]
        else:
            caption = f"Summary Statistics: {csv_file.replace('.csv', '').replace('_', ' ').title()}"
            label = f"tab:{csv_file.replace('.csv', '').replace('_', '')}"
        
        # Create LaTeX table
        latex_table = create_latex_table(df_rounded, csv_file, caption, label)
        
        # Save LaTeX table
        latex_filename = csv_file.replace('.csv', '.tex')
        with open(latex_path + latex_filename, 'w') as f:
            f.write(latex_table)
        
        converted_files.append(latex_filename)
        print(f"  ✓ {latex_filename} created")
        
    except Exception as e:
        print(f"  ✗ Error processing {csv_file}: {str(e)}")

# Create comprehensive master LaTeX file
print(f"\nCreating comprehensive master LaTeX file...")

master_latex = f"""\\documentclass{{article}}
\\usepackage{{booktabs}}
\\usepackage{{longtable}}
\\usepackage{{geometry}}
\\geometry{{margin=1in}}
\\usepackage{{rotating}}
\\usepackage{{array}}
\\usepackage{{tabularx}}

\\title{{Beyond Salary: Comprehensive Occupation-Year Analysis}}
\\author{{Non-Monetary Benefits for AI Skills}}
\\date{{\\today}}

\\begin{{document}}
\\maketitle

\\tableofcontents
\\newpage

\\section{{Dataset Overview and Sample Description}}
\\subsection{{Sample Characteristics}}
\\input{{occupation_year_sample_characteristics.tex}}

\\subsection{{Dataset Overview}}
\\input{{dataset_overview.tex}}

\\subsection{{AI Demand Distribution}}
\\input{{ai_demand_distribution.tex}}

\\section{{Descriptive Statistics}}
\\subsection{{AI Demand Summary}}
\\input{{ai_demand_summary.tex}}

\\subsection{{Salary Summary by Population}}
\\input{{salary_summary.tex}}

\\subsection{{Benefit Prevalence Rate Distributions}}
\\input{{benefit_prevalence_rate_distributions.tex}}

\\subsection{{Benefits Summary Statistics}}
\\input{{benefits_summary.tex}}

\\section{{Variation Analysis}}
\\subsection{{Temporal and Cross-Sectional Variation}}
\\input{{temporal_variation_analysis.tex}}

\\subsection{{Occupation-Year Correlations}}
\\input{{occupation_year_correlations_matrix.tex}}

\\section{{Benefit-Demand Relationships}}
\\subsection{{Benefit Prevalence by Population}}
\\input{{benefit_prevalence_rates.tex}}

\\subsection{{Benefits by AI Salary Quartiles}}
\\input{{benefits_by_salary_quartile.tex}}

\\subsection{{Salary by Benefit Availability}}
\\input{{salary_by_benefit_availability.tex}}

\\subsection{{Benefit-Salary Correlations}}
\\input{{benefit_salary_correlations.tex}}

\\section{{Temporal Trends}}
\\subsection{{Aggregate Yearly Trends}}
\\input{{yearly_trends.tex}}

\\subsection{{Benefit Trends Over Time}}
\\input{{yearly_benefit_trends.tex}}

\\section{{2024 Cross-Section Analysis}}
\\subsection{{Occupation Characteristics}}
\\input{{occupation_characteristics_2024.tex}}

\\subsection{{Top AI Occupations}}
\\input{{top_occupations_ai_demand_2024.tex}}

\\subsection{{Benefits in Top AI Occupations}}
\\input{{benefits_by_top_ai_occupations_2024.tex}}

\\section{{Extreme Cases Analysis}}
\\subsection{{Highest AI Demand Occupation-Years}}
\\input{{top_occupation_years_ai_demand.tex}}

\\subsection{{Lowest AI Demand Occupation-Years}}
\\input{{bottom_occupation_years_ai_demand.tex}}

\\end{{document}}"""

with open(latex_path + 'comprehensive_occupation_year_analysis.tex', 'w') as f:
    f.write(master_latex)

print("✓ comprehensive_occupation_year_analysis.tex created")

# Update README
readme_content = f"""# LaTeX Tables for Occupation-Year Analysis

This directory contains LaTeX table files generated from the occupation-year analysis of non-monetary benefits for AI skills.

## Research Focus
- Unit of analysis: Occupation-Year (105 observations: 15 occupations × 7 years, 2018-2024)
- Research question: Do non-monetary benefits differ for AI skills, and does demand influence benefit offerings?

## Files Converted ({len(converted_files)} total):
"""

for latex_file in sorted(converted_files):
    csv_name = latex_file.replace('.tex', '.csv')
    if csv_name in captions_labels:
        caption, _ = captions_labels[csv_name]
        readme_content += f"- {latex_file} - {caption}\\n"
    else:
        readme_content += f"- {latex_file}\\n"

readme_content += f"""
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
1. Include individual .tex files: \\input{{filename.tex}}
2. Compile full document: pdflatex comprehensive_occupation_year_analysis.tex

## Required LaTeX packages:
- booktabs, longtable, geometry, rotating, array, tabularx

## Compilation:
```bash
cd {latex_path}
pdflatex comprehensive_occupation_year_analysis.tex
pdflatex comprehensive_occupation_year_analysis.tex  # Run twice for TOC
```
"""

with open(latex_path + 'README.md', 'w') as f:
    f.write(readme_content)

print("✓ README.md updated")

print(f"\n🎉 Successfully processed {len(converted_files)} tables!")
print(f"📁 All files saved to: {latex_path}")
print(f"📄 Main document: comprehensive_occupation_year_analysis.tex")
print(f"💡 All numeric values rounded to 2 decimal places")
print(f"📊 Ready for academic paper on non-monetary benefits for AI skills")

print(f"\n📋 All LaTeX files:")
for f in sorted(converted_files):
    print(f"   - {f}")
print("   - comprehensive_occupation_year_analysis.tex")
print("   - README.md")