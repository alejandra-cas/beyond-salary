import pandas as pd
import os

# Create latex exports directory
os.makedirs('exports/tables/summary_statistics/occupation-years/latex', exist_ok=True)

print("Converting CSV tables to LaTeX format...")

# List of CSV files to convert
csv_files = [
    'benefit_prevalence_rates.csv',
    'salary_by_benefit_availability.csv', 
    'benefits_by_salary_quartile.csv',
    'yearly_benefit_trends.csv',
    'benefit_salary_correlations.csv'
]

# Also include the original summary tables
original_csv_files = [
    'dataset_overview.csv',
    'ai_demand_summary.csv',
    'salary_summary.csv', 
    'benefits_summary.csv',
    'top_occupations_ai_demand_2024.csv',
    'yearly_trends.csv',
    'occupation_characteristics_2024.csv',
    'benefits_by_top_ai_occupations_2024.csv'
]

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

# Convert occupation-years tables
base_path = 'exports/tables/summary_statistics/occupation-years/'

for csv_file in csv_files:
    if os.path.exists(base_path + csv_file):
        print(f"Converting {csv_file}...")
        
        df = pd.read_csv(base_path + csv_file)
        
        # Create appropriate caption and label based on filename
        if 'benefit_prevalence_rates' in csv_file:
            caption = "Benefit Prevalence Rates by Population (Occupation-Year Level)"
            label = "tab:benefit_prevalence_rates"
        elif 'salary_by_benefit_availability' in csv_file:
            caption = "Median Salaries by Benefit Availability Quartiles (2024)"
            label = "tab:salary_by_benefit"
        elif 'benefits_by_salary_quartile' in csv_file:
            caption = "Benefit Prevalence Rates by AI Salary Quartiles (2024)"
            label = "tab:benefits_by_salary"
        elif 'yearly_benefit_trends' in csv_file:
            caption = "Year-over-Year Benefit Prevalence Trends (2018-2024)"
            label = "tab:yearly_benefit_trends"
        elif 'benefit_salary_correlations' in csv_file:
            caption = "Correlations between Benefit Prevalence and Salary Levels (2024)"
            label = "tab:benefit_salary_corr"
        else:
            caption = f"Summary Statistics Table"
            label = f"tab:{csv_file.replace('.csv', '').replace('_', '')}"
        
        latex_table = create_latex_table(df, csv_file, caption, label)
        
        # Save LaTeX table
        latex_filename = csv_file.replace('.csv', '.tex')
        with open(base_path + 'latex/' + latex_filename, 'w') as f:
            f.write(latex_table)
        
        print(f"✓ {latex_filename} created")

# Convert original summary tables  
summary_path = 'exports/tables/summary_statistics/'

for csv_file in original_csv_files:
    if os.path.exists(summary_path + csv_file):
        print(f"Converting {csv_file}...")
        
        df = pd.read_csv(summary_path + csv_file)
        
        # Create appropriate caption and label based on filename
        if 'dataset_overview' in csv_file:
            caption = "Dataset Overview and Summary Statistics"
            label = "tab:dataset_overview"
        elif 'ai_demand_summary' in csv_file:
            caption = "AI Demand Summary Statistics"
            label = "tab:ai_demand_summary"
        elif 'salary_summary' in csv_file:
            caption = "Salary Summary Statistics by Population"
            label = "tab:salary_summary"
        elif 'benefits_summary' in csv_file:
            caption = "Benefits Prevalence Summary Statistics"
            label = "tab:benefits_summary"
        elif 'top_occupations_ai_demand_2024' in csv_file:
            caption = "Top Occupations by AI Demand (2024)"
            label = "tab:top_ai_occupations"
        elif 'yearly_trends' in csv_file:
            caption = "Year-over-Year Aggregate Trends (2018-2024)"
            label = "tab:yearly_trends"
        elif 'occupation_characteristics_2024' in csv_file:
            caption = "Occupation Characteristics (2024)"
            label = "tab:occupation_chars_2024"
        elif 'benefits_by_top_ai_occupations_2024' in csv_file:
            caption = "Benefit Prevalence in Top AI Occupations (2024)"
            label = "tab:benefits_top_ai_occs"
        else:
            caption = f"Summary Statistics Table"
            label = f"tab:{csv_file.replace('.csv', '').replace('_', '')}"
        
        latex_table = create_latex_table(df, csv_file, caption, label)
        
        # Save LaTeX table
        latex_filename = csv_file.replace('.csv', '.tex')
        with open(base_path + 'latex/' + latex_filename, 'w') as f:
            f.write(latex_table)
        
        print(f"✓ {latex_filename} created")

# Create a master LaTeX file that includes all tables
print("\nCreating master LaTeX file...")

master_latex = """\\documentclass{article}
\\usepackage{booktabs}
\\usepackage{longtable}
\\usepackage{geometry}
\\geometry{margin=1in}
\\usepackage{rotating}

\\title{Beyond Salary: Occupation-Year Analysis Summary Tables}
\\author{Analysis Summary}
\\date{\\today}

\\begin{document}
\\maketitle

\\section{Dataset Overview}
\\input{dataset_overview.tex}

\\section{AI Demand Analysis}
\\input{ai_demand_summary.tex}

\\section{Salary Analysis}
\\input{salary_summary.tex}

\\section{Benefits Analysis}
\\input{benefits_summary.tex}
\\input{benefit_prevalence_rates.tex}

\\section{Salary-Benefit Relationships}
\\input{salary_by_benefit_availability.tex}
\\input{benefits_by_salary_quartile.tex}
\\input{benefit_salary_correlations.tex}

\\section{Temporal Trends}
\\input{yearly_trends.tex}
\\input{yearly_benefit_trends.tex}

\\section{2024 Detailed Analysis}
\\input{occupation_characteristics_2024.tex}
\\input{top_occupations_ai_demand_2024.tex}
\\input{benefits_by_top_ai_occupations_2024.tex}

\\end{document}"""

with open(base_path + 'latex/master_tables.tex', 'w') as f:
    f.write(master_latex)

print("✓ master_tables.tex created with all table includes")

print(f"\nAll LaTeX tables exported to: {base_path}latex/")
print("\nTo compile the full document:")
print(f"cd {base_path}latex/")
print("pdflatex master_tables.tex")

# Create a README for the LaTeX files
readme_content = """# LaTeX Tables for Occupation-Year Analysis

This directory contains LaTeX table files generated from the occupation-year analysis.

## Individual Table Files:
- dataset_overview.tex - Basic dataset metrics
- ai_demand_summary.tex - AI demand statistics 
- salary_summary.tex - Salary analysis by population
- benefits_summary.tex - Benefits prevalence summary
- benefit_prevalence_rates.tex - Benefit rates by population
- salary_by_benefit_availability.tex - Salaries by benefit quartiles
- benefits_by_salary_quartile.tex - Benefits by salary quartiles
- benefit_salary_correlations.tex - Benefit-salary correlations
- yearly_trends.tex - Aggregate yearly trends
- yearly_benefit_trends.tex - Benefit trends over time
- occupation_characteristics_2024.tex - 2024 occupation profiles
- top_occupations_ai_demand_2024.tex - Top AI occupations
- benefits_by_top_ai_occupations_2024.tex - Benefits in top AI occupations

## Master Document:
- master_tables.tex - Includes all tables in a single document

## Usage:
1. Include individual .tex files in your LaTeX document using \\input{filename.tex}
2. Or compile the master document: pdflatex master_tables.tex

## Required LaTeX packages:
- booktabs
- longtable  
- geometry
- rotating
"""

with open(base_path + 'latex/README.md', 'w') as f:
    f.write(readme_content)

print("✓ README.md created with usage instructions")