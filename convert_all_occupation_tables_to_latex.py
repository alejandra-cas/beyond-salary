import pandas as pd
import os
import glob

# Path to occupation-years folder
base_path = 'exports/tables/summary_statistics/occupation-years/'
latex_path = base_path + 'latex/'

# Create latex directory if it doesn't exist
os.makedirs(latex_path, exist_ok=True)

print("Converting all CSV tables in occupation-years folder to LaTeX format...")

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

# Get all CSV files in the occupation-years directory
csv_files = glob.glob(base_path + '*.csv')
csv_files = [os.path.basename(f) for f in csv_files]

print(f"Found {len(csv_files)} CSV files to convert:")
for f in csv_files:
    print(f"  - {f}")

# Caption and label mappings
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
    'benefits_by_top_ai_occupations_2024.csv': ("Benefit Prevalence in Top AI Occupations (2024)", "tab:benefits_top_ai_occs")
}

# Convert each CSV file
converted_files = []
for csv_file in csv_files:
    print(f"\nConverting {csv_file}...")
    
    try:
        df = pd.read_csv(base_path + csv_file)
        
        # Get caption and label, or create default ones
        if csv_file in captions_labels:
            caption, label = captions_labels[csv_file]
        else:
            caption = f"Summary Statistics: {csv_file.replace('.csv', '').replace('_', ' ').title()}"
            label = f"tab:{csv_file.replace('.csv', '').replace('_', '')}"
        
        latex_table = create_latex_table(df, csv_file, caption, label)
        
        # Save LaTeX table
        latex_filename = csv_file.replace('.csv', '.tex')
        with open(latex_path + latex_filename, 'w') as f:
            f.write(latex_table)
        
        converted_files.append(latex_filename)
        print(f"✓ {latex_filename} created")
        
    except Exception as e:
        print(f"✗ Error converting {csv_file}: {str(e)}")

# Create a comprehensive master LaTeX file
print("\nCreating comprehensive master LaTeX file...")

master_latex = f"""\\documentclass{{article}}
\\usepackage{{booktabs}}
\\usepackage{{longtable}}
\\usepackage{{geometry}}
\\geometry{{margin=1in}}
\\usepackage{{rotating}}
\\usepackage{{array}}
\\usepackage{{tabularx}}

\\title{{Beyond Salary: Comprehensive Occupation-Year Analysis Tables}}
\\author{{Analysis Summary}}
\\date{{\\today}}

\\begin{{document}}
\\maketitle

\\tableofcontents
\\newpage

\\section{{Dataset Overview}}
\\input{{dataset_overview.tex}}

\\section{{AI Demand Analysis}}
\\input{{ai_demand_summary.tex}}

\\section{{Salary Analysis}}
\\input{{salary_summary.tex}}

\\section{{Benefits Analysis}}
\\subsection{{Overall Benefits Summary}}
\\input{{benefits_summary.tex}}

\\subsection{{Benefit Prevalence Rates}}
\\input{{benefit_prevalence_rates.tex}}

\\section{{Salary-Benefit Relationships}}
\\subsection{{Salaries by Benefit Availability}}
\\input{{salary_by_benefit_availability.tex}}

\\subsection{{Benefits by Salary Quartiles}}
\\input{{benefits_by_salary_quartile.tex}}

\\subsection{{Benefit-Salary Correlations}}
\\input{{benefit_salary_correlations.tex}}

\\section{{Temporal Trends}}
\\subsection{{Aggregate Yearly Trends}}
\\input{{yearly_trends.tex}}

\\subsection{{Benefit Trends Over Time}}
\\input{{yearly_benefit_trends.tex}}

\\section{{2024 Detailed Analysis}}
\\subsection{{Occupation Characteristics}}
\\input{{occupation_characteristics_2024.tex}}

\\subsection{{Top AI Occupations}}
\\input{{top_occupations_ai_demand_2024.tex}}

\\subsection{{Benefits in Top AI Occupations}}
\\input{{benefits_by_top_ai_occupations_2024.tex}}

\\end{{document}}"""

with open(latex_path + 'comprehensive_tables.tex', 'w') as f:
    f.write(master_latex)

print("✓ comprehensive_tables.tex created")

# Update README
readme_content = f"""# LaTeX Tables for Occupation-Year Analysis

This directory contains LaTeX table files generated from the occupation-year analysis.

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
## Master Documents:
- comprehensive_tables.tex - Complete document with all tables organized by section

## Usage:
1. Include individual .tex files in your LaTeX document using \\input{{filename.tex}}
2. Compile the comprehensive document: pdflatex comprehensive_tables.tex

## Required LaTeX packages:
- booktabs
- longtable  
- geometry
- rotating
- array
- tabularx

## Compilation Instructions:
```bash
cd {latex_path}
pdflatex comprehensive_tables.tex
pdflatex comprehensive_tables.tex  # Run twice for TOC
```

## Notes:
- All special characters (%, &, _) have been properly escaped for LaTeX
- Tables use consistent formatting with captions and labels
- The comprehensive document includes a table of contents
"""

with open(latex_path + 'README.md', 'w') as f:
    f.write(readme_content)

print("✓ README.md updated")

print(f"\n🎉 Successfully converted {len(converted_files)} tables to LaTeX!")
print(f"📁 All LaTeX files saved to: {latex_path}")
print(f"📄 Main document: comprehensive_tables.tex")
print("\n📋 Files created:")
for f in sorted(converted_files):
    print(f"   - {f}")
print("   - comprehensive_tables.tex")
print("   - README.md")