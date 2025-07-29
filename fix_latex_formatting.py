import pandas as pd
import numpy as np
import os
import glob
import re

# Paths
base_path = 'exports/tables/summary_statistics/occupation-years/'
latex_path = base_path + 'latex/'

print("Fixing LaTeX table formatting to remove trailing zeros...")

def format_number(val):
    """Format numbers to remove unnecessary trailing zeros"""
    if pd.isna(val):
        return ''
    
    # If it's already a string, try to process it
    if isinstance(val, str):
        # Check if it contains a percentage
        if '%' in val:
            try:
                num = float(val.replace('%', ''))
                if num == int(num):
                    return f"{int(num)}\\%"
                else:
                    return f"{num:.2f}".rstrip('0').rstrip('.') + "\\%"
            except:
                return val.replace('%', '\\%')
        
        # Check if it contains a dollar sign
        elif '$' in val:
            try:
                num_str = val.replace('$', '').replace(',', '')
                num = float(num_str)
                if num >= 1000:
                    return f"\\${num:,.0f}"
                else:
                    return f"\\${num:.2f}".rstrip('0').rstrip('.')
            except:
                return val.replace('$', '\\$')
        
        # Try to process as a regular number string
        else:
            try:
                num = float(val)
                if num == int(num):
                    return str(int(num))
                else:
                    formatted = f"{num:.2f}".rstrip('0').rstrip('.')
                    return formatted if formatted else '0'
            except:
                return val
    
    # If it's a numeric type
    elif isinstance(val, (int, float)):
        if np.isnan(val):
            return ''
        if val == int(val):
            return str(int(val))
        else:
            formatted = f"{val:.2f}".rstrip('0').rstrip('.')
            return formatted if formatted else '0'
    
    return str(val)

def create_clean_latex_table(df, caption, label):
    """Convert DataFrame to LaTeX table with clean formatting"""
    
    # Create a copy and format all values
    df_clean = df.copy()
    
    # Format all cells
    for col in df_clean.columns:
        df_clean[col] = df_clean[col].apply(format_number)
    
    # Clean up column names for LaTeX
    df_clean.columns = [col.replace('&', '\\&').replace('%', '\\%').replace('_', '\\_') for col in df_clean.columns]
    
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
    
    # Additional cleanup - remove any remaining .00 patterns
    latex_table = re.sub(r'(\d+)\.00(?=\s|&|\\\\)', r'\1', latex_table)
    
    return latex_table

# Get all CSV files that aren't empty
csv_files = []
for file in glob.glob(base_path + '*.csv'):
    filename = os.path.basename(file)
    try:
        df = pd.read_csv(file)
        if not df.empty:
            csv_files.append(filename)
        else:
            print(f"Skipping empty file: {filename}")
    except:
        print(f"Skipping problematic file: {filename}")

print(f"Processing {len(csv_files)} non-empty CSV files:")
for f in sorted(csv_files):
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
    'benefits_by_top_ai_occupations_2024.csv': ("Benefit Prevalence in Top AI Occupations (2024)", "tab:benefits_top_ai_occs"),
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
        
        # Get caption and label
        if csv_file in captions_labels:
            caption, label = captions_labels[csv_file]
        else:
            caption = f"Summary Statistics: {csv_file.replace('.csv', '').replace('_', ' ').title()}"
            label = f"tab:{csv_file.replace('.csv', '').replace('_', '')}"
        
        # Create clean LaTeX table
        latex_table = create_clean_latex_table(df, caption, label)
        
        # Save LaTeX table
        latex_filename = csv_file.replace('.csv', '.tex')
        with open(latex_path + latex_filename, 'w') as f:
            f.write(latex_table)
        
        converted_files.append(latex_filename)
        print(f"  ✓ {latex_filename} created with clean formatting")
        
    except Exception as e:
        print(f"  ✗ Error processing {csv_file}: {str(e)}")

# Create master LaTeX file with only available tables
print(f"\nCreating master LaTeX file with {len(converted_files)} tables...")

# Only include sections for tables that actually exist
available_tables = {f.replace('.tex', '.csv'): f for f in converted_files}

sections = []

if 'occupation_year_sample_characteristics.csv' in available_tables:
    sections.append("""\\section{{Dataset Overview and Sample Description}}
\\subsection{{Sample Characteristics}}
\\input{{occupation_year_sample_characteristics.tex}}""")

if 'dataset_overview.csv' in available_tables:
    sections.append("""\\subsection{{Dataset Overview}}
\\input{{dataset_overview.tex}}""")

if 'ai_demand_distribution.csv' in available_tables:
    sections.append("""\\subsection{{AI Demand Distribution}}
\\input{{ai_demand_distribution.tex}}""")

if any(f in available_tables for f in ['ai_demand_summary.csv', 'salary_summary.csv']):
    section = "\\section{Descriptive Statistics}\n"
    if 'ai_demand_summary.csv' in available_tables:
        section += "\\subsection{AI Demand Summary}\n\\input{ai_demand_summary.tex}\n\n"
    if 'salary_summary.csv' in available_tables:
        section += "\\subsection{Salary Summary by Population}\n\\input{salary_summary.tex}\n\n"
    sections.append(section.rstrip())

if any(f in available_tables for f in ['benefit_prevalence_rate_distributions.csv', 'benefits_summary.csv']):
    section = "\\section{Benefit Statistics}\n"
    if 'benefit_prevalence_rate_distributions.csv' in available_tables:
        section += "\\subsection{Benefit Prevalence Rate Distributions}\n\\input{benefit_prevalence_rate_distributions.tex}\n\n"
    if 'benefits_summary.csv' in available_tables:
        section += "\\subsection{Benefits Summary Statistics}\n\\input{benefits_summary.tex}\n\n"
    sections.append(section.rstrip())

if any(f in available_tables for f in ['temporal_variation_analysis.csv', 'occupation_year_correlations_matrix.csv']):
    section = "\\section{Variation Analysis}\n"
    if 'temporal_variation_analysis.csv' in available_tables:
        section += "\\subsection{Temporal and Cross-Sectional Variation}\n\\input{temporal_variation_analysis.tex}\n\n"
    if 'occupation_year_correlations_matrix.csv' in available_tables:
        section += "\\subsection{Occupation-Year Correlations}\n\\input{occupation_year_correlations_matrix.tex}\n\n"
    sections.append(section.rstrip())

# Add more sections as needed for other available tables...
# Core analysis tables
core_tables = ['benefit_prevalence_rates.csv', 'benefits_by_salary_quartile.csv', 
               'salary_by_benefit_availability.csv', 'benefit_salary_correlations.csv']
if any(f in available_tables for f in core_tables):
    section = "\\section{Benefit-Demand Relationships}\n"
    for table in core_tables:
        if table in available_tables:
            table_name = table.replace('.csv', '.tex')
            section_name = table.replace('.csv', '').replace('_', ' ').title()
            section += f"\\subsection{{{section_name}}}\n\\input{{{table_name}}}\n\n"
    sections.append(section.rstrip())

master_latex = f"""\\documentclass{{article}}
\\usepackage{{booktabs}}
\\usepackage{{longtable}}
\\usepackage{{geometry}}
\\geometry{{margin=1in}}
\\usepackage{{rotating}}
\\usepackage{{array}}
\\usepackage{{tabularx}}

\\title{{Beyond Salary: Occupation-Year Analysis (Clean Formatting)}}
\\author{{Non-Monetary Benefits for AI Skills}}
\\date{{\\today}}

\\begin{{document}}
\\maketitle

\\tableofcontents
\\newpage

{chr(10).join(sections)}

\\end{{document}}"""

with open(latex_path + 'clean_formatted_tables.tex', 'w') as f:
    f.write(master_latex)

print("✓ clean_formatted_tables.tex created")

print(f"\n🎉 Successfully cleaned and formatted {len(converted_files)} LaTeX tables!")
print(f"📁 All files saved to: {latex_path}")
print(f"📄 Clean master document: clean_formatted_tables.tex")
print(f"✨ All trailing zeros removed from numeric values")

print(f"\n📋 Clean LaTeX files created:")
for f in sorted(converted_files):
    print(f"   - {f}")
print("   - clean_formatted_tables.tex")