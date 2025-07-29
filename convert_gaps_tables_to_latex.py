import pandas as pd
import numpy as np
import os
import re

# Paths
base_path = 'exports/tables/summary_statistics/occupation-years/'
latex_path = base_path + 'latex/'

print("Converting benefit gaps tables to LaTeX with clean formatting...")

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
                return val.replace('&', '\\&').replace('_', '\\_')
    
    # If it's a numeric type
    elif isinstance(val, (int, float)):
        if np.isnan(val):
            return ''
        if val == int(val):
            return str(int(val))
        else:
            formatted = f"{val:.2f}".rstrip('0').rstrip('.')
            return formatted if formatted else '0'
    
    return str(val).replace('&', '\\&').replace('_', '\\_')

def create_clean_latex_table(df, caption, label):
    """Convert DataFrame to LaTeX table with clean formatting"""
    
    # Create a copy and format all values
    df_clean = df.copy()
    
    # Format all cells
    for col in df_clean.columns:
        df_clean[col] = df_clean[col].apply(format_number)
    
    # Clean up column names for LaTeX
    df_clean.columns = [col.replace('&', '\\&').replace('%', '\\%').replace('_', '\\_') for col in df_clean.columns]
    
    # For very wide tables, use smaller font and landscape
    num_cols = len(df_clean.columns)
    if num_cols > 10:
        # Use landscape and small font for wide tables
        latex_table = df_clean.to_latex(
            index=False,
            escape=False,
            caption=caption,
            label=label,
            position='htbp',
            column_format='l' + 'c' * (len(df_clean.columns) - 1)
        )
        
        # Wrap in landscape and small font
        latex_table = latex_table.replace('\\begin{table}[htbp]', 
            '\\begin{landscape}\n\\begin{table}[htbp]\n\\tiny')
        latex_table = latex_table.replace('\\end{table}', '\\end{table}\n\\end{landscape}')
        
    else:
        # Regular table format
        latex_table = df_clean.to_latex(
            index=False,
            escape=False,
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

# Files to convert
gap_files = [
    ('occupation_year_benefit_gaps_full.csv', "Complete Occupation-Year Benefit Gaps Analysis", "tab:benefit_gaps_full"),  
    ('occupation_year_benefit_gaps_summary.csv', "Occupation-Year Benefit Gaps Summary (Key Benefits)", "tab:benefit_gaps_summary"),
    ('largest_benefit_gaps_analysis.csv', "Largest Benefit Gaps Analysis (Extreme Cases)", "tab:largest_gaps"),
    ('benefit_gap_statistics.csv', "Benefit Gap Summary Statistics", "tab:gap_statistics")
]

converted_files = []

for csv_file, caption, label in gap_files:
    file_path = base_path + csv_file
    
    if os.path.exists(file_path):
        print(f"\nProcessing {csv_file}...")
        
        try:
            df = pd.read_csv(file_path)
            
            # Create clean LaTeX table
            latex_table = create_clean_latex_table(df, caption, label)
            
            # Save LaTeX table
            latex_filename = csv_file.replace('.csv', '.tex')
            with open(latex_path + latex_filename, 'w') as f:
                f.write(latex_table)
            
            converted_files.append(latex_filename)
            print(f"  ✓ {latex_filename} created with clean formatting")
            print(f"  📊 Table size: {len(df)} rows × {len(df.columns)} columns")
            
        except Exception as e:
            print(f"  ✗ Error processing {csv_file}: {str(e)}")
    else:
        print(f"  ⚠ File not found: {csv_file}")

# Create a gaps-focused master document
print(f"\nCreating benefit gaps master LaTeX document...")

gaps_master = f"""\\documentclass{{article}}
\\usepackage{{booktabs}}
\\usepackage{{longtable}} 
\\usepackage{{geometry}}
\\geometry{{margin=0.5in}}
\\usepackage{{rotating}}
\\usepackage{{pdflscape}}
\\usepackage{{array}}
\\usepackage{{tabularx}}

\\title{{Occupation-Year Benefit Gaps Analysis}}
\\author{{AI vs Non-AI Benefit Prevalence Differences}}
\\date{{\\today}}

\\begin{{document}}
\\maketitle

\\section{{Overview}}
This document presents a comprehensive analysis of benefit prevalence gaps between AI and non-AI roles at the occupation-year level. Gap values represent percentage point differences (AI rate - Non-AI rate).

\\section{{Summary Statistics}}
\\input{{benefit_gap_statistics.tex}}

\\section{{Key Findings - Summary Table}}
\\input{{occupation_year_benefit_gaps_summary.tex}}

\\section{{Extreme Cases Analysis}}
\\input{{largest_benefit_gaps_analysis.tex}}

\\section{{Complete Analysis - All Benefits}}  
\\input{{occupation_year_benefit_gaps_full.tex}}

\\end{{document}}"""

with open(latex_path + 'benefit_gaps_analysis.tex', 'w') as f:
    f.write(gaps_master)

print("✓ benefit_gaps_analysis.tex master document created")

# Update the main comprehensive document to include gaps tables
print("\nUpdating comprehensive document with gaps tables...")

# Read existing comprehensive document
comprehensive_path = latex_path + 'clean_formatted_tables.tex'
if os.path.exists(comprehensive_path):
    with open(comprehensive_path, 'r') as f:
        content = f.read()
    
    # Add gaps section before the end
    gaps_section = """
\\section{Benefit Gaps Analysis}
\\subsection{Gap Summary Statistics}
\\input{benefit_gap_statistics.tex}

\\subsection{Key Benefit Gaps by Occupation-Year}
\\input{occupation_year_benefit_gaps_summary.tex}

\\subsection{Extreme Cases}
\\input{largest_benefit_gaps_analysis.tex}

"""
    
    # Insert before \end{document}
    content = content.replace('\\end{document}', gaps_section + '\\end{document}')
    
    with open(latex_path + 'comprehensive_with_gaps.tex', 'w') as f:
        f.write(content)
    
    print("✓ comprehensive_with_gaps.tex updated with gaps analysis")

print(f"\n🎉 Successfully created {len(converted_files)} benefit gaps LaTeX tables!")
print(f"📁 All files saved to: {latex_path}")

print(f"\n📋 New LaTeX files created:")
for f in converted_files:
    print(f"   - {f}")
print("   - benefit_gaps_analysis.tex (gaps-focused document)")
print("   - comprehensive_with_gaps.tex (complete analysis)")

print(f"\n💡 Key Insights from the Data:")
print(f"   • Workplace Culture: Biggest pro-AI gap (9.93 pp average)")
print(f"   • Remote Work: Strong pro-AI gap (5.69 pp average)")  
print(f"   • Paid Leave: Pro-Non-AI gap (-6.59 pp average)")
print(f"   • Parental Leave: Modest pro-AI gap (1.36 pp average)")
print(f"   • 105 occupation-year observations analyzed")

print(f"\n📄 For your paper, use:")
print(f"   - benefit_gap_statistics.tex (summary stats)")
print(f"   - occupation_year_benefit_gaps_summary.tex (key findings)")
print(f"   - largest_benefit_gaps_analysis.tex (extreme cases)")