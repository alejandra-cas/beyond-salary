import pandas as pd
import numpy as np
import os
import re

# Read the full data
df_full = pd.read_csv('exports/tables/summary_statistics/occupation-years/occupation_year_benefit_gaps_full.csv')

print("Fixing Health & Wellbeing in summary table and LaTeX formatting...")

# Create a corrected summary table with ALL 6 benefits
summary_cols = ['Occupation', 'Year', 'AI Demand (%)', 'Total Jobs', 'AI Jobs']

# Add ALL 6 benefits (not just 4)
all_benefits = ['Tuition Assistance', 'Paid Leave', 'Health & Wellbeing', 'Parental Leave', 'Workplace Culture', 'Remote Work']
for benefit in all_benefits:
    summary_cols.extend([
        f'{benefit} AI (%)',
        f'{benefit} Non-AI (%)', 
        f'{benefit} Gap (pp)'
    ])

# Create corrected summary table
summary_df_corrected = df_full[summary_cols].copy()

# Save corrected summary table
summary_df_corrected.to_csv('exports/tables/summary_statistics/occupation-years/occupation_year_benefit_gaps_summary_corrected.csv', index=False)
print("✓ Corrected summary table created with all 6 benefits")

# Also create a concise version with just the key metrics for each benefit
concise_cols = ['Occupation', 'Year', 'AI Demand (%)']
for benefit in all_benefits:
    concise_cols.append(f'{benefit} Gap (pp)')

concise_df = df_full[concise_cols].copy()
concise_df.to_csv('exports/tables/summary_statistics/occupation-years/occupation_year_benefit_gaps_concise.csv', index=False)
print("✓ Concise gaps table created (gaps only)")

# Function to format numbers cleanly
def format_number(val):
    """Format numbers to remove unnecessary trailing zeros"""
    if pd.isna(val):
        return ''
    
    if isinstance(val, str):
        try:
            num = float(val)
            if num == int(num):
                return str(int(num))
            else:
                formatted = f"{num:.2f}".rstrip('0').rstrip('.')
                return formatted if formatted else '0'
        except:
            return val.replace('&', '\\&').replace('_', '\\_')
    
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
    if num_cols > 12:
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
            '\\begin{landscape}\n\\begin{table}[htbp]\n\\scriptsize')
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

# Convert corrected tables to LaTeX
latex_path = 'exports/tables/summary_statistics/occupation-years/latex/'

tables_to_convert = [
    ('occupation_year_benefit_gaps_summary_corrected.csv', 
     "Occupation-Year Benefit Gaps Summary (All Benefits)", 
     "tab:benefit_gaps_summary_corrected"),
    ('occupation_year_benefit_gaps_concise.csv', 
     "Occupation-Year Benefit Gaps (Gaps Only)", 
     "tab:benefit_gaps_concise")
]

print("\nConverting corrected tables to LaTeX...")

for csv_file, caption, label in tables_to_convert:
    file_path = f'exports/tables/summary_statistics/occupation-years/{csv_file}'
    
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
            
            print(f"  ✓ {latex_filename} created")
            print(f"  📊 Table size: {len(df)} rows × {len(df.columns)} columns")
            
            # Show first few Health & Wellbeing values to verify
            if 'Health & Wellbeing Gap (pp)' in df.columns:
                hw_sample = df[['Occupation', 'Year', 'Health & Wellbeing Gap (pp)']].head(3)
                print(f"  🔍 Health & Wellbeing sample:")
                for _, row in hw_sample.iterrows():
                    print(f"     {row['Occupation'][:30]}... {row['Year']}: {row['Health & Wellbeing Gap (pp)']} pp")
            
        except Exception as e:
            print(f"  ✗ Error processing {csv_file}: {str(e)}")

# Also re-generate the LaTeX for the original summary table to make sure it's clean
print(f"\nRe-generating original summary table LaTeX...")
original_summary = pd.read_csv('exports/tables/summary_statistics/occupation-years/occupation_year_benefit_gaps_summary.csv')
latex_table = create_clean_latex_table(
    original_summary, 
    "Occupation-Year Benefit Gaps Summary (Key Benefits)", 
    "tab:benefit_gaps_summary"
)
with open(latex_path + 'occupation_year_benefit_gaps_summary.tex', 'w') as f:
    f.write(latex_table)
print("✓ Original summary table LaTeX updated")

print(f"\n📊 Summary of Health & Wellbeing Data:")
hw_stats = df_full['Health & Wellbeing Gap (pp)']
print(f"   Mean Gap: {hw_stats.mean():.2f} pp")
print(f"   Median Gap: {hw_stats.median():.2f} pp")
print(f"   Min Gap: {hw_stats.min():.2f} pp")
print(f"   Max Gap: {hw_stats.max():.2f} pp")
print(f"   Pro-AI cases: {(hw_stats > 0).sum()}/105")
print(f"   Pro-Non-AI cases: {(hw_stats < 0).sum()}/105")

print(f"\n✅ Health & Wellbeing tables fixed and verified!")
print(f"📄 New files created:")
print(f"   - occupation_year_benefit_gaps_summary_corrected.csv/.tex (all 6 benefits)")
print(f"   - occupation_year_benefit_gaps_concise.csv/.tex (gaps only)")
print(f"   - Original summary table LaTeX regenerated")