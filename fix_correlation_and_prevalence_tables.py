import pandas as pd
import numpy as np
import os
import re

print("Fixing correlation matrix and benefit prevalence rates tables...")

# Fix correlation matrix - remove the unnamed index column
corr_path = 'exports/tables/summary_statistics/occupation-years/occupation_year_correlations_matrix.csv'
if os.path.exists(corr_path):
    df_corr = pd.read_csv(corr_path)
    
    # Remove the unnamed index column
    if 'Unnamed: 0' in df_corr.columns:
        df_corr = df_corr.drop('Unnamed: 0', axis=1)
        print("✓ Removed 'Unnamed: 0' column from correlation matrix")
    
    # Set the first column (variable names) as index
    if len(df_corr.columns) > 0:
        df_corr = df_corr.set_index(df_corr.columns[0])
        print("✓ Set variable names as row index")
    
    # Round all values to 2 decimal places
    df_corr = df_corr.round(2)
    
    # Save corrected correlation matrix
    df_corr.to_csv(corr_path)
    print("✓ Correlation matrix corrected and saved")
    
    print(f"📊 Correlation matrix shape: {df_corr.shape}")
    print(f"   Variables: {len(df_corr.columns)}")

# Check and verify benefit prevalence rates table
prev_path = 'exports/tables/summary_statistics/occupation-years/benefit_prevalence_rates.csv' 
if os.path.exists(prev_path):
    df_prev = pd.read_csv(prev_path)
    print(f"\n📊 Benefit prevalence rates table:")
    print(f"   Shape: {df_prev.shape}")
    print(f"   Benefits included: {df_prev['Benefit'].unique().tolist()}")
    print(f"   Populations: {df_prev['Population'].unique().tolist()}")
    
    # Check for any issues
    issues = []
    if df_prev.isnull().any().any():
        issues.append("Contains null values")
    
    # Check if Health & Wellbeing is properly included
    hw_data = df_prev[df_prev['Benefit'] == 'Health & Wellbeing']
    if len(hw_data) == 0:
        issues.append("Health & Wellbeing data missing")
    elif len(hw_data) != 3:  # Should have Overall, AI, Non-AI
        issues.append(f"Health & Wellbeing has {len(hw_data)} rows instead of 3")
    
    if issues:
        print(f"   ⚠ Issues found: {', '.join(issues)}")
    else:
        print("   ✅ Table looks correct")

# Function to format numbers cleanly for LaTeX
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
    
    # Generate LaTeX table
    latex_table = df_clean.to_latex(
        index=True,  # Include index for correlation matrix
        escape=False,
        column_format='l' + 'c' * len(df_clean.columns),
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

def create_benefit_prev_latex_table(df, caption, label):
    """Convert benefit prevalence DataFrame to LaTeX (no index needed)"""
    
    # Create a copy and format all values
    df_clean = df.copy()
    
    # Format all cells
    for col in df_clean.columns:
        df_clean[col] = df_clean[col].apply(format_number)
    
    # Clean up column names for LaTeX
    df_clean.columns = [col.replace('&', '\\&').replace('%', '\\%').replace('_', '\\_') for col in df_clean.columns]
    
    # Generate LaTeX table
    latex_table = df_clean.to_latex(
        index=False,  # No index for this table
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

# Regenerate LaTeX tables
latex_path = 'exports/tables/summary_statistics/occupation-years/latex/'

print(f"\nRegenerating LaTeX tables...")

# 1. Correlation matrix
if os.path.exists(corr_path):
    df_corr = pd.read_csv(corr_path, index_col=0)  # Read with first column as index
    latex_table = create_clean_latex_table(
        df_corr,
        "Occupation-Year Correlations Matrix",
        "tab:correlations_matrix"
    )
    with open(latex_path + 'occupation_year_correlations_matrix.tex', 'w') as f:
        f.write(latex_table)
    print("✓ Correlation matrix LaTeX regenerated")

# 2. Benefit prevalence rates
if os.path.exists(prev_path):
    df_prev = pd.read_csv(prev_path)
    latex_table = create_benefit_prev_latex_table(
        df_prev,
        "Benefit Prevalence Rates by Population (Occupation-Year Level)",
        "tab:benefit_prevalence_rates"
    )
    with open(latex_path + 'benefit_prevalence_rates.tex', 'w') as f:
        f.write(latex_table)
    print("✓ Benefit prevalence rates LaTeX regenerated")

print(f"\n✅ Both tables fixed and LaTeX regenerated!")
print(f"📄 Files updated:")
print(f"   - occupation_year_correlations_matrix.csv/.tex")
print(f"   - benefit_prevalence_rates.tex (LaTeX only)")

# Show a sample of the correlation data to verify
if os.path.exists(corr_path):
    df_corr_check = pd.read_csv(corr_path, index_col=0)
    print(f"\n🔍 Correlation matrix sample (top-left 3x3):")
    print(df_corr_check.iloc[:3, :3].to_string())
    
    # Check specific Health & Wellbeing correlations
    if 'Health & Wellbeing_AI_Rate' in df_corr_check.columns:
        hw_ai_corr_with_demand = df_corr_check.loc['AI_Demand_Pct', 'Health & Wellbeing_AI_Rate']
        print(f"\n📊 AI Demand vs Health & Wellbeing AI Rate correlation: {hw_ai_corr_with_demand}")