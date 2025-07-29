import pandas as pd
import numpy as np
import os

# Read the full data
df = pd.read_csv('exports/occ_year_data/occ_year_analysis_2024_raw.csv')

print("Recreating benefit-salary correlations table with correct data...")

# Benefits mapping
benefits_mapping = {
    'Tuition Assistance': 'Prevalence: Tuition Assistance',
    'Paid Leave': 'Prevalence: Paid Leave',
    'Health & Wellbeing': 'Prevalence: Health and Wellbeing', 
    'Parental Leave': 'Prevalence: Parental Leave',
    'Workplace Culture': 'Prevalence: Workplace Culture',
    'Remote Work': 'Prevalence: Remote Work'
}

# Create correlation analysis using ALL data (2018-2024), not just 2024
print("Analyzing correlations using full dataset (105 occupation-years)...")

correlation_data = []
df_analysis = df.copy()

for benefit_name, col_name in benefits_mapping.items():
    if col_name in df.columns:
        ai_col = col_name + ' (AI)'
        non_ai_col = col_name + ' (Non-AI)'
        
        # Calculate benefit rates for all occupation-years
        overall_rate = df_analysis[col_name] / df_analysis['job_count'] * 100
        
        # AI rate (handle division by zero)
        ai_rate_series = df_analysis[ai_col] / df_analysis['ai_role_count'] * 100
        ai_rate = ai_rate_series.fillna(0).replace([float('inf'), -float('inf')], 0)
        
        # Non-AI rate  
        non_ai_rate = df_analysis[non_ai_col] / (df_analysis['job_count'] - df_analysis['ai_role_count']) * 100
        
        # Correlations with salaries
        corr_overall_ai_salary = overall_rate.corr(df_analysis['MEDIAN_SALARY_ai'])
        corr_overall_nonai_salary = overall_rate.corr(df_analysis['MEDIAN_SALARY_non_ai'])
        corr_ai_rate_ai_salary = ai_rate.corr(df_analysis['MEDIAN_SALARY_ai'])
        
        # Also correlate with AI demand (this is important!)
        corr_overall_ai_demand = overall_rate.corr(df_analysis['AI Demand'])
        corr_ai_rate_ai_demand = ai_rate.corr(df_analysis['AI Demand'])
        
        correlation_data.append({
            'Benefit': benefit_name,
            'Overall Rate vs AI Salary': round(corr_overall_ai_salary, 2),
            'Overall Rate vs Non-AI Salary': round(corr_overall_nonai_salary, 2),
            'AI Rate vs AI Salary': round(corr_ai_rate_ai_salary, 2),
            'Overall Rate vs AI Demand': round(corr_overall_ai_demand, 2),
            'AI Rate vs AI Demand': round(corr_ai_rate_ai_demand, 2),
            'Mean Overall Rate (%)': round(overall_rate.mean(), 2),
            'Mean AI Rate (%)': round(ai_rate.mean(), 2),
            'Mean Non-AI Rate (%)': round(non_ai_rate.mean(), 2)
        })

correlation_table = pd.DataFrame(correlation_data)

# Save the corrected table
correlation_table.to_csv('exports/tables/summary_statistics/occupation-years/benefit_salary_correlations.csv', index=False)
print("✓ Corrected benefit-salary correlations table created")

# Also create a version focused on AI demand correlations
demand_corr_data = []
for benefit_name, col_name in benefits_mapping.items():
    if col_name in df.columns:
        ai_col = col_name + ' (AI)'
        
        # Calculate benefit rates
        overall_rate = df_analysis[col_name] / df_analysis['job_count'] * 100
        ai_rate_series = df_analysis[ai_col] / df_analysis['ai_role_count'] * 100
        ai_rate = ai_rate_series.fillna(0).replace([float('inf'), -float('inf')], 0)
        
        # Correlations with AI demand
        corr_overall_demand = overall_rate.corr(df_analysis['AI Demand'])
        corr_ai_rate_demand = ai_rate.corr(df_analysis['AI Demand'])
        
        demand_corr_data.append({
            'Benefit': benefit_name,
            'Overall Rate vs AI Demand': round(corr_overall_demand, 2),
            'AI Rate vs AI Demand': round(corr_ai_rate_demand, 2),
            'Mean Overall Rate (%)': round(overall_rate.mean(), 2),
            'Mean AI Rate (%)': round(ai_rate.mean(), 2),
            'Observations': len(df_analysis)
        })

demand_corr_table = pd.DataFrame(demand_corr_data)
demand_corr_table.to_csv('exports/tables/summary_statistics/occupation-years/benefit_ai_demand_correlations.csv', index=False)
print("✓ Benefit-AI demand correlations table created")

# Convert to LaTeX with clean formatting
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
    
    return latex_table

# Create LaTeX versions
latex_path = 'exports/tables/summary_statistics/occupation-years/latex/'

# Main correlations table
latex_table = create_clean_latex_table(
    correlation_table,
    "Correlations between Benefit Prevalence and Salary/Demand Levels (All Years)",
    "tab:benefit_salary_corr"
)
with open(latex_path + 'benefit_salary_correlations.tex', 'w') as f:
    f.write(latex_table)
print("✓ Corrected benefit-salary correlations LaTeX created")

# AI demand correlations table
latex_table_demand = create_clean_latex_table(
    demand_corr_table,
    "Correlations between Benefit Prevalence and AI Demand (All Years)",
    "tab:benefit_ai_demand_corr"
)
with open(latex_path + 'benefit_ai_demand_correlations.tex', 'w') as f:
    f.write(latex_table_demand)
print("✓ Benefit-AI demand correlations LaTeX created")

print(f"\n📊 Key Findings from Corrected Analysis:")
print(f"📈 Correlations with AI Demand:")
for _, row in demand_corr_table.iterrows():
    benefit = row['Benefit']
    overall_corr = row['Overall Rate vs AI Demand']
    ai_corr = row['AI Rate vs AI Demand']
    print(f"   {benefit}: Overall={overall_corr}, AI Rate={ai_corr}")

print(f"\n✅ Tables fixed and recreated!")
print(f"📄 Files created/updated:")
print(f"   - benefit_salary_correlations.csv/.tex (corrected)")
print(f"   - benefit_ai_demand_correlations.csv/.tex (new)")
print(f"\n💡 Now uses all 105 occupation-year observations (2018-2024)")
print(f"💡 Includes correlations with AI demand (key for your research question)")