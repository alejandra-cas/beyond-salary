import pandas as pd
import os

# Read the data
df = pd.read_csv('exports/occ_year_data/occ_year_analysis_2024_raw.csv')

# Create exports directory if it doesn't exist
os.makedirs('exports/tables/summary_statistics', exist_ok=True)

print("Exporting occupation-year summary tables...")

# 1. Dataset Overview Table
overview_data = {
    'Metric': [
        'Total Observations',
        'Number of Occupations', 
        'Time Period',
        'Total Jobs (All Years)',
        'Total AI Roles (All Years)',
        'Average Years per Occupation'
    ],
    'Value': [
        f"{df.shape[0]:,}",
        f"{df['SOC_2021_2_NAME'].nunique()}",
        f"{df['YEAR'].min()}-{df['YEAR'].max()}",
        f"{df['job_count'].sum():,}",
        f"{df['ai_role_count'].sum():,}",
        f"{df.groupby('SOC_2021_2_NAME')['YEAR'].count().mean():.1f}"
    ]
}
overview_table = pd.DataFrame(overview_data)
overview_table.to_csv('exports/tables/summary_statistics/dataset_overview.csv', index=False)
print("✓ Dataset overview table exported")

# 2. AI Demand Summary Statistics
ai_metrics = ['AI Demand', 'ai_role_count']
ai_summary_data = []

for metric in ai_metrics:
    ai_summary_data.append({
        'Variable': 'AI Demand (%)' if metric == 'AI Demand' else 'AI Role Count',
        'Mean': f"{df[metric].mean():.2f}" if metric == 'AI Demand' else f"{df[metric].mean():.0f}",
        'Median': f"{df[metric].median():.2f}" if metric == 'AI Demand' else f"{df[metric].median():.0f}",
        'Min': f"{df[metric].min():.2f}" if metric == 'AI Demand' else f"{df[metric].min():.0f}",
        'Max': f"{df[metric].max():.2f}" if metric == 'AI Demand' else f"{df[metric].max():.0f}",
        'Std Dev': f"{df[metric].std():.2f}" if metric == 'AI Demand' else f"{df[metric].std():.0f}"
    })

ai_summary_table = pd.DataFrame(ai_summary_data)
ai_summary_table.to_csv('exports/tables/summary_statistics/ai_demand_summary.csv', index=False)
print("✓ AI demand summary table exported")

# 3. Salary Summary Statistics
salary_metrics = {
    'MEAN SALARY': 'Mean Salary (All Jobs)',
    'MEDIAN_SALARY_ai': 'Median Salary (AI Jobs)', 
    'MEDIAN_SALARY_non_ai': 'Median Salary (Non-AI Jobs)',
    'SALARY_PREMIUM': 'Salary Premium (%)'
}

salary_summary_data = []
for metric, label in salary_metrics.items():
    if metric in df.columns:
        if 'PREMIUM' in metric:
            salary_summary_data.append({
                'Variable': label,
                'Mean': f"{df[metric].mean():.1f}%",
                'Median': f"{df[metric].median():.1f}%", 
                'Min': f"{df[metric].min():.1f}%",
                'Max': f"{df[metric].max():.1f}%",
                'Std Dev': f"{df[metric].std():.1f}%"
            })
        else:
            salary_summary_data.append({
                'Variable': label,
                'Mean': f"${df[metric].mean():.0f}",
                'Median': f"${df[metric].median():.0f}",
                'Min': f"${df[metric].min():.0f}",
                'Max': f"${df[metric].max():.0f}",
                'Std Dev': f"${df[metric].std():.0f}"
            })

salary_summary_table = pd.DataFrame(salary_summary_data)
salary_summary_table.to_csv('exports/tables/summary_statistics/salary_summary.csv', index=False)
print("✓ Salary summary table exported")

# 4. Benefits Prevalence Summary
benefits_mapping = {
    'Tuition Assistance': 'Prevalence: Tuition Assistance',
    'Paid Leave': 'Prevalence: Paid Leave',
    'Health & Wellbeing': 'Prevalence: Health and Wellbeing', 
    'Parental Leave': 'Prevalence: Parental Leave',
    'Workplace Culture': 'Prevalence: Workplace Culture',
    'Remote Work': 'Prevalence: Remote Work'
}

benefits_summary_data = []
for benefit_name, col_name in benefits_mapping.items():
    if col_name in df.columns:
        ai_col = col_name + ' (AI)'
        non_ai_col = col_name + ' (Non-AI)'
        
        # Overall prevalence
        benefits_summary_data.append({
            'Benefit': benefit_name,
            'Population': 'Overall',
            'Mean': f"{df[col_name].mean():.0f}",
            'Median': f"{df[col_name].median():.0f}",
            'Min': f"{df[col_name].min():.0f}",
            'Max': f"{df[col_name].max():.0f}",
            'Std Dev': f"{df[col_name].std():.0f}"
        })
        
        # AI roles
        if ai_col in df.columns:
            benefits_summary_data.append({
                'Benefit': benefit_name,
                'Population': 'AI Roles',
                'Mean': f"{df[ai_col].mean():.0f}",
                'Median': f"{df[ai_col].median():.0f}",
                'Min': f"{df[ai_col].min():.0f}",
                'Max': f"{df[ai_col].max():.0f}",
                'Std Dev': f"{df[ai_col].std():.0f}"
            })
        
        # Non-AI roles
        if non_ai_col in df.columns:
            benefits_summary_data.append({
                'Benefit': benefit_name,
                'Population': 'Non-AI Roles',
                'Mean': f"{df[non_ai_col].mean():.0f}",
                'Median': f"{df[non_ai_col].median():.0f}",
                'Min': f"{df[non_ai_col].min():.0f}",
                'Max': f"{df[non_ai_col].max():.0f}",
                'Std Dev': f"{df[non_ai_col].std():.0f}"
            })

benefits_summary_table = pd.DataFrame(benefits_summary_data)
benefits_summary_table.to_csv('exports/tables/summary_statistics/benefits_summary.csv', index=False)
print("✓ Benefits summary table exported")

# 5. Top Occupations by AI Demand (2024)
df_2024 = df[df['YEAR'] == 2024].copy()
top_ai_demand = df_2024.nlargest(10, 'AI Demand')[['SOC_2021_2_NAME', 'AI Demand', 'ai_role_count', 'job_count', 'MEDIAN_SALARY_ai']].copy()
top_ai_demand['AI Demand'] = top_ai_demand['AI Demand'].round(2)
top_ai_demand['MEDIAN_SALARY_ai'] = top_ai_demand['MEDIAN_SALARY_ai'].round(0)
top_ai_demand.columns = ['Occupation', 'AI Demand (%)', 'AI Role Count', 'Total Job Count', 'Median AI Salary ($)']
top_ai_demand.to_csv('exports/tables/summary_statistics/top_occupations_ai_demand_2024.csv', index=False)
print("✓ Top occupations by AI demand (2024) table exported")

# 6. Year-over-year trends table
yearly_trends = df.groupby('YEAR').agg({
    'AI Demand': 'mean',
    'ai_role_count': 'sum', 
    'job_count': 'sum',
    'MEDIAN_SALARY_ai': 'mean',
    'SALARY_PREMIUM': 'mean'
}).round(2)

yearly_trends.columns = ['Avg AI Demand (%)', 'Total AI Roles', 'Total Jobs', 'Avg AI Salary ($)', 'Avg Salary Premium (%)']
yearly_trends.to_csv('exports/tables/summary_statistics/yearly_trends.csv')
print("✓ Yearly trends table exported")

# 7. Occupation characteristics table (2024)
occ_chars_2024 = df_2024[['SOC_2021_2_NAME', 'AI Demand', 'ai_role_count', 'job_count', 
                          'MEDIAN_SALARY_ai', 'MEDIAN_SALARY_non_ai', 'SALARY_PREMIUM']].copy()
occ_chars_2024.columns = ['Occupation', 'AI Demand (%)', 'AI Role Count', 'Total Jobs',
                          'Median AI Salary ($)', 'Median Non-AI Salary ($)', 'Salary Premium (%)']
occ_chars_2024 = occ_chars_2024.round(2)
occ_chars_2024.to_csv('exports/tables/summary_statistics/occupation_characteristics_2024.csv', index=False)
print("✓ Occupation characteristics (2024) table exported")

# 8. Benefits prevalence by top AI occupations (2024)
top_5_ai_occs = df_2024.nlargest(5, 'AI Demand')['SOC_2021_2_NAME'].tolist()
benefits_cols = [col for col in df.columns if 'Prevalence:' in col and '(AI)' in col]

benefits_by_occ = df_2024[df_2024['SOC_2021_2_NAME'].isin(top_5_ai_occs)][['SOC_2021_2_NAME'] + benefits_cols].copy()
benefits_by_occ.columns = ['Occupation'] + [col.replace('Prevalence: ', '').replace(' (AI)', '') for col in benefits_cols]
benefits_by_occ.to_csv('exports/tables/summary_statistics/benefits_by_top_ai_occupations_2024.csv', index=False)
print("✓ Benefits by top AI occupations (2024) table exported")

print(f"\nAll tables exported to: exports/tables/summary_statistics/")
print("Files created:")
for file in [
    'dataset_overview.csv',
    'ai_demand_summary.csv', 
    'salary_summary.csv',
    'benefits_summary.csv',
    'top_occupations_ai_demand_2024.csv',
    'yearly_trends.csv',
    'occupation_characteristics_2024.csv',
    'benefits_by_top_ai_occupations_2024.csv'
]:
    print(f"  - {file}")