import pandas as pd
import os

# Read the data
df = pd.read_csv('exports/occ_year_data/occ_year_analysis_2024_raw.csv')

# Create exports directory if it doesn't exist
os.makedirs('exports/tables/summary_statistics/occupation-years', exist_ok=True)

print("Exporting benefit prevalence and salary analysis tables...")

# Benefits mapping
benefits_mapping = {
    'Tuition Assistance': 'Prevalence: Tuition Assistance',
    'Paid Leave': 'Prevalence: Paid Leave',
    'Health & Wellbeing': 'Prevalence: Health and Wellbeing', 
    'Parental Leave': 'Prevalence: Parental Leave',
    'Workplace Culture': 'Prevalence: Workplace Culture',
    'Remote Work': 'Prevalence: Remote Work'
}

# 1. Benefit Prevalence Rates (as percentages of total jobs)
print("Creating benefit prevalence rates table...")

prevalence_rates_data = []
for benefit_name, col_name in benefits_mapping.items():
    if col_name in df.columns:
        ai_col = col_name + ' (AI)'
        non_ai_col = col_name + ' (Non-AI)'
        
        # Calculate prevalence rates as percentage of total jobs
        overall_rate = (df[col_name] / df['job_count'] * 100)
        ai_rate = (df[ai_col] / df['ai_role_count'] * 100) if ai_col in df.columns else None
        non_ai_rate = (df[non_ai_col] / (df['job_count'] - df['ai_role_count']) * 100) if non_ai_col in df.columns else None
        
        # Overall rates
        prevalence_rates_data.append({
            'Benefit': benefit_name,
            'Population': 'Overall',
            'Mean Rate (%)': f"{overall_rate.mean():.2f}",
            'Median Rate (%)': f"{overall_rate.median():.2f}",
            'Min Rate (%)': f"{overall_rate.min():.2f}",
            'Max Rate (%)': f"{overall_rate.max():.2f}",
            'Std Dev (%)': f"{overall_rate.std():.2f}"
        })
        
        # AI rates
        if ai_rate is not None:
            # Filter out infinite/NaN values that occur when ai_role_count is 0
            ai_rate_clean = ai_rate[ai_rate.notna() & (ai_rate != float('inf'))]
            if len(ai_rate_clean) > 0:
                prevalence_rates_data.append({
                    'Benefit': benefit_name,
                    'Population': 'AI Roles',
                    'Mean Rate (%)': f"{ai_rate_clean.mean():.2f}",
                    'Median Rate (%)': f"{ai_rate_clean.median():.2f}",
                    'Min Rate (%)': f"{ai_rate_clean.min():.2f}",
                    'Max Rate (%)': f"{ai_rate_clean.max():.2f}",
                    'Std Dev (%)': f"{ai_rate_clean.std():.2f}"
                })
        
        # Non-AI rates
        if non_ai_rate is not None:
            prevalence_rates_data.append({
                'Benefit': benefit_name,
                'Population': 'Non-AI Roles',
                'Mean Rate (%)': f"{non_ai_rate.mean():.2f}",
                'Median Rate (%)': f"{non_ai_rate.median():.2f}",
                'Min Rate (%)': f"{non_ai_rate.min():.2f}",
                'Max Rate (%)': f"{non_ai_rate.max():.2f}",
                'Std Dev (%)': f"{non_ai_rate.std():.2f}"
            })

prevalence_rates_table = pd.DataFrame(prevalence_rates_data)
prevalence_rates_table.to_csv('exports/tables/summary_statistics/occupation-years/benefit_prevalence_rates.csv', index=False)
print("✓ Benefit prevalence rates table exported")

# 2. Median Salary by Benefit Availability (2024 data)
print("Creating median salary by benefit availability table...")

df_2024 = df[df['YEAR'] == 2024].copy()
salary_by_benefit_data = []

for benefit_name, col_name in benefits_mapping.items():
    if col_name in df.columns:
        ai_col = col_name + ' (AI)'
        
        # Calculate benefit rate for each occupation
        df_2024[f'{benefit_name}_rate_overall'] = df_2024[col_name] / df_2024['job_count'] * 100
        df_2024[f'{benefit_name}_rate_ai'] = df_2024[ai_col] / df_2024['ai_role_count'] * 100
        
        # Create quartiles based on overall benefit prevalence rate
        df_2024[f'{benefit_name}_quartile'] = pd.qcut(df_2024[f'{benefit_name}_rate_overall'], 
                                                     q=4, labels=['Low', 'Med-Low', 'Med-High', 'High'])
        
        # Calculate median salaries by quartile
        for quartile in ['Low', 'Med-Low', 'Med-High', 'High']:
            quartile_data = df_2024[df_2024[f'{benefit_name}_quartile'] == quartile]
            if len(quartile_data) > 0:
                salary_by_benefit_data.append({
                    'Benefit': benefit_name,
                    'Prevalence Quartile': quartile,
                    'N Occupations': len(quartile_data),
                    'Median AI Salary ($)': f"{quartile_data['MEDIAN_SALARY_ai'].median():.0f}",
                    'Median Non-AI Salary ($)': f"{quartile_data['MEDIAN_SALARY_non_ai'].median():.0f}",
                    'Avg Benefit Rate (%)': f"{quartile_data[f'{benefit_name}_rate_overall'].mean():.2f}",
                    'Avg AI Benefit Rate (%)': f"{quartile_data[f'{benefit_name}_rate_ai'].mean():.2f}"
                })

salary_by_benefit_table = pd.DataFrame(salary_by_benefit_data)
salary_by_benefit_table.to_csv('exports/tables/summary_statistics/occupation-years/salary_by_benefit_availability.csv', index=False)
print("✓ Salary by benefit availability table exported")

# 3. Benefit Prevalence by Salary Quartiles (2024)
print("Creating benefit prevalence by salary quartiles table...")

# Create salary quartiles
df_2024['ai_salary_quartile'] = pd.qcut(df_2024['MEDIAN_SALARY_ai'], 
                                       q=4, labels=['Low Salary', 'Med-Low Salary', 'Med-High Salary', 'High Salary'])

benefit_by_salary_data = []
for quartile in ['Low Salary', 'Med-Low Salary', 'Med-High Salary', 'High Salary']:
    quartile_data = df_2024[df_2024['ai_salary_quartile'] == quartile]
    if len(quartile_data) > 0:
        row_data = {
            'Salary Quartile': quartile,
            'N Occupations': len(quartile_data),
            'Median AI Salary ($)': f"{quartile_data['MEDIAN_SALARY_ai'].median():.0f}",
            'Salary Range ($)': f"{quartile_data['MEDIAN_SALARY_ai'].min():.0f} - {quartile_data['MEDIAN_SALARY_ai'].max():.0f}"
        }
        
        # Add benefit rates for this salary quartile
        for benefit_name, col_name in benefits_mapping.items():
            if col_name in df.columns:
                ai_col = col_name + ' (AI)'
                benefit_rate = (quartile_data[ai_col] / quartile_data['ai_role_count'] * 100).mean()
                row_data[f'{benefit_name} Rate (%)'] = f"{benefit_rate:.2f}"
        
        benefit_by_salary_data.append(row_data)

benefit_by_salary_table = pd.DataFrame(benefit_by_salary_data)
benefit_by_salary_table.to_csv('exports/tables/summary_statistics/occupation-years/benefits_by_salary_quartile.csv', index=False)
print("✓ Benefits by salary quartile table exported")

# 4. Year-over-Year Benefit Prevalence Trends
print("Creating year-over-year benefit trends table...")

yearly_benefit_trends = []
for year in sorted(df['YEAR'].unique()):
    year_data = df[df['YEAR'] == year]
    row_data = {'Year': year}
    
    for benefit_name, col_name in benefits_mapping.items():
        if col_name in df.columns:
            ai_col = col_name + ' (AI)'
            
            # Overall prevalence rate
            overall_rate = (year_data[col_name] / year_data['job_count'] * 100).mean()
            
            # AI prevalence rate (filter out inf/nan)
            ai_rate_series = year_data[ai_col] / year_data['ai_role_count'] * 100
            ai_rate_clean = ai_rate_series[ai_rate_series.notna() & (ai_rate_series != float('inf'))]
            ai_rate = ai_rate_clean.mean() if len(ai_rate_clean) > 0 else 0
            
            row_data[f'{benefit_name} Overall (%)'] = f"{overall_rate:.2f}"
            row_data[f'{benefit_name} AI (%)'] = f"{ai_rate:.2f}"
    
    yearly_benefit_trends.append(row_data)

yearly_benefit_trends_table = pd.DataFrame(yearly_benefit_trends)
yearly_benefit_trends_table.to_csv('exports/tables/summary_statistics/occupation-years/yearly_benefit_trends.csv', index=False)
print("✓ Yearly benefit trends table exported")

# 5. Correlation between Benefits and Salaries (2024)
print("Creating benefit-salary correlation table...")

correlation_data = []
df_2024_clean = df_2024.copy()

for benefit_name, col_name in benefits_mapping.items():
    if col_name in df.columns:
        ai_col = col_name + ' (AI)'
        
        # Calculate benefit rates
        overall_rate = df_2024_clean[col_name] / df_2024_clean['job_count'] * 100
        ai_rate_series = df_2024_clean[ai_col] / df_2024_clean['ai_role_count'] * 100
        ai_rate = ai_rate_series.fillna(0).replace([float('inf'), -float('inf')], 0)
        
        # Correlations with salaries
        corr_overall_ai_salary = overall_rate.corr(df_2024_clean['MEDIAN_SALARY_ai'])
        corr_overall_nonai_salary = overall_rate.corr(df_2024_clean['MEDIAN_SALARY_non_ai'])
        corr_ai_rate_ai_salary = ai_rate.corr(df_2024_clean['MEDIAN_SALARY_ai'])
        
        correlation_data.append({
            'Benefit': benefit_name,
            'Overall Rate vs AI Salary': f"{corr_overall_ai_salary:.3f}",
            'Overall Rate vs Non-AI Salary': f"{corr_overall_nonai_salary:.3f}",
            'AI Rate vs AI Salary': f"{corr_ai_rate_ai_salary:.3f}",
            'Mean Overall Rate (%)': f"{overall_rate.mean():.2f}",
            'Mean AI Rate (%)': f"{ai_rate.mean():.2f}"
        })

correlation_table = pd.DataFrame(correlation_data)
correlation_table.to_csv('exports/tables/summary_statistics/occupation-years/benefit_salary_correlations.csv', index=False)
print("✓ Benefit-salary correlation table exported")

print(f"\nAll benefit analysis tables exported to: exports/tables/summary_statistics/occupation-years/")
print("New files created:")
for file in [
    'benefit_prevalence_rates.csv',
    'salary_by_benefit_availability.csv',
    'benefits_by_salary_quartile.csv', 
    'yearly_benefit_trends.csv',
    'benefit_salary_correlations.csv'
]:
    print(f"  - {file}")