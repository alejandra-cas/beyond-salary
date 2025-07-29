import pandas as pd
import numpy as np
import os

# Read the data
df = pd.read_csv('exports/occ_year_data/occ_year_analysis_2024_raw.csv')

# Create directory
os.makedirs('exports/tables/summary_statistics/occupation-years', exist_ok=True)

print("Creating occupation-year level summary statistics tables...")

# Benefits mapping
benefits_mapping = {
    'Tuition Assistance': 'Prevalence: Tuition Assistance',
    'Paid Leave': 'Prevalence: Paid Leave',
    'Health & Wellbeing': 'Prevalence: Health and Wellbeing', 
    'Parental Leave': 'Prevalence: Parental Leave',
    'Workplace Culture': 'Prevalence: Workplace Culture',
    'Remote Work': 'Prevalence: Remote Work'
}

# 1. Occupation-Year Sample Characteristics
print("Creating occupation-year sample characteristics table...")

sample_chars_data = []

# Basic sample info
sample_chars_data.append({
    'Characteristic': 'Total Occupation-Year Observations',
    'Value': f"{len(df)}"
})

sample_chars_data.append({
    'Characteristic': 'Number of Unique Occupations', 
    'Value': f"{df['SOC_2021_2_NAME'].nunique()}"
})

sample_chars_data.append({
    'Characteristic': 'Time Period',
    'Value': f"{df['YEAR'].min()}-{df['YEAR'].max()}"
})

sample_chars_data.append({
    'Characteristic': 'Years per Occupation (Average)',
    'Value': f"{df.groupby('SOC_2021_2_NAME')['YEAR'].count().mean():.2f}"
})

sample_chars_data.append({
    'Characteristic': 'Occupation-Years per Year (Average)',
    'Value': f"{df.groupby('YEAR')['SOC_2021_2_NAME'].count().mean():.2f}"
})

# Distribution by year
year_counts = df['YEAR'].value_counts().sort_index()
for year, count in year_counts.items():
    sample_chars_data.append({
        'Characteristic': f'Observations in {year}',
        'Value': f"{count}"
    })

sample_chars_table = pd.DataFrame(sample_chars_data)
sample_chars_table.to_csv('exports/tables/summary_statistics/occupation-years/occupation_year_sample_characteristics.csv', index=False)
print("✓ Occupation-year sample characteristics table created")

# 2. Benefit Prevalence Rate Distributions (Occupation-Year Level)
print("Creating benefit prevalence rate distributions table...")

# Calculate prevalence rates for each benefit
prevalence_distributions_data = []

for benefit_name, col_name in benefits_mapping.items():
    if col_name in df.columns:
        ai_col = col_name + ' (AI)'
        non_ai_col = col_name + ' (Non-AI)'
        
        # Overall prevalence rates (as % of total jobs)
        overall_rates = (df[col_name] / df['job_count'] * 100)
        
        prevalence_distributions_data.append({
            'Benefit': benefit_name,
            'Population': 'Overall',
            'Mean (%)': f"{overall_rates.mean():.2f}",
            'Median (%)': f"{overall_rates.median():.2f}",
            'Min (%)': f"{overall_rates.min():.2f}",
            'Max (%)': f"{overall_rates.max():.2f}",
            'Std Dev (%)': f"{overall_rates.std():.2f}",
            'Q25 (%)': f"{overall_rates.quantile(0.25):.2f}",
            'Q75 (%)': f"{overall_rates.quantile(0.75):.2f}"
        })
        
        # AI prevalence rates
        if ai_col in df.columns:
            ai_rates_series = df[ai_col] / df['ai_role_count'] * 100
            ai_rates_clean = ai_rates_series[ai_rates_series.notna() & (ai_rates_series != float('inf'))]
            
            if len(ai_rates_clean) > 0:
                prevalence_distributions_data.append({
                    'Benefit': benefit_name,
                    'Population': 'AI Roles',
                    'Mean (%)': f"{ai_rates_clean.mean():.2f}",
                    'Median (%)': f"{ai_rates_clean.median():.2f}",
                    'Min (%)': f"{ai_rates_clean.min():.2f}",
                    'Max (%)': f"{ai_rates_clean.max():.2f}",
                    'Std Dev (%)': f"{ai_rates_clean.std():.2f}",
                    'Q25 (%)': f"{ai_rates_clean.quantile(0.25):.2f}",
                    'Q75 (%)': f"{ai_rates_clean.quantile(0.75):.2f}"
                })
        
        # Non-AI prevalence rates
        if non_ai_col in df.columns:
            non_ai_rates = df[non_ai_col] / (df['job_count'] - df['ai_role_count']) * 100
            
            prevalence_distributions_data.append({
                'Benefit': benefit_name,
                'Population': 'Non-AI Roles',
                'Mean (%)': f"{non_ai_rates.mean():.2f}",
                'Median (%)': f"{non_ai_rates.median():.2f}",
                'Min (%)': f"{non_ai_rates.min():.2f}",
                'Max (%)': f"{non_ai_rates.max():.2f}",
                'Std Dev (%)': f"{non_ai_rates.std():.2f}",
                'Q25 (%)': f"{non_ai_rates.quantile(0.25):.2f}",
                'Q75 (%)': f"{non_ai_rates.quantile(0.75):.2f}"
            })

prevalence_distributions_table = pd.DataFrame(prevalence_distributions_data)
prevalence_distributions_table.to_csv('exports/tables/summary_statistics/occupation-years/benefit_prevalence_rate_distributions.csv', index=False)
print("✓ Benefit prevalence rate distributions table created")

# 3. AI Demand Distribution Across Occupation-Years
print("Creating AI demand distribution table...")

ai_demand_dist_data = []

# Overall AI demand statistics
ai_demand_dist_data.append({
    'Statistic': 'Mean AI Demand (%)',
    'Value': f"{df['AI Demand'].mean():.2f}"
})

ai_demand_dist_data.append({
    'Statistic': 'Median AI Demand (%)',
    'Value': f"{df['AI Demand'].median():.2f}"
})

ai_demand_dist_data.append({
    'Statistic': 'Min AI Demand (%)',
    'Value': f"{df['AI Demand'].min():.2f}"
})

ai_demand_dist_data.append({
    'Statistic': 'Max AI Demand (%)',
    'Value': f"{df['AI Demand'].max():.2f}"
})

ai_demand_dist_data.append({
    'Statistic': 'Standard Deviation (%)',
    'Value': f"{df['AI Demand'].std():.2f}"
})

# Percentiles
percentiles = [10, 25, 50, 75, 90, 95, 99]
for p in percentiles:
    ai_demand_dist_data.append({
        'Statistic': f'{p}th Percentile (%)',
        'Value': f"{df['AI Demand'].quantile(p/100):.2f}"
    })

# Coefficient of variation
cv = df['AI Demand'].std() / df['AI Demand'].mean()
ai_demand_dist_data.append({
    'Statistic': 'Coefficient of Variation',
    'Value': f"{cv:.2f}"
})

ai_demand_dist_table = pd.DataFrame(ai_demand_dist_data)
ai_demand_dist_table.to_csv('exports/tables/summary_statistics/occupation-years/ai_demand_distribution.csv', index=False)
print("✓ AI demand distribution table created")

# 4. Temporal and Cross-Sectional Variation Analysis
print("Creating temporal variation analysis table...")

variation_data = []

# Within-occupation variation over time (average std dev across occupations)
within_occ_variation = df.groupby('SOC_2021_2_NAME')['AI Demand'].std().mean()
variation_data.append({
    'Variation Type': 'Within-Occupation Variation (AI Demand %)',
    'Average Std Dev': f"{within_occ_variation:.2f}",
    'Description': 'Average standard deviation of AI demand within occupations over time'
})

# Between-occupation variation within years (average std dev across years)
between_occ_variation = df.groupby('YEAR')['AI Demand'].std().mean()
variation_data.append({
    'Variation Type': 'Between-Occupation Variation (AI Demand %)',
    'Average Std Dev': f"{between_occ_variation:.2f}",
    'Description': 'Average standard deviation of AI demand between occupations within years'
})

# Year-to-year changes in AI demand
df_sorted = df.sort_values(['SOC_2021_2_NAME', 'YEAR'])
df_sorted['AI_Demand_Change'] = df_sorted.groupby('SOC_2021_2_NAME')['AI Demand'].diff()
avg_year_change = df_sorted['AI_Demand_Change'].mean()
variation_data.append({
    'Variation Type': 'Average Year-to-Year Change (AI Demand %)',
    'Average Std Dev': f"{avg_year_change:.2f}",
    'Description': 'Average change in AI demand from year to year within occupations'
})

# For each benefit, calculate temporal variation
for benefit_name, col_name in benefits_mapping.items():
    if col_name in df.columns:
        overall_rates = df[col_name] / df['job_count'] * 100
        df[f'{benefit_name}_rate'] = overall_rates
        
        within_occ_benefit_var = df.groupby('SOC_2021_2_NAME')[f'{benefit_name}_rate'].std().mean()
        variation_data.append({
            'Variation Type': f'Within-Occupation Variation ({benefit_name} %)',
            'Average Std Dev': f"{within_occ_benefit_var:.2f}",
            'Description': f'Average std dev of {benefit_name} prevalence within occupations over time'
        })

variation_table = pd.DataFrame(variation_data)
variation_table.to_csv('exports/tables/summary_statistics/occupation-years/temporal_variation_analysis.csv', index=False)
print("✓ Temporal variation analysis table created")

# 5. Occupation-Year Correlations Matrix
print("Creating occupation-year correlations matrix...")

# Create dataset with rates for correlation analysis
corr_data = df.copy()
corr_data['AI_Demand_Pct'] = corr_data['AI Demand']

# Calculate benefit rates
for benefit_name, col_name in benefits_mapping.items():
    if col_name in df.columns:
        ai_col = col_name + ' (AI)'
        
        # Overall rate
        corr_data[f'{benefit_name}_Overall_Rate'] = corr_data[col_name] / corr_data['job_count'] * 100
        
        # AI rate (handle division by zero)
        ai_rate_series = corr_data[ai_col] / corr_data['ai_role_count'] * 100
        corr_data[f'{benefit_name}_AI_Rate'] = ai_rate_series.fillna(0).replace([float('inf'), -float('inf')], 0)

# Select variables for correlation matrix
corr_vars = ['AI_Demand_Pct'] + [f'{benefit}_Overall_Rate' for benefit in benefits_mapping.keys()] + [f'{benefit}_AI_Rate' for benefit in benefits_mapping.keys()]
corr_matrix = corr_data[corr_vars].corr()

# Round to 2 decimal places
corr_matrix = corr_matrix.round(2)

# Save correlation matrix
corr_matrix.to_csv('exports/tables/summary_statistics/occupation-years/occupation_year_correlations_matrix.csv')
print("✓ Occupation-year correlations matrix created")

# 6. Top/Bottom Occupation-Years by AI Demand and Benefits
print("Creating top/bottom occupation-years analysis...")

# Top 10 occupation-years by AI demand
top_ai_demand = df.nlargest(10, 'AI Demand')[['SOC_2021_2_NAME', 'YEAR', 'AI Demand', 'ai_role_count', 'job_count']].copy()
top_ai_demand['AI Demand'] = top_ai_demand['AI Demand'].round(2)
top_ai_demand.to_csv('exports/tables/summary_statistics/occupation-years/top_occupation_years_ai_demand.csv', index=False)

# Bottom 10 occupation-years by AI demand (excluding zeros)
bottom_ai_demand = df[df['AI Demand'] > 0].nsmallest(10, 'AI Demand')[['SOC_2021_2_NAME', 'YEAR', 'AI Demand', 'ai_role_count', 'job_count']].copy()
bottom_ai_demand['AI Demand'] = bottom_ai_demand['AI Demand'].round(2)
bottom_ai_demand.to_csv('exports/tables/summary_statistics/occupation-years/bottom_occupation_years_ai_demand.csv', index=False)

print("✓ Top/bottom occupation-years analysis created")

print(f"\n🎉 All occupation-year summary tables created!")
print("📁 Tables saved to: exports/tables/summary_statistics/occupation-years/")
print("\n📋 New files created:")
files_created = [
    'occupation_year_sample_characteristics.csv',
    'benefit_prevalence_rate_distributions.csv', 
    'ai_demand_distribution.csv',
    'temporal_variation_analysis.csv',
    'occupation_year_correlations_matrix.csv',
    'top_occupation_years_ai_demand.csv',
    'bottom_occupation_years_ai_demand.csv'
]

for f in files_created:
    print(f"   - {f}")

print(f"\n📊 Total: {len(files_created)} new summary tables created")
print("💡 All values rounded to 2 decimal places as requested")