import pandas as pd
import numpy as np

# Read the data
df = pd.read_csv('exports/occ_year_data/occ_year_analysis_2024_raw.csv')

print("=== OCCUPATION-YEAR SUMMARY STATISTICS (RAW NUMBERS) ===\n")
print(f"Dataset dimensions: {df.shape[0]} observations, {df.shape[1]} variables")
print(f"Time period: {df['YEAR'].min()}-{df['YEAR'].max()}")
print(f"Number of occupations: {df['SOC_2021_2_NAME'].nunique()}")
print(f"Years per occupation: {df.groupby('SOC_2021_2_NAME')['YEAR'].count().mean():.1f} (average)")

print("\n=== KEY VARIABLES SUMMARY ===")

# AI Demand metrics
print("\n--- AI DEMAND ---")
print(f"AI Demand (% of jobs):")
print(f"  Mean: {df['AI Demand'].mean():.2f}%")
print(f"  Median: {df['AI Demand'].median():.2f}%") 
print(f"  Min: {df['AI Demand'].min():.2f}%")
print(f"  Max: {df['AI Demand'].max():.2f}%")
print(f"  Std Dev: {df['AI Demand'].std():.2f}%")

print(f"\nAI Role Count:")
print(f"  Mean: {df['ai_role_count'].mean():.0f}")
print(f"  Median: {df['ai_role_count'].median():.0f}")
print(f"  Min: {df['ai_role_count'].min():.0f}")
print(f"  Max: {df['ai_role_count'].max():.0f}")
print(f"  Total AI roles across all occ-years: {df['ai_role_count'].sum():,}")

# Job counts
print("\n--- JOB COUNTS ---")
print(f"Total Job Count:")
print(f"  Mean: {df['job_count'].mean():.0f}")
print(f"  Median: {df['job_count'].median():.0f}")
print(f"  Min: {df['job_count'].min():.0f}")
print(f"  Max: {df['job_count'].max():.0f}")
print(f"  Total jobs across all occ-years: {df['job_count'].sum():,}")

# Salary metrics
print("\n--- SALARY METRICS ---")
print(f"Mean Salary (All Jobs):")
print(f"  Mean: ${df['MEAN SALARY'].mean():.0f}")
print(f"  Median: ${df['MEAN SALARY'].median():.0f}")
print(f"  Min: ${df['MEAN SALARY'].min():.0f}")
print(f"  Max: ${df['MEAN SALARY'].max():.0f}")

print(f"\nMedian Salary (AI Jobs):")
print(f"  Mean: ${df['MEDIAN_SALARY_ai'].mean():.0f}")
print(f"  Median: ${df['MEDIAN_SALARY_ai'].median():.0f}")
print(f"  Min: ${df['MEDIAN_SALARY_ai'].min():.0f}")
print(f"  Max: ${df['MEDIAN_SALARY_ai'].max():.0f}")

print(f"\nSalary Premium (AI vs Non-AI):")
print(f"  Mean: {df['SALARY_PREMIUM'].mean():.1f}%")
print(f"  Median: {df['SALARY_PREMIUM'].median():.1f}%")
print(f"  Min: {df['SALARY_PREMIUM'].min():.1f}%")
print(f"  Max: {df['SALARY_PREMIUM'].max():.1f}%")

# Benefits prevalence summary
print("\n=== BENEFITS PREVALENCE SUMMARY ===")

benefits_mapping = {
    'Tuition Assistance': 'Prevalence: Tuition Assistance',
    'Paid Leave': 'Prevalence: Paid Leave', 
    'Health & Wellbeing': 'Prevalence: Health and Wellbeing',
    'Parental Leave': 'Prevalence: Parental Leave',
    'Workplace Culture': 'Prevalence: Workplace Culture',
    'Remote Work': 'Prevalence: Remote Work'
}

for benefit_name, col_name in benefits_mapping.items():
    if col_name in df.columns:
        ai_col = col_name + ' (AI)'
        non_ai_col = col_name + ' (Non-AI)'
        
        print(f"\n--- {benefit_name.upper()} ---")
        print(f"Overall prevalence:")
        print(f"  Mean: {df[col_name].mean():.0f}")
        print(f"  Median: {df[col_name].median():.0f}")
        print(f"  Max: {df[col_name].max():.0f}")
        
        if ai_col in df.columns:
            print(f"AI roles:")
            print(f"  Mean: {df[ai_col].mean():.0f}")
            print(f"  Median: {df[ai_col].median():.0f}")
            print(f"  Max: {df[ai_col].max():.0f}")
            
        if non_ai_col in df.columns:
            print(f"Non-AI roles:")
            print(f"  Mean: {df[non_ai_col].mean():.0f}")
            print(f"  Median: {df[non_ai_col].median():.0f}")
            print(f"  Max: {df[non_ai_col].max():.0f}")

# Top occupations by AI demand
print("\n=== TOP OCCUPATIONS BY AI DEMAND (2024) ===")
df_2024 = df[df['YEAR'] == 2024].copy()
top_ai_demand = df_2024.nlargest(5, 'AI Demand')[['SOC_2021_2_NAME', 'AI Demand', 'ai_role_count', 'job_count']]
print(top_ai_demand.to_string(index=False))

print("\n=== HIGHEST AI JOB COUNTS (2024) ===") 
top_ai_count = df_2024.nlargest(5, 'ai_role_count')[['SOC_2021_2_NAME', 'ai_role_count', 'AI Demand', 'job_count']]
print(top_ai_count.to_string(index=False))

print("\n=== OCCUPATION-YEAR OBSERVATIONS BY YEAR ===")
year_counts = df['YEAR'].value_counts().sort_index()
for year, count in year_counts.items():
    print(f"{year}: {count} occupations")

print(f"\nTotal occupation-year observations: {len(df)}")