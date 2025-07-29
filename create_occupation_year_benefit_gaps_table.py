import pandas as pd
import numpy as np
import os

# Read the data
df = pd.read_csv('exports/occ_year_data/occ_year_analysis_2024_raw.csv')

# Create directory
os.makedirs('exports/tables/summary_statistics/occupation-years', exist_ok=True)

print("Creating occupation-year benefit gaps table...")

# Benefits mapping
benefits_mapping = {
    'Tuition Assistance': 'Prevalence: Tuition Assistance',
    'Paid Leave': 'Prevalence: Paid Leave',
    'Health & Wellbeing': 'Prevalence: Health and Wellbeing', 
    'Parental Leave': 'Prevalence: Parental Leave',
    'Workplace Culture': 'Prevalence: Workplace Culture',
    'Remote Work': 'Prevalence: Remote Work'
}

# Create the comprehensive table
occupation_year_gaps = []

for idx, row in df.iterrows():
    occupation = row['SOC_2021_2_NAME']
    year = row['YEAR']
    ai_demand = row['AI Demand']
    total_jobs = row['job_count']
    ai_jobs = row['ai_role_count']
    non_ai_jobs = total_jobs - ai_jobs
    
    # Base info for this occupation-year
    base_row = {
        'Occupation': occupation,
        'Year': year,
        'AI Demand (%)': round(ai_demand, 2),
        'Total Jobs': total_jobs,
        'AI Jobs': ai_jobs,
        'Non-AI Jobs': non_ai_jobs
    }
    
    # Add benefit prevalence data for each benefit
    for benefit_name, col_name in benefits_mapping.items():
        if col_name in df.columns:
            ai_col = col_name + ' (AI)'
            non_ai_col = col_name + ' (Non-AI)'
            
            # Raw counts
            overall_count = row[col_name]
            ai_count = row[ai_col] if ai_col in df.columns else 0
            non_ai_count = row[non_ai_col] if non_ai_col in df.columns else 0
            
            # Calculate prevalence rates (as percentages)
            overall_rate = (overall_count / total_jobs * 100) if total_jobs > 0 else 0
            ai_rate = (ai_count / ai_jobs * 100) if ai_jobs > 0 else 0
            non_ai_rate = (non_ai_count / non_ai_jobs * 100) if non_ai_jobs > 0 else 0
            
            # Calculate gap (AI rate - Non-AI rate)
            gap = ai_rate - non_ai_rate
            
            # Add to row
            base_row[f'{benefit_name} Overall (%)'] = round(overall_rate, 2)
            base_row[f'{benefit_name} AI (%)'] = round(ai_rate, 2)
            base_row[f'{benefit_name} Non-AI (%)'] = round(non_ai_rate, 2)
            base_row[f'{benefit_name} Gap (pp)'] = round(gap, 2)  # pp = percentage points
    
    occupation_year_gaps.append(base_row)

# Convert to DataFrame
gaps_df = pd.DataFrame(occupation_year_gaps)

# Sort by AI Demand descending, then by Year
gaps_df = gaps_df.sort_values(['AI Demand (%)', 'Year'], ascending=[False, True])

# Save the full table
gaps_df.to_csv('exports/tables/summary_statistics/occupation-years/occupation_year_benefit_gaps_full.csv', index=False)
print("✓ Full occupation-year benefit gaps table created")

# Create a summary version with just key metrics
summary_cols = ['Occupation', 'Year', 'AI Demand (%)', 'Total Jobs', 'AI Jobs']

# Add the most important benefits
key_benefits = ['Workplace Culture', 'Remote Work', 'Paid Leave', 'Parental Leave']
for benefit in key_benefits:
    summary_cols.extend([
        f'{benefit} AI (%)',
        f'{benefit} Non-AI (%)', 
        f'{benefit} Gap (pp)'
    ])

summary_df = gaps_df[summary_cols].copy()
summary_df.to_csv('exports/tables/summary_statistics/occupation-years/occupation_year_benefit_gaps_summary.csv', index=False)
print("✓ Summary occupation-year benefit gaps table created")

# Create a table focused on largest gaps
print("Creating largest gaps analysis...")

gap_analysis = []
for benefit_name in benefits_mapping.keys():
    gap_col = f'{benefit_name} Gap (pp)'
    if gap_col in gaps_df.columns:
        # Find occupation-years with largest positive and negative gaps
        largest_positive = gaps_df.nlargest(3, gap_col)[['Occupation', 'Year', 'AI Demand (%)', gap_col]].copy()
        largest_negative = gaps_df.nsmallest(3, gap_col)[['Occupation', 'Year', 'AI Demand (%)', gap_col]].copy()
        
        for idx, row in largest_positive.iterrows():
            gap_analysis.append({
                'Benefit': benefit_name,
                'Gap Type': 'Largest Pro-AI',
                'Occupation': row['Occupation'],
                'Year': row['Year'],
                'AI Demand (%)': row['AI Demand (%)'],
                'Gap (pp)': row[gap_col]
            })
        
        for idx, row in largest_negative.iterrows():
            gap_analysis.append({
                'Benefit': benefit_name,
                'Gap Type': 'Largest Pro-Non-AI',
                'Occupation': row['Occupation'], 
                'Year': row['Year'],
                'AI Demand (%)': row['AI Demand (%)'],
                'Gap (pp)': row[gap_col]
            })

gap_analysis_df = pd.DataFrame(gap_analysis)
gap_analysis_df.to_csv('exports/tables/summary_statistics/occupation-years/largest_benefit_gaps_analysis.csv', index=False)
print("✓ Largest gaps analysis table created")

# Create aggregate statistics by benefit
print("Creating gap statistics by benefit...")

gap_stats = []
for benefit_name in benefits_mapping.keys():
    gap_col = f'{benefit_name} Gap (pp)'
    ai_col = f'{benefit_name} AI (%)'
    non_ai_col = f'{benefit_name} Non-AI (%)'
    
    if gap_col in gaps_df.columns:
        gaps = gaps_df[gap_col]
        ai_rates = gaps_df[ai_col]
        non_ai_rates = gaps_df[non_ai_col]
        
        gap_stats.append({
            'Benefit': benefit_name,
            'Mean Gap (pp)': round(gaps.mean(), 2),
            'Median Gap (pp)': round(gaps.median(), 2),
            'Min Gap (pp)': round(gaps.min(), 2),
            'Max Gap (pp)': round(gaps.max(), 2),
            'Std Dev Gap (pp)': round(gaps.std(), 2),
            'Positive Gaps (N)': (gaps > 0).sum(),
            'Negative Gaps (N)': (gaps < 0).sum(),
            'Zero Gaps (N)': (gaps == 0).sum(),
            'Mean AI Rate (%)': round(ai_rates.mean(), 2),
            'Mean Non-AI Rate (%)': round(non_ai_rates.mean(), 2)
        })

gap_stats_df = pd.DataFrame(gap_stats)
gap_stats_df.to_csv('exports/tables/summary_statistics/occupation-years/benefit_gap_statistics.csv', index=False)
print("✓ Benefit gap statistics table created")

# Show some key findings
print(f"\n📊 KEY FINDINGS:")
print(f"📁 Total occupation-year observations: {len(gaps_df)}")

print(f"\n🎯 Average Gaps (AI % - Non-AI %):")
for benefit_name in benefits_mapping.keys():
    gap_col = f'{benefit_name} Gap (pp)'
    if gap_col in gaps_df.columns:
        mean_gap = gaps_df[gap_col].mean()
        pos_gaps = (gaps_df[gap_col] > 0).sum()
        total_obs = len(gaps_df)
        print(f"  {benefit_name}: {mean_gap:.2f} pp (Pro-AI in {pos_gaps}/{total_obs} cases)")

print(f"\n📄 Files created:")
files_created = [
    'occupation_year_benefit_gaps_full.csv - Complete data for all benefits',
    'occupation_year_benefit_gaps_summary.csv - Key benefits only',
    'largest_benefit_gaps_analysis.csv - Extreme cases analysis', 
    'benefit_gap_statistics.csv - Summary statistics by benefit'
]

for f in files_created:
    print(f"   - {f}")

print(f"\n💡 Use the 'full' table for comprehensive analysis")
print(f"💡 Use the 'summary' table for paper presentation")
print(f"💡 'Gap (pp)' = percentage point difference (AI rate - Non-AI rate)")