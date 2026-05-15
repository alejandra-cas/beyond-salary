import pandas as pd
import numpy as np
import sys
import os
from pathlib import Path

# Add parent directory to sys.path
parent_dir = os.path.abspath(os.path.join(os.getcwd()) + "/src")
sys.path.append(parent_dir)
from package_files.benefits_defns import *

_base = Path(__file__).parent.parent / "data" / "processed"
path = _base / "labeled_v1.parquet"
print("reading data")
data = pd.read_parquet(path)
remote_kw = _base / "labeled_v2.parquet"
remote_df = pd.read_parquet(remote_kw)
data = data.merge(remote_df[['ID', 'REMOTE_KW']], on='ID', how='left')
print("data read")

COUNTY_COL = 'COUNTY'  # adjust if your county column has a different name


def run_analysis(data, group_col, output_name):
    print(f"\n=== Running analysis by {group_col} ===")
    data_select = data[data[group_col].notna()]
    data_select['LOG_SALARY'] = np.log(data_select['SALARY'])

    # get % AI roles per group-year
    ai_role_group = data_select.groupby([group_col, 'YEAR'])[['AI ROLE', 'SALARY']].mean().reset_index()
    ai_role_group.rename(columns={'AI ROLE':'AI ROLE %'}, inplace=True)
    ai_role_group.rename(columns={'SALARY':'MEAN SALARY'}, inplace=True)
    ai_role_group['AI ROLE %'] = ai_role_group['AI ROLE %']*100

    # # get % with benefit
    # group_benefits = data_select.groupby([group_col, 'YEAR', 'AI ROLE'])[benefits3].mean().reset_index()

    # benefits_ai = group_benefits[group_benefits['AI ROLE'] == 1]
    # benefits_non_ai = group_benefits[group_benefits['AI ROLE'] == 0]

    # merged_df = pd.merge(benefits_ai, benefits_non_ai, on=[group_col, 'YEAR'], suffixes=('_ai', '_non_ai'))

    # salary premium
    group_salaries = data_select.groupby([group_col, 'YEAR', 'AI ROLE'])[['LOG_SALARY', 'SALARY']].mean().reset_index()
    # get median
    group_salaries_median = data_select.groupby([group_col, 'YEAR', 'AI ROLE'])[['LOG_SALARY','SALARY']].median().reset_index()
    group_salaries_median.rename(columns={'SALARY':'MEDIAN_SALARY', 'LOG_SALARY': 'MEDIAN_LOG_SALARY'}, inplace=True)
    group_salaries = pd.merge(group_salaries, group_salaries_median, on=[group_col, 'YEAR', 'AI ROLE'])
    group_salaries_ai = group_salaries[group_salaries['AI ROLE'] == 1]
    group_salaries_non_ai = group_salaries[group_salaries['AI ROLE'] == 0]
    group_salaries = pd.merge(group_salaries_ai, group_salaries_non_ai, on=[group_col, 'YEAR'], suffixes=('_ai', '_non_ai'))
    group_salaries['SALARY_PREMIUM_LOG'] = ((group_salaries['LOG_SALARY_ai']/group_salaries['LOG_SALARY_non_ai'])*100)
    group_salaries['SALARY_PREMIUM'] = ((group_salaries['SALARY_ai']/group_salaries['SALARY_non_ai'])*100)

    # % change in demand
    ai_role_group = ai_role_group.sort_values(by=[group_col, 'YEAR'])
    ai_role_group['AI ROLE % CHANGE'] = ai_role_group.groupby(group_col)['AI ROLE %'].pct_change()
    ai_role_group['AI ROLE % CHANGE'] = ai_role_group['AI ROLE % CHANGE']*100

    # prior year % change
    ai_role_group['PRIOR YEAR % CHANGE'] = ai_role_group.groupby(group_col)['AI ROLE % CHANGE'].shift(1)
    group_year_df = ai_role_group.merge(group_salaries, on=[group_col, 'YEAR'])
    group_year_df.drop(columns=['AI ROLE_ai', 'AI ROLE_non_ai'], inplace=True)
    print("group_year_df columns")
    # print(group_year_df.columns)

    # 2026-05-15: Coefficient merge from models_occ_year_results.csv was previously
    # done here but was commented out. Renamed coeff_df -> analysis_df for clarity.
    # print("reading results")
    # results = pd.read_csv('../exports/models_occ_year_results.csv')
    # analysis_df = results.merge(group_year_df, left_on=['Occupation', 'Year'], right_on = [group_col, 'YEAR'])
    # [group_col, 'YEAR','AI ROLE %',
    #        'AI ROLE % CHANGE', 'PRIOR YEAR % CHANGE', 'LOG_SALARY_ai', 'SALARY_ai',
    #        'LOG_SALARY_non_ai', 'SALARY_non_ai', 'SALARY_PREMIUM_LOG',
    #        'SALARY_PREMIUM', 'DURATION_CALC_ai', 'DURATION_CALC_non_ai', 'MEAN SALARY']
    analysis_df = group_year_df.copy()

    # % benefits by group
    # group_year_agg = data_select.groupby([group_col, 'YEAR']).size().reset_index(name='job_count')
    group_year_agg = data_select.groupby([group_col, 'YEAR']).agg(
        job_count=('ID', 'size'),   # Counts the number of rows (jobs) in each group
        ai_role_count=('AI ROLE', 'sum') # Counts the number of True values in AI ROLE
    ).reset_index()
    group_year_agg['AI Job Count Change'] = group_year_agg.groupby(group_col)['ai_role_count'].diff()
    # MBB added for county filter
    if group_col == COUNTY_COL:
        group_year_agg = group_year_agg[group_year_agg["job_count"] > 1000]

    for benefit in benefits4:
        print(benefit)
        label = benefits_labels_map[benefit]
        benefit_group = data_select.groupby([group_col,'YEAR'])[benefit].sum().reset_index(name=f'Prevalence: {label}')
        # benefit_group[f'Prevalence: {label}'] = benefit_group[f'Prevalence: {label}']*100
        print("merging group_year_agg and benefit_group")
        group_year_agg = group_year_agg.merge(benefit_group, on = [group_col,'YEAR'], how = 'left')
        print("group_year_agg")
        # print(group_year_agg.columns)

        benefit_role = data_select.groupby([group_col,'YEAR', 'AI ROLE'])[benefit].sum().reset_index(name=f'Prevalence: {label}')
        # benefit_role[f'Prevalence: {label}'] = benefit_role[f'Prevalence: {label}']*100
        print("benefit_role")
        # print(benefit_role.columns)
        benefit_role_ai = benefit_role[benefit_role['AI ROLE'] == 1]
        benefit_role_non_ai = benefit_role[benefit_role['AI ROLE'] == 0]
        print("merging benefit roles")
        benefit_role_all = pd.merge(benefit_role_ai, benefit_role_non_ai, on=[group_col, 'YEAR'], suffixes=(' (AI)', ' (Non-AI)'))
        print("benefit_role_all")
        # print(benefit_role_all.columns)
        print("merging")
        benefit_role_all.drop(columns=['AI ROLE (AI)', 'AI ROLE (Non-AI)'], inplace=True)
        group_year_agg = group_year_agg.merge(benefit_role_all, on = [group_col,'YEAR'], how = 'left')

    analysis_df = analysis_df.merge(group_year_agg, on = [group_col,'YEAR'])
    analysis_df['Log Job Count'] = analysis_df['job_count'].apply(lambda x: np.log(x))
    # group_year_df = group_year_df.merge(mean_salary, on=[group_col, 'YEAR'])
    # pd.set_option('display.max_rows', 100)
    # group_year_df.rename(columns={'SALARY':'MEAN SALARY'}, inplace=True)
    # group_year_df.drop(columns='SALARY', inplace=True)
    # group_year_df.rename(columns={'LOG_SALARY':'MEAN LOG SALARY'}, inplace=True)

    ai_jobs = data[data['AI ROLE'] == 1]
    non_ai_jobs = data[data['AI ROLE'] == 0]

    # Group by group_col and YEAR and sum benefits for AI and non-AI jobs
    ai_counts = ai_jobs.groupby([group_col, 'YEAR'])[benefits4].mean().reset_index()
    non_ai_counts = non_ai_jobs.groupby([group_col, 'YEAR'])[benefits4].mean().reset_index()

    # Merge the counts for AI and non-AI jobs
    merged_counts = ai_counts.merge(non_ai_counts, on=[group_col, 'YEAR'], suffixes=('_ai', '_non_ai'))

    # Calculate the difference between AI and non-AI jobs for each benefit
    for benefit in benefits4:
        merged_counts[f'{benefit}_difference'] = (merged_counts[f'{benefit}_ai'] - merged_counts[f'{benefit}_non_ai'])*100

    # Select only the columns with differences and group-year identifiers
    difference_summary = merged_counts[[group_col, 'YEAR'] + [f'{benefit}_difference' for benefit in benefits4]]
    analysis_df = analysis_df.merge(difference_summary, on=[group_col, 'YEAR'])

    analysis_df.rename(columns={'AI ROLE %': 'AI Demand'}, inplace=True)
    analysis_df.rename(columns={'SALARY_PREMIUM_LOG': 'Salary (Log) Premium'}, inplace=True)
    analysis_df.rename(columns={'AI ROLE % CHANGE': 'AI Demand % Change'}, inplace=True)
    analysis_df.rename(columns={'AI ROLE %': 'AI Demand'}, inplace=True)

    out_path = _base / f'{output_name}.parquet'
    print(f"exporting to {out_path}")
    print(analysis_df.columns)
    analysis_df.to_parquet(out_path, index=False)
    return analysis_df


# Run occupation-year analysis
occ_df = run_analysis(data, 'SOC_MAJOR_GROUP', 'occ_year_analysis_raw')

# Run industry-year analysis
ind_df = run_analysis(data, 'NAICS_2022_2_NAME', 'ind_year_analysis_raw')
