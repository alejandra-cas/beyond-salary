import pandas as pd
import numpy as np
import sys
import os

# Add parent directory to sys.path
parent_dir = os.path.abspath(os.path.join(os.getcwd(), os.pardir))
sys.path.append(parent_dir)
from package_files.benefits_defns import *

path = input("Please enter the input file path: ")
# path = '../data/us_10m_nointernship_ai_skills_benefits.parquet.gzip'
print("reading data")
if path[-3:] == 'csv:':
    data = pd.read_csv(path)
elif path[-3:] == 'zip':
    data = pd.read_parquet(path)
print("data read")
occupations_select = ['Architecture and Engineering Occupations','Arts, Design, Entertainment, Sports, and Media Occupations','Business and Financial Operations Occupations','Community and Social Service Occupations','Computer and Mathematical Occupations',
'Educational Instruction and Library Occupations',
'Healthcare Practitioners and Technical Occupations',
'Legal Occupations',
'Life, Physical, and Social Science Occupations',
'Management Occupations', 
'Office and Administrative Support Occupations',
'Personal Care and Service Occupations', 'Production Occupations',
'Sales and Related Occupations',
'Transportation and Material Moving Occupations',
]

data_select = data[data[occupation].isin(occupations_select)]
data_select['LOG_SALARY'] = np.log(data_select['SALARY'])
# get % AI roles per occupation-year
ai_role_occupation = data_select.groupby([occupation, 'YEAR'])[['AI ROLE', 'SALARY']].mean().reset_index()
ai_role_occupation.rename(columns={'AI ROLE':'AI ROLE %'}, inplace=True)
ai_role_occupation.rename(columns={'SALARY':'MEAN SALARY'}, inplace=True)
ai_role_occupation['AI ROLE %'] = ai_role_occupation['AI ROLE %']*100

# # get % with benefit 
# occupation_benefits = data_select.groupby([occupation, 'YEAR', 'AI ROLE'])[benefits3].mean().reset_index()

# benefits_ai = occupation_benefits[occupation_benefits['AI ROLE'] == 1]
# benefits_non_ai = occupation_benefits[occupation_benefits['AI ROLE'] == 0]

# merged_df = pd.merge(benefits_ai, benefits_non_ai, on=['SOC_2021_2_NAME', 'YEAR'], suffixes=('_ai', '_non_ai'))

# salary premium
occupation_salaries = data_select.groupby([occupation, 'YEAR', 'AI ROLE'])[['LOG_SALARY', 'SALARY']].mean().reset_index()
# get median
occupation_salaries_median = data_select.groupby([occupation, 'YEAR', 'AI ROLE'])[['LOG_SALARY','SALARY']].median().reset_index()
occupation_salaries_median.rename(columns={'SALARY':'MEDIAN_SALARY', 'LOG_SALARY': 'MEDIAN_LOG_SALARY'}, inplace=True)
occupation_salaries = pd.merge(occupation_salaries, occupation_salaries_median, on=[occupation, 'YEAR', 'AI ROLE'])
occupation_salaries_ai = occupation_salaries[occupation_salaries['AI ROLE'] == 1]
occupation_salaries_non_ai = occupation_salaries[occupation_salaries['AI ROLE'] == 0]
occupation_salaries = pd.merge(occupation_salaries_ai, occupation_salaries_non_ai, on=[occupation, 'YEAR'], suffixes=('_ai', '_non_ai'))
occupation_salaries['SALARY_PREMIUM_LOG'] = ((occupation_salaries['LOG_SALARY_ai']/occupation_salaries['LOG_SALARY_non_ai'])*100)
occupation_salaries['SALARY_PREMIUM'] = ((occupation_salaries['SALARY_ai']/occupation_salaries['SALARY_non_ai'])*100)

# % change in demand
ai_role_occupation = ai_role_occupation.sort_values(by=[occupation, 'YEAR'])
ai_role_occupation['AI ROLE % CHANGE'] = ai_role_occupation.groupby(occupation)['AI ROLE %'].pct_change()
ai_role_occupation['AI ROLE % CHANGE'] = ai_role_occupation['AI ROLE % CHANGE']*100

# prior year % change
ai_role_occupation['PRIOR YEAR % CHANGE'] = ai_role_occupation.groupby(occupation)['AI ROLE % CHANGE'].shift(1)
occ_year_df = ai_role_occupation.merge(occupation_salaries, on=[occupation, 'YEAR'])
occ_year_df.drop(columns=['AI ROLE_ai', 'AI ROLE_non_ai'], inplace=True)
print("occ_year_df columns")
# print(occ_year_df.columns)

# print("reading results")
# results = pd.read_csv('../exports/models_occ_year_results.csv')
coeff_df = occ_year_df.copy()
# add coefficients
# coeff_df = results.merge(occ_year_df, left_on=['Occupation', 'Year'], right_on = [occupation, 'YEAR'])
# ['SOC_2021_2_NAME', 'YEAR','AI ROLE %',
#        'AI ROLE % CHANGE', 'PRIOR YEAR % CHANGE', 'LOG_SALARY_ai', 'SALARY_ai',
#        'LOG_SALARY_non_ai', 'SALARY_non_ai', 'SALARY_PREMIUM_LOG',
#        'SALARY_PREMIUM', 'DURATION_CALC_ai', 'DURATION_CALC_non_ai', 'MEAN SALARY']

# % benefits by occ
# occ_year_group = data_select.groupby([occupation, 'YEAR']).size().reset_index(name='job_count')
occ_year_group = data_select.groupby([occupation, 'YEAR']).agg(
    job_count=('ID', 'size'),   # Counts the number of rows (jobs) in each group
    ai_role_count=('AI ROLE', 'sum') # Counts the number of True values in AI ROLE
).reset_index()

for benefit in benefits4:
    print(benefit)
    label = benefits_labels_map[benefit]
    occ_benefit_group = data_select.groupby([occupation,'YEAR'])[benefit].sum().reset_index(name=f'Prevalence: {label}')
    # occ_benefit_group[f'Prevalence: {label}'] = occ_benefit_group[f'Prevalence: {label}']*100
    print("merging occ_year_group and occ_benefit_group")
    occ_year_group = occ_year_group.merge(occ_benefit_group, on = ['SOC_2021_2_NAME','YEAR'], how = 'left')
    print("occ_year_group")
    # print(occ_year_group.columns)
    
    occ_benefit_role = data_select.groupby([occupation,'YEAR', 'AI ROLE'])[benefit].sum().reset_index(name=f'Prevalence: {label}')
    # occ_benefit_role[f'Prevalence: {label}'] = occ_benefit_role[f'Prevalence: {label}']*100
    print("occ_benefit_role")
    # print(occ_benefit_role.columns)
    occ_benefit_role_ai = occ_benefit_role[occ_benefit_role['AI ROLE'] == 1]
    occ_benefit_role_non_ai = occ_benefit_role[occ_benefit_role['AI ROLE'] == 0]
    print("merging occ benefit roles")
    occ_benefit_role_all = pd.merge(occ_benefit_role_ai, occ_benefit_role_non_ai, on=[occupation, 'YEAR'], suffixes=(' (AI)', ' (Non-AI)'))
    print("occ_benefit_role_all")
    # print(occ_benefit_role_all.columns)
    print("merging")
    occ_benefit_role_all.drop(columns=['AI ROLE (AI)', 'AI ROLE (Non-AI)'], inplace=True)
    occ_year_group = occ_year_group.merge(occ_benefit_role_all, on = ['SOC_2021_2_NAME','YEAR'], how = 'left')

coeff_df = coeff_df.merge(occ_year_group, on = ['SOC_2021_2_NAME','YEAR'])
coeff_df['Log Job Count'] = coeff_df['job_count'].apply(lambda x: np.log(x))
# occ_year_df = occ_year_df.merge(mean_salary, on=[occupation, 'YEAR'])
# pd.set_option('display.max_rows', 100)
# occ_year_df.rename(columns={'SALARY':'MEAN SALARY'}, inplace=True)
# occ_year_df.drop(columns='SALARY', inplace=True)
# occ_year_df.rename(columns={'LOG_SALARY':'MEAN LOG SALARY'}, inplace=True)

coeff_df.rename(columns={'AI ROLE %': 'AI Demand'}, inplace=True)
coeff_df.rename(columns={'SALARY_PREMIUM_LOG': 'Salary (Log) Premium'}, inplace=True)
coeff_df.rename(columns={'AI ROLE % CHANGE': 'AI Demand % Change'}, inplace=True)
coeff_df.rename(columns={'AI ROLE %': 'AI Demand'}, inplace=True)

print("exporting coeff_df")
print(coeff_df.columns)
coeff_df.to_csv('../exports/occ_year_data/occ_year_analysis_2024_raw.csv', index=False)