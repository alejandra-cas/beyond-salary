import pandas as pd
import sys
import os

# Add parent directory to sys.path
parent_dir = os.path.abspath(os.path.join(os.getcwd(), os.pardir))
sys.path.append(parent_dir)
from package_files.benefits_defns import *

path = '../../data/salary_sample_body_benefits_v3.parquet.gzip'
if path[-3:] == 'csv:':
    even_sample = pd.read_csv(path)
elif path[-3:] == 'zip':
    even_sample = pd.read_parquet(path)

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

even_sample_select = even_sample[even_sample[occupation].isin(occupations_select)]

# get % AI roles per occupation-year
ai_role_occupation = even_sample_select.groupby([occupation, 'YEAR'])['AI ROLE'].mean().reset_index()
ai_role_occupation.rename(columns={'AI ROLE':'AI ROLE %'}, inplace=True)

# # get % with benefit 
# occupation_benefits = even_sample_select.groupby([occupation, 'YEAR', 'AI ROLE'])[benefits3].mean().reset_index()

# benefits_ai = occupation_benefits[occupation_benefits['AI ROLE'] == 1]
# benefits_non_ai = occupation_benefits[occupation_benefits['AI ROLE'] == 0]

# merged_df = pd.merge(benefits_ai, benefits_non_ai, on=['SOC_2021_2_NAME', 'YEAR'], suffixes=('_ai', '_non_ai'))

# salary premium
occupation_salaries = even_sample_select.groupby([occupation, 'YEAR', 'AI ROLE'])[['LOG_SALARY', 'SALARY']].mean().reset_index()
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


results = pd.read_csv('../exports/models_occ_year_results.csv')
# add coefficients
coeff_df = results.merge(occ_year_df[['SOC_2021_2_NAME', 'YEAR','AI ROLE %',
       'AI ROLE % CHANGE', 'PRIOR YEAR % CHANGE', 'LOG_SALARY_ai', 'SALARY_ai',
       'LOG_SALARY_non_ai', 'SALARY_non_ai', 'SALARY_PREMIUM_LOG',
       'SALARY_PREMIUM', 'MEAN SALARY', 'MEAN LOG SALARY']], left_on=['Occupation', 'Year'], right_on = [occupation, 'YEAR'])

# % benefits by occ
occ_benefit_group = even_sample_select.groupby([occupation, 'AI ROLE',benefit]).size().reset_index(name='benefit_count')
occ_year_group = even_sample_select.groupby([occupation, 'YEAR']).size().reset_index(name='job_count')
occ_year_benefit_group = occ_year_group.merge(occ_benefit_group, on = ['SOC_2021_2_NAME','YEAR'], how = 'left')
occ_year_benefit_group[f'Percent with {benefit}'] = occ_year_benefit_group['benefit_count']/occ_year_benefit_group['count']
