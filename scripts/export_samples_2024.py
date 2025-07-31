import pandas as pd
all_data = pd.read_parquet('../data/us_10m_nointernship_2018_2024_benefits.parquet.gzip')
all_data = all_data[all_data['QUARTER'] != '2024Q3']
# all_data_body = pd.read_parquet('../data/us_10m_nointernship_ai_skills_body.parquet.gzip')
# Exports

print("finished loading data")

## Random Sample with Salary

salary_data = all_data[all_data['SALARY'].notnull()]
# salary_wham_data = salary_data[salary_data['wfh_wham'].notnull()]
len(salary_data)
salary_ai = salary_data[salary_data['AI ROLE'] == True]
salary_no_ai = salary_data[salary_data['AI ROLE'] == False]
salary_ai_sample = salary_ai.sample(n=10000, random_state=42)
salary_no_ai_sample = salary_no_ai.sample(n=10000, random_state=42)
salary_sample_all = pd.concat([salary_ai_sample, salary_no_ai_sample])
# salary_sample_all = salary_sample_all.merge(body, left_on='ID', right_on='ID', how='left')
len(salary_sample_all)
salary_sample_all.to_parquet('data/small_samples/2024_salary_sample.parquet.gzip', compression='gzip')


## Random Sample without Salary
# all_data_wham = all_data[all_data['wfh_wham'].notnull()]
all_data_ai = all_data[all_data['AI ROLE'] == True]
all_data_no_ai = all_data[all_data['AI ROLE'] == False]
all_data_ai_sample = all_data_ai.sample(n=10000, random_state=42)
all_data_no_ai_sample = all_data_no_ai.sample(n=10000, random_state=42)
all_data_sample_all = pd.concat([all_data_ai_sample, all_data_no_ai_sample])
# all_data_sample_all = all_data_sample_all.merge(body, left_on='ID', right_on='ID', how='left')
all_data_sample_all.to_parquet('data/small_samples/2024_nosalary_sample.parquet.gzip', compression='gzip')
# print("saving just benefits all data")
# all_data.drop(columns=['BODY']).to_parquet('data/us_10m_nointernship_ai_skills_benefits.parquet.gzip', compression='gzip')
