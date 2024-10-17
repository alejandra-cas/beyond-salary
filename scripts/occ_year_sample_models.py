# from logit_model import *
import statsmodels.api as sm
import matplotlib.pyplot as plt
from tqdm import tqdm
from statsmodels.stats.outliers_influence import variance_inflation_factor
import pandas as pd
import importlib 

import sys
import os

# Add parent directory to sys.path
parent_dir = os.path.abspath(os.path.join(os.getcwd(), os.pardir))
sys.path.append(parent_dir)

from package_files.benefits_defns import *
from package_files.logit_model import *

print(benefits4)

path = '../data/us_10m_nointernship_ai_skills_benefits.parquet.gzip'
if path[-3:] == 'csv:':
    df = pd.read_csv(path)
elif path[-3:] == 'zip':
    df = pd.read_parquet(path)
    
# occupations to analyze
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

# filter for occupations
df_select = df[df[occupation].isin(occupations_select)]

# get years
years = df['YEAR'].unique()

# 
def get_occ_year_data(df, occ, year):
    """
    Filter the DataFrame to include only rows with the specified occupation and year.

    Args:
        df (pd.DataFrame): The input DataFrame containing the data.
        occ (str): The occupation to filter by.
        year (int): The year to filter by.

    Returns:
        pd.DataFrame: A new DataFrame containing only the rows that match the specified occupation and year.
    """
    df_occ_year = df[(df[occupation] == occ) & (df['YEAR'] == year)].copy()
    return df_occ_year

models_dict = {}
for benefit in benefits4[6:]:
    print(benefit)
    for occ, year in tqdm([(occ, year) for occ in occupations_select for year in years]):
        print(occ, year)
        df_occ_year = get_occ_year_data(df_select, occ, year)
        # print length of df_occ_year
        print("df length:", len(df_occ_year))
        model = run_logit_model(df_occ_year, dependent = benefit, predictor = 'AI ROLE', cat_controls = [education, experience], ref_category = {education: "No Education Listed", experience: 'None Listed'})
        models_dict[(benefit, occ, year)] = model
        

import sys

# Estimate the size of the models dictionary in bytes
size_in_bytes = sys.getsizeof(models_dict)
size_in_megabytes = size_in_bytes / (1024 ** 2)  # Convert to MB
print(f"Size of models_dict: {size_in_megabytes:.2f} MB")

coefficients = []
errors = []
pvalues = []
benefits = []
occupations = []
yrs = []

keys_list = list(models_dict.keys())
for key, value in models_dict.items(): 
    model = value
    try:
        coef = model.params['AI ROLE']
        err = model.bse['AI ROLE']
        pvalue = model.pvalues['AI ROLE'].round(3)
    except:
        coef = None
        err = None
        pvalue = None
    benefit = key[0]
    occ = key[1]
    year = key[2]
    
    coefficients.append(coef)
    errors.append(err)
    pvalues.append(pvalue)
    benefits.append(benefit)
    occupations.append(occ)
    yrs.append(year)

# Creating DataFrame
results = {
    'Benefit': benefits,
    'Occupation': occupations,
    'Year': yrs,
    'AI Coefficient': coefficients,
    'Error': errors,
    'P-Value': pvalues,
}

results_df = pd.DataFrame(results)

print("exporting...")
# export results_df
results_df.to_csv('../exports/models_occ_year_results_remote.csv')

# export models_dict to pickle
import pickle
with open('../exports/models_occ_year_dict_remote.pickle', 'wb') as f:
    pickle.dump(models_dict, f)