# SkillScale Beyond Salary Project

# Pipeline
- merge_body.py
- label_benefits.py
- prepare_data.ipynb
- regression_models.ipynb

# Data
- salary_sample_body_benefits.parquet.gzip
    - 20k sample extracted from dataset of 10 million job postings, filtered for no internships and only jobs with salary information
- even_sample.csv
    - 20k sample extracted from original shared dataset of 1 million job postings, filtered for no internships
- us-003_noint.parquet
    - 1 million OJV sample filtered for no internships

# Scripts
- prepare_data.ipynb
    - loads original OJV data
    - loads data from thesis
    - merges data from thesis to get AI and WHAM job labels 
    - adds 'YEAR' column
    - adds any missing WHAM labels from WHAM data
    - exports random samples of 20k (with and without salary) and checks industry compositions

- classify_ai_skills.py
    - classifies jobs as AI or non-AI for any jobs with missing label after merge in prepare_data
    
- merge_body.py
    - merges body column to OJV data, exports OJV data

- label_benefits.py
    - Checks whether the BODY column of OJV data contains specified keywords for each benefit. Adds a column for each benefit to a dataframe indicating True or False. Imports and exports using given path

- prepare_data_2.ipynb
    - imports OJV dataset 
    - renames 'Has AI Skills' column to AI ROLE
    - adds EXPERIENCE_BUCKET column
    - joins salary data
    - adds LOG_SALARY column
    - fills in missing AI labels
    - exports data

- export_samples.py
    - export samples of 20k OJVs with salary data, with salary & remote data, and with remote data but without requiring salary data 

- descriptive_statistics.ipynb
    - AI and non-AI role demand over time (monthly)
    - change in benefits over time
    - top companies offering benefits

- logit_model.py

- model_defn.py

- regression_models.ipynb
    - series of logit models with different controls

- occ_year_analysis.py
    - export coeff df for occ_year_analysis_coeffs

- occ_year_analysis_coeffs.ipynb
    - regressions on AI coefficient from occ-year models

# other notebooks
- preliminary_analysis.ipynb
- skillscale_preliminary.ipynb
    

