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

# py files
- merge_body.py
    - adds body column to OJV data sample and exports samples

- label_benefits.py
    - Checks whether the BODY column of OJV data contains specified keywords for each benefit. Adds a column for each benefit to a dataframe indicating True or False. Exports to 'data/salary_sample_body_benefits.parquet.gzip'

- logit_model.py

- model_defn.py

# notebooks
- prepare_data.ipynb
    - imports salary_sample_body_benefits.parquet
    - renames AI skills column to AI ROLE
    - adds EXPERIENCE_BUCKET column
    - adds LOG_SALARY column
    - exports to salary_sample_body_benefits.parquet

- descriptive_statistics.ipynb
    - AI and non-AI role demand over time (monthly)
    - change in benefits over time
    - top companies offering benefits

- regression_models.ipynb
    - series of logit models with different controls

- preliminary_analysis.ipynb
- skillscale_preliminary.ipynb
    

