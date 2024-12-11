# SkillScale Beyond Salary Project

# Pipeline
- prepare_data.ipynb
- classify_ai_skills.py
- merge_body.py
- label_benefits.py
- prepare_data_2.ipynb
- export_samples.py

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
    - filters out internships
    - merges data from thesis to get AI and WHAM job labels 
    - rename 'Has AI Skills' column to 'AI ROLE'
    - adds 'YEAR' column
    - adds any missing WHAM labels from WHAM data


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
    - export df with descriptive statistics for occupation-year model
    - ~~export coeff df for occ_year_analysis_coeffs~~

- occ_year_model_new.ipynb
    - model: % benefit for ai role = a + B1 * AI Demand + B2* Overall Prevalence...

# other notebooks
- preliminary_analysis.ipynb
- skillscale_preliminary.ipynb
- occ_year_sample_models.ipynb
    - run job-level models on each occupation-year to get AI coeffs
- occ_year_analysis_coeffs.ipynb
    - regressions on AI coefficient from occ-year models
- export_samples.ipynb
    - comparison to national statistics



# SkillScale Beyond Salary Project

## Overview

This project explores non-monetary benefit offerings for AI vs. non-AI jobs using a dataset of nearly ten million online job vacancies. It uncovers trends in recent years for benefits such as paid leave, tuition assistance, health and wellbeing, parental leave, workplace culture, and remote work. Additionally, it examines the relationship between benefits offered for AI roles and job demand to determine whether high levels of AI demand are associated with higher levels of non-monetary benefits. This project is structured around a set of Python scripts and Jupyter notebooks.

## Pipeline

### Scripts

1. **prepare_data.ipynb**
   - Loads original online job vacancy (OJV) data.
   - Filters out internships.
   - Merges data from thesis to include AI and WHAM job labels.
   - Adds columns such as `YEAR` and updates missing WHAM labels.

2. **classify_ai_skills.py**
   - Classifies jobs as AI or non-AI for any jobs with missing labels after the merge in `prepare_data.ipynb`.

3. **merge_body.py**
   - Merges the `BODY` column to OJV data.
   - Exports updated OJV data.

4. **label_benefits.py**
   - Checks whether the `BODY` column of OJV data contains specified keywords for each benefit.
   - Adds a column for each benefit indicating `True` or `False`.
   - Imports and exports data using specified paths.

5. **prepare_data_2.ipynb**
   - Renames the `Has AI Skills` column to `AI ROLE`.
   - Adds an `EXPERIENCE_BUCKET` column and joins salary data.
   - Adds a `LOG_SALARY` column and fills in missing AI labels.
   - Exports processed data.

6. **export_samples.py**
   - Exports samples of 20k OJVs with different criteria, such as salary and remote data.

7. **descriptive_statistics.ipynb**
   - Analyzes demand for AI and non-AI roles over time.
   - Tracks changes in benefits over time.
   - Identifies top companies offering benefits.

8. **regression_models.ipynb**
   - Runs logit models with different controls to explore relationships between benefits and demand.

9. **occ_year_analysis.py**
   - Exports dataframes with descriptive statistics for occupation-year models.
   - Generates coefficients for further analysis.

10. **occ_year_model_new.ipynb**
    - Builds models exploring the relationship between AI role benefits and demand, including control variables.

## Data

- **salary_sample_body_benefits.parquet.gzip**: 20k sample extracted from a dataset of 10 million job postings, filtered for no internships and only jobs with salary information.
- **even_sample.csv**: 20k sample from a shared dataset of 1 million job postings, filtered for no internships.
- **us-003_noint.parquet**: 1 million OJV sample filtered for no internships.

