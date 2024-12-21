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

### Data Preparation

1. **prepare\_data.ipynb**
   - Loads original online job vacancy (OJV) data.
   - Filters out internships.
   - Merges data from thesis to include AI and WHAM job labels.
   - Adds columns such as `YEAR` and updates missing WHAM labels.
   - **Data Used**: `us-003_noint.parquet`

2. **classify\_ai\_skills.py**
   - Classifies jobs as AI or non-AI for any jobs with missing labels after the merge in `prepare_data.ipynb`.
   - **Data Used**: Data output from `prepare_data.ipynb`

3. **merge\_body.py**
   - Merges the `BODY` column to OJV data.
   - Exports updated OJV data.
   - **Data Used**: Updated OJV data and `salary_sample_body_benefits.parquet.gzip`

4. **label\_benefits.py**
   - Checks whether the `BODY` column of OJV data contains specified keywords for each benefit.
   - Adds a column for each benefit indicating `True` or `False`.
   - **Data Used**: Merged OJV data from `merge_body.py`

5. **prepare\_data\_2.ipynb**
   - Renames the `Has AI Skills` column to `AI ROLE`.
   - Adds an `EXPERIENCE_BUCKET` column and joins salary data.
   - Adds a `LOG_SALARY` column and fills in missing AI labels.
   - Exports processed data.
   - **Data Used**: Updated OJV dataset from `label_benefits.py`

### Data Sampling

6. **export\_samples.py**
   - Exports samples of 20k OJVs with different criteria, such as salary and remote data.
   - **Data Used**: Processed OJV data

### Descriptive Analysis

7. **descriptive\_statistics.ipynb**
   - Analyzes demand for AI and non-AI roles over time.
   - Tracks changes in benefits over time.
   - Identifies top companies offering benefits.
   - **Data Used**: Sampled data from `export_samples.py`

8. **join_remote_kw.ipynb**
   - Analyzes and joins data for remote keyword classifications.
   - **Data Used**: Data with remote work classifications

9. **duration.ipynb**
   - Explores the duration of job postings and related trends.
   - **Data Used**: Processed OJV data

10. **national_comparison.ipynb**
    - Compares AI-related benefits and demand across national datasets.
    - **Data Used**: Sampled and national datasets

### Modeling

11. **regression\_models.ipynb**
    - Runs logit models with different controls to explore relationships between benefits and demand.
    - **Data Used**: Processed OJV data

12. **occ\_year\_analysis.py**
    - Exports dataframes with descriptive statistics for occupation-year models.
    - Generates coefficients for further analysis.
    - **Data Used**: Processed OJV data

13. **occ\_year\_model\_new.ipynb**
    - Builds models exploring the relationship between AI role benefits and demand, including control variables.
    - **Data Used**: Occupation-year model data

14. **compounding_effects.ipynb**
    - Investigates the compounding effects of AI demand on job benefits.
    - **Data Used**: Processed OJV data

15. **occ_year_analysis_coeffs.ipynb**
    - Generates coefficients from occupation-year analysis models.
    - **Data Used**: Occupation-year model data

### Validation

16. **check_ai_skills.ipynb**
    - Validates AI skill classifications and labels in the dataset.
    - **Data Used**: Processed OJV data

17. **evaluate_accuracy.ipynb**
    - Evaluates the accuracy of benefit and AI skill classifications.
    - **Data Used**: Labeled benefits data

18. **manual_check_export.ipynb**
    - Manually checks and exports data for validation purposes.
    - **Data Used**: Labeled and processed OJV data

### Visualization

19. **scatterplot.ipynb**
    - Generates scatterplots to visualize AI demand and benefit trends.
    - **Data Used**: Sampled data

20. **keyword_definitions.ipynb**
    - Documents and defines keywords used for benefit labeling.
    - **Data Used**: Keywords metadata

21. **remote_keywords.ipynb**
    - Identifies and analyzes keywords related to remote work.
    - **Data Used**: Processed OJV data

22. **wage_info.ipynb**
    - Processes and analyzes wage information for job postings.
    - **Data Used**: Wage-related OJV data

23. **test_keyword_search.ipynb**
    - Tests and refines the keyword search functionality.
    - **Data Used**: Processed OJV data

## Additional Scripts

The following scripts define key functions and lists that are used throughout the pipeline:

1. **benefits_defns.py**
   - Contains definitions and lists for benefit keywords used in labeling.
   - Supports `label_benefits.py` and other related scripts.

2. **logit_model.py**
   - Implements functions for building and analyzing logistic regression models.
   - Supports notebooks and scripts in the modeling pipeline.

## Data

- **salary\_sample\_body\_benefits.parquet.gzip**: 20k sample extracted from a dataset of 10 million job postings, filtered for no internships and only jobs with salary information.
- **even\_sample.csv**: 20k sample from a shared dataset of 1 million job postings, filtered for no internships.
- **us-003\_noint.parquet**: 1 million OJV sample filtered for no internships.




