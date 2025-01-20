# SkillScale Beyond Salary Project

## Overview

This project explores non-monetary benefit offerings for AI vs. non-AI jobs using a dataset of nearly ten million online job vacancies. It uncovers trends in recent years for benefits such as paid leave, tuition assistance, health and wellbeing, parental leave, workplace culture, and remote work. Additionally, it examines the relationship between benefits offered for AI roles and job demand to determine whether high levels of AI demand are associated with higher levels of non-monetary benefits.

## Pipeline

### Data Preparation

1. **prepare\_data.ipynb**
   - Loads original online job vacancy (OJV) data.
   - Filters out internships.
   - Merges data from thesis to include AI and WHAM job labels.
   - Adds columns such as `YEAR` and updates missing WHAM labels.

2. **classify\_ai\_skills.py**
   - Classifies jobs as AI or non-AI for any jobs with missing labels after the merge in `prepare_data.ipynb`.
   - **Data Used**: Data output from `prepare_data.ipynb`

3. **merge\_body.py**
   - Merges the `BODY` column to OJV data.
   - Exports updated OJV data.

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
   - e.g. export samples of 20k OJVs with salary data, with salary & remote data, and with remote data but without requiring salary data 
   - **Data Used**: Processed OJV data

### Descriptive Analysis

7. **descriptive\_statistics.ipynb**
   - Analyzes demand for AI and non-AI roles over time.
   - Tracks changes in benefits over time.
   - Identifies top companies offering benefits.

8. **join_remote_kw.ipynb**
   - Join remote keyword classifications to processed OJV data and export updated df

9. **duration.ipynb**
   - Explores the duration of job postings and related trends.

10. **national_comparison.ipynb**
    - Compares representativeness of OJV data vs. national statistics

### Modeling

11. **regression\_models.ipynb**
    - Runs logit models with different controls to explore relationships between benefits and demand.

12. **occ\_year\_analysis.py**
    - Exports dataframes with descriptive statistics for occupation-year models.
   
12. **occ\_year\_analysis_raw.py**
    - Exports dataframes with descriptive statistics for occupation-year models with raw numbers.

13. **occ\_year\_model\_new.ipynb**
    - Builds models exploring the relationship between AI role benefits and demand on occ-year level, including control variables.
    - **Data Used**: Occupation-year data

14. **compounding_effects.ipynb**
    - Investigates the compounding effects of job benefits.
    - **Data Used**: Processed OJV data

15. **occ_year_analysis_coeffs.ipynb**
    - Runs models looking at the effect of demand on the strength of AI role coefficients
    - **Data Used**: "occ_year_coeff_analysis.csv"

### Validation

16. **evaluate_accuracy.ipynb**
    - Evaluates the accuracy of benefit and AI skill classifications.
    - **Data Used**: highlighted_job_postings_2_exploded.xlsx

17. **manual_check_export.ipynb**
    - Manually checks and exports data for validation purposes.
    - **Data Used**: Sample of labeled data with benefits

### Visualization

18. **scatterplot.ipynb**
    - Generates scatterplots to visualize AI demand and benefit correlations.
    - **Data Used**: Occ-year analysis df

19. **wage_info.ipynb**
    - Processes and analyzes wage information
    - **Data Used**: OJV data with salary information

## Supporting Scripts

The following scripts define key functions and lists that are used throughout the pipeline:

1. **benefits_defns.py**
   - Contains definitions and lists for benefit keywords used in labeling.
   - Supports `label_benefits.py` and other related scripts.

2. **logit_model.py**
   - Implements functions for building and analyzing logistic regression models.
   - Supports notebooks and scripts in the modeling pipeline.

## Archive
1. **check_ai_skills.ipynb**
    - Validates length of AI skills used to classify AI roles.
2. **keyword_definitions.ipynb**
    - Documents and defines keywords used for benefit labeling.
    - **Data Used**: Keywords metadata
3. **remote_keywords.ipynb**
    - Scrutinizes text data to identify remote work keywords
    - **Data Used**: OJV data samples
4. **test_keyword_search.ipynb**
    - Tests and refines the keyword search functionality.
    - **Data Used**: Processed OJV data
5. **preliminary_analysis.ipynb**
   - preliminary exploratory analysis
6. **skillscale_preliminary.ipynb**
    - preliminary exploratory analysis
7. **occ_year_sample_models.ipynb**
    - run job-level models on each occupation-year to get AI coeffs
8. **occ_year_analysis_coeffs.ipynb**
    - regressions on AI coefficient from occ-year models
9. **export_samples.ipynb**
