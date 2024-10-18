
import statsmodels.api as sm
import pandas as pd
import numpy as np
from statsmodels.stats.outliers_influence import variance_inflation_factor
from tqdm import tqdm


def run_logit_model(data, dependent, cat_controls=[], cont_controls = [], binary_vars = [], predictor = 'Has AI Skills', ref_category = None, get_vif = False):
    """
    Runs a logistic regression model to predict work-from-home dependent status based on the specified predictor
    and control variables.

    Parameters:
    -----------
    controls : list of str
        A list of column names to be used as control variables. These variables will be converted to categorical
        variables and dummy variables will be created for them.
    predictor : str, default='Has AI Skills'
        The column name of the primary predictor variable.
    data : pd.DataFrame, default=data
        The DataFrame containing the data.
    ref_category : dict, optional
        A dictionary specifying the reference category for each control variable where keys are column names
        and values are the reference categories.

    Returns:
    --------
    logit_model : statsmodels.discrete.discrete_model.BinaryResultsWrapper
        The fitted logistic regression model.

    Example:
    --------
    >>> controls = ['region_column_name', 'industry_column_name']
    >>> model = run_wfh_logit(controls, predictor='Has AI Skills', data=data)
    >>> print(model.summary())
    """
    if 'NAICS_2022_2_NAME' in cat_controls:
        industry_counts = data['NAICS_2022_2_NAME'].value_counts()
        small_industries = industry_counts[industry_counts < 50].index  # Set a threshold (e.g., 10 observations)
        data['NAICS_2022_2_NAME'] = data['NAICS_2022_2_NAME'].replace(small_industries, 'Other')
    X = data[[predictor]]
    # count nas in predictor
    print(f"Number of NAs in dependent: {data[dependent].isna().sum()}")
    data = data.dropna(subset=[dependent])
    if (dependent == 'wfh_wham' or dependent == 'PARENTAL_LEAVE'):
        data = data[data['YEAR'] != 2018]
    if binary_vars:
        for var in binary_vars:
            X = pd.concat([X, data[[var]]], axis = 1)
    
    if cont_controls:
        for cc in cont_controls:
            X = pd.concat([X, data[[cc]]], axis = 1)
    
    if cat_controls:
        for c in cat_controls:    
            # Create categorical variables 
            data[c] = data[c].astype('category')
            
            # Create dummy variables with specified reference category
            if ref_category and c in ref_category:
                # Ensure the reference category is present in the data
                if ref_category[c] in data[c].cat.categories:
                    data[c] = data[c].cat.reorder_categories(
                        [ref_category[c]] + [cat for cat in data[c].cat.categories if cat != ref_category[c]],
                        ordered=True
                    )
                    dummies = pd.get_dummies(data[c], drop_first=True)
            # Create dummy variables
                else:
                    raise ValueError(f"Reference category '{ref_category[c]}' not found in column '{c}'")
            else:
                dummies = pd.get_dummies(data[c], drop_first=True)
            X = pd.concat([X, dummies], axis = 1)

    X = X.apply(pd.to_numeric, errors='coerce')

    y = data[dependent]
    data = pd.concat([X, y], axis=1).dropna()
    X = data.drop(columns = dependent)
    y = data[dependent]
    X = sm.add_constant(X)
    y = pd.to_numeric(y, errors='coerce')

    X.columns = X.columns.astype(str)

    X = X.astype(float)
    y = y.astype(float)
    # print(X)
    # print(X.corr())
    
    if get_vif:
        print("Correlation Matrix")
        # create new df combine X and y for correlation matrix
        dummy_df = pd.concat([X, y], axis=1)
        dummy_corr_matrix = dummy_df.corr()
        # print any correlations above 0.7
        print(dummy_corr_matrix)
        
        print("VIF Results")
        vif_data = []
        for i in tqdm(range(X.shape[1]), desc="Calculating VIF"):
            vif = variance_inflation_factor(X.values, i)
            vif_data.append(vif)
        vif = pd.DataFrame()
        vif["features"] = X.columns
        vif["VIF"] = vif_data
        print(vif)

        
    # if error, continue to next model
    try:
        logit_model = sm.Logit(y, X).fit()
    except Exception as error:
        # print error
        print(error)
        return "Error"

    # Print the summary of the model
    print(logit_model.summary())
    # print(summary_col(logit_model, stars=True, float_format='%0.2f'))
    return logit_model


def save_model_summaries(models, save_path):
    summaries = []
    for model in models:
        summary = {
            'params': model.params,
            'pvalues': model.pvalues,
            'conf_int': model.conf_int(),
            'summary': model.summary().as_text()
        }
        summaries.append(summary)
    
    with open(save_path, 'wb') as f:
        pickle.dump(summaries, f)