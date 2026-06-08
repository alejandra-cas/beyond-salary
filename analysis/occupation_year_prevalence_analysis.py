#!/usr/bin/env python3
"""
Occupation-Year Benefit Prevalence Analysis

This script computes AI benefit prevalence by group-year and runs OLS regression
models on those outcomes using the full dataset.

It generates both HTML and LaTeX table outputs for the regression results.
"""

import pandas as pd
import numpy as np
import sys
import os
from pathlib import Path
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor

sys.path.append(str(Path(__file__).parent.parent / "src"))

from package_files.config_utils import get_processed_dir, get_repo_root

# Configuration and definitions (from benefits_defns.py)
benefits4 = ['EDU_ASSISTANCE','PAID LEAVE','HEALTH_WELLBEING', 'PARENTAL_LEAVE', 'CULTURE', 'REMOTE_KW']
benefits_labels_map = {
    'EDU_ASSISTANCE': 'Tuition Assistance', 
    'PAID LEAVE': 'Paid Leave', 
    'HEALTH_WELLBEING': 'Health and Wellbeing', 
    'PARENTAL_LEAVE': 'Parental Leave', 
    'CULTURE': 'Workplace Culture', 
    'REMOTE_KW': 'Remote Work'
}
benefits4_labels = ['Tuition Assistance', 'Paid Leave', 'Health and Wellbeing', 'Parental Leave', 'Workplace Culture', 'Remote Work']
occupation = 'COUNTY_NAME'  # updated per-run in run_group_analysis()

REPO_ROOT = get_repo_root()
PROCESSED_DIR = get_processed_dir()

def create_regression_table_html(models, model_names, output_path):
    """
    Create HTML regression table similar to Stargazer output.
    """
    html_content = """
    <!DOCTYPE html>
    <html>
    <head>
        <style>
            table { border-collapse: collapse; margin: 25px 0; font-size: 0.9em; font-family: sans-serif; min-width: 400px; }
            th, td { padding: 8px 12px; text-align: center; border-bottom: 1px solid #dddddd; }
            th { background-color: #f2f2f2; font-weight: bold; }
            .model-header { background-color: #e6f3ff; font-weight: bold; }
            .variable-name { text-align: left; font-weight: bold; }
            .coefficient { text-align: center; }
            .se { font-style: italic; color: #666; }
            .stars { color: #ff0000; }
        </style>
    </head>
    <body>
        <h2>AI Benefit Prevalence Regression Results</h2>
        <table>
    """
    
    # Create header row with model names
    html_content += "<tr><th></th>"
    for i, name in enumerate(model_names):
        html_content += f"<th class='model-header' colspan='2'>{name}</th>"
    html_content += "</tr>\n"
    
    # Sub-header row for Model A and Model B
    html_content += "<tr><th></th>"
    for _ in model_names:
        html_content += "<th>Model A</th><th>Model B</th>"
    html_content += "</tr>\n"
    
    # Get all parameter names from all models
    all_params = set()
    for i in range(0, len(models), 2):  # Every pair of models
        all_params.update(models[i].params.index)
        if i+1 < len(models):
            all_params.update(models[i+1].params.index)
    
    # Sort parameters for consistent ordering
    param_order = ['const'] + [p for p in sorted(all_params) if p != 'const']
    
    # Add parameter rows
    for param in param_order:
        html_content += f"<tr><td class='variable-name'>{param}</td>"
        
        for i in range(0, len(models), 2):  # Every pair of models
            # Model A
            if param in models[i].params.index:
                coef = models[i].params[param]
                se = models[i].bse[param]
                pval = models[i].pvalues[param]
                stars = get_significance_stars(pval)
                html_content += f"<td class='coefficient'>{coef:.4f}{stars}</td>"
            else:
                html_content += "<td>-</td>"
                
            # Model B
            if i+1 < len(models) and param in models[i+1].params.index:
                coef = models[i+1].params[param]
                se = models[i+1].bse[param]
                pval = models[i+1].pvalues[param]
                stars = get_significance_stars(pval)
                html_content += f"<td class='coefficient'>{coef:.4f}{stars}</td>"
            else:
                html_content += "<td>-</td>"
        
        html_content += "</tr>\n"
        
        # Add standard error row
        html_content += f"<tr><td></td>"
        for i in range(0, len(models), 2):  # Every pair of models
            # Model A SE
            if param in models[i].params.index:
                se = models[i].bse[param]
                html_content += f"<td class='se'>({se:.4f})</td>"
            else:
                html_content += "<td>-</td>"
                
            # Model B SE
            if i+1 < len(models) and param in models[i+1].params.index:
                se = models[i+1].bse[param]
                html_content += f"<td class='se'>({se:.4f})</td>"
            else:
                html_content += "<td>-</td>"
        
        html_content += "</tr>\n"
    
    # Add model statistics
    html_content += "<tr><td class='variable-name'>R-squared</td>"
    for i in range(0, len(models), 2):
        html_content += f"<td>{models[i].rsquared:.4f}</td>"
        if i+1 < len(models):
            html_content += f"<td>{models[i+1].rsquared:.4f}</td>"
    html_content += "</tr>\n"
    
    html_content += "<tr><td class='variable-name'>Adj. R-squared</td>"
    for i in range(0, len(models), 2):
        html_content += f"<td>{models[i].rsquared_adj:.4f}</td>"
        if i+1 < len(models):
            html_content += f"<td>{models[i+1].rsquared_adj:.4f}</td>"
    html_content += "</tr>\n"
    
    html_content += "<tr><td class='variable-name'>Observations</td>"
    for i in range(0, len(models), 2):
        html_content += f"<td>{int(models[i].nobs)}</td>"
        if i+1 < len(models):
            html_content += f"<td>{int(models[i+1].nobs)}</td>"
    html_content += "</tr>\n"
    
    html_content += """
        </table>
        <p><small>*** p&lt;0.01, ** p&lt;0.05, * p&lt;0.1</small></p>
    </body>
    </html>
    """
    
    with open(output_path, 'w') as f:
        print(output_path)
        f.write(html_content)
    
    return html_content

def create_regression_table_latex(models, model_names, output_path):
    """
    Create LaTeX regression table similar to Stargazer output.
    """
    latex_content = """\\begin{table}[htbp]
\\centering
\\caption{AI Benefit Prevalence Regression Results}
\\begin{tabular}{l"""
    
    # Add column specifications
    for _ in model_names:
        latex_content += "cc"
    latex_content += "}\n\\hline\\hline\n"
    
    # Header row
    latex_content += " & "
    for i, name in enumerate(model_names):
        latex_content += f"\\multicolumn{{2}}{{c}}{{{name}}}"
        if i < len(model_names) - 1:
            latex_content += " & "
    latex_content += " \\\\\n"
    
    # Sub-header row
    latex_content += " & "
    for i in range(len(model_names)):
        latex_content += "Model A & Model B"
        if i < len(model_names) - 1:
            latex_content += " & "
    latex_content += " \\\\\n\\hline\n"
    
    # Get all parameter names
    all_params = set()
    for i in range(0, len(models), 2):
        all_params.update(models[i].params.index)
        if i+1 < len(models):
            all_params.update(models[i+1].params.index)
    
    param_order = ['const'] + [p for p in sorted(all_params) if p != 'const']
    
    # Add parameter rows
    for param in param_order:
        param_name = param.replace('_', '\\_')
        latex_content += f"{param_name} & "
        
        coeffs = []
        for i in range(0, len(models), 2):
            # Model A
            if param in models[i].params.index:
                coef = models[i].params[param]
                pval = models[i].pvalues[param]
                stars = get_significance_stars(pval)
                coeffs.append(f"{coef:.4f}{stars}")
            else:
                coeffs.append("-")
                
            # Model B
            if i+1 < len(models) and param in models[i+1].params.index:
                coef = models[i+1].params[param]
                pval = models[i+1].pvalues[param]
                stars = get_significance_stars(pval)
                coeffs.append(f"{coef:.4f}{stars}")
            else:
                coeffs.append("-")
        
        latex_content += " & ".join(coeffs) + " \\\\\n"
        
        # Standard errors row
        latex_content += " & "
        ses = []
        for i in range(0, len(models), 2):
            # Model A SE
            if param in models[i].params.index:
                se = models[i].bse[param]
                ses.append(f"({se:.4f})")
            else:
                ses.append("-")
                
            # Model B SE
            if i+1 < len(models) and param in models[i+1].params.index:
                se = models[i+1].bse[param]
                ses.append(f"({se:.4f})")
            else:
                ses.append("-")
        
        latex_content += " & ".join(ses) + " \\\\\n"
    
    latex_content += "\\hline\n"
    
    # Model statistics
    latex_content += "R-squared & "
    r2s = []
    for i in range(0, len(models), 2):
        r2s.append(f"{models[i].rsquared:.4f}")
        if i+1 < len(models):
            r2s.append(f"{models[i+1].rsquared:.4f}")
    latex_content += " & ".join(r2s) + " \\\\\n"
    
    latex_content += "Adj. R-squared & "
    adj_r2s = []
    for i in range(0, len(models), 2):
        adj_r2s.append(f"{models[i].rsquared_adj:.4f}")
        if i+1 < len(models):
            adj_r2s.append(f"{models[i+1].rsquared_adj:.4f}")
    latex_content += " & ".join(adj_r2s) + " \\\\\n"
    
    latex_content += "Observations & "
    obs = []
    for i in range(0, len(models), 2):
        obs.append(f"{int(models[i].nobs)}")
        if i+1 < len(models):
            obs.append(f"{int(models[i+1].nobs)}")
    latex_content += " & ".join(obs) + " \\\\\n"
    
    latex_content += """\\hline\\hline
\\end{tabular}
\\begin{tablenotes}[flushleft]
\\footnotesize
\\item *** p$<$0.01, ** p$<$0.05, * p$<$0.1
\\end{tablenotes}
\\end{table}
"""
    
    with open(output_path, 'w') as f:
        f.write(latex_content)
    
    return latex_content

def get_significance_stars(pval):
    """Return significance stars based on p-value."""
    if pval < 0.01:
        return "***"
    elif pval < 0.05:
        return "**"
    elif pval < 0.1:
        return "*"
    else:
        return ""

def calculate_ai_benefit_prevalence(data):
    """
    Calculate mean benefit prevalence for AI jobs by group-year.
    Previously computed differences (AI - non-AI); now returns AI prevalence only.
    """
    print("Calculating AI benefit prevalence...")

    ai_jobs = data[data['AI ROLE'] == 1]

    # Mean prevalence of each benefit for AI roles per group-year
    ai_prevalence = ai_jobs.groupby([occupation, 'YEAR'])[benefits4].mean().reset_index()

    # Rename to standard column names used by downstream models
    for benefit in benefits4:
        ai_prevalence.rename(columns={benefit: f'{benefit}_ai_prevalence'}, inplace=True)

    print(ai_prevalence)
    return ai_prevalence

def load_occupation_year_data(occ_year_data_path):
    """
    Load the main occupation-year analysis data.
    This contains the covariates for the regression models.
    """
    print("Loading occupation-year data...")
    df = pd.read_parquet(occ_year_data_path)
    
    # Add log job count
    df['Log Job Count'] = df['job_count'].apply(lambda x: np.log(x))
    
    return df

def run_prevalence_models(df):
    """
    Run OLS regression models with AI benefit prevalence as the dependent variable.
    """
    print("Running AI benefit prevalence regression models...")
    
    diff_models = []
    
    for benefit in benefits4:
        benefit_df = df.copy()
        benefit_df = benefit_df.rename(columns={
            f'ai_role_count': 'AI Job Count', 
            'MEDIAN_LOG_SALARY_ai': 'Median Log Salary AI',
            f'Prevalence: {benefits_labels_map[benefit]}': f'Overall Benefit Prevalence'
        })
        
        # Handle remote work (exclude 2018)
        if benefit == 'REMOTE_KW':
            benefit_df = benefit_df[benefit_df['YEAR'] != 2018]
        
        label = benefits_labels_map[benefit]
        log_cols = [f'Overall Benefit Prevalence']
        benefit_df = benefit_df.dropna(subset=log_cols)
        
        # Apply log transformation
        for col in log_cols:
            benefit_df[f'Log {col}'] = benefit_df[col].apply(lambda x: np.sign(x) * np.log(abs(x) + 1))
        
        X_cols = [f'Log {col}' for col in log_cols] + ['Median Log Salary AI', 'Log Job Count'] 
        y_col = f'{benefit}_ai_prevalence'

        benefit_df = benefit_df.dropna(subset=X_cols + [y_col])
        X = benefit_df[X_cols]
        y = benefit_df[y_col]

        # Add year dummies (excluding 2019 as reference)
        year_dummies = pd.get_dummies(benefit_df['YEAR'], prefix='Year').drop(columns=['Year_2019'], errors='ignore')
        # Convert boolean dummies to integers for statsmodels compatibility
        year_dummies = year_dummies.astype(int)
        X = pd.concat([X, year_dummies], axis=1)
        X = sm.add_constant(X)

        # Fit the OLS model
        model = sm.OLS(y, X).fit()
        print(f"\n{benefit} Model Summary:")
        print(f"R-squared: {model.rsquared:.4f}")
        print(f"Adj. R-squared: {model.rsquared_adj:.4f}")

        # Calculate VIF (Variance Inflation Factor)
        print("VIF Results:")
        vif_data = []
        for i in range(X.shape[1]):
            vif = variance_inflation_factor(X.values, i)
            vif_data.append(vif)
        vif_df = pd.DataFrame()
        vif_df["features"] = X.columns
        vif_df["VIF"] = vif_data
        print(vif_df)

        diff_models.append(model)

    return diff_models

def run_prevalence_models_extended(df):
    """
    Run extended OLS regression models (Model 2).
    Includes AI Job Count as an additional variable.
    """
    print("Running extended AI benefit prevalence regression models...")
    
    diff_models_2 = []
    
    for benefit in benefits4:
        benefit_df = df.copy()
        benefit_df = benefit_df.rename(columns={
            f'ai_role_count': 'AI Job Count', 
            'MEDIAN_LOG_SALARY_ai': 'Median Log Salary AI',
            f'Prevalence: {benefits_labels_map[benefit]}': f'Overall Benefit Prevalence'
        })
        
        # Handle remote work (exclude 2018)
        if benefit == 'REMOTE_KW':
            benefit_df = benefit_df[benefit_df['YEAR'] != 2018]
        
        label = benefits_labels_map[benefit]
        log_cols = [f'Overall Benefit Prevalence', 'AI Job Count']
        benefit_df = benefit_df.dropna(subset=log_cols)
        
        # Apply log transformation
        for col in log_cols:
            benefit_df[f'Log {col}'] = benefit_df[col].apply(lambda x: np.sign(x) * np.log(abs(x) + 1))
        
        X_cols = [f'Log {col}' for col in log_cols] + ['Median Log Salary AI', 'Log Job Count'] 
        y_col = f'{benefit}_ai_prevalence'

        benefit_df = benefit_df.dropna(subset=X_cols + [y_col])
        X = benefit_df[X_cols]
        y = benefit_df[y_col]

        # Add year dummies (excluding 2019 as reference)
        year_dummies = pd.get_dummies(benefit_df['YEAR'], prefix='Year').drop(columns=['Year_2019'], errors='ignore')
        # Convert boolean dummies to integers for statsmodels compatibility
        year_dummies = year_dummies.astype(int)
        X = pd.concat([X, year_dummies], axis=1)
        X = sm.add_constant(X)

        # Fit the OLS model
        model = sm.OLS(y, X).fit()
        print(f"\n{benefit} Extended Model Summary:")
        print(f"R-squared: {model.rsquared:.4f}")
        print(f"Adj. R-squared: {model.rsquared_adj:.4f}")

        diff_models_2.append(model)
    
    return diff_models_2

def generate_combined_tables(diff_models, diff_models_2, output_dir):
    """
    Generate combined HTML and LaTeX tables for the difference regression models.
    Reproduces the table generation from occ_year_model_new.ipynb
    """
    print("Generating combined regression tables...")
    
    # Create combined models list (alternating between Model A and Model B for each benefit)
    combined_models = []
    for model_a, model_b in zip(diff_models, diff_models_2):
        combined_models.extend([model_a, model_b])
    
    # Generate HTML table
    html_output_path = os.path.join(output_dir, "difference_regression_table_combined.html")
    html_table = create_regression_table_html(combined_models, benefits4_labels, html_output_path)
    print(f"HTML table saved to: {html_output_path}")
    
    # Generate LaTeX table
    latex_output_path = os.path.join(output_dir, "difference_regression_table_combined.tex")
    latex_table = create_regression_table_latex(combined_models, benefits4_labels, latex_output_path)
    print(f"LaTeX table saved to: {latex_output_path}")
    
    return html_table, latex_table

def run_group_analysis(data, group_col, analysis_parquet, output_dir):
    """
    Run the full prevalence analysis for a given grouping column.

    Parameters
    ----------
    data : DataFrame - full labeled dataset (labeled_v2.parquet)
    group_col : str - column to group by (e.g. 'SOC_MAJOR_GROUP', 'NAICS_2022_2_NAME')
    analysis_parquet : Path - path to the group-year analysis parquet
    output_dir : str - directory for HTML/LaTeX table output
    """
    global occupation
    occupation = group_col

    print(f"\n{'=' * 60}")
    print(f"Running analysis by {group_col}")
    print(f"{'=' * 60}")

    os.makedirs(output_dir, exist_ok=True)

    # Step 1: Calculate AI benefit prevalence from full dataset
    ai_prevalence = calculate_ai_benefit_prevalence(data)

    # Step 2: Load group-year data
    group_year_df = load_occupation_year_data(analysis_parquet)

    # Remove existing prevalence columns to avoid conflicts
    prev_cols = [f"{benefit}_ai_prevalence" for benefit in benefits4]
    existing_prev_cols = [col for col in group_year_df.columns if col in prev_cols]
    if existing_prev_cols:
        print(f"Removing existing prevalence columns: {existing_prev_cols}")
        group_year_df = group_year_df.drop(columns=existing_prev_cols)

    # Step 3: Merge with AI benefit prevalence
    print("Merging group-year data with AI benefit prevalence...")
    df = group_year_df.merge(ai_prevalence, on=[occupation, 'YEAR'])

    print(f"Final dataset shape: {df.shape}")
    print(f"Group-years: {len(df)}")
    print(f"Years covered: {sorted(df['YEAR'].unique())}")
    print(f"Groups: {len(df[occupation].unique())}")

    # Step 4: Run regression models
    diff_models = run_prevalence_models(df)
    diff_models_2 = run_prevalence_models_extended(df)

    # Step 5: Generate combined tables
    html_table, latex_table = generate_combined_tables(diff_models, diff_models_2, output_dir)

    print(f"\nAnalysis by {group_col} completed!")
    return {
        'data': df,
        'models_basic': diff_models,
        'models_extended': diff_models_2,
        'html_table': html_table,
        'latex_table': latex_table
    }


def main():
    """Run the prevalence analysis for both occupation-year and industry-year."""
    # Load data once
    print("Loading data...")
    data = pd.read_parquet(PROCESSED_DIR / "labeled_v2.parquet")

    # Occupation-year analysis
    occ_results = run_group_analysis(
        data,
        group_col='SOC_MAJOR_GROUP',
        analysis_parquet=PROCESSED_DIR / "occ_year_analysis_raw.parquet",
        output_dir=REPO_ROOT / "results" / "tables" / "occ_year_models",
    )

    # Industry-year analysis
    ind_results = run_group_analysis(
        data,
        group_col='NAICS_2022_2_NAME',
        analysis_parquet=PROCESSED_DIR / "ind_year_analysis_raw.parquet",
        output_dir=REPO_ROOT / "results" / "tables" / "ind_year_models",
    )

    print("\n" + "=" * 60)
    print("All analyses completed successfully!")
    print("=" * 60)

    return {'occupation': occ_results, 'industry': ind_results}


if __name__ == "__main__":
    results = main()
