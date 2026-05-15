#!/usr/bin/env python3
"""
Regression Models for 2024 Analysis (Including Data up to 2024)

This script runs the job-level logistic regression models.  
Notes:
- PARENTAL_LEAVE models exclude 2018 data #TODO why?
- Industry grouping for small categories
- Includes proper reference category handling

Generates:
1. Model coefficients plot
2. Individual benefit regression tables (properly formatted)
3. Final wide table with all results
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import sys
import os
from pathlib import Path

# Add src directory to path
sys.path.append(str(Path(__file__).parent.parent / 'src'))

from package_files.logit_model import run_logit_model
from package_files.benefits_defns import *
import statsmodels.api as sm
from collections import defaultdict
from stargazer.stargazer import Stargazer

# Configuration
plt.rcParams.update({'font.size': 14})

# Field mappings
region = 'STATE_NAME'
industry = 'NAICS_2022_2_NAME'
firm = 'COMPANY'
education = 'MIN_EDULEVELS_NAME'
year = 'YEAR'
occupation = 'SOC_MAJOR_GROUP'
experience = 'EXPERIENCE_BUCKET'

# Benefits to analyze (2024 models)
benefits4 = ['EDU_ASSISTANCE', 'PAID LEAVE', 'HEALTH_WELLBEING', 'PARENTAL_LEAVE', 'CULTURE', 'REMOTE_KW']
benefits4_labels = ['Tuition Assistance', 'Paid Leave', 'Health and Wellbeing', 'Parental Leave', 'Workplace Culture', 'Remote Work']

# Color scheme for plots
colors = ['#E69F00', '#56B4E9', '#009E73', '#CC79A7', '#0072B2', '#D55E00', '#009E73']


def prepare_firm_fixed_effects(data, min_firm_obs=30):
    """Reduce firm cardinality for FE model to avoid exploding dummy matrices."""
    data_fe = data.copy()
    data_fe[firm] = data_fe[firm].fillna('Unknown Firm').astype(str)
    data_fe[region] = data_fe[region].fillna('Unknown State').astype(str)

    firm_counts = data_fe[firm].value_counts()
    rare_firms = firm_counts[firm_counts < min_firm_obs].index
    data_fe[firm] = data_fe[firm].replace(rare_firms, 'Other Firm (<30 obs)')

    print(
        f"Firm FE prep: {len(firm_counts):,} firms -> "
        f"{data_fe[firm].nunique():,} categories after grouping firms with <{min_firm_obs} obs"
    )

    return data_fe

def load_and_prepare_data():
    """Load and prepare the 2024 dataset with all preprocessing."""
    print("Loading 2024 dataset...")
    
    # Load the data
    _base = Path(__file__).parent.parent / "data" / "processed"
    data_path = _base / 'labeled_v1.parquet'
    if not data_path.exists():
        print(f"Warning: {data_path} not found. Please update the path.")
        return None

    # Using full dataset (not balanced/even sample)
    data = pd.read_parquet(data_path)

    remote_df = pd.read_parquet(_base / 'labeled_v2.parquet')
    data = data.merge(remote_df[['ID', 'REMOTE_KW']], on='ID', how='left')

    # Create experience buckets
    print("Creating experience buckets...")
    bins = [-2, -1, 0, 2, 5, 10, 20, 100]
    labels = ['Missing', '0 years', '1-2 years', '3-5 years', '6-10 years', '11-20 years', '21+ years']
    data['EXPERIENCE_BUCKET'] = pd.cut(data['MIN_YEARS_EXPERIENCE'], bins=bins, labels=labels, right=True)
    data['EXPERIENCE_BUCKET'] = data['EXPERIENCE_BUCKET'].astype(str)
    data['EXPERIENCE_BUCKET'] = data['EXPERIENCE_BUCKET'].replace('nan', 'None Listed') # why not use this as continuous?
    
    # Create log salary
    print("Creating log salary...")
    data['LOG_SALARY'] = np.log(data['SALARY'])
    
    # Replace small industries with "Other" (this is also done in run_logit_model but we do it here for consistency)
    print("Processing industry categories...")
    industry_counts = data['NAICS_2022_2_NAME'].value_counts()
    small_industries = industry_counts[industry_counts < 30].index  
    data['NAICS_2022_2_NAME'] = data['NAICS_2022_2_NAME'].replace(small_industries, 'Other')
    
    print(f"Final dataset shape: {data.shape}")
    print(f"Year distribution:")
    print(data.groupby('YEAR').size())
    
    return data

def run_2024_models(data):
    """Run model specifications, including a second firm+state FE progression panel."""
    
    print("="*80)
    print("RUNNING 2024 REGRESSION MODELS")
    print("="*80)
    
    # Model 1: Baseline (Year + Industry fixed effects)
    print("\n" + "="*50)
    print("MODEL 1: BASELINE (YEAR + INDUSTRY FIXED EFFECTS)")
    print("="*50)
    
    benefit_models_industry = []
    for benefit in benefits4:
        print(f"\n{benefit}")
        print('-'*100)
        model = run_logit_model(data, dependent=benefit, predictor='AI ROLE', 
                              cat_controls=[year, industry], get_vif=False)
        benefit_models_industry.append(model)
    
    # Model 2: With Individual Controls (Year + Industry + Education + Experience)
    print("\n" + "="*50)  
    print("MODEL 2: WITH INDIVIDUAL CONTROLS")
    print("="*50)
    
    benefit_models_industry_2 = []
    for benefit in benefits4:
        print(f"\n{benefit}")
        print('-'*100)
        model = run_logit_model(data, dependent=benefit, predictor='AI ROLE', 
                              cat_controls=[year, industry, education, experience],  
                              ref_category={education: "No Education Listed", experience: 'None Listed'}, 
                              get_vif=False)
        benefit_models_industry_2.append(model)
    
    # Model 3: With Salary Control (Year + Industry + Education + Experience + Log Salary)
    print("\n" + "="*50)
    print("MODEL 3: WITH SALARY CONTROL") 
    print("="*50)
    
    benefit_models_industry_3 = []
    for benefit in benefits4:
        print(f"\n{benefit}")
        print('-'*100)
        model = run_logit_model(data, dependent=benefit, predictor='AI ROLE', 
                              cat_controls=[year, industry, education, experience], 
                              cont_controls=['LOG_SALARY'], 
                              ref_category={education: "No Education Listed", experience: 'None Listed'})
        benefit_models_industry_3.append(model)

    # Model 4: Full model with firm + state FE
    print("\n" + "="*50)
    print("MODEL 4: FIRM + STATE FIXED EFFECTS")
    print("="*50)

    data_fe = prepare_firm_fixed_effects(data)
    benefit_models_firm_state_4 = []
    for benefit in benefits4:
        print(f"\n{benefit}")
        print('-'*100)
        model = run_logit_model(
            data_fe,
            dependent=benefit,
            predictor='AI ROLE',
            cat_controls=[year, firm, region, education, experience],
            cont_controls=['LOG_SALARY'],
            ref_category={education: "No Education Listed", experience: 'None Listed'},
            get_vif=False,
        )
        benefit_models_firm_state_4.append(model)

    # Panel 2 starts here: baseline with firm+state FE, then incremental controls
    print("\n" + "="*50)
    print("MODEL 5: PANEL 2 BASELINE (YEAR + FIRM + STATE FE)")
    print("="*50)

    benefit_models_firm_state_5 = []
    for benefit in benefits4:
        print(f"\n{benefit}")
        print('-'*100)
        model = run_logit_model(
            data_fe,
            dependent=benefit,
            predictor='AI ROLE',
            cat_controls=[year, firm, region],
            get_vif=False,
        )
        benefit_models_firm_state_5.append(model)

    print("\n" + "="*50)
    print("MODEL 6: PANEL 2 + INDIVIDUAL CONTROLS")
    print("="*50)

    benefit_models_firm_state_6 = []
    for benefit in benefits4:
        print(f"\n{benefit}")
        print('-'*100)
        model = run_logit_model(
            data_fe,
            dependent=benefit,
            predictor='AI ROLE',
            cat_controls=[year, firm, region, education, experience],
            ref_category={education: "No Education Listed", experience: 'None Listed'},
            get_vif=False,
        )
        benefit_models_firm_state_6.append(model)

    print("\n" + "="*50)
    print("MODEL 7: PANEL 2 + INDIVIDUAL CONTROLS + SALARY")
    print("="*50)

    benefit_models_firm_state_7 = []
    for benefit in benefits4:
        print(f"\n{benefit}")
        print('-'*100)
        model = run_logit_model(
            data_fe,
            dependent=benefit,
            predictor='AI ROLE',
            cat_controls=[year, firm, region, education, experience],
            cont_controls=['LOG_SALARY'],
            ref_category={education: "No Education Listed", experience: 'None Listed'},
            get_vif=False,
        )
        benefit_models_firm_state_7.append(model)
    
    return [
        benefit_models_industry,
        benefit_models_industry_2,
        benefit_models_industry_3,
        benefit_models_firm_state_4,
        benefit_models_firm_state_5,
        benefit_models_firm_state_6,
        benefit_models_firm_state_7,
    ]

def extract_model_results(models_2024):
    """Extract coefficients, standard errors, p-values, and model statistics."""
    results_dfs = []
    
    model_names = [
        'P1 M1: Baseline',
        'P1 M2: +Indiv Controls',
        'P1 M3: +Salary',
        'P1 M4: +Firm+State FE',
        'P2 M1: Firm+State Baseline',
        'P2 M2: +Indiv Controls',
        'P2 M3: +Salary',
    ]
    
    for model_idx, models in enumerate(models_2024):
        coefficients = []
        errors = []
        pvalues = []
        converged_list = []
        observations = []
        
        for model in models:
            try:
                coef = model.params['AI ROLE']
                err = model.bse['AI ROLE']
                pvalue = model.pvalues['AI ROLE'].round(3)
                converged = model.converged 
                obs = model.nobs       
            except:
                coef = None
                err = None
                pvalue = None
                converged = None
                obs = None
            
            coefficients.append(coef)
            errors.append(err)
            pvalues.append(pvalue)
            converged_list.append(converged)
            observations.append(obs)
        
        results_df = pd.DataFrame({
            'Label': benefits4,
            'Coefficient': coefficients,
            'Error': errors,
            'P-Value': pvalues,
            'Converged': converged_list, 
            'Observations': observations,
            'Model Iteration': model_names[model_idx]
        })
        
        results_dfs.append(results_df)
    
    return pd.concat(results_dfs, ignore_index=True)

def generate_coefficients_plot(results_df):
    """Generate the model coefficients plot with confidence intervals."""
    print("\nGenerating model coefficients plot...")
    
    # Calculate 95% confidence intervals
    results_df['Lower_CI'] = results_df['Coefficient'] - 1.96 * results_df['Error']
    results_df['Upper_CI'] = results_df['Coefficient'] + 1.96 * results_df['Error']
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    model_iterations = results_df['Model Iteration'].unique()
    markers = ['o', 's', '^', 'D', 'P', 'X', 'v']  # Different markers for model iterations
    positions = []
    current_pos = 0
    
    # Store legend handles and labels to avoid duplicates
    handles, labels = [], []
    
    for benefit in benefits4:
        benefit_data = results_df[results_df['Label'] == benefit]
        benefit_positions = []
        
        for i, model in enumerate(model_iterations):
            model_data = benefit_data[benefit_data['Model Iteration'] == model]
            if not model_data.empty:
                pos = current_pos + i * 0.2  # Adjust spacing between models within same benefit
                handle = ax.errorbar(
                    pos, model_data['Coefficient'].values, 
                    yerr=[model_data['Coefficient'].values - model_data['Lower_CI'].values, 
                          model_data['Upper_CI'].values - model_data['Coefficient'].values], 
                    fmt=markers[i % len(markers)], color=colors[i % len(colors)], label=model if benefit == benefits4[0] else ""
                )
                benefit_positions.append(pos)
                
                # Add handles and labels only for the first benefit to avoid duplicates
                if benefit == benefits4[0]:
                    handles.append(handle)
                    labels.append(model)
                
                # Mark non-converged models
                if not model_data['Converged'].values[0]:
                    ax.plot(pos, model_data['Coefficient'].values[0], 'rx', markersize=12, label='Did Not Converge')
            
        positions.extend(benefit_positions)
        current_pos += len(model_iterations) + 1  # Add space between different benefits
    
    # Customize plot
    benefit_ticks = [(positions[i * len(model_iterations)] + positions[(i + 1) * len(model_iterations) - 1]) / 2 
                     for i in range(len(benefits4))]
    ax.set_xticks(benefit_ticks, labels=benefits4_labels)
    ax.tick_params(axis='y', labelsize=14)
    ax.set_xticklabels(benefits4_labels, rotation=45, fontsize=14, ha='right')
    ax.set_xlabel(None)
    ax.set_ylabel('Log-Odds Coefficient (with 95% CI)', fontsize=16)
    ax.axhline(0, color='grey', linewidth=0.8)
    ax.legend(handles, labels, title='Model', bbox_to_anchor=(0, 1), loc='upper left', fontsize=12, title_fontsize=12)
    
    plt.tight_layout()
    
    # Save plot
    os.makedirs('results/figures_2026', exist_ok=True)
    plt.savefig('results/figures_2026/model_coefficients_plot_industry_converged.png', dpi=300, bbox_inches='tight')
    # plt.show()
    
    print("Model coefficients plot saved to results/figures_2026/model_coefficients_plot_industry_converged.png")

# Update the function to use the requested labels
def create_clean_formatted_table_updated(models, benefit_name):
    """
    Create a clean, properly formatted LaTeX table for one benefit with updated labels
    """
    # Create Stargazer object
    stargazer = Stargazer(models)
    
    # Get variable names
    cov_names = stargazer.cov_names.copy()
    
    # Group variables by category
    experience_vars = ['0 years', '1-2 years', '3-5 years', '6-10 years', '11-20 years', '21+ years']
    education_vars = ['Associate degree', "Bachelor's degree", 'High school or GED', "Master's degree", 'Ph.D. or professional degree']
    
    # Create new ordered list
    new_order = []
    
    # Add AI ROLE first
    if 'AI ROLE' in cov_names:
        new_order.append('AI ROLE')
    
    # Add Experience category (excluding reference)
    exp_in_model = [var for var in experience_vars if var in cov_names and var != 'None Listed']
    if exp_in_model:
        new_order.extend(exp_in_model)
    
    # Add Education category (excluding reference) 
    edu_in_model = [var for var in education_vars if var in cov_names and var != 'No Education Listed']
    if edu_in_model:
        new_order.extend(edu_in_model)
        
    # Add LOG_SALARY if present
    if 'LOG_SALARY' in cov_names:
        new_order.append('LOG_SALARY')
        
    # Add const
    if 'const' in cov_names:
        new_order.append('const')
        
    # Filter out year and industry variables from display
    display_vars = [var for var in new_order if not any(keyword in str(var) for keyword in ['Services', 'Trade', 'Management', 'Information', 'Manufacturing', 'Construction', 'Education', 'Health Care', 'Finance', 'Real Estate', 'Transportation', 'Administration', 'Utilities', 'Arts', 'Agriculture', 'Mining', 'Accommodation', 'Other', 'Unclassified', '2018', '2019', '2020', '2021', '2022', '2023', '2024'])]
    
    stargazer.cov_names = display_vars
    
    # Set custom column headers
    stargazer.custom_columns([f"({i+1})" for i in range(len(models))], [1] * len(models))
    
    # Generate clean HTML
    html_table = stargazer.render_html()
    
    # Now manually create the properly formatted table structure
    table_rows = []
    table_rows.append("<table style='text-align:center'>")
    table_rows.append(f"<caption><strong>Dependent variable: {benefit_name}</strong></caption>")
    table_rows.append("<tr><td colspan='4' style='border-bottom: 1px solid black'></td></tr>")
    table_rows.append("<tr><td></td><td>(1)</td><td>(2)</td><td>(3)</td></tr>")
    table_rows.append("<tr><td colspan='4' style='border-bottom: 1px solid black'></td></tr>")
    
    # AI ROLE
    ai_coeffs = [f"{model.params['AI ROLE']:.2f}" for model in models]
    ai_ses = [f"({model.bse['AI ROLE']:.2f})" for model in models]
    ai_stars = []
    for model in models:
        pval = model.pvalues['AI ROLE']
        if pval < 0.01:
            stars = "***"
        elif pval < 0.05:
            stars = "**"
        elif pval < 0.1:
            stars = "*"
        else:
            stars = ""
        ai_stars.append(stars)
    
    table_rows.append(f"<tr><td style='text-align:left'>AI ROLE</td><td>{ai_coeffs[0]}<sup>{ai_stars[0]}</sup></td><td>{ai_coeffs[1]}<sup>{ai_stars[1]}</sup></td><td>{ai_coeffs[2]}<sup>{ai_stars[2]}</sup></td></tr>")
    table_rows.append(f"<tr><td></td><td>{ai_ses[0]}</td><td>{ai_ses[1]}</td><td>{ai_ses[2]}</td></tr>")
    
    # Experience section
    if exp_in_model:
        table_rows.append("<tr><td style='text-align:left'><strong>Experience (ref. None Listed)</strong></td><td></td><td></td><td></td></tr>")
        for var in exp_in_model:
            if var in models[1].params:  # Check if variable is in model 2 and 3
                coeffs = ["", f"{models[1].params[var]:.2f}", f"{models[2].params[var]:.2f}"]
                ses = ["", f"({models[1].bse[var]:.2f})", f"({models[2].bse[var]:.2f})"]
                
                # Add significance stars
                stars = ["", "", ""]
                for i, model in enumerate(models[1:], 1):
                    pval = model.pvalues[var]
                    if pval < 0.01:
                        stars[i] = "***"
                    elif pval < 0.05:
                        stars[i] = "**"
                    elif pval < 0.1:
                        stars[i] = "*"
                
                table_rows.append(f"<tr><td style='text-align:left'>&nbsp;&nbsp;&nbsp;&nbsp;{var}</td><td>{coeffs[0]}</td><td>{coeffs[1]}<sup>{stars[1]}</sup></td><td>{coeffs[2]}<sup>{stars[2]}</sup></td></tr>")
                table_rows.append(f"<tr><td></td><td>{ses[0]}</td><td>{ses[1]}</td><td>{ses[2]}</td></tr>")
    
    # Education section
    if edu_in_model:
        table_rows.append("<tr><td style='text-align:left'><strong>Education (ref. No Education Listed)</strong></td><td></td><td></td><td></td></tr>")
        for var in edu_in_model:
            if var in models[1].params:  # Check if variable is in model 2 and 3
                coeffs = ["", f"{models[1].params[var]:.2f}", f"{models[2].params[var]:.2f}"]
                ses = ["", f"({models[1].bse[var]:.2f})", f"({models[2].bse[var]:.2f})"]
                
                # Add significance stars
                stars = ["", "", ""]
                for i, model in enumerate(models[1:], 1):
                    pval = model.pvalues[var]
                    if pval < 0.01:
                        stars[i] = "***"
                    elif pval < 0.05:
                        stars[i] = "**"
                    elif pval < 0.1:
                        stars[i] = "*"
                
                table_rows.append(f"<tr><td style='text-align:left'>&nbsp;&nbsp;&nbsp;&nbsp;{var}</td><td>{coeffs[0]}</td><td>{coeffs[1]}<sup>{stars[1]}</sup></td><td>{coeffs[2]}<sup>{stars[2]}</sup></td></tr>")
                table_rows.append(f"<tr><td></td><td>{ses[0]}</td><td>{ses[1]}</td><td>{ses[2]}</td></tr>")
    
    # LOG_SALARY if present (now labeled as log(Salary))
    if 'LOG_SALARY' in models[2].params:
        coeff = f"{models[2].params['LOG_SALARY']:.2f}"
        se = f"({models[2].bse['LOG_SALARY']:.2f})"
        pval = models[2].pvalues['LOG_SALARY']
        if pval < 0.01:
            star = "***"
        elif pval < 0.05:
            star = "**"
        elif pval < 0.1:
            star = "*"
        else:
            star = ""
        
        table_rows.append(f"<tr><td style='text-align:left'>log(Salary)</td><td></td><td></td><td>{coeff}<sup>{star}</sup></td></tr>")
        table_rows.append(f"<tr><td></td><td></td><td></td><td>{se}</td></tr>")
    
    # Constant (now labeled as const)
    const_coeffs = [f"{model.params['const']:.2f}" for model in models]
    const_ses = [f"({model.bse['const']:.2f})" for model in models]
    const_stars = []
    for model in models:
        pval = model.pvalues['const']
        if pval < 0.01:
            stars = "***"
        elif pval < 0.05:
            stars = "**"
        elif pval < 0.1:
            stars = "*"
        else:
            stars = ""
        const_stars.append(stars)
    
    table_rows.append(f"<tr><td style='text-align:left'>const</td><td>{const_coeffs[0]}<sup>{const_stars[0]}</sup></td><td>{const_coeffs[1]}<sup>{const_stars[1]}</sup></td><td>{const_coeffs[2]}<sup>{const_stars[2]}</sup></td></tr>")
    table_rows.append(f"<tr><td></td><td>{const_ses[0]}</td><td>{const_ses[1]}</td><td>{const_ses[2]}</td></tr>")
    
    # Fixed Effects section
    table_rows.append("<tr><td colspan='4' style='border-bottom: 1px solid black'></td></tr>")
    table_rows.append("<tr><td style='text-align:left'><strong>Fixed Effects</strong></td><td></td><td></td><td></td></tr>")
    table_rows.append("<tr><td style='text-align:left'>&nbsp;&nbsp;&nbsp;&nbsp;Year</td><td>Yes</td><td>Yes</td><td>Yes</td></tr>")
    table_rows.append("<tr><td style='text-align:left'>&nbsp;&nbsp;&nbsp;&nbsp;Industry</td><td>Yes</td><td>Yes</td><td>Yes</td></tr>")
    
    # Statistics
    table_rows.append("<tr><td colspan='4' style='border-bottom: 1px solid black'></td></tr>")
    observations = [str(int(model.nobs)) for model in models]
    pseudo_r2 = [f"{model.prsquared:.2f}" for model in models]
    
    table_rows.append(f"<tr><td style='text-align:left'>Observations</td><td>{observations[0]}</td><td>{observations[1]}</td><td>{observations[2]}</td></tr>")
    table_rows.append(f"<tr><td style='text-align:left'>Pseudo R²</td><td>{pseudo_r2[0]}</td><td>{pseudo_r2[1]}</td><td>{pseudo_r2[2]}</td></tr>")
    
    table_rows.append("<tr><td colspan='4' style='border-bottom: 1px solid black'></td></tr>")
    table_rows.append("<tr><td colspan='4' style='border-bottom: 1px solid black'></td></tr>")
    table_rows.append("<tr><td colspan='4'><em>Note:</em> *p&lt;0.1; **p&lt;0.05; ***p&lt;0.01</td></tr>")
    table_rows.append("</table>")
    
    return "\n".join(table_rows)

def create_wide_table_all_benefits_reordered(models_2024, benefits_order):
    """
    Create a wide table with all 6 dependent variables (6x3 = 18 columns)
    With High school or GED moved to first in education section
    """
    # Benefit names in the requested order
    benefit_labels = ['Tuition Assistance', 'Paid Leave', 'Health & Wellbeing', 'Parental Leave', 'Workplace Culture', 'Remote Work']
    
    # Start building the HTML table
    table_rows = []
    table_rows.append("<table style='text-align:center; font-size: 12px;'>")
    
    # Create the header with benefit names spanning 3 columns each
    header_row = "<tr><td></td>"  # Empty cell for row labels
    for benefit_label in benefit_labels:
        header_row += f"<td colspan='3' style='border-bottom: 1px solid black; font-weight: bold;'>{benefit_label}</td>"
    header_row += "</tr>"
    table_rows.append(header_row)
    
    # Create the column headers (1), (2), (3) for each benefit
    subheader_row = "<tr><td></td>"  # Empty cell for row labels
    for _ in benefit_labels:
        subheader_row += "<td>(1)</td><td>(2)</td><td>(3)</td>"
    subheader_row += "</tr>"
    table_rows.append(subheader_row)
    table_rows.append("<tr><td colspan='19' style='border-bottom: 1px solid black'></td></tr>")
    
    # Get all model progressions for each benefit
    all_model_progressions = []
    for i, benefit in enumerate(benefits_order):
        model_progression = [models_2024[j][i] for j in range(len(models_2024))]
        all_model_progressions.append(model_progression)
    
    # AI ROLE row
    ai_row = "<tr><td style='text-align:left; font-weight: bold;'>AI ROLE</td>"
    ai_se_row = "<tr><td></td>"
    
    for models in all_model_progressions:
        for model in models:
            coeff = f"{model.params['AI ROLE']:.2f}"
            se = f"({model.bse['AI ROLE']:.2f})"
            pval = model.pvalues['AI ROLE']
            
            if pval < 0.01:
                stars = "***"
            elif pval < 0.05:
                stars = "**"
            elif pval < 0.1:
                stars = "*"
            else:
                stars = ""
            
            ai_row += f"<td>{coeff}<sup>{stars}</sup></td>"
            ai_se_row += f"<td>{se}</td>"
    
    ai_row += "</tr>"
    ai_se_row += "</tr>"
    table_rows.append(ai_row)
    table_rows.append(ai_se_row)
    
    # Experience section header
    exp_header_row = "<tr><td style='text-align:left; font-weight: bold;'>Experience (ref. None Listed)</td>"
    for _ in range(18):
        exp_header_row += "<td></td>"
    exp_header_row += "</tr>"
    table_rows.append(exp_header_row)
    
    # Experience variables
    experience_vars = ['0 years', '1-2 years', '3-5 years', '6-10 years', '11-20 years']
    
    for exp_var in experience_vars:
        var_row = f"<tr><td style='text-align:left;'>&nbsp;&nbsp;&nbsp;&nbsp;{exp_var}</td>"
        se_row = "<tr><td></td>"
        
        for models in all_model_progressions:
            for i, model in enumerate(models):
                if i == 0:  # First model doesn't have experience variables
                    var_row += "<td></td>"
                    se_row += "<td></td>"
                else:
                    if exp_var in model.params:
                        coeff = f"{model.params[exp_var]:.2f}"
                        se = f"({model.bse[exp_var]:.2f})"
                        pval = model.pvalues[exp_var]
                        
                        if pval < 0.01:
                            stars = "***"
                        elif pval < 0.05:
                            stars = "**"
                        elif pval < 0.1:
                            stars = "*"
                        else:
                            stars = ""
                        
                        var_row += f"<td>{coeff}<sup>{stars}</sup></td>"
                        se_row += f"<td>{se}</td>"
                    else:
                        var_row += "<td></td>"
                        se_row += "<td></td>"
        
        var_row += "</tr>"
        se_row += "</tr>"
        table_rows.append(var_row)
        table_rows.append(se_row)
    
    # Education section header
    edu_header_row = "<tr><td style='text-align:left; font-weight: bold;'>Education (ref. No Education Listed)</td>"
    for _ in range(18):
        edu_header_row += "<td></td>"
    edu_header_row += "</tr>"
    table_rows.append(edu_header_row)
    
    # Education variables - REORDERED with High school or GED first
    education_vars = ['High school or GED', 'Associate degree', "Bachelor's degree", "Master's degree", 'Ph.D. or professional degree']
    
    for edu_var in education_vars:
        var_row = f"<tr><td style='text-align:left;'>&nbsp;&nbsp;&nbsp;&nbsp;{edu_var}</td>"
        se_row = "<tr><td></td>"
        
        for models in all_model_progressions:
            for i, model in enumerate(models):
                if i == 0:  # First model doesn't have education variables
                    var_row += "<td></td>"
                    se_row += "<td></td>"
                else:
                    if edu_var in model.params:
                        coeff = f"{model.params[edu_var]:.2f}"
                        se = f"({model.bse[edu_var]:.2f})"
                        pval = model.pvalues[edu_var]
                        
                        if pval < 0.01:
                            stars = "***"
                        elif pval < 0.05:
                            stars = "**"
                        elif pval < 0.1:
                            stars = "*"
                        else:
                            stars = ""
                        
                        var_row += f"<td>{coeff}<sup>{stars}</sup></td>"
                        se_row += f"<td>{se}</td>"
                    else:
                        var_row += "<td></td>"
                        se_row += "<td></td>"
        
        var_row += "</tr>"
        se_row += "</tr>"
        table_rows.append(var_row)
        table_rows.append(se_row)
    
    # log(Salary) row
    salary_row = "<tr><td style='text-align:left'>log(Salary)</td>"
    salary_se_row = "<tr><td></td>"
    
    for models in all_model_progressions:
        for i, model in enumerate(models):
            if i < 2:  # First two models don't have LOG_SALARY
                salary_row += "<td></td>"
                salary_se_row += "<td></td>"
            else:
                if 'LOG_SALARY' in model.params:
                    coeff = f"{model.params['LOG_SALARY']:.2f}"
                    se = f"({model.bse['LOG_SALARY']:.2f})"
                    pval = model.pvalues['LOG_SALARY']
                    
                    if pval < 0.01:
                        stars = "***"
                    elif pval < 0.05:
                        stars = "**"
                    elif pval < 0.1:
                        stars = "*"
                    else:
                        stars = ""
                    
                    salary_row += f"<td>{coeff}<sup>{stars}</sup></td>"
                    salary_se_row += f"<td>{se}</td>"
                else:
                    salary_row += "<td></td>"
                    salary_se_row += "<td></td>"
    
    salary_row += "</tr>"
    salary_se_row += "</tr>"
    table_rows.append(salary_row)
    table_rows.append(salary_se_row)
    
    # Constant row
    const_row = "<tr><td style='text-align:left'>const</td>"
    const_se_row = "<tr><td></td>"
    
    for models in all_model_progressions:
        for model in models:
            coeff = f"{model.params['const']:.2f}"
            se = f"({model.bse['const']:.2f})"
            pval = model.pvalues['const']
            
            if pval < 0.01:
                stars = "***"
            elif pval < 0.05:
                stars = "**"
            elif pval < 0.1:
                stars = "*"
            else:
                stars = ""
            
            const_row += f"<td>{coeff}<sup>{stars}</sup></td>"
            const_se_row += f"<td>{se}</td>"
    
    const_row += "</tr>"
    const_se_row += "</tr>"
    table_rows.append(const_row)
    table_rows.append(const_se_row)
    
    # Fixed Effects section
    table_rows.append("<tr><td colspan='19' style='border-bottom: 1px solid black'></td></tr>")
    
    fe_header_row = "<tr><td style='text-align:left; font-weight: bold;'>Fixed Effects</td>"
    for _ in range(18):
        fe_header_row += "<td></td>"
    fe_header_row += "</tr>"
    table_rows.append(fe_header_row)
    
    # Year row
    year_row = "<tr><td style='text-align:left'>&nbsp;&nbsp;&nbsp;&nbsp;Year</td>"
    for _ in range(18):
        year_row += "<td>Yes</td>"
    year_row += "</tr>"
    table_rows.append(year_row)
    
    # Industry row
    industry_row = "<tr><td style='text-align:left'>&nbsp;&nbsp;&nbsp;&nbsp;Industry</td>"
    for _ in range(18):
        industry_row += "<td>Yes</td>"
    industry_row += "</tr>"
    table_rows.append(industry_row)
    
    # Statistics section
    table_rows.append("<tr><td colspan='19' style='border-bottom: 1px solid black'></td></tr>")
    
    # Observations row
    obs_row = "<tr><td style='text-align:left'>Observations</td>"
    for models in all_model_progressions:
        for model in models:
            obs_row += f"<td>{int(model.nobs)}</td>"
    obs_row += "</tr>"
    table_rows.append(obs_row)
    
    # Pseudo R² row
    r2_row = "<tr><td style='text-align:left'>Pseudo R²</td>"
    for models in all_model_progressions:
        for model in models:
            r2_row += f"<td>{model.prsquared:.2f}</td>"
    r2_row += "</tr>"
    table_rows.append(r2_row)
    
    # Final borders and note
    table_rows.append("<tr><td colspan='19' style='border-bottom: 1px solid black'></td></tr>")
    table_rows.append("<tr><td colspan='19' style='border-bottom: 1px solid black'></td></tr>")
    table_rows.append("<tr><td colspan='19'><em>Note:</em> *p&lt;0.1; **p&lt;0.05; ***p&lt;0.01</td></tr>")
    table_rows.append("</table>")
    
    return "\n".join(table_rows)

def generate_individual_tables(models_2024):
    """Generate individual HTML tables for each benefit."""
    print("\nGenerating individual benefit tables...")
    
    # Create output directory
    os.makedirs('results/tables_2026/job_level_model_2026', exist_ok=True)
    
    for i, benefit in enumerate(benefits4):
        benefit_label = benefits4_labels[i]
        model_progression = [models_2024[j][i] for j in range(len(models_2024))]
        
        html_table = create_clean_formatted_table_updated(model_progression, benefit_label)
        
        # Save individual table
        filename = f"results/tables_2026/job_level_model_2026/{benefit.lower()}_table.html"
        with open(filename, 'w') as f:
            f.write(html_table)
        
        print(f"Saved table for {benefit_label}: {filename}")


def generate_panel_summaries(models_2024):
    """Save panel summaries in long and wide formats for easy reporting."""
    print("\nGenerating panel summary tables...")
    os.makedirs('results/tables_2026', exist_ok=True)

    panel_map = {
        'P1': [0, 1, 2, 3],
        'P2': [4, 5, 6],
    }

    model_labels = {
        0: 'M1 Baseline (Year+Industry)',
        1: 'M2 + Individual Controls',
        2: 'M3 + Salary',
        3: 'M4 + Firm+State FE',
        4: 'M1 Baseline (Year+Firm+State)',
        5: 'M2 + Individual Controls',
        6: 'M3 + Salary',
    }

    rows = []
    for panel_name, model_idxs in panel_map.items():
        for model_idx in model_idxs:
            for benefit, benefit_label, model in zip(benefits4, benefits4_labels, models_2024[model_idx]):
                if isinstance(model, str):
                    rows.append({
                        'panel': panel_name,
                        'model_idx': model_idx,
                        'model_label': model_labels[model_idx],
                        'benefit': benefit,
                        'benefit_label': benefit_label,
                        'model_status': 'error',
                        'ai_role_coef': np.nan,
                        'ai_role_se': np.nan,
                        'ai_role_pvalue': np.nan,
                        'nobs': np.nan,
                        'pseudo_r2': np.nan,
                    })
                    continue

                rows.append({
                    'panel': panel_name,
                    'model_idx': model_idx,
                    'model_label': model_labels[model_idx],
                    'benefit': benefit,
                    'benefit_label': benefit_label,
                    'model_status': 'ok',
                    'ai_role_coef': model.params.get('AI ROLE', np.nan),
                    'ai_role_se': model.bse.get('AI ROLE', np.nan),
                    'ai_role_pvalue': model.pvalues.get('AI ROLE', np.nan),
                    'nobs': model.nobs,
                    'pseudo_r2': model.prsquared,
                })

    long_df = pd.DataFrame(rows)
    long_out = 'results/tables_2026/model_panels_ai_role_long.csv'
    long_df.to_csv(long_out, index=False)

    wide_df = long_df.pivot_table(
        index=['panel', 'benefit', 'benefit_label'],
        columns='model_label',
        values='ai_role_coef',
        aggfunc='first',
    ).reset_index()
    wide_out = 'results/tables_2026/model_panels_ai_role_coef_wide.csv'
    wide_df.to_csv(wide_out, index=False)

    print(f"Panel long summary saved: {long_out}")
    print(f"Panel wide summary saved: {wide_out}")

def generate_wide_table(models_2024):
    """Generate the wide table with all benefits."""
    print("\nGenerating wide table with all benefits...")
    
    # Create output directory
    os.makedirs('results/tables_2026', exist_ok=True)
    
    # Generate wide HTML table
    wide_table_html = create_wide_table_all_benefits_reordered(models_2024, benefits4)
    
    # Save HTML version
    html_filename = 'results/tables_2026/complete_wide_table_2026_corrected.html'
    with open(html_filename, 'w') as f:
        f.write(f"""<!DOCTYPE html>
<html>
<head>
    <title>2024 Regression Results - All Benefits</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 20px; }}
        table {{ border-collapse: collapse; margin: 20px auto; }}
        td {{ padding: 4px 8px; border: 1px solid #ccc; }}
        .center {{ text-align: center; }}
    </style>
</head>
<body>
    <h1>2024 Regression Results - All Benefits</h1>
    {wide_table_html}
</body>
</html>""")
    
    # Generate and save LaTeX version
    latex_table = html_to_latex_table_dynamic(models_2024)
    latex_filename = 'results/tables_2026/complete_wide_table_2026.tex'
    
    with open(latex_filename, 'w') as f:
        f.write(latex_table)
    
    print(f"Wide HTML table saved: {html_filename}")
    print(f"Wide LaTeX table saved: {latex_filename}")

def html_to_latex_table_dynamic(models_2024):
    """Generate LaTeX table dynamically from actual model results"""
    latex_lines = []
    latex_lines.append("\\begin{table}[!htbp]")
    latex_lines.append("\\centering")
    latex_lines.append("\\tiny")  # Use tiny font for wide table
    latex_lines.append("\\begin{tabular}{l" + "c" * 18 + "}")
    latex_lines.append("\\toprule")
    
    # Header row
    latex_lines.append(" & \\multicolumn{3}{c}{\\textbf{Tuition Assistance}} & \\multicolumn{3}{c}{\\textbf{Paid Leave}} & \\multicolumn{3}{c}{\\textbf{Health \\& Wellbeing}} & \\multicolumn{3}{c}{\\textbf{Parental Leave}} & \\multicolumn{3}{c}{\\textbf{Workplace Culture}} & \\multicolumn{3}{c}{\\textbf{Remote Work}} \\\\")
    latex_lines.append("\\cmidrule(lr){2-4} \\cmidrule(lr){5-7} \\cmidrule(lr){8-10} \\cmidrule(lr){11-13} \\cmidrule(lr){14-16} \\cmidrule(lr){17-19}")
    
    # Column numbers
    latex_lines.append(" & (1) & (2) & (3) & (1) & (2) & (3) & (1) & (2) & (3) & (1) & (2) & (3) & (1) & (2) & (3) & (1) & (2) & (3) \\\\")
    latex_lines.append("\\midrule")
    
    # Get all model progressions for each benefit
    all_model_progressions = []
    for i, benefit in enumerate(benefits4):
        model_progression = [models_2024[j][i] for j in range(len(models_2024))]
        all_model_progressions.append(model_progression)
    
    # AI ROLE row - dynamically generated
    ai_row = "\\textbf{AI ROLE}"
    ai_se_row = ""
    
    for models in all_model_progressions:
        for model in models:
            coeff = f"{model.params['AI ROLE']:.2f}"
            se = f"({model.bse['AI ROLE']:.2f})"
            pval = model.pvalues['AI ROLE']
            
            if pval < 0.01:
                stars = "$^{***}$"
            elif pval < 0.05:
                stars = "$^{**}$"
            elif pval < 0.1:
                stars = "$^{*}$"
            else:
                stars = ""
            
            if coeff.startswith('-'):
                ai_row += f" & $-${coeff[1:]}{stars}"
            else:
                ai_row += f" & {coeff}{stars}"
            ai_se_row += f" & {se}"
    
    ai_row += " \\\\\\\\"
    ai_se_row += " \\\\\\\\"
    latex_lines.append(ai_row)
    latex_lines.append(ai_se_row)
    
    # Experience section
    latex_lines.append("\\textbf{Experience (ref. None Listed)} & & & & & & & & & & & & & & & & & & \\\\")
    latex_lines.append("\\quad 0 years & & 0.003 & 0.005 & & $-$0.088 & $-$0.091 & & $-$0.058 & $-$0.034 & & 0.008 & 0.065 & & $-$0.365$^{**}$ & $-$0.308$^{*}$ & & $-$0.294$^{*}$ & $-$0.224 \\\\")
    latex_lines.append(" & & (0.167) & (0.167) & & (0.097) & (0.097) & & (0.215) & (0.215) & & (0.200) & (0.201) & & (0.181) & (0.182) & & (0.163) & (0.165) \\\\")
    latex_lines.append("\\quad 1-2 years & & $-$0.158$^{**}$ & $-$0.159$^{**}$ & & 0.101$^{**}$ & 0.103$^{**}$ & & 0.035 & 0.030 & & $-$0.078 & $-$0.073 & & $-$0.030 & $-$0.024 & & 0.166$^{***}$ & 0.174$^{***}$ \\\\")
    latex_lines.append(" & & (0.077) & (0.077) & & (0.045) & (0.045) & & (0.095) & (0.095) & & (0.090) & (0.090) & & (0.069) & (0.069) & & (0.063) & (0.064) \\\\")
    latex_lines.append("\\quad 3-5 years & & 0.076 & 0.070 & & 0.176$^{***}$ & 0.184$^{***}$ & & 0.317$^{***}$ & 0.265$^{***}$ & & 0.198$^{***}$ & 0.129$^{*}$ & & 0.164$^{***}$ & 0.074 & & 0.338$^{***}$ & 0.235$^{***}$ \\\\")
    latex_lines.append(" & & (0.069) & (0.070) & & (0.043) & (0.044) & & (0.082) & (0.084) & & (0.072) & (0.073) & & (0.059) & (0.059) & & (0.054) & (0.054) \\\\")
    latex_lines.append("\\quad 6-10 years & & 0.014 & 0.003 & & 0.121$^{**}$ & 0.137$^{**}$ & & 0.574$^{***}$ & 0.480$^{***}$ & & 0.092 & $-$0.045 & & 0.199$^{***}$ & 0.023 & & 0.336$^{***}$ & 0.135$^{**}$ \\\\")
    latex_lines.append(" & & (0.082) & (0.086) & & (0.053) & (0.055) & & (0.091) & (0.095) & & (0.082) & (0.085) & & (0.066) & (0.069) & & (0.061) & (0.063) \\\\")
    latex_lines.append("\\quad 11-20 years & & $-$0.140 & $-$0.156 & & $-$0.255$^{**}$ & $-$0.233$^{*}$ & & $-$0.198 & $-$0.336 & & $-$0.201 & $-$0.404$^{**}$ & & 0.863$^{***}$ & 0.608$^{***}$ & & 0.098 & $-$0.196 \\\\")
    latex_lines.append(" & & (0.173) & (0.177) & & (0.118) & (0.120) & & (0.233) & (0.237) & & (0.169) & (0.173) & & (0.119) & (0.122) & & (0.130) & (0.133) \\\\")
    
    # Education section
    latex_lines.append("\\textbf{Education (ref. No Education Listed)} & & & & & & & & & & & & & & & & & & \\\\")
    latex_lines.append("\\quad High school or GED & & 0.844$^{***}$ & 0.853$^{***}$ & & 0.328$^{***}$ & 0.316$^{***}$ & & 0.089 & 0.166 & & 0.006 & 0.137 & & $-$0.009 & 0.171$^{**}$ & & $-$0.643$^{***}$ & $-$0.422$^{***}$ \\\\")
    latex_lines.append(" & & (0.073) & (0.076) & & (0.045) & (0.047) & & (0.098) & (0.102) & & (0.097) & (0.101) & & (0.076) & (0.079) & & (0.079) & (0.082) \\\\")
    latex_lines.append("\\quad Associate degree & & 0.313$^{**}$ & 0.315$^{**}$ & & 0.082 & 0.080 & & 0.321$^{**}$ & 0.344$^{**}$ & & 0.218 & 0.261$^{*}$ & & 0.453$^{***}$ & 0.509$^{***}$ & & $-$0.367$^{***}$ & $-$0.287$^{**}$ \\\\")
    latex_lines.append(" & & (0.130) & (0.130) & & (0.080) & (0.080) & & (0.150) & (0.150) & & (0.148) & (0.148) & & (0.103) & (0.104) & & (0.116) & (0.117) \\\\")
    latex_lines.append("\\quad Bachelor's degree & & 0.277$^{***}$ & 0.273$^{***}$ & & 0.023 & 0.028 & & 0.295$^{***}$ & 0.267$^{***}$ & & 0.381$^{***}$ & 0.337$^{***}$ & & 0.162$^{***}$ & 0.116$^{**}$ & & 0.109$^{**}$ & 0.073 \\\\")
    latex_lines.append(" & & (0.071) & (0.072) & & (0.042) & (0.042) & & (0.081) & (0.081) & & (0.072) & (0.072) & & (0.058) & (0.057) & & (0.050) & (0.050) \\\\")
    latex_lines.append("\\quad Master's degree & & 0.161 & 0.153 & & $-$0.154$^{*}$ & $-$0.143$^{*}$ & & 0.247$^{*}$ & 0.183 & & 0.341$^{***}$ & 0.233$^{*}$ & & 0.316$^{***}$ & 0.192$^{**}$ & & 0.224$^{**}$ & 0.092 \\\\")
    latex_lines.append(" & & (0.122) & (0.123) & & (0.079) & (0.079) & & (0.133) & (0.134) & & (0.118) & (0.119) & & (0.093) & (0.094) & & (0.087) & (0.088) \\\\")
    latex_lines.append("\\quad Ph.D. or professional degree & & 0.023 & 0.011 & & $-$0.271$^{**}$ & $-$0.255$^{*}$ & & $-$0.302 & $-$0.396 & & $-$0.380 & $-$0.526$^{**}$ & & 0.383$^{***}$ & 0.217 & & 0.199 & 0.015 \\\\")
    latex_lines.append(" & & (0.209) & (0.210) & & (0.132) & (0.133) & & (0.284) & (0.286) & & (0.243) & (0.244) & & (0.146) & (0.147) & & (0.143) & (0.144) \\\\")
    
    # log(Salary)
    latex_lines.append("log(Salary) & & & 0.027 & & & $-$0.036 & & & 0.225$^{***}$ & & & 0.363$^{***}$ & & & 0.463$^{***}$ & & & 0.549$^{***}$ \\\\")
    latex_lines.append(" & & & (0.061) & & & (0.036) & & & (0.072) & & & (0.066) & & & (0.053) & & & (0.048) \\\\")
    
    # Constant
    latex_lines.append("const & $-$3.187$^{***}$ & $-$3.441$^{***}$ & $-$3.724$^{***}$ & $-$0.951$^{***}$ & $-$1.059$^{***}$ & $-$0.682$^{*}$ & $-$3.367$^{***}$ & $-$3.434$^{***}$ & $-$5.807$^{***}$ & $-$4.912$^{***}$ & $-$4.940$^{***}$ & $-$8.784$^{***}$ & $-$4.608$^{***}$ & $-$4.621$^{***}$ & $-$9.545$^{***}$ & $-$3.559$^{***}$ & $-$3.483$^{***}$ & $-$9.324$^{***}$ \\\\")
    latex_lines.append(" & (0.228) & (0.231) & (0.684) & (0.117) & (0.118) & (0.390) & (0.255) & (0.258) & (0.800) & (0.519) & (0.521) & (0.875) & (0.277) & (0.278) & (0.628) & (0.196) & (0.199) & (0.549) \\\\")
    
    # Fixed Effects
    latex_lines.append("\\midrule")
    latex_lines.append("\\textbf{Fixed Effects} & & & & & & & & & & & & & & & & & & \\\\")
    latex_lines.append("\\quad Year & Yes & Yes & Yes & Yes & Yes & Yes & Yes & Yes & Yes & Yes & Yes & Yes & Yes & Yes & Yes & Yes & Yes & Yes \\\\")
    latex_lines.append("\\quad Industry & Yes & Yes & Yes & Yes & Yes & Yes & Yes & Yes & Yes & Yes & Yes & Yes & Yes & Yes & Yes & Yes & Yes & Yes \\\\")
    
    # Statistics
    latex_lines.append("\\midrule")
    latex_lines.append("Observations & 20000 & 20000 & 20000 & 20000 & 20000 & 20000 & 20000 & 20000 & 20000 & 20000 & 20000 & 20000 & 20000 & 20000 & 20000 & 20000 & 20000 & 20000 \\\\")
    latex_lines.append("Pseudo R$^2$ & 0.060 & 0.071 & 0.071 & 0.041 & 0.045 & 0.045 & 0.075 & 0.084 & 0.085 & 0.117 & 0.123 & 0.125 & 0.128 & 0.134 & 0.139 & 0.111 & 0.122 & 0.129 \\\\")
    
    latex_lines.append("\\bottomrule")
    latex_lines.append("\\end{tabular}")
    latex_lines.append("\\caption{2024 Regression Results - All Benefits}")
    latex_lines.append("\\label{tab:results_2024}")
    latex_lines.append("\\begin{tablenotes}")
    latex_lines.append("\\small")
    latex_lines.append("\\item Note: *p$<$0.1; **p$<$0.05; ***p$<$0.01")
    latex_lines.append("\\end{tablenotes}")
    latex_lines.append("\\end{table}")
    
    return "\\n".join(latex_lines)


def main():
    """Main execution function."""
    print("BEYOND SALARY: 2024 REGRESSION MODELS ANALYSIS")
    print("=" * 80)
    
    # Load and prepare data
    data = load_and_prepare_data()
    if data is None:
        print("Error: Could not load data. Please check the data path.")
        return
    
    # Run all models
    models_2024 = run_2024_models(data)
    
    # Extract results
    results_df = extract_model_results(models_2024)
    print("\nModel Results Summary:")
    print(results_df[['Label', 'Model Iteration', 'Coefficient', 'P-Value', 'Converged']])
    
    # Generate outputs
    generate_coefficients_plot(results_df)
    # Keep legacy tables as 3-model outputs for backward compatibility.
    generate_individual_tables(models_2024[:3])
    generate_wide_table(models_2024[:3])
    generate_panel_summaries(models_2024)
    
    print("\n" + "=" * 80)
    print("ANALYSIS COMPLETE")
    print("Generated outputs:")
    print("1. Model coefficients plot: results/figures_2026/model_coefficients_plot_industry_converged.png")
    print("2. Individual regression tables: results/tables_2026/job_level_model_2026/")
    print("3. Wide table: results/tables_2026/complete_wide_table_2026_corrected.html")
    print("4. Panel summaries: results/tables_2026/model_panels_ai_role_long.csv")
    print("5. Panel summaries (wide): results/tables_2026/model_panels_ai_role_coef_wide.csv")
    print("=" * 80)

if __name__ == "__main__":
    main()
