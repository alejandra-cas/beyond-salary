#!/usr/bin/env python3
"""
Regression Models for 2024 Analysis (Including Data up to 2024)

This script runs the logistic regression models for the 2024 paper analysis,
preserving all original model logic including:
- PARENTAL_LEAVE models exclude 2018 data
- Industry grouping for small categories
- Proper reference category handling
- All fixed effects and control variables

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

# Configuration
plt.rcParams.update({'font.size': 14})

# Field mappings
region = 'STATE_NAME'
industry = 'NAICS_2022_2_NAME'
education = 'MIN_EDULEVELS_NAME'
year = 'YEAR'
occupation = 'SOC_2021_2_NAME'
experience = 'EXPERIENCE_BUCKET'

# Benefits to analyze (2024 models)
benefits4 = ['EDU_ASSISTANCE', 'PAID_LEAVE', 'HEALTH_WELLBEING', 'PARENTAL_LEAVE', 'CULTURE', 'REMOTE_KW']
benefits4_labels = ['Tuition Assistance', 'Paid Leave', 'Health and Wellbeing', 'Parental Leave', 'Workplace Culture', 'Remote Work']

# Color scheme for plots
colors = ['#E69F00', '#56B4E9', '#009E73', '#CC79A7']

def load_and_prepare_data():
    """Load and prepare the 2024 dataset with all preprocessing."""
    print("Loading 2024 dataset...")
    
    # Load the data (adjust path as needed)
    data_path = '../data/2024_salary_sample.parquet.gzip'
    if not os.path.exists(data_path):
        print(f"Warning: {data_path} not found. Please update the path.")
        return None
    
    even_sample = pd.read_parquet(data_path)
    
    # Create experience buckets
    print("Creating experience buckets...")
    bins = [-2, -1, 0, 2, 5, 10, 20, 100]
    labels = ['Missing', '0 years', '1-2 years', '3-5 years', '6-10 years', '11-20 years', '21+ years']
    even_sample['EXPERIENCE_BUCKET'] = pd.cut(even_sample['MIN_YEARS_EXPERIENCE'], bins=bins, labels=labels, right=True)
    even_sample['EXPERIENCE_BUCKET'] = even_sample['EXPERIENCE_BUCKET'].astype(str)
    even_sample['EXPERIENCE_BUCKET'] = even_sample['EXPERIENCE_BUCKET'].replace('nan', 'None Listed')
    
    # Create log salary
    print("Creating log salary...")
    even_sample['LOG_SALARY'] = np.log(even_sample['SALARY'])
    
    # Replace small industries with "Other" (this is also done in run_logit_model but we do it here for consistency)
    print("Processing industry categories...")
    industry_counts = even_sample['NAICS_2022_2_NAME'].value_counts()
    small_industries = industry_counts[industry_counts < 30].index  
    even_sample['NAICS_2022_2_NAME'] = even_sample['NAICS_2022_2_NAME'].replace(small_industries, 'Other')
    
    print(f"Final dataset shape: {even_sample.shape}")
    print(f"Year distribution:")
    print(even_sample.groupby('YEAR').size())
    
    return even_sample

def run_2024_models(even_sample):
    """Run all three model specifications for each benefit following original logic."""
    
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
        model = run_logit_model(even_sample, dependent=benefit, predictor='AI ROLE', 
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
        model = run_logit_model(even_sample, dependent=benefit, predictor='AI ROLE', 
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
        model = run_logit_model(even_sample, dependent=benefit, predictor='AI ROLE', 
                              cat_controls=[year, industry, education, experience], 
                              cont_controls=['LOG_SALARY'], 
                              ref_category={education: "No Education Listed", experience: 'None Listed'})
        benefit_models_industry_3.append(model)
    
    return [benefit_models_industry, benefit_models_industry_2, benefit_models_industry_3]

def extract_model_results(models_2024):
    """Extract coefficients, standard errors, p-values, and model statistics."""
    results_dfs = []
    
    model_names = ['Baseline', 'Individual Controls', 'With Salary']
    
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
    markers = ['o', 's', '^']  # Different markers for model iterations
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
                    fmt=markers[i], color=colors[i], label=model if benefit == benefits4[0] else ""
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
    os.makedirs('../results/figures', exist_ok=True)
    plt.savefig('../results/figures/model_coefficients_plot_industry_converged.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print("Model coefficients plot saved to ../results/figures/model_coefficients_plot_industry_converged.png")

def generate_individual_tables(models_2024):
    """Generate properly formatted individual regression tables for each benefit."""
    print("\nGenerating individual regression tables...")
    
    try:
        from stargazer.stargazer import Stargazer
    except ImportError:
        print("Warning: stargazer not available. Skipping table generation.")
        return
    
    # Create output directory
    table_dir = '../results/tables/job_level_model_2025'
    os.makedirs(table_dir, exist_ok=True)
    
    # Fixed effects setup
    fixed_effects_variables = [year, industry]
    fixed_effects_var_dict = {var: list(set(pd.read_parquet('../data/2024_salary_sample.parquet.gzip')[var])) 
                             for var in fixed_effects_variables}
    
    cat_var_dict = {}
    for category, values in fixed_effects_var_dict.items():
        for value in values:
            cat_var_dict[value] = category
    
    fixed_effects_categories = [item for sublist in fixed_effects_var_dict.values() for item in sublist]
    
    def create_table(model_progression, benefit_name):
        """Create a formatted table for one benefit."""
        stargazer = Stargazer(model_progression)
        
        # Customize variable order
        cov_names = stargazer.cov_names
        if 'AI ROLE' in cov_names:
            cov_names.remove('AI ROLE')
            cov_names.insert(0, 'AI ROLE')
        stargazer.cov_names = cov_names
        
        # Remove fixed effects variables from display
        fixed_effects_list = [cat for cat in fixed_effects_categories if cat in stargazer.cov_names]
        for covariate in fixed_effects_list:
            if covariate in stargazer.cov_names:
                stargazer.cov_names.remove(covariate)
        
        # Add fixed effects indicators
        stargazer.add_line("\\textbf{Fixed Effects}", [""] * len(model_progression))
        
        model_fixed_effects_dict = defaultdict(list)
        for i, model in enumerate(model_progression): 
            model_fixed_effects_dict[i] = list(set([value for key, value in cat_var_dict.items() 
                                                  if key in model.model.exog_names]))
        
        fixed_effects_names = {'NAICS_2022_2_NAME': 'Industry', 'YEAR': 'Year'}
        
        for variable in fixed_effects_variables:
            variable_name = fixed_effects_names[variable]
            inclusion_list = ['Yes' if variable in model_fixed_effects_dict[model] else 'Yes' 
                            for model in model_fixed_effects_dict]
            stargazer.add_line(f"{variable_name}", inclusion_list)
        
        stargazer.significance_levels([0.1, 0.05, 0.01])
        
        # Generate LaTeX
        latex_table = stargazer.render_latex()
        
        # Save table
        filename = f'{benefit_name}_reg_table.tex'
        filepath = os.path.join(table_dir, filename)
        with open(filepath, 'w') as f:
            f.write(latex_table)
        
        print(f"  Saved {filename}")
        
        return latex_table
    
    # Generate table for each benefit
    for i, benefit in enumerate(benefits4):
        print(f"  Generating table for {benefit}...")
        model_progression = [models[i] for models in models_2024]
        create_table(model_progression, benefit)
    
    print(f"Individual tables saved to {table_dir}/")

def generate_wide_table(models_2024):
    """Generate the final wide table with all benefits and models."""
    print("\nGenerating wide table...")
    
    def create_wide_table_html(models_2024, benefits_order):
        """Create a wide HTML table with all 6 dependent variables (6x3 = 18 columns)."""
        
        table_rows = []
        table_rows.append("<table style='text-align:center; font-size: 12px; border-collapse: collapse;'>")
        
        # Create header with benefit names spanning 3 columns each
        header_row = "<tr><td style='border-bottom: 2px solid black;'></td>"  # Empty cell for row labels
        for benefit_label in benefits4_labels:
            header_row += f"<td colspan='3' style='border-bottom: 2px solid black; font-weight: bold; text-align: center;'>{benefit_label}</td>"
        header_row += "</tr>"
        table_rows.append(header_row)
        
        # Create column headers (1), (2), (3) for each benefit
        subheader_row = "<tr><td style='border-bottom: 1px solid black;'></td>"  # Empty cell for row labels
        for _ in benefits4_labels:
            subheader_row += "<td style='border-bottom: 1px solid black;'>(1)</td><td style='border-bottom: 1px solid black;'>(2)</td><td style='border-bottom: 1px solid black;'>(3)</td>"
        subheader_row += "</tr>"
        table_rows.append(subheader_row)
        
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
                coeff = f"{model.params['AI ROLE']:.3f}"
                se = f"({model.bse['AI ROLE']:.3f})"
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
                ai_se_row += f"<td style='font-style: italic;'>{se}</td>"
        
        ai_row += "</tr>"
        ai_se_row += "</tr>"
        table_rows.append(ai_row)
        table_rows.append(ai_se_row)
        
        # Fixed Effects section
        table_rows.append("<tr><td colspan='19' style='border-top: 1px solid black;'></td></tr>")
        
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
        
        # Individual Controls row
        controls_row = "<tr><td style='text-align:left'>&nbsp;&nbsp;&nbsp;&nbsp;Individual Controls</td>"
        for models in all_model_progressions:
            controls_row += "<td>No</td><td>Yes</td><td>Yes</td>"
        controls_row += "</tr>"
        table_rows.append(controls_row)
        
        # Salary Controls row
        salary_row = "<tr><td style='text-align:left'>&nbsp;&nbsp;&nbsp;&nbsp;Salary Controls</td>"
        for models in all_model_progressions:
            salary_row += "<td>No</td><td>No</td><td>Yes</td>"
        salary_row += "</tr>"
        table_rows.append(salary_row)
        
        # Statistics section
        table_rows.append("<tr><td colspan='19' style='border-top: 1px solid black;'></td></tr>")
        
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
                r2_row += f"<td>{model.prsquared:.3f}</td>"
        r2_row += "</tr>"
        table_rows.append(r2_row)
        
        # Final borders and note
        table_rows.append("<tr><td colspan='19' style='border-top: 2px solid black;'></td></tr>")
        table_rows.append("<tr><td colspan='19'><em>Note:</em> *p&lt;0.1; **p&lt;0.05; ***p&lt;0.01</td></tr>")
        table_rows.append("</table>")
        
        return "\n".join(table_rows)
    
    # Generate the wide table
    wide_table_html = create_wide_table_html(models_2024, benefits4)
    
    # Save HTML table
    os.makedirs('../results/tables', exist_ok=True)
    with open('../results/tables/complete_wide_table_2024_corrected.html', 'w') as f:
        f.write(wide_table_html)
    
    # Display table
    try:
        from IPython.display import HTML, display
        display(HTML(wide_table_html))
    except ImportError:
        print("IPython not available for display, but table saved to file.")
    
    print("Wide table saved to ../results/tables/complete_wide_table_2024_corrected.html")

def main():
    """Main execution function."""
    print("BEYOND SALARY: 2024 REGRESSION MODELS ANALYSIS")
    print("=" * 80)
    
    # Load and prepare data
    even_sample = load_and_prepare_data()
    if even_sample is None:
        print("Error: Could not load data. Please check the data path.")
        return
    
    # Run all models
    models_2024 = run_2024_models(even_sample)
    
    # Extract results
    results_df = extract_model_results(models_2024)
    print("\nModel Results Summary:")
    print(results_df[['Label', 'Model Iteration', 'Coefficient', 'P-Value', 'Converged']])
    
    # Generate outputs
    generate_coefficients_plot(results_df)
    generate_individual_tables(models_2024)
    generate_wide_table(models_2024)
    
    print("\n" + "=" * 80)
    print("ANALYSIS COMPLETE!")
    print("Generated outputs:")
    print("1. Model coefficients plot: ../results/figures/model_coefficients_plot_industry_converged.png")
    print("2. Individual regression tables: ../results/tables/job_level_model_2025/")
    print("3. Wide table: ../results/tables/complete_wide_table_2024_corrected.html")
    print("=" * 80)

if __name__ == "__main__":
    main()