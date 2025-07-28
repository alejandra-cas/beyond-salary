def get_summary_table(models, export_name = 'table.tex'):
    """
    Generate a LaTeX table for a list of models with an indication of fixed effects variables.

    Parameters:
    - models: list of fitted model objects.
    - fixed_effects_variables: list of fixed effects variables.
    - filename: name of the output LaTeX file.
    """   
    # Assuming model1 and model2 are your fitted model objects
    # Create the Stargazer object with the models
    stargazer = Stargazer(models)
    # List of fixed effects variables
    # Automatically determine covariates to include (excluding fixed effects)
    
    cov_names = stargazer.cov_names
    if 'Has AI Skills' in cov_names:
        cov_names.remove('Has AI Skills')
        cov_names.insert(0, 'Has AI Skills')
    stargazer.cov_names = cov_names
    
    
    fixed_effects_list = [cat for cat in fixed_effects_categories if cat in stargazer.cov_names]
    covariates_to_include = [covariate for covariate in stargazer.cov_names if covariate not in fixed_effects_list]
    
    # Remove unwanted covariates from stargazer
    for covariate in fixed_effects_list:
        if covariate in stargazer.cov_names:
            stargazer.cov_names.remove(covariate)

    
    # Add custom line for each fixed effect variable
    # stargazer.add_line("Fixed Effects")
    model_fixed_effects_dict = defaultdict(list)
    for i, model in enumerate(models): 
        model_fixed_effects_dict[i] = list(set([value for key, value in cat_var_dict.items() if key in model.model.exog_names]))
        
    fixed_effects_names = {'STATE_NAME': 'State',  'NAICS_2022_2_NAME': 'Industry', 'POSTED_YEAR': 'Year', 'SOC_2021_2_NAME': 'Occupation'}    
    
    stargazer.add_line("\\textbf{Fixed Effects}", [""] * len(models))

    for variable in fixed_effects_variables:
        # fixed_effects = [value for key: value in cat_var_dict if key in stargazer.cov_names]
        variable_name = fixed_effects_names[variable]
        inclusion_list = ['Yes' if variable in model_fixed_effects_dict[model] else 'No' for model in model_fixed_effects_dict]
        stargazer.add_line(f"{variable_name}", inclusion_list)

    # Customize the Stargazer table (optional)
    # stargazer.title("Regression Results")
    # stargazer.custom_columns([f"Model {i+1}" for i in range(len(models))], [1] * len(models))

    stargazer.significance_levels([0.1, 0.05, 0.01])
    # stargazer.add_line("Observations", [len(model.model.endog) for model in models])

    # Render LaTeX table
    latex_table = stargazer.render_latex()
    print(latex_table)
    # export latex table
    with open(f'../exports/{export_name}', 'w') as file:
        file.write(latex_table)
    
    display(HTML(stargazer.render_html()))
    # Print the LaTeX table to verify

    # Save the LaTeX code to a .tex file
    # with open("regression_results.tex", "w") as f:
        # f.write(latex_table)
