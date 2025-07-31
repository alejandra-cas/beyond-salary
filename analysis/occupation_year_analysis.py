#!/usr/bin/env python3
"""
Occupation-Year Balanced Sample Differences Analysis

This script reproduces the balanced sample differences analysis from the notebooks:
1. occ_year_even_sample.ipynb - for data preparation and balanced sampling
2. occ_year_model_new.ipynb - for the difference regression models

It generates both HTML and LaTeX table outputs for the balanced sample differences analysis.
"""

import pandas as pd
import numpy as np
import sys
import os
from pathlib import Path
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor

# Add src directory to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

try:
    from package_files.benefits_defns import *
except ImportError:
    # Fallback definitions
    benefits4 = [
        "EDU_ASSISTANCE",
        "PAID_LEAVE",
        "HEALTH_WELLBEING",
        "PARENTAL_LEAVE",
        "CULTURE",
        "REMOTE_KW",
    ]
    benefits_labels_map = {
        "EDU_ASSISTANCE": "Tuition Assistance",
        "PAID_LEAVE": "Paid Leave",
        "HEALTH_WELLBEING": "Health and Wellbeing",
        "PARENTAL_LEAVE": "Parental Leave",
        "CULTURE": "Workplace Culture",
        "REMOTE_KW": "Remote Keywords",
    }
    benefits4_labels = [
        "Tuition Assistance",
        "Paid Leave",
        "Health and Wellbeing",
        "Parental Leave",
        "Workplace Culture",
        "Remote Keywords",
    ]

occupation = "SOC_2021_2_NAME"


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


def create_regression_table_html(models, model_names, output_path):
    """Create HTML regression table similar to Stargazer output."""
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
        <h2>Balanced Sample Differences Regression Results</h2>
        <table>
    """

    # Create header row with model names
    html_content += "<tr><th></th>"
    for name in model_names:
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
        if i + 1 < len(models):
            all_params.update(models[i + 1].params.index)

    # Sort parameters for consistent ordering
    param_order = ["const"] + [p for p in sorted(all_params) if p != "const"]

    # Add parameter rows
    for param in param_order:
        html_content += f"<tr><td class='variable-name'>{param}</td>"

        for i in range(0, len(models), 2):  # Every pair of models
            # Model A
            if param in models[i].params.index:
                coef = models[i].params[param]
                pval = models[i].pvalues[param]
                stars = get_significance_stars(pval)
                html_content += f"<td class='coefficient'>{coef:.4f}{stars}</td>"
            else:
                html_content += "<td>-</td>"

            # Model B
            if i + 1 < len(models) and param in models[i + 1].params.index:
                coef = models[i + 1].params[param]
                pval = models[i + 1].pvalues[param]
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
            if i + 1 < len(models) and param in models[i + 1].params.index:
                se = models[i + 1].bse[param]
                html_content += f"<td class='se'>({se:.4f})</td>"
            else:
                html_content += "<td>-</td>"

        html_content += "</tr>\n"

    # Add model statistics
    html_content += "<tr><td class='variable-name'>R-squared</td>"
    for i in range(0, len(models), 2):
        html_content += f"<td>{models[i].rsquared:.4f}</td>"
        if i + 1 < len(models):
            html_content += f"<td>{models[i+1].rsquared:.4f}</td>"
    html_content += "</tr>\n"

    html_content += "<tr><td class='variable-name'>Adj. R-squared</td>"
    for i in range(0, len(models), 2):
        html_content += f"<td>{models[i].rsquared_adj:.4f}</td>"
        if i + 1 < len(models):
            html_content += f"<td>{models[i+1].rsquared_adj:.4f}</td>"
    html_content += "</tr>\n"

    html_content += "<tr><td class='variable-name'>Observations</td>"
    for i in range(0, len(models), 2):
        html_content += f"<td>{int(models[i].nobs)}</td>"
        if i + 1 < len(models):
            html_content += f"<td>{int(models[i+1].nobs)}</td>"
    html_content += "</tr>\n"

    html_content += """
        </table>
        <p><small>*** p&lt;0.01, ** p&lt;0.05, * p&lt;0.1</small></p>
    </body>
    </html>
    """

    with open(output_path, "w") as f:
        f.write(html_content)

    return html_content


def create_regression_table_latex(models, model_names, output_path):
    """Create LaTeX regression table similar to Stargazer output."""
    latex_content = """\\begin{table}[htbp]
\\centering
\\caption{Balanced Sample Differences Regression Results}
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
        if i + 1 < len(models):
            all_params.update(models[i + 1].params.index)

    param_order = ["const"] + [p for p in sorted(all_params) if p != "const"]

    # Add parameter rows
    for param in param_order:
        param_name = param.replace("_", "\\_")
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
            if i + 1 < len(models) and param in models[i + 1].params.index:
                coef = models[i + 1].params[param]
                pval = models[i + 1].pvalues[param]
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
            if i + 1 < len(models) and param in models[i + 1].params.index:
                se = models[i + 1].bse[param]
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
        if i + 1 < len(models):
            r2s.append(f"{models[i+1].rsquared:.4f}")
    latex_content += " & ".join(r2s) + " \\\\\n"

    latex_content += "Adj. R-squared & "
    adj_r2s = []
    for i in range(0, len(models), 2):
        adj_r2s.append(f"{models[i].rsquared_adj:.4f}")
        if i + 1 < len(models):
            adj_r2s.append(f"{models[i+1].rsquared_adj:.4f}")
    latex_content += " & ".join(adj_r2s) + " \\\\\n"

    latex_content += "Observations & "
    obs = []
    for i in range(0, len(models), 2):
        obs.append(f"{int(models[i].nobs)}")
        if i + 1 < len(models):
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

    with open(output_path, "w") as f:
        f.write(latex_content)

    return latex_content


def create_balanced_sample(data_path):
    """Create balanced sample from the main dataset."""
    print("Reading data...")
    data = pd.read_parquet(data_path)

    # Select specific occupations (from notebook)
    occupations_select = [
        "Architecture and Engineering Occupations",
        "Arts, Design, Entertainment, Sports, and Media Occupations",
        "Business and Financial Operations Occupations",
        "Community and Social Service Occupations",
        "Computer and Mathematical Occupations",
        "Educational Instruction and Library Occupations",
        "Healthcare Practitioners and Technical Occupations",
        "Legal Occupations",
        "Life, Physical, and Social Science Occupations",
        "Management Occupations",
        "Office and Administrative Support Occupations",
        "Personal Care and Service Occupations",
        "Production Occupations",
        "Sales and Related Occupations",
        "Transportation and Material Moving Occupations",
    ]

    data_select = data[data[occupation].isin(occupations_select)]

    print("Creating balanced sample...")
    balanced_sample = []

    for (occ, year), group in data_select.groupby([occupation, "YEAR"]):
        # Separate AI and non-AI roles
        ai_roles = group[group["AI ROLE"] == 1]
        non_ai_roles = group[group["AI ROLE"] == 0]

        # Determine the smaller size for sample size
        sample_size = min(len(ai_roles), len(non_ai_roles))

        if sample_size == 0:
            continue

        # Random sample with fixed seed for reproducibility
        sampled_ai = ai_roles.sample(n=sample_size, random_state=42)
        sampled_non_ai = non_ai_roles.sample(n=sample_size, random_state=42)

        # Combine samples
        balanced_group = pd.concat([sampled_ai, sampled_non_ai])
        balanced_sample.append(balanced_group)

    # Combine all balanced groups into a single DataFrame
    final_balanced_sample = pd.concat(balanced_sample, ignore_index=True)

    return final_balanced_sample


def calculate_benefit_differences(balanced_sample):
    """Calculate benefit differences between AI and non-AI jobs."""
    print("Calculating benefit differences...")

    ai_jobs = balanced_sample[balanced_sample["AI ROLE"] == 1]
    non_ai_jobs = balanced_sample[balanced_sample["AI ROLE"] == 0]

    # Group by occupation and YEAR and sum benefits for AI and non-AI jobs
    ai_counts = ai_jobs.groupby([occupation, "YEAR"])[benefits4].sum().reset_index()
    non_ai_counts = (
        non_ai_jobs.groupby([occupation, "YEAR"])[benefits4].sum().reset_index()
    )

    # Merge the counts for AI and non-AI jobs
    merged_counts = ai_counts.merge(
        non_ai_counts, on=[occupation, "YEAR"], suffixes=("_ai", "_non_ai")
    )

    # Calculate the difference between AI and non-AI jobs for each benefit
    for benefit in benefits4:
        merged_counts[f"{benefit}_difference"] = (
            merged_counts[f"{benefit}_ai"] - merged_counts[f"{benefit}_non_ai"]
        )

    # Select only the columns with differences and occupation-year identifiers
    difference_summary = merged_counts[
        [occupation, "YEAR"] + [f"{benefit}_difference" for benefit in benefits4]
    ]

    return difference_summary


def load_occupation_year_data(occ_year_data_path):
    """Load the main occupation-year analysis data."""
    print("Loading occupation-year data...")
    df = pd.read_csv(occ_year_data_path)

    # Add log job count
    df["Log Job Count"] = df["job_count"].apply(
        lambda x: np.log(x + 1)
    )  # Add 1 to avoid log(0)

    return df


def run_regression_models(df, include_ai_count=False):
    """Run the balanced sample difference regression models."""
    model_type = "Extended" if include_ai_count else "Basic"
    print(
        f"Running {model_type.lower()} balanced sample difference regression models..."
    )

    diff_models = []

    for benefit in benefits4:
        benefit_df = df.copy()
        benefit_df.rename(
            columns={
                "ai_role_count": "AI Job Count",
                "MEDIAN_LOG_SALARY_ai": "Median Log Salary AI",
                f"Prevalence: {benefits_labels_map[benefit]}": "Overall Benefit Prevalence",
            },
            inplace=True,
        )

        # Handle remote work (exclude 2018)
        if benefit == "REMOTE_KW":
            benefit_df = benefit_df[benefit_df["YEAR"] != 2018]

        # Define log columns based on model type
        if include_ai_count:
            log_cols = ["Overall Benefit Prevalence", "AI Job Count"]
        else:
            log_cols = ["Overall Benefit Prevalence"]

        benefit_df = benefit_df.dropna(subset=log_cols)

        # Apply log transformation with sign preservation
        for col in log_cols:
            benefit_df[f"Log {col}"] = benefit_df[col].apply(
                lambda x: np.sign(x) * np.log(abs(x) + 1)
            )

        X_cols = [f"Log {col}" for col in log_cols] + [
            "Median Log Salary AI",
            "Log Job Count",
        ]
        y_col = f"{benefit}_difference"

        benefit_df = benefit_df.dropna(subset=X_cols + [y_col])
        X = benefit_df[X_cols]
        y = benefit_df[y_col]

        # Add year dummies (excluding 2019 as reference)
        year_dummies = pd.get_dummies(benefit_df["YEAR"], prefix="Year").drop(
            columns=["Year_2019"], errors="ignore"
        )
        X = pd.concat([X, year_dummies], axis=1)
        X = sm.add_constant(X)

        # Fit the OLS model
        try:
            model = sm.OLS(y, X).fit()
            print(f"\n{benefit} {model_type} Model Summary:")
            print(f"R-squared: {model.rsquared:.4f}")
            print(f"Adj. R-squared: {model.rsquared_adj:.4f}")
            print(f"Observations: {int(model.nobs)}")

            diff_models.append(model)
        except Exception as e:
            print(f"Error fitting model for {benefit}: {e}")
            diff_models.append(None)

    return diff_models


def generate_combined_tables(diff_models, diff_models_2, output_dir):
    """Generate combined HTML and LaTeX tables for the difference regression models."""
    print("Generating combined regression tables...")

    # Filter out None models
    valid_models = []
    valid_models_2 = []
    valid_labels = []

    for i, (model_a, model_b) in enumerate(zip(diff_models, diff_models_2)):
        if model_a is not None and model_b is not None:
            valid_models.extend([model_a, model_b])
            valid_labels.append(benefits4_labels[i])

    if not valid_models:
        print("No valid models to generate tables for.")
        return None, None

    # Generate HTML table
    html_output_path = os.path.join(
        output_dir, "difference_regression_table_combined.html"
    )
    html_table = create_regression_table_html(
        valid_models, valid_labels, html_output_path
    )
    print(f"HTML table saved to: {html_output_path}")

    # Generate LaTeX table
    latex_output_path = os.path.join(
        output_dir, "difference_regression_table_combined.tex"
    )
    latex_table = create_regression_table_latex(
        valid_models, valid_labels, latex_output_path
    )
    print(f"LaTeX table saved to: {latex_output_path}")

    return html_table, latex_table


def main():
    """Main function to run the complete balanced sample differences analysis."""
    print("OCCUPATION-YEAR BALANCED SAMPLE DIFFERENCES ANALYSIS")
    print("=" * 80)

    # File paths (adjust as needed for clean repo structure)
    possible_data_paths = [
        "../data/us_10m_nointernship_2018_2024_benefits.parquet.gzip",
        "../data/2024_salary_sample.parquet.gzip",
        "data/us_10m_nointernship_2018_2024_benefits.parquet.gzip",
    ]

    possible_occ_year_paths = [
        "../data/occ_year_analysis_2024_raw.csv",
        "exports/occ_year_data/occ_year_analysis_2024_raw.csv",
    ]

    output_dir = "../results/tables"

    # Find data files
    data_path = None
    for path in possible_data_paths:
        if os.path.exists(path):
            data_path = path
            break

    occ_year_data_path = None
    for path in possible_occ_year_paths:
        if os.path.exists(path):
            occ_year_data_path = path
            break

    if not data_path:
        print("❌ Error: Could not find main data file")
        print(f"Looked for: {possible_data_paths}")
        return None

    if not occ_year_data_path:
        print("❌ Error: Could not find occupation-year data file")
        print(f"Looked for: {possible_occ_year_paths}")
        return None

    print(f"✅ Using data file: {data_path}")
    print(f"✅ Using occupation-year file: {occ_year_data_path}")

    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Check if we need to create balanced sample
    balanced_diffs_path = "../data/balanced_sample_diffs.csv"

    if os.path.exists(balanced_diffs_path):
        print("Loading existing balanced sample differences...")
        balanced_diffs = pd.read_csv(balanced_diffs_path)
    else:
        print("Creating balanced sample from scratch...")
        balanced_sample = create_balanced_sample(data_path)
        balanced_diffs = calculate_benefit_differences(balanced_sample)

        # Save the balanced sample differences
        os.makedirs(os.path.dirname(balanced_diffs_path), exist_ok=True)
        balanced_diffs.to_csv(balanced_diffs_path, index=False)
        print(f"✅ Balanced sample differences saved to: {balanced_diffs_path}")

    # Load occupation-year data
    occ_year_df = load_occupation_year_data(occ_year_data_path)

    # Merge with balanced sample differences
    print("Merging occupation-year data with balanced sample differences...")
    df = occ_year_df.merge(balanced_diffs, on=[occupation, "YEAR"])

    print(f"Final dataset shape: {df.shape}")
    print(f"Occupation-years: {len(df)}")
    print(f"Years covered: {sorted(df['YEAR'].unique())}")
    print(f"Occupations: {len(df[occupation].unique())}")

    # Run regression models
    diff_models = run_regression_models(df, include_ai_count=False)
    diff_models_2 = run_regression_models(df, include_ai_count=True)

    # Generate combined tables
    html_table, latex_table = generate_combined_tables(
        diff_models, diff_models_2, output_dir
    )

    print("\n" + "=" * 80)
    print("✅ ANALYSIS COMPLETED SUCCESSFULLY!")
    print("📊 Generated outputs:")
    print(f"   - HTML table: {output_dir}/difference_regression_table_combined.html")
    print(f"   - LaTeX table: {output_dir}/difference_regression_table_combined.tex")
    print("=" * 80)

    # Return results for further use if needed
    return {
        "data": df,
        "models_basic": diff_models,
        "models_extended": diff_models_2,
        "html_table": html_table,
        "latex_table": latex_table,
    }


if __name__ == "__main__":
    results = main()
