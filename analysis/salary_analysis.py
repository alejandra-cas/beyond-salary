#!/usr/bin/env python3
"""
Salary Analysis Script

Generates the salary_by_benefit_combined figure showing median annual salary
by AI skills and various benefits with confidence intervals.

This script creates a combined figure with 6 subplots (2 rows x 3 columns)
showing salary comparisons for different benefit categories.
"""

import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns
import os
import sys
from pathlib import Path

# Add src directory to path
sys.path.append(str(Path(__file__).parent.parent / 'src'))

try:
    from package_files.benefits_defns import *
    from package_files.config_utils import get_processed_dir, get_repo_root
except ImportError:
    # Fallback benefit definitions if import fails
    benefits4 = ['EDU_ASSISTANCE', 'PAID LEAVE', 'HEALTH_WELLBEING', 'PARENTAL_LEAVE', 'CULTURE', 'REMOTE_KW']
    
    benefits_labels_map = {
        'EDU_ASSISTANCE': 'Tuition Assistance', 
        'PAID LEAVE': 'Paid Leave', 
        'HEALTH_WELLBEING': 'Health and Wellbeing', 
        'PARENTAL_LEAVE': 'Parental Leave', 
        'CULTURE': 'Inclusive Workplace', 
        'REMOTE_KW': 'Remote Work'
    }
    
    benefit_colors = {
        'EDU_ASSISTANCE': '#41afaa',
        'PAID LEAVE': '#466eb4',
        'HEALTH_WELLBEING': '#e6a532',
        'PARENTAL_LEAVE': '#00a0e1',
        'CULTURE': '#d7642c',
        'REMOTE_KW': '#c765a6'
    }

    def get_processed_dir():
        return Path(__file__).parent.parent / "data" / "processed"

    def get_repo_root():
        return Path(__file__).parent.parent

REPO_ROOT = get_repo_root()
PROCESSED_DIR = get_processed_dir()
RESULTS_DIR = REPO_ROOT / "results" / "figures_2026" / "salary"

def median_ci(data, confidence=0.95):
    """Calculate median with confidence interval."""
    data = np.array(data)
    data = data[~np.isnan(data)]  # Remove NaN values
    
    if len(data) == 0:
        return np.nan, np.nan, np.nan
    
    n = len(data)
    median = np.median(data)
    
    # Use bootstrap method for small samples, normal approximation for large samples
    if n < 30:
        # Bootstrap method
        n_bootstrap = 1000
        bootstrap_medians = []
        for _ in range(n_bootstrap):
            bootstrap_sample = np.random.choice(data, size=n, replace=True)
            bootstrap_medians.append(np.median(bootstrap_sample))
        
        ci_lower = np.percentile(bootstrap_medians, (1 - confidence) / 2 * 100)
        ci_upper = np.percentile(bootstrap_medians, (1 + confidence) / 2 * 100)
    else:
        # Normal approximation method
        z = stats.norm.ppf(0.5 + confidence / 2.0)
        margin_of_error = z * np.std(data, ddof=1) / np.sqrt(n)
        ci_lower = median - margin_of_error
        ci_upper = median + margin_of_error
    
    return median, ci_lower, ci_upper

def load_and_prepare_data():
    """Load and prepare the salary data."""
    print("Loading salary data...")
    usdf_salary = pd.read_parquet(PROCESSED_DIR / "labeled_v2.parquet")
    
    # Filter out rows with null salaries
    print(f"Initial data shape: {usdf_salary.shape}")
    usdf_salary = usdf_salary[usdf_salary['SALARY'].notnull()]
    print(f"After removing null salaries: {usdf_salary.shape}")
    
    # Ensure we have the required columns
    required_columns = ['YEAR', 'AI ROLE', 'SALARY'] + benefits4
    missing_columns = [col for col in required_columns if col not in usdf_salary.columns]
    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")
    
    # Ensure AI ROLE is boolean
    if 'AI ROLE' in usdf_salary.columns:
        usdf_salary['AI ROLE'] = usdf_salary['AI ROLE'].astype(bool)
    
    # Ensure benefits are boolean
    for benefit in benefits4:
        if benefit in usdf_salary.columns:
            usdf_salary[benefit] = usdf_salary[benefit].astype(bool)
    
    print(f"Final data shape: {usdf_salary.shape}")
    # print(f"Year range: {usdf_salary['YEAR'].min()} - {usdf_salary['YEAR'].max()}")
    print(f"AI Role distribution: {usdf_salary['AI ROLE'].value_counts()}")
    
    return usdf_salary

def calculate_salary_stats(usdf_salary):
    """Calculate salary statistics by year, AI role, and benefit."""
    print("Calculating salary statistics...")
    
    all_stats = []
    
    for benefit in benefits4:
        print(f"  Processing {benefit}...")
        
        # Group by year, AI role, and benefit presence
        grouped = usdf_salary.groupby(['YEAR', 'AI ROLE', benefit])['SALARY']
        
        for (year, ai_role, benefit_present), group in grouped:
            median, ci_lower, ci_upper = median_ci(group.values)
            
            all_stats.append({
                'YEAR': year,
                'AI ROLE': ai_role,
                'Benefit': benefit,
                'Benefit_Present': benefit_present,
                'Median_Salary': median,
                'CI_Lower': ci_lower,
                'CI_Upper': ci_upper,
                'Count': len(group)
            })
    
    return pd.DataFrame(all_stats)

def generate_combined_figure(salary_stats):
    """Generate the combined salary by benefit figure."""
    print("Generating combined salary figure...")
    
    # Set up the figure
    fig, axes = plt.subplots(nrows=2, ncols=3, figsize=(20, 12))
    axes = axes.flatten()
    
    # Set plot style
    plt.style.use('default')
    sns.set_palette("husl")
    
    for i, benefit in enumerate(benefits4):
        print(f"  Creating subplot for {benefit}...")
        
        # Filter data for this benefit
        benefit_data = salary_stats[salary_stats['Benefit'] == benefit]
        
        if len(benefit_data) == 0:
            print(f"    No data available for {benefit}")
            continue
        
        ax = axes[i]
        
        # Plot lines for each combination of AI role and benefit presence
        for ai_role in [True, False]:
            for benefit_present in [True, False]:
                subset = benefit_data[
                    (benefit_data['AI ROLE'] == ai_role) & 
                    (benefit_data['Benefit_Present'] == benefit_present)
                ]
                
                if len(subset) == 0:
                    continue
                
                # Determine color and style
                if ai_role:
                    color = benefit_colors[benefit]
                    alpha = 1.0 if benefit_present else 0.6
                    linestyle = '-' if benefit_present else '--'
                else:
                    color = 'gray'
                    alpha = 1.0 if benefit_present else 0.6
                    linestyle = '-' if benefit_present else '--'
                
                # Create label
                ai_label = "AI Role" if ai_role else "Non-AI Role"
                benefit_label = "With Benefit" if benefit_present else "Without Benefit"
                label = f"{ai_label}, {benefit_label}"
                
                # Plot line
                ax.plot(subset['YEAR'], subset['Median_Salary'], 
                       color=color, alpha=alpha, linestyle=linestyle,
                       marker='o', markersize=4, label=label)
                
                # Add confidence intervals
                ax.fill_between(subset['YEAR'], 
                              subset['CI_Lower'], 
                              subset['CI_Upper'],
                              color=color, alpha=0.2)
        
        # Customize subplot
        ax.set_title(benefits_labels_map[benefit], fontsize=20, fontweight='bold')
        ax.set_xlabel('Year', fontsize=16)
        ax.set_ylabel('Median Annual Salary (USD)', fontsize=16)
        ax.tick_params(labelsize=14)
        
        # Format y-axis to show salary in thousands
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x/1000:.0f}K'))
        
        # Set x-axis ticks
        years = sorted(benefit_data['YEAR'].unique())
        ax.set_xticks(years)
        ax.set_xlim(min(years) - 0.5, max(years) + 0.5)
        
        # Add legend only to first subplot
        if i == 0:
            ax.legend(bbox_to_anchor=(.05, 1), loc='upper left', fontsize=14)
        
        # Add grid
        ax.grid(True, alpha=0.3)
    
    # Adjust layout
    plt.tight_layout()
    
    # Save figure
    output_path = RESULTS_DIR / "salary_by_benefit_combined.png"
    os.makedirs(output_path.parent, exist_ok=True)
    
    print(f"Saving figure to: {output_path}")
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    # plt.show()
    
    print("Salary analysis complete!")
    return output_path

def print_summary_stats(salary_stats):
    """Print summary statistics."""
    print("\n" + "="*60)
    print("SALARY ANALYSIS SUMMARY")
    print("="*60)
    
    for benefit in benefits4:
        benefit_data = salary_stats[salary_stats['Benefit'] == benefit]
        
        if len(benefit_data) == 0:
            continue
        
        print(f"\n{benefits_labels_map[benefit]}:")
        print("-" * 40)
        
        # AI roles with/without benefit
        ai_with = benefit_data[(benefit_data['AI ROLE'] == True) & (benefit_data['Benefit_Present'] == True)]
        ai_without = benefit_data[(benefit_data['AI ROLE'] == True) & (benefit_data['Benefit_Present'] == False)]
        
        if len(ai_with) > 0 and len(ai_without) > 0:
            avg_with = ai_with['Median_Salary'].mean()
            avg_without = ai_without['Median_Salary'].mean()
            premium = ((avg_with - avg_without) / avg_without) * 100
            
            print(f"  AI roles with benefit: ${avg_with:,.0f}")
            print(f"  AI roles without benefit: ${avg_without:,.0f}")
            print(f"  Premium: {premium:+.1f}%")

def main():
    """Main function to generate the salary analysis."""
    print("BEYOND SALARY: SALARY ANALYSIS")
    print("=" * 60)
    
    try:
        # Load and prepare data
        usdf_salary = load_and_prepare_data()
        
        # Calculate salary statistics
        salary_stats = calculate_salary_stats(usdf_salary)

        # Export stats so figures can be rebuilt without the raw data
        # (consumed by scripts/generate_combined_figures.py for paper Figure 4a)
        stats_path = REPO_ROOT / "results" / "tables_2026" / "salary" / "salary_by_benefit_stats.csv"
        os.makedirs(stats_path.parent, exist_ok=True)
        salary_stats.to_csv(stats_path, index=False)
        print(f"Salary statistics saved to: {stats_path}")

        # Generate the combined figure
        output_path = generate_combined_figure(salary_stats)
        
        # Print summary statistics
        print_summary_stats(salary_stats)
        
        print(f"\n✅ Analysis complete! Figure saved to: {output_path}")
        
    except FileNotFoundError as e:
        print(f"❌ Error: Data file not found. {e}")
        print("Please ensure the salary data file exists in the data directory.")
    except Exception as e:
        print(f"❌ An error occurred: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
