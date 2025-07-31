#!/usr/bin/env python3
"""
Scatterplot Analysis Script

This script generates raw number scatterplots showing correlations between
various factors and AI benefit prevalence across different benefits.

Creates both percentage-based and log-transformed scatterplots for
occupation-year analysis.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import sys
from pathlib import Path

# Add src directory to path
sys.path.append(str(Path(__file__).parent.parent / 'src'))

try:
    from package_files.benefits_defns import *
except ImportError:
    # Fallback definitions
    benefits4 = ['EDU_ASSISTANCE', 'PAID_LEAVE', 'HEALTH_WELLBEING', 'PARENTAL_LEAVE', 'CULTURE', 'REMOTE_KW']
    
    benefits_labels_map = {
        'EDU_ASSISTANCE': 'Tuition Assistance', 
        'PAID_LEAVE': 'Paid Leave', 
        'HEALTH_WELLBEING': 'Health and Wellbeing', 
        'PARENTAL_LEAVE': 'Parental Leave', 
        'CULTURE': 'Workplace Culture', 
        'REMOTE_KW': 'Remote Work'
    }
    
    benefits4_labels = ['Tuition Assistance', 'Paid Leave', 'Health and Wellbeing', 'Parental Leave', 'Workplace Culture', 'Remote Work']
    
    benefit_colors = {
        'EDU_ASSISTANCE': '#41afaa',
        'PAID_LEAVE': '#466eb4',
        'HEALTH_WELLBEING': '#e6a532',
        'PARENTAL_LEAVE': '#00a0e1',
        'CULTURE': '#d7642c',
        'REMOTE_KW': '#c765a6'
    }

def remove_outliers(df, columns):
    """Remove outliers using IQR method for specified columns."""
    result_df = df.copy()
    
    for column in columns:
        if column in result_df.columns:
            Q1 = result_df[column].quantile(0.25)
            Q3 = result_df[column].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            result_df = result_df[(result_df[column] >= lower_bound) & (result_df[column] <= upper_bound)]
    
    return result_df

def load_occupation_year_data():
    """Load occupation-year analysis data."""
    print("Loading occupation-year data...")
    
    # Try different possible data paths
    possible_paths = [
        '../data/occ_year_analysis_2024_raw.csv',
        'exports/occ_year_data/occ_year_analysis_2024_raw.csv'
    ]
    
    for path in possible_paths:
        if os.path.exists(path):
            print(f"✅ Found data at: {path}")
            return pd.read_csv(path)
    
    raise FileNotFoundError(f"Could not find occupation-year data file in any of: {possible_paths}")

def generate_percentage_scatterplots(coeff_df, output_dir):
    """Generate percentage-based scatterplots (from original remotekw analysis)."""
    print("Generating percentage-based scatterplots...")
    
    # Set the colors for different perks
    palette = list(benefit_colors.values())
    
    x_factors = ['Benefit Prevalence', 'AI Demand', 'AI Demand % Change', 'MEDIAN_SALARY_ai']
    
    for x in range(4):
        print(f"  Creating scatterplot for {x_factors[x]}...")
        
        plt.figure(figsize=(10, 8))
        
        for i, benefit in enumerate(benefits4):
            # Prepare benefit data
            benefit_df = coeff_df[[
                f'Prevalence: {benefits_labels_map[benefit]} (AI)', 
                f'Prevalence: {benefits_labels_map[benefit]}', 
                'AI Demand', 
                'AI Demand % Change', 
                'MEDIAN_SALARY_ai'
            ]].copy()
            
            benefit_df.rename(columns={
                f'Prevalence: {benefits_labels_map[benefit]} (AI)': 'Benefit Prevalence (AI Roles)', 
                f'Prevalence: {benefits_labels_map[benefit]}': 'Benefit Prevalence'
            }, inplace=True)
            
            factor = x_factors[x]
            benefit_df = benefit_df.dropna(subset=[factor, 'Benefit Prevalence (AI Roles)'])
            
            # Remove outliers
            scatter_df = remove_outliers(benefit_df, [factor, 'Benefit Prevalence (AI Roles)'])
            
            if len(scatter_df) == 0:
                print(f"    No data for {benefit} after outlier removal")
                continue
            
            # Plot scatter points
            sns.scatterplot(
                data=scatter_df,
                x=factor,
                y='Benefit Prevalence (AI Roles)',
                color=palette[i],
                label=benefits4_labels[i],
                alpha=0.7
            )
            
            # Plot regression line
            sns.regplot(
                data=scatter_df,
                x=factor,
                y='Benefit Prevalence (AI Roles)',
                scatter=False,
                color=palette[i],
                line_kws={'label': benefits4_labels[i], 'linewidth': 2}
            )
        
        # Customize plot
        plt.xlabel(factor)
        plt.ylabel('Benefit Prevalence (AI)')
        plt.title(f'Correlation between {factor} and AI Benefit Prevalence for Different Benefits')
        plt.legend(title='Benefit', bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        
        # Save plot
        output_path = os.path.join(output_dir, f'scatterplot_pct_{factor.replace(" ", "_")}.png')
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"    Saved: {output_path}")

def generate_raw_number_scatterplots(df, output_dir):
    """Generate raw number (log-transformed) scatterplots."""
    print("Generating raw number (log-transformed) scatterplots...")
    
    # Set the colors for different perks
    palette = list(benefit_colors.values())
    
    # Define x-factors for raw number analysis
    base_cols = ['Overall Benefit Prevalence', 'AI Job Count Change', 'AI Job Count', 'Benefit Prevalence (AI Roles)']
    x_factors = ['Log Overall Benefit Prevalence', 'Log AI Job Count Change', 'Log AI Job Count', 'Median Log Salary AI', 'Log Job Count', 'AI Job Count Change']
    
    for x in range(6):
        print(f"  Creating raw number scatterplot for {x_factors[x]}...")
        
        plt.figure(figsize=(10, 8))
        
        for i, benefit in enumerate(benefits4):
            # Prepare benefit data
            benefit_df = df.copy()
            benefit_df.rename(columns={
                f'Prevalence: {benefits_labels_map[benefit]}': 'Overall Benefit Prevalence',
                f'Prevalence: {benefits_labels_map[benefit]} (AI)': 'Benefit Prevalence (AI Roles)',
                'ai_role_count': 'AI Job Count', 
                'MEDIAN_LOG_SALARY_ai': 'Median Log Salary AI'
            }, inplace=True)
            
            # Apply log transformation with sign preservation
            log_cols = ['Overall Benefit Prevalence', 'AI Job Count Change', 'AI Job Count', 'Benefit Prevalence (AI Roles)']
            for col in log_cols:
                if col in benefit_df.columns:
                    benefit_df[f'Log {col}'] = benefit_df[col].apply(lambda x: np.sign(x) * np.log(abs(x) + 1))
            
            factor = x_factors[x]
            y_var = 'Log Benefit Prevalence (AI Roles)'
            
            # Filter data
            benefit_df = benefit_df.dropna(subset=[factor, y_var])
            
            if len(benefit_df) == 0:
                print(f"    No data for {benefit}")
                continue
            
            # Use all data (no outlier removal for raw numbers as in original)
            scatter_df = benefit_df.copy()
            
            # Plot scatter points
            sns.scatterplot(
                data=scatter_df,
                x=factor,
                y=y_var,
                color=palette[i],
                label=benefits4_labels[i],
                alpha=0.7
            )
            
            # Plot regression line
            sns.regplot(
                data=scatter_df,
                x=factor,
                y=y_var,
                scatter=False,
                color=palette[i],
                line_kws={'label': benefits4_labels[i], 'linewidth': 2}
            )
        
        # Customize plot
        plt.xlabel(factor)
        plt.ylabel('Log Benefit Prevalence (AI)')
        # No title (as in original notebook)
        plt.legend(title='Benefit', bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        
        # Save plot
        output_path = os.path.join(output_dir, f'scatterplot_log_{factor.replace(" ", "_")}.png')
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"    Saved: {output_path}")

def main():
    """Main function to generate all scatterplots."""
    print("SCATTERPLOT ANALYSIS")
    print("=" * 50)
    
    # Setup output directory
    output_dir = '../results/figures/scatterplots'
    os.makedirs(output_dir, exist_ok=True)
    
    try:
        # Load the raw occupation-year data for log scatterplots
        df = load_occupation_year_data()
        print(f"Loaded data with {len(df)} rows and {len(df.columns)} columns")
        
        # Generate raw number (log-transformed) scatterplots
        generate_raw_number_scatterplots(df, output_dir)
        
        # Try to load percentage data for percentage scatterplots (optional)
        try:
            coeff_df_path = '../exports/occ_year_data/occ_year_analysis_remotekw.csv'
            if os.path.exists(coeff_df_path):
                coeff_df = pd.read_csv(coeff_df_path)
                print(f"Also found percentage data with {len(coeff_df)} rows")
                generate_percentage_scatterplots(coeff_df, output_dir)
            else:
                print("Percentage data not found, skipping percentage scatterplots")
        except Exception as e:
            print(f"Warning: Could not generate percentage scatterplots: {e}")
        
        print("\n" + "=" * 50)
        print("✅ SCATTERPLOT ANALYSIS COMPLETE!")
        print(f"📊 Outputs saved to: {output_dir}")
        print("=" * 50)
        
    except Exception as e:
        print(f"❌ Error during scatterplot analysis: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()