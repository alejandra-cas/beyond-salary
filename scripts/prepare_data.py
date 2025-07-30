"""
Data preparation script for Beyond Salary analysis.
Merges job posting data with AI skills labels and remote work classification.
"""

import pandas as pd
import numpy as np
import pickle
import os
from pathlib import Path

def load_data_paths():
    """Define input and output data paths."""
    return {
        'jobs_csv': '../data/US_10M_SAMP_2018_2024.csv',
        'processed_ai_data': '/data/sant6443/thesis/data/us_10m_wham_nointernship_2019_2023.parquet.gzip',
        'wham_data': '../data/ID_CNTRY_ALL_WHAM.csv',
        'ai_skills_pkl': '../data/ai_skill_ids.pkl',
        'output_parquet': '../data/us_10m_nointernship_2018_2024.parquet.gzip'
    }

def merge_ai_skills_data(jobs_df, processed_df):
    """Merge main jobs data with AI skills classification."""
    # Select relevant columns from processed data
    processed_subset = processed_df[['ID', 'Has AI Skills', 'wfh_wham_prob', 'wfh_wham']]
    
    # Merge with main dataset
    merged_df = jobs_df.merge(processed_subset, left_on='ID', right_on='ID', how='left')
    
    print("AI Skills distribution:")
    print(merged_df.groupby('Has AI Skills', dropna=False).size())
    
    return merged_df

def add_year_column(df):
    """Extract year from POSTED date column."""
    df['YEAR'] = pd.to_datetime(df['POSTED']).dt.year
    
    print("Year distribution:")
    print(df.groupby('YEAR').size())
    
    return df

def merge_wham_data(df, wham_df):
    """Fill missing remote work classification data."""
    # Filter for US data only
    wham_us = wham_df[wham_df['country2'] == 'US']
    
    # Merge and fill missing values
    df = df.merge(wham_us[['id', 'wfh_wham_prob', 'wfh_wham']], 
                  left_on='ID', right_on='id', how='left', 
                  suffixes=('', '_wham'))
    
    # Combine first (fill NaNs with wham data)
    df['wfh_wham_prob'] = df['wfh_wham_prob'].combine_first(df['wfh_wham_prob_wham'])
    df['wfh_wham'] = df['wfh_wham'].combine_first(df['wfh_wham_wham'])
    
    # Clean up temporary columns
    df.drop(columns=['id', 'wfh_wham_prob_wham', 'wfh_wham_wham'], inplace=True)
    
    print("Missing WHAM data after merge:")
    print(df[df['wfh_wham_prob'].isna()].groupby('YEAR').size())
    
    return df

def main():
    """Main data preparation pipeline."""
    paths = load_data_paths()
    
    print("Loading main jobs data...")
    all_data = pd.read_csv(paths['jobs_csv'])
    
    # Filter out internships
    all_data = all_data[all_data['IS_INTERNSHIP'] == False]
    print(f"Data after removing internships: {len(all_data):,} rows")
    
    # Load processed data with AI skills
    print("Loading processed AI skills data...")
    if os.path.exists(paths['processed_ai_data']):
        p_data = pd.read_parquet(paths['processed_ai_data'])
        all_data = merge_ai_skills_data(all_data, p_data)
    else:
        print("Warning: AI skills data not found, continuing without merge")
    
    # Add year column
    all_data = add_year_column(all_data)
    
    # Load and merge WHAM data for missing remote work classifications
    print("Loading WHAM data...")
    if os.path.exists(paths['wham_data']):
        wham_data = pd.read_csv(paths['wham_data'])
        all_data = merge_wham_data(all_data, wham_data)
    else:
        print("Warning: WHAM data not found, continuing without merge")
    
    # Rename column for consistency
    if 'Has AI Skills' in all_data.columns:
        all_data.rename(columns={'Has AI Skills': 'AI ROLE'}, inplace=True)
    
    # Save processed data
    print(f"Saving processed data to {paths['output_parquet']}")
    all_data.to_parquet(paths['output_parquet'], compression='gzip')
    
    print("Data preparation complete!")
    print(f"Final dataset shape: {all_data.shape}")

if __name__ == "__main__":
    main()