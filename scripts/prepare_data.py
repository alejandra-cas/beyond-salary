"""
Data preparation script for Beyond Salary analysis.
Merges job posting data with AI skills labels and remote work classification.
"""

import pandas as pd
import numpy as np
from pathlib import Path


def load_data_paths():
    """Define input and output data paths."""
    base = Path(__file__).parent.parent / "data"
    return {
        "posts_csv": base / "OII_US_10M_POSTS_MAY26_SUBSAMPLE.csv",
        "skills_csv": base / "OII_US_10M_SKILLS_MAY26_SUBSAMPLE.csv",
        "body_csv": base / "OII_US_10M_BODY_MAY26_SUBSAMPLE.csv",
        "wham_data": base / "ID_CNTRY_ALL_WHAM.csv",
        "output_parquet": base / "processed" / "data_v1.parquet"
    }


def classify_ai_roles(jobs_df, skills_df):
    """Flag jobs that require AI/ML skills and count AI skills per posting."""
    ai_skills = skills_df[
        skills_df["SKILL_SUBCATEGORY_NAME"] == "Artificial Intelligence and Machine Learning (AI/ML)"
    ]

    # Count AI/ML skills per posting
    ai_skill_counts = ai_skills.groupby("ID").size().rename("AI_SKILL_COUNT")
    jobs_df = jobs_df.merge(ai_skill_counts, on="ID", how="left")
    jobs_df["AI_SKILL_COUNT"] = jobs_df["AI_SKILL_COUNT"].fillna(0).astype(int)

    # Binary flag (1+ AI skills)
    jobs_df["AI ROLE"] = jobs_df["AI_SKILL_COUNT"] > 0

    print("AI Role distribution:")
    print(jobs_df["AI ROLE"].value_counts(dropna=False))
    print("\nAI skill count distribution (among AI roles):")
    print(jobs_df.loc[jobs_df["AI ROLE"], "AI_SKILL_COUNT"].describe())

    return jobs_df


def create_experience_buckets(df):
    """Create experience level categories from MIN_YEARS_EXPERIENCE."""
    bins = [-2, -1, 0, 2, 5, 10, 20, 100]
    labels = [
        "Missing",
        "0 years",
        "1-2 years",
        "3-5 years",
        "6-10 years",
        "11-20 years",
        "21+ years",
    ]

    df["EXPERIENCE_BUCKET"] = pd.cut(
        df["MIN_YEARS_EXPERIENCE"], bins=bins, labels=labels, right=True
    )
    df["EXPERIENCE_BUCKET"] = df["EXPERIENCE_BUCKET"].astype(str)
    df["EXPERIENCE_BUCKET"] = df["EXPERIENCE_BUCKET"].replace("nan", "None Listed")

    print("Experience bucket distribution:")
    print(df["EXPERIENCE_BUCKET"].value_counts(dropna=False))

    return df


def add_log_salary(df):
    """Add log-transformed salary column."""
    df["LOG_SALARY"] = np.log(df["SALARY"])
    return df


def add_year_column(df):
    """Extract year from POSTED date column."""
    df["YEAR"] = pd.to_datetime(df["POSTED"]).dt.year

    print("Year distribution:")
    print(df.groupby("YEAR").size())

    return df


def merge_wham_data(df, wham_df):
    """Add remote work classification from WHAM data."""
    # Filter for US data only
    wham_us = wham_df[wham_df["country2"] == "US"]

    df = df.merge(
        wham_us[["id", "wfh_wham_prob", "wfh_wham"]],
        left_on="ID",
        right_on="id",
        how="left",
    )
    df.drop(columns=["id"], inplace=True)

    matched = df["wfh_wham_prob"].notna().sum()
    print(f"WHAM matches: {matched:,} / {len(df):,} ({matched / len(df) * 100:.1f}%)")
    print("Missing WHAM data after merge:")
    print(df[df["wfh_wham_prob"].isna()].groupby("YEAR").size())

    return df


def main():
    """Main data preparation pipeline."""
    paths = load_data_paths()

    print("Loading posts data...")
    all_data = pd.read_csv(paths["posts_csv"])
    print(f"Posts loaded: {len(all_data):,} rows")

    print("Loading skills data...")
    skills_df = pd.read_csv(paths["skills_csv"])
    print(f"Skills loaded: {len(skills_df):,} rows")

    print("Loading body data...")
    body_df = pd.read_csv(paths["body_csv"])
    all_data = all_data.merge(body_df, on="ID", how="left")
    print(f"Body merged: {all_data['BODY'].notna().sum():,} / {len(all_data):,} posts have body text")

    # Classify AI roles from skills subcategory
    all_data = classify_ai_roles(all_data, skills_df)

    # Add year column
    all_data = add_year_column(all_data)

    # Add experience buckets
    all_data = create_experience_buckets(all_data)

    # Add log salary
    if "SALARY" in all_data.columns:
        all_data = add_log_salary(all_data)
        print("Log salary column added")

    # Load and merge WHAM data for missing remote work classifications
    print("Loading WHAM data...")
    if Path(paths["wham_data"]).exists():
        wham_data = pd.read_csv(paths["wham_data"])
        all_data = merge_wham_data(all_data, wham_data)
    else:
        print("Warning: WHAM data not found, continuing without merge")

    # Derive 2-digit NAICS industry name
    NAICS_2_DIGIT = {
        '11': 'Agriculture, Forestry, Fishing and Hunting',
        '21': 'Mining, Quarrying, and Oil and Gas Extraction',
        '22': 'Utilities',
        '23': 'Construction',
        '31': 'Manufacturing', '32': 'Manufacturing', '33': 'Manufacturing',
        '42': 'Wholesale Trade',
        '44': 'Retail Trade', '45': 'Retail Trade',
        '48': 'Transportation and Warehousing', '49': 'Transportation and Warehousing',
        '51': 'Information',
        '52': 'Finance and Insurance',
        '53': 'Real Estate and Rental and Leasing',
        '54': 'Professional, Scientific, and Technical Services',
        '55': 'Management of Companies and Enterprises',
        '56': 'Administrative and Support and Waste Management and Remediation Services',
        '61': 'Educational Services',
        '62': 'Health Care and Social Assistance',
        '71': 'Arts, Entertainment, and Recreation',
        '72': 'Accommodation and Food Services',
        '81': 'Other Services (except Public Administration)',
        '92': 'Public Administration',
    }
    all_data['NAICS_2022_2_DIGIT'] = all_data['NAICS_2022_6'].astype(str).str[:2]
    all_data['NAICS_2022_2_NAME'] = all_data['NAICS_2022_2_DIGIT'].map(NAICS_2_DIGIT)

    # Derive SOC major group from ONET code
    SOC_MAJOR_GROUPS = {
        '11': 'Management Occupations',
        '13': 'Business and Financial Operations Occupations',
        '15': 'Computer and Mathematical Occupations',
        '17': 'Architecture and Engineering Occupations',
        '19': 'Life, Physical, and Social Science Occupations',
        '21': 'Community and Social Service Occupations',
        '23': 'Legal Occupations',
        '25': 'Educational Instruction and Library Occupations',
        '27': 'Arts, Design, Entertainment, Sports, and Media Occupations',
        '29': 'Healthcare Practitioners and Technical Occupations',
        '31': 'Healthcare Support Occupations',
        '33': 'Protective Service Occupations',
        '35': 'Food Preparation and Serving Related Occupations',
        '37': 'Building and Grounds Cleaning and Maintenance Occupations',
        '39': 'Personal Care and Service Occupations',
        '41': 'Sales and Related Occupations',
        '43': 'Office and Administrative Support Occupations',
        '45': 'Farming, Fishing, and Forestry Occupations',
        '47': 'Construction and Extraction Occupations',
        '49': 'Installation, Maintenance, and Repair Occupations',
        '51': 'Production Occupations',
        '53': 'Transportation and Material Moving Occupations',
    }
    all_data['SOC_MAJOR_GROUP'] = all_data['ONET'].str[:2].map(SOC_MAJOR_GROUPS)

    # Save processed data
    output_path = Path(paths["output_parquet"])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"Saving processed data to {output_path}")
    all_data.to_parquet(output_path, compression="gzip")

    print("Data preparation complete!")
    print(f"Final dataset shape: {all_data.shape}")


if __name__ == "__main__":
    main()
