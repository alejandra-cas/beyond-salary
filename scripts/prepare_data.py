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
    """Flag jobs that require AI/ML skills using SKILL_SUBCATEGORY_NAME."""
    ai_job_ids = skills_df[
        skills_df["SKILL_SUBCATEGORY_NAME"] == "Artificial Intelligence and Machine Learning (AI/ML)"
    ]["ID"].unique()

    jobs_df["AI ROLE"] = jobs_df["ID"].isin(ai_job_ids)

    print("AI Role distribution:")
    print(jobs_df["AI ROLE"].value_counts(dropna=False))

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

    # Save processed data
    output_path = Path(paths["output_parquet"])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"Saving processed data to {output_path}")
    all_data.to_parquet(output_path, compression="gzip")

    print("Data preparation complete!")
    print(f"Final dataset shape: {all_data.shape}")


if __name__ == "__main__":
    main()
