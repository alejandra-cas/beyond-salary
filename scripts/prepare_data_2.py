"""
Secondary data preparation script for Beyond Salary analysis.
Adds experience buckets, AI skills classification, and salary processing.
"""

import pandas as pd
import numpy as np
import pickle
import swifter
from pathlib import Path


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


def merge_additional_salaries(df, salary_file):
    """Merge additional salary data if available."""
    if Path(salary_file).exists():
        salaries = pd.read_csv(salary_file)
        df = df.merge(salaries, left_on="ID", right_on="ID", how="left")
        print("Additional salary data merged successfully")
    else:
        print("Warning: Additional salary file not found, skipping merge")

    return df


def classify_ai_skills(df, ai_skill_ids):
    """Classify remaining jobs for AI skills using skill IDs."""
    # Find rows where AI ROLE is missing
    ai_na = df[df["AI ROLE"].isna()].copy()

    if len(ai_na) == 0:
        print("No missing AI classifications found")
        return df

    print(f"Classifying {len(ai_na):,} jobs for AI skills...")

    def has_ai_skills(skill_list, ai_skill_ids):
        """Check if any AI skill ID is in the job's skills."""
        try:
            skill_list = eval(skill_list)  # Convert string to list
            return any(skill_id in skill_list for skill_id in ai_skill_ids)
        except:
            return False

    # Apply AI skills classification
    ai_na["AI ROLE"] = ai_na.swifter.progress_bar(True).apply(
        lambda x: has_ai_skills(x["SKILLS"], ai_skill_ids), axis=1
    )

    # Update original dataframe
    df.loc[ai_na.index, "AI ROLE"] = ai_na["AI ROLE"]

    # Verify no missing values remain
    remaining_na = df[df["AI ROLE"].isna()]
    print(f"Remaining NA values: {len(remaining_na)}")

    return df


def print_sample_ai_skills(df, year=2018, n_samples=3):
    """Print sample AI role skills for verification."""
    print(f"\nSample AI role skills from {year}:")
    sample_data = df[(df["AI ROLE"] == True) & (df["YEAR"] == year)].head(n_samples)

    for i, row in sample_data.iterrows():
        print(f"\nSample {i+1}:")
        print(row["SKILLS_NAME"])


def main():
    """Main data preparation pipeline."""
    # Configuration
    input_file = "../data/us_10m_nointernship_2018_2024_benefits.parquet.gzip"
    salary_file = "../data/SALARIES.csv"
    ai_skills_file = "../data/ai_skill_ids.pkl"
    output_with_body = "../data/us_10m_nointernship_ai_skills_body.parquet.gzip"
    output_without_body = "../data/us_10m_nointernship_ai_skills_benefits.parquet.gzip"

    print("Loading data...")
    if input_file.endswith(".csv"):
        df = pd.read_csv(input_file)
    else:
        df = pd.read_parquet(input_file)

    print(f"Initial data shape: {df.shape}")

    # Create experience buckets
    df = create_experience_buckets(df)

    # Add log salary
    if "SALARY" in df.columns:
        df = add_log_salary(df)
        print("Log salary column added")

    # Merge additional salary data
    df = merge_additional_salaries(df, salary_file)

    # Load AI skill IDs for classification
    if Path(ai_skills_file).exists():
        with open(ai_skills_file, "rb") as f:
            ai_skill_ids = pickle.load(f)

        print(f"Loaded {len(ai_skill_ids)} AI skill IDs")

        # Classify remaining AI roles
        df = classify_ai_skills(df, ai_skill_ids)

        # Print sample for verification
        print_sample_ai_skills(df)
    else:
        print("Warning: AI skills file not found, skipping AI classification")

    # Save outputs
    print(f"Saving data with body to {output_with_body}")
    df.to_parquet(output_with_body, compression="gzip")

    print(f"Saving data without body to {output_without_body}")
    df.drop(columns=["BODY"], errors="ignore").to_parquet(
        output_without_body, compression="gzip"
    )

    print("Data preparation complete!")
    print(f"Final dataset shape: {df.shape}")

    # Print final AI role distribution
    if "AI ROLE" in df.columns:
        print("\nFinal AI role distribution:")
        print(df["AI ROLE"].value_counts(dropna=False))


if __name__ == "__main__":
    main()
