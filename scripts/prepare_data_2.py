"""
Secondary data preparation script for Beyond Salary analysis.
Adds experience buckets and log salary to prepared data.
"""

import pandas as pd
import numpy as np
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


def main():
    """Main data preparation pipeline."""
    base = Path(__file__).parent.parent / "data" / "processed"
    input_file = base / "data_v1.parquet"
    output_file = base / "data_v2.parquet"

    print("Loading data...")
    df = pd.read_parquet(input_file)
    print(f"Initial data shape: {df.shape}")

    # Create experience buckets
    df = create_experience_buckets(df)

    # Add log salary
    if "SALARY" in df.columns:
        df = add_log_salary(df)
        print("Log salary column added")

    print(f"Saving to {output_file}")
    df.to_parquet(output_file, compression="gzip")

    print("Data preparation complete!")
    print(f"Final dataset shape: {df.shape}")

    if "AI ROLE" in df.columns:
        print("\nFinal AI role distribution:")
        print(df["AI ROLE"].value_counts(dropna=False))


if __name__ == "__main__":
    main()
