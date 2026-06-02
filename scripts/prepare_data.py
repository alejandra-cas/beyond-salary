"""
Data preparation script for Beyond Salary analysis.
Merges job posting data with AI skills labels and remote work classification.
"""

import pandas as pd
import numpy as np
import yaml
from pathlib import Path


def load_data_paths():
    """Load input/output data paths from config.yaml.

    Falls back to defaults if config.yaml is not found.
    See config.example.yaml for the expected format.
    """
    repo_root = Path(__file__).parent.parent
    config_path = repo_root / "config.yaml"

    if config_path.exists():
        with open(config_path) as f:
            cfg = yaml.safe_load(f)["data"]
        raw_dir = Path(cfg["raw_dir"])
        if not raw_dir.is_absolute():
            raw_dir = repo_root / raw_dir
        processed_dir = Path(cfg["processed_dir"])
        if not processed_dir.is_absolute():
            processed_dir = repo_root / processed_dir
        return {
            "posts_csv": raw_dir / cfg["posts_csv"],
            "skills_csv": raw_dir / cfg["skills_csv"],
            "body_csv": raw_dir / cfg["body_csv"],
            "wham_data": raw_dir / cfg["wham_csv"],
            "sp500_csv": raw_dir / cfg["sp500_csv"] if cfg.get("sp500_csv") else None,
            "output_parquet": processed_dir / "data_v1.parquet",
            "sp500_audit_csv": processed_dir / "sp500_match_audit.csv",
        }
    else:
        print("Warning: config.yaml not found, using default paths. "
              "Copy config.example.yaml to config.yaml to configure.")
        base = repo_root / "data"
        return {
            "posts_csv": base / "OII_US_10M_POSTS_MAY26_SUBSAMPLE.csv",
            "skills_csv": base / "OII_US_10M_SKILLS_MAY26_SUBSAMPLE.csv",
            "body_csv": base / "OII_US_10M_BODY_MAY26_SUBSAMPLE.csv",
            "wham_data": base / "ID_CNTRY_ALL_WHAM.csv",
            "sp500_csv": None,
            "output_parquet": base / "processed" / "data_v1.parquet",
            "sp500_audit_csv": base / "processed" / "sp500_match_audit.csv",
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


def add_industry_columns(df):
    """Derive 2-digit and 3-digit NAICS industry columns."""
    naics_code = df["NAICS_2022_6"].astype("Int64").astype(str)
    df["NAICS_2022_2_DIGIT"] = naics_code.str[:2]
    df["NAICS_2022_3_DIGIT"] = naics_code.str[:3]
    return df


def add_firm_size(df):
    """Add posting-volume firm size buckets, excluding unclassified employers."""
    df = df.copy()
    classified_firms = df["COMPANY"].notna() & (df["COMPANY"] != 0)
    firm_counts = df.loc[classified_firms, "COMPANY"].value_counts()

    df["FIRM_POSTING_COUNT"] = df["COMPANY"].map(firm_counts).astype("Int64")
    df["FIRM_SIZE_BUCKET"] = pd.cut(
        df["FIRM_POSTING_COUNT"],
        bins=[0, 2, 9, np.inf],
        labels=["Small (<=2)", "Medium (3-9)", "Large (>=10)"],
    )

    print("Firm size bucket distribution:")
    print(df["FIRM_SIZE_BUCKET"].value_counts(dropna=False))
    return df


def merge_sp500_snapshot(df, snapshot_path, audit_path):
    """Join a fixed S&P 500 constituent snapshot keyed by COMPANY."""
    df = df.copy()
    df["SP500"] = False

    if snapshot_path is None or not Path(snapshot_path).exists():
        print("Warning: S&P 500 snapshot not configured or not found; SP500 set to False.")
        return df

    snapshot = pd.read_csv(snapshot_path)
    if "COMPANY" not in snapshot.columns:
        raise ValueError("S&P 500 snapshot must contain a COMPANY column.")

    snapshot = snapshot.copy()
    snapshot["COMPANY"] = pd.to_numeric(snapshot["COMPANY"], errors="coerce")
    snapshot = snapshot[snapshot["COMPANY"].notna() & (snapshot["COMPANY"] != 0)]
    snapshot["COMPANY"] = snapshot["COMPANY"].astype(df["COMPANY"].dtype)
    snapshot = snapshot.drop_duplicates(subset=["COMPANY"])

    sp500_ids = set(snapshot["COMPANY"])
    df["SP500"] = df["COMPANY"].isin(sp500_ids) & (df["COMPANY"] != 0)

    audit_columns = ["COMPANY"]
    if "COMPANY_NAME" in snapshot.columns:
        audit_columns.append("COMPANY_NAME")
    audit = snapshot[audit_columns].copy()
    audit["MATCHED_POSTINGS"] = audit["COMPANY"].map(df["COMPANY"].value_counts()).fillna(0).astype(int)
    audit["MATCHED_IN_POSTINGS"] = audit["MATCHED_POSTINGS"] > 0
    Path(audit_path).parent.mkdir(parents=True, exist_ok=True)
    audit.to_csv(audit_path, index=False)

    print(f"S&P 500 snapshot companies: {len(snapshot):,}")
    print(f"S&P 500 companies matched to postings: {audit['MATCHED_IN_POSTINGS'].sum():,}")
    print(f"S&P 500 postings: {df['SP500'].sum():,}")
    print(f"S&P 500 match audit saved to {audit_path}")
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

    # Add industry columns
    all_data = add_industry_columns(all_data)

    # Add posting-volume firm size buckets
    all_data = add_firm_size(all_data)

    # Join fixed S&P 500 constituent snapshot when configured
    all_data = merge_sp500_snapshot(
        all_data,
        paths["sp500_csv"],
        paths["sp500_audit_csv"],
    )

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
