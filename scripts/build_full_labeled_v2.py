"""Assemble the full-data labeled_v2 parquet.

The large joins happen in PyArrow before converting to pandas:
posts + keyword benefit labels + remote keyword labels + AI skill counts.
Pandas is used only for the downstream transformations that mirror
scripts/prepare_data.py.
"""

import argparse
import re
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.csv as pacsv
import pyarrow.dataset as ds
import pyarrow.parquet as pq


REPO_ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = REPO_ROOT / "data"
PROCESSED_DIR = RAW_DIR / "processed"
FULL_PROCESSED_DIR = RAW_DIR / "processed_full"

DEFAULT_POSTS_CSV = RAW_DIR / "OII_US_10M_POSTS_MAY26.csv"
DEFAULT_BENEFIT_LABELS = PROCESSED_DIR / "benefit_labels_body.parquet"
DEFAULT_REMOTE_LABELS = PROCESSED_DIR / "remote_labels_body.parquet"
DEFAULT_AI_COUNTS = PROCESSED_DIR / "ai_skill_counts_by_id.parquet"
DEFAULT_SP500 = RAW_DIR / "sp500_snapshot.csv"
DEFAULT_OUTPUT = FULL_PROCESSED_DIR / "labeled_v2.parquet"

POST_COLUMNS = [
    "ID",
    "POSTED",
    "COMPANY",
    "COMPANY_NAME",
    "MIN_EDULEVELS_NAME",
    "MIN_YEARS_EXPERIENCE",
    "SALARY",
    "REMOTE_TYPE_NAME",
    "STATE_NAME",
    "NAICS_2022_6",
    "NAICS_2022_6_NAME",
    "ONET",
    "BENEFIT_SUBCATEGORY_NAME",
    "QUARTER",
]

BENEFIT_COLUMNS = [
    "ID",
    "EDU_ASSISTANCE",
    "PAID LEAVE",
    "WELLBEING",
    "HEALTH_WELLBEING",
    "PARENTAL_LEAVE",
    "CULTURE",
]

REMOTE_COLUMNS = ["ID", "REMOTE_KW", "REMOTE_KW_POSITION"]

NAICS_2_DIGIT = {
    "11": "Agriculture, Forestry, Fishing and Hunting",
    "21": "Mining, Quarrying, and Oil and Gas Extraction",
    "22": "Utilities",
    "23": "Construction",
    "31": "Manufacturing",
    "32": "Manufacturing",
    "33": "Manufacturing",
    "42": "Wholesale Trade",
    "44": "Retail Trade",
    "45": "Retail Trade",
    "48": "Transportation and Warehousing",
    "49": "Transportation and Warehousing",
    "51": "Information",
    "52": "Finance and Insurance",
    "53": "Real Estate and Rental and Leasing",
    "54": "Professional, Scientific, and Technical Services",
    "55": "Management of Companies and Enterprises",
    "56": "Administrative and Support and Waste Management and Remediation Services",
    "61": "Educational Services",
    "62": "Health Care and Social Assistance",
    "71": "Arts, Entertainment, and Recreation",
    "72": "Accommodation and Food Services",
    "81": "Other Services (except Public Administration)",
    "92": "Public Administration",
}

SOC_MAJOR_GROUPS = {
    "11": "Management Occupations",
    "13": "Business and Financial Operations Occupations",
    "15": "Computer and Mathematical Occupations",
    "17": "Architecture and Engineering Occupations",
    "19": "Life, Physical, and Social Science Occupations",
    "21": "Community and Social Service Occupations",
    "23": "Legal Occupations",
    "25": "Educational Instruction and Library Occupations",
    "27": "Arts, Design, Entertainment, Sports, and Media Occupations",
    "29": "Healthcare Practitioners and Technical Occupations",
    "31": "Healthcare Support Occupations",
    "33": "Protective Service Occupations",
    "35": "Food Preparation and Serving Related Occupations",
    "37": "Building and Grounds Cleaning and Maintenance Occupations",
    "39": "Personal Care and Service Occupations",
    "41": "Sales and Related Occupations",
    "43": "Office and Administrative Support Occupations",
    "45": "Farming, Fishing, and Forestry Occupations",
    "47": "Construction and Extraction Occupations",
    "49": "Installation, Maintenance, and Repair Occupations",
    "51": "Production Occupations",
    "53": "Transportation and Material Moving Occupations",
}


def read_csv_table(path, columns, block_mb, limit_rows=None):
    read_options = pacsv.ReadOptions(use_threads=True, block_size=block_mb << 20)
    parse_options = pacsv.ParseOptions(newlines_in_values=True)
    convert_options = pacsv.ConvertOptions(
        include_columns=columns,
        strings_can_be_null=True,
    )

    if limit_rows is None:
        return pacsv.read_csv(
            path,
            read_options=read_options,
            parse_options=parse_options,
            convert_options=convert_options,
        )

    reader = pacsv.open_csv(
        path,
        read_options=read_options,
        parse_options=parse_options,
        convert_options=convert_options,
    )
    batches = []
    rows = 0
    while rows < limit_rows:
        try:
            batch = reader.read_next_batch()
        except StopIteration:
            break
        if batch is None or batch.num_rows == 0:
            break
        if rows + batch.num_rows > limit_rows:
            batch = batch.slice(0, limit_rows - rows)
        batches.append(batch)
        rows += batch.num_rows
    return pa.Table.from_batches(batches) if batches else pa.table({c: [] for c in columns})


def read_parquet_columns(path, columns=None):
    table = pq.read_table(path, columns=columns)
    return table.combine_chunks()


def read_remote_labels(path, required_rows=None, allow_partial=False):
    columns = None
    if path.exists():
        schema_names = set(pq.ParquetFile(path).schema.names)
        columns = [c for c in REMOTE_COLUMNS if c in schema_names]
        table = pq.read_table(path, columns=columns).combine_chunks()
    else:
        parts_dir = path.with_name(path.name + ".parts")
        if not parts_dir.exists():
            raise FileNotFoundError(
                f"Remote label source not found: {path} or {parts_dir}"
            )
        dataset = ds.dataset(parts_dir, format="parquet")
        schema_names = set(dataset.schema.names)
        columns = [c for c in REMOTE_COLUMNS if c in schema_names]
        table = dataset.to_table(columns=columns).combine_chunks()

    if "REMOTE_KW" not in table.column_names:
        raise ValueError("Remote label source must contain REMOTE_KW.")
    if required_rows is not None and table.num_rows != required_rows and not allow_partial:
        raise RuntimeError(
            f"Remote labels have {table.num_rows:,} rows but posts have "
            f"{required_rows:,}. Finish scripts/label_benefits_remote.py before "
            "building the full labeled_v2, or rerun with --allow-partial-remote "
            "only for smoke tests."
        )
    return table


def left_join(left, right, key="ID"):
    before = left.num_rows
    joined = left.join(right, keys=key, join_type="left outer", use_threads=True)
    if joined.num_rows != before:
        raise RuntimeError(
            f"Join on {key} changed row count from {before:,} to {joined.num_rows:,}."
        )
    return joined.combine_chunks()


def add_structured_labels(df):
    subcat = df["BENEFIT_SUBCATEGORY_NAME"].fillna("")

    def contains_any(terms):
        pattern = "|".join(re.escape(term) for term in terms)
        return subcat.str.contains(pattern, case=False, regex=True, na=False)

    df["S_FLEX_WORK"] = contains_any(["Flexible Work Schedules"])
    df["S_PROF_DEV"] = contains_any(
        ["Professional Development", "Leadership Development", "Mentorships"]
    )
    df["S_HEALTH_WELLNESS"] = contains_any(
        [
            "Health and Wellness Programs",
            "Mental Health",
            "Bereavement/Mental Health Leave",
            "Health and Wellness Stipends",
            "Health and Wellness Applications",
        ]
    )
    df["S_REMOTE"] = df["REMOTE_TYPE_NAME"].isin(["Remote", "Hybrid Remote"])


def add_prepare_data_transforms(df, sp500_path, audit_path):
    df["POSTED"] = pd.to_datetime(df["POSTED"], errors="coerce")
    df["YEAR"] = df["POSTED"].dt.year.astype("Int64")
    df["POSTED"] = df["POSTED"].dt.strftime("%Y-%m-%d")

    naics_code = df["NAICS_2022_6"].astype("Int64").astype(str)
    df["NAICS_2022_2_DIGIT"] = naics_code.str[:2]
    df["NAICS_2022_3_DIGIT"] = naics_code.str[:3]

    classified_firms = df["COMPANY"].notna() & (df["COMPANY"] != 0)
    firm_counts = df.loc[classified_firms, "COMPANY"].value_counts()
    df["FIRM_POSTING_COUNT"] = df["COMPANY"].map(firm_counts).astype("Int64")
    df["FIRM_SIZE_BUCKET"] = pd.cut(
        df["FIRM_POSTING_COUNT"],
        bins=[0, 49, np.inf],
        labels=["SMEs (<50)", "Large firms (>=50)"],
    )

    df["SP500"] = False
    if sp500_path and sp500_path.exists():
        snapshot = pd.read_csv(sp500_path)
        if "COMPANY" not in snapshot.columns:
            raise ValueError("S&P 500 snapshot must contain a COMPANY column.")
        snapshot["COMPANY"] = pd.to_numeric(snapshot["COMPANY"], errors="coerce")
        snapshot = snapshot[snapshot["COMPANY"].notna() & (snapshot["COMPANY"] != 0)]
        snapshot["COMPANY"] = snapshot["COMPANY"].astype(df["COMPANY"].dtype)
        snapshot = snapshot.drop_duplicates(subset=["COMPANY"])

        sp500_ids = set(snapshot["COMPANY"])
        df["SP500"] = df["COMPANY"].isin(sp500_ids) & (df["COMPANY"] != 0)
        df["FIRM_SIZE_BUCKET"] = df["FIRM_SIZE_BUCKET"].astype("object")
        df.loc[df["SP500"], "FIRM_SIZE_BUCKET"] = "S&P 500 firms"

        audit_columns = ["COMPANY"]
        if "COMPANY_NAME" in snapshot.columns:
            audit_columns.append("COMPANY_NAME")
        audit = snapshot[audit_columns].copy()
        audit["MATCHED_POSTINGS"] = (
            audit["COMPANY"].map(df["COMPANY"].value_counts()).fillna(0).astype(int)
        )
        audit["MATCHED_IN_POSTINGS"] = audit["MATCHED_POSTINGS"] > 0
        audit_path.parent.mkdir(parents=True, exist_ok=True)
        audit.to_csv(audit_path, index=False)

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
    df["EXPERIENCE_BUCKET"] = df["EXPERIENCE_BUCKET"].astype("object").where(
        df["EXPERIENCE_BUCKET"].notna(), "None Listed"
    )
    df["EXPERIENCE_BUCKET"] = df["EXPERIENCE_BUCKET"].astype(str)

    df["LOG_SALARY"] = np.log(df["SALARY"])
    df["NAICS_2022_2_NAME"] = df["NAICS_2022_2_DIGIT"].map(NAICS_2_DIGIT)
    df["SOC_MAJOR_GROUP"] = df["ONET"].str[:2].map(SOC_MAJOR_GROUPS)


def finalize_columns(df):
    bool_cols = [
        "AI ROLE",
        "EDU_ASSISTANCE",
        "PAID LEAVE",
        "WELLBEING",
        "HEALTH_WELLBEING",
        "PARENTAL_LEAVE",
        "CULTURE",
        "S_FLEX_WORK",
        "S_PROF_DEV",
        "S_HEALTH_WELLNESS",
        "S_REMOTE",
        "REMOTE_KW",
        "SP500",
    ]
    for col in bool_cols:
        if col in df.columns:
            df[col] = df[col].fillna(False).astype(bool)

    if "AI_SKILL_COUNT" in df.columns:
        df["AI_SKILL_COUNT"] = df["AI_SKILL_COUNT"].fillna(0).astype("int64")
    else:
        df["AI_SKILL_COUNT"] = 0
    df["AI ROLE"] = df["AI_SKILL_COUNT"] > 0

    ordered = [
        "ID",
        "POSTED",
        "COMPANY",
        "COMPANY_NAME",
        "MIN_EDULEVELS_NAME",
        "MIN_YEARS_EXPERIENCE",
        "SALARY",
        "REMOTE_TYPE_NAME",
        "STATE_NAME",
        "NAICS_2022_6",
        "NAICS_2022_6_NAME",
        "ONET",
        "BENEFIT_SUBCATEGORY_NAME",
        "YEAR",
        "QUARTER",
        "AI_SKILL_COUNT",
        "AI ROLE",
        "NAICS_2022_2_DIGIT",
        "NAICS_2022_3_DIGIT",
        "FIRM_POSTING_COUNT",
        "FIRM_SIZE_BUCKET",
        "SP500",
        "EXPERIENCE_BUCKET",
        "LOG_SALARY",
        "NAICS_2022_2_NAME",
        "SOC_MAJOR_GROUP",
        "EDU_ASSISTANCE",
        "PAID LEAVE",
        "WELLBEING",
        "HEALTH_WELLBEING",
        "PARENTAL_LEAVE",
        "CULTURE",
        "S_FLEX_WORK",
        "S_PROF_DEV",
        "S_HEALTH_WELLNESS",
        "S_REMOTE",
        "REMOTE_KW",
        "REMOTE_KW_POSITION",
    ]
    return df[[col for col in ordered if col in df.columns]]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--posts-csv", type=Path, default=DEFAULT_POSTS_CSV)
    parser.add_argument("--benefit-labels", type=Path, default=DEFAULT_BENEFIT_LABELS)
    parser.add_argument("--remote-labels", type=Path, default=DEFAULT_REMOTE_LABELS)
    parser.add_argument("--ai-counts", type=Path, default=DEFAULT_AI_COUNTS)
    parser.add_argument("--sp500-csv", type=Path, default=DEFAULT_SP500)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--block-mb", type=int, default=256)
    parser.add_argument("--limit-rows", type=int, default=None)
    parser.add_argument("--allow-partial-remote", action="store_true")
    args = parser.parse_args()

    start = time.time()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    audit_path = args.output.parent / "sp500_match_audit.csv"

    print(f"Reading posts: {args.posts_csv}")
    posts = read_csv_table(args.posts_csv, POST_COLUMNS, args.block_mb, args.limit_rows)
    print(f"  posts rows: {posts.num_rows:,}")

    print(f"Reading benefit labels: {args.benefit_labels}")
    benefits = read_parquet_columns(args.benefit_labels, BENEFIT_COLUMNS)
    print(f"  benefit rows: {benefits.num_rows:,}")

    print(f"Reading remote labels: {args.remote_labels}")
    required_remote_rows = None if args.limit_rows else posts.num_rows
    remote = read_remote_labels(
        args.remote_labels,
        required_rows=required_remote_rows,
        allow_partial=args.allow_partial_remote,
    )
    print(f"  remote rows: {remote.num_rows:,}")

    print(f"Reading AI skill counts: {args.ai_counts}")
    ai_counts = read_parquet_columns(args.ai_counts, ["ID", "AI_SKILL_COUNT"])
    print(f"  AI-count rows: {ai_counts.num_rows:,}")

    print("Joining in Arrow...")
    table = left_join(posts, benefits)
    table = left_join(table, remote)
    table = left_join(table, ai_counts)
    print(f"  joined rows: {table.num_rows:,}, columns: {table.num_columns}")

    print("Converting joined table to pandas...")
    df = table.to_pandas(split_blocks=True, self_destruct=True)
    del table, posts, benefits, remote, ai_counts
    print(f"  pandas shape: {df.shape}")

    print("Applying prepare_data.py transforms and structured labels...")
    add_prepare_data_transforms(df, args.sp500_csv, audit_path)
    add_structured_labels(df)
    df = finalize_columns(df)

    print(f"Writing {args.output}")
    output_table = pa.Table.from_pandas(df, preserve_index=False)
    pq.write_table(output_table, args.output, compression="gzip")

    print("Done.")
    print(f"Rows: {len(df):,}")
    print(f"Columns: {len(df.columns):,}")
    print(f"Elapsed: {(time.time() - start) / 60:.1f} minutes")


if __name__ == "__main__":
    main()
