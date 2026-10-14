"""Assemble the full-data labeled_v2.parquet WITHOUT ever holding body text.

Full-data replacement for prepare_data.py + label_benefits.py + label_benefits_remote.py:
the benefit/remote keyword flags are produced separately by the streaming
label_benefits*.py scripts (ID + flags only), so here we just join those compact
label files onto the posts data and replicate prepare_data.py's derived columns.

Inputs (config.yaml [data]):
  posts_csv                     full posts CSV (structured fields; no BODY)
  processed_dir/                benefit_labels_body.parquet   (ID + 6 keyword flags)
                                remote_labels_body.parquet    (ID + REMOTE_KW)
                                ai_skill_counts_by_id.parquet (ID + AI_SKILL_COUNT)
  wham_csv, sp500_csv

Output:
  <out-dir>/labeled_v2.parquet  (default: processed_dir/full/labeled_v2.parquet)

Memory: the posts CSV is read once with pyarrow; heavy unused text columns are
dropped before the pandas conversion, and joins use small ID-keyed frames.
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.csv as pacsv
import pyarrow.parquet as pq
import yaml

REPO = Path(__file__).resolve().parent.parent

# Columns no analysis/plot script reads — dropped to save memory (verified 2026-07).
DROP_COLUMNS = [
    "TITLE_RAW", "BENEFIT_NAME", "BENEFIT_CATEGORIES_NAME",
    "CITY_NAME", "ONET_NAME", "EMPLOYMENT_TYPE_NAME",
]

# Structured benefit labels derived from BENEFIT_SUBCATEGORY_NAME (no body needed).
STRUCTURED_SUBCATEGORY_LABELS = [
    ("S_FLEX_WORK", ["Flexible Work Schedules"]),
    ("S_PROF_DEV", ["Professional Development", "Leadership Development", "Mentorships"]),
    ("S_HEALTH_WELLNESS", [
        "Health and Wellness Programs", "Mental Health", "Bereavement/Mental Health Leave",
        "Health and Wellness Stipends", "Health and Wellness Applications",
    ]),
]

NAICS_2_DIGIT = {
    '11': 'Agriculture, Forestry, Fishing and Hunting',
    '21': 'Mining, Quarrying, and Oil and Gas Extraction',
    '22': 'Utilities', '23': 'Construction',
    '31': 'Manufacturing', '32': 'Manufacturing', '33': 'Manufacturing',
    '42': 'Wholesale Trade', '44': 'Retail Trade', '45': 'Retail Trade',
    '48': 'Transportation and Warehousing', '49': 'Transportation and Warehousing',
    '51': 'Information', '52': 'Finance and Insurance',
    '53': 'Real Estate and Rental and Leasing',
    '54': 'Professional, Scientific, and Technical Services',
    '55': 'Management of Companies and Enterprises',
    '56': 'Administrative and Support and Waste Management and Remediation Services',
    '61': 'Educational Services', '62': 'Health Care and Social Assistance',
    '71': 'Arts, Entertainment, and Recreation',
    '72': 'Accommodation and Food Services',
    '81': 'Other Services (except Public Administration)',
    '92': 'Public Administration',
}
SOC_MAJOR_GROUPS = {
    '11': 'Management Occupations', '13': 'Business and Financial Operations Occupations',
    '15': 'Computer and Mathematical Occupations', '17': 'Architecture and Engineering Occupations',
    '19': 'Life, Physical, and Social Science Occupations', '21': 'Community and Social Service Occupations',
    '23': 'Legal Occupations', '25': 'Educational Instruction and Library Occupations',
    '27': 'Arts, Design, Entertainment, Sports, and Media Occupations',
    '29': 'Healthcare Practitioners and Technical Occupations', '31': 'Healthcare Support Occupations',
    '33': 'Protective Service Occupations', '35': 'Food Preparation and Serving Related Occupations',
    '37': 'Building and Grounds Cleaning and Maintenance Occupations',
    '39': 'Personal Care and Service Occupations', '41': 'Sales and Related Occupations',
    '43': 'Office and Administrative Support Occupations', '45': 'Farming, Fishing, and Forestry Occupations',
    '47': 'Construction and Extraction Occupations', '49': 'Installation, Maintenance, and Repair Occupations',
    '51': 'Production Occupations', '53': 'Transportation and Material Moving Occupations',
}


def load_cfg():
    cfg_path = REPO / "config.yaml"
    if not cfg_path.exists():
        raise SystemExit("config.yaml not found; copy config.example.yaml and set the full-data paths.")
    with open(cfg_path) as f:
        cfg = yaml.safe_load(f)["data"]
    raw = Path(cfg["raw_dir"])
    raw = raw if raw.is_absolute() else REPO / raw
    proc = Path(cfg["processed_dir"])
    proc = proc if proc.is_absolute() else REPO / proc
    return {
        "posts_csv": (raw / cfg["posts_csv"]) if not Path(cfg["posts_csv"]).is_absolute() else Path(cfg["posts_csv"]),
        "wham_csv": raw / cfg["wham_csv"],
        "sp500_csv": (raw / cfg["sp500_csv"]) if cfg.get("sp500_csv") else None,
        "processed_dir": proc,
    }


def parse_benefit_list(val):
    if isinstance(val, str) and val.startswith("["):
        try:
            return json.loads(val)
        except (json.JSONDecodeError, ValueError):
            return []
    return []


def add_structured_labels(df):
    parsed = df["BENEFIT_SUBCATEGORY_NAME"].apply(parse_benefit_list)
    for name, terms in STRUCTURED_SUBCATEGORY_LABELS:
        terms_lower = {t.lower() for t in terms}
        df[name] = parsed.apply(lambda lst: any(v.lower() in terms_lower for v in lst if isinstance(v, str)))
    df["S_REMOTE"] = df["REMOTE_TYPE_NAME"].isin(["Remote", "Hybrid Remote"])
    return df


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, default=None,
                    help="Output parquet (default: processed_dir/full/labeled_v2.parquet)")
    ap.add_argument("--keep-body-cols", action="store_true",
                    help="Do not drop the heavy unused text columns (default: drop them)")
    args = ap.parse_args()

    cfg = load_cfg()
    proc = cfg["processed_dir"]
    out = args.out or (proc / "full" / "labeled_v2.parquet")
    out.parent.mkdir(parents=True, exist_ok=True)
    start = time.time()

    # --- Load posts CSV via pyarrow, drop heavy columns before pandas ---
    print(f"Reading posts CSV: {cfg['posts_csv']}")
    table = pacsv.read_csv(
        cfg["posts_csv"],
        read_options=pacsv.ReadOptions(use_threads=True, block_size=256 << 20),
        parse_options=pacsv.ParseOptions(newlines_in_values=True),
    )
    if not args.keep_body_cols:
        drop = [c for c in DROP_COLUMNS if c in table.column_names]
        table = table.drop(drop)
        print(f"  dropped unused columns: {drop}")
    df = table.to_pandas()
    del table
    print(f"  posts rows: {len(df):,} ({time.time() - start:.0f}s)")

    # --- Join label files (ID-keyed, small) ---
    def join_labels(path, cols, how="left"):
        nonlocal df
        if not path.exists():
            raise SystemExit(f"Required label file missing: {path}")
        lab = pd.read_parquet(path, columns=["ID"] + cols)
        df = df.merge(lab, on="ID", how=how)
        print(f"  joined {cols} from {path.name}")

    join_labels(proc / "benefit_labels_body.parquet",
                ["EDU_ASSISTANCE", "PAID LEAVE", "WELLBEING", "HEALTH_WELLBEING",
                 "PARENTAL_LEAVE", "CULTURE"])
    join_labels(proc / "remote_labels_body.parquet", ["REMOTE_KW"])
    # Benefit keyword flags are False (not null) where a posting had no body row.
    for col in ["EDU_ASSISTANCE", "PAID LEAVE", "WELLBEING", "HEALTH_WELLBEING",
                "PARENTAL_LEAVE", "CULTURE", "REMOTE_KW"]:
        df[col] = df[col].fillna(False).astype(bool)

    # --- AI role from precomputed skill counts ---
    ai = pd.read_parquet(proc / "ai_skill_counts_by_id.parquet")
    df = df.merge(ai, on="ID", how="left")
    df["AI_SKILL_COUNT"] = df["AI_SKILL_COUNT"].fillna(0).astype(int)
    df["AI ROLE"] = df["AI_SKILL_COUNT"] > 0
    print(f"  AI roles: {int(df['AI ROLE'].sum()):,}")

    # --- Derived columns (mirror prepare_data.py) ---
    df["YEAR"] = pd.to_datetime(df["POSTED"]).dt.year
    naics = df["NAICS_2022_6"].astype("Int64").astype(str)
    df["NAICS_2022_2_DIGIT"] = naics.str[:2]
    df["NAICS_2022_3_DIGIT"] = naics.str[:3]
    df["NAICS_2022_2_NAME"] = df["NAICS_2022_2_DIGIT"].map(NAICS_2_DIGIT)
    df["SOC_MAJOR_GROUP"] = df["ONET"].str[:2].map(SOC_MAJOR_GROUPS)

    classified = df["COMPANY"].notna() & (df["COMPANY"] != 0)
    firm_counts = df.loc[classified, "COMPANY"].value_counts()
    df["FIRM_POSTING_COUNT"] = df["COMPANY"].map(firm_counts).astype("Int64")
    df["FIRM_SIZE_BUCKET"] = pd.cut(df["FIRM_POSTING_COUNT"], bins=[0, 49, np.inf],
                                    labels=["SMEs (<50)", "Large firms (>=50)"]).astype("object")

    # S&P 500 from the corrected snapshot
    df["SP500"] = False
    if cfg["sp500_csv"] and cfg["sp500_csv"].exists():
        snap = pd.read_csv(cfg["sp500_csv"])
        snap["COMPANY"] = pd.to_numeric(snap["COMPANY"], errors="coerce")
        snap = snap[snap["COMPANY"].notna() & (snap["COMPANY"] != 0)]
        sp_ids = set(snap["COMPANY"].astype(df["COMPANY"].dtype))
        df["SP500"] = df["COMPANY"].isin(sp_ids) & (df["COMPANY"] != 0)
        df.loc[df["SP500"], "FIRM_SIZE_BUCKET"] = "S&P 500 firms"
        print(f"  S&P 500 postings: {int(df['SP500'].sum()):,} (from {len(sp_ids):,} firms)")
    else:
        print("  WARNING: no S&P snapshot configured; SP500 all False")

    df["EXPERIENCE_BUCKET"] = pd.cut(
        df["MIN_YEARS_EXPERIENCE"], bins=[-2, -1, 0, 2, 5, 10, 20, 100],
        labels=["Missing", "0 years", "1-2 years", "3-5 years", "6-10 years", "11-20 years", "21+ years"],
        right=True,
    ).astype(str).replace("nan", "None Listed")

    if "SALARY" in df.columns:
        df["LOG_SALARY"] = np.log(df["SALARY"])

    df = add_structured_labels(df)

    # WHAM remote classification (US only)
    if cfg["wham_csv"].exists():
        wham = pd.read_csv(cfg["wham_csv"])
        wham = wham[wham["country2"] == "US"][["id", "wfh_wham_prob", "wfh_wham"]]
        df = df.merge(wham, left_on="ID", right_on="id", how="left").drop(columns=["id"])
        print(f"  WHAM matched: {df['wfh_wham_prob'].notna().sum():,}")
    else:
        print("  WARNING: WHAM file not found; skipping")

    print(f"Writing {out}  shape={df.shape}")
    df.to_parquet(out, compression="gzip")
    print(f"Done in {time.time() - start:.0f}s")


if __name__ == "__main__":
    main()
