#!/usr/bin/env python3
"""Apply the fixed S&P 500 snapshot to existing processed parquet files."""

import argparse
from collections import Counter
from pathlib import Path

import pandas as pd
import pyarrow.csv as pacsv
import yaml


DEFAULT_PARQUETS = ("data_v1.parquet", "labeled_v1.parquet", "labeled_v2.parquet")


def load_default_paths():
    """Return default processed and S&P 500 snapshot paths."""
    repo_root = Path(__file__).parent.parent
    config_path = repo_root / "config.yaml"

    processed_dir = repo_root / "data" / "processed"
    snapshot_path = repo_root / "data" / "sp500_snapshot.csv"

    if config_path.exists():
        with open(config_path) as f:
            cfg = yaml.safe_load(f)["data"]

        processed_dir = Path(cfg["processed_dir"])
        if not processed_dir.is_absolute():
            processed_dir = repo_root / processed_dir

        if cfg.get("sp500_csv"):
            raw_dir = Path(cfg["raw_dir"])
            if not raw_dir.is_absolute():
                raw_dir = repo_root / raw_dir
            snapshot_path = raw_dir / cfg["sp500_csv"]
    else:
        print(
            "Warning: config.yaml not found, using default paths. "
            "Copy config.example.yaml to config.yaml to configure."
        )

    return processed_dir, snapshot_path


def load_sp500_ids(snapshot_path):
    """Load valid posting-system company IDs from a snapshot CSV."""
    snapshot = pd.read_csv(snapshot_path)
    if "COMPANY" not in snapshot.columns:
        raise ValueError("S&P 500 snapshot must contain a COMPANY column.")

    company_ids = pd.to_numeric(snapshot["COMPANY"], errors="coerce")
    company_ids = company_ids[company_ids.notna() & (company_ids != 0)].astype("int64")
    return set(company_ids)


def relabel_file(path, sp500_ids, dry_run=False):
    """Update SP500 in one parquet file and return audit metadata."""
    df = pd.read_parquet(path)
    if "COMPANY" not in df.columns:
        raise ValueError(f"{path} must contain a COMPANY column.")

    old_sp500 = df["SP500"].astype(bool) if "SP500" in df.columns else pd.Series(False, index=df.index)
    company_ids = pd.to_numeric(df["COMPANY"], errors="coerce")
    new_sp500 = company_ids.isin(sp500_ids) & company_ids.ne(0)

    df["SP500"] = new_sp500
    if not dry_run:
        df.to_parquet(path, index=False)

    return {
        "FILE": path.name,
        "ROWS": len(df),
        "OLD_SP500_POSTINGS": int(old_sp500.sum()),
        "NEW_SP500_POSTINGS": int(new_sp500.sum()),
        "CHANGED_ROWS": int((old_sp500 != new_sp500).sum()),
        "DRY_RUN": dry_run,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Refresh SP500 labels in existing processed parquet files."
    )
    parser.add_argument("--snapshot", type=Path, help="S&P 500 snapshot CSV keyed by COMPANY")
    parser.add_argument("--processed-dir", type=Path, help="Directory containing processed parquet files")
    parser.add_argument(
        "--files",
        nargs="+",
        default=list(DEFAULT_PARQUETS),
        help="Processed parquet filenames to relabel",
    )
    parser.add_argument("--dry-run", action="store_true", help="Report changes without writing parquet files")
    args = parser.parse_args()

    default_processed_dir, default_snapshot_path = load_default_paths()
    processed_dir = args.processed_dir or default_processed_dir
    snapshot_path = args.snapshot or default_snapshot_path

    sp500_ids = load_sp500_ids(snapshot_path)
    print(f"S&P 500 snapshot companies: {len(sp500_ids):,}")
    print(f"S&P 500 snapshot path: {snapshot_path}")

    audit_rows = []
    for filename in args.files:
        path = processed_dir / filename
        if not path.exists():
            print(f"Skipping missing file: {path}")
            continue
        audit = relabel_file(path, sp500_ids, dry_run=args.dry_run)
        audit_rows.append(audit)
        print(
            f"{filename}: {audit['OLD_SP500_POSTINGS']:,} -> "
            f"{audit['NEW_SP500_POSTINGS']:,} SP500 postings "
            f"({audit['CHANGED_ROWS']:,} rows changed)"
        )

    if not audit_rows:
        raise FileNotFoundError(f"No requested parquet files found in {processed_dir}")

    audit_df = pd.DataFrame(audit_rows)
    audit_path = processed_dir / "sp500_relabel_audit.csv"
    audit_df.to_csv(audit_path, index=False)
    print(f"S&P 500 relabel audit saved to {audit_path}")


if __name__ == "__main__":
    main()
