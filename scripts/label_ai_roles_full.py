#!/usr/bin/env python3
"""Count full-data AI skills without loading the skills CSV into memory."""

import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.full_data_pipeline import count_ai_skills, data_paths


def main():
    paths = data_paths()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--skills-csv", type=Path, default=paths["skills_csv"])
    parser.add_argument("--output", type=Path, default=paths["processed_dir"] / "ai_skill_counts_by_id.parquet")
    parser.add_argument("--batch-rows", type=int, default=100_000)
    parser.add_argument("--csv-block-mb", type=int, default=64)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    count_ai_skills(args.skills_csv, args.output, batch_rows=args.batch_rows,
                    block_mb=args.csv_block_mb, overwrite=args.overwrite)


if __name__ == "__main__":
    main()
