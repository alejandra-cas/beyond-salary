#!/usr/bin/env python3
"""Label AI roles for the full-sample postings without loading the 16GB skills
CSV into memory.

Mirrors prepare_data.py:classify_ai_roles -- a posting is an "AI ROLE" when it
has at least one skill in the "Artificial Intelligence and Machine Learning
(AI/ML)" subcategory -- but streams both large CSVs so it runs on the full data.

Pass 1: stream the skills CSV, keep only AI/ML rows, count them per posting ID.
Pass 2: stream the posts CSV, attach AI_SKILL_COUNT / AI ROLE, write a parquet
        keyed by posting ID for downstream merges.

Output: data/processed/ai_roles_full.parquet  (columns: ID, AI_SKILL_COUNT, AI_ROLE)
"""
from __future__ import annotations

from collections import Counter
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.csv as pacsv
import pyarrow.parquet as pq

SKILLS_CSV = Path("data/OII_US_10M_SKILLS_MAY26.csv")
POSTS_CSV = Path("data/OII_US_10M_POSTS_MAY26.csv")
OUT_PARQUET = Path("data/processed/ai_roles_full.parquet")
# Cache of pass-1 results so a re-run skips the multi-minute 16GB skills scan.
COUNTS_CACHE = Path("data/processed/ai_skill_counts_by_id.parquet")
AI_SUBCATEGORY = "Artificial Intelligence and Machine Learning (AI/ML)"

# Embedded newlines occur in free-text skill/title fields; block_size caps the
# pyarrow read buffer so memory stays bounded on these multi-GB files.
READ = pacsv.ReadOptions(use_threads=True, block_size=128 << 20)
PARSE = pacsv.ParseOptions(newlines_in_values=True)


def count_ai_skills_per_id() -> Counter:
    """Pass 1: stream skills, count AI/ML skills per posting ID."""
    reader = pacsv.open_csv(
        SKILLS_CSV,
        read_options=READ,
        parse_options=PARSE,
        convert_options=pacsv.ConvertOptions(
            include_columns=["ID", "SKILL_SUBCATEGORY_NAME"]
        ),
    )
    counts: Counter = Counter()
    rows_seen = 0
    for batch in reader:
        df = batch.to_pandas()
        rows_seen += len(df)
        ai = df[df["SKILL_SUBCATEGORY_NAME"] == AI_SUBCATEGORY]
        if len(ai):
            for posting_id, n in ai["ID"].value_counts().items():
                counts[posting_id] += int(n)
        if rows_seen % 50_000_000 < len(df):
            print(f"  skills rows scanned: {rows_seen:,} | AI postings so far: {len(counts):,}")
    print(f"Pass 1 done: {rows_seen:,} skill rows, {len(counts):,} postings with >=1 AI/ML skill")
    return counts


def load_or_build_counts() -> dict:
    """Return AI-skill-count-per-ID, using a cached parquet when available."""
    if COUNTS_CACHE.exists():
        print(f"Loading cached AI skill counts from {COUNTS_CACHE}")
        cache = pd.read_parquet(COUNTS_CACHE)
        return dict(zip(cache["ID"].tolist(), cache["AI_SKILL_COUNT"].tolist()))
    counts = count_ai_skills_per_id()
    COUNTS_CACHE.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        {"ID": list(counts.keys()), "AI_SKILL_COUNT": list(counts.values())}
    ).to_parquet(COUNTS_CACHE, index=False)
    print(f"Cached AI skill counts to {COUNTS_CACHE}")
    return counts


def write_post_labels(ai_counts: Counter) -> dict:
    """Pass 2: stream posts, attach AI labels, write parquet, return summary."""
    OUT_PARQUET.parent.mkdir(parents=True, exist_ok=True)
    # Posting IDs are 40-char SHA-1 hex strings, not numbers; key on the string.
    schema = pa.schema([
        ("ID", pa.string()),
        ("AI_SKILL_COUNT", pa.int32()),
        ("AI_ROLE", pa.bool_()),
    ])
    writer = pq.ParquetWriter(OUT_PARQUET, schema)
    total = ai_total = dropped = 0
    reader = pacsv.open_csv(
        POSTS_CSV,
        read_options=READ,
        parse_options=PARSE,
        convert_options=pacsv.ConvertOptions(include_columns=["ID"]),
    )
    try:
        for batch in reader:
            ids = batch.column(0).to_pandas()
            total += len(ids)
            # Drop rows with a missing/blank posting ID: they cannot be keyed or
            # merged downstream.
            valid = ids.notna() & (ids.astype(str).str.len() > 0)
            dropped += int((~valid).sum())
            ids = ids[valid].astype(str).reset_index(drop=True)
            cnt = ids.map(lambda x: ai_counts.get(x, 0)).astype("int32")
            role = cnt > 0
            ai_total += int(role.sum())
            tbl = pa.table(
                {
                    "ID": pa.array(ids, type=pa.string()),
                    "AI_SKILL_COUNT": pa.array(cnt, type=pa.int32()),
                    "AI_ROLE": pa.array(role.to_numpy(), type=pa.bool_()),
                },
                schema=schema,
            )
            writer.write_table(tbl)
    finally:
        writer.close()
    return {"total_postings": total, "ai_postings": ai_total, "dropped_no_id": dropped}


def main() -> None:
    ai_counts = load_or_build_counts()
    summary = write_post_labels(ai_counts)
    total = summary["total_postings"]
    ai = summary["ai_postings"]
    dropped = summary["dropped_no_id"]
    written = total - dropped
    print(f"\nWrote {OUT_PARQUET}")
    print(f"Total postings   : {total:,}")
    print(f"Dropped (no ID)  : {dropped:,}")
    print(f"Labeled postings : {written:,}")
    print(f"AI ROLE = True   : {ai:,} ({ai/written:.2%} of labeled)")


if __name__ == "__main__":
    main()
