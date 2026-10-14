#!/usr/bin/env python3
"""Join compact full-data labels using the current MBB preparation rules.

Avoids holding posting body text or the complete skills file in memory. The
selected structured posts and joined labels must still fit in memory. Output
is separate from existing analysis data unless explicitly selected/overwritten.
"""

import argparse
import json
import os
from pathlib import Path
import sys
import tempfile

import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts import prepare_data
from scripts.full_data_pipeline import (
    BENEFIT_NAMES, POSITION_NAMES, add_structured_labels, atomic_json,
    data_paths, fingerprint, iter_batches, metadata_path,
)
from package_files.firm_size import FIRM_SIZE_CUTOFF

POST_COLUMNS = [
    "ID", "POSTED", "COMPANY", "COMPANY_NAME", "MIN_EDULEVELS_NAME",
    "MIN_YEARS_EXPERIENCE", "SALARY", "REMOTE_TYPE_NAME", "STATE_NAME",
    "NAICS_2022_6", "NAICS_2022_6_NAME", "ONET", "BENEFIT_SUBCATEGORY_NAME", "QUARTER",
]
POST_TYPES = {name: pa.string() for name in POST_COLUMNS}
POST_TYPES.update({"COMPANY": pa.int64(), "SALARY": pa.float64(),
                   "MIN_YEARS_EXPERIENCE": pa.float64(), "NAICS_2022_6": pa.int64(),
                   "QUARTER": pa.int64()})


def check_ids(table, name):
    if "ID" not in table.column_names:
        raise ValueError(f"{name} needs an ID column.")
    ids = table["ID"]
    if not pa.types.is_string(ids.type) and not pa.types.is_large_string(ids.type):
        raise ValueError(f"{name} posting IDs must be strings.")
    if ids.null_count or pc.any(pc.equal(pc.utf8_trim_whitespace(ids), "")).as_py():
        raise ValueError(f"{name} has missing or empty IDs.")
    if pc.count_distinct(ids).as_py() != table.num_rows:
        raise ValueError(f"{name} has duplicate IDs; joins must be one-to-one.")


def read_labels(path, columns, *, boolean_columns=(), position_columns=()):
    path = Path(path)
    schema = pq.ParquetFile(path).schema_arrow
    missing = set(columns) - set(schema.names)
    if missing:
        raise ValueError(f"{path.name} is missing required columns: {sorted(missing)}")
    for column in boolean_columns:
        if not pa.types.is_boolean(schema.field(column).type):
            raise ValueError(f"{path.name}: {column} must be boolean.")
    table = pq.read_table(path, columns=columns).combine_chunks()
    check_ids(table, path.name)
    for column in boolean_columns:
        if table[column].null_count:
            raise ValueError(f"{path.name}: {column} contains null labels.")
    for column in position_columns:
        try:
            values = pc.cast(table[column], pa.float64())
        except (pa.ArrowInvalid, pa.ArrowNotImplementedError) as error:
            raise ValueError(f"{path.name}: {column} must be numeric.") from error
        invalid = pc.or_(pc.less(values, 0), pc.greater(values, 100))
        if pc.any(pc.fill_null(invalid, False)).as_py():
            raise ValueError(f"{path.name}: {column} must be between 0 and 100.")
    sidecar = metadata_path(path)
    metadata = json.loads(sidecar.read_text()) if sidecar.exists() else {}
    if metadata and (not metadata.get("complete") or metadata.get("rows") != table.num_rows
                     or metadata.get("output") != fingerprint(path)):
        raise ValueError(f"{path.name}: metadata does not describe this completed output.")
    return table, metadata


def join_one_to_one(left, right):
    joined = left.join(right, keys="ID", join_type="left outer", use_threads=True)
    if joined.num_rows != left.num_rows:
        raise ValueError("Label join changed the posting row count.")
    return joined.combine_chunks()


def assemble_dataset(posts, benefits, remote, ai_counts, *, sp500=None, wham=None,
                     allow_missing_body=False, batch_rows=100_000, block_mb=64):
    sources = {name: Path(path) for name, path in {
        "posts": posts, "benefits": benefits, "remote": remote, "ai_counts": ai_counts,
    }.items()}
    original = {name: fingerprint(path) for name, path in sources.items()}
    for name, path in {"sp500": sp500, "wham": wham}.items():
        if path is not None and not Path(path).exists():
            raise FileNotFoundError(f"Configured {name} file not found: {path}")
        if path is not None:
            original[name] = fingerprint(path)

    reader = iter_batches(posts, POST_COLUMNS, batch_rows, block_mb, POST_TYPES)
    batches = list(reader)
    if not batches:
        raise ValueError("Posts input is empty.")
    table = pa.Table.from_batches(batches).combine_chunks()
    del batches
    check_ids(table, "posts")
    table = table.append_column("__ROW_ORDER", pa.array(range(table.num_rows), type=pa.int64()))
    benefits_table, benefit_meta = read_labels(
        benefits, ["ID"] + BENEFIT_NAMES + POSITION_NAMES,
        boolean_columns=BENEFIT_NAMES, position_columns=POSITION_NAMES,
    )
    remote_table, remote_meta = read_labels(
        remote, ["ID", "REMOTE_KW", "REMOTE_KW_POSITION"],
        boolean_columns=["REMOTE_KW"], position_columns=["REMOTE_KW_POSITION"],
    )
    if benefits_table.num_rows != remote_table.num_rows or not pc.all(
        pc.is_in(benefits_table["ID"], value_set=remote_table["ID"])
    ).as_py():
        raise ValueError("Benefit and remote labels must cover the same body IDs.")
    if benefit_meta and remote_meta and benefit_meta.get("input") != remote_meta.get("input"):
        raise ValueError("Benefit and remote labels come from different body inputs.")

    missing_body = int(pc.sum(pc.invert(pc.is_in(table["ID"], value_set=benefits_table["ID"]))).as_py() or 0)
    if missing_body and not allow_missing_body:
        raise ValueError(f"{missing_body:,} posts have no body labels; review coverage and use --allow-missing-body if expected.")
    ai_table, ai_meta = read_labels(ai_counts, ["ID", "AI_SKILL_COUNT"])
    if not pa.types.is_integer(ai_table["AI_SKILL_COUNT"].type) or ai_table["AI_SKILL_COUNT"].null_count:
        raise ValueError("AI_SKILL_COUNT must contain nonnull integers.")
    if pc.any(pc.less(ai_table["AI_SKILL_COUNT"], 0)).as_py():
        raise ValueError("AI_SKILL_COUNT cannot be negative.")

    table = join_one_to_one(table, benefits_table)
    table = join_one_to_one(table, remote_table)
    table = join_one_to_one(table, ai_table)
    del benefits_table, remote_table, ai_table
    frame = table.to_pandas()
    del table
    for name in BENEFIT_NAMES + ["REMOTE_KW"]:
        frame[name] = frame[name].astype("boolean").fillna(False).astype(bool)
    frame["AI_SKILL_COUNT"] = frame["AI_SKILL_COUNT"].fillna(0).astype("int64")
    frame["AI ROLE"] = frame["AI_SKILL_COUNT"] > 0

    # Reuse Matthew's current transforms; never classify using all-year counts.
    frame = prepare_data.add_year_column(frame)
    frame = prepare_data.add_industry_columns(frame)
    frame = prepare_data.add_firm_size(frame)
    with tempfile.TemporaryDirectory(prefix="sp500-assembly-") as temporary:
        frame = prepare_data.merge_sp500_snapshot(frame, sp500, Path(temporary) / "match_audit.csv")
    frame = prepare_data.create_experience_buckets(frame)
    frame = prepare_data.add_log_salary(frame)
    if wham is not None:
        wham_frame = pd.read_csv(wham)
        us = wham_frame.loc[wham_frame["country2"].eq("US"), ["id", "wfh_wham_prob", "wfh_wham"]]
        if us["id"].isna().any() or us["id"].duplicated().any():
            raise ValueError("WHAM US rows need unique, nonnull posting IDs.")
        frame = prepare_data.merge_wham_data(frame, wham_frame)
    else:
        print("WHAM was not supplied; WHAM fields are omitted, as in the original pipeline.")
    frame = prepare_data.add_group_names(frame)
    frame = add_structured_labels(frame)
    frame = frame.sort_values("__ROW_ORDER").drop(columns="__ROW_ORDER").reset_index(drop=True)
    for name, path in sources.items():
        if fingerprint(path) != original[name]:
            raise ValueError(f"{name} changed while building the dataset.")
    for name, path in {"sp500": sp500, "wham": wham}.items():
        if path is not None and fingerprint(path) != original[name]:
            raise ValueError(f"{name} changed while building the dataset.")
    audit = {
        "inputs": original, "rows": len(frame), "posts_without_body": missing_body,
        "ai_postings": int(frame["AI ROLE"].sum()), "firm_size_base_year": 2018,
        "firm_size_cutoff": FIRM_SIZE_CUTOFF,
        "remote_keyword_correction": bool(remote_meta.get("correct_remote_keywords", False)),
        "label_metadata_available": bool(benefit_meta and remote_meta and ai_meta),
        "wham_included": wham is not None,
    }
    return frame, audit


def build_dataset(output, *, overwrite=False, **inputs):
    output = Path(output)
    if output.exists() and not overwrite:
        raise FileExistsError(f"Output exists: {output}; use --overwrite explicitly.")
    for name in ["posts", "benefits", "remote", "ai_counts", "sp500", "wham"]:
        if inputs.get(name) is not None and output.resolve() == Path(inputs[name]).resolve():
            raise ValueError("Output cannot replace an input file.")
    frame, audit = assemble_dataset(**inputs)
    output.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=output.name + ".", suffix=".tmp", dir=output.parent)
    os.close(descriptor)
    try:
        frame.to_parquet(temporary, index=False, compression="gzip")
        os.replace(temporary, output)
        atomic_json(metadata_path(output), {**audit, "complete": True, "output": fingerprint(output)})
    finally:
        Path(temporary).unlink(missing_ok=True)
    print(f"Wrote {len(frame):,} rows to {output}; audit: {metadata_path(output)}")
    return audit


def main():
    paths = data_paths()
    processed = paths["processed_dir"]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--posts-csv", type=Path, default=paths["posts_csv"])
    parser.add_argument("--benefit-labels", type=Path, default=processed / "benefit_labels_body.parquet")
    parser.add_argument("--remote-labels", type=Path, default=processed / "remote_labels_body.parquet")
    parser.add_argument("--ai-counts", type=Path, default=processed / "ai_skill_counts_by_id.parquet")
    parser.add_argument("--sp500-csv", type=Path, default=paths["sp500_csv"])
    parser.add_argument("--wham-csv", type=Path, default=paths["wham_csv"] if paths["wham_csv"].exists() else None)
    parser.add_argument("--output", type=Path, default=processed / "full" / "labeled_v2.parquet")
    parser.add_argument("--batch-rows", type=int, default=100_000)
    parser.add_argument("--csv-block-mb", type=int, default=64)
    parser.add_argument("--allow-missing-body", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    build_dataset(
        args.output, posts=args.posts_csv, benefits=args.benefit_labels,
        remote=args.remote_labels, ai_counts=args.ai_counts,
        sp500=args.sp500_csv, wham=args.wham_csv, allow_missing_body=args.allow_missing_body,
        batch_rows=args.batch_rows, block_mb=args.csv_block_mb, overwrite=args.overwrite,
    )


if __name__ == "__main__":
    main()
