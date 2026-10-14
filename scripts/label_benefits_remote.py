"""Stream-label remote-work keyword mentions (REMOTE_KW) from body text.

Reads the body file batch-by-batch (never fully into memory), writes checkpoint
part files (safe to kill and rerun), and merges to a single ID + labels parquet.
Shares its regex/labeling machinery with label_benefits.py.
"""
import argparse
import sys
import time
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import yaml

sys.path.insert(0, str(Path(__file__).parent))
from label_benefits import check_benefit_position, check_benefits, iter_input_batches


def load_processed_paths():
    """Load input/output paths from config.yaml.

    The default input is data.body_csv, which may be an absolute path
    (e.g. a full-size CSV in ~/Downloads); relative values resolve against
    data.raw_dir. Falls back to repo-local defaults if config.yaml is missing.
    """
    repo_root = Path(__file__).parent.parent
    config_path = repo_root / "config.yaml"

    if config_path.exists():
        with open(config_path) as f:
            cfg = yaml.safe_load(f)["data"]
        processed_dir = Path(cfg["processed_dir"])
        if not processed_dir.is_absolute():
            processed_dir = repo_root / processed_dir
        raw_dir = Path(cfg.get("raw_dir", "data"))
        if not raw_dir.is_absolute():
            raw_dir = repo_root / raw_dir
        body_csv = Path(cfg.get("body_csv", "OII_US_10M_BODY_MAY26_SUBSAMPLE.csv")).expanduser()
        if not body_csv.is_absolute():
            body_csv = raw_dir / body_csv
    else:
        print(
            "Warning: config.yaml not found, using default processed paths. "
            "Copy config.example.yaml to config.yaml to configure."
        )
        processed_dir = repo_root / "data" / "processed"
        body_csv = repo_root / "data" / "OII_US_10M_BODY_MAY26_SUBSAMPLE.csv"

    return {
        "input_path": body_csv,
        "output_path": processed_dir / "remote_labels_body.parquet",
    }


paths = load_processed_paths()
input_path = paths["input_path"]
output_path = paths["output_path"]

remote_keywords = [
    "fully remote", "100% remote", "work from home", "remote role", "work remotely",
    "remote eligible", "open to remote", "remote work environment", "remote work policy",
    "remote and onsite", "virtual role", "remote first", "home office", "telecommute",
    "distributed team", "remote-first", "remote-friendly", "remote options", "virtual work", "work from home",
    "remote position", "wfh", "Remote - us", "telework", "home office", "(remote)", "remote work flexibility", "remote work: hybrid",
    "remote: yes", "location: remote", "remote usa", "remote flexibility", "(remote", "- remote", "remote-flexibility",
    "remote,", "\nremote\n", "\n remote \n", "remotely", "us-remote", "remote work eligible", "remote - united states",
    "remote - work at home", "can be remote", "remote within the us", "remote in north america", "remote available", ", remote",
    "remote full-time", "us remote", "#li-remote", "open to remote",
    "hybrid work", "split time between office and remote", "office and remote options",
    "partially remote", "hybrid position", "3 days remote", "hybrid workplace",
    "some remote days", "in-office and remote", "hybrid onsite", "remote or in-office", "work-from-home days", "mix of working in the office and from home", "#li-hybrid", "hybrid remote"
]

remote_to_exclude = [
    "not considering remote", "in-office only",
    "remote monitoring", "remote sensing", "remote access systems", "remote control",
    "remote diagnostics", "remote delivery", "#li-onsite", "onsite job", "work from home not available", "telework:no",
    "remote: no", "100% on-site", "work at home option: No",
    "remotely: no", "remote: n", "remotely: n", "telework: no", "remote: * no", "remotely: * no",
    "remotely piloted", "data remotely", "remote type on-site", "interviewed remotely", "work remotely: * no",
    "remote desktop", "must be able to work on-site", "not applicable for 100% remote", "remote testing", "remotely upgrading", "remote usability",
    "remote site", "remote machine", "remote iot", "no remote", "work remotely no", "remotely:no", "remotely? n", "remotely no", "remote areas",
    "remote access", "remote position? no", "remotely tucked away", "supporting remote", "remotely sensed"
]


def clean_body(series):
    """Normalize body text the same way the original pipeline did before matching."""
    return (
        series.fillna("")
        .str.strip()
        .str.replace(r"\s+", " ", regex=True)
        .str.lower()
    )


def process_batch(data, part_idx, parts_dir, skip_positions):
    """Label one batch and write its checkpoint part file. Returns run stats."""
    parts_dir = Path(parts_dir)
    stats = {
        "rows": len(data),
        "empty_body": int(data["BODY"].isna().sum() + data["BODY"].str.strip().eq("").sum()),
    }

    body = clean_body(data["BODY"])
    out = pd.DataFrame({"ID": data["ID"]})
    out["REMOTE_KW"] = check_benefits(body, remote_keywords, remote_to_exclude, assume_lowered=True)
    stats["remote_kw"] = int(out["REMOTE_KW"].sum())

    if not skip_positions:
        out["REMOTE_KW_POSITION"] = check_benefit_position(
            body, remote_keywords, remote_to_exclude, assume_lowered=True
        )
        valid = out["REMOTE_KW_POSITION"].dropna()
        stats["position_sum"] = float(valid.sum())
        stats["position_count"] = len(valid)

    # Atomic checkpoint write: tmp file then rename.
    table = pa.Table.from_pandas(out, preserve_index=False)
    part_path = parts_dir / f"part-{part_idx:05d}.parquet"
    tmp_path = part_path.with_suffix(".tmp")
    pq.write_table(table, tmp_path, compression="gzip")
    tmp_path.rename(part_path)
    return stats


def main():
    parser = argparse.ArgumentParser(
        description="Stream-label REMOTE_KW; input is never fully loaded into memory."
    )
    parser.add_argument("--input", type=Path, default=input_path,
                        help="Input parquet or CSV with ID+BODY (default: data.body_csv from config.yaml)")
    parser.add_argument("--output", type=Path, default=output_path,
                        help="Output parquet (default: processed_dir/remote_labels_body.parquet)")
    parser.add_argument("--parquet-batch-rows", type=int, default=100_000,
                        help="Rows per batch for parquet inputs (default: 100000)")
    parser.add_argument("--csv-block-mb", type=int, default=128,
                        help="Read-buffer size in MB for CSV inputs (default: 128)")
    parser.add_argument("--skip-positions", action="store_true",
                        help="Skip the slow per-row prominence-position pass")
    parser.add_argument("--workers", type=int, default=6,
                        help="Parallel labeling processes (default: 6, one per M2 performance core)")
    args = parser.parse_args()

    start = time.time()
    input_resolved = args.input.expanduser().resolve()
    size_gb = input_resolved.stat().st_size / 1e9
    print(f"Input body file: {input_resolved} ({size_gb:.1f} GB)")
    print(f"Writing to {args.output} (ID + REMOTE_KW"
          + ("" if args.skip_positions else " + REMOTE_KW_POSITION") + ")")

    # Checkpointing: batch i always becomes part-i (batching is deterministic
    # for the same input/settings), so a rerun relabels only the missing parts.
    parts_dir = args.output.with_name(args.output.name + ".parts")
    parts_dir.mkdir(parents=True, exist_ok=True)
    done_indices = {int(p.stem.split("-")[1]) for p in parts_dir.glob("part-*.parquet")}
    if done_indices:
        print(f"Resuming: {len(done_indices)} checkpoint parts found; "
              "relabeling only the missing batches (use the same batch-size flags "
              f"as the original run, or delete {parts_dir} to start fresh).")

    total_rows = 0
    empty_body_count = 0
    remote_count = 0
    position_sum = 0.0
    position_count = 0

    def merge_stats(stats):
        nonlocal total_rows, empty_body_count, remote_count, position_sum, position_count
        total_rows += stats["rows"]
        empty_body_count += stats["empty_body"]
        remote_count += stats["remote_kw"]
        position_sum += stats.get("position_sum", 0.0)
        position_count += stats.get("position_count", 0)
        print(f"  {total_rows:,} new rows labeled ({time.time() - start:.0f}s)", flush=True)

    batches = (
        (i, data)
        for i, data in enumerate(iter_input_batches(input_resolved, args.parquet_batch_rows, args.csv_block_mb))
        if i not in done_indices
    )

    if args.workers <= 1:
        for i, data in batches:
            merge_stats(process_batch(data, i, parts_dir, args.skip_positions))
    else:
        import concurrent.futures as cf
        # Bound in-flight batches so RAM stays capped at ~(workers+2) blocks.
        with cf.ProcessPoolExecutor(max_workers=args.workers) as pool:
            in_flight = set()
            for i, data in batches:
                while len(in_flight) >= args.workers + 2:
                    done, in_flight = cf.wait(in_flight, return_when=cf.FIRST_COMPLETED)
                    for fut in done:
                        merge_stats(fut.result())
                in_flight.add(pool.submit(process_batch, data, i, parts_dir, args.skip_positions))
            for fut in cf.as_completed(in_flight):
                merge_stats(fut.result())

    # Merge parts into the single output parquet, streaming one part at a time.
    print("Merging checkpoint parts into final output...")
    all_parts = sorted(parts_dir.glob("part-*.parquet"))
    indices = [int(p.stem.split("-")[1]) for p in all_parts]
    if indices != list(range(len(all_parts))):
        raise RuntimeError(f"Checkpoint parts in {parts_dir} are not contiguous; "
                           "rerun to fill the gaps before merging.")
    merged_rows = 0
    writer = None
    try:
        for p in all_parts:
            table = pq.read_table(p)
            if writer is None:
                writer = pq.ParquetWriter(args.output, table.schema, compression="gzip")
            else:
                table = table.cast(writer.schema)
            writer.write_table(table)
            merged_rows += table.num_rows
    finally:
        if writer is not None:
            writer.close()
    for p in all_parts:
        p.unlink()
    parts_dir.rmdir()

    print(f"\nLabeled {total_rows:,} new rows ({merged_rows:,} total in output) "
          f"in {time.time() - start:.1f}s")
    print(f"Empty body text (this run): {empty_body_count:,}")
    print(f"  REMOTE_KW: {remote_count:,} rows")
    if position_count:
        print(f"  REMOTE_KW_POSITION: {position_count:,} matches, "
              f"mean prominence {position_sum / position_count:.1f}")


if __name__ == "__main__":
    main()
