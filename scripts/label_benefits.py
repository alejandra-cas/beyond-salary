import argparse
import pandas as pd
import pyarrow as pa
import pyarrow.csv as pacsv
import pyarrow.parquet as pq
import re
import json
import time
import yaml

from pathlib import Path


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
        "output_path": processed_dir / "benefit_labels_body.parquet",
    }


paths = load_processed_paths()
input_path = paths["input_path"]
output_path = paths["output_path"]

# --- Keyword definitions ---

edu_assistance = (
    "education assistance",
    "tuition reimbursement",
    "tuition assistance",
    "education reimbursement",
)
leave = (
    "paid time off",
    "PTO",
    "extra vacation",
    "flexible vacation",
    "mental health day",
    "generous time-off",
    "generous time off",
    "paid vacation",
    "paid holidays",
    "holiday pay",
    "vacation days",
    "company holiday",
    "sick leave",
    "vacation time",
    "paid days off",
    "paid flexible holidays",
)
wellbeing = (
    "wellness stipend",
    "mental health support",
    "gym membership",
    "wellness program",
    "well-being stipend",
    "wellness programs",
    "mental health benefits",
    "employee well-being",
    "mental wellness support",
    "wellness and global well-being",
    "wellness stipend",
)
health_wellbeing = (
    "wellness stipend",
    "health benefits",
    "mental health support",
    "gym membership",
    "wellness program",
    "well-being stipend",
    "wellness programs",
    "mental health benefits",
    "employee well-being",
    "health care benefits",
)
parental_leave = (
    "parental leave",
    "family support",
    "flexible maternity leave",
    "paternity leave",
    "maternity leave",
    "paid family leave",
    "paid caregiver/parental",
    "paid parental",
    "paid new parent leave",
    "paid bonding leave",
    "parental bonding leave",
)
culture = (
    "diversity and inclusion",
    "team culture",
    "creative freedom",
    "values-driven",
    "inclusive culture",
    "diverse team",
    "inclusive environment",
    "value diversity",
    "diversity, equity",
    "culture of diversity",
    "diversity, inclusion",
    "inclusion and diversity",
    "commitment to diversity",
    "diversity, inclusion",
    "diversity, equity and inclusion",
    "diversity is respected",
    "inclusive diversity",
    "workforce diversity",
    "embracing diversity",
    "value diversity",
    "values diversity",
    "committed to diversity",
    "equity and diversity",
    "promoting diversity",
    "celebrate diversity",
    "encourage diversity",
    "support diversity",
    "diversity in the workplace",
    "equity, inclusion",
)

# --- Exclusion lists ---

career_dev_to_exclude = [
    "youth leadership development",
    "provide mentorship",
    "professional development expertise",
    "planning professional development",
    "forecasting growth opportunities",
    "teen leadership development",
    "develop training programs",
    "professional development experience",
    "training programs as required",
    "providing mentorship",
    "providing and encouraging mentorship",
    "implementing training programs",
    "provide leadership and mentorship",
    "provide mentorship",
    "identify growth opportunities",
    "manage the Symbotic Tuition Reimbursement",
]
tuition_to_exclude = ["manage the Symbotic Tuition Reimbursement"]
wellbeing_to_exclude = [
    "health benefits companies",
    "wellness program coordinator",
    "familiarity with mental health support",
    "health benefits administration",
]
recognition_to_exclude = ["managing employee recognition"]
family_to_exclude = [
    "accredited childcare program",
    "maui family support services",
    "teaching assistant - childcare",
    "experience in childcare",
    "childcare state licensing",
    "experience with children",
]
culture_to_exclude = ["maintaining a collaborative environment"]


def _keyword_regex(keywords):
    """Case-sensitive regex over lowercased keywords (matching lowercased text).

    Matching lowercased text with a case-sensitive pattern is ~2.5x faster than
    re.IGNORECASE over the original text and produces identical labels.
    """
    return re.compile(r"\b(?:" + "|".join(map(re.escape, [k.lower() for k in keywords])) + r")\b")


def check_benefit_position(body_series, keywords, exclusions=None, assume_lowered=False):
    """Return the prominence score (0–100) of the first keyword match in each posting.

    Score = (1 - match.start() / len(text)) * 100
      - 100: match at the very start of the text (most prominent)
      - 0:   match at the very end of the text (least prominent)
      - NaN: no match (or empty body)

    This mirrors the boolean logic of check_benefits() — exclusion-overlapping
    matches are skipped before recording position.

    Pass assume_lowered=True when body_series is already fillna("")+lowercased.
    """
    include_re = _keyword_regex(keywords)
    exclude_re = _keyword_regex(exclusions) if exclusions else None

    body_filled = body_series if assume_lowered else body_series.fillna("").str.lower()
    positions = pd.array([float("nan")] * len(body_series), dtype="Float64")

    for i, text in enumerate(body_filled):
        if not text:
            continue
        text_len = len(text)
        for m in include_re.finditer(text):
            # Skip if this match is covered by an exclusion
            if exclude_re:
                skip = False
                for exc in exclude_re.finditer(text):
                    if exc.start() <= m.start() < exc.end() or exc.start() < m.end() <= exc.end():
                        skip = True
                        break
                if skip:
                    continue
            # First valid match — record prominence and stop
            positions[i] = (1 - m.start() / text_len) * 100
            break

    return positions


def check_benefits(body_series, keywords, exclusions=None, assume_lowered=False):
    """Vectorized benefit labeling using compiled regex.

    Uses pandas str.contains() for the bulk of rows, then falls back to
    row-level overlap checking only for the small subset that matched
    both inclusion and exclusion patterns.

    Pass assume_lowered=True when body_series is already fillna("")+lowercased.

    Returns a boolean numpy array.
    """
    include_re = _keyword_regex(keywords)

    # Vectorized inclusion check
    body_filled = body_series if assume_lowered else body_series.fillna("").str.lower()
    included = body_filled.str.contains(include_re.pattern, regex=True, na=False)

    if exclusions is None:
        return included.to_numpy(dtype=bool)

    # Only check exclusions on the subset that matched inclusion
    exclude_re = _keyword_regex(exclusions)
    excluded = body_filled.str.contains(exclude_re.pattern, regex=True, na=False)

    # Rows with inclusion but no exclusion are True
    # Rows with both need overlap check
    needs_overlap_check = included & excluded

    if not needs_overlap_check.any():
        return included.to_numpy(dtype=bool)

    # Row-level overlap check only for the small subset with both matches
    result = included.to_numpy(dtype=bool, copy=True)
    check_idx = needs_overlap_check[needs_overlap_check].index

    for idx in check_idx:
        text = body_filled.iloc[idx]
        include_matches = list(include_re.finditer(text))
        exclude_matches = list(exclude_re.finditer(text))

        # Check if any exclusion overlaps with an inclusion match
        has_overlap = False
        for exc in exclude_matches:
            for inc in include_matches:
                if inc.start() <= exc.start() < inc.end() or inc.start() <= exc.end() <= inc.end():
                    has_overlap = True
                    break
            if has_overlap:
                break

        if has_overlap:
            result[idx] = False

    return result


KEYWORD_LABELS = [
    ("EDU_ASSISTANCE", edu_assistance, tuition_to_exclude),
    ("PAID LEAVE", leave, None),
    ("WELLBEING", wellbeing, wellbeing_to_exclude),
    ("HEALTH_WELLBEING", health_wellbeing, wellbeing_to_exclude),
    ("PARENTAL_LEAVE", parental_leave, None),
    ("CULTURE", culture, None),
]
POSITION_LABELS = [
    ("EDU_ASSISTANCE_POSITION", edu_assistance, tuition_to_exclude),
    ("PAID_LEAVE_POSITION", leave, None),
    ("HEALTH_WELLBEING_POSITION", health_wellbeing, wellbeing_to_exclude),
    ("PARENTAL_LEAVE_POSITION", parental_leave, None),
    ("CULTURE_POSITION", culture, None),
]
STRUCTURED_SUBCATEGORY_LABELS = [
    ("S_FLEX_WORK", ["Flexible Work Schedules"]),
    ("S_PROF_DEV", ["Professional Development", "Leadership Development", "Mentorships"]),
    ("S_HEALTH_WELLNESS", [
        "Health and Wellness Programs", "Mental Health", "Bereavement/Mental Health Leave",
        "Health and Wellness Stipends", "Health and Wellness Applications",
    ]),
]
LABEL_COLUMNS = (
    [name for name, _, _ in KEYWORD_LABELS]
    + [name for name, _ in STRUCTURED_SUBCATEGORY_LABELS]
    + ["S_REMOTE"]
    + [name for name, _, _ in POSITION_LABELS]
)


def parse_benefit_list(val):
    if isinstance(val, str) and val.startswith("["):
        try:
            return json.loads(val)
        except (json.JSONDecodeError, ValueError):
            return []
    return []


def has_any(series, terms):
    terms_lower = {t.lower() for t in terms}
    return series.apply(lambda lst: any(v.lower() in terms_lower for v in lst if isinstance(v, str)))


def label_batch(data, skip_positions=False, skip_structured=False):
    """Add all keyword, structured, and position label columns to one batch.

    Structured labels are only computed when their source columns are present
    (e.g. the full BODY CSV carries just ID + BODY).
    """
    body = data["BODY"].fillna("").str.lower()

    for name, keywords, exclusions in KEYWORD_LABELS:
        data[name] = check_benefits(body, keywords, exclusions=exclusions, assume_lowered=True)

    if not skip_structured and "BENEFIT_SUBCATEGORY_NAME" in data.columns:
        parsed = data["BENEFIT_SUBCATEGORY_NAME"].apply(parse_benefit_list)
        for name, terms in STRUCTURED_SUBCATEGORY_LABELS:
            data[name] = has_any(parsed, terms)
    if not skip_structured and "REMOTE_TYPE_NAME" in data.columns:
        data["S_REMOTE"] = data["REMOTE_TYPE_NAME"].isin(["Remote", "Hybrid Remote"])

    if not skip_positions:
        for name, keywords, exclusions in POSITION_LABELS:
            data[name] = check_benefit_position(body, keywords, exclusions=exclusions, assume_lowered=True)

    return data


def iter_input_batches(path, parquet_batch_rows, csv_block_mb):
    """Yield pandas DataFrames from a parquet or CSV input without loading it all.

    CSV reads use a bounded block size and tolerate embedded newlines in BODY.
    """
    if path.suffix.lower() == ".csv":
        reader = pacsv.open_csv(
            path,
            read_options=pacsv.ReadOptions(use_threads=True, block_size=csv_block_mb << 20),
            parse_options=pacsv.ParseOptions(newlines_in_values=True),
        )
        for batch in reader:
            yield batch.to_pandas()
    else:
        parquet_file = pq.ParquetFile(path)
        for batch in parquet_file.iter_batches(batch_size=parquet_batch_rows):
            yield batch.to_pandas()


def process_batch(data, part_idx, parts_dir, labels_only, skip_positions, skip_structured):
    """Label one batch and write its checkpoint part file. Returns run stats.

    Module-level so ProcessPoolExecutor workers can pickle it.
    """
    parts_dir = Path(parts_dir)
    stats = {
        "rows": len(data),
        "empty_body": int(data["BODY"].isna().sum() + data["BODY"].str.strip().eq("").sum()),
        "true_counts": {},
        "position_sums": {},
        "position_counts": {},
        "has_structured": "BENEFIT_SUBCATEGORY_NAME" in data.columns,
    }

    data = label_batch(data, skip_positions=skip_positions, skip_structured=skip_structured)

    for col in LABEL_COLUMNS:
        if col not in data.columns:
            continue
        if col.endswith("_POSITION"):
            valid = data[col].dropna()
            stats["position_sums"][col] = float(valid.sum())
            stats["position_counts"][col] = len(valid)
        else:
            stats["true_counts"][col] = int(data[col].sum())

    if labels_only:
        keep = ["ID"] + [c for c in LABEL_COLUMNS if c in data.columns]
        data = data[keep]

    # Atomic checkpoint write: tmp file then rename, so a crash mid-write
    # never leaves a truncated part behind.
    table = pa.Table.from_pandas(data, preserve_index=False)
    part_path = parts_dir / f"part-{part_idx:05d}.parquet"
    tmp_path = part_path.with_suffix(".tmp")
    pq.write_table(table, tmp_path, compression="gzip")
    tmp_path.rename(part_path)
    return stats


def main():
    parser = argparse.ArgumentParser(
        description="Stream-label benefit mentions; input is never fully loaded into memory."
    )
    parser.add_argument("--input", type=Path, default=input_path,
                        help="Input parquet or CSV (default: data.body_csv from config.yaml)")
    parser.add_argument("--output", type=Path, default=output_path,
                        help="Output parquet (default: processed_dir/benefit_labels_body.parquet)")
    parser.add_argument("--labels-only", action="store_true",
                        help="Write only ID + label columns (recommended for the full BODY CSV)")
    parser.add_argument("--parquet-batch-rows", type=int, default=100_000,
                        help="Rows per batch for parquet inputs (default: 100000)")
    parser.add_argument("--csv-block-mb", type=int, default=128,
                        help="Read-buffer size in MB for CSV inputs (default: 128)")
    parser.add_argument("--skip-positions", action="store_true",
                        help="Skip the slow per-row prominence-position pass")
    parser.add_argument("--skip-structured", action="store_true",
                        help="Skip the structured BENEFIT_SUBCATEGORY_NAME/REMOTE_TYPE_NAME labels")
    parser.add_argument("--workers", type=int, default=6,
                        help="Parallel labeling processes (default: 6, one per M2 performance core)")
    args = parser.parse_args()

    start = time.time()
    input_resolved = args.input.expanduser().resolve()
    size_gb = input_resolved.stat().st_size / 1e9
    print(f"Input body file: {input_resolved} ({size_gb:.1f} GB)")
    print(f"Writing to {args.output}" + (" (labels only)" if args.labels_only else ""))

    # Checkpointing: batch i always becomes part-i (batching is deterministic
    # for the same input/settings), so a rerun relabels only the missing parts.
    parts_dir = args.output.with_name(args.output.name + ".parts")
    parts_dir.mkdir(parents=True, exist_ok=True)
    done_indices = {int(p.stem.split("-")[1]) for p in parts_dir.glob("part-*.parquet")}
    if done_indices:
        print(f"Resuming: {len(done_indices)} checkpoint parts found; "
              "relabeling only the missing batches (use the same batch-size flags "
              f"as the original run, or delete {parts_dir} to start fresh).")
        print("Note: end-of-run label counts will cover the resumed portion only.")

    total_rows = 0
    empty_body_count = 0
    true_counts = {}
    position_sums = {}
    position_counts = {}
    structured_seen = False
    batches_done = 0

    def merge_stats(stats):
        nonlocal total_rows, empty_body_count, structured_seen, batches_done
        total_rows += stats["rows"]
        empty_body_count += stats["empty_body"]
        structured_seen = structured_seen or stats["has_structured"]
        for col, v in stats["true_counts"].items():
            true_counts[col] = true_counts.get(col, 0) + v
        for col, v in stats["position_sums"].items():
            position_sums[col] = position_sums.get(col, 0.0) + v
        for col, v in stats["position_counts"].items():
            position_counts[col] = position_counts.get(col, 0) + v
        batches_done += 1
        print(f"  {total_rows:,} new rows labeled ({time.time() - start:.0f}s)", flush=True)

    batch_args = (parts_dir, args.labels_only, args.skip_positions, args.skip_structured)
    batches = (
        (i, data)
        for i, data in enumerate(iter_input_batches(input_resolved, args.parquet_batch_rows, args.csv_block_mb))
        if i not in done_indices
    )

    if args.workers <= 1:
        for i, data in batches:
            merge_stats(process_batch(data, i, *batch_args))
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
                in_flight.add(pool.submit(process_batch, data, i, *batch_args))
            for fut in cf.as_completed(in_flight):
                merge_stats(fut.result())

    if not args.skip_structured and total_rows and not structured_seen:
        print("Note: no BENEFIT_SUBCATEGORY_NAME/REMOTE_TYPE_NAME columns; "
              "structured labels were skipped (join them from the posts data separately).")

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
    for col, count in true_counts.items():
        print(f"  {col}: {count:,} rows")
    for col in position_counts:
        if position_counts[col]:
            print(f"  {col}: {position_counts[col]:,} matches, "
                  f"mean prominence {position_sums[col] / position_counts[col]:.1f}")


if __name__ == "__main__":
    main()
