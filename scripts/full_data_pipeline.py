"""Shared streaming I/O for the optional full-data preparation path.

Keyword matching deliberately uses the existing labelers, including their
case and exclusion rules. Only the optional remote-keyword correction changes
labels. Existing no-argument preparation commands remain available.
"""

import argparse
from collections import Counter
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
import hashlib
import json
import os
from pathlib import Path
import tempfile

import pandas as pd
import pyarrow as pa
import pyarrow.csv as pacsv
import pyarrow.parquet as pq
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
BENEFIT_NAMES = [
    "EDU_ASSISTANCE", "PAID LEAVE", "WELLBEING", "HEALTH_WELLBEING",
    "PARENTAL_LEAVE", "CULTURE",
]
POSITION_NAMES = [
    "EDU_ASSISTANCE_POSITION", "PAID_LEAVE_POSITION",
    "HEALTH_WELLBEING_POSITION", "PARENTAL_LEAVE_POSITION", "CULTURE_POSITION",
]
STRUCTURED_TERMS = {
    "S_FLEX_WORK": ["Flexible Work Schedules"],
    "S_PROF_DEV": ["Professional Development", "Leadership Development", "Mentorships"],
    "S_HEALTH_WELLNESS": [
        "Health and Wellness Programs", "Mental Health", "Bereavement/Mental Health Leave",
        "Health and Wellness Stipends", "Health and Wellness Applications",
    ],
}


def data_paths():
    config_path = REPO_ROOT / "config.yaml"
    config = yaml.safe_load(config_path.read_text())["data"] if config_path.exists() else {}

    def resolve(value, root):
        path = Path(value).expanduser()
        return path if path.is_absolute() else root / path

    raw = resolve(config.get("raw_dir", "data"), REPO_ROOT)
    processed = resolve(config.get("processed_dir", "data/processed"), REPO_ROOT)
    paths = {"processed_dir": processed}
    for key, default in {
        "posts_csv": "OII_US_10M_POSTS_MAY26_SUBSAMPLE.csv",
        "skills_csv": "OII_US_10M_SKILLS_MAY26_SUBSAMPLE.csv",
        "body_csv": "OII_US_10M_BODY_MAY26_SUBSAMPLE.csv",
        "wham_csv": "ID_CNTRY_ALL_WHAM.csv",
    }.items():
        paths[key] = resolve(config.get(key, default), raw)
    paths["sp500_csv"] = resolve(config["sp500_csv"], raw) if config.get("sp500_csv") else None
    return paths


def fingerprint(path):
    path = Path(path).resolve()
    stat = path.stat()
    return {"path": str(path), "size": stat.st_size, "mtime_ns": stat.st_mtime_ns}


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=path.name + ".", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w") as handle:
            json.dump(value, handle, indent=2, sort_keys=True)
            handle.write("\n")
        os.replace(temporary, path)
    finally:
        Path(temporary).unlink(missing_ok=True)


def metadata_path(path):
    return Path(str(path) + ".metadata.json")


def iter_batches(path, columns, batch_rows=100_000, block_mb=64, column_types=None):
    if batch_rows <= 0 or block_mb <= 0:
        raise ValueError("Batch rows and CSV block size must be positive.")
    path = Path(path)
    if path.suffix.lower() == ".csv":
        types = {"ID": pa.string(), "BODY": pa.string(), "SKILL_SUBCATEGORY_NAME": pa.string()}
        types.update(column_types or {})
        types = {name: value for name, value in types.items() if name in columns}
        reader = pacsv.open_csv(
            path,
            read_options=pacsv.ReadOptions(block_size=block_mb << 20),
            parse_options=pacsv.ParseOptions(newlines_in_values=True),
            convert_options=pacsv.ConvertOptions(
                include_columns=columns, column_types=types, strings_can_be_null=True,
            ),
        )
        for batch in reader:
            for offset in range(0, batch.num_rows, batch_rows):
                yield batch.slice(offset, batch_rows)
    else:
        yield from pq.ParquetFile(path).iter_batches(batch_size=batch_rows, columns=columns)


def keyword_specs():
    from scripts import label_benefits as labels

    return [
        ("EDU_ASSISTANCE", "EDU_ASSISTANCE_POSITION", labels.edu_assistance, labels.tuition_to_exclude),
        ("PAID LEAVE", "PAID_LEAVE_POSITION", labels.leave, None),
        ("WELLBEING", None, labels.wellbeing, labels.wellbeing_to_exclude),
        ("HEALTH_WELLBEING", "HEALTH_WELLBEING_POSITION", labels.health_wellbeing, labels.wellbeing_to_exclude),
        ("PARENTAL_LEAVE", "PARENTAL_LEAVE_POSITION", labels.parental_leave, None),
        ("CULTURE", "CULTURE_POSITION", labels.culture, None),
    ]


def label_schema(kind, positions):
    names = BENEFIT_NAMES if kind == "benefits" else ["REMOTE_KW"]
    position_names = POSITION_NAMES if kind == "benefits" else ["REMOTE_KW_POSITION"]
    fields = [("ID", pa.string())] + [(name, pa.bool_()) for name in names]
    if positions:
        fields += [(name, pa.float64()) for name in position_names]
    return pa.schema(fields)


def label_batch(batch, kind, positions=True, correct_remote_keywords=False):
    from scripts import label_benefits as labels
    from scripts import label_benefits_remote as remote

    frame = batch.to_pandas().reset_index(drop=True)
    if frame["ID"].isna().any() or frame["ID"].str.strip().eq("").any():
        raise ValueError("Body labels require nonempty posting IDs.")
    output = pd.DataFrame({"ID": frame["ID"]})
    body = frame["BODY"]
    if kind == "benefits":
        for name, position, keywords, exclusions in keyword_specs():
            output[name] = labels.check_benefits(body, keywords, exclusions)
            if positions and position:
                output[position] = labels.check_benefit_position(body, keywords, exclusions)
    else:
        # Match the original remote labeler's normalization, not the benefit one.
        body = body.fillna("").str.strip().str.replace(r"\s+", " ", regex=True).str.lower()
        keywords = remote.get_remote_keywords(correct_remote_keywords)
        output["REMOTE_KW"] = remote.check_benefits(body, keywords, remote.remote_to_exclude)
        if positions:
            output["REMOTE_KW_POSITION"] = remote.check_benefit_position(body, keywords, remote.remote_to_exclude)
    return pa.Table.from_pandas(output, schema=label_schema(kind, positions), preserve_index=False)


def write_label_part(batch, path, kind, positions, correction):
    table = label_batch(batch, kind, positions, correction)
    temporary = Path(str(path) + ".tmp")
    pq.write_table(table, temporary, compression="gzip")
    os.replace(temporary, path)
    return table.num_rows


def stream_labels(source, output, kind, *, batch_rows=100_000, block_mb=64,
                  workers=1, positions=True, correct_remote_keywords=False, overwrite=False):
    source, output = Path(source), Path(output)
    if kind not in {"benefits", "remote"} or workers < 1:
        raise ValueError("Expected benefits/remote labels and at least one worker.")
    if output.resolve() == source.resolve():
        raise ValueError("Input and output must be different files.")
    if output.exists() and not overwrite:
        raise FileExistsError(f"Output exists: {output}; use --overwrite explicitly.")
    output.parent.mkdir(parents=True, exist_ok=True)
    parts = output.with_name(output.name + ".parts")
    parts.mkdir(exist_ok=True)
    manifest_path = parts / "manifest.json"
    source_files = [Path(__file__), REPO_ROOT / "scripts/label_benefits.py",
                    REPO_ROOT / "scripts/label_benefits_remote.py"]
    manifest = {
        "input": fingerprint(source), "kind": kind, "batch_rows": batch_rows,
        "block_mb": block_mb, "positions": positions,
        "correct_remote_keywords": correct_remote_keywords,
        "code_sha256": {path.name: hashlib.sha256(path.read_bytes()).hexdigest() for path in source_files},
    }
    if manifest_path.exists():
        if json.loads(manifest_path.read_text()) != manifest:
            raise ValueError("Checkpoint input, code, or settings changed; choose a new output path.")
    elif any(parts.iterdir()):
        raise ValueError("Checkpoint directory has no manifest; choose a new output path.")
    else:
        atomic_json(manifest_path, manifest)

    schema = label_schema(kind, positions)
    expected = []
    futures = set()

    def consume(completed):
        for future in completed:
            future.result()

    pool = ProcessPoolExecutor(max_workers=workers) if workers > 1 else None
    try:
        for index, batch in enumerate(iter_batches(source, ["ID", "BODY"], batch_rows, block_mb)):
            part = parts / f"part-{index:06d}.parquet"
            expected.append(part)
            if part.exists():
                saved = pq.ParquetFile(part)
                if saved.metadata.num_rows != batch.num_rows or not saved.schema_arrow.equals(schema):
                    raise ValueError(f"Invalid checkpoint part: {part}")
                continue
            if pool is None:
                write_label_part(batch, part, kind, positions, correct_remote_keywords)
            else:
                while len(futures) >= workers + 2:
                    completed, futures = wait(futures, return_when=FIRST_COMPLETED)
                    consume(completed)
                futures.add(pool.submit(write_label_part, batch, part, kind, positions, correct_remote_keywords))
        consume(futures)
    finally:
        if pool is not None:
            pool.shutdown(wait=True, cancel_futures=True)

    if set(parts.glob("part-*.parquet")) != set(expected):
        raise ValueError("Checkpoint part count does not match the complete input.")
    descriptor, temporary = tempfile.mkstemp(prefix=output.name + ".", suffix=".tmp", dir=output.parent)
    os.close(descriptor)
    total = 0
    try:
        with pq.ParquetWriter(temporary, schema, compression="gzip") as writer:
            for part in expected:
                for batch in pq.ParquetFile(part).iter_batches(batch_size=batch_rows):
                    writer.write_batch(batch)
                    total += batch.num_rows
        if fingerprint(source) != manifest["input"]:
            raise ValueError("Input changed during labeling; output was not replaced.")
        os.replace(temporary, output)
        atomic_json(metadata_path(output), {**manifest, "complete": True, "rows": total,
                                            "output": fingerprint(output)})
    finally:
        Path(temporary).unlink(missing_ok=True)
    for part in expected:
        part.unlink()
    manifest_path.unlink()
    parts.rmdir()
    print(f"Wrote {total:,} {kind} labels to {output}")
    return total


def add_structured_labels(frame):
    def parse(value):
        if isinstance(value, str) and value.startswith("["):
            try:
                parsed = json.loads(value)
                return parsed if isinstance(parsed, list) else []
            except (json.JSONDecodeError, ValueError):
                pass
        return []

    parsed = frame["BENEFIT_SUBCATEGORY_NAME"].apply(parse)
    for name, terms in STRUCTURED_TERMS.items():
        allowed = {term.lower() for term in terms}
        frame[name] = parsed.apply(lambda values: any(isinstance(v, str) and v.lower() in allowed for v in values))
    frame["S_REMOTE"] = frame["REMOTE_TYPE_NAME"].isin(["Remote", "Hybrid Remote"])
    return frame


def label_cli(kind, legacy_main):
    parser = argparse.ArgumentParser(description=f"Label {kind}; add --stream for bounded-memory body processing.")
    parser.add_argument("--stream", action="store_true")
    parser.add_argument("--input", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--batch-rows", type=int, default=100_000)
    parser.add_argument("--csv-block-mb", type=int, default=64)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--skip-positions", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    if kind == "remote":
        parser.add_argument("--correct-remote-keywords", action="store_true",
                            help="Opt into the missing-comma correction; may change labels and results.")
    args = parser.parse_args()
    if not args.stream:
        if any([args.input, args.output, args.skip_positions, args.overwrite,
                args.batch_rows != 100_000, args.csv_block_mb != 64, args.workers != 1,
                getattr(args, "correct_remote_keywords", False)]):
            parser.error("Processing options require --stream; no arguments keeps the original pipeline.")
        return legacy_main()
    paths = data_paths()
    filename = "benefit_labels_body.parquet" if kind == "benefits" else "remote_labels_body.parquet"
    return stream_labels(
        args.input or paths["body_csv"], args.output or paths["processed_dir"] / filename, kind,
        batch_rows=args.batch_rows, block_mb=args.csv_block_mb, workers=args.workers,
        positions=not args.skip_positions, correct_remote_keywords=getattr(args, "correct_remote_keywords", False),
        overwrite=args.overwrite,
    )


def count_ai_skills(source, output, *, batch_rows=100_000, block_mb=64, overwrite=False):
    source, output = Path(source), Path(output)
    if source.resolve() == output.resolve():
        raise ValueError("Input and output must be different files.")
    if output.exists() and not overwrite:
        raise FileExistsError(f"Output exists: {output}; use --overwrite explicitly.")
    original = fingerprint(source)
    counts = Counter()
    rows = 0
    for batch in iter_batches(source, ["ID", "SKILL_SUBCATEGORY_NAME"], batch_rows, block_mb):
        frame = batch.to_pandas()
        rows += len(frame)
        ai = frame.loc[frame["SKILL_SUBCATEGORY_NAME"].eq("Artificial Intelligence and Machine Learning (AI/ML)"), "ID"]
        counts.update(ai.dropna().loc[lambda ids: ids.str.strip().ne("")])
    if fingerprint(source) != original:
        raise ValueError("Skills input changed during counting.")
    output.parent.mkdir(parents=True, exist_ok=True)
    table = pa.table({"ID": pa.array(list(counts), type=pa.string()),
                      "AI_SKILL_COUNT": pa.array(list(counts.values()), type=pa.int64())})
    descriptor, temporary = tempfile.mkstemp(prefix=output.name + ".", suffix=".tmp", dir=output.parent)
    os.close(descriptor)
    try:
        pq.write_table(table, temporary, compression="gzip")
        os.replace(temporary, output)
        atomic_json(metadata_path(output), {"input": original, "complete": True, "skill_rows": rows,
                                            "rows": len(counts), "output": fingerprint(output)})
    finally:
        Path(temporary).unlink(missing_ok=True)
    print(f"Counted {rows:,} skill rows; {len(counts):,} AI posting IDs in {output}")
    return len(counts)
