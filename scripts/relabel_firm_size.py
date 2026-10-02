#!/usr/bin/env python3
"""Refresh existing Parquets with one common first-year firm-size definition."""

import argparse
import json
import os
from pathlib import Path
import sys
import tempfile

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

sys.path.append(str(Path(__file__).resolve().parents[1] / 'src'))
from package_files.config_utils import get_processed_dir
from package_files.firm_size import FIRM_SIZE_CUTOFF, baseline_firm_counts, firm_size_columns

DEFAULT_FILES = ('data_v1.parquet', 'labeled_v1.parquet', 'labeled_v2.parquet')


def relabel_file(path, counts, baseline_year, dry_run=False, batch_size=100_000):
    """Rewrite in batches, replacing the original only after a successful write."""
    source = pq.ParquetFile(path)
    names = source.schema_arrow.names
    if 'COMPANY' not in names:
        raise ValueError(f'{path} must contain COMPANY.')
    read_columns = None if not dry_run else [
        name for name in ('COMPANY', 'SP500', 'FIRM_POSTING_COUNT', 'FIRM_SIZE_BUCKET') if name in names
    ]
    rows = changed_counts = changed_buckets = unknown_size = 0
    writer = None
    temporary_path = None
    try:
        if not dry_run:
            descriptor, temporary_name = tempfile.mkstemp(prefix=path.name + '.', suffix='.tmp', dir=path.parent)
            os.close(descriptor)
            temporary_path = Path(temporary_name)
        for batch in source.iter_batches(batch_size=batch_size, columns=read_columns):
            table = pa.Table.from_batches([batch])
            selected = [name for name in ('COMPANY', 'SP500') if name in table.column_names]
            identifiers = table.select(selected).to_pandas().reset_index(drop=True)
            columns = firm_size_columns(identifiers, counts, baseline_year)
            old_counts = pd.to_numeric(table['FIRM_POSTING_COUNT'].to_pandas(), errors='coerce') if 'FIRM_POSTING_COUNT' in table.column_names else pd.Series(float('nan'), index=identifiers.index)
            changed_counts += int(old_counts.fillna(-1).ne(columns['FIRM_POSTING_COUNT'].fillna(-1)).sum())
            old_buckets = table['FIRM_SIZE_BUCKET'].to_pandas().astype('string') if 'FIRM_SIZE_BUCKET' in table.column_names else pd.Series(pd.NA, index=identifiers.index, dtype='string')
            changed_buckets += int(old_buckets.fillna('NA').ne(columns['FIRM_SIZE_BUCKET'].fillna('NA')).sum())
            unknown_size += int(columns['FIRM_SIZE_BUCKET'].isna().sum())
            rows += len(identifiers)
            if dry_run:
                continue
            replacement = pa.Table.from_pandas(columns, preserve_index=False)
            table = table.drop([name for name in columns if name in table.column_names])
            for name in replacement.column_names:
                table = table.append_column(name, replacement[name])
            # Retain metadata for every original column, including categorical
            # fields and indices, but update descriptions of the replaced fields.
            metadata = dict(table.schema.metadata or {})
            if b'pandas' in metadata:
                pandas_metadata = json.loads(metadata[b'pandas'])
                new_metadata = json.loads(replacement.schema.metadata[b'pandas'])
                pandas_metadata['columns'] = [
                    item for item in pandas_metadata['columns'] if item['field_name'] not in columns
                ] + new_metadata['columns']
                metadata[b'pandas'] = json.dumps(pandas_metadata).encode()
            table = table.replace_schema_metadata(metadata)
            if writer is None:
                writer = pq.ParquetWriter(temporary_path, table.schema, compression='gzip')
            writer.write_table(table)
        if not dry_run:
            if writer is None:
                raise ValueError(f'{path} contains no postings.')
            writer.close()
            writer = None
            source.close()
            os.replace(temporary_path, path)
            temporary_path = None
    finally:
        source.close()
        if writer is not None:
            writer.close()
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)
    return {
        'FILE': path.name, 'ROWS': rows, 'BASELINE_YEAR': baseline_year,
        'CHANGED_POSTING_COUNTS': changed_counts, 'CHANGED_SIZE_BUCKETS': changed_buckets,
        'FIRM_SIZE_CUTOFF': FIRM_SIZE_CUTOFF, 'UNKNOWN_SIZE_POSTINGS': unknown_size,
        'ABSENT_FIRMS': 'unknown', 'DRY_RUN': dry_run,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--processed-dir', type=Path, help='Directory containing existing processed Parquets.')
    parser.add_argument('--source', type=Path, help='Complete posting dataset for baseline counts; defaults to labeled_v2.parquet, then data_v1.parquet.')
    parser.add_argument('--files', nargs='+', default=list(DEFAULT_FILES), help='Parquet filenames to refresh.')
    parser.add_argument('--dry-run', action='store_true', help='Report changes without writing Parquets or an audit file.')
    args = parser.parse_args()
    processed_dir = args.processed_dir or get_processed_dir()
    source_path = args.source
    if source_path is None:
        source_path = processed_dir / 'labeled_v2.parquet'
        if not source_path.exists():
            source_path = processed_dir / 'data_v1.parquet'
    if not source_path.exists():
        raise FileNotFoundError(f'Baseline source dataset not found: {source_path}')
    schema = pq.read_schema(source_path)
    date_column = 'YEAR' if 'YEAR' in schema.names else 'POSTED'
    baseline_data = pd.read_parquet(source_path, columns=['COMPANY', date_column])
    counts, baseline_year = baseline_firm_counts(baseline_data)
    del baseline_data
    print(f'Baseline source: {source_path}; year: {baseline_year}; observed firms: {len(counts):,}')
    paths = [processed_dir / filename for filename in args.files if (processed_dir / filename).exists()]
    if not paths:
        raise FileNotFoundError(f'No requested Parquet files found in {processed_dir}')
    audits = []
    for path in paths:
        audit = relabel_file(path, counts, baseline_year, dry_run=args.dry_run)
        audits.append(audit)
        print(f"{path.name}: {audit['CHANGED_POSTING_COUNTS']:,} counts changed; {audit['CHANGED_SIZE_BUCKETS']:,} size buckets changed (cutoff {FIRM_SIZE_CUTOFF}); {audit['UNKNOWN_SIZE_POSTINGS']:,} postings with unknown size")
    if not args.dry_run:
        audit_path = processed_dir / 'firm_size_relabel_audit.csv'
        pd.DataFrame(audits).to_csv(audit_path, index=False)
        print(f'Audit saved: {audit_path}')


if __name__ == '__main__':
    main()
