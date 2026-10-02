"""Firm-size proxies measured once in the 2018 baseline year."""

import numpy as np
import pandas as pd

# Selected by macro F1 weighted by 2018 postings in the web validation.
FIRM_SIZE_CUTOFF = 6


def baseline_firm_counts(data):
    """Return 2018 posting counts, including an empty mapping if 2018 is absent."""
    if 'COMPANY' not in data:
        raise ValueError('Firm-size classification requires COMPANY.')
    if 'YEAR' in data:
        years = pd.to_numeric(data['YEAR'], errors='coerce')
    elif 'POSTED' in data:
        years = pd.to_datetime(data['POSTED'], errors='coerce').dt.year
    else:
        raise ValueError('First-year firm size requires YEAR or POSTED.')
    valid_years = years.dropna()
    if valid_years.empty:
        raise ValueError('No valid years are available for first-year firm size.')
    baseline_year = 2018
    classified = data['COMPANY'].notna() & data['COMPANY'].ne(0)
    counts = data.loc[classified & years.eq(baseline_year), 'COMPANY'].value_counts()
    return counts, baseline_year


def firm_size_columns(data, counts, baseline_year):
    """Apply the same baseline counts to every year; never use later volumes."""
    classified = data['COMPANY'].notna() & data['COMPANY'].ne(0)
    posting_count = data['COMPANY'].map(counts).astype('Int64').where(classified)
    bucket = pd.cut(posting_count, bins=[0, FIRM_SIZE_CUTOFF - 1, np.inf],
                    labels=[f'SMEs (<{FIRM_SIZE_CUTOFF})', f'Large firms (>={FIRM_SIZE_CUTOFF})']).astype('string')
    if 'SP500' in data:
        bucket = bucket.mask(posting_count.notna() & data['SP500'].fillna(False).astype(bool), 'S&P 500 firms')
    return pd.DataFrame({
        'FIRM_POSTING_COUNT': posting_count,
        'FIRM_SIZE_BUCKET': bucket,
        'FIRM_SIZE_BASE_YEAR': pd.Series(baseline_year, index=data.index, dtype='Int64'),
    }, index=data.index)


def add_baseline_firm_size(data):
    counts, baseline_year = baseline_firm_counts(data)
    result = data.copy()
    columns = firm_size_columns(data, counts, baseline_year)
    for column in columns:
        result[column] = columns[column]
    return result
