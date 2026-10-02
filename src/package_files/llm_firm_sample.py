"""Select employers with explicit SME/large LLM labels for robustness runs."""

from pathlib import Path

import pandas as pd


def company_keys(values):
    """Match integer COMPANY IDs across CSV and Parquet representations."""
    return values.astype('string').str.strip().str.replace(r'\.0+$', '', regex=True)


def read_firm_labels(path):
    labels = pd.read_csv(Path(path), dtype={'COMPANY': 'string'})
    if not {'COMPANY', 'llm_label'}.issubset(labels.columns):
        raise ValueError('LLM labels CSV must contain COMPANY and llm_label.')
    labels['COMPANY'] = company_keys(labels['COMPANY'])
    labels['llm_label'] = labels['llm_label'].astype('string').str.strip().str.lower()
    labels = labels[labels['llm_label'].isin(['sme', 'large'])
                    & labels['COMPANY'].notna() & labels['COMPANY'].ne('0')]
    if labels.groupby('COMPANY')['llm_label'].nunique().gt(1).any():
        raise ValueError('Conflicting SME/large labels for the same COMPANY.')
    if labels.empty:
        raise ValueError('No explicit SME/large firm labels found.')
    return labels.drop_duplicates('COMPANY').set_index('COMPANY')['llm_label']


def select_llm_firms(data, labels):
    """Keep all years for labeled employers with an observed 2018 size."""
    mapped = company_keys(data['COMPANY']).map(labels)
    eligible = mapped.notna() & pd.to_numeric(data['FIRM_POSTING_COUNT'], errors='coerce').notna()
    result = data.loc[eligible].copy()
    result['LLM_FIRM_LABEL'] = mapped.loc[eligible]
    if result.empty:
        raise ValueError('No labeled firms with observed 2018 size match the posting dataset.')
    return result
