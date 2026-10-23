#!/usr/bin/env python3
"""Plot within-occupation and within-firm AI minus non-AI benefit-prevalence gaps."""

import argparse
from collections import Counter
import json
import os
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from scipy.ndimage import gaussian_filter1d
from scipy.stats import ttest_1samp

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from package_files.benefits_defns import benefits4, benefits_labels_map, occupation
from package_files.config_utils import get_processed_dir, get_repo_root


def binary_values(series, name):
    values = pd.to_numeric(series, errors='coerce')
    if (series.notna() & ~values.isin([0, 1])).any():
        raise ValueError(f'{name} must contain only True/False, 0/1, or missing values.')
    return values


def summarize_postings(path, occupation_column=occupation, benefits=None, batch_size=100_000):
    """Aggregate benefit numerators/denominators in batches without reading BODY."""
    benefits = benefits4 if benefits is None else benefits
    source = pq.ParquetFile(path)
    names = source.schema_arrow.names
    required = {'COMPANY', 'AI ROLE', occupation_column, *benefits}
    if not required.issubset(names):
        raise ValueError(f'Missing required columns: {sorted(required - set(names))}')
    date_column = 'YEAR' if 'YEAR' in names else 'POSTED'
    if date_column not in names:
        raise ValueError('YEAR or POSTED is required to identify firms observed in 2018.')
    columns = list(dict.fromkeys(['COMPANY', 'AI ROLE', occupation_column, date_column] + list(benefits)))
    summaries = {occupation_column: None, 'COMPANY': None}
    baseline_counts = Counter()
    rows = missing_ai = missing_occupation = invalid_firm = 0
    for batch_number, batch in enumerate(source.iter_batches(batch_size=batch_size, columns=columns), 1):
        data = batch.to_pandas()
        rows += len(data)
        years = (pd.to_numeric(data['YEAR'], errors='coerce') if date_column == 'YEAR'
                 else pd.to_datetime(data['POSTED'], errors='coerce').dt.year)
        classified = data['COMPANY'].notna() & ~data['COMPANY'].astype('string').isin(['0', '0.0', ''])
        baseline_counts.update(data.loc[classified & years.eq(2018), 'COMPANY'].value_counts().to_dict())
        ai = binary_values(data['AI ROLE'], 'AI ROLE')
        valid_ai = ai.notna()
        missing_ai += int((~valid_ai).sum())
        data = data.assign(_ai=ai.fillna(0).astype(bool))
        known_occupation = data[occupation_column].notna() & data[occupation_column].astype('string').str.strip().ne('')
        missing_occupation += int((valid_ai & ~known_occupation).sum())
        invalid_firm += int((valid_ai & ~classified).sum())
        measures = pd.DataFrame({'POSTINGS': np.ones(len(data), dtype='int64')}, index=data.index)
        for benefit in benefits:
            values = binary_values(data[benefit], benefit)
            measures[benefit + '__valid'] = values.notna().astype('int64')
            measures[benefit + '__present'] = values.fillna(0).astype('int64')
        for group, mask in [(occupation_column, valid_ai & known_occupation), ('COMPANY', valid_ai & classified)]:
            frame = pd.concat([data[[group, '_ai']], measures], axis=1).loc[mask]
            aggregated = frame.groupby([group, '_ai'], observed=True)[measures.columns.tolist()].sum()
            summaries[group] = aggregated if summaries[group] is None else summaries[group].add(aggregated, fill_value=0)
        if batch_number % 10 == 0:
            print(f'Aggregated {rows:,} postings...', flush=True)

    def long_frame(summary, group):
        if summary is None or summary.empty:
            raise ValueError(f'No usable postings for {group}.')
        # Retain explicit NA prevalence for groups with no AI or no non-AI jobs.
        groups = summary.index.get_level_values(group).unique()
        pairs = pd.MultiIndex.from_product([groups, [False, True]], names=[group, '_ai'])
        wide = summary.reindex(pairs, fill_value=0).fillna(0).astype('int64').reset_index()
        blocks = []
        for benefit in benefits:
            table = wide[[group, 'POSTINGS']].copy()
            table['ROLE_TYPE'] = np.where(wide['_ai'], 'AI', 'Non-AI')
            table['BENEFIT'] = benefit
            table['BENEFIT_LABEL'] = benefits_labels_map.get(benefit, benefit)
            table['VALID_POSTINGS'] = wide[benefit + '__valid']
            table['BENEFIT_POSTINGS'] = wide[benefit + '__present']
            table['PREVALENCE_PERCENT'] = 100 * table['BENEFIT_POSTINGS'] / table['VALID_POSTINGS'].replace(0, np.nan)
            blocks.append(table)
        return pd.concat(blocks, ignore_index=True)

    occupations = long_frame(summaries[occupation_column], occupation_column)
    all_firms = long_frame(summaries['COMPANY'], 'COMPANY')
    firms = all_firms[all_firms['COMPANY'].isin(baseline_counts)].copy()
    firms['POSTINGS_2018'] = firms['COMPANY'].map(baseline_counts).astype('int64')
    audit = {
        'input_path': str(Path(path).resolve()), 'occupation_column': occupation_column,
        'benefits': list(benefits), 'rows_read': rows,
        'missing_ai_flag_postings_excluded': missing_ai,
        'missing_occupation_postings_excluded': missing_occupation,
        'unclassified_firm_postings_excluded': invalid_firm,
        'firms_absent_in_2018_excluded': int(all_firms['COMPANY'].nunique() - firms['COMPANY'].nunique()),
        'occupations': int(occupations[occupation_column].nunique()), 'firms': int(firms['COMPANY'].nunique()),
        'firm_baseline_year': 2018,
        'posting_years': 'All available years; 2018 determines firm eligibility only.',
        'sp500': 'Included among individual firms if observed in 2018.',
        'denominator': 'Within group and AI/non-AI type, postings with a known flag for the given benefit.',
        'zero_denominators': 'Prevalence is NA, never zero; excluded from density estimates.',
        'distribution_weighting': 'Each occupation/firm has equal weight.',
    }
    return occupations, firms, audit


def filter_firms(table, min_postings=100):
    """Filter on total AI + non-AI postings, counting each role only once."""
    if min_postings < 1:
        raise ValueError('Minimum firm postings must be at least 1.')
    # POSTINGS is repeated for each benefit in the long prevalence table.
    totals = table.drop_duplicates(['COMPANY', 'ROLE_TYPE']).groupby('COMPANY')['POSTINGS'].sum()
    result = table.assign(TOTAL_FIRM_POSTINGS=table['COMPANY'].map(totals))
    return result.loc[result['TOTAL_FIRM_POSTINGS'].ge(min_postings)].copy()


def smooth_density(values, bandwidth=2.0, bounds=(0, 100)):
    """Binned Gaussian density reflected at the possible prevalence/gap bounds."""
    values = np.asarray(values, dtype=float)
    lower, upper = bounds
    if not len(values) or not np.isfinite(values).all() or ((values < lower) | (values > upper)).any():
        raise ValueError(f'Density requires finite values between {lower} and {upper}.')
    if not np.isfinite(bandwidth) or bandwidth <= 0:
        raise ValueError('Bandwidth must be positive, in percentage points.')
    edges = np.linspace(lower, upper, round((upper - lower) * 10) + 1)
    mass, _ = np.histogram(values, bins=edges)
    width = edges[1] - edges[0]
    density = gaussian_filter1d(mass.astype(float), bandwidth / width, mode='reflect') / (len(values) * width)
    return (edges[:-1] + edges[1:]) / 2, density


def plot_distributions(table, group_column, title, output_path, bandwidth=2.0, min_postings=1, paired_only=False):
    os.environ.setdefault('MPLCONFIGDIR', '/tmp/beyond-salary-matplotlib')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    benefit_keys = table['BENEFIT'].unique()
    ncols = min(3, len(benefit_keys))
    nrows = int(np.ceil(len(benefit_keys) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.2 * ncols, 3.5 * nrows), squeeze=False)
    for ax, benefit in zip(axes.ravel(), benefit_keys):
        subset = table[(table['BENEFIT'] == benefit) & table['VALID_POSTINGS'].ge(min_postings)
                       & table['PREVALENCE_PERCENT'].notna()]
        if paired_only:
            valid_groups = subset.groupby(group_column)['ROLE_TYPE'].nunique()
            subset = subset[subset[group_column].isin(valid_groups[valid_groups.eq(2)].index)]
        for role, label, color in [('AI', 'AI jobs', '#1f77b4'), ('Non-AI', 'Non-AI jobs', '#ff7f0e')]:
            values = subset.loc[subset['ROLE_TYPE'] == role, 'PREVALENCE_PERCENT']
            if values.empty:
                ax.plot([], [], color=color, label=f'{label} (n=0)')
                continue
            x, density = smooth_density(values, bandwidth)
            ax.plot(x, density, color=color, linewidth=2, label=f'{label} (n={len(values):,})')
            ax.fill_between(x, density, color=color, alpha=0.22)
        ax.set_title(benefits_labels_map.get(benefit, benefit))
        ax.set_xlim(0, 100)
        ax.set_ylim(bottom=0)
        ax.set_xlabel('Posts offering benefit (%)')
        ax.set_ylabel('Density')
        ax.grid(axis='y', alpha=0.2)
        ax.legend(frameon=False, fontsize=9)
    for ax in axes.ravel()[len(benefit_keys):]:
        ax.set_visible(False)
    fig.suptitle(title)
    pairing = 'Both job types required per group.' if paired_only else 'Each curve includes groups with that job type.'
    fig.text(0.5, 0.01, f'Equal weight per group; minimum {min_postings} valid posts per job type; bandwidth {bandwidth:g} pp. {pairing}',
             ha='center', fontsize=8)
    fig.tight_layout(rect=(0, 0.04, 1, 0.97))
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300, bbox_inches='tight')
    fig.savefig(output_path.with_suffix('.pdf'), bbox_inches='tight')
    plt.close(fig)


def prevalence_differences(table, group_column, min_postings=1):
    """Match job types within each group/benefit, then subtract their prevalences."""
    usable = table[table['VALID_POSTINGS'].ge(min_postings) & table['PREVALENCE_PERCENT'].notna()]
    keys = [group_column, 'BENEFIT']
    ai = usable.loc[usable['ROLE_TYPE'].eq('AI'), keys + ['PREVALENCE_PERCENT', 'VALID_POSTINGS']].rename(
        columns={'PREVALENCE_PERCENT': 'AI_PREVALENCE_PERCENT', 'VALID_POSTINGS': 'AI_VALID_POSTINGS'})
    non_ai = usable.loc[usable['ROLE_TYPE'].eq('Non-AI'), keys + ['PREVALENCE_PERCENT', 'VALID_POSTINGS']].rename(
        columns={'PREVALENCE_PERCENT': 'NON_AI_PREVALENCE_PERCENT', 'VALID_POSTINGS': 'NON_AI_VALID_POSTINGS'})
    result = ai.merge(non_ai, on=keys, how='inner', validate='one_to_one')
    result['DIFFERENCE_PP'] = result['AI_PREVALENCE_PERCENT'] - result['NON_AI_PREVALENCE_PERCENT']
    result['BENEFIT_LABEL'] = result['BENEFIT'].map(lambda key: benefits_labels_map.get(key, key))
    for column in ['TOTAL_FIRM_POSTINGS', 'POSTINGS_2018']:
        if column in table:
            result[column] = result[group_column].map(table.drop_duplicates(group_column).set_index(group_column)[column])
    return result


def mean_difference_test(values):
    """Test the equally weighted mean of observed group gaps against zero."""
    values = pd.to_numeric(pd.Series(values), errors='raise').dropna().to_numpy(dtype=float)
    if not np.isfinite(values).all():
        raise ValueError('Mean tests require finite differences.')
    n = len(values)
    result = {
        'N_GROUPS': n, 'MEAN_DIFFERENCE_PP': float(values.mean()) if n else np.nan,
        'STANDARD_ERROR_PP': np.nan, 'T_STATISTIC': np.nan,
        'DEGREES_OF_FREEDOM': n - 1 if n else np.nan, 'P_VALUE': np.nan,
        'CI95_LOWER_PP': np.nan, 'CI95_UPPER_PP': np.nan,
        'SIGNIFICANT_05': pd.NA, 'TEST_STATUS': 'insufficient_observations',
    }
    if n < 2:
        return result
    std = values.std(ddof=1)
    result['STANDARD_ERROR_PP'] = float(std / np.sqrt(n))
    if std == 0:
        result['TEST_STATUS'] = 'zero_variance'
        return result
    test = ttest_1samp(values, popmean=0, alternative='two-sided')
    interval = test.confidence_interval(confidence_level=.95)
    result.update(T_STATISTIC=float(test.statistic), P_VALUE=float(test.pvalue),
                  CI95_LOWER_PP=float(interval.low), CI95_UPPER_PP=float(interval.high),
                  SIGNIFICANT_05=bool(test.pvalue < .05), TEST_STATUS='ok')
    return result


def mean_difference_tests(table, benefit_keys, group_type):
    rows = []
    for benefit in benefit_keys:
        values = table.loc[table['BENEFIT'].eq(benefit), 'DIFFERENCE_PP']
        rows.append({'GROUP_TYPE': group_type, 'BENEFIT': benefit,
                     'BENEFIT_LABEL': benefits_labels_map.get(benefit, benefit),
                     **mean_difference_test(values)})
    return pd.DataFrame(rows)


def plot_difference_distributions(table, title, output_path, benefit_keys, bandwidth=2.0, min_postings=1, mean_tests=None):
    os.environ.setdefault('MPLCONFIGDIR', '/tmp/beyond-salary-matplotlib')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    if mean_tests is None:
        mean_tests = mean_difference_tests(table, benefit_keys, None)
    mean_tests = mean_tests.set_index('BENEFIT')
    ncols = min(3, len(benefit_keys))
    nrows = int(np.ceil(len(benefit_keys) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.2 * ncols, 3.5 * nrows), squeeze=False)
    maximum = table['DIFFERENCE_PP'].abs().max() if not table.empty else 0
    limit = min(100, max(10, np.ceil((maximum + 4 * bandwidth) / 5) * 5))
    for ax, benefit in zip(axes.ravel(), benefit_keys):
        values = table.loc[table['BENEFIT'].eq(benefit), 'DIFFERENCE_PP'].dropna()
        test = mean_tests.loc[benefit]
        if len(values):
            x, density = smooth_density(values, bandwidth, bounds=(-100, 100))
            ax.plot(x, density, color='#1f77b4', linewidth=2)
            ax.fill_between(x, density, color='#1f77b4', alpha=0.22)
            mean = test['MEAN_DIFFERENCE_PP']
            ax.axvline(mean, color='#9c2c77', linestyle=':', linewidth=1.8)
            ax.text(.97, .85, f'Mean: {mean:+.2f} pp', transform=ax.transAxes,
                    ha='right', va='top', fontsize=10, color='#9c2c77')
        else:
            ax.text(.5, .5, 'No groups with both job types', transform=ax.transAxes, ha='center', fontsize=9)
        p_label = 'test unavailable'
        if test['TEST_STATUS'] == 'ok':
            p_label = 'p < 0.001' if test['P_VALUE'] < .001 else f'p = {test["P_VALUE"]:.3g}'
        ax.text(.97, .76, p_label, transform=ax.transAxes, ha='right', va='top', fontsize=9)
        ax.axvline(0, color='#555555', linestyle='--', linewidth=1)
        ax.text(.97, .94, f'n={len(values):,}', transform=ax.transAxes, ha='right', va='top', fontsize=10)
        ax.set_title(benefits_labels_map.get(benefit, benefit))
        ax.set_xlim(-limit, limit)
        ax.set_ylim(bottom=0)
        ax.set_xlabel('AI − non-AI prevalence (percentage points)')
        ax.set_ylabel('Density')
        ax.grid(axis='y', alpha=.2)
    for ax in axes.ravel()[len(benefit_keys):]:
        ax.set_visible(False)
    fig.suptitle(title)
    fig.text(.5, .01,
             f'Positive = higher prevalence in AI jobs; negative = higher in non-AI jobs.\n'
             f'Equal weight per group; both job types required; minimum {min_postings} valid posts per type; bandwidth {bandwidth:g} pp.\n'
             'Unadjusted two-sided t-tests; significance threshold p < 0.05.',
             ha='center', fontsize=8)
    fig.tight_layout(rect=(0, .075, 1, .97))
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300, bbox_inches='tight')
    fig.savefig(output_path.with_suffix('.pdf'), bbox_inches='tight')
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input-path', type=Path, default=get_processed_dir() / 'labeled_v2.parquet')
    parser.add_argument('--occupation-column', default=occupation, help='Default: SOC_MAJOR_GROUP (SOC2).')
    parser.add_argument('--output-dir', type=Path, default=get_repo_root() / 'results/benefit_prevalence_distributions')
    parser.add_argument('--bandwidth', type=float, default=2.0, help='Smoothing width in percentage points (default: 2).')
    parser.add_argument('--min-postings', type=int, default=1, help='Minimum non-missing benefit flags per group/job type for plotting.')
    parser.add_argument('--min-firm-postings', type=int, default=100,
                        help='Minimum total AI + non-AI postings per firm across all years (default: 100).')
    parser.add_argument('--paired-only', action='store_true', help='Plot only groups with enough valid AI and non-AI posts for each benefit.')
    parser.add_argument('--view', choices=['difference', 'prevalence'], default='difference',
                        help='Plot within-group AI minus non-AI gaps (default), or separate prevalence curves.')
    parser.add_argument('--plot-only', action='store_true', help='Reuse saved prevalence CSVs instead of reading postings.')
    args = parser.parse_args()
    if not np.isfinite(args.bandwidth) or args.bandwidth <= 0 or args.min_postings < 1 or args.min_firm_postings < 1:
        parser.error('--bandwidth must be positive; minimum posting counts must be at least 1.')
    args.output_dir.mkdir(parents=True, exist_ok=True)
    occupation_csv = args.output_dir / 'occupation_benefit_prevalence.csv'
    firm_csv = args.output_dir / 'firm_benefit_prevalence.csv'
    audit_path = args.output_dir / 'sample_audit.json'
    if args.plot_only:
        occupations, firms = pd.read_csv(occupation_csv), pd.read_csv(firm_csv)
        args.occupation_column = json.loads(audit_path.read_text())['occupation_column']
    else:
        print(f'Reading {args.input_path}', flush=True)
        occupations, firms, audit = summarize_postings(args.input_path, args.occupation_column)
        if firms.empty:
            raise ValueError('No firms observed in 2018 have usable postings.')
        occupations.to_csv(occupation_csv, index=False)
        firms.to_csv(firm_csv, index=False)
        audit_path.write_text(json.dumps(audit, indent=2) + '\n')
    original_firms = firms['COMPANY'].nunique()
    firms = filter_firms(firms, args.min_firm_postings)
    if firms.empty:
        raise ValueError('No firms meet --min-firm-postings.')
    firms.to_csv(args.output_dir / 'firm_benefit_prevalence_filtered.csv', index=False)
    print(f'Firm posting filter: {original_firms:,} -> {firms["COMPANY"].nunique():,} employers with at least {args.min_firm_postings:,} total postings.', flush=True)
    settings = {'bandwidth_percentage_points': args.bandwidth, 'min_postings': args.min_postings,
                'min_firm_postings': args.min_firm_postings, 'firms_before_posting_filter': int(original_firms),
                'firms_after_posting_filter': int(firms['COMPANY'].nunique()), 'paired_only': args.paired_only or args.view == 'difference',
                'view': args.view, 'difference_definition': 'AI prevalence minus non-AI prevalence, in percentage points',
                'mean_test': 'unadjusted two-sided one-sample t-test against zero', 'significance_level': .05}
    (args.output_dir / 'plot_settings.json').write_text(json.dumps(settings, indent=2) + '\n')
    test_summaries = []
    for table, group, title, filename in [
        (occupations, args.occupation_column, f'Benefit prevalence across occupations ({args.occupation_column})', 'occupation_benefit_distributions.png'),
        (firms, 'COMPANY', f'Benefit prevalence across firms observed in 2018 (at least {args.min_firm_postings:,} total posts)', 'firm_benefit_distributions.png'),
    ]:
        if args.view == 'difference':
            differences = prevalence_differences(table, group, args.min_postings)
            prefix = 'firm' if group == 'COMPANY' else 'occupation'
            differences.to_csv(args.output_dir / f'{prefix}_benefit_prevalence_differences.csv', index=False)
            tests = mean_difference_tests(differences, table['BENEFIT'].unique(), prefix.capitalize())
            test_summaries.append(tests)
            plot_difference_distributions(differences, title.replace('Benefit prevalence', 'AI − non-AI benefit-prevalence gaps'),
                                          args.output_dir / filename, table['BENEFIT'].unique(), args.bandwidth, args.min_postings,
                                          mean_tests=tests)
            print(f'{prefix}: {differences[group].nunique():,} groups have both job types for at least one benefit.', flush=True)
        else:
            plot_distributions(table, group, title, args.output_dir / filename, args.bandwidth, args.min_postings, args.paired_only)
    if test_summaries:
        pd.concat(test_summaries, ignore_index=True).to_csv(args.output_dir / 'mean_difference_tests.csv', index=False)
    print(f'Saved prevalence tables, audit, PNGs and PDFs to {args.output_dir}', flush=True)


if __name__ == '__main__':
    main()
