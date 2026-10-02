import contextlib
import io
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

os.environ.setdefault('MPLCONFIGDIR', '/tmp/beyond-salary-matplotlib')
os.environ.setdefault('MPLBACKEND', 'Agg')
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))

import pandas as pd
import pyarrow.parquet as pq
from package_files.firm_size import add_baseline_firm_size, baseline_firm_counts, firm_size_columns
from scripts import prepare_data, relabel_firm_size
from analysis import regression_models, descriptive_analysis, wage_perk_interaction_analysis, high_ai_firm_analysis
from scripts import plot_firm_category_figure1


class BaselineFirmSizeTests(unittest.TestCase):
    def test_later_growth_does_not_change_size_or_baseline_counts(self):
        data = pd.DataFrame({'COMPANY': [1] * 5 + [2] * 6 + [1] * 80 + [2],
                             'YEAR': [2018] * 11 + [2025] * 81})
        result = prepare_data.add_firm_size(data)
        self.assertTrue(result.loc[result['COMPANY'].eq(1), 'FIRM_POSTING_COUNT'].eq(5).all())
        self.assertTrue(result.loc[result['COMPANY'].eq(2), 'FIRM_POSTING_COUNT'].eq(6).all())
        self.assertEqual(result.iloc[-1]['FIRM_SIZE_BUCKET'], 'Large firms (>=6)')
        self.assertEqual(result.iloc[11]['FIRM_SIZE_BUCKET'], 'SMEs (<6)')
        self.assertTrue(result['FIRM_SIZE_BASE_YEAR'].eq(2018).all())
        self.assertNotIn('FIRM_POSTING_COUNT', data)
        for module in (regression_models, descriptive_analysis, plot_firm_category_figure1):
            categories = module.add_firm_category(result)
            self.assertTrue(categories.loc[result.COMPANY.eq(1), 'FIRM_CATEGORY'].eq('SMEs').all())
            self.assertTrue(categories.loc[result.COMPANY.eq(2), 'FIRM_CATEGORY'].eq('Large firms').all())
        renamed = result.assign(SP500=False).rename(columns={'SP500': 'sp500', 'FIRM_POSTING_COUNT': 'firm_posting_count'})
        self.assertTrue(wage_perk_interaction_analysis.SAMPLE_FILTERS['SMEs'](renamed).eq(renamed.COMPANY.eq(1)).all())
        self.assertTrue(wage_perk_interaction_analysis.SAMPLE_FILTERS['Large firms'](renamed).eq(renamed.COMPANY.eq(2)).all())

    def test_later_entrants_and_unclassified_firms_remain_unknown(self):
        data = pd.DataFrame({'COMPANY': [1, 1, 2, 0, None, 3],
                             'YEAR': [2018, 2025, 2025, 2018, 2025, 2025],
                             'SP500': [False, False, False, False, False, True]})
        result = add_baseline_firm_size(data)
        self.assertTrue(result.loc[[2, 3, 4, 5], 'FIRM_POSTING_COUNT'].isna().all())
        self.assertTrue(result.loc[[2, 3, 4, 5], 'FIRM_SIZE_BUCKET'].isna().all())
        for module in (regression_models, descriptive_analysis, plot_firm_category_figure1):
            categories = module.add_firm_category(result)
            self.assertTrue(categories.loc[[2, 3, 4, 5], 'FIRM_CATEGORY'].isna().all())
            self.assertEqual(categories.loc[0, 'FIRM_CATEGORY'], 'SMEs')
        renamed = result.rename(columns={'SP500': 'sp500', 'FIRM_POSTING_COUNT': 'firm_posting_count'})
        for name in ('SMEs', 'Large firms', 'S&P 500 firms'):
            self.assertFalse(wage_perk_interaction_analysis.SAMPLE_FILTERS[name](renamed).fillna(False).iloc[2:].any())
        self.assertTrue(wage_perk_interaction_analysis.SAMPLE_FILTERS['Full sample'](renamed).all())

    def test_baseline_is_common_calendar_year_not_each_firms_first_year(self):
        data = pd.DataFrame({'COMPANY': [0, 1, 1], 'YEAR': [2018, 2019, 2020]})
        result = add_baseline_firm_size(data)
        self.assertTrue(result['FIRM_POSTING_COUNT'].isna().all())
        self.assertTrue(result['FIRM_SIZE_BASE_YEAR'].eq(2018).all())
        dated = pd.DataFrame({'COMPANY': [1, 1, 2], 'POSTED': ['2020-01-01', '2021-01-01', '2021-01-01']})
        counts, year = baseline_firm_counts(dated)
        self.assertEqual(year, 2018)
        self.assertTrue(counts.empty)
        with self.assertRaisesRegex(ValueError, 'No valid years'):
            baseline_firm_counts(pd.DataFrame({'COMPANY': [1], 'YEAR': [None]}))

    def test_missing_baseline_cannot_be_resurrected_by_stale_buckets(self):
        data = pd.DataFrame({'COMPANY': [1, 2], 'YEAR': [2025, 2025], 'SP500': [True, False]})
        result = add_baseline_firm_size(data)
        result['FIRM_SIZE_BUCKET'] = ['S&P 500 firms', 'SMEs (<50)']
        self.assertTrue(descriptive_analysis.add_firm_category(result)['FIRM_CATEGORY'].isna().all())

    def test_snapshot_merge_preserves_unknown_size_for_later_entrants(self):
        data = add_baseline_firm_size(pd.DataFrame({'COMPANY': [1, 2], 'YEAR': [2018, 2025]}))
        with tempfile.TemporaryDirectory() as directory:
            snapshot = Path(directory) / 'snapshot.csv'
            pd.DataFrame({'COMPANY': [1, 2]}).to_csv(snapshot, index=False)
            result = prepare_data.merge_sp500_snapshot(data, snapshot, Path(directory) / 'audit.csv')
        self.assertTrue(result['SP500'].all())
        self.assertEqual(result.loc[0, 'FIRM_SIZE_BUCKET'], 'S&P 500 firms')
        self.assertTrue(pd.isna(result.loc[1, 'FIRM_SIZE_BUCKET']))

    def test_high_ai_firm_analysis_excludes_unknown_size(self):
        data = pd.DataFrame({'COMPANY': [1] * 35 + [2] * 35,
                             'AI ROLE': [True] * 70,
                             'FIRM_POSTING_COUNT': pd.array([1] * 35 + [None] * 35, dtype='Int64')})
        filtered, stats = high_ai_firm_analysis.compute_firm_stats(data)
        self.assertEqual(stats.index.tolist(), ['1'])
        self.assertEqual(len(filtered), 35)

    def test_batch_relabel_preserves_labels_text_categories_and_shared_counts(self):
        data = pd.DataFrame({
            'COMPANY': [1, 1, 2, 1, 2, 3], 'YEAR': [2018, 2018, 2018, 2025, 2025, 2025],
            'BODY': ['posting ' + str(i) for i in range(6)],
            'SP500': [False, False, True, False, True, False],
            'REMOTE_KW': [True, False, True, False, True, False],
            'EXPERIENCE_BUCKET': pd.Categorical(['None Listed', '0 years'] * 3),
            'FIRM_POSTING_COUNT': pd.array([3, 3, 2, 3, 2, 1], dtype='Int64'),
            'FIRM_SIZE_BUCKET': pd.Categorical(['SMEs (<50)'] * 6),
        })
        counts, year = baseline_firm_counts(data)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'labeled_v2.parquet'
            data.to_parquet(path, index=False)
            before = path.read_bytes()
            audit = relabel_firm_size.relabel_file(path, counts, year, dry_run=True, batch_size=2)
            self.assertEqual(audit['ROWS'], 6)
            self.assertEqual(audit['FIRM_SIZE_CUTOFF'], 6)
            self.assertEqual(audit['CHANGED_SIZE_BUCKETS'], 6)
            self.assertEqual(path.read_bytes(), before)
            relabel_firm_size.relabel_file(path, counts, year, batch_size=2)
            result = pd.read_parquet(path)
            unchanged = ['COMPANY', 'YEAR', 'BODY', 'SP500', 'REMOTE_KW', 'EXPERIENCE_BUCKET']
            pd.testing.assert_frame_equal(result[unchanged], data[unchanged])
            self.assertEqual(result['FIRM_POSTING_COUNT'].iloc[:5].tolist(), [2, 2, 1, 2, 1])
            self.assertTrue(pd.isna(result.loc[5, 'FIRM_POSTING_COUNT']))
            self.assertEqual(result.loc[2, 'FIRM_SIZE_BUCKET'], 'S&P 500 firms')
            # A later-year target file must receive the same mapping as the full source.
            other = Path(directory) / 'data_v1.parquet'
            data.iloc[3:].reset_index(drop=True).to_parquet(other, index=False)
            relabel_firm_size.relabel_file(other, counts, year, batch_size=2)
            self.assertEqual(pd.read_parquet(other)['FIRM_POSTING_COUNT'].iloc[:2].tolist(), [2, 1])

    def test_failed_rewrite_leaves_original_intact_and_cleans_temporary_file(self):
        data = pd.DataFrame({'COMPANY': [1, 1], 'YEAR': [2018, 2025]})
        counts, year = baseline_firm_counts(data)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'source.parquet'
            data.to_parquet(path, index=False)
            before = path.read_bytes()
            with patch.object(pq.ParquetWriter, 'write_table', side_effect=RuntimeError('write failed')):
                with self.assertRaisesRegex(RuntimeError, 'write failed'):
                    relabel_firm_size.relabel_file(path, counts, year, batch_size=1)
            self.assertEqual(path.read_bytes(), before)
            self.assertEqual(list(Path(directory).iterdir()), [path])

    def test_cli_updates_all_three_files_from_one_baseline_source(self):
        data = pd.DataFrame({'COMPANY': [1, 1, 1, 2], 'YEAR': [2018, 2018, 2025, 2025]})
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for filename in relabel_firm_size.DEFAULT_FILES:
                data.to_parquet(root / filename, index=False)
            with patch.object(sys, 'argv', ['relabel_firm_size.py', '--processed-dir', directory]), \
                    contextlib.redirect_stdout(io.StringIO()):
                relabel_firm_size.main()
            for filename in relabel_firm_size.DEFAULT_FILES:
                result = pd.read_parquet(root / filename)
                self.assertEqual(result['FIRM_POSTING_COUNT'].iloc[:3].tolist(), [2, 2, 2])
                self.assertTrue(pd.isna(result.loc[3, 'FIRM_POSTING_COUNT']))
            self.assertTrue((root / 'firm_size_relabel_audit.csv').exists())


if __name__ == '__main__':
    unittest.main()
