from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np
import pandas as pd
from scipy.stats import sem, ttest_1samp

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from analysis.benefit_prevalence_distributions import (
    summarize_postings, smooth_density, plot_distributions, filter_firms, prevalence_differences,
    mean_difference_test, mean_difference_tests, plot_difference_distributions,
)


class BenefitPrevalenceDistributionTests(unittest.TestCase):
    def test_mean_gap_tests_match_scipy_for_positive_negative_and_zero_means(self):
        for values in [[1, 2, 3, 5], [-1, -2, -3, -5], [-2, -1, 1, 2]]:
            with self.subTest(values=values):
                result = mean_difference_test(values)
                expected = ttest_1samp(values, 0, alternative='two-sided')
                interval = expected.confidence_interval(confidence_level=.95)
                self.assertEqual(result['TEST_STATUS'], 'ok')
                self.assertEqual(result['N_GROUPS'], 4)
                self.assertAlmostEqual(result['MEAN_DIFFERENCE_PP'], np.mean(values))
                self.assertAlmostEqual(result['STANDARD_ERROR_PP'], sem(values))
                self.assertAlmostEqual(result['T_STATISTIC'], expected.statistic)
                self.assertAlmostEqual(result['P_VALUE'], expected.pvalue)
                self.assertAlmostEqual(result['CI95_LOWER_PP'], interval.low)
                self.assertAlmostEqual(result['CI95_UPPER_PP'], interval.high)
                self.assertEqual(result['SIGNIFICANT_05'], expected.pvalue < .05)

    def test_mean_test_unavailable_for_small_samples_or_zero_variance(self):
        for values, status in [([], 'insufficient_observations'), ([2], 'insufficient_observations'),
                               ([0, 0], 'zero_variance'), ([2, 2], 'zero_variance')]:
            with self.subTest(values=values):
                result = mean_difference_test(values)
                self.assertEqual(result['TEST_STATUS'], status)
                self.assertTrue(pd.isna(result['P_VALUE']))
                self.assertTrue(pd.isna(result['SIGNIFICANT_05']))
        with tempfile.TemporaryDirectory() as directory:
            table = pd.DataFrame({'BENEFIT': ['A'], 'DIFFERENCE_PP': [2.]})
            summary = mean_difference_tests(table, ['A', 'B'], 'Firm')
            self.assertEqual(summary.N_GROUPS.tolist(), [1, 0])
            plot_difference_distributions(table, 'Unavailable tests', Path(directory) / 'plot.png', ['A', 'B'],
                                          mean_tests=summary)
            self.assertTrue((Path(directory) / 'plot.png').exists())

    def test_differences_match_within_firm_and_require_both_job_types(self):
        table = pd.DataFrame({'COMPANY': [1, 1, 2, 2, 3], 'BENEFIT': ['A'] * 5,
                              'ROLE_TYPE': ['AI', 'Non-AI', 'AI', 'Non-AI', 'Non-AI'],
                              'VALID_POSTINGS': [2, 100, 1, 0, 5],
                              'PREVALENCE_PERCENT': [80, 20, 10, np.nan, 90]})
        result = prevalence_differences(table, 'COMPANY')
        self.assertEqual(result.COMPANY.tolist(), [1])
        self.assertEqual(result.DIFFERENCE_PP.tolist(), [60])
        self.assertTrue(prevalence_differences(table, 'COMPANY', min_postings=3).empty)
        x, y = smooth_density([-100, 0, 100], bounds=(-100, 100))
        self.assertAlmostEqual(y.sum() * (x[1] - x[0]), 1)

    def test_100_post_threshold_combines_roles_without_double_counting_benefits(self):
        table = pd.DataFrame({'COMPANY': [1, 1, 2, 2] * 2,
                              'ROLE_TYPE': ['AI', 'Non-AI'] * 4,
                              'POSTINGS': [1, 98, 1, 99] * 2,
                              'BENEFIT': ['A'] * 4 + ['B'] * 4})
        result = filter_firms(table)
        self.assertEqual(result.COMPANY.unique().tolist(), [2])
        self.assertTrue(result.TOTAL_FIRM_POSTINGS.eq(100).all())
        self.assertEqual(len(result), 4)

    def test_role_denominators_missing_values_and_2018_firm_eligibility_across_batches(self):
        data = pd.DataFrame({
            'COMPANY': [1, 1, 1, 1, 2, 2, 3, 0],
            'YEAR': [2018, 2025, 2025, 2018, 2018, 2025, 2025, 2018],
            'SOC_MAJOR_GROUP': ['A', 'A', 'A', 'A', 'B', 'B', 'B', 'C'],
            'AI ROLE': [True, False, False, True, None, True, False, False],
            'REMOTE_KW': [True, False, True, None, None, False, True, True],
            'BODY': ['Unused posting text'] * 8,
        })
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'postings.parquet'
            data.to_parquet(path, index=False)
            occupations, firms, audit = summarize_postings(path, benefits=['REMOTE_KW'], batch_size=2)
            by_firm = firms.set_index(['COMPANY', 'ROLE_TYPE'])
            self.assertEqual(set(firms.COMPANY), {1, 2})
            self.assertEqual(by_firm.loc[(1, 'AI'), 'POSTINGS'], 2)
            self.assertEqual(by_firm.loc[(1, 'AI'), 'VALID_POSTINGS'], 1)
            self.assertEqual(by_firm.loc[(1, 'AI'), 'PREVALENCE_PERCENT'], 100)
            self.assertEqual(by_firm.loc[(1, 'Non-AI'), 'PREVALENCE_PERCENT'], 50)
            self.assertEqual(by_firm.loc[(2, 'AI'), 'PREVALENCE_PERCENT'], 0)
            self.assertTrue(pd.isna(by_firm.loc[(2, 'Non-AI'), 'PREVALENCE_PERCENT']))
            self.assertEqual(audit['firms_absent_in_2018_excluded'], 1)
            self.assertEqual(audit['missing_ai_flag_postings_excluded'], 1)
            self.assertEqual(occupations.set_index(['SOC_MAJOR_GROUP', 'ROLE_TYPE']).loc[('B', 'Non-AI'), 'PREVALENCE_PERCENT'], 100)
            plot_distributions(firms, 'COMPANY', 'Test firms', Path(directory) / 'plot.png', paired_only=True)
            self.assertTrue((Path(directory) / 'plot.png').exists())
            self.assertTrue((Path(directory) / 'plot.pdf').exists())

    def test_density_normalizes_and_handles_boundary_only_groups(self):
        x, y = smooth_density([0, 0, 0])
        self.assertTrue(np.isfinite(y).all())
        self.assertAlmostEqual(y.sum() * (x[1] - x[0]), 1)
        self.assertLess(x[y.argmax()], 1)
        _, reflected = smooth_density([100, 100, 100])
        np.testing.assert_allclose(y, reflected[::-1])
        with self.assertRaises(ValueError):
            smooth_density([101])


if __name__ == '__main__':
    unittest.main()
