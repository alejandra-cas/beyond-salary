import os
from pathlib import Path
from types import SimpleNamespace
import sys
import tempfile
import unittest
from unittest.mock import patch

os.environ.setdefault('MPLCONFIGDIR', '/tmp/beyond-salary-matplotlib')
os.environ.setdefault('MPLBACKEND', 'Agg')
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))

import pandas as pd
from analysis import regression_models as regression
from package_files.llm_firm_sample import read_firm_labels, select_llm_firms


class LLMFirmRegressionTests(unittest.TestCase):
    def test_explicit_labels_override_proxy_and_exclude_unknown_and_missing_2018(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'labels.csv'
            pd.DataFrame({'COMPANY': ['1.0', '2', '3', '4'],
                          'llm_label': ['sme', 'large', 'unknown', 'large']}).to_csv(path, index=False)
            labels = read_firm_labels(path)
        data = pd.DataFrame({'COMPANY': [1, 1, 2, 3, 4, 5],
                             'FIRM_POSTING_COUNT': [100, 100, 1, 1, None, 1],
                             'YEAR': [2018, 2025, 2025, 2018, 2025, 2018],
                             'SP500': [False, False, True, False, False, False]})
        selected = select_llm_firms(data, labels)
        self.assertEqual(selected.COMPANY.tolist(), [1, 1, 2])
        self.assertEqual(regression.add_firm_category(selected).FIRM_CATEGORY.tolist(),
                         ['SMEs', 'SMEs', 'Large firms'])
        self.assertEqual(regression.add_firm_category(data).FIRM_CATEGORY.iloc[0], 'Large firms')

    def test_conflicting_labels_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'labels.csv'
            pd.DataFrame({'COMPANY': ['1', '1.0'], 'llm_label': ['sme', 'large']}).to_csv(path, index=False)
            with self.assertRaisesRegex(ValueError, 'Conflicting'):
                read_firm_labels(path)

    def test_filtered_parquet_read_and_separate_pooled_outputs(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            labels_path = root / 'labels.csv'
            pd.DataFrame({'COMPANY': [1, 2, 3], 'llm_label': ['sme', 'large', 'unknown']}).to_csv(labels_path, index=False)
            data = pd.DataFrame({
                'COMPANY': [1, 2, 3], 'FIRM_POSTING_COUNT': [100, 1, 1],
                'YEAR': [2018, 2025, 2025], 'STATE_NAME': ['CA'] * 3,
                'NAICS_2022_2_NAME': ['Industry'] * 3,
                'NAICS_2022_3_DIGIT': ['123'] * 3,
                'MIN_EDULEVELS_NAME': ['No Education Listed'] * 3,
                'EXPERIENCE_BUCKET': ['None Listed'] * 3,
                'LOG_SALARY': [10.] * 3, 'AI ROLE': [False, True, False],
                'SP500': [False] * 3, 'BODY': ['Unused text'] * 3,
                **{benefit: [True, False, True] for benefit in regression.benefits4},
            })
            data.to_parquet(root / 'labeled_v2.parquet', index=False)
            args = SimpleNamespace(llm_only=True, llm_labels=labels_path, firm_output_dir=root / 'isolated')
            with patch.object(regression, 'PROCESSED_DIR', root):
                selected, tables, figures = regression.prepare_firm_run(args)
            self.assertEqual(selected.COMPANY.tolist(), [1, 2])
            self.assertNotIn('BODY', selected)
            self.assertTrue((tables / 'llm_firm_sample_audit.csv').exists())
            calls = []

            def fit(subset, category, benefit, **kwargs):
                calls.append((category, subset.COMPANY.tolist()))
                return SimpleNamespace(params=pd.Series({'AI ROLE': 0.2}),
                                       bse=pd.Series({'AI ROLE': 0.1}), pvalues=pd.Series({'AI ROLE': 0.04}),
                                       nobs=len(subset), prsquared=0.1, converged=True)

            with patch.object(regression, 'fit_firm_category_logit_model', side_effect=fit):
                results = regression.run_firm_category_logit_models(selected, output_dir=tables)
            self.assertEqual(len(results), 12)
            self.assertTrue(all(ids == ([1] if category == 'SMEs' else [2]) for category, ids in calls))
            regression.plot_firm_category_logit_results(results, output_dir=figures, title='LLM-labeled firms')
            self.assertTrue((tables / 'firm_category_ai_role_logit_results.csv').exists())
            self.assertTrue((figures / 'firm_category_ai_role_logit_coefficients.png').exists())


if __name__ == '__main__':
    unittest.main()
