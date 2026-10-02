import contextlib
import io
import os
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

os.environ.setdefault('MPLCONFIGDIR', '/tmp/beyond-salary-matplotlib')
os.environ.setdefault('MPLBACKEND', 'Agg')
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd

from analysis import regression_models as regressions


class M5Tests(unittest.TestCase):
    def test_existing_soc_groups_retain_missing_postings_and_do_not_mutate_input(self):
        data = pd.DataFrame({'SOC_MAJOR_GROUP': ['Computer and Mathematical Occupations', 'Management Occupations', None, '']})
        result = regressions.prepare_occupation_fixed_effects(data)
        self.assertEqual(result['SOC_MAJOR_GROUP'].tolist(), ['Computer and Mathematical Occupations', 'Management Occupations', 'Unknown SOC', 'Unknown SOC'])
        self.assertIsNone(data.loc[2, 'SOC_MAJOR_GROUP'])
        self.assertNotIn('SOC_2_DIGIT', data)
        with self.assertRaisesRegex(ValueError, 'M5 requires SOC_MAJOR_GROUP'):
            regressions.prepare_occupation_fixed_effects(pd.DataFrame({'ONET': ['15-1252.00']}))

    def test_m5_only_cli_returns_before_other_models(self):
        with patch.object(sys, 'argv', ['regression_models.py', '--m5-only']), \
                patch.object(regressions, 'run_m5_only') as m5, \
                patch.object(regressions, 'run_2024_models') as all_models:
            regressions.main()
        m5.assert_called_once()
        all_models.assert_not_called()

    def test_existing_output_directory_is_rejected_before_fitting(self):
        with tempfile.TemporaryDirectory() as directory, \
                patch.object(regressions, 'load_and_prepare_data') as load:
            sentinel = Path(directory) / 'existing.csv'
            sentinel.write_text('preserve me')
            with self.assertRaises(FileExistsError):
                regressions.run_m5_only(SimpleNamespace(output_dir=Path(directory)))
            load.assert_not_called()
            self.assertEqual(sentinel.read_text(), 'preserve me')

    def test_plot_only_combines_saved_models_without_fitting_or_overwriting_tables(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            base_csv = root / 'main.csv'
            base = pd.DataFrame([
                {'panel': 'Main', 'model_idx': model_idx, 'benefit': benefit,
                 'ai_role_coef': model_idx / 10, 'ai_role_se': 0.02, 'model_status': 'ok', 'nobs': 400}
                for model_idx in range(4) for benefit in regressions.benefits4
            ])
            base.to_csv(base_csv, index=False)
            m5_run = root / 'm5' / '20261001T125301Z'
            (m5_run / 'tables').mkdir(parents=True)
            m5_csv = m5_run / 'tables' / 'm5_ai_role_summary.csv'
            pd.DataFrame({
                'Label': regressions.benefits4, 'Coefficient': 0.5, 'Error': 0.03,
                'Converged': True, 'Observations': 400,
            }).to_csv(m5_csv, index=False)
            original_base = base_csv.read_bytes()
            original_m5 = m5_csv.read_bytes()
            output = root / 'five_models.png'
            with patch.object(regressions, 'RESULTS_DIR', root):
                results = regressions.load_five_model_plot_results(base_csv)
                self.assertEqual(len(results), 30)
                self.assertEqual(results['Model Iteration'].nunique(), 5)
                self.assertTrue((results.loc[results['model_idx'].eq(4), 'Coefficient'] == 0.5).all())
                with patch.object(sys, 'argv', [
                    'regression_models.py', '--plot-only', '--results-csv', str(base_csv),
                    '--m5-results', str(m5_run), '--plot-output', str(output),
                ]), patch.object(regressions, 'load_and_prepare_data') as load, \
                        patch.object(regressions, 'run_2024_models') as fit:
                    regressions.main()
                load.assert_not_called()
                fit.assert_not_called()
            self.assertTrue(output.exists())
            self.assertEqual(base_csv.read_bytes(), original_base)
            self.assertEqual(m5_csv.read_bytes(), original_m5)
            base.iloc[:-1].to_csv(base_csv, index=False)
            with self.assertRaisesRegex(ValueError, 'one row per model'):
                regressions.load_five_model_plot_results(base_csv, m5_csv)

    def test_m5_fit_exports_soc_coefficients_and_isolates_all_outputs(self):
        rng = np.random.default_rng(42)
        n = 400
        data = pd.DataFrame({
            'SOC_MAJOR_GROUP': rng.choice(['Management Occupations', 'Computer and Mathematical Occupations'], n),
            'ONET': '11-1021.00',  # M5 must use SOC_MAJOR_GROUP even if ONET differs.
            'AI ROLE': rng.random(n) < 0.35,
            'YEAR': rng.choice([2021, 2022], n),
            'NAICS_2022_3_DIGIT': rng.choice(['541', '621'], n),
            'STATE_NAME': rng.choice(['California', 'Texas'], n),
            'MIN_EDULEVELS_NAME': rng.choice(['No Education Listed', "Bachelor's degree"], n),
            'EXPERIENCE_BUCKET': rng.choice(['None Listed', '1-2 years'], n),
            'LOG_SALARY': rng.normal(11, 0.3, n),
            'SP500': rng.random(n) < 0.25,
        })
        for benefit in regressions.benefits4:
            data[benefit] = rng.random(n) < 0.4
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            legacy = root / 'tables_2026' / 'regression'
            legacy.mkdir(parents=True)
            sentinel = legacy / 'model_panels_ai_role_long.csv'
            sentinel.write_text('existing results')
            output = root / 'm5' / 'test_run'
            with patch.object(regressions, 'RESULTS_DIR', root), \
                    patch.object(regressions, 'TABLES_DIR', legacy), \
                    patch.object(regressions, 'load_and_prepare_data', return_value=data), \
                    contextlib.redirect_stdout(io.StringIO()):
                regressions.run_m5_only(SimpleNamespace(output_dir=output))
                models = regressions.load_cached_models(output / 'model_cache' / 'm5.pkl', expected_panels=1)
                self.assertIsNone(regressions.load_cached_models(output / 'model_cache' / 'm5.pkl'))
            self.assertEqual(sentinel.read_text(), 'existing results')
            self.assertEqual(list(legacy.iterdir()), [sentinel])
            summary = pd.read_csv(output / 'tables' / 'm5_ai_role_summary.csv')
            self.assertEqual(len(summary), 6)
            self.assertTrue((summary['Model Status'] == 'ok').all())
            self.assertTrue(summary['Converged'].all())
            self.assertTrue((summary['Observations'] == n).all())
            coefficients = pd.read_csv(output / 'tables' / 'm5_full_coefficients.csv')
            self.assertEqual(coefficients['benefit'].nunique(), 6)
            for model in models[0]:
                self.assertIn('Management Occupations', model.params)
                self.assertIn('SP500', model.params)
                self.assertIn('LOG_SALARY', model.params)
            self.assertTrue((output / 'figures' / 'm5_ai_role_coefficients.png').exists())
            self.assertEqual(len(list((output / 'tables').glob('*_m5.html'))), 6)


if __name__ == '__main__':
    unittest.main()
