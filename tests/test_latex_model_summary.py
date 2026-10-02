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

import pandas as pd
from analysis import regression_models as regressions


def example_models():
    panels = []
    for i in range(5):
        terms = {'AI ROLE': .111 * (i + 1), 'const': -1.234}
        if i >= 1:
            terms['1-2 years'] = .222
            terms["Bachelor's degree"] = .333
        if i >= 2:
            terms['LOG_SALARY'] = .444
        if i >= 3:
            terms['SP500'] = .555
        model = SimpleNamespace(params=pd.Series(terms), bse=pd.Series(.012, index=terms),
                                pvalues=pd.Series(.001, index=terms), nobs=3559853,
                                prsquared=.123, converged=True)
        panels.append([model for _ in regressions.benefits4])
    return panels


class LatexSummaryTests(unittest.TestCase):
    def test_five_models_keep_original_layout_and_use_actual_values(self):
        text = regressions.html_to_latex_table_dynamic(example_models())
        self.assertTrue(text.startswith('\\begin{table}[!htbp]\n\\centering\n\\tiny\n'))
        self.assertEqual(text.count(r'\multicolumn{5}'), 6)
        self.assertIn(r'\cmidrule(lr){2-6}', text)
        self.assertIn(r'\cmidrule(lr){27-31}', text)
        self.assertEqual(text.count('(5)'), 6)
        self.assertIn(r'\quad 1-2 years', text)
        self.assertIn(r'\caption{Regression Results - All Benefits}', text)
        self.assertIn(r'\label{tab:results_all_benefits}', text)
        self.assertIn(r'\begin{tablenotes}', text)
        self.assertIn('0.56$^{***}$', text)  # M5 AI coefficient, original two-decimal style.
        self.assertIn('0.555$^{***}$', text)  # S&P 500 control uses fitted values.
        self.assertIn('3559853', text)
        self.assertNotIn('20000', text)
        self.assertNotIn('M2: M1', text)  # Numeric column headings, as in the original.
        for line in text.splitlines():
            if line.endswith(r' \\'):
                self.assertEqual(len(line.split(' & ')), 7 if '\\multicolumn' in line else 31)
        soc = next(line for line in text.splitlines() if line.startswith(r'\quad SOC2'))
        self.assertEqual(soc.split(' & ')[1:], ['No', 'No', 'No', 'No', 'Yes'] * 5 + ['No', 'No', 'No', 'No', r'Yes \\'])

    def test_latex_only_writes_one_file_and_never_fits_models(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / 'summary.tex'
            sentinel = Path(directory) / 'existing.html'
            sentinel.write_text('preserve existing table')
            with patch.object(sys, 'argv', ['regression_models.py', '--latex-only', '--latex-output', str(output)]), \
                    patch.object(regressions, 'load_latex_summary_models', return_value=example_models()), \
                    patch.object(regressions, 'load_and_prepare_data') as load, \
                    patch.object(regressions, 'run_2024_models') as fit, \
                    patch.object(regressions, 'generate_wide_table') as html, \
                    contextlib.redirect_stdout(io.StringIO()):
                regressions.main()
            load.assert_not_called()
            fit.assert_not_called()
            html.assert_not_called()
            self.assertEqual(sentinel.read_text(), 'preserve existing table')
            self.assertEqual(len(list(Path(directory).iterdir())), 2)
            self.assertIn('(5)', output.read_text())


if __name__ == '__main__':
    unittest.main()
