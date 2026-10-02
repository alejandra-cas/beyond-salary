import os
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

os.environ.setdefault('MPLCONFIGDIR', '/tmp/beyond-salary-matplotlib')
os.environ.setdefault('MPLBACKEND', 'Agg')
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib.pyplot as plt
import pandas as pd
from scripts import generate_combined_figures as figures


class Figure3Tests(unittest.TestCase):
    def test_panel_plots_five_models_for_all_six_benefits_with_cumulative_legend(self):
        names = ['M1: Baseline', 'M2: +Indiv Controls', 'M3: +Salary',
                 'M4: +S&P 500', 'M5: M4 + 2-digit SOC FE']
        results = pd.DataFrame([
            {'benefit': benefit, 'model_label': name, 'ai_role_coef': i / 10, 'ai_role_se': .02}
            for i, name in enumerate(names) for benefit in figures.benefits4
        ])
        fig, ax = plt.subplots()
        try:
            figures.plot_model_coefficients(ax, results)
            self.assertEqual(len(ax.containers), 30)
            self.assertEqual([text.get_text() for text in ax.get_legend().get_texts()], [
                'M1: Baseline (Year+NAICS3)', 'M2: M1 + Individ. + State',
                'M3: M2 + Salary', 'M4: M3 + S&P 500', 'M5: M4 + SOC2',
            ])
            self.assertEqual(len(ax.get_xticks()), 6)
        finally:
            plt.close(fig)

    def test_figure3_cli_does_not_regenerate_figure4(self):
        with patch.object(sys, 'argv', ['generate_combined_figures.py', '--figure', '3',
                                       '--output-dir', 'results/figures_2026_paper']), \
                patch.object(figures, 'generate_figure3') as figure3, \
                patch.object(figures, 'generate_figure4') as figure4:
            figures.main()
        figure3.assert_called_once_with(Path('results/figures_2026_paper'), figures.MODEL_PANELS_CSV, None)
        figure4.assert_not_called()


if __name__ == '__main__':
    unittest.main()
