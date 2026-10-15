import unittest
from pathlib import Path
import sys

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from package_files.firm_category_figure import prepare_mixed_wage_series


class MixedFrequencyFigureTests(unittest.TestCase):
    def test_annual_preperiod_and_quarterly_postperiod_do_not_overlap(self):
        quarters = pd.period_range("2018Q1", "2022Q4", freq="Q")
        positions = {quarter: index for index, quarter in enumerate(quarters)}
        annual = pd.DataFrame(
            {
                "firm_category": ["SMEs"] * 5,
                "period": ["2018", "2019", "2020", "2021", "2018–2022"],
                "ai_role_beta": [0.1] * 5,
                "lower_ci": [0.0] * 5,
                "upper_ci": [0.2] * 5,
                "status": ["ok"] * 5,
            }
        )
        quarterly = pd.DataFrame(
            {
                "FIRM_CATEGORY": ["SMEs"] * 4,
                "QUARTER": ["2020Q4", "2021Q1", "2021Q2", "2022Q1"],
                "ai_role_beta": [0.1] * 4,
                "lower_ci": [0.0] * 4,
                "upper_ci": [0.2] * 4,
                "status": ["ok"] * 4,
            }
        )

        annual_result, quarterly_result = prepare_mixed_wage_series(
            annual,
            quarterly,
            positions,
        )

        self.assertEqual(annual_result["YEAR"].tolist(), [2018, 2019, 2020])
        self.assertEqual(
            quarterly_result["QUARTER"].tolist(),
            [pd.Period("2021Q1"), pd.Period("2021Q2"), pd.Period("2022Q1")],
        )
        self.assertEqual(annual_result["x"].tolist(), [1.5, 5.5, 9.5])
        self.assertEqual(quarterly_result["x"].tolist(), [12, 13, 16])


if __name__ == "__main__":
    unittest.main()
