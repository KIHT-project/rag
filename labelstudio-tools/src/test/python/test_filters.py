import unittest
import pandas as pd
from unittest.mock import patch
import filters


class TestApplyFilters(unittest.TestCase):
    def setUp(self):
        # Given a baseline dataframe with all relevant flags
        self.df = pd.DataFrame([
            {"id": 1, "Excluded": True,  "Not related to venous thrombosis": False, "Reporting on risk factors": True},
            {"id": 2, "Excluded": False, "Not related to venous thrombosis": True,  "Reporting on risk factors": True},
            {"id": 3, "Excluded": False, "Not related to venous thrombosis": False, "Reporting on risk factors": False},
            {"id": 4, "Excluded": False, "Not related to venous thrombosis": False, "Reporting on risk factors": True},
        ])

    @patch("filters.FILTER_EXCLUDED", True)
    @patch("filters.FILTER_NOT_RELATED_TO_VTE", False)
    @patch("filters.FILTER_REQUIRE_RISK_FACTORS", False)
    def test_given_excluded_true_when_apply_filters_then_excluded_rows_removed(self):
        # When applying filters with FILTER_EXCLUDED enabled
        result = filters.apply_filters(self.df)

        # Then rows with Excluded=True should be removed
        self.assertNotIn(1, result["id"].tolist())
        self.assertIn(2, result["id"].tolist())

    @patch("filters.FILTER_EXCLUDED", False)
    @patch("filters.FILTER_NOT_RELATED_TO_VTE", True)
    @patch("filters.FILTER_REQUIRE_RISK_FACTORS", False)
    def test_given_not_related_true_when_apply_filters_then_not_related_rows_removed(self):
        # When applying filters with FILTER_NOT_RELATED_TO_VTE enabled
        result = filters.apply_filters(self.df)

        # Then rows with Not related to venous thrombosis=True should be removed
        self.assertNotIn(2, result["id"].tolist())
        self.assertIn(1, result["id"].tolist())

    @patch("filters.FILTER_EXCLUDED", False)
    @patch("filters.FILTER_NOT_RELATED_TO_VTE", False)
    @patch("filters.FILTER_REQUIRE_RISK_FACTORS", True)
    def test_given_require_risk_factors_true_when_apply_filters_then_rows_without_risk_factors_removed(self):
        # When applying filters with FILTER_REQUIRE_RISK_FACTORS enabled
        result = filters.apply_filters(self.df)

        # Then rows with Reporting on risk factors=False should be removed
        self.assertNotIn(3, result["id"].tolist())
        self.assertIn(4, result["id"].tolist())

    @patch("filters.FILTER_EXCLUDED", False)
    @patch("filters.FILTER_NOT_RELATED_TO_VTE", False)
    @patch("filters.FILTER_REQUIRE_RISK_FACTORS", False)
    def test_given_all_filters_false_when_apply_filters_then_dataframe_unchanged(self):
        # When applying filters with all toggles disabled
        result = filters.apply_filters(self.df)

        # Then the dataframe should remain unchanged
        self.assertEqual(len(result), len(self.df))


if __name__ == "__main__":
    unittest.main()
