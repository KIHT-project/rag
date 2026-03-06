# src/test/python/test_data_loader.py
import unittest
import pandas as pd
from unittest.mock import patch
import data_loader


class TestToBool(unittest.TestCase):
    def test_given_none_or_nan_when_to_bool_then_returns_false(self):
        # Given None or NaN values
        # When converting to bool
        # Then result should be False
        self.assertFalse(data_loader.to_bool(None))
        self.assertFalse(data_loader.to_bool(float("nan")))

    def test_given_truthy_strings_when_to_bool_then_returns_true(self):
        # Given strings that should evaluate as True
        truthy = ["x", "X", "yes", "True", "1", "Y"]

        # When converting each to bool
        # Then result should be True
        for v in truthy:
            self.assertTrue(data_loader.to_bool(v))

    def test_given_falsy_strings_when_to_bool_then_returns_false(self):
        # Given strings that should evaluate as False
        falsy = ["", "no", "false", "0", "random"]

        # When converting each to bool
        # Then result should be False
        for v in falsy:
            self.assertFalse(data_loader.to_bool(v))


class TestLoadDataframe(unittest.TestCase):
    @patch("data_loader.pd.read_excel")
    def test_given_valid_excel_with_optional_bool_col_when_load_dataframe_then_bools_normalized(self, mock_read_excel):
        # Given a DataFrame with required cols + one optional boolean-like col
        df = pd.DataFrame([
            {"PMID": "1", "ArticleTitle": "T", "Abstract": "A", "Excluded": "x"},
            {"PMID": "2", "ArticleTitle": "T2", "Abstract": "A2", "Excluded": ""},
        ])
        mock_read_excel.return_value = df

        # When loading the dataframe
        out = data_loader.load_dataframe("fake.xlsx", "Sheet1")

        # Then required cols should be present and 'Excluded' normalized to bool
        self.assertEqual(list(out.columns[:3]), ["PMID", "ArticleTitle", "Abstract"])
        self.assertTrue(out.loc[0, "Excluded"])
        self.assertFalse(out.loc[1, "Excluded"])

    @patch("data_loader.pd.read_excel", side_effect=Exception("boom"))
    def test_given_excel_read_failure_when_load_dataframe_then_raises_runtime_error(self, mock_read_excel):
        # Given read_excel raises an error
        # When loading the dataframe
        # Then RuntimeError should be raised
        with self.assertRaises(RuntimeError) as ctx:
            data_loader.load_dataframe("fake.xlsx", "Sheet1")
        self.assertIn("Error reading Excel", str(ctx.exception))

    @patch("data_loader.pd.read_excel")
    def test_given_empty_dataframe_when_load_dataframe_then_raises_runtime_error(self, mock_read_excel):
        # Given an empty DataFrame
        mock_read_excel.return_value = pd.DataFrame()

        # When loading the dataframe
        # Then RuntimeError should be raised
        with self.assertRaises(RuntimeError) as ctx:
            data_loader.load_dataframe("fake.xlsx", "S")
        self.assertIn("empty", str(ctx.exception))

    @patch("data_loader.pd.read_excel")
    def test_given_missing_required_columns_when_load_dataframe_then_raises_runtime_error(self, mock_read_excel):
        # Given DataFrame missing a required column
        df = pd.DataFrame([{"PMID": "1", "ArticleTitle": "T"}])  # missing Abstract
        mock_read_excel.return_value = df

        # When loading the dataframe
        # Then RuntimeError should mention missing columns
        with self.assertRaises(RuntimeError) as ctx:
            data_loader.load_dataframe("f.xlsx", "S")
        self.assertIn("Missing required columns", str(ctx.exception))


if __name__ == "__main__":
    unittest.main()
