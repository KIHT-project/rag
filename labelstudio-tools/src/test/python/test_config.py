import os
import unittest
from unittest.mock import patch
from importlib import reload
import config


class TestEnvParsing(unittest.TestCase):
    def test_given_env_vars_for_bool_and_int_when_reload_config_then_parsed_correctly(self):
        # Given environment variables for FILTER_EXCLUDED and ROWS
        with patch.dict(os.environ, {"FILTER_EXCLUDED": "TrUe", "ROWS": "25"}, clear=False):
            # When reloading config
            reload(config)

            # Then values should be parsed correctly
            self.assertTrue(config.FILTER_EXCLUDED)
            self.assertEqual(config.ROWS, 25)

    def test_given_rows_zero_when_reload_config_then_rows_is_none(self):
        # Given ROWS set to "0"
        with patch.dict(os.environ, {"ROWS": "0"}, clear=False):
            # When reloading config
            reload(config)

            # Then ROWS should resolve to None
            self.assertIsNone(config.ROWS)

    def test_given_url_api_key_and_project_id_when_reload_config_then_values_set_correctly(self):
        # Given environment variables for URL, API key and Project ID
        with patch.dict(os.environ, {
            "LABEL_STUDIO_URL": "https://example.org/",
            "LABEL_STUDIO_API_KEY": "secret123",
            "PROJECT_ID": "42"
        }, clear=False):
            # When reloading config
            reload(config)

            # Then URL, API_KEY and PROJECT_ID should be correctly parsed
            self.assertEqual(config.LABEL_STUDIO_URL, "https://example.org")
            self.assertEqual(config.API_KEY, "secret123")
            self.assertEqual(config.PROJECT_ID, 42)

    def test_given_excel_file_and_sheet_name_when_reload_config_then_values_override_defaults(self):
        # Given environment variables for Excel file and sheet name
        with patch.dict(os.environ, {
            "EXCEL_FILE": "/tmp/foo.xlsx",
            "SHEET_NAME": "CustomSheet"
        }, clear=False):
            # When reloading config
            reload(config)

            # Then overrides should apply
            self.assertEqual(config.EXCEL_FILE, "/tmp/foo.xlsx")
            self.assertEqual(config.SHEET_NAME, "CustomSheet")

    def test_given_filter_flags_and_sample_flag_when_reload_config_then_values_parsed_correctly(self):
        # Given environment variables for filter toggles and sample flag
        with patch.dict(os.environ, {
            "FILTER_EXCLUDED": "yes",
            "FILTER_NOT_RELATED_TO_VTE": "1",
            "FILTER_REQUIRE_RISK_FACTORS": "on",
            "SAMPLE_AFTER_FILTER": "false"
        }, clear=False):
            # When reloading config
            reload(config)

            # Then filter toggles and SAMPLE_AFTER_FILTER should be parsed correctly
            self.assertTrue(config.FILTER_EXCLUDED)
            self.assertTrue(config.FILTER_NOT_RELATED_TO_VTE)
            self.assertTrue(config.FILTER_REQUIRE_RISK_FACTORS)
            self.assertFalse(config.SAMPLE_AFTER_FILTER)

    def test_given_import_retries_and_backoff_when_reload_config_then_values_parsed_as_numbers(self):
        # Given environment variables for import retries and backoff
        with patch.dict(os.environ, {
            "IMPORT_RETRIES": "7",
            "IMPORT_BACKOFF": "2.5"
        }, clear=False):
            # When reloading config
            reload(config)

            # Then values should be converted to int/float
            self.assertEqual(config.IMPORT_RETRIES, 7)
            self.assertEqual(config.IMPORT_BACKOFF, 2.5)

    def test_given_csv_overrides_when_reload_config_then_lists_parsed_correctly(self):
        # Given CSV environment variables for required, optional bool, and category cols
        with patch.dict(os.environ, {
            "REQUIRED_COLS_CSV": "ID,Title,Abs",
            "OPTIONAL_BOOL_COLS_CSV": "Flag1,Flag2",
            "CATEGORY_COLS_CSV": "CatA,CatB"
        }, clear=False):
            # When reloading config
            reload(config)

            # Then the lists should be parsed correctly
            self.assertEqual(config.REQUIRED_COLS, ["ID", "Title", "Abs"])
            self.assertEqual(config.OPTIONAL_BOOL_COLS, ["Flag1", "Flag2"])
            self.assertEqual(config.CATEGORY_COLS, ["CatA", "CatB"])


if __name__ == "__main__":
    unittest.main()
