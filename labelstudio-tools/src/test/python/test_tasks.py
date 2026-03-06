import unittest
import pandas as pd
import tasks


class TestValidateRow(unittest.TestCase):
    def test_valid_pmid(self):
        # Given a row with a valid PMID
        row = pd.Series({"PMID": "12345"})

        # When validating the row
        result = tasks.validate_row(row)

        # Then it should return None (no errors)
        self.assertIsNone(result)

    def test_missing_pmid_nan(self):
        # Given a row with NaN PMID
        row = pd.Series({"PMID": float("nan")})

        # When validating the row
        result = tasks.validate_row(row)

        # Then it should return "PMID missing"
        self.assertEqual(result, "PMID missing")

    def test_missing_pmid_blank(self):
        # Given a row with blank string PMID
        row = pd.Series({"PMID": "   "})

        # When validating the row
        result = tasks.validate_row(row)

        # Then it should return "PMID missing"
        self.assertEqual(result, "PMID missing")


class TestTransformToTasks(unittest.TestCase):
    def test_transforms_valid_rows(self):
        # Given a dataframe with valid rows
        df = pd.DataFrame([
            {"PMID": "111", "ArticleTitle": "Title 1", "Abstract": "Abs 1"},
            {"PMID": "222", "ArticleTitle": "Title 2", "Abstract": "Abs 2"},
        ])

        # When transforming to tasks
        tasks_out = tasks.transform_to_tasks(df)

        # Then both rows should be transformed correctly
        self.assertEqual(len(tasks_out), 2)
        self.assertEqual(tasks_out[0]["data"]["pmid"], "111")
        self.assertEqual(tasks_out[1]["data"]["title"], "Title 2")

    def test_skips_invalid_rows(self):
        # Given a dataframe with one valid row and one invalid row (missing PMID)
        df = pd.DataFrame([
            {"PMID": "333", "ArticleTitle": "Title ok", "Abstract": "Abs ok"},
            {"PMID": "", "ArticleTitle": "Bad title", "Abstract": "Bad abs"},
        ])

        # When transforming to tasks
        tasks_out = tasks.transform_to_tasks(df)

        # Then only the valid row should be included
        self.assertEqual(len(tasks_out), 1)
        self.assertEqual(tasks_out[0]["data"]["pmid"], "333")

    def test_raises_if_all_invalid(self):
        # Given a dataframe where all rows are invalid
        df = pd.DataFrame([
            {"PMID": None, "ArticleTitle": "Bad", "Abstract": "Bad"},
            {"PMID": "   ", "ArticleTitle": "Bad2", "Abstract": "Bad2"},
        ])

        # When transforming to tasks
        # Then it should raise RuntimeError
        with self.assertRaises(RuntimeError):
            tasks.transform_to_tasks(df)


if __name__ == "__main__":
    unittest.main()
