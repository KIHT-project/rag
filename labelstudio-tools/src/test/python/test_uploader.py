import unittest
from unittest.mock import MagicMock, patch
import uploader


class TestImportWithRetries(unittest.TestCase):
    def test_import_succeeds_first_try(self):
        # Given a LabelStudio client that succeeds immediately
        ls = MagicMock()
        ls.projects.import_tasks.return_value = None

        # When calling _import_with_retries
        uploader._import_with_retries(ls, 3, [{"data": "x"}], retries=3, backoff=1)

        # Then it should call import_tasks once and succeed
        ls.projects.import_tasks.assert_called_once()

    def test_import_retries_then_succeeds(self):
        # Given a client that fails twice then succeeds
        ls = MagicMock()
        ls.projects.import_tasks.side_effect = [Exception("fail1"), Exception("fail2"), None]

        # When calling _import_with_retries
        uploader._import_with_retries(ls, 3, [{"data": "x"}], retries=3, backoff=1)

        # Then it should retry 3 times total
        self.assertEqual(ls.projects.import_tasks.call_count, 3)

    def test_import_fails_after_retries(self):
        # Given a client that always fails
        ls = MagicMock()
        ls.projects.import_tasks.side_effect = Exception("always fails")

        # When calling _import_with_retries with retries=2
        # Then it should raise RuntimeError after exceeding retries
        with self.assertRaises(RuntimeError):
            uploader._import_with_retries(ls, 3, [{"data": "x"}], retries=2, backoff=1)


class TestUploadTasks(unittest.TestCase):
    def test_no_api_key(self):
        # Given no API key
        with patch("uploader.API_KEY", ""):
            # When upload_tasks is called
            # Then it should raise ValueError
            with self.assertRaises(ValueError):
                uploader.upload_tasks([{"data": "x"}])

    def test_authentication_failure(self):
        # Given an invalid API key and a client that fails on projects.list
        with patch("uploader.API_KEY", "fake"), \
             patch("uploader.LabelStudio") as mock_ls_class:
            mock_ls = mock_ls_class.return_value
            mock_ls.projects.list.side_effect = Exception("auth failed")

            # When upload_tasks is called
            # Then it should raise RuntimeError
            with self.assertRaises(RuntimeError):
                uploader.upload_tasks([{"data": "x"}])

    def test_project_not_found(self):
        # Given an API key but project ID is not in visible projects
        with patch("uploader.API_KEY", "fake"), \
             patch("uploader.LabelStudio") as mock_ls_class:
            mock_ls = mock_ls_class.return_value
            mock_ls.projects.list.return_value = [MagicMock(id=999)]

            # When upload_tasks is called
            # Then it should raise PermissionError
            with self.assertRaises(PermissionError):
                uploader.upload_tasks([{"data": "x"}])

    def test_successful_upload(self):
        # Given an API key, project 3 visible, and a mock import
        with patch("uploader.API_KEY", "fake"), \
             patch("uploader.LabelStudio") as mock_ls_class, \
             patch("uploader._import_with_retries") as mock_import:
            mock_ls = mock_ls_class.return_value
            mock_ls.projects.list.return_value = [MagicMock(id=3)]

            # When upload_tasks is called with 2 tasks
            uploader.upload_tasks([{"data": "x"}, {"data": "y"}])

            # Then _import_with_retries should be called
            mock_import.assert_called()


if __name__ == "__main__":
    unittest.main()
