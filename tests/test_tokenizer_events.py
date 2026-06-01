import unittest

from src.tokenizer.core.events import JobEvent


class TokenizerEventTests(unittest.TestCase):
    def test_valid_progress_event(self):
        event = JobEvent(kind="progress", total_windows=100, completed_windows=25, eta_seconds=12.5)
        event.validate()

    def test_invalid_progress_event_missing_counts(self):
        event = JobEvent(kind="progress")
        with self.assertRaises(ValueError):
            event.validate()

    def test_invalid_progress_event_overrun(self):
        event = JobEvent(kind="progress", total_windows=10, completed_windows=11)
        with self.assertRaises(ValueError):
            event.validate()


if __name__ == "__main__":
    unittest.main()
