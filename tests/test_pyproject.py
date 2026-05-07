import re
import unittest
from pathlib import Path


class PyprojectTests(unittest.TestCase):
    def test_pyannote_is_optional_dependency(self):
        content = Path("pyproject.toml").read_text(encoding="utf-8")
        project_dependencies = re.search(r"dependencies = \[(.*?)\]\n\n", content, re.S)
        optional_dependencies = re.search(r"\[project.optional-dependencies\](.*?)(?:\n\[|\Z)", content, re.S)

        self.assertIsNotNone(project_dependencies)
        self.assertIsNotNone(optional_dependencies)
        self.assertNotIn("pyannote-audio", project_dependencies.group(1))
        self.assertIn("diarize = [", optional_dependencies.group(1))
        self.assertIn("full = [", optional_dependencies.group(1))
        self.assertIn('"pyannote-audio"', optional_dependencies.group(1))
