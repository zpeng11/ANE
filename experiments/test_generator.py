#!/usr/bin/env python3
"""Offline generator determinism checks; never calls ANE."""
from pathlib import Path
import subprocess
import sys
import unittest

ROOT = Path(__file__).resolve().parent

import verify_cases  # noqa: E402  (local module, path is the test directory)


class GeneratorDeterminismTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.ok, cls.report = verify_cases.verify()

    def test_generator_reproduces_committed_cases(self):
        self.assertTrue(self.ok, self.report)
        commit = self.report["against_commit"]
        self.assertEqual(commit["expected_files"], commit["generated_files"])
        self.assertEqual(commit["missing"], [])
        self.assertEqual(commit["extra"], [])
        self.assertEqual(commit["differing"], [])

    def test_two_runs_are_byte_identical(self):
        repeat = self.report["run1_vs_run2"]
        self.assertEqual(repeat["missing"], [])
        self.assertEqual(repeat["extra"], [])
        self.assertEqual(repeat["differing"], [])

    def test_expected_index_is_76_unique_cases(self):
        manifest = self.report["manifest"]
        self.assertEqual(manifest["declared"], 76)
        self.assertEqual(manifest["unique"], 76)
        self.assertEqual(manifest["duplicates"], [])
        self.assertEqual(manifest["missing_mil"], [])
        self.assertEqual(manifest["missing_meta"], [])
        self.assertEqual(manifest["unindexed_mil"], [])
        self.assertTrue(manifest["sorted"])

    def test_weighted_cases_is_the_delegated_module(self):
        self.assertTrue((ROOT / "weighted_cases.py").is_file())
        self.assertFalse((ROOT / "mil_gen2.py").exists())

    def test_mil_gen_is_the_only_entry_point(self):
        result = subprocess.run([sys.executable, str(ROOT / "mil_gen.py"), "--help"],
                                cwd=ROOT, capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("--output", result.stdout)

    def test_generator_has_no_numpy_dependency(self):
        code = ("import sys; sys.path.insert(0, %r); import mil_gen; "
                "print('numpy' in sys.modules)" % str(ROOT))
        result = subprocess.run([sys.executable, "-c", code], cwd=ROOT,
                                capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(result.stdout.strip(), "False")


if __name__ == "__main__":
    unittest.main(verbosity=2)
