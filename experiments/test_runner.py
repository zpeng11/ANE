#!/usr/bin/env python3
"""Offline runner provenance checks; no build, no probe, no ANE."""
import csv
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import types
import unittest

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
import run_validation  # noqa: E402


class RunnerProvenanceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp = tempfile.TemporaryDirectory(prefix="ane-runner-provenance-")
        cls.output = Path(cls.temp.name) / "evidence"
        cls.result = subprocess.run(
            [sys.executable, "run_validation.py", "--provenance-only",
             "--output", str(cls.output)],
            cwd=ROOT, capture_output=True, text=True)
        cls.metadata_text = (cls.output / "metadata.json").read_text()
        cls.metadata = json.loads(cls.metadata_text)
        cls.summary = json.loads((cls.output / "summary.json").read_text())

    @classmethod
    def tearDownClass(cls):
        cls.temp.cleanup()

    def test_provenance_only_succeeds_without_hardware(self):
        self.assertEqual(self.result.returncode, 0,
                         self.result.stdout + self.result.stderr)
        self.assertEqual(self.summary, [])
        self.assertEqual(self.metadata["binary_sha256"], {})
        self.assertEqual(self.metadata["binary_hashes_recorded_at"],
                         "not built (provenance-only)")

    def test_recorded_paths_are_portable(self):
        self.assertNotIn("/Users/", self.metadata_text)
        self.assertFalse(self.metadata["output"].startswith("/"))
        for key in self.metadata["source_sha256"]:
            self.assertFalse(key.startswith("/"), key)
        for key in self.metadata["case_sha256"]:
            self.assertFalse(key.startswith("/"), key)

    def test_git_commit_and_dirty_state_are_recorded(self):
        expected = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT,
                                  capture_output=True, text=True)
        self.assertEqual(self.metadata["git"]["commit"], expected.stdout.strip())
        self.assertIsInstance(self.metadata["git"]["dirty"], bool)
        self.assertIsInstance(self.metadata["git"]["status"], list)

    def test_clang_version_is_recorded(self):
        clang = self.metadata["clang"]
        self.assertIn(clang["command"][0], {"xcrun", "clang"})
        if clang["exit_code"] == 0:
            self.assertIn("clang version", clang["stdout"])

    def test_source_and_case_hashes_are_recorded(self):
        self.assertIn("Makefile", self.metadata["source_sha256"])
        self.assertIn("mil_gen.py", self.metadata["source_sha256"])
        self.assertIn("weighted_cases.py", self.metadata["source_sha256"])
        self.assertIn("run_survey.sh", self.metadata["source_sha256"])
        self.assertIn("opcases/_index.txt", self.metadata["case_sha256"])
        self.assertEqual(len(self.metadata["case_sha256"]), 163)

    def test_manifest_reports_76_unique_cases(self):
        manifest = self.metadata["manifest"]
        self.assertEqual(manifest["declared"], 76)
        self.assertEqual(manifest["unique"], 76)
        self.assertEqual(manifest["missing_mil"], [])
        self.assertEqual(manifest["missing_meta"], [])
        self.assertEqual(manifest["extra_mil"], [])
        self.assertEqual(manifest["duplicate_names"], [])
        self.assertEqual(manifest["meta_errors"], [])
        self.assertTrue(manifest["ok"])
        self.assertEqual(manifest["errors"], [])
        self.assertEqual(len(manifest["names"]), 76)

    def test_check_manifest_cli_accepts_committed_cases(self):
        result = subprocess.run(
            [sys.executable, "run_validation.py", "--check-manifest"],
            cwd=ROOT, capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertIn("manifest: PASS (76 unique cases)", result.stdout)

    def test_check_manifest_cli_rejects_missing_mil_and_duplicate_index(self):
        with tempfile.TemporaryDirectory(prefix="ane-manifest-bad-") as temp:
            cases = Path(temp) / "opcases"
            cases.mkdir()
            (cases / "_index.txt").write_text("a\nb\nb\n")
            (cases / "a.mil").write_text("a\n")
            (cases / "a.meta").write_text("a,4096,4096,-\n")
            (cases / "b.meta").write_text("b,4096,4096,-\n")
            result = subprocess.run(
                [sys.executable, "run_validation.py", "--check-manifest",
                 "--cases", str(cases)],
                cwd=ROOT, capture_output=True, text=True)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("duplicate cases: b", result.stderr)
        self.assertIn("missing MIL: b", result.stderr)

    def test_check_manifest_cli_rejects_extra_mil_and_bad_meta(self):
        with tempfile.TemporaryDirectory(prefix="ane-manifest-extra-") as temp:
            cases = Path(temp) / "opcases"
            cases.mkdir()
            (cases / "_index.txt").write_text("a\n")
            (cases / "a.mil").write_text("a\n")
            (cases / "a.meta").write_text("a,4096,4096,-\n")
            (cases / "extra.mil").write_text("extra\n")
            result = subprocess.run(
                [sys.executable, "run_validation.py", "--check-manifest",
                 "--cases", str(cases)],
                cwd=ROOT, capture_output=True, text=True)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("unindexed MIL: extra", result.stderr)

            (cases / "extra.mil").unlink()
            (cases / "a.meta").write_text("a,4096\n")
            result = subprocess.run(
                [sys.executable, "run_validation.py", "--check-manifest",
                 "--cases", str(cases)],
                cwd=ROOT, capture_output=True, text=True)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("expected 4 meta fields", result.stderr)

    def test_hardware_mode_refuses_bad_manifest_before_build(self):
        with tempfile.TemporaryDirectory(prefix="ane-manifest-gate-") as temp:
            cases = Path(temp) / "opcases"
            cases.mkdir()
            (cases / "_index.txt").write_text("a\nb\n")
            (cases / "a.mil").write_text("a\n")
            (cases / "a.meta").write_text("a,4096,4096,-\n")
            output = Path(temp) / "evidence"
            result = subprocess.run(
                [sys.executable, "run_validation.py", "--cases", str(cases),
                 "--output", str(output)],
                cwd=ROOT, capture_output=True, text=True)
            metadata = json.loads((output / "metadata.json").read_text())
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("refusing to run hardware", result.stderr)
        self.assertEqual(metadata["status"], "manifest-failed")
        self.assertFalse(metadata["manifest"]["ok"])
        self.assertEqual(metadata["binary_sha256"], {})
        self.assertFalse((output / "build.log").exists())

    def test_survey_row_check_requires_exact_declared_set(self):
        with tempfile.TemporaryDirectory(prefix="ane-row-check-") as temp:
            table = Path(temp) / "survey_results.csv"
            table.write_text("op,status,compile,load,eval,numeric,exit\n"
                             "a,OK,OK,OK,OK,NA,0\n"
                             "b,OK,OK,OK,OK,NA,0\n")
            self.assertTrue(run_validation.survey_row_check(table, ["a", "b"])["ok"])

            table.write_text("op,status,compile,load,eval,numeric,exit\n"
                             "a,OK,OK,OK,OK,NA,0\n")
            missing = run_validation.survey_row_check(table, ["a", "b"])
            self.assertFalse(missing["ok"])
            self.assertEqual(missing["missing_rows"], ["b"])

            table.write_text("op,status,compile,load,eval,numeric,exit\n"
                             "a,OK,OK,OK,OK,NA,0\n"
                             "b,OK,OK,OK,OK,NA,0\n"
                             "c,OK,OK,OK,OK,NA,0\n")
            extra = run_validation.survey_row_check(table, ["a", "b"])
            self.assertFalse(extra["ok"])
            self.assertEqual(extra["extra_rows"], ["c"])

            table.write_text("op,status,compile,load,eval,numeric,exit\n"
                             "a,OK,OK,OK,OK,NA,0\n"
                             "a,OK,OK,OK,OK,NA,0\n"
                             "b,OK,OK,OK,OK,NA,0\n")
            duplicate = run_validation.survey_row_check(table, ["a", "b"])
            self.assertFalse(duplicate["ok"])
            self.assertEqual(duplicate["duplicate_rows"], ["a"])

    def test_no_performance_reproducibility_claim(self):
        self.assertIn("not byte-identical",
                      self.metadata["performance_reproducibility"])
        self.assertEqual(self.metadata["survey_classification"]["status"], "not_run")

    def test_survey_classification_separates_numeric_from_acceptance(self):
        with tempfile.TemporaryDirectory(prefix="ane-survey-classify-") as temp:
            table = Path(temp) / "survey_results.csv"
            table.write_text(
                "op,status,compile,load,eval,numeric,exit\n"
                "a,OK,OK,OK,OK,PASS,0\n"
                "b,OK,OK,OK,OK,ACCEPTANCE_ONLY,0\n"
                "c,COMPILE_FAIL,FAIL,NA,NA,NA,10\n"
                "d,NUMERIC_FAIL,OK,OK,OK,FAIL,13\n")
            counts = run_validation.classify_survey_csv(table)
        self.assertEqual(counts["total"], 4)
        self.assertEqual(counts["numeric_verified"], 1)
        self.assertEqual(counts["acceptance_only"], 1)
        self.assertEqual(counts["compile_rejected"], 1)
        self.assertEqual(counts["numeric_failures"], 1)

    def test_hardware_commands_use_build_and_current_names(self):
        args = types.SimpleNamespace(bench=True)
        commands = run_validation.build_commands(args, ROOT / "results/example")
        flat = [part for _, command, _ in commands for part in command]
        for needed in ("build/multi_input", "build/multi_io",
                       "build/static_shape_compile", "build/surface_capacity",
                       "build/dispatch_latency", "build/mapping_overhead"):
            self.assertIn(needed, flat)
        for old in ("multi_input_test2", "multi_io_gen", "shape_test",
                    "shape_test2", "dispatch_floor", "dispatch_floor2"):
            self.assertFalse(any(old in part for part in flat), old)


if __name__ == "__main__":
    unittest.main(verbosity=2)
