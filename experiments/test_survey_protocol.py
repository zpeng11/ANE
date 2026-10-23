#!/usr/bin/env python3
"""Offline checks for survey status/exit-code handling; never calls ANE."""
import csv
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).resolve().parent


class SurveyProtocolTests(unittest.TestCase):
    def make_tree(self, temp, probe_exit=0, status="OK", index="test\n"):
        root = Path(temp)
        shutil.copy2(ROOT / "run_survey.sh", root / "run_survey.sh")
        (root / "opcases").mkdir()
        (root / "opcases/test.mil").write_text("unused by offline probe\n")
        (root / "opcases/test.meta").write_text("test,4096,4096,-\n")
        (root / "opcases/_index.txt").write_text(index)
        probe = root / "op_probe"
        probe.write_text(
            "#!/bin/sh\n"
            "printf '%s\\n' 'multiline diagnostic' 'NSUnderlyingError = details' >&2\n"
            f"printf '%s\\n' 'CASE test STATUS {status} COMPILE=OK LOAD=OK EVAL=OK NUMERIC=NA EXIT={probe_exit}'\n"
            f"exit {probe_exit}\n"
        )
        probe.chmod(0o755)
        return root

    def run_survey(self, root, *arguments, probe=True):
        env = dict(os.environ)
        if probe:
            env["OP_PROBE"] = "./op_probe"
        else:
            env.pop("OP_PROBE", None)
        return subprocess.run(["bash", "run_survey.sh", *arguments], cwd=root,
                              env=env, capture_output=True, text=True)

    def read_rows(self, path):
        with Path(path).open() as table:
            return list(csv.DictReader(table))

    def default_runs(self, root):
        return sorted((Path(root) / "results").glob("run-*"))

    def run_case(self, status, probe_exit, expected_exit, expected_status=None, allow=False):
        with tempfile.TemporaryDirectory(prefix="ane-survey-protocol-") as temp:
            root = self.make_tree(temp, probe_exit, status)
            command = ["--csv", "table.csv", "--results", "results"]
            if allow:
                command.append("--allow-unverified")
            result = self.run_survey(root, *command)
            self.assertEqual(result.returncode, expected_exit, result.stdout + result.stderr)
            rows = self.read_rows(root / "table.csv")
            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0]["op"], "test")
            self.assertEqual(rows[0]["status"], expected_status or status)
            self.assertIn("NSUnderlyingError",
                          (root / "results/survey_errors.log").read_text())

    def test_success(self):
        self.run_case("OK", 0, 0)

    def test_compile_rejection_is_an_observation(self):
        self.run_case("COMPILE_FAIL", 10, 0)

    def test_runtime_and_numerical_failures_are_not_success(self):
        for status, code in [("LOAD_FAIL", 11), ("EVAL_FAIL", 12),
                             ("NUMERIC_FAIL", 13), ("RUN_ERROR", 15)]:
            with self.subTest(status=status):
                self.run_case(status, code, 1)

    def test_inconsistent_exit_codes_fail(self):
        for status, code in [("OK", 1), ("COMPILE_FAIL", 0)]:
            with self.subTest(status=status):
                self.run_case(status, code, 1, "RUN_ERROR")

    def test_unverified_layout_requires_explicit_opt_in(self):
        for status in ["NUMERIC_UNVERIFIED", "NUMERIC_LAYOUT_CANDIDATE"]:
            with self.subTest(status=status):
                self.run_case(status, 0, 1)
                self.run_case(status, 0, 0, allow=True)

    def test_default_layout_uses_a_fresh_run_directory(self):
        with tempfile.TemporaryDirectory(prefix="ane-survey-default-") as temp:
            root = self.make_tree(temp)
            result = self.run_survey(root)
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            runs = self.default_runs(root)
            self.assertEqual(len(runs), 1)
            run = runs[0]
            self.assertEqual([row["status"] for row in
                              self.read_rows(run / "survey_results.csv")], ["OK"])
            for name in ("survey_errors.log", "survey_stdout.log",
                         "run_meta.txt", "summary.txt"):
                self.assertTrue((run / name).is_file(), name)
            self.assertTrue((run / "stdout/test.log").is_file())
            self.assertIn("git_commit=", (run / "run_meta.txt").read_text())

    def test_repeated_default_runs_never_overwrite(self):
        with tempfile.TemporaryDirectory(prefix="ane-survey-repeat-") as temp:
            root = self.make_tree(temp)
            first = self.run_survey(root)
            second = self.run_survey(root)
            self.assertEqual(first.returncode, 0, first.stdout + first.stderr)
            self.assertEqual(second.returncode, 0, second.stdout + second.stderr)
            runs = self.default_runs(root)
            self.assertEqual(len(runs), 2)
            self.assertNotEqual(runs[0].name, runs[1].name)
            for run in runs:
                self.assertTrue((run / "survey_results.csv").is_file())

    def test_explicit_results_directory_owns_csv_and_logs(self):
        with tempfile.TemporaryDirectory(prefix="ane-survey-results-") as temp:
            root = self.make_tree(temp)
            result = self.run_survey(root, "--results", "out")
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            self.assertTrue((root / "out/survey_results.csv").is_file())
            self.assertTrue((root / "out/survey_errors.log").is_file())

    def test_refuses_existing_csv(self):
        with tempfile.TemporaryDirectory(prefix="ane-survey-csv-") as temp:
            root = self.make_tree(temp)
            existing = root / "table.csv"
            existing.write_text("keep me\n")
            result = self.run_survey(root, "--csv", "table.csv")
            self.assertEqual(result.returncode, 2)
            self.assertIn("refusing to overwrite existing CSV", result.stderr)
            self.assertEqual(existing.read_text(), "keep me\n")

    def test_refuses_nonempty_results_directory(self):
        with tempfile.TemporaryDirectory(prefix="ane-survey-dir-") as temp:
            root = self.make_tree(temp)
            (root / "results").mkdir()
            old = root / "results/old.log"
            old.write_text("old evidence\n")
            result = self.run_survey(root, "--results", "results", "--csv", "new.csv")
            self.assertEqual(result.returncode, 2)
            self.assertIn("refusing to reuse non-empty results directory", result.stderr)
            self.assertEqual(old.read_text(), "old evidence\n")
            self.assertFalse((root / "new.csv").exists())

    def test_list_only_writes_the_requested_csv(self):
        with tempfile.TemporaryDirectory(prefix="ane-survey-list-") as temp:
            root = self.make_tree(temp)
            result = self.run_survey(root, "--list", "--csv", "list.csv", probe=False)
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            self.assertEqual([row["status"] for row in
                              self.read_rows(root / "list.csv")], ["NOT_RUN"])
            self.assertFalse((root / "results").exists())
            self.assertFalse((root / "survey_errors.log").exists())

    def test_list_default_writes_only_the_csv_in_a_fresh_run_directory(self):
        with tempfile.TemporaryDirectory(prefix="ane-survey-list-default-") as temp:
            root = self.make_tree(temp)
            result = self.run_survey(root, "--list", probe=False)
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            runs = self.default_runs(root)
            self.assertEqual(len(runs), 1)
            self.assertTrue((runs[0] / "survey_results.csv").is_file())
            for name in ("survey_errors.log", "survey_stdout.log",
                         "run_meta.txt", "summary.txt", "stdout"):
                self.assertFalse((runs[0] / name).exists(), name)

    def test_list_refuses_existing_csv(self):
        with tempfile.TemporaryDirectory(prefix="ane-survey-list-existing-") as temp:
            root = self.make_tree(temp)
            existing = root / "list.csv"
            existing.write_text("keep me\n")
            result = self.run_survey(root, "--list", "--csv", "list.csv", probe=False)
            self.assertEqual(result.returncode, 2)
            self.assertIn("refusing to overwrite existing CSV", result.stderr)
            self.assertEqual(existing.read_text(), "keep me\n")

    def test_missing_probe_fails_outside_list_mode(self):
        with tempfile.TemporaryDirectory(prefix="ane-survey-noprobe-") as temp:
            root = self.make_tree(temp)
            result = self.run_survey(root, "--csv", "table.csv", probe=False)
            self.assertEqual(result.returncode, 1)
            self.assertIn("not found or not executable", result.stderr)

    def test_manifest_rejects_missing_indexed_mil(self):
        with tempfile.TemporaryDirectory(prefix="ane-survey-missing-mil-") as temp:
            root = self.make_tree(temp, index="ghost\ntest\n")
            result = self.run_survey(root, "--list", "--csv", "list.csv", probe=False)
            self.assertEqual(result.returncode, 1)
            self.assertIn("case manifest mismatch", result.stderr)
            self.assertIn("ghost", result.stderr)
            self.assertFalse((root / "list.csv").exists())

    def test_manifest_rejects_duplicate_index(self):
        with tempfile.TemporaryDirectory(prefix="ane-survey-duplicate-") as temp:
            root = self.make_tree(temp, index="test\ntest\n")
            result = self.run_survey(root, "--list", "--csv", "list.csv", probe=False)
            self.assertEqual(result.returncode, 1)
            self.assertIn("duplicate case names", result.stderr)
            self.assertFalse((root / "list.csv").exists())

    def test_manifest_rejects_unindexed_mil(self):
        with tempfile.TemporaryDirectory(prefix="ane-survey-extra-") as temp:
            root = self.make_tree(temp)
            (root / "opcases/extra.mil").write_text("extra\n")
            (root / "opcases/extra.meta").write_text("extra,4096,4096,-\n")
            result = self.run_survey(root, "--list", "--csv", "list.csv", probe=False)
            self.assertEqual(result.returncode, 1)
            self.assertIn("extra MIL", result.stderr)
            self.assertFalse((root / "list.csv").exists())

    def test_manifest_rejects_missing_meta(self):
        with tempfile.TemporaryDirectory(prefix="ane-survey-missing-meta-") as temp:
            root = self.make_tree(temp)
            (root / "opcases/test.meta").unlink()
            result = self.run_survey(root, "--list", "--csv", "list.csv", probe=False)
            self.assertEqual(result.returncode, 1)
            self.assertIn("missing meta", result.stderr)
            self.assertFalse((root / "list.csv").exists())

    def test_manifest_rejects_meta_label_mismatch(self):
        with tempfile.TemporaryDirectory(prefix="ane-survey-label-") as temp:
            root = self.make_tree(temp)
            (root / "opcases/test.meta").write_text("other,4096,4096,-\n")
            result = self.run_survey(root, "--list", "--csv", "list.csv", probe=False)
            self.assertEqual(result.returncode, 1)
            self.assertIn("label mismatch", result.stderr)
            self.assertFalse((root / "list.csv").exists())


if __name__ == "__main__":
    unittest.main(verbosity=2)
