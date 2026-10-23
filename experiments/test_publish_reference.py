#!/usr/bin/env python3
"""Offline tests for curated evidence publication; no ANE access."""
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from publish_reference import publish


class ReferencePublicationTests(unittest.TestCase):
    def fixture(self, root):
        source = root / "original"
        source.mkdir()
        (source / "metadata.json").write_text(json.dumps({
            "started_at": "2027-02-03", "scope": "historical application scope",
            "source_sha256": {"old_probe.m": "abc"},
        }) + "\n")
        (source / "summary.json").write_text('[{"command": ["/Users/private/ANE/experiments/results/run/table.csv"]}]\n')
        (source / "survey_results.csv").write_text("op,status\nadd,OK\n")
        (source / "device.log").write_text("device=h16g temp=/var/folders/xx/private-user/T/compile/file  \t\n")
        (source / "delivery-hashes.json").write_text("{}\n")
        (source / "survey" / "stdout").mkdir(parents=True)
        (source / "survey" / "stdout" / "add.log").write_text("NUMERIC=ACCEPTANCE_ONLY LAYOUT=compact\n")
        (source / "survey" / "survey_stdout.log").write_text("duplicate\n")
        (source / "survey" / "dump.bin").write_bytes(b"binary")
        return source

    def test_publication_is_curated_traceable_and_non_destructive(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            source = self.fixture(root)
            before = {p.relative_to(source): p.read_bytes()
                      for p in source.rglob("*") if p.is_file()}
            output = root / "published"
            count = publish(source, output)
            self.assertEqual(count, 5)
            title = (output / "README.md").read_text().splitlines()[0]
            self.assertEqual(title, "# Historical ANE reference evidence")
            self.assertNotIn("2026-10-18", title)
            self.assertNotIn("M4 Air", title)
            self.assertFalse((output / "delivery-hashes.json").exists())
            self.assertFalse((output / "survey" / "survey_stdout.log").exists())
            self.assertFalse((output / "survey" / "dump.bin").exists())
            self.assertIn("${EXPERIMENTS}/results/run/table.csv", (output / "summary.json").read_text())
            self.assertIn("${TMPDIR}/compile/file", (output / "device.log").read_text())
            self.assertNotIn("private-user", (output / "device.log").read_text())
            self.assertTrue((output / "device.log").read_text().endswith("/compile/file\n"))
            manifest = json.loads((output / "publication.json").read_text())
            for name, record in manifest["files"].items():
                self.assertEqual(record["original_sha256"], hashlib.sha256(before[Path(name)]).hexdigest())
                self.assertEqual(record["published_sha256"], hashlib.sha256((output / name).read_bytes()).hexdigest())
            self.assertEqual(before, {p.relative_to(source): p.read_bytes()
                                      for p in source.rglob("*") if p.is_file()})
            self.assertIn("not the current cleaned source tree", manifest["source_hashes_describe"])

    def test_committed_reference_hashes_and_case_set(self):
        reference = Path(__file__).resolve().parent / "reference" / "m4-air-20261018"
        manifest = json.loads((reference / "publication.json").read_text())
        for name, record in manifest["files"].items():
            data = (reference / name).read_bytes()
            self.assertEqual(hashlib.sha256(data).hexdigest(), record["published_sha256"], name)
            self.assertNotIn(b"/Users/", data, name)
        import csv
        with (reference / "survey_results.csv").open() as source:
            rows = list(csv.DictReader(source))
        expected = set((reference.parents[1] / "opcases" / "_index.txt").read_text().splitlines())
        self.assertEqual(len(rows), len(expected))
        self.assertEqual({row["op"] for row in rows}, expected)
        self.assertTrue(manifest["historical_evidence"])

    def test_refuses_overwrite(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            source = self.fixture(root)
            output = root / "published"
            output.mkdir()
            with self.assertRaises(FileExistsError):
                publish(source, output)
            self.assertEqual(list(output.iterdir()), [])

    def test_rejects_incomplete_run(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            source = root / "original"
            source.mkdir()
            with self.assertRaises(ValueError):
                publish(source, root / "published")
            self.assertFalse((root / "published").exists())


if __name__ == "__main__":
    unittest.main(verbosity=2)
