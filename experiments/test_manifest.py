#!/usr/bin/env python3
"""Offline manifest integrity checks for the committed opcases fixtures."""
from pathlib import Path
import struct
import unittest

ROOT = Path(__file__).resolve().parent
CASES = ROOT / "opcases"


class CaseManifestTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.names = [line.strip() for line in
                     (CASES / "_index.txt").read_text().splitlines() if line.strip()]

    def test_index_has_exactly_76_unique_sorted_names(self):
        self.assertEqual(len(self.names), 76)
        self.assertEqual(len(set(self.names)), 76)
        self.assertEqual(self.names, sorted(self.names))

    def test_every_indexed_case_has_mil_and_meta(self):
        for name in self.names:
            with self.subTest(case=name):
                self.assertTrue((CASES / (name + ".mil")).is_file())
                self.assertTrue((CASES / (name + ".meta")).is_file())
        self.assertEqual({p.stem for p in CASES.glob("*.mil")}, set(self.names))
        self.assertEqual({p.stem for p in CASES.glob("*.meta")}, set(self.names))

    def test_meta_rows_are_well_formed_and_weights_exist(self):
        weight_files = set()
        for name in self.names:
            with self.subTest(case=name):
                fields = (CASES / (name + ".meta")).read_text().rstrip("\n").split(",")
                self.assertEqual(len(fields), 4)
                self.assertEqual(fields[0], name)
                self.assertGreater(int(fields[1]), 0)
                self.assertGreater(int(fields[2]), 0)
                weights = fields[3]
                if weights != "-":
                    path = Path(weights)
                    self.assertFalse(path.is_absolute())
                    self.assertNotIn("..", path.parts)
                    self.assertEqual(path.parts[0], "opcases")
                    self.assertTrue((ROOT / path).is_file(), weights)
                    weight_files.add(path.name)
        self.assertEqual(len(weight_files), 10)

    def test_weight_blobs_use_the_v2_container(self):
        for blob in sorted(CASES.glob("*.wblob")):
            with self.subTest(blob=blob.name):
                data = blob.read_bytes()
                self.assertGreaterEqual(len(data), 64)
                count, version = struct.unpack_from("<II", data, 0)
                self.assertGreater(count, 0)
                self.assertEqual(version, 2)
                self.assertEqual(data[64:68], b"\xef\xbe\xad\xde")

    def test_gitignore_does_not_hide_case_fixtures(self):
        lines = [line.strip() for line in
                 (ROOT / ".gitignore").read_text().splitlines()]
        self.assertNotIn("*.bin", lines)
        self.assertNotIn("*.wblob", lines)
        self.assertNotIn("opcases/", lines)
        self.assertNotIn("opcases", lines)

    def test_load_sensitivity_executes_build_outputs(self):
        script = (ROOT / "load_sensitivity.sh").read_text()
        self.assertIn('BUILD_DIR=${BUILD_DIR:-build}', script)
        self.assertIn('"$BUILD_DIR/sustained"', script)
        self.assertIn('"$BUILD_DIR/cpuhog"', script)
        self.assertNotIn("./sustained", script)
        self.assertNotIn("./cpuhog", script)


if __name__ == "__main__":
    unittest.main(verbosity=2)
