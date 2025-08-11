#!/usr/bin/env python3
import os
import sys
import sqlite3
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import riddle as irr


class TestStemsGeneration(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name).resolve()
        self.cwd = os.getcwd()
        os.chdir(self.root)

    def tearDown(self):
        os.chdir(self.cwd)
        self.tmp.cleanup()

    def test_stems_and_vault_records(self):
        orig_duration = irr.sample_duration_seconds
        irr.sample_duration_seconds = lambda prng, theme, bucket: 1
        try:
            outdir = self.root / "out"
            db_path = self.root / "vault.db"
            irr.run_riddle("glass", outdir, db_path, "short", True, 0, -14.0, 0)
            stems = list(outdir.glob("*_STEM_*wav"))
            self.assertEqual(len(stems), 4)
            conn = sqlite3.connect(db_path)
            cur = conn.cursor()
            cur.execute("SELECT COUNT(*) FROM artifacts WHERE kind='stem'")
            self.assertEqual(cur.fetchone()[0], 4)
            conn.close()
        finally:
            irr.sample_duration_seconds = orig_duration


if __name__ == "__main__":
    unittest.main()
