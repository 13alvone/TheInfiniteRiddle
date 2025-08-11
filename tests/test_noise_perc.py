#!/usr/bin/env python3
import os
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import riddle as irr


class TestNoisePercDeterminism(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name).resolve()
        self.cwd = os.getcwd()
        os.chdir(self.root)
        self.seed = "feedfacefeedfacefeedfacefeedface"

    def tearDown(self):
        os.chdir(self.cwd)
        self.tmp.cleanup()

    def run_once(self, outdir: Path, db_path: Path) -> Path:
        orig_duration = irr.sample_duration_seconds
        irr.sample_duration_seconds = lambda prng, theme, bucket: 1
        try:
            irr.run_riddle("glass", outdir, db_path, "short", True, 0, -14.0, self.seed, 0)
        finally:
            irr.sample_duration_seconds = orig_duration
        return next(outdir.glob("*_STEM_PERC_*wav"))

    def test_percussion_deterministic_with_seed(self):
        out1 = self.root / "run1"
        out2 = self.root / "run2"
        perc1 = self.run_once(out1, self.root / "db1.db")
        perc2 = self.run_once(out2, self.root / "db2.db")
        self.assertEqual(perc1.read_bytes(), perc2.read_bytes())


if __name__ == "__main__":
    unittest.main()
