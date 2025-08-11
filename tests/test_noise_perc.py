#!/usr/bin/env python3
import os
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import riddle as irr
import riddle.synth as synth
from riddle.core import Xoshiro256StarStar


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


class TestNoisePercRender(unittest.TestCase):
    def test_renders_silence_without_hit(self):
        prng = Xoshiro256StarStar(b"\x00" * 32)
        drums = synth.NoisePerc(48000, prng)
        buf = drums.render(16)
        self.assertEqual(buf, [0.0] * 16)

    def test_hit_produces_output(self):
        prng = Xoshiro256StarStar(b"\x00" * 32)
        drums = synth.NoisePerc(48000, prng)
        drums.hit(1.0)
        buf = drums.render(16)
        self.assertTrue(any(abs(s) > 0 for s in buf))


if __name__ == "__main__":
    unittest.main()
