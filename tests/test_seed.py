#!/usr/bin/env python3
import os
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import riddle as irr


class TestSeedHandling(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name).resolve()
        self.cwd = os.getcwd()
        os.chdir(self.root)

    def tearDown(self):
        os.chdir(self.cwd)
        self.tmp.cleanup()

    def test_parse_args_valid_seed(self):
        seed = "deadbeefdeadbeefdeadbeefdeadbeef"
        args = irr.parse_args(["generate", "glass", "out", "--db", "vault.db", "--seed", seed])
        self.assertEqual(args.seed, seed)

    def test_parse_args_invalid_seed_length(self):
        with self.assertRaises(SystemExit):
            irr.parse_args(["generate", "glass", "out", "--db", "vault.db", "--seed", "abc"])

    def test_parse_args_invalid_seed_hex(self):
        bad = "g" * 32
        with self.assertRaises(SystemExit):
            irr.parse_args(["generate", "glass", "out", "--db", "vault.db", "--seed", bad])

    def test_run_riddle_invalid_seed_logs(self):
        with self.assertLogs(level="ERROR") as cm:
            irr.run_riddle("glass", self.root, self.root / "vault.db", "short", False, 0, -14.0, "xyz", 0)
        self.assertIn("Seed must be 32 hex characters", "\n".join(cm.output))


if __name__ == "__main__":
    unittest.main()
