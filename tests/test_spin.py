#!/usr/bin/env python3
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import riddle.__main__ as riddle_main


class TestSpin(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name).resolve()
        self.cwd = os.getcwd()
        os.chdir(self.root)

    def tearDown(self):
        os.chdir(self.cwd)
        self.tmp.cleanup()

    @mock.patch("riddle.__main__._resolve_artifact_paths")
    @mock.patch("riddle.__main__.run_riddle")
    def test_overrides(self, mock_run, mock_resolve):
        mock_resolve.return_value = (self.root / "out", self.root / "db.db")
        seed = "deadbeefdeadbeefdeadbeefdeadbeef"
        argv = [
            "riddle",
            "spin",
            "--outdir",
            "out",
            "--db",
            "db.db",
            "--theme",
            "glass",
            "--bucket",
            "med",
            "--stems",
            "--mythic-max",
            "3",
            "--lufs-target",
            "-15.0",
            "--seed",
            seed,
        ]
        with mock.patch.object(sys, "argv", argv):
            riddle_main.main()
        args = mock_run.call_args[0]
        self.assertEqual(args[0], "glass")
        self.assertEqual(args[3], "med")
        self.assertTrue(args[4])
        self.assertEqual(args[5], 3)
        self.assertEqual(args[6], -15.0)
        self.assertEqual(args[7], seed)

    @mock.patch("riddle.__main__._resolve_artifact_paths")
    @mock.patch("riddle.__main__.run_riddle")
    def test_random_defaults(self, mock_run, mock_resolve):
        mock_resolve.return_value = (self.root / "out", self.root / "db.db")
        argv = ["riddle", "spin"]
        with mock.patch.object(sys, "argv", argv):
            riddle_main.main()
        args1 = mock_run.call_args_list[-1][0]
        with mock.patch.object(sys, "argv", argv):
            riddle_main.main()
        args2 = mock_run.call_args_list[-1][0]

        def unpack(args):
            theme, _, _, bucket, stems, mythic_max, lufs_target, seed, _ = args
            return theme, bucket, stems, mythic_max, lufs_target, seed

        params1 = unpack(args1)
        params2 = unpack(args2)

        for params in (params1, params2):
            theme, bucket, stems, mythic_max, lufs_target, seed = params
            self.assertIn(theme, ["glass", "salt"])
            self.assertIn(bucket, ["short", "med", "long"])
            self.assertIsInstance(stems, bool)
            self.assertIsInstance(mythic_max, int)
            self.assertIsInstance(lufs_target, float)
            self.assertEqual(len(seed), 32)
            self.assertTrue(all(c in "0123456789abcdef" for c in seed))

        self.assertNotEqual(params1, params2)


if __name__ == "__main__":
    unittest.main()
