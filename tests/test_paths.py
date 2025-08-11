#!/usr/bin/env python3
import os
import tempfile
import unittest
from pathlib import Path

import infinite_riddle_root as irr


class TestPathResolution(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name).resolve()
        self.cwd = os.getcwd()
        os.chdir(self.root)

    def tearDown(self):
        os.chdir(self.cwd)
        self.tmp.cleanup()

    def test_relative_paths(self):
        args = irr.parse_args(["glass", "relout", "--db", "vault.db"])
        outdir, db = irr._resolve_artifact_paths(args.outdir, args.db, self.root)
        self.assertEqual(outdir, self.root / "relout")
        self.assertEqual(db, self.root / "vault.db")
        self.assertTrue(outdir.is_dir())
        self.assertTrue(db.parent.is_dir())

    def test_absolute_paths(self):
        abs_out = self.root / "absout"
        abs_db = self.root / "dbdir" / "vault.db"
        args = irr.parse_args(["glass", str(abs_out), "--db", str(abs_db)])
        outdir, db = irr._resolve_artifact_paths(args.outdir, args.db, self.root)
        self.assertEqual(outdir, abs_out)
        self.assertEqual(db, abs_db)
        self.assertTrue(outdir.is_dir())
        self.assertTrue(db.parent.is_dir())

    def test_traversal_rejected(self):
        args = irr.parse_args(["glass", "../escape", "--db", "vault.db"])
        with self.assertRaises(ValueError):
            irr._resolve_artifact_paths(args.outdir, args.db, self.root)
        args = irr.parse_args(["glass", "relout", "--db", "../vault.db"])
        with self.assertRaises(ValueError):
            irr._resolve_artifact_paths(args.outdir, args.db, self.root)


if __name__ == "__main__":
    unittest.main()
