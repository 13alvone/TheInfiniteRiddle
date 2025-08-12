#!/usr/bin/env python3
import json
import os
import sqlite3
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


class TestHarmonyEngineCLI(unittest.TestCase):
    def _run(self, extra):
        cmd = [sys.executable, "-m", "riddle.harmony_engine", "gen", "--key", "C", "--mode", "ionian", "--seed", "0123456789abcdef0123456789abcdef", "--stdout-json"]
        cmd.extend(extra)
        return subprocess.run(cmd, capture_output=True, text=True)

    def test_reproducible_json(self):
        r1 = self._run([])
        r2 = self._run([])
        self.assertEqual(r1.returncode, 0)
        self.assertEqual(r1.stdout, r2.stdout)
        self.assertTrue(r1.stdout.strip().startswith("{"))

    def test_db_insert(self):
        with tempfile.TemporaryDirectory() as tmp:
            db = Path(tmp) / "harmony.db"
            r = self._run(["--db", str(db)])
            self.assertEqual(r.returncode, 0, r.stderr)
            con = sqlite3.connect(db)
            cur = con.cursor()
            cur.execute("SELECT COUNT(*) FROM harmony_runs")
            self.assertGreater(cur.fetchone()[0], 0)
            cur.execute("SELECT COUNT(*) FROM harmony_events")
            self.assertGreater(cur.fetchone()[0], 0)
            con.close()

    def test_invalid_key(self):
        cmd = [sys.executable, "-m", "riddle.harmony_engine", "gen", "--key", "H"]
        r = subprocess.run(cmd, capture_output=True, text=True)
        self.assertNotEqual(r.returncode, 0)
        self.assertIn("[x]", r.stderr)
        self.assertNotIn("Traceback", r.stderr)


if __name__ == "__main__":
    unittest.main()
