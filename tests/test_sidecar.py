#!/usr/bin/env python3
import json
import sqlite3
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import riddle as irr


@pytest.fixture
def short_run(tmp_path: Path, root_seed: bytes):
    outdir = tmp_path / "out"
    db_path = tmp_path / "vault.db"
    outdir.mkdir()
    orig_entropy = irr.gather_entropy
    orig_duration = irr.sample_duration_seconds
    irr.gather_entropy = lambda: root_seed
    irr.sample_duration_seconds = lambda prng, theme, bucket: 1
    try:
        irr.run_riddle("glass", outdir, db_path, "short", False, 0, -14.0, 0)
        yield outdir
    finally:
        irr.gather_entropy = orig_entropy
        irr.sample_duration_seconds = orig_duration


def test_sidecar_fields(short_run: Path):
    sidecars = list(short_run.glob("*.riddle.json"))
    assert sidecars, "sidecar not generated"
    data = json.loads(sidecars[0].read_text())
    for field in ("seed_commitment", "theme", "form_nodes", "artifact_hashes"):
        assert field in data
