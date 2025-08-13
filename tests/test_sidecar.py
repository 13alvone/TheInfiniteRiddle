#!/usr/bin/env python3
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import riddle as irr
from riddle.core import SECTION_PCT_RANGES


@pytest.fixture
def short_run(tmp_path: Path, root_seed: bytes):
    outdir = tmp_path / "out"
    db_path = tmp_path / "vault.db"
    outdir.mkdir()
    orig_entropy = irr.gather_entropy
    orig_duration = irr.sample_duration_seconds
    irr.gather_entropy = lambda: root_seed
    irr.sample_duration_seconds = lambda prng, theme, bucket: (1, bucket or "short")
    try:
        irr.run_riddle("glass", outdir, db_path, "short", False, 0, -14.0, None, 0)
        yield outdir
    finally:
        irr.gather_entropy = orig_entropy
        irr.sample_duration_seconds = orig_duration


def test_sidecar_fields(short_run: Path):
    sidecars = list(short_run.glob("*.riddle.json"))
    assert sidecars, "sidecar not generated"
    data = json.loads(sidecars[0].read_text())
    for field in ("seed_commitment", "theme", "duration_bucket", "form_nodes", "artifact_hashes"):
        assert field in data
    assert "durations" in data and data["durations"]
    assert all("pct" in seg for seg in data["durations"])


def test_sidecar_pct_ranges(short_run: Path):
    sidecar = json.loads(next(short_run.glob("*.riddle.json")).read_text())
    ranges = SECTION_PCT_RANGES[sidecar["theme"]]
    for seg in sidecar["durations"]:
        mn, mx = ranges[seg["node"]]
        assert mn <= seg["pct"] <= mx


def test_seed_repro(tmp_path: Path):
    outdir = tmp_path / "out"
    db_path = tmp_path / "vault.db"
    outdir.mkdir()
    seed = "deadbeefdeadbeefdeadbeefdeadbeef"
    irr.run_riddle("glass", outdir, db_path, "short", False, 0, -14.0, seed, 0)
    sidecar = json.loads(next(outdir.glob("*.riddle.json")).read_text())
    assert sidecar["seed_commitment"] == irr.seed_commitment(bytes.fromhex(seed))
