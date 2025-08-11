#!/usr/bin/env python3
import json
import math
import struct
import sys
import wave
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import riddle as irr


def _write_sine(path: Path, amplitude: float = 0.5, freq: float = 1000.0, dur: float = 0.1, rate: int = 48000) -> None:
    """Create a simple sine wave WAV for testing."""
    frames = int(rate * dur)
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(rate)
        buf = bytearray()
        for i in range(frames):
            sample = amplitude * math.sin(2 * math.pi * freq * (i / rate))
            buf.extend(struct.pack("<h", int(sample * 32767)))
        wf.writeframes(buf)


def test_measure_functions(tmp_path: Path):
    wav = tmp_path / "tone.wav"
    _write_sine(wav, amplitude=0.5)
    peak = irr.measure_peak(wav)
    rms = irr.measure_rms(wav)
    assert peak == pytest.approx(-6.0, abs=0.5)
    assert rms == pytest.approx(-9.0, abs=0.5)
    assert irr.validate_lufs(rms, -9.0, tolerance=1.0)
    assert not irr.validate_lufs(rms, -14.0, tolerance=1.0)


def test_run_riddle_metrics(tmp_path: Path, root_seed: bytes):
    outdir = tmp_path / "out"
    db_path = tmp_path / "vault.db"
    outdir.mkdir()
    orig_entropy = irr.gather_entropy
    orig_duration = irr.sample_duration_seconds
    irr.gather_entropy = lambda: root_seed
    irr.sample_duration_seconds = lambda prng, theme, bucket: 1
    try:
        irr.run_riddle("glass", outdir, db_path, "short", False, 0, -14.0, None, 0)
    finally:
        irr.gather_entropy = orig_entropy
        irr.sample_duration_seconds = orig_duration
    sidecar = json.loads(next(outdir.glob("*.riddle.json")).read_text())
    assert sidecar["true_peak_est"] is not None
    assert sidecar["crest_factor_est"] is not None
