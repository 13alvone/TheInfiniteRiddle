#!/usr/bin/env python3
"""Basic audio quality analysis utilities using the Python standard library."""

from __future__ import annotations

import audioop
import math
import wave
from pathlib import Path
from typing import Tuple


def _read_frames(path: Path) -> Tuple[bytes, int]:
    """Return raw frames and sample width from a WAV file."""
    with wave.open(str(path), "rb") as wf:
        frames = wf.readframes(wf.getnframes())
        width = wf.getsampwidth()
    return frames, width


def _full_scale(width: int) -> float:
    """Compute the full-scale value for a given sample width."""
    return float(2 ** (8 * width - 1))


def measure_peak(path: Path) -> float:
    """Estimate digital true peak (dBFS) of a WAV file."""
    frames, width = _read_frames(path)
    peak = audioop.max(frames, width)
    if peak <= 0:
        return float("-inf")
    return 20.0 * math.log10(peak / _full_scale(width))


def measure_rms(path: Path) -> float:
    """Measure RMS level (dBFS) of a WAV file."""
    frames, width = _read_frames(path)
    rms = audioop.rms(frames, width)
    if rms <= 0:
        return float("-inf")
    return 20.0 * math.log10(rms / _full_scale(width))


def validate_lufs(measured_lufs: float, target_lufs: float, tolerance: float = 1.0) -> bool:
    """Validate loudness against target using RMS as LUFS approximation."""
    return abs(measured_lufs - target_lufs) <= tolerance
