#!/usr/bin/env python3
"""Affective audio metrics for The Infinite Riddle."""
from __future__ import annotations

import audioop
import math
import struct
import wave
from pathlib import Path


def _mono_16bit(path: Path):
    """Load a WAV file as mono 16-bit samples."""
    with wave.open(str(path), "rb") as wf:
        frames = wf.readframes(wf.getnframes())
        width = wf.getsampwidth()
        rate = wf.getframerate()
        channels = wf.getnchannels()
    if channels > 1:
        frames = audioop.tomono(frames, width, 0.5, 0.5)
    if width != 2:
        frames = audioop.lin2lin(frames, width, 2)
    samples = struct.unpack("<" + "h" * (len(frames) // 2), frames)
    return samples, rate


def rhythm_stability(path: Path, frame: int = 1024) -> float:
    """Estimate rhythm stability as a coherence metric in [0, 1]."""
    samples, _ = _mono_16bit(path)
    if len(samples) <= frame:
        return 1.0
    energies = []
    for i in range(0, len(samples), frame):
        window = samples[i : i + frame]
        if not window:
            break
        rms = math.sqrt(sum(x * x for x in window) / len(window))
        energies.append(rms)
    if len(energies) < 2:
        return 1.0
    mean = sum(energies) / len(energies)
    if mean == 0:
        return 0.0
    variance = sum((e - mean) ** 2 for e in energies) / len(energies)
    stdev = math.sqrt(variance)
    coherence = 1.0 - min(1.0, stdev / (mean + 1e-9))
    return max(0.0, min(1.0, coherence))


def spectral_centroid(path: Path) -> float:
    """Approximate spectral centroid as a presence metric in [0, 1]."""
    samples, _ = _mono_16bit(path)
    if len(samples) < 2:
        return 0.0
    zero_crossings = sum(
        1
        for a, b in zip(samples[:-1], samples[1:])
        if (a >= 0) != (b >= 0)
    )
    presence = zero_crossings / (len(samples) - 1)
    return max(0.0, min(1.0, presence))
