#!/usr/bin/env python3
import sys
from pathlib import Path
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import riddle as irr


def _bucket_distribution(theme: str, root_seed: bytes, n: int = 2000):
    prng = irr.domain_prngs(root_seed)["form"]
    counts = {"short": 0, "med": 0, "long": 0}
    for _ in range(n):
        _, bucket = irr.sample_duration_seconds(prng, theme, None)
        counts[bucket] += 1
    return {k: v / n for k, v in counts.items()}


def test_glass_bucket_bias(root_seed: bytes):
    dist = _bucket_distribution("glass", root_seed, 2000)
    assert dist["short"] == pytest.approx(0.56, abs=0.05)
    assert dist["med"] == pytest.approx(0.34, abs=0.05)
    assert dist["long"] == pytest.approx(0.10, abs=0.03)


def test_salt_bucket_bias(root_seed: bytes):
    dist = _bucket_distribution("salt", root_seed, 2000)
    assert dist["short"] == pytest.approx(0.75, abs=0.05)
    assert dist["med"] == pytest.approx(0.21, abs=0.05)
    assert dist["long"] == pytest.approx(0.04, abs=0.03)
