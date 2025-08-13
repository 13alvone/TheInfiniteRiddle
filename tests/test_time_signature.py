#!/usr/bin/env python3
import sys
from pathlib import Path
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import riddle as irr


def _ts_distribution(theme: str, root_seed: bytes, n: int = 4000):
    prng = irr.domain_prngs(root_seed)["rhythm"]
    counts = {}
    for _ in range(n):
        ts = irr.pick_time_signature(prng, theme)
        counts[ts] = counts.get(ts, 0) + 1
    return {k: v / n for k, v in counts.items()}


def test_glass_time_signature_distribution(root_seed: bytes):
    dist = _ts_distribution("glass", root_seed, 4000)
    assert dist[(4, 4)] == pytest.approx(0.35, abs=0.05)
    assert dist[(5, 4)] == pytest.approx(0.22, abs=0.05)
    assert dist[(7, 8)] == pytest.approx(0.18, abs=0.05)
    assert dist[(9, 8)] == pytest.approx(0.12, abs=0.03)
    assert dist[(11, 8)] == pytest.approx(0.08, abs=0.03)
    assert dist[(13, 8)] == pytest.approx(0.05, abs=0.02)


def test_salt_time_signature_distribution(root_seed: bytes):
    dist = _ts_distribution("salt", root_seed, 4000)
    assert dist[(7, 8)] == pytest.approx(0.42, abs=0.05)
    assert dist[(5, 4)] == pytest.approx(0.22, abs=0.05)
    assert dist[(3, 4)] == pytest.approx(0.18, abs=0.05)
    assert dist[(4, 4)] == pytest.approx(0.18, abs=0.05)


def test_pick_time_signature_chaotic_deterministic(root_seed: bytes):
    prng1 = irr.domain_prngs(root_seed)["rhythm"]
    prng2 = irr.domain_prngs(root_seed)["rhythm"]
    seq1 = irr.pick_time_signature(prng1, "glass", chaotic=True, count=16)
    seq2 = irr.pick_time_signature(prng2, "glass", chaotic=True, count=16)
    assert seq1 == seq2 and isinstance(seq1, list) and len(seq1) == 16


def test_pick_time_signatures_sequence(root_seed: bytes):
    prng = irr.domain_prngs(root_seed)["rhythm"]
    nodes = ["INV", "PRC", "CLM"]
    base, meters = irr.pick_time_signatures(prng, "glass", nodes, bars=3)
    assert list(meters.keys()) == nodes
    assert len(base) == len(nodes)
    for i, node in enumerate(nodes):
        assert meters[node][0] == base[i]
