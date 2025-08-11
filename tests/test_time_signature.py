#!/usr/bin/env python3
import sys
from pathlib import Path
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import riddle as irr

ODD_METERS = {(3, 4), (5, 4), (5, 8), (7, 8), (9, 8)}


def test_pick_time_signature_odd(root_seed: bytes):
    prng = irr.domain_prngs(root_seed)["rhythm"]
    ts = irr.pick_time_signature(prng, "glass")
    assert ts in ODD_METERS


def test_pick_time_signature_chaotic_list(root_seed: bytes):
    prng = irr.domain_prngs(root_seed)["rhythm"]
    tss = irr.pick_time_signature(prng, "glass", chaotic=True)
    assert isinstance(tss, list) and tss
    for ts in tss:
        assert ts in ODD_METERS


def test_pick_time_signatures_sequence(root_seed: bytes):
    prng = irr.domain_prngs(root_seed)["rhythm"]
    form_nodes = ["INV", "PRC", "CLM"]
    seq = irr.pick_time_signatures(prng, "glass", form_nodes)
    assert len(seq) == len(form_nodes)
    for ts in seq:
        assert ts in ODD_METERS
