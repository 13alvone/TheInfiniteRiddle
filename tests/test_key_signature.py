#!/usr/bin/env python3
import sys
from pathlib import Path
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import riddle as irr


@pytest.mark.parametrize("pc, sf", [
    (0, 0),   # C major
    (1, -5),  # Db major
    (2, 2),   # D major
    (3, -3),  # Eb major
    (4, 4),   # E major
    (5, -1),  # F major
    (6, 6),   # F# major
    (7, 1),   # G major
    (8, -4),  # Ab major
    (9, 3),   # A major
    (10, -2), # Bb major
    (11, 5),  # B major
])
def test_pitch_class_to_sf(pc, sf):
    assert irr.pitch_class_to_sf(pc) == sf
