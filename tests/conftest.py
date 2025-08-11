#!/usr/bin/env python3
import pytest
from pathlib import Path
import sys

# Ensure repository root on path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def pytest_addoption(parser: pytest.Parser) -> None:
    parser.addoption(
        "--seed",
        action="store",
        default="deadbeef" * 8,
        help="hex-encoded root seed for deterministic PRNG",
    )


@pytest.fixture
def root_seed(request: pytest.FixtureRequest) -> bytes:
    hex_seed = request.config.getoption("--seed")
    return bytes.fromhex(hex_seed)
