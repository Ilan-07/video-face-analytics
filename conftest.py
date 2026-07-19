"""Put the project root on sys.path so tests can import pipeline modules."""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))


# A `slow` marker for tests that load real models (detect, caption, embed,
# whisper). They exercise the true stage code end to end, but cost seconds to
# minutes and download weights, so they are opt-in: `pytest --run-slow`.
def pytest_addoption(parser):
    parser.addoption("--run-slow", action="store_true", default=False,
                     help="run tests marked @pytest.mark.slow (real models)")


def pytest_configure(config):
    config.addinivalue_line(
        "markers", "slow: exercises real models; run only with --run-slow")


def pytest_collection_modifyitems(config, items):
    if config.getoption("--run-slow"):
        return
    skip = pytest.mark.skip(reason="needs --run-slow (loads real models)")
    for item in items:
        if "slow" in item.keywords:
            item.add_marker(skip)
