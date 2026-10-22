import os

import pytest

os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")


def pytest_configure(config):
    config.addinivalue_line("markers", "live: loads the small 0.8B model (MACK_RUN_MODEL=1 to run)")
    config.addinivalue_line("markers", "model: needs the real 4B model; run by hand, one at a time (MACK_RUN_MODEL=1)")


def pytest_collection_modifyitems(config, items):
    """The default suite never loads a model, so it runs in well under a minute."""
    if os.environ.get("MACK_RUN_MODEL") == "1":
        return
    skip = pytest.mark.skip(reason="needs a real model; set MACK_RUN_MODEL=1")
    for item in items:
        if "model" in item.keywords or "live" in item.keywords:
            item.add_marker(skip)
