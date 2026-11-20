import os
from pathlib import Path

import pytest

from gesturelink.data.labels import load_labels

FIXTURE_DIR = Path(__file__).parent / "fixtures" / "landmarks"


def pytest_collection_modifyitems(config, items):
    # tests marked `board` need real hardware (or the Wokwi simulator): opt in with
    # GESTURELINK_PORT=COM3 (or rfc2217://localhost:4000) so CI and normal runs skip them
    if os.environ.get("GESTURELINK_PORT"):
        return
    skip = pytest.mark.skip(reason="set GESTURELINK_PORT to run live board tests")
    for item in items:
        if "board" in item.keywords:
            item.add_marker(skip)


@pytest.fixture(scope="session")
def labels():
    return load_labels()


@pytest.fixture(scope="session")
def fixture_csv() -> Path:
    # 140 synthetic rows (20 per gesture), made with `gesturelink synthesize --seed 7`
    return FIXTURE_DIR / "samples.csv"
