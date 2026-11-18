from pathlib import Path

import pytest

from gesturelink.data.labels import load_labels

FIXTURE_DIR = Path(__file__).parent / "fixtures" / "landmarks"


@pytest.fixture(scope="session")
def labels():
    return load_labels()


@pytest.fixture(scope="session")
def fixture_csv() -> Path:
    # 140 synthetic rows (20 per gesture), made with `gesturelink synthesize --seed 7`
    return FIXTURE_DIR / "samples.csv"
