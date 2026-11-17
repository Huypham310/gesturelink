import numpy as np
import pytest

from gesturelink.data.io import SampleWriter, read_samples
from gesturelink.data.labels import load_labels
from gesturelink.data.schema import (
    COORD_COLUMNS,
    CSV_COLUMNS,
    FEATURE_DIM,
    NUM_LANDMARKS,
    Sample,
    SchemaError,
    check_header,
)


def make_sample(**overrides):
    values = {
        "label": "fist",
        "label_id": 1,
        "session_id": "s1",
        "timestamp_ms": 120,
        "handedness": "Right",
        "landmarks": np.random.default_rng(0).random((NUM_LANDMARKS, 3)),
    }
    values.update(overrides)
    return Sample(**values)


def test_column_layout():
    assert FEATURE_DIM == 63
    assert len(COORD_COLUMNS) == 63
    assert COORD_COLUMNS[:4] == ["x0", "y0", "z0", "x1"]
    assert COORD_COLUMNS[-1] == "z20"
    assert CSV_COLUMNS[:5] == ["label", "label_id", "session_id", "timestamp_ms", "handedness"]
    assert len(CSV_COLUMNS) == 68


def test_sample_roundtrip_through_row():
    sample = make_sample()
    row = dict(zip(CSV_COLUMNS, sample.to_row(), strict=True))
    back = Sample.from_row(row)
    assert back.label == "fist" and back.label_id == 1 and back.timestamp_ms == 120
    np.testing.assert_allclose(back.landmarks, sample.landmarks, atol=1e-6)


def test_flat_landmarks_are_reshaped():
    sample = make_sample(landmarks=np.zeros(63))
    assert sample.landmarks.shape == (21, 3)


@pytest.mark.parametrize(
    "overrides",
    [
        {"landmarks": np.zeros((20, 3))},  # wrong number of landmarks
        {"handedness": "Both"},
        {"label": ""},
        {"landmarks": np.full((21, 3), np.nan)},
    ],
)
def test_sample_validation(overrides):
    with pytest.raises(SchemaError):
        make_sample(**overrides)


def test_from_row_rejects_bad_rows():
    row = dict(zip(CSV_COLUMNS, make_sample().to_row(), strict=True))
    bad = dict(row, x3="abc")
    with pytest.raises(SchemaError):
        Sample.from_row(bad)
    del row["z20"]
    with pytest.raises(SchemaError):
        Sample.from_row(row)


def test_check_header():
    check_header(CSV_COLUMNS)
    with pytest.raises(SchemaError):
        check_header(CSV_COLUMNS[:-1])
    with pytest.raises(SchemaError):
        check_header(list(reversed(CSV_COLUMNS)))


def test_writer_appends_header_once(tmp_path):
    path = tmp_path / "out.csv"
    with SampleWriter(path) as w:
        w.write(make_sample())
    with SampleWriter(path) as w:
        w.write(make_sample(timestamp_ms=200))
    assert len(path.read_text().splitlines()) == 3  # 1 header + 2 rows
    assert [s.timestamp_ms for s in read_samples(path)] == [120, 200]


def test_read_reports_line_number(tmp_path):
    path = tmp_path / "bad.csv"
    row = make_sample().to_row()
    row[4] = "Nobody"
    path.write_text(",".join(CSV_COLUMNS) + "\n" + ",".join(row) + "\n")
    with pytest.raises(SchemaError, match=r"bad.csv:2"):
        read_samples(path)


def test_fixture_file_is_valid(fixture_csv):
    samples = read_samples(fixture_csv)
    assert len(samples) == 140
    assert {s.label for s in samples} == set(load_labels().names)


def test_labels_file():
    labels = load_labels()
    assert labels.names == ("open_palm", "fist", "thumbs_up", "peace", "point", "ok", "none")
    assert labels.id_of("ok") == 5
    assert labels.name_of(6) == "none"
    with pytest.raises(SchemaError):
        labels.id_of("wave")
    with pytest.raises(SchemaError):
        labels.check_pair("fist", 3)
