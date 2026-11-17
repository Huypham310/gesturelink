import json
import shutil

import numpy as np
import pytest

from gesturelink.data.io import SampleWriter, read_samples
from gesturelink.data.prepare import (
    load_dataset,
    load_raw,
    prepare_dataset,
    stratified_split,
)
from gesturelink.data.schema import Sample, SchemaError


def test_stratified_split_keeps_class_ratio():
    y = np.repeat(np.arange(3), [50, 30, 10])
    train, val = stratified_split(y, 0.2, seed=42)
    assert len(set(train) & set(val)) == 0
    assert len(train) + len(val) == len(y)
    assert np.bincount(y[val]).tolist() == [10, 6, 2]


def test_split_is_deterministic_and_seed_dependent():
    y = np.repeat(np.arange(4), 25)
    a = stratified_split(y, 0.2, seed=42)
    b = stratified_split(y, 0.2, seed=42)
    c = stratified_split(y, 0.2, seed=1)
    assert np.array_equal(a[1], b[1])
    assert not np.array_equal(a[1], c[1])


def test_tiny_classes_stay_usable():
    y = np.array([0, 0, 0, 1, 1, 2])
    train, val = stratified_split(y, 0.2, seed=0)
    assert 2 in y[train] and 2 not in y[val]  # single sample -> train only
    assert 1 in y[val] and 1 in y[train]  # two samples -> one each


def test_split_rejects_bad_fraction():
    with pytest.raises(ValueError):
        stratified_split(np.arange(10), 1.5)


def test_prepare_fixture(fixture_csv, labels, tmp_path):
    meta = prepare_dataset(fixture_csv, tmp_path, labels)
    data = load_dataset(tmp_path / "dataset.npz")

    assert data["X_train"].shape == (112, 63) and data["X_val"].shape == (28, 63)
    assert data["X_train"].dtype == np.float32
    assert data["y_train"].shape == (112,)
    # 20 per class -> 16 train / 4 val each
    assert set(meta["counts"]["train"].values()) == {16}
    assert set(meta["counts"]["val"].values()) == {4}
    assert np.bincount(data["y_val"], minlength=7).tolist() == [4] * 7

    saved = json.loads((tmp_path / "meta.json").read_text())
    assert saved["split"] == {"val_fraction": 0.2, "seed": 42, "stratified": True}
    assert saved["labels"] == list(labels.names)
    assert saved["feature_dim"] == 63
    assert saved["source_files"] == ["samples.csv"]
    assert saved["num_dropped_degenerate"] == 0


def test_prepare_is_reproducible(fixture_csv, labels, tmp_path):
    prepare_dataset(fixture_csv, tmp_path / "a", labels)
    prepare_dataset(fixture_csv, tmp_path / "b", labels)
    a = load_dataset(tmp_path / "a" / "dataset.npz")
    b = load_dataset(tmp_path / "b" / "dataset.npz")
    for key in a:
        assert np.array_equal(a[key], b[key])


def test_prepare_reads_a_directory(fixture_csv, labels, tmp_path):
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    shutil.copy(fixture_csv, raw_dir / "one.csv")
    shutil.copy(fixture_csv, raw_dir / "two.csv")
    meta = prepare_dataset(raw_dir, tmp_path / "out", labels)
    assert meta["num_samples"] == 280
    assert meta["source_files"] == ["one.csv", "two.csv"]


def test_prepare_drops_degenerate_rows(fixture_csv, labels, tmp_path):
    samples = read_samples(fixture_csv)[:14]
    flat = samples[0].landmarks.copy()
    flat[9] = flat[0]  # wrist == middle MCP, no palm size
    samples.append(Sample("fist", 1, "bad", 0, "Left", flat))
    path = tmp_path / "raw.csv"
    with SampleWriter(path) as w:
        for s in samples:
            w.write(s)
    meta = prepare_dataset(path, tmp_path / "out", labels)
    assert meta["num_dropped_degenerate"] == 1
    assert meta["num_samples"] == 14


def test_mirror_left_flips_x_for_left_hands(fixture_csv, labels, tmp_path):
    plain = prepare_dataset(fixture_csv, tmp_path / "plain", labels)
    mirrored = prepare_dataset(fixture_csv, tmp_path / "mirror", labels, mirror_left=True)
    assert mirrored["mirror_left"] is True and plain["mirror_left"] is False
    a = load_dataset(tmp_path / "plain" / "dataset.npz")["X_train"]
    b = load_dataset(tmp_path / "mirror" / "dataset.npz")["X_train"]
    assert not np.array_equal(a, b)
    # y and z never change, only x
    np.testing.assert_allclose(a.reshape(-1, 21, 3)[:, :, 1:], b.reshape(-1, 21, 3)[:, :, 1:])


def test_label_id_mismatch_is_rejected(fixture_csv, labels, tmp_path):
    s = read_samples(fixture_csv)[0]
    wrong = Sample(s.label, (s.label_id + 1) % 7, "x", 0, s.handedness, s.landmarks)
    path = tmp_path / "wrong.csv"
    with SampleWriter(path) as w:
        w.write(wrong)
    with pytest.raises(SchemaError):
        load_raw(path, labels)


def test_missing_input(labels, tmp_path):
    with pytest.raises(FileNotFoundError):
        prepare_dataset(tmp_path / "nope", tmp_path / "out", labels)
