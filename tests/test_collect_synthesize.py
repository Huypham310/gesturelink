import numpy as np
import pytest

from gesturelink.cli import main
from gesturelink.data.collect import collect_samples
from gesturelink.data.io import SampleWriter, read_samples
from gesturelink.data.normalize import normalize_batch
from gesturelink.data.prepare import load_dataset
from gesturelink.data.schema import SchemaError
from gesturelink.data.synthesize import make_pose, synthesize_csv, synthesize_samples

# ---------- synthetic generator ----------

def test_synthesize_is_balanced_and_deterministic(labels):
    a = synthesize_samples(labels, per_class=5, seed=3)
    b = synthesize_samples(labels, per_class=5, seed=3)
    assert len(a) == 35
    assert [s.label for s in a].count("fist") == 5
    for x, y in zip(a, b, strict=True):
        np.testing.assert_array_equal(x.landmarks, y.landmarks)


def test_synthetic_landmarks_look_like_mediapipe_output():
    rng = np.random.default_rng(0)
    for gesture in ("open_palm", "fist", "ok"):
        pts, handedness = make_pose(gesture, rng)
        assert pts.shape == (21, 3)
        assert handedness in ("Left", "Right")
        assert pts[:, :2].min() >= 0.0 and pts[:, :2].max() <= 1.0


def test_synthetic_gestures_are_separable(labels):
    """Nearest-centroid on normalized features should already do well, else the recipe is broken."""
    samples = synthesize_samples(labels, per_class=60, seed=11)
    feats, valid = normalize_batch(np.stack([s.landmarks.reshape(-1) for s in samples]))
    assert valid.all()
    # mirror left hands so both sides share one centroid
    y = np.array([s.label_id for s in samples])
    left = np.array([s.handedness == "Left" for s in samples])
    feats = feats.reshape(-1, 21, 3).copy()
    feats[left, :, 0] *= -1
    feats = feats.reshape(-1, 63)

    idx = np.arange(len(y))
    train, test = idx % 3 != 0, idx % 3 == 0
    centroids = np.stack([feats[train & (y == c)].mean(axis=0) for c in range(7)])
    dist = ((feats[test][:, None, :] - centroids[None]) ** 2).sum(axis=2)
    accuracy = (dist.argmin(axis=1) == y[test]).mean()
    assert accuracy > 0.85


def test_fist_tips_are_closer_to_wrist_than_open_palm(labels):
    samples = synthesize_samples(labels, per_class=20, seed=5)
    feats, _ = normalize_batch(np.stack([s.landmarks.reshape(-1) for s in samples]))
    feats = feats.reshape(-1, 21, 3)
    tips = np.linalg.norm(feats[:, [8, 12, 16, 20]], axis=2).mean(axis=1)
    by_label = {n: tips[[s.label == n for s in samples]].mean() for n in labels.names}
    assert by_label["fist"] < by_label["open_palm"] - 0.5


def test_synthesize_csv_overwrites(labels, tmp_path):
    path = tmp_path / "syn.csv"
    synthesize_csv(path, labels, per_class=3, seed=1)
    n = synthesize_csv(path, labels, per_class=2, seed=1)
    assert n == 14
    assert len(read_samples(path)) == 14


# ---------- offline collect (fake frames + fake detector, no camera) ----------

def fake_frames(n, step_ms=33):
    frame = np.zeros((4, 4, 3), dtype=np.uint8)
    return [(i * step_ms, frame) for i in range(n)]


def fake_detector(landmarks, every=1):
    calls = {"n": 0}

    def detect(_frame):
        calls["n"] += 1
        if calls["n"] % every:
            return None  # simulate frames without a hand
        return landmarks, "Right"

    return detect


@pytest.fixture
def a_hand(labels):
    return synthesize_samples(labels, per_class=1, seed=0)[0].landmarks


def test_collect_saves_with_interval(labels, a_hand, tmp_path):
    path = tmp_path / "s.csv"
    with SampleWriter(path) as writer:
        stats = collect_samples(
            fake_frames(30), fake_detector(a_hand), writer, "peace", labels, "sess",
            max_samples=100, min_interval_ms=100,
        )
    rows = read_samples(path)
    assert stats.frames_seen == 30 and stats.frames_with_hand == 30
    # 33 ms frames -> a sample every 4th frame (132 ms): frames 0, 4, ..., 28
    assert stats.saved == len(rows) == 8
    assert all(r.label == "peace" and r.label_id == 3 and r.session_id == "sess" for r in rows)
    gaps = np.diff([r.timestamp_ms for r in rows])
    assert (gaps >= 100).all()


def test_collect_stops_at_max_samples(labels, a_hand, tmp_path):
    with SampleWriter(tmp_path / "s.csv") as writer:
        stats = collect_samples(
            fake_frames(50), fake_detector(a_hand), writer, "fist", labels, "s",
            max_samples=5, min_interval_ms=0,
        )
    assert stats.saved == 5 and stats.frames_seen == 5


def test_collect_skips_frames_without_hand(labels, a_hand, tmp_path):
    with SampleWriter(tmp_path / "s.csv") as writer:
        stats = collect_samples(
            fake_frames(20), fake_detector(a_hand, every=4), writer, "ok", labels, "s",
            max_samples=100, min_interval_ms=0,
        )
    assert stats.frames_with_hand == 5 and stats.saved == 5


def test_collect_callback_can_stop_early(labels, a_hand, tmp_path):
    with SampleWriter(tmp_path / "s.csv") as writer:
        stats = collect_samples(
            fake_frames(50), fake_detector(a_hand), writer, "ok", labels, "s",
            min_interval_ms=0, on_frame=lambda f, d, saved: saved < 3,
        )
    assert stats.saved == 3


def test_collect_rejects_unknown_label(labels, a_hand, tmp_path):
    with SampleWriter(tmp_path / "s.csv") as writer, pytest.raises(SchemaError):
        collect_samples(fake_frames(3), fake_detector(a_hand), writer, "wave", labels, "s")


# ---------- CLI, end to end without a camera ----------

def test_cli_synthesize_then_prepare(tmp_path, capsys):
    raw = tmp_path / "raw" / "syn.csv"
    assert main(["synthesize", "--out", str(raw), "--per-class", "10", "--seed", "1"]) == 0
    out_dir = tmp_path / "processed"
    assert main(["prepare", "--input", str(raw.parent), "--out-dir", str(out_dir)]) == 0
    text = capsys.readouterr().out
    assert "70 samples" in text
    data = load_dataset(out_dir / "dataset.npz")
    assert data["X_train"].shape == (56, 63) and data["X_val"].shape == (14, 63)
    assert (out_dir / "meta.json").exists()


def test_cli_download_model_from_local_url(tmp_path, capsys):
    source = tmp_path / "fake.task"
    source.write_bytes(b"not a real model" * 10)
    dest = tmp_path / "models" / "hand_landmarker.task"
    assert main(["download-model", "--dest", str(dest), "--url", source.as_uri()]) == 0
    assert dest.read_bytes() == source.read_bytes()
    assert not dest.with_suffix(".task.part").exists()

    # second call is a no-op unless --force
    source.write_bytes(b"new")
    main(["download-model", "--dest", str(dest), "--url", source.as_uri()])
    assert dest.stat().st_size == 160
    main(["download-model", "--dest", str(dest), "--url", source.as_uri(), "--force"])
    assert dest.read_bytes() == b"new"


def test_hand_extractor_needs_model_file(tmp_path):
    from gesturelink.data.landmarks import HandExtractor

    with pytest.raises(FileNotFoundError, match="download-model"):
        HandExtractor(tmp_path / "missing.task")
