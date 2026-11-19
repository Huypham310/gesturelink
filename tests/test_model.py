import json

import numpy as np
import pytest
import torch

from gesturelink.cli import main
from gesturelink.data.io import read_samples
from gesturelink.data.prepare import prepare_dataset
from gesturelink.data.schema import FEATURE_DIM
from gesturelink.data.synthesize import synthesize_csv
from gesturelink.model.dataset import LandmarkDataset, load_splits
from gesturelink.model.eval import confusion_matrix, per_class_report
from gesturelink.model.net import GestureNet
from gesturelink.model.predict import GesturePredictor
from gesturelink.model.train import TrainConfig, load_checkpoint, load_config, train


@pytest.fixture(scope="module")
def trained(tmp_path_factory, fixture_csv, labels):
    """Train once on the 140-row fixture (80 epochs, well under a few seconds on CPU)."""
    root = tmp_path_factory.mktemp("model")
    prepare_dataset(fixture_csv, root / "processed", labels)
    cfg = TrainConfig(data=str(root / "processed" / "dataset.npz"), out_dir=str(root / "out"))
    metrics = train(cfg, labels, verbose=False)
    return root, cfg, metrics


def test_net_shapes_and_size():
    net = GestureNet()
    out = net(torch.zeros(5, FEATURE_DIM))
    assert out.shape == (5, 7)
    # (63*128+128) + (128*64+64) + (64*7+7)
    assert net.num_parameters() == 8_192 + 8_256 + 455
    assert net.arch()["hidden"] == [128, 64]


def test_net_rejects_bad_dropout():
    with pytest.raises(ValueError):
        GestureNet(dropout=1.0)


def test_dataset_checks_shapes():
    with pytest.raises(ValueError):
        LandmarkDataset(np.zeros((4, 10)), np.zeros(4))
    with pytest.raises(ValueError):
        LandmarkDataset(np.zeros((4, FEATURE_DIM)), np.zeros(3))


def test_augmentation_only_when_enabled():
    x = np.random.default_rng(0).normal(size=(3, FEATURE_DIM)).astype(np.float32)
    plain = LandmarkDataset(x, np.zeros(3))
    noisy = LandmarkDataset(x, np.zeros(3), noise_std=0.05)
    flipped = LandmarkDataset(x, np.zeros(3), mirror_prob=1.0)
    assert torch.equal(plain[0][0], torch.from_numpy(x[0]))
    assert not torch.equal(noisy[0][0], plain[0][0])
    # mirror only negates the x coordinates (every third value)
    assert torch.allclose(flipped[0][0][0::3], -plain[0][0][0::3])
    assert torch.allclose(flipped[0][0][1::3], plain[0][0][1::3])


def test_confusion_and_report():
    y_true = np.array([0, 0, 1, 1, 2])
    y_pred = np.array([0, 1, 1, 1, 0])
    cm = confusion_matrix(y_true, y_pred, 3)
    assert cm.tolist() == [[1, 1, 0], [0, 2, 0], [1, 0, 0]]
    rep = per_class_report(cm, ["a", "b", "c"])
    assert rep["a"]["precision"] == 0.5 and rep["a"]["recall"] == 0.5
    assert rep["b"]["precision"] == pytest.approx(0.6667, abs=1e-4) and rep["b"]["recall"] == 1.0
    assert rep["c"]["f1"] == 0.0 and rep["c"]["support"] == 1  # never predicted, no crash


def test_train_on_fixture_beats_chance(trained):
    _, _, metrics = trained
    # chance is 1/7 = 0.14; clean synthetic data should be nearly separable
    assert metrics["accuracy"] >= 0.85
    assert metrics["num_samples"] == 28 and metrics["num_train"] == 112
    assert np.array(metrics["confusion_matrix"]).sum() == 28


def test_train_writes_checkpoint_and_metrics(trained):
    root, _, metrics = trained
    assert (root / "out" / "gesturelink.pt").exists()
    saved = json.loads((root / "out" / "metrics.json").read_text())
    assert saved["accuracy"] == metrics["accuracy"]
    assert saved["labels"][0] == "open_palm" and len(saved["history"]) == saved["epochs_run"]


def test_checkpoint_roundtrip(trained, labels):
    root, cfg, _ = trained
    model, ckpt = load_checkpoint(root / "out" / "gesturelink.pt")
    assert ckpt["labels"] == list(labels.names)
    assert ckpt["config"]["hidden"] == cfg.hidden
    assert not model.training
    # reloading must give the exact same logits
    again, _ = load_checkpoint(root / "out" / "gesturelink.pt")
    x = torch.randn(4, FEATURE_DIM)
    assert torch.equal(model(x), again(x))


def test_predictor_on_raw_landmarks(trained, fixture_csv):
    root, _, _ = trained
    predictor = GesturePredictor(root / "out" / "gesturelink.pt")
    samples = read_samples(fixture_csv)
    preds = predictor.predict_batch([s.landmarks for s in samples])
    hits = sum(p.label == s.label for p, s in zip(preds, samples, strict=True))
    assert hits / len(samples) >= 0.85  # includes training rows, so a loose bound
    one = predictor.predict(samples[0].landmarks)
    assert one is not None and one.label in predictor.labels
    assert sum(one.probabilities.values()) == pytest.approx(1.0, abs=1e-2)
    assert one.confidence == pytest.approx(max(one.probabilities.values()), abs=1e-3)


def test_predictor_ignores_hand_position_and_size(trained, fixture_csv):
    root, _, _ = trained
    predictor = GesturePredictor(root / "out" / "gesturelink.pt")
    lm = read_samples(fixture_csv)[30].landmarks
    moved = lm * 0.5 + 0.1  # smaller hand, different place in the frame
    assert predictor.predict(lm).label == predictor.predict(moved).label


def test_predictor_degenerate_hand_gives_none(trained, fixture_csv):
    root, _, _ = trained
    predictor = GesturePredictor(root / "out" / "gesturelink.pt")
    flat = read_samples(fixture_csv)[0].landmarks.copy()
    flat[9] = flat[0]  # no palm size
    assert predictor.predict(flat) is None


def test_more_data_gives_high_accuracy(tmp_path, labels):
    synthesize_csv(tmp_path / "big.csv", labels, per_class=120, seed=3)
    prepare_dataset(tmp_path / "big.csv", tmp_path / "p", labels)
    cfg = TrainConfig(
        data=str(tmp_path / "p" / "dataset.npz"), out_dir=str(tmp_path / "o"), epochs=40
    )
    metrics = train(cfg, labels, verbose=False)
    assert metrics["accuracy"] >= 0.95


def test_training_is_reproducible(tmp_path, fixture_csv, labels):
    prepare_dataset(fixture_csv, tmp_path / "p", labels)
    accs = []
    for run in ("a", "b"):
        cfg = TrainConfig(
            data=str(tmp_path / "p" / "dataset.npz"), out_dir=str(tmp_path / run), epochs=10
        )
        accs.append(train(cfg, labels, verbose=False)["history"])
    assert accs[0] == accs[1]


def test_load_config_defaults_and_overrides():
    cfg = load_config()
    assert cfg.hidden == [128, 64] and cfg.seed == 42
    assert load_config(epochs=5, lr=None).epochs == 5


def test_load_config_rejects_unknown_key(tmp_path):
    bad = tmp_path / "bad.json"
    bad.write_text(json.dumps({"epochs": 3, "typo_key": 1}))
    with pytest.raises(ValueError):
        load_config(bad)


def test_missing_dataset_and_checkpoint(tmp_path, labels):
    with pytest.raises(FileNotFoundError):
        train(TrainConfig(data=str(tmp_path / "nope.npz")), labels, verbose=False)
    with pytest.raises(FileNotFoundError):
        load_checkpoint(tmp_path / "nope.pt")


def test_load_splits_needs_all_arrays(tmp_path):
    np.savez(tmp_path / "bad.npz", X_train=np.zeros((2, FEATURE_DIM)))
    with pytest.raises(ValueError):
        load_splits(tmp_path / "bad.npz")


def test_cli_train_and_predict(tmp_path, fixture_csv, capsys):
    assert main(["prepare", "--input", str(fixture_csv), "--out-dir", str(tmp_path / "p")]) == 0
    code = main([
        "train", "--data", str(tmp_path / "p" / "dataset.npz"),
        "--out-dir", str(tmp_path / "m"), "--epochs", "30",
    ])
    assert code == 0 and (tmp_path / "m" / "gesturelink.pt").exists()
    capsys.readouterr()
    code = main([
        "predict", "--input", str(fixture_csv),
        "--checkpoint", str(tmp_path / "m" / "gesturelink.pt"), "--limit", "20",
    ])
    assert code == 0
    assert "correct" in capsys.readouterr().out
