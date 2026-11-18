import numpy as np
import pytest

from gesturelink.data.io import read_samples
from gesturelink.data.normalize import (
    DegenerateHandError,
    mirror_x,
    normalize_batch,
    normalize_landmarks,
    palm_size,
)


@pytest.fixture
def hand():
    rng = np.random.default_rng(1)
    pts = rng.random((21, 3)).astype(np.float32)
    pts[9] = pts[0] + np.array([0.0, -0.2, 0.0])  # known palm size of 0.2
    return pts


def test_wrist_is_origin_and_palm_is_unit(hand):
    out = normalize_landmarks(hand).reshape(21, 3)
    np.testing.assert_allclose(out[0], 0, atol=1e-6)
    assert np.linalg.norm(out[9]) == pytest.approx(1.0, abs=1e-5)


def test_translation_invariant(hand):
    shifted = hand + np.array([0.3, -0.1, 0.05], dtype=np.float32)
    np.testing.assert_allclose(normalize_landmarks(hand), normalize_landmarks(shifted), atol=1e-5)


def test_scale_invariant(hand):
    np.testing.assert_allclose(
        normalize_landmarks(hand), normalize_landmarks(hand * 2.5), atol=1e-5
    )


def test_accepts_flat_input(hand):
    out = normalize_landmarks(hand.reshape(-1))
    assert out.shape == (63,) and out.dtype == np.float32


def test_palm_size(hand):
    assert palm_size(hand) == pytest.approx(0.2, abs=1e-6)


def test_degenerate_hand_raises(hand):
    hand[9] = hand[0]
    with pytest.raises(DegenerateHandError):
        normalize_landmarks(hand)


def test_batch_matches_single_and_flags_bad_rows(hand):
    bad = hand.copy()
    bad[9] = bad[0]
    nan_row = hand.copy()
    nan_row[3, 1] = np.nan
    batch = np.stack([hand.reshape(-1), bad.reshape(-1), nan_row.reshape(-1)])
    out, valid = normalize_batch(batch)
    assert valid.tolist() == [True, False, False]
    np.testing.assert_allclose(out[0], normalize_landmarks(hand), atol=1e-6)
    assert not out[1].any() and np.isfinite(out).all()


def test_mirror_x_is_involution(hand):
    feats = normalize_landmarks(hand)
    flipped = mirror_x(feats)
    np.testing.assert_allclose(flipped.reshape(21, 3)[:, 0], -feats.reshape(21, 3)[:, 0])
    np.testing.assert_allclose(mirror_x(flipped), feats)


def test_fixture_hands_all_normalize(fixture_csv):
    raw = np.stack([s.landmarks.reshape(-1) for s in read_samples(fixture_csv)])
    feats, valid = normalize_batch(raw)
    assert valid.all()
    # palm size is the unit, so the middle MCP always sits at distance 1
    mcp = feats.reshape(-1, 21, 3)[:, 9]
    np.testing.assert_allclose(np.linalg.norm(mcp, axis=1), 1.0, atol=1e-4)
