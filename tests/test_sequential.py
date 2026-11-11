"""Tests for the grouped sequential decoder (src/sequential)."""
import numpy as np
import pytest

from src.sequential import decode as D
from src.sequential import groups as G

rng = np.random.default_rng(1)


def rand_probs(n=50):
    p = rng.random((n, 34)) ** 4
    return p / p.sum(1, keepdims=True)


@pytest.mark.parametrize("name", ["H5", "G8", "A1"])
def test_anatomical_cover_all_gestures(name):
    gmap = G.ANATOMICAL[name]
    assert gmap.shape == (34,)
    assert set(gmap) == set(range(G.n_groups(gmap)))


def test_group_sizes():
    assert G.n_groups(G.ANATOMICAL["H5"]) == 5
    assert G.n_groups(G.ANATOMICAL["G8"]) == 8
    assert G.n_groups(G.ANATOMICAL["A1"]) == 22


def test_group_probs_sum_to_one():
    p = rand_probs()
    q = D.group_probs(p, G.ANATOMICAL["H5"])
    np.testing.assert_allclose(q.sum(1), 1.0)


def test_gate_above_one_is_flat():
    p = rand_probs()
    dec = D.sequential_decode(p, G.ANATOMICAL["H5"], tau_group=1.01)
    np.testing.assert_array_equal(dec.gesture, p.argmax(1))
    assert not dec.routed.any()


def test_singleton_groups_equal_flat():
    p = rand_probs()
    dec = D.sequential_decode(p, np.arange(34))
    np.testing.assert_array_equal(dec.gesture, p.argmax(1))


def test_seq_gesture_inside_chosen_group():
    p = rand_probs()
    gmap = G.ANATOMICAL["G8"]
    dec = D.sequential_decode(p, gmap)
    np.testing.assert_array_equal(gmap[dec.gesture], dec.group)


def test_trained_heads_are_used():
    p = rand_probs(10)
    gmap = G.ANATOMICAL["H5"]
    k = G.n_groups(gmap)
    pg = np.zeros((10, k)); pg[:, 4] = 1.0                    # always the pinch group
    mem = G.members(gmap)
    pw = [np.full((10, len(m)), 1 / len(m)) for m in mem]
    pw[4][:, 2] = 1.0                                          # third pinch, G34
    dec = D.sequential_decode(p, gmap, p_group=pg, p_within=pw)
    assert (dec.gesture == 33).all()


def test_severity_bounds_and_values():
    gmap = G.ANATOMICAL["H5"]
    y = np.array([0, 0, 0])
    yhat = np.array([0, 1, 5])        # correct, same group (finger ext), other group (wrist)
    assert D.severity_score(y, yhat, gmap) == pytest.approx(1 - (0 + 0.5 + 1) / 3)
    assert D.error_containment(y, yhat, gmap) == pytest.approx(0.5)


def test_confidence_weighted_vote_is_distribution():
    ps = [rand_probs(), rand_probs(), rand_probs()]
    v = D.confidence_weighted_vote(ps)
    np.testing.assert_allclose(v.sum(1), 1.0)


def test_cluster_groups_k():
    conf = rng.random((34, 34))
    for k in (5, 8):
        assert G.n_groups(G.cluster_groups(conf, k, 12)) == k


def test_accuracy_at_full_coverage():
    y = np.array([1, 2, 3, 4])
    yhat = np.array([1, 2, 0, 0])
    assert D.accuracy_at_coverage(y, yhat, np.array([.9, .8, .7, .6]), 1.0) == 0.5
    assert D.accuracy_at_coverage(y, yhat, np.array([.9, .8, .7, .6]), 0.5) == 1.0


# ---------------------------------------------------------------- trained heads (no TensorFlow needed)
from src.sequential import heads as Hd  # noqa: E402


def _fake_features(n_per=12, d=8, seed=0, gmap=None):
    """Clusters per gesture, with gestures of one group close together (as in real confusions)."""
    r = np.random.default_rng(seed)
    gmap = G.ANATOMICAL["H5"] if gmap is None else gmap
    centres = r.normal(size=(G.n_groups(gmap), d))[gmap] * 6 + r.normal(size=(34, d)) * 2
    y = np.repeat(np.arange(34), n_per)
    return centres[y] + r.normal(size=(len(y), d)), y


def test_grouped_heads_shapes_and_sums():
    f, y = _fake_features()
    gmap = G.ANATOMICAL["H5"]
    gh = Hd.GroupedHeads(gmap).fit(f, y, f, y)
    pg, pw = gh.proba(f)
    assert pg.shape == (len(f), 5) and pw.shape == (len(f), 34)
    np.testing.assert_allclose(pg.sum(1), 1.0, atol=1e-6)
    for idx in G.members(gmap):                     # each group's within-distribution sums to one
        np.testing.assert_allclose(pw[:, idx].sum(1), 1.0, atol=1e-6)
    dec = D.sequential_decode(pg @ np.eye(5)[gmap].T, gmap, p_group=pg, p_within=Hd.within_list(pw, gmap))
    assert (dec.gesture == y).mean() > 0.9          # separable fake data is decoded


def test_pick_shots_nested_and_per_gesture():
    y = np.repeat(np.arange(34), 6)
    s = Hd.pick_shots(y, 2, seed=0)
    assert set(s[1]) <= set(s[2])
    assert np.all(np.bincount(y[s[2]], minlength=34) == 2)


def test_trial_mean():
    p = np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 0.0]])
    t = np.array([0, 0, 1])
    np.testing.assert_allclose(Hd.trial_mean(p, t, 2), [[0.5, 0.5], [1.0, 0.0]])
