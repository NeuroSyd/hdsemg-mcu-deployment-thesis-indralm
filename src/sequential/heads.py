"""Trained grouped sequential classifier on a frozen flat network.

Design (one subject, one architecture)
--------------------------------------
The flat 34-class network trained on session 1 is kept as it is. Its
penultimate layer (the input of the final softmax layer) is used as a fixed
feature extractor. On these features three kinds of small linear heads are
trained, all on session 1 training trials only:

    group head        K-way softmax that predicts the gesture group
    specialist heads  one softmax per group with more than one gesture,
                      trained only on the windows of that group's gestures
    (the flat head)   the network's own 34-way output, kept as the fallback

A linear softmax head on frozen features is a single dense layer, so the
group head and all specialists together add a few thousand parameters to a
network that already fits on the microcontroller. The regularisation of each
head is chosen on the session 1 validation trials.

Adaptation conditions evaluated on session 2
--------------------------------------------
    s1stats   session 2 normalised with session 1 statistics (no calibration)
    cal{N}    label-free: the per-channel statistics are re-estimated from N
              unlabelled session 2 trials spread over the session, and every
              session 2 window is re-normalised with them. Because z-scoring is
              an affine map per channel, re-normalising the stored windows is
              equivalent to re-normalising the filtered signal (up to window
              overlap at trial edges).
    ll{k}     labelled baseline: the last layer is re-fitted on session 1
              training features plus k labelled session 2 trials per gesture
              (each group of shots weighted to match session 1 in total)

All conditions are scored on the same session 2 trials: every trial that
could be a labelled shot (k_max per gesture) is removed from scoring, so the
comparison between conditions is paired and fair.
"""
from __future__ import annotations

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from .groups import members, n_groups

C_GRID = (0.01, 0.1, 1.0, 10.0)


# --------------------------------------------------------------- feature extraction
def split_model(model):
    """Return a model with two outputs: penultimate features and the flat softmax."""
    import tensorflow as tf
    head = model.layers[-1]
    return tf.keras.Model(model.inputs, [head.input, head.output])


def trial_windows(path) -> np.ndarray:
    return np.load(path)["X"].astype(np.float32)


def channel_stats(paths) -> tuple[np.ndarray, np.ndarray]:
    """Per-channel mean and std over all samples of the given trials' windows."""
    s = ss = n = 0
    for p in paths:
        w = trial_windows(p)                       # (n_win, C, T)
        x = w.transpose(1, 0, 2).reshape(w.shape[1], -1).astype(np.float64)
        s = s + x.sum(1)
        ss = ss + np.square(x).sum(1)
        n += x.shape[1]
    mean = s / n
    std = np.sqrt(np.maximum(ss / n - mean ** 2, 0.0))
    return mean.astype(np.float32), std.astype(np.float32)


def extract(two_out, paths, renorm=None, batch=128):
    """Features and flat probabilities, per window, for a list of trials.

    renorm = (mean, std) re-normalises every window per channel first.
    Returns feats (n_win, d), probs (n_win, 34), trial_index (n_win,).
    """
    F, P, T = [], [], []
    for i, p in enumerate(paths):
        w = trial_windows(p)
        if renorm is not None:
            m, s = renorm
            w = (w - m[None, :, None]) / (s[None, :, None] + 1e-8)
        f, pr = two_out.predict(w, batch_size=batch, verbose=0)
        F.append(f.astype(np.float32)); P.append(pr.astype(np.float32)); T.append(np.full(len(w), i))
    return np.concatenate(F), np.concatenate(P), np.concatenate(T)


def trial_mean(prob, tid, n_trials):
    out = np.zeros((n_trials, prob.shape[1]), np.float64)
    for t in range(n_trials):
        out[t] = prob[tid == t].mean(0)
    return out


# --------------------------------------------------------------- linear heads
class Head:
    """Standardise features, then multinomial logistic regression (one dense layer)."""

    def __init__(self, classes):
        self.classes = np.asarray(classes)
        self.scaler = None
        self.lr = None
        self.C = None

    def fit(self, f_tr, y_tr, f_va=None, y_va=None, w_tr=None):
        self.scaler = StandardScaler().fit(f_tr)
        xt = self.scaler.transform(f_tr)
        best = (-1.0, None, None)
        grid = C_GRID if f_va is not None and len(np.unique(y_tr)) > 1 else (1.0,)
        for c in grid:
            lr = LogisticRegression(C=c, max_iter=2000)
            lr.fit(xt, y_tr, sample_weight=w_tr)
            if f_va is None:
                best = (0.0, c, lr)
                break
            acc = (lr.predict(self.scaler.transform(f_va)) == y_va).mean()
            if acc > best[0]:
                best = (acc, c, lr)
        _, self.C, self.lr = best
        return self

    def proba(self, f):
        """Probabilities over self.classes (columns in that order)."""
        p = self.lr.predict_proba(self.scaler.transform(f))
        out = np.zeros((len(f), len(self.classes)))
        cols = [int(np.flatnonzero(self.classes == c)[0]) for c in self.lr.classes_]
        out[:, cols] = p
        return out

    def n_params(self):
        return int(self.lr.coef_.size + self.lr.intercept_.size)


class GroupedHeads:
    """Group head plus one specialist per multi-gesture group."""

    def __init__(self, gmap):
        self.gmap = np.asarray(gmap)
        self.mem = members(self.gmap)
        self.group = None
        self.spec = {}

    def fit(self, f_tr, y_tr, f_va, y_va):
        k = n_groups(self.gmap)
        self.group = Head(np.arange(k)).fit(f_tr, self.gmap[y_tr], f_va, self.gmap[y_va])
        for g, idx in enumerate(self.mem):
            if len(idx) < 2:
                continue
            mtr, mva = np.isin(y_tr, idx), np.isin(y_va, idx)
            if len(np.unique(y_tr[mtr])) < 2:
                continue
            self.spec[g] = Head(idx).fit(f_tr[mtr], y_tr[mtr],
                                         f_va[mva] if mva.any() else None,
                                         y_va[mva] if mva.any() else None)
        return self

    def proba(self, f):
        """Returns p_group (n, K) and p_within_full (n, 34): each group's
        specialist distribution placed at its member gestures."""
        pg = self.group.proba(f)
        pw = np.zeros((len(f), len(self.gmap)))
        for g, idx in enumerate(self.mem):
            if g in self.spec:
                pw[:, idx] = self.spec[g].proba(f)
            else:
                pw[:, idx] = 1.0 / len(idx)
        return pg, pw

    def n_params(self):
        return self.group.n_params() + sum(h.n_params() for h in self.spec.values())


def within_list(pw_full, gmap):
    """Convert the (n, 34) within-group array to the list format used by decode.sequential_decode."""
    return [pw_full[:, idx] for idx in members(np.asarray(gmap))]


def pick_shots(y_trials, k, seed=0):
    """Indices of k trials per gesture (seeded). Returns a dict k_i -> indices for k_i = 1..k,
    nested so the 1-shot set is contained in the 2-shot set."""
    rng = np.random.default_rng(seed)
    order = {}
    for g in np.unique(y_trials):
        idx = np.flatnonzero(y_trials == g)
        order[g] = rng.permutation(idx)
    return {ki: np.sort(np.concatenate([order[g][:ki] for g in order])) for ki in range(1, k + 1)}
