"""Gesture groupings for the 34 Hyser PR gestures.

A grouping is an integer array of length 34 (index 0 is gesture G1). Gestures
that share a value belong to the same group. Group ids are made contiguous
(0..K-1) by `as_map`.

Anatomical groupings are fixed before any data are seen:
    H5  five groups by the joint that moves (finger extensions, wrist,
        wrist plus hand, whole hand, pinch). This is the five-group layout of
        the hand-drawn sequential classifier.
    G8  the eight families of the Hyser gesture list (Table G1 of the weekly
        reports).
    A1  each wrist movement joined with its "plus hand close" and "plus hand
        open" versions (22 groups, the grouping already used for retraining).

Data-driven groupings are learned from predictions:
    soft_confusion()  : mean predicted probability matrix (true class x class)
    cluster_groups()  : average-linkage clustering of that matrix into K groups
The caller decides which data the matrix comes from. The evaluation script
only uses session 1 data, or session 2 data of OTHER subjects, so the target
subject's test session never shapes its own groups.
"""
from __future__ import annotations

import numpy as np

N_GESTURES = 34

GESTURE_NAMES = [
    "Thumb ext", "Index ext", "Middle ext", "Ring ext", "Little ext",
    "Wrist flex", "Wrist ext", "Wrist radial", "Wrist ulnar", "Wrist pronation",
    "Wrist supination", "Thumb+index ext", "Index+middle ext",
    "Wrist flex + close", "Wrist ext + close", "Wrist radial + close",
    "Wrist ulnar + close", "Wrist pron + close", "Wrist sup + close",
    "Wrist flex + open", "Wrist ext + open", "Wrist radial + open",
    "Wrist ulnar + open", "Wrist pron + open", "Wrist sup + open",
    "Thumb+index+middle ext", "Index+middle+ring ext", "Middle+ring+little ext",
    "Index to little ext", "Hand close", "Hand open",
    "Thumb+index pinch", "Thumb+index+middle pinch", "Thumb+middle pinch",
]


def _from_ranges(spec: list[tuple[list[int], int]]) -> np.ndarray:
    """spec: list of (gesture numbers 1..34, group id)."""
    mp = np.full(N_GESTURES, -1, int)
    for gestures, gid in spec:
        for g in gestures:
            mp[g - 1] = gid
    assert (mp >= 0).all(), "every gesture needs a group"
    return mp


def r(a: int, b: int) -> list[int]:
    return list(range(a, b + 1))


H5_NAMES = ["Finger extension", "Wrist", "Wrist + hand", "Whole hand", "Pinch"]
H5 = _from_ranges([
    (r(1, 5) + [12, 13] + r(26, 29), 0),   # finger extensions
    (r(6, 11), 1),                          # single wrist movements
    (r(14, 25), 2),                         # wrist movement with hand close or open
    ([30, 31], 3),                          # hand close, hand open
    (r(32, 34), 4),                         # pinches
])

G8_NAMES = ["Single finger ext", "Wrist", "Two finger ext", "Wrist + close",
            "Wrist + open", "Three/four finger ext", "Hand close/open", "Pinch"]
G8 = _from_ranges([
    (r(1, 5), 0), (r(6, 11), 1), ([12, 13], 2), (r(14, 19), 3),
    (r(20, 25), 4), (r(26, 29), 5), ([30, 31], 6), (r(32, 34), 7),
])


def a1() -> np.ndarray:
    mp = list(range(N_GESTURES))
    for i in range(13, 19):
        mp[i] = i - 8      # G14-19 -> G6-11
    for i in range(19, 25):
        mp[i] = i - 14     # G20-25 -> G6-11
    return as_map(mp)


def as_map(raw) -> np.ndarray:
    """Make group ids contiguous, 0..K-1, in order of first appearance."""
    raw = np.asarray(raw, int)
    assert raw.shape == (N_GESTURES,)
    _, first = np.unique(raw, return_index=True)
    order = raw[np.sort(first)]
    lut = {g: i for i, g in enumerate(order)}
    return np.array([lut[g] for g in raw], int)


ANATOMICAL = {"H5": as_map(H5), "G8": as_map(G8), "A1": a1()}


def n_groups(gmap: np.ndarray) -> int:
    return int(gmap.max()) + 1


def members(gmap: np.ndarray) -> list[np.ndarray]:
    return [np.flatnonzero(gmap == g) for g in range(n_groups(gmap))]


def soft_confusion(y_true_list, prob_list) -> np.ndarray:
    """Row-normalised mean probability matrix: M[i, j] = mean P(j | true i)."""
    m = np.zeros((N_GESTURES, N_GESTURES))
    n = np.zeros(N_GESTURES)
    for y, p in zip(y_true_list, prob_list):
        for yi, pi in zip(y, p):
            m[yi] += pi
            n[yi] += 1
    n[n == 0] = 1
    return m / n[:, None]


def cluster_groups(conf: np.ndarray, k: int, max_size: int | None = None) -> np.ndarray:
    """Average-linkage agglomerative clustering on the symmetric confusion.

    Similarity between two groups is the mean of the symmetric confusion over
    all cross pairs. The pair with the highest similarity is merged until k
    groups remain. `max_size` stops one group from absorbing everything, which
    happens when one gesture is a common destination of errors.
    """
    s = conf + conf.T
    np.fill_diagonal(s, 0.0)
    groups = [[i] for i in range(N_GESTURES)]
    while len(groups) > k:
        best, pair = -1.0, None
        for a in range(len(groups)):
            for b in range(a + 1, len(groups)):
                if max_size and len(groups[a]) + len(groups[b]) > max_size:
                    continue
                v = s[np.ix_(groups[a], groups[b])].mean()
                if v > best:
                    best, pair = v, (a, b)
        if pair is None:  # size cap blocks every merge
            break
        a, b = pair
        groups[a] += groups[b]
        del groups[b]
    mp = np.zeros(N_GESTURES, int)
    for gi, g in enumerate(groups):
        mp[g] = gi
    return as_map(mp)


def describe(gmap: np.ndarray, names: list[str] | None = None) -> list[str]:
    out = []
    for g, idx in enumerate(members(gmap)):
        label = names[g] if names else f"Group {g + 1}"
        out.append(f"{label}: " + ", ".join(f"G{i + 1}" for i in idx))
    return out
