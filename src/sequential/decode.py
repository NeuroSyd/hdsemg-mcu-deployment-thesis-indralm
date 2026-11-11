"""Arbiter, two-stage decoding and evaluation metrics.

All functions work on probability matrices of shape (n_trials, 34) so the same
code serves the offline study (stored flat-model probabilities) and the
trained version (a group head and within-group specialists).

Two-stage decoding with a confidence gate
-----------------------------------------
Stage 1 picks the group with the largest group probability. When a trained
group head exists its output is used; otherwise the group probability is the
sum of the flat class probabilities inside the group.
    if stage 1 confidence >= tau_group:
        Stage 2 picks the gesture inside the chosen group (specialist output,
        or the flat probabilities restricted to the group and renormalised)
    else:
        fall back to the flat 34-class decision
A second threshold, tau_gesture, lets the decoder stop at the group level when
the within-group decision is unsure. Such a trial is "deferred": the group is
committed but the gesture is not, which is what a device would do by holding
the last safe command of that group.

Severity-aware score
--------------------
A correct gesture costs 0, a wrong gesture inside the true group costs
`within_cost` (0.5 by default) and a wrong group costs 1. The score is
1 - mean cost. It follows the lowest-common-ancestor idea of mistake severity
of Bertinetto et al. (CVPR 2020) on a two-level hierarchy. It is offered as
one reading of the "gradient of incorrectness" suggested by the supervisor.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .groups import members, n_groups


# ---------------------------------------------------------------- arbiter
def soft_vote(probs: list[np.ndarray]) -> np.ndarray:
    """Mean probability over networks."""
    return np.mean(np.stack(probs), axis=0)


def confidence_weighted_vote(probs: list[np.ndarray]) -> np.ndarray:
    """Arbiter: each network's vote is weighted by its own top probability,
    so a confident network outweighs an unsure one on that trial."""
    st = np.stack(probs)                       # (m, n, c)
    w = st.max(axis=2, keepdims=True)          # (m, n, 1)
    out = (w * st).sum(0) / w.sum(0)
    return out / out.sum(1, keepdims=True)


# ---------------------------------------------------------------- decoding
def group_probs(p: np.ndarray, gmap: np.ndarray) -> np.ndarray:
    """Sum of class probabilities inside each group, shape (n, K)."""
    k = n_groups(gmap)
    q = np.zeros((p.shape[0], k))
    for g, idx in enumerate(members(gmap)):
        q[:, g] = p[:, idx].sum(1)
    return q


@dataclass
class Decision:
    gesture: np.ndarray        # final gesture (flat id), always defined
    group: np.ndarray          # committed group
    conf_group: np.ndarray     # stage 1 confidence
    conf_gesture: np.ndarray   # within-group confidence of the final gesture
    routed: np.ndarray         # True if the two-stage path was used (not the fallback)
    deferred: np.ndarray       # True if the decoder stopped at the group level


def sequential_decode(p_flat: np.ndarray, gmap: np.ndarray,
                      tau_group: float = 0.0, tau_gesture: float = 0.0,
                      p_group: np.ndarray | None = None,
                      p_within: list[np.ndarray] | None = None) -> Decision:
    """Two-stage decision.

    p_flat   : (n, 34) flat probabilities, used for the fallback and, when no
               trained heads are given, to derive both stages.
    p_group  : optional (n, K) output of a trained group head.
    p_within : optional list of K arrays, (n, |group k|), outputs of trained
               within-group specialists, columns ordered as members(gmap)[k].
    """
    n = p_flat.shape[0]
    mem = members(gmap)
    q = group_probs(p_flat, gmap) if p_group is None else p_group
    g_hat = q.argmax(1)
    c_group = q[np.arange(n), g_hat]
    routed = c_group >= tau_group

    flat_hat = p_flat.argmax(1)
    gesture = flat_hat.copy()
    conf_g = np.zeros(n)
    for i in range(n):
        idx = mem[g_hat[i]]
        if p_within is None:
            w = p_flat[i, idx]
            s = w.sum()
            w = w / s if s > 0 else np.full(len(idx), 1 / len(idx))
        else:
            w = p_within[g_hat[i]][i]
        j = int(np.argmax(w))
        if routed[i]:
            gesture[i] = idx[j]
            conf_g[i] = w[j]
        else:
            conf_g[i] = p_flat[i, flat_hat[i]]
    group = np.where(routed, g_hat, gmap[gesture])
    deferred = routed & (conf_g < tau_gesture)
    return Decision(gesture, group, c_group, conf_g, routed, deferred)


# ---------------------------------------------------------------- metrics
def gesture_accuracy(y, yhat) -> float:
    return float(np.mean(np.asarray(y) == np.asarray(yhat)))


def group_accuracy(y, group_hat, gmap) -> float:
    return float(np.mean(gmap[np.asarray(y)] == np.asarray(group_hat)))


def error_containment(y, yhat, gmap) -> float:
    """Share of gesture errors whose prediction lies in the true group."""
    y, yhat = np.asarray(y), np.asarray(yhat)
    wrong = y != yhat
    if not wrong.any():
        return float("nan")
    return float(np.mean(gmap[y[wrong]] == gmap[yhat[wrong]]))


def severity_score(y, yhat, gmap, within_cost: float = 0.5,
                   group_hat=None, deferred=None) -> float:
    """1 - mean mistake cost on a two-level hierarchy.

    A deferred trial (group committed, gesture withheld) costs `within_cost`
    when the group is right and 1 when it is wrong.
    """
    y, yhat = np.asarray(y), np.asarray(yhat)
    cost = np.where(y == yhat, 0.0, np.where(gmap[y] == gmap[yhat], within_cost, 1.0))
    if deferred is not None and group_hat is not None:
        d = np.asarray(deferred)
        gh = np.asarray(group_hat)
        cost = np.where(d, np.where(gmap[y] == gh, within_cost, 1.0), cost)
    return float(1.0 - cost.mean())


def risk_coverage(y, yhat, conf, taus) -> tuple[np.ndarray, np.ndarray]:
    """Selective classification: keep trials with conf >= tau.

    Returns (coverage, accuracy on kept trials) for each tau. Accuracy is NaN
    when nothing is kept.
    """
    y, yhat, conf = map(np.asarray, (y, yhat, conf))
    cov, acc = [], []
    for t in taus:
        keep = conf >= t
        cov.append(keep.mean())
        acc.append((y[keep] == yhat[keep]).mean() if keep.any() else np.nan)
    return np.array(cov), np.array(acc)


def accuracy_at_coverage(y, yhat, conf, coverage: float) -> float:
    """Accuracy on the most confident `coverage` share of trials."""
    y, yhat, conf = map(np.asarray, (y, yhat, conf))
    n_keep = max(1, int(round(coverage * len(y))))
    order = np.argsort(-conf, kind="stable")[:n_keep]
    return float(np.mean(y[order] == yhat[order]))
