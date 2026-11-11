"""
Class maps for Solution A (gesture grouping): 34 Hyser gestures -> fewer classes.

A class map is a list of 34 ints, one per gesture (index 0 = gesture 1). Gestures with the same value
become one class. Specs accepted by --class-map:
    A1            anatomical, 22 classes: wrist + hand close (G14-19) and wrist + hand open (G20-25)
                  join their wrist movement (G6-11)
    A1b           A1, plus the three/four finger extensions G26-29 join G12, 18 classes
    A2-s1:K       data-driven, K classes, groups chosen from SESSION 1 confusion only
                  (read from reports/grouping/groups_A2.json, written by scripts/test_gesture_grouping.py)
    A2-s2:K       same but chosen with session 2 confusion: OPTIMISTIC, for comparison only
    path.json     a JSON list of 34 ints
The split is still stratified by the original 34 gestures, so the train, validation and test trials are
the same trials as in the 34-class runs and results can be compared trial by trial.
"""
import json
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]


def anatomical(level: str) -> list[int]:
    mp = list(range(34))
    for i in range(13, 19): mp[i] = i - 8      # G14-19 -> G6-11
    for i in range(19, 25): mp[i] = i - 14     # G20-25 -> G6-11
    if level == "A1b":
        for i in (25, 26, 27, 28): mp[i] = 11  # G26-29 -> G12
    return mp


def from_groups(groups: list[list[str]]) -> list[int]:
    """groups = merged groups such as [["G1","G2","G12"], ...]; every other gesture stays alone."""
    mp = list(range(34))
    for g in groups:
        ids = sorted(int(x[1:]) - 1 for x in g)
        for i in ids: mp[i] = ids[0]
    return mp


def resolve_class_map(spec: str) -> np.ndarray:
    """Return a contiguous class id (0..K-1) for each of the 34 gestures."""
    if spec in ("A1", "A1b"):
        raw = anatomical(spec)
    elif spec.startswith("A2-"):
        key = spec.replace(":", " K=")
        groups_file = REPO / "reports" / "grouping" / "groups_A2.json"
        groups = json.load(open(groups_file))
        if key not in groups:
            raise SystemExit(f"{key!r} not in {groups_file}; run scripts/test_gesture_grouping.py first")
        raw = from_groups(groups[key])
    else:
        raw = json.load(open(spec))
    raw = np.asarray(raw, int)
    assert raw.shape == (34,), "a class map needs 34 entries"
    return np.unique(raw, return_inverse=True)[1]


def safe_name(spec: str) -> str:
    return spec.replace(":", "_K").replace("/", "_").replace(".json", "").replace(" ", "")
