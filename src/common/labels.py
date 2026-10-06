"""
Hyser gesture-label parsing.

Each Hyser session folder ships label_<task>.txt: one comma-separated list
of gesture IDs (1 to 34), where the k-th entry is the label of
<task>_<sig>_sample<k>. Trials that PhysioNet excluded for failed
performance are simply absent from both the sample files and the label
list, which is why e.g. gesture 22 can have 4 trials instead of 6.
"""

import functools
from pathlib import Path


@functools.lru_cache(maxsize=None)
def read_label_file(path: str) -> tuple[int, ...]:
    text = Path(path).read_text()
    return tuple(int(t) for t in text.replace("\n", ",").split(",") if t.strip())


def label_file_for(row) -> Path:
    """label_<task>.txt sitting next to the row's signal file."""
    return Path(row["file_path"]).parent / f"label_{row['task_type']}.txt"


def gesture_label_for_row(row) -> int:
    """Gesture ID (1 to 34) for one manifest row (one trial)."""
    labels = read_label_file(str(label_file_for(row)))
    idx = int(row["sample_index"]) - 1
    if not 0 <= idx < len(labels):
        raise ValueError(
            f"sample_index {row['sample_index']} outside label list "
            f"(len {len(labels)}) for {label_file_for(row)}"
        )
    return labels[idx]
