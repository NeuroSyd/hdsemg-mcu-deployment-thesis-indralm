"""
Manifest loading and filtering shared across all pipeline stages.

Every stage (extraction, preprocessing, windowing, features, ...) reads its
input as a manifest CSV and writes its output as a new manifest CSV with an
extra set of columns, rather than passing file paths around directly. This
keeps each stage independently re-runnable and auditable against the stage
before it.
"""

from pathlib import Path

import pandas as pd


def load_manifest(path: str | Path) -> pd.DataFrame:
    """Load a stage manifest CSV into a DataFrame."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"Manifest not found at {path}. Run the producing stage first."
        )
    return pd.read_csv(path)


def filter_manifest(
    df: pd.DataFrame,
    subjects: set[str] | None = None,
    sessions: set[str] | None = None,
    task_type: str | None = None,
    sig_type: str | None = None,
    file_kind: str | None = None,
) -> pd.DataFrame:
    """
    Narrow a manifest DataFrame by any combination of the standard columns.

    Any filter left as None is not applied. `subjects` matches against the
    numeric part of subject_id (e.g. {"01"} matches "subject01").
    """
    out = df

    if subjects:
        subject_num = out["subject_id"].str.replace("subject", "", regex=False)
        out = out[subject_num.isin(subjects)]
    if sessions:
        out = out[out["session"].astype(str).isin(sessions)]
    if task_type:
        out = out[out["task_type"] == task_type]
    if sig_type:
        out = out[out["sig_type"] == sig_type]
    if file_kind:
        out = out[out["file_kind"] == file_kind]

    return out.reset_index(drop=True)


def write_manifest(df: pd.DataFrame, path: str | Path) -> None:
    """Write a stage manifest CSV, creating parent directories as needed."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
