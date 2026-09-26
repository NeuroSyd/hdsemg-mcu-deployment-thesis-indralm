"""
Hyser-specific signal loading for Stage 1.

Each row in the extraction manifest with file_kind == "signal" corresponds
to one WFDB record (a .dat/.hea pair) for one subject/session/task/sample.
CEMHSEY loading (.mat files) will need its own module here once that
dataset's extraction stage is built; Stage 1's filtering and normalisation
code above is dataset-agnostic and does not depend on this loader.
"""

from pathlib import Path

import wfdb


def read_hyser_record(dat_path: str | Path):
    """
    Read one Hyser WFDB record and return (signal, fs, channel_names).

    `dat_path` may be given with or without the .dat extension; wfdb expects
    the record path without extension. signal is returned as
    (n_channels, n_samples), transposed from wfdb's (n_samples, n_channels)
    convention to match this pipeline's array layout.
    """
    record_path = Path(dat_path)
    record_base = str(record_path.with_suffix(""))

    record = wfdb.rdrecord(record_base)
    signal = record.p_signal.T
    return signal, record.fs, record.sig_name
