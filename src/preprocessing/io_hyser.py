"""
Hyser-specific signal loading for Stage 1.

Each row in the extraction manifest with file_kind == "signal" corresponds
to one WFDB record (a .dat/.hea pair) for one subject/session/task/sample.
CEMHSEY loading (.mat files) will need its own module here once that
dataset's extraction stage is built; Stage 1's filtering and normalisation
code is dataset-agnostic and does not depend on this loader.
"""

import time
from pathlib import Path

import wfdb


def read_hyser_record(dat_path: str | Path, retries: int = 5):
    """
    Read one Hyser WFDB record and return (signal, fs, channel_names).

    `dat_path` may be given with or without the .dat extension. signal is
    returned as (n_channels, n_samples), transposed from wfdb's
    (n_samples, n_channels) convention to match this pipeline's layout.
    Reads are retried a few times because cloud-synced folders (OneDrive)
    occasionally raise a transient "Resource deadlock avoided" OSError.
    """
    record_base = str(Path(dat_path).with_suffix(""))

    for attempt in range(retries):
        try:
            record = wfdb.rdrecord(record_base)
            break
        except OSError:
            if attempt == retries - 1:
                raise
            time.sleep(0.5 * (attempt + 1))

    return record.p_signal.T, record.fs, record.sig_name
