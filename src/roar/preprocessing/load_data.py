import re
from datetime import datetime
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import polars as pl

from roar import DATA_DIR, MEASUREMENTS_CLEAN_NAMES


def load_data_df(data_dir: Path = DATA_DIR) -> pl.DataFrame:
    """Load the h5 files from the data dir into a data frame that contains some further information about the files

    Args:
        data_dir (Path, optional): Directory where the data is stored. Defaults to DATA_DIR.

    Returns:
        pl.DataFrame: Dataframe containing the
            file_path,
            file_stem,
            track_ID (from the h5 metadata if available),
            tyre_ID (from the path),
            vehicle (from the path),
            and if the data follows the naming convention.
    """
    h5_files = list(data_dir.rglob("*.h5"))

    parsed = [parse_filename(f) for f in h5_files]

    df = pl.DataFrame(parsed)
    df = df.with_columns(pl.col("measure").replace(MEASUREMENTS_CLEAN_NAMES).cast(pl.Categorical))
    return df


def parse_filename(filename: Path) -> dict[str, Any]:
    """
    Parse the filename stem (without extension) and return a dict of groups
    or None if the stem doesn't match the expected pattern.

    Example stems:
    - track211_ID.4_tyre3_2pt6_vr45_2025-07-11_10-24-28
    - track211_ID.4_tyre3_2pt6_vr50_b50_2025-07-11_10-41-07
    - track211_ID.4_tyre1_meas5_2p5_1_2025-08-07_10-48-15
    - track150_Q8 e-tron_tyre6_meas3_2p5_1_2025-09-29_17-28-02
    """
    pattern = re.compile(
        r"^track(?P<track>\d+)_"  # track digits only (e.g. 211)
        r"(?P<vehicle>[^_]+)_"
        r"tyre(?P<tyre>\d+)"
        r"(?:_[^_]+)*?"  # allow zero or more intermediate underscore tokens like 2pt6 or 2p5_1
        r"_"  # measurement token must be prefixed by underscore
        r"(?P<measure>(?:meas\d+|vr\d+)(?:_b\d+)?)"  # allow any digits for meas, vr, and optional b
        r"(?:_[^_]+)*?"  # allow extra tokens between measure and the date token
        r"_"  # date must be prefixed by underscore
        r"(?P<date>\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2})$"
    )

    stem = filename.stem
    m = pattern.match(stem)
    if not m:
        raise ValueError(f"Filename stem '{stem}' does not match expected pattern.")
    info = m.groupdict()

    return {
        "file_path": str(filename),
        "file_stem": stem,
        "vehicle": info["vehicle"],
        "tyre_ID": int(tyre_id) if (tyre_id := info["tyre"]) else None,
        "track_ID": int(track_id) if (track_id := info["track"]) else None,
        "measure": info["measure"],  # 'meas5' or 'vr50_b50' (includes optional _bNN)
        "date": datetime.strptime(info["date"], "%Y-%m-%d_%H-%M-%S"),
    }


def load_h5_channel(file_path: str | Path, channel_name: str) -> tuple[np.ndarray, int]:
    """Load a specific channel from an H5 file.

    Args:
        file_path (str | Path): Path to the H5 file.
        channel_name (str): Name of the channel to load.

    Returns:
        tuple[np.ndarray, int]: Tuple containing the data array and sample rate.
                                       Or None if channel not found.
    """
    if isinstance(file_path, Path):
        file_path = str(file_path)

    with h5py.File(file_path, "r") as f:
        if channel_name not in f:
            raise KeyError(f"Channel '{channel_name}' not found in {file_path}.")
        data = np.array(f[channel_name])
        # Flatten if 2D with single row/column
        if data.ndim == 2:
            if data.shape[0] == 1 or data.shape[1] == 1:
                data = data.flatten()
        sample_rate = f[channel_name].attrs.get("sample_rate", None)
        if isinstance(sample_rate, np.ndarray):
            sample_rate = sample_rate.item()
        return data, sample_rate
