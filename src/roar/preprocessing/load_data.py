import re
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path

import h5py
import numpy as np
import polars as pl

from roar import DATA_DIR, EXTRAS_DIR


def load_data(data_dir: Path = DATA_DIR) -> pl.DataFrame:
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

    def _parse_filename(filename: Path) -> dict | None:
        """
        Parse the filename stem (without extension) and return a dict of groups
        or None if the stem doesn't match the expected pattern.

        Example stems:
        - track211_ID.4_tyre3_2pt6_vr45_2025-07-11_10-24-28
        - track211_ID.4_tyre3_2pt6_vr50_b50_2025-07-11_10-41-07
        - track211_ID.4_tyre1_meas5_2p5_1_2025-08-07_10-48-15
        - track150_Q8 e-tron_tyre6_meas3_2p5_1_2025-09-29_17-28-02
        """
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

    parsed = [_parse_filename(f) for f in h5_files]

    df = pl.DataFrame(parsed)
    return df


def get_channel_mapping_dict(mapping_file: Path | None = None) -> dict:
    """Get a mapping dictionary for old channel names to new channel names.

    Returns:
        dict: Mapping dictionary.
    """
    if mapping_file is None:
        mapping_file = EXTRAS_DIR / "all_measurement_channels_name.csv"
    df = pl.read_csv(mapping_file)

    syn_dict = {}
    for syn in ["synonym_1", "synonym_2"]:
        dicts = df.filter(pl.col(syn).is_not_null()).select(["channel_name", syn]).to_dicts()
        for d in dicts:
            if (synonym := d[syn]) in d:
                raise KeyError(f"Key '{synonym}' already exists.")
            syn_dict[synonym] = d["channel_name"]

    return syn_dict


def load_h5_fix_channel_names(file_path: Path, mapping: dict, verbose=False) -> h5py.File:
    """Load a single h5 file and fix the channel names if necessary.

    Args:
        file_path (Path): Path to the h5 file.
        mapping (dict): Mapping of old channel names to new channel names.

    Returns:
        h5py.File: Loaded h5 file with fixed channel names.
    """
    h5_file = h5py.File(file_path, "r+")

    # Get all keys that need to be renamed
    keys_to_rename = [key for key in h5_file.keys() if key in mapping]

    # Rename datasets by copying to new name and deleting old
    for old_name in keys_to_rename:
        new_name = mapping[old_name]
        # Copy the dataset with all its attributes
        try:
            h5_file.copy(old_name, new_name)
            # Delete the old dataset
            del h5_file[old_name]
            if verbose:
                print(f"Renamed dataset '{old_name}' to '{new_name}' in file '{file_path.name}'")
        except Exception:
            # Renaming already happened in a previous run
            if verbose:
                print(
                    f"Dataset '{old_name}' could not be renamed to '{new_name}'. It may already exist."
                )
            pass

    return h5_file


@contextmanager
def h5_with_fixed_channels(
    file_path: Path | str, mapping: dict | None = None, mode: str = "r", *args, **kwargs
):
    """Context manager to open an h5 file with fixed channel names.

    Args:
        file_path (Path): Path to the h5 file.
        mapping (dict | None): Mapping of old channel names to new channel names.
            If None, uses get_channel_mapping_dict(). Only applied if mode allows writing.
        mode (str): File mode. Use "r+" or "a" to enable channel renaming. Defaults to "r".

    Yields:
        h5py.File: Loaded h5 file with fixed channel names (if mode allows).

    Example:
        >>> with h5_with_fixed_channels(file_path, mapping) as f:
        ...     data = f["NAWSSound"][:]
    """
    if isinstance(file_path, str):
        file_path = Path(file_path)

    if mapping is None:
        mapping = get_channel_mapping_dict()

    # Only rename channels if mode allows writing
    if mode in ("r+", "a"):
        h5_file = load_h5_fix_channel_names(file_path, mapping, *args, **kwargs)
    else:
        h5_file = h5py.File(file_path, mode)

    try:
        yield h5_file
    finally:
        h5_file.close()


def load_h5_channel(file_path: str, mapping: dict | None, channel_name: str):
    """Load a specific channel from an H5 file."""
    if mapping is None:
        mapping = get_channel_mapping_dict()
    with h5_with_fixed_channels(file_path, mapping, "r+") as f:
        if channel_name in f:
            data = f[channel_name][:]  # type: ignore
            # Flatten if 2D with single row/column
            if data.ndim == 2:  # type: ignore
                if data.shape[0] == 1:  # type: ignore
                    data = data.flatten()  # type: ignore
                elif data.shape[1] == 1:  # type: ignore
                    data = data.flatten()  # type: ignore
            sample_rate = f[channel_name].attrs.get("sample_rate", None)
            if isinstance(sample_rate, np.ndarray):
                sample_rate = sample_rate.item()
            return data, sample_rate
    return None, None
