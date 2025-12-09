import re
from datetime import datetime
from pathlib import Path

import h5py
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


def load_h5_fix_channel_names(file_path: Path, mapping: dict) -> h5py.File:
    """Load a single h5 file and fix the channel names if necessary.

    Args:
        file_path (Path): Path to the h5 file.
        mapping (dict): Mapping of old channel names to new channel names.

    Returns:
        h5py.File: Loaded h5 file with fixed channel names.
    """
    h5_file = h5py.File(file_path, "r+")
    ...  # TODO
    return h5_file
