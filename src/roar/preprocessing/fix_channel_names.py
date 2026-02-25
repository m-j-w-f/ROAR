from pathlib import Path

import h5py
import numpy as np
import polars as pl

from roar import EXTRAS_DIR

SYNONYMS_FILE = EXTRAS_DIR / "all_measurement_channels_name.csv"


def get_channel_mapping(mapping_file: Path = SYNONYMS_FILE) -> dict[str, str]:
    """Get a mapping dictionary for synonyms to canonical channel names.

    Args:
        mapping_file (Path, optional): Path to the mapping file. Defaults to SYNONYMS_FILE.
    Returns:
        dict[str, str]: Mapping dictionary from synonyms to channel names.
    """
    df = pl.read_csv(mapping_file)

    syn_dict = {}
    for syn in ["synonym_1", "synonym_2"]:
        dicts = df.filter(pl.col(syn).is_not_null()).select(["channel_name", syn]).to_dicts()
        for d in dicts:
            if (synonym := d[syn]) in d:
                raise KeyError(f"Key '{synonym}' already exists.")
            syn_dict[synonym] = d["channel_name"]

    return syn_dict


def fix_channel_names(file_path: Path, mapping: dict[str, str], verbose: bool = False) -> None:
    """Fix channel names in a measurement file using a mapping dictionary.

    Args:
        file_path (Path): Path to the measurement file.
        mapping (dict[str, str]): Mapping dictionary from synonyms to canonical channel names.
        verbose (bool, optional): Whether to print verbose output. Defaults to False.

    Returns:
        None
    """
    with h5py.File(file_path, "r+") as h5_file:
        synonyms_to_rename = [synonym for synonym in h5_file.keys() if synonym in mapping]
        if not synonyms_to_rename and verbose:
            print("No channel names to fix.")
            return
        for synonym in synonyms_to_rename:
            canonical_name = mapping[synonym]
            # Copy the dataset to the canonical name
            if canonical_name not in h5_file:
                h5_file.copy(synonym, canonical_name)
                # Delete the old dataset
                del h5_file[synonym]
                if verbose:
                    print(f"Renamed dataset '{synonym}' to '{canonical_name}'.")
            else:
                # The canonical name already exists, need to compare data
                if verbose:
                    print(
                        f"Dataset '{synonym}' could not be renamed to '{canonical_name}', as it already exists. Comparing data..."
                    )

                # canonical_data is empty and synonym_data is not -> replace canonical with synonym
                canonical_all_zero = np.all(np.array(h5_file[canonical_name]) == 0)
                synonym_all_zero = np.all(np.array(h5_file[synonym]) == 0)
                array_equal = np.array_equal(
                    np.array(h5_file[canonical_name]), np.array(h5_file[synonym])
                )

                if canonical_all_zero and not synonym_all_zero:
                    # Replace the canonical dataset with data from the synonym
                    if verbose:
                        print(
                            f"Replaced dataset '{canonical_name}' with data from '{synonym}' as it was all zeros."
                        )
                    del h5_file[canonical_name]
                    h5_file.copy(synonym, canonical_name)
                    del h5_file[synonym]

                # synonym_data is empty and canonical_data is not -> delete synonym
                elif synonym_all_zero and not canonical_all_zero:
                    if verbose:
                        print(f"Dataset '{synonym}' is all zeros. Deleting it.")
                    del h5_file[synonym]
                # Both are identical and nonzero -> delete synonym
                elif array_equal and not (canonical_all_zero and synonym_all_zero):
                    if verbose:
                        print(
                            f"Datasets '{canonical_name}' and '{synonym}' are identical. Deleting '{synonym}'."
                        )
                    del h5_file[synonym]
                else:
                    # Both datasets have data but are different
                    raise ValueError(
                        f"Datasets '{canonical_name}' and '{synonym}' have different data. Manual intervention required."
                    )
