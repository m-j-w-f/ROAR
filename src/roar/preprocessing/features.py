from pathlib import Path

import librosa
import numpy as np
import scipy
from scipy.signal import welch

from roar import MIC_CHANNELS, MIC_CHANNELS_CLEANED

from .load_data import load_h5_channel


def extract_features_from_h5_file(file_path: Path | str, channels: list[str]) -> dict:
    """
    Extract features for channels specified in a channels.

    Args:
        file_path (Path | str): Path to the H5 file.
        channels (list[str]): List of channels to extract features for.

    Returns:
        dict: Flattened dictionary of features with keys like "{channel_name}_{feature_name}".
    """
    all_feats = {}
    for channel_name in channels:
        try:
            sig, fs = load_h5_channel(file_path, channel_name)
        except KeyError:
            sig, fs = None, None
            pass
        if channel_name in MIC_CHANNELS or channel_name in MIC_CHANNELS_CLEANED:
            if sig is not None and fs is not None:
                feats = extract_audio_features_from_signal(sig, fs)
        elif channel_name not in ["speed"]:
            # For non-audio channels, we can add other feature extraction methods here
            if sig is not None:
                feats = extract_statistic_features_from_signal(sig)
        else:
            feats = {}

        # Flatten features with prefix
        for k, v in feats.items():  # pyright: ignore[reportPossiblyUnboundVariable]
            all_feats[f"{channel_name}_{k}"] = v

        if channel_name == "speed":
            speed_accel_feats = get_speed_accel_features(Path(file_path))
            all_feats.update(speed_accel_feats)

    return all_feats


def extract_audio_features_from_signal(
    signal: np.ndarray, fs: int, n_mfcc=13, roll_percent=0.95
) -> dict[str, float]:
    """
    Extract single-scalar audio features from a 1D signal.

    Time-domain:
        - rms, mean, std, max, crest, zcr

    Frequency-domain (Welch PSD):
        - spec_centroid
        - spec_rolloff (e.g. 95% of spectral energy)
        - spec_flatness (geometric mean / arithmetic mean of PSD)
        - spec_bandwidth (spectral spread around centroid)
        - band_i (absolute band powers over fixed frequency ranges)

    MFCC (if librosa is available):
        - mfcc_1 ... mfcc_n_mfcc (mean over time for each coefficient)
    """
    if signal is None or fs is None:
        return {}

    # Make sure signal is 1D float
    x = np.asarray(signal, dtype=float).squeeze()
    if x.ndim != 1 or x.size < 10:
        # degenerate / empty channel
        return {}

    feats = {}

    # --- Time-domain ---
    feats["rms"] = float(np.sqrt(np.mean(x**2)))
    feats["mean"] = float(np.mean(x))
    feats["std"] = float(np.std(x))
    feats["max"] = float(np.max(np.abs(x)))
    feats["crest"] = float(feats["max"] / (feats["rms"] + 1e-9))

    # zero-crossing rate
    feats["zcr"] = float(((x[:-1] * x[1:]) < 0).mean())

    # --- Frequency-domain using Welch PSD ---
    f, psd = welch(x, fs, nperseg=min(4096, x.size))

    # ensure 1D PSD
    psd = np.asarray(psd).squeeze()
    f = np.asarray(f).squeeze()

    if f.ndim != 1 or psd.ndim != 1 or f.size != psd.size:
        # something weird, bail out on frequency features
        return feats

    total_power = np.sum(psd) + 1e-9

    # Spectral centroid (power-weighted mean frequency)
    spec_centroid = np.sum(f * psd) / total_power
    feats["spec_centroid"] = float(spec_centroid)

    # --- NEW: Spectral rolloff ---
    # Frequency below which `roll_percent` of total spectral energy lies.
    # Common choice: roll_percent = 0.95
    cumulative_power = np.cumsum(psd)
    threshold = roll_percent * cumulative_power[-1]
    idx = np.searchsorted(cumulative_power, threshold)
    idx = min(idx, len(f) - 1)
    feats["spec_rolloff"] = float(f[idx])

    # --- NEW: Spectral flatness ---
    # Ratio of geometric mean to arithmetic mean of PSD.
    # ~1 → noise-like, <<1 → tone-like.
    psd_safe = psd + 1e-12  # avoid log(0) / division by zero
    geo_mean = float(np.exp(np.mean(np.log(psd_safe))))
    arith_mean = float(np.mean(psd_safe))
    feats["spec_flatness"] = float(geo_mean / (arith_mean + 1e-12))

    # --- NEW: Spectral bandwidth (spread around centroid) ---
    # Here: standard deviation of frequency around centroid, weighted by PSD.
    feats["spec_bandwidth"] = float(np.sqrt(np.sum(((f - spec_centroid) ** 2) * psd) / total_power))

    # --- Band powers (already present, kept as-is) ---
    bands = [(0, 200), (200, 500), (500, 1000), (1000, 2000), (2000, 5000)]
    for i, (lo, hi) in enumerate(bands):
        mask = (f >= lo) & (f < hi)
        if not np.any(mask):
            feats[f"band_{i}"] = 0.0
        else:
            feats[f"band_{i}"] = float(np.sum(psd[mask]))

    # --- NEW: MFCC features (single scalars per coefficient) ---
    # We use librosa if available. Each coefficient is averaged over time.
    # librosa expects float32 and mono; x is already 1D float64
    mfcc = librosa.feature.mfcc(
        y=x.astype(np.float32), sr=fs, n_mfcc=n_mfcc
    )  # shape: (n_mfcc, n_frames)

    mfcc_means = mfcc.mean(axis=1)  # mean over time for each coeff
    for i, coeff in enumerate(mfcc_means, start=1):
        feats[f"mfcc_{i}"] = float(coeff)

    return feats


def extract_statistic_features_from_signal(signal: np.ndarray, *args, **kwargs) -> dict[str, float]:
    """Placeholder for other feature extraction methods."""
    return {
        "mean": float(np.mean(signal)),
        "std": float(np.std(signal)),
        "max": float(np.max(signal)),
        "min": float(np.min(signal)),
        "median": float(np.median(signal)),
    }


def get_speed_accel_features(file_path: Path) -> dict[str, float]:
    speed_east, _ = load_h5_channel(file_path=file_path, channel_name="v_east_CAN_Sig_")
    speed_north, sample_rate = load_h5_channel(file_path=file_path, channel_name="v_north_CAN_Sig")

    speed_ms = np.sqrt(speed_east**2 + speed_north**2).flatten()  # in m/s
    dt = 1 / sample_rate  # in seconds
    accel = np.gradient(speed_ms, dt)  # in m/s²

    res = []
    for arr in [speed_ms, accel]:
        res.append(extract_statistic_features_from_signal(arr))
    features = {}
    for i, prefix in enumerate(["speed", "accel"]):
        for k, v in res[i].items():
            features[f"{prefix}_{k}"] = v
    return features


def compute_fft_spectrum(data: np.ndarray, sample_rate: float):
    """Compute FFT spectrum of the signal."""
    n = len(data)
    # Apply window to reduce spectral leakage
    window = scipy.signal.windows.hann(n, sym=False)  # use periodic for FFT (optional)
    cg = np.sum(window) / n  # coherent gain
    windowed_data = data * window

    yf = scipy.fft.fft(windowed_data)
    xf = scipy.fft.fftfreq(n, 1 / sample_rate)[: n // 2]

    magnitude = 2.0 / (n * cg) * np.abs(yf[0 : n // 2])  # type: ignore

    # Calculate SPL (dB re 20 microPa)
    ref_pressure = 2e-5  # 20 microPascals
    magnitude_db = 20 * np.log10((magnitude + 1e-10) / ref_pressure)

    return xf, magnitude_db
