"""
Feature extraction for CTG records.

This module builds a tabular feature set from paired fetal heart rate (FHR)
and uterine contraction (UC) time series stored as NumPy arrays.

Optional install notes:
    pip install numpy pandas scipy tqdm

The non-linear features (ApEn, SampEn and DFA) are implemented locally to keep
the pipeline reproducible without requiring antropy or nolds.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.ndimage import median_filter, uniform_filter1d
from scipy.signal import find_peaks
from scipy.spatial import cKDTree
from tqdm import tqdm


FS_DEFAULT = 4
RAW_DIR = Path("data/raw")
CLINICAL_METADATA_PATH = Path("data/clinical_metadata.csv")
OUTPUT_PATH = Path("data/processed_features.csv")


@dataclass(frozen=True)
class Event:
    """A sustained FHR excursion relative to baseline."""

    start_idx: int
    end_idx: int
    peak_idx: int
    amplitude_bpm: float
    duration_s: float


def _odd_window(samples: int, minimum: int = 3) -> int:
    """Return an odd positive window length suitable for median filters."""
    samples = max(int(samples), minimum)
    return samples if samples % 2 == 1 else samples + 1


def interpolate_missing(
    signal: np.ndarray,
    *,
    invalid_zero: bool = True,
    valid_min: float | None = None,
    valid_max: float | None = None,
) -> tuple[np.ndarray, float]:
    """
    Replace NaNs, zeros and out-of-range samples by linear interpolation.

    Edges are filled with the nearest valid value, matching numpy.interp
    behavior. If the whole signal is invalid, a vector of NaNs is returned.

    Returns:
        cleaned signal and percentage of samples marked invalid before filling.
    """
    arr = np.asarray(signal, dtype=np.float64).reshape(-1)
    invalid = ~np.isfinite(arr)
    if invalid_zero:
        invalid |= arr == 0
    if valid_min is not None:
        invalid |= arr < valid_min
    if valid_max is not None:
        invalid |= arr > valid_max

    invalid_pct = float(invalid.mean() * 100.0) if arr.size else np.nan
    if arr.size == 0 or invalid.all():
        return np.full(arr.shape, np.nan, dtype=np.float64), invalid_pct

    idx = np.arange(arr.size)
    cleaned = arr.copy()
    cleaned[invalid] = np.interp(idx[invalid], idx[~invalid], arr[~invalid])
    return cleaned, invalid_pct


def estimate_fhr_baseline(fhr: np.ndarray, fs: int = FS_DEFAULT) -> np.ndarray:
    """
    Estimate a robust FHR baseline using repeated median filtering.

    The first pass removes short artifacts and beat-to-beat oscillations; the
    second pass produces a slow trend suitable as a CTG baseline proxy.
    """
    if np.isnan(fhr).all():
        return np.full_like(fhr, np.nan, dtype=np.float64)

    short_window = _odd_window(15 * fs)
    long_window = _odd_window(120 * fs)
    baseline = median_filter(fhr, size=short_window, mode="nearest")
    baseline = median_filter(baseline, size=long_window, mode="nearest")
    return baseline.astype(np.float64, copy=False)


def baseline_slope_bpm_per_min(baseline: np.ndarray, fs: int = FS_DEFAULT) -> float:
    """Estimate linear baseline slope in bpm per minute."""
    valid = np.isfinite(baseline)
    if valid.sum() < 2:
        return np.nan
    t_min = np.arange(baseline.size, dtype=np.float64)[valid] / (fs * 60.0)
    slope, _intercept = np.polyfit(t_min, baseline[valid], deg=1)
    return float(slope)


def _find_sustained_events(
    delta: np.ndarray,
    *,
    threshold_bpm: float,
    min_duration_s: float,
    fs: int,
    direction: str,
) -> list[Event]:
    """Find sustained positive or negative FHR excursions from baseline."""
    if direction not in {"above", "below"}:
        raise ValueError("direction must be 'above' or 'below'")

    mask = delta >= threshold_bpm if direction == "above" else delta <= -threshold_bpm
    if not mask.any():
        return []

    min_len = int(round(min_duration_s * fs))
    padded = np.pad(mask.astype(np.int8), (1, 1), mode="constant")
    transitions = np.diff(padded)
    starts = np.flatnonzero(transitions == 1)
    ends = np.flatnonzero(transitions == -1)

    events: list[Event] = []
    for start, end in zip(starts, ends):
        if end - start < min_len:
            continue
        segment = delta[start:end]
        local_idx = int(np.argmax(segment) if direction == "above" else np.argmin(segment))
        peak_idx = start + local_idx
        amplitude = float(segment[local_idx] if direction == "above" else -segment[local_idx])
        events.append(
            Event(
                start_idx=int(start),
                end_idx=int(end),
                peak_idx=int(peak_idx),
                amplitude_bpm=amplitude,
                duration_s=float((end - start) / fs),
            )
        )
    return events


def extract_variability_features(
    fhr: np.ndarray,
    baseline: np.ndarray,
    fs: int = FS_DEFAULT,
    *,
    window_s: int = 60,
) -> dict[str, float]:
    """
    Compute long-term variability as mean oscillation amplitude per window.

    Windows with sustained deceleration content are excluded because they
    inflate oscillation amplitude without representing baseline variability.
    """
    window = int(window_s * fs)
    if window <= 0 or fhr.size < window:
        return {
            "ltv_mean_amp_bpm": np.nan,
            "ltv_median_amp_bpm": np.nan,
            "ltv_valid_windows": 0.0,
        }

    usable = (fhr.size // window) * window
    residual = (fhr[:usable] - baseline[:usable]).reshape(-1, window)
    valid = np.isfinite(residual)
    decel_fraction = np.mean((residual <= -15.0) & valid, axis=1)
    enough_valid = valid.mean(axis=1) >= 0.80
    keep = enough_valid & (decel_fraction < 0.10)
    if not keep.any():
        return {
            "ltv_mean_amp_bpm": np.nan,
            "ltv_median_amp_bpm": np.nan,
            "ltv_valid_windows": 0.0,
        }

    amplitudes = np.nanpercentile(residual[keep], 95, axis=1) - np.nanpercentile(
        residual[keep], 5, axis=1
    )
    return {
        "ltv_mean_amp_bpm": float(np.nanmean(amplitudes)),
        "ltv_median_amp_bpm": float(np.nanmedian(amplitudes)),
        "ltv_valid_windows": float(keep.sum()),
    }


def detect_uc_peaks(uc: np.ndarray, fs: int = FS_DEFAULT) -> np.ndarray:
    """
    Detect contraction peaks in the UC signal.

    The detector uses a smoothed UC trace and a dynamic height/prominence rule
    so it can adapt to recordings with different contraction amplitudes.
    """
    if uc.size == 0 or np.isnan(uc).all():
        return np.array([], dtype=np.int64)

    smoothed = uniform_filter1d(uc, size=max(1, int(10 * fs)), mode="nearest")
    finite = smoothed[np.isfinite(smoothed)]
    if finite.size == 0:
        return np.array([], dtype=np.int64)

    height = max(float(np.nanpercentile(finite, 60)), 10.0)
    prominence = max(float(np.nanstd(finite) * 0.4), 5.0)
    min_distance = int(60 * fs)
    peaks, _properties = find_peaks(
        smoothed,
        height=height,
        prominence=prominence,
        distance=min_distance,
    )
    return peaks.astype(np.int64, copy=False)


def classify_decelerations(
    decelerations: list[Event],
    uc_peak_indices: np.ndarray,
    fs: int = FS_DEFAULT,
) -> dict[str, float]:
    """
    Classify decelerations by temporal lag from the nearest UC peak.

    Heuristic:
        early: FHR nadir occurs within +/- 15 s of UC peak.
        late: FHR nadir occurs 15 to 90 s after UC peak.
        variable_unclassified: no nearby contraction or lag outside those rules.
    """
    early = 0
    late = 0
    variable = 0
    lags_s: list[float] = []

    for decel in decelerations:
        if uc_peak_indices.size == 0:
            variable += 1
            continue
        nearest_peak = uc_peak_indices[np.argmin(np.abs(uc_peak_indices - decel.peak_idx))]
        lag_s = float((decel.peak_idx - nearest_peak) / fs)
        lags_s.append(lag_s)
        if abs(lag_s) <= 15.0:
            early += 1
        elif 15.0 < lag_s <= 90.0:
            late += 1
        else:
            variable += 1

    return {
        "decelerations_early_count": float(early),
        "decelerations_late_count": float(late),
        "decelerations_variable_count": float(variable),
        "deceleration_uc_lag_mean_s": float(np.mean(lags_s)) if lags_s else np.nan,
        "deceleration_uc_lag_median_s": float(np.median(lags_s)) if lags_s else np.nan,
    }


def _coarse_grain(signal: np.ndarray, fs: int, target_fs: int = 1) -> np.ndarray:
    """Downsample by averaging complete blocks for non-linear feature speed."""
    if fs <= target_fs:
        return signal.astype(np.float64, copy=False)
    block = int(round(fs / target_fs))
    usable = (signal.size // block) * block
    if usable == 0:
        return signal.astype(np.float64, copy=False)
    return np.nanmean(signal[:usable].reshape(-1, block), axis=1)


def _limit_entropy_length(signal: np.ndarray, max_samples: int = 1200) -> np.ndarray:
    """Uniformly subsample long signals to bound KD-tree entropy cost."""
    valid = signal[np.isfinite(signal)]
    if valid.size <= max_samples:
        return valid
    indices = np.linspace(0, valid.size - 1, max_samples).astype(np.int64)
    return valid[indices]


def _embedded_view(signal: np.ndarray, order: int) -> np.ndarray:
    """Create an embedding matrix with consecutive samples."""
    if signal.size < order:
        return np.empty((0, order), dtype=np.float64)
    return np.lib.stride_tricks.sliding_window_view(signal, order)


def approximate_entropy(signal: np.ndarray, order: int = 2, r: float | None = None) -> float:
    """
    Compute Approximate Entropy (ApEn) using Chebyshev-distance KD-trees.

    Self matches are intentionally included, as in the original ApEn
    definition.
    """
    x = np.asarray(signal, dtype=np.float64)
    x = x[np.isfinite(x)]
    if x.size < order + 2 or np.nanstd(x) == 0:
        return np.nan
    tolerance = float(0.2 * np.nanstd(x) if r is None else r)
    if tolerance <= 0:
        return np.nan

    def phi(m: int) -> float:
        emb = _embedded_view(x, m)
        if emb.size == 0:
            return np.nan
        tree = cKDTree(emb)
        counts = np.asarray(tree.query_ball_point(emb, tolerance, p=np.inf, return_length=True))
        return float(np.mean(np.log(counts / emb.shape[0])))

    return float(phi(order) - phi(order + 1))


def sample_entropy(signal: np.ndarray, order: int = 2, r: float | None = None) -> float:
    """Compute Sample Entropy (SampEn) using Chebyshev-distance KD-trees."""
    x = np.asarray(signal, dtype=np.float64)
    x = x[np.isfinite(x)]
    if x.size < order + 2 or np.nanstd(x) == 0:
        return np.nan
    tolerance = float(0.2 * np.nanstd(x) if r is None else r)
    if tolerance <= 0:
        return np.nan

    def pair_count(m: int) -> float:
        emb = _embedded_view(x, m)
        if emb.shape[0] < 2:
            return 0.0
        tree = cKDTree(emb)
        counts = np.asarray(tree.query_ball_point(emb, tolerance, p=np.inf, return_length=True))
        return float(np.sum(counts - 1) / 2.0)

    b = pair_count(order)
    a = pair_count(order + 1)
    if a <= 0 or b <= 0:
        return np.nan
    return float(-np.log(a / b))


def detrended_fluctuation_analysis(
    signal: np.ndarray,
    *,
    scales: tuple[int, ...] = (4, 8, 16, 32, 64, 128),
) -> dict[str, float]:
    """
    Estimate DFA scaling coefficients from a detrended cumulative profile.

    Scales are expressed in samples of the signal provided to this function.
    The returned short/long slopes split the available log-log points in half.
    """
    x = np.asarray(signal, dtype=np.float64)
    x = x[np.isfinite(x)]
    if x.size < max(scales) * 2 or np.nanstd(x) == 0:
        return {
            "dfa_alpha": np.nan,
            "dfa_intercept": np.nan,
            "dfa_alpha_short": np.nan,
            "dfa_alpha_long": np.nan,
        }

    profile = np.cumsum(x - np.mean(x))
    used_scales: list[int] = []
    fluctuations: list[float] = []

    for scale in scales:
        if scale < 4 or x.size < scale * 2:
            continue
        n_segments = x.size // scale
        segments = profile[: n_segments * scale].reshape(n_segments, scale)
        t = np.arange(scale, dtype=np.float64)
        rms_values = []
        for segment in segments:
            coeffs = np.polyfit(t, segment, deg=1)
            trend = coeffs[0] * t + coeffs[1]
            rms_values.append(np.sqrt(np.mean((segment - trend) ** 2)))
        fluctuation = float(np.sqrt(np.mean(np.square(rms_values))))
        if np.isfinite(fluctuation) and fluctuation > 0:
            used_scales.append(scale)
            fluctuations.append(fluctuation)

    if len(used_scales) < 2:
        return {
            "dfa_alpha": np.nan,
            "dfa_intercept": np.nan,
            "dfa_alpha_short": np.nan,
            "dfa_alpha_long": np.nan,
        }

    log_scales = np.log(np.asarray(used_scales, dtype=np.float64))
    log_fluctuations = np.log(np.asarray(fluctuations, dtype=np.float64))
    alpha, intercept = np.polyfit(log_scales, log_fluctuations, deg=1)

    midpoint = max(2, len(used_scales) // 2)
    alpha_short = np.nan
    alpha_long = np.nan
    if midpoint >= 2:
        alpha_short = float(np.polyfit(log_scales[:midpoint], log_fluctuations[:midpoint], deg=1)[0])
    if len(used_scales) - midpoint >= 2:
        alpha_long = float(np.polyfit(log_scales[midpoint:], log_fluctuations[midpoint:], deg=1)[0])

    return {
        "dfa_alpha": float(alpha),
        "dfa_intercept": float(intercept),
        "dfa_alpha_short": alpha_short,
        "dfa_alpha_long": alpha_long,
    }


def extract_features_from_record(
    fhr_signal: np.ndarray,
    uc_signal: np.ndarray,
    fs: int = FS_DEFAULT,
) -> dict[str, float]:
    """
    Extract FIGO-inspired temporal, UC-coupled and non-linear FHR features.

    Args:
        fhr_signal: FHR signal in beats per minute.
        uc_signal: UC signal in mmHg.
        fs: Sampling frequency in Hz. The CTU-UHB CTG files here use 4 Hz.

    Returns:
        A flat dictionary ready to become one row of a pandas DataFrame.
    """
    fhr_clean, fhr_invalid_pct = interpolate_missing(
        fhr_signal,
        invalid_zero=True,
        valid_min=50.0,
        valid_max=240.0,
    )
    uc_clean, uc_invalid_pct = interpolate_missing(
        uc_signal,
        invalid_zero=True,
        valid_min=0.0,
        valid_max=150.0,
    )

    baseline = estimate_fhr_baseline(fhr_clean, fs=fs)
    delta = fhr_clean - baseline

    accelerations = _find_sustained_events(
        delta,
        threshold_bpm=15.0,
        min_duration_s=15.0,
        fs=fs,
        direction="above",
    )
    decelerations = _find_sustained_events(
        delta,
        threshold_bpm=15.0,
        min_duration_s=15.0,
        fs=fs,
        direction="below",
    )
    uc_peaks = detect_uc_peaks(uc_clean, fs=fs)

    entropy_signal = _limit_entropy_length(_coarse_grain(fhr_clean, fs=fs, target_fs=1))
    dfa_features = detrended_fluctuation_analysis(entropy_signal)

    features: dict[str, float] = {
        "signal_duration_min": float(fhr_clean.size / (fs * 60.0)) if fhr_clean.size else np.nan,
        "fhr_invalid_pct": fhr_invalid_pct,
        "uc_invalid_pct": uc_invalid_pct,
        "fhr_baseline_mean_bpm": float(np.nanmean(baseline)),
        "fhr_baseline_median_bpm": float(np.nanmedian(baseline)),
        "fhr_baseline_std_bpm": float(np.nanstd(baseline)),
        "fhr_baseline_slope_bpm_min": baseline_slope_bpm_per_min(baseline, fs=fs),
        "accelerations_count": float(len(accelerations)),
        "accelerations_mean_amp_bpm": float(np.mean([e.amplitude_bpm for e in accelerations]))
        if accelerations
        else 0.0,
        "accelerations_mean_duration_s": float(np.mean([e.duration_s for e in accelerations]))
        if accelerations
        else 0.0,
        "decelerations_count": float(len(decelerations)),
        "decelerations_mean_amp_bpm": float(np.mean([e.amplitude_bpm for e in decelerations]))
        if decelerations
        else 0.0,
        "decelerations_mean_duration_s": float(np.mean([e.duration_s for e in decelerations]))
        if decelerations
        else 0.0,
        "uc_contractions_count": float(uc_peaks.size),
        "fhr_apen": approximate_entropy(entropy_signal, order=2),
        "fhr_sampen": sample_entropy(entropy_signal, order=2),
    }
    features.update(extract_variability_features(fhr_clean, baseline, fs=fs))
    features.update(classify_decelerations(decelerations, uc_peaks, fs=fs))
    features.update(dfa_features)
    return features


def _record_id_from_fhr_path(path: Path) -> str:
    """Extract record id from a path like data/raw/1001_fhr.npy."""
    return path.name.removesuffix("_fhr.npy")


def process_raw_folder(
    raw_dir: Path = RAW_DIR,
    *,
    fs: int = FS_DEFAULT,
) -> pd.DataFrame:
    """Process every paired FHR/UC record in raw_dir into a features DataFrame."""
    rows: list[dict[str, Any]] = []
    fhr_paths = sorted(raw_dir.glob("*_fhr.npy"))
    if not fhr_paths:
        raise FileNotFoundError(f"No *_fhr.npy files found in {raw_dir}")

    for fhr_path in tqdm(fhr_paths, desc="Extracting CTG features"):
        record = _record_id_from_fhr_path(fhr_path)
        uc_path = raw_dir / f"{record}_uc.npy"
        if not uc_path.exists():
            raise FileNotFoundError(f"Missing paired UC file for record {record}: {uc_path}")

        fhr_signal = np.load(fhr_path)
        uc_signal = np.load(uc_path)
        features = extract_features_from_record(fhr_signal, uc_signal, fs=fs)
        features["record"] = int(record) if record.isdigit() else record
        rows.append(features)

    return pd.DataFrame(rows)


def build_processed_dataset(
    raw_dir: Path = RAW_DIR,
    clinical_metadata_path: Path = CLINICAL_METADATA_PATH,
    output_path: Path = OUTPUT_PATH,
    *,
    fs: int = FS_DEFAULT,
) -> pd.DataFrame:
    """
    Extract signal features, merge clinical metadata and save the final dataset.

    The merge uses the real metadata key present in this project: ``record``.
    """
    feature_df = process_raw_folder(raw_dir, fs=fs)
    clinical_df = pd.read_csv(clinical_metadata_path)
    if "record" not in clinical_df.columns:
        raise KeyError(f"'record' column not found in {clinical_metadata_path}")

    clinical_df["record"] = clinical_df["record"].astype(feature_df["record"].dtype)
    dataset = clinical_df.merge(feature_df, on="record", how="inner", validate="one_to_one")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    dataset.to_csv(output_path, index=False)
    return dataset


def main() -> None:
    """Run the complete extraction pipeline for the original CTG records."""
    dataset = build_processed_dataset()
    print(f"Saved {dataset.shape[0]} rows x {dataset.shape[1]} columns to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
