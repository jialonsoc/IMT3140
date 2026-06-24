"""
Pipeline 01 - Clean window feature engineering for clinical real-time modeling.

Outputs:
    data/processed_features_windows_real.csv
    data/processed_features_windows_synthetic.csv

Design rule:
    No future/outcome descriptor is retained as a modelable predictor. The
    exported files contain target, tracking/grouping columns, valid baseline
    admission metadata and real-time FHR/UC window features only.
"""

from __future__ import annotations

import os
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from feature_extractor import (
    FS_DEFAULT,
    MAX_INVALID_WINDOW_PCT,
    RAW_DIR,
    WINDOW_LENGTH_MIN,
    WINDOW_STRIDE_MIN,
    build_processed_windowed_dataset,
    extract_windowed_features_for_record,
)


REAL_SOURCE_PATH = Path("data/processed_features_windows.csv")
REAL_OUTPUT_PATH = Path("data/processed_features_windows_real.csv")
SYNTH_DIR = Path("data/synthetic")
SYNTHETIC_METADATA_PATH = Path("data/dataset_synthetic.csv")
SYNTHETIC_OUTPUT_PATH = Path("data/processed_features_windows_synthetic.csv")
CLINICAL_METADATA_PATH = Path("data/clinical_metadata.csv")

PROHIBITED_COLUMNS = {
    "Deliv_type",
    "II_stage",
    "I_stage",
    "NoProgress",
    "Pos_IIst",
    "Sig2Birth",
    "pH",
    "BDecf",
    "pCO2",
    "BE",
    "Apgar1",
    "Apgar5",
    "NICU_days",
    "Seizures",
    "HIE",
    "Intubation",
    "Main_diag",
    "Other_diag",
    "dbID",
}

BASAL_METADATA_COLUMNS = [
    "Age",
    "Gravidity",
    "Parity",
    "Diabetes",
    "Pyrexia",
    "Weight_g",
    "Gest_weeks",
    "Sex",
]

SIGNAL_FEATURE_COLUMNS = [
    "signal_duration_min",
    "fhr_invalid_pct",
    "uc_invalid_pct",
    "fhr_baseline_mean_bpm",
    "fhr_baseline_median_bpm",
    "fhr_baseline_std_bpm",
    "fhr_baseline_slope_bpm_min",
    "accelerations_count",
    "accelerations_mean_amp_bpm",
    "accelerations_mean_duration_s",
    "decelerations_count",
    "decelerations_mean_amp_bpm",
    "decelerations_mean_duration_s",
    "uc_contractions_count",
    "fhr_apen",
    "fhr_sampen",
    "ltv_mean_amp_bpm",
    "ltv_median_amp_bpm",
    "ltv_valid_windows",
    "decelerations_early_count",
    "decelerations_late_count",
    "decelerations_variable_count",
    "deceleration_uc_lag_mean_s",
    "deceleration_uc_lag_median_s",
    "dfa_alpha",
    "dfa_intercept",
    "dfa_alpha_short",
    "dfa_alpha_long",
]

TREND_FEATURE_COLUMNS = [
    "delta_baseline_std_bpm",
    "delta_sampen",
    "fhr_std_falling_streak",
    "fhr_sampen_falling_streak",
    "slope_sampen_30min",
    "slope_baseline_mean_30min",
]

TRACKING_COLUMNS_REAL = [
    "record",
    "window_id",
    "window_start_min",
    "window_end_min",
    "time_to_birth_min",
    "target",
    "sample_source",
    "sequence_id",
    "group_record",
]

TRACKING_COLUMNS_SYNTHETIC = [
    "record",
    "source_record",
    "bootstrap_idx",
    "window_id",
    "window_start_min",
    "window_end_min",
    "time_to_birth_min",
    "target",
    "sample_source",
    "sequence_id",
    "group_record",
]


def create_composite_target(df: pd.DataFrame) -> pd.Series:
    """Strategy B outcome: pH < 7.10 OR Apgar5 < 7 OR NICU_days > 0."""
    return (
        (df["pH"] < 7.10) | (df["Apgar5"] < 7.0) | (df["NICU_days"] > 0.0)
    ).astype(int)


def falling_streak(values: pd.Series) -> pd.Series:
    """Count consecutive decreases against the previous chronological window."""
    arr = values.to_numpy(dtype=float)
    out = np.zeros(arr.shape[0], dtype=float)
    for idx in range(1, arr.shape[0]):
        if np.isfinite(arr[idx]) and np.isfinite(arr[idx - 1]) and arr[idx] < arr[idx - 1]:
            out[idx] = out[idx - 1] + 1.0
    return pd.Series(out, index=values.index)


def rolling_slope(values: pd.Series, times_min: pd.Series, window_size: int = 6) -> pd.Series:
    """Slope over the last six continuous 5-minute windows, in units/minute."""
    y = values.to_numpy(dtype=float)
    t = times_min.to_numpy(dtype=float)
    out = np.full(y.shape[0], np.nan, dtype=float)
    for idx in range(window_size - 1, y.shape[0]):
        y_window = y[idx - window_size + 1 : idx + 1]
        t_window = t[idx - window_size + 1 : idx + 1]
        if not np.all(np.isfinite(t_window)):
            continue
        if not np.allclose(np.diff(t_window), 5.0, atol=1e-6):
            continue
        valid = np.isfinite(y_window)
        if valid.sum() >= 2:
            out[idx] = float(np.polyfit(t_window[valid], y_window[valid], deg=1)[0])
    return pd.Series(out, index=values.index)


def add_sequential_features(
    df: pd.DataFrame,
    group_columns: list[str],
    sort_columns: list[str],
) -> pd.DataFrame:
    """Add deltas, falling streaks and 30-minute slopes per patient/sequence."""
    result = df.sort_values(sort_columns).copy()
    grouped = result.groupby(group_columns, sort=False, group_keys=False)
    result["delta_baseline_std_bpm"] = grouped["fhr_baseline_std_bpm"].diff(2)
    result["delta_sampen"] = grouped["fhr_sampen"].diff(2)
    result["fhr_std_falling_streak"] = grouped["fhr_baseline_std_bpm"].apply(falling_streak)
    result["fhr_sampen_falling_streak"] = grouped["fhr_sampen"].apply(falling_streak)
    result["slope_sampen_30min"] = grouped.apply(
        lambda group: rolling_slope(group["fhr_sampen"], group["window_start_min"]),
        include_groups=False,
    )
    result["slope_baseline_mean_30min"] = grouped.apply(
        lambda group: rolling_slope(group["fhr_baseline_mean_bpm"], group["window_start_min"]),
        include_groups=False,
    )
    result[TREND_FEATURE_COLUMNS] = result[TREND_FEATURE_COLUMNS].fillna(0.0)
    return result


def valid_predictor_columns(df: pd.DataFrame) -> list[str]:
    """Return the strict whitelist of modelable clinical + signal predictors."""
    allowed = BASAL_METADATA_COLUMNS + SIGNAL_FEATURE_COLUMNS + TREND_FEATURE_COLUMNS
    return [column for column in allowed if column in df.columns and column not in PROHIBITED_COLUMNS]


def _select_clean_columns(df: pd.DataFrame, tracking_columns: list[str]) -> pd.DataFrame:
    """Keep only tracking columns and valid modelable predictors."""
    columns = [column for column in tracking_columns if column in df.columns]
    columns += [column for column in valid_predictor_columns(df) if column not in columns]
    return df[columns].copy()


def build_real_windows() -> pd.DataFrame:
    """Load existing real windows or build them, then export the clean real matrix."""
    if not REAL_SOURCE_PATH.exists():
        build_processed_windowed_dataset(output_path=REAL_SOURCE_PATH)

    df = pd.read_csv(REAL_SOURCE_PATH)
    df["target"] = create_composite_target(df)
    df["sample_source"] = "real"
    df["sequence_id"] = df["record"].astype(int).astype(str)
    df["group_record"] = df["record"].astype(int)
    df = add_sequential_features(
        df,
        group_columns=["record"],
        sort_columns=["record", "window_start_min"],
    )
    clean_df = _select_clean_columns(df, TRACKING_COLUMNS_REAL)
    clean_df.to_csv(REAL_OUTPUT_PATH, index=False)
    return clean_df


def _extract_one_synthetic(row: dict[str, Any]) -> list[dict[str, Any]]:
    """Worker: extract window features for one synthetic FHR series."""
    synthetic_record = str(row["record"])
    source_record = int(row["source_record"])
    bootstrap_idx = int(row["bootstrap_idx"])
    fhr_path = SYNTH_DIR / f"{synthetic_record}_fhr.npy"
    uc_path = RAW_DIR / f"{source_record}_uc.npy"
    if not fhr_path.exists() or not uc_path.exists():
        return []

    fhr_signal = np.load(fhr_path)
    uc_signal = np.load(uc_path)
    metadata_row = pd.Series(row)
    rows = extract_windowed_features_for_record(
        synthetic_record,
        fhr_signal,
        uc_signal,
        metadata_row,
        fs=FS_DEFAULT,
        window_length_min=WINDOW_LENGTH_MIN,
        stride_min=WINDOW_STRIDE_MIN,
        max_invalid_pct=MAX_INVALID_WINDOW_PCT,
    )
    target = int(
        (float(row["pH"]) < 7.10)
        or (float(row["Apgar5"]) < 7.0)
        or (float(row["NICU_days"]) > 0.0)
    )
    sequence_id = f"{source_record}_boot_{bootstrap_idx}"
    for item in rows:
        item["record"] = synthetic_record
        item["source_record"] = source_record
        item["bootstrap_idx"] = bootstrap_idx
        item["target"] = target
        item["sample_source"] = "synthetic"
        item["sequence_id"] = sequence_id
        item["group_record"] = source_record
        for column in BASAL_METADATA_COLUMNS:
            item[column] = row.get(column, np.nan)
    return rows


def build_synthetic_windows(max_workers: int | None = None) -> pd.DataFrame:
    """
    Extract synthetic sliding windows.

    Synthetic FHR is paired with the source record's real UC because the current
    repository contains synthetic FHR files only. Downstream splits still filter
    synthetic rows to training source records only.
    """
    if not SYNTHETIC_METADATA_PATH.exists():
        raise FileNotFoundError(f"Missing {SYNTHETIC_METADATA_PATH}")

    synthetic_metadata = pd.read_csv(SYNTHETIC_METADATA_PATH)
    tasks = synthetic_metadata.to_dict(orient="records")
    workers = max_workers or max(1, min(4, (os.cpu_count() or 2) - 1))
    rows: list[dict[str, Any]] = []
    with ProcessPoolExecutor(max_workers=workers) as executor:
        futures = [executor.submit(_extract_one_synthetic, row) for row in tasks]
        for future in tqdm(as_completed(futures), total=len(futures), desc="Synthetic windows"):
            rows.extend(future.result())

    if not rows:
        raise RuntimeError("No synthetic windows were extracted.")

    df = pd.DataFrame(rows)
    df = add_sequential_features(
        df,
        group_columns=["source_record", "bootstrap_idx"],
        sort_columns=["source_record", "bootstrap_idx", "window_start_min"],
    )
    clean_df = _select_clean_columns(df, TRACKING_COLUMNS_SYNTHETIC)
    clean_df.to_csv(SYNTHETIC_OUTPUT_PATH, index=False)
    return clean_df


def main() -> None:
    """Run clean feature engineering for real and synthetic windows."""
    REAL_OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    real_df = build_real_windows()
    synthetic_df = build_synthetic_windows()
    print(f"Real clean windows: {real_df.shape} -> {REAL_OUTPUT_PATH}")
    print(f"Synthetic clean windows: {synthetic_df.shape} -> {SYNTHETIC_OUTPUT_PATH}")


if __name__ == "__main__":
    main()
