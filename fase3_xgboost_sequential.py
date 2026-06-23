"""
Phase 3: sequential XGBoost with longitudinal memory and live-test simulation.

Inputs:
    data/processed_features_windows.csv
    data/clinical_metadata.csv
    data/dataset_synthetic.csv

Outputs:
    data/results_fase3_xgboost_sequential_comparison.csv
    data/results_fase3_xgboost_sequential_feature_importance.csv
    data/results_fase3_xgboost_sequential_best_params.json
    data/results_fase2_model_comparison.csv  (updated with sequential rows)
    reports/xgboost_sequential_feature_importance_gain.png

Install notes:
    pip install xgboost numpy pandas scikit-learn matplotlib
"""

from __future__ import annotations

import json
import math
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import RandomizedSearchCV, StratifiedGroupKFold, train_test_split
from sklearn.pipeline import Pipeline


WINDOWS_INPUT_PATH = Path("data/processed_features_windows.csv")
CLINICAL_METADATA_PATH = Path("data/clinical_metadata.csv")
SYNTHETIC_DATASET_PATH = Path("data/dataset_synthetic.csv")
PHASE2_COMPARISON_PATH = Path("data/results_fase2_model_comparison.csv")

SEQUENTIAL_COMPARISON_OUTPUT_PATH = Path("data/results_fase3_xgboost_sequential_comparison.csv")
BEST_PARAMS_OUTPUT_PATH = Path("data/results_fase3_xgboost_sequential_best_params.json")
FEATURE_IMPORTANCE_OUTPUT_PATH = Path("data/results_fase3_xgboost_sequential_feature_importance.csv")
FEATURE_IMPORTANCE_FIG_PATH = Path("reports/xgboost_sequential_feature_importance_gain.png")

RANDOM_STATE = 42
N_ITER_SEARCH = 30
MIN_RECALL_FOR_THRESHOLD = 0.75

TREND_FEATURES = [
    "delta_baseline_std_bpm",
    "delta_sampen",
    "fhr_std_falling_streak",
    "fhr_sampen_falling_streak",
    "slope_sampen_30min",
    "slope_baseline_mean_30min",
]

TARGET_COLUMNS = ["pH", "Apgar5", "NICU_days"]
EXCLUDED_OUTCOME_COLUMNS = [
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
]
EXCLUDED_IDENTIFIER_COLUMNS = [
    "record",
    "dbID",
    "source_record",
    "bootstrap_idx",
    "is_synthetic",
    "group_record",
    "sample_source",
    "sequence_id",
]
EXCLUDED_WINDOW_TRACKING_COLUMNS = [
    "window_id",
    "window_start_min",
    "window_end_min",
    "time_to_birth_min",
]


@dataclass(frozen=True)
class PatientSplit:
    """Patient-level train/test split mapped to window rows."""

    train_df: pd.DataFrame
    test_df: pd.DataFrame
    train_records: set[int]
    test_records: set[int]


@dataclass(frozen=True)
class TrainingBundle:
    """Training matrix and augmentation metadata for one experiment."""

    name: str
    train_df: pd.DataFrame
    synthetic_rows_added: int
    synthetic_positive_rows_added: int
    note: str


def import_xgb_classifier() -> type[Any]:
    """Import XGBClassifier lazily with a clear installation message."""
    try:
        from xgboost import XGBClassifier
    except ImportError as exc:
        raise ImportError(
            "xgboost no esta instalado. Ejecuta: python -m pip install xgboost"
        ) from exc
    return XGBClassifier


def create_composite_target(df: pd.DataFrame) -> pd.Series:
    """Create Strategy B target: pH < 7.10 OR Apgar5 < 7 OR NICU_days > 0."""
    missing = [column for column in TARGET_COLUMNS if column not in df.columns]
    if missing:
        raise KeyError(f"Missing target columns: {missing}")
    return (
        (df["pH"] < 7.10) | (df["Apgar5"] < 7.0) | (df["NICU_days"] > 0.0)
    ).astype(int).rename("target")


def falling_streak(values: pd.Series) -> pd.Series:
    """Count consecutive decreases relative to the immediately previous window."""
    arr = values.to_numpy(dtype=float)
    streak = np.zeros(arr.shape[0], dtype=float)
    for idx in range(1, arr.shape[0]):
        if np.isfinite(arr[idx]) and np.isfinite(arr[idx - 1]) and arr[idx] < arr[idx - 1]:
            streak[idx] = streak[idx - 1] + 1.0
        else:
            streak[idx] = 0.0
    return pd.Series(streak, index=values.index)


def rolling_slope(
    values: pd.Series,
    times_min: pd.Series,
    *,
    window_size: int = 6,
    expected_stride_min: float = 5.0,
) -> pd.Series:
    """
    Compute rolling linear slope over the last six continuous 5-minute windows.

    The slope is expressed per minute. If the previous six rows are not
    temporally continuous, or contain fewer than two finite values, the result
    remains NaN and is later imputed to 0 for initial/gap windows.
    """
    y = values.to_numpy(dtype=float)
    t = times_min.to_numpy(dtype=float)
    out = np.full(y.shape[0], np.nan, dtype=float)
    for idx in range(window_size - 1, y.shape[0]):
        y_window = y[idx - window_size + 1 : idx + 1]
        t_window = t[idx - window_size + 1 : idx + 1]
        if not np.all(np.isfinite(t_window)):
            continue
        if not np.allclose(np.diff(t_window), expected_stride_min, atol=1e-6):
            continue
        valid = np.isfinite(y_window)
        if valid.sum() < 2:
            continue
        out[idx] = float(np.polyfit(t_window[valid], y_window[valid], deg=1)[0])
    return pd.Series(out, index=values.index)


def add_longitudinal_features(
    df: pd.DataFrame,
    *,
    group_columns: list[str],
    sort_columns: list[str],
) -> pd.DataFrame:
    """Add deltas, falling streaks and 30-minute rolling slopes per sequence."""
    result = df.sort_values(sort_columns).copy()
    if "window_start_min" not in result.columns:
        result["window_start_min"] = result.groupby(group_columns).cumcount() * 5.0
    required_base_columns = [
        "fhr_baseline_std_bpm",
        "fhr_sampen",
        "fhr_baseline_mean_bpm",
    ]
    for column in required_base_columns:
        if column not in result.columns:
            result[column] = np.nan

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
    result[TREND_FEATURES] = result[TREND_FEATURES].fillna(0.0)
    return result


def load_real_windows() -> pd.DataFrame:
    """Load real windowed features, map target and add longitudinal features."""
    windows_df = pd.read_csv(WINDOWS_INPUT_PATH)
    clinical_df = pd.read_csv(CLINICAL_METADATA_PATH)
    clinical_df["target"] = create_composite_target(clinical_df)
    windows_df["target"] = windows_df["record"].map(
        clinical_df.set_index("record")["target"]
    ).astype(int)
    windows_df["sample_source"] = "real"
    windows_df["sequence_id"] = windows_df["record"].astype(str)
    windows_df = add_longitudinal_features(
        windows_df,
        group_columns=["record"],
        sort_columns=["record", "window_start_min"],
    )
    windows_df["group_record"] = windows_df["record"].astype(str)
    return windows_df


def patient_level_split(real_windows_df: pd.DataFrame) -> PatientSplit:
    """Split unique patient IDs 70/30 before mapping windows to train/test."""
    clinical_df = pd.read_csv(CLINICAL_METADATA_PATH)
    clinical_df["target"] = create_composite_target(clinical_df)
    train_records, test_records = train_test_split(
        clinical_df["record"],
        test_size=0.30,
        stratify=clinical_df["target"],
        random_state=RANDOM_STATE,
    )
    train_record_set = set(train_records.astype(int))
    test_record_set = set(test_records.astype(int))
    train_df = real_windows_df[real_windows_df["record"].astype(int).isin(train_record_set)].copy()
    test_df = real_windows_df[real_windows_df["record"].astype(int).isin(test_record_set)].copy()
    overlap = set(train_df["record"].astype(int)).intersection(set(test_df["record"].astype(int)))
    if overlap:
        raise RuntimeError(f"Patient-level leakage detected: {sorted(overlap)[:5]}")
    return PatientSplit(train_df, test_df, train_record_set, test_record_set)


def get_predictor_columns(df: pd.DataFrame) -> list[str]:
    """Select numeric predictors excluding outcome, IDs and direct temporal tracking."""
    excluded = set(
        EXCLUDED_OUTCOME_COLUMNS
        + EXCLUDED_IDENTIFIER_COLUMNS
        + EXCLUDED_WINDOW_TRACKING_COLUMNS
        + ["target"]
    )
    candidates = [column for column in df.columns if column not in excluded]
    numeric_columns = df[candidates].select_dtypes(include=[np.number]).columns
    return [column for column in numeric_columns if df[column].nunique(dropna=True) >= 2]


def load_synthetic_rows(
    train_records: set[int],
    predictor_columns: list[str],
) -> tuple[pd.DataFrame, str]:
    """
    Load synthetic rows, add trend features by source_record + bootstrap_idx.

    The current synthetic file has one row per full synthetic signal rather than
    sliding windows. Trend features are still computed with the exact same
    function, but they become 0 because each synthetic sequence has length 1.
    """
    if not SYNTHETIC_DATASET_PATH.exists():
        return pd.DataFrame(), "Synthetic CSV not found; augmentation skipped."

    synthetic_df = pd.read_csv(SYNTHETIC_DATASET_PATH)
    if "source_record" not in synthetic_df.columns or "bootstrap_idx" not in synthetic_df.columns:
        return pd.DataFrame(), "Synthetic CSV missing source_record/bootstrap_idx; augmentation skipped."

    synthetic_df["source_record"] = pd.to_numeric(synthetic_df["source_record"], errors="coerce")
    synthetic_df = synthetic_df[synthetic_df["source_record"].isin(train_records)].copy()
    if synthetic_df.empty:
        return pd.DataFrame(), "No synthetic rows matched training source records."

    synthetic_df["target"] = create_composite_target(synthetic_df)
    synthetic_df["sample_source"] = "synthetic"
    synthetic_df["sequence_id"] = (
        synthetic_df["source_record"].astype(int).astype(str)
        + "_boot_"
        + synthetic_df["bootstrap_idx"].astype(int).astype(str)
    )
    synthetic_df["group_record"] = synthetic_df["sequence_id"]
    synthetic_df["window_start_min"] = 0.0

    # Map full-record synthetic summaries to the closest validated window names.
    if "fhr_baseline_std_bpm" not in synthetic_df.columns and "fhr_std" in synthetic_df.columns:
        synthetic_df["fhr_baseline_std_bpm"] = synthetic_df["fhr_std"]
    if "fhr_baseline_mean_bpm" not in synthetic_df.columns and "fhr_mean" in synthetic_df.columns:
        synthetic_df["fhr_baseline_mean_bpm"] = synthetic_df["fhr_mean"]
    if "fhr_invalid_pct" not in synthetic_df.columns and "nan_pct" in synthetic_df.columns:
        synthetic_df["fhr_invalid_pct"] = synthetic_df["nan_pct"]

    synthetic_df = add_longitudinal_features(
        synthetic_df,
        group_columns=["source_record", "bootstrap_idx"],
        sort_columns=["source_record", "bootstrap_idx", "window_start_min"],
    )
    for column in predictor_columns:
        if column not in synthetic_df.columns:
            synthetic_df[column] = np.nan

    output_columns = predictor_columns + ["target", "group_record", "sample_source", "sequence_id"]
    note = (
        "Synthetic rows filtered to train source_record only. Current synthetic data is "
        "full-record, so sequential trend features are 0 within each one-row simulation."
    )
    return synthetic_df[output_columns].copy(), note


def make_training_bundles(split: PatientSplit, predictor_columns: list[str]) -> list[TrainingBundle]:
    """Create real-only and real+synthetic training sets for comparison."""
    base_columns = predictor_columns + ["target", "group_record", "sample_source", "sequence_id"]
    real_train = split.train_df[base_columns].copy()
    synthetic_rows, note = load_synthetic_rows(split.train_records, predictor_columns)
    if synthetic_rows.empty:
        augmented = real_train.copy()
        rows_added = 0
        positives_added = 0
    else:
        augmented = pd.concat([real_train, synthetic_rows], axis=0, ignore_index=True)
        rows_added = int(synthetic_rows.shape[0])
        positives_added = int(synthetic_rows["target"].sum())
    return [
        TrainingBundle(
            name="XGBoost sequential real-only",
            train_df=real_train,
            synthetic_rows_added=0,
            synthetic_positive_rows_added=0,
            note="Only real training windows.",
        ),
        TrainingBundle(
            name="XGBoost sequential real+synthetic",
            train_df=augmented,
            synthetic_rows_added=rows_added,
            synthetic_positive_rows_added=positives_added,
            note=note,
        ),
    ]


def import_xgb_pipeline(scale_pos_weight: float, params: dict[str, Any] | None = None) -> Pipeline:
    """Build imputer + XGBClassifier pipeline."""
    xgb_classifier = import_xgb_classifier()
    classifier = xgb_classifier(
        objective="binary:logistic",
        eval_metric="logloss",
        tree_method="hist",
        random_state=RANDOM_STATE,
        n_jobs=-1,
        scale_pos_weight=scale_pos_weight,
    )
    pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("xgb", classifier),
        ]
    )
    if params:
        pipeline.set_params(**params)
    return pipeline


def compute_scale_pos_weight(y: pd.Series) -> float:
    """Compute class-balance weight for XGBoost."""
    positives = int(y.sum())
    negatives = int(y.shape[0] - positives)
    return float(negatives / positives) if positives > 0 else 1.0


def parameter_distributions(scale_pos_weight: float) -> dict[str, list[Any]]:
    """Hyperparameter search space for sequential XGBoost."""
    return {
        "xgb__max_depth": [2, 3, 4, 5],
        "xgb__learning_rate": [0.01, 0.03, 0.05, 0.10],
        "xgb__n_estimators": [100, 200, 300, 500],
        "xgb__subsample": [0.70, 0.85, 1.00],
        "xgb__colsample_bytree": [0.70, 0.85, 1.00],
        "xgb__min_child_weight": [1, 3, 5, 10],
        "xgb__reg_lambda": [1.0, 3.0, 10.0],
        "xgb__scale_pos_weight": [1.0, math.sqrt(scale_pos_weight), scale_pos_weight],
    }


def make_group_cv(train_df: pd.DataFrame) -> StratifiedGroupKFold:
    """Create a 5-fold StratifiedGroupKFold bounded by available class groups."""
    group_target = train_df[["group_record", "target"]].drop_duplicates("group_record")
    min_class_groups = int(group_target["target"].value_counts().min())
    n_splits = min(5, min_class_groups)
    if n_splits < 2:
        raise ValueError("Not enough positive/negative groups for grouped CV.")
    return StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_STATE)


def tune_hyperparameters(train_df: pd.DataFrame, predictor_columns: list[str]) -> RandomizedSearchCV:
    """Tune XGBoost on grouped training data optimizing AUPRC."""
    y_train = train_df["target"].astype(int)
    scale_pos_weight = compute_scale_pos_weight(y_train)
    cv = make_group_cv(train_df)
    search = RandomizedSearchCV(
        estimator=import_xgb_pipeline(scale_pos_weight),
        param_distributions=parameter_distributions(scale_pos_weight),
        n_iter=N_ITER_SEARCH,
        scoring="average_precision",
        cv=cv,
        n_jobs=-1,
        random_state=RANDOM_STATE,
        verbose=1,
        refit=True,
        error_score=np.nan,
    )
    search.fit(
        train_df[predictor_columns],
        y_train,
        groups=train_df["group_record"],
    )
    return search


def out_of_fold_probabilities(
    train_df: pd.DataFrame,
    predictor_columns: list[str],
    best_params: dict[str, Any],
) -> np.ndarray:
    """Generate validation probabilities from grouped CV using fixed best params."""
    y = train_df["target"].astype(int)
    cv = make_group_cv(train_df)
    probabilities = np.full(train_df.shape[0], np.nan, dtype=float)
    scale_pos_weight = compute_scale_pos_weight(y)

    for train_idx, valid_idx in cv.split(train_df[predictor_columns], y, groups=train_df["group_record"]):
        fold_model = import_xgb_pipeline(scale_pos_weight, params=best_params)
        fold_model.fit(
            train_df.iloc[train_idx][predictor_columns],
            y.iloc[train_idx],
        )
        probabilities[valid_idx] = fold_model.predict_proba(
            train_df.iloc[valid_idx][predictor_columns]
        )[:, 1]

    if np.isnan(probabilities).any():
        raise RuntimeError("Some train rows did not receive out-of-fold probabilities.")
    return probabilities


def calibrate_threshold_for_min_recall(
    y_true: np.ndarray,
    probabilities: np.ndarray,
    *,
    min_recall: float = MIN_RECALL_FOR_THRESHOLD,
) -> float:
    """
    Choose the highest threshold whose validation recall is at least min_recall.

    This reduces alert burden while enforcing the requested sensitivity target
    on out-of-fold training predictions.
    """
    candidate_thresholds = np.unique(probabilities)
    best_threshold = float(candidate_thresholds.min()) if candidate_thresholds.size else 0.5
    for threshold in candidate_thresholds:
        predictions = (probabilities >= threshold).astype(int)
        recall = recall_score(y_true, predictions, zero_division=0)
        if recall >= min_recall:
            best_threshold = float(threshold)
        else:
            break
    return best_threshold


def compute_window_metrics(
    y_true: np.ndarray,
    probabilities: np.ndarray,
    threshold: float,
) -> dict[str, float]:
    """Compute window-level performance metrics."""
    predictions = (probabilities >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, predictions, labels=[0, 1]).ravel()
    alerts = tp + fp
    return {
        "AUROC": float(roc_auc_score(y_true, probabilities)),
        "AUPRC": float(average_precision_score(y_true, probabilities)),
        "Sensitivity": float(tp / (tp + fn)) if (tp + fn) else np.nan,
        "Specificity": float(tn / (tn + fp)) if (tn + fp) else np.nan,
        "Brier_Score": float(brier_score_loss(y_true, probabilities)),
        "Window_NNA": float(alerts / tp) if tp else float("inf"),
        "Window_PPV": float(tp / alerts) if alerts else np.nan,
        "Window_Alerts": float(alerts),
        "Window_TP": float(tp),
    }


def live_monitoring_metrics(
    test_df: pd.DataFrame,
    probabilities: np.ndarray,
    threshold: float,
) -> dict[str, float]:
    """
    Simulate chronological monitoring and compute patient-level alert metrics.

    For positive patients, Early Warning Time is the time_to_birth_min at the
    first alert in the final persistent alert suffix. If the final window is not
    above threshold, the patient has no persistent early warning.
    """
    live_df = test_df.copy()
    live_df["probability"] = probabilities
    live_df["alert"] = live_df["probability"] >= threshold
    live_df = live_df.sort_values(["record", "window_start_min"])

    patient_rows: list[dict[str, float | int]] = []
    for record, group in live_df.groupby("record", sort=False):
        target = int(group["target"].max())
        alerts = group["alert"].to_numpy(dtype=bool)
        any_alert = bool(alerts.any())
        early_warning = np.nan
        persistent_alert = False

        if target == 1 and alerts.size > 0 and alerts[-1]:
            suffix_start = alerts.size - 1
            while suffix_start > 0 and alerts[suffix_start - 1]:
                suffix_start -= 1
            persistent_alert = True
            early_warning = float(group.iloc[suffix_start]["time_to_birth_min"])
        elif target == 0 and alerts.size > 0 and alerts[-1]:
            persistent_alert = True

        patient_rows.append(
            {
                "record": int(record),
                "target": target,
                "any_alert": int(any_alert),
                "persistent_alert": int(persistent_alert),
                "early_warning_time_min": early_warning,
            }
        )

    patient_df = pd.DataFrame(patient_rows)
    positives = patient_df[patient_df["target"] == 1]
    persistent_detected = positives["early_warning_time_min"].notna()
    any_tp = int(((patient_df["target"] == 1) & (patient_df["any_alert"] == 1)).sum())
    any_alerts = int(patient_df["any_alert"].sum())

    return {
        "Patient_NNA": float(any_alerts / any_tp) if any_tp else float("inf"),
        "Patient_Alerts": float(any_alerts),
        "Patient_TP_AnyAlert": float(any_tp),
        "Patient_Recall_AnyAlert": float(any_tp / positives.shape[0]) if positives.shape[0] else np.nan,
        "Persistent_TP": float(persistent_detected.sum()),
        "Persistent_Recall": float(persistent_detected.mean()) if positives.shape[0] else np.nan,
        "Early_Warning_Median_Min": float(positives.loc[persistent_detected, "early_warning_time_min"].median())
        if persistent_detected.any()
        else np.nan,
        "Early_Warning_Mean_Min": float(positives.loc[persistent_detected, "early_warning_time_min"].mean())
        if persistent_detected.any()
        else np.nan,
    }


def evaluate_live_test(
    model: Pipeline,
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    predictor_columns: list[str],
    threshold: float,
) -> dict[str, float]:
    """Evaluate model probabilities on chronological real test windows."""
    test_sorted = test_df.sort_values(["record", "window_start_min"]).copy()
    probabilities = model.predict_proba(test_sorted[predictor_columns])[:, 1]
    y_test = test_sorted["target"].to_numpy(dtype=int)
    metrics = compute_window_metrics(y_test, probabilities, threshold)
    metrics.update(live_monitoring_metrics(test_sorted, probabilities, threshold))
    return metrics


def feature_importance_gain(
    model: Pipeline,
    predictor_columns: list[str],
    model_name: str,
) -> pd.DataFrame:
    """Extract gain-based XGBoost feature importance."""
    booster = model.named_steps["xgb"].get_booster()
    raw_scores = booster.get_score(importance_type="gain")
    rows = []
    for key, gain in raw_scores.items():
        if key.startswith("f") and key[1:].isdigit():
            index = int(key[1:])
            feature = predictor_columns[index] if index < len(predictor_columns) else key
        else:
            feature = key
        rows.append({"Model": model_name, "Feature": feature, "Gain": float(gain)})
    return pd.DataFrame(rows).sort_values("Gain", ascending=False).reset_index(drop=True)


def save_importance_artifacts(importance_df: pd.DataFrame) -> None:
    """Save importance table and a bar plot for the best sequential model."""
    FEATURE_IMPORTANCE_OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    importance_df.to_csv(FEATURE_IMPORTANCE_OUTPUT_PATH, index=False)

    try:
        import matplotlib.pyplot as plt

        best_model_name = importance_df.groupby("Model")["Gain"].sum().idxmax()
        top = (
            importance_df[importance_df["Model"] == best_model_name]
            .head(20)
            .sort_values("Gain", ascending=True)
        )
        FEATURE_IMPORTANCE_FIG_PATH.parent.mkdir(parents=True, exist_ok=True)
        plt.figure(figsize=(10, max(5, 0.35 * len(top))))
        plt.barh(top["Feature"], top["Gain"], color="#365f91")
        plt.xlabel("Ganancia media")
        plt.ylabel("Feature")
        plt.title(f"XGBoost Sequential Feature Importance - {best_model_name}")
        plt.tight_layout()
        plt.savefig(FEATURE_IMPORTANCE_FIG_PATH, dpi=180)
        plt.close()
    except Exception as exc:
        warnings.warn(f"Could not save feature importance plot: {exc}", RuntimeWarning)


def append_to_phase2_comparison(new_rows: pd.DataFrame) -> None:
    """Append sequential XGBoost rows to the requested comparison CSV."""
    if PHASE2_COMPARISON_PATH.exists():
        existing = pd.read_csv(PHASE2_COMPARISON_PATH)
        if "Model" in existing.columns:
            existing = existing[~existing["Model"].isin(new_rows["Model"])]
        combined = pd.concat([existing, new_rows], ignore_index=True, sort=False)
    else:
        combined = new_rows.copy()
    PHASE2_COMPARISON_PATH.parent.mkdir(parents=True, exist_ok=True)
    combined.to_csv(PHASE2_COMPARISON_PATH, index=False)


def train_and_evaluate_bundle(
    bundle: TrainingBundle,
    test_df: pd.DataFrame,
    predictor_columns: list[str],
) -> tuple[dict[str, Any], pd.DataFrame, dict[str, Any]]:
    """Tune, calibrate threshold and evaluate one training bundle."""
    print(f"\nEntrenando: {bundle.name}")
    print(
        f"Filas train={len(bundle.train_df)}, positivos={int(bundle.train_df['target'].sum())} "
        f"({bundle.train_df['target'].mean() * 100:.2f}%), sintéticos={bundle.synthetic_rows_added}"
    )
    search = tune_hyperparameters(bundle.train_df, predictor_columns)
    oof_probs = out_of_fold_probabilities(bundle.train_df, predictor_columns, search.best_params_)
    threshold = calibrate_threshold_for_min_recall(
        bundle.train_df["target"].to_numpy(dtype=int),
        oof_probs,
    )
    oof_auprc = float(average_precision_score(bundle.train_df["target"].astype(int), oof_probs))
    oof_recall = float(
        recall_score(
            bundle.train_df["target"].astype(int),
            (oof_probs >= threshold).astype(int),
            zero_division=0,
        )
    )
    metrics = evaluate_live_test(search.best_estimator_, bundle.train_df, test_df, predictor_columns, threshold)
    row = {
        "Model": bundle.name,
        "N_Features": len(predictor_columns),
        "Features": "all eligible predictors + sequential trend features",
        "Training_Data": "real-only" if bundle.synthetic_rows_added == 0 else "real + train-only synthetic",
        "Synthetic_Rows": bundle.synthetic_rows_added,
        "Synthetic_Positive_Rows": bundle.synthetic_positive_rows_added,
        "Best_CV_AUPRC": float(search.best_score_),
        "OOF_AUPRC": oof_auprc,
        "OOF_Recall_At_Threshold": oof_recall,
        "Threshold": threshold,
        **metrics,
    }
    metadata = {
        "model": bundle.name,
        "best_params": search.best_params_,
        "best_cv_auprc": float(search.best_score_),
        "oof_auprc": oof_auprc,
        "threshold": threshold,
        "synthetic_note": bundle.note,
    }
    importance = feature_importance_gain(search.best_estimator_, predictor_columns, bundle.name)
    return row, importance, metadata


def run_sequential_xgboost() -> pd.DataFrame:
    """Run the full sequential XGBoost experiment."""
    real_windows = load_real_windows()
    split = patient_level_split(real_windows)
    predictor_columns = get_predictor_columns(split.train_df)
    bundles = make_training_bundles(split, predictor_columns)

    print(
        f"Records train/test: {len(split.train_records)}/{len(split.test_records)}; "
        f"ventanas train/test: {len(split.train_df)}/{len(split.test_df)}"
    )
    print(f"Predictores elegibles: {len(predictor_columns)}")
    print(f"Features secuenciales añadidos: {', '.join(TREND_FEATURES)}")

    rows: list[dict[str, Any]] = []
    importance_frames: list[pd.DataFrame] = []
    params_payload: dict[str, Any] = {
        "random_state": RANDOM_STATE,
        "n_iter_search": N_ITER_SEARCH,
        "min_recall_for_threshold": MIN_RECALL_FOR_THRESHOLD,
        "train_records": len(split.train_records),
        "test_records": len(split.test_records),
        "train_windows": int(split.train_df.shape[0]),
        "test_windows": int(split.test_df.shape[0]),
        "trend_features": TREND_FEATURES,
        "models": [],
    }

    for bundle in bundles:
        row, importance, metadata = train_and_evaluate_bundle(bundle, split.test_df, predictor_columns)
        rows.append(row)
        importance_frames.append(importance)
        params_payload["models"].append(metadata)

    results_df = pd.DataFrame(rows)
    SEQUENTIAL_COMPARISON_OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    results_df.to_csv(SEQUENTIAL_COMPARISON_OUTPUT_PATH, index=False)
    append_to_phase2_comparison(results_df)

    importance_df = pd.concat(importance_frames, ignore_index=True)
    save_importance_artifacts(importance_df)

    BEST_PARAMS_OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with BEST_PARAMS_OUTPUT_PATH.open("w", encoding="utf-8") as file:
        json.dump(params_payload, file, indent=2, ensure_ascii=False)

    print("\nResultados secuenciales en TEST cronológico real:")
    display_columns = [
        "Model",
        "AUROC",
        "AUPRC",
        "Sensitivity",
        "Specificity",
        "Brier_Score",
        "Patient_NNA",
        "Persistent_Recall",
        "Early_Warning_Median_Min",
        "Threshold",
    ]
    print(results_df[display_columns].to_string(index=False))

    print("\nImportancia por ganancia - variables secuenciales:")
    sequential_importance = importance_df[importance_df["Feature"].isin(TREND_FEATURES)]
    if sequential_importance.empty:
        print("Ninguna feature secuencial fue usada por los árboles.")
    else:
        print(sequential_importance.sort_values(["Model", "Gain"], ascending=[True, False]).to_string(index=False))

    print(f"\nResultados guardados en: {SEQUENTIAL_COMPARISON_OUTPUT_PATH}")
    print(f"Comparacion solicitada actualizada en: {PHASE2_COMPARISON_PATH}")
    print(f"Parametros guardados en: {BEST_PARAMS_OUTPUT_PATH}")
    print(f"Importancias guardadas en: {FEATURE_IMPORTANCE_OUTPUT_PATH}")
    print(f"Grafico guardado en: {FEATURE_IMPORTANCE_FIG_PATH}")
    return results_df


def main() -> None:
    """CLI entrypoint."""
    run_sequential_xgboost()


if __name__ == "__main__":
    main()
