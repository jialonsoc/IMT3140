"""Nested patient-grouped validation for XGBoost and CatBoost.

Outer folds estimate performance on real patients. Inner folds tune the model
and calibrate its decision threshold. Synthetic windows are allowed only in a
training fold and only when their source patient belongs to that fold.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import RandomizedSearchCV, StratifiedGroupKFold

from pipeline_04_fase3_advanced import (
    MIN_RECALL_FOR_THRESHOLD,
    ModelSpec,
    candidate_features,
    load_real_windows,
    load_synthetic_windows,
    model_specs,
    patient_level_split,
    scale_pos_weight,
)


OUTPUT_DIR = Path("data/nuevo_intento")
FOLD_RESULTS_PATH = OUTPUT_DIR / "cross_validation_fold_metrics.csv"
SUMMARY_PATH = OUTPUT_DIR / "cross_validation_summary.csv"
DETAILS_PATH = OUTPUT_DIR / "cross_validation_details.json"

METRICS = ("Accuracy", "Precision", "Recall", "F1", "ROC_AUC", "AUPRC")


@dataclass(frozen=True)
class CrossValidationConfig:
    """Configuration for nested grouped cross-validation."""

    n_splits: int = 5
    inner_splits: int = 3
    n_iter_search: int = 8
    random_state: int = 42
    n_jobs: int = -1
    min_recall: float = MIN_RECALL_FOR_THRESHOLD


@dataclass(frozen=True)
class InnerFold:
    """Indices for one inner fold in real-only and combined matrices."""

    real_train_idx: np.ndarray
    real_valid_idx: np.ndarray
    combined_train_idx: np.ndarray
    combined_valid_idx: np.ndarray


def _bounded_split_count(df: pd.DataFrame, requested: int) -> int:
    """Bound folds by the number of positive and negative patient groups."""
    target_counts = df.groupby("group_record")["target"].nunique(dropna=False)
    if (target_counts != 1).any():
        raise ValueError("Each patient must have exactly one target.")
    patient_targets = df[["group_record", "target"]].drop_duplicates("group_record")
    smallest_class = int(patient_targets["target"].value_counts().min())
    n_splits = min(requested, smallest_class, patient_targets.shape[0])
    if n_splits < 2:
        raise ValueError("At least two patient groups per class are required.")
    return n_splits


def _splitter(df: pd.DataFrame, requested: int, random_state: int) -> StratifiedGroupKFold:
    """Create a reproducible stratified patient-group splitter."""
    return StratifiedGroupKFold(
        n_splits=_bounded_split_count(df, requested),
        shuffle=True,
        random_state=random_state,
    )


def _combine_training_rows(
    real_train: pd.DataFrame,
    synthetic_df: pd.DataFrame,
    features: list[str],
    use_synthetic: bool,
) -> pd.DataFrame:
    """Add only synthetic derivatives of patients present in real_train."""
    columns = features + ["target", "group_record", "sample_source"]
    real = real_train[columns].copy()
    if not use_synthetic:
        return real.reset_index(drop=True)

    allowed_groups = set(real_train["group_record"].astype(int))
    synthetic = synthetic_df[
        synthetic_df["source_record"].astype(int).isin(allowed_groups)
    ][columns].copy()
    combined = pd.concat([real, synthetic], ignore_index=True)
    if not set(combined.loc[combined["sample_source"].eq("synthetic"), "group_record"]).issubset(allowed_groups):
        raise RuntimeError("Synthetic source leakage detected in a training fold.")
    return combined


def _inner_folds(
    real_train: pd.DataFrame,
    combined_train: pd.DataFrame,
    config: CrossValidationConfig,
) -> list[InnerFold]:
    """Build inner folds whose validation rows are always 100% real."""
    splitter = _splitter(real_train, config.inner_splits, config.random_state)
    y = real_train["target"].astype(int)
    groups = real_train["group_record"].astype(int)
    folds: list[InnerFold] = []

    for real_train_idx, real_valid_idx in splitter.split(real_train, y, groups):
        train_groups = set(groups.iloc[real_train_idx])
        valid_groups = set(groups.iloc[real_valid_idx])
        if train_groups.intersection(valid_groups):
            raise RuntimeError("Patient leakage detected in an inner fold.")

        combined_groups = combined_train["group_record"].astype(int)
        combined_train_idx = np.flatnonzero(combined_groups.isin(train_groups).to_numpy())
        combined_valid_idx = np.flatnonzero(
            combined_groups.isin(valid_groups).to_numpy()
            & combined_train["sample_source"].eq("real").to_numpy()
        )
        folds.append(
            InnerFold(
                real_train_idx=np.asarray(real_train_idx),
                real_valid_idx=np.asarray(real_valid_idx),
                combined_train_idx=combined_train_idx,
                combined_valid_idx=combined_valid_idx,
            )
        )
    return folds


def _tune_in_outer_fold(
    spec: ModelSpec,
    combined_train: pd.DataFrame,
    features: list[str],
    inner_folds: list[InnerFold],
    config: CrossValidationConfig,
) -> RandomizedSearchCV:
    """Tune with explicit splits so synthetic rows never enter validation."""
    y = combined_train["target"].astype(int)
    pos_weight = scale_pos_weight(y)
    estimator = spec.estimator_factory(pos_weight)
    if spec.name == "XGBoost":
        estimator.set_params(model__n_jobs=1)
    elif spec.name == "CatBoost":
        estimator.set_params(model__thread_count=1)

    search = RandomizedSearchCV(
        estimator=estimator,
        param_distributions=spec.param_distributions_factory(pos_weight),
        n_iter=config.n_iter_search,
        scoring="average_precision",
        cv=[(fold.combined_train_idx, fold.combined_valid_idx) for fold in inner_folds],
        n_jobs=config.n_jobs,
        random_state=config.random_state,
        refit=True,
        error_score="raise",
    )
    search.fit(combined_train[features], y)
    return search


def _calibrate_inner_threshold(
    estimator: Any,
    real_train: pd.DataFrame,
    combined_train: pd.DataFrame,
    features: list[str],
    inner_folds: list[InnerFold],
    min_recall: float,
) -> float:
    """Calibrate the threshold from real inner-fold out-of-fold predictions."""
    probabilities = np.full(real_train.shape[0], np.nan, dtype=float)
    for fold in inner_folds:
        model = clone(estimator)
        train_rows = combined_train.iloc[fold.combined_train_idx]
        valid_rows = real_train.iloc[fold.real_valid_idx]
        model.fit(train_rows[features], train_rows["target"].astype(int))
        probabilities[fold.real_valid_idx] = model.predict_proba(valid_rows[features])[:, 1]
    if np.isnan(probabilities).any():
        raise RuntimeError("Threshold calibration left real training rows without OOF predictions.")
    y = real_train["target"].to_numpy(dtype=int)
    thresholds = np.unique(probabilities)
    best = float(thresholds.min()) if thresholds.size else 0.5
    for threshold in thresholds:
        recall = float(np.mean(probabilities[y == 1] >= threshold)) if np.any(y == 1) else 0.0
        if recall >= min_recall:
            best = float(threshold)
        else:
            break
    return best


def _classification_metrics(
    y_true: np.ndarray,
    probabilities: np.ndarray,
    threshold: float,
) -> dict[str, float]:
    """Compute thresholded and ranking metrics for one outer fold."""
    predictions = (probabilities >= threshold).astype(int)
    return {
        "Accuracy": float(accuracy_score(y_true, predictions)),
        "Precision": float(precision_score(y_true, predictions, zero_division=0)),
        "Recall": float(recall_score(y_true, predictions, zero_division=0)),
        "F1": float(f1_score(y_true, predictions, zero_division=0)),
        "ROC_AUC": float(roc_auc_score(y_true, probabilities)),
        "AUPRC": float(average_precision_score(y_true, probabilities)),
    }


def cross_validate_model(
    spec: ModelSpec,
    real_train_df: pd.DataFrame,
    synthetic_df: pd.DataFrame,
    features: list[str],
    use_synthetic: bool,
    config: CrossValidationConfig,
) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    """Run nested grouped CV for one model and one training-data experiment."""
    outer = _splitter(real_train_df, config.n_splits, config.random_state)
    y = real_train_df["target"].astype(int)
    groups = real_train_df["group_record"].astype(int)
    training_label = "Real+Synthetic" if use_synthetic else "Real"
    rows: list[dict[str, Any]] = []
    details: list[dict[str, Any]] = []

    for fold_number, (train_idx, valid_idx) in enumerate(
        outer.split(real_train_df, y, groups), start=1
    ):
        outer_real_train = real_train_df.iloc[train_idx].reset_index(drop=True)
        outer_real_valid = real_train_df.iloc[valid_idx].reset_index(drop=True)
        train_groups = set(outer_real_train["group_record"].astype(int))
        valid_groups = set(outer_real_valid["group_record"].astype(int))
        if train_groups.intersection(valid_groups):
            raise RuntimeError("Patient leakage detected in an outer fold.")

        combined_train = _combine_training_rows(
            outer_real_train, synthetic_df, features, use_synthetic
        )
        inner_folds = _inner_folds(outer_real_train, combined_train, config)
        search = _tune_in_outer_fold(spec, combined_train, features, inner_folds, config)
        threshold = _calibrate_inner_threshold(
            search.best_estimator_,
            outer_real_train,
            combined_train,
            features,
            inner_folds,
            config.min_recall,
        )
        probabilities = search.best_estimator_.predict_proba(outer_real_valid[features])[:, 1]
        metrics = _classification_metrics(
            outer_real_valid["target"].to_numpy(dtype=int), probabilities, threshold
        )
        synthetic_rows = int(combined_train["sample_source"].eq("synthetic").sum())
        row = {
            "Model": spec.name,
            "Training_Data": training_label,
            "Fold": fold_number,
            "Train_Patients": len(train_groups),
            "Validation_Patients": len(valid_groups),
            "Train_Real_Windows": int(outer_real_train.shape[0]),
            "Train_Synthetic_Windows": synthetic_rows,
            "Validation_Real_Windows": int(outer_real_valid.shape[0]),
            "Threshold": float(threshold),
            **metrics,
        }
        rows.append(row)
        details.append(
            {
                "model": spec.name,
                "training_data": training_label,
                "fold": fold_number,
                "best_inner_auprc": float(search.best_score_),
                "best_params": search.best_params_,
                "threshold": float(threshold),
                "train_patient_ids": [int(value) for value in sorted(train_groups)],
                "validation_patient_ids": [int(value) for value in sorted(valid_groups)],
            }
        )
        print(
            f"{spec.name} | {training_label} | fold {fold_number}: "
            f"AUPRC={metrics['AUPRC']:.4f}, ROC-AUC={metrics['ROC_AUC']:.4f}, "
            f"Recall={metrics['Recall']:.4f}"
        )
    return pd.DataFrame(rows), details


def summarize_fold_metrics(fold_results: pd.DataFrame) -> pd.DataFrame:
    """Aggregate mean and sample standard deviation for each metric."""
    rows: list[dict[str, Any]] = []
    for (model, training_data), group in fold_results.groupby(["Model", "Training_Data"]):
        row: dict[str, Any] = {
            "Model": model,
            "Training_Data": training_data,
            "N_Folds": int(group.shape[0]),
        }
        for metric in METRICS:
            row[f"{metric}_Mean"] = float(group[metric].mean())
            row[f"{metric}_Std"] = float(group[metric].std(ddof=1))
        row["Threshold_Mean"] = float(group["Threshold"].mean())
        row["Threshold_Std"] = float(group["Threshold"].std(ddof=1))
        rows.append(row)
    return pd.DataFrame(rows).sort_values("AUPRC_Mean", ascending=False).reset_index(drop=True)


def run_validation(config: CrossValidationConfig) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Validate XGBoost/CatBoost with real-only and mixed training data."""
    real_df = load_real_windows()
    synthetic_df = load_synthetic_windows()
    split = patient_level_split(real_df)
    features = candidate_features(split.train_df)
    if "Weight_g" in features:
        raise RuntimeError("Weight_g must not be present in the clean predictor set.")

    all_results: list[pd.DataFrame] = []
    all_details: list[dict[str, Any]] = []
    for use_synthetic in (False, True):
        for spec in model_specs():
            fold_df, details = cross_validate_model(
                spec=spec,
                real_train_df=split.train_df,
                synthetic_df=synthetic_df,
                features=features,
                use_synthetic=use_synthetic,
                config=config,
            )
            all_results.append(fold_df)
            all_details.extend(details)

    fold_results = pd.concat(all_results, ignore_index=True)
    summary = summarize_fold_metrics(fold_results)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    fold_results.to_csv(FOLD_RESULTS_PATH, index=False)
    summary.to_csv(SUMMARY_PATH, index=False)
    with DETAILS_PATH.open("w", encoding="utf-8") as file:
        json.dump(
            {
                "problem_type": "binary_classification",
                "primary_metric": "AUPRC",
                "configuration": asdict(config),
                "n_features": len(features),
                "features": features,
                "external_test_untouched": True,
                "folds": all_details,
            },
            file,
            indent=2,
            ensure_ascii=False,
        )

    print("\nNested grouped cross-validation summary:")
    print(summary.to_string(index=False))
    best = summary.iloc[0]
    stable = summary.sort_values("AUPRC_Std", ascending=True).iloc[0]
    print(
        f"\nBest mean AUPRC: {best['Model']} ({best['Training_Data']}) "
        f"= {best['AUPRC_Mean']:.4f}"
    )
    print(
        f"Most stable AUPRC: {stable['Model']} ({stable['Training_Data']}) "
        f"std = {stable['AUPRC_Std']:.4f}"
    )
    return fold_results, summary
