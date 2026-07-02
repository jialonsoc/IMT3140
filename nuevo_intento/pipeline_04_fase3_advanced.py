"""
Pipeline 04 - Advanced clean tree models and final comparative report.

Experiments:
    A. Sequential clean features + 100% real training windows.
    B. Sequential clean features + real training windows plus train-only synthetic windows.

Models:
    xgboost.XGBClassifier
    catboost.CatBoostClassifier

Final output:
    data/nuevo_intento/final_comparative_report.csv
"""

from __future__ import annotations

import json
import math
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

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

from pipeline_01_feature_engineering import (
    CLINICAL_METADATA_PATH,
    REAL_OUTPUT_PATH,
    SYNTHETIC_OUTPUT_PATH,
    TREND_FEATURE_COLUMNS,
    create_composite_target,
    valid_predictor_columns,
)


FINAL_REPORT_PATH = Path("data/nuevo_intento/final_comparative_report.csv")
BEST_PARAMS_PATH = Path("data/nuevo_intento/fase3_best_params.json")
IMPORTANCE_OUTPUT_PATH = Path("data/nuevo_intento/fase3_feature_importance.csv")
IMPORTANCE_FIG_PATH = Path("reports/nuevo_intento_fase3_feature_importance.png")
FASE2_CLEAN_PATH = Path("data/nuevo_intento/results_fase2_windows_clean.csv")

RANDOM_STATE = 42
N_ITER_SEARCH = 20
MIN_RECALL_FOR_THRESHOLD = 0.75

FINAL_COLUMNS = [
    "Model",
    "N_Features",
    "Sequential_Features_Used",
    "Training_Data",
    "AUROC",
    "AUPRC",
    "Sensitivity",
    "Specificity",
    "Brier_Score",
    "Threshold",
    "Patient_NNA",
    "Patient_Recall_AnyAlert",
    "Early_Warning_Median_Min",
]


@dataclass(frozen=True)
class PatientSplit:
    """Patient-level train/test split."""

    train_df: pd.DataFrame
    test_df: pd.DataFrame
    train_records: set[int]
    test_records: set[int]


@dataclass(frozen=True)
class Experiment:
    """One advanced-model training experiment."""

    name: str
    training_data_label: str
    train_df: pd.DataFrame


@dataclass(frozen=True)
class ModelSpec:
    """Model factory and search-space definition."""

    name: str
    estimator_factory: Callable[[float], Pipeline]
    param_distributions_factory: Callable[[float], dict[str, list[Any]]]


def import_xgb_classifier() -> type[Any]:
    """Import XGBoost lazily."""
    try:
        from xgboost import XGBClassifier
    except ImportError as exc:
        raise ImportError("Install XGBoost with: python -m pip install xgboost") from exc
    return XGBClassifier


def import_catboost_classifier() -> type[Any]:
    """Import CatBoost lazily."""
    try:
        from catboost import CatBoostClassifier
    except ImportError as exc:
        raise ImportError("Install CatBoost with: python -m pip install catboost") from exc
    return CatBoostClassifier


def load_real_windows() -> pd.DataFrame:
    """Load clean real windows."""
    if not REAL_OUTPUT_PATH.exists():
        raise FileNotFoundError(f"Run pipeline_01_feature_engineering.py first: {REAL_OUTPUT_PATH}")
    return pd.read_csv(REAL_OUTPUT_PATH)


def load_synthetic_windows() -> pd.DataFrame:
    """Load clean synthetic windows."""
    if not SYNTHETIC_OUTPUT_PATH.exists():
        raise FileNotFoundError(f"Run pipeline_01_feature_engineering.py first: {SYNTHETIC_OUTPUT_PATH}")
    return pd.read_csv(SYNTHETIC_OUTPUT_PATH)


def patient_level_split(real_df: pd.DataFrame) -> PatientSplit:
    """Split patients before selecting windows."""
    clinical = pd.read_csv(CLINICAL_METADATA_PATH)
    clinical["target"] = create_composite_target(clinical)
    train_records, test_records = train_test_split(
        clinical["record"],
        test_size=0.30,
        stratify=clinical["target"],
        random_state=RANDOM_STATE,
    )
    train_set = set(train_records.astype(int))
    test_set = set(test_records.astype(int))
    train_df = real_df[real_df["record"].astype(int).isin(train_set)].copy()
    test_df = real_df[real_df["record"].astype(int).isin(test_set)].copy()
    if set(train_df["record"].astype(int)).intersection(set(test_df["record"].astype(int))):
        raise RuntimeError("Patient-level leakage detected.")
    return PatientSplit(train_df, test_df, train_set, test_set)


def candidate_features(train_df: pd.DataFrame) -> list[str]:
    """Valid non-constant predictors under the golden rule."""
    columns = valid_predictor_columns(train_df)
    numeric = train_df[columns].select_dtypes(include=[np.number]).columns
    return [column for column in numeric if train_df[column].nunique(dropna=True) >= 2]


def build_experiments(split: PatientSplit, synthetic_df: pd.DataFrame, features: list[str]) -> list[Experiment]:
    """Create real-only and real+synthetic training matrices."""
    base_columns = features + ["target", "group_record", "sample_source"]
    real_train = split.train_df[base_columns].copy()
    synthetic_train = synthetic_df[synthetic_df["source_record"].astype(int).isin(split.train_records)].copy()
    synthetic_train = synthetic_train[base_columns].copy()
    mixed_train = pd.concat([real_train, synthetic_train], ignore_index=True)
    print(
        f"Synthetic train-only rows: {len(synthetic_train)} "
        f"({int(synthetic_train['target'].sum())} positives)"
    )
    return [
        Experiment("Sequential real-only", "Train 100% Real", real_train),
        Experiment("Sequential real+synthetic", "Train Mixed Real+Synthetic", mixed_train),
    ]


def scale_pos_weight(y: pd.Series) -> float:
    """Class imbalance ratio for weighted tree training."""
    positives = int(y.sum())
    negatives = int(len(y) - positives)
    return float(negatives / positives) if positives else 1.0


def xgb_pipeline_factory(pos_weight: float) -> Pipeline:
    """XGBoost pipeline."""
    xgb = import_xgb_classifier()
    return Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            (
                "model",
                xgb(
                    objective="binary:logistic",
                    eval_metric="logloss",
                    tree_method="hist",
                    n_jobs=-1,
                    random_state=RANDOM_STATE,
                    scale_pos_weight=pos_weight,
                ),
            ),
        ]
    )


def xgb_param_space(pos_weight: float) -> dict[str, list[Any]]:
    """XGBoost randomized-search space."""
    return {
        "model__max_depth": [2, 3, 4, 5],
        "model__learning_rate": [0.01, 0.03, 0.05, 0.10],
        "model__n_estimators": [100, 200, 300, 500],
        "model__subsample": [0.70, 0.85, 1.00],
        "model__colsample_bytree": [0.70, 0.85, 1.00],
        "model__min_child_weight": [1, 3, 5, 10],
        "model__reg_lambda": [1.0, 3.0, 10.0],
        "model__scale_pos_weight": [1.0, math.sqrt(pos_weight), pos_weight],
    }


def catboost_pipeline_factory(_pos_weight: float) -> Pipeline:
    """CatBoost pipeline."""
    catboost = import_catboost_classifier()
    return Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            (
                "model",
                catboost(
                    loss_function="Logloss",
                    eval_metric="AUC",
                    auto_class_weights="Balanced",
                    random_seed=RANDOM_STATE,
                    verbose=False,
                    allow_writing_files=False,
                    thread_count=-1,
                ),
            ),
        ]
    )


def catboost_param_space(_pos_weight: float) -> dict[str, list[Any]]:
    """CatBoost randomized-search space."""
    return {
        "model__depth": [2, 3, 4, 5],
        "model__learning_rate": [0.01, 0.03, 0.05, 0.10],
        "model__iterations": [100, 200, 300, 500],
        "model__l2_leaf_reg": [1.0, 3.0, 10.0],
        "model__random_strength": [0.5, 1.0, 2.0],
    }


def model_specs() -> list[ModelSpec]:
    """Advanced model specifications."""
    return [
        ModelSpec("XGBoost", xgb_pipeline_factory, xgb_param_space),
        ModelSpec("CatBoost", catboost_pipeline_factory, catboost_param_space),
    ]


def grouped_cv(train_df: pd.DataFrame) -> StratifiedGroupKFold:
    """Grouped stratified CV bounded by positive/negative patient groups."""
    group_targets = train_df[["group_record", "target"]].drop_duplicates("group_record")
    n_splits = min(5, int(group_targets["target"].value_counts().min()))
    if n_splits < 2:
        raise ValueError("Not enough groups for StratifiedGroupKFold.")
    return StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_STATE)


def tune_model(spec: ModelSpec, train_df: pd.DataFrame, features: list[str]) -> RandomizedSearchCV:
    """Tune one model with grouped CV optimizing AUPRC."""
    y = train_df["target"].astype(int)
    pos_weight = scale_pos_weight(y)
    search = RandomizedSearchCV(
        estimator=spec.estimator_factory(pos_weight),
        param_distributions=spec.param_distributions_factory(pos_weight),
        n_iter=N_ITER_SEARCH,
        scoring="average_precision",
        cv=grouped_cv(train_df),
        n_jobs=-1,
        random_state=RANDOM_STATE,
        verbose=1,
        refit=True,
        error_score=np.nan,
    )
    search.fit(train_df[features], y, groups=train_df["group_record"])
    return search


def out_of_fold_probabilities(
    estimator: Pipeline,
    train_df: pd.DataFrame,
    features: list[str],
) -> np.ndarray:
    """Generate grouped out-of-fold probabilities for threshold calibration."""
    y = train_df["target"].astype(int)
    probs = np.full(len(train_df), np.nan, dtype=float)
    for train_idx, valid_idx in grouped_cv(train_df).split(train_df[features], y, groups=train_df["group_record"]):
        model = clone(estimator)
        model.fit(train_df.iloc[train_idx][features], y.iloc[train_idx])
        probs[valid_idx] = model.predict_proba(train_df.iloc[valid_idx][features])[:, 1]
    if np.isnan(probs).any():
        raise RuntimeError("Missing out-of-fold predictions.")
    return probs


def calibrate_threshold(y: np.ndarray, probabilities: np.ndarray) -> float:
    """Highest threshold with train OOF recall >= 75%."""
    thresholds = np.unique(probabilities)
    best = float(thresholds.min()) if thresholds.size else 0.5
    for threshold in thresholds:
        recall = np.mean((probabilities >= threshold)[y == 1]) if np.any(y == 1) else 0.0
        if recall >= MIN_RECALL_FOR_THRESHOLD:
            best = float(threshold)
        else:
            break
    return best


def live_test_metrics(test_df: pd.DataFrame, probabilities: np.ndarray, threshold: float) -> dict[str, float]:
    """Chronological live-monitoring metrics on real test windows."""
    ordered = test_df.sort_values(["record", "window_start_min"]).copy()
    ordered["probability"] = probabilities
    ordered["alert"] = ordered["probability"] >= threshold
    y = ordered["target"].to_numpy(dtype=int)
    pred = ordered["alert"].to_numpy(dtype=int)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()

    patient_rows: list[dict[str, float | int]] = []
    for record, group in ordered.groupby("record", sort=False):
        target = int(group["target"].max())
        alerts = group["alert"].to_numpy(dtype=bool)
        any_alert = bool(alerts.any())
        early_warning = np.nan
        if target == 1 and alerts.size and alerts[-1]:
            start = alerts.size - 1
            while start > 0 and alerts[start - 1]:
                start -= 1
            early_warning = float(group.iloc[start]["time_to_birth_min"])
        patient_rows.append(
            {
                "target": target,
                "any_alert": int(any_alert),
                "early_warning_time_min": early_warning,
            }
        )
    patient_df = pd.DataFrame(patient_rows)
    positives = patient_df[patient_df["target"] == 1]
    patient_tp_any = int(((patient_df["target"] == 1) & (patient_df["any_alert"] == 1)).sum())
    patient_alerts = int(patient_df["any_alert"].sum())
    persistent = positives["early_warning_time_min"].notna()

    return {
        "AUROC": float(roc_auc_score(y, probabilities)),
        "AUPRC": float(average_precision_score(y, probabilities)),
        "Sensitivity": float(tp / (tp + fn)) if (tp + fn) else np.nan,
        "Specificity": float(tn / (tn + fp)) if (tn + fp) else np.nan,
        "Brier_Score": float(brier_score_loss(y, probabilities)),
        "Patient_NNA": float(patient_alerts / patient_tp_any) if patient_tp_any else float("inf"),
        "Patient_Recall_AnyAlert": float(patient_tp_any / positives.shape[0]) if positives.shape[0] else np.nan,
        "Early_Warning_Median_Min": float(positives.loc[persistent, "early_warning_time_min"].median())
        if persistent.any()
        else np.nan,
    }


def feature_importance(estimator: Pipeline, features: list[str], model_name: str) -> pd.DataFrame:
    """Extract model feature importance for audit."""
    model = estimator.named_steps["model"]
    rows: list[dict[str, float | str]] = []
    if hasattr(model, "get_booster"):
        scores = model.get_booster().get_score(importance_type="gain")
        for key, gain in scores.items():
            idx = int(key[1:]) if key.startswith("f") and key[1:].isdigit() else -1
            feature = features[idx] if 0 <= idx < len(features) else key
            rows.append({"Model": model_name, "Feature": feature, "Gain": float(gain)})
    elif hasattr(model, "get_feature_importance"):
        values = model.get_feature_importance()
        for feature, gain in zip(features, values):
            rows.append({"Model": model_name, "Feature": feature, "Gain": float(gain)})
    return pd.DataFrame(rows).sort_values("Gain", ascending=False)


def save_importance_plot(importance_df: pd.DataFrame) -> None:
    """Save feature-importance CSV and a compact plot."""
    IMPORTANCE_OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    importance_df.to_csv(IMPORTANCE_OUTPUT_PATH, index=False)
    try:
        import matplotlib.pyplot as plt

        top = importance_df.head(25).sort_values("Gain", ascending=True)
        IMPORTANCE_FIG_PATH.parent.mkdir(parents=True, exist_ok=True)
        plt.figure(figsize=(11, max(6, 0.35 * len(top))))
        plt.barh(top["Model"] + " | " + top["Feature"], top["Gain"], color="#345f8c")
        plt.xlabel("Gain / importance")
        plt.tight_layout()
        plt.savefig(IMPORTANCE_FIG_PATH, dpi=180)
        plt.close()
    except Exception as exc:
        warnings.warn(f"Could not save feature importance plot: {exc}", RuntimeWarning)


def load_fase2_baselines() -> pd.DataFrame:
    """Load clean AIC/BIC baselines if pipeline 03 was run."""
    if not FASE2_CLEAN_PATH.exists():
        return pd.DataFrame(columns=FINAL_COLUMNS)
    df = pd.read_csv(FASE2_CLEAN_PATH)
    for column in FINAL_COLUMNS:
        if column not in df.columns:
            df[column] = np.nan
    return df[FINAL_COLUMNS].copy()


def run_advanced() -> pd.DataFrame:
    """Run all advanced experiments and export the final comparative report."""
    real_df = load_real_windows()
    synthetic_df = load_synthetic_windows()
    split = patient_level_split(real_df)
    features = candidate_features(split.train_df)
    experiments = build_experiments(split, synthetic_df, features)

    print(
        f"Train/test records: {len(split.train_records)}/{len(split.test_records)}; "
        f"real windows: {len(split.train_df)}/{len(split.test_df)}"
    )
    print(f"Valid features: {len(features)}")

    rows: list[dict[str, Any]] = []
    importance_frames: list[pd.DataFrame] = []
    params_payload: dict[str, Any] = {
        "random_state": RANDOM_STATE,
        "n_iter_search": N_ITER_SEARCH,
        "min_recall_for_threshold": MIN_RECALL_FOR_THRESHOLD,
        "features": features,
        "models": [],
    }

    for experiment in experiments:
        for spec in model_specs():
            model_name = f"{spec.name} - {experiment.name}"
            print(f"\nTraining {model_name}")
            search = tune_model(spec, experiment.train_df, features)
            oof = out_of_fold_probabilities(search.best_estimator_, experiment.train_df, features)
            threshold = calibrate_threshold(experiment.train_df["target"].to_numpy(dtype=int), oof)
            test_ordered = split.test_df.sort_values(["record", "window_start_min"]).copy()
            test_prob = search.best_estimator_.predict_proba(test_ordered[features])[:, 1]
            metrics = live_test_metrics(test_ordered, test_prob, threshold)
            rows.append(
                {
                    "Model": model_name,
                    "N_Features": len(features),
                    "Sequential_Features_Used": ", ".join([f for f in TREND_FEATURE_COLUMNS if f in features]),
                    "Training_Data": experiment.training_data_label,
                    **metrics,
                    "Threshold": float(threshold),
                }
            )
            importance_frames.append(feature_importance(search.best_estimator_, features, model_name))
            params_payload["models"].append(
                {
                    "model": model_name,
                    "best_params": search.best_params_,
                    "best_cv_auprc": float(search.best_score_),
                    "threshold": float(threshold),
                }
            )

    advanced_df = pd.DataFrame(rows)
    final_df = pd.concat([load_fase2_baselines(), advanced_df[FINAL_COLUMNS]], ignore_index=True)
    FINAL_REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    final_df[FINAL_COLUMNS].to_csv(FINAL_REPORT_PATH, index=False)

    importance_df = pd.concat(importance_frames, ignore_index=True) if importance_frames else pd.DataFrame()
    save_importance_plot(importance_df)
    with BEST_PARAMS_PATH.open("w", encoding="utf-8") as file:
        json.dump(params_payload, file, indent=2, ensure_ascii=False)

    print("\nFinal comparative report:")
    print(final_df[FINAL_COLUMNS].to_string(index=False))
    print("\nSequential feature gain audit:")
    trend_importance = importance_df[importance_df["Feature"].isin(TREND_FEATURE_COLUMNS)]
    print(trend_importance.sort_values(["Model", "Gain"], ascending=[True, False]).to_string(index=False))
    print(f"\nSaved final report: {FINAL_REPORT_PATH}")
    return final_df


def main() -> None:
    """CLI entrypoint."""
    run_advanced()


if __name__ == "__main__":
    main()
