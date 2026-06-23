"""
Phase 3: non-linear windowed modeling with XGBoost.

Inputs:
    data/processed_features_windows.csv
    data/clinical_metadata.csv
    data/dataset_synthetic.csv or data/processed_features_windows_synthetic.csv (optional)
    data/results_fase2_model_comparison_windows.csv (optional baseline comparison)

Outputs:
    data/results_fase3_xgboost_comparison.csv
    data/results_fase3_xgboost_best_params.json
    data/results_fase3_xgboost_feature_importance.csv
    reports/xgboost_feature_importance_gain.png

Leakage controls:
    - Train/test split is performed at patient level using ``record``.
    - Synthetic rows are added only after the split and only when their
      ``source_record`` belongs to the training patient set.
    - Hyperparameter tuning uses StratifiedGroupKFold with patient groups.
      Synthetic rows are grouped by ``source_record``.

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
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import RandomizedSearchCV, StratifiedGroupKFold, train_test_split
from sklearn.pipeline import Pipeline


WINDOWS_INPUT_PATH = Path("data/processed_features_windows.csv")
CLINICAL_METADATA_PATH = Path("data/clinical_metadata.csv")
SYNTHETIC_WINDOWS_PATH = Path("data/processed_features_windows_synthetic.csv")
SYNTHETIC_DATASET_PATH = Path("data/dataset_synthetic.csv")
PHASE2_BASELINE_PATH = Path("data/results_fase2_model_comparison_windows.csv")

COMPARISON_OUTPUT_PATH = Path("data/results_fase3_xgboost_comparison.csv")
BEST_PARAMS_OUTPUT_PATH = Path("data/results_fase3_xgboost_best_params.json")
FEATURE_IMPORTANCE_OUTPUT_PATH = Path("data/results_fase3_xgboost_feature_importance.csv")
FEATURE_IMPORTANCE_FIG_PATH = Path("reports/xgboost_feature_importance_gain.png")

RANDOM_STATE = 42
N_ITER_SEARCH = 30

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
EXCLUDED_IDENTIFIER_COLUMNS = ["record", "dbID", "source_record", "bootstrap_idx", "is_synthetic"]
EXCLUDED_WINDOW_TRACKING_COLUMNS = [
    "window_id",
    "window_start_min",
    "window_end_min",
    "time_to_birth_min",
]


@dataclass(frozen=True)
class PatientSplit:
    """Window rows after strict patient-level splitting."""

    train_df: pd.DataFrame
    test_df: pd.DataFrame
    train_records: set[int]
    test_records: set[int]


@dataclass(frozen=True)
class SyntheticAugmentationReport:
    """Summary of synthetic rows appended to training."""

    source_path: str | None
    rows_added: int
    positive_rows_added: int
    negative_rows_added: int
    note: str


def import_xgboost_classifier() -> type[Any]:
    """Import XGBClassifier lazily so the script has a clear missing-package error."""
    try:
        from xgboost import XGBClassifier
    except ImportError as exc:
        raise ImportError(
            "xgboost no esta instalado. Instala la dependencia con: "
            "python -m pip install xgboost"
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


def load_windows_with_target(
    windows_path: Path = WINDOWS_INPUT_PATH,
    clinical_metadata_path: Path = CLINICAL_METADATA_PATH,
) -> pd.DataFrame:
    """Load window rows and map record-level causal target if absent."""
    windows_df = pd.read_csv(windows_path)
    if "target" in windows_df.columns:
        return windows_df

    clinical_df = pd.read_csv(clinical_metadata_path)
    clinical_df["target"] = create_composite_target(clinical_df)
    windows_df["target"] = windows_df["record"].map(
        clinical_df.set_index("record")["target"]
    ).astype(int)
    return windows_df


def patient_level_split(
    windows_df: pd.DataFrame,
    clinical_metadata_path: Path = CLINICAL_METADATA_PATH,
) -> PatientSplit:
    """Split unique patient IDs 70/30 before selecting window rows."""
    clinical_df = pd.read_csv(clinical_metadata_path)
    clinical_df["target"] = create_composite_target(clinical_df)

    train_records, test_records = train_test_split(
        clinical_df["record"],
        test_size=0.30,
        stratify=clinical_df["target"],
        random_state=RANDOM_STATE,
    )
    train_record_set = set(train_records.astype(int))
    test_record_set = set(test_records.astype(int))

    train_df = windows_df[windows_df["record"].astype(int).isin(train_record_set)].copy()
    test_df = windows_df[windows_df["record"].astype(int).isin(test_record_set)].copy()
    overlap = set(train_df["record"].astype(int)).intersection(set(test_df["record"].astype(int)))
    if overlap:
        raise RuntimeError(f"Patient-level leakage detected: {sorted(overlap)[:5]}")

    train_df["group_record"] = train_df["record"].astype(int)
    test_df["group_record"] = test_df["record"].astype(int)
    train_df["sample_source"] = "real"
    test_df["sample_source"] = "real"
    return PatientSplit(train_df, test_df, train_record_set, test_record_set)


def get_predictor_columns(df: pd.DataFrame) -> list[str]:
    """Select numeric predictors while excluding leakage-prone columns."""
    excluded = set(
        EXCLUDED_OUTCOME_COLUMNS
        + EXCLUDED_IDENTIFIER_COLUMNS
        + EXCLUDED_WINDOW_TRACKING_COLUMNS
        + ["target", "group_record", "sample_source"]
    )
    candidates = [column for column in df.columns if column not in excluded]
    numeric_columns = df[candidates].select_dtypes(include=[np.number]).columns
    return [column for column in numeric_columns if df[column].nunique(dropna=True) >= 2]


def _candidate_synthetic_path() -> Path | None:
    """Prefer windowed synthetic features; otherwise use the available synthetic CSV."""
    if SYNTHETIC_WINDOWS_PATH.exists():
        return SYNTHETIC_WINDOWS_PATH
    if SYNTHETIC_DATASET_PATH.exists():
        return SYNTHETIC_DATASET_PATH
    return None


def load_synthetic_training_rows(
    train_records: set[int],
    predictor_columns: list[str],
) -> tuple[pd.DataFrame, SyntheticAugmentationReport]:
    """
    Load synthetic rows and keep only rows derived from training patients.

    Current project data has synthetic FHR-only full-record summaries in
    ``data/dataset_synthetic.csv``. Those rows are aligned to the model feature
    columns; missing window-specific features are left as NaN and imputed inside
    the training pipeline. This avoids test leakage but should be interpreted as
    auxiliary augmentation, not replacement for real windowed CTG samples.
    """
    path = _candidate_synthetic_path()
    if path is None:
        return pd.DataFrame(), SyntheticAugmentationReport(
            None,
            0,
            0,
            0,
            "No synthetic feature CSV was found; training uses real windows only.",
        )

    synthetic_df = pd.read_csv(path)
    if "source_record" not in synthetic_df.columns:
        return pd.DataFrame(), SyntheticAugmentationReport(
            str(path),
            0,
            0,
            0,
            "Synthetic file skipped because source_record is missing; leakage cannot be ruled out.",
        )

    synthetic_df["source_record"] = pd.to_numeric(synthetic_df["source_record"], errors="coerce")
    synthetic_df = synthetic_df[synthetic_df["source_record"].isin(train_records)].copy()
    if synthetic_df.empty:
        return pd.DataFrame(), SyntheticAugmentationReport(
            str(path),
            0,
            0,
            0,
            "No synthetic rows matched training source records.",
        )

    if "target" not in synthetic_df.columns:
        synthetic_df["target"] = create_composite_target(synthetic_df)

    for column in predictor_columns:
        if column not in synthetic_df.columns:
            synthetic_df[column] = np.nan

    synthetic_df["group_record"] = synthetic_df["source_record"].astype(int)
    synthetic_df["sample_source"] = "synthetic"
    output_columns = predictor_columns + ["target", "group_record", "sample_source"]
    synthetic_df = synthetic_df[output_columns].copy()

    positives = int(synthetic_df["target"].sum())
    rows = int(synthetic_df.shape[0])
    return synthetic_df, SyntheticAugmentationReport(
        str(path),
        rows,
        positives,
        rows - positives,
        "Synthetic rows filtered to training source_record only.",
    )


def augment_training_set(
    train_df: pd.DataFrame,
    train_records: set[int],
    predictor_columns: list[str],
) -> tuple[pd.DataFrame, SyntheticAugmentationReport]:
    """Append leakage-safe synthetic rows to real training windows."""
    synthetic_df, report = load_synthetic_training_rows(train_records, predictor_columns)
    real_train = train_df[predictor_columns + ["target", "group_record", "sample_source"]].copy()
    if synthetic_df.empty:
        return real_train, report
    augmented = pd.concat([real_train, synthetic_df], axis=0, ignore_index=True)
    return augmented, report


def compute_scale_pos_weight(y: pd.Series) -> float:
    """Compute XGBoost scale_pos_weight from the augmented training set."""
    positives = int(y.sum())
    negatives = int(y.shape[0] - positives)
    if positives <= 0:
        return 1.0
    return float(negatives / positives)


def make_xgb_pipeline(xgb_classifier: type[Any], scale_pos_weight: float) -> Pipeline:
    """Build an imputer + XGBClassifier pipeline for hyperparameter search."""
    classifier = xgb_classifier(
        objective="binary:logistic",
        eval_metric="logloss",
        tree_method="hist",
        random_state=RANDOM_STATE,
        n_jobs=-1,
        scale_pos_weight=scale_pos_weight,
    )
    return Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("xgb", classifier),
        ]
    )


def parameter_distributions(scale_pos_weight: float) -> dict[str, list[Any]]:
    """Search space for the most clinically relevant XGBoost controls."""
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


def tune_hyperparameters(
    train_augmented: pd.DataFrame,
    predictor_columns: list[str],
    xgb_classifier: type[Any],
) -> RandomizedSearchCV:
    """Tune XGBoost with StratifiedGroupKFold to respect patient grouping."""
    x_train = train_augmented[predictor_columns]
    y_train = train_augmented["target"].astype(int)
    groups = train_augmented["group_record"].astype(int)
    scale_pos_weight = compute_scale_pos_weight(y_train)
    group_target = train_augmented[["group_record", "target"]].drop_duplicates("group_record")
    min_class_groups = int(group_target["target"].value_counts().min())
    n_splits = min(5, min_class_groups)
    if n_splits < 2:
        raise ValueError("Not enough positive/negative patient groups for StratifiedGroupKFold.")

    cv = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_STATE)
    search = RandomizedSearchCV(
        estimator=make_xgb_pipeline(xgb_classifier, scale_pos_weight),
        param_distributions=parameter_distributions(scale_pos_weight),
        n_iter=N_ITER_SEARCH,
        scoring="average_precision",
        cv=cv,
        n_jobs=-1,
        random_state=RANDOM_STATE,
        verbose=1,
        refit=True,
    )
    search.fit(x_train, y_train, groups=groups)
    return search


def choose_youden_threshold(y_true: np.ndarray, probabilities: np.ndarray) -> float:
    """Choose a classification threshold using train predictions only."""
    if np.unique(y_true).size < 2:
        return 0.5
    fpr, tpr, thresholds = roc_curve(y_true, probabilities)
    finite = np.isfinite(thresholds)
    if not finite.any():
        return 0.5
    scores = tpr[finite] - fpr[finite]
    return float(thresholds[finite][int(np.argmax(scores))])


def compute_metrics(y_true: np.ndarray, probabilities: np.ndarray, threshold: float) -> dict[str, float]:
    """Compute window-level discrimination, calibration and alert burden metrics."""
    predictions = (probabilities >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, predictions, labels=[0, 1]).ravel()
    alerts = tp + fp
    nna = float(alerts / tp) if tp > 0 else float("inf")
    ppv = float(tp / alerts) if alerts > 0 else np.nan

    return {
        "AUROC": float(roc_auc_score(y_true, probabilities)),
        "AUPRC": float(average_precision_score(y_true, probabilities)),
        "Sensitivity": float(tp / (tp + fn)) if (tp + fn) else np.nan,
        "Specificity": float(tn / (tn + fp)) if (tn + fp) else np.nan,
        "Brier_Score": float(brier_score_loss(y_true, probabilities)),
        "NNA": nna,
        "PPV": ppv,
        "Threshold": float(threshold),
        "Alerts": float(alerts),
        "True_Positives": float(tp),
    }


def evaluate_best_model(
    best_pipeline: Pipeline,
    train_augmented: pd.DataFrame,
    test_df: pd.DataFrame,
    predictor_columns: list[str],
) -> dict[str, float]:
    """Evaluate the refit XGBoost model on real held-out patient test windows."""
    train_probabilities = best_pipeline.predict_proba(train_augmented[predictor_columns])[:, 1]
    threshold = choose_youden_threshold(
        train_augmented["target"].to_numpy(dtype=int),
        train_probabilities,
    )
    test_probabilities = best_pipeline.predict_proba(test_df[predictor_columns])[:, 1]
    return compute_metrics(test_df["target"].to_numpy(dtype=int), test_probabilities, threshold)


def load_phase2_baselines(path: Path = PHASE2_BASELINE_PATH) -> pd.DataFrame:
    """Load phase-2 windowed baselines for direct comparison, if available."""
    if not path.exists():
        return pd.DataFrame()
    baselines = pd.read_csv(path)
    baselines["Evaluation"] = "Patient-level test, real windows"
    baselines["Training_Data"] = "Real train windows"
    for column in ["NNA", "PPV", "Alerts", "True_Positives", "Best_CV_AUPRC"]:
        if column not in baselines.columns:
            baselines[column] = np.nan
    return baselines


def save_feature_importance(
    best_pipeline: Pipeline,
    predictor_columns: list[str],
    output_csv: Path = FEATURE_IMPORTANCE_OUTPUT_PATH,
    output_figure: Path = FEATURE_IMPORTANCE_FIG_PATH,
    *,
    top_n: int = 20,
) -> pd.DataFrame:
    """Save XGBoost gain-based feature importance as CSV and bar chart."""
    booster = best_pipeline.named_steps["xgb"].get_booster()
    raw_scores = booster.get_score(importance_type="gain")
    rows = []
    for key, gain in raw_scores.items():
        if key.startswith("f") and key[1:].isdigit():
            index = int(key[1:])
            feature = predictor_columns[index] if index < len(predictor_columns) else key
        else:
            feature = key
        rows.append({"Feature": feature, "Gain": float(gain)})

    importance_df = pd.DataFrame(rows)
    if importance_df.empty:
        importance_df = pd.DataFrame({"Feature": predictor_columns, "Gain": np.zeros(len(predictor_columns))})
    importance_df = importance_df.sort_values("Gain", ascending=False).reset_index(drop=True)

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    importance_df.to_csv(output_csv, index=False)

    try:
        import matplotlib.pyplot as plt

        top = importance_df.head(top_n).sort_values("Gain", ascending=True)
        output_figure.parent.mkdir(parents=True, exist_ok=True)
        plt.figure(figsize=(10, max(5, 0.35 * len(top))))
        plt.barh(top["Feature"], top["Gain"], color="#2f6f73")
        plt.xlabel("Ganancia media")
        plt.ylabel("Feature")
        plt.title("XGBoost Feature Importance por Ganancia")
        plt.tight_layout()
        plt.savefig(output_figure, dpi=180)
        plt.close()
    except Exception as exc:
        warnings.warn(f"Could not save feature importance plot: {exc}", RuntimeWarning)

    return importance_df


def save_best_params(
    search: RandomizedSearchCV,
    synthetic_report: SyntheticAugmentationReport,
    split: PatientSplit,
    train_augmented: pd.DataFrame,
    output_path: Path = BEST_PARAMS_OUTPUT_PATH,
) -> None:
    """Persist tuning, split and augmentation metadata."""
    payload = {
        "random_state": RANDOM_STATE,
        "search_scoring": "average_precision",
        "n_iter_search": N_ITER_SEARCH,
        "best_cv_auprc": float(search.best_score_),
        "best_params": search.best_params_,
        "train_records": len(split.train_records),
        "test_records": len(split.test_records),
        "real_train_windows": int(split.train_df.shape[0]),
        "real_test_windows": int(split.test_df.shape[0]),
        "augmented_train_rows": int(train_augmented.shape[0]),
        "synthetic_augmentation": synthetic_report.__dict__,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as file:
        json.dump(payload, file, indent=2, ensure_ascii=False)


def run_phase3_xgboost() -> pd.DataFrame:
    """Run patient-safe XGBoost tuning, test evaluation and reporting."""
    xgb_classifier = import_xgboost_classifier()
    windows_df = load_windows_with_target()
    split = patient_level_split(windows_df)
    predictor_columns = get_predictor_columns(split.train_df)
    train_augmented, synthetic_report = augment_training_set(
        split.train_df,
        split.train_records,
        predictor_columns,
    )

    print(
        f"Records train/test: {len(split.train_records)}/{len(split.test_records)}; "
        f"ventanas train/test: {len(split.train_df)}/{len(split.test_df)}"
    )
    print(f"Predictores XGBoost: {len(predictor_columns)}")
    print(
        "Train real positivo/window: "
        f"{int(split.train_df['target'].sum())}/{len(split.train_df)} "
        f"({split.train_df['target'].mean() * 100:.2f}%)"
    )
    print(
        "Synthetic augmentation: "
        f"{synthetic_report.rows_added} filas "
        f"({synthetic_report.positive_rows_added} positivas). {synthetic_report.note}"
    )
    print(
        "Train aumentado positivo/filas: "
        f"{int(train_augmented['target'].sum())}/{len(train_augmented)} "
        f"({train_augmented['target'].mean() * 100:.2f}%)"
    )

    search = tune_hyperparameters(train_augmented, predictor_columns, xgb_classifier)
    metrics = evaluate_best_model(search.best_estimator_, train_augmented, split.test_df, predictor_columns)
    importance_df = save_feature_importance(search.best_estimator_, predictor_columns)
    save_best_params(search, synthetic_report, split, train_augmented)

    xgb_row = {
        "Model": "XGBoost windowed",
        "N_Features": len(predictor_columns),
        "Features": "all eligible predictors",
        "Evaluation": "Patient-level test, real windows",
        "Training_Data": "Real train windows + leakage-safe synthetic rows",
        "Best_CV_AUPRC": float(search.best_score_),
        **metrics,
    }
    comparison_df = pd.concat(
        [load_phase2_baselines(), pd.DataFrame([xgb_row])],
        ignore_index=True,
        sort=False,
    )
    COMPARISON_OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    comparison_df.to_csv(COMPARISON_OUTPUT_PATH, index=False)

    print("\nMejores hiperparametros:")
    print(json.dumps(search.best_params_, indent=2, ensure_ascii=False))
    print("\nEvaluacion comparativa:")
    metric_columns = [
        "Model",
        "AUROC",
        "AUPRC",
        "Sensitivity",
        "Specificity",
        "Brier_Score",
        "NNA",
        "Threshold",
    ]
    print(comparison_df[metric_columns].to_string(index=False))
    print("\nTop 15 features por ganancia:")
    print(importance_df.head(15).to_string(index=False))
    print(f"\nComparacion guardada en: {COMPARISON_OUTPUT_PATH}")
    print(f"Parametros guardados en: {BEST_PARAMS_OUTPUT_PATH}")
    print(f"Importancia CSV guardada en: {FEATURE_IMPORTANCE_OUTPUT_PATH}")
    print(f"Grafico guardado en: {FEATURE_IMPORTANCE_FIG_PATH}")
    return comparison_df


def main() -> None:
    """CLI entrypoint."""
    run_phase3_xgboost()


if __name__ == "__main__":
    main()
