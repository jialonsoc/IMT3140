"""
Pipeline 03 - Clean multivariate forward logistic models with AIC/BIC.

Inputs:
    data/nuevo_intento_processed_features_windows_real.csv

Outputs:
    data/nuevo_intento_results_fase2_windows_clean.csv
    data/nuevo_intento_results_fase2_windows_clean_coefficients.json
"""

from __future__ import annotations

import json
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from statsmodels.tools.sm_exceptions import ConvergenceWarning, PerfectSeparationError

from pipeline_01_feature_engineering import (
    CLINICAL_METADATA_PATH,
    REAL_OUTPUT_PATH,
    TREND_FEATURE_COLUMNS,
    create_composite_target,
    valid_predictor_columns,
)


RESULTS_OUTPUT_PATH = Path("data/nuevo_intento_results_fase2_windows_clean.csv")
COEFFICIENTS_OUTPUT_PATH = Path("data/nuevo_intento_results_fase2_windows_clean_coefficients.json")
RANDOM_STATE = 42
MIN_RECALL_FOR_THRESHOLD = 0.75
Criterion = Literal["aic", "bic"]


@dataclass(frozen=True)
class PatientSplit:
    """Patient-level split mapped to real windows."""

    train_df: pd.DataFrame
    test_df: pd.DataFrame
    train_records: set[int]
    test_records: set[int]


@dataclass(frozen=True)
class SelectionStep:
    """One accepted forward-selection step."""

    step: int
    feature: str
    criterion_value: float
    delta: float


@dataclass(frozen=True)
class SelectionResult:
    """Forward-selection result."""

    criterion: Criterion
    selected_features: list[str]
    steps: list[SelectionStep]
    final_criterion_value: float


def load_real_windows() -> pd.DataFrame:
    """Load pipeline 01 clean real windows."""
    if not REAL_OUTPUT_PATH.exists():
        raise FileNotFoundError(f"Run pipeline_01_feature_engineering.py first: {REAL_OUTPUT_PATH}")
    return pd.read_csv(REAL_OUTPUT_PATH)


def patient_level_split(df: pd.DataFrame) -> PatientSplit:
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
    train_df = df[df["record"].astype(int).isin(train_set)].copy()
    test_df = df[df["record"].astype(int).isin(test_set)].copy()
    if set(train_df["record"].astype(int)).intersection(set(test_df["record"].astype(int))):
        raise RuntimeError("Patient-level leakage detected.")
    return PatientSplit(train_df, test_df, train_set, test_set)


def candidate_features(train_df: pd.DataFrame) -> list[str]:
    """Valid non-constant model features."""
    columns = valid_predictor_columns(train_df)
    numeric = train_df[columns].select_dtypes(include=[np.number]).columns
    return [column for column in numeric if train_df[column].nunique(dropna=True) >= 2]


def fit_preprocessor(train_df: pd.DataFrame, features: list[str]) -> tuple[pd.DataFrame, SimpleImputer, StandardScaler]:
    """Fit imputer/scaler on train only for statsmodels selection."""
    imputer = SimpleImputer(strategy="median")
    scaler = StandardScaler()
    x_imputed = imputer.fit_transform(train_df[features])
    x_scaled = scaler.fit_transform(x_imputed)
    return pd.DataFrame(x_scaled, columns=features, index=train_df.index), imputer, scaler


def fit_logit(
    x_scaled: pd.DataFrame,
    y: pd.Series,
    selected_features: list[str],
    *,
    groups: pd.Series | None = None,
    cluster_robust: bool = False,
) -> Any | None:
    """Fit statsmodels Logit and return None on non-convergence/separation."""
    if selected_features:
        design = sm.add_constant(x_scaled[selected_features], has_constant="add")
    else:
        design = pd.DataFrame({"const": np.ones(len(y), dtype=float)}, index=y.index)
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=ConvergenceWarning)
            warnings.simplefilter("ignore", category=RuntimeWarning)
            model = sm.Logit(y.to_numpy(dtype=float), design.to_numpy(dtype=float))
            if cluster_robust and groups is not None:
                result = model.fit(
                    disp=False,
                    maxiter=300,
                    cov_type="cluster",
                    cov_kwds={"groups": groups.to_numpy()},
                )
            else:
                result = model.fit(disp=False, maxiter=300)
    except (PerfectSeparationError, np.linalg.LinAlgError, ValueError, FloatingPointError):
        return None
    if not bool(result.mle_retvals.get("converged", True)):
        return None
    return result


def value_for(result: Any, criterion: Criterion) -> float:
    """Read AIC/BIC value."""
    return float(result.aic if criterion == "aic" else result.bic)


def forward_selection(x_scaled: pd.DataFrame, y: pd.Series, criterion: Criterion) -> SelectionResult:
    """Forward stepwise selection by AIC or BIC."""
    null_model = fit_logit(x_scaled, y, [])
    if null_model is None:
        raise RuntimeError("Intercept-only model did not converge.")
    current = value_for(null_model, criterion)
    selected: list[str] = []
    remaining = list(x_scaled.columns)
    steps: list[SelectionStep] = []

    while remaining:
        best_feature: str | None = None
        best_value = np.inf
        for candidate in remaining:
            result = fit_logit(x_scaled, y, selected + [candidate])
            if result is None:
                continue
            criterion_value = value_for(result, criterion)
            if criterion_value < best_value:
                best_feature = candidate
                best_value = criterion_value
        if best_feature is None:
            break
        delta = current - best_value
        if delta <= 1e-8:
            break
        selected.append(best_feature)
        remaining.remove(best_feature)
        current = best_value
        steps.append(SelectionStep(len(steps) + 1, best_feature, current, float(delta)))
    return SelectionResult(criterion, selected, steps, current)


def make_logistic_pipeline() -> Pipeline:
    """Build fold-free train-only logistic model for final evaluation."""
    return Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            (
                "logit",
                LogisticRegression(
                    penalty=None,
                    solver="lbfgs",
                    max_iter=2000,
                    random_state=RANDOM_STATE,
                ),
            ),
        ]
    )


def calibrate_threshold(y_true: np.ndarray, probabilities: np.ndarray) -> float:
    """Highest threshold satisfying minimum train recall."""
    thresholds = np.unique(probabilities)
    best = float(thresholds.min()) if thresholds.size else 0.5
    for threshold in thresholds:
        recall = recall_score(y_true, (probabilities >= threshold).astype(int), zero_division=0)
        if recall >= MIN_RECALL_FOR_THRESHOLD:
            best = float(threshold)
        else:
            break
    return best


def live_metrics(test_df: pd.DataFrame, probabilities: np.ndarray, threshold: float) -> dict[str, float]:
    """Window and patient-level chronological metrics."""
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
                "record": int(record),
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
        "Threshold": float(threshold),
        "Patient_NNA": float(patient_alerts / patient_tp_any) if patient_tp_any else float("inf"),
        "Patient_Recall_AnyAlert": float(patient_tp_any / positives.shape[0]) if positives.shape[0] else np.nan,
        "Early_Warning_Median_Min": float(positives.loc[persistent, "early_warning_time_min"].median())
        if persistent.any()
        else np.nan,
    }


def evaluate_model(train_df: pd.DataFrame, test_df: pd.DataFrame, features: list[str]) -> dict[str, float]:
    """Train on real train windows and evaluate chronologically on real test windows."""
    model = make_logistic_pipeline()
    model.fit(train_df[features], train_df["target"].astype(int))
    train_prob = model.predict_proba(train_df[features])[:, 1]
    threshold = calibrate_threshold(train_df["target"].to_numpy(dtype=int), train_prob)
    test_ordered = test_df.sort_values(["record", "window_start_min"]).copy()
    test_prob = model.predict_proba(test_ordered[features])[:, 1]
    return live_metrics(test_ordered, test_prob, threshold)


def _safe_exp(value: float) -> float:
    """Safe exponentiation for odds ratios."""
    if value > np.log(np.finfo(float).max):
        return float("inf")
    if value < np.log(np.finfo(float).tiny):
        return 0.0
    return float(np.exp(value))


def coefficient_summary(
    x_scaled: pd.DataFrame,
    train_df: pd.DataFrame,
    selected: list[str],
) -> list[dict[str, float | str]]:
    """Cluster-robust coefficient summary for final selected model."""
    result = fit_logit(
        x_scaled,
        train_df["target"].astype(int),
        selected,
        groups=train_df["record"].astype(int),
        cluster_robust=True,
    )
    if result is None:
        return []
    params = np.asarray(result.params, dtype=float)
    pvalues = np.asarray(result.pvalues, dtype=float)
    rows: list[dict[str, float | str]] = []
    for idx, feature in enumerate(["Intercept"] + selected):
        coef = float(params[idx])
        rows.append(
            {
                "Feature": feature,
                "Coefficient": coef,
                "Odds_Ratio": _safe_exp(coef),
                "P_Value": float(pvalues[idx]),
            }
        )
    return rows


def run_fase2() -> pd.DataFrame:
    """Execute clean AIC/BIC forward logistic models."""
    df = load_real_windows()
    split = patient_level_split(df)
    features = candidate_features(split.train_df)
    x_scaled, _imputer, _scaler = fit_preprocessor(split.train_df, features)
    y_train = split.train_df["target"].astype(int)

    selections = {
        "AIC": forward_selection(x_scaled, y_train, "aic"),
        "BIC": forward_selection(x_scaled, y_train, "bic"),
    }
    for name, selection in selections.items():
        print(f"\n{name} entry order:")
        for step in selection.steps:
            print(f"  {step.step:02d}. {step.feature} ({name}={step.criterion_value:.3f}, delta={step.delta:.3f})")

    rows: list[dict[str, Any]] = []
    coefficient_payload: dict[str, Any] = {
        "trend_features_available": TREND_FEATURE_COLUMNS,
        "models": {},
    }
    for name, selection in selections.items():
        metrics = evaluate_model(split.train_df, split.test_df, selection.selected_features)
        rows.append(
            {
                "Model": f"Clean Forward {name}",
                "N_Features": len(selection.selected_features),
                "Sequential_Features_Used": ", ".join(
                    [f for f in selection.selected_features if f in TREND_FEATURE_COLUMNS]
                ),
                "Training_Data": "Real train windows",
                "Features": ", ".join(selection.selected_features),
                **metrics,
            }
        )
        coefficient_payload["models"][name] = {
            "selection_order": [step.__dict__ for step in selection.steps],
            "final_criterion": selection.final_criterion_value,
            "coefficients": coefficient_summary(x_scaled, split.train_df, selection.selected_features),
        }

    results = pd.DataFrame(rows)
    RESULTS_OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(RESULTS_OUTPUT_PATH, index=False)
    with COEFFICIENTS_OUTPUT_PATH.open("w", encoding="utf-8") as file:
        json.dump(coefficient_payload, file, indent=2, ensure_ascii=False)

    print("\nClean Fase 2 test metrics:")
    print(results.to_string(index=False))
    print(f"\nSaved: {RESULTS_OUTPUT_PATH}")
    print(f"Saved coefficients: {COEFFICIENTS_OUTPUT_PATH}")
    return results


def main() -> None:
    """CLI entrypoint."""
    run_fase2()


if __name__ == "__main__":
    main()
