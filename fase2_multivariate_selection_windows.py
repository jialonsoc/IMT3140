"""
Phase 2 windowed multivariate logistic selection with patient-level holdout.

Inputs:
    data/processed_features_windows.csv
    data/clinical_metadata.csv
    data/results_fase1_univariado_windows.csv  (optional baseline source)

Outputs:
    data/results_fase2_coefficients_windows.json
    data/results_fase2_model_comparison_windows.csv

Forward selection uses global train-set Logit AIC/BIC. Final coefficient
summaries use cluster-robust covariance by ``record`` to reflect repeated
windows per patient. Final predictive metrics are computed on an independent
patient-level test split.

Install notes:
    pip install numpy pandas scikit-learn statsmodels
"""

from __future__ import annotations

import json
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from statsmodels.tools.sm_exceptions import ConvergenceWarning, PerfectSeparationError


INPUT_PATH = Path("data/processed_features_windows.csv")
CLINICAL_METADATA_PATH = Path("data/clinical_metadata.csv")
PHASE1_RESULTS_PATH = Path("data/results_fase1_univariado_windows.csv")
COEFFICIENTS_OUTPUT_PATH = Path("data/results_fase2_coefficients_windows.json")
COMPARISON_OUTPUT_PATH = Path("data/results_fase2_model_comparison_windows.csv")
RANDOM_STATE = 42

Criterion = Literal["aic", "bic"]

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
EXCLUDED_IDENTIFIER_COLUMNS = ["record", "dbID"]
EXCLUDED_WINDOW_TRACKING_COLUMNS = [
    "window_id",
    "window_start_min",
    "window_end_min",
    "time_to_birth_min",
]


@dataclass(frozen=True)
class PatientSplit:
    """Window rows after patient-level splitting."""

    train_df: pd.DataFrame
    test_df: pd.DataFrame
    train_records: set[int]
    test_records: set[int]


@dataclass(frozen=True)
class SelectionStep:
    """One accepted step in forward selection."""

    step: int
    feature: str
    criterion_value: float
    delta: float


@dataclass(frozen=True)
class ForwardSelectionResult:
    """Forward-selection output for one information criterion."""

    criterion: Criterion
    selected_features: list[str]
    steps: list[SelectionStep]
    final_criterion_value: float


def create_composite_target(df: pd.DataFrame) -> pd.Series:
    """Create Strategy B composite outcome."""
    missing = [column for column in TARGET_COLUMNS if column not in df.columns]
    if missing:
        raise KeyError(f"Missing target columns: {missing}")
    return (
        (df["pH"] < 7.10) | (df["Apgar5"] < 7.0) | (df["NICU_days"] > 0.0)
    ).astype(int).rename("target")


def load_windows_with_record_target(
    windows_path: Path = INPUT_PATH,
    clinical_metadata_path: Path = CLINICAL_METADATA_PATH,
) -> pd.DataFrame:
    """Load window rows and map target from unique patient records."""
    windows_df = pd.read_csv(windows_path)
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
    """Split unique records 70/30 before mapping windows to train/test."""
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

    return PatientSplit(train_df, test_df, train_record_set, test_record_set)


def get_predictor_columns(df: pd.DataFrame) -> list[str]:
    """Select numeric predictors, excluding outcome, IDs and temporal tracking."""
    excluded = set(
        EXCLUDED_OUTCOME_COLUMNS
        + EXCLUDED_IDENTIFIER_COLUMNS
        + EXCLUDED_WINDOW_TRACKING_COLUMNS
        + ["target"]
    )
    candidates = [column for column in df.columns if column not in excluded]
    numeric_columns = df[candidates].select_dtypes(include=[np.number]).columns
    return [column for column in numeric_columns if df[column].nunique(dropna=True) >= 2]


def fit_train_preprocessor(x_train: pd.DataFrame) -> tuple[pd.DataFrame, SimpleImputer, StandardScaler]:
    """Fit median imputer and scaler on train windows only."""
    imputer = SimpleImputer(strategy="median")
    scaler = StandardScaler()
    x_imputed = imputer.fit_transform(x_train)
    x_scaled = scaler.fit_transform(x_imputed)
    return pd.DataFrame(x_scaled, columns=x_train.columns, index=x_train.index), imputer, scaler


def fit_statsmodels_logit(
    x_scaled: pd.DataFrame,
    y: pd.Series,
    selected_features: list[str],
    *,
    groups: pd.Series | None = None,
    cluster_robust: bool = False,
) -> sm.discrete.discrete_model.BinaryResultsWrapper | None:
    """Fit a Logit model, optionally with cluster-robust covariance by record."""
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
                    maxiter=250,
                    cov_type="cluster",
                    cov_kwds={"groups": groups.to_numpy()},
                )
            else:
                result = model.fit(disp=False, maxiter=250)
    except (PerfectSeparationError, np.linalg.LinAlgError, ValueError, FloatingPointError):
        return None

    if not bool(result.mle_retvals.get("converged", True)):
        return None
    return result


def criterion_value(result: sm.discrete.discrete_model.BinaryResultsWrapper, criterion: Criterion) -> float:
    """Read AIC or BIC from a fitted statsmodels Logit result."""
    return float(result.aic if criterion == "aic" else result.bic)


def forward_selection(
    x_train_scaled: pd.DataFrame,
    y_train: pd.Series,
    *,
    criterion: Criterion,
    min_delta: float = 1e-8,
) -> ForwardSelectionResult:
    """Run forward selection by reducing AIC/BIC on the training windows."""
    null_model = fit_statsmodels_logit(x_train_scaled, y_train, [])
    if null_model is None:
        raise RuntimeError("The intercept-only Logit model did not converge.")

    current_value = criterion_value(null_model, criterion)
    selected: list[str] = []
    remaining = list(x_train_scaled.columns)
    steps: list[SelectionStep] = []

    while remaining:
        best_feature: str | None = None
        best_value = np.inf

        for candidate in remaining:
            candidate_features = selected + [candidate]
            result = fit_statsmodels_logit(x_train_scaled, y_train, candidate_features)
            if result is None:
                continue
            value = criterion_value(result, criterion)
            if value < best_value:
                best_feature = candidate
                best_value = value

        if best_feature is None:
            break

        delta = current_value - best_value
        if delta <= min_delta:
            break

        selected.append(best_feature)
        remaining.remove(best_feature)
        current_value = best_value
        steps.append(
            SelectionStep(
                step=len(steps) + 1,
                feature=best_feature,
                criterion_value=current_value,
                delta=float(delta),
            )
        )

    return ForwardSelectionResult(criterion, selected, steps, current_value)


def make_logistic_pipeline() -> Pipeline:
    """Build train-only imputer/scaler and unpenalized logistic regression."""
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


def choose_youden_threshold(y_true: np.ndarray, probabilities: np.ndarray) -> float:
    """Choose threshold on train predictions only."""
    if np.unique(y_true).size < 2:
        return 0.5
    fpr, tpr, thresholds = roc_curve(y_true, probabilities)
    finite = np.isfinite(thresholds)
    if not finite.any():
        return 0.5
    scores = tpr[finite] - fpr[finite]
    return float(thresholds[finite][int(np.argmax(scores))])


def fit_predict_test(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    selected_features: list[str],
) -> tuple[np.ndarray, float]:
    """Train on train windows and return test probabilities plus train threshold."""
    if not selected_features:
        prevalence = float(train_df["target"].mean())
        return np.full(test_df.shape[0], prevalence, dtype=float), 0.5

    pipeline = make_logistic_pipeline()
    try:
        pipeline.fit(train_df[selected_features], train_df["target"])
        train_probabilities = pipeline.predict_proba(train_df[selected_features])[:, 1]
        test_probabilities = pipeline.predict_proba(test_df[selected_features])[:, 1]
    except Exception:
        fallback = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
                (
                    "logit",
                    LogisticRegression(
                        solver="liblinear",
                        class_weight="balanced",
                        random_state=RANDOM_STATE,
                    ),
                ),
            ]
        )
        fallback.fit(train_df[selected_features], train_df["target"])
        train_probabilities = fallback.predict_proba(train_df[selected_features])[:, 1]
        test_probabilities = fallback.predict_proba(test_df[selected_features])[:, 1]

    threshold = choose_youden_threshold(train_df["target"].to_numpy(), train_probabilities)
    return test_probabilities, threshold


def evaluate_on_test(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    selected_features: list[str],
) -> dict[str, float]:
    """Evaluate a fixed feature set on the independent patient-level test windows."""
    probabilities, threshold = fit_predict_test(train_df, test_df, selected_features)
    y_test = test_df["target"].to_numpy(dtype=int)
    predictions = (probabilities >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_test, predictions, labels=[0, 1]).ravel()

    return {
        "AUROC": float(roc_auc_score(y_test, probabilities)),
        "AUPRC": float(average_precision_score(y_test, probabilities)),
        "Sensitivity": float(tp / (tp + fn)) if (tp + fn) else np.nan,
        "Specificity": float(tn / (tn + fp)) if (tn + fp) else np.nan,
        "Brier_Score": float(brier_score_loss(y_test, probabilities)),
        "Threshold": float(threshold),
    }


def get_best_phase1_feature(available_features: list[str]) -> str:
    """Read the top windowed Phase 1 feature, falling back to first candidate."""
    if PHASE1_RESULTS_PATH.exists():
        phase1 = pd.read_csv(PHASE1_RESULTS_PATH)
        for feature in phase1["Feature"].dropna().astype(str):
            if feature in available_features:
                return feature
    return available_features[0]


def _safe_exp(value: float) -> float:
    """Exponentiate safely for odds ratios."""
    if value > np.log(np.finfo(float).max):
        return float("inf")
    if value < np.log(np.finfo(float).tiny):
        return 0.0
    return float(np.exp(value))


def coefficient_summary(
    x_train_scaled: pd.DataFrame,
    y_train: pd.Series,
    groups: pd.Series,
    selected_features: list[str],
) -> list[dict[str, float | str]]:
    """Summarize final model coefficients with clustered p-values by record."""
    result = fit_statsmodels_logit(
        x_train_scaled,
        y_train,
        selected_features,
        groups=groups,
        cluster_robust=True,
    )
    if result is None:
        return []

    params = np.asarray(result.params, dtype=float)
    pvalues = np.asarray(result.pvalues, dtype=float)
    rows: list[dict[str, float | str]] = []
    for idx, feature in enumerate(["Intercept"] + selected_features):
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


def print_selection_order(selection: ForwardSelectionResult) -> None:
    """Print selected variables in entry order."""
    print(f"\nOrden de ingreso - {selection.criterion.upper()}:")
    if not selection.steps:
        print("  Modelo nulo; no ingreso ninguna variable.")
        return
    for step in selection.steps:
        print(
            f"  {step.step:02d}. {step.feature} "
            f"({selection.criterion.upper()}={step.criterion_value:.3f}, "
            f"delta={step.delta:.3f})"
        )


def build_comparison_table(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    selections: dict[str, list[str]],
) -> pd.DataFrame:
    """Evaluate baseline, AIC and BIC models on the held-out patient test set."""
    rows: list[dict[str, float | int | str]] = []
    for model_name, features in selections.items():
        metrics = evaluate_on_test(train_df, test_df, features)
        rows.append(
            {
                "Model": model_name,
                "N_Features": len(features),
                "Features": ", ".join(features) if features else "(intercept-only)",
                **metrics,
            }
        )
    return pd.DataFrame(rows)


def run_phase2_windows(
    input_path: Path = INPUT_PATH,
    coefficients_output_path: Path = COEFFICIENTS_OUTPUT_PATH,
    comparison_output_path: Path = COMPARISON_OUTPUT_PATH,
) -> tuple[pd.DataFrame, dict[str, object]]:
    """Run windowed forward selection and independent patient-test evaluation."""
    windows_df = load_windows_with_record_target(input_path, CLINICAL_METADATA_PATH)
    split = patient_level_split(windows_df, CLINICAL_METADATA_PATH)
    predictor_columns = get_predictor_columns(split.train_df)

    print(
        f"Records train/test: {len(split.train_records)}/{len(split.test_records)}; "
        f"ventanas train/test: {len(split.train_df)}/{len(split.test_df)}"
    )
    print(
        f"Prevalencia ventanas train/test: "
        f"{split.train_df['target'].mean() * 100:.2f}%/"
        f"{split.test_df['target'].mean() * 100:.2f}%"
    )
    print(f"Predictores candidatos: {len(predictor_columns)}")

    x_train_scaled, _imputer, _scaler = fit_train_preprocessor(split.train_df[predictor_columns])
    y_train = split.train_df["target"]
    groups_train = split.train_df["record"]

    aic_selection = forward_selection(x_train_scaled, y_train, criterion="aic")
    bic_selection = forward_selection(x_train_scaled, y_train, criterion="bic")
    print_selection_order(aic_selection)
    print_selection_order(bic_selection)

    best_phase1_feature = get_best_phase1_feature(predictor_columns)
    selections = {
        f"Fase 1 univariado windowed ({best_phase1_feature})": [best_phase1_feature],
        "Multivariado Forward AIC windowed": aic_selection.selected_features,
        "Multivariado Forward BIC windowed": bic_selection.selected_features,
    }
    comparison_df = build_comparison_table(split.train_df, split.test_df, selections)
    comparison_output_path.parent.mkdir(parents=True, exist_ok=True)
    comparison_df.to_csv(comparison_output_path, index=False)

    report: dict[str, object] = {
        "target_definition": "pH < 7.10 OR Apgar5 < 7 OR NICU_days > 0",
        "split_level": "record",
        "random_state": RANDOM_STATE,
        "train_records": len(split.train_records),
        "test_records": len(split.test_records),
        "train_windows": int(split.train_df.shape[0]),
        "test_windows": int(split.test_df.shape[0]),
        "aic_selection_order": [step.__dict__ for step in aic_selection.steps],
        "bic_selection_order": [step.__dict__ for step in bic_selection.steps],
        "aic_final_criterion": aic_selection.final_criterion_value,
        "bic_final_criterion": bic_selection.final_criterion_value,
        "aic_coefficients": coefficient_summary(
            x_train_scaled,
            y_train,
            groups_train,
            aic_selection.selected_features,
        ),
        "bic_coefficients": coefficient_summary(
            x_train_scaled,
            y_train,
            groups_train,
            bic_selection.selected_features,
        ),
    }
    with coefficients_output_path.open("w", encoding="utf-8") as file:
        json.dump(report, file, indent=2, ensure_ascii=False)

    print("\nComparacion final en TEST independiente por paciente:")
    print(comparison_df.to_string(index=False))
    print(f"\nCoeficientes guardados en: {coefficients_output_path}")
    print(f"Comparacion guardada en: {comparison_output_path}")
    return comparison_df, report


def main() -> None:
    """CLI entrypoint."""
    run_phase2_windows()


if __name__ == "__main__":
    main()
