"""
Phase 2: multivariate logistic models with forward selection by AIC and BIC.

Inputs:
    data/processed_features.csv
    data/results_fase1_univariado.csv  (optional, used to identify phase-1 best)

Outputs:
    data/results_fase2_coefficients.json
    data/results_fase2_model_comparison.csv

The target matches Phase 1 Strategy B:
    target = 1 if pH < 7.10 OR Apgar5 < 7 OR NICU_days > 0.

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
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from statsmodels.tools.sm_exceptions import ConvergenceWarning, PerfectSeparationError


INPUT_PATH = Path("data/processed_features.csv")
PHASE1_RESULTS_PATH = Path("data/results_fase1_univariado.csv")
COEFFICIENTS_OUTPUT_PATH = Path("data/results_fase2_coefficients.json")
COMPARISON_OUTPUT_PATH = Path("data/results_fase2_model_comparison.csv")
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


def get_predictor_columns(df: pd.DataFrame) -> list[str]:
    """
    Keep numeric predictors only, excluding direct outcome variables and IDs.

    Constant columns are removed because they make logistic models singular and
    cannot discriminate the target.
    """
    excluded = set(EXCLUDED_OUTCOME_COLUMNS + EXCLUDED_IDENTIFIER_COLUMNS + ["target"])
    candidate_columns = [column for column in df.columns if column not in excluded]
    numeric_columns = df[candidate_columns].select_dtypes(include=[np.number]).columns
    return [column for column in numeric_columns if df[column].nunique(dropna=True) >= 2]


def split_train_test(
    x: pd.DataFrame,
    y: pd.Series,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Create the same stratified 70/30 split strategy used in Phase 1."""
    return train_test_split(
        x,
        y,
        test_size=0.30,
        stratify=y,
        random_state=RANDOM_STATE,
    )


def fit_train_preprocessor(x_train: pd.DataFrame) -> tuple[pd.DataFrame, SimpleImputer, StandardScaler]:
    """Median-impute and standardize training predictors for statsmodels selection."""
    imputer = SimpleImputer(strategy="median")
    scaler = StandardScaler()
    x_imputed = imputer.fit_transform(x_train)
    x_scaled = scaler.fit_transform(x_imputed)
    x_scaled_df = pd.DataFrame(x_scaled, columns=x_train.columns, index=x_train.index)
    return x_scaled_df, imputer, scaler


def fit_statsmodels_logit(
    x: pd.DataFrame,
    y: pd.Series,
    selected_features: list[str],
) -> sm.discrete.discrete_model.BinaryResultsWrapper | None:
    """Fit an unpenalized statsmodels Logit model, returning None on failure."""
    if selected_features:
        design = sm.add_constant(x[selected_features], has_constant="add")
    else:
        design = pd.DataFrame({"const": np.ones(len(y), dtype=float)}, index=y.index)

    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=ConvergenceWarning)
            warnings.simplefilter("ignore", category=RuntimeWarning)
            model = sm.Logit(y.to_numpy(dtype=float), design.to_numpy(dtype=float))
            result = model.fit(disp=False, maxiter=250)
    except (PerfectSeparationError, np.linalg.LinAlgError, ValueError, FloatingPointError):
        return None

    if not bool(result.mle_retvals.get("converged", True)):
        return None
    return result


def criterion_value(result: sm.discrete.discrete_model.BinaryResultsWrapper, criterion: Criterion) -> float:
    """Read AIC or BIC from a fitted Logit result."""
    return float(result.aic if criterion == "aic" else result.bic)


def forward_selection(
    x_train_scaled: pd.DataFrame,
    y_train: pd.Series,
    *,
    criterion: Criterion,
    min_delta: float = 1e-8,
) -> ForwardSelectionResult:
    """
    Run forward selection using AIC or BIC as stopping criterion.

    The algorithm starts at the intercept-only model and accepts a candidate
    only if it reduces the chosen information criterion.
    """
    null_model = fit_statsmodels_logit(x_train_scaled, y_train, [])
    if null_model is None:
        raise RuntimeError("The intercept-only Logit model did not converge.")

    current_value = criterion_value(null_model, criterion)
    selected: list[str] = []
    steps: list[SelectionStep] = []
    remaining = list(x_train_scaled.columns)

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
                best_value = value
                best_feature = candidate

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

    return ForwardSelectionResult(
        criterion=criterion,
        selected_features=selected,
        steps=steps,
        final_criterion_value=current_value,
    )


def choose_youden_threshold(y_true: np.ndarray, probabilities: np.ndarray) -> float:
    """Choose a fold-training threshold that maximizes sensitivity + specificity - 1."""
    if np.unique(y_true).size < 2:
        return 0.5
    fpr, tpr, thresholds = roc_curve(y_true, probabilities)
    finite = np.isfinite(thresholds)
    if not finite.any():
        return 0.5
    scores = tpr[finite] - fpr[finite]
    return float(thresholds[finite][int(np.argmax(scores))])


def make_logistic_pipeline() -> Pipeline:
    """Build a fold-local imputer, scaler and unpenalized logistic model."""
    return Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            (
                "logit",
                LogisticRegression(
                    penalty=None,
                    solver="lbfgs",
                    max_iter=1000,
                    random_state=RANDOM_STATE,
                ),
            ),
        ]
    )


def predict_with_feature_set(
    x_train_fold: pd.DataFrame,
    y_train_fold: pd.Series,
    x_valid_fold: pd.DataFrame,
    selected_features: list[str],
) -> tuple[np.ndarray, float]:
    """
    Fit a fold-local model and return validation probabilities plus threshold.

    If no features are selected, the model is an intercept-only prevalence
    model and the Youden threshold falls back to 0.5.
    """
    if not selected_features:
        prevalence = float(y_train_fold.mean())
        return np.full(x_valid_fold.shape[0], prevalence, dtype=float), 0.5

    pipeline = make_logistic_pipeline()
    try:
        pipeline.fit(x_train_fold[selected_features], y_train_fold)
        train_probabilities = pipeline.predict_proba(x_train_fold[selected_features])[:, 1]
        valid_probabilities = pipeline.predict_proba(x_valid_fold[selected_features])[:, 1]
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
        fallback.fit(x_train_fold[selected_features], y_train_fold)
        train_probabilities = fallback.predict_proba(x_train_fold[selected_features])[:, 1]
        valid_probabilities = fallback.predict_proba(x_valid_fold[selected_features])[:, 1]

    threshold = choose_youden_threshold(y_train_fold.to_numpy(), train_probabilities)
    return valid_probabilities, threshold


def evaluate_model_cv(
    x_train: pd.DataFrame,
    y_train: pd.Series,
    selected_features: list[str],
    *,
    n_splits: int = 5,
) -> dict[str, float]:
    """Evaluate a fixed feature set by stratified CV on the training split."""
    min_class_count = int(y_train.value_counts().min())
    splits = min(n_splits, min_class_count)
    if splits < 2:
        raise ValueError("Not enough positive/negative cases for stratified CV.")

    cv = StratifiedKFold(n_splits=splits, shuffle=True, random_state=RANDOM_STATE)
    y_all: list[int] = []
    p_all: list[float] = []
    pred_all: list[int] = []

    for train_idx, valid_idx in cv.split(x_train, y_train):
        x_fold_train = x_train.iloc[train_idx]
        y_fold_train = y_train.iloc[train_idx]
        x_fold_valid = x_train.iloc[valid_idx]
        y_fold_valid = y_train.iloc[valid_idx]

        probabilities, threshold = predict_with_feature_set(
            x_fold_train,
            y_fold_train,
            x_fold_valid,
            selected_features,
        )
        predictions = (probabilities >= threshold).astype(int)
        y_all.extend(y_fold_valid.astype(int).tolist())
        p_all.extend(probabilities.astype(float).tolist())
        pred_all.extend(predictions.astype(int).tolist())

    y_array = np.asarray(y_all, dtype=int)
    p_array = np.asarray(p_all, dtype=float)
    pred_array = np.asarray(pred_all, dtype=int)
    tn, fp, fn, tp = confusion_matrix(y_array, pred_array, labels=[0, 1]).ravel()

    return {
        "AUROC": float(roc_auc_score(y_array, p_array)),
        "AUPRC": float(average_precision_score(y_array, p_array)),
        "Sensitivity": float(tp / (tp + fn)) if (tp + fn) else np.nan,
        "Specificity": float(tn / (tn + fp)) if (tn + fp) else np.nan,
        "Brier_Score": float(brier_score_loss(y_array, p_array)),
    }


def get_best_phase1_feature(available_features: list[str]) -> str:
    """Read the best Phase 1 feature, falling back to the first available feature."""
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
    selected_features: list[str],
) -> list[dict[str, float | str]]:
    """Fit the final selected model and summarize coefficients and p-values."""
    result = fit_statsmodels_logit(x_train_scaled, y_train, selected_features)
    if result is None:
        return []

    rows: list[dict[str, float | str]] = []
    params = np.asarray(result.params, dtype=float)
    pvalues = np.asarray(result.pvalues, dtype=float)

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
    title = selection.criterion.upper()
    print(f"\nOrden de ingreso - {title}:")
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
    x_train: pd.DataFrame,
    y_train: pd.Series,
    selections: dict[str, list[str]],
) -> pd.DataFrame:
    """Evaluate Phase 1 baseline, AIC model and BIC model in train CV."""
    rows: list[dict[str, float | int | str]] = []
    for model_name, features in selections.items():
        metrics = evaluate_model_cv(x_train, y_train, features, n_splits=5)
        rows.append(
            {
                "Model": model_name,
                "N_Features": len(features),
                "Features": ", ".join(features) if features else "(intercept-only)",
                **metrics,
            }
        )
    return pd.DataFrame(rows)


def run_phase2(
    input_path: Path = INPUT_PATH,
    coefficients_output_path: Path = COEFFICIENTS_OUTPUT_PATH,
    comparison_output_path: Path = COMPARISON_OUTPUT_PATH,
) -> tuple[pd.DataFrame, dict[str, object]]:
    """Run all Phase 2 selection, evaluation and reporting steps."""
    df = pd.read_csv(input_path)
    y = create_composite_target(df)
    x = df[get_predictor_columns(df)]
    prevalence = float(y.mean())
    print(f"Prevalencia target compuesto: {int(y.sum())}/{len(y)} ({prevalence * 100:.2f}%)")
    print(f"Predictores candidatos: {x.shape[1]}")

    x_train, x_test, y_train, y_test = split_train_test(x, y)
    print(
        "Split estratificado: "
        f"train={len(y_train)} ({y_train.mean() * 100:.2f}% positivos), "
        f"test={len(y_test)} ({y_test.mean() * 100:.2f}% positivos)"
    )

    x_train_scaled, _imputer, _scaler = fit_train_preprocessor(x_train)
    aic_selection = forward_selection(x_train_scaled, y_train, criterion="aic")
    bic_selection = forward_selection(x_train_scaled, y_train, criterion="bic")

    print_selection_order(aic_selection)
    print_selection_order(bic_selection)

    best_phase1_feature = get_best_phase1_feature(list(x.columns))
    selections = {
        f"Fase 1 univariado ({best_phase1_feature})": [best_phase1_feature],
        "Multivariado Forward AIC": aic_selection.selected_features,
        "Multivariado Forward BIC": bic_selection.selected_features,
    }
    comparison_df = build_comparison_table(x_train, y_train, selections)
    comparison_output_path.parent.mkdir(parents=True, exist_ok=True)
    comparison_df.to_csv(comparison_output_path, index=False)

    report: dict[str, object] = {
        "target_definition": "pH < 7.10 OR Apgar5 < 7 OR NICU_days > 0",
        "random_state": RANDOM_STATE,
        "train_size": int(len(y_train)),
        "test_size": int(len(y_test)),
        "target_prevalence_total": prevalence,
        "target_prevalence_train": float(y_train.mean()),
        "target_prevalence_test": float(y_test.mean()),
        "aic_selection_order": [step.__dict__ for step in aic_selection.steps],
        "bic_selection_order": [step.__dict__ for step in bic_selection.steps],
        "aic_final_criterion": aic_selection.final_criterion_value,
        "bic_final_criterion": bic_selection.final_criterion_value,
        "aic_coefficients": coefficient_summary(
            x_train_scaled,
            y_train,
            aic_selection.selected_features,
        ),
        "bic_coefficients": coefficient_summary(
            x_train_scaled,
            y_train,
            bic_selection.selected_features,
        ),
    }
    with coefficients_output_path.open("w", encoding="utf-8") as file:
        json.dump(report, file, indent=2, ensure_ascii=False)

    print("\nComparacion final por validacion cruzada en train:")
    print(comparison_df.to_string(index=False))
    print(f"\nCoeficientes guardados en: {coefficients_output_path}")
    print(f"Comparacion guardada en: {comparison_output_path}")
    return comparison_df, report


def main() -> None:
    """CLI entrypoint."""
    run_phase2()


if __name__ == "__main__":
    main()
