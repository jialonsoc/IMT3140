"""
Phase 1: univariate clinical ML analysis for intrapartum asphyxia.

Input:
    data/processed_features.csv

Output:
    data/results_fase1_univariado.csv

The binary outcome follows "Strategy B - composite outcome":
    target = 1 if pH < 7.10 OR Apgar5 < 7 OR NICU_days > 0.

Install notes:
    pip install numpy pandas scikit-learn statsmodels
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


INPUT_PATH = Path("data/processed_features.csv")
OUTPUT_PATH = Path("data/results_fase1_univariado.csv")
RANDOM_STATE = 42

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
class UnivariateResult:
    """Container for one feature's univariate analysis result."""

    feature: str
    auroc_cv: float
    odds_ratio: float
    ci_lower: float
    ci_upper: float
    p_value: float


def create_composite_target(df: pd.DataFrame) -> pd.Series:
    """
    Create the Strategy B binary target.

    A row is positive when any of the following is true:
    pH < 7.10, Apgar5 < 7, or NICU_days > 0.
    """
    missing = [column for column in TARGET_COLUMNS if column not in df.columns]
    if missing:
        raise KeyError(f"Missing target columns: {missing}")

    target = (
        (df["pH"] < 7.10) | (df["Apgar5"] < 7.0) | (df["NICU_days"] > 0.0)
    ).astype(int)
    return target.rename("target")


def print_target_prevalence(target: pd.Series) -> None:
    """Print positive-class prevalence and flag unexpected class balance."""
    positives = int(target.sum())
    total = int(target.shape[0])
    prevalence = positives / total if total else np.nan
    print(
        f"Prevalencia clase positiva: {positives}/{total} "
        f"({prevalence * 100:.2f}%)"
    )
    if not 0.02 <= prevalence <= 0.04:
        print(
            "Advertencia: la prevalencia no esta entre 2% y 4% con la "
            "definicion compuesta solicitada."
        )


def get_predictor_columns(df: pd.DataFrame) -> list[str]:
    """
    Select numeric predictor columns after removing outcome leakage and IDs.

    Non-numeric columns are excluded because this phase asks for single-feature
    logistic models without categorical encoding.
    """
    excluded = set(EXCLUDED_OUTCOME_COLUMNS + EXCLUDED_IDENTIFIER_COLUMNS + ["target"])
    candidate_columns = [column for column in df.columns if column not in excluded]
    numeric_columns = df[candidate_columns].select_dtypes(include=[np.number]).columns
    return list(numeric_columns)


def stratified_train_test_split(
    df: pd.DataFrame,
    target: pd.Series,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Create a 70/30 stratified split to keep test data untouched."""
    return train_test_split(
        df,
        target,
        test_size=0.30,
        stratify=target,
        random_state=RANDOM_STATE,
    )


def compute_cv_auroc(
    x: pd.Series,
    y: pd.Series,
    *,
    n_splits: int = 5,
) -> float:
    """
    Estimate univariate AUROC using stratified CV on the training set.

    Imputation and scaling are fitted inside each fold to avoid leakage from
    validation folds into preprocessing parameters.
    """
    if x.nunique(dropna=True) < 2:
        return np.nan

    min_class_count = int(y.value_counts().min())
    splits = min(n_splits, min_class_count)
    if splits < 2:
        return np.nan

    cv = StratifiedKFold(n_splits=splits, shuffle=True, random_state=RANDOM_STATE)
    aucs: list[float] = []

    for train_idx, valid_idx in cv.split(x.to_frame(), y):
        x_train = x.iloc[train_idx].to_frame()
        y_train = y.iloc[train_idx]
        x_valid = x.iloc[valid_idx].to_frame()
        y_valid = y.iloc[valid_idx]

        if y_valid.nunique() < 2:
            continue

        model = Pipeline(
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
        model.fit(x_train, y_train)
        probabilities = model.predict_proba(x_valid)[:, 1]
        aucs.append(float(roc_auc_score(y_valid, probabilities)))

    return float(np.mean(aucs)) if aucs else np.nan


def _safe_exp(value: float) -> float:
    """Exponentiate a log-odds value and return inf instead of warning on overflow."""
    if value > np.log(np.finfo(float).max):
        return float("inf")
    if value < np.log(np.finfo(float).tiny):
        return 0.0
    return float(np.exp(value))


def fit_statsmodels_logit(x: pd.Series, y: pd.Series) -> tuple[float, float, float, float]:
    """
    Fit one standardized-feature Logit model and return OR, CI95 and p-value.

    The odds ratio is per one standard deviation increase in the feature,
    because the predictor is median-imputed and standardized before fitting.
    """
    if x.nunique(dropna=True) < 2:
        return np.nan, np.nan, np.nan, np.nan

    imputer = SimpleImputer(strategy="median")
    scaler = StandardScaler()
    x_imputed = imputer.fit_transform(x.to_frame())
    x_scaled = scaler.fit_transform(x_imputed).reshape(-1)

    design = sm.add_constant(x_scaled, has_constant="add")
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model = sm.Logit(y.to_numpy(dtype=float), design)
            result = model.fit(disp=False, maxiter=200)
    except Exception:
        # Fallback for rare separation/singularity cases.
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                model = sm.Logit(y.to_numpy(dtype=float), design)
                result = model.fit_regularized(alpha=1e-6, disp=False, maxiter=500)
            coef = float(result.params[1])
            return _safe_exp(coef), np.nan, np.nan, np.nan
        except Exception:
            return np.nan, np.nan, np.nan, np.nan

    coef = float(result.params[1])
    conf_int = result.conf_int(alpha=0.05)
    ci_lower = _safe_exp(float(conf_int[1, 0]))
    ci_upper = _safe_exp(float(conf_int[1, 1]))
    p_value = float(result.pvalues[1])
    odds_ratio = _safe_exp(coef)
    return odds_ratio, ci_lower, ci_upper, p_value


def analyze_feature(feature: str, x_train: pd.DataFrame, y_train: pd.Series) -> UnivariateResult:
    """Run CV AUROC and inferential Logit statistics for one feature."""
    x = x_train[feature]
    auroc = compute_cv_auroc(x, y_train)
    odds_ratio, ci_lower, ci_upper, p_value = fit_statsmodels_logit(x, y_train)
    return UnivariateResult(
        feature=feature,
        auroc_cv=auroc,
        odds_ratio=odds_ratio,
        ci_lower=ci_lower,
        ci_upper=ci_upper,
        p_value=p_value,
    )


def run_univariate_analysis(
    input_path: Path = INPUT_PATH,
    output_path: Path = OUTPUT_PATH,
    *,
    top_n: int = 10,
) -> pd.DataFrame:
    """Execute Phase 1 and write the ranked univariate summary table."""
    df = pd.read_csv(input_path)
    df["target"] = create_composite_target(df)
    print_target_prevalence(df["target"])

    predictor_columns = get_predictor_columns(df)
    print(f"Features predictivos evaluados: {len(predictor_columns)}")

    x_train, x_test, y_train, y_test = stratified_train_test_split(
        df[predictor_columns],
        df["target"],
    )
    print(
        "Split estratificado: "
        f"train={len(y_train)} ({y_train.mean() * 100:.2f}% positivos), "
        f"test={len(y_test)} ({y_test.mean() * 100:.2f}% positivos)"
    )

    results = [
        analyze_feature(feature, x_train, y_train)
        for feature in predictor_columns
    ]
    results_df = pd.DataFrame(
        [
            {
                "Feature": result.feature,
                "AUROC_CV": result.auroc_cv,
                "Odds_Ratio": result.odds_ratio,
                "CI_Lower": result.ci_lower,
                "CI_Upper": result.ci_upper,
                "P_Value": result.p_value,
            }
            for result in results
        ]
    )
    results_df = results_df.sort_values(
        by=["AUROC_CV", "P_Value"],
        ascending=[False, True],
        na_position="last",
    ).reset_index(drop=True)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    results_df.to_csv(output_path, index=False)

    print("\nTop 10 de caracteristicas mas informativas:")
    print(results_df.head(top_n).to_string(index=False))
    print(f"\nResultados guardados en: {output_path}")
    return results_df


def main() -> None:
    """CLI entrypoint."""
    run_univariate_analysis()


if __name__ == "__main__":
    main()
