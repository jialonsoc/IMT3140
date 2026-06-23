"""
Phase 1 windowed univariate analysis with strict patient-level leakage control.

Inputs:
    data/processed_features_windows.csv
    data/clinical_metadata.csv

Output:
    data/results_fase1_univariado_windows.csv

Target Strategy B:
    target = 1 if pH < 7.10 OR Apgar5 < 7 OR NICU_days > 0.

All windows inherit their parent record target. Train/test splitting is done
before mapping windows, using unique patient/record IDs.

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
from sklearn.model_selection import StratifiedGroupKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


INPUT_PATH = Path("data/processed_features_windows.csv")
CLINICAL_METADATA_PATH = Path("data/clinical_metadata.csv")
OUTPUT_PATH = Path("data/results_fase1_univariado_windows.csv")
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
class UnivariateResult:
    """Container for one feature's grouped univariate analysis result."""

    feature: str
    auroc_cv: float
    odds_ratio: float
    ci_lower: float
    ci_upper: float
    p_value: float


def create_composite_target(df: pd.DataFrame) -> pd.Series:
    """Create Strategy B binary target."""
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
    """Load windows and map one causal target from parent records."""
    windows_df = pd.read_csv(windows_path)
    clinical_df = pd.read_csv(clinical_metadata_path)
    clinical_df["target"] = create_composite_target(clinical_df)

    record_target = clinical_df.set_index("record")["target"]
    windows_df["target"] = windows_df["record"].map(record_target).astype(int)
    return windows_df


def patient_level_split(
    windows_df: pd.DataFrame,
    clinical_metadata_path: Path = CLINICAL_METADATA_PATH,
) -> PatientSplit:
    """Split unique records 70/30 with stratification before selecting windows."""
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
        raise RuntimeError(f"Patient-level leakage detected for records: {sorted(overlap)[:5]}")

    return PatientSplit(
        train_df=train_df,
        test_df=test_df,
        train_records=train_record_set,
        test_records=test_record_set,
    )


def print_target_prevalence(df: pd.DataFrame, label: str) -> None:
    """Print patient/window target prevalence."""
    positives = int(df["target"].sum())
    total = int(df.shape[0])
    prevalence = positives / total if total else np.nan
    print(f"{label}: {positives}/{total} positivos ({prevalence * 100:.2f}%)")


def get_predictor_columns(df: pd.DataFrame) -> list[str]:
    """Select numeric candidate predictors after removing leakage columns."""
    excluded = set(
        EXCLUDED_OUTCOME_COLUMNS
        + EXCLUDED_IDENTIFIER_COLUMNS
        + EXCLUDED_WINDOW_TRACKING_COLUMNS
        + ["target"]
    )
    candidates = [column for column in df.columns if column not in excluded]
    numeric_columns = df[candidates].select_dtypes(include=[np.number]).columns
    return [column for column in numeric_columns if df[column].nunique(dropna=True) >= 2]


def _n_group_splits(y: pd.Series, groups: pd.Series, max_splits: int = 5) -> int:
    """Bound Group CV folds by the number of positive/negative patient groups."""
    group_target = pd.DataFrame({"target": y, "group": groups}).drop_duplicates("group")
    class_group_counts = group_target["target"].value_counts()
    if class_group_counts.shape[0] < 2:
        return 0
    return int(min(max_splits, class_group_counts.min(), group_target.shape[0]))


def compute_grouped_cv_auroc(
    x: pd.Series,
    y: pd.Series,
    groups: pd.Series,
    *,
    n_splits: int = 5,
) -> float:
    """Estimate univariate AUROC using patient-grouped stratified CV."""
    if x.nunique(dropna=True) < 2:
        return np.nan

    splits = _n_group_splits(y, groups, max_splits=n_splits)
    if splits < 2:
        return np.nan

    cv = StratifiedGroupKFold(n_splits=splits, shuffle=True, random_state=RANDOM_STATE)
    aucs: list[float] = []
    for train_idx, valid_idx in cv.split(x.to_frame(), y, groups=groups):
        x_train = x.iloc[train_idx].to_frame()
        y_train = y.iloc[train_idx]
        x_valid = x.iloc[valid_idx].to_frame()
        y_valid = y.iloc[valid_idx]
        if y_valid.nunique() < 2 or x_train.iloc[:, 0].nunique(dropna=True) < 2:
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
    """Exponentiate a log-odds value safely."""
    if value > np.log(np.finfo(float).max):
        return float("inf")
    if value < np.log(np.finfo(float).tiny):
        return 0.0
    return float(np.exp(value))


def fit_clustered_logit(
    x: pd.Series,
    y: pd.Series,
    groups: pd.Series,
) -> tuple[float, float, float, float]:
    """Fit univariate Logit with cluster-robust SE by patient record."""
    if x.nunique(dropna=True) < 2:
        return np.nan, np.nan, np.nan, np.nan

    imputer = SimpleImputer(strategy="median")
    scaler = StandardScaler()
    x_scaled = scaler.fit_transform(imputer.fit_transform(x.to_frame())).reshape(-1)
    design = sm.add_constant(x_scaled, has_constant="add")

    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model = sm.Logit(y.to_numpy(dtype=float), design)
            result = model.fit(
                disp=False,
                maxiter=250,
                cov_type="cluster",
                cov_kwds={"groups": groups.to_numpy()},
            )
    except Exception:
        return np.nan, np.nan, np.nan, np.nan

    coef = float(result.params[1])
    conf_int = result.conf_int(alpha=0.05)
    return (
        _safe_exp(coef),
        _safe_exp(float(conf_int[1, 0])),
        _safe_exp(float(conf_int[1, 1])),
        float(result.pvalues[1]),
    )


def analyze_feature(
    feature: str,
    train_df: pd.DataFrame,
) -> UnivariateResult:
    """Run grouped CV AUROC and clustered inferential Logit for one feature."""
    x = train_df[feature]
    y = train_df["target"]
    groups = train_df["record"]
    auroc = compute_grouped_cv_auroc(x, y, groups)
    odds_ratio, ci_lower, ci_upper, p_value = fit_clustered_logit(x, y, groups)
    return UnivariateResult(feature, auroc, odds_ratio, ci_lower, ci_upper, p_value)


def run_univariate_analysis_windows(
    input_path: Path = INPUT_PATH,
    output_path: Path = OUTPUT_PATH,
    *,
    top_n: int = 10,
) -> pd.DataFrame:
    """Execute grouped window-level univariate analysis and export results."""
    windows_df = load_windows_with_record_target(input_path, CLINICAL_METADATA_PATH)
    split = patient_level_split(windows_df, CLINICAL_METADATA_PATH)

    print_target_prevalence(windows_df.drop_duplicates("record"), "Pacientes totales")
    print_target_prevalence(windows_df, "Ventanas totales")
    print_target_prevalence(split.train_df, "Ventanas train")
    print_target_prevalence(split.test_df, "Ventanas test")

    predictor_columns = get_predictor_columns(split.train_df)
    print(f"Features predictivos evaluados: {len(predictor_columns)}")
    print(
        f"Records train/test: {len(split.train_records)}/{len(split.test_records)}; "
        f"ventanas train/test: {len(split.train_df)}/{len(split.test_df)}"
    )

    results = [analyze_feature(feature, split.train_df) for feature in predictor_columns]
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

    print("\nTop 10 de caracteristicas mas informativas (CV agrupada por record):")
    print(results_df.head(top_n).to_string(index=False))
    print(f"\nResultados guardados en: {output_path}")
    return results_df


def main() -> None:
    """CLI entrypoint."""
    run_univariate_analysis_windows()


if __name__ == "__main__":
    main()
