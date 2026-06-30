"""
Pipeline 05 - Visualizaciones finales para el informe de tesis.

Lee los resultados consolidados de data/final_comparative_report.csv y genera
figuras listas para insertar en el informe. La tercera figura reconstruye las
probabilidades del XGBoost ganador usando los hiperparametros guardados, para
evitar graficar datos inventados.
"""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt


PROJECT_ROOT = Path(__file__).resolve().parent
FASE3_DIR = PROJECT_ROOT / "fase3_exploration"
if str(FASE3_DIR) not in sys.path:
    sys.path.insert(0, str(FASE3_DIR))

from pipeline_04_fase3_advanced import (  # noqa: E402
    build_experiments,
    candidate_features,
    load_real_windows,
    load_synthetic_windows,
    patient_level_split,
    xgb_pipeline_factory,
)


FINAL_REPORT_PATH = Path("data/final_comparative_report.csv")
BEST_PARAMS_PATH = Path("data/fase3_exploration_best_params.json")
OUTPUT_DIR = Path("output_plots")

PLOT_01_PATH = OUTPUT_DIR / "plot_01_comparativa_auroc_auprc.png"
PLOT_02_PATH = OUTPUT_DIR / "plot_02_tradeoff_clinico.png"
PLOT_03_PATH = OUTPUT_DIR / "plot_03_perfil_alerta_temprana.png"

WINNING_MODEL = "XGBoost - Sequential real+synthetic"
TARGET_EARLY_WARNING_MIN = 50.0

MODEL_LABELS = {
    "Clean Forward AIC": "Forward AIC",
    "Clean Forward BIC": "Forward BIC",
    "XGBoost - Sequential real-only": "XGBoost Real",
    "CatBoost - Sequential real-only": "CatBoost Real",
    "XGBoost - Sequential real+synthetic": "XGBoost Real+Sint.",
    "CatBoost - Sequential real+synthetic": "CatBoost Real+Sint.",
}


@dataclass(frozen=True)
class WinnerCurve:
    """Predicted chronological risk curve for one positive test patient."""

    record: int
    threshold: float
    early_warning_min: float
    curve: pd.DataFrame


def configure_style() -> None:
    """Apply a formal and reproducible plotting style."""
    sns.set_theme(
        context="paper",
        style="whitegrid",
        font="DejaVu Sans",
        rc={
            "figure.dpi": 150,
            "savefig.dpi": 300,
            "axes.edgecolor": "#2f2f2f",
            "axes.labelcolor": "#222222",
            "axes.titleweight": "bold",
            "grid.color": "#e5e7eb",
            "grid.linewidth": 0.8,
            "legend.frameon": False,
        },
    )


def load_final_report() -> pd.DataFrame:
    """Load and validate the final comparative report."""
    if not FINAL_REPORT_PATH.exists():
        raise FileNotFoundError(f"No existe {FINAL_REPORT_PATH}. Ejecuta primero pipeline_04_fase3_advanced.py.")
    df = pd.read_csv(FINAL_REPORT_PATH)
    required = {
        "Model",
        "AUROC",
        "AUPRC",
        "Sensitivity",
        "Patient_NNA",
        "Threshold",
        "Early_Warning_Median_Min",
    }
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"Faltan columnas en {FINAL_REPORT_PATH}: {sorted(missing)}")
    df = df.copy()
    df["Modelo"] = df["Model"].map(MODEL_LABELS).fillna(df["Model"])
    return df


def plot_metric_comparison(report: pd.DataFrame) -> Path:
    """Grouped bar plot comparing AUROC and AUPRC across final models."""
    long_df = report.melt(
        id_vars=["Model", "Modelo"],
        value_vars=["AUROC", "AUPRC"],
        var_name="Metrica",
        value_name="Valor",
    )
    long_df["Metrica"] = long_df["Metrica"].map({"AUROC": "AUROC", "AUPRC": "AUPRC"})
    order = report["Modelo"].tolist()

    fig, ax = plt.subplots(figsize=(12.5, 6.8))
    palette = {"AUROC": "#1f77b4", "AUPRC": "#ff9f1c"}
    sns.barplot(
        data=long_df,
        x="Modelo",
        y="Valor",
        hue="Metrica",
        order=order,
        palette=palette,
        ax=ax,
        edgecolor="#1f2937",
        linewidth=0.35,
    )
    ax.set_title("Comparacion de capacidad discriminante por modelo")
    ax.set_xlabel("")
    ax.set_ylabel("Valor de metrica")
    ax.set_ylim(0.0, max(0.75, float(long_df["Valor"].max()) + 0.08))
    ax.tick_params(axis="x", rotation=25)
    ax.legend(title="")

    for container in ax.containers:
        ax.bar_label(container, fmt="%.3f", padding=2, fontsize=8)

    winner_label = MODEL_LABELS.get(WINNING_MODEL, WINNING_MODEL)
    if winner_label in order:
        winner_idx = order.index(winner_label)
        ax.axvspan(winner_idx - 0.42, winner_idx + 0.42, color="#2ca58d", alpha=0.08, zorder=0)
        ax.text(
            winner_idx,
            ax.get_ylim()[1] * 0.96,
            "XGBoost mixto",
            ha="center",
            va="top",
            fontsize=9,
            color="#146356",
            fontweight="bold",
        )

    sns.despine(ax=ax)
    fig.tight_layout()
    fig.savefig(PLOT_01_PATH, bbox_inches="tight")
    plt.close(fig)
    return PLOT_01_PATH


def plot_clinical_tradeoff(report: pd.DataFrame) -> Path:
    """Scatter plot of sensitivity versus patient-level NNA."""
    fig, ax = plt.subplots(figsize=(10.8, 6.5))
    ordered = report.sort_values("Patient_NNA").copy()
    ordered["Modelo"] = ordered["Model"].map(MODEL_LABELS).fillna(ordered["Model"])

    x_min = max(0.0, float(ordered["Patient_NNA"].min()) - 0.45)
    ax.axvspan(x_min, 20.0, color="#dcfce7", alpha=0.28, zorder=0)
    ax.text(
        x_min + 0.15,
        0.625,
        "Zona segura de carga de alarmas",
        color="#166534",
        fontsize=9,
        va="bottom",
        fontweight="bold",
    )

    palette = {
        "Forward AIC": "#2ca25f",
        "Forward BIC": "#d1495b",
        "XGBoost Real": "#3366aa",
        "CatBoost Real": "#e07a3f",
        "XGBoost Real+Sint.": "#756bb1",
        "CatBoost Real+Sint.": "#8d6e63",
    }
    offsets = {
        "Forward AIC": (8, 4),
        "Forward BIC": (10, -18),
        "XGBoost Real": (8, 14),
        "CatBoost Real": (8, -18),
        "XGBoost Real+Sint.": (8, 10),
        "CatBoost Real+Sint.": (8, -2),
    }
    auprc_min = float(ordered["AUPRC"].min())
    auprc_range = max(1e-12, float(ordered["AUPRC"].max()) - auprc_min)

    for _, row in ordered.iterrows():
        label = str(row["Modelo"])
        size = 120.0 + 240.0 * ((float(row["AUPRC"]) - auprc_min) / auprc_range)
        ax.scatter(
            row["Patient_NNA"],
            row["Sensitivity"],
            s=size,
            color=palette.get(label, "#4b5563"),
            edgecolor="#111827",
            linewidth=0.7,
            zorder=3,
        )
        ax.annotate(
            label,
            (row["Patient_NNA"], row["Sensitivity"]),
            xytext=offsets.get(label, (6, 4)),
            textcoords="offset points",
            fontsize=9,
            color="#1f2937",
            arrowprops={"arrowstyle": "-", "color": "#9ca3af", "lw": 0.8},
        )

    ax.axvline(20, color="#b91c1c", linestyle="--", linewidth=1.5)
    ax.text(
        20.15,
        0.70,
        "NNA = 20\nlimite de fatiga",
        color="#991b1b",
        fontsize=9,
        va="bottom",
    )

    ax.set_xlim(x_min, 20.8)
    ax.set_ylim(0.62, 0.95)
    ax.set_title("Trade-off clinico: sensibilidad versus carga de alarmas")
    ax.set_xlabel("Numero necesario de alertas por verdadero positivo paciente")
    ax.set_ylabel("Sensibilidad por ventana")
    sns.despine(ax=ax)
    fig.tight_layout()
    fig.savefig(PLOT_02_PATH, bbox_inches="tight")
    plt.close(fig)
    return PLOT_02_PATH


def _load_winning_model_config() -> tuple[list[str], dict[str, Any], float]:
    """Read feature list, best parameters and threshold for the winning XGBoost."""
    if not BEST_PARAMS_PATH.exists():
        raise FileNotFoundError(f"No existe {BEST_PARAMS_PATH}. Ejecuta primero el pipeline 04.")
    with BEST_PARAMS_PATH.open("r", encoding="utf-8") as file:
        payload = json.load(file)
    features = [str(feature) for feature in payload["features"]]
    for model in payload["models"]:
        if model.get("model") == WINNING_MODEL:
            return features, dict(model["best_params"]), float(model["threshold"])
    raise ValueError(f"No se encontro el modelo ganador en {BEST_PARAMS_PATH}: {WINNING_MODEL}")


def _persistent_alert_start(group: pd.DataFrame) -> float | None:
    """Return time-to-birth at the first point of the final persistent alert suffix."""
    alerts = group["alert"].to_numpy(dtype=bool)
    if alerts.size == 0 or not alerts[-1]:
        return None
    start = alerts.size - 1
    while start > 0 and alerts[start - 1]:
        start -= 1
    return float(group.iloc[start]["time_to_birth_min"])


def reconstruct_winning_xgb_curve() -> WinnerCurve:
    """Fit the winning XGBoost and extract a real positive test-patient curve."""
    features, best_params, threshold = _load_winning_model_config()
    real_df = load_real_windows()
    synthetic_df = load_synthetic_windows()
    split = patient_level_split(real_df)

    selected_features = [feature for feature in candidate_features(split.train_df) if feature in features]
    if selected_features != features:
        missing = sorted(set(features).difference(selected_features))
        raise ValueError(f"No se pudieron reconstruir todos los predictores del modelo: {missing}")

    experiments = build_experiments(split, synthetic_df, selected_features)
    mixed_train = next(exp.train_df for exp in experiments if exp.name == "Sequential real+synthetic")
    y_train = mixed_train["target"].astype(int)
    negatives = int((y_train == 0).sum())
    positives = int((y_train == 1).sum())
    pos_weight = float(negatives / positives) if positives else 1.0

    estimator = xgb_pipeline_factory(pos_weight)
    estimator.set_params(**best_params)
    estimator.fit(mixed_train[selected_features], y_train)

    test_ordered = split.test_df.sort_values(["record", "window_start_min"]).copy()
    test_ordered["probability"] = estimator.predict_proba(test_ordered[selected_features])[:, 1]
    test_ordered["alert"] = test_ordered["probability"] >= threshold

    candidates: list[tuple[float, int, pd.DataFrame]] = []
    for record, group in test_ordered.groupby("record", sort=False):
        if int(group["target"].max()) != 1:
            continue
        group = group.sort_values("window_start_min").copy()
        early_warning = _persistent_alert_start(group)
        if early_warning is not None:
            candidates.append((abs(early_warning - TARGET_EARLY_WARNING_MIN), int(record), group))

    if not candidates:
        positives_df = test_ordered[test_ordered["target"] == 1].copy()
        if positives_df.empty:
            raise RuntimeError("No hay pacientes positivos en el set de test.")
        record = int(positives_df.groupby("record")["probability"].max().idxmax())
        group = positives_df[positives_df["record"] == record].sort_values("window_start_min").copy()
        early_warning = float(group.loc[group["probability"].idxmax(), "time_to_birth_min"])
        return WinnerCurve(record=record, threshold=threshold, early_warning_min=early_warning, curve=group)

    _, record, curve = sorted(candidates, key=lambda item: (item[0], item[1]))[0]
    early_warning = float(_persistent_alert_start(curve))
    return WinnerCurve(record=record, threshold=threshold, early_warning_min=early_warning, curve=curve)


def plot_early_warning_profile(curve: WinnerCurve) -> Path:
    """Plot a chronological live-monitoring risk profile for one test patient."""
    data = curve.curve.sort_values("time_to_birth_min", ascending=False).copy()
    x = data["time_to_birth_min"].to_numpy(dtype=float)
    y = data["probability"].to_numpy(dtype=float)

    fig, (ax_top, ax_prob) = plt.subplots(
        2,
        1,
        figsize=(12.5, 7.2),
        sharex=True,
        gridspec_kw={"height_ratios": [0.75, 2.4], "hspace": 0.12},
        constrained_layout=True,
    )

    ax_top.plot([x.max(), 0.0], [1.0, 1.0], color="#64748b", linewidth=5, solid_capstyle="round")
    ax_top.scatter([x.max()], [1], s=95, color="#0f766e", zorder=3)
    ax_top.scatter([0], [1], s=105, color="#b91c1c", zorder=3)
    ax_top.axvline(curve.early_warning_min, color="#ea580c", linestyle="--", linewidth=1.5)
    ax_top.text(x.max(), 1.10, "Inicio monitorizacion", ha="left", va="bottom", fontsize=9)
    ax_top.text(0, 1.10, "Parto", ha="right", va="bottom", fontsize=9, color="#7f1d1d")
    ax_top.text(
        curve.early_warning_min,
        0.84,
        f"Alerta persistente\n{curve.early_warning_min:.0f} min",
        ha="center",
        va="top",
        fontsize=9,
        color="#9a3412",
    )
    ax_top.set_yticks([])
    ax_top.set_ylabel("Linea temporal")
    ax_top.set_title(f"Perfil de alerta temprana en simulacion en vivo - registro {curve.record}")
    sns.despine(ax=ax_top, left=True, bottom=True)

    ax_prob.plot(x, y, color="#155e75", linewidth=2.3, marker="o", markersize=3.8)
    ax_prob.fill_between(x, y, curve.threshold, where=y >= curve.threshold, color="#f97316", alpha=0.22)
    ax_prob.axhline(curve.threshold, color="#b91c1c", linestyle="--", linewidth=1.4)
    ax_prob.axvline(curve.early_warning_min, color="#ea580c", linestyle="--", linewidth=1.5)
    ax_prob.annotate(
        f"Early Warning Time: {curve.early_warning_min:.0f} min",
        xy=(curve.early_warning_min, curve.threshold),
        xytext=(
            max(float(x.min()) + 2.0, curve.early_warning_min - 18.0),
            min(0.98, max(float(y.max()), curve.threshold) + 0.08),
        ),
        arrowprops={"arrowstyle": "->", "color": "#9a3412", "lw": 1.5},
        color="#9a3412",
        fontsize=10,
        fontweight="bold",
    )
    ax_prob.text(
        x.min() + 1.5,
        curve.threshold + 0.01,
        f"Umbral XGBoost = {curve.threshold:.4f}",
        color="#7f1d1d",
        fontsize=9,
    )
    ax_prob.set_xlabel("Minutos antes del parto")
    ax_prob.set_ylabel("Probabilidad estimada de asfixia")
    ax_prob.set_ylim(0.0, min(1.0, max(0.18, float(y.max()) + 0.12)))
    ax_prob.invert_xaxis()
    sns.despine(ax=ax_prob)

    fig.savefig(PLOT_03_PATH, bbox_inches="tight")
    plt.close(fig)
    return PLOT_03_PATH


def main() -> None:
    """Generate all final visualizations."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    configure_style()
    report = load_final_report()

    created = [
        plot_metric_comparison(report),
        plot_clinical_tradeoff(report),
        plot_early_warning_profile(reconstruct_winning_xgb_curve()),
    ]

    print("Graficos generados:")
    for path in created:
        print(f"- {path.resolve()}")


if __name__ == "__main__":
    main()
