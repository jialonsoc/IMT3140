"""
Pipeline 06 - Diccionario de predictores limpios.

Genera data/diccionario_predictores_limpios.csv con las variables permitidas
en el entrenamiento actual bajo la regla de censura absoluta del futuro.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


BEST_PARAMS_PATH = Path("data/fase3_exploration_best_params.json")
REAL_WINDOWS_PATH = Path("data/processed_features_windows_real.csv")
OUTPUT_PATH = Path("data/diccionario_predictores_limpios.csv")

OUTPUT_COLUMNS = ["Variable", "Tipo_Feature", "Descripción_Clínica_Matemática"]

BASAL_FEATURES = [
    "Age",
    "Gravidity",
    "Parity",
    "Diabetes",
    "Pyrexia",
    "Weight_g",
    "Gest_weeks",
    "Sex",
]

SIGNAL_FEATURES = [
    "fhr_invalid_pct",
    "uc_invalid_pct",
    "fhr_baseline_mean_bpm",
    "fhr_baseline_median_bpm",
    "fhr_baseline_std_bpm",
    "fhr_baseline_slope_bpm_min",
    "accelerations_count",
    "accelerations_mean_amp_bpm",
    "accelerations_mean_duration_s",
    "decelerations_count",
    "decelerations_mean_amp_bpm",
    "decelerations_mean_duration_s",
    "uc_contractions_count",
    "fhr_apen",
    "fhr_sampen",
    "ltv_mean_amp_bpm",
    "ltv_median_amp_bpm",
    "ltv_valid_windows",
    "decelerations_early_count",
    "decelerations_late_count",
    "decelerations_variable_count",
    "deceleration_uc_lag_mean_s",
    "deceleration_uc_lag_median_s",
    "dfa_alpha",
    "dfa_intercept",
    "dfa_alpha_short",
    "dfa_alpha_long",
]

SEQUENTIAL_FEATURES = [
    "delta_baseline_std_bpm",
    "delta_sampen",
    "fhr_std_falling_streak",
    "fhr_sampen_falling_streak",
    "slope_sampen_30min",
    "slope_baseline_mean_30min",
]

ALLOWED_ORDER = BASAL_FEATURES + SIGNAL_FEATURES + SEQUENTIAL_FEATURES

FEATURE_TYPES = {
    **{feature: "Basal materno-fetal" for feature in BASAL_FEATURES},
    "fhr_invalid_pct": "Calidad de señal",
    "uc_invalid_pct": "Calidad de señal",
    "fhr_baseline_mean_bpm": "Morfología temporal FIGO",
    "fhr_baseline_median_bpm": "Morfología temporal FIGO",
    "fhr_baseline_std_bpm": "Morfología temporal FIGO",
    "fhr_baseline_slope_bpm_min": "Morfología temporal FIGO",
    "accelerations_count": "Morfología temporal FIGO",
    "accelerations_mean_amp_bpm": "Morfología temporal FIGO",
    "accelerations_mean_duration_s": "Morfología temporal FIGO",
    "decelerations_count": "Morfología temporal FIGO",
    "decelerations_mean_amp_bpm": "Morfología temporal FIGO",
    "decelerations_mean_duration_s": "Morfología temporal FIGO",
    "uc_contractions_count": "Actividad uterina / acoplamiento FHR-UC",
    "fhr_apen": "Complejidad no lineal",
    "fhr_sampen": "Complejidad no lineal",
    "ltv_mean_amp_bpm": "Morfología temporal FIGO",
    "ltv_median_amp_bpm": "Morfología temporal FIGO",
    "ltv_valid_windows": "Calidad de señal",
    "decelerations_early_count": "Actividad uterina / acoplamiento FHR-UC",
    "decelerations_late_count": "Actividad uterina / acoplamiento FHR-UC",
    "decelerations_variable_count": "Actividad uterina / acoplamiento FHR-UC",
    "deceleration_uc_lag_mean_s": "Actividad uterina / acoplamiento FHR-UC",
    "deceleration_uc_lag_median_s": "Actividad uterina / acoplamiento FHR-UC",
    "dfa_alpha": "Complejidad no lineal",
    "dfa_intercept": "Complejidad no lineal",
    "dfa_alpha_short": "Complejidad no lineal",
    "dfa_alpha_long": "Complejidad no lineal",
    **{feature: "Gradiente longitudinal / Time Mix" for feature in SEQUENTIAL_FEATURES},
}

DESCRIPTIONS = {
    "Age": "Edad materna en años al ingreso; covariable basal previa al desenlace.",
    "Gravidity": "Número total de gestaciones maternas registradas antes del parto índice.",
    "Parity": "Número de partos previos; aproxima experiencia obstétrica basal sin usar evolución intraparto futura.",
    "Diabetes": "Indicador binario de diabetes materna; factor metabólico basal asociado a riesgo obstétrico.",
    "Pyrexia": "Indicador binario de fiebre materna registrada; marcador basal/intraparto temprano de posible infección o estrés.",
    "Weight_g": "Peso fetal/neonatal en gramos disponible como covariable biométrica basal del caso.",
    "Gest_weeks": "Edad gestacional en semanas; ajusta madurez fetal y tolerancia fisiológica.",
    "Sex": "Sexo fetal/neonatal codificado numéricamente; covariable basal de susceptibilidad biológica.",
    "fhr_invalid_pct": "Porcentaje de muestras FHR inválidas en la ventana, definidas como cero o NaN antes de interpolación; mide calidad de contacto.",
    "uc_invalid_pct": "Porcentaje de muestras UC inválidas en la ventana, definidas como cero o NaN antes de interpolación; mide calidad de señal uterina.",
    "fhr_baseline_mean_bpm": "Media robusta de la línea base FHR de la ventana, en latidos por minuto, tras limpieza de pérdidas de contacto.",
    "fhr_baseline_median_bpm": "Mediana robusta de la línea base FHR de la ventana, menos sensible a aceleraciones y desaceleraciones transitorias.",
    "fhr_baseline_std_bpm": "Desviación estándar de la línea base FHR en la ventana; resume dispersión de la frecuencia fetal basal.",
    "fhr_baseline_slope_bpm_min": "Pendiente lineal de la línea base FHR dentro de la ventana, en lpm/min; captura deriva temporal local.",
    "accelerations_count": "Conteo de aceleraciones FHR de al menos 15 lpm sostenidas al menos 15 segundos en la ventana.",
    "accelerations_mean_amp_bpm": "Amplitud media, en lpm, de las aceleraciones detectadas respecto de la línea base local.",
    "accelerations_mean_duration_s": "Duración media, en segundos, de las aceleraciones detectadas en la ventana actual.",
    "decelerations_count": "Conteo total de desaceleraciones FHR de al menos 15 lpm sostenidas al menos 15 segundos.",
    "decelerations_mean_amp_bpm": "Amplitud media, en lpm, de las caídas FHR clasificadas como desaceleraciones.",
    "decelerations_mean_duration_s": "Duración media, en segundos, de las desaceleraciones detectadas en la ventana.",
    "uc_contractions_count": "Conteo de contracciones uterinas detectadas en UC durante la ventana de 20 minutos.",
    "fhr_apen": "Entropía aproximada de la FHR interpolada; cuantifica irregularidad temporal y complejidad de corto plazo.",
    "fhr_sampen": "Entropía muestral de la FHR interpolada; estima complejidad sin autocoincidencias y es menos sesgada que ApEn.",
    "ltv_mean_amp_bpm": "Amplitud media de la variabilidad FHR de largo plazo en subventanas válidas, excluyendo eventos extremos.",
    "ltv_median_amp_bpm": "Mediana de la variabilidad FHR de largo plazo en subventanas válidas; descriptor robusto de oscilación basal.",
    "ltv_valid_windows": "Número de subventanas válidas usadas para estimar la variabilidad de largo plazo.",
    "decelerations_early_count": "Conteo de desaceleraciones tempranas, clasificadas por sincronía cercana con el pico de contracción UC.",
    "decelerations_late_count": "Conteo de desaceleraciones tardías, clasificadas por ocurrir después del pico UC; marcador clínico de hipoxia progresiva.",
    "decelerations_variable_count": "Conteo de desaceleraciones variables sin patrón temporal fijo respecto de UC.",
    "deceleration_uc_lag_mean_s": "Desfase medio, en segundos, entre el nadir de desaceleración FHR y el pico UC más cercano.",
    "deceleration_uc_lag_median_s": "Mediana del desfase, en segundos, entre desaceleraciones FHR y picos UC; versión robusta del acoplamiento temporal.",
    "dfa_alpha": "Exponente alfa global del análisis de fluctuación sin tendencia (DFA) sobre FHR; resume correlación fractal.",
    "dfa_intercept": "Intercepto del ajuste log-log del DFA; componente de escala de las fluctuaciones FHR.",
    "dfa_alpha_short": "Exponente DFA en escalas cortas; captura correlación local de la FHR en horizontes breves.",
    "dfa_alpha_long": "Exponente DFA en escalas largas; captura persistencia de la dinámica FHR en horizontes extendidos.",
    "delta_baseline_std_bpm": "Cambio de fhr_baseline_std_bpm respecto de la ventana de hace 10 minutos (lag de dos pasos de 5 min).",
    "delta_sampen": "Cambio de fhr_sampen respecto de la ventana de hace 10 minutos; cuantifica pérdida o ganancia reciente de complejidad.",
    "fhr_std_falling_streak": "Racha de ventanas consecutivas donde fhr_baseline_std_bpm disminuye respecto de la ventana previa; se reinicia al subir o mantenerse.",
    "fhr_sampen_falling_streak": "Racha de ventanas consecutivas donde fhr_sampen disminuye respecto de la ventana previa; indicador de simplificación sostenida.",
    "slope_sampen_30min": "Pendiente lineal de fhr_sampen calculada sobre las últimas seis ventanas continuas (30 minutos).",
    "slope_baseline_mean_30min": "Pendiente lineal de fhr_baseline_mean_bpm sobre las últimas seis ventanas continuas (30 minutos).",
}


def load_current_training_features() -> list[str]:
    """Return the feature set used by the current clean advanced training."""
    if BEST_PARAMS_PATH.exists():
        with BEST_PARAMS_PATH.open("r", encoding="utf-8") as file:
            payload = json.load(file)
        features = [str(feature) for feature in payload.get("features", [])]
        if features:
            return [feature for feature in ALLOWED_ORDER if feature in features]

    if not REAL_WINDOWS_PATH.exists():
        raise FileNotFoundError(
            f"No existe {BEST_PARAMS_PATH} ni {REAL_WINDOWS_PATH}; no se puede inferir el espacio de predictores."
        )
    columns = pd.read_csv(REAL_WINDOWS_PATH, nrows=0).columns
    return [feature for feature in ALLOWED_ORDER if feature in columns]


def build_dictionary() -> pd.DataFrame:
    """Build the predictor dictionary dataframe in the requested order."""
    rows = []
    for feature in load_current_training_features():
        rows.append(
            {
                "Variable": feature,
                "Tipo_Feature": FEATURE_TYPES[feature],
                "Descripción_Clínica_Matemática": DESCRIPTIONS[feature],
            }
        )
    dictionary = pd.DataFrame(rows, columns=OUTPUT_COLUMNS)
    if dictionary.empty:
        raise RuntimeError("El diccionario quedó vacío; revisa los archivos de entrada.")
    return dictionary


def main() -> None:
    """Generate and save the clean predictor dictionary."""
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    dictionary = build_dictionary()
    dictionary.to_csv(OUTPUT_PATH, index=False, encoding="utf-8-sig")
    print(f"Diccionario generado: {OUTPUT_PATH.resolve()}")
    print(f"Variables documentadas: {len(dictionary)}")


if __name__ == "__main__":
    main()
