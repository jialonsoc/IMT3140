"""
utils/signal.py
---------------
Limpieza y estadísticas de señales CTG (FHR y UC).

Decisiones de NaN e interpolación
----------------------------------
Se distinguen dos tipos de ausencia de dato:

  1. Valores fisiológicamente imposibles (FHR < 60 o > 200 bpm):
     → Se marcan como NaN. NO se clipean al límite porque eso introduciría
       valores artificiales en los bordes del rango que distorsionan la
       distribución y la autocorrelación.

  2. Gaps de señal (NaN consecutivos):
     → Gaps CORTOS (≤ MAX_GAP_S segundos, por defecto 15 s):
         Se interpolan linealmente. Justificación: corresponden a artefactos
         de movimiento o recolocación del electrodo; la señal subyacente
         existe y es continua (Ayres-de-Campos et al., 2015 — guías FIGO).
     → Gaps LARGOS (> MAX_GAP_S):
         Se dejan como NaN. Representan pérdida real de señal. Interpolarlo
         sería inventar fisiología. El script de bootstrap (03) excluirá o
         tratará estos registros según NAN_PCT_MAX.

Referencia:
  Ayres-de-Campos, D., Spong, C. Y., & Chandraharan, E. (2015).
  FIGO consensus guidelines on intrapartum fetal monitoring: Cardiotocography.
  International Journal of Gynecology & Obstetrics, 131(1), 13–24.
  https://doi.org/10.1016/j.ijgo.2015.06.020
"""

import numpy as np
from config import FS, FHR_MIN, FHR_MAX

# Umbral para interpolación: gaps más largos que esto se dejan como NaN
MAX_GAP_S = 15   # segundos
MAX_GAP_N = int(MAX_GAP_S * FS)  # en muestras (= 60 a 4 Hz)


# ── Utilidades internas ────────────────────────────────────────────────────────

def _gap_lengths(mask_nan: np.ndarray) -> list[tuple[int, int]]:
    """
    Devuelve lista de (inicio, longitud) de cada run de NaNs contiguos.
    mask_nan: array booleano True donde hay NaN.
    """
    gaps = []
    in_gap = False
    start = 0
    for i, is_nan in enumerate(mask_nan):
        if is_nan and not in_gap:
            in_gap = True
            start = i
        elif not is_nan and in_gap:
            in_gap = False
            gaps.append((start, i - start))
    if in_gap:
        gaps.append((start, len(mask_nan) - start))
    return gaps


def _interpolate_short_gaps(arr: np.ndarray, max_gap: int) -> np.ndarray:
    """
    Interpola linealmente solo los gaps de NaN con longitud ≤ max_gap.
    Los gaps más largos permanecen como NaN.
    """
    out = arr.copy()
    gaps = _gap_lengths(np.isnan(out))
    for start, length in gaps:
        if length > max_gap:
            continue  # gap largo → no tocar
        # Necesita un valor válido a cada lado para interpolar
        left  = start - 1
        right = start + length
        if left < 0 or right >= len(out):
            continue  # en el borde de la serie → no interpolar
        if np.isnan(out[left]) or np.isnan(out[right]):
            continue  # vecinos también NaN → no interpolar
        out[start:right] = np.linspace(out[left], out[right], length + 2)[1:-1]
    return out


# ── API pública ────────────────────────────────────────────────────────────────

def clean_fhr(x: np.ndarray) -> np.ndarray:
    """
    Limpia la señal FHR:
      1. Valores fuera de [FHR_MIN, FHR_MAX] → NaN  (no clip)
      2. Gaps ≤ MAX_GAP_S → interpolación lineal
      3. Gaps > MAX_GAP_S → permanecen NaN

    Retorna array float32 de la misma longitud que x.
    """
    arr = x.copy().astype(np.float32)
    arr[(arr < FHR_MIN) | (arr > FHR_MAX)] = np.nan
    arr = _interpolate_short_gaps(arr, MAX_GAP_N)
    return arr


def clean_uc(x: np.ndarray) -> np.ndarray:
    """
    Limpia la señal UC (unidades arbitrarias de presión).
    No tiene límites fisiológicos tan estrictos como FHR, pero se eliminan
    valores claramente erróneos (negativos o > 200 unidades).
    Misma estrategia de interpolación de gaps cortos.
    """
    arr = x.copy().astype(np.float32)
    arr[arr < 0] = np.nan          # presión negativa no tiene sentido físico
    arr[arr > 200] = np.nan        # umbral conservador para outliers extremos
    arr = _interpolate_short_gaps(arr, MAX_GAP_N)
    return arr


def nan_summary(arr: np.ndarray) -> dict:
    """
    Describe la estructura de NaNs de una señal.
    Útil para decidir si un registro es elegible para bootstrap.
    """
    mask = np.isnan(arr)
    gaps = _gap_lengths(mask)
    long_gaps = [g for g in gaps if g[1] > MAX_GAP_N]
    return {
        "nan_pct"        : float(mask.mean() * 100),
        "n_gaps"         : len(gaps),
        "n_long_gaps"    : len(long_gaps),
        "max_gap_samples": max((g[1] for g in gaps), default=0),
        "max_gap_s"      : max((g[1] for g in gaps), default=0) / FS,
    }


def signal_stats(arr: np.ndarray, label: str = "") -> dict:
    """
    Estadísticas descriptivas de una señal limpia.
    Ignora los NaN restantes (gaps largos no interpolados).
    label: prefijo para los nombres de columna (ej. "fhr" o "uc").
    """
    clean = arr[~np.isnan(arr)]
    prefix = f"{label}_" if label else ""
    if len(clean) == 0:
        return {f"{prefix}mean": np.nan, f"{prefix}std": np.nan,
                f"{prefix}min": np.nan,  f"{prefix}max": np.nan,
                f"{prefix}nan_pct": 100.0, "n_samples": len(arr),
                "duration_min": round(len(arr) / FS / 60, 2), "fs_hz": FS}
    return {
        f"{prefix}mean"   : round(float(np.mean(clean)),  3),
        f"{prefix}std"    : round(float(np.std(clean)),   3),
        f"{prefix}min"    : round(float(np.min(clean)),   3),
        f"{prefix}max"    : round(float(np.max(clean)),   3),
        f"{prefix}nan_pct": round(float(np.isnan(arr).mean() * 100), 2),
        "n_samples"       : len(arr),
        "duration_min"    : round(len(arr) / FS / 60, 2),
        "fs_hz"           : FS,
    }
