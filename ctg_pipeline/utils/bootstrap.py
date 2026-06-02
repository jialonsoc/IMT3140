"""
utils/bootstrap.py
------------------
Stationary Bootstrap multivariado para pares de señales CTG (FHR, UC).

Decisiones implementadas (ver DECISIONS.md D-02, D-03, D-04)
-------------------------------------------------------------

D-02 — Stationary Bootstrap (Politis & Romano 1994)
    Se usa un solo proceso de Markov para generar los índices: con
    probabilidad p = 1/b se elige una posición aleatoria nueva (inicio
    de bloque nuevo), con probabilidad 1-p se avanza un paso.
    Esto produce bloques de longitud geométrica con media b, y la serie
    resultante es débilmente estacionaria.

D-03 — Tamaño de bloque óptimo b*
    b* = 2 × (primer lag donde ACF_FHR cae por debajo de ACF_ZERO_THR),
    acotado entre BLOCK_MIN_S y BLOCK_MAX_S segundos.
    Se calcula sobre FHR (más autocorrelacionada que UC) porque esa es
    la señal que fija la estructura temporal que queremos preservar.

D-02 — Sincronización FHR + UC
    El MISMO vector de índices se aplica a ambas señales. Esto garantiza
    que cada par (FHR_t, UC_t) sintético corresponde al mismo instante
    temporal del registro original, preservando la relación fisiológica
    contracción → desaceleración FHR.

Referencia:
    Politis, D. N., & Romano, J. P. (1994). The stationary bootstrap.
    Journal of the American Statistical Association, 89(428), 1303–1313.
    https://doi.org/10.1080/01621459.1994.10476870

    Politis, D. N., & White, H. (2004). Automatic block-length selection
    for the dependent bootstrap. Econometric Reviews, 23(1), 53–70.
    https://doi.org/10.1081/ETC-120028836
"""

import numpy as np
from config import FS, BLOCK_MIN_S, BLOCK_MAX_S, ACF_ZERO_THR


# ── Cálculo del tamaño de bloque óptimo ───────────────────────────────────────

def _acf(x: np.ndarray, max_lag: int) -> np.ndarray:
    """Autocorrelación normalizada de x hasta max_lag."""
    x = x - np.nanmean(x)
    n = len(x)
    acf = []
    for lag in range(max_lag + 1):
        if lag >= n:
            acf.append(0.0)
        else:
            num = np.nanmean(x[:n - lag] * x[lag:])
            acf.append(num)
    arr = np.array(acf, dtype=np.float64)
    if arr[0] != 0:
        arr /= arr[0]
    return arr


def optimal_block_length(fhr: np.ndarray) -> int:
    """
    Estima el tamaño óptimo de bloque b* en muestras a partir de la
    ventana de autocorrelación significativa de FHR.

    Heurística (Politis & White 2004):
        b* = 2 × primer lag donde ACF < ACF_ZERO_THR
    Acotado entre BLOCK_MIN_S y BLOCK_MAX_S segundos.

    Se usa FHR (no UC) porque es la señal con mayor autocorrelación
    y la que define la dinámica que el bootstrap debe preservar.

    Retorna: b* en número de muestras.
    """
    b_min = BLOCK_MIN_S * FS
    b_max = BLOCK_MAX_S * FS

    clean = fhr[~np.isnan(fhr)]
    if len(clean) < b_min * 2:
        return int(b_min)

    max_lag = int(b_max)
    acf = _acf(clean, max_lag)

    # Primer lag donde ACF cae por debajo del umbral
    first_zero = next(
        (lag for lag, a in enumerate(acf) if lag > 0 and abs(a) < ACF_ZERO_THR),
        int(b_max / 2)   # fallback si nunca cae
    )

    b_star = int(np.clip(2 * first_zero, b_min, b_max))
    return b_star


# ── Bootstrap multivariado sincronizado ───────────────────────────────────────

def _generate_indices(n: int, mean_block: int) -> np.ndarray:
    """
    Genera n índices según el proceso Markov del Stationary Bootstrap.
    p = 1/mean_block = probabilidad de saltar a una posición aleatoria nueva.
    """
    p = 1.0 / mean_block
    indices = []
    pos = np.random.randint(0, n)
    while len(indices) < n:
        indices.append(pos % n)
        if np.random.rand() < p:
            pos = np.random.randint(0, n)
        else:
            pos += 1
    return np.array(indices[:n], dtype=np.intp)


def stationary_bootstrap(fhr: np.ndarray,
                          uc: np.ndarray,
                          mean_block: int) -> tuple[np.ndarray, np.ndarray]:
    """
    Stationary Bootstrap multivariado sincronizado (Politis & Romano 1994).

    Aplica el MISMO vector de índices a FHR y UC, preservando la
    relación temporal entre ambas señales.

    Parámetros
    ----------
    fhr        : array float32 limpio (sin NaN, o con NaN residuales)
    uc         : array float32 de la misma longitud que fhr
    mean_block : longitud media de bloque en muestras (b*)

    Retorna
    -------
    fhr_syn, uc_syn : arrays float32 de la misma longitud que las entradas
    """
    if len(fhr) != len(uc):
        raise ValueError(
            f"FHR y UC deben tener la misma longitud ({len(fhr)} ≠ {len(uc)})"
        )

    idx = _generate_indices(len(fhr), mean_block)
    return fhr[idx].astype(np.float32), uc[idx].astype(np.float32)
