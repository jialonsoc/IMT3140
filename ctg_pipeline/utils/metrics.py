"""
utils/metrics.py
----------------
Métricas de fidelidad estadística para comparar series originales
vs sintéticas. Usadas por 05_validation.py.

Métricas implementadas
----------------------
  - KS-test (Kolmogorov-Smirnov): ¿las distribuciones marginales son iguales?
  - Wasserstein-1: distancia entre distribuciones (interpretable en bpm / u.a.)
  - ACF media: ¿se preserva la estructura de autocorrelación?
  - PSD Welch: ¿se preserva el contenido espectral?
  - Correlación cruzada FHR↔UC: ¿se preserva la relación entre señales?

Todas las funciones reciben listas de arrays (una por registro) para
poder calcular promedios entre registros.
"""

import numpy as np
from scipy import signal as sp_signal
from scipy.stats import ks_2samp, wasserstein_distance
from config import FS


# ── Autocorrelación ────────────────────────────────────────────────────────────

def mean_acf(series_list: list[np.ndarray], max_lag: int = 200) -> np.ndarray:
    """
    Autocorrelación normalizada promediada sobre una lista de series.
    Ignora NaN en cada serie.
    """
    acfs = []
    for x in series_list:
        x = x[~np.isnan(x)]
        if len(x) < max_lag + 1:
            continue
        x = x - x.mean()
        n = len(x)
        row = []
        for lag in range(max_lag + 1):
            c = np.mean(x[:n - lag] * x[lag:])
            row.append(c)
        arr = np.array(row)
        if arr[0] != 0:
            arr /= arr[0]
        acfs.append(arr)
    return np.mean(acfs, axis=0) if acfs else np.zeros(max_lag + 1)


# ── Densidad espectral de potencia ─────────────────────────────────────────────

def mean_psd(series_list: list[np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    """
    PSD de Welch promediada sobre una lista de series.
    Retorna (freqs, power_mean).
    """
    psds = []
    freqs_ref = None
    for x in series_list:
        x = x[~np.isnan(x)]
        if len(x) < 256:
            continue
        nperseg = min(256, len(x) // 4)
        freqs, power = sp_signal.welch(x, fs=FS, nperseg=nperseg)
        if freqs_ref is None:
            freqs_ref = freqs
        if len(power) == len(freqs_ref):
            psds.append(power)
    if not psds:
        return np.array([0.0]), np.array([0.0])
    return freqs_ref, np.mean(psds, axis=0)


# ── Correlación cruzada FHR↔UC ────────────────────────────────────────────────

def mean_cross_corr(fhr_list: list[np.ndarray],
                    uc_list: list[np.ndarray],
                    max_lag: int = 200) -> np.ndarray:
    """
    Correlación cruzada normalizada FHR↔UC promediada sobre los pares.
    Un lag positivo significa que UC precede a FHR (fisiológicamente esperado:
    la contracción aparece antes de la desaceleración).
    """
    lags = np.arange(-max_lag, max_lag + 1)
    xcorrs = []
    for fhr, uc in zip(fhr_list, uc_list):
        # Solo muestras donde ambas señales son válidas
        valid = ~(np.isnan(fhr) | np.isnan(uc))
        f = fhr[valid] - fhr[valid].mean()
        u = uc[valid] - uc[valid].mean()
        if len(f) < max_lag * 2 + 1:
            continue
        xc = np.correlate(f, u, mode="full")
        center = len(xc) // 2
        xc = xc[center - max_lag: center + max_lag + 1]
        norm = (np.std(f) * np.std(u) * len(f))
        if norm > 0:
            xc = xc / norm
        xcorrs.append(xc)
    return np.mean(xcorrs, axis=0) if xcorrs else np.zeros(len(lags))


# ── Métricas de distribución ───────────────────────────────────────────────────

def distribution_metrics(orig_list: list[np.ndarray],
                          synt_list: list[np.ndarray]) -> dict:
    """
    KS-test y Wasserstein-1 entre la distribución empírica de los valores
    originales vs sintéticos (aplanando todas las series).
    """
    orig_flat = np.concatenate([x[~np.isnan(x)] for x in orig_list])
    synt_flat = np.concatenate([x[~np.isnan(x)] for x in synt_list])

    ks_stat, ks_p = ks_2samp(orig_flat, synt_flat)
    wass          = wasserstein_distance(orig_flat, synt_flat)

    return {
        "ks_statistic"  : round(float(ks_stat), 4),
        "ks_pvalue"     : round(float(ks_p),    4),
        "wasserstein"   : round(float(wass),     4),
    }


# ── Resumen compacto ───────────────────────────────────────────────────────────

def fidelity_summary(orig_fhr: list[np.ndarray],
                     synt_fhr: list[np.ndarray],
                     orig_uc:  list[np.ndarray],
                     synt_uc:  list[np.ndarray]) -> dict:
    """
    Calcula todas las métricas de fidelidad y retorna un dict con resultados
    e interpretación automática.
    """
    fhr_dist  = distribution_metrics(orig_fhr, synt_fhr)
    uc_dist   = distribution_metrics(orig_uc,  synt_uc)

    orig_acf  = mean_acf(orig_fhr)
    synt_acf  = mean_acf(synt_fhr)
    acf_r     = float(np.corrcoef(orig_acf, synt_acf)[0, 1])
    acf_mae   = float(np.mean(np.abs(orig_acf - synt_acf)))

    _, orig_psd = mean_psd(orig_fhr)
    _, synt_psd = mean_psd(synt_fhr)
    min_len     = min(len(orig_psd), len(synt_psd))
    psd_r       = float(np.corrcoef(
        np.log1p(orig_psd[:min_len]),
        np.log1p(synt_psd[:min_len])
    )[0, 1])

    orig_xc   = mean_cross_corr(orig_fhr, orig_uc)
    synt_xc   = mean_cross_corr(synt_fhr, synt_uc)
    xc_r      = float(np.corrcoef(orig_xc, synt_xc)[0, 1])

    summary = {
        "fhr_distribution" : fhr_dist,
        "uc_distribution"  : uc_dist,
        "fhr_acf_pearson_r": round(acf_r,   4),
        "fhr_acf_mae"      : round(acf_mae, 5),
        "fhr_psd_log_corr" : round(psd_r,   4),
        "fhr_uc_xcorr_r"   : round(xc_r,    4),
    }

    # Interpretación automática con umbrales de la literatura
    checks = []
    if fhr_dist["ks_statistic"] < 0.05:
        checks.append("FHR KS: distribuciones indistinguibles (D < 0.05)")
    else:
        checks.append(f"FHR KS: diferencia detectada (D = {fhr_dist['ks_statistic']})")

    if fhr_dist["wasserstein"] < 1.0:
        checks.append(f"FHR Wasserstein: {fhr_dist['wasserstein']} bpm (< 1 bpm, excelente)")
    else:
        checks.append(f"FHR Wasserstein: {fhr_dist['wasserstein']} bpm (revisar)")

    if acf_r > 0.98:
        checks.append(f"ACF Pearson r = {acf_r:.4f} (> 0.98, estructura temporal preservada)")
    elif acf_r > 0.90:
        checks.append(f"ACF Pearson r = {acf_r:.4f} (aceptable, > 0.90)")
    else:
        checks.append(f"ACF Pearson r = {acf_r:.4f} (degradado, revisar b*)")

    if xc_r > 0.90:
        checks.append(f"Correlación cruzada FHR↔UC r = {xc_r:.4f} (relación fisiológica preservada)")
    else:
        checks.append(f"Correlación cruzada FHR↔UC r = {xc_r:.4f} (relación FHR↔UC alterada)")

    summary["checks"] = checks
    return summary
