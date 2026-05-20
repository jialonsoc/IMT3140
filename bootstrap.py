"""
02_bootstrap_ctg.py
====================
Bootstrapping de series de tiempo CTG para aumento de muestra.
Implementa 4 estrategias + métricas estadísticas de fidelidad.

Estrategias disponibles:
  1. block_bootstrap     – Bootstrap por bloques (preserva autocorrelación)
  2. moving_block        – Moving Block Bootstrap (Künsch 1989)
  3. stationary_bootstrap– Bloques de longitud aleatoria (Politis & Romano 1994)
  4. jitter_warp         – Jitter + Time Warping (data augmentation clásico)

Métricas de fidelidad:
  - Distribución: KS-test, Wasserstein distance
  - Estructura temporal: autocorrelación (ACF), DTW distance
  - Espectral: comparación de densidad espectral de potencia (PSD)
  - Resumen: mean, std, skewness, kurtosis, percentiles

Uso:
    pip install numpy pandas scipy scikit-learn tslearn matplotlib seaborn tqdm
    python 02_bootstrap_ctg.py
"""

import os
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from pathlib import Path
from tqdm import tqdm
from scipy import stats, signal as sp_signal
from scipy.stats import ks_2samp, wasserstein_distance
from sklearn.preprocessing import StandardScaler
import json

warnings.filterwarnings("ignore")
np.random.seed(42)

# ── Configuración ──────────────────────────────────────────────────────────────
RAW_DIR        = "data/raw"
SYNTH_DIR      = "data/synthetic"
REPORT_DIR     = "reports"
METADATA_FILE  = "data/metadata.csv"
FS             = 4          # Hz
N_SYNTHETIC    = 1104       # meta objetivo (2× la muestra original)
BLOCK_SIZE     = 120        # muestras por bloque (30 s a 4 Hz)
STRATEGY       = "stationary_bootstrap"  # cambiar a: block_bootstrap | stationary_bootstrap | jitter_warp

os.makedirs(SYNTH_DIR, exist_ok=True)
os.makedirs(REPORT_DIR, exist_ok=True)

# ══════════════════════════════════════════════════════════════════════════════
# 1. CARGA DE DATOS
# ══════════════════════════════════════════════════════════════════════════════

def load_all_fhr(raw_dir: str) -> list[np.ndarray]:
    """Carga todas las series FHR disponibles en raw_dir."""
    files = sorted(Path(raw_dir).glob("*_fhr.npy"))
    series = []
    for f in files:
        arr = np.load(f).astype(np.float32)
        # Mantener sólo señales con suficientes datos válidos (>50% no-NaN)
        if np.mean(~np.isnan(arr)) > 0.5:
            series.append(arr)
    print(f"Cargadas {len(series)} series FHR desde '{raw_dir}'")
    return series

def interpolate_nans(x: np.ndarray) -> np.ndarray:
    """Interpolación lineal de NaNs internos; relleno de bordes con vecinos."""
    arr = x.copy()
    nans = np.isnan(arr)
    if not nans.any():
        return arr
    idx = np.arange(len(arr))
    valid = ~nans
    if valid.sum() < 2:
        return np.zeros_like(arr)
    arr[nans] = np.interp(idx[nans], idx[valid], arr[valid])
    return arr

# ══════════════════════════════════════════════════════════════════════════════
# 2. ESTRATEGIAS DE BOOTSTRAP
# ══════════════════════════════════════════════════════════════════════════════

def block_bootstrap(series: np.ndarray, target_len: int, block_size: int) -> np.ndarray:
    """Bootstrap por bloques no solapados."""
    n = len(series)
    n_blocks = (target_len // block_size) + 1
    starts = np.arange(0, n - block_size, block_size)
    chosen = np.random.choice(starts, size=n_blocks, replace=True)
    synthetic = np.concatenate([series[s:s + block_size] for s in chosen])
    return synthetic[:target_len]


def moving_block_bootstrap(series: np.ndarray, target_len: int, block_size: int) -> np.ndarray:
    """Moving Block Bootstrap (bloques solapados, Künsch 1989)."""
    n = len(series)
    n_blocks = (target_len // block_size) + 1
    max_start = n - block_size
    if max_start <= 0:
        return series[:target_len] if len(series) >= target_len else np.resize(series, target_len)
    starts = np.random.randint(0, max_start, size=n_blocks)
    synthetic = np.concatenate([series[s:s + block_size] for s in starts])
    return synthetic[:target_len]


def stationary_bootstrap(series: np.ndarray, target_len: int, mean_block: int = 60) -> np.ndarray:
    """Stationary Bootstrap con longitud de bloque geométrica (Politis & Romano 1994)."""
    n   = len(series)
    p   = 1.0 / mean_block   # probabilidad de nueva start
    out = []
    pos = np.random.randint(0, n)
    while len(out) < target_len:
        out.append(series[pos % n])
        if np.random.rand() < p:
            pos = np.random.randint(0, n)
        else:
            pos += 1
    return np.array(out[:target_len], dtype=np.float32)


def jitter_warp(series: np.ndarray, target_len: int,
                sigma_jitter: float = 1.5,
                warp_sigma: float = 0.05) -> np.ndarray:
    """
    Data augmentation: jitter gaussiano + time warping suave.
    Genera una nueva serie de la misma longitud con variaciones realistas.
    """
    n = len(series)
    # — jitter —
    jittered = series + np.random.normal(0, sigma_jitter, n).astype(np.float32)
    # — time warping suave mediante interpolación con puntos de anclaje deformados —
    n_anchors = max(10, n // 50)
    anchors   = np.linspace(0, n - 1, n_anchors)
    offsets   = np.random.normal(0, warp_sigma * n, n_anchors)
    warped_src = np.clip(anchors + offsets, 0, n - 1)
    warped_src = np.sort(warped_src)
    warped_tgt = np.linspace(0, n - 1, len(warped_src))
    # Interpolar desde la posición "warpada" a la posición original
    new_idx   = np.interp(np.arange(n), warped_tgt, warped_src)
    synthetic = np.interp(new_idx, np.arange(n), jittered).astype(np.float32)

    # Ajustar al target_len
    if len(synthetic) >= target_len:
        return synthetic[:target_len]
    else:
        return np.resize(synthetic, target_len)


STRATEGIES = {
    "block_bootstrap"      : block_bootstrap,
    "moving_block"         : moving_block_bootstrap,
    "stationary_bootstrap" : stationary_bootstrap,
    "jitter_warp"          : jitter_warp,
}

# ══════════════════════════════════════════════════════════════════════════════
# 3. GENERACIÓN DE SERIES SINTÉTICAS
# ══════════════════════════════════════════════════════════════════════════════
def clean_fhr(x: np.ndarray) -> np.ndarray:
    """Elimina valores fisiológicamente imposibles antes del bootstrap."""
    arr = x.copy()
    # Marcar como NaN valores fuera del rango fisiológico
    arr[(arr < 60) | (arr > 200)] = np.nan
    # Interpolar
    nans = np.isnan(arr)
    if nans.all():
        return np.full_like(arr, 140.0)
    idx = np.arange(len(arr))
    arr[nans] = np.interp(idx[nans], idx[~nans], arr[~nans])
    return arr

def generate_synthetic(series_list: list[np.ndarray],
                       n_synthetic: int,
                       strategy: str,
                       block_size: int) -> list[np.ndarray]:
    """Genera n_synthetic series usando la estrategia elegida."""
    func = STRATEGIES[strategy]
    synthetic = []
    for i in tqdm(range(n_synthetic), desc=f"Generando sintéticas [{strategy}]"):
        src = clean_fhr(interpolate_nans(series_list[i % len(series_list)]))
        target_len = len(src)
        syn = func(src, target_len, block_size) if strategy != "jitter_warp" \
              else func(src, target_len)
        synthetic.append(syn)
    return synthetic

# ══════════════════════════════════════════════════════════════════════════════
# 4. MÉTRICAS DE FIDELIDAD
# ══════════════════════════════════════════════════════════════════════════════

def summary_stats(arr: np.ndarray) -> dict:
    a = arr[~np.isnan(arr)]
    return {
        "mean"    : float(np.mean(a)),
        "std"     : float(np.std(a)),
        "skew"    : float(stats.skew(a)),
        "kurtosis": float(stats.kurtosis(a)),
        "p5"      : float(np.percentile(a, 5)),
        "p25"     : float(np.percentile(a, 25)),
        "p50"     : float(np.percentile(a, 50)),
        "p75"     : float(np.percentile(a, 75)),
        "p95"     : float(np.percentile(a, 95)),
    }


def acf(x: np.ndarray, max_lag: int = 100) -> np.ndarray:
    """Autocorrelación normalizada hasta max_lag."""
    x = x - np.nanmean(x)
    n = len(x)
    result = []
    for lag in range(max_lag + 1):
        if lag >= n:
            result.append(0.0)
        else:
            c = np.nanmean(x[:n - lag] * x[lag:])
            result.append(c)
    arr = np.array(result)
    if arr[0] != 0:
        arr /= arr[0]
    return arr


def psd(x: np.ndarray, fs: int = 4) -> tuple[np.ndarray, np.ndarray]:
    """Densidad espectral de potencia (Welch)."""
    x_clean = interpolate_nans(x)
    freqs, power = sp_signal.welch(x_clean, fs=fs, nperseg=min(256, len(x_clean) // 4))
    return freqs, power


def compute_fidelity_metrics(originals: list[np.ndarray],
                              synthetics: list[np.ndarray],
                              max_lag: int = 100) -> dict:
    """
    Calcula métricas de fidelidad entre distribución original y sintética.
    Retorna un dict con todas las métricas y sus interpretaciones.
    """
    print("\nCalculando métricas de fidelidad...")

    # — Aplanar para métricas de distribución —
    n_sample = min(len(originals), len(synthetics), 200)
    orig_flat = np.concatenate([interpolate_nans(s) for s in originals[:n_sample]])
    synt_flat = np.concatenate([s for s in synthetics[:n_sample]])

    # Remover outliers extremos (FHR razonable: 50–210 bpm)
    orig_flat = orig_flat[(orig_flat > 50) & (orig_flat < 210)]
    synt_flat = synt_flat[(synt_flat > 50) & (synt_flat < 210)]

    # ── 4.1 Distribución ──────────────────────────────────────────────────────
    ks_stat, ks_p   = ks_2samp(orig_flat, synt_flat)
    wass            = wasserstein_distance(orig_flat, synt_flat)
    orig_stats      = summary_stats(orig_flat)
    synt_stats      = summary_stats(synt_flat)

    # ── 4.2 Autocorrelación (media sobre series) ──────────────────────────────
    orig_acf = np.mean([acf(interpolate_nans(s), max_lag) for s in originals[:50]], axis=0)
    synt_acf = np.mean([acf(s, max_lag)                   for s in synthetics[:50]], axis=0)
    acf_mae  = float(np.mean(np.abs(orig_acf - synt_acf)))
    acf_corr = float(np.corrcoef(orig_acf, synt_acf)[0, 1])

    # ── 4.3 PSD media ─────────────────────────────────────────────────────────
    orig_psd_list = [psd(interpolate_nans(s))[1] for s in originals[:50]]
    synt_psd_list = [psd(s)[1]                   for s in synthetics[:50]]
    min_len   = min(min(len(p) for p in orig_psd_list), min(len(p) for p in synt_psd_list))
    orig_psd  = np.mean([p[:min_len] for p in orig_psd_list], axis=0)
    synt_psd  = np.mean([p[:min_len] for p in synt_psd_list], axis=0)
    psd_corr  = float(np.corrcoef(np.log1p(orig_psd), np.log1p(synt_psd))[0, 1])
    psd_mae   = float(np.mean(np.abs(np.log1p(orig_psd) - np.log1p(synt_psd))))

    metrics = {
        "distribution": {
            "ks_statistic"      : round(ks_stat, 4),
            "ks_pvalue"         : round(ks_p,    4),
            "wasserstein_dist"  : round(wass,    4),
            "original_stats"    : {k: round(v, 3) for k, v in orig_stats.items()},
            "synthetic_stats"   : {k: round(v, 3) for k, v in synt_stats.items()},
        },
        "temporal_structure": {
            "acf_mae"           : round(acf_mae,  5),
            "acf_pearson_r"     : round(acf_corr, 4),
            "orig_acf_lag1"     : round(float(orig_acf[1]), 4),
            "synt_acf_lag1"     : round(float(synt_acf[1]), 4),
        },
        "spectral": {
            "psd_log_corr"      : round(psd_corr, 4),
            "psd_log_mae"       : round(psd_mae,  4),
        },
        "interpretation": {},
        "_raw": {          # guardar para gráficos
            "orig_acf"  : orig_acf.tolist(),
            "synt_acf"  : synt_acf.tolist(),
            "orig_psd"  : orig_psd.tolist(),
            "synt_psd"  : synt_psd.tolist(),
            "orig_flat_sample": orig_flat[:5000].tolist(),
            "synt_flat_sample": synt_flat[:5000].tolist(),
        }
    }

    # ── 4.4 Interpretación automática ─────────────────────────────────────────
    interpret = []
    if ks_stat < 0.05:
        interpret.append("✅ KS-test: distribuciones muy similares (D<0.05)")
    elif ks_stat < 0.10:
        interpret.append("⚠️  KS-test: diferencia moderada (D<0.10) — aceptable")
    else:
        interpret.append("❌ KS-test: diferencia significativa (D≥0.10) — revisar estrategia")

    if wass < 1.0:
        interpret.append("✅ Wasserstein: distancia < 1 bpm — distribución preservada")
    elif wass < 3.0:
        interpret.append("⚠️  Wasserstein: distancia 1–3 bpm — sesgo leve")
    else:
        interpret.append("❌ Wasserstein: distancia > 3 bpm — distribución desplazada")

    if acf_corr > 0.98:
        interpret.append("✅ ACF Pearson r > 0.98 — estructura temporal excelente")
    elif acf_corr > 0.90:
        interpret.append("⚠️  ACF Pearson r > 0.90 — estructura temporal aceptable")
    else:
        interpret.append("❌ ACF Pearson r ≤ 0.90 — estructura temporal degradada")

    if psd_corr > 0.95:
        interpret.append("✅ PSD log-corr > 0.95 — contenido espectral preservado")
    elif psd_corr > 0.85:
        interpret.append("⚠️  PSD log-corr > 0.85 — espectro levemente alterado")
    else:
        interpret.append("❌ PSD log-corr ≤ 0.85 — distribución espectral alterada")

    metrics["interpretation"] = interpret
    return metrics


# ══════════════════════════════════════════════════════════════════════════════
# 5. VISUALIZACIONES
# ══════════════════════════════════════════════════════════════════════════════

def plot_fidelity_report(metrics: dict, strategy: str, out_dir: str):
    raw       = metrics["_raw"]
    orig_acf  = np.array(raw["orig_acf"])
    synt_acf  = np.array(raw["synt_acf"])
    orig_flat = np.array(raw["orig_flat_sample"])
    synt_flat = np.array(raw["synt_flat_sample"])
    orig_psd  = np.array(raw["orig_psd"])
    synt_psd  = np.array(raw["synt_psd"])

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle(f"Fidelity Report — CTG Bootstrap [{strategy}]",
                 fontsize=15, fontweight="bold", y=1.01)

    # ── Panel 1: Histograma FHR ───────────────────────────────────────────────
    ax = axes[0, 0]
    ax.hist(orig_flat, bins=80, alpha=0.6, color="#2196F3", label="Original", density=True)
    ax.hist(synt_flat, bins=80, alpha=0.6, color="#FF5722", label="Sintética", density=True)
    ax.set_xlabel("FHR (bpm)"); ax.set_ylabel("Densidad")
    ax.set_title("Distribución FHR")
    d  = metrics["distribution"]
    ax.legend(title=f"KS={d['ks_statistic']}, W={d['wasserstein_dist']:.2f}")

    # ── Panel 2: Q-Q plot ─────────────────────────────────────────────────────
    ax = axes[0, 1]
    n_qq = min(5000, len(orig_flat), len(synt_flat))
    o_q  = np.sort(np.random.choice(orig_flat, n_qq, replace=False))
    s_q  = np.sort(np.random.choice(synt_flat, n_qq, replace=False))
    ax.scatter(o_q, s_q, alpha=0.2, s=1, color="#7E57C2")
    mn, mx = min(o_q[0], s_q[0]), max(o_q[-1], s_q[-1])
    ax.plot([mn, mx], [mn, mx], "r--", lw=1.5, label="y=x (ideal)")
    ax.set_xlabel("Cuantiles originales"); ax.set_ylabel("Cuantiles sintéticos")
    ax.set_title("Q-Q Plot")
    ax.legend()

    # ── Panel 3: Box plot comparativo ────────────────────────────────────────
    ax = axes[0, 2]
    ax.boxplot([orig_flat, synt_flat], labels=["Original", "Sintética"],
               patch_artist=True,
               boxprops=dict(facecolor="#B3E5FC"),
               medianprops=dict(color="red", lw=2))
    ax.set_ylabel("FHR (bpm)"); ax.set_title("Box Plot FHR")

    # ── Panel 4: ACF ──────────────────────────────────────────────────────────
    ax = axes[1, 0]
    lags = np.arange(len(orig_acf))
    ax.plot(lags, orig_acf, color="#2196F3", lw=2, label="Original")
    ax.plot(lags, synt_acf, color="#FF5722", lw=2, ls="--", label="Sintética")
    ax.axhline(0, color="gray", lw=0.8, ls=":")
    ax.set_xlabel("Lag (muestras)"); ax.set_ylabel("ACF")
    t = metrics["temporal_structure"]
    ax.set_title(f"Autocorrelación  (r={t['acf_pearson_r']}, MAE={t['acf_mae']:.4f})")
    ax.legend()

    # ── Panel 5: PSD ──────────────────────────────────────────────────────────
    ax = axes[1, 1]
    freqs = np.linspace(0, FS / 2, len(orig_psd))
    ax.semilogy(freqs, orig_psd, color="#2196F3", lw=2, label="Original")
    ax.semilogy(freqs, synt_psd, color="#FF5722", lw=2, ls="--", label="Sintética")
    ax.set_xlabel("Frecuencia (Hz)"); ax.set_ylabel("PSD (log)")
    s = metrics["spectral"]
    ax.set_title(f"Densidad Espectral  (corr={s['psd_log_corr']})")
    ax.legend()

    # ── Panel 6: Tabla de métricas ────────────────────────────────────────────
    ax = axes[1, 2]
    ax.axis("off")
    rows = [
        ["Métrica", "Valor", "Umbral OK"],
        ["KS statistic",   f"{d['ks_statistic']:.4f}", "< 0.05"],
        ["KS p-value",     f"{d['ks_pvalue']:.4f}",    "> 0.05"],
        ["Wasserstein",    f"{d['wasserstein_dist']:.4f} bpm", "< 1.0"],
        ["ACF Pearson r",  f"{t['acf_pearson_r']:.4f}", "> 0.98"],
        ["ACF MAE",        f"{t['acf_mae']:.5f}",       "< 0.01"],
        ["PSD log-corr",   f"{s['psd_log_corr']:.4f}",  "> 0.95"],
        ["PSD log-MAE",    f"{s['psd_log_mae']:.4f}",   "< 0.5"],
    ]
    tbl = ax.table(cellText=rows[1:], colLabels=rows[0],
                   cellLoc="center", loc="center")
    tbl.auto_set_font_size(True)
    tbl.scale(1, 1.6)
    ax.set_title("Resumen de Métricas", pad=12)

    plt.tight_layout()
    out_path = os.path.join(out_dir, f"fidelity_report_{strategy}.png")
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Gráfico guardado → {out_path}")
    return out_path


def plot_sample_traces(originals, synthetics, strategy, out_dir, n=4):
    """Compara trazados individuales originales vs sintéticos."""
    fig, axes = plt.subplots(n, 1, figsize=(16, 3 * n))
    fig.suptitle(f"Trazados CTG — Original vs Sintético [{strategy}]",
                 fontsize=13, fontweight="bold")
    t_max = 60 * FS  # mostrar 60 s
    for i, ax in enumerate(axes):
        orig = interpolate_nans(originals[i])[:t_max]
        synt = synthetics[i][:t_max]
        t    = np.arange(len(orig)) / FS
        ax.plot(t, orig, color="#2196F3", lw=1.2, label="Original", alpha=0.9)
        ax.plot(t, synt, color="#FF5722", lw=1.2, label="Sintética", alpha=0.7, ls="--")
        ax.set_ylabel("FHR (bpm)"); ax.set_xlabel("Tiempo (s)")
        ax.legend(loc="upper right", fontsize=8)
        ax.set_title(f"Registro #{i + 1}")
    plt.tight_layout()
    out_path = os.path.join(out_dir, f"sample_traces_{strategy}.png")
    plt.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close()
    print(f"Trazados guardados → {out_path}")
    return out_path


# ══════════════════════════════════════════════════════════════════════════════
# 6. PIPELINE PRINCIPAL
# ══════════════════════════════════════════════════════════════════════════════

def run_all_strategies(series_list):
    """Corre todas las estrategias y produce un reporte comparativo."""
    results = {}
    for strat_name in STRATEGIES:
        print(f"\n{'='*60}")
        print(f"  ESTRATEGIA: {strat_name}")
        print(f"{'='*60}")
        synthetics = generate_synthetic(series_list, len(series_list), strat_name, BLOCK_SIZE)
        metrics    = compute_fidelity_metrics(series_list, synthetics)
        metrics.pop("_raw", None)   # excluir datos crudos del JSON final
        results[strat_name] = metrics
        print(f"\nInterpretación:")
        for msg in metrics["interpretation"]:
            print(f"  {msg}")
    return results


def main():
    # ── Cargar datos ──────────────────────────────────────────────────────────
    if not os.path.isdir(RAW_DIR) or not list(Path(RAW_DIR).glob("*_fhr.npy")):
        print("⚠️  No se encontraron datos en 'data/raw/'.")
        print("   Ejecuta primero: python 01_download_ctg.py")
        print("\n   [DEMO] Generando 20 series sintéticas para demostración...\n")
        # Datos de demo con parámetros realistas de FHR
        series_list = []
        for _ in range(20):
            n    = np.random.randint(5000, 20000)
            base = np.random.normal(140, 10, n)
            # Agregar variabilidad de baja frecuencia realista
            lf   = 5 * np.sin(2 * np.pi * np.arange(n) / (FS * 300))
            base = np.clip(base + lf, 80, 200).astype(np.float32)
            # Añadir algunos NaNs (~5%)
            nan_idx = np.random.choice(n, size=int(n * 0.05), replace=False)
            base[nan_idx] = np.nan
            series_list.append(base)
    else:
        series_list = load_all_fhr(RAW_DIR)

    print(f"\nTotal de series disponibles: {len(series_list)}")

    # ── Estrategia principal ──────────────────────────────────────────────────
    print(f"\nEstrategia seleccionada: {STRATEGY}")
    synthetics = generate_synthetic(series_list, N_SYNTHETIC, STRATEGY, BLOCK_SIZE)

    # Guardar sintéticas
    for i, s in enumerate(tqdm(synthetics, desc="Guardando sintéticas")):
        np.save(os.path.join(SYNTH_DIR, f"syn_{i:04d}_fhr.npy"), s)

    # ── Métricas ──────────────────────────────────────────────────────────────
    metrics = compute_fidelity_metrics(series_list, synthetics)

    print(f"\n{'─'*50}")
    print("INTERPRETACIÓN AUTOMÁTICA:")
    for msg in metrics["interpretation"]:
        print(f"  {msg}")
    print(f"{'─'*50}")

    # ── Gráficos ──────────────────────────────────────────────────────────────
    plot_fidelity_report(metrics, STRATEGY, REPORT_DIR)
    plot_sample_traces(series_list, synthetics, STRATEGY, REPORT_DIR, n=4)

    # ── Guardar reporte JSON ───────────────────────────────────────────────────
    report = {k: v for k, v in metrics.items() if k != "_raw"}
    report_path = os.path.join(REPORT_DIR, f"fidelity_{STRATEGY}.json")
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\nReporte JSON → {report_path}")

    # ── Comparar todas las estrategias ───────────────────────────────────────
    print("\n" + "="*60)
    resp = input("¿Comparar todas las estrategias? [s/N]: ").strip().lower()
    if resp == "s":
        all_results = run_all_strategies(series_list)
        with open(os.path.join(REPORT_DIR, "comparison_all_strategies.json"), "w") as f:
            json.dump(all_results, f, indent=2)

        # Tabla resumen
        rows = []
        for strat, res in all_results.items():
            rows.append({
                "strategy"    : strat,
                "KS_stat"     : res["distribution"]["ks_statistic"],
                "Wasserstein" : res["distribution"]["wasserstein_dist"],
                "ACF_r"       : res["temporal_structure"]["acf_pearson_r"],
                "PSD_corr"    : res["spectral"]["psd_log_corr"],
            })
        df = pd.DataFrame(rows)
        print("\n── Tabla comparativa ──")
        print(df.to_string(index=False))
        df.to_csv(os.path.join(REPORT_DIR, "strategy_comparison.csv"), index=False)
        print(f"\nCSV guardado → {REPORT_DIR}/strategy_comparison.csv")

    print("\n✓ Pipeline completo.")


if __name__ == "__main__":
    main()