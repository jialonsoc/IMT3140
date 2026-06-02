"""
05_validation.py
----------------
Validación estadística de fidelidad del bootstrap.
Compara la distribución y estructura temporal de las series originales
vs las sintéticas en FHR, UC y la relación cruzada FHR↔UC.

Métricas calculadas
-------------------
  1. Distribución marginal FHR  : KS-test, Wasserstein-1
  2. Distribución marginal UC   : KS-test, Wasserstein-1
  3. Autocorrelación FHR        : Pearson r entre curvas ACF media
  4. Densidad espectral FHR     : Pearson r entre curvas PSD log-media
  5. Correlación cruzada FHR↔UC : Pearson r entre funciones de xcorr media

Interpretación de umbrales
---------------------------
  KS statistic < 0.05   → distribuciones indistinguibles
  Wasserstein  < 1.0    → diferencia < 1 bpm / u.a. (excelente)
  ACF Pearson r > 0.98  → estructura temporal preservada
  PSD log-corr > 0.95   → contenido espectral preservado
  XCorr r > 0.90        → relación fisiológica FHR↔UC preservada

Salida
------
  reports/05_fidelity.json  — métricas numéricas
  reports/05_fidelity.png   — figura de 6 paneles

Ejecutar después de 03_bootstrap.py:
    cd ctg_pipeline
    python 05_validation.py
"""

import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from config import FULL_CSV, SYNTH_DIR, RAW_DIR, REPORT_DIR, FS
from utils.signal import clean_fhr, clean_uc
from utils.metrics import (
    distribution_metrics, mean_acf, mean_psd,
    mean_cross_corr, fidelity_summary,
)

REPORT_DIR.mkdir(parents=True, exist_ok=True)

MAX_RECORDS = 100   # muestrea hasta 100 pares para que sea manejable en memoria


# ── Carga de señales ──────────────────────────────────────────────────────────

def load_originals(df_orig: pd.DataFrame, n: int) -> tuple[list, list]:
    """Carga hasta n pares FHR+UC de los registros originales."""
    fhr_list, uc_list = [], []
    for rec_id in df_orig["record"].astype(str).head(n):
        fhr_path = RAW_DIR  / f"{rec_id}_fhr.npy"
        uc_path  = RAW_DIR  / f"{rec_id}_uc.npy"
        if not fhr_path.exists() or not uc_path.exists():
            continue
        fhr_list.append(clean_fhr(np.load(fhr_path)))
        uc_list.append(clean_uc(np.load(uc_path)))
    return fhr_list, uc_list


def load_synthetics(df_synt: pd.DataFrame, n: int) -> tuple[list, list]:
    """Carga hasta n pares FHR+UC de las series sintéticas."""
    fhr_list, uc_list = [], []
    for syn_id in df_synt["record"].astype(str).head(n):
        fhr_path = SYNTH_DIR / f"{syn_id}_fhr.npy"
        uc_path  = SYNTH_DIR / f"{syn_id}_uc.npy"
        if not fhr_path.exists() or not uc_path.exists():
            continue
        fhr_list.append(np.load(fhr_path))
        uc_list.append(np.load(uc_path))
    return fhr_list, uc_list


# ── Figura de fidelidad ───────────────────────────────────────────────────────

def plot_fidelity(orig_fhr: list, synt_fhr: list,
                  orig_uc:  list, synt_uc:  list,
                  summary:  dict):
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle("Validación de Fidelidad — Bootstrap CTG (FHR + UC)",
                 fontsize=14, fontweight="bold")

    # ── A: Histograma FHR ─────────────────────────────────────────────────────
    ax = axes[0, 0]
    orig_flat = np.concatenate([x[~np.isnan(x)] for x in orig_fhr])
    synt_flat = np.concatenate([x[~np.isnan(x)] for x in synt_fhr])
    bins = np.linspace(60, 200, 60)
    ax.hist(orig_flat, bins=bins, alpha=0.6, density=True,
            color="#2196F3", label="Original")
    ax.hist(synt_flat, bins=bins, alpha=0.6, density=True,
            color="#FF5722", label="Sintética")
    d = summary["fhr_distribution"]
    ax.set_title(f"FHR — KS={d['ks_statistic']:.4f}  W={d['wasserstein']:.3f} bpm")
    ax.set_xlabel("FHR (bpm)"); ax.set_ylabel("Densidad"); ax.legend(fontsize=8)

    # ── B: Histograma UC ──────────────────────────────────────────────────────
    ax = axes[0, 1]
    orig_uc_flat = np.concatenate([x[~np.isnan(x)] for x in orig_uc])
    synt_uc_flat = np.concatenate([x[~np.isnan(x)] for x in synt_uc])
    bins_uc = np.linspace(0, 120, 50)
    ax.hist(orig_uc_flat, bins=bins_uc, alpha=0.6, density=True,
            color="#2196F3", label="Original")
    ax.hist(synt_uc_flat, bins=bins_uc, alpha=0.6, density=True,
            color="#FF5722", label="Sintética")
    d_uc = summary["uc_distribution"]
    ax.set_title(f"UC — KS={d_uc['ks_statistic']:.4f}  W={d_uc['wasserstein']:.3f}")
    ax.set_xlabel("UC (u.a.)"); ax.set_ylabel("Densidad"); ax.legend(fontsize=8)

    # ── C: ACF FHR ────────────────────────────────────────────────────────────
    ax = axes[0, 2]
    max_lag = 200
    o_acf = mean_acf(orig_fhr, max_lag)
    s_acf = mean_acf(synt_fhr, max_lag)
    lags  = np.arange(max_lag + 1) / FS   # en segundos
    ax.plot(lags, o_acf, color="#2196F3", lw=2,   label="Original")
    ax.plot(lags, s_acf, color="#FF5722", lw=2,
            ls="--", label="Sintética")
    ax.axhline(0, color="gray", lw=0.8, ls=":")
    r = summary["fhr_acf_pearson_r"]
    ax.set_title(f"ACF FHR  (Pearson r = {r:.4f})")
    ax.set_xlabel("Lag (s)"); ax.set_ylabel("ACF"); ax.legend(fontsize=8)

    # ── D: PSD FHR ────────────────────────────────────────────────────────────
    ax = axes[1, 0]
    freqs, o_psd = mean_psd(orig_fhr)
    _,     s_psd = mean_psd(synt_fhr)
    min_len = min(len(o_psd), len(s_psd))
    ax.semilogy(freqs[:min_len], o_psd[:min_len],
                color="#2196F3", lw=2, label="Original")
    ax.semilogy(freqs[:min_len], s_psd[:min_len],
                color="#FF5722", lw=2, ls="--", label="Sintética")
    r_psd = summary["fhr_psd_log_corr"]
    ax.set_title(f"PSD FHR  (log-corr = {r_psd:.4f})")
    ax.set_xlabel("Frecuencia (Hz)"); ax.set_ylabel("PSD (log)")
    ax.legend(fontsize=8)

    # ── E: Correlación cruzada FHR↔UC ─────────────────────────────────────────
    ax = axes[1, 1]
    max_lag_xc = 200
    o_xc = mean_cross_corr(orig_fhr, orig_uc, max_lag_xc)
    s_xc = mean_cross_corr(synt_fhr, synt_uc, max_lag_xc)
    lags_xc = np.arange(-max_lag_xc, max_lag_xc + 1) / FS
    ax.plot(lags_xc, o_xc, color="#2196F3", lw=2,       label="Original")
    ax.plot(lags_xc, s_xc, color="#FF5722", lw=2, ls="--", label="Sintética")
    ax.axvline(0, color="gray", lw=0.8, ls=":")
    r_xc = summary["fhr_uc_xcorr_r"]
    ax.set_title(f"XCorr FHR↔UC  (Pearson r = {r_xc:.4f})")
    ax.set_xlabel("Lag (s)"); ax.set_ylabel("Corr. cruzada normalizada")
    ax.legend(fontsize=8)

    # ── F: Tabla resumen ──────────────────────────────────────────────────────
    ax = axes[1, 2]
    ax.axis("off")
    checks = summary.get("checks", [])
    rows = [["Métrica", "Valor", "Umbral"]]
    rows += [
        ["FHR KS stat",      f"{d['ks_statistic']:.4f}",    "< 0.05"],
        ["FHR Wasserstein",  f"{d['wasserstein']:.3f} bpm",  "< 1.0"],
        ["UC KS stat",       f"{d_uc['ks_statistic']:.4f}",  "< 0.05"],
        ["UC Wasserstein",   f"{d_uc['wasserstein']:.3f}",   "< 1.0"],
        ["ACF Pearson r",    f"{summary['fhr_acf_pearson_r']:.4f}", "> 0.98"],
        ["PSD log-corr",     f"{summary['fhr_psd_log_corr']:.4f}",  "> 0.95"],
        ["XCorr r FHR↔UC",  f"{summary['fhr_uc_xcorr_r']:.4f}",   "> 0.90"],
    ]
    tbl = ax.table(cellText=rows[1:], colLabels=rows[0],
                   cellLoc="center", loc="center")
    tbl.auto_set_font_size(True)
    tbl.scale(1, 1.5)
    ax.set_title("Resumen de fidelidad", pad=10)

    out = REPORT_DIR / "05_fidelity.png"
    fig.tight_layout()
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    return out


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    print("Cargando dataset_full.csv...")
    df   = pd.read_csv(FULL_CSV)
    orig = df[df["is_synthetic"] == False]
    synt = df[df["is_synthetic"] == True]

    print(f"Cargando señales (hasta {MAX_RECORDS} pares por grupo)...")
    orig_fhr, orig_uc = load_originals(orig,  MAX_RECORDS)
    synt_fhr, synt_uc = load_synthetics(synt, MAX_RECORDS)
    print(f"  Originales cargados: {len(orig_fhr)}")
    print(f"  Sintéticas cargadas: {len(synt_fhr)}")

    print("\nCalculando métricas de fidelidad...")
    summary = fidelity_summary(orig_fhr, synt_fhr, orig_uc, synt_uc)

    print("\n── Resultados ──────────────────────────────────────────────")
    for check in summary.get("checks", []):
        print(f"  {check}")

    print(f"\n  FHR ACF Pearson r : {summary['fhr_acf_pearson_r']}")
    print(f"  FHR PSD log-corr  : {summary['fhr_psd_log_corr']}")
    print(f"  XCorr FHR↔UC r    : {summary['fhr_uc_xcorr_r']}")

    # Guardar JSON (sin los arrays crudos)
    summary_clean = {k: v for k, v in summary.items()}
    out_json = REPORT_DIR / "05_fidelity.json"
    with open(out_json, "w") as f:
        json.dump(summary_clean, f, indent=2)
    print(f"\n  JSON → {out_json}")

    # Figura
    out_png = plot_fidelity(orig_fhr, synt_fhr, orig_uc, synt_uc, summary)
    print(f"  PNG  → {out_png}")
    print("\nValidación completa.")


if __name__ == "__main__":
    main()
