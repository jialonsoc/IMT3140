"""
diag_acf.py — diagnóstico de ACF con métrica correcta
------------------------------------------------------
Compara ACF original vs sintética punto a punto (bias por lag),
en lugar del Pearson r global que puede ser engañoso.

Ejecutar:
    cd ctg_pipeline
    python diag_acf.py
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from config import RAW_DIR, SYNTH_DIR, REPORT_DIR, FS
from utils.signal import clean_fhr

N = 80   # pares a comparar
MAX_LAG_S = 300   # hasta 5 minutos

# ── Cargar series ──────────────────────────────────────────────────────────────
orig_files = sorted(RAW_DIR.glob("*_fhr.npy"))[:N]
synt_files = sorted(SYNTH_DIR.glob("syn_*_fhr.npy"))[:N]

orig_series = [clean_fhr(np.load(f)) for f in orig_files]
synt_series = [np.load(f) for f in synt_files]

print(f"Series cargadas: {len(orig_series)} orig, {len(synt_series)} synt")

# ── ACF por serie ──────────────────────────────────────────────────────────────
MAX_LAG_N = MAX_LAG_S * FS   # en muestras

def acf_single(x, max_lag):
    x = x[~np.isnan(x)]
    if len(x) < max_lag + 1:
        return None
    x = x - x.mean()
    n = len(x)
    result = []
    for lag in range(max_lag + 1):
        c = np.mean(x[:n - lag] * x[lag:])
        result.append(c)
    arr = np.array(result)
    if arr[0] != 0:
        arr /= arr[0]
    return arr

orig_acfs = [acf_single(s, MAX_LAG_N) for s in orig_series]
synt_acfs = [acf_single(s, MAX_LAG_N) for s in synt_series]
orig_acfs = [a for a in orig_acfs if a is not None]
synt_acfs = [a for a in synt_acfs if a is not None]

orig_mean = np.mean(orig_acfs, axis=0)
synt_mean = np.mean(synt_acfs, axis=0)
bias      = orig_mean - synt_mean   # positivo = sintética decae más rápido

lags_s = np.arange(MAX_LAG_N + 1) / FS   # eje en segundos

# ── Tabla de bias en lags clave ────────────────────────────────────────────────
checkpoints = [10, 20, 30, 60, 120, 180, 240, 300]
print(f"\n{'Lag (s)':>8} {'ACF orig':>10} {'ACF synt':>10} {'Bias':>10}")
print("─" * 42)
for s in checkpoints:
    idx = s * FS
    if idx <= MAX_LAG_N:
        print(f"{s:>8}  {orig_mean[idx]:>10.4f}  {synt_mean[idx]:>10.4f}  {bias[idx]:>+10.4f}")

# ── Figura ─────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
fig.suptitle("Diagnóstico ACF — Original vs Sintética (FHR)", fontsize=13)

# Panel 1: ACF media con banda de ±1 std
ax = axes[0]
orig_std = np.std(orig_acfs, axis=0)
synt_std = np.std(synt_acfs, axis=0)
ax.plot(lags_s, orig_mean, color="#2196F3", lw=2, label="Original")
ax.fill_between(lags_s, orig_mean - orig_std, orig_mean + orig_std,
                alpha=0.15, color="#2196F3")
ax.plot(lags_s, synt_mean, color="#FF5722", lw=2, ls="--", label="Sintética")
ax.fill_between(lags_s, synt_mean - synt_std, synt_mean + synt_std,
                alpha=0.15, color="#FF5722")
ax.axhline(1/np.e, color="gray", ls=":", lw=1, label=f"1/e ≈ 0.368")
ax.set_xlabel("Lag (s)"); ax.set_ylabel("ACF normalizada")
ax.set_title("ACF media ± 1 std"); ax.legend(fontsize=9)
ax.set_xlim(0, MAX_LAG_S)

# Panel 2: Bias = ACF_orig - ACF_synt
ax = axes[1]
ax.plot(lags_s, bias, color="#7B1FA2", lw=2)
ax.axhline(0, color="gray", lw=0.8, ls=":")
ax.axhline(0.05, color="orange", lw=1, ls="--", label="±0.05")
ax.axhline(-0.05, color="orange", lw=1, ls="--")
# Marcar checkpoints
for s in [20, 60, 120, 240]:
    idx = s * FS
    if idx <= MAX_LAG_N:
        ax.axvline(s, color="lightgray", lw=0.8, ls=":")
        ax.annotate(f"{bias[idx]:+.3f}", xy=(s, bias[idx]),
                    xytext=(s + 5, bias[idx] + 0.02), fontsize=8, color="#7B1FA2")
ax.set_xlabel("Lag (s)"); ax.set_ylabel("Bias (orig − synt)")
ax.set_title("Bias por lag\n(positivo = sintética decae más rápido)")
ax.legend(fontsize=9); ax.set_xlim(0, MAX_LAG_S)

# Panel 3: lag donde ACF cruza 1/e por primera vez (distribución)
ax = axes[2]
thresh = 1 / np.e
def first_cross(acf_arr, thr):
    for i, v in enumerate(acf_arr):
        if v < thr:
            return i / FS
    return MAX_LAG_S

orig_cross = [first_cross(a, thresh) for a in orig_acfs]
synt_cross = [first_cross(a, thresh) for a in synt_acfs]
bins = np.linspace(0, MAX_LAG_S, 40)
ax.hist(orig_cross, bins=bins, alpha=0.7, color="#2196F3", label="Original", density=True)
ax.hist(synt_cross, bins=bins, alpha=0.7, color="#FF5722", label="Sintética", density=True)
ax.axvline(np.median(orig_cross), color="#2196F3", lw=2, ls="--",
           label=f"Mediana orig: {np.median(orig_cross):.0f}s")
ax.axvline(np.median(synt_cross), color="#FF5722", lw=2, ls="--",
           label=f"Mediana synt: {np.median(synt_cross):.0f}s")
ax.set_xlabel("Lag donde ACF < 1/e (s)")
ax.set_title("Distribución del 'tiempo de memoria'\npor registro"); ax.legend(fontsize=8)

plt.tight_layout()
out = REPORT_DIR / "diag_acf.png"
fig.savefig(out, dpi=150, bbox_inches="tight")
plt.close()
print(f"\nFigura guardada → {out}")
