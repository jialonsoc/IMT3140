"""
05_apgar_analysis.py
=====================
Análisis de casos con Apgar < 7 (sin considerar pH).
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import os

DATA_PATH  = "data/dataset_full.csv"
REPORT_DIR = "reports"
os.makedirs(REPORT_DIR, exist_ok=True)

# ── Cargar solo originales ─────────────────────────────────────────────────────
df   = pd.read_csv(DATA_PATH)
orig = df[~df["is_synthetic"]].copy()
n    = len(orig)

print(f"\nRegistros originales: {n}")
print("=" * 55)

# ── Conteos Apgar ──────────────────────────────────────────────────────────────
criterios = {
    "Apgar1 < 7  (depresión min 1)"      : orig["Apgar1"] < 7,
    "Apgar1 < 4  (depresión severa min 1)": orig["Apgar1"] < 4,
    "Apgar5 < 7  (depresión min 5)"      : orig["Apgar5"] < 7,
    "Apgar5 < 4  (depresión severa min 5)": orig["Apgar5"] < 4,
    "Apgar1 < 7  Y  Apgar5 < 7"          : (orig["Apgar1"] < 7) & (orig["Apgar5"] < 7),
    "Apgar1 < 7  O  Apgar5 < 7"          : (orig["Apgar1"] < 7) | (orig["Apgar5"] < 7),
}

resumen = []
for nombre, mask in criterios.items():
    n_pos = int(mask.sum())
    n_neg = n - n_pos
    pct   = n_pos / n * 100
    print(f"  {nombre:<42} → {n_pos:>4} casos ({pct:5.1f}%) | negativos: {n_neg}")
    resumen.append({"criterio": nombre, "n_pos": n_pos, "n_neg": n_neg, "pct": pct})

# ── Distribución detallada por puntuación ─────────────────────────────────────
print("\n── Distribución por puntuación Apgar ────────────────────────")
print(f"{'Score':<8} {'Apgar1':>10} {'%':>7}  {'Apgar5':>10} {'%':>7}")
print("-" * 48)
for score in range(11):
    n1 = int((orig["Apgar1"] == score).sum())
    n5 = int((orig["Apgar5"] == score).sum())
    print(f"  {score:<6} {n1:>10} {n1/n*100:>6.1f}%  {n5:>10} {n5/n*100:>6.1f}%")

# ── Tabla cruzada Apgar1 × Apgar5 ────────────────────────────────────────────
print("\n── Tabla cruzada Apgar1 × Apgar5 ───────────────────────────")

def grupo_apgar(x):
    if x < 4:  return "0–3 (severo)"
    if x < 7:  return "4–6 (moderado)"
    return "7–10 (normal)"

orig["Apgar1_cat"] = orig["Apgar1"].apply(grupo_apgar)
orig["Apgar5_cat"] = orig["Apgar5"].apply(grupo_apgar)

tabla = pd.crosstab(orig["Apgar1_cat"], orig["Apgar5_cat"],
                    margins=True, margins_name="TOTAL")
print(tabla.to_string())

# ── Recuperación: Apgar1<7 que mejoran a Apgar5≥7 ────────────────────────────
print("\n── Recuperación neonatal ────────────────────────────────────")
deprimidos_1  = orig[orig["Apgar1"] < 7]
recuperados   = deprimidos_1[deprimidos_1["Apgar5"] >= 7]
no_recuperados= deprimidos_1[deprimidos_1["Apgar5"] < 7]

print(f"  Apgar1 < 7:                    {len(deprimidos_1)} casos")
print(f"  → Se recuperan  (Apgar5 ≥ 7): {len(recuperados)}  ({len(recuperados)/len(deprimidos_1)*100:.1f}%)")
print(f"  → No recuperan  (Apgar5 < 7): {len(no_recuperados)}  ({len(no_recuperados)/len(deprimidos_1)*100:.1f}%)")

# ══════════════════════════════════════════════════════════════════════════════
# GRÁFICOS
# ══════════════════════════════════════════════════════════════════════════════
fig, axes = plt.subplots(1, 3, figsize=(16, 5))
fig.suptitle("Análisis Apgar — CTU-UHB (n=552 originales)",
             fontsize=13, fontweight="bold")

# Panel 1: Distribución Apgar1 y Apgar5
ax = axes[0]
scores = range(11)
n1_vals = [(orig["Apgar1"] == s).sum() for s in scores]
n5_vals = [(orig["Apgar5"] == s).sum() for s in scores]
x = np.array(list(scores))
ax.bar(x - 0.2, n1_vals, width=0.4, color="#1565C0", alpha=0.8, label="Apgar 1 min")
ax.bar(x + 0.2, n5_vals, width=0.4, color="#2E7D32", alpha=0.8, label="Apgar 5 min")
ax.axvline(6.5, color="red", lw=2, ls="--", label="Umbral = 7")
ax.set_xlabel("Puntuación Apgar")
ax.set_ylabel("N registros")
ax.set_title("Distribución por puntuación")
ax.set_xticks(list(scores))
ax.legend()

# Panel 2: Barras de criterios
ax = axes[1]
df_res = pd.DataFrame(resumen)
colores = ["#1565C0","#0D47A1","#2E7D32","#1B5E20","#D32F2F","#B71C1C"]
bars = ax.barh(range(len(df_res)), df_res["n_pos"],
               color=colores[:len(df_res)], edgecolor="white")
ax.set_yticks(range(len(df_res)))
ax.set_yticklabels([r["criterio"] for r in resumen], fontsize=8)
ax.set_xlabel("N casos")
ax.set_title("Casos por criterio Apgar")
for i, (bar, row) in enumerate(zip(bars, resumen)):
    ax.text(bar.get_width() + 0.5, i,
            f"{row['n_pos']} ({row['pct']:.1f}%)", va="center", fontsize=9)
ax.set_xlim(0, df_res["n_pos"].max() * 1.4)

# Panel 3: Scatter Apgar1 vs Apgar5
ax = axes[2]
ax.scatter(orig["Apgar1"], orig["Apgar5"],
           alpha=0.4, s=25, color="#546E7A", edgecolors="none")
# Resaltar casos con algún Apgar < 7
mask_any = (orig["Apgar1"] < 7) | (orig["Apgar5"] < 7)
ax.scatter(orig.loc[mask_any, "Apgar1"], orig.loc[mask_any, "Apgar5"],
           alpha=0.8, s=40, color="#D32F2F", edgecolors="none", label="Algún Apgar<7")
ax.axvline(6.5, color="orange", lw=1.5, ls="--")
ax.axhline(6.5, color="blue",   lw=1.5, ls="--")
ax.set_xlabel("Apgar 1 minuto")
ax.set_ylabel("Apgar 5 minutos")
ax.set_title("Apgar1 vs Apgar5")
ax.set_xticks(range(0, 11))
ax.set_yticks(range(0, 11))
ax.legend(fontsize=9)

plt.tight_layout()
out = f"{REPORT_DIR}/apgar_analysis.png"
plt.savefig(out, dpi=150, bbox_inches="tight")
plt.close()
print(f"\nGráfico → {out}")