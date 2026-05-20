"""
04_clasificar_asfixia.py
=========================
Clasifica y cuenta casos de asfixia intraparto usando criterios combinados:
  - pH de arteria umbilical
  - Apgar al minuto 1
  - Apgar al minuto 5

Criterios clínicos implementados:
  1. Solo pH         : pH < 7.05
  2. Solo Apgar      : Apgar1 < 7 y/o Apgar5 < 7
  3. Combinado ACOG  : pH < 7.00 + Apgar5 < 7 (criterio American College OB-GYN)
  4. Combinado amplio: pH < 7.10 + Apgar1 < 7
  5. Asfixia clínica real (gold standard): pH < 7.05 + Apgar5 < 7 + HIE o NICU

Uso:
    python 04_clasificar_asfixia.py
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import os

# ── Configuración ──────────────────────────────────────────────────────────────
DATA_PATH   = "data/dataset_full.csv"
REPORT_DIR  = "reports"
os.makedirs(REPORT_DIR, exist_ok=True)

# ══════════════════════════════════════════════════════════════════════════════
# 1. CARGAR DATOS
# ══════════════════════════════════════════════════════════════════════════════
print("Cargando dataset...")
df = pd.read_csv(DATA_PATH)
orig = df[~df["is_synthetic"]].copy()   # solo originales para el análisis clínico
synt = df[ df["is_synthetic"]].copy()

print(f"  Total filas:    {len(df)}")
print(f"  Originales:     {len(orig)}")
print(f"  Sintéticas:     {len(synt)}")

# ══════════════════════════════════════════════════════════════════════════════
# 2. DEFINIR CRITERIOS DE CLASIFICACIÓN
# ══════════════════════════════════════════════════════════════════════════════

def clasificar_asfixia(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    ph     = df["pH"]
    ap1    = df["Apgar1"]
    ap5    = df["Apgar5"]
    hie    = df["HIE"]    if "HIE"      in df.columns else pd.Series(0, index=df.index)
    nicu   = df["NICU_days"] if "NICU_days" in df.columns else pd.Series(0, index=df.index)

    # ── Criterio 1: Solo pH ───────────────────────────────────────────────────
    df["asf_pH_severa"]   = (ph < 7.05).astype(int)
    df["asf_pH_moderada"] = ((ph >= 7.05) & (ph < 7.10)).astype(int)
    df["asf_pH_leve"]     = ((ph >= 7.10) & (ph < 7.20)).astype(int)

    # ── Criterio 2: Solo Apgar ────────────────────────────────────────────────
    # Depresión neonatal definida por Apgar bajo
    df["depresion_ap1"]     = (ap1 < 7).astype(int)   # depresión al min 1
    df["depresion_ap5"]     = (ap5 < 7).astype(int)   # depresión al min 5 (más grave)
    df["depresion_ap1_sev"] = (ap1 < 4).astype(int)   # depresión severa al min 1
    df["depresion_ap5_sev"] = (ap5 < 4).astype(int)   # depresión severa al min 5

    # ── Criterio 3: ACOG (American College of OB-GYN) ────────────────────────
    # Asfixia intraparto = pH < 7.00 + Apgar5 < 7 (criterio conservador oficial)
    df["asf_ACOG"] = ((ph < 7.00) & (ap5 < 7)).astype(int)

    # ── Criterio 4: Combinado amplio ──────────────────────────────────────────
    # pH < 7.10 + Apgar1 < 7 (detecta más casos borderline)
    df["asf_combinado_amplio"] = ((ph < 7.10) & (ap1 < 7)).astype(int)

    # ── Criterio 5: pH < 7.05 + Apgar5 < 7 ──────────────────────────────────
    df["asf_ph_apgar5"]   = ((ph < 7.05) & (ap5 < 7)).astype(int)

    # ── Criterio 6: Clínico estricto (gold standard) ─────────────────────────
    # pH < 7.05 + Apgar5 < 7 + evidencia de compromiso (HIE o NICU)
    df["asf_gold_standard"] = (
        (ph < 7.05) & (ap5 < 7) & ((hie > 0) | (nicu > 0))
    ).astype(int)

    # ── Categoría resumen ─────────────────────────────────────────────────────
    def categoria(row):
        ph_  = row["pH"]
        ap1_ = row["Apgar1"]
        ap5_ = row["Apgar5"]

        if pd.isna(ph_) or pd.isna(ap1_) or pd.isna(ap5_):
            return "Sin dato completo"

        # Asfixia clara: pH bajo + Apgar bajo en ambos
        if ph_ < 7.05 and ap5_ < 7:
            return "Asfixia confirmada (pH<7.05 + Apgar5<7)"
        # pH bajo pero Apgar recuperado
        if ph_ < 7.05 and ap5_ >= 7:
            return "Acidosis sin depresión Apgar5"
        # Apgar bajo pero pH normal
        if ph_ >= 7.05 and ap5_ < 7:
            return "Depresión neonatal sin acidosis severa"
        # Borderline
        if ph_ < 7.10 and ap1_ < 7:
            return "Borderline (pH<7.10 + Apgar1<7)"
        return "Normal"

    df["categoria_asfixia"] = df.apply(categoria, axis=1)

    return df

# ══════════════════════════════════════════════════════════════════════════════
# 3. APLICAR CLASIFICACIÓN
# ══════════════════════════════════════════════════════════════════════════════
orig = clasificar_asfixia(orig)

# ══════════════════════════════════════════════════════════════════════════════
# 4. REPORTE DE CONTEOS
# ══════════════════════════════════════════════════════════════════════════════

print("\n" + "="*65)
print("  REPORTE DE ASFIXIA INTRAPARTO — REGISTROS ORIGINALES (n=552)")
print("="*65)

criterios = {
    "1. pH < 7.05 (solo pH severo)"              : "asf_pH_severa",
    "2. pH < 7.10 (solo pH moderado+severo)"     : lambda d: ((d["pH"] < 7.10)).astype(int),
    "3. Apgar1 < 7 (depresión al min 1)"         : "depresion_ap1",
    "4. Apgar5 < 7 (depresión al min 5)"         : "depresion_ap5",
    "5. Apgar1 < 4 (depresión severa min 1)"     : "depresion_ap1_sev",
    "6. Apgar5 < 4 (depresión severa min 5)"     : "depresion_ap5_sev",
    "7. ACOG: pH<7.00 + Apgar5<7"               : "asf_ACOG",
    "8. pH<7.05 + Apgar5<7"                      : "asf_ph_apgar5",
    "9. pH<7.10 + Apgar1<7 (amplio)"             : "asf_combinado_amplio",
    "10. Gold standard: pH<7.05+Apgar5<7+HIE/NICU":"asf_gold_standard",
}

resumen = []
for nombre, col in criterios.items():
    if callable(col):
        mask = col(orig)
    else:
        mask = orig[col]
    n   = int(mask.sum())
    pct = n / len(orig) * 100
    print(f"  {nombre:<45} → {n:>4} casos ({pct:5.1f}%)")
    resumen.append({"criterio": nombre, "n_casos": n, "pct": round(pct, 1)})

# ── Tabla de categoría resumen ────────────────────────────────────────────────
print("\n── Categoría clínica combinada ──────────────────────────────────────")
cat_counts = orig["categoria_asfixia"].value_counts()
for cat, cnt in cat_counts.items():
    pct = cnt / len(orig) * 100
    print(f"  {cat:<50} → {cnt:>4} ({pct:5.1f}%)")

# ── Tabla cruzada pH × Apgar5 ─────────────────────────────────────────────────
print("\n── Tabla cruzada: pH vs Apgar5 ──────────────────────────────────────")
orig["pH_grupo"] = pd.cut(
    orig["pH"],
    bins  = [0, 7.00, 7.05, 7.10, 7.20, 10],
    labels= ["<7.00", "7.00–7.05", "7.05–7.10", "7.10–7.20", "≥7.20"],
    right = False
)
orig["Apgar5_grupo"] = pd.cut(
    orig["Apgar5"],
    bins  = [0, 4, 7, 11],
    labels= ["<4 (severo)", "4–6 (leve-mod)", "7–10 (normal)"],
    right = False
)
tabla_cruzada = pd.crosstab(
    orig["pH_grupo"], orig["Apgar5_grupo"],
    margins=True, margins_name="TOTAL"
)
print(tabla_cruzada.to_string())

# ── Tabla cruzada pH × Apgar1 ─────────────────────────────────────────────────
print("\n── Tabla cruzada: pH vs Apgar1 ──────────────────────────────────────")
orig["Apgar1_grupo"] = pd.cut(
    orig["Apgar1"],
    bins  = [0, 4, 7, 11],
    labels= ["<4 (severo)", "4–6 (leve-mod)", "7–10 (normal)"],
    right = False
)
tabla_cruzada2 = pd.crosstab(
    orig["pH_grupo"], orig["Apgar1_grupo"],
    margins=True, margins_name="TOTAL"
)
print(tabla_cruzada2.to_string())

# ══════════════════════════════════════════════════════════════════════════════
# 5. VISUALIZACIONES
# ══════════════════════════════════════════════════════════════════════════════

fig, axes = plt.subplots(2, 3, figsize=(18, 11))
fig.suptitle("Análisis de Asfixia Intraparto — CTU-UHB (n=552 originales)",
             fontsize=14, fontweight="bold")

COLORS = {
    "Asfixia confirmada (pH<7.05 + Apgar5<7)"    : "#D32F2F",
    "Acidosis sin depresión Apgar5"               : "#FF7043",
    "Depresión neonatal sin acidosis severa"       : "#FFA000",
    "Borderline (pH<7.10 + Apgar1<7)"             : "#7B1FA2",
    "Normal"                                       : "#388E3C",
    "Sin dato completo"                            : "#9E9E9E",
}

# ── Panel 1: Categorías de asfixia ───────────────────────────────────────────
ax = axes[0, 0]
cat_data  = orig["categoria_asfixia"].value_counts()
bar_colors = [COLORS.get(c, "#607D8B") for c in cat_data.index]
bars = ax.barh(range(len(cat_data)), cat_data.values, color=bar_colors, edgecolor="white")
ax.set_yticks(range(len(cat_data)))
ax.set_yticklabels([t[:40] for t in cat_data.index], fontsize=8)
ax.set_xlabel("N casos")
ax.set_title("Categorías clínicas de asfixia")
for i, (bar, val) in enumerate(zip(bars, cat_data.values)):
    ax.text(bar.get_width() + 2, i, f"{val} ({val/len(orig)*100:.1f}%)",
            va="center", fontsize=8)
ax.set_xlim(0, cat_data.max() * 1.35)

# ── Panel 2: Scatter pH vs Apgar5 ────────────────────────────────────────────
ax = axes[0, 1]
color_map = orig["categoria_asfixia"].map(COLORS).fillna("#607D8B")
ax.scatter(orig["pH"], orig["Apgar5"],
           c=color_map, alpha=0.6, s=30, edgecolors="none")
ax.axvline(7.05, color="red",    lw=1.5, ls="--", label="pH=7.05")
ax.axvline(7.10, color="orange", lw=1.2, ls=":",  label="pH=7.10")
ax.axhline(7,    color="blue",   lw=1.2, ls="--", label="Apgar5=7")
ax.set_xlabel("pH arteria umbilical")
ax.set_ylabel("Apgar 5 minutos")
ax.set_title("pH vs Apgar5 — dispersión clínica")
ax.legend(fontsize=8)
patches = [mpatches.Patch(color=v, label=k[:35]) for k, v in COLORS.items()
           if k in orig["categoria_asfixia"].values]
ax.legend(handles=patches, fontsize=6, loc="upper left")

# ── Panel 3: Scatter pH vs Apgar1 ────────────────────────────────────────────
ax = axes[0, 2]
ax.scatter(orig["pH"], orig["Apgar1"],
           c=color_map, alpha=0.6, s=30, edgecolors="none")
ax.axvline(7.05, color="red",    lw=1.5, ls="--", label="pH=7.05")
ax.axvline(7.10, color="orange", lw=1.2, ls=":",  label="pH=7.10")
ax.axhline(7,    color="blue",   lw=1.2, ls="--", label="Apgar1=7")
ax.set_xlabel("pH arteria umbilical")
ax.set_ylabel("Apgar 1 minuto")
ax.set_title("pH vs Apgar1 — dispersión clínica")
ax.legend(fontsize=8)

# ── Panel 4: Histograma pH coloreado ─────────────────────────────────────────
ax = axes[1, 0]
ph_vals = orig["pH"].dropna()
ax.hist(ph_vals, bins=40, color="#90CAF9", edgecolor="white", label="Todos")
ax.axvspan(0,    7.00, alpha=0.25, color="#D32F2F", label="pH<7.00")
ax.axvspan(7.00, 7.05, alpha=0.20, color="#FF7043", label="7.00–7.05")
ax.axvspan(7.05, 7.10, alpha=0.15, color="#FFA000", label="7.05–7.10")
ax.axvspan(7.10, 7.20, alpha=0.10, color="#7B1FA2", label="7.10–7.20")
ax.set_xlabel("pH arteria umbilical")
ax.set_ylabel("N registros")
ax.set_title("Distribución de pH")
ax.legend(fontsize=8)

# ── Panel 5: Distribución Apgar1 y Apgar5 ────────────────────────────────────
ax = axes[1, 1]
bins = np.arange(0, 12) - 0.2
ax.hist(orig["Apgar1"].dropna(), bins=bins,       alpha=0.7,
        color="#1565C0", label="Apgar 1 min", width=0.4)
ax.hist(orig["Apgar5"].dropna(), bins=bins + 0.4, alpha=0.7,
        color="#2E7D32", label="Apgar 5 min", width=0.4)
ax.axvline(6.5, color="red", lw=1.5, ls="--", label="Umbral = 7")
ax.set_xlabel("Puntuación Apgar")
ax.set_ylabel("N registros")
ax.set_title("Distribución Apgar1 vs Apgar5")
ax.set_xticks(range(0, 11))
ax.legend()

# ── Panel 6: Barras comparativas por criterio ─────────────────────────────────
ax = axes[1, 2]
df_res = pd.DataFrame(resumen)
short_labels = [r["criterio"].split(":")[0].strip() for r in resumen]
bar_c = ["#D32F2F" if "severo" in r["criterio"] or "ACOG" in r["criterio"]
         or "gold" in r["criterio"].lower() or "Apgar5<7" in r["criterio"]
         else "#FF7043" for r in resumen]
bars2 = ax.barh(range(len(df_res)), df_res["n_casos"], color=bar_c, edgecolor="white")
ax.set_yticks(range(len(df_res)))
ax.set_yticklabels(short_labels, fontsize=8)
ax.set_xlabel("N casos")
ax.set_title("Conteo por criterio diagnóstico")
for i, (bar, row) in enumerate(zip(bars2, resumen)):
    ax.text(bar.get_width() + 1, i, f"{row['n_casos']} ({row['pct']}%)",
            va="center", fontsize=8)
ax.set_xlim(0, df_res["n_casos"].max() * 1.4)

plt.tight_layout()
out_path = f"{REPORT_DIR}/asfixia_analysis.png"
plt.savefig(out_path, dpi=150, bbox_inches="tight")
plt.close()
print(f"\nGráfico guardado → {out_path}")

# ══════════════════════════════════════════════════════════════════════════════
# 6. GUARDAR CSV CON CLASIFICACIÓN
# ══════════════════════════════════════════════════════════════════════════════
out_csv = "data/dataset_original_clasificado.csv"
orig.to_csv(out_csv, index=False)
print(f"CSV clasificado guardado → {out_csv}")

# ══════════════════════════════════════════════════════════════════════════════
# 7. RECOMENDACIÓN DE ETIQUETA PARA MODELADO
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "="*65)
print("  RECOMENDACIÓN DE ETIQUETA PARA MODELADO")
print("="*65)

for nombre_et, col_et, desc in [
    ("Estricta (ACOG)",   "asf_ACOG",          "pH<7.00 + Apgar5<7"),
    ("Recomendada",       "asf_ph_apgar5",      "pH<7.05 + Apgar5<7"),
    ("Amplia",            "asf_combinado_amplio","pH<7.10 + Apgar1<7"),
    ("Solo pH",           "asf_pH_severa",       "pH<7.05"),
]:
    pos = int(orig[col_et].sum())
    neg = len(orig) - pos
    ratio = neg / pos if pos > 0 else float("inf")
    print(f"\n  [{nombre_et}] — {desc}")
    print(f"    Positivos (asfixia): {pos}  |  Negativos: {neg}  |  Ratio: {ratio:.1f}:1")

print(f"""
  → Para detección clínica real:    usar 'asf_ph_apgar5'  (pH<7.05 + Apgar5<7)
  → Para maximizar sensibilidad:    usar 'asf_combinado_amplio' (pH<7.10 + Apgar1<7)
  → Para criterio académico estricto: usar 'asf_ACOG' (pH<7.00 + Apgar5<7)
""")