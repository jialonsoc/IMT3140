"""
04_analysis.py
--------------
Análisis exploratorio (EDA) del dataset completo (originales + sintéticas).
Enfocado en asfixia intraparto y Apgar.

Secciones
---------
  1. Tamaño y calidad del dataset
  2. Variables de outcome (pH, Apgar, HIE)
  3. Clasificación de asfixia (criterios ACOG/FIGO)
  4. Factores de riesgo maternos
  5. Comparación originales vs sintéticas
  6. Figuras (guardadas en reports/)

Definiciones de asfixia usadas
-------------------------------
  - Apgar1 < 7          : depresión neonatal al minuto
  - Apgar5 < 7          : depresión neonatal a los 5 min (mayor relevancia)
  - pH < 7.10           : acidosis neonatal (criterio ACOG)
  - pH < 7.00           : acidosis severa
  - HIE > 0             : encefalopatía hipóxico-isquémica

Referencias
-----------
  ACOG (2014). Neonatal encephalopathy and neurologic outcome (2nd ed.).
  American College of Obstetricians and Gynecologists.

  Ayres-de-Campos, D., et al. (2015). FIGO consensus guidelines on
  intrapartum fetal monitoring: Cardiotocography.
  Int J Gynecol Obstet, 131(1), 13–24.
  https://doi.org/10.1016/j.ijgo.2015.06.020

Ejecutar después de 03_bootstrap.py:
    cd ctg_pipeline
    python 04_analysis.py
"""

import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from config import FULL_CSV, REPORT_DIR

REPORT_DIR.mkdir(parents=True, exist_ok=True)

# ── Clasificadores clínicos ────────────────────────────────────────────────────

def classify_ph(ph: float) -> str:
    if pd.isna(ph):     return "Sin dato"
    if ph < 7.00:       return "Acidosis severa (pH < 7.00)"
    if ph < 7.10:       return "Acidosis moderada (7.00–7.10)"
    if ph < 7.20:       return "Acidosis leve (7.10–7.20)"
    return "Normal (pH ≥ 7.20)"

def classify_apgar1(a: float) -> str:
    if pd.isna(a):  return "Sin dato"
    if a <= 3:      return "Depresión severa (≤ 3)"
    if a < 7:       return "Depresión moderada (4–6)"
    return "Normal (≥ 7)"

def classify_apgar5(a: float) -> str:
    if pd.isna(a):  return "Sin dato"
    if a < 7:       return "Depresión (< 7)"
    return "Normal (≥ 7)"

def asphyxia_acog(row: pd.Series) -> bool:
    """
    Criterio ACOG de asfixia perinatal (presencia de al menos uno):
      - pH < 7.00
      - Apgar5 < 7
      - HIE > 0
    """
    ph_crit    = (not pd.isna(row.get("pH")))    and row["pH"]    < 7.00
    apgar_crit = (not pd.isna(row.get("Apgar5"))) and row["Apgar5"] < 7
    hie_crit   = (not pd.isna(row.get("HIE")))   and row["HIE"]   > 0
    return ph_crit or apgar_crit or hie_crit


# ── Helpers de impresión ───────────────────────────────────────────────────────

def section(title: str):
    print(f"\n{'═' * 60}")
    print(f"  {title}")
    print(f"{'═' * 60}")

def freq_table(series: pd.Series, label: str = ""):
    counts = series.value_counts(dropna=False).sort_index()
    total  = len(series)
    if label:
        print(f"\n  {label}")
    for val, n in counts.items():
        print(f"    {str(val):<40} {n:>5}  ({n/total*100:5.1f}%)")


# ── Sección 1: Tamaño y calidad ────────────────────────────────────────────────

def section_dataset(df: pd.DataFrame, orig: pd.DataFrame, synt: pd.DataFrame):
    section("1. TAMAÑO Y CALIDAD DEL DATASET")
    print(f"\n  Registros originales : {len(orig)}")
    print(f"  Series sintéticas    : {len(synt)}")
    print(f"  Total dataset        : {len(df)}")
    print(f"  N_MULTIPLIER         : {len(synt) // len(orig) if len(orig) > 0 else '—'}")
    print(f"\n  Columnas disponibles : {len(df.columns)}")
    print(f"  Duración media (orig): {orig['duration_min'].mean():.1f} min "
          f"(rango {orig['duration_min'].min():.0f}–{orig['duration_min'].max():.0f})")
    print(f"  FHR media (orig)     : {orig['fhr_mean'].mean():.1f} bpm "
          f"± {orig['fhr_mean'].std():.1f}")

    missing = orig[["pH", "Apgar1", "Apgar5", "HIE"]].isna().sum()
    print("\n  Valores faltantes en originales:")
    for col, n in missing.items():
        print(f"    {col:<10} {n}")


# ── Sección 2: Outcomes neonatales ────────────────────────────────────────────

def section_outcomes(orig: pd.DataFrame) -> dict:
    section("2. VARIABLES DE OUTCOME (originales)")

    orig = orig.copy()
    orig["pH_class"]     = orig["pH"].apply(classify_ph)
    orig["Apgar1_class"] = orig["Apgar1"].apply(classify_apgar1)
    orig["Apgar5_class"] = orig["Apgar5"].apply(classify_apgar5)

    freq_table(orig["pH_class"],     "pH arterial umbilical")
    freq_table(orig["Apgar1_class"], "Apgar al minuto 1")
    freq_table(orig["Apgar5_class"], "Apgar a los 5 minutos")

    print("\n  HIE (encefalopatía hipóxico-isquémica):")
    hie_counts = orig["HIE"].value_counts(dropna=False)
    for val, n in hie_counts.items():
        print(f"    HIE = {val}: {n}  ({n/len(orig)*100:.1f}%)")

    stats = {
        "pH"    : orig["pH"].describe().round(3).to_dict(),
        "Apgar1": orig["Apgar1"].describe().round(2).to_dict(),
        "Apgar5": orig["Apgar5"].describe().round(2).to_dict(),
    }
    print("\n  Estadísticas descriptivas:")
    for var, s in stats.items():
        print(f"    {var}: media={s['mean']:.2f}  std={s['std']:.2f}  "
              f"min={s['min']:.2f}  max={s['max']:.2f}")

    return stats


# ── Sección 3: Clasificación de asfixia ───────────────────────────────────────

def section_asphyxia(orig: pd.DataFrame) -> pd.DataFrame:
    section("3. CLASIFICACIÓN DE ASFIXIA (criterios ACOG, originales)")

    orig = orig.copy()
    orig["asphyxia_acog"] = orig.apply(asphyxia_acog, axis=1)

    n = len(orig)
    n_asf = orig["asphyxia_acog"].sum()
    print(f"\n  Asfixia perinatal (ACOG): {n_asf} / {n}  ({n_asf/n*100:.1f}%)")

    print("\n  Desglose por criterio individual:")
    ph_crit    = (orig["pH"] < 7.00).sum()
    apgar_crit = (orig["Apgar5"] < 7).sum()
    hie_crit   = (orig["HIE"] > 0).sum()
    print(f"    pH < 7.00      : {ph_crit}  ({ph_crit/n*100:.1f}%)")
    print(f"    Apgar5 < 7     : {apgar_crit}  ({apgar_crit/n*100:.1f}%)")
    print(f"    HIE > 0        : {hie_crit}  ({hie_crit/n*100:.1f}%)")

    print("\n  Apgar1 < 7 (depresión al minuto):")
    apgar1_low = (orig["Apgar1"] < 7).sum()
    print(f"    {apgar1_low} / {n}  ({apgar1_low/n*100:.1f}%)")

    return orig


# ── Sección 4: Factores de riesgo maternos ────────────────────────────────────

def section_risk_factors(orig: pd.DataFrame):
    section("4. FACTORES DE RIESGO MATERNOS (originales)")

    binary_factors = {
        "Diabetes"    : "Diabetes",
        "Hypertension": "Hipertensión",
        "Preeclampsia": "Preeclampsia",
        "Liq_praecox" : "Rotura prematura de membranas",
        "Pyrexia"     : "Fiebre materna",
        "Meconium"    : "Meconio",
        "Induced"     : "Inducción del parto",
        "NoProgress"  : "No progresión",
    }
    n = len(orig)
    for col, label in binary_factors.items():
        if col in orig.columns:
            pos = (orig[col] == 1).sum()
            print(f"  {label:<40} {pos:>4} / {n}  ({pos/n*100:4.1f}%)")

    if "Age" in orig.columns:
        print(f"\n  Edad materna: media={orig['Age'].mean():.1f}  "
              f"std={orig['Age'].std():.1f}  "
              f"rango [{orig['Age'].min():.0f}–{orig['Age'].max():.0f}]")

    if "Deliv_type" in orig.columns:
        print("\n  Tipo de parto:")
        freq_table(orig["Deliv_type"], "")


# ── Sección 5: Comparación originales vs sintéticas ──────────────────────────

def section_comparison(orig: pd.DataFrame, synt: pd.DataFrame):
    section("5. COMPARACIÓN ORIGINALES vs SINTÉTICAS")

    sig_vars = ["fhr_mean", "fhr_std", "fhr_nan_pct", "uc_mean"]
    print(f"\n  {'Variable':<20} {'Orig media':>12} {'Orig std':>10} "
          f"{'Synt media':>12} {'Synt std':>10}")
    print("  " + "─" * 66)
    for var in sig_vars:
        if var in orig.columns and var in synt.columns:
            print(f"  {var:<20} {orig[var].mean():>12.3f} {orig[var].std():>10.3f} "
                  f"{synt[var].mean():>12.3f} {synt[var].std():>10.3f}")

    clin_vars = ["pH", "Apgar1", "Apgar5"]
    print(f"\n  {'Variable clínica':<20} {'Orig media':>12} {'Synt media':>12}  (debe ser igual)")
    print("  " + "─" * 48)
    for var in clin_vars:
        if var in orig.columns and var in synt.columns:
            print(f"  {var:<20} {orig[var].mean():>12.4f} {synt[var].mean():>12.4f}")

    print(f"\n  Proporción Apgar1 < 7 — orig: {(orig['Apgar1'] < 7).mean():.4f}  "
          f"synt: {(synt['Apgar1'] < 7).mean():.4f}")
    print(f"  Proporción Apgar5 < 7 — orig: {(orig['Apgar5'] < 7).mean():.4f}  "
          f"synt: {(synt['Apgar5'] < 7).mean():.4f}")


# ── Sección 6: Figuras ────────────────────────────────────────────────────────

def make_figures(orig: pd.DataFrame, synt: pd.DataFrame):
    section("6. GENERANDO FIGURAS")

    fig = plt.figure(figsize=(18, 14))
    gs  = gridspec.GridSpec(3, 3, figure=fig, hspace=0.42, wspace=0.35)
    fig.suptitle("EDA — CTG Dataset (originales + sintéticas)", fontsize=14, y=1.01)

    # ── Panel A: Distribución pH ───────────────────────────────────────────────
    ax = fig.add_subplot(gs[0, 0])
    bins = np.linspace(6.8, 7.5, 30)
    ax.hist(orig["pH"].dropna(), bins=bins, alpha=0.7, label="Original", color="#2196F3")
    ax.hist(synt["pH"].dropna(), bins=bins, alpha=0.5, label="Sintética", color="#FF5722")
    ax.axvline(7.00, color="red",    ls="--", lw=1.2, label="pH 7.00")
    ax.axvline(7.10, color="orange", ls="--", lw=1.2, label="pH 7.10")
    ax.set_xlabel("pH umbilical"); ax.set_ylabel("Frecuencia")
    ax.set_title("Distribución pH"); ax.legend(fontsize=7)

    # ── Panel B: Distribución Apgar1 ──────────────────────────────────────────
    ax = fig.add_subplot(gs[0, 1])
    apgar_vals = sorted(orig["Apgar1"].dropna().unique())
    orig_counts = orig["Apgar1"].value_counts().sort_index()
    synt_counts = synt["Apgar1"].value_counts().sort_index()
    x = np.arange(len(apgar_vals))
    w = 0.35
    ax.bar(x - w/2, [orig_counts.get(v, 0) for v in apgar_vals],
           width=w, label="Original", color="#2196F3", alpha=0.8)
    ax.bar(x + w/2, [synt_counts.get(v, 0) for v in apgar_vals],
           width=w, label="Sintética", color="#FF5722", alpha=0.8)
    ax.set_xticks(x); ax.set_xticklabels([int(v) for v in apgar_vals])
    ax.axvline(x[list(apgar_vals).index(7)] - 0.5 if 7 in list(apgar_vals) else -1,
               color="gray", ls=":", lw=1)
    ax.set_xlabel("Apgar 1 min"); ax.set_ylabel("Frecuencia")
    ax.set_title("Distribución Apgar1"); ax.legend(fontsize=7)

    # ── Panel C: Distribución Apgar5 ──────────────────────────────────────────
    ax = fig.add_subplot(gs[0, 2])
    apgar5_vals = sorted(orig["Apgar5"].dropna().unique())
    orig5 = orig["Apgar5"].value_counts().sort_index()
    synt5 = synt["Apgar5"].value_counts().sort_index()
    x5 = np.arange(len(apgar5_vals))
    ax.bar(x5 - w/2, [orig5.get(v, 0) for v in apgar5_vals],
           width=w, label="Original", color="#2196F3", alpha=0.8)
    ax.bar(x5 + w/2, [synt5.get(v, 0) for v in apgar5_vals],
           width=w, label="Sintética", color="#FF5722", alpha=0.8)
    ax.set_xticks(x5); ax.set_xticklabels([int(v) for v in apgar5_vals])
    ax.set_xlabel("Apgar 5 min"); ax.set_ylabel("Frecuencia")
    ax.set_title("Distribución Apgar5"); ax.legend(fontsize=7)

    # ── Panel D: Scatter pH vs Apgar1 (originales) ────────────────────────────
    ax = fig.add_subplot(gs[1, 0])
    mask = orig["pH"].notna() & orig["Apgar1"].notna()
    sc = ax.scatter(orig.loc[mask, "pH"], orig.loc[mask, "Apgar1"],
                    c=orig.loc[mask, "Apgar1"], cmap="RdYlGn",
                    alpha=0.6, s=18, vmin=0, vmax=10)
    ax.axvline(7.00, color="red",    ls="--", lw=1, label="pH 7.00")
    ax.axvline(7.10, color="orange", ls="--", lw=1, label="pH 7.10")
    ax.axhline(7,    color="gray",   ls=":",  lw=1, label="Apgar 7")
    ax.set_xlabel("pH umbilical"); ax.set_ylabel("Apgar 1 min")
    ax.set_title("pH vs Apgar1 (originales)"); ax.legend(fontsize=7)
    plt.colorbar(sc, ax=ax, label="Apgar1")

    # ── Panel E: Scatter pH vs Apgar5 (originales) ────────────────────────────
    ax = fig.add_subplot(gs[1, 1])
    mask5 = orig["pH"].notna() & orig["Apgar5"].notna()
    sc5 = ax.scatter(orig.loc[mask5, "pH"], orig.loc[mask5, "Apgar5"],
                     c=orig.loc[mask5, "Apgar5"], cmap="RdYlGn",
                     alpha=0.6, s=18, vmin=0, vmax=10)
    ax.axvline(7.00, color="red",    ls="--", lw=1)
    ax.axvline(7.10, color="orange", ls="--", lw=1)
    ax.axhline(7,    color="gray",   ls=":",  lw=1)
    ax.set_xlabel("pH umbilical"); ax.set_ylabel("Apgar 5 min")
    ax.set_title("pH vs Apgar5 (originales)")
    plt.colorbar(sc5, ax=ax, label="Apgar5")

    # ── Panel F: FHR media por grupo Apgar1 ───────────────────────────────────
    ax = fig.add_subplot(gs[1, 2])
    orig_copy = orig.copy()
    orig_copy["grupo"] = orig_copy["Apgar1"].apply(
        lambda a: "Apgar1 < 7" if (not pd.isna(a) and a < 7) else "Apgar1 ≥ 7"
    )
    for grupo, color in [("Apgar1 < 7", "#E53935"), ("Apgar1 ≥ 7", "#43A047")]:
        vals = orig_copy.loc[orig_copy["grupo"] == grupo, "fhr_mean"].dropna()
        ax.hist(vals, bins=20, alpha=0.7, label=f"{grupo} (n={len(vals)})",
                color=color, density=True)
    ax.set_xlabel("FHR media (bpm)"); ax.set_ylabel("Densidad")
    ax.set_title("FHR media por grupo Apgar1"); ax.legend(fontsize=7)

    # ── Panel G: Criterios de asfixia ACOG (barras) ──────────────────────────
    ax = fig.add_subplot(gs[2, 0])
    n_orig = len(orig)
    criterios = {
        "pH < 7.00"    : (orig["pH"] < 7.00).sum(),
        "Apgar5 < 7"   : (orig["Apgar5"] < 7).sum(),
        "HIE > 0"      : (orig["HIE"] > 0).sum(),
        "Asfixia ACOG" : orig.apply(asphyxia_acog, axis=1).sum(),
    }
    bars = ax.bar(list(criterios.keys()), list(criterios.values()),
                  color=["#EF9A9A", "#FFCC80", "#CE93D8", "#EF5350"])
    ax.bar_label(bars, labels=[f"{v}\n({v/n_orig*100:.1f}%)" for v in criterios.values()],
                 fontsize=8, padding=2)
    ax.set_ylabel("n registros (originales)")
    ax.set_title("Criterios de asfixia ACOG")
    ax.tick_params(axis="x", labelsize=8)

    # ── Panel H: Factores de riesgo ───────────────────────────────────────────
    ax = fig.add_subplot(gs[2, 1])
    factors = {
        "Diabetes": "Diabetes",
        "Hypertension": "HTA",
        "Preeclampsia": "Preecl.",
        "Liq_praecox": "RPM",
        "Pyrexia": "Fiebre",
        "Meconium": "Meconio",
        "Induced": "Inducido",
    }
    vals_f = [(orig[col] == 1).sum() / n_orig * 100
              for col in factors if col in orig.columns]
    labs_f = [v for col, v in factors.items() if col in orig.columns]
    bars_f = ax.barh(labs_f, vals_f, color="#90CAF9")
    ax.bar_label(bars_f, labels=[f"{v:.1f}%" for v in vals_f], fontsize=8, padding=3)
    ax.set_xlabel("% registros")
    ax.set_title("Factores de riesgo maternos")

    # ── Panel I: Tipo de parto ────────────────────────────────────────────────
    ax = fig.add_subplot(gs[2, 2])
    if "Deliv_type" in orig.columns:
        dt_map  = {1.0: "Vaginal espontáneo", 2.0: "Forceps/Vacuum", 3.0: "Cesárea"}
        dt_vals = orig["Deliv_type"].map(dt_map).fillna("Otro")
        counts  = dt_vals.value_counts()
        ax.pie(counts.values, labels=counts.index,
               autopct="%1.1f%%", startangle=90,
               colors=["#A5D6A7", "#FFF176", "#EF9A9A"])
        ax.set_title("Tipo de parto (originales)")

    out_path = REPORT_DIR / "04_eda.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Figura guardada → {out_path}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    print("Cargando dataset_full.csv...")
    df   = pd.read_csv(FULL_CSV)
    orig = df[df["is_synthetic"] == False].copy()
    synt = df[df["is_synthetic"] == True].copy()

    section_dataset(df, orig, synt)
    stats = section_outcomes(orig)
    orig_with_asf = section_asphyxia(orig)
    section_risk_factors(orig)
    section_comparison(orig, synt)
    make_figures(orig, synt)

    # Guardar resumen numérico
    summary = {
        "n_original"         : int(len(orig)),
        "n_synthetic"        : int(len(synt)),
        "n_total"            : int(len(df)),
        "n_asphyxia_acog"    : int(orig_with_asf["asphyxia_acog"].sum()),
        "pct_asphyxia_acog"  : round(orig_with_asf["asphyxia_acog"].mean() * 100, 2),
        "pct_apgar1_lt7"     : round((orig["Apgar1"] < 7).mean() * 100, 2),
        "pct_apgar5_lt7"     : round((orig["Apgar5"] < 7).mean() * 100, 2),
        "pct_ph_lt7"         : round((orig["pH"] < 7.00).mean() * 100, 2),
        "pct_ph_lt710"       : round((orig["pH"] < 7.10).mean() * 100, 2),
        "outcome_stats"      : stats,
    }
    out_json = REPORT_DIR / "04_summary.json"
    with open(out_json, "w") as f:
        import json
        json.dump(summary, f, indent=2)
    print(f"\n  Resumen JSON → {out_json}")
    print("\nAnálisis completo.")


if __name__ == "__main__":
    main()
