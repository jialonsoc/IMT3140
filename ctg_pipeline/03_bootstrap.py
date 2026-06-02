"""
03_bootstrap.py
---------------
Bootstrap multivariado sincronizado FHR+UC sobre los registros de
dataset_original.csv. Produce dataset_synthetic.csv y dataset_full.csv.

Flujo
-----
  1. Leer dataset_original.csv  (fuente de verdad — D-05)
  2. Filtrar registros por calidad de señal  (D-06)
  3. Calcular b* óptimo por registro  (D-03)
  4. Determinar N_MULTIPLIER por convergencia del estimador  (D-04)
  5. Generar réplicas con stationary bootstrap sincronizado  (D-02)
  6. Heredar metadata clínica + recalcular stats de señal
  7. Guardar .npy y CSVs

Ejecutar después de 02_metadata.py:
    cd ctg_pipeline
    python 03_bootstrap.py
"""

import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm

from config import (
    RAW_DIR, SYNTH_DIR, ORIGINAL_CSV, SYNTHETIC_CSV, FULL_CSV,
    FS, NAN_PCT_MAX, DURATION_MIN,
    N_MULTIPLIER_MAX, CONVERGENCE_THR, SEED,
)
from utils.signal import clean_fhr, clean_uc, signal_stats
from utils.bootstrap import stationary_bootstrap, optimal_block_length

np.random.seed(SEED)
SYNTH_DIR.mkdir(parents=True, exist_ok=True)


# ── 1. Cargar y filtrar dataset_original.csv ──────────────────────────────────

def load_eligible(csv_path: Path) -> pd.DataFrame:
    """
    Lee dataset_original.csv y aplica los filtros de calidad (D-06):
      - fhr_nan_pct ≤ NAN_PCT_MAX
      - duration_min ≥ DURATION_MIN
      - has_uc == True  (necesitamos UC para el bootstrap sincronizado)
    """
    df = pd.read_csv(csv_path)
    n_total = len(df)

    mask = (
        (df["fhr_nan_pct"] <= NAN_PCT_MAX) &
        (df["duration_min"] >= DURATION_MIN) &
        (df["has_uc"] == True)
    )
    df_ok = df[mask].reset_index(drop=True)

    print(f"Registros totales:           {n_total}")
    print(f"  fhr_nan_pct > {NAN_PCT_MAX}%:    {(df['fhr_nan_pct'] > NAN_PCT_MAX).sum()}")
    print(f"  duration_min < {DURATION_MIN} min:  {(df['duration_min'] < DURATION_MIN).sum()}")
    print(f"  sin UC válida:               {(df['has_uc'] != True).sum()}")
    print(f"Registros elegibles:         {len(df_ok)}\n")

    return df_ok


# ── 2. Cargar señales limpias para un registro ────────────────────────────────

def load_pair(rec_id: str) -> tuple[np.ndarray, np.ndarray] | None:
    """
    Carga y limpia el par (FHR, UC) de un registro.
    Retorna None si el archivo no existe o tiene demasiados NaN residuales.
    """
    fhr_path = RAW_DIR / f"{rec_id}_fhr.npy"
    uc_path  = RAW_DIR / f"{rec_id}_uc.npy"

    if not fhr_path.exists() or not uc_path.exists():
        return None

    fhr = clean_fhr(np.load(fhr_path))
    uc  = clean_uc(np.load(uc_path))

    # Ambas deben tener la misma longitud (garantizado por la descarga)
    n = min(len(fhr), len(uc))
    return fhr[:n], uc[:n]


# ── 3. Convergencia del estimador para N_MULTIPLIER ──────────────────────────

def find_n_multiplier(df: pd.DataFrame,
                      series_cache: dict[str, tuple]) -> int:
    """
    Determina N_MULTIPLIER por convergencia del estimador de fhr_mean
    sobre el pool sintético acumulado (D-04).

    Estrategia:
      - Para m = 1, 2, ..., N_MULTIPLIER_MAX:
          Generar 1 réplica adicional por cada registro y calcular
          la std del fhr_mean acumulado en el pool.
      - Detener cuando |Δstd| < CONVERGENCE_THR entre dos iteraciones.

    Se usa fhr_mean (estadístico de señal) como estimador porque es
    el único parámetro que varía entre réplicas del mismo registro;
    la metadata clínica (Apgar, pH) es constante por herencia.
    """
    print("Calculando N_MULTIPLIER óptimo por convergencia del estimador...")
    rec_ids = df["record"].astype(str).tolist()

    pool_means = []   # fhr_mean de cada réplica generada
    prev_std   = None
    chosen_n   = N_MULTIPLIER_MAX

    for m in range(1, N_MULTIPLIER_MAX + 1):
        batch_means = []
        for rec_id in rec_ids:
            pair = series_cache.get(rec_id)
            if pair is None:
                continue
            fhr, uc = pair
            b_star = optimal_block_length(fhr)
            fhr_syn, _ = stationary_bootstrap(fhr, uc, b_star)
            # Estadístico: media de FHR sintética (ignora NaN residuales)
            batch_means.append(float(np.nanmean(fhr_syn)))

        pool_means.extend(batch_means)
        curr_std = float(np.std(pool_means))

        delta = abs(curr_std - prev_std) if prev_std is not None else float("inf")
        print(f"  m={m}  pool_size={len(pool_means):4d}  "
              f"std(fhr_mean)={curr_std:.4f}  Δstd={delta:.4f}")

        if prev_std is not None and delta < CONVERGENCE_THR:
            chosen_n = m
            print(f"  → Convergencia alcanzada en m={m} (Δstd < {CONVERGENCE_THR})\n")
            break
        prev_std = curr_std
    else:
        print(f"  → No convergió en {N_MULTIPLIER_MAX} iteraciones. "
              f"Se usa N_MULTIPLIER = {N_MULTIPLIER_MAX}\n")

    return chosen_n


# ── 4. Generación de sintéticas ───────────────────────────────────────────────

def build_row(syn_id: str,
              rec_id: str,
              bootstrap_idx: int,
              fhr_syn: np.ndarray,
              uc_syn: np.ndarray,
              clinical_row: pd.Series,
              b_star: int) -> dict:
    """Construye la fila de metadata para una serie sintética."""
    fhr_stats = signal_stats(fhr_syn, label="fhr")
    uc_stats  = {
        "uc_mean"   : round(float(np.nanmean(uc_syn)), 3),
        "uc_std"    : round(float(np.nanstd(uc_syn)),  3),
        "uc_min"    : round(float(np.nanmin(uc_syn)),  3),
        "uc_max"    : round(float(np.nanmax(uc_syn)),  3),
        "uc_nan_pct": round(float(np.isnan(uc_syn).mean() * 100), 2),
    }

    # Correlación FHR↔UC en la réplica (validación de coherencia fisiológica)
    valid = ~(np.isnan(fhr_syn) | np.isnan(uc_syn))
    f_v, u_v = fhr_syn[valid], uc_syn[valid]
    if valid.sum() > 10 and f_v.std() > 0 and u_v.std() > 0:
        fhr_uc_corr = float(np.corrcoef(f_v, u_v)[0, 1])
    else:
        fhr_uc_corr = np.nan

    row = {
        "record"        : syn_id,
        "source_record" : rec_id,
        "bootstrap_idx" : bootstrap_idx,
        "is_synthetic"  : True,
        "block_size_used": b_star,
        "fhr_uc_corr"   : round(fhr_uc_corr, 4),
        # Metadata clínica heredada del registro origen
        **{col: clinical_row[col]
           for col in clinical_row.index
           if col not in ("record", "has_uc", "n_samples", "duration_min",
                          "fs_hz", "fhr_mean", "fhr_std", "fhr_min", "fhr_max",
                          "fhr_nan_pct", "uc_mean", "uc_std", "uc_min",
                          "uc_max", "uc_nan_pct")},
        # Stats de señal recalculadas desde la serie sintética
        **fhr_stats,
        **uc_stats,
        "has_uc"        : True,
    }
    return row


def generate_synthetics(df: pd.DataFrame,
                         series_cache: dict,
                         n_multiplier: int) -> tuple[list[dict], int]:
    """
    Genera n_multiplier réplicas por cada registro elegible.
    Retorna (lista de filas metadata, contador total de sintéticas).
    """
    rows   = []
    counter = 0

    for _, orig_row in tqdm(df.iterrows(), total=len(df),
                             desc=f"Bootstrap (N={n_multiplier}×)"):
        rec_id = str(orig_row["record"])
        pair   = series_cache.get(rec_id)
        if pair is None:
            continue

        fhr, uc = pair
        b_star  = optimal_block_length(fhr)

        for i in range(n_multiplier):
            fhr_syn, uc_syn = stationary_bootstrap(fhr, uc, b_star)

            syn_id = f"syn_{counter:05d}"
            np.save(SYNTH_DIR / f"{syn_id}_fhr.npy", fhr_syn)
            np.save(SYNTH_DIR / f"{syn_id}_uc.npy",  uc_syn)

            row = build_row(syn_id, rec_id, i, fhr_syn, uc_syn, orig_row, b_star)
            rows.append(row)
            counter += 1

    return rows, counter


# ── 5. Main ───────────────────────────────────────────────────────────────────

def main():
    # Paso 1: filtrar registros elegibles
    df = load_eligible(ORIGINAL_CSV)

    # Paso 2: cargar todas las señales en memoria (evita leer disco N veces)
    print("Cargando señales en memoria...")
    series_cache = {}
    for rec_id in tqdm(df["record"].astype(str), desc="Cargando pares FHR+UC"):
        pair = load_pair(rec_id)
        if pair is not None:
            series_cache[rec_id] = pair
    print(f"Señales cargadas: {len(series_cache)}\n")

    # Paso 3: determinar N_MULTIPLIER por convergencia
    n_multiplier = find_n_multiplier(df, series_cache)

    # Paso 4: generar todas las sintéticas
    print(f"Generando {n_multiplier} × {len(series_cache)} series sintéticas...")
    syn_rows, total = generate_synthetics(df, series_cache, n_multiplier)
    print(f"\nSeries sintéticas generadas: {total}")

    # Paso 5: construir y guardar CSVs
    df_syn  = pd.DataFrame(syn_rows)
    df_orig = df.copy()
    df_orig["is_synthetic"]   = False
    df_orig["source_record"]  = df_orig["record"]
    df_orig["bootstrap_idx"]  = 0
    df_orig["block_size_used"] = np.nan
    df_orig["fhr_uc_corr"]    = np.nan

    df_full = pd.concat([df_orig, df_syn], ignore_index=True)

    df_syn.to_csv(SYNTHETIC_CSV, index=False)
    df_full.to_csv(FULL_CSV,     index=False)

    print(f"\nArchivos guardados:")
    print(f"  → {SYNTHETIC_CSV}  ({len(df_syn)} filas)")
    print(f"  → {FULL_CSV}       ({len(df_full)} filas)")
    print(f"\nN_MULTIPLIER usado: {n_multiplier}")
    print(f"Proporción Apgar1 < 7 — originales: "
          f"{(df['Apgar1'] < 7).mean():.3f}  "
          f"sintéticas: {(df_syn['Apgar1'] < 7).mean():.3f}")


if __name__ == "__main__":
    main()
