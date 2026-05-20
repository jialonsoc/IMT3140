"""
03_extract_metadata_and_bootstrap.py
=====================================
Pipeline integrado que:
1. Extrae TODA la metadata clínica de los .hea de PhysioNet (552 registros)
2. Limpia las señales FHR (elimina valores fisiológicamente imposibles)
3. Genera series sintéticas via stationary bootstrap
4. Produce metadata_synthetic.csv con trazabilidad completa:
   - ID sintético único
   - ID del registro origen
   - Metadata clínica HEREDADA del origen (pH, Apgar, etc.)
   - Estadísticas de señal RECALCULADAS desde la serie sintética
   - Columna 'is_synthetic' para distinguir originales de sintéticas

Uso:
    python 03_extract_metadata_and_bootstrap.py
"""

import os
import re
import json
import numpy as np
import pandas as pd
import wfdb
from pathlib import Path
from tqdm import tqdm
from scipy.stats import ks_2samp, wasserstein_distance

# ── Configuración ──────────────────────────────────────────────────────────────
PHYSIONET_DB  = "ctu-uhb-ctgdb"
RAW_DIR       = "data/raw"
SYNTH_DIR     = "data/synthetic"
DATA_DIR      = "data"
FS            = 4
N_MULTIPLIER  = 2          # generar N_MULTIPLIER × 552 series sintéticas
MEAN_BLOCK    = 240        # longitud media de bloque stationary (60 s a 4 Hz)
FHR_MIN       = 60.0       # límite fisiológico inferior (bpm)
FHR_MAX       = 200.0      # límite fisiológico superior (bpm)

os.makedirs(SYNTH_DIR, exist_ok=True)
os.makedirs(DATA_DIR,  exist_ok=True)

# ══════════════════════════════════════════════════════════════════════════════
# 1. PARSER DE METADATA CLÍNICA DESDE .HEA
# ══════════════════════════════════════════════════════════════════════════════

# Mapa: nombre en el header → nombre de columna limpio
FIELD_MAP = {
    # Outcome measures
    "ph"          : "pH",
    "bdecf"       : "BDecf",
    "pco2"        : "pCO2",
    "be"          : "BE",
    "apgar1"      : "Apgar1",
    "apgar5"      : "Apgar5",
    # Neonatology
    "nicu days"   : "NICU_days",
    "seizures"    : "Seizures",
    "hie"         : "HIE",
    "intubation"  : "Intubation",
    "main diag."  : "Main_diag",
    "other diag." : "Other_diag",
    # Fetus/Neonate
    "gest. weeks" : "Gest_weeks",
    "weight(g)"   : "Weight_g",
    "sex"         : "Sex",
    # Maternal
    "age"         : "Age",
    "gravidity"   : "Gravidity",
    "parity"      : "Parity",
    "diabetes"    : "Diabetes",
    "hypertension": "Hypertension",
    "preeclampsia": "Preeclampsia",
    "liq. praecox": "Liq_praecox",
    "pyrexia"     : "Pyrexia",
    "meconium"    : "Meconium",
    # Delivery
    "presentation": "Presentation",
    "induced"     : "Induced",
    "i.stage"     : "I_stage",
    "noprogress"  : "NoProgress",
    "ck/kp"       : "CK_KP",
    "ii.stage"    : "II_stage",
    "deliv. type" : "Deliv_type",
    # Signal info
    "dbid"        : "dbID",
    "rec. type"   : "Rec_type",
    "pos. ii.st." : "Pos_IIst",
    "sig2birth"   : "Sig2Birth",
}

def parse_header_comments(comments: list[str]) -> dict:
    """Parsea los comentarios del .hea y retorna dict con todos los campos clínicos."""
    result = {}
    for line in comments:
        line = line.strip()
        # Ignorar líneas de sección (empiezan con --)
        if line.startswith("--") or not line:
            continue
        # Formato: "Campo    valor"
        # Separar por 2+ espacios o tab
        parts = re.split(r'\s{2,}|\t', line, maxsplit=1)
        if len(parts) == 2:
            key_raw = parts[0].strip().lower()
            val_raw = parts[1].strip()
            # Buscar en el mapa de campos
            col_name = FIELD_MAP.get(key_raw)
            if col_name:
                # Convertir a número si es posible
                try:
                    result[col_name] = float(val_raw)
                except ValueError:
                    result[col_name] = val_raw
    return result


def extract_all_clinical_metadata(records: list[str]) -> pd.DataFrame:
    """Extrae metadata clínica de todos los registros via wfdb."""
    print(f"\nExtrayendo metadata clínica de {len(records)} registros...")
    rows = []
    failed = []

    for rec in tqdm(records, desc="Leyendo headers .hea"):
        try:
            header = wfdb.rdheader(rec, pn_dir=PHYSIONET_DB)
            clinical = parse_header_comments(header.comments)
            clinical["record"] = rec
            rows.append(clinical)
        except Exception as e:
            failed.append({"record": rec, "error": str(e)})

    if failed:
        print(f"  [AVISO] {len(failed)} registros fallaron en extracción de header")

    df = pd.DataFrame(rows)
    # Poner 'record' como primera columna
    cols = ["record"] + [c for c in df.columns if c != "record"]
    df = df[cols]
    return df


# ══════════════════════════════════════════════════════════════════════════════
# 2. CARGA Y LIMPIEZA DE SEÑALES FHR
# ══════════════════════════════════════════════════════════════════════════════

def clean_fhr(x: np.ndarray,
              fhr_min: float = FHR_MIN,
              fhr_max: float = FHR_MAX) -> np.ndarray:
    """
    Limpieza fisiológica de la señal FHR:
    1. Marca como NaN valores fuera del rango fisiológico
    2. Interpola linealmente los NaNs
    """
    arr = x.copy().astype(np.float32)
    # Marcar outliers fisiológicos
    arr[(arr < fhr_min) | (arr > fhr_max)] = np.nan
    # Interpolar
    nans = np.isnan(arr)
    if nans.all():
        return np.full_like(arr, 140.0)
    if not nans.any():
        return arr
    idx   = np.arange(len(arr))
    valid = ~nans
    arr[nans] = np.interp(idx[nans], idx[valid], arr[valid])
    return arr


def load_fhr_series(raw_dir: str) -> dict[str, np.ndarray]:
    """Carga todas las series FHR limpias. Retorna {record_id: array}."""
    files = sorted(Path(raw_dir).glob("*_fhr.npy"))
    series = {}
    for f in files:
        rec_id = f.stem.replace("_fhr", "")
        arr    = np.load(f).astype(np.float32)
        # Mantener solo si tiene suficientes datos válidos
        valid_pct = np.mean((arr >= FHR_MIN) & (arr <= FHR_MAX))
        if valid_pct > 0.3:
            series[rec_id] = clean_fhr(arr)
    print(f"Cargadas {len(series)} series FHR válidas desde '{raw_dir}'")
    return series


# ══════════════════════════════════════════════════════════════════════════════
# 3. ESTADÍSTICAS DE SEÑAL (para metadata recalculada)
# ══════════════════════════════════════════════════════════════════════════════

def signal_stats(arr: np.ndarray, fs: int = FS) -> dict:
    """Calcula estadísticas de señal para la metadata de una serie."""
    clean = arr[~np.isnan(arr)]
    clean = clean[(clean >= FHR_MIN) & (clean <= FHR_MAX)]
    if len(clean) == 0:
        clean = np.array([140.0])
    return {
        "n_samples"   : len(arr),
        "duration_min": round(len(arr) / fs / 60, 2),
        "fs_hz"       : fs,
        "fhr_mean"    : round(float(np.mean(clean)), 3),
        "fhr_std"     : round(float(np.std(clean)),  3),
        "fhr_min"     : round(float(np.min(clean)),  3),
        "fhr_max"     : round(float(np.max(clean)),  3),
        "nan_pct"     : round(float(np.mean(np.isnan(arr))) * 100, 2),
    }


# ══════════════════════════════════════════════════════════════════════════════
# 4. STATIONARY BOOTSTRAP (con trazabilidad)
# ══════════════════════════════════════════════════════════════════════════════

def stationary_bootstrap(series: np.ndarray,
                          target_len: int,
                          mean_block: int = MEAN_BLOCK) -> np.ndarray:
    """Stationary Bootstrap (Politis & Romano 1994)."""
    n   = len(series)
    p   = 1.0 / mean_block
    out = []
    pos = np.random.randint(0, n)
    while len(out) < target_len:
        out.append(series[pos % n])
        pos = np.random.randint(0, n) if np.random.rand() < p else pos + 1
    return np.array(out[:target_len], dtype=np.float32)


def generate_synthetic_with_metadata(
        series_dict   : dict[str, np.ndarray],
        clinical_df   : pd.DataFrame,
        n_multiplier  : int = N_MULTIPLIER,
        mean_block    : int = MEAN_BLOCK,
) -> tuple[list[dict], list[np.ndarray]]:
    """
    Genera series sintéticas con metadata completa y trazable.

    Por cada registro origen genera n_multiplier series sintéticas.
    Cada sintética hereda TODA la metadata clínica del origen y
    recalcula las estadísticas de señal desde la nueva serie.

    Retorna:
        meta_rows : lista de dicts (una fila por serie sintética)
        syn_series: lista de arrays numpy
    """
    # Indexar metadata clínica por record
    clinical_idx = clinical_df.set_index("record").to_dict(orient="index")

    meta_rows  = []
    syn_series = []
    counter    = 0

    rec_ids = sorted(series_dict.keys())

    for rec_id in tqdm(rec_ids, desc="Generando sintéticas"):
        src      = series_dict[rec_id]
        clinical = clinical_idx.get(rec_id, {})

        for i in range(n_multiplier):
            syn = stationary_bootstrap(src, len(src), mean_block)

            # ── ID único y trazabilidad ───────────────────────────────────────
            syn_id = f"syn_{counter:05d}"

            # ── Metadata de la fila ───────────────────────────────────────────
            row = {
                # Identificación
                "record"       : syn_id,
                "source_record": rec_id,
                "bootstrap_idx": i,
                "is_synthetic" : True,

                # Metadata clínica HEREDADA del registro origen
                **clinical,

                # Estadísticas de señal RECALCULADAS desde la serie sintética
                **signal_stats(syn),
            }
            meta_rows.append(row)
            syn_series.append(syn)
            counter += 1

    return meta_rows, syn_series


# ══════════════════════════════════════════════════════════════════════════════
# 5. CONSTRUIR DATASET COMPLETO (originales + sintéticas)
# ══════════════════════════════════════════════════════════════════════════════

def build_full_dataset(
        series_dict : dict[str, np.ndarray],
        clinical_df : pd.DataFrame,
        syn_meta    : list[dict],
) -> pd.DataFrame:
    """
    Combina metadata de originales y sintéticas en un único DataFrame.
    Agrega columna 'is_synthetic' a los originales (False).
    """
    clinical_idx = clinical_df.set_index("record").to_dict(orient="index")

    orig_rows = []
    for rec_id, arr in series_dict.items():
        clinical = clinical_idx.get(rec_id, {})
        row = {
            "record"       : rec_id,
            "source_record": rec_id,
            "bootstrap_idx": 0,
            "is_synthetic" : False,
            **clinical,
            **signal_stats(arr),
        }
        orig_rows.append(row)

    df_orig = pd.DataFrame(orig_rows)
    df_synt = pd.DataFrame(syn_meta)
    df_full = pd.concat([df_orig, df_synt], ignore_index=True)

    return df_full


# ══════════════════════════════════════════════════════════════════════════════
# 6. VALIDACIÓN DE CONCORDANCIA METADATA ↔ SEÑAL
# ══════════════════════════════════════════════════════════════════════════════

def validate_metadata_concordance(df_full: pd.DataFrame) -> dict:
    """
    Verifica que las estadísticas de señal de las sintéticas son
    coherentes con las del registro origen (fhr_mean, std dentro de rangos).
    """
    print("\nValidando concordancia metadata ↔ señal...")

    orig = df_full[~df_full["is_synthetic"]]
    synt = df_full[ df_full["is_synthetic"]]

    # Para cada sintética, comparar fhr_mean con la del origen
    synt_merged = synt.merge(
        orig[["record", "fhr_mean", "fhr_std"]].rename(
            columns={"record": "source_record",
                     "fhr_mean": "orig_fhr_mean",
                     "fhr_std" : "orig_fhr_std"}),
        on="source_record", how="left"
    )

    diff_mean = (synt_merged["fhr_mean"] - synt_merged["orig_fhr_mean"]).abs()
    diff_std  = (synt_merged["fhr_std"]  - synt_merged["orig_fhr_std"]).abs()

    results = {
        "n_original"             : int(len(orig)),
        "n_synthetic"            : int(len(synt)),
        "n_total"                : int(len(df_full)),
        "fhr_mean_diff_mean"     : round(float(diff_mean.mean()), 3),
        "fhr_mean_diff_max"      : round(float(diff_mean.max()),  3),
        "fhr_std_diff_mean"      : round(float(diff_std.mean()),  3),
        "fhr_std_diff_max"       : round(float(diff_std.max()),   3),
        "pct_within_5bpm"        : round(float((diff_mean < 5).mean()  * 100), 1),
        "pct_within_10bpm"       : round(float((diff_mean < 10).mean() * 100), 1),
        # Verificar que pH se heredó correctamente (no debe haber variación)
        "pH_unique_per_source"   : int(
            synt.groupby("source_record")["pH"].nunique().max()
        ) if "pH" in synt.columns else None,
    }

    print(f"  Originales:  {results['n_original']}")
    print(f"  Sintéticas:  {results['n_synthetic']}")
    print(f"  Total:       {results['n_total']}")
    print(f"  FHR mean diff promedio: {results['fhr_mean_diff_mean']} bpm")
    print(f"  FHR mean diff máximo:   {results['fhr_mean_diff_max']} bpm")
    print(f"  % sintéticas dentro de ±5 bpm del origen:  {results['pct_within_5bpm']}%")
    print(f"  % sintéticas dentro de ±10 bpm del origen: {results['pct_within_10bpm']}%")
    if results["pH_unique_per_source"] is not None:
        ok = results["pH_unique_per_source"] == 1
        print(f"  pH heredado correctamente: {'✅ Sí' if ok else '❌ No'}")

    return results


# ══════════════════════════════════════════════════════════════════════════════
# 7. DISTRIBUCIÓN DE pH EN ORIGINALES VS SINTÉTICAS
# ══════════════════════════════════════════════════════════════════════════════

def ph_distribution_report(df_full: pd.DataFrame) -> pd.DataFrame:
    """Muestra la distribución de clases de pH en originales y sintéticas."""
    if "pH" not in df_full.columns:
        print("Columna 'pH' no disponible")
        return pd.DataFrame()

    def classify_ph(ph):
        if pd.isna(ph):   return "Sin dato"
        if ph < 7.05:     return "Asfixia severa (pH<7.05)"
        if ph < 7.10:     return "Acidosis moderada (7.05–7.10)"
        if ph < 7.20:     return "Acidosis leve (7.10–7.20)"
        return "Normal (pH≥7.20)"

    df_full = df_full.copy()
    df_full["pH_class"] = df_full["pH"].apply(classify_ph)

    print("\n── Distribución de clases de pH ──────────────────────────────")
    for group_name, group in [("ORIGINALES", df_full[~df_full["is_synthetic"]]),
                               ("SINTÉTICAS", df_full[ df_full["is_synthetic"]])]:
        counts = group["pH_class"].value_counts()
        total  = len(group)
        print(f"\n  {group_name} (n={total}):")
        for cls, cnt in counts.items():
            print(f"    {cls}: {cnt} ({cnt/total*100:.1f}%)")

    return df_full


# ══════════════════════════════════════════════════════════════════════════════
# 8. MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main():
    np.random.seed(42)

    # ── Paso 1: Obtener lista de registros ────────────────────────────────────
    print("Obteniendo lista de registros desde PhysioNet...")
    records = wfdb.get_record_list(PHYSIONET_DB)
    print(f"  → {len(records)} registros")

    # ── Paso 2: Extraer metadata clínica de todos los .hea ────────────────────
    df_clinical = extract_all_clinical_metadata(records)
    df_clinical.to_csv(f"{DATA_DIR}/clinical_metadata.csv", index=False)
    print(f"\nMetadata clínica guardada → {DATA_DIR}/clinical_metadata.csv")
    print(f"Columnas disponibles: {list(df_clinical.columns)}")

    # ── Paso 3: Cargar y limpiar señales FHR ──────────────────────────────────
    series_dict = load_fhr_series(RAW_DIR)

    # Verificar que los record IDs del CSV coinciden con los .npy
    clinical_records = set(df_clinical["record"].astype(str))
    signal_records   = set(series_dict.keys())
    match = clinical_records & signal_records
    print(f"\nRegistros con señal Y metadata clínica: {len(match)}/{len(records)}")

    # Filtrar series_dict a solo los que tienen metadata
    series_dict = {k: v for k, v in series_dict.items() if k in clinical_records}

    # ── Paso 4: Generar sintéticas con metadata trazable ──────────────────────
    print(f"\nGenerando {N_MULTIPLIER}× {len(series_dict)} = "
          f"{N_MULTIPLIER * len(series_dict)} series sintéticas...")

    syn_meta, syn_series = generate_synthetic_with_metadata(
        series_dict  = series_dict,
        clinical_df  = df_clinical,
        n_multiplier = N_MULTIPLIER,
        mean_block   = MEAN_BLOCK,
    )

    # ── Paso 5: Guardar series sintéticas ─────────────────────────────────────
    print("\nGuardando series sintéticas...")
    for i, (meta, arr) in enumerate(tqdm(zip(syn_meta, syn_series),
                                          total=len(syn_series),
                                          desc="Guardando .npy")):
        syn_id = meta["record"]
        np.save(f"{SYNTH_DIR}/{syn_id}_fhr.npy", arr)

    # ── Paso 6: Construir dataset completo ────────────────────────────────────
    df_full = build_full_dataset(series_dict, df_clinical, syn_meta)

    # Reordenar columnas: identidad → clínica → señal
    id_cols     = ["record", "source_record", "bootstrap_idx", "is_synthetic"]
    signal_cols = ["n_samples", "duration_min", "fs_hz",
                   "fhr_mean", "fhr_std", "fhr_min", "fhr_max", "nan_pct"]
    clinical_cols = [c for c in df_full.columns
                     if c not in id_cols + signal_cols]
    final_cols  = id_cols + clinical_cols + signal_cols
    final_cols  = [c for c in final_cols if c in df_full.columns]
    df_full     = df_full[final_cols]

    # Guardar CSVs
    df_full.to_csv(f"{DATA_DIR}/dataset_full.csv", index=False)
    df_full[~df_full["is_synthetic"]].to_csv(
        f"{DATA_DIR}/dataset_original.csv", index=False)
    df_full[ df_full["is_synthetic"]].to_csv(
        f"{DATA_DIR}/dataset_synthetic.csv", index=False)

    print(f"\nDatasets guardados:")
    print(f"  → {DATA_DIR}/dataset_full.csv      ({len(df_full)} filas)")
    print(f"  → {DATA_DIR}/dataset_original.csv  ({(~df_full['is_synthetic']).sum()} filas)")
    print(f"  → {DATA_DIR}/dataset_synthetic.csv ({df_full['is_synthetic'].sum()} filas)")

    # ── Paso 7: Validar concordancia ──────────────────────────────────────────
    validation = validate_metadata_concordance(df_full)
    with open(f"{DATA_DIR}/validation_concordance.json", "w") as f:
        json.dump(validation, f, indent=2)

    # ── Paso 8: Distribución de pH ────────────────────────────────────────────
    df_full = ph_distribution_report(df_full)

    # ── Resumen final ─────────────────────────────────────────────────────────
    print("\n" + "="*60)
    print("RESUMEN FINAL")
    print("="*60)
    print(f"  Registros originales con señal+metadata: {len(series_dict)}")
    print(f"  Series sintéticas generadas:             {len(syn_series)}")
    print(f"  Total dataset:                           {len(df_full)}")
    print(f"  Columnas de metadata:                    {len(df_full.columns)}")
    print(f"\nEstructura de dataset_full.csv:")
    print(df_full.head(3).to_string())
    print("\n✓ Pipeline completo.")


if __name__ == "__main__":
    main()