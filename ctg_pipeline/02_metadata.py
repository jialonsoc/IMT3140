"""
02_metadata.py
--------------
Construye dataset_original.csv combinando:
  1. Metadata clínica extraída de los headers .hea (PhysioNet)
  2. Estadísticas de señal calculadas sobre FHR y UC ya limpias

El CSV resultante tiene una fila por registro y es la fuente de entrada
del paso 03 (bootstrap). No modifica ni genera series de tiempo.

Ejecutar después de 01_download.py:
    cd ctg_pipeline
    python 02_metadata.py
"""

import re
import numpy as np
import pandas as pd
import wfdb
from tqdm import tqdm
from config import PHYSIONET_DB, RAW_DIR, ORIGINAL_CSV, DATA_DIR
from utils.signal import clean_fhr, clean_uc, signal_stats, nan_summary

DATA_DIR.mkdir(parents=True, exist_ok=True)

# ── Mapa de campos del header .hea → nombre de columna ────────────────────────
# El header tiene líneas como "pH    7.14", "Apgar1    6", etc.
FIELD_MAP = {
    # Outcomes neonatales
    "ph"          : "pH",
    "bdecf"       : "BDecf",
    "pco2"        : "pCO2",
    "be"          : "BE",
    "apgar1"      : "Apgar1",
    "apgar5"      : "Apgar5",
    # Neonatología
    "nicu days"   : "NICU_days",
    "seizures"    : "Seizures",
    "hie"         : "HIE",
    "intubation"  : "Intubation",
    "main diag."  : "Main_diag",
    "other diag." : "Other_diag",
    # Feto / neonato
    "gest. weeks" : "Gest_weeks",
    "weight(g)"   : "Weight_g",
    "sex"         : "Sex",
    # Factores maternos
    "age"         : "Age",
    "gravidity"   : "Gravidity",
    "parity"      : "Parity",
    "diabetes"    : "Diabetes",
    "hypertension": "Hypertension",
    "preeclampsia": "Preeclampsia",
    "liq. praecox": "Liq_praecox",
    "pyrexia"     : "Pyrexia",
    "meconium"    : "Meconium",
    # Parto
    "presentation": "Presentation",
    "induced"     : "Induced",
    "i.stage"     : "I_stage",
    "noprogress"  : "NoProgress",
    "ck/kp"       : "CK_KP",
    "ii.stage"    : "II_stage",
    "deliv. type" : "Deliv_type",
    # Info de señal (del header, no recalculada)
    "dbid"        : "dbID",
    "rec. type"   : "Rec_type",
    "pos. ii.st." : "Pos_IIst",
    "sig2birth"   : "Sig2Birth",
}


def parse_header(rec_id: str) -> dict:
    """Extrae los campos clínicos del header .hea de un registro."""
    try:
        header = wfdb.rdheader(rec_id, pn_dir=PHYSIONET_DB)
    except Exception as e:
        return {"record": rec_id, "_parse_error": str(e)}

    result = {"record": rec_id}
    for line in header.comments:
        line = line.strip()
        if line.startswith("--") or not line:
            continue
        # Separar por 2+ espacios o tabulador
        parts = re.split(r"\s{2,}|\t", line, maxsplit=1)
        if len(parts) != 2:
            continue
        key = parts[0].strip().lower()
        col = FIELD_MAP.get(key)
        if col is None:
            continue
        try:
            result[col] = float(parts[1].strip())
        except ValueError:
            result[col] = parts[1].strip()
    return result


def signal_row(rec_id: str) -> dict:
    """
    Carga FHR y UC, aplica limpieza, calcula estadísticas.
    Retorna dict con columnas prefijadas 'fhr_*' y 'uc_*'.
    """
    fhr_path = RAW_DIR / f"{rec_id}_fhr.npy"
    uc_path  = RAW_DIR / f"{rec_id}_uc.npy"

    if not fhr_path.exists():
        return {"_signal_error": "fhr_no_encontrado"}

    fhr_raw = np.load(fhr_path)
    fhr     = clean_fhr(fhr_raw)

    row = {}
    row.update(signal_stats(fhr, label="fhr"))

    if uc_path.exists():
        uc_raw = np.load(uc_path)
        uc     = clean_uc(uc_raw)
        uc_nan = nan_summary(uc_raw)
        row["uc_mean"]    = round(float(np.nanmean(uc)), 3) if not np.all(np.isnan(uc)) else np.nan
        row["uc_std"]     = round(float(np.nanstd(uc)),  3) if not np.all(np.isnan(uc)) else np.nan
        row["uc_min"]     = round(float(np.nanmin(uc)),  3) if not np.all(np.isnan(uc)) else np.nan
        row["uc_max"]     = round(float(np.nanmax(uc)),  3) if not np.all(np.isnan(uc)) else np.nan
        row["uc_nan_pct"] = round(uc_nan["nan_pct"], 2)
        row["has_uc"]     = not np.all(np.isnan(uc))
    else:
        row.update({"uc_mean": np.nan, "uc_std": np.nan,
                    "uc_min": np.nan,  "uc_max": np.nan,
                    "uc_nan_pct": 100.0, "has_uc": False})
    return row


def main():
    print(f"Obteniendo lista de registros de '{PHYSIONET_DB}'...")
    records = wfdb.get_record_list(PHYSIONET_DB)
    print(f"  → {len(records)} registros\n")

    rows = []
    for rec_id in tqdm(records, desc="Extrayendo metadata"):
        clinical = parse_header(rec_id)
        signal   = signal_row(rec_id)
        row = {**clinical, **signal}
        rows.append(row)

    df = pd.DataFrame(rows)

    # Orden de columnas: identidad → clínica → señal FHR → señal UC
    id_cols  = ["record"]
    fhr_cols = ["n_samples", "duration_min", "fs_hz",
                "fhr_mean", "fhr_std", "fhr_min", "fhr_max", "fhr_nan_pct"]
    uc_cols  = ["has_uc", "uc_mean", "uc_std", "uc_min", "uc_max", "uc_nan_pct"]
    clin_cols = [c for c in df.columns
                 if c not in id_cols + fhr_cols + uc_cols
                 and not c.startswith("_")]
    final_order = id_cols + clin_cols + fhr_cols + uc_cols
    final_order = [c for c in final_order if c in df.columns]
    df = df[final_order]

    df.to_csv(ORIGINAL_CSV, index=False)
    print(f"\nDataset original guardado → {ORIGINAL_CSV}")
    print(f"  Filas: {len(df)}  |  Columnas: {len(df.columns)}")
    print(f"  Registros con UC válida: {df['has_uc'].sum()}")
    print(f"  Registros con error de señal: {df['_signal_error'].notna().sum()}"
          if "_signal_error" in df.columns else "")
    print(f"\nPrimeras 3 filas:")
    print(df[["record", "pH", "Apgar1", "Apgar5",
              "fhr_mean", "fhr_nan_pct", "has_uc"]].head(3).to_string(index=False))


if __name__ == "__main__":
    main()
