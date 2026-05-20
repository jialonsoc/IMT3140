"""
01_download_ctg.py
==================
Descarga las 552 grabaciones CTG (CTU-UHB Intrapartum CTG Database)
desde PhysioNet usando wfdb.

Señales disponibles por registro:
  - FHR  : Fetal Heart Rate  (latidos/min, 4 Hz)
  - UC   : Uterine Contractions (mmHg, 4 Hz)

Uso:
    pip install wfdb numpy pandas tqdm
    python 01_download_ctg.py
"""

import os
import json
import numpy as np
import pandas as pd
import wfdb
from tqdm import tqdm

# ── Configuración ──────────────────────────────────────────────────────────────
PHYSIONET_DB  = "ctu-uhb-ctgdb"          # nombre del dataset en PhysioNet
OUTPUT_DIR    = "data/raw"               # carpeta de salida para .npy y metadatos
METADATA_FILE = "data/metadata.csv"      # resumen clínico de cada registro
FS            = 4                        # Hz (frecuencia de muestreo oficial)

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs("data", exist_ok=True)

# ── 1. Listar todos los registros ──────────────────────────────────────────────
print("Obteniendo lista de registros desde PhysioNet...")
records = wfdb.get_record_list(PHYSIONET_DB)
print(f"  → {len(records)} registros encontrados\n")

# ── 2. Descargar y guardar cada serie ─────────────────────────────────────────
meta_rows = []
failed    = []

for rec_name in tqdm(records, desc="Descargando CTGs"):
    try:
        # Leer señal
        record = wfdb.rdrecord(
            rec_name,
            pn_dir=PHYSIONET_DB,
            smooth_frames=True
        )

        # Intentar leer anotaciones de cabecera (metadatos clínicos)
        header = wfdb.rdheader(rec_name, pn_dir=PHYSIONET_DB)

        sig_names = [s.lower() for s in record.sig_name]
        p_signal  = record.p_signal  # shape: (n_samples, n_channels)

        # Extraer FHR y UC según índice de canal
        fhr_idx = next((i for i, n in enumerate(sig_names) if "fhr" in n), None)
        uc_idx  = next((i for i, n in enumerate(sig_names) if "uc"  in n), None)

        fhr = p_signal[:, fhr_idx] if fhr_idx is not None else np.full(len(p_signal), np.nan)
        uc  = p_signal[:, uc_idx]  if uc_idx  is not None else np.full(len(p_signal), np.nan)

        # Guardar como .npy
        np.save(os.path.join(OUTPUT_DIR, f"{rec_name}_fhr.npy"), fhr.astype(np.float32))
        np.save(os.path.join(OUTPUT_DIR, f"{rec_name}_uc.npy"),  uc.astype(np.float32))

        # Metadatos básicos
        duration_min = len(fhr) / FS / 60
        fhr_clean    = fhr[~np.isnan(fhr)]

        meta_rows.append({
            "record"        : rec_name,
            "n_samples"     : len(fhr),
            "duration_min"  : round(duration_min, 2),
            "fs_hz"         : FS,
            "fhr_mean"      : round(float(np.nanmean(fhr)), 2) if len(fhr_clean) else np.nan,
            "fhr_std"       : round(float(np.nanstd(fhr)),  2) if len(fhr_clean) else np.nan,
            "fhr_min"       : round(float(np.nanmin(fhr)),  2) if len(fhr_clean) else np.nan,
            "fhr_max"       : round(float(np.nanmax(fhr)),  2) if len(fhr_clean) else np.nan,
            "nan_pct"       : round(float(np.mean(np.isnan(fhr))) * 100, 2),
            "n_channels"    : record.n_sig,
            "sig_names"     : str(record.sig_name),
        })

    except Exception as e:
        tqdm.write(f"  [ERROR] {rec_name}: {e}")
        failed.append({"record": rec_name, "error": str(e)})

# ── 3. Guardar metadatos ───────────────────────────────────────────────────────
df_meta = pd.DataFrame(meta_rows)
df_meta.to_csv(METADATA_FILE, index=False)
print(f"\nMetadatos guardados → {METADATA_FILE}")
print(df_meta.describe())

if failed:
    with open("data/failed_records.json", "w") as f:
        json.dump(failed, f, indent=2)
    print(f"\n[AVISO] {len(failed)} registros fallaron → data/failed_records.json")

print(f"\n✓ Descarga completa: {len(meta_rows)}/{len(records)} registros guardados en '{OUTPUT_DIR}'")