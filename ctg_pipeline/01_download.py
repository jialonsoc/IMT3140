"""
01_download.py
--------------
Descarga los registros CTG desde PhysioNet (CTU-UHB, 552 registros).

Por cada registro guarda:
  - data/raw/{id}_fhr.npy   → señal FHR (bpm), array float32 1-D
  - data/raw/{id}_uc.npy    → señal UC (u.a.), array float32 1-D

Además genera:
  - data/raw/download_log.csv → estado de cada registro (ok / sin_uc / error)

Notas sobre los canales del registro WFDB:
  Canal 0 = FHR  (señal principal)
  Canal 1 = UC   (en algunos registros puede estar ausente o ser todo ceros)

El script es idempotente: si el .npy ya existe, omite ese registro.
Ejecutar con:
    cd ctg_pipeline
    python 01_download.py
"""

import numpy as np
import pandas as pd
import wfdb
from tqdm import tqdm
from config import PHYSIONET_DB, RAW_DIR

RAW_DIR.mkdir(parents=True, exist_ok=True)


def download_record(rec_id: str) -> dict:
    """
    Descarga un registro y guarda FHR y UC como .npy.
    Retorna un dict con el estado del registro.
    """
    fhr_path = RAW_DIR / f"{rec_id}_fhr.npy"
    uc_path  = RAW_DIR / f"{rec_id}_uc.npy"

    # Idempotencia: si ya están los dos archivos, saltar
    if fhr_path.exists() and uc_path.exists():
        return {"record": rec_id, "status": "ya_existe", "n_samples": None}

    try:
        rec = wfdb.rdrecord(rec_id, pn_dir=PHYSIONET_DB)
    except Exception as e:
        return {"record": rec_id, "status": f"error: {e}", "n_samples": None}

    signal = rec.p_signal  # shape (n_samples, n_channels), float64

    # Canal 0: FHR
    fhr = signal[:, 0].astype(np.float32)
    np.save(fhr_path, fhr)

    # Canal 1: UC (puede no existir o ser todo NaN)
    if signal.shape[1] > 1:
        uc = signal[:, 1].astype(np.float32)
        has_uc = not np.all(np.isnan(uc))
    else:
        uc = np.full(len(fhr), np.nan, dtype=np.float32)
        has_uc = False

    np.save(uc_path, uc)

    status = "ok" if has_uc else "sin_uc"
    return {"record": rec_id, "status": status, "n_samples": len(fhr)}


def main():
    print(f"Obteniendo lista de registros de '{PHYSIONET_DB}'...")
    records = wfdb.get_record_list(PHYSIONET_DB)
    print(f"  → {len(records)} registros encontrados\n")

    log = []
    for rec_id in tqdm(records, desc="Descargando"):
        result = download_record(rec_id)
        log.append(result)

    df_log = pd.DataFrame(log)
    log_path = RAW_DIR / "download_log.csv"
    df_log.to_csv(log_path, index=False)

    # Resumen
    counts = df_log["status"].value_counts()
    print("\nResumen de descarga:")
    for status, n in counts.items():
        print(f"  {status}: {n}")
    print(f"\nLog guardado → {log_path}")


if __name__ == "__main__":
    main()
