"""
config.py
---------
Única fuente de verdad para rutas, constantes fisiológicas y
parámetros del bootstrap. Todos los scripts importan desde aquí.
"""

from pathlib import Path

# ── Rutas ──────────────────────────────────────────────────────────────────────
BASE_DIR   = Path(__file__).parent          # ctg_pipeline/
DATA_DIR   = BASE_DIR / "data"
RAW_DIR    = DATA_DIR / "raw"
SYNTH_DIR  = DATA_DIR / "synthetic"
REPORT_DIR = BASE_DIR / "reports"

ORIGINAL_CSV  = DATA_DIR / "dataset_original.csv"
SYNTHETIC_CSV = DATA_DIR / "dataset_synthetic.csv"
FULL_CSV      = DATA_DIR / "dataset_full.csv"

# ── Señal ─────────────────────────────────────────────────────────────────────
FS       = 4      # Hz  (250 ms entre muestras, estándar CTU-UHB)
FHR_MIN  = 60.0   # bpm
FHR_MAX  = 200.0  # bpm

# ── Filtros de calidad (aplicados antes de bootstrapear) ──────────────────────
NAN_PCT_MAX   = 30.0   # % máximo de NaNs permitidos en FHR
DURATION_MIN  = 30.0   # minutos mínimos de registro válido

# ── Bootstrap (D-02, D-03, D-04 en DECISIONS.md) ─────────────────────────────
BLOCK_MIN_S      = 60    # segundos — mínimo clínico: ventana LTV (1 min)
BLOCK_MAX_S      = 300   # segundos — escala del ciclo contracción→FHR (≤5 min)
ACF_ZERO_THR     = 0.05  # umbral ACF (no se alcanza en FHR → fallback = b_max/2 = 150s → b*=300s)
N_MULTIPLIER_MAX = 10    # tope de búsqueda del N óptimo
CONVERGENCE_THR  = 0.01  # |Δstd(estimador)| para declarar convergencia

# ── Base de datos PhysioNet ───────────────────────────────────────────────────
PHYSIONET_DB = "ctu-uhb-ctgdb"

# ── Reproducibilidad ─────────────────────────────────────────────────────────
SEED = 42
