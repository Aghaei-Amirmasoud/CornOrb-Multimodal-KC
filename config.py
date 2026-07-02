import os
import torch

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE_DIR    = os.path.dirname(os.path.abspath(__file__))
DATASET_DIR = os.path.join(BASE_DIR, "dataset")
CSV_PATH    = os.path.join(DATASET_DIR, "clinical_data_and_labels.csv")
IMAGE_ROOT  = os.path.join(DATASET_DIR, "ORBSCAN_Dataset")
OUTPUT_DIR  = os.path.join(BASE_DIR, "outputs")
FIGURES_DIR = os.path.join(OUTPUT_DIR, "figures")
WEIGHTS_DIR = os.path.join(OUTPUT_DIR, "weights")

for d in [DATASET_DIR, OUTPUT_DIR, FIGURES_DIR, WEIGHTS_DIR]:
    os.makedirs(d, exist_ok=True)

# ── Zenodo URLs ───────────────────────────────────────────────────────────────
ZENODO_CSV = "https://zenodo.org/records/20542091/files/clinical_data_and_labels.csv?download=1"
ZENODO_ZIP = "https://zenodo.org/records/20542091/files/ORBSCAN_Dataset.zip?download=1"

# ── Google Drive paths (update if using DATA_SOURCE = "drive") ────────────────
DRIVE_CSV = "/content/drive/MyDrive/CornOrb/clinical_data_and_labels.csv"
DRIVE_ZIP = "/content/drive/MyDrive/CornOrb/ORBSCAN_Dataset.zip"

# ── Pretrained weight paths (used when RUN_MODE = "eval_only") ────────────────
PRETRAINED_WEIGHTS = {
    "image_only"   : os.path.join(WEIGHTS_DIR, "best_image_only.pt"),
    "clinical_only": os.path.join(WEIGHTS_DIR, "best_clinical_only.pt"),
    "fused"        : os.path.join(WEIGHTS_DIR, "best_fused.pt"),
}

# ── Data ──────────────────────────────────────────────────────────────────────
MAP_TYPES = ["Axial", "Anterior", "Posterior", "Pachymetry"]

CLINICAL_COLS = [
    "age_years",
    "astig_value_D",
    "kmax_value_D",
    "pachy_central_um",
    "pachy_thinnest_um",
    "asphericity_anterior",
    "asphericity_posterior",
]

# ── Training ──────────────────────────────────────────────────────────────────
SEED         = 42
IMG_SIZE     = 224
BATCH_SIZE   = 32
EPOCHS       = 30
LR           = 1e-4
WEIGHT_DECAY = 1e-4
VAL_SIZE     = 0.15      # fraction of train patients held out for validation

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ── Ablation modes ────────────────────────────────────────────────────────────
MODES = ["image_only", "clinical_only", "fused"]

# ── Plot colours per mode ─────────────────────────────────────────────────────
MODE_COLORS = {
    "image_only"   : "steelblue",
    "clinical_only": "darkorange",
    "fused"        : "green",
}
