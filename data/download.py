"""
data/download.py
Handles dataset acquisition from Zenodo or Google Drive.
Usage:
    from data.download import load_data
    load_data(source="zenodo")   # or "drive"
"""

import os
import shutil
from config import (
    DATASET_DIR, CSV_PATH, IMAGE_ROOT,
    ZENODO_CSV, ZENODO_ZIP,
    DRIVE_CSV, DRIVE_ZIP,
)


def load_data(source: str = "zenodo") -> None:
    """
    Ensure CSV and image folder are present under dataset/.

    Args:
        source: "zenodo" — download from Zenodo (default)
                "drive"  — copy from Google Drive (must be mounted)
    """
    if source == "zenodo":
        _from_zenodo()
    elif source == "drive":
        _from_drive()
    else:
        raise ValueError(f"Unknown source: {source!r}. Use 'zenodo' or 'drive'.")

    # Final check
    if not os.path.exists(CSV_PATH):
        raise FileNotFoundError(f"CSV not found at {CSV_PATH}")
    if not os.path.exists(IMAGE_ROOT):
        raise FileNotFoundError(f"Image folder not found at {IMAGE_ROOT}")

    print(f"Dataset ready.\n  CSV   : {CSV_PATH}\n  Images: {IMAGE_ROOT}")


def _from_zenodo() -> None:
    if not os.path.exists(CSV_PATH):
        print("Downloading CSV from Zenodo ...")
        os.system(f'wget -q "{ZENODO_CSV}" -O "{CSV_PATH}"')
        print("CSV downloaded.")
    else:
        print("CSV already present — skipping.")

    if not os.path.exists(IMAGE_ROOT):
        zip_path = os.path.join(DATASET_DIR, "ORBSCAN_Dataset.zip")
        print("Downloading image ZIP from Zenodo (~680 MB) ...")
        os.system(f'wget -q --show-progress "{ZENODO_ZIP}" -O "{zip_path}"')
        print("Extracting ...")
        os.system(f'unzip -q "{zip_path}" -d "{DATASET_DIR}/"')
        print("Extracted.")
    else:
        print("Images already extracted — skipping.")


def _from_drive() -> None:
    try:
        from google.colab import drive
        drive.mount("/content/drive", force_remount=False)
    except ImportError:
        raise EnvironmentError("Google Drive mounting is only available in Google Colab.")

    if not os.path.exists(CSV_PATH):
        if not os.path.exists(DRIVE_CSV):
            raise FileNotFoundError(
                f"CSV not found on Drive at {DRIVE_CSV}\n"
                "Update DRIVE_CSV in config.py."
            )
        shutil.copy(DRIVE_CSV, CSV_PATH)
        print(f"CSV copied from Drive → {CSV_PATH}")
    else:
        print("CSV already present — skipping.")

    if not os.path.exists(IMAGE_ROOT):
        if not os.path.exists(DRIVE_ZIP):
            raise FileNotFoundError(
                f"ZIP not found on Drive at {DRIVE_ZIP}\n"
                "Update DRIVE_ZIP in config.py."
            )
        print("Extracting ZIP from Drive ...")
        os.system(f'unzip -q "{DRIVE_ZIP}" -d "{DATASET_DIR}/"')
        print("Extracted.")
    else:
        print("Images already extracted — skipping.")
