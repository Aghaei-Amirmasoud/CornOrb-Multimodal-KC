"""
preprocessing/splits.py
Patient-level train/val/test splitting and clinical feature standardization.
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from config import SEED, VAL_SIZE, CLINICAL_COLS


def make_splits(df: pd.DataFrame, clinical_cols: list = CLINICAL_COLS,
                val_size: float = VAL_SIZE, seed: int = SEED):
    """
    Split df into train / val / test using the predefined split_80_20 column,
    then carve a patient-level validation set from train.

    Standardizes clinical features using statistics from train only.

    Args:
        df           : full dataframe loaded from CSV
        clinical_cols: list of clinical feature column names
        val_size     : fraction of train patients to use for validation
        seed         : random seed for reproducibility

    Returns:
        train_df, val_df, test_df : standardized DataFrames
        scaler                    : fitted StandardScaler (for inference)
    """
    # ── Use predefined split column ───────────────────────────────────────────
    print("Split values:", df["split_80_20"].value_counts().to_dict())

    train_df = df[df["split_80_20"] == "train"].reset_index(drop=True)
    test_df  = df[df["split_80_20"] == "test"].reset_index(drop=True)

    # ── Patient-level validation carve-out ────────────────────────────────────
    train_pids = train_df["patient_code"].unique()
    train_pids, val_pids = train_test_split(
        train_pids, test_size=val_size, random_state=seed
    )

    val_df   = train_df[train_df["patient_code"].isin(val_pids)].reset_index(drop=True)
    train_df = train_df[train_df["patient_code"].isin(train_pids)].reset_index(drop=True)

    print(f"Train: {len(train_df)} eyes | Val: {len(val_df)} | Test: {len(test_df)}")
    print(f"Train balance: {train_df['label'].value_counts().to_dict()}")
    print(f"Test  balance: {test_df['label'].value_counts().to_dict()}")

    # ── Standardize clinical features (fit on train only) ─────────────────────
    scaler = StandardScaler()
    train_df = train_df.copy()
    val_df   = val_df.copy()
    test_df  = test_df.copy()

    train_df[clinical_cols] = scaler.fit_transform(train_df[clinical_cols])
    val_df[clinical_cols]   = scaler.transform(val_df[clinical_cols])
    test_df[clinical_cols]  = scaler.transform(test_df[clinical_cols])

    print("Clinical features standardized (fit on train only).")

    return train_df, val_df, test_df, scaler


def verify_no_leakage(train_df: pd.DataFrame, val_df: pd.DataFrame,
                      test_df: pd.DataFrame) -> None:
    """Assert no patient appears in more than one split."""
    train_p = set(train_df["patient_code"])
    val_p   = set(val_df["patient_code"])
    test_p  = set(test_df["patient_code"])

    assert len(train_p & val_p)  == 0, f"Train ∩ Val leak:  {train_p & val_p}"
    assert len(train_p & test_p) == 0, f"Train ∩ Test leak: {train_p & test_p}"
    assert len(val_p   & test_p) == 0, f"Val ∩ Test leak:   {val_p & test_p}"

    print("Leakage check passed — no patient overlap across splits.")
