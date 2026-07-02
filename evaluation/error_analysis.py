"""
evaluation/error_analysis.py
Exports misclassified cases and compares their clinical feature profiles
against correctly classified cases.
"""

import os
import pandas as pd

from config import OUTPUT_DIR, CLINICAL_COLS


def export_misclassified(test_df: pd.DataFrame, results: dict,
                         mode: str = "fused",
                         clinical_cols: list = CLINICAL_COLS) -> pd.DataFrame:
    """
    Build a DataFrame of misclassified cases for the given mode,
    save to CSV, and return it.

    Args:
        test_df      : raw (standardized) test DataFrame
        results      : dict from run_ablation()
        mode         : which model variant to analyse (default "fused")
        clinical_cols: list of clinical feature column names

    Returns:
        misclassified_df : DataFrame of wrong predictions
    """
    res = results[mode]
    df  = test_df.copy().reset_index(drop=True)

    df["true_label"]      = res["labels"]
    df["predicted_label"] = res["preds"]
    df["pred_proba_KC"]   = res["probs"]
    df["label_name"]      = df["true_label"].map({0: "Normal", 1: "Keratoconus"})
    df["predicted_name"]  = df["predicted_label"].map({0: "Normal", 1: "Keratoconus"})

    misclassified_df = df[df["true_label"] != df["predicted_label"]].copy()
    misclassified_df["error_type"] = misclassified_df.apply(
        lambda row: "False Negative (missed KC)" if row["true_label"] == 1
                    else "False Positive (over-called KC)",
        axis=1,
    )

    print(f"Total misclassified: {len(misclassified_df)} / {len(df)}")
    print(misclassified_df["error_type"].value_counts())

    export_cols = [
        "patient_code", "eye", "age_years", "gender",
        "kmax_value_D", "pachy_central_um", "pachy_thinnest_um",
        "astig_value_D", "asphericity_anterior", "asphericity_posterior",
        "true_label", "label_name", "predicted_label", "predicted_name",
        "pred_proba_KC", "error_type",
    ]
    out = os.path.join(OUTPUT_DIR, f"misclassified_{mode}_test.csv")
    misclassified_df[export_cols].to_csv(out, index=False)
    print(f"Saved → {out}")

    return misclassified_df


def clinical_feature_comparison(test_df: pd.DataFrame, results: dict,
                                 mode: str = "fused",
                                 clinical_cols: list = CLINICAL_COLS) -> None:
    """
    Print a comparison of clinical feature means/stds between
    misclassified and correctly classified cases.
    """
    res = results[mode]
    df  = test_df.copy().reset_index(drop=True)
    df["true_label"]      = res["labels"]
    df["predicted_label"] = res["preds"]

    mis_df = df[df["true_label"] != df["predicted_label"]]
    cor_df = df[df["true_label"] == df["predicted_label"]]

    print("\n" + "="*60)
    print("CLINICAL FEATURE COMPARISON: Misclassified vs. Correct")
    print("="*60)

    rows = []
    for col in clinical_cols:
        rows.append({
            "Feature"           : col,
            "Misclassified_mean": round(mis_df[col].mean(), 3),
            "Misclassified_std" : round(mis_df[col].std(), 3),
            "Correct_mean"      : round(cor_df[col].mean(), 3),
            "Correct_std"       : round(cor_df[col].std(), 3),
            "Mean_diff"         : round(mis_df[col].mean() - cor_df[col].mean(), 3),
        })

    print(pd.DataFrame(rows).to_string(index=False))

    print("\n" + "="*60)
    print("EYE LATERALITY (OD/OS) IN ERRORS")
    print("="*60)
    print(mis_df["eye"].value_counts(normalize=True).round(3))
