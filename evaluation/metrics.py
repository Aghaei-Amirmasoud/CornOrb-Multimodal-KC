"""
evaluation/metrics.py
Classification metrics, ROC curves, confusion matrices, and training curves.
"""

import os
import matplotlib.pyplot as plt
from sklearn.metrics import (
    classification_report, confusion_matrix,
    roc_auc_score, roc_curve, ConfusionMatrixDisplay,
)

from config import FIGURES_DIR, MODE_COLORS


def print_classification_reports(results: dict) -> None:
    """Print ROC-AUC and classification report for each model mode."""
    print(f"{'='*55}\n  TEST SET RESULTS (Ablation Study)\n{'='*55}")
    for mode, res in results.items():
        auc = roc_auc_score(res["labels"], res["probs"])
        print(f"\n--- {mode.upper()} ---  ROC-AUC: {auc:.4f}")
        print(classification_report(
            res["labels"], res["preds"],
            target_names=["Normal", "Keratoconus"]
        ))


def plot_confusion_matrices(results: dict) -> None:
    """Plot side-by-side confusion matrices for all modes."""
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    for ax, (mode, res) in zip(axes, results.items()):
        cm  = confusion_matrix(res["labels"], res["preds"])
        auc = roc_auc_score(res["labels"], res["probs"])
        ConfusionMatrixDisplay(
            cm, display_labels=["Normal", "Keratoconus"]
        ).plot(ax=ax, colorbar=False)
        ax.set_title(f"{mode}\nROC-AUC: {auc:.3f}")
    plt.tight_layout()
    out = os.path.join(FIGURES_DIR, "confusion_matrices.png")
    plt.savefig(out, dpi=150)
    plt.show()
    print(f"Saved → {out}")


def plot_roc_curves(results: dict) -> None:
    """Plot ROC curves for all modes on one figure."""
    plt.figure(figsize=(7, 6))
    for mode, res in results.items():
        fpr, tpr, _ = roc_curve(res["labels"], res["probs"])
        auc = roc_auc_score(res["labels"], res["probs"])
        plt.plot(fpr, tpr,
                 label=f"{mode} (AUC={auc:.3f})",
                 color=MODE_COLORS[mode])
    plt.plot([0, 1], [0, 1], "k--", alpha=0.4)
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curves — Ablation Study")
    plt.legend()
    plt.tight_layout()
    out = os.path.join(FIGURES_DIR, "roc_curves.png")
    plt.savefig(out, dpi=150)
    plt.show()
    print(f"Saved → {out}")


def plot_training_curves(results: dict) -> None:
    """Plot validation loss and accuracy curves (skipped in eval_only mode)."""
    if all(res["history"] is None for res in results.values()):
        print("Training curves not available in eval_only mode.")
        return

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for mode, res in results.items():
        if res["history"] is None:
            continue
        h = res["history"]
        axes[0].plot(h["val_loss"], label=mode, color=MODE_COLORS[mode])
        axes[1].plot(h["val_acc"],  label=mode, color=MODE_COLORS[mode])

    axes[0].set_title("Validation Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].legend()
    axes[1].set_title("Validation Accuracy")
    axes[1].set_xlabel("Epoch")
    axes[1].legend()
    plt.tight_layout()
    out = os.path.join(FIGURES_DIR, "training_curves.png")
    plt.savefig(out, dpi=150)
    plt.show()
    print(f"Saved → {out}")


def summary_table(results: dict) -> None:
    """Print a compact summary table of all metrics."""
    from sklearn.metrics import f1_score, accuracy_score
    import pandas as pd

    rows = []
    for mode, res in results.items():
        auc = roc_auc_score(res["labels"], res["probs"])
        rows.append({
            "Mode"    : mode,
            "ROC-AUC" : round(auc, 4),
            "Accuracy": round(accuracy_score(res["labels"], res["preds"]), 4),
            "F1-macro": round(f1_score(res["labels"], res["preds"],
                                       average="macro"), 4),
            "KC Recall": round(
                (res["preds"][res["labels"] == 1] == 1).mean(), 4
            ),
        })
    print(pd.DataFrame(rows).to_string(index=False))
