"""
training/ablation.py
Ablation study: trains or loads all three model variants
(image_only, clinical_only, fused) and evaluates on the test set.
"""

import os
import torch
import torch.nn as nn
import torch.optim as optim

from config import (
    DEVICE, EPOCHS, LR, WEIGHT_DECAY,
    CLINICAL_COLS, WEIGHTS_DIR, PRETRAINED_WEIGHTS, MODES,
)
from models.fusion_net import MultimodalFusionNet
from training.train import train_one_epoch, evaluate


def run_training(mode: str, train_loader, val_loader,
                 epochs: int = EPOCHS) -> tuple:
    """
    Train one model variant and save the best checkpoint.

    Returns:
        model   : best-checkpoint model
        history : dict with train/val loss and accuracy lists
        ckpt    : path to saved checkpoint
    """
    print(f"\n{'='*55}\n  Training mode: {mode.upper()}\n{'='*55}")

    model     = MultimodalFusionNet(len(CLINICAL_COLS), mode=mode).to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=LR,
                            weight_decay=WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    history = {"train_loss": [], "val_loss": [],
               "train_acc": [],  "val_acc": []}
    best_val_loss = float("inf")
    ckpt = os.path.join(WEIGHTS_DIR, f"best_{mode}.pt")

    for epoch in range(1, epochs + 1):
        tr_loss, tr_acc          = train_one_epoch(model, train_loader,
                                                   optimizer, criterion)
        vl_loss, vl_acc, _, _, _ = evaluate(model, val_loader, criterion)
        scheduler.step()

        history["train_loss"].append(tr_loss)
        history["val_loss"].append(vl_loss)
        history["train_acc"].append(tr_acc)
        history["val_acc"].append(vl_acc)

        if vl_loss < best_val_loss:
            best_val_loss = vl_loss
            torch.save(model.state_dict(), ckpt)

        if epoch % 5 == 0 or epoch == 1:
            print(f"Ep {epoch:3d}/{epochs} | "
                  f"Train loss: {tr_loss:.4f}  acc: {tr_acc:.3f} | "
                  f"Val loss: {vl_loss:.4f}  acc: {vl_acc:.3f}")

    print(f"Best checkpoint saved → {ckpt}")
    model.load_state_dict(torch.load(ckpt, map_location=DEVICE))
    return model, history, ckpt


def run_ablation(train_loader, val_loader, test_loader,
                 run_mode: str = "train") -> dict:
    """
    Run the full ablation study for all three modes.

    Args:
        train_loader, val_loader, test_loader : DataLoaders
        run_mode : "train"     — train from scratch
                   "eval_only" — load saved weights from PRETRAINED_WEIGHTS

    Returns:
        results : dict keyed by mode, each containing
                  history, probs, preds, labels, model
    """
    results = {}
    criterion = nn.CrossEntropyLoss()

    if run_mode == "train":
        for mode in MODES:
            model, history, ckpt = run_training(mode, train_loader,
                                                val_loader)
            _, _, probs, preds, labels_true = evaluate(
                model, test_loader, criterion
            )
            results[mode] = {
                "history": history,
                "probs"  : probs,
                "preds"  : preds,
                "labels" : labels_true,
                "model"  : model,
            }

    elif run_mode == "eval_only":
        try:
            from google.colab import drive
            drive.mount("/content/drive", force_remount=False)
        except ImportError:
            pass  # not in Colab — weights must already be local

        for mode in MODES:
            weight_path = PRETRAINED_WEIGHTS[mode]
            if not os.path.exists(weight_path):
                raise FileNotFoundError(
                    f"Weight file not found: {weight_path}\n"
                    "Update PRETRAINED_WEIGHTS in config.py."
                )
            print(f"Loading weights for {mode.upper()} from {weight_path} ...")
            model = MultimodalFusionNet(
                len(CLINICAL_COLS), mode=mode
            ).to(DEVICE)
            model.load_state_dict(
                torch.load(weight_path, map_location=DEVICE)
            )
            _, _, probs, preds, labels_true = evaluate(
                model, test_loader, criterion
            )
            results[mode] = {
                "history": None,   # no history in eval_only mode
                "probs"  : probs,
                "preds"  : preds,
                "labels" : labels_true,
                "model"  : model,
            }
        print("\nAll models loaded and evaluated.")

    else:
        raise ValueError(
            f"Unknown run_mode: {run_mode!r}. Use 'train' or 'eval_only'."
        )

    return results
