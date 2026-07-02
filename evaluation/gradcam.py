"""
evaluation/gradcam.py
Grad-CAM visualizations for the fused model's image branch.
One figure per eye showing all 4 corneal map types (original + heatmap).
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

import torch
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from torchvision import transforms

from config import (
    IMG_SIZE, IMAGE_ROOT, MAP_TYPES, GRADCAM_DIR, DEVICE
)

# ── Inference transform (no augmentation) ────────────────────────────────────
_infer_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

_inv_normalize = transforms.Normalize(
    mean=[-0.485/0.229, -0.456/0.224, -0.406/0.225],
    std=[1/0.229, 1/0.224, 1/0.225],
)


def _load_map_tensor(patient_code: str, eye: str, map_type: str,
                     image_root: str = IMAGE_ROOT):
    """Load one corneal map, return (tensor, rgb_np_array)."""
    path = os.path.join(
        image_root, str(patient_code), eye,
        f"{patient_code}_{eye}_{map_type}.png"
    )
    img_pil = (Image.open(path).convert("RGB")
               if os.path.exists(path)
               else Image.new("RGB", (IMG_SIZE, IMG_SIZE), color=0))

    tensor = _infer_transform(img_pil)          # (3, H, W)
    rgb_np = np.array(img_pil.resize((IMG_SIZE, IMG_SIZE))) / 255.0
    return tensor, rgb_np


class ImageOnlyWrapper(torch.nn.Module):
    """Wraps MultimodalFusionNet to accept only image input for Grad-CAM."""
    def __init__(self, model, clinical_tensor):
        super().__init__()
        self.model           = model
        self.clinical_tensor = clinical_tensor  # fixed clinical input

    def forward(self, images):
        return self.model(images, self.clinical_tensor)


def gradcam_for_eye(model, patient_code: str, eye: str,
                    label_str: str, clinical_tensor,
                    save_path: str = None) -> plt.Figure:
    """
    Run Grad-CAM on all 4 corneal maps for one eye and return a figure.

    Args:
        model          : trained MultimodalFusionNet (fused mode)
        patient_code   : e.g. "125GC"
        eye            : "OD" or "OS"
        label_str      : display label for the figure title
        clinical_tensor: (1, n_features) clinical feature tensor
        save_path      : optional path to save the figure PNG

    Returns:
        matplotlib Figure
    """
    model.eval()

    map_tensors, rgb_nps = [], []
    for m in MAP_TYPES:
        t, rgb = _load_map_tensor(patient_code, eye, m)
        map_tensors.append(t)
        rgb_nps.append(rgb)

    stacked = torch.cat(map_tensors, dim=0).unsqueeze(0).to(DEVICE)  # (1,12,H,W)
    clin    = clinical_tensor.to(DEVICE)

    wrapped_model = ImageOnlyWrapper(model, clin)
    target_layer = wrapped_model.model.image_branch[-2][-1].conv2
    cam = GradCAM(model=wrapped_model, target_layers=[target_layer])

    fig, axes = plt.subplots(2, 4, figsize=(20, 10))
    fig.suptitle(
        f"Grad-CAM — {label_str} ({patient_code} {eye})", fontsize=13
    )

    for col_idx, (m, rgb_np) in enumerate(zip(MAP_TYPES, rgb_nps)):
        # Compute Grad-CAM for this map's channel slice
        grayscale_cam = cam(
            input_tensor=stacked,
            targets=None,
        )[0]
        cam_image = show_cam_on_image(rgb_np.astype(np.float32),
                                      grayscale_cam, use_rgb=True)

        axes[0][col_idx].imshow(rgb_np)
        axes[0][col_idx].set_title(f"{m}\nOriginal", fontsize=9)
        axes[0][col_idx].axis("off")

        axes[1][col_idx].imshow(cam_image)
        axes[1][col_idx].set_title(f"{m}\nGrad-CAM", fontsize=9)
        axes[1][col_idx].axis("off")

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved → {save_path}")

    return fig


def run_gradcam_analysis(model, results: dict, test_df, scaler=None,
                         clinical_cols=None, n_top: int = 5) -> None:
    """
    Generate Grad-CAM figures for:
      - Top-n most confidently misclassified KC eyes (false negatives)
      - Top-n most confidently correct KC eyes
      - The false positive case (if any)
      - Top-n most confidently correct Normal eyes

    Saves all figures to FIGURES_DIR.
    """
    import pandas as pd
    from config import CLINICAL_COLS
    clinical_cols = clinical_cols or CLINICAL_COLS

    res = results["fused"]
    df  = test_df.copy().reset_index(drop=True)
    df["true_label"]      = res["labels"]
    df["predicted_label"] = res["preds"]
    df["pred_proba_KC"]   = res["probs"]

    def _get_clinical(row):
        return torch.tensor(
            row[clinical_cols].values.astype(np.float32)
        ).unsqueeze(0)

    # False negatives (missed KC) — sorted by confidence descending (most wrong first)
    fn_df = df[(df["true_label"] == 1) & (df["predicted_label"] == 0)].copy()
    fn_df = fn_df.sort_values("pred_proba_KC")  # lowest KC prob = most confident FN

    print(f"Generating Grad-CAM for top {n_top} false negatives ...")
    for _, row in fn_df.head(n_top).iterrows():
        save = os.path.join(
            GRADCAM_DIR,
            f"gradcam_error_{row['patient_code']}_{row['eye']}.png"
        )
        gradcam_for_eye(
            model, row["patient_code"], row["eye"],
            f"False Negative (missed KC) — conf={row['pred_proba_KC']:.3f}",
            _get_clinical(row), save_path=save
        )
        plt.close()

    # Correct KC — sorted by confidence descending
    tp_df = df[(df["true_label"] == 1) & (df["predicted_label"] == 1)].copy()
    tp_df = tp_df.sort_values("pred_proba_KC", ascending=False)

    print(f"Generating Grad-CAM for top {n_top} correct KC detections ...")
    for _, row in tp_df.head(n_top).iterrows():
        save = os.path.join(
            GRADCAM_DIR,
            f"gradcam_correct_{row['patient_code']}_{row['eye']}.png"
        )
        gradcam_for_eye(
            model, row["patient_code"], row["eye"],
            f"Correctly Detected KC — conf={row['pred_proba_KC']:.3f}",
            _get_clinical(row), save_path=save
        )
        plt.close()

    # False positive(s)
    fp_df = df[(df["true_label"] == 0) & (df["predicted_label"] == 1)].copy()
    fp_df = fp_df.sort_values("pred_proba_KC", ascending=False)

    print(f"Generating Grad-CAM for {len(fp_df)} false positive(s) ...")
    for _, row in fp_df.iterrows():
        save = os.path.join(
            GRADCAM_DIR,
            f"gradcam_FP_{row['patient_code']}_{row['eye']}.png"
        )
        gradcam_for_eye(
            model, row["patient_code"], row["eye"],
            f"False Positive — conf={row['pred_proba_KC']:.3f}",
            _get_clinical(row), save_path=save
        )
        plt.close()

    # Correct Normal — sorted by lowest KC probability
    tn_df = df[(df["true_label"] == 0) & (df["predicted_label"] == 0)].copy()
    tn_df = tn_df.sort_values("pred_proba_KC")

    print(f"Generating Grad-CAM for top {n_top} correct Normal detections ...")
    for _, row in tn_df.head(n_top).iterrows():
        save = os.path.join(
            GRADCAM_DIR,
            f"gradcam_correct_normal_{row['patient_code']}_{row['eye']}.png"
        )
        gradcam_for_eye(
            model, row["patient_code"], row["eye"],
            f"Correctly Detected Normal — conf={1 - row['pred_proba_KC']:.3f}",
            _get_clinical(row), save_path=save
        )
        plt.close()

    print("Grad-CAM analysis complete.")
