# CornOrb Multimodal Keratoconus Detection

A multimodal deep learning framework for binary keratoconus classification using the [CornOrb dataset](https://zenodo.org/records/20542091).  
Combines four Orbscan corneal topography maps (image branch) with seven structured clinical biomarkers (clinical branch) via a late-fusion architecture (**MultimodalFusionNet**).

## Results (Ablation Study)

| Model | ROC-AUC | Accuracy | KC Recall | KC F1 |
|---|---|---|---|---|
| Image-Only | 0.996 | 0.97 | 0.94 | 0.95 |
| Clinical-Only | 0.996 | 0.95 | 0.86 | 0.92 |
| **Fused** | **0.999** | **0.98** | **0.95** | **0.97** |

## Project Structure

```
CornOrb-Multimodal-KC/
├── config.py                  # All hyperparameters and paths
├── requirements.txt
├── data/
│   ├── download.py            # Zenodo / Google Drive data loading
│   └── dataset.py             # CornOrbDataset + DataLoader builders
├── models/
│   ├── clinical_mlp.py        # Clinical feature MLP branch
│   └── fusion_net.py          # MultimodalFusionNet
├── training/
│   ├── train.py               # train_one_epoch + evaluate
│   └── ablation.py            # Ablation study runner
├── evaluation/
│   ├── metrics.py             # ROC, confusion matrix, training curves
│   ├── error_analysis.py      # Misclassified case export + clinical comparison
│   └── gradcam.py             # Grad-CAM visualizations
├── preprocessing/
│   └── splits.py              # Patient-level split + StandardScaler
├── outputs/
│   ├── figures/               # Saved plots
│   └── weights/               # Saved model checkpoints (.pt)
├── dataset/                   # CornOrb CSV + images (downloaded at runtime)
└── CornOrb_Multimodal_Keratoconus.ipynb
```

## Quickstart (Google Colab)

```python
# 1. Clone the repo
!git clone https://github.com/your-username/CornOrb-Multimodal-KC.git
%cd CornOrb-Multimodal-KC

# 2. Install the only missing dependency
!pip install grad-cam

# 3. Open and run the notebook
# CornOrb_Multimodal_Keratoconus.ipynb
```

## Reproducing Results Without Retraining

Pre-trained weights (`best_image_only.pt`, `best_clinical_only.pt`, `best_fused.pt`) are available in the repository under `outputs/weights/`.  
In the notebook, set:
```python
RUN_MODE = "eval_only"
```

## Dataset

Set `DATA_SOURCE = "zenodo"` to download automatically, or `DATA_SOURCE = "drive"` to load from Google Drive.  
Update `DRIVE_CSV` and `DRIVE_ZIP` in `config.py` to match your Drive layout.

- **Source:** [Zenodo DOI: 10.5281/zenodo.17127265](https://doi.org/10.5281/zenodo.17127265)  
- **License:** CC BY 4.0

## Citation

```bibtex
@article{lazouni2026cornorb,
  title   = {CornOrb: A Multimodal Dataset of Orbscan Corneal Topography and Clinical Annotations for Keratoconus Detection},
  author  = {Lazouni, M.E.A. and others},
  journal = {arXiv preprint arXiv:2603.21245},
  year    = {2026}
}
```
