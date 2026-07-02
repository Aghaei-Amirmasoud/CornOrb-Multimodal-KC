"""
training/train.py
Core training and evaluation loop functions.
"""

import torch
import torch.nn as nn

from config import DEVICE


def train_one_epoch(model: nn.Module, loader, optimizer, criterion) -> tuple:
    """
    Run one training epoch.

    Returns:
        avg_loss : float
        accuracy : float
    """
    model.train()
    total_loss, correct, total = 0.0, 0, 0

    for imgs, clin, labels in loader:
        imgs, clin, labels = (imgs.to(DEVICE), clin.to(DEVICE),
                              labels.to(DEVICE))
        optimizer.zero_grad()
        logits = model(imgs, clin)
        loss   = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * labels.size(0)
        correct    += (logits.argmax(1) == labels).sum().item()
        total      += labels.size(0)

    return total_loss / total, correct / total


@torch.no_grad()
def evaluate(model: nn.Module, loader, criterion) -> tuple:
    """
    Evaluate model on a DataLoader.

    Returns:
        avg_loss  : float
        accuracy  : float
        probs     : np.ndarray  — predicted probability of KC (class 1)
        preds     : np.ndarray  — predicted class labels
        labels    : np.ndarray  — true labels
    """
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    all_probs, all_preds, all_labels = [], [], []

    for imgs, clin, labels in loader:
        imgs, clin, labels = (imgs.to(DEVICE), clin.to(DEVICE),
                              labels.to(DEVICE))
        logits = model(imgs, clin)
        loss   = criterion(logits, labels)
        probs  = torch.softmax(logits, dim=1)[:, 1]

        total_loss += loss.item() * labels.size(0)
        correct    += (logits.argmax(1) == labels).sum().item()
        total      += labels.size(0)

        all_probs.append(probs.cpu())
        all_preds.append(logits.argmax(1).cpu())
        all_labels.append(labels.cpu())

    return (
        total_loss / total,
        correct / total,
        torch.cat(all_probs).numpy(),
        torch.cat(all_preds).numpy(),
        torch.cat(all_labels).numpy(),
    )
