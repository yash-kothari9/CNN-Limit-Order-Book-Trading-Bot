import argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from sklearn.metrics import (
    classification_report, confusion_matrix,
    f1_score, matthews_corrcoef
)
import seaborn as sns

from dataset import (
    load_fi2010, make_demo_data, get_dataloaders,
    compute_class_weights, NUM_CLASSES
)
from model import LOBPredictorCNN


CLASS_NAMES = ["DOWN", "STATIONARY", "UP"]


# ── Argument Parser ──────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate LOB Predictor")
    parser.add_argument("--model",      type=str, default="best_model.pt")
    parser.add_argument("--demo",       action="store_true")
    parser.add_argument("--data_dir",   type=str, default="data/")
    parser.add_argument("--batch_size", type=int, default=64)
    return parser.parse_args()


# ── Model Inference ──────────────────────────────────────────────────

@torch.no_grad()
def get_predictions(model, loader, device):
    model.eval()
    all_preds, all_targets, all_probs = [], [], []

    for batch_x, batch_y in loader:
        batch_x  = batch_x.to(device)
        logits   = model(batch_x)
        probs    = torch.softmax(logits, dim=1)
        preds    = logits.argmax(dim=1)

        all_preds  .extend(preds.cpu().numpy())
        all_targets.extend(batch_y.numpy())
        all_probs  .extend(probs.cpu().numpy())

    return (
        np.array(all_preds),
        np.array(all_targets),
        np.array(all_probs)
    )


# ── Classification Report ───────────────────────────────────────────

def print_classification_report(preds, targets):
    print("\n" + "=" * 55)
    print("  CLASSIFICATION REPORT")
    print("=" * 55)
    report = classification_report(
        targets, preds,
        target_names=CLASS_NAMES,
        digits=4
    )
    print(report)

    wf1 = f1_score(targets, preds, average="weighted")
    mcc = matthews_corrcoef(targets, preds)
    acc = (preds == targets).mean()

    print(f"  Accuracy (overall) : {acc:.4f}  ({acc*100:.1f}%)")
    print(f"  Weighted F1        : {wf1:.4f}")
    print(f"  MCC                : {mcc:.4f}")
    print("=" * 55)


# ── Confusion Matrix Plot ────────────────────────────────────────────

def plot_confusion_matrix(preds, targets, save_path="confusion_matrix.png"):
    cm = confusion_matrix(targets, preds)
    cm_pct = cm.astype(float) / cm.sum(axis=1, keepdims=True) * 100

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Confusion Matrix — LOB Mid-Price Predictor", fontsize=14, fontweight="bold")

    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES, ax=axes[0])
    axes[0].set_title("Raw Counts")
    axes[0].set_xlabel("Predicted Class")
    axes[0].set_ylabel("True Class")

    sns.heatmap(cm_pct, annot=True, fmt=".1f", cmap="Blues",
                xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES, ax=axes[1])
    axes[1].set_title("Row-Normalised (%)")
    axes[1].set_xlabel("Predicted Class")
    axes[1].set_ylabel("True Class")

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\n  ✅ Confusion matrix saved to: {save_path}")


# ── Training Curves Plot ─────────────────────────────────────────────

def plot_training_curves(history_path="training_history.npy",
                         save_path="training_curves.png"):
    try:
        history = np.load(history_path, allow_pickle=True).item()
    except FileNotFoundError:
        print(f"  ⚠️  {history_path} not found — skipping training curves plot")
        return

    epochs = range(1, len(history["train_loss"]) + 1)

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    fig.suptitle("Training History — LOB Predictor CNN", fontsize=14, fontweight="bold")

    axes[0].plot(epochs, history["train_loss"], label="Train", color="#2563EB")
    axes[0].plot(epochs, history["val_loss"],   label="Val",   color="#DC2626", linestyle="--")
    axes[0].set_title("Cross-Entropy Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(epochs, history["train_acc"], label="Train", color="#2563EB")
    axes[1].plot(epochs, history["val_acc"],   label="Val",   color="#DC2626", linestyle="--")
    axes[1].set_title("Accuracy")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy")
    axes[1].set_ylim(0, 1)
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    axes[2].plot(epochs, history["val_f1"],  label="Val F1",  color="#059669")
    axes[2].plot(epochs, history["val_mcc"], label="Val MCC", color="#7C3AED", linestyle="--")
    axes[2].set_title("Val F1 & MCC")
    axes[2].set_xlabel("Epoch")
    axes[2].set_ylabel("Score")
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  ✅ Training curves saved to: {save_path}")


# ── Probability Distribution Plot ────────────────────────────────────

def plot_class_probability_distribution(probs, targets, save_path="prob_distribution.png"):
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    fig.suptitle("Predicted Probability Distributions by True Class",
                 fontsize=13, fontweight="bold")

    colors = ["#DC2626", "#6B7280", "#2563EB"]

    for cls_idx, (cls_name, color) in enumerate(zip(CLASS_NAMES, colors)):
        mask = targets == cls_idx
        if mask.sum() == 0:
            continue
        cls_probs = probs[mask, cls_idx]

        axes[cls_idx].hist(cls_probs, bins=30, color=color, alpha=0.8, edgecolor="white")
        axes[cls_idx].set_title(f"True Class: {cls_name}")
        axes[cls_idx].set_xlabel(f"P({cls_name})")
        axes[cls_idx].set_ylabel("Count")
        axes[cls_idx].axvline(0.65, color="black", linestyle="--", linewidth=1.5,
                              label="Trade threshold (0.65)")
        axes[cls_idx].legend(fontsize=8)
        axes[cls_idx].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  ✅ Probability distribution saved to: {save_path}")


# ── Main ─────────────────────────────────────────────────────────────

def main():
    args   = parse_args()
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    print("\n" + "=" * 55)
    print("   LOB Predictor — Evaluation")
    print("=" * 55)

    if args.demo:
        _, _, test_ds = make_demo_data()
    else:
        _, _, test_ds = load_fi2010(args.data_dir)

    _, _, test_loader = get_dataloaders(test_ds, test_ds, test_ds,
                                        batch_size=args.batch_size)

    print(f"\n  Loading model from: {args.model}")
    model = LOBPredictorCNN(num_classes=NUM_CLASSES).to(device)
    checkpoint = torch.load(args.model, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state"])

    saved_epoch = checkpoint.get("epoch", "?")
    saved_f1    = checkpoint.get("val_f1", 0)
    print(f"  Checkpoint: epoch {saved_epoch}, val F1 = {saved_f1:.4f}")

    preds, targets, probs = get_predictions(model, test_loader, device)

    print_classification_report(preds, targets)
    plot_confusion_matrix(preds, targets)
    plot_training_curves()
    plot_class_probability_distribution(probs, targets)

    print("\n  All evaluation plots saved. ✅")
    print("  Files: confusion_matrix.png, training_curves.png, prob_distribution.png")


if __name__ == "__main__":
    main()
