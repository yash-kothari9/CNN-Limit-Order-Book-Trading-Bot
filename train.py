import argparse
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import f1_score, matthews_corrcoef

from dataset import (
    load_fi2010, make_demo_data, get_dataloaders,
    compute_class_weights, NUM_CLASSES
)
from model import LOBPredictorCNN, model_summary


# ── Argument Parser ──────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(description="Train LOB Predictor CNN")
    parser.add_argument("--demo",       action="store_true",
                        help="Use synthetic demo data (no download needed)")
    parser.add_argument("--data_dir",   type=str, default="data/",
                        help="Path to FI-2010 data files")
    parser.add_argument("--epochs",     type=int, default=50,
                        help="Maximum training epochs (default: 50)")
    parser.add_argument("--batch_size", type=int, default=32,
                        help="Batch size (default: 32)")
    parser.add_argument("--lr",         type=float, default=1e-3,
                        help="Learning rate (default: 0.001)")
    parser.add_argument("--patience",   type=int, default=15,
                        help="Early stopping patience (default: 15)")
    parser.add_argument("--save_path",  type=str, default="best_model.pt",
                        help="Where to save the best model weights")
    return parser.parse_args()


# ── Training & Validation ────────────────────────────────────────────

def train_one_epoch(model, loader, criterion, optimizer, device, grad_clip=1.0):
    model.train()
    total_loss, correct, total = 0.0, 0, 0

    for batch_x, batch_y in loader:
        batch_x = batch_x.to(device)
        batch_y = batch_y.to(device)

        optimizer.zero_grad()
        logits = model(batch_x)
        loss = criterion(logits, batch_y)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()

        total_loss += loss.item() * len(batch_y)
        preds       = logits.argmax(dim=1)
        correct    += (preds == batch_y).sum().item()
        total      += len(batch_y)

    return total_loss / total, correct / total


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss, all_preds, all_targets = 0.0, [], []

    for batch_x, batch_y in loader:
        batch_x = batch_x.to(device)
        batch_y = batch_y.to(device)

        logits  = model(batch_x)
        loss    = criterion(logits, batch_y)
        preds   = logits.argmax(dim=1)

        total_loss  += loss.item() * len(batch_y)
        all_preds   .extend(preds.cpu().numpy())
        all_targets .extend(batch_y.cpu().numpy())

    all_preds   = np.array(all_preds)
    all_targets = np.array(all_targets)

    n = len(all_targets)
    return {
        "loss"    : total_loss / n,
        "accuracy": (all_preds == all_targets).mean(),
        "f1"      : f1_score(all_targets, all_preds, average="weighted", zero_division=0),
        "mcc"     : matthews_corrcoef(all_targets, all_preds),
        "preds"   : all_preds,
        "targets" : all_targets,
    }


# ── Main Training Loop ───────────────────────────────────────────────

def train(args):
    print("\n" + "=" * 60)
    print("   High-Frequency Order Book Predictor — Training")
    print("=" * 60)

    # Device selection
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"\n  Device: {device}")
    if device.type == "cuda":
        print(f"  GPU   : {torch.cuda.get_device_name(0)}")
    elif device.type == "mps":
        print(f"  GPU   : Apple Silicon (Metal Performance Shaders)")

    # Data loading
    print(f"\n{'─'*60}")
    print("  Loading data...")
    if args.demo:
        train_ds, val_ds, test_ds = make_demo_data()
    else:
        train_ds, val_ds, test_ds = load_fi2010(args.data_dir)

    train_loader, val_loader, test_loader = get_dataloaders(
        train_ds, val_ds, test_ds, batch_size=args.batch_size
    )
    print(f"  Train batches : {len(train_loader)}")
    print(f"  Val   batches : {len(val_loader)}")
    print(f"  Test  batches : {len(test_loader)}")

    # Model initialisation
    print(f"\n{'─'*60}")
    model = LOBPredictorCNN(num_classes=NUM_CLASSES).to(device)
    model_summary(model)

    # Loss function with class weights
    class_weights = compute_class_weights(train_ds).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    print(f"\n  Class weights : {class_weights.tolist()}")

    # Optimizer
    optimizer = optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=1e-4
    )

    # Learning rate scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=7
    )

    # Training history
    history = {
        "train_loss": [], "train_acc": [],
        "val_loss"  : [], "val_acc"  : [], "val_f1": [], "val_mcc": []
    }

    # Early stopping state
    best_val_f1     = 0.0
    patience_count  = 0

    # Training loop
    print(f"\n{'─'*60}")
    print(f"  Training for up to {args.epochs} epochs (early stop patience={args.patience})")
    print(f"{'─'*60}")
    print(f"  {'Epoch':>5}  {'TrLoss':>7}  {'TrAcc':>6}  "
          f"{'VaLoss':>7}  {'VaAcc':>6}  {'VaF1':>6}  {'VaMCC':>6}  {'Time':>5}")
    print(f"  {'─'*5}  {'─'*7}  {'─'*6}  {'─'*7}  {'─'*6}  {'─'*6}  {'─'*6}  {'─'*5}")

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()

        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, device
        )
        val_metrics = evaluate(model, val_loader, criterion, device)

        elapsed = time.time() - t0

        history["train_loss"].append(train_loss)
        history["train_acc"] .append(train_acc)
        history["val_loss"]  .append(val_metrics["loss"])
        history["val_acc"]   .append(val_metrics["accuracy"])
        history["val_f1"]    .append(val_metrics["f1"])
        history["val_mcc"]   .append(val_metrics["mcc"])

        print(f"  {epoch:>5}  {train_loss:>7.4f}  {train_acc:>6.3f}  "
              f"{val_metrics['loss']:>7.4f}  {val_metrics['accuracy']:>6.3f}  "
              f"{val_metrics['f1']:>6.3f}  {val_metrics['mcc']:>6.3f}  "
              f"{elapsed:>4.1f}s")

        scheduler.step(val_metrics["f1"])

        if val_metrics["f1"] > best_val_f1:
            best_val_f1    = val_metrics["f1"]
            patience_count = 0
            torch.save({
                "epoch"          : epoch,
                "model_state"    : model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "val_f1"         : best_val_f1,
                "val_acc"        : val_metrics["accuracy"],
                "val_mcc"        : val_metrics["mcc"],
            }, args.save_path)
            print(f"  ✅ Best model saved (val F1 = {best_val_f1:.4f})")
        else:
            patience_count += 1
            if patience_count >= args.patience:
                print(f"\n  ⏹  Early stopping at epoch {epoch} "
                      f"(no improvement for {args.patience} epochs)")
                break

    # Final test evaluation
    print(f"\n{'─'*60}")
    print("  Loading best model and evaluating on test set...")

    checkpoint = torch.load(args.save_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state"])
    test_metrics = evaluate(model, test_loader, criterion, device)

    print(f"\n{'=' * 60}")
    print("  FINAL TEST RESULTS")
    print(f"{'=' * 60}")
    print(f"  Accuracy  : {test_metrics['accuracy']:.4f}  ({test_metrics['accuracy']*100:.1f}%)")
    print(f"  Weighted F1: {test_metrics['f1']:.4f}")
    print(f"  MCC        : {test_metrics['mcc']:.4f}")
    print(f"{'=' * 60}")

    if args.demo:
        print("\n  ℹ  Note: Demo results are near-random (~33% accuracy).")
        print("     This is expected — synthetic data has no real signal.")
        print("     Use --data_dir with FI-2010 data for real results.")

    np.save("training_history.npy", history)
    print(f"\n  Training history saved to: training_history.npy")
    print(f"  Best model saved to      : {args.save_path}")
    print("\n  Run: python evaluate.py  to see detailed results & plots")

    return model, test_metrics, history


# ── Entry Point ──────────────────────────────────────────────────────

if __name__ == "__main__":
    args = parse_args()
    train(args)
