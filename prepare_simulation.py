import json
import numpy as np
import torch
import torch.nn.functional as F
from dataset import load_fi2010, get_dataloaders, NUM_FEATURES
from model import LOBPredictorCNN


# ── Constants ────────────────────────────────────────────────────────

CLASS_DOWN, CLASS_STAT, CLASS_UP = 0, 1, 2
THRESHOLD = 0.65
SLIPPAGE_BPS = 0.75
COMMISSION_BPS = 0.2
COST_PER_TRADE = (SLIPPAGE_BPS + COMMISSION_BPS) / 10_000
TICK_SIZE = 0.01


# ── Main ─────────────────────────────────────────────────────────────

def main():
    print("=" * 58)
    print("  Preparing FULL Simulation Data (all test steps)")
    print("=" * 58)

    # Load model
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    print(f"  Device: {device}")

    model = LOBPredictorCNN(num_classes=3).to(device)
    ckpt = torch.load("best_model.pt", map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    print("  Model loaded")

    # Load test data
    train_ds, val_ds, test_ds = load_fi2010("data/")
    _, _, test_loader = get_dataloaders(train_ds, val_ds, test_ds, batch_size=256)
    total_test = len(test_ds)
    print(f"  Total test samples: {total_test:,}")

    # Batch inference
    print("  Running batched inference...")
    all_probs = []
    all_labels = []
    with torch.no_grad():
        for batch_x, batch_y in test_loader:
            logits = model(batch_x.to(device))
            probs = F.softmax(logits, dim=1).cpu().numpy()
            all_probs.append(probs)
            all_labels.append(batch_y.numpy())
    all_probs = np.vstack(all_probs)
    all_labels = np.concatenate(all_labels)
    print(f"  Inference complete: {len(all_probs):,} predictions")

    # Sequential backtest
    print("  Running full backtest...")
    mid_price = 100.0
    position = 0
    entry_price = None
    cumulative_pnl = 0.0
    completed_trades = 0
    winning_trades = 0
    raw_pnls = []
    steps = []
    trade_entries = []

    for i in range(total_test):
        true_label = int(all_labels[i])
        p_down, p_stat, p_up = float(all_probs[i, 0]), float(all_probs[i, 1]), float(all_probs[i, 2])

        if true_label == CLASS_UP:
            mid_price += TICK_SIZE
        elif true_label == CLASS_DOWN:
            mid_price -= TICK_SIZE

        confidence = max(p_down, p_stat, p_up)
        sig = 0
        if p_up > THRESHOLD and position == 0:
            sig = 1
        elif p_down > THRESHOLD and position == 0:
            sig = 2
        elif position != 0 and p_stat > THRESHOLD:
            sig = 3

        te = None
        if sig == 1 and position == 0:
            position = 1
            entry_price = mid_price
            te = {"a": "BUY", "pr": round(mid_price, 3)}
        elif sig == 2 and position == 0:
            position = -1
            entry_price = mid_price
            te = {"a": "SELL", "pr": round(mid_price, 3)}
        elif sig == 3 and position != 0:
            gross = position * (mid_price - entry_price)
            cost = 2 * COST_PER_TRADE * mid_price
            net = gross - cost
            raw_pnls.append(gross)
            cumulative_pnl += net
            completed_trades += 1
            if gross > 0:
                winning_trades += 1
            te = {"a": "EXIT", "pr": round(mid_price, 3), "pnl": round(net, 4)}
            position = 0
            entry_price = None

        if te:
            te["i"] = i
            trade_entries.append(te)

        hr = round(winning_trades / completed_trades * 100, 1) if completed_trades > 0 else 0
        sr = 0.0
        if len(raw_pnls) > 1 and np.std(raw_pnls) > 0:
            sr = round(float(np.mean(raw_pnls) / np.std(raw_pnls) * np.sqrt(252)), 2)

        pos_code = 0 if position == 0 else (1 if position == 1 else 2)

        steps.append([
            round(cumulative_pnl, 4),
            round(p_down * 100, 1), round(p_stat * 100, 1), round(p_up * 100, 1),
            sig, pos_code,
            completed_trades, hr, sr,
            round(mid_price, 3)
        ])

        if i % 10000 == 0:
            print(f"    Step {i:,}/{total_test:,}  PnL=${cumulative_pnl:.4f}  Trades={completed_trades}")

    print(f"    Step {total_test:,}/{total_test:,}  PnL=${cumulative_pnl:.4f}  Trades={completed_trades}")

    # Extract order book for all steps
    print(f"  Extracting order book snapshots for all {total_test} steps...")
    ob_list = []
    for idx in range(total_test):
        x, _ = test_ds[idx]
        features = x[0, -1, :].numpy()
        mp = steps[idx][9]
        asks, bids = [], []
        for lv in range(5):
            fi = lv * 4
            spread = 0.005 + lv * 0.014 + abs(float(features[fi])) * 0.003
            asks.append([round(mp + spread, 3), max(50, int(300 + float(features[fi+1]) * 180))])
            bids.append([round(mp - spread, 3), max(50, int(300 + float(features[fi+3]) * 180))])
        ob_list.append([asks, bids])

    # Save as JS
    print("  Saving...")
    output = {
        "total": total_test,
        "threshold": THRESHOLD,
        "steps": steps,
        "trades": trade_entries,
        "ob": ob_list
    }

    class NpEncoder(json.JSONEncoder):
        def default(self, obj):
            if isinstance(obj, (np.integer,)): return int(obj)
            if isinstance(obj, (np.floating,)): return float(obj)
            if isinstance(obj, np.ndarray): return obj.tolist()
            return super().default(obj)

    js = f"const SIM={json.dumps(output, separators=(',',':'), cls=NpEncoder)};"
    with open("simulation_data.js", "w") as f:
        f.write(js)

    sz = len(js) / (1024 * 1024)
    print(f"\n  ✅ Saved: simulation_data.js ({sz:.1f} MB)")
    print(f"  Total steps     : {total_test:,}")
    print(f"  Final PnL       : ${cumulative_pnl:.4f}")
    print(f"  Completed trades: {completed_trades}")
    print(f"  Hit rate        : {hr}%")
    print(f"  Sharpe ratio    : {sr}")
    print(f"  Trade entries   : {len(trade_entries)}")
    print(f"  OB snapshots    : {len(ob_list)} (Every step)")
    print(f"\n  Open simulation.html in your browser")

if __name__ == "__main__":
    main()
