import argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F

from dataset import load_fi2010, make_demo_data, get_dataloaders, NUM_CLASSES
from model import LOBPredictorCNN


# ── Constants ────────────────────────────────────────────────────────

CLASS_DOWN       = 0
CLASS_STATIONARY = 1
CLASS_UP         = 2


# ── Argument Parser ──────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(description="Backtest LOB Predictor")
    parser.add_argument("--model",       type=str,   default="best_model.pt")
    parser.add_argument("--demo",        action="store_true")
    parser.add_argument("--data_dir",    type=str,   default="data/")
    parser.add_argument("--threshold",   type=float, default=0.65)
    parser.add_argument("--slippage_bps",type=float, default=0.75)
    parser.add_argument("--commission_bps",type=float, default=0.2)
    parser.add_argument("--batch_size",  type=int,   default=64)
    return parser.parse_args()


# ── Model Inference ──────────────────────────────────────────────────

@torch.no_grad()
def get_all_probabilities(model, loader, device):
    model.eval()
    all_probs   = []
    all_targets = []

    for batch_x, batch_y in loader:
        logits = model(batch_x.to(device))
        probs  = F.softmax(logits, dim=1)
        all_probs  .append(probs.cpu().numpy())
        all_targets.append(batch_y.numpy())

    return np.vstack(all_probs), np.concatenate(all_targets)


# ── Backtesting Engine ───────────────────────────────────────────────

class BacktestEngine:

    def __init__(self, threshold: float, slippage_bps: float, commission_bps: float):
        self.threshold      = threshold
        self.cost_per_trade = (slippage_bps + commission_bps) / 10_000

        self.current_price  = 100.0
        self.position       = 0
        self.entry_price    = None

        self.pnl_series     = []
        self.raw_pnl        = []
        self.trade_log      = []
        self.cumulative_pnl = 0.0

    def step(self, probs: np.ndarray, true_label: int, step_idx: int):
        p_down, p_stat, p_up = probs

        # Update cumulative price based on true label
        tick_size = 0.01
        price_change = {CLASS_UP: tick_size, CLASS_DOWN: -tick_size,
                        CLASS_STATIONARY: 0.0}[true_label]
        self.current_price += price_change

        # Determine signal
        if p_up > self.threshold and self.position == 0:
            signal = "BUY"
        elif p_down > self.threshold and self.position == 0:
            signal = "SELL"
        elif self.position != 0 and p_stat > self.threshold:
            signal = "EXIT"
        else:
            signal = "HOLD"

        # Execute signal
        trade_pnl = 0.0

        if signal == "BUY" and self.position == 0:
            self.position    = 1
            self.entry_price = self.current_price
            self.trade_log.append({"step": step_idx, "action": "BUY",
                                   "price": self.current_price, "p_up": p_up})

        elif signal == "SELL" and self.position == 0:
            self.position    = -1
            self.entry_price = self.current_price
            self.trade_log.append({"step": step_idx, "action": "SELL",
                                   "price": self.current_price, "p_down": p_down})

        elif signal == "EXIT" and self.position != 0:
            gross_pnl = self.position * (self.current_price - self.entry_price)
            cost       = 2 * self.cost_per_trade * self.current_price
            trade_pnl  = gross_pnl - cost

            self.raw_pnl.append(gross_pnl)
            self.cumulative_pnl += trade_pnl
            self.trade_log.append({"step": step_idx, "action": "EXIT",
                                   "price": self.current_price, "pnl": trade_pnl})
            self.position    = 0
            self.entry_price = None

        self.pnl_series.append(self.cumulative_pnl)

    def results(self) -> dict:
        if len(self.raw_pnl) == 0:
            print("  ⚠️  No completed trades — try lowering --threshold")
            return {}

        raw_pnl = np.array(self.raw_pnl)
        n_trades = len(raw_pnl)

        hit_rate = (raw_pnl > 0).mean()
        if raw_pnl.std() > 0:
            sharpe = raw_pnl.mean() / raw_pnl.std() * np.sqrt(252)
        else:
            sharpe = 0.0

        pnl_arr    = np.array(self.pnl_series)
        running_max = np.maximum.accumulate(pnl_arr)
        drawdowns   = running_max - pnl_arr
        max_drawdown = drawdowns.max()

        total_return = self.cumulative_pnl
        calmar = total_return / max_drawdown if max_drawdown > 0 else float("inf")

        return {
            "n_trades"      : n_trades,
            "total_pnl"     : self.cumulative_pnl,
            "hit_rate"      : hit_rate,
            "sharpe_ratio"  : sharpe,
            "max_drawdown"  : max_drawdown,
            "calmar_ratio"  : calmar,
            "pnl_series"    : self.pnl_series,
        }


# ── Run Backtest ─────────────────────────────────────────────────────

def run_backtest(probs, targets, threshold, slippage_bps, commission_bps):
    engine = BacktestEngine(threshold, slippage_bps, commission_bps)
    for i in range(len(probs)):
        engine.step(probs[i], targets[i], i)
    return engine.results()


# ── Results Display ──────────────────────────────────────────────────

def print_results(results: dict, threshold: float):
    if not results:
        return

    print("\n" + "=" * 55)
    print("  BACKTEST RESULTS")
    print("=" * 55)
    print(f"  Confidence threshold : {threshold:.2f}")
    print(f"  Total completed trades: {results['n_trades']}")
    print(f"  {'─'*51}")
    print(f"  {'Metric':<22} {'Value':>12}  {'Target':>10}  {'Met':>3}")
    print(f"  {'─'*51}")

    metrics = [
        ("Total PnL",       f"{results['total_pnl']:>+.4f}",    "> 0",      results['total_pnl'] > 0),
        ("Hit Rate",        f"{results['hit_rate']:.1%}",        "≥ 52%",    results['hit_rate'] >= 0.52),
        ("Sharpe Ratio",    f"{results['sharpe_ratio']:.3f}",    "≥ 1.5",    results['sharpe_ratio'] >= 1.5),
        ("Max Drawdown",    f"{results['max_drawdown']:.4f}",    "< 5%",     results['max_drawdown'] < 0.05),
        ("Calmar Ratio",    f"{results['calmar_ratio']:.3f}",    "> 1.0",    results['calmar_ratio'] > 1.0),
    ]

    for name, val, target, met in metrics:
        icon = "✅" if met else "❌"
        print(f"  {name:<22} {val:>12}  {target:>10}  {icon:>3}")

    print("=" * 55)


# ── PnL Plot ─────────────────────────────────────────────────────────

def plot_pnl(pnl_series: list, save_path="pnl_curve.png"):
    pnl = np.array(pnl_series)

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(pnl, color="#2563EB", linewidth=1.2, label="Strategy PnL")
    ax.axhline(0, color="black", linewidth=0.8, linestyle="--")
    ax.fill_between(range(len(pnl)), pnl, 0,
                    where=(pnl >= 0), color="#DCFCE7", alpha=0.5, label="Profit")
    ax.fill_between(range(len(pnl)), pnl, 0,
                    where=(pnl < 0),  color="#FEE2E2", alpha=0.5, label="Loss")

    ax.set_title("Cumulative PnL — LOB Predictor Agent", fontsize=14, fontweight="bold")
    ax.set_xlabel("Test Set Time Steps")
    ax.set_ylabel("Cumulative PnL (simulated units)")
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\n  ✅ PnL curve saved to: {save_path}")


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
    print("   LOB Predictor — Backtesting Engine")
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

    print("  Running inference on test set...")
    probs, targets = get_all_probabilities(model, test_loader, device)

    print(f"  Running backtest (threshold={args.threshold}, "
          f"slippage={args.slippage_bps}bps, commission={args.commission_bps}bps)...")
    results = run_backtest(
        probs, targets,
        threshold=args.threshold,
        slippage_bps=args.slippage_bps,
        commission_bps=args.commission_bps
    )

    if results:
        print_results(results, args.threshold)
        plot_pnl(results["pnl_series"])


if __name__ == "__main__":
    main()
