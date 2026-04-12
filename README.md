# High-Frequency Trading Bot — Deep Learning on Limit Order Books

An end-to-end deep learning pipeline for predicting mid-price movements from Level-2 Limit Order Book (LOB) data, built on the **FI-2010 benchmark dataset**. The system includes model training, rigorous out-of-sample backtesting with realistic transaction costs, and a real-time browser-based simulation dashboard.

[![Python](https://img.shields.io/badge/Python-3.9%2B-blue)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-orange)](https://pytorch.org)
[![Dataset](https://img.shields.io/badge/Dataset-FI--2010-green)](https://www.kaggle.com/datasets/freemanone/fi2010)

---

## Overview

The model is a **9-layer 1D Convolutional Neural Network (CNN)** with ~1.07 million parameters that processes rolling windows of 100 consecutive order book snapshots (each with 40 features across 10 bid/ask levels) and outputs a 3-class probability distribution: **DOWN**, **STATIONARY**, or **UP**.

A threshold-based trading agent converts these probabilities into actionable BUY, SELL, HOLD, or EXIT signals, with positions evaluated against a cost model that includes slippage (0.75 bps) and exchange commissions (0.20 bps) per leg.

---

## Results

### Classification (Test Set — Day 10, Unseen)

| Metric         | Value  |
|----------------|--------|
| Accuracy       | 62.4%  |
| Weighted F1    | 0.57   |

### Backtest (55,379 sequential market states)

| Metric           | Value    |
|------------------|----------|
| Completed Trades | 1,153    |
| Hit Rate         | 54.0%    |
| Cumulative PnL   | +$17.90  |
| Sharpe Ratio     | 3.14     |

---

## Dataset

This project uses the **FI-2010 Benchmark Dataset** for Limit Order Book mid-price prediction.

**Download the dataset from Kaggle:**

> [https://www.kaggle.com/datasets/freemanone/fi2010](https://www.kaggle.com/datasets/freemanone/fi2010)

After downloading, extract the contents into a `data/` directory at the project root. The data pipeline auto-discovers the nested directory structure (`NoAuction/1.NoAuction_Zscore/...`), so no manual restructuring is needed.

Only the **NoAuction Z-Score Cross-Fold 7** subset is used:
- **Training + Validation:** `Train_Dst_NoAuction_ZScore_CF_7.txt` (Days 1–9, 80/20 temporal split)
- **Testing:** `Test_Dst_NoAuction_ZScore_CF_7.txt` (Day 10)

> **Note:** The full extracted dataset is ~30 GB due to redundant normalization variants and cross-fold copies. Only ~840 MB is actually consumed by this project.

---

## Setup

```bash
git clone https://github.com/YOUR_USERNAME/HFT_Trading_Bot.git
cd HFT_Trading_Bot
pip install -r requirements.txt
```

---

## Usage

### 1. Train the Model

```bash
python train.py --data_dir data/ --epochs 50
```

The best model checkpoint is saved to `best_model.pt` based on validation F1-score with early stopping (patience=15).

### 2. Evaluate

```bash
python evaluate.py --model best_model.pt --data_dir data/
```

Generates: `confusion_matrix.png`, `training_curves.png`, `prob_distribution.png`

### 3. Backtest

```bash
python backtest.py --model best_model.pt --data_dir data/ --threshold 0.65
```

Generates: `pnl_curve.png`

### 4. Run Live Simulation Dashboard

```bash
python prepare_simulation.py
```

Then open `simulation.html` in a browser. The dashboard visualises the model stepping through all 55,379 test states with a live order book, CNN output probabilities, trade log, and cumulative PnL chart.

---

## Project Structure

```
HFT_Trading_Bot/
├── model.py                 # 9-layer CNN architecture (~1.07M parameters)
├── dataset.py               # FI-2010 data loading, windowing, label mapping
├── train.py                 # Training loop with AdamW, early stopping, LR scheduling
├── evaluate.py              # Classification metrics, confusion matrix, training curves
├── backtest.py              # Event-driven backtesting engine with transaction cost model
├── prepare_simulation.py    # Generates simulation_data.js for the browser dashboard
├── simulation.html          # Real-time browser-based simulation dashboard
├── requirements.txt         # Python dependencies
├── paper.tex                # IEEE journal article (LaTeX)
└── data/                    # FI-2010 dataset (not included — download from Kaggle)
```

---

## Architecture

```
Input: [batch, 1, 100, 40]

Block 1 — Conv2d(1→64)  × 3 layers    Spatial feature extraction
Block 2 — Conv2d(64→128) × 3 layers   Deep temporal patterns
Block 3 — Conv2d(128→256) × 3 layers  High-level abstraction

→ AdaptiveAvgPool2d(1,1) → Flatten → Dropout(0.5) → FC(256→128→3)

Output: [P(DOWN), P(STATIONARY), P(UP)]
```

Each convolutional layer uses LeakyReLU activation and Batch Normalization. The classifier head applies dropout regularization before the fully connected layers.

---

## Cost Model

Transaction costs are modeled per leg:
- **Slippage:** 0.75 basis points
- **Commission:** 0.20 basis points
- **Round-trip cost:** 1.90 bps (~$0.019 on a $100 asset)

The model must predict a movement of at least 2 ticks ($0.02) to break even on any trade.

---

## Key Design Decisions

- **Temporal Split:** Train on Days 1–7, validate on Days 8–9, test on Day 10. Random splitting is never used on time-series data to avoid look-ahead bias.
- **Class Weighting:** Inverse-frequency weighted CrossEntropyLoss to counteract STATIONARY class dominance.
- **Confidence Threshold:** Trades are only executed when model confidence exceeds 65%, filtering out noisy low-conviction predictions.

---

## References

1. Ntakaris, A., et al. (2018). *Benchmark Dataset for Mid-Price Forecasting of Limit Order Book Data with Machine Learning Methods.* Journal of Forecasting, 37(8), 852–866.
2. Sirignano, J. & Cont, R. (2019). *Universal Features of Price Formation in Financial Markets: Perspectives from Deep Learning.* Quantitative Finance, 19(9), 1449–1459.
3. Zhang, Z., Zohren, S., & Roberts, S. (2019). *DeepLOB: Deep Learning for Limit Order Books.* IEEE Transactions on Signal Processing, 67(11), 3001–3012.

---

*This project is for academic and research purposes only. It does not constitute financial advice.*
