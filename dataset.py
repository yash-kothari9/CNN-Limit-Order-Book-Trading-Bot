import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader


# ── Constants ────────────────────────────────────────────────────────

NUM_FEATURES = 40
SEQUENCE_LEN = 100
NUM_CLASSES = 3
LABEL_MAP = {1: 0, 2: 1, 3: 2}
HORIZON_INDEX = 4


# ── Dataset ──────────────────────────────────────────────────────────

class LOBDataset(Dataset):

    def __init__(self, data: np.ndarray, labels: np.ndarray):
        self.data = torch.tensor(data, dtype=torch.float32)
        self.labels = torch.tensor(labels, dtype=torch.long)
        self.num_samples = len(self.data) - SEQUENCE_LEN + 1

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        x = self.data[idx : idx + SEQUENCE_LEN]
        x = x.unsqueeze(0)
        y = self.labels[idx + SEQUENCE_LEN - 1]
        return x, y


# ── FI-2010 Data Loading ─────────────────────────────────────────────

def load_fi2010(data_dir: str):
    import os

    print(f"Loading FI-2010 data from: {data_dir}")

    train_candidates = [
        (os.path.join(data_dir, "Train_Dst_NoAuction_ZScore_CF_7.npy"), "npy"),
        (os.path.join(data_dir, "Train_Dst_NoAuction_ZScore_CF_7.txt"), "txt"),
        (os.path.join(data_dir, "NoAuction", "1.NoAuction_Zscore",
                      "NoAuction_Zscore_Training",
                      "Train_Dst_NoAuction_ZScore_CF_7.txt"), "txt"),
        (os.path.join(data_dir, "NoAuction", "1.NoAuction_Zscore",
                      "NoAuction_Zscore_Training",
                      "Train_Dst_NoAuction_ZScore_CF_7.npy"), "npy"),
    ]
    test_candidates = [
        (os.path.join(data_dir, "Test_Dst_NoAuction_ZScore_CF_7.npy"), "npy"),
        (os.path.join(data_dir, "Test_Dst_NoAuction_ZScore_CF_7.txt"), "txt"),
        (os.path.join(data_dir, "NoAuction", "1.NoAuction_Zscore",
                      "NoAuction_Zscore_Testing",
                      "Test_Dst_NoAuction_ZScore_CF_7.txt"), "txt"),
        (os.path.join(data_dir, "NoAuction", "1.NoAuction_Zscore",
                      "NoAuction_Zscore_Testing",
                      "Test_Dst_NoAuction_ZScore_CF_7.npy"), "npy"),
    ]

    def find_file(candidates, label):
        for path, fmt in candidates:
            if os.path.exists(path):
                print(f"  Found {label}: {path}  (format: {fmt})")
                return path, fmt
        tried = "\n    ".join(p for p, _ in candidates)
        raise FileNotFoundError(
            f"Could not find {label} file. Searched:\n    {tried}\n"
            "Please download FI-2010 from:\n"
            "https://www.kaggle.com/datasets/freemanone/fi2010\n"
        )

    train_file, train_fmt = find_file(train_candidates, "training data")
    test_file,  test_fmt  = find_file(test_candidates,  "testing data")

    def load_data(path, fmt):
        if fmt == "npy":
            return np.load(path)
        else:
            return np.loadtxt(path)

    print("  Loading training data (this may take a moment for .txt files)...")
    train_raw = load_data(train_file, train_fmt)
    print("  Loading testing data...")
    test_raw  = load_data(test_file, test_fmt)
    print(f"  Train raw shape: {train_raw.shape}")
    print(f"  Test  raw shape: {test_raw.shape}")

    # Auto-detect label layout
    def get_label_row_index(raw_data):
        candidate_npy = NUM_FEATURES + HORIZON_INDEX
        candidate_txt = raw_data.shape[0] - 5 + HORIZON_INDEX
        unique_vals = np.unique(raw_data[candidate_npy, :])
        if set(unique_vals.astype(int)).issubset({1, 2, 3}) and len(unique_vals) <= 3:
            print(f"  Label layout: .npy-style (labels at row {candidate_npy})")
            return candidate_npy
        print(f"  Label layout: .txt-style (labels at row {candidate_txt})")
        return candidate_txt

    train_label_idx = get_label_row_index(train_raw)
    test_label_idx  = get_label_row_index(test_raw)

    train_features   = train_raw[:NUM_FEATURES, :].T
    train_labels_raw = train_raw[train_label_idx, :]
    test_features    = test_raw[:NUM_FEATURES, :].T
    test_labels_raw  = test_raw[test_label_idx, :]

    train_labels = np.vectorize(LABEL_MAP.get)(train_labels_raw.astype(int))
    test_labels  = np.vectorize(LABEL_MAP.get)(test_labels_raw.astype(int))

    # Temporal split: last 20% of training data as validation
    split_idx = int(len(train_features) * 0.8)
    val_features   = train_features[split_idx:]
    val_labels     = train_labels[split_idx:]
    train_features = train_features[:split_idx]
    train_labels   = train_labels[:split_idx]

    print(f"  Train snapshots : {len(train_features):,}")
    print(f"  Val snapshots   : {len(val_features):,}")
    print(f"  Test snapshots  : {len(test_features):,}")

    for name, lbls in [("Train", train_labels), ("Val", val_labels), ("Test", test_labels)]:
        counts = np.bincount(lbls, minlength=3)
        total  = len(lbls)
        print(f"  {name} class dist → DOWN:{counts[0]/total:.1%}  "
              f"STAT:{counts[1]/total:.1%}  UP:{counts[2]/total:.1%}")

    return (
        LOBDataset(train_features, train_labels),
        LOBDataset(val_features,   val_labels),
        LOBDataset(test_features,  test_labels),
    )


# ── Synthetic Demo Data ──────────────────────────────────────────────

def make_demo_data(n_train=5000, n_val=1000, n_test=1000):
    print("⚠️  DEMO MODE: Using synthetic data (no real signal)")
    print("   For real results, download FI-2010 and use --data_dir\n")

    rng = np.random.default_rng(42)

    def make_split(n):
        features = rng.standard_normal((n, NUM_FEATURES)).astype(np.float32)
        labels   = rng.integers(0, NUM_CLASSES, size=n)
        return LOBDataset(features, labels)

    return make_split(n_train), make_split(n_val), make_split(n_test)


# ── DataLoader Factory ───────────────────────────────────────────────

def get_dataloaders(train_ds, val_ds, test_ds, batch_size=32):
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,  num_workers=0)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False, num_workers=0)
    test_loader  = DataLoader(test_ds,  batch_size=batch_size, shuffle=False, num_workers=0)
    return train_loader, val_loader, test_loader


# ── Class Weights ────────────────────────────────────────────────────

def compute_class_weights(dataset: LOBDataset) -> torch.Tensor:
    labels = dataset.labels.numpy()
    counts = np.bincount(labels, minlength=NUM_CLASSES).astype(float)
    weights = len(labels) / (NUM_CLASSES * counts)
    return torch.tensor(weights, dtype=torch.float32)
