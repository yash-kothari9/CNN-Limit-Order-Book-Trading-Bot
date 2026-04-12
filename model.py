import torch
import torch.nn as nn
import torch.nn.functional as F


# ── CNN Architecture ─────────────────────────────────────────────────

class LOBPredictorCNN(nn.Module):

    def __init__(self, num_classes: int = 3, dropout_rate: float = 0.5):
        super(LOBPredictorCNN, self).__init__()

        # Block 1: Spatial-Temporal Feature Extraction
        self.block1 = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=(1, 2), stride=(1, 2)),
            nn.LeakyReLU(negative_slope=0.01),
            nn.BatchNorm2d(64),
            nn.Conv2d(64, 64, kernel_size=(4, 1)),
            nn.LeakyReLU(negative_slope=0.01),
            nn.BatchNorm2d(64),
            nn.Conv2d(64, 64, kernel_size=(4, 1)),
            nn.LeakyReLU(negative_slope=0.01),
            nn.BatchNorm2d(64),
        )

        # Block 2: Deep Spatial-Temporal Patterns
        self.block2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=(1, 2), stride=(1, 2)),
            nn.LeakyReLU(negative_slope=0.01),
            nn.BatchNorm2d(128),
            nn.Conv2d(128, 128, kernel_size=(4, 1)),
            nn.LeakyReLU(negative_slope=0.01),
            nn.BatchNorm2d(128),
            nn.Conv2d(128, 128, kernel_size=(4, 1)),
            nn.LeakyReLU(negative_slope=0.01),
            nn.BatchNorm2d(128),
        )

        # Block 3: High-Level Temporal Abstraction
        self.block3 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=(1, 10), stride=(1, 1)),
            nn.LeakyReLU(negative_slope=0.01),
            nn.BatchNorm2d(256),
            nn.Conv2d(256, 256, kernel_size=(4, 1)),
            nn.LeakyReLU(negative_slope=0.01),
            nn.BatchNorm2d(256),
            nn.Conv2d(256, 256, kernel_size=(4, 1)),
            nn.LeakyReLU(negative_slope=0.01),
            nn.BatchNorm2d(256),
        )

        # Global Average Pooling
        self.global_avg_pool = nn.AdaptiveAvgPool2d((1, 1))

        # Classifier Head
        self.flatten = nn.Flatten()
        self.fc = nn.Sequential(
            nn.Dropout(p=dropout_rate),
            nn.Linear(256, 128),
            nn.LeakyReLU(negative_slope=0.01),
            nn.Dropout(p=dropout_rate),
            nn.Linear(128, num_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.global_avg_pool(x)
        x = self.flatten(x)
        x = self.fc(x)
        return x

    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        self.eval()
        with torch.no_grad():
            logits = self.forward(x)
            probs = F.softmax(logits, dim=1)
        return probs


# ── Utilities ────────────────────────────────────────────────────────

def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def model_summary(model: nn.Module):
    print("=" * 55)
    print("   LOB Predictor CNN — Architecture Summary")
    print("=" * 55)
    print(f"  Input shape  : [batch, 1, 100, 40]")
    print(f"  Output shape : [batch, 3]  (DOWN / STAT / UP)")
    print()
    for name, module in model.named_children():
        params = sum(p.numel() for p in module.parameters())
        print(f"  {name:<20} {str(module.__class__.__name__):<20} {params:>8,} params")
    print("-" * 55)
    total = count_parameters(model)
    print(f"  {'TOTAL':<20} {'':20} {total:>8,} params")
    print("=" * 55)


# ── Sanity Check ─────────────────────────────────────────────────────

if __name__ == "__main__":
    model = LOBPredictorCNN(num_classes=3)
    model_summary(model)

    dummy_input = torch.randn(8, 1, 100, 40)
    output = model(dummy_input)
    probs = model.predict_proba(dummy_input)

    print(f"\n  Dummy input shape  : {dummy_input.shape}")
    print(f"  Logits output shape: {output.shape}")
    print(f"  Probabilities shape: {probs.shape}")
    print(f"  Sample probs row 0 : {probs[0].tolist()}")
    print(f"  Sum of probs row 0 : {probs[0].sum().item():.4f}  (should be 1.0)")
    print("\n  ✅ Architecture check passed!")
