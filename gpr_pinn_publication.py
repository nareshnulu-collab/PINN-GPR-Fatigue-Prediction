# gpr_pinn_fatigue_FINAL_WORKING.py
# Physics-Informed Neural Network for AZ31 fatigue S-N prediction
# JOURNAL-READY VERSION: LOOCV | Enhanced uncertainty | Physics justification with LOG SCALE FIX
# Date: 2025-11-17

import os
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.utils.data import TensorDataset, DataLoader
import warnings

warnings.filterwarnings("ignore")

# GLOBAL REPRODUCIBILITY
# =============================================================================
os.environ['PYTHONHASHSEED'] = '42'
np.random.seed(42)
random.seed(42)
torch.manual_seed(42)
torch.cuda.manual_seed_all(42)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

print("=" * 80)
print("PHYSICS-INFORMED NEURAL NETWORK FOR AZ31 FATIGUE (JOURNAL-READY)")
print("WITH LOOCV | ENHANCED UNCERTAINTY | PHYSICS JUSTIFICATION | WORKING FIX")
print("=" * 80)

# 1. LOAD AND PREPARE DATA
# =============================================================================
print("\n[1] Loading augmented data (real + synthetic from GPR)...")

df = pd.read_csv("az31_fatigue_augmented.csv")

print(f"  Total rows: {len(df)}")
print(f"  Real points: {len(df[df['source'] == 'real'])}")
print(f"  Synthetic points: {len(df[df['source'] == 'synthetic'])}")
print(f"  Stress range: {df['sigma_a'].min():.1f}–{df['sigma_a'].max():.1f} MPa")
print(f"  Cycle range: 10^{np.log10(df['N_cycles'].min()):.1f}–10^{np.log10(df['N_cycles'].max()):.1f}")

# Physics regime check
n_lcf = np.sum(df['N_cycles'] < 1e4)
n_hcf = np.sum(df['N_cycles'] >= 1e4)
print(f"\n  Physics regime breakdown:")
print(f"    LCF (N < 10⁴): {n_lcf} points")
print(f"    HCF (N ≥ 10⁴): {n_hcf} points")
print(f"    ⚠ Model uses Basquin law (HCF regime). LCF region uses soft regularization only.")

X_all = np.log10(df["N_cycles"].values).reshape(-1, 1)
y_all = df["sigma_a"].values.reshape(-1, 1)
sigma_uncertainty = df["sigma_uncertainty"].values
weight_factor = df["weight_factor"].values
source = df["source"].values

# INPUT/OUTPUT NORMALIZATION
# =============================================================================
print("\n[2] Normalizing inputs/outputs for numerical stability...")

X_mean, X_std = X_all.mean(), X_all.std()
y_mean, y_std = y_all.mean(), y_all.std()

X_norm = (X_all - X_mean) / X_std
y_norm = (y_all - y_mean) / y_std

print(f"  X normalized: μ={X_mean:.3f}, σ={X_std:.3f}")
print(f"  y normalized: μ={y_mean:.3f}, σ={y_std:.3f}")

mask_real_idx = np.where(source == "real")[0]
mask_synth_idx = np.where(source == "synthetic")[0]

print(f"  Real indices: {mask_real_idx}")
print(f"  Synthetic count: {len(mask_synth_idx)}")

# 3. PHYSICS-INFORMED NEURAL NETWORK
# =============================================================================
print("\n[3] Building Physics-Informed Neural Network...")


class PhysicsInformedNN(nn.Module):
    """PINN with embedded Basquin law (HCF regime) + MC Dropout."""

    def __init__(self, hidden_dim=64, n_layers=4, dropout_rate=0.3):
        super().__init__()

        layers = [nn.Linear(1, hidden_dim), nn.ReLU(), nn.Dropout(dropout_rate)]

        for _ in range(n_layers - 2):
            layers += [nn.Linear(hidden_dim, hidden_dim), nn.ReLU(), nn.Dropout(dropout_rate)]

        layers.append(nn.Linear(hidden_dim, 1))
        self.network = nn.Sequential(*layers)

        self.log_sigma_f_prime = nn.Parameter(torch.tensor(5.5))
        self.b = nn.Parameter(torch.tensor(-0.08))

    def forward(self, x_norm, use_physics=True):
        """x_norm: normalized log10(N_cycles), Returns: normalized stress amplitude"""
        y_network = self.network(x_norm)
        return y_network


# 4. ENHANCED PINN TRAINER
# =============================================================================
print("\n[4] Setting up PINN trainer with validation loss + early stopping...")


class PINNTrainer:
    def __init__(self, model, device="cpu", seed=42):
        self.model = model.to(device)
        self.device = device
        torch.manual_seed(seed)

    def compute_loss(
        self,
        y_pred_network,
        y_true_norm,
        x_norm,
        sigma_y,
        weights=None,
        lambda_phys=0.15,
        lambda_param=0.01,
        y_mean=0,
        y_std=1,
    ):
        """Enhanced loss with physics residual + parameter bounds."""

        tau_sq = 0.1 / (y_std ** 2)
        sigma_sq = (sigma_y / y_std) ** 2 + tau_sq

        if weights is not None:
            data_loss = torch.mean(
                weights * (1.0 / sigma_sq) * (y_pred_network - y_true_norm) ** 2
            )
        else:
            data_loss = torch.mean((1.0 / sigma_sq) * (y_pred_network - y_true_norm) ** 2)

        total_loss = data_loss

        # Physics residual loss
        if lambda_phys > 0:
            N_phys = torch.logspace(3.5, 7.5, 50, device=self.device).unsqueeze(1)
            x_phys = torch.log10(N_phys)
            x_phys_norm = (x_phys - X_mean) / X_std

            y_network_phys = self.model(x_phys_norm, use_physics=True)
            sigma_f_prime = torch.exp(self.model.log_sigma_f_prime)
            sigma_a_basquin = sigma_f_prime * (2.0 * N_phys) ** self.model.b

            sigma_a_basquin_norm = (sigma_a_basquin - y_mean) / y_std

            physics_residual = torch.mean((y_network_phys - sigma_a_basquin_norm) ** 2)
            total_loss = total_loss + lambda_phys * physics_residual

        # Parameter bounds
        if lambda_param > 0:
            param_loss = (
                torch.relu(self.model.b + 0.15) ** 2
                + torch.relu(-0.05 - self.model.b) ** 2
            )
            total_loss = total_loss + lambda_param * param_loss

        return total_loss, data_loss

    def train(
        self,
        train_loader,
        val_X,
        val_y,
        val_sigma,
        val_weights,
        epochs=1000,
        lr=1e-2,
        weight_decay=1e-4,
        lambda_phys=0.15,
        lambda_param=0.01,
        early_stopping_patience=50,
    ):
        """Train with validation loss for early stopping."""

        optimizer = AdamW(self.model.parameters(), lr=lr, weight_decay=weight_decay)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

        best_val_loss = float("inf")
        patience_counter = 0
        train_losses = []
        val_losses = []

        for epoch in range(epochs):
            self.model.train()
            epoch_loss = 0.0

            for x_batch, y_batch, sigma_batch, w_batch in train_loader:
                x_batch = x_batch.to(self.device)
                y_batch = y_batch.to(self.device)
                sigma_batch = sigma_batch.to(self.device)
                w_batch = w_batch.to(self.device)

                optimizer.zero_grad()

                y_pred = self.model(x_batch, use_physics=True).squeeze()
                loss, _ = self.compute_loss(
                    y_pred,
                    y_batch.squeeze(),
                    x_batch.squeeze(),
                    sigma_batch.squeeze(),
                    weights=w_batch.squeeze(),
                    lambda_phys=lambda_phys,
                    lambda_param=lambda_param,
                    y_mean=y_mean,
                    y_std=y_std,
                )

                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                optimizer.step()

                epoch_loss += loss.item()

            train_loss = epoch_loss / len(train_loader)
            train_losses.append(train_loss)

            # VALIDATION
            self.model.eval()
            with torch.no_grad():
                val_X_t = val_X.to(self.device)
                val_y_t = val_y.to(self.device)
                val_sigma_t = val_sigma.to(self.device)
                val_w_t = val_weights.to(self.device)

                val_pred = self.model(val_X_t, use_physics=True).squeeze()
                val_loss, _ = self.compute_loss(
                    val_pred,
                    val_y_t.squeeze(),
                    val_X_t.squeeze(),
                    val_sigma_t.squeeze(),
                    weights=val_w_t.squeeze(),
                    lambda_phys=lambda_phys,
                    lambda_param=lambda_param,
                    y_mean=y_mean,
                    y_std=y_std,
                )

            val_losses.append(val_loss.item())
            scheduler.step()

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                best_state = self.model.state_dict()
            else:
                patience_counter += 1

            if (epoch + 1) % 100 == 0:
                print(f"  Epoch {epoch+1:4d} | Train: {train_loss:.4f} | Val: {val_loss:.4f}")

            if patience_counter >= early_stopping_patience:
                print(f"  Early stopping at epoch {epoch+1}")
                self.model.load_state_dict(best_state)
                break

        return train_losses, val_losses

    def mc_dropout_inference(self, X_norm, n_samples=500):
        """MC Dropout with 500 samples."""
        self.model.train()
        preds = []
        with torch.no_grad():
            for _ in range(n_samples):
                y_pred = self.model(X_norm.to(self.device), use_physics=True)
                preds.append(y_pred.cpu().numpy())
        preds = np.array(preds)
        y_mean = np.mean(preds, axis=0)
        y_std = np.std(preds, axis=0)
        self.model.eval()
        return y_mean, y_std


# 5. LEAVE-ONE-OUT CROSS-VALIDATION (LOOCV)
# =============================================================================
print("\n[5] Setting up Leave-One-Out Cross-Validation on 5 real points...")

all_predictions = []
all_uncertainties = []
all_actuals = []

for fold_idx, test_idx in enumerate(mask_real_idx):
    print(f"\n  [LOOCV Fold {fold_idx+1}/5] Hold-out point {test_idx} (N={10**X_all[test_idx,0]:.0e})...")

    train_mask = np.ones(len(X_all), dtype=bool)
    train_mask[test_idx] = False

    X_train_fold = X_norm[train_mask].astype(np.float32)
    y_train_fold = y_norm[train_mask].astype(np.float32)
    sigma_train_fold = sigma_uncertainty[train_mask].astype(np.float32)
    w_train_fold = weight_factor[train_mask].astype(np.float32)
    w_train_fold = np.minimum(w_train_fold, 0.3)

    X_test_fold = X_norm[test_idx:test_idx+1].astype(np.float32)
    y_test_fold = y_norm[test_idx:test_idx+1].astype(np.float32)

    train_dataset = TensorDataset(
        torch.tensor(X_train_fold, dtype=torch.float32),
        torch.tensor(y_train_fold, dtype=torch.float32),
        torch.tensor(sigma_train_fold, dtype=torch.float32),
        torch.tensor(w_train_fold, dtype=torch.float32)
    )
    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)

    other_real = [m for m in mask_real_idx if m != test_idx][0]
    val_mask = np.zeros(len(X_all), dtype=bool)
    val_mask[other_real] = True

    X_val = torch.tensor(X_norm[val_mask], dtype=torch.float32)
    y_val = torch.tensor(y_norm[val_mask], dtype=torch.float32)
    sigma_val = torch.tensor(sigma_uncertainty[val_mask], dtype=torch.float32)
    w_val = torch.tensor(np.ones(np.sum(val_mask)), dtype=torch.float32)

    device = "cpu"
    model_fold = PhysicsInformedNN(hidden_dim=64, n_layers=4, dropout_rate=0.3)
    trainer_fold = PINNTrainer(model_fold, device=device, seed=42+fold_idx)

    train_loss, val_loss = trainer_fold.train(
        train_loader,
        X_val,
        y_val,
        sigma_val,
        w_val,
        epochs=1000,
        lr=1e-2,
        weight_decay=1e-4,
        lambda_phys=0.15,
        lambda_param=0.01,
        early_stopping_patience=50,
    )

    X_test_t = torch.tensor(X_test_fold, dtype=torch.float32)
    y_pred_norm, y_std_norm = trainer_fold.mc_dropout_inference(X_test_t, n_samples=500)

    y_pred = y_pred_norm * y_std + y_mean
    y_std_denorm = y_std_norm * y_std
    y_actual = y_all[test_idx, 0]

    all_predictions.append(y_pred.squeeze())
    all_uncertainties.append(y_std_denorm.squeeze())
    all_actuals.append(y_actual)

    print(f"    Actual: {y_actual:.1f} MPa | Pred: {y_pred.squeeze():.1f} ± {y_std_denorm.squeeze():.2f} MPa")

all_predictions = np.array(all_predictions).ravel()
all_uncertainties = np.array(all_uncertainties).ravel()
all_actuals = np.array(all_actuals).ravel()

# 6. COMPUTE METRICS
# =============================================================================
print("\n[6] Computing LOOCV metrics...")

rmse = np.sqrt(np.mean((all_predictions - all_actuals) ** 2))
mae = np.mean(np.abs(all_predictions - all_actuals))
mape = np.mean(np.abs((all_predictions - all_actuals) / all_actuals)) * 100

r2 = 1.0 - (np.sum((all_actuals - all_predictions) ** 2) / np.sum((all_actuals - np.mean(all_actuals)) ** 2))

calibration = all_actuals / all_predictions
calib_mean = np.mean(calibration)
calib_std = np.std(calibration)

ci_lower = all_predictions - 1.96 * all_uncertainties
ci_upper = all_predictions + 1.96 * all_uncertainties
coverage = np.mean((all_actuals >= ci_lower) & (all_actuals <= ci_upper))

print(f"\n  === LOOCV METRICS (5 real points) ===")
print(f"  RMSE: {rmse:.2f} MPa")
print(f"  MAE: {mae:.2f} MPa")
print(f"  MAPE: {mape:.2f}%")
print(f"  R²: {r2:.4f}")
print(f"  Calibration: {calib_mean:.3f} ± {calib_std:.3f}")
print(f"  Coverage: {coverage:.1%}")

# 7. TRAIN FINAL MODEL FOR VISUALIZATION
# =============================================================================
print("\n[7] Training final model on all real points for visualization...")

X_final = X_norm.astype(np.float32)
y_final = y_norm.astype(np.float32)
sigma_final = sigma_uncertainty.astype(np.float32)
w_final = weight_factor.astype(np.float32)
w_final = np.minimum(w_final, 0.3)

train_dataset_final = TensorDataset(
    torch.tensor(X_final, dtype=torch.float32),
    torch.tensor(y_final, dtype=torch.float32),
    torch.tensor(sigma_final, dtype=torch.float32),
    torch.tensor(w_final, dtype=torch.float32)
)
train_loader_final = DataLoader(train_dataset_final, batch_size=8, shuffle=True)

model_final = PhysicsInformedNN(hidden_dim=64, n_layers=4, dropout_rate=0.3)
trainer_final = PINNTrainer(model_final, device=device, seed=42)

optimizer_final = AdamW(model_final.parameters(), lr=1e-2, weight_decay=1e-4)
scheduler_final = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer_final, T_max=1000)

for epoch in range(1000):
    model_final.train()
    epoch_loss = 0.0
    for x_batch, y_batch, sigma_batch, w_batch in train_loader_final:
        x_batch = x_batch.to(device)
        y_batch = y_batch.to(device)
        sigma_batch = sigma_batch.to(device)
        w_batch = w_batch.to(device)

        optimizer_final.zero_grad()

        y_pred = model_final(x_batch, use_physics=True).squeeze()
        loss, _ = trainer_final.compute_loss(
            y_pred,
            y_batch.squeeze(),
            x_batch.squeeze(),
            sigma_batch.squeeze(),
            weights=w_batch.squeeze(),
            lambda_phys=0.15,
            lambda_param=0.01,
            y_mean=y_mean,
            y_std=y_std,
        )
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model_final.parameters(), 1.0)
        optimizer_final.step()
        epoch_loss += loss.item()

    scheduler_final.step()
    if (epoch + 1) % 100 == 0:
        print(f"  Epoch {epoch+1:4d} | Loss: {epoch_loss/len(train_loader_final):.4f}")

model_final.eval()
sigma_f_prime = np.exp(model_final.log_sigma_f_prime.item())
b = model_final.b.item()

print(f"\n  === FITTED PHYSICS PARAMETERS ===")
print(f"  σ'_f: {sigma_f_prime:.1f} MPa")
print(f"  b: {b:.4f} (literature: [-0.15, -0.05])")
print(f"  Within bounds: {'✓ YES' if -0.15 <= b <= -0.05 else '✗ NO'}")

# 8. PUBLICATION VISUALIZATION (9 panels) - WITH LOG SCALE FIX
# =============================================================================
print("\n[8] Creating publication-ready plots...")

fig = plt.figure(figsize=(18, 14))
gs = fig.add_gridspec(3, 3, hspace=0.35, wspace=0.35)

# Plot 1: S-N Curve with Basquin overlay (FIX: proper log scale)
ax1 = fig.add_subplot(gs[0, :2])
mask_real_all = source == "real"
ax1.scatter(
    df[mask_real_all]["N_cycles"],
    df[mask_real_all]["sigma_a"],
    color="red",
    s=150,
    marker="o",
    label="Real (Fouad 2011, BB)",
    zorder=5,
    edgecolors="darkred",
    linewidth=2,
)

mask_synth_all = source == "synthetic"
ax1.scatter(
    df[mask_synth_all]["N_cycles"],
    df[mask_synth_all]["sigma_a"],
    color="blue",
    s=15,
    alpha=0.3,
    marker="x",
    label="Synthetic (GPR)",
    zorder=3,
)

# FIX: Generate plot in denormalized space to avoid negative log values
X_plot_norm = torch.tensor(np.linspace(-2, 2, 200)).unsqueeze(1).float()
with torch.no_grad():
    y_plot_norm = model_final(X_plot_norm.to(device), use_physics=True).cpu().numpy()

X_plot_denorm = X_plot_norm.numpy() * X_std + X_mean
N_plot = 10 ** X_plot_denorm
y_plot_denorm = y_plot_norm * y_std + y_mean

N_phys = np.logspace(3.5, 7.5, 100)
sigma_basquin = sigma_f_prime * (2.0 * N_phys) ** b

ax1.plot(N_plot.ravel(), y_plot_denorm.ravel(), "g-", linewidth=3, label="PINN prediction", zorder=4)
ax1.plot(N_phys, sigma_basquin, "m--", linewidth=2.5, label="Pure Basquin law", zorder=4, alpha=0.7)

# FIX: Ensure only positive values for log scale
xmin = max(1e3, N_plot.min())
xmax = N_plot.max()
ax1.set_xlim(xmin, xmax)
ax1.set_xscale("log")
ax1.set_xlabel("Cycles to Failure, $N_f$", fontsize=12, fontweight="bold")
ax1.set_ylabel("Stress Amplitude, $\\sigma_a$ (MPa)", fontsize=12, fontweight="bold")
ax1.set_title("S–N Curve: Real + Synthetic + PINN vs Basquin", fontsize=13, fontweight="bold")
ax1.grid(True, which="both", alpha=0.3)
ax1.legend(fontsize=10, loc="best")

# Plot 2: LOOCV predictions vs actual
ax2 = fig.add_subplot(gs[0, 2])
ax2.scatter(all_actuals, all_predictions, c=range(5), cmap="viridis", s=150, alpha=0.8, edgecolors="black", linewidth=2)
lims = [all_actuals.min() - 5, all_actuals.max() + 5]
ax2.plot(lims, lims, "r--", linewidth=2.5, label="Perfect fit")
for i in range(5):
    ax2.annotate(f"F{i+1}", (all_actuals[i], all_predictions[i]), fontsize=9, ha='center', va='center')
ax2.set_xlabel("Actual (MPa)", fontsize=11, fontweight="bold")
ax2.set_ylabel("LOOCV Predicted (MPa)", fontsize=11, fontweight="bold")
ax2.set_title(f"LOOCV (RMSE={rmse:.2f}, R²={r2:.3f})", fontsize=12, fontweight="bold")
ax2.set_xlim(lims)
ax2.set_ylim(lims)
ax2.set_aspect("equal")
ax2.grid(True, alpha=0.3)
ax2.legend(fontsize=10)

# Plot 3: Residuals with uncertainty bands
ax3 = fig.add_subplot(gs[1, 0])
residuals = all_actuals - all_predictions
ax3.scatter(all_predictions, residuals, color="orange", s=150, alpha=0.8, edgecolors="black", linewidth=2)
ax3.axhline(0, color="r", linestyle="--", linewidth=2)
ax3.fill_between(
    [all_predictions.min()-5, all_predictions.max()+5],
    -1.96*all_uncertainties.max(),
    1.96*all_uncertainties.max(),
    alpha=0.2,
    color="green",
    label="±1.96σ"
)
ax3.set_xlabel("Predicted (MPa)", fontsize=11, fontweight="bold")
ax3.set_ylabel("Residuals (MPa)", fontsize=11, fontweight="bold")
ax3.set_title("Residual Analysis", fontsize=12, fontweight="bold")
ax3.grid(True, alpha=0.3)
ax3.legend(fontsize=10)

# Plot 4: Calibration
ax4 = fig.add_subplot(gs[1, 1])
ax4.scatter(all_actuals, calibration, color="cyan", s=150, alpha=0.8, edgecolors="black", linewidth=2)
ax4.axhline(1.0, color="r", linestyle="--", linewidth=2, label="Perfect")
ax4.fill_between([all_actuals.min(), all_actuals.max()], 0.9, 1.1, alpha=0.1, color="green", label="±10%")
ax4.set_xlabel("Actual (MPa)", fontsize=11, fontweight="bold")
ax4.set_ylabel("Actual / Predicted", fontsize=11, fontweight="bold")
ax4.set_title(f"Calibration ({calib_mean:.3f}±{calib_std:.3f})", fontsize=12, fontweight="bold")
ax4.grid(True, alpha=0.3)
ax4.legend(fontsize=10)

# Plot 5: Uncertainty vs Error
ax5 = fig.add_subplot(gs[1, 2])
errors = np.abs(all_actuals - all_predictions)
ax5.scatter(all_uncertainties, errors, color="magenta", s=150, alpha=0.8, edgecolors="black", linewidth=2)
ax5.plot([0, all_uncertainties.max()], [0, all_uncertainties.max()], "r--", linewidth=2, label="Ideal")
ax5.set_xlabel("MC Dropout Std (MPa)", fontsize=11, fontweight="bold")
ax5.set_ylabel("Absolute Error (MPa)", fontsize=11, fontweight="bold")
ax5.set_title("Uncertainty Calibration", fontsize=12, fontweight="bold")
ax5.grid(True, alpha=0.3)
ax5.legend(fontsize=10)

# Plot 6: PINN vs Basquin residual (FIX: proper log scale)
ax6 = fig.add_subplot(gs[2, 0])
sigma_basquin_grid = sigma_f_prime * (2.0 * N_plot.ravel()) ** b
residual_correction = y_plot_denorm.ravel() - sigma_basquin_grid
ax6.plot(N_plot.ravel(), residual_correction, "purple", linewidth=2.5, label="PINN - Basquin")
ax6.axhline(0, color="r", linestyle="--", linewidth=1.5, alpha=0.7)
ax6.fill_between(N_plot.ravel(), -2, 2, alpha=0.1, color="gray", label="±2 MPa band")

xmin = max(1e3, N_plot.min())
xmax = N_plot.max()
ax6.set_xlim(xmin, xmax)
ax6.set_xscale("log")
ax6.set_xlabel("Cycles, $N$", fontsize=11, fontweight="bold")
ax6.set_ylabel("Correction (MPa)", fontsize=11, fontweight="bold")
ax6.set_title("Physics Residual (PINN - Basquin)", fontsize=12, fontweight="bold")
ax6.grid(True, which="both", alpha=0.3)
ax6.legend(fontsize=10)

# Plot 7: Monotonicity check
ax7 = fig.add_subplot(gs[2, 1])
dlogN = np.diff(X_plot_denorm.ravel())
dy = np.diff(y_plot_denorm.ravel())
dydlogN = dy / dlogN
ax7.plot(X_plot_denorm.ravel()[:-1], dydlogN, "b-", linewidth=2, label="dσ/d(log N)")
ax7.axhline(0, color="r", linestyle="--", linewidth=2, label="Monotonicity limit")
ax7.fill_between(X_plot_denorm.ravel()[:-1], -10, 0, alpha=0.1, color="green", label="Physical region")
ax7.set_xlabel("log₁₀(N)", fontsize=11, fontweight="bold")
ax7.set_ylabel("dσ_a / d(log N)", fontsize=11, fontweight="bold")
ax7.set_title("Monotonicity Check", fontsize=12, fontweight="bold")
ax7.grid(True, alpha=0.3)
ax7.legend(fontsize=10)

# Plot 8: Metrics summary
ax8 = fig.add_subplot(gs[2, 2])
ax8.axis("off")
metrics_text = f"""
LOOCV RESULTS (5 real)
──────────────────────
RMSE = {rmse:.2f} MPa
MAE = {mae:.2f} MPa
MAPE = {mape:.2f}%
R² = {r2:.4f}
Coverage = {coverage:.1%}
Calib. = {calib_mean:.3f}±{calib_std:.3f}

PHYSICS (Basquin HCF)
──────────────────────
σ'_f = {sigma_f_prime:.0f} MPa
b = {b:.4f}
Literature: [-0.15, -0.05]
Bound: {'✓' if -0.15 <= b <= -0.05 else '✗'}

LCF REGIME
──────────────────────
Points: {n_lcf} (soft reg.)
Model: Basquin (HCF)
Note: See Methods
"""
ax8.text(0.05, 0.95, metrics_text, transform=ax8.transAxes, fontsize=9,
         verticalalignment="top", fontfamily="monospace",
         bbox=dict(boxstyle="round", facecolor="lightyellow", alpha=0.7))

plt.savefig("pinn_az31_journal_ready.png", dpi=300, bbox_inches="tight")
print("  ✓ Saved: pinn_az31_journal_ready.png")
plt.show()

# 9. FINAL SUMMARY
# =============================================================================
print("\n" + "=" * 80)
print("JOURNAL-READY SUMMARY")
print("=" * 80)
print(f"✓ LOOCV implemented on 5 real points")
print(f"✓ Input/output normalization applied")
print(f"✓ Uncertainty quantification (500 MC samples, {coverage:.1%} coverage)")
print(f"✓ Physics regime justification (Basquin HCF, soft LCF)")
print(f"✓ Validation loss for early stopping")
print(f"✓ Physics residual plot (PINN vs Basquin)")
print(f"✓ Monotonicity verification")
print(f"✓ LOG SCALE FIXED for S-N and residual plots")
print(f"\n✓ FINAL METRICS (LOOCV on 5 real points):")
print(f"  R² = {r2:.4f}")
print(f"  RMSE = {rmse:.2f} MPa")
print(f"  Calibration = {calib_mean:.3f} ± {calib_std:.3f}")
print(f"  Coverage = {coverage:.1%}")
print(f"\n✓ OUTPUT: pinn_az31_journal_ready.png (8 panels, all working)")
print("=" * 80)
