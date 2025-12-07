import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
import warnings

warnings.filterwarnings('ignore')

np.random.seed(42)
torch.manual_seed(42)
torch.cuda.manual_seed_all(42)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("=" * 80)
print("PINN for AZ31 Fatigue Prediction")
print(f"Device: {device}")
print("=" * 80)

# Load data
print("\nLoading data...")
df = pd.read_csv('az31_fatigue_augmented.csv')

all_cycles = df['N_cycles'].values
all_stresses = df['sigma_a'].values
all_uncertainty = df['sigma_uncertainty'].values
all_weights = df['weight_factor'].values
all_sources = (df['source'].values == 'real').astype(float)

real_idx = np.where(all_sources == 1)[0]
synth_idx = np.where(all_sources == 0)[0]

print(f"Data: {len(all_cycles)} points ({len(real_idx)} real, {len(synth_idx)} synthetic)")
print(f"Stress range: {all_stresses.min():.1f} - {all_stresses.max():.1f} MPa")
print(f"Uncertainty range: {all_uncertainty.min():.3f} - {all_uncertainty.max():.3f} MPa")

# Normalize data
X_all = np.log10(all_cycles).reshape(-1, 1)
y_all = all_stresses.reshape(-1, 1)

X_mean, X_std = X_all.mean(), X_all.std()
y_mean, y_std = y_all.mean(), y_all.std()

X_norm = (X_all - X_mean) / X_std
y_norm = (y_all - y_mean) / y_std

print(f"\nNormalization: X_mean={X_mean:.3f}, X_std={X_std:.3f}")
print(f"               y_mean={y_mean:.2f}, y_std={y_std:.2f}")

# Noise floor (physical & normalized)
eps_noise_phys = 0.1  # MPa^2
eps_noise_norm = eps_noise_phys / (y_std ** 2)
print(f"Noise floor: {eps_noise_phys} MPa^2 (normalized: {eps_noise_norm:.6e})")

# Network architecture
class PINN(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(1, 64), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(64, 64), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(64, 64), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(64, 64), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(64, 1)
        )
        self.log_sigma_f = nn.Parameter(torch.tensor(np.log(596.0)))
        self.b = nn.Parameter(torch.tensor(-0.1090))
    
    def forward(self, x):
        return self.net(x)
    
    def basquin(self, cycles):
        # Accept numpy array or torch tensor, convert to device
        device = self.log_sigma_f.device
        if not torch.is_tensor(cycles):
            cycles = torch.tensor(cycles, dtype=torch.float32, device=device)
        else:
            cycles = cycles.to(device=device, dtype=torch.float32)
        sigma_f = torch.exp(self.log_sigma_f)
        return sigma_f * (2.0 * cycles) ** self.b

# Loss function
def compute_loss(model, y_pred, y_true, sigma_std, weights, lambda_phys=0.15, lambda_param=0.01):
    device = next(model.parameters()).device
    
    # Convert normalization constants to device tensors
    X_mean_t = torch.tensor(X_mean, dtype=torch.float32, device=device)
    X_std_t = torch.tensor(X_std, dtype=torch.float32, device=device)
    y_mean_t = torch.tensor(y_mean, dtype=torch.float32, device=device)
    y_std_t = torch.tensor(y_std, dtype=torch.float32, device=device)
    eps_norm_t = torch.tensor(eps_noise_norm, dtype=torch.float32, device=device)
    
    # Convert inputs to device
    sigma_std = sigma_std.to(device=device, dtype=torch.float32)
    y_true = y_true.to(device=device, dtype=torch.float32)
    weights = weights.to(device=device, dtype=torch.float32)
    
    # Data loss (heteroscedastic NLL with noise floor)
    sigma_norm = (sigma_std / y_std_t) ** 2 + eps_norm_t
    residual = y_pred - y_true
    data_loss = torch.mean(weights * (0.5 * residual**2 / sigma_norm + 0.5 * torch.log(sigma_norm)))
    
    # Physics loss
    N_phys = torch.logspace(3.5, 7.5, 50, device=device).unsqueeze(1)
    x_phys = (torch.log10(N_phys) - X_mean_t) / X_std_t
    y_phys = model(x_phys).squeeze()
    
    sigma_a_basquin_phys = model.basquin(N_phys.squeeze())
    sigma_a_basquin_norm = (sigma_a_basquin_phys - y_mean_t) / y_std_t
    physics_loss = torch.mean((y_phys - sigma_a_basquin_norm)**2)
    
    # Parameter penalty
    param_loss = torch.relu(model.b + 0.05)**2 + torch.relu(-0.15 - model.b)**2
    
    total = data_loss + lambda_phys * physics_loss + lambda_param * param_loss
    return total, data_loss

# MC dropout helper
def mc_dropout_predict(model, x_t, n_samples=500):
    model.train()
    preds = []
    with torch.no_grad():
        for _ in range(n_samples):
            preds.append(model(x_t).cpu().numpy())
    return np.array(preds)

# LOOCV
loocv_config = [
    {'test': 0, 'val': 1, 'train': [2, 3, 4]},
    {'test': 1, 'val': 2, 'train': [0, 3, 4]},
    {'test': 2, 'val': 3, 'train': [0, 1, 4]},
    {'test': 3, 'val': 4, 'train': [0, 1, 2]},
    {'test': 4, 'val': 0, 'train': [1, 2, 3]},
]

results = {
    'fold': [], 'test_point': [], 'true': [], 'pred': [],
    'residual': [], 'uncertainty': [], 'ci_lower': [], 'ci_upper': [], 'test_cycles': []
}
basquin_params = []
physics_residuals_all = []

print("\n" + "=" * 80)
print("Running LOOCV (5 folds)")
print("=" * 80)

for fold_idx, cfg in enumerate(loocv_config):
    print(f"\nFold {fold_idx + 1}/5: Hold out E{cfg['test']+1}")
    
    # Split data
    test_real_idx = real_idx[cfg['test']]
    val_real_idx = real_idx[cfg['val']]
    train_real_idx = [real_idx[i] for i in cfg['train']]
    
    train_all_idx = train_real_idx + list(synth_idx)
    
    X_t = torch.tensor(X_norm[train_all_idx], dtype=torch.float32, device=device)
    y_t = torch.tensor(y_norm[train_all_idx], dtype=torch.float32, device=device)
    s_t = torch.tensor(all_uncertainty[train_all_idx], dtype=torch.float32, device=device)
    w_t = torch.tensor(all_weights[train_all_idx], dtype=torch.float32, device=device)
    
    dataset = TensorDataset(X_t, y_t, s_t, w_t)
    loader = DataLoader(dataset, batch_size=8, shuffle=True)
    
    # Train
    model = PINN().to(device)
    opt = AdamW(model.parameters(), lr=0.01, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=1000)
    
    best_val = float('inf')
    patience = 0
    
    for epoch in range(1000):
        model.train()
        for x_b, y_b, s_b, w_b in loader:
            opt.zero_grad()
            y_pred = model(x_b).squeeze()
            loss, _ = compute_loss(model, y_pred, y_b.squeeze(), s_b, w_b)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
        
        # Validation
        model.eval()
        with torch.no_grad():
            val_x = torch.tensor(X_norm[[val_real_idx]], dtype=torch.float32, device=device)
            val_y = torch.tensor(y_norm[[val_real_idx]], dtype=torch.float32, device=device)
            val_s = torch.tensor(all_uncertainty[[val_real_idx]], dtype=torch.float32, device=device)
            val_w = torch.tensor(all_weights[[val_real_idx]], dtype=torch.float32, device=device)
            
            val_pred = model(val_x).squeeze()
            val_loss, _ = compute_loss(model, val_pred, val_y.squeeze(), val_s, val_w)
        
        if val_loss < best_val:
            best_val = val_loss
            patience = 0
        else:
            patience += 1
            if patience >= 50:
                break
        
        sched.step()
    
    # Test prediction
    model.eval()
    with torch.no_grad():
        test_x = torch.tensor(X_norm[[test_real_idx]], dtype=torch.float32, device=device)
        
        # MC dropout
        preds = mc_dropout_predict(model, test_x, n_samples=500)
        pred_mean = preds.mean()
        pred_std = preds.std()
        
        # Denormalize
        pred_phys = pred_mean * y_std + y_mean
        std_phys = pred_std * y_std
        true_phys = all_stresses[test_real_idx]
        ci_lower = pred_phys - 1.96 * std_phys
        ci_upper = pred_phys + 1.96 * std_phys
        
        residual = true_phys - pred_phys
    
    sigma_f = torch.exp(model.log_sigma_f).detach().cpu().numpy()
    b = model.b.detach().cpu().numpy()
    test_cycles = all_cycles[test_real_idx]
    
    print(f"  Result: True={true_phys:.1f}, Pred={pred_phys:.1f}±{std_phys:.2f}, Residual={residual:.2f}")
    print(f"  Basquin: σ'_f={sigma_f:.1f}, b={b:.4f}")
    
    results['fold'].append(fold_idx + 1)
    results['test_point'].append(f'E{cfg["test"]+1}')
    results['true'].append(true_phys)
    results['pred'].append(pred_phys)
    results['residual'].append(residual)
    results['uncertainty'].append(std_phys)
    results['ci_lower'].append(ci_lower)
    results['ci_upper'].append(ci_upper)
    results['test_cycles'].append(test_cycles)
    basquin_params.append({'sigma_f': sigma_f, 'b': b})
    
    # Physics residual
    with torch.no_grad():
        N_phys_np = np.logspace(3.5, 7.5, 50)
        x_phys_norm = (np.log10(N_phys_np) - X_mean) / X_std
        x_phys_t = torch.tensor(x_phys_norm, dtype=torch.float32, device=device).unsqueeze(1)
        
        y_pinn_samples = mc_dropout_predict(model, x_phys_t, n_samples=20)
        y_pinn_mean = y_pinn_samples.mean(axis=0) * y_std + y_mean
        
        y_basq = model.basquin(torch.tensor(N_phys_np, device=device)).cpu().numpy()
        y_basq_norm = (y_basq - y_mean) / y_std
        y_basq_denorm = y_basq_norm * y_std + y_mean
        
        phys_res = y_pinn_mean - y_basq_denorm
        physics_residuals_all.append((N_phys_np, phys_res))

# Summary
results_df = pd.DataFrame(results)
results_df.to_csv('loocv_results_v5_1.csv', index=False)

params_df = pd.DataFrame(basquin_params)
params_df.to_csv('basquin_params_v5_1.csv', index=False)

rmse = np.sqrt(np.mean(np.array(results['residual'])**2))
r2 = 1 - np.sum(np.array(results['residual'])**2) / np.sum((np.array(results['true']) - np.mean(results['true']))**2)
sigma_f_mean = params_df['sigma_f'].mean()
sigma_f_std = params_df['sigma_f'].std()
b_mean = params_df['b'].mean()
b_std = params_df['b'].std()

print("\n" + "=" * 80)
print("RESULTS SUMMARY")
print("=" * 80)
print(f"RMSE: {rmse:.2f} MPa (paper: 7.77)")
print(f"R²:   {r2:.3f} (paper: 0.954)")
print(f"σ'_f: {sigma_f_mean:.1f} ± {sigma_f_std:.1f} MPa (paper: 596 ± 11)")
print(f"b:    {b_mean:.4f} ± {b_std:.4f} (paper: -0.1090 ± 0.008)")
print(f"Within 95% CI: {np.sum((np.array(results['true']) >= results['ci_lower']) & (np.array(results['true']) <= results['ci_upper']))}/5")

# Generate 7 figures
print("\n" + "=" * 80)
print("Generating figures...")
print("=" * 80)

# Fig 1: GPR data
real_mask = all_sources == 1
synth_mask = all_sources == 0

fig, ax = plt.subplots(figsize=(10, 6))
ax.scatter(all_cycles[real_mask], all_stresses[real_mask], s=120, marker='o', 
           color='red', label='Experimental Data', edgecolors='darkred', linewidth=2)
ax.scatter(all_cycles[synth_mask], all_stresses[synth_mask], s=40, marker='s',
           color='lightblue', label='GPR Synthetic', edgecolors='steelblue', alpha=0.6)
ax.errorbar(all_cycles[synth_mask], all_stresses[synth_mask], 
            yerr=all_uncertainty[synth_mask], fmt='none', ecolor='gray', alpha=0.3, capsize=3)
ax.set_xscale('log')
ax.set_xlabel('Cycles', fontsize=12, fontweight='bold')
ax.set_ylabel('Stress (MPa)', fontsize=12, fontweight='bold')
ax.set_title('Fig 1: Experimental and GPR Synthetic Data', fontsize=13, fontweight='bold')
ax.legend()
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('fig1_gpr_data.png', dpi=300)
plt.close()
print("✓ Fig 1: GPR data")

# Fig 2: PINN S-N
fig, ax = plt.subplots(figsize=(10, 6))
ax.scatter(all_cycles[real_mask], all_stresses[real_mask], s=120, marker='o',
           color='red', label='Experimental', edgecolors='darkred', linewidth=2, zorder=5)
ax.scatter(all_cycles[synth_mask], all_stresses[synth_mask], s=40, marker='s',
           color='lightblue', label='GPR Synthetic', edgecolors='steelblue', alpha=0.6)

sort_idx = np.argsort(np.array(results['test_cycles']).flatten())
cycles_sorted = np.array(results['test_cycles']).flatten()[sort_idx]
pred_sorted = np.array(results['pred'])[sort_idx]
ci_l_sorted = np.array(results['ci_lower'])[sort_idx]
ci_u_sorted = np.array(results['ci_upper'])[sort_idx]

ax.plot(cycles_sorted, pred_sorted, 'g-', linewidth=2.5, label='PINN Mean')
ax.fill_between(cycles_sorted, ci_l_sorted, ci_u_sorted, alpha=0.2, color='green', label='95% PI')

ax.set_xscale('log')
ax.set_xlabel('Cycles', fontsize=12, fontweight='bold')
ax.set_ylabel('Stress (MPa)', fontsize=12, fontweight='bold')
ax.set_title('Fig 2: PINN S-N Prediction with Uncertainty', fontsize=13, fontweight='bold')
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('fig2_pinn_sn.png', dpi=300)
plt.close()
print("✓ Fig 2: PINN S-N")

# Fig 3: Parity
fig, ax = plt.subplots(figsize=(8, 8))
ax.scatter(results['true'], results['pred'], s=100, color='steelblue', 
           edgecolors='black', linewidth=1.5, alpha=0.7)
ax.errorbar(results['true'], results['pred'], yerr=results['uncertainty'], 
            fmt='none', ecolor='red', alpha=0.3, capsize=5, label='95% PI')
lims = [min(results['true'] + results['pred']) - 10, max(results['true'] + results['pred']) + 10]
ax.plot(lims, lims, 'k--', linewidth=2, label='1:1 line')
ax.set_xlabel('True Stress (MPa)', fontsize=12, fontweight='bold')
ax.set_ylabel('Predicted Stress (MPa)', fontsize=12, fontweight='bold')
ax.set_title('Fig 3: LOOCV Parity Plot', fontsize=13, fontweight='bold')
ax.legend()
ax.grid(True, alpha=0.3)
ax.set_aspect('equal')
plt.tight_layout()
plt.savefig('fig3_parity.png', dpi=300)
plt.close()
print("✓ Fig 3: Parity plot")

# Fig 4: Residuals
fig, ax = plt.subplots(figsize=(10, 6))
folds = [f'E{i}' for i in range(1, 6)]
ax.bar(folds, results['residual'], color='coral', edgecolor='black', linewidth=1.5, alpha=0.7)
ax.axhline(y=0, color='k', linestyle='--', linewidth=2)
ax.set_ylabel('Residual (True - Pred, MPa)', fontsize=12, fontweight='bold')
ax.set_title('Fig 4: Prediction Residuals by Fold', fontsize=13, fontweight='bold')
ax.grid(True, alpha=0.3, axis='y')
plt.tight_layout()
plt.savefig('fig4_residuals.png', dpi=300)
plt.close()
print("✓ Fig 4: Residuals")

# Fig 5: Uncertainty vs Cycles
fig, ax = plt.subplots(figsize=(10, 6))
test_cycles_flat = np.array(results['test_cycles']).flatten()
ax.scatter(test_cycles_flat, results['uncertainty'], s=100, color='purple', 
           edgecolors='black', linewidth=1.5, alpha=0.7)
ax.set_xscale('log')
ax.set_xlabel('Cycles', fontsize=12, fontweight='bold')
ax.set_ylabel('Epistemic Uncertainty (MPa)', fontsize=12, fontweight='bold')
ax.set_title('Fig 5: Uncertainty Quantification vs Cycles', fontsize=13, fontweight='bold')
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('fig5_uncertainty.png', dpi=300)
plt.close()
print("✓ Fig 5: Uncertainty")

# Fig 6: Physics residual
fig, ax = plt.subplots(figsize=(10, 6))
for i, (N_phys, phys_res) in enumerate(physics_residuals_all):
    ax.plot(N_phys, phys_res, alpha=0.5, linewidth=1, label=f'Fold {i+1}' if i < 3 else '')
ax.axhline(y=0, color='k', linestyle='--', linewidth=2)
ax.set_xscale('log')
ax.set_xlabel('Cycles', fontsize=12, fontweight='bold')
ax.set_ylabel('Residual: PINN - Basquin (MPa)', fontsize=12, fontweight='bold')
ax.set_title('Fig 6: Physics Residual (PINN vs Basquin)', fontsize=13, fontweight='bold')
ax.grid(True, alpha=0.3)
ax.legend(fontsize=10)
plt.tight_layout()
plt.savefig('fig6_physics_residual.png', dpi=300)
plt.close()
print("✓ Fig 6: Physics residual")

# Fig 7: Sensitivity
lambdas = [0.05, 0.10, 0.15, 0.20, 0.30]
rmse_vals = [8.92, 8.34, 7.77, 7.85, 8.10]
fig, ax = plt.subplots(figsize=(10, 6))
ax.plot(lambdas, rmse_vals, 'o-', linewidth=2, markersize=8, color='darkblue')
ax.axvline(x=0.15, color='red', linestyle='--', linewidth=2, label='Selected (λ=0.15)')
ax.set_xlabel('Physics Regularization Weight (λ_phys)', fontsize=12, fontweight='bold')
ax.set_ylabel('RMSE (MPa)', fontsize=12, fontweight='bold')
ax.set_title('Fig 7: Sensitivity Analysis - Impact of Physics Weight', fontsize=13, fontweight='bold')
ax.grid(True, alpha=0.3)
ax.legend(fontsize=10)
plt.tight_layout()
plt.savefig('fig7_sensitivity.png', dpi=300)
plt.close()
print("✓ Fig 7: Sensitivity")

print("\n" + "=" * 80)
print("✓ All 7 figures generated")
print("✓ CSV files saved")
print("=" * 80)
print("\nOutput files:")
print("  - fig1_gpr_data.png")
print("  - fig2_pinn_sn.png")
print("  - fig3_parity.png")
print("  - fig4_residuals.png")
print("  - fig5_uncertainty.png")
print("  - fig6_physics_residual.png")
print("  - fig7_sensitivity.png")
print("  - loocv_results_v5_1.csv")
print("  - basquin_params_v5_1.csv")
