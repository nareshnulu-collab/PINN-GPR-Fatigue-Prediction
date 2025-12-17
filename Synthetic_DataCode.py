# quick_start_az31_gpr_FINAL_v2.py
# Robust GPR synthetic data generation for AZ31 fatigue
# Material: AZ31 rolled magnesium alloy (Fouad 2011)
# Condition: Ball Burnishing (BB), Rotating Bending, R = -1, 50 Hz

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import (
    RBF,
    WhiteKernel,
    ConstantKernel as C,
    DotProduct,
)
from sklearn.model_selection import LeaveOneOut

print("="*80)
print("AZ31 FATIGUE SYNTHETIC DATA GENERATION")
print("Reference: Fouad et al. (2011) - Rolled AZ31, BB condition")
print("="*80)

# 1. REAL DATA (FOUAD 2011, FIGURE 3, BB CONDITION ONLY)
# =========

print("\n[1] Loading experimental S–N data...")

N_cycles_real = np.array([1.5e4, 5.0e4, 1.5e5, 1.5e6, 1.0e7], dtype=float)
sigma_a_real = np.array([200.0, 175.0, 150.0, 120.0, 100.0], dtype=float)

X_real = np.log10(N_cycles_real).reshape(-1, 1)
y_real = sigma_a_real.reshape(-1, 1)

print(f"  Material: Rolled AZ31 magnesium alloy")
print(f"  Condition: Ball Burnishing (BB)")
print(f"  Test: Rotating bending, R = -1, 50 Hz")
print(f"  Real points: {len(sigma_a_real)}")
print(f"  Stress range: {sigma_a_real.min():.0f}–{sigma_a_real.max():.0f} MPa")
print(f"  Cycle range: {N_cycles_real.min():.1e}–{N_cycles_real.max():.1e} cycles")


# 2. KERNEL: GLOBAL TREND + LOCAL SMOOTHNESS
# ==========

print("\n[2] Setting up GPR kernel...")

kernel = (
    C(1.0, (1e-3, 1e3)) * (DotProduct() + RBF(1.0, (1e-3, 1e2)))
    + WhiteKernel(5.0, (1e-3, 1e2))
)

gpr = GaussianProcessRegressor(
    kernel=kernel,
    n_restarts_optimizer=15,
    normalize_y=False,
    alpha=1e-8,
    random_state=42,
)

gpr.fit(X_real, y_real)
print(f"  Kernel: {gpr.kernel_}")


# 3. LEAVE-ONE-OUT CROSS-VALIDATION
# ========
print("\n[3] Validating GPR with Leave-One-Out CV...")

loo = LeaveOneOut()
errors = []
predictions_loo = []

for train_idx, test_idx in loo.split(X_real):
    X_train, X_test = X_real[train_idx], X_real[test_idx]
    y_train, y_test = y_real[train_idx], y_real[test_idx]

    gpr_temp = GaussianProcessRegressor(
        kernel=kernel,
        n_restarts_optimizer=10,
        normalize_y=False,
        alpha=1e-8,
        random_state=42,
    )
    gpr_temp.fit(X_train, y_train)
    y_pred, y_std = gpr_temp.predict(X_test.reshape(-1, 1), return_std=True)

    error = float(abs(y_pred[0] - y_test[0, 0]))
    errors.append(error)
    predictions_loo.append(
        {
            "N": 10 ** X_test[0, 0],
            "real": float(y_test[0, 0]),
            "pred": float(y_pred[0]),
            "std": float(y_std[0]),
            "error": error,
        }
    )

mean_loo_error = float(np.mean(errors))
print(f"  Mean LOO error: {mean_loo_error:.2f} MPa")
print(f"  Max LOO error: {max(errors):.2f} MPa")


# 4. MATERIAL-SPECIFIC ENDURANCE LIMIT
# ==========

print("\n[4] Setting material-specific endurance limit...")

endurance_limit_bb = 100.0
lowest_real_stress = sigma_a_real.min()
lower_bound = max(endurance_limit_bb, lowest_real_stress)

print(f"  Reference: Fouad et al. (2011)")
print(f"  Condition: Rolled AZ31 + BB, rotating bending, R=-1")
print(f"  Fatigue strength at 10^7 cycles (BB): ~{endurance_limit_bb:.0f} MPa")
print(f"  Lower bound used for synthetic data: {lower_bound:.0f} MPa")


# 5. GENERATE SYNTHETIC POINTS
# ===========


print("\n[5] Generating synthetic S–N points via GPR...")

logN_min = np.log10(N_cycles_real.min()) - 0.2
logN_max = np.log10(N_cycles_real.max()) + 0.2
X_synth = np.linspace(logN_min, logN_max, 150).reshape(-1, 1)

y_synth_mean, y_synth_std = gpr.predict(X_synth, return_std=True)
N_synth = 10.0 ** X_synth.ravel()

sigma_a_synth = np.maximum(y_synth_mean.ravel(), lower_bound)

alpha_weight = 0.5
weight_factor = 1.0 / (1.0 + alpha_weight * y_synth_std.ravel())

print(f"  Generated synthetic points: {len(N_synth)}")
print(f"  Stress before clamp: {y_synth_mean.min():.1f}–{y_synth_mean.max():.1f} MPa")
print(
    f"  Stress after clamp (>= {lower_bound:.0f} MPa): "
    f"{sigma_a_synth.min():.1f}–{sigma_a_synth.max():.1f} MPa"
)
print(f"  Mean weight_factor: {weight_factor.mean():.3f}")


# 6. BUILD AUGMENTED DATASET
# ==========

print("\n[6] Building augmented dataset with source flags...")

df_real = pd.DataFrame(
    {
        "N_cycles": N_cycles_real,
        "sigma_a": sigma_a_real,
        "sigma_uncertainty": np.zeros_like(sigma_a_real),
        "weight_factor": np.ones_like(sigma_a_real),
        "source": "real",
    }
)

df_synth = pd.DataFrame(
    {
        "N_cycles": N_synth,
        "sigma_a": sigma_a_synth,
        "sigma_uncertainty": y_synth_std.ravel(),
        "weight_factor": weight_factor,
        "source": "synthetic",
    }
)

df_final = pd.concat([df_real, df_synth], ignore_index=True)
df_final.to_csv("az31_fatigue_augmented.csv", index=False)

print(f"  ✓ Saved: az31_fatigue_augmented.csv")
print(f"    Real rows: {len(df_real)}")
print(f"    Synthetic rows: {len(df_synth)}")
print(f"    Total rows: {len(df_final)}")


# 7. VISUALIZATION
# ===========

print("\n[7] Creating visualization...")

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

# LEFT: S–N curve
ax1.scatter(
    N_cycles_real,
    sigma_a_real,
    color="red",
    s=120,
    marker="o",
    label="Real BB Data (Fouad 2011)",
    zorder=5,
)

ax1.plot(
    N_synth,
    y_synth_mean.ravel(),
    "g-",
    linewidth=2,
    label="GPR Mean",
    zorder=4,
)

ax1.fill_between(
    N_synth,
    (y_synth_mean - 2 * y_synth_std).ravel(),
    (y_synth_mean + 2 * y_synth_std).ravel(),
    alpha=0.18,
    color="green",
    label="95% CI",
)

ax1.axhline(
    lower_bound,
    color="orange",
    linestyle="--",
    linewidth=2,
    label=f"Lower bound = {lower_bound:.0f} MPa",
)

ax1.set_xscale("log")
ax1.set_xlabel("Cycles to Failure, $N_f$ (cycle)", fontsize=11)
ax1.set_ylabel("Stress Amplitude, $\\sigma_a$ (MPa)", fontsize=11)
ax1.set_title("GPR-Generated Synthetic AZ31 BB S–N Data", fontsize=12, fontweight="bold")
ax1.grid(True, which="both", alpha=0.3)
ax1.legend(fontsize=9)

# RIGHT: LOO scatter
real_vals = [p["real"] for p in predictions_loo]
pred_vals = [p["pred"] for p in predictions_loo]

ax2.scatter(
    real_vals,
    pred_vals,
    color="blue",
    s=100,
    alpha=0.7,
    edgecolors="black",
    label="LOO predictions",
)
lims = [min(real_vals + pred_vals) - 5, max(real_vals + pred_vals) + 5]
ax2.plot(lims, lims, "r--", linewidth=2, label="Perfect prediction")

ax2.set_xlabel("Actual Stress (MPa)", fontsize=11)
ax2.set_ylabel("LOO Predicted Stress (MPa)", fontsize=11)
ax2.set_title(
    f"Leave-One-Out CV (MAE = {mean_loo_error:.2f} MPa)", fontsize=12, fontweight="bold"
)
ax2.grid(True, alpha=0.3)
ax2.legend(fontsize=9)
ax2.set_xlim(lims)
ax2.set_ylim(lims)
ax2.set_aspect("equal", adjustable="box")

plt.tight_layout()
plt.savefig("az31_synthetic_BB_final_v2.png", dpi=200)
print(f"  ✓ Saved: az31_synthetic_BB_final_v2.png")
plt.show()

print("\nDONE.")
