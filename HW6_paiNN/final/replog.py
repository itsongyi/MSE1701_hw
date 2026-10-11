import numpy as np
import matplotlib.pyplot as plt


data = np.load("loss_history_direct.npz")

train_losses = data["train_losses"]
val_losses = data["val_losses"]

train_energy_maes = data["train_energy_maes"]
val_energy_maes = data["val_energy_maes"]

train_force_maes = data["train_force_maes"]
val_force_maes = data["val_force_maes"]

# ========= epoch =========

epochs = np.arange(1, len(train_losses) + 1)

# =========================================================
# 1. LOSS
# =========================================================

plt.figure(figsize=(8,6))

plt.plot(epochs, train_losses, label="Train Loss")
plt.plot(epochs, val_losses, label="Validation Loss")

plt.yscale("log")

plt.xlabel("Epoch", fontsize=14)
plt.ylabel("Loss", fontsize=14)

plt.title("Loss Curves(Direct, log)", fontsize=16)

plt.legend()

plt.tight_layout()

plt.savefig("Direct_loss_log.png", dpi=300)

plt.show()

# =========================================================
# 2. ENERGY MAE
# =========================================================

plt.figure(figsize=(8,6))

plt.plot(epochs, train_energy_maes, label="Train Energy MAE")
plt.plot(epochs, val_energy_maes, label="Validation Energy MAE")

plt.yscale("log")

plt.xlabel("Epoch", fontsize=14)
plt.ylabel("Energy MAE", fontsize=14)

plt.title("Energy MAE Curves(Direct, log)", fontsize=16)

plt.legend()

plt.tight_layout()

plt.savefig("Direct_energy_mae_log.png", dpi=300)

plt.show()

# =========================================================
# 3. FORCE MAE
# =========================================================

plt.figure(figsize=(8,6))

plt.plot(epochs, train_force_maes, label="Train Force MAE")
plt.plot(epochs, val_force_maes, label="Validation Force MAE")

plt.yscale("log")

plt.xlabel("Epoch", fontsize=14)
plt.ylabel("Force MAE", fontsize=14)

plt.title("Force MAE Curves(Direct, log)", fontsize=16)

plt.legend()

plt.tight_layout()

plt.savefig("Direct_force_mae_log.png", dpi=300)

plt.show()
