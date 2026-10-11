import random
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde

# ========= 参数 =========

snapshot_size = 801      # 1 energy + 800 atom lines
train_ratio = 0.8

long_md_file = "long_md.dat"
test_md_file = "test_md.dat"

# ========= 读取 snapshot =========

def read_snapshots(filename, snapshot_size):

    with open(filename, "r") as f:
        lines = f.readlines()

    n_snapshots = len(lines) // snapshot_size

    snapshots = []

    for i in range(n_snapshots):

        start = i * snapshot_size
        end = start + snapshot_size

        snap = lines[start:end]

        snapshots.append(snap)

    return snapshots

# ========= 提取势能 =========

def get_energies(snapshots):

    energies = []

    for snap in snapshots:

        energy = float(snap[0].split()[0])

        energies.append(energy)

    return np.array(energies)

train_set = read_snapshots("./train_2602_fixed.dat",snapshot_size)
val_set = read_snapshots("./val_700_fixed.dat",snapshot_size)
test_set = read_snapshots("./test_200_fixed.dat",snapshot_size)

print("Train:", len(train_set))
print("Validation:", len(val_set))
print("Test:", len(test_set))

train_E = get_energies(train_set)
val_E = get_energies(val_set)
test_E = get_energies(test_set)

train_E = train_E[train_E > 0]
val_E = val_E[val_E > 0]
test_E = test_E[test_E > 0]

# ========= KDE 分布 =========

xmin = min(train_E.min(), val_E.min(), test_E.min())
xmax = max(train_E.max(), val_E.max(), test_E.max())

x = np.linspace(xmin, xmax, 1000)

kde_train = gaussian_kde(train_E)
kde_val = gaussian_kde(val_E)
kde_test = gaussian_kde(test_E)

# ========= 画图 =========

plt.figure(figsize=(8,6))

plt.plot(x, kde_train(x), label="Train")
plt.plot(x, kde_val(x), label="Validation")
plt.plot(x, kde_test(x), label="Test")

plt.xlabel("Potential Energy(E_pot)")
plt.ylabel("Probability Density")

plt.title("Energy Distribution of Datasets")

plt.legend()

plt.tight_layout()

plt.savefig("energy_distribution.png", dpi=300)

plt.show()




#-------------------------------------
import matplotlib.pyplot as plt
import numpy as np

# ========= 自动统一 bins =========

all_E = np.concatenate([train_E, val_E, test_E])

bins = np.linspace(all_E.min(), all_E.max(), 40)

# ========= 画 histogram =========

plt.figure(figsize=(10,7))

plt.hist(
    train_E,
    bins=bins,
    alpha=0.7,
    label=f"Train (n={len(train_E)})",
    edgecolor='black'
)

plt.hist(
    val_E,
    bins=bins,
    alpha=0.7,
    label=f"Validation (n={len(val_E)})",
    edgecolor='black'
)

plt.hist(
    test_E,
    bins=bins,
    alpha=0.7,
    label=f"Test (n={len(test_E)})",
    edgecolor='black'
)

# ========= 标签 =========

plt.xlabel("Potential Energy (eV)", fontsize=16)
plt.ylabel("Frame Count", fontsize=16)

plt.title(
    "Stacked Energy Distribution by Split (E_pot)",
    fontsize=20
)

plt.legend(fontsize=14)

plt.tight_layout()

plt.savefig("energy_histogram.png", dpi=300)

plt.show()

# 读取
train_set = read_snapshots("./train_2602_fixed.dat", snapshot_size)
val_set = read_snapshots("./val_700_fixed.dat", snapshot_size)
test_set = read_snapshots("./test_200_fixed.dat", snapshot_size)

def filter_snapshots_by_energy(snapshots, threshold=2.0):

    filtered = []

    for snap in snapshots:

        energy = float(snap[0].split()[0])

        if energy >= threshold:
            filtered.append(snap)

    return filtered
# 筛选
train_set = filter_snapshots_by_energy(train_set, 2.0)
val_set = filter_snapshots_by_energy(val_set, 2.0)
test_set = filter_snapshots_by_energy(test_set, 2.0)
def write_dataset(filename, snapshots):

    with open(filename, "w") as f:

        for snap in snapshots:
            f.writelines(snap)
# 写文件
write_dataset("train_filtered.dat", train_set)
write_dataset("val_filtered.dat", val_set)
write_dataset("test_filtered.dat", test_set)

print("Filtered train:", len(train_set))
print("Filtered val:", len(val_set))
print("Filtered test:", len(test_set))