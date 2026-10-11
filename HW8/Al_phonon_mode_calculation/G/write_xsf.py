import yaml
import numpy as np

# =====================
# 读取yaml
# =====================

with open("band_G.yaml", "r") as f:
    data = yaml.safe_load(f)

# =====================
# 晶格矢量
# =====================

lattice = np.array(data["lattice"])

# =====================
# 原子信息
# =====================

atoms = data["points"]

# 原胞晶格矩阵
cell = lattice

# 分数坐标 -> 笛卡尔坐标
cart_coords = []

for atom in atoms:

    cart = np.array(atom["coordinates"])

    cart_coords.append(cart)

# =====================
# Γ点模式
# =====================

bands = data["phonon"][0]["band"]

# =====================
# 逐模式输出
# =====================

for mode_id, mode in enumerate(bands, start=1):

    filename = f"mode{mode_id}.xsf"

    eigvec = mode["eigenvector"]

    with open(filename, "w") as f:

        f.write("CRYSTAL\n")

        f.write("PRIMVEC\n")

        for vec in cell:
            f.write(
                f"   {vec[0]:12.8f}"
                f"   {vec[1]:12.8f}"
                f"   {vec[2]:12.8f}\n"
            )

        f.write("PRIMCOORD\n")
        f.write(f"      {len(atoms)}   1\n")

        for i, atom in enumerate(atoms):

            symbol = atom["symbol"]

            x, y, z = cart_coords[i]

            # 取实部
            vx = eigvec[i][0][0]
            vy = eigvec[i][1][0]
            vz = eigvec[i][2][0]

            f.write(
                f"   {symbol:2s}"
                f"   {x:12.8f}"
                f"   {y:12.8f}"
                f"   {z:12.8f}"
                f"   {vx:16.14f}"
                f"   {vy:16.14f}"
                f"   {vz:16.14f}\n"
            )

    print(f"Generated {filename}")

print("Done.")