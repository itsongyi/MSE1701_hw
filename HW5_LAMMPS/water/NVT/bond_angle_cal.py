# since we have known that the system reached equilibrium after 0.65ps
# since time_ps = step * 0.5 / 1000
# so the step_equilibrium = 1300
# and since dump file dumps every 1000 steps
# our caculation start from the second frame, which is at TIMESTEP 2000

import numpy as np
import matplotlib.pyplot as plt

dump_file = "./330.dump"
sample_interval = 2000
start_timestep = 2000

###--- PBC check ---###
def pbc_vector(r1, r2, box):
    dr = r2 - r1
    dr -= box * np.round(dr / box)
    return dr

def bond_length(O, H, box):
    return np.linalg.norm(pbc_vector(O, H, box))

def angle(H1, O, H2, box):
    v1 = pbc_vector(O, H1, box)
    v2 = pbc_vector(O, H2, box)
    cos_theta = np.dot(v1, v2) / (np.linalg.norm(v1)*np.linalg.norm(v2))
    return np.degrees(np.arccos(cos_theta))   # unit is degree

###--- MAIN ---###
bond_lengths = []
angles = []

with open(dump_file, "r") as f:
    lines = f.readlines()

i = 0
n_lines = len(lines)

while i < n_lines:
    if "ITEM: TIMESTEP" in lines[i]:
        timestep = int(lines[i+1].strip())
        
        i += 4
        xlo, xhi = map(float, lines[i+1].split())
        ylo, yhi = map(float, lines[i+2].split())
        zlo, zhi = map(float, lines[i+3].split())
        box = np.array([xhi-xlo, yhi-ylo, zhi-zlo])

        i += 4
        if timestep < start_timestep or timestep % sample_interval != 0:
            i += 600
            continue

        coords = []
        types = []
        i += 1

        for j in range(600):
            parts = lines[i+j].split()
            atom_type = int(parts[1])
            x, y, z = map(float, parts[2:5])

            coords.append([x, y, z])
            types.append(atom_type)

        coords = np.array(coords)

        for k in range(0, 600, 3):
            O = coords[k]
            H1 = coords[k+1]
            H2 = coords[k+2]

            r1 = bond_length(O, H1, box)
            r2 = bond_length(O, H2, box)
            theta = angle(H1, O, H2, box)

            bond_lengths.extend([r1, r2])
            angles.append(theta)

        i += 600
    else:
        i += 1


bad = [r for r in bond_lengths if r > 2]
print(f"Abnormal bonds (>2 Å): {len(bad)}")

plt.figure()
plt.hist(bond_lengths, bins=50)
plt.xlabel("O-H bond length (Å)")
plt.ylabel("Count")
plt.title("O-H Bond Length Distribution")
plt.savefig("./bond_length_distribution.png", dpi=300)
#plt.show()

plt.figure()
plt.hist(angles, bins=50)
plt.xlabel("H-O-H angle (degree)")
plt.ylabel("Count")
plt.title("H-O-H Angle Distribution")
plt.savefig("./angle_distribution.png", dpi=300)
plt.show()