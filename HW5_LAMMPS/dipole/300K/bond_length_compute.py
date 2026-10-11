# since we have known that the system reached equilibrium after 38.45ps
# since time_ps = step * 0.5 / 1000
# so the step_equilibrium = 76900
# and since dump file dumps every 40 steps (20fs)
# our caculation start from the second frame, which is at TIMESTEP 76920
# 4-N 5-C

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

###--- PBC check ---###
def pbc_vector(r1, r2, box):
    dr = r2 - r1
    dr -= box * np.round(dr / box)
    return dr


###--- MAIN ---###
def read_dump(filename, start_timestep=0, sample_interval=40):

    with open(filename, "r") as f:
        lines = f.readlines()

    i = 0
    coords_all = []
    timesteps = []
    box = None

    while i < len(lines):
        if "ITEM: TIMESTEP" in lines[i]:
            timestep = int(lines[i+1].strip())
            natoms = int(lines[i+3].strip())       
            i += 4
            xlo, xhi = map(float, lines[i+1].split())
            ylo, yhi = map(float, lines[i+2].split())
            zlo, zhi = map(float, lines[i+3].split())
            box = np.array([xhi-xlo, yhi-ylo, zhi-zlo])

            i += 5
            if timestep < start_timestep or timestep % sample_interval != 0:
                i += natoms
                continue

            coords = np.zeros((natoms, 3))
            types = np.zeros(natoms, dtype=int)

            for j in range(natoms):
                parts = lines[i+j].split()
                types[j] = int(parts[1])
                coords[j] = list(map(float, parts[2:5]))

            coords_all.append((coords, types))
            timesteps.append(timestep)

            i += natoms

        else:
            i += 1

    return coords_all, np.array(timesteps), box

### --- EXTRACT --- ###
def extract_CN(coords, types):
    C_list = coords[types == 5]
    N_list = coords[types == 4]
    return C_list, N_list

### --- MATCH --- ###
def match_CN(C_list, N_list, box):

    n_mol = len(C_list)
    dipoles = np.zeros((n_mol, 3))
    used = np.zeros(n_mol, dtype=bool)

    for i, C in enumerate(C_list):

        min_dist = 1e9
        best_j = -1

        for j, N in enumerate(N_list):

            if used[j]:
                continue

            dr = pbc_vector(C, N, box)
            dist = np.linalg.norm(dr)

            if dist < min_dist:
                min_dist = dist
                best_j = j

        if best_j == -1:
            continue
        if min_dist > 2.0:
            print("Warning: large C-N distance:", min_dist)

        used[best_j] = True

        mu = pbc_vector(C, N_list[best_j], box)
        mu /= np.linalg.norm(mu)

        dipoles[i] = mu

    return dipoles

### --- COMPUTE --- ###
def compute_dipoles(coords_all, box):

    all_dipoles = []

    for coords, types in coords_all:

        C_list, N_list = extract_CN(coords, types)

        if len(C_list) != len(N_list):
            print("ERROR: C and N count mismatch")
            break

        dipoles = match_CN(C_list, N_list, box)

        all_dipoles.append(dipoles)

    return np.array(all_dipoles)

### --- CORRELATION --- ###
def dipole_correlation(dipoles, dt_ps, max_time_ps=15):

    n_frames = dipoles.shape[0]
    max_dt = min(int(max_time_ps / dt_ps), n_frames - 1)

    C = []
    t_axis = []

    for dt in range(max_dt):

        vals = []

        for t0 in range(n_frames - dt):

            mu0 = dipoles[t0]
            mut = dipoles[t0 + dt]

            dot = np.sum(mu0 * mut, axis=1)
            vals.append(np.mean(dot))

        C.append(np.mean(vals))
        t_axis.append(dt * dt_ps)

    return np.array(t_axis), np.array(C)


### --- FIT --- ###
def fit_exp(t, C):

    def f(t, A0, tau):
        return A0 * np.exp(-t / tau)

    popt, _ = curve_fit(f, t, C, p0=[1, 5], maxfev=5000)
    return popt


### --- MAIN --- ###
if __name__ == "__main__":

    filename = "./npt_product.dump"
    coords_all, timesteps, box = read_dump(filename, start_timestep=76920, sample_interval=40)
    dt_ps = 0.02 # 0.0005 * 40 = 0.02ps
    time_ps = timesteps * dt_ps

    print("Frames used:", len(coords_all))

    # ========= dipole =========
    dipoles = compute_dipoles(coords_all, box)

    print("Dipole shape:", dipoles.shape)

    # sanity check
    print("Norm check:", np.linalg.norm(dipoles[0], axis=1)[:5])

    # ========= correlation =========
    t, C = dipole_correlation(dipoles, dt_ps)

    # ========= 拟合 =========
    A0, tau = fit_exp(t, C)
    print("Tau =", tau, "ps")

    # ========= 画图 =========
    plt.figure(figsize=(6,4))
    plt.plot(t, C, label="C(t)")
    plt.plot(t, A0*np.exp(-t/tau), '--', label=f"fit τ={tau:.2f} ps")

    plt.xlabel("Time (ps)")
    plt.ylabel("Dipole Correlation")
    plt.legend()

    plt.savefig("dipole_corr.png", dpi=300)
    plt.show()