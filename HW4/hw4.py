import os, glob
import numpy as np
import matplotlib.pyplot as plt
import sympy as sp
from numba import jit, prange

#-------------------
#  define parameters
#-------------------
mass = 39.948   # amu Ar
epsilon = 0.00802236   # eV
sigma = 3.97   # A
kB = 8.617e-5   # eV/K
r0 = 10.0   # A
r_cut = 11.0   # A

SIGMA6 = sigma**6
SIGMA12 = sigma**12
EPS4 = 4*epsilon

r = sp.symbols('r')
A, B = sp.symbols('A B')

## def lennard-jones potential
U_LJ_sym = 4*epsilon*((sigma/r)**12-(sigma/r)**6)
dU_LJ_sym = sp.diff(U_LJ_sym, r)

## def tail function
U_tail_sym = A*(r-r_cut)**4 + B*(r-r_cut)**2 #eV
dU_tail_sym = sp.diff(U_tail_sym,r)

### solve for A and B 
eq1 = sp.Eq(U_tail_sym.subs(r,r0),U_LJ_sym.subs(r,r0))
eq2 = sp.Eq(dU_tail_sym.subs(r,r0),dU_LJ_sym.subs(r,r0))
solution = sp.solve((eq1,eq2),(A,B))

A_val = float(solution[A])
B_val = float(solution[B])

#-----------------------------------------------------------------------------

@jit(nopython=True, cache=True)
def compute_forces_and_energy(coords,L):
    N = len(coords)
    F = np.zeros_like(coords)
    PE = 0.0
    
    for i in range(N):
        for j in range(i+1, N):
            dx = coords[j, 0] - coords[i, 0]
            dy = coords[j, 1] - coords[i, 1]

            dx -= L * np.round(dx / L)
            dy -= L * np.round(dy / L)
            
            r = np.sqrt(dx*dx + dy*dy)

            if r < 1e-8:
                continue

            if 1e-8 < r < r0:
                PE += EPS4 * (SIGMA12 / r**12 - SIGMA6 / r**6)
                f_mag = EPS4 * (-12 * SIGMA12 / r**13 + 6 * SIGMA6 / r**7)
                
            elif r <= r_cut:
                dr = r - r_cut
                PE += A_val * dr**4 + B_val * dr**2
                f_mag = 4 * A_val * dr**3 + 2 * B_val * dr
            else:
                continue
            
            fx = f_mag * dx / r
            fy = f_mag * dy / r
            
            F[i, 0] += fx
            F[i, 1] += fy
            F[j, 0] -= fx
            F[j, 1] -= fy
    
    return F, PE

# confirm temperature from atom velocity

@jit(nopython=True, cache=True)
def get_temperature_numba(vel):
    KE = 0.5 * mass * np.sum(vel**2)
    dof = vel.size
    T = 2 * KE / (dof * kB)
    return T, KE

def get_temperature(vel):
    return get_temperature_numba(vel)

# set atom velocity from temperature
def set_initial_vel(T_init, coords):
    vel = np.random.randn(*coords.shape)

    T_now,_ = get_temperature(vel)
    vel *= np.sqrt(T_init / T_now)

    vel -= np.mean(vel, axis=0)
    return vel


# molecular dynamics simulation
#@jit(nopython=True, cache=True)
def load_traj(steps,T_target):
    traj = []
    for step in range(0, steps, 20):
        data = np.load(f'./MD_steps_save/MD_{T_target}K_{step}.npz')
        traj.append(data['coords'])
    return np.array(traj)

def MD_simulation(coords, vel, dt, steps, L, T_target=None, d_traj=20, thermostat='none',tau=200):
    coords = coords.copy()
    vel = vel.copy()

    time_log, E_log, T_log = [], [], []

    forces, PE = compute_forces_and_energy(coords, L)
    _,KE = get_temperature(vel)
#    PE = total_energy(coords)

    for step in range(steps):
        vel += 0.5 * forces / mass * dt
        coords += vel * dt
        coords %= L

        forces_new, PE = compute_forces_and_energy(coords, L)
        #PE = total_energy(coords)

        vel += 0.5 * forces_new / mass * dt
        forces = forces_new
        T, KE = get_temperature(vel)

        if thermostat == 'rescale' and step % 50 == 0:
            vel *= np.sqrt(T_target/T)
            T, KE = get_temperature(vel)

        elif thermostat == 'berendsen':
            lam = np.sqrt(1 + dt/tau * (T_target/T - 1))
            vel *= lam
            T, KE = get_temperature(vel)

#        T, KE = get_temperature(vel)

        if step % 10 == 0:
            time_log.append(step * dt)
            E_log.append(KE + PE)
            T_log.append(T)

        if step % d_traj == 0:
            np.savez(f'./MD_steps_save/MD_{T_target}K_{step}.npz', coords=coords,vel=vel)

    return coords, vel, time_log, E_log, T_log

def write_boxes(filename, coords):
    np.savetxt(filename, coords)

def scale_boxes(coords, L_old, L_new):
    s = coords/L_old
    return s*L_new

def unwrap(traj, L):
    unwrapped = np.zeros_like(traj)
    unwrapped[0] = traj[0]

    for t in range(1, len(traj)):
        delta = traj[t] - traj[t-1]

        delta[delta >  L/2] -= L
        delta[delta < -L/2] += L

        unwrapped[t] = unwrapped[t-1] + delta

    return unwrapped

def compute_MSD(traj):
    r0 = traj[0]
    msd = []

    for t in range(len(traj)):
        dr = traj[t] - r0
        msd.append(np.mean(np.sum(dr**2,axis=1)))

    return np.array(msd)


# =========================
if __name__ == "__main__":

    L = 7.5 # nm
    L = L*10 # A
    dt = 1   #fs

    target_boxes = [75.0, 80.0, 100.0, 120.0]   #A
    temps = [60, 100, 200, 250]

    # PART 1
    coords0 = np.loadtxt("./random_structure_after_SD.dat")
    all_D = []
    d_traj = 20
    n_steps = 15000

 #   plt.figure(figsize=(6,5))   # all_temp
    fig_all, ax_all = plt.subplots(figsize=(6,5))

    for T, L_new in zip(temps, target_boxes):
        for f in glob.glob(f"./MD_steps_save/MD_{T}K_*.npz"):
            os.remove(f)
        coords = scale_boxes(coords0, L, L_new)
        out_file = f"coords_L{int(L_new)}.dat"
        write_boxes(out_file, coords)
        print(f"Generated: {out_file}")

        vel = set_initial_vel(T, coords)
        coords_5ps,vel_5ps,_,_,_ = MD_simulation(coords,vel,dt,steps=5000,L=L_new,T_target=T,d_traj=d_traj,thermostat='berendsen')

        for f in glob.glob(f"./MD_steps_save/MD_{T}K_*.npz"):
            os.remove(f)
        MD_simulation(coords_5ps,vel_5ps,dt,steps=n_steps,L=L_new,d_traj=d_traj,T_target=T,thermostat='berendsen')
        
        coords_traj = load_traj(steps=n_steps, T_target=T)
        coords_unwrap = unwrap(coords_traj, L_new)
         
        msd = compute_MSD(coords_unwrap)

        time = np.arange(len(msd)) * 0.001 * dt * d_traj   #ps

        start = 20
        slope, intercept = np.polyfit(time[start:], msd[start:], 1)
        #slope, intercept = np.polyfit(time, msd, 1)

        D = slope / 4   # 2D
        all_D.append(D)

        #plt.plot(time, msd, label=f"{T}K")
        fig, ax = plt.subplots()
        #plt.figure()
        ax.plot(time, msd, label="MSD")
        ax.plot(time[start:], slope*time[start:] + intercept, '--', label=f"slope={slope:.4f}, D={D:.4e} Å²/ps")
#        plt.text(
#            0.6*max(time), 
#            0.2*max(msd), 
#            f"slope = {slope:.4f} Å²/ps,D={D:.4e} Å²/ps",
#            fontsize=10
#        )
        ax.legend()
        ax.set_xlabel("Time (ps)")
        ax.set_ylabel("MSD (Å$^2$)")
        ax.grid()
        ax.set_title(f"MSD vs Time at {T}K")
        fig.savefig(f"MSD_Time_{T}K.png", dpi=300)
        #plt.show()
        plt.close(fig)

        #plt.figure(1)  # all_tamp
        ax_all.plot(time, msd, label=f"{T}K (D={D:.2e})")

    ax_all.set_xlabel("Time (ps)")
    ax_all.set_ylabel("MSD (Å$^2$)")
    ax_all.grid()
    ax_all.set_title("MSD vs Time at Different Temperatures")
    ax_all.legend()

    fig_all.savefig("MSD_all_temperatures.png", dpi=300)
    plt.show()

    temps_np = np.array(temps)
    D_np = np.array(all_D)

    x = 1000 / temps_np
    y = np.log(D_np + 1e-20)

    slope, ln_D0 = np.polyfit(x, y, 1)
    Ea = -slope * kB * 1000

    print("\nDiffusion coefficients:")
    for T, D in zip(temps, D_np):
        print(f"T={T} K, D={D:.3e}")

    print(f"\nActivation Energy Ea = {Ea:.4f} eV")

    plt.figure()
    plt.scatter(x, y)
    plt.plot(x, slope*x + ln_D0, '--', label="Fit")
    plt.legend()
    plt.xlabel("1000/T (s$^-1$)")
    plt.ylabel("ln(D)")
    plt.grid()
    plt.title("Arrhenius Plot")
    plt.savefig("Arrhenius_Plot.png", dpi=300)
    plt.show()

    print("Done.")
