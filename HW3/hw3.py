import numpy as np
import matplotlib.pyplot as plt
import sympy as sp
from numba import jit, prange

#-------------------
#  define parameters
#-------------------
mass = 39.948 * 103.6427   # g/mol Ar
epsilon = 0.00802236   # eV
sigma = 3.97   # A
kB = 8.617e-5   # eV/K
L = 7.5 # nm
L = L*10 # A
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
def compute_forces_and_energy(coords):
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

# Steepest Descent relax from hw2
def SD_relax(coords, alpha=0.05, MaxMoveLength=0.1, MaxStep=800,save=False):
    energies = []
    traj = []
    coords = coords.copy()
    for step in range(MaxStep):
        # 1. Compute Forces
        F_all,PE = compute_forces_and_energy(coords)
        energies.append(PE)

        DeltaR = alpha*F_all
        # 2. Safety Check (Crucial step for Part 2 report)
        for i in range(len(coords)):
            if abs(DeltaR[i][0]) > MaxMoveLength:
                DeltaR[i][0] = MaxMoveLength if DeltaR[i][0] > 0 else -MaxMoveLength
            if abs(DeltaR[i][1]) > MaxMoveLength:
                DeltaR[i][1] = MaxMoveLength if DeltaR[i][1] > 0 else -MaxMoveLength       
            # length = np.linalg.norm(DeltaR[i])
            # if length > MaxMoveLength:
            #     DeltaR[i] = DeltaR[i]/length*MaxMoveLength
        coords += DeltaR
        coords %= L
         # 3. Compute Energy (for monitoring)
#        PE = total_energy(coords)
#        energies.append(PE)
        #print(f'Energy in function SD_relax [{step:>4}/{MaxStep:>4}] {E:>5.3f}')
        # 4. Check Convergence
        if save and step%10==0:
            traj.append(coords.copy())        
        if np.max(np.linalg.norm(F_all,axis=1)) < 0.05:
            print(f"SD收敛于第{step}步")
            break
    return coords, energies, traj


# molecular dynamics simulation
#@jit(nopython=True, cache=True)
def MD_simulation(coords, vel, dt, steps, T_target=None, thermostat='none',tau=200):
    coords = coords.copy()
    vel = vel.copy()

    time_log, E_log, T_log = [], [], []

    forces, PE = compute_forces_and_energy(coords)
    _,KE = get_temperature(vel)
#    PE = total_energy(coords)

    for step in range(steps):
        vel += 0.5 * forces / mass * dt
        coords += vel * dt
        coords %= L

        forces_new, PE = compute_forces_and_energy(coords)
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

        if step % 500 == 0:
            np.savez(f'./MD_steps_save/MD--{step}.npz', coords=coords,vel=vel)

    return coords, vel, time_log, E_log, T_log



###------###
#  PART 1  #
###------###
def time_step_test(coords):
    dt_list = [2, 20, 40]

#    plt.figure()

    for dt in dt_list:
        vel = set_initial_vel(600,coords)

        _,_,t,E,_ = MD_simulation(coords,vel,dt,1000)

        E0 = E[0]
        drift = [abs((e - E0)/E0) for e in E]

        plt.figure()
        plt.plot(t,drift)
#        plt.plot(t, drift, label=f"dt={dt} fs")
        plt.xlabel("Time (fs)")
        plt.ylabel(r"$|E(t)-E(0)| / E(0)$")
#        plt.ylabel("Relative Energy Drift")
        plt.title(f"NVE Stability Test: dt = {dt} fs")
        plt.savefig(f"./time_step_test_dt{dt}.png",dpi=300)
 


###------###
#  PART 2  #
###------###
def thermostat_test(coords):

    vel1 = set_initial_vel(600,coords)
    vel2 = vel1.copy()

    _,_,t1,_,T1 = MD_simulation(coords,vel1,dt=1,steps=2000,T_target=600,thermostat='rescale')
    _,_,t2,_,T2 = MD_simulation(coords,vel2,dt=1,steps=2000,T_target=600,thermostat='berendsen',tau=200)

    T_mean1, T_std1 = np.mean(T1), np.std(T1)
    T_mean2, T_std2 = np.mean(T2), np.std(T2)

    plt.figure()
    plt.plot(t1, T1, label="Hard Rescale")
    plt.plot(t2, T2, label="Berendsen",color='orange')

    plt.axhline(T_mean1, linestyle='--', label=f"Rescle Mean={T_mean1:.1f}K, std={T_std1:.1f}")
    plt.axhline(T_mean2, linestyle='--', label=f"Berendsen Mean={T_mean2:.1f}K, std={T_std2:.1f}", color='orange')
    plt.axhline(600, linestyle=':', label="Target 600K")


    plt.xlabel("Time (fs)")
    plt.ylabel("Temperature (K)")
    plt.legend()
    plt.grid(True)
    plt.title("Thermostat Comparision")
    plt.savefig("./thermostat_comparision.png",dpi=300)
#    plt.show()

def annealing(coords):
    dt = 1
    # Heat
    vel = set_initial_vel(1600,coords)
    coords,vel,_,_,_ = MD_simulation(coords,vel,dt,steps=50000,T_target=1600,thermostat='berendsen',tau=200)

    # Cool
    T_final = 114514
    max_loops = 18
    loop = 0
    while T_final > 10 and loop < max_loops:
        coords,vel,_,_,T = MD_simulation(coords,vel,dt,steps=100000,T_target=10,thermostat='berendsen',tau=50000)
        T_final = T[-1]
        print(f"[Cooling loop {loop}]Final T of annealing is {T_final:.2f} K.")

        loop += 1
    
    if T_final > 10 :
        print("Cooling did not finished in 10 loops.")
 
    # Quench
    coords,energies,_ = SD_relax(coords)

    return coords,energies


def compare_structures(coords_A, coords_B):

    plt.figure(figsize=(8,4))

    plt.subplot(1,2,1)
    plt.scatter(coords_A[:,0], coords_A[:,1])
    plt.title("Structure A")

    plt.subplot(1,2,2)
    plt.scatter(coords_B[:,0], coords_B[:,1])
    plt.title("Structure B")

    plt.savefig("structures.png", dpi=300)
#    plt.show()


# =========================
if __name__ == "__main__":

    # PART 1
    coords0 = np.loadtxt("./particle_coordinates_random.dat")

    # Structure A（HW2）
    coords_A,E_A,_ = SD_relax(coords0.copy())
    np.savetxt("./random_structure_after_SD.dat", coords_A)

    #coords_A = np.loadtxt("./random_structure_after_SD.dat")
    time_step_test(coords_A)

    # Part 2
#    coords, vel, t, E, T = MD_simulation(coords_A,vel,steps=1000,dt=2,thermostat='none')
#    coords2, vel2, t2, E2, T2 = MD_simulation(coords_A,vel,steps=1000,dt=2,T_target=600,thermostat='berendsen')
    thermostat_test(coords_A)

    # Part 3
    coords_B,E_B = annealing(coords_A.copy())

    E_A_final = E_A[-1]
    E_B_final = E_B[-1]

    print("The Total Energy of Structure A is", E_A_final)
    print("The Total Energy of Structure B is", E_B_final)
    if E_B_final < E_A_final:
        print("Annealing is useful.")
    else:
        print("Annealing is useless.")
     
    compare_structures(coords_A, coords_B)

    np.savetxt("final_coords_berendsen.dat",coords_B)
#    np.savetxt("final_vel_berendsen.dat",vel2)
    print("Done.")
