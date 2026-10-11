import numpy as np
import matplotlib.pyplot as plt
import sympy as sp

###Consider 单位！！！

###------###
#  PART 1  #
###------###
epsilon = 0.00802236   # eV
sigma = 3.97   # A
L = 7.5 # nm
L = L*10 # A
r0 = 10.0   # A
r_cut = 11.0   # A
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
U_tail_sym = U_tail_sym.subs(solution)
dU_tail_sym = dU_tail_sym.subs(solution)

U_LJ = sp.lambdify(r,U_LJ_sym,'numpy')
dU_LJ = sp.lambdify(r,dU_LJ_sym,'numpy')
U_tail = sp.lambdify(r,U_tail_sym,'numpy')
dU_tail = sp.lambdify(r,dU_tail_sym,'numpy')

## def the potential function we use
def U(r):
    if 1e-8 < r < r0:
        return U_LJ(r)
    elif r <= r_cut:
        return U_tail(r)
    else:
        return 0.0

def dU_dr(r):
    if 1e-8 < r < r0:
        return dU_LJ(r)
    elif r <= r_cut:
        return dU_tail(r)
    else:
        return 0.0

# def analytic force as pair force
def pair_force(rij):
    r = np.linalg.norm(rij)
    if r < 1e-8:
        return np.zeros_like(rij)
    return -dU_dr(r) * rij / r

def total_energy(coords):
    E = 0
    N = len(coords)
    for i in range(N):
        for j in range(i+1,N):
            rij = coords[j] - coords[i]
            rij -= L*np.round(rij/L)
            E += U(np.linalg.norm(rij))
    return E   #eV


def numeric_force(coords, atom_idx, delta=1e-3):
    F_num = np.zeros(2)

    for d in range(2):
        shift = np.zeros_like(coords)
        shift[atom_idx, d] = delta

        E_plus = total_energy(coords + shift)
        E_minus = total_energy(coords - shift)

        F_num[d] = -(E_plus - E_minus) / (2 * delta)

    return F_num


def LJ_force(coords):
    N = len(coords)
    F = np.zeros_like(coords)
    for i in range(N):
        for j in range(i+1,N):
            rij = coords[j]-coords[i]
            rij -= L*np.round(rij/L)
            fij = pair_force(rij)
            F[i] -= fij
            F[j] += fij
    return F

#delta = 1e-3
#def numeric_force(coords,idx):
 #   F = np.zeros(2)
 #   for d in range(2):
 #       shift = np.zeros_like(coords)
 #       shift[idx,d] = delta
 #       F[d] = -(total_energy(coords+shift)-total_energy(coords-shift))/(2*delta)
#    return F

def force_test():
    rlist = np.linspace(3,12,200)
    err1, err2 = [],[]

    for r in rlist:
        coords = np.array([[0,0],[r,0]])
        rij = coords[1] - coords[0]
        rij -= L * np.round(rij / L)   
        Fa = pair_force(rij)
        
        Fn = numeric_force(coords, 1)
        Fp = LJ_force(coords)[1]

        err1.append(np.linalg.norm(Fa-Fn))
        err2.append(np.linalg.norm(Fa-Fp))
    
    plt.plot(rlist,err1,label="analytic vs numeric")
    plt.plot(rlist,err2,label="analytic vs program")
    plt.legend()
    plt.xlabel("r (A)")
    plt.ylabel("Error (eV/A)")
    plt.title("Force validation")
    plt.savefig("./Force_validation.png",dpi=300)
    plt.show()
# consider the program again!

def PBC_test():
    rlist = np.linspace(3,12,200)
    err = []

    for r in rlist:
        coords = np.array([[74,0],[r-2,0]])
        rij = coords[1]-coords[0]
        rij -= L*np.round(rij/L)

        Fa = pair_force(rij)
        Fp = LJ_force(coords)[1]

        err.append(np.linalg.norm(Fa-Fp))

    plt.plot(rlist,err)
    plt.xlabel("r (A)")
    plt.ylabel("Error (eV/A)")
    plt.title("PBC test")
    plt.savefig("./PBC_test",dpi=300)
    plt.show()

###----------###
#  PART 2 & 3  #
###----------###

def SD_relax(coords, alpha=0.05, MaxMoveLength=0.1, MaxStep=800,save=False):
    energies = []
    traj = []

    for step in range(MaxStep):
        # 1. Compute Forces
        F_all = LJ_force(coords)
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
        E = total_energy(coords)
        energies.append(E)
        #print(f'Energy in function SD_relax [{step:>4}/{MaxStep:>4}] {E:>5.3f}')
        
        # 4. Check Convergence
        if save and step%10==0:
            traj.append(coords.copy())
        
        if np.max(np.linalg.norm(F_all,axis=1)) < 0.05:
            break

    return coords, energies, traj

def failure_test(coords):
    _, energies, _ = SD_relax(coords.copy(), alpha=0.05, MaxMoveLength=1e10)
    
    plt.plot(energies)
    plt.title("Failure: no MaxMoveLength")
    plt.xlabel("Step")
    plt.ylabel("Energy (eV)")
    plt.savefig("./Failure_with_no_MaxMoveLength",dpi=300)
    plt.show()

def load_coords(filename):
    return np.loadtxt(filename)

def plot_coords(coords,title):
    plt.scatter(coords[:,0],coords[:,1],s=5)
    plt.title(title)
    plt.axis('equal')
    plt.savefig(title,dpi=300)
    plt.show()

def energy_comparison(all_results):
    plt.figure(figsize=(6,8))

    for name, energies in all_results.items():
        steps = np.arange(len(energies))
        plt.plot(steps[20:], energies[20:], label=name)

   # for name, energies in all_results.items():
   #     plt.plot(energies, label=name)

    plt.xlabel("Step")
    plt.ylabel("Energy (eV)")
    plt.title("Energy vs Step")

    plt.legend()
    plt.grid(True)

    plt.savefig("Energy_comparison.png", dpi=300)
    plt.show()

### MAIN ###
if __name__ == "__main__":
    force_test()
    PBC_test()

    coords_rand = load_coords("particle_coordinates_random.dat")
    
    failure_test(coords_rand)

    coordsA = load_coords("particle_coordinates_random.dat")
    coordsB = load_coords("./scripts&outputs/generated_random_structure.dat")
    coordsC = load_coords("particle_coordinates_square.dat")
    coordsD = load_coords("particle_coordinates_hcp.dat")

    
    all_results = {}
    for name,coords in zip(["A","B","C","D"],[coordsA,coordsB,coordsC,coordsD]):
    #    if name != 'B': continue 
        final,E,_ = SD_relax(coords,alpha=0.05)
        all_results[name] = E

        #plt.plot(E,label=name)

        plot_coords(final,f"Final_structure{name}")
    
    #energy_comparison({"A": all_results["A"]})
    energy_comparison(all_results)

   # plt.legend()
   # plt.xlabel("Step")
    #plt.ylabel("Energy")
    #plt.title("Energy comparision")
    #plt.show()
