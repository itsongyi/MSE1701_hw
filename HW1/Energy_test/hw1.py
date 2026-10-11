###------###
#  PART 1  #
###------###
import numpy as np
import matplotlib.pyplot as plt
import sympy as sp

## define the constants
epsilon = 0.185   # kcal/mol
sigma = 3.97   # A
r = sp.symbols('r')
r0 = 10.0   # A
r_cut = 11.0   # A
A, B = sp.symbols('A B')

## def lennard-jones potential
U_LJ = 4*epsilon*((sigma/r)**12-(sigma/r)**6)

## get the derivative of U_LJ
dU_LJ = sp.diff(U_LJ,r)

## def tail function
U_tail = A*(r-r_cut)**4 + B*(r-r_cut)**2

## get the derivative of U_tail
dU_tail = sp.diff(U_tail,r)

### solve for A and B 
eq1 = sp.Eq(U_tail.subs(r,r0),U_LJ.subs(r,r0))
eq2 = sp.Eq(dU_tail.subs(r,r0),dU_LJ.subs(r,r0))
solution = sp.solve((eq1,eq2),(A,B))
print('The solution is',solution)
U_tail = U_tail.subs(solution)

U_LJ = sp.lambdify(r,U_LJ,'numpy')
dU_LJ = sp.lambdify(r,dU_LJ,'numpy')
U_tail = sp.lambdify(r,U_tail,'numpy')
dU_tail = sp.lambdify(r,dU_tail,'numpy')

## def the potential function we use
def U(r):
    U_total = np.zeros_like(r)
    for i,val in enumerate(r):
        if val < r0 :
            U_total[i] = U_LJ(val)
        elif val <= r_cut :
            U_total[i] = U_tail(val)
        else :
            U_total[i] = 0
    return U_total

## generate r values
r = np.linspace(3.0,12.0,20000)
U_vals = U(r)
## numerical force
dr = r[1]- r[0]

F = np.zeros_like(U_vals)
for i in range(1,len(r)-1):
    F[i] = -(U_vals[i+1]-U_vals[i-1])/(2*dr)

## plot potential
plt.figure()
plt.plot(r,U_vals)
plt.xlabel("r (Å)")
plt.ylabel("U(r) kcal/mol")
plt.title("Potential Energy")
plt.axvline(10,color='red',linestyle='--',label='r0')
plt.axvline(11,color='black',linestyle='--',label='rcut')
plt.legend()
plt.savefig("./Energy_test/Potential_Energy.png",dpi=300)
plt.show()

## plot force
plt.figure()
plt.plot(r,F)
plt.xlabel("r (Å)")
plt.ylabel("F(r)")
plt.title("Force vs Distance")
plt.savefig("./Energy_test/Force.png",dpi=300)
plt.show()

## plot Zoom 
plt.figure()
mask = (r>=9.0) & (r<=11.5)
plt.plot(r[mask],U_vals[mask])
plt.xlabel("Distance r (Å)")
plt.ylabel("Potential Energy U(r) (kcal/mol)")
plt.title("Zoom of Potential near r0 = 10 Å")
plt.savefig("./Energy_test/potential_zoom.png",dpi=300)
plt.show()



###------###
#  PART 2  #
###------###
L = 7.5 # nm
L = L*10 # A
Atom1 = [1,50]
Atom2 = [72,50]
atoms_test = np.array([Atom1,
                       Atom2])
dx = atoms_test[0,0] - atoms_test[1,0]
dy = atoms_test[0,1] - atoms_test[1,1]

# apply PBC
dx = dx - L*np.round(dx/L)
dy = dy - L*np.round(dy/L)

r_12 = np.sqrt(dx**2 + dy**2)

U_total = U(np.array([r_12]))[0]
print('The potential energy is',U_total,'when the distance is',r_12)



###------###
#  PART 3  #
###------###
# define the constants
N = 400

# random atoms
atoms = np.random.rand(N,2)*L
np.savetxt("./Energy_test/generated_random_structure.dat", atoms)
corner = np.array([0,0])
dist = np.linalg.norm(atoms - corner,axis=1)
target_index = np.argmin(dist)
target_atom = atoms[target_index]

# draw boxes
shifts = [-L,0,L]
# images for neighbor search
images = []
for dx in shifts:
    for dy in shifts:
        shift = np.array([dx,dy])
        images.append(atoms + shift)
images = np.vstack(images)

# find neighbors
neighbors = []
for pos in images:
    d = np.linalg.norm(pos - target_atom)
    if d < r_cut and d > 1e-12:
        neighbors.append(pos)

neighbors = np.array(neighbors)

# create the 2D scatter plot of the atoms
plt.figure(figsize=(12,12))
# draw atoms in 9 boxes
for dx in shifts :
    for dy in shifts:
        shift = np.array([dx,dy])
        atoms_shift = atoms + shift
        plt.scatter(atoms_shift[:,0],atoms_shift[:,1],s=15,color='blue')
plt.scatter(atoms[:,0],atoms[:,1],s=15)
plt.scatter(target_atom[0],target_atom[1],color='red',s=120,label='Target atom')

# draw the circle with R=r_cut
circle = plt.Circle((target_atom[0],target_atom[1]),r_cut,fill=False,linestyle='--')
plt.gca().add_patch(circle)

for nb in neighbors:
    plt.plot([target_atom[0],nb[0]],
             [target_atom[1],nb[1]],
            color='blue',linewidth=0.8)

for dx in shifts:
    for dy in shifts:
        x0 = dx
        y0 = dy
        rect = plt.Rectangle((x0,y0),L,L,
                             fill=False,
                             linestyle='--',
                             linewidth=0.8,
                             color='gray') 
        plt.gca().add_patch(rect)

plt.xlim(-L,2*L)
plt.ylim(-L,2*L)
plt.xlabel("length(A)")
plt.ylabel("length(A)")
plt.gca().set_aspect('equal')
plt.title("Neighbor search with PBC")
plt.legend()
plt.savefig("./Energy_test/Neighbor_search.png",dpi=300)
plt.show()



###------###
#  PART 4  #
###------###

# Plot the structure of the random structure which just generated
plt.figure(figsize=(8,8))
plt.scatter(atoms[:,0],atoms[:,1],s=15,color='blue')
plt.xlabel("x (Å)")
plt.ylabel("y (Å)")
plt.title("Generated Random Structure")
plt.gca().set_aspect('equal')
plt.savefig("./Energy_test/generated_structure.png",dpi=300)
plt.show()

# Plot the structure of the structure load from the board
# read the structure
atoms_repo = np.loadtxt("particle_coordinates_random.dat")
plt.figure(figsize=(8,8))
plt.scatter(atoms_repo[:,0],atoms_repo[:,1],s=15,color='blue')
plt.xlabel("x (Å)")
plt.ylabel("y (Å)")
plt.title("Repository Random Structure")
plt.gca().set_aspect('equal')
plt.savefig("./Energy_test/repository_structure.png",dpi=300)
plt.show()

# def a function to compute energy
def compute_energy(atoms):
    N = len(atoms)
    E_total = 0.0
    for i in range(N):
        for j in range(i+1,N):
            dx = atoms[i,0] - atoms[j,0]
            dy = atoms[i,1] - atoms[j,1]
            # PBC
            dx = dx - L*np.round(dx/L)
            dy = dy - L*np.round(dy/L)

            r_ij = np.sqrt(dx**2 + dy**2)

            if 2 < r_ij < r_cut:
                E_total += U(np.array([r_ij]))[0]

    return E_total


# Compute the energy of the random structure which just generated
E_generated = compute_energy(atoms)
print('Total energy of the random structure just generated is',E_generated,'kcal/mol')

# Compute the energy of the random structure load from the board
E_repo = compute_energy(atoms_repo)
print('Total energy of the random structure given is',E_repo,'kcal/mol')