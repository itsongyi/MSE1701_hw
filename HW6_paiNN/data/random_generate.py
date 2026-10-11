import numpy as np

L= 75.0
N = 400

# random atoms
atoms = np.random.rand(N,2)*L
np.savetxt("./generated_random_structure_3.dat", atoms)