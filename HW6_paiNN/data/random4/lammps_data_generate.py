from ase import Atoms
from ase.io import write
import numpy as np

coords = np.loadtxt("generated_random_structure_2.dat")

positions = [(x, y, 0.0) for x, y in coords]

atoms = Atoms(
    symbols=['Ar'] * len(coords),
    positions=positions,
    cell=[75, 75, 1],
    pbc=[True, True, False]
)

write("struc.data", atoms, format="lammps-data")