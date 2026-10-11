import numpy as np

input_file = "C2_after_optimization.vasp"

scale_factors = [0.98, 0.99, 1.01, 1.02]

with open(input_file, "r") as f:
    lines = f.readlines()

title = lines[0]
global_scale = float(lines[1].split()[0])

lattice = np.array([
    list(map(float, lines[2].split())),
    list(map(float, lines[3].split())),
    list(map(float, lines[4].split()))
])

rest_lines = lines[5:]

for factor in scale_factors:

    new_lattice = lattice * factor

    output_file = f"POSCAR_{factor:.2f}.vasp"

    with open(output_file, "w") as f:

        f.write(title)
        f.write("1.0\n")

        for vec in new_lattice:
            f.write(
                f"{vec[0]:20.16f} "
                f"{vec[1]:20.16f} "
                f"{vec[2]:20.16f}\n"
            )

        f.writelines(rest_lines)

    print(f"Generated {output_file}")

print("Done.")