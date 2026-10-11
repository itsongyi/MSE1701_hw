import numpy as np
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt

# ==========================
# Birch-Murnaghan EOS
# ==========================

def birch_murnaghan(V, E0, V0, B0, B0p):

    E = E0 - (9*B0/16)*((4-B0p)*V0**3/V**2-(14-3*B0p)*V0**(7/3)/V**(4/3)+(16-3*B0p)*V0**(5/3)/V**(2/3))

    return E


# ==========================
# read data
# ==========================

data = np.loadtxt("data.txt")

V = data[:,0]
E = data[:,1]

# ==========================
# initial guess
# ==========================

E0_guess = np.min(E)

V0_guess = V[np.argmin(E)]

B0_guess = 2.5      # eV/Å^3

B0p_guess = 4.0

p0 = [E0_guess, V0_guess, B0_guess, B0p_guess]

# ==========================
# fitting
# ==========================

popt, pcov = curve_fit(
    birch_murnaghan,
    V,
    E,
    p0=p0
)

E0, V0, B0, B0p = popt

# ==========================
# convert bulk modulus
# ==========================

EV_A3_TO_GPA = 160.21766208

B0_GPa = B0 * EV_A3_TO_GPA

# ==========================
# print results
# ==========================

print("\n===== EOS Fit Results =====")

print(f"E0      = {E0:.8f} eV")
print(f"V0      = {V0:.8f} Å³")
print(f"B0      = {B0:.8f} eV/Å³")
print(f"B0      = {B0_GPa:.2f} GPa")
print(f"B0'     = {B0p:.4f}")

# ==========================
# plot
# ==========================

Vfit = np.linspace(
    min(V),
    max(V),
    300
)

Efit = birch_murnaghan(
    Vfit,
    *popt
)

plt.figure(figsize=(6,4))

plt.scatter(V,E,label="DFT data")

plt.plot(
    Vfit,
    Efit,
    label="Birch-Murnaghan Fit"
)

plt.xlabel("Volume (Å³)")
plt.ylabel("Energy (eV)")
plt.legend()

plt.tight_layout()

plt.savefig(
    "EOS_fit.png",
    dpi=300
)

plt.show()
residuals = E - birch_murnaghan(V, *popt)

ss_res = np.sum(residuals**2)

ss_tot = np.sum((E - np.mean(E))**2)

r2 = 1 - ss_res/ss_tot

print(f"R² = {r2:.8f}")