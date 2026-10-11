import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import inset_axes, mark_inset

### for NPT folder
data = np.loadtxt("./form.txt", skiprows=1)

step = data[:, 0]
etot = data[:, 1]
pe   = data[:, 2]
ke   = data[:, 3]
temp = data[:, 4]
press = data[:, 5]
vol = data[:, 6] # only use in npt data 

# timestep = 0.5 fs → ps
time_ps = step * 0.5 / 1000

def moving_avg(x, w):
    return np.convolve(x, np.ones(w)/w, mode='valid')
 
def find_equilibrium(x, window_check=20, threshold = 10, name="data" ):
    window = 3
    smooth = moving_avg(x, window)
    diff = np.abs(np.diff(smooth))
 
    for i in range(len(diff) - window_check):
        if np.all(diff[i:i+window_check] < threshold):
            eq_index = i
            break

    eq_time = time_ps[eq_index]
    print(f"Equilibrium of {name} at:", eq_time, "ps")
    return eq_time

eq_time_temp = find_equilibrium(temp, name="Temperature")
eq_time_etot = find_equilibrium(x=etot, threshold=20, name="Total Enargy")
eq_time_ke = find_equilibrium(x=ke, threshold=20, name="Kenetic Energy")
eq_time_pe = find_equilibrium(x=pe, threshold=20, name="Portential Energy")
eq_time_vol = find_equilibrium(x=vol, threshold=70, name="Volume",)

# ===== figure Temp =====
plt.figure()
plt.plot(time_ps, temp)
plt.xlabel("Time (ps)")
plt.ylabel("Temperature (K)")
plt.title("Temperature vs Time")
plt.axvline(eq_time_temp, linestyle='--', color='g',
            label=f"Equilibrium ~ {eq_time_temp:.2f} ps")
plt.legend()
plt.savefig("./Temp_t.png", dpi=300)

fig, ax = plt.subplots(figsize=(7, 5))
ax.plot(time_ps, temp)
ax.set_xlabel("Time (ps)")
ax.set_ylabel("Temperature (K)")
ax.set_title("Temperature vs Time")
ax.axvline(eq_time_temp, linestyle='--', color='g',
            label=f"Equilibrium ~ {eq_time_temp:.2f} ps")
axins = inset_axes(ax, width="35%", height="35%", loc="lower right")
axins.set_xlim(-0.3, 4.2)
axins.set_ylim(175, 375)
axins.plot(time_ps, temp)
axins.axvline(eq_time_temp, linestyle='--', color='g',
            label=f"Equilibrium ~ {eq_time_temp:.2f} ps")
mark_inset(ax, axins, loc1=2, loc2=4, fc="none", ec="0.5")
ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
fig.tight_layout()
fig.savefig("./Temp_t_detail.png", dpi=300, bbox_inches="tight")

# ===== figure Energy =====
plt.figure()
plt.plot(time_ps, ke, label="KE")
plt.plot(time_ps, pe, label="PE")
plt.plot(time_ps, etot, label="Total Energy")
plt.xlabel("Time (ps)")
plt.ylabel("Energy (kcal/mol)")
plt.title("Energy vs Time")
plt.axvline(eq_time_etot, linestyle='--', color='g',
            label=f"Etot eq ~ {eq_time_etot:.2f} ps")
plt.axvline(eq_time_ke, linestyle='--', color='b',
            label=f"KE eq ~ {eq_time_ke:.2f} ps")
plt.axvline(eq_time_pe, linestyle='--', color='r',
            label=f"PE eq ~ {eq_time_pe:.2f} ps")
plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
plt.tight_layout()
plt.savefig("./E_t.png", dpi=300)

fig, ax = plt.subplots(figsize=(7, 5))
ax.plot(time_ps, ke, label="KE")
ax.plot(time_ps, pe, label="PE")
ax.plot(time_ps, etot, label="Total Energy")
ax.set_xlabel("Time (ps)")
ax.set_ylabel("Energy (kcal/mol)")
ax.set_title("Energy vs Time")
ax.axvline(eq_time_etot, linestyle='--', color='g',
            label=f"Etot eq ~ {eq_time_etot:.2f} ps")
ax.axvline(eq_time_ke, linestyle='--', color='b',
            label=f"KE eq ~ {eq_time_ke:.2f} ps")
ax.axvline(eq_time_pe, linestyle='--', color='r',
            label=f"PE eq ~ {eq_time_pe:.2f} ps")
axins = inset_axes(ax, width="35%", height="35%", loc="upper right")
axins.set_xlim(-0.3, 2.1)
axins.set_ylim(-2500, 800)
axins.plot(time_ps, ke, label="KE")
axins.plot(time_ps, pe, label="PE")
axins.plot(time_ps, etot, label="Total Energy")
axins.axvline(eq_time_etot, linestyle='--', color='g',
            label=f"Etot eq ~ {eq_time_etot:.2f} ps")
axins.axvline(eq_time_ke, linestyle='--', color='b',
            label=f"KE eq ~ {eq_time_ke:.2f} ps")
axins.axvline(eq_time_pe, linestyle='--', color='r',
            label=f"PE eq ~ {eq_time_pe:.2f} ps")
mark_inset(ax, axins, loc1=2, loc2=4, fc="none", ec="0.5")
ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
fig.tight_layout()
fig.savefig("./E_t_detail.png", dpi=300, bbox_inches="tight")

# ===== figure Volume =====
plt.figure()
plt.plot(time_ps, vol)
plt.xlabel("Time (ps)")
plt.ylabel(r"Volume ($\mathrm{\AA}^3$)")
plt.title("Volume vs Time")
plt.axvline(eq_time_vol, linestyle='--', color='g',
            label=f"Equilibrium ~ {eq_time_vol:.2f} ps")
plt.legend()
plt.savefig("./Vol_t.png", dpi=300)

fig, ax = plt.subplots(figsize=(7, 5))
ax.plot(time_ps, vol)
ax.set_xlabel("Time (ps)")
ax.set_ylabel(r"Volume ($\mathrm{\AA}^3$)")
ax.set_title("Volume vs Time")
ax.axvline(eq_time_vol, linestyle='--', color='g',
            label=f"Equilibrium ~ {eq_time_vol:.2f} ps")
axins = inset_axes(ax, width="35%", height="35%", loc="lower right")
axins.set_xlim(-0.3, 4.2)
axins.set_ylim(5600, 6200)
axins.plot(time_ps, vol)
axins.axvline(eq_time_vol, linestyle='--', color='g',
            label=f"Equilibrium ~ {eq_time_vol:.2f} ps")
mark_inset(ax, axins, loc1=2, loc2=4, fc="none", ec="0.5")
ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
fig.tight_layout()
fig.savefig("./Vol_t_detail.png", dpi=300, bbox_inches="tight")

plt.show()