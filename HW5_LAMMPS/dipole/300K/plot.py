import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import inset_axes, mark_inset

### for NPT folder
data = np.loadtxt("./form.txt", skiprows=1)

step = data[:, 0]
temp = data[:, 1]
vol = data[:, 7] # only use in npt data 

# timestep = 0.5 fs → ps
time_ps = step * 0.5 / 1000

def moving_avg(x, w):
    return np.convolve(x, np.ones(w)/w, mode='valid')
 
def find_equilibrium(x, target, tol, window_check=70, threshold=70, name="data" ):
    window = 10
    smooth = moving_avg(x, window)
    diff = np.abs(np.diff(smooth))
 
    for i in range(len(diff) - window_check):
        if abs(smooth[i] - target) > tol:
            continue

        if np.all(diff[i:i+window_check] < threshold):
            eq_index = i
            break

    eq_time = time_ps[eq_index + window//2]
    print(f"Equilibrium of {name} at:", eq_time, "ps")
    return eq_time

eq_time_temp = find_equilibrium(x=temp, target=300, tol=6, threshold=30, name="Temperature")
eq_time_vol = find_equilibrium(x=vol, target=63250, tol=60, threshold=70, name="Volume",)

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
axins = inset_axes(ax, width="35%", height="35%", loc="center right")
axins.set_xlim(25, 35)
axins.set_ylim(250, 320)
axins.plot(time_ps, temp)
axins.axvline(eq_time_temp, linestyle='--', color='g',
            label=f"Equilibrium ~ {eq_time_temp:.2f} ps")
mark_inset(ax, axins, loc1=2, loc2=4, fc="none", ec="0.5")
ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
fig.tight_layout()
fig.savefig("./Temp_t_detail.png", dpi=300, bbox_inches="tight")

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
axins = inset_axes(ax, width="35%", height="35%", loc="center right")
axins.set_xlim(30, 42)
axins.set_ylim(62500, 63800)
axins.plot(time_ps, vol)
axins.axvline(eq_time_vol, linestyle='--', color='g',
            label=f"Equilibrium ~ {eq_time_vol:.2f} ps")
mark_inset(ax, axins, loc1=2, loc2=4, fc="none", ec="0.5")
ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
fig.tight_layout()
fig.savefig("./Vol_t_detail.png", dpi=300, bbox_inches="tight")

plt.show()