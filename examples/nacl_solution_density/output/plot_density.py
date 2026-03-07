import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

data = np.loadtxt('/workspace/group/nacl_density/density_prod.dat', comments='#')
steps = data[:, 0]
rho   = data[:, 1]

# Convert steps to time (ps): timestep=2 fs, so time = step * 2e-3 ps
time_ps = steps * 2e-3

rho_mean = rho.mean()
rho_std  = rho.std() / np.sqrt(len(rho))

fig, ax = plt.subplots(figsize=(9, 5))
ax.plot(time_ps, rho, 'o-', color='steelblue', linewidth=1.5,
        markersize=5, markerfacecolor='white', markeredgewidth=1.5, label='Block average (10 ps)')
ax.axhline(rho_mean, color='tomato', linewidth=2, linestyle='--',
           label=f'Mean = {rho_mean:.4f} g/cm³')
ax.fill_between(time_ps, rho_mean - rho_std, rho_mean + rho_std,
                color='tomato', alpha=0.2)
ax.axhline(1.0418, color='gray', linewidth=1.5, linestyle=':', label='Expt. (1.0418 g/cm³)')

ax.set_xlabel('Simulation time (ps)', fontsize=13)
ax.set_ylabel('Density (g/cm³)', fontsize=13)
ax.set_title('1 mol/L NaCl solution — NPT MD at 298 K, 1 atm\nSPC/E water + Joung-Cheatham ions', fontsize=12)
ax.legend(fontsize=11)
ax.set_ylim(1.005, 1.06)
ax.grid(True, alpha=0.3)
ax.tick_params(labelsize=11)

plt.tight_layout()
plt.savefig('/workspace/group/nacl_density/density_nacl.png', dpi=150, bbox_inches='tight')
plt.close()
print("Saved density_nacl.png")
