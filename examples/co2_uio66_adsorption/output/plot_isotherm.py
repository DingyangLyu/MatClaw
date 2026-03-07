import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

pressures = [0.1, 1.0, 5.0, 10.0]
loading_molkg = [0.3593, 2.1150, 4.3912, 5.3766]
errors_molkg  = [0.0029, 0.0295, 0.0619, 0.0479]
loading_moluc = [2.3859, 14.044, 29.158, 35.701]
errors_moluc  = [0.0192, 0.1956, 0.4108, 0.3183]

fig, axes = plt.subplots(1, 2, figsize=(11, 5))

# Left: mol/kg
ax = axes[0]
ax.errorbar(pressures, loading_molkg, yerr=errors_molkg,
            fmt='o-', color='steelblue', capsize=5, linewidth=2,
            markersize=8, markerfacecolor='white', markeredgewidth=2)
ax.set_xlabel('Pressure (bar)', fontsize=13)
ax.set_ylabel('CO₂ Loading (mol/kg)', fontsize=13)
ax.set_title('CO₂ Adsorption in UiO-66\n298 K, GCMC/TraPPE+UFF', fontsize=12)
ax.set_xlim(-0.3, 11)
ax.set_ylim(0, 6.5)
ax.grid(True, alpha=0.3)
ax.tick_params(labelsize=11)

# Right: molecules/uc
ax = axes[1]
ax.errorbar(pressures, loading_moluc, yerr=errors_moluc,
            fmt='s-', color='coral', capsize=5, linewidth=2,
            markersize=8, markerfacecolor='white', markeredgewidth=2)
ax.set_xlabel('Pressure (bar)', fontsize=13)
ax.set_ylabel('CO₂ Loading (molecules/unit cell)', fontsize=13)
ax.set_title('CO₂ Adsorption in UiO-66\n298 K, GCMC/TraPPE+UFF', fontsize=12)
ax.set_xlim(-0.3, 11)
ax.set_ylim(0, 42)
ax.grid(True, alpha=0.3)
ax.tick_params(labelsize=11)

plt.tight_layout()
plt.savefig('/workspace/group/uio66_co2/co2_isotherm.png', dpi=150, bbox_inches='tight')
plt.close()
print("Saved co2_isotherm.png")
