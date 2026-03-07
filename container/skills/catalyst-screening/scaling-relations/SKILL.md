# Scaling Relations and Volcano Plots

## When to Use

- Screening catalysts using a single descriptor (e.g., E_OH) instead of computing all intermediate binding energies
- Predicting catalytic activity trends across a series of surfaces
- Constructing volcano plots to identify optimal catalysts for OER, ORR, or HER
- Estimating unknown adsorption energies from known ones via linear scaling
- Identifying the thermodynamic optimum and overpotential for a given mechanism

## Prerequisites

- Python packages: `numpy`, `scipy`, `matplotlib`
- Pre-computed adsorption energies (from QE, MACE, or literature)
- ASE + MACE-torch for rapid adsorption energy estimation (optional)
- Reference gas-phase energies: E(H2O), E(H2)

## Theory

### Linear Scaling Relations

Adsorption energies of oxygen intermediates (OH, O, OOH) on transition metal and oxide
surfaces are linearly correlated:

```
E_O   = a1 * E_OH + b1     (typically a1 ~ 2.0, b1 ~ 0.0-0.5 eV)
E_OOH = a2 * E_OH + b2     (typically a2 ~ 1.0, b2 ~ 3.2 eV)
```

This means that a single descriptor (E_OH or E_O) determines all intermediate binding
energies. The universal scaling E_OOH = E_OH + 3.2 eV is a key constraint.

### Volcano Plots

A volcano plot shows catalytic activity (negative overpotential, or equivalently the
rate-limiting step free energy) as a function of the descriptor. The "volcano" shape
arises because:
- Left leg (strong binding): surface is poisoned by intermediates
- Right leg (weak binding): intermediates do not bind well enough
- Apex: optimal catalyst, minimum overpotential

### Free Energy Steps

For the oxygen evolution reaction (OER) at standard conditions:

```
DeltaG1 = G_OH* - G_* - G_H2O + 0.5*G_H2
DeltaG2 = G_O* - G_OH*          + 0.5*G_H2
DeltaG3 = G_OOH* - G_O* - G_H2O + 0.5*G_H2
DeltaG4 = G_O2 - G_OOH*         + 0.5*G_H2
```

Sum = 4 * 1.23 eV = 4.92 eV (thermodynamic requirement)

Overpotential: eta = max(DeltaG_i) / e - 1.23 V

## Detailed Steps

### Step 1: Define Adsorption Energy Database

```python
#!/usr/bin/env python3
"""
Complete scaling relations and volcano plot workflow for OER/ORR catalysis.
Uses literature/computed adsorption energies on transition metal oxide surfaces.
"""
import os
import numpy as np
from scipy.optimize import curve_fit
from scipy.stats import linregress
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

WORK_DIR = os.path.abspath("scaling_volcano")
os.makedirs(WORK_DIR, exist_ok=True)

# ------------------------------------------------------------------
# Adsorption energy database (eV)
# Reference: clean surface + H2O(g) + H2(g)
# DeltaE_OH = E(surface+OH) - E(surface) - E(H2O) + 0.5*E(H2)
# DeltaE_O  = E(surface+O)  - E(surface) - E(H2O) + E(H2)
# DeltaE_OOH= E(surface+OOH)- E(surface) - 2*E(H2O) + 1.5*E(H2)
#
# Values are DFT (PBE) adsorption energies in eV
# Sources: Man et al., ChemCatChem 2011; Rossmeisl et al., JECS 2007
# ------------------------------------------------------------------

# Format: {"surface": {"E_OH": val, "E_O": val, "E_OOH": val}}
ADSORPTION_DATA = {
    # Pure metal (111) surfaces
    "Pt(111)":   {"E_OH": 0.80, "E_O": 1.57, "E_OOH": 3.95},
    "Pd(111)":   {"E_OH": 0.60, "E_O": 1.27, "E_OOH": 3.75},
    "Ir(111)":   {"E_OH": 0.55, "E_O": 1.10, "E_OOH": 3.70},
    "Au(111)":   {"E_OH": 1.70, "E_O": 3.20, "E_OOH": 4.90},
    "Ag(111)":   {"E_OH": 1.50, "E_O": 2.90, "E_OOH": 4.70},
    "Cu(111)":   {"E_OH": 0.35, "E_O": 0.70, "E_OOH": 3.55},
    "Ni(111)":   {"E_OH": 0.20, "E_O": 0.40, "E_OOH": 3.40},
    "Ru(0001)":  {"E_OH": 0.10, "E_O": 0.05, "E_OOH": 3.30},
    # Metal oxide surfaces (rutile 110)
    "RuO2(110)": {"E_OH": 0.45, "E_O": 1.10, "E_OOH": 3.65},
    "IrO2(110)": {"E_OH": 0.65, "E_O": 1.40, "E_OOH": 3.80},
    "MnO2(110)": {"E_OH": 0.05, "E_O": 0.30, "E_OOH": 3.25},
    "TiO2(110)": {"E_OH": -0.20, "E_O": 0.60, "E_OOH": 3.00},
    "Co3O4(311)":{"E_OH": 0.50, "E_O": 1.20, "E_OOH": 3.70},
    "NiOOH":     {"E_OH": 0.70, "E_O": 1.50, "E_OOH": 3.90},
    "FeOOH":     {"E_OH": 0.30, "E_O": 0.80, "E_OOH": 3.50},
}

# ------------------------------------------------------------------
# Free energy corrections (ZPE and entropy at T=298.15 K)
# These are typical values for adsorbates on metal/oxide surfaces
# Reference: Norskov et al., J. Phys. Chem. B 108, 17886 (2004)
# ------------------------------------------------------------------
ZPE_CORRECTIONS = {
    "OH*":   0.355,   # eV (ZPE of adsorbed OH)
    "O*":    0.065,   # eV (ZPE of adsorbed O)
    "OOH*":  0.400,   # eV (ZPE of adsorbed OOH)
    "H2O":   0.560,   # eV (ZPE of gas-phase H2O)
    "H2":    0.268,   # eV (ZPE of gas-phase H2)
    "O2":    0.098,   # eV (ZPE of gas-phase O2)
}

TS_CORRECTIONS = {
    "OH*":   0.000,   # eV (immobile adsorbate, negligible TS)
    "O*":    0.000,
    "OOH*":  0.000,
    "H2O":   0.670,   # eV (T*S at 298.15 K, gas phase)
    "H2":    0.410,   # eV
    "O2":    0.635,   # eV
}

# Combined: DeltaZPE - T*DeltaS for each step
# These are the corrections applied to DeltaE to get DeltaG
# DeltaG = DeltaE + (ZPE_prod - ZPE_react) - T*(S_prod - S_react)
FREE_ENERGY_CORRECTIONS = {
    "OH":  ZPE_CORRECTIONS["OH*"] - ZPE_CORRECTIONS["H2O"] + 0.5*ZPE_CORRECTIONS["H2"]
           - (TS_CORRECTIONS["OH*"] - TS_CORRECTIONS["H2O"] + 0.5*TS_CORRECTIONS["H2"]),
    "O":   ZPE_CORRECTIONS["O*"] - ZPE_CORRECTIONS["OH*"] + 0.5*ZPE_CORRECTIONS["H2"]
           - (TS_CORRECTIONS["O*"] - TS_CORRECTIONS["OH*"] + 0.5*TS_CORRECTIONS["H2"]),
    "OOH": ZPE_CORRECTIONS["OOH*"] - ZPE_CORRECTIONS["O*"] - ZPE_CORRECTIONS["H2O"]
           + 0.5*ZPE_CORRECTIONS["H2"]
           - (TS_CORRECTIONS["OOH*"] - TS_CORRECTIONS["O*"] - TS_CORRECTIONS["H2O"]
              + 0.5*TS_CORRECTIONS["H2"]),
    "O2":  ZPE_CORRECTIONS["O2"] - ZPE_CORRECTIONS["OOH*"] + 0.5*ZPE_CORRECTIONS["H2"]
           - (TS_CORRECTIONS["O2"] - TS_CORRECTIONS["OOH*"] + 0.5*TS_CORRECTIONS["H2"]),
}

# Simpler commonly used total corrections for each intermediate
# DeltaG_OH  = DeltaE_OH + 0.35 eV
# DeltaG_O   = DeltaE_O  + 0.05 eV
# DeltaG_OOH = DeltaE_OOH + 0.40 eV
SIMPLE_CORRECTIONS = {
    "OH":  0.35,   # eV
    "O":   0.05,   # eV
    "OOH": 0.40,   # eV
}

print("="*60)
print("Adsorption Energy Database Loaded")
print(f"Number of surfaces: {len(ADSORPTION_DATA)}")
print("="*60)
```

### Step 2: Fit Scaling Relations

```python
#!/usr/bin/env python3
"""
Fit linear scaling relations between adsorption energies.
"""
import numpy as np
from scipy.stats import linregress
import matplotlib.pyplot as plt
import os

WORK_DIR = os.path.abspath("scaling_volcano")
os.makedirs(WORK_DIR, exist_ok=True)

# Use the database from Step 1 (paste or import)
# ... (ADSORPTION_DATA defined above)

# Extract arrays
surfaces = list(ADSORPTION_DATA.keys())
E_OH  = np.array([ADSORPTION_DATA[s]["E_OH"]  for s in surfaces])
E_O   = np.array([ADSORPTION_DATA[s]["E_O"]   for s in surfaces])
E_OOH = np.array([ADSORPTION_DATA[s]["E_OOH"] for s in surfaces])


def fit_scaling_relation(x, y, x_label, y_label):
    """Fit and print linear scaling relation."""
    slope, intercept, r_value, p_value, std_err = linregress(x, y)
    r_squared = r_value**2
    print(f"  {y_label} = {slope:.3f} * {x_label} + {intercept:.3f}  "
          f"(R^2 = {r_squared:.4f})")
    return slope, intercept, r_squared


print("\nScaling Relations:")
print("-" * 60)

# E_O vs E_OH
slope_O, intercept_O, r2_O = fit_scaling_relation(E_OH, E_O, "E_OH", "E_O")

# E_OOH vs E_OH
slope_OOH, intercept_OOH, r2_OOH = fit_scaling_relation(E_OH, E_OOH, "E_OH", "E_OOH")

# E_OOH - E_OH (should be ~3.2 eV, the "universal" constraint)
diff_OOH_OH = E_OOH - E_OH
print(f"\n  E_OOH - E_OH = {np.mean(diff_OOH_OH):.3f} +/- {np.std(diff_OOH_OH):.3f} eV")
print(f"  (Universal value ~ 3.2 eV)")

# ------------------------------------------------------------------
# Plot scaling relations
# ------------------------------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))

# --- E_O vs E_OH ---
ax = axes[0]
ax.scatter(E_OH, E_O, s=80, c="steelblue", edgecolor="black", zorder=5, linewidth=0.8)

# Labels
for s, x, y in zip(surfaces, E_OH, E_O):
    ax.annotate(s.replace("(", "\n("), (x, y), textcoords="offset points",
                xytext=(6, 6), fontsize=7.5, alpha=0.85)

# Fit line
x_fit = np.linspace(min(E_OH) - 0.3, max(E_OH) + 0.3, 100)
y_fit = slope_O * x_fit + intercept_O
ax.plot(x_fit, y_fit, "r--", linewidth=1.5,
        label=f"$\\Delta E_O$ = {slope_O:.2f}$\\Delta E_{{OH}}$ + {intercept_O:.2f}\n"
              f"$R^2$ = {r2_O:.3f}")

ax.set_xlabel(r"$\Delta E_{OH}$ (eV)", fontsize=12)
ax.set_ylabel(r"$\Delta E_{O}$ (eV)", fontsize=12)
ax.set_title("Scaling: O vs OH", fontsize=13)
ax.legend(fontsize=10, loc="upper left")

# --- E_OOH vs E_OH ---
ax = axes[1]
ax.scatter(E_OH, E_OOH, s=80, c="darkorange", edgecolor="black", zorder=5, linewidth=0.8)

for s, x, y in zip(surfaces, E_OH, E_OOH):
    ax.annotate(s.replace("(", "\n("), (x, y), textcoords="offset points",
                xytext=(6, 6), fontsize=7.5, alpha=0.85)

y_fit2 = slope_OOH * x_fit + intercept_OOH
ax.plot(x_fit, y_fit2, "r--", linewidth=1.5,
        label=f"$\\Delta E_{{OOH}}$ = {slope_OOH:.2f}$\\Delta E_{{OH}}$ + {intercept_OOH:.2f}\n"
              f"$R^2$ = {r2_OOH:.3f}")

# Universal constraint line
ax.plot(x_fit, x_fit + 3.2, "g:", linewidth=1.5, alpha=0.7,
        label=r"$\Delta E_{OOH} = \Delta E_{OH} + 3.2$")

ax.set_xlabel(r"$\Delta E_{OH}$ (eV)", fontsize=12)
ax.set_ylabel(r"$\Delta E_{OOH}$ (eV)", fontsize=12)
ax.set_title("Scaling: OOH vs OH", fontsize=13)
ax.legend(fontsize=10, loc="upper left")

plt.suptitle("Linear Scaling Relations for OER Intermediates", fontsize=14, y=1.02)
plt.tight_layout()
plt.savefig(os.path.join(WORK_DIR, "scaling_relations.png"), dpi=150, bbox_inches="tight")
plt.close()
print(f"\nScaling plot saved to {os.path.join(WORK_DIR, 'scaling_relations.png')}")
```

### Step 3: Construct Volcano Plot

```python
#!/usr/bin/env python3
"""
Construct volcano plot for OER.
The descriptor is DeltaG_OH (or equivalently DeltaE_OH).
"""
import numpy as np
import matplotlib.pyplot as plt
import os

WORK_DIR = os.path.abspath("scaling_volcano")

# ------------------------------------------------------------------
# OER mechanism (4 proton-electron transfer steps)
# *       + H2O -> OH*  + H+ + e-   DeltaG1
# OH*             -> O*   + H+ + e-   DeltaG2
# O*      + H2O -> OOH* + H+ + e-   DeltaG3
# OOH*           -> O2   + H+ + e-   DeltaG4
#
# Sum: 2 H2O -> O2 + 4H+ + 4e-   DeltaG_total = 4*1.23 = 4.92 eV
#
# Free energy steps as functions of DeltaG_OH:
# DeltaG1 = DeltaG_OH
# DeltaG2 = DeltaG_O - DeltaG_OH
# DeltaG3 = DeltaG_OOH - DeltaG_O
# DeltaG4 = 4.92 - DeltaG_OOH
# ------------------------------------------------------------------

# Simple corrections
CORR_OH  = 0.35
CORR_O   = 0.05
CORR_OOH = 0.40


def compute_oer_overpotential(E_OH, E_O, E_OOH):
    """
    Compute OER overpotential from adsorption energies.

    Parameters
    ----------
    E_OH, E_O, E_OOH : float or array
        Adsorption energies in eV

    Returns
    -------
    eta : float or array, overpotential in V
    pds : int, potential-determining step (1-4)
    dG : (4,) array, free energy steps in eV
    """
    # Apply free energy corrections
    G_OH  = E_OH  + CORR_OH
    G_O   = E_O   + CORR_O
    G_OOH = E_OOH + CORR_OOH

    # Free energy steps at U = 0
    dG1 = G_OH
    dG2 = G_O - G_OH
    dG3 = G_OOH - G_O
    dG4 = 4.92 - G_OOH

    dG = np.array([dG1, dG2, dG3, dG4])

    # Potential-determining step (largest DeltaG)
    pds = np.argmax(dG) + 1
    max_dG = np.max(dG)

    # Overpotential
    eta = max_dG / 1.0 - 1.23  # in V (each step involves 1 electron)

    return eta, pds, dG


def compute_oer_volcano_from_scaling(G_OH_range, slope_O, intercept_O,
                                      slope_OOH, intercept_OOH):
    """
    Compute OER volcano curve using scaling relations.
    Uses DeltaG_OH as the single descriptor.

    Parameters
    ----------
    G_OH_range : 1D array
        Range of DeltaG_OH values
    slope_O, intercept_O : float
        Scaling: DeltaG_O = slope_O * DeltaG_OH + intercept_O
    slope_OOH, intercept_OOH : float
        Scaling: DeltaG_OOH = slope_OOH * DeltaG_OH + intercept_OOH

    Returns
    -------
    eta : 1D array, overpotential (V)
    pds : 1D array, potential-determining step
    dG_all : (N, 4) array, free energy steps
    """
    N = len(G_OH_range)
    eta = np.zeros(N)
    pds = np.zeros(N, dtype=int)
    dG_all = np.zeros((N, 4))

    for i, G_OH in enumerate(G_OH_range):
        # Apply scaling relations (on DeltaG, not DeltaE)
        G_O   = slope_O * G_OH + intercept_O
        G_OOH = slope_OOH * G_OH + intercept_OOH

        dG1 = G_OH
        dG2 = G_O - G_OH
        dG3 = G_OOH - G_O
        dG4 = 4.92 - G_OOH

        dG = np.array([dG1, dG2, dG3, dG4])
        dG_all[i] = dG

        pds[i] = np.argmax(dG) + 1
        eta[i] = np.max(dG) - 1.23

    return eta, pds, dG_all


# ------------------------------------------------------------------
# Compute overpotential for each surface in the database
# ------------------------------------------------------------------
# Use the ADSORPTION_DATA from Step 1
ADSORPTION_DATA = {
    "Pt(111)":   {"E_OH": 0.80, "E_O": 1.57, "E_OOH": 3.95},
    "Pd(111)":   {"E_OH": 0.60, "E_O": 1.27, "E_OOH": 3.75},
    "Ir(111)":   {"E_OH": 0.55, "E_O": 1.10, "E_OOH": 3.70},
    "Au(111)":   {"E_OH": 1.70, "E_O": 3.20, "E_OOH": 4.90},
    "Ag(111)":   {"E_OH": 1.50, "E_O": 2.90, "E_OOH": 4.70},
    "Cu(111)":   {"E_OH": 0.35, "E_O": 0.70, "E_OOH": 3.55},
    "Ni(111)":   {"E_OH": 0.20, "E_O": 0.40, "E_OOH": 3.40},
    "Ru(0001)":  {"E_OH": 0.10, "E_O": 0.05, "E_OOH": 3.30},
    "RuO2(110)": {"E_OH": 0.45, "E_O": 1.10, "E_OOH": 3.65},
    "IrO2(110)": {"E_OH": 0.65, "E_O": 1.40, "E_OOH": 3.80},
    "MnO2(110)": {"E_OH": 0.05, "E_O": 0.30, "E_OOH": 3.25},
    "TiO2(110)": {"E_OH": -0.20, "E_O": 0.60, "E_OOH": 3.00},
    "Co3O4(311)":{"E_OH": 0.50, "E_O": 1.20, "E_OOH": 3.70},
    "NiOOH":     {"E_OH": 0.70, "E_O": 1.50, "E_OOH": 3.90},
    "FeOOH":     {"E_OH": 0.30, "E_O": 0.80, "E_OOH": 3.50},
}

surfaces = list(ADSORPTION_DATA.keys())

print("="*70)
print(f"{'Surface':>15s} | {'E_OH':>6s} | {'E_O':>6s} | {'E_OOH':>6s} | "
      f"{'eta(V)':>7s} | {'PDS':>3s}")
print("-"*70)

surface_eta = {}
surface_G_OH = {}

for s in surfaces:
    d = ADSORPTION_DATA[s]
    eta, pds, dG = compute_oer_overpotential(d["E_OH"], d["E_O"], d["E_OOH"])
    surface_eta[s] = eta
    surface_G_OH[s] = d["E_OH"] + CORR_OH
    print(f"{s:>15s} | {d['E_OH']:>6.2f} | {d['E_O']:>6.2f} | {d['E_OOH']:>6.2f} | "
          f"{eta:>7.3f} | {pds:>3d}")

print("="*70)

# ------------------------------------------------------------------
# Construct volcano curve from scaling relations
# ------------------------------------------------------------------

# Fit scaling in DeltaG space
G_OH_arr  = np.array([ADSORPTION_DATA[s]["E_OH"] + CORR_OH for s in surfaces])
G_O_arr   = np.array([ADSORPTION_DATA[s]["E_O"] + CORR_O for s in surfaces])
G_OOH_arr = np.array([ADSORPTION_DATA[s]["E_OOH"] + CORR_OOH for s in surfaces])

from scipy.stats import linregress

res_O = linregress(G_OH_arr, G_O_arr)
res_OOH = linregress(G_OH_arr, G_OOH_arr)

print(f"\nScaling (in DeltaG space):")
print(f"  DeltaG_O   = {res_O.slope:.3f} * DeltaG_OH + {res_O.intercept:.3f}")
print(f"  DeltaG_OOH = {res_OOH.slope:.3f} * DeltaG_OH + {res_OOH.intercept:.3f}")

# Volcano curve
G_OH_range = np.linspace(-0.5, 2.5, 500)
eta_volcano, pds_volcano, dG_volcano = compute_oer_volcano_from_scaling(
    G_OH_range, res_O.slope, res_O.intercept, res_OOH.slope, res_OOH.intercept
)

# ------------------------------------------------------------------
# Plot volcano
# ------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(10, 7))

# Volcano curve (negative eta so higher = better)
ax.plot(G_OH_range, -eta_volcano, "k-", linewidth=2.5, label="Volcano curve", zorder=3)

# Color regions by PDS
colors_pds = {1: "#e74c3c", 2: "#3498db", 3: "#2ecc71", 4: "#9b59b6"}
pds_labels = {1: "PDS: OH* formation", 2: "PDS: O* formation",
              3: "PDS: OOH* formation", 4: "PDS: O$_2$ release"}

for step in [1, 2, 3, 4]:
    mask = pds_volcano == step
    if np.any(mask):
        ax.fill_between(G_OH_range[mask], -4, -eta_volcano[mask],
                        color=colors_pds[step], alpha=0.08)

# Plot individual surfaces
marker_styles = {"(111)": "o", "(0001)": "s", "(110)": "D", "(311)": "^"}
for s in surfaces:
    x = surface_G_OH[s]
    y = -surface_eta[s]
    marker = "o"
    for key, m in marker_styles.items():
        if key in s:
            marker = m
            break
    if "O" in s and "(" in s:
        color = "red"
    elif "OOH" in s.lower() or "ooh" in s.lower():
        color = "orange"
    else:
        color = "steelblue"

    ax.scatter(x, y, s=100, marker=marker, c=color, edgecolor="black",
               zorder=5, linewidth=0.8)
    ax.annotate(s, (x, y), textcoords="offset points", xytext=(8, 5),
                fontsize=8.5, alpha=0.9)

# Ideal overpotential line
ax.axhline(y=0, color="gray", linewidth=0.5, linestyle=":")
ax.axhline(y=-0.3, color="green", linewidth=0.8, linestyle="--", alpha=0.5,
           label=r"$\eta$ = 0.3 V target")

ax.set_xlabel(r"$\Delta G_{OH*}$ (eV)", fontsize=13)
ax.set_ylabel(r"$-\eta_{OER}$ (V)", fontsize=13)
ax.set_title("OER Volcano Plot", fontsize=15)
ax.set_xlim(-0.3, 2.3)
ax.set_ylim(-2.5, 0.5)

# Legend
legend_elements = [
    Line2D([0], [0], color="k", linewidth=2.5, label="Volcano curve"),
    Line2D([0], [0], marker="o", color="w", markerfacecolor="steelblue",
           markersize=10, markeredgecolor="black", label="Pure metals"),
    Line2D([0], [0], marker="D", color="w", markerfacecolor="red",
           markersize=10, markeredgecolor="black", label="Metal oxides"),
    Line2D([0], [0], color="green", linewidth=0.8, linestyle="--",
           label=r"$\eta$ = 0.3 V target"),
]
ax.legend(handles=legend_elements, fontsize=10, loc="lower left")

plt.tight_layout()
plt.savefig(os.path.join(WORK_DIR, "oer_volcano.png"), dpi=150)
plt.close()
print(f"\nVolcano plot saved to {os.path.join(WORK_DIR, 'oer_volcano.png')}")

# ------------------------------------------------------------------
# Print the optimal catalyst from the volcano
# ------------------------------------------------------------------
idx_best = np.argmin(eta_volcano)
print(f"\nVolcano apex at DeltaG_OH = {G_OH_range[idx_best]:.3f} eV")
print(f"Minimum theoretical overpotential = {eta_volcano[idx_best]:.3f} V")

# Find closest real surface to apex
distances = {s: abs(surface_G_OH[s] - G_OH_range[idx_best]) for s in surfaces}
best_surface = min(distances, key=distances.get)
print(f"Closest surface to apex: {best_surface} "
      f"(eta = {surface_eta[best_surface]:.3f} V)")
```

### Step 4: MACE-Based Adsorption Energy Screening (Optional)

```python
#!/usr/bin/env python3
"""
Use MACE for rapid adsorption energy screening.
Build surfaces, place adsorbates, optimize, compute binding energies.
"""
import os
import numpy as np
from ase.build import fcc111, add_adsorbate, molecule
from ase.optimize import BFGS
from ase.constraints import FixAtoms
from ase.io import write

WORK_DIR = os.path.abspath("scaling_volcano")

try:
    from mace.calculators import mace_mp
    calc = mace_mp(model="medium", device="cpu", default_dtype="float64")
    MACE_AVAILABLE = True
except ImportError:
    MACE_AVAILABLE = False
    print("MACE not available; skipping MACE screening")


def compute_adsorption_energy_mace(metal, adsorbate_name, site="ontop",
                                    a=None, height=None):
    """
    Compute adsorption energy using MACE.

    E_ads = E(slab+adsorbate) - E(slab) - E(adsorbate_gas)

    Parameters
    ----------
    metal : str
    adsorbate_name : str ('OH', 'O', 'OOH', 'H')
    site : str ('ontop', 'bridge', 'fcc', 'hcp')
    a : float or None, lattice constant
    height : float or None, initial adsorbate height

    Returns
    -------
    E_ads : float (eV)
    """
    if not MACE_AVAILABLE:
        return None

    lattice_constants = {"Cu": 3.615, "Ag": 4.086, "Au": 4.078,
                         "Pd": 3.890, "Pt": 3.924, "Ni": 3.524,
                         "Ir": 3.839, "Ru": 2.706}

    if a is None:
        a = lattice_constants.get(metal, 3.9)

    # Default heights
    default_heights = {"O": 1.3, "OH": 1.5, "OOH": 1.8, "H": 1.0}
    if height is None:
        height = default_heights.get(adsorbate_name, 1.5)

    # Build slab
    slab = fcc111(metal, size=(3, 3, 4), a=a, vacuum=15.0, periodic=True)
    positions = slab.get_positions()
    z_coords = positions[:, 2]
    z_sorted = np.sort(np.unique(np.round(z_coords, 2)))
    z_fix = z_sorted[1] + 0.1
    slab.set_constraint(FixAtoms(mask=z_coords < z_fix))

    # Optimize clean slab
    slab.calc = calc
    opt = BFGS(slab, logfile=None)
    opt.run(fmax=0.03, steps=200)
    E_slab = slab.get_potential_energy()

    # Build adsorbate molecule
    if adsorbate_name == "OH":
        adsorbate = molecule("OH")
    elif adsorbate_name == "O":
        from ase import Atoms
        adsorbate = Atoms("O", positions=[[0, 0, 0]])
    elif adsorbate_name == "OOH":
        from ase import Atoms
        adsorbate = Atoms("OOH", positions=[[0, 0, 0], [1.3, 0, 0], [1.8, 0.9, 0]])
    elif adsorbate_name == "H":
        from ase import Atoms
        adsorbate = Atoms("H", positions=[[0, 0, 0]])
    else:
        raise ValueError(f"Unknown adsorbate: {adsorbate_name}")

    # Gas-phase reference energy
    adsorbate_gas = adsorbate.copy()
    adsorbate_gas.cell = [15, 15.5, 16]  # box
    adsorbate_gas.pbc = True
    adsorbate_gas.calc = calc
    opt_gas = BFGS(adsorbate_gas, logfile=None)
    opt_gas.run(fmax=0.03, steps=100)
    E_gas = adsorbate_gas.get_potential_energy()

    # Add adsorbate to slab
    slab_ads = slab.copy()
    # Remove constraint, add adsorbate, re-constrain
    slab_ads.set_constraint()
    add_adsorbate(slab_ads, adsorbate_name, height=height, position=site)
    # Re-fix bottom layers
    positions_ads = slab_ads.get_positions()
    z_coords_ads = positions_ads[:, 2]
    slab_ads.set_constraint(FixAtoms(mask=z_coords_ads < z_fix))

    # Optimize
    slab_ads.calc = calc
    opt_ads = BFGS(slab_ads, logfile=None)
    opt_ads.run(fmax=0.03, steps=300)
    E_slab_ads = slab_ads.get_potential_energy()

    E_ads = E_slab_ads - E_slab - E_gas
    return E_ads


if MACE_AVAILABLE:
    print("="*60)
    print("MACE Adsorption Energy Screening")
    print("="*60)

    metals_screen = ["Cu", "Ag", "Au", "Pd", "Pt", "Ni"]
    adsorbates = ["OH", "O"]

    mace_results = {}
    for metal in metals_screen:
        mace_results[metal] = {}
        for ads in adsorbates:
            print(f"\n{metal} + {ads}*...", end=" ", flush=True)
            E_ads = compute_adsorption_energy_mace(metal, ads)
            if E_ads is not None:
                mace_results[metal][ads] = E_ads
                print(f"E_ads = {E_ads:.3f} eV")
            else:
                print("Failed")

    # Print summary
    print("\n" + "="*60)
    print("MACE Screening Results")
    print(f"{'Metal':>6s} | {'E_OH (eV)':>10s} | {'E_O (eV)':>10s}")
    print("-"*40)
    for metal in metals_screen:
        e_oh = mace_results.get(metal, {}).get("OH", float("nan"))
        e_o = mace_results.get(metal, {}).get("O", float("nan"))
        print(f"{metal:>6s} | {e_oh:>10.3f} | {e_o:>10.3f}")
    print("="*60)
    print("\nNote: MACE energies may differ from DFT. Use for relative trends only.")
```

## Key Parameters

| Parameter | Typical Value | Notes |
|-----------|--------------|-------|
| Scaling slope (E_O vs E_OH) | ~2.0 | Varies: 1.5-2.5 depending on surface type |
| Scaling offset (E_OOH - E_OH) | ~3.2 eV | "Universal" constraint for OER |
| ZPE correction (OH*) | +0.35 eV | Approximate; system-dependent |
| ZPE correction (O*) | +0.05 eV | Small for atomic adsorbate |
| ZPE correction (OOH*) | +0.40 eV | Similar to OH* |
| Equilibrium potential (OER) | 1.23 V | Standard thermodynamic value |
| DeltaG_total (OER) | 4.92 eV | 4 x 1.23 eV |
| Slab size for adsorption | 3x3x4 minimum | Avoid adsorbate-adsorbate interactions |
| k-grid for slab DFT | 4x4x1 | Gamma-centered; test convergence |

## Interpreting Results

**Scaling relations:**
- R^2 > 0.9 indicates a reliable scaling relation
- Deviations from scaling can indicate unusual binding mechanisms
- The OOH-OH correlation (slope ~1.0, intercept ~3.2) is most universal
- The O-OH correlation (slope ~2.0) reflects the double-bond nature of O*

**Volcano plot:**
- Apex = thermodynamic optimum catalyst
- Left side: too-strong binding (surface poisoned by intermediates)
- Right side: too-weak binding (hard to activate H2O)
- Best catalysts for OER: RuO2, IrO2 (near apex)
- Minimum possible overpotential ~0.3-0.4 V (limited by OOH-OH scaling)

**Overpotential values:**
- eta < 0.3 V: excellent catalyst
- eta = 0.3-0.5 V: good catalyst
- eta = 0.5-1.0 V: moderate
- eta > 1.0 V: poor

## Common Issues

1. **Scaling relation has poor R^2:**
   - Different surface types (metals vs oxides) may have different scaling
   - Fit separately for each class of materials
   - Check for outliers due to computational errors

2. **Volcano apex is unrealistically low:**
   - The OOH-OH constraint (~3.2 eV) sets a fundamental lower limit
   - If you get eta < 0.3 V, check whether the universal constraint is violated

3. **MACE adsorption energies differ from DFT:**
   - MACE is approximate; use for trends, not absolute values
   - Compare MACE scaling slopes to DFT slopes
   - Validate on a few known systems before screening

4. **Free energy corrections vary significantly:**
   - Use system-specific ZPE and entropy if available (from phonon calculations)
   - Tabulated values are approximate averages
   - Temperature and solvent effects can shift values by 0.1-0.2 eV

5. **Adsorbate finds wrong binding site:**
   - Try multiple initial sites (ontop, bridge, fcc, hcp)
   - Use the lowest energy configuration
   - For OOH, initial orientation matters significantly
