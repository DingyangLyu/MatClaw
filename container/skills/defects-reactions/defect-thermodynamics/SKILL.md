# Defect Thermodynamics and Transition Levels

## When to Use

- Plot defect formation energy as a function of Fermi level across the band gap
- Determine charge transition levels epsilon(q1/q2) for defects
- Identify whether a defect is a deep or shallow donor/acceptor
- Compute equilibrium defect concentrations as a function of temperature
- Determine the Fermi level self-consistently from charge neutrality
- Interpret defect behavior using Kroeger-Vink notation

## Method Selection

| Criterion | Post-Processing (DFT data) | MACE-Assisted |
|---|---|---|
| Input | DFT formation energies for each charge state | MACE neutral energies only |
| Transition levels | Full analysis possible | Not possible (no charged states) |
| Fermi level diagrams | Complete | Neutral defects only (horizontal lines) |
| Defect concentrations | Temperature-dependent with charge states | Approximate, neutral only |
| Use case | Publication-quality defect analysis | Quick neutral defect thermodynamics |

**Decision flow:**

```
Have DFT formation energies for multiple charge states?
  --> Full thermodynamic analysis (Method A)

Only have neutral defect energies (from MACE or DFT)?
  --> Limited analysis: neutral concentrations vs temperature (Method B)

Need self-consistent Fermi level and defect concentrations?
  --> Method A with charge neutrality solver

Want to use pymatgen-analysis-defects automated workflow?
  --> Method C (requires completed DFT calculations)
```

## Prerequisites

- Completed defect formation energy calculations (from vacancy-formation/, substitution-defect/, or interstitial-defect/ skills)
- Band gap and VBM energy of the host from DFT
- Dielectric constant for Freysoldt corrections (charged defects)
- Chemical potential references
- Python packages: numpy, scipy, matplotlib (pre-installed)
- Optional: `pip install pymatgen-analysis-defects` for automated workflows

## Detailed Steps

### Kroeger-Vink Notation Reference

```
Standard notation for point defects in ionic crystals:

  Symbol:  X_S^C
  where:
    X = species occupying the site
    S = site in the lattice
    C = effective charge relative to the perfect lattice

  Charge symbols:
    .  (dot)   = +1 effective charge (positive)
    '  (prime) = -1 effective charge (negative)
    x          = neutral (zero effective charge)

  Examples in MgO:
    V_O^{..}    = oxygen vacancy, 2+ effective charge
    V_O^{.}     = oxygen vacancy, 1+ effective charge
    V_O^{x}     = oxygen vacancy, neutral
    V_Mg^{''}   = magnesium vacancy, 2- effective charge
    Al_Mg^{.}   = Al on Mg site, 1+ effective charge (donor)
    Li_Mg^{'}   = Li on Mg site, 1- effective charge (acceptor)
    O_i^{''}    = oxygen interstitial, 2- effective charge
    Mg_i^{..}   = Mg interstitial, 2+ effective charge

  Schottky defect (pair):
    V_Mg^{''} + V_O^{..} --> null  (charge neutral pair)

  Frenkel defect (pair):
    Mg_Mg^{x} --> V_Mg^{''} + Mg_i^{..}
```

### Method A: Full Defect Thermodynamic Analysis

#### Step 1: Collect Formation Energies and Build Defect Database

```python
#!/usr/bin/env python3
"""
Build a defect database from completed DFT calculations.
Each defect has formation energies computed at different charge states.

The formation energy at a given Fermi level E_F is:
  E_f(D^q, E_F) = E_f(D^q, E_F=0) + q * E_F

where E_f(D^q, E_F=0) is the formation energy at the VBM (E_F = 0).
"""

import numpy as np
import json

# ============================================================
# Defect formation energy database
# ============================================================
# Each entry: defect name, charge state, E_f at VBM (already corrected)
#
# These values should come from your DFT calculations with
# Freysoldt corrections applied. The examples below use
# representative values for MgO.

defect_data = {
    "system": "MgO",
    "band_gap_eV": 7.7,        # experimental; DFT-PBE gives ~5.0
    "vbm_eV": 0.0,             # reference energy
    "defects": [
        {
            "name": "V_O",
            "kroeger_vink": "V_O",
            "charge_states": {
                0:  {"e_form_at_vbm": 7.2},   # V_O^x
                1:  {"e_form_at_vbm": 5.8},   # V_O^.
                2:  {"e_form_at_vbm": 3.1},   # V_O^{..}
            },
        },
        {
            "name": "V_Mg",
            "kroeger_vink": "V_Mg",
            "charge_states": {
                0:   {"e_form_at_vbm": 7.5},  # V_Mg^x
                -1:  {"e_form_at_vbm": 9.0},  # V_Mg^'
                -2:  {"e_form_at_vbm": 9.8},  # V_Mg^{''}
            },
        },
        {
            "name": "Al_Mg",
            "kroeger_vink": "Al_Mg",
            "charge_states": {
                0:  {"e_form_at_vbm": 2.5},   # Al_Mg^x
                1:  {"e_form_at_vbm": 1.2},   # Al_Mg^.
            },
        },
        {
            "name": "Li_Mg",
            "kroeger_vink": "Li_Mg",
            "charge_states": {
                0:   {"e_form_at_vbm": 3.0},  # Li_Mg^x
                -1:  {"e_form_at_vbm": 4.5},  # Li_Mg^'
            },
        },
    ],
}

with open("defect_database.json", "w") as f:
    json.dump(defect_data, f, indent=2)
print("Defect database saved to defect_database.json")
print(f"System: {defect_data['system']}")
print(f"Band gap: {defect_data['band_gap_eV']} eV")
print(f"Number of defect types: {len(defect_data['defects'])}")
```

#### Step 2: Plot Formation Energy vs. Fermi Level

```python
#!/usr/bin/env python3
"""
Plot defect formation energy as a function of Fermi level.
This is the standard defect diagram used in publications.

For each defect D at charge state q:
  E_f(D^q, E_F) = E_f(D^q, VBM) + q * E_F

The lowest-energy charge state at each Fermi level determines
the stable charge configuration. Crossings between charge states
give the charge transition levels.
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import json

# ============================================================
# Load defect database
# ============================================================
with open("defect_database.json") as f:
    db = json.load(f)

band_gap = db["band_gap_eV"]
E_F = np.linspace(0, band_gap, 500)  # Fermi level from VBM to CBM

# ============================================================
# Color and style assignments
# ============================================================
defect_colors = {
    "V_O": "firebrick",
    "V_Mg": "steelblue",
    "Al_Mg": "forestgreen",
    "Li_Mg": "darkorange",
}
default_colors = ["purple", "brown", "teal", "olive", "navy", "crimson"]

# ============================================================
# Compute and plot formation energies
# ============================================================
fig, ax = plt.subplots(figsize=(8, 6))

transition_levels = []

for defect in db["defects"]:
    name = defect["name"]
    charges = defect["charge_states"]
    color = defect_colors.get(name, default_colors.pop(0) if default_colors else "gray")

    # Compute E_f(E_F) for each charge state
    charge_list = sorted(charges.keys(), key=lambda x: int(x))
    e_f_arrays = {}

    for q_str in charge_list:
        q = int(q_str)
        e_f_vbm = charges[q_str]["e_form_at_vbm"]
        e_f_arrays[q] = e_f_vbm + q * E_F

    # Find the lowest-energy charge state at each E_F
    all_qs = sorted(e_f_arrays.keys())
    e_f_min = np.full_like(E_F, np.inf)
    q_min = np.zeros_like(E_F, dtype=int)

    for q in all_qs:
        mask = e_f_arrays[q] < e_f_min
        e_f_min[mask] = e_f_arrays[q][mask]
        q_min[mask] = q

    # Plot the minimum envelope (thick line)
    ax.plot(E_F, e_f_min, color=color, linewidth=2.5, label=name)

    # Plot individual charge state lines (thin dashed)
    for q in all_qs:
        ax.plot(E_F, e_f_arrays[q], color=color, linewidth=0.8,
                linestyle="--", alpha=0.4)

        # Label charge state at appropriate location
        if q >= 0:
            x_label = band_gap * 0.1
        else:
            x_label = band_gap * 0.9
        y_label = e_f_arrays[q][int(x_label / band_gap * len(E_F))]
        ax.annotate(f"q={q:+d}", xy=(x_label, y_label),
                    fontsize=7, color=color, alpha=0.6)

    # Find charge transition levels (crossings between adjacent charge states)
    for i in range(len(all_qs) - 1):
        q1 = all_qs[i]
        q2 = all_qs[i + 1]
        if q1 == q2:
            continue
        e_f1_vbm = charges[str(q1)]["e_form_at_vbm"]
        e_f2_vbm = charges[str(q2)]["e_form_at_vbm"]

        # Crossing: e_f1 + q1*E_F = e_f2 + q2*E_F
        # E_F_cross = (e_f1 - e_f2) / (q2 - q1)
        if q2 != q1:
            e_f_cross = (e_f1_vbm - e_f2_vbm) / (q2 - q1)
            if 0 <= e_f_cross <= band_gap:
                e_form_at_cross = e_f1_vbm + q1 * e_f_cross

                ax.plot(e_f_cross, e_form_at_cross, "o",
                        color=color, markersize=6, zorder=5)

                transition_levels.append({
                    "defect": name,
                    "q1": q1,
                    "q2": q2,
                    "epsilon_eV": e_f_cross,
                    "e_form_at_transition_eV": e_form_at_cross,
                })

# ============================================================
# Formatting
# ============================================================
ax.set_xlim(0, band_gap)
ax.set_ylim(bottom=0)
ax.set_xlabel("Fermi level (eV above VBM)", fontsize=13)
ax.set_ylabel("Formation energy (eV)", fontsize=13)
ax.set_title(f"Defect formation energies in {db['system']}", fontsize=14)

# Band edge markers
ax.axvline(x=0, color="black", linewidth=1.5)
ax.axvline(x=band_gap, color="black", linewidth=1.5)
ax.text(0.02, ax.get_ylim()[1] * 0.95, "VBM", fontsize=10, ha="left")
ax.text(band_gap - 0.02, ax.get_ylim()[1] * 0.95, "CBM", fontsize=10, ha="right")

ax.legend(fontsize=10, loc="upper left")
ax.grid(True, alpha=0.2)
fig.tight_layout()
fig.savefig("defect_formation_diagram.png", dpi=150)
print("Defect formation diagram saved to defect_formation_diagram.png")

# ============================================================
# Print transition levels
# ============================================================
print("\n=== Charge Transition Levels ===")
print(f"{'Defect':<12} {'Transition':<15} {'epsilon (eV)':<15} {'E_f (eV)'}")
print("-" * 55)
for tl in sorted(transition_levels, key=lambda x: x["epsilon_eV"]):
    trans = f"({tl['q1']:+d}/{tl['q2']:+d})"
    print(f"{tl['defect']:<12} {trans:<15} {tl['epsilon_eV']:<15.4f} {tl['e_form_at_transition_eV']:.4f}")

# Classify as deep or shallow
for tl in transition_levels:
    if tl["q2"] > tl["q1"]:  # donor-like (positive charge)
        depth = band_gap - tl["epsilon_eV"]
        dtype = "donor"
    else:  # acceptor-like
        depth = tl["epsilon_eV"]
        dtype = "acceptor"

    shallow = "shallow" if depth < 0.3 else "deep"
    print(f"  {tl['defect']} {tl['q1']:+d}/{tl['q2']:+d}: {shallow} {dtype} "
          f"(depth = {depth:.3f} eV from nearest band edge)")

with open("transition_levels.json", "w") as f:
    json.dump(transition_levels, f, indent=2, default=float)
print("\nTransition levels saved to transition_levels.json")
```

#### Step 3: Equilibrium Defect Concentrations vs. Temperature

```python
#!/usr/bin/env python3
"""
Compute equilibrium defect concentrations as a function of temperature.
Solves the charge neutrality condition self-consistently to find
the equilibrium Fermi level at each temperature.

Charge neutrality:
  sum_D sum_q (q * c_D^q) + p - n = 0

where:
  c_D^q = N_sites * g * exp(-E_f(D^q, E_F) / k_B T)
  p = N_V * F_{1/2}((E_VBM - E_F) / k_B T)   (hole concentration)
  n = N_C * F_{1/2}((E_F - E_CBM) / k_B T)   (electron concentration)
"""

import numpy as np
from scipy.optimize import brentq
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import json

# ============================================================
# Physical constants
# ============================================================
kB = 8.617333262e-5  # eV/K

# ============================================================
# Load defect database
# ============================================================
with open("defect_database.json") as f:
    db = json.load(f)

band_gap = db["band_gap_eV"]

# Effective density of states (approximate)
# N_V, N_C ~ 1e19 cm^-3 for wide-gap oxides at 300 K
# Scale with T^{3/2}
N_V_300 = 1.0e19  # cm^-3 at 300 K
N_C_300 = 1.0e19

# Site density (number of possible defect sites per cm^3)
# For MgO: 4 formula units per unit cell, a=4.212 A
a_MgO = 4.212e-8  # cm
n_fu_per_cell = 4
vol_cell = a_MgO**3
site_density = n_fu_per_cell / vol_cell  # ~5.3e22 cm^-3
print(f"Site density: {site_density:.2e} cm^-3")

# ============================================================
# Helper functions
# ============================================================
def defect_concentration(e_form, T, n_sites, degeneracy=1):
    """
    Defect concentration: c = n_sites * g * exp(-E_f / kBT).
    """
    if T < 1:
        return 0.0
    arg = -e_form / (kB * T)
    if arg < -200:
        return 0.0
    return n_sites * degeneracy * np.exp(arg)

def electron_concentration(E_F, T, band_gap, N_C_300=1e19):
    """Electron concentration in conduction band (Boltzmann approx)."""
    if T < 1:
        return 0.0
    N_C = N_C_300 * (T / 300)**1.5
    return N_C * np.exp(-(band_gap - E_F) / (kB * T))

def hole_concentration(E_F, T, N_V_300=1e19):
    """Hole concentration in valence band (Boltzmann approx)."""
    if T < 1:
        return 0.0
    N_V = N_V_300 * (T / 300)**1.5
    return N_V * np.exp(-E_F / (kB * T))

def charge_neutrality(E_F, T, defects, band_gap, site_density):
    """
    Returns the net charge: sum of all charged species.
    = 0 at the equilibrium Fermi level.
    """
    total_charge = 0.0

    # Defect contributions
    for defect in defects:
        charges = defect["charge_states"]
        for q_str, data in charges.items():
            q = int(q_str)
            e_f = data["e_form_at_vbm"] + q * E_F
            c = defect_concentration(e_f, T, site_density, degeneracy=1)
            total_charge += q * c

    # Free carriers
    n = electron_concentration(E_F, T, band_gap, N_C_300)
    p = hole_concentration(E_F, T, N_V_300)
    total_charge += p - n  # holes are positive, electrons negative

    return total_charge

# ============================================================
# Solve for equilibrium Fermi level at each temperature
# ============================================================
temperatures = np.concatenate([
    np.arange(300, 1000, 50),
    np.arange(1000, 2200, 100),
])

E_F_eq = []
concentrations = {defect["name"]: {str(q): [] for q in defect["charge_states"]}
                  for defect in db["defects"]}
electron_concs = []
hole_concs = []

for T in temperatures:
    # Find E_F where charge neutrality holds
    try:
        E_F_sol = brentq(
            charge_neutrality, 0.01, band_gap - 0.01,
            args=(T, db["defects"], band_gap, site_density),
            xtol=1e-6,
        )
    except ValueError:
        # Charge neutrality not satisfied in this range; use midgap
        E_F_sol = band_gap / 2

    E_F_eq.append(E_F_sol)

    # Compute individual defect concentrations at equilibrium E_F
    for defect in db["defects"]:
        for q_str, data in defect["charge_states"].items():
            q = int(q_str)
            e_f = data["e_form_at_vbm"] + q * E_F_sol
            c = defect_concentration(e_f, T, site_density)
            concentrations[defect["name"]][q_str].append(c)

    electron_concs.append(electron_concentration(E_F_sol, T, band_gap, N_C_300))
    hole_concs.append(hole_concentration(E_F_sol, T, N_V_300))

E_F_eq = np.array(E_F_eq)
electron_concs = np.array(electron_concs)
hole_concs = np.array(hole_concs)

# ============================================================
# Plot 1: Fermi level vs. temperature
# ============================================================
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

ax1.plot(temperatures, E_F_eq, "b-", linewidth=2)
ax1.axhline(y=0, color="gray", linestyle="--", alpha=0.5, label="VBM")
ax1.axhline(y=band_gap, color="gray", linestyle="--", alpha=0.5, label="CBM")
ax1.set_xlabel("Temperature (K)", fontsize=12)
ax1.set_ylabel("Equilibrium Fermi level (eV above VBM)", fontsize=12)
ax1.set_title("Fermi level vs. temperature", fontsize=13)
ax1.set_ylim(-0.5, band_gap + 0.5)
ax1.legend(fontsize=10)
ax1.grid(True, alpha=0.3)

# ============================================================
# Plot 2: Defect concentrations vs. 1000/T (Arrhenius)
# ============================================================
inv_T = 1000 / temperatures

# Plot each defect/charge state
defect_colors = {
    "V_O": "firebrick",
    "V_Mg": "steelblue",
    "Al_Mg": "forestgreen",
    "Li_Mg": "darkorange",
}
charge_styles = {-2: ":", -1: "--", 0: "-", 1: "-.", 2: ":"}

for defect in db["defects"]:
    name = defect["name"]
    color = defect_colors.get(name, "gray")
    for q_str in defect["charge_states"]:
        q = int(q_str)
        c_arr = np.array(concentrations[name][q_str])
        c_arr = np.maximum(c_arr, 1e-30)  # floor for log plot
        ls = charge_styles.get(q, "-")
        label = f"{name}$^{{{q:+d}}}$"
        ax2.semilogy(inv_T, c_arr, color=color, linestyle=ls,
                     linewidth=1.5, label=label)

# Free carriers
ax2.semilogy(inv_T, np.maximum(electron_concs, 1e-30),
             "k-", linewidth=1.5, label="n (electrons)")
ax2.semilogy(inv_T, np.maximum(hole_concs, 1e-30),
             "k--", linewidth=1.5, label="p (holes)")

ax2.set_xlabel("1000/T (K$^{-1}$)", fontsize=12)
ax2.set_ylabel("Concentration (cm$^{-3}$)", fontsize=12)
ax2.set_title("Defect concentrations (Arrhenius)", fontsize=13)
ax2.set_ylim(1e0, 1e23)
ax2.legend(fontsize=8, ncol=2, loc="upper right")
ax2.grid(True, alpha=0.3)

fig.tight_layout()
fig.savefig("defect_thermodynamics.png", dpi=150)
print("Defect thermodynamics plot saved to defect_thermodynamics.png")

# ============================================================
# Print summary at key temperatures
# ============================================================
for T_target in [300, 600, 1000, 1500, 2000]:
    idx = np.argmin(np.abs(temperatures - T_target))
    T = temperatures[idx]
    print(f"\n--- T = {T:.0f} K ---")
    print(f"  E_F = {E_F_eq[idx]:.4f} eV")
    print(f"  n = {electron_concs[idx]:.2e} cm^-3")
    print(f"  p = {hole_concs[idx]:.2e} cm^-3")
    for defect in db["defects"]:
        name = defect["name"]
        for q_str in defect["charge_states"]:
            c = concentrations[name][q_str][idx]
            if c > 1.0:
                print(f"  [{name}^{{{int(q_str):+d}}}] = {c:.2e} cm^-3")
```

#### Step 4: Chemical Potential Dependence

```python
#!/usr/bin/env python3
"""
Plot defect formation energies under different chemical potential conditions.
Shows how the defect landscape changes from O-rich to Mg-rich conditions.
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import json

# ============================================================
# Chemical potential bounds for MgO
# ============================================================
# Delta_Hf(MgO) ~ -6.1 eV (PBE value; experimental ~ -6.2 eV)
delta_Hf = -6.1

# O-rich limit:  Delta_mu_O = 0,   Delta_mu_Mg = Delta_Hf
# Mg-rich limit: Delta_mu_Mg = 0,  Delta_mu_O = Delta_Hf
# (Delta_mu = mu - mu_ref, always <= 0 for stable compounds)

conditions = {
    "O-rich": {"Delta_mu_Mg": delta_Hf, "Delta_mu_O": 0.0},
    "Intermediate": {"Delta_mu_Mg": delta_Hf / 2, "Delta_mu_O": delta_Hf / 2},
    "Mg-rich": {"Delta_mu_Mg": 0.0, "Delta_mu_O": delta_Hf},
}

# Defect formation energies at VBM need adjustment for chemical potentials.
# E_f(V_O, condition) = E_f(V_O, O-rich) - Delta_mu_O(condition)
# (removing O costs more when O is abundant)
#
# General formula for the chemical potential shift:
# For vacancy V_X:   shift = -Delta_mu_X (1 atom of X removed)
# For substitution A_B: shift = -Delta_mu_A + Delta_mu_B (A added, B removed)

defect_adjustments = {
    "V_O":   {"n_Mg": 0, "n_O": -1},   # removes 1 O
    "V_Mg":  {"n_Mg": -1, "n_O": 0},   # removes 1 Mg
    "Al_Mg": {"n_Mg": -1, "n_O": 0, "n_Al": 1},  # Al added, Mg removed
    "Li_Mg": {"n_Mg": -1, "n_O": 0, "n_Li": 1},  # Li added, Mg removed
}

# Base formation energies at VBM (O-rich reference from DFT)
# These are the q=most stable charge state at midgap as a representative
base_e_form = {
    "V_O": {"q": 2, "e_f_at_vbm_Orich": 3.1},
    "V_Mg": {"q": -2, "e_f_at_vbm_Orich": 9.8},
    "Al_Mg": {"q": 1, "e_f_at_vbm_Orich": 1.2},
    "Li_Mg": {"q": -1, "e_f_at_vbm_Orich": 4.5},
}

band_gap = 7.7
E_F = np.linspace(0, band_gap, 500)

fig, axes = plt.subplots(1, 3, figsize=(18, 6), sharey=True)

for ax, (cond_name, cond) in zip(axes, conditions.items()):
    ax.set_title(f"{cond_name}\n$\\Delta\\mu_{{Mg}}$ = {cond['Delta_mu_Mg']:.1f} eV, "
                 f"$\\Delta\\mu_O$ = {cond['Delta_mu_O']:.1f} eV",
                 fontsize=11)

    colors = {"V_O": "firebrick", "V_Mg": "steelblue",
              "Al_Mg": "forestgreen", "Li_Mg": "darkorange"}

    for defect_name, base in base_e_form.items():
        q = base["q"]
        e_f_base = base["e_f_at_vbm_Orich"]

        # Chemical potential adjustment
        adj = defect_adjustments[defect_name]
        shift = 0.0
        shift += adj.get("n_O", 0) * cond["Delta_mu_O"]
        shift += adj.get("n_Mg", 0) * cond["Delta_mu_Mg"]
        # For foreign species (Al, Li), their chemical potential is fixed
        # (independent of MgO growth conditions), so no shift needed here.

        e_f_adjusted = e_f_base + shift + q * E_F

        ax.plot(E_F, e_f_adjusted, color=colors[defect_name],
                linewidth=2, label=f"{defect_name}$^{{{q:+d}}}$")

    ax.set_xlim(0, band_gap)
    ax.set_ylim(bottom=0, top=15)
    ax.set_xlabel("Fermi level (eV above VBM)", fontsize=11)
    ax.axvline(x=0, color="black", linewidth=1.2)
    ax.axvline(x=band_gap, color="black", linewidth=1.2)
    ax.grid(True, alpha=0.2)
    ax.legend(fontsize=9)

axes[0].set_ylabel("Formation energy (eV)", fontsize=12)
fig.suptitle(f"Defect formation energies in MgO under different conditions",
             fontsize=14, y=1.02)
fig.tight_layout()
fig.savefig("defect_chemical_potential_dependence.png", dpi=150, bbox_inches="tight")
print("Chemical potential dependence plot saved to defect_chemical_potential_dependence.png")
```

### Method B: Neutral Defect Thermodynamics (MACE-Compatible)

```python
#!/usr/bin/env python3
"""
Simplified defect thermodynamics using only neutral formation energies.
Compatible with MACE (which cannot compute charged states).

For neutral defects, the formation energy is independent of the Fermi level.
Concentrations follow a simple Boltzmann distribution:
  c(T) = N_sites * exp(-E_f / k_B T)
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import json

kB = 8.617333262e-5  # eV/K

# Neutral formation energies from MACE calculations
neutral_defects = {
    "V_O":   {"e_form_eV": 7.5, "n_sites_per_fu": 1},
    "V_Mg":  {"e_form_eV": 7.8, "n_sites_per_fu": 1},
    "O_i":   {"e_form_eV": 9.2, "n_sites_per_fu": 1},
    "Mg_i":  {"e_form_eV": 10.5, "n_sites_per_fu": 1},
}

# Site density
a_MgO = 4.212e-8  # cm
n_fu_per_cell = 4
site_density = n_fu_per_cell / a_MgO**3

temperatures = np.arange(300, 2500, 10)

fig, ax = plt.subplots(figsize=(8, 6))

colors = {"V_O": "firebrick", "V_Mg": "steelblue",
          "O_i": "darkorange", "Mg_i": "forestgreen"}

for name, data in neutral_defects.items():
    e_f = data["e_form_eV"]
    n_sites = site_density * data["n_sites_per_fu"]

    conc = n_sites * np.exp(-e_f / (kB * temperatures))
    ax.semilogy(temperatures, conc, color=colors.get(name, "gray"),
                linewidth=2, label=f"{name} (E_f = {e_f:.1f} eV)")

ax.set_xlabel("Temperature (K)", fontsize=12)
ax.set_ylabel("Defect concentration (cm$^{-3}$)", fontsize=12)
ax.set_title("Neutral defect concentrations in MgO (MACE)", fontsize=13)
ax.set_ylim(1e0, 1e23)
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3)

# Add reference lines
ax.axhline(y=1e15, color="gray", linestyle=":", alpha=0.5)
ax.text(350, 2e15, "1e15 cm$^{-3}$ (typical significant level)",
        fontsize=8, color="gray")

fig.tight_layout()
fig.savefig("neutral_defect_concentrations.png", dpi=150)
print("Plot saved to neutral_defect_concentrations.png")

# Print concentrations at selected temperatures
print("\nDefect concentrations (cm^-3):")
print(f"{'Defect':<10}", end="")
for T in [300, 600, 1000, 1500, 2000]:
    print(f"{'T=' + str(T) + 'K':<15}", end="")
print()
print("-" * 80)
for name, data in neutral_defects.items():
    e_f = data["e_form_eV"]
    n_sites = site_density * data["n_sites_per_fu"]
    print(f"{name:<10}", end="")
    for T in [300, 600, 1000, 1500, 2000]:
        c = n_sites * np.exp(-e_f / (kB * T))
        print(f"{c:<15.2e}", end="")
    print()
```

### Method C: Automated Workflow with pymatgen-analysis-defects

```python
#!/usr/bin/env python3
"""
Use pymatgen-analysis-defects for automated defect thermodynamics.
This package provides high-level classes for managing defect calculations
and computing formation energy diagrams.

Requires: pip install pymatgen-analysis-defects
"""

print("""
=== pymatgen-analysis-defects Automated Workflow ===

Installation:
  pip install pymatgen-analysis-defects

Key classes:
  - DefectEntry: Stores a single defect calculation result
  - DefectPhaseDiagram / FormationEnergyDiagram: Plots E_f vs E_F
  - FreysoldtCorrection: Computes finite-size corrections

Workflow:
  1. Run DFT calculations (bulk + each defect + each charge state)
  2. Create DefectEntry objects from the results
  3. Build the formation energy diagram
  4. Compute transition levels and concentrations

Example usage:
""")

# NOTE: This code requires completed DFT calculations and
# pymatgen-analysis-defects installed. It shows the API pattern.

example_code = '''
from pymatgen.analysis.defects.thermo import DefectEntry, FormationEnergyDiagram
from pymatgen.analysis.defects.corrections.freysoldt import get_freysoldt_correction
from pymatgen.io.vasp import Vasprun, Locpot

# 1. Parse DFT results
bulk_vr = Vasprun("vasp_bulk/vasprun.xml")
bulk_energy = bulk_vr.final_energy
bulk_struct = bulk_vr.final_structure
bulk_vbm = bulk_vr.get_band_structure().get_vbm()["energy"]

# 2. Create DefectEntry for each defect/charge combination
entries = []
for defect_dir, defect_obj, charge in your_defect_list:
    vr = Vasprun(f"{defect_dir}/vasprun.xml")

    # Freysoldt correction for charged defects
    correction = 0.0
    if charge != 0:
        bulk_locpot = Locpot.from_file("vasp_bulk/LOCPOT")
        defect_locpot = Locpot.from_file(f"{defect_dir}/LOCPOT")
        correction = get_freysoldt_correction(
            charge, dielectric_constant, bulk_locpot, defect_locpot
        )

    entry = DefectEntry(
        defect=defect_obj,
        charge_state=charge,
        sc_entry=vr.get_computed_entry(),
        corrections={"freysoldt": correction},
    )
    entries.append(entry)

# 3. Build formation energy diagram
fed = FormationEnergyDiagram(
    bulk_entry=bulk_vr.get_computed_entry(),
    defect_entries=entries,
    vbm=bulk_vbm,
    band_gap=band_gap,
    # chemical potentials as a dict
)

# 4. Plot
fig = fed.get_plot()
fig.savefig("defect_diagram_auto.png", dpi=150)

# 5. Get transition levels
for entry in entries:
    tls = fed.get_transition_levels(entry)
    print(f"{entry.defect.name}: {tls}")
'''

print(example_code)
print("See pymatgen-analysis-defects documentation for full API reference.")
```

## Key Parameters

| Parameter | Recommended Value | Notes |
|---|---|---|
| Band gap | Experimental or HSE06 value | PBE underestimates; use experimental for E_f vs E_F plots |
| Chemical potentials | From phase diagram analysis | Constrained by thermodynamic stability of the host |
| Dielectric constant | DFT (DFPT) or experimental | Needed for Freysoldt correction; use static (ionic + electronic) |
| Site density | From crystal structure | Typically 1e22-1e23 cm^-3 for oxides |
| Effective DOS | ~1e19 cm^-3 at 300 K | Approximate; scale with T^{3/2} |
| Temperature range | 300-2000 K | Depends on material stability |
| Fermi level range | 0 to band gap | VBM = 0, CBM = band gap |
| Degeneracy factor g | 1-3 depending on defect symmetry | Spin degeneracy, orbital degeneracy |

## Interpreting Results

1. **Formation energy diagram**: The lowest line at each Fermi level gives the dominant charge state. Kinks (slope changes) correspond to charge transitions. Lower formation energies mean higher defect concentrations.

2. **Charge transition levels**: epsilon(q1/q2) is where two charge states have equal formation energy. Deep levels (far from band edges) create recombination centers; shallow levels (close to band edges) contribute free carriers.

3. **Donor vs. acceptor**: A defect is a donor if its transition level is close to the CBM (releases electrons). It is an acceptor if close to the VBM (captures electrons). A defect can be amphoteric (both donor and acceptor at different E_F).

4. **Negative-U behavior**: If epsilon(0/2+) lies below epsilon(0/1+), the q=+1 state is never thermodynamically stable. This is negative-U behavior, common for V_O in many oxides.

5. **Chemical potential dependence**: Formation energies shift rigidly with chemical potential changes. O-rich conditions favor V_Mg over V_O; Mg-rich conditions favor V_O. This determines intrinsic vs. extrinsic defect dominance.

6. **Equilibrium Fermi level**: The self-consistent Fermi level balances all charged defects and free carriers. In wide-gap insulators, it typically sits near midgap. Doping shifts it toward a band edge.

7. **Kroeger-Vink insight**: The notation immediately reveals the effective charge. Charge-neutral defect reactions (e.g., Schottky, Frenkel) maintain overall neutrality. The equilibrium constant for a defect reaction relates to the sum of formation energies.

## Common Issues

| Issue | Solution |
|---|---|
| PBE band gap too small for E_f vs E_F plot | Use experimental or HSE06 band gap; apply scissors correction if needed |
| Formation energy goes negative for some E_F range | Physical: means the defect forms spontaneously. The material is unstable under those conditions. |
| Charge neutrality solver does not converge | Broaden the search range; check that at least one positive and one negative charge defect exist |
| Defect concentrations seem unreasonably high | Check site density; ensure formation energy units are correct (eV, not Ry) |
| Transition level is outside the band gap | The charge state is never stable within the gap; it may pin the Fermi level at a band edge |
| MACE gives different formation energies than DFT | Expected; MACE is approximate. Use DFT values for the thermodynamic analysis. |
| Chemical potential bounds are unclear | Compute from the convex hull of competing phases using pymatgen PhaseDiagram |
| Spin degeneracy not accounted for | Include g = 2S+1 factor in the Boltzmann concentration formula |
| Results sensitive to supercell size | Ensure formation energies are converged (vacancy-formation/ skill); rerun with corrections |
