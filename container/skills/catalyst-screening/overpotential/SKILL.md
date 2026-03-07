# Electrocatalytic Overpotential Calculation

## When to Use

- Computing theoretical overpotential for oxygen evolution reaction (OER)
- Computing theoretical overpotential for oxygen reduction reaction (ORR)
- Computing theoretical overpotential for hydrogen evolution reaction (HER)
- Constructing free energy diagrams for electrocatalytic mechanisms
- Identifying the potential-determining step for a given catalyst
- Comparing catalyst activity at different applied potentials

## Prerequisites

- Python packages: `numpy`, `matplotlib`, `ase`
- MACE-torch for structure optimization and adsorption energies
- Quantum ESPRESSO for DFT reference calculations (optional, for higher accuracy)
- Tabulated ZPE and entropy corrections (provided below)

## Theory

### OER Mechanism (4-step, acidic conditions)

```
Step 1:  *       + H2O(l) -> OH*  + H+(aq) + e-     DeltaG1
Step 2:  OH*              -> O*   + H+(aq) + e-     DeltaG2
Step 3:  O*      + H2O(l) -> OOH* + H+(aq) + e-     DeltaG3
Step 4:  OOH*             -> *    + O2(g)  + H+(aq) + e-     DeltaG4
-----------------------------------------------------------------
Overall: 2 H2O(l) -> O2(g) + 4 H+(aq) + 4 e-    DeltaG = 4.92 eV
```

### ORR Mechanism (reverse of OER)

```
Step 1:  * + O2(g) + H+(aq) + e-  -> OOH*          DeltaG1 = -DeltaG4(OER)
Step 2:  OOH*      + H+(aq) + e-  -> O* + H2O(l)   DeltaG2 = -DeltaG3(OER)
Step 3:  O*        + H+(aq) + e-  -> OH*            DeltaG3 = -DeltaG2(OER)
Step 4:  OH*       + H+(aq) + e-  -> * + H2O(l)     DeltaG4 = -DeltaG1(OER)
```

### HER Mechanism

**Volmer-Heyrovsky:**
```
Step 1 (Volmer):    * + H+(aq) + e- -> H*              DeltaG1 = DeltaG_H*
Step 2 (Heyrovsky): H* + H+(aq) + e- -> * + H2(g)     DeltaG2 = -DeltaG_H*
```

**Volmer-Tafel:**
```
Step 1 (Volmer): * + H+(aq) + e- -> H*                DeltaG1 = DeltaG_H*
Step 2 (Tafel):  2 H* -> 2* + H2(g)                   DeltaG2 = -2 * DeltaG_H*
```

For HER, the optimal catalyst has DeltaG_H* ~ 0 eV (Sabatier principle).

### Free Energy Formula

```
DeltaG = DeltaE + DeltaZPE - T*DeltaS + n*e*U + kT*ln(10)*pH
```

where:
- DeltaE: DFT adsorption energy
- DeltaZPE: zero-point energy correction
- T*DeltaS: entropy correction at temperature T
- n*e*U: potential-dependent term (n electrons transferred, potential U vs RHE)
- pH term: for RHE scale, pH correction is already included (U_RHE = U_SHE + 0.059*pH)

**At RHE scale, the pH term drops out for proton-electron transfer steps.**

### Overpotential

```
eta_OER = max(DeltaG_i) / e - 1.23 V    (i = 1,2,3,4)
eta_ORR = 1.23 V - max(-DeltaG_i) / e   (reverse steps)
eta_HER = |DeltaG_H*| / e               (for both Volmer-Heyrovsky and Volmer-Tafel)
```

## Detailed Steps

### Step 1: Compute Adsorption Energies with MACE

```python
#!/usr/bin/env python3
"""
Complete overpotential calculation workflow.
Uses MACE for adsorption energy estimation, with DFT-quality corrections.
"""
import os
import numpy as np
import matplotlib.pyplot as plt
from ase.build import fcc111, add_adsorbate, molecule
from ase.optimize import BFGS
from ase.constraints import FixAtoms
from ase import Atoms
from ase.io import write, read

WORK_DIR = os.path.abspath("overpotential")
os.makedirs(WORK_DIR, exist_ok=True)

# ------------------------------------------------------------------
# Load MACE calculator
# ------------------------------------------------------------------
try:
    from mace.calculators import mace_mp
    calc = mace_mp(model="medium", device="cpu", default_dtype="float64")
    MACE_AVAILABLE = True
    print("MACE calculator loaded")
except ImportError:
    MACE_AVAILABLE = False
    print("MACE not available; using pre-computed reference values")

# ------------------------------------------------------------------
# Thermodynamic corrections (eV, at T = 298.15 K, P = 1 bar)
# Source: NIST-JANAF + computed ZPE from DFT (PBE)
# ------------------------------------------------------------------
# ZPE values for adsorbates and gas-phase molecules
ZPE = {
    "OH*":   0.355,   # adsorbed OH
    "O*":    0.065,   # adsorbed O (atomic)
    "OOH*":  0.400,   # adsorbed OOH
    "H*":    0.170,   # adsorbed H
    "H2O_g": 0.560,   # gas-phase H2O
    "H2_g":  0.268,   # gas-phase H2
    "O2_g":  0.098,   # gas-phase O2
}

# T*S values (entropy contributions at 298.15 K)
TS = {
    "OH*":   0.000,   # immobile adsorbate
    "O*":    0.000,
    "OOH*":  0.000,
    "H*":    0.000,
    "H2O_g": 0.670,   # gas phase (from NIST)
    "H2_g":  0.410,
    "O2_g":  0.635,
}

# ------------------------------------------------------------------
# Reference energies
# Using the standard computational hydrogen electrode (CHE):
#   G(H+ + e-) = 0.5 * G(H2) at U = 0 V vs RHE, pH = 0
#   G(O2) = 2*G(H2O) - 2*G(H2) + 4*1.23  (fixed by equilibrium)
# ------------------------------------------------------------------
# Equilibrium potentials
U_OER_eq = 1.23   # V vs RHE
U_ORR_eq = 1.23   # V vs RHE
U_HER_eq = 0.00   # V vs RHE


def compute_reference_energies(calc):
    """
    Compute gas-phase reference energies for H2O and H2 using MACE.

    Returns
    -------
    E_H2O : float (eV)
    E_H2 : float (eV)
    """
    # H2O
    h2o = molecule("H2O")
    h2o.cell = [15, 15.5, 16]
    h2o.pbc = True
    h2o.calc = calc
    opt = BFGS(h2o, logfile=None)
    opt.run(fmax=0.01, steps=100)
    E_H2O = h2o.get_potential_energy()

    # H2
    h2 = molecule("H2")
    h2.cell = [15, 15.5, 16]
    h2.pbc = True
    h2.calc = calc
    opt = BFGS(h2, logfile=None)
    opt.run(fmax=0.01, steps=100)
    E_H2 = h2.get_potential_energy()

    print(f"Reference energies: E(H2O) = {E_H2O:.4f} eV, E(H2) = {E_H2:.4f} eV")
    return E_H2O, E_H2


def build_slab(metal, a, size=(3, 3, 4), vacuum=15.0):
    """Build and relax a (111) surface slab."""
    slab = fcc111(metal, size=size, a=a, vacuum=vacuum, periodic=True)

    # Fix bottom 2 layers
    positions = slab.get_positions()
    z_coords = positions[:, 2]
    z_sorted = np.sort(np.unique(np.round(z_coords, 2)))
    z_fix = z_sorted[1] + 0.1
    slab.set_constraint(FixAtoms(mask=z_coords < z_fix))

    return slab


def optimize_structure(atoms, calc, fmax=0.03, steps=300, label=""):
    """Optimize structure with MACE."""
    atoms.calc = calc
    opt = BFGS(atoms, logfile=None)
    opt.run(fmax=fmax, steps=steps)
    energy = atoms.get_potential_energy()
    if label:
        print(f"  {label}: E = {energy:.4f} eV ({opt.nsteps} steps)")
    return energy


def compute_adsorption_energies(metal, a, calc, E_H2O, E_H2):
    """
    Compute adsorption energies of OER/ORR intermediates on a metal surface.

    DeltaE_OH  = E(slab+OH)  - E(slab) - E(H2O) + 0.5*E(H2)
    DeltaE_O   = E(slab+O)   - E(slab) - E(H2O) + E(H2)
    DeltaE_OOH = E(slab+OOH) - E(slab) - 2*E(H2O) + 1.5*E(H2)
    DeltaE_H   = E(slab+H)   - E(slab) - 0.5*E(H2)

    Parameters
    ----------
    metal : str
    a : float, lattice constant
    calc : ASE calculator
    E_H2O : float, gas-phase H2O energy
    E_H2 : float, gas-phase H2 energy

    Returns
    -------
    dict with adsorption energies (eV)
    """
    print(f"\nComputing adsorption energies for {metal}(111)...")

    # Build and optimize clean slab
    slab = build_slab(metal, a)
    E_slab = optimize_structure(slab, calc, label="clean slab")

    results = {}

    # --- OH* ---
    slab_oh = slab.copy()
    slab_oh.set_constraint()
    add_adsorbate(slab_oh, "OH", height=1.5, position="fcc")
    positions = slab_oh.get_positions()
    z_coords = positions[:, 2]
    z_sorted = np.sort(np.unique(np.round(z_coords[:len(slab)], 2)))
    z_fix = z_sorted[1] + 0.1
    slab_oh.set_constraint(FixAtoms(indices=[i for i in range(len(slab))
                                             if z_coords[i] < z_fix]))
    E_slab_oh = optimize_structure(slab_oh, calc, label="slab+OH")
    results["E_OH"] = E_slab_oh - E_slab - E_H2O + 0.5 * E_H2

    # --- O* ---
    slab_o = slab.copy()
    slab_o.set_constraint()
    add_adsorbate(slab_o, "O", height=1.3, position="fcc")
    positions = slab_o.get_positions()
    z_coords = positions[:, 2]
    slab_o.set_constraint(FixAtoms(indices=[i for i in range(len(slab))
                                            if z_coords[i] < z_fix]))
    E_slab_o = optimize_structure(slab_o, calc, label="slab+O")
    results["E_O"] = E_slab_o - E_slab - E_H2O + E_H2

    # --- OOH* ---
    slab_ooh = slab.copy()
    slab_ooh.set_constraint()
    # Build OOH manually: O-O-H with appropriate geometry
    ooh = Atoms("OOH", positions=[[0, 0, 0], [1.3, 0, 0.3], [1.8, 0.8, 0.6]])
    add_adsorbate(slab_ooh, ooh, height=1.8, position="ontop")
    positions = slab_ooh.get_positions()
    z_coords = positions[:, 2]
    n_slab = len(slab)
    slab_ooh.set_constraint(FixAtoms(indices=[i for i in range(n_slab)
                                              if z_coords[i] < z_fix]))
    E_slab_ooh = optimize_structure(slab_ooh, calc, label="slab+OOH")
    results["E_OOH"] = E_slab_ooh - E_slab - 2 * E_H2O + 1.5 * E_H2

    # --- H* (for HER) ---
    slab_h = slab.copy()
    slab_h.set_constraint()
    add_adsorbate(slab_h, "H", height=1.0, position="fcc")
    positions = slab_h.get_positions()
    z_coords = positions[:, 2]
    slab_h.set_constraint(FixAtoms(indices=[i for i in range(len(slab))
                                            if z_coords[i] < z_fix]))
    E_slab_h = optimize_structure(slab_h, calc, label="slab+H")
    results["E_H"] = E_slab_h - E_slab - 0.5 * E_H2

    print(f"\n  Results for {metal}(111):")
    for key, val in results.items():
        print(f"    {key} = {val:.3f} eV")

    return results


# ------------------------------------------------------------------
# Run for a target metal (or use pre-computed values)
# ------------------------------------------------------------------
if MACE_AVAILABLE:
    E_H2O, E_H2 = compute_reference_energies(calc)

    # Example: Pt(111)
    pt_results = compute_adsorption_energies("Pt", 3.924, calc, E_H2O, E_H2)
else:
    # Pre-computed DFT values for demonstration
    pt_results = {
        "E_OH":  0.80,
        "E_O":   1.57,
        "E_OOH": 3.95,
        "E_H":  -0.30,
    }
    print("Using pre-computed DFT values for Pt(111)")
    for key, val in pt_results.items():
        print(f"  {key} = {val:.3f} eV")
```

### Step 2: Compute Free Energy Steps and Overpotential

```python
#!/usr/bin/env python3
"""
Compute free energy steps and overpotential for OER, ORR, and HER.
"""
import numpy as np

# ------------------------------------------------------------------
# Corrections (from Step 1 definitions above)
# ------------------------------------------------------------------
ZPE = {
    "OH*": 0.355, "O*": 0.065, "OOH*": 0.400, "H*": 0.170,
    "H2O_g": 0.560, "H2_g": 0.268, "O2_g": 0.098,
}
TS = {
    "OH*": 0.000, "O*": 0.000, "OOH*": 0.000, "H*": 0.000,
    "H2O_g": 0.670, "H2_g": 0.410, "O2_g": 0.635,
}


def dft_energy_to_free_energy(E_ads, species):
    """
    Convert DFT adsorption energy to free energy.

    DeltaG = DeltaE + DeltaZPE - T*DeltaS

    For each OER step, the correction is pre-computed.
    """
    # Corrections per intermediate (DeltaZPE - T*DeltaS relative to references)
    corrections = {
        "OH":  (ZPE["OH*"] - ZPE["H2O_g"] + 0.5*ZPE["H2_g"])
               - (TS["OH*"] - TS["H2O_g"] + 0.5*TS["H2_g"]),
        "O":   (ZPE["O*"] - ZPE["H2O_g"] + ZPE["H2_g"])
               - (TS["O*"] - TS["H2O_g"] + TS["H2_g"]),
        "OOH": (ZPE["OOH*"] - 2*ZPE["H2O_g"] + 1.5*ZPE["H2_g"])
               - (TS["OOH*"] - 2*TS["H2O_g"] + 1.5*TS["H2_g"]),
        "H":   (ZPE["H*"] - 0.5*ZPE["H2_g"])
               - (TS["H*"] - 0.5*TS["H2_g"]),
    }
    return E_ads + corrections.get(species, 0.0)


def compute_oer_free_energies(E_OH, E_O, E_OOH, U=0.0, pH=0.0):
    """
    Compute OER free energy steps at potential U (vs RHE).

    DeltaG_i = DeltaG_i(U=0) - e*U   (each step involves 1 e-)

    pH correction is zero at RHE scale.

    Parameters
    ----------
    E_OH, E_O, E_OOH : float
        DFT adsorption energies (eV)
    U : float
        Applied potential vs RHE (V)
    pH : float
        pH value (0 for standard conditions at RHE)

    Returns
    -------
    dG : (4,) array of free energy steps (eV)
    eta : float, overpotential (V)
    pds : int, potential-determining step (1-4)
    G_levels : (5,) array, cumulative free energy levels for diagram
    """
    # Convert to free energies
    G_OH  = dft_energy_to_free_energy(E_OH, "OH")
    G_O   = dft_energy_to_free_energy(E_O, "O")
    G_OOH = dft_energy_to_free_energy(E_OOH, "OOH")

    # Free energy steps at U = 0
    dG = np.zeros(4)
    dG[0] = G_OH                           # * -> OH*
    dG[1] = G_O - G_OH                     # OH* -> O*
    dG[2] = G_OOH - G_O                    # O* -> OOH*
    dG[3] = 4.92 - G_OOH                   # OOH* -> O2

    # Apply potential
    # At RHE, pH correction is absorbed: G(H+ + e-) = 0.5*G(H2) - eU_RHE
    dG_U = dG - U  # each step transfers 1 electron

    # Overpotential
    pds = np.argmax(dG) + 1
    eta = np.max(dG) / 1.0 - 1.23

    # Cumulative free energy levels for diagram
    # Levels: *, OH*, O*, OOH*, O2 + *
    G_levels = np.zeros(5)
    G_levels[0] = 0.0                       # reference: *
    G_levels[1] = dG_U[0]                   # OH*
    G_levels[2] = G_levels[1] + dG_U[1]     # O*
    G_levels[3] = G_levels[2] + dG_U[2]     # OOH*
    G_levels[4] = G_levels[3] + dG_U[3]     # O2 + *

    return dG_U, eta, pds, G_levels


def compute_orr_free_energies(E_OH, E_O, E_OOH, U=0.0):
    """
    Compute ORR free energy steps (reverse of OER).
    At U = 1.23 V, all steps should be downhill for ideal catalyst.

    Returns dG_orr, eta_orr, pds_orr, G_levels_orr
    """
    G_OH  = dft_energy_to_free_energy(E_OH, "OH")
    G_O   = dft_energy_to_free_energy(E_O, "O")
    G_OOH = dft_energy_to_free_energy(E_OOH, "OOH")

    # ORR steps (reverse of OER)
    dG = np.zeros(4)
    dG[0] = -(4.92 - G_OOH)      # O2 -> OOH*
    dG[1] = -(G_OOH - G_O)       # OOH* -> O*
    dG[2] = -(G_O - G_OH)        # O* -> OH*
    dG[3] = -G_OH                 # OH* -> *

    # Apply potential (ORR at cathode, electrons are consumed)
    dG_U = dG + U  # each step consumes 1 electron at potential U

    # Overpotential: at U = 1.23 V, ideal catalyst has all dG <= 0
    # eta_ORR = max(dG at U=1.23) = 1.23 - min(-dG at U=0)
    pds = np.argmax(dG_U) + 1
    eta = 1.23 - np.min(np.abs(dG))  # limiting potential

    G_levels = np.zeros(5)
    G_levels[0] = 0.0               # O2 + *
    for i in range(4):
        G_levels[i+1] = G_levels[i] + dG_U[i]

    return dG_U, eta, pds, G_levels


def compute_her_free_energies(E_H, U=0.0):
    """
    Compute HER free energy (Volmer-Heyrovsky mechanism).

    DeltaG_H = E_H + DeltaZPE - T*DeltaS
    Overpotential = |DeltaG_H| (optimal at DeltaG_H = 0)

    Returns dG_steps, eta, G_levels
    """
    G_H = dft_energy_to_free_energy(E_H, "H")

    dG = np.zeros(2)
    dG[0] = G_H - U            # Volmer: H+ + e- -> H*
    dG[1] = -G_H - U           # Heyrovsky: H* + H+ + e- -> H2

    eta = abs(G_H)  # overpotential

    G_levels = np.zeros(3)
    G_levels[0] = 0.0           # H+ + e-
    G_levels[1] = dG[0]         # H*
    G_levels[2] = G_levels[1] + dG[1]   # H2

    return dG, eta, G_levels


# ------------------------------------------------------------------
# Example: Pt(111)
# ------------------------------------------------------------------
print("="*60)
print("Overpotential Calculation for Pt(111)")
print("="*60)

# Use pre-computed or MACE values
E_OH  = 0.80   # eV
E_O   = 1.57   # eV
E_OOH = 3.95   # eV
E_H   = -0.30  # eV

# OER
print("\n--- OER ---")
dG_oer, eta_oer, pds_oer, G_oer = compute_oer_free_energies(E_OH, E_O, E_OOH)
print(f"  DeltaG steps: {[f'{g:.3f}' for g in dG_oer]} eV")
print(f"  Potential-determining step: {pds_oer}")
print(f"  Overpotential: {eta_oer:.3f} V")

# ORR at U = 1.23 V
print("\n--- ORR (at U = 1.23 V) ---")
dG_orr, eta_orr, pds_orr, G_orr = compute_orr_free_energies(E_OH, E_O, E_OOH, U=1.23)
print(f"  DeltaG steps: {[f'{g:.3f}' for g in dG_orr]} eV")
print(f"  Potential-determining step: {pds_orr}")
print(f"  Overpotential: {eta_orr:.3f} V")

# HER
print("\n--- HER ---")
dG_her, eta_her, G_her = compute_her_free_energies(E_H)
print(f"  DeltaG_H = {dft_energy_to_free_energy(E_H, 'H'):.3f} eV")
print(f"  Overpotential: {eta_her:.3f} V")
```

### Step 3: Plot Free Energy Diagrams

```python
#!/usr/bin/env python3
"""
Plot free energy diagrams at multiple potentials.
"""
import numpy as np
import matplotlib.pyplot as plt
import os

WORK_DIR = os.path.abspath("overpotential")
os.makedirs(WORK_DIR, exist_ok=True)

# Re-use functions from Step 2 (or import them)
# ... (compute_oer_free_energies, etc. defined above)

# ZPE/TS corrections (from above)
ZPE = {
    "OH*": 0.355, "O*": 0.065, "OOH*": 0.400, "H*": 0.170,
    "H2O_g": 0.560, "H2_g": 0.268, "O2_g": 0.098,
}
TS = {
    "OH*": 0.000, "O*": 0.000, "OOH*": 0.000, "H*": 0.000,
    "H2O_g": 0.670, "H2_g": 0.410, "O2_g": 0.635,
}


def dft_energy_to_free_energy(E_ads, species):
    corrections = {
        "OH":  (ZPE["OH*"] - ZPE["H2O_g"] + 0.5*ZPE["H2_g"])
               - (TS["OH*"] - TS["H2O_g"] + 0.5*TS["H2_g"]),
        "O":   (ZPE["O*"] - ZPE["H2O_g"] + ZPE["H2_g"])
               - (TS["O*"] - TS["H2O_g"] + TS["H2_g"]),
        "OOH": (ZPE["OOH*"] - 2*ZPE["H2O_g"] + 1.5*ZPE["H2_g"])
               - (TS["OOH*"] - 2*TS["H2O_g"] + 1.5*TS["H2_g"]),
        "H":   (ZPE["H*"] - 0.5*ZPE["H2_g"])
               - (TS["H*"] - 0.5*TS["H2_g"]),
    }
    return E_ads + corrections.get(species, 0.0)


def compute_oer_free_energies(E_OH, E_O, E_OOH, U=0.0, pH=0.0):
    G_OH  = dft_energy_to_free_energy(E_OH, "OH")
    G_O   = dft_energy_to_free_energy(E_O, "O")
    G_OOH = dft_energy_to_free_energy(E_OOH, "OOH")
    dG = np.zeros(4)
    dG[0] = G_OH
    dG[1] = G_O - G_OH
    dG[2] = G_OOH - G_O
    dG[3] = 4.92 - G_OOH
    dG_U = dG - U
    pds = np.argmax(dG) + 1
    eta = np.max(dG) / 1.0 - 1.23
    G_levels = np.zeros(5)
    G_levels[0] = 0.0
    for i in range(4):
        G_levels[i+1] = G_levels[i] + dG_U[i]
    return dG_U, eta, pds, G_levels


def compute_her_free_energies(E_H, U=0.0):
    G_H = dft_energy_to_free_energy(E_H, "H")
    dG = np.zeros(2)
    dG[0] = G_H - U
    dG[1] = -G_H - U
    eta = abs(G_H)
    G_levels = np.zeros(3)
    G_levels[0] = 0.0
    G_levels[1] = dG[0]
    G_levels[2] = G_levels[1] + dG[1]
    return dG, eta, G_levels


def plot_oer_free_energy_diagram(E_OH, E_O, E_OOH, potentials, metal_name,
                                  save_path=None):
    """
    Plot OER free energy diagram at multiple potentials.

    Parameters
    ----------
    E_OH, E_O, E_OOH : float
        DFT adsorption energies (eV)
    potentials : list of float
        Applied potentials vs RHE (V)
    metal_name : str
        Label for the catalyst
    save_path : str or None
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    species_labels = ["$*$", "$OH^*$", "$O^*$", "$OOH^*$", "$O_2 + *$"]
    colors = plt.cm.viridis(np.linspace(0.1, 0.9, len(potentials)))

    step_width = 1.5  # width of each horizontal bar
    gap = 0.3         # gap between bars

    for idx, U in enumerate(potentials):
        dG_U, eta, pds, G_levels = compute_oer_free_energies(E_OH, E_O, E_OOH, U=U)
        color = colors[idx]
        label = f"U = {U:.2f} V"
        if U == 0:
            label = "U = 0 V"
        elif abs(U - 1.23) < 0.01:
            label = "U = 1.23 V (eq.)"

        for i in range(5):
            x_start = i * (step_width + gap)
            x_end = x_start + step_width
            ax.plot([x_start, x_end], [G_levels[i], G_levels[i]],
                    color=color, linewidth=2.5, solid_capstyle="round")

            # Connect to next level with dashed line
            if i < 4:
                x_next_start = (i + 1) * (step_width + gap)
                ax.plot([x_end, x_next_start], [G_levels[i], G_levels[i+1]],
                        color=color, linewidth=1.0, linestyle="--", alpha=0.5)

        # Add to legend
        ax.plot([], [], color=color, linewidth=2.5, label=label)

    # Add species labels at the bottom
    for i, label in enumerate(species_labels):
        x_center = i * (step_width + gap) + step_width / 2
        ax.text(x_center, ax.get_ylim()[0] - 0.15, label, ha="center",
                fontsize=12, fontweight="bold")

    # Ideal thermodynamic line (all steps equal at U = 1.23 V)
    ideal_levels = np.array([0, 1.23, 2.46, 3.69, 4.92]) - 1.23 * np.arange(5)
    for i in range(5):
        x_start = i * (step_width + gap)
        x_end = x_start + step_width
        ax.plot([x_start, x_end], [ideal_levels[i], ideal_levels[i]],
                color="gray", linewidth=1.0, linestyle=":", alpha=0.5)

    ax.set_ylabel(r"$\Delta G$ (eV)", fontsize=13)
    ax.set_title(f"OER Free Energy Diagram - {metal_name}", fontsize=14)
    ax.legend(fontsize=10, loc="upper left")
    ax.set_xlim(-0.5, 5 * (step_width + gap))

    # Remove x-axis ticks (labels are manual)
    ax.set_xticks([])

    # Add step labels
    step_labels = [r"$\Delta G_1$", r"$\Delta G_2$",
                   r"$\Delta G_3$", r"$\Delta G_4$"]
    dG_U0, _, pds, _ = compute_oer_free_energies(E_OH, E_O, E_OOH, U=0)
    for i, (label, dg) in enumerate(zip(step_labels, dG_U0)):
        x_mid = i * (step_width + gap) + step_width + gap / 2
        y_top = ax.get_ylim()[1] - 0.3
        marker = " **" if (i + 1) == pds else ""
        ax.text(x_mid, y_top, f"{label}\n{dg:.2f} eV{marker}",
                ha="center", fontsize=9, color="black", alpha=0.8)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"OER diagram saved to {save_path}")
    plt.close()


def plot_her_free_energy_diagram(E_H, metal_name, save_path=None):
    """
    Plot HER free energy diagram.
    """
    fig, ax = plt.subplots(figsize=(6, 5))

    G_H = dft_energy_to_free_energy(E_H, "H")

    species_labels = ["$H^+ + e^-$", "$H^*$", r"$\frac{1}{2}H_2$"]

    potentials = [0.0, -abs(G_H)]
    colors = ["steelblue", "red"]
    labels_U = ["U = 0 V", f"U = {-abs(G_H):.2f} V (onset)"]

    step_width = 1.5
    gap = 0.3

    for idx, U in enumerate(potentials):
        dG, eta, G_levels = compute_her_free_energies(E_H, U=U)

        for i in range(3):
            x_start = i * (step_width + gap)
            x_end = x_start + step_width
            ax.plot([x_start, x_end], [G_levels[i], G_levels[i]],
                    color=colors[idx], linewidth=2.5)
            if i < 2:
                x_next = (i + 1) * (step_width + gap)
                ax.plot([x_end, x_next], [G_levels[i], G_levels[i+1]],
                        color=colors[idx], linewidth=1.0, linestyle="--", alpha=0.5)

        ax.plot([], [], color=colors[idx], linewidth=2.5, label=labels_U[idx])

    for i, label in enumerate(species_labels):
        x_center = i * (step_width + gap) + step_width / 2
        ax.text(x_center, ax.get_ylim()[0] + 0.05, label, ha="center",
                fontsize=12, fontweight="bold",
                transform=ax.get_xaxis_transform())

    ax.set_ylabel(r"$\Delta G$ (eV)", fontsize=13)
    ax.set_title(f"HER Free Energy Diagram - {metal_name}\n"
                 f"$\\Delta G_H$ = {G_H:.3f} eV, $\\eta$ = {abs(G_H):.3f} V",
                 fontsize=13)
    ax.legend(fontsize=10)
    ax.set_xticks([])
    ax.axhline(y=0, color="gray", linewidth=0.5, linestyle=":")

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"HER diagram saved to {save_path}")
    plt.close()


# ------------------------------------------------------------------
# Generate plots for Pt(111)
# ------------------------------------------------------------------
E_OH, E_O, E_OOH, E_H = 0.80, 1.57, 3.95, -0.30

# OER diagram at multiple potentials
plot_oer_free_energy_diagram(
    E_OH, E_O, E_OOH,
    potentials=[0.0, 1.23, 1.23 + 0.56],  # U=0, equilibrium, onset
    metal_name="Pt(111)",
    save_path=os.path.join(WORK_DIR, "oer_diagram_Pt.png"),
)

# HER diagram
plot_her_free_energy_diagram(
    E_H,
    metal_name="Pt(111)",
    save_path=os.path.join(WORK_DIR, "her_diagram_Pt.png"),
)
```

### Step 4: Multi-Surface Comparison

```python
#!/usr/bin/env python3
"""
Compare overpotentials across multiple catalyst surfaces.
"""
import numpy as np
import matplotlib.pyplot as plt
import os

WORK_DIR = os.path.abspath("overpotential")
os.makedirs(WORK_DIR, exist_ok=True)

# ZPE/TS corrections
ZPE = {
    "OH*": 0.355, "O*": 0.065, "OOH*": 0.400, "H*": 0.170,
    "H2O_g": 0.560, "H2_g": 0.268, "O2_g": 0.098,
}
TS = {
    "OH*": 0.000, "O*": 0.000, "OOH*": 0.000, "H*": 0.000,
    "H2O_g": 0.670, "H2_g": 0.410, "O2_g": 0.635,
}


def dft_energy_to_free_energy(E_ads, species):
    corrections = {
        "OH":  (ZPE["OH*"] - ZPE["H2O_g"] + 0.5*ZPE["H2_g"])
               - (TS["OH*"] - TS["H2O_g"] + 0.5*TS["H2_g"]),
        "O":   (ZPE["O*"] - ZPE["H2O_g"] + ZPE["H2_g"])
               - (TS["O*"] - TS["H2O_g"] + TS["H2_g"]),
        "OOH": (ZPE["OOH*"] - 2*ZPE["H2O_g"] + 1.5*ZPE["H2_g"])
               - (TS["OOH*"] - 2*TS["H2O_g"] + 1.5*TS["H2_g"]),
        "H":   (ZPE["H*"] - 0.5*ZPE["H2_g"])
               - (TS["H*"] - 0.5*TS["H2_g"]),
    }
    return E_ads + corrections.get(species, 0.0)


# ------------------------------------------------------------------
# Database of adsorption energies and HER descriptor
# ------------------------------------------------------------------
CATALYST_DATA = {
    "Pt(111)":    {"E_OH": 0.80, "E_O": 1.57, "E_OOH": 3.95, "E_H": -0.30},
    "Pd(111)":    {"E_OH": 0.60, "E_O": 1.27, "E_OOH": 3.75, "E_H": -0.35},
    "Ir(111)":    {"E_OH": 0.55, "E_O": 1.10, "E_OOH": 3.70, "E_H": -0.25},
    "Au(111)":    {"E_OH": 1.70, "E_O": 3.20, "E_OOH": 4.90, "E_H":  0.40},
    "Ag(111)":    {"E_OH": 1.50, "E_O": 2.90, "E_OOH": 4.70, "E_H":  0.50},
    "Cu(111)":    {"E_OH": 0.35, "E_O": 0.70, "E_OOH": 3.55, "E_H": -0.15},
    "Ni(111)":    {"E_OH": 0.20, "E_O": 0.40, "E_OOH": 3.40, "E_H": -0.40},
    "RuO2(110)":  {"E_OH": 0.45, "E_O": 1.10, "E_OOH": 3.65, "E_H": -0.10},
    "IrO2(110)":  {"E_OH": 0.65, "E_O": 1.40, "E_OOH": 3.80, "E_H":  0.05},
}


def compute_overpotentials(data):
    """Compute OER and HER overpotentials for all catalysts."""
    results = {}
    for name, d in data.items():
        G_OH  = dft_energy_to_free_energy(d["E_OH"], "OH")
        G_O   = dft_energy_to_free_energy(d["E_O"], "O")
        G_OOH = dft_energy_to_free_energy(d["E_OOH"], "OOH")
        G_H   = dft_energy_to_free_energy(d["E_H"], "H")

        # OER
        dG_oer = [G_OH, G_O - G_OH, G_OOH - G_O, 4.92 - G_OOH]
        pds_oer = np.argmax(dG_oer) + 1
        eta_oer = max(dG_oer) - 1.23

        # HER
        eta_her = abs(G_H)

        results[name] = {
            "eta_oer": eta_oer,
            "pds_oer": pds_oer,
            "dG_oer": dG_oer,
            "eta_her": eta_her,
            "G_H": G_H,
        }

    return results


results = compute_overpotentials(CATALYST_DATA)

# Print summary
print("="*75)
print(f"{'Catalyst':>15s} | {'eta_OER (V)':>11s} | {'PDS':>3s} | "
      f"{'eta_HER (V)':>11s} | {'DeltaG_H (eV)':>13s}")
print("-"*75)
for name in CATALYST_DATA:
    r = results[name]
    print(f"{name:>15s} | {r['eta_oer']:>11.3f} | {r['pds_oer']:>3d} | "
          f"{r['eta_her']:>11.3f} | {r['G_H']:>13.3f}")
print("="*75)

# ------------------------------------------------------------------
# Plot: OER vs HER overpotential comparison
# ------------------------------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

catalyst_names = list(results.keys())
x_pos = range(len(catalyst_names))

# OER overpotentials
eta_oer_vals = [results[n]["eta_oer"] for n in catalyst_names]
colors_oer = ["#2ecc71" if eta < 0.5 else "#e74c3c" if eta > 1.0 else "#f39c12"
               for eta in eta_oer_vals]

axes[0].bar(x_pos, eta_oer_vals, color=colors_oer, edgecolor="black", linewidth=0.8)
axes[0].set_xticks(x_pos)
axes[0].set_xticklabels(catalyst_names, rotation=45, ha="right", fontsize=9)
axes[0].set_ylabel(r"$\eta_{OER}$ (V)", fontsize=12)
axes[0].set_title("OER Overpotential", fontsize=13)
axes[0].axhline(y=0.3, color="green", linewidth=1, linestyle="--", alpha=0.7,
                label=r"$\eta$ = 0.3 V target")
axes[0].legend(fontsize=9)

# HER overpotentials
eta_her_vals = [results[n]["eta_her"] for n in catalyst_names]
colors_her = ["#2ecc71" if eta < 0.2 else "#e74c3c" if eta > 0.5 else "#f39c12"
               for eta in eta_her_vals]

axes[1].bar(x_pos, eta_her_vals, color=colors_her, edgecolor="black", linewidth=0.8)
axes[1].set_xticks(x_pos)
axes[1].set_xticklabels(catalyst_names, rotation=45, ha="right", fontsize=9)
axes[1].set_ylabel(r"$\eta_{HER}$ (V)", fontsize=12)
axes[1].set_title("HER Overpotential", fontsize=13)
axes[1].axhline(y=0.1, color="green", linewidth=1, linestyle="--", alpha=0.7,
                label=r"$\eta$ = 0.1 V target")
axes[1].legend(fontsize=9)

plt.suptitle("Electrocatalytic Overpotentials", fontsize=15, y=1.02)
plt.tight_layout()
plt.savefig(os.path.join(WORK_DIR, "overpotential_comparison.png"),
            dpi=150, bbox_inches="tight")
plt.close()
print(f"\nComparison plot saved to {os.path.join(WORK_DIR, 'overpotential_comparison.png')}")

# ------------------------------------------------------------------
# Bifunctional activity plot (OER + ORR)
# ------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(8, 7))

for name in catalyst_names:
    r = results[name]
    ax.scatter(r["eta_oer"], r["eta_her"], s=120, edgecolor="black",
               zorder=5, linewidth=0.8)
    ax.annotate(name, (r["eta_oer"], r["eta_her"]),
                textcoords="offset points", xytext=(6, 6), fontsize=9)

# Ideal corner (low eta for both)
ax.axhline(y=0.1, color="green", linewidth=0.8, linestyle="--", alpha=0.5)
ax.axvline(x=0.3, color="green", linewidth=0.8, linestyle="--", alpha=0.5)
ax.fill_between([0, 0.3], [0, 0], [0.1, 0.1], color="green", alpha=0.05)
ax.text(0.15, 0.05, "Ideal\nregion", ha="center", fontsize=10, color="green",
        fontweight="bold", alpha=0.7)

ax.set_xlabel(r"$\eta_{OER}$ (V)", fontsize=13)
ax.set_ylabel(r"$\eta_{HER}$ (V)", fontsize=13)
ax.set_title("Bifunctional Electrocatalytic Activity", fontsize=14)
ax.set_xlim(0, max(eta_oer_vals) + 0.3)
ax.set_ylim(0, max(eta_her_vals) + 0.2)

plt.tight_layout()
plt.savefig(os.path.join(WORK_DIR, "bifunctional_activity.png"), dpi=150)
plt.close()
print(f"Bifunctional plot saved to {os.path.join(WORK_DIR, 'bifunctional_activity.png')}")
```

## Key Parameters

| Parameter | Typical Value | Notes |
|-----------|--------------|-------|
| OER equilibrium potential | 1.23 V vs RHE | Standard thermodynamic value |
| HER equilibrium potential | 0.00 V vs RHE | By definition at RHE |
| ZPE(OH*) | 0.355 eV | Depends on binding site and surface |
| ZPE(O*) | 0.065 eV | Small for atomic adsorbate |
| ZPE(OOH*) | 0.400 eV | Similar to OH* |
| ZPE(H*) | 0.170 eV | Depends on binding mode |
| TS(H2O, 298K) | 0.670 eV | From NIST-JANAF tables |
| TS(H2, 298K) | 0.410 eV | From NIST-JANAF tables |
| TS(O2, 298K) | 0.635 eV | From NIST-JANAF tables |
| DeltaG(OER total) | 4.92 eV | 4 x 1.23 eV, fixed by thermodynamics |
| Slab size | 3x3x4 minimum | Adsorbate coverage = 1/9 ML at 3x3 |
| Force convergence | 0.03 eV/Ang | For geometry optimization |

## Interpreting Results

**OER overpotential:**
- Best experimental OER catalysts: RuO2 (~0.37 V), IrO2 (~0.42 V)
- Theoretical minimum with OOH-OH scaling constraint: ~0.3-0.4 V
- Potential-determining step is usually step 2 (OH* -> O*) or step 3 (O* -> OOH*)
- Breaking the OOH-OH scaling relation is the key to lowering overpotential

**HER overpotential:**
- Pt is near-optimal: |DeltaG_H| ~ 0.09 eV
- Good HER catalysts: Pt, Pd, Ir, RuO2 (|DeltaG_H| < 0.2 eV)
- DeltaG_H > 0: weak H binding (right leg of HER volcano)
- DeltaG_H < 0: strong H binding (left leg of HER volcano)

**Free energy diagram:**
- At U = 0: all steps uphill (sum = 4.92 eV for OER)
- At U = 1.23 V: ideal catalyst has all steps = 1.23 eV (flat)
- At U = 1.23 + eta: all steps become downhill -> reaction onset
- The PDS (tallest step at U=0 minus 1.23) determines the overpotential

**ORR:**
- ORR is the reverse of OER
- Good OER catalyst is often a poor ORR catalyst and vice versa
- Pt(111) is better for ORR than OER

## Common Issues

1. **Overpotential is unrealistically low or negative:**
   - Check adsorption energy signs and references
   - Verify that E(H2O) and E(H2) reference energies are consistent
   - Ensure DeltaG1 + DeltaG2 + DeltaG3 + DeltaG4 = 4.92 eV (sum rule)

2. **Free energy corrections change the ranking:**
   - ZPE and entropy corrections are approximate (~0.1-0.2 eV uncertainty)
   - For precise ranking, compute system-specific ZPE from phonon calculations
   - Use consistent corrections across all surfaces being compared

3. **MACE gives different adsorption energies than DFT:**
   - MACE is a machine learning potential; accuracy varies
   - Validate on a few known systems before large-scale screening
   - MACE trends (ranking) are more reliable than absolute values
   - Apply a systematic offset if needed (calibrate against DFT)

4. **Adsorbate dissociates or desorbs during optimization:**
   - Start from a reasonable initial geometry
   - Reduce optimization step size
   - Constrain adsorbate height initially, then fully relax
   - Try different binding sites (ontop, bridge, fcc, hcp, hollow)

5. **OOH* is unstable on certain surfaces:**
   - OOH can dissociate into O* + OH* on reactive surfaces
   - This indicates the 4-step mechanism may not be valid
   - Consider alternative mechanisms (e.g., direct O-O coupling)

6. **pH dependence:**
   - At RHE scale, pH correction is zero for proton-coupled electron transfer
   - For non-PCET steps or at SHE scale, add kT*ln(10)*pH per proton
   - For alkaline conditions (OER in KOH), use OH- mechanism instead
