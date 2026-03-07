# Thermal Corrections for Catalysis

## When to Use

- Compute zero-point energy (ZPE) corrections for adsorbed species or gas-phase molecules
- Obtain Gibbs free energy corrections (ZPE + thermal enthalpy + entropy) at finite temperature
- Build free energy diagrams for catalytic reaction mechanisms
- Calculate thermodynamic corrections for adsorbates using the harmonic approximation
- Calculate thermodynamic properties of gas-phase molecules using the ideal gas approximation
- Corresponds to VASPKIT tasks 501 (ZPE for adsorbates) and 502 (thermodynamic corrections for molecules)

## Method Selection

| Criterion | ASE + MACE (Method A) | QE Finite Differences (Method B) | VASP (Method C) |
|---|---|---|---|
| Speed | Minutes | Hours | Hours |
| Accuracy | Good for screening | DFT quality | DFT quality |
| ZPE adsorbate | Harmonic approx on fixed slab | Harmonic approx on fixed slab | IBRION=5 or 6 |
| Gas-phase molecule | IdealGasThermo | IdealGasThermo from QE freqs | IBRION=5 |
| When to use | Rapid screening, trends | Publication quality | VASP license available |

```
Need quick ZPE/thermal corrections for screening?
  --> Method A: ASE + MACE harmonic frequencies (minutes)

Need publication-quality free energies?
  --> Method B: QE finite differences + ASE thermochemistry
  --> Method C: VASP IBRION=5 + VASPKIT 501/502

Adsorbate on surface?
  --> Use HarmonicThermo (fix slab, vibrate only adsorbate)

Gas-phase molecule?
  --> Use IdealGasThermo (includes translation, rotation, vibration)
```

## Prerequisites

- ASE with thermochemistry module (`ase.thermochemistry`)
- MACE-MP-0 (Method A)
- Quantum ESPRESSO pw.x (Method B)
- numpy, scipy, matplotlib
- A relaxed structure (adsorbate on slab or gas-phase molecule)

## Detailed Steps

### Method A: ASE + MACE -- Adsorbate on Surface (Harmonic Approximation)

```python
#!/usr/bin/env python3
"""
Compute ZPE and thermal corrections for an adsorbate on a surface
using MACE and the harmonic approximation.

For adsorbates, we freeze the slab atoms and compute vibrational
frequencies only for the adsorbate atoms. This follows VASPKIT 501.

Example: CO adsorbed on Cu(111).
"""

import numpy as np
from ase.build import fcc111, molecule
from ase.optimize import BFGS
from ase.constraints import FixAtoms
from ase.vibrations import Vibrations
from ase.thermochemistry import HarmonicThermo
from mace.calculators import mace_mp
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import json

# ============================================================
# 1. Build and relax adsorbate+slab system
# ============================================================
calc = mace_mp(model="medium", dispersion=False, default_dtype="float64")

# Build Cu(111) 3x3 slab with 4 layers + 15 A vacuum
slab = fcc111("Cu", size=(3, 3, 4), vacuum=15.0)
n_slab = len(slab)

# Add CO molecule (C-down) at a hollow site
co = molecule("CO")
co_height = 1.9  # initial C-surface distance in Angstrom

# Place CO above the center of the slab
slab_pos = slab.get_positions()
x_center = np.mean(slab_pos[:, 0])
y_center = np.mean(slab_pos[:, 1])
z_top = np.max(slab_pos[:, 2])

co.set_positions([
    [x_center, y_center, z_top + co_height],
    [x_center, y_center, z_top + co_height + 1.128],
])

from ase import Atoms
adslab = slab + co
n_total = len(adslab)
n_adsorbate = n_total - n_slab

print(f"System: {n_slab} slab atoms + {n_adsorbate} adsorbate atoms")

# Fix bottom 2 layers of slab
z_coords = adslab.get_positions()[:, 2]
z_layers = sorted(set(np.round(z_coords[:n_slab], 1)))
fix_threshold = z_layers[1] + 0.5  # fix bottom 2 layers
fix_mask = z_coords < fix_threshold
fix_mask[n_slab:] = False  # never fix adsorbate atoms

adslab.set_constraint(FixAtoms(mask=fix_mask))
adslab.calc = calc

# Relax (only adsorbate + top slab layers move)
print("\nRelaxing adsorbate+slab...")
opt = BFGS(adslab, logfile="relax_adslab.log")
opt.run(fmax=0.01, steps=300)
e_adslab = adslab.get_potential_energy()
print(f"E_adslab = {e_adslab:.6f} eV")

# ============================================================
# 2. Compute vibrational frequencies (adsorbate atoms only)
# ============================================================
# Identify adsorbate atom indices
adsorbate_indices = list(range(n_slab, n_total))
print(f"\nComputing vibrations for atoms: {adsorbate_indices}")

# Fix ALL slab atoms for the vibrational analysis
fix_slab = list(range(n_slab))
adslab.set_constraint(FixAtoms(indices=fix_slab))

vib = Vibrations(adslab, indices=adsorbate_indices, name="vib_adsorbate",
                 delta=0.01)  # displacement in Angstrom
vib.run()

# Get frequencies (in cm^-1)
freq_cm = vib.get_frequencies()
print(f"\nVibrational frequencies (cm^-1):")
for i, f in enumerate(freq_cm):
    if np.isreal(f):
        print(f"  Mode {i+1}: {f.real:10.2f} cm^-1")
    else:
        print(f"  Mode {i+1}: {f.real:10.2f} cm^-1 (imaginary: {f.imag:.2f}i)")

# Get energies in eV for thermochemistry
freq_ev = vib.get_energies()  # in eV, complex for imaginary modes

# Filter: keep only real, positive frequencies
real_freqs_ev = []
for f in freq_ev:
    if np.isreal(f) and f.real > 0.001:  # threshold: > ~8 cm^-1
        real_freqs_ev.append(f.real)
    elif np.isreal(f) and f.real > 0:
        print(f"  Skipping near-zero frequency: {f.real*8065.54:.1f} cm^-1")

print(f"\n{len(real_freqs_ev)} real positive frequencies retained")

# ============================================================
# 3. Compute ZPE and thermal corrections (Harmonic)
# ============================================================
# For adsorbates: use HarmonicThermo
# Assumes adsorbate has NO translational or rotational freedom
# (all DOF are vibrational when bound to surface)

thermo = HarmonicThermo(vib_energies=real_freqs_ev)

# ZPE
zpe = thermo.get_ZPE_correction()
print(f"\nZero-Point Energy (ZPE) = {zpe:.4f} eV")

# Thermal corrections at various temperatures
temperatures = [298.15, 400, 500, 600, 700, 800]
print(f"\n{'T (K)':<10} {'U_vib (eV)':<12} {'S_vib (eV/K)':<14} {'F_vib (eV)':<12}")
print("-" * 50)

results_T = []
for T in temperatures:
    # Internal energy (includes ZPE)
    u_vib = thermo.get_internal_energy(T, verbose=False)
    # Entropy contribution
    s_vib = thermo.get_entropy(T, verbose=False)
    # Helmholtz free energy: A = U - TS
    f_vib = thermo.get_helmholtz_energy(T, verbose=False)

    print(f"{T:<10.2f} {u_vib:<12.6f} {s_vib:<14.8f} {f_vib:<12.6f}")
    results_T.append({
        "T_K": T,
        "U_vib_eV": float(u_vib),
        "S_vib_eV_per_K": float(s_vib),
        "F_vib_eV": float(f_vib),
    })

# ============================================================
# 4. Save results
# ============================================================
results = {
    "system": "CO on Cu(111)",
    "n_slab_atoms": n_slab,
    "n_adsorbate_atoms": n_adsorbate,
    "e_adslab_eV": float(e_adslab),
    "frequencies_cm-1": [float(f.real * 8065.54) for f in freq_ev if np.isreal(f)],
    "zpe_eV": float(zpe),
    "thermal_corrections": results_T,
}

with open("thermal_corrections_adsorbate.json", "w") as f:
    json.dump(results, f, indent=2)
print("\nSaved to thermal_corrections_adsorbate.json")

vib.clean()
```

### Method A (continued): Gas-Phase Molecule (Ideal Gas Approximation)

```python
#!/usr/bin/env python3
"""
Compute ZPE and Gibbs free energy corrections for a gas-phase molecule
using MACE and the ideal gas approximation.

For gas-phase molecules, we include:
  - Translational entropy (Sackur-Tetrode equation)
  - Rotational entropy (rigid rotor)
  - Vibrational entropy and ZPE (harmonic oscillator)

This follows VASPKIT 502 functionality.

Example: CO, H2O, H2, CO2
"""

import numpy as np
from ase.build import molecule
from ase.optimize import BFGS
from ase.vibrations import Vibrations
from ase.thermochemistry import IdealGasThermo
from mace.calculators import mace_mp
import json

calc = mace_mp(model="medium", dispersion=False, default_dtype="float64")

# ============================================================
# 1. Define molecules to process
# ============================================================
molecules_info = {
    "H2": {"geometry": "linear", "spin": 0, "symmetrynumber": 2},
    "H2O": {"geometry": "nonlinear", "spin": 0, "symmetrynumber": 2},
    "CO": {"geometry": "linear", "spin": 0, "symmetrynumber": 1},
    "CO2": {"geometry": "linear", "spin": 0, "symmetrynumber": 2},
    "N2": {"geometry": "linear", "spin": 0, "symmetrynumber": 2},
    "O2": {"geometry": "linear", "spin": 1, "symmetrynumber": 2},
    "CH4": {"geometry": "nonlinear", "spin": 0, "symmetrynumber": 12},
    "NH3": {"geometry": "nonlinear", "spin": 0, "symmetrynumber": 3},
}

# Select which molecules to compute
target_molecules = ["H2", "H2O", "CO", "CO2"]

T = 298.15         # Temperature in K
P = 101325.0        # Pressure in Pa (1 atm)

all_results = {}

for mol_name in target_molecules:
    info = molecules_info[mol_name]
    print(f"\n{'='*50}")
    print(f"Processing: {mol_name}")
    print(f"{'='*50}")

    # ============================================================
    # 2. Build and relax molecule in vacuum box
    # ============================================================
    atoms = molecule(mol_name)
    atoms.center(vacuum=10.0)
    atoms.calc = calc

    opt = BFGS(atoms, logfile=f"relax_{mol_name}.log")
    opt.run(fmax=0.001, steps=200)
    e_pot = atoms.get_potential_energy()
    print(f"  Potential energy: {e_pot:.6f} eV")

    # ============================================================
    # 3. Compute vibrational frequencies
    # ============================================================
    vib = Vibrations(atoms, name=f"vib_{mol_name}", delta=0.01)
    vib.run()

    freq_ev = vib.get_energies()  # eV, complex for imaginary
    freq_cm = vib.get_frequencies()  # cm^-1

    print(f"  Frequencies (cm^-1):")
    for i, f in enumerate(freq_cm):
        tag = ""
        if not np.isreal(f):
            tag = f" (imaginary: {f.imag:.1f}i)"
        elif f.real < 10:
            tag = " (translation/rotation -- will be excluded)"
        print(f"    Mode {i+1}: {f.real:8.1f}{tag}")

    # Filter: keep only real, positive frequencies above threshold
    # For a molecule with N atoms:
    #   Linear: 3N-5 vibrational modes
    #   Nonlinear: 3N-6 vibrational modes
    n_atoms = len(atoms)
    if info["geometry"] == "linear":
        n_vib_expected = 3 * n_atoms - 5
    else:
        n_vib_expected = 3 * n_atoms - 6

    # Sort real positive frequencies by magnitude, take the top n_vib_expected
    real_pos = sorted(
        [f.real for f in freq_ev if np.isreal(f) and f.real > 0.001],
        reverse=True
    )
    vib_energies = real_pos[:n_vib_expected]
    print(f"  Expected {n_vib_expected} vibrational modes, using {len(vib_energies)}")

    # ============================================================
    # 4. Compute thermodynamic properties (Ideal Gas)
    # ============================================================
    thermo = IdealGasThermo(
        vib_energies=vib_energies,
        potentialenergy=e_pot,
        atoms=atoms,
        geometry=info["geometry"],
        symmetrynumber=info["symmetrynumber"],
        spin=info["spin"],
    )

    # ZPE
    zpe = sum(vib_energies) / 2
    print(f"\n  ZPE = {zpe:.4f} eV")

    # Enthalpy: H = U + PV = E_pot + ZPE + thermal_corrections + kT (PV=nRT for ideal gas)
    enthalpy = thermo.get_enthalpy(T, verbose=False)
    print(f"  Enthalpy H({T:.0f}K) = {enthalpy:.4f} eV")

    # Entropy
    entropy = thermo.get_entropy(T, P, verbose=False)
    print(f"  Entropy S({T:.0f}K) = {entropy:.6f} eV/K")
    print(f"  T*S = {T * entropy:.4f} eV")

    # Gibbs free energy: G = H - TS
    gibbs = thermo.get_gibbs_energy(T, P, verbose=False)
    print(f"  Gibbs free energy G({T:.0f}K) = {gibbs:.4f} eV")

    # Thermal correction: G - E_pot
    thermal_correction = gibbs - e_pot
    print(f"  Thermal correction (G - E_pot) = {thermal_correction:.4f} eV")

    all_results[mol_name] = {
        "e_pot_eV": float(e_pot),
        "zpe_eV": float(zpe),
        "enthalpy_eV": float(enthalpy),
        "entropy_eV_per_K": float(entropy),
        "TS_eV": float(T * entropy),
        "gibbs_eV": float(gibbs),
        "thermal_correction_eV": float(thermal_correction),
        "frequencies_cm-1": [float(f.real) for f in freq_cm if np.isreal(f)],
        "geometry": info["geometry"],
        "symmetrynumber": info["symmetrynumber"],
    }

    vib.clean()

# ============================================================
# 5. Summary table
# ============================================================
print(f"\n{'='*70}")
print(f"SUMMARY at T={T:.2f} K, P={P:.0f} Pa")
print(f"{'='*70}")
print(f"{'Molecule':<10} {'E_pot (eV)':<12} {'ZPE (eV)':<10} {'G (eV)':<12} {'G-E (eV)':<10}")
print("-" * 55)
for mol_name in target_molecules:
    r = all_results[mol_name]
    print(f"{mol_name:<10} {r['e_pot_eV']:<12.4f} {r['zpe_eV']:<10.4f} "
          f"{r['gibbs_eV']:<12.4f} {r['thermal_correction_eV']:<10.4f}")

with open("thermal_corrections_gas.json", "w") as f:
    json.dump(all_results, f, indent=2)
print("\nSaved to thermal_corrections_gas.json")
```

### Method A (continued): Free Energy Diagram Construction

```python
#!/usr/bin/env python3
"""
Build a free energy diagram for a catalytic reaction using
DFT energies + thermal corrections.

Example: Oxygen Reduction Reaction (ORR) on a metal surface.
4-electron pathway:
  O2 + * -> OOH*
  OOH* -> O* + OH-
  O* + H2O -> OH* + OH-
  OH* -> * + OH-

This script takes pre-computed energies and thermal corrections
and builds the free energy diagram at a given potential U.
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ============================================================
# 1. Input: DFT energies and thermal corrections
# ============================================================
# These would come from your calculations (Methods A or B)
# Replace with actual computed values

# Reference energies (eV)
E_H2O_gas = -14.213      # gas-phase H2O (DFT)
E_H2_gas = -6.770        # gas-phase H2 (DFT)

# ZPE and thermal corrections (G - E) at 298.15 K
# From IdealGasThermo for gas molecules
dG_H2O = 0.57            # thermal correction for H2O
dG_H2 = 0.27             # thermal correction for H2

# Slab + adsorbate DFT energies
E_slab = -236.500         # clean slab
E_OOH_slab = -253.120     # OOH* on slab
E_O_slab = -241.890       # O* on slab
E_OH_slab = -243.200      # OH* on slab

# ZPE corrections for adsorbates (from HarmonicThermo)
ZPE_OOH = 0.44
ZPE_O = 0.07
ZPE_OH = 0.36

# Entropy corrections for adsorbates (T*S at 298 K) -- typically small
TS_OOH = 0.00
TS_O = 0.00
TS_OH = 0.00

# ============================================================
# 2. Compute adsorption free energies
# ============================================================
# Standard hydrogen electrode (SHE):
# G(H+ + e-) = 0.5 * G(H2) at U=0 V, pH=0
# At potential U: G(H+ + e-) = 0.5 * G(H2) - eU

# Free energy of liquid water
G_H2O = E_H2O_gas + dG_H2O
G_H2 = E_H2_gas + dG_H2

# Free energies of adsorbed species
G_OOH = E_OOH_slab + ZPE_OOH - TS_OOH
G_O = E_O_slab + ZPE_O - TS_O
G_OH = E_OH_slab + ZPE_OH - TS_OH
G_slab = E_slab

# Adsorption free energies referenced to H2O and H2
# dG_OOH* = G(OOH*) - G(*) - 2*G(H2O) + 3/2*G(H2)
dG_OOH_ads = G_OOH - G_slab - 2 * G_H2O + 1.5 * G_H2
# dG_O* = G(O*) - G(*) - G(H2O) + G(H2)
dG_O_ads = G_O - G_slab - G_H2O + G_H2
# dG_OH* = G(OH*) - G(*) - G(H2O) + 0.5*G(H2)
dG_OH_ads = G_OH - G_slab - G_H2O + 0.5 * G_H2

# Standard Gibbs free energy of ORR: O2 + 2H2 -> 2H2O, dG = -4.92 eV
dG_ORR = -4.92

print("Adsorption free energies (at U=0 V):")
print(f"  dG(OOH*) = {dG_OOH_ads:.4f} eV")
print(f"  dG(O*)   = {dG_O_ads:.4f} eV")
print(f"  dG(OH*)  = {dG_OH_ads:.4f} eV")

# ============================================================
# 3. Build free energy diagram at various potentials
# ============================================================
def orr_free_energy_steps(dG_OOH, dG_O, dG_OH, U=0.0):
    """
    Compute free energy changes for each ORR step at potential U.

    Step 1: O2 + H+ + e- -> OOH*
    Step 2: OOH* + H+ + e- -> O* + H2O
    Step 3: O* + H+ + e- -> OH*
    Step 4: OH* + H+ + e- -> * + H2O

    Each step involves one (H+ + e-) transfer.
    At potential U: each step shifts by -eU.
    """
    dG_ORR = -4.92  # total ORR free energy

    dG1 = dG_OOH - dG_ORR + (-0) - (-1) * U  # simplified
    # More precisely:
    # G levels: * + O2 + 4(H++e-) -> OOH* + 3(H++e-) -> O* + H2O + 2(H++e-)
    #           -> OH* + H2O + (H++e-) -> * + 2H2O

    # Step free energies at U=0:
    dG1 = dG_OOH - dG_ORR  # O2 -> OOH*
    dG2 = dG_O - dG_OOH     # OOH* -> O* + H2O
    dG3 = dG_OH - dG_O       # O* -> OH*
    dG4 = -dG_OH              # OH* -> * + H2O

    # Correct: at the equilibrium potential U_eq = 1.23 V, all steps have dG=0
    # Step energies at U=0 V (vs SHE):
    dG1_U0 = 4.92 + dG_OOH   # should be ~0 at ideal catalyst
    dG2_U0 = -dG_OOH + dG_O + (-4.92 + 4.92)  # simplified
    # Let's use the standard formulation directly:

    # Free energy levels (cumulative):
    G0 = 0.0                              # * + O2 + 4(H++e-)
    G1 = dG_OOH + 4.92                    # OOH* + 3(H++e-)
    G2 = dG_O + 4.92 - (-0)              # O* + H2O + 2(H++e-)
    G3 = dG_OH + 4.92 - (-0)             # OH* + H2O + (H++e-)
    G4 = 0.0                              # * + 2H2O (product)

    # Simpler: Norskov's ORR formulation
    # dG1 = dG_OOH - 4.92
    # dG2 = dG_O - dG_OOH
    # dG3 = dG_OH - dG_O
    # dG4 = -dG_OH
    dG1 = dG_OOH - 4.92
    dG2 = dG_O - dG_OOH
    dG3 = dG_OH - dG_O
    dG4 = -dG_OH

    # Apply potential: each electron transfer step shifts by -eU
    dG1_U = dG1 + U
    dG2_U = dG2 + U
    dG3_U = dG3 + U
    dG4_U = dG4 + U

    return [dG1_U, dG2_U, dG3_U, dG4_U]

# ============================================================
# 4. Plot free energy diagram
# ============================================================
fig, ax = plt.subplots(figsize=(10, 6))

potentials = [0.0, 0.5, 1.23]
colors = ["#2ecc71", "#3498db", "#e74c3c"]
labels_U = [f"U = {u:.2f} V" for u in potentials]

step_labels = [
    r"$*$ + O$_2$",
    r"OOH$*$",
    r"O$*$ + H$_2$O",
    r"OH$*$ + H$_2$O",
    r"$*$ + 2H$_2$O",
]

for U, color, label in zip(potentials, colors, labels_U):
    dGs = orr_free_energy_steps(dG_OOH_ads, dG_O_ads, dG_OH_ads, U)

    # Cumulative free energy levels
    levels = [0.0]
    for dg in dGs:
        levels.append(levels[-1] + dg)

    # Plot horizontal lines for each state
    x_positions = np.arange(len(levels))
    width = 0.35

    for i, (x, g) in enumerate(zip(x_positions, levels)):
        ax.plot([x - width, x + width], [g, g], color=color, linewidth=2.5)
        if i < len(levels) - 1:
            ax.plot([x + width, x + 1 - width], [levels[i], levels[i + 1]],
                    color=color, linewidth=1, linestyle="--", alpha=0.5)

    # Label the first point only (for legend)
    ax.plot([], [], color=color, linewidth=2.5, label=label)

ax.set_xticks(x_positions)
ax.set_xticklabels(step_labels, fontsize=11)
ax.set_ylabel("Free energy (eV)", fontsize=13)
ax.set_title("ORR Free Energy Diagram", fontsize=14)
ax.axhline(y=0, color="gray", linewidth=0.5, linestyle="-")
ax.legend(fontsize=11, loc="upper right")
ax.grid(axis="y", alpha=0.3)

fig.tight_layout()
fig.savefig("free_energy_diagram.png", dpi=150)
print("\nSaved free_energy_diagram.png")

# Overpotential
dGs_eq = orr_free_energy_steps(dG_OOH_ads, dG_O_ads, dG_OH_ads, U=1.23)
max_dG = max(dGs_eq)
eta = max_dG  # overpotential
print(f"\nPotential-determining step at U=1.23V: max(dG) = {max_dG:.4f} eV")
print(f"Overpotential eta = {eta:.4f} V")
```

### Method B: QE Finite Differences + ASE Thermochemistry

```python
#!/usr/bin/env python3
"""
Compute vibrational frequencies using QE finite differences,
then apply ASE thermochemistry for ZPE and thermal corrections.

Step 1: Run QE SCF for each displaced configuration
Step 2: Extract forces and compute frequencies
Step 3: Apply ASE HarmonicThermo or IdealGasThermo
"""

import numpy as np
import subprocess
import os
import json
from ase.io import read
from ase.thermochemistry import HarmonicThermo, IdealGasThermo

# ============================================================
# 1. Configuration
# ============================================================
STRUCTURE_FILE = "relaxed_molecule.cif"  # relaxed structure
PSEUDO_DIR = os.path.abspath("./pseudo")
WORK_DIR = os.path.abspath("./qe_freq")
os.makedirs(WORK_DIR, exist_ok=True)

ECUTWFC = 50.0
ECUTRHO = 400.0
DELTA = 0.01  # displacement in Angstrom

# Read structure
atoms = read(STRUCTURE_FILE)
n_atoms = len(atoms)
positions = atoms.get_positions().copy()
cell = atoms.get_cell()

# Element info
elements = sorted(set(atoms.get_chemical_symbols()))
pseudos = {
    "C": "C.pbe-n-kjpaw_psl.1.0.0.UPF",
    "O": "O.pbe-n-kjpaw_psl.1.0.0.UPF",
    "H": "H.pbe-kjpaw_psl.1.0.0.UPF",
    "Cu": "Cu.pbe-dn-kjpaw_psl.1.0.0.UPF",
}

print(f"System: {atoms.get_chemical_formula()}, {n_atoms} atoms")


def write_qe_input(atoms, filename, pseudo_dir, pseudos, ecutwfc, ecutrho):
    """Write a QE SCF input file for force calculation."""
    symbols = atoms.get_chemical_symbols()
    positions = atoms.get_positions()
    cell = atoms.get_cell()
    elements = sorted(set(symbols))

    from pymatgen.core.periodic_table import Element as PmgElement

    inp = f"""&CONTROL
    calculation = 'scf'
    prefix      = 'freq'
    outdir      = './tmp'
    pseudo_dir  = '{pseudo_dir}'
    tprnfor     = .true.
/
&SYSTEM
    ibrav       = 0
    nat         = {len(atoms)}
    ntyp        = {len(elements)}
    ecutwfc     = {ecutwfc}
    ecutrho     = {ecutrho}
    occupations = 'smearing'
    smearing    = 'gaussian'
    degauss     = 0.005
/
&ELECTRONS
    conv_thr = 1.0d-8
/

ATOMIC_SPECIES
"""
    for el in elements:
        mass = PmgElement(el).atomic_mass
        inp += f"  {el}  {mass:.4f}  {pseudos[el]}\n"

    inp += "\nCELL_PARAMETERS angstrom\n"
    for vec in cell:
        inp += f"  {vec[0]:.10f}  {vec[1]:.10f}  {vec[2]:.10f}\n"

    inp += "\nATOMIC_POSITIONS angstrom\n"
    for sym, pos in zip(symbols, positions):
        inp += f"  {sym}  {pos[0]:.10f}  {pos[1]:.10f}  {pos[2]:.10f}\n"

    inp += "\nK_POINTS gamma\n"

    with open(filename, "w") as f:
        f.write(inp)


def parse_qe_forces(output_file):
    """Parse forces from QE SCF output. Returns forces in eV/Angstrom."""
    import re
    forces = []
    in_forces = False
    with open(output_file) as f:
        for line in f:
            if "Forces acting on atoms" in line:
                in_forces = True
                forces = []
                continue
            if in_forces and "force =" in line:
                parts = line.split("force =")[1].split()
                fx, fy, fz = float(parts[0]), float(parts[1]), float(parts[2])
                # QE forces are in Ry/Bohr, convert to eV/Angstrom
                # 1 Ry/Bohr = 25.7112 eV/Angstrom
                forces.append([fx * 25.7112, fy * 25.7112, fz * 25.7112])
            if in_forces and line.strip() == "":
                if forces:
                    in_forces = False
    return np.array(forces)


# ============================================================
# 2. Generate displaced structures and run QE
# ============================================================
# Which atoms to displace (for adsorbate on slab, only adsorbate atoms)
# For gas-phase molecule, displace all atoms
displace_indices = list(range(n_atoms))
n_disp = len(displace_indices)

print(f"Generating {6 * n_disp} displaced configurations...")

# Collect forces: force_matrix[atom_idx][direction][+/-]
force_data = {}

nproc = os.cpu_count() or 4

for atom_i in displace_indices:
    force_data[atom_i] = {}
    for direction in range(3):  # x, y, z
        force_data[atom_i][direction] = {}
        for sign, sign_label in [(+1, "plus"), (-1, "minus")]:
            # Create displaced structure
            disp_atoms = atoms.copy()
            disp_pos = positions.copy()
            disp_pos[atom_i, direction] += sign * DELTA
            disp_atoms.set_positions(disp_pos)

            # Write QE input
            label = f"disp_a{atom_i}_d{direction}_{sign_label}"
            inp_file = os.path.join(WORK_DIR, f"{label}.in")
            out_file = os.path.join(WORK_DIR, f"{label}.out")

            write_qe_input(disp_atoms, inp_file, PSEUDO_DIR, pseudos,
                           ECUTWFC, ECUTRHO)

            # Run QE SCF
            os.makedirs(os.path.join(WORK_DIR, "tmp"), exist_ok=True)
            result = subprocess.run(
                ["mpirun", "--allow-run-as-root", "-np", str(nproc),
                 "pw.x", "-in", inp_file],
                capture_output=True, text=True, timeout=600,
                cwd=WORK_DIR,
            )
            with open(out_file, "w") as f:
                f.write(result.stdout)

            # Parse forces
            forces = parse_qe_forces(out_file)
            force_data[atom_i][direction][sign_label] = forces

    print(f"  Atom {atom_i} done")

# ============================================================
# 3. Build Hessian and compute frequencies
# ============================================================
n_modes = 3 * n_disp
hessian = np.zeros((n_modes, n_modes))

for i, atom_i in enumerate(displace_indices):
    for d in range(3):
        row = 3 * i + d
        f_plus = force_data[atom_i][d]["plus"]
        f_minus = force_data[atom_i][d]["minus"]

        # Central difference: dF/dx = (F(+dx) - F(-dx)) / (2*dx)
        for j, atom_j in enumerate(displace_indices):
            for dd in range(3):
                col = 3 * j + dd
                # Hessian: H_ij = -dF_i/dx_j
                hessian[row, col] = -(f_plus[atom_j, dd] - f_minus[atom_j, dd]) / (2 * DELTA)

# Symmetrize
hessian = 0.5 * (hessian + hessian.T)

# Mass-weight the Hessian
from ase.data import atomic_masses, atomic_numbers
from ase.units import _hbar, _e, _amu

masses = atoms.get_masses()
disp_masses = masses[displace_indices]

mass_vec = np.repeat(disp_masses, 3)
mass_matrix = np.sqrt(np.outer(mass_vec, mass_vec))
dyn_matrix = hessian / mass_matrix

# Diagonalize
eigenvalues, eigenvectors = np.linalg.eigh(dyn_matrix)

# Convert eigenvalues to frequencies
# eigenvalue units: eV/(Angstrom^2 * amu)
# frequency = sqrt(eigenvalue) / (2*pi), convert to eV
# Using ASE units: 1 eV = 1.602e-19 J, 1 amu = 1.661e-27 kg, 1 A = 1e-10 m
freq_factor = np.sqrt(abs(eigenvalues))
# Convert to eV: hbar * omega
# omega = sqrt(k/m), k in eV/A^2, m in amu
# omega (rad/s) = sqrt(k * eV_to_J / (m * amu_to_kg * A_to_m^2))
eV_to_J = 1.602176634e-19
amu_to_kg = 1.66053906660e-27
A_to_m = 1e-10
hbar_eV_s = 6.582119569e-16  # hbar in eV*s

frequencies_eV = []
frequencies_cm = []

for ev in eigenvalues:
    if ev > 0:
        omega = np.sqrt(ev * eV_to_J / (amu_to_kg * A_to_m**2))
        freq_eV = hbar_eV_s * omega
        freq_cm_val = freq_eV * 8065.54
    else:
        omega = np.sqrt(abs(ev) * eV_to_J / (amu_to_kg * A_to_m**2))
        freq_eV = -hbar_eV_s * omega  # negative = imaginary
        freq_cm_val = -freq_eV * 8065.54
    frequencies_eV.append(freq_eV)
    frequencies_cm.append(freq_cm_val)

print("\nVibrational frequencies from QE finite differences:")
for i, (fev, fcm) in enumerate(zip(frequencies_eV, frequencies_cm)):
    tag = " (imaginary)" if fev < 0 else ""
    print(f"  Mode {i+1}: {abs(fcm):8.1f} cm^-1  ({abs(fev)*1000:8.3f} meV){tag}")

# ============================================================
# 4. Apply ASE thermochemistry
# ============================================================
# Keep only real positive frequencies
real_pos_eV = [f for f in frequencies_eV if f > 0.001 * 0.001]  # > ~1 meV

# For adsorbate: HarmonicThermo
thermo = HarmonicThermo(vib_energies=real_pos_eV)
T = 298.15
zpe = thermo.get_ZPE_correction()
helmholtz = thermo.get_helmholtz_energy(T, verbose=False)

print(f"\nZPE = {zpe:.4f} eV")
print(f"Helmholtz free energy at {T} K = {helmholtz:.4f} eV")

results = {
    "frequencies_cm-1": frequencies_cm,
    "frequencies_eV": frequencies_eV,
    "zpe_eV": float(zpe),
    "helmholtz_eV": float(helmholtz),
    "T_K": T,
}
with open(os.path.join(WORK_DIR, "qe_freq_results.json"), "w") as f:
    json.dump(results, f, indent=2)
print(f"Saved to {WORK_DIR}/qe_freq_results.json")
```

### Method C: VASP Workflow (Future)

```
VASP thermal corrections workflow (requires VASP license):

1. Relax structure:
   IBRION = 2, ISIF = 2 (adsorbate) or ISIF = 3 (bulk/molecule)
   NSW = 300, EDIFF = 1E-6, EDIFFG = -0.01

2. Frequency calculation:
   IBRION = 5 (finite differences) or IBRION = 6 (DFPT for VASP >=6)
   NSW = 1, NFREE = 2, POTIM = 0.015
   For adsorbates: use selective dynamics to freeze slab atoms

3. Post-process with VASPKIT:
   - Task 501: ZPE and thermal corrections for adsorbates
     Reads OUTCAR frequencies, applies harmonic approximation
   - Task 502: Thermodynamic properties for gas-phase molecules
     Reads OUTCAR frequencies, applies ideal gas + rigid rotor + harmonic oscillator

4. VASPKIT 501 output: THERMO_ADSORBATE.dat
   Contains: ZPE, thermal energy U(T), entropy S(T), free energy F(T)

5. VASPKIT 502 output: THERMO_MOLECULE.dat
   Contains: ZPE, H(T), S(T), G(T) with translational/rotational contributions

Note: When VASP is available, generate POSCAR and INCAR with the above settings,
run VASP, then use VASPKIT or the ASE thermochemistry module (parsing OUTCAR)
to extract thermal corrections.
```

## Key Parameters

| Parameter | Recommended Value | Notes |
|---|---|---|
| Displacement delta | 0.01 A | For finite differences. 0.015 for VASP POTIM. |
| fmax for relaxation | 0.001-0.01 eV/A | Must be very tight before frequency calculation |
| Temperature | 298.15 K | Standard conditions; adjust for operating T |
| Pressure | 101325 Pa | Standard pressure for gas-phase molecules |
| Frequency threshold | > 50 cm^-1 | Ignore very low frequencies (numerical noise) |
| Symmetry number | Molecule-dependent | H2: 2, H2O: 2, NH3: 3, CH4: 12, CO: 1 |
| Spin multiplicity | Molecule-dependent | O2: triplet (spin=1), most closed-shell: spin=0 |
| Vacuum box for molecule | 10-15 A | Large enough to avoid periodic interactions |

## Interpreting Results

1. **ZPE magnitude**: Typically 0.3-0.5 eV for small molecules (H2O, CO). Can significantly change relative reaction energies.
2. **Thermal correction (G - E)**: For gas-phase molecules at 298 K, typically -0.5 to -1.0 eV due to large translational and rotational entropy.
3. **Adsorbate corrections**: Smaller than gas-phase (no translation/rotation). Typically ZPE of 0.05-0.5 eV depending on the adsorbate.
4. **Entropy contribution**: T*S at 298 K is ~0.6 eV for typical gas molecules. This makes desorption favorable at high T.
5. **Imaginary frequencies**: For a minimum, all frequencies should be real. Imaginary frequencies indicate the structure is not at a minimum; re-relax with tighter convergence.
6. **Free energy diagrams**: The overpotential (eta) is the minimum potential at which all steps become downhill. Lower eta = better catalyst.

## Common Issues

| Issue | Solution |
|---|---|
| Imaginary frequencies for a supposed minimum | Structure not fully relaxed. Re-relax with fmax < 0.001 eV/A. |
| Too many near-zero frequencies | Increase vacuum box size for molecules. For adsorbates, ensure slab is properly frozen. |
| Wrong number of vibrational modes | Linear molecule: 3N-5 modes. Nonlinear: 3N-6 modes. Check geometry assignment. |
| Large discrepancy MACE vs QE frequencies | Expected for systems outside MACE training data. Use QE for quantitative results. |
| IdealGasThermo gives wrong entropy | Check symmetry number and spin. Wrong values cause large entropy errors. |
| HarmonicThermo negative entropy at low T | Numerical issue with very low frequencies. Increase frequency threshold. |
| ZPE too large / unreasonable | Check for unphysical high frequencies (> 4000 cm^-1 for non-H bonds). May indicate bad relaxation. |
