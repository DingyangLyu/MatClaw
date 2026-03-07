# d-Band Center Calculation

## When to Use

- Compute d-band center from projected density of states (PDOS) as a descriptor for catalytic activity
- Apply the Norskov d-band model to predict adsorption energies on transition metal surfaces
- Compare d-band centers across different metals, alloys, or surface facets
- Analyze the effect of strain, alloying, or surface modification on the d-band position
- Corresponds to VASPKIT task 503

## Method Selection

| Criterion | QE DFT (Method A) | VASP + VASPKIT (Method B) | MACE (N/A) |
|---|---|---|---|
| d-band center | From projwfc.x PDOS | From DOSCAR/PROCAR via VASPKIT 503 | Not available |
| Accuracy | Full DFT | Full DFT | Force field only |
| Notes | Parse PDOS files, integrate d-states | VASPKIT reads DOSCAR automatically | MACE has no electronic states |

```
Need d-band center from DFT calculation?
  Have QE data?
    YES --> Method A: Parse projwfc.x PDOS output
  Have VASP data?
    YES --> Method B: Use VASPKIT 503 or parse DOSCAR/PROCAR

Need to pre-relax surface before DFT?
  --> Use MACE for structure relaxation, then run QE/VASP for PDOS
```

**Important**: MACE cannot compute electronic properties. The d-band center requires DFT eigenvalues. Use MACE only for structure preparation.

## Prerequisites

- Quantum ESPRESSO (pw.x, projwfc.x) -- for Method A
- A completed NSCF + projwfc.x calculation (see `electronic-structure/density-of-states/`)
- numpy, scipy (numerical integration)
- matplotlib (plotting)
- pymatgen (optional, for structure manipulation)

## Detailed Steps

### Method A: d-Band Center from QE PDOS

#### Step 1: Run SCF + NSCF + projwfc.x (Complete Workflow)

```python
#!/usr/bin/env python3
"""
Complete workflow to compute d-band center for a transition metal surface.
Steps:
  1. Relax slab with MACE (fast)
  2. Run QE SCF
  3. Run QE NSCF (dense k-grid)
  4. Run projwfc.x to get PDOS
  5. Parse PDOS and compute d-band center

Example: Pt(111) surface slab.
"""

import os
import subprocess
import re
import glob
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from ase.build import fcc111
from ase.io import write
from ase.optimize import BFGS
from ase.constraints import FixAtoms
from mace.calculators import mace_mp
import json

# ============================================================
# 1. Build and relax slab with MACE
# ============================================================
calc = mace_mp(model="medium", dispersion=False, default_dtype="float64")

slab = fcc111("Pt", size=(2, 2, 4), vacuum=15.0)
n_atoms = len(slab)

# Fix bottom 2 layers
z = slab.get_positions()[:, 2]
z_layers = sorted(set(np.round(z, 1)))
fix_z = z_layers[1] + 0.5
slab.set_constraint(FixAtoms(mask=z < fix_z))
slab.calc = calc

opt = BFGS(slab, logfile="relax_Pt111.log")
opt.run(fmax=0.01, steps=200)
write("Pt111_relaxed.cif", slab)
print(f"Relaxed Pt(111) slab: {n_atoms} atoms")

# ============================================================
# 2. Generate QE input files
# ============================================================
PSEUDO_DIR = os.path.abspath("./pseudo")
OUTDIR = os.path.abspath("./tmp_dband")
os.makedirs(PSEUDO_DIR, exist_ok=True)
os.makedirs(OUTDIR, exist_ok=True)
PREFIX = "pt111"

# Get cell and positions from relaxed slab
cell = slab.get_cell()
positions = slab.get_positions()
symbols = slab.get_chemical_symbols()

pseudo_file = "Pt.pbe-n-rrkjus_psl.1.0.0.UPF"

# Download pseudopotential
pp_path = os.path.join(PSEUDO_DIR, pseudo_file)
if not os.path.exists(pp_path):
    subprocess.run([
        "wget", "-q",
        f"https://pseudopotentials.quantum-espresso.org/upf_files/{pseudo_file}",
        "-O", pp_path
    ], check=True)
    print(f"Downloaded {pseudo_file}")

# --- SCF input ---
scf_input = f"""&CONTROL
    calculation = 'scf'
    prefix      = '{PREFIX}'
    outdir      = '{OUTDIR}'
    pseudo_dir  = '{PSEUDO_DIR}'
/
&SYSTEM
    ibrav       = 0
    nat         = {n_atoms}
    ntyp        = 1
    ecutwfc     = 50.0
    ecutrho     = 400.0
    occupations = 'smearing'
    smearing    = 'cold'
    degauss     = 0.02
/
&ELECTRONS
    conv_thr = 1.0d-8
    mixing_beta = 0.3
/

ATOMIC_SPECIES
  Pt  195.078  {pseudo_file}

CELL_PARAMETERS angstrom
"""
for vec in cell:
    scf_input += f"  {vec[0]:.10f}  {vec[1]:.10f}  {vec[2]:.10f}\n"

scf_input += "\nATOMIC_POSITIONS angstrom\n"
for sym, pos in zip(symbols, positions):
    scf_input += f"  {sym}  {pos[0]:.10f}  {pos[1]:.10f}  {pos[2]:.10f}\n"

scf_input += "\nK_POINTS automatic\n  6 6 1  0 0 0\n"

with open(f"{PREFIX}_scf.in", "w") as f:
    f.write(scf_input)

# --- NSCF input (dense k-grid) ---
nscf_input = scf_input.replace("'scf'", "'nscf'")
nscf_input = nscf_input.replace("6 6 1  0 0 0", "12 12 1  0 0 0")
# Add nbnd and change occupations
nscf_input = nscf_input.replace(
    "occupations = 'smearing'",
    "occupations = 'smearing'\n    nbnd        = 100"
)
nscf_input = nscf_input.replace(
    "conv_thr = 1.0d-8",
    "conv_thr = 1.0d-8\n    diago_full_acc = .true."
)

with open(f"{PREFIX}_nscf.in", "w") as f:
    f.write(nscf_input)

# --- projwfc.x input ---
projwfc_input = f"""&PROJWFC
    prefix  = '{PREFIX}'
    outdir  = '{OUTDIR}'
    filpdos = '{PREFIX}_pdos'
    Emin    = -15.0
    Emax    = 5.0
    DeltaE  = 0.01
    ngauss  = 0
    degauss = 0.01
/
"""

with open(f"{PREFIX}_projwfc.in", "w") as f:
    f.write(projwfc_input)

print("Generated QE input files:")
print(f"  {PREFIX}_scf.in")
print(f"  {PREFIX}_nscf.in")
print(f"  {PREFIX}_projwfc.in")
```

#### Step 2: Run QE Calculations

```bash
#!/bin/bash
# Run QE SCF -> NSCF -> projwfc.x for d-band center calculation

PREFIX="pt111"
NPROC=$(nproc)

echo "=== Step 1: SCF ==="
mpirun --allow-run-as-root -np $NPROC pw.x -in ${PREFIX}_scf.in > ${PREFIX}_scf.out 2>&1
grep "convergence has been achieved" ${PREFIX}_scf.out && echo "SCF converged" || echo "SCF FAILED"

echo "=== Step 2: NSCF ==="
mpirun --allow-run-as-root -np $NPROC pw.x -in ${PREFIX}_nscf.in > ${PREFIX}_nscf.out 2>&1
echo "NSCF done (exit code: $?)"

echo "=== Step 3: projwfc.x ==="
projwfc.x -in ${PREFIX}_projwfc.in > ${PREFIX}_projwfc.out 2>&1
echo "projwfc.x done (exit code: $?)"

echo "=== PDOS files generated ==="
ls -la ${PREFIX}_pdos.pdos_atm* 2>/dev/null | head -20
```

#### Step 3: Compute d-Band Center from PDOS

```python
#!/usr/bin/env python3
"""
Compute d-band center from QE projwfc.x PDOS output.

The d-band center is defined as:
  epsilon_d = integral(E * PDOS_d(E) dE) / integral(PDOS_d(E) dE)

where PDOS_d(E) is the projected DOS onto d-orbitals of the surface atoms.

This is a key descriptor in the Norskov d-band model for predicting
adsorption energies on transition metal surfaces.
"""

import os
import re
import glob
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import json

PREFIX = "pt111"

# ============================================================
# 1. Extract Fermi energy from NSCF output
# ============================================================
def get_fermi_energy(output_file):
    """Extract Fermi energy (eV) from QE output."""
    e_fermi = 0.0
    with open(output_file) as f:
        for line in f:
            if "the Fermi energy is" in line:
                m = re.search(r"is\s+([-\d.]+)", line)
                if m:
                    e_fermi = float(m.group(1))
            if "highest occupied, lowest unoccupied" in line:
                m = re.search(r":\s+([-\d.]+)\s+([-\d.]+)", line)
                if m:
                    e_fermi = (float(m.group(1)) + float(m.group(2))) / 2
    return e_fermi

e_fermi = get_fermi_energy(f"{PREFIX}_nscf.out")
print(f"Fermi energy: {e_fermi:.4f} eV")

# ============================================================
# 2. Parse PDOS files and identify d-orbital contributions
# ============================================================
def parse_pdos_file(filename):
    """Parse a projwfc.x PDOS file. Returns (energy, pdos)."""
    data = np.loadtxt(filename, comments="#")
    return data[:, 0], data[:, 1]  # energy, LDOS

# Find all PDOS files
pdos_files = sorted(glob.glob(f"{PREFIX}_pdos.pdos_atm#*"))
print(f"Found {len(pdos_files)} PDOS files")

# Parse and categorize by atom and orbital
pdos_data = {}  # {(atom_idx, element, orbital): (energy, pdos)}
for fpath in pdos_files:
    fname = os.path.basename(fpath)
    m = re.search(r"atm#(\d+)\((\w+)\)_wfc#(\d+)\((\w+)\)", fname)
    if m:
        atom_idx = int(m.group(1))
        element = m.group(2)
        orbital = m.group(4)  # s, p, d, f
        energy, pdos = parse_pdos_file(fpath)
        pdos_data[(atom_idx, element, orbital)] = (energy, pdos)

# ============================================================
# 3. Identify surface atoms and compute d-band center
# ============================================================
# Surface atoms: those with the highest z-coordinates
# For a 4-layer slab, surface = top layer

# Get atom indices that have d-orbital contributions
d_atoms = set()
for (atom_idx, element, orbital) in pdos_data:
    if orbital == "d":
        d_atoms.add((atom_idx, element))

print(f"\nAtoms with d-orbitals: {len(d_atoms)}")

# For surface analysis, we typically want only the top-layer atoms
# If we know the atom ordering, we can select specific indices
# Here we sum all d-orbital PDOS (or specify surface_atoms list)

# Option 1: Sum d-PDOS over ALL atoms (bulk d-band center)
d_pdos_total = None
energy_grid = None

for (atom_idx, element, orbital), (energy, pdos) in pdos_data.items():
    if orbital == "d":
        if d_pdos_total is None:
            energy_grid = energy
            d_pdos_total = np.zeros_like(pdos)
        d_pdos_total += pdos

if d_pdos_total is None:
    print("ERROR: No d-orbital PDOS found!")
    exit(1)

# Option 2: Sum d-PDOS over specified surface atoms
# surface_atom_indices = [13, 14, 15, 16]  # example: top-layer Pt atoms
# d_pdos_surface = np.zeros_like(energy_grid)
# for (atom_idx, element, orbital), (energy, pdos) in pdos_data.items():
#     if orbital == "d" and atom_idx in surface_atom_indices:
#         d_pdos_surface += pdos

# ============================================================
# 4. Compute d-band center and higher moments
# ============================================================
def compute_band_center(energy, pdos, e_fermi=None, integrate_up_to=None):
    """
    Compute the band center (first moment of the DOS).

    Parameters
    ----------
    energy : array
        Energy grid (eV).
    pdos : array
        Projected DOS values.
    e_fermi : float, optional
        Fermi energy. If given, shift energy scale so E_F = 0.
    integrate_up_to : float, optional
        Upper integration limit (eV). Default: integrate over all states.
        Use e_fermi to integrate only filled states.

    Returns
    -------
    center : float
        Band center (first moment) in eV.
    width : float
        Band width (second moment) in eV.
    skewness : float
        Band skewness (third moment).
    filling : float
        Band filling fraction.
    """
    e = energy.copy()
    if e_fermi is not None:
        e = e - e_fermi

    if integrate_up_to is not None:
        mask = energy <= integrate_up_to
    else:
        mask = np.ones(len(energy), dtype=bool)

    e_masked = e[mask]
    p_masked = pdos[mask]

    # Zeroth moment: total number of states
    n_states = np.trapz(p_masked, e_masked)

    if n_states < 1e-10:
        return 0.0, 0.0, 0.0, 0.0

    # First moment: band center
    center = np.trapz(e_masked * p_masked, e_masked) / n_states

    # Second moment: band width (variance)
    variance = np.trapz((e_masked - center)**2 * p_masked, e_masked) / n_states
    width = np.sqrt(variance)

    # Third moment: skewness
    if width > 0:
        skew = np.trapz((e_masked - center)**3 * p_masked, e_masked) / (n_states * width**3)
    else:
        skew = 0.0

    # Band filling
    total_states = np.trapz(pdos, energy)
    filling = n_states / total_states if total_states > 0 else 0.0

    return center, width, skew, filling


# Compute d-band center (all states, referenced to E_F)
d_center, d_width, d_skew, d_filling = compute_band_center(
    energy_grid, d_pdos_total, e_fermi=e_fermi
)

# Compute d-band center (filled states only)
d_center_filled, d_width_filled, _, d_filling_filled = compute_band_center(
    energy_grid, d_pdos_total, e_fermi=e_fermi, integrate_up_to=e_fermi
)

print(f"\n{'='*50}")
print(f"d-BAND CENTER ANALYSIS")
print(f"{'='*50}")
print(f"d-band center (all states):    {d_center:.4f} eV  (relative to E_F)")
print(f"d-band center (filled only):   {d_center_filled:.4f} eV")
print(f"d-band width:                  {d_width:.4f} eV")
print(f"d-band skewness:               {d_skew:.4f}")
print(f"d-band filling:                {d_filling:.2%}")

# ============================================================
# 5. Per-atom d-band centers (for identifying surface vs bulk)
# ============================================================
print(f"\nPer-atom d-band centers:")
print(f"{'Atom':>6} {'Element':>8} {'d-center (eV)':>14} {'d-width (eV)':>13}")
print("-" * 45)

per_atom_results = []
for (atom_idx, element) in sorted(d_atoms):
    if (atom_idx, element, "d") in pdos_data:
        e, p = pdos_data[(atom_idx, element, "d")]
        center, width, _, _ = compute_band_center(e, p, e_fermi=e_fermi)
        print(f"  {atom_idx:4d}   {element:>6}   {center:12.4f}   {width:11.4f}")
        per_atom_results.append({
            "atom_index": atom_idx,
            "element": element,
            "d_center_eV": float(center),
            "d_width_eV": float(width),
        })

# ============================================================
# 6. Plot d-band DOS and center
# ============================================================
fig, ax = plt.subplots(figsize=(8, 5))

e_shifted = energy_grid - e_fermi

# Plot d-PDOS
ax.plot(e_shifted, d_pdos_total, color="steelblue", linewidth=1.5, label="d-PDOS (total)")
ax.fill_between(e_shifted, d_pdos_total, alpha=0.15, color="steelblue")

# Shade filled region
fill_mask = energy_grid <= e_fermi
ax.fill_between(e_shifted[fill_mask], d_pdos_total[fill_mask], alpha=0.3,
                color="steelblue", label="Filled d-states")

# Mark d-band center
ax.axvline(d_center, color="red", linestyle="--", linewidth=2,
           label=f"d-band center: {d_center:.2f} eV")

# Mark Fermi level
ax.axvline(0, color="black", linestyle="-", linewidth=1, label="$E_F$")

ax.set_xlabel("$E - E_F$ (eV)", fontsize=13)
ax.set_ylabel("d-PDOS (states/eV)", fontsize=13)
ax.set_title("d-Band Analysis", fontsize=14)
ax.set_xlim(-12, 5)
ax.set_ylim(0, None)
ax.legend(fontsize=10)
ax.grid(alpha=0.3)

fig.tight_layout()
fig.savefig("d_band_center.png", dpi=150)
print("\nSaved d_band_center.png")

# ============================================================
# 7. Save results
# ============================================================
results = {
    "system": "Pt(111)",
    "e_fermi_eV": float(e_fermi),
    "d_band_center_all_eV": float(d_center),
    "d_band_center_filled_eV": float(d_center_filled),
    "d_band_width_eV": float(d_width),
    "d_band_skewness": float(d_skew),
    "d_band_filling": float(d_filling),
    "per_atom": per_atom_results,
}

with open("d_band_results.json", "w") as f:
    json.dump(results, f, indent=2)
print("Saved d_band_results.json")
```

#### Comparative d-Band Center Analysis (Multiple Metals)

```python
#!/usr/bin/env python3
"""
Compare d-band centers across different transition metals.
Uses pre-computed PDOS data from QE projwfc.x runs.

This script reads PDOS files from multiple calculation directories
and produces a comparative bar chart and correlation plot.
"""

import os
import re
import glob
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import json


def compute_d_center_from_pdos_dir(calc_dir, prefix, nscf_out=None):
    """
    Compute d-band center from a QE calculation directory.

    Parameters
    ----------
    calc_dir : str
        Directory containing PDOS files.
    prefix : str
        PDOS file prefix (e.g., "pt111_pdos").
    nscf_out : str, optional
        Path to NSCF output for Fermi energy.

    Returns
    -------
    dict with d-band center, width, and Fermi energy.
    """
    # Get Fermi energy
    e_fermi = 0.0
    if nscf_out and os.path.exists(nscf_out):
        with open(nscf_out) as f:
            for line in f:
                if "the Fermi energy is" in line:
                    m = re.search(r"is\s+([-\d.]+)", line)
                    if m:
                        e_fermi = float(m.group(1))

    # Find and sum d-orbital PDOS
    pdos_files = sorted(glob.glob(os.path.join(calc_dir, f"{prefix}.pdos_atm*")))
    d_pdos = None
    energy = None

    for fpath in pdos_files:
        fname = os.path.basename(fpath)
        if "(d)" in fname:
            data = np.loadtxt(fpath, comments="#")
            if d_pdos is None:
                energy = data[:, 0]
                d_pdos = np.zeros(len(energy))
            d_pdos += data[:, 1]

    if d_pdos is None:
        return None

    # Compute center
    e_shifted = energy - e_fermi
    n_states = np.trapz(d_pdos, e_shifted)
    if n_states < 1e-10:
        return None

    center = np.trapz(e_shifted * d_pdos, e_shifted) / n_states
    variance = np.trapz((e_shifted - center)**2 * d_pdos, e_shifted) / n_states
    width = np.sqrt(variance)

    return {
        "d_center_eV": float(center),
        "d_width_eV": float(width),
        "e_fermi_eV": float(e_fermi),
    }


# ============================================================
# Compare d-band centers
# ============================================================
# Example: Known d-band centers from literature (relative to E_F)
# These serve as reference values for validation
literature_d_centers = {
    "Cu": -2.67,
    "Ag": -4.30,
    "Au": -3.56,
    "Ni": -1.29,
    "Pd": -1.83,
    "Pt": -2.25,
    "Co": -1.17,
    "Fe": -0.92,
    "Ru": -1.41,
    "Rh": -1.73,
    "Ir": -2.11,
}

# Plot literature d-band centers vs position in periodic table
metals = ["Fe", "Co", "Ni", "Cu", "Ru", "Rh", "Pd", "Ag", "Ir", "Pt", "Au"]
d_centers = [literature_d_centers[m] for m in metals]

fig, ax = plt.subplots(figsize=(10, 5))
colors = plt.cm.coolwarm(np.linspace(0.2, 0.8, len(metals)))

bars = ax.bar(metals, d_centers, color=colors, edgecolor="black", alpha=0.8)

for bar, val in zip(bars, d_centers):
    ax.text(bar.get_x() + bar.get_width()/2, val - 0.15,
            f"{val:.2f}", ha="center", va="top", fontsize=9, fontweight="bold")

ax.set_ylabel("d-band center (eV, rel. to $E_F$)", fontsize=13)
ax.set_title("d-Band Centers of Transition Metals (111) Surfaces", fontsize=14)
ax.axhline(0, color="gray", linewidth=0.5)
ax.grid(axis="y", alpha=0.3)

fig.tight_layout()
fig.savefig("d_band_comparison.png", dpi=150)
print("Saved d_band_comparison.png")
```

### Method B: d-Band Center from VASP DOSCAR/PROCAR

```python
#!/usr/bin/env python3
"""
Compute d-band center from VASP DOSCAR or PROCAR output.
This replicates VASPKIT task 503 functionality.

Requires a completed VASP calculation with:
  LORBIT = 11   (atom- and orbital-projected DOS in DOSCAR)
  or LORBIT = 12 (m-resolved)
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import json


def parse_doscar(filename="DOSCAR", n_atoms=None):
    """
    Parse VASP DOSCAR file.

    Returns
    -------
    e_fermi : float
    energy : array
    total_dos : array
    pdos : dict mapping atom_index -> {orbital: dos_array}
         orbitals: s, p, d (or s, py, pz, px, dxy, dyz, dz2, dxz, dx2-y2)
    """
    with open(filename) as f:
        lines = f.readlines()

    # Line 6: NEDOS, E_fermi, ...
    header = lines[5].split()
    e_max = float(header[0])
    e_min = float(header[1])
    nedos = int(header[2])
    e_fermi = float(header[3])

    # Read total DOS (lines 6 to 6+NEDOS)
    total_data = []
    for i in range(6, 6 + nedos):
        parts = [float(x) for x in lines[i].split()]
        total_data.append(parts)
    total_data = np.array(total_data)
    energy = total_data[:, 0]
    total_dos = total_data[:, 1]

    # Read projected DOS for each atom
    pdos = {}
    offset = 6 + nedos

    if n_atoms is None:
        # Try to detect from file
        n_atoms_detected = int(lines[0].split()[0])
        n_atoms = n_atoms_detected

    for atom_i in range(n_atoms):
        if offset >= len(lines):
            break

        # Skip header line for this atom
        offset += 1

        atom_data = []
        for i in range(offset, min(offset + nedos, len(lines))):
            parts = [float(x) for x in lines[i].split()]
            atom_data.append(parts)
        offset += nedos

        if not atom_data:
            continue

        atom_data = np.array(atom_data)
        # Columns depend on LORBIT:
        # LORBIT=11, non-spin: E, s, p, d (4 cols) or E, s, py, pz, px, dxy, dyz, dz2, dxz, dx2 (10 cols)
        # LORBIT=11, spin: E, s_up, s_dn, p_up, p_dn, d_up, d_dn

        n_cols = atom_data.shape[1]

        if n_cols >= 4:
            # Simplified: s, p, d columns (LORBIT=10 or summed LORBIT=11)
            pdos[atom_i] = {
                "s": atom_data[:, 1],
                "p": atom_data[:, 2] if n_cols > 2 else np.zeros(nedos),
                "d": atom_data[:, 3] if n_cols > 3 else np.zeros(nedos),
            }

            # If more columns (m-resolved), sum into s, p, d
            if n_cols == 10:
                # E, s, py, pz, px, dxy, dyz, dz2, dxz, dx2-y2
                pdos[atom_i] = {
                    "s": atom_data[:, 1],
                    "p": atom_data[:, 2] + atom_data[:, 3] + atom_data[:, 4],
                    "d": (atom_data[:, 5] + atom_data[:, 6] + atom_data[:, 7] +
                          atom_data[:, 8] + atom_data[:, 9]),
                }

    return e_fermi, energy, total_dos, pdos


def compute_d_band_center(energy, d_pdos, e_fermi):
    """Compute d-band center relative to Fermi energy."""
    e = energy - e_fermi
    n = np.trapz(d_pdos, e)
    if n < 1e-10:
        return 0.0, 0.0
    center = np.trapz(e * d_pdos, e) / n
    var = np.trapz((e - center)**2 * d_pdos, e) / n
    return center, np.sqrt(var)


# ============================================================
# Parse DOSCAR and compute d-band center
# ============================================================
doscar_file = "DOSCAR"
e_fermi, energy, total_dos, pdos = parse_doscar(doscar_file)
print(f"Fermi energy: {e_fermi:.4f} eV")
print(f"Number of atoms with PDOS: {len(pdos)}")

# Sum d-PDOS over all atoms (or specify surface atoms)
d_pdos_total = np.zeros_like(energy)
for atom_i, orbitals in pdos.items():
    if "d" in orbitals:
        d_pdos_total += orbitals["d"]

center, width = compute_d_band_center(energy, d_pdos_total, e_fermi)
print(f"\nd-band center (all atoms): {center:.4f} eV (rel. to E_F)")
print(f"d-band width: {width:.4f} eV")

# Surface atoms only (specify indices)
# surface_atoms = [12, 13, 14, 15]
# d_pdos_surface = np.zeros_like(energy)
# for atom_i in surface_atoms:
#     if atom_i in pdos and "d" in pdos[atom_i]:
#         d_pdos_surface += pdos[atom_i]["d"]
# center_surf, width_surf = compute_d_band_center(energy, d_pdos_surface, e_fermi)
# print(f"d-band center (surface): {center_surf:.4f} eV")

# Plot
fig, ax = plt.subplots(figsize=(8, 5))
e_shifted = energy - e_fermi
ax.plot(e_shifted, d_pdos_total, color="steelblue", linewidth=1.5)
ax.fill_between(e_shifted, d_pdos_total, where=(energy <= e_fermi),
                alpha=0.3, color="steelblue", label="Filled d-states")
ax.axvline(center, color="red", linestyle="--", linewidth=2,
           label=f"d-center: {center:.2f} eV")
ax.axvline(0, color="black", linewidth=1)
ax.set_xlabel("$E - E_F$ (eV)", fontsize=13)
ax.set_ylabel("d-PDOS (states/eV)", fontsize=13)
ax.set_title("d-Band Analysis (VASP)", fontsize=14)
ax.set_xlim(-12, 5)
ax.legend(fontsize=11)
ax.grid(alpha=0.3)
fig.tight_layout()
fig.savefig("d_band_vasp.png", dpi=150)
print("Saved d_band_vasp.png")

results = {
    "e_fermi_eV": float(e_fermi),
    "d_band_center_eV": float(center),
    "d_band_width_eV": float(width),
}
with open("d_band_vasp_results.json", "w") as f:
    json.dump(results, f, indent=2)
print("Saved d_band_vasp_results.json")
```

## Key Parameters

| Parameter | Recommended Value | Notes |
|---|---|---|
| NSCF k-grid | 12x12x1 or denser | Dense in-plane grid needed for accurate d-band integration |
| DeltaE (projwfc.x) | 0.01 eV | Energy grid spacing for PDOS; finer = more accurate integration |
| degauss (projwfc.x) | 0.01-0.02 Ry | Gaussian broadening; too large smears out features |
| LORBIT (VASP) | 11 | Required for orbital-projected DOS in DOSCAR |
| ecutwfc | 50-80 Ry | Depends on pseudopotential; converge for your system |
| Slab layers | 4-6 | Converge d-band center vs number of layers |
| Surface atom selection | Top 1-2 layers | d-band center of surface atoms differs from bulk |

## Interpreting Results

1. **d-band center position**: Higher (closer to E_F) d-band center correlates with stronger adsorption. The d-band model predicts: as the d-band center moves up, the anti-bonding states shift above E_F and become empty, strengthening the adsorbate-surface bond.
2. **Ordering**: Cu < Ag < Au < Pt < Pd < Rh < Ir < Ru < Ni < Co < Fe (from lower to higher d-band center, i.e., from weaker to stronger adsorption).
3. **d-band width**: Narrower d-bands (late 3d metals) have more localized d-states and stronger correlation effects. Width affects the accuracy of the d-band model.
4. **Surface vs bulk**: Surface atoms have narrower d-bands shifted up relative to bulk, due to reduced coordination.
5. **Scaling relations**: Linear correlations between d-band center and adsorption energies of *OH, *O, *OOH enable catalyst screening without computing every intermediate.
6. **Strain effects**: Tensile strain narrows the d-band and shifts it up; compressive strain broadens and shifts it down.

## Common Issues

| Issue | Solution |
|---|---|
| No d-orbital PDOS files found | Check projwfc.x ran successfully. Ensure species has d-electrons. |
| d-band center seems wrong | Verify Fermi energy extraction. Check energy alignment. |
| Integration gives zero | Energy range too narrow; extend Emin/Emax in projwfc.x input. |
| Different result with different k-grid | Converge with respect to k-grid density; 12x12x1 minimum for surfaces. |
| DOSCAR has wrong number of columns | Check LORBIT setting in VASP. LORBIT=11 gives orbital-resolved; LORBIT=10 gives l-resolved. |
| Surface vs bulk d-band center differ | This is expected. Report which atoms were included in the analysis. |
| d-band center does not correlate with adsorption | The d-band model is approximate. Works best for simple adsorbates (CO, O, H) on close-packed surfaces. |
