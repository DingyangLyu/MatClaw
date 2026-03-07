# Vacuum Resize for 2D / Slab Models

## When to Use

- Adjust the vacuum thickness in a slab or 2D material calculation
- Converge total energy with respect to vacuum size
- Prepare 2D structures with appropriate vacuum for DFT calculations
- Eliminate spurious interactions between periodic slab images
- Test convergence of work function, band gap, or surface energy vs vacuum thickness
- Corresponds to VASPKIT task 922

## Method Selection

| Task | Method | Notes |
|---|---|---|
| Set vacuum to specific thickness | pymatgen (Method A) | Rescale c-axis, reposition atoms |
| Convergence test (vacuum vs energy) | ASE + MACE (Method B) | Fast screening of optimal vacuum |
| Convergence test (vacuum vs DFT) | QE (Method C) | Publication-quality convergence |
| Symmetric vacuum (centered slab) | pymatgen (Method A) | Equal vacuum on both sides |

```
Need to set a specific vacuum thickness?
  --> Method A: resize_vacuum() function

Need to test convergence of a property vs vacuum?
  --> Method B (MACE, fast) then Method C (QE, accurate)

Need symmetric vacuum for work function calculation?
  --> Method A with center_slab=True
```

## Prerequisites

- pymatgen (structure manipulation)
- ASE + MACE (Method B)
- Quantum ESPRESSO (Method C)
- numpy, matplotlib

## Detailed Steps

### Method A: Set Vacuum Thickness

```python
#!/usr/bin/env python3
"""
Resize vacuum thickness for 2D/slab structures.

Provides tools to:
  - Set vacuum to a specific thickness
  - Add or remove vacuum
  - Create symmetric vacuum (centered slab)
  - Batch process multiple structures

Corresponds to VASPKIT task 922.
"""

import numpy as np
from pymatgen.core import Structure, Lattice
from pymatgen.io.cif import CifWriter
import json


def get_slab_thickness(structure, axis=2):
    """
    Measure the thickness of the slab (atomic extent) along the given axis.

    Parameters
    ----------
    structure : Structure
        Slab or 2D material structure.
    axis : int
        Vacuum direction (default: 2 = c).

    Returns
    -------
    thickness : float
        Slab thickness in Angstrom.
    vacuum : float
        Current vacuum thickness in Angstrom.
    """
    cart_z = np.array([s.coords[axis] for s in structure])
    z_min = np.min(cart_z)
    z_max = np.max(cart_z)
    thickness = z_max - z_min

    # Total cell length along axis
    c_length = np.linalg.norm(structure.lattice.matrix[axis])
    vacuum = c_length - thickness

    return thickness, vacuum


def resize_vacuum(structure, target_vacuum, axis=2, center=True):
    """
    Resize the vacuum region of a slab/2D structure.

    Parameters
    ----------
    structure : Structure
        Input structure with slab and vacuum.
    target_vacuum : float
        Desired vacuum thickness in Angstrom.
    axis : int
        Vacuum direction (default: 2 = c).
    center : bool
        If True, center the slab in the new cell.

    Returns
    -------
    Structure
        New structure with adjusted vacuum.
    """
    # Get current slab thickness
    thickness, current_vacuum = get_slab_thickness(structure, axis)

    # New c-axis length
    new_c_length = thickness + target_vacuum

    # Get Cartesian coordinates
    cart_coords = np.array([s.coords for s in structure])
    species = [str(s.specie) for s in structure]

    # Shift slab to start at z = 0 (if centering, will adjust later)
    z_cart = cart_coords[:, axis]
    z_min = np.min(z_cart)
    cart_coords[:, axis] -= z_min

    if center:
        # Center slab in the new cell
        cart_coords[:, axis] += target_vacuum / 2

    # Build new lattice
    old_matrix = structure.lattice.matrix.copy()
    c_hat = old_matrix[axis] / np.linalg.norm(old_matrix[axis])
    new_matrix = old_matrix.copy()
    new_matrix[axis] = c_hat * new_c_length

    new_lattice = Lattice(new_matrix)

    # Create new structure
    new_struct = Structure(
        new_lattice,
        species,
        cart_coords,
        coords_are_cartesian=True,
    )

    return new_struct


def add_vacuum(structure, additional_vacuum, axis=2, side="both"):
    """
    Add vacuum to an existing structure.

    Parameters
    ----------
    structure : Structure
        Input structure.
    additional_vacuum : float
        Vacuum to add (Angstrom).
    axis : int
        Vacuum direction.
    side : str
        "both" (split equally), "top", or "bottom".

    Returns
    -------
    Structure
        Structure with added vacuum.
    """
    thickness, current_vacuum = get_slab_thickness(structure, axis)
    new_vacuum = current_vacuum + additional_vacuum
    return resize_vacuum(structure, new_vacuum, axis, center=(side == "both"))


# ============================================================
# Example usage
# ============================================================
# Build a 2D MoS2 monolayer
mos2 = Structure(
    Lattice.hexagonal(3.16, 20.0),  # initial 20 A cell height
    ["Mo", "S", "S"],
    [[1/3, 2/3, 0.50], [1/3, 2/3, 0.562], [1/3, 2/3, 0.438]],
)

thickness, vacuum = get_slab_thickness(mos2)
print(f"Original MoS2:")
print(f"  Slab thickness: {thickness:.3f} A")
print(f"  Vacuum: {vacuum:.3f} A")
print(f"  c parameter: {mos2.lattice.c:.3f} A")

# Resize to various vacuum thicknesses
for target_vac in [10.0, 15.0, 20.0, 25.0, 30.0]:
    new_struct = resize_vacuum(mos2, target_vac, center=True)
    t, v = get_slab_thickness(new_struct)
    print(f"\n  Vacuum = {target_vac:.1f} A:")
    print(f"    c parameter: {new_struct.lattice.c:.3f} A")
    print(f"    Actual vacuum: {v:.3f} A")
    print(f"    Slab thickness: {t:.3f} A")

    # Save the 20 A vacuum version
    if abs(target_vac - 20.0) < 0.1:
        new_struct.to(filename="MoS2_vac20.cif")
        new_struct.to(filename="MoS2_vac20.vasp", fmt="poscar")
        print("    Saved: MoS2_vac20.cif, MoS2_vac20.vasp")
```

### Method B: Vacuum Convergence Test with MACE

```python
#!/usr/bin/env python3
"""
Test convergence of total energy with respect to vacuum thickness
using MACE for fast screening.

This helps determine the minimum vacuum needed to avoid
spurious interactions between periodic slab images.
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pymatgen.core import Structure, Lattice
from pymatgen.io.ase import AseAtomsAdaptor
from ase.optimize import LBFGS
from ase.constraints import FixAtoms
from mace.calculators import mace_mp
import json

# ============================================================
# 1. Build base structure
# ============================================================
# MoS2 monolayer
mos2 = Structure(
    Lattice.hexagonal(3.16, 30.0),  # start with large vacuum
    ["Mo", "S", "S"],
    [[1/3, 2/3, 0.50], [1/3, 2/3, 0.552], [1/3, 2/3, 0.448]],
)

# Make a 3x3 supercell for better statistics
mos2.make_supercell([3, 3, 1])

calc = mace_mp(model="medium", dispersion=True, default_dtype="float64")
adaptor = AseAtomsAdaptor()


def get_slab_thickness(structure, axis=2):
    cart_z = np.array([s.coords[axis] for s in structure])
    return np.max(cart_z) - np.min(cart_z)


def resize_vacuum(structure, target_vacuum, axis=2):
    cart_coords = np.array([s.coords for s in structure])
    species = [str(s.specie) for s in structure]
    z_cart = cart_coords[:, axis]
    thickness = np.max(z_cart) - np.min(z_cart)
    cart_coords[:, axis] -= np.min(z_cart)
    cart_coords[:, axis] += target_vacuum / 2
    new_c = thickness + target_vacuum
    matrix = structure.lattice.matrix.copy()
    c_hat = matrix[axis] / np.linalg.norm(matrix[axis])
    matrix[axis] = c_hat * new_c
    return Structure(Lattice(matrix), species, cart_coords,
                     coords_are_cartesian=True)


# ============================================================
# 2. Convergence test
# ============================================================
vacuum_values = [8, 10, 12, 14, 16, 18, 20, 22, 25, 30]
energies = []
energies_per_atom = []

print(f"{'Vacuum (A)':<12} {'c (A)':<10} {'E_total (eV)':<14} {'E/atom (eV)':<12}")
print("-" * 50)

for vac in vacuum_values:
    struct = resize_vacuum(mos2, vac)
    atoms = adaptor.get_atoms(struct)
    atoms.calc = mace_mp(model="medium", dispersion=True, default_dtype="float64")

    e_total = atoms.get_potential_energy()
    e_per_atom = e_total / len(atoms)

    energies.append(e_total)
    energies_per_atom.append(e_per_atom)

    print(f"{vac:<12.1f} {struct.lattice.c:<10.3f} {e_total:<14.6f} {e_per_atom:<12.6f}")

# ============================================================
# 3. Determine convergence
# ============================================================
# Reference: largest vacuum value
e_ref = energies_per_atom[-1]
errors = [abs(e - e_ref) * 1000 for e in energies_per_atom]  # in meV/atom

print(f"\nConvergence analysis (reference: vacuum = {vacuum_values[-1]} A):")
print(f"{'Vacuum (A)':<12} {'Error (meV/atom)':<18} {'Converged?'}")
print("-" * 40)

converged_vacuum = vacuum_values[-1]
for vac, err in zip(vacuum_values, errors):
    converged = "YES" if err < 1.0 else "no"  # 1 meV/atom threshold
    print(f"{vac:<12.1f} {err:<18.3f} {converged}")
    if err < 1.0 and vac < converged_vacuum:
        converged_vacuum = vac

print(f"\nMinimum converged vacuum (< 1 meV/atom): {converged_vacuum} A")

# ============================================================
# 4. Plot convergence
# ============================================================
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# Left: Energy per atom vs vacuum
ax1 = axes[0]
ax1.plot(vacuum_values, energies_per_atom, "o-", color="steelblue", markersize=8)
ax1.set_xlabel("Vacuum thickness (A)", fontsize=12)
ax1.set_ylabel("Energy per atom (eV)", fontsize=12)
ax1.set_title("Energy vs Vacuum Thickness", fontsize=13)
ax1.grid(alpha=0.3)

# Right: Error vs vacuum (convergence)
ax2 = axes[1]
ax2.semilogy(vacuum_values, errors, "o-", color="crimson", markersize=8)
ax2.axhline(1.0, color="green", linestyle="--", linewidth=1.5,
            label="1 meV/atom threshold")
ax2.axhline(0.1, color="blue", linestyle="--", linewidth=1,
            label="0.1 meV/atom threshold")
ax2.set_xlabel("Vacuum thickness (A)", fontsize=12)
ax2.set_ylabel("Error (meV/atom)", fontsize=12)
ax2.set_title("Vacuum Convergence", fontsize=13)
ax2.legend(fontsize=10)
ax2.grid(True, alpha=0.3)

fig.tight_layout()
fig.savefig("vacuum_convergence.png", dpi=150)
print("\nSaved vacuum_convergence.png")

# Save results
results = {
    "system": "MoS2 monolayer (3x3 supercell)",
    "vacuum_values_A": vacuum_values,
    "energies_per_atom_eV": energies_per_atom,
    "errors_meV_per_atom": errors,
    "converged_vacuum_A": converged_vacuum,
    "convergence_threshold_meV": 1.0,
}

with open("vacuum_convergence.json", "w") as f:
    json.dump(results, f, indent=2, default=float)
print("Saved vacuum_convergence.json")
```

### Method C: QE Vacuum Convergence Test

```python
#!/usr/bin/env python3
"""
Generate QE input files for vacuum convergence testing.

Creates a series of SCF calculations with different vacuum thicknesses
to verify that the chosen vacuum is sufficient.
"""

import os
import numpy as np
from pymatgen.core import Structure, Lattice
import subprocess
import re
import json

# ============================================================
# 1. Configuration
# ============================================================
PSEUDO_DIR = os.path.abspath("./pseudo")
WORK_DIR = os.path.abspath("./vac_convergence")
os.makedirs(WORK_DIR, exist_ok=True)

ECUTWFC = 50.0
ECUTRHO = 400.0
PREFIX = "mos2"

# Base monolayer structure
base_struct = Structure(
    Lattice.hexagonal(3.16, 25.0),
    ["Mo", "S", "S"],
    [[1/3, 2/3, 0.50], [1/3, 2/3, 0.562], [1/3, 2/3, 0.438]],
)

pseudos = {
    "Mo": "Mo.pbe-spn-kjpaw_psl.1.0.0.UPF",
    "S": "S.pbe-n-kjpaw_psl.1.0.0.UPF",
}

vacuum_values = [10, 12, 15, 18, 20, 25, 30]


# ============================================================
# 2. Generate input files for each vacuum
# ============================================================
def resize_vacuum_struct(structure, target_vacuum, axis=2):
    """Resize vacuum to target thickness, center slab."""
    cart_coords = np.array([s.coords for s in structure])
    species = [str(s.specie) for s in structure]
    z = cart_coords[:, axis]
    thickness = np.max(z) - np.min(z)
    cart_coords[:, axis] -= np.min(z)
    cart_coords[:, axis] += target_vacuum / 2
    new_c = thickness + target_vacuum
    matrix = structure.lattice.matrix.copy()
    c_hat = matrix[axis] / np.linalg.norm(matrix[axis])
    matrix[axis] = c_hat * new_c
    return Structure(Lattice(matrix), species, cart_coords,
                     coords_are_cartesian=True)


for vac in vacuum_values:
    struct = resize_vacuum_struct(base_struct, vac)
    n_atoms = len(struct)
    elements = sorted(set(str(s.specie) for s in struct))

    inp = f"""&CONTROL
    calculation = 'scf'
    prefix      = '{PREFIX}_v{vac}'
    outdir      = '{WORK_DIR}/tmp'
    pseudo_dir  = '{PSEUDO_DIR}'
    tprnfor     = .true.
/
&SYSTEM
    ibrav       = 0
    nat         = {n_atoms}
    ntyp        = {len(elements)}
    ecutwfc     = {ECUTWFC}
    ecutrho     = {ECUTRHO}
    occupations = 'smearing'
    smearing    = 'cold'
    degauss     = 0.01
    assume_isolated = '2D'
/
&ELECTRONS
    conv_thr = 1.0d-8
/

ATOMIC_SPECIES
"""
    from pymatgen.core.periodic_table import Element
    for el in elements:
        mass = Element(el).atomic_mass
        inp += f"  {el}  {mass:.4f}  {pseudos[el]}\n"

    inp += "\nCELL_PARAMETERS angstrom\n"
    for vec in struct.lattice.matrix:
        inp += f"  {vec[0]:.10f}  {vec[1]:.10f}  {vec[2]:.10f}\n"

    inp += "\nATOMIC_POSITIONS crystal\n"
    for site in struct:
        el = str(site.specie)
        fc = site.frac_coords
        inp += f"  {el}  {fc[0]:.10f}  {fc[1]:.10f}  {fc[2]:.10f}\n"

    inp += "\nK_POINTS automatic\n  12 12 1  0 0 0\n"

    inp_file = os.path.join(WORK_DIR, f"scf_v{vac}.in")
    with open(inp_file, "w") as f:
        f.write(inp)

print(f"Generated {len(vacuum_values)} QE input files in {WORK_DIR}/")
print("Note: assume_isolated = '2D' adds 2D Coulomb cutoff (recommended)")
print("\nRun with:")
for vac in vacuum_values:
    print(f"  mpirun -np $(nproc) pw.x -in {WORK_DIR}/scf_v{vac}.in > "
          f"{WORK_DIR}/scf_v{vac}.out")

# ============================================================
# 3. Parse results (run after QE calculations complete)
# ============================================================
print("\nAfter running, parse results with:")
print("""
import re, json
results = []
for vac in vacuum_values:
    outfile = f"{WORK_DIR}/scf_v{vac}.out"
    energy = None
    with open(outfile) as f:
        for line in f:
            if "!" in line and "total energy" in line:
                m = re.search(r"=\s+([-\\d.]+)\\s+Ry", line)
                if m: energy = float(m.group(1)) * 13.605693123
    results.append({"vacuum_A": vac, "energy_eV": energy})
    print(f"Vacuum {vac} A: E = {energy:.6f} eV")
""")
```

## Key Parameters

| Parameter | Recommended Value | Notes |
|---|---|---|
| Minimum vacuum | 15 A | Absolute minimum for most calculations |
| Recommended vacuum | 20 A | Safe default for most 2D/slab systems |
| Work function convergence | 20-25 A | Work function is sensitive to vacuum; needs more than total energy |
| Convergence threshold | < 1 meV/atom | For total energy; use < 0.01 eV for work function |
| `assume_isolated = '2D'` | Recommended for QE | Adds 2D Coulomb cutoff, reduces vacuum needed |
| Dipole correction | Use for asymmetric slabs | `dipfield=.true., edir=3` in QE |
| K-points perpendicular | 1 | Only 1 k-point along vacuum direction (no periodicity) |

## Interpreting Results

1. **Energy convergence**: Total energy per atom should change by < 1 meV/atom when increasing vacuum by 5 A. If not, use more vacuum.
2. **Electrostatic potential**: For work function calculations, the electrostatic potential must be flat (constant) in the vacuum region. If it is not flat, the vacuum is too thin.
3. **2D Coulomb cutoff**: QE's `assume_isolated = '2D'` truncates the Coulomb interaction at the cell boundary, reducing the required vacuum thickness significantly (often 10-12 A is sufficient with this option).
4. **Dipole correction**: For asymmetric slabs (e.g., one side has an adsorbate), a dipole correction is essential to remove the artificial electric field across the vacuum.
5. **Surface density**: Denser surfaces (more atoms per area) may need slightly more vacuum due to longer-range electrostatic interactions.

## Common Issues

| Issue | Solution |
|---|---|
| Energy does not converge with vacuum | Check for charged surfaces or dipoles. Enable dipole correction. Use `assume_isolated = '2D'`. |
| Slab atoms are split across periodic boundary | Center the slab first using the `center_slab()` function. |
| Too many k-points along c | For slab/2D systems, use only 1 k-point along the vacuum direction (e.g., 12 12 1). |
| Work function not converged at 20 A | Work function is more sensitive than total energy. Try 25-30 A or use 2D Coulomb cutoff. |
| MACE gives different convergence than QE | Expected. MACE uses periodic interactions differently. QE is the reference. |
| `assume_isolated` crashes | Only available in QE >= 6.5. Check your QE version. |
