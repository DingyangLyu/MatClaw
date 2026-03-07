# Grain Boundary Construction and Calculation

## When to Use

- You need to compute grain boundary (GB) energies for a material.
- You want to study the relationship between GB geometry (sigma value, misorientation angle) and energy.
- You need to build symmetric tilt or twist grain boundary structures for further simulation (segregation, diffusion, mechanical response).

## Prerequisites

- Python packages: `pymatgen`, `ase`, `mace-torch`, `numpy`, `matplotlib`, `spglib`.
- For QE validation: Quantum ESPRESSO 7.5 (`pw.x`) with appropriate pseudopotentials.

## Detailed Steps

### Background: Grain Boundary Theory

A **grain boundary** is the interface between two crystallites of the same material related by a rotation. Key concepts:

- **CSL (Coincidence Site Lattice)**: When one grain is rotated relative to another, a fraction 1/Sigma of lattice sites coincide. Lower Sigma values produce "special" boundaries with lower energy.
- **Tilt boundary**: Rotation axis lies in the GB plane.
- **Twist boundary**: Rotation axis is perpendicular to the GB plane.
- **Symmetric tilt GB**: The GB plane is a mirror plane for both grains. Notation example: Sigma5(210)[001] means Sigma=5, GB plane=(210), rotation axis=[001].

### Step 1 -- Generate a Grain Boundary with pymatgen

```python
#!/usr/bin/env python3
"""
Generate symmetric tilt grain boundaries using pymatgen's
GrainBoundaryGenerator. Demonstrates Sigma3, Sigma5, Sigma7, etc.
in FCC Cu.
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from pymatgen.core import Structure, Lattice
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
from pymatgen.analysis.gb.grain import GrainBoundaryGenerator

from ase import Atoms
from ase.io import write as ase_write
from ase.optimize import LBFGS, FIRE
from ase.constraints import FixAtoms

from pymatgen.io.ase import AseAtomsAdaptor

# ------------------------------------------------------------------ #
#  Configuration
# ------------------------------------------------------------------ #
MACE_MODEL = "medium"
FMAX = 0.03          # eV/Ang convergence for relaxation
VACUUM = 0.0         # Angstrom -- 0 for fully periodic bicrystal
ELEMENT = "Cu"
A_LAT = 3.615        # Cu lattice constant (Ang)

# Sigma values and corresponding rotation axes / GB planes for FCC
# Format: (sigma, rotation_axis, gb_plane)
GB_CONFIGS = [
    (3,  [1, 1, 0], None),
    (5,  [1, 0, 0], None),
    (7,  [1, 1, 1], None),
    (9,  [1, 1, 0], None),
    (11, [1, 1, 0], None),
]

adaptor = AseAtomsAdaptor()


# ------------------------------------------------------------------ #
#  Step 1: Build the conventional Cu unit cell
# ------------------------------------------------------------------ #
cu_structure = Structure.from_spacegroup(
    "Fm-3m",
    Lattice.cubic(A_LAT),
    [ELEMENT],
    [[0, 0, 0]],
)

print(f"Bulk {ELEMENT}: {cu_structure}")
print(f"  Space group: {SpacegroupAnalyzer(cu_structure).get_space_group_symbol()}")


# ------------------------------------------------------------------ #
#  Step 2: Generate GB structures
# ------------------------------------------------------------------ #
def generate_gb(structure, sigma, rotation_axis, gb_plane=None,
                expand_times=2, vacuum=VACUUM):
    """
    Generate a symmetric tilt GB using pymatgen.
    Returns the GB structure and metadata.
    """
    gb_gen = GrainBoundaryGenerator(structure)

    # Get all possible GB planes for this sigma and rotation axis
    gb_structs = gb_gen.get_grains(
        rotation_axis=rotation_axis,
        sigma=sigma,
        expand_times=expand_times,
        vacuum_thickness=vacuum,
        ab_shift=0.0,
        rm_ratio=0.0,
    )

    if not gb_structs:
        print(f"  No GB found for Sigma{sigma} {rotation_axis}")
        return None, None

    # Take the first (usually lowest-index) GB plane
    gb_struct = gb_structs[0]
    print(f"  Sigma{sigma} {rotation_axis}: "
          f"{len(gb_struct)} atoms, "
          f"rotation angle = {gb_struct.rotation_angle:.2f} deg")

    return gb_struct, gb_struct.rotation_angle


# ------------------------------------------------------------------ #
#  Step 3: Compute bulk energy per atom with MACE
# ------------------------------------------------------------------ #
def compute_bulk_energy(structure, model=MACE_MODEL):
    """Relax bulk and return energy per atom."""
    from mace.calculators import mace_mp
    calc = mace_mp(model=model, default_dtype="float64")

    atoms = adaptor.get_atoms(structure)
    atoms.calc = calc

    opt = LBFGS(atoms, logfile=None)
    opt.run(fmax=FMAX, steps=300)

    e_per_atom = atoms.get_potential_energy() / len(atoms)
    print(f"  Bulk energy per atom: {e_per_atom:.4f} eV")
    return e_per_atom, atoms


# ------------------------------------------------------------------ #
#  Step 4: Relax GB and compute GB energy
# ------------------------------------------------------------------ #
def relax_and_compute_gb_energy(gb_struct, e_bulk_per_atom,
                                model=MACE_MODEL, fix_fraction=0.3):
    """
    Relax a GB structure with MACE.
    Fix the atoms furthest from the GB plane (bulk-like region).
    Compute GB energy:
        gamma_GB = (E_slab - N * E_bulk) / (2 * A)
    Factor of 2 accounts for two GB planes in a periodic bicrystal.

    Returns GB energy in J/m^2.
    """
    from mace.calculators import mace_mp
    calc = mace_mp(model=model, default_dtype="float64")

    atoms = adaptor.get_atoms(gb_struct)
    atoms.calc = calc

    # Identify the GB plane: it is perpendicular to the c-axis
    # in pymatgen's convention. Fix atoms in the middle of the slab
    # (far from the GB planes at top and bottom).
    positions = atoms.get_scaled_positions()[:, 2]  # fractional c-coords
    center = 0.5
    dist_from_center = np.abs(positions - center)

    # Fix the fraction of atoms closest to the center (bulk-like)
    n_fix = int(fix_fraction * len(atoms))
    fix_indices = np.argsort(dist_from_center)[:n_fix]
    atoms.set_constraint(FixAtoms(indices=fix_indices))

    # Relax
    opt = FIRE(atoms, logfile=None)
    opt.run(fmax=FMAX, steps=500)

    # GB energy
    e_total = atoms.get_potential_energy()
    n_atoms = len(atoms)
    cell = atoms.get_cell()
    # GB area = |a x b| (the cross-section perpendicular to c)
    a_vec = cell[0]
    b_vec = cell[1]
    area = np.linalg.norm(np.cross(a_vec, b_vec))  # Ang^2

    # Two GB planes in a periodic bicrystal
    gamma = (e_total - n_atoms * e_bulk_per_atom) / (2 * area)
    gamma_si = gamma * 16.02176634  # eV/Ang^2 -> J/m^2

    return gamma_si, atoms, area


# ------------------------------------------------------------------ #
#  Step 5: Loop over sigma values
# ------------------------------------------------------------------ #
print("\n=== Computing bulk reference ===")
e_bulk, bulk_atoms = compute_bulk_energy(cu_structure)

results = []
gb_atoms_list = []

print("\n=== Generating and relaxing grain boundaries ===")
for sigma, rot_axis, gb_plane in GB_CONFIGS:
    print(f"\nSigma{sigma}, rotation axis {rot_axis}:")
    gb_struct, angle = generate_gb(cu_structure, sigma, rot_axis, gb_plane)

    if gb_struct is None:
        continue

    gamma, gb_atoms, area = relax_and_compute_gb_energy(
        gb_struct, e_bulk, fix_fraction=0.3
    )

    results.append({
        "sigma": sigma,
        "rotation_axis": str(rot_axis),
        "angle_deg": angle,
        "n_atoms": len(gb_atoms),
        "area_ang2": area,
        "gb_energy_J_m2": gamma,
    })
    gb_atoms_list.append(gb_atoms)

    print(f"  GB energy: {gamma:.3f} J/m^2")
    print(f"  GB area:   {area:.1f} Ang^2")

    # Save structure
    ase_write(f"gb_sigma{sigma}_{ELEMENT}.xyz", gb_atoms)

# ------------------------------------------------------------------ #
#  Step 6: Summary table
# ------------------------------------------------------------------ #
import pandas as pd

df = pd.DataFrame(results)
df.sort_values("angle_deg", inplace=True)
print("\n=== Grain Boundary Energy Summary ===")
print(df.to_string(index=False))
df.to_csv(f"gb_energies_{ELEMENT}.csv", index=False)


# ------------------------------------------------------------------ #
#  Step 7: Plot GB energy vs misorientation angle
# ------------------------------------------------------------------ #
fig, ax = plt.subplots(figsize=(8, 5))
ax.plot(df["angle_deg"], df["gb_energy_J_m2"], "o-", markersize=8,
        color="C0", linewidth=1.5)

for _, row in df.iterrows():
    ax.annotate(
        f"$\\Sigma${int(row['sigma'])}",
        (row["angle_deg"], row["gb_energy_J_m2"]),
        textcoords="offset points", xytext=(8, 5), fontsize=10,
    )

ax.set_xlabel("Misorientation angle (degrees)", fontsize=12)
ax.set_ylabel("GB energy (J/m$^2$)", fontsize=12)
ax.set_title(f"Grain boundary energy vs. misorientation: {ELEMENT} (FCC)", fontsize=13)
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(f"gb_energy_vs_angle_{ELEMENT}.png", dpi=150)
plt.close()
print(f"\nSaved gb_energy_vs_angle_{ELEMENT}.png")


# ------------------------------------------------------------------ #
#  Step 8: Visualize GB structure (atomic positions along c-axis)
# ------------------------------------------------------------------ #
if gb_atoms_list:
    gb_show = gb_atoms_list[0]
    fig, ax = plt.subplots(figsize=(6, 8))

    pos = gb_show.get_positions()
    cell = gb_show.get_cell()
    c_len = np.linalg.norm(cell[2])

    # Project onto a-c plane
    ax.scatter(pos[:, 0], pos[:, 2], s=30, c="C0", edgecolors="black",
               linewidth=0.3)

    # Mark approximate GB plane locations
    ax.axhline(y=0, color="red", ls="--", alpha=0.7, label="GB plane")
    ax.axhline(y=c_len, color="red", ls="--", alpha=0.7)
    ax.axhline(y=c_len / 2, color="red", ls="--", alpha=0.4,
               label="GB plane (periodic image)")

    ax.set_xlabel("x (Ang)", fontsize=11)
    ax.set_ylabel("z (Ang) -- normal to GB", fontsize=11)
    ax.set_title(f"GB structure: $\\Sigma${results[0]['sigma']} {ELEMENT}",
                 fontsize=12)
    ax.legend()
    ax.set_aspect("equal")
    plt.tight_layout()
    plt.savefig(f"gb_structure_sigma{results[0]['sigma']}_{ELEMENT}.png", dpi=150)
    plt.close()
    print(f"Saved gb_structure_sigma{results[0]['sigma']}_{ELEMENT}.png")

print("\nDone.")
```

### Step 2 -- GB Energy with QE Validation

For high-accuracy validation of the MACE GB energy, run a QE SCF calculation on the relaxed GB structure:

```python
#!/usr/bin/env python3
"""
Write QE input for a grain boundary structure and a bulk reference,
then compare energies.
"""

from ase.io import read as ase_read
from ase.io.espresso import write_espresso_in
import numpy as np

# Read the MACE-relaxed GB structure
gb_atoms = ase_read("gb_sigma5_Cu.xyz")

# QE parameters
input_data = {
    "control": {
        "calculation": "scf",
        "prefix": "gb_sigma5",
        "outdir": "./tmp_gb",
        "pseudo_dir": "/opt/pseudo",
        "tprnfor": True,
        "tstress": True,
    },
    "system": {
        "ecutwfc": 50,
        "ecutrho": 400,
        "occupations": "smearing",
        "smearing": "mv",
        "degauss": 0.02,
    },
    "electrons": {
        "conv_thr": 1.0e-6,
        "mixing_beta": 0.3,
    },
}

pseudopotentials = {"Cu": "Cu.pbe-dn-kjpaw_psl.1.0.0.UPF"}

# Determine k-points: fine in-plane, coarse along GB normal (c-axis)
cell = gb_atoms.get_cell()
a_len = np.linalg.norm(cell[0])
b_len = np.linalg.norm(cell[1])
c_len = np.linalg.norm(cell[2])

# Roughly 30/length Angstrom rule for k-spacing
ka = max(1, int(round(30 / a_len)))
kb = max(1, int(round(30 / b_len)))
kc = max(1, int(round(30 / c_len)))
kpts = (ka, kb, kc)
print(f"k-points: {kpts} (cell: {a_len:.1f} x {b_len:.1f} x {c_len:.1f} Ang)")

with open("gb_sigma5.pwi", "w") as f:
    write_espresso_in(
        f, gb_atoms,
        input_data=input_data,
        pseudopotentials=pseudopotentials,
        kpts=kpts,
    )
print("Wrote gb_sigma5.pwi")

# Also write a bulk reference (single FCC Cu unit cell)
from ase.build import bulk
cu_bulk = bulk("Cu", "fcc", a=3.615, cubic=True)

input_data["control"]["prefix"] = "cu_bulk"
input_data["control"]["outdir"] = "./tmp_bulk"

with open("cu_bulk.pwi", "w") as f:
    write_espresso_in(
        f, cu_bulk,
        input_data=input_data,
        pseudopotentials=pseudopotentials,
        kpts=(8, 8, 8),
    )
print("Wrote cu_bulk.pwi")
print("\nRun both with: mpirun -np 4 pw.x -in <file>.pwi > <file>.pwo")
```

## Key Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `sigma` | varies | CSL sigma value (3, 5, 7, 9, 11, ...); lower = more coincident sites |
| `rotation_axis` | varies | Crystallographic axis of rotation, e.g., [1,1,0] |
| `expand_times` | 2 | Number of bulk repeats on each side of the GB; larger = thicker slab |
| `vacuum_thickness` | 0.0 | Vacuum in Angstrom; 0 = fully periodic bicrystal (recommended) |
| `fix_fraction` | 0.3 | Fraction of atoms to fix during relaxation (bulk-like interior) |
| `FMAX` | 0.03 eV/A | Force convergence for relaxation |
| `MACE_MODEL` | `"medium"` | MACE-MP-0 model size |

## Interpreting Results

- **GB energy (gamma)** is typically reported in J/m^2. For FCC metals, values range from 0.3 to 1.5 J/m^2 depending on the GB type.
- **Low-sigma cusps**: Sigma3 (twin boundary, ~60.0 deg in FCC) and Sigma11 boundaries often show energy minima. The gamma vs. angle plot should show these cusps.
- **Sigma3 in FCC**: The coherent twin boundary is the lowest-energy GB in most FCC metals (~0.02-0.05 J/m^2 for Cu). If your value is much higher, it may be an incoherent twin or a different GB plane.
- **Factor of 2**: The periodic bicrystal contains two GB planes. The formula divides by 2A to get the energy per single GB.
- **Convergence with slab thickness**: Increase `expand_times` from 2 to 4 and check that gamma changes by less than 5%. If not, the slab is too thin.
- **MACE accuracy**: MACE-MP-0 typically reproduces GB energy trends (relative ordering) well, but absolute values may differ from DFT by 10-30%.

## Common Issues

| Issue | Solution |
|-------|----------|
| `GrainBoundaryGenerator` returns empty list | Not all sigma values have valid GBs for a given rotation axis and crystal structure. Try a different axis or sigma. |
| Atoms overlap at the GB | Reduce `rm_ratio` (default 0) or use `ab_shift` to shift grains relative to each other |
| Very high GB energy | Check that the bulk reference energy is correct; ensure `expand_times` is large enough |
| Relaxation does not converge | Use FIRE instead of LBFGS; increase `steps`; reduce `FMAX` |
| Negative GB energy | Usually a sign that the bulk energy reference is wrong (e.g., different calculator settings). Recompute bulk with the same MACE model and settings. |
| Large cell causes OOM | Reduce `expand_times` or use a smaller MACE model (`"small"`) |
| Structure distorted after relaxation | Fix more atoms (increase `fix_fraction` to 0.5) to prevent bulk-like atoms from shifting |
