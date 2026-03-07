# Surface Slab Builder

## When to Use

- You need to build a surface slab model from a bulk crystal given Miller indices (hkl).
- You want to study surface energies, adsorption, or catalysis with DFT or MACE.
- You need both symmetric (for surface energy) and asymmetric (for adsorption) slab models.
- You want to explore all possible surface terminations for a given (hkl) orientation.
- You need a slab with a vacuum layer for periodic boundary condition calculations.

## Method Selection

| Criterion | pymatgen SlabGenerator | ASE surface module | Manual construction |
|---|---|---|---|
| Miller index support | Any (hkl) | Limited to low-index for common lattices | Any, but tedious |
| Termination control | Auto-detects all terminations | Limited | Full manual control |
| Symmetry handling | Preserves symmetry, can enforce symmetric slabs | Basic | Manual |
| Best for | General slab generation, all crystal systems | Quick FCC/BCC/HCP low-index surfaces | Custom geometry |

## Prerequisites

- pymatgen (pre-installed)
- ASE (pre-installed)
- numpy, matplotlib (pre-installed)

---

## Detailed Steps

### Method A: pymatgen SlabGenerator (Recommended)

```python
#!/usr/bin/env python3
"""
Build surface slabs using pymatgen SlabGenerator.
Covers any Miller index for any crystal structure.
Handles terminations, vacuum, symmetry, and orthogonality.
"""

import numpy as np
import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from pymatgen.core import Structure, Lattice
from pymatgen.core.surface import SlabGenerator, Slab
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
from pymatgen.io.vasp import Poscar
from pymatgen.io.ase import AseAtomsAdaptor
from pathlib import Path

# ============================================================
# 1. Load or build the bulk structure
# ============================================================
# Option 1: Load from file
# bulk = Structure.from_file("POSCAR")
# bulk = Structure.from_file("structure.cif")

# Option 2: Build from spacegroup
# FCC Cu
bulk = Structure.from_spacegroup(
    "Fm-3m",
    lattice=Lattice.cubic(3.615),
    species=["Cu"],
    coords=[[0, 0, 0]],
)

# Option 3: From Materials Project
# from mp_api.client import MPRester
# with MPRester("YOUR_API_KEY") as mpr:
#     bulk = mpr.get_structure_by_material_id("mp-30")

print(f"Bulk structure: {bulk.composition.reduced_formula}")
sga = SpacegroupAnalyzer(bulk)
print(f"Space group: {sga.get_space_group_symbol()}")

# ============================================================
# 2. Generate slabs for a given Miller index
# ============================================================
MILLER_INDEX = (1, 1, 1)    # Surface orientation
MIN_SLAB_SIZE = 10.0        # Minimum slab thickness in Angstrom
MIN_VACUUM_SIZE = 15.0      # Minimum vacuum thickness in Angstrom
CENTER_SLAB = True          # Center the slab in the cell
IN_UNIT_PLANES = False      # If True, min_slab_size is in units of planes
PRIMITIVE = True            # Use primitive cell (smaller, faster)
MAX_NORMAL_SEARCH = 2       # Search range for orthogonal c vector

slabgen = SlabGenerator(
    initial_structure=bulk,
    miller_index=MILLER_INDEX,
    min_slab_size=MIN_SLAB_SIZE,
    min_vacuum_size=MIN_VACUUM_SIZE,
    center_slab=CENTER_SLAB,
    in_unit_planes=IN_UNIT_PLANES,
    primitive=PRIMITIVE,
    max_normal_search=MAX_NORMAL_SEARCH,
    lll_reduce=True,            # LLL-reduce the slab cell
    reorient_lattice=True,      # Put c perpendicular to surface
)

# Get all possible terminations
all_slabs = slabgen.get_slabs(
    symmetrize=False,     # Set True for symmetric slabs (surface energy)
    repair=True,          # Repair broken bonds at surface
)

print(f"\nMiller index: {MILLER_INDEX}")
print(f"Found {len(all_slabs)} unique slab termination(s)")

output_dir = Path(f"slab_{''.join(map(str, MILLER_INDEX))}")
output_dir.mkdir(exist_ok=True)

slab_info = []
for i, slab in enumerate(all_slabs):
    # Slab properties
    n_atoms = len(slab)
    thickness = slab.lattice.c  # total cell height
    # Estimate actual slab thickness
    z_coords = slab.frac_coords[:, 2]
    z_range = (z_coords.max() - z_coords.min()) * slab.lattice.c

    print(f"\n--- Termination {i} ---")
    print(f"  Atoms: {n_atoms}")
    print(f"  Slab thickness: {z_range:.2f} A")
    print(f"  Cell c: {thickness:.2f} A")
    print(f"  Surface area: {slab.surface_area:.2f} A^2")
    print(f"  Formula: {slab.composition.reduced_formula}")
    print(f"  Is symmetric: {slab.is_symmetric()}")
    print(f"  Is polar: {slab.is_polar()}")

    # Save
    slab.to(str(output_dir / f"slab_term{i}.cif"))
    Poscar(slab).write_file(str(output_dir / f"POSCAR_term{i}"))
    print(f"  Saved: {output_dir}/slab_term{i}.cif, POSCAR_term{i}")

    slab_info.append({
        "termination": i,
        "n_atoms": n_atoms,
        "slab_thickness_A": round(z_range, 2),
        "surface_area_A2": round(slab.surface_area, 2),
        "is_symmetric": slab.is_symmetric(),
        "is_polar": slab.is_polar(),
    })

# Save summary
with open(str(output_dir / "slab_summary.json"), "w") as f:
    json.dump({
        "miller_index": list(MILLER_INDEX),
        "bulk": bulk.composition.reduced_formula,
        "terminations": slab_info,
    }, f, indent=2)

# ============================================================
# 3. Generate symmetric slabs (for surface energy calculations)
# ============================================================
print("\n=== Symmetric slabs (for surface energy) ===")
sym_slabs = slabgen.get_slabs(
    symmetrize=True,    # Force mirror symmetry
    repair=True,
)

for i, slab in enumerate(sym_slabs):
    if slab.is_symmetric():
        print(f"  Symmetric slab {i}: {len(slab)} atoms, "
              f"symmetric={slab.is_symmetric()}, polar={slab.is_polar()}")
        slab.to(str(output_dir / f"slab_symmetric_{i}.cif"))

# ============================================================
# 4. Multiple Miller indices scan
# ============================================================
print("\n=== Low-index surfaces scan ===")
low_index_surfaces = [
    (1, 0, 0), (1, 1, 0), (1, 1, 1),
    (2, 1, 0), (2, 1, 1), (2, 2, 1),
]

for hkl in low_index_surfaces:
    sg = SlabGenerator(
        bulk, hkl,
        min_slab_size=8.0,
        min_vacuum_size=12.0,
        center_slab=True,
        primitive=True,
    )
    slabs = sg.get_slabs(symmetrize=False)
    n_terms = len(slabs)
    n_atoms_min = min(len(s) for s in slabs) if slabs else 0
    print(f"  ({hkl[0]}{hkl[1]}{hkl[2]}): {n_terms} termination(s), "
          f"min {n_atoms_min} atoms")

# ============================================================
# 5. Visualize slab structure (side view)
# ============================================================
def plot_slab_side_view(slab, filename="slab_side_view.png"):
    """Plot the slab in side view (xz plane)."""
    fig, ax = plt.subplots(figsize=(6, 8))

    species_colors = {}
    color_cycle = plt.cm.tab10.colors
    for idx, sp in enumerate(set(str(s.specie) for s in slab)):
        species_colors[sp] = color_cycle[idx % len(color_cycle)]

    cart_coords = slab.cart_coords
    for site in slab:
        color = species_colors[str(site.specie)]
        ax.scatter(site.coords[0], site.coords[2], s=80, c=[color],
                   edgecolors="black", linewidth=0.5, zorder=3)

    # Draw cell boundaries
    a = slab.lattice.matrix[0]
    c = slab.lattice.matrix[2]
    corners = np.array([
        [0, 0], [a[0], a[2]], [a[0]+c[0], a[2]+c[2]], [c[0], c[2]], [0, 0]
    ])
    ax.plot(corners[:, 0], corners[:, 1], "k-", linewidth=1, alpha=0.5)

    # Mark vacuum
    z_coords = cart_coords[:, 2]
    z_min, z_max = z_coords.min(), z_coords.max()
    cell_height = slab.lattice.c
    ax.axhspan(z_max + 1, cell_height, alpha=0.1, color="blue", label="vacuum")
    ax.axhspan(0, z_min - 1, alpha=0.1, color="blue")

    # Legend
    for sp, color in species_colors.items():
        ax.scatter([], [], c=[color], s=60, label=sp, edgecolors="black")
    ax.legend(fontsize=10, loc="upper right")

    ax.set_xlabel("x (A)", fontsize=12)
    ax.set_ylabel("z (A)", fontsize=12)
    ax.set_title(f"Slab side view - {slab.composition.reduced_formula}", fontsize=13)
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(filename, dpi=150, bbox_inches="tight")
    print(f"Slab visualization saved: {filename}")

if all_slabs:
    plot_slab_side_view(all_slabs[0], str(output_dir / "slab_side_view.png"))

# ============================================================
# 6. Generate QE input for slab
# ============================================================
def write_qe_slab_input(slab, pseudo_dir="./pseudo", filename="slab_scf.in"):
    """Generate QE pw.x input for a slab calculation."""
    cell = slab.lattice.matrix
    elements = sorted(set(str(s.specie) for s in slab))

    cell_lines = "\n".join(
        f"  {v[0]:.10f}  {v[1]:.10f}  {v[2]:.10f}" for v in cell
    )

    species_lines = []
    for el in elements:
        from pymatgen.core.periodic_table import Element as PmgElement
        mass = PmgElement(el).atomic_mass
        species_lines.append(f"  {el:4s} {mass:10.4f}  {el}.pbe-n-rrkjus_psl.1.0.0.UPF")

    pos_lines = []
    for site in slab:
        fc = site.frac_coords
        pos_lines.append(f"  {str(site.specie):4s} {fc[0]:.10f} {fc[1]:.10f} {fc[2]:.10f}")

    # For slab: use smearing, k-mesh with 1 along c
    qe_input = f"""&CONTROL
    calculation = 'relax'
    prefix      = 'slab'
    outdir      = './tmp'
    pseudo_dir  = '{pseudo_dir}'
    tprnfor     = .true.
    tstress     = .true.
/
&SYSTEM
    ibrav       = 0
    nat         = {len(slab)}
    ntyp        = {len(elements)}
    ecutwfc     = 50.0
    ecutrho     = 400.0
    occupations = 'smearing'
    smearing    = 'cold'
    degauss     = 0.01
    assume_isolated = '2D'
/
&ELECTRONS
    conv_thr    = 1.0d-8
    mixing_beta = 0.4
/
&IONS
    ion_dynamics = 'bfgs'
/

CELL_PARAMETERS angstrom
{cell_lines}

ATOMIC_SPECIES
{chr(10).join(species_lines)}

ATOMIC_POSITIONS crystal
{chr(10).join(pos_lines)}

K_POINTS automatic
  8 8 1  0 0 0
"""

    with open(filename, "w") as f:
        f.write(qe_input)
    print(f"QE slab input written: {filename}")

if all_slabs:
    write_qe_slab_input(all_slabs[0], filename=str(output_dir / "slab_scf.in"))

print(f"\n=== All files saved to {output_dir}/ ===")
```

### Method B: ASE Surface Module

```python
#!/usr/bin/env python3
"""
Build surface slabs using ASE's built-in surface builders.
Best for common low-index surfaces of FCC, BCC, HCP metals.
"""

import numpy as np
from ase.build import (
    fcc111, fcc110, fcc100,
    bcc111, bcc110, bcc100,
    hcp0001, hcp10m10,
    surface, add_vacuum,
)
from ase.io import write
from ase.visualize.plot import plot_atoms
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ============================================================
# 1. FCC surfaces using dedicated builders
# ============================================================
print("=== FCC Cu surfaces ===")

# FCC (111) surface
slab_111 = fcc111("Cu", size=(3, 3, 5), a=3.615, vacuum=15.0)
print(f"  Cu(111): {len(slab_111)} atoms, cell = {slab_111.cell.lengths()}")
write("Cu_111.cif", slab_111)

# FCC (100) surface
slab_100 = fcc100("Cu", size=(3, 3, 5), a=3.615, vacuum=15.0)
print(f"  Cu(100): {len(slab_100)} atoms, cell = {slab_100.cell.lengths()}")
write("Cu_100.cif", slab_100)

# FCC (110) surface
slab_110 = fcc110("Cu", size=(3, 3, 5), a=3.615, vacuum=15.0)
print(f"  Cu(110): {len(slab_110)} atoms, cell = {slab_110.cell.lengths()}")
write("Cu_110.cif", slab_110)

# ============================================================
# 2. General surface builder (any Miller index)
# ============================================================
print("\n=== General surface builder ===")

from ase.build import bulk

cu_bulk = bulk("Cu", "fcc", a=3.615)

# Any (hkl) using the general surface() function
slab_211 = surface(cu_bulk, (2, 1, 1), layers=6, vacuum=15.0)
print(f"  Cu(211): {len(slab_211)} atoms")
write("Cu_211.cif", slab_211)

slab_321 = surface(cu_bulk, (3, 2, 1), layers=4, vacuum=15.0)
print(f"  Cu(321): {len(slab_321)} atoms")
write("Cu_321.cif", slab_321)

# ============================================================
# 3. Add adsorbate site finding
# ============================================================
from ase.build import add_adsorbate

# Add a CO molecule on the FCC(111) surface
slab_ads = fcc111("Cu", size=(3, 3, 5), a=3.615, vacuum=15.0)

# Add CO at a top site
co = add_adsorbate(slab_ads, "C", height=1.8, position="ontop")

print(f"\n  Cu(111) with C adsorbate: {len(slab_ads)} atoms")
write("Cu_111_with_C.cif", slab_ads)

# ============================================================
# 4. Relax slab with MACE
# ============================================================
print("\n=== Relax slab with MACE ===")

import warnings
warnings.filterwarnings("ignore")
from mace.calculators import mace_mp
from ase.optimize import BFGS
from ase.constraints import FixAtoms

slab = fcc111("Cu", size=(3, 3, 5), a=3.615, vacuum=15.0)

# Fix bottom 2 layers
z_positions = slab.positions[:, 2]
z_sorted = np.sort(np.unique(np.round(z_positions, 2)))
cutoff = z_sorted[1] + 0.1  # Fix bottom 2 layers
mask = slab.positions[:, 2] < cutoff
constraint = FixAtoms(mask=mask)
slab.set_constraint(constraint)
print(f"  Fixed {mask.sum()} atoms in bottom layers")

calc = mace_mp(model="medium", dispersion=False, default_dtype="float64")
slab.calc = calc

opt = BFGS(slab, logfile="slab_relax.log")
opt.run(fmax=0.01, steps=200)

e_slab = slab.get_potential_energy()
print(f"  Relaxed slab energy: {e_slab:.4f} eV")
write("Cu_111_relaxed.cif", slab)
```

### Method C: VASP (Future External Access)

```python
#!/usr/bin/env python3
"""
Generate VASP input files for surface slab calculations.
Includes POSCAR, INCAR, and KPOINTS for slab DFT.
"""

from pymatgen.core import Structure, Lattice
from pymatgen.core.surface import SlabGenerator
from pymatgen.io.vasp import Poscar, Incar, Kpoints

# Build bulk
bulk = Structure.from_spacegroup(
    "Fm-3m",
    lattice=Lattice.cubic(3.615),
    species=["Cu"],
    coords=[[0, 0, 0]],
)

# Generate slab
slabgen = SlabGenerator(
    bulk,
    miller_index=(1, 1, 1),
    min_slab_size=12.0,
    min_vacuum_size=15.0,
    center_slab=True,
    primitive=True,
)

slabs = slabgen.get_slabs(symmetrize=True)
slab = slabs[0]
print(f"Slab: {slab.composition}, {len(slab)} atoms")
print(f"Symmetric: {slab.is_symmetric()}")

# POSCAR
Poscar(slab).write_file("POSCAR_slab")

# KPOINTS (use 1 along c for slab)
kpts = Kpoints.gamma_automatic(kpts=(8, 8, 1), shift=(0, 0, 0))
kpts.write_file("KPOINTS_slab")

# INCAR for slab relaxation
incar = Incar({
    "SYSTEM": f"Cu(111) slab",
    "ENCUT": 520,
    "PREC": "Accurate",
    "EDIFF": 1e-6,
    "EDIFFG": -0.01,
    "IBRION": 2,
    "NSW": 100,
    "ISIF": 2,            # Relax ions only, not cell
    "ISMEAR": 1,
    "SIGMA": 0.1,
    "LWAVE": False,
    "LCHARG": True,
    "LDIPOL": True,        # Dipole correction
    "IDIPOL": 3,           # Along c (vacuum direction)
    "LORBIT": 11,
    "LVDW": True,          # vdW for surfaces
    "IVDW": 12,            # DFT-D3(BJ)
})
incar.write_file("INCAR_slab")

# Generate selective dynamics POSCAR (fix bottom layers)
import numpy as np
frac_z = slab.frac_coords[:, 2]
z_sorted = np.sort(np.unique(np.round(frac_z, 4)))
n_layers_to_fix = 2
z_cutoff = z_sorted[n_layers_to_fix - 1] + 0.01

selective_dynamics = []
for site in slab:
    if site.frac_coords[2] < z_cutoff:
        selective_dynamics.append([False, False, False])  # Fixed
    else:
        selective_dynamics.append([True, True, True])     # Free

poscar_sd = Poscar(slab, selective_dynamics=selective_dynamics)
poscar_sd.write_file("POSCAR_slab_sd")
print(f"\nPOSCAR with selective dynamics: {sum(1 for sd in selective_dynamics if sd[0])} "
      f"free atoms, {sum(1 for sd in selective_dynamics if not sd[0])} fixed atoms")

print("\nVASP slab files written:")
print("  POSCAR_slab, POSCAR_slab_sd, KPOINTS_slab, INCAR_slab")
print("\nKey settings: ISIF=2, LDIPOL=True, IDIPOL=3, K-mesh: Nx Ny 1")
```

---

## Key Parameters

| Parameter | Typical Value | Notes |
|---|---|---|
| `min_slab_size` | 10--15 A | Minimum slab thickness. Converge surface energy vs. slab thickness. |
| `min_vacuum_size` | 15--20 A | Prevents interaction between periodic images of the slab. |
| `center_slab` | True | Centers atoms in the cell, useful for visualization and dipole correction. |
| `primitive` | True | Uses primitive in-plane cell (fewer atoms). Set False for adsorption supercells. |
| `symmetrize` | True for surface energy, False for adsorption | Symmetric slabs have identical top and bottom surfaces. |
| `max_normal_search` | 2 | Search range for c vector perpendicular to surface. Increase for high-index surfaces. |
| `in_unit_planes` | False | If True, min_slab_size counts atomic planes instead of Angstroms. |
| Fixed layers | 2--3 bottom layers | Fix bottom layers to mimic bulk-like behavior. |
| `ISIF` (VASP) | 2 | Relax ions only, not cell shape, for slab calculations. |
| `assume_isolated` (QE) | `'2D'` | Coulomb truncation for slab geometry. |

## Common Issues

| Problem | Solution |
|---|---|
| SlabGenerator returns zero slabs | Check the Miller index is valid. Try increasing `max_normal_search` to 3 or 4. |
| Slab is polar (net dipole) | Use `symmetrize=True` for symmetric slabs, or apply dipole correction (LDIPOL/IDIPOL in VASP, assume_isolated='2D' in QE). |
| Too many atoms in slab | Set `primitive=True`, reduce `min_slab_size`, or use a lower-index surface. |
| Surface energy not converged | Increase slab thickness. Test 4--8 layer slabs. Symmetric slabs give more reliable surface energies. |
| Adsorbate sinks into surface during relaxation | Fix more substrate layers. Use tighter fmax. Check that the adsorbate starting position is reasonable. |
| High-index surface has reconstructed geometry | Run relaxation with MACE or DFT. High-index surfaces often reconstruct to lower-energy configurations. |
| In-plane cell too small for adsorption | Set `primitive=False` or build a supercell of the slab: `slab.make_supercell([2, 2, 1])`. |
