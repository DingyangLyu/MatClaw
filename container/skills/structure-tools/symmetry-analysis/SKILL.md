# Symmetry Analysis

## When to Use

- You need to determine the space group, point group, or crystal system of a crystal structure.
- You want to find the primitive cell or conventional standard cell.
- You need to identify Wyckoff positions and symmetrically equivalent atoms.
- You want to check if a relaxed structure has maintained its expected symmetry.
- You need to symmetrize a slightly distorted structure back to its ideal space group.
- You want to determine the molecular point group of a molecule or cluster.
- You need site symmetry information for spectroscopy or selection rule analysis.
- You want to compare symmetry before and after a phase transition (VASPKIT 601-609 equivalent).

## Method Selection

| Criterion | pymatgen + spglib (Python) | QE | VASP |
|---|---|---|---|
| Space group detection | `SpacegroupAnalyzer` (spglib backend) | Not built-in (post-process) | Not built-in (post-process) |
| Primitive cell | `get_primitive_standard_structure()` | N/A | N/A |
| Conventional cell | `get_conventional_standard_structure()` | N/A | N/A |
| Wyckoff positions | `get_symmetrized_structure()` | N/A | N/A |
| Symmetry operations | Full list via spglib | `verbosity='high'` prints ops | ISYM in OUTCAR |
| Molecular symmetry | pymatgen `PointGroupAnalyzer` | N/A | N/A |
| Available now | Yes | Post-processing only | Post-processing only |

## Prerequisites

- A crystal structure in any common format (CIF, POSCAR, XYZ).
- Python packages: `pymatgen`, `spglib`, `ase`, `numpy` (pre-installed).
- For molecular symmetry: a molecule or cluster (not periodic).

---

## Detailed Steps

### Method A: pymatgen + spglib -- Full Symmetry Analysis

#### Step A1: Space group and basic symmetry information

```python
#!/usr/bin/env python3
"""
Determine space group, point group, crystal system, and lattice type.
Equivalent to VASPKIT function 601.
"""
import numpy as np
import spglib
from pymatgen.core.structure import Structure
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer

# ── Load structure ────────────────────────────────────────────────
structure = Structure.from_file("structure.cif")
print(f"Structure: {structure.composition.reduced_formula}")
print(f"Atoms: {len(structure)}")
print(f"Volume: {structure.volume:.4f} A^3")
print()

# ── Symmetry analysis with different tolerances ──────────────────
# symprec controls how precisely atoms must match symmetry positions
for symprec in [0.001, 0.01, 0.1]:
    sga = SpacegroupAnalyzer(structure, symprec=symprec, angle_tolerance=5)

    print(f"=== symprec = {symprec} ===")
    print(f"  Space group symbol:  {sga.get_space_group_symbol()}")
    print(f"  Space group number:  {sga.get_space_group_number()}")
    print(f"  Point group:         {sga.get_point_group_symbol()}")
    print(f"  Crystal system:      {sga.get_crystal_system()}")
    print(f"  Lattice type:        {sga.get_lattice_type()}")
    print(f"  Hall symbol:         {sga.get_hall()}")
    print(f"  Number of symmetry operations: {len(sga.get_symmetry_operations())}")
    print()

# ── Detailed analysis with default symprec ────────────────────────
sga = SpacegroupAnalyzer(structure, symprec=0.01)

# International Tables notation
print(f"International symbol: {sga.get_space_group_symbol()}")
print(f"Space group number:   {sga.get_space_group_number()}")
print(f"Point group (Schoenflies): {sga.get_point_group_symbol()}")
print(f"Crystal system:       {sga.get_crystal_system()}")
print(f"Lattice type:         {sga.get_lattice_type()}")
print(f"Is Laue class:        {sga.get_crystal_system()}")

# Check centering
sg_symbol = sga.get_space_group_symbol()
centering = sg_symbol[0] if sg_symbol else "?"
centering_names = {
    "P": "Primitive",
    "I": "Body-centered",
    "F": "Face-centered",
    "A": "A-centered",
    "B": "B-centered",
    "C": "C-centered",
    "R": "Rhombohedral",
}
print(f"Centering: {centering} ({centering_names.get(centering, 'unknown')})")
```

#### Step A2: Primitive and conventional cells

```python
#!/usr/bin/env python3
"""
Find primitive cell and conventional standard cell.
Equivalent to VASPKIT functions 602-603.
"""
import numpy as np
from pymatgen.core.structure import Structure
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
from pymatgen.io.vasp import Poscar

# ── Load structure ────────────────────────────────────────────────
structure = Structure.from_file("structure.cif")
sga = SpacegroupAnalyzer(structure, symprec=0.01)

print(f"Input structure: {structure.composition.reduced_formula}")
print(f"  Atoms: {len(structure)}")
print(f"  Lattice: a={structure.lattice.a:.4f}, b={structure.lattice.b:.4f}, "
      f"c={structure.lattice.c:.4f}")
print(f"  Volume: {structure.volume:.4f} A^3")
print(f"  Space group: {sga.get_space_group_symbol()}")
print()

# ── Primitive standard cell ──────────────────────────────────────
primitive = sga.get_primitive_standard_structure()
print(f"Primitive cell:")
print(f"  Atoms: {len(primitive)}")
print(f"  Lattice: a={primitive.lattice.a:.4f}, b={primitive.lattice.b:.4f}, "
      f"c={primitive.lattice.c:.4f}")
print(f"  Angles: alpha={primitive.lattice.alpha:.2f}, beta={primitive.lattice.beta:.2f}, "
      f"gamma={primitive.lattice.gamma:.2f}")
print(f"  Volume: {primitive.volume:.4f} A^3")

# Verify: volume ratio should equal natom ratio
ratio = structure.volume / primitive.volume
print(f"  Volume ratio (input/primitive): {ratio:.1f}")

primitive.to("POSCAR_primitive", fmt="poscar")
primitive.to("primitive.cif", fmt="cif")
print(f"  Saved to: POSCAR_primitive, primitive.cif")
print()

# ── Conventional standard cell ───────────────────────────────────
conventional = sga.get_conventional_standard_structure()
print(f"Conventional cell:")
print(f"  Atoms: {len(conventional)}")
print(f"  Lattice: a={conventional.lattice.a:.4f}, b={conventional.lattice.b:.4f}, "
      f"c={conventional.lattice.c:.4f}")
print(f"  Angles: alpha={conventional.lattice.alpha:.2f}, "
      f"beta={conventional.lattice.beta:.2f}, gamma={conventional.lattice.gamma:.2f}")
print(f"  Volume: {conventional.volume:.4f} A^3")

conventional.to("POSCAR_conventional", fmt="poscar")
conventional.to("conventional.cif", fmt="cif")
print(f"  Saved to: POSCAR_conventional, conventional.cif")
print()

# ── Refined structure (symmetrized) ──────────────────────────────
refined = sga.get_refined_structure()
print(f"Refined (symmetrized) cell:")
print(f"  Atoms: {len(refined)}")
print(f"  Lattice: a={refined.lattice.a:.4f}, b={refined.lattice.b:.4f}, "
      f"c={refined.lattice.c:.4f}")
refined.to("POSCAR_refined", fmt="poscar")
```

#### Step A3: Wyckoff positions and equivalent atoms

```python
#!/usr/bin/env python3
"""
Find Wyckoff positions and symmetrically equivalent atoms.
Equivalent to VASPKIT functions 604-605.
"""
import numpy as np
import spglib
from pymatgen.core.structure import Structure
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer

# ── Load structure ────────────────────────────────────────────────
structure = Structure.from_file("structure.cif")
sga = SpacegroupAnalyzer(structure, symprec=0.01)

print(f"Structure: {structure.composition.reduced_formula}")
print(f"Space group: {sga.get_space_group_symbol()} (#{sga.get_space_group_number()})")
print()

# ── Get symmetrized structure with Wyckoff labels ─────────────────
sym_struct = sga.get_symmetrized_structure()

# Group sites by Wyckoff position
print("=== Wyckoff Positions ===")
print(f"{'Index':>6s}  {'Element':>8s}  {'Wyckoff':>8s}  {'Multiplicity':>12s}  "
      f"{'Fractional Coordinates':>30s}")
print("-" * 75)

for i, (site, wyckoff) in enumerate(zip(sym_struct, sym_struct.wyckoff_symbols)):
    fc = site.frac_coords
    print(f"{i:6d}  {str(site.specie):>8s}  {wyckoff:>8s}  "
          f"{'':>12s}  "
          f"({fc[0]:8.5f}, {fc[1]:8.5f}, {fc[2]:8.5f})")

# ── Group equivalent atoms ───────────────────────────────────────
print("\n=== Equivalent Atom Groups ===")
equiv_groups = sym_struct.equivalent_indices
for group_idx, group in enumerate(equiv_groups):
    sites_in_group = [sym_struct[i] for i in group]
    species = str(sites_in_group[0].specie)
    wyckoff = sym_struct.wyckoff_symbols[group[0]]
    print(f"\nGroup {group_idx + 1}: {species} ({wyckoff}), multiplicity = {len(group)}")
    for idx in group:
        fc = sym_struct[idx].frac_coords
        print(f"  Site {idx:3d}: ({fc[0]:8.5f}, {fc[1]:8.5f}, {fc[2]:8.5f})")

# ── Using spglib directly for more detailed information ───────────
print("\n=== spglib Analysis ===")
# Convert pymatgen structure to spglib format
lattice = structure.lattice.matrix
positions = structure.frac_coords
numbers = [site.specie.Z for site in structure]
cell = (lattice, positions, numbers)

# Get symmetry dataset
dataset = spglib.get_symmetry_dataset(cell, symprec=0.01)

print(f"Space group (intl):  {dataset['international']}")
print(f"Space group number:  {dataset['number']}")
print(f"Hall symbol:         {dataset['hall']}")
print(f"Hall number:         {dataset['hall_number']}")
print(f"Point group:         {dataset['pointgroup']}")
print(f"Transformation matrix:\n{dataset['transformation_matrix']}")
print(f"Origin shift: {dataset['origin_shift']}")
print()

# Wyckoff letters from spglib
print("Wyckoff letters (spglib):")
for i, (letter, equiv) in enumerate(zip(dataset['wyckoffs'], dataset['equivalent_atoms'])):
    species = structure[i].specie
    fc = structure[i].frac_coords
    print(f"  Atom {i:3d}: {str(species):4s}  Wyckoff: {letter}  "
          f"Equiv group: {equiv}  "
          f"({fc[0]:8.5f}, {fc[1]:8.5f}, {fc[2]:8.5f})")

# ── Site symmetry ─────────────────────────────────────────────────
print("\n=== Site Symmetry ===")
site_symmetry = dataset.get('site_symmetry_symbols', None)
if site_symmetry is not None:
    for i, sym in enumerate(site_symmetry):
        species = structure[i].specie
        print(f"  Atom {i:3d} ({str(species):4s}): site symmetry = {sym}")
```

#### Step A4: Symmetry operations

```python
#!/usr/bin/env python3
"""
List all symmetry operations of the space group.
Includes rotation matrices and translation vectors.
Equivalent to VASPKIT function 606.
"""
import numpy as np
import spglib
from pymatgen.core.structure import Structure
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer

# ── Load structure ────────────────────────────────────────────────
structure = Structure.from_file("structure.cif")
sga = SpacegroupAnalyzer(structure, symprec=0.01)

print(f"Structure: {structure.composition.reduced_formula}")
print(f"Space group: {sga.get_space_group_symbol()} (#{sga.get_space_group_number()})")

# ── Get symmetry operations from pymatgen ─────────────────────────
symmops = sga.get_symmetry_operations()
print(f"\nNumber of symmetry operations: {len(symmops)}")

print("\n=== Symmetry Operations (fractional) ===")
for i, op in enumerate(symmops):
    rot = op.rotation_matrix
    trans = op.translation_vector
    # Determine operation type
    det = np.linalg.det(rot)
    trace = np.trace(rot)

    # Classify the operation
    if np.allclose(rot, np.eye(3)):
        op_type = "Identity (E)"
    elif np.allclose(rot, -np.eye(3)):
        op_type = "Inversion (i)"
    elif abs(det - 1) < 1e-6:
        if abs(trace - (-1)) < 1e-6:
            op_type = "C2 rotation"
        elif abs(trace - 0) < 1e-6:
            op_type = "C3 rotation"
        elif abs(trace - 1) < 1e-6:
            op_type = "C4 rotation"
        elif abs(trace - (1 + np.sqrt(3))) < 1e-6:
            op_type = "C6 rotation"
        else:
            op_type = f"Rotation (tr={trace:.2f})"
    else:
        if abs(trace - 1) < 1e-6:
            op_type = "Mirror (sigma)"
        else:
            op_type = f"Improper rotation (tr={trace:.2f})"

    has_translation = not np.allclose(trans, 0, atol=1e-6)
    if has_translation:
        op_type += " + translation"

    print(f"\nOperation {i+1}: {op_type}")
    print(f"  Rotation matrix:       Translation:")
    for j in range(3):
        print(f"  [{rot[j,0]:7.3f} {rot[j,1]:7.3f} {rot[j,2]:7.3f}]   "
              f"[{trans[j]:7.4f}]")

# ── Cartesian symmetry operations ────────────────────────────────
print("\n=== Symmetry Operations (Cartesian) ===")
symmops_cart = sga.get_symmetry_operations(cartesian=True)
print(f"Number of operations: {len(symmops_cart)}")
for i, op in enumerate(symmops_cart[:5]):  # Show first 5
    print(f"\nOperation {i+1}:")
    print(f"  Rotation (Cartesian): {op.rotation_matrix.tolist()}")

# ── Point group operations ────────────────────────────────────────
print("\n=== Point Group Operations ===")
pg_ops = sga.get_point_group_operations()
print(f"Point group: {sga.get_point_group_symbol()}")
print(f"Order (number of operations): {len(pg_ops)}")
```

#### Step A5: Check symmetry of relaxed structures

```python
#!/usr/bin/env python3
"""
Compare symmetry before and after relaxation.
Checks if relaxation broke symmetry (common issue with DFT).
Equivalent to VASPKIT function 607.
"""
import numpy as np
from pymatgen.core.structure import Structure
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
from pymatgen.analysis.structure_matcher import StructureMatcher

# ── Load original and relaxed structures ──────────────────────────
original = Structure.from_file("POSCAR_original")   # Before relaxation
relaxed = Structure.from_file("CONTCAR")             # After relaxation (or relaxed.cif)

print("=== Original Structure ===")
sga_orig = SpacegroupAnalyzer(original, symprec=0.01)
print(f"  Space group: {sga_orig.get_space_group_symbol()} (#{sga_orig.get_space_group_number()})")
print(f"  Point group: {sga_orig.get_point_group_symbol()}")
print(f"  Crystal system: {sga_orig.get_crystal_system()}")
print(f"  Symmetry operations: {len(sga_orig.get_symmetry_operations())}")
print(f"  Lattice: a={original.lattice.a:.4f}, b={original.lattice.b:.4f}, "
      f"c={original.lattice.c:.4f}")

print("\n=== Relaxed Structure ===")
sga_relax = SpacegroupAnalyzer(relaxed, symprec=0.01)
print(f"  Space group: {sga_relax.get_space_group_symbol()} (#{sga_relax.get_space_group_number()})")
print(f"  Point group: {sga_relax.get_point_group_symbol()}")
print(f"  Crystal system: {sga_relax.get_crystal_system()}")
print(f"  Symmetry operations: {len(sga_relax.get_symmetry_operations())}")
print(f"  Lattice: a={relaxed.lattice.a:.4f}, b={relaxed.lattice.b:.4f}, "
      f"c={relaxed.lattice.c:.4f}")

# ── Check if symmetry was preserved ──────────────────────────────
orig_sg = sga_orig.get_space_group_number()
relax_sg = sga_relax.get_space_group_number()

if orig_sg == relax_sg:
    print("\n[OK] Symmetry PRESERVED after relaxation.")
else:
    print(f"\n[WARNING] Symmetry CHANGED: {sga_orig.get_space_group_symbol()} -> "
          f"{sga_relax.get_space_group_symbol()}")

    # Try with looser tolerance
    for tol in [0.05, 0.1, 0.2, 0.5]:
        sga_loose = SpacegroupAnalyzer(relaxed, symprec=tol)
        if sga_loose.get_space_group_number() == orig_sg:
            print(f"  Original symmetry recovered at symprec = {tol}")
            break
    else:
        print("  Symmetry NOT recoverable even at symprec = 0.5")
        print("  This may indicate a genuine symmetry-breaking distortion.")

# ── Quantify structural changes ──────────────────────────────────
print("\n=== Structural Changes ===")

# Lattice parameter changes
print("Lattice parameter changes:")
for param, orig_val, relax_val in zip(
    ["a", "b", "c", "alpha", "beta", "gamma"],
    original.lattice.abc + original.lattice.angles,
    relaxed.lattice.abc + relaxed.lattice.angles
):
    change = relax_val - orig_val
    pct = 100 * change / orig_val if orig_val != 0 else 0
    print(f"  {param}: {orig_val:.4f} -> {relax_val:.4f} "
          f"(change: {change:+.4f}, {pct:+.2f}%)")

# Volume change
vol_change = (relaxed.volume - original.volume) / original.volume * 100
print(f"\nVolume change: {original.volume:.4f} -> {relaxed.volume:.4f} "
      f"({vol_change:+.2f}%)")

# Maximum atomic displacement
matcher = StructureMatcher(ltol=0.3, stol=0.3, angle_tol=5)
if matcher.fit(original, relaxed):
    rms_disp = matcher.get_rms_dist(original, relaxed)
    if rms_disp:
        print(f"\nRMS displacement: {rms_disp[0]:.4f} A (normalized), "
              f"{rms_disp[1]:.4f} A (max)")
else:
    print("\nStructures are too different for site matching.")

# ── Symmetrize relaxed structure ──────────────────────────────────
print("\n=== Symmetrization ===")
sym_relaxed = sga_relax.get_refined_structure()
print(f"Symmetrized relaxed structure:")
print(f"  Space group: {SpacegroupAnalyzer(sym_relaxed, symprec=0.01).get_space_group_symbol()}")
sym_relaxed.to("POSCAR_symmetrized", fmt="poscar")
sym_relaxed.to("symmetrized.cif", fmt="cif")
print(f"  Saved to: POSCAR_symmetrized, symmetrized.cif")
```

#### Step A6: Molecular point group analysis

```python
#!/usr/bin/env python3
"""
Determine the point group symmetry of a molecule or cluster.
Equivalent to VASPKIT function 609.
"""
import numpy as np
from pymatgen.core.structure import Molecule
from pymatgen.symmetry.analyzer import PointGroupAnalyzer

# ── Method 1: Load molecule from file ────────────────────────────
# molecule = Molecule.from_file("molecule.xyz")

# ── Method 2: Build molecule programmatically ────────────────────
# Water molecule (C2v)
water = Molecule(
    species=["O", "H", "H"],
    coords=[
        [0.000, 0.000, 0.117],
        [0.000, 0.757, -0.469],
        [0.000, -0.757, -0.469],
    ]
)

# Methane (Td)
methane = Molecule(
    species=["C", "H", "H", "H", "H"],
    coords=[
        [0.000, 0.000, 0.000],
        [0.629, 0.629, 0.629],
        [-0.629, -0.629, 0.629],
        [-0.629, 0.629, -0.629],
        [0.629, -0.629, -0.629],
    ]
)

# Benzene (D6h)
import math
benzene_species = ["C"] * 6 + ["H"] * 6
benzene_coords = []
r_CC = 1.397
r_CH = 1.084
for i in range(6):
    angle = i * 60 * math.pi / 180
    benzene_coords.append([r_CC * math.cos(angle), r_CC * math.sin(angle), 0.0])
for i in range(6):
    angle = i * 60 * math.pi / 180
    r = r_CC + r_CH
    benzene_coords.append([r * math.cos(angle), r * math.sin(angle), 0.0])
benzene = Molecule(benzene_species, benzene_coords)

# ── Analyze point groups ──────────────────────────────────────────
molecules = {"Water": water, "Methane": methane, "Benzene": benzene}

for name, mol in molecules.items():
    pga = PointGroupAnalyzer(mol, tolerance=0.3, eigen_tolerance=0.01)

    print(f"\n=== {name} ===")
    print(f"  Formula: {mol.composition.reduced_formula}")
    print(f"  Point group (Schoenflies): {pga.sch_symbol}")
    print(f"  Number of symmetry operations: {len(pga.symmops)}")

    # List symmetry elements
    print(f"  Symmetry elements:")
    # Identify key elements
    has_inversion = any(
        np.allclose(op.rotation_matrix, -np.eye(3), atol=0.1)
        for op in pga.symmops
    )
    print(f"    Inversion center: {'Yes' if has_inversion else 'No'}")

    # Rotation axes
    n_proper = sum(1 for op in pga.symmops
                   if abs(np.linalg.det(op.rotation_matrix) - 1) < 0.1
                   and not np.allclose(op.rotation_matrix, np.eye(3), atol=0.1))
    n_improper = sum(1 for op in pga.symmops
                     if abs(np.linalg.det(op.rotation_matrix) + 1) < 0.1
                     and not np.allclose(op.rotation_matrix, -np.eye(3), atol=0.1))
    print(f"    Proper rotations (excluding E): {n_proper}")
    print(f"    Improper rotations (excluding i): {n_improper}")

# ── Extract molecule from crystal for molecular symmetry ──────────
# If you have a molecular crystal, extract a molecule first:
# from pymatgen.core.structure import Structure
# structure = Structure.from_file("molecular_crystal.cif")
# # Get connected components (molecules)
# from pymatgen.analysis.graphs import StructureGraph
# from pymatgen.analysis.local_env import CrystalNN
# sg = StructureGraph.with_local_env_strategy(structure, CrystalNN())
# molecules = sg.get_subgraphs_as_molecules()
# for i, mol in enumerate(molecules):
#     pga = PointGroupAnalyzer(mol)
#     print(f"Molecule {i}: {pga.sch_symbol}")
```

#### Step A7: Comprehensive symmetry report

```python
#!/usr/bin/env python3
"""
Generate a comprehensive symmetry report for a crystal structure.
Combines all symmetry analysis tools into a single report.
"""
import json
import numpy as np
import spglib
from pymatgen.core.structure import Structure
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
from pymatgen.symmetry.bandstructure import HighSymmKpath

# ── Load structure ────────────────────────────────────────────────
structure = Structure.from_file("structure.cif")
sga = SpacegroupAnalyzer(structure, symprec=0.01)

# ── Build report ──────────────────────────────────────────────────
report = {}

# Basic info
report["formula"] = str(structure.composition.reduced_formula)
report["n_atoms"] = len(structure)
report["volume_A3"] = round(structure.volume, 4)

# Lattice
report["lattice"] = {
    "a": round(structure.lattice.a, 6),
    "b": round(structure.lattice.b, 6),
    "c": round(structure.lattice.c, 6),
    "alpha": round(structure.lattice.alpha, 4),
    "beta": round(structure.lattice.beta, 4),
    "gamma": round(structure.lattice.gamma, 4),
}

# Symmetry
report["space_group"] = {
    "symbol": sga.get_space_group_symbol(),
    "number": sga.get_space_group_number(),
    "hall": sga.get_hall(),
    "point_group": sga.get_point_group_symbol(),
    "crystal_system": sga.get_crystal_system(),
    "lattice_type": sga.get_lattice_type(),
    "n_symmetry_operations": len(sga.get_symmetry_operations()),
}

# Primitive and conventional cells
primitive = sga.get_primitive_standard_structure()
conventional = sga.get_conventional_standard_structure()

report["primitive_cell"] = {
    "n_atoms": len(primitive),
    "volume_A3": round(primitive.volume, 4),
    "a": round(primitive.lattice.a, 6),
    "b": round(primitive.lattice.b, 6),
    "c": round(primitive.lattice.c, 6),
}

report["conventional_cell"] = {
    "n_atoms": len(conventional),
    "volume_A3": round(conventional.volume, 4),
    "a": round(conventional.lattice.a, 6),
    "b": round(conventional.lattice.b, 6),
    "c": round(conventional.lattice.c, 6),
}

# Wyckoff positions
sym_struct = sga.get_symmetrized_structure()
wyckoff_groups = []
for group_idx, group in enumerate(sym_struct.equivalent_indices):
    site = sym_struct[group[0]]
    wyckoff_groups.append({
        "element": str(site.specie),
        "wyckoff_letter": sym_struct.wyckoff_symbols[group[0]],
        "multiplicity": len(group),
        "representative_position": [round(x, 6) for x in site.frac_coords.tolist()],
        "site_indices": list(group),
    })
report["wyckoff_positions"] = wyckoff_groups

# High-symmetry k-path
kpath = HighSymmKpath(structure)
report["high_symmetry_kpath"] = {
    "path": kpath.kpath["path"],
    "kpoints": {k: [round(x, 6) for x in v] for k, v in kpath.kpath["kpoints"].items()},
}

# spglib detailed info
lattice = structure.lattice.matrix
positions = structure.frac_coords
numbers = [site.specie.Z for site in structure]
cell = (lattice, positions, numbers)
dataset = spglib.get_symmetry_dataset(cell, symprec=0.01)

report["spglib"] = {
    "international": dataset["international"],
    "hall_number": int(dataset["hall_number"]),
    "transformation_matrix": dataset["transformation_matrix"].tolist(),
    "origin_shift": dataset["origin_shift"].tolist(),
    "wyckoff_letters": dataset["wyckoffs"],
    "equivalent_atoms": dataset["equivalent_atoms"].tolist(),
}

# ── Print report ──────────────────────────────────────────────────
print("=" * 70)
print(f"SYMMETRY ANALYSIS REPORT: {report['formula']}")
print("=" * 70)
print(f"Formula:           {report['formula']}")
print(f"Atoms:             {report['n_atoms']}")
print(f"Volume:            {report['volume_A3']} A^3")
print(f"Space group:       {report['space_group']['symbol']} "
      f"(#{report['space_group']['number']})")
print(f"Point group:       {report['space_group']['point_group']}")
print(f"Crystal system:    {report['space_group']['crystal_system']}")
print(f"Lattice type:      {report['space_group']['lattice_type']}")
print(f"Symmetry ops:      {report['space_group']['n_symmetry_operations']}")

print(f"\nLattice parameters:")
L = report["lattice"]
print(f"  a = {L['a']:.6f} A    alpha = {L['alpha']:.4f} deg")
print(f"  b = {L['b']:.6f} A    beta  = {L['beta']:.4f} deg")
print(f"  c = {L['c']:.6f} A    gamma = {L['gamma']:.4f} deg")

print(f"\nPrimitive cell:    {report['primitive_cell']['n_atoms']} atoms, "
      f"{report['primitive_cell']['volume_A3']} A^3")
print(f"Conventional cell: {report['conventional_cell']['n_atoms']} atoms, "
      f"{report['conventional_cell']['volume_A3']} A^3")

print(f"\nWyckoff positions:")
for wp in report["wyckoff_positions"]:
    pos = wp["representative_position"]
    print(f"  {wp['multiplicity']}{wp['wyckoff_letter']}  {wp['element']:4s}  "
          f"({pos[0]:.6f}, {pos[1]:.6f}, {pos[2]:.6f})")

print(f"\nHigh-symmetry k-path:")
for path_segment in report["high_symmetry_kpath"]["path"]:
    path_str = " -> ".join(path_segment)
    print(f"  {path_str}")

# ── Save report to JSON ──────────────────────────────────────────
with open("symmetry_report.json", "w") as f:
    json.dump(report, f, indent=2)
print(f"\nFull report saved to: symmetry_report.json")
```

---

### Method B: QE DFT -- Symmetry in Quantum ESPRESSO

QE uses symmetry internally during SCF calculations. This section shows how to extract and verify the symmetry information from QE output.

```python
#!/usr/bin/env python3
"""
Extract symmetry information from a QE pw.x output file.
QE detects and uses crystal symmetry automatically.
Use verbosity='high' in &CONTROL to get full symmetry output.
"""
import re
import numpy as np

def parse_qe_symmetry(filename):
    """Parse symmetry information from QE output."""
    results = {
        "bravais_lattice": None,
        "crystal_system": None,
        "space_group": None,
        "point_group": None,
        "n_symmetry_ops": None,
        "n_symmetry_ops_with_frac_trans": None,
        "symmetry_operations": [],
        "inversion_symmetry": None,
    }

    with open(filename) as f:
        lines = f.readlines()

    for i, line in enumerate(lines):
        # Bravais lattice
        if "bravais-lattice index" in line:
            m = re.search(r"=\s+(\d+)", line)
            if m:
                results["bravais_lattice"] = int(m.group(1))

        # Number of symmetry operations
        if "Sym. Ops." in line or "symmetry operations" in line.lower():
            m = re.search(r"(\d+)\s+Sym", line)
            if m:
                results["n_symmetry_ops"] = int(m.group(1))

        # Point group
        if "point group" in line.lower():
            parts = line.split()
            for j, p in enumerate(parts):
                if p.lower() == "group":
                    if j + 1 < len(parts):
                        results["point_group"] = parts[j + 1].strip()
                    break

        # Inversion
        if "inversion" in line.lower():
            if "with" in line.lower() and "inversion" in line.lower():
                results["inversion_symmetry"] = True
            elif "without" in line.lower() and "inversion" in line.lower():
                results["inversion_symmetry"] = False

    return results

# ── Parse and display ─────────────────────────────────────────────
results = parse_qe_symmetry("scf.out")

print("=== QE Symmetry Analysis ===")
for key, val in results.items():
    if val is not None and key != "symmetry_operations":
        print(f"  {key}: {val}")

# ── Generate QE input to force symmetry analysis ──────────────────
print("\n=== Tip: Force symmetry output in QE ===")
print("Add these to your QE input for full symmetry info:")
print("  &CONTROL")
print("    verbosity = 'high'")
print("  /")
print("  &SYSTEM")
print("    nosym = .false.    ! Use symmetry (default)")
print("    noinv = .false.    ! Use inversion (default)")
print("  /")
print("\nTo DISABLE symmetry (e.g., for NEB, MD):")
print("    nosym = .true.")
print("    noinv = .true.")
```

---

### Method C: VASP -- Symmetry Analysis of VASP Structures

```python
#!/usr/bin/env python3
"""
Analyze symmetry of VASP structures (POSCAR/CONTCAR).
Generate symmetry report and check symmetry preservation.
"""
import numpy as np
from pymatgen.core.structure import Structure
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
from pymatgen.io.vasp import Poscar, Incar

# ── Load VASP structure ──────────────────────────────────────────
structure = Structure.from_file("POSCAR")

# Full symmetry analysis (same as Method A)
sga = SpacegroupAnalyzer(structure, symprec=0.01)
print(f"Space group: {sga.get_space_group_symbol()} (#{sga.get_space_group_number()})")
print(f"Point group: {sga.get_point_group_symbol()}")
print(f"Crystal system: {sga.get_crystal_system()}")
print(f"Number of symmetry operations: {len(sga.get_symmetry_operations())}")

# ── VASP symmetry-related INCAR settings ─────────────────────────
print("\n=== Recommended VASP INCAR settings ===")
n_ops = len(sga.get_symmetry_operations())

if n_ops > 1:
    print(f"  ISYM = 2  (use symmetry, {n_ops} operations detected)")
    print(f"  SYMPREC = 1E-5  (default symmetry tolerance)")
else:
    print(f"  ISYM = 0  (low symmetry, consider disabling)")

# For MD or NEB
print("\nFor MD or NEB calculations:")
print("  ISYM = 0  (always disable symmetry)")

# ── Generate standardized POSCAR ──────────────────────────────────
# Write POSCAR using conventional cell
conventional = sga.get_conventional_standard_structure()
poscar_conv = Poscar(conventional,
                     comment=f"{sga.get_space_group_symbol()} conventional cell")
poscar_conv.write_file("POSCAR_conventional")
print(f"\nConventional POSCAR: {len(conventional)} atoms")

# Write POSCAR using primitive cell
primitive = sga.get_primitive_standard_structure()
poscar_prim = Poscar(primitive,
                     comment=f"{sga.get_space_group_symbol()} primitive cell")
poscar_prim.write_file("POSCAR_primitive")
print(f"Primitive POSCAR: {len(primitive)} atoms")

# ── Check symmetry of CONTCAR (relaxed structure) ────────────────
# import os
# if os.path.exists("CONTCAR"):
#     relaxed = Structure.from_file("CONTCAR")
#     sga_r = SpacegroupAnalyzer(relaxed, symprec=0.01)
#     print(f"\nRelaxed (CONTCAR):")
#     print(f"  Space group: {sga_r.get_space_group_symbol()}")
#     orig_sg = sga.get_space_group_number()
#     relax_sg = sga_r.get_space_group_number()
#     if orig_sg == relax_sg:
#         print("  Symmetry PRESERVED")
#     else:
#         print(f"  Symmetry CHANGED: {orig_sg} -> {relax_sg}")
```

---

## Key Parameters

| Parameter | Typical Value | Notes |
|---|---|---|
| `symprec` | 0.01 A | Tolerance for atomic position matching. Larger values find higher symmetry. |
| `angle_tolerance` | 5 degrees | Tolerance for lattice angle matching. |
| `primitive_matrix` | Auto | Transformation from input to primitive cell. |
| `nosym` (QE) | `.false.` | Set `.true.` to disable symmetry in QE (needed for NEB, MD). |
| `ISYM` (VASP) | 2 | 0=no symmetry, 1=use symmetry (US-PP), 2=use symmetry (PAW). |
| `SYMPREC` (VASP) | 1E-5 | Symmetry tolerance in VASP. |

## Interpreting Results

- **Space group**: Reported in Hermann-Mauguin (international) notation. The number (1-230) uniquely identifies the space group.
- **Point group**: Crystal class. Determines which physical properties are allowed by symmetry (e.g., piezoelectricity requires a non-centrosymmetric point group).
- **Crystal system**: One of triclinic, monoclinic, orthorhombic, tetragonal, trigonal, hexagonal, cubic. Determines the number of independent lattice parameters.
- **Wyckoff positions**: Each atom sits on a specific Wyckoff site with a multiplicity and letter. Atoms on the same Wyckoff site are symmetrically equivalent.
- **Symmetry tolerance**: If the detected space group changes significantly with symprec, the structure may be slightly distorted from ideal symmetry. Use the result at a physically reasonable tolerance (0.01-0.1 A).
- **Primitive vs conventional**: The primitive cell has the minimum number of atoms; the conventional cell follows International Tables conventions and is more intuitive for some space groups (e.g., BCC, FCC).

## Common Issues

| Problem | Solution |
|---|---|
| Wrong space group detected | Adjust `symprec`. Too tight (0.001) misses near-symmetric atoms; too loose (0.5) finds false symmetry. |
| Symmetry breaks during relaxation | Use `nosym=.true.` in QE or `ISYM=0` in VASP if intentional. If unintentional, tighten convergence criteria. |
| Wyckoff sites differ from literature | Check if you have the primitive vs conventional cell. Convert to the standard setting. |
| spglib and pymatgen give different results | Both use spglib internally, but with possibly different default tolerances. Set `symprec` explicitly. |
| Molecular crystal shows P1 | Molecular crystals may have subtle symmetry. Increase `symprec` to 0.1 or use `angle_tolerance=10`. |
| Too many symmetry operations | This is correct for high-symmetry structures (e.g., cubic Fm-3m has 192 operations). |
| Relaxed structure has lower symmetry | May be a genuine Jahn-Teller or Peierls distortion. Symmetrize and re-check with `get_refined_structure()`. |
