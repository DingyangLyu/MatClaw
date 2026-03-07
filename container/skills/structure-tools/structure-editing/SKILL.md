# Structure Editing

## When to Use

- You need to build a supercell from a unit cell for MD, phonon, or defect calculations.
- You want to fix (freeze) certain atoms during relaxation (e.g., bottom layers of a slab).
- You need to substitute one element for another (e.g., doping, alloying).
- You need to delete specific atoms from a structure (e.g., creating vacancies).
- You need to move or displace atoms (e.g., breaking symmetry for phonon calculations).
- You want to redefine the lattice vectors (e.g., rotate the cell, create a non-orthogonal supercell).
- You need to convert between fractional and Cartesian coordinates.
- You need to sort atoms by element type, z-coordinate, or other criteria (VASPKIT 400-415 equivalent).

## Method Selection

| Criterion | ASE + pymatgen (Python) | QE | VASP |
|---|---|---|---|
| Supercell generation | `structure * (n1,n2,n3)` or `make_supercell()` | Not directly (pre-process) | Not directly (pre-process) |
| Fix atoms | ASE `FixAtoms` constraint | QE: if_pos flags (0/1) | VASP: Selective dynamics in POSCAR |
| Substitute atoms | `structure.replace()` or ASE indexing | Edit input manually | Edit POSCAR manually |
| Delete atoms | `structure.remove_sites()` | Edit input manually | Edit POSCAR manually |
| Redefine lattice | pymatgen `Structure.make_supercell()` with matrix | N/A | N/A |
| Sort atoms | pymatgen `structure.sort()` | N/A | N/A |
| Coordinate conversion | pymatgen/ASE built-in | N/A | N/A |
| Available now | Yes | Yes (input prep) | Input prep only |

## Prerequisites

- A crystal structure in any common format (CIF, POSCAR, XYZ).
- Python packages: `pymatgen`, `ase`, `numpy`, `spglib` (pre-installed).

---

## Detailed Steps

### Method A: ASE + pymatgen -- Structure Editing Operations

#### Step A1: Build supercells

```python
#!/usr/bin/env python3
"""
Build supercells from a unit cell using pymatgen and ASE.
Covers simple diagonal supercells, non-diagonal (transformation matrix) supercells,
and minimum-image supercells for a given cutoff radius.
Equivalent to VASPKIT function 400.
"""
import numpy as np
from pymatgen.core.structure import Structure, Lattice
from pymatgen.io.vasp import Poscar
from pymatgen.io.ase import AseAtomsAdaptor
from ase.io import write as ase_write
from ase.build import make_supercell as ase_make_supercell

# ── Load structure ────────────────────────────────────────────────
structure = Structure.from_file("structure.cif")
print(f"Unit cell: {structure.composition.reduced_formula}, {len(structure)} atoms")
print(f"Lattice: a={structure.lattice.a:.4f}, b={structure.lattice.b:.4f}, "
      f"c={structure.lattice.c:.4f}")

# ── Method 1: Simple diagonal supercell (pymatgen) ────────────────
# Create a 2x2x2 supercell
supercell_222 = structure.copy()
supercell_222.make_supercell([2, 2, 2])
print(f"\n2x2x2 supercell: {len(supercell_222)} atoms, V={supercell_222.volume:.2f} A^3")
supercell_222.to("POSCAR_222", fmt="poscar")

# Create a 3x3x1 supercell (common for slab models)
supercell_331 = structure.copy()
supercell_331.make_supercell([3, 3, 1])
print(f"3x3x1 supercell: {len(supercell_331)} atoms, V={supercell_331.volume:.2f} A^3")
supercell_331.to("POSCAR_331", fmt="poscar")

# ── Method 2: Non-diagonal supercell matrix (pymatgen) ────────────
# Transformation matrix: new_lattice = M * old_lattice
# Example: create a sqrt(2) x sqrt(2) x 1 supercell rotated 45 degrees
M = [[1, 1, 0],
     [-1, 1, 0],
     [0, 0, 1]]
supercell_rot = structure.copy()
supercell_rot.make_supercell(M)
print(f"\nsqrt(2)xsqrt(2)x1 rotated: {len(supercell_rot)} atoms")
print(f"  New lattice: a={supercell_rot.lattice.a:.4f}, b={supercell_rot.lattice.b:.4f}")
supercell_rot.to("POSCAR_rot", fmt="poscar")

# ── Method 3: Non-diagonal supercell with ASE ────────────────────
# ASE make_supercell takes a 3x3 transformation matrix
atoms = AseAtomsAdaptor.get_atoms(structure)
P = np.array([[2, 0, 0],
              [0, 2, 0],
              [0, 0, 2]])
supercell_ase = ase_make_supercell(atoms, P)
print(f"\n2x2x2 supercell (ASE): {len(supercell_ase)} atoms")
ase_write("supercell_ase.vasp", supercell_ase, format="vasp")

# ── Method 4: Minimum supercell for a given cutoff radius ─────────
def min_supercell_for_cutoff(structure, cutoff):
    """
    Find the smallest supercell where the minimum image distance
    exceeds the given cutoff. Useful for defect calculations.
    """
    abc = structure.lattice.abc
    scaling = [max(1, int(np.ceil(2 * cutoff / a))) for a in abc]
    return scaling

cutoff = 10.0  # Angstrom
scaling = min_supercell_for_cutoff(structure, cutoff)
print(f"\nMin supercell for {cutoff} A cutoff: {scaling[0]}x{scaling[1]}x{scaling[2]}")
supercell_min = structure.copy()
supercell_min.make_supercell(scaling)
print(f"  Atoms: {len(supercell_min)}, min lattice param: "
      f"{min(supercell_min.lattice.abc):.2f} A")
supercell_min.to("POSCAR_mincutoff", fmt="poscar")
```

#### Step A2: Fix (freeze) atoms

```python
#!/usr/bin/env python3
"""
Fix atoms during relaxation. Common use cases:
- Fix bottom layers of a slab model.
- Fix certain atom types.
- Fix atoms below a z-coordinate threshold.
Equivalent to VASPKIT function 409.

Outputs POSCAR with selective dynamics for VASP,
and QE input with if_pos flags.
"""
import numpy as np
from pymatgen.core.structure import Structure
from pymatgen.io.vasp import Poscar
from pymatgen.io.ase import AseAtomsAdaptor
from ase.constraints import FixAtoms
from ase.io import write as ase_write

# ── Load structure ────────────────────────────────────────────────
structure = Structure.from_file("POSCAR")
print(f"Structure: {structure.composition.reduced_formula}, {len(structure)} atoms")

# ── Method 1: Fix atoms by z-coordinate (fractional) ─────────────
# Fix atoms in the bottom half (z < 0.5 in fractional coordinates)
z_threshold = 0.5
fixed_indices = []
for i, site in enumerate(structure):
    if site.frac_coords[2] < z_threshold:
        fixed_indices.append(i)

print(f"Fixing {len(fixed_indices)} atoms with z_frac < {z_threshold}")

# ── Method 2: Fix atoms by element type ──────────────────────────
# fixed_indices = [i for i, site in enumerate(structure)
#                  if str(site.specie) in ["O", "N"]]

# ── Method 3: Fix atoms by distance from bottom ──────────────────
# cart_z = [site.coords[2] for site in structure]
# z_min = min(cart_z)
# fix_height = 3.0  # Angstrom from bottom
# fixed_indices = [i for i, site in enumerate(structure)
#                  if site.coords[2] - z_min < fix_height]

# ── Generate VASP POSCAR with Selective Dynamics ──────────────────
# selective_dynamics: list of [bool, bool, bool] per atom
# True = free to move, False = fixed
sd = []
for i in range(len(structure)):
    if i in fixed_indices:
        sd.append([False, False, False])  # Fixed
    else:
        sd.append([True, True, True])     # Free

poscar = Poscar(structure, selective_dynamics=sd,
                comment=f"{structure.composition.reduced_formula} - selective dynamics")
poscar.write_file("POSCAR_SD")
print(f"POSCAR with selective dynamics written to POSCAR_SD")

# ── Generate QE input with if_pos flags ───────────────────────────
def write_qe_with_fixed_atoms(structure, fixed_indices, filename="pw_fixed.in"):
    """Write QE ATOMIC_POSITIONS with if_pos flags (0=fixed, 1=free)."""
    lines = []
    lines.append("ATOMIC_POSITIONS crystal")
    for i, site in enumerate(structure):
        fc = site.frac_coords
        if i in fixed_indices:
            # if_pos = 0 0 0 means fixed
            lines.append(f"  {str(site.specie):4s} {fc[0]:16.10f} {fc[1]:16.10f} "
                         f"{fc[2]:16.10f}  0 0 0")
        else:
            lines.append(f"  {str(site.specie):4s} {fc[0]:16.10f} {fc[1]:16.10f} "
                         f"{fc[2]:16.10f}  1 1 1")

    with open(filename, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"QE atomic positions with if_pos written to {filename}")

write_qe_with_fixed_atoms(structure, fixed_indices)

# ── ASE approach (for MACE or other calculators) ──────────────────
atoms = AseAtomsAdaptor.get_atoms(structure)
constraint = FixAtoms(indices=fixed_indices)
atoms.set_constraint(constraint)

# Verify constraints
print(f"\nASE constraints: {atoms.constraints}")
print(f"Fixed atoms: {len(fixed_indices)}/{len(atoms)}")

# Write with constraints preserved (extxyz format stores constraints)
ase_write("structure_fixed.xyz", atoms, format="extxyz")
print("ASE structure with constraints written to structure_fixed.xyz")
```

#### Step A3: Substitute atoms (element replacement / doping)

```python
#!/usr/bin/env python3
"""
Substitute atoms in a crystal structure.
Use cases: doping, alloying, cation/anion exchange.
Equivalent to VASPKIT function 404.
"""
import numpy as np
from pymatgen.core.structure import Structure
from pymatgen.io.vasp import Poscar
from pymatgen.transformations.site_transformations import (
    ReplaceSiteSpeciesTransformation,
    RemoveSitesTransformation,
)
from pymatgen.transformations.standard_transformations import (
    SubstitutionTransformation,
)

# ── Load structure ────────────────────────────────────────────────
structure = Structure.from_file("structure.cif")
print(f"Original: {structure.composition.reduced_formula}, {len(structure)} atoms")

# ── Method 1: Replace ALL atoms of one element ───────────────────
# Replace all Si with Ge (full substitution)
sub_all = structure.copy()
sub_trans = SubstitutionTransformation({"Si": "Ge"})
sub_all = sub_trans.apply_transformation(sub_all)
print(f"\nAll Si -> Ge: {sub_all.composition.reduced_formula}")
sub_all.to("POSCAR_sub_all", fmt="poscar")

# ── Method 2: Replace a SINGLE specific atom ─────────────────────
# Replace atom at site index 0
sub_single = structure.copy()
sub_single.replace(0, "Ge")
print(f"Site 0 Si -> Ge: {sub_single.composition}")
sub_single.to("POSCAR_sub_single", fmt="poscar")

# ── Method 3: Replace atoms by fractional coordinates ─────────────
# Find the atom closest to a specific position and replace it
target_frac = [0.25, 0.25, 0.25]
sub_pos = structure.copy()
distances = [np.linalg.norm(site.frac_coords - target_frac) for site in sub_pos]
closest_idx = np.argmin(distances)
print(f"\nClosest atom to {target_frac}: site {closest_idx} "
      f"({sub_pos[closest_idx].specie} at {sub_pos[closest_idx].frac_coords})")
sub_pos.replace(closest_idx, "P")
print(f"After substitution: {sub_pos.composition}")
sub_pos.to("POSCAR_sub_pos", fmt="poscar")

# ── Method 4: Random doping at a specified concentration ──────────
def random_doping(structure, host_element, dopant_element, fraction, seed=42):
    """
    Randomly substitute a fraction of host_element with dopant_element.

    Parameters
    ----------
    structure : pymatgen Structure
    host_element : str
        Element to replace (e.g., "Si").
    dopant_element : str
        Replacement element (e.g., "B").
    fraction : float
        Fraction of host atoms to replace (0 to 1).
    seed : int
        Random seed for reproducibility.

    Returns
    -------
    pymatgen Structure with substitutions applied.
    """
    rng = np.random.default_rng(seed)
    doped = structure.copy()

    host_indices = [i for i, site in enumerate(doped)
                    if str(site.specie) == host_element]

    n_to_replace = max(1, int(round(len(host_indices) * fraction)))
    replace_indices = rng.choice(host_indices, size=n_to_replace, replace=False)

    for idx in sorted(replace_indices, reverse=True):
        doped.replace(idx, dopant_element)

    return doped

# Example: 12.5% B-doped Si (1 out of 8 Si atoms in conventional cell)
supercell = structure.copy()
supercell.make_supercell([2, 2, 2])
doped = random_doping(supercell, "Si", "B", fraction=0.125)
print(f"\nB-doped Si: {doped.composition}")
doped.to("POSCAR_doped", fmt="poscar")

# ── Method 5: Ordered substitution using pymatgen transformations ─
# Create all symmetrically distinct substituted structures
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
from pymatgen.analysis.structure_matcher import StructureMatcher
from itertools import combinations

def enumerate_substitutions(structure, host_element, dopant_element, n_sub):
    """
    Enumerate all symmetrically distinct structures with n_sub substitutions.
    Uses symmetry to identify unique configurations.
    """
    host_indices = [i for i, site in enumerate(structure)
                    if str(site.specie) == host_element]

    if n_sub > len(host_indices):
        raise ValueError(f"Cannot substitute {n_sub} atoms, only {len(host_indices)} "
                         f"{host_element} atoms available.")

    matcher = StructureMatcher(ltol=0.1, stol=0.1, angle_tol=2)
    unique_structures = []

    for combo in combinations(host_indices, n_sub):
        s = structure.copy()
        for idx in combo:
            s.replace(idx, dopant_element)

        # Check if this is a new unique structure
        is_duplicate = False
        for existing in unique_structures:
            if matcher.fit(s, existing):
                is_duplicate = True
                break

        if not is_duplicate:
            unique_structures.append(s)

    return unique_structures

# Example: Find all unique single-substitution configurations
# (For large supercells, limit n_sub to keep enumeration tractable)
unique = enumerate_substitutions(structure, "Si", "Ge", 1)
print(f"\nUnique single Ge substitutions in unit cell: {len(unique)}")
for i, s in enumerate(unique):
    s.to(f"POSCAR_unique_{i}", fmt="poscar")
    print(f"  Config {i}: {s.composition}")
```

#### Step A4: Delete atoms (create vacancies)

```python
#!/usr/bin/env python3
"""
Delete atoms from a crystal structure.
Use cases: vacancy creation, surface termination removal.
Equivalent to VASPKIT function 405.
"""
import numpy as np
from pymatgen.core.structure import Structure
from pymatgen.io.vasp import Poscar

# ── Load structure ────────────────────────────────────────────────
structure = Structure.from_file("structure.cif")
print(f"Original: {structure.composition.reduced_formula}, {len(structure)} atoms")

# ── Method 1: Remove atom at a specific site index ────────────────
vacancy = structure.copy()
removed_site = vacancy[0]
vacancy.remove_sites([0])
print(f"\nRemoved site 0 ({removed_site.specie} at {removed_site.frac_coords})")
print(f"After removal: {vacancy.composition}, {len(vacancy)} atoms")
vacancy.to("POSCAR_vacancy", fmt="poscar")

# ── Method 2: Remove atom closest to a given position ────────────
target_frac = [0.0, 0.0, 0.0]
vac2 = structure.copy()
distances = [np.linalg.norm(site.frac_coords - target_frac) for site in vac2]
idx = np.argmin(distances)
print(f"\nRemoving atom closest to {target_frac}: "
      f"site {idx} ({vac2[idx].specie} at {vac2[idx].frac_coords})")
vac2.remove_sites([idx])
vac2.to("POSCAR_vacancy_pos", fmt="poscar")

# ── Method 3: Remove all atoms of a specific element ─────────────
vac3 = structure.copy()
remove_element = "O"
indices_to_remove = [i for i, site in enumerate(vac3)
                     if str(site.specie) == remove_element]
print(f"\nRemoving all {remove_element} atoms: {len(indices_to_remove)} atoms")
vac3.remove_sites(indices_to_remove)
print(f"After removal: {vac3.composition}")
vac3.to("POSCAR_no_O", fmt="poscar")

# ── Method 4: Remove atoms outside a slab region ─────────────────
# Keep only atoms within a z-range (useful for cleaning up slabs)
vac4 = structure.copy()
z_min, z_max = 0.1, 0.9  # fractional coordinates
indices_outside = [i for i, site in enumerate(vac4)
                   if site.frac_coords[2] < z_min or site.frac_coords[2] > z_max]
print(f"\nRemoving {len(indices_outside)} atoms outside z=[{z_min},{z_max}]")
vac4.remove_sites(indices_outside)
vac4.to("POSCAR_trimmed", fmt="poscar")
```

#### Step A5: Move and displace atoms

```python
#!/usr/bin/env python3
"""
Move or displace atoms in a crystal structure.
Use cases: break symmetry, create distortions, manual adjustments.
Equivalent to VASPKIT function 406.
"""
import numpy as np
from pymatgen.core.structure import Structure
from pymatgen.io.vasp import Poscar
from pymatgen.io.ase import AseAtomsAdaptor
from ase.io import write as ase_write

# ── Load structure ────────────────────────────────────────────────
structure = Structure.from_file("structure.cif")
print(f"Original: {structure.composition.reduced_formula}, {len(structure)} atoms")

# ── Method 1: Translate a single atom (fractional coords) ────────
moved = structure.copy()
# Move atom 0 by [0.01, 0.0, 0.0] in fractional coordinates
old_pos = moved[0].frac_coords.copy()
new_pos = old_pos + np.array([0.01, 0.0, 0.0])
moved.replace(0, moved[0].specie, coords=new_pos)
print(f"Moved atom 0: {old_pos} -> {new_pos}")
moved.to("POSCAR_moved", fmt="poscar")

# ── Method 2: Translate a single atom (Cartesian displacement) ───
moved2 = structure.copy()
# Displace atom 0 by 0.1 Angstrom along x in Cartesian
displacement_cart = np.array([0.1, 0.0, 0.0])  # Angstrom
old_cart = moved2[0].coords.copy()
new_cart = old_cart + displacement_cart
new_frac = moved2.lattice.get_fractional_coords(new_cart)
moved2.replace(0, moved2[0].specie, coords=new_frac)
print(f"\nCartesian displacement: {old_cart} -> {new_cart}")
moved2.to("POSCAR_displaced", fmt="poscar")

# ── Method 3: Random perturbation of all atoms ───────────────────
# Useful for breaking symmetry before phonon calculations
def perturb_structure(structure, amplitude=0.01, seed=42):
    """
    Apply random Gaussian perturbations to all atoms.

    Parameters
    ----------
    structure : pymatgen Structure
    amplitude : float
        Standard deviation of Gaussian displacement (Angstrom).
    seed : int
        Random seed.
    """
    rng = np.random.default_rng(seed)
    perturbed = structure.copy()

    for i in range(len(perturbed)):
        displacement = rng.normal(0, amplitude, size=3)  # Cartesian, Angstrom
        old_cart = perturbed[i].coords
        new_cart = old_cart + displacement
        new_frac = perturbed.lattice.get_fractional_coords(new_cart)
        perturbed.replace(i, perturbed[i].specie, coords=new_frac)

    return perturbed

perturbed = perturb_structure(structure, amplitude=0.01)
print(f"\nPerturbed structure (amplitude=0.01 A): {perturbed.composition}")
perturbed.to("POSCAR_perturbed", fmt="poscar")

# ── Method 4: Translate entire structure (shift origin) ───────────
# Shift all atoms so that a specific atom is at the origin
shifted = structure.copy()
shift_vector = -shifted[0].frac_coords  # Move atom 0 to origin
shifted.translate_sites(range(len(shifted)), shift_vector, frac_coords=True)
print(f"\nShifted structure (atom 0 at origin)")
shifted.to("POSCAR_shifted", fmt="poscar")

# ── Method 5: Apply strain/deformation ────────────────────────────
from pymatgen.analysis.elasticity import Strain, Deformation

strained = structure.copy()
# Apply 1% tensile strain along x
strain_voigt = [0.01, 0.0, 0.0, 0.0, 0.0, 0.0]
strain = Strain.from_voigt(strain_voigt)
strained.apply_strain(strain)
print(f"\n1% x-strain: a={strained.lattice.a:.4f} (was {structure.lattice.a:.4f})")
strained.to("POSCAR_strained", fmt="poscar")
```

#### Step A6: Redefine lattice vectors

```python
#!/usr/bin/env python3
"""
Redefine lattice vectors using a transformation matrix.
Use cases: create non-orthogonal cells, rotate cell, Niggli reduction.
Equivalent to VASPKIT function 401.
"""
import numpy as np
from pymatgen.core.structure import Structure, Lattice
from pymatgen.core.lattice import get_points_in_spheres
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
from pymatgen.transformations.standard_transformations import (
    ConventionalCellTransformation,
    PrimitiveCellTransformation,
)

# ── Load structure ────────────────────────────────────────────────
structure = Structure.from_file("structure.cif")
print(f"Original: {structure.composition.reduced_formula}")
print(f"Lattice:\n{structure.lattice.matrix}")

# ── Method 1: Apply transformation matrix to lattice ──────────────
# new_lattice = M @ old_lattice (rows are lattice vectors)
# Example: create a 45-degree rotated cell
M = np.array([[1, 1, 0],
              [-1, 1, 0],
              [0, 0, 1]])

transformed = structure.copy()
transformed.make_supercell(M)
print(f"\nTransformed lattice (M = [[1,1,0],[-1,1,0],[0,0,1]]):")
print(f"  a={transformed.lattice.a:.4f}, b={transformed.lattice.b:.4f}, "
      f"c={transformed.lattice.c:.4f}")
print(f"  Atoms: {len(transformed)}")
transformed.to("POSCAR_transformed", fmt="poscar")

# ── Method 2: Niggli reduction (find shortest lattice vectors) ────
niggli_lattice = structure.lattice.get_niggli_reduced_lattice()
niggli_structure = Structure(
    niggli_lattice,
    structure.species,
    structure.cart_coords,
    coords_are_cartesian=True,
)
print(f"\nNiggli-reduced lattice:")
print(f"  a={niggli_lattice.a:.4f}, b={niggli_lattice.b:.4f}, c={niggli_lattice.c:.4f}")
print(f"  alpha={niggli_lattice.alpha:.2f}, beta={niggli_lattice.beta:.2f}, "
      f"gamma={niggli_lattice.gamma:.2f}")

# ── Method 3: Get conventional standard cell ─────────────────────
conv_trans = ConventionalCellTransformation(symprec=0.01)
conventional = conv_trans.apply_transformation(structure)
print(f"\nConventional cell: {conventional.composition}, {len(conventional)} atoms")
print(f"  a={conventional.lattice.a:.4f}, b={conventional.lattice.b:.4f}, "
      f"c={conventional.lattice.c:.4f}")
conventional.to("POSCAR_conventional", fmt="poscar")

# ── Method 4: Get primitive cell ──────────────────────────────────
prim_trans = PrimitiveCellTransformation()
primitive = prim_trans.apply_transformation(structure)
print(f"\nPrimitive cell: {primitive.composition}, {len(primitive)} atoms")
print(f"  a={primitive.lattice.a:.4f}, b={primitive.lattice.b:.4f}, "
      f"c={primitive.lattice.c:.4f}")
primitive.to("POSCAR_primitive", fmt="poscar")

# ── Method 5: Create orthogonal supercell ─────────────────────────
def make_orthogonal_supercell(structure, max_atoms=200):
    """
    Find a near-orthogonal supercell by testing transformation matrices.
    Returns the smallest supercell with angles close to 90 degrees.
    """
    best = None
    best_score = float('inf')

    for i in range(-3, 4):
        for j in range(-3, 4):
            for k in range(-3, 4):
                for l in range(-3, 4):
                    for m in range(-3, 4):
                        for n in range(-3, 4):
                            M = np.array([[1, i, j],
                                          [0, 1, k],
                                          [0, 0, 1]])
                            # Quick check: keep it manageable
                            det = abs(np.linalg.det(M))
                            if det < 0.5 or det > 4:
                                continue
                            new_lat = Lattice(M @ structure.lattice.matrix)
                            angles = new_lat.angles
                            # Score: deviation from 90 degrees
                            score = sum((a - 90)**2 for a in angles)
                            natoms = int(round(det)) * len(structure)
                            if natoms > max_atoms:
                                continue
                            if score < best_score:
                                best_score = score
                                best = M

    if best is not None:
        ortho = structure.copy()
        ortho.make_supercell(best)
        return ortho
    return None

# This is a simplified approach; for complex cases, use the full algorithm
# ortho = make_orthogonal_supercell(structure)
```

#### Step A7: Sort atoms

```python
#!/usr/bin/env python3
"""
Sort atoms in a crystal structure by various criteria.
Equivalent to VASPKIT functions 407-408.
"""
import numpy as np
from pymatgen.core.structure import Structure
from pymatgen.io.vasp import Poscar

# ── Load structure ────────────────────────────────────────────────
structure = Structure.from_file("structure.cif")
print(f"Original order:")
for i, site in enumerate(structure):
    print(f"  {i}: {site.specie} at {site.frac_coords}")

# ── Method 1: Sort by element (default pymatgen behavior) ────────
sorted_by_element = structure.get_sorted_structure()
print(f"\nSorted by element (electronegativity):")
for i, site in enumerate(sorted_by_element):
    print(f"  {i}: {site.specie} at [{site.frac_coords[0]:.4f}, "
          f"{site.frac_coords[1]:.4f}, {site.frac_coords[2]:.4f}]")
sorted_by_element.to("POSCAR_sorted_element", fmt="poscar")

# ── Method 2: Sort by z-coordinate ───────────────────────────────
# Useful for slab models where you want layers ordered by height
sorted_by_z = structure.get_sorted_structure(key=lambda site: site.frac_coords[2])
print(f"\nSorted by z (fractional):")
for i, site in enumerate(sorted_by_z):
    print(f"  {i}: {site.specie} z={site.frac_coords[2]:.4f}")
sorted_by_z.to("POSCAR_sorted_z", fmt="poscar")

# ── Method 3: Sort by element, then by z-coordinate ──────────────
sorted_el_z = structure.get_sorted_structure(
    key=lambda site: (str(site.specie), site.frac_coords[2])
)
print(f"\nSorted by element then z:")
for i, site in enumerate(sorted_el_z):
    print(f"  {i}: {site.specie} z={site.frac_coords[2]:.4f}")
sorted_el_z.to("POSCAR_sorted_el_z", fmt="poscar")

# ── Method 4: Custom sort order for VASP ──────────────────────────
# VASP requires atoms grouped by species in the order listed in POTCAR.
# pymatgen's Poscar class handles this automatically.
poscar = Poscar(structure)
poscar.write_file("POSCAR_vasp_order")
print(f"\nVASP-ordered POSCAR written (species: {poscar.site_symbols})")
```

#### Step A8: Coordinate conversion

```python
#!/usr/bin/env python3
"""
Convert between fractional (direct) and Cartesian coordinates.
Equivalent to VASPKIT function 410.
"""
import numpy as np
from pymatgen.core.structure import Structure
from pymatgen.io.vasp import Poscar

# ── Load structure ────────────────────────────────────────────────
structure = Structure.from_file("structure.cif")
print(f"Structure: {structure.composition.reduced_formula}")

# ── Display both coordinate systems ──────────────────────────────
print("\n--- Fractional (Direct) Coordinates ---")
for i, site in enumerate(structure):
    fc = site.frac_coords
    print(f"  {i:3d} {str(site.specie):4s} {fc[0]:12.8f} {fc[1]:12.8f} {fc[2]:12.8f}")

print("\n--- Cartesian Coordinates (Angstrom) ---")
for i, site in enumerate(structure):
    cc = site.coords
    print(f"  {i:3d} {str(site.specie):4s} {cc[0]:12.8f} {cc[1]:12.8f} {cc[2]:12.8f}")

# ── Conversion formulas ──────────────────────────────────────────
# Cartesian = M @ Fractional  where M = lattice matrix (rows = vectors)
# Fractional = M^{-1} @ Cartesian
M = structure.lattice.matrix
M_inv = structure.lattice.inv_matrix

print(f"\nLattice matrix M (rows = a, b, c vectors):\n{M}")
print(f"\nM^-1:\n{M_inv}")

# Example conversion
frac_coords = np.array([0.25, 0.25, 0.25])
cart_coords = M.T @ frac_coords  # Note: pymatgen uses column convention
# Or equivalently:
cart_coords_pmg = structure.lattice.get_cartesian_coords(frac_coords)
frac_back = structure.lattice.get_fractional_coords(cart_coords_pmg)

print(f"\nExample conversion:")
print(f"  Fractional: {frac_coords}")
print(f"  Cartesian:  {cart_coords_pmg}")
print(f"  Back to fractional: {frac_back}")

# ── Write POSCAR in Cartesian coordinates ─────────────────────────
poscar_cart = Poscar(structure, direct=False,
                     comment=f"{structure.composition.reduced_formula} (Cartesian)")
poscar_cart.write_file("POSCAR_cartesian")
print("\nPOSCAR written in Cartesian coordinates: POSCAR_cartesian")

# ── Write POSCAR in fractional/direct coordinates ─────────────────
poscar_frac = Poscar(structure, direct=True,
                     comment=f"{structure.composition.reduced_formula} (Direct)")
poscar_frac.write_file("POSCAR_direct")
print("POSCAR written in Direct coordinates: POSCAR_direct")
```

---

### Method B: QE DFT -- Structure Editing for QE Input

Structure editing for QE is typically done in Python (pymatgen/ASE) and then the modified structure is written as a QE input file.

```python
#!/usr/bin/env python3
"""
Edit a structure and generate a QE input file with the modifications.
Combines pymatgen structure editing with QE input generation.
"""
import os
import numpy as np
from pymatgen.core.structure import Structure
from pymatgen.io.pwscf import PWInput

# ── Load and edit structure ───────────────────────────────────────
structure = Structure.from_file("structure.cif")

# Example: create a 2x2x2 supercell and substitute one atom
supercell = structure.copy()
supercell.make_supercell([2, 2, 2])
supercell.replace(0, "Ge")  # Substitute first Si with Ge

print(f"Edited structure: {supercell.composition}")
print(f"Atoms: {len(supercell)}")

# ── Set up pseudopotential map ────────────────────────────────────
PSEUDO_DIR = os.path.abspath("./pseudo")
elements = list(set(str(sp) for sp in supercell.species))
pseudo_map = {}
for el in elements:
    if os.path.exists(PSEUDO_DIR):
        for fname in os.listdir(PSEUDO_DIR):
            if fname.endswith(".UPF") and fname.lower().startswith(el.lower()):
                pseudo_map[el] = fname
                break
    if el not in pseudo_map:
        pseudo_map[el] = f"{el}.UPF"

# ── QE input with selective relaxation (if_pos) ──────────────────
# Fix bottom half of atoms
fixed_indices = [i for i, site in enumerate(supercell)
                 if site.frac_coords[2] < 0.5]

# Build QE input manually to include if_pos flags
cell = supercell.lattice.matrix
species_order = list(dict.fromkeys(str(sp) for sp in supercell.species))

lines = []
lines.append("&CONTROL")
lines.append(f"    calculation  = 'relax'")
lines.append(f"    prefix       = 'edited'")
lines.append(f"    outdir       = './tmp'")
lines.append(f"    pseudo_dir   = '{PSEUDO_DIR}'")
lines.append(f"    tprnfor      = .true.")
lines.append(f"    tstress      = .true.")
lines.append(f"    forc_conv_thr = 1.0d-4")
lines.append("/\n")

lines.append("&SYSTEM")
lines.append(f"    ibrav        = 0")
lines.append(f"    nat          = {len(supercell)}")
lines.append(f"    ntyp         = {len(species_order)}")
lines.append(f"    ecutwfc      = 50.0")
lines.append(f"    ecutrho      = 400.0")
lines.append(f"    occupations  = 'smearing'")
lines.append(f"    smearing     = 'cold'")
lines.append(f"    degauss      = 0.01")
lines.append("/\n")

lines.append("&ELECTRONS")
lines.append(f"    conv_thr     = 1.0d-8")
lines.append(f"    mixing_beta  = 0.7")
lines.append("/\n")

lines.append("&IONS")
lines.append(f"    ion_dynamics = 'bfgs'")
lines.append("/\n")

lines.append("ATOMIC_SPECIES")
for el in species_order:
    from pymatgen.core.periodic_table import Element
    mass = float(Element(el).atomic_mass)
    lines.append(f"  {el:4s} {mass:10.4f}  {pseudo_map.get(el, el + '.UPF')}")
lines.append("")

lines.append("CELL_PARAMETERS angstrom")
for row in cell:
    lines.append(f"  {row[0]:16.10f} {row[1]:16.10f} {row[2]:16.10f}")
lines.append("")

lines.append("ATOMIC_POSITIONS crystal")
for i, site in enumerate(supercell):
    fc = site.frac_coords
    if_pos = "0 0 0" if i in fixed_indices else "1 1 1"
    lines.append(f"  {str(site.specie):4s} {fc[0]:16.10f} {fc[1]:16.10f} "
                 f"{fc[2]:16.10f}  {if_pos}")
lines.append("")

lines.append("K_POINTS automatic")
# Reduce k-grid for supercell
recip = supercell.lattice.reciprocal_lattice.abc
kgrid = tuple(max(1, int(round(40 / (2 * np.pi) * rl))) for rl in recip)
lines.append(f"  {kgrid[0]} {kgrid[1]} {kgrid[2]}  0 0 0")

input_text = "\n".join(lines) + "\n"
with open("pw_edited.in", "w") as f:
    f.write(input_text)
print(f"QE input with edited structure written to pw_edited.in")
print(f"  Fixed atoms: {len(fixed_indices)}/{len(supercell)}")
```

---

### Method C: VASP -- Structure Editing for VASP Input

```python
#!/usr/bin/env python3
"""
Edit a structure and generate VASP input files with modifications.
Covers supercell generation, selective dynamics, and substitution
for VASP workflows.
"""
import numpy as np
from pymatgen.core.structure import Structure
from pymatgen.io.vasp import Poscar, Incar, Kpoints

# ── Load structure ────────────────────────────────────────────────
structure = Structure.from_file("structure.cif")

# ── Build supercell ───────────────────────────────────────────────
supercell = structure.copy()
supercell.make_supercell([2, 2, 2])
print(f"Supercell: {supercell.composition}, {len(supercell)} atoms")

# ── Selective dynamics (fix bottom layers) ────────────────────────
z_threshold = 0.5
selective_dynamics = []
for site in supercell:
    if site.frac_coords[2] < z_threshold:
        selective_dynamics.append([False, False, False])
    else:
        selective_dynamics.append([True, True, True])

n_fixed = sum(1 for sd in selective_dynamics if not sd[0])
print(f"Selective dynamics: {n_fixed} fixed, {len(supercell) - n_fixed} free")

# ── Write POSCAR with selective dynamics ──────────────────────────
poscar = Poscar(supercell, selective_dynamics=selective_dynamics,
                comment=f"{supercell.composition.reduced_formula} supercell with SD")
poscar.write_file("POSCAR")

# ── Generate matching INCAR ──────────────────────────────────────
incar_dict = {
    "ENCUT": 520,
    "PREC": "Accurate",
    "EDIFF": 1e-6,
    "NSW": 200,
    "IBRION": 2,
    "ISIF": 2,          # Fix cell shape/volume, relax ions
    "EDIFFG": -0.01,
    "ISMEAR": 0,
    "SIGMA": 0.05,
    "LREAL": "Auto",
    "NCORE": 4,
    "LWAVE": False,
    "LCHARG": False,
}
incar = Incar(incar_dict)
incar.write_file("INCAR")

# ── Generate KPOINTS (reduced for supercell) ─────────────────────
kpoints = Kpoints.automatic_density(supercell, kppa=500, force_gamma=True)
kpoints.write_file("KPOINTS")

print(f"\nVASP input files generated:")
print(f"  POSCAR: {len(supercell)} atoms with selective dynamics")
print(f"  INCAR: NSW={incar_dict['NSW']}, ISIF={incar_dict['ISIF']}")
with open("KPOINTS") as f:
    kpt_line = f.readlines()[3].strip()
    print(f"  KPOINTS: {kpt_line}")
```

---

## Key Parameters

| Operation | Key Parameter | Notes |
|---|---|---|
| Supercell size | `[n1, n2, n3]` or 3x3 matrix | Use minimum image criterion for defect calcs: 2*cutoff < min(lattice_param) |
| Selective dynamics | Per-atom `[bool, bool, bool]` | VASP: True=free. QE: if_pos 1=free, 0=fixed. |
| Doping fraction | 0.0 -- 1.0 | For ordered enumeration, keep total substitutions small (combinatorial explosion). |
| Perturbation amplitude | 0.005 -- 0.05 A | 0.01 A is typical for breaking symmetry. Larger values may cause convergence issues. |
| Strain magnitude | 0.001 -- 0.05 | Keep small for elastic response. Larger for stress-strain curves. |
| Sort key | Element, z-coordinate, custom | VASP requires atoms grouped by species. |

## Interpreting Results

- **Supercell**: Verify atom count = unit_cell_atoms * det(M). Check that no atoms overlap (minimum distance > 0.5 A).
- **Selective dynamics**: In VASP output CONTCAR, fixed atoms should remain at their original positions. In QE, check forces on fixed atoms are not printed.
- **Substitution**: After relaxation, check that the dopant site relaxes to a physically reasonable position. Large displacements may indicate an unstable configuration.
- **Sorting**: VASP POSCAR species order must match POTCAR order. pymatgen handles this automatically.

## Common Issues

| Problem | Solution |
|---|---|
| Atoms overlap after supercell | Check transformation matrix determinant > 0. Use `structure.get_all_distances()` to verify minimum distances. |
| Selective dynamics not applied in VASP | Ensure the T/F flags appear in POSCAR line 8+. Must have "Selective dynamics" on line 7. |
| QE if_pos not working | if_pos flags must follow atomic coordinates on the same line, separated by spaces. |
| Substitution creates unreasonable structure | Relax the structure after substitution. Use MACE for quick pre-relaxation. |
| Wrong number of atoms after make_supercell | `det(M)` must be a positive integer. Non-integer determinants are invalid. |
| Coordinate wrapping issues | Use `structure.frac_coords % 1.0` or `structure.get_sorted_structure()` to wrap coordinates into [0,1). |
| Atoms at cell boundaries appear doubled | Set `symprec` appropriately. Use `structure.merge_sites(tol=0.01)` to merge near-duplicate sites. |
