# Supercell Builder

## When to Use

- You need to build a supercell (NxMxL) of a crystal structure for defect, phonon, or alloy calculations.
- You want an orthogonal (cubic or orthorhombic) supercell from a non-orthogonal primitive cell (e.g., FCC, HCP, rhombohedral).
- You need a supercell with a minimum dimension in each direction (e.g., >= 10 A for defect calculations).
- You want to convert a primitive cell to a conventional cell.

## Method Selection

| Criterion | pymatgen make_supercell | pymatgen CubicSupercellTransformation | ASE make_supercell |
|---|---|---|---|
| Diagonal supercells | Simple (n1, n2, n3) | Not needed | Simple repeat |
| Orthogonal supercells | Requires manual matrix | Automatic | Requires manual matrix |
| Non-diagonal supercells | Full 3x3 matrix | Automatic | Full 3x3 matrix |
| Minimum length constraint | Manual calculation | Built-in min_length | Manual calculation |
| Best for | Quick supercells | Orthogonal cells for LAMMPS/classical MD | ASE workflows |

## Prerequisites

- pymatgen, ASE, numpy (pre-installed)
- spglib (pre-installed, for conventional cell)

---

## Detailed Steps

### Method A: pymatgen Supercell Construction

```python
#!/usr/bin/env python3
"""
Build supercells using pymatgen.
Covers diagonal, non-diagonal, and orthogonal supercell construction.
Also includes minimum-length supercell determination.
"""

import numpy as np
import json
from pymatgen.core import Structure, Lattice
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
from pymatgen.io.vasp import Poscar
from pymatgen.transformations.advanced_transformations import (
    CubicSupercellTransformation,
)
from pymatgen.transformations.standard_transformations import (
    SupercellTransformation,
)

# ============================================================
# 1. Load or build a structure
# ============================================================
# Option 1: Load from file
# structure = Structure.from_file("POSCAR")

# Option 2: Build FCC Si
structure = Structure.from_spacegroup(
    "Fd-3m",
    lattice=[[5.431, 0, 0], [0, 5.431, 0], [0, 0, 5.431]],
    species=["Si"],
    coords=[[0.0, 0.0, 0.0]],
)

# Option 3: HCP Mg (non-orthogonal)
# structure = Structure.from_spacegroup(
#     "P6_3/mmc",
#     lattice=Lattice.hexagonal(3.209, 5.211),
#     species=["Mg", "Mg"],
#     coords=[[1/3, 2/3, 0.25], [2/3, 1/3, 0.75]],
# )

print(f"Input structure: {structure.composition.reduced_formula}")
print(f"  Atoms: {len(structure)}")
print(f"  Cell: a={structure.lattice.a:.4f}, b={structure.lattice.b:.4f}, "
      f"c={structure.lattice.c:.4f}")
print(f"  Angles: alpha={structure.lattice.alpha:.1f}, "
      f"beta={structure.lattice.beta:.1f}, gamma={structure.lattice.gamma:.1f}")
print(f"  Volume: {structure.volume:.2f} A^3")

# ============================================================
# 2. Simple diagonal supercell (NxNxN)
# ============================================================
print("\n=== Diagonal supercells ===")

for n in [2, 3, 4]:
    sc = structure.copy()
    sc.make_supercell([n, n, n])
    print(f"  {n}x{n}x{n}: {len(sc)} atoms, volume = {sc.volume:.1f} A^3, "
          f"cell = ({sc.lattice.a:.2f}, {sc.lattice.b:.2f}, {sc.lattice.c:.2f})")
    sc.to(f"supercell_{n}x{n}x{n}.cif")

# ============================================================
# 3. Anisotropic supercell (NxMxL)
# ============================================================
print("\n=== Anisotropic supercell ===")
sc_nml = structure.copy()
sc_nml.make_supercell([2, 3, 4])
print(f"  2x3x4: {len(sc_nml)} atoms, "
      f"cell = ({sc_nml.lattice.a:.2f}, {sc_nml.lattice.b:.2f}, {sc_nml.lattice.c:.2f})")
sc_nml.to("supercell_2x3x4.cif")

# ============================================================
# 4. Non-diagonal supercell (general 3x3 matrix)
# ============================================================
print("\n=== Non-diagonal supercell ===")

# Example: transform FCC primitive to conventional cubic
# FCC primitive: a1 = (0, a/2, a/2), a2 = (a/2, 0, a/2), a3 = (a/2, a/2, 0)
# Conventional cubic: A1 = (a, 0, 0), A2 = (0, a, 0), A3 = (0, 0, a)
# Transformation matrix M such that [A1, A2, A3] = M . [a1, a2, a3]:
# M = [[-1, 1, 1], [1, -1, 1], [1, 1, -1]]

sga = SpacegroupAnalyzer(structure)
prim = sga.get_primitive_standard_structure()
print(f"  Primitive: {len(prim)} atoms, cell = "
      f"({prim.lattice.a:.4f}, {prim.lattice.b:.4f}, {prim.lattice.c:.4f})")

conv = sga.get_conventional_standard_structure()
print(f"  Conventional: {len(conv)} atoms, cell = "
      f"({conv.lattice.a:.4f}, {conv.lattice.b:.4f}, {conv.lattice.c:.4f})")

# General non-diagonal supercell
matrix = [[2, 1, 0], [0, 2, 0], [0, 0, 1]]
sc_nd = structure.copy()
sc_nd.make_supercell(matrix)
print(f"  Matrix [[2,1,0],[0,2,0],[0,0,1]]: {len(sc_nd)} atoms")
sc_nd.to("supercell_nondiag.cif")

# ============================================================
# 5. Minimum-length supercell
# ============================================================
print("\n=== Minimum-length supercell ===")

def get_min_length_supercell(structure, min_length=10.0):
    """
    Determine the smallest supercell such that all lattice vector lengths
    are >= min_length.
    """
    cell_lengths = np.array(structure.lattice.abc)
    multiples = np.ceil(min_length / cell_lengths).astype(int)
    multiples = np.maximum(multiples, 1)

    sc = structure.copy()
    sc.make_supercell(multiples.tolist())

    print(f"  Input cell lengths: {cell_lengths}")
    print(f"  Required min_length: {min_length} A")
    print(f"  Supercell: {multiples[0]}x{multiples[1]}x{multiples[2]}")
    print(f"  Output cell lengths: ({sc.lattice.a:.2f}, {sc.lattice.b:.2f}, "
          f"{sc.lattice.c:.2f})")
    print(f"  Atoms: {len(sc)}")

    return sc, multiples

sc_min, mult = get_min_length_supercell(structure, min_length=10.0)
sc_min.to("supercell_min10A.cif")

# For defect calculations, typically need >= 10 A
sc_defect, _ = get_min_length_supercell(structure, min_length=12.0)
sc_defect.to("supercell_defect.cif")

# ============================================================
# 6. Orthogonal supercell (CubicSupercellTransformation)
# ============================================================
print("\n=== Orthogonal (cubic) supercell ===")

# This is the key feature for converting non-orthogonal cells
# (e.g., HCP, rhombohedral) to orthogonal cells needed by
# LAMMPS or some visualization tools.

# Build an HCP structure for this demo
hcp_mg = Structure.from_spacegroup(
    "P6_3/mmc",
    lattice=Lattice.hexagonal(3.209, 5.211),
    species=["Mg", "Mg"],
    coords=[[1/3, 2/3, 0.25], [2/3, 1/3, 0.75]],
)
print(f"\n  HCP Mg primitive:")
print(f"    Angles: {hcp_mg.lattice.alpha:.1f}, {hcp_mg.lattice.beta:.1f}, "
      f"{hcp_mg.lattice.gamma:.1f}")

try:
    cst = CubicSupercellTransformation(
        min_length=10.0,       # Minimum length of each axis
        max_atoms=200,         # Maximum allowed atoms
        min_atoms=20,          # Minimum atoms (for efficiency)
        force_diagonal=False,  # Allow non-diagonal transformations
    )
    ortho_sc = cst.apply_transformation(hcp_mg)
    print(f"  Orthogonal supercell:")
    print(f"    Atoms: {len(ortho_sc)}")
    print(f"    Cell: ({ortho_sc.lattice.a:.2f}, {ortho_sc.lattice.b:.2f}, "
          f"{ortho_sc.lattice.c:.2f})")
    print(f"    Angles: {ortho_sc.lattice.alpha:.1f}, {ortho_sc.lattice.beta:.1f}, "
          f"{ortho_sc.lattice.gamma:.1f}")
    ortho_sc.to("Mg_orthogonal_supercell.cif")
    Poscar(ortho_sc).write_file("POSCAR_ortho")
except Exception as e:
    print(f"  CubicSupercellTransformation failed: {e}")
    print("  Trying manual orthogonalization...")

    # Manual orthogonal supercell for HCP
    # HCP to orthorhombic: M = [[1, -1, 0], [1, 1, 0], [0, 0, 1]]
    # This gives an orthorhombic cell with a_orth = a, b_orth = a*sqrt(3), c_orth = c
    ortho_matrix = [[1, -1, 0], [1, 1, 0], [0, 0, 1]]
    ortho_sc = hcp_mg.copy()
    ortho_sc.make_supercell(ortho_matrix)

    # Then scale up
    ortho_sc.make_supercell([2, 2, 2])
    print(f"  Manual orthogonal supercell:")
    print(f"    Atoms: {len(ortho_sc)}")
    print(f"    Cell: ({ortho_sc.lattice.a:.2f}, {ortho_sc.lattice.b:.2f}, "
          f"{ortho_sc.lattice.c:.2f})")
    print(f"    Angles: {ortho_sc.lattice.alpha:.1f}, {ortho_sc.lattice.beta:.1f}, "
          f"{ortho_sc.lattice.gamma:.1f}")
    ortho_sc.to("Mg_orthogonal_supercell.cif")

# ============================================================
# 7. Write QE input for supercell
# ============================================================
def write_qe_supercell(sc, pseudo_dir="./pseudo", filename="supercell_scf.in"):
    """Write QE input for a supercell calculation."""
    cell = sc.lattice.matrix
    elements = sorted(set(str(s.specie) for s in sc))

    cell_lines = "\n".join(f"  {v[0]:.10f}  {v[1]:.10f}  {v[2]:.10f}" for v in cell)

    species_lines = []
    for el in elements:
        from pymatgen.core.periodic_table import Element
        mass = Element(el).atomic_mass
        species_lines.append(f"  {el:4s} {mass:10.4f}  {el}.pbe-n-rrkjus_psl.1.0.0.UPF")

    pos_lines = []
    for site in sc:
        fc = site.frac_coords
        pos_lines.append(f"  {str(site.specie):4s} {fc[0]:.10f} {fc[1]:.10f} {fc[2]:.10f}")

    # Determine k-mesh (inverse proportional to cell size)
    lengths = sc.lattice.abc
    k_grid = [max(1, int(np.ceil(30.0 / l))) for l in lengths]

    qe_input = f"""&CONTROL
    calculation = 'scf'
    prefix      = 'supercell'
    outdir      = './tmp'
    pseudo_dir  = '{pseudo_dir}'
/
&SYSTEM
    ibrav       = 0
    nat         = {len(sc)}
    ntyp        = {len(elements)}
    ecutwfc     = 50.0
    ecutrho     = 400.0
    occupations = 'smearing'
    smearing    = 'cold'
    degauss     = 0.01
/
&ELECTRONS
    conv_thr = 1.0d-8
/

CELL_PARAMETERS angstrom
{cell_lines}

ATOMIC_SPECIES
{chr(10).join(species_lines)}

ATOMIC_POSITIONS crystal
{chr(10).join(pos_lines)}

K_POINTS automatic
  {k_grid[0]} {k_grid[1]} {k_grid[2]}  0 0 0
"""
    with open(filename, "w") as f:
        f.write(qe_input)
    print(f"\nQE input written: {filename} ({len(sc)} atoms, k-grid: {k_grid})")

write_qe_supercell(sc_min)
```

### Method B: ASE Supercell Construction

```python
#!/usr/bin/env python3
"""
Build supercells using ASE.
Includes diagonal, non-diagonal, and orthogonal supercells.
"""

import numpy as np
from ase.build import bulk, make_supercell
from ase.io import write, read

# ============================================================
# 1. Simple diagonal supercell
# ============================================================
print("=== Diagonal supercells ===")

atoms = bulk("Si", "diamond", a=5.431)
print(f"  Primitive: {len(atoms)} atoms, cell = {atoms.cell.lengths()}")

# NxNxN supercell
for n in [2, 3, 4]:
    sc = atoms.repeat((n, n, n))
    print(f"  {n}x{n}x{n}: {len(sc)} atoms, cell = {np.round(sc.cell.lengths(), 2)}")
    write(f"Si_sc_{n}x{n}x{n}.cif", sc)

# ============================================================
# 2. Non-diagonal supercell with make_supercell
# ============================================================
print("\n=== Non-diagonal supercell ===")

# Transformation matrix P: new_cell = P @ old_cell
P = np.array([[2, 1, 0],
              [0, 2, 0],
              [0, 0, 2]])

sc_nd = make_supercell(atoms, P)
print(f"  Matrix P={P.tolist()}: {len(sc_nd)} atoms")
print(f"  Cell: {np.round(sc_nd.cell.lengths(), 2)}")
print(f"  Angles: {np.round(sc_nd.cell.cellpar()[3:], 1)}")
write("Si_nondiag.cif", sc_nd)

# ============================================================
# 3. Orthogonal supercell from HCP
# ============================================================
print("\n=== Orthogonal supercell from HCP ===")

mg_hcp = bulk("Mg", "hcp", a=3.209, c=5.211)
print(f"  HCP Mg: {len(mg_hcp)} atoms, angles = {np.round(mg_hcp.cell.cellpar()[3:], 1)}")

# HCP to orthorhombic transformation
# This creates an orthorhombic cell from HCP
P_ortho = np.array([[1, -1, 0],
                     [1,  1, 0],
                     [0,  0, 1]])

mg_ortho = make_supercell(mg_hcp, P_ortho)
print(f"  Orthorhombic: {len(mg_ortho)} atoms")
print(f"  Cell: {np.round(mg_ortho.cell.lengths(), 3)}")
print(f"  Angles: {np.round(mg_ortho.cell.cellpar()[3:], 1)}")

# Scale up
mg_ortho_big = mg_ortho.repeat((3, 3, 2))
print(f"  3x3x2 orthorhombic: {len(mg_ortho_big)} atoms")
write("Mg_orthorhombic.cif", mg_ortho_big)

# ============================================================
# 4. FCC primitive to conventional cubic
# ============================================================
print("\n=== FCC primitive to conventional ===")

cu_prim = bulk("Cu", "fcc", a=3.615)  # primitive (1 atom)
print(f"  FCC primitive: {len(cu_prim)} atoms, "
      f"cell = {np.round(cu_prim.cell.lengths(), 3)}")

# Conventional cubic: M = [[-1,1,1],[1,-1,1],[1,1,-1]]
P_conv = np.array([[-1, 1, 1],
                    [ 1,-1, 1],
                    [ 1, 1,-1]])

cu_conv = make_supercell(cu_prim, P_conv)
print(f"  Conventional cubic: {len(cu_conv)} atoms, "
      f"cell = {np.round(cu_conv.cell.lengths(), 3)}")
print(f"  Angles: {np.round(cu_conv.cell.cellpar()[3:], 1)}")
write("Cu_conventional.cif", cu_conv)

# ============================================================
# 5. Minimum-length supercell
# ============================================================
print("\n=== Minimum-length supercell ===")

def min_length_supercell(atoms, min_length=10.0):
    """Build the smallest supercell with all dimensions >= min_length."""
    lengths = atoms.cell.lengths()
    multiples = np.ceil(min_length / lengths).astype(int)
    multiples = np.maximum(multiples, 1)
    sc = atoms.repeat(multiples)
    print(f"  Input: {np.round(lengths, 2)}")
    print(f"  Multiples: {multiples}")
    print(f"  Output: {len(sc)} atoms, {np.round(sc.cell.lengths(), 2)}")
    return sc

sc = min_length_supercell(atoms, min_length=12.0)
write("Si_min12A.cif", sc)

# ============================================================
# 6. Write LAMMPS data file (orthogonal cell)
# ============================================================
print("\n=== LAMMPS output ===")
write("orthogonal.lmp", mg_ortho_big, format="lammps-data")
print("  Written: orthogonal.lmp (LAMMPS data format)")
```

### Method C: VASP (Future External Access)

```python
#!/usr/bin/env python3
"""
Generate VASP POSCAR files for various supercell types.
"""

from pymatgen.core import Structure, Lattice
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
from pymatgen.io.vasp import Poscar
from pymatgen.transformations.advanced_transformations import (
    CubicSupercellTransformation,
)
import numpy as np

# Load structure
structure = Structure.from_spacegroup(
    "P6_3/mmc",
    lattice=Lattice.hexagonal(3.209, 5.211),
    species=["Mg", "Mg"],
    coords=[[1/3, 2/3, 0.25], [2/3, 1/3, 0.75]],
)

# 1. Conventional cell POSCAR
sga = SpacegroupAnalyzer(structure)
conv = sga.get_conventional_standard_structure()
Poscar(conv).write_file("POSCAR_conventional")
print(f"Conventional: {len(conv)} atoms, a={conv.lattice.a:.3f}")

# 2. Supercell POSCAR
sc = structure.copy()
sc.make_supercell([3, 3, 3])
Poscar(sc).write_file("POSCAR_supercell")
print(f"Supercell 3x3x3: {len(sc)} atoms")

# 3. Orthogonal POSCAR
try:
    cst = CubicSupercellTransformation(
        min_length=10.0, max_atoms=300,
    )
    ortho = cst.apply_transformation(structure)
    Poscar(ortho).write_file("POSCAR_orthogonal")
    print(f"Orthogonal: {len(ortho)} atoms, "
          f"angles = ({ortho.lattice.alpha:.1f}, {ortho.lattice.beta:.1f}, "
          f"{ortho.lattice.gamma:.1f})")
except Exception as e:
    print(f"Orthogonal transformation failed: {e}")

# 4. INCAR for supercell SCF
from pymatgen.io.vasp import Incar
incar = Incar({
    "SYSTEM": "Supercell",
    "ENCUT": 520,
    "PREC": "Accurate",
    "EDIFF": 1e-6,
    "ISMEAR": 1,
    "SIGMA": 0.1,
    "NSW": 0,
    "LWAVE": True,
    "LCHARG": True,
})
incar.write_file("INCAR_supercell")
print("\nVASP files written: POSCAR_conventional, POSCAR_supercell, "
      "POSCAR_orthogonal, INCAR_supercell")
```

---

## Key Parameters

| Parameter | Typical Value | Notes |
|---|---|---|
| Supercell size | 2x2x2 to 4x4x4 | Depends on property: phonons need ~20 A, defects need ~10 A |
| `min_length` (CubicSupercellTransformation) | 10.0 A | Minimum dimension of the orthogonal cell |
| `max_atoms` | 200--500 | Upper limit to prevent excessively large cells |
| k-mesh for supercell | Scale inversely with cell | If primitive uses 8x8x8, a 3x3x3 supercell uses ~3x3x3 |
| Transformation matrix | 3x3 integer matrix | Columns are new lattice vectors in units of old vectors |

## Common Issues

| Problem | Solution |
|---|---|
| CubicSupercellTransformation fails | Increase `max_atoms` or decrease `min_length`. Some structures require large cells for orthogonality. |
| Supercell has wrong number of atoms | Check the transformation matrix determinant: det(M) gives the volume ratio (number of primitive cells in supercell). |
| Orthogonal cell has slightly non-90 angles | Numerical precision. Round angles < 90.01 to 90.0, or use `structure.lattice = Lattice.orthorhombic(a, b, c)`. |
| LAMMPS requires orthogonal cell | Use CubicSupercellTransformation or the HCP-to-orthorhombic matrix shown above. |
| Supercell loses symmetry | Expected for non-diagonal transformations. The supercell has the translational symmetry of the new cell. |
| Too many atoms for DFT | Use Gamma-only k-point for large supercells. Consider MACE for quick relaxation. |
