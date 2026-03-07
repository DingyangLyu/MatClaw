# Heterostructure Builder

## When to Use

- You need to build a heterostructure (interface) from two different materials.
- You want to find lattice-matched interfaces between two slabs.
- You are studying interface adhesion, band alignment, or charge transfer.
- You need to generate a van der Waals heterostructure (e.g., graphene/h-BN).
- You want to explore multiple interface orientations and terminations systematically.

## Method Selection

| Criterion | pymatgen ZSLGenerator | pymatgen CoherentInterfaceBuilder | Manual stacking |
|---|---|---|---|
| Lattice matching | Automatic (Zur-McGill algorithm) | Automatic (based on ZSL) | Manual, requires known match |
| Strain handling | Reports strain, user applies | Built-in strain application | Manual |
| Complexity | Moderate | Higher-level API | Simple but limited |
| Best for | Systematic search for matched interfaces | Ready-to-use interface structures | Simple 2D/2D stacking |

## Prerequisites

- pymatgen (pre-installed)
- pymatgen.analysis.interfaces (included in pymatgen)
- ASE (pre-installed)
- numpy, matplotlib (pre-installed)

---

## Detailed Steps

### Method A: pymatgen Interface Builder (Recommended)

```python
#!/usr/bin/env python3
"""
Build heterostructures using pymatgen interface tools.
Uses the Zur and McGill (ZSL) algorithm for lattice matching
and CoherentInterfaceBuilder for generating interface structures.

Example: Si/Ge heterostructure.
"""

import numpy as np
import json
import warnings
warnings.filterwarnings("ignore")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from pymatgen.core import Structure, Lattice
from pymatgen.core.surface import SlabGenerator
from pymatgen.analysis.interfaces.zsl import ZSLGenerator
from pymatgen.analysis.interfaces.coherent_interfaces import (
    CoherentInterfaceBuilder,
)
from pymatgen.io.vasp import Poscar
from pathlib import Path

output_dir = Path("heterostructure")
output_dir.mkdir(exist_ok=True)

# ============================================================
# 1. Define the two bulk materials
# ============================================================
# Material 1: Si (diamond structure)
si_bulk = Structure.from_spacegroup(
    "Fd-3m",
    lattice=Lattice.cubic(5.431),
    species=["Si"],
    coords=[[0.0, 0.0, 0.0]],
)

# Material 2: Ge (diamond structure)
ge_bulk = Structure.from_spacegroup(
    "Fd-3m",
    lattice=Lattice.cubic(5.658),
    species=["Ge"],
    coords=[[0.0, 0.0, 0.0]],
)

print(f"Material 1: {si_bulk.composition.reduced_formula}, a = {si_bulk.lattice.a:.3f} A")
print(f"Material 2: {ge_bulk.composition.reduced_formula}, a = {ge_bulk.lattice.a:.3f} A")
lattice_mismatch = abs(si_bulk.lattice.a - ge_bulk.lattice.a) / si_bulk.lattice.a * 100
print(f"Lattice mismatch: {lattice_mismatch:.2f}%")

# ============================================================
# 2. Use CoherentInterfaceBuilder
# ============================================================
MILLER_INDEX = (1, 0, 0)  # Interface orientation
MAX_AREA = 50.0            # Maximum interface area in A^2
MAX_STRAIN = 0.05          # Maximum allowed strain (5%)
MIN_SLAB_SIZE = 8.0        # Minimum slab thickness in A

print(f"\n=== Building {MILLER_INDEX} interface ===")

cib = CoherentInterfaceBuilder(
    substrate_structure=si_bulk,
    film_structure=ge_bulk,
    film_miller_index=MILLER_INDEX,
    substrate_miller_index=MILLER_INDEX,
    zslgen=ZSLGenerator(
        max_area_ratio_tol=0.09,
        max_area=MAX_AREA,
        max_length_tol=0.03,
        max_angle_tol=0.01,
    ),
)

# Get all matched interfaces
interfaces = list(cib.get_interfaces(
    termination=None,           # Try all terminations
    gap=2.0,                    # Initial gap between slabs (A)
    vacuum_over_film=15.0,      # Vacuum above the film
    film_thickness=MIN_SLAB_SIZE,
    substrate_thickness=MIN_SLAB_SIZE,
))

print(f"Found {len(interfaces)} interface configurations")

# ============================================================
# 3. Analyze and save interfaces
# ============================================================
interface_data = []
for i, iface in enumerate(interfaces[:10]):  # Take top 10
    n_atoms = len(iface)
    area = iface.lattice.a * iface.lattice.b * np.sin(
        np.radians(iface.lattice.gamma)
    )

    print(f"\n--- Interface {i} ---")
    print(f"  Atoms: {n_atoms}")
    print(f"  Interface area: {area:.2f} A^2")
    print(f"  Cell: a={iface.lattice.a:.2f}, b={iface.lattice.b:.2f}, "
          f"c={iface.lattice.c:.2f}")
    print(f"  Composition: {iface.composition}")

    iface.to(str(output_dir / f"interface_{i}.cif"))
    Poscar(iface).write_file(str(output_dir / f"POSCAR_interface_{i}"))

    interface_data.append({
        "index": i,
        "n_atoms": n_atoms,
        "interface_area_A2": round(area, 2),
        "composition": str(iface.composition),
    })

with open(str(output_dir / "interface_summary.json"), "w") as f:
    json.dump(interface_data, f, indent=2)

# ============================================================
# 4. Direct ZSL lattice matching (lower-level)
# ============================================================
print("\n=== Direct ZSL lattice matching ===")

zsl = ZSLGenerator(
    max_area_ratio_tol=0.09,
    max_area=100.0,
    max_length_tol=0.03,
    max_angle_tol=0.01,
)

# Generate slabs first
si_slab_gen = SlabGenerator(
    si_bulk, MILLER_INDEX, min_slab_size=8.0, min_vacuum_size=0.1,
    center_slab=True, primitive=True,
)
ge_slab_gen = SlabGenerator(
    ge_bulk, MILLER_INDEX, min_slab_size=8.0, min_vacuum_size=0.1,
    center_slab=True, primitive=True,
)

si_slabs = si_slab_gen.get_slabs(symmetrize=False)
ge_slabs = ge_slab_gen.get_slabs(symmetrize=False)

if si_slabs and ge_slabs:
    si_slab = si_slabs[0]
    ge_slab = ge_slabs[0]

    # Get 2D lattice vectors (in-plane)
    si_2d = si_slab.lattice.matrix[:2, :2]  # 2x2 in-plane
    ge_2d = ge_slab.lattice.matrix[:2, :2]

    matches = list(zsl(si_2d, ge_2d))
    print(f"  Found {len(matches)} ZSL matches for {MILLER_INDEX}")

    for j, match in enumerate(matches[:5]):
        strain = np.linalg.norm(match.match_area)
        print(f"  Match {j}: substrate_sl={match.substrate_sl_vectors}, "
              f"film_sl={match.film_sl_vectors}, "
              f"match_area={match.match_area:.2f}")

# ============================================================
# 5. Relax interface with MACE
# ============================================================
if interfaces:
    print("\n=== MACE relaxation of interface ===")

    from mace.calculators import mace_mp
    from ase.optimize import BFGS
    from ase.constraints import FixAtoms
    from pymatgen.io.ase import AseAtomsAdaptor

    adaptor = AseAtomsAdaptor()
    calc = mace_mp(model="medium", dispersion=True, default_dtype="float64")

    iface = interfaces[0]
    atoms = adaptor.get_atoms(iface)

    # Fix bottom half of substrate
    z = atoms.positions[:, 2]
    z_mid = (z.max() + z.min()) / 2
    mask = z < (z.min() + (z_mid - z.min()) * 0.5)
    atoms.set_constraint(FixAtoms(mask=mask))
    print(f"  Fixed {mask.sum()} atoms in lower substrate")

    atoms.calc = calc
    opt = BFGS(atoms, logfile=str(output_dir / "interface_relax.log"))
    opt.run(fmax=0.01, steps=300)

    e_interface = atoms.get_potential_energy()
    print(f"  Relaxed interface energy: {e_interface:.4f} eV")

    from ase.io import write as ase_write
    ase_write(str(output_dir / "interface_relaxed.cif"), atoms)
```

### Method B: Manual 2D/2D Heterostructure Stacking

```python
#!/usr/bin/env python3
"""
Build van der Waals heterostructures by manually stacking 2D layers.
Example: graphene/h-BN bilayer.
"""

import numpy as np
from pymatgen.core import Structure, Lattice, Element
from pymatgen.io.vasp import Poscar
from pymatgen.io.ase import AseAtomsAdaptor
from ase.io import write as ase_write

# ============================================================
# 1. Build the two monolayers
# ============================================================
# Graphene
a_gr = 2.46
graphene = Structure(
    lattice=Lattice.hexagonal(a_gr, 20.0),
    species=["C", "C"],
    coords=[[0.0, 0.0, 0.5], [1/3, 2/3, 0.5]],
)

# h-BN
a_bn = 2.50
hbn = Structure(
    lattice=Lattice.hexagonal(a_bn, 20.0),
    species=["B", "N"],
    coords=[[0.0, 0.0, 0.5], [1/3, 2/3, 0.5]],
)

print(f"Graphene: a = {a_gr:.3f} A")
print(f"h-BN:    a = {a_bn:.3f} A")
print(f"Lattice mismatch: {abs(a_gr - a_bn) / a_gr * 100:.2f}%")

# ============================================================
# 2. Build the heterostructure
# ============================================================
def stack_2d_layers(layer1, layer2, interlayer_distance=3.35,
                    vacuum=15.0, strain_to="average"):
    """
    Stack two 2D layers to form a heterostructure.

    Parameters
    ----------
    layer1, layer2 : Structure
        Two monolayer structures (vacuum along c).
    interlayer_distance : float
        Distance between layers in Angstrom.
    vacuum : float
        Vacuum above/below the bilayer.
    strain_to : str
        "layer1", "layer2", or "average" -- which lattice to use.

    Returns
    -------
    Structure
        Heterostructure.
    """
    # Determine common in-plane lattice
    a1 = layer1.lattice.a
    a2 = layer2.lattice.a

    if strain_to == "layer1":
        a_common = a1
    elif strain_to == "layer2":
        a_common = a2
    else:  # average
        a_common = (a1 + a2) / 2

    strain1 = (a_common - a1) / a1
    strain2 = (a_common - a2) / a2
    print(f"  Common lattice parameter: {a_common:.4f} A")
    print(f"  Strain on layer 1: {strain1*100:+.2f}%")
    print(f"  Strain on layer 2: {strain2*100:+.2f}%")

    # Total cell height
    c_total = interlayer_distance + vacuum * 2

    # New lattice (hexagonal for graphene/h-BN)
    new_lattice = Lattice.hexagonal(a_common, c_total)

    # Position layers
    z_center = 0.5
    dz = interlayer_distance / c_total

    # Layer 1 positions (below center)
    species = []
    coords = []
    for site in layer1:
        species.append(site.specie)
        fc = site.frac_coords.copy()
        fc[2] = z_center - dz / 2
        # Scale in-plane to new lattice
        coords.append(fc)

    # Layer 2 positions (above center)
    for site in layer2:
        species.append(site.specie)
        fc = site.frac_coords.copy()
        fc[2] = z_center + dz / 2
        coords.append(fc)

    hetero = Structure(new_lattice, species, coords)
    return hetero

hetero = stack_2d_layers(graphene, hbn, interlayer_distance=3.35,
                         vacuum=15.0, strain_to="average")

print(f"\nHeterostructure: {hetero.composition}")
print(f"  Atoms: {len(hetero)}")
print(f"  Cell: a={hetero.lattice.a:.3f}, c={hetero.lattice.c:.3f}")

hetero.to("graphene_hBN_hetero.cif")
Poscar(hetero).write_file("POSCAR_hetero")

# ============================================================
# 3. Build supercell heterostructure (for Moire patterns)
# ============================================================
print("\n=== Moire supercell (approximate) ===")

def find_commensurate_supercell(a1, a2, max_n=10, tol=0.01):
    """
    Find commensurate supercell for two lattices.
    Finds n1, n2 such that |n1*a1 - n2*a2| / (n1*a1) < tol.
    """
    best = None
    best_strain = 1.0

    for n1 in range(1, max_n + 1):
        for n2 in range(1, max_n + 1):
            l1 = n1 * a1
            l2 = n2 * a2
            strain = abs(l1 - l2) / l1
            if strain < tol and strain < best_strain:
                best = (n1, n2, strain)
                best_strain = strain

    return best

result = find_commensurate_supercell(a_gr, a_bn, max_n=15, tol=0.02)
if result:
    n1, n2, strain = result
    print(f"  Commensurate: {n1}x graphene ({n1*a_gr:.3f} A) ~ "
          f"{n2}x h-BN ({n2*a_bn:.3f} A), strain = {strain*100:.2f}%")

    # Build commensurate heterostructure
    gr_sc = graphene.copy()
    gr_sc.make_supercell([n1, n1, 1])

    bn_sc = hbn.copy()
    bn_sc.make_supercell([n2, n2, 1])

    # Stack with average lattice
    hetero_moire = stack_2d_layers(gr_sc, bn_sc, strain_to="average")
    print(f"  Moire heterostructure: {len(hetero_moire)} atoms")
    hetero_moire.to("graphene_hBN_moire.cif")

# ============================================================
# 4. Twist bilayer
# ============================================================
print("\n=== Twisted bilayer (manual rotation) ===")

def rotate_2d_layer(structure, angle_deg):
    """
    Rotate a 2D layer by angle_deg around the z-axis.
    Returns the rotated structure with a new lattice.
    """
    theta = np.radians(angle_deg)
    R = np.array([
        [np.cos(theta), -np.sin(theta), 0],
        [np.sin(theta),  np.cos(theta), 0],
        [0, 0, 1]
    ])

    new_lattice_matrix = structure.lattice.matrix @ R.T
    new_lattice = Lattice(new_lattice_matrix)

    return Structure(
        new_lattice,
        structure.species,
        structure.cart_coords,
        coords_are_cartesian=True,
    )

twist_angle = 21.8  # degrees (common commensurate angle for graphene)
gr_twisted = rotate_2d_layer(graphene, twist_angle)
print(f"  Twisted graphene layer by {twist_angle} degrees")
print(f"  Original lattice: {np.round(graphene.lattice.matrix[:2, :2], 3)}")
print(f"  Rotated lattice:  {np.round(gr_twisted.lattice.matrix[:2, :2], 3)}")

# ============================================================
# 5. Write QE input for heterostructure
# ============================================================
def write_qe_hetero(structure, pseudo_dir="./pseudo",
                     filename="hetero_scf.in"):
    """Write QE input for heterostructure calculation."""
    cell = structure.lattice.matrix
    elements = sorted(set(str(s.specie) for s in structure))

    cell_lines = "\n".join(
        f"  {v[0]:.10f}  {v[1]:.10f}  {v[2]:.10f}" for v in cell
    )

    species_lines = []
    for el in elements:
        from pymatgen.core.periodic_table import Element as PmgElem
        mass = PmgElem(el).atomic_mass
        species_lines.append(
            f"  {el:4s} {mass:10.4f}  {el}.pbe-n-rrkjus_psl.1.0.0.UPF"
        )

    pos_lines = []
    for site in structure:
        fc = site.frac_coords
        pos_lines.append(
            f"  {str(site.specie):4s} {fc[0]:.10f} {fc[1]:.10f} {fc[2]:.10f}"
        )

    qe_input = f"""&CONTROL
    calculation = 'relax'
    prefix      = 'hetero'
    outdir      = './tmp'
    pseudo_dir  = '{pseudo_dir}'
    tprnfor     = .true.
/
&SYSTEM
    ibrav       = 0
    nat         = {len(structure)}
    ntyp        = {len(elements)}
    ecutwfc     = 60.0
    ecutrho     = 480.0
    occupations = 'smearing'
    smearing    = 'cold'
    degauss     = 0.01
    vdw_corr    = 'dft-d3'
    assume_isolated = '2D'
/
&ELECTRONS
    conv_thr    = 1.0d-8
    mixing_beta = 0.3
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
  12 12 1  0 0 0
"""
    with open(filename, "w") as f:
        f.write(qe_input)
    print(f"\nQE input written: {filename}")
    print(f"  Key settings: vdw_corr='dft-d3', assume_isolated='2D'")

write_qe_hetero(hetero)
```

### Method C: VASP (Future External Access)

```python
#!/usr/bin/env python3
"""
Generate VASP input files for heterostructure calculations.
"""

import numpy as np
from pymatgen.core import Structure, Lattice
from pymatgen.io.vasp import Poscar, Incar, Kpoints

# Build a simple heterostructure (see Method A/B above)
a_common = 2.48
c_total = 36.7  # interlayer + vacuum

hetero = Structure(
    lattice=Lattice.hexagonal(a_common, c_total),
    species=["C", "C", "B", "N"],
    coords=[
        [0.0, 0.0, 0.455],
        [1/3, 2/3, 0.455],
        [0.0, 0.0, 0.545],
        [1/3, 2/3, 0.545],
    ],
)

# POSCAR
Poscar(hetero).write_file("POSCAR_hetero")

# KPOINTS
kpts = Kpoints.gamma_automatic(kpts=(12, 12, 1))
kpts.write_file("KPOINTS_hetero")

# INCAR
incar = Incar({
    "SYSTEM": "Graphene/h-BN heterostructure",
    "ENCUT": 520,
    "PREC": "Accurate",
    "EDIFF": 1e-6,
    "EDIFFG": -0.01,
    "IBRION": 2,
    "NSW": 100,
    "ISIF": 2,             # Relax ions only (fixed cell for hetero)
    "ISMEAR": 0,
    "SIGMA": 0.05,
    "LVDW": True,
    "IVDW": 12,            # DFT-D3(BJ) for vdW interactions
    "LDIPOL": True,
    "IDIPOL": 3,
    "LWAVE": False,
    "LCHARG": True,
})
incar.write_file("INCAR_hetero")

print("VASP heterostructure files written:")
print("  POSCAR_hetero, KPOINTS_hetero, INCAR_hetero")
print("  Key: IVDW=12 (DFT-D3), ISIF=2 (relax ions only), LDIPOL=True")
```

---

## Key Parameters

| Parameter | Typical Value | Notes |
|---|---|---|
| `max_area` (ZSL) | 50--200 A^2 | Maximum interface supercell area. Larger allows more matches but more atoms. |
| `max_length_tol` (ZSL) | 0.03 | Tolerance for lattice vector length matching |
| `max_angle_tol` (ZSL) | 0.01 | Tolerance for angle matching (radians) |
| `gap` | 2.0--3.5 A | Initial interlayer distance. 3.35 A typical for vdW systems. |
| `vacuum_over_film` | 15--20 A | Vacuum above the top layer |
| `film_thickness` / `substrate_thickness` | 8--15 A | Slab thickness for each material |
| `vdw_corr` (QE) | `'dft-d3'` | Van der Waals correction, critical for heterostructures |
| `IVDW` (VASP) | 12 (DFT-D3(BJ)) | Van der Waals correction for VASP |
| `strain_to` | `"average"` or `"substrate"` | Which lattice to adopt. "substrate" is physical for epitaxial growth. |

## Common Issues

| Problem | Solution |
|---|---|
| ZSLGenerator finds no matches | Increase `max_area_ratio_tol`, `max_length_tol`, or `max_area`. Materials with large mismatch need larger supercells. |
| Too many atoms in interface | Reduce `max_area` or choose a lower-index interface. Use MACE for initial relaxation. |
| Interface structure has unphysical overlapping atoms | Increase the `gap` parameter. Check that both slabs are properly oriented. |
| Van der Waals correction missing | Always include vdW for heterostructures: `vdw_corr='dft-d3'` in QE or `IVDW=12` in VASP. |
| Strain is too large | Look for higher-order commensurate supercells. Some interfaces are inherently poorly matched. |
| Band alignment calculation needed | Compute the vacuum level (electrostatic potential) for each material separately and aligned at the interface. |
