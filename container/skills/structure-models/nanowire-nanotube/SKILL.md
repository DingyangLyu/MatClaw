# Nanowire, Nanotube, and Nanoribbon Builder

## When to Use

- You need to build a nanowire model by carving a cylinder from a bulk crystal.
- You want to construct a nanotube by rolling a 2D sheet (e.g., carbon nanotube, BN nanotube).
- You need a nanoribbon model (e.g., graphene nanoribbon with armchair or zigzag edges).
- You want to build a quantum dot (nanoparticle) model.
- You need 1D periodic structures for band structure or transport calculations.

## Method Selection

| Criterion | ASE nanotube builder | pymatgen (carve from bulk) | Manual construction |
|---|---|---|---|
| Carbon nanotubes | Built-in `nanotube()` | Not specialized | Complex |
| Nanowires from bulk | Not specialized | Carve cylinder from supercell | Carve + passivate |
| Nanoribbons | Not specialized | Carve from 2D sheet | ASE or manual |
| Quantum dots | Not specialized | Carve sphere from bulk | Carve + passivate |

## Prerequisites

- ASE (pre-installed)
- pymatgen (pre-installed)
- numpy, matplotlib (pre-installed)

---

## Detailed Steps

### Method A: Carbon Nanotubes (ASE)

```python
#!/usr/bin/env python3
"""
Build carbon nanotubes using ASE's built-in nanotube builder.
Covers single-wall CNTs with various chiralities (n,m).
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from ase.build import nanotube
from ase.io import write
from ase.visualize.plot import plot_atoms

# ============================================================
# 1. Single-Wall Carbon Nanotube (SWCNT)
# ============================================================
print("=== Single-Wall Carbon Nanotubes ===")

# CNT chirality (n, m):
#   (n, 0)   = zigzag
#   (n, n)   = armchair
#   (n, m)   = chiral (n != m, m != 0)

# Armchair (5, 5)
cnt_55 = nanotube(5, 5, length=4, bond=1.42, symbol="C")
print(f"  (5,5) armchair: {len(cnt_55)} atoms, "
      f"diameter = {cnt_55.cell[0, 0]:.2f} A")
write("CNT_5_5.cif", cnt_55)

# Zigzag (10, 0)
cnt_100 = nanotube(10, 0, length=4, bond=1.42, symbol="C")
print(f"  (10,0) zigzag:  {len(cnt_100)} atoms, "
      f"diameter = {cnt_100.cell[0, 0]:.2f} A")
write("CNT_10_0.cif", cnt_100)

# Chiral (6, 4)
cnt_64 = nanotube(6, 4, length=2, bond=1.42, symbol="C")
print(f"  (6,4) chiral:   {len(cnt_64)} atoms")
write("CNT_6_4.cif", cnt_64)

# ============================================================
# 2. CNT properties
# ============================================================
def cnt_properties(n, m, a_cc=1.42):
    """
    Compute CNT properties from chiral indices (n, m).

    Parameters
    ----------
    n, m : int
        Chiral indices.
    a_cc : float
        C-C bond length in Angstrom (default: 1.42 for graphene).

    Returns
    -------
    dict with diameter, chiral_angle, electronic_type, etc.
    """
    a = a_cc * np.sqrt(3)  # graphene lattice constant

    # Chiral vector length
    C = a * np.sqrt(n**2 + n*m + m**2)
    diameter = C / np.pi

    # Chiral angle
    theta = np.arctan2(np.sqrt(3) * m, 2 * n + m)
    theta_deg = np.degrees(theta)

    # Electronic type
    if (n - m) % 3 == 0:
        if n == m:
            e_type = "metallic (armchair)"
        else:
            e_type = "metallic"
    else:
        e_type = "semiconducting"

    # Translation vector length
    d_R = np.gcd(2*n + m, n + 2*m)
    T = np.sqrt(3) * C / d_R

    # Number of hexagons per unit cell
    N_hex = int(round(2 * (n**2 + n*m + m**2) / d_R))

    return {
        "n": n, "m": m,
        "diameter_A": diameter,
        "chiral_angle_deg": theta_deg,
        "electronic_type": e_type,
        "translation_period_A": T,
        "atoms_per_unit_cell": 2 * N_hex,
        "type": "armchair" if n == m else ("zigzag" if m == 0 else "chiral"),
    }

print("\n=== CNT Property Table ===")
cnts_to_check = [(5, 5), (10, 0), (6, 4), (8, 0), (9, 0), (10, 10), (7, 3)]
print(f"{'(n,m)':>8} {'Type':>12} {'d (A)':>8} {'theta':>8} {'Electronic':>20} {'Atoms/UC':>10}")
for n, m in cnts_to_check:
    props = cnt_properties(n, m)
    print(f"  ({n},{m})  {props['type']:>12} {props['diameter_A']:8.2f} "
          f"{props['chiral_angle_deg']:7.1f}  {props['electronic_type']:>20} "
          f"{props['atoms_per_unit_cell']:>10}")

# ============================================================
# 3. BN Nanotube
# ============================================================
print("\n=== BN Nanotube ===")

# ASE nanotube() only does single-element tubes.
# For BN, we build a C nanotube and replace alternating atoms.
cnt_bn = nanotube(6, 6, length=3, bond=1.44, symbol="C")

# In a (6,6) armchair tube, atoms alternate on A and B sublattices.
# Replace A-sublattice with B, B-sublattice with N.
positions = cnt_bn.positions.copy()
symbols = list(cnt_bn.get_chemical_symbols())

# Simple heuristic: alternate B and N along the circumference
for i in range(len(symbols)):
    if i % 2 == 0:
        symbols[i] = "B"
    else:
        symbols[i] = "N"

cnt_bn.set_chemical_symbols(symbols)
print(f"  BN (6,6) nanotube: {len(cnt_bn)} atoms, "
      f"{cnt_bn.get_chemical_formula()}")
write("BNNT_6_6.cif", cnt_bn)

# ============================================================
# 4. Add vacuum for periodic calculations
# ============================================================
print("\n=== Adding vacuum for periodic DFT ===")

cnt = nanotube(5, 5, length=4, bond=1.42, symbol="C")

# The nanotube is periodic along z (c axis).
# We need vacuum in x and y (a and b axes).
# ASE's nanotube() already puts the tube axis along z
# and sets a, b large enough to contain the tube.

# Ensure enough vacuum
cell = cnt.cell.copy()
positions = cnt.positions.copy()

# Center the tube and add vacuum
tube_radius = np.max(np.sqrt(positions[:, 0]**2 + positions[:, 1]**2))
vacuum = 15.0
new_a = 2 * (tube_radius + vacuum)

# Recenter
center_x = positions[:, 0].mean()
center_y = positions[:, 1].mean()
positions[:, 0] -= center_x - new_a / 2
positions[:, 1] -= center_y - new_a / 2

cnt.positions = positions
cnt.cell[0, 0] = new_a
cnt.cell[1, 1] = new_a

print(f"  Cell: a={cnt.cell[0,0]:.1f}, b={cnt.cell[1,1]:.1f}, "
      f"c={cnt.cell[2,2]:.2f} A")
print(f"  Tube radius: {tube_radius:.2f} A, vacuum: {vacuum:.1f} A")
write("CNT_5_5_with_vacuum.cif", cnt)

# ============================================================
# 5. Cross-section visualization
# ============================================================
fig, ax = plt.subplots(figsize=(6, 6))
ax.scatter(cnt.positions[:, 0], cnt.positions[:, 1], s=30, c="steelblue")
circle = plt.Circle((new_a/2, new_a/2), tube_radius, fill=False,
                      linestyle="--", color="red", linewidth=1.5)
ax.add_patch(circle)
ax.set_aspect("equal")
ax.set_xlabel("x (A)")
ax.set_ylabel("y (A)")
ax.set_title("CNT (5,5) cross-section")
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("cnt_cross_section.png", dpi=150)
print("  Saved: cnt_cross_section.png")
```

### Method B: Nanowires from Bulk (pymatgen/ASE)

```python
#!/usr/bin/env python3
"""
Build nanowires by carving a cylinder from a bulk crystal.
Also covers quantum dots (spherical carving).
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from pymatgen.core import Structure, Lattice
from pymatgen.io.vasp import Poscar
from pymatgen.io.ase import AseAtomsAdaptor
from ase.io import write as ase_write

# ============================================================
# 1. Build a large supercell
# ============================================================
# Si nanowire along [001]
bulk = Structure.from_spacegroup(
    "Fd-3m",
    lattice=Lattice.cubic(5.431),
    species=["Si"],
    coords=[[0.0, 0.0, 0.0]],
)

# Need a supercell large enough to carve from
sc = bulk.copy()
sc.make_supercell([8, 8, 2])  # Large in xy, periodic along z
print(f"Bulk supercell: {len(sc)} atoms")
print(f"  Cell: {np.round(sc.lattice.abc, 2)}")

# ============================================================
# 2. Carve nanowire (cylinder along z)
# ============================================================
def carve_nanowire(structure, radius, axis="c"):
    """
    Carve a nanowire from a supercell by removing atoms outside
    a cylinder of given radius centered on the cell center.

    Parameters
    ----------
    structure : Structure
        Large supercell.
    radius : float
        Nanowire radius in Angstrom.
    axis : str
        Periodic axis ('a', 'b', or 'c').

    Returns
    -------
    Structure
        Nanowire structure.
    """
    axis_map = {"a": 0, "b": 1, "c": 2}
    ax_idx = axis_map[axis]
    perp_idx = [i for i in range(3) if i != ax_idx]

    cart = structure.cart_coords
    center = np.mean(cart, axis=0)

    # Distance from axis in perpendicular plane
    r_perp = np.sqrt(
        (cart[:, perp_idx[0]] - center[perp_idx[0]])**2 +
        (cart[:, perp_idx[1]] - center[perp_idx[1]])**2
    )

    # Keep atoms inside cylinder
    keep = r_perp <= radius
    sites_to_remove = [i for i, k in enumerate(keep) if not k]

    nw = structure.copy()
    nw.remove_sites(sites_to_remove)

    # Add vacuum in perpendicular directions
    vacuum = 15.0
    cell = nw.lattice.matrix.copy()
    for idx in perp_idx:
        cell[idx] = cell[idx] / np.linalg.norm(cell[idx]) * (2 * radius + 2 * vacuum)

    # Recenter atoms
    new_coords = nw.cart_coords.copy()
    new_center = np.mean(new_coords, axis=0)
    for idx in perp_idx:
        new_coords[:, idx] -= new_center[idx]
        new_coords[:, idx] += (2 * radius + 2 * vacuum) / 2

    nw = Structure(
        Lattice(cell),
        nw.species,
        new_coords,
        coords_are_cartesian=True,
    )

    return nw

# Carve nanowires of different radii
for radius in [5.0, 8.0, 12.0]:
    nw = carve_nanowire(sc, radius=radius, axis="c")
    print(f"\n  Nanowire r={radius:.1f} A: {len(nw)} atoms")
    print(f"    Cell: {np.round(nw.lattice.abc, 2)}")
    nw.to(f"Si_nanowire_r{radius:.0f}.cif")
    Poscar(nw).write_file(f"POSCAR_nw_r{radius:.0f}")

# ============================================================
# 3. Quantum dot (spherical nanoparticle)
# ============================================================
print("\n=== Quantum Dot ===")

sc_qd = bulk.copy()
sc_qd.make_supercell([6, 6, 6])

def carve_quantum_dot(structure, radius):
    """
    Carve a quantum dot (sphere) from a supercell.
    """
    cart = structure.cart_coords
    center = np.mean(cart, axis=0)
    distances = np.linalg.norm(cart - center, axis=1)

    keep = distances <= radius
    sites_to_remove = [i for i, k in enumerate(keep) if not k]

    qd = structure.copy()
    qd.remove_sites(sites_to_remove)

    # Add vacuum box
    vacuum = 15.0
    box_size = 2 * radius + 2 * vacuum
    new_cell = np.eye(3) * box_size

    # Recenter
    new_coords = qd.cart_coords.copy()
    new_center = np.mean(new_coords, axis=0)
    new_coords -= new_center
    new_coords += box_size / 2

    qd = Structure(
        Lattice(new_cell),
        qd.species,
        new_coords,
        coords_are_cartesian=True,
    )

    return qd

for radius in [5.0, 8.0, 12.0]:
    qd = carve_quantum_dot(sc_qd, radius=radius)
    print(f"  Quantum dot r={radius:.0f} A: {len(qd)} atoms")
    qd.to(f"Si_QD_r{radius:.0f}.cif")

# ============================================================
# 4. Graphene nanoribbon
# ============================================================
print("\n=== Graphene Nanoribbons ===")

from ase.build import graphene_nanoribbon

# Armchair nanoribbon (AGNR)
agnr = graphene_nanoribbon(
    n=7,              # Width (number of dimer lines)
    m=1,              # Length (unit cells along ribbon)
    type="armchair",
    saturated=True,    # Passivate edges with H
    vacuum=15.0,
)
print(f"  7-AGNR: {len(agnr)} atoms, formula: {agnr.get_chemical_formula()}")
print(f"    Cell: {np.round(agnr.cell.lengths(), 2)}")
write("AGNR_7.cif", agnr)

# Zigzag nanoribbon (ZGNR)
zgnr = graphene_nanoribbon(
    n=6,
    m=1,
    type="zigzag",
    saturated=True,
    vacuum=15.0,
)
print(f"  6-ZGNR: {len(zgnr)} atoms, formula: {zgnr.get_chemical_formula()}")
write("ZGNR_6.cif", zgnr)

# Unsaturated (without H passivation)
zgnr_bare = graphene_nanoribbon(
    n=6, m=1, type="zigzag",
    saturated=False, vacuum=15.0,
)
print(f"  6-ZGNR (bare): {len(zgnr_bare)} atoms")
write("ZGNR_6_bare.cif", zgnr_bare)

# ============================================================
# 5. Visualize nanoribbon cross-section
# ============================================================
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

for ax, ribbon, title in [
    (axes[0], agnr, "7-AGNR"),
    (axes[1], zgnr, "6-ZGNR"),
]:
    pos = ribbon.positions
    symbols = ribbon.get_chemical_symbols()

    for sym, color, size in [("C", "black", 50), ("H", "lightgray", 20)]:
        mask = [s == sym for s in symbols]
        if any(mask):
            coords = pos[mask]
            ax.scatter(coords[:, 0], coords[:, 1], s=size, c=color,
                       label=sym, edgecolors="gray", linewidth=0.5)

    ax.set_title(title, fontsize=13)
    ax.set_xlabel("x (A)")
    ax.set_ylabel("y (A)")
    ax.set_aspect("equal")
    ax.legend()
    ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("nanoribbons.png", dpi=150)
print("  Saved: nanoribbons.png")

# ============================================================
# 6. Relax nanostructure with MACE
# ============================================================
print("\n=== MACE relaxation ===")

import warnings
warnings.filterwarnings("ignore")
from mace.calculators import mace_mp
from ase.optimize import BFGS

# Relax a nanoribbon
ribbon = graphene_nanoribbon(n=7, m=3, type="armchair",
                              saturated=True, vacuum=15.0)
calc = mace_mp(model="medium", dispersion=False, default_dtype="float64")
ribbon.calc = calc

opt = BFGS(ribbon, logfile="ribbon_relax.log")
opt.run(fmax=0.01, steps=200)

e_ribbon = ribbon.get_potential_energy()
print(f"  Relaxed 7-AGNR energy: {e_ribbon:.4f} eV ({len(ribbon)} atoms)")
write("AGNR_7_relaxed.cif", ribbon)
```

### Method C: VASP (Future External Access)

```python
#!/usr/bin/env python3
"""
Generate VASP input files for nanostructure calculations.
"""

import numpy as np
from pymatgen.core import Structure, Lattice
from pymatgen.io.vasp import Poscar, Incar, Kpoints
from ase.build import nanotube, graphene_nanoribbon
from ase.io import write
from pymatgen.io.ase import AseAtomsAdaptor

adaptor = AseAtomsAdaptor()

# ============================================================
# 1. CNT for VASP
# ============================================================
cnt = nanotube(5, 5, length=4, bond=1.42, symbol="C")

# Add vacuum
vacuum = 15.0
r_max = np.max(np.sqrt(cnt.positions[:, 0]**2 + cnt.positions[:, 1]**2))
new_a = 2 * (r_max + vacuum)
cnt.positions[:, 0] -= cnt.positions[:, 0].mean() - new_a / 2
cnt.positions[:, 1] -= cnt.positions[:, 1].mean() - new_a / 2
cnt.cell[0, 0] = new_a
cnt.cell[1, 1] = new_a

structure = adaptor.get_structure(cnt)
Poscar(structure).write_file("POSCAR_cnt")

# KPOINTS: 1x1xN (periodic only along tube axis)
Kpoints.gamma_automatic(kpts=(1, 1, 12)).write_file("KPOINTS_cnt")

# INCAR
incar = Incar({
    "SYSTEM": "CNT (5,5)",
    "ENCUT": 520,
    "PREC": "Accurate",
    "EDIFF": 1e-6,
    "EDIFFG": -0.01,
    "IBRION": 2,
    "NSW": 100,
    "ISIF": 2,       # Relax ions only
    "ISMEAR": 0,
    "SIGMA": 0.05,
    "LWAVE": False,
    "LCHARG": False,
})
incar.write_file("INCAR_cnt")

# ============================================================
# 2. Nanoribbon for VASP
# ============================================================
ribbon = graphene_nanoribbon(n=7, m=1, type="armchair",
                              saturated=True, vacuum=15.0)
structure_r = adaptor.get_structure(ribbon)
Poscar(structure_r).write_file("POSCAR_ribbon")

# KPOINTS: 1xNx1 or Nx1x1 depending on periodic axis
Kpoints.gamma_automatic(kpts=(1, 1, 12)).write_file("KPOINTS_ribbon")

print("VASP nanostructure files written:")
print("  CNT: POSCAR_cnt, KPOINTS_cnt, INCAR_cnt")
print("  Ribbon: POSCAR_ribbon, KPOINTS_ribbon")
print("  Key: ISIF=2, k-mesh dense only along periodic axis")
```

---

## Key Parameters

| Parameter | Typical Value | Notes |
|---|---|---|
| CNT chirality (n, m) | Varies | (n,n) = armchair, (n,0) = zigzag. (n-m) mod 3 = 0 for metallic. |
| C-C bond length | 1.42 A | Standard for sp2 carbon (graphene/CNT) |
| Nanowire radius | 5--20 A | Converge properties vs. radius |
| Vacuum | 15--20 A | Between periodic images in non-periodic directions |
| Edge passivation | H atoms | For nanoribbons/nanowires with dangling bonds |
| Nanoribbon width (n) | 3--15 | Number of dimer lines (AGNR) or zigzag chains (ZGNR) |
| SCF k-mesh | 1x1xN | Dense only along periodic axis for 1D systems |

## Common Issues

| Problem | Solution |
|---|---|
| ASE `nanotube()` only supports single elements | Build C nanotube first, then replace atoms for BN, MoS2, etc. |
| Nanowire surface has dangling bonds | Passivate with H atoms or reconstruct the surface. |
| Quantum dot is non-neutral (charged) | Ensure stoichiometry is maintained when carving. Adjust carve radius. |
| Nanoribbon edge states dominate band structure | This is physical for zigzag edges. Use armchair edges for gapped ribbons. |
| Periodic images interact | Increase vacuum to >= 15 A in non-periodic directions. |
| MACE gives poor results for nanostructure | Surfaces and edges may be outside MACE training data. Validate with DFT. |
