# 2D Material K-Path Generation

## When to Use

- You need a k-path for a 2D material (monolayer, few-layer, or slab model).
- You are computing the electronic band structure of graphene, MoS2, h-BN, or other 2D materials.
- You need the 2D Brillouin zone high-symmetry points for hexagonal, square, or rectangular lattices.
- You want to avoid artifacts from the vacuum direction in a slab supercell.

## Method Selection

| Criterion | Manual 2D path | seekpath (3D, adapted) | pymatgen HighSymmKpath |
|---|---|---|---|
| BZ type | True 2D BZ (no kz) | 3D BZ of slab supercell | 3D BZ of slab supercell |
| Accuracy | Correct for 2D | Includes spurious kz points | Includes spurious kz points |
| Generality | Requires knowing the lattice type | Automatic | Automatic |
| Best for | Publication-quality 2D band structures | Quick check, if you filter kz=0 | Quick check |

**Recommendation**: For 2D materials, manually specify the k-path using known 2D BZ high-symmetry points. seekpath and pymatgen are designed for 3D crystals and may include irrelevant kz components for slab models.

## Prerequisites

- pymatgen, numpy, matplotlib (pre-installed)
- A 2D structure (monolayer CIF/POSCAR with vacuum along c, or built with pymatgen/ASE)
- Optional: `pip install seekpath` for comparison with 3D paths

---

## Detailed Steps

### Method A: Manual 2D K-Path (Python)

```python
#!/usr/bin/env python3
"""
Generate k-paths for 2D materials with proper 2D Brillouin zone handling.
Covers hexagonal, square, rectangular, centered rectangular, and oblique lattices.
Outputs for QE, VASP, and Wannier90.
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pymatgen.core import Structure, Lattice

# ============================================================
# 1. Define 2D high-symmetry points for common lattice types
# ============================================================

# All coordinates in fractional reciprocal space (kx, ky, kz=0)
# kz=0 because we only care about the in-plane BZ

KPOINTS_2D = {
    "hexagonal": {
        # Hexagonal BZ: graphene, h-BN, MoS2, etc.
        "GAMMA": [0.0, 0.0, 0.0],
        "M":     [0.5, 0.0, 0.0],
        "K":     [1/3, 1/3, 0.0],
        "K'":    [2/3, -1/3, 0.0],  # Time-reversed K
    },
    "square": {
        # Square BZ: square lattice materials
        "GAMMA": [0.0, 0.0, 0.0],
        "X":     [0.5, 0.0, 0.0],
        "M":     [0.5, 0.5, 0.0],
    },
    "rectangular": {
        # Rectangular BZ
        "GAMMA": [0.0, 0.0, 0.0],
        "X":     [0.5, 0.0, 0.0],
        "S":     [0.5, 0.5, 0.0],
        "Y":     [0.0, 0.5, 0.0],
    },
    "centered_rectangular": {
        # Centered rectangular (e.g., black phosphorus)
        "GAMMA": [0.0, 0.0, 0.0],
        "X":     [0.5, 0.0, 0.0],
        "S":     [0.5, 0.5, 0.0],
        "Y":     [0.0, 0.5, 0.0],
    },
    "oblique": {
        # General oblique lattice
        "GAMMA": [0.0, 0.0, 0.0],
        "X":     [0.5, 0.0, 0.0],
        "Y":     [0.0, 0.5, 0.0],
        "S":     [0.5, 0.5, 0.0],
    },
}

# Standard paths for each lattice type
PATHS_2D = {
    "hexagonal":            [("GAMMA", "M"), ("M", "K"), ("K", "GAMMA")],
    "square":               [("GAMMA", "X"), ("X", "M"), ("M", "GAMMA")],
    "rectangular":          [("GAMMA", "X"), ("X", "S"), ("S", "Y"), ("Y", "GAMMA")],
    "centered_rectangular": [("GAMMA", "X"), ("X", "S"), ("S", "Y"), ("Y", "GAMMA")],
    "oblique":              [("GAMMA", "X"), ("X", "S"), ("S", "Y"), ("Y", "GAMMA")],
}

# ============================================================
# 2. Detect 2D lattice type from structure
# ============================================================
def detect_2d_lattice_type(structure, tol=0.01):
    """
    Detect the 2D lattice type from a slab structure.
    Assumes the vacuum direction is along c (third lattice vector).
    Returns one of: hexagonal, square, rectangular, centered_rectangular, oblique.
    """
    a, b, c = structure.lattice.abc
    alpha, beta, gamma = structure.lattice.angles

    # Only use in-plane lattice parameters
    # For a slab, c is the vacuum direction
    ratio = a / b

    if abs(gamma - 120.0) < tol and abs(ratio - 1.0) < tol:
        return "hexagonal"
    elif abs(gamma - 90.0) < tol and abs(ratio - 1.0) < tol:
        return "square"
    elif abs(gamma - 90.0) < tol:
        return "rectangular"
    else:
        return "oblique"

# ============================================================
# 3. Build or load 2D structure
# ============================================================

# Example 1: Graphene
graphene = Structure(
    lattice=Lattice.hexagonal(2.46, 20.0),  # 20 A vacuum
    species=["C", "C"],
    coords=[[0.0, 0.0, 0.5], [1/3, 2/3, 0.5]],
)
print(f"Structure: {graphene.composition.reduced_formula}")

# Example 2: MoS2 monolayer
# mos2 = Structure(
#     lattice=Lattice.hexagonal(3.16, 20.0),
#     species=["Mo", "S", "S"],
#     coords=[
#         [0.0, 0.0, 0.5],
#         [1/3, 2/3, 0.5 + 1.58/20.0],
#         [1/3, 2/3, 0.5 - 1.58/20.0],
#     ],
# )

# Example 3: Load from file
# structure = Structure.from_file("POSCAR_2d")

structure = graphene
lattice_type = detect_2d_lattice_type(structure)
print(f"Detected 2D lattice type: {lattice_type}")

kpoints = KPOINTS_2D[lattice_type]
path = PATHS_2D[lattice_type]

print(f"\n=== 2D High-symmetry points ===")
for label, coords in kpoints.items():
    print(f"  {label:8s}: ({coords[0]:.4f}, {coords[1]:.4f}, {coords[2]:.4f})")

print(f"\n=== 2D Path ===")
for s, e in path:
    print(f"  {s} --> {e}")

# ============================================================
# 4. Output: QE K_POINTS {crystal_b} card
# ============================================================
def write_qe_2d_kpath(kpoints, path, npoints=40, filename="kpath_2d_qe.txt"):
    """Write QE K_POINTS card for 2D material."""
    klines = []
    for i, (start, end) in enumerate(path):
        sc = kpoints[start]
        ec = kpoints[end]

        if i == 0:
            klines.append(
                f"  {sc[0]:.10f}  {sc[1]:.10f}  {sc[2]:.10f}  {npoints}  ! {start}"
            )

        end_npts = npoints if i < len(path) - 1 else 0
        klines.append(
            f"  {ec[0]:.10f}  {ec[1]:.10f}  {ec[2]:.10f}  {end_npts}  ! {end}"
        )

    content = f"K_POINTS {{crystal_b}}\n{len(klines)}\n" + "\n".join(klines) + "\n"
    with open(filename, "w") as f:
        f.write(content)
    print(f"\nQE K_POINTS card written to {filename}")
    print(content)
    return content

write_qe_2d_kpath(kpoints, path, npoints=40)

# ============================================================
# 5. Output: VASP KPOINTS file
# ============================================================
def write_vasp_2d_kpoints(kpoints, path, npoints=40, filename="KPOINTS_2d"):
    """Write VASP KPOINTS file for 2D material (line mode)."""
    lines = [
        "K-path for 2D material",
        f"{npoints}",
        "Line-mode",
        "Reciprocal",
    ]

    for start, end in path:
        sc = kpoints[start]
        ec = kpoints[end]
        lines.append(f"  {sc[0]:.10f}  {sc[1]:.10f}  {sc[2]:.10f}  ! {start}")
        lines.append(f"  {ec[0]:.10f}  {ec[1]:.10f}  {ec[2]:.10f}  ! {end}")
        lines.append("")

    content = "\n".join(lines) + "\n"
    with open(filename, "w") as f:
        f.write(content)
    print(f"VASP KPOINTS written to {filename}")
    return content

write_vasp_2d_kpoints(kpoints, path, npoints=40)

# ============================================================
# 6. Output: Wannier90 kpath block
# ============================================================
def write_wannier90_2d_kpath(kpoints, path, filename="wannier90_kpath_2d.txt"):
    """Write Wannier90 kpoint_path block for 2D material."""
    lines = ["begin kpoint_path"]
    for start, end in path:
        sc = kpoints[start]
        ec = kpoints[end]
        lines.append(
            f"  {start:6s} {sc[0]:8.5f} {sc[1]:8.5f} {sc[2]:8.5f}  "
            f"{end:6s} {ec[0]:8.5f} {ec[1]:8.5f} {ec[2]:8.5f}"
        )
    lines.append("end kpoint_path")

    content = "\n".join(lines) + "\n"
    with open(filename, "w") as f:
        f.write(content)
    print(f"Wannier90 kpath written to {filename}")
    return content

write_wannier90_2d_kpath(kpoints, path)

# ============================================================
# 7. Visualize 2D Brillouin zone
# ============================================================
def plot_2d_bz(structure, kpoints, path, filename="brillouin_zone_2d.png"):
    """
    Plot the 2D Brillouin zone with high-symmetry points and paths.
    """
    # Get reciprocal lattice vectors (in-plane only)
    recip = structure.lattice.reciprocal_lattice.matrix
    b1 = recip[0, :2]  # in-plane components only
    b2 = recip[1, :2]

    fig, ax = plt.subplots(figsize=(8, 7))

    # Draw BZ boundary using Wigner-Seitz construction in 2D
    from scipy.spatial import Voronoi

    # Generate grid of reciprocal lattice points
    grid_2d = []
    for i in range(-2, 3):
        for j in range(-2, 3):
            grid_2d.append(i * b1 + j * b2)
    grid_2d = np.array(grid_2d)

    vor = Voronoi(grid_2d)
    gamma_idx = np.argmin(np.linalg.norm(grid_2d, axis=1))
    region = vor.regions[vor.point_region[gamma_idx]]

    if -1 not in region and len(region) > 0:
        bz_verts = vor.vertices[region]
        # Order by angle
        center = bz_verts.mean(axis=0)
        angles = np.arctan2(bz_verts[:, 1] - center[1], bz_verts[:, 0] - center[0])
        order = np.argsort(angles)
        bz_verts = bz_verts[order]

        # Draw BZ boundary
        bz_polygon = plt.Polygon(bz_verts, fill=True, facecolor="lightyellow",
                                  edgecolor="black", linewidth=2, alpha=0.5)
        ax.add_patch(bz_polygon)

    # Convert k-points to Cartesian 2D
    b_full = recip[:2, :2]  # 2x2 matrix of b1, b2 in-plane

    cart_kpoints = {}
    for label, frac in kpoints.items():
        cart = frac[0] * b1 + frac[1] * b2
        cart_kpoints[label] = cart

    # Draw path segments
    for start, end in path:
        sc = cart_kpoints[start]
        ec = cart_kpoints[end]
        ax.plot([sc[0], ec[0]], [sc[1], ec[1]], "r-", linewidth=2.5, alpha=0.7)

    # Draw high-symmetry points
    for label, cart in cart_kpoints.items():
        ax.scatter(cart[0], cart[1], s=80, c="red", zorder=5, edgecolors="black")
        display = label.replace("GAMMA", r"$\Gamma$").replace("K'", "K'")
        ax.annotate(display, (cart[0], cart[1]),
                    textcoords="offset points", xytext=(8, 8),
                    fontsize=14, fontweight="bold")

    # Draw reciprocal lattice vectors
    ax.annotate("", xy=b1, xytext=(0, 0),
                arrowprops=dict(arrowstyle="->", color="blue", lw=2))
    ax.text(b1[0] * 1.1, b1[1] * 1.1, r"$\mathbf{b}_1$", fontsize=13, color="blue")
    ax.annotate("", xy=b2, xytext=(0, 0),
                arrowprops=dict(arrowstyle="->", color="blue", lw=2))
    ax.text(b2[0] * 1.1, b2[1] * 1.1, r"$\mathbf{b}_2$", fontsize=13, color="blue")

    ax.set_aspect("equal")
    ax.set_xlabel(r"$k_x$ (1/$\AA$)", fontsize=13)
    ax.set_ylabel(r"$k_y$ (1/$\AA$)", fontsize=13)
    ax.set_title("2D Brillouin Zone", fontsize=15)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(filename, dpi=200, bbox_inches="tight")
    print(f"\n2D BZ plot saved: {filename}")

plot_2d_bz(structure, kpoints, path)

# ============================================================
# 8. Special: Extended hexagonal path with K' and M'
# ============================================================
print("\n=== Extended hexagonal path (with Dirac cones at K and K') ===")

hex_extended_kpoints = {
    "GAMMA": [0.0, 0.0, 0.0],
    "M":     [0.5, 0.0, 0.0],
    "K":     [1/3, 1/3, 0.0],
    "K'":    [2/3, -1/3, 0.0],
}
hex_extended_path = [
    ("GAMMA", "M"), ("M", "K"), ("K", "GAMMA"),
    ("GAMMA", "K'"),
]

for s, e in hex_extended_path:
    sc = hex_extended_kpoints[s]
    ec = hex_extended_kpoints[e]
    print(f"  {s} ({sc[0]:.4f} {sc[1]:.4f} {sc[2]:.4f}) --> "
          f"{e} ({ec[0]:.4f} {ec[1]:.4f} {ec[2]:.4f})")

# ============================================================
# 9. QE K_POINTS for 2D slab: use only in-plane k-mesh for SCF
# ============================================================
print("\n=== Recommended QE k-mesh for 2D slab SCF ===")
print("K_POINTS automatic")
print("  12 12 1  0 0 0")
print("\nNote: Use 1 k-point along c (vacuum direction) for slab models.")
print("The k-path for bands should have kz=0 for all points.")

# ============================================================
# 10. Save summary
# ============================================================
print("\n" + "=" * 60)
print("GENERATED FILES")
print("=" * 60)
print("  kpath_2d_qe.txt         -- QE K_POINTS {crystal_b}")
print("  KPOINTS_2d              -- VASP KPOINTS (line mode)")
print("  wannier90_kpath_2d.txt  -- Wannier90 kpoint_path block")
print("  brillouin_zone_2d.png   -- 2D BZ visualization")
```

### Method B: QE Band Structure for 2D Material

```python
#!/usr/bin/env python3
"""
Complete QE band structure workflow for a 2D material (graphene example).
Key difference from 3D: use kz=0 for all k-points, 1 k-point along c for SCF.
"""

import os
import subprocess
import numpy as np
from pymatgen.core import Structure, Lattice

# ============================================================
# 1. Build graphene structure with vacuum
# ============================================================
vacuum = 20.0  # Angstrom
a_graphene = 2.46

graphene = Structure(
    lattice=Lattice.hexagonal(a_graphene, vacuum),
    species=["C", "C"],
    coords=[[0.0, 0.0, 0.5], [1/3, 2/3, 0.5]],
)
graphene.to("graphene.cif")
print(f"Graphene: {graphene.composition}, vacuum = {vacuum} A")

# ============================================================
# 2. Generate QE inputs
# ============================================================
PSEUDO_DIR = os.path.abspath("./pseudo")
OUTDIR = os.path.abspath("./tmp_graphene")
os.makedirs(PSEUDO_DIR, exist_ok=True)
os.makedirs(OUTDIR, exist_ok=True)

# Cell parameters
cell = graphene.lattice.matrix
cell_lines = "\n".join(
    f"  {v[0]:.10f}  {v[1]:.10f}  {v[2]:.10f}" for v in cell
)

# SCF input -- note K_POINTS: Nx Ny 1 (only 1 along vacuum direction)
scf_input = f"""&CONTROL
    calculation = 'scf'
    prefix      = 'graphene'
    outdir      = '{OUTDIR}'
    pseudo_dir  = '{PSEUDO_DIR}'
    tprnfor     = .true.
    tstress     = .true.
/
&SYSTEM
    ibrav       = 0
    nat         = 2
    ntyp        = 1
    ecutwfc     = 60.0
    ecutrho     = 480.0
    occupations = 'smearing'
    smearing    = 'mv'
    degauss     = 0.02
    assume_isolated = '2D'
/
&ELECTRONS
    conv_thr = 1.0d-10
/

CELL_PARAMETERS angstrom
{cell_lines}

ATOMIC_SPECIES
  C  12.011  C.pbe-n-kjpaw_psl.1.0.0.UPF

ATOMIC_POSITIONS crystal
  C  0.0000000000  0.0000000000  0.5000000000
  C  0.3333333333  0.6666666667  0.5000000000

K_POINTS automatic
  18 18 1  0 0 0
"""

with open("graphene_scf.in", "w") as f:
    f.write(scf_input)

# Bands input -- 2D k-path: GAMMA-M-K-GAMMA, all kz=0
kpath_2d = """K_POINTS {crystal_b}
4
  0.0000000000  0.0000000000  0.0000000000  40  ! GAMMA
  0.5000000000  0.0000000000  0.0000000000  40  ! M
  0.3333333333  0.3333333333  0.0000000000  40  ! K
  0.0000000000  0.0000000000  0.0000000000   0  ! GAMMA
"""

bands_input = f"""&CONTROL
    calculation = 'bands'
    prefix      = 'graphene'
    outdir      = '{OUTDIR}'
    pseudo_dir  = '{PSEUDO_DIR}'
    verbosity   = 'high'
/
&SYSTEM
    ibrav       = 0
    nat         = 2
    ntyp        = 1
    ecutwfc     = 60.0
    ecutrho     = 480.0
    nbnd        = 12
    assume_isolated = '2D'
/
&ELECTRONS
    conv_thr = 1.0d-10
/

CELL_PARAMETERS angstrom
{cell_lines}

ATOMIC_SPECIES
  C  12.011  C.pbe-n-kjpaw_psl.1.0.0.UPF

ATOMIC_POSITIONS crystal
  C  0.0000000000  0.0000000000  0.5000000000
  C  0.3333333333  0.6666666667  0.5000000000

{kpath_2d}
"""

with open("graphene_bands.in", "w") as f:
    f.write(bands_input)

print("QE input files written:")
print("  graphene_scf.in   -- SCF with 18x18x1 k-mesh")
print("  graphene_bands.in -- bands along GAMMA-M-K-GAMMA (kz=0)")
print("\nKey 2D settings:")
print("  K_POINTS: Nx Ny 1 (1 point along vacuum)")
print("  assume_isolated = '2D' (Coulomb truncation)")
print("  All k-path points have kz = 0")
```

### Method C: VASP (Future External Access)

```python
#!/usr/bin/env python3
"""
Generate VASP input files for 2D material band structure.
Key settings for 2D: IVDW for vdW, appropriate KPOINTS.
"""

from pymatgen.core import Structure, Lattice
from pymatgen.io.vasp import Poscar, Incar

# Build 2D structure
graphene = Structure(
    lattice=Lattice.hexagonal(2.46, 20.0),
    species=["C", "C"],
    coords=[[0.0, 0.0, 0.5], [1/3, 2/3, 0.5]],
)

# POSCAR
Poscar(graphene).write_file("POSCAR_2d")

# KPOINTS for SCF (2D: only 1 along c)
kpoints_scf = """Gamma-centered mesh for 2D material
0
Gamma
  18 18 1
  0  0  0
"""
with open("KPOINTS_scf_2d", "w") as f:
    f.write(kpoints_scf)

# KPOINTS for bands (line mode, kz=0)
kpoints_bands = """K-path for 2D hexagonal BZ
40
Line-mode
Reciprocal
  0.0000  0.0000  0.0000  ! GAMMA
  0.5000  0.0000  0.0000  ! M

  0.5000  0.0000  0.0000  ! M
  0.3333  0.3333  0.0000  ! K

  0.3333  0.3333  0.0000  ! K
  0.0000  0.0000  0.0000  ! GAMMA
"""
with open("KPOINTS_bands_2d", "w") as f:
    f.write(kpoints_bands)

# INCAR for SCF
incar_scf = Incar({
    "SYSTEM": "Graphene SCF",
    "ENCUT": 520,
    "PREC": "Accurate",
    "EDIFF": 1e-7,
    "ISMEAR": 0,
    "SIGMA": 0.05,
    "IBRION": -1,
    "NSW": 0,
    "LWAVE": True,
    "LCHARG": True,
    "LVDW": True,       # Enable vdW correction
    "IVDW": 12,          # DFT-D3(BJ) for 2D materials
    "LDIPOL": True,       # Dipole correction
    "IDIPOL": 3,          # Along c (vacuum direction)
})
incar_scf.write_file("INCAR_scf_2d")

# INCAR for bands
incar_bands = Incar({
    "SYSTEM": "Graphene bands",
    "ENCUT": 520,
    "PREC": "Accurate",
    "EDIFF": 1e-7,
    "ISMEAR": 0,
    "SIGMA": 0.05,
    "IBRION": -1,
    "NSW": 0,
    "ICHARG": 11,
    "LORBIT": 11,
    "LWAVE": False,
    "LCHARG": False,
    "NBANDS": 24,
})
incar_bands.write_file("INCAR_bands_2d")

print("VASP input files for 2D band structure generated:")
print("  POSCAR_2d, KPOINTS_scf_2d, KPOINTS_bands_2d")
print("  INCAR_scf_2d, INCAR_bands_2d")
print("\nWorkflow:")
print("  1. SCF: use KPOINTS_scf_2d (18x18x1 mesh)")
print("  2. Bands: copy CHGCAR, use KPOINTS_bands_2d + INCAR_bands_2d")
```

---

## Key Parameters

| Parameter | Typical Value | Notes |
|---|---|---|
| Vacuum thickness | 15--25 A | Minimum 15 A to avoid periodic image interaction. 20 A is standard. |
| kz in k-path | 0.0 for all points | 2D materials have no dispersion along kz |
| SCF k-mesh kz | 1 | Only 1 k-point along vacuum direction (e.g., 18x18x1) |
| `assume_isolated` (QE) | `'2D'` | Coulomb truncation for 2D; avoids spurious interlayer interaction |
| `LDIPOL` / `IDIPOL` (VASP) | True / 3 | Dipole correction along vacuum for asymmetric slabs |
| npoints per segment | 30--50 | More for fine features (e.g., Dirac cone at K) |

## Common Issues

| Problem | Solution |
|---|---|
| Bands show dispersion along kz | Vacuum is too thin. Increase vacuum to >= 20 A. Ensure all k-path points have kz=0. |
| seekpath gives 3D path with kz != 0 | seekpath treats the slab as 3D. Use manual 2D k-paths instead. |
| Dirac point not at K for graphene | Check lattice vectors match hexagonal convention. Ensure Gamma-M-K path is correct. |
| QE: convergence issues for 2D | Add `assume_isolated = '2D'` to avoid artificial Coulomb interaction between periodic images. |
| VASP: incorrect vacuum potential | Enable `LDIPOL=True, IDIPOL=3` for dipole correction along vacuum direction. |
| Wrong lattice type detected | Verify the structure has vacuum along c and in-plane vectors match expected symmetry. |
