# 1D K-Path Generation

## When to Use

- You need a k-path for a 1D periodic structure (nanowire, nanotube, polymer chain, atomic chain).
- You are computing the electronic band structure of a system periodic along one axis only.
- You need the 1D Brillouin zone path from Gamma to X (or Z) along the periodic direction.
- You want to set up the k-mesh and band path for QE or VASP calculations of quasi-1D systems.

## Method Selection

| Criterion | Manual 1D path | seekpath (3D, adapted) |
|---|---|---|
| Applicability | Directly correct for 1D | Designed for 3D; includes spurious directions |
| Simplicity | Very simple (Gamma to X) | Overcomplicated for 1D |
| Best for | Nanowires, nanotubes, chains | Not recommended for 1D |

**Recommendation**: Always use the manual 1D k-path. The 1D BZ is simply a line segment from Gamma (0) to X (0.5) along the periodic axis. seekpath and pymatgen HighSymmKpath are not designed for 1D systems.

## Prerequisites

- pymatgen, numpy, matplotlib (pre-installed)
- A 1D structure file with periodicity along one axis and vacuum along the other two

---

## Detailed Steps

### Method A: Python 1D K-Path Generation

```python
#!/usr/bin/env python3
"""
Generate k-paths for 1D periodic structures (nanowires, nanotubes, chains).
The 1D BZ is a line segment: Gamma (0) --> X (pi/a) --> Gamma (-pi/a).

For a 1D system periodic along axis `periodic_axis` (default: c/z),
the k-path has only one component: k along that axis.
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pymatgen.core import Structure, Lattice

# ============================================================
# 1. Define 1D BZ high-symmetry points
# ============================================================
# For a 1D system periodic along c (third lattice vector):
# Gamma = (0, 0, 0), X = (0, 0, 0.5)
# If periodic along a: Gamma = (0, 0, 0), X = (0.5, 0, 0)

def get_1d_kpath(periodic_axis="c"):
    """
    Return the high-symmetry points and path for a 1D BZ.

    Parameters
    ----------
    periodic_axis : str
        Which lattice vector is periodic: 'a', 'b', or 'c'.

    Returns
    -------
    kpoints : dict
    path : list of tuples
    """
    axis_map = {"a": 0, "b": 1, "c": 2}
    idx = axis_map[periodic_axis]

    gamma = [0.0, 0.0, 0.0]
    x = [0.0, 0.0, 0.0]
    x[idx] = 0.5

    kpoints = {
        "GAMMA": gamma,
        "X": x,
    }

    # Standard 1D path: Gamma -> X
    # Optionally include -X -> Gamma -> X for full BZ
    path = [("GAMMA", "X")]

    return kpoints, path

# ============================================================
# 2. Build or load 1D structure
# ============================================================

# Example 1: Carbon chain (carbyne) periodic along c
a_vac = 15.0  # vacuum in a and b directions
c_chain = 2.56  # C-C bond length along chain * 2 (for alternating bonds)

carbyne = Structure(
    lattice=Lattice([[a_vac, 0, 0], [0, a_vac, 0], [0, 0, c_chain]]),
    species=["C", "C"],
    coords=[[0.5, 0.5, 0.0], [0.5, 0.5, 0.5]],
)
print(f"Structure: {carbyne.composition.reduced_formula}")
print(f"Periodic along c with period = {c_chain:.3f} A")
print(f"Vacuum: {a_vac:.1f} A in a and b directions")

# Example 2: GaN nanowire (would be loaded from file)
# nanowire = Structure.from_file("nanowire.cif")

structure = carbyne
periodic_axis = "c"  # the periodic direction
kpoints, path = get_1d_kpath(periodic_axis)

print(f"\n=== 1D BZ ===")
for label, coords in kpoints.items():
    print(f"  {label}: ({coords[0]:.4f}, {coords[1]:.4f}, {coords[2]:.4f})")
for s, e in path:
    print(f"  Path: {s} --> {e}")

# ============================================================
# 3. Output: QE K_POINTS card for 1D system
# ============================================================
def write_qe_1d_kpath(kpoints, path, npoints=80, filename="kpath_1d_qe.txt"):
    """
    Write QE K_POINTS {crystal_b} card for 1D band structure.
    Uses more points since there is only one segment.
    """
    klines = []
    for i, (start, end) in enumerate(path):
        sc = kpoints[start]
        ec = kpoints[end]

        if i == 0:
            klines.append(
                f"  {sc[0]:.10f}  {sc[1]:.10f}  {sc[2]:.10f}  {npoints}  ! {start}"
            )

        end_npts = 0  # last point
        klines.append(
            f"  {ec[0]:.10f}  {ec[1]:.10f}  {ec[2]:.10f}  {end_npts}  ! {end}"
        )

    content = f"K_POINTS {{crystal_b}}\n{len(klines)}\n" + "\n".join(klines) + "\n"
    with open(filename, "w") as f:
        f.write(content)
    print(f"\nQE K_POINTS card written to {filename}")
    print(content)
    return content

write_qe_1d_kpath(kpoints, path, npoints=80)

# Also write the SCF k-mesh recommendation
print("=== Recommended QE SCF k-mesh for 1D system ===")
print("K_POINTS automatic")
if periodic_axis == "c":
    print("  1 1 24  0 0 0")
elif periodic_axis == "b":
    print("  1 24 1  0 0 0")
else:
    print("  24 1 1  0 0 0")
print("(Dense mesh along periodic axis, 1 along vacuum directions)")

# ============================================================
# 4. Output: VASP KPOINTS files for 1D system
# ============================================================
def write_vasp_1d_kpoints(kpoints, path, npoints=80, periodic_axis="c",
                          filename_bands="KPOINTS_1d_bands",
                          filename_scf="KPOINTS_1d_scf"):
    """Write VASP KPOINTS for 1D band structure and SCF."""
    # Bands
    lines = [
        "K-path for 1D system",
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
    with open(filename_bands, "w") as f:
        f.write(content)
    print(f"VASP KPOINTS (bands) written to {filename_bands}")

    # SCF mesh
    axis_map = {"a": 0, "b": 1, "c": 2}
    mesh = [1, 1, 1]
    mesh[axis_map[periodic_axis]] = 24
    scf_content = f"""Gamma-centered mesh for 1D system
0
Gamma
  {mesh[0]} {mesh[1]} {mesh[2]}
  0  0  0
"""
    with open(filename_scf, "w") as f:
        f.write(scf_content)
    print(f"VASP KPOINTS (SCF) written to {filename_scf}")

write_vasp_1d_kpoints(kpoints, path, periodic_axis=periodic_axis)

# ============================================================
# 5. Output: Wannier90 kpath
# ============================================================
def write_wannier90_1d_kpath(kpoints, path, filename="wannier90_kpath_1d.txt"):
    """Write Wannier90 kpoint_path block for 1D system."""
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

write_wannier90_1d_kpath(kpoints, path)

# ============================================================
# 6. Visualize 1D BZ
# ============================================================
def plot_1d_bz(structure, periodic_axis="c", filename="brillouin_zone_1d.png"):
    """Plot the 1D Brillouin zone as a line segment."""
    axis_map = {"a": 0, "b": 1, "c": 2}
    idx = axis_map[periodic_axis]

    recip = structure.lattice.reciprocal_lattice.matrix
    b = recip[idx]
    b_length = np.linalg.norm(b)

    fig, ax = plt.subplots(figsize=(10, 3))

    # Draw BZ line
    ax.plot([-b_length/2, b_length/2], [0, 0], "k-", linewidth=3)

    # Mark high-symmetry points
    points = {
        r"$-X$": -b_length/2,
        r"$\Gamma$": 0.0,
        "$X$": b_length/2,
    }

    for label, pos in points.items():
        ax.plot(pos, 0, "ro", markersize=12, zorder=5)
        ax.annotate(label, (pos, 0), textcoords="offset points",
                    xytext=(0, 15), fontsize=14, ha="center", fontweight="bold")

    # Draw path
    ax.annotate("", xy=(b_length/2, -0.02), xytext=(0, -0.02),
                arrowprops=dict(arrowstyle="->", color="red", lw=2))
    ax.text(b_length/4, -0.05, "band path", fontsize=11, ha="center", color="red")

    ax.set_xlim(-b_length * 0.7, b_length * 0.7)
    ax.set_ylim(-0.1, 0.1)
    ax.set_xlabel(f"$k_{periodic_axis}$ (1/$\\AA$)", fontsize=13)
    ax.set_title(f"1D Brillouin Zone (periodic along {periodic_axis})", fontsize=14)
    ax.set_yticks([])
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(False)

    plt.tight_layout()
    plt.savefig(filename, dpi=200, bbox_inches="tight")
    print(f"\n1D BZ plot saved: {filename}")

plot_1d_bz(structure, periodic_axis=periodic_axis)

# ============================================================
# 7. Full 1D band path: -X -> Gamma -> X (both sides)
# ============================================================
print("\n=== Full 1D BZ path (-X -> Gamma -> X) ===")
axis_map = {"a": 0, "b": 1, "c": 2}
idx = axis_map[periodic_axis]

neg_x = [0.0, 0.0, 0.0]
neg_x[idx] = -0.5
gamma = [0.0, 0.0, 0.0]
pos_x = [0.0, 0.0, 0.0]
pos_x[idx] = 0.5

full_kpath = f"""K_POINTS {{crystal_b}}
3
  {neg_x[0]:.10f}  {neg_x[1]:.10f}  {neg_x[2]:.10f}  80  ! -X
  {gamma[0]:.10f}  {gamma[1]:.10f}  {gamma[2]:.10f}  80  ! GAMMA
  {pos_x[0]:.10f}  {pos_x[1]:.10f}  {pos_x[2]:.10f}   0  ! X
"""
print(full_kpath)

with open("kpath_1d_full_qe.txt", "w") as f:
    f.write(full_kpath)
print("Written to kpath_1d_full_qe.txt")
```

### Method B: QE Setup for 1D System

```python
#!/usr/bin/env python3
"""
Generate complete QE input files for a 1D system band structure.
Example: carbon chain (carbyne) periodic along c.
"""

import os
import numpy as np
from pymatgen.core import Structure, Lattice

# ============================================================
# Configuration
# ============================================================
PSEUDO_DIR = os.path.abspath("./pseudo")
OUTDIR = os.path.abspath("./tmp_1d")
os.makedirs(PSEUDO_DIR, exist_ok=True)
os.makedirs(OUTDIR, exist_ok=True)

vacuum = 15.0
c_period = 2.56

structure = Structure(
    lattice=Lattice([[vacuum, 0, 0], [0, vacuum, 0], [0, 0, c_period]]),
    species=["C", "C"],
    coords=[[0.5, 0.5, 0.0], [0.5, 0.5, 0.5]],
)

cell = structure.lattice.matrix
cell_lines = "\n".join(f"  {v[0]:.10f}  {v[1]:.10f}  {v[2]:.10f}" for v in cell)

# ============================================================
# SCF input
# ============================================================
scf_input = f"""&CONTROL
    calculation = 'scf'
    prefix      = 'chain1d'
    outdir      = '{OUTDIR}'
    pseudo_dir  = '{PSEUDO_DIR}'
    tprnfor     = .true.
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
    assume_isolated = '1D'
/
&ELECTRONS
    conv_thr = 1.0d-10
/

CELL_PARAMETERS angstrom
{cell_lines}

ATOMIC_SPECIES
  C  12.011  C.pbe-n-kjpaw_psl.1.0.0.UPF

ATOMIC_POSITIONS crystal
  C  0.5000000000  0.5000000000  0.0000000000
  C  0.5000000000  0.5000000000  0.5000000000

K_POINTS automatic
  1 1 24  0 0 0
"""

with open("chain1d_scf.in", "w") as f:
    f.write(scf_input)

# ============================================================
# Bands input
# ============================================================
bands_input = f"""&CONTROL
    calculation = 'bands'
    prefix      = 'chain1d'
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
    assume_isolated = '1D'
/
&ELECTRONS
    conv_thr = 1.0d-10
/

CELL_PARAMETERS angstrom
{cell_lines}

ATOMIC_SPECIES
  C  12.011  C.pbe-n-kjpaw_psl.1.0.0.UPF

ATOMIC_POSITIONS crystal
  C  0.5000000000  0.5000000000  0.0000000000
  C  0.5000000000  0.5000000000  0.5000000000

K_POINTS {{crystal_b}}
2
  0.0000000000  0.0000000000  0.0000000000  80  ! GAMMA
  0.0000000000  0.0000000000  0.5000000000   0  ! X
"""

with open("chain1d_bands.in", "w") as f:
    f.write(bands_input)

print("QE inputs for 1D system:")
print("  chain1d_scf.in   -- SCF with 1x1x24 k-mesh")
print("  chain1d_bands.in -- bands along Gamma-X (kz only)")
print("\nKey 1D settings:")
print("  assume_isolated = '1D' (Coulomb truncation for 1D)")
print("  K_POINTS: 1 1 N (dense only along periodic direction)")
```

### Method C: VASP (Future External Access)

```python
#!/usr/bin/env python3
"""
Generate VASP input files for 1D system band structure.
"""

from pymatgen.core import Structure, Lattice
from pymatgen.io.vasp import Poscar, Incar

vacuum = 15.0
c_period = 2.56

structure = Structure(
    lattice=Lattice([[vacuum, 0, 0], [0, vacuum, 0], [0, 0, c_period]]),
    species=["C", "C"],
    coords=[[0.5, 0.5, 0.0], [0.5, 0.5, 0.5]],
)

Poscar(structure).write_file("POSCAR_1d")

# SCF KPOINTS: 1x1xN
kpoints_scf = """Gamma mesh for 1D system
0
Gamma
  1 1 24
  0 0 0
"""
with open("KPOINTS_scf_1d", "w") as f:
    f.write(kpoints_scf)

# Band KPOINTS: Gamma -> X along c
kpoints_bands = """K-path for 1D system along c
80
Line-mode
Reciprocal
  0.0000  0.0000  0.0000  ! GAMMA
  0.0000  0.0000  0.5000  ! X
"""
with open("KPOINTS_bands_1d", "w") as f:
    f.write(kpoints_bands)

# INCAR for SCF
incar = Incar({
    "SYSTEM": "1D system SCF",
    "ENCUT": 520,
    "PREC": "Accurate",
    "EDIFF": 1e-7,
    "ISMEAR": 0,
    "SIGMA": 0.05,
    "NSW": 0,
    "LWAVE": True,
    "LCHARG": True,
})
incar.write_file("INCAR_scf_1d")

# INCAR for bands
incar_bands = Incar({
    "SYSTEM": "1D system bands",
    "ENCUT": 520,
    "PREC": "Accurate",
    "EDIFF": 1e-7,
    "ISMEAR": 0,
    "SIGMA": 0.05,
    "NSW": 0,
    "ICHARG": 11,
    "LORBIT": 11,
    "NBANDS": 24,
    "LWAVE": False,
    "LCHARG": False,
})
incar_bands.write_file("INCAR_bands_1d")

print("VASP inputs for 1D band structure:")
print("  POSCAR_1d, KPOINTS_scf_1d, KPOINTS_bands_1d")
print("  INCAR_scf_1d, INCAR_bands_1d")
```

---

## Key Parameters

| Parameter | Typical Value | Notes |
|---|---|---|
| Vacuum thickness | 12--20 A | Along both non-periodic directions. 15 A standard. |
| SCF k-mesh | 1 x 1 x N (N=20--30) | Dense only along the periodic axis |
| Band path points | 60--100 | Single segment Gamma-X, so use more points for smooth bands |
| `assume_isolated` (QE) | `'1D'` | Coulomb truncation for 1D periodicity |
| Periodic direction | Usually c | Convention: place the 1D periodic direction along c |

## Common Issues

| Problem | Solution |
|---|---|
| Bands show dispersion in non-periodic directions | Vacuum too small. Increase to >= 15 A. Ensure k-path only varies along periodic axis. |
| seekpath gives a 3D path | seekpath is for 3D crystals. Use the manual 1D path (Gamma -> X along periodic axis). |
| QE fails with `assume_isolated='1D'` | Ensure the periodic axis is c (third vector). QE's 1D truncation may require the wire along z. |
| Flat bands dominate | Normal for isolated 1D systems. Molecular-like states appear as flat bands. |
| Band crossings are missing | Increase npoints along the path to resolve fine features. |
