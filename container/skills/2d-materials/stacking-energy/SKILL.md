# Stacking-Dependent Potential Energy Surface

## When to Use

- Map the interlayer sliding energy landscape (gamma surface / GSFE) of a bilayer or multilayer 2D material
- Compute the stacking fault energy for different registry configurations (AA, AB, SP, etc.)
- Investigate bilayer registry preferences and moiré-scale energetics
- Determine the energy barrier for interlayer sliding -- relevant to tribology and lubrication
- Identify the most stable stacking order in van der Waals heterostructures
- Study commensurate-incommensurate transitions in twisted bilayers
- Corresponds to VASPKIT task 926

## Method Selection

| Criterion | ASE + MACE (Method A) | QE DFT (Method B) | VASP (Method C) |
|---|---|---|---|
| Speed | Very fast (~seconds per point) | Slow (~minutes per point) | Slow (~minutes per point) |
| Accuracy | MLIP quality (good trends, ~meV/atom) | Full DFT with vdW-DF | Full DFT with DFT-D3/optB88 |
| vdW treatment | Built-in dispersion (D3BJ via MACE-MP) | vdW-DF, rVV10, or DFT-D3 | IVDW = 11 (DFT-D3) or optB88 |
| Grid density | 20x20 or finer (trivial cost) | 10x10 typical (expensive) | 10x10 typical (expensive) |
| Best for | Rapid screening, large grids, many systems | Quantitative PES, benchmarking | Production PES, VASPKIT 926 workflow |
| Limitations | Accuracy depends on MACE training data | Computationally expensive | Requires VASP license |

```
Need a quick stacking energy landscape for screening?
  --> Method A (ASE + MACE): fast, captures qualitative trends

Need quantitative PES with first-principles accuracy?
  --> Method B (QE DFT) or Method C (VASP)

Need to use VASPKIT 926 workflow?
  --> Method C (VASP)

Which vdW correction?
  --> vdW-DF or rVV10 (QE) / optB88-vdW or DFT-D3 (VASP)
  --> Dispersion is essential for interlayer binding in 2D materials
```

## Prerequisites

- ASE, pymatgen, numpy, matplotlib -- all methods
- MACE (`mace-torch`) with `mace_mp` pretrained model -- Method A
- Quantum ESPRESSO (pw.x) with vdW-DF support -- Method B
- VASP with IVDW support -- Method C
- A relaxed 2D bilayer structure

## Detailed Steps

### Method A: ASE + MACE -- Stacking Energy Surface

#### Complete Workflow

```python
#!/usr/bin/env python3
"""
Compute the stacking-dependent potential energy surface (PES) for a
bilayer 2D material using ASE + MACE.

Workflow:
  1. Build a bilayer (e.g., bilayer graphene or MoS2)
  2. Define a grid of (dx, dy) displacements across one unit cell
  3. At each grid point, shift the top layer by (dx, dy), relax only
     the interlayer distance z, and record the total energy
  4. Plot the 2D contour map E(dx, dy) -- the gamma surface / GSFE

Example: Bilayer graphene.
"""

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from ase import Atoms
from ase.build import graphene
from ase.optimize import LBFGS
from ase.constraints import FixAtoms, FixedPlane
from mace.calculators import mace_mp
import json
import time

# ============================================================
# 1. Build bilayer graphene
# ============================================================
def build_bilayer_graphene(a=2.46, d_interlayer=3.35, vacuum=20.0):
    """
    Build AB-stacked bilayer graphene.

    Parameters
    ----------
    a : float
        In-plane lattice constant (Angstrom).
    d_interlayer : float
        Initial interlayer distance (Angstrom).
    vacuum : float
        Vacuum thickness above and below (Angstrom).

    Returns
    -------
    bilayer : ase.Atoms
        Bilayer graphene structure.
    n_bottom : int
        Number of atoms in the bottom layer.
    """
    # Bottom layer: two C atoms in hexagonal cell
    # Positions: (0, 0, 0) and (1/3, 2/3, 0) in fractional
    cell = [
        [a, 0.0, 0.0],
        [a * (-0.5), a * (np.sqrt(3) / 2), 0.0],
        [0.0, 0.0, d_interlayer + vacuum],
    ]

    z_bottom = vacuum / 2.0
    z_top = z_bottom + d_interlayer

    # Bottom layer atoms (A sublattice at origin, B sublattice at (1/3,2/3))
    pos_bottom = [
        [0.0, 0.0, z_bottom],
        [a / 3.0 + a * (-0.5) / 3.0 * 2, a * np.sqrt(3) / 3.0, z_bottom],
    ]

    # Top layer atoms (AB stacking: shifted by (a1+a2)/3)
    shift_x = (cell[0][0] + cell[1][0]) / 3.0
    shift_y = (cell[0][1] + cell[1][1]) / 3.0
    pos_top = [
        [shift_x, shift_y, z_top],
        [pos_bottom[1][0] + shift_x, pos_bottom[1][1] + shift_y, z_top],
    ]

    positions = pos_bottom + pos_top
    symbols = ["C"] * 4
    bilayer = Atoms(symbols=symbols, positions=positions, cell=cell, pbc=[True, True, False])
    return bilayer, 2  # 2 atoms in bottom layer


def build_bilayer_mos2(a=3.16, d_interlayer=6.15, vacuum=20.0):
    """
    Build 2H-stacked bilayer MoS2.

    In 2H stacking the top layer is rotated 180 degrees relative to the bottom,
    so that Mo atoms in the top layer sit above S atoms in the bottom layer.

    Parameters
    ----------
    a : float
        In-plane lattice constant (Angstrom).
    d_interlayer : float
        Mo-Mo interlayer distance (Angstrom).
    vacuum : float
        Vacuum thickness (Angstrom).

    Returns
    -------
    bilayer : ase.Atoms
        Bilayer MoS2 structure.
    n_bottom : int
        Number of atoms in the bottom layer.
    """
    cell = [
        [a, 0.0, 0.0],
        [a * (-0.5), a * (np.sqrt(3) / 2), 0.0],
        [0.0, 0.0, d_interlayer + vacuum],
    ]

    z_center = (d_interlayer + vacuum) / 2.0
    d_SeMo = 1.56  # S-Mo vertical distance in MoS2

    # Bottom layer: Mo at (1/3, 2/3), S above and below
    frac_Mo_bot = np.array([1.0 / 3, 2.0 / 3])
    frac_S_bot = np.array([2.0 / 3, 1.0 / 3])

    cart_Mo_bot = frac_Mo_bot[0] * np.array(cell[0][:2]) + frac_Mo_bot[1] * np.array(cell[1][:2])
    cart_S_bot = frac_S_bot[0] * np.array(cell[0][:2]) + frac_S_bot[1] * np.array(cell[1][:2])

    z_Mo_bot = z_center - d_interlayer / 2.0
    pos_bottom = [
        [cart_Mo_bot[0], cart_Mo_bot[1], z_Mo_bot],           # Mo
        [cart_S_bot[0], cart_S_bot[1], z_Mo_bot + d_SeMo],    # S top
        [cart_S_bot[0], cart_S_bot[1], z_Mo_bot - d_SeMo],    # S bottom
    ]

    # Top layer (2H): Mo at (2/3, 1/3), S at (1/3, 2/3)
    frac_Mo_top = np.array([2.0 / 3, 1.0 / 3])
    frac_S_top = np.array([1.0 / 3, 2.0 / 3])

    cart_Mo_top = frac_Mo_top[0] * np.array(cell[0][:2]) + frac_Mo_top[1] * np.array(cell[1][:2])
    cart_S_top = frac_S_top[0] * np.array(cell[0][:2]) + frac_S_top[1] * np.array(cell[1][:2])

    z_Mo_top = z_center + d_interlayer / 2.0
    pos_top = [
        [cart_Mo_top[0], cart_Mo_top[1], z_Mo_top],           # Mo
        [cart_S_top[0], cart_S_top[1], z_Mo_top + d_SeMo],    # S top
        [cart_S_top[0], cart_S_top[1], z_Mo_top - d_SeMo],    # S bottom
    ]

    positions = pos_bottom + pos_top
    symbols = ["Mo", "S", "S", "Mo", "S", "S"]
    bilayer = Atoms(symbols=symbols, positions=positions, cell=cell, pbc=[True, True, False])
    return bilayer, 3  # 3 atoms in bottom layer


# ============================================================
# 2. Set up MACE calculator
# ============================================================
calc = mace_mp(model="medium", dispersion=True, default_dtype="float64")

# ============================================================
# 3. Choose system and build bilayer
# ============================================================
SYSTEM = "graphene"  # Change to "MoS2" for bilayer MoS2

if SYSTEM == "graphene":
    bilayer, n_bottom = build_bilayer_graphene(a=2.46, d_interlayer=3.35, vacuum=20.0)
    high_symmetry_stackings = {
        "AA": (0.0, 0.0),
        "AB": (1.0 / 3, 2.0 / 3),
        "SP": (0.5, 0.0),
    }
elif SYSTEM == "MoS2":
    bilayer, n_bottom = build_bilayer_mos2(a=3.16, d_interlayer=6.15, vacuum=20.0)
    high_symmetry_stackings = {
        "AA'(2H)": (0.0, 0.0),
        "AA": (1.0 / 3, 1.0 / 3),
        "AB": (2.0 / 3, 1.0 / 3),
    }
else:
    raise ValueError(f"Unknown system: {SYSTEM}")

print(f"System: bilayer {SYSTEM}")
print(f"Number of atoms: {len(bilayer)}")
print(f"Bottom layer atoms: indices 0..{n_bottom - 1}")
print(f"Top layer atoms: indices {n_bottom}..{len(bilayer) - 1}")

# ============================================================
# 4. Compute stacking energy on a grid
# ============================================================
N_GRID = 21  # Grid points along each lattice vector direction

# Fractional displacements along a1 and a2
frac_a = np.linspace(0, 1, N_GRID, endpoint=False)
frac_b = np.linspace(0, 1, N_GRID, endpoint=False)

cell_2d = bilayer.cell[:2, :2]  # 2x2 matrix of in-plane lattice vectors

# Store results
energies = np.zeros((N_GRID, N_GRID))
interlayer_dists = np.zeros((N_GRID, N_GRID))

# Reference positions: save the original top-layer positions
pos_orig = bilayer.get_positions().copy()
top_indices = list(range(n_bottom, len(bilayer)))
bottom_indices = list(range(n_bottom))

# Reference energy: isolated bilayer at current stacking
bilayer.calc = calc
e_ref = bilayer.get_potential_energy()
n_formula = len(bilayer) // 2  # per formula unit normalization

print(f"\nComputing PES on {N_GRID}x{N_GRID} grid...")
t0 = time.time()

for i, fa in enumerate(frac_a):
    for j, fb in enumerate(frac_b):
        # Compute Cartesian displacement from fractional
        dx = fa * cell_2d[0, 0] + fb * cell_2d[1, 0]
        dy = fa * cell_2d[0, 1] + fb * cell_2d[1, 1]

        # Create a copy and shift top layer
        atoms = bilayer.copy()
        atoms.calc = calc
        pos = pos_orig.copy()
        for idx in top_indices:
            pos[idx, 0] += dx
            pos[idx, 1] += dy
        atoms.set_positions(pos)

        # Fix bottom layer completely; fix top layer xy, relax z only
        constraints = []
        constraints.append(FixAtoms(indices=bottom_indices))
        for idx in top_indices:
            constraints.append(FixedPlane(idx, [0, 0, 1]))
        # FixedPlane with normal [0,0,1] fixes z -- we want to relax z
        # Instead, fix x and y of top layer atoms using FixedPlane normals
        constraints = [FixAtoms(indices=bottom_indices)]
        for idx in top_indices:
            # Fix motion along x
            constraints.append(FixedPlane(idx, [1, 0, 0]))
            # Fix motion along y
            constraints.append(FixedPlane(idx, [0, 1, 0]))
        atoms.set_constraint(constraints)

        # Relax interlayer distance (z only)
        opt = LBFGS(atoms, logfile=None)
        try:
            opt.run(fmax=0.01, steps=100)
        except Exception:
            pass  # Use last geometry if optimization fails

        e = atoms.get_potential_energy()
        energies[i, j] = e

        # Compute average interlayer distance
        z_bottom_avg = np.mean([atoms.positions[k, 2] for k in bottom_indices])
        z_top_avg = np.mean([atoms.positions[k, 2] for k in top_indices])
        interlayer_dists[i, j] = z_top_avg - z_bottom_avg

    if (i + 1) % 5 == 0:
        print(f"  Row {i + 1}/{N_GRID} done ({time.time() - t0:.1f} s)")

elapsed = time.time() - t0
print(f"PES computation completed in {elapsed:.1f} s")

# ============================================================
# 5. Convert to meV/atom relative to minimum
# ============================================================
e_min = np.min(energies)
n_total = len(bilayer)
pes_meV = (energies - e_min) / n_total * 1000.0  # meV/atom

print(f"\nEnergy range: {np.min(pes_meV):.2f} -- {np.max(pes_meV):.2f} meV/atom")
print(f"Interlayer distance range: {np.min(interlayer_dists):.3f} -- "
      f"{np.max(interlayer_dists):.3f} A")

# ============================================================
# 6. Plot 2D contour map (Gamma Surface)
# ============================================================
fig, axes = plt.subplots(1, 2, figsize=(16, 7))

# Build Cartesian coordinate grid for plotting
X = np.zeros((N_GRID, N_GRID))
Y = np.zeros((N_GRID, N_GRID))
for i, fa in enumerate(frac_a):
    for j, fb in enumerate(frac_b):
        X[i, j] = fa * cell_2d[0, 0] + fb * cell_2d[1, 0]
        Y[i, j] = fa * cell_2d[0, 1] + fb * cell_2d[1, 1]

# -- Left panel: Stacking energy PES --
ax1 = axes[0]
levels = np.linspace(0, np.max(pes_meV), 30)
cf = ax1.contourf(X, Y, pes_meV, levels=levels, cmap="RdYlBu_r")
ax1.contour(X, Y, pes_meV, levels=levels, colors="k", linewidths=0.3, alpha=0.4)
cbar = fig.colorbar(cf, ax=ax1, label="Stacking energy (meV/atom)")

# Mark high-symmetry stackings
for label, (fa_hs, fb_hs) in high_symmetry_stackings.items():
    x_hs = fa_hs * cell_2d[0, 0] + fb_hs * cell_2d[1, 0]
    y_hs = fa_hs * cell_2d[0, 1] + fb_hs * cell_2d[1, 1]
    # Find energy at this stacking (nearest grid point)
    i_near = int(round(fa_hs * (N_GRID - 1))) % N_GRID
    j_near = int(round(fb_hs * (N_GRID - 1))) % N_GRID
    e_hs = pes_meV[i_near, j_near]
    ax1.plot(x_hs, y_hs, "ko", markersize=8)
    ax1.annotate(f"{label}\n({e_hs:.1f})",
                 (x_hs, y_hs), textcoords="offset points",
                 xytext=(10, 10), fontsize=10, fontweight="bold",
                 bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.8))

ax1.set_xlabel("x (A)", fontsize=12)
ax1.set_ylabel("y (A)", fontsize=12)
ax1.set_title(f"Stacking Energy PES -- Bilayer {SYSTEM}", fontsize=14)
ax1.set_aspect("equal")

# -- Right panel: Interlayer distance --
ax2 = axes[1]
levels_d = np.linspace(np.min(interlayer_dists), np.max(interlayer_dists), 30)
cf2 = ax2.contourf(X, Y, interlayer_dists, levels=levels_d, cmap="viridis")
ax2.contour(X, Y, interlayer_dists, levels=levels_d, colors="k", linewidths=0.3, alpha=0.4)
fig.colorbar(cf2, ax=ax2, label="Interlayer distance (A)")

for label, (fa_hs, fb_hs) in high_symmetry_stackings.items():
    x_hs = fa_hs * cell_2d[0, 0] + fb_hs * cell_2d[1, 0]
    y_hs = fa_hs * cell_2d[0, 1] + fb_hs * cell_2d[1, 1]
    ax2.plot(x_hs, y_hs, "wo", markersize=8, markeredgecolor="k")
    ax2.annotate(label, (x_hs, y_hs), textcoords="offset points",
                 xytext=(10, 10), fontsize=10, fontweight="bold",
                 bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.8))

ax2.set_xlabel("x (A)", fontsize=12)
ax2.set_ylabel("y (A)", fontsize=12)
ax2.set_title(f"Interlayer Distance -- Bilayer {SYSTEM}", fontsize=14)
ax2.set_aspect("equal")

fig.tight_layout()
fig.savefig("stacking_pes.png", dpi=200, bbox_inches="tight")
print("\nSaved stacking_pes.png")

# ============================================================
# 7. Print high-symmetry stacking energies
# ============================================================
print(f"\n{'='*60}")
print(f"HIGH-SYMMETRY STACKING ENERGIES -- Bilayer {SYSTEM}")
print(f"{'='*60}")
print(f"{'Stacking':<15} {'E (meV/atom)':<15} {'d_inter (A)':<15}")
print("-" * 45)
for label, (fa_hs, fb_hs) in high_symmetry_stackings.items():
    i_near = int(round(fa_hs * (N_GRID - 1))) % N_GRID
    j_near = int(round(fb_hs * (N_GRID - 1))) % N_GRID
    print(f"{label:<15} {pes_meV[i_near, j_near]:<15.3f} "
          f"{interlayer_dists[i_near, j_near]:<15.3f}")

# Sliding barrier: max energy along minimum energy path
print(f"\nMax stacking energy (barrier upper bound): {np.max(pes_meV):.3f} meV/atom")
print(f"Min stacking energy (ground state):        {np.min(pes_meV):.3f} meV/atom")

# ============================================================
# 8. Save results
# ============================================================
results = {
    "system": f"bilayer_{SYSTEM}",
    "grid_size": N_GRID,
    "e_min_eV": float(e_min),
    "pes_range_meV_per_atom": [float(np.min(pes_meV)), float(np.max(pes_meV))],
    "interlayer_distance_range_A": [
        float(np.min(interlayer_dists)),
        float(np.max(interlayer_dists)),
    ],
    "high_symmetry_stackings": {},
}

for label, (fa_hs, fb_hs) in high_symmetry_stackings.items():
    i_near = int(round(fa_hs * (N_GRID - 1))) % N_GRID
    j_near = int(round(fb_hs * (N_GRID - 1))) % N_GRID
    results["high_symmetry_stackings"][label] = {
        "frac_a": float(fa_hs),
        "frac_b": float(fb_hs),
        "energy_meV_per_atom": float(pes_meV[i_near, j_near]),
        "interlayer_distance_A": float(interlayer_dists[i_near, j_near]),
    }

with open("stacking_pes_results.json", "w") as f:
    json.dump(results, f, indent=2)
print("\nSaved stacking_pes_results.json")

# Save raw data as numpy arrays
np.savez(
    "stacking_pes_data.npz",
    frac_a=frac_a,
    frac_b=frac_b,
    X=X,
    Y=Y,
    pes_meV=pes_meV,
    interlayer_dists=interlayer_dists,
)
print("Saved stacking_pes_data.npz")
```

### Method B: QE DFT -- Stacking Energy Surface with vdW-DF

#### Step 1: Generate QE Inputs for a Grid of Stackings

```python
#!/usr/bin/env python3
"""
Generate Quantum ESPRESSO input files for a stacking-dependent PES
calculation using a vdW density functional.

Workflow:
  1. Build bilayer graphene at the reference stacking
  2. For each (dx, dy) grid point, generate a pw.x input with the
     top layer shifted
  3. Use vdW-DF2 (or rVV10) for proper interlayer interaction

Example: Bilayer graphene with vdW-DF2.
"""

import os
import numpy as np
from pymatgen.core import Structure, Lattice
from pymatgen.core.periodic_table import Element

# ============================================================
# Configuration
# ============================================================
PSEUDO_DIR = os.path.abspath("./pseudo")
BASE_OUTDIR = os.path.abspath("./tmp_stacking")
os.makedirs(PSEUDO_DIR, exist_ok=True)
os.makedirs(BASE_OUTDIR, exist_ok=True)

PREFIX = "blg"
ECUTWFC = 60.0   # Ry -- higher cutoff needed for vdW-DF
ECUTRHO = 600.0  # Ry
N_GRID = 11      # 11x11 grid (121 calculations)
A_LAT = 2.46     # graphene lattice constant (A)
D_INTER = 3.35   # interlayer distance (A)
VACUUM = 20.0    # vacuum (A)

pseudos = {
    "C": "C.pbe-n-kjpaw_psl.1.0.0.UPF",
}

# Download pseudopotentials
import subprocess
for el, pp in pseudos.items():
    pp_path = os.path.join(PSEUDO_DIR, pp)
    if not os.path.exists(pp_path):
        subprocess.run([
            "wget", "-q",
            f"https://pseudopotentials.quantum-espresso.org/upf_files/{pp}",
            "-O", pp_path
        ], check=True)

# ============================================================
# Build reference bilayer graphene (AB stacking)
# ============================================================
a = A_LAT
c = D_INTER + VACUUM

# Cell vectors
cell = np.array([
    [a, 0.0, 0.0],
    [-a / 2, a * np.sqrt(3) / 2, 0.0],
    [0.0, 0.0, c],
])

# Bottom layer positions (Cartesian)
z_bot = VACUUM / 2.0
z_top = z_bot + D_INTER

# Bottom layer: A sublattice (0,0) and B sublattice (1/3,2/3)
pos_bot_A = np.array([0.0, 0.0, z_bot])
pos_bot_B = (1.0 / 3) * cell[0] + (2.0 / 3) * cell[1]
pos_bot_B[2] = z_bot

# Top layer at reference AB stacking
shift_AB = (1.0 / 3) * cell[0] + (2.0 / 3) * cell[1]
pos_top_A = np.array([0.0, 0.0, z_top]) + np.array([shift_AB[0], shift_AB[1], 0.0])
pos_top_B = pos_bot_B.copy()
pos_top_B[2] = z_top
pos_top_B += np.array([shift_AB[0], shift_AB[1], 0.0])

positions_ref = np.array([pos_bot_A, pos_bot_B, pos_top_A, pos_top_B])
symbols = ["C"] * 4

# ============================================================
# Generate QE inputs for each grid point
# ============================================================
frac_a_grid = np.linspace(0, 1, N_GRID, endpoint=False)
frac_b_grid = np.linspace(0, 1, N_GRID, endpoint=False)

cell_2d = cell[:2, :2]

input_dirs = []

for i, fa in enumerate(frac_a_grid):
    for j, fb in enumerate(frac_b_grid):
        # Cartesian displacement from fractional shift
        dx = fa * cell_2d[0, 0] + fb * cell_2d[1, 0]
        dy = fa * cell_2d[0, 1] + fb * cell_2d[1, 1]

        # Shift top layer (indices 2, 3)
        positions = positions_ref.copy()
        positions[2, 0] += dx
        positions[2, 1] += dy
        positions[3, 0] += dx
        positions[3, 1] += dy

        # Convert to fractional coordinates
        frac_coords = np.linalg.solve(cell.T, positions.T).T

        # Create directory
        dirname = f"stacking_{i:02d}_{j:02d}"
        dirpath = os.path.join(BASE_OUTDIR, dirname)
        os.makedirs(dirpath, exist_ok=True)

        outdir = os.path.join(dirpath, "tmp")
        os.makedirs(outdir, exist_ok=True)

        # Write QE input
        qe_input = f"""&CONTROL
    calculation = 'relax'
    prefix      = '{PREFIX}'
    outdir      = '{outdir}'
    pseudo_dir  = '{PSEUDO_DIR}'
    tprnfor     = .true.
    forc_conv_thr = 1.0d-3
/
&SYSTEM
    ibrav       = 0
    nat         = 4
    ntyp        = 1
    ecutwfc     = {ECUTWFC}
    ecutrho     = {ECUTRHO}
    input_dft   = 'vdw-df2-b86r'
    occupations = 'smearing'
    smearing    = 'cold'
    degauss     = 0.005
    assume_isolated = '2D'
/
&ELECTRONS
    conv_thr = 1.0d-8
    mixing_beta = 0.3
/
&IONS
    ion_dynamics = 'bfgs'
/

ATOMIC_SPECIES
  C  12.0107  {pseudos["C"]}

CELL_PARAMETERS angstrom
  {cell[0, 0]:.10f}  {cell[0, 1]:.10f}  {cell[0, 2]:.10f}
  {cell[1, 0]:.10f}  {cell[1, 1]:.10f}  {cell[1, 2]:.10f}
  {cell[2, 0]:.10f}  {cell[2, 1]:.10f}  {cell[2, 2]:.10f}

ATOMIC_POSITIONS angstrom
"""
        # Bottom layer: fixed
        qe_input += f"  C  {positions[0, 0]:.10f}  {positions[0, 1]:.10f}  {positions[0, 2]:.10f}  0 0 0\n"
        qe_input += f"  C  {positions[1, 0]:.10f}  {positions[1, 1]:.10f}  {positions[1, 2]:.10f}  0 0 0\n"
        # Top layer: only z relaxed (fix x, y)
        qe_input += f"  C  {positions[2, 0]:.10f}  {positions[2, 1]:.10f}  {positions[2, 2]:.10f}  0 0 1\n"
        qe_input += f"  C  {positions[3, 0]:.10f}  {positions[3, 1]:.10f}  {positions[3, 2]:.10f}  0 0 1\n"

        qe_input += "\nK_POINTS automatic\n  18 18 1  0 0 0\n"

        input_file = os.path.join(dirpath, f"{PREFIX}.in")
        with open(input_file, "w") as f:
            f.write(qe_input)

        input_dirs.append(dirname)

print(f"Generated {len(input_dirs)} QE input directories in {BASE_OUTDIR}/")
print(f"Grid: {N_GRID} x {N_GRID} = {N_GRID**2} points")

# Write a batch run script
batch_script = f"""#!/bin/bash
# Run all stacking PES calculations
NPROC=$(nproc)
BASE="{BASE_OUTDIR}"

for dir in {BASE_OUTDIR}/stacking_*/; do
    name=$(basename "$dir")
    echo "Running $name ..."
    cd "$dir"
    mpirun --allow-run-as-root -np $NPROC pw.x -in {PREFIX}.in > {PREFIX}.out 2>&1
    cd -
done

echo "All stacking calculations completed."
"""

with open("run_stacking_qe.sh", "w") as f:
    f.write(batch_script)
os.chmod("run_stacking_qe.sh", 0o755)
print("Generated run_stacking_qe.sh")
```

#### Step 2: Run All QE Calculations

```bash
#!/bin/bash
# Run the batch stacking calculation
chmod +x run_stacking_qe.sh
./run_stacking_qe.sh
```

#### Step 3: Parse Energies and Plot PES

```python
#!/usr/bin/env python3
"""
Parse QE outputs from the stacking PES grid and plot the gamma surface.
"""

import os
import re
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import json

BASE_OUTDIR = os.path.abspath("./tmp_stacking")
PREFIX = "blg"
N_GRID = 11
A_LAT = 2.46

# Cell vectors for coordinate mapping
cell_2d = np.array([
    [A_LAT, 0.0],
    [-A_LAT / 2, A_LAT * np.sqrt(3) / 2],
])

frac_a_grid = np.linspace(0, 1, N_GRID, endpoint=False)
frac_b_grid = np.linspace(0, 1, N_GRID, endpoint=False)

# ============================================================
# Parse energies from QE outputs
# ============================================================
energies = np.full((N_GRID, N_GRID), np.nan)

for i in range(N_GRID):
    for j in range(N_GRID):
        outfile = os.path.join(BASE_OUTDIR, f"stacking_{i:02d}_{j:02d}", f"{PREFIX}.out")
        if not os.path.exists(outfile):
            print(f"WARNING: missing {outfile}")
            continue

        with open(outfile) as f:
            content = f.read()

        # Find the final total energy (last occurrence)
        matches = re.findall(r"!\s+total energy\s+=\s+([-\d.]+)\s+Ry", content)
        if matches:
            energies[i, j] = float(matches[-1]) * 13.605693123  # Ry -> eV
        else:
            print(f"WARNING: no energy found in {outfile}")

# Check for missing data
n_missing = np.sum(np.isnan(energies))
if n_missing > 0:
    print(f"WARNING: {n_missing} grid points have no energy data")

# ============================================================
# Convert to meV/atom relative to minimum
# ============================================================
e_min = np.nanmin(energies)
n_atoms = 4
pes_meV = (energies - e_min) / n_atoms * 1000.0

print(f"Energy range: {np.nanmin(pes_meV):.2f} -- {np.nanmax(pes_meV):.2f} meV/atom")

# ============================================================
# Build coordinate grid and plot
# ============================================================
X = np.zeros((N_GRID, N_GRID))
Y = np.zeros((N_GRID, N_GRID))
for i, fa in enumerate(frac_a_grid):
    for j, fb in enumerate(frac_b_grid):
        X[i, j] = fa * cell_2d[0, 0] + fb * cell_2d[1, 0]
        Y[i, j] = fa * cell_2d[0, 1] + fb * cell_2d[1, 1]

fig, ax = plt.subplots(figsize=(9, 8))

levels = np.linspace(0, np.nanmax(pes_meV), 30)
cf = ax.contourf(X, Y, pes_meV, levels=levels, cmap="RdYlBu_r")
ax.contour(X, Y, pes_meV, levels=levels, colors="k", linewidths=0.3, alpha=0.4)
fig.colorbar(cf, ax=ax, label="Stacking energy (meV/atom)", shrink=0.8)

# Mark high-symmetry points
hs_points = {
    "AA": (0.0, 0.0),
    "AB": (1.0 / 3, 2.0 / 3),
    "SP": (0.5, 0.0),
}
for label, (fa_hs, fb_hs) in hs_points.items():
    x_hs = fa_hs * cell_2d[0, 0] + fb_hs * cell_2d[1, 0]
    y_hs = fa_hs * cell_2d[0, 1] + fb_hs * cell_2d[1, 1]
    i_near = int(round(fa_hs * (N_GRID - 1))) % N_GRID
    j_near = int(round(fb_hs * (N_GRID - 1))) % N_GRID
    e_hs = pes_meV[i_near, j_near]
    ax.plot(x_hs, y_hs, "ko", markersize=8)
    ax.annotate(f"{label}\n({e_hs:.1f})", (x_hs, y_hs),
                textcoords="offset points", xytext=(10, 10), fontsize=11,
                fontweight="bold",
                bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.8))

ax.set_xlabel("x (A)", fontsize=13)
ax.set_ylabel("y (A)", fontsize=13)
ax.set_title("Stacking PES -- Bilayer Graphene (QE vdW-DF2)", fontsize=14)
ax.set_aspect("equal")

fig.tight_layout()
fig.savefig("stacking_pes_qe.png", dpi=200, bbox_inches="tight")
print("Saved stacking_pes_qe.png")

# Save results
results = {
    "method": "QE_vdW-DF2",
    "system": "bilayer_graphene",
    "grid_size": N_GRID,
    "e_min_eV": float(e_min),
    "pes_range_meV_per_atom": [float(np.nanmin(pes_meV)), float(np.nanmax(pes_meV))],
}
with open("stacking_pes_qe_results.json", "w") as f:
    json.dump(results, f, indent=2)
print("Saved stacking_pes_qe_results.json")
```

### Method C: VASP -- Stacking Energy Surface (VASPKIT 926)

#### Step 1: Prepare VASP Inputs

```python
#!/usr/bin/env python3
"""
Generate VASP input files for a stacking-dependent PES calculation.

Uses DFT-D3(BJ) for van der Waals corrections (IVDW = 12) and
generates a grid of POSCAR files with the top layer shifted.

The VASPKIT 926 task provides an automated workflow for this.
Alternatively, this script generates inputs for manual submission.

Example: Bilayer graphene.
"""

import os
import numpy as np
from pymatgen.core import Structure, Lattice
from pymatgen.io.vasp import Poscar, Incar, Kpoints

# ============================================================
# Configuration
# ============================================================
BASE_DIR = os.path.abspath("./vasp_stacking")
os.makedirs(BASE_DIR, exist_ok=True)
N_GRID = 11
A_LAT = 2.46
D_INTER = 3.35
VACUUM = 20.0

# ============================================================
# Build reference bilayer graphene
# ============================================================
a = A_LAT
c = D_INTER + VACUUM

lattice = Lattice(np.array([
    [a, 0.0, 0.0],
    [-a / 2, a * np.sqrt(3) / 2, 0.0],
    [0.0, 0.0, c],
]))

z_bot = VACUUM / (2.0 * c)
z_top = z_bot + D_INTER / c

# AB stacking reference in fractional coords
frac_coords_ref = np.array([
    [0.0, 0.0, z_bot],            # C1 bottom A
    [1.0 / 3, 2.0 / 3, z_bot],    # C2 bottom B
    [1.0 / 3, 2.0 / 3, z_top],    # C3 top A (AB shift)
    [2.0 / 3, 1.0 / 3, z_top],    # C4 top B (AB shift)
])

species = ["C"] * 4

# ============================================================
# INCAR with vdW correction
# ============================================================
incar_dict = {
    "SYSTEM": "bilayer_graphene_stacking",
    "ISTART": 0,
    "ICHARG": 2,
    "ENCUT": 520,
    "EDIFF": 1e-6,
    "EDIFFG": -0.01,
    "NSW": 50,
    "IBRION": 2,
    "ISIF": 0,        # Fix cell, relax ions
    "ISMEAR": 0,
    "SIGMA": 0.05,
    "IVDW": 12,        # DFT-D3(BJ) van der Waals correction
    "LREAL": "Auto",
    "PREC": "Accurate",
    "LWAVE": False,
    "LCHARG": False,
    "NPAR": 4,
}
incar = Incar(incar_dict)

# ============================================================
# KPOINTS
# ============================================================
kpoints = Kpoints.gamma_automatic(kpts=(18, 18, 1), shift=(0, 0, 0))

# ============================================================
# Generate POSCARs for each grid point
# ============================================================
frac_a_grid = np.linspace(0, 1, N_GRID, endpoint=False)
frac_b_grid = np.linspace(0, 1, N_GRID, endpoint=False)

# Selective dynamics: fix bottom layer, relax only z for top layer
selective_dynamics = [
    [False, False, False],  # C1 bottom - fixed
    [False, False, False],  # C2 bottom - fixed
    [False, False, True],   # C3 top - relax z only
    [False, False, True],   # C4 top - relax z only
]

for i, fa in enumerate(frac_a_grid):
    for j, fb in enumerate(frac_b_grid):
        # Shift top layer in fractional coordinates
        frac_coords = frac_coords_ref.copy()
        frac_coords[2, 0] += fa  # shift along a1
        frac_coords[2, 1] += fb  # shift along a2
        frac_coords[3, 0] += fa
        frac_coords[3, 1] += fb

        # Wrap to [0, 1)
        frac_coords = frac_coords % 1.0

        struct = Structure(lattice, species, frac_coords)

        # Create directory
        dirname = f"stacking_{i:02d}_{j:02d}"
        dirpath = os.path.join(BASE_DIR, dirname)
        os.makedirs(dirpath, exist_ok=True)

        # Write POSCAR with selective dynamics
        poscar = Poscar(struct, selective_dynamics=selective_dynamics)
        poscar.write_file(os.path.join(dirpath, "POSCAR"))

        # Write INCAR
        incar.write_file(os.path.join(dirpath, "INCAR"))

        # Write KPOINTS
        kpoints.write_file(os.path.join(dirpath, "KPOINTS"))

        # POTCAR must be generated separately (requires VASP pseudopotentials)
        # Typically: cat ~/VASP_PP/PBE/C/POTCAR > POTCAR

print(f"Generated {N_GRID**2} VASP input directories in {BASE_DIR}/")
print("Note: POTCAR files must be generated separately from your VASP PP library.")
print()
print("To use VASPKIT 926 instead of this manual approach:")
print("  1. Prepare a single relaxed bilayer POSCAR")
print("  2. Run: vaspkit -task 926")
print("  3. Follow the prompts to define the sliding grid")
print("  4. VASPKIT generates all POSCARs and a submission script")
```

#### Step 2: Run VASP Calculations

```bash
#!/bin/bash
# Run all VASP stacking calculations
BASE="./vasp_stacking"

for dir in ${BASE}/stacking_*/; do
    name=$(basename "$dir")
    echo "Running $name ..."
    cd "$dir"
    mpirun -np $(nproc) vasp_std > vasp.log 2>&1
    cd - > /dev/null
done

echo "All VASP stacking calculations completed."
```

#### Step 3: Parse VASP Energies and Plot

```python
#!/usr/bin/env python3
"""
Parse VASP OSZICAR files from the stacking grid and plot the PES.
"""

import os
import re
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import json

BASE_DIR = os.path.abspath("./vasp_stacking")
N_GRID = 11
A_LAT = 2.46
N_ATOMS = 4

cell_2d = np.array([
    [A_LAT, 0.0],
    [-A_LAT / 2, A_LAT * np.sqrt(3) / 2],
])

frac_a_grid = np.linspace(0, 1, N_GRID, endpoint=False)
frac_b_grid = np.linspace(0, 1, N_GRID, endpoint=False)

# Parse energies from OSZICAR
energies = np.full((N_GRID, N_GRID), np.nan)

for i in range(N_GRID):
    for j in range(N_GRID):
        oszicar = os.path.join(BASE_DIR, f"stacking_{i:02d}_{j:02d}", "OSZICAR")
        if not os.path.exists(oszicar):
            continue

        with open(oszicar) as f:
            lines = f.readlines()

        # Last line with "E0=" contains the converged energy
        for line in reversed(lines):
            m = re.search(r"E0=\s*([-\d.E+]+)", line)
            if m:
                energies[i, j] = float(m.group(1))
                break

# Relative energies in meV/atom
e_min = np.nanmin(energies)
pes_meV = (energies - e_min) / N_ATOMS * 1000.0

# Build coordinate grid
X = np.zeros((N_GRID, N_GRID))
Y = np.zeros((N_GRID, N_GRID))
for i, fa in enumerate(frac_a_grid):
    for j, fb in enumerate(frac_b_grid):
        X[i, j] = fa * cell_2d[0, 0] + fb * cell_2d[1, 0]
        Y[i, j] = fa * cell_2d[0, 1] + fb * cell_2d[1, 1]

# Plot
fig, ax = plt.subplots(figsize=(9, 8))
levels = np.linspace(0, np.nanmax(pes_meV), 30)
cf = ax.contourf(X, Y, pes_meV, levels=levels, cmap="RdYlBu_r")
ax.contour(X, Y, pes_meV, levels=levels, colors="k", linewidths=0.3, alpha=0.4)
fig.colorbar(cf, ax=ax, label="Stacking energy (meV/atom)", shrink=0.8)

# High-symmetry labels
hs_points = {"AA": (0.0, 0.0), "AB": (1.0 / 3, 2.0 / 3), "SP": (0.5, 0.0)}
for label, (fa_hs, fb_hs) in hs_points.items():
    x_hs = fa_hs * cell_2d[0, 0] + fb_hs * cell_2d[1, 0]
    y_hs = fa_hs * cell_2d[0, 1] + fb_hs * cell_2d[1, 1]
    ax.plot(x_hs, y_hs, "ko", markersize=8)
    ax.annotate(label, (x_hs, y_hs), textcoords="offset points",
                xytext=(10, 10), fontsize=11, fontweight="bold",
                bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.8))

ax.set_xlabel("x (A)", fontsize=13)
ax.set_ylabel("y (A)", fontsize=13)
ax.set_title("Stacking PES -- Bilayer Graphene (VASP DFT-D3)", fontsize=14)
ax.set_aspect("equal")
fig.tight_layout()
fig.savefig("stacking_pes_vasp.png", dpi=200, bbox_inches="tight")
print("Saved stacking_pes_vasp.png")

# Save JSON
results = {
    "method": "VASP_DFT-D3(BJ)",
    "system": "bilayer_graphene",
    "grid_size": N_GRID,
    "pes_range_meV_per_atom": [float(np.nanmin(pes_meV)), float(np.nanmax(pes_meV))],
}
with open("stacking_pes_vasp_results.json", "w") as f:
    json.dump(results, f, indent=2)
print("Saved stacking_pes_vasp_results.json")
```

#### VASPKIT 926 Automated Workflow

```
VASPKIT 926 provides an automated workflow for stacking-dependent PES:

1. Prepare a relaxed bilayer POSCAR with appropriate vacuum.

2. Run VASPKIT:
     vaspkit -task 926

3. VASPKIT prompts:
   - Select the two layers (top and bottom atom indices)
   - Define the sliding grid (N1 x N2 along a1 and a2)
   - Choose whether to relax interlayer distance

4. VASPKIT generates:
   - A directory for each grid point with POSCAR
   - A batch submission script
   - Post-processing scripts to collect energies and plot

5. After all calculations finish, run VASPKIT 926 again in
   post-processing mode to generate:
   - STACKING_PES.dat -- energy vs displacement data
   - STACKING_PES.png -- contour plot of the gamma surface

Note: VASPKIT 926 handles the grid generation, selective dynamics,
and post-processing automatically. The manual approach above gives
more control over vdW functional choice and plotting.
```

## Key Parameters

| Parameter | Recommended Value | Notes |
|---|---|---|
| Grid density | 11x11 (DFT), 21x21 (MACE) | Finer grid for smoother contours; MACE is cheap enough for dense grids |
| Vacuum thickness | >= 20 A | Must be large enough to avoid periodic image interaction |
| vdW correction | Essential | Use DFT-D3(BJ), vdW-DF2, rVV10, or optB88-vdW; PBE alone gives no binding |
| ecutwfc (QE) | 60 Ry | vdW-DF needs higher cutoff than standard PBE |
| K-points (in-plane) | 18x18x1 | Dense k-mesh for small unit cell; reduce for larger supercells |
| Selective dynamics | Fix bottom layer; relax z of top layer | Keeps xy displacement fixed, allows interlayer relaxation |
| ISIF (VASP) | 0 | Fix cell shape and volume; relax only ionic positions |
| IVDW (VASP) | 11 (D3) or 12 (D3BJ) | DFT-D3(BJ) recommended for layered materials |
| `assume_isolated = '2D'` (QE) | Recommended | 2D Coulomb cutoff reduces vacuum needed |
| MACE dispersion | `dispersion=True` | Adds D3BJ correction to MACE energies |

## Interpreting Results

1. **Ground-state stacking**: The global minimum of E(dx, dy) identifies the most stable stacking configuration. For bilayer graphene this is AB (Bernal) stacking; for MoS2 it is the 2H (AA') stacking.
2. **Energy scale**: Typical stacking energy differences are 1-20 meV/atom for graphene, 5-50 meV/atom for TMDs. If values are much larger, check vdW settings.
3. **Sliding barrier**: The saddle point energy along the minimum energy path between two equivalent ground states gives the interlayer sliding barrier -- relevant to tribology and superlubricity.
4. **AA stacking**: Usually the highest energy configuration (atoms directly atop each other); serves as a useful reference point.
5. **Symmetry check**: The PES should reflect the symmetry of the lattice (C3 or C6 for hexagonal). Asymmetric PES indicates an error in the geometry setup.
6. **Interlayer distance variation**: The relaxed interlayer distance varies with stacking; AA stacking typically has a larger interlayer gap than AB. A flat interlayer distance map suggests z was not relaxed.
7. **MACE vs DFT comparison**: MACE typically reproduces the correct qualitative topology of the PES (correct ground state, correct symmetry) but may differ by 20-50% in absolute energy scale compared to DFT with high-quality vdW functionals.

## Common Issues

| Issue | Solution |
|---|---|
| PES shows no energy variation | vdW correction is missing or disabled; PBE alone gives nearly zero interlayer binding |
| PES has wrong symmetry | Check that the top layer shift uses the correct lattice vectors, not Cartesian x/y |
| Energy scale is unreasonably large (>100 meV/atom) | Units error (Ry vs eV), or cell/atom count normalization is wrong |
| Interlayer distance does not relax | Check constraints: z of top-layer atoms must be free; bottom layer fixed |
| QE vdW-DF calculation is very slow | vdW-DF is computationally expensive; use `assume_isolated = '2D'` and reduce vacuum |
| MACE gives negative interlayer binding | Check `dispersion=True`; without D3 correction the MACE potential may not bind layers |
| Contour plot has artifacts at grid edges | The grid should span exactly one unit cell [0, 1) without including the endpoint |
| VASP does not converge for some stackings | Increase NELM, reduce SIGMA, or use different mixing (AMIX, BMIX) |
| VASPKIT 926 does not recognize layers | Ensure the two layers are clearly separated in z and the vacuum gap is unambiguous |
| PES not periodic at boundaries | Verify the grid does not include the endpoint (use `endpoint=False` in linspace) |
