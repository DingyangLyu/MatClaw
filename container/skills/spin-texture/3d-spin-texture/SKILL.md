# 3D Spin Texture Calculation

## When to Use

- You need to compute the spin expectation values on a 3D Fermi surface or constant-energy surface in a bulk material with spin-orbit coupling.
- You are studying topological insulator bulk bands (e.g., Bi2Se3, Bi2Te3) and need the spin polarization of bulk states throughout the 3D Brillouin zone.
- You are investigating Weyl semimetals (e.g., TaAs, NbAs) and want to visualize spin-momentum locking on Fermi arcs or Weyl cones in full 3D k-space.
- You are analyzing heavy fermion systems (e.g., CeCoIn5, SmB6) or heavy-element metals (e.g., Pt, Au, W) where strong SOC produces complex 3D spin textures on Fermi surfaces.
- You need to characterize Rashba or Dresselhaus spin splitting in bulk non-centrosymmetric crystals (e.g., BiTeI bulk, GeTe).
- You want to compare computed 3D spin textures with spin-resolved ARPES or de Haas-van Alphen experiments.

## Method Selection

| Criterion | QE DFT | VASP DFT |
|---|---|---|
| SOC setup | `noncolin=.true.`, `lspinorb=.true.` | `LSORBIT=.TRUE.`, `LNONCOLLINEAR=.TRUE.` |
| Spin extraction | `projwfc.x` with `lsym=.false.` or parse `pw.x` output (`verbosity='high'`) | Parse PROCAR for Sx, Sy, Sz per band per k-point |
| Pseudopotentials | Fully relativistic (`*_rel_*` or `*_FR_*`) | PAW potentials (standard, SOC handled internally) |
| 3D k-mesh | Dense uniform 3D grid or custom k-list via `K_POINTS (crystal)` | Dense Gamma-centered mesh or explicit k-list with `KPOINTS` |
| Output format | Text output parsed with Python | PROCAR file (fixed-format, machine-readable) |
| VASPKIT task | N/A | Menu 65 (tasks 654--656): 3D Fermi surface spin texture |
| Isosurface tools | Python (matplotlib, plotly, mayavi) | VASPKIT + Python post-processing |

## Prerequisites

- Quantum ESPRESSO 7.x (`pw.x`, `projwfc.x`) with noncollinear + SOC support
- Fully relativistic pseudopotentials (FR-ONCV or PSlibrary `_rel` PPs)
- Python packages: `numpy`, `matplotlib`, `scipy`, `pymatgen`, `ase`
- Optional: `plotly` or `mayavi` for interactive 3D visualization
- A relaxed bulk crystal structure (no vacuum needed -- this is a 3D periodic system)
- For VASP: PROCAR file from a SOC NSCF calculation with `LORBIT=11` or `LORBIT=12`
- For VASPKIT: version >= 1.3.0 for tasks 654--656

---

## Detailed Steps

### Method A: QE DFT -- 3D Spin Texture of Bulk Bi2Se3

Bi2Se3 is a prototypical 3D topological insulator with strong SOC. Its bulk bands exhibit spin-polarized Dirac-cone surface states, and the bulk states themselves carry nontrivial spin textures due to the rhombohedral crystal field and SOC. This example computes the spin expectation values on a dense 3D k-grid and visualizes the spin texture on a constant-energy isosurface.

#### Step A1: SCF Calculation with SOC

```python
#!/usr/bin/env python3
"""
Step 1: SCF calculation for bulk Bi2Se3 with spin-orbit coupling.
Bi2Se3 has a rhombohedral structure (space group R-3m, #166).
Uses the hexagonal conventional cell (3 formula units, 15 atoms).
"""
import os
import subprocess
import numpy as np

PSEUDO_DIR = os.path.abspath("./pseudo")
OUTDIR = os.path.abspath("./tmp_spintex_3d")
os.makedirs(OUTDIR, exist_ok=True)
PREFIX = "bi2se3_soc"

# Bi2Se3 hexagonal cell parameters (experimental)
a = 4.138   # Angstrom (in-plane)
c = 28.636  # Angstrom (out-of-plane, 3 quintuple layers)

# Atomic positions in fractional coordinates (hexagonal setting)
# 5-atom primitive rhombohedral cell (1 formula unit)
# Using the primitive rhombohedral cell for efficiency
a_rhomb = 9.841   # rhombohedral lattice parameter (Angstrom)
alpha_deg = 24.27  # rhombohedral angle (degrees)
alpha = np.radians(alpha_deg)

# Convert rhombohedral to Cartesian vectors
cos_a = np.cos(alpha)
sin_a = np.sin(alpha)
a1_len = a_rhomb

scf_input = f"""&CONTROL
    calculation  = 'scf'
    prefix       = '{PREFIX}'
    outdir       = '{OUTDIR}'
    pseudo_dir   = '{PSEUDO_DIR}'
    tprnfor      = .true.
/
&SYSTEM
    ibrav        = 0
    nat          = 5
    ntyp         = 2
    ecutwfc      = 50.0
    ecutrho      = 400.0
    occupations  = 'smearing'
    smearing     = 'cold'
    degauss      = 0.005
    noncolin     = .true.
    lspinorb     = .true.
/
&ELECTRONS
    conv_thr     = 1.0d-9
    mixing_beta  = 0.3
/

CELL_PARAMETERS (angstrom)
  {a:.10f}   0.0000000000   0.0000000000
  {-a/2:.10f}   {a*np.sqrt(3)/2:.10f}   0.0000000000
  0.0000000000   0.0000000000   {c:.10f}

ATOMIC_SPECIES
  Bi  208.98  Bi.rel-pbe-dn-kjpaw_psl.1.0.0.UPF
  Se  78.96   Se.rel-pbe-dn-kjpaw_psl.1.0.0.UPF

ATOMIC_POSITIONS (crystal)
  Bi  0.0000000000  0.0000000000  0.3990000000
  Bi  0.0000000000  0.0000000000  0.6010000000
  Se  0.0000000000  0.0000000000  0.0000000000
  Se  0.0000000000  0.0000000000  0.2060000000
  Se  0.0000000000  0.0000000000  0.7940000000

K_POINTS (automatic)
  8 8 8  0 0 0
"""

with open(f"{PREFIX}_scf.in", "w") as f:
    f.write(scf_input)

print("Running SCF for Bi2Se3 with SOC...")
result = subprocess.run(
    ["mpirun", "-np", "4", "pw.x", "-in", f"{PREFIX}_scf.in"],
    capture_output=True, text=True, timeout=3600
)
with open(f"{PREFIX}_scf.out", "w") as f:
    f.write(result.stdout)

if "convergence has been achieved" in result.stdout:
    print("SCF converged.")
else:
    print("WARNING: SCF may not have converged. Check output.")
    print("Try reducing mixing_beta or increasing electron_maxstep.")
```

#### Step A2: Dense 3D K-Grid NSCF Calculation

```python
#!/usr/bin/env python3
"""
Step 2: Generate a dense uniform 3D k-grid and run NSCF.
For 3D spin texture, we need a full 3D k-mesh covering the Brillouin zone
(or a region of interest). We use nosym/noinv to prevent symmetry reduction
so every k-point is explicitly computed with spin expectation values.
"""
import os
import subprocess
import numpy as np

PSEUDO_DIR = os.path.abspath("./pseudo")
OUTDIR = os.path.abspath("./tmp_spintex_3d")
PREFIX = "bi2se3_soc"

a = 4.138
c = 28.636

# Dense 3D k-grid: nk x nk x nk uniform mesh
# For Fermi surface spin texture, we need enough points to resolve the surface.
# A 16x16x16 grid gives 4096 k-points -- feasible but thorough.
nk1, nk2, nk3 = 16, 16, 16

# Generate explicit k-point list for full BZ sampling
kpoints = []
for i in range(nk1):
    for j in range(nk2):
        for k in range(nk3):
            kx = i / nk1
            ky = j / nk2
            kz = k / nk3
            kpoints.append(f"  {kx:.10f}  {ky:.10f}  {kz:.10f}  1.0")

nk_total = len(kpoints)
print(f"Generated {nk_total} k-points on {nk1}x{nk2}x{nk3} grid")

kpoints_card = f"K_POINTS (crystal)\n{nk_total}\n" + "\n".join(kpoints) + "\n"

# Save the k-grid coordinates for later use
kgrid = np.zeros((nk_total, 3))
idx = 0
for i in range(nk1):
    for j in range(nk2):
        for k in range(nk3):
            kgrid[idx] = [i / nk1, j / nk2, k / nk3]
            idx += 1
np.save("kgrid_3d.npy", kgrid)

# Write NSCF input
nscf_input = f"""&CONTROL
    calculation  = 'nscf'
    prefix       = '{PREFIX}'
    outdir       = '{OUTDIR}'
    pseudo_dir   = '{PSEUDO_DIR}'
    verbosity    = 'high'
/
&SYSTEM
    ibrav        = 0
    nat          = 5
    ntyp         = 2
    ecutwfc      = 50.0
    ecutrho      = 400.0
    occupations  = 'smearing'
    smearing     = 'cold'
    degauss      = 0.005
    nbnd         = 40
    noncolin     = .true.
    lspinorb     = .true.
    nosym        = .true.
    noinv        = .true.
/
&ELECTRONS
    conv_thr     = 1.0d-9
/

CELL_PARAMETERS (angstrom)
  {a:.10f}   0.0000000000   0.0000000000
  {-a/2:.10f}   {a*np.sqrt(3)/2:.10f}   0.0000000000
  0.0000000000   0.0000000000   {c:.10f}

ATOMIC_SPECIES
  Bi  208.98  Bi.rel-pbe-dn-kjpaw_psl.1.0.0.UPF
  Se  78.96   Se.rel-pbe-dn-kjpaw_psl.1.0.0.UPF

ATOMIC_POSITIONS (crystal)
  Bi  0.0000000000  0.0000000000  0.3990000000
  Bi  0.0000000000  0.0000000000  0.6010000000
  Se  0.0000000000  0.0000000000  0.0000000000
  Se  0.0000000000  0.0000000000  0.2060000000
  Se  0.0000000000  0.0000000000  0.7940000000

{kpoints_card}
"""

with open(f"{PREFIX}_nscf_3d.in", "w") as f:
    f.write(nscf_input)

print(f"Running NSCF on {nk_total} k-points (3D grid)...")
result = subprocess.run(
    ["mpirun", "-np", "4", "pw.x", "-in", f"{PREFIX}_nscf_3d.in"],
    capture_output=True, text=True, timeout=7200
)
with open(f"{PREFIX}_nscf_3d.out", "w") as f:
    f.write(result.stdout)

if result.returncode == 0:
    print("NSCF completed successfully.")
else:
    print("ERROR in NSCF!")
    print(result.stderr[-500:] if result.stderr else "")
```

#### Step A3: Parse Spin Expectation Values from QE Output

```python
#!/usr/bin/env python3
"""
Step 3: Parse spin expectation values <Sx>, <Sy>, <Sz> for each band at
each k-point from the pw.x NSCF output (verbosity='high').

QE with noncolin=.true. and verbosity='high' prints spin components as:
    e(  1) =  -5.000 eV   Sx=  0.000  Sy=  0.000  Sz=  0.500

This script parses all k-points on the 3D grid into arrays suitable
for isosurface extraction and visualization.
"""
import re
import numpy as np
import os

PREFIX = "bi2se3_soc"
NSCF_OUTPUT = f"{PREFIX}_nscf_3d.out"


def parse_qe_spin_texture_3d(nscf_output, nk_grid=None):
    """
    Parse spin expectation values from QE NSCF output (verbosity='high').

    Parameters
    ----------
    nscf_output : str
        Path to the NSCF output file.
    nk_grid : tuple of int, optional
        (nk1, nk2, nk3) grid dimensions for reshaping.

    Returns
    -------
    kpoints : ndarray, shape (nk, 3) -- k-point coordinates (Cartesian, 2pi/a)
    eigenvalues : ndarray, shape (nk, nbnd) -- eigenvalues in eV
    sx, sy, sz : ndarray, shape (nk, nbnd) -- spin components per band
    """
    with open(nscf_output, "r") as f:
        content = f.read()

    kpoints = []
    eigenvalues = []
    sx_all, sy_all, sz_all = [], [], []

    lines = content.split("\n")
    current_k_idx = -1
    current_eigs = []
    sx_k, sy_k, sz_k = [], [], []

    for line in lines:
        # Match k-point line
        km = re.match(r"\s*k\s*=\s*([-\d.]+)\s+([-\d.]+)\s+([-\d.]+)", line)
        if km:
            # Save previous k-point data
            if current_k_idx >= 0 and sx_k:
                sx_all.append(sx_k)
                sy_all.append(sy_k)
                sz_all.append(sz_k)
            elif current_k_idx >= 0:
                sx_all.append([])
                sy_all.append([])
                sz_all.append([])

            current_k_idx += 1
            kpoints.append([float(km.group(1)), float(km.group(2)),
                            float(km.group(3))])
            sx_k, sy_k, sz_k = [], [], []
            continue

        # Match spin expectation value line
        sm = re.search(
            r"Sx=\s*([-\d.]+)\s+Sy=\s*([-\d.]+)\s+Sz=\s*([-\d.]+)", line)
        if sm:
            sx_k.append(float(sm.group(1)))
            sy_k.append(float(sm.group(2)))
            sz_k.append(float(sm.group(3)))

    # Append last k-point
    if sx_k:
        sx_all.append(sx_k)
        sy_all.append(sy_k)
        sz_all.append(sz_k)

    kpoints = np.array(kpoints)

    # Convert to uniform arrays
    if sx_all:
        nbnd = max(len(s) for s in sx_all)
        nk = len(sx_all)
        sx_arr = np.zeros((nk, nbnd))
        sy_arr = np.zeros((nk, nbnd))
        sz_arr = np.zeros((nk, nbnd))
        for i in range(nk):
            n = len(sx_all[i])
            sx_arr[i, :n] = sx_all[i]
            sy_arr[i, :n] = sy_all[i]
            sz_arr[i, :n] = sz_all[i]
    else:
        sx_arr = np.zeros((len(kpoints), 1))
        sy_arr = np.zeros((len(kpoints), 1))
        sz_arr = np.zeros((len(kpoints), 1))
        print("WARNING: No spin expectation values found.")
        print("Ensure verbosity='high' and noncolin=.true. in NSCF input.")

    # Parse eigenvalues from band blocks
    eig_pattern = re.compile(
        r"k\s*=\s*[-\d.]+\s+[-\d.]+\s+[-\d.]+.*?\n"
        r"\s*bands \(ev\):\s*\n(.*?)\n\s*\n",
        re.DOTALL
    )
    eig_matches = eig_pattern.findall(content)
    eigenvalues = []
    for eig_text in eig_matches:
        eigs = [float(x) for x in re.findall(r'([-\d.]+)', eig_text)]
        eigenvalues.append(eigs)

    if eigenvalues:
        nbnd_eig = max(len(e) for e in eigenvalues)
        eig_arr = np.full((len(eigenvalues), nbnd_eig), np.nan)
        for i, eigs in enumerate(eigenvalues):
            eig_arr[i, :len(eigs)] = eigs
    else:
        eig_arr = np.full_like(sx_arr, np.nan)

    return kpoints, eig_arr, sx_arr, sy_arr, sz_arr


# Parse the 3D spin texture data
kpoints, eigenvalues, sx, sy, sz = parse_qe_spin_texture_3d(NSCF_OUTPUT)

print(f"Parsed {len(kpoints)} k-points, {eigenvalues.shape[1]} bands")
print(f"Eigenvalue range: [{np.nanmin(eigenvalues):.3f}, "
      f"{np.nanmax(eigenvalues):.3f}] eV")
print(f"Sx range: [{sx.min():.4f}, {sx.max():.4f}]")
print(f"Sy range: [{sy.min():.4f}, {sy.max():.4f}]")
print(f"Sz range: [{sz.min():.4f}, {sz.max():.4f}]")

# Save parsed data
np.savez("spin_texture_3d_data.npz",
         kpoints=kpoints, eigenvalues=eigenvalues,
         sx=sx, sy=sy, sz=sz)
print("Data saved to spin_texture_3d_data.npz")
```

#### Step A4: Visualize 3D Spin Texture on Constant-Energy Isosurface

```python
#!/usr/bin/env python3
"""
Step 4: Visualize the 3D spin texture on a constant-energy isosurface.

Uses scipy marching_cubes to extract the Fermi surface (or constant-energy
surface) from the 3D eigenvalue grid, then colors the surface by the spin
component and overlays spin-direction arrows.
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from scipy.ndimage import uniform_filter
from matplotlib.colors import Normalize
from matplotlib import cm
import os

# ======================================================================
# Load parsed data
# ======================================================================
data_file = "spin_texture_3d_data.npz"
kgrid_file = "kgrid_3d.npy"

use_synthetic = False
if os.path.exists(data_file) and os.path.exists(kgrid_file):
    print("Loading QE spin texture data...")
    data = np.load(data_file)
    kpoints = data["kpoints"]
    eigenvalues = data["eigenvalues"]
    sx = data["sx"]
    sy = data["sy"]
    sz = data["sz"]
    kgrid = np.load(kgrid_file)
else:
    print("QE data not found -- using synthetic data for demonstration.")
    use_synthetic = True

# Grid dimensions
nk1, nk2, nk3 = 16, 16, 16

if use_synthetic:
    # Generate synthetic Bi2Se3-like Fermi surface with spin texture
    kgrid = np.zeros((nk1 * nk2 * nk3, 3))
    idx = 0
    for i in range(nk1):
        for j in range(nk2):
            for k in range(nk3):
                kgrid[idx] = [i / nk1, j / nk2, k / nk3]
                idx += 1

    # Shift to [-0.5, 0.5] range for centering at Gamma
    kgrid_centered = kgrid - 0.5
    kgrid_centered[kgrid_centered > 0.5] -= 1.0

    # Synthetic band: ellipsoidal Fermi surface centered at Gamma
    # E(k) = A*(kx^2 + ky^2) + B*kz^2 - E0
    A, B, E0 = 2.0, 3.0, 0.3
    kx = kgrid_centered[:, 0]
    ky = kgrid_centered[:, 1]
    kz = kgrid_centered[:, 2]
    nk_total = len(kgrid)

    energy_band = A * (kx**2 + ky**2) + B * kz**2 - E0
    eigenvalues = energy_band.reshape(nk_total, 1)

    # Synthetic spin texture: Rashba-like helical winding
    # Sx ~ -ky/|k|, Sy ~ kx/|k|, Sz ~ small
    k_perp = np.sqrt(kx**2 + ky**2 + 1e-10)
    sx = (-ky / k_perp * 0.4).reshape(nk_total, 1)
    sy = (kx / k_perp * 0.4).reshape(nk_total, 1)
    sz = (0.1 * kz / (np.sqrt(kz**2 + k_perp**2) + 1e-10)).reshape(
        nk_total, 1)

    print(f"Synthetic data: {nk_total} k-points, 1 band")

# ======================================================================
# Extract constant-energy isosurface using marching cubes
# ======================================================================
# Select the band of interest (e.g., band crossing Fermi level)
# For Bi2Se3, find the band index that crosses E_F
e_fermi = 0.0  # adjust based on SCF output

band_idx = 0
if eigenvalues.shape[1] > 1:
    # Find band whose eigenvalue range spans E_Fermi
    for ib in range(eigenvalues.shape[1]):
        emin = np.nanmin(eigenvalues[:, ib])
        emax = np.nanmax(eigenvalues[:, ib])
        if emin <= e_fermi <= emax:
            band_idx = ib
            break

print(f"Selected band index: {band_idx}")
print(f"Band energy range: [{np.nanmin(eigenvalues[:, band_idx]):.3f}, "
      f"{np.nanmax(eigenvalues[:, band_idx]):.3f}] eV")

# Reshape eigenvalues and spin to 3D grids
E_3d = eigenvalues[:, band_idx].reshape(nk1, nk2, nk3)
Sx_3d = sx[:, band_idx].reshape(nk1, nk2, nk3)
Sy_3d = sy[:, band_idx].reshape(nk1, nk2, nk3)
Sz_3d = sz[:, band_idx].reshape(nk1, nk2, nk3)

# Smooth for cleaner isosurface
E_smooth = uniform_filter(E_3d, size=2)

# Extract isosurface at E = E_Fermi using marching cubes
try:
    from skimage.measure import marching_cubes
    verts, faces, normals, values = marching_cubes(
        E_smooth, level=e_fermi,
        spacing=(1.0 / nk1, 1.0 / nk2, 1.0 / nk3)
    )
except ImportError:
    from scipy.ndimage import map_coordinates
    print("skimage not available; using manual isosurface slicing.")
    verts, faces = None, None

if verts is not None and len(verts) > 0:
    print(f"Isosurface: {len(verts)} vertices, {len(faces)} triangles")

    # Interpolate spin components onto isosurface vertices
    # Convert vertex positions to grid indices
    vert_grid = verts * np.array([nk1, nk2, nk3])

    from scipy.interpolate import RegularGridInterpolator
    grid_x = np.arange(nk1)
    grid_y = np.arange(nk2)
    grid_z = np.arange(nk3)

    interp_sx = RegularGridInterpolator(
        (grid_x, grid_y, grid_z), Sx_3d, bounds_error=False,
        fill_value=0.0)
    interp_sy = RegularGridInterpolator(
        (grid_x, grid_y, grid_z), Sy_3d, bounds_error=False,
        fill_value=0.0)
    interp_sz = RegularGridInterpolator(
        (grid_x, grid_y, grid_z), Sz_3d, bounds_error=False,
        fill_value=0.0)

    sx_surf = interp_sx(vert_grid)
    sy_surf = interp_sy(vert_grid)
    sz_surf = interp_sz(vert_grid)

    # ================================================================
    # Plot 1: Isosurface colored by Sz
    # ================================================================
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection="3d")

    # Color each face by average Sz of its vertices
    face_colors = np.zeros(len(faces))
    for i, face in enumerate(faces):
        face_colors[i] = np.mean(sz_surf[face])

    norm = Normalize(vmin=-0.5, vmax=0.5)
    facecolors = cm.coolwarm(norm(face_colors))

    mesh = Poly3DCollection(verts[faces], alpha=0.7)
    mesh.set_facecolor(facecolors)
    mesh.set_edgecolor("none")
    ax.add_collection3d(mesh)

    # Set axes limits
    ax.set_xlim(verts[:, 0].min(), verts[:, 0].max())
    ax.set_ylim(verts[:, 1].min(), verts[:, 1].max())
    ax.set_zlim(verts[:, 2].min(), verts[:, 2].max())

    ax.set_xlabel(r"$k_x$ (crystal)", fontsize=12)
    ax.set_ylabel(r"$k_y$ (crystal)", fontsize=12)
    ax.set_zlabel(r"$k_z$ (crystal)", fontsize=12)
    ax.set_title(
        f"3D Fermi Surface Spin Texture (band {band_idx})\n"
        r"Color: $\langle S_z \rangle$",
        fontsize=14, fontweight="bold")

    # Add colorbar
    sm = cm.ScalarMappable(norm=norm, cmap="coolwarm")
    cb = plt.colorbar(sm, ax=ax, shrink=0.6, pad=0.1)
    cb.set_label(r"$\langle S_z \rangle$ ($\hbar/2$)", fontsize=12)

    plt.savefig("spin_texture_3d_isosurface_sz.png",
                dpi=200, bbox_inches="tight")
    print("Saved: spin_texture_3d_isosurface_sz.png")
    plt.close()

    # ================================================================
    # Plot 2: Isosurface with spin-direction arrows (quiver3D)
    # ================================================================
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection="3d")

    # Re-draw the surface with reduced alpha
    mesh2 = Poly3DCollection(verts[faces], alpha=0.3)
    mesh2.set_facecolor(facecolors)
    mesh2.set_edgecolor("none")
    ax.add_collection3d(mesh2)

    # Subsample vertices for quiver arrows (every Nth vertex)
    stride = max(1, len(verts) // 200)
    v_sub = verts[::stride]
    sx_sub = sx_surf[::stride]
    sy_sub = sy_surf[::stride]
    sz_sub = sz_surf[::stride]

    # Color arrows by |S| magnitude
    s_mag = np.sqrt(sx_sub**2 + sy_sub**2 + sz_sub**2)
    arrow_colors = cm.viridis(s_mag / (s_mag.max() + 1e-10))

    ax.quiver(v_sub[:, 0], v_sub[:, 1], v_sub[:, 2],
              sx_sub, sy_sub, sz_sub,
              length=0.02, normalize=True,
              color=arrow_colors, alpha=0.8, linewidth=1.2)

    ax.set_xlim(verts[:, 0].min(), verts[:, 0].max())
    ax.set_ylim(verts[:, 1].min(), verts[:, 1].max())
    ax.set_zlim(verts[:, 2].min(), verts[:, 2].max())

    ax.set_xlabel(r"$k_x$ (crystal)", fontsize=12)
    ax.set_ylabel(r"$k_y$ (crystal)", fontsize=12)
    ax.set_zlabel(r"$k_z$ (crystal)", fontsize=12)
    ax.set_title(
        f"3D Spin Texture with Arrows (band {band_idx})\n"
        r"Arrows: $(S_x, S_y, S_z)$, Color: $|S|$",
        fontsize=14, fontweight="bold")

    plt.savefig("spin_texture_3d_arrows.png",
                dpi=200, bbox_inches="tight")
    print("Saved: spin_texture_3d_arrows.png")
    plt.close()

    # ================================================================
    # Plot 3: Constant-kz slices showing in-plane spin texture
    # ================================================================
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    kz_slices = [0, nk3 // 4, nk3 // 2]  # kz = 0, 0.25, 0.5
    kz_labels = ["$k_z = 0$", "$k_z = 0.25$", "$k_z = 0.5$"]

    for ax, kz_idx, label in zip(axes, kz_slices, kz_labels):
        E_slice = E_3d[:, :, kz_idx]
        Sx_slice = Sx_3d[:, :, kz_idx]
        Sy_slice = Sy_3d[:, :, kz_idx]
        Sz_slice = Sz_3d[:, :, kz_idx]

        kx_1d = np.linspace(0, 1, nk1)
        ky_1d = np.linspace(0, 1, nk2)
        KX, KY = np.meshgrid(kx_1d, ky_1d, indexing="ij")

        # Color background by energy
        c = ax.pcolormesh(KX, KY, E_slice, cmap="RdYlBu_r",
                          shading="auto", alpha=0.4)
        plt.colorbar(c, ax=ax, label="E (eV)", shrink=0.8)

        # Quiver plot for in-plane spin
        norm = Normalize(vmin=-0.5, vmax=0.5)
        ax.quiver(KX, KY, Sx_slice, Sy_slice,
                  Sz_slice, cmap="coolwarm", norm=norm,
                  scale=10, width=0.005, headwidth=3)

        # Mark Gamma point
        ax.plot(0, 0, "k+", markersize=10, markeredgewidth=2)

        ax.set_xlabel(r"$k_x$ (crystal)", fontsize=11)
        ax.set_ylabel(r"$k_y$ (crystal)", fontsize=11)
        ax.set_title(f"{label}", fontsize=13, fontweight="bold")
        ax.set_aspect("equal")

    plt.suptitle("Constant-$k_z$ Slices: In-Plane Spin Texture",
                 fontsize=15, fontweight="bold")
    plt.tight_layout()
    plt.savefig("spin_texture_3d_kz_slices.png",
                dpi=200, bbox_inches="tight")
    print("Saved: spin_texture_3d_kz_slices.png")
    plt.close()

else:
    print("No isosurface found at E_Fermi. Adjusting energy level...")
    print("Try changing e_fermi or selecting a different band index.")
```

### Method B: VASP -- 3D Spin Texture from PROCAR (VASPKIT Tasks 654--656)

#### Step B1: VASP Input Files for Bulk SOC Calculation on Dense 3D K-Mesh

```python
#!/usr/bin/env python3
"""
Generate VASP input files for a 3D spin texture calculation of bulk Pt.
Pt is a heavy 5d metal with strong SOC and a complex multi-sheet Fermi surface.

Workflow:
  1. SCF on coarse k-mesh (ICHARG=2)
  2. NSCF on dense 3D k-mesh (ICHARG=11)
  3. Parse PROCAR for Sx, Sy, Sz per band per k-point
  4. Visualize spin texture on 3D Fermi surface

Alternatively, use VASPKIT menu 65, tasks 654-656 to extract 3D spin texture
directly from PROCAR.
"""
import os
import numpy as np

WORK_DIR = os.path.abspath("vasp_spintex_3d")
os.makedirs(WORK_DIR, exist_ok=True)

# -- POSCAR: Bulk Pt (FCC) -----------------------------------------------
a = 3.924  # Pt lattice constant (Angstrom)

poscar = f"""Pt bulk FCC
1.0
  {a/2:.10f}  {a/2:.10f}  0.0000000000
  0.0000000000  {a/2:.10f}  {a/2:.10f}
  {a/2:.10f}  0.0000000000  {a/2:.10f}
Pt
1
Direct
  0.0000000000  0.0000000000  0.0000000000
"""

with open(os.path.join(WORK_DIR, "POSCAR"), "w") as f:
    f.write(poscar)

# -- INCAR for SCF (step 1) ----------------------------------------------
incar_scf = """# Pt bulk SOC SCF
SYSTEM   = Pt_bulk_SOC
ENCUT    = 400
PREC     = Accurate
EDIFF    = 1E-8
ISMEAR   = 1
SIGMA    = 0.1
LREAL    = .FALSE.
# SOC settings
LSORBIT  = .TRUE.
LNONCOLLINEAR = .TRUE.
# Write CHGCAR for NSCF
LCHARG   = .TRUE.
LWAVE    = .FALSE.
# Electronic
NELM     = 200
"""

with open(os.path.join(WORK_DIR, "INCAR_SCF"), "w") as f:
    f.write(incar_scf)

# -- INCAR for NSCF (step 2) ---------------------------------------------
incar_nscf = """# Pt bulk SOC NSCF for 3D spin texture
SYSTEM   = Pt_bulk_SOC_3D_SPINTEX
ENCUT    = 400
PREC     = Accurate
EDIFF    = 1E-8
ISMEAR   = 1
SIGMA    = 0.1
LREAL    = .FALSE.
# SOC settings
LSORBIT  = .TRUE.
LNONCOLLINEAR = .TRUE.
# NSCF: read charge density
ICHARG   = 11
# Write PROCAR with spin projections
LORBIT   = 11
LWAVE    = .FALSE.
LCHARG   = .FALSE.
# Electronic
NELM     = 1
"""

with open(os.path.join(WORK_DIR, "INCAR_NSCF"), "w") as f:
    f.write(incar_nscf)

# -- KPOINTS for SCF ------------------------------------------------------
kpoints_scf = """Automatic mesh for SCF
0
Gamma
12 12 12
0  0  0
"""

with open(os.path.join(WORK_DIR, "KPOINTS_SCF"), "w") as f:
    f.write(kpoints_scf)

# -- KPOINTS for NSCF: dense 3D mesh -------------------------------------
# For 3D Fermi surface spin texture we need a fine uniform grid.
# 20x20x20 = 8000 k-points is a good balance of cost and resolution.
nk1, nk2, nk3 = 20, 20, 20

kpts = []
for i in range(nk1):
    for j in range(nk2):
        for k in range(nk3):
            kx = i / nk1
            ky = j / nk2
            kz = k / nk3
            kpts.append(f"  {kx:.10f}  {ky:.10f}  {kz:.10f}  1.0")

nk_total = len(kpts)
kpoints_nscf = f"""Explicit k-points for 3D spin texture
{nk_total}
Reciprocal
""" + "\n".join(kpts) + "\n"

with open(os.path.join(WORK_DIR, "KPOINTS_NSCF"), "w") as f:
    f.write(kpoints_nscf)

print(f"VASP inputs written to {WORK_DIR}/")
print(f"  SCF k-points: 12x12x12 Gamma-centered")
print(f"  NSCF k-points: {nk_total} ({nk1}x{nk2}x{nk3} uniform grid)")
print()
print("Run workflow:")
print("  1. cp INCAR_SCF INCAR && cp KPOINTS_SCF KPOINTS && "
      "mpirun -np N vasp_ncl")
print("  2. cp INCAR_NSCF INCAR && cp KPOINTS_NSCF KPOINTS && "
      "mpirun -np N vasp_ncl")
print("  3. Parse PROCAR for 3D spin texture (Step B2)")
print()
print("VASPKIT alternative (after NSCF):")
print("  vaspkit -task 654  # Extract Sx on 3D Fermi surface")
print("  vaspkit -task 655  # Extract Sy on 3D Fermi surface")
print("  vaspkit -task 656  # Extract Sz on 3D Fermi surface")

# Save k-grid metadata
np.savez(os.path.join(WORK_DIR, "kgrid_meta.npz"),
         nk1=nk1, nk2=nk2, nk3=nk3, nk_total=nk_total)
```

#### Step B2: Parse PROCAR for 3D Spin Components

```python
#!/usr/bin/env python3
"""
Parse VASP PROCAR file from a 3D SOC NSCF calculation to extract
Sx, Sy, Sz per band per k-point across the full 3D Brillouin zone.

PROCAR with LSORBIT=.TRUE. and LORBIT=11 contains four blocks per
k-point per band:
  Block 0: charge projection
  Block 1: Sx projection
  Block 2: Sy projection
  Block 3: Sz projection

The "tot" row at the end of each block gives the total over all ions.
"""
import re
import numpy as np
import os


def parse_procar_spin_3d(procar_path):
    """
    Parse PROCAR from a VASP SOC noncollinear calculation.

    Returns
    -------
    kpoints : ndarray, shape (nk, 3)
    energies : ndarray, shape (nk, nbnd)
    occupations : ndarray, shape (nk, nbnd)
    sx, sy, sz : ndarray, shape (nk, nbnd) -- total spin per band per k
    """
    with open(procar_path, "r") as f:
        lines = f.readlines()

    # Parse header
    header = lines[1].strip()
    m = re.search(
        r"k-points:\s*(\d+).*bands:\s*(\d+).*ions:\s*(\d+)", header)
    if not m:
        raise ValueError(f"Cannot parse PROCAR header: {header}")

    nk = int(m.group(1))
    nbnd = int(m.group(2))
    nions = int(m.group(3))
    print(f"PROCAR: {nk} k-points, {nbnd} bands, {nions} ions")

    kpoints = np.zeros((nk, 3))
    energies = np.zeros((nk, nbnd))
    occupations = np.zeros((nk, nbnd))
    sx = np.zeros((nk, nbnd))
    sy = np.zeros((nk, nbnd))
    sz = np.zeros((nk, nbnd))

    ik = -1
    ib = -1
    block_count = 0

    for line in lines:
        line_s = line.strip()

        # k-point line
        km = re.match(
            r"k-point\s+(\d+)\s*:\s*([-\d.E+]+)\s+([-\d.E+]+)"
            r"\s+([-\d.E+]+)", line_s)
        if km:
            ik = int(km.group(1)) - 1
            kpoints[ik] = [float(km.group(2)), float(km.group(3)),
                           float(km.group(4))]
            block_count = 0
            continue

        # band line
        bm = re.match(
            r"band\s+(\d+)\s*#\s*energy\s+([-\d.E+]+)"
            r"\s*#\s*occ\.\s+([-\d.E+]+)", line_s)
        if bm:
            ib = int(bm.group(1)) - 1
            energies[ik, ib] = float(bm.group(2))
            occupations[ik, ib] = float(bm.group(3))
            block_count = 0
            continue

        # "tot" summary line (appears after ion rows in each block)
        if line_s.startswith("tot") and ik >= 0 and ib >= 0:
            parts = line_s.split()
            if len(parts) >= 2:
                tot_val = float(parts[-1])
                if block_count == 1:
                    sx[ik, ib] = tot_val
                elif block_count == 2:
                    sy[ik, ib] = tot_val
                elif block_count == 3:
                    sz[ik, ib] = tot_val
                block_count += 1

    return kpoints, energies, occupations, sx, sy, sz


# -- Parse and save -------------------------------------------------------
WORK_DIR = os.path.abspath("vasp_spintex_3d")
procar_path = os.path.join(WORK_DIR, "PROCAR")

if os.path.exists(procar_path):
    kpoints, energies, occ, sx, sy, sz = parse_procar_spin_3d(procar_path)

    # Read Fermi energy from OUTCAR
    e_fermi = 0.0
    outcar = os.path.join(WORK_DIR, "OUTCAR")
    if os.path.exists(outcar):
        with open(outcar, "r") as f:
            for line in f:
                if "E-fermi" in line:
                    m = re.search(r"E-fermi\s*:\s*([-\d.]+)", line)
                    if m:
                        e_fermi = float(m.group(1))

    np.savez(os.path.join(WORK_DIR, "spin_texture_3d_vasp.npz"),
             kpoints=kpoints, energies=energies,
             sx=sx, sy=sy, sz=sz, e_fermi=e_fermi)
    print(f"3D spin texture data saved.")
    print(f"E_Fermi = {e_fermi:.4f} eV")
    print(f"Energy range: [{energies.min():.3f}, {energies.max():.3f}] eV")
    print(f"Sx range: [{sx.min():.4f}, {sx.max():.4f}]")
    print(f"Sy range: [{sy.min():.4f}, {sy.max():.4f}]")
    print(f"Sz range: [{sz.min():.4f}, {sz.max():.4f}]")
else:
    print(f"PROCAR not found at {procar_path}")
    print("Run VASP NSCF calculation first (see Step B1).")
```

#### Step B3: VASPKIT Tasks 654--656 for 3D Spin Texture

```python
#!/usr/bin/env python3
"""
Use VASPKIT tasks 654-656 to extract 3D Fermi surface spin texture.

VASPKIT menu 65 provides automated spin texture extraction:
  Task 654: Extract Sx on 3D Fermi surface
  Task 655: Extract Sy on 3D Fermi surface
  Task 656: Extract Sz on 3D Fermi surface

These tasks read PROCAR (and OUTCAR for E_Fermi) and produce output
files suitable for visualization with tools like FermiSurfer or XCrySDen.

Prerequisites: PROCAR from NSCF with LSORBIT=.TRUE. and LORBIT=11.
"""
import subprocess
import os

WORK_DIR = os.path.abspath("vasp_spintex_3d")
os.chdir(WORK_DIR)

# Check prerequisites
for f in ["PROCAR", "OUTCAR", "POSCAR"]:
    if not os.path.exists(f):
        print(f"ERROR: {f} not found in {WORK_DIR}")
        print("Run the VASP SCF and NSCF steps first.")
        exit(1)

# Run VASPKIT tasks 654, 655, 656
tasks = {
    654: "Sx (3D Fermi surface)",
    655: "Sy (3D Fermi surface)",
    656: "Sz (3D Fermi surface)",
}

for task_id, description in tasks.items():
    print(f"Running VASPKIT task {task_id}: {description}...")
    result = subprocess.run(
        ["vaspkit"],
        input=f"{task_id}\n",
        capture_output=True, text=True, timeout=300
    )
    with open(f"vaspkit_task{task_id}.log", "w") as f:
        f.write(result.stdout)
    if result.returncode == 0:
        print(f"  Task {task_id} completed.")
    else:
        print(f"  WARNING: Task {task_id} may have failed.")
        print(f"  Check vaspkit_task{task_id}.log for details.")

print()
print("VASPKIT output files:")
print("  FERMISURFACE_Sx.bxsf  -- Sx on 3D Fermi surface (BXSF format)")
print("  FERMISURFACE_Sy.bxsf  -- Sy on 3D Fermi surface")
print("  FERMISURFACE_Sz.bxsf  -- Sz on 3D Fermi surface")
print()
print("Visualize with XCrySDen:")
print("  xcrysden --bxsf FERMISURFACE_Sx.bxsf")
print()
print("Or use FermiSurfer:")
print("  fermisurfer FERMISURFACE_Sx.bxsf")
```

#### Step B4: Visualize 3D Spin Texture from VASP Data

```python
#!/usr/bin/env python3
"""
Visualize 3D Fermi surface spin texture from VASP PROCAR data.
Produces:
  1. Isosurface colored by Sz with spin-direction arrows
  2. Constant-kz slices with in-plane spin quiver plots
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from matplotlib.colors import Normalize
from matplotlib import cm
from scipy.ndimage import uniform_filter
import os

WORK_DIR = os.path.abspath("vasp_spintex_3d")
data_file = os.path.join(WORK_DIR, "spin_texture_3d_vasp.npz")
meta_file = os.path.join(WORK_DIR, "kgrid_meta.npz")

if not os.path.exists(data_file):
    print(f"Data file not found: {data_file}")
    print("Run the PROCAR parser first (Step B2).")
    exit(1)

data = np.load(data_file)
kpoints = data["kpoints"]
energies = data["energies"]
sx = data["sx"]
sy = data["sy"]
sz = data["sz"]
e_fermi = float(data["e_fermi"])

meta = np.load(meta_file)
nk1, nk2, nk3 = int(meta["nk1"]), int(meta["nk2"]), int(meta["nk3"])
nk, nbnd = energies.shape

print(f"Loaded: {nk} k-points, {nbnd} bands")
print(f"E_Fermi = {e_fermi:.4f} eV")

# Find bands crossing the Fermi level
fermi_bands = []
for ib in range(nbnd):
    emin = energies[:, ib].min()
    emax = energies[:, ib].max()
    if emin <= e_fermi <= emax:
        fermi_bands.append(ib)

if not fermi_bands:
    print("No bands cross E_Fermi. Selecting closest band.")
    dist_to_ef = np.abs(energies - e_fermi).min(axis=0)
    fermi_bands = [np.argmin(dist_to_ef)]

print(f"Fermi-crossing bands: {fermi_bands}")

for band_idx in fermi_bands:
    E_3d = energies[:, band_idx].reshape(nk1, nk2, nk3)
    Sx_3d = sx[:, band_idx].reshape(nk1, nk2, nk3)
    Sy_3d = sy[:, band_idx].reshape(nk1, nk2, nk3)
    Sz_3d = sz[:, band_idx].reshape(nk1, nk2, nk3)

    E_smooth = uniform_filter(E_3d, size=2)

    # Extract isosurface at E_Fermi
    try:
        from skimage.measure import marching_cubes
        verts, faces, normals, values = marching_cubes(
            E_smooth, level=e_fermi,
            spacing=(1.0 / nk1, 1.0 / nk2, 1.0 / nk3)
        )
    except (ImportError, ValueError):
        print(f"  Skipping band {band_idx}: isosurface extraction failed.")
        continue

    if len(verts) == 0:
        print(f"  No isosurface for band {band_idx} at E_Fermi.")
        continue

    print(f"  Band {band_idx}: {len(verts)} vertices, "
          f"{len(faces)} triangles")

    # Interpolate spin onto surface vertices
    from scipy.interpolate import RegularGridInterpolator
    grid_x = np.arange(nk1)
    grid_y = np.arange(nk2)
    grid_z = np.arange(nk3)
    vert_grid = verts * np.array([nk1, nk2, nk3])

    interp_sx = RegularGridInterpolator(
        (grid_x, grid_y, grid_z), Sx_3d,
        bounds_error=False, fill_value=0.0)
    interp_sy = RegularGridInterpolator(
        (grid_x, grid_y, grid_z), Sy_3d,
        bounds_error=False, fill_value=0.0)
    interp_sz = RegularGridInterpolator(
        (grid_x, grid_y, grid_z), Sz_3d,
        bounds_error=False, fill_value=0.0)

    sx_surf = interp_sx(vert_grid)
    sy_surf = interp_sy(vert_grid)
    sz_surf = interp_sz(vert_grid)

    # -- Plot: Isosurface with Sz color and spin arrows ---------------
    fig = plt.figure(figsize=(14, 10))
    ax = fig.add_subplot(111, projection="3d")

    # Face colors from average Sz
    face_sz = np.array([np.mean(sz_surf[f]) for f in faces])
    norm = Normalize(vmin=-0.5, vmax=0.5)
    facecolors = cm.coolwarm(norm(face_sz))

    mesh = Poly3DCollection(verts[faces], alpha=0.5)
    mesh.set_facecolor(facecolors)
    mesh.set_edgecolor("none")
    ax.add_collection3d(mesh)

    # Subsample spin arrows
    stride = max(1, len(verts) // 150)
    v_sub = verts[::stride]
    ax.quiver(v_sub[:, 0], v_sub[:, 1], v_sub[:, 2],
              sx_surf[::stride], sy_surf[::stride], sz_surf[::stride],
              length=0.015, normalize=True,
              color="black", alpha=0.6, linewidth=0.8)

    ax.set_xlim(verts[:, 0].min(), verts[:, 0].max())
    ax.set_ylim(verts[:, 1].min(), verts[:, 1].max())
    ax.set_zlim(verts[:, 2].min(), verts[:, 2].max())

    ax.set_xlabel(r"$k_x$", fontsize=12)
    ax.set_ylabel(r"$k_y$", fontsize=12)
    ax.set_zlabel(r"$k_z$", fontsize=12)
    ax.set_title(
        f"Pt 3D Fermi Surface Spin Texture (band {band_idx})\n"
        r"Surface color: $\langle S_z \rangle$, "
        r"Arrows: $(\langle S_x \rangle, \langle S_y \rangle, "
        r"\langle S_z \rangle)$",
        fontsize=13, fontweight="bold")

    sm = cm.ScalarMappable(norm=norm, cmap="coolwarm")
    cb = plt.colorbar(sm, ax=ax, shrink=0.55, pad=0.1)
    cb.set_label(r"$\langle S_z \rangle$ ($\hbar/2$)", fontsize=12)

    outname = os.path.join(
        WORK_DIR, f"spin_texture_3d_vasp_band{band_idx}.png")
    plt.savefig(outname, dpi=200, bbox_inches="tight")
    print(f"  Saved: {outname}")
    plt.close()

# -- Constant-kz slices for all Fermi-crossing bands ---------------------
fig, axes = plt.subplots(1, 3, figsize=(18, 6))
kz_indices = [0, nk3 // 4, nk3 // 2]
kz_labels = [r"$k_z = 0$", r"$k_z = \pi/2a$", r"$k_z = \pi/a$"]

band_idx = fermi_bands[0]  # use first Fermi-crossing band

for ax, kz_idx, label in zip(axes, kz_indices, kz_labels):
    Sx_slice = sx[:, band_idx].reshape(nk1, nk2, nk3)[:, :, kz_idx]
    Sy_slice = sy[:, band_idx].reshape(nk1, nk2, nk3)[:, :, kz_idx]
    Sz_slice = sz[:, band_idx].reshape(nk1, nk2, nk3)[:, :, kz_idx]
    E_slice = energies[:, band_idx].reshape(nk1, nk2, nk3)[:, :, kz_idx]

    kx_1d = np.linspace(0, 1, nk1)
    ky_1d = np.linspace(0, 1, nk2)
    KX, KY = np.meshgrid(kx_1d, ky_1d, indexing="ij")

    # Contour of E = E_Fermi (Fermi contour at this kz)
    ax.contour(KX, KY, E_slice, levels=[e_fermi],
               colors="black", linewidths=1.5)

    # Spin quiver
    norm = Normalize(vmin=-0.5, vmax=0.5)
    ax.quiver(KX, KY, Sx_slice, Sy_slice,
              Sz_slice, cmap="coolwarm", norm=norm,
              scale=12, width=0.004, headwidth=3)

    ax.set_xlabel(r"$k_x$ (crystal)", fontsize=11)
    ax.set_ylabel(r"$k_y$ (crystal)", fontsize=11)
    ax.set_title(f"{label}\n(band {band_idx})", fontsize=13,
                 fontweight="bold")
    ax.set_aspect("equal")

plt.suptitle("Pt: Constant-$k_z$ Slices with Spin Texture",
             fontsize=14, fontweight="bold")
plt.tight_layout()
outname = os.path.join(WORK_DIR, "spin_texture_3d_kz_slices_vasp.png")
plt.savefig(outname, dpi=200, bbox_inches="tight")
print(f"Saved: {outname}")
plt.close()
```

---

## Key Parameters

| Parameter | Typical Value | Notes |
|---|---|---|
| `noncolin` / `LNONCOLLINEAR` | `.true.` / `.TRUE.` | Required for spin texture (noncollinear spinors) |
| `lspinorb` / `LSORBIT` | `.true.` / `.TRUE.` | Required for spin-orbit coupling |
| `nosym`, `noinv` (QE) | `.true.` | Disable symmetry in NSCF for explicit 3D k-mesh |
| `LORBIT` (VASP) | 11 or 12 | Writes spin-projected PROCAR; 12 includes phase factors |
| `ICHARG` (VASP) | 11 | Read charge density for NSCF |
| Pseudopotentials (QE) | `*_rel_*` or `*_FR_*` | Must be fully relativistic for SOC |
| `ecutwfc` / `ENCUT` | 50--60 Ry / 400--500 eV | 10--20% higher than scalar-relativistic values |
| `nbnd` (QE) / `NBANDS` (VASP) | 2x non-SOC value | SOC doubles bands (spinor representation) |
| 3D k-mesh | 16x16x16 to 24x24x24 | Denser = smoother isosurface; 4096--13824 k-points |
| `verbosity` (QE) | `'high'` | Required to print spin expectation values in pw.x output |
| `degauss` / `SIGMA` | 0.005 Ry / 0.05--0.1 eV | Appropriate smearing for metals; smaller for semiconductors |
| VASPKIT tasks | 654, 655, 656 | Extract Sx, Sy, Sz on 3D Fermi surface from PROCAR |
| `KPAR` (VASP) | 2--8 | k-point parallelization to handle large 3D k-meshes |

## Interpreting Results

- **Spin-momentum locking in 3D**: On a 3D Fermi surface, spin-momentum locking manifests as a systematic alignment of the spin polarization vector with respect to the local momentum direction. In topological surface states embedded in bulk calculations, the spin winds helically around the Fermi contour.

- **Spin chirality on Fermi sheets**: Each Fermi surface sheet may carry a distinct spin chirality. For time-reversal invariant systems, the spin at k and -k must be opposite: S(k) = -S(-k). This produces a characteristic pattern where opposite sides of the Fermi surface have opposite spin colors.

- **Weyl semimetal signatures**: Near Weyl nodes, the spin texture exhibits a hedgehog-like pattern (radially outward or inward) in 3D k-space, directly reflecting the monopole character of the Berry curvature. The spin winding number corresponds to the topological charge of the Weyl point.

- **Rashba bulk splitting (non-centrosymmetric crystals)**: In bulk crystals lacking inversion symmetry (e.g., BiTeI, GeTe), the Fermi surface splits into spin-polarized sheets with opposite helicity. The inner and outer Fermi surfaces carry opposite spin windings.

- **Heavy fermion and 5d metal textures**: In Pt, Au, or W, the multi-sheet Fermi surface exhibits complex spin textures arising from band hybridization under strong SOC. Different sheets (electron pockets, hole pockets) may show different spin polarization patterns.

- **Spin magnitude**: Each component ranges from -1/2 to +1/2 in units of hbar. For bulk states with strong orbital mixing, the total spin polarization |S| may be significantly less than 1/2, indicating that orbital angular momentum contributes to the total angular momentum.

- **Constant-kz slices**: Examining the spin texture at different kz values reveals the three-dimensional structure of spin-momentum locking and is directly comparable to spin-resolved ARPES measurements at different photon energies.

## Common Issues

| Problem | Solution |
|---|---|
| No spin components in QE output | Set `verbosity = 'high'` in `&CONTROL`. Ensure `noncolin = .true.` and `lspinorb = .true.` are set. |
| PROCAR has no spin blocks (VASP) | Ensure `LSORBIT = .TRUE.` and `LORBIT = 11` (or 12) in INCAR. The PROCAR must contain 4 blocks per band (charge, Sx, Sy, Sz). Use `vasp_ncl` executable. |
| Isosurface extraction fails or is empty | Adjust the energy level (e_fermi). Verify the band index crosses the target energy. Increase k-mesh density for smoother interpolation. |
| Memory error with dense 3D k-mesh | Reduce grid from 24x24x24 to 16x16x16. Use k-point parallelization (`-nk` in QE, `KPAR` in VASP). Reduce `nbnd`/`NBANDS` to include only bands near E_F. |
| VASPKIT tasks 654--656 produce no output | Confirm PROCAR exists and was generated with `LSORBIT = .TRUE.` and `LORBIT = 11`. Check that OUTCAR contains E-fermi. Ensure POSCAR is in the same directory. |
| Spin texture looks random or noisy | Increase 3D k-mesh density. Ensure SCF is well converged (tight EDIFF). Check that `nosym = .true.` is set in the NSCF step. Apply light smoothing (uniform_filter) before isosurface extraction. |
| Wrong Fermi surface topology | Verify E_Fermi from SCF output. For metals, ensure adequate smearing. Compare Fermi surface shape with known literature before analyzing spin texture. |
| SCF does not converge with SOC | Reduce `mixing_beta` to 0.1--0.2 (QE) or use `AMIX=0.1, BMIX=0.01` (VASP). Start from a non-SOC converged charge density. Increase `electron_maxstep`/`NELM`. |
| Pseudopotential error with SOC (QE) | Must use fully relativistic PPs (`*.rel-*` or `*_FR_*`). Scalar-relativistic PPs with `lspinorb=.true.` will crash or give incorrect results. |
| Spin texture violates time-reversal symmetry | Numerical artifact from insufficient k-mesh or convergence. Verify S(k) = -S(-k) as a consistency check. Tighten `conv_thr`/`EDIFF`. |
| 3D visualization is hard to interpret | Use constant-kz slices (2D quiver plots) for clearer analysis. Use interactive 3D viewers (plotly, mayavi, FermiSurfer) instead of static matplotlib. Export BXSF files via VASPKIT for XCrySDen. |
