# 3D Fermi Surface Calculation and Visualization

## When to Use

- You need to visualize the Fermi surface of a bulk metal (Cu, Al, Fe, Nb, etc.).
- You want to understand the electronic topology: closed vs open sheets, nesting features, connectivity.
- You are studying de Haas--van Alphen oscillations and need extremal orbits on the Fermi surface.
- You need output in standard formats: XcrySDen (`.xsf`), FermiSurfer (`.frmsf`), or raw data for custom visualization.
- You want to compare your DFT Fermi surface to ARPES or positron annihilation data.

## Method Selection

| Criterion | QE DFT | VASP DFT | ASE + MACE |
|---|---|---|---|
| Availability | Full Fermi surface workflow | Full Fermi surface workflow | Cannot compute |
| Tools | `pw.x` + `pp.x` or Python parsing | `EIGENVAL` parsing | N/A |
| Output formats | XcrySDen (`.xsf`), cube, Python | FermiSurfer (`.frmsf`), Python | N/A |
| Reason | Solves Kohn-Sham equations | Solves Kohn-Sham equations | Force field, no electrons |

**MACE cannot compute Fermi surfaces.** Always use DFT.

## Prerequisites

- Quantum ESPRESSO 7.5 (`pw.x`, `pp.x`) or VASP.
- A metallic system (bands must cross the Fermi level).
- Pseudopotential files in `./pseudo/`.
- Python: `numpy`, `scipy` (for `marching_cubes`), `matplotlib`, `pymatgen`, `ase`.
- For FermiSurfer visualization: the FermiSurfer program (optional, for `.frmsf` files).

---

## Detailed Steps

### Method A: QE Fermi Surface

#### Step A1: SCF Calculation

```python
#!/usr/bin/env python3
"""
Step 1: SCF for metallic system.
Example: FCC Cu.
"""
import os
import subprocess

PSEUDO_DIR = os.path.abspath("./pseudo")
OUTDIR = os.path.abspath("./tmp_fermi")
os.makedirs(OUTDIR, exist_ok=True)
PREFIX = "cu"

scf_input = f"""&CONTROL
    calculation  = 'scf'
    prefix       = '{PREFIX}'
    outdir       = '{OUTDIR}'
    pseudo_dir   = '{PSEUDO_DIR}'
    tprnfor      = .true.
/
&SYSTEM
    ibrav        = 2
    celldm(1)    = 6.82
    nat          = 1
    ntyp         = 1
    ecutwfc      = 60.0
    ecutrho      = 480.0
    occupations  = 'smearing'
    smearing     = 'mv'
    degauss      = 0.02
/
&ELECTRONS
    conv_thr     = 1.0d-8
    mixing_beta  = 0.7
/

ATOMIC_SPECIES
  Cu  63.546  Cu.pbe-dn-kjpaw_psl.1.0.0.UPF

ATOMIC_POSITIONS (crystal)
  Cu  0.0  0.0  0.0

K_POINTS (automatic)
  12 12 12  0 0 0
"""

with open(f"{PREFIX}_scf.in", "w") as f:
    f.write(scf_input)

print("Running SCF...")
result = subprocess.run(
    ["mpirun", "-np", "4", "pw.x", "-in", f"{PREFIX}_scf.in"],
    capture_output=True, text=True, timeout=600
)
with open(f"{PREFIX}_scf.out", "w") as f:
    f.write(result.stdout)

if "convergence has been achieved" in result.stdout:
    print("SCF converged.")
else:
    print("WARNING: Check convergence.")
```

#### Step A2: NSCF on Dense Uniform 3D K-Grid

```python
#!/usr/bin/env python3
"""
Step 2: NSCF on a dense 3D k-grid for Fermi surface.
Must use a uniform grid covering the full BZ.
"""
import os
import subprocess

PSEUDO_DIR = os.path.abspath("./pseudo")
OUTDIR = os.path.abspath("./tmp_fermi")
PREFIX = "cu"

# Dense uniform k-grid
NK = 30  # 30x30x30 grid
kpoints = []
for i in range(NK):
    for j in range(NK):
        for k in range(NK):
            kx = i / NK
            ky = j / NK
            kz = k / NK
            kpoints.append(f"  {kx:.10f}  {ky:.10f}  {kz:.10f}  1.0")

kpoints_card = f"K_POINTS (crystal)\n{len(kpoints)}\n" + "\n".join(kpoints)

nscf_input = f"""&CONTROL
    calculation  = 'nscf'
    prefix       = '{PREFIX}'
    outdir       = '{OUTDIR}'
    pseudo_dir   = '{PSEUDO_DIR}'
    verbosity    = 'high'
/
&SYSTEM
    ibrav        = 2
    celldm(1)    = 6.82
    nat          = 1
    ntyp         = 1
    ecutwfc      = 60.0
    ecutrho      = 480.0
    occupations  = 'smearing'
    smearing     = 'mv'
    degauss      = 0.005
    nbnd         = 12
/
&ELECTRONS
    conv_thr     = 1.0d-8
/

ATOMIC_SPECIES
  Cu  63.546  Cu.pbe-dn-kjpaw_psl.1.0.0.UPF

ATOMIC_POSITIONS (crystal)
  Cu  0.0  0.0  0.0

{kpoints_card}
"""

with open(f"{PREFIX}_nscf_fs.in", "w") as f:
    f.write(nscf_input)

print(f"Running NSCF on {NK}x{NK}x{NK} = {NK**3} k-points...")
result = subprocess.run(
    ["mpirun", "-np", "4", "pw.x", "-in", f"{PREFIX}_nscf_fs.in"],
    capture_output=True, text=True, timeout=7200
)
with open(f"{PREFIX}_nscf_fs.out", "w") as f:
    f.write(result.stdout)

if result.returncode == 0:
    print("NSCF completed.")
else:
    print("ERROR in NSCF!")
```

#### Step A3: Extract Fermi Surface with pp.x (XcrySDen Format)

QE's `pp.x` can directly generate Fermi surface data using `plot_num = 10` (integrated LDOS up to E_F):

```python
#!/usr/bin/env python3
"""
Step 3a: Use pp.x to generate Fermi surface in XcrySDen format.
Alternatively, parse eigenvalues directly (Step 3b).
"""
import os
import subprocess

OUTDIR = os.path.abspath("./tmp_fermi")
PREFIX = "cu"

# pp.x for Fermi surface visualization
# plot_num = 10: integrated local DOS up to E_Fermi
# This gives a 3D scalar field that, when isosurfaced, approximates the Fermi surface.
pp_input = f"""&INPUTPP
    prefix   = '{PREFIX}'
    outdir   = '{OUTDIR}'
    filplot  = '{PREFIX}_fermi.dat'
    plot_num = 10
/
&PLOT
    nfile       = 1
    filepp(1)   = '{PREFIX}_fermi.dat'
    weight(1)   = 1.0
    iflag       = 3
    output_format = 5
    fileout     = '{PREFIX}_fermi.xsf'
/
"""

with open(f"{PREFIX}_pp_fermi.in", "w") as f:
    f.write(pp_input)

print("Running pp.x for Fermi surface...")
result = subprocess.run(
    ["pp.x", "-in", f"{PREFIX}_pp_fermi.in"],
    capture_output=True, text=True, timeout=300
)
with open(f"{PREFIX}_pp_fermi.out", "w") as f:
    f.write(result.stdout)

if result.returncode == 0:
    print(f"Fermi surface saved to {PREFIX}_fermi.xsf (XcrySDen format)")
    print("Open with XcrySDen: xcrysden --xsf cu_fermi.xsf")
else:
    print("ERROR in pp.x!")
```

#### Step A4: Parse Eigenvalues and Generate Fermi Surface with Python

```python
#!/usr/bin/env python3
"""
Step 3b: Parse eigenvalues from QE NSCF output, construct E(k) on a 3D grid,
extract the Fermi surface using marching cubes, and visualize.
Also outputs .frmsf format for FermiSurfer.
"""
import re
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

# ── Configuration ──────────────────────────────────────────────────
PREFIX = "cu"
NK = 30  # must match NSCF

# ── Parse Fermi energy ─────────────────────────────────────────────
def get_fermi_energy(output_file):
    with open(output_file, "r") as f:
        for line in f:
            if "the Fermi energy is" in line:
                m = re.search(r"is\s+([-\d.]+)\s+ev", line, re.IGNORECASE)
                if m:
                    return float(m.group(1))
    return 0.0

e_fermi = get_fermi_energy(f"{PREFIX}_scf.out")
print(f"Fermi energy: {e_fermi:.4f} eV")

# ── Parse eigenvalues from NSCF ───────────────────────────────────
def parse_nscf_eigenvalues(output_file):
    kpoints = []
    eigenvalues = []

    with open(output_file, "r") as f:
        content = f.read()

    kpt_pattern = re.compile(
        r"k\s*=\s*([-\d.]+)\s+([-\d.]+)\s+([-\d.]+)\s.*?\n"
        r"(.*?)(?=\n\s*k\s*=|\n\s*the Fermi energy|\n\s*highest occupied|\n\s*Writing|\Z)",
        re.DOTALL
    )

    for match in kpt_pattern.finditer(content):
        kx = float(match.group(1))
        ky = float(match.group(2))
        kz = float(match.group(3))
        kpoints.append([kx, ky, kz])

        eig_text = match.group(4)
        eig_vals = re.findall(r"([-\d.]+)", eig_text)
        eigs = [float(v) for v in eig_vals if -1000 < float(v) < 1000]
        eigenvalues.append(eigs)

    return np.array(kpoints), eigenvalues

kpoints, eigenvalues_list = parse_nscf_eigenvalues(f"{PREFIX}_nscf_fs.out")
nbnd = min(len(e) for e in eigenvalues_list)
eigenvalues = np.array([e[:nbnd] for e in eigenvalues_list])
print(f"Parsed {len(kpoints)} k-points, {nbnd} bands")

# ── Reshape eigenvalues to 3D grid ────────────────────────────────
eig_3d = eigenvalues.reshape(NK, NK, NK, nbnd)

# ── Find bands crossing the Fermi level ───────────────────────────
crossing_bands = []
for ib in range(nbnd):
    band_min = eig_3d[:, :, :, ib].min()
    band_max = eig_3d[:, :, :, ib].max()
    if band_min < e_fermi < band_max:
        crossing_bands.append(ib)
        print(f"  Band {ib+1}: min={band_min:.3f}, max={band_max:.3f} eV (crosses E_F)")

if not crossing_bands:
    print("No bands cross the Fermi level! Is this really a metal?")

# ── Extract Fermi surface using marching cubes ─────────────────────
from scipy.ndimage import gaussian_filter

# FCC Cu reciprocal lattice vectors (for celldm(1)=6.82 bohr)
a_bohr = 6.82
a_ang = a_bohr * 0.529177
recip = 2 * np.pi / a_ang * np.array([
    [-1, 1, 1],
    [1, -1, 1],
    [1, 1, -1]
])

# Fractional k-grid
kx_frac = np.linspace(0, 1, NK, endpoint=False)
ky_frac = np.linspace(0, 1, NK, endpoint=False)
kz_frac = np.linspace(0, 1, NK, endpoint=False)

fig = plt.figure(figsize=(10, 10))
ax = fig.add_subplot(111, projection='3d')

colors = ['royalblue', 'tomato', 'green', 'orange']

for idx, ib in enumerate(crossing_bands):
    # Extract band on 3D grid and shift by Fermi energy
    band_3d = eig_3d[:, :, :, ib] - e_fermi

    # Smooth slightly to reduce noise
    band_smooth = gaussian_filter(band_3d, sigma=0.5)

    # Marching cubes to extract isosurface at E=0
    try:
        from skimage.measure import marching_cubes
        verts, faces, normals, values = marching_cubes(
            band_smooth, level=0.0,
            spacing=(1.0/NK, 1.0/NK, 1.0/NK)
        )
    except ImportError:
        # Fall back to scipy
        from scipy.ndimage import generate_binary_structure
        print("skimage not available, using manual isosurface extraction")
        continue

    # Convert vertices from fractional to Cartesian reciprocal space
    verts_cart = verts @ recip

    # Create mesh for 3D plot
    mesh = Poly3DCollection(
        verts_cart[faces],
        alpha=0.6,
        facecolor=colors[idx % len(colors)],
        edgecolor='none'
    )
    ax.add_collection3d(mesh)
    print(f"  Band {ib+1}: {len(verts)} vertices, {len(faces)} faces")

# Set plot limits
all_verts = np.vstack([v @ recip for v in [np.array([[0,0,0],[1,1,1]])]])
margin = np.max(np.abs(recip)) * 0.6
ax.set_xlim(-margin, margin)
ax.set_ylim(-margin, margin)
ax.set_zlim(-margin, margin)

ax.set_xlabel(r"$k_x$ ($\AA^{-1}$)", fontsize=12)
ax.set_ylabel(r"$k_y$ ($\AA^{-1}$)", fontsize=12)
ax.set_zlabel(r"$k_z$ ($\AA^{-1}$)", fontsize=12)
ax.set_title("3D Fermi Surface (Cu)", fontsize=14)
plt.tight_layout()
plt.savefig("fermi_surface_3d.png", dpi=200, bbox_inches="tight")
plt.close()
print("Saved: fermi_surface_3d.png")

# ── Output in FermiSurfer .frmsf format ────────────────────────────
def write_frmsf(filename, eig_3d, e_fermi, recip_vectors, nk, crossing_bands):
    """
    Write eigenvalues in FermiSurfer (.frmsf) format.

    Format:
    Line 1: nk1 nk2 nk3
    Line 2: 1 (no spin-orbit)
    Line 3: number of bands
    Lines 4-6: reciprocal lattice vectors (rows)
    Then for each band: nk1*nk2*nk3 eigenvalues (E - E_F)
    """
    with open(filename, "w") as f:
        f.write(f"{nk} {nk} {nk}\n")
        f.write("1\n")  # no spin polarization
        f.write(f"{len(crossing_bands)}\n")
        for i in range(3):
            f.write(f"{recip_vectors[i, 0]:.10f} {recip_vectors[i, 1]:.10f} {recip_vectors[i, 2]:.10f}\n")

        for ib in crossing_bands:
            for iz in range(nk):
                for iy in range(nk):
                    for ix in range(nk):
                        f.write(f"{eig_3d[ix, iy, iz, ib] - e_fermi:.10f}\n")

    print(f"Saved FermiSurfer file: {filename}")

write_frmsf(f"{PREFIX}_fermi.frmsf", eig_3d, e_fermi, recip, NK, crossing_bands)

# ── Output in XcrySDen .xsf format ────────────────────────────────
def write_xsf_fermi(filename, eig_3d, e_fermi, cell_vectors, nk, band_idx):
    """
    Write a 3D data grid in XcrySDen XSF format for Fermi surface visualization.
    """
    # XSF needs the data on a grid spanning the unit cell
    nx, ny, nz = nk, nk, nk

    with open(filename, "w") as f:
        f.write("BEGIN_BLOCK_DATAGRID_3D\n")
        f.write("  fermi_surface\n")
        f.write("  BEGIN_DATAGRID_3D_fermi\n")
        f.write(f"    {nx+1} {ny+1} {nz+1}\n")  # XSF uses n+1 points (periodic)
        f.write(f"    0.0 0.0 0.0\n")  # origin

        # Spanning vectors (reciprocal lattice vectors in bohr^-1 or Angstrom^-1)
        for i in range(3):
            f.write(f"    {cell_vectors[i, 0]:.10f} {cell_vectors[i, 1]:.10f} {cell_vectors[i, 2]:.10f}\n")

        # Data values: E(k) - E_F
        band_data = eig_3d[:, :, :, band_idx] - e_fermi
        # Extend periodically: add first point at end
        data_ext = np.zeros((nx+1, ny+1, nz+1))
        data_ext[:nx, :ny, :nz] = band_data
        data_ext[nx, :ny, :nz] = band_data[0, :, :]
        data_ext[:nx, ny, :nz] = band_data[:, 0, :]
        data_ext[:nx, :ny, nz] = band_data[:, :, 0]
        data_ext[nx, ny, :nz] = band_data[0, 0, :]
        data_ext[nx, :ny, nz] = band_data[0, :, 0]
        data_ext[:nx, ny, nz] = band_data[:, 0, 0]
        data_ext[nx, ny, nz] = band_data[0, 0, 0]

        count = 0
        for iz in range(nz+1):
            for iy in range(ny+1):
                for ix in range(nx+1):
                    f.write(f"  {data_ext[ix, iy, iz]:.8f}")
                    count += 1
                    if count % 6 == 0:
                        f.write("\n")
        if count % 6 != 0:
            f.write("\n")

        f.write("  END_DATAGRID_3D_fermi\n")
        f.write("END_BLOCK_DATAGRID_3D\n")

    print(f"Saved XcrySDen file: {filename}")

for ib in crossing_bands:
    write_xsf_fermi(f"{PREFIX}_fermi_band{ib+1}.xsf", eig_3d, e_fermi, recip, NK, ib)
```

#### Complete Single-Script Workflow

```python
#!/usr/bin/env python3
"""
Complete Fermi surface workflow: SCF -> NSCF -> extract -> plot.
Example: FCC Cu.
"""
import os
import subprocess
import re
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from scipy.ndimage import gaussian_filter

# ── Configuration ──────────────────────────────────────────────────
PSEUDO_DIR = os.path.abspath("./pseudo")
OUTDIR = os.path.abspath("./tmp_fermi_full")
os.makedirs(OUTDIR, exist_ok=True)
PREFIX = "cu"
NPROC = 4
NK = 30  # 3D k-mesh density per direction
NBND = 12

# -- FCC Cu: celldm(1) = 6.82 bohr --
CELLDM1 = 6.82
a_ang = CELLDM1 * 0.529177

# ── Step 1: SCF ────────────────────────────────────────────────────
scf_input = f"""&CONTROL
    calculation = 'scf', prefix = '{PREFIX}',
    outdir = '{OUTDIR}', pseudo_dir = '{PSEUDO_DIR}'
/
&SYSTEM
    ibrav = 2, celldm(1) = {CELLDM1}, nat = 1, ntyp = 1,
    ecutwfc = 60.0, ecutrho = 480.0,
    occupations = 'smearing', smearing = 'mv', degauss = 0.02
/
&ELECTRONS
    conv_thr = 1.0d-8
/
ATOMIC_SPECIES
  Cu  63.546  Cu.pbe-dn-kjpaw_psl.1.0.0.UPF
ATOMIC_POSITIONS (crystal)
  Cu  0.0  0.0  0.0
K_POINTS (automatic)
  12 12 12  0 0 0
"""
with open(f"{PREFIX}_scf.in", "w") as f:
    f.write(scf_input)

print("[1/3] Running SCF...")
r = subprocess.run(["mpirun", "-np", str(NPROC), "pw.x", "-in", f"{PREFIX}_scf.in"],
                    capture_output=True, text=True, timeout=600)
with open(f"{PREFIX}_scf.out", "w") as f:
    f.write(r.stdout)
assert "convergence has been achieved" in r.stdout, "SCF did not converge!"

e_fermi = 0.0
for line in r.stdout.split("\n"):
    if "the Fermi energy is" in line:
        m = re.search(r"is\s+([-\d.]+)", line)
        if m:
            e_fermi = float(m.group(1))
print(f"      Fermi energy: {e_fermi:.4f} eV")

# ── Step 2: NSCF on dense 3D k-grid ───────────────────────────────
kpts = []
for i in range(NK):
    for j in range(NK):
        for k in range(NK):
            kpts.append(f"  {i/NK:.10f}  {j/NK:.10f}  {k/NK:.10f}  1.0")

kpts_card = f"K_POINTS (crystal)\n{len(kpts)}\n" + "\n".join(kpts)

nscf_input = f"""&CONTROL
    calculation = 'nscf', prefix = '{PREFIX}',
    outdir = '{OUTDIR}', pseudo_dir = '{PSEUDO_DIR}', verbosity = 'high'
/
&SYSTEM
    ibrav = 2, celldm(1) = {CELLDM1}, nat = 1, ntyp = 1,
    ecutwfc = 60.0, ecutrho = 480.0,
    occupations = 'smearing', smearing = 'mv', degauss = 0.005,
    nbnd = {NBND}
/
&ELECTRONS
    conv_thr = 1.0d-8
/
ATOMIC_SPECIES
  Cu  63.546  Cu.pbe-dn-kjpaw_psl.1.0.0.UPF
ATOMIC_POSITIONS (crystal)
  Cu  0.0  0.0  0.0
{kpts_card}
"""
with open(f"{PREFIX}_nscf_fs.in", "w") as f:
    f.write(nscf_input)

print(f"[2/3] Running NSCF on {NK}^3 = {NK**3} k-points...")
r = subprocess.run(["mpirun", "-np", str(NPROC), "pw.x", "-in", f"{PREFIX}_nscf_fs.in"],
                    capture_output=True, text=True, timeout=7200)
with open(f"{PREFIX}_nscf_fs.out", "w") as f:
    f.write(r.stdout)
assert r.returncode == 0, "NSCF failed!"
print("      NSCF completed.")

# ── Step 3: Parse, extract, plot ───────────────────────────────────
print("[3/3] Extracting Fermi surface...")

kpt_pattern = re.compile(
    r"k\s*=\s*([-\d.]+)\s+([-\d.]+)\s+([-\d.]+)\s.*?\n"
    r"(.*?)(?=\n\s*k\s*=|\n\s*the Fermi energy|\n\s*highest occupied|\n\s*Writing|\Z)",
    re.DOTALL
)

with open(f"{PREFIX}_nscf_fs.out", "r") as f:
    content = f.read()

kpoints = []
eigenvalues_list = []
for match in kpt_pattern.finditer(content):
    kpoints.append([float(match.group(1)), float(match.group(2)), float(match.group(3))])
    eig_text = match.group(4)
    eigs = [float(v) for v in re.findall(r"([-\d.]+)", eig_text) if -1000 < float(v) < 1000]
    eigenvalues_list.append(eigs)

nbnd_actual = min(len(e) for e in eigenvalues_list)
eigenvalues = np.array([e[:nbnd_actual] for e in eigenvalues_list])
eig_3d = eigenvalues.reshape(NK, NK, NK, nbnd_actual)
print(f"      Parsed {len(kpoints)} k-points, {nbnd_actual} bands")

# Find crossing bands
crossing = []
for ib in range(nbnd_actual):
    bmin = eig_3d[:, :, :, ib].min()
    bmax = eig_3d[:, :, :, ib].max()
    if bmin < e_fermi < bmax:
        crossing.append(ib)
        print(f"      Band {ib+1} crosses E_F: [{bmin:.3f}, {bmax:.3f}] eV")

# Reciprocal lattice vectors for FCC
recip = 2 * np.pi / a_ang * np.array([[-1, 1, 1], [1, -1, 1], [1, 1, -1]])

# Marching cubes
try:
    from skimage.measure import marching_cubes
    HAS_SKIMAGE = True
except ImportError:
    HAS_SKIMAGE = False
    print("      skimage not available. Install with: pip install scikit-image")

if HAS_SKIMAGE and crossing:
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')
    colors_list = ['royalblue', 'tomato', 'forestgreen', 'gold']

    for idx, ib in enumerate(crossing):
        band_shifted = eig_3d[:, :, :, ib] - e_fermi
        band_smooth = gaussian_filter(band_shifted, sigma=0.5)
        verts, faces, _, _ = marching_cubes(band_smooth, level=0.0,
                                             spacing=(1.0/NK, 1.0/NK, 1.0/NK))
        verts_cart = verts @ recip
        mesh = Poly3DCollection(verts_cart[faces], alpha=0.5,
                                facecolor=colors_list[idx % len(colors_list)],
                                edgecolor='none')
        ax.add_collection3d(mesh)

    lim = np.max(np.abs(recip)) * 0.6
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_zlim(-lim, lim)
    ax.set_xlabel(r"$k_x$ ($\AA^{-1}$)")
    ax.set_ylabel(r"$k_y$ ($\AA^{-1}$)")
    ax.set_zlabel(r"$k_z$ ($\AA^{-1}$)")
    ax.set_title("Fermi Surface (Cu)", fontsize=14)
    plt.tight_layout()
    plt.savefig("fermi_surface_3d.png", dpi=200, bbox_inches="tight")
    plt.close()
    print("      Saved: fermi_surface_3d.png")

# Write .frmsf
with open(f"{PREFIX}_fermi.frmsf", "w") as f:
    f.write(f"{NK} {NK} {NK}\n1\n{len(crossing)}\n")
    for i in range(3):
        f.write(f"{recip[i, 0]:.10f} {recip[i, 1]:.10f} {recip[i, 2]:.10f}\n")
    for ib in crossing:
        for iz in range(NK):
            for iy in range(NK):
                for ix in range(NK):
                    f.write(f"{eig_3d[ix, iy, iz, ib] - e_fermi:.10f}\n")
print(f"      Saved: {PREFIX}_fermi.frmsf")
print("\nDone.")
```

### Method B: VASP Fermi Surface

#### Step B1: VASP Input Files

**INCAR (SCF):**
```
SYSTEM = Cu FCC - SCF
ENCUT = 500
EDIFF = 1E-6
ISMEAR = 1
SIGMA = 0.1
IBRION = -1
NSW = 0
LWAVE = .TRUE.
LCHARG = .TRUE.
PREC = Accurate
```

**KPOINTS (SCF):**
```
Automatic mesh
0
Gamma
12 12 12
0.0 0.0 0.0
```

**INCAR (NSCF for Fermi surface):**
```
SYSTEM = Cu FCC - NSCF dense
ENCUT = 500
EDIFF = 1E-6
ISMEAR = 1
SIGMA = 0.05
IBRION = -1
NSW = 0
ICHARG = 11
NBANDS = 12
LWAVE = .FALSE.
PREC = Accurate
```

**Generate dense KPOINTS:**
```python
#!/usr/bin/env python3
"""Generate dense 3D KPOINTS for VASP Fermi surface."""
NK = 30
with open("KPOINTS", "w") as f:
    f.write(f"Dense {NK}x{NK}x{NK} k-mesh for Fermi surface\n")
    f.write(f"{NK**3}\n")
    f.write("Reciprocal lattice\n")
    for i in range(NK):
        for j in range(NK):
            for k in range(NK):
                f.write(f"  {i/NK:.10f}  {j/NK:.10f}  {k/NK:.10f}  1.0\n")
print(f"Written KPOINTS with {NK**3} points")
```

#### Step B2: Parse VASP EIGENVAL and Generate Fermi Surface

```python
#!/usr/bin/env python3
"""
Parse VASP EIGENVAL for Fermi surface and generate .frmsf output.
Compatible with VASPKIT tasks 261-263.
"""
import re
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from scipy.ndimage import gaussian_filter

NK = 30

# ── Parse EIGENVAL ─────────────────────────────────────────────────
def parse_eigenval(filename="EIGENVAL"):
    with open(filename, "r") as f:
        lines = f.readlines()
    header = lines[5].split()
    nelect = int(header[0])
    nkpts = int(header[1])
    nbands = int(header[2])
    kpoints = []
    eigenvalues = []
    idx = 7
    for ik in range(nkpts):
        kline = lines[idx].split()
        kpoints.append([float(kline[0]), float(kline[1]), float(kline[2])])
        eigs = []
        for ib in range(nbands):
            idx += 1
            parts = lines[idx].split()
            eigs.append(float(parts[1]))
        eigenvalues.append(eigs)
        idx += 2
    return np.array(kpoints), np.array(eigenvalues), nelect

def get_efermi_outcar(filename="OUTCAR"):
    with open(filename, "r") as f:
        for line in f:
            if "E-fermi" in line:
                m = re.search(r"E-fermi\s*:\s*([-\d.]+)", line)
                if m:
                    return float(m.group(1))
    return 0.0

kpoints, eigenvalues, nelect = parse_eigenval()
e_fermi = get_efermi_outcar()
print(f"NKPTS = {len(kpoints)}, NBANDS = {eigenvalues.shape[1]}, E_Fermi = {e_fermi:.4f} eV")

# ── Reshape to 3D grid ────────────────────────────────────────────
nbands = eigenvalues.shape[1]
eig_3d = eigenvalues.reshape(NK, NK, NK, nbands)

# ── Find crossing bands ───────────────────────────────────────────
crossing = []
for ib in range(nbands):
    bmin = eig_3d[:, :, :, ib].min()
    bmax = eig_3d[:, :, :, ib].max()
    if bmin < e_fermi < bmax:
        crossing.append(ib)
        print(f"  Band {ib+1} crosses E_F")

# ── Get reciprocal lattice ────────────────────────────────────────
from pymatgen.io.vasp import Poscar
poscar = Poscar.from_file("POSCAR")
recip = poscar.structure.lattice.reciprocal_lattice.matrix

# ── Write .frmsf ──────────────────────────────────────────────────
with open("fermi_surface.frmsf", "w") as f:
    f.write(f"{NK} {NK} {NK}\n1\n{len(crossing)}\n")
    for i in range(3):
        f.write(f"{recip[i, 0]:.10f} {recip[i, 1]:.10f} {recip[i, 2]:.10f}\n")
    for ib in crossing:
        for iz in range(NK):
            for iy in range(NK):
                for ix in range(NK):
                    f.write(f"{eig_3d[ix, iy, iz, ib] - e_fermi:.10f}\n")
print("Saved: fermi_surface.frmsf")

# ── 3D plot with marching cubes ────────────────────────────────────
try:
    from skimage.measure import marching_cubes

    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')
    colors_map = ['royalblue', 'tomato', 'forestgreen']

    for idx, ib in enumerate(crossing):
        band_shifted = eig_3d[:, :, :, ib] - e_fermi
        band_smooth = gaussian_filter(band_shifted, sigma=0.5)
        verts, faces, _, _ = marching_cubes(band_smooth, level=0.0,
                                             spacing=(1.0/NK, 1.0/NK, 1.0/NK))
        verts_cart = verts @ recip
        mesh = Poly3DCollection(verts_cart[faces], alpha=0.5,
                                facecolor=colors_map[idx % len(colors_map)],
                                edgecolor='none')
        ax.add_collection3d(mesh)

    lim = np.max(np.abs(recip)) * 0.6
    ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim); ax.set_zlim(-lim, lim)
    ax.set_xlabel(r"$k_x$ ($\AA^{-1}$)")
    ax.set_ylabel(r"$k_y$ ($\AA^{-1}$)")
    ax.set_zlabel(r"$k_z$ ($\AA^{-1}$)")
    ax.set_title("Fermi Surface (VASP)", fontsize=14)
    plt.tight_layout()
    plt.savefig("vasp_fermi_surface_3d.png", dpi=200, bbox_inches="tight")
    plt.close()
    print("Saved: vasp_fermi_surface_3d.png")
except ImportError:
    print("Install scikit-image for marching cubes: pip install scikit-image")
```

---

## Key Parameters

| Parameter | Typical Value | Notes |
|---|---|---|
| `NK` (3D mesh) | 20--40 per direction | Higher = smoother Fermi surface. 30 is a good compromise. |
| `degauss` / `SIGMA` | 0.005--0.01 Ry / 0.05--0.1 eV | Small smearing for NSCF. Too large smears the Fermi surface. |
| `nbnd` / `NBANDS` | All occupied + 2--4 empty | Need the bands that cross the Fermi level. |
| `ecutwfc` / `ENCUT` | Same as SCF | Must be converged. |
| Gaussian smoothing `sigma` | 0.3--1.0 grid points | Applied before marching cubes to reduce noise. Too large over-smooths. |
| `ICHARG` | `11` (VASP) | Read SCF charge, do not update. |
| `smearing` | `'mv'` (QE) or `ISMEAR=1` (VASP) | Methfessel-Paxton or Marzari-Vanderbilt for metals. |
| `plot_num` | `10` (QE pp.x) | Integrated LDOS up to Fermi energy -- alternative to eigenvalue parsing. |

## Interpreting Results

- **Cu Fermi surface**: Nearly free-electron-like sphere with necks connecting along [111] directions at the L points of the BZ. The neck radius is related to the dHvA frequency.
- **Closed sheets**: Electron-like (spherical) or hole-like pockets. The volume enclosed equals the carrier count (Luttinger's theorem).
- **Open sheets**: Indicate quasi-1D or quasi-2D transport. Look for flat or cylindrical sections.
- **Nesting**: Parallel flat sections of the Fermi surface connected by a nesting vector **q** drive charge-density waves and phonon anomalies. Large parallel areas suggest strong nesting.
- **Topology changes**: Under pressure or doping, Fermi surface topology can change (Lifshitz transitions). Monitor the number and shape of sheets.
- **Multiple sheets**: Each band crossing E_F produces a separate Fermi surface sheet. Color-code them in the plot.

## Common Issues

| Problem | Solution |
|---|---|
| Fermi surface is jagged/incomplete | Increase k-mesh density. 30x30x30 minimum; use 40x40x40 for smooth surfaces. |
| No bands cross the Fermi level | The system may be an insulator/semiconductor, or the Fermi energy is wrong. Check the SCF output. |
| marching_cubes import error | Install scikit-image: `pip install scikit-image`. |
| FermiSurfer format errors | Ensure the grid dimensions match exactly. The .frmsf format is strict about ordering (ix fastest). |
| XcrySDen shows artifacts | The XSF file must include periodic images (+1 in each direction). Check the write_xsf function. |
| Multiple spin channels | For spin-polarized metals, run with `nspin=2` and extract separate Fermi surfaces for spin-up and spin-down. Write separate .frmsf files. |
| Fermi surface looks wrong for non-cubic cell | The reciprocal lattice conversion must match the actual cell. Use pymatgen `reciprocal_lattice.matrix` for accuracy. |
| NSCF is extremely slow | Use k-point parallelization: `mpirun -np N pw.x -npool P -in ...` where P divides N. |
