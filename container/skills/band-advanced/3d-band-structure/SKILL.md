# 3D Band Structure for 2D Materials

## When to Use

- You need the full E(kx, ky) energy surface across the entire 2D Brillouin zone for a layered or 2D material.
- You want to visualize Dirac cones, saddle points, van Hove singularities, or band touching points.
- You are studying the anisotropy of band dispersion beyond what a single k-path reveals.
- You need to identify the exact location of band extrema (VBM/CBM) in the full 2D BZ.
- You are comparing DFT results to ARPES data, which maps the full BZ.

## Method Selection

| Criterion | ASE + MACE | QE DFT | VASP DFT |
|---|---|---|---|
| Availability | Cannot compute electronic bands | Full 3D band structure | Full 3D band structure |
| Reason | MACE is a force field with no electronic states | Solves Kohn-Sham equations on dense 2D k-mesh | Solves Kohn-Sham equations on dense 2D k-mesh |
| Use case | Pre-relax slab structure only | Production 3D band calculation | Production 3D band calculation |
| 2D handling | N/A | `assume_isolated = '2D'` for Coulomb cutoff | Vacuum > 15 A along z |

**MACE cannot produce band structures.** Always use QE or VASP for electronic dispersion.

## Prerequisites

- Relaxed 2D slab structure with sufficient vacuum (> 15 A) along the non-periodic direction.
- Quantum ESPRESSO 7.5 (`pw.x`, `bands.x`) or VASP.
- Pseudopotential files in `./pseudo/`.
- Python: `numpy`, `scipy`, `matplotlib`, `pymatgen`, `ase`.
- Optional: `pip install seekpath` for automatic high-symmetry point identification.

---

## Detailed Steps

### Method A: ASE + MACE (Structure Preparation Only)

MACE cannot compute band structures but can quickly relax a 2D slab before DFT.

```python
#!/usr/bin/env python3
"""
Relax a 2D slab structure with MACE, then export for QE 3D band calculation.
Example: monolayer MoS2.
"""
import warnings
warnings.filterwarnings("ignore")

import numpy as np
from ase import Atoms
from ase.io import write
from ase.optimize import LBFGS
from ase.filters import FrechetCellFilter
from ase.constraints import FixedPlane
from mace.calculators import mace_mp

# -- Build monolayer MoS2 --
a = 3.16  # in-plane lattice constant (Angstrom)
c_vac = 20.0  # vacuum along z
cell = [[a, 0, 0],
        [-a/2, a*np.sqrt(3)/2, 0],
        [0, 0, c_vac]]

# Fractional positions: Mo at (1/3, 2/3, 0.5), S at (2/3, 1/3, 0.5 +/- dz)
dz_frac = 1.56 / c_vac  # S-Mo vertical separation ~ 1.56 A
positions_frac = [
    [1/3, 2/3, 0.5],            # Mo
    [2/3, 1/3, 0.5 + dz_frac],  # S (top)
    [2/3, 1/3, 0.5 - dz_frac],  # S (bottom)
]
atoms = Atoms(symbols='MoS2', scaled_positions=positions_frac,
              cell=cell, pbc=[True, True, False])

# -- Relax with MACE --
calc = mace_mp(model="medium", dispersion=False, device="cpu")
atoms.calc = calc

# Only relax in-plane (fix z of cell, allow atom relaxation)
filtered = FrechetCellFilter(atoms, mask=[True, True, False, False, False, False])
opt = LBFGS(filtered, logfile="relax_2d.log")
opt.run(fmax=0.01)

print(f"Relaxed in-plane lattice: a = {np.linalg.norm(atoms.cell[0]):.4f} A")
print(f"Vacuum: {atoms.cell[2][2]:.1f} A")
write("MoS2_relaxed.cif", atoms)
write("MoS2_relaxed.xsf", atoms)
print("Structure saved to MoS2_relaxed.cif")
```

### Method B: QE DFT 3D Band Structure

The workflow has four steps:
1. **SCF**: Self-consistent calculation on a standard k-grid.
2. **NSCF on dense 2D mesh**: Non-self-consistent calculation on a dense kx-ky grid at kz=0.
3. **Extract eigenvalues**: Parse QE output for all eigenvalues on the 2D mesh.
4. **Plot**: 3D surface plot of E(kx, ky).

#### Step B1: SCF Calculation

```python
#!/usr/bin/env python3
"""
Step 1: SCF calculation for monolayer MoS2.
Uses ibrav=0 with explicit cell parameters for a 2D slab.
"""
import os
import subprocess
import numpy as np

PSEUDO_DIR = os.path.abspath("./pseudo")
OUTDIR = os.path.abspath("./tmp_3dband")
os.makedirs(OUTDIR, exist_ok=True)
PREFIX = "mos2"

# Lattice parameters
a = 3.16
c_vac = 20.0

scf_input = f"""&CONTROL
    calculation  = 'scf'
    prefix       = '{PREFIX}'
    outdir       = '{OUTDIR}'
    pseudo_dir   = '{PSEUDO_DIR}'
    tprnfor      = .true.
    tstress      = .true.
/
&SYSTEM
    ibrav        = 0
    nat          = 3
    ntyp         = 2
    ecutwfc      = 60.0
    ecutrho      = 480.0
    occupations  = 'smearing'
    smearing     = 'cold'
    degauss      = 0.005
    assume_isolated = '2D'
/
&ELECTRONS
    conv_thr     = 1.0d-8
/

CELL_PARAMETERS (angstrom)
  {a:.10f}   0.0000000000   0.0000000000
  {-a/2:.10f}   {a*np.sqrt(3)/2:.10f}   0.0000000000
  0.0000000000   0.0000000000   {c_vac:.10f}

ATOMIC_SPECIES
  Mo  95.94   Mo.pbe-spn-kjpaw_psl.1.0.0.UPF
  S   32.065  S.pbe-n-kjpaw_psl.1.0.0.UPF

ATOMIC_POSITIONS (crystal)
  Mo  0.3333333333  0.6666666667  0.5000000000
  S   0.6666666667  0.3333333333  {0.5 + 1.56/c_vac:.10f}
  S   0.6666666667  0.3333333333  {0.5 - 1.56/c_vac:.10f}

K_POINTS (automatic)
  12 12 1  0 0 0
"""

with open(f"{PREFIX}_scf.in", "w") as f:
    f.write(scf_input)

print("Running SCF for 3D band structure...")
result = subprocess.run(
    ["mpirun", "-np", "4", "pw.x", "-in", f"{PREFIX}_scf.in"],
    capture_output=True, text=True, timeout=600
)
with open(f"{PREFIX}_scf.out", "w") as f:
    f.write(result.stdout)

if "convergence has been achieved" in result.stdout:
    print("SCF converged.")
else:
    print("WARNING: SCF may not have converged. Check output.")
```

#### Step B2: Generate Dense 2D K-Mesh and Run NSCF

```python
#!/usr/bin/env python3
"""
Step 2: Generate a dense 2D k-mesh covering the full BZ and run NSCF.
The k-mesh is uniform in fractional coordinates (b1, b2) with kz=0.
"""
import os
import subprocess
import numpy as np

PSEUDO_DIR = os.path.abspath("./pseudo")
OUTDIR = os.path.abspath("./tmp_3dband")
PREFIX = "mos2"

a = 3.16
c_vac = 20.0

# -- Generate dense 2D k-mesh --
nk1, nk2 = 40, 40  # density: 40x40 = 1600 k-points
kpoints = []
for i in range(nk1):
    for j in range(nk2):
        kx = i / nk1  # fractional coordinate along b1
        ky = j / nk2  # fractional coordinate along b2
        kz = 0.0
        weight = 1.0
        kpoints.append(f"  {kx:.10f}  {ky:.10f}  {kz:.10f}  {weight:.1f}")

nk_total = len(kpoints)
kpoints_card = f"K_POINTS (crystal)\n{nk_total}\n" + "\n".join(kpoints) + "\n"

# -- Write NSCF input --
nscf_input = f"""&CONTROL
    calculation  = 'nscf'
    prefix       = '{PREFIX}'
    outdir       = '{OUTDIR}'
    pseudo_dir   = '{PSEUDO_DIR}'
    verbosity    = 'high'
/
&SYSTEM
    ibrav        = 0
    nat          = 3
    ntyp         = 2
    ecutwfc      = 60.0
    ecutrho      = 480.0
    occupations  = 'smearing'
    smearing     = 'cold'
    degauss      = 0.005
    nbnd         = 20
    assume_isolated = '2D'
/
&ELECTRONS
    conv_thr     = 1.0d-8
/

CELL_PARAMETERS (angstrom)
  {a:.10f}   0.0000000000   0.0000000000
  {-a/2:.10f}   {a*np.sqrt(3)/2:.10f}   0.0000000000
  0.0000000000   0.0000000000   {c_vac:.10f}

ATOMIC_SPECIES
  Mo  95.94   Mo.pbe-spn-kjpaw_psl.1.0.0.UPF
  S   32.065  S.pbe-n-kjpaw_psl.1.0.0.UPF

ATOMIC_POSITIONS (crystal)
  Mo  0.3333333333  0.6666666667  0.5000000000
  S   0.6666666667  0.3333333333  {0.5 + 1.56/c_vac:.10f}
  S   0.6666666667  0.3333333333  {0.5 - 1.56/c_vac:.10f}

{kpoints_card}
"""

with open(f"{PREFIX}_nscf_3d.in", "w") as f:
    f.write(nscf_input)

print(f"Running NSCF on {nk1}x{nk2} = {nk_total} k-points...")
result = subprocess.run(
    ["mpirun", "-np", "4", "pw.x", "-in", f"{PREFIX}_nscf_3d.in"],
    capture_output=True, text=True, timeout=3600
)
with open(f"{PREFIX}_nscf_3d.out", "w") as f:
    f.write(result.stdout)

if result.returncode == 0:
    print("NSCF completed successfully.")
else:
    print("ERROR in NSCF calculation!")
    print(result.stderr[-500:] if result.stderr else "")
```

#### Step B3: Parse Eigenvalues and Plot 3D Band Surface

```python
#!/usr/bin/env python3
"""
Step 3: Parse eigenvalues from the QE NSCF output and plot E(kx, ky) surfaces.
Produces 3D surface plots for selected bands.
"""
import re
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm

# ── Configuration ──────────────────────────────────────────────────
PREFIX = "mos2"
nk1, nk2 = 40, 40   # must match the NSCF mesh
NBND = 20
BAND_INDICES = [12, 13]  # 0-indexed: VBM and CBM bands for MoS2

# ── Parse Fermi energy from SCF output ─────────────────────────────
def get_fermi_energy(scf_output):
    """Extract Fermi energy from QE SCF output."""
    with open(scf_output, "r") as f:
        for line in f:
            if "the Fermi energy is" in line:
                m = re.search(r"is\s+([-\d.]+)\s+ev", line, re.IGNORECASE)
                if m:
                    return float(m.group(1))
            if "highest occupied, lowest unoccupied" in line:
                m = re.search(r":\s+([-\d.]+)\s+([-\d.]+)", line)
                if m:
                    return (float(m.group(1)) + float(m.group(2))) / 2
            if "highest occupied" in line:
                m = re.search(r":\s+([-\d.]+)", line)
                if m:
                    return float(m.group(1))
    return 0.0

e_fermi = get_fermi_energy(f"{PREFIX}_scf.out")
print(f"Fermi energy: {e_fermi:.4f} eV")

# ── Parse eigenvalues from NSCF output ─────────────────────────────
def parse_nscf_eigenvalues(nscf_output, nbnd):
    """
    Parse all eigenvalues from QE NSCF verbose output.
    Returns: kpoints (N x 3), eigenvalues (N x nbnd) in eV.
    """
    kpoints = []
    eigenvalues = []

    with open(nscf_output, "r") as f:
        content = f.read()

    # Find all k-point blocks
    # Pattern: "k = 0.0000 0.0000 0.0000 (...)" followed by eigenvalue lines
    kpt_pattern = re.compile(
        r"k\s*=\s*([-\d.]+)\s+([-\d.]+)\s+([-\d.]+)\s+.*?\n"
        r"(.*?)(?=\n\s*k\s*=|\n\s*the Fermi energy|\n\s*highest occupied|\Z)",
        re.DOTALL
    )

    for match in kpt_pattern.finditer(content):
        kx = float(match.group(1))
        ky = float(match.group(2))
        kz = float(match.group(3))
        kpoints.append([kx, ky, kz])

        eig_text = match.group(4)
        eig_vals = re.findall(r"([-\d.]+)", eig_text)
        # Filter out very large/invalid numbers (band indices, etc.)
        eigs = []
        for val in eig_vals:
            v = float(val)
            if -1000 < v < 1000:
                eigs.append(v)
        if len(eigs) >= nbnd:
            eigenvalues.append(eigs[:nbnd])

    return np.array(kpoints), np.array(eigenvalues)

kpoints, eigenvalues = parse_nscf_eigenvalues(f"{PREFIX}_nscf_3d.out", NBND)
print(f"Parsed {len(kpoints)} k-points, {eigenvalues.shape[1]} bands each")

# ── Reshape into 2D grid ──────────────────────────────────────────
kx_grid = kpoints[:, 0].reshape(nk1, nk2)
ky_grid = kpoints[:, 1].reshape(nk1, nk2)

# ── Convert fractional k to Cartesian (for hexagonal BZ) ──────────
a_lat = 3.16  # lattice constant in Angstrom
# Reciprocal lattice vectors for hexagonal lattice
b1 = np.array([2*np.pi/a_lat, 2*np.pi/(a_lat*np.sqrt(3)), 0])
b2 = np.array([0, 4*np.pi/(a_lat*np.sqrt(3)), 0])

kx_cart = kx_grid * b1[0] + ky_grid * b2[0]
ky_cart = kx_grid * b1[1] + ky_grid * b2[1]

# ── Plot 3D band surface ──────────────────────────────────────────
for band_idx in BAND_INDICES:
    eig_grid = eigenvalues[:, band_idx].reshape(nk1, nk2) - e_fermi

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    surf = ax.plot_surface(
        kx_cart, ky_cart, eig_grid,
        cmap=cm.coolwarm, alpha=0.9,
        linewidth=0, antialiased=True,
        rcount=nk1, ccount=nk2
    )

    ax.set_xlabel(r"$k_x$ ($\AA^{-1}$)", fontsize=12, labelpad=10)
    ax.set_ylabel(r"$k_y$ ($\AA^{-1}$)", fontsize=12, labelpad=10)
    ax.set_zlabel(r"$E - E_F$ (eV)", fontsize=12, labelpad=10)
    ax.set_title(f"3D Band Structure -- Band {band_idx + 1}", fontsize=14)

    fig.colorbar(surf, ax=ax, shrink=0.6, label="E - E_F (eV)")
    plt.tight_layout()
    plt.savefig(f"3d_band_surface_band{band_idx+1}.png", dpi=200, bbox_inches="tight")
    plt.close()
    print(f"Saved: 3d_band_surface_band{band_idx+1}.png")

# ── 2D color map (top-down view) ──────────────────────────────────
for band_idx in BAND_INDICES:
    eig_grid = eigenvalues[:, band_idx].reshape(nk1, nk2) - e_fermi

    fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.pcolormesh(kx_cart, ky_cart, eig_grid, cmap='coolwarm', shading='gouraud')
    ax.set_xlabel(r"$k_x$ ($\AA^{-1}$)", fontsize=13)
    ax.set_ylabel(r"$k_y$ ($\AA^{-1}$)", fontsize=13)
    ax.set_title(f"E(kx, ky) -- Band {band_idx + 1}", fontsize=14)
    ax.set_aspect('equal')
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("E - E_F (eV)", fontsize=12)
    plt.tight_layout()
    plt.savefig(f"3d_band_colormap_band{band_idx+1}.png", dpi=200, bbox_inches="tight")
    plt.close()
    print(f"Saved: 3d_band_colormap_band{band_idx+1}.png")
```

#### Complete Single-Script Workflow

```python
#!/usr/bin/env python3
"""
Complete 3D band structure workflow for a 2D material.
Runs: SCF -> NSCF on dense 2D mesh -> parse -> 3D surface plot.
Example: monolayer MoS2.
"""
import os
import subprocess
import re
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm

# ── Configuration ──────────────────────────────────────────────────
PSEUDO_DIR = os.path.abspath("./pseudo")
OUTDIR = os.path.abspath("./tmp_3dband_full")
os.makedirs(OUTDIR, exist_ok=True)
PREFIX = "mos2"
NPROC = 4
ECUTWFC = 60.0
ECUTRHO = 480.0
NK1, NK2 = 40, 40  # 2D mesh density
NBND = 20
BAND_VBM = 12  # 0-indexed VBM band
BAND_CBM = 13  # 0-indexed CBM band

# -- Lattice --
a = 3.16
c_vac = 20.0
dz = 1.56

cell_card = f"""CELL_PARAMETERS (angstrom)
  {a:.10f}   0.0000000000   0.0000000000
  {-a/2:.10f}   {a*np.sqrt(3)/2:.10f}   0.0000000000
  0.0000000000   0.0000000000   {c_vac:.10f}"""

atoms_card = f"""ATOMIC_SPECIES
  Mo  95.94   Mo.pbe-spn-kjpaw_psl.1.0.0.UPF
  S   32.065  S.pbe-n-kjpaw_psl.1.0.0.UPF

ATOMIC_POSITIONS (crystal)
  Mo  0.3333333333  0.6666666667  0.5000000000
  S   0.6666666667  0.3333333333  {0.5 + dz/c_vac:.10f}
  S   0.6666666667  0.3333333333  {0.5 - dz/c_vac:.10f}"""

system_card = f"""&SYSTEM
    ibrav        = 0
    nat          = 3
    ntyp         = 2
    ecutwfc      = {ECUTWFC}
    ecutrho      = {ECUTRHO}
    assume_isolated = '2D'"""

# ── Step 1: SCF ────────────────────────────────────────────────────
scf_input = f"""&CONTROL
    calculation  = 'scf'
    prefix       = '{PREFIX}'
    outdir       = '{OUTDIR}'
    pseudo_dir   = '{PSEUDO_DIR}'
/
{system_card}
    occupations  = 'smearing'
    smearing     = 'cold'
    degauss      = 0.005
/
&ELECTRONS
    conv_thr     = 1.0d-8
/

{cell_card}

{atoms_card}

K_POINTS (automatic)
  12 12 1  0 0 0
"""
with open(f"{PREFIX}_scf.in", "w") as f:
    f.write(scf_input)

print("[1/3] Running SCF...")
r = subprocess.run(["mpirun", "-np", str(NPROC), "pw.x", "-in", f"{PREFIX}_scf.in"],
                    capture_output=True, text=True, timeout=600)
with open(f"{PREFIX}_scf.out", "w") as f:
    f.write(r.stdout)
assert "convergence has been achieved" in r.stdout, "SCF did not converge!"

# Extract Fermi energy
e_fermi = 0.0
for line in r.stdout.split("\n"):
    if "the Fermi energy is" in line:
        m = re.search(r"is\s+([-\d.]+)", line)
        if m:
            e_fermi = float(m.group(1))
    if "highest occupied, lowest unoccupied" in line:
        m = re.search(r":\s+([-\d.]+)\s+([-\d.]+)", line)
        if m:
            e_fermi = (float(m.group(1)) + float(m.group(2))) / 2
    if "highest occupied" in line and "lowest" not in line:
        m = re.search(r":\s+([-\d.]+)", line)
        if m:
            e_fermi = float(m.group(1))
print(f"      Fermi energy: {e_fermi:.4f} eV")

# ── Step 2: NSCF on dense 2D k-mesh ───────────────────────────────
kpoints_list = []
for i in range(NK1):
    for j in range(NK2):
        kx = i / NK1
        ky = j / NK2
        kpoints_list.append(f"  {kx:.10f}  {ky:.10f}  0.0  1.0")

kpoints_card = f"K_POINTS (crystal)\n{len(kpoints_list)}\n" + "\n".join(kpoints_list)

nscf_input = f"""&CONTROL
    calculation  = 'nscf'
    prefix       = '{PREFIX}'
    outdir       = '{OUTDIR}'
    pseudo_dir   = '{PSEUDO_DIR}'
    verbosity    = 'high'
/
{system_card}
    occupations  = 'smearing'
    smearing     = 'cold'
    degauss      = 0.005
    nbnd         = {NBND}
/
&ELECTRONS
    conv_thr     = 1.0d-8
/

{cell_card}

{atoms_card}

{kpoints_card}
"""
with open(f"{PREFIX}_nscf_3d.in", "w") as f:
    f.write(nscf_input)

print(f"[2/3] Running NSCF on {NK1}x{NK2} = {NK1*NK2} k-points...")
r = subprocess.run(["mpirun", "-np", str(NPROC), "pw.x", "-in", f"{PREFIX}_nscf_3d.in"],
                    capture_output=True, text=True, timeout=3600)
with open(f"{PREFIX}_nscf_3d.out", "w") as f:
    f.write(r.stdout)
assert r.returncode == 0, "NSCF calculation failed!"
print("      NSCF completed.")

# ── Step 3: Parse and plot ─────────────────────────────────────────
print("[3/3] Parsing and plotting...")

# Parse eigenvalues from verbose NSCF output
kpoints_parsed = []
eigenvalues_parsed = []

kpt_pattern = re.compile(
    r"k\s*=\s*([-\d.]+)\s+([-\d.]+)\s+([-\d.]+)\s+.*?\n"
    r"(.*?)(?=\n\s*k\s*=|\n\s*the Fermi energy|\n\s*highest occupied|\Z)",
    re.DOTALL
)

for match in kpt_pattern.finditer(r.stdout if os.path.exists(f"{PREFIX}_nscf_3d.out") else ""):
    pass  # handled below

with open(f"{PREFIX}_nscf_3d.out", "r") as f:
    content = f.read()

for match in kpt_pattern.finditer(content):
    kx = float(match.group(1))
    ky = float(match.group(2))
    kz = float(match.group(3))
    kpoints_parsed.append([kx, ky, kz])

    eig_text = match.group(4)
    eig_vals = re.findall(r"([-\d.]+)", eig_text)
    eigs = [float(v) for v in eig_vals if -1000 < float(v) < 1000]
    if len(eigs) >= NBND:
        eigenvalues_parsed.append(eigs[:NBND])

kpoints_arr = np.array(kpoints_parsed)
eigenvalues_arr = np.array(eigenvalues_parsed)
print(f"      Parsed {len(kpoints_arr)} k-points")

# Reshape to 2D grid
kx_frac = kpoints_arr[:, 0].reshape(NK1, NK2)
ky_frac = kpoints_arr[:, 1].reshape(NK1, NK2)

# Convert to Cartesian reciprocal space
b1 = np.array([2*np.pi/a, 2*np.pi/(a*np.sqrt(3))])
b2 = np.array([0, 4*np.pi/(a*np.sqrt(3))])
kx_cart = kx_frac * b1[0] + ky_frac * b2[0]
ky_cart = kx_frac * b1[1] + ky_frac * b2[1]

# Plot 3D surfaces for VBM and CBM
for band_idx, label in [(BAND_VBM, "VBM"), (BAND_CBM, "CBM")]:
    eig_grid = eigenvalues_arr[:, band_idx].reshape(NK1, NK2) - e_fermi

    # 3D surface
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    surf = ax.plot_surface(kx_cart, ky_cart, eig_grid,
                           cmap=cm.coolwarm, alpha=0.9,
                           linewidth=0, antialiased=True)
    ax.set_xlabel(r"$k_x$ ($\AA^{-1}$)", fontsize=12, labelpad=10)
    ax.set_ylabel(r"$k_y$ ($\AA^{-1}$)", fontsize=12, labelpad=10)
    ax.set_zlabel(r"$E - E_F$ (eV)", fontsize=12, labelpad=10)
    ax.set_title(f"3D Band Surface -- {label} (Band {band_idx+1})", fontsize=14)
    fig.colorbar(surf, ax=ax, shrink=0.6, label="E - E_F (eV)")
    plt.tight_layout()
    plt.savefig(f"3d_band_{label}.png", dpi=200, bbox_inches="tight")
    plt.close()
    print(f"      Saved: 3d_band_{label}.png")

    # 2D colormap
    fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.pcolormesh(kx_cart, ky_cart, eig_grid, cmap='coolwarm', shading='gouraud')
    ax.set_xlabel(r"$k_x$ ($\AA^{-1}$)", fontsize=13)
    ax.set_ylabel(r"$k_y$ ($\AA^{-1}$)", fontsize=13)
    ax.set_title(f"E(kx, ky) -- {label}", fontsize=14)
    ax.set_aspect('equal')
    fig.colorbar(im, ax=ax, label="E - E_F (eV)")
    plt.tight_layout()
    plt.savefig(f"3d_band_{label}_2d.png", dpi=200, bbox_inches="tight")
    plt.close()
    print(f"      Saved: 3d_band_{label}_2d.png")

print("\nDone. Check output images.")
```

### Method C: VASP 3D Band Structure

#### Step C1: VASP Input Files

**INCAR:**
```
SYSTEM = MoS2 monolayer - SCF
ENCUT = 500
EDIFF = 1E-6
ISMEAR = 0
SIGMA = 0.05
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
12 12 1
0.0 0.0 0.0
```

After SCF converges, run the NSCF step with a dense 2D k-mesh:

**INCAR (NSCF):**
```
SYSTEM = MoS2 monolayer - NSCF dense 2D mesh
ENCUT = 500
EDIFF = 1E-6
ISMEAR = 0
SIGMA = 0.05
IBRION = -1
NSW = 0
ICHARG = 11
NBANDS = 20
LWAVE = .FALSE.
LCHARG = .FALSE.
PREC = Accurate
```

**Generate dense KPOINTS file with Python:**

```python
#!/usr/bin/env python3
"""
Generate a dense 2D k-mesh KPOINTS file for VASP 3D band structure.
"""

nk1, nk2 = 40, 40
kpoints = []
for i in range(nk1):
    for j in range(nk2):
        kx = i / nk1
        ky = j / nk2
        kz = 0.0
        weight = 1.0
        kpoints.append(f"  {kx:.10f}  {ky:.10f}  {kz:.10f}  {weight:.1f}")

with open("KPOINTS", "w") as f:
    f.write("Dense 2D k-mesh for 3D band structure\n")
    f.write(f"{len(kpoints)}\n")
    f.write("Reciprocal lattice\n")
    for kp in kpoints:
        f.write(kp + "\n")

print(f"Generated KPOINTS with {len(kpoints)} points")
```

#### Step C2: Parse VASP EIGENVAL and Plot

```python
#!/usr/bin/env python3
"""
Parse VASP EIGENVAL for 3D band structure and plot E(kx, ky) surface.
Also compatible with VASPKIT tasks 231-233 output.
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm

def parse_eigenval(filename="EIGENVAL"):
    """
    Parse VASP EIGENVAL file.
    Returns: kpoints (N x 3), eigenvalues (N x nbnd), e_fermi (from OUTCAR).
    """
    with open(filename, "r") as f:
        lines = f.readlines()

    # Line 6: NELECT, NKPTS, NBANDS
    header = lines[5].split()
    nelect = int(header[0])
    nkpts = int(header[1])
    nbands = int(header[2])

    kpoints = []
    eigenvalues = []

    idx = 7  # start of first k-point block
    for ik in range(nkpts):
        # k-point line: kx ky kz weight
        kline = lines[idx].split()
        kpoints.append([float(kline[0]), float(kline[1]), float(kline[2])])

        eigs = []
        for ib in range(nbands):
            idx += 1
            parts = lines[idx].split()
            eigs.append(float(parts[1]))  # eigenvalue in eV

        eigenvalues.append(eigs)
        idx += 2  # skip blank line + next k-point header

    return np.array(kpoints), np.array(eigenvalues), nelect

def get_efermi_outcar(filename="OUTCAR"):
    """Extract Fermi energy from VASP OUTCAR."""
    import re
    with open(filename, "r") as f:
        for line in f:
            if "E-fermi" in line:
                m = re.search(r"E-fermi\s*:\s*([-\d.]+)", line)
                if m:
                    return float(m.group(1))
    return 0.0

# ── Parse ──────────────────────────────────────────────────────────
kpoints, eigenvalues, nelect = parse_eigenval("EIGENVAL")
e_fermi = get_efermi_outcar("OUTCAR")
print(f"NKPTS = {len(kpoints)}, NBANDS = {eigenvalues.shape[1]}, E_Fermi = {e_fermi:.4f} eV")

# ── Reshape and plot ───────────────────────────────────────────────
nk1, nk2 = 40, 40  # must match the k-mesh generation
assert len(kpoints) == nk1 * nk2, f"Mismatch: {len(kpoints)} != {nk1*nk2}"

kx_frac = kpoints[:, 0].reshape(nk1, nk2)
ky_frac = kpoints[:, 1].reshape(nk1, nk2)

# Convert to Cartesian (adjust for your lattice)
# Read POSCAR for reciprocal lattice vectors
from pymatgen.io.vasp import Poscar
poscar = Poscar.from_file("POSCAR")
recip = poscar.structure.lattice.reciprocal_lattice.matrix
kx_cart = kx_frac * recip[0, 0] + ky_frac * recip[1, 0]
ky_cart = kx_frac * recip[0, 1] + ky_frac * recip[1, 1]

# Determine VBM and CBM band indices
n_occ = int(nelect) // 2  # for non-spin-polarized
vbm_band = n_occ - 1  # 0-indexed
cbm_band = n_occ

for band_idx, label in [(vbm_band, "VBM"), (cbm_band, "CBM")]:
    eig_grid = eigenvalues[:, band_idx].reshape(nk1, nk2) - e_fermi

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    surf = ax.plot_surface(kx_cart, ky_cart, eig_grid,
                           cmap=cm.coolwarm, alpha=0.9,
                           linewidth=0, antialiased=True)
    ax.set_xlabel(r"$k_x$ ($\AA^{-1}$)", fontsize=12, labelpad=10)
    ax.set_ylabel(r"$k_y$ ($\AA^{-1}$)", fontsize=12, labelpad=10)
    ax.set_zlabel(r"$E - E_F$ (eV)", fontsize=12, labelpad=10)
    ax.set_title(f"3D Band Surface -- {label}", fontsize=14)
    fig.colorbar(surf, ax=ax, shrink=0.6, label="E - E_F (eV)")
    plt.tight_layout()
    plt.savefig(f"vasp_3d_band_{label}.png", dpi=200, bbox_inches="tight")
    plt.close()
    print(f"Saved: vasp_3d_band_{label}.png")
```

---

## Key Parameters

| Parameter | Typical Value | Notes |
|---|---|---|
| `NK1, NK2` | 30--60 | 2D k-mesh density. Higher = smoother surface but longer calculation. 40x40 is a good starting point. |
| `nbnd` | 1.5--2x occupied bands | Must include enough conduction bands for the energy range of interest. |
| `ecutwfc` | 50--80 Ry | Must match the SCF calculation. |
| `assume_isolated` | `'2D'` | QE: enables Coulomb cutoff for 2D systems. Prevents spurious interlayer interactions. |
| Vacuum | > 15 A | Minimum vacuum thickness along the non-periodic direction. 20 A is safer. |
| `verbosity` | `'high'` | Required in QE NSCF to print eigenvalues for each k-point. |
| `ICHARG` | `11` | VASP: read charge density from SCF, do not update (NSCF mode). |
| `ENCUT` | 400--600 eV | VASP: plane-wave cutoff. Must match SCF. |

## Interpreting Results

- **Dirac cone**: Appears as a conical surface with linear dispersion meeting at a point (e.g., graphene at K point).
- **Parabolic bands**: Typical for semiconductors near band edges. The effective mass is related to the curvature.
- **Saddle points**: Appear as hyperbolic surfaces and correspond to van Hove singularities in the DOS.
- **Band touching**: If VBM and CBM surfaces touch at a point, the material has a zero or near-zero gap at that k-point.
- **Anisotropy**: Elongated contours indicate direction-dependent effective mass and transport properties.
- **Mexican hat dispersion**: Ring-shaped VBM/CBM (seen in some TMDs with SOC) indicates band inversion or topological features.
- **Comparison to ARPES**: The 2D colormap plot directly corresponds to ARPES intensity maps. The energy range and momentum coverage should match the experimental setup.

## Common Issues

| Problem | Solution |
|---|---|
| Surface is jagged or has discontinuities | Increase `NK1, NK2` mesh density. 40x40 minimum for smooth surfaces. |
| Missing bands at some k-points | Ensure `verbosity = 'high'` in QE NSCF. Check NBND is consistent. |
| Wrong BZ shape in plot | Verify the reciprocal lattice vectors used for Cartesian conversion match the actual cell. |
| Eigenvalue parsing errors | The regex parser expects QE 7.x verbose output format. Check the output file manually if parsing fails. |
| Very slow NSCF | Reduce mesh to 30x30 for initial tests. Use k-point parallelization (`-npool`). |
| Folded bands from supercell | For supercell calculations, the BZ is smaller. Use band unfolding instead (see `band-unfolding/` skill). |
| SOC effects missing | For TMDs, spin-orbit coupling is critical. Add `lspinorb = .true.` and `noncolin = .true.` in QE, or `LSORBIT = .TRUE.` in VASP. This doubles the number of bands. |
