# 2D Fermi Surface (Fermi Contour)

## When to Use

- You are studying a 2D or quasi-2D (layered) metal and need the Fermi contour in the kx-ky plane.
- You want to visualize constant-energy contours at or near the Fermi level for a 2D material.
- You are comparing with ARPES Fermi surface maps of layered materials (graphene, TMDs, cuprates).
- You need to identify nesting vectors, Dirac points, or saddle points in the 2D BZ.
- You are analyzing the Fermi surface topology of a material with weak interlayer coupling (kz-independent).

## Method Selection

| Criterion | QE DFT | VASP DFT | ASE + MACE |
|---|---|---|---|
| Availability | Full 2D Fermi contour | Full 2D Fermi contour | Cannot compute |
| Best for | 2D materials with `assume_isolated='2D'` | Slab models with vacuum | N/A |
| Key tool | `pw.x` NSCF + Python | EIGENVAL + Python | N/A |

**MACE cannot compute Fermi contours.** It has no electronic states. Use DFT.

## Prerequisites

- Quantum ESPRESSO 7.5 (`pw.x`) or VASP.
- A metallic 2D or layered material (bands crossing E_F).
- Pseudopotential files in `./pseudo/`.
- Python: `numpy`, `scipy`, `matplotlib`, `pymatgen`, `ase`.

---

## Detailed Steps

### Method A: QE 2D Fermi Contour

#### Step A1: SCF Calculation for 2D Metal

```python
#!/usr/bin/env python3
"""
Step 1: SCF for a 2D metal.
Example: monolayer graphene (metallic at the Dirac point).
"""
import os
import subprocess
import numpy as np

PSEUDO_DIR = os.path.abspath("./pseudo")
OUTDIR = os.path.abspath("./tmp_fermi2d")
os.makedirs(OUTDIR, exist_ok=True)
PREFIX = "graphene"

a = 2.46  # graphene lattice constant (Angstrom)
c_vac = 20.0  # vacuum

scf_input = f"""&CONTROL
    calculation  = 'scf'
    prefix       = '{PREFIX}'
    outdir       = '{OUTDIR}'
    pseudo_dir   = '{PSEUDO_DIR}'
/
&SYSTEM
    ibrav        = 0
    nat          = 2
    ntyp         = 1
    ecutwfc      = 60.0
    ecutrho      = 480.0
    occupations  = 'smearing'
    smearing     = 'mv'
    degauss      = 0.01
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
  C  12.011  C.pbe-n-kjpaw_psl.1.0.0.UPF

ATOMIC_POSITIONS (crystal)
  C  0.3333333333  0.6666666667  0.5000000000
  C  0.6666666667  0.3333333333  0.5000000000

K_POINTS (automatic)
  18 18 1  0 0 0
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

#### Step A2: NSCF on Dense 2D K-Mesh

```python
#!/usr/bin/env python3
"""
Step 2: NSCF on a dense kx-ky mesh at kz=0.
For 2D materials, we only need the in-plane k-points.
"""
import os
import subprocess
import numpy as np

PSEUDO_DIR = os.path.abspath("./pseudo")
OUTDIR = os.path.abspath("./tmp_fermi2d")
PREFIX = "graphene"

a = 2.46
c_vac = 20.0

# Dense 2D k-mesh
NK = 60  # 60x60 in-plane
kpoints = []
for i in range(NK):
    for j in range(NK):
        kx = i / NK
        ky = j / NK
        kpoints.append(f"  {kx:.10f}  {ky:.10f}  0.0  1.0")

kpoints_card = f"K_POINTS (crystal)\n{len(kpoints)}\n" + "\n".join(kpoints)

nscf_input = f"""&CONTROL
    calculation  = 'nscf'
    prefix       = '{PREFIX}'
    outdir       = '{OUTDIR}'
    pseudo_dir   = '{PSEUDO_DIR}'
    verbosity    = 'high'
/
&SYSTEM
    ibrav        = 0
    nat          = 2
    ntyp         = 1
    ecutwfc      = 60.0
    ecutrho      = 480.0
    occupations  = 'smearing'
    smearing     = 'mv'
    degauss      = 0.005
    nbnd         = 10
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
  C  12.011  C.pbe-n-kjpaw_psl.1.0.0.UPF

ATOMIC_POSITIONS (crystal)
  C  0.3333333333  0.6666666667  0.5000000000
  C  0.6666666667  0.3333333333  0.5000000000

{kpoints_card}
"""

with open(f"{PREFIX}_nscf_2d.in", "w") as f:
    f.write(nscf_input)

print(f"Running NSCF on {NK}x{NK} = {NK**2} k-points...")
result = subprocess.run(
    ["mpirun", "-np", "4", "pw.x", "-in", f"{PREFIX}_nscf_2d.in"],
    capture_output=True, text=True, timeout=3600
)
with open(f"{PREFIX}_nscf_2d.out", "w") as f:
    f.write(result.stdout)

if result.returncode == 0:
    print("NSCF completed.")
else:
    print("ERROR in NSCF!")
```

#### Step A3: Parse and Plot 2D Fermi Contour

```python
#!/usr/bin/env python3
"""
Step 3: Parse eigenvalues and plot 2D Fermi contour.
Shows constant-energy contours at E_F in the kx-ky plane.
"""
import re
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.interpolate import RegularGridInterpolator

# ── Configuration ──────────────────────────────────────────────────
PREFIX = "graphene"
NK = 60  # must match NSCF mesh
a = 2.46  # lattice constant

# ── Parse Fermi energy ─────────────────────────────────────────────
def get_fermi_energy(output_file):
    with open(output_file, "r") as f:
        for line in f:
            if "the Fermi energy is" in line:
                m = re.search(r"is\s+([-\d.]+)\s+ev", line, re.IGNORECASE)
                if m:
                    return float(m.group(1))
            if "highest occupied, lowest unoccupied" in line:
                m = re.search(r":\s+([-\d.]+)\s+([-\d.]+)", line)
                if m:
                    return (float(m.group(1)) + float(m.group(2))) / 2
    return 0.0

e_fermi = get_fermi_energy(f"{PREFIX}_scf.out")
print(f"Fermi energy: {e_fermi:.4f} eV")

# ── Parse eigenvalues ──────────────────────────────────────────────
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

kpoints, eigenvalues_list = parse_nscf_eigenvalues(f"{PREFIX}_nscf_2d.out")
nbnd = min(len(e) for e in eigenvalues_list)
eigenvalues = np.array([e[:nbnd] for e in eigenvalues_list])
print(f"Parsed {len(kpoints)} k-points, {nbnd} bands")

# ── Reshape to 2D grid ────────────────────────────────────────────
eig_2d = eigenvalues.reshape(NK, NK, nbnd)

# ── Convert fractional k to Cartesian ──────────────────────────────
# Hexagonal reciprocal lattice vectors
b1 = np.array([2*np.pi/a, 2*np.pi/(a*np.sqrt(3))])
b2 = np.array([0, 4*np.pi/(a*np.sqrt(3))])

kx_frac = np.linspace(0, 1, NK, endpoint=False)
ky_frac = np.linspace(0, 1, NK, endpoint=False)
KX_frac, KY_frac = np.meshgrid(kx_frac, ky_frac, indexing='ij')

KX_cart = KX_frac * b1[0] + KY_frac * b2[0]
KY_cart = KX_frac * b1[1] + KY_frac * b2[1]

# ── Plot Fermi contour ────────────────────────────────────────────
# Find bands near the Fermi level
crossing_bands = []
for ib in range(nbnd):
    band_min = eig_2d[:, :, ib].min()
    band_max = eig_2d[:, :, ib].max()
    if band_min < e_fermi < band_max:
        crossing_bands.append(ib)
        print(f"  Band {ib+1} crosses E_F: [{band_min:.3f}, {band_max:.3f}] eV")

fig, ax = plt.subplots(figsize=(8, 8))

colors = ['royalblue', 'tomato', 'forestgreen', 'gold']
for idx, ib in enumerate(crossing_bands):
    band_shifted = eig_2d[:, :, ib] - e_fermi
    cs = ax.contour(KX_cart, KY_cart, band_shifted, levels=[0.0],
                    colors=colors[idx % len(colors)], linewidths=2.0)
    # Label
    ax.clabel(cs, inline=False, fontsize=0)  # no numeric labels

# ── Draw hexagonal BZ boundary ────────────────────────────────────
def hexagonal_bz_vertices(b1, b2):
    """Compute vertices of the first Brillouin zone for hexagonal lattice."""
    # For hexagonal, the BZ is a regular hexagon
    # The K and K' points and M points define the boundary
    # Vertices at 2/3, 1/3 and permutations in fractional coords
    vertices_frac = [
        [2/3, 1/3], [1/3, 2/3], [-1/3, 1/3],
        [-2/3, -1/3], [-1/3, -2/3], [1/3, -1/3]
    ]
    vertices_cart = []
    for vf in vertices_frac:
        vx = vf[0] * b1[0] + vf[1] * b2[0]
        vy = vf[0] * b1[1] + vf[1] * b2[1]
        vertices_cart.append([vx, vy])
    vertices_cart.append(vertices_cart[0])  # close the hexagon
    return np.array(vertices_cart)

bz = hexagonal_bz_vertices(b1, b2)
ax.plot(bz[:, 0], bz[:, 1], 'k-', linewidth=1.5, alpha=0.5, label='BZ boundary')

# Mark high-symmetry points
K_point_frac = np.array([2/3, 1/3])
K_cart = K_point_frac[0] * b1 + K_point_frac[1] * b2
M_point_frac = np.array([1/2, 0])
M_cart = M_point_frac[0] * b1 + M_point_frac[1] * b2
G_cart = np.array([0, 0])

ax.plot(*K_cart, 'ko', markersize=6)
ax.annotate('K', K_cart, textcoords="offset points", xytext=(5, 5), fontsize=12)
ax.plot(*M_cart, 'ks', markersize=6)
ax.annotate('M', M_cart, textcoords="offset points", xytext=(5, 5), fontsize=12)
ax.plot(*G_cart, 'k^', markersize=6)
ax.annotate(r'$\Gamma$', G_cart, textcoords="offset points", xytext=(5, 5), fontsize=13)

ax.set_xlabel(r"$k_x$ ($\AA^{-1}$)", fontsize=13)
ax.set_ylabel(r"$k_y$ ($\AA^{-1}$)", fontsize=13)
ax.set_title("2D Fermi Surface (Fermi Contour)", fontsize=14)
ax.set_aspect('equal')
ax.legend(fontsize=11)
ax.grid(True, alpha=0.2)
plt.tight_layout()
plt.savefig("fermi_contour_2d.png", dpi=200, bbox_inches="tight")
plt.close()
print("Saved: fermi_contour_2d.png")

# ── Constant-energy contours at multiple energies ──────────────────
fig, ax = plt.subplots(figsize=(8, 8))

energy_offsets = [-0.5, -0.2, 0.0, 0.2, 0.5]  # eV relative to E_F
cmap = plt.cm.coolwarm
norm = plt.Normalize(vmin=min(energy_offsets), vmax=max(energy_offsets))

for ib in crossing_bands:
    for dE in energy_offsets:
        band_shifted = eig_2d[:, :, ib] - e_fermi - dE
        color = cmap(norm(dE))
        cs = ax.contour(KX_cart, KY_cart, band_shifted, levels=[0.0],
                        colors=[color], linewidths=1.5, alpha=0.8)

ax.plot(bz[:, 0], bz[:, 1], 'k-', linewidth=1.5, alpha=0.5)

# Colorbar for energy
sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
sm.set_array([])
cbar = fig.colorbar(sm, ax=ax)
cbar.set_label("E - E_F (eV)", fontsize=12)

ax.set_xlabel(r"$k_x$ ($\AA^{-1}$)", fontsize=13)
ax.set_ylabel(r"$k_y$ ($\AA^{-1}$)", fontsize=13)
ax.set_title("Constant-Energy Contours", fontsize=14)
ax.set_aspect('equal')
plt.tight_layout()
plt.savefig("constant_energy_contours.png", dpi=200, bbox_inches="tight")
plt.close()
print("Saved: constant_energy_contours.png")

# ── Filled color map of band energy ────────────────────────────────
for ib in crossing_bands:
    fig, ax = plt.subplots(figsize=(8, 8))

    band_shifted = eig_2d[:, :, ib] - e_fermi
    vmax = min(abs(band_shifted.min()), abs(band_shifted.max()), 3.0)

    im = ax.pcolormesh(KX_cart, KY_cart, band_shifted,
                       cmap='coolwarm', shading='gouraud',
                       vmin=-vmax, vmax=vmax)
    ax.contour(KX_cart, KY_cart, band_shifted, levels=[0.0],
               colors='black', linewidths=2.0)

    ax.plot(bz[:, 0], bz[:, 1], 'k-', linewidth=1.5, alpha=0.5)
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("E - E_F (eV)", fontsize=12)

    ax.set_xlabel(r"$k_x$ ($\AA^{-1}$)", fontsize=13)
    ax.set_ylabel(r"$k_y$ ($\AA^{-1}$)", fontsize=13)
    ax.set_title(f"Band {ib+1} Energy Map with Fermi Contour", fontsize=14)
    ax.set_aspect('equal')
    plt.tight_layout()
    plt.savefig(f"fermi_contour_band{ib+1}_colormap.png", dpi=200, bbox_inches="tight")
    plt.close()
    print(f"Saved: fermi_contour_band{ib+1}_colormap.png")
```

### Method B: VASP 2D Fermi Contour

#### Step B1: VASP Input Files

**INCAR:**
```
SYSTEM = Graphene - SCF
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
18 18 1
0.0 0.0 0.0
```

**INCAR (NSCF with dense 2D mesh):**
```
SYSTEM = Graphene - NSCF dense 2D
ENCUT = 500
EDIFF = 1E-6
ISMEAR = 1
SIGMA = 0.05
IBRION = -1
NSW = 0
ICHARG = 11
NBANDS = 10
LWAVE = .FALSE.
PREC = Accurate
```

#### Step B2: Generate KPOINTS and Parse EIGENVAL

```python
#!/usr/bin/env python3
"""
VASP 2D Fermi contour: generate dense 2D KPOINTS, parse EIGENVAL, plot.
Compatible with VASPKIT tasks 264-265.
"""
import re
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ── Generate dense 2D KPOINTS ─────────────────────────────────────
NK = 60
with open("KPOINTS", "w") as f:
    f.write(f"Dense {NK}x{NK} 2D k-mesh\n")
    f.write(f"{NK * NK}\n")
    f.write("Reciprocal lattice\n")
    for i in range(NK):
        for j in range(NK):
            f.write(f"  {i/NK:.10f}  {j/NK:.10f}  0.0  1.0\n")
print(f"Written KPOINTS with {NK*NK} points")

# ── After VASP run, parse EIGENVAL ─────────────────────────────────
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

try:
    kpoints, eigenvalues, nelect = parse_eigenval()
    e_fermi = get_efermi_outcar()
except FileNotFoundError:
    print("EIGENVAL/OUTCAR not found. Run VASP first.")
    raise SystemExit

nbands = eigenvalues.shape[1]
eig_2d = eigenvalues.reshape(NK, NK, nbands)
print(f"Parsed {NK}x{NK} grid, {nbands} bands, E_F = {e_fermi:.4f} eV")

# ── Convert to Cartesian k-space ──────────────────────────────────
from pymatgen.io.vasp import Poscar
poscar = Poscar.from_file("POSCAR")
recip = poscar.structure.lattice.reciprocal_lattice.matrix

kx_frac = np.linspace(0, 1, NK, endpoint=False)
ky_frac = np.linspace(0, 1, NK, endpoint=False)
KX_frac, KY_frac = np.meshgrid(kx_frac, ky_frac, indexing='ij')

KX_cart = KX_frac * recip[0, 0] + KY_frac * recip[1, 0]
KY_cart = KX_frac * recip[0, 1] + KY_frac * recip[1, 1]

# ── Find crossing bands and plot ──────────────────────────────────
crossing = []
for ib in range(nbands):
    bmin = eig_2d[:, :, ib].min()
    bmax = eig_2d[:, :, ib].max()
    if bmin < e_fermi < bmax:
        crossing.append(ib)

fig, ax = plt.subplots(figsize=(8, 8))
colors = ['royalblue', 'tomato', 'forestgreen']

for idx, ib in enumerate(crossing):
    band_shifted = eig_2d[:, :, ib] - e_fermi
    ax.contour(KX_cart, KY_cart, band_shifted, levels=[0.0],
               colors=colors[idx % len(colors)], linewidths=2.0)

ax.set_xlabel(r"$k_x$ ($\AA^{-1}$)", fontsize=13)
ax.set_ylabel(r"$k_y$ ($\AA^{-1}$)", fontsize=13)
ax.set_title("2D Fermi Contour (VASP)", fontsize=14)
ax.set_aspect('equal')
ax.grid(True, alpha=0.2)
plt.tight_layout()
plt.savefig("vasp_fermi_contour_2d.png", dpi=200, bbox_inches="tight")
plt.close()
print("Saved: vasp_fermi_contour_2d.png")

# ── Energy color map ──────────────────────────────────────────────
for ib in crossing:
    fig, ax = plt.subplots(figsize=(8, 8))
    band_shifted = eig_2d[:, :, ib] - e_fermi
    vmax = min(abs(band_shifted.min()), abs(band_shifted.max()), 3.0)

    im = ax.pcolormesh(KX_cart, KY_cart, band_shifted,
                       cmap='coolwarm', shading='gouraud',
                       vmin=-vmax, vmax=vmax)
    ax.contour(KX_cart, KY_cart, band_shifted, levels=[0.0],
               colors='black', linewidths=2.0)
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("E - E_F (eV)")
    ax.set_xlabel(r"$k_x$ ($\AA^{-1}$)")
    ax.set_ylabel(r"$k_y$ ($\AA^{-1}$)")
    ax.set_title(f"Band {ib+1} with Fermi Contour (VASP)")
    ax.set_aspect('equal')
    plt.tight_layout()
    plt.savefig(f"vasp_fermi_contour_band{ib+1}.png", dpi=200, bbox_inches="tight")
    plt.close()
    print(f"Saved: vasp_fermi_contour_band{ib+1}.png")
```

#### Complete Single-Script Workflow (QE)

```python
#!/usr/bin/env python3
"""
Complete 2D Fermi contour workflow: SCF -> NSCF -> parse -> plot.
Example: monolayer graphene.
"""
import os
import subprocess
import re
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ── Configuration ──────────────────────────────────────────────────
PSEUDO_DIR = os.path.abspath("./pseudo")
OUTDIR = os.path.abspath("./tmp_fermi2d_full")
os.makedirs(OUTDIR, exist_ok=True)
PREFIX = "graphene"
NPROC = 4
NK = 60  # 2D mesh density
NBND = 10

a = 2.46
c_vac = 20.0

cell_card = f"""CELL_PARAMETERS (angstrom)
  {a:.10f}   0.0000000000   0.0000000000
  {-a/2:.10f}   {a*np.sqrt(3)/2:.10f}   0.0000000000
  0.0000000000   0.0000000000   {c_vac:.10f}"""

atoms_card = """ATOMIC_SPECIES
  C  12.011  C.pbe-n-kjpaw_psl.1.0.0.UPF

ATOMIC_POSITIONS (crystal)
  C  0.3333333333  0.6666666667  0.5000000000
  C  0.6666666667  0.3333333333  0.5000000000"""

# ── Step 1: SCF ────────────────────────────────────────────────────
scf_input = f"""&CONTROL
    calculation = 'scf', prefix = '{PREFIX}',
    outdir = '{OUTDIR}', pseudo_dir = '{PSEUDO_DIR}'
/
&SYSTEM
    ibrav = 0, nat = 2, ntyp = 1,
    ecutwfc = 60.0, ecutrho = 480.0,
    occupations = 'smearing', smearing = 'mv', degauss = 0.01,
    assume_isolated = '2D'
/
&ELECTRONS
    conv_thr = 1.0d-8
/
{cell_card}
{atoms_card}
K_POINTS (automatic)
  18 18 1  0 0 0
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
print(f"      E_F = {e_fermi:.4f} eV")

# ── Step 2: NSCF on dense 2D mesh ─────────────────────────────────
kpts = []
for i in range(NK):
    for j in range(NK):
        kpts.append(f"  {i/NK:.10f}  {j/NK:.10f}  0.0  1.0")

nscf_input = f"""&CONTROL
    calculation = 'nscf', prefix = '{PREFIX}',
    outdir = '{OUTDIR}', pseudo_dir = '{PSEUDO_DIR}', verbosity = 'high'
/
&SYSTEM
    ibrav = 0, nat = 2, ntyp = 1,
    ecutwfc = 60.0, ecutrho = 480.0,
    occupations = 'smearing', smearing = 'mv', degauss = 0.005,
    nbnd = {NBND}, assume_isolated = '2D'
/
&ELECTRONS
    conv_thr = 1.0d-8
/
{cell_card}
{atoms_card}
K_POINTS (crystal)
{len(kpts)}
""" + "\n".join(kpts) + "\n"

with open(f"{PREFIX}_nscf_2d.in", "w") as f:
    f.write(nscf_input)

print(f"[2/3] Running NSCF on {NK}x{NK} = {NK**2} k-points...")
r = subprocess.run(["mpirun", "-np", str(NPROC), "pw.x", "-in", f"{PREFIX}_nscf_2d.in"],
                    capture_output=True, text=True, timeout=3600)
with open(f"{PREFIX}_nscf_2d.out", "w") as f:
    f.write(r.stdout)
assert r.returncode == 0, "NSCF failed!"
print("      NSCF completed.")

# ── Step 3: Parse and plot ─────────────────────────────────────────
print("[3/3] Plotting 2D Fermi contour...")

with open(f"{PREFIX}_nscf_2d.out", "r") as f:
    content = f.read()

kpt_pattern = re.compile(
    r"k\s*=\s*([-\d.]+)\s+([-\d.]+)\s+([-\d.]+)\s.*?\n"
    r"(.*?)(?=\n\s*k\s*=|\n\s*the Fermi energy|\n\s*highest occupied|\n\s*Writing|\Z)",
    re.DOTALL
)

kpoints_parsed = []
eigenvalues_parsed = []
for match in kpt_pattern.finditer(content):
    kpoints_parsed.append([float(match.group(1)), float(match.group(2)), float(match.group(3))])
    eig_text = match.group(4)
    eigs = [float(v) for v in re.findall(r"([-\d.]+)", eig_text) if -1000 < float(v) < 1000]
    eigenvalues_parsed.append(eigs)

nbnd_actual = min(len(e) for e in eigenvalues_parsed)
eigenvalues = np.array([e[:nbnd_actual] for e in eigenvalues_parsed])
eig_2d = eigenvalues.reshape(NK, NK, nbnd_actual)

# Cartesian k-coordinates
b1 = np.array([2*np.pi/a, 2*np.pi/(a*np.sqrt(3))])
b2 = np.array([0, 4*np.pi/(a*np.sqrt(3))])
kx_frac = np.linspace(0, 1, NK, endpoint=False)
ky_frac = np.linspace(0, 1, NK, endpoint=False)
KX_frac, KY_frac = np.meshgrid(kx_frac, ky_frac, indexing='ij')
KX_cart = KX_frac * b1[0] + KY_frac * b2[0]
KY_cart = KX_frac * b1[1] + KY_frac * b2[1]

# Find crossing bands
crossing = []
for ib in range(nbnd_actual):
    if eig_2d[:, :, ib].min() < e_fermi < eig_2d[:, :, ib].max():
        crossing.append(ib)

# Plot Fermi contour
fig, ax = plt.subplots(figsize=(8, 8))
colors = ['royalblue', 'tomato', 'forestgreen']
for idx, ib in enumerate(crossing):
    ax.contour(KX_cart, KY_cart, eig_2d[:, :, ib] - e_fermi, levels=[0.0],
               colors=colors[idx % len(colors)], linewidths=2.0)

ax.set_xlabel(r"$k_x$ ($\AA^{-1}$)", fontsize=13)
ax.set_ylabel(r"$k_y$ ($\AA^{-1}$)", fontsize=13)
ax.set_title("2D Fermi Contour (Graphene)", fontsize=14)
ax.set_aspect('equal')
ax.grid(True, alpha=0.2)
plt.tight_layout()
plt.savefig("fermi_contour_2d.png", dpi=200, bbox_inches="tight")
plt.close()
print("      Saved: fermi_contour_2d.png")

# Multi-energy contour plot
fig, ax = plt.subplots(figsize=(8, 8))
energy_offsets = [-0.5, -0.2, 0.0, 0.2, 0.5]
cmap = plt.cm.coolwarm
norm = plt.Normalize(vmin=-0.5, vmax=0.5)
for ib in crossing:
    for dE in energy_offsets:
        color = cmap(norm(dE))
        ax.contour(KX_cart, KY_cart, eig_2d[:, :, ib] - e_fermi - dE,
                   levels=[0.0], colors=[color], linewidths=1.5)
sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
sm.set_array([])
fig.colorbar(sm, ax=ax, label="E - E_F (eV)")
ax.set_xlabel(r"$k_x$ ($\AA^{-1}$)", fontsize=13)
ax.set_ylabel(r"$k_y$ ($\AA^{-1}$)", fontsize=13)
ax.set_title("Constant-Energy Contours", fontsize=14)
ax.set_aspect('equal')
plt.tight_layout()
plt.savefig("constant_energy_contours.png", dpi=200, bbox_inches="tight")
plt.close()
print("      Saved: constant_energy_contours.png")

print("\nDone.")
```

---

## Key Parameters

| Parameter | Typical Value | Notes |
|---|---|---|
| `NK` (2D mesh) | 40--80 per direction | Higher = smoother contours. 60 is a good starting point for hexagonal BZ. |
| `degauss` | 0.005--0.01 Ry (QE) | Small smearing for NSCF. Larger values smear the contour. |
| `nbnd` / `NBANDS` | All occupied + 2--4 empty | Only bands crossing E_F matter. |
| `assume_isolated` | `'2D'` (QE) | Coulomb cutoff for 2D systems. Prevents interlayer interactions. |
| Vacuum | > 15 A | Along the non-periodic direction. |
| `ISMEAR` | `1` (VASP) | Methfessel-Paxton for metals. |
| Energy offsets | -1.0 to +1.0 eV | Range for constant-energy contour plots. |

## Interpreting Results

- **Graphene Fermi contour**: At charge neutrality, the Fermi surface consists of points at K and K' -- the Dirac points. With doping, the contour expands into circles (trigonally warped at higher doping).
- **Nesting**: Parallel sections of the Fermi contour connected by a vector **q** indicate nesting, which can drive CDW or SDW instabilities.
- **Topology**: Closed contours indicate electron or hole pockets. The enclosed area is proportional to the carrier density (Luttinger's theorem in 2D).
- **Lifshitz transition**: Under doping or strain, the Fermi contour topology can change (e.g., from electron-like to hole-like), signaling a Lifshitz transition.
- **Anisotropy**: Elongated contours indicate direction-dependent transport. Circular contours indicate isotropic in-plane transport.
- **van Hove singularity**: When the Fermi contour passes through a saddle point (M point in graphene), the DOS diverges logarithmically.

## Common Issues

| Problem | Solution |
|---|---|
| No bands cross E_F | The material may be semiconducting. Check if you are using the correct doping level or if SOC opens a gap. |
| Contour is jagged or fragmented | Increase k-mesh density. Use `scipy.ndimage.gaussian_filter` with small sigma to smooth the data. |
| Contour wraps incorrectly across BZ boundary | The k-mesh goes from 0 to 1, so contours near the boundary may split. Extend the mesh periodically (tile 2x2) before contouring. |
| Wrong BZ shape | The Cartesian conversion must use the correct reciprocal lattice vectors. Verify with pymatgen. |
| Vacuum direction has dispersion | Increase vacuum to > 20 A or use `assume_isolated = '2D'` in QE. |
| NSCF is very slow | Use k-point parallelization. For 60x60 = 3600 k-points, use `-npool 4` or more. |
