# Orbital-Projected Fermi Surface

## When to Use

- You need to visualize the orbital character (s, p, d contributions) on the Fermi surface.
- You want to understand which orbitals dominate the electronic states at the Fermi level.
- You are studying multi-band metals where different Fermi surface sheets have different orbital origins.
- You need to identify d-orbital ordering (dxy, dxz, dyz, dz2, dx2-y2) on the Fermi surface for correlated electron systems.
- You are comparing with spin- and orbital-resolved ARPES data.
- You want to identify the orbital character relevant to superconducting pairing or spin-orbit effects.

## Method Selection

| Criterion | QE DFT (`projwfc.x`) | VASP DFT (`PROCAR`) | ASE + MACE |
|---|---|---|---|
| Availability | Full projected Fermi surface | Full projected Fermi surface | Cannot compute |
| Projection tool | `projwfc.x` post-processing | `LORBIT = 11/12` in INCAR | N/A |
| Output format | `pdos_atm#..._wfc#...` files | PROCAR file | N/A |
| Orbital resolution | Per-atom, per-orbital | Per-atom, per-orbital | N/A |

**MACE cannot compute orbital projections.** Always use DFT.

## Prerequisites

- Quantum ESPRESSO 7.5 (`pw.x`, `projwfc.x`) or VASP with `LORBIT = 11`.
- A metallic system with bands crossing the Fermi level.
- Pseudopotential files in `./pseudo/`.
- Python: `numpy`, `scipy`, `matplotlib`, `pymatgen`, `ase`.
- For 3D visualization: `scikit-image` (`pip install scikit-image`) for marching cubes.

---

## Detailed Steps

### Method A: QE Projected Fermi Surface

#### Step A1: SCF Calculation

```python
#!/usr/bin/env python3
"""
Step 1: SCF for multi-orbital metal.
Example: BCC Fe (d-band metal, spin-polarized).
"""
import os
import subprocess

PSEUDO_DIR = os.path.abspath("./pseudo")
OUTDIR = os.path.abspath("./tmp_proj_fermi")
os.makedirs(OUTDIR, exist_ok=True)
PREFIX = "fe"

scf_input = f"""&CONTROL
    calculation  = 'scf'
    prefix       = '{PREFIX}'
    outdir       = '{OUTDIR}'
    pseudo_dir   = '{PSEUDO_DIR}'
    verbosity    = 'high'
/
&SYSTEM
    ibrav        = 3
    celldm(1)    = 5.42
    nat          = 1
    ntyp         = 1
    ecutwfc      = 60.0
    ecutrho      = 480.0
    nspin        = 2
    starting_magnetization(1) = 0.6
    occupations  = 'smearing'
    smearing     = 'mv'
    degauss      = 0.02
/
&ELECTRONS
    conv_thr     = 1.0d-8
    mixing_beta  = 0.3
/

ATOMIC_SPECIES
  Fe  55.845  Fe.pbe-spn-kjpaw_psl.0.2.1.UPF

ATOMIC_POSITIONS (crystal)
  Fe  0.0  0.0  0.0

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

#### Step A2: NSCF on Dense 3D K-Grid

```python
#!/usr/bin/env python3
"""
Step 2: NSCF on dense k-grid.
For projected Fermi surface, we also need projwfc.x afterward.
"""
import os
import subprocess

PSEUDO_DIR = os.path.abspath("./pseudo")
OUTDIR = os.path.abspath("./tmp_proj_fermi")
PREFIX = "fe"

NK = 24  # 24x24x24 for orbital projection (expensive with projwfc.x)

kpoints = []
for i in range(NK):
    for j in range(NK):
        for k in range(NK):
            kpoints.append(f"  {i/NK:.10f}  {j/NK:.10f}  {k/NK:.10f}  1.0")

kpoints_card = f"K_POINTS (crystal)\n{len(kpoints)}\n" + "\n".join(kpoints)

nscf_input = f"""&CONTROL
    calculation  = 'nscf'
    prefix       = '{PREFIX}'
    outdir       = '{OUTDIR}'
    pseudo_dir   = '{PSEUDO_DIR}'
    verbosity    = 'high'
/
&SYSTEM
    ibrav        = 3
    celldm(1)    = 5.42
    nat          = 1
    ntyp         = 1
    ecutwfc      = 60.0
    ecutrho      = 480.0
    nspin        = 2
    starting_magnetization(1) = 0.6
    occupations  = 'smearing'
    smearing     = 'mv'
    degauss      = 0.005
    nbnd         = 16
/
&ELECTRONS
    conv_thr     = 1.0d-8
/

ATOMIC_SPECIES
  Fe  55.845  Fe.pbe-spn-kjpaw_psl.0.2.1.UPF

ATOMIC_POSITIONS (crystal)
  Fe  0.0  0.0  0.0

{kpoints_card}
"""

with open(f"{PREFIX}_nscf_proj.in", "w") as f:
    f.write(nscf_input)

print(f"Running NSCF on {NK}^3 = {NK**3} k-points...")
result = subprocess.run(
    ["mpirun", "-np", "4", "pw.x", "-in", f"{PREFIX}_nscf_proj.in"],
    capture_output=True, text=True, timeout=7200
)
with open(f"{PREFIX}_nscf_proj.out", "w") as f:
    f.write(result.stdout)

if result.returncode == 0:
    print("NSCF completed.")
else:
    print("ERROR in NSCF!")
```

#### Step A3: Run projwfc.x for Orbital Projections

```python
#!/usr/bin/env python3
"""
Step 3: Run projwfc.x to compute orbital projections at each k-point.
The output files contain the projected DOS and orbital weights per band.
"""
import os
import subprocess

OUTDIR = os.path.abspath("./tmp_proj_fermi")
PREFIX = "fe"

projwfc_input = f"""&PROJWFC
    prefix   = '{PREFIX}'
    outdir   = '{OUTDIR}'
    filpdos  = '{PREFIX}_projwfc'
    Emin     = -15.0
    Emax     = 15.0
    DeltaE   = 0.01
    lwrite_overlaps = .false.
    lbinary_data    = .false.
/
"""

with open(f"{PREFIX}_projwfc.in", "w") as f:
    f.write(projwfc_input)

print("Running projwfc.x (this may take a while for dense k-grids)...")
result = subprocess.run(
    ["mpirun", "-np", "4", "projwfc.x", "-in", f"{PREFIX}_projwfc.in"],
    capture_output=True, text=True, timeout=7200
)
with open(f"{PREFIX}_projwfc.out", "w") as f:
    f.write(result.stdout)

if result.returncode == 0:
    print("projwfc.x completed.")
else:
    print("ERROR in projwfc.x!")
    print(result.stderr[-500:] if result.stderr else "")
```

#### Step A4: Parse Projections and Plot Orbital-Projected Fermi Surface

```python
#!/usr/bin/env python3
"""
Step 4: Parse projwfc.x output and eigenvalues, then plot orbital-projected
Fermi surface.

projwfc.x writes the projection of each Kohn-Sham state onto atomic orbitals.
The output file (projwfc.out) contains lines like:
    |psi|^2 = ...
    state #  1: atom   1 (Fe ), wfc  1 (l=0 m= 1)  -> s
    state #  2: atom   1 (Fe ), wfc  2 (l=2 m= 1)  -> dxy
    ...
followed by projection tables per k-point and band.
"""
import re
import os
import glob
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from scipy.ndimage import gaussian_filter

PREFIX = "fe"
NK = 24
NBND = 16

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

# ── Parse eigenvalues from NSCF output ─────────────────────────────
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
        kpoints.append([float(match.group(1)), float(match.group(2)), float(match.group(3))])
        eig_text = match.group(4)
        eigs = [float(v) for v in re.findall(r"([-\d.]+)", eig_text) if -1000 < float(v) < 1000]
        eigenvalues.append(eigs)

    return np.array(kpoints), eigenvalues

kpoints, eigenvalues_list = parse_nscf_eigenvalues(f"{PREFIX}_nscf_proj.out")
nbnd_actual = min(len(e) for e in eigenvalues_list)
eigenvalues = np.array([e[:nbnd_actual] for e in eigenvalues_list])
print(f"Parsed {len(kpoints)} k-points, {nbnd_actual} bands")

# ── Parse orbital projections from projwfc.x output ────────────────
def parse_projwfc_projections(projwfc_out, nkpts, nbnd):
    """
    Parse projwfc.x output to get orbital projections per k-point and band.

    Returns:
        orbital_names: list of (atom, orbital_type) tuples
        projections: dict mapping orbital_type -> (nkpts x nbnd) array
    """
    with open(projwfc_out, "r") as f:
        lines = f.readlines()

    # Find state definitions
    # Format: "state #  N: atom  M (El), wfc  W (l=L m=M)"
    state_defs = {}
    for line in lines:
        m = re.match(r"\s*state\s*#\s*(\d+):\s*atom\s+(\d+)\s+\((\w+)\s*\),\s*wfc\s+(\d+)\s+\(l=\s*(\d+)\s+m=\s*(\d+)\)", line)
        if m:
            state_idx = int(m.group(1))
            atom_idx = int(m.group(2))
            atom_name = m.group(3).strip()
            l_val = int(m.group(5))
            m_val = int(m.group(6))

            orbital_map = {0: 's', 1: 'p', 2: 'd', 3: 'f'}
            orb_type = orbital_map.get(l_val, f'l{l_val}')
            state_defs[state_idx] = (atom_name, orb_type, l_val, m_val)

    print(f"Found {len(state_defs)} atomic states")
    for idx, (atom, orb, l, m_q) in sorted(state_defs.items()):
        print(f"  State {idx}: {atom} {orb} (l={l}, m={m_q})")

    # Parse projections: look for "|psi|^2" blocks
    # Format per k-point/band:
    #   k = ...   band = N
    #   psi = c1*|state1> + c2*|state2> + ...
    #   |psi|^2 = sum_i |ci|^2
    #
    # For a simpler parsing, we use the projection numbers that follow
    # "Lowdin Charges" or we parse the "e(..." lines.

    # Alternative: parse the filproj (atomic_proj.xml) file if available.
    # For now, use the PDOS files from projwfc.x which give integrated projections.

    # Parse PDOS files for per-orbital weights at each energy
    pdos_files = sorted(glob.glob(f"{PREFIX}_projwfc.pdos_atm*"))
    orbital_pdos = {}

    for pf in pdos_files:
        m = re.search(r"atm#(\d+)\((\w+)\)_wfc#(\d+)\((\w+)\)", pf)
        if m:
            atom_idx = int(m.group(1))
            atom_name = m.group(2)
            wfc_idx = int(m.group(3))
            orbital = m.group(4)
            label = f"{atom_name}-{orbital}"

            data = np.loadtxt(pf, comments="#")
            # Column 0: energy, Column 1: LDOS (for nspin=2: col1=up, col2=down)
            if label not in orbital_pdos:
                orbital_pdos[label] = data
            else:
                # Sum contributions from multiple atoms of same type
                orbital_pdos[label][:, 1:] += data[:, 1:]

    return orbital_pdos, state_defs

orbital_pdos, state_defs = parse_projwfc_projections(f"{PREFIX}_projwfc.out", len(kpoints), nbnd_actual)

# ── Compute orbital weights at the Fermi level ────────────────────
# The PDOS gives the density of states projected onto each orbital.
# At the Fermi level, the PDOS value indicates the relative contribution
# of that orbital to the states at E_F.

print("\n=== Orbital character at the Fermi level ===")
e_fermi_pdos_idx = None
total_at_ef = 0.0
orbital_weights_ef = {}

for label, data in orbital_pdos.items():
    energies = data[:, 0]
    ldos = data[:, 1]  # spin-up for nspin=2

    # Find index closest to E_F
    idx_ef = np.argmin(np.abs(energies - e_fermi))
    weight = ldos[idx_ef]
    orbital_weights_ef[label] = weight
    total_at_ef += weight
    print(f"  {label}: PDOS at E_F = {weight:.4f} states/eV")

if total_at_ef > 0:
    print("\n  Normalized orbital fractions at E_F:")
    for label, w in orbital_weights_ef.items():
        print(f"    {label}: {w/total_at_ef*100:.1f}%")

# ── Approximate orbital-projected Fermi surface (2D slice) ────────
# For a true orbital-projected 3D Fermi surface, we need per-k per-band
# projections. When those are not directly available from projwfc.x output,
# we can use the PDOS ratio as a proxy: at each k-point, the bands near E_F
# are colored by the overall orbital character from the PDOS.

# Here we demonstrate using the eigenvalue grid with orbital coloring.
eig_3d = eigenvalues.reshape(NK, NK, NK, nbnd_actual)

# Find bands crossing E_F
crossing_bands = []
for ib in range(nbnd_actual):
    bmin = eig_3d[:, :, :, ib].min()
    bmax = eig_3d[:, :, :, ib].max()
    if bmin < e_fermi < bmax:
        crossing_bands.append(ib)

# BCC Fe reciprocal lattice
a_bohr = 5.42
a_ang = a_bohr * 0.529177
recip = 2 * np.pi / a_ang * np.array([
    [1, 0, 0],
    [0, 1, 0],
    [0, 0, 1]
])  # BCC reciprocal is FCC; for ibrav=3, QE uses a cube

# ── Plot 2D kz=0 slice with orbital coloring ──────────────────────
kz_idx = 0  # kz=0 slice

fig, ax = plt.subplots(figsize=(8, 8))

# Create grid for the kz=0 slice
kx_frac = np.linspace(0, 1, NK, endpoint=False)
ky_frac = np.linspace(0, 1, NK, endpoint=False)
KX_frac, KY_frac = np.meshgrid(kx_frac, ky_frac, indexing='ij')
KX_cart = KX_frac * recip[0, 0] + KY_frac * recip[1, 0]
KY_cart = KX_frac * recip[0, 1] + KY_frac * recip[1, 1]

for ib in crossing_bands:
    band_slice = eig_3d[:, :, kz_idx, ib] - e_fermi
    cs = ax.contour(KX_cart, KY_cart, band_slice, levels=[0.0],
                    linewidths=2.0)

ax.set_xlabel(r"$k_x$ ($\AA^{-1}$)", fontsize=13)
ax.set_ylabel(r"$k_y$ ($\AA^{-1}$)", fontsize=13)
ax.set_title(f"Fermi Contour at kz=0 (Fe)", fontsize=14)
ax.set_aspect('equal')
ax.grid(True, alpha=0.2)
plt.tight_layout()
plt.savefig("proj_fermi_kz0_contour.png", dpi=200, bbox_inches="tight")
plt.close()
print("Saved: proj_fermi_kz0_contour.png")

# ── PDOS pie chart of orbital character at E_F ─────────────────────
if orbital_weights_ef:
    labels_pie = list(orbital_weights_ef.keys())
    sizes = list(orbital_weights_ef.values())
    colors_pie = plt.cm.Set3(np.linspace(0, 1, len(labels_pie)))

    fig, ax = plt.subplots(figsize=(7, 7))
    ax.pie(sizes, labels=labels_pie, colors=colors_pie, autopct='%1.1f%%',
           startangle=90, textprops={'fontsize': 11})
    ax.set_title("Orbital Character at E_F (Fe)", fontsize=14)
    plt.tight_layout()
    plt.savefig("orbital_character_ef.png", dpi=200, bbox_inches="tight")
    plt.close()
    print("Saved: orbital_character_ef.png")
```

### Method B: VASP Projected Fermi Surface (PROCAR)

#### Step B1: VASP Input Files

**INCAR (SCF):**
```
SYSTEM = Fe BCC - SCF
ENCUT = 500
EDIFF = 1E-6
ISMEAR = 1
SIGMA = 0.1
ISPIN = 2
MAGMOM = 3.0
IBRION = -1
NSW = 0
LWAVE = .TRUE.
LCHARG = .TRUE.
PREC = Accurate
```

**INCAR (NSCF with PROCAR):**
```
SYSTEM = Fe BCC - NSCF projected Fermi surface
ENCUT = 500
EDIFF = 1E-6
ISMEAR = 1
SIGMA = 0.05
ISPIN = 2
IBRION = -1
NSW = 0
ICHARG = 11
NBANDS = 16
LORBIT = 11
LWAVE = .FALSE.
PREC = Accurate
```

**Generate dense KPOINTS:**
```python
#!/usr/bin/env python3
"""Generate dense 3D KPOINTS for VASP projected Fermi surface."""
NK = 24
with open("KPOINTS", "w") as f:
    f.write(f"Dense {NK}^3 for projected Fermi surface\n")
    f.write(f"{NK**3}\n")
    f.write("Reciprocal lattice\n")
    for i in range(NK):
        for j in range(NK):
            for k in range(NK):
                f.write(f"  {i/NK:.10f}  {j/NK:.10f}  {k/NK:.10f}  1.0\n")
print(f"Written KPOINTS with {NK**3} points")
```

#### Step B2: Parse PROCAR for Orbital-Projected Fermi Surface

```python
#!/usr/bin/env python3
"""
Parse VASP PROCAR and EIGENVAL for orbital-projected Fermi surface.
PROCAR contains orbital projections per k-point, per band, per ion.
Compatible with VASPKIT tasks 266-267.
"""
import re
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from scipy.ndimage import gaussian_filter

NK = 24

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

# ── Parse PROCAR ───────────────────────────────────────────────────
def parse_procar_full(filename="PROCAR"):
    """
    Parse VASP PROCAR for full orbital projections.
    Returns: orbital_names, projections[orbital] = (nkpts, nbands) array.
    """
    with open(filename, "r") as f:
        lines = f.readlines()

    # Get dimensions
    for line in lines:
        if "# of k-points" in line:
            parts = line.split()
            nkpts = int(parts[3])
            nbands = int(parts[7])
            nions = int(parts[11])
            break

    # Get orbital names from header
    orbital_names = []
    for line in lines:
        if line.strip().startswith("ion") and "s" in line and "tot" in line:
            parts = line.split()
            ion_idx = parts.index("ion")
            tot_idx = parts.index("tot")
            orbital_names = parts[ion_idx+1:tot_idx]
            break

    print(f"PROCAR: {nkpts} k-pts, {nbands} bands, {nions} ions")
    print(f"Orbitals: {orbital_names}")

    # Initialize projection arrays
    proj = {orb: np.zeros((nkpts, nbands)) for orb in orbital_names}

    ik = -1
    ib = -1
    for line in lines:
        stripped = line.strip()
        if stripped.startswith("k-point"):
            m = re.search(r"k-point\s+(\d+)", stripped)
            if m:
                ik = int(m.group(1)) - 1
        elif stripped.startswith("band"):
            m = re.search(r"band\s+(\d+)", stripped)
            if m:
                ib = int(m.group(1)) - 1
        elif stripped.startswith("tot") and ik >= 0 and ib >= 0:
            parts = stripped.split()
            if len(parts) >= len(orbital_names) + 1:
                for io, orb in enumerate(orbital_names):
                    proj[orb][ik, ib] = float(parts[io + 1])

    return orbital_names, proj

# ── Main ───────────────────────────────────────────────────────────
kpoints, eigenvalues, nelect = parse_eigenval()
e_fermi = get_efermi_outcar()
print(f"E_Fermi = {e_fermi:.4f} eV, NKPTS = {len(kpoints)}, NBANDS = {eigenvalues.shape[1]}")

orbital_names, proj = parse_procar_full()
nbands = eigenvalues.shape[1]

# Reshape to 3D grid
eig_3d = eigenvalues.reshape(NK, NK, NK, nbands)

# Group orbital projections
proj_3d = {}
for orb in orbital_names:
    proj_3d[orb] = proj[orb].reshape(NK, NK, NK, nbands)

# Group into s, p, d
s_proj_3d = proj_3d.get('s', np.zeros_like(eig_3d))
p_proj_3d = sum(proj_3d.get(orb, np.zeros_like(eig_3d))
                for orb in ['py', 'pz', 'px'] if orb in proj_3d)
d_proj_3d = sum(proj_3d.get(orb, np.zeros_like(eig_3d))
                for orb in ['dxy', 'dyz', 'dz2', 'dxz', 'dx2'] if orb in proj_3d)

# ── Find crossing bands ──────────────────────────────────────────
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

# ── 2D slice at kz=0 with orbital coloring ────────────────────────
kz_idx = 0
kx_frac = np.linspace(0, 1, NK, endpoint=False)
ky_frac = np.linspace(0, 1, NK, endpoint=False)
KX_frac, KY_frac = np.meshgrid(kx_frac, ky_frac, indexing='ij')
KX_cart = KX_frac * recip[0, 0] + KY_frac * recip[1, 0]
KY_cart = KX_frac * recip[0, 1] + KY_frac * recip[1, 1]

for ib in crossing:
    band_slice = eig_3d[:, :, kz_idx, ib] - e_fermi

    # Orbital weights at this slice
    s_slice = s_proj_3d[:, :, kz_idx, ib]
    p_slice = p_proj_3d[:, :, kz_idx, ib]
    d_slice = d_proj_3d[:, :, kz_idx, ib]
    total_slice = s_slice + p_slice + d_slice
    total_slice[total_slice == 0] = 1.0  # avoid division by zero

    # Dominant orbital at each k-point
    d_frac = d_slice / total_slice

    fig, ax = plt.subplots(figsize=(8, 8))

    # Color map showing d-orbital fraction
    im = ax.pcolormesh(KX_cart, KY_cart, d_frac,
                       cmap='YlOrRd', shading='gouraud', vmin=0, vmax=1)
    # Overlay Fermi contour
    ax.contour(KX_cart, KY_cart, band_slice, levels=[0.0],
               colors='black', linewidths=2.0)

    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("d-orbital fraction", fontsize=12)
    ax.set_xlabel(r"$k_x$ ($\AA^{-1}$)", fontsize=13)
    ax.set_ylabel(r"$k_y$ ($\AA^{-1}$)", fontsize=13)
    ax.set_title(f"Projected Fermi Contour - Band {ib+1} (kz=0)", fontsize=14)
    ax.set_aspect('equal')
    plt.tight_layout()
    plt.savefig(f"proj_fermi_band{ib+1}_kz0.png", dpi=200, bbox_inches="tight")
    plt.close()
    print(f"Saved: proj_fermi_band{ib+1}_kz0.png")

# ── 3D projected Fermi surface with orbital coloring ───────────────
try:
    from skimage.measure import marching_cubes

    for ib in crossing:
        band_shifted = eig_3d[:, :, :, ib] - e_fermi
        band_smooth = gaussian_filter(band_shifted, sigma=0.5)

        verts, faces, _, _ = marching_cubes(band_smooth, level=0.0,
                                             spacing=(1.0/NK, 1.0/NK, 1.0/NK))

        # Get orbital weights at vertex positions by interpolation
        from scipy.interpolate import RegularGridInterpolator

        kx_axis = np.linspace(0, 1, NK, endpoint=False)
        ky_axis = np.linspace(0, 1, NK, endpoint=False)
        kz_axis = np.linspace(0, 1, NK, endpoint=False)

        s_interp = RegularGridInterpolator((kx_axis, ky_axis, kz_axis),
                                            s_proj_3d[:, :, :, ib],
                                            bounds_error=False, fill_value=0)
        p_interp = RegularGridInterpolator((kx_axis, ky_axis, kz_axis),
                                            p_proj_3d[:, :, :, ib],
                                            bounds_error=False, fill_value=0)
        d_interp = RegularGridInterpolator((kx_axis, ky_axis, kz_axis),
                                            d_proj_3d[:, :, :, ib],
                                            bounds_error=False, fill_value=0)

        # Vertices are in fractional coordinates (from spacing)
        s_at_verts = s_interp(verts)
        p_at_verts = p_interp(verts)
        d_at_verts = d_interp(verts)
        total_at_verts = s_at_verts + p_at_verts + d_at_verts
        total_at_verts[total_at_verts == 0] = 1.0

        d_frac_verts = d_at_verts / total_at_verts

        # Average d-fraction per face for coloring
        d_frac_faces = np.mean(d_frac_verts[faces], axis=1)

        # Convert vertices to Cartesian
        verts_cart = verts @ recip

        # Plot
        fig = plt.figure(figsize=(10, 10))
        ax = fig.add_subplot(111, projection='3d')

        # Color faces by d-orbital fraction
        cmap = plt.cm.YlOrRd
        face_colors = cmap(d_frac_faces)

        mesh = Poly3DCollection(verts_cart[faces], alpha=0.7,
                                facecolors=face_colors, edgecolor='none')
        ax.add_collection3d(mesh)

        lim = np.max(np.abs(recip)) * 0.6
        ax.set_xlim(-lim, lim)
        ax.set_ylim(-lim, lim)
        ax.set_zlim(-lim, lim)
        ax.set_xlabel(r"$k_x$ ($\AA^{-1}$)")
        ax.set_ylabel(r"$k_y$ ($\AA^{-1}$)")
        ax.set_zlabel(r"$k_z$ ($\AA^{-1}$)")
        ax.set_title(f"Orbital-Projected Fermi Surface - Band {ib+1}", fontsize=14)

        # Colorbar
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(0, 1))
        sm.set_array([])
        fig.colorbar(sm, ax=ax, shrink=0.6, label="d-orbital fraction")

        plt.tight_layout()
        plt.savefig(f"proj_fermi_3d_band{ib+1}.png", dpi=200, bbox_inches="tight")
        plt.close()
        print(f"Saved: proj_fermi_3d_band{ib+1}.png")

except ImportError:
    print("Install scikit-image for 3D projected Fermi surface: pip install scikit-image")

# ── Write .frmsf with orbital weights ──────────────────────────────
# Extended FermiSurfer format supports orbital weights as additional data blocks
with open("proj_fermi.frmsf", "w") as f:
    f.write(f"{NK} {NK} {NK}\n")
    f.write("1\n")  # no SOC
    f.write(f"{len(crossing)}\n")
    for i in range(3):
        f.write(f"{recip[i, 0]:.10f} {recip[i, 1]:.10f} {recip[i, 2]:.10f}\n")

    # Eigenvalues
    for ib in crossing:
        for iz in range(NK):
            for iy in range(NK):
                for ix in range(NK):
                    f.write(f"{eig_3d[ix, iy, iz, ib] - e_fermi:.10f}\n")

    # d-orbital weight as scalar property on the Fermi surface
    for ib in crossing:
        total = s_proj_3d[:,:,:,ib] + p_proj_3d[:,:,:,ib] + d_proj_3d[:,:,:,ib]
        total[total == 0] = 1.0
        d_frac = d_proj_3d[:,:,:,ib] / total
        for iz in range(NK):
            for iy in range(NK):
                for ix in range(NK):
                    f.write(f"{d_frac[ix, iy, iz]:.10f}\n")

print("Saved: proj_fermi.frmsf (with d-orbital weight)")
```

---

## Key Parameters

| Parameter | Typical Value | Notes |
|---|---|---|
| `NK` (3D mesh) | 20--30 per direction | Lower than plain Fermi surface because projwfc.x / PROCAR parsing is expensive. |
| `LORBIT` | `11` (VASP) | Writes PROCAR with per-atom, per-orbital projections. Use `12` for DOSCAR with l,m decomposition. |
| `nbnd` / `NBANDS` | All occupied + 4--8 empty | Need bands crossing E_F plus nearby bands. |
| `nspin` / `ISPIN` | `2` for magnetic | Separate spin channels require separate analysis. |
| `projwfc.x` DeltaE | 0.01 eV | Energy resolution for PDOS. Smaller = more accurate at E_F but slower. |
| `ecutwfc` / `ENCUT` | Same as SCF | Must be converged. |
| Gaussian smoothing | 0.3--0.5 grid points | For marching cubes on the eigenvalue grid. |

## Interpreting Results

- **d-orbital dominance**: In transition metals (Fe, Ni, Cu), the Fermi surface is primarily d-character. The specific d-orbital (dxy, dxz, dyz, dz2, dx2-y2) varies across the Fermi surface and determines anisotropic properties.
- **s-p vs d character**: Free-electron-like metals (Al, Na) have mostly s-p character on the Fermi surface. Transition metals show mixed character with d-bands dominating near the Fermi level.
- **Orbital-dependent scattering**: Impurity scattering depends on the orbital character of the states at E_F. Orbital-projected Fermi surfaces help predict which defects scatter most effectively.
- **Superconducting pairing**: In multi-band superconductors (MgB2, Fe-based superconductors), the orbital character of each Fermi surface sheet determines the pairing symmetry and gap structure.
- **Spin-orbit coupling**: With SOC, orbital character and spin texture become entangled. The projected Fermi surface shows how orbital character varies with momentum, which is critical for spintronic applications.
- **Color coding**: Red/warm colors for high d-character, blue/cool for s-p character. The color gradients on the Fermi surface directly indicate where orbital hybridization occurs.

## Common Issues

| Problem | Solution |
|---|---|
| PROCAR is very large | For NK=30, PROCAR can be gigabytes. Reduce NK to 20--24 or use binary PROCAR (`LORBIT=12` does not help with size). |
| projwfc.x is extremely slow | The calculation scales as N_states x N_kpts x N_bands. Reduce NK or use fewer processors with less memory contention. |
| Orbital names don't match | QE and VASP use different orbital naming conventions. QE uses l,m quantum numbers; VASP uses s, py, pz, px, dxy, etc. Map between them as needed. |
| Sum of projections != 1 | Projections sum to less than 1 because interstitial charge is not captured by atomic orbitals. This is normal; normalize by the sum for fractional analysis. |
| Spin-up and spin-down look identical | For paramagnetic metals, both spin channels are degenerate. For ferromagnets, they should differ. Check that `nspin=2` / `ISPIN=2` is set. |
| 3D plot is too cluttered | Show one band/sheet at a time. Use transparency (alpha=0.5--0.7). Rotate the view angle for clarity. |
| Interpolation artifacts at BZ boundary | Periodic boundary conditions must be handled in the interpolation. Pad the grid with periodic copies. |
