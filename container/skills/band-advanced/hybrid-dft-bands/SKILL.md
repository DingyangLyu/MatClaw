# Hybrid-DFT Band Structure (HSE06 / PBE0)

## When to Use

- You need an accurate band gap that accounts for the self-interaction error in standard PBE/GGA.
- You are studying a semiconductor or insulator where PBE underestimates the gap by 30--50%.
- You need quantitative comparison with experimental photoemission or optical gap measurements.
- You want projected (fat) band structures with hybrid functional accuracy.
- You are studying materials where PBE gives qualitatively wrong results (e.g., predicts metallic when the material is an insulator).

## Method Selection

| Criterion | QE DFT (HSE06) | VASP DFT (HSE06) | PBE (standard) |
|---|---|---|---|
| Band gap accuracy | Good (within ~0.3 eV of experiment) | Good (within ~0.3 eV of experiment) | Poor (underestimates by 30--50%) |
| Computational cost | Very expensive (10--100x PBE) | Very expensive (10--100x PBE) | Cheap |
| Implementation | `input_dft='hse'` | `LHFCALC=.TRUE.`, `HFSCREEN=0.2` | Default |
| When to use | Publication-quality gaps | Publication-quality gaps | Screening, structure relaxation |
| Projected bands | `projwfc.x` on hybrid wavefunctions | PROCAR from hybrid calc | `projwfc.x` on PBE wavefunctions |

**MACE cannot compute band structures.** Hybrid functionals are a DFT concept and require electronic structure codes.

## Prerequisites

- Quantum ESPRESSO 7.5 (`pw.x`, `bands.x`, `projwfc.x`) or VASP with hybrid functional support.
- Well-relaxed structure (relax with PBE first, then optionally re-relax with hybrid).
- Pseudopotential files (norm-conserving recommended for QE hybrid; PAW for VASP).
- Python: `numpy`, `matplotlib`, `pymatgen`, `ase`, `seekpath`.
- Significant computational resources: hybrid calculations are 10--100x more expensive than PBE.

---

## Detailed Steps

### Method A: QE HSE06 Band Structure

The QE workflow for hybrid bands differs from PBE bands because the exact exchange operator is nonlocal and expensive. The strategy is:

1. **SCF with hybrid on a uniform k-grid** (converge charge density).
2. **Bands calculation**: QE evaluates the hybrid Hamiltonian at both the uniform k-grid (for exchange) and the band k-path points simultaneously.

#### Step A1: Generate K-Path

```python
#!/usr/bin/env python3
"""
Generate the high-symmetry k-path for hybrid band structure.
For QE hybrid bands, we need the band-path k-points appended to the
uniform mesh k-points with zero weight.
"""
import numpy as np
import seekpath
from ase.build import bulk

# Build or load structure
atoms = bulk("Si", "diamond", a=5.431)

# Get standardized path from seekpath
cell = atoms.cell[:]
positions = atoms.get_scaled_positions()
numbers = atoms.get_atomic_numbers()
path_data = seekpath.get_path((cell, positions, numbers))

kpoint_coords = path_data["point_coords"]
kpath_segments = path_data["path"]

print("=== High-symmetry points ===")
for label, coords in sorted(kpoint_coords.items()):
    print(f"  {label:6s}: ({coords[0]:.4f}, {coords[1]:.4f}, {coords[2]:.4f})")

print("\n=== Path segments ===")
for seg in kpath_segments:
    print(f"  {seg[0]} -> {seg[1]}")

# ── Generate band-path k-points with zero weight ──────────────────
def generate_band_kpoints(path_data, npoints_per_segment=20):
    """
    Generate k-points along the band path with zero weight.
    These will be appended to the uniform mesh for QE hybrid bands.
    Returns list of (kx, ky, kz, weight, label_or_empty).
    """
    kpath_segments = path_data["path"]
    kpoint_coords = path_data["point_coords"]
    band_kpoints = []
    labels = []

    for i, (start_label, end_label) in enumerate(kpath_segments):
        start = np.array(kpoint_coords[start_label])
        end = np.array(kpoint_coords[end_label])

        for j in range(npoints_per_segment):
            t = j / npoints_per_segment
            kpt = start + t * (end - start)
            label = ""
            if j == 0:
                label = start_label
            band_kpoints.append((kpt[0], kpt[1], kpt[2], 0.0, label))

        # Add end point of last segment
        if i == len(kpath_segments) - 1:
            band_kpoints.append((end[0], end[1], end[2], 0.0, end_label))

    return band_kpoints

band_kpts = generate_band_kpoints(path_data, npoints_per_segment=20)
print(f"\nGenerated {len(band_kpts)} band-path k-points (zero weight)")

# ── Generate uniform k-grid with weights ──────────────────────────
def generate_uniform_kgrid(nk1, nk2, nk3):
    """Generate uniform Monkhorst-Pack grid with weights."""
    kpts = []
    total = nk1 * nk2 * nk3
    weight = 1.0 / total
    for i in range(nk1):
        for j in range(nk2):
            for k in range(nk3):
                kx = (i + 0.5) / nk1 if nk1 > 1 else 0.0
                ky = (j + 0.5) / nk2 if nk2 > 1 else 0.0
                kz = (k + 0.5) / nk3 if nk3 > 1 else 0.0
                # Shift to first BZ
                kx = kx - 1.0 if kx > 0.5 else kx
                ky = ky - 1.0 if ky > 0.5 else ky
                kz = kz - 1.0 if kz > 0.5 else kz
                kpts.append((kx, ky, kz, weight))
    return kpts

uniform_kpts = generate_uniform_kgrid(4, 4, 4)
print(f"Generated {len(uniform_kpts)} uniform k-points")

# ── Combine: uniform (with weight) + band path (zero weight) ──────
all_kpoints = []
for kx, ky, kz, w in uniform_kpts:
    all_kpoints.append(f"  {kx:.10f}  {ky:.10f}  {kz:.10f}  {w:.10f}")
for kx, ky, kz, w, label in band_kpts:
    comment = f"  ! {label}" if label else ""
    all_kpoints.append(f"  {kx:.10f}  {ky:.10f}  {kz:.10f}  {w:.10f}{comment}")

kpoints_card = f"K_POINTS (crystal)\n{len(all_kpoints)}\n" + "\n".join(all_kpoints) + "\n"

with open("hybrid_kpoints_card.txt", "w") as f:
    f.write(kpoints_card)
print(f"\nTotal k-points: {len(all_kpoints)} ({len(uniform_kpts)} uniform + {len(band_kpts)} band path)")
print("Saved to hybrid_kpoints_card.txt")

# Save labels and their indices for plotting
label_indices = []
idx_offset = len(uniform_kpts)
for i, (kx, ky, kz, w, label) in enumerate(band_kpts):
    if label:
        label_indices.append((i + idx_offset, label))

with open("hybrid_band_labels.txt", "w") as f:
    for idx, label in label_indices:
        f.write(f"{idx} {label}\n")
print("Saved label indices to hybrid_band_labels.txt")
```

#### Step A2: SCF + Bands with Hybrid Functional (QE)

```python
#!/usr/bin/env python3
"""
Run HSE06 band structure calculation with QE.
Uses a single pw.x call with uniform+band-path k-points.
"""
import os
import subprocess
import re

PSEUDO_DIR = os.path.abspath("./pseudo")
OUTDIR = os.path.abspath("./tmp_hse_bands")
os.makedirs(OUTDIR, exist_ok=True)
PREFIX = "si_hse"

# Read the combined k-points card
with open("hybrid_kpoints_card.txt", "r") as f:
    kpoints_card = f.read()

# ── HSE06 SCF + bands in one shot ─────────────────────────────────
# In QE, for hybrid bands, you run a single 'scf' calculation with
# both uniform (weighted) and band-path (zero-weight) k-points.
# QE computes exact exchange on the uniform mesh and evaluates
# eigenvalues at all k-points.

hse_input = f"""&CONTROL
    calculation  = 'scf'
    prefix       = '{PREFIX}'
    outdir       = '{OUTDIR}'
    pseudo_dir   = '{PSEUDO_DIR}'
    verbosity    = 'high'
    tprnfor      = .true.
/
&SYSTEM
    ibrav        = 0
    nat          = 2
    ntyp         = 1
    ecutwfc      = 60.0
    ecutrho      = 480.0
    input_dft    = 'hse'
    nqx1 = 4, nqx2 = 4, nqx3 = 4
    exxdiv_treatment = 'gygi-baldereschi'
    x_gamma_extrapolation = .true.
    occupations  = 'smearing'
    smearing     = 'cold'
    degauss      = 0.005
    nbnd         = 12
/
&ELECTRONS
    conv_thr         = 1.0d-8
    mixing_beta      = 0.4
    electron_maxstep = 200
    adaptive_thr     = .true.
    conv_thr_init    = 1.0d-4
/

CELL_PARAMETERS (angstrom)
  0.0000000000   2.7155000000   2.7155000000
  2.7155000000   0.0000000000   2.7155000000
  2.7155000000   2.7155000000   0.0000000000

ATOMIC_SPECIES
  Si  28.0855  Si.pbe-n-rrkjus_psl.1.0.0.UPF

ATOMIC_POSITIONS (crystal)
  Si  0.0000000000  0.0000000000  0.0000000000
  Si  0.2500000000  0.2500000000  0.2500000000

{kpoints_card}
"""

with open(f"{PREFIX}_scf.in", "w") as f:
    f.write(hse_input)

print("Running HSE06 SCF + bands (this may take a long time)...")
result = subprocess.run(
    ["mpirun", "-np", "4", "pw.x", "-in", f"{PREFIX}_scf.in"],
    capture_output=True, text=True, timeout=7200
)
with open(f"{PREFIX}_scf.out", "w") as f:
    f.write(result.stdout)

if "convergence has been achieved" in result.stdout:
    print("HSE06 calculation converged.")
else:
    print("WARNING: Check convergence in output file.")

# Extract Fermi energy
e_fermi = 0.0
for line in result.stdout.split("\n"):
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
print(f"Fermi energy: {e_fermi:.4f} eV")
```

#### Step A3: Parse and Plot HSE06 Band Structure

```python
#!/usr/bin/env python3
"""
Parse eigenvalues from QE HSE06 output and plot band structure.
Only the zero-weight k-points (band path) are used for plotting.
"""
import re
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

PREFIX = "si_hse"
N_UNIFORM = 64  # 4x4x4 uniform grid k-points

# ── Parse Fermi energy ─────────────────────────────────────────────
def get_fermi_energy(scf_output):
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

# ── Parse all eigenvalues from verbose output ──────────────────────
def parse_eigenvalues_verbose(output_file):
    """
    Parse k-points and eigenvalues from QE verbose output.
    Returns: kpoints list, eigenvalues list (each entry is a list of eV values).
    """
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

    return kpoints, eigenvalues

kpoints, eigenvalues = parse_eigenvalues_verbose(f"{PREFIX}_scf.out")
print(f"Total k-points parsed: {len(kpoints)}")

# ── Extract only band-path k-points (zero-weight, after uniform) ──
band_kpoints = kpoints[N_UNIFORM:]
band_eigenvalues = eigenvalues[N_UNIFORM:]
nbnd = min(len(e) for e in band_eigenvalues) if band_eigenvalues else 0
print(f"Band-path k-points: {len(band_kpoints)}, bands: {nbnd}")

# ── Compute k-distance along path ─────────────────────────────────
# Load primitive reciprocal lattice
from pymatgen.core import Lattice
# Si FCC primitive cell
lat = Lattice([[0.0, 2.7155, 2.7155],
               [2.7155, 0.0, 2.7155],
               [2.7155, 2.7155, 0.0]])
recip = lat.reciprocal_lattice.matrix

def kfrac_to_cart(kfrac, recip_matrix):
    return np.dot(kfrac, recip_matrix)

k_cart = np.array([kfrac_to_cart(k, recip) for k in band_kpoints])
k_dist = np.zeros(len(k_cart))
for i in range(1, len(k_cart)):
    k_dist[i] = k_dist[i-1] + np.linalg.norm(k_cart[i] - k_cart[i-1])

# ── Read labels ────────────────────────────────────────────────────
label_positions = []
label_names = []
if os.path.exists("hybrid_band_labels.txt"):
    with open("hybrid_band_labels.txt", "r") as f:
        for line in f:
            parts = line.strip().split()
            idx = int(parts[0]) - N_UNIFORM  # offset to band-path index
            name = parts[1]
            if 0 <= idx < len(k_dist):
                label_positions.append(k_dist[idx])
                name_fmt = r"$\Gamma$" if name.upper() in ("GAMMA", "G") else name
                label_names.append(name_fmt)

# ── Plot ───────────────────────────────────────────────────────────
import os
eig_array = np.array([e[:nbnd] for e in band_eigenvalues])

fig, ax = plt.subplots(figsize=(8, 6))
for ib in range(nbnd):
    ax.plot(k_dist, eig_array[:, ib] - e_fermi, color="darkblue", linewidth=1.5)

ax.axhline(0, color="red", linestyle="--", linewidth=0.8, label="$E_F$")
for xpos in label_positions:
    ax.axvline(xpos, color="black", linewidth=0.5, alpha=0.5)

if len(label_names) == len(label_positions):
    ax.set_xticks(label_positions)
    ax.set_xticklabels(label_names, fontsize=13)

ax.set_xlim(k_dist[0], k_dist[-1])
ax.set_ylim(-6, 8)
ax.set_ylabel("$E - E_F$ (eV)", fontsize=14)
ax.set_title("HSE06 Band Structure", fontsize=15)
ax.legend(fontsize=11)
ax.grid(axis="y", alpha=0.3)
plt.tight_layout()
plt.savefig("hse06_bands.png", dpi=200, bbox_inches="tight")
plt.close()
print("Saved: hse06_bands.png")

# ── Band gap ───────────────────────────────────────────────────────
vbm = np.max(eig_array[eig_array <= e_fermi])
cbm = np.min(eig_array[eig_array > e_fermi])
gap = cbm - vbm
print(f"\n=== HSE06 Band Gap ===")
print(f"Band gap: {gap:.4f} eV")
print(f"VBM: {vbm:.4f} eV, CBM: {cbm:.4f} eV")
print(f"(Si experimental gap: 1.17 eV; HSE06 typically gives ~1.15 eV)")
```

#### Step A4: Projected (Fat) Band Structure with QE Hybrid

```python
#!/usr/bin/env python3
"""
Run projwfc.x on the hybrid calculation to get orbital-projected bands.
Then parse and plot fat bands showing orbital character.
"""
import os
import subprocess
import re
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection

PREFIX = "si_hse"
OUTDIR = os.path.abspath("./tmp_hse_bands")
N_UNIFORM = 64

# ── Run projwfc.x ─────────────────────────────────────────────────
projwfc_input = f"""&PROJWFC
    prefix  = '{PREFIX}'
    outdir  = '{OUTDIR}'
    filpdos = '{PREFIX}_pdos'
    Emin    = -15.0
    Emax    = 15.0
    DeltaE  = 0.01
    lwrite_overlaps = .false.
    lbinary_data    = .false.
/
"""

with open(f"{PREFIX}_projwfc.in", "w") as f:
    f.write(projwfc_input)

print("Running projwfc.x...")
result = subprocess.run(
    ["mpirun", "-np", "4", "projwfc.x", "-in", f"{PREFIX}_projwfc.in"],
    capture_output=True, text=True, timeout=3600
)
with open(f"{PREFIX}_projwfc.out", "w") as f:
    f.write(result.stdout)

if result.returncode == 0:
    print("projwfc.x completed.")
else:
    print("ERROR in projwfc.x!")

# ── Parse projwfc.x output for orbital weights ────────────────────
def parse_projwfc_output(projwfc_out, n_uniform, nbnd):
    """
    Parse projwfc.x output to extract orbital character per k-point per band.
    Returns dict: orbital_name -> (n_band_kpoints x nbnd) weight array.
    """
    with open(projwfc_out, "r") as f:
        content = f.read()

    # projwfc.x prints atomic wavefunctions and their projections
    # We look for the projection summary: |psi_nk|^2 for each state

    # For simplicity, parse the PDOS files instead
    # projwfc.x writes files like: prefix_pdos.pdos_atm#1(Si)_wfc#1(s)
    import glob
    pdos_files = sorted(glob.glob(f"{PREFIX}_pdos.pdos_atm*"))

    orbital_data = {}
    for pf in pdos_files:
        # Extract orbital name from filename
        m = re.search(r"atm#(\d+)\((\w+)\)_wfc#(\d+)\((\w+)\)", pf)
        if m:
            atom_idx = int(m.group(1))
            atom_name = m.group(2)
            wfc_idx = int(m.group(3))
            orbital = m.group(4)
            label = f"{atom_name}-{orbital}"

            data = np.loadtxt(pf, comments="#")
            if label not in orbital_data:
                orbital_data[label] = data[:, 1]  # LDOS column
            else:
                orbital_data[label] += data[:, 1]

    return orbital_data

print("\nPDOS files can be plotted to show orbital character.")
print("For fat bands, use the eigenvalue + projection data from projwfc.x output.")
```

### Method B: VASP HSE06 Band Structure

#### Step B1: VASP Input Files

The VASP HSE06 band structure uses a two-step approach:
1. SCF on a uniform k-grid with hybrid functional.
2. Non-SCF on uniform + band-path k-points (zero-weight band path).

**INCAR (SCF step):**
```
SYSTEM = Si HSE06 SCF
ENCUT = 400
EDIFF = 1E-6
ISMEAR = 0
SIGMA = 0.05
IBRION = -1
NSW = 0
LHFCALC = .TRUE.
HFSCREEN = 0.2
ALGO = All
TIME = 0.4
PRECFOCK = Fast
NKRED = 2
LWAVE = .TRUE.
LCHARG = .TRUE.
PREC = Accurate
```

**KPOINTS (SCF, Gamma-centered):**
```
Automatic mesh
0
Gamma
4 4 4
0.0 0.0 0.0
```

**INCAR (Bands step -- read WAVECAR):**
```
SYSTEM = Si HSE06 Bands
ENCUT = 400
EDIFF = 1E-6
ISMEAR = 0
SIGMA = 0.05
IBRION = -1
NSW = 0
LHFCALC = .TRUE.
HFSCREEN = 0.2
ALGO = All
TIME = 0.4
PRECFOCK = Fast
NKRED = 2
ICHARG = 11
NBANDS = 16
LORBIT = 11
LWAVE = .FALSE.
PREC = Accurate
```

#### Step B2: Generate KPOINTS with Uniform + Band Path (VASP)

```python
#!/usr/bin/env python3
"""
Generate KPOINTS file for VASP HSE06 band structure:
uniform k-grid (with weights) + band path (zero weight).
Compatible with VASPKIT tasks 250-257.
"""
import numpy as np
import seekpath
from pymatgen.core import Structure

# Load structure
structure = Structure.from_file("POSCAR")

# Get seekpath k-path
cell = structure.lattice.matrix
positions = structure.frac_coords
numbers = [s.Z for s in structure.species]
path_data = seekpath.get_path((cell, positions, numbers))

kpoint_coords = path_data["point_coords"]
kpath_segments = path_data["path"]

# ── Uniform k-grid (Gamma-centered) ───────────────────────────────
nk1, nk2, nk3 = 4, 4, 4
uniform_kpts = []
weight = 1.0 / (nk1 * nk2 * nk3)
for i in range(nk1):
    for j in range(nk2):
        for k in range(nk3):
            kx = i / nk1
            ky = j / nk2
            kz = k / nk3
            uniform_kpts.append((kx, ky, kz, weight))

# ── Band path k-points (zero weight) ──────────────────────────────
npts_per_seg = 20
band_kpts = []
labels_in_order = []

for i, (start_label, end_label) in enumerate(kpath_segments):
    start = np.array(kpoint_coords[start_label])
    end = np.array(kpoint_coords[end_label])

    for j in range(npts_per_seg):
        t = j / npts_per_seg
        kpt = start + t * (end - start)
        label = start_label if j == 0 else ""
        band_kpts.append((kpt[0], kpt[1], kpt[2], 0.0, label))

    if i == len(kpath_segments) - 1:
        band_kpts.append((end[0], end[1], end[2], 0.0, end_label))

# ── Write KPOINTS ─────────────────────────────────────────────────
with open("KPOINTS", "w") as f:
    f.write("HSE06 band structure: uniform + band path\n")
    f.write(f"{len(uniform_kpts) + len(band_kpts)}\n")
    f.write("Reciprocal lattice\n")

    for kx, ky, kz, w in uniform_kpts:
        f.write(f"  {kx:.10f}  {ky:.10f}  {kz:.10f}  {w:.10f}\n")
    for kx, ky, kz, w, label in band_kpts:
        comment = f"  ! {label}" if label else ""
        f.write(f"  {kx:.10f}  {ky:.10f}  {kz:.10f}  {w:.10f}{comment}\n")

total = len(uniform_kpts) + len(band_kpts)
print(f"Written KPOINTS: {len(uniform_kpts)} uniform + {len(band_kpts)} band path = {total} total")

# Save label info
with open("band_labels.dat", "w") as f:
    f.write(f"# n_uniform = {len(uniform_kpts)}\n")
    for i, (kx, ky, kz, w, label) in enumerate(band_kpts):
        if label:
            f.write(f"{i} {label}\n")
print("Saved band_labels.dat")
```

#### Step B3: Parse VASP EIGENVAL and PROCAR for Projected Bands

```python
#!/usr/bin/env python3
"""
Parse VASP EIGENVAL and PROCAR for HSE06 projected (fat) band structure.
"""
import re
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ── Configuration ──────────────────────────────────────────────────
N_UNIFORM = 64  # 4x4x4

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

# ── Parse PROCAR for orbital projections ───────────────────────────
def parse_procar(filename="PROCAR", nkpts=None, nbands=None, nions=None):
    """
    Parse VASP PROCAR file for orbital-resolved projections.
    Returns: projections dict with keys like 's', 'p', 'd'
             each value is (nkpts x nbands) array summed over atoms.
    """
    with open(filename, "r") as f:
        lines = f.readlines()

    # Find header to get orbital names
    orbital_names = []
    for line in lines:
        if "ion" in line and "s" in line and "tot" in line:
            parts = line.split()
            ion_idx = parts.index("ion")
            tot_idx = parts.index("tot")
            orbital_names = parts[ion_idx+1:tot_idx]
            break

    if not orbital_names:
        print("Could not parse orbital names from PROCAR")
        return {}

    print(f"Orbitals found: {orbital_names}")

    # Parse projections
    # PROCAR format: blocks per k-point, then per band, then per ion
    projections = {orb: [] for orb in orbital_names}

    # Find nkpts, nbands, nions from header
    for line in lines:
        if "# of k-points" in line:
            parts = line.split()
            nkpts_file = int(parts[3])
            nbands_file = int(parts[7])
            nions_file = int(parts[11])
            break

    nkpts = nkpts or nkpts_file
    nbands = nbands or nbands_file
    nions = nions or nions_file

    # Initialize arrays
    proj_arrays = {orb: np.zeros((nkpts, nbands)) for orb in orbital_names}

    ik = -1
    ib = -1
    for line in lines:
        if line.strip().startswith("k-point"):
            m = re.search(r"k-point\s+(\d+)", line)
            if m:
                ik = int(m.group(1)) - 1
                ib = -1
        elif line.strip().startswith("band"):
            m = re.search(r"band\s+(\d+)", line)
            if m:
                ib = int(m.group(1)) - 1
        elif line.strip().startswith("tot") and ik >= 0 and ib >= 0:
            # This is the "tot" line summed over all ions
            parts = line.split()
            if len(parts) >= len(orbital_names) + 1:
                for io, orb in enumerate(orbital_names):
                    proj_arrays[orb][ik, ib] = float(parts[io + 1])

    return proj_arrays

# ── Main ───────────────────────────────────────────────────────────
kpoints, eigenvalues, nelect = parse_eigenval("EIGENVAL")

# Get Fermi energy
def get_efermi_outcar(filename="OUTCAR"):
    with open(filename, "r") as f:
        for line in f:
            if "E-fermi" in line:
                m = re.search(r"E-fermi\s*:\s*([-\d.]+)", line)
                if m:
                    return float(m.group(1))
    return 0.0

e_fermi = get_efermi_outcar()
print(f"E_Fermi = {e_fermi:.4f} eV")

# Extract band-path data
band_kpts = kpoints[N_UNIFORM:]
band_eigs = eigenvalues[N_UNIFORM:]
nbands = band_eigs.shape[1]

# Compute k-distance
from pymatgen.io.vasp import Poscar
poscar = Poscar.from_file("POSCAR")
recip = poscar.structure.lattice.reciprocal_lattice.matrix

k_cart = np.dot(band_kpts, recip)
k_dist = np.zeros(len(k_cart))
for i in range(1, len(k_cart)):
    k_dist[i] = k_dist[i-1] + np.linalg.norm(k_cart[i] - k_cart[i-1])

# ── Parse PROCAR projections ──────────────────────────────────────
proj = parse_procar("PROCAR")
if proj:
    # Extract band-path projections
    band_proj = {orb: arr[N_UNIFORM:] for orb, arr in proj.items()}

    # Group into s, p, d
    s_weight = band_proj.get('s', np.zeros_like(band_eigs))
    p_weight = sum(band_proj.get(orb, np.zeros_like(band_eigs))
                   for orb in ['py', 'pz', 'px'] if orb in band_proj)
    d_weight = sum(band_proj.get(orb, np.zeros_like(band_eigs))
                   for orb in ['dxy', 'dyz', 'dz2', 'dxz', 'dx2'] if orb in band_proj)

    # Normalize
    total = s_weight + p_weight + d_weight
    total[total == 0] = 1.0
    s_frac = s_weight / total
    p_frac = p_weight / total
    d_frac = d_weight / total

    # ── Fat band plot with RGB coloring ────────────────────────────
    fig, ax = plt.subplots(figsize=(8, 6))

    for ib in range(nbands):
        for i in range(len(k_dist) - 1):
            r_val = s_frac[i, ib]
            g_val = p_frac[i, ib]
            b_val = d_frac[i, ib]
            color = (r_val, g_val, b_val)
            ax.plot(k_dist[i:i+2], band_eigs[i:i+2, ib] - e_fermi,
                    color=color, linewidth=2.0)

    ax.axhline(0, color="gray", linestyle="--", linewidth=0.8)

    # Legend patches
    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor='red', label='s'),
                       Patch(facecolor='green', label='p'),
                       Patch(facecolor='blue', label='d')]
    ax.legend(handles=legend_elements, fontsize=11, loc='upper right')

    ax.set_xlim(k_dist[0], k_dist[-1])
    ax.set_ylim(-6, 8)
    ax.set_ylabel("$E - E_F$ (eV)", fontsize=14)
    ax.set_title("HSE06 Projected Band Structure", fontsize=15)
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig("hse06_fat_bands.png", dpi=200, bbox_inches="tight")
    plt.close()
    print("Saved: hse06_fat_bands.png")
else:
    print("PROCAR not parsed. Run VASP with LORBIT=11 for projected bands.")
```

### PBE0 Variant

To use PBE0 instead of HSE06:

**QE:** Replace `input_dft = 'hse'` with `input_dft = 'pbe0'` and remove `nqx1/nqx2/nqx3` (PBE0 uses the full Fock exchange without screening).

**VASP:** Set `HFSCREEN = 0` (no screening) instead of `HFSCREEN = 0.2`. Optionally adjust `AEXX = 0.25` (default for PBE0).

To adjust the exact exchange fraction:
- **QE:** Add `exx_fraction = 0.30` (e.g., 30% instead of default 25%).
- **VASP:** Set `AEXX = 0.30`.

---

## Key Parameters

| Parameter | QE | VASP | Default / Typical | Notes |
|---|---|---|---|---|
| Functional | `input_dft = 'hse'` | `LHFCALC = .TRUE.` | -- | Enables hybrid functional |
| Screening | `nqx1/2/3` define q-mesh | `HFSCREEN = 0.2` | 0.2 A^-1 for HSE06 | 0 = PBE0 (unscreened) |
| Exchange fraction | `exx_fraction` | `AEXX` | 0.25 | 25% exact exchange (standard) |
| Q-mesh (QE) | `nqx1=4, nqx2=4, nqx3=4` | N/A | Same as k-grid | Defines Fock operator sampling |
| Convergence trick | `adaptive_thr = .true.` | `PRECFOCK = Fast` | -- | Speeds up hybrid SCF |
| Algorithm | -- | `ALGO = All` | -- | Required for hybrid in VASP |
| Band count | `nbnd = 12` | `NBANDS = 16` | 1.5--2x occupied | More bands for wider energy range |
| K-grid for exchange | 4x4x4 typical | 4x4x4 typical | -- | Smaller than PBE because of cost |

## Interpreting Results

- **HSE06 vs PBE band gaps**: HSE06 typically gives band gaps within 0.3 eV of experiment. PBE underestimates by 30--50%.
  - Si: PBE ~0.6 eV, HSE06 ~1.15 eV, Experiment ~1.17 eV.
  - GaAs: PBE ~0.5 eV, HSE06 ~1.3 eV, Experiment ~1.42 eV.
  - ZnO: PBE ~0.7 eV, HSE06 ~2.5 eV, Experiment ~3.4 eV (HSE06 still underestimates wide-gap materials).
- **Band dispersion**: Hybrid functionals generally do not change the shape of bands dramatically compared to PBE, but they open the gap. The effective masses are often more accurate with HSE06.
- **Projected bands**: The orbital character should be similar to PBE, but the relative positions of s, p, d states may shift due to the improved exchange potential.
- **When HSE06 is insufficient**: For strongly correlated materials (Mott insulators, heavy fermions), even HSE06 may fail. Consider GW or DFT+U approaches.

## Common Issues

| Problem | Solution |
|---|---|
| Extremely slow convergence | Use `adaptive_thr = .true.` in QE. In VASP, use `PRECFOCK = Fast` and `NKRED = 2`. |
| Out of memory | Reduce the uniform k-grid (3x3x3 or even 2x2x2). Reduce `nbnd`. |
| SCF oscillates, does not converge | Reduce `mixing_beta` to 0.2-0.3 in QE. In VASP, try `ALGO = Damped` + `TIME = 0.2`. |
| Band path looks wrong | Ensure band-path k-points have exactly zero weight. Nonzero weight changes the charge density. |
| Gap is still too small | Check `exx_fraction` / `AEXX`. Try increasing to 0.30. Some materials require tuned exchange fractions. |
| QE crashes with hybrid | Ensure norm-conserving pseudopotentials are used (USPP/PAW may not work with all hybrid implementations in older QE). QE 7.x supports PAW+hybrid. |
| VASP PRECFOCK warning | `PRECFOCK = Fast` may reduce accuracy slightly. For final production, use `PRECFOCK = Accurate`. |
| k-grid too coarse for exchange | The exact exchange is sampled on the uniform k-grid. If too coarse, the gap may be inaccurate. Test convergence with 4x4x4 vs 6x6x6. |
