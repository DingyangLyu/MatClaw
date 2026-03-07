# Band Structure Calculation

## When to Use

- You need the electronic band structure (energy vs. k-path) of a crystal.
- You want to determine if a material is a metal, semiconductor, or insulator.
- You need the band gap value (direct or indirect) and its location in the Brillouin zone.
- You are comparing DFT results to experimental ARPES or optical data.

## Method Selection (MACE vs QE)

| Criterion | ASE + MACE | QE DFT |
|---|---|---|
| Availability | Cannot compute electronic bands | Full band structure |
| Reason | MACE is a force field with no electronic degrees of freedom | DFT solves the Kohn-Sham equations |
| Use case | Relax the structure before DFT band calc | Actual band structure |

**MACE cannot produce band structures.** It predicts energies and forces via an interatomic potential that has no concept of electronic wavefunctions or eigenvalues. Always use QE for band structures. MACE can be used to pre-relax the structure to save DFT time.

## Prerequisites

- A relaxed crystal structure (use the `scf-relax` skill first).
- Pseudopotential files in `./pseudo/` (see `scf-relax` skill for download instructions).
- Python packages: `pymatgen`, `seekpath` (install via `pip install seekpath`), `numpy`, `matplotlib`.

---

## Detailed Steps

### Method A: ASE + MACE (Structure Preparation Only)

MACE cannot compute band structures, but it is useful for quickly relaxing the input structure before a DFT band calculation.

```python
#!/usr/bin/env python3
"""
Use MACE to relax a structure, then prepare it for a QE band calculation.
"""
import warnings
warnings.filterwarnings("ignore")

from ase.build import bulk
from ase.io import write
from ase.optimize import LBFGS
from ase.filters import FrechetCellFilter
from mace.calculators import mace_mp

# ── 1. Relax with MACE ─────────────────────────────────────────────
atoms = bulk("Si", "diamond", a=5.43)
calc = mace_mp(model="medium", dispersion=False, device="cpu")
atoms.calc = calc

filtered = FrechetCellFilter(atoms)
opt = LBFGS(filtered, logfile="relax_for_bands.log")
opt.run(fmax=0.005)

print(f"Relaxed lattice: {atoms.cell.cellpar()}")
write("relaxed_for_bands.cif", atoms)
print("Structure saved to relaxed_for_bands.cif -- use this for QE band calculation.")
```

### Method B: QE DFT Band Structure

The QE band structure workflow has four steps:
1. **SCF**: Self-consistent calculation to get the charge density.
2. **bands**: Non-self-consistent calculation along the k-path (reads SCF charge density).
3. **bands.x**: Post-processing to extract eigenvalues in plottable format.
4. **Plot**: Python script to produce the band structure figure.

#### Step B1: Generate the High-Symmetry K-Path

```python
#!/usr/bin/env python3
"""
Generate the high-symmetry k-path for band structure calculations.
Two methods: seekpath (recommended) and pymatgen HighSymmKpath.
"""
import numpy as np

# ── Method 1: Using seekpath (recommended) ─────────────────────────
# pip install seekpath  (if not already installed)
import seekpath
from ase.build import bulk
from ase.io import read

# Load or build structure
atoms = bulk("Si", "diamond", a=5.431)
# atoms = read("relaxed_for_bands.cif")

# seekpath needs: (cell, positions, numbers)
cell = atoms.cell[:]
positions = atoms.get_scaled_positions()
numbers = atoms.get_atomic_numbers()

path_data = seekpath.get_path((cell, positions, numbers))

# Extract k-path info
kpoint_labels = path_data["point_coords"]  # dict: label -> fractional coords
path_segments = path_data["path"]  # list of (start_label, end_label) tuples

print("=== High-symmetry points ===")
for label, coords in sorted(kpoint_labels.items()):
    print(f"  {label:6s}: ({coords[0]:.4f}, {coords[1]:.4f}, {coords[2]:.4f})")

print("\n=== Path segments ===")
for seg in path_segments:
    print(f"  {seg[0]} -> {seg[1]}")

# ── Build QE K_POINTS card for bands calculation ───────────────────
def build_qe_kpath(path_data, npoints_per_segment=30):
    """
    Build the K_POINTS {crystal_b} card for QE bands calculation.
    Returns the string to paste into a QE input file.
    """
    kpath_segments = path_data["path"]
    kpoint_coords = path_data["point_coords"]

    kpoints_lines = []
    n_kpoints = 0

    for i, (start, end) in enumerate(kpath_segments):
        sc = kpoint_coords[start]
        ec = kpoint_coords[end]

        # Check if this segment connects to the previous one
        if i > 0:
            prev_end = kpath_segments[i - 1][1]
            if start != prev_end:
                # Discontinuity: add previous end with 0 weight, then start
                # The previous end was already added; add start with npoints
                kpoints_lines.append(
                    f"  {sc[0]:.10f}  {sc[1]:.10f}  {sc[2]:.10f}  {npoints_per_segment}  ! {start}"
                )
                n_kpoints += 1
            # else: continuous path, start was already added as previous end
        else:
            # First segment: add start point
            kpoints_lines.append(
                f"  {sc[0]:.10f}  {sc[1]:.10f}  {sc[2]:.10f}  {npoints_per_segment}  ! {start}"
            )
            n_kpoints += 1

        # Add end point
        npts = npoints_per_segment if i < len(kpath_segments) - 1 else 0
        # For the last point of the last segment, use 0
        if i == len(kpath_segments) - 1:
            npts = 0
        kpoints_lines.append(
            f"  {ec[0]:.10f}  {ec[1]:.10f}  {ec[2]:.10f}  {npts}  ! {end}"
        )
        n_kpoints += 1

    header = f"K_POINTS {{crystal_b}}\n{n_kpoints}"
    return header + "\n" + "\n".join(kpoints_lines) + "\n"

kpath_card = build_qe_kpath(path_data, npoints_per_segment=30)
print("\n=== QE K_POINTS card ===")
print(kpath_card)

# Save for later use
with open("kpath_card.txt", "w") as f:
    f.write(kpath_card)

# ── Method 2: Using pymatgen HighSymmKpath ──────────────────────────
from pymatgen.core import Structure
from pymatgen.symmetry.bandstructure import HighSymmKpath

struct = Structure.from_file("relaxed_for_bands.cif") if False else \
    Structure(
        lattice=atoms.cell[:],
        species=atoms.get_chemical_symbols(),
        coords=atoms.get_scaled_positions(),
    )

kpath_pmg = HighSymmKpath(struct)
kpts_dict = kpath_pmg.kpath
print("\n=== pymatgen high-symmetry points ===")
for label, coords in kpts_dict["kpoints"].items():
    print(f"  {label}: {coords}")
print("Path:", kpts_dict["path"])
```

#### Step B2: SCF Calculation (charge density)

```python
#!/usr/bin/env python3
"""
Step 1 of band structure: SCF to get the charge density.
This is essentially the same as a normal SCF but we ensure
outdir and prefix are consistent with the bands step.
"""
import os
import subprocess

PSEUDO_DIR = os.path.abspath("./pseudo")
OUTDIR = os.path.abspath("./tmp_bands")
os.makedirs(OUTDIR, exist_ok=True)

scf_input = f"""&CONTROL
    calculation = 'scf'
    prefix      = 'si'
    outdir      = '{OUTDIR}'
    pseudo_dir  = '{PSEUDO_DIR}'
    tprnfor     = .true.
    tstress     = .true.
/

&SYSTEM
    ibrav       = 2
    celldm(1)   = 10.26
    nat         = 2
    ntyp        = 1
    ecutwfc     = 50.0
    ecutrho     = 400.0
    occupations = 'smearing'
    smearing    = 'cold'
    degauss     = 0.01
/

&ELECTRONS
    conv_thr = 1.0d-8
/

ATOMIC_SPECIES
  Si  28.0855  Si.pbe-n-rrkjus_psl.1.0.0.UPF

ATOMIC_POSITIONS (crystal)
  Si  0.00  0.00  0.00
  Si  0.25  0.25  0.25

K_POINTS (automatic)
  8 8 8  0 0 0
"""

with open("si_bands_scf.in", "w") as f:
    f.write(scf_input)

print("Running SCF for band structure...")
result = subprocess.run(
    ["mpirun", "-np", "4", "pw.x", "-in", "si_bands_scf.in"],
    capture_output=True, text=True, timeout=600
)
with open("si_bands_scf.out", "w") as f:
    f.write(result.stdout)

if "convergence has been achieved" in result.stdout:
    print("SCF converged. Charge density saved.")
else:
    print("WARNING: SCF may not have converged. Check si_bands_scf.out")
```

#### Step B3: Bands Calculation (non-SCF along k-path)

```python
#!/usr/bin/env python3
"""
Step 2 of band structure: non-SCF calculation along the k-path.
Reads the charge density from the SCF step.
"""
import os
import subprocess

PSEUDO_DIR = os.path.abspath("./pseudo")
OUTDIR = os.path.abspath("./tmp_bands")  # MUST match SCF step

# Read the k-path card generated earlier
with open("kpath_card.txt", "r") as f:
    kpath_card = f.read()

bands_input = f"""&CONTROL
    calculation = 'bands'
    prefix      = 'si'
    outdir      = '{OUTDIR}'
    pseudo_dir  = '{PSEUDO_DIR}'
    verbosity   = 'high'
/

&SYSTEM
    ibrav       = 2
    celldm(1)   = 10.26
    nat         = 2
    ntyp        = 1
    ecutwfc     = 50.0
    ecutrho     = 400.0
    nbnd        = 12
/

&ELECTRONS
    conv_thr = 1.0d-8
/

ATOMIC_SPECIES
  Si  28.0855  Si.pbe-n-rrkjus_psl.1.0.0.UPF

ATOMIC_POSITIONS (crystal)
  Si  0.00  0.00  0.00
  Si  0.25  0.25  0.25

{kpath_card}
"""

with open("si_bands.in", "w") as f:
    f.write(bands_input)

print("Running bands calculation...")
result = subprocess.run(
    ["mpirun", "-np", "4", "pw.x", "-in", "si_bands.in"],
    capture_output=True, text=True, timeout=600
)
with open("si_bands.out", "w") as f:
    f.write(result.stdout)

if result.returncode == 0:
    print("Bands calculation completed.")
else:
    print("ERROR in bands calculation!")
    print(result.stderr[-500:] if result.stderr else "")
```

#### Step B4: Post-processing with bands.x

```python
#!/usr/bin/env python3
"""
Step 3: Run bands.x to extract eigenvalues in a plottable format.
"""
import os
import subprocess

OUTDIR = os.path.abspath("./tmp_bands")

bands_pp_input = f"""&BANDS
    prefix  = 'si'
    outdir  = '{OUTDIR}'
    filband = 'si_bands.dat'
    lsym    = .true.
/
"""

with open("si_bands_pp.in", "w") as f:
    f.write(bands_pp_input)

print("Running bands.x post-processing...")
result = subprocess.run(
    ["bands.x", "-in", "si_bands_pp.in"],
    capture_output=True, text=True, timeout=120
)
with open("si_bands_pp.out", "w") as f:
    f.write(result.stdout)

if result.returncode == 0:
    print("bands.x completed. Output files:")
    print("  si_bands.dat       -- raw band data")
    print("  si_bands.dat.gnu   -- gnuplot-friendly format")
    print("  si_bands.dat.rap   -- symmetry labels (if lsym=.true.)")
else:
    print("ERROR in bands.x!")
    print(result.stderr[-500:] if result.stderr else "")
```

#### Step B5: Plot Band Structure with Python

```python
#!/usr/bin/env python3
"""
Step 4: Plot band structure from QE bands.x output.
Complete plotting script with Fermi level alignment and high-symmetry labels.
"""
import numpy as np
import re
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ── Parse the Fermi energy from the SCF output ─────────────────────
def get_fermi_energy(scf_output_file):
    """Extract Fermi energy (eV) from QE SCF output."""
    with open(scf_output_file, "r") as f:
        for line in f:
            if "the Fermi energy is" in line:
                m = re.search(r"is\s+([-\d.]+)\s+ev", line, re.IGNORECASE)
                if m:
                    return float(m.group(1))
            # For insulators, QE reports highest occupied / lowest unoccupied
            if "highest occupied" in line:
                m = re.search(r"highest occupied.*?:\s+([-\d.]+)", line)
                if m:
                    return float(m.group(1))
            if "highest occupied, lowest unoccupied" in line:
                m = re.search(r":\s+([-\d.]+)\s+([-\d.]+)", line)
                if m:
                    vbm = float(m.group(1))
                    cbm = float(m.group(2))
                    return (vbm + cbm) / 2  # midgap
    return 0.0

# ── Parse band data from .gnu file ─────────────────────────────────
def parse_bands_gnu(gnu_file):
    """
    Parse the .gnu file produced by bands.x.
    Returns: k_distances (1D array), eigenvalues (2D array: n_kpoints x n_bands)
    """
    with open(gnu_file, "r") as f:
        text = f.read()

    # The .gnu file has blocks separated by blank lines, one block per band.
    blocks = text.strip().split("\n\n")
    bands = []
    k_dist = None

    for block in blocks:
        lines = block.strip().split("\n")
        k_vals = []
        e_vals = []
        for line in lines:
            parts = line.split()
            if len(parts) >= 2:
                k_vals.append(float(parts[0]))
                e_vals.append(float(parts[1]))
        if k_dist is None:
            k_dist = np.array(k_vals)
        bands.append(np.array(e_vals))

    eigenvalues = np.column_stack(bands)  # shape: (n_kpoints, n_bands)
    return k_dist, eigenvalues

# ── Parse high-symmetry point positions from bands.x output ────────
def parse_high_sym_points(bands_pp_output):
    """
    Parse high-symmetry point positions from bands.x output.
    Returns list of (k_distance, label) tuples.
    """
    sym_points = []
    with open(bands_pp_output, "r") as f:
        for line in f:
            # bands.x prints lines like:
            #  high-symmetry point:  0.0000 0.0000 0.0000   x coordinate   0.0000
            if "high-symmetry point" in line:
                m = re.search(r"x coordinate\s+([\d.]+)", line)
                coords = re.search(r"point:\s+([-\d.]+)\s+([-\d.]+)\s+([-\d.]+)", line)
                if m:
                    x = float(m.group(1))
                    sym_points.append((x, ""))
    return sym_points

# ── Alternative: manually define labels ─────────────────────────────
# If you know the path (e.g., from seekpath), define labels here.
# These must match the order of high-symmetry points in your k-path.
MANUAL_LABELS = [
    r"$\Gamma$", "X", "U|K", r"$\Gamma$", "L", "W", "X"
]
# For silicon FCC: GAMMA -> X -> U|K -> GAMMA -> L -> W -> X

# ── Main plotting ──────────────────────────────────────────────────
e_fermi = get_fermi_energy("si_bands_scf.out")
print(f"Fermi energy: {e_fermi:.4f} eV")

k_dist, eigenvalues = parse_bands_gnu("si_bands.dat.gnu")
print(f"Number of k-points: {len(k_dist)}")
print(f"Number of bands: {eigenvalues.shape[1]}")

# Shift eigenvalues to Fermi level
eigenvalues_shifted = eigenvalues - e_fermi

# Get high-symmetry point positions
sym_points = parse_high_sym_points("si_bands_pp.out")
sym_k_positions = [sp[0] for sp in sym_points]

# Use manual labels if we have them, otherwise use indices
if len(MANUAL_LABELS) == len(sym_k_positions):
    sym_labels = MANUAL_LABELS
else:
    sym_labels = [f"K{i}" for i in range(len(sym_k_positions))]

# ── Create the plot ─────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(8, 6))

# Plot each band
for i in range(eigenvalues_shifted.shape[1]):
    ax.plot(k_dist, eigenvalues_shifted[:, i], color="steelblue", linewidth=1.2)

# Fermi level
ax.axhline(y=0, color="red", linestyle="--", linewidth=0.8, alpha=0.7, label="$E_F$")

# High-symmetry point lines
for kpos in sym_k_positions:
    ax.axvline(x=kpos, color="black", linestyle="-", linewidth=0.5, alpha=0.5)

# Labels and formatting
ax.set_xticks(sym_k_positions)
ax.set_xticklabels(sym_labels, fontsize=13)
ax.set_xlim(k_dist[0], k_dist[-1])
ax.set_ylim(-12, 8)  # Adjust range as needed
ax.set_ylabel("Energy (eV)", fontsize=14)
ax.set_title("Band Structure (QE PBE)", fontsize=15)
ax.legend(fontsize=11, loc="upper right")
ax.grid(axis="y", alpha=0.3)

plt.tight_layout()
plt.savefig("band_structure.png", dpi=200, bbox_inches="tight")
print("Band structure plot saved: band_structure.png")

# ── Extract band gap ────────────────────────────────────────────────
def extract_band_gap(eigenvalues, n_electrons_per_kpt=None, e_fermi=0.0):
    """
    Extract band gap from eigenvalue array.
    For semiconductors/insulators, finds VBM and CBM.

    eigenvalues: shape (n_kpoints, n_bands), NOT shifted by Fermi level.
    """
    # Find bands below and above Fermi level
    vbm = -np.inf
    cbm = np.inf
    vbm_k_idx = 0
    cbm_k_idx = 0

    for ik in range(eigenvalues.shape[0]):
        for ib in range(eigenvalues.shape[1]):
            e = eigenvalues[ik, ib]
            if e <= e_fermi:
                if e > vbm:
                    vbm = e
                    vbm_k_idx = ik
            else:
                if e < cbm:
                    cbm = e
                    cbm_k_idx = ik

    gap = cbm - vbm
    is_direct = (vbm_k_idx == cbm_k_idx)
    return {
        "band_gap_eV": gap,
        "VBM_eV": vbm,
        "CBM_eV": cbm,
        "VBM_k_index": vbm_k_idx,
        "CBM_k_index": cbm_k_idx,
        "is_direct": is_direct,
    }

gap_info = extract_band_gap(eigenvalues, e_fermi=e_fermi)
print(f"\n=== Band Gap Analysis ===")
print(f"Band gap: {gap_info['band_gap_eV']:.4f} eV")
print(f"VBM: {gap_info['VBM_eV']:.4f} eV at k-index {gap_info['VBM_k_idx']}")
print(f"CBM: {gap_info['CBM_eV']:.4f} eV at k-index {gap_info['CBM_k_idx']}")
print(f"Type: {'Direct' if gap_info['is_direct'] else 'Indirect'}")
print(f"(Note: PBE typically underestimates band gaps by ~30-50%)")
```

#### Complete Single-Script Workflow

For convenience, here is a single script that runs the entire SCF -> bands -> plot pipeline:

```python
#!/usr/bin/env python3
"""
Complete band structure workflow in one script.
Runs: SCF -> bands -> bands.x -> plot
"""
import os
import subprocess
import re
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seekpath
from ase.build import bulk

# ── Configuration ──────────────────────────────────────────────────
PSEUDO_DIR = os.path.abspath("./pseudo")
OUTDIR = os.path.abspath("./tmp_bands_full")
os.makedirs(OUTDIR, exist_ok=True)
PREFIX = "si"
ECUTWFC = 50.0
ECUTRHO = 400.0
NPROC = 4
NBND = 12  # number of bands to compute

# ── Structure ──────────────────────────────────────────────────────
atoms = bulk("Si", "diamond", a=5.431)
cell = atoms.cell[:]
positions = atoms.get_scaled_positions()
numbers = atoms.get_atomic_numbers()

# ── K-path via seekpath ────────────────────────────────────────────
path_data = seekpath.get_path((cell, positions, numbers))
kpoint_coords = path_data["point_coords"]
kpath_segments = path_data["path"]
primitive_cell = path_data["primitive_lattice"]
primitive_positions = path_data["primitive_positions"]
primitive_numbers = path_data["primitive_types"]

# Build K_POINTS card
npts_per_seg = 30
klines = []
labels_in_order = []
for i, (s_label, e_label) in enumerate(kpath_segments):
    sc = kpoint_coords[s_label]
    ec = kpoint_coords[e_label]
    if i == 0:
        klines.append(f"  {sc[0]:.10f}  {sc[1]:.10f}  {sc[2]:.10f}  {npts_per_seg}  ! {s_label}")
        labels_in_order.append(s_label)
    elif s_label != kpath_segments[i-1][1]:
        # Discontinuity
        klines.append(f"  {sc[0]:.10f}  {sc[1]:.10f}  {sc[2]:.10f}  {npts_per_seg}  ! {s_label}")
        # Merge previous end label with current start for display
        labels_in_order[-1] = labels_in_order[-1] + "|" + s_label
    end_npts = npts_per_seg if i < len(kpath_segments) - 1 else 0
    klines.append(f"  {ec[0]:.10f}  {ec[1]:.10f}  {ec[2]:.10f}  {end_npts}  ! {e_label}")
    labels_in_order.append(e_label)

n_kpt_spec = len(klines)
kpath_card = f"K_POINTS {{crystal_b}}\n{n_kpt_spec}\n" + "\n".join(klines) + "\n"

# Build ATOMIC_SPECIES and ATOMIC_POSITIONS from seekpath primitive cell
from ase.data import chemical_symbols, atomic_masses
unique_numbers = sorted(set(primitive_numbers))
species_lines = []
for z in unique_numbers:
    sym = chemical_symbols[z]
    mass = atomic_masses[z]
    # You must have the UPF file for each element
    species_lines.append(f"  {sym}  {mass:.4f}  {sym}.pbe-n-rrkjus_psl.1.0.0.UPF")
species_card = "\n".join(species_lines)

pos_lines = []
for pos, z in zip(primitive_positions, primitive_numbers):
    sym = chemical_symbols[z]
    pos_lines.append(f"  {sym}  {pos[0]:.10f}  {pos[1]:.10f}  {pos[2]:.10f}")
pos_card = "\n".join(pos_lines)

nat = len(primitive_numbers)
ntyp = len(unique_numbers)

# Cell parameters in Angstrom
cell_lines = []
for row in primitive_cell:
    cell_lines.append(f"  {row[0]:.10f}  {row[1]:.10f}  {row[2]:.10f}")
cell_card = "\n".join(cell_lines)

# ── Step 1: SCF ────────────────────────────────────────────────────
scf_input = f"""&CONTROL
    calculation = 'scf'
    prefix      = '{PREFIX}'
    outdir      = '{OUTDIR}'
    pseudo_dir  = '{PSEUDO_DIR}'
/
&SYSTEM
    ibrav       = 0
    nat         = {nat}
    ntyp        = {ntyp}
    ecutwfc     = {ECUTWFC}
    ecutrho     = {ECUTRHO}
    occupations = 'smearing'
    smearing    = 'cold'
    degauss     = 0.01
/
&ELECTRONS
    conv_thr = 1.0d-8
/
CELL_PARAMETERS (angstrom)
{cell_card}

ATOMIC_SPECIES
{species_card}

ATOMIC_POSITIONS (crystal)
{pos_card}

K_POINTS (automatic)
  8 8 8  0 0 0
"""
with open(f"{PREFIX}_scf.in", "w") as f:
    f.write(scf_input)

print("[1/4] Running SCF...")
r = subprocess.run(["mpirun", "-np", str(NPROC), "pw.x", "-in", f"{PREFIX}_scf.in"],
                    capture_output=True, text=True, timeout=600)
with open(f"{PREFIX}_scf.out", "w") as f:
    f.write(r.stdout)
assert "convergence has been achieved" in r.stdout, "SCF did not converge!"
print("      SCF converged.")

# Extract Fermi energy
e_fermi = 0.0
for line in r.stdout.split("\n"):
    if "the Fermi energy is" in line:
        m = re.search(r"is\s+([-\d.]+)", line)
        if m: e_fermi = float(m.group(1))
    if "highest occupied" in line:
        m = re.search(r":\s+([-\d.]+)", line)
        if m: e_fermi = float(m.group(1))
    if "highest occupied, lowest unoccupied" in line:
        m = re.search(r":\s+([-\d.]+)\s+([-\d.]+)", line)
        if m: e_fermi = (float(m.group(1)) + float(m.group(2))) / 2
print(f"      Fermi energy: {e_fermi:.4f} eV")

# ── Step 2: Bands ──────────────────────────────────────────────────
bands_input = f"""&CONTROL
    calculation = 'bands'
    prefix      = '{PREFIX}'
    outdir      = '{OUTDIR}'
    pseudo_dir  = '{PSEUDO_DIR}'
    verbosity   = 'high'
/
&SYSTEM
    ibrav       = 0
    nat         = {nat}
    ntyp        = {ntyp}
    ecutwfc     = {ECUTWFC}
    ecutrho     = {ECUTRHO}
    nbnd        = {NBND}
/
&ELECTRONS
    conv_thr = 1.0d-8
/
CELL_PARAMETERS (angstrom)
{cell_card}

ATOMIC_SPECIES
{species_card}

ATOMIC_POSITIONS (crystal)
{pos_card}

{kpath_card}
"""
with open(f"{PREFIX}_bands.in", "w") as f:
    f.write(bands_input)

print("[2/4] Running bands calculation...")
r = subprocess.run(["mpirun", "-np", str(NPROC), "pw.x", "-in", f"{PREFIX}_bands.in"],
                    capture_output=True, text=True, timeout=600)
with open(f"{PREFIX}_bands.out", "w") as f:
    f.write(r.stdout)
print("      Bands calculation completed.")

# ── Step 3: bands.x ───────────────────────────────────────────────
pp_input = f"""&BANDS
    prefix  = '{PREFIX}'
    outdir  = '{OUTDIR}'
    filband = '{PREFIX}_bands.dat'
    lsym    = .true.
/
"""
with open(f"{PREFIX}_bands_pp.in", "w") as f:
    f.write(pp_input)

print("[3/4] Running bands.x...")
r = subprocess.run(["bands.x", "-in", f"{PREFIX}_bands_pp.in"],
                    capture_output=True, text=True, timeout=120)
with open(f"{PREFIX}_bands_pp.out", "w") as f:
    f.write(r.stdout)
print("      bands.x completed.")

# ── Step 4: Plot ───────────────────────────────────────────────────
print("[4/4] Plotting...")

# Parse .gnu file
with open(f"{PREFIX}_bands.dat.gnu", "r") as f:
    text = f.read()
blocks = text.strip().split("\n\n")
bands_data = []
k_dist = None
for block in blocks:
    lines = block.strip().split("\n")
    kvs, evs = [], []
    for line in lines:
        parts = line.split()
        if len(parts) >= 2:
            kvs.append(float(parts[0]))
            evs.append(float(parts[1]))
    if k_dist is None:
        k_dist = np.array(kvs)
    bands_data.append(np.array(evs))
eigenvalues = np.column_stack(bands_data)

# Parse high-symmetry positions from bands.x output
sym_positions = []
with open(f"{PREFIX}_bands_pp.out", "r") as f:
    for line in f:
        if "high-symmetry point" in line:
            m = re.search(r"x coordinate\s+([\d.]+)", line)
            if m:
                sym_positions.append(float(m.group(1)))

# Format labels (replace GAMMA with symbol)
def format_label(label):
    if label.upper() in ("GAMMA", "G"):
        return r"$\Gamma$"
    return label

formatted_labels = [format_label(l) for l in labels_in_order]
# Handle |
for i, lab in enumerate(formatted_labels):
    if "|" in lab:
        parts = lab.split("|")
        formatted_labels[i] = format_label(parts[0]) + "|" + format_label(parts[1])

fig, ax = plt.subplots(figsize=(8, 6))

for i in range(eigenvalues.shape[1]):
    ax.plot(k_dist, eigenvalues[:, i] - e_fermi, color="steelblue", linewidth=1.5)

ax.axhline(0, color="red", linestyle="--", linewidth=0.8, label="$E_F$")
for xpos in sym_positions:
    ax.axvline(xpos, color="black", linewidth=0.5, alpha=0.5)

if len(formatted_labels) == len(sym_positions):
    ax.set_xticks(sym_positions)
    ax.set_xticklabels(formatted_labels, fontsize=13)

ax.set_xlim(k_dist[0], k_dist[-1])
ax.set_ylim(-12, 8)
ax.set_ylabel("$E - E_F$ (eV)", fontsize=14)
ax.set_title("Electronic Band Structure", fontsize=15)
ax.legend(fontsize=11)
ax.grid(axis="y", alpha=0.3)
plt.tight_layout()
plt.savefig("band_structure.png", dpi=200, bbox_inches="tight")
print(f"      Saved: band_structure.png")

# Band gap
vbm = np.max(eigenvalues[eigenvalues <= e_fermi])
cbm = np.min(eigenvalues[eigenvalues > e_fermi])
gap = cbm - vbm
print(f"\n=== Results ===")
print(f"Band gap: {gap:.4f} eV ({'indirect' if True else 'direct'})")
print(f"VBM: {vbm:.4f} eV, CBM: {cbm:.4f} eV")
```

---

## Key Parameters

| Parameter | Typical Value | Notes |
|---|---|---|
| `nbnd` | 1.5x--2x occupied bands | Must be large enough to capture conduction bands of interest |
| `ecutwfc` | Same as SCF | Must match the SCF calculation exactly |
| `K_POINTS` | `{crystal_b}` format | Use seekpath or pymatgen for correct high-symmetry path |
| `npts_per_segment` | 20--40 | More points = smoother bands, longer computation |
| `verbosity` | `'high'` | Needed to print eigenvalues in bands output |

## Interpreting Results

- **Metallic**: Bands cross the Fermi level (E=0 in the shifted plot).
- **Semiconductor/Insulator**: A gap separates occupied (below E_F) and unoccupied (above E_F) bands.
- **Direct gap**: VBM and CBM occur at the same k-point.
- **Indirect gap**: VBM and CBM occur at different k-points (e.g., Si has an indirect gap with VBM at Gamma and CBM near X).
- **PBE underestimates band gaps** by 30--50% compared to experiment. For better gaps, use hybrid functionals or GW (not covered here).
- **Band character**: Flat bands indicate localized states; dispersive bands indicate delocalized/itinerant states.

## Common Issues

| Problem | Solution |
|---|---|
| Bands are discontinuous/jagged | Increase `npts_per_segment`. Ensure k-path follows correct BZ for the space group. |
| bands.x crashes: "prefix not found" | Ensure `prefix` and `outdir` match between SCF and bands steps exactly. |
| No bands above Fermi level | Increase `nbnd` in the bands input to include more empty states. |
| Wrong k-path for crystal system | Use seekpath which automatically determines the correct standardized cell and path. |
| Bands shifted incorrectly | Double-check the Fermi energy. For insulators, use midgap or VBM as reference. |
| Plot labels don't match points | Verify the number of high-symmetry points from bands.x output matches your label list. |
| SCF charge density not found | The bands `calculation='bands'` reads charge density from `outdir/prefix.save/`. Do not delete it between steps. |
