# Wannier90 Workflow: QE + Wannier90

## When to Use

- You need maximally localized Wannier functions (MLWFs) from a DFT calculation.
- You want a tight-binding Hamiltonian in the Wannier basis (for transport, topological analysis, etc.).
- You need Wannier-interpolated band structures (smoother and cheaper than dense NSCF grids).
- You are preparing input for WannierTools (topological invariants) or EPW (electron-phonon coupling).
- You need to extract hopping parameters for a model Hamiltonian.

## Method Selection

| Approach | When |
|---|---|
| Isolated bands (no disentanglement) | All target bands are separated from other bands by a gap at every k-point (e.g., valence bands of an insulator) |
| Entangled bands (disentanglement) | Target bands overlap with other bands in some k-region (e.g., d-bands in a metal, conduction bands touching higher bands) |

## Prerequisites

- Relaxed crystal structure (use `electronic-structure/scf-relax` skill).
- Pseudopotential files in `./pseudo/` (SSSP or PSlibrary).
- Quantum ESPRESSO: `pw.x`, `pw2wannier90.x`.
- Wannier90: `wannier90.x` binary. If not available, see the "Alternative: No Wannier90 Binary" section at the end.
- Python: `numpy`, `matplotlib`, `pymatgen` for post-processing.

---

## Detailed Steps

The full workflow is:

```
SCF (pw.x)
  |
  v
NSCF on uniform k-grid, nosym (pw.x)
  |
  v
wannier90.x -pp  (pre-processing: generates .nnkp file)
  |
  v
pw2wannier90.x   (extracts overlaps .mmn and projections .amn from QE)
  |
  v
wannier90.x      (Wannierization: spread minimization, band interpolation)
```

### Example System: Silicon (FCC diamond)

The complete example below uses silicon. Adapt atomic species, pseudopotentials, lattice parameters, projections, and energy windows for your material.

### Step 1: SCF Calculation

```python
#!/usr/bin/env python3
"""
Step 1: SCF calculation to obtain the ground-state charge density.
"""
import os
import subprocess

# ── Configuration ───────────────────────────────────────────────────
PSEUDO_DIR = os.path.abspath("./pseudo")
OUTDIR = os.path.abspath("./tmp_wannier")
PREFIX = "si"
os.makedirs(OUTDIR, exist_ok=True)
os.makedirs(PSEUDO_DIR, exist_ok=True)

# Download pseudopotential if not present
pp_file = os.path.join(PSEUDO_DIR, "Si.pbe-n-rrkjus_psl.1.0.0.UPF")
if not os.path.exists(pp_file):
    subprocess.run([
        "wget", "-q", "-O", pp_file,
        "https://pseudopotentials.quantum-espresso.org/upf_files/Si.pbe-n-rrkjus_psl.1.0.0.UPF"
    ], check=True)
    print(f"Downloaded pseudopotential to {pp_file}")

scf_input = f"""&CONTROL
    calculation = 'scf'
    prefix      = '{PREFIX}'
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
    degauss     = 0.005
/

&ELECTRONS
    conv_thr = 1.0d-10
    mixing_beta = 0.7
/

ATOMIC_SPECIES
  Si  28.0855  Si.pbe-n-rrkjus_psl.1.0.0.UPF

ATOMIC_POSITIONS (crystal)
  Si  0.00  0.00  0.00
  Si  0.25  0.25  0.25

K_POINTS (automatic)
  8 8 8  0 0 0
"""

with open(f"{PREFIX}_scf.in", "w") as f:
    f.write(scf_input)

print("Running SCF...")
result = subprocess.run(
    ["mpirun", "--allow-run-as-root", "-np", "4", "pw.x", "-in", f"{PREFIX}_scf.in"],
    capture_output=True, text=True, timeout=600
)
with open(f"{PREFIX}_scf.out", "w") as f:
    f.write(result.stdout)

if "convergence has been achieved" in result.stdout:
    print("SCF converged successfully.")
else:
    print("WARNING: SCF did not converge. Check output file.")
    print(result.stderr[-300:] if result.stderr else "")
```

### Step 2: NSCF on Uniform K-Grid (No Symmetry)

Wannier90 requires a uniform Monkhorst-Pack grid with all k-points explicitly listed (no symmetry reduction). The key flags are `nosym = .true.` and `noinv = .true.` in `&SYSTEM`, and `calculation = 'nscf'`.

```python
#!/usr/bin/env python3
"""
Step 2: NSCF calculation on a uniform k-grid without symmetry.
Wannier90 requires the full (unreduced) k-point grid.
"""
import os
import subprocess
import numpy as np

PSEUDO_DIR = os.path.abspath("./pseudo")
OUTDIR = os.path.abspath("./tmp_wannier")
PREFIX = "si"

# ── Generate explicit k-point grid ─────────────────────────────────
# Wannier90 needs the full unreduced grid. We generate it explicitly.
nk1, nk2, nk3 = 4, 4, 4  # k-grid for Wannier90 (can be different from SCF)
# For production, use at least 6x6x6 or 8x8x8.

kpoints = []
for i in range(nk1):
    for j in range(nk2):
        for k in range(nk3):
            kx = i / nk1
            ky = j / nk2
            kz = k / nk3
            kpoints.append((kx, ky, kz))

nktot = len(kpoints)
weight = 1.0 / nktot

kpoints_card = f"K_POINTS (crystal)\n{nktot}\n"
for kx, ky, kz in kpoints:
    kpoints_card += f"  {kx:.10f}  {ky:.10f}  {kz:.10f}  {weight:.10f}\n"

# Number of bands: must include enough empty states for disentanglement
# Si has 8 valence electrons -> 4 occupied bands. We want 8 Wannier functions
# (sp3 on each Si), so include at least 8+ bands.
NBND = 12

nscf_input = f"""&CONTROL
    calculation = 'nscf'
    prefix      = '{PREFIX}'
    outdir      = '{OUTDIR}'
    pseudo_dir  = '{PSEUDO_DIR}'
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
    degauss     = 0.005
    nbnd        = {NBND}
    nosym       = .true.
    noinv       = .true.
/

&ELECTRONS
    conv_thr = 1.0d-10
/

ATOMIC_SPECIES
  Si  28.0855  Si.pbe-n-rrkjus_psl.1.0.0.UPF

ATOMIC_POSITIONS (crystal)
  Si  0.00  0.00  0.00
  Si  0.25  0.25  0.25

{kpoints_card}
"""

with open(f"{PREFIX}_nscf.in", "w") as f:
    f.write(nscf_input)

print(f"Running NSCF with {nktot} k-points, {NBND} bands...")
result = subprocess.run(
    ["mpirun", "--allow-run-as-root", "-np", "4", "pw.x", "-in", f"{PREFIX}_nscf.in"],
    capture_output=True, text=True, timeout=1200
)
with open(f"{PREFIX}_nscf.out", "w") as f:
    f.write(result.stdout)

if result.returncode == 0:
    print("NSCF completed successfully.")
else:
    print("ERROR in NSCF calculation!")
    print(result.stderr[-500:] if result.stderr else "")
```

### Step 3: Prepare the Wannier90 Input File (`.win`)

The `.win` file controls all aspects of the Wannierization: number of Wannier functions, initial projections, disentanglement windows, and band interpolation k-path.

```python
#!/usr/bin/env python3
"""
Step 3: Generate the wannier90.win input file.
This is the master control file for the Wannier90 calculation.
"""
import numpy as np

PREFIX = "si"

# ── K-grid (must match NSCF) ───────────────────────────────────────
nk1, nk2, nk3 = 4, 4, 4

# ── Wannierization parameters ──────────────────────────────────────
num_bands = 12    # total bands from NSCF (nbnd)
num_wann = 8      # number of Wannier functions to construct

# For silicon sp3 hybridization: project onto sp3 orbitals on each Si atom.
# This gives 4 WFs per Si atom x 2 atoms = 8 WFs.
# Alternative: project onto s and p on each Si: also 8 WFs (1s + 3p per atom).

# ── Energy windows for disentanglement ─────────────────────────────
# dis_froz_min/max: frozen window -- states inside are included exactly
# dis_win_min/max: outer window -- states inside are candidates for disentanglement
# For silicon: valence bands (4) are isolated; conduction bands are entangled.
# If extracting only the 4 valence bands (num_wann=4), no disentanglement needed.
# For 8 WFs (valence + low conduction), we need disentanglement:
dis_win_min = -100.0   # eV, below all bands (include everything from bottom)
dis_win_max = 17.0     # eV, outer window upper bound
dis_froz_min = -100.0  # eV, frozen window lower bound
dis_froz_max = 6.5     # eV, frozen window upper bound (below entanglement region)
# States in [dis_froz_min, dis_froz_max] are kept exactly.
# States in [dis_win_min, dis_win_max] but outside frozen window are disentangled.

# ── Band structure k-path for interpolation ────────────────────────
# Silicon FCC Brillouin zone high-symmetry points (crystal coordinates):
kpath_points = {
    "G":     (0.000, 0.000, 0.000),
    "X":     (0.500, 0.000, 0.500),
    "W":     (0.500, 0.250, 0.750),
    "K":     (0.375, 0.375, 0.750),
    "L":     (0.500, 0.500, 0.500),
    "U":     (0.625, 0.250, 0.625),
}

# Path segments: G-X-W-K-G-L-U-W-L-K|U-X
kpath_segments = [
    ("G", "X", 50),
    ("X", "W", 30),
    ("W", "K", 20),
    ("K", "G", 50),
    ("G", "L", 50),
    ("L", "U", 30),
    ("U", "W", 20),
    ("W", "L", 30),
    ("L", "K", 30),
]

# ── Build the .win file ────────────────────────────────────────────
win_content = f"""\
! Wannier90 input file for Silicon
! Generated by Python script

num_bands = {num_bands}
num_wann  = {num_wann}

! ── Disentanglement ──────────────────────────────────────────────
! Required when num_bands > num_wann (entangled bands)
dis_win_min  = {dis_win_min}
dis_win_max  = {dis_win_max}
dis_froz_min = {dis_froz_min}
dis_froz_max = {dis_froz_max}
dis_num_iter = 200
dis_conv_tol = 1.0e-10

! ── Spread minimization ─────────────────────────────────────────
num_iter     = 200
conv_tol     = 1.0e-10
conv_window  = 5

! ── Initial projections ─────────────────────────────────────────
! Option A: sp3 hybrids centered on bond midpoints
! Option B: atomic s and p orbitals on each Si (simpler, often works well)
! We use Option B here.

begin projections
Si : s; p
end projections

! ── Band structure plot ──────────────────────────────────────────
bands_plot      = .true.
bands_num_points = 50

begin kpoint_path
"""

for start_label, end_label, npts in kpath_segments:
    sx, sy, sz = kpath_points[start_label]
    ex, ey, ez = kpath_points[end_label]
    win_content += f"{start_label}  {sx:.6f} {sy:.6f} {sz:.6f}   {end_label}  {ex:.6f} {ey:.6f} {ez:.6f}\n"

win_content += """end kpoint_path

! ── Write Hamiltonian ────────────────────────────────────────────
write_hr       = .true.
write_xyz      = .true.

! ── Unit cell ────────────────────────────────────────────────────
! Silicon FCC conventional -> primitive cell
! a = 5.431 Angstrom, celldm(1)=10.26 bohr -> a/2 in each pair of Cartesian dirs

begin unit_cell_cart
bohr
-5.13  0.00  5.13
 0.00  5.13  5.13
-5.13  5.13  0.00
end unit_cell_cart

begin atoms_frac
Si  0.00  0.00  0.00
Si  0.25  0.25  0.25
end atoms_frac

! ── K-point grid ─────────────────────────────────────────────────
mp_grid = {nk1} {nk2} {nk3}

begin kpoints
"""

# Write explicit k-points (same as NSCF)
for i in range(nk1):
    for j in range(nk2):
        for k in range(nk3):
            kx = i / nk1
            ky = j / nk2
            kz = k / nk3
            win_content += f"  {kx:.10f}  {ky:.10f}  {kz:.10f}\n"

win_content += "end kpoints\n"

with open(f"{PREFIX}.win", "w") as f:
    f.write(win_content)

print(f"Wannier90 input file written: {PREFIX}.win")
print(f"  num_bands = {num_bands}")
print(f"  num_wann  = {num_wann}")
print(f"  k-grid    = {nk1}x{nk2}x{nk3}")
print(f"  frozen window: [{dis_froz_min}, {dis_froz_max}] eV")
print(f"  outer window:  [{dis_win_min}, {dis_win_max}] eV")
```

### Step 4: Run Wannier90 Pre-Processing

```python
#!/usr/bin/env python3
"""
Step 4: Run wannier90.x -pp to generate the .nnkp file.
The .nnkp file tells pw2wannier90.x which overlaps to compute.
"""
import subprocess

PREFIX = "si"

print("Running wannier90.x -pp (pre-processing)...")
result = subprocess.run(
    ["wannier90.x", "-pp", PREFIX],
    capture_output=True, text=True, timeout=120
)

print(result.stdout)
if result.returncode != 0:
    print("ERROR in wannier90.x -pp!")
    print(result.stderr)
else:
    print(f"Pre-processing complete. Generated {PREFIX}.nnkp")
    print("This file is needed by pw2wannier90.x.")
```

### Step 5: Run pw2wannier90.x (QE -> Wannier90 Interface)

```python
#!/usr/bin/env python3
"""
Step 5: Run pw2wannier90.x to compute overlap matrices (.mmn)
and projection matrices (.amn) from QE wavefunctions.
"""
import os
import subprocess

OUTDIR = os.path.abspath("./tmp_wannier")
PREFIX = "si"

pw2wan_input = f"""&inputpp
    outdir    = '{OUTDIR}'
    prefix    = '{PREFIX}'
    seedname  = '{PREFIX}'
    write_mmn = .true.
    write_amn = .true.
    write_unk = .false.
/
"""
# write_unk = .true. generates real-space wavefunctions for plotting WFs (large files).
# Set to .false. unless you need to visualize Wannier functions in real space.

with open(f"{PREFIX}_pw2wan.in", "w") as f:
    f.write(pw2wan_input)

print("Running pw2wannier90.x...")
result = subprocess.run(
    ["mpirun", "--allow-run-as-root", "-np", "4", "pw2wannier90.x", "-in", f"{PREFIX}_pw2wan.in"],
    capture_output=True, text=True, timeout=600
)
with open(f"{PREFIX}_pw2wan.out", "w") as f:
    f.write(result.stdout)

if result.returncode == 0:
    print("pw2wannier90.x completed successfully.")
    print(f"Generated files: {PREFIX}.mmn, {PREFIX}.amn, {PREFIX}.eig")
else:
    print("ERROR in pw2wannier90.x!")
    print(result.stderr[-500:] if result.stderr else "")
```

### Step 6: Run Wannier90 (Full Wannierization)

```python
#!/usr/bin/env python3
"""
Step 6: Run wannier90.x to perform:
  - Disentanglement (if entangled bands)
  - Spread minimization (maximal localization)
  - Band interpolation
  - Tight-binding Hamiltonian output
"""
import subprocess

PREFIX = "si"

print("Running wannier90.x (full Wannierization)...")
result = subprocess.run(
    ["wannier90.x", PREFIX],
    capture_output=True, text=True, timeout=600
)
with open(f"{PREFIX}_wannier90.out", "w") as f:
    f.write(result.stdout)

if result.returncode == 0:
    print("Wannier90 completed successfully.")
    print(f"\nKey output files:")
    print(f"  {PREFIX}.wout          -- main output (spreads, centers, convergence)")
    print(f"  {PREFIX}_band.dat      -- interpolated band structure data")
    print(f"  {PREFIX}_band.gnu      -- gnuplot-formatted band data")
    print(f"  {PREFIX}_band.labelinfo.dat -- k-path label positions")
    print(f"  {PREFIX}_hr.dat        -- tight-binding Hamiltonian in WF basis")
    print(f"  {PREFIX}_centres.xyz   -- Wannier function centers")
else:
    print("ERROR in wannier90.x!")
    print(result.stderr[-500:] if result.stderr else "")
    print(result.stdout[-1000:])
```

### Step 7: Post-Processing and Plotting

#### 7a: Check Spread Convergence

```python
#!/usr/bin/env python3
"""
Parse the wannier90 .wout file to check spread convergence
and extract Wannier function centers and spreads.
"""
import re
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

PREFIX = "si"

# ── Parse .wout file ────────────────────────────────────────────────
with open(f"{PREFIX}.wout", "r") as f:
    wout_text = f.read()

# Extract spread at each iteration
# Lines look like:
#   <-- SPREAD  1   23.456789   0.123456   0.000012  ...
iterations = []
omega_total = []
omega_invariant = []
omega_tilde = []

for line in wout_text.split("\n"):
    # During spread minimization, lines contain spread info
    m = re.match(r"\s*(\d+)\s+([\d.E+-]+)\s+([\d.E+-]+)\s+([\d.E+-]+)\s+([\d.E+-]+)", line)
    if m and "CONV" not in line and "Cycle" not in line:
        # Try the SPREAD pattern
        pass

    # Better: look for the O_D + O_OD = O_total lines
    if "O_D=" in line and "O_OD=" in line:
        m = re.search(r"O_D=\s*([\d.E+-]+)\s+O_OD=\s*([\d.E+-]+)\s+O_TOT=\s*([\d.E+-]+)", line)
        if m:
            omega_invariant.append(float(m.group(1)))
            omega_tilde.append(float(m.group(2)))
            omega_total.append(float(m.group(3)))
            iterations.append(len(omega_total))

    # Also match spread output like: " SPRD: ..."
    if "<-- SPRD" in line:
        m = re.search(r"<-- SPRD\s+(\d+)\s+([\d.E+-]+)\s+([\d.E+-]+)\s+([\d.E+-]+)", line)
        if m and len(omega_total) == 0:
            # Fallback pattern
            pass

# If no data from above, try alternative parsing
if len(omega_total) == 0:
    for line in wout_text.split("\n"):
        # Pattern: "     1     0.123E+02  0.1234E-03  ..."
        if "CONV" not in line:
            m = re.match(r"\s+(\d+)\s+([\d.]+(?:E[+-]?\d+)?)\s+([\d.]+(?:E[+-]?\d+)?)\s+([\d.]+(?:E[+-]?\d+)?)", line)
            if m:
                it = int(m.group(1))
                if 1 <= it <= 5000:
                    iterations.append(it)
                    omega_total.append(float(m.group(2)))

print(f"Parsed {len(omega_total)} spread minimization iterations")

# ── Extract final WF centers and spreads ────────────────────────────
# Look for the "Final State" block
wf_centers = []
wf_spreads = []
in_final = False
for line in wout_text.split("\n"):
    if "Final State" in line:
        in_final = True
        continue
    if in_final:
        # Lines like: WF centre and target_spread:   1  (  -1.234,   2.345,   0.567 )     3.456
        m = re.search(r"WF\s+centre.*?(\d+)\s+\(\s*([-\d.]+),\s*([-\d.]+),\s*([-\d.]+)\s*\)\s+([\d.]+)", line)
        if m:
            idx = int(m.group(1))
            cx, cy, cz = float(m.group(2)), float(m.group(3)), float(m.group(4))
            spread = float(m.group(5))
            wf_centers.append([cx, cy, cz])
            wf_spreads.append(spread)
        elif line.strip() == "" or "Sum" in line or "---" in line:
            if len(wf_spreads) > 0:
                in_final = False

if len(wf_centers) > 0:
    print(f"\n=== Wannier Function Centers and Spreads ===")
    for i, (center, spread) in enumerate(zip(wf_centers, wf_spreads)):
        print(f"  WF {i+1}: center = ({center[0]:8.4f}, {center[1]:8.4f}, {center[2]:8.4f}) Ang,  "
              f"spread = {spread:.4f} Ang^2")
    print(f"\n  Total spread: {sum(wf_spreads):.4f} Ang^2")
    print(f"  Average spread: {np.mean(wf_spreads):.4f} Ang^2")

# ── Plot spread convergence ─────────────────────────────────────────
if len(omega_total) > 1:
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(iterations, omega_total, 'b-', linewidth=1.5, label=r"$\Omega_{total}$")
    if len(omega_invariant) == len(iterations):
        ax.plot(iterations, omega_invariant, 'r--', linewidth=1.0, label=r"$\Omega_D$ (gauge invariant)")
        ax.plot(iterations, omega_tilde, 'g--', linewidth=1.0, label=r"$\widetilde{\Omega}$ (minimized)")
    ax.set_xlabel("Iteration", fontsize=13)
    ax.set_ylabel(r"Spread ($\AA^2$)", fontsize=13)
    ax.set_title("Wannier Spread Minimization Convergence", fontsize=14)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("wannier_spread_convergence.png", dpi=200)
    print("\nSaved: wannier_spread_convergence.png")
```

#### 7b: Plot Wannier-Interpolated Bands Overlaid on DFT Bands

```python
#!/usr/bin/env python3
"""
Plot Wannier90-interpolated band structure overlaid on DFT bands.
Reads wannier90 _band.dat and optionally QE bands.x output.
"""
import numpy as np
import re
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

PREFIX = "si"

# ── Parse Wannier90 band data ───────────────────────────────────────
def parse_wannier_bands(band_dat_file):
    """
    Parse PREFIX_band.dat from Wannier90.
    Format: k_distance  energy (eV)
    Bands are separated by blank lines.
    """
    with open(band_dat_file, "r") as f:
        text = f.read()

    blocks = text.strip().split("\n\n")
    bands = []
    k_dist = None

    for block in blocks:
        lines = block.strip().split("\n")
        kvs, evs = [], []
        for line in lines:
            parts = line.split()
            if len(parts) >= 2:
                kvs.append(float(parts[0]))
                evs.append(float(parts[1]))
        if len(kvs) > 0:
            if k_dist is None:
                k_dist = np.array(kvs)
            bands.append(np.array(evs))

    if len(bands) == 0:
        return None, None

    eigenvalues = np.column_stack(bands)  # shape: (n_kpoints, n_bands)
    return k_dist, eigenvalues


def parse_wannier_labels(labelinfo_file):
    """
    Parse PREFIX_band.labelinfo.dat for high-symmetry point positions.
    Format: label  k_index  k_distance
    """
    labels = []
    positions = []
    if not os.path.exists(labelinfo_file):
        return labels, positions

    with open(labelinfo_file, "r") as f:
        for line in f:
            parts = line.split()
            if len(parts) >= 3:
                label = parts[0]
                # k_distance is the third column
                try:
                    k_pos = float(parts[2])
                    labels.append(label)
                    positions.append(k_pos)
                except ValueError:
                    pass
    return labels, positions


# ── Parse DFT bands (from QE bands.x .gnu file) ────────────────────
def parse_dft_bands_gnu(gnu_file):
    """Parse QE bands.x .gnu output."""
    if not os.path.exists(gnu_file):
        return None, None

    with open(gnu_file, "r") as f:
        text = f.read()

    blocks = text.strip().split("\n\n")
    bands = []
    k_dist = None

    for block in blocks:
        lines = block.strip().split("\n")
        kvs, evs = [], []
        for line in lines:
            parts = line.split()
            if len(parts) >= 2:
                kvs.append(float(parts[0]))
                evs.append(float(parts[1]))
        if len(kvs) > 0:
            if k_dist is None:
                k_dist = np.array(kvs)
            bands.append(np.array(evs))

    if len(bands) == 0:
        return None, None
    return k_dist, np.column_stack(bands)


# ── Extract Fermi energy from SCF output ────────────────────────────
def get_fermi_energy(scf_output_file):
    """Extract Fermi energy from QE SCF output."""
    e_fermi = 0.0
    if not os.path.exists(scf_output_file):
        return e_fermi
    with open(scf_output_file, "r") as f:
        for line in f:
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
    return e_fermi


# ── Main plotting ──────────────────────────────────────────────────
e_fermi = get_fermi_energy(f"{PREFIX}_scf.out")
print(f"Fermi energy: {e_fermi:.4f} eV")

# Wannier-interpolated bands
w_kdist, w_bands = parse_wannier_bands(f"{PREFIX}_band.dat")
if w_bands is not None:
    print(f"Wannier bands: {w_bands.shape[0]} k-points, {w_bands.shape[1]} bands")
else:
    print("ERROR: Could not parse Wannier band data!")

# DFT bands (optional overlay)
dft_kdist, dft_bands = parse_dft_bands_gnu(f"{PREFIX}_bands.dat.gnu")
has_dft = dft_bands is not None

# Labels
labels, label_positions = parse_wannier_labels(f"{PREFIX}_band.labelinfo.dat")

# Format labels
def format_label(label):
    if label.upper() in ("GAMMA", "G"):
        return r"$\Gamma$"
    return label

formatted_labels = [format_label(l) for l in labels]

# ── Create figure ──────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(9, 6))

# Plot Wannier-interpolated bands
if w_bands is not None:
    for i in range(w_bands.shape[1]):
        label_str = "Wannier90" if i == 0 else None
        ax.plot(w_kdist, w_bands[:, i] - e_fermi,
                color="steelblue", linewidth=1.5, label=label_str)

# Overlay DFT bands if available
if has_dft:
    for i in range(dft_bands.shape[1]):
        label_str = "DFT (QE)" if i == 0 else None
        ax.plot(dft_kdist, dft_bands[:, i] - e_fermi,
                color="red", linewidth=0.8, linestyle="--",
                alpha=0.7, label=label_str)

# Fermi level
ax.axhline(0, color="black", linestyle=":", linewidth=0.8, alpha=0.5)

# High-symmetry point lines
for kpos in label_positions:
    ax.axvline(kpos, color="gray", linewidth=0.5, alpha=0.5)

# Labels
if len(formatted_labels) == len(label_positions):
    ax.set_xticks(label_positions)
    ax.set_xticklabels(formatted_labels, fontsize=13)

if w_kdist is not None:
    ax.set_xlim(w_kdist[0], w_kdist[-1])
ax.set_ylim(-12, 8)
ax.set_ylabel("$E - E_F$ (eV)", fontsize=14)
ax.set_title("Wannier-Interpolated Band Structure", fontsize=15)
ax.legend(fontsize=11, loc="upper right")
ax.grid(axis="y", alpha=0.3)

plt.tight_layout()
plt.savefig("wannier_bands.png", dpi=200, bbox_inches="tight")
print("Saved: wannier_bands.png")
```

#### 7c: Read the Tight-Binding Hamiltonian

```python
#!/usr/bin/env python3
"""
Read the Wannier90 tight-binding Hamiltonian from PREFIX_hr.dat.
This Hamiltonian can be used for transport calculations, topological analysis, etc.
"""
import numpy as np

PREFIX = "si"

def read_hr_dat(filename):
    """
    Read the _hr.dat file from Wannier90.
    Returns:
        num_wann: number of Wannier functions
        nrpts: number of R-vectors
        degeneracy: degeneracy of each R-vector
        rvecs: array of R-vectors (nrpts, 3)
        hr: Hamiltonian dict: {(R1,R2,R3): complex array (num_wann, num_wann)}
    """
    with open(filename, "r") as f:
        lines = f.readlines()

    # Line 1: header/comment
    # Line 2: num_wann
    num_wann = int(lines[1].strip())
    # Line 3: nrpts
    nrpts = int(lines[2].strip())

    # Next ceil(nrpts/15) lines: degeneracy weights
    ndeg_lines = (nrpts + 14) // 15
    degeneracy = []
    for i in range(3, 3 + ndeg_lines):
        degeneracy.extend([int(x) for x in lines[i].split()])
    degeneracy = np.array(degeneracy)

    # Remaining lines: R1 R2 R3  m  n  Re(H)  Im(H)
    hr = {}
    start_line = 3 + ndeg_lines
    for line in lines[start_line:]:
        parts = line.split()
        if len(parts) < 7:
            continue
        r1, r2, r3 = int(parts[0]), int(parts[1]), int(parts[2])
        m, n = int(parts[3]) - 1, int(parts[4]) - 1  # 0-indexed
        re_h, im_h = float(parts[5]), float(parts[6])

        key = (r1, r2, r3)
        if key not in hr:
            hr[key] = np.zeros((num_wann, num_wann), dtype=complex)
        hr[key][m, n] = re_h + 1j * im_h

    rvecs = np.array(list(hr.keys()))

    return num_wann, nrpts, degeneracy, rvecs, hr


num_wann, nrpts, degeneracy, rvecs, hr = read_hr_dat(f"{PREFIX}_hr.dat")

print(f"Number of Wannier functions: {num_wann}")
print(f"Number of R-vectors: {nrpts}")
print(f"\nOn-site Hamiltonian H(R=0) diagonal (eV):")
h_onsite = hr.get((0, 0, 0), np.zeros((num_wann, num_wann)))
for i in range(num_wann):
    print(f"  WF {i+1}: {h_onsite[i, i].real:.6f} eV")

# Nearest-neighbor hoppings
print(f"\nSample hopping parameters (first 5 R-vectors):")
for rv in sorted(hr.keys())[:5]:
    h_r = hr[rv]
    max_hop = np.max(np.abs(h_r))
    print(f"  R = {rv}: max|H| = {max_hop:.6f} eV")

# ── Wannier-interpolate H(k) at an arbitrary k-point ───────────────
def interpolate_hk(k, hr, degeneracy_dict, num_wann):
    """
    Fourier-interpolate H(k) = sum_R H(R) * exp(i k.R) / deg(R)
    k: fractional coordinates (3,)
    Returns: H(k) complex matrix (num_wann, num_wann)
    """
    hk = np.zeros((num_wann, num_wann), dtype=complex)
    for (r1, r2, r3), h_r in hr.items():
        r_vec = np.array([r1, r2, r3])
        phase = np.exp(2j * np.pi * np.dot(k, r_vec))
        deg = degeneracy_dict.get((r1, r2, r3), 1)
        hk += h_r * phase / deg
    return hk

# Build degeneracy dict
deg_dict = {}
for i, rv in enumerate(sorted(hr.keys())):
    if i < len(degeneracy):
        deg_dict[rv] = degeneracy[i]
    else:
        deg_dict[rv] = 1

# Example: eigenvalues at Gamma
k_gamma = np.array([0.0, 0.0, 0.0])
hk_gamma = interpolate_hk(k_gamma, hr, deg_dict, num_wann)
eigs_gamma = np.sort(np.linalg.eigvalsh(hk_gamma).real)
print(f"\nEigenvalues at Gamma from Wannier interpolation (eV):")
for i, e in enumerate(eigs_gamma):
    print(f"  Band {i+1}: {e:.6f} eV")
```

### Complete Single-Script Workflow

For convenience, here is a single script that runs the entire pipeline from SCF through plotting:

```python
#!/usr/bin/env python3
"""
Complete Wannier90 workflow for Silicon.
Runs: SCF -> NSCF -> wannier90 -pp -> pw2wannier90 -> wannier90 -> plot
"""
import os
import subprocess
import re
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ══════════════════════════════════════════════════════════════════════
# Configuration
# ══════════════════════════════════════════════════════════════════════
PSEUDO_DIR = os.path.abspath("./pseudo")
OUTDIR = os.path.abspath("./tmp_wannier")
PREFIX = "si"
NPROC = 4
ECUTWFC = 50.0
ECUTRHO = 400.0
NK1, NK2, NK3 = 4, 4, 4    # Wannier k-grid (use 6-8 for production)
NUM_BANDS = 12              # total bands in NSCF
NUM_WANN = 8                # Wannier functions (sp3 on 2 Si atoms)

os.makedirs(OUTDIR, exist_ok=True)
os.makedirs(PSEUDO_DIR, exist_ok=True)

# Download pseudopotential
pp_file = os.path.join(PSEUDO_DIR, "Si.pbe-n-rrkjus_psl.1.0.0.UPF")
if not os.path.exists(pp_file):
    subprocess.run([
        "wget", "-q", "-O", pp_file,
        "https://pseudopotentials.quantum-espresso.org/upf_files/Si.pbe-n-rrkjus_psl.1.0.0.UPF"
    ], check=True)

def run_cmd(cmd, input_file=None, output_file=None, timeout=600):
    """Run a command, optionally saving stdout to a file."""
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    if output_file:
        with open(output_file, "w") as f:
            f.write(r.stdout)
    return r

# ══════════════════════════════════════════════════════════════════════
# Step 1: SCF
# ══════════════════════════════════════════════════════════════════════
scf_input = f"""&CONTROL
    calculation = 'scf', prefix = '{PREFIX}', outdir = '{OUTDIR}',
    pseudo_dir = '{PSEUDO_DIR}'
/
&SYSTEM
    ibrav = 2, celldm(1) = 10.26, nat = 2, ntyp = 1,
    ecutwfc = {ECUTWFC}, ecutrho = {ECUTRHO},
    occupations = 'smearing', smearing = 'cold', degauss = 0.005
/
&ELECTRONS
    conv_thr = 1.0d-10
/
ATOMIC_SPECIES
  Si 28.0855 Si.pbe-n-rrkjus_psl.1.0.0.UPF
ATOMIC_POSITIONS (crystal)
  Si 0.00 0.00 0.00
  Si 0.25 0.25 0.25
K_POINTS (automatic)
  8 8 8 0 0 0
"""
with open(f"{PREFIX}_scf.in", "w") as f:
    f.write(scf_input)

print("[1/6] Running SCF...")
r = run_cmd(["mpirun", "--allow-run-as-root", "-np", str(NPROC), "pw.x",
             "-in", f"{PREFIX}_scf.in"], output_file=f"{PREFIX}_scf.out")
assert "convergence has been achieved" in r.stdout, "SCF failed!"
print("      SCF converged.")

# Extract Fermi energy
e_fermi = 0.0
for line in r.stdout.split("\n"):
    if "the Fermi energy is" in line:
        m = re.search(r"is\s+([-\d.]+)", line)
        if m: e_fermi = float(m.group(1))
    if "highest occupied, lowest unoccupied" in line:
        m = re.search(r":\s+([-\d.]+)\s+([-\d.]+)", line)
        if m: e_fermi = (float(m.group(1)) + float(m.group(2))) / 2
    if "highest occupied" in line and "lowest" not in line:
        m = re.search(r":\s+([-\d.]+)", line)
        if m: e_fermi = float(m.group(1))
print(f"      Fermi energy: {e_fermi:.4f} eV")

# ══════════════════════════════════════════════════════════════════════
# Step 2: NSCF (uniform k-grid, no symmetry)
# ══════════════════════════════════════════════════════════════════════
kpoints = []
for i in range(NK1):
    for j in range(NK2):
        for k in range(NK3):
            kpoints.append((i/NK1, j/NK2, k/NK3))

nktot = len(kpoints)
w = 1.0 / nktot
kpts_card = f"K_POINTS (crystal)\n{nktot}\n"
for kx, ky, kz in kpoints:
    kpts_card += f"  {kx:.10f} {ky:.10f} {kz:.10f} {w:.10f}\n"

nscf_input = f"""&CONTROL
    calculation = 'nscf', prefix = '{PREFIX}', outdir = '{OUTDIR}',
    pseudo_dir = '{PSEUDO_DIR}'
/
&SYSTEM
    ibrav = 2, celldm(1) = 10.26, nat = 2, ntyp = 1,
    ecutwfc = {ECUTWFC}, ecutrho = {ECUTRHO},
    occupations = 'smearing', smearing = 'cold', degauss = 0.005,
    nbnd = {NUM_BANDS}, nosym = .true., noinv = .true.
/
&ELECTRONS
    conv_thr = 1.0d-10
/
ATOMIC_SPECIES
  Si 28.0855 Si.pbe-n-rrkjus_psl.1.0.0.UPF
ATOMIC_POSITIONS (crystal)
  Si 0.00 0.00 0.00
  Si 0.25 0.25 0.25
{kpts_card}
"""
with open(f"{PREFIX}_nscf.in", "w") as f:
    f.write(nscf_input)

print(f"[2/6] Running NSCF ({nktot} k-points, {NUM_BANDS} bands)...")
r = run_cmd(["mpirun", "--allow-run-as-root", "-np", str(NPROC), "pw.x",
             "-in", f"{PREFIX}_nscf.in"], output_file=f"{PREFIX}_nscf.out")
assert r.returncode == 0, "NSCF failed!"
print("      NSCF completed.")

# ══════════════════════════════════════════════════════════════════════
# Step 3: Generate wannier90.win and run -pp
# ══════════════════════════════════════════════════════════════════════
win_content = f"""\
num_bands = {NUM_BANDS}
num_wann  = {NUM_WANN}

dis_win_min  = -100.0
dis_win_max  = 17.0
dis_froz_min = -100.0
dis_froz_max = 6.5
dis_num_iter = 200
dis_conv_tol = 1.0e-10

num_iter     = 200
conv_tol     = 1.0e-10
conv_window  = 5

begin projections
Si : s; p
end projections

bands_plot       = .true.
bands_num_points = 50

begin kpoint_path
G  0.000 0.000 0.000   X  0.500 0.000 0.500
X  0.500 0.000 0.500   W  0.500 0.250 0.750
W  0.500 0.250 0.750   K  0.375 0.375 0.750
K  0.375 0.375 0.750   G  0.000 0.000 0.000
G  0.000 0.000 0.000   L  0.500 0.500 0.500
end kpoint_path

write_hr  = .true.
write_xyz = .true.

begin unit_cell_cart
bohr
-5.13  0.00  5.13
 0.00  5.13  5.13
-5.13  5.13  0.00
end unit_cell_cart

begin atoms_frac
Si 0.00 0.00 0.00
Si 0.25 0.25 0.25
end atoms_frac

mp_grid = {NK1} {NK2} {NK3}

begin kpoints
"""
for kx, ky, kz in kpoints:
    win_content += f"  {kx:.10f} {ky:.10f} {kz:.10f}\n"
win_content += "end kpoints\n"

with open(f"{PREFIX}.win", "w") as f:
    f.write(win_content)

print("[3/6] Running wannier90.x -pp...")
r = run_cmd(["wannier90.x", "-pp", PREFIX])
assert r.returncode == 0, f"wannier90 -pp failed!\n{r.stderr}"
print(f"      Generated {PREFIX}.nnkp")

# ══════════════════════════════════════════════════════════════════════
# Step 4: pw2wannier90.x
# ══════════════════════════════════════════════════════════════════════
pw2wan_input = f"""&inputpp
    outdir    = '{OUTDIR}'
    prefix    = '{PREFIX}'
    seedname  = '{PREFIX}'
    write_mmn = .true.
    write_amn = .true.
    write_unk = .false.
/
"""
with open(f"{PREFIX}_pw2wan.in", "w") as f:
    f.write(pw2wan_input)

print("[4/6] Running pw2wannier90.x...")
r = run_cmd(["mpirun", "--allow-run-as-root", "-np", str(NPROC),
             "pw2wannier90.x", "-in", f"{PREFIX}_pw2wan.in"],
            output_file=f"{PREFIX}_pw2wan.out")
assert r.returncode == 0, f"pw2wannier90 failed!\n{r.stderr}"
print(f"      Generated {PREFIX}.mmn, {PREFIX}.amn, {PREFIX}.eig")

# ══════════════════════════════════════════════════════════════════════
# Step 5: wannier90.x (full run)
# ══════════════════════════════════════════════════════════════════════
print("[5/6] Running wannier90.x (Wannierization)...")
r = run_cmd(["wannier90.x", PREFIX], output_file=f"{PREFIX}_wannier90.out")
assert r.returncode == 0, f"wannier90 failed!\n{r.stderr}"
print("      Wannierization complete.")

# ══════════════════════════════════════════════════════════════════════
# Step 6: Plot
# ══════════════════════════════════════════════════════════════════════
print("[6/6] Plotting Wannier-interpolated band structure...")

# Parse Wannier band data
with open(f"{PREFIX}_band.dat", "r") as f:
    text = f.read()
blocks = text.strip().split("\n\n")
bands_list = []
k_dist = None
for block in blocks:
    lines = block.strip().split("\n")
    kvs, evs = [], []
    for line in lines:
        parts = line.split()
        if len(parts) >= 2:
            kvs.append(float(parts[0]))
            evs.append(float(parts[1]))
    if len(kvs) > 0:
        if k_dist is None:
            k_dist = np.array(kvs)
        bands_list.append(np.array(evs))

w_bands = np.column_stack(bands_list)

# Parse label info
labels, label_pos = [], []
label_file = f"{PREFIX}_band.labelinfo.dat"
if os.path.exists(label_file):
    with open(label_file, "r") as f:
        for line in f:
            parts = line.split()
            if len(parts) >= 3:
                labels.append(parts[0])
                try:
                    label_pos.append(float(parts[2]))
                except ValueError:
                    pass

def fmt_label(l):
    return r"$\Gamma$" if l.upper() in ("GAMMA", "G") else l

fig, ax = plt.subplots(figsize=(9, 6))

for i in range(w_bands.shape[1]):
    lbl = "Wannier90" if i == 0 else None
    ax.plot(k_dist, w_bands[:, i] - e_fermi, color="steelblue", linewidth=1.5, label=lbl)

ax.axhline(0, color="black", linestyle=":", linewidth=0.8, alpha=0.5)
for kp in label_pos:
    ax.axvline(kp, color="gray", linewidth=0.5, alpha=0.5)
if len(labels) == len(label_pos):
    ax.set_xticks(label_pos)
    ax.set_xticklabels([fmt_label(l) for l in labels], fontsize=13)
ax.set_xlim(k_dist[0], k_dist[-1])
ax.set_ylim(-12, 8)
ax.set_ylabel("$E - E_F$ (eV)", fontsize=14)
ax.set_title("Si: Wannier-Interpolated Band Structure", fontsize=15)
ax.legend(fontsize=11)
ax.grid(axis="y", alpha=0.3)
plt.tight_layout()
plt.savefig("wannier_bands.png", dpi=200, bbox_inches="tight")
print("Saved: wannier_bands.png")
print("\nDone. All output files in current directory.")
```

### Alternative: No Wannier90 Binary Available

If `wannier90.x` is not installed in the container, you have several options:

**Option A: Install via pip**

```bash
pip install wannier90
# This installs Python bindings but NOT the standalone wannier90.x binary.
# The Python bindings provide wannier90.run() which calls the library directly.
```

```python
#!/usr/bin/env python3
"""
Use the wannier90 Python bindings if the binary is not available.
After: pip install wannier90
"""
try:
    import wannier90
    # The Python bindings call the Wannier90 library functions directly.
    # Usage depends on the specific Python wrapper version.
    # Typical usage:
    # wannier90.run(seedname="si")
    print("wannier90 Python bindings available.")
except ImportError:
    print("wannier90 Python bindings not available.")
    print("Try: pip install wannier90")
```

**Option B: Build from source**

```bash
# Download and compile wannier90
cd /tmp
wget https://github.com/wannier-developers/wannier90/archive/refs/tags/v3.1.0.tar.gz
tar xzf v3.1.0.tar.gz
cd wannier90-3.1.0
cp config/make.inc.gfort make.inc
# Edit make.inc if needed (set F90=gfortran, LIBS=-llapack -lblas)
make
# Copy binaries
cp wannier90.x postw90.x /usr/local/bin/
```

**Option C: Use only pw2wannier90.x outputs**

If you only need the overlap/projection matrices (`.mmn`, `.amn`, `.eig`) for another code (e.g., WannierTools, Z2Pack), you can run only the QE side (steps 1-5) and process the output files with Python:

```python
#!/usr/bin/env python3
"""
Process .amn, .mmn, .eig files directly with Python for
simple Wannier analysis without the wannier90.x binary.
This is a minimal implementation -- for production use the full Wannier90.
"""
import numpy as np

PREFIX = "si"

# ── Read .eig file (eigenvalues at each k-point) ───────────────────
def read_eig(filename):
    """Read PREFIX.eig: band_index k_index eigenvalue(eV)"""
    eig_data = {}
    with open(filename, "r") as f:
        for line in f:
            parts = line.split()
            if len(parts) == 3:
                ib = int(parts[0])
                ik = int(parts[1])
                ev = float(parts[2])
                if ik not in eig_data:
                    eig_data[ik] = {}
                eig_data[ik][ib] = ev
    return eig_data

eig_data = read_eig(f"{PREFIX}.eig")
nk = len(eig_data)
nb = len(eig_data[1])
print(f"Read {nk} k-points, {nb} bands from .eig file")

# Print eigenvalues at Gamma (k-point 1)
print("Eigenvalues at k=1 (Gamma):")
for ib in sorted(eig_data[1].keys()):
    print(f"  Band {ib}: {eig_data[1][ib]:.6f} eV")

# ── Read .amn file (projection matrix) ─────────────────────────────
def read_amn(filename):
    """Read PREFIX.amn: band_index k_index wf_index Re(A) Im(A)"""
    with open(filename, "r") as f:
        header = f.readline()  # comment
        parts = f.readline().split()
        nb, nk, nw = int(parts[0]), int(parts[1]), int(parts[2])

        amn = np.zeros((nk, nb, nw), dtype=complex)
        for line in f:
            parts = line.split()
            ib = int(parts[0]) - 1
            ik = int(parts[1]) - 1
            iw = int(parts[2]) - 1
            amn[ik, ib, iw] = float(parts[3]) + 1j * float(parts[4])

    return amn, nb, nk, nw

amn, nb, nk, nw = read_amn(f"{PREFIX}.amn")
print(f"\nProjection matrix A(k): shape = ({nk}, {nb}, {nw})")
print(f"  |A|^2 sum at k=0: {np.sum(np.abs(amn[0])**2):.6f}")
```

---

## Key Parameters

| Parameter | Typical Value | Notes |
|---|---|---|
| `num_wann` | Depends on chemistry | Must match number of target orbitals (e.g., 5 for d-bands, 4 for sp3 per atom) |
| `num_bands` | >= `num_wann` | Total bands in NSCF. If > `num_wann`, disentanglement is needed |
| `mp_grid` | 4x4x4 to 12x12x12 | Uniform k-grid. Must match NSCF grid. Denser = better interpolation |
| `dis_froz_max` | Just above target band manifold | Frozen window top: states below this are kept exactly |
| `dis_win_max` | Well above `dis_froz_max` | Outer window top: states above this are excluded |
| `num_iter` | 100-500 | Spread minimization iterations. Watch convergence. |
| `dis_num_iter` | 100-500 | Disentanglement iterations. Usually converges faster. |
| `projections` | Element : orbitals | Initial guess strongly affects convergence. Match expected chemistry. |
| `nosym`, `noinv` | `.true.` | MUST be set in NSCF. Wannier90 needs the full unreduced k-grid. |
| `conv_thr` (NSCF) | 1e-10 or tighter | Tighter than SCF to ensure accurate wavefunctions |

### Projection Guidelines

| Material | Recommended Projections | num_wann |
|---|---|---|
| Si (diamond) | `Si : s; p` | 8 (4 per atom) |
| GaAs | `Ga : s; p` and `As : s; p` | 8 |
| Fe (BCC) | `Fe : s; p; d` | 9 |
| Transition metal oxide | `TM : d` and `O : p` | 5 + 3*n_O |
| Topological insulator (Bi2Se3) | `Bi : p` and `Se : p` | 6 + 9 = 15 |

### Disentanglement Window Guidelines

```
Energy
  ^
  |  ....  bands above outer window (excluded)
  |  ----  dis_win_max (outer window top)
  |  ::::  disentanglement region (optimized)
  |  ----  dis_froz_max (frozen window top)
  |  ####  frozen region (kept exactly)
  |  ----  dis_froz_min (frozen window bottom, usually very negative)
  |  ----  dis_win_min (outer window bottom, usually very negative)
```

- Set `dis_froz_max` just above the highest band you want to reproduce exactly.
- Set `dis_win_max` high enough to include all bands that might hybridize with your target manifold.
- For isolated bands (e.g., d-bands separated by gaps), no disentanglement is needed. Set `num_bands = num_wann`.

## Interpreting Results

- **Spread convergence**: The total spread should decrease monotonically and plateau. If it oscillates or does not converge, try different projections or a finer k-grid.
- **Wannier function spreads**: Typical spreads are 1-5 Angstrom^2 for well-localized WFs. Very large spreads (>10 Ang^2) indicate poor Wannierization.
- **Band interpolation quality**: Wannier-interpolated bands should match DFT bands exactly within the frozen window. Deviations outside the frozen window are expected.
- **Hopping parameters**: On-site energies appear on the diagonal of H(R=0). Nearest-neighbor hoppings are in H(R=nearest lattice vector). These should match physical expectations.
- **Symmetry**: Well-converged WFs should respect the crystal symmetry. Check that symmetry-equivalent WFs have the same spread.

## Common Issues

| Problem | Solution |
|---|---|
| `wannier90 -pp` fails: "Error reading .win" | Check `.win` file syntax. Ensure `mp_grid` and `kpoints` block match exactly. |
| `pw2wannier90.x` fails: "prefix not found" | Ensure `prefix` and `outdir` match the NSCF calculation. The NSCF must have completed successfully. |
| `pw2wannier90.x` fails: "k-points mismatch" | The k-grid in `.win` must match the NSCF k-grid exactly (same order, same coordinates). Both must use `nosym=.true., noinv=.true.`. |
| Spread does not converge | Try different initial projections. Increase `num_iter`. Try a finer k-grid. Check that the disentanglement window is reasonable. |
| Bands look wrong outside frozen window | This is expected -- only the frozen window bands are guaranteed to match. Widen the frozen window if needed. |
| Very large spreads | Poor initial projections. Try projections closer to the expected Wannier function character. For metals, ensure the disentanglement window is appropriate. |
| `wannier90.x` not found | See "Alternative: No Wannier90 Binary" section. Build from source or use Python bindings. |
| NSCF k-points out of order | Generate k-points in the same nested-loop order as Wannier90 expects: `for i in nk1: for j in nk2: for k in nk3`. |
| Disentanglement does not converge | Widen the outer window. Increase `dis_num_iter`. Check that `num_wann` matches the physical band count. |
