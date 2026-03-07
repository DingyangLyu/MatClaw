# Dielectric Function Calculation

## When to Use

- You need the frequency-dependent dielectric function (real and imaginary parts) of a crystalline material.
- You want to determine the static dielectric constant (electronic contribution).
- You need to identify optical transitions and the optical gap from Im(epsilon).
- You want Born effective charges and ionic contributions to the static dielectric constant.
- You are computing inputs for subsequent optical property derivations (absorption, reflectivity, conductivity).

## Method Selection

| Criterion | ASE + MACE | QE DFT (epsilon.x) |
|---|---|---|
| Dielectric function | Not available | Full frequency-dependent epsilon(omega) via RPA |
| Static dielectric constant | Not available | Electronic part from epsilon.x; ionic part from ph.x |
| Born effective charges | Not available | Via ph.x (DFPT) |

**MACE cannot compute dielectric functions.** It is a force field with no electronic wavefunctions. Always use QE. MACE can pre-relax the structure before DFT.

## Prerequisites

- A relaxed crystal structure (use the `scf-relax` skill first, or relax with MACE).
- Pseudopotential files in `./pseudo/` (norm-conserving recommended for epsilon.x).
- QE executables: `pw.x`, `epsilon.x`, optionally `ph.x` (for Born effective charges).
- Python packages: `numpy`, `matplotlib`.

**Important**: QE `epsilon.x` works best with **norm-conserving** pseudopotentials. Ultrasoft or PAW pseudopotentials require reconstruction of the all-electron wavefunction, which epsilon.x does not perform. Use ONCV or similar norm-conserving pseudopotentials for reliable optical properties.

---

## Detailed Steps

### Step 0: Download Pseudopotentials

```python
#!/usr/bin/env python3
"""
Download norm-conserving (ONCV) pseudopotentials for optical calculations.
"""
import os
import subprocess

PSEUDO_DIR = os.path.abspath("./pseudo")
os.makedirs(PSEUDO_DIR, exist_ok=True)

# Example: Silicon with norm-conserving ONCV pseudopotential
# From the PseudoDojo or SG15 collection
pseudo_files = {
    "Si": "Si.pbe-n-kjpaw_psl.1.0.0.UPF",  # fallback if NC not available
}

# Try SG15 ONCV first (recommended for epsilon.x)
sg15_base = "http://www.quantum-simulation.org/potentials/sg15_oncv"
oncv_files = {
    "Si": "Si_ONCV_PBE-1.2.upf",
}

for elem, fname in oncv_files.items():
    fpath = os.path.join(PSEUDO_DIR, fname)
    if not os.path.exists(fpath):
        url = f"{sg15_base}/{fname}"
        print(f"Downloading {fname}...")
        result = subprocess.run(["wget", "-q", "-O", fpath, url], capture_output=True)
        if result.returncode != 0:
            print(f"  SG15 download failed, trying SSSP...")
            # Fallback to SSSP
            sssp_url = f"https://pseudopotentials.quantum-espresso.org/upf_files/{pseudo_files[elem]}"
            subprocess.run(["wget", "-q", "-O", os.path.join(PSEUDO_DIR, pseudo_files[elem]), sssp_url])
        else:
            print(f"  Downloaded {fname}")
    else:
        print(f"  {fname} already exists")
```

### Step 1: Structure Preparation (Optional MACE Pre-Relaxation)

```python
#!/usr/bin/env python3
"""
Optional: relax structure with MACE before DFT optical calculation.
"""
import warnings
warnings.filterwarnings("ignore")

from ase.build import bulk
from ase.io import write
from ase.optimize import LBFGS
from ase.filters import FrechetCellFilter
from mace.calculators import mace_mp

# Build or load structure
atoms = bulk("Si", "diamond", a=5.43)
# atoms = read("input_structure.cif")

calc = mace_mp(model="medium", dispersion=False, device="cpu")
atoms.calc = calc

filtered = FrechetCellFilter(atoms)
opt = LBFGS(filtered, logfile="relax_optical.log")
opt.run(fmax=0.005)

print(f"Relaxed lattice parameters: {atoms.cell.cellpar()}")
write("relaxed_for_optical.cif", atoms)
print("Structure saved to relaxed_for_optical.cif")
```

### Step 2: SCF Calculation (Dense K-Grid, Many Bands)

The SCF must use a **dense k-grid** and include enough **empty bands** for optical transitions up to the desired photon energy.

```python
#!/usr/bin/env python3
"""
Step 1: SCF calculation for dielectric function.
Key: dense k-grid + norm-conserving pseudopotentials.
"""
import os
import subprocess

PSEUDO_DIR = os.path.abspath("./pseudo")
OUTDIR = os.path.abspath("./tmp_epsilon")
os.makedirs(OUTDIR, exist_ok=True)
PREFIX = "si"

# For optical properties:
# - Dense k-grid (at least 12x12x12 for bulk Si)
# - ecutwfc: follow pseudopotential recommendation
# - nbnd will be set in the nscf step

scf_input = f"""&CONTROL
    calculation  = 'scf'
    prefix       = '{PREFIX}'
    outdir       = '{OUTDIR}'
    pseudo_dir   = '{PSEUDO_DIR}'
    tprnfor      = .true.
    tstress      = .true.
/

&SYSTEM
    ibrav        = 2
    celldm(1)    = 10.26
    nat          = 2
    ntyp         = 1
    ecutwfc      = 60.0
    ecutrho      = 240.0
    occupations  = 'fixed'
/

&ELECTRONS
    conv_thr = 1.0d-10
/

ATOMIC_SPECIES
  Si  28.0855  Si_ONCV_PBE-1.2.upf

ATOMIC_POSITIONS (crystal)
  Si  0.00  0.00  0.00
  Si  0.25  0.25  0.25

K_POINTS (automatic)
  12 12 12  0 0 0
"""

with open(f"{PREFIX}_scf.in", "w") as f:
    f.write(scf_input)

print("Running SCF calculation...")
result = subprocess.run(
    ["mpirun", "-np", "4", "pw.x", "-in", f"{PREFIX}_scf.in"],
    capture_output=True, text=True, timeout=600
)
with open(f"{PREFIX}_scf.out", "w") as f:
    f.write(result.stdout)

if "convergence has been achieved" in result.stdout:
    print("SCF converged successfully.")
else:
    print("WARNING: SCF may not have converged. Check output file.")
    print(result.stderr[-500:] if result.stderr else "")
```

### Step 3: NSCF Calculation (Many Empty Bands on Dense Grid)

epsilon.x requires a non-self-consistent calculation with many empty (conduction) bands. The number of bands determines the maximum photon energy you can probe.

```python
#!/usr/bin/env python3
"""
Step 2: NSCF calculation with many empty bands for epsilon.x.
Must use same prefix and outdir as SCF.
"""
import os
import subprocess

PSEUDO_DIR = os.path.abspath("./pseudo")
OUTDIR = os.path.abspath("./tmp_epsilon")
PREFIX = "si"

# Rule of thumb for nbnd:
# - Si has 4 valence electrons/atom, 2 atoms -> 8 electrons -> 4 occupied bands
# - For optical properties up to ~30 eV, need ~40-60 bands
# - More bands = higher energy transitions captured
NBND = 50

nscf_input = f"""&CONTROL
    calculation  = 'nscf'
    prefix       = '{PREFIX}'
    outdir       = '{OUTDIR}'
    pseudo_dir   = '{PSEUDO_DIR}'
/

&SYSTEM
    ibrav        = 2
    celldm(1)    = 10.26
    nat          = 2
    ntyp         = 1
    ecutwfc      = 60.0
    ecutrho      = 240.0
    occupations  = 'fixed'
    nbnd         = {NBND}
    nosym        = .true.
    noinv        = .true.
/

&ELECTRONS
    conv_thr = 1.0d-10
/

ATOMIC_SPECIES
  Si  28.0855  Si_ONCV_PBE-1.2.upf

ATOMIC_POSITIONS (crystal)
  Si  0.00  0.00  0.00
  Si  0.25  0.25  0.25

K_POINTS (automatic)
  12 12 12  0 0 0
"""

with open(f"{PREFIX}_nscf.in", "w") as f:
    f.write(nscf_input)

print("Running NSCF calculation with many empty bands...")
result = subprocess.run(
    ["mpirun", "-np", "4", "pw.x", "-in", f"{PREFIX}_nscf.in"],
    capture_output=True, text=True, timeout=1200
)
with open(f"{PREFIX}_nscf.out", "w") as f:
    f.write(result.stdout)

if result.returncode == 0:
    print(f"NSCF completed with {NBND} bands.")
else:
    print("ERROR in NSCF calculation!")
    print(result.stderr[-500:] if result.stderr else "")
```

### Step 4: epsilon.x -- Compute Dielectric Function

```python
#!/usr/bin/env python3
"""
Step 3: Run epsilon.x to compute the RPA dielectric function.
epsilon.x reads the wavefunctions from the NSCF step and computes
the imaginary part of epsilon via the Fermi golden rule, then obtains
the real part via Kramers-Kronig transformation.
"""
import os
import subprocess

OUTDIR = os.path.abspath("./tmp_epsilon")
PREFIX = "si"

# Key epsilon.x parameters:
# - intersmear: Lorentzian broadening in eV (smearing of delta functions)
# - wmax: maximum frequency (energy) in eV
# - nw: number of frequency points
# - shift: energy shift (scissors correction) in eV -- use if needed
# - calculation: 'eps' for dielectric function

epsilon_input = f"""&INPUTPP
    prefix       = '{PREFIX}'
    outdir       = '{OUTDIR}'
    calculation  = 'eps'
/

&ENERGY_GRID
    smeartype    = 'gauss'
    intersmear   = 0.1
    wmin         = 0.0
    wmax         = 30.0
    nw           = 1000
    shift        = 0.0
/
"""

with open(f"{PREFIX}_epsilon.in", "w") as f:
    f.write(epsilon_input)

print("Running epsilon.x...")
result = subprocess.run(
    ["epsilon.x", "-in", f"{PREFIX}_epsilon.in"],
    capture_output=True, text=True, timeout=600
)
with open(f"{PREFIX}_epsilon.out", "w") as f:
    f.write(result.stdout)

if result.returncode == 0:
    print("epsilon.x completed successfully.")
    print("Output files:")
    print("  epsr_Si.dat  -- Real part of dielectric function (xx, yy, zz)")
    print("  epsi_Si.dat  -- Imaginary part of dielectric function")
    # Note: actual filenames may vary; epsilon.x typically writes:
    #   epsr.dat, epsi.dat (or with prefix/element in name)
    # Check the output directory for actual filenames
else:
    print("ERROR in epsilon.x!")
    print(result.stdout[-1000:])
    print(result.stderr[-500:] if result.stderr else "")
```

### Step 5: Parse and Plot Dielectric Function

```python
#!/usr/bin/env python3
"""
Step 4: Parse epsilon.x output and plot the dielectric function.
Handles both the standard epsilon.x output format and produces
publication-quality plots.
"""
import numpy as np
import os
import glob
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ── Locate output files ─────────────────────────────────────────────
# epsilon.x output filenames vary by QE version. Common patterns:
#   epsr.dat / epsi.dat
#   epsr_<prefix>.dat / epsi_<prefix>.dat
#   or printed in the main output file

def find_epsilon_files():
    """Find the epsilon output data files."""
    patterns = [
        ("epsr*.dat", "epsi*.dat"),
        ("EPSR*.dat", "EPSI*.dat"),
    ]
    for real_pat, imag_pat in patterns:
        real_files = sorted(glob.glob(real_pat))
        imag_files = sorted(glob.glob(imag_pat))
        if real_files and imag_files:
            return real_files[0], imag_files[0]
    return None, None

def parse_epsilon_dat(filename):
    """
    Parse epsilon.x .dat output file.

    Format (typical):
      # energy(eV)  eps_xx  eps_yy  eps_zz
      0.000   12.345   12.345   12.345
      0.030   12.340   12.340   12.340
      ...

    Returns: energy (eV), eps_xx, eps_yy, eps_zz arrays
    """
    data = []
    with open(filename, "r") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or line.startswith("!"):
                continue
            parts = line.split()
            if len(parts) >= 4:
                try:
                    vals = [float(x) for x in parts[:4]]
                    data.append(vals)
                except ValueError:
                    continue
            elif len(parts) >= 2:
                # Some versions output only averaged epsilon
                try:
                    vals = [float(x) for x in parts[:2]]
                    data.append(vals + [vals[1], vals[1]])  # replicate for xx=yy=zz
                except ValueError:
                    continue
    data = np.array(data)
    return data[:, 0], data[:, 1], data[:, 2], data[:, 3]

def parse_epsilon_from_output(output_file):
    """
    Alternative parser: extract dielectric function directly from
    epsilon.x stdout if .dat files are not generated.
    """
    energy_re = []
    energy_im = []
    eps_re = []
    eps_im = []
    reading_real = False
    reading_imag = False

    with open(output_file, "r") as f:
        for line in f:
            if "Real part" in line or "REAL PART" in line:
                reading_real = True
                reading_imag = False
                continue
            if "Imaginary part" in line or "IMAGINARY PART" in line:
                reading_real = False
                reading_imag = True
                continue
            if reading_real or reading_imag:
                parts = line.split()
                if len(parts) >= 2:
                    try:
                        e = float(parts[0])
                        val = float(parts[1])
                        if reading_real:
                            energy_re.append(e)
                            eps_re.append(val)
                        else:
                            energy_im.append(e)
                            eps_im.append(val)
                    except ValueError:
                        reading_real = False
                        reading_imag = False

    return (np.array(energy_re), np.array(eps_re),
            np.array(energy_im), np.array(eps_im))

# ── Load data ────────────────────────────────────────────────────────
real_file, imag_file = find_epsilon_files()

if real_file and imag_file:
    print(f"Reading: {real_file}, {imag_file}")
    energy_r, eps1_xx, eps1_yy, eps1_zz = parse_epsilon_dat(real_file)
    energy_i, eps2_xx, eps2_yy, eps2_zz = parse_epsilon_dat(imag_file)

    # For cubic systems, xx = yy = zz; use average
    eps1_avg = (eps1_xx + eps1_yy + eps1_zz) / 3.0
    eps2_avg = (eps2_xx + eps2_yy + eps2_zz) / 3.0
else:
    print("No .dat files found, parsing from epsilon.x output...")
    energy_r, eps1_avg, energy_i, eps2_avg = parse_epsilon_from_output("si_epsilon.out")

# ── Extract key quantities ───────────────────────────────────────────
# Static dielectric constant (electronic contribution) = eps1(omega->0)
eps_static = eps1_avg[0] if len(eps1_avg) > 0 else float("nan")
print(f"\nStatic dielectric constant (electronic): {eps_static:.3f}")
print(f"  (Experimental value for Si: ~11.7)")

# Optical gap from onset of Im(epsilon)
# Find first energy where eps2 exceeds a threshold
threshold = 0.01 * np.max(eps2_avg) if len(eps2_avg) > 0 else 0
onset_indices = np.where(eps2_avg > threshold)[0]
if len(onset_indices) > 0:
    optical_gap = energy_i[onset_indices[0]]
    print(f"Optical gap (onset of absorption): {optical_gap:.2f} eV")
    print(f"  (Experimental direct gap of Si: ~3.4 eV, PBE underestimates)")

# ── Plot dielectric function ────────────────────────────────────────
fig, axes = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

# Real part
ax1 = axes[0]
ax1.plot(energy_r, eps1_avg, color="steelblue", linewidth=1.5, label=r"$\varepsilon_1(\omega)$ (avg)")
ax1.axhline(0, color="gray", linestyle="--", linewidth=0.5)
ax1.set_ylabel(r"$\varepsilon_1(\omega)$ (Real part)", fontsize=13)
ax1.set_title("Dielectric Function -- Si (QE epsilon.x, PBE)", fontsize=14)
ax1.legend(fontsize=11)
ax1.grid(alpha=0.3)
ax1.set_xlim(0, 15)

# Imaginary part
ax2 = axes[1]
ax2.plot(energy_i, eps2_avg, color="crimson", linewidth=1.5, label=r"$\varepsilon_2(\omega)$ (avg)")
ax2.fill_between(energy_i, 0, eps2_avg, alpha=0.15, color="crimson")
ax2.axhline(0, color="gray", linestyle="--", linewidth=0.5)
ax2.set_xlabel("Photon Energy (eV)", fontsize=13)
ax2.set_ylabel(r"$\varepsilon_2(\omega)$ (Imaginary part)", fontsize=13)
ax2.legend(fontsize=11)
ax2.grid(alpha=0.3)
ax2.set_xlim(0, 15)

plt.tight_layout()
plt.savefig("dielectric_function.png", dpi=200, bbox_inches="tight")
print("\nPlot saved: dielectric_function.png")

# ── Plot anisotropic components (for non-cubic systems) ──────────────
if real_file and imag_file:
    fig2, axes2 = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

    ax1 = axes2[0]
    ax1.plot(energy_r, eps1_xx, label=r"$\varepsilon_1^{xx}$", linewidth=1.2)
    ax1.plot(energy_r, eps1_yy, label=r"$\varepsilon_1^{yy}$", linewidth=1.2, linestyle="--")
    ax1.plot(energy_r, eps1_zz, label=r"$\varepsilon_1^{zz}$", linewidth=1.2, linestyle="-.")
    ax1.axhline(0, color="gray", linestyle="--", linewidth=0.5)
    ax1.set_ylabel(r"$\varepsilon_1(\omega)$", fontsize=13)
    ax1.set_title("Dielectric Function -- Anisotropic Components", fontsize=14)
    ax1.legend(fontsize=10)
    ax1.grid(alpha=0.3)
    ax1.set_xlim(0, 15)

    ax2 = axes2[1]
    ax2.plot(energy_i, eps2_xx, label=r"$\varepsilon_2^{xx}$", linewidth=1.2)
    ax2.plot(energy_i, eps2_yy, label=r"$\varepsilon_2^{yy}$", linewidth=1.2, linestyle="--")
    ax2.plot(energy_i, eps2_zz, label=r"$\varepsilon_2^{zz}$", linewidth=1.2, linestyle="-.")
    ax2.axhline(0, color="gray", linestyle="--", linewidth=0.5)
    ax2.set_xlabel("Photon Energy (eV)", fontsize=13)
    ax2.set_ylabel(r"$\varepsilon_2(\omega)$", fontsize=13)
    ax2.legend(fontsize=10)
    ax2.grid(alpha=0.3)
    ax2.set_xlim(0, 15)

    plt.tight_layout()
    plt.savefig("dielectric_function_anisotropic.png", dpi=200, bbox_inches="tight")
    print("Plot saved: dielectric_function_anisotropic.png")
```

### Step 6: Born Effective Charges and Ionic Dielectric Contribution (via ph.x)

For the **total** static dielectric constant (electronic + ionic), you need the Born effective charges and phonon frequencies from DFPT. The electronic part comes from epsilon.x; the ionic part comes from ph.x.

```python
#!/usr/bin/env python3
"""
Step 5 (optional): Compute Born effective charges and ionic contribution
to the static dielectric constant using ph.x (DFPT).

The total static dielectric constant is:
  epsilon_total = epsilon_electronic + epsilon_ionic

where epsilon_electronic comes from epsilon.x (Step 4) and
epsilon_ionic comes from the infrared-active phonon modes.
"""
import os
import subprocess
import re
import numpy as np

OUTDIR = os.path.abspath("./tmp_epsilon")
PSEUDO_DIR = os.path.abspath("./pseudo")
PREFIX = "si"

# NOTE: For Si (diamond structure), Born effective charges are zero
# by symmetry (no LO-TO splitting). This example shows the workflow
# for general materials. For polar materials (e.g., GaAs, BN, SiC),
# Born charges are nonzero and contribute to epsilon_ionic.

# Step 5a: Run ph.x at Gamma to get Born effective charges and
# the high-frequency dielectric constant
ph_input = f"""phonon calculation at Gamma for Born effective charges
&INPUTPH
    prefix     = '{PREFIX}'
    outdir     = '{OUTDIR}'
    tr2_ph     = 1.0d-14
    epsil      = .true.
    fildyn     = '{PREFIX}_gamma.dyn'
/

0.0 0.0 0.0
"""

with open(f"{PREFIX}_ph.in", "w") as f:
    f.write(ph_input)

print("Running ph.x at Gamma (this may take several minutes)...")
result = subprocess.run(
    ["mpirun", "-np", "4", "ph.x", "-in", f"{PREFIX}_ph.in"],
    capture_output=True, text=True, timeout=1800
)
with open(f"{PREFIX}_ph.out", "w") as f:
    f.write(result.stdout)

if result.returncode == 0:
    print("ph.x completed.")
else:
    print("ERROR in ph.x!")
    print(result.stderr[-500:] if result.stderr else "")

# Step 5b: Parse the output for dielectric tensor and Born charges
def parse_ph_output(filename):
    """Parse ph.x output for dielectric tensor and Born effective charges."""
    eps_inf = np.zeros((3, 3))
    born_charges = {}
    current_atom = None

    with open(filename, "r") as f:
        lines = f.readlines()

    # Find dielectric constant tensor
    for i, line in enumerate(lines):
        if "Dielectric constant in cartesian axes" in line:
            for j in range(1, 4):
                if i + j < len(lines):
                    parts = lines[i + j].strip().replace("(", "").replace(")", "").split()
                    vals = [float(x) for x in parts if x.replace(".", "").replace("-", "").replace("+", "").replace("e", "").replace("E", "").isdigit() or "." in x]
                    if len(vals) >= 3:
                        eps_inf[j-1, :] = vals[:3]
            break

    # Find Born effective charges
    for i, line in enumerate(lines):
        if "Effective charges" in line and "E-field" in line:
            j = i + 1
            while j < len(lines):
                if "atom" in lines[j]:
                    m = re.search(r"atom\s+(\d+)\s+(\w+)", lines[j])
                    if m:
                        atom_idx = int(m.group(1))
                        atom_sym = m.group(2)
                        zstar = np.zeros((3, 3))
                        for k in range(3):
                            if j + 1 + k < len(lines):
                                parts = lines[j + 1 + k].split()
                                vals = []
                                for p in parts:
                                    try:
                                        vals.append(float(p))
                                    except ValueError:
                                        pass
                                if len(vals) >= 3:
                                    zstar[k, :] = vals[:3]
                        born_charges[f"{atom_sym}_{atom_idx}"] = zstar
                        j += 4
                    else:
                        j += 1
                elif lines[j].strip() == "" or "---" in lines[j]:
                    j += 1
                else:
                    break

    return eps_inf, born_charges

eps_inf, born_charges = parse_ph_output(f"{PREFIX}_ph.out")

print(f"\nHigh-frequency dielectric tensor (epsilon_infinity):")
for i in range(3):
    print(f"  [{eps_inf[i, 0]:8.4f}  {eps_inf[i, 1]:8.4f}  {eps_inf[i, 2]:8.4f}]")
print(f"\nAverage epsilon_infinity = {np.trace(eps_inf)/3:.4f}")

if born_charges:
    print(f"\nBorn effective charges (Z*):")
    for atom, zstar in born_charges.items():
        print(f"\n  {atom}:")
        for i in range(3):
            print(f"    [{zstar[i, 0]:8.4f}  {zstar[i, 1]:8.4f}  {zstar[i, 2]:8.4f}]")
else:
    print("\nNo Born effective charges found (zero by symmetry for Si diamond).")
    print("For polar materials, Born charges will be printed here.")
```

### Complete Single-Script Workflow

```python
#!/usr/bin/env python3
"""
Complete dielectric function workflow in one script.
Runs: SCF -> NSCF -> epsilon.x -> parse and plot

Example: Silicon (diamond structure)
"""
import os
import subprocess
import re
import glob
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ── Configuration ────────────────────────────────────────────────────
PSEUDO_DIR = os.path.abspath("./pseudo")
OUTDIR = os.path.abspath("./tmp_epsilon")
os.makedirs(OUTDIR, exist_ok=True)
os.makedirs(PSEUDO_DIR, exist_ok=True)
PREFIX = "si"
ECUTWFC = 60.0
ECUTRHO = 240.0  # 4*ecutwfc for NC pseudopotentials
NBND = 50        # many empty bands for optical transitions
NPROC = 4
KGRID = "12 12 12"

# Pseudopotential filename (must exist in PSEUDO_DIR)
PP_FILE = "Si_ONCV_PBE-1.2.upf"

# ── Step 1: SCF ──────────────────────────────────────────────────────
scf_input = f"""&CONTROL
    calculation  = 'scf'
    prefix       = '{PREFIX}'
    outdir       = '{OUTDIR}'
    pseudo_dir   = '{PSEUDO_DIR}'
/
&SYSTEM
    ibrav        = 2
    celldm(1)    = 10.26
    nat          = 2
    ntyp         = 1
    ecutwfc      = {ECUTWFC}
    ecutrho      = {ECUTRHO}
    occupations  = 'fixed'
/
&ELECTRONS
    conv_thr = 1.0d-10
/
ATOMIC_SPECIES
  Si  28.0855  {PP_FILE}
ATOMIC_POSITIONS (crystal)
  Si  0.00  0.00  0.00
  Si  0.25  0.25  0.25
K_POINTS (automatic)
  {KGRID}  0 0 0
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

# ── Step 2: NSCF with many bands ────────────────────────────────────
nscf_input = f"""&CONTROL
    calculation  = 'nscf'
    prefix       = '{PREFIX}'
    outdir       = '{OUTDIR}'
    pseudo_dir   = '{PSEUDO_DIR}'
/
&SYSTEM
    ibrav        = 2
    celldm(1)    = 10.26
    nat          = 2
    ntyp         = 1
    ecutwfc      = {ECUTWFC}
    ecutrho      = {ECUTRHO}
    occupations  = 'fixed'
    nbnd         = {NBND}
    nosym        = .true.
    noinv        = .true.
/
&ELECTRONS
    conv_thr = 1.0d-10
/
ATOMIC_SPECIES
  Si  28.0855  {PP_FILE}
ATOMIC_POSITIONS (crystal)
  Si  0.00  0.00  0.00
  Si  0.25  0.25  0.25
K_POINTS (automatic)
  {KGRID}  0 0 0
"""
with open(f"{PREFIX}_nscf.in", "w") as f:
    f.write(nscf_input)

print("[2/4] Running NSCF with nbnd={NBND}...")
r = subprocess.run(["mpirun", "-np", str(NPROC), "pw.x", "-in", f"{PREFIX}_nscf.in"],
                    capture_output=True, text=True, timeout=1200)
with open(f"{PREFIX}_nscf.out", "w") as f:
    f.write(r.stdout)
assert r.returncode == 0, "NSCF failed!"
print("      NSCF completed.")

# ── Step 3: epsilon.x ────────────────────────────────────────────────
epsilon_input = f"""&INPUTPP
    prefix       = '{PREFIX}'
    outdir       = '{OUTDIR}'
    calculation  = 'eps'
/
&ENERGY_GRID
    smeartype    = 'gauss'
    intersmear   = 0.1
    wmin         = 0.0
    wmax         = 30.0
    nw           = 1000
    shift        = 0.0
/
"""
with open(f"{PREFIX}_epsilon.in", "w") as f:
    f.write(epsilon_input)

print("[3/4] Running epsilon.x...")
r = subprocess.run(["epsilon.x", "-in", f"{PREFIX}_epsilon.in"],
                    capture_output=True, text=True, timeout=600)
with open(f"{PREFIX}_epsilon.out", "w") as f:
    f.write(r.stdout)
assert r.returncode == 0, f"epsilon.x failed!\n{r.stderr}"
print("      epsilon.x completed.")

# ── Step 4: Parse and plot ───────────────────────────────────────────
print("[4/4] Parsing and plotting...")

# Find output files
real_file = None
imag_file = None
for pat_r, pat_i in [("epsr*.dat", "epsi*.dat"), ("EPSR*.dat", "EPSI*.dat")]:
    rf = sorted(glob.glob(pat_r))
    imf = sorted(glob.glob(pat_i))
    if rf and imf:
        real_file, imag_file = rf[0], imf[0]
        break

def load_eps_dat(filename):
    data = []
    with open(filename) as fh:
        for line in fh:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            try:
                vals = [float(x) for x in parts[:4]]
                if len(vals) == 4:
                    data.append(vals)
                elif len(vals) >= 2:
                    data.append([vals[0], vals[1], vals[1], vals[1]])
            except ValueError:
                continue
    return np.array(data)

if real_file and imag_file:
    dr = load_eps_dat(real_file)
    di = load_eps_dat(imag_file)
    energy = dr[:, 0]
    eps1 = (dr[:, 1] + dr[:, 2] + dr[:, 3]) / 3.0
    eps2 = (di[:, 1] + di[:, 2] + di[:, 3]) / 3.0
else:
    raise FileNotFoundError("Could not find epsr/epsi .dat files. Check epsilon.x output.")

# Static dielectric constant
print(f"\n=== Results ===")
print(f"Static dielectric constant (electronic): eps1(0) = {eps1[0]:.3f}")
print(f"  (Experimental Si: ~11.7)")

# Optical gap
threshold = 0.01 * np.max(eps2)
onset = np.where(eps2 > threshold)[0]
if len(onset) > 0:
    print(f"Optical gap (onset of Im eps): {energy[onset[0]]:.2f} eV")

# Peak positions in eps2
from scipy.signal import find_peaks
peaks, props = find_peaks(eps2, height=0.5*np.max(eps2), distance=20)
for p in peaks:
    print(f"Peak in eps2 at {energy[p]:.2f} eV, value = {eps2[p]:.2f}")

# Plot
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

ax1.plot(energy, eps1, "b-", linewidth=1.5, label=r"$\varepsilon_1(\omega)$")
ax1.axhline(0, color="gray", linestyle="--", linewidth=0.5)
ax1.set_ylabel(r"$\varepsilon_1(\omega)$", fontsize=14)
ax1.set_title("Dielectric Function -- Si (PBE, RPA)", fontsize=15)
ax1.legend(fontsize=12)
ax1.grid(alpha=0.3)

ax2.plot(energy, eps2, "r-", linewidth=1.5, label=r"$\varepsilon_2(\omega)$")
ax2.fill_between(energy, 0, eps2, alpha=0.15, color="red")
ax2.set_xlabel("Photon Energy (eV)", fontsize=14)
ax2.set_ylabel(r"$\varepsilon_2(\omega)$", fontsize=14)
ax2.legend(fontsize=12)
ax2.grid(alpha=0.3)

for ax in (ax1, ax2):
    ax.set_xlim(0, 15)

plt.tight_layout()
plt.savefig("dielectric_function.png", dpi=200, bbox_inches="tight")
print(f"\nPlot saved: dielectric_function.png")
```

---

## Key Parameters

| Parameter | Where | Typical Value | Notes |
|---|---|---|---|
| `ecutwfc` | SCF/NSCF | 60--80 Ry | Follow pseudopotential recommendation; NC PPs often need higher cutoff |
| `nbnd` | NSCF | 2--4x occupied | More bands = higher energy transitions; 50+ for up to ~30 eV |
| `K_POINTS` | SCF/NSCF | 12x12x12+ | Optical properties need dense k-grids; converge with respect to k-mesh |
| `nosym` / `noinv` | NSCF | `.true.` | epsilon.x requires the full k-point set without symmetry reduction |
| `intersmear` | epsilon.x | 0.05--0.3 eV | Gaussian/Lorentzian broadening; too small = noisy, too large = washed out |
| `wmax` | epsilon.x | 20--40 eV | Maximum photon energy; must be consistent with nbnd |
| `nw` | epsilon.x | 500--2000 | Number of frequency grid points; more = finer resolution |
| `shift` | epsilon.x | 0.0--1.0 eV | Scissors correction to open the PBE gap; set to (expt_gap - PBE_gap) |
| `smeartype` | epsilon.x | 'gauss' or 'lorentz' | Gaussian gives smoother spectra; Lorentzian has longer tails |
| `occupations` | SCF/NSCF | 'fixed' (insulators) | For metals, use 'smearing' with appropriate degauss |

## Interpreting Results

- **Static dielectric constant**: eps1(omega=0) gives the electronic (high-frequency) dielectric constant. For Si, PBE gives ~12, experiment ~11.7. For total static epsilon (including ionic), add the phonon contribution from ph.x.

- **Imaginary part peaks**: Peaks in eps2 correspond to strong interband transitions (joint density of states peaks). The first peak onset corresponds to the optical gap.

- **Real part zero crossings**: Where eps1 crosses zero from positive to negative with large eps2 indicates strong absorption. Where eps1 crosses zero from negative to positive near eps2~0 indicates a plasma frequency (relevant for metals).

- **Optical gap**: The onset of eps2 gives the direct optical gap. PBE systematically underestimates this by 30--50%. Apply a scissors correction (`shift` in epsilon.x) to align with experiment.

- **Anisotropy**: For non-cubic crystals, eps_xx, eps_yy, eps_zz differ. The ordinary (o) and extraordinary (e) refractive indices can be derived from these components.

## Common Issues

| Problem | Solution |
|---|---|
| epsilon.x crashes with "wrong number of k-points" | Set `nosym=.true.` and `noinv=.true.` in the NSCF input to generate the full k-mesh |
| Spectra too noisy / spiky | Increase `intersmear` (broadening) or use a denser k-grid |
| eps2 is zero everywhere | Check that `nbnd` includes enough empty states; verify the NSCF completed successfully |
| Static epsilon too high | Common with PBE for small-gap materials; the gap underestimation inflates epsilon |
| No .dat output files found | Check the epsilon.x stdout for error messages; ensure prefix/outdir match the NSCF |
| Negative eps2 values | Should not happen physically; indicates numerical issues. Increase k-points and ecutwfc |
| Need ultrasoft pseudopotentials | epsilon.x is designed for NC PPs. With US/PAW, results may be unreliable. Switch to ONCV PPs |
| Born effective charges are all zero | Expected for non-polar materials (e.g., Si, Ge). For polar materials (GaAs, BN), they will be nonzero |
| epsilon.x very slow | Reduce `nbnd` or `nw`. epsilon.x scales as O(nbnd^2 * nk * nw) |
