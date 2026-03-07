# Density of States (DOS / PDOS)

## When to Use

- You want to understand the electronic structure of a material in terms of energy distribution of states.
- You need to identify orbital contributions (s, p, d, f) to bands near the Fermi level.
- You are analyzing bonding character, hybridization, or crystal field splitting.
- You want to confirm metallic vs. insulating character from the DOS at the Fermi level.
- You need atom-resolved or orbital-resolved DOS for a multi-element compound.

## Method Selection (MACE vs QE)

| Criterion | ASE + MACE | QE DFT |
|---|---|---|
| Total DOS | Not available | Full DOS from Kohn-Sham eigenvalues |
| Projected DOS | Not available | Atom- and orbital-resolved PDOS via projwfc.x |
| Reason | MACE is a force field with no electronic eigenvalues | DFT provides the full electronic spectrum |

**MACE cannot compute DOS.** It has no concept of electronic states. Always use QE for DOS. MACE can pre-relax the structure.

## Prerequisites

- A relaxed crystal structure (use the `scf-relax` skill).
- Pseudopotential files in `./pseudo/`.
- Python packages: `numpy`, `matplotlib`.
- QE executables: `pw.x`, `dos.x`, `projwfc.x`.

---

## Detailed Steps

### Method A: ASE + MACE (Structure Preparation Only)

MACE cannot compute electronic DOS. Use it only to relax the structure before a QE DOS calculation.

```python
#!/usr/bin/env python3
"""
Relax structure with MACE before QE DOS calculation.
"""
import warnings
warnings.filterwarnings("ignore")

from ase.build import bulk
from ase.io import write
from ase.optimize import LBFGS
from ase.filters import FrechetCellFilter
from mace.calculators import mace_mp

atoms = bulk("Si", "diamond", a=5.43)
calc = mace_mp(model="medium", dispersion=False, device="cpu")
atoms.calc = calc

filtered = FrechetCellFilter(atoms)
opt = LBFGS(filtered, logfile="relax_for_dos.log")
opt.run(fmax=0.005)

print(f"Relaxed cell: {atoms.cell.cellpar()}")
write("relaxed_for_dos.cif", atoms)
print("Saved: relaxed_for_dos.cif -- use for QE DOS calculation.")
```

### Method B: QE DFT Density of States

The QE DOS workflow has three main steps:
1. **SCF**: Self-consistent calculation on a standard k-grid.
2. **NSCF**: Non-self-consistent calculation on a much denser k-grid (reads SCF charge density).
3. **dos.x**: Compute total DOS. And/or **projwfc.x**: Compute projected DOS (PDOS).

#### Step B1: SCF Calculation

```python
#!/usr/bin/env python3
"""
Step 1 of DOS workflow: SCF calculation.
Example: silicon (diamond structure).
"""
import os
import subprocess

PSEUDO_DIR = os.path.abspath("./pseudo")
OUTDIR = os.path.abspath("./tmp_dos")
os.makedirs(OUTDIR, exist_ok=True)
PREFIX = "si"

scf_input = f"""&CONTROL
    calculation = 'scf'
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

with open(f"{PREFIX}_dos_scf.in", "w") as f:
    f.write(scf_input)

print("[1/4] Running SCF...")
result = subprocess.run(
    ["mpirun", "-np", "4", "pw.x", "-in", f"{PREFIX}_dos_scf.in"],
    capture_output=True, text=True, timeout=600
)
with open(f"{PREFIX}_dos_scf.out", "w") as f:
    f.write(result.stdout)

if "convergence has been achieved" in result.stdout:
    print("      SCF converged.")
else:
    print("      WARNING: SCF may not have converged!")
```

#### Step B2: NSCF Calculation (Dense K-Grid)

The NSCF step recalculates eigenvalues on a much denser k-grid using the converged charge density from the SCF step. This is critical for a smooth DOS.

```python
#!/usr/bin/env python3
"""
Step 2 of DOS workflow: NSCF on a dense k-grid.
IMPORTANT: prefix and outdir MUST match the SCF step.
"""
import os
import subprocess

PSEUDO_DIR = os.path.abspath("./pseudo")
OUTDIR = os.path.abspath("./tmp_dos")
PREFIX = "si"

# Dense k-grid: typically 2x--3x the SCF grid in each direction
# For metals, use even denser grids (e.g., 24x24x24)
DENSE_KGRID = (16, 16, 16)

nscf_input = f"""&CONTROL
    calculation = 'nscf'
    prefix      = '{PREFIX}'
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
    occupations = 'tetrahedra'
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

K_POINTS (automatic)
  {DENSE_KGRID[0]} {DENSE_KGRID[1]} {DENSE_KGRID[2]}  0 0 0
"""

with open(f"{PREFIX}_dos_nscf.in", "w") as f:
    f.write(nscf_input)

print("[2/4] Running NSCF on dense k-grid...")
result = subprocess.run(
    ["mpirun", "-np", "4", "pw.x", "-in", f"{PREFIX}_dos_nscf.in"],
    capture_output=True, text=True, timeout=1200
)
with open(f"{PREFIX}_dos_nscf.out", "w") as f:
    f.write(result.stdout)

if result.returncode == 0:
    print("      NSCF completed.")
else:
    print("      ERROR in NSCF!")
    print(result.stderr[-500:] if result.stderr else "")
```

**Important notes on the NSCF input:**
- `calculation = 'nscf'` reads the charge density from the SCF and recomputes eigenvalues.
- `occupations = 'tetrahedra'` is recommended for DOS (more accurate integration than smearing). For metals with complicated Fermi surfaces, `'tetrahedra_lin'` or `'tetrahedra_opt'` may be better.
- `nbnd` should include empty bands above the Fermi level so you can see the conduction band DOS.
- The k-grid must be **much denser** than the SCF grid. A 16x16x16 grid is a good starting point for bulk Si.

#### Step B3a: Total DOS with dos.x

```python
#!/usr/bin/env python3
"""
Step 3a: Compute total DOS using dos.x.
"""
import os
import subprocess
import re

OUTDIR = os.path.abspath("./tmp_dos")
PREFIX = "si"

# Extract Fermi energy from NSCF output
e_fermi = 0.0
with open(f"{PREFIX}_dos_nscf.out", "r") as f:
    for line in f:
        if "the Fermi energy is" in line:
            m = re.search(r"is\s+([-\d.]+)", line)
            if m:
                e_fermi = float(m.group(1))
        if "highest occupied" in line:
            m = re.search(r":\s+([-\d.]+)", line)
            if m:
                e_fermi = float(m.group(1))
        if "highest occupied, lowest unoccupied" in line:
            m = re.search(r":\s+([-\d.]+)\s+([-\d.]+)", line)
            if m:
                e_fermi = (float(m.group(1)) + float(m.group(2))) / 2

print(f"Fermi energy: {e_fermi:.4f} eV")

dos_input = f"""&DOS
    prefix  = '{PREFIX}'
    outdir  = '{OUTDIR}'
    fildos  = '{PREFIX}_dos.dat'
    Emin    = {e_fermi - 15.0}
    Emax    = {e_fermi + 10.0}
    DeltaE  = 0.01
/
"""

with open(f"{PREFIX}_dos.in", "w") as f:
    f.write(dos_input)

print("[3a/4] Running dos.x...")
result = subprocess.run(
    ["dos.x", "-in", f"{PREFIX}_dos.in"],
    capture_output=True, text=True, timeout=120
)
with open(f"{PREFIX}_dos_pp.out", "w") as f:
    f.write(result.stdout)

if result.returncode == 0:
    print(f"      Total DOS saved to {PREFIX}_dos.dat")
else:
    print("      ERROR in dos.x!")
    print(result.stderr[-500:] if result.stderr else "")
```

#### Step B3b: Projected DOS with projwfc.x

```python
#!/usr/bin/env python3
"""
Step 3b: Compute projected DOS (PDOS) using projwfc.x.
This gives atom-resolved and orbital-resolved DOS.
"""
import os
import subprocess

OUTDIR = os.path.abspath("./tmp_dos")
PREFIX = "si"

projwfc_input = f"""&PROJWFC
    prefix  = '{PREFIX}'
    outdir  = '{OUTDIR}'
    filpdos = '{PREFIX}_pdos'
    Emin    = -15.0
    Emax    = 10.0
    DeltaE  = 0.01
    ngauss  = 0
    degauss = 0.01
/
"""

with open(f"{PREFIX}_projwfc.in", "w") as f:
    f.write(projwfc_input)

print("[3b/4] Running projwfc.x...")
result = subprocess.run(
    ["projwfc.x", "-in", f"{PREFIX}_projwfc.in"],
    capture_output=True, text=True, timeout=300
)
with open(f"{PREFIX}_projwfc_pp.out", "w") as f:
    f.write(result.stdout)

if result.returncode == 0:
    print("      projwfc.x completed.")
    # List output files
    pdos_files = [f for f in os.listdir(".") if f.startswith(f"{PREFIX}_pdos")]
    print("      PDOS files generated:")
    for pf in sorted(pdos_files):
        print(f"        {pf}")
else:
    print("      ERROR in projwfc.x!")
    print(result.stderr[-500:] if result.stderr else "")
```

**Understanding projwfc.x output files:**
- `{prefix}_pdos.pdos_tot` -- Total DOS (same as dos.x but computed differently).
- `{prefix}_pdos.pdos_atm#N(Element)_wfc#M(orbital)` -- PDOS for atom N, orbital M.
  - Example: `si_pdos.pdos_atm#1(Si)_wfc#1(s)` = s-orbital PDOS of atom 1 (Si).
  - Example: `si_pdos.pdos_atm#1(Si)_wfc#2(p)` = p-orbital PDOS of atom 1 (Si).

#### Step B4: Plot DOS and PDOS with Python

```python
#!/usr/bin/env python3
"""
Step 4: Complete plotting script for total DOS and projected DOS.
Handles Fermi level alignment, atom-projected, and orbital-projected plots.
"""
import os
import re
import glob
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

PREFIX = "si"

# ── Parse Fermi energy ─────────────────────────────────────────────
def get_fermi_energy(nscf_output):
    """Extract Fermi energy from NSCF output."""
    e_fermi = 0.0
    with open(nscf_output, "r") as f:
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

e_fermi = get_fermi_energy(f"{PREFIX}_dos_nscf.out")
print(f"Fermi energy: {e_fermi:.4f} eV")

# ── Parse total DOS from dos.x output ──────────────────────────────
def parse_dos_file(filename):
    """
    Parse QE dos.x output file.
    Returns: energy (eV), dos (states/eV), integrated_dos
    """
    data = np.loadtxt(filename, comments="#")
    # Columns: E(eV)  dos(E)  int_dos(E)
    energy = data[:, 0]
    dos = data[:, 1]
    int_dos = data[:, 2] if data.shape[1] > 2 else None
    return energy, dos, int_dos

# ── Parse PDOS files from projwfc.x ────────────────────────────────
def parse_pdos_file(filename):
    """
    Parse a single PDOS file from projwfc.x.
    Returns: energy array, pdos array(s).
    For non-spin-polarized: columns are E, ldos, pdos1, pdos2, ...
    """
    # Read header to check format
    with open(filename, "r") as f:
        header = f.readline()

    data = np.loadtxt(filename, comments="#")
    energy = data[:, 0]
    # Column 1 is the total LDOS for this projection, columns 2+ are m-resolved
    ldos = data[:, 1]
    return energy, ldos

def parse_all_pdos(prefix):
    """
    Parse all PDOS files and organize by atom and orbital.
    Returns a dict: {(atom_index, element, orbital): (energy, pdos)}
    """
    pdos_dict = {}
    pattern = f"{prefix}_pdos.pdos_atm#*"
    files = glob.glob(pattern)

    for fpath in sorted(files):
        fname = os.path.basename(fpath)
        # Parse filename: prefix_pdos.pdos_atm#1(Si)_wfc#1(s)
        m = re.search(r"atm#(\d+)\((\w+)\)_wfc#(\d+)\((\w+)\)", fname)
        if m:
            atom_idx = int(m.group(1))
            element = m.group(2)
            wfc_idx = int(m.group(3))
            orbital = m.group(4)

            energy, pdos = parse_pdos_file(fpath)
            pdos_dict[(atom_idx, element, orbital)] = (energy, pdos)

    return pdos_dict

# ── Parse total PDOS ───────────────────────────────────────────────
def parse_pdos_total(prefix):
    """Parse the total PDOS file from projwfc.x."""
    fname = f"{prefix}_pdos.pdos_tot"
    if os.path.exists(fname):
        data = np.loadtxt(fname, comments="#")
        return data[:, 0], data[:, 1]  # energy, total_dos
    return None, None

# ══════════════════════════════════════════════════════════════════
#  PLOT 1: Total DOS
# ══════════════════════════════════════════════════════════════════
print("\nPlotting total DOS...")

energy_dos, dos, int_dos = parse_dos_file(f"{PREFIX}_dos.dat")

fig, ax = plt.subplots(figsize=(8, 5))
ax.plot(energy_dos - e_fermi, dos, color="steelblue", linewidth=1.2, label="Total DOS")
ax.axvline(0, color="red", linestyle="--", linewidth=0.8, label="$E_F$")
ax.fill_between(energy_dos - e_fermi, dos, alpha=0.15, color="steelblue")

ax.set_xlabel("$E - E_F$ (eV)", fontsize=14)
ax.set_ylabel("DOS (states/eV/cell)", fontsize=14)
ax.set_title("Total Density of States", fontsize=15)
ax.set_xlim(-12, 8)
ax.set_ylim(0, None)
ax.legend(fontsize=12)
ax.grid(alpha=0.3)

plt.tight_layout()
plt.savefig("dos_total.png", dpi=200, bbox_inches="tight")
print("  Saved: dos_total.png")

# ══════════════════════════════════════════════════════════════════
#  PLOT 2: Orbital-Projected DOS
# ══════════════════════════════════════════════════════════════════
print("\nPlotting orbital-projected DOS...")

pdos_dict = parse_all_pdos(PREFIX)

if pdos_dict:
    # Group by orbital type (sum over all atoms of same element)
    orbital_dos = {}
    for (atom_idx, element, orbital), (energy, pdos) in pdos_dict.items():
        key = f"{element}-{orbital}"
        if key not in orbital_dos:
            orbital_dos[key] = (energy, np.zeros_like(pdos))
        orbital_dos[key] = (energy, orbital_dos[key][1] + pdos)

    # Color map for orbitals
    orbital_colors = {
        "s": "#e74c3c",    # red
        "p": "#3498db",    # blue
        "d": "#2ecc71",    # green
        "f": "#9b59b6",    # purple
    }

    fig, ax = plt.subplots(figsize=(8, 5))

    for key in sorted(orbital_dos.keys()):
        energy, pdos = orbital_dos[key]
        element, orb = key.split("-")
        color = orbital_colors.get(orb, "gray")
        ax.plot(energy - e_fermi, pdos, linewidth=1.5, label=f"{element} ({orb})",
                color=color)
        ax.fill_between(energy - e_fermi, pdos, alpha=0.1, color=color)

    ax.axvline(0, color="red", linestyle="--", linewidth=0.8, label="$E_F$")
    ax.set_xlabel("$E - E_F$ (eV)", fontsize=14)
    ax.set_ylabel("PDOS (states/eV/cell)", fontsize=14)
    ax.set_title("Orbital-Projected Density of States", fontsize=15)
    ax.set_xlim(-12, 8)
    ax.set_ylim(0, None)
    ax.legend(fontsize=11, ncol=2)
    ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig("dos_orbital_projected.png", dpi=200, bbox_inches="tight")
    print("  Saved: dos_orbital_projected.png")

# ══════════════════════════════════════════════════════════════════
#  PLOT 3: Atom-Projected DOS (useful for multi-element systems)
# ══════════════════════════════════════════════════════════════════
print("\nPlotting atom-projected DOS...")

if pdos_dict:
    # Group by atom (sum all orbitals for each atom)
    atom_dos = {}
    for (atom_idx, element, orbital), (energy, pdos) in pdos_dict.items():
        key = f"Atom {atom_idx} ({element})"
        if key not in atom_dos:
            atom_dos[key] = (energy, np.zeros_like(pdos))
        atom_dos[key] = (energy, atom_dos[key][1] + pdos)

    # For systems with equivalent atoms, group by element instead
    element_dos = {}
    for (atom_idx, element, orbital), (energy, pdos) in pdos_dict.items():
        if element not in element_dos:
            element_dos[element] = (energy, np.zeros_like(pdos))
        element_dos[element] = (energy, element_dos[element][1] + pdos)

    fig, ax = plt.subplots(figsize=(8, 5))

    colors = plt.cm.Set1(np.linspace(0, 1, max(len(element_dos), 3)))
    for i, (element, (energy, pdos)) in enumerate(sorted(element_dos.items())):
        ax.plot(energy - e_fermi, pdos, linewidth=1.5, label=element, color=colors[i])
        ax.fill_between(energy - e_fermi, pdos, alpha=0.1, color=colors[i])

    ax.axvline(0, color="red", linestyle="--", linewidth=0.8, label="$E_F$")
    ax.set_xlabel("$E - E_F$ (eV)", fontsize=14)
    ax.set_ylabel("PDOS (states/eV/cell)", fontsize=14)
    ax.set_title("Atom-Projected Density of States", fontsize=15)
    ax.set_xlim(-12, 8)
    ax.set_ylim(0, None)
    ax.legend(fontsize=12)
    ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig("dos_atom_projected.png", dpi=200, bbox_inches="tight")
    print("  Saved: dos_atom_projected.png")

# ══════════════════════════════════════════════════════════════════
#  PLOT 4: Combined DOS + PDOS (publication style)
# ══════════════════════════════════════════════════════════════════
print("\nPlotting publication-style combined figure...")

fig, axes = plt.subplots(2, 1, figsize=(8, 8), sharex=True,
                          gridspec_kw={"height_ratios": [1, 1], "hspace": 0.05})

# Top panel: Total DOS
ax1 = axes[0]
ax1.plot(energy_dos - e_fermi, dos, color="black", linewidth=1.2, label="Total DOS")
ax1.fill_between(energy_dos - e_fermi, dos, alpha=0.15, color="gray")
ax1.axvline(0, color="red", linestyle="--", linewidth=0.8)
ax1.set_ylabel("DOS (states/eV/cell)", fontsize=13)
ax1.set_title("Density of States", fontsize=15)
ax1.legend(fontsize=11, loc="upper right")
ax1.set_xlim(-12, 8)
ax1.set_ylim(0, None)
ax1.grid(alpha=0.3)

# Bottom panel: PDOS by orbital
ax2 = axes[1]
if pdos_dict:
    for key in sorted(orbital_dos.keys()):
        energy, pdos = orbital_dos[key]
        element, orb = key.split("-")
        color = orbital_colors.get(orb, "gray")
        ax2.plot(energy - e_fermi, pdos, linewidth=1.5, label=f"{element} ({orb})",
                 color=color)
        ax2.fill_between(energy - e_fermi, pdos, alpha=0.1, color=color)

ax2.axvline(0, color="red", linestyle="--", linewidth=0.8)
ax2.set_xlabel("$E - E_F$ (eV)", fontsize=13)
ax2.set_ylabel("PDOS (states/eV/cell)", fontsize=13)
ax2.legend(fontsize=11, ncol=2, loc="upper right")
ax2.set_ylim(0, None)
ax2.grid(alpha=0.3)

plt.savefig("dos_combined.png", dpi=200, bbox_inches="tight")
print("  Saved: dos_combined.png")

print("\nDone. All DOS plots generated.")
```

#### Complete Single-Script Workflow

```python
#!/usr/bin/env python3
"""
Complete DOS workflow in one script.
Runs: SCF -> NSCF (dense k-grid) -> dos.x + projwfc.x -> plot
"""
import os
import subprocess
import re
import glob
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ══════════════════════════════════════════════════════════════════
#  Configuration
# ══════════════════════════════════════════════════════════════════
PSEUDO_DIR = os.path.abspath("./pseudo")
OUTDIR = os.path.abspath("./tmp_dos_full")
os.makedirs(OUTDIR, exist_ok=True)
PREFIX = "si"
ECUTWFC = 50.0
ECUTRHO = 400.0
NPROC = 4

# Structure: Silicon (diamond)
# Using ibrav=2 for FCC. Modify for your system.
ALAT_BOHR = 10.26
NAT = 2
NTYP = 1
NBND = 12  # include empty bands for conduction DOS
SCF_KGRID = "8 8 8"
NSCF_KGRID = "16 16 16"  # dense grid for DOS

ATOMIC_SPECIES = "  Si  28.0855  Si.pbe-n-rrkjus_psl.1.0.0.UPF"
ATOMIC_POSITIONS = """\
  Si  0.00  0.00  0.00
  Si  0.25  0.25  0.25"""

# ══════════════════════════════════════════════════════════════════
#  Step 1: SCF
# ══════════════════════════════════════════════════════════════════
scf_input = f"""&CONTROL
    calculation = 'scf'
    prefix      = '{PREFIX}'
    outdir      = '{OUTDIR}'
    pseudo_dir  = '{PSEUDO_DIR}'
/
&SYSTEM
    ibrav       = 2
    celldm(1)   = {ALAT_BOHR}
    nat         = {NAT}
    ntyp        = {NTYP}
    ecutwfc     = {ECUTWFC}
    ecutrho     = {ECUTRHO}
    occupations = 'smearing'
    smearing    = 'cold'
    degauss     = 0.01
/
&ELECTRONS
    conv_thr = 1.0d-8
/
ATOMIC_SPECIES
{ATOMIC_SPECIES}
ATOMIC_POSITIONS (crystal)
{ATOMIC_POSITIONS}
K_POINTS (automatic)
  {SCF_KGRID}  0 0 0
"""

with open(f"{PREFIX}_scf.in", "w") as f:
    f.write(scf_input)

print("[1/5] Running SCF...")
r = subprocess.run(["mpirun", "-np", str(NPROC), "pw.x", "-in", f"{PREFIX}_scf.in"],
                    capture_output=True, text=True, timeout=600)
with open(f"{PREFIX}_scf.out", "w") as f:
    f.write(r.stdout)
assert "convergence has been achieved" in r.stdout, "SCF did not converge!"
print("      SCF converged.")

# ══════════════════════════════════════════════════════════════════
#  Step 2: NSCF (dense k-grid)
# ══════════════════════════════════════════════════════════════════
nscf_input = f"""&CONTROL
    calculation = 'nscf'
    prefix      = '{PREFIX}'
    outdir      = '{OUTDIR}'
    pseudo_dir  = '{PSEUDO_DIR}'
    verbosity   = 'high'
/
&SYSTEM
    ibrav       = 2
    celldm(1)   = {ALAT_BOHR}
    nat         = {NAT}
    ntyp        = {NTYP}
    ecutwfc     = {ECUTWFC}
    ecutrho     = {ECUTRHO}
    occupations = 'tetrahedra'
    nbnd        = {NBND}
/
&ELECTRONS
    conv_thr = 1.0d-8
/
ATOMIC_SPECIES
{ATOMIC_SPECIES}
ATOMIC_POSITIONS (crystal)
{ATOMIC_POSITIONS}
K_POINTS (automatic)
  {NSCF_KGRID}  0 0 0
"""

with open(f"{PREFIX}_nscf.in", "w") as f:
    f.write(nscf_input)

print("[2/5] Running NSCF on dense k-grid...")
r = subprocess.run(["mpirun", "-np", str(NPROC), "pw.x", "-in", f"{PREFIX}_nscf.in"],
                    capture_output=True, text=True, timeout=1200)
with open(f"{PREFIX}_nscf.out", "w") as f:
    f.write(r.stdout)
assert r.returncode == 0, "NSCF failed!"
print("      NSCF completed.")

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

# ══════════════════════════════════════════════════════════════════
#  Step 3: dos.x (total DOS)
# ══════════════════════════════════════════════════════════════════
dos_input = f"""&DOS
    prefix  = '{PREFIX}'
    outdir  = '{OUTDIR}'
    fildos  = '{PREFIX}_dos.dat'
    Emin    = {e_fermi - 15.0}
    Emax    = {e_fermi + 10.0}
    DeltaE  = 0.01
/
"""
with open(f"{PREFIX}_dos.in", "w") as f:
    f.write(dos_input)

print("[3/5] Running dos.x...")
r = subprocess.run(["dos.x", "-in", f"{PREFIX}_dos.in"],
                    capture_output=True, text=True, timeout=120)
with open(f"{PREFIX}_dos_out.log", "w") as f:
    f.write(r.stdout)
assert r.returncode == 0, "dos.x failed!"
print(f"      Total DOS saved to {PREFIX}_dos.dat")

# ══════════════════════════════════════════════════════════════════
#  Step 4: projwfc.x (projected DOS)
# ══════════════════════════════════════════════════════════════════
projwfc_input = f"""&PROJWFC
    prefix  = '{PREFIX}'
    outdir  = '{OUTDIR}'
    filpdos = '{PREFIX}_pdos'
    Emin    = {e_fermi - 15.0}
    Emax    = {e_fermi + 10.0}
    DeltaE  = 0.01
    ngauss  = 0
    degauss = 0.01
/
"""
with open(f"{PREFIX}_projwfc.in", "w") as f:
    f.write(projwfc_input)

print("[4/5] Running projwfc.x...")
r = subprocess.run(["projwfc.x", "-in", f"{PREFIX}_projwfc.in"],
                    capture_output=True, text=True, timeout=300)
with open(f"{PREFIX}_projwfc_out.log", "w") as f:
    f.write(r.stdout)
assert r.returncode == 0, "projwfc.x failed!"
print("      PDOS files generated.")

# ══════════════════════════════════════════════════════════════════
#  Step 5: Plot everything
# ══════════════════════════════════════════════════════════════════
print("[5/5] Plotting...")

# Parse total DOS
dos_data = np.loadtxt(f"{PREFIX}_dos.dat", comments="#")
energy_total = dos_data[:, 0]
dos_total = dos_data[:, 1]

# Parse PDOS files
pdos_files = sorted(glob.glob(f"{PREFIX}_pdos.pdos_atm#*"))
orbital_groups = {}  # {element-orbital: (energy, summed_pdos)}
element_groups = {}  # {element: (energy, summed_pdos)}

for fpath in pdos_files:
    fname = os.path.basename(fpath)
    m = re.search(r"atm#(\d+)\((\w+)\)_wfc#(\d+)\((\w+)\)", fname)
    if m:
        atom_idx = int(m.group(1))
        element = m.group(2)
        orbital = m.group(4)

        data = np.loadtxt(fpath, comments="#")
        energy = data[:, 0]
        pdos = data[:, 1]

        # Orbital-grouped
        key = f"{element}-{orbital}"
        if key not in orbital_groups:
            orbital_groups[key] = (energy, np.zeros_like(pdos))
        orbital_groups[key] = (energy, orbital_groups[key][1] + pdos)

        # Element-grouped
        if element not in element_groups:
            element_groups[element] = (energy, np.zeros_like(pdos))
        element_groups[element] = (energy, element_groups[element][1] + pdos)

# ── Combined figure: Total DOS + Orbital PDOS + Atom PDOS ──────────
n_panels = 1 + (1 if orbital_groups else 0) + (1 if len(element_groups) > 1 else 0)
fig, axes = plt.subplots(n_panels, 1, figsize=(8, 4 * n_panels), sharex=True,
                          gridspec_kw={"hspace": 0.05})
if n_panels == 1:
    axes = [axes]

panel = 0

# Panel 1: Total DOS
ax = axes[panel]
ax.plot(energy_total - e_fermi, dos_total, color="black", linewidth=1.2, label="Total DOS")
ax.fill_between(energy_total - e_fermi, dos_total, alpha=0.15, color="gray")
ax.axvline(0, color="red", linestyle="--", linewidth=0.8)
ax.set_ylabel("DOS\n(states/eV/cell)", fontsize=12)
ax.set_title("Density of States", fontsize=15)
ax.legend(fontsize=11)
ax.set_ylim(0, None)
ax.grid(alpha=0.3)
panel += 1

# Panel 2: Orbital-projected
orbital_colors = {"s": "#e74c3c", "p": "#3498db", "d": "#2ecc71", "f": "#9b59b6"}
if orbital_groups and panel < n_panels:
    ax = axes[panel]
    for key in sorted(orbital_groups.keys()):
        e, p = orbital_groups[key]
        element, orb = key.split("-")
        color = orbital_colors.get(orb, "gray")
        ax.plot(e - e_fermi, p, linewidth=1.5, label=f"{element} ({orb})", color=color)
        ax.fill_between(e - e_fermi, p, alpha=0.1, color=color)
    ax.axvline(0, color="red", linestyle="--", linewidth=0.8)
    ax.set_ylabel("PDOS\n(states/eV/cell)", fontsize=12)
    ax.legend(fontsize=10, ncol=2)
    ax.set_ylim(0, None)
    ax.grid(alpha=0.3)
    panel += 1

# Panel 3: Atom-projected (only for multi-element systems)
if len(element_groups) > 1 and panel < n_panels:
    ax = axes[panel]
    colors = plt.cm.Set1(np.linspace(0, 1, max(len(element_groups), 3)))
    for i, (element, (e, p)) in enumerate(sorted(element_groups.items())):
        ax.plot(e - e_fermi, p, linewidth=1.5, label=element, color=colors[i])
        ax.fill_between(e - e_fermi, p, alpha=0.1, color=colors[i])
    ax.axvline(0, color="red", linestyle="--", linewidth=0.8)
    ax.set_ylabel("PDOS\n(states/eV/cell)", fontsize=12)
    ax.legend(fontsize=11)
    ax.set_ylim(0, None)
    ax.grid(alpha=0.3)

axes[-1].set_xlabel("$E - E_F$ (eV)", fontsize=13)
axes[-1].set_xlim(-12, 8)

plt.savefig("dos_all.png", dpi=200, bbox_inches="tight")
print("      Saved: dos_all.png")

# ── Print summary ──────────────────────────────────────────────────
# DOS at Fermi level
idx_fermi = np.argmin(np.abs(energy_total - e_fermi))
dos_at_ef = dos_total[idx_fermi]
print(f"\n=== Results ===")
print(f"Fermi energy: {e_fermi:.4f} eV")
print(f"DOS at E_F: {dos_at_ef:.4f} states/eV/cell")
if dos_at_ef < 0.01:
    print("Material is likely a semiconductor/insulator (DOS ~ 0 at E_F).")
else:
    print("Material is likely a metal (finite DOS at E_F).")
print(f"\nGenerated files:")
print(f"  {PREFIX}_dos.dat         -- total DOS data")
print(f"  {PREFIX}_pdos.*          -- projected DOS data files")
print(f"  dos_all.png              -- combined plot")
```

---

## Key Parameters

| Parameter | Typical Value | Notes |
|---|---|---|
| SCF `K_POINTS` | 8x8x8 | Standard grid for the SCF step |
| NSCF `K_POINTS` | 16x16x16 to 24x24x24 | Dense grid for smooth DOS. Denser = smoother but slower. |
| `occupations` (NSCF) | `'tetrahedra'` | Best for DOS integration. Use `'tetrahedra_lin'` for metals. |
| `nbnd` | 1.5x--2x occupied | Include empty bands to see conduction band DOS |
| `DeltaE` (dos.x) | 0.01--0.05 eV | Energy grid spacing for DOS output |
| `Emin`, `Emax` | E_F - 15, E_F + 10 | Energy window. Adjust to your needs. |
| `degauss` (projwfc.x) | 0.01--0.05 Ry | Gaussian broadening for PDOS. Smaller = sharper features. |
| `ngauss` (projwfc.x) | 0 | 0 = simple Gaussian, -1 = Fermi-Dirac, 1 = Methfessel-Paxton |

## Interpreting Results

- **DOS at Fermi level**: If DOS(E_F) > 0, the material is metallic. If DOS(E_F) = 0, it is a semiconductor or insulator.
- **Band gap from DOS**: The energy range where DOS = 0 around the Fermi level gives the band gap (though less precise than from band structure).
- **Orbital character**: The PDOS tells you which orbitals dominate at each energy:
  - Transition metal d-states typically appear as sharp peaks near E_F.
  - Oxygen/nitrogen p-states typically dominate the valence band in oxides/nitrides.
  - s-states are usually at the bottom of the valence band.
- **Crystal field splitting**: For transition metals, compare t2g vs. eg peak positions in the d-orbital PDOS.
- **Hybridization**: Overlapping peaks from different elements at the same energy indicate orbital hybridization (bonding).
- **Integrated DOS**: The integrated DOS at E_F should equal the number of electrons.

## Common Issues

| Problem | Solution |
|---|---|
| DOS is too spiky/noisy | Use a denser NSCF k-grid (e.g., 20x20x20 or more). Increase `degauss` slightly. |
| DOS is too smooth, features washed out | Decrease `degauss`. Use tetrahedron method instead of smearing. |
| projwfc.x crashes | Ensure NSCF completed successfully. Check that `outdir` and `prefix` match. Verify UPF files support projection (some simplified pseudos do not). |
| Missing PDOS for some orbitals | The pseudopotential may not include that orbital as a projector. Check the UPF file header. |
| Negative DOS values | Can happen with tetrahedron method at sharp features. Use a small Gaussian broadening or denser k-grid. |
| `occupations = 'tetrahedra'` fails | Tetrahedra require a uniform (automatic) k-grid with no offset. Use `0 0 0` for the k-grid shift. |
| NSCF reads wrong charge density | Ensure `prefix` and `outdir` are identical between SCF and NSCF. Do not delete `outdir/prefix.save/` between steps. |
| DOS not aligned to Fermi level | Always shift by E_F from the NSCF (not SCF) output. For insulators, align to VBM instead. |
