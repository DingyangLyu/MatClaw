# Systematic Convergence Testing for DFT Calculations

## When to Use

- You are starting DFT calculations for a new material and need to determine optimal computational parameters.
- You need to converge the plane-wave cutoff energy (ecutwfc in QE, ENCUT in VASP).
- You need to converge the k-point grid density.
- You need to converge the smearing parameter (degauss/SIGMA) for metallic systems.
- You want to document convergence for a publication (required for credible results).
- You need to balance accuracy against computational cost for a large-scale study.

## Method Selection

| Parameter | QE | VASP | What It Controls |
|---|---|---|---|
| Cutoff energy | `ecutwfc` (Ry) | `ENCUT` (eV) | Completeness of plane-wave basis. Affects total energy, forces, stress. |
| Charge density cutoff | `ecutrho` (Ry) | Automatic | QE only: typically 4x (NC) or 8--12x (US/PAW) of ecutwfc. |
| K-point grid | `K_POINTS (automatic) Nx Ny Nz` | `KPOINTS` file | Brillouin zone sampling. Affects total energy, DOS smoothness. |
| Smearing width | `degauss` (Ry) | `SIGMA` (eV) | Electronic temperature. Affects total energy for metals. |
| SCF convergence | `conv_thr` (Ry) | `EDIFF` (eV) | Self-consistency convergence threshold. |
| Force convergence | `forc_conv_thr` (Ry/au) | `EDIFFG` (eV/Ang) | Relaxation convergence. Usually not a convergence "test" but must be set consistently. |

## Prerequisites

- A crystal structure (CIF, POSCAR, or built programmatically).
- **QE**: `pw.x` executable, pseudopotential files.
- **VASP**: VASP executable, POTCAR files.
- Python packages: `numpy`, `matplotlib`, `pymatgen`, `ase`.

---

## Detailed Steps

### Method A: QE Convergence Testing

#### Step A1: Cutoff Energy Convergence

```python
#!/usr/bin/env python3
"""
Systematic ecutwfc convergence test for Quantum ESPRESSO.
Runs a series of SCF calculations with increasing cutoff energy,
keeping k-grid fixed. Plots energy vs. cutoff and determines
the optimal cutoff within a specified accuracy threshold.
"""
import os
import subprocess
import re
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ══════════════════════════════════════════════════════════════════
#  Configuration
# ══════════════════════════════════════════════════════════════════
PSEUDO_DIR = os.path.abspath("./pseudo")
OUTDIR_BASE = os.path.abspath("./conv_ecutwfc")
os.makedirs(OUTDIR_BASE, exist_ok=True)
PREFIX = "si"
NPROC = 4

# Convergence criterion: energy difference per atom (meV/atom)
THRESHOLD_MEV = 1.0
NAT = 2  # Number of atoms in the unit cell

# K-grid: use a well-converged k-grid so we only test ecutwfc
KGRID = "8 8 8"

# Range of ecutwfc values to test (in Ry)
ECUTWFC_VALUES = [20, 25, 30, 35, 40, 45, 50, 55, 60, 70, 80, 90, 100]

# Ratio of ecutrho to ecutwfc
# 4 for norm-conserving, 8-12 for ultrasoft/PAW
ECUTRHO_RATIO = 8

# Structure definition (silicon diamond)
STRUCTURE_BLOCK = """\
&SYSTEM
    ibrav       = 2
    celldm(1)   = 10.26
    nat         = 2
    ntyp        = 1
    ecutwfc     = {ecutwfc}
    ecutrho     = {ecutrho}
    occupations = 'smearing'
    smearing    = 'cold'
    degauss     = 0.01
/

ATOMIC_SPECIES
  Si  28.0855  Si.pbe-n-rrkjus_psl.1.0.0.UPF

ATOMIC_POSITIONS (crystal)
  Si  0.00  0.00  0.00
  Si  0.25  0.25  0.25

K_POINTS (automatic)
  {kgrid}  0 0 0
"""

# ══════════════════════════════════════════════════════════════════
#  Run calculations
# ══════════════════════════════════════════════════════════════════
def run_qe_scf(ecutwfc, kgrid, tag, prefix, pseudo_dir, outdir_base, nproc,
               ecutrho_ratio, structure_block):
    """Run a single QE SCF and return total energy in eV."""
    outdir = os.path.join(outdir_base, tag)
    os.makedirs(outdir, exist_ok=True)
    ecutrho = ecutwfc * ecutrho_ratio

    inp = f"""&CONTROL
    calculation = 'scf'
    prefix      = '{prefix}'
    outdir      = '{outdir}'
    pseudo_dir  = '{pseudo_dir}'
    tprnfor     = .true.
    tstress     = .true.
/
{structure_block.format(ecutwfc=ecutwfc, ecutrho=ecutrho, kgrid=kgrid)}
"""
    infile = os.path.join(outdir_base, f"scf_{tag}.in")
    outfile = os.path.join(outdir_base, f"scf_{tag}.out")

    with open(infile, "w") as f:
        f.write(inp)

    result = subprocess.run(
        ["mpirun", "-np", str(nproc), "pw.x", "-in", infile],
        capture_output=True, text=True, timeout=600
    )
    with open(outfile, "w") as f:
        f.write(result.stdout)

    # Parse total energy (eV), forces, and timing
    energy_ry = None
    max_force = None
    wall_time = None

    for line in result.stdout.split("\n"):
        if line.strip().startswith("!") and "total energy" in line:
            m = re.search(r"=\s+([-\d.]+)\s+Ry", line)
            if m:
                energy_ry = float(m.group(1))
        if "Total force" in line:
            m = re.search(r"Total force\s*=\s*([\d.]+)", line)
            if m:
                max_force = float(m.group(1))
        if "WALL" in line:
            m = re.search(r"([\d.]+)s WALL", line)
            if m:
                wall_time = float(m.group(1))

    energy_ev = energy_ry * 13.605693123 if energy_ry else None
    return {
        "ecutwfc": ecutwfc,
        "energy_Ry": energy_ry,
        "energy_eV": energy_ev,
        "total_force_Ry_au": max_force,
        "wall_time_s": wall_time,
        "converged": "convergence has been achieved" in result.stdout,
    }

print("=" * 60)
print("  ecutwfc Convergence Test")
print("=" * 60)

results = []
for ec in ECUTWFC_VALUES:
    tag = f"ecut_{ec:03d}"
    print(f"  ecutwfc = {ec:3d} Ry  ...", end="", flush=True)
    r = run_qe_scf(ec, KGRID, tag, PREFIX, PSEUDO_DIR, OUTDIR_BASE, NPROC,
                   ECUTRHO_RATIO, STRUCTURE_BLOCK)
    results.append(r)
    if r["energy_eV"] is not None:
        print(f"  E = {r['energy_eV']:.6f} eV  [{'OK' if r['converged'] else 'WARN'}]")
    else:
        print("  FAILED")

# Save results
with open(os.path.join(OUTDIR_BASE, "ecutwfc_results.json"), "w") as f:
    json.dump(results, f, indent=2)

# ══════════════════════════════════════════════════════════════════
#  Analysis and plotting
# ══════════════════════════════════════════════════════════════════
valid = [r for r in results if r["energy_eV"] is not None]
if not valid:
    print("ERROR: No calculations succeeded!")
    exit(1)

ecutwfc_arr = np.array([r["ecutwfc"] for r in valid])
energy_arr = np.array([r["energy_eV"] for r in valid])

# Reference: highest cutoff
e_ref = energy_arr[-1]
de_meV_per_atom = (energy_arr - e_ref) * 1000.0 / NAT

# Determine optimal cutoff
optimal_ecut = None
for i, de in enumerate(de_meV_per_atom):
    if abs(de) <= THRESHOLD_MEV:
        optimal_ecut = ecutwfc_arr[i]
        break

print(f"\n{'='*60}")
print(f"  Results (reference: ecutwfc = {ecutwfc_arr[-1]} Ry)")
print(f"{'='*60}")
print(f"  {'ecutwfc (Ry)':>12s}  {'E (eV)':>14s}  {'dE (meV/atom)':>14s}")
print(f"  {'-'*12}  {'-'*14}  {'-'*14}")
for ec, e, de in zip(ecutwfc_arr, energy_arr, de_meV_per_atom):
    marker = " <-- optimal" if optimal_ecut and ec == optimal_ecut else ""
    print(f"  {ec:12.0f}  {e:14.6f}  {de:14.4f}{marker}")

if optimal_ecut:
    print(f"\nOptimal ecutwfc = {optimal_ecut} Ry (converged to {THRESHOLD_MEV} meV/atom)")
else:
    print(f"\nWARNING: Not converged to {THRESHOLD_MEV} meV/atom within tested range!")
    print("Extend ECUTWFC_VALUES to higher values.")

# ── Plot ───────────────────────────────────────────────────────────
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

# Left: absolute energy vs cutoff
ax1.plot(ecutwfc_arr, energy_arr, "o-", color="steelblue", linewidth=2, markersize=8)
ax1.set_xlabel("ecutwfc (Ry)", fontsize=13)
ax1.set_ylabel("Total energy (eV)", fontsize=13)
ax1.set_title("Total Energy vs. Cutoff", fontsize=14)
ax1.grid(True, alpha=0.3)

# Right: energy difference per atom
ax2.plot(ecutwfc_arr, np.abs(de_meV_per_atom), "s-", color="darkorange",
         linewidth=2, markersize=8)
ax2.axhline(THRESHOLD_MEV, color="red", linestyle="--", linewidth=1.5,
            label=f"Threshold: {THRESHOLD_MEV} meV/atom")
ax2.set_xlabel("ecutwfc (Ry)", fontsize=13)
ax2.set_ylabel("|dE| (meV/atom)", fontsize=13)
ax2.set_title("Convergence: Energy Difference", fontsize=14)
ax2.set_yscale("log")
ax2.legend(fontsize=11)
ax2.grid(True, alpha=0.3, which="both")

if optimal_ecut:
    ax2.axvline(optimal_ecut, color="green", linestyle=":", linewidth=1.5,
                label=f"Optimal: {optimal_ecut} Ry")
    ax2.legend(fontsize=11)

plt.tight_layout()
plt.savefig(os.path.join(OUTDIR_BASE, "ecutwfc_convergence.png"),
            dpi=200, bbox_inches="tight")
print(f"\nPlot saved: {OUTDIR_BASE}/ecutwfc_convergence.png")
```

#### Step A2: K-Point Grid Convergence

```python
#!/usr/bin/env python3
"""
Systematic k-point grid convergence test for Quantum ESPRESSO.
Runs SCF with increasing k-grid density, fixed ecutwfc.
"""
import os
import subprocess
import re
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ══════════════════════════════════════════════════════════════════
#  Configuration
# ══════════════════════════════════════════════════════════════════
PSEUDO_DIR = os.path.abspath("./pseudo")
OUTDIR_BASE = os.path.abspath("./conv_kgrid")
os.makedirs(OUTDIR_BASE, exist_ok=True)
PREFIX = "si"
NPROC = 4
NAT = 2

THRESHOLD_MEV = 1.0

# Use the converged ecutwfc from Step A1
ECUTWFC = 50.0
ECUTRHO = ECUTWFC * 8

# K-grid values to test (NxNxN)
# For non-cubic cells, test each direction independently
KGRID_VALUES = [2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 14, 16]

def run_qe_scf_kgrid(kgrid_n, ecutwfc, ecutrho, tag):
    """Run single SCF with given k-grid."""
    outdir = os.path.join(OUTDIR_BASE, tag)
    os.makedirs(outdir, exist_ok=True)

    inp = f"""&CONTROL
    calculation = 'scf'
    prefix      = '{PREFIX}'
    outdir      = '{outdir}'
    pseudo_dir  = '{PSEUDO_DIR}'
/
&SYSTEM
    ibrav       = 2
    celldm(1)   = 10.26
    nat         = {NAT}
    ntyp        = 1
    ecutwfc     = {ecutwfc}
    ecutrho     = {ecutrho}
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
  Si  0.00 0.00 0.00
  Si  0.25 0.25 0.25
K_POINTS (automatic)
  {kgrid_n} {kgrid_n} {kgrid_n}  0 0 0
"""
    infile = os.path.join(OUTDIR_BASE, f"scf_{tag}.in")
    outfile = os.path.join(OUTDIR_BASE, f"scf_{tag}.out")

    with open(infile, "w") as f:
        f.write(inp)

    result = subprocess.run(
        ["mpirun", "-np", str(NPROC), "pw.x", "-in", infile],
        capture_output=True, text=True, timeout=600
    )
    with open(outfile, "w") as f:
        f.write(result.stdout)

    energy_ry = None
    n_kpts_irr = None
    for line in result.stdout.split("\n"):
        if line.strip().startswith("!") and "total energy" in line:
            m = re.search(r"=\s+([-\d.]+)\s+Ry", line)
            if m: energy_ry = float(m.group(1))
        if "number of k points=" in line:
            m = re.search(r"=\s+(\d+)", line)
            if m: n_kpts_irr = int(m.group(1))

    energy_ev = energy_ry * 13.605693123 if energy_ry else None
    return {
        "kgrid": kgrid_n,
        "n_kpts_irr": n_kpts_irr,
        "energy_eV": energy_ev,
        "converged": "convergence has been achieved" in result.stdout,
    }

print("=" * 60)
print("  K-point Grid Convergence Test")
print(f"  ecutwfc = {ECUTWFC} Ry (fixed)")
print("=" * 60)

results = []
for kg in KGRID_VALUES:
    tag = f"kgrid_{kg:02d}"
    print(f"  k-grid = {kg:2d}x{kg}x{kg}  ...", end="", flush=True)
    r = run_qe_scf_kgrid(kg, ECUTWFC, ECUTRHO, tag)
    results.append(r)
    if r["energy_eV"] is not None:
        print(f"  E = {r['energy_eV']:.6f} eV  ({r['n_kpts_irr']} irr. k-pts)")
    else:
        print("  FAILED")

# Save results
with open(os.path.join(OUTDIR_BASE, "kgrid_results.json"), "w") as f:
    json.dump(results, f, indent=2)

# Analysis
valid = [r for r in results if r["energy_eV"] is not None]
kgrid_arr = np.array([r["kgrid"] for r in valid])
energy_arr = np.array([r["energy_eV"] for r in valid])
nkpts_arr = np.array([r["n_kpts_irr"] for r in valid])

e_ref = energy_arr[-1]
de_meV = (energy_arr - e_ref) * 1000.0 / NAT

optimal_kgrid = None
for i, de in enumerate(de_meV):
    if abs(de) <= THRESHOLD_MEV:
        optimal_kgrid = kgrid_arr[i]
        break

print(f"\n{'='*60}")
print(f"  Results (reference: k-grid = {kgrid_arr[-1]}x{kgrid_arr[-1]}x{kgrid_arr[-1]})")
print(f"{'='*60}")
print(f"  {'k-grid':>8s}  {'irr. k-pts':>10s}  {'E (eV)':>14s}  {'dE (meV/atom)':>14s}")
for kg, nk, e, de in zip(kgrid_arr, nkpts_arr, energy_arr, de_meV):
    marker = " <--" if optimal_kgrid and kg == optimal_kgrid else ""
    print(f"  {kg:5.0f}^3    {nk:10d}  {e:14.6f}  {de:14.4f}{marker}")

if optimal_kgrid:
    print(f"\nOptimal k-grid = {optimal_kgrid}x{optimal_kgrid}x{optimal_kgrid} "
          f"(converged to {THRESHOLD_MEV} meV/atom)")

# Plot
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

ax1.plot(kgrid_arr, energy_arr, "o-", color="steelblue", linewidth=2, markersize=8)
ax1.set_xlabel("k-grid (NxNxN)", fontsize=13)
ax1.set_ylabel("Total energy (eV)", fontsize=13)
ax1.set_title("Total Energy vs. k-grid", fontsize=14)
ax1.grid(True, alpha=0.3)

ax2.plot(kgrid_arr, np.abs(de_meV), "s-", color="darkorange", linewidth=2, markersize=8)
ax2.axhline(THRESHOLD_MEV, color="red", linestyle="--", linewidth=1.5,
            label=f"Threshold: {THRESHOLD_MEV} meV/atom")
if optimal_kgrid:
    ax2.axvline(optimal_kgrid, color="green", linestyle=":", linewidth=1.5,
                label=f"Optimal: {optimal_kgrid}^3")
ax2.set_xlabel("k-grid (NxNxN)", fontsize=13)
ax2.set_ylabel("|dE| (meV/atom)", fontsize=13)
ax2.set_title("Convergence: Energy Difference", fontsize=14)
ax2.set_yscale("log")
ax2.legend(fontsize=11)
ax2.grid(True, alpha=0.3, which="both")

plt.tight_layout()
plt.savefig(os.path.join(OUTDIR_BASE, "kgrid_convergence.png"),
            dpi=200, bbox_inches="tight")
print(f"\nPlot saved: {OUTDIR_BASE}/kgrid_convergence.png")
```

#### Step A3: Smearing Convergence (for Metals)

```python
#!/usr/bin/env python3
"""
Smearing parameter (degauss) convergence test for metallic systems.
For semiconductors/insulators, smearing should have negligible effect;
this test is primarily for metals.
"""
import os
import subprocess
import re
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

PSEUDO_DIR = os.path.abspath("./pseudo")
OUTDIR_BASE = os.path.abspath("./conv_smearing")
os.makedirs(OUTDIR_BASE, exist_ok=True)
PREFIX = "al"
NPROC = 4
NAT = 1  # FCC Al, 1 atom in primitive cell

ECUTWFC = 40.0
ECUTRHO = ECUTWFC * 8
KGRID = "12 12 12"  # Well-converged k-grid

# Smearing values to test (in Ry)
# Also test different smearing types
DEGAUSS_VALUES = [0.001, 0.002, 0.005, 0.01, 0.02, 0.03, 0.05, 0.07, 0.10]
SMEARING_TYPES = {
    "cold": "Marzari-Vanderbilt cold smearing",
    "mp": "Methfessel-Paxton order 1",
    "gauss": "Simple Gaussian",
}

def run_scf_smearing(degauss, smearing_type, tag):
    """Run SCF with given smearing parameters."""
    outdir = os.path.join(OUTDIR_BASE, tag)
    os.makedirs(outdir, exist_ok=True)

    # Map smearing type to QE keyword
    smearing_qe = {"cold": "'cold'", "mp": "'mp'", "gauss": "'gauss'"}

    inp = f"""&CONTROL
    calculation = 'scf'
    prefix      = '{PREFIX}'
    outdir      = '{outdir}'
    pseudo_dir  = '{PSEUDO_DIR}'
/
&SYSTEM
    ibrav       = 2
    celldm(1)   = 7.63
    nat         = {NAT}
    ntyp        = 1
    ecutwfc     = {ECUTWFC}
    ecutrho     = {ECUTRHO}
    occupations = 'smearing'
    smearing    = {smearing_qe[smearing_type]}
    degauss     = {degauss}
/
&ELECTRONS
    conv_thr = 1.0d-10
/
ATOMIC_SPECIES
  Al  26.9815  Al.pbe-n-rrkjus_psl.1.0.0.UPF
ATOMIC_POSITIONS (crystal)
  Al  0.00 0.00 0.00
K_POINTS (automatic)
  {KGRID}  0 0 0
"""
    infile = os.path.join(OUTDIR_BASE, f"scf_{tag}.in")
    outfile = os.path.join(OUTDIR_BASE, f"scf_{tag}.out")

    with open(infile, "w") as f:
        f.write(inp)

    result = subprocess.run(
        ["mpirun", "-np", str(NPROC), "pw.x", "-in", infile],
        capture_output=True, text=True, timeout=600
    )
    with open(outfile, "w") as f:
        f.write(result.stdout)

    energy_ry = None
    for line in result.stdout.split("\n"):
        if line.strip().startswith("!") and "total energy" in line:
            m = re.search(r"=\s+([-\d.]+)\s+Ry", line)
            if m: energy_ry = float(m.group(1))

    energy_ev = energy_ry * 13.605693123 if energy_ry else None
    return {"degauss": degauss, "smearing": smearing_type,
            "energy_eV": energy_ev,
            "converged": "convergence has been achieved" in result.stdout}

print("=" * 60)
print("  Smearing Convergence Test")
print("=" * 60)

all_results = {}
for stype, sdesc in SMEARING_TYPES.items():
    print(f"\n  --- {sdesc} ({stype}) ---")
    results = []
    for dg in DEGAUSS_VALUES:
        tag = f"{stype}_dg{dg:.4f}"
        print(f"    degauss = {dg:.4f} Ry  ...", end="", flush=True)
        r = run_scf_smearing(dg, stype, tag)
        results.append(r)
        if r["energy_eV"] is not None:
            print(f"  E = {r['energy_eV']:.6f} eV")
        else:
            print("  FAILED")
    all_results[stype] = results

# Save
with open(os.path.join(OUTDIR_BASE, "smearing_results.json"), "w") as f:
    json.dump(all_results, f, indent=2)

# Plot
fig, ax = plt.subplots(figsize=(8, 6))
colors = {"cold": "steelblue", "mp": "darkorange", "gauss": "green"}

for stype, results in all_results.items():
    valid = [r for r in results if r["energy_eV"] is not None]
    if not valid:
        continue
    dg = np.array([r["degauss"] for r in valid])
    en = np.array([r["energy_eV"] for r in valid])
    # Plot relative to smallest degauss result
    e_ref = en[0]  # smallest degauss is the most accurate
    de = (en - e_ref) * 1000.0 / NAT
    ax.plot(dg * 13605.693,  # Convert Ry to meV for x-axis
            np.abs(de), "o-", color=colors[stype], linewidth=2,
            markersize=8, label=f"{stype}")

ax.axhline(1.0, color="red", linestyle="--", linewidth=1.5, label="1 meV/atom")
ax.set_xlabel("degauss (meV)", fontsize=13)
ax.set_ylabel("|dE| relative to smallest degauss (meV/atom)", fontsize=13)
ax.set_title("Smearing Convergence (Metal)", fontsize=14)
ax.set_xscale("log")
ax.set_yscale("log")
ax.legend(fontsize=11)
ax.grid(True, alpha=0.3, which="both")

plt.tight_layout()
plt.savefig(os.path.join(OUTDIR_BASE, "smearing_convergence.png"),
            dpi=200, bbox_inches="tight")
print(f"\nPlot saved: {OUTDIR_BASE}/smearing_convergence.png")
```

### Method B: VASP Convergence Testing

#### Step B1: ENCUT Convergence

```python
#!/usr/bin/env python3
"""
Systematic ENCUT convergence test for VASP.
Generates and runs VASP calculations with varying ENCUT.
Parses OSZICAR for total energy.
"""
import os
import subprocess
import re
import json
import shutil
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pymatgen.core import Structure
from pymatgen.io.vasp.inputs import Incar, Kpoints, Poscar, Potcar
from pymatgen.io.vasp.sets import MPStaticSet

# ══════════════════════════════════════════════════════════════════
#  Configuration
# ══════════════════════════════════════════════════════════════════
STRUCTURE_FILE = "POSCAR"  # Input structure
CONV_DIR = os.path.abspath("./conv_encut")
os.makedirs(CONV_DIR, exist_ok=True)
NAT = None  # Will be determined from structure

THRESHOLD_MEV = 1.0

# ENCUT values to test (eV)
ENCUT_VALUES = [200, 250, 300, 350, 400, 450, 500, 550, 600, 650, 700]

# Fixed k-grid (well-converged)
KPOINTS_GRID = (8, 8, 8)

# ── Load structure ─────────────────────────────────────────────────
struct = Structure.from_file(STRUCTURE_FILE)
NAT = struct.num_sites
print(f"Structure: {struct.formula}, {NAT} atoms")

# ══════════════════════════════════════════════════════════════════
#  Generate and run calculations
# ══════════════════════════════════════════════════════════════════
def parse_vasp_energy(directory):
    """Parse total energy from VASP OSZICAR or OUTCAR."""
    # Try OSZICAR first (faster)
    oszicar = os.path.join(directory, "OSZICAR")
    if os.path.exists(oszicar):
        with open(oszicar, "r") as f:
            lines = f.readlines()
        for line in reversed(lines):
            if "F=" in line or "E0=" in line:
                m = re.search(r"E0=\s*([-\d.E+]+)", line)
                if m:
                    return float(m.group(1))

    # Fallback: OUTCAR
    outcar = os.path.join(directory, "OUTCAR")
    if os.path.exists(outcar):
        with open(outcar, "r") as f:
            for line in f:
                if "free  energy   TOTEN" in line:
                    m = re.search(r"=\s*([-\d.]+)", line)
                    if m:
                        energy = float(m.group(1))
        return energy  # Last occurrence is the converged value

    return None

results = []
print(f"\n{'='*60}")
print(f"  ENCUT Convergence Test")
print(f"{'='*60}")

for encut in ENCUT_VALUES:
    calc_dir = os.path.join(CONV_DIR, f"encut_{encut:04d}")
    os.makedirs(calc_dir, exist_ok=True)

    print(f"  ENCUT = {encut:4d} eV  ...", end="", flush=True)

    # Generate inputs using pymatgen
    vasp_set = MPStaticSet(struct, user_incar_settings={
        "ENCUT": encut,
        "EDIFF": 1e-6,
        "ISMEAR": -5,
        "LWAVE": False,
        "LCHARG": False,
        "PREC": "Accurate",
    }, user_kpoints_settings={"grid_density": None,
                              "reciprocal_density": None})

    # Write input files
    vasp_set.write_input(calc_dir)

    # Override KPOINTS with our fixed grid
    kpts = Kpoints.gamma_automatic(kpts=KPOINTS_GRID)
    kpts.write_file(os.path.join(calc_dir, "KPOINTS"))

    # Run VASP
    result = subprocess.run(
        ["mpirun", "-np", "4", "vasp_std"],
        capture_output=True, text=True, timeout=1800,
        cwd=calc_dir
    )

    energy = parse_vasp_energy(calc_dir)
    if energy is not None:
        print(f"  E = {energy:.6f} eV")
    else:
        print("  FAILED")

    results.append({
        "encut": encut,
        "energy_eV": energy,
    })

# Save results
with open(os.path.join(CONV_DIR, "encut_results.json"), "w") as f:
    json.dump(results, f, indent=2)

# ── Analysis ───────────────────────────────────────────────────────
valid = [r for r in results if r["energy_eV"] is not None]
encut_arr = np.array([r["encut"] for r in valid])
energy_arr = np.array([r["energy_eV"] for r in valid])

e_ref = energy_arr[-1]
de_meV = (energy_arr - e_ref) * 1000.0 / NAT

optimal = None
for i, de in enumerate(de_meV):
    if abs(de) <= THRESHOLD_MEV:
        optimal = encut_arr[i]
        break

print(f"\n{'='*60}")
print(f"  {'ENCUT (eV)':>10s}  {'E (eV)':>14s}  {'dE (meV/atom)':>14s}")
for ec, e, de in zip(encut_arr, energy_arr, de_meV):
    marker = " <--" if optimal and ec == optimal else ""
    print(f"  {ec:10.0f}  {e:14.6f}  {de:14.4f}{marker}")

if optimal:
    print(f"\nOptimal ENCUT = {optimal} eV (converged to {THRESHOLD_MEV} meV/atom)")

# ── Plot ───────────────────────────────────────────────────────────
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

ax1.plot(encut_arr, energy_arr, "o-", color="steelblue", linewidth=2, markersize=8)
ax1.set_xlabel("ENCUT (eV)", fontsize=13)
ax1.set_ylabel("Total energy (eV)", fontsize=13)
ax1.set_title("Total Energy vs. ENCUT", fontsize=14)
ax1.grid(True, alpha=0.3)

ax2.plot(encut_arr, np.abs(de_meV), "s-", color="darkorange", linewidth=2, markersize=8)
ax2.axhline(THRESHOLD_MEV, color="red", linestyle="--", linewidth=1.5,
            label=f"{THRESHOLD_MEV} meV/atom")
if optimal:
    ax2.axvline(optimal, color="green", linestyle=":", linewidth=1.5,
                label=f"Optimal: {optimal} eV")
ax2.set_xlabel("ENCUT (eV)", fontsize=13)
ax2.set_ylabel("|dE| (meV/atom)", fontsize=13)
ax2.set_title("Convergence: Energy Difference", fontsize=14)
ax2.set_yscale("log")
ax2.legend(fontsize=11)
ax2.grid(True, alpha=0.3, which="both")

plt.tight_layout()
plt.savefig(os.path.join(CONV_DIR, "encut_convergence.png"),
            dpi=200, bbox_inches="tight")
print(f"\nPlot saved: {CONV_DIR}/encut_convergence.png")
```

#### Step B2: VASP K-Point Convergence

```python
#!/usr/bin/env python3
"""
VASP k-point convergence test.
Similar to QE version but uses pymatgen for VASP input generation.
"""
import os
import subprocess
import re
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pymatgen.core import Structure
from pymatgen.io.vasp.sets import MPStaticSet
from pymatgen.io.vasp.inputs import Kpoints

STRUCTURE_FILE = "POSCAR"
CONV_DIR = os.path.abspath("./conv_kpoints")
os.makedirs(CONV_DIR, exist_ok=True)

struct = Structure.from_file(STRUCTURE_FILE)
NAT = struct.num_sites
THRESHOLD_MEV = 1.0

# Fixed ENCUT (from convergence test)
ENCUT = 520  # eV

# K-grid values (NxNxN for cubic; adjust ratios for non-cubic)
KGRID_VALUES = [2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 14, 16]

def parse_vasp_energy(directory):
    """Parse VASP total energy from OSZICAR."""
    oszicar = os.path.join(directory, "OSZICAR")
    if os.path.exists(oszicar):
        with open(oszicar, "r") as f:
            lines = f.readlines()
        for line in reversed(lines):
            if "E0=" in line:
                m = re.search(r"E0=\s*([-\d.E+]+)", line)
                if m:
                    return float(m.group(1))
    return None

print(f"{'='*60}")
print(f"  VASP K-Point Convergence Test")
print(f"  ENCUT = {ENCUT} eV (fixed)")
print(f"{'='*60}")

results = []
for kg in KGRID_VALUES:
    calc_dir = os.path.join(CONV_DIR, f"kgrid_{kg:02d}")
    os.makedirs(calc_dir, exist_ok=True)

    print(f"  k-grid = {kg:2d}x{kg}x{kg}  ...", end="", flush=True)

    vasp_set = MPStaticSet(struct, user_incar_settings={
        "ENCUT": ENCUT,
        "EDIFF": 1e-6,
        "ISMEAR": -5,
        "LWAVE": False,
        "LCHARG": False,
    })
    vasp_set.write_input(calc_dir)

    kpts = Kpoints.gamma_automatic(kpts=(kg, kg, kg))
    kpts.write_file(os.path.join(calc_dir, "KPOINTS"))

    subprocess.run(
        ["mpirun", "-np", "4", "vasp_std"],
        capture_output=True, text=True, timeout=1800,
        cwd=calc_dir
    )

    energy = parse_vasp_energy(calc_dir)
    if energy is not None:
        print(f"  E = {energy:.6f} eV")
    else:
        print("  FAILED")

    results.append({"kgrid": kg, "energy_eV": energy})

# Save
with open(os.path.join(CONV_DIR, "kgrid_results.json"), "w") as f:
    json.dump(results, f, indent=2)

# Analysis and plot
valid = [r for r in results if r["energy_eV"] is not None]
kgrid_arr = np.array([r["kgrid"] for r in valid])
energy_arr = np.array([r["energy_eV"] for r in valid])
e_ref = energy_arr[-1]
de_meV = (energy_arr - e_ref) * 1000.0 / NAT

optimal = None
for i, de in enumerate(de_meV):
    if abs(de) <= THRESHOLD_MEV:
        optimal = kgrid_arr[i]
        break

print(f"\n{'='*60}")
for kg, e, de in zip(kgrid_arr, energy_arr, de_meV):
    marker = " <--" if optimal and kg == optimal else ""
    print(f"  {kg:3.0f}^3  {e:14.6f} eV  {de:10.4f} meV/atom{marker}")
if optimal:
    print(f"\nOptimal: {optimal}x{optimal}x{optimal}")

fig, ax = plt.subplots(figsize=(8, 5))
ax.plot(kgrid_arr, np.abs(de_meV), "s-", color="darkorange", linewidth=2, markersize=8)
ax.axhline(THRESHOLD_MEV, color="red", linestyle="--", linewidth=1.5,
           label=f"{THRESHOLD_MEV} meV/atom")
if optimal:
    ax.axvline(optimal, color="green", linestyle=":", linewidth=1.5,
               label=f"Optimal: {optimal}^3")
ax.set_xlabel("k-grid (NxNxN)", fontsize=13)
ax.set_ylabel("|dE| (meV/atom)", fontsize=13)
ax.set_title("VASP K-Point Convergence", fontsize=14)
ax.set_yscale("log")
ax.legend(fontsize=11)
ax.grid(True, alpha=0.3, which="both")
plt.tight_layout()
plt.savefig(os.path.join(CONV_DIR, "vasp_kgrid_convergence.png"),
            dpi=200, bbox_inches="tight")
print(f"\nPlot saved: {CONV_DIR}/vasp_kgrid_convergence.png")
```

### Method C: Automated Convergence Workflow (Both Codes)

```python
#!/usr/bin/env python3
"""
Automated convergence testing framework.
Runs cutoff and k-grid convergence sequentially, using the converged
cutoff from the first test in the second test.
Supports both QE and VASP via a unified interface.
Generates a summary report and combined convergence plot.
"""
import os
import subprocess
import re
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ══════════════════════════════════════════════════════════════════
#  Configuration
# ══════════════════════════════════════════════════════════════════
CODE = "qe"  # "qe" or "vasp"
THRESHOLD_MEV = 1.0  # meV/atom

# QE-specific settings
QE_CONFIG = {
    "pseudo_dir": os.path.abspath("./pseudo"),
    "nproc": 4,
    "ecutwfc_values": [20, 30, 40, 50, 60, 70, 80, 90, 100],
    "ecutrho_ratio": 8,  # ecutrho = ratio * ecutwfc
    "kgrid_values": [2, 4, 6, 8, 10, 12, 14],
    "default_kgrid": "8 8 8",
    "default_ecutwfc": 50,
    # Template for QE input (will be formatted with ecutwfc, ecutrho, kgrid)
    "template": """\
&CONTROL
    calculation = 'scf'
    prefix      = 'conv'
    outdir      = '{outdir}'
    pseudo_dir  = '{pseudo_dir}'
/
&SYSTEM
    ibrav       = 2
    celldm(1)   = 10.26
    nat         = 2
    ntyp        = 1
    ecutwfc     = {ecutwfc}
    ecutrho     = {ecutrho}
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
  Si  0.00 0.00 0.00
  Si  0.25 0.25 0.25
K_POINTS (automatic)
  {kgrid}  0 0 0
""",
    "nat": 2,
}

CONV_DIR = os.path.abspath("./convergence_test")
os.makedirs(CONV_DIR, exist_ok=True)

# ══════════════════════════════════════════════════════════════════
#  QE Runner
# ══════════════════════════════════════════════════════════════════
def run_qe(config, ecutwfc, kgrid_str, tag):
    """Run a single QE SCF calculation."""
    outdir = os.path.join(CONV_DIR, "tmp", tag)
    os.makedirs(outdir, exist_ok=True)
    ecutrho = ecutwfc * config["ecutrho_ratio"]

    inp = config["template"].format(
        outdir=outdir,
        pseudo_dir=config["pseudo_dir"],
        ecutwfc=ecutwfc,
        ecutrho=ecutrho,
        kgrid=kgrid_str,
    )
    infile = os.path.join(CONV_DIR, f"{tag}.in")
    outfile = os.path.join(CONV_DIR, f"{tag}.out")

    with open(infile, "w") as f:
        f.write(inp)

    result = subprocess.run(
        ["mpirun", "-np", str(config["nproc"]), "pw.x", "-in", infile],
        capture_output=True, text=True, timeout=600
    )
    with open(outfile, "w") as f:
        f.write(result.stdout)

    # Parse energy
    energy_ry = None
    for line in result.stdout.split("\n"):
        if line.strip().startswith("!") and "total energy" in line:
            m = re.search(r"=\s+([-\d.]+)\s+Ry", line)
            if m: energy_ry = float(m.group(1))

    return energy_ry * 13.605693123 if energy_ry else None

# ══════════════════════════════════════════════════════════════════
#  Step 1: Cutoff convergence
# ══════════════════════════════════════════════════════════════════
print("=" * 70)
print("  STEP 1: Cutoff Energy Convergence")
print("=" * 70)

config = QE_CONFIG
nat = config["nat"]

ecut_results = []
for ec in config["ecutwfc_values"]:
    tag = f"ecut_{ec:03d}"
    kgrid = config["default_kgrid"]
    print(f"  ecutwfc = {ec:3d} Ry (k={kgrid})  ...", end="", flush=True)

    energy = run_qe(config, ec, kgrid, tag)
    ecut_results.append({"ecutwfc": ec, "energy_eV": energy})

    if energy is not None:
        print(f"  E = {energy:.6f} eV")
    else:
        print("  FAILED")

# Find optimal cutoff
valid_ecut = [r for r in ecut_results if r["energy_eV"] is not None]
ecut_arr = np.array([r["ecutwfc"] for r in valid_ecut])
en_ecut = np.array([r["energy_eV"] for r in valid_ecut])
de_ecut = (en_ecut - en_ecut[-1]) * 1000.0 / nat

optimal_ecut = config["default_ecutwfc"]
for i, de in enumerate(de_ecut):
    if abs(de) <= THRESHOLD_MEV:
        optimal_ecut = ecut_arr[i]
        break

print(f"\n  --> Optimal ecutwfc = {optimal_ecut} Ry")

# ══════════════════════════════════════════════════════════════════
#  Step 2: K-grid convergence (using optimal cutoff)
# ══════════════════════════════════════════════════════════════════
print(f"\n{'='*70}")
print(f"  STEP 2: K-Point Convergence (ecutwfc = {optimal_ecut} Ry)")
print(f"{'='*70}")

kgrid_results = []
for kg in config["kgrid_values"]:
    tag = f"kgrid_{kg:02d}"
    kgrid_str = f"{kg} {kg} {kg}"
    print(f"  k-grid = {kg:2d}^3 (ecutwfc={optimal_ecut})  ...", end="", flush=True)

    energy = run_qe(config, optimal_ecut, kgrid_str, tag)
    kgrid_results.append({"kgrid": kg, "energy_eV": energy})

    if energy is not None:
        print(f"  E = {energy:.6f} eV")
    else:
        print("  FAILED")

valid_kgrid = [r for r in kgrid_results if r["energy_eV"] is not None]
kg_arr = np.array([r["kgrid"] for r in valid_kgrid])
en_kg = np.array([r["energy_eV"] for r in valid_kgrid])
de_kgrid = (en_kg - en_kg[-1]) * 1000.0 / nat

optimal_kgrid = None
for i, de in enumerate(de_kgrid):
    if abs(de) <= THRESHOLD_MEV:
        optimal_kgrid = kg_arr[i]
        break

if optimal_kgrid:
    print(f"\n  --> Optimal k-grid = {optimal_kgrid}^3")

# ══════════════════════════════════════════════════════════════════
#  Summary Report
# ══════════════════════════════════════════════════════════════════
report = f"""
{'='*70}
  CONVERGENCE TEST SUMMARY
{'='*70}

  Code:              {CODE.upper()}
  Structure:         Si (diamond)
  Atoms:             {nat}
  Threshold:         {THRESHOLD_MEV} meV/atom

  Optimal ecutwfc:   {optimal_ecut} Ry ({optimal_ecut * 13.606:.0f} eV)
  Optimal k-grid:    {optimal_kgrid}x{optimal_kgrid}x{optimal_kgrid}

  Cutoff convergence:
    {'ecutwfc (Ry)':>12s}  {'dE (meV/atom)':>14s}
"""
for ec, de in zip(ecut_arr, de_ecut):
    marker = " <--" if ec == optimal_ecut else ""
    report += f"    {ec:12.0f}  {de:14.4f}{marker}\n"

report += f"""
  K-grid convergence:
    {'k-grid':>8s}  {'dE (meV/atom)':>14s}
"""
for kg, de in zip(kg_arr, de_kgrid):
    marker = " <--" if optimal_kgrid and kg == optimal_kgrid else ""
    report += f"    {kg:5.0f}^3    {de:14.4f}{marker}\n"

print(report)

# Save report
with open(os.path.join(CONV_DIR, "convergence_report.txt"), "w") as f:
    f.write(report)

# Save all results as JSON
all_results = {
    "threshold_meV": THRESHOLD_MEV,
    "optimal_ecutwfc_Ry": float(optimal_ecut),
    "optimal_kgrid": int(optimal_kgrid) if optimal_kgrid else None,
    "ecut_results": ecut_results,
    "kgrid_results": kgrid_results,
}
with open(os.path.join(CONV_DIR, "convergence_results.json"), "w") as f:
    json.dump(all_results, f, indent=2)

# ══════════════════════════════════════════════════════════════════
#  Combined Plot
# ══════════════════════════════════════════════════════════════════
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

# Cutoff convergence
ax1.plot(ecut_arr, np.abs(de_ecut), "o-", color="steelblue", linewidth=2, markersize=8)
ax1.axhline(THRESHOLD_MEV, color="red", linestyle="--", linewidth=1.5)
if optimal_ecut:
    ax1.axvline(optimal_ecut, color="green", linestyle=":", linewidth=1.5,
                label=f"Optimal: {optimal_ecut} Ry")
ax1.set_xlabel("ecutwfc (Ry)", fontsize=13)
ax1.set_ylabel("|dE| (meV/atom)", fontsize=13)
ax1.set_title("Cutoff Convergence", fontsize=14)
ax1.set_yscale("log")
ax1.legend(fontsize=11)
ax1.grid(True, alpha=0.3, which="both")

# K-grid convergence
ax2.plot(kg_arr, np.abs(de_kgrid), "s-", color="darkorange", linewidth=2, markersize=8)
ax2.axhline(THRESHOLD_MEV, color="red", linestyle="--", linewidth=1.5)
if optimal_kgrid:
    ax2.axvline(optimal_kgrid, color="green", linestyle=":", linewidth=1.5,
                label=f"Optimal: {optimal_kgrid}^3")
ax2.set_xlabel("k-grid (NxNxN)", fontsize=13)
ax2.set_ylabel("|dE| (meV/atom)", fontsize=13)
ax2.set_title("K-Point Convergence", fontsize=14)
ax2.set_yscale("log")
ax2.legend(fontsize=11)
ax2.grid(True, alpha=0.3, which="both")

fig.suptitle(f"DFT Convergence Test ({CODE.upper()}) -- Threshold: {THRESHOLD_MEV} meV/atom",
             fontsize=15, y=1.02)
plt.tight_layout()
plt.savefig(os.path.join(CONV_DIR, "convergence_combined.png"),
            dpi=200, bbox_inches="tight")
print(f"\nCombined plot saved: {CONV_DIR}/convergence_combined.png")
```

---

## Key Parameters

| Parameter | QE | VASP | Typical Range | Notes |
|---|---|---|---|---|
| Cutoff energy | `ecutwfc` (Ry) | `ENCUT` (eV) | 30--100 Ry / 400--800 eV | Depends on pseudopotential. Always converge. |
| Charge density cutoff | `ecutrho` (Ry) | Automatic | 4--12x ecutwfc | 4x for NC, 8--12x for US/PAW |
| K-grid | `K_POINTS (automatic) N N N` | KPOINTS file | 4^3 to 16^3 | Scale inversely with cell size |
| Smearing width | `degauss` (Ry) | `SIGMA` (eV) | 0.005--0.05 Ry / 0.05--0.2 eV | Critical for metals; irrelevant for insulators |
| SCF convergence | `conv_thr` (Ry) | `EDIFF` (eV) | 1e-8 Ry / 1e-6 eV | Tight enough that it does not limit accuracy |
| Accuracy threshold | N/A | N/A | 1--5 meV/atom | 1 meV/atom for publication quality |

## Interpreting Results

- **Convergence curve shape**: Total energy should decrease monotonically with increasing cutoff and approach a plateau. Oscillatory behavior may indicate numerical issues.
- **Optimal parameter**: The smallest cutoff/k-grid where the energy per atom differs from the reference by less than the threshold (e.g., 1 meV/atom).
- **Pseudopotential-suggested cutoff**: Check the pseudopotential file header for recommended cutoff values. The converged value should be at or above the recommendation.
- **K-grid scaling**: For the same accuracy, larger cells need coarser k-grids. A rule of thumb: k * a ~ constant, where a is the lattice parameter.
- **Smearing for metals**: The total energy depends on degauss/SIGMA. Use the smallest value that still allows SCF convergence. Cold smearing and Methfessel-Paxton converge faster than simple Gaussian.
- **Force convergence**: If you need accurate forces (for relaxation), converge forces as well as energies. Forces typically require higher cutoff than energies alone.
- **Non-cubic cells**: Test each k-grid direction independently. The required density scales with the reciprocal lattice vector length: Ni = round(density * |bi|).

## Common Issues

| Problem | Solution |
|---|---|
| Total energy not converging with cutoff | Check the ecutrho/ecutwfc ratio. For ultrasoft PP, ecutrho should be 8--12x ecutwfc. Try increasing the ratio. |
| K-grid convergence oscillates | For metals, use a consistent smearing scheme. For insulators, ensure `occupations='tetrahedra'` or `ISMEAR=-5`. Odd vs. even k-grids can give different results due to Gamma-point inclusion. |
| Very expensive calculations for large k-grids | Use k-point parallelism (`-nk` flag in QE, `NCORE`/`KPAR` in VASP). Consider only testing k-grids that are multiples of the MPI parallelization. |
| Smearing test shows no trend | The system is likely an insulator. Smearing convergence is only meaningful for metals. For insulators, use `occupations='tetrahedra'` or `ISMEAR=-5` and skip smearing tests. |
| Different pseudopotentials give different converged cutoffs | This is expected. Each pseudopotential has its own cutoff requirements. Always converge the cutoff for the specific PP you are using. |
| Energy not monotonic with k-grid | This can happen if symmetry is on and different k-grids sample different irreducible k-points. Turn off symmetry (`nosym=.true.` in QE or `ISYM=-1` in VASP) for a cleaner convergence test, then re-enable for production. |
| Results differ between QE and VASP | Expected due to different PP types (US vs PAW), different augmentation charge handling, and implementation differences. Converge each code independently. |
