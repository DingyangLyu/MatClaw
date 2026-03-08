# Automated DFT Convergence Testing

## When to Use

- You need to determine optimal DFT parameters for a new material or pseudopotential set before production calculations.
- You want to automate ecutwfc (plane-wave cutoff) convergence sweeps and find the minimum cutoff that meets accuracy requirements.
- You want to automate k-point grid convergence sweeps and find the coarsest grid that meets accuracy requirements.
- You need a combined workflow that first converges the cutoff, then uses the converged cutoff to converge the k-grid.
- You need to run convergence testing for multiple materials in batch mode (e.g., all compounds in a study).
- You want publication-quality convergence plots and a machine-readable report with recommended parameters.
- You need fully standalone Python scripts with no pyiron or workflow-framework dependency.

## Design Principles (inspired by pyiron_atomistics)

This skill is inspired by the convergence testing architecture in pyiron_atomistics (`ConvEncutParallel`, `ConvKpointParallel`), which uses a parallel master/child pattern:

1. **Parameter sweep as a generator**: define min, max, and number of points (or step) for the parameter to converge.
2. **Each point is an independent calculation**: run all points, collect energies, compare against the highest-fidelity reference.
3. **Convergence criterion**: energy per atom change < threshold (default 1 meV/atom) relative to the most expensive calculation.
4. **Sequential workflow**: converge the cutoff first (with a well-converged k-grid), then converge the k-grid (with the newly determined cutoff).

The scripts below implement this pattern as standalone Python, driving QE `pw.x` via subprocess.

## Prerequisites

- Quantum ESPRESSO 7.x installed (`pw.x` in PATH, optionally `mpirun`).
- Pseudopotential files in a local `./pseudo/` directory (SSSP or pslibrary UPF format).
- Python packages: `numpy`, `matplotlib`, `json` (stdlib).
- Optional: `pymatgen` for structure I/O; `ase` for structure building. Neither is required for the core scripts.

---

## Detailed Steps

### Method A: Automated ecutwfc Convergence

Sweeps ecutwfc from a minimum to a maximum value, runs SCF at each point with a fixed (well-converged) k-grid, and determines the optimal cutoff where the energy per atom converges to within a threshold of the highest-cutoff reference.

#### Step A1: ecutwfc Convergence Sweep

```python
#!/usr/bin/env python3
"""
Automated ecutwfc convergence test for Quantum ESPRESSO.

Inspired by pyiron_atomistics ConvEncutParallel:
  - Sweeps ecutwfc over a linspace(min, max, num_points) range.
  - Runs independent SCF calculations at each cutoff with a fixed k-grid.
  - Determines the optimal cutoff using an energy-per-atom threshold.
  - Generates a convergence plot and JSON report.

No pyiron dependency -- standalone script using subprocess to call pw.x.
"""
import os
import subprocess
import re
import json
import sys
import time
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ================================================================== #
#  CONFIGURATION -- edit this section for your system
# ================================================================== #

# --- Paths ---
PSEUDO_DIR = os.path.abspath("./pseudo")
OUTDIR_BASE = os.path.abspath("./conv_ecutwfc")
PW_COMMAND = "pw.x"           # or full path, e.g. "/usr/bin/pw.x"
MPI_COMMAND = "mpirun"         # set to None to run serial
NPROC = 4                      # MPI ranks (ignored if MPI_COMMAND is None)

# --- Convergence parameters (pyiron-style: min, max, num_points) ---
ECUTWFC_MIN = 20.0             # Ry
ECUTWFC_MAX = 100.0            # Ry
ECUTWFC_NUM_POINTS = 17        # number of sample points in the sweep
# Alternatively, supply an explicit list (overrides min/max/num_points):
# ECUTWFC_LIST = [20, 25, 30, 35, 40, 45, 50, 55, 60, 70, 80, 90, 100]
ECUTWFC_LIST = None

# --- ecutrho / ecutwfc ratio ---
# 4 for norm-conserving pseudopotentials
# 8-12 for ultrasoft / PAW pseudopotentials
ECUTRHO_RATIO = 8

# --- Fixed k-grid (must be well-converged so only ecutwfc is tested) ---
KGRID = (8, 8, 8)

# --- Convergence criterion ---
THRESHOLD_MEV_PER_ATOM = 1.0   # meV/atom relative to highest cutoff

# --- Structure definition ---
# For quick editing, the QE input template is defined inline.
# Replace the SYSTEM / ATOMIC_SPECIES / ATOMIC_POSITIONS blocks for your material.
PREFIX = "conv"
NAT = 2   # number of atoms in the unit cell

STRUCTURE_TEMPLATE = """\
&SYSTEM
    ibrav       = 2
    celldm(1)   = 10.26
    nat         = {nat}
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
  {kx} {ky} {kz}  0 0 0
"""

# ================================================================== #
#  Build the parameter list
# ================================================================== #
if ECUTWFC_LIST is not None:
    ecutwfc_values = sorted(ECUTWFC_LIST)
else:
    ecutwfc_values = [
        round(v, 2)
        for v in np.linspace(ECUTWFC_MIN, ECUTWFC_MAX, ECUTWFC_NUM_POINTS)
    ]

os.makedirs(OUTDIR_BASE, exist_ok=True)

# ================================================================== #
#  QE runner
# ================================================================== #
RY_TO_EV = 13.605693123


def run_qe_scf(ecutwfc, tag):
    """Run a single QE SCF calculation. Returns a result dict."""
    outdir = os.path.join(OUTDIR_BASE, tag)
    os.makedirs(outdir, exist_ok=True)
    ecutrho = ecutwfc * ECUTRHO_RATIO

    inp = f"""&CONTROL
    calculation = 'scf'
    prefix      = '{PREFIX}'
    outdir      = '{outdir}'
    pseudo_dir  = '{PSEUDO_DIR}'
    tprnfor     = .true.
    tstress     = .true.
/
{STRUCTURE_TEMPLATE.format(
    nat=NAT, ecutwfc=ecutwfc, ecutrho=ecutrho,
    kx=KGRID[0], ky=KGRID[1], kz=KGRID[2]
)}
"""
    infile = os.path.join(OUTDIR_BASE, f"scf_{tag}.in")
    outfile = os.path.join(OUTDIR_BASE, f"scf_{tag}.out")

    with open(infile, "w") as f:
        f.write(inp)

    cmd = []
    if MPI_COMMAND is not None:
        cmd = [MPI_COMMAND, "-np", str(NPROC)]
    cmd += [PW_COMMAND, "-in", infile]

    t0 = time.time()
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=1800)
    wall = time.time() - t0

    with open(outfile, "w") as f:
        f.write(proc.stdout)
        if proc.stderr:
            f.write("\n--- STDERR ---\n")
            f.write(proc.stderr)

    # --- Parse output ---
    energy_ry = None
    total_force = None
    pressure_kbar = None
    scf_converged = False

    for line in proc.stdout.splitlines():
        if line.strip().startswith("!") and "total energy" in line:
            m = re.search(r"=\s+([-\d.]+)\s+Ry", line)
            if m:
                energy_ry = float(m.group(1))
        if "Total force" in line:
            m = re.search(r"Total force\s*=\s*([\d.]+)", line)
            if m:
                total_force = float(m.group(1))
        if "P=" in line:
            m = re.search(r"P=\s*([-\d.]+)", line)
            if m:
                pressure_kbar = float(m.group(1))
        if "convergence has been achieved" in line:
            scf_converged = True

    energy_ev = energy_ry * RY_TO_EV if energy_ry is not None else None

    return {
        "ecutwfc_Ry": ecutwfc,
        "ecutrho_Ry": ecutwfc * ECUTRHO_RATIO,
        "energy_Ry": energy_ry,
        "energy_eV": energy_ev,
        "energy_per_atom_eV": energy_ev / NAT if energy_ev is not None else None,
        "total_force_Ry_bohr": total_force,
        "pressure_kbar": pressure_kbar,
        "scf_converged": scf_converged,
        "wall_time_s": round(wall, 2),
    }


# ================================================================== #
#  Run all calculations
# ================================================================== #
print("=" * 70)
print("  Automated ecutwfc Convergence Test")
print(f"  {len(ecutwfc_values)} points: "
      f"{ecutwfc_values[0]} -- {ecutwfc_values[-1]} Ry")
print(f"  Fixed k-grid: {KGRID[0]}x{KGRID[1]}x{KGRID[2]}")
print(f"  Threshold: {THRESHOLD_MEV_PER_ATOM} meV/atom")
print("=" * 70)

results = []
for ec in ecutwfc_values:
    tag = f"ecut_{ec:07.2f}".replace(".", "p")
    print(f"  ecutwfc = {ec:7.2f} Ry  ...", end="", flush=True)
    r = run_qe_scf(ec, tag)
    results.append(r)
    if r["energy_eV"] is not None:
        print(f"  E/atom = {r['energy_per_atom_eV']:.6f} eV  "
              f"({r['wall_time_s']:.1f}s)  "
              f"[{'OK' if r['scf_converged'] else 'WARN: not converged'}]")
    else:
        print("  FAILED -- check output file")

# ================================================================== #
#  Analysis
# ================================================================== #
valid = [r for r in results if r["energy_eV"] is not None]
if not valid:
    print("\nERROR: No calculations produced an energy. Aborting.")
    sys.exit(1)

ecutwfc_arr = np.array([r["ecutwfc_Ry"] for r in valid])
energy_arr = np.array([r["energy_per_atom_eV"] for r in valid])
wall_arr = np.array([r["wall_time_s"] for r in valid])

# Reference: highest cutoff (last valid point)
e_ref = energy_arr[-1]
de_meV = (energy_arr - e_ref) * 1000.0   # meV/atom

# Find optimal cutoff (first point within threshold)
optimal_ecutwfc = None
optimal_index = None
for i, de in enumerate(de_meV):
    if abs(de) <= THRESHOLD_MEV_PER_ATOM:
        optimal_ecutwfc = ecutwfc_arr[i]
        optimal_index = i
        break

# ================================================================== #
#  Print summary table
# ================================================================== #
print(f"\n{'=' * 70}")
print(f"  CONVERGENCE RESULTS")
print(f"  Reference: ecutwfc = {ecutwfc_arr[-1]:.2f} Ry")
print(f"  Threshold: {THRESHOLD_MEV_PER_ATOM} meV/atom")
print(f"{'=' * 70}")
print(f"  {'ecutwfc (Ry)':>12s}  {'E/atom (eV)':>14s}  "
      f"{'dE (meV/atom)':>14s}  {'Wall (s)':>9s}")
print(f"  {'-'*12}  {'-'*14}  {'-'*14}  {'-'*9}")
for ec, e, de, w in zip(ecutwfc_arr, energy_arr, de_meV, wall_arr):
    marker = "  <-- optimal" if optimal_ecutwfc is not None and ec == optimal_ecutwfc else ""
    print(f"  {ec:12.2f}  {e:14.6f}  {de:14.4f}  {w:9.1f}{marker}")

if optimal_ecutwfc is not None:
    print(f"\n  Recommended ecutwfc = {optimal_ecutwfc} Ry "
          f"({optimal_ecutwfc * RY_TO_EV:.1f} eV)")
    print(f"  Converged to {THRESHOLD_MEV_PER_ATOM} meV/atom "
          f"relative to {ecutwfc_arr[-1]:.2f} Ry reference.")
else:
    print(f"\n  WARNING: energy did NOT converge to {THRESHOLD_MEV_PER_ATOM} meV/atom "
          f"within the tested range.")
    print(f"  Extend ECUTWFC_MAX or increase ECUTWFC_NUM_POINTS.")

# ================================================================== #
#  Save results as JSON
# ================================================================== #
report = {
    "parameter": "ecutwfc",
    "threshold_meV_per_atom": THRESHOLD_MEV_PER_ATOM,
    "reference_ecutwfc_Ry": float(ecutwfc_arr[-1]),
    "reference_energy_per_atom_eV": float(e_ref),
    "optimal_ecutwfc_Ry": float(optimal_ecutwfc) if optimal_ecutwfc is not None else None,
    "optimal_ecutwfc_eV": float(optimal_ecutwfc * RY_TO_EV) if optimal_ecutwfc is not None else None,
    "kgrid_fixed": list(KGRID),
    "nat": NAT,
    "data": results,
}
report_path = os.path.join(OUTDIR_BASE, "ecutwfc_convergence.json")
with open(report_path, "w") as f:
    json.dump(report, f, indent=2)
print(f"\n  JSON report: {report_path}")

# ================================================================== #
#  Plot
# ================================================================== #
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# (a) Absolute energy vs cutoff
axes[0].plot(ecutwfc_arr, energy_arr, "o-", color="steelblue",
             linewidth=2, markersize=7)
axes[0].set_xlabel("ecutwfc (Ry)", fontsize=13)
axes[0].set_ylabel("Energy per atom (eV)", fontsize=13)
axes[0].set_title("(a) Total Energy vs. Cutoff", fontsize=14)
axes[0].grid(True, alpha=0.3)

# (b) Energy difference (log scale)
axes[1].plot(ecutwfc_arr, np.abs(de_meV), "s-", color="darkorange",
             linewidth=2, markersize=7)
axes[1].axhline(THRESHOLD_MEV_PER_ATOM, color="red", linestyle="--",
                linewidth=1.5, label=f"Threshold: {THRESHOLD_MEV_PER_ATOM} meV/atom")
if optimal_ecutwfc is not None:
    axes[1].axvline(optimal_ecutwfc, color="green", linestyle=":",
                    linewidth=1.5, label=f"Optimal: {optimal_ecutwfc} Ry")
axes[1].set_xlabel("ecutwfc (Ry)", fontsize=13)
axes[1].set_ylabel("|dE| (meV/atom)", fontsize=13)
axes[1].set_title("(b) Convergence", fontsize=14)
axes[1].set_yscale("log")
axes[1].legend(fontsize=10, loc="upper right")
axes[1].grid(True, alpha=0.3, which="both")

# (c) Computational cost
axes[2].bar(ecutwfc_arr, wall_arr, width=(ecutwfc_arr[1] - ecutwfc_arr[0]) * 0.7,
            color="mediumseagreen", edgecolor="black", linewidth=0.5)
if optimal_ecutwfc is not None:
    axes[2].axvline(optimal_ecutwfc, color="green", linestyle=":",
                    linewidth=1.5, label=f"Optimal: {optimal_ecutwfc} Ry")
    axes[2].legend(fontsize=10)
axes[2].set_xlabel("ecutwfc (Ry)", fontsize=13)
axes[2].set_ylabel("Wall time (s)", fontsize=13)
axes[2].set_title("(c) Computational Cost", fontsize=14)
axes[2].grid(True, alpha=0.3)

fig.suptitle("ecutwfc Convergence Test", fontsize=16, y=1.02)
plt.tight_layout()
plot_path = os.path.join(OUTDIR_BASE, "ecutwfc_convergence.png")
plt.savefig(plot_path, dpi=200, bbox_inches="tight")
plt.close()
print(f"  Plot saved: {plot_path}")
```

---

### Method B: Automated k-Point Convergence

Sweeps k-point grids from coarse to dense, runs SCF at each point with a fixed ecutwfc, and determines the optimal grid.

#### Step B1: k-Point Convergence Sweep

```python
#!/usr/bin/env python3
"""
Automated k-point grid convergence test for Quantum ESPRESSO.

Inspired by pyiron_atomistics ConvKpointParallel:
  - Sweeps k-grid from min to max in steps.
  - Runs independent SCF calculations at each grid density.
  - Determines the optimal k-grid using an energy-per-atom threshold.
  - Supports isotropic (NxNxN) and anisotropic grids.
  - Generates convergence plot and JSON report.

No pyiron dependency -- standalone script using subprocess to call pw.x.
"""
import os
import subprocess
import re
import json
import sys
import time
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ================================================================== #
#  CONFIGURATION
# ================================================================== #
PSEUDO_DIR = os.path.abspath("./pseudo")
OUTDIR_BASE = os.path.abspath("./conv_kgrid")
PW_COMMAND = "pw.x"
MPI_COMMAND = "mpirun"
NPROC = 4

# --- K-grid sweep (pyiron-style: min, max, step) ---
# For isotropic grids (NxNxN for cubic cells):
KGRID_MIN = 2
KGRID_MAX = 16
KGRID_STEP = 2
# Alternatively, supply an explicit list:
# KGRID_LIST = [2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 14, 16]
KGRID_LIST = None

# For anisotropic grids (non-cubic cells), set this to True
# and define lattice parameters so the script computes proportional grids.
ANISOTROPIC = False
LATTICE_ABC = (5.43, 5.43, 5.43)  # lattice parameters in Angstrom

# --- Fixed ecutwfc (use the converged value from Method A) ---
ECUTWFC = 50.0   # Ry
ECUTRHO_RATIO = 8

# --- Convergence criterion ---
THRESHOLD_MEV_PER_ATOM = 1.0

# --- Structure ---
PREFIX = "conv"
NAT = 2

STRUCTURE_TEMPLATE = """\
&SYSTEM
    ibrav       = 2
    celldm(1)   = 10.26
    nat         = {nat}
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
  {kx} {ky} {kz}  0 0 0
"""

# ================================================================== #
#  Build the k-grid list
# ================================================================== #
if KGRID_LIST is not None:
    kgrid_n_values = sorted(KGRID_LIST)
else:
    kgrid_n_values = list(range(KGRID_MIN, KGRID_MAX + KGRID_STEP, KGRID_STEP))


def compute_kgrid(n):
    """Return (kx, ky, kz) for a given density parameter n."""
    if not ANISOTROPIC:
        return (n, n, n)
    # For anisotropic: scale inversely with lattice parameter
    # so that k_i * a_i ~ constant
    a_max = max(LATTICE_ABC)
    kx = max(1, round(n * a_max / LATTICE_ABC[0]))
    ky = max(1, round(n * a_max / LATTICE_ABC[1]))
    kz = max(1, round(n * a_max / LATTICE_ABC[2]))
    return (kx, ky, kz)


os.makedirs(OUTDIR_BASE, exist_ok=True)

# ================================================================== #
#  QE runner
# ================================================================== #
RY_TO_EV = 13.605693123


def run_qe_scf_kgrid(kgrid, tag):
    """Run a single SCF with the given k-grid. Returns a result dict."""
    kx, ky, kz = kgrid
    outdir = os.path.join(OUTDIR_BASE, tag)
    os.makedirs(outdir, exist_ok=True)
    ecutrho = ECUTWFC * ECUTRHO_RATIO

    inp = f"""&CONTROL
    calculation = 'scf'
    prefix      = '{PREFIX}'
    outdir      = '{outdir}'
    pseudo_dir  = '{PSEUDO_DIR}'
/
{STRUCTURE_TEMPLATE.format(
    nat=NAT, ecutwfc=ECUTWFC, ecutrho=ecutrho,
    kx=kx, ky=ky, kz=kz
)}
"""
    infile = os.path.join(OUTDIR_BASE, f"scf_{tag}.in")
    outfile = os.path.join(OUTDIR_BASE, f"scf_{tag}.out")

    with open(infile, "w") as f:
        f.write(inp)

    cmd = []
    if MPI_COMMAND is not None:
        cmd = [MPI_COMMAND, "-np", str(NPROC)]
    cmd += [PW_COMMAND, "-in", infile]

    t0 = time.time()
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=1800)
    wall = time.time() - t0

    with open(outfile, "w") as f:
        f.write(proc.stdout)

    energy_ry = None
    n_kpts_irr = None
    scf_converged = False

    for line in proc.stdout.splitlines():
        if line.strip().startswith("!") and "total energy" in line:
            m = re.search(r"=\s+([-\d.]+)\s+Ry", line)
            if m:
                energy_ry = float(m.group(1))
        if "number of k points=" in line:
            m = re.search(r"=\s+(\d+)", line)
            if m:
                n_kpts_irr = int(m.group(1))
        if "convergence has been achieved" in line:
            scf_converged = True

    energy_ev = energy_ry * RY_TO_EV if energy_ry is not None else None

    return {
        "kgrid": list(kgrid),
        "kgrid_n": kgrid[0],  # for isotropic: just the N value
        "n_kpts_irr": n_kpts_irr,
        "energy_eV": energy_ev,
        "energy_per_atom_eV": energy_ev / NAT if energy_ev is not None else None,
        "scf_converged": scf_converged,
        "wall_time_s": round(wall, 2),
    }


# ================================================================== #
#  Run all calculations
# ================================================================== #
print("=" * 70)
print("  Automated K-Point Convergence Test")
print(f"  {len(kgrid_n_values)} grids: "
      f"{kgrid_n_values[0]} -- {kgrid_n_values[-1]}")
print(f"  Fixed ecutwfc = {ECUTWFC} Ry")
print(f"  Threshold: {THRESHOLD_MEV_PER_ATOM} meV/atom")
print("=" * 70)

results = []
for n in kgrid_n_values:
    kgrid = compute_kgrid(n)
    tag = f"kgrid_{kgrid[0]:02d}_{kgrid[1]:02d}_{kgrid[2]:02d}"
    print(f"  k-grid = {kgrid[0]}x{kgrid[1]}x{kgrid[2]}  ...", end="", flush=True)
    r = run_qe_scf_kgrid(kgrid, tag)
    results.append(r)
    if r["energy_eV"] is not None:
        irr = f"({r['n_kpts_irr']} irr. k-pts)" if r["n_kpts_irr"] else ""
        print(f"  E/atom = {r['energy_per_atom_eV']:.6f} eV  "
              f"({r['wall_time_s']:.1f}s)  {irr}")
    else:
        print("  FAILED")

# ================================================================== #
#  Analysis
# ================================================================== #
valid = [r for r in results if r["energy_eV"] is not None]
if not valid:
    print("\nERROR: No calculations produced an energy. Aborting.")
    sys.exit(1)

kgrid_arr = np.array([r["kgrid_n"] for r in valid])
energy_arr = np.array([r["energy_per_atom_eV"] for r in valid])
nkpts_arr = np.array([r["n_kpts_irr"] if r["n_kpts_irr"] else 0 for r in valid])
wall_arr = np.array([r["wall_time_s"] for r in valid])

e_ref = energy_arr[-1]
de_meV = (energy_arr - e_ref) * 1000.0

optimal_kgrid = None
for i, de in enumerate(de_meV):
    if abs(de) <= THRESHOLD_MEV_PER_ATOM:
        optimal_kgrid = int(kgrid_arr[i])
        break

print(f"\n{'=' * 70}")
print(f"  CONVERGENCE RESULTS")
print(f"  Reference: k-grid = {int(kgrid_arr[-1])}^3")
print(f"  Threshold: {THRESHOLD_MEV_PER_ATOM} meV/atom")
print(f"{'=' * 70}")
print(f"  {'k-grid':>8s}  {'irr. k-pts':>10s}  {'E/atom (eV)':>14s}  "
      f"{'dE (meV/atom)':>14s}  {'Wall (s)':>9s}")
print(f"  {'-'*8}  {'-'*10}  {'-'*14}  {'-'*14}  {'-'*9}")
for kg, nk, e, de, w in zip(kgrid_arr, nkpts_arr, energy_arr, de_meV, wall_arr):
    marker = "  <-- optimal" if optimal_kgrid is not None and int(kg) == optimal_kgrid else ""
    print(f"  {int(kg):5d}^3  {int(nk):10d}  {e:14.6f}  {de:14.4f}  {w:9.1f}{marker}")

if optimal_kgrid is not None:
    print(f"\n  Recommended k-grid = {optimal_kgrid}x{optimal_kgrid}x{optimal_kgrid}")
else:
    print(f"\n  WARNING: energy did NOT converge to {THRESHOLD_MEV_PER_ATOM} meV/atom.")
    print(f"  Extend KGRID_MAX or decrease KGRID_STEP.")

# ================================================================== #
#  Save report
# ================================================================== #
report = {
    "parameter": "kgrid",
    "threshold_meV_per_atom": THRESHOLD_MEV_PER_ATOM,
    "reference_kgrid": int(kgrid_arr[-1]),
    "reference_energy_per_atom_eV": float(e_ref),
    "optimal_kgrid": optimal_kgrid,
    "ecutwfc_fixed_Ry": ECUTWFC,
    "nat": NAT,
    "anisotropic": ANISOTROPIC,
    "data": results,
}
report_path = os.path.join(OUTDIR_BASE, "kgrid_convergence.json")
with open(report_path, "w") as f:
    json.dump(report, f, indent=2)
print(f"  JSON report: {report_path}")

# ================================================================== #
#  Plot
# ================================================================== #
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

axes[0].plot(kgrid_arr, energy_arr, "o-", color="steelblue",
             linewidth=2, markersize=7)
axes[0].set_xlabel("k-grid (NxNxN)", fontsize=13)
axes[0].set_ylabel("Energy per atom (eV)", fontsize=13)
axes[0].set_title("(a) Total Energy vs. k-grid", fontsize=14)
axes[0].grid(True, alpha=0.3)

axes[1].plot(kgrid_arr, np.abs(de_meV), "s-", color="darkorange",
             linewidth=2, markersize=7)
axes[1].axhline(THRESHOLD_MEV_PER_ATOM, color="red", linestyle="--",
                linewidth=1.5, label=f"Threshold: {THRESHOLD_MEV_PER_ATOM} meV/atom")
if optimal_kgrid is not None:
    axes[1].axvline(optimal_kgrid, color="green", linestyle=":",
                    linewidth=1.5, label=f"Optimal: {optimal_kgrid}^3")
axes[1].set_xlabel("k-grid (NxNxN)", fontsize=13)
axes[1].set_ylabel("|dE| (meV/atom)", fontsize=13)
axes[1].set_title("(b) Convergence", fontsize=14)
axes[1].set_yscale("log")
axes[1].legend(fontsize=10)
axes[1].grid(True, alpha=0.3, which="both")

axes[2].plot(nkpts_arr, np.abs(de_meV), "D-", color="mediumpurple",
             linewidth=2, markersize=7)
axes[2].axhline(THRESHOLD_MEV_PER_ATOM, color="red", linestyle="--",
                linewidth=1.5)
axes[2].set_xlabel("Irreducible k-points", fontsize=13)
axes[2].set_ylabel("|dE| (meV/atom)", fontsize=13)
axes[2].set_title("(c) Convergence vs. Cost", fontsize=14)
axes[2].set_yscale("log")
axes[2].grid(True, alpha=0.3, which="both")

fig.suptitle(f"K-Point Convergence (ecutwfc = {ECUTWFC} Ry)", fontsize=16, y=1.02)
plt.tight_layout()
plot_path = os.path.join(OUTDIR_BASE, "kgrid_convergence.png")
plt.savefig(plot_path, dpi=200, bbox_inches="tight")
plt.close()
print(f"  Plot saved: {plot_path}")
```

---

### Method C: Combined Convergence Workflow

Runs Method A (ecutwfc convergence) first, then feeds the optimal cutoff into Method B (k-grid convergence). Produces a unified report and combined plot. This is the recommended workflow for new materials.

#### Step C1: Full Convergence Pipeline

```python
#!/usr/bin/env python3
"""
Combined automated convergence workflow for Quantum ESPRESSO.

Pipeline:
  1. Sweep ecutwfc with a well-converged k-grid -> find optimal ecutwfc.
  2. Sweep k-grid with the optimal ecutwfc      -> find optimal k-grid.
  3. Generate a unified report and combined convergence plot.

Inspired by pyiron_atomistics' sequential convergence pattern
(ConvEncutParallel followed by ConvKpointParallel).

No pyiron dependency -- standalone script using subprocess to call pw.x.
"""
import os
import subprocess
import re
import json
import sys
import time
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ================================================================== #
#  CONFIGURATION
# ================================================================== #
PSEUDO_DIR = os.path.abspath("./pseudo")
WORK_DIR = os.path.abspath("./convergence_full")
PW_COMMAND = "pw.x"
MPI_COMMAND = "mpirun"         # set to None for serial execution
NPROC = 4

# --- ecutwfc sweep ---
ECUTWFC_VALUES = [20, 25, 30, 35, 40, 45, 50, 55, 60, 70, 80, 90, 100]
ECUTRHO_RATIO = 8
KGRID_FOR_ECUT = (8, 8, 8)    # well-converged k-grid for ecutwfc test

# --- k-grid sweep ---
KGRID_VALUES = [2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 14, 16]
FALLBACK_ECUTWFC = 60.0        # used if ecutwfc sweep fails to converge

# --- Convergence criterion ---
THRESHOLD_MEV = 1.0            # meV/atom

# --- Structure ---
PREFIX = "conv"
NAT = 2

STRUCTURE_TEMPLATE = """\
&SYSTEM
    ibrav       = 2
    celldm(1)   = 10.26
    nat         = {nat}
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
  Si  0.00  0.00  0.00
  Si  0.25  0.25  0.25

K_POINTS (automatic)
  {kx} {ky} {kz}  0 0 0
"""

os.makedirs(WORK_DIR, exist_ok=True)

RY_TO_EV = 13.605693123

# ================================================================== #
#  QE runner (shared by both phases)
# ================================================================== #
def run_scf(ecutwfc, kgrid, tag, work_subdir):
    """Run a single QE SCF and return a result dict."""
    basedir = os.path.join(WORK_DIR, work_subdir)
    os.makedirs(basedir, exist_ok=True)
    outdir = os.path.join(basedir, tag)
    os.makedirs(outdir, exist_ok=True)
    ecutrho = ecutwfc * ECUTRHO_RATIO

    inp = f"""&CONTROL
    calculation = 'scf'
    prefix      = '{PREFIX}'
    outdir      = '{outdir}'
    pseudo_dir  = '{PSEUDO_DIR}'
    tprnfor     = .true.
    tstress     = .true.
/
{STRUCTURE_TEMPLATE.format(
    nat=NAT, ecutwfc=ecutwfc, ecutrho=ecutrho,
    kx=kgrid[0], ky=kgrid[1], kz=kgrid[2]
)}
"""
    infile = os.path.join(basedir, f"scf_{tag}.in")
    outfile = os.path.join(basedir, f"scf_{tag}.out")

    with open(infile, "w") as f:
        f.write(inp)

    cmd = []
    if MPI_COMMAND is not None:
        cmd = [MPI_COMMAND, "-np", str(NPROC)]
    cmd += [PW_COMMAND, "-in", infile]

    t0 = time.time()
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=1800)
    wall = time.time() - t0

    with open(outfile, "w") as f:
        f.write(proc.stdout)

    energy_ry = None
    n_kpts_irr = None
    scf_ok = False
    for line in proc.stdout.splitlines():
        if line.strip().startswith("!") and "total energy" in line:
            m = re.search(r"=\s+([-\d.]+)\s+Ry", line)
            if m:
                energy_ry = float(m.group(1))
        if "number of k points=" in line:
            m = re.search(r"=\s+(\d+)", line)
            if m:
                n_kpts_irr = int(m.group(1))
        if "convergence has been achieved" in line:
            scf_ok = True

    energy_ev = energy_ry * RY_TO_EV if energy_ry is not None else None

    return {
        "ecutwfc_Ry": ecutwfc,
        "kgrid": list(kgrid),
        "n_kpts_irr": n_kpts_irr,
        "energy_eV": energy_ev,
        "energy_per_atom_eV": energy_ev / NAT if energy_ev is not None else None,
        "scf_converged": scf_ok,
        "wall_time_s": round(wall, 2),
    }


def find_optimal(param_arr, de_meV_arr, threshold):
    """Return the first parameter value where |dE| <= threshold, or None."""
    for i, de in enumerate(de_meV_arr):
        if abs(de) <= threshold:
            return param_arr[i]
    return None


# ================================================================== #
#  PHASE 1: ecutwfc convergence
# ================================================================== #
print("=" * 70)
print("  PHASE 1: ecutwfc Convergence")
print(f"  k-grid fixed at {KGRID_FOR_ECUT[0]}x{KGRID_FOR_ECUT[1]}x{KGRID_FOR_ECUT[2]}")
print("=" * 70)

ecut_results = []
for ec in ECUTWFC_VALUES:
    tag = f"ecut_{ec:03d}"
    print(f"  ecutwfc = {ec:3d} Ry  ...", end="", flush=True)
    r = run_scf(ec, KGRID_FOR_ECUT, tag, "phase1_ecutwfc")
    ecut_results.append(r)
    if r["energy_per_atom_eV"] is not None:
        print(f"  E/atom = {r['energy_per_atom_eV']:.6f} eV  ({r['wall_time_s']:.1f}s)")
    else:
        print("  FAILED")

valid_ecut = [r for r in ecut_results if r["energy_per_atom_eV"] is not None]
if not valid_ecut:
    print("ERROR: No ecutwfc calculations succeeded. Aborting.")
    sys.exit(1)

ecut_arr = np.array([r["ecutwfc_Ry"] for r in valid_ecut])
en_ecut = np.array([r["energy_per_atom_eV"] for r in valid_ecut])
de_ecut = (en_ecut - en_ecut[-1]) * 1000.0

optimal_ecutwfc = find_optimal(ecut_arr, de_ecut, THRESHOLD_MEV)
if optimal_ecutwfc is None:
    optimal_ecutwfc = FALLBACK_ECUTWFC
    print(f"\n  WARNING: ecutwfc not converged. Using fallback = {FALLBACK_ECUTWFC} Ry")
else:
    print(f"\n  --> Optimal ecutwfc = {optimal_ecutwfc} Ry "
          f"({optimal_ecutwfc * RY_TO_EV:.1f} eV)")

# ================================================================== #
#  PHASE 2: k-grid convergence (using optimal ecutwfc)
# ================================================================== #
print(f"\n{'=' * 70}")
print(f"  PHASE 2: K-Point Convergence")
print(f"  ecutwfc fixed at {optimal_ecutwfc} Ry")
print("=" * 70)

kgrid_results = []
for kg in KGRID_VALUES:
    kgrid = (kg, kg, kg)
    tag = f"kgrid_{kg:02d}"
    print(f"  k-grid = {kg:2d}^3  ...", end="", flush=True)
    r = run_scf(optimal_ecutwfc, kgrid, tag, "phase2_kgrid")
    kgrid_results.append(r)
    if r["energy_per_atom_eV"] is not None:
        irr = f"({r['n_kpts_irr']} irr.)" if r["n_kpts_irr"] else ""
        print(f"  E/atom = {r['energy_per_atom_eV']:.6f} eV  "
              f"({r['wall_time_s']:.1f}s)  {irr}")
    else:
        print("  FAILED")

valid_kgrid = [r for r in kgrid_results if r["energy_per_atom_eV"] is not None]
if not valid_kgrid:
    print("ERROR: No k-grid calculations succeeded.")
    sys.exit(1)

kg_arr = np.array([r["kgrid"][0] for r in valid_kgrid])
en_kg = np.array([r["energy_per_atom_eV"] for r in valid_kgrid])
de_kgrid = (en_kg - en_kg[-1]) * 1000.0

optimal_kgrid = find_optimal(kg_arr, de_kgrid, THRESHOLD_MEV)
if optimal_kgrid is not None:
    optimal_kgrid = int(optimal_kgrid)
    print(f"\n  --> Optimal k-grid = {optimal_kgrid}x{optimal_kgrid}x{optimal_kgrid}")
else:
    print(f"\n  WARNING: k-grid not converged within tested range.")

# ================================================================== #
#  SUMMARY REPORT
# ================================================================== #
report_text = f"""
{'=' * 70}
  CONVERGENCE TEST SUMMARY
{'=' * 70}

  Structure:         Si (diamond, {NAT} atoms)
  Pseudopotential:   Si.pbe-n-rrkjus_psl.1.0.0.UPF
  ecutrho/ecutwfc:   {ECUTRHO_RATIO}x
  Threshold:         {THRESHOLD_MEV} meV/atom

  RESULTS:
    Optimal ecutwfc:   {optimal_ecutwfc} Ry ({optimal_ecutwfc * RY_TO_EV:.1f} eV)
    Optimal k-grid:    {f'{optimal_kgrid}x{optimal_kgrid}x{optimal_kgrid}' if optimal_kgrid else 'NOT CONVERGED'}

  ECUTWFC CONVERGENCE (k-grid = {KGRID_FOR_ECUT[0]}x{KGRID_FOR_ECUT[1]}x{KGRID_FOR_ECUT[2]}):
    {'ecutwfc (Ry)':>12s}  {'E/atom (eV)':>14s}  {'dE (meV/atom)':>14s}
"""
for ec, e, de in zip(ecut_arr, en_ecut, de_ecut):
    marker = " <--" if ec == optimal_ecutwfc else ""
    report_text += f"    {ec:12.1f}  {e:14.6f}  {de:14.4f}{marker}\n"

report_text += f"""
  K-GRID CONVERGENCE (ecutwfc = {optimal_ecutwfc} Ry):
    {'k-grid':>8s}  {'E/atom (eV)':>14s}  {'dE (meV/atom)':>14s}
"""
for kg, e, de in zip(kg_arr, en_kg, de_kgrid):
    marker = " <--" if optimal_kgrid and int(kg) == optimal_kgrid else ""
    report_text += f"    {int(kg):5d}^3    {e:14.6f}  {de:14.4f}{marker}\n"

print(report_text)

# Save text report
with open(os.path.join(WORK_DIR, "convergence_report.txt"), "w") as f:
    f.write(report_text)

# Save JSON report
json_report = {
    "threshold_meV_per_atom": THRESHOLD_MEV,
    "nat": NAT,
    "ecutrho_ratio": ECUTRHO_RATIO,
    "optimal_ecutwfc_Ry": float(optimal_ecutwfc),
    "optimal_ecutwfc_eV": float(optimal_ecutwfc * RY_TO_EV),
    "optimal_kgrid": optimal_kgrid,
    "phase1_kgrid_fixed": list(KGRID_FOR_ECUT),
    "phase1_ecut_results": ecut_results,
    "phase2_ecutwfc_fixed_Ry": float(optimal_ecutwfc),
    "phase2_kgrid_results": kgrid_results,
}
json_path = os.path.join(WORK_DIR, "convergence_report.json")
with open(json_path, "w") as f:
    json.dump(json_report, f, indent=2)
print(f"  JSON report: {json_path}")
print(f"  Text report: {os.path.join(WORK_DIR, 'convergence_report.txt')}")

# ================================================================== #
#  Combined Plot
# ================================================================== #
fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))

# (a) ecutwfc convergence
ax = axes[0]
ax.plot(ecut_arr, np.abs(de_ecut), "o-", color="steelblue",
        linewidth=2, markersize=7, label="ecutwfc sweep")
ax.axhline(THRESHOLD_MEV, color="red", linestyle="--", linewidth=1.5,
           label=f"Threshold: {THRESHOLD_MEV} meV/atom")
if optimal_ecutwfc in ecut_arr:
    ax.axvline(optimal_ecutwfc, color="green", linestyle=":", linewidth=1.5,
               label=f"Optimal: {optimal_ecutwfc} Ry")
ax.set_xlabel("ecutwfc (Ry)", fontsize=13)
ax.set_ylabel("|dE| (meV/atom)", fontsize=13)
ax.set_title("(a) Cutoff Convergence", fontsize=14)
ax.set_yscale("log")
ax.legend(fontsize=10, loc="upper right")
ax.grid(True, alpha=0.3, which="both")

# (b) k-grid convergence
ax = axes[1]
ax.plot(kg_arr, np.abs(de_kgrid), "s-", color="darkorange",
        linewidth=2, markersize=7, label="k-grid sweep")
ax.axhline(THRESHOLD_MEV, color="red", linestyle="--", linewidth=1.5,
           label=f"Threshold: {THRESHOLD_MEV} meV/atom")
if optimal_kgrid is not None:
    ax.axvline(optimal_kgrid, color="green", linestyle=":", linewidth=1.5,
               label=f"Optimal: {optimal_kgrid}^3")
ax.set_xlabel("k-grid (NxNxN)", fontsize=13)
ax.set_ylabel("|dE| (meV/atom)", fontsize=13)
ax.set_title(f"(b) K-Point Convergence (ecutwfc={optimal_ecutwfc} Ry)", fontsize=14)
ax.set_yscale("log")
ax.legend(fontsize=10, loc="upper right")
ax.grid(True, alpha=0.3, which="both")

fig.suptitle(f"DFT Convergence Test -- Threshold: {THRESHOLD_MEV} meV/atom",
             fontsize=15, y=1.02)
plt.tight_layout()
plot_path = os.path.join(WORK_DIR, "convergence_combined.png")
plt.savefig(plot_path, dpi=200, bbox_inches="tight")
plt.close()
print(f"  Combined plot: {plot_path}")
```

---

### Method D: Batch Convergence for Multiple Materials

Runs the combined convergence workflow (Method C) for a list of materials. Each material gets its own convergence report. A summary CSV ranks materials by optimal parameters and computational cost.

#### Step D1: Multi-Material Convergence Automation

```python
#!/usr/bin/env python3
"""
Batch convergence testing for multiple materials.

For each material:
  1. Run ecutwfc convergence with a safe k-grid.
  2. Run k-grid convergence with the optimal ecutwfc.
  3. Record optimal parameters and timing.

Produces:
  - Per-material JSON reports and convergence plots.
  - A summary CSV with recommended parameters for all materials.
  - A combined overview plot.

No pyiron dependency -- standalone script.
"""
import os
import subprocess
import re
import json
import sys
import time
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ================================================================== #
#  CONFIGURATION
# ================================================================== #
PSEUDO_DIR = os.path.abspath("./pseudo")
BATCH_DIR = os.path.abspath("./convergence_batch")
PW_COMMAND = "pw.x"
MPI_COMMAND = "mpirun"
NPROC = 4

THRESHOLD_MEV = 1.0
ECUTWFC_VALUES = [20, 30, 40, 50, 60, 70, 80, 90, 100]
KGRID_VALUES = [2, 4, 6, 8, 10, 12, 14]
ECUTRHO_RATIO = 8
KGRID_FOR_ECUT = (8, 8, 8)    # well-converged k-grid for ecutwfc phase
FALLBACK_ECUTWFC = 60.0

RY_TO_EV = 13.605693123

# ================================================================== #
#  MATERIALS DEFINITIONS
# ================================================================== #
# Each entry defines a material by its QE input template.
# The template must accept {ecutwfc}, {ecutrho}, {kx}, {ky}, {kz} placeholders.
# It must also set nat (used to normalize energy per atom).

MATERIALS = [
    {
        "name": "Si",
        "formula": "Si",
        "nat": 2,
        "prefix": "si",
        "template": """\
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
  Si  0.00  0.00  0.00
  Si  0.25  0.25  0.25
K_POINTS (automatic)
  {kx} {ky} {kz}  0 0 0
""",
    },
    {
        "name": "Al_FCC",
        "formula": "Al",
        "nat": 1,
        "prefix": "al",
        "template": """\
&SYSTEM
    ibrav       = 2
    celldm(1)   = 7.63
    nat         = 1
    ntyp        = 1
    ecutwfc     = {ecutwfc}
    ecutrho     = {ecutrho}
    occupations = 'smearing'
    smearing    = 'cold'
    degauss     = 0.02
/
&ELECTRONS
    conv_thr = 1.0d-8
/
ATOMIC_SPECIES
  Al  26.9815  Al.pbe-n-rrkjus_psl.1.0.0.UPF
ATOMIC_POSITIONS (crystal)
  Al  0.00  0.00  0.00
K_POINTS (automatic)
  {kx} {ky} {kz}  0 0 0
""",
    },
    {
        "name": "NaCl_rocksalt",
        "formula": "NaCl",
        "nat": 2,
        "prefix": "nacl",
        "template": """\
&SYSTEM
    ibrav       = 2
    celldm(1)   = 10.68
    nat         = 2
    ntyp        = 2
    ecutwfc     = {ecutwfc}
    ecutrho     = {ecutrho}
    occupations = 'smearing'
    smearing    = 'cold'
    degauss     = 0.005
/
&ELECTRONS
    conv_thr = 1.0d-8
/
ATOMIC_SPECIES
  Na  22.9898  Na.pbe-spn-rrkjus_psl.1.0.0.UPF
  Cl  35.4530  Cl.pbe-n-rrkjus_psl.1.0.0.UPF
ATOMIC_POSITIONS (crystal)
  Na  0.00  0.00  0.00
  Cl  0.50  0.50  0.50
K_POINTS (automatic)
  {kx} {ky} {kz}  0 0 0
""",
    },
]

# ================================================================== #
#  QE runner
# ================================================================== #
def run_scf(material, ecutwfc, kgrid, tag, work_subdir):
    """Run a single QE SCF for a given material."""
    mat_dir = os.path.join(BATCH_DIR, material["name"])
    basedir = os.path.join(mat_dir, work_subdir)
    os.makedirs(basedir, exist_ok=True)
    outdir = os.path.join(basedir, tag)
    os.makedirs(outdir, exist_ok=True)
    ecutrho = ecutwfc * ECUTRHO_RATIO

    inp = f"""&CONTROL
    calculation = 'scf'
    prefix      = '{material["prefix"]}'
    outdir      = '{outdir}'
    pseudo_dir  = '{PSEUDO_DIR}'
    tprnfor     = .true.
/
{material["template"].format(
    ecutwfc=ecutwfc, ecutrho=ecutrho,
    kx=kgrid[0], ky=kgrid[1], kz=kgrid[2]
)}
"""
    infile = os.path.join(basedir, f"scf_{tag}.in")
    outfile = os.path.join(basedir, f"scf_{tag}.out")

    with open(infile, "w") as f:
        f.write(inp)

    cmd = []
    if MPI_COMMAND is not None:
        cmd = [MPI_COMMAND, "-np", str(NPROC)]
    cmd += [PW_COMMAND, "-in", infile]

    t0 = time.time()
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=1800)
    wall = time.time() - t0

    with open(outfile, "w") as f:
        f.write(proc.stdout)

    energy_ry = None
    n_kpts_irr = None
    scf_ok = False
    for line in proc.stdout.splitlines():
        if line.strip().startswith("!") and "total energy" in line:
            m = re.search(r"=\s+([-\d.]+)\s+Ry", line)
            if m:
                energy_ry = float(m.group(1))
        if "number of k points=" in line:
            m = re.search(r"=\s+(\d+)", line)
            if m:
                n_kpts_irr = int(m.group(1))
        if "convergence has been achieved" in line:
            scf_ok = True

    nat = material["nat"]
    energy_ev = energy_ry * RY_TO_EV if energy_ry is not None else None

    return {
        "ecutwfc_Ry": ecutwfc,
        "kgrid": list(kgrid),
        "n_kpts_irr": n_kpts_irr,
        "energy_eV": energy_ev,
        "energy_per_atom_eV": energy_ev / nat if energy_ev is not None else None,
        "scf_converged": scf_ok,
        "wall_time_s": round(wall, 2),
    }


def find_optimal(values, de_meV, threshold):
    """Return first value where |dE| <= threshold, or None."""
    for v, de in zip(values, de_meV):
        if abs(de) <= threshold:
            return v
    return None


def converge_parameter(material, param_name, param_values, ecutwfc, kgrid_base,
                       work_subdir):
    """
    Run convergence sweep for one parameter.
    Returns (optimal_value, param_arr, de_meV_arr, results_list).
    """
    results = []
    for val in param_values:
        if param_name == "ecutwfc":
            tag = f"ecut_{val:03.0f}"
            r = run_scf(material, val, kgrid_base, tag, work_subdir)
        elif param_name == "kgrid":
            tag = f"kgrid_{val:02d}"
            kgrid = (val, val, val)
            r = run_scf(material, ecutwfc, kgrid, tag, work_subdir)
        else:
            raise ValueError(f"Unknown parameter: {param_name}")
        results.append(r)

        label = f"ecutwfc={val} Ry" if param_name == "ecutwfc" else f"k={val}^3"
        if r["energy_per_atom_eV"] is not None:
            print(f"    {label:>20s}  E/atom = {r['energy_per_atom_eV']:.6f} eV  "
                  f"({r['wall_time_s']:.1f}s)")
        else:
            print(f"    {label:>20s}  FAILED")

    valid = [r for r in results if r["energy_per_atom_eV"] is not None]
    if not valid:
        return None, np.array([]), np.array([]), results

    if param_name == "ecutwfc":
        p_arr = np.array([r["ecutwfc_Ry"] for r in valid])
    else:
        p_arr = np.array([r["kgrid"][0] for r in valid])

    en_arr = np.array([r["energy_per_atom_eV"] for r in valid])
    de = (en_arr - en_arr[-1]) * 1000.0

    optimal = find_optimal(p_arr, de, THRESHOLD_MEV)
    return optimal, p_arr, de, results


# ================================================================== #
#  Run convergence for each material
# ================================================================== #
os.makedirs(BATCH_DIR, exist_ok=True)

summary_rows = []

for mat in MATERIALS:
    name = mat["name"]
    print(f"\n{'#' * 70}")
    print(f"  Material: {name} ({mat['formula']}, {mat['nat']} atoms)")
    print(f"{'#' * 70}")

    # Phase 1: ecutwfc
    print(f"\n  Phase 1: ecutwfc convergence "
          f"(k-grid = {KGRID_FOR_ECUT[0]}x{KGRID_FOR_ECUT[1]}x{KGRID_FOR_ECUT[2]})")
    opt_ecut, ecut_arr, de_ecut, ecut_results = converge_parameter(
        mat, "ecutwfc", ECUTWFC_VALUES, None, KGRID_FOR_ECUT, "phase1_ecutwfc"
    )
    if opt_ecut is None:
        opt_ecut = FALLBACK_ECUTWFC
        print(f"    WARNING: not converged, using fallback = {FALLBACK_ECUTWFC} Ry")
    else:
        print(f"    --> Optimal ecutwfc = {opt_ecut} Ry")

    # Phase 2: k-grid
    print(f"\n  Phase 2: k-grid convergence (ecutwfc = {opt_ecut} Ry)")
    opt_kgrid, kg_arr, de_kgrid, kgrid_results = converge_parameter(
        mat, "kgrid", KGRID_VALUES, opt_ecut, None, "phase2_kgrid"
    )
    if opt_kgrid is not None:
        opt_kgrid = int(opt_kgrid)
        print(f"    --> Optimal k-grid = {opt_kgrid}^3")
    else:
        print(f"    WARNING: k-grid not converged.")

    # Save per-material report
    mat_dir = os.path.join(BATCH_DIR, name)
    mat_report = {
        "material": name,
        "formula": mat["formula"],
        "nat": mat["nat"],
        "threshold_meV": THRESHOLD_MEV,
        "optimal_ecutwfc_Ry": float(opt_ecut) if opt_ecut else None,
        "optimal_kgrid": opt_kgrid,
        "ecut_results": ecut_results,
        "kgrid_results": kgrid_results,
    }
    with open(os.path.join(mat_dir, "convergence_report.json"), "w") as f:
        json.dump(mat_report, f, indent=2)

    # Per-material plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    if len(ecut_arr) > 0:
        ax1.plot(ecut_arr, np.abs(de_ecut), "o-", color="steelblue",
                 linewidth=2, markersize=7)
        ax1.axhline(THRESHOLD_MEV, color="red", linestyle="--", linewidth=1.5)
        if opt_ecut in ecut_arr:
            ax1.axvline(opt_ecut, color="green", linestyle=":", linewidth=1.5,
                        label=f"Optimal: {opt_ecut} Ry")
            ax1.legend(fontsize=10)
    ax1.set_xlabel("ecutwfc (Ry)", fontsize=13)
    ax1.set_ylabel("|dE| (meV/atom)", fontsize=13)
    ax1.set_title("Cutoff Convergence", fontsize=14)
    ax1.set_yscale("log")
    ax1.grid(True, alpha=0.3, which="both")

    if len(kg_arr) > 0:
        ax2.plot(kg_arr, np.abs(de_kgrid), "s-", color="darkorange",
                 linewidth=2, markersize=7)
        ax2.axhline(THRESHOLD_MEV, color="red", linestyle="--", linewidth=1.5)
        if opt_kgrid is not None:
            ax2.axvline(opt_kgrid, color="green", linestyle=":", linewidth=1.5,
                        label=f"Optimal: {opt_kgrid}^3")
            ax2.legend(fontsize=10)
    ax2.set_xlabel("k-grid (NxNxN)", fontsize=13)
    ax2.set_ylabel("|dE| (meV/atom)", fontsize=13)
    ax2.set_title("K-Point Convergence", fontsize=14)
    ax2.set_yscale("log")
    ax2.grid(True, alpha=0.3, which="both")

    fig.suptitle(f"{name} ({mat['formula']}) Convergence", fontsize=15, y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(mat_dir, "convergence.png"),
                dpi=200, bbox_inches="tight")
    plt.close()

    # Collect summary
    total_wall = sum(r["wall_time_s"] for r in ecut_results + kgrid_results
                     if r.get("wall_time_s"))
    summary_rows.append({
        "material": name,
        "formula": mat["formula"],
        "nat": mat["nat"],
        "optimal_ecutwfc_Ry": opt_ecut,
        "optimal_ecutwfc_eV": round(opt_ecut * RY_TO_EV, 1) if opt_ecut else None,
        "optimal_kgrid": opt_kgrid,
        "total_wall_time_s": round(total_wall, 1),
    })

# ================================================================== #
#  Batch summary
# ================================================================== #
print(f"\n\n{'=' * 70}")
print(f"  BATCH CONVERGENCE SUMMARY ({len(MATERIALS)} materials)")
print(f"  Threshold: {THRESHOLD_MEV} meV/atom")
print(f"{'=' * 70}")
print(f"  {'Material':>15s}  {'ecutwfc (Ry)':>12s}  {'ecutwfc (eV)':>12s}  "
      f"{'k-grid':>8s}  {'Total time (s)':>14s}")
print(f"  {'-'*15}  {'-'*12}  {'-'*12}  {'-'*8}  {'-'*14}")

for row in summary_rows:
    kg_str = f"{row['optimal_kgrid']}^3" if row["optimal_kgrid"] else "N/A"
    ec_ry = f"{row['optimal_ecutwfc_Ry']:.1f}" if row["optimal_ecutwfc_Ry"] else "N/A"
    ec_ev = f"{row['optimal_ecutwfc_eV']:.1f}" if row["optimal_ecutwfc_eV"] else "N/A"
    print(f"  {row['material']:>15s}  {ec_ry:>12s}  {ec_ev:>12s}  "
          f"{kg_str:>8s}  {row['total_wall_time_s']:>14.1f}")

# Save summary CSV
import csv
csv_path = os.path.join(BATCH_DIR, "convergence_summary.csv")
with open(csv_path, "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=summary_rows[0].keys())
    writer.writeheader()
    writer.writerows(summary_rows)
print(f"\n  Summary CSV: {csv_path}")

# Save summary JSON
with open(os.path.join(BATCH_DIR, "convergence_summary.json"), "w") as f:
    json.dump(summary_rows, f, indent=2)
print(f"  Summary JSON: {os.path.join(BATCH_DIR, 'convergence_summary.json')}")

# ================================================================== #
#  Overview plot (all materials on one figure)
# ================================================================== #
n_mats = len(summary_rows)
fig, ax = plt.subplots(figsize=(max(6, n_mats * 2), 5))

x = np.arange(n_mats)
names = [r["material"] for r in summary_rows]
ecuts = [r["optimal_ecutwfc_Ry"] or 0 for r in summary_rows]
kgrids = [r["optimal_kgrid"] or 0 for r in summary_rows]

bar_width = 0.35
bars1 = ax.bar(x - bar_width/2, ecuts, bar_width, color="steelblue",
               edgecolor="black", linewidth=0.5, label="ecutwfc (Ry)")
ax2 = ax.twinx()
bars2 = ax2.bar(x + bar_width/2, kgrids, bar_width, color="darkorange",
                edgecolor="black", linewidth=0.5, label="k-grid (N)")

ax.set_xlabel("Material", fontsize=13)
ax.set_ylabel("Optimal ecutwfc (Ry)", fontsize=13, color="steelblue")
ax2.set_ylabel("Optimal k-grid (N)", fontsize=13, color="darkorange")
ax.set_xticks(x)
ax.set_xticklabels(names, fontsize=11, rotation=30, ha="right")
ax.set_title(f"Converged Parameters (threshold = {THRESHOLD_MEV} meV/atom)",
             fontsize=14)

# Combined legend
lines1, labels1 = ax.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax.legend(lines1 + lines2, labels1 + labels2, fontsize=10, loc="upper left")
ax.grid(True, alpha=0.2)

plt.tight_layout()
overview_path = os.path.join(BATCH_DIR, "convergence_overview.png")
plt.savefig(overview_path, dpi=200, bbox_inches="tight")
plt.close()
print(f"  Overview plot: {overview_path}")
```

---

## Key Parameters

| Parameter | Default | Description |
|---|---|---|
| `ECUTWFC_MIN` / `ECUTWFC_MAX` | 20 / 100 Ry | Range for cutoff sweep (pyiron uses `min`/`max`/`num_points`) |
| `ECUTWFC_NUM_POINTS` | 17 | Number of evenly-spaced points in the sweep |
| `ECUTRHO_RATIO` | 8 | ecutrho = ratio * ecutwfc (4 for NC, 8-12 for US/PAW) |
| `KGRID_MIN` / `KGRID_MAX` / `KGRID_STEP` | 2 / 16 / 2 | Range and step for k-grid sweep (pyiron uses `min`/`max`/`steps`) |
| `THRESHOLD_MEV_PER_ATOM` | 1.0 | Energy per atom change threshold in meV/atom |
| `KGRID_FOR_ECUT` | (8, 8, 8) | Fixed k-grid used during ecutwfc convergence (must be well-converged) |
| `FALLBACK_ECUTWFC` | 60.0 Ry | Used if ecutwfc sweep does not converge within the tested range |
| `NPROC` | 4 | MPI processes per calculation |

## Interpreting Results

- **Monotonic energy decrease with cutoff**: Expected behavior. The plane-wave basis becomes more complete at higher cutoff, lowering the total energy.
- **Oscillatory behavior with k-grid**: Can occur when symmetry causes different k-grids to sample different irreducible k-points. Use `nosym=.true.` for a cleaner test, then re-enable symmetry for production.
- **Optimal cutoff below pseudopotential recommendation**: The pseudopotential file header often lists a recommended cutoff. If your converged cutoff is lower, use the pseudopotential recommendation instead.
- **k-grid vs. cell size**: Larger cells need coarser k-grids. A rule of thumb: k * a = constant (where a is the lattice parameter). For a 2x2x2 supercell, halve the k-grid in each direction.
- **Threshold selection**: 1 meV/atom is publication quality. 5 meV/atom is acceptable for screening. 0.1 meV/atom is needed for phonon calculations or equation of state fitting.
- **Combined workflow rationale**: The cutoff and k-grid are weakly coupled. Converging the cutoff first with a safe k-grid, then converging the k-grid with the optimal cutoff, avoids a costly 2D parameter sweep while producing reliable results.

## Common Issues

| Problem | Cause | Solution |
|---|---|---|
| All calculations fail | Wrong pseudopotential path or filename | Verify `PSEUDO_DIR` and that UPF files exist with expected names |
| Energy not monotonic with cutoff | ecutrho/ecutwfc ratio too low for ultrasoft PP | Increase `ECUTRHO_RATIO` from 4 to 8 or 12 |
| k-grid convergence oscillates | Gamma-point inclusion changes with odd/even grids | Test only even grids, or use `nosym=.true.` |
| Optimal cutoff at the upper boundary | Sweep range is too narrow | Increase `ECUTWFC_MAX` and re-run |
| Very slow for large cells | Many atoms + dense k-grids | Use k-point parallelism (`-nk` flag), reduce `KGRID_FOR_ECUT` |
| Batch mode fails on one material | Pseudopotential missing for that element | Add the missing UPF to `./pseudo/` or remove the material from `MATERIALS` |
| JSON report has null energies | Calculation crashed (check `.out` file) | Look for "Error in routine" in the output; fix input and re-run |
| Threshold too tight (0.01 meV/atom) | May never converge at feasible parameters | Use 0.1-1.0 meV/atom for standard calculations; tighter only for phonons or EOS |
