# SCF Calculation and Structure Relaxation

## When to Use

- You have a crystal structure and need to optimize atomic positions and/or cell parameters.
- You need the DFT total energy, forces, or stress tensor for a given structure.
- You are preparing a relaxed structure as input for band structure or DOS calculations.

## Method Selection (MACE vs QE)

| Criterion | ASE + MACE | QE DFT |
|---|---|---|
| Speed | Seconds | Minutes to hours |
| Accuracy | Good for structures, approximate for energies | Reference-quality |
| Electronic properties | Not available | Available (eigenvalues, charge density) |
| Cell optimization | Supported (FrechetCellFilter) | Supported (vc-relax) |
| Use case | Quick screening, initial relaxation | Publication results, input for bands/DOS |

**Recommendation**: Use MACE first to get a good starting geometry, then refine with QE if DFT accuracy is needed.

## Prerequisites

- A crystal structure in any common format (CIF, POSCAR, extxyz, or built programmatically).
- For QE: pseudopotential files (download script provided below).
- Python packages: `ase`, `mace-torch`, `pymatgen`, `numpy`.

---

## Detailed Steps

### Method A: ASE + MACE Relaxation

This is the fastest way to relax a structure. MACE is a universal machine learning interatomic potential that covers most of the periodic table.

```python
#!/usr/bin/env python3
"""
Relax a crystal structure using ASE + MACE.
This example uses silicon as a demo. Replace with your own structure.
"""
import warnings
warnings.filterwarnings("ignore")

from ase.build import bulk
from ase.io import read, write
from ase.optimize import LBFGS
from ase.filters import FrechetCellFilter
from mace.calculators import mace_mp
import numpy as np

# ── 1. Build or load structure ──────────────────────────────────────
# Option A: build a known bulk structure
atoms = bulk("Si", "diamond", a=5.43)

# Option B: read from a CIF file
# atoms = read("my_structure.cif")

# Option C: read from a POSCAR
# atoms = read("POSCAR", format="vasp")

print(f"Input structure: {atoms.get_chemical_formula()}")
print(f"Cell volume: {atoms.get_volume():.3f} A^3")

# ── 2. Attach MACE calculator ──────────────────────────────────────
# model="medium" is a good balance of speed and accuracy.
# Options: "small", "medium", "large"
# device="cpu" or "cuda" if GPU is available.
calc = mace_mp(model="medium", dispersion=False, device="cpu")
atoms.calc = calc

# ── 3a. Relax atomic positions only (fixed cell) ───────────────────
opt = LBFGS(atoms, logfile="relax_fixcell.log")
opt.run(fmax=0.01)  # eV/Angstrom
print(f"\nAfter fixed-cell relaxation:")
print(f"  Energy: {atoms.get_potential_energy():.6f} eV")
print(f"  Max force: {np.max(np.abs(atoms.get_forces())):.6f} eV/A")

# ── 3b. Relax both positions AND cell (variable cell) ──────────────
atoms2 = bulk("Si", "diamond", a=5.43)  # fresh copy
atoms2.calc = calc
filtered = FrechetCellFilter(atoms2)
opt2 = LBFGS(filtered, logfile="relax_varcell.log")
opt2.run(fmax=0.01)

print(f"\nAfter variable-cell relaxation:")
print(f"  Energy: {atoms2.get_potential_energy():.6f} eV")
print(f"  Lattice parameter a: {atoms2.cell.cellpar()[0]:.4f} A")
print(f"  Cell volume: {atoms2.get_volume():.3f} A^3")
print(f"  Max force: {np.max(np.abs(atoms2.get_forces())):.6f} eV/A")
stress_GPa = atoms2.get_stress(voigt=True) * 160.2176634  # eV/A^3 -> GPa
print(f"  Stress (GPa): {np.array2string(stress_GPa, precision=3)}")

# ── 4. Save the relaxed structure ───────────────────────────────────
write("relaxed_MACE.cif", atoms2)
write("relaxed_MACE.vasp", atoms2, format="vasp")
write("relaxed_MACE.xyz", atoms2, format="extxyz")
print("\nRelaxed structure saved to relaxed_MACE.cif / .vasp / .xyz")
```

---

### Method B: Quantum ESPRESSO DFT

#### Step B0: Download pseudopotentials (run once)

```python
#!/usr/bin/env python3
"""
Download SSSP Efficiency pseudopotentials for Quantum ESPRESSO.
Run this once. Pseudopotentials are saved to ./pseudo/
"""
import os
import json
import urllib.request

PSEUDO_DIR = os.path.abspath("./pseudo")
os.makedirs(PSEUDO_DIR, exist_ok=True)

# SSSP Efficiency 1.3.0 metadata
SSSP_URL = "https://raw.githubusercontent.com/aiidateam/pseudo-data/main/sssp/sssp_efficiency_1.3.0_pbe.json"

print("Downloading SSSP metadata...")
try:
    with urllib.request.urlopen(SSSP_URL, timeout=30) as resp:
        sssp_data = json.loads(resp.read().decode())
except Exception as e:
    print(f"Could not download SSSP metadata: {e}")
    print("Falling back to manual pseudopotential specification.")
    sssp_data = {}

BASE = "https://raw.githubusercontent.com/aiidateam/pseudo-data/main/sssp/pseudos/"

def download_pseudo(element):
    """Download pseudopotential for a single element."""
    if element in sssp_data:
        fname = sssp_data[element]["filename"]
    else:
        # Common fallback naming convention for SSSP PBE
        fname = f"{element}.upf"
    dest = os.path.join(PSEUDO_DIR, fname)
    if os.path.exists(dest):
        print(f"  {fname} already exists, skipping.")
        return fname
    url = BASE + fname
    print(f"  Downloading {fname} ...")
    try:
        urllib.request.urlretrieve(url, dest)
    except Exception as e:
        print(f"  WARNING: Could not download {fname}: {e}")
        print(f"  You may need to manually place a UPF for {element} in {PSEUDO_DIR}")
    return fname

# Download for the elements you need (example: Si)
elements_needed = ["Si"]
pseudo_map = {}
for el in elements_needed:
    pseudo_map[el] = download_pseudo(el)

print(f"\nPseudopotentials in {PSEUDO_DIR}:")
for f in sorted(os.listdir(PSEUDO_DIR)):
    print(f"  {f}")
```

**Alternative: direct wget approach** (simpler, if you know the filename):

```bash
mkdir -p ./pseudo
# SSSP Efficiency pseudopotentials (PBE, PAW or ultrasoft)
wget -q -nc -P ./pseudo "https://pseudopotentials.quantum-espresso.org/upf_files/Si.pbe-n-rrkjus_psl.1.0.0.UPF"
```

#### Step B1: QE SCF Calculation

```python
#!/usr/bin/env python3
"""
Generate and run a QE SCF input file.
Demonstrates a complete workflow for silicon.
"""
import os
import subprocess
import numpy as np

PSEUDO_DIR = os.path.abspath("./pseudo")
OUTDIR = os.path.abspath("./tmp")
os.makedirs(OUTDIR, exist_ok=True)

# ── Structure parameters (silicon diamond) ──────────────────────────
alat = 10.26  # lattice parameter in Bohr (5.43 Angstrom * 1.8897)
ecutwfc = 40.0  # plane-wave cutoff in Ry
ecutrho = 320.0  # charge density cutoff in Ry (8x ecutwfc for ultrasoft)
k_grid = (6, 6, 6)

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
    celldm(1)   = {alat}
    nat         = 2
    ntyp        = 1
    ecutwfc     = {ecutwfc}
    ecutrho     = {ecutrho}
    occupations = 'smearing'
    smearing    = 'cold'
    degauss     = 0.01
/

&ELECTRONS
    conv_thr    = 1.0d-8
    mixing_beta = 0.7
/

ATOMIC_SPECIES
  Si  28.0855  Si.pbe-n-rrkjus_psl.1.0.0.UPF

ATOMIC_POSITIONS (crystal)
  Si  0.00  0.00  0.00
  Si  0.25  0.25  0.25

K_POINTS (automatic)
  {k_grid[0]} {k_grid[1]} {k_grid[2]}  0 0 0
"""

# Write input file
with open("si_scf.in", "w") as f:
    f.write(scf_input)
print("Written: si_scf.in")

# Run QE (adjust -nk for k-point parallelism)
print("Running pw.x ...")
result = subprocess.run(
    ["mpirun", "-np", "4", "pw.x", "-in", "si_scf.in"],
    capture_output=True, text=True, timeout=600
)

with open("si_scf.out", "w") as f:
    f.write(result.stdout)
print("Output written to si_scf.out")

if result.returncode != 0:
    print("ERROR: pw.x failed!")
    print(result.stderr[-500:] if result.stderr else "No stderr")
else:
    print("SCF completed successfully.")
```

#### Step B2: Parse QE SCF Output

```python
#!/usr/bin/env python3
"""
Parse key quantities from a QE SCF output file.
"""
import re

def parse_qe_scf(filename):
    """Parse total energy, forces, stress, and convergence from QE output."""
    results = {
        "converged": False,
        "total_energy_Ry": None,
        "total_energy_eV": None,
        "forces": [],
        "max_force_Ry_au": None,
        "stress_kbar": [],
        "pressure_kbar": None,
        "n_scf_steps": 0,
        "fermi_energy_eV": None,
    }

    with open(filename, "r") as f:
        lines = f.readlines()

    for i, line in enumerate(lines):
        # Convergence
        if "convergence has been achieved" in line:
            results["converged"] = True
            m = re.search(r"in\s+(\d+)\s+iterations", line)
            if m:
                results["n_scf_steps"] = int(m.group(1))

        # Total energy
        if line.strip().startswith("!") and "total energy" in line:
            m = re.search(r"=\s+([-\d.]+)\s+Ry", line)
            if m:
                e_ry = float(m.group(1))
                results["total_energy_Ry"] = e_ry
                results["total_energy_eV"] = e_ry * 13.605693123  # Ry -> eV

        # Forces
        if "force =" in line and "atom" in line:
            m = re.search(r"force\s*=\s*([-\d.]+)\s+([-\d.]+)\s+([-\d.]+)", line)
            if m:
                results["forces"].append([float(m.group(j)) for j in range(1, 4)])

        # Total force
        if "Total force" in line:
            m = re.search(r"Total force\s*=\s*([\d.]+)", line)
            if m:
                results["max_force_Ry_au"] = float(m.group(1))

        # Stress tensor
        if "total   stress" in line:
            m = re.search(r"P=\s*([-\d.]+)", line)
            if m:
                results["pressure_kbar"] = float(m.group(1))
            # Next 3 lines contain the stress tensor
            for j in range(1, 4):
                if i + j < len(lines):
                    parts = lines[i + j].split()
                    if len(parts) >= 3:
                        results["stress_kbar"].append(
                            [float(parts[0]), float(parts[1]), float(parts[2])]
                        )

        # Fermi energy
        if "the Fermi energy is" in line:
            m = re.search(r"is\s+([-\d.]+)\s+ev", line, re.IGNORECASE)
            if m:
                results["fermi_energy_eV"] = float(m.group(1))

    return results


# ── Use the parser ──────────────────────────────────────────────────
results = parse_qe_scf("si_scf.out")

print("=== QE SCF Results ===")
print(f"Converged:       {results['converged']}")
print(f"SCF iterations:  {results['n_scf_steps']}")
print(f"Total energy:    {results['total_energy_Ry']:.8f} Ry"
      f"  =  {results['total_energy_eV']:.6f} eV")
if results["fermi_energy_eV"] is not None:
    print(f"Fermi energy:    {results['fermi_energy_eV']:.4f} eV")
if results["max_force_Ry_au"] is not None:
    print(f"Total force:     {results['max_force_Ry_au']:.6f} Ry/au")
if results["pressure_kbar"] is not None:
    print(f"Pressure:        {results['pressure_kbar']:.2f} kbar")
if results["stress_kbar"]:
    print("Stress tensor (kbar):")
    for row in results["stress_kbar"]:
        print(f"  {row[0]:10.2f} {row[1]:10.2f} {row[2]:10.2f}")
```

#### Step B3: QE Variable-Cell Relaxation (vc-relax)

```python
#!/usr/bin/env python3
"""
Generate and run a QE vc-relax input file.
Relaxes both atomic positions and cell parameters.
"""
import os
import subprocess

PSEUDO_DIR = os.path.abspath("./pseudo")
OUTDIR = os.path.abspath("./tmp")
os.makedirs(OUTDIR, exist_ok=True)

alat = 10.26  # Bohr
ecutwfc = 40.0
ecutrho = 320.0

vcrelax_input = f"""&CONTROL
    calculation  = 'vc-relax'
    prefix       = 'si'
    outdir       = '{OUTDIR}'
    pseudo_dir   = '{PSEUDO_DIR}'
    tprnfor      = .true.
    tstress      = .true.
    forc_conv_thr = 1.0d-4
    etot_conv_thr = 1.0d-6
/

&SYSTEM
    ibrav       = 2
    celldm(1)   = {alat}
    nat         = 2
    ntyp        = 1
    ecutwfc     = {ecutwfc}
    ecutrho     = {ecutrho}
    occupations = 'smearing'
    smearing    = 'cold'
    degauss     = 0.01
/

&ELECTRONS
    conv_thr    = 1.0d-8
    mixing_beta = 0.7
/

&IONS
    ion_dynamics = 'bfgs'
/

&CELL
    cell_dynamics = 'bfgs'
    press         = 0.0
    press_conv_thr = 0.5
/

ATOMIC_SPECIES
  Si  28.0855  Si.pbe-n-rrkjus_psl.1.0.0.UPF

ATOMIC_POSITIONS (crystal)
  Si  0.00  0.00  0.00
  Si  0.25  0.25  0.25

K_POINTS (automatic)
  6 6 6  0 0 0
"""

with open("si_vcrelax.in", "w") as f:
    f.write(vcrelax_input)
print("Written: si_vcrelax.in")

print("Running pw.x for vc-relax ...")
result = subprocess.run(
    ["mpirun", "-np", "4", "pw.x", "-in", "si_vcrelax.in"],
    capture_output=True, text=True, timeout=1800
)

with open("si_vcrelax.out", "w") as f:
    f.write(result.stdout)
print("Output written to si_vcrelax.out")

if result.returncode != 0:
    print("ERROR: pw.x vc-relax failed!")
    print(result.stderr[-500:] if result.stderr else "No stderr")
else:
    print("vc-relax completed successfully.")
```

#### Step B4: Extract Relaxed Structure from vc-relax Output

```python
#!/usr/bin/env python3
"""
Extract the final relaxed structure from a QE vc-relax output
and convert it to CIF, POSCAR, and a new QE input.
"""
from pymatgen.io.pwscf import PWInput, PWOutput
from ase.io.espresso import read_espresso_out
from ase.io import write

# ── Method 1: Using ASE ────────────────────────────────────────────
# read_espresso_out reads the LAST configuration from the output
atoms = read_espresso_out("si_vcrelax.out")
print(f"Relaxed cell parameters: {atoms.cell.cellpar()}")
print(f"Relaxed volume: {atoms.get_volume():.4f} A^3")

write("si_relaxed.cif", atoms)
write("si_relaxed.vasp", atoms, format="vasp")
print("Saved: si_relaxed.cif, si_relaxed.vasp")

# ── Method 2: Manual extraction for new QE input ───────────────────
import re
import numpy as np

def extract_final_structure_qe(filename):
    """Extract final cell and positions from QE vc-relax output."""
    with open(filename, "r") as f:
        text = f.read()

    # Find the last CELL_PARAMETERS block
    cell_blocks = re.findall(
        r"CELL_PARAMETERS\s*\(angstrom\)\s*\n((?:\s*[-\d.]+\s+[-\d.]+\s+[-\d.]+\s*\n){3})",
        text
    )
    # Find the last ATOMIC_POSITIONS block
    pos_blocks = re.findall(
        r"ATOMIC_POSITIONS\s*\(crystal\)\s*\n((?:\s*\w+\s+[-\d.]+\s+[-\d.]+\s+[-\d.]+.*\n)+)",
        text
    )

    if not cell_blocks or not pos_blocks:
        print("Could not find final structure in output.")
        return None, None

    # Parse last cell
    cell_lines = cell_blocks[-1].strip().split("\n")
    cell = np.array([[float(x) for x in line.split()] for line in cell_lines])

    # Parse last positions
    pos_lines = pos_blocks[-1].strip().split("\n")
    species = []
    positions = []
    for line in pos_lines:
        parts = line.split()
        species.append(parts[0])
        positions.append([float(parts[1]), float(parts[2]), float(parts[3])])

    return cell, list(zip(species, positions))

cell, atoms_list = extract_final_structure_qe("si_vcrelax.out")
if cell is not None:
    print("\nFinal cell (Angstrom):")
    for row in cell:
        print(f"  {row[0]:12.8f} {row[1]:12.8f} {row[2]:12.8f}")
    print("Final atomic positions (crystal):")
    for sp, pos in atoms_list:
        print(f"  {sp}  {pos[0]:.8f}  {pos[1]:.8f}  {pos[2]:.8f}")
```

#### Step B5: Convergence Testing

```python
#!/usr/bin/env python3
"""
Convergence testing for ecutwfc and k-points.
Runs a series of SCF calculations and plots energy vs. parameter.
"""
import os
import subprocess
import re
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

PSEUDO_DIR = os.path.abspath("./pseudo")
OUTDIR_BASE = os.path.abspath("./tmp_conv")

def run_scf(ecutwfc, kgrid, tag):
    """Run a single SCF and return total energy in Ry."""
    outdir = os.path.join(OUTDIR_BASE, tag)
    os.makedirs(outdir, exist_ok=True)

    inp = f"""&CONTROL
    calculation = 'scf'
    prefix      = 'si'
    outdir      = '{outdir}'
    pseudo_dir  = '{PSEUDO_DIR}'
/
&SYSTEM
    ibrav       = 2
    celldm(1)   = 10.26
    nat         = 2
    ntyp        = 1
    ecutwfc     = {ecutwfc}
    ecutrho     = {ecutwfc * 8}
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
  {kgrid} {kgrid} {kgrid}  0 0 0
"""
    infile = f"scf_{tag}.in"
    outfile = f"scf_{tag}.out"
    with open(infile, "w") as f:
        f.write(inp)

    result = subprocess.run(
        ["mpirun", "-np", "2", "pw.x", "-in", infile],
        capture_output=True, text=True, timeout=600
    )
    with open(outfile, "w") as f:
        f.write(result.stdout)

    # Extract total energy
    for line in result.stdout.split("\n"):
        if line.strip().startswith("!") and "total energy" in line:
            m = re.search(r"=\s+([-\d.]+)\s+Ry", line)
            if m:
                return float(m.group(1))
    return None

# ── Test 1: ecutwfc convergence (fixed k=6) ────────────────────────
ecutwfc_values = [20, 30, 40, 50, 60, 70, 80, 90, 100]
energies_ecut = []
print("=== ecutwfc convergence test ===")
for ec in ecutwfc_values:
    e = run_scf(ec, 6, f"ecut{ec}")
    energies_ecut.append(e)
    if e is not None:
        print(f"  ecutwfc = {ec:3d} Ry  ->  E = {e:.8f} Ry")
    else:
        print(f"  ecutwfc = {ec:3d} Ry  ->  FAILED")

# ── Test 2: k-grid convergence (fixed ecutwfc=50) ──────────────────
kgrid_values = [2, 4, 6, 8, 10, 12]
energies_kgrid = []
print("\n=== k-grid convergence test ===")
for kg in kgrid_values:
    e = run_scf(50, kg, f"kgrid{kg}")
    energies_kgrid.append(e)
    if e is not None:
        print(f"  k-grid = {kg:2d}x{kg}x{kg}  ->  E = {e:.8f} Ry")
    else:
        print(f"  k-grid = {kg:2d}x{kg}x{kg}  ->  FAILED")

# ── Plot results ────────────────────────────────────────────────────
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

# ecutwfc plot
valid_ecut = [(ec, e) for ec, e in zip(ecutwfc_values, energies_ecut) if e is not None]
if valid_ecut:
    ecs, es = zip(*valid_ecut)
    es_meV = [(e - es[-1]) * 13605.693 for e in es]  # relative to most converged, in meV
    ax1.plot(ecs, es_meV, "o-", color="steelblue", linewidth=2, markersize=8)
    ax1.axhline(y=1.0, color="red", linestyle="--", alpha=0.7, label="1 meV threshold")
    ax1.axhline(y=-1.0, color="red", linestyle="--", alpha=0.7)
    ax1.set_xlabel("ecutwfc (Ry)", fontsize=13)
    ax1.set_ylabel("Energy relative to highest cutoff (meV/atom)", fontsize=13)
    ax1.set_title("Plane-wave cutoff convergence", fontsize=14)
    ax1.legend(fontsize=11)
    ax1.grid(True, alpha=0.3)

# k-grid plot
valid_kgrid = [(kg, e) for kg, e in zip(kgrid_values, energies_kgrid) if e is not None]
if valid_kgrid:
    kgs, es = zip(*valid_kgrid)
    es_meV = [(e - es[-1]) * 13605.693 for e in es]
    ax2.plot(kgs, es_meV, "s-", color="darkorange", linewidth=2, markersize=8)
    ax2.axhline(y=1.0, color="red", linestyle="--", alpha=0.7, label="1 meV threshold")
    ax2.axhline(y=-1.0, color="red", linestyle="--", alpha=0.7)
    ax2.set_xlabel("k-grid (NxNxN)", fontsize=13)
    ax2.set_ylabel("Energy relative to densest grid (meV/atom)", fontsize=13)
    ax2.set_title("k-point grid convergence", fontsize=14)
    ax2.legend(fontsize=11)
    ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("convergence_test.png", dpi=150, bbox_inches="tight")
print("\nPlot saved: convergence_test.png")
```

---

## Key Parameters

| Parameter | Typical Value | Notes |
|---|---|---|
| `ecutwfc` | 40--80 Ry | Depends on pseudopotential. Always converge. |
| `ecutrho` | 4x--12x `ecutwfc` | 4x for norm-conserving, 8--12x for ultrasoft/PAW |
| `k-grid` | 6x6x6 to 12x12x12 | Denser for metals; sparser OK for large cells |
| `conv_thr` | 1.0d-8 | SCF energy convergence (Ry). Tighter for forces. |
| `degauss` | 0.005--0.02 Ry | Smearing width. Smaller = more accurate but harder to converge. |
| `smearing` | `'cold'` or `'mp'` | Marzari-Vanderbilt (cold) recommended for metals |
| `forc_conv_thr` | 1.0d-4 Ry/Bohr | Force convergence for relax/vc-relax |
| `press_conv_thr` | 0.5 kbar | Pressure convergence for vc-relax |
| `mixing_beta` | 0.3--0.7 | Charge mixing. Lower = more stable but slower. |
| `nbnd` | auto or manual | Number of Kohn-Sham bands. QE auto-sets to n_electrons/2 (+ a few). **For doped/substituted systems, you MUST set this manually** — see Common Issues below. |

## Interpreting Results

- **Total energy**: Look for the line starting with `!    total energy`. This is the converged energy in Ry.
- **Forces**: Lines with `atom   N  type  X  force =`. Units are Ry/Bohr. Multiply by 25.711 to get eV/Angstrom.
- **Stress**: The `total   stress` block gives the stress tensor in kbar. `P=` gives the hydrostatic pressure.
- **Convergence**: Look for `convergence has been achieved in N iterations`. If you see `convergence NOT achieved`, reduce `mixing_beta` or increase `electron_maxstep`.
- **vc-relax completion**: Look for `End final coordinates` or `bfgs converged`.

## Common Issues

| Problem | Solution |
|---|---|
| SCF not converging | Reduce `mixing_beta` to 0.3 or 0.2. Try `mixing_mode = 'local-TF'`. Increase `electron_maxstep`. |
| vc-relax oscillating | Use a tighter `conv_thr` (1.0d-10). Reduce `press_conv_thr`. Start from MACE-relaxed structure. |
| Pseudopotential not found | Check `pseudo_dir` path is absolute. Verify the UPF filename matches `ATOMIC_SPECIES`. |
| Negative frequencies after relax | Structure may be at a saddle point. Try breaking symmetry slightly and re-relaxing. |
| Memory issues | Reduce `ecutwfc`, reduce k-points, or increase MPI ranks. Use `disk_io = 'low'`. |
| Forces too large after "convergence" | The SCF converged but ionic relaxation did not. Run more BFGS steps or restart from the last geometry. |
| `too few bands` error | Happens when aliovalent substitution changes the electron count (e.g., La³⁺ replacing Na⁺ adds 2 electrons). **Fix**: read each element's `z_valence` from its UPF header, sum across all atoms to get `n_electrons`, then set `nbnd = int(n_electrons / 2 * 1.2) + 4`. Always do this for doped/substituted supercells. |
