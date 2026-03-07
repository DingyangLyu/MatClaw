# Batch DFT Calculations

## When to Use

- You have a set of crystal structures (CIF, POSCAR, or pymatgen objects) and need to run the same type of DFT calculation on all of them.
- You want to generate standardized input files for Quantum ESPRESSO or VASP using pymatgen input sets (MPRelaxSet, MPStaticSet).
- You need parallel execution of multiple independent DFT jobs with progress tracking.
- You want automatic result collection, tabulation, error detection, and restart logic for failed calculations.

## Method Selection

| Approach | Code | Input Sets | Use When |
|---|---|---|---|
| QE batch (direct) | Quantum ESPRESSO `pw.x` | pymatgen `PWInput` | QE is installed and available in the container |
| VASP batch (external) | VASP `vasp_std` | pymatgen `MPRelaxSet`, `MPStaticSet` | VASP is available via external access (future) |
| MACE batch (local) | ASE + MACE-MP-0 | None (Python API) | Fast pre-screening, no DFT needed |

## Prerequisites

- A directory of structure files (CIF, POSCAR, or any format readable by pymatgen).
- For QE: Quantum ESPRESSO 7.5 installed (`pw.x` in PATH), SSSP pseudopotentials downloaded.
- For VASP: VASP binary and POTCAR files available (future external access).
- Python packages: `pymatgen`, `ase`, `numpy`, `pandas`, `matplotlib`.
- Optional: `pip install custodian` for error-correcting job management.

## Detailed Steps

### Workflow A: Quantum ESPRESSO Batch Calculations

#### Step A1 -- Generate QE input files for all structures

```python
#!/usr/bin/env python3
"""
Generate Quantum ESPRESSO input files for batch calculations.

Reads all structure files from an input directory, generates standardized
pw.x input files using pymatgen, and creates a batch run script.

Supports: relax, vc-relax, scf, bands calculations.
"""

import os
import json
import glob
import numpy as np
from pymatgen.core import Structure
from pymatgen.io.pwscf import PWInput
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer

# ================================================================== #
#  CONFIGURATION
# ================================================================== #
INPUT_DIR = "structures"          # directory with CIF/POSCAR files
WORK_DIR = "batch_qe"            # output directory for QE calculations
PSEUDO_DIR = "./pseudo"           # pseudopotential directory (absolute or relative)
CALCULATION = "vc-relax"          # "scf", "relax", "vc-relax", "bands"
ECUTWFC = 60.0                    # Ry
ECUTRHO = 480.0                   # Ry
KPOINT_DENSITY = 40               # k-points per reciprocal Angstrom
SMEARING = "mv"                   # "mv", "cold", "gauss", "mp"
DEGAUSS = 0.02                    # Ry
CONV_THR = 1.0e-8                 # SCF convergence (Ry)
FORC_CONV_THR = 1.0e-4            # force convergence (Ry/bohr)
NPROCS = 4                        # MPI processes per job
MAX_CONCURRENT = 1                # max simultaneous jobs (1 = serial)
# ================================================================== #

os.makedirs(WORK_DIR, exist_ok=True)

# Standard SSSP pseudopotential naming convention
# Adjust this map if your pseudopotential files have different names
def get_pseudo_name(element_symbol):
    """Return the SSSP pseudopotential filename for an element."""
    return f"{element_symbol}.pbe-n-rrkjus_psl.1.0.0.UPF"


def auto_kgrid(structure, density=KPOINT_DENSITY):
    """Compute k-point grid from lattice parameters and target density."""
    abc = structure.lattice.abc
    return tuple(max(1, int(round(density / a))) for a in abc)


def generate_qe_input(structure, calc_dir, prefix, calculation=CALCULATION):
    """Generate a complete QE pw.x input file."""
    os.makedirs(calc_dir, exist_ok=True)

    # Pseudopotential map
    pseudo_map = {}
    for el in structure.composition.elements:
        pseudo_map[el.symbol] = get_pseudo_name(el.symbol)

    kpts = auto_kgrid(structure)

    control = {
        "calculation": calculation,
        "prefix": prefix,
        "outdir": "./tmp",
        "pseudo_dir": os.path.abspath(PSEUDO_DIR),
        "tprnfor": True,
        "tstress": True,
        "etot_conv_thr": 1.0e-6,
    }

    system = {
        "ecutwfc": ECUTWFC,
        "ecutrho": ECUTRHO,
        "occupations": "smearing",
        "smearing": SMEARING,
        "degauss": DEGAUSS,
    }

    electrons = {
        "conv_thr": CONV_THR,
        "mixing_beta": 0.4,
    }

    # Add relaxation-specific blocks
    ions = {}
    cell = {}
    if calculation in ("relax", "vc-relax"):
        control["forc_conv_thr"] = FORC_CONV_THR
        ions["ion_dynamics"] = "bfgs"
    if calculation == "vc-relax":
        cell["cell_dynamics"] = "bfgs"
        cell["press_conv_thr"] = 0.1

    pw_input = PWInput(
        structure,
        pseudo=pseudo_map,
        control=control,
        system=system,
        electrons=electrons,
        ions=ions if ions else None,
        cell=cell if cell else None,
        kpoints_grid=kpts,
    )

    input_file = os.path.join(calc_dir, f"{calculation}.in")
    pw_input.write_file(input_file)
    return input_file, kpts


# --- Process all structures ---
structure_files = sorted(
    glob.glob(os.path.join(INPUT_DIR, "*.cif"))
    + glob.glob(os.path.join(INPUT_DIR, "*.vasp"))
    + glob.glob(os.path.join(INPUT_DIR, "POSCAR*"))
    + glob.glob(os.path.join(INPUT_DIR, "*.json"))
)

if not structure_files:
    print(f"No structure files found in {INPUT_DIR}/")
    print("Supported formats: .cif, .vasp, POSCAR*, .json (pymatgen)")
    exit(1)

job_metadata = []
for idx, filepath in enumerate(structure_files):
    basename = os.path.splitext(os.path.basename(filepath))[0]
    # Sanitize prefix for QE (alphanumeric + underscore only)
    prefix = basename.replace("-", "_").replace(" ", "_")

    try:
        structure = Structure.from_file(filepath)
    except Exception as e:
        print(f"  ERROR reading {filepath}: {e}")
        continue

    calc_dir = os.path.join(WORK_DIR, f"{idx:03d}_{prefix}")
    input_file, kpts = generate_qe_input(structure, calc_dir, prefix)

    # Save structure as CIF for reference
    structure.to(os.path.join(calc_dir, f"{prefix}.cif"))

    sga = SpacegroupAnalyzer(structure, symprec=0.01)
    sg = sga.get_space_group_symbol()

    metadata = {
        "index": idx,
        "source_file": filepath,
        "prefix": prefix,
        "formula": structure.composition.reduced_formula,
        "nsites": len(structure),
        "spacegroup": sg,
        "kgrid": list(kpts),
        "calc_dir": calc_dir,
        "input_file": input_file,
        "status": "pending",
    }
    job_metadata.append(metadata)
    print(f"  [{idx+1}/{len(structure_files)}] {filepath} -> {input_file}  "
          f"({structure.composition.reduced_formula}, {sg}, k={kpts})")

# Save metadata
meta_path = os.path.join(WORK_DIR, "job_metadata.json")
with open(meta_path, "w") as f:
    json.dump(job_metadata, f, indent=2)

print(f"\nGenerated {len(job_metadata)} QE input files in {WORK_DIR}/")
print(f"Metadata saved to {meta_path}")

# --- Generate batch run script ---
run_script = "#!/bin/bash\n"
run_script += f"# Batch QE {CALCULATION} calculations\n"
run_script += f"# Generated for {len(job_metadata)} structures\n"
run_script += f"# MPI processes per job: {NPROCS}\n\n"
run_script += "FAILED=0\n"
run_script += "PASSED=0\n\n"

for job in job_metadata:
    calc_dir = job["calc_dir"]
    prefix = job["prefix"]
    run_script += f'echo "[$(date +%H:%M:%S)] Running {prefix} ..."\n'
    run_script += f"cd {os.path.abspath(calc_dir)}\n"
    run_script += (f"mpirun -np {NPROCS} pw.x -in {CALCULATION}.in "
                   f"> {CALCULATION}.out 2>&1\n")
    run_script += "if [ $? -eq 0 ]; then\n"
    run_script += "  PASSED=$((PASSED + 1))\n"
    run_script += "else\n"
    run_script += f'  echo "  FAILED: {prefix}"\n'
    run_script += "  FAILED=$((FAILED + 1))\n"
    run_script += "fi\n"
    run_script += "cd -\n\n"

run_script += 'echo ""\n'
run_script += 'echo "==========================="\n'
run_script += 'echo "Batch complete: $PASSED passed, $FAILED failed"\n'
run_script += 'echo "==========================="\n'

script_path = os.path.join(WORK_DIR, "run_all.sh")
with open(script_path, "w") as f:
    f.write(run_script)
os.chmod(script_path, 0o755)
print(f"Batch run script: {script_path}")
print(f"\nTo run: bash {script_path}")
```

#### Step A2 -- Collect results and tabulate

```python
#!/usr/bin/env python3
"""
Collect results from batch QE calculations.
Parses output files, extracts energies, forces, stresses, and convergence status.
Produces a summary CSV and identifies failed calculations.
"""

import os
import re
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ================================================================== #
#  CONFIGURATION
# ================================================================== #
WORK_DIR = "batch_qe"
CALCULATION = "vc-relax"    # must match what was used in generation
OUTPUT_CSV = "batch_results.csv"
# ================================================================== #


def parse_qe_output(output_file):
    """
    Parse a QE pw.x output file and extract key quantities.
    Returns a dict with energy, forces, stress, convergence, timing.
    """
    result = {
        "total_energy_Ry": None,
        "total_energy_eV": None,
        "energy_per_atom_eV": None,
        "n_atoms": 0,
        "n_scf_steps": 0,
        "max_force_Ry_bohr": None,
        "pressure_kbar": None,
        "converged": False,
        "wall_time_s": None,
        "error_message": None,
    }

    if not os.path.exists(output_file):
        result["error_message"] = "Output file not found"
        return result

    try:
        with open(output_file) as f:
            lines = f.readlines()
    except Exception as e:
        result["error_message"] = str(e)
        return result

    for line in lines:
        # Total energy
        if "!" in line and "total energy" in line:
            try:
                result["total_energy_Ry"] = float(
                    line.split("=")[1].split("Ry")[0].strip())
            except (ValueError, IndexError):
                pass

        # Number of atoms
        if "number of atoms/cell" in line:
            try:
                result["n_atoms"] = int(line.split("=")[1].strip())
            except (ValueError, IndexError):
                pass

        # SCF convergence count
        if "convergence has been achieved in" in line:
            result["n_scf_steps"] += 1

        # Maximum force
        if "Total force" in line:
            try:
                result["max_force_Ry_bohr"] = float(line.split("=")[1].split()[0])
            except (ValueError, IndexError):
                pass

        # Pressure
        if "P=" in line and "total   stress" in line:
            try:
                result["pressure_kbar"] = float(line.split("P=")[1].strip())
            except (ValueError, IndexError):
                pass

        # Final convergence
        if "End final coordinates" in line or "Final energy" in line:
            result["converged"] = True

        # Also check for "JOB DONE" which indicates clean exit
        if "JOB DONE" in line:
            result["converged"] = True

        # Wall time
        if "WALL" in line.upper() and ("PWSCF" in line or "pw.x" in line):
            # Parse timing like "1h 2m" or "3m 4.5s" or "4.5s"
            time_match = re.findall(
                r"(\d+)h\s*(\d+)m|(\d+)m\s*([\d.]+)s|^\s*([\d.]+)s",
                line)
            # Simpler: just look for the last number-like thing
            numbers = re.findall(r"[\d.]+", line.split("WALL")[0])
            if numbers:
                try:
                    result["wall_time_s"] = float(numbers[-1])
                except ValueError:
                    pass

        # Error detection
        if "Error in routine" in line or "CRASH" in line:
            result["error_message"] = line.strip()
            result["converged"] = False

    # Compute derived quantities
    if result["total_energy_Ry"] is not None:
        result["total_energy_eV"] = result["total_energy_Ry"] * 13.605693123
        if result["n_atoms"] > 0:
            result["energy_per_atom_eV"] = (result["total_energy_eV"]
                                             / result["n_atoms"])

    return result


# --- Load metadata ---
meta_path = os.path.join(WORK_DIR, "job_metadata.json")
with open(meta_path) as f:
    job_metadata = json.load(f)

# --- Parse all outputs ---
rows = []
failed_jobs = []
for job in job_metadata:
    calc_dir = job["calc_dir"]
    output_file = os.path.join(calc_dir, f"{CALCULATION}.out")

    result = parse_qe_output(output_file)
    result.update({
        "index": job["index"],
        "prefix": job["prefix"],
        "formula": job["formula"],
        "nsites": job["nsites"],
        "spacegroup": job["spacegroup"],
        "source_file": job["source_file"],
    })
    rows.append(result)

    status = "OK" if result["converged"] else "FAILED"
    e_str = (f'{result["energy_per_atom_eV"]:.4f}'
             if result["energy_per_atom_eV"] is not None else "N/A")
    print(f"  [{job['index']+1}] {job['prefix']:>20s}  E/atom={e_str} eV  "
          f"{status}")

    if not result["converged"]:
        failed_jobs.append(job)

# --- Build DataFrame and export ---
df = pd.DataFrame(rows)
df.to_csv(OUTPUT_CSV, index=False)
print(f"\nResults saved to {OUTPUT_CSV}")
print(f"  Total: {len(rows)}, Converged: {len(rows) - len(failed_jobs)}, "
      f"Failed: {len(failed_jobs)}")

# --- Summary statistics ---
df_ok = df[df["converged"]].copy()
if len(df_ok) > 0:
    print(f"\nEnergy per atom statistics (converged):")
    print(f"  Mean:  {df_ok['energy_per_atom_eV'].mean():.4f} eV")
    print(f"  Std:   {df_ok['energy_per_atom_eV'].std():.4f} eV")
    print(f"  Range: {df_ok['energy_per_atom_eV'].min():.4f} to "
          f"{df_ok['energy_per_atom_eV'].max():.4f} eV")

# --- Report failed jobs ---
if failed_jobs:
    print(f"\nFailed calculations ({len(failed_jobs)}):")
    for job in failed_jobs:
        err = df[df["index"] == job["index"]]["error_message"].values[0]
        print(f"  {job['prefix']}: {err}")

# --- Visualization ---
if len(df_ok) > 2:
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    axes[0].bar(range(len(df_ok)), df_ok["energy_per_atom_eV"].values,
                color="steelblue", edgecolor="black")
    axes[0].set_xlabel("Structure index")
    axes[0].set_ylabel("Energy per atom (eV)")
    axes[0].set_title("Energy per atom")
    axes[0].set_xticks(range(len(df_ok)))
    axes[0].set_xticklabels(df_ok["formula"].values, rotation=45,
                            ha="right", fontsize=7)

    wt = df_ok["wall_time_s"].dropna()
    if len(wt) > 0:
        axes[1].bar(range(len(wt)), wt.values, color="C1", edgecolor="black")
        axes[1].set_xlabel("Structure index")
        axes[1].set_ylabel("Wall time (s)")
        axes[1].set_title("Calculation time")

    ns = df_ok["n_scf_steps"]
    axes[2].bar(range(len(ns)), ns.values, color="C2", edgecolor="black")
    axes[2].set_xlabel("Structure index")
    axes[2].set_ylabel("SCF cycles")
    axes[2].set_title("SCF convergence steps")

    plt.tight_layout()
    plt.savefig("batch_results_summary.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("Saved batch_results_summary.png")
```

#### Step A3 -- Error handling and restart logic

```python
#!/usr/bin/env python3
"""
Restart failed QE batch calculations with adjusted parameters.

Common failure modes and automatic fixes:
  - SCF not converged: reduce mixing_beta, increase electron_maxstep
  - Crash due to memory: reduce parallelization
  - Cell relaxation oscillating: reduce dt, switch cell_dynamics
  - Pseudopotential not found: report and skip
"""

import os
import json
import re
from pymatgen.core import Structure
from pymatgen.io.pwscf import PWInput

# ================================================================== #
#  CONFIGURATION
# ================================================================== #
WORK_DIR = "batch_qe"
CALCULATION = "vc-relax"
PSEUDO_DIR = "./pseudo"
NPROCS = 4
MAX_RESTARTS = 2        # maximum number of restart attempts per job
# ================================================================== #

# Load metadata
meta_path = os.path.join(WORK_DIR, "job_metadata.json")
with open(meta_path) as f:
    job_metadata = json.load(f)


def diagnose_failure(output_file):
    """Analyze a QE output file to determine the failure mode."""
    if not os.path.exists(output_file):
        return "no_output"

    with open(output_file) as f:
        content = f.read()

    # Check for specific error patterns
    if "convergence NOT achieved" in content:
        return "scf_not_converged"
    if "Error in routine" in content:
        if "pseudo" in content.lower():
            return "pseudo_not_found"
        if "out of memory" in content.lower() or "mpi" in content.lower():
            return "memory_error"
        return "runtime_error"
    if "Maximum number of iterations reached" in content:
        return "max_iterations"
    if "CRASH" in content:
        return "crash"
    if "JOB DONE" not in content:
        return "incomplete"

    return "unknown"


def generate_restart_input(job, failure_mode, restart_num):
    """
    Generate a modified input file to address the specific failure mode.
    Returns the path to the new input file, or None if not restartable.
    """
    calc_dir = job["calc_dir"]
    prefix = job["prefix"]

    # Load the original structure
    cif_files = [f for f in os.listdir(calc_dir) if f.endswith(".cif")]
    if not cif_files:
        print(f"  No CIF file found in {calc_dir}, cannot restart.")
        return None

    structure = Structure.from_file(os.path.join(calc_dir, cif_files[0]))

    # Build pseudopotential map
    pseudo_map = {}
    for el in structure.composition.elements:
        pseudo_map[el.symbol] = f"{el.symbol}.pbe-n-rrkjus_psl.1.0.0.UPF"

    # Base parameters
    abc = structure.lattice.abc
    kpts = tuple(max(1, int(round(40 / a))) for a in abc)

    control = {
        "calculation": CALCULATION,
        "prefix": prefix,
        "outdir": "./tmp",
        "pseudo_dir": os.path.abspath(PSEUDO_DIR),
        "tprnfor": True,
        "tstress": True,
        "etot_conv_thr": 1.0e-6,
        "forc_conv_thr": 1.0e-4,
    }

    system = {
        "ecutwfc": 60.0,
        "ecutrho": 480.0,
        "occupations": "smearing",
        "smearing": "mv",
        "degauss": 0.02,
    }

    electrons = {
        "conv_thr": 1.0e-8,
        "mixing_beta": 0.4,
    }

    ions = {"ion_dynamics": "bfgs"}
    cell = {"cell_dynamics": "bfgs", "press_conv_thr": 0.1}

    # --- Apply fixes based on failure mode ---
    if failure_mode == "scf_not_converged":
        # Reduce mixing_beta for better SCF convergence
        electrons["mixing_beta"] = 0.2 if restart_num == 1 else 0.1
        electrons["electron_maxstep"] = 200
        # Try different diagonalization
        if restart_num >= 2:
            electrons["diagonalization"] = "cg"
        print(f"  Fix: mixing_beta={electrons['mixing_beta']}, "
              f"maxstep={electrons['electron_maxstep']}")

    elif failure_mode == "max_iterations":
        # Increase iteration limits
        control["nstep"] = 200
        electrons["electron_maxstep"] = 200
        electrons["mixing_beta"] = 0.3
        print(f"  Fix: nstep=200, electron_maxstep=200, mixing_beta=0.3")

    elif failure_mode == "pseudo_not_found":
        print(f"  Cannot fix: pseudopotential files missing. "
              f"Check {PSEUDO_DIR}/ directory.")
        return None

    elif failure_mode == "memory_error":
        print(f"  Cannot fix automatically: out of memory. "
              f"Reduce NPROCS or ecutwfc.")
        return None

    elif failure_mode == "crash":
        # Try with safer parameters
        electrons["mixing_beta"] = 0.1
        electrons["electron_maxstep"] = 300
        system["degauss"] = 0.03
        # Try to restart from saved wavefunction
        control["restart_mode"] = "restart"
        print(f"  Fix: restart_mode=restart, mixing_beta=0.1, degauss=0.03")

    else:
        # Generic fix: use conservative parameters
        electrons["mixing_beta"] = 0.2
        electrons["electron_maxstep"] = 200
        print(f"  Fix: conservative parameters (mixing_beta=0.2)")

    # Write new input file
    pw_input = PWInput(
        structure,
        pseudo=pseudo_map,
        control=control,
        system=system,
        electrons=electrons,
        ions=ions,
        cell=cell if CALCULATION == "vc-relax" else None,
        kpoints_grid=kpts,
    )

    input_file = os.path.join(calc_dir, f"{CALCULATION}_restart{restart_num}.in")
    pw_input.write_file(input_file)
    return input_file


# --- Identify failed jobs and attempt restarts ---
restart_jobs = []
for job in job_metadata:
    calc_dir = job["calc_dir"]
    output_file = os.path.join(calc_dir, f"{CALCULATION}.out")

    failure = diagnose_failure(output_file)
    if failure == "unknown":
        continue  # already succeeded or unrecognizable

    # Check how many restarts have been attempted
    existing_restarts = len([f for f in os.listdir(calc_dir)
                             if "restart" in f and f.endswith(".in")])
    if existing_restarts >= MAX_RESTARTS:
        print(f"  {job['prefix']}: max restarts ({MAX_RESTARTS}) reached, "
              f"skipping.")
        continue

    restart_num = existing_restarts + 1
    print(f"\n{job['prefix']}: failure={failure}, restart #{restart_num}")

    new_input = generate_restart_input(job, failure, restart_num)
    if new_input is not None:
        restart_jobs.append({
            "job": job,
            "input_file": new_input,
            "restart_num": restart_num,
            "failure_mode": failure,
        })

# --- Generate restart script ---
if restart_jobs:
    script = "#!/bin/bash\n"
    script += f"# Restart script for {len(restart_jobs)} failed calculations\n\n"
    for rj in restart_jobs:
        calc_dir = rj["job"]["calc_dir"]
        input_file = os.path.basename(rj["input_file"])
        output_file = input_file.replace(".in", ".out")
        script += f'echo "Restarting {rj["job"]["prefix"]} '\
                  f'(was: {rj["failure_mode"]}) ..."\n'
        script += f"cd {os.path.abspath(calc_dir)}\n"
        script += f"mpirun -np {NPROCS} pw.x -in {input_file} > {output_file} 2>&1\n"
        script += "cd -\n\n"

    script_path = os.path.join(WORK_DIR, "restart_failed.sh")
    with open(script_path, "w") as f:
        f.write(script)
    os.chmod(script_path, 0o755)
    print(f"\nRestart script: {script_path}")
    print(f"  {len(restart_jobs)} jobs to restart. Run: bash {script_path}")
else:
    print("\nNo jobs to restart.")
```

### Workflow B: VASP Batch Calculations (using pymatgen input sets)

#### Step B1 -- Generate VASP input files with MPRelaxSet / MPStaticSet

```python
#!/usr/bin/env python3
"""
Generate VASP input files for batch calculations using pymatgen input sets.

Uses MPRelaxSet for relaxation and MPStaticSet for static calculations.
Generates INCAR, POSCAR, KPOINTS, and a POTCAR-generation command
(actual POTCAR requires VASP license).
"""

import os
import json
import glob
import numpy as np
from pymatgen.core import Structure
from pymatgen.io.vasp.sets import MPRelaxSet, MPStaticSet
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer

# ================================================================== #
#  CONFIGURATION
# ================================================================== #
INPUT_DIR = "structures"          # directory with CIF/POSCAR files
WORK_DIR = "batch_vasp"          # output directory for VASP calculations
CALC_TYPE = "relax"              # "relax" or "static"
USER_INCAR = {                   # override INCAR settings
    "ENCUT": 520,                # plane-wave cutoff (eV)
    "EDIFF": 1e-6,               # SCF convergence (eV)
    "EDIFFG": -0.02,             # force convergence (eV/Ang, negative = force)
    "ISMEAR": 0,                 # Gaussian smearing (0) or MP (1)
    "SIGMA": 0.05,               # smearing width (eV)
    "LORBIT": 11,                # write DOSCAR and lm-decomposed
    "LWAVE": False,              # do not write WAVECAR (saves disk)
    "LCHARG": False,             # do not write CHGCAR (saves disk)
}
# For relaxation:
USER_INCAR_RELAX = {
    "ISIF": 3,                   # relax ions + cell shape + cell volume
    "NSW": 200,                  # max ionic steps
    "IBRION": 2,                 # conjugate gradient
}
# For static:
USER_INCAR_STATIC = {
    "NSW": 0,
    "IBRION": -1,
    "LORBIT": 11,
    "LCHARG": True,              # need charge density for post-processing
}
# ================================================================== #

os.makedirs(WORK_DIR, exist_ok=True)


def generate_vasp_input(structure, calc_dir, calc_type="relax"):
    """Generate VASP input files using pymatgen MPRelaxSet or MPStaticSet."""
    os.makedirs(calc_dir, exist_ok=True)

    if calc_type == "relax":
        input_set = MPRelaxSet(
            structure,
            user_incar_settings={**USER_INCAR, **USER_INCAR_RELAX},
        )
    elif calc_type == "static":
        input_set = MPStaticSet(
            structure,
            user_incar_settings={**USER_INCAR, **USER_INCAR_STATIC},
        )
    else:
        raise ValueError(f"Unknown calc_type: {calc_type}")

    # Write INCAR, POSCAR, KPOINTS (but not POTCAR -- requires license)
    input_set.incar.write_file(os.path.join(calc_dir, "INCAR"))
    input_set.poscar.write_file(os.path.join(calc_dir, "POSCAR"))
    input_set.kpoints.write_file(os.path.join(calc_dir, "KPOINTS"))

    # Write a helper script to generate POTCAR
    elements = [el.symbol for el in structure.composition.elements]
    potcar_script = "#!/bin/bash\n"
    potcar_script += "# Generate POTCAR from VASP pseudopotential library\n"
    potcar_script += "# Set VASP_PSP_DIR to your VASP pseudopotential directory\n\n"
    potcar_script += 'if [ -z "$VASP_PSP_DIR" ]; then\n'
    potcar_script += '  echo "Error: VASP_PSP_DIR not set"\n'
    potcar_script += "  exit 1\n"
    potcar_script += "fi\n\n"
    potcar_script += "cat"
    for el in elements:
        potcar_script += f" $VASP_PSP_DIR/POT_GGA_PAW_PBE/{el}_pv/POTCAR"
    potcar_script += " > POTCAR\n"

    potcar_path = os.path.join(calc_dir, "generate_potcar.sh")
    with open(potcar_path, "w") as f:
        f.write(potcar_script)
    os.chmod(potcar_path, 0o755)

    return calc_dir


# --- Process all structures ---
structure_files = sorted(
    glob.glob(os.path.join(INPUT_DIR, "*.cif"))
    + glob.glob(os.path.join(INPUT_DIR, "*.vasp"))
    + glob.glob(os.path.join(INPUT_DIR, "POSCAR*"))
)

job_metadata = []
for idx, filepath in enumerate(structure_files):
    basename = os.path.splitext(os.path.basename(filepath))[0]
    prefix = basename.replace("-", "_").replace(" ", "_")

    try:
        structure = Structure.from_file(filepath)
    except Exception as e:
        print(f"  ERROR reading {filepath}: {e}")
        continue

    calc_dir = os.path.join(WORK_DIR, f"{idx:03d}_{prefix}")
    generate_vasp_input(structure, calc_dir, CALC_TYPE)

    sga = SpacegroupAnalyzer(structure, symprec=0.01)
    metadata = {
        "index": idx,
        "source_file": filepath,
        "prefix": prefix,
        "formula": structure.composition.reduced_formula,
        "nsites": len(structure),
        "spacegroup": sga.get_space_group_symbol(),
        "calc_dir": calc_dir,
        "calc_type": CALC_TYPE,
        "status": "pending",
    }
    job_metadata.append(metadata)
    print(f"  [{idx+1}/{len(structure_files)}] {filepath} -> {calc_dir}/")

# Save metadata
meta_path = os.path.join(WORK_DIR, "job_metadata.json")
with open(meta_path, "w") as f:
    json.dump(job_metadata, f, indent=2)

# --- Batch run script for VASP ---
run_script = "#!/bin/bash\n"
run_script += f"# Batch VASP {CALC_TYPE} calculations\n"
run_script += f"# {len(job_metadata)} structures\n\n"
run_script += "# First generate POTCARs\n"
for job in job_metadata:
    run_script += f"cd {os.path.abspath(job['calc_dir'])} && "
    run_script += "bash generate_potcar.sh && cd -\n"
run_script += "\n# Then run VASP\n"
for job in job_metadata:
    run_script += f'echo "Running {job["prefix"]} ..."\n'
    run_script += f"cd {os.path.abspath(job['calc_dir'])}\n"
    run_script += "mpirun -np 4 vasp_std > vasp.log 2>&1\n"
    run_script += "cd -\n\n"

script_path = os.path.join(WORK_DIR, "run_all.sh")
with open(script_path, "w") as f:
    f.write(run_script)
os.chmod(script_path, 0o755)

print(f"\nGenerated {len(job_metadata)} VASP inputs in {WORK_DIR}/")
print(f"Run script: {script_path}")
```

#### Step B2 -- Collect VASP results

```python
#!/usr/bin/env python3
"""
Collect results from batch VASP calculations.
Parses vasprun.xml and OUTCAR to extract energies, forces, and convergence.
"""

import os
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from pymatgen.io.vasp.outputs import Vasprun, Outcar

# ================================================================== #
WORK_DIR = "batch_vasp"
OUTPUT_CSV = "batch_vasp_results.csv"
# ================================================================== #

meta_path = os.path.join(WORK_DIR, "job_metadata.json")
with open(meta_path) as f:
    job_metadata = json.load(f)

rows = []
failed = []
for job in job_metadata:
    calc_dir = job["calc_dir"]
    prefix = job["prefix"]

    vasprun_file = os.path.join(calc_dir, "vasprun.xml")
    outcar_file = os.path.join(calc_dir, "OUTCAR")

    result = {
        "index": job["index"],
        "prefix": prefix,
        "formula": job["formula"],
        "nsites": job["nsites"],
        "spacegroup": job["spacegroup"],
    }

    try:
        vr = Vasprun(vasprun_file, parse_dos=False, parse_eigen=False)
        result["total_energy_eV"] = vr.final_energy
        result["energy_per_atom_eV"] = vr.final_energy / job["nsites"]
        result["converged_electronic"] = vr.converged_electronic
        result["converged_ionic"] = vr.converged_ionic
        result["converged"] = vr.converged_electronic and vr.converged_ionic
        result["band_gap_eV"] = vr.get_band_structure().get_band_gap()["energy"]

        # Extract final forces
        forces = np.array(vr.ionic_steps[-1]["forces"])
        result["max_force_eV_Ang"] = np.max(np.linalg.norm(forces, axis=1))

        status = "OK" if result["converged"] else "PARTIAL"
    except Exception as e:
        result["total_energy_eV"] = None
        result["energy_per_atom_eV"] = None
        result["converged"] = False
        result["error"] = str(e)
        status = "FAILED"
        failed.append(prefix)

    rows.append(result)
    e_str = (f'{result["energy_per_atom_eV"]:.4f}'
             if result.get("energy_per_atom_eV") is not None else "N/A")
    print(f"  [{job['index']+1}] {prefix:>20s}  E/atom={e_str} eV  {status}")

df = pd.DataFrame(rows)
df.to_csv(OUTPUT_CSV, index=False)
print(f"\nResults saved to {OUTPUT_CSV}")
print(f"Converged: {len(df[df['converged']==True])}/{len(df)}, "
      f"Failed: {len(failed)}")
```

## Key Parameters

### QE Parameters

| Parameter | Typical Value | Notes |
|---|---|---|
| `CALCULATION` | `"vc-relax"` | `"scf"` for single-point, `"relax"` for ionic, `"vc-relax"` for full |
| `ECUTWFC` | 60--80 Ry | Must be converged for pseudopotentials; 60 Ry is typical for SSSP |
| `KPOINT_DENSITY` | 30--50 | k-points per reciprocal Angstrom per direction; higher for metals |
| `CONV_THR` | 1e-8 Ry | SCF convergence; tighten to 1e-10 for stress calculations |
| `FORC_CONV_THR` | 1e-4 Ry/bohr | Force convergence for relaxation; ~0.005 eV/Ang |
| `mixing_beta` | 0.2--0.7 | Lower values for difficult convergence (metals, magnetic) |
| `NPROCS` | 2--8 | MPI processes; match to available cores |

### VASP Parameters (via pymatgen sets)

| Parameter | Typical Value | Notes |
|---|---|---|
| `ENCUT` | 520 eV | MP default; override only if needed |
| `EDIFF` | 1e-6 eV | SCF convergence |
| `EDIFFG` | -0.02 eV/Ang | Ionic convergence (negative = force criterion) |
| `ISIF` | 3 | 3 = relax everything; 2 = ions only; 4 = ions + cell shape |
| `ISMEAR` | 0 (Gaussian) or 1 (MP) | 0 for insulators, 1 for metals; -5 for DOS |
| `SIGMA` | 0.05 eV | Smearing width; check that entropy T*S < 1 meV/atom |
| `NSW` | 100--200 | Max ionic steps |

### Batch Execution Parameters

| Parameter | Value | Notes |
|---|---|---|
| `MAX_CONCURRENT` | 1 | Serial execution; increase if cluster allows |
| `MAX_RESTARTS` | 2 | Number of automatic restart attempts for failed jobs |
| `NPROCS` | 4 | MPI processes per calculation; adjust to hardware |

## Interpreting Results

- **Converged = True**: Both electronic SCF and ionic relaxation converged. The total energy and structure are reliable.
- **Converged = False**: The calculation did not finish cleanly. Check the output for error messages. The energy may be usable if SCF converged but ionic relaxation hit the step limit (the structure is simply not fully relaxed).
- **Energy per atom**: Comparable across structures with the same composition. For comparing across different compositions, compute formation energies (see `phase-stability/` skill).
- **Max force**: Should be below `FORC_CONV_THR` for a properly converged relaxation. Values above 0.05 eV/Ang indicate the structure is not at equilibrium.
- **Wall time**: Scales roughly as N^3 with number of electrons (N_atoms * Z_valence). Use to estimate total time for the batch.
- **SCF steps**: More than 100 SCF steps suggests convergence difficulties. Reduce `mixing_beta` or try different `diagonalization`.

## Common Issues

| Problem | Cause | Solution |
|---|---|---|
| SCF not converged | Mixing parameter too aggressive for the system | Reduce `mixing_beta` to 0.2 or 0.1; try `diagonalization = "cg"` |
| Ionic relaxation oscillates | Force convergence too tight or poor initial geometry | Relax `FORC_CONV_THR`; use MACE pre-relaxation; increase `NSW` |
| Pseudopotential not found | Filename mismatch or missing file | Check `PSEUDO_DIR` contents; adjust `get_pseudo_name()` function |
| Out of memory | Too many k-points or large cell | Reduce k-grid density; reduce `NPROCS` (less memory per process) |
| VASP POTCAR missing | No VASP license / POTCAR files | VASP requires a license; use `generate_potcar.sh` with valid `VASP_PSP_DIR` |
| Batch script hangs | A single calculation runs forever | Add timeout to run script: `timeout 3600 mpirun ...`; set `NSW` limit |
| Results CSV has NaN | Failed calculations produce None values | Filter with `df[df["converged"] == True]` before analysis |
| Different pseudopotentials across batch | Elements mapped inconsistently | Use a single `get_pseudo_name()` function for all inputs |
