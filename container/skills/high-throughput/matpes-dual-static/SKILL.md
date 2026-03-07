# MatPES Dual-Functional Static Calculation

## When to Use

- You need both PBE (GGA) and r2SCAN (meta-GGA) total energies for a set of structures in a single efficient pipeline.
- You are building training datasets for machine-learning interatomic potentials (MLIPs) and want energies at two levels of theory.
- You want to cross-validate GGA vs. meta-GGA stability predictions (formation energies, convex hulls, phase ordering).
- You are doing Materials Project-style high-throughput data generation following the MatPES (Materials Project Energy Surface) protocol.
- You want to run r2SCAN calculations efficiently by bootstrapping from converged PBE wavefunctions, reducing SCF iterations by 50--80%.
- You optionally need PBE+U energies for structures containing transition metals or lanthanides with Hubbard U corrections.

## Method Selection

```
What do you need?

Both PBE and r2SCAN energies for many structures efficiently?
  --> Use this skill (MatPES dual-functional static pipeline)

Only PBE static energies?
  --> Use batch-calculations/ with CALCULATION="scf"

Only r2SCAN energies (no PBE bootstrap)?
  --> Use batch-calculations/ with METAGGA=R2SCAN (slower, but simpler)

PBE+U energies for correlated systems?
  --> This skill handles it automatically when Hubbard-U elements are detected

Quick MLIP-based energy comparison before committing to DFT?
  --> Method C below (ASE + MACE single-point)

Formation energies and convex hull analysis from dual-functional data?
  --> Run this skill first, then use phase-stability/ on the collected energies
```

## Prerequisites

- For QE: Quantum ESPRESSO 7.5+ (`pw.x` in PATH), SSSP pseudopotentials downloaded.
- For VASP: VASP 6.x binary (`vasp_std`), PAW PBE pseudopotentials (POTCAR files), valid VASP license.
- Python packages: `pymatgen`, `ase`, `numpy`, `pandas`, `matplotlib`.
- Optional: `mace-torch` for Method C (MACE comparison).
- Optional: `mp-api` and `MP_API_KEY` for fetching structures from Materials Project.
- Structures in CIF, POSCAR, or pymatgen JSON format in an input directory.

## Detailed Steps

### Method A: QE (pw.x) Implementation

Complete pipeline generating Quantum ESPRESSO inputs for PBE followed by r2SCAN (SCAN) static calculations. QE uses `input_dft='scan'` for the SCAN meta-GGA functional (r2SCAN is available in QE 7.2+ via `input_dft='r2scan'`).

#### Step A1 -- Generate PBE and r2SCAN QE input files

```python
#!/usr/bin/env python3
"""
MatPES dual-functional static workflow for Quantum ESPRESSO.

Pipeline:
  1. PBE GGA static calculation (save wavefunctions)
  2. r2SCAN meta-GGA static calculation (restart from PBE wavefunctions)
  3. (Optional) PBE+U static if Hubbard-U elements are present

The PBE wavefunction provides a warm start for the meta-GGA SCF,
reducing r2SCAN convergence from ~60-100 iterations to ~10-30 iterations.
"""

import os
import json
import glob
import numpy as np
from pymatgen.core import Structure, Element
from pymatgen.io.pwscf import PWInput
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer

# ================================================================== #
#  CONFIGURATION
# ================================================================== #
INPUT_DIR = "structures"          # directory with CIF/POSCAR files
WORK_DIR = "matpes_qe"           # output directory
PSEUDO_DIR = "./pseudo"          # pseudopotential directory

# PBE static settings
ECUTWFC_PBE = 64.0               # Ry -- plane-wave cutoff
ECUTRHO_PBE = 512.0              # Ry -- charge density cutoff (8x ecutwfc)
CONV_THR_PBE = 1.0e-8            # Ry -- SCF convergence

# r2SCAN static settings (higher cutoff recommended for meta-GGA)
ECUTWFC_R2SCAN = 72.0            # Ry -- meta-GGA needs tighter basis
ECUTRHO_R2SCAN = 576.0           # Ry -- 8x ecutwfc
CONV_THR_R2SCAN = 1.0e-8         # Ry -- SCF convergence

# k-point settings
KPOINT_DENSITY = 45              # k-points per reciprocal Angstrom

# Smearing
SMEARING = "mv"                  # Marzari-Vanderbilt cold smearing
DEGAUSS = 0.01                   # Ry -- narrower for static

# Parallelization
NPROCS = 4                       # MPI processes per job

# Hubbard U values (eV) -- applied to PBE+U if element is present
# Following Materials Project conventions (based on Wang et al.)
HUBBARD_U = {
    "V":  3.25, "Cr": 3.7,  "Mn": 3.9,  "Fe": 5.3,
    "Co": 3.32, "Ni": 6.2,  "Cu": 7.0,  "Mo": 4.38,
    "W":  6.2,  "Ce": 5.0,  "Er": 5.0,  "Eu": 5.0,
    "Gd": 5.0,  "Ho": 5.0,  "La": 5.0,  "Lu": 5.0,
    "Nd": 5.0,  "Pm": 5.0,  "Pr": 5.0,  "Sm": 5.0,
    "Tb": 5.0,  "Tm": 5.0,  "Yb": 5.0,  "Dy": 5.0,
}
# ================================================================== #

os.makedirs(WORK_DIR, exist_ok=True)


def get_pseudo_name(element_symbol):
    """Return the SSSP pseudopotential filename for an element."""
    return f"{element_symbol}.pbe-n-rrkjus_psl.1.0.0.UPF"


def auto_kgrid(structure, density=KPOINT_DENSITY):
    """Compute k-point grid from lattice parameters and target density."""
    abc = structure.lattice.abc
    return tuple(max(1, int(round(density / a))) for a in abc)


def has_hubbard_elements(structure):
    """Check if the structure contains elements that need Hubbard U."""
    for el in structure.composition.elements:
        if el.symbol in HUBBARD_U:
            return True
    return False


def get_hubbard_species(structure):
    """Return dict of {element: U_value} for elements present in structure."""
    u_dict = {}
    for el in structure.composition.elements:
        if el.symbol in HUBBARD_U:
            u_dict[el.symbol] = HUBBARD_U[el.symbol]
    return u_dict


def write_qe_input_manual(filepath, structure, pseudo_map, kpts,
                          control, system, electrons,
                          hubbard_block=None):
    """
    Write a QE pw.x input file manually to support fields that
    pymatgen's PWInput may not handle (e.g., input_dft, startingwfc,
    HUBBARD block).
    """
    from pymatgen.io.pwscf import PWInput

    # Use PWInput for the base structure, then patch the file
    pw = PWInput(
        structure,
        pseudo=pseudo_map,
        control=control,
        system=system,
        electrons=electrons,
        kpoints_grid=kpts,
    )
    pw.write_file(filepath)

    # Read back and patch system namelist with additional keywords
    with open(filepath, "r") as f:
        content = f.read()

    # Inject extra system keywords before the closing /
    extra_system_lines = []
    if "input_dft" in system:
        # pymatgen may not write input_dft; ensure it is present
        if "input_dft" not in content.lower():
            extra_system_lines.append(
                f"  input_dft = '{system['input_dft']}'")
    if "startingwfc" in electrons:
        if "startingwfc" not in content.lower():
            # Add to &ELECTRONS block
            content = content.replace(
                "&ELECTRONS",
                f"&ELECTRONS\n  startingwfc = '{electrons['startingwfc']}'",
                1)

    if extra_system_lines:
        # Insert before the / that closes &SYSTEM
        # Find the &SYSTEM block and its closing /
        import re
        system_pattern = re.compile(
            r"(&SYSTEM.*?)(^\s*/)", re.MULTILINE | re.DOTALL)
        match = system_pattern.search(content)
        if match:
            insert_point = match.start(2)
            extra = "\n".join(extra_system_lines) + "\n"
            content = content[:insert_point] + extra + content[insert_point:]

    # Add HUBBARD block if needed (QE 7.x new-style)
    if hubbard_block:
        # Insert HUBBARD block after ATOMIC_SPECIES
        content += "\n" + hubbard_block + "\n"

    with open(filepath, "w") as f:
        f.write(content)


def generate_pbe_input(structure, calc_dir, prefix):
    """Generate PBE static input with LWAVE=True equivalent (disk_io='high')."""
    os.makedirs(calc_dir, exist_ok=True)

    pseudo_map = {el.symbol: get_pseudo_name(el.symbol)
                  for el in structure.composition.elements}
    kpts = auto_kgrid(structure)

    control = {
        "calculation": "scf",
        "prefix": prefix,
        "outdir": "./tmp",
        "pseudo_dir": os.path.abspath(PSEUDO_DIR),
        "tprnfor": True,
        "tstress": True,
        "disk_io": "high",      # Save wavefunctions to disk (like LWAVE=True)
        "wf_collect": True,     # Collect wavefunctions for restart
    }

    system = {
        "ecutwfc": ECUTWFC_PBE,
        "ecutrho": ECUTRHO_PBE,
        "occupations": "smearing",
        "smearing": SMEARING,
        "degauss": DEGAUSS,
    }

    electrons = {
        "conv_thr": CONV_THR_PBE,
        "mixing_beta": 0.4,
        "electron_maxstep": 200,
    }

    pw_input = PWInput(
        structure,
        pseudo=pseudo_map,
        control=control,
        system=system,
        electrons=electrons,
        kpoints_grid=kpts,
    )

    input_file = os.path.join(calc_dir, "pbe_static.in")
    pw_input.write_file(input_file)
    return input_file, kpts


def generate_r2scan_input(structure, calc_dir, prefix, kpts):
    """
    Generate r2SCAN static input that restarts from PBE wavefunctions.
    Uses startingwfc='file' to read the saved PBE wavefunction.
    """
    os.makedirs(calc_dir, exist_ok=True)

    pseudo_map = {el.symbol: get_pseudo_name(el.symbol)
                  for el in structure.composition.elements}

    control = {
        "calculation": "scf",
        "prefix": prefix,
        "outdir": "./tmp",          # Same outdir as PBE to read wavefunctions
        "pseudo_dir": os.path.abspath(PSEUDO_DIR),
        "tprnfor": True,
        "tstress": True,
        "disk_io": "low",           # No need to save wavefunctions again
    }

    system = {
        "ecutwfc": ECUTRHO_R2SCAN / 8,  # Use the higher meta-GGA cutoff
        "ecutrho": ECUTRHO_R2SCAN,
        "occupations": "smearing",
        "smearing": SMEARING,
        "degauss": DEGAUSS,
        "input_dft": "r2scan",      # r2SCAN meta-GGA functional
    }

    electrons = {
        "conv_thr": CONV_THR_R2SCAN,
        "mixing_beta": 0.4,
        "electron_maxstep": 200,
        "startingwfc": "file",       # Read PBE wavefunctions as starting point
    }

    write_qe_input_manual(
        filepath=os.path.join(calc_dir, "r2scan_static.in"),
        structure=structure,
        pseudo_map=pseudo_map,
        kpts=kpts,
        control=control,
        system=system,
        electrons=electrons,
    )

    return os.path.join(calc_dir, "r2scan_static.in")


def generate_pbe_u_input(structure, calc_dir, prefix, kpts):
    """
    Generate PBE+U static input for structures with Hubbard-U elements.
    Uses the dudarev (simplified) DFT+U approach.
    """
    os.makedirs(calc_dir, exist_ok=True)

    pseudo_map = {el.symbol: get_pseudo_name(el.symbol)
                  for el in structure.composition.elements}

    u_species = get_hubbard_species(structure)

    control = {
        "calculation": "scf",
        "prefix": prefix,
        "outdir": "./tmp_u",        # Separate outdir to avoid conflict
        "pseudo_dir": os.path.abspath(PSEUDO_DIR),
        "tprnfor": True,
        "tstress": True,
        "disk_io": "low",
    }

    system = {
        "ecutwfc": ECUTWFC_PBE,
        "ecutrho": ECUTRHO_PBE,
        "occupations": "smearing",
        "smearing": SMEARING,
        "degauss": DEGAUSS,
        "lda_plus_u": True,
        "lda_plus_u_kind": 0,       # Dudarev (simplified) DFT+U
    }

    electrons = {
        "conv_thr": CONV_THR_PBE,
        "mixing_beta": 0.3,         # More conservative for DFT+U
        "electron_maxstep": 300,
    }

    # Build Hubbard U specification for QE
    # In QE, Hubbard_U is set per atomic species
    elements_list = [el.symbol for el in structure.composition.elements]
    hubbard_lines = []
    for el_sym in elements_list:
        if el_sym in u_species:
            hubbard_lines.append(
                f"  Hubbard_U({elements_list.index(el_sym) + 1}) = "
                f"{u_species[el_sym]}")

    # Write base input with PWInput, then patch
    pw_input = PWInput(
        structure,
        pseudo=pseudo_map,
        control=control,
        system=system,
        electrons=electrons,
        kpoints_grid=kpts,
    )

    input_file = os.path.join(calc_dir, "pbe_u_static.in")
    pw_input.write_file(input_file)

    # Patch in Hubbard U values into &SYSTEM block
    with open(input_file, "r") as f:
        content = f.read()

    import re
    system_close = re.compile(r"(^\s*/)", re.MULTILINE)
    # Find the first / after &SYSTEM
    system_start = content.find("&SYSTEM")
    if system_start >= 0:
        match = system_close.search(content, system_start + 7)
        if match and hubbard_lines:
            insert_point = match.start()
            extra = "\n".join(hubbard_lines) + "\n"
            content = content[:insert_point] + extra + content[insert_point:]

    with open(input_file, "w") as f:
        f.write(content)

    return input_file


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
    prefix = basename.replace("-", "_").replace(" ", "_")

    try:
        structure = Structure.from_file(filepath)
    except Exception as e:
        print(f"  ERROR reading {filepath}: {e}")
        continue

    calc_dir = os.path.join(WORK_DIR, f"{idx:03d}_{prefix}")
    sga = SpacegroupAnalyzer(structure, symprec=0.01)
    sg = sga.get_space_group_symbol()

    # Step 1: PBE static (with wavefunction saving)
    pbe_input, kpts = generate_pbe_input(structure, calc_dir, prefix)

    # Step 2: r2SCAN static (restart from PBE wavefunction)
    r2scan_input = generate_r2scan_input(structure, calc_dir, prefix, kpts)

    # Step 3 (optional): PBE+U if Hubbard elements present
    needs_u = has_hubbard_elements(structure)
    pbe_u_input = None
    if needs_u:
        pbe_u_input = generate_pbe_u_input(structure, calc_dir, prefix, kpts)

    # Save structure as CIF for reference
    structure.to(os.path.join(calc_dir, f"{prefix}.cif"))

    metadata = {
        "index": idx,
        "source_file": filepath,
        "prefix": prefix,
        "formula": structure.composition.reduced_formula,
        "nsites": len(structure),
        "spacegroup": sg,
        "kgrid": list(kpts),
        "calc_dir": calc_dir,
        "pbe_input": pbe_input,
        "r2scan_input": r2scan_input,
        "needs_hubbard_u": needs_u,
        "pbe_u_input": pbe_u_input,
        "status_pbe": "pending",
        "status_r2scan": "pending",
        "status_pbe_u": "pending" if needs_u else "n/a",
    }
    job_metadata.append(metadata)

    u_tag = " [+U]" if needs_u else ""
    print(f"  [{idx+1}/{len(structure_files)}] {filepath} -> {calc_dir}/  "
          f"({structure.composition.reduced_formula}, {sg}, k={kpts}){u_tag}")

# Save metadata
meta_path = os.path.join(WORK_DIR, "job_metadata.json")
with open(meta_path, "w") as f:
    json.dump(job_metadata, f, indent=2)

print(f"\nGenerated inputs for {len(job_metadata)} structures in {WORK_DIR}/")
print(f"Metadata saved to {meta_path}")

# --- Generate batch run script ---
# The key: run PBE first, THEN r2SCAN in the same directory so it can
# read the saved wavefunction from tmp/prefix.wfc
run_script = "#!/bin/bash\n"
run_script += "# MatPES dual-functional static pipeline (QE)\n"
run_script += "# PBE -> r2SCAN (with wavefunction restart)\n"
run_script += f"# {len(job_metadata)} structures\n\n"
run_script += "FAILED=0\nPASSED=0\n\n"

for job in job_metadata:
    calc_dir = os.path.abspath(job["calc_dir"])
    prefix = job["prefix"]

    # PBE static
    run_script += f'echo "[$(date +%H:%M:%S)] {prefix}: PBE static ..."\n'
    run_script += f"cd {calc_dir}\n"
    run_script += (f"mpirun -np {NPROCS} pw.x -in pbe_static.in "
                   f"> pbe_static.out 2>&1\n")
    run_script += "PBE_EXIT=$?\n\n"

    # r2SCAN static (only if PBE succeeded)
    run_script += "if [ $PBE_EXIT -eq 0 ]; then\n"
    run_script += f'  echo "[$(date +%H:%M:%S)] {prefix}: r2SCAN static '
    run_script += f'(from PBE wavefunction) ..."\n'
    run_script += (f"  mpirun -np {NPROCS} pw.x -in r2scan_static.in "
                   f"> r2scan_static.out 2>&1\n")
    run_script += "  if [ $? -eq 0 ]; then\n"
    run_script += "    PASSED=$((PASSED + 1))\n"
    run_script += "  else\n"
    run_script += f'    echo "  FAILED r2SCAN: {prefix}"\n'
    run_script += "    FAILED=$((FAILED + 1))\n"
    run_script += "  fi\n"
    run_script += "else\n"
    run_script += f'  echo "  FAILED PBE: {prefix} (skipping r2SCAN)"\n'
    run_script += "  FAILED=$((FAILED + 1))\n"
    run_script += "fi\n"

    # PBE+U if needed
    if job["needs_hubbard_u"]:
        run_script += f'\necho "[$(date +%H:%M:%S)] {prefix}: PBE+U static ..."\n'
        run_script += (f"mpirun -np {NPROCS} pw.x -in pbe_u_static.in "
                       f"> pbe_u_static.out 2>&1\n")
        run_script += "if [ $? -ne 0 ]; then\n"
        run_script += f'  echo "  FAILED PBE+U: {prefix}"\n'
        run_script += "fi\n"

    run_script += "cd -\n\n"

run_script += 'echo ""\n'
run_script += 'echo "==================================="\n'
run_script += 'echo "MatPES pipeline: $PASSED passed, $FAILED failed"\n'
run_script += 'echo "==================================="\n'

script_path = os.path.join(WORK_DIR, "run_matpes.sh")
with open(script_path, "w") as f:
    f.write(run_script)
os.chmod(script_path, 0o755)
print(f"Batch run script: {script_path}")
print(f"\nTo run: bash {script_path}")
```

#### Step A2 -- Collect PBE and r2SCAN results

```python
#!/usr/bin/env python3
"""
Collect results from MatPES dual-functional QE calculations.
Parses PBE and r2SCAN outputs, compares energies, tabulates results.
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
WORK_DIR = "matpes_qe"
OUTPUT_CSV = "matpes_qe_results.csv"
# ================================================================== #


def parse_qe_output(output_file):
    """Parse a QE pw.x output and extract energy, forces, SCF steps."""
    result = {
        "total_energy_Ry": None,
        "total_energy_eV": None,
        "energy_per_atom_eV": None,
        "n_atoms": 0,
        "n_scf_steps": 0,
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
        if "!" in line and "total energy" in line:
            try:
                result["total_energy_Ry"] = float(
                    line.split("=")[1].split("Ry")[0].strip())
            except (ValueError, IndexError):
                pass
        if "number of atoms/cell" in line:
            try:
                result["n_atoms"] = int(line.split("=")[1].strip())
            except (ValueError, IndexError):
                pass
        if "convergence has been achieved in" in line:
            try:
                n_iter = int(re.search(r"in\s+(\d+)", line).group(1))
                result["n_scf_steps"] = n_iter
            except (AttributeError, ValueError):
                result["n_scf_steps"] += 1
        if "JOB DONE" in line:
            result["converged"] = True
        if "Error in routine" in line or "CRASH" in line:
            result["error_message"] = line.strip()
            result["converged"] = False

    if result["total_energy_Ry"] is not None:
        result["total_energy_eV"] = result["total_energy_Ry"] * 13.605693123
        if result["n_atoms"] > 0:
            result["energy_per_atom_eV"] = (
                result["total_energy_eV"] / result["n_atoms"])

    return result


# --- Load metadata ---
meta_path = os.path.join(WORK_DIR, "job_metadata.json")
with open(meta_path) as f:
    job_metadata = json.load(f)

# --- Parse all outputs ---
rows = []
for job in job_metadata:
    calc_dir = job["calc_dir"]
    prefix = job["prefix"]

    # PBE results
    pbe = parse_qe_output(os.path.join(calc_dir, "pbe_static.out"))
    # r2SCAN results
    r2s = parse_qe_output(os.path.join(calc_dir, "r2scan_static.out"))
    # PBE+U results (if applicable)
    pbe_u = None
    if job["needs_hubbard_u"]:
        pbe_u = parse_qe_output(os.path.join(calc_dir, "pbe_u_static.out"))

    # Compute energy difference
    delta_e = None
    if (pbe["energy_per_atom_eV"] is not None
            and r2s["energy_per_atom_eV"] is not None):
        delta_e = r2s["energy_per_atom_eV"] - pbe["energy_per_atom_eV"]

    row = {
        "index": job["index"],
        "prefix": prefix,
        "formula": job["formula"],
        "nsites": job["nsites"],
        "spacegroup": job["spacegroup"],
        # PBE
        "pbe_energy_per_atom_eV": pbe["energy_per_atom_eV"],
        "pbe_total_energy_eV": pbe["total_energy_eV"],
        "pbe_scf_steps": pbe["n_scf_steps"],
        "pbe_converged": pbe["converged"],
        # r2SCAN
        "r2scan_energy_per_atom_eV": r2s["energy_per_atom_eV"],
        "r2scan_total_energy_eV": r2s["total_energy_eV"],
        "r2scan_scf_steps": r2s["n_scf_steps"],
        "r2scan_converged": r2s["converged"],
        # Delta
        "delta_E_r2scan_minus_pbe_eV": delta_e,
    }

    if pbe_u is not None:
        row["pbe_u_energy_per_atom_eV"] = pbe_u["energy_per_atom_eV"]
        row["pbe_u_total_energy_eV"] = pbe_u["total_energy_eV"]
        row["pbe_u_scf_steps"] = pbe_u["n_scf_steps"]
        row["pbe_u_converged"] = pbe_u["converged"]

    rows.append(row)

    # Print summary line
    pbe_e = (f'{pbe["energy_per_atom_eV"]:.4f}'
             if pbe["energy_per_atom_eV"] else "N/A")
    r2s_e = (f'{r2s["energy_per_atom_eV"]:.4f}'
             if r2s["energy_per_atom_eV"] else "N/A")
    delta_str = f"{delta_e:+.4f}" if delta_e is not None else "N/A"
    scf_str = (f"PBE:{pbe['n_scf_steps']}, r2SCAN:{r2s['n_scf_steps']}")
    print(f"  [{job['index']+1}] {prefix:>20s}  "
          f"PBE={pbe_e}  r2SCAN={r2s_e}  delta={delta_str}  SCF({scf_str})")

# --- Export ---
df = pd.DataFrame(rows)
df.to_csv(OUTPUT_CSV, index=False)
print(f"\nResults saved to {OUTPUT_CSV}")

# --- Summary statistics ---
df_ok = df[df["pbe_converged"] & df["r2scan_converged"]].copy()
n_total = len(df)
n_ok = len(df_ok)
n_u = df[df.columns[df.columns.str.contains("pbe_u")]].dropna(
    how="all").shape[0] if "pbe_u_converged" in df.columns else 0

print(f"\nSummary: {n_ok}/{n_total} both PBE+r2SCAN converged")
if n_u > 0:
    print(f"  PBE+U calculations: {n_u}")

if n_ok > 0:
    print(f"\nEnergy difference (r2SCAN - PBE) per atom:")
    deltas = df_ok["delta_E_r2scan_minus_pbe_eV"].dropna()
    print(f"  Mean:  {deltas.mean():+.4f} eV/atom")
    print(f"  Std:   {deltas.std():.4f} eV/atom")
    print(f"  Range: {deltas.min():+.4f} to {deltas.max():+.4f} eV/atom")
    print(f"\nSCF iteration savings (r2SCAN with PBE restart):")
    print(f"  PBE mean SCF steps:    {df_ok['pbe_scf_steps'].mean():.1f}")
    print(f"  r2SCAN mean SCF steps: {df_ok['r2scan_scf_steps'].mean():.1f}")

# --- Visualization ---
if n_ok >= 2:
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    # PBE vs r2SCAN parity
    x = df_ok["pbe_energy_per_atom_eV"].values
    y = df_ok["r2scan_energy_per_atom_eV"].values
    axes[0].scatter(x, y, s=50, edgecolors="black", zorder=5)
    lim = [min(x.min(), y.min()) - 0.2, max(x.max(), y.max()) + 0.2]
    axes[0].plot(lim, lim, "k--", alpha=0.4, label="y=x")
    axes[0].set_xlabel("PBE energy/atom (eV)")
    axes[0].set_ylabel("r2SCAN energy/atom (eV)")
    axes[0].set_title("PBE vs r2SCAN energies")
    axes[0].legend()

    # Energy difference distribution
    axes[1].hist(deltas.values, bins=max(5, n_ok // 3),
                 edgecolor="black", color="C1", alpha=0.8)
    axes[1].axvline(deltas.mean(), color="red", linestyle="--",
                    label=f"mean={deltas.mean():+.3f}")
    axes[1].set_xlabel("r2SCAN - PBE (eV/atom)")
    axes[1].set_ylabel("Count")
    axes[1].set_title("Energy difference distribution")
    axes[1].legend()

    # SCF steps comparison
    pbe_steps = df_ok["pbe_scf_steps"].values
    r2s_steps = df_ok["r2scan_scf_steps"].values
    x_pos = np.arange(n_ok)
    w = 0.35
    axes[2].bar(x_pos - w/2, pbe_steps, w, label="PBE", color="C0",
                edgecolor="black")
    axes[2].bar(x_pos + w/2, r2s_steps, w, label="r2SCAN", color="C2",
                edgecolor="black")
    axes[2].set_xlabel("Structure index")
    axes[2].set_ylabel("SCF iterations")
    axes[2].set_title("SCF convergence: PBE vs r2SCAN (warm start)")
    axes[2].legend()
    if n_ok <= 20:
        axes[2].set_xticks(x_pos)
        axes[2].set_xticklabels(df_ok["formula"].values, rotation=45,
                                ha="right", fontsize=7)

    plt.tight_layout()
    plot_path = "matpes_qe_summary.png"
    plt.savefig(plot_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\nSaved {plot_path}")
```

### Method B: VASP Implementation

Complete pipeline generating VASP input files for the PBE-to-r2SCAN dual static workflow, following atomate2's `MatPesStaticFlowMaker` approach.

#### Step B1 -- Generate VASP inputs for PBE and r2SCAN pipeline

```python
#!/usr/bin/env python3
"""
MatPES dual-functional static workflow for VASP.

Follows the atomate2 MatPesStaticFlowMaker pattern:
  1. PBE GGA static with LWAVE=True (saves WAVECAR)
  2. r2SCAN meta-GGA static reading PBE WAVECAR (warm SCF start)
  3. (Optional) PBE+U static if Hubbard-U elements are present

VASP-specific advantages:
  - WAVECAR from PBE provides excellent starting wavefunctions for r2SCAN
  - ICHARG=1 + pre-converged CHGCAR can also help (but WAVECAR is better)
  - r2SCAN SCF converges in 10-30 iterations instead of 60-100
"""

import os
import json
import glob
import numpy as np
from pymatgen.core import Structure, Element
from pymatgen.io.vasp import Poscar, Kpoints, Incar
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
from pymatgen.symmetry.bandstructure import HighSymmKpath

# ================================================================== #
#  CONFIGURATION
# ================================================================== #
INPUT_DIR = "structures"          # directory with CIF/POSCAR files
WORK_DIR = "matpes_vasp"          # output directory

# PBE static INCAR settings
PBE_INCAR = {
    # General
    "SYSTEM": "MatPES PBE static",
    "PREC": "Accurate",
    "ALGO": "Normal",           # blocked Davidson
    "ENCUT": 520,               # eV -- MP standard
    "NELM": 200,                # max SCF steps
    "EDIFF": 1e-6,              # eV -- SCF convergence
    "EDIFFG": -1,               # not relaxing (NSW=0)

    # Electronic
    "ISMEAR": 0,                # Gaussian smearing
    "SIGMA": 0.05,              # eV
    "LORBIT": 11,               # l-projected DOS
    "LREAL": "Auto",            # real-space projection

    # Static calculation
    "NSW": 0,                   # no ionic steps
    "IBRION": -1,               # no relaxation
    "ISIF": 2,                  # compute stress tensor

    # Wavefunction and charge output -- THE KEY SETTINGS
    "LWAVE": True,              # SAVE WAVECAR for r2SCAN restart
    "LCHARG": True,             # Save CHGCAR (useful for analysis)

    # Spin polarization (enable for magnetic systems)
    "ISPIN": 2,                 # spin-polarized
    "LMAXMIX": 6,              # for d/f elements

    # Performance
    "NCORE": 4,                 # cores per orbital band
    "KPAR": 1,                  # k-point parallelism
}

# r2SCAN static INCAR settings
R2SCAN_INCAR = {
    # General
    "SYSTEM": "MatPES r2SCAN static",
    "PREC": "Accurate",
    "ALGO": "All",              # recommended for meta-GGA
    "ENCUT": 680,               # eV -- higher cutoff for meta-GGA
    "NELM": 200,                # max SCF steps
    "EDIFF": 1e-6,              # eV -- SCF convergence

    # Meta-GGA specification
    "METAGGA": "R2SCAN",        # r2SCAN meta-GGA functional
    "LASPH": True,              # aspherical contributions (REQUIRED for meta-GGA)

    # Electronic
    "ISMEAR": 0,                # Gaussian smearing
    "SIGMA": 0.05,              # eV
    "LORBIT": 11,
    "LREAL": "Auto",

    # Static calculation
    "NSW": 0,
    "IBRION": -1,
    "ISIF": 2,

    # Wavefunction restart from PBE -- THE KEY SETTINGS
    "ISTART": 1,                # Read WAVECAR from PBE calculation
    "ICHARG": 1,                # Read CHGCAR and calculate from WAVECAR

    # Output
    "LWAVE": False,             # No need to save r2SCAN WAVECAR
    "LCHARG": True,             # Save r2SCAN CHGCAR

    # Spin
    "ISPIN": 2,
    "LMAXMIX": 6,

    # Performance
    "NCORE": 4,
    "KPAR": 1,
}

# PBE+U settings (Dudarev approach, applied only when U elements present)
PBE_U_EXTRA = {
    "SYSTEM": "MatPES PBE+U static",
    "LDAU": True,
    "LDAUTYPE": 2,              # Dudarev (simplified rotationally invariant)
    "LDAUPRINT": 1,             # print occupancy matrices
    "LWAVE": False,             # no WAVECAR needed for PBE+U
}

# Hubbard U values (eV) -- Materials Project conventions
HUBBARD_U = {
    "V":  3.25, "Cr": 3.7,  "Mn": 3.9,  "Fe": 5.3,
    "Co": 3.32, "Ni": 6.2,  "Cu": 7.0,  "Mo": 4.38,
    "W":  6.2,  "Ce": 5.0,  "Er": 5.0,  "Eu": 5.0,
    "Gd": 5.0,  "Ho": 5.0,  "La": 5.0,  "Lu": 5.0,
    "Nd": 5.0,  "Pm": 5.0,  "Pr": 5.0,  "Sm": 5.0,
    "Tb": 5.0,  "Tm": 5.0,  "Yb": 5.0,  "Dy": 5.0,
}

# k-point density (per reciprocal Angstrom)
KPOINT_DENSITY = 45
# ================================================================== #

os.makedirs(WORK_DIR, exist_ok=True)


def auto_kpoints(structure, density=KPOINT_DENSITY):
    """Generate Gamma-centered k-point grid from lattice parameters."""
    abc = structure.lattice.abc
    grid = tuple(max(1, int(round(density / a))) for a in abc)
    return Kpoints.gamma_automatic(grid)


def has_hubbard_elements(structure):
    """Check if structure contains elements needing Hubbard U correction."""
    return any(el.symbol in HUBBARD_U
               for el in structure.composition.elements)


def get_ldau_params(structure):
    """
    Build LDAUL, LDAUU, LDAUJ lists for VASP DFT+U.
    Order must match POSCAR element order.
    """
    elements = [el.symbol for el in sorted(
        structure.composition.elements, key=lambda x: x.X)]
    # pymatgen Poscar sorts elements by electronegativity by default
    poscar = Poscar(structure)
    elements = [site.specie.symbol for site in poscar.structure]
    # Get unique elements in POSCAR order
    seen = set()
    unique_elements = []
    for el in elements:
        if el not in seen:
            unique_elements.append(el)
            seen.add(el)

    ldaul = []
    ldauu = []
    ldauj = []
    for el in unique_elements:
        if el in HUBBARD_U:
            ldaul.append(2)                 # d-electrons
            ldauu.append(HUBBARD_U[el])
            ldauj.append(0.0)
        else:
            ldaul.append(-1)                # no U
            ldauu.append(0.0)
            ldauj.append(0.0)

    return ldaul, ldauu, ldauj


def write_vasp_inputs(structure, calc_dir, incar_settings, prefix="calc"):
    """Write INCAR, POSCAR, KPOINTS to calc_dir."""
    os.makedirs(calc_dir, exist_ok=True)

    # POSCAR
    poscar = Poscar(structure, comment=prefix)
    poscar.write_file(os.path.join(calc_dir, "POSCAR"))

    # KPOINTS
    kpts = auto_kpoints(structure)
    kpts.write_file(os.path.join(calc_dir, "KPOINTS"))

    # INCAR
    incar = Incar(incar_settings)
    incar.write_file(os.path.join(calc_dir, "INCAR"))

    # POTCAR generation helper script
    elements_in_poscar = []
    seen = set()
    for site in structure:
        sym = site.specie.symbol
        if sym not in seen:
            elements_in_poscar.append(sym)
            seen.add(sym)

    potcar_script = "#!/bin/bash\n"
    potcar_script += "# Generate POTCAR from VASP pseudopotential library\n"
    potcar_script += 'if [ -z "$VASP_PSP_DIR" ]; then\n'
    potcar_script += '  echo "Error: Set VASP_PSP_DIR to your PP directory"\n'
    potcar_script += "  exit 1\n"
    potcar_script += "fi\n\n"
    potcar_script += "cat"
    for el in elements_in_poscar:
        # Use _pv / _sv variants for transition metals (MP convention)
        if el in ("Ti", "V", "Cr", "Mn", "Zr", "Nb", "Mo", "Hf", "Ta", "W"):
            pp_name = f"{el}_pv"
        elif el in ("Fe", "Co", "Ni", "Cu", "Zn", "Ga", "Ge"):
            pp_name = el
        elif el in ("La", "Ce", "Pr", "Nd", "Pm", "Sm", "Eu", "Gd",
                     "Tb", "Dy", "Ho", "Er", "Tm", "Yb", "Lu"):
            pp_name = f"{el}_3"
        elif el in ("K", "Ca", "Rb", "Sr", "Cs", "Ba"):
            pp_name = f"{el}_sv"
        elif el in ("Na", "Li"):
            pp_name = f"{el}_sv"
        else:
            pp_name = el
        potcar_script += f" $VASP_PSP_DIR/POT_GGA_PAW_PBE/{pp_name}/POTCAR"
    potcar_script += " > POTCAR\n"

    with open(os.path.join(calc_dir, "generate_potcar.sh"), "w") as f:
        f.write(potcar_script)
    os.chmod(os.path.join(calc_dir, "generate_potcar.sh"), 0o755)

    return calc_dir


# --- Process all structures ---
structure_files = sorted(
    glob.glob(os.path.join(INPUT_DIR, "*.cif"))
    + glob.glob(os.path.join(INPUT_DIR, "*.vasp"))
    + glob.glob(os.path.join(INPUT_DIR, "POSCAR*"))
    + glob.glob(os.path.join(INPUT_DIR, "*.json"))
)

if not structure_files:
    print(f"No structure files found in {INPUT_DIR}/")
    exit(1)

job_metadata = []
for idx, filepath in enumerate(structure_files):
    basename = os.path.splitext(os.path.basename(filepath))[0]
    prefix = basename.replace("-", "_").replace(" ", "_")

    try:
        structure = Structure.from_file(filepath)
    except Exception as e:
        print(f"  ERROR reading {filepath}: {e}")
        continue

    calc_base = os.path.join(WORK_DIR, f"{idx:03d}_{prefix}")
    sga = SpacegroupAnalyzer(structure, symprec=0.01)
    sg = sga.get_space_group_symbol()

    # --- Step 1: PBE static with LWAVE=True ---
    pbe_dir = os.path.join(calc_base, "pbe_static")
    write_vasp_inputs(structure, pbe_dir, PBE_INCAR, prefix=f"PBE_{prefix}")

    # --- Step 2: r2SCAN static (will read WAVECAR from PBE) ---
    r2scan_dir = os.path.join(calc_base, "r2scan_static")
    write_vasp_inputs(structure, r2scan_dir, R2SCAN_INCAR,
                      prefix=f"R2SCAN_{prefix}")

    # --- Step 3 (optional): PBE+U ---
    needs_u = has_hubbard_elements(structure)
    pbe_u_dir = None
    if needs_u:
        ldaul, ldauu, ldauj = get_ldau_params(structure)
        pbe_u_incar = {**PBE_INCAR, **PBE_U_EXTRA}
        pbe_u_incar["LDAUL"] = ldaul
        pbe_u_incar["LDAUU"] = ldauu
        pbe_u_incar["LDAUJ"] = ldauj
        pbe_u_dir = os.path.join(calc_base, "pbe_u_static")
        write_vasp_inputs(structure, pbe_u_dir, pbe_u_incar,
                          prefix=f"PBE_U_{prefix}")

    metadata = {
        "index": idx,
        "source_file": filepath,
        "prefix": prefix,
        "formula": structure.composition.reduced_formula,
        "nsites": len(structure),
        "spacegroup": sg,
        "pbe_dir": pbe_dir,
        "r2scan_dir": r2scan_dir,
        "needs_hubbard_u": needs_u,
        "pbe_u_dir": pbe_u_dir,
        "status_pbe": "pending",
        "status_r2scan": "pending",
        "status_pbe_u": "pending" if needs_u else "n/a",
    }
    job_metadata.append(metadata)

    u_tag = " [+U]" if needs_u else ""
    print(f"  [{idx+1}/{len(structure_files)}] {filepath} -> {calc_base}/  "
          f"({structure.composition.reduced_formula}, {sg}){u_tag}")

# Save metadata
meta_path = os.path.join(WORK_DIR, "job_metadata.json")
with open(meta_path, "w") as f:
    json.dump(job_metadata, f, indent=2)

# --- Generate batch run script ---
run_script = "#!/bin/bash\n"
run_script += "# MatPES dual-functional static pipeline (VASP)\n"
run_script += "# PBE (LWAVE=True) -> copy WAVECAR -> r2SCAN\n"
run_script += f"# {len(job_metadata)} structures\n\n"
run_script += "NPROCS=${NPROCS:-4}\n"
run_script += "FAILED=0\nPASSED=0\n\n"

for job in job_metadata:
    pbe_dir = os.path.abspath(job["pbe_dir"])
    r2scan_dir = os.path.abspath(job["r2scan_dir"])
    prefix = job["prefix"]

    # Generate POTCAR for both directories
    run_script += f"# --- {prefix} ---\n"
    run_script += f"cd {pbe_dir} && bash generate_potcar.sh && cd -\n"
    run_script += f"cd {r2scan_dir} && bash generate_potcar.sh && cd -\n\n"

    # PBE static
    run_script += f'echo "[$(date +%H:%M:%S)] {prefix}: PBE static ..."\n'
    run_script += f"cd {pbe_dir}\n"
    run_script += "mpirun -np $NPROCS vasp_std > vasp.log 2>&1\n"
    run_script += "PBE_EXIT=$?\n\n"

    # Copy WAVECAR and CHGCAR to r2SCAN directory, then run r2SCAN
    run_script += "if [ $PBE_EXIT -eq 0 ]; then\n"
    run_script += f'  echo "[$(date +%H:%M:%S)] {prefix}: Copying WAVECAR '
    run_script += f'to r2SCAN directory ..."\n'
    run_script += f"  cp {pbe_dir}/WAVECAR {r2scan_dir}/WAVECAR\n"
    run_script += f"  cp {pbe_dir}/CHGCAR  {r2scan_dir}/CHGCAR\n\n"
    run_script += f'  echo "[$(date +%H:%M:%S)] {prefix}: r2SCAN static '
    run_script += f'(from PBE WAVECAR) ..."\n'
    run_script += f"  cd {r2scan_dir}\n"
    run_script += "  mpirun -np $NPROCS vasp_std > vasp.log 2>&1\n"
    run_script += "  if [ $? -eq 0 ]; then\n"
    run_script += "    PASSED=$((PASSED + 1))\n"
    run_script += f'    echo "  OK: {prefix}"\n'
    run_script += "  else\n"
    run_script += f'    echo "  FAILED r2SCAN: {prefix}"\n'
    run_script += "    FAILED=$((FAILED + 1))\n"
    run_script += "  fi\n"
    run_script += "else\n"
    run_script += f'  echo "  FAILED PBE: {prefix} (skipping r2SCAN)"\n'
    run_script += "  FAILED=$((FAILED + 1))\n"
    run_script += "fi\n"

    # PBE+U if needed
    if job["needs_hubbard_u"]:
        pbe_u_dir = os.path.abspath(job["pbe_u_dir"])
        run_script += f"\ncd {pbe_u_dir} && bash generate_potcar.sh && cd -\n"
        run_script += f'echo "[$(date +%H:%M:%S)] {prefix}: PBE+U static ..."\n'
        run_script += f"cd {pbe_u_dir}\n"
        run_script += "mpirun -np $NPROCS vasp_std > vasp.log 2>&1\n"
        run_script += "if [ $? -ne 0 ]; then\n"
        run_script += f'  echo "  FAILED PBE+U: {prefix}"\n'
        run_script += "fi\n"

    run_script += "cd -\n\n"

run_script += 'echo ""\n'
run_script += 'echo "==================================="\n'
run_script += 'echo "MatPES VASP pipeline: $PASSED passed, $FAILED failed"\n'
run_script += 'echo "==================================="\n'

script_path = os.path.join(WORK_DIR, "run_matpes.sh")
with open(script_path, "w") as f:
    f.write(run_script)
os.chmod(script_path, 0o755)

print(f"\nGenerated inputs for {len(job_metadata)} structures in {WORK_DIR}/")
print(f"Batch run script: {script_path}")
print(f"\nTo run: NPROCS=4 bash {script_path}")
```

#### Step B2 -- Collect VASP results from PBE and r2SCAN

```python
#!/usr/bin/env python3
"""
Collect results from MatPES dual-functional VASP calculations.
Parses vasprun.xml from PBE and r2SCAN directories, compares energies.
"""

import os
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pymatgen.io.vasp.outputs import Vasprun

# ================================================================== #
WORK_DIR = "matpes_vasp"
OUTPUT_CSV = "matpes_vasp_results.csv"
# ================================================================== #

meta_path = os.path.join(WORK_DIR, "job_metadata.json")
with open(meta_path) as f:
    job_metadata = json.load(f)


def parse_vasp_dir(calc_dir):
    """Parse vasprun.xml from a VASP calculation directory."""
    result = {
        "total_energy_eV": None,
        "energy_per_atom_eV": None,
        "converged": False,
        "n_scf_steps": 0,
        "band_gap_eV": None,
        "error": None,
    }

    vasprun_file = os.path.join(calc_dir, "vasprun.xml")
    if not os.path.exists(vasprun_file):
        result["error"] = "vasprun.xml not found"
        return result

    try:
        vr = Vasprun(vasprun_file, parse_dos=False, parse_eigen=False)
        result["total_energy_eV"] = vr.final_energy
        n_atoms = len(vr.final_structure)
        result["energy_per_atom_eV"] = vr.final_energy / n_atoms
        result["converged"] = vr.converged_electronic
        result["n_scf_steps"] = len(vr.ionic_steps[0]["electronic_steps"])
        try:
            bs = vr.get_band_structure()
            result["band_gap_eV"] = bs.get_band_gap()["energy"]
        except Exception:
            pass
    except Exception as e:
        result["error"] = str(e)

    return result


rows = []
for job in job_metadata:
    prefix = job["prefix"]

    # PBE
    pbe = parse_vasp_dir(job["pbe_dir"])
    # r2SCAN
    r2s = parse_vasp_dir(job["r2scan_dir"])
    # PBE+U
    pbe_u = None
    if job["needs_hubbard_u"] and job["pbe_u_dir"]:
        pbe_u = parse_vasp_dir(job["pbe_u_dir"])

    delta_e = None
    if (pbe["energy_per_atom_eV"] is not None
            and r2s["energy_per_atom_eV"] is not None):
        delta_e = r2s["energy_per_atom_eV"] - pbe["energy_per_atom_eV"]

    row = {
        "index": job["index"],
        "prefix": prefix,
        "formula": job["formula"],
        "nsites": job["nsites"],
        "spacegroup": job["spacegroup"],
        # PBE
        "pbe_energy_per_atom_eV": pbe["energy_per_atom_eV"],
        "pbe_total_energy_eV": pbe["total_energy_eV"],
        "pbe_scf_steps": pbe["n_scf_steps"],
        "pbe_converged": pbe["converged"],
        "pbe_band_gap_eV": pbe["band_gap_eV"],
        # r2SCAN
        "r2scan_energy_per_atom_eV": r2s["energy_per_atom_eV"],
        "r2scan_total_energy_eV": r2s["total_energy_eV"],
        "r2scan_scf_steps": r2s["n_scf_steps"],
        "r2scan_converged": r2s["converged"],
        "r2scan_band_gap_eV": r2s["band_gap_eV"],
        # Delta
        "delta_E_r2scan_minus_pbe_eV": delta_e,
    }

    if pbe_u is not None:
        row["pbe_u_energy_per_atom_eV"] = pbe_u["energy_per_atom_eV"]
        row["pbe_u_scf_steps"] = pbe_u["n_scf_steps"]
        row["pbe_u_converged"] = pbe_u["converged"]
        row["pbe_u_band_gap_eV"] = pbe_u["band_gap_eV"]

    rows.append(row)

    pbe_e = f'{pbe["energy_per_atom_eV"]:.4f}' if pbe["energy_per_atom_eV"] else "N/A"
    r2s_e = f'{r2s["energy_per_atom_eV"]:.4f}' if r2s["energy_per_atom_eV"] else "N/A"
    delta_str = f"{delta_e:+.4f}" if delta_e is not None else "N/A"
    scf_info = f"PBE:{pbe['n_scf_steps']}, r2SCAN:{r2s['n_scf_steps']}"
    print(f"  [{job['index']+1}] {prefix:>20s}  "
          f"PBE={pbe_e}  r2SCAN={r2s_e}  delta={delta_str}  SCF({scf_info})")

df = pd.DataFrame(rows)
df.to_csv(OUTPUT_CSV, index=False)

df_ok = df[df["pbe_converged"] & df["r2scan_converged"]].copy()
print(f"\nResults saved to {OUTPUT_CSV}")
print(f"Both PBE+r2SCAN converged: {len(df_ok)}/{len(df)}")

if len(df_ok) > 0:
    deltas = df_ok["delta_E_r2scan_minus_pbe_eV"].dropna()
    print(f"\nr2SCAN - PBE energy difference:")
    print(f"  Mean:  {deltas.mean():+.4f} eV/atom")
    print(f"  Std:   {deltas.std():.4f} eV/atom")
    print(f"\nSCF iteration savings:")
    print(f"  PBE mean:    {df_ok['pbe_scf_steps'].mean():.1f} steps")
    print(f"  r2SCAN mean: {df_ok['r2scan_scf_steps'].mean():.1f} steps "
          f"(with PBE warm start)")

# Visualization
if len(df_ok) >= 2:
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    x = df_ok["pbe_energy_per_atom_eV"].values
    y = df_ok["r2scan_energy_per_atom_eV"].values
    axes[0].scatter(x, y, s=50, edgecolors="black", zorder=5)
    lim = [min(x.min(), y.min()) - 0.2, max(x.max(), y.max()) + 0.2]
    axes[0].plot(lim, lim, "k--", alpha=0.4)
    axes[0].set_xlabel("PBE energy/atom (eV)")
    axes[0].set_ylabel("r2SCAN energy/atom (eV)")
    axes[0].set_title("PBE vs r2SCAN (VASP)")

    deltas_arr = deltas.values
    axes[1].hist(deltas_arr, bins=max(5, len(df_ok) // 3),
                 edgecolor="black", color="C1", alpha=0.8)
    axes[1].axvline(deltas.mean(), color="red", linestyle="--",
                    label=f"mean={deltas.mean():+.3f}")
    axes[1].set_xlabel("r2SCAN - PBE (eV/atom)")
    axes[1].set_ylabel("Count")
    axes[1].set_title("Energy difference distribution")
    axes[1].legend()

    pbe_s = df_ok["pbe_scf_steps"].values
    r2s_s = df_ok["r2scan_scf_steps"].values
    x_pos = np.arange(len(df_ok))
    w = 0.35
    axes[2].bar(x_pos - w/2, pbe_s, w, label="PBE", edgecolor="black")
    axes[2].bar(x_pos + w/2, r2s_s, w, label="r2SCAN", edgecolor="black")
    axes[2].set_xlabel("Structure index")
    axes[2].set_ylabel("SCF iterations")
    axes[2].set_title("SCF steps: PBE vs r2SCAN (warm start)")
    axes[2].legend()

    plt.tight_layout()
    plt.savefig("matpes_vasp_summary.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("Saved matpes_vasp_summary.png")
```

### Method C: ASE + MACE Comparison

Quick MACE single-point calculations for comparison and validation before running expensive DFT. Useful for sanity-checking structures and getting a rapid energy baseline.

```python
#!/usr/bin/env python3
"""
Quick MACE single-point energy for comparison with MatPES DFT results.

Runs MACE-MP-0 on each structure to get a fast energy baseline.
Compare with PBE and r2SCAN DFT energies to check for outliers
or structural issues before committing to expensive calculations.
"""

import os
import glob
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from ase.io import read as ase_read
from pymatgen.core import Structure
from pymatgen.io.ase import AseAtomsAdaptor

# ================================================================== #
#  CONFIGURATION
# ================================================================== #
INPUT_DIR = "structures"          # directory with CIF/POSCAR files
MACE_MODEL = "medium"            # "small", "medium", "large"
OUTPUT_CSV = "matpes_mace_comparison.csv"

# If MatPES DFT results are available, provide path for comparison
DFT_RESULTS_CSV = None           # e.g., "matpes_vasp_results.csv"
# ================================================================== #

from mace.calculators import mace_mp

calc = mace_mp(model=MACE_MODEL, default_dtype="float64")
adaptor = AseAtomsAdaptor()

# --- Load structures ---
structure_files = sorted(
    glob.glob(os.path.join(INPUT_DIR, "*.cif"))
    + glob.glob(os.path.join(INPUT_DIR, "*.vasp"))
    + glob.glob(os.path.join(INPUT_DIR, "POSCAR*"))
    + glob.glob(os.path.join(INPUT_DIR, "*.json"))
)

if not structure_files:
    print(f"No structure files found in {INPUT_DIR}/")
    exit(1)

rows = []
for idx, filepath in enumerate(structure_files):
    basename = os.path.splitext(os.path.basename(filepath))[0]
    prefix = basename.replace("-", "_").replace(" ", "_")

    try:
        structure = Structure.from_file(filepath)
        atoms = adaptor.get_atoms(structure)
        atoms.calc = calc

        energy = atoms.get_potential_energy()
        energy_per_atom = energy / len(atoms)
        forces = atoms.get_forces()
        max_force = np.max(np.linalg.norm(forces, axis=1))
        stress = atoms.get_stress()  # Voigt notation, eV/Ang^3

        row = {
            "index": idx,
            "prefix": prefix,
            "formula": structure.composition.reduced_formula,
            "nsites": len(structure),
            "mace_energy_eV": energy,
            "mace_energy_per_atom_eV": energy_per_atom,
            "mace_max_force_eV_Ang": max_force,
            "mace_pressure_GPa": -np.mean(stress[:3]) * 160.2176634,
        }
        rows.append(row)

        print(f"  [{idx+1}/{len(structure_files)}] {prefix:>20s}  "
              f"E={energy_per_atom:.4f} eV/atom  "
              f"Fmax={max_force:.4f} eV/Ang")

    except Exception as e:
        print(f"  [{idx+1}/{len(structure_files)}] {prefix}: ERROR - {e}")

df_mace = pd.DataFrame(rows)
df_mace.to_csv(OUTPUT_CSV, index=False)
print(f"\nMACE single-point results saved to {OUTPUT_CSV}")
print(f"Computed: {len(rows)}/{len(structure_files)}")

# --- Compare with DFT if available ---
if DFT_RESULTS_CSV and os.path.exists(DFT_RESULTS_CSV):
    df_dft = pd.read_csv(DFT_RESULTS_CSV)
    df_merged = df_mace.merge(df_dft, on="prefix", how="inner",
                               suffixes=("_mace", "_dft"))

    if len(df_merged) >= 2:
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))

        # MACE vs PBE
        if "pbe_energy_per_atom_eV" in df_merged.columns:
            x = df_merged["mace_energy_per_atom_eV"].values
            y = df_merged["pbe_energy_per_atom_eV"].values
            axes[0].scatter(x, y, s=50, edgecolors="black")
            lim = [min(x.min(), y.min()) - 0.2,
                   max(x.max(), y.max()) + 0.2]
            axes[0].plot(lim, lim, "k--", alpha=0.4)
            axes[0].set_xlabel("MACE energy/atom (eV)")
            axes[0].set_ylabel("PBE energy/atom (eV)")
            axes[0].set_title("MACE vs PBE")

        # MACE vs r2SCAN
        if "r2scan_energy_per_atom_eV" in df_merged.columns:
            x = df_merged["mace_energy_per_atom_eV"].values
            y = df_merged["r2scan_energy_per_atom_eV"].values
            axes[1].scatter(x, y, s=50, edgecolors="black", color="C1")
            lim = [min(x.min(), y.min()) - 0.2,
                   max(x.max(), y.max()) + 0.2]
            axes[1].plot(lim, lim, "k--", alpha=0.4)
            axes[1].set_xlabel("MACE energy/atom (eV)")
            axes[1].set_ylabel("r2SCAN energy/atom (eV)")
            axes[1].set_title("MACE vs r2SCAN")

        plt.tight_layout()
        plt.savefig("matpes_mace_vs_dft.png", dpi=150, bbox_inches="tight")
        plt.close()
        print("Saved matpes_mace_vs_dft.png")

        df_merged.to_csv("matpes_comparison_merged.csv", index=False)
        print("Saved matpes_comparison_merged.csv")
else:
    print("\nNo DFT results CSV provided for comparison.")
    print("Set DFT_RESULTS_CSV to compare after running DFT pipeline.")
```

## Key Parameters

### VASP-Specific Parameters

| Parameter | PBE Value | r2SCAN Value | Notes |
|---|---|---|---|
| `ENCUT` | 520 eV | 680 eV | Higher cutoff needed for meta-GGA accuracy |
| `METAGGA` | (not set) | `R2SCAN` | Activates r2SCAN functional |
| `LASPH` | (optional) | `True` | **Required** for meta-GGA -- aspherical PAW contributions |
| `LWAVE` | `True` | `False` | PBE must save WAVECAR; r2SCAN reads it |
| `ISTART` | 0 | 1 | 1 = read WAVECAR from previous calculation |
| `ICHARG` | 2 | 1 | 1 = read CHGCAR, recalculate from WAVECAR |
| `ALGO` | `Normal` | `All` | `All` (Davidson + RMM-DIIS) recommended for meta-GGA |
| `ISPIN` | 2 | 2 | Spin-polarized for magnetic systems |
| `SIGMA` | 0.05 eV | 0.05 eV | Gaussian smearing width |
| `EDIFF` | 1e-6 eV | 1e-6 eV | SCF convergence criterion |

### QE-Specific Parameters

| Parameter | PBE Value | r2SCAN Value | Notes |
|---|---|---|---|
| `ecutwfc` | 64 Ry | 72 Ry | Higher for meta-GGA |
| `input_dft` | (default PBE) | `'r2scan'` | Sets r2SCAN functional (QE 7.2+) |
| `disk_io` | `'high'` | `'low'` | PBE saves wavefunctions; r2SCAN reads them |
| `startingwfc` | (default) | `'file'` | Read PBE wavefunctions as starting point |
| `conv_thr` | 1e-8 Ry | 1e-8 Ry | SCF convergence |
| `mixing_beta` | 0.4 | 0.4 | Mixing parameter for SCF |

### PBE+U Parameters

| Parameter | Value | Notes |
|---|---|---|
| `LDAU` / `lda_plus_u` | `True` | Enable DFT+U |
| `LDAUTYPE` / `lda_plus_u_kind` | 2 / 0 | Dudarev (simplified) approach |
| U values | See `HUBBARD_U` dict | Materials Project conventions (Wang et al.) |

## Interpreting Results

- **r2SCAN vs PBE energy difference**: Typically r2SCAN gives more negative (lower) energies per atom than PBE. The shift is composition-dependent but usually -0.1 to -0.5 eV/atom. The important quantity is the *relative* ordering of phases, not absolute energies.
- **SCF iteration count**: With PBE warm start, r2SCAN should converge in 10--30 SCF steps compared to 60--100 from scratch. If r2SCAN takes more steps than PBE, the wavefunction restart may not be working (check that WAVECAR was copied correctly or that `startingwfc='file'` is set).
- **Phase ordering changes**: If PBE and r2SCAN disagree on which polymorph is most stable, r2SCAN is generally more reliable for relative energies. This is a valuable finding for your dataset.
- **PBE+U energies**: These are on a different energy scale than bare PBE. Do not directly compare PBE+U with PBE or r2SCAN total energies. PBE+U is useful for band gaps and magnetic ordering in correlated systems.
- **Band gaps**: PBE underestimates band gaps by 30--50%. r2SCAN improves slightly but is still not quantitative. Use PBE+U or hybrid functionals for reliable gaps.
- **WAVECAR file size**: The WAVECAR file can be very large (hundreds of MB to GB). After the r2SCAN calculation completes, the PBE WAVECAR can be deleted to save disk space.

## Common Issues

| Problem | Cause | Solution |
|---|---|---|
| r2SCAN SCF not converging despite warm start | WAVECAR not copied or incompatible | Verify WAVECAR exists in r2SCAN directory; check ISTART=1 in INCAR; ensure same k-grid and ENCUT >= PBE ENCUT |
| `METAGGA` tag not recognized | VASP version too old | r2SCAN requires VASP 6.2+; use SCAN for VASP 5.4.4+ |
| `input_dft='r2scan'` fails in QE | QE version too old | r2SCAN requires QE 7.2+; use `input_dft='scan'` for older versions |
| WAVECAR file too large | Many bands, high ENCUT, large cell | Use `NBANDS` to limit number of bands; consider LWAVE=False for PBE and use only CHGCAR restart (ICHARG=1) -- slightly less effective but smaller files |
| `LASPH` not set for r2SCAN | Missing required tag | Always set `LASPH=True` for any meta-GGA calculation; results without it are unreliable |
| Negative pressure / wrong stress | Meta-GGA stress requires LASPH | Set `LASPH=True`; also check that ENCUT is sufficient |
| PBE+U elements not detected | Element not in `HUBBARD_U` dict | Add the element and its U value to the dictionary; consult literature for appropriate U values |
| r2SCAN takes as many steps as from scratch | ecutwfc/ENCUT mismatch between PBE and r2SCAN | r2SCAN ENCUT must be >= PBE ENCUT for the WAVECAR to be useful; if much larger, extra plane waves start from scratch |
| Memory error during WAVECAR copy | Large WAVECAR on limited disk | Use `mv` instead of `cp` for WAVECAR (moves instead of copies); or increase disk allocation |
| Batch script fails partway | One structure crashes and blocks the rest | Add `set +e` at top of script to continue on errors; use `timeout 7200 mpirun ...` to cap individual job time |
