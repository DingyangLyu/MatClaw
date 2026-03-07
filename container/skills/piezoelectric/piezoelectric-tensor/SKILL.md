# Piezoelectric Tensor

## When to Use

Use this skill to compute the piezoelectric stress tensor **e_ij** (C/m^2) of a non-centrosymmetric crystalline material. The piezoelectric tensor relates the macroscopic polarization change to an applied strain:

e_ij = dP_i / d(epsilon_j)

where i = x, y, z (polarization direction) and j = 1..6 (Voigt strain index: xx, yy, zz, yz, xz, xy).

Typical target materials: ZnO (wurtzite), BaTiO3 (tetragonal), AlN (wurtzite), GaN, PbTiO3.

## Method Selection

| Method | Description | When to Use |
|--------|-------------|-------------|
| **QE Berry Phase (finite differences)** | Apply small strains, compute polarization via Berry phase at each strain, differentiate | Gold standard for piezoelectric tensors. Works for any insulator/semiconductor. |
| **QE DFPT (linear response)** | Use `ph.x` with electric field perturbations | Faster but requires careful setup; less transparent |

This skill covers the **Berry phase finite-difference** approach, which is more robust and easier to validate.

### Clamped-ion vs Relaxed-ion

- **Clamped-ion (electronic):** Apply strain, do NOT relax internal coordinates. Captures electronic redistribution only.
- **Relaxed-ion (total):** Apply strain, relax internal coordinates, then compute polarization. Includes ionic contribution from internal displacements.
- **Ionic contribution:** e_relaxed - e_clamped. Often dominates (especially in perovskites).

## Prerequisites

- Quantum ESPRESSO 7.5 with `pw.x` (Berry phase via `lberry=.true.`)
- Python packages: `numpy`, `pymatgen`, `matplotlib`
- Pseudopotentials (norm-conserving recommended for Berry phase; PAW also works in modern QE)

**Important:** The Berry phase approach requires an **insulating** system (band gap > 0). Metals do not have a well-defined polarization.

---

## Detailed Steps

### Overview

1. Relax the structure at zero strain (QE `vc-relax`).
2. For each of the 6 Voigt strain components (xx, yy, zz, yz, xz, xy):
   a. Apply +delta and -delta strain to the cell.
   b. (Relaxed-ion) Relax internal coordinates at fixed cell.
   c. Compute Berry phase polarization P_x, P_y, P_z.
3. Compute e_ij = (P_i(+delta) - P_i(-delta)) / (2 * delta).
4. Assemble the 3x6 piezoelectric tensor.

### Complete Workflow Script

```python
#!/usr/bin/env python3
"""
Piezoelectric tensor via QE Berry phase (finite strain differences).
Complete workflow: generates all QE inputs, runner script, and post-processing.

Produces:
  - Relaxation input
  - 12 strained structure inputs (6 strains x 2 signs)
  - For each: optional internal relaxation + Berry phase calculation
  - Post-processing to extract e_ij
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pymatgen.core import Structure, Lattice
from pymatgen.core.periodic_table import Element as PmgElement
from pathlib import Path
import json
import copy

# ── 0. Configuration ──────────────────────────────────────────────
STRUCTURE_FILE   = "POSCAR"              # Input structure
PSEUDO_DIR       = "/home/pps/"
ECUTWFC          = 60.0                  # Ry
ECUTRHO          = 480.0                 # Ry
KPOINTS_RELAX    = [8, 8, 8]             # k-mesh for relaxation
KPOINTS_SCF      = [8, 8, 8]             # k-mesh for SCF
KPOINTS_BERRY    = [8, 8, 8]             # k-mesh for Berry phase (dense along polarization dir)
STRAIN_MAGNITUDE = 0.005                 # Applied strain (dimensionless). 0.005 = 0.5%
CONV_THR         = 1.0e-10               # Tight SCF convergence for polarization
FORC_CONV_THR    = 1.0e-5                # Ry/Bohr for internal relaxation
NBERRYCYC        = 1                     # Berry phase cycles (increase if convergence issues)
QE_CMD           = "mpirun -np 4 pw.x"
OUTPUT_DIR       = Path("piezo_berry_phase")
DO_RELAXED_ION   = True                 # Compute relaxed-ion response (True recommended)
DO_CLAMPED_ION   = True                 # Also compute clamped-ion response
# ──────────────────────────────────────────────────────────────────

OUTPUT_DIR.mkdir(exist_ok=True)

# Pseudopotential mapping (adjust for your pseudopotential library)
PSEUDO_MAP = {
    "Si": "Si.pbe-n-rrkjus_psl.1.0.0.UPF",
    "Ge": "Ge.pbe-dn-rrkjus_psl.1.0.0.UPF",
    "C":  "C.pbe-n-kjpaw_psl.1.0.0.UPF",
    "O":  "O.pbe-n-kjpaw_psl.1.0.0.UPF",
    "N":  "N.pbe-n-kjpaw_psl.1.0.0.UPF",
    "Al": "Al.pbe-n-kjpaw_psl.1.0.0.UPF",
    "Ga": "Ga.pbe-dn-kjpaw_psl.1.0.0.UPF",
    "Zn": "Zn.pbe-dn-kjpaw_psl.1.0.0.UPF",
    "Ti": "Ti.pbe-spn-kjpaw_psl.1.0.0.UPF",
    "Ba": "Ba.pbe-spn-kjpaw_psl.1.0.0.UPF",
    "Pb": "Pb.pbe-dn-kjpaw_psl.1.0.0.UPF",
}

def get_pseudo(element):
    if element in PSEUDO_MAP:
        return PSEUDO_MAP[element]
    return f"{element}.pbe-n-kjpaw_psl.1.0.0.UPF"


# Voigt strain definitions
# Index: 1=xx, 2=yy, 3=zz, 4=yz, 5=xz, 6=xy
VOIGT_LABELS = ["xx", "yy", "zz", "yz", "xz", "xy"]

def voigt_strain_matrix(voigt_index, magnitude):
    """
    Return the 3x3 strain tensor for a given Voigt index (0-5) and magnitude.

    Voigt convention:
      0: eps_xx     -> e_11
      1: eps_yy     -> e_22
      2: eps_zz     -> e_33
      3: eps_yz     -> e_23 = e_32 (engineering strain / 2)
      4: eps_xz     -> e_13 = e_31
      5: eps_xy     -> e_12 = e_21

    The deformation gradient is F = I + eps, so the strained cell is:
      cell_strained = F @ cell_original
    """
    eps = np.zeros((3, 3))
    if voigt_index == 0:    # xx
        eps[0, 0] = magnitude
    elif voigt_index == 1:  # yy
        eps[1, 1] = magnitude
    elif voigt_index == 2:  # zz
        eps[2, 2] = magnitude
    elif voigt_index == 3:  # yz
        eps[1, 2] = magnitude / 2.0
        eps[2, 1] = magnitude / 2.0
    elif voigt_index == 4:  # xz
        eps[0, 2] = magnitude / 2.0
        eps[2, 0] = magnitude / 2.0
    elif voigt_index == 5:  # xy
        eps[0, 1] = magnitude / 2.0
        eps[1, 0] = magnitude / 2.0
    return eps


def apply_strain(structure, voigt_index, magnitude):
    """
    Apply a homogeneous strain to a pymatgen Structure.
    Returns new Structure with strained cell. Fractional coordinates preserved
    (clamped-ion approximation).
    """
    eps = voigt_strain_matrix(voigt_index, magnitude)
    F = np.eye(3) + eps  # Deformation gradient
    new_matrix = structure.lattice.matrix @ F.T  # Row-vector convention
    new_lattice = Lattice(new_matrix)
    # Keep fractional coordinates (clamped-ion)
    return Structure(
        new_lattice,
        structure.species,
        structure.frac_coords,
        coords_are_cartesian=False,
    )


# ── 1. Load structure ─────────────────────────────────────────────
struct = Structure.from_file(STRUCTURE_FILE)
elements = sorted(set(str(s) for s in struct.species))
print(f"Structure: {struct.formula}, {struct.num_sites} atoms")
print(f"Space group: {struct.get_space_group_info()}")

# Check: must be non-centrosymmetric for piezoelectricity
import spglib
spg_cell = (
    struct.lattice.matrix,
    struct.frac_coords,
    [s.Z for s in struct.species],
)
sym_data = spglib.get_symmetry_dataset(spg_cell)
if sym_data is not None:
    pg = sym_data['pointgroup']
    print(f"Point group: {pg}")
    # Centrosymmetric point groups
    centrosymmetric = ['-1', '2/m', 'mmm', '4/m', '4/mmm', '-3', '-3m',
                       '6/m', '6/mmm', 'm-3', 'm-3m']
    if pg in centrosymmetric:
        print("WARNING: Structure is centrosymmetric! Piezoelectric tensor will be zero.")
        print("         Check that this is the correct structure.")

# ── 2. Write relaxation input ─────────────────────────────────────
def write_qe_input(struct_pmg, filename, calc_type, prefix, kpoints,
                   lberry=False, gdir=1, nppstr=8, relax_ions_only=False):
    """Write a QE input file."""

    elems = sorted(set(str(s) for s in struct_pmg.species))

    control_block = f"""&CONTROL
  calculation  = '{calc_type}'
  prefix       = '{prefix}'
  outdir       = './tmp_{prefix}'
  pseudo_dir   = '{PSEUDO_DIR}'
  tprnfor      = .true.
  tstress      = .true.
"""
    if calc_type == 'vc-relax':
        control_block += f"  forc_conv_thr = {FORC_CONV_THR}\n"
        control_block += f"  etot_conv_thr = 1.0d-6\n"
    if lberry:
        control_block += f"  lberry       = .true.\n"
        control_block += f"  gdir         = {gdir}\n"
        control_block += f"  nppstr       = {nppstr}\n"
        control_block += f"  nberrycyc    = {NBERRYCYC}\n"
    control_block += "/\n"

    system_block = f"""&SYSTEM
  ibrav        = 0
  nat          = {struct_pmg.num_sites}
  ntyp         = {len(elems)}
  ecutwfc      = {ECUTWFC}
  ecutrho      = {ECUTRHO}
  occupations  = 'fixed'
  nosym        = .true.
  noinv        = .true.
/
"""

    electrons_block = f"""&ELECTRONS
  conv_thr     = {CONV_THR}
  mixing_beta  = 0.7
/
"""

    ions_block = ""
    cell_block = ""
    if calc_type in ('relax', 'vc-relax'):
        ions_block = """&IONS
  ion_dynamics = 'bfgs'
/
"""
    if calc_type == 'vc-relax':
        cell_block = """&CELL
  cell_dynamics = 'bfgs'
  press         = 0.0
/
"""
    if calc_type == 'relax' and relax_ions_only:
        # For relaxed-ion: relax ions at fixed cell
        pass  # ions_block already set, no cell block

    input_text = control_block + system_block + electrons_block + ions_block + cell_block

    input_text += "\nATOMIC_SPECIES\n"
    for el in elems:
        atomic_mass = PmgElement(el).atomic_mass
        input_text += f"  {el}  {atomic_mass:.4f}  {get_pseudo(el)}\n"

    input_text += "\nATOMIC_POSITIONS crystal\n"
    for site in struct_pmg:
        el = str(site.specie)
        input_text += f"  {el}  {site.frac_coords[0]:.10f}  {site.frac_coords[1]:.10f}  {site.frac_coords[2]:.10f}\n"

    input_text += f"\nK_POINTS automatic\n"
    input_text += f"  {kpoints[0]} {kpoints[1]} {kpoints[2]}  0 0 0\n"

    input_text += "\nCELL_PARAMETERS angstrom\n"
    for vec in struct_pmg.lattice.matrix:
        input_text += f"  {vec[0]:.10f}  {vec[1]:.10f}  {vec[2]:.10f}\n"

    with open(filename, "w") as f:
        f.write(input_text)


# Relaxation input
write_qe_input(struct, OUTPUT_DIR / "relax.in", "vc-relax", "relax", KPOINTS_RELAX)
print(f"Written: {OUTPUT_DIR / 'relax.in'}")

# ── 3. Generate strained structures and QE inputs ─────────────────
strain_dir = OUTPUT_DIR / "strained"
strain_dir.mkdir(exist_ok=True)

print(f"\nGenerating strained structures (delta = +/-{STRAIN_MAGNITUDE})...")

for v_idx in range(6):
    for sign_label, sign in [("pos", +1), ("neg", -1)]:
        delta = sign * STRAIN_MAGNITUDE
        strained = apply_strain(struct, v_idx, delta)
        label = f"strain_{VOIGT_LABELS[v_idx]}_{sign_label}"

        # Clamped-ion: SCF + Berry phase (no ionic relaxation)
        if DO_CLAMPED_ION:
            for gdir in [1, 2, 3]:
                fname = strain_dir / f"{label}_clamped_berry_gdir{gdir}.in"
                # nppstr: number of k-points along the Berry phase string
                # Should be dense enough (typically 6-10 for each reciprocal direction)
                nppstr = KPOINTS_BERRY[gdir - 1]
                write_qe_input(
                    strained, fname, "scf", f"{label}_c_g{gdir}",
                    KPOINTS_SCF, lberry=True, gdir=gdir, nppstr=nppstr,
                )

        # Relaxed-ion: relax internal coordinates at fixed cell, then Berry phase
        if DO_RELAXED_ION:
            # Step 1: Relax ions at fixed strained cell
            fname_relax = strain_dir / f"{label}_relax.in"
            write_qe_input(
                strained, fname_relax, "relax", f"{label}_r",
                KPOINTS_SCF, relax_ions_only=True,
            )
            # Berry phase inputs will be generated after parsing relaxed positions
            # (see runner script below)

print(f"  Written {len(list(strain_dir.glob('*.in')))} input files to {strain_dir}/")

# ── 4. Also write reference Berry phase at zero strain ─────────────
for gdir in [1, 2, 3]:
    fname = OUTPUT_DIR / f"ref_berry_gdir{gdir}.in"
    nppstr = KPOINTS_BERRY[gdir - 1]
    write_qe_input(
        struct, fname, "scf", f"ref_g{gdir}",
        KPOINTS_SCF, lberry=True, gdir=gdir, nppstr=nppstr,
    )
print("Written reference (zero-strain) Berry phase inputs.")

# ── 5. Runner script ──────────────────────────────────────────────
runner = f"""#!/bin/bash
# Run all QE calculations for piezoelectric tensor
# Usage: bash run_piezo.sh

QE_CMD="{QE_CMD}"
STRAIN_DIR="{strain_dir}"
OUTPUT_DIR="{OUTPUT_DIR}"

echo "=== Step 1: Relaxation ==="
cd $OUTPUT_DIR
mkdir -p tmp_relax
$QE_CMD < relax.in > relax.out 2>&1
echo "Relaxation done."

echo "=== Step 2: Reference Berry phase (zero strain) ==="
for gdir in 1 2 3; do
    echo "  gdir=$gdir"
    mkdir -p tmp_ref_g${{gdir}}
    $QE_CMD < ref_berry_gdir${{gdir}}.in > ref_berry_gdir${{gdir}}.out 2>&1
done

echo "=== Step 3: Clamped-ion Berry phase calculations ==="
cd $STRAIN_DIR
for inp in *_clamped_berry_gdir*.in; do
    out="${{inp%.in}}.out"
    if [ ! -f "$out" ]; then
        echo "  Running $inp"
        prefix=$(echo $inp | sed 's/.in//')
        mkdir -p tmp_$prefix
        $QE_CMD < $inp > $out 2>&1
    fi
done

echo "=== Step 4: Relaxed-ion calculations ==="
# First relax internal coordinates
for inp in *_relax.in; do
    out="${{inp%.in}}.out"
    if [ ! -f "$out" ]; then
        echo "  Relaxing $inp"
        prefix=$(echo $inp | sed 's/.in//')
        mkdir -p tmp_$prefix
        $QE_CMD < $inp > $out 2>&1
    fi
done

# Then run Berry phase on relaxed structures
# (Post-processing script generates these inputs after parsing relaxed positions)
echo "  Now run: python generate_relaxed_berry.py"
echo "  Then re-run this script section for relaxed Berry phase"

for inp in *_relaxed_berry_gdir*.in; do
    if [ -f "$inp" ]; then
        out="${{inp%.in}}.out"
        if [ ! -f "$out" ]; then
            echo "  Running $inp"
            prefix=$(echo $inp | sed 's/.in//')
            mkdir -p tmp_$prefix
            $QE_CMD < $inp > $out 2>&1
        fi
    fi
done

echo "=== All piezoelectric calculations complete ==="
echo "Run: python postprocess_piezo.py"
"""

with open(OUTPUT_DIR / "run_piezo.sh", "w") as f:
    f.write(runner)
print(f"Written: {OUTPUT_DIR / 'run_piezo.sh'}")

# ── 6. Script to generate Berry phase inputs from relaxed structures ──
gen_relaxed_script = f'''#!/usr/bin/env python3
"""
After internal coordinate relaxation at each strained cell,
parse the relaxed positions and generate Berry phase inputs.
"""
import numpy as np
from pymatgen.core import Structure, Lattice
from pymatgen.core.periodic_table import Element as PmgElement
from pathlib import Path

STRAIN_DIR = Path("{strain_dir}")
PSEUDO_DIR = "{PSEUDO_DIR}"
ECUTWFC = {ECUTWFC}
ECUTRHO = {ECUTRHO}
KPOINTS_SCF = {KPOINTS_SCF}
KPOINTS_BERRY = {KPOINTS_BERRY}
CONV_THR = {CONV_THR}
NBERRYCYC = {NBERRYCYC}
VOIGT_LABELS = {VOIGT_LABELS}

PSEUDO_MAP = {json.dumps(PSEUDO_MAP)}

def get_pseudo(element):
    return PSEUDO_MAP.get(element, f"{{element}}.pbe-n-kjpaw_psl.1.0.0.UPF")

def parse_relaxed_structure(qe_output_file):
    """Parse final relaxed structure from QE relax output."""
    lines = open(qe_output_file).readlines()
    cell = np.zeros((3, 3))
    positions = []
    species = []

    # Find final CELL_PARAMETERS
    cell_found = False
    for i in range(len(lines)-1, -1, -1):
        if "CELL_PARAMETERS" in lines[i]:
            scale = 1.0
            if "bohr" in lines[i].lower():
                scale = 0.529177249
            for j in range(3):
                vals = lines[i+1+j].split()
                cell[j] = [float(v) * scale for v in vals]
            cell_found = True
            break

    if not cell_found:
        # Cell did not change (relax, not vc-relax): parse from input
        for i, line in enumerate(lines):
            if "CELL_PARAMETERS" in line:
                scale = 1.0
                if "bohr" in line.lower():
                    scale = 0.529177249
                for j in range(3):
                    vals = lines[i+1+j].split()
                    cell[j] = [float(v) * scale for v in vals]
                cell_found = True
                break

    # Find final ATOMIC_POSITIONS
    for i in range(len(lines)-1, -1, -1):
        if "ATOMIC_POSITIONS" in lines[i]:
            coord_type = "crystal" if "crystal" in lines[i].lower() else "angstrom"
            j = i + 1
            while j < len(lines):
                parts = lines[j].split()
                if len(parts) >= 4 and parts[0].isalpha():
                    species.append(parts[0])
                    positions.append([float(parts[1]), float(parts[2]), float(parts[3])])
                elif len(parts) < 4 and positions:
                    break
                j += 1
            break

    lattice = Lattice(cell)
    if coord_type == "crystal":
        return Structure(lattice, species, positions)
    else:
        return Structure(lattice, species, positions, coords_are_cartesian=True)


def write_berry_input(struct_pmg, filename, prefix, kpoints, gdir, nppstr):
    elems = sorted(set(str(s) for s in struct_pmg.species))
    input_text = f"""&CONTROL
  calculation  = 'scf'
  prefix       = '{{prefix}}'
  outdir       = './tmp_{{prefix}}'
  pseudo_dir   = '{{PSEUDO_DIR}}'
  tprnfor      = .true.
  tstress      = .true.
  lberry       = .true.
  gdir         = {{gdir}}
  nppstr       = {{nppstr}}
  nberrycyc    = {{NBERRYCYC}}
/
&SYSTEM
  ibrav        = 0
  nat          = {{struct_pmg.num_sites}}
  ntyp         = {{len(elems)}}
  ecutwfc      = {{ECUTWFC}}
  ecutrho      = {{ECUTRHO}}
  occupations  = 'fixed'
  nosym        = .true.
  noinv        = .true.
/
&ELECTRONS
  conv_thr     = {{CONV_THR}}
  mixing_beta  = 0.7
/

ATOMIC_SPECIES
"""
    for el in elems:
        atomic_mass = PmgElement(el).atomic_mass
        input_text += f"  {{el}}  {{atomic_mass:.4f}}  {{get_pseudo(el)}}\\n"

    input_text += "\\nATOMIC_POSITIONS crystal\\n"
    for site in struct_pmg:
        el = str(site.specie)
        input_text += f"  {{el}}  {{site.frac_coords[0]:.10f}}  {{site.frac_coords[1]:.10f}}  {{site.frac_coords[2]:.10f}}\\n"

    input_text += f"\\nK_POINTS automatic\\n"
    input_text += f"  {{kpoints[0]}} {{kpoints[1]}} {{kpoints[2]}}  0 0 0\\n"

    input_text += "\\nCELL_PARAMETERS angstrom\\n"
    for vec in struct_pmg.lattice.matrix:
        input_text += f"  {{vec[0]:.10f}}  {{vec[1]:.10f}}  {{vec[2]:.10f}}\\n"

    with open(filename, "w") as f:
        f.write(input_text)


# Process each relaxed output
for v_idx, vlabel in enumerate(VOIGT_LABELS):
    for sign_label in ["pos", "neg"]:
        relax_out = STRAIN_DIR / f"strain_{{vlabel}}_{{sign_label}}_relax.out"
        if not relax_out.exists():
            print(f"  Skipping {{relax_out}} (not found)")
            continue

        try:
            relaxed_struct = parse_relaxed_structure(relax_out)
            label = f"strain_{{vlabel}}_{{sign_label}}"
            for gdir in [1, 2, 3]:
                nppstr = KPOINTS_BERRY[gdir - 1]
                fname = STRAIN_DIR / f"{{label}}_relaxed_berry_gdir{{gdir}}.in"
                write_berry_input(
                    relaxed_struct, fname, f"{{label}}_rx_g{{gdir}}",
                    KPOINTS_SCF, gdir, nppstr,
                )
            print(f"  Generated Berry phase inputs for {{label}}")
        except Exception as e:
            print(f"  ERROR parsing {{relax_out}}: {{e}}")

print("Done. Now run the Berry phase calculations.")
'''

with open(OUTPUT_DIR / "generate_relaxed_berry.py", "w") as f:
    f.write(gen_relaxed_script)
print(f"Written: {OUTPUT_DIR / 'generate_relaxed_berry.py'}")

# ── 7. Post-processing script ─────────────────────────────────────
postproc = f'''#!/usr/bin/env python3
"""
Post-process Berry phase outputs to extract the piezoelectric tensor.

Parses QE Berry phase polarization, computes e_ij = dP_i / d(eps_j).
"""
import numpy as np
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
from pymatgen.core import Structure

OUTPUT_DIR = Path("{OUTPUT_DIR}")
STRAIN_DIR = Path("{strain_dir}")
STRAIN_MAGNITUDE = {STRAIN_MAGNITUDE}
VOIGT_LABELS = {VOIGT_LABELS}
STRUCTURE_FILE = "{STRUCTURE_FILE}"

# Physical constants
E_CHARGE = 1.602176634e-19       # C
BOHR_TO_M = 0.529177249e-10      # m
ANG_TO_M = 1.0e-10               # m


def parse_berry_phase_polarization(qe_output_file):
    """
    Parse the Berry phase polarization from QE output.

    QE prints the electronic + ionic polarization in the output.
    The Berry phase is given as a phase (in units of 2*pi) which maps
    to a polarization modulo a quantum.

    Returns the polarization component along gdir in C/m^2.
    """
    lines = open(qe_output_file).readlines()

    # Look for "P = " line or "POLARIZATION" section
    pol_value = None
    pol_quantum = None

    for i, line in enumerate(lines):
        # QE Berry phase output format:
        # "VALUES OF POLARIZATION"
        # or directly "P = xxx (mod yyy) (C/m^2)"
        if "P =" in line and "C/m^2" in line:
            # Parse: "P =   0.12345678  (mod   1.23456789)  (C/m^2)"
            parts = line.split("P =")[1]
            pol_str = parts.split("(mod")[0].strip()
            pol_value = float(pol_str)
            mod_str = parts.split("(mod")[1].split(")")[0].strip()
            pol_quantum = float(mod_str)

        # Alternative format in some QE versions
        if "Electronic" in line and "Polarization" in line:
            # Look for total polarization nearby
            pass

        # Another format: "The polarization direction is:"
        # followed by component values
        if "POLARIZATION" in line.upper() and "BERRY" in line.upper():
            # Scan for P value
            for j in range(i, min(i+20, len(lines))):
                if "P =" in lines[j]:
                    parts = lines[j].split("P =")[1]
                    pol_str = parts.split("(")[0].strip()
                    pol_value = float(pol_str)
                    if "mod" in lines[j]:
                        mod_str = parts.split("mod")[1].split(")")[0].strip()
                        pol_quantum = float(mod_str)
                    break

    if pol_value is None:
        # Try parsing from "Ionic + Electronic" line
        for line in lines:
            if "ionic" in line.lower() and "electronic" in line.lower() and "=" in line:
                parts = line.split("=")[-1].strip()
                try:
                    pol_value = float(parts.split()[0])
                except (ValueError, IndexError):
                    pass

    if pol_value is None:
        raise ValueError(f"Could not parse polarization from {{qe_output_file}}")

    return pol_value, pol_quantum


def resolve_berry_phase_branch(pol_strained, pol_ref, quantum):
    """
    Resolve the Berry phase branch cut.

    The Berry phase polarization is defined modulo a quantum.
    Choose the branch closest to the reference polarization.
    """
    if quantum is None or quantum == 0:
        return pol_strained

    diff = pol_strained - pol_ref
    # Find closest branch
    n = round(diff / quantum)
    return pol_strained - n * quantum


# ── Parse reference polarization ───────────────────────────────────
print("Parsing reference (zero-strain) polarization...")
P_ref = np.zeros(3)
Q_ref = np.zeros(3)
for gdir in [1, 2, 3]:
    fname = OUTPUT_DIR / f"ref_berry_gdir{{gdir}}.out"
    if fname.exists():
        p, q = parse_berry_phase_polarization(fname)
        P_ref[gdir-1] = p
        Q_ref[gdir-1] = q if q else 0
        print(f"  gdir={{gdir}}: P = {{p:.8f}} C/m^2 (quantum = {{q}})")
    else:
        print(f"  WARNING: {{fname}} not found!")

# ── Parse strained polarizations ──────────────────────────────────
# e_ij: i = polarization direction (0,1,2 = x,y,z), j = Voigt strain (0..5)
e_clamped = np.zeros((3, 6))
e_relaxed = np.zeros((3, 6))

print("\\nParsing strained structure polarizations...")
for v_idx, vlabel in enumerate(VOIGT_LABELS):
    # ── Clamped-ion ──
    P_plus_clamped = np.zeros(3)
    P_minus_clamped = np.zeros(3)
    clamped_ok = True

    for gdir in [1, 2, 3]:
        fname_pos = STRAIN_DIR / f"strain_{{vlabel}}_pos_clamped_berry_gdir{{gdir}}.out"
        fname_neg = STRAIN_DIR / f"strain_{{vlabel}}_neg_clamped_berry_gdir{{gdir}}.out"

        if fname_pos.exists() and fname_neg.exists():
            p_pos, q_pos = parse_berry_phase_polarization(fname_pos)
            p_neg, q_neg = parse_berry_phase_polarization(fname_neg)

            # Resolve branch
            q = q_pos if q_pos else Q_ref[gdir-1]
            P_plus_clamped[gdir-1] = resolve_berry_phase_branch(p_pos, P_ref[gdir-1], q)
            P_minus_clamped[gdir-1] = resolve_berry_phase_branch(p_neg, P_ref[gdir-1], q)
        else:
            clamped_ok = False

    if clamped_ok:
        # Central difference: e_ij = (P_i(+eps) - P_i(-eps)) / (2 * eps)
        for i in range(3):
            e_clamped[i, v_idx] = (P_plus_clamped[i] - P_minus_clamped[i]) / (2.0 * STRAIN_MAGNITUDE)

    # ── Relaxed-ion ──
    P_plus_relaxed = np.zeros(3)
    P_minus_relaxed = np.zeros(3)
    relaxed_ok = True

    for gdir in [1, 2, 3]:
        fname_pos = STRAIN_DIR / f"strain_{{vlabel}}_pos_relaxed_berry_gdir{{gdir}}.out"
        fname_neg = STRAIN_DIR / f"strain_{{vlabel}}_neg_relaxed_berry_gdir{{gdir}}.out"

        if fname_pos.exists() and fname_neg.exists():
            p_pos, q_pos = parse_berry_phase_polarization(fname_pos)
            p_neg, q_neg = parse_berry_phase_polarization(fname_neg)

            q = q_pos if q_pos else Q_ref[gdir-1]
            P_plus_relaxed[gdir-1] = resolve_berry_phase_branch(p_pos, P_ref[gdir-1], q)
            P_minus_relaxed[gdir-1] = resolve_berry_phase_branch(p_neg, P_ref[gdir-1], q)
        else:
            relaxed_ok = False

    if relaxed_ok:
        for i in range(3):
            e_relaxed[i, v_idx] = (P_plus_relaxed[i] - P_minus_relaxed[i]) / (2.0 * STRAIN_MAGNITUDE)

    status_c = "OK" if clamped_ok else "MISSING"
    status_r = "OK" if relaxed_ok else "MISSING"
    print(f"  eps_{{vlabel}}: clamped={{status_c}}, relaxed={{status_r}}")

# Ionic contribution
e_ionic = e_relaxed - e_clamped

# ── Print results ──────────────────────────────────────────────────
def print_tensor(name, tensor):
    print(f"\\n=== {{name}} (C/m^2) ===")
    print(f"         eps_xx     eps_yy     eps_zz     eps_yz     eps_xz     eps_xy")
    for i, label in enumerate(["P_x", "P_y", "P_z"]):
        row = "  ".join(f"{{tensor[i,j]:10.5f}}" for j in range(6))
        print(f"  {{label}}   {{row}}")

print_tensor("Clamped-ion piezoelectric tensor (e_clamped)", e_clamped)
print_tensor("Relaxed-ion piezoelectric tensor (e_relaxed)", e_relaxed)
print_tensor("Ionic contribution (e_ionic = e_relaxed - e_clamped)", e_ionic)

# ── Save results ───────────────────────────────────────────────────
results = {{
    "e_clamped_Cm2": e_clamped.tolist(),
    "e_relaxed_Cm2": e_relaxed.tolist(),
    "e_ionic_Cm2": e_ionic.tolist(),
    "voigt_labels": VOIGT_LABELS,
    "polarization_dirs": ["P_x", "P_y", "P_z"],
    "strain_magnitude": STRAIN_MAGNITUDE,
    "unit": "C/m^2",
    "method": "QE Berry phase (finite differences)",
}}
with open(OUTPUT_DIR / "piezoelectric_tensor.json", "w") as f:
    json.dump(results, f, indent=2)
print(f"\\nResults saved to {{OUTPUT_DIR / 'piezoelectric_tensor.json'}}")

# ── Plot ───────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(15, 5))
titles = ["Clamped-ion", "Relaxed-ion", "Ionic contribution"]
tensors = [e_clamped, e_relaxed, e_ionic]

for ax, title, tensor in zip(axes, titles, tensors):
    im = ax.imshow(tensor, cmap="RdBu_r", aspect="auto",
                   vmin=-max(abs(tensor.min()), abs(tensor.max())),
                   vmax=max(abs(tensor.min()), abs(tensor.max())))
    ax.set_xticks(range(6))
    ax.set_xticklabels(VOIGT_LABELS, fontsize=10)
    ax.set_yticks(range(3))
    ax.set_yticklabels(["$P_x$", "$P_y$", "$P_z$"], fontsize=11)
    ax.set_title(title, fontsize=12)
    ax.set_xlabel("Strain component", fontsize=11)

    # Annotate cells with values
    for i in range(3):
        for j in range(6):
            val = tensor[i, j]
            color = "white" if abs(val) > 0.5 * max(abs(tensor.min()), abs(tensor.max())) else "black"
            ax.text(j, i, f"{{val:.3f}}", ha="center", va="center", fontsize=8, color=color)

    plt.colorbar(im, ax=ax, label="C/m$^2$", shrink=0.8)

fig.suptitle("Piezoelectric Stress Tensor $e_{{ij}}$", fontsize=14, y=1.02)
fig.tight_layout()
fig.savefig(OUTPUT_DIR / "piezoelectric_tensor.png", dpi=150, bbox_inches="tight")
print(f"Plot saved to {{OUTPUT_DIR / 'piezoelectric_tensor.png'}}")

# ── Print reference values for common materials ────────────────────
print("\\n=== Reference values (experimental, C/m^2) ===")
print("  ZnO (wurtzite):  e_33 = 0.89,  e_31 = -0.51,  e_15 = -0.45")
print("  AlN (wurtzite):  e_33 = 1.55,  e_31 = -0.58,  e_15 = -0.48")
print("  BaTiO3 (tetra):  e_33 = 3.36,  e_31 = -2.69,  e_15 = 21.3")
print("  GaN (wurtzite):  e_33 = 0.73,  e_31 = -0.49,  e_15 = -0.30")
print("  PbTiO3 (tetra):  e_33 = 3.23,  e_31 = -3.73,  e_15 = 5.01")
'''

with open(OUTPUT_DIR / "postprocess_piezo.py", "w") as f:
    f.write(postproc)
print(f"Written: {OUTPUT_DIR / 'postprocess_piezo.py'}")

# ── 8. Summary ────────────────────────────────────────────────────
print(f"""
=== Piezoelectric Tensor Workflow ===
Directory: {OUTPUT_DIR}

Step 1: Relax structure
  {QE_CMD} < relax.in > relax.out

Step 2: Run all strained calculations
  bash run_piezo.sh

Step 3: (If relaxed-ion) Generate Berry phase inputs from relaxed structures
  python generate_relaxed_berry.py

Step 4: Run relaxed Berry phase calculations (re-run relevant part of run_piezo.sh)

Step 5: Post-process
  python postprocess_piezo.py

Output: piezoelectric_tensor.json, piezoelectric_tensor.png
""")
```

---

## Key Parameters

### Strain magnitude
| Value | Use Case |
|-------|----------|
| 0.005 (0.5%) | Standard. Good balance of signal vs linearity. |
| 0.002 (0.2%) | More accurate for stiff materials. Requires tighter convergence. |
| 0.01 (1%) | Use if signal is too noisy at 0.5%. Risk of nonlinear effects. |

**Convergence check:** Run at both 0.005 and 0.002 and verify e_ij changes < 5%.

### SCF convergence
- Use `conv_thr = 1.0e-10` Ry or tighter for Berry phase calculations. Polarization is sensitive to wavefunction convergence.
- Use `nosym = .true.` and `noinv = .true.` to prevent symmetry from interfering with the Berry phase calculation under strain.

### k-points
- The Berry phase string direction (`nppstr`) needs enough k-points: at least 6, preferably 8-10.
- The total k-mesh should be reasonably dense: 8x8x8 for bulk materials.
- For materials with small band gaps, use denser meshes.

### Berry phase branch resolution
- The Berry phase polarization is defined modulo an "electronic quantum" of polarization: `e * R / Omega`, where R is a lattice vector and Omega is the cell volume.
- When comparing P(+strain) and P(-strain), ensure they are on the same branch. The post-processing script handles this automatically.

### Pseudopotentials
- Norm-conserving pseudopotentials are traditionally preferred for Berry phase calculations.
- PAW pseudopotentials work in modern QE (7.x) but may require the PAW Berry phase implementation.
- USPP: supported but less tested for Berry phase.

---

## Interpreting Results

### Units
- **e_ij:** Piezoelectric stress tensor in C/m^2. This is the "proper" piezoelectric tensor.
- **d_ij:** Piezoelectric strain tensor in pC/N (or pm/V). Related by: d = e . S, where S is the elastic compliance tensor.
- To convert: compute the elastic constants (see elastic tensor skill) and invert.

### Physical meaning
- `e_33`: Polarization along z due to strain along z. Dominant component in wurtzite (ZnO, AlN, GaN).
- `e_31`: Polarization along z due to strain along x. Often negative in wurtzites.
- `e_15`: Polarization along x due to shear strain yz. Related to transverse response.

### Symmetry constraints
- **Wurtzite (6mm):** Only e_31 = e_32, e_33, e_15 = e_24 are nonzero. 3 independent components.
- **Tetragonal (4mm, BaTiO3):** e_31 = e_32, e_33, e_15 = e_24. 3 independent components.
- **Cubic (m3m):** No piezoelectric response (centrosymmetric).
- **Zinc blende (43m):** Only e_14 = e_25 = e_36. 1 independent component.

### Clamped vs Relaxed
- **Clamped-ion:** Only electronic contribution. Usually smaller magnitude.
- **Relaxed-ion:** Includes internal strain effect (ions move to new equilibrium). Usually dominates.
- If `e_relaxed >> e_clamped`, the piezoelectric response is primarily ionic (common in perovskites).

### Sanity checks
1. Verify symmetry: for wurtzite, e_31 should equal e_32 (within numerical noise ~0.01 C/m^2).
2. Centrosymmetric materials should give e_ij = 0 (within noise).
3. Compare with published DFT values (typically within 10-20% of experiment due to exchange-correlation approximation).

---

## Common Issues

| Problem | Cause | Solution |
|---------|-------|----------|
| All e_ij = 0 | Centrosymmetric structure | Check space group. Piezoelectricity requires broken inversion symmetry. |
| Berry phase not converging | Metallic or near-metallic system | Berry phase requires an insulator. Check band gap. Use `occupations = 'fixed'`. |
| e_ij varies wildly with strain magnitude | Nonlinear regime or branch-cut error | Use smaller strain (0.002). Check branch resolution in post-processing. |
| Inconsistent branches | Berry phase quantum not properly resolved | Compare P(+eps) and P(-eps) with P(ref). Shift by integer multiples of quantum. |
| `nosym` not set | Symmetry breaks Berry phase under strain | Always use `nosym = .true.` and `noinv = .true.` for strained calculations. |
| Large ionic contribution oscillates | Internal relaxation not converged | Tighten `forc_conv_thr` to 1e-6 Ry/Bohr for the ionic relaxation step. |
| e_31 != e_32 for wurtzite | Numerical noise or cell not properly oriented | Verify cell orientation matches convention. Check with tighter k-mesh. |
| PAW Berry phase errors | Older QE version or incompatible pseudo | Use norm-conserving pseudopotentials, or update to QE 7.x. |
| Polarization in wrong units | Misinterpreting QE output | QE reports P in C/m^2. Verify by checking against known polarization of BaTiO3 (~0.26 C/m^2). |
