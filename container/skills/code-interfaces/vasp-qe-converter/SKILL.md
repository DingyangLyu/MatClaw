# VASP-QE Input/Output Converter

## When to Use

- You have VASP input files (POSCAR, INCAR, KPOINTS) and need to run the same system in Quantum ESPRESSO.
- You have QE input/output and need to convert it to VASP format for collaborators or VASP-dependent tools.
- You need to compare results between VASP and QE for the same system (benchmarking, validation).
- You are migrating a workflow from VASP to QE (or vice versa) and need parameter mapping.
- You need to parse VASP OUTCAR and extract data in a format compatible with QE post-processing tools.

## Method Selection

| Conversion | Method | Key Tools |
|---|---|---|
| POSCAR to QE input (ibrav=0) | `pymatgen.io.vasp.Poscar` + `pymatgen.io.pwscf.PWInput` | Structure-level conversion with automatic namelist generation |
| QE output to POSCAR | `pymatgen.io.pwscf.PWOutput` + `pymatgen.io.vasp.Poscar` | Parse relaxed structure from QE, write VASP format |
| INCAR to QE namelists | Manual mapping via Python dict | Parameter-by-parameter mapping (see table below) |
| KPOINTS to K_POINTS | `pymatgen.io.vasp.Kpoints` parser + formatting | Grid, line-mode, and explicit k-point conversion |
| OUTCAR energies/forces to QE format | `pymatgen.io.vasp.Outcar` parser | Extract and reformat for QE-compatible post-processing |
| Full VASP input set to QE input | Combined workflow (all above) | Single script for complete conversion |

## Prerequisites

- Python packages: `pymatgen`, `ase`, `numpy` (pre-installed).
- For QE execution: QE binaries in `/opt/qe/bin/` (pre-installed).
- For VASP POTCAR handling: `PMG_VASP_PSP_DIR` environment variable (only needed if generating POTCAR).
- SSSP pseudopotentials for QE (download script provided below).

---

## Detailed Steps

### 1. POSCAR to QE Input (Complete Conversion)

This converts a VASP POSCAR to a complete QE pw.x input file with ibrav=0, including automatic pseudopotential selection and k-point mapping.

```python
#!/usr/bin/env python3
"""
Convert VASP POSCAR to Quantum ESPRESSO pw.x input file.
Handles structure, pseudopotentials, k-points, and all namelists.
"""
import os
import json
import numpy as np
import urllib.request
from pymatgen.core.structure import Structure
from pymatgen.io.vasp import Poscar, Incar, Kpoints
from pymatgen.io.pwscf import PWInput

# ============================================================
# CONFIGURATION
# ============================================================

POSCAR_FILE = "POSCAR"              # Input VASP POSCAR
INCAR_FILE = "INCAR"                # Optional: read VASP INCAR for parameter mapping
KPOINTS_FILE = "KPOINTS"            # Optional: read VASP KPOINTS
PSEUDO_DIR = os.path.abspath("./pseudo")
OUTPUT_QE = "pw_from_vasp.in"

os.makedirs(PSEUDO_DIR, exist_ok=True)

# ============================================================
# 1. LOAD POSCAR
# ============================================================

poscar = Poscar.from_file(POSCAR_FILE)
structure = poscar.structure

print(f"Structure: {structure.composition.reduced_formula}")
print(f"  Atoms: {len(structure)}")
print(f"  Lattice: a={structure.lattice.a:.4f}, b={structure.lattice.b:.4f}, "
      f"c={structure.lattice.c:.4f} A")
print(f"  Volume: {structure.volume:.4f} A^3")

# ============================================================
# 2. MAP INCAR PARAMETERS TO QE NAMELISTS
# ============================================================

# Default QE parameters
control = {
    "calculation": "scf",
    "prefix": structure.composition.reduced_formula.replace(" ", ""),
    "outdir": "./tmp",
    "pseudo_dir": PSEUDO_DIR,
    "tprnfor": True,
    "tstress": True,
    "verbosity": "high",
}

system = {
    "ecutwfc": 50.0,
    "ecutrho": 400.0,
    "occupations": "smearing",
    "smearing": "cold",
    "degauss": 0.01,
}

electrons = {
    "conv_thr": 1.0e-8,
    "mixing_beta": 0.7,
}

ions = None
cell = None

# Read INCAR if available and map parameters
if os.path.exists(INCAR_FILE):
    incar = Incar.from_file(INCAR_FILE)
    print(f"\nMapping INCAR parameters to QE namelists...")

    # ENCUT (eV) -> ecutwfc (Ry): 1 eV = 0.0735 Ry
    if "ENCUT" in incar:
        ecutwfc_ry = incar["ENCUT"] / 13.6057
        system["ecutwfc"] = round(ecutwfc_ry, 1)
        system["ecutrho"] = round(ecutwfc_ry * 8, 1)  # 8x for PAW
        print(f"  ENCUT={incar['ENCUT']} eV -> ecutwfc={system['ecutwfc']:.1f} Ry")

    # EDIFF (eV) -> conv_thr (Ry): factor of 2 (Ry vs Ha convention)
    if "EDIFF" in incar:
        conv_thr = incar["EDIFF"] / 13.6057
        electrons["conv_thr"] = conv_thr
        print(f"  EDIFF={incar['EDIFF']} eV -> conv_thr={conv_thr:.1e}")

    # ISMEAR / SIGMA -> smearing / degauss
    if "ISMEAR" in incar:
        ismear = incar["ISMEAR"]
        sigma = incar.get("SIGMA", 0.05)
        degauss_ry = sigma / 13.6057
        if ismear == -5:
            system["occupations"] = "tetrahedra"
            del system["smearing"]
            del system["degauss"]
            print(f"  ISMEAR=-5 -> occupations='tetrahedra'")
        elif ismear == 0:
            system["smearing"] = "gauss"
            system["degauss"] = round(degauss_ry, 4)
            print(f"  ISMEAR=0, SIGMA={sigma} -> smearing='gauss', degauss={degauss_ry:.4f}")
        elif ismear == 1:
            system["smearing"] = "mp"
            system["degauss"] = round(degauss_ry, 4)
            print(f"  ISMEAR=1, SIGMA={sigma} -> smearing='mp', degauss={degauss_ry:.4f}")
        elif ismear == 2:
            system["smearing"] = "mv"
            system["degauss"] = round(degauss_ry, 4)
            print(f"  ISMEAR=2, SIGMA={sigma} -> smearing='mv', degauss={degauss_ry:.4f}")

    # NSW / IBRION / ISIF -> calculation type
    nsw = incar.get("NSW", 0)
    ibrion = incar.get("IBRION", -1)
    isif = incar.get("ISIF", 2)

    if nsw == 0 or ibrion == -1:
        control["calculation"] = "scf"
    elif isif <= 2:
        control["calculation"] = "relax"
        ions = {"ion_dynamics": "bfgs"}
    elif isif >= 3:
        control["calculation"] = "vc-relax"
        ions = {"ion_dynamics": "bfgs"}
        cell = {"cell_dynamics": "bfgs", "press": 0.0}
    print(f"  NSW={nsw}, IBRION={ibrion}, ISIF={isif} -> calculation='{control['calculation']}'")

    # EDIFFG -> forc_conv_thr
    if "EDIFFG" in incar and incar["EDIFFG"] < 0:
        # VASP: negative EDIFFG = force criterion in eV/A
        # QE: forc_conv_thr in Ry/Bohr (1 eV/A = 0.0389 Ry/Bohr)
        forc_conv = abs(incar["EDIFFG"]) * 0.0389
        control["forc_conv_thr"] = forc_conv
        print(f"  EDIFFG={incar['EDIFFG']} eV/A -> forc_conv_thr={forc_conv:.1e} Ry/Bohr")

    # ISPIN -> nspin
    if incar.get("ISPIN", 1) == 2:
        system["nspin"] = 2
        print(f"  ISPIN=2 -> nspin=2")

    # LSORBIT -> noncolin + lspinorb
    if incar.get("LSORBIT", False):
        system["noncolin"] = True
        system["lspinorb"] = True
        print(f"  LSORBIT=.TRUE. -> noncolin=.true., lspinorb=.true.")

    # GGA / METAGGA -> input_dft
    if "METAGGA" in incar:
        metagga = incar["METAGGA"]
        dft_map = {"SCAN": "scan", "R2SCAN": "r2scan", "TPSS": "tpss"}
        if metagga in dft_map:
            system["input_dft"] = dft_map[metagga]
            print(f"  METAGGA={metagga} -> input_dft='{dft_map[metagga]}'")
    elif "GGA" in incar:
        gga_map = {"PE": "pbe", "PS": "pbesol", "91": "pw91"}
        gga = incar["GGA"]
        if gga in gga_map:
            system["input_dft"] = gga_map[gga]
            print(f"  GGA={gga} -> input_dft='{gga_map[gga]}'")

    # IVDW -> vdw_corr
    if "IVDW" in incar:
        ivdw = incar["IVDW"]
        vdw_map = {11: "dft-d3", 12: "dft-d3", 1: "dft-d", 2: "dft-d"}
        if ivdw in vdw_map:
            system["vdw_corr"] = vdw_map[ivdw]
            print(f"  IVDW={ivdw} -> vdw_corr='{vdw_map[ivdw]}'")

    # LDAU -> lda_plus_u
    if incar.get("LDAU", False):
        system["lda_plus_u"] = True
        species_order = list(dict.fromkeys(str(sp) for sp in structure.species))
        if "LDAUU" in incar:
            ldauu = [float(x) for x in str(incar["LDAUU"]).split()]
            for i, (el, u) in enumerate(zip(species_order, ldauu), 1):
                if u > 0:
                    system[f"Hubbard_U({i})"] = u
                    print(f"  LDAUU: {el} -> Hubbard_U({i})={u}")

# ============================================================
# 3. MAP KPOINTS
# ============================================================

kgrid = (6, 6, 6)  # default

if os.path.exists(KPOINTS_FILE):
    kpoints = Kpoints.from_file(KPOINTS_FILE)
    if kpoints.style.name in ("Gamma", "Monkhorst"):
        kgrid = tuple(kpoints.kpts[0])
        kgrid = tuple(int(k) for k in kgrid)
        print(f"\nKPOINTS: {kgrid[0]}x{kgrid[1]}x{kgrid[2]} "
              f"({'Gamma' if kpoints.style.name == 'Gamma' else 'MP'})")
    else:
        print(f"\nKPOINTS style '{kpoints.style.name}' -- using default grid {kgrid}")
else:
    # Auto-determine from structure
    recip = structure.lattice.reciprocal_lattice.abc
    kgrid = tuple(max(1, int(round(40 / (2 * np.pi) * r))) for r in recip)
    print(f"\nAuto k-grid: {kgrid[0]}x{kgrid[1]}x{kgrid[2]}")

# ============================================================
# 4. GET PSEUDOPOTENTIALS (SSSP)
# ============================================================

SSSP_URL = "https://raw.githubusercontent.com/aiidateam/pseudo-data/main/sssp/sssp_efficiency_1.3.0_pbe.json"
sssp_json = os.path.join(PSEUDO_DIR, "sssp_efficiency_1.3.0_pbe.json")

if not os.path.exists(sssp_json):
    print("\nDownloading SSSP metadata...")
    urllib.request.urlretrieve(SSSP_URL, sssp_json)

with open(sssp_json) as f:
    sssp_data = json.load(f)

BASE_PP = "https://raw.githubusercontent.com/aiidateam/pseudo-data/main/sssp/pseudos/"
pseudo_map = {}
elements = [str(el) for el in structure.composition.elements]

for el in elements:
    if el in sssp_data:
        fname = sssp_data[el]["filename"]
        pseudo_map[el] = fname
        dest = os.path.join(PSEUDO_DIR, fname)
        if not os.path.exists(dest):
            print(f"  Downloading {fname}...")
            urllib.request.urlretrieve(BASE_PP + fname, dest)
        # Update cutoffs from SSSP recommendation
        sssp_ecutwfc = sssp_data[el].get("cutoff_wfc", 50)
        sssp_ecutrho = sssp_data[el].get("cutoff_rho", 400)
        system["ecutwfc"] = max(system["ecutwfc"], sssp_ecutwfc)
        system["ecutrho"] = max(system["ecutrho"], sssp_ecutrho)
    else:
        pseudo_map[el] = f"{el}.UPF"
        print(f"  WARNING: {el} not in SSSP database, using {el}.UPF")

print(f"  Final cutoffs: ecutwfc={system['ecutwfc']} Ry, ecutrho={system['ecutrho']} Ry")

# ============================================================
# 5. WRITE QE INPUT
# ============================================================

pw_in = PWInput(
    structure,
    pseudo=pseudo_map,
    control=control,
    system=system,
    electrons=electrons,
    ions=ions,
    cell=cell,
    kpoints_grid=kgrid,
)

pw_in.write_file(OUTPUT_QE)
print(f"\nQE input written to: {OUTPUT_QE}")

with open(OUTPUT_QE) as f:
    print("\n--- QE Input ---")
    print(f.read())
```

### 2. QE Output to VASP POSCAR

Convert a relaxed structure from QE back to VASP POSCAR format.

```python
#!/usr/bin/env python3
"""
Convert QE pw.x output (relaxed structure) to VASP POSCAR.
Handles vc-relax, relax, and scf output.
"""
import re
import numpy as np
from pymatgen.core.structure import Structure, Lattice
from pymatgen.io.vasp import Poscar

# ============================================================
# CONFIGURATION
# ============================================================

QE_OUTPUT = "pw.out"      # QE output file
QE_INPUT = "pw.in"        # QE input file (for initial structure if output parsing fails)
OUTPUT_POSCAR = "POSCAR_from_qe"

# ============================================================
# PARSE QE OUTPUT FOR FINAL STRUCTURE
# ============================================================

def parse_qe_output(filename):
    """Parse QE pw.x output for the final (relaxed) structure."""
    with open(filename) as f:
        content = f.read()

    # Check if calculation converged
    if "convergence has been achieved" not in content and "bfgs converged" not in content:
        print("WARNING: Calculation may not have converged.")

    lines = content.split("\n")

    # Try to find CELL_PARAMETERS from final relaxed structure
    cell_matrix = None
    coords = []
    species = []
    coord_type = "crystal"

    # Look for the last CELL_PARAMETERS block (for vc-relax)
    cell_indices = [i for i, line in enumerate(lines)
                    if "CELL_PARAMETERS" in line and "CELL_PARA" in line.upper()]
    if cell_indices:
        idx = cell_indices[-1]  # last occurrence
        unit_line = lines[idx]
        # Parse unit (angstrom or bohr)
        is_bohr = "bohr" in unit_line.lower()
        cell_matrix = []
        for j in range(1, 4):
            vals = [float(x) for x in lines[idx + j].split()]
            cell_matrix.append(vals)
        cell_matrix = np.array(cell_matrix)
        if is_bohr:
            cell_matrix *= 0.529177  # Bohr to Angstrom

    # Look for the last ATOMIC_POSITIONS block
    pos_indices = [i for i, line in enumerate(lines)
                   if "ATOMIC_POSITIONS" in line and "ATOMIC_POS" in line.upper()]
    if pos_indices:
        idx = pos_indices[-1]
        header = lines[idx]
        if "angstrom" in header.lower():
            coord_type = "cartesian"
        elif "crystal" in header.lower():
            coord_type = "crystal"
        elif "bohr" in header.lower():
            coord_type = "bohr"

        coords = []
        species = []
        j = idx + 1
        while j < len(lines) and lines[j].strip():
            parts = lines[j].split()
            if len(parts) >= 4:
                try:
                    species.append(parts[0])
                    coords.append([float(parts[1]), float(parts[2]), float(parts[3])])
                except ValueError:
                    break
            else:
                break
            j += 1
        coords = np.array(coords)

    # If no cell was found in output, parse from input
    if cell_matrix is None:
        print("  No CELL_PARAMETERS in output, reading from input file...")
        structure = parse_qe_input_structure(QE_INPUT)
        if structure is not None:
            cell_matrix = structure.lattice.matrix

    if cell_matrix is None or len(coords) == 0:
        print("ERROR: Could not parse structure from QE output.")
        return None

    # Build pymatgen Structure
    lattice = Lattice(cell_matrix)
    if coord_type == "crystal":
        structure = Structure(lattice, species, coords)
    elif coord_type == "cartesian" or coord_type == "angstrom":
        structure = Structure(lattice, species, coords, coords_are_cartesian=True)
    elif coord_type == "bohr":
        coords_ang = coords * 0.529177
        structure = Structure(lattice, species, coords_ang, coords_are_cartesian=True)

    return structure


def parse_qe_input_structure(filename):
    """Parse QE pw.x input for structure (fallback)."""
    try:
        from pymatgen.io.pwscf import PWInput
        pw_in = PWInput.from_file(filename)
        return pw_in.structure
    except Exception as e:
        print(f"  Could not parse QE input: {e}")
        return None


# ============================================================
# CONVERT AND WRITE
# ============================================================

structure = parse_qe_output(QE_OUTPUT)

if structure is not None:
    poscar = Poscar(structure,
                    comment=f"{structure.composition.reduced_formula} from QE")
    poscar.write_file(OUTPUT_POSCAR)
    print(f"POSCAR written to: {OUTPUT_POSCAR}")
    print(f"  Formula: {structure.composition.reduced_formula}")
    print(f"  Atoms: {len(structure)}")
    print(f"  Lattice: a={structure.lattice.a:.4f}, b={structure.lattice.b:.4f}, "
          f"c={structure.lattice.c:.4f} A")

    with open(OUTPUT_POSCAR) as f:
        print("\n--- POSCAR ---")
        print(f.read())
```

### 3. INCAR to QE Parameter Mapping Reference

Complete mapping between VASP INCAR tags and QE namelist parameters.

```python
#!/usr/bin/env python3
"""
VASP INCAR -> QE namelist parameter mapping.
Prints a reference table and converts a specific INCAR to QE parameters.
"""
from pymatgen.io.vasp import Incar

# Complete INCAR -> QE mapping reference
PARAMETER_MAP = {
    # === ELECTRONIC ===
    "ENCUT": {
        "qe_param": "ecutwfc (Ry) = ENCUT / 13.6057",
        "qe_namelist": "&SYSTEM",
        "notes": "Also set ecutrho = 8*ecutwfc for PAW, 4*ecutwfc for NC"
    },
    "EDIFF": {
        "qe_param": "conv_thr (Ry) = EDIFF / 13.6057",
        "qe_namelist": "&ELECTRONS",
        "notes": "QE convergence is on total energy in Ry"
    },
    "NELM": {
        "qe_param": "electron_maxstep",
        "qe_namelist": "&ELECTRONS",
        "notes": "Same meaning: max SCF iterations"
    },
    "ALGO": {
        "qe_param": "diagonalization",
        "qe_namelist": "&ELECTRONS",
        "notes": "ALGO=Normal -> 'david', ALGO=Fast -> 'david', ALGO=All -> 'cg'"
    },
    "ISMEAR": {
        "qe_param": "occupations / smearing",
        "qe_namelist": "&SYSTEM",
        "notes": "-5->tetrahedra, 0->gauss, 1->mp, 2->mv (cold)"
    },
    "SIGMA": {
        "qe_param": "degauss (Ry) = SIGMA / 13.6057",
        "qe_namelist": "&SYSTEM",
        "notes": "VASP in eV, QE in Ry"
    },
    "NBANDS": {
        "qe_param": "nbnd",
        "qe_namelist": "&SYSTEM",
        "notes": "Same meaning: number of bands"
    },
    # === IONIC ===
    "NSW": {
        "qe_param": "nstep (+ calculation type)",
        "qe_namelist": "&CONTROL / &IONS",
        "notes": "NSW>0 with ISIF<=2 -> calculation='relax', ISIF>=3 -> 'vc-relax'"
    },
    "IBRION": {
        "qe_param": "ion_dynamics",
        "qe_namelist": "&IONS",
        "notes": "IBRION=1,2 -> 'bfgs', IBRION=0 -> 'verlet' (MD)"
    },
    "ISIF": {
        "qe_param": "calculation",
        "qe_namelist": "&CONTROL",
        "notes": "ISIF<=2 -> 'relax', ISIF=3 -> 'vc-relax', ISIF=0 NSW>0 -> MD"
    },
    "EDIFFG": {
        "qe_param": "forc_conv_thr (Ry/Bohr)",
        "qe_namelist": "&CONTROL",
        "notes": "Negative EDIFFG (eV/A) * 0.0389 = forc_conv_thr (Ry/Bohr)"
    },
    "POTIM": {
        "qe_param": "dt (a.u.) for MD, trust_radius_ini for relax",
        "qe_namelist": "&IONS",
        "notes": "POTIM(fs) * 41.341 = dt (a.u.) for MD"
    },
    # === SPIN / MAGNETISM ===
    "ISPIN": {
        "qe_param": "nspin",
        "qe_namelist": "&SYSTEM",
        "notes": "ISPIN=1 -> nspin=1, ISPIN=2 -> nspin=2"
    },
    "MAGMOM": {
        "qe_param": "starting_magnetization(i)",
        "qe_namelist": "&SYSTEM",
        "notes": "VASP: per-atom absolute moment. QE: per-type fractional (-1 to 1)"
    },
    "LSORBIT": {
        "qe_param": "lspinorb + noncolin",
        "qe_namelist": "&SYSTEM",
        "notes": "LSORBIT=T -> noncolin=.true., lspinorb=.true."
    },
    # === DFT+U ===
    "LDAU": {
        "qe_param": "lda_plus_u",
        "qe_namelist": "&SYSTEM",
        "notes": "Enable Hubbard correction"
    },
    "LDAUU": {
        "qe_param": "Hubbard_U(i)",
        "qe_namelist": "&SYSTEM",
        "notes": "Same U values (eV), indexed by species type number"
    },
    # === FUNCTIONAL ===
    "GGA": {
        "qe_param": "input_dft",
        "qe_namelist": "&SYSTEM",
        "notes": "PE->pbe, PS->pbesol, 91->pw91. QE: input_dft overrides PP functional"
    },
    "METAGGA": {
        "qe_param": "input_dft",
        "qe_namelist": "&SYSTEM",
        "notes": "SCAN->scan, R2SCAN->r2scan, TPSS->tpss"
    },
    "LHFCALC": {
        "qe_param": "input_dft = 'hse'",
        "qe_namelist": "&SYSTEM",
        "notes": "HSE06: input_dft='hse', exx_fraction=0.25, screening_parameter=0.106"
    },
    # === VAN DER WAALS ===
    "IVDW": {
        "qe_param": "vdw_corr",
        "qe_namelist": "&SYSTEM",
        "notes": "11->dft-d3, 12->dft-d3 (BJ damping), 1,2->dft-d"
    },
    # === OUTPUT ===
    "LWAVE": {
        "qe_param": "disk_io",
        "qe_namelist": "&CONTROL",
        "notes": "QE always writes wavefunctions; disk_io='none' to minimize"
    },
    "LCHARG": {
        "qe_param": "disk_io",
        "qe_namelist": "&CONTROL",
        "notes": "QE: charge density saved by default"
    },
    "LORBIT": {
        "qe_param": "projwfc.x post-processing",
        "qe_namelist": "N/A",
        "notes": "VASP: LORBIT=11 for PDOS. QE: run projwfc.x separately"
    },
}

# Print mapping table
print("=" * 90)
print(f"{'VASP INCAR':>12} | {'QE Parameter':<35} | {'Namelist':<15} | Notes")
print("-" * 90)
for vasp_tag, info in PARAMETER_MAP.items():
    print(f"{vasp_tag:>12} | {info['qe_param']:<35} | {info['qe_namelist']:<15} | {info['notes']}")
print("=" * 90)

# Convert a specific INCAR
if __name__ == "__main__":
    import os
    if os.path.exists("INCAR"):
        incar = Incar.from_file("INCAR")
        print(f"\nConverting INCAR ({len(incar)} parameters):")
        for tag in incar:
            if tag in PARAMETER_MAP:
                print(f"  {tag}={incar[tag]} -> {PARAMETER_MAP[tag]['qe_param']}")
            else:
                print(f"  {tag}={incar[tag]} -> [no direct mapping]")
```

### 4. KPOINTS to K_POINTS Conversion

```python
#!/usr/bin/env python3
"""
Convert VASP KPOINTS file to QE K_POINTS card.
Supports automatic, Gamma, Monkhorst-Pack, line-mode, and explicit k-points.
"""
import numpy as np
from pymatgen.io.vasp import Kpoints
from pymatgen.core.structure import Structure
from pymatgen.symmetry.bandstructure import HighSymmKpath

KPOINTS_FILE = "KPOINTS"
STRUCTURE_FILE = "POSCAR"

kpoints = Kpoints.from_file(KPOINTS_FILE)
structure = Structure.from_file(STRUCTURE_FILE)

print(f"KPOINTS style: {kpoints.style.name}")
print(f"KPOINTS comment: {kpoints.comment}")

qe_kpoints = ""

if kpoints.style.name in ("Gamma", "Monkhorst"):
    nk = kpoints.kpts[0]
    shift = kpoints.kpts_shift if kpoints.kpts_shift else [0, 0, 0]

    # QE uses 0/1 shift (0 = Gamma-centered, 1 = half-grid shift)
    qe_shift = [int(round(s * 2)) for s in shift]

    qe_kpoints = f"K_POINTS automatic\n"
    qe_kpoints += f"  {int(nk[0])} {int(nk[1])} {int(nk[2])}  "
    qe_kpoints += f"{qe_shift[0]} {qe_shift[1]} {qe_shift[2]}\n"

    print(f"\nConverted to QE:")
    print(qe_kpoints)

elif kpoints.style.name == "Line_mode":
    # Convert line-mode KPOINTS to QE K_POINTS crystal_b
    labels = kpoints.labels
    kpts = kpoints.kpts
    num_kpts = kpoints.num_kpts  # points per segment

    qe_lines = []
    # QE crystal_b format: each point gets a weight (number of points to next)
    n_segments = len(kpts) // 2

    for i in range(n_segments):
        k1 = kpts[2 * i]
        k2 = kpts[2 * i + 1]
        l1 = labels[2 * i] if labels else ""
        l2 = labels[2 * i + 1] if labels else ""

        if i == 0:
            qe_lines.append(f"  {k1[0]:10.6f} {k1[1]:10.6f} {k1[2]:10.6f}  "
                           f"{num_kpts}  ! {l1}")
        qe_lines.append(f"  {k2[0]:10.6f} {k2[1]:10.6f} {k2[2]:10.6f}  "
                        f"{'0' if i == n_segments - 1 else str(num_kpts)}  ! {l2}")

    qe_kpoints = f"K_POINTS crystal_b\n"
    qe_kpoints += f"  {len(qe_lines)}\n"
    qe_kpoints += "\n".join(qe_lines) + "\n"

    print(f"\nConverted to QE (crystal_b):")
    print(qe_kpoints)

elif kpoints.style.name == "Reciprocal":
    # Explicit k-points with weights
    kpts = kpoints.kpts
    weights = kpoints.kpts_weights

    qe_kpoints = f"K_POINTS crystal\n"
    qe_kpoints += f"  {len(kpts)}\n"
    for k, w in zip(kpts, weights):
        qe_kpoints += f"  {k[0]:12.8f} {k[1]:12.8f} {k[2]:12.8f}  {w:12.8f}\n"

    print(f"\nConverted to QE (crystal):")
    print(qe_kpoints[:500])
    if len(kpts) > 10:
        print(f"  ... ({len(kpts)} k-points total)")

# Save to file
with open("K_POINTS_qe.txt", "w") as f:
    f.write(qe_kpoints)
print(f"\nSaved to: K_POINTS_qe.txt")
```

### 5. Parse VASP OUTCAR and Extract Results

```python
#!/usr/bin/env python3
"""
Parse VASP OUTCAR and extract key results.
Convert to QE-compatible format where applicable.
"""
import numpy as np
from pymatgen.io.vasp import Outcar, Vasprun
from pymatgen.core.structure import Structure

OUTCAR_FILE = "OUTCAR"
VASPRUN_FILE = "vasprun.xml"

# ============================================================
# METHOD A: Parse from OUTCAR
# ============================================================

if __name__ == "__main__":
    import os

    if os.path.exists(OUTCAR_FILE):
        outcar = Outcar(OUTCAR_FILE)

        print("=== VASP OUTCAR Results ===")
        print(f"  Final energy: {outcar.final_energy:.6f} eV")
        print(f"  Final energy (per atom): -- (need structure for this)")

        # Forces
        if outcar.ionic_steps:
            last_step = outcar.ionic_steps[-1]
            if "forces" in last_step:
                forces = np.array(last_step["forces"])
                max_force = np.sqrt((forces**2).sum(axis=1)).max()
                print(f"  Max force: {max_force:.6f} eV/A")

                # Convert forces to QE units (Ry/Bohr)
                # 1 eV/A = 0.03889 Ry/Bohr
                forces_qe = forces * 0.03889
                print(f"  Max force (QE): {max_force * 0.03889:.6f} Ry/Bohr")

        # Stress tensor
        if outcar.ionic_steps and "stress" in outcar.ionic_steps[-1]:
            stress = np.array(outcar.ionic_steps[-1]["stress"])
            print(f"  Stress tensor (kBar):")
            for row in stress:
                print(f"    {row[0]:10.4f} {row[1]:10.4f} {row[2]:10.4f}")

            # Convert to QE units (kbar -> kbar, same unit)
            # VASP stress in kBar, QE reports in kbar too
            print(f"  Pressure: {np.trace(stress) / 3:.4f} kBar")

        # Band gap
        if os.path.exists(VASPRUN_FILE):
            try:
                vasprun = Vasprun(VASPRUN_FILE, parse_dos=False, parse_eigen=True)
                bs = vasprun.get_band_structure()
                if bs.is_metal():
                    print("  Band gap: 0 (metallic)")
                else:
                    bg = bs.get_band_gap()
                    print(f"  Band gap: {bg['energy']:.4f} eV ({bg['transition']})")
            except Exception as e:
                print(f"  Could not parse band structure: {e}")

        # Magnetic moments
        if outcar.magnetization:
            print(f"  Total magnetization: {outcar.total_mag:.4f} mu_B")
            print(f"  Site magnetizations:")
            for i, mag in enumerate(outcar.magnetization):
                print(f"    Site {i}: {mag['tot']:.4f} mu_B")

        # Print unit conversion summary
        print("\n=== Unit Conversions (VASP -> QE) ===")
        print(f"  Energy:  1 eV     = 0.07350 Ry = 0.03675 Ha")
        print(f"  Force:   1 eV/A   = 0.03889 Ry/Bohr")
        print(f"  Length:  1 A      = 1.8897 Bohr")
        print(f"  Stress:  1 kBar   = 1 kbar (same)")
        print(f"  Cutoff:  1 eV     = 0.07350 Ry")
        print(f"  Smearing: 1 eV    = 0.07350 Ry")

    else:
        print(f"OUTCAR not found at {OUTCAR_FILE}")
        print("Provide a valid OUTCAR file path.")
```

### 6. Batch Conversion: Full VASP Directory to QE

```python
#!/usr/bin/env python3
"""
Convert a complete VASP calculation directory (POSCAR, INCAR, KPOINTS)
to a ready-to-run QE pw.x input file in one step.
"""
import os
import sys
import json
import numpy as np
import urllib.request
from pymatgen.core.structure import Structure
from pymatgen.io.vasp import Poscar, Incar, Kpoints
from pymatgen.io.pwscf import PWInput


def vasp_to_qe(vasp_dir, output_file="pw.in", pseudo_dir="./pseudo"):
    """
    Convert VASP input directory to QE pw.x input.

    Parameters
    ----------
    vasp_dir : str
        Directory containing POSCAR, INCAR, KPOINTS.
    output_file : str
        Output QE input file path.
    pseudo_dir : str
        Directory for pseudopotentials.
    """
    os.makedirs(pseudo_dir, exist_ok=True)

    # Load POSCAR
    poscar_path = os.path.join(vasp_dir, "POSCAR")
    if not os.path.exists(poscar_path):
        # Try CONTCAR
        poscar_path = os.path.join(vasp_dir, "CONTCAR")
    structure = Structure.from_file(poscar_path)

    # Load INCAR
    incar_path = os.path.join(vasp_dir, "INCAR")
    incar = Incar.from_file(incar_path) if os.path.exists(incar_path) else Incar()

    # Load KPOINTS
    kpoints_path = os.path.join(vasp_dir, "KPOINTS")
    kpoints = Kpoints.from_file(kpoints_path) if os.path.exists(kpoints_path) else None

    # --- Map calculation type ---
    nsw = incar.get("NSW", 0)
    isif = incar.get("ISIF", 2)
    ibrion = incar.get("IBRION", -1)

    if nsw == 0 or ibrion == -1:
        calc_type = "scf"
    elif isif >= 3:
        calc_type = "vc-relax"
    elif ibrion == 0:
        calc_type = "md"  # not standard QE calculation keyword
    else:
        calc_type = "relax"

    # --- Map cutoffs ---
    encut_ev = incar.get("ENCUT", 520)
    ecutwfc = round(encut_ev / 13.6057, 1)
    ecutrho = round(ecutwfc * 8, 1)

    # --- Map smearing ---
    ismear = incar.get("ISMEAR", 0)
    sigma_ev = incar.get("SIGMA", 0.05)
    degauss = round(sigma_ev / 13.6057, 4)
    smearing_map = {-5: None, 0: "gauss", 1: "mp", 2: "mv"}
    smearing = smearing_map.get(ismear, "cold")

    # --- Build QE input dicts ---
    control = {
        "calculation": calc_type if calc_type != "md" else "md",
        "prefix": structure.composition.reduced_formula.replace(" ", ""),
        "outdir": "./tmp",
        "pseudo_dir": os.path.abspath(pseudo_dir),
        "tprnfor": True,
        "tstress": True,
    }

    system_dict = {
        "ecutwfc": ecutwfc,
        "ecutrho": ecutrho,
    }

    if ismear == -5:
        system_dict["occupations"] = "tetrahedra"
    else:
        system_dict["occupations"] = "smearing"
        system_dict["smearing"] = smearing if smearing else "cold"
        system_dict["degauss"] = degauss

    if incar.get("ISPIN", 1) == 2:
        system_dict["nspin"] = 2

    electrons = {"conv_thr": 1.0e-8, "mixing_beta": 0.7}

    ions = {"ion_dynamics": "bfgs"} if calc_type in ("relax", "vc-relax") else None
    cell_dict = {"cell_dynamics": "bfgs"} if calc_type == "vc-relax" else None

    # --- K-grid ---
    kgrid = (6, 6, 6)
    if kpoints and kpoints.style.name in ("Gamma", "Monkhorst"):
        kgrid = tuple(int(k) for k in kpoints.kpts[0])

    # --- Pseudopotentials (SSSP) ---
    sssp_json = os.path.join(pseudo_dir, "sssp_efficiency_1.3.0_pbe.json")
    sssp_data = {}
    if os.path.exists(sssp_json):
        with open(sssp_json) as f:
            sssp_data = json.load(f)

    elements = [str(el) for el in structure.composition.elements]
    pseudo_map = {}
    for el in elements:
        if el in sssp_data:
            pseudo_map[el] = sssp_data[el]["filename"]
        else:
            pseudo_map[el] = f"{el}.UPF"

    # --- Write QE input ---
    pw_in = PWInput(
        structure,
        pseudo=pseudo_map,
        control=control,
        system=system_dict,
        electrons=electrons,
        ions=ions,
        cell=cell_dict,
        kpoints_grid=kgrid,
    )

    pw_in.write_file(output_file)
    print(f"Converted: {vasp_dir} -> {output_file}")
    print(f"  Calculation: {calc_type}")
    print(f"  ecutwfc: {ecutwfc} Ry (from ENCUT={encut_ev} eV)")
    print(f"  K-grid: {kgrid}")

    return output_file


# Usage
if __name__ == "__main__":
    vasp_dir = sys.argv[1] if len(sys.argv) > 1 else "."
    vasp_to_qe(vasp_dir)
```

## Key Parameters

| VASP Parameter | QE Equivalent | Conversion Factor | Notes |
|---|---|---|---|
| `ENCUT` (eV) | `ecutwfc` (Ry) | / 13.6057 | Also set `ecutrho` = 8x ecutwfc for PAW |
| `EDIFF` (eV) | `conv_thr` (Ry) | / 13.6057 | Total energy convergence |
| `SIGMA` (eV) | `degauss` (Ry) | / 13.6057 | Smearing width |
| `EDIFFG` (eV/A) | `forc_conv_thr` (Ry/Bohr) | * 0.0389 | Force convergence (use negative EDIFFG) |
| `POTIM` (fs) | `dt` (a.u.) | * 41.341 | MD timestep |
| `ISIF=3` | `calculation='vc-relax'` | -- | Variable-cell relaxation |
| `ISMEAR=0` | `smearing='gauss'` | -- | Gaussian smearing |
| `ISMEAR=1` | `smearing='mp'` | -- | Methfessel-Paxton |
| `ISMEAR=-5` | `occupations='tetrahedra'` | -- | Tetrahedron method |
| `LORBIT=11` | `projwfc.x` | -- | Projected DOS: separate post-processing in QE |
| `ICHARG=11` | `calculation='nscf'` | -- | Non-self-consistent from charge density |

## Interpreting Results

### Energy comparison
- VASP reports total energy in eV. QE reports in Ry (1 Ry = 13.6057 eV).
- Absolute energies differ between codes due to different pseudopotentials and implementations.
- Compare **relative** energies: formation energies, energy differences between structures, equation of state.
- Agreement within 1-5 meV/atom between VASP PAW and QE PAW/US for the same functional is typical.

### Force comparison
- VASP: eV/A. QE: Ry/Bohr. Conversion: 1 eV/A = 0.03889 Ry/Bohr.
- Forces should agree to within ~1 meV/A after proper convergence.

### Stress comparison
- Both codes report stress in kBar (VASP) / kbar (QE). Values should agree closely.
- QE also prints stress in Ry/Bohr^3 internally.

### Band gap
- Compare with same functional and similar pseudopotentials.
- PAW vs ultrasoft can give small (~0.05 eV) band gap differences.

## Common Issues

| Issue | Cause | Fix |
|---|---|---|
| Large energy difference between VASP and QE | Different pseudopotentials | Use equivalent PAW sets (SSSP for QE, PAW_PBE for VASP). Compare relative energies only. |
| QE input has wrong number of atoms | POSCAR parsing issue (selective dynamics, Cartesian vs Direct) | Use pymatgen `Poscar.from_file()` which handles all POSCAR variants. |
| KPOINTS conversion fails for line-mode | Line-mode format differs between codes | Use pymatgen `HighSymmKpath` to generate fresh k-path for either code. |
| QE crashes with "pseudopotential not found" | Filename mismatch | Check that `pseudo_dir` contains the exact filenames referenced in `ATOMIC_SPECIES`. |
| VASP DFT+U results differ from QE Hubbard | Different DFT+U implementations (Dudarev vs Liechtenstein) | Ensure both use Dudarev (simplified). VASP: LDAUTYPE=2. QE: default lda_plus_u. |
| Cell shape changes differently in vc-relax | Different optimization algorithms | Use same symmetry constraints. QE: `cell_dofree` to restrict cell degrees of freedom. |
| Magnetic moments differ | Different initial magnetization / convergence | QE `starting_magnetization` is fractional (-1 to 1). VASP MAGMOM is absolute. Adjust accordingly. |
