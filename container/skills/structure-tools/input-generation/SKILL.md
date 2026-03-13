# DFT Input File Generation

## When to Use

- You have a crystal structure (CIF, POSCAR, Materials Project ID) and need to set up a DFT calculation.
- You need to generate complete QE pw.x input files with automatic pseudopotential selection and k-point grids.
- You need to generate VASP input files (INCAR, KPOINTS, POSCAR, POTCAR) using pymatgen.
- You want to set up a series of calculations (convergence tests, different functionals, magnetic configurations).
- You need to convert a structure from any format into a ready-to-run DFT input (VASPKIT 101-109 equivalent).

## Method Selection

| Criterion | ASE + pymatgen (Python) | QE (pw.x) | VASP |
|---|---|---|---|
| Input generation | Full control via Python API | Generate .in files with pymatgen or manually | Generate INCAR/KPOINTS/POSCAR/POTCAR via pymatgen |
| Pseudopotentials | SSSP library (auto-download) | UPF files in pseudo_dir | POTCAR from VASP distribution |
| K-point grid | pymatgen automatic mesh or manual | Specified in input file | KPOINTS file (auto or explicit) |
| Available now | Yes | Yes | Input generation only (no VASP binary in container) |
| Best for | Rapid prototyping, batch generation | Production DFT runs | When VASP is available externally |

## Prerequisites

- A crystal structure in any common format (CIF, POSCAR, XYZ) or a Materials Project ID.
- For QE: pseudopotential files (SSSP download script provided below).
- For VASP: a configured `PMG_VASP_PSP_DIR` environment variable pointing to POTCAR files (for POTCAR generation only).
- Python packages: `pymatgen`, `ase`, `numpy`, `spglib` (pre-installed).

---

## Detailed Steps

### Method A: ASE + pymatgen -- Generate QE Input Files

This method uses pymatgen's `PWInput` class and ASE to build complete Quantum ESPRESSO input files from any structure source, with automatic pseudopotential selection and k-point grid generation.

#### Step A1: Load structure from various sources

```python
#!/usr/bin/env python3
"""
Load a crystal structure from CIF, POSCAR, Materials Project, or build it
programmatically. This is the starting point for all input generation.
"""
from pymatgen.core.structure import Structure, Lattice
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
from ase.io import read as ase_read
from ase.build import bulk

# ── Option 1: Load from CIF file ─────────────────────────────────
# structure = Structure.from_file("my_structure.cif")

# ── Option 2: Load from POSCAR ───────────────────────────────────
# structure = Structure.from_file("POSCAR")

# ── Option 3: Load from Materials Project ────────────────────────
# from mp_api.client import MPRester
# with MPRester("YOUR_API_KEY") as mpr:
#     structure = mpr.get_structure_by_material_id("mp-149")  # Silicon

# ── Option 4: Build programmatically with pymatgen ───────────────
structure = Structure(
    lattice=Lattice.cubic(5.43),
    species=["Si", "Si", "Si", "Si", "Si", "Si", "Si", "Si"],
    coords=[
        [0.00, 0.00, 0.00], [0.50, 0.50, 0.00],
        [0.50, 0.00, 0.50], [0.00, 0.50, 0.50],
        [0.25, 0.25, 0.25], [0.75, 0.75, 0.25],
        [0.75, 0.25, 0.75], [0.25, 0.75, 0.75],
    ],
)

# ── Option 5: Build with ASE, then convert to pymatgen ───────────
# atoms = bulk("Si", "diamond", a=5.43)
# from pymatgen.io.ase import AseAtomsAdaptor
# structure = AseAtomsAdaptor.get_structure(atoms)

# ── Inspect the structure ─────────────────────────────────────────
print(f"Formula: {structure.composition.reduced_formula}")
print(f"Number of atoms: {len(structure)}")
print(f"Lattice parameters: a={structure.lattice.a:.4f}, b={structure.lattice.b:.4f}, "
      f"c={structure.lattice.c:.4f}")
print(f"Angles: alpha={structure.lattice.alpha:.2f}, beta={structure.lattice.beta:.2f}, "
      f"gamma={structure.lattice.gamma:.2f}")
print(f"Volume: {structure.volume:.4f} A^3")

sga = SpacegroupAnalyzer(structure, symprec=0.01)
print(f"Space group: {sga.get_space_group_symbol()} (#{sga.get_space_group_number()})")
print(f"Crystal system: {sga.get_crystal_system()}")
```

#### Step A2: Download SSSP pseudopotentials

```python
#!/usr/bin/env python3
"""
Download SSSP Efficiency pseudopotentials for a given set of elements.
Run once per element set. Pseudopotentials are saved to ./pseudo/
"""
import os
import json
import urllib.request

PSEUDO_DIR = os.path.abspath("./pseudo")
os.makedirs(PSEUDO_DIR, exist_ok=True)

# SSSP Efficiency 1.3.0 metadata (PBE)
SSSP_URL = "https://raw.githubusercontent.com/aiidateam/pseudo-data/main/sssp/sssp_efficiency_1.3.0_pbe.json"

print("Downloading SSSP metadata...")
try:
    with urllib.request.urlopen(SSSP_URL, timeout=30) as resp:
        sssp_data = json.loads(resp.read().decode())
except Exception as e:
    print(f"Could not download SSSP metadata: {e}")
    sssp_data = {}

BASE = "https://raw.githubusercontent.com/aiidateam/pseudo-data/main/sssp/pseudos/"

def get_sssp_info(element):
    """Get the SSSP recommended pseudopotential filename, cutoffs, and dual."""
    if element not in sssp_data:
        return None
    entry = sssp_data[element]
    return {
        "filename": entry["filename"],
        "ecutwfc": entry.get("cutoff_wfc", 50),
        "ecutrho": entry.get("cutoff_rho", 400),
        "dual": entry.get("cutoff_rho", 400) / entry.get("cutoff_wfc", 50),
    }

def download_pseudo(element):
    """Download pseudopotential for a single element. Returns filename."""
    info = get_sssp_info(element)
    if info is None:
        print(f"  WARNING: {element} not found in SSSP database")
        return None
    fname = info["filename"]
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
        return None
    return fname

# ── Download for your elements ────────────────────────────────────
elements_needed = ["Si"]  # <-- Change this to your elements
pseudo_map = {}
cutoff_info = {}
for el in elements_needed:
    fname = download_pseudo(el)
    if fname:
        pseudo_map[el] = fname
        cutoff_info[el] = get_sssp_info(el)

print(f"\nPseudopotentials in {PSEUDO_DIR}:")
for f in sorted(os.listdir(PSEUDO_DIR)):
    print(f"  {f}")

# Report recommended cutoffs
if cutoff_info:
    max_ecutwfc = max(info["ecutwfc"] for info in cutoff_info.values())
    max_ecutrho = max(info["ecutrho"] for info in cutoff_info.values())
    print(f"\nRecommended cutoffs (max across all elements):")
    print(f"  ecutwfc = {max_ecutwfc} Ry")
    print(f"  ecutrho = {max_ecutrho} Ry")
    for el, info in cutoff_info.items():
        print(f"  {el}: ecutwfc={info['ecutwfc']}, ecutrho={info['ecutrho']}, dual={info['dual']:.1f}")
```

#### Step A3: Generate complete QE input file with pymatgen

```python
#!/usr/bin/env python3
"""
Generate a complete QE pw.x input file using pymatgen PWInput.
Handles structure loading, pseudopotential mapping, k-point grid,
and all control parameters automatically.
"""
import os
import json
import numpy as np
from pymatgen.core.structure import Structure
from pymatgen.io.pwscf import PWInput
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
from pymatgen.symmetry.bandstructure import HighSymmKpath

# ── CONFIGURATION ────────────────────────────────────────────────
INPUT_STRUCTURE = "structure.cif"       # Input structure file
PSEUDO_DIR = os.path.abspath("./pseudo")
CALCULATION = "scf"                     # "scf", "relax", "vc-relax", "nscf", "bands"
ECUTWFC = 50.0                          # Plane-wave cutoff (Ry) -- use SSSP recommended
ECUTRHO = 400.0                         # Charge density cutoff (Ry)
K_GRID = None                           # Auto-determine if None, or specify e.g. (6,6,6)
K_DENSITY = 40                          # k-points per reciprocal atom (A^-1), used if K_GRID is None
SMEARING = "cold"                       # "cold" (mv), "mp", "gauss", "fd"
DEGAUSS = 0.01                          # Smearing width (Ry)
MAGNETIC = False                        # Enable spin-polarized calculation
OUTPUT_FILE = "pw_input.in"
# ─────────────────────────────────────────────────────────────────

# ── Load structure ────────────────────────────────────────────────
structure = Structure.from_file(INPUT_STRUCTURE)
print(f"Structure: {structure.composition.reduced_formula}, {len(structure)} atoms")

sga = SpacegroupAnalyzer(structure, symprec=0.01)
print(f"Space group: {sga.get_space_group_symbol()} (#{sga.get_space_group_number()})")

# ── Determine pseudopotential mapping ────────────────────────────
# Try to load SSSP metadata for automatic mapping
SSSP_JSON = os.path.join(PSEUDO_DIR, "sssp_efficiency_1.3.0_pbe.json")
sssp_data = {}
if os.path.exists(SSSP_JSON):
    with open(SSSP_JSON) as f:
        sssp_data = json.load(f)

pseudo_map = {}
for el in structure.composition.elements:
    symbol = el.symbol
    if symbol in sssp_data:
        pseudo_map[symbol] = sssp_data[symbol]["filename"]
    else:
        # Fallback: list UPF files and try to match
        if os.path.exists(PSEUDO_DIR):
            for fname in os.listdir(PSEUDO_DIR):
                if fname.endswith(".UPF") and fname.startswith(symbol):
                    pseudo_map[symbol] = fname
                    break
        if symbol not in pseudo_map:
            pseudo_map[symbol] = f"{symbol}.UPF"
            print(f"  WARNING: No pseudopotential found for {symbol}, using {pseudo_map[symbol]}")

print(f"Pseudopotentials: {pseudo_map}")

# ── Determine k-point grid ───────────────────────────────────────
if K_GRID is None:
    # Automatic k-grid based on reciprocal lattice lengths
    recip_lengths = structure.lattice.reciprocal_lattice.abc
    K_GRID = tuple(max(1, int(round(K_DENSITY / (2 * np.pi) * rl))) for rl in recip_lengths)
    print(f"Auto k-grid: {K_GRID[0]}x{K_GRID[1]}x{K_GRID[2]} (density={K_DENSITY})")
else:
    print(f"Manual k-grid: {K_GRID[0]}x{K_GRID[1]}x{K_GRID[2]}")

# ── Build control parameters ─────────────────────────────────────
control = {
    "calculation": CALCULATION,
    "prefix": structure.composition.reduced_formula.replace(" ", ""),
    "outdir": "./tmp",
    "pseudo_dir": PSEUDO_DIR,
    "tprnfor": True,
    "tstress": True,
    "verbosity": "high",
}

system = {
    "ecutwfc": ECUTWFC,
    "ecutrho": ECUTRHO,
    "occupations": "smearing",
    "smearing": SMEARING,
    "degauss": DEGAUSS,
}

electrons = {
    "conv_thr": 1.0e-8,
    "mixing_beta": 0.7,
    "electron_maxstep": 200,
}

ions = None
cell = None

if CALCULATION in ("relax", "vc-relax"):
    control["forc_conv_thr"] = 1.0e-4
    control["etot_conv_thr"] = 1.0e-6
    ions = {"ion_dynamics": "bfgs"}

if CALCULATION == "vc-relax":
    cell = {
        "cell_dynamics": "bfgs",
        "press": 0.0,
        "press_conv_thr": 0.5,
    }

if MAGNETIC:
    system["nspin"] = 2
    # Set initial magnetization for each element type
    for i, el in enumerate(structure.composition.elements, 1):
        system[f"starting_magnetization({i})"] = 0.5

# ── Generate QE input ────────────────────────────────────────────
pw_input = PWInput(
    structure,
    pseudo=pseudo_map,
    control=control,
    system=system,
    electrons=electrons,
    ions=ions,
    cell=cell,
    kpoints_grid=K_GRID,
)

pw_input.write_file(OUTPUT_FILE)
print(f"\nQE input written to: {OUTPUT_FILE}")

# ── Display the generated input ──────────────────────────────────
with open(OUTPUT_FILE) as f:
    print("\n--- Generated QE Input ---")
    print(f.read())
```

#### Step A4: Generate QE input manually (full control)

```python
#!/usr/bin/env python3
"""
Generate a QE pw.x input file with full manual control over all parameters.
Useful when pymatgen's PWInput does not support a specific feature
(e.g., DFT+U, SOC, fixed-occupations, custom k-paths).
"""
import os
import numpy as np
from pymatgen.core.structure import Structure
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer

# ── CONFIGURATION ────────────────────────────────────────────────
INPUT_STRUCTURE = "structure.cif"
PSEUDO_DIR = os.path.abspath("./pseudo")
CALCULATION = "scf"        # scf / relax / vc-relax / nscf / bands
ECUTWFC = 50.0
ECUTRHO = 400.0
K_GRID = (6, 6, 6)
# DFT+U parameters (set to None to disable)
HUBBARD_U = None           # e.g., {"Fe": 4.0, "Co": 3.5}  (eV)
# SOC (spin-orbit coupling)
SOC = False
# Van der Waals correction
VDW = None                 # "dft-d3", "dft-d2", "vdw-df", "vdw-df2"
OUTPUT_FILE = "pw_manual.in"
# ─────────────────────────────────────────────────────────────────

structure = Structure.from_file(INPUT_STRUCTURE)
sga = SpacegroupAnalyzer(structure, symprec=0.01)

# Get species info
species_list = list(dict.fromkeys(str(sp) for sp in structure.species))
atomic_masses = {str(sp): sp.atomic_mass for sp in structure.composition.elements}

# Pseudopotential map
pseudo_map = {}
for el in species_list:
    # Try to find UPF in pseudo_dir
    if os.path.exists(PSEUDO_DIR):
        for fname in os.listdir(PSEUDO_DIR):
            if fname.endswith(".UPF") and fname.lower().startswith(el.lower()):
                pseudo_map[el] = fname
                break
    if el not in pseudo_map:
        pseudo_map[el] = f"{el}.UPF"

# Build the input string
lines = []

# &CONTROL
lines.append("&CONTROL")
lines.append(f"    calculation  = '{CALCULATION}'")
lines.append(f"    prefix       = '{structure.composition.reduced_formula}'")
lines.append(f"    outdir       = './tmp'")
lines.append(f"    pseudo_dir   = '{PSEUDO_DIR}'")
lines.append(f"    tprnfor      = .true.")
lines.append(f"    tstress      = .true.")
lines.append(f"    verbosity    = 'high'")
if CALCULATION in ("relax", "vc-relax"):
    lines.append(f"    forc_conv_thr = 1.0d-4")
    lines.append(f"    etot_conv_thr = 1.0d-6")
lines.append("/\n")

# &SYSTEM
lines.append("&SYSTEM")
lines.append(f"    ibrav        = 0")
lines.append(f"    nat          = {len(structure)}")
lines.append(f"    ntyp         = {len(species_list)}")
lines.append(f"    ecutwfc      = {ECUTWFC}")
lines.append(f"    ecutrho      = {ECUTRHO}")
lines.append(f"    occupations  = 'smearing'")
lines.append(f"    smearing     = 'cold'")
lines.append(f"    degauss      = 0.01")

# ── nbnd: MUST set explicitly for doped/substituted systems ──
# Read z_valence from each UPF to compute total electrons
import re as _re
n_electrons = 0
for el in species_list:
    upf_path = os.path.join(PSEUDO_DIR, pseudo_map.get(el, f"{el}.UPF"))
    if os.path.exists(upf_path):
        with open(upf_path, 'r', errors='ignore') as _f:
            header = _f.read(4000)
        m = _re.search(r'z_valence\s*=\s*"?([\d.]+)', header)
        zv = float(m.group(1)) if m else 0
    else:
        zv = 0
    count = sum(1 for s in structure if str(s.specie) == el)
    n_electrons += zv * count
if n_electrons > 0:
    nbnd = int(n_electrons / 2 * 1.2) + 4
    lines.append(f"    nbnd         = {nbnd}")

if SOC:
    lines.append(f"    noncolin     = .true.")
    lines.append(f"    lspinorb     = .true.")

if HUBBARD_U:
    lines.append(f"    lda_plus_u   = .true.")
    for i, el in enumerate(species_list, 1):
        if el in HUBBARD_U:
            lines.append(f"    Hubbard_U({i}) = {HUBBARD_U[el]}")

if VDW:
    if VDW == "dft-d3":
        lines.append(f"    vdw_corr     = 'dft-d3'")
        lines.append(f"    dftd3_version = 4")
    elif VDW == "dft-d2":
        lines.append(f"    vdw_corr     = 'dft-d2'")
    else:
        lines.append(f"    input_dft    = '{VDW}'")

lines.append("/\n")

# &ELECTRONS
lines.append("&ELECTRONS")
lines.append(f"    conv_thr     = 1.0d-8")
lines.append(f"    mixing_beta  = 0.7")
lines.append(f"    electron_maxstep = 200")
lines.append("/\n")

# &IONS (if relax or vc-relax)
if CALCULATION in ("relax", "vc-relax"):
    lines.append("&IONS")
    lines.append(f"    ion_dynamics = 'bfgs'")
    lines.append("/\n")

# &CELL (if vc-relax)
if CALCULATION == "vc-relax":
    lines.append("&CELL")
    lines.append(f"    cell_dynamics = 'bfgs'")
    lines.append(f"    press         = 0.0")
    lines.append(f"    press_conv_thr = 0.5")
    lines.append("/\n")

# ATOMIC_SPECIES
lines.append("ATOMIC_SPECIES")
for el in species_list:
    mass = float(atomic_masses.get(el, 1.0))
    lines.append(f"  {el:4s} {mass:10.4f}  {pseudo_map[el]}")
lines.append("")

# CELL_PARAMETERS (angstrom)
lines.append("CELL_PARAMETERS angstrom")
cell_matrix = structure.lattice.matrix
for row in cell_matrix:
    lines.append(f"  {row[0]:16.10f} {row[1]:16.10f} {row[2]:16.10f}")
lines.append("")

# ATOMIC_POSITIONS (crystal)
lines.append("ATOMIC_POSITIONS crystal")
for site in structure:
    fc = site.frac_coords
    lines.append(f"  {str(site.specie):4s} {fc[0]:16.10f} {fc[1]:16.10f} {fc[2]:16.10f}")
lines.append("")

# K_POINTS
lines.append("K_POINTS automatic")
lines.append(f"  {K_GRID[0]} {K_GRID[1]} {K_GRID[2]}  0 0 0")

# Write output
input_text = "\n".join(lines) + "\n"
with open(OUTPUT_FILE, "w") as f:
    f.write(input_text)
print(f"QE input written to: {OUTPUT_FILE}")
print(f"\n--- Generated QE Input ---")
print(input_text)
```

#### Step A5: Automatic k-point grid estimation

```python
#!/usr/bin/env python3
"""
Estimate optimal k-point grid for a given structure based on
reciprocal lattice lengths and target density, following the
convention used by VASPKIT and pymatgen.
"""
import numpy as np
from pymatgen.core.structure import Structure

def estimate_kgrid(structure, kppa=1000, force_gamma=True):
    """
    Estimate k-point grid for a structure.

    Parameters
    ----------
    structure : pymatgen Structure
        Input structure.
    kppa : int
        Target k-points per reciprocal atom (default 1000).
        Typical values: 500 (coarse), 1000 (standard), 2000 (fine), 4000 (very fine).
    force_gamma : bool
        If True, ensure Gamma-centered grid (odd grid values).

    Returns
    -------
    tuple : (nk1, nk2, nk3)
    """
    natoms = len(structure)
    total_kpts = kppa / natoms
    recip_lengths = np.array(structure.lattice.reciprocal_lattice.abc)

    # Distribute k-points proportionally to reciprocal lattice lengths
    ratios = recip_lengths / recip_lengths.min()
    base = (total_kpts / np.prod(ratios)) ** (1.0 / 3.0)
    kgrid = np.maximum(1, np.round(base * ratios)).astype(int)

    if force_gamma:
        # Make all values odd for Gamma-centered mesh
        kgrid = np.array([k if k % 2 == 1 else k + 1 for k in kgrid])

    return tuple(kgrid)

def estimate_kgrid_by_length(structure, min_length=30.0):
    """
    Estimate k-point grid using minimum k-point line density.
    kgrid_i = max(1, ceil(min_length / lattice_parameter_i))

    Parameters
    ----------
    structure : pymatgen Structure
    min_length : float
        Minimum real-space supercell length along each direction (Angstrom).
        VASP KSPACING equivalent: kspacing = 2*pi/min_length.
        Typical: 20 (coarse), 30 (standard), 50 (fine).
    """
    abc = structure.lattice.abc
    kgrid = tuple(max(1, int(np.ceil(min_length / a))) for a in abc)
    return kgrid

# ── Example usage ─────────────────────────────────────────────────
structure = Structure.from_file("structure.cif")

kgrid_kppa = estimate_kgrid(structure, kppa=1000)
kgrid_len = estimate_kgrid_by_length(structure, min_length=30.0)

print(f"Structure: {structure.composition.reduced_formula} ({len(structure)} atoms)")
print(f"Lattice: a={structure.lattice.a:.3f}, b={structure.lattice.b:.3f}, c={structure.lattice.c:.3f}")
print(f"K-grid (KPPA=1000):     {kgrid_kppa[0]}x{kgrid_kppa[1]}x{kgrid_kppa[2]}")
print(f"K-grid (min_length=30): {kgrid_len[0]}x{kgrid_len[1]}x{kgrid_len[2]}")

# For metals, use denser grids:
kgrid_metal = estimate_kgrid(structure, kppa=3000)
print(f"K-grid (metals, KPPA=3000): {kgrid_metal[0]}x{kgrid_metal[1]}x{kgrid_metal[2]}")
```

---

### Method B: QE DFT -- Generate Input from Scratch

This method generates a complete QE input file and runs it. The structure is loaded, pseudopotentials are selected, and the input is generated and executed in one script.

```python
#!/usr/bin/env python3
"""
Complete QE input generation and execution workflow.
Loads a structure, downloads pseudopotentials, generates input, and runs pw.x.
"""
import os
import subprocess
import json
import urllib.request
import numpy as np
from pymatgen.core.structure import Structure

# ── CONFIGURATION ────────────────────────────────────────────────
INPUT_FILE = "structure.cif"
CALCULATION = "vc-relax"    # scf, relax, vc-relax
PSEUDO_DIR = os.path.abspath("./pseudo")
OUTDIR = os.path.abspath("./tmp")
NPROCS = 4                 # MPI processes
K_DENSITY = 40              # k-points per reciprocal Angstrom
# ─────────────────────────────────────────────────────────────────

os.makedirs(PSEUDO_DIR, exist_ok=True)
os.makedirs(OUTDIR, exist_ok=True)

# Load structure
structure = Structure.from_file(INPUT_FILE)
elements = [str(el) for el in structure.composition.elements]
print(f"Structure: {structure.composition.reduced_formula}")
print(f"Elements: {elements}")

# Download SSSP pseudopotentials
SSSP_URL = "https://raw.githubusercontent.com/aiidateam/pseudo-data/main/sssp/sssp_efficiency_1.3.0_pbe.json"
sssp_json_path = os.path.join(PSEUDO_DIR, "sssp_efficiency_1.3.0_pbe.json")

if not os.path.exists(sssp_json_path):
    print("Downloading SSSP metadata...")
    urllib.request.urlretrieve(SSSP_URL, sssp_json_path)

with open(sssp_json_path) as f:
    sssp_data = json.load(f)

BASE_URL = "https://raw.githubusercontent.com/aiidateam/pseudo-data/main/sssp/pseudos/"
pseudo_map = {}
max_ecutwfc = 0
max_ecutrho = 0

for el in elements:
    if el not in sssp_data:
        raise ValueError(f"Element {el} not in SSSP database")
    info = sssp_data[el]
    fname = info["filename"]
    pseudo_map[el] = fname
    max_ecutwfc = max(max_ecutwfc, info.get("cutoff_wfc", 50))
    max_ecutrho = max(max_ecutrho, info.get("cutoff_rho", 400))
    dest = os.path.join(PSEUDO_DIR, fname)
    if not os.path.exists(dest):
        print(f"  Downloading {fname}...")
        urllib.request.urlretrieve(BASE_URL + fname, dest)

print(f"Recommended cutoffs: ecutwfc={max_ecutwfc} Ry, ecutrho={max_ecutrho} Ry")

# Auto k-grid
recip_lengths = structure.lattice.reciprocal_lattice.abc
kgrid = tuple(max(1, int(round(K_DENSITY / (2 * np.pi) * rl))) for rl in recip_lengths)
print(f"Auto k-grid: {kgrid[0]}x{kgrid[1]}x{kgrid[2]}")

# Build the input file
species_order = list(dict.fromkeys(str(sp) for sp in structure.species))
cell = structure.lattice.matrix

lines = []
lines.append("&CONTROL")
lines.append(f"    calculation  = '{CALCULATION}'")
lines.append(f"    prefix       = '{structure.composition.reduced_formula}'")
lines.append(f"    outdir       = '{OUTDIR}'")
lines.append(f"    pseudo_dir   = '{PSEUDO_DIR}'")
lines.append(f"    tprnfor      = .true.")
lines.append(f"    tstress      = .true.")
if CALCULATION in ("relax", "vc-relax"):
    lines.append(f"    forc_conv_thr = 1.0d-4")
    lines.append(f"    etot_conv_thr = 1.0d-6")
lines.append("/\n")

lines.append("&SYSTEM")
lines.append(f"    ibrav        = 0")
lines.append(f"    nat          = {len(structure)}")
lines.append(f"    ntyp         = {len(species_order)}")
lines.append(f"    ecutwfc      = {max_ecutwfc}")
lines.append(f"    ecutrho      = {max_ecutrho}")
lines.append(f"    occupations  = 'smearing'")
lines.append(f"    smearing     = 'cold'")
lines.append(f"    degauss      = 0.01")
lines.append("/\n")

lines.append("&ELECTRONS")
lines.append(f"    conv_thr     = 1.0d-8")
lines.append(f"    mixing_beta  = 0.7")
lines.append("/\n")

if CALCULATION in ("relax", "vc-relax"):
    lines.append("&IONS")
    lines.append(f"    ion_dynamics = 'bfgs'")
    lines.append("/\n")

if CALCULATION == "vc-relax":
    lines.append("&CELL")
    lines.append(f"    cell_dynamics = 'bfgs'")
    lines.append(f"    press         = 0.0")
    lines.append(f"    press_conv_thr = 0.5")
    lines.append("/\n")

lines.append("ATOMIC_SPECIES")
for el in species_order:
    from pymatgen.core.periodic_table import Element
    mass = Element(el).atomic_mass
    lines.append(f"  {el:4s} {float(mass):10.4f}  {pseudo_map[el]}")
lines.append("")

lines.append("CELL_PARAMETERS angstrom")
for row in cell:
    lines.append(f"  {row[0]:16.10f} {row[1]:16.10f} {row[2]:16.10f}")
lines.append("")

lines.append("ATOMIC_POSITIONS crystal")
for site in structure:
    fc = site.frac_coords
    lines.append(f"  {str(site.specie):4s} {fc[0]:16.10f} {fc[1]:16.10f} {fc[2]:16.10f}")
lines.append("")

lines.append("K_POINTS automatic")
lines.append(f"  {kgrid[0]} {kgrid[1]} {kgrid[2]}  0 0 0")

input_text = "\n".join(lines) + "\n"
input_filename = f"{structure.composition.reduced_formula}_{CALCULATION}.in"
output_filename = input_filename.replace(".in", ".out")

with open(input_filename, "w") as f:
    f.write(input_text)
print(f"\nInput written to: {input_filename}")

# Run QE
print(f"Running pw.x with {NPROCS} MPI processes...")
result = subprocess.run(
    ["mpirun", "-np", str(NPROCS), "pw.x", "-in", input_filename],
    capture_output=True, text=True, timeout=3600
)

with open(output_filename, "w") as f:
    f.write(result.stdout)

if result.returncode != 0:
    print(f"ERROR: pw.x failed!")
    print(result.stderr[-500:] if result.stderr else "No stderr")
else:
    print(f"Output written to: {output_filename}")
    # Check for convergence
    if "convergence has been achieved" in result.stdout:
        print("SCF convergence achieved.")
    if "bfgs converged" in result.stdout or "End final coordinates" in result.stdout:
        print("Ionic relaxation converged.")
```

---

### Method C: VASP -- Generate Input Files with pymatgen

This method generates complete VASP input files (INCAR, KPOINTS, POSCAR, POTCAR) using pymatgen. VASP itself is not available in the container, but the input files can be prepared here for execution elsewhere.

#### Step C1: Generate POSCAR

```python
#!/usr/bin/env python3
"""
Generate VASP POSCAR from any structure source.
Equivalent to VASPKIT function 101.
"""
from pymatgen.core.structure import Structure
from pymatgen.io.vasp import Poscar
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer

# Load structure from any format
structure = Structure.from_file("structure.cif")

# Option: use primitive cell to reduce atom count
sga = SpacegroupAnalyzer(structure, symprec=0.01)
# primitive = sga.get_primitive_standard_structure()
# conventional = sga.get_conventional_standard_structure()

# Generate POSCAR (direct/fractional coordinates)
poscar = Poscar(structure, comment=f"{structure.composition.reduced_formula} - generated by pymatgen")
poscar.write_file("POSCAR")
print(f"POSCAR written for {structure.composition.reduced_formula}")
print(f"  Atoms: {len(structure)}")
print(f"  Cell volume: {structure.volume:.4f} A^3")

# Display POSCAR content
with open("POSCAR") as f:
    print("\n--- POSCAR ---")
    print(f.read())
```

#### Step C2: Generate INCAR

```python
#!/usr/bin/env python3
"""
Generate VASP INCAR for various calculation types.
Equivalent to VASPKIT function 102.
Supports: SCF, relaxation, band structure, DOS, MD.
"""
from pymatgen.core.structure import Structure
from pymatgen.io.vasp import Incar
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer

# ── CONFIGURATION ────────────────────────────────────────────────
CALCULATION_TYPE = "relax"   # "scf", "relax", "band", "dos", "md"
STRUCTURE_FILE = "POSCAR"
FUNCTIONAL = "PBE"          # "PBE", "PBEsol", "SCAN", "r2SCAN", "HSE06"
ENCUT = 520                 # Plane-wave cutoff (eV)
IS_METAL = False             # True for metals (affects smearing)
IS_MAGNETIC = False          # True for spin-polarized
HUBBARD_U = None             # e.g., {"Fe": 4.0} for DFT+U
VDW_CORRECTION = None        # "D3", "D3BJ", "dDsC", "TS"
SOC = False                  # Spin-orbit coupling
# ─────────────────────────────────────────────────────────────────

structure = Structure.from_file(STRUCTURE_FILE)

# Base INCAR parameters
incar_dict = {
    # Electronic
    "ENCUT": ENCUT,
    "PREC": "Accurate",
    "EDIFF": 1e-6,
    "NELM": 200,
    "ALGO": "Normal",
    "LREAL": "Auto" if len(structure) > 20 else False,
    "LWAVE": False,
    "LCHARG": False,

    # Parallelization
    "NCORE": 4,
    "KPAR": 2,
}

# Smearing
if IS_METAL:
    incar_dict["ISMEAR"] = 1       # Methfessel-Paxton for metals
    incar_dict["SIGMA"] = 0.2
else:
    incar_dict["ISMEAR"] = 0       # Gaussian for insulators/semiconductors
    incar_dict["SIGMA"] = 0.05

# Calculation-type-specific settings
if CALCULATION_TYPE == "scf":
    incar_dict["NSW"] = 0
    incar_dict["IBRION"] = -1
    incar_dict["LCHARG"] = True

elif CALCULATION_TYPE == "relax":
    incar_dict["NSW"] = 200
    incar_dict["IBRION"] = 2       # Conjugate gradient
    incar_dict["ISIF"] = 3         # Relax ions + cell shape + cell volume
    incar_dict["EDIFFG"] = -0.01   # Force convergence (eV/A, negative = force criterion)
    incar_dict["POTIM"] = 0.5

elif CALCULATION_TYPE == "band":
    incar_dict["NSW"] = 0
    incar_dict["IBRION"] = -1
    incar_dict["ICHARG"] = 11      # Read CHGCAR from SCF
    incar_dict["ISMEAR"] = 0
    incar_dict["SIGMA"] = 0.05
    incar_dict["LORBIT"] = 11
    incar_dict["LCHARG"] = False

elif CALCULATION_TYPE == "dos":
    incar_dict["NSW"] = 0
    incar_dict["IBRION"] = -1
    incar_dict["ICHARG"] = 11
    incar_dict["ISMEAR"] = -5      # Tetrahedron method for DOS
    incar_dict["LORBIT"] = 11
    incar_dict["NEDOS"] = 3001
    incar_dict["LCHARG"] = False

elif CALCULATION_TYPE == "md":
    incar_dict["NSW"] = 5000
    incar_dict["IBRION"] = 0       # Molecular dynamics
    incar_dict["ISIF"] = 2         # Fix cell, relax ions
    incar_dict["POTIM"] = 1.0      # Timestep in fs
    incar_dict["SMASS"] = 0        # Nose-Hoover thermostat
    incar_dict["TEBEG"] = 300      # Start temperature (K)
    incar_dict["TEEND"] = 300      # End temperature (K)
    incar_dict["ISYM"] = 0         # No symmetry for MD
    incar_dict["LREAL"] = "Auto"
    incar_dict["LWAVE"] = False
    incar_dict["LCHARG"] = False

# Functional
if FUNCTIONAL == "PBEsol":
    incar_dict["GGA"] = "PS"
elif FUNCTIONAL == "SCAN":
    incar_dict["METAGGA"] = "SCAN"
    incar_dict["LASPH"] = True
    incar_dict["LMIXTAU"] = True
    incar_dict["ALGO"] = "All"
elif FUNCTIONAL == "r2SCAN":
    incar_dict["METAGGA"] = "R2SCAN"
    incar_dict["LASPH"] = True
    incar_dict["LMIXTAU"] = True
    incar_dict["ALGO"] = "All"
elif FUNCTIONAL == "HSE06":
    incar_dict["LHFCALC"] = True
    incar_dict["HFSCREEN"] = 0.2
    incar_dict["ALGO"] = "Damped"
    incar_dict["TIME"] = 0.4
    incar_dict["PRECFOCK"] = "Fast"
    incar_dict["LASPH"] = True

# Magnetism
if IS_MAGNETIC:
    incar_dict["ISPIN"] = 2
    magmom = []
    for site in structure:
        el = str(site.specie)
        # Default magnetic moments for common magnetic elements
        default_mag = {"Fe": 5.0, "Co": 3.0, "Ni": 2.0, "Mn": 5.0, "Cr": 5.0,
                       "V": 3.0, "Ti": 2.0, "Cu": 1.0, "O": 0.6}
        magmom.append(default_mag.get(el, 0.6))
    incar_dict["MAGMOM"] = " ".join(f"{m:.1f}" for m in magmom)

# DFT+U
if HUBBARD_U:
    incar_dict["LDAU"] = True
    incar_dict["LDAUTYPE"] = 2
    species_order = list(dict.fromkeys(str(sp) for sp in structure.species))
    ldauu = [HUBBARD_U.get(el, 0.0) for el in species_order]
    ldauj = [0.0] * len(species_order)
    ldaul = [2 if el in HUBBARD_U else -1 for el in species_order]  # l=2 for d-electrons
    incar_dict["LDAUU"] = " ".join(f"{u:.1f}" for u in ldauu)
    incar_dict["LDAUJ"] = " ".join(f"{j:.1f}" for j in ldauj)
    incar_dict["LDAUL"] = " ".join(f"{l}" for l in ldaul)
    incar_dict["LMAXMIX"] = 4

# Van der Waals
if VDW_CORRECTION:
    if VDW_CORRECTION in ("D3", "D3BJ"):
        incar_dict["IVDW"] = 12 if VDW_CORRECTION == "D3BJ" else 11
    elif VDW_CORRECTION == "dDsC":
        incar_dict["IVDW"] = 4
    elif VDW_CORRECTION == "TS":
        incar_dict["IVDW"] = 20

# Spin-orbit coupling
if SOC:
    incar_dict["LSORBIT"] = True
    incar_dict["LNONCOLLINEAR"] = True
    incar_dict["ISYM"] = -1

# Write INCAR
incar = Incar(incar_dict)
incar.write_file("INCAR")
print(f"INCAR written for {CALCULATION_TYPE} calculation ({FUNCTIONAL})")

with open("INCAR") as f:
    print("\n--- INCAR ---")
    print(f.read())
```

#### Step C3: Generate KPOINTS

```python
#!/usr/bin/env python3
"""
Generate VASP KPOINTS file for various calculation types.
Equivalent to VASPKIT functions 102-103.
Supports automatic mesh, line-mode for band structure, and custom grids.
"""
import numpy as np
from pymatgen.core.structure import Structure
from pymatgen.io.vasp import Kpoints
from pymatgen.symmetry.bandstructure import HighSymmKpath

# ── CONFIGURATION ────────────────────────────────────────────────
STRUCTURE_FILE = "POSCAR"
KPOINTS_TYPE = "auto"       # "auto", "band", "line_density", "manual"
KPPA = 1000                 # k-points per reciprocal atom (for auto)
LINE_DENSITY = 40           # Points per segment (for band structure)
MANUAL_GRID = (6, 6, 6)    # Manual grid specification
# ─────────────────────────────────────────────────────────────────

structure = Structure.from_file(STRUCTURE_FILE)

if KPOINTS_TYPE == "auto":
    # Automatic Gamma-centered mesh based on KPPA
    kpoints = Kpoints.automatic_density(structure, kppa=KPPA, force_gamma=True)
    kpoints.write_file("KPOINTS")
    print(f"KPOINTS (auto, KPPA={KPPA}):")

elif KPOINTS_TYPE == "band":
    # Line-mode KPOINTS for band structure
    kpath = HighSymmKpath(structure)
    kpoints = Kpoints.automatic_linemode(LINE_DENSITY, kpath)
    kpoints.write_file("KPOINTS")
    print(f"KPOINTS (band structure, {LINE_DENSITY} pts/segment):")
    print(f"  Path: {' -> '.join(kpath.kpath['path'][0])}")

elif KPOINTS_TYPE == "line_density":
    # Line-mode with specified density
    kpath = HighSymmKpath(structure)
    kpoints = Kpoints.automatic_linemode(LINE_DENSITY, kpath)
    kpoints.write_file("KPOINTS")
    print(f"KPOINTS (line mode, density={LINE_DENSITY}):")

elif KPOINTS_TYPE == "manual":
    # Manual Gamma-centered grid
    kpoints = Kpoints.gamma_automatic(MANUAL_GRID)
    kpoints.write_file("KPOINTS")
    print(f"KPOINTS (manual, {MANUAL_GRID[0]}x{MANUAL_GRID[1]}x{MANUAL_GRID[2]}):")

with open("KPOINTS") as f:
    print(f.read())
```

#### Step C4: Generate POTCAR (requires VASP pseudopotential files)

```python
#!/usr/bin/env python3
"""
Generate VASP POTCAR file using pymatgen.
Equivalent to VASPKIT function 103.

IMPORTANT: This requires VASP pseudopotential files to be available
and PMG_VASP_PSP_DIR to be set in ~/.config/pymatgen/config.yml or
as an environment variable.
"""
import os
from pymatgen.core.structure import Structure
from pymatgen.io.vasp import Potcar, PotcarSingle

# ── CONFIGURATION ────────────────────────────────────────────────
STRUCTURE_FILE = "POSCAR"
FUNCTIONAL = "PBE"          # "PBE", "PBE_52", "PBE_54", "LDA", "PW91"
# POTCAR variant preference: use _sv, _pv variants for transition metals
POTCAR_SPEC = None          # e.g., {"Fe": "Fe_pv", "O": "O"}, None for defaults
# ─────────────────────────────────────────────────────────────────

structure = Structure.from_file(STRUCTURE_FILE)
species_order = list(dict.fromkeys(str(sp) for sp in structure.species))

# Check if VASP pseudopotentials are available
psp_dir = os.environ.get("PMG_VASP_PSP_DIR", "")
if not psp_dir:
    print("WARNING: PMG_VASP_PSP_DIR not set.")
    print("To configure, run:")
    print("  pmg config -p <EXTRACTED_VASP_PSP> <MY_PSP>")
    print("Or set in ~/.config/pymatgen/config.yml:")
    print("  PMG_VASP_PSP_DIR: /path/to/VASP/pseudopotentials")
    print("\nGenerating POTCAR specification instead...")

    # Print recommended POTCAR types (VASP wiki recommendations)
    recommended = {
        "Li": "Li_sv", "Na": "Na_pv", "K": "K_sv", "Rb": "Rb_sv", "Cs": "Cs_sv",
        "Ca": "Ca_sv", "Sr": "Sr_sv", "Ba": "Ba_sv",
        "Sc": "Sc_sv", "Ti": "Ti_pv", "V": "V_pv", "Cr": "Cr_pv",
        "Mn": "Mn_pv", "Fe": "Fe_pv", "Co": "Co", "Ni": "Ni_pv",
        "Cu": "Cu_pv", "Zn": "Zn", "Y": "Y_sv", "Zr": "Zr_sv",
        "Nb": "Nb_pv", "Mo": "Mo_pv", "Ru": "Ru_pv", "Rh": "Rh_pv",
        "Pd": "Pd", "Ag": "Ag", "Hf": "Hf_pv", "Ta": "Ta_pv",
        "W": "W_pv", "Re": "Re_pv", "Os": "Os_pv", "Ir": "Ir",
        "Pt": "Pt", "Au": "Au",
        "Ga": "Ga_d", "Ge": "Ge_d", "In": "In_d", "Sn": "Sn_d",
    }
    print(f"\nRecommended POTCAR types for {species_order}:")
    for el in species_order:
        rec = recommended.get(el, el)
        print(f"  {el} -> {rec}")
else:
    # Generate POTCAR
    if POTCAR_SPEC:
        symbols = [POTCAR_SPEC.get(el, el) for el in species_order]
    else:
        symbols = species_order

    try:
        potcar = Potcar(symbols, functional=FUNCTIONAL)
        potcar.write_file("POTCAR")
        print(f"POTCAR written for: {', '.join(symbols)} ({FUNCTIONAL})")
        for p in potcar:
            print(f"  {p.symbol}: ENMAX={p.enmax:.1f} eV, ZVAL={p.zval:.1f}")
        # Recommend ENCUT
        enmax_values = [p.enmax for p in potcar]
        print(f"\nRecommended ENCUT: {max(enmax_values) * 1.3:.0f} eV (1.3 x max ENMAX)")
    except Exception as e:
        print(f"ERROR generating POTCAR: {e}")
        print("Check that PMG_VASP_PSP_DIR is correctly configured.")
```

#### Step C5: Generate complete VASP input set with MPRelaxSet

```python
#!/usr/bin/env python3
"""
Generate a complete set of VASP input files using pymatgen's MPRelaxSet,
MPStaticSet, MPNonSCFSet, or other predefined input sets.
Equivalent to VASPKIT functions 101-109 combined.
"""
from pymatgen.core.structure import Structure
from pymatgen.io.vasp.sets import (
    MPRelaxSet,
    MPStaticSet,
    MPMetalRelaxSet,
)
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer

# ── CONFIGURATION ────────────────────────────────────────────────
STRUCTURE_FILE = "structure.cif"
CALC_TYPE = "relax"         # "relax", "static", "metal_relax"
OUTPUT_DIR = "./vasp_input"
USER_INCAR = {}             # Override specific INCAR parameters
# ─────────────────────────────────────────────────────────────────

structure = Structure.from_file(STRUCTURE_FILE)
print(f"Structure: {structure.composition.reduced_formula}")

sga = SpacegroupAnalyzer(structure, symprec=0.01)
print(f"Space group: {sga.get_space_group_symbol()} (#{sga.get_space_group_number()})")

# Select the appropriate input set
if CALC_TYPE == "relax":
    input_set = MPRelaxSet(structure, user_incar_settings=USER_INCAR)
elif CALC_TYPE == "static":
    input_set = MPStaticSet(structure, user_incar_settings=USER_INCAR)
elif CALC_TYPE == "metal_relax":
    input_set = MPMetalRelaxSet(structure, user_incar_settings=USER_INCAR)
else:
    raise ValueError(f"Unknown calc type: {CALC_TYPE}")

# Write all input files
input_set.write_input(OUTPUT_DIR)
print(f"\nVASP input files written to: {OUTPUT_DIR}/")

# List generated files
import os
for fname in sorted(os.listdir(OUTPUT_DIR)):
    fpath = os.path.join(OUTPUT_DIR, fname)
    size = os.path.getsize(fpath)
    print(f"  {fname} ({size} bytes)")

# Display INCAR
print("\n--- INCAR ---")
with open(os.path.join(OUTPUT_DIR, "INCAR")) as f:
    print(f.read())

# Display KPOINTS
print("--- KPOINTS ---")
with open(os.path.join(OUTPUT_DIR, "KPOINTS")) as f:
    print(f.read())
```

---

## Key Parameters

| Parameter | QE | VASP | Notes |
|---|---|---|---|
| Plane-wave cutoff | `ecutwfc` (Ry) | `ENCUT` (eV) | 1 Ry = 13.6 eV. Use SSSP recommended for QE, 1.3x ENMAX for VASP. |
| Charge density cutoff | `ecutrho` (Ry) | N/A (auto) | 4x ecutwfc for norm-conserving, 8-12x for ultrasoft/PAW. |
| K-point grid | `K_POINTS automatic` | `KPOINTS` file | KPPA=1000 for semiconductors, 3000+ for metals. |
| SCF convergence | `conv_thr` (Ry) | `EDIFF` (eV) | QE: 1e-8 Ry. VASP: 1e-6 eV. |
| Force convergence | `forc_conv_thr` (Ry/Bohr) | `EDIFFG` (eV/A) | QE: 1e-4. VASP: -0.01 (negative = force criterion). |
| Smearing | `smearing`/`degauss` | `ISMEAR`/`SIGMA` | QE: 'cold'/0.01. VASP: 0/0.05 (insulator), 1/0.2 (metal). |
| Ionic relaxation | `calculation='relax'` | `IBRION=2`, `NSW=200` | QE: 'relax' (ions) or 'vc-relax' (ions+cell). VASP: ISIF=2 (ions), ISIF=3 (ions+cell). |
| Pseudopotentials | UPF files in pseudo_dir | POTCAR | QE: SSSP library. VASP: PAW_PBE recommended. |
| Number of bands | `nbnd` | `NBANDS` | QE auto-sets to n_electrons/2 (+few). **For doped/substituted systems**, you MUST calculate `nbnd` manually: read `z_valence` from each UPF, sum total electrons, set `nbnd = int(n_electrons/2 * 1.2) + 4`. |

## Interpreting Results

- **Input validation**: Before running, check that atom count, species, and cell parameters match expectations. Verify pseudopotentials exist for all elements.
- **K-point convergence**: For production results, always run convergence tests. Energy should converge to within 1 meV/atom.
- **Cutoff convergence**: SSSP cutoffs are pre-converged for most properties. For stresses or elastic constants, use higher cutoffs.
- **INCAR checks**: Verify ENCUT is above 1.3x the maximum ENMAX in POTCAR. Check ISMEAR is appropriate for the system (metal vs insulator).

## Common Issues

| Problem | Solution |
|---|---|
| Pseudopotential not found (QE) | Check `pseudo_dir` path is absolute. Verify UPF filename matches `ATOMIC_SPECIES`. Run the download script. |
| POTCAR generation fails (VASP) | Set `PMG_VASP_PSP_DIR` via `pmg config -p`. Ensure pseudopotential files are extracted. |
| Wrong k-point grid | Use `estimate_kgrid()` function or pymatgen's `Kpoints.automatic_density()`. Increase KPPA for metals. |
| INCAR parameter conflict | METAGGA and GGA cannot be set simultaneously. Remove GGA when using SCAN/r2SCAN. |
| Structure has wrong symmetry | Symmetrize first with `SpacegroupAnalyzer.get_refined_structure()`. Check `symprec`. |
| Too many atoms for DFT | Use primitive cell (`get_primitive_standard_structure()`). Consider MACE for screening. |
| QE `too few bands` for doped systems | Aliovalent doping changes total electron count (e.g., La³⁺→Na⁺ adds 2e). Read `z_valence` from each UPF header, sum for all atoms, set `nbnd = int(n_electrons/2 * 1.2) + 4`. |
| QE input has ibrav != 0 | When using `ibrav=0`, provide CELL_PARAMETERS. pymatgen always uses `ibrav=0`. |
| VASP POSCAR coordinate type | pymatgen writes direct (fractional) by default. Use `Poscar(structure, direct=False)` for Cartesian. |
