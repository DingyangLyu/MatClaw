# Spontaneous Polarization via Berry Phase

## When to Use

- You need the spontaneous (switchable) polarization of a ferroelectric material in C/m^2.
- You have identified both a polar phase and a centrosymmetric reference phase.
- You need to track polarization along a switching path (lambda = 0 centrosymmetric to lambda = 1 polar) to resolve the polarization quantum ambiguity.
- You want to compare computed polarization with experimental values (e.g., BaTiO3 ~0.26 C/m^2, PbTiO3 ~0.75 C/m^2).

## Prerequisites

- A relaxed polar structure (ferroelectric phase) and a relaxed centrosymmetric reference structure (paraelectric / nonpolar phase).
- Both structures must have the same number of atoms, same species, and compatible unit cells (the polar distortion should be a continuous deformation of the reference).
- Quantum ESPRESSO 7.5 with `pw.x` available on PATH.
- Pseudopotential files (SSSP or PSlibrary recommended).
- Python packages: `pymatgen`, `ase`, `numpy`, `matplotlib`.

## Background: Modern Theory of Polarization

The absolute polarization of a bulk crystal is not a well-defined observable -- only polarization *differences* are measurable. The Berry phase approach (King-Smith and Vanderbilt, 1993) computes the electronic contribution to polarization as a geometric phase of the occupied Bloch states. The total polarization is:

```
P_total = P_electronic + P_ionic
```

where `P_ionic = (e / Omega) * sum_i Z_i * r_i` is the classical ionic contribution and `P_electronic` is the Berry phase. Both are defined only modulo the **polarization quantum** `e*R / Omega` (where R is a lattice vector and Omega is the cell volume). To resolve this ambiguity, one computes the polarization along an adiabatic path from a centrosymmetric reference (where P = 0 by symmetry) to the polar phase, ensuring the polarization branch is continuous.

## Detailed Steps

### Overview of the Full Workflow

```
Step 0: Download pseudopotentials
Step 1: Relax the centrosymmetric (nonpolar) reference structure
Step 2: Relax the ferroelectric (polar) structure
Step 3: Generate interpolated structures (lambda = 0 to 1)
Step 4: Run SCF for all structures
Step 5: Run Berry phase calculation for all structures
Step 6: Parse Berry phase output, compute polarization, resolve branches
Step 7: Plot polarization vs. lambda and extract spontaneous polarization
```

This mirrors the atomate2 `FerroelectricMaker` workflow: relax both endpoints, interpolate N images, compute LCALCPOL (Berry phase) at each image, then run `polarization_analysis` to resolve branches.

### Step 0: Download Pseudopotentials

```python
#!/usr/bin/env python3
"""Download SSSP Efficiency pseudopotentials for BaTiO3."""
import os
import urllib.request

PSEUDO_DIR = os.path.abspath("./pseudo")
os.makedirs(PSEUDO_DIR, exist_ok=True)

# SSSP Efficiency pseudopotentials (PBE)
pseudos = {
    "Ba": "Ba.pbe-spn-kjpaw_psl.1.0.0.UPF",
    "Ti": "Ti.pbe-spn-kjpaw_psl.1.0.0.UPF",
    "O":  "O.pbe-n-kjpaw_psl.1.0.0.UPF",
}

BASE_URL = "https://pseudopotentials.quantum-espresso.org/upf_files/"

for element, fname in pseudos.items():
    dest = os.path.join(PSEUDO_DIR, fname)
    if os.path.exists(dest):
        print(f"  {fname} already exists, skipping.")
        continue
    url = BASE_URL + fname
    print(f"  Downloading {fname} ...")
    try:
        urllib.request.urlretrieve(url, dest)
        print(f"  OK: {fname}")
    except Exception as e:
        print(f"  WARNING: Could not download {fname}: {e}")

print(f"\nPseudopotentials in {PSEUDO_DIR}:")
for f in sorted(os.listdir(PSEUDO_DIR)):
    print(f"  {f}")
```

### Step 1: Relax the Centrosymmetric Reference (Nonpolar Phase)

```python
#!/usr/bin/env python3
"""
Relax the cubic (centrosymmetric) BaTiO3 structure using QE vc-relax.
Space group Pm-3m (#221). This is the nonpolar reference where P = 0 by symmetry.
"""
import os
import subprocess

PSEUDO_DIR = os.path.abspath("./pseudo")
OUTDIR = os.path.abspath("./tmp_nonpolar")
os.makedirs(OUTDIR, exist_ok=True)

# Cubic BaTiO3: a = 4.00 Angstrom (experimental ~4.00 A)
# Convert to Bohr: 4.00 * 1.8897259886 = 7.5589
alat_bohr = 7.5589

vcrelax_input = f"""&CONTROL
    calculation  = 'vc-relax'
    prefix       = 'batio3_cubic'
    outdir       = '{OUTDIR}'
    pseudo_dir   = '{PSEUDO_DIR}'
    tprnfor      = .true.
    tstress      = .true.
    forc_conv_thr = 1.0d-4
    etot_conv_thr = 1.0d-6
/

&SYSTEM
    ibrav       = 1
    celldm(1)   = {alat_bohr}
    nat         = 5
    ntyp        = 3
    ecutwfc     = 50.0
    ecutrho     = 400.0
    occupations = 'smearing'
    smearing    = 'cold'
    degauss     = 0.005
/

&ELECTRONS
    conv_thr    = 1.0d-10
    mixing_beta = 0.5
/

&IONS
    ion_dynamics = 'bfgs'
/

&CELL
    cell_dynamics = 'bfgs'
    press         = 0.0
    press_conv_thr = 0.1
/

ATOMIC_SPECIES
  Ba  137.327  Ba.pbe-spn-kjpaw_psl.1.0.0.UPF
  Ti   47.867  Ti.pbe-spn-kjpaw_psl.1.0.0.UPF
  O    15.999  O.pbe-n-kjpaw_psl.1.0.0.UPF

ATOMIC_POSITIONS (crystal)
  Ba  0.000  0.000  0.000
  Ti  0.500  0.500  0.500
  O   0.500  0.500  0.000
  O   0.500  0.000  0.500
  O   0.000  0.500  0.500

K_POINTS (automatic)
  6 6 6  1 1 1
"""

with open("batio3_nonpolar_vcrelax.in", "w") as f:
    f.write(vcrelax_input)
print("Written: batio3_nonpolar_vcrelax.in")

print("Running pw.x vc-relax for nonpolar phase ...")
result = subprocess.run(
    ["mpirun", "-np", "4", "pw.x", "-in", "batio3_nonpolar_vcrelax.in"],
    capture_output=True, text=True, timeout=3600
)
with open("batio3_nonpolar_vcrelax.out", "w") as f:
    f.write(result.stdout)

if result.returncode != 0:
    print("ERROR: vc-relax failed!")
    print(result.stderr[-500:] if result.stderr else "No stderr")
else:
    print("Nonpolar vc-relax completed successfully.")
    print("Output: batio3_nonpolar_vcrelax.out")
```

### Step 2: Relax the Ferroelectric (Polar) Phase

```python
#!/usr/bin/env python3
"""
Relax tetragonal (ferroelectric) BaTiO3 with QE vc-relax.
Space group P4mm (#99). Polar axis along z.
Start from experimental tetragonal structure with Ti/O displacements.
"""
import os
import subprocess

PSEUDO_DIR = os.path.abspath("./pseudo")
OUTDIR = os.path.abspath("./tmp_polar")
os.makedirs(OUTDIR, exist_ok=True)

# Tetragonal BaTiO3: a = 3.994 A, c = 4.034 A (experimental)
# Use ibrav=0 with explicit CELL_PARAMETERS for tetragonal cell
a_ang = 3.994
c_ang = 4.034

vcrelax_input = f"""&CONTROL
    calculation  = 'vc-relax'
    prefix       = 'batio3_tetra'
    outdir       = '{OUTDIR}'
    pseudo_dir   = '{PSEUDO_DIR}'
    tprnfor      = .true.
    tstress      = .true.
    forc_conv_thr = 1.0d-4
    etot_conv_thr = 1.0d-6
/

&SYSTEM
    ibrav       = 0
    nat         = 5
    ntyp        = 3
    ecutwfc     = 50.0
    ecutrho     = 400.0
    occupations = 'smearing'
    smearing    = 'cold'
    degauss     = 0.005
/

&ELECTRONS
    conv_thr    = 1.0d-10
    mixing_beta = 0.5
/

&IONS
    ion_dynamics = 'bfgs'
/

&CELL
    cell_dynamics = 'bfgs'
    press         = 0.0
    press_conv_thr = 0.1
/

CELL_PARAMETERS (angstrom)
  {a_ang:.6f}  0.000000  0.000000
  0.000000  {a_ang:.6f}  0.000000
  0.000000  0.000000  {c_ang:.6f}

ATOMIC_SPECIES
  Ba  137.327  Ba.pbe-spn-kjpaw_psl.1.0.0.UPF
  Ti   47.867  Ti.pbe-spn-kjpaw_psl.1.0.0.UPF
  O    15.999  O.pbe-n-kjpaw_psl.1.0.0.UPF

ATOMIC_POSITIONS (crystal)
  Ba  0.000  0.000  0.000
  Ti  0.500  0.500  0.520
  O   0.500  0.500  -0.020
  O   0.500  0.000  0.520
  O   0.000  0.500  0.520

K_POINTS (automatic)
  6 6 6  1 1 1
"""

with open("batio3_polar_vcrelax.in", "w") as f:
    f.write(vcrelax_input)
print("Written: batio3_polar_vcrelax.in")

print("Running pw.x vc-relax for polar phase ...")
result = subprocess.run(
    ["mpirun", "-np", "4", "pw.x", "-in", "batio3_polar_vcrelax.in"],
    capture_output=True, text=True, timeout=3600
)
with open("batio3_polar_vcrelax.out", "w") as f:
    f.write(result.stdout)

if result.returncode != 0:
    print("ERROR: vc-relax failed!")
    print(result.stderr[-500:] if result.stderr else "No stderr")
else:
    print("Polar vc-relax completed successfully.")
    print("Output: batio3_polar_vcrelax.out")
```

### Step 3: Generate Interpolated Structures

```python
#!/usr/bin/env python3
"""
Generate interpolated structures between nonpolar (lambda=0) and polar (lambda=1)
endpoints. This creates a continuous path to track the polarization branch.

Mirrors atomate2's interpolate_structures() function:
  p_st.interpolate(np_st, nimages+1, interpolate_lattices=True, autosort_tol=0.0)
"""
import numpy as np
from pymatgen.core import Structure, Lattice
import os
import json

# ── 1. Load relaxed structures ────────────────────────────────────
# In practice, parse these from the vc-relax outputs.
# Here we build them explicitly for BaTiO3 as an example.

# Nonpolar (cubic) BaTiO3 -- lambda = 0
a_cubic = 4.00  # Angstrom (use your relaxed value)
nonpolar = Structure(
    Lattice.cubic(a_cubic),
    species=["Ba", "Ti", "O", "O", "O"],
    coords=[
        [0.000, 0.000, 0.000],
        [0.500, 0.500, 0.500],
        [0.500, 0.500, 0.000],
        [0.500, 0.000, 0.500],
        [0.000, 0.500, 0.500],
    ],
    coords_are_cartesian=False,
)

# Polar (tetragonal) BaTiO3 -- lambda = 1
a_tetra = 3.994
c_tetra = 4.034
polar = Structure(
    Lattice.tetragonal(a_tetra, c_tetra),
    species=["Ba", "Ti", "O", "O", "O"],
    coords=[
        [0.000, 0.000, 0.000],
        [0.500, 0.500, 0.520],
        [0.500, 0.500, -0.020],
        [0.500, 0.000, 0.520],
        [0.000, 0.500, 0.520],
    ],
    coords_are_cartesian=False,
)

# ── 2. Interpolate ────────────────────────────────────────────────
nimages = 8  # Number of intermediate images (matching atomate2 default)

# pymatgen's interpolate gives nimages+2 structures (including endpoints)
images = nonpolar.interpolate(
    polar,
    nimages=nimages + 1,  # +1 to match atomate2 convention
    interpolate_lattices=True,
    autosort_tol=0.0,
)

print(f"Generated {len(images)} structures along interpolation path:")
print(f"  lambda = 0 (nonpolar) to lambda = 1 (polar)")

# ── 3. Save structures ────────────────────────────────────────────
os.makedirs("interpolated_structures", exist_ok=True)

for i, struct in enumerate(images):
    lam = i / (len(images) - 1)
    fname = f"interpolated_structures/image_{i:02d}_lambda_{lam:.3f}.cif"
    struct.to(filename=fname)
    print(f"  Image {i:2d}: lambda = {lam:.3f}, "
          f"V = {struct.volume:.3f} A^3, "
          f"c/a = {struct.lattice.c / struct.lattice.a:.4f}")

# Also save as JSON for easy loading later
struct_data = []
for i, struct in enumerate(images):
    struct_data.append({
        "index": i,
        "lambda": i / (len(images) - 1),
        "structure": struct.as_dict(),
    })

with open("interpolated_structures/all_images.json", "w") as f:
    json.dump(struct_data, f, indent=2)

print(f"\nAll structures saved to interpolated_structures/")
print(f"Total images (including endpoints): {len(images)}")
```

### Step 4: Run SCF for Each Image

```python
#!/usr/bin/env python3
"""
Run SCF calculation for each interpolated structure.
Must converge to insulating ground state (no metallic states).
"""
import os
import subprocess
import json
import numpy as np
from pymatgen.core import Structure

PSEUDO_DIR = os.path.abspath("./pseudo")
BASE_OUTDIR = os.path.abspath("./tmp_berry")

# ── 1. Load interpolated structures ───────────────────────────────
with open("interpolated_structures/all_images.json", "r") as f:
    struct_data = json.load(f)

pseudos = {
    "Ba": ("137.327", "Ba.pbe-spn-kjpaw_psl.1.0.0.UPF"),
    "Ti": ("47.867",  "Ti.pbe-spn-kjpaw_psl.1.0.0.UPF"),
    "O":  ("15.999",  "O.pbe-n-kjpaw_psl.1.0.0.UPF"),
}


def write_scf_input(struct, image_idx, outdir):
    """Write a QE SCF input file for a given structure."""
    os.makedirs(outdir, exist_ok=True)

    lattice = struct.lattice
    cell_params = (
        f"  {lattice.matrix[0][0]:.10f}  {lattice.matrix[0][1]:.10f}  {lattice.matrix[0][2]:.10f}\n"
        f"  {lattice.matrix[1][0]:.10f}  {lattice.matrix[1][1]:.10f}  {lattice.matrix[1][2]:.10f}\n"
        f"  {lattice.matrix[2][0]:.10f}  {lattice.matrix[2][1]:.10f}  {lattice.matrix[2][2]:.10f}"
    )

    # Get unique species and their counts
    species_list = [str(s) for s in struct.species]
    unique_species = list(dict.fromkeys(species_list))  # preserves order

    atomic_species_block = ""
    for sp in unique_species:
        mass, pp = pseudos[sp]
        atomic_species_block += f"  {sp}  {mass}  {pp}\n"

    atomic_positions_block = ""
    for site in struct:
        sp = str(site.specie)
        fc = site.frac_coords
        atomic_positions_block += f"  {sp}  {fc[0]:.10f}  {fc[1]:.10f}  {fc[2]:.10f}\n"

    inp = f"""&CONTROL
    calculation = 'scf'
    prefix      = 'image_{image_idx:02d}'
    outdir      = '{outdir}'
    pseudo_dir  = '{PSEUDO_DIR}'
    tprnfor     = .true.
    tstress     = .true.
    verbosity   = 'high'
/

&SYSTEM
    ibrav       = 0
    nat         = {struct.num_sites}
    ntyp        = {len(unique_species)}
    ecutwfc     = 50.0
    ecutrho     = 400.0
    occupations = 'fixed'
    nosym       = .true.
/

&ELECTRONS
    conv_thr    = 1.0d-10
    mixing_beta = 0.5
/

CELL_PARAMETERS (angstrom)
{cell_params}

ATOMIC_SPECIES
{atomic_species_block}
ATOMIC_POSITIONS (crystal)
{atomic_positions_block}
K_POINTS (automatic)
  6 6 8  1 1 1
"""
    return inp


# ── 2. Write and run SCF for each image ───────────────────────────
for entry in struct_data:
    idx = entry["index"]
    lam = entry["lambda"]
    struct = Structure.from_dict(entry["structure"])
    outdir = os.path.join(BASE_OUTDIR, f"image_{idx:02d}")

    inp = write_scf_input(struct, idx, outdir)
    infile = f"image_{idx:02d}_scf.in"
    outfile = f"image_{idx:02d}_scf.out"

    with open(infile, "w") as f:
        f.write(inp)

    print(f"Running SCF for image {idx:02d} (lambda = {lam:.3f}) ...")
    result = subprocess.run(
        ["mpirun", "-np", "4", "pw.x", "-in", infile],
        capture_output=True, text=True, timeout=1800
    )
    with open(outfile, "w") as f:
        f.write(result.stdout)

    if result.returncode != 0:
        print(f"  ERROR: SCF failed for image {idx:02d}")
        print(result.stderr[-300:] if result.stderr else "")
    else:
        # Check convergence
        if "convergence has been achieved" in result.stdout:
            print(f"  OK: SCF converged for image {idx:02d}")
        else:
            print(f"  WARNING: SCF may not have converged for image {idx:02d}")

        # Check for metallic states
        if "the Fermi energy is" in result.stdout:
            print(f"  WARNING: metallic state detected at image {idx:02d}!")
            print(f"  Berry phase is only defined for insulators.")
            print(f"  Consider using 'occupations = fixed' or adjusting smearing.")

print("\nAll SCF calculations complete.")
```

**Important notes on SCF setup for Berry phase:**
- Use `occupations = 'fixed'` (no smearing) -- Berry phase requires an insulator with a gap at every k-point. If any image along the path becomes metallic, the Berry phase is undefined.
- Use `nosym = .true.` -- symmetry must be disabled for Berry phase calculations.
- Use a k-grid that is compatible with `nppstr` (the number of k-points along the Berry phase string direction). The k-grid dimension along the polarization direction must be divisible by `nppstr`.
- The k-grid along the polarization direction should be **dense** (at least 8, ideally 10+). For BaTiO3 with polarization along z: use e.g. `6 6 8` or `6 6 10`.

### Step 5: Berry Phase Calculation for Each Image

```python
#!/usr/bin/env python3
"""
Run Berry phase calculation for each image using pw.x with lberry=.true.
QE computes the electronic polarization via the Berry phase approach.

CRITICAL: This must be run AFTER the SCF converges for each image.
The Berry phase calculation reads the SCF wavefunctions from outdir.
"""
import os
import subprocess
import json
from pymatgen.core import Structure

PSEUDO_DIR = os.path.abspath("./pseudo")
BASE_OUTDIR = os.path.abspath("./tmp_berry")

# Load structures
with open("interpolated_structures/all_images.json", "r") as f:
    struct_data = json.load(f)

pseudos = {
    "Ba": ("137.327", "Ba.pbe-spn-kjpaw_psl.1.0.0.UPF"),
    "Ti": ("47.867",  "Ti.pbe-spn-kjpaw_psl.1.0.0.UPF"),
    "O":  ("15.999",  "O.pbe-n-kjpaw_psl.1.0.0.UPF"),
}

# The polarization direction to compute.
# gdir = 1 -> x, gdir = 2 -> y, gdir = 3 -> z
# For tetragonal BaTiO3, the polar axis is along z, so gdir = 3.
# Compute all three directions to get the full polarization vector.
GDIRS = [1, 2, 3]  # compute along x, y, and z

# nppstr = number of k-points along the Berry phase string.
# Must divide the k-grid along the corresponding direction.
# For k-grid 6 6 8 with gdir=3, nppstr=8 uses all k-points along z.
NPPSTR_MAP = {1: 6, 2: 6, 3: 8}  # nppstr for each gdir


def write_berry_input(struct, image_idx, gdir, nppstr, outdir):
    """Write QE input with lberry=.true. for Berry phase calculation."""
    lattice = struct.lattice
    cell_params = (
        f"  {lattice.matrix[0][0]:.10f}  {lattice.matrix[0][1]:.10f}  {lattice.matrix[0][2]:.10f}\n"
        f"  {lattice.matrix[1][0]:.10f}  {lattice.matrix[1][1]:.10f}  {lattice.matrix[1][2]:.10f}\n"
        f"  {lattice.matrix[2][0]:.10f}  {lattice.matrix[2][1]:.10f}  {lattice.matrix[2][2]:.10f}"
    )

    species_list = [str(s) for s in struct.species]
    unique_species = list(dict.fromkeys(species_list))

    atomic_species_block = ""
    for sp in unique_species:
        mass, pp = pseudos[sp]
        atomic_species_block += f"  {sp}  {mass}  {pp}\n"

    atomic_positions_block = ""
    for site in struct:
        sp = str(site.specie)
        fc = site.frac_coords
        atomic_positions_block += f"  {sp}  {fc[0]:.10f}  {fc[1]:.10f}  {fc[2]:.10f}\n"

    # NOTE: lberry=.true. replaces the normal SCF. It reads the converged
    # charge density from outdir and computes the Berry phase.
    # The K_POINTS block is IGNORED when lberry=.true. -- the k-point
    # sampling is determined by the perpendicular grid and nppstr.
    inp = f"""&CONTROL
    calculation = 'scf'
    prefix      = 'image_{image_idx:02d}'
    outdir      = '{outdir}'
    pseudo_dir  = '{PSEUDO_DIR}'
    verbosity   = 'high'
/

&SYSTEM
    ibrav       = 0
    nat         = {struct.num_sites}
    ntyp        = {len(unique_species)}
    ecutwfc     = 50.0
    ecutrho     = 400.0
    occupations = 'fixed'
    nosym       = .true.
    lberry      = .true.
    gdir        = {gdir}
    nppstr      = {nppstr}
/

&ELECTRONS
    conv_thr    = 1.0d-10
    mixing_beta = 0.5
/

CELL_PARAMETERS (angstrom)
{cell_params}

ATOMIC_SPECIES
{atomic_species_block}
ATOMIC_POSITIONS (crystal)
{atomic_positions_block}
K_POINTS (automatic)
  6 6 8  1 1 1
"""
    return inp


# ── Run Berry phase for each image and each direction ─────────────
for entry in struct_data:
    idx = entry["index"]
    lam = entry["lambda"]
    struct = Structure.from_dict(entry["structure"])
    outdir = os.path.join(BASE_OUTDIR, f"image_{idx:02d}")

    for gdir in GDIRS:
        nppstr = NPPSTR_MAP[gdir]
        inp = write_berry_input(struct, idx, gdir, nppstr, outdir)

        infile = f"image_{idx:02d}_berry_gdir{gdir}.in"
        outfile = f"image_{idx:02d}_berry_gdir{gdir}.out"

        with open(infile, "w") as f:
            f.write(inp)

        print(f"Running Berry phase: image {idx:02d}, gdir={gdir}, "
              f"nppstr={nppstr} ...")
        result = subprocess.run(
            ["mpirun", "-np", "4", "pw.x", "-in", infile],
            capture_output=True, text=True, timeout=1800
        )
        with open(outfile, "w") as f:
            f.write(result.stdout)

        if result.returncode != 0:
            print(f"  ERROR for image {idx:02d}, gdir={gdir}")
        else:
            # Quick check for Berry phase output
            if "POLARIZATION" in result.stdout or "Berry phase" in result.stdout:
                print(f"  OK: Berry phase computed for image {idx:02d}, gdir={gdir}")
            else:
                print(f"  WARNING: Berry phase output not found for "
                      f"image {idx:02d}, gdir={gdir}")

print("\nAll Berry phase calculations complete.")
```

### Step 6: Parse Berry Phase Output and Compute Polarization

```python
#!/usr/bin/env python3
"""
Parse Berry phase output from QE pw.x and compute the polarization vector.

QE outputs the electronic polarization as a Berry phase in units of (2*pi).
The ionic polarization must be computed separately from the atomic positions
and nominal ionic charges (valence charges from pseudopotentials).

Total polarization = P_electronic + P_ionic (mod polarization quantum)
"""
import re
import os
import json
import numpy as np
from pymatgen.core import Structure

# ── Physical constants ────────────────────────────────────────────
E_CHARGE = 1.602176634e-19      # C
BOHR_TO_M = 5.29177210903e-11   # m
ANGSTROM_TO_M = 1e-10           # m
HARTREE_TO_J = 4.3597447222071e-18  # J
# In QE, polarization is reported in (e * Bohr) / Omega_Bohr^3
# or equivalently C/m^2. We parse the QE output directly.


def parse_berry_phase_output(filename):
    """
    Parse QE Berry phase output.

    Returns:
        dict with keys:
            'phase_el': electronic Berry phase in units of 2*pi
            'P_el_Cm2': electronic polarization in C/m^2 (as reported by QE)
            'P_ion_Cm2': ionic polarization in C/m^2 (as reported by QE)
            'P_tot_Cm2': total polarization in C/m^2
            'P_quantum_Cm2': polarization quantum in C/m^2
    """
    result = {
        "phase_el": None,
        "P_el_Cm2": None,
        "P_ion_Cm2": None,
        "P_tot_Cm2": None,
        "P_quantum_Cm2": None,
    }

    if not os.path.exists(filename):
        print(f"  File not found: {filename}")
        return result

    with open(filename, "r") as f:
        text = f.read()

    # Parse the electronic Berry phase (in units of 2*pi)
    # QE outputs: "Electronic Phase:   X.XXXXXX"
    m = re.search(r"Electronic Phase:\s+([-\d.]+)", text)
    if m:
        result["phase_el"] = float(m.group(1))

    # Parse electronic polarization
    # QE outputs: "el. pol. in the direction of ... = X.XXXXXX (mod Y.YYYYYY) (e/Omega).bohr"
    # Or: "P_ele  =  X.XXXXX  (mod  Y.YYYYY)  C/m^2"
    m = re.search(r"P_ele\s*=\s*([-\d.]+)\s*\(mod\s+([-\d.]+)\)", text)
    if m:
        result["P_el_Cm2"] = float(m.group(1))
        result["P_quantum_Cm2"] = float(m.group(2))

    # Alternative format in some QE versions
    if result["P_el_Cm2"] is None:
        m = re.search(
            r"The\s+electronic\s+contribution\s+to\s+the\s+polarization.*?"
            r"P\s*=\s*([-\d.]+)\s+\(mod\s+([-\d.]+)\)\s+C/m\^2",
            text, re.DOTALL
        )
        if m:
            result["P_el_Cm2"] = float(m.group(1))
            result["P_quantum_Cm2"] = float(m.group(2))

    # Parse ionic polarization
    m = re.search(r"P_ion\s*=\s*([-\d.]+)", text)
    if m:
        result["P_ion_Cm2"] = float(m.group(1))

    # Alternative ionic format
    if result["P_ion_Cm2"] is None:
        m = re.search(
            r"The\s+ionic\s+contribution.*?P\s*=\s*([-\d.]+)\s+C/m\^2",
            text, re.DOTALL
        )
        if m:
            result["P_ion_Cm2"] = float(m.group(1))

    # Parse total polarization
    m = re.search(r"P_tot\s*=\s*([-\d.]+)", text)
    if m:
        result["P_tot_Cm2"] = float(m.group(1))

    # Compute total if not directly parsed
    if (result["P_tot_Cm2"] is None and
            result["P_el_Cm2"] is not None and
            result["P_ion_Cm2"] is not None):
        result["P_tot_Cm2"] = result["P_el_Cm2"] + result["P_ion_Cm2"]

    return result


def compute_ionic_polarization(struct, zval_dict, gdir):
    """
    Compute the ionic contribution to polarization manually.

    Args:
        struct: pymatgen Structure
        zval_dict: dict mapping element symbol to valence charge (from PP)
        gdir: direction index (0=x, 1=y, 2=z)

    Returns:
        P_ion in C/m^2
    """
    volume_m3 = struct.volume * ANGSTROM_TO_M**3

    # Compute ionic dipole along gdir direction
    # P_ion = (e / Omega) * sum_i Z_i * r_i (in Cartesian)
    dipole = 0.0
    for site in struct:
        sp = str(site.specie)
        z_val = zval_dict[sp]
        # Cartesian coordinate in meters
        r_cart = site.coords[gdir] * ANGSTROM_TO_M
        dipole += z_val * r_cart

    P_ion = E_CHARGE * dipole / volume_m3  # C/m^2
    return P_ion


# ── Parse all images ──────────────────────────────────────────────
with open("interpolated_structures/all_images.json", "r") as f:
    struct_data = json.load(f)

# Pseudopotential valence charges (check your PP files!)
# For PAW PPs: Ba has 10 valence electrons, Ti has 12, O has 6
zval_dict = {"Ba": 10.0, "Ti": 12.0, "O": 6.0}

GDIRS = [1, 2, 3]
DIR_LABELS = {1: "x", 2: "y", 3: "z"}

all_results = []

for entry in struct_data:
    idx = entry["index"]
    lam = entry["lambda"]
    struct = Structure.from_dict(entry["structure"])

    image_result = {
        "index": idx,
        "lambda": lam,
        "volume_A3": struct.volume,
        "P_el": {},
        "P_ion": {},
        "P_tot": {},
        "P_quantum": {},
    }

    for gdir in GDIRS:
        outfile = f"image_{idx:02d}_berry_gdir{gdir}.out"
        parsed = parse_berry_phase_output(outfile)
        d = DIR_LABELS[gdir]

        image_result["P_el"][d] = parsed["P_el_Cm2"]
        image_result["P_ion"][d] = parsed["P_ion_Cm2"]
        image_result["P_tot"][d] = parsed["P_tot_Cm2"]
        image_result["P_quantum"][d] = parsed["P_quantum_Cm2"]

    all_results.append(image_result)

# ── Print summary ─────────────────────────────────────────────────
print("=" * 75)
print(f"{'Image':>5} {'lambda':>7} {'P_el_z':>12} {'P_ion_z':>12} "
      f"{'P_tot_z':>12} {'P_quant_z':>12}")
print(f"{'':>5} {'':>7} {'(C/m2)':>12} {'(C/m2)':>12} "
      f"{'(C/m2)':>12} {'(C/m2)':>12}")
print("-" * 75)

for r in all_results:
    P_el = r["P_el"].get("z", None)
    P_ion = r["P_ion"].get("z", None)
    P_tot = r["P_tot"].get("z", None)
    P_q = r["P_quantum"].get("z", None)

    def fmt(v):
        return f"{v:12.6f}" if v is not None else f"{'N/A':>12}"

    print(f"{r['index']:5d} {r['lambda']:7.3f} {fmt(P_el)} {fmt(P_ion)} "
          f"{fmt(P_tot)} {fmt(P_q)}")

# Save to JSON
with open("berry_phase_results.json", "w") as f:
    json.dump(all_results, f, indent=2)
print(f"\nResults saved to berry_phase_results.json")
```

### Step 7: Branch Selection and Spontaneous Polarization

```python
#!/usr/bin/env python3
"""
Resolve polarization quantum ambiguity and compute spontaneous polarization.

The Berry phase polarization is defined modulo the polarization quantum
  Q = e * |R_i| / Omega
where R_i is a lattice vector and Omega is the cell volume. At each image,
the reported P_tot could be shifted by any integer multiple of Q. The
correct branch is the one where P varies continuously from the centrosymmetric
reference (where P = 0) to the polar phase.

Algorithm:
1. Start from image 0 (nonpolar, P = 0 by symmetry).
2. For each subsequent image, pick the branch of P that is closest to
   the previous image's P value, i.e., add/subtract multiples of Q
   to minimize |P_i - P_{i-1}|.
3. The spontaneous polarization is P_s = P(lambda=1) - P(lambda=0).
"""
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def resolve_branch(P_values, P_quantum):
    """
    Resolve polarization branch by ensuring continuity.

    Args:
        P_values: list of raw polarization values (C/m^2)
        P_quantum: polarization quantum (C/m^2)

    Returns:
        list of branch-resolved polarization values
    """
    if P_quantum is None or P_quantum == 0 or any(p is None for p in P_values):
        return P_values

    resolved = [P_values[0]]

    for i in range(1, len(P_values)):
        P_raw = P_values[i]
        P_prev = resolved[-1]

        # Find the branch closest to previous value
        # P_raw + n * P_quantum should be closest to P_prev
        n = round((P_prev - P_raw) / P_quantum)
        P_corrected = P_raw + n * P_quantum
        resolved.append(P_corrected)

    return resolved


# ── Load parsed results ───────────────────────────────────────────
with open("berry_phase_results.json", "r") as f:
    all_results = json.load(f)

lambdas = [r["lambda"] for r in all_results]
directions = ["x", "y", "z"]

# ── Resolve branches for each direction ───────────────────────────
P_resolved = {}
P_spontaneous = {}

for d in directions:
    P_raw = [r["P_tot"].get(d) for r in all_results]
    P_q_values = [r["P_quantum"].get(d) for r in all_results]

    # Use the polarization quantum from the first valid image
    P_q = None
    for pq in P_q_values:
        if pq is not None and pq > 0:
            P_q = pq
            break

    if all(p is not None for p in P_raw) and P_q is not None:
        P_branch = resolve_branch(P_raw, P_q)
        P_resolved[d] = P_branch
        P_spontaneous[d] = P_branch[-1] - P_branch[0]
        print(f"Direction {d}: P_s = {P_spontaneous[d]:.6f} C/m^2  "
              f"(quantum = {P_q:.6f} C/m^2)")
    else:
        P_resolved[d] = P_raw
        P_spontaneous[d] = None
        print(f"Direction {d}: Could not resolve (missing data)")

# ── Compute |P_s| ─────────────────────────────────────────────────
Ps_vec = [P_spontaneous.get(d, 0.0) or 0.0 for d in directions]
Ps_magnitude = np.linalg.norm(Ps_vec)
print(f"\nSpontaneous polarization vector: "
      f"({Ps_vec[0]:.6f}, {Ps_vec[1]:.6f}, {Ps_vec[2]:.6f}) C/m^2")
print(f"|P_s| = {Ps_magnitude:.6f} C/m^2")
print(f"|P_s| = {Ps_magnitude * 100:.4f} uC/cm^2")

# ── Plot polarization vs. lambda ──────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(15, 5))
colors = {"x": "steelblue", "y": "darkorange", "z": "forestgreen"}

for ax, d in zip(axes, directions):
    P = P_resolved.get(d)
    if P is not None and all(p is not None for p in P):
        ax.plot(lambdas, P, "o-", color=colors[d], linewidth=2, markersize=8)
        ax.set_xlabel("$\\lambda$ (interpolation parameter)", fontsize=13)
        ax.set_ylabel(f"$P_{d}$ (C/m$^2$)", fontsize=13)
        ax.set_title(f"Polarization along {d}", fontsize=14)
        ax.grid(True, alpha=0.3)

        # Show quantum as dashed lines
        P_q_val = None
        for r in all_results:
            pq = r["P_quantum"].get(d)
            if pq is not None:
                P_q_val = pq
                break
        if P_q_val and P_q_val > 0:
            for n in range(-3, 4):
                ax.axhline(y=P[0] + n * P_q_val, color="red",
                           linestyle="--", alpha=0.2)
    else:
        ax.text(0.5, 0.5, "No data", ha="center", va="center",
                transform=ax.transAxes, fontsize=14)
        ax.set_title(f"Polarization along {d}", fontsize=14)

plt.suptitle("Berry Phase Polarization Along Switching Path", fontsize=15, y=1.02)
plt.tight_layout()
plt.savefig("polarization_vs_lambda.png", dpi=150, bbox_inches="tight")
print("\nPlot saved: polarization_vs_lambda.png")

# ── Save final results ────────────────────────────────────────────
final_output = {
    "P_spontaneous_Cm2": P_spontaneous,
    "P_spontaneous_magnitude_Cm2": Ps_magnitude,
    "P_spontaneous_magnitude_uC_cm2": Ps_magnitude * 100,
    "P_resolved": {d: P_resolved.get(d) for d in directions},
    "lambdas": lambdas,
}
with open("spontaneous_polarization.json", "w") as f:
    json.dump(final_output, f, indent=2)
print("Final results saved: spontaneous_polarization.json")
```

### Alternative: All-in-One Script (Minimal Example for PbTiO3)

```python
#!/usr/bin/env python3
"""
Minimal all-in-one Berry phase polarization workflow for PbTiO3.
Generates input files, runs QE, parses output, and plots results.

PbTiO3: Polar phase P4mm, nonpolar phase Pm-3m.
Experimental P_s ~ 0.75 C/m^2.
"""
import os
import subprocess
import re
import json
import numpy as np
from pymatgen.core import Structure, Lattice
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ── Configuration ─────────────────────────────────────────────────
PSEUDO_DIR = os.path.abspath("./pseudo")
WORK_DIR = os.path.abspath("./pbtio3_berry")
os.makedirs(WORK_DIR, exist_ok=True)

PSEUDOS = {
    "Pb": ("207.2", "Pb.pbe-dn-kjpaw_psl.1.0.0.UPF"),
    "Ti": ("47.867", "Ti.pbe-spn-kjpaw_psl.1.0.0.UPF"),
    "O":  ("15.999", "O.pbe-n-kjpaw_psl.1.0.0.UPF"),
}
ZVAL = {"Pb": 14.0, "Ti": 12.0, "O": 6.0}

NIMAGES = 6
KGRID = (6, 6, 8)  # dense along z (polarization direction)
NPPSTR = 8          # for gdir=3


def make_structure(a, c, dz_Ti, dz_O1, dz_O2):
    """Build PbTiO3 structure with given distortions."""
    return Structure(
        Lattice([[a, 0, 0], [0, a, 0], [0, 0, c]]),
        species=["Pb", "Ti", "O", "O", "O"],
        coords=[
            [0.0, 0.0, 0.0],
            [0.5, 0.5, 0.5 + dz_Ti],
            [0.5, 0.5, 0.0 + dz_O1],
            [0.5, 0.0, 0.5 + dz_O2],
            [0.0, 0.5, 0.5 + dz_O2],
        ],
        coords_are_cartesian=False,
    )


# Nonpolar (cubic Pm-3m): a = c, no distortions
nonpolar = make_structure(a=3.97, c=3.97, dz_Ti=0.0, dz_O1=0.0, dz_O2=0.0)

# Polar (tetragonal P4mm): c/a ~ 1.065, Ti and O displaced
polar = make_structure(a=3.90, c=4.15, dz_Ti=0.04, dz_O1=-0.11, dz_O2=-0.04)

# Interpolate
images = nonpolar.interpolate(
    polar, nimages=NIMAGES + 1,
    interpolate_lattices=True, autosort_tol=0.0
)

print(f"Generated {len(images)} images")


def write_and_run(struct, prefix, outdir, lberry=False, gdir=3, nppstr=8):
    """Write QE input and run pw.x."""
    os.makedirs(outdir, exist_ok=True)
    lat = struct.lattice.matrix

    species_set = list(dict.fromkeys([str(s) for s in struct.species]))
    sp_block = "".join(
        f"  {sp} {PSEUDOS[sp][0]} {PSEUDOS[sp][1]}\n" for sp in species_set
    )
    pos_block = "".join(
        f"  {str(s.specie)} {s.frac_coords[0]:.10f} "
        f"{s.frac_coords[1]:.10f} {s.frac_coords[2]:.10f}\n"
        for s in struct
    )

    berry_block = ""
    if lberry:
        berry_block = f"""    lberry      = .true.
    gdir        = {gdir}
    nppstr      = {nppstr}"""

    inp = f"""&CONTROL
    calculation = 'scf'
    prefix      = '{prefix}'
    outdir      = '{outdir}'
    pseudo_dir  = '{PSEUDO_DIR}'
    verbosity   = 'high'
/
&SYSTEM
    ibrav       = 0
    nat         = {struct.num_sites}
    ntyp        = {len(species_set)}
    ecutwfc     = 50.0
    ecutrho     = 400.0
    occupations = 'fixed'
    nosym       = .true.
{berry_block}
/
&ELECTRONS
    conv_thr    = 1.0d-10
    mixing_beta = 0.4
/
CELL_PARAMETERS (angstrom)
  {lat[0][0]:.10f} {lat[0][1]:.10f} {lat[0][2]:.10f}
  {lat[1][0]:.10f} {lat[1][1]:.10f} {lat[1][2]:.10f}
  {lat[2][0]:.10f} {lat[2][1]:.10f} {lat[2][2]:.10f}
ATOMIC_SPECIES
{sp_block}ATOMIC_POSITIONS (crystal)
{pos_block}K_POINTS (automatic)
  {KGRID[0]} {KGRID[1]} {KGRID[2]}  1 1 1
"""
    tag = "berry" if lberry else "scf"
    infile = os.path.join(WORK_DIR, f"{prefix}_{tag}.in")
    outfile = os.path.join(WORK_DIR, f"{prefix}_{tag}.out")

    with open(infile, "w") as f:
        f.write(inp)

    result = subprocess.run(
        ["mpirun", "-np", "4", "pw.x", "-in", infile],
        capture_output=True, text=True, timeout=1800
    )
    with open(outfile, "w") as f:
        f.write(result.stdout)

    return outfile, result.returncode == 0


# ── Run SCF then Berry phase for each image ───────────────────────
for i, struct in enumerate(images):
    lam = i / (len(images) - 1)
    prefix = f"img{i:02d}"
    outdir = os.path.join(WORK_DIR, f"tmp_{prefix}")

    # Step 1: SCF
    scf_out, ok = write_and_run(struct, prefix, outdir, lberry=False)
    status = "OK" if ok else "FAIL"
    print(f"Image {i:2d} (lambda={lam:.3f}): SCF {status}")

    # Step 2: Berry phase along z
    berry_out, ok = write_and_run(
        struct, prefix, outdir, lberry=True, gdir=3, nppstr=NPPSTR
    )
    status = "OK" if ok else "FAIL"
    print(f"Image {i:2d} (lambda={lam:.3f}): Berry {status}")

print("\nAll calculations complete. Parse the Berry phase outputs next.")
```

---

## Key Parameters

| Parameter | Typical Value | Notes |
|---|---|---|
| `lberry` | `.true.` | Activates Berry phase calculation in pw.x |
| `gdir` | 1, 2, or 3 | Direction of Berry phase string (1=x, 2=y, 3=z) |
| `nppstr` | 6--12 | Number of k-points along the Berry phase string. Must divide the k-grid dimension along gdir. Larger = more accurate. |
| `occupations` | `'fixed'` | **Required** -- Berry phase is only defined for insulators. Never use smearing. |
| `nosym` | `.true.` | **Required** -- symmetry must be disabled for Berry phase |
| `ecutwfc` | 50--80 Ry | Converge carefully; polarization is sensitive to basis set |
| `conv_thr` | 1.0d-10 | Tight SCF convergence needed for accurate Berry phase |
| k-grid perpendicular | 6x6 or denser | The k-grid perpendicular to gdir; denser = better |
| k-grid along gdir | 8--12 | Must equal or be divisible by nppstr |
| `nimages` | 6--10 | Number of interpolation images (not counting endpoints) |

### Choosing nppstr

The parameter `nppstr` controls the density of the k-point string along the Berry phase direction. Guidelines:

- `nppstr` = k-grid dimension along gdir is the standard choice.
- Minimum recommended: `nppstr = 6`. For accurate results: `nppstr >= 8`.
- Test convergence: run with nppstr = 6, 8, 10, 12 and check that the polarization does not change by more than 0.01 C/m^2.

### Choosing nimages

More images = smoother polarization curve = easier branch resolution. Use at least 6 images. If the polarization jumps discontinuously between adjacent images, increase nimages.

## Interpreting Results

### QE Berry Phase Output

QE prints the following in the output file when `lberry=.true.`:

```
VALUES OF POLARIZATION
   ...
   The electronic contribution to the polarization
   P_ele  =  -0.012345  (mod   1.234567)  C/m^2

   The ionic contribution to the polarization
   P_ion  =   0.567890  C/m^2

   The total polarization
   P_tot  =   0.555545  C/m^2
```

- `P_ele`: electronic Berry phase contribution (mod polarization quantum).
- `P_ion`: ionic contribution (classical point charges at atomic positions).
- `P_tot = P_ele + P_ion`: total, still defined mod the quantum.
- The `(mod X.XXXXXX)` value is the **polarization quantum** for that direction.

### Polarization Quantum

The polarization quantum along direction i is:

```
Q_i = e * |a_i| / Omega
```

where `a_i` is the lattice vector along direction i and Omega is the cell volume. For BaTiO3 with a ~ 4.0 A and Omega ~ 64 A^3:

```
Q_z = (1.602e-19 * 4.0e-10) / (64e-30) = 1.00 C/m^2
```

The *physical* polarization must be chosen from the set `P_tot + n * Q` (n integer) such that the path from the centrosymmetric reference is continuous.

### Expected Values for Common Ferroelectrics

| Material | Space Group (polar) | P_s (C/m^2) | P_s (uC/cm^2) |
|---|---|---|---|
| BaTiO3 | P4mm | ~0.26 | ~26 |
| PbTiO3 | P4mm | ~0.75 | ~75 |
| LiNbO3 | R3c | ~0.71 | ~71 |
| BiFeO3 | R3c | ~1.00 | ~100 |
| KNbO3 | Amm2 | ~0.37 | ~37 |

### Checking Your Results

1. **Continuity**: Plot P vs. lambda. The curve should be smooth and monotonic (or at least continuous). Discontinuities indicate branch errors.
2. **Endpoints**: P(lambda=0) for the centrosymmetric phase should be zero (or very close, within numerical noise ~0.001 C/m^2). If not, check that the reference is truly centrosymmetric.
3. **Symmetry**: For a tetragonal ferroelectric with polar axis along z, P_x and P_y should be zero (within noise). Only P_z should be nonzero.
4. **Magnitude**: Compare with experimental or literature DFT values. PBE typically overestimates polarization by 5--15%.

## Common Issues

| Problem | Solution |
|---|---|
| **Polarization quantum ambiguity** | Use interpolation path with enough images (>= 6). Resolve branches by continuity from the centrosymmetric endpoint. If jumps persist, increase nimages. |
| **Metallic states at intermediate lambda** | Some interpolated structures may close the gap. Use `occupations = 'fixed'` and ensure the gap stays open. If the gap closes, try a different interpolation path, use a Hubbard U correction, or use a hybrid functional. |
| **P(lambda=0) is not zero** | Verify the reference structure is truly centrosymmetric. Check with `spglib` that the space group has an inversion center. Small numerical noise (< 0.005 C/m^2) is acceptable. |
| **Large discontinuities in P vs. lambda** | Branch resolution failed. Try: (1) more images, (2) check that all SCFs converged, (3) increase nppstr, (4) check for metallic states. |
| **P_ele has wrong sign** | This is normal -- the sign depends on the branch. The physical quantity is P_s = P(polar) - P(nonpolar), which should have a definite sign matching the direction of polar distortion. |
| **SCF not converging for Berry phase** | The Berry phase run reads converged wavefunctions. If the SCF step failed, fix that first (reduce mixing_beta, check structure). The Berry phase step itself does not iterate. |
| **Different P from different nppstr** | Increase nppstr until P converges (changes < 0.01 C/m^2). Typical convergence at nppstr >= 8. |
| **Inconsistent P_ionic between QE and manual calculation** | QE uses the pseudopotential valence charges (zval). Verify your zval_dict matches the PP files. Use `grep 'z_valence' *.UPF` to check. |
| **Structures have different numbers of atoms** | Polar and nonpolar structures must have the same atoms in the same order. The interpolation requires a one-to-one mapping. Use `autosort_tol=0.0` to prevent reordering. |
| **ibrav != 0 causes issues** | Always use `ibrav = 0` with explicit CELL_PARAMETERS for Berry phase calculations. Non-zero ibrav with lberry can cause coordinate transformation issues in some QE versions. |
