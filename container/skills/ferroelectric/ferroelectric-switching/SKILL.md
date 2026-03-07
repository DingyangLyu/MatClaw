# Ferroelectric Switching Pathway and Double-Well Energy Profile

## When to Use

- You need to compute the energy barrier for ferroelectric switching between two equivalent polar states (e.g., +P and -P) through a centrosymmetric transition state.
- You want to map the double-well energy profile E(lambda) along the switching coordinate to characterize the depth and shape of the ferroelectric potential well.
- You need the switching barrier height to estimate coercive field, switching kinetics, or thermal stability of polarization.
- You want to track the evolution of polarization along the switching path (P vs. lambda) to understand the switching mechanism.
- You want to perform a quick screening of switching barriers using MACE-MP-0 before committing to expensive DFT NEB calculations.
- You need to assess whether a polar distortion is robust (deep double-well) or fragile (shallow well, easy to switch).

## Method Selection

| Method | Code | What You Get | Cost | Accuracy |
|---|---|---|---|---|
| Linear interpolation + single-point energies | QE pw.x | Approximate energy profile; misses relaxation of transverse modes | Low | Qualitative |
| Nudged Elastic Band (NEB) | QE neb.x | Minimum energy path (MEP) and accurate barrier | High | High |
| CI-NEB (climbing image) | QE neb.x | Accurate saddle point and barrier height | High | High |
| VASP NEB | VASP + VTST | MEP and barrier with PAW accuracy | High | High |
| ASE NEB + MACE-MP-0 | ASE + MACE | Fast approximate MEP for screening | Low | Screening quality |
| ASE NEB + DFT calculator | ASE + QE/VASP | MEP with DFT accuracy via ASE interface | High | High |
| Frozen-phonon double-well | QE pw.x | Energy vs. soft-mode amplitude; 1D energy profile | Low | Good for single-mode switching |

## Prerequisites

- Two equivalent polar structures related by symmetry (e.g., +P and -P states), and ideally the centrosymmetric (paraelectric) reference structure as the transition state candidate.
- For NEB: an initial guess for the switching path (interpolated images between +P and -P states).
- Quantum ESPRESSO 7.5 with `pw.x` and `neb.x` available on PATH.
- For MACE screening: `mace-torch` package with MACE-MP-0 pretrained model.
- Python packages: `pymatgen`, `ase`, `numpy`, `scipy`, `matplotlib`.
- Pseudopotential files for QE calculations.

## Background

### Ferroelectric double-well potential

In a ferroelectric material, the energy as a function of polar distortion amplitude (lambda) has a characteristic double-well shape:

```
E(lambda)
  ^
  |     *           *
  |    * *         * *
  |   *   *       *   *
  |  *     *     *     *
  | *       *   *       *
  |*         * *         *
  |           *           --> lambda
  -P    0 (centrosym)    +P
```

The barrier height Delta_E is the energy difference between the centrosymmetric transition state (lambda = 0) and the polar minimum (lambda = +/- 1). This barrier determines:

- **Coercive field**: E_c ~ Delta_E / (P_s * Omega), where P_s is the spontaneous polarization.
- **Thermal stability**: Polarization is stable when Delta_E >> k_B * T.
- **Switching kinetics**: The Arrhenius rate of thermally activated switching ~ exp(-Delta_E / k_B T).

### Switching coordinate

The switching coordinate lambda parameterizes the path from one polar state to the other:

- lambda = -1: Polar state with -P (mirror of +P)
- lambda = 0: Centrosymmetric reference (transition state)
- lambda = +1: Polar state with +P

For simple perovskites like BaTiO3, the switching path is approximately a linear interpolation of atomic coordinates between +P and -P. For more complex ferroelectrics (e.g., BiFeO3 with coupled polarization-tilt modes), the minimum energy path may deviate significantly from linear interpolation, requiring NEB.

### Nudged Elastic Band (NEB)

NEB finds the minimum energy path (MEP) between two endpoint configurations by optimizing a chain of images connected by spring forces. The climbing-image variant (CI-NEB) drives the highest-energy image to the exact saddle point, giving an accurate barrier height.

## Detailed Steps

### Overview

```
Step 0: Prepare +P and -P endpoint structures
Step 1: Quick screening with linear interpolation
Step 2: (Optional) ASE + MACE-MP-0 NEB for fast screening
Step 3: QE NEB for accurate barrier
Step 4: Track polarization along the switching path
Step 5: Analyze double-well profile and extract barrier
Step 6: (Optional) VASP NEB reference
```

### Step 0: Prepare Endpoint Structures

```python
#!/usr/bin/env python3
"""
Prepare +P and -P endpoint structures for ferroelectric switching.

For a simple perovskite ABO3 with polarization along z:
  +P state: Ti displaced +dz relative to centrosymmetric position
  -P state: Ti displaced -dz (mirror image through z -> -z)

The centrosymmetric structure is the midpoint (transition state candidate).
"""
import os
import json
import numpy as np
from pymatgen.core import Structure, Lattice

# ---- Example: Tetragonal BaTiO3 ----
# Relaxed tetragonal parameters
a = 3.994  # Angstrom
c = 4.034

# +P state: Ti and O displaced along z
polar_plus = Structure(
    Lattice([[a, 0, 0], [0, a, 0], [0, 0, c]]),
    species=["Ba", "Ti", "O", "O", "O"],
    coords=[
        [0.000, 0.000, 0.000],   # Ba at origin
        [0.500, 0.500, 0.520],   # Ti shifted +z
        [0.500, 0.500, -0.020],  # O1 shifted -z (apical)
        [0.500, 0.000, 0.520],   # O2 shifted +z (equatorial)
        [0.000, 0.500, 0.520],   # O3 shifted +z (equatorial)
    ],
    coords_are_cartesian=False,
)

# -P state: Mirror of +P through the centrosymmetric reference.
# Invert the z-displacements relative to centrosymmetric positions.
# Centrosymmetric positions: Ti at (0.5, 0.5, 0.5), O at (0.5,0.5,0), etc.
# dz(Ti) = +0.020 in +P --> -0.020 in -P
# dz(O1) = -0.020 in +P --> +0.020 in -P
# dz(O2,O3) = +0.020 in +P --> -0.020 in -P
polar_minus = Structure(
    Lattice([[a, 0, 0], [0, a, 0], [0, 0, c]]),
    species=["Ba", "Ti", "O", "O", "O"],
    coords=[
        [0.000, 0.000, 0.000],
        [0.500, 0.500, 0.480],   # Ti shifted -z
        [0.500, 0.500, 0.020],   # O1 shifted +z
        [0.500, 0.000, 0.480],   # O2 shifted -z
        [0.000, 0.500, 0.480],   # O3 shifted -z
    ],
    coords_are_cartesian=False,
)

# Centrosymmetric reference (cubic BaTiO3)
centro = Structure(
    Lattice.cubic(a),  # Use cubic for the reference
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

# Save structures
os.makedirs("switching_structures", exist_ok=True)
polar_plus.to(filename="switching_structures/polar_plus.cif")
polar_minus.to(filename="switching_structures/polar_minus.cif")
centro.to(filename="switching_structures/centrosymmetric.cif")

# Also save as JSON
struct_data = {
    "polar_plus": polar_plus.as_dict(),
    "polar_minus": polar_minus.as_dict(),
    "centrosymmetric": centro.as_dict(),
}
with open("switching_structures/endpoints.json", "w") as f:
    json.dump(struct_data, f, indent=2)

print("Endpoint structures saved to switching_structures/")
print(f"  +P state: polar_plus.cif")
print(f"  -P state: polar_minus.cif")
print(f"  Centrosymmetric: centrosymmetric.cif")

# Print displacements
print("\nAtomic displacements from centrosymmetric reference:")
for i, (s_plus, s_centro) in enumerate(zip(polar_plus, centro)):
    d = polar_plus.frac_coords[i] - centro.frac_coords[i]
    d = d - np.round(d)  # wrap
    d_cart = d @ polar_plus.lattice.matrix
    print(f"  {str(s_plus.specie):3s}: dz = {d_cart[2]:+.4f} A "
          f"(frac = {d[2]:+.4f})")
```

### Step 1: Linear Interpolation Energy Profile

```python
#!/usr/bin/env python3
"""
Compute energy along the linear interpolation path from -P to +P
through the centrosymmetric state.

This is a quick approximation: energies at linearly interpolated
structures without relaxation of transverse modes. The true MEP
(from NEB) may have a lower barrier.

The path is parameterized as lambda in [-1, +1]:
  lambda = -1 : -P state
  lambda =  0 : centrosymmetric
  lambda = +1 : +P state
"""
import os
import subprocess
import json
import numpy as np
from pymatgen.core import Structure

PSEUDO_DIR = os.path.abspath("./pseudo")
BASE_OUTDIR = os.path.abspath("./tmp_switching")

pseudos = {
    "Ba": ("137.327", "Ba.pbe-spn-kjpaw_psl.1.0.0.UPF"),
    "Ti": ("47.867",  "Ti.pbe-spn-kjpaw_psl.1.0.0.UPF"),
    "O":  ("15.999",  "O.pbe-n-kjpaw_psl.1.0.0.UPF"),
}

# Load endpoints
with open("switching_structures/endpoints.json", "r") as f:
    data = json.load(f)

polar_plus = Structure.from_dict(data["polar_plus"])
polar_minus = Structure.from_dict(data["polar_minus"])

# Generate images along the path from -P to +P
# lambda = -1 to +1 in N steps
N_IMAGES = 11  # odd number so lambda=0 (centrosymmetric) is included

lambdas = np.linspace(-1, 1, N_IMAGES)

# Interpolate: use polar_minus as start, polar_plus as end
images = polar_minus.interpolate(
    polar_plus,
    nimages=N_IMAGES - 1,  # pymatgen adds 1 for endpoint
    interpolate_lattices=True,
    autosort_tol=0.0,
)

print(f"Generated {len(images)} images for lambda = -1 to +1")


def write_scf_input(struct, prefix, outdir):
    """Write QE SCF input for a given structure."""
    os.makedirs(outdir, exist_ok=True)
    lat = struct.lattice.matrix

    species_set = list(dict.fromkeys([str(s) for s in struct.species]))
    sp_block = ""
    for sp in species_set:
        mass, pp = pseudos[sp]
        sp_block += f"  {sp}  {mass}  {pp}\n"

    pos_block = ""
    for site in struct:
        sp = str(site.specie)
        fc = site.frac_coords
        pos_block += f"  {sp}  {fc[0]:.10f}  {fc[1]:.10f}  {fc[2]:.10f}\n"

    inp = f"""&CONTROL
    calculation = 'scf'
    prefix      = '{prefix}'
    outdir      = '{outdir}'
    pseudo_dir  = '{PSEUDO_DIR}'
    tprnfor     = .true.
    tstress     = .true.
    verbosity   = 'high'
/

&SYSTEM
    ibrav       = 0
    nat         = {struct.num_sites}
    ntyp        = {len(species_set)}
    ecutwfc     = 50.0
    ecutrho     = 400.0
    occupations = 'fixed'
/

&ELECTRONS
    conv_thr    = 1.0d-10
    mixing_beta = 0.5
/

CELL_PARAMETERS (angstrom)
  {lat[0][0]:.10f}  {lat[0][1]:.10f}  {lat[0][2]:.10f}
  {lat[1][0]:.10f}  {lat[1][1]:.10f}  {lat[1][2]:.10f}
  {lat[2][0]:.10f}  {lat[2][1]:.10f}  {lat[2][2]:.10f}

ATOMIC_SPECIES
{sp_block}
ATOMIC_POSITIONS (crystal)
{pos_block}
K_POINTS (automatic)
  6 6 6  1 1 1
"""
    return inp


# Run SCF for each image
energies = []
for i, struct in enumerate(images):
    lam = lambdas[i]
    prefix = f"switch_{i:02d}"
    outdir = os.path.join(BASE_OUTDIR, prefix)

    inp = write_scf_input(struct, prefix, outdir)
    infile = f"{prefix}_scf.in"
    outfile = f"{prefix}_scf.out"

    with open(infile, "w") as f:
        f.write(inp)

    print(f"Running SCF for image {i:2d} (lambda = {lam:+.3f}) ...")
    result = subprocess.run(
        ["mpirun", "-np", "4", "pw.x", "-in", infile],
        capture_output=True, text=True, timeout=1800
    )
    with open(outfile, "w") as f:
        f.write(result.stdout)

    # Parse total energy
    energy = None
    if result.returncode == 0:
        import re
        m = re.search(r"!\s+total energy\s+=\s+([-\d.]+)\s+Ry", result.stdout)
        if m:
            energy = float(m.group(1))
            print(f"  E = {energy:.8f} Ry")
        else:
            print(f"  WARNING: Could not parse energy from output.")
    else:
        print(f"  ERROR: SCF failed for image {i}")

    energies.append({"index": i, "lambda": float(lam), "energy_Ry": energy})

# Save raw results
with open("switching_energies_raw.json", "w") as f:
    json.dump(energies, f, indent=2)
print(f"\nRaw energies saved to switching_energies_raw.json")
```

### Step 2: ASE + MACE-MP-0 NEB for Fast Screening

```python
#!/usr/bin/env python3
"""
Fast NEB screening of the ferroelectric switching barrier using MACE-MP-0.

MACE-MP-0 is a universal machine learning interatomic potential that provides
reasonable energetics for many materials without any DFT calculations.
This is useful for:
  - Quick screening of switching barriers before expensive DFT NEB
  - Exploring multiple switching paths
  - Estimating barrier trends across material families

Note: MACE-MP-0 accuracy varies by material. Always validate with DFT
for quantitative results.
"""
import os
import json
import numpy as np
from ase.io import read, write
from ase.build import bulk
from ase.neb import NEB
from ase.optimize import BFGS, FIRE
from ase import Atoms
from pymatgen.core import Structure
from pymatgen.io.ase import AseAtomsAdaptor
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ---- Load MACE-MP-0 calculator ----
from mace.calculators import mace_mp
calc = mace_mp(model="medium", dispersion=False, default_dtype="float64")

# ---- Load endpoint structures ----
with open("switching_structures/endpoints.json", "r") as f:
    data = json.load(f)

polar_plus = Structure.from_dict(data["polar_plus"])
polar_minus = Structure.from_dict(data["polar_minus"])

adaptor = AseAtomsAdaptor()
atoms_plus = adaptor.get_atoms(polar_plus)
atoms_minus = adaptor.get_atoms(polar_minus)

# ---- Relax endpoints with MACE ----
print("Relaxing +P endpoint with MACE-MP-0 ...")
atoms_plus.calc = calc
opt = BFGS(atoms_plus, logfile="mace_relax_plus.log")
opt.run(fmax=0.01, steps=200)
e_plus = atoms_plus.get_potential_energy()
print(f"  E(+P) = {e_plus:.6f} eV")

print("Relaxing -P endpoint with MACE-MP-0 ...")
atoms_minus.calc = calc
opt = BFGS(atoms_minus, logfile="mace_relax_minus.log")
opt.run(fmax=0.01, steps=200)
e_minus = atoms_minus.get_potential_energy()
print(f"  E(-P) = {e_minus:.6f} eV")

print(f"  E(+P) - E(-P) = {(e_plus - e_minus)*1000:.2f} meV "
      f"(should be ~0 by symmetry)")

# ---- Set up NEB ----
N_IMAGES = 7  # intermediate images (not counting endpoints)

# Create interpolated images
images_neb = [atoms_minus.copy()]
for i in range(N_IMAGES):
    image = atoms_minus.copy()
    image.calc = mace_mp(model="medium", dispersion=False,
                         default_dtype="float64")
    images_neb.append(image)
images_neb.append(atoms_plus.copy())

# Interpolate positions
neb = NEB(images_neb, climb=True, parallel=False)
neb.interpolate(method="idpp")  # IDPP interpolation for better initial path

print(f"\nRunning CI-NEB with {N_IMAGES} intermediate images ...")
print("Using FIRE optimizer ...")

optimizer = FIRE(neb, logfile="mace_neb.log")
optimizer.run(fmax=0.03, steps=500)

# ---- Extract energy profile ----
energies_neb = [image.get_potential_energy() for image in images_neb]
e_ref = min(energies_neb)
energies_rel = [(e - e_ref) * 1000 for e in energies_neb]  # meV

# Map to lambda coordinates
n_total = len(images_neb)
lambdas_neb = np.linspace(-1, 1, n_total)

print("\nNEB Energy Profile (MACE-MP-0):")
print(f"{'Image':>5} {'lambda':>8} {'E_rel (meV)':>12}")
print("-" * 30)
for i, (lam, e_rel) in enumerate(zip(lambdas_neb, energies_rel)):
    marker = ""
    if e_rel == max(energies_rel):
        marker = " <-- barrier"
    elif abs(e_rel) < 0.1:
        marker = " <-- minimum"
    print(f"{i:5d} {lam:8.3f} {e_rel:12.2f}{marker}")

barrier_meV = max(energies_rel)
barrier_eV = barrier_meV / 1000
barrier_meV_per_atom = barrier_meV / atoms_plus.get_global_number_of_atoms()

print(f"\nSwitching barrier (MACE-MP-0):")
print(f"  Delta_E = {barrier_meV:.1f} meV/f.u. = {barrier_eV:.4f} eV/f.u.")
print(f"  Delta_E = {barrier_meV_per_atom:.1f} meV/atom")
print(f"\nCAUTION: MACE-MP-0 barriers are approximate. Validate with DFT NEB.")

# ---- Save NEB trajectory ----
os.makedirs("mace_neb_results", exist_ok=True)
for i, image in enumerate(images_neb):
    write(f"mace_neb_results/image_{i:02d}.xyz", image)

# ---- Plot ----
fig, ax = plt.subplots(figsize=(8, 5))
ax.plot(lambdas_neb, energies_rel, "o-", color="steelblue",
        linewidth=2, markersize=8)
ax.fill_between(lambdas_neb, 0, energies_rel, alpha=0.1, color="steelblue")
ax.axhline(y=0, color="k", linestyle="-", linewidth=0.5)
ax.axvline(x=0, color="gray", linestyle="--", alpha=0.5,
           label="Centrosymmetric")
ax.set_xlabel("$\\lambda$ (switching coordinate)", fontsize=13)
ax.set_ylabel("$\\Delta E$ (meV/f.u.)", fontsize=13)
ax.set_title(f"Ferroelectric Switching Profile (MACE-MP-0)\n"
             f"Barrier = {barrier_meV:.1f} meV/f.u.", fontsize=14)
ax.legend(fontsize=11)
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("mace_switching_profile.png", dpi=150, bbox_inches="tight")
print("\nPlot saved: mace_switching_profile.png")

# ---- Save results ----
neb_results = {
    "method": "MACE-MP-0 CI-NEB",
    "n_images": N_IMAGES,
    "lambdas": lambdas_neb.tolist(),
    "energies_meV_per_fu": energies_rel,
    "barrier_meV_per_fu": barrier_meV,
    "barrier_eV_per_fu": barrier_eV,
}
with open("mace_neb_results/neb_results.json", "w") as f:
    json.dump(neb_results, f, indent=2)
print("Results saved to mace_neb_results/neb_results.json")
```

### Step 3: QE NEB for Accurate Barrier

```python
#!/usr/bin/env python3
"""
Run QE NEB calculation for accurate ferroelectric switching barrier.

QE's neb.x implements the NEB method with:
  - CI-NEB (climbing image) for accurate saddle point
  - Variable-cell NEB for simultaneous cell + atom optimization
  - String method as an alternative to NEB

The path is from -P to +P through the centrosymmetric transition state.
"""
import os
import subprocess
import json
import numpy as np
from pymatgen.core import Structure

PSEUDO_DIR = os.path.abspath("./pseudo")
OUTDIR = os.path.abspath("./tmp_neb")
os.makedirs(OUTDIR, exist_ok=True)

# Load endpoint structures
with open("switching_structures/endpoints.json", "r") as f:
    data = json.load(f)

polar_plus = Structure.from_dict(data["polar_plus"])
polar_minus = Structure.from_dict(data["polar_minus"])

# ---- Write NEB input ----
# QE neb.x input has a special format with BEGIN/END blocks.

pseudos = {
    "Ba": ("137.327", "Ba.pbe-spn-kjpaw_psl.1.0.0.UPF"),
    "Ti": ("47.867",  "Ti.pbe-spn-kjpaw_psl.1.0.0.UPF"),
    "O":  ("15.999",  "O.pbe-n-kjpaw_psl.1.0.0.UPF"),
}


def format_positions(struct):
    """Format atomic positions for QE NEB input."""
    lines = []
    for site in struct:
        sp = str(site.specie)
        fc = site.frac_coords
        lines.append(
            f"  {sp}  {fc[0]:.10f}  {fc[1]:.10f}  {fc[2]:.10f}"
        )
    return "\n".join(lines)


def format_cell(struct):
    """Format cell parameters for QE NEB input."""
    lat = struct.lattice.matrix
    return (
        f"  {lat[0][0]:.10f}  {lat[0][1]:.10f}  {lat[0][2]:.10f}\n"
        f"  {lat[1][0]:.10f}  {lat[1][1]:.10f}  {lat[1][2]:.10f}\n"
        f"  {lat[2][0]:.10f}  {lat[2][1]:.10f}  {lat[2][2]:.10f}"
    )


N_IMAGES = 7  # intermediate images

neb_input = f"""BEGIN
BEGIN_PATH_INPUT
&PATH
    string_method  = 'neb'
    nstep_path     = 100
    num_of_images  = {N_IMAGES + 2}
    opt_scheme     = 'broyden'
    CI_scheme      = 'auto'
    first_last_opt = .false.
    ds             = 1.0
    k_max          = 0.3
    k_min          = 0.2
    path_thr       = 0.05
/
END_PATH_INPUT

BEGIN_ENGINE_INPUT
&CONTROL
    prefix      = 'batio3_neb'
    outdir      = '{OUTDIR}'
    pseudo_dir  = '{PSEUDO_DIR}'
/

&SYSTEM
    ibrav       = 0
    nat         = 5
    ntyp        = 3
    ecutwfc     = 50.0
    ecutrho     = 400.0
    occupations = 'fixed'
/

&ELECTRONS
    conv_thr    = 1.0d-8
    mixing_beta = 0.5
/

ATOMIC_SPECIES
  Ba  {pseudos["Ba"][0]}  {pseudos["Ba"][1]}
  Ti  {pseudos["Ti"][0]}  {pseudos["Ti"][1]}
  O   {pseudos["O"][0]}   {pseudos["O"][1]}

K_POINTS (automatic)
  6 6 6  1 1 1

CELL_PARAMETERS (angstrom)
{format_cell(polar_minus)}

BEGIN_POSITIONS
FIRST_IMAGE
ATOMIC_POSITIONS (crystal)
{format_positions(polar_minus)}
LAST_IMAGE
ATOMIC_POSITIONS (crystal)
{format_positions(polar_plus)}
END_POSITIONS

END_ENGINE_INPUT
END
"""

with open("batio3_neb.in", "w") as f:
    f.write(neb_input)
print("Written: batio3_neb.in")

print(f"\nRunning QE NEB with {N_IMAGES + 2} total images ...")
print("This may take several hours depending on system size and parallelism.")
result = subprocess.run(
    ["mpirun", "-np", "4", "neb.x", "-in", "batio3_neb.in"],
    capture_output=True, text=True, timeout=14400  # 4 hours
)
with open("batio3_neb.out", "w") as f:
    f.write(result.stdout)

if result.returncode != 0:
    print("ERROR: NEB failed!")
    print(result.stderr[-500:] if result.stderr else "No stderr")
else:
    if "JOB DONE" in result.stdout:
        print("NEB completed successfully.")
    else:
        print("WARNING: NEB may not have completed. Check output.")
    print("Output: batio3_neb.out")
```

### Step 4: Parse NEB Results and Analyze Switching Profile

```python
#!/usr/bin/env python3
"""
Parse QE NEB output and analyze the ferroelectric switching profile.

QE neb.x outputs:
  - Energy of each image at each NEB step
  - Path coordinates (reaction coordinate)
  - Forces on each image
  - Climbing image info

The output also creates files in the outdir:
  - prefix.path : energy and path coordinate for each image
  - prefix.dat  : summary data
"""
import re
import os
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def parse_neb_output(filename):
    """
    Parse QE neb.x output for the energy profile.

    Returns:
        dict with:
            'energies_Ry': list of energies per image (final iteration)
            'reaction_coords': list of reaction coordinates
            'barrier_Ry': energy barrier in Ry
            'barrier_eV': energy barrier in eV
            'n_iterations': number of NEB iterations completed
    """
    result = {
        "energies_Ry": [],
        "reaction_coords": [],
        "barrier_Ry": None,
        "barrier_eV": None,
        "n_iterations": 0,
    }

    if not os.path.exists(filename):
        print(f"File not found: {filename}")
        return result

    with open(filename, "r") as f:
        text = f.read()

    # Parse iteration count
    iterations = re.findall(r"neb:\s+(\d+)\s+", text)
    if iterations:
        result["n_iterations"] = int(iterations[-1])

    # Parse the final energy profile
    # QE NEB output format (near end of file):
    #   image    energy (eV)    error (eV/A)
    #     1      -XXX.XXXX        0.XXXX
    #     2      -XXX.XXXX        0.XXXX
    #     ...
    #
    # Or from the .path file:
    #   reaction_coord  energy(Ry)
    energy_blocks = re.findall(
        r"image\s+energy\s+\(eV\)\s+error\s+\(eV/A\)(.*?)(?:neb:|$)",
        text, re.DOTALL
    )

    if energy_blocks:
        last_block = energy_blocks[-1]
        for line in last_block.strip().split("\n"):
            m = re.match(r"\s*(\d+)\s+([-\d.]+)\s+([-\d.]+)", line)
            if m:
                result["energies_Ry"].append(float(m.group(2)) / 13.6057)
                result["reaction_coords"].append(float(m.group(1)))

    # Alternative: parse from .dat file if it exists
    dat_file = filename.replace(".out", ".dat")
    if not result["energies_Ry"] and os.path.exists(dat_file):
        with open(dat_file) as f:
            for line in f:
                parts = line.split()
                if len(parts) >= 2:
                    try:
                        result["reaction_coords"].append(float(parts[0]))
                        result["energies_Ry"].append(float(parts[1]))
                    except ValueError:
                        continue

    # Alternative: parse from prefix.path file
    path_file = os.path.join(
        os.path.dirname(filename) or ".",
        "tmp_neb", "batio3_neb.path"
    )
    if not result["energies_Ry"] and os.path.exists(path_file):
        with open(path_file) as f:
            for line in f:
                parts = line.split()
                if len(parts) >= 2:
                    try:
                        result["reaction_coords"].append(float(parts[0]))
                        result["energies_Ry"].append(float(parts[1]))
                    except ValueError:
                        continue

    # Compute barrier
    if result["energies_Ry"]:
        e_min = min(result["energies_Ry"])
        e_max = max(result["energies_Ry"])
        result["barrier_Ry"] = e_max - e_min
        result["barrier_eV"] = result["barrier_Ry"] * 13.6057

    return result


# ---- Parse NEB output ----
neb_data = parse_neb_output("batio3_neb.out")

print("=" * 60)
print("QE NEB RESULTS: FERROELECTRIC SWITCHING")
print("=" * 60)
print(f"NEB iterations completed: {neb_data['n_iterations']}")

if neb_data["energies_Ry"]:
    e_min = min(neb_data["energies_Ry"])
    energies_rel_meV = [(e - e_min) * 13605.7 for e in neb_data["energies_Ry"]]

    print(f"\n{'Image':>5} {'E_rel (meV)':>12}")
    print("-" * 20)
    for i, e_meV in enumerate(energies_rel_meV):
        marker = ""
        if e_meV == max(energies_rel_meV):
            marker = " <-- saddle point"
        elif e_meV < 0.1:
            marker = " <-- minimum"
        print(f"{i:5d} {e_meV:12.2f}{marker}")

    barrier_meV = neb_data["barrier_eV"] * 1000 if neb_data["barrier_eV"] else None
    if barrier_meV is not None:
        print(f"\nSwitching barrier:")
        print(f"  Delta_E = {barrier_meV:.1f} meV/f.u.")
        print(f"  Delta_E = {neb_data['barrier_eV']:.4f} eV/f.u.")
        print(f"  Delta_E = {barrier_meV * 0.0963:.2f} kJ/mol")

        # Estimate coercive field (very rough)
        # E_c ~ Delta_E / (P_s * Omega)
        # BaTiO3: P_s ~ 0.26 C/m^2, Omega ~ 64.3 A^3
        P_s = 0.26  # C/m^2
        Omega = 64.3 * 1e-30  # m^3
        Delta_E_J = neb_data["barrier_eV"] * 1.602e-19
        E_c_estimate = Delta_E_J / (P_s * Omega)
        print(f"\n  Estimated coercive field: ~{E_c_estimate/1e6:.0f} MV/m")
        print(f"  (Very rough: E_c = Delta_E / (P_s * Omega))")

    # Plot
    n_images = len(energies_rel_meV)
    lambdas = np.linspace(-1, 1, n_images)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(lambdas, energies_rel_meV, "o-", color="darkred",
            linewidth=2, markersize=8)
    ax.fill_between(lambdas, 0, energies_rel_meV,
                    alpha=0.1, color="darkred")
    ax.axhline(y=0, color="k", linestyle="-", linewidth=0.5)
    ax.axvline(x=0, color="gray", linestyle="--", alpha=0.5)

    ax.set_xlabel("Switching coordinate $\\lambda$", fontsize=13)
    ax.set_ylabel("$\\Delta E$ (meV/f.u.)", fontsize=13)
    ax.set_title(
        f"Ferroelectric Switching Barrier (QE NEB)\n"
        f"$\\Delta E$ = {barrier_meV:.1f} meV/f.u.",
        fontsize=14,
    )
    ax.grid(True, alpha=0.3)

    # Annotate barrier
    i_max = np.argmax(energies_rel_meV)
    ax.annotate(
        f"{barrier_meV:.1f} meV",
        xy=(lambdas[i_max], energies_rel_meV[i_max]),
        xytext=(lambdas[i_max] + 0.2, energies_rel_meV[i_max] + 5),
        fontsize=11,
        arrowprops=dict(arrowstyle="->", color="black"),
    )

    plt.tight_layout()
    plt.savefig("neb_switching_barrier.png", dpi=150, bbox_inches="tight")
    print("\nPlot saved: neb_switching_barrier.png")
else:
    print("No energy data found. Check NEB output for errors.")
```

### Step 5: Polarization Along the Switching Path

```python
#!/usr/bin/env python3
"""
Compute polarization at each image along the NEB switching path.

This combines the switching energy profile with Berry phase polarization
to create a P-E-like hysteresis-related plot: P(lambda) and E(lambda)
on the same switching coordinate.

Uses the Born effective charge linear approximation for efficiency.
For accurate results, run full Berry phase at each image (see
polarization/ sub-skill).
"""
import os
import json
import numpy as np
from pymatgen.core import Structure, Lattice
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

E_CHARGE = 1.602176634e-19
ANGSTROM_TO_M = 1e-10

# ---- Load structures ----
with open("switching_structures/endpoints.json", "r") as f:
    data = json.load(f)

polar_plus = Structure.from_dict(data["polar_plus"])
polar_minus = Structure.from_dict(data["polar_minus"])
centro = Structure.from_dict(data["centrosymmetric"])

# Interpolate images from -P to +P
N_IMAGES = 11
images = polar_minus.interpolate(
    polar_plus,
    nimages=N_IMAGES - 1,
    interpolate_lattices=True,
    autosort_tol=0.0,
)

lambdas = np.linspace(-1, 1, len(images))

# ---- Use Born effective charges for polarization estimate ----
# Load from born_charges_results.json if available, else use typical values
Z_star_diag = {"Ba": 2.75, "Ti": 7.25, "O": -2.15}  # isotropic average

# Compute polarization for each image
P_z_values = []
for struct in images:
    Omega_m3 = struct.volume * ANGSTROM_TO_M**3

    # Displacement from centrosymmetric reference
    delta_frac = struct.frac_coords - centro.frac_coords
    delta_frac = delta_frac - np.round(delta_frac)
    delta_cart = delta_frac @ struct.lattice.matrix  # Angstrom

    # P_z = (e / Omega) * sum_i Z*_i * dz_i
    P_z = 0.0
    for i, site in enumerate(struct):
        sp = str(site.specie)
        Z = Z_star_diag.get(sp, 0.0)
        dz_m = delta_cart[i, 2] * ANGSTROM_TO_M
        P_z += E_CHARGE * Z * dz_m / Omega_m3

    P_z_values.append(P_z)

P_z_uC_cm2 = [P * 100 for P in P_z_values]  # convert C/m^2 to uC/cm^2

# ---- Load energy profile ----
energies_meV = None
if os.path.exists("switching_energies_raw.json"):
    with open("switching_energies_raw.json") as f:
        e_data = json.load(f)
    energies_Ry = [e["energy_Ry"] for e in e_data if e["energy_Ry"] is not None]
    if energies_Ry:
        e_min = min(energies_Ry)
        energies_meV = [(e - e_min) * 13605.7 for e in energies_Ry]

# ---- Plot P and E vs lambda ----
fig, ax1 = plt.subplots(figsize=(8, 5))

# Polarization
color1 = "steelblue"
ax1.plot(lambdas, P_z_uC_cm2, "o-", color=color1,
         linewidth=2, markersize=8, label="$P_z$")
ax1.set_xlabel("Switching coordinate $\\lambda$", fontsize=13)
ax1.set_ylabel("$P_z$ ($\\mu$C/cm$^2$)", color=color1, fontsize=13)
ax1.tick_params(axis="y", labelcolor=color1)
ax1.axhline(y=0, color="gray", linestyle=":", alpha=0.5)

# Energy on secondary axis
if energies_meV and len(energies_meV) == len(lambdas):
    ax2 = ax1.twinx()
    color2 = "darkred"
    ax2.plot(lambdas, energies_meV, "s--", color=color2,
             linewidth=1.5, markersize=6, label="$\\Delta E$")
    ax2.set_ylabel("$\\Delta E$ (meV/f.u.)", color=color2, fontsize=13)
    ax2.tick_params(axis="y", labelcolor=color2)

ax1.set_title("Polarization and Energy Along Switching Path", fontsize=14)
ax1.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("polarization_switching_path.png", dpi=150, bbox_inches="tight")
print("Plot saved: polarization_switching_path.png")

# ---- Print summary ----
print("\nPolarization along switching path:")
print(f"{'lambda':>8} {'P_z (C/m2)':>14} {'P_z (uC/cm2)':>14}")
print("-" * 40)
for lam, Pz, Pz_uC in zip(lambdas, P_z_values, P_z_uC_cm2):
    print(f"{lam:8.3f} {Pz:14.6f} {Pz_uC:14.4f}")

print(f"\nP_s(+P) = {P_z_uC_cm2[-1]:.2f} uC/cm^2")
print(f"P_s(-P) = {P_z_uC_cm2[0]:.2f} uC/cm^2")
print(f"2*P_s   = {P_z_uC_cm2[-1] - P_z_uC_cm2[0]:.2f} uC/cm^2")

# Save
results = {
    "lambdas": lambdas.tolist(),
    "P_z_Cm2": P_z_values,
    "P_z_uC_cm2": P_z_uC_cm2,
}
with open("switching_polarization.json", "w") as f:
    json.dump(results, f, indent=2)
print("Results saved: switching_polarization.json")
```

### Step 6: VASP NEB Reference

```python
#!/usr/bin/env python3
"""
VASP NEB workflow for ferroelectric switching barrier.

VASP NEB requires the VTST (Henkelman group) scripts and patches,
or can be run with the built-in NEB in VASP 6.x.

NOTE: VASP is not available in the current container.
This script generates input files for reference.
"""
import os
import json
import numpy as np
from pymatgen.core import Structure

# Load endpoint structures
with open("switching_structures/endpoints.json", "r") as f:
    data = json.load(f)

polar_plus = Structure.from_dict(data["polar_plus"])
polar_minus = Structure.from_dict(data["polar_minus"])

N_IMAGES = 5  # VASP NEB intermediate images

# ---- Create directory structure ----
# VASP NEB uses directories: 00/ 01/ 02/ ... (N+1)/
# 00/ = initial image (-P), (N+1)/ = final image (+P)
# 01/ to N/ = intermediate images
neb_dir = "vasp_neb"
os.makedirs(neb_dir, exist_ok=True)

# Interpolate
images = polar_minus.interpolate(
    polar_plus,
    nimages=N_IMAGES + 1,
    interpolate_lattices=True,
    autosort_tol=0.0,
)

for i, struct in enumerate(images):
    img_dir = os.path.join(neb_dir, f"{i:02d}")
    os.makedirs(img_dir, exist_ok=True)

    # Write POSCAR
    poscar_lines = ["BaTiO3 ferroelectric switching"]
    poscar_lines.append("1.0")
    lat = struct.lattice.matrix
    for v in lat:
        poscar_lines.append(f"  {v[0]:.10f}  {v[1]:.10f}  {v[2]:.10f}")

    species_set = list(dict.fromkeys([str(s) for s in struct.species]))
    counts = [sum(1 for s in struct.species if str(s) == sp) for sp in species_set]
    poscar_lines.append("  ".join(species_set))
    poscar_lines.append("  ".join(str(c) for c in counts))
    poscar_lines.append("Direct")
    for site in struct:
        fc = site.frac_coords
        poscar_lines.append(
            f"  {fc[0]:.10f}  {fc[1]:.10f}  {fc[2]:.10f}"
        )

    with open(os.path.join(img_dir, "POSCAR"), "w") as f:
        f.write("\n".join(poscar_lines) + "\n")

# ---- Write INCAR for NEB ----
incar = """SYSTEM = BaTiO3 ferroelectric switching NEB

# Electronic
ENCUT  = 550
EDIFF  = 1E-6
PREC   = Accurate
LREAL  = .FALSE.
ALGO   = Normal
ISMEAR = 0
SIGMA  = 0.05

# NEB settings
IBRION  = 3        ! Damped MD for NEB optimization
POTIM   = 0.0      ! Set by optimizer
ICHAIN  = 0        ! NEB method
IMAGES  = {N_IMAGES}       ! Number of intermediate images
SPRING  = -5.0     ! Spring constant (eV/Ang^2)
LCLIMB  = .TRUE.   ! Climbing image NEB
LNEBCELL = .FALSE. ! Fixed cell NEB (set .TRUE. for variable-cell)

# Ionic
NSW     = 200      ! Max NEB steps
EDIFFG  = -0.03    ! Force convergence (eV/Ang)

# Output
LWAVE  = .FALSE.
LCHARG = .FALSE.
"""

with open(os.path.join(neb_dir, "INCAR"), "w") as f:
    f.write(incar)

# KPOINTS
kpoints = """Automatic mesh
0
Gamma
  6  6  6
  0  0  0
"""
with open(os.path.join(neb_dir, "KPOINTS"), "w") as f:
    f.write(kpoints)

print(f"VASP NEB input files written to {neb_dir}/")
print(f"  {N_IMAGES + 2} image directories: 00/ to {N_IMAGES+1:02d}/")
print(f"  INCAR with CI-NEB settings")
print(f"  KPOINTS: 6x6x6 Gamma-centered")
print(f"\nTo run:")
print(f"  cd {neb_dir}")
print(f"  cp /path/to/POTCAR .")
print(f"  mpirun -np N vasp_std")
print(f"\nAfter completion, parse with:")
print(f"  nebresults.pl  (VTST scripts)")
print(f"  or use pymatgen.io.vasp.outputs.Vasprun")
```

### Step 7: Frozen-Phonon Double-Well (Alternative 1D Approach)

```python
#!/usr/bin/env python3
"""
Compute the double-well energy profile along the ferroelectric soft mode.

Instead of NEB, this approach:
1. Identifies the soft mode eigenvector at the centrosymmetric structure
2. Displaces atoms along this eigenvector with varying amplitude
3. Computes total energy at each amplitude

This gives a 1D cross-section of the energy surface along the primary
switching coordinate. It is simpler than NEB but only captures the
single-mode component of switching.

The resulting E(Q) curve is fit to a double-well potential:
  E(Q) = a*Q^4 - b*Q^2
where Q is the mode amplitude.
"""
import os
import subprocess
import re
import json
import numpy as np
from pymatgen.core import Structure, Lattice
from scipy.optimize import curve_fit
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

PSEUDO_DIR = os.path.abspath("./pseudo")
BASE_OUTDIR = os.path.abspath("./tmp_frozen_phonon")

pseudos = {
    "Ba": ("137.327", "Ba.pbe-spn-kjpaw_psl.1.0.0.UPF"),
    "Ti": ("47.867",  "Ti.pbe-spn-kjpaw_psl.1.0.0.UPF"),
    "O":  ("15.999",  "O.pbe-n-kjpaw_psl.1.0.0.UPF"),
}

# ---- Centrosymmetric reference ----
a = 4.00
centro = Structure(
    Lattice.cubic(a),
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

# ---- Soft mode eigenvector (Gamma_4- / Slater mode) ----
# For cubic BaTiO3, the ferroelectric soft mode displaces Ti against O
# along z. The eigenvector (in Cartesian, normalized):
#   Ba:  0        (Ba barely moves)
#   Ti:  +0.60 z  (Ti moves up)
#   O1:  -0.50 z  (apical O moves down)
#   O2:  -0.25 z  (equatorial O moves down)
#   O3:  -0.25 z  (equatorial O moves down)
# These are approximate; exact eigenvectors come from ph.x output.

soft_mode_z = np.array([
    [0.0, 0.0, 0.00],   # Ba
    [0.0, 0.0, 0.60],   # Ti
    [0.0, 0.0, -0.50],  # O1
    [0.0, 0.0, -0.25],  # O2
    [0.0, 0.0, -0.25],  # O3
])

# Normalize
norm = np.sqrt(np.sum(soft_mode_z**2))
soft_mode_z /= norm

# ---- Generate displaced structures ----
# Q ranges from -Q_max to +Q_max in Angstrom * sqrt(AMU)
# For now, use displacement amplitude in Angstrom directly
amplitudes = np.linspace(-0.15, 0.15, 15)  # Angstrom

structures = []
for amp in amplitudes:
    # Displacement in Cartesian coordinates
    disp_cart = amp * soft_mode_z  # Angstrom
    # Convert to fractional
    disp_frac = disp_cart @ np.linalg.inv(centro.lattice.matrix)
    new_frac = centro.frac_coords + disp_frac

    struct = Structure(
        centro.lattice,
        species=[str(s.specie) for s in centro],
        coords=new_frac,
        coords_are_cartesian=False,
    )
    structures.append(struct)


def write_scf(struct, prefix, outdir):
    """Write QE SCF input."""
    os.makedirs(outdir, exist_ok=True)
    lat = struct.lattice.matrix
    species_set = list(dict.fromkeys([str(s) for s in struct.species]))
    sp_block = "".join(
        f"  {sp} {pseudos[sp][0]} {pseudos[sp][1]}\n" for sp in species_set
    )
    pos_block = "".join(
        f"  {str(s.specie)} {s.frac_coords[0]:.10f} "
        f"{s.frac_coords[1]:.10f} {s.frac_coords[2]:.10f}\n"
        for s in struct
    )
    return f"""&CONTROL
    calculation = 'scf'
    prefix      = '{prefix}'
    outdir      = '{outdir}'
    pseudo_dir  = '{PSEUDO_DIR}'
/
&SYSTEM
    ibrav       = 0
    nat         = {struct.num_sites}
    ntyp        = {len(species_set)}
    ecutwfc     = 50.0
    ecutrho     = 400.0
    occupations = 'fixed'
/
&ELECTRONS
    conv_thr    = 1.0d-10
    mixing_beta = 0.5
/
CELL_PARAMETERS (angstrom)
  {lat[0][0]:.10f}  {lat[0][1]:.10f}  {lat[0][2]:.10f}
  {lat[1][0]:.10f}  {lat[1][1]:.10f}  {lat[1][2]:.10f}
  {lat[2][0]:.10f}  {lat[2][1]:.10f}  {lat[2][2]:.10f}
ATOMIC_SPECIES
{sp_block}
ATOMIC_POSITIONS (crystal)
{pos_block}
K_POINTS (automatic)
  6 6 6  1 1 1
"""


# Run SCF for each displacement
energies = []
for i, (amp, struct) in enumerate(zip(amplitudes, structures)):
    prefix = f"frozen_{i:02d}"
    outdir = os.path.join(BASE_OUTDIR, prefix)

    inp = write_scf(struct, prefix, outdir)
    infile = f"{prefix}.in"
    outfile = f"{prefix}.out"

    with open(infile, "w") as f:
        f.write(inp)

    print(f"Running SCF: amplitude = {amp:+.4f} A ...")
    result = subprocess.run(
        ["mpirun", "-np", "4", "pw.x", "-in", infile],
        capture_output=True, text=True, timeout=1800
    )
    with open(outfile, "w") as f:
        f.write(result.stdout)

    energy = None
    if result.returncode == 0:
        m = re.search(r"!\s+total energy\s+=\s+([-\d.]+)\s+Ry", result.stdout)
        if m:
            energy = float(m.group(1))
    energies.append(energy)

# ---- Analyze double-well profile ----
valid = [(a, e) for a, e in zip(amplitudes, energies) if e is not None]
if not valid:
    print("ERROR: No valid energies computed.")
else:
    amps_valid = np.array([v[0] for v in valid])
    E_valid = np.array([v[1] for v in valid])  # Ry
    E_ref = E_valid[len(E_valid)//2]  # centrosymmetric energy
    E_meV = (E_valid - E_ref) * 13605.7  # meV

    # Fit to double-well: E(Q) = a*Q^4 - b*Q^2 + c
    def double_well(Q, a_coeff, b_coeff, c_coeff):
        return a_coeff * Q**4 - b_coeff * Q**2 + c_coeff

    try:
        popt, pcov = curve_fit(double_well, amps_valid, E_meV,
                               p0=[1e4, 100, 0])
        a_fit, b_fit, c_fit = popt

        # Equilibrium displacement: Q_0 = sqrt(b / 2a)
        Q_0 = np.sqrt(b_fit / (2 * a_fit)) if b_fit > 0 and a_fit > 0 else 0
        # Barrier height: E_barrier = b^2 / (4a)
        E_barrier = b_fit**2 / (4 * a_fit) if a_fit > 0 else 0

        print(f"\nDouble-well fit: E(Q) = {a_fit:.1f}*Q^4 - {b_fit:.1f}*Q^2 + {c_fit:.2f}")
        print(f"Equilibrium displacement: Q_0 = {Q_0:.4f} A")
        print(f"Barrier height: {E_barrier:.1f} meV/f.u.")
    except RuntimeError:
        print("WARNING: Double-well fit failed. Data may be noisy.")
        a_fit, b_fit, c_fit = None, None, None
        Q_0, E_barrier = 0, max(E_meV) - min(E_meV)

    # ---- Plot ----
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(amps_valid, E_meV, "o", color="steelblue", markersize=8,
            label="DFT (QE)")

    if a_fit is not None:
        Q_fine = np.linspace(amps_valid[0], amps_valid[-1], 200)
        E_fit = double_well(Q_fine, a_fit, b_fit, c_fit)
        ax.plot(Q_fine, E_fit, "-", color="darkred", linewidth=2,
                label="$aQ^4 - bQ^2$ fit")

    ax.axhline(y=0, color="k", linestyle="-", linewidth=0.5)
    ax.axvline(x=0, color="gray", linestyle="--", alpha=0.5)
    ax.set_xlabel("Soft mode amplitude $Q$ (A)", fontsize=13)
    ax.set_ylabel("$\\Delta E$ (meV/f.u.)", fontsize=13)
    ax.set_title(
        f"Frozen-Phonon Double-Well: BaTiO$_3$ Soft Mode\n"
        f"Barrier = {E_barrier:.1f} meV/f.u., $Q_0$ = {Q_0:.4f} A",
        fontsize=14,
    )
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("frozen_phonon_double_well.png", dpi=150, bbox_inches="tight")
    print("Plot saved: frozen_phonon_double_well.png")

    # Save results
    output = {
        "amplitudes_A": amps_valid.tolist(),
        "energies_meV": E_meV.tolist(),
        "fit_a": a_fit,
        "fit_b": b_fit,
        "fit_c": c_fit,
        "Q0_A": float(Q_0),
        "barrier_meV": float(E_barrier),
    }
    with open("frozen_phonon_results.json", "w") as f:
        json.dump(output, f, indent=2)
    print("Results saved: frozen_phonon_results.json")
```

---

## Key Parameters

| Parameter | Typical Value | Notes |
|---|---|---|
| **NEB: `num_of_images`** (QE) | 7-9 total | Including endpoints. More images give smoother MEP but cost more. |
| **NEB: `CI_scheme`** (QE) | `'auto'` | Climbing image activates automatically after initial convergence |
| **NEB: `path_thr`** (QE) | 0.05 eV/A | Force convergence threshold on NEB path. Use 0.03 for high accuracy. |
| **NEB: `k_max`, `k_min`** (QE) | 0.3, 0.2 | Spring constants (Ry/Bohr^2). Adjust if images bunch or spread unevenly. |
| **NEB: `IMAGES`** (VASP) | 5-7 | Number of intermediate images (excludes endpoints). |
| **NEB: `SPRING`** (VASP) | -5.0 | Spring constant (eV/Ang^2). Negative = variable spring. |
| **NEB: `LCLIMB`** (VASP) | `.TRUE.` | Enable climbing-image NEB for accurate barrier. |
| **SCF: `conv_thr`** (QE) | 1.0d-8 to 1.0d-10 | For NEB, 1e-8 is usually sufficient. For frozen phonon, use 1e-10. |
| **MACE: `fmax`** | 0.03 eV/A | NEB force convergence with MACE. Looser than DFT since MACE is approximate. |
| **Frozen phonon: amplitude range** | -0.15 to +0.15 A | Must span beyond the equilibrium displacement Q_0. |

## Interpreting Results

### Switching barrier magnitude

- **BaTiO3 (tetragonal)**: Barrier ~20-30 meV/f.u. Relatively easy to switch; low coercive field (~1 kV/cm experimentally).
- **PbTiO3**: Barrier ~100-200 meV/f.u. Harder to switch; higher coercive field.
- **BiFeO3**: Barrier ~500+ meV/f.u. Very robust polarization.
- **Barrier < 10 meV/f.u.**: Polarization is fragile and may not survive at room temperature (k_B T ~ 25 meV).
- **Barrier > 100 meV/f.u.**: Polarization is thermally robust. Switching requires significant electric field.

### Double-well shape

- **Symmetric double-well**: Expected for equivalent +P and -P states. If asymmetric, check that the two endpoints are truly related by symmetry.
- **Flat bottom (anharmonic)**: Indicates strong anharmonicity. The quartic term dominates -- the material is a "displacive" ferroelectric.
- **Deep, narrow wells**: Characteristic of "order-disorder" ferroelectrics with well-localized polar distortions.
- **Fit quality**: If E(Q) = aQ^4 - bQ^2 gives a poor fit, higher-order terms or coupling to other modes (e.g., octahedral tilts) are important.

### Polarization along switching path

- **Linear P(lambda)**: Simple single-mode switching. P changes monotonically from -P_s to +P_s.
- **Nonlinear P(lambda)**: Multiple modes involved in switching. May indicate a complex switching mechanism (e.g., rotation of polarization rather than direct reversal).
- **P = 0 at lambda = 0**: Confirms the centrosymmetric state is indeed nonpolar.

### NEB convergence

- **Path converged**: Forces on all images below `path_thr`. The MEP is reliable.
- **Climbing image at position 0**: The transition state is the centrosymmetric structure, as expected for simple perovskites.
- **Climbing image NOT at midpoint**: The transition state differs from the centrosymmetric reference -- a more complex switching mechanism is at play. This is common in materials with coupled order parameters (e.g., BiFeO3).

### MACE vs DFT comparison

- MACE-MP-0 typically captures qualitative trends but may over- or underestimate barriers by 20-50%.
- Use MACE for screening multiple materials or paths, then validate the most promising candidates with DFT NEB.

### Typical switching barriers for reference

| Material | Barrier (meV/f.u.) | Method | Reference |
|---|---|---|---|
| BaTiO3 (tetragonal) | 20-30 | DFT-PBE NEB | Cohen & Krakauer 1992 |
| PbTiO3 | 100-200 | DFT-LDA | Waghmare & Rabe 1997 |
| BiFeO3 | 500-800 | DFT-PBE NEB | Ravindran et al. 2006 |
| KNbO3 | 30-50 | DFT-PBE | Yu & Krakauer 1995 |
| HfO2 (ortho) | 200-400 | DFT-PBE NEB | Huan et al. 2014 |

## Common Issues

| Problem | Solution |
|---|---|
| **NEB does not converge** | Reduce `path_thr` gradually. Try `opt_scheme = 'broyden2'` or `'sd'` instead of `'broyden'`. Increase `nstep_path`. Check that endpoints are fully relaxed. |
| **Images bunch near one endpoint** | The spring constants are too weak. Increase `k_max` and `k_min` in QE, or use a more negative `SPRING` in VASP. |
| **Climbing image oscillates** | Set `CI_scheme = 'manual'` and specify which image climbs after the path is roughly converged. Or disable CI for initial convergence, then enable it. |
| **Barrier is zero or negative** | The centrosymmetric structure is already a minimum, not a saddle point. The material may not be ferroelectric at this level of theory. Check the phonon spectrum for soft modes. |
| **Asymmetric double-well** | The +P and -P states are not exactly related by symmetry. Check lattice parameters and atomic positions. Use `spglib` to verify space groups. |
| **NEB path deviates from linear interpolation** | This is expected for materials with coupled modes (tilts, Jahn-Teller, etc.). The NEB has found a lower-energy pathway. Analyze the intermediate images to understand the mechanism. |
| **MACE gives qualitatively wrong barrier** | MACE-MP-0 may not be accurate for all materials. Try MACE-MP-0 "large" model. If still inaccurate, fall back to DFT. |
| **QE neb.x out of memory** | NEB stores all images simultaneously. Reduce k-grid, ecutwfc, or number of images. Use `-nimage N` parallelization flag to distribute images across processor groups. |
| **Frozen phonon gives single well** | The soft mode eigenvector may be wrong, or the material is not ferroelectric. Verify the eigenvector from ph.x output. |
| **Very slow NEB convergence** | The initial path (linear interpolation) may be far from the MEP. Try IDPP interpolation or use MACE-NEB structures as the starting path for DFT NEB. |
| **VASP NEB: EDIFFG not reached** | Increase NSW. For VTST, ensure the VTST patch is correctly applied. Check that IBRION=3 and POTIM=0. |
