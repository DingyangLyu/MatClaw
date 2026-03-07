# NEB Transition State Calculations

## When to Use

- Calculate migration barriers for atomic diffusion (vacancy, interstitial, or impurity migration)
- Find minimum energy paths (MEP) for chemical reactions in solids or at surfaces
- Determine transition state geometries
- Compare diffusion barriers across different materials or pathways
- Estimate diffusion coefficients via Arrhenius relation: D ~ exp(-E_barrier / k_B T)

## Method Selection

```
Quick barrier estimate for screening?
  --> ASE + MACE NEB (Method A): minutes, good for trends and ranking paths

Publication-quality migration barrier?
  --> QE NEB via neb.x (Method B): hours to days, DFT accuracy

System outside MACE training data?
  --> QE NEB required (Method B)

Very long reaction path or many intermediates?
  --> Consider ASE + MACE first to find approximate MEP,
      then refine critical segment with QE NEB
```

## Prerequisites

- pymatgen (structure manipulation, endpoint generation)
- ASE + mace-torch (Method A)
- Quantum ESPRESSO neb.x (Method B)
- SSSP pseudopotentials for QE
- Optional: `pip install pymatgen-diffusion` for advanced diffusion path analysis

## Detailed Steps

### Method A: ASE + MACE NEB

#### Complete Workflow: Vacancy Migration in MgO

```python
#!/usr/bin/env python3
"""
NEB calculation for O vacancy migration in MgO using ASE + MACE.
Pattern inspired by atomate2's ForceFieldNebFromEndpointsMaker.
"""

import numpy as np
from pymatgen.core import Structure, Element
from pymatgen.io.ase import AseAtomsAdaptor
from ase.optimize import BFGS, FIRE
from ase.mep.neb import NEB, NEBTools
from ase.mep.neb import idpp_interpolate
from mace.calculators import mace_mp
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
import json

# ============================================================
# 1. Set up MACE calculator
# ============================================================
calc = mace_mp(model="medium", dispersion=False, default_dtype="float64")
adaptor = AseAtomsAdaptor()

# ============================================================
# 2. Build bulk supercell and create endpoints
# ============================================================
# MgO rocksalt
primitive = Structure.from_spacegroup(
    "Fm-3m",
    lattice=[[4.212, 0, 0], [0, 4.212, 0], [0, 0, 4.212]],
    species=["Mg", "O"],
    coords=[[0.0, 0.0, 0.0], [0.5, 0.5, 0.5]],
)

# 3x3x3 supercell
bulk_sc = primitive.copy()
bulk_sc.make_supercell([3, 3, 3])
print(f"Supercell: {bulk_sc.formula}, {len(bulk_sc)} atoms")

# Find two neighboring O sites for vacancy hop
o_indices = [i for i, s in enumerate(bulk_sc) if s.specie == Element("O")]

# Pick first O and find its nearest O neighbor
site0 = bulk_sc[o_indices[0]]
distances = []
for idx in o_indices[1:]:
    d = site0.distance(bulk_sc[idx])
    distances.append((idx, d))
distances.sort(key=lambda x: x[1])

# Nearest O-O neighbor (should be ~2.98 A in MgO = a/sqrt(2))
nn_idx = distances[0][0]
nn_dist = distances[0][1]
print(f"Vacancy hop: site {o_indices[0]} -> site {nn_idx}, distance = {nn_dist:.3f} A")

# Create initial image: vacancy at site 0
initial_struct = bulk_sc.copy()
initial_struct.remove_sites([o_indices[0]])

# Create final image: vacancy at nearest neighbor site
# Need to re-index since we removed a site
final_struct = bulk_sc.copy()
final_struct.remove_sites([nn_idx])

print(f"Initial endpoint: {len(initial_struct)} atoms")
print(f"Final endpoint:   {len(final_struct)} atoms")

# ============================================================
# 3. Relax endpoints
# ============================================================
print("\nRelaxing initial endpoint...")
atoms_initial = adaptor.get_atoms(initial_struct)
atoms_initial.calc = calc
opt_i = BFGS(atoms_initial, logfile="relax_endpoint_initial.log")
opt_i.run(fmax=0.01, steps=300)
e_initial = atoms_initial.get_potential_energy()
print(f"  E_initial = {e_initial:.6f} eV")

print("Relaxing final endpoint...")
atoms_final = adaptor.get_atoms(final_struct)
atoms_final.calc = calc
opt_f = BFGS(atoms_final, logfile="relax_endpoint_final.log")
opt_f.run(fmax=0.01, steps=300)
e_final = atoms_final.get_potential_energy()
print(f"  E_final = {e_final:.6f} eV")
print(f"  Delta_E = {e_final - e_initial:.6f} eV")

# ============================================================
# 4. Create NEB images with IDPP interpolation
# ============================================================
n_images = 7  # number of intermediate images (5-9 typical)

# Build image list: [initial, copy, copy, ..., final]
images = [atoms_initial.copy()]
for _ in range(n_images):
    images.append(atoms_initial.copy())
images.append(atoms_final.copy())

# IDPP (Image Dependent Pair Potential) interpolation
# Much better than linear for avoiding atomic clashes
idpp_interpolate(images, traj=None, log=None)
print(f"\nIDPP interpolation done: {n_images} intermediate images")

# Assign calculators to intermediate images (endpoints are fixed)
for img in images[1:-1]:
    img.calc = mace_mp(model="medium", dispersion=False, default_dtype="float64")

# ============================================================
# 5. Run NEB (or CI-NEB)
# ============================================================
# Standard NEB first
neb = NEB(
    images,
    climb=False,         # set True for CI-NEB after initial convergence
    k=0.1,               # spring constant (eV/A^2)
    parallel=False,      # set True if using multiple processes
)

print("\n=== Running NEB (phase 1: standard) ===")
optimizer = FIRE(neb, logfile="neb_standard.log")
optimizer.run(fmax=0.05, steps=500)
print(f"  Converged in {optimizer.nsteps} steps")

# Switch to CI-NEB for accurate barrier
neb_ci = NEB(
    images,
    climb=True,          # climbing image finds true saddle point
    k=0.1,
    parallel=False,
)

print("\n=== Running CI-NEB (phase 2: climbing image) ===")
optimizer_ci = FIRE(neb_ci, logfile="neb_cineb.log")
optimizer_ci.run(fmax=0.03, steps=500)
print(f"  Converged in {optimizer_ci.nsteps} steps")

# ============================================================
# 6. Extract results
# ============================================================
neb_tools = NEBTools(images)

# Get the barrier
barrier_forward = neb_tools.get_barrier()[0]   # forward barrier
barrier_reverse = neb_tools.get_barrier()[1]   # reverse barrier

print(f"\n=== NEB Results ===")
print(f"Forward barrier:  {barrier_forward:.4f} eV")
print(f"Reverse barrier:  {barrier_reverse:.4f} eV")

# Get energies and positions along the path
energies = [img.get_potential_energy() for img in images]
e_ref = energies[0]
relative_energies = [e - e_ref for e in energies]

# Reaction coordinate (cumulative distance)
reaction_coord = [0.0]
for i in range(1, len(images)):
    diff = images[i].get_positions() - images[i-1].get_positions()
    d = np.sqrt(np.sum(diff**2))
    reaction_coord.append(reaction_coord[-1] + d)

# Normalize to [0, 1]
rc_norm = [r / reaction_coord[-1] for r in reaction_coord]

# Identify transition state (highest energy image)
ts_idx = np.argmax(relative_energies)
ts_energy = relative_energies[ts_idx]
print(f"Transition state at image {ts_idx}, E_rel = {ts_energy:.4f} eV")

# ============================================================
# 7. Visualization: Energy profile
# ============================================================
fig, ax = plt.subplots(figsize=(8, 5))

# Plot discrete image energies
ax.plot(rc_norm, relative_energies, "o-", color="steelblue",
        markersize=8, linewidth=2, label="NEB images")

# Highlight transition state
ax.plot(rc_norm[ts_idx], relative_energies[ts_idx], "r*",
        markersize=15, zorder=5, label=f"TS (E = {ts_energy:.3f} eV)")

# Spline fit for smooth curve
try:
    from scipy.interpolate import CubicSpline
    cs = CubicSpline(rc_norm, relative_energies)
    rc_fine = np.linspace(0, 1, 200)
    ax.plot(rc_fine, cs(rc_fine), "--", color="gray", alpha=0.6, label="Spline fit")
except ImportError:
    pass

ax.set_xlabel("Reaction coordinate", fontsize=12)
ax.set_ylabel("Energy relative to initial (eV)", fontsize=12)
ax.set_title("NEB Energy Profile: O vacancy migration in MgO", fontsize=13)
ax.axhline(y=0, color="black", linestyle="-", linewidth=0.5)
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3)
fig.tight_layout()
fig.savefig("neb_energy_profile.png", dpi=150)
print("\nEnergy profile saved to neb_energy_profile.png")

# ============================================================
# 8. Save results
# ============================================================
results = {
    "system": "MgO",
    "defect": "O vacancy migration",
    "n_images": n_images,
    "forward_barrier_eV": barrier_forward,
    "reverse_barrier_eV": barrier_reverse,
    "reaction_coordinate": rc_norm,
    "relative_energies_eV": relative_energies,
    "transition_state_image": int(ts_idx),
}

with open("neb_results.json", "w") as f:
    json.dump(results, f, indent=2, default=float)
print("Results saved to neb_results.json")

# Save transition state structure
ts_struct = adaptor.get_structure(images[ts_idx])
ts_struct.to(filename="transition_state.cif")
print("Transition state structure saved to transition_state.cif")
```

#### Alternative: Using Linear Interpolation

```python
"""
For simple hops where IDPP is not needed,
use ASE's linear interpolation.
"""
from ase.mep.neb import interpolate

# After creating images list with copies of initial:
images = [atoms_initial.copy()]
for _ in range(n_images):
    images.append(atoms_initial.copy())
images.append(atoms_final.copy())

# Linear interpolation (simpler, but can cause atom overlap)
interpolate(images)

# Then proceed with NEB as above
```

#### Multi-Step NEB Strategy

```python
#!/usr/bin/env python3
"""
Robust multi-step NEB strategy for difficult paths.
1. Coarse NEB with few images and loose convergence
2. Refine with more images around the barrier
3. CI-NEB for accurate saddle point
"""

import numpy as np
from ase.mep.neb import NEB, NEBTools, idpp_interpolate
from ase.optimize import FIRE, BFGS
from mace.calculators import mace_mp


def run_neb_multistep(atoms_initial, atoms_final, calc_factory, logprefix="neb"):
    """
    Run NEB in multiple steps for robust convergence.

    Parameters
    ----------
    atoms_initial, atoms_final : ase.Atoms
        Relaxed endpoint structures.
    calc_factory : callable
        Function that returns a new calculator instance.
    logprefix : str
        Prefix for log files.

    Returns
    -------
    images : list of ase.Atoms
        Final NEB images.
    results : dict
        Barrier heights and convergence info.
    """
    # --- Phase 1: Coarse NEB with 5 images ---
    print("Phase 1: Coarse NEB (5 images, fmax=0.1)")
    n_coarse = 5
    images = [atoms_initial.copy()]
    for _ in range(n_coarse):
        images.append(atoms_initial.copy())
    images.append(atoms_final.copy())

    idpp_interpolate(images, traj=None, log=None)

    for img in images[1:-1]:
        img.calc = calc_factory()

    neb = NEB(images, climb=False, k=0.1)
    opt = FIRE(neb, logfile=f"{logprefix}_phase1.log")
    opt.run(fmax=0.1, steps=300)
    print(f"  Converged in {opt.nsteps} steps")

    # Find approximate barrier location
    energies = [img.get_potential_energy() for img in images]
    ts_idx = np.argmax(energies)
    barrier_coarse = energies[ts_idx] - energies[0]
    print(f"  Approximate barrier: {barrier_coarse:.3f} eV at image {ts_idx}")

    # --- Phase 2: Fine NEB with 9 images ---
    print("\nPhase 2: Fine NEB (9 images, fmax=0.05)")
    n_fine = 9
    images_fine = [atoms_initial.copy()]
    for _ in range(n_fine):
        images_fine.append(atoms_initial.copy())
    images_fine.append(atoms_final.copy())

    idpp_interpolate(images_fine, traj=None, log=None)

    for img in images_fine[1:-1]:
        img.calc = calc_factory()

    neb_fine = NEB(images_fine, climb=False, k=0.1)
    opt_fine = FIRE(neb_fine, logfile=f"{logprefix}_phase2.log")
    opt_fine.run(fmax=0.05, steps=500)
    print(f"  Converged in {opt_fine.nsteps} steps")

    # --- Phase 3: CI-NEB ---
    print("\nPhase 3: CI-NEB (climbing image, fmax=0.03)")
    neb_ci = NEB(images_fine, climb=True, k=0.1)
    opt_ci = FIRE(neb_ci, logfile=f"{logprefix}_phase3.log")
    opt_ci.run(fmax=0.03, steps=500)
    print(f"  Converged in {opt_ci.nsteps} steps")

    # Extract final results
    tools = NEBTools(images_fine)
    fwd_barrier, rev_barrier = tools.get_barrier()

    results = {
        "forward_barrier_eV": float(fwd_barrier),
        "reverse_barrier_eV": float(rev_barrier),
        "n_images": n_fine,
        "phase1_steps": opt.nsteps,
        "phase2_steps": opt_fine.nsteps,
        "phase3_steps": opt_ci.nsteps,
    }

    return images_fine, results


# Example usage:
# images, results = run_neb_multistep(
#     atoms_initial, atoms_final,
#     calc_factory=lambda: mace_mp(model="medium", dispersion=False, default_dtype="float64"),
# )
```

### Method B: QE NEB (neb.x)

#### Step 1: Generate NEB Input File

```python
#!/usr/bin/env python3
"""
Generate QE neb.x input file from two endpoint structures.
"""

import numpy as np
from pymatgen.core import Structure, Element
from pathlib import Path

# ============================================================
# Build endpoint structures (same as Method A)
# ============================================================
primitive = Structure.from_spacegroup(
    "Fm-3m",
    lattice=[[4.212, 0, 0], [0, 4.212, 0], [0, 0, 4.212]],
    species=["Mg", "O"],
    coords=[[0.0, 0.0, 0.0], [0.5, 0.5, 0.5]],
)

sc_size = 2  # smaller for DFT (2x2x2 = 16 atoms; 15 with vacancy)
bulk_sc = primitive.copy()
bulk_sc.make_supercell([sc_size, sc_size, sc_size])

# Two neighboring O sites
o_indices = [i for i, s in enumerate(bulk_sc) if s.specie == Element("O")]
site0 = bulk_sc[o_indices[0]]
dists = [(idx, site0.distance(bulk_sc[idx])) for idx in o_indices[1:]]
dists.sort(key=lambda x: x[1])
nn_idx = dists[0][0]

initial_struct = bulk_sc.copy()
initial_struct.remove_sites([o_indices[0]])

final_struct = bulk_sc.copy()
final_struct.remove_sites([nn_idx])


def struct_to_atomic_positions(struct, pseudo_map):
    """Convert pymatgen Structure to QE ATOMIC_POSITIONS block."""
    lines = []
    for site in struct:
        sp = str(site.specie)
        x, y, z = site.frac_coords
        lines.append(f"  {sp}  {x:.10f}  {y:.10f}  {z:.10f}")
    return "\n".join(lines)


def struct_to_cell_parameters(struct):
    """Convert pymatgen Structure to QE CELL_PARAMETERS block."""
    lines = ["CELL_PARAMETERS angstrom"]
    for vec in struct.lattice.matrix:
        lines.append(f"  {vec[0]:.10f}  {vec[1]:.10f}  {vec[2]:.10f}")
    return "\n".join(lines)


# ============================================================
# Write neb.x input
# ============================================================
pseudo_dir = "./pseudo"
pseudos = {
    "Mg": "Mg.pbe-spnl-kjpaw_psl.1.0.0.UPF",
    "O":  "O.pbe-n-kjpaw_psl.1.0.0.UPF",
}

n_images_neb = 7  # intermediate images for QE NEB
nat = len(initial_struct)

# Get unique species and their pseudopotentials
species_list = sorted(set(str(s.specie) for s in initial_struct))
ntyp = len(species_list)

# Atomic species block
atomic_species_lines = []
for sp in species_list:
    # Mass (approximate, QE uses these for dynamics)
    mass = Element(sp).atomic_mass
    atomic_species_lines.append(f"  {sp}  {mass:.4f}  {pseudos[sp]}")

neb_input = f"""BEGIN
BEGIN_PATH_INPUT
&PATH
  restart_mode  = 'from_scratch',
  string_method = 'neb',
  nstep_path    = 200,
  ds            = 1.0,
  opt_scheme    = 'broyden',
  num_of_images = {n_images_neb + 2},
  k_max         = 0.3,
  k_min         = 0.1,
  CI_scheme     = 'auto',
  path_thr      = 0.05,
/
END_PATH_INPUT

BEGIN_ENGINE_INPUT
&CONTROL
  prefix        = 'neb_vacancy',
  pseudo_dir    = '{pseudo_dir}',
  outdir        = './tmp',
/
&SYSTEM
  ibrav         = 0,
  nat           = {nat},
  ntyp          = {ntyp},
  ecutwfc       = 50.0,
  ecutrho       = 400.0,
  occupations   = 'smearing',
  smearing      = 'cold',
  degauss       = 0.01,
/
&ELECTRONS
  conv_thr      = 1.0e-6,
  mixing_beta   = 0.3,
/

ATOMIC_SPECIES
{chr(10).join(atomic_species_lines)}

{struct_to_cell_parameters(initial_struct)}

K_POINTS automatic
  2 2 2  0 0 0

BEGIN_POSITIONS
FIRST_IMAGE
ATOMIC_POSITIONS crystal
{struct_to_atomic_positions(initial_struct, pseudos)}

LAST_IMAGE
ATOMIC_POSITIONS crystal
{struct_to_atomic_positions(final_struct, pseudos)}
END_POSITIONS
END_ENGINE_INPUT
END
"""

with open("neb.in", "w") as f:
    f.write(neb_input)
print(f"Written neb.in ({nat} atoms, {n_images_neb + 2} total images)")
```

#### Step 2: Run QE NEB

```bash
#!/bin/bash
# Run QE NEB calculation

NPROC=4   # total MPI processes
NIMAGE=1  # number of image groups for parallelization
            # set to number of images for best parallel scaling
            # (each image group handles ceil(n_images/NIMAGE) images)

# Download pseudopotentials if needed
mkdir -p pseudo tmp
cd pseudo
wget -nc "https://pseudopotentials.quantum-espresso.org/upf_files/Mg.pbe-spnl-kjpaw_psl.1.0.0.UPF" 2>/dev/null
wget -nc "https://pseudopotentials.quantum-espresso.org/upf_files/O.pbe-n-kjpaw_psl.1.0.0.UPF" 2>/dev/null
cd ..

# Run NEB
echo "=== Running QE NEB ==="
mpirun -np $NPROC neb.x -ni $NIMAGE -in neb.in > neb.out 2>&1

# Check convergence
if grep -q "neb: convergence achieved" neb.out; then
    echo "NEB CONVERGED"
else
    echo "WARNING: NEB did not converge. Check neb.out"
fi

# Extract barrier from output
echo ""
echo "=== Energy along path ==="
grep "image" neb.out | grep "eV"
```

#### Step 3: Parse QE NEB Results

```python
#!/usr/bin/env python3
"""
Parse QE neb.x output and generate energy profile.
"""

import re
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import json

def parse_neb_output(filename):
    """
    Parse neb.x output file for energies and reaction coordinates.
    """
    energies = []
    positions = []
    converged = False

    with open(filename) as f:
        content = f.read()

    # Check convergence
    if "neb: convergence achieved" in content:
        converged = True

    # Parse the final path energies
    # Format: "     image        energy (eV)        error (eV/A)"
    # followed by lines like "     1        -1234.5678           0.0012"
    pattern = r"^\s+(\d+)\s+([-\d.]+)\s+([-\d.]+)"
    lines = content.split("\n")
    in_energy_block = False
    for line in lines:
        if "image" in line and "energy (eV)" in line:
            in_energy_block = True
            energies = []  # reset - take last block
            continue
        if in_energy_block:
            match = re.match(pattern, line)
            if match:
                img_num = int(match.group(1))
                energy = float(match.group(2))
                error = float(match.group(3))
                energies.append({
                    "image": img_num,
                    "energy_eV": energy,
                    "error_eV_per_A": error,
                })
            elif line.strip() == "" and energies:
                in_energy_block = False

    # Also try parsing the activation energy summary
    barrier_fwd = None
    barrier_rev = None
    for line in lines:
        if "activation energy" in line.lower() and "--->" in line:
            match = re.search(r"([-\d.]+)\s+eV", line)
            if match:
                barrier_fwd = float(match.group(1))
        if "activation energy" in line.lower() and "<---" in line:
            match = re.search(r"([-\d.]+)\s+eV", line)
            if match:
                barrier_rev = float(match.group(1))

    return {
        "converged": converged,
        "images": energies,
        "barrier_forward_eV": barrier_fwd,
        "barrier_reverse_eV": barrier_rev,
    }


# Parse results
results = parse_neb_output("neb.out")

if not results["images"]:
    print("ERROR: Could not parse NEB energies from neb.out")
    print("Check if the calculation completed successfully.")
    exit(1)

print(f"Converged: {results['converged']}")
print(f"Number of images: {len(results['images'])}")

# Energy profile
e_ref = results["images"][0]["energy_eV"]
image_nums = [img["image"] for img in results["images"]]
rel_energies = [img["energy_eV"] - e_ref for img in results["images"]]
errors = [img["error_eV_per_A"] for img in results["images"]]

print("\nImage | Rel. Energy (eV) | Force Error (eV/A)")
print("-" * 50)
for img, e, err in zip(image_nums, rel_energies, errors):
    marker = " <-- TS" if e == max(rel_energies) else ""
    print(f"  {img:3d}  |  {e:12.6f}       |  {err:10.6f}{marker}")

if results["barrier_forward_eV"] is not None:
    print(f"\nForward barrier: {results['barrier_forward_eV']:.4f} eV")
if results["barrier_reverse_eV"] is not None:
    print(f"Reverse barrier: {results['barrier_reverse_eV']:.4f} eV")

# Plot
fig, ax = plt.subplots(figsize=(8, 5))
rc_norm = np.linspace(0, 1, len(rel_energies))
ax.plot(rc_norm, rel_energies, "o-", color="steelblue", markersize=8, linewidth=2)

ts_idx = np.argmax(rel_energies)
ax.plot(rc_norm[ts_idx], rel_energies[ts_idx], "r*", markersize=15, zorder=5,
        label=f"TS ({rel_energies[ts_idx]:.3f} eV)")

try:
    from scipy.interpolate import CubicSpline
    cs = CubicSpline(rc_norm, rel_energies)
    rc_fine = np.linspace(0, 1, 200)
    ax.plot(rc_fine, cs(rc_fine), "--", color="gray", alpha=0.6)
except ImportError:
    pass

ax.set_xlabel("Reaction coordinate", fontsize=12)
ax.set_ylabel("Energy relative to initial (eV)", fontsize=12)
ax.set_title("QE NEB Energy Profile", fontsize=13)
ax.axhline(y=0, color="black", linestyle="-", linewidth=0.5)
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3)
fig.tight_layout()
fig.savefig("qe_neb_energy_profile.png", dpi=150)
print("\nPlot saved to qe_neb_energy_profile.png")

# Save JSON
with open("qe_neb_results.json", "w") as f:
    json.dump(results, f, indent=2, default=float)
print("Results saved to qe_neb_results.json")
```

### Method C: Hybrid MACE Screening + QE Refinement

```python
#!/usr/bin/env python3
"""
Use MACE NEB to quickly screen multiple migration paths,
then refine the most promising path with QE.
"""

import numpy as np
from pymatgen.core import Structure, Element
from pymatgen.io.ase import AseAtomsAdaptor
from ase.optimize import FIRE
from ase.mep.neb import NEB, NEBTools, idpp_interpolate
from mace.calculators import mace_mp


def quick_neb_barrier(atoms_initial, atoms_final, n_images=5, fmax=0.05):
    """
    Run a quick MACE NEB to estimate the barrier height.

    Returns
    -------
    float : forward barrier in eV
    """
    calc_factory = lambda: mace_mp(
        model="medium", dispersion=False, default_dtype="float64"
    )

    images = [atoms_initial.copy()]
    for _ in range(n_images):
        images.append(atoms_initial.copy())
    images.append(atoms_final.copy())

    idpp_interpolate(images, traj=None, log=None)
    for img in images[1:-1]:
        img.calc = calc_factory()

    neb = NEB(images, climb=True, k=0.1)
    opt = FIRE(neb, logfile=None)  # suppress output
    opt.run(fmax=fmax, steps=300)

    tools = NEBTools(images)
    return tools.get_barrier()[0]


# Example: screen 3 different vacancy hop paths
# (In practice, generate these from symmetry analysis)
adaptor = AseAtomsAdaptor()

primitive = Structure.from_spacegroup(
    "Fm-3m",
    lattice=[[4.212, 0, 0], [0, 4.212, 0], [0, 0, 4.212]],
    species=["Mg", "O"],
    coords=[[0.0, 0.0, 0.0], [0.5, 0.5, 0.5]],
)
bulk_sc = primitive.copy()
bulk_sc.make_supercell([3, 3, 3])

calc = mace_mp(model="medium", dispersion=False, default_dtype="float64")

o_indices = [i for i, s in enumerate(bulk_sc) if s.specie == Element("O")]
site0 = bulk_sc[o_indices[0]]

# Find nearest O neighbors (multiple paths)
dists = [(idx, site0.distance(bulk_sc[idx])) for idx in o_indices[1:]]
dists.sort(key=lambda x: x[1])

# Screen first 3 nearest-neighbor hops
print("Screening vacancy migration paths with MACE NEB:")
print("-" * 50)
barriers = []
for path_idx in range(min(3, len(dists))):
    nn_idx = dists[path_idx][0]
    nn_dist = dists[path_idx][1]

    initial = bulk_sc.copy()
    initial.remove_sites([o_indices[0]])
    final = bulk_sc.copy()
    final.remove_sites([nn_idx])

    # Quick relax endpoints
    ai = adaptor.get_atoms(initial)
    ai.calc = calc
    from ase.optimize import BFGS
    BFGS(ai, logfile=None).run(fmax=0.02, steps=100)

    af = adaptor.get_atoms(final)
    af.calc = calc
    BFGS(af, logfile=None).run(fmax=0.02, steps=100)

    barrier = quick_neb_barrier(ai, af)
    barriers.append((path_idx, nn_dist, barrier))
    print(f"  Path {path_idx}: hop distance = {nn_dist:.3f} A, barrier = {barrier:.3f} eV")

# Rank paths
barriers.sort(key=lambda x: x[2])
print(f"\nLowest barrier: Path {barriers[0][0]} ({barriers[0][2]:.3f} eV)")
print("Refine this path with QE NEB for publication quality.")
```

## Key Parameters

| Parameter | Recommended Value | Notes |
|---|---|---|
| Number of images | 5-9 (intermediate, between endpoints) | More images for complex paths; 7 is a good default |
| Spring constant k | 0.05-0.3 eV/A^2 | Lower k allows more path flexibility; too low causes image bunching |
| fmax (convergence) | 0.03-0.05 eV/A | Tighter for accurate barriers; 0.03 for CI-NEB |
| Climbing image | Always use for final result | Finds the true saddle point; start without climb, then turn on |
| Interpolation method | IDPP preferred over linear | IDPP respects interatomic distances, avoids atom overlap |
| Optimizer | FIRE preferred for NEB | More robust than BFGS for NEB; BFGS can oscillate |
| QE path_thr | 0.05 eV/A | Convergence threshold for forces along the path |
| QE CI_scheme | 'auto' | Automatically switch to CI-NEB when forces are low enough |
| QE num_of_images | 7-11 (total including endpoints) | Must include endpoints in the count for neb.x |

## Interpreting Results

1. **Barrier height**: The forward barrier (from initial state to transition state) determines the rate-limiting step. Barriers < 0.5 eV indicate fast diffusion at room temperature; > 1.5 eV means negligible diffusion below ~1000 K.
2. **Arrhenius estimate**: D ~ D_0 * exp(-E_barrier / k_B T), with D_0 ~ 10^-3 cm^2/s for vacancy diffusion in oxides. At 300 K, k_B T = 0.026 eV.
3. **Path asymmetry**: If forward and reverse barriers differ, the path connects inequivalent sites. Symmetric barriers indicate equivalent endpoints (e.g., vacancy hopping between equivalent lattice sites).
4. **Transition state geometry**: Examine the highest-energy image for bonding and coordination changes at the saddle point.
5. **MACE vs. QE**: MACE barriers are typically within 0.1-0.3 eV of DFT for well-represented systems. Always validate with QE for publication.

## Common Issues

| Issue | Solution |
|---|---|
| NEB does not converge | Reduce fmax; increase steps; try FIRE optimizer; check endpoint relaxation quality |
| Images bunch up near endpoints | Increase spring constant k; use IDPP interpolation; add more images |
| Path finds wrong saddle point | Try different initial path; use more images; check that endpoints are correct minima |
| Atom overlap in initial path | Use IDPP instead of linear interpolation; manually adjust problematic images |
| MACE gives unphysical barrier | System outside training data; fall back to QE NEB |
| QE NEB very slow | Reduce number of images; use image parallelization (-ni flag); use coarser k-mesh first |
| Negative barrier | Endpoints are not true minima; re-relax endpoints with tighter convergence |
| Different barriers from MACE vs QE | Expected; report QE value for publication. Use MACE for qualitative trends only. |
| neb.x crashes with "images are too close" | Increase ds parameter; reduce num_of_images; check input geometry |
