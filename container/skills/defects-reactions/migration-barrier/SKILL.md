# Migration Barrier Calculation (CI-NEB)

## When to Use

- Calculate diffusion barriers for vacancy, interstitial, or impurity migration
- Screen migration pathways to identify the rate-limiting step
- Extract activation energies for Arrhenius diffusion models
- Estimate diffusion coefficients and prefactors
- Compare migration mechanisms (vacancy-mediated, interstitial, interstitialcy, ring)
- Find minimum energy paths (MEP) for defect migration

## Method Selection

| Criterion | ASE + MACE NEB | QE NEB (neb.x) | VASP VTST NEB |
|---|---|---|---|
| Speed | Minutes | Hours to days | Hours to days |
| Accuracy | ~0.1-0.3 eV of DFT | DFT accuracy | DFT accuracy |
| Climbing image | Supported (ASE CI-NEB) | Supported (CI_scheme) | Supported (LCLIMB) |
| Image parallelism | Via Python multiprocessing | Built-in (-ni flag) | Built-in (separate dirs) |
| Use case | Quick screening, path ranking | Publication quality | Publication quality |
| Install | Pre-installed | Pre-installed (neb.x) | VASP + VTST scripts (future) |

**Decision flow:**

```
Quick barrier estimate for multiple pathways?
  --> ASE + MACE NEB (Method A): minutes, rank paths by barrier

Publication-quality single barrier?
  --> QE NEB (Method B) or VASP VTST NEB (Method C)

System outside MACE training data?
  --> QE or VASP NEB required

Hybrid: screen with MACE, refine with DFT?
  --> Method A first, then Method B or C for the lowest-barrier path
```

## Prerequisites

- pymatgen (structure manipulation, endpoint generation)
- ASE + mace-torch (Method A)
- Quantum ESPRESSO neb.x (Method B)
- SSSP pseudopotentials for QE
- VASP with VTST extensions (Method C -- future external access)
- Optional: `pip install pymatgen-diffusion` for diffusion path analysis

## Detailed Steps

### Method A: ASE + MACE NEB

#### Complete Workflow: Vacancy Migration with CI-NEB

```python
#!/usr/bin/env python3
"""
Vacancy migration barrier calculation using ASE + MACE CI-NEB.

Workflow:
  1. Build supercell with vacancy at two neighboring sites (endpoints)
  2. Relax both endpoints
  3. Interpolate intermediate images (IDPP)
  4. Run standard NEB to get approximate path
  5. Switch to CI-NEB for accurate saddle point
  6. Extract barrier, plot MEP, save transition state
"""

import numpy as np
from pymatgen.core import Structure, Element
from pymatgen.io.ase import AseAtomsAdaptor
from ase.optimize import BFGS, FIRE
from ase.filters import FrechetCellFilter
from ase.mep.neb import NEB, NEBTools, idpp_interpolate
from mace.calculators import mace_mp
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.interpolate import CubicSpline
import json

# ============================================================
# 1. Setup
# ============================================================
calc = mace_mp(model="medium", dispersion=False, default_dtype="float64")
adaptor = AseAtomsAdaptor()

# Host: MgO
primitive = Structure.from_spacegroup(
    "Fm-3m",
    lattice=[[4.212, 0, 0], [0, 4.212, 0], [0, 0, 4.212]],
    species=["Mg", "O"],
    coords=[[0.0, 0.0, 0.0], [0.5, 0.5, 0.5]],
)

# Relax primitive
atoms_prim = adaptor.get_atoms(primitive)
atoms_prim.calc = calc
ecf = FrechetCellFilter(atoms_prim)
BFGS(ecf, logfile=None).run(fmax=0.005, steps=500)
prim_relaxed = adaptor.get_structure(atoms_prim)

# Build 3x3x3 supercell
SC_SIZE = 3
bulk_sc = prim_relaxed.copy()
bulk_sc.make_supercell([SC_SIZE, SC_SIZE, SC_SIZE])
print(f"Supercell: {bulk_sc.formula}, {len(bulk_sc)} atoms")

# ============================================================
# 2. Create vacancy migration endpoints
# ============================================================
VACANCY_SPECIES = "O"

# Find target atoms for the vacancy hop
sp_indices = [i for i, s in enumerate(bulk_sc) if str(s.specie) == VACANCY_SPECIES]
site0 = bulk_sc[sp_indices[0]]

# Find nearest same-species neighbor (the hop destination)
dists = [(idx, site0.distance(bulk_sc[idx])) for idx in sp_indices[1:]]
dists.sort(key=lambda x: x[1])
nn_idx = dists[0][0]
hop_distance = dists[0][1]
print(f"Vacancy hop: site {sp_indices[0]} -> site {nn_idx}, distance = {hop_distance:.3f} A")

# Initial endpoint: vacancy at site 0
initial_struct = bulk_sc.copy()
initial_struct.remove_sites([sp_indices[0]])

# Final endpoint: vacancy at nearest neighbor
final_struct = bulk_sc.copy()
final_struct.remove_sites([nn_idx])

# ============================================================
# 3. Relax endpoints
# ============================================================
print("\nRelaxing endpoints...")
atoms_initial = adaptor.get_atoms(initial_struct)
atoms_initial.calc = calc
BFGS(atoms_initial, logfile="relax_endpoint_initial.log").run(fmax=0.005, steps=300)
e_initial = atoms_initial.get_potential_energy()
print(f"  Initial: E = {e_initial:.6f} eV")

atoms_final = adaptor.get_atoms(final_struct)
atoms_final.calc = calc
BFGS(atoms_final, logfile="relax_endpoint_final.log").run(fmax=0.005, steps=300)
e_final = atoms_final.get_potential_energy()
print(f"  Final:   E = {e_final:.6f} eV")
print(f"  Delta_E = {e_final - e_initial:.6f} eV")

# ============================================================
# 4. Set up NEB images with IDPP interpolation
# ============================================================
N_IMAGES = 7  # intermediate images (not counting endpoints)

images = [atoms_initial.copy()]
for _ in range(N_IMAGES):
    images.append(atoms_initial.copy())
images.append(atoms_final.copy())

# IDPP interpolation: respects interatomic distances, avoids clashes
idpp_interpolate(images, traj=None, log=None)
print(f"\nIDPP interpolation done: {N_IMAGES} intermediate images")

# Assign fresh calculator to each intermediate image
for img in images[1:-1]:
    img.calc = mace_mp(model="medium", dispersion=False, default_dtype="float64")

# ============================================================
# 5. Phase 1: Standard NEB (coarse convergence)
# ============================================================
print("\n=== Phase 1: Standard NEB ===")
neb = NEB(images, climb=False, k=0.1, parallel=False)
opt = FIRE(neb, logfile="neb_phase1.log")
opt.run(fmax=0.05, steps=500)
print(f"  Converged in {opt.nsteps} steps")

# Quick barrier check
energies_p1 = [img.get_potential_energy() for img in images]
barrier_p1 = max(energies_p1) - energies_p1[0]
print(f"  Approximate barrier: {barrier_p1:.4f} eV")

# ============================================================
# 6. Phase 2: CI-NEB (climbing image for accurate saddle point)
# ============================================================
print("\n=== Phase 2: CI-NEB ===")
neb_ci = NEB(images, climb=True, k=0.1, parallel=False)
opt_ci = FIRE(neb_ci, logfile="neb_phase2_cineb.log")
opt_ci.run(fmax=0.03, steps=500)
print(f"  Converged in {opt_ci.nsteps} steps")

# ============================================================
# 7. Extract results
# ============================================================
neb_tools = NEBTools(images)
barrier_fwd, barrier_rev = neb_tools.get_barrier()

energies = [img.get_potential_energy() for img in images]
e_ref = energies[0]
rel_energies = [e - e_ref for e in energies]

# Reaction coordinate (cumulative atomic displacement)
reaction_coord = [0.0]
for i in range(1, len(images)):
    diff = images[i].get_positions() - images[i-1].get_positions()
    d = np.sqrt(np.sum(diff**2))
    reaction_coord.append(reaction_coord[-1] + d)
rc_norm = np.array(reaction_coord) / reaction_coord[-1]

# Transition state
ts_idx = np.argmax(rel_energies)
ts_energy = rel_energies[ts_idx]

print(f"\n{'='*50}")
print(f"NEB Results: {VACANCY_SPECIES} vacancy migration in {primitive.composition.reduced_formula}")
print(f"{'='*50}")
print(f"Forward barrier:  {barrier_fwd:.4f} eV")
print(f"Reverse barrier:  {barrier_rev:.4f} eV")
print(f"Transition state: image {ts_idx}, E_rel = {ts_energy:.4f} eV")
print(f"Hop distance:     {hop_distance:.3f} A")

# ============================================================
# 8. Plot minimum energy path
# ============================================================
fig, ax = plt.subplots(figsize=(8, 5))

# Discrete image energies
ax.plot(rc_norm, rel_energies, "o", color="steelblue", markersize=9, zorder=4)

# Spline fit for smooth curve
cs = CubicSpline(rc_norm, rel_energies)
rc_fine = np.linspace(0, 1, 300)
ax.plot(rc_fine, cs(rc_fine), "-", color="steelblue", linewidth=2, alpha=0.7)

# Highlight transition state
ax.plot(rc_norm[ts_idx], ts_energy, "r*", markersize=16, zorder=5,
        label=f"TS ({ts_energy:.3f} eV)")

# Annotations
ax.annotate(f"$E_a^{{\\rightarrow}}$ = {barrier_fwd:.3f} eV",
            xy=(0.35, barrier_fwd * 0.7), fontsize=11, color="firebrick")
if abs(barrier_fwd - barrier_rev) > 0.01:
    ax.annotate(f"$E_a^{{\\leftarrow}}$ = {barrier_rev:.3f} eV",
                xy=(0.65, barrier_rev * 0.7), fontsize=11, color="navy")

ax.set_xlabel("Reaction coordinate", fontsize=13)
ax.set_ylabel("Energy relative to initial (eV)", fontsize=13)
ax.set_title(f"V_{VACANCY_SPECIES} migration in {primitive.composition.reduced_formula} (MACE CI-NEB)",
             fontsize=13)
ax.axhline(y=0, color="black", linewidth=0.5)
ax.legend(fontsize=11)
ax.grid(True, alpha=0.3)
fig.tight_layout()
fig.savefig("migration_barrier.png", dpi=150)
print("\nEnergy profile saved to migration_barrier.png")

# Save transition state structure
ts_struct = adaptor.get_structure(images[ts_idx])
ts_struct.to(filename="transition_state.cif")
print("Transition state saved to transition_state.cif")

# Save JSON results
results = {
    "system": primitive.composition.reduced_formula,
    "vacancy_species": VACANCY_SPECIES,
    "hop_distance_A": hop_distance,
    "n_images": N_IMAGES,
    "forward_barrier_eV": float(barrier_fwd),
    "reverse_barrier_eV": float(barrier_rev),
    "reaction_coordinate": rc_norm.tolist(),
    "relative_energies_eV": rel_energies,
    "transition_state_image": int(ts_idx),
}
with open("migration_barrier_results.json", "w") as f:
    json.dump(results, f, indent=2, default=float)
print("Results saved to migration_barrier_results.json")
```

#### Screen Multiple Migration Pathways

```python
#!/usr/bin/env python3
"""
Screen multiple migration pathways using MACE NEB.
Identifies the lowest-barrier path for subsequent DFT refinement.
"""

import numpy as np
from pymatgen.core import Structure, Element
from pymatgen.io.ase import AseAtomsAdaptor
from ase.optimize import FIRE, BFGS
from ase.mep.neb import NEB, NEBTools, idpp_interpolate
from mace.calculators import mace_mp
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import json

adaptor = AseAtomsAdaptor()
calc = mace_mp(model="medium", dispersion=False, default_dtype="float64")

# Build supercell
primitive = Structure.from_spacegroup(
    "Fm-3m",
    lattice=[[4.212, 0, 0], [0, 4.212, 0], [0, 0, 4.212]],
    species=["Mg", "O"],
    coords=[[0.0, 0.0, 0.0], [0.5, 0.5, 0.5]],
)
bulk_sc = primitive.copy()
bulk_sc.make_supercell([3, 3, 3])

VACANCY_SPECIES = "O"
sp_indices = [i for i, s in enumerate(bulk_sc) if str(s.specie) == VACANCY_SPECIES]
site0 = bulk_sc[sp_indices[0]]

# Find nearest neighbors of different shell distances
dists = [(idx, site0.distance(bulk_sc[idx])) for idx in sp_indices[1:]]
dists.sort(key=lambda x: x[1])

# Group by distance shells (within 0.1 A tolerance)
shells = []
current_shell = [dists[0]]
for d in dists[1:]:
    if abs(d[1] - current_shell[-1][1]) < 0.1:
        current_shell.append(d)
    else:
        shells.append(current_shell)
        current_shell = [d]
        if len(shells) >= 3:
            break
shells.append(current_shell)

# Screen one path per shell
def quick_neb(atoms_i, atoms_f, n_images=5, fmax=0.05):
    """Run a quick CI-NEB and return the forward barrier."""
    imgs = [atoms_i.copy()]
    for _ in range(n_images):
        imgs.append(atoms_i.copy())
    imgs.append(atoms_f.copy())

    idpp_interpolate(imgs, traj=None, log=None)
    for img in imgs[1:-1]:
        img.calc = mace_mp(model="medium", dispersion=False, default_dtype="float64")

    neb = NEB(imgs, climb=True, k=0.1)
    FIRE(neb, logfile=None).run(fmax=fmax, steps=300)
    return NEBTools(imgs).get_barrier()[0], imgs

print("Screening vacancy migration paths:")
print(f"{'Path':<8} {'Hop dist (A)':<15} {'Barrier (eV)':<15} {'Status'}")
print("-" * 50)

screening_results = []
all_path_images = {}

for shell_idx, shell in enumerate(shells[:3]):
    nn_idx, nn_dist = shell[0]  # take first member of each shell

    # Create endpoints
    initial = bulk_sc.copy()
    initial.remove_sites([sp_indices[0]])
    final = bulk_sc.copy()
    final.remove_sites([nn_idx])

    # Quick relax
    ai = adaptor.get_atoms(initial)
    ai.calc = calc
    BFGS(ai, logfile=None).run(fmax=0.01, steps=100)

    af = adaptor.get_atoms(final)
    af.calc = calc
    BFGS(af, logfile=None).run(fmax=0.01, steps=100)

    try:
        barrier, imgs = quick_neb(ai, af)
        status = "OK"
    except Exception as e:
        barrier = np.nan
        imgs = None
        status = f"FAILED: {e}"

    print(f"  {shell_idx:<6} {nn_dist:<15.3f} {barrier:<15.4f} {status}")
    screening_results.append({
        "path_index": shell_idx,
        "hop_distance_A": nn_dist,
        "barrier_eV": float(barrier),
        "target_site_index": nn_idx,
    })
    if imgs is not None:
        all_path_images[shell_idx] = imgs

# Rank paths
valid = [r for r in screening_results if not np.isnan(r["barrier_eV"])]
valid.sort(key=lambda x: x["barrier_eV"])

print(f"\nLowest barrier: Path {valid[0]['path_index']} "
      f"({valid[0]['barrier_eV']:.4f} eV, hop = {valid[0]['hop_distance_A']:.3f} A)")
print("Refine this path with QE or VASP NEB for publication quality.")

# Comparative plot
fig, ax = plt.subplots(figsize=(8, 5))
for shell_idx, imgs in all_path_images.items():
    energies = [img.get_potential_energy() for img in imgs]
    e_ref = energies[0]
    rel_e = [e - e_ref for e in energies]
    rc = np.linspace(0, 1, len(rel_e))
    d = screening_results[shell_idx]["hop_distance_A"]
    ax.plot(rc, rel_e, "o-", markersize=6, linewidth=1.5,
            label=f"Path {shell_idx} (d={d:.2f} A)")

ax.set_xlabel("Reaction coordinate", fontsize=12)
ax.set_ylabel("Energy (eV)", fontsize=12)
ax.set_title("Migration pathway comparison", fontsize=13)
ax.axhline(y=0, color="black", linewidth=0.5)
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3)
fig.tight_layout()
fig.savefig("pathway_comparison.png", dpi=150)
print("\nComparison plot saved to pathway_comparison.png")

with open("pathway_screening.json", "w") as f:
    json.dump(screening_results, f, indent=2, default=float)
```

#### Compute Diffusion Coefficient and Prefactor

```python
#!/usr/bin/env python3
"""
Estimate diffusion coefficient from migration barrier using the
Arrhenius relation and transition state theory.

D(T) = D_0 * exp(-E_a / k_B T)

where:
  D_0 = (1/2d) * lambda^2 * nu_0 * z
  d = dimensionality (3 for bulk)
  lambda = hop distance
  nu_0 = attempt frequency (from phonon calculation or estimated ~10^12-10^13 Hz)
  z = coordination number of migration paths
  E_a = activation energy (from CI-NEB)
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import json

# ============================================================
# Input: migration barrier from NEB calculation
# ============================================================
E_a = 2.4          # eV (forward barrier from CI-NEB)
hop_distance = 2.98  # Angstrom (O-O distance in MgO)
z = 12              # number of equivalent hop paths (fcc nearest neighbors)
nu_0 = 1.0e13      # Hz (typical attempt frequency for oxides)
d = 3               # dimensionality

kB = 8.617333262e-5  # eV/K

# ============================================================
# Diffusion prefactor
# ============================================================
lambda_cm = hop_distance * 1e-8  # A -> cm
D_0 = (1.0 / (2 * d)) * lambda_cm**2 * nu_0 * z
print(f"Diffusion parameters:")
print(f"  E_a = {E_a:.4f} eV")
print(f"  hop distance = {hop_distance:.3f} A")
print(f"  attempt frequency = {nu_0:.1e} Hz")
print(f"  coordination = {z}")
print(f"  D_0 = {D_0:.4e} cm^2/s")

# ============================================================
# D(T) over a temperature range
# ============================================================
temperatures = np.arange(300, 2500, 10)
D_T = D_0 * np.exp(-E_a / (kB * temperatures))

# Print at selected temperatures
print(f"\n{'T (K)':<10} {'D (cm^2/s)':<20} {'Mean free path (nm)':<25}")
print("-" * 55)
for T_target in [300, 500, 800, 1000, 1200, 1500, 2000]:
    D = D_0 * np.exp(-E_a / (kB * T_target))
    # Mean displacement in 1 hour: x = sqrt(2*d*D*t)
    t_hr = 3600  # 1 hour in seconds
    x_cm = np.sqrt(2 * d * D * t_hr)
    x_nm = x_cm * 1e7
    print(f"  {T_target:<8} {D:<20.4e} {x_nm:<25.4e}")

# ============================================================
# Arrhenius plot
# ============================================================
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

# Log(D) vs 1000/T
inv_T = 1000 / temperatures
ax1.semilogy(inv_T, D_T, "b-", linewidth=2)
ax1.set_xlabel("1000/T (K$^{-1}$)", fontsize=12)
ax1.set_ylabel("D (cm$^2$/s)", fontsize=12)
ax1.set_title(f"Arrhenius plot: $E_a$ = {E_a:.3f} eV", fontsize=13)
ax1.grid(True, alpha=0.3)

# Add temperature labels on top axis
ax1_top = ax1.twiny()
T_labels = [2000, 1500, 1000, 800, 600, 400]
ax1_top.set_xlim(ax1.get_xlim())
ax1_top.set_xticks([1000/T for T in T_labels])
ax1_top.set_xticklabels([str(T) for T in T_labels])
ax1_top.set_xlabel("Temperature (K)", fontsize=11)

# D vs T (linear scale)
ax2.semilogy(temperatures, D_T, "r-", linewidth=2)
ax2.set_xlabel("Temperature (K)", fontsize=12)
ax2.set_ylabel("D (cm$^2$/s)", fontsize=12)
ax2.set_title("Diffusion coefficient vs. temperature", fontsize=13)
ax2.grid(True, alpha=0.3)

# Reference lines
for D_ref, label in [(1e-12, "1e-12"), (1e-8, "1e-8"), (1e-4, "1e-4")]:
    ax2.axhline(y=D_ref, color="gray", linestyle=":", alpha=0.4)
    ax2.text(350, D_ref * 1.5, f"D = {label} cm$^2$/s", fontsize=8, color="gray")

fig.tight_layout()
fig.savefig("diffusion_arrhenius.png", dpi=150)
print("\nArrhenius plot saved to diffusion_arrhenius.png")

# Save
results = {
    "E_a_eV": E_a,
    "D_0_cm2_per_s": D_0,
    "hop_distance_A": hop_distance,
    "attempt_frequency_Hz": nu_0,
    "coordination": z,
}
with open("diffusion_parameters.json", "w") as f:
    json.dump(results, f, indent=2, default=float)
print("Parameters saved to diffusion_parameters.json")
```

### Method B: QE NEB (neb.x)

#### Step 1: Generate neb.x Input File

```python
#!/usr/bin/env python3
"""
Generate QE neb.x input file for vacancy migration barrier.
Uses relaxed endpoint structures from Method A or from separate QE relaxations.
"""

import numpy as np
from pymatgen.core import Structure, Element
from pathlib import Path

# ============================================================
# Build endpoint structures
# ============================================================
primitive = Structure.from_spacegroup(
    "Fm-3m",
    lattice=[[4.212, 0, 0], [0, 4.212, 0], [0, 0, 4.212]],
    species=["Mg", "O"],
    coords=[[0.0, 0.0, 0.0], [0.5, 0.5, 0.5]],
)

sc_size = 2  # 2x2x2 for DFT (keep smaller for tractability)
bulk_sc = primitive.copy()
bulk_sc.make_supercell([sc_size, sc_size, sc_size])

# Two neighboring O sites for vacancy hop
o_indices = [i for i, s in enumerate(bulk_sc) if s.specie == Element("O")]
site0 = bulk_sc[o_indices[0]]
dists = [(idx, site0.distance(bulk_sc[idx])) for idx in o_indices[1:]]
dists.sort(key=lambda x: x[1])
nn_idx = dists[0][0]

initial_struct = bulk_sc.copy()
initial_struct.remove_sites([o_indices[0]])

final_struct = bulk_sc.copy()
final_struct.remove_sites([nn_idx])

# ============================================================
# Helper functions
# ============================================================
def struct_to_atomic_positions(struct):
    lines = []
    for site in struct:
        sp = str(site.specie)
        x, y, z = site.frac_coords
        lines.append(f"  {sp}  {x:.10f}  {y:.10f}  {z:.10f}")
    return "\n".join(lines)

def struct_to_cell_parameters(struct):
    lines = ["CELL_PARAMETERS angstrom"]
    for vec in struct.lattice.matrix:
        lines.append(f"  {vec[0]:.10f}  {vec[1]:.10f}  {vec[2]:.10f}")
    return "\n".join(lines)

# ============================================================
# NEB input parameters
# ============================================================
pseudo_dir = "./pseudo"
pseudos = {
    "Mg": "Mg.pbe-spnl-kjpaw_psl.1.0.0.UPF",
    "O":  "O.pbe-n-kjpaw_psl.1.0.0.UPF",
}

N_IMAGES = 7  # intermediate images
nat = len(initial_struct)
species_list = sorted(set(str(s.specie) for s in initial_struct))
ntyp = len(species_list)

atomic_species_lines = []
for sp in species_list:
    mass = Element(sp).atomic_mass
    atomic_species_lines.append(f"  {sp}  {mass:.4f}  {pseudos[sp]}")

neb_input = f"""BEGIN
BEGIN_PATH_INPUT
&PATH
  restart_mode  = 'from_scratch',
  string_method = 'neb',
  nstep_path    = 300,
  ds            = 1.0,
  opt_scheme    = 'broyden',
  num_of_images = {N_IMAGES + 2},
  k_max         = 0.3,
  k_min         = 0.1,
  CI_scheme     = 'auto',
  path_thr      = 0.05,
/
END_PATH_INPUT

BEGIN_ENGINE_INPUT
&CONTROL
  prefix        = 'neb_migration',
  pseudo_dir    = '{pseudo_dir}',
  outdir        = './tmp',
/
&SYSTEM
  ibrav         = 0,
  nat           = {nat},
  ntyp          = {ntyp},
  ecutwfc       = 60.0,
  ecutrho       = 480.0,
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
{struct_to_atomic_positions(initial_struct)}

LAST_IMAGE
ATOMIC_POSITIONS crystal
{struct_to_atomic_positions(final_struct)}
END_POSITIONS
END_ENGINE_INPUT
END
"""

Path(pseudo_dir).mkdir(exist_ok=True)
Path("tmp").mkdir(exist_ok=True)

with open("neb.in", "w") as f:
    f.write(neb_input)
print(f"Written neb.in ({nat} atoms, {N_IMAGES + 2} total images)")
print(f"  CI_scheme = 'auto' (switches to CI-NEB when forces drop)")
print(f"  path_thr = 0.05 eV/A (force convergence along path)")
```

#### Step 2: Run QE NEB

```bash
#!/bin/bash
# Run QE NEB calculation for migration barrier
NPROC=4
NIMAGE=1  # Image parallelism: set to num_of_images for best scaling

# Download pseudopotentials
mkdir -p pseudo tmp
cd pseudo
wget -nc "https://pseudopotentials.quantum-espresso.org/upf_files/Mg.pbe-spnl-kjpaw_psl.1.0.0.UPF" 2>/dev/null
wget -nc "https://pseudopotentials.quantum-espresso.org/upf_files/O.pbe-n-kjpaw_psl.1.0.0.UPF" 2>/dev/null
cd ..

echo "=== Running QE NEB ==="
echo "This may take hours for DFT-quality barriers."
mpirun -np $NPROC neb.x -ni $NIMAGE -in neb.in > neb.out 2>&1

# Check convergence
if grep -q "neb: convergence achieved" neb.out; then
    echo "NEB CONVERGED"
else
    echo "WARNING: NEB did not converge. Check neb.out"
    echo "Consider increasing nstep_path or loosening path_thr."
fi

# Extract barrier
echo ""
echo "=== Path Energies ==="
grep "image" neb.out | grep "eV"

echo ""
echo "=== Activation Energies ==="
grep -i "activation energy" neb.out
```

#### Step 3: Parse QE NEB Results

```python
#!/usr/bin/env python3
"""
Parse QE neb.x output: extract energies, barriers, and plot MEP.
Also read intermediate image structures from the NEB output directory.
"""

import re
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.interpolate import CubicSpline
import json
import os

# ============================================================
# Parse neb.out
# ============================================================
def parse_neb_output(filename):
    energies = []
    converged = False
    barrier_fwd = None
    barrier_rev = None

    with open(filename) as f:
        content = f.read()

    if "neb: convergence achieved" in content:
        converged = True

    # Parse image energies (take the last block)
    pattern = r"^\s+(\d+)\s+([-\d.]+)\s+([-\d.]+)"
    lines = content.split("\n")
    in_block = False
    for line in lines:
        if "image" in line and "energy (eV)" in line:
            in_block = True
            energies = []
            continue
        if in_block:
            match = re.match(pattern, line)
            if match:
                energies.append({
                    "image": int(match.group(1)),
                    "energy_eV": float(match.group(2)),
                    "error_eV_A": float(match.group(3)),
                })
            elif line.strip() == "" and energies:
                in_block = False

    # Parse activation energies
    for line in lines:
        if "activation energy" in line.lower() and "--->" in line:
            m = re.search(r"([-\d.]+)\s+eV", line)
            if m:
                barrier_fwd = float(m.group(1))
        if "activation energy" in line.lower() and "<---" in line:
            m = re.search(r"([-\d.]+)\s+eV", line)
            if m:
                barrier_rev = float(m.group(1))

    return {
        "converged": converged,
        "images": energies,
        "barrier_forward_eV": barrier_fwd,
        "barrier_reverse_eV": barrier_rev,
    }

results = parse_neb_output("neb.out")

if not results["images"]:
    print("ERROR: No image energies found in neb.out")
    exit(1)

print(f"Converged: {results['converged']}")
print(f"Images: {len(results['images'])}")

# ============================================================
# Energy profile
# ============================================================
e_ref = results["images"][0]["energy_eV"]
rel_energies = [img["energy_eV"] - e_ref for img in results["images"]]
image_nums = [img["image"] for img in results["images"]]
errors = [img["error_eV_A"] for img in results["images"]]

print("\n  Image | Rel. Energy (eV) | Force Error (eV/A)")
print("  " + "-" * 50)
for n, e, err in zip(image_nums, rel_energies, errors):
    marker = " <-- TS" if e == max(rel_energies) else ""
    print(f"  {n:5d}  | {e:12.6f}       | {err:10.6f}{marker}")

if results["barrier_forward_eV"] is not None:
    print(f"\nForward barrier: {results['barrier_forward_eV']:.4f} eV")
if results["barrier_reverse_eV"] is not None:
    print(f"Reverse barrier: {results['barrier_reverse_eV']:.4f} eV")

# ============================================================
# Plot
# ============================================================
rc = np.linspace(0, 1, len(rel_energies))

fig, ax = plt.subplots(figsize=(8, 5))
ax.plot(rc, rel_energies, "o", color="steelblue", markersize=9, zorder=4)

# Spline fit
cs = CubicSpline(rc, rel_energies)
rc_fine = np.linspace(0, 1, 300)
ax.plot(rc_fine, cs(rc_fine), "-", color="steelblue", linewidth=2, alpha=0.7)

ts_idx = np.argmax(rel_energies)
ax.plot(rc[ts_idx], rel_energies[ts_idx], "r*", markersize=16, zorder=5,
        label=f"TS ({rel_energies[ts_idx]:.3f} eV)")

ax.set_xlabel("Reaction coordinate", fontsize=13)
ax.set_ylabel("Energy (eV)", fontsize=13)
ax.set_title("QE NEB Migration Barrier", fontsize=14)
ax.axhline(y=0, color="black", linewidth=0.5)
ax.legend(fontsize=11)
ax.grid(True, alpha=0.3)
fig.tight_layout()
fig.savefig("qe_neb_migration_barrier.png", dpi=150)
print("\nPlot saved to qe_neb_migration_barrier.png")

with open("qe_neb_results.json", "w") as f:
    json.dump(results, f, indent=2, default=float)
print("Results saved to qe_neb_results.json")
```

### Method C: VASP VTST NEB (Future External Access)

#### Generate VASP NEB Input Files

```python
#!/usr/bin/env python3
"""
Generate VASP NEB input files using the VTST (VASP Transition State Tools) format.
VASP NEB uses separate directories for each image: 00/, 01/, ..., N+1/

Note: VASP execution will be available via future external access.
This script generates the input file structure.
"""

import numpy as np
from pymatgen.core import Structure, Element
from pymatgen.io.vasp import Incar, Poscar, Kpoints
from pymatgen.io.ase import AseAtomsAdaptor
from ase.mep.neb import idpp_interpolate
from mace.calculators import mace_mp
from ase.optimize import BFGS
from pathlib import Path
import json

# ============================================================
# Build endpoints
# ============================================================
primitive = Structure.from_spacegroup(
    "Fm-3m",
    lattice=[[4.212, 0, 0], [0, 4.212, 0], [0, 0, 4.212]],
    species=["Mg", "O"],
    coords=[[0.0, 0.0, 0.0], [0.5, 0.5, 0.5]],
)

sc_size = 2
bulk_sc = primitive.copy()
bulk_sc.make_supercell([sc_size, sc_size, sc_size])

o_indices = [i for i, s in enumerate(bulk_sc) if s.specie == Element("O")]
site0 = bulk_sc[o_indices[0]]
dists = [(idx, site0.distance(bulk_sc[idx])) for idx in o_indices[1:]]
dists.sort(key=lambda x: x[1])
nn_idx = dists[0][0]

initial_struct = bulk_sc.copy()
initial_struct.remove_sites([o_indices[0]])
final_struct = bulk_sc.copy()
final_struct.remove_sites([nn_idx])

N_IMAGES = 5  # intermediate images

# ============================================================
# Generate interpolated images using IDPP (via ASE + MACE)
# ============================================================
adaptor = AseAtomsAdaptor()
calc = mace_mp(model="medium", dispersion=False, default_dtype="float64")

# Quick relax endpoints with MACE
atoms_i = adaptor.get_atoms(initial_struct)
atoms_i.calc = calc
BFGS(atoms_i, logfile=None).run(fmax=0.01, steps=100)

atoms_f = adaptor.get_atoms(final_struct)
atoms_f.calc = calc
BFGS(atoms_f, logfile=None).run(fmax=0.01, steps=100)

# IDPP interpolation
images = [atoms_i.copy()]
for _ in range(N_IMAGES):
    images.append(atoms_i.copy())
images.append(atoms_f.copy())
idpp_interpolate(images, traj=None, log=None)

# ============================================================
# Write VASP NEB directory structure
# ============================================================
neb_dir = Path("vasp_neb")
neb_dir.mkdir(exist_ok=True)

# Write INCAR
incar = Incar({
    "SYSTEM": f"NEB V_O migration in {primitive.composition.reduced_formula}",
    "ENCUT": 520,
    "PREC": "Accurate",
    "EDIFF": 1e-6,
    "EDIFFG": -0.03,         # force convergence for NEB
    "IBRION": 3,             # damped MD (recommended for VTST NEB)
    "POTIM": 0.0,            # VTST handles the step size
    "ISIF": 2,               # relax ions, fixed cell
    "NSW": 500,
    "ISMEAR": 0,
    "SIGMA": 0.05,
    "LREAL": "Auto",
    "LWAVE": False,
    "LCHARG": False,
    "ALGO": "Normal",
    "NELM": 200,
    # VTST NEB parameters
    "IMAGES": N_IMAGES,       # number of intermediate images
    "SPRING": -5.0,           # spring constant (eV/A^2; negative = variable spring)
    "LCLIMB": True,           # climbing image NEB
    "IOPT": 3,                # optimizer: 3 = L-BFGS (VTST)
    "ICHAIN": 0,              # 0 = NEB
})
incar.write_file(str(neb_dir / "INCAR"))
print(f"Written {neb_dir}/INCAR")

# Write KPOINTS
kpoints = Kpoints.gamma_automatic(kpts=(2, 2, 2), shift=(0, 0, 0))
kpoints.write_file(str(neb_dir / "KPOINTS"))

# Write image directories (00 = initial, 01-N = intermediate, N+1 = final)
for i, img in enumerate(images):
    img_dir = neb_dir / f"{i:02d}"
    img_dir.mkdir(exist_ok=True)
    struct = adaptor.get_structure(img)
    Poscar(struct).write_file(str(img_dir / "POSCAR"))
    print(f"  Written {img_dir}/POSCAR")

print(f"""
=== VASP VTST NEB Setup ===
Directory structure:
  {neb_dir}/
    INCAR
    KPOINTS
    POTCAR  (generate with: cat POTCAR_Mg POTCAR_O > POTCAR)
    00/POSCAR  (initial endpoint - fixed)
    01/POSCAR  (image 1 - optimized)
    ...
    {N_IMAGES:02d}/POSCAR  (image {N_IMAGES} - optimized)
    {N_IMAGES+1:02d}/POSCAR  (final endpoint - fixed)

Run with:
  cd {neb_dir}
  mpirun -np $NPROC vasp_std

VTST NEB parameters in INCAR:
  IMAGES = {N_IMAGES}    (intermediate images)
  LCLIMB = .TRUE.  (climbing image NEB)
  IOPT = 3         (L-BFGS optimizer from VTST)
  SPRING = -5.0    (variable spring constant)

Parallelism:
  NPROC should be divisible by IMAGES for image parallelism.
  E.g., IMAGES=5 with NPROC=20 gives 4 cores per image.
""")
```

#### Parse VASP NEB Results

```python
#!/usr/bin/env python3
"""
Parse VASP VTST NEB results.
Reads energies from each image's OUTCAR or from the nebef.dat file
generated by VTST tools.

Note: For use when VASP is available via external access.
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.interpolate import CubicSpline
import json
import os

N_IMAGES = 5
neb_dir = "vasp_neb"

print("=== Parsing VASP NEB Results ===")
print("(Uncomment the parsing code when VASP outputs are available)")

# Example parsing code (uncomment when VASP outputs exist):
#
# from pymatgen.io.vasp import Outcar
#
# energies = []
# for i in range(N_IMAGES + 2):
#     outcar_path = os.path.join(neb_dir, f"{i:02d}", "OUTCAR")
#     if os.path.exists(outcar_path):
#         outcar = Outcar(outcar_path)
#         energies.append(outcar.final_energy)
#     else:
#         print(f"WARNING: {outcar_path} not found")
#
# # Alternative: parse nebef.dat from VTST
# nebef_path = os.path.join(neb_dir, "nebef.dat")
# if os.path.exists(nebef_path):
#     data = np.loadtxt(nebef_path)
#     # Columns: image_number, energy, force_along_path
#     energies = data[:, 1]
#
# # Compute barrier
# e_ref = energies[0]
# rel_energies = [e - e_ref for e in energies]
# barrier_fwd = max(rel_energies)
# barrier_rev = max(rel_energies) - rel_energies[-1]
#
# print(f"Forward barrier: {barrier_fwd:.4f} eV")
# print(f"Reverse barrier: {barrier_rev:.4f} eV")

print("""
VASP NEB output files:
  - OUTCAR in each image directory (00/ through {N+1}/)
  - nebef.dat (if VTST tools): energy and forces along the path
  - vasprun.xml in each image directory

Parse with pymatgen:
  from pymatgen.io.vasp import Outcar
  outcar = Outcar("vasp_neb/03/OUTCAR")
  energy = outcar.final_energy
""")
```

## Key Parameters

| Parameter | Recommended Value | Notes |
|---|---|---|
| Number of images | 5-9 intermediate (7 is good default) | More images for complex paths; fewer for quick screening |
| Spring constant k | 0.05-0.3 eV/A^2 (ASE), -5.0 (VASP VTST) | Negative in VASP = variable spring; too low causes bunching |
| fmax convergence | 0.03 eV/A for CI-NEB | Tighter for accurate barriers; 0.05 for initial NEB |
| Climbing image | Always use for final barrier | CI-NEB finds the true saddle point |
| Interpolation | IDPP strongly preferred over linear | IDPP avoids atom overlap and gives better initial path |
| Optimizer | FIRE for ASE NEB | More robust than BFGS for NEB optimization |
| QE path_thr | 0.05 eV/A | Force convergence along the path |
| QE CI_scheme | 'auto' | Automatically enables CI when forces are low enough |
| VASP IOPT | 3 (L-BFGS) or 7 (FIRE) | VTST optimizer choice; 3 is generally reliable |
| VASP LCLIMB | .TRUE. | Enable climbing image in VTST NEB |
| Supercell size | >= 2x2x2, ideally 3x3x3 | Migrating atom must not interact with its periodic image |

## Interpreting Results

1. **Barrier height and diffusion regime**:
   - < 0.3 eV: Very fast diffusion, even at room temperature
   - 0.3-1.0 eV: Moderate diffusion, significant above 300-500 K
   - 1.0-2.0 eV: Slow diffusion, relevant only at high T (>800 K)
   - > 2.0 eV: Negligible diffusion below ~1500 K

2. **Forward vs. reverse barrier**: Asymmetric barriers indicate inequivalent endpoints. The rate-limiting step is determined by the higher barrier in a multi-hop path.

3. **Transition state geometry**: The climbing image reveals the saddle-point configuration. Examine bond lengths and coordination to understand the migration mechanism (e.g., through a bottleneck between atoms).

4. **Arrhenius relation**: D = D_0 * exp(-E_a / kBT). At 300 K, kBT = 0.026 eV. An activation energy of 1 eV means the diffusion rate is suppressed by exp(-38) compared to the prefactor.

5. **Attempt frequency**: Estimated from phonon frequencies at the initial site, or approximated as 10^12-10^13 Hz for most solid-state processes. More accurate values come from harmonic transition state theory.

6. **MACE vs. DFT**: MACE NEB barriers typically agree with DFT within 0.1-0.3 eV for well-represented systems. Use MACE for screening and ranking pathways, DFT for publication-quality barriers.

7. **Multi-step paths**: If migration involves several elementary hops (e.g., through multiple inequivalent sites), compute each hop separately. The overall barrier is determined by the highest cumulative energy along the full path.

## Common Issues

| Issue | Solution |
|---|---|
| NEB does not converge | Use FIRE optimizer; reduce fmax; increase max steps; check endpoint quality |
| Images bunch near endpoints | Increase spring constant; use IDPP interpolation; add more images |
| Path finds wrong saddle point | Try different initial interpolation; add intermediate endpoint; increase images |
| Atom overlap in initial path | Use IDPP instead of linear interpolation; manually inspect and fix |
| MACE gives unphysical barrier | System outside training data; use DFT NEB |
| QE NEB very slow | Use image parallelism (-ni flag); reduce images for initial run; coarser k-mesh |
| VASP NEB: images not equally spaced | Expected with variable spring (SPRING < 0); images cluster near the TS |
| Negative barrier | Endpoints are not true minima; re-relax with tighter convergence |
| CI-NEB does not improve barrier | Ensure standard NEB is reasonably converged first; CI-NEB refines, not rescues |
| neb.x: "images are too close" | Increase ds parameter; reduce num_of_images; check geometry |
| Different barriers from MACE vs. QE | Report QE for publication; MACE is for trends and screening |
| Asymmetric path for equivalent sites | Check supercell symmetry; endpoints may not be truly equivalent due to PBC effects |
