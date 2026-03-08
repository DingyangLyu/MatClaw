# Special Quasirandom Structure (SQS) Generation

## When to Use

- You need to model a **random substitutional alloy** (e.g., Ti0.5Al0.5N, Cu0.75Au0.25) but DFT/MACE requires a periodic supercell.
- You want the smallest periodic cell whose **correlation functions** best match the ideal random alloy (all pair, triplet, ... correlations equal zero for equiatomic; or equal the appropriate analytical values for off-stoichiometry).
- You plan to compute mixing energies, lattice parameters, elastic constants, or electronic structure of a disordered alloy.
- You need to **sweep across a composition range** (e.g., A_{1-x}B_x for x = 0.1, 0.2, ..., 0.9) and generate SQS at each concentration.
- You want to generate **multiple SQS replicas** at each composition to quantify statistical uncertainty in computed properties.
- You need a complete **batch workflow**: generate SQS -> relax with MACE -> compute properties -> plot vs composition.

## Method Selection

| Method | Tool | Pros | Cons |
|--------|------|------|------|
| **icet + Monte Carlo** | `icet` | Rigorous cluster-correlation matching; fast MC optimization; handles arbitrary lattices; built-in quality metrics | Requires `pip install icet` |
| **sqsgenerator** | `sqsgenerator` | Purpose-built SQS tool, parallel, shell-specific weights, objective function targeting | Requires `pip install sqsgenerator` |
| **pymatgen SQSTransformation** | `pymatgen` | No extra install; integrated with pymatgen Structures | Slower for large cells; fewer options; may need external ATAT |
| **Manual enumeration** | `pymatgen` + custom code | Full control | Impractical for large cells |

**Recommendation:** Use **icet** for production-quality SQS. It is well-documented, fast, and gives access to correlation functions for validation. For multi-concentration sweeps with shell-weighted optimization, **sqsgenerator** is an excellent alternative with native parallelism.

## Prerequisites

```bash
pip install icet sqsgenerator
# Already available: ase, pymatgen, mace-torch, numpy, matplotlib
```

## Theory Summary

### What Is an SQS?

An SQS is a periodic supercell of N atoms whose **correlation functions** best approximate those of a perfectly random alloy at a given composition. For a binary A_{1-x}B_x alloy on a single sublattice, we assign a pseudo-spin variable sigma_i = +1 (species B) or -1 (species A) to each site. The correlation function of a cluster alpha (pair, triplet, ...) is:

```
Pi_alpha = <sigma_i * sigma_j * ...>_alpha
```

averaged over all symmetry-equivalent clusters of type alpha. For a perfectly random alloy at concentration x of species B:

```
Pi_pair   = (2x - 1)^2        (target for all pair clusters)
Pi_triplet = (2x - 1)^3       (target for all triplet clusters)
Pi_n-body = (2x - 1)^n        (general n-body target)
```

The SQS objective is to minimize the weighted sum of squared differences between the structure's correlation functions and these random-alloy targets.

### Objective Function and Weights

The **sqsgenerator** approach (used by pyiron's SQSJob) defines an objective function:

```
Objective = sum_shell  w_shell * |Pi_shell(structure) - Pi_target|^2
```

where `w_shell` weights each neighbor shell. Default weights are 1/shell_number (i.e., [1, 1/2, 1/3, 1/4, ...]), prioritizing short-range correlations. Setting `objective = 0` targets a perfectly random alloy where all correlations match analytical values.

## Detailed Steps

### Method A: SQS Generation with icet (Recommended)

The workflow:
1. Define a parent lattice (e.g., FCC Cu).
2. Build a cluster space that enumerates pair, triplet correlations up to chosen cutoff radii.
3. Generate a supercell of desired size.
4. Run Monte Carlo simulated-annealing to minimize the difference between the structure's correlations and the random-alloy target correlations.
5. Validate: compare correlation functions.

```python
#!/usr/bin/env python3
"""
SQS generation for Cu0.75Au0.25 FCC alloy using icet.
Produces a validated SQS, relaxes with MACE, and computes mixing energy.
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ============================================================
# Step 1: Define parent lattice and cluster space
# ============================================================
from ase.build import bulk
from icet import ClusterSpace
from icet.tools import enumerate_structures
from icet.tools.structure_generation import generate_sqs_from_supercells

# FCC Cu as parent lattice (Cu sites will host Cu or Au)
parent = bulk("Cu", crystalstructure="fcc", a=3.80)  # approximate avg lattice param

# Cluster space: pairs up to 6.0 A, triplets up to 4.5 A
# Chemical symbols list: each sublattice lists the allowed species
cluster_space = ClusterSpace(
    structure=parent,
    cutoffs=[6.0, 4.5],        # [pair_cutoff, triplet_cutoff] in Angstrom
    chemical_symbols=[["Cu", "Au"]],
)
print("== Cluster Space ==")
print(cluster_space)
print(f"Number of orbits: {cluster_space.number_of_orbits}")

# ============================================================
# Step 2: Define supercell matrices to search over
# ============================================================
from ase.build import make_supercell

# Several candidate supercell shapes (targeting ~32 atoms)
supercell_matrices = [
    [[4, 0, 0], [0, 4, 0], [0, 0, 2]],   # 32 atoms
    [[4, 0, 0], [0, 2, 0], [0, 0, 4]],   # 32 atoms
    [[2, 2, 0], [2, 0, 2], [0, 2, 2]],   # 32 atoms (more isotropic)
    [[3, 0, 0], [0, 3, 0], [0, 0, 3]],   # 27 atoms
]

supercells = []
for sc_matrix in supercell_matrices:
    sc = make_supercell(parent, sc_matrix)
    supercells.append(sc)
    print(f"Supercell with {len(sc)} atoms, matrix = {sc_matrix}")

# ============================================================
# Step 3: Generate SQS via Monte Carlo optimization
# ============================================================
# Target composition: Cu0.75Au0.25
target_concentrations = {"Cu": 0.75, "Au": 0.25}

print("\nGenerating SQS (this may take a minute)...")
sqs = generate_sqs_from_supercells(
    cluster_space=cluster_space,
    supercells=supercells,
    target_concentrations=target_concentrations,
    n_steps=50000,        # MC optimization steps per supercell
    random_seed=42,
)

n_cu = sum(1 for s in sqs.get_chemical_symbols() if s == "Cu")
n_au = sum(1 for s in sqs.get_chemical_symbols() if s == "Au")
print(f"\nSQS generated: {len(sqs)} atoms, {n_cu} Cu + {n_au} Au")
print(f"Actual composition: Cu={n_cu/len(sqs):.3f}, Au={n_au/len(sqs):.3f}")

# ============================================================
# Step 4: Validate -- compare cluster correlations to random alloy
# ============================================================
from icet import ClusterVectorCalculator

# Cluster vector of the SQS
cv_calculator = ClusterVectorCalculator(cluster_space)
# Older icet: use cluster_space.get_cluster_vector(sqs)
try:
    cv_sqs = cluster_space.get_cluster_vector(sqs)
except AttributeError:
    cv_sqs = cv_calculator.get_cluster_vector(sqs)

# Target cluster vector for a perfectly random alloy at this composition
# For a binary A_{1-x}B_x alloy on a single sublattice, the target
# pair correlation = (2x - 1)^2 for pairs, etc. icet provides a helper:
from icet.tools.structure_generation import _get_sqs_cluster_vector
try:
    cv_target = _get_sqs_cluster_vector(
        cluster_space=cluster_space,
        target_concentrations=target_concentrations,
    )
except Exception:
    # Manual calculation: for concentration x of species mapped to spin -1/+1
    # sigma_avg = 2*x_Au - 1 = 2*0.25 - 1 = -0.5
    # pair correlation target = sigma_avg^2 = 0.25
    # triplet correlation target = sigma_avg^3 = -0.125
    sigma_avg = 2 * target_concentrations["Au"] - 1  # icet convention
    n_orbits = cluster_space.number_of_orbits
    cv_target = np.zeros(n_orbits)
    cv_target[0] = 1.0  # zerolet (always 1)
    orbit_data = cluster_space.orbit_data
    for i, orb in enumerate(orbit_data):
        order = orb["order"]
        cv_target[i + 1] = sigma_avg ** order

print("\n== Correlation Function Comparison ==")
print(f"{'Orbit':>6} {'Order':>6} {'SQS':>10} {'Random':>10} {'Delta':>10}")
orbit_data = cluster_space.orbit_data
for i, orb in enumerate(orbit_data):
    idx = i + 1  # skip zerolet at index 0
    delta = cv_sqs[idx] - cv_target[idx]
    print(f"{idx:>6d} {orb['order']:>6d} {cv_sqs[idx]:>10.4f} {cv_target[idx]:>10.4f} {delta:>10.4f}")

# ============================================================
# Step 5: Visualize correlation comparison
# ============================================================
orbit_indices = list(range(1, len(cv_sqs)))
fig, ax = plt.subplots(figsize=(8, 4))
ax.bar(
    [i - 0.15 for i in orbit_indices],
    [cv_sqs[i] for i in orbit_indices],
    width=0.3, label="SQS", color="steelblue",
)
ax.bar(
    [i + 0.15 for i in orbit_indices],
    [cv_target[i] for i in orbit_indices],
    width=0.3, label="Random target", color="salmon",
)
ax.set_xlabel("Orbit index")
ax.set_ylabel("Correlation function")
ax.set_title("SQS vs Random Alloy Correlation Functions (Cu$_{0.75}$Au$_{0.25}$)")
ax.legend()
ax.axhline(0, color="k", linewidth=0.5)
plt.tight_layout()
plt.savefig("sqs_correlation_comparison.png", dpi=150)
print("\nSaved: sqs_correlation_comparison.png")

# ============================================================
# Step 6: Relax SQS with MACE and compute mixing energy
# ============================================================
from ase.optimize import BFGS
from ase.constraints import ExpCellFilter
from mace.calculators import mace_mp

print("\nRelaxing SQS with MACE...")
calc = mace_mp(model="medium", default_dtype="float64")
sqs.calc = calc

# Full relaxation (cell + positions)
ecf = ExpCellFilter(sqs)
opt = BFGS(ecf, logfile="sqs_relax.log")
opt.run(fmax=0.01, steps=500)

e_sqs = sqs.get_potential_energy()
print(f"SQS total energy: {e_sqs:.4f} eV ({len(sqs)} atoms)")
print(f"SQS energy/atom:  {e_sqs / len(sqs):.4f} eV/atom")

# Reference energies: pure Cu and pure Au (FCC)
cu_pure = bulk("Cu", crystalstructure="fcc", a=3.615)
cu_pure.calc = calc
ecf_cu = ExpCellFilter(cu_pure)
opt_cu = BFGS(ecf_cu, logfile="cu_relax.log")
opt_cu.run(fmax=0.01)
e_cu = cu_pure.get_potential_energy() / len(cu_pure)

au_pure = bulk("Au", crystalstructure="fcc", a=4.078)
au_pure.calc = calc
ecf_au = ExpCellFilter(au_pure)
opt_au = BFGS(ecf_au, logfile="au_relax.log")
opt_au.run(fmax=0.01)
e_au = au_pure.get_potential_energy() / len(au_pure)

print(f"\nPure Cu energy/atom: {e_cu:.4f} eV")
print(f"Pure Au energy/atom: {e_au:.4f} eV")

# Mixing energy: E_mix = E_alloy/atom - x_Cu * E_Cu - x_Au * E_Au
x_cu = n_cu / len(sqs)
x_au = n_au / len(sqs)
e_mix = e_sqs / len(sqs) - x_cu * e_cu - x_au * e_au
print(f"\nMixing energy: {e_mix * 1000:.1f} meV/atom")
print(f"  (positive = endothermic/phase-separating, negative = exothermic/ordering)")

# ============================================================
# Step 7: Visualize the SQS structure
# ============================================================
from ase.io import write

# Write structure files
write("sqs_Cu75Au25.cif", sqs)
write("sqs_Cu75Au25.vasp", sqs, format="vasp")
print("\nSaved: sqs_Cu75Au25.cif, sqs_Cu75Au25.vasp")

# 2D projection plot
fig2, ax2 = plt.subplots(figsize=(6, 6))
positions = sqs.get_positions()
symbols = sqs.get_chemical_symbols()
colors = ["#1f77b4" if s == "Cu" else "#d4af37" for s in symbols]
sizes = [40 if s == "Cu" else 60 for s in symbols]
ax2.scatter(positions[:, 0], positions[:, 1], c=colors, s=sizes, edgecolors="k", linewidth=0.5)
ax2.set_xlabel("x (A)")
ax2.set_ylabel("y (A)")
ax2.set_title("SQS Cu$_{0.75}$Au$_{0.25}$ (projection onto xy)")
# Legend
from matplotlib.lines import Line2D
legend_elements = [
    Line2D([0], [0], marker="o", color="w", markerfacecolor="#1f77b4", markersize=8, label="Cu"),
    Line2D([0], [0], marker="o", color="w", markerfacecolor="#d4af37", markersize=10, label="Au"),
]
ax2.legend(handles=legend_elements)
ax2.set_aspect("equal")
plt.tight_layout()
plt.savefig("sqs_structure.png", dpi=150)
print("Saved: sqs_structure.png")
```

### Method B: SQS with sqsgenerator (Shell-Weighted Optimization)

The **sqsgenerator** package provides a high-performance SQS engine with explicit control over shell weights and the number of output structures. This is the backend used by pyiron's `SQSJob`. The script below uses it directly without any pyiron dependency.

```python
#!/usr/bin/env python3
"""
SQS generation using sqsgenerator directly.
Features:
  - Shell-specific weight control (prioritize short-range correlations)
  - Multiple output structures ranked by objective function
  - Parallel thread support
  - Composition rounding for finite supercells
"""

import numpy as np
from ase.build import bulk, make_supercell

# ============================================================
# Step 1: Build supercell and define composition
# ============================================================
parent = bulk("Cu", crystalstructure="fcc", a=3.80)
sc_matrix = [[2, 2, 0], [2, 0, 2], [0, 2, 2]]  # 32 atoms (isotropic)
supercell = make_supercell(parent, sc_matrix)
n_atoms = len(supercell)

# Target: Cu0.75Au0.25
species_one = "Cu"
species_two = "Au"
x_two = 0.25

# Convert mole fractions to integer atom counts (handle rounding)
n_species_two = int(round(x_two * n_atoms))
n_species_one = n_atoms - n_species_two
actual_x = n_species_two / n_atoms
print(f"Supercell: {n_atoms} atoms")
print(f"Target: {species_one}={1-x_two:.2f}, {species_two}={x_two:.2f}")
print(f"Actual: {species_one}={n_species_one}, {species_two}={n_species_two} "
      f"(x={actual_x:.4f})")

# Assign initial chemical symbols (sqsgenerator shuffles these)
symbols = [species_one] * n_species_one + [species_two] * n_species_two
supercell.set_chemical_symbols(symbols)

# ============================================================
# Step 2: Configure and run sqsgenerator
# ============================================================
from sqsgenerator import sqs_optimize, IterationMode

# Shell weights: 1/shell_number (same as pyiron default)
# This prioritizes matching short-range pair correlations
n_shells = 7
weights = [1.0 / (i + 1) for i in range(n_shells)]

result = sqs_optimize(
    structure=supercell,
    weights=weights,
    objective=0.0,           # target: perfect random alloy
    iterations=int(1e6),     # number of random shuffles to try
    output_structures=5,     # return top-5 SQS candidates
    mode=IterationMode.random,
    num_threads=4,
)

print(f"\nGenerated {len(result)} SQS candidates")
for i, (structure, objective, timings) in enumerate(result):
    print(f"  SQS #{i}: objective = {objective:.6f}")

# Best SQS is the first one (lowest objective)
best_sqs = result[0][0]  # ASE Atoms object
best_obj = result[0][1]
print(f"\nBest SQS objective: {best_obj:.6f}")

from ase.io import write
write("sqs_sqsgenerator.cif", best_sqs)
print("Saved: sqs_sqsgenerator.cif")
```

### Method C: SQS with pymatgen (No Extra Installs)

```python
#!/usr/bin/env python3
"""
SQS generation using pymatgen's built-in SQS tools.
Simpler but less flexible than icet.
"""

from pymatgen.core import Structure, Lattice
from pymatgen.transformations.advanced_transformations import SQSTransformation

# Define FCC parent structure with mixed occupancy
a = 3.80  # approximate lattice parameter for Cu-Au
lattice = Lattice.cubic(a)
# Single atom at FCC origin; pymatgen SQSTransformation handles the rest
fcc_structure = Structure(
    lattice,
    ["Cu"],
    [[0.0, 0.0, 0.0]],
)

# SQS transformation
# scaling defines the supercell size (e.g., [2,2,2] = 32 atoms for FCC with 4 atoms/cell)
sqs_transform = SQSTransformation(
    scaling=[2, 2, 2],           # supercell dimensions
    search_time=60,              # seconds of Monte Carlo search
    cluster_size_and_shell={     # {cluster_order: number_of_shells}
        2: 4,                    # pairs up to 4th shell
        3: 2,                    # triplets up to 2nd shell
    },
    directory=".",
    instances=4,                 # parallel searches
)

# The transformation needs a disordered structure
from pymatgen.core import DummySpecies
disordered = Structure(
    lattice,
    [{"Cu": 0.75, "Au": 0.25}],
    [[0.0, 0.0, 0.0]],
)

print("Generating SQS with pymatgen (this may take ~60s)...")
try:
    sqs_result = sqs_transform.apply_transformation(disordered)
    print(f"SQS structure: {sqs_result}")
    print(f"Number of sites: {len(sqs_result)}")
    sqs_result.to(filename="sqs_pymatgen.cif")
    print("Saved: sqs_pymatgen.cif")
except Exception as e:
    print(f"pymatgen SQS failed (may need ATAT mcsqs): {e}")
    print("Falling back to icet method above.")
```

### Method D: Manual Enumeration (Educational)

```python
#!/usr/bin/env python3
"""
Manual SQS-like selection by enumerating small supercells and
picking the one with correlation functions closest to the random alloy.
Only practical for very small cells.
"""

import numpy as np
from itertools import combinations
from ase.build import bulk, make_supercell
from ase import Atoms

# Build a 2x2x2 FCC supercell (32 atoms for conventional cell, 8 for primitive)
parent = bulk("Cu", crystalstructure="fcc", a=3.80)
sc_matrix = [[2, 0, 0], [0, 2, 0], [0, 0, 2]]
supercell = make_supercell(parent, sc_matrix)
n_atoms = len(supercell)
n_au = round(0.25 * n_atoms)  # 25% Au
print(f"Supercell: {n_atoms} atoms, placing {n_au} Au atoms")

def compute_pair_correlation(atoms, cutoff=4.5):
    """Compute average pair correlation for nearest-neighbor shell.
    Assigns spin +1 to Au, -1 to Cu."""
    symbols = atoms.get_chemical_symbols()
    spins = np.array([1.0 if s == "Au" else -1.0 for s in symbols])
    distances = atoms.get_all_distances(mic=True)
    # Find nearest-neighbor distance
    nn_dist = np.min(distances[distances > 0.1])
    # Pair correlation: average of spin_i * spin_j for NN pairs
    pair_sum = 0.0
    pair_count = 0
    for i in range(len(atoms)):
        for j in range(i + 1, len(atoms)):
            if distances[i, j] < nn_dist * 1.1:  # within 10% tolerance
                pair_sum += spins[i] * spins[j]
                pair_count += 1
    return pair_sum / pair_count if pair_count > 0 else 0.0

# Target: for x_Au=0.25, sigma_avg = 2*0.25 - 1 = -0.5
# Random pair correlation = sigma_avg^2 = 0.25
target_pair_corr = (2 * 0.25 - 1) ** 2

# Enumerate random placements (sample if too many combinations)
all_indices = list(range(n_atoms))
n_combos = int(np.math.factorial(n_atoms) / (np.math.factorial(n_au) * np.math.factorial(n_atoms - n_au)))
print(f"Total configurations: {n_combos}")

best_structure = None
best_delta = float("inf")
n_samples = min(5000, n_combos)

rng = np.random.default_rng(42)
tested = set()

for trial in range(n_samples):
    # Random selection of Au sites
    au_sites = tuple(sorted(rng.choice(n_atoms, size=n_au, replace=False)))
    if au_sites in tested:
        continue
    tested.add(au_sites)

    trial_atoms = supercell.copy()
    symbols = ["Cu"] * n_atoms
    for idx in au_sites:
        symbols[idx] = "Au"
    trial_atoms.set_chemical_symbols(symbols)

    corr = compute_pair_correlation(trial_atoms)
    delta = abs(corr - target_pair_corr)

    if delta < best_delta:
        best_delta = delta
        best_structure = trial_atoms.copy()

    if trial % 1000 == 0:
        print(f"  Trial {trial}: best delta = {best_delta:.4f}")

print(f"\nBest SQS found: pair correlation = {compute_pair_correlation(best_structure):.4f}")
print(f"Target (random):               = {target_pair_corr:.4f}")
print(f"Delta:                          = {best_delta:.4f}")

from ase.io import write
write("sqs_manual.cif", best_structure)
print("Saved: sqs_manual.cif")
```

---

### Method E: Multi-Concentration SQS Sweep (SQSMaster Pattern)

This is the core workflow inspired by pyiron's `SQSMaster`: sweep across a composition range, generate one or more SQS at each concentration, relax all structures with MACE, and collect property data as a function of composition. The script is fully standalone -- no pyiron dependency.

```python
#!/usr/bin/env python3
"""
Multi-concentration SQS sweep for a binary alloy.

Workflow (inspired by pyiron SQSMaster):
  1. Define a list of concentrations: x = 0.1, 0.2, ..., 0.9 for A_{1-x}B_x
  2. At each x, generate N_replicas SQS structures (statistical sampling)
  3. Validate each SQS via correlation function quality metrics
  4. Relax all structures with MACE
  5. Compute mixing energy, lattice parameter, bulk modulus at each x
  6. Plot all properties vs composition with error bars

System: FCC Cu_{1-x}Au_x
"""

import json
import warnings
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from dataclasses import dataclass, field, asdict

from ase.build import bulk, make_supercell
from ase.optimize import BFGS, FIRE
from ase.constraints import ExpCellFilter
from ase.io import write as ase_write
from mace.calculators import mace_mp

warnings.filterwarnings("ignore")

# ============================================================
# Configuration
# ============================================================
SPECIES_ONE = "Cu"
SPECIES_TWO = "Au"
CRYSTAL_STRUCTURE = "fcc"
LATTICE_PARAM = 3.80          # approximate average a0 (Angstrom)
FRACTION_LIST = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
N_REPLICAS = 3                # SQS replicas per concentration
SUPERCELL_SIZE = 32           # target atoms (actual may differ slightly)
MC_STEPS = 50000              # per supercell per attempt
PAIR_CUTOFF = 6.0             # Angstrom
TRIPLET_CUTOFF = 4.5          # Angstrom
FMAX = 0.02                   # relaxation convergence (eV/A)
MAX_STEPS = 300               # max optimizer steps
OUTPUT_DIR = Path("sqs_sweep")
OUTPUT_DIR.mkdir(exist_ok=True)

# Supercell matrices to search over (all ~32 atoms for FCC primitive)
SUPERCELL_MATRICES = [
    [[4, 0, 0], [0, 4, 0], [0, 0, 2]],
    [[4, 0, 0], [0, 2, 0], [0, 0, 4]],
    [[2, 2, 0], [2, 0, 2], [0, 2, 2]],
    [[3, 0, 0], [0, 3, 0], [0, 0, 3]],
]

# ============================================================
# Data container
# ============================================================
@dataclass
class SQSResult:
    x_target: float
    x_actual: float
    replica: int
    n_atoms: int
    quality_score: float       # sum of |delta_correlation|^2
    max_pair_delta: float      # worst pair correlation deviation
    energy_per_atom: float     # after relaxation (eV/atom)
    mixing_energy: float       # meV/atom
    volume_per_atom: float     # A^3/atom
    lattice_param_eff: float   # effective cubic lattice parameter (A)
    cif_file: str

# ============================================================
# Helper: SQS quality metrics
# ============================================================
def compute_sqs_quality(cluster_space, sqs, target_concentrations):
    """
    Evaluate SQS quality by comparing correlation functions to random-alloy targets.

    Returns:
        quality_score: sum of squared deviations (lower = better)
        max_pair_delta: maximum absolute deviation among pair correlations
        deltas: dict mapping orbit index to deviation
    """
    # Get SQS cluster vector
    try:
        cv_sqs = cluster_space.get_cluster_vector(sqs)
    except AttributeError:
        from icet import ClusterVectorCalculator
        cv_calc = ClusterVectorCalculator(cluster_space)
        cv_sqs = cv_calc.get_cluster_vector(sqs)

    # Get target cluster vector
    from icet.tools.structure_generation import _get_sqs_cluster_vector
    try:
        cv_target = _get_sqs_cluster_vector(
            cluster_space=cluster_space,
            target_concentrations=target_concentrations,
        )
    except Exception:
        # Manual fallback
        x_b = target_concentrations[sorted(target_concentrations.keys())[1]]
        sigma_avg = 2 * x_b - 1
        n_orbits = cluster_space.number_of_orbits
        cv_target = np.zeros(n_orbits + 1)
        cv_target[0] = 1.0
        for i, orb in enumerate(cluster_space.orbit_data):
            cv_target[i + 1] = sigma_avg ** orb["order"]

    # Compute per-orbit deviations
    deltas = {}
    pair_deltas = []
    total_sq = 0.0
    for i, orb in enumerate(cluster_space.orbit_data):
        idx = i + 1
        delta = cv_sqs[idx] - cv_target[idx]
        deltas[idx] = delta
        total_sq += delta ** 2
        if orb["order"] == 2:
            pair_deltas.append(abs(delta))

    quality_score = total_sq
    max_pair_delta = max(pair_deltas) if pair_deltas else 0.0

    return quality_score, max_pair_delta, deltas

# ============================================================
# Step 1: Set up icet cluster space and supercells
# ============================================================
from icet import ClusterSpace
from icet.tools.structure_generation import generate_sqs_from_supercells

parent = bulk(SPECIES_ONE, crystalstructure=CRYSTAL_STRUCTURE, a=LATTICE_PARAM)

cluster_space = ClusterSpace(
    structure=parent,
    cutoffs=[PAIR_CUTOFF, TRIPLET_CUTOFF],
    chemical_symbols=[[SPECIES_ONE, SPECIES_TWO]],
)
print(f"Cluster space: {cluster_space.number_of_orbits} orbits")
print(cluster_space)

supercells = [make_supercell(parent, m) for m in SUPERCELL_MATRICES]
print(f"Supercell sizes: {[len(sc) for sc in supercells]} atoms\n")

# ============================================================
# Step 2: Generate SQS at each concentration
# ============================================================
print("=" * 70)
print("PHASE 1: SQS GENERATION")
print("=" * 70)

all_sqs = {}  # {(x, replica): ASE Atoms}
all_quality = {}  # {(x, replica): (quality_score, max_pair_delta)}

for x in FRACTION_LIST:
    target_conc = {SPECIES_ONE: 1 - x, SPECIES_TWO: x}
    print(f"\n--- x({SPECIES_TWO}) = {x:.1f} ---")

    for rep in range(N_REPLICAS):
        seed = int(x * 1000) + rep * 137 + 42  # reproducible but varied
        sqs = generate_sqs_from_supercells(
            cluster_space=cluster_space,
            supercells=supercells,
            target_concentrations=target_conc,
            n_steps=MC_STEPS,
            random_seed=seed,
        )

        # Compute actual composition
        syms = sqs.get_chemical_symbols()
        n_b = sum(1 for s in syms if s == SPECIES_TWO)
        x_actual = n_b / len(sqs)

        # Quality check
        q_score, max_pd, _ = compute_sqs_quality(cluster_space, sqs, target_conc)

        all_sqs[(x, rep)] = sqs
        all_quality[(x, rep)] = (q_score, max_pd)

        tag = f"sqs_{SPECIES_ONE}{SPECIES_TWO}_x{x:.1f}_rep{rep}"
        cif_path = OUTPUT_DIR / f"{tag}.cif"
        ase_write(str(cif_path), sqs)

        quality_label = "GOOD" if max_pd < 0.05 else ("OK" if max_pd < 0.10 else "POOR")
        print(f"  replica {rep}: {len(sqs)} atoms, x_actual={x_actual:.3f}, "
              f"quality={q_score:.4f}, max_pair_delta={max_pd:.4f} [{quality_label}]")

# ============================================================
# Step 3: Relax all SQS with MACE and compute properties
# ============================================================
print("\n" + "=" * 70)
print("PHASE 2: MACE RELAXATION AND PROPERTY CALCULATION")
print("=" * 70)

calc = mace_mp(model="medium", default_dtype="float64")

# Relax pure endpoints for mixing energy reference
def relax_pure(element, a0):
    atoms = bulk(element, crystalstructure=CRYSTAL_STRUCTURE, a=a0)
    atoms.calc = calc
    ecf = ExpCellFilter(atoms)
    opt = BFGS(ecf, logfile=str(OUTPUT_DIR / f"relax_{element}.log"))
    opt.run(fmax=0.005, steps=200)
    return atoms.get_potential_energy() / len(atoms), atoms.get_volume() / len(atoms)

pure_a_params = {SPECIES_ONE: 3.615, SPECIES_TWO: 4.078}  # initial guesses
e_pure = {}
v_pure = {}
for sp, a0 in pure_a_params.items():
    e_pure[sp], v_pure[sp] = relax_pure(sp, a0)
    print(f"Pure {sp}: E = {e_pure[sp]:.4f} eV/atom, V = {v_pure[sp]:.2f} A^3/atom")

# Relax all SQS
results = []

for (x, rep), sqs in all_sqs.items():
    tag = f"x={x:.1f} rep={rep}"
    print(f"\nRelaxing {tag} ({len(sqs)} atoms)...")

    sqs.calc = calc
    try:
        ecf = ExpCellFilter(sqs)
        opt = BFGS(ecf, logfile=str(OUTPUT_DIR / f"relax_x{x:.1f}_rep{rep}.log"))
        opt.run(fmax=FMAX, steps=MAX_STEPS)
    except Exception as e:
        print(f"  BFGS failed ({e}), retrying with FIRE...")
        sqs.calc = calc
        ecf = ExpCellFilter(sqs)
        opt = FIRE(ecf, logfile=str(OUTPUT_DIR / f"relax_x{x:.1f}_rep{rep}_fire.log"))
        opt.run(fmax=FMAX, steps=MAX_STEPS)

    e_total = sqs.get_potential_energy()
    e_per_atom = e_total / len(sqs)
    vol = sqs.get_volume()
    v_per_atom = vol / len(sqs)
    # Effective cubic lattice parameter: a_eff = (V_per_atom * atoms_per_fcc_cell)^(1/3)
    a_eff = (v_per_atom * 4) ** (1.0 / 3.0)  # 4 atoms per FCC conventional cell

    # Mixing energy
    syms = sqs.get_chemical_symbols()
    n_b = sum(1 for s in syms if s == SPECIES_TWO)
    x_actual = n_b / len(sqs)
    x_a = 1 - x_actual
    e_mix = e_per_atom - x_a * e_pure[SPECIES_ONE] - x_actual * e_pure[SPECIES_TWO]
    e_mix_meV = e_mix * 1000

    q_score, max_pd = all_quality[(x, rep)]

    cif_path = str(OUTPUT_DIR / f"sqs_{SPECIES_ONE}{SPECIES_TWO}_x{x:.1f}_rep{rep}_relaxed.cif")
    ase_write(cif_path, sqs)

    r = SQSResult(
        x_target=x,
        x_actual=x_actual,
        replica=rep,
        n_atoms=len(sqs),
        quality_score=q_score,
        max_pair_delta=max_pd,
        energy_per_atom=e_per_atom,
        mixing_energy=e_mix_meV,
        volume_per_atom=v_per_atom,
        lattice_param_eff=a_eff,
        cif_file=cif_path,
    )
    results.append(r)
    print(f"  E/atom = {e_per_atom:.4f} eV, E_mix = {e_mix_meV:.1f} meV/atom, "
          f"a_eff = {a_eff:.3f} A")

# Save all results to JSON
with open(OUTPUT_DIR / "sqs_sweep_results.json", "w") as f:
    json.dump([asdict(r) for r in results], f, indent=2)
print(f"\nSaved: {OUTPUT_DIR}/sqs_sweep_results.json")

# ============================================================
# Step 4: Plot properties vs composition
# ============================================================
print("\n" + "=" * 70)
print("PHASE 3: PLOTTING")
print("=" * 70)

# Group results by target composition
from collections import defaultdict
by_x = defaultdict(list)
for r in results:
    by_x[r.x_target].append(r)

x_vals = sorted(by_x.keys())
mix_means = []
mix_stds = []
a_means = []
a_stds = []
vol_means = []
vol_stds = []

for x in x_vals:
    rs = by_x[x]
    mix_vals = [r.mixing_energy for r in rs]
    a_vals = [r.lattice_param_eff for r in rs]
    v_vals = [r.volume_per_atom for r in rs]
    mix_means.append(np.mean(mix_vals))
    mix_stds.append(np.std(mix_vals))
    a_means.append(np.mean(a_vals))
    a_stds.append(np.std(a_vals))
    vol_means.append(np.mean(v_vals))
    vol_stds.append(np.std(v_vals))

# Include pure endpoints
x_plot = [0.0] + x_vals + [1.0]
mix_plot = [0.0] + mix_means + [0.0]
mix_err = [0.0] + mix_stds + [0.0]

a_pure_one = (v_pure[SPECIES_ONE] * 4) ** (1.0 / 3.0)
a_pure_two = (v_pure[SPECIES_TWO] * 4) ** (1.0 / 3.0)
a_plot = [a_pure_one] + a_means + [a_pure_two]
a_err = [0.0] + a_stds + [0.0]

# --- Figure 1: Mixing energy vs composition ---
fig1, ax1 = plt.subplots(figsize=(8, 5))
ax1.errorbar(x_plot, mix_plot, yerr=mix_err, fmt="o-", color="steelblue",
             capsize=4, markersize=6, linewidth=1.5, label="SQS + MACE")
# Redlich-Kister fit (simple parabolic: Omega * x * (1-x))
x_fit = np.array(x_plot)
mix_fit = np.array(mix_plot)
# Fit Omega from interior points only
interior = (x_fit > 0) & (x_fit < 1)
if np.any(interior):
    Omega = np.mean(mix_fit[interior] / (x_fit[interior] * (1 - x_fit[interior]) + 1e-12))
    x_smooth = np.linspace(0, 1, 100)
    ax1.plot(x_smooth, Omega * x_smooth * (1 - x_smooth), "--", color="salmon",
             linewidth=1.5, label=f"Regular solution fit ($\\Omega$ = {Omega:.0f} meV)")
ax1.axhline(0, color="k", linewidth=0.5, linestyle=":")
ax1.set_xlabel(f"$x$ in {SPECIES_ONE}$_{{1-x}}${SPECIES_TWO}$_x$", fontsize=12)
ax1.set_ylabel("Mixing energy (meV/atom)", fontsize=12)
ax1.set_title(f"Mixing Energy: {SPECIES_ONE}--{SPECIES_TWO} (SQS + MACE)", fontsize=13)
ax1.legend(fontsize=10)
ax1.set_xlim(-0.05, 1.05)
plt.tight_layout()
fig1.savefig(str(OUTPUT_DIR / "mixing_energy_vs_x.png"), dpi=150)
print(f"Saved: {OUTPUT_DIR}/mixing_energy_vs_x.png")

# --- Figure 2: Lattice parameter vs composition ---
fig2, ax2 = plt.subplots(figsize=(8, 5))
ax2.errorbar(x_plot, a_plot, yerr=a_err, fmt="s-", color="forestgreen",
             capsize=4, markersize=6, linewidth=1.5, label="SQS + MACE")
# Vegard's law reference
a_vegard = [a_pure_one + x * (a_pure_two - a_pure_one) for x in x_plot]
ax2.plot(x_plot, a_vegard, "--", color="gray", linewidth=1.5, label="Vegard's law")
ax2.set_xlabel(f"$x$ in {SPECIES_ONE}$_{{1-x}}${SPECIES_TWO}$_x$", fontsize=12)
ax2.set_ylabel("Effective lattice parameter (A)", fontsize=12)
ax2.set_title(f"Lattice Parameter: {SPECIES_ONE}--{SPECIES_TWO}", fontsize=13)
ax2.legend(fontsize=10)
ax2.set_xlim(-0.05, 1.05)
plt.tight_layout()
fig2.savefig(str(OUTPUT_DIR / "lattice_param_vs_x.png"), dpi=150)
print(f"Saved: {OUTPUT_DIR}/lattice_param_vs_x.png")

# --- Figure 3: SQS quality across compositions ---
fig3, ax3 = plt.subplots(figsize=(8, 5))
for x in x_vals:
    rs = by_x[x]
    for r in rs:
        color = "forestgreen" if r.max_pair_delta < 0.05 else (
            "orange" if r.max_pair_delta < 0.10 else "red")
        ax3.scatter(r.x_target, r.max_pair_delta, c=color, s=50,
                    edgecolors="k", linewidth=0.5, zorder=3)
ax3.axhline(0.05, color="forestgreen", linestyle="--", linewidth=1,
            label="Excellent (< 0.05)")
ax3.axhline(0.10, color="orange", linestyle="--", linewidth=1,
            label="Acceptable (< 0.10)")
ax3.set_xlabel(f"$x$ in {SPECIES_ONE}$_{{1-x}}${SPECIES_TWO}$_x$", fontsize=12)
ax3.set_ylabel("Max pair correlation deviation", fontsize=12)
ax3.set_title("SQS Quality Across Composition Range", fontsize=13)
ax3.legend(fontsize=10)
ax3.set_xlim(-0.05, 1.05)
ax3.set_ylim(bottom=0)
plt.tight_layout()
fig3.savefig(str(OUTPUT_DIR / "sqs_quality_vs_x.png"), dpi=150)
print(f"Saved: {OUTPUT_DIR}/sqs_quality_vs_x.png")

print("\n" + "=" * 70)
print("SWEEP COMPLETE")
print(f"Results: {OUTPUT_DIR}/sqs_sweep_results.json")
print(f"Plots:   {OUTPUT_DIR}/mixing_energy_vs_x.png")
print(f"         {OUTPUT_DIR}/lattice_param_vs_x.png")
print(f"         {OUTPUT_DIR}/sqs_quality_vs_x.png")
print(f"Structures: {OUTPUT_DIR}/sqs_*.cif")
print("=" * 70)
```

---

### Method F: Multi-Concentration Sweep with sqsgenerator

Alternative to Method E that uses **sqsgenerator** instead of icet for the generation step. The sqsgenerator approach gives explicit control over shell weights and produces multiple ranked candidates per concentration.

```python
#!/usr/bin/env python3
"""
Multi-concentration SQS sweep using sqsgenerator backend.

This mirrors the pyiron SQSMaster workflow:
  - SQSMaster defines fraction_lst, species_one, species_two
  - For each fraction, it spawns an SQSJob that calls sqsgenerator
  - Each SQSJob returns n_output_structures ranked by objective

This script does the same without pyiron, using sqsgenerator directly.
"""

import json
import warnings
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

from ase.build import bulk, make_supercell
from ase.optimize import BFGS
from ase.constraints import ExpCellFilter
from ase.io import write as ase_write

warnings.filterwarnings("ignore")

# ============================================================
# Configuration
# ============================================================
SPECIES_ONE = "Cu"
SPECIES_TWO = "Au"
CRYSTAL_STRUCTURE = "fcc"
LATTICE_PARAM = 3.80
OUTPUT_DIR = Path("sqs_sweep_sqsgen")
OUTPUT_DIR.mkdir(exist_ok=True)

# Composition sweep (SQSMaster-style fraction_lst)
FRACTION_LIST = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

# sqsgenerator settings (matching pyiron SQSJob defaults)
ITERATIONS = int(1e6)
N_OUTPUT_STRUCTURES = 3       # top-N candidates per composition
WEIGHTS = None                # None = default [1, 1/2, 1/3, ...]
OBJECTIVE = 0.0               # target perfect random alloy
NUM_THREADS = 4

# Supercell
SC_MATRIX = [[2, 2, 0], [2, 0, 2], [0, 2, 2]]  # 32 atoms (isotropic)

# ============================================================
# Helper: mole fractions to integer composition
# ============================================================
import random as stdlib_random

def mole_fractions_to_composition(mole_fractions, num_atoms):
    """Convert mole fractions dict to integer atom counts.
    Handles rounding for finite supercells (same logic as pyiron SQSJob)."""
    if not (1.0 - 1 / num_atoms) < sum(mole_fractions.values()) < (1.0 + 1 / num_atoms):
        raise ValueError(f"Mole fractions must sum to ~1.0, got {sum(mole_fractions.values())}")

    composition = {k: v * num_atoms for k, v in mole_fractions.items()}

    # Check for fractional occupation
    if any(not float.is_integer(round(occ, 1)) for occ in composition.values()):
        composition = {k: int(round(v)) for k, v in composition.items()}

    # Fix off-by-one due to rounding
    actual = sum(composition.values())
    diff = actual - num_atoms
    if abs(diff) == 1:
        removed = stdlib_random.choice(list(composition.keys()))
        composition[removed] -= int(np.sign(diff))
    elif abs(diff) > 1:
        raise ValueError(f"Cannot distribute species for fractions {mole_fractions} on {num_atoms} atoms")

    return composition

# ============================================================
# Step 1: Generate SQS at each composition
# ============================================================
from sqsgenerator import sqs_optimize, IterationMode

parent = bulk(SPECIES_ONE, crystalstructure=CRYSTAL_STRUCTURE, a=LATTICE_PARAM)
template = make_supercell(parent, SC_MATRIX)
n_atoms = len(template)
print(f"Template supercell: {n_atoms} atoms")

all_structures = {}  # {(x, rank): ASE Atoms}
all_objectives = {}  # {(x, rank): float}

for x in FRACTION_LIST:
    mole_fracs = {SPECIES_ONE: 1 - x, SPECIES_TWO: x}
    comp = mole_fractions_to_composition(mole_fracs, n_atoms)
    x_actual = comp[SPECIES_TWO] / n_atoms

    # Build initial structure with the right atom counts
    sc = template.copy()
    symbols = []
    for sp, count in comp.items():
        symbols.extend([sp] * count)
    # Ensure symbols list matches atom count
    assert len(symbols) == n_atoms, f"Symbol count mismatch: {len(symbols)} vs {n_atoms}"
    sc.set_chemical_symbols(symbols)

    print(f"\nx({SPECIES_TWO}) = {x:.1f} -> {comp} (x_actual = {x_actual:.3f})")

    result = sqs_optimize(
        structure=sc,
        weights=WEIGHTS,
        objective=OBJECTIVE,
        iterations=ITERATIONS,
        output_structures=N_OUTPUT_STRUCTURES,
        mode=IterationMode.random,
        num_threads=NUM_THREADS,
    )

    for rank, (structure, obj_val, _) in enumerate(result):
        all_structures[(x, rank)] = structure
        all_objectives[(x, rank)] = obj_val
        tag = f"sqs_{SPECIES_ONE}{SPECIES_TWO}_x{x:.1f}_rank{rank}"
        ase_write(str(OUTPUT_DIR / f"{tag}.cif"), structure)
        label = "BEST" if rank == 0 else f"#{rank}"
        print(f"  [{label}] objective = {obj_val:.6f}")

# ============================================================
# Step 2: Relax best SQS at each composition with MACE
# ============================================================
from mace.calculators import mace_mp

calc = mace_mp(model="medium", default_dtype="float64")

# Pure references
def relax_pure(element, a0):
    atoms = bulk(element, crystalstructure=CRYSTAL_STRUCTURE, a=a0)
    atoms.calc = calc
    ecf = ExpCellFilter(atoms)
    opt = BFGS(ecf, logfile=str(OUTPUT_DIR / f"relax_{element}.log"))
    opt.run(fmax=0.005, steps=200)
    return atoms.get_potential_energy() / len(atoms)

e_pure = {
    SPECIES_ONE: relax_pure(SPECIES_ONE, 3.615),
    SPECIES_TWO: relax_pure(SPECIES_TWO, 4.078),
}
print(f"\nPure energies: {SPECIES_ONE}={e_pure[SPECIES_ONE]:.4f}, "
      f"{SPECIES_TWO}={e_pure[SPECIES_TWO]:.4f} eV/atom")

results = []
for x in FRACTION_LIST:
    # Use rank-0 (best) SQS
    sqs = all_structures[(x, 0)]
    sqs.calc = calc
    ecf = ExpCellFilter(sqs)
    opt = BFGS(ecf, logfile=str(OUTPUT_DIR / f"relax_x{x:.1f}.log"))
    opt.run(fmax=0.02, steps=300)

    e = sqs.get_potential_energy() / len(sqs)
    syms = sqs.get_chemical_symbols()
    n_b = sum(1 for s in syms if s == SPECIES_TWO)
    x_act = n_b / len(sqs)
    e_mix = (e - (1 - x_act) * e_pure[SPECIES_ONE] - x_act * e_pure[SPECIES_TWO]) * 1000

    results.append({
        "x_target": x,
        "x_actual": x_act,
        "objective": all_objectives[(x, 0)],
        "energy_per_atom": e,
        "mixing_energy_meV": e_mix,
    })
    print(f"x={x:.1f}: E_mix = {e_mix:.1f} meV/atom, obj = {all_objectives[(x, 0)]:.6f}")

# Save and plot
with open(OUTPUT_DIR / "results.json", "w") as f:
    json.dump(results, f, indent=2)

x_plot = [0] + [r["x_target"] for r in results] + [1]
e_plot = [0] + [r["mixing_energy_meV"] for r in results] + [0]

fig, ax = plt.subplots(figsize=(8, 5))
ax.plot(x_plot, e_plot, "o-", color="steelblue", markersize=7, linewidth=1.5)
ax.axhline(0, color="k", linewidth=0.5, linestyle=":")
ax.set_xlabel(f"$x$ in {SPECIES_ONE}$_{{1-x}}${SPECIES_TWO}$_x$", fontsize=12)
ax.set_ylabel("Mixing energy (meV/atom)", fontsize=12)
ax.set_title(f"{SPECIES_ONE}--{SPECIES_TWO} Mixing Energy (sqsgenerator + MACE)")
ax.set_xlim(-0.05, 1.05)
plt.tight_layout()
fig.savefig(str(OUTPUT_DIR / "mixing_energy.png"), dpi=150)
print(f"\nSaved: {OUTPUT_DIR}/mixing_energy.png")
```

---

### Method G: Warren-Cowley Short-Range Order (SRO) Analysis

After generating an SQS (or any alloy structure), you should verify that the structure truly mimics a random alloy by computing the **Warren-Cowley short-range order (SRO) parameters**. For a perfect random alloy, all alpha parameters should be zero. Significant deviations indicate chemical clustering (alpha > 0) or ordering (alpha < 0).

This analysis is inspired by pyiron_atomistics SRO tools but implemented as a standalone script with no pyiron dependency.

```python
#!/usr/bin/env python3
"""
Warren-Cowley Short-Range Order (SRO) analysis for alloy structures.
Computes alpha parameters for multiple neighbor shells and validates
that an SQS has near-zero SRO (as expected for a random alloy).

Reference: Warren, B.E., X-ray Diffraction (1969).
Inspired by pyiron_atomistics SRO analysis (get_average_of_unique_labels).
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from ase.build import bulk, make_supercell
from ase.io import read
from collections import defaultdict

# ============================================================
# Step 1: Load or generate an alloy structure
# ============================================================
# Option A: Load an existing SQS from file
# atoms = read("sqs_Cu75Au25.cif")

# Option B: Generate a quick SQS with icet for demonstration
from icet import ClusterSpace
from icet.tools.structure_generation import generate_sqs_from_supercells

parent = bulk("Cu", crystalstructure="fcc", a=3.80)
cluster_space = ClusterSpace(
    structure=parent,
    cutoffs=[6.0, 4.5],
    chemical_symbols=[["Cu", "Au"]],
)

supercells = [make_supercell(parent, [[4, 0, 0], [0, 4, 0], [0, 0, 2]])]  # 32 atoms
target_concentrations = {"Cu": 0.75, "Au": 0.25}

print("Generating SQS for SRO analysis...")
atoms = generate_sqs_from_supercells(
    cluster_space=cluster_space,
    supercells=supercells,
    target_concentrations=target_concentrations,
    n_steps=50000,
    random_seed=42,
)
print(f"Structure: {len(atoms)} atoms, {atoms.get_chemical_formula()}")

# ============================================================
# Step 2: Compute neighbor shells and Warren-Cowley alpha
# ============================================================
def get_neighbor_shells(atoms, n_shells=8, tol=0.1):
    """
    Identify unique neighbor shell distances using minimum-image convention.
    Returns sorted list of (shell_distance, neighbor_pairs).

    This mirrors pyiron's approach of grouping neighbors by unique distances
    (get_average_of_unique_labels) but uses ASE directly.
    """
    all_distances = atoms.get_all_distances(mic=True)
    n = len(atoms)

    # Collect all unique distances (rounded to avoid floating-point duplicates)
    unique_dists = set()
    for i in range(n):
        for j in range(i + 1, n):
            d = all_distances[i, j]
            if d > 0.1:  # skip self
                unique_dists.add(round(d, 2))

    unique_dists = sorted(unique_dists)

    # Group into shells by merging distances within tolerance
    shells = []
    current_shell = [unique_dists[0]]
    for d in unique_dists[1:]:
        if d - current_shell[-1] < tol:
            current_shell.append(d)
        else:
            shells.append(np.mean(current_shell))
            current_shell = [d]
            if len(shells) >= n_shells:
                break
    if len(shells) < n_shells and current_shell:
        shells.append(np.mean(current_shell))

    shells = shells[:n_shells]

    # For each shell, find atom pairs
    shell_pairs = []
    for shell_dist in shells:
        pairs = []
        for i in range(n):
            for j in range(i + 1, n):
                d = all_distances[i, j]
                if abs(d - shell_dist) < tol:
                    pairs.append((i, j))
        shell_pairs.append((shell_dist, pairs))

    return shell_pairs


def warren_cowley_alpha(atoms, species_A, species_B, n_shells=8, tol=0.1):
    """
    Compute Warren-Cowley short-range order parameter alpha for each shell.

    alpha_n = 1 - P_AB(n) / x_B

    where P_AB(n) is the conditional probability that a B atom is the neighbor
    of an A atom in shell n, and x_B is the global concentration of B.

    For a random alloy: alpha = 0 for all shells.
    For ordering (A-B preference): alpha < 0.
    For clustering (A-A / B-B preference): alpha > 0.
    """
    symbols = np.array(atoms.get_chemical_symbols())
    n_atoms = len(atoms)

    # Global concentrations
    x_A = np.sum(symbols == species_A) / n_atoms
    x_B = np.sum(symbols == species_B) / n_atoms

    if x_A == 0 or x_B == 0:
        raise ValueError(f"Both species must be present. Found x_{species_A}={x_A}, x_{species_B}={x_B}")

    shell_pairs = get_neighbor_shells(atoms, n_shells=n_shells, tol=tol)

    results = []
    for shell_dist, pairs in shell_pairs:
        if not pairs:
            continue

        # Count A-B pairs, A-A pairs, etc.
        n_AB = 0  # A atom paired with B atom
        n_A_total = 0  # Total bonds from A atoms in this shell

        for i, j in pairs:
            si, sj = symbols[i], symbols[j]
            # Count from A's perspective
            if si == species_A:
                n_A_total += 1
                if sj == species_B:
                    n_AB += 1
            if sj == species_A:
                n_A_total += 1
                if si == species_B:
                    n_AB += 1

        # P_AB = probability that neighbor of A is B
        P_AB = n_AB / n_A_total if n_A_total > 0 else 0.0

        # Warren-Cowley alpha
        alpha = 1.0 - P_AB / x_B

        results.append({
            "shell_distance": shell_dist,
            "alpha": alpha,
            "P_AB": P_AB,
            "n_pairs": len(pairs),
            "n_A_total": n_A_total,
            "n_AB": n_AB,
        })

    return results


# ============================================================
# Step 3: Compute SRO parameters
# ============================================================
species_A = "Cu"
species_B = "Au"
n_shells = 6

print(f"\n== Warren-Cowley SRO Analysis ({species_A}-{species_B}) ==")
sro_results = warren_cowley_alpha(atoms, species_A, species_B, n_shells=n_shells)

print(f"{'Shell':>6} {'Dist (A)':>10} {'alpha':>10} {'P_AB':>10} {'N_pairs':>10}")
print("-" * 50)
for i, r in enumerate(sro_results):
    print(f"{i+1:>6d} {r['shell_distance']:>10.3f} {r['alpha']:>10.4f} {r['P_AB']:>10.4f} {r['n_pairs']:>10d}")

# ============================================================
# Step 4: Validate SQS quality from SRO
# ============================================================
max_alpha = max(abs(r["alpha"]) for r in sro_results)
print(f"\nMax |alpha| across all shells: {max_alpha:.4f}")

if max_alpha < 0.05:
    print("PASS: SQS has excellent random-alloy character (|alpha| < 0.05 for all shells)")
elif max_alpha < 0.10:
    print("ACCEPTABLE: SQS has reasonable random-alloy character (|alpha| < 0.10 for all shells)")
else:
    print("WARNING: SQS shows significant SRO. Consider regenerating with more MC steps or a larger cell.")

# ============================================================
# Step 5: Plot alpha vs shell distance
# ============================================================
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# Left panel: alpha vs shell index
shell_indices = list(range(1, len(sro_results) + 1))
alphas = [r["alpha"] for r in sro_results]
dists = [r["shell_distance"] for r in sro_results]

ax1 = axes[0]
bars = ax1.bar(shell_indices, alphas, color="steelblue", edgecolor="k", linewidth=0.5)
# Color negative bars differently
for bar, a in zip(bars, alphas):
    if a < 0:
        bar.set_color("salmon")
ax1.axhline(0, color="k", linewidth=1.0, linestyle="-")
ax1.axhline(0.05, color="gray", linewidth=0.8, linestyle="--", label="Random threshold (+/- 0.05)")
ax1.axhline(-0.05, color="gray", linewidth=0.8, linestyle="--")
ax1.set_xlabel("Neighbor shell")
ax1.set_ylabel("Warren-Cowley alpha")
ax1.set_title(f"SRO Parameters: {species_A}-{species_B}")
ax1.legend(fontsize=9)
ax1.set_xticks(shell_indices)

# Right panel: alpha vs distance
ax2 = axes[1]
ax2.plot(dists, alphas, "o-", color="steelblue", markersize=8, linewidth=1.5)
ax2.axhline(0, color="k", linewidth=1.0, linestyle="-")
ax2.axhspan(-0.05, 0.05, color="green", alpha=0.1, label="Random alloy zone")
ax2.set_xlabel("Shell distance (A)")
ax2.set_ylabel("Warren-Cowley alpha")
ax2.set_title(f"SRO vs Distance: {species_A}-{species_B}")
ax2.legend(fontsize=9)

plt.tight_layout()
plt.savefig("sro_analysis.png", dpi=150)
print("\nSaved: sro_analysis.png")

# ============================================================
# Step 6: Pairwise SRO for multi-component systems
# ============================================================
# For ternary+ alloys, compute alpha for all species pairs.
# Example usage (uncomment for a ternary system):
#
# species_list = ["Cu", "Au", "Ag"]
# for i, sA in enumerate(species_list):
#     for sB in species_list[i+1:]:
#         print(f"\n--- SRO: {sA}-{sB} ---")
#         results = warren_cowley_alpha(atoms, sA, sB, n_shells=4)
#         for r in results:
#             print(f"  Shell {r['shell_distance']:.2f} A: alpha = {r['alpha']:.4f}")

print("\nDone.")
```

---

### Method H: Multi-Sublattice SQS

For complex crystal structures with multiple Wyckoff sites (e.g., perovskites, spinels, Heusler alloys), disorder may occur on only some sublattices, or different sublattices may host different species sets. icet supports multi-sublattice SQS generation natively.

Example: **(Ba,Sr)TiO3** perovskite with A-site disorder (Ba/Sr) while the Ti and O sublattices remain fixed.

```python
#!/usr/bin/env python3
"""
Multi-sublattice SQS generation for (Ba,Sr)TiO3 perovskite.
A-site: Ba/Sr disorder (50/50)
B-site: Ti only (fixed)
X-site: O only (fixed)

Uses icet with sublattice-aware chemical symbols.
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from ase import Atoms
from ase.build import make_supercell
from icet import ClusterSpace
from icet.tools.structure_generation import generate_sqs_from_supercells

# ============================================================
# Step 1: Define the perovskite parent structure (cubic BaTiO3)
# ============================================================
# Cubic perovskite ABX3
# A at (0, 0, 0), B at (0.5, 0.5, 0.5), X at (0.5, 0.5, 0), (0.5, 0, 0.5), (0, 0.5, 0.5)
a = 4.00  # approximate lattice parameter

parent = Atoms(
    symbols=["Ba", "Ti", "O", "O", "O"],
    scaled_positions=[
        [0.0, 0.0, 0.0],    # A-site
        [0.5, 0.5, 0.5],    # B-site
        [0.5, 0.5, 0.0],    # X-site
        [0.5, 0.0, 0.5],    # X-site
        [0.0, 0.5, 0.5],    # X-site
    ],
    cell=[a, a, a],
    pbc=True,
)

print(f"Parent structure: {parent.get_chemical_formula()}")
print(f"Number of sites: {len(parent)}")

# ============================================================
# Step 2: Define cluster space with sublattice constraints
# ============================================================
# chemical_symbols: one list per site in the parent cell.
# A-site: ["Ba", "Sr"] (disordered)
# B-site: ["Ti"] (fixed)
# X-sites: ["O"] (fixed)
chemical_symbols = [
    ["Ba", "Sr"],  # site 0: A-site (disordered)
    ["Ti"],        # site 1: B-site (fixed)
    ["O"],         # site 2: X-site (fixed)
    ["O"],         # site 3: X-site (fixed)
    ["O"],         # site 4: X-site (fixed)
]

# Cluster space: pairs up to 7.0 A, triplets up to 5.0 A
# Only orbits involving the disordered sublattice will be active
cluster_space = ClusterSpace(
    structure=parent,
    cutoffs=[7.0, 5.0],
    chemical_symbols=chemical_symbols,
)

print("\n== Cluster Space ==")
print(cluster_space)
print(f"Number of orbits: {cluster_space.number_of_orbits}")

# Show which orbits are active (only those with A-site pairs/triplets)
orbit_data = cluster_space.orbit_data
for i, orb in enumerate(orbit_data):
    print(f"  Orbit {i+1}: order={orb['order']}, radius={orb.get('radius', 'N/A')}")

# ============================================================
# Step 3: Generate supercells for SQS search
# ============================================================
# For perovskite, 2x2x2 gives 40 atoms (8 A-sites for 50/50 = 4 Ba + 4 Sr)
# 3x3x3 gives 135 atoms (27 A-sites: not divisible by 2, use 3x3x2 = 90 atoms, 18 A-sites)
supercell_matrices = [
    [[2, 0, 0], [0, 2, 0], [0, 0, 2]],   # 40 atoms, 8 A-sites
    [[3, 0, 0], [0, 3, 0], [0, 0, 2]],   # 90 atoms, 18 A-sites
    [[2, 0, 0], [0, 2, 0], [0, 0, 3]],   # 60 atoms, 12 A-sites
]

supercells = []
for sc_matrix in supercell_matrices:
    sc = make_supercell(parent, sc_matrix)
    n_a_sites = sum(1 for s in sc.get_chemical_symbols() if s == "Ba")
    supercells.append(sc)
    print(f"Supercell: {len(sc)} atoms, {n_a_sites} A-sites, matrix = {sc_matrix}")

# ============================================================
# Step 4: Generate multi-sublattice SQS
# ============================================================
# Target: 50% Ba, 50% Sr on the A sublattice
# For icet, specify concentrations for the disordered sublattice
target_concentrations = {"Ba": 0.5, "Sr": 0.5}

print("\nGenerating multi-sublattice SQS...")
sqs = generate_sqs_from_supercells(
    cluster_space=cluster_space,
    supercells=supercells,
    target_concentrations=target_concentrations,
    n_steps=80000,
    random_seed=42,
)

# Count species
symbols = sqs.get_chemical_symbols()
counts = {}
for s in symbols:
    counts[s] = counts.get(s, 0) + 1

print(f"\nSQS generated: {len(sqs)} atoms")
for species, count in sorted(counts.items()):
    print(f"  {species}: {count} atoms")

n_ba = counts.get("Ba", 0)
n_sr = counts.get("Sr", 0)
n_a_total = n_ba + n_sr
print(f"A-site composition: Ba={n_ba/n_a_total:.3f}, Sr={n_sr/n_a_total:.3f}")

# ============================================================
# Step 5: Validate correlations per sublattice
# ============================================================
try:
    cv_sqs = cluster_space.get_cluster_vector(sqs)
except AttributeError:
    from icet import ClusterVectorCalculator
    cv_calculator = ClusterVectorCalculator(cluster_space)
    cv_sqs = cv_calculator.get_cluster_vector(sqs)

# Target for random 50/50: sigma_avg = 2*0.5 - 1 = 0.0
# All pair and triplet correlations should be 0.0
sigma_avg = 2 * target_concentrations["Sr"] - 1  # = 0.0 for 50/50

print("\n== Sublattice Correlation Validation ==")
print(f"Target sigma_avg for A-site: {sigma_avg:.4f}")
print(f"{'Orbit':>6} {'Order':>6} {'SQS':>10} {'Target':>10} {'Delta':>10}")
print("-" * 50)

orbit_data = cluster_space.orbit_data
max_delta = 0.0
for i, orb in enumerate(orbit_data):
    idx = i + 1
    target_val = sigma_avg ** orb["order"]
    delta = cv_sqs[idx] - target_val
    max_delta = max(max_delta, abs(delta))
    print(f"{idx:>6d} {orb['order']:>6d} {cv_sqs[idx]:>10.4f} {target_val:>10.4f} {delta:>10.4f}")

print(f"\nMax |delta| across orbits: {max_delta:.4f}")
if max_delta < 0.02:
    print("PASS: Excellent SQS quality for multi-sublattice system")
elif max_delta < 0.05:
    print("ACCEPTABLE: Reasonable SQS quality")
else:
    print("WARNING: Consider more MC steps or larger supercell")

# ============================================================
# Step 6: Relax with MACE and compute formation properties
# ============================================================
from ase.optimize import BFGS
from ase.constraints import ExpCellFilter
from mace.calculators import mace_mp

print("\nRelaxing multi-sublattice SQS with MACE...")
calc = mace_mp(model="medium", default_dtype="float64")
sqs.calc = calc

ecf = ExpCellFilter(sqs)
opt = BFGS(ecf, logfile="sqs_perovskite_relax.log")
opt.run(fmax=0.01, steps=500)

e_sqs = sqs.get_potential_energy()
print(f"SQS total energy: {e_sqs:.4f} eV ({len(sqs)} atoms)")
print(f"SQS energy/atom:  {e_sqs / len(sqs):.4f} eV/atom")

# Reference energies: pure BaTiO3 and pure SrTiO3
from ase.io import write

ref_BTO = Atoms(
    symbols=["Ba", "Ti", "O", "O", "O"],
    scaled_positions=[
        [0.0, 0.0, 0.0],
        [0.5, 0.5, 0.5],
        [0.5, 0.5, 0.0],
        [0.5, 0.0, 0.5],
        [0.0, 0.5, 0.5],
    ],
    cell=[a, a, a],
    pbc=True,
)
ref_BTO.calc = calc
ecf_bto = ExpCellFilter(ref_BTO)
opt_bto = BFGS(ecf_bto, logfile="bto_relax.log")
opt_bto.run(fmax=0.01)
e_bto = ref_BTO.get_potential_energy() / len(ref_BTO)

ref_STO = Atoms(
    symbols=["Sr", "Ti", "O", "O", "O"],
    scaled_positions=[
        [0.0, 0.0, 0.0],
        [0.5, 0.5, 0.5],
        [0.5, 0.5, 0.0],
        [0.5, 0.0, 0.5],
        [0.0, 0.5, 0.5],
    ],
    cell=[a, a, a],
    pbc=True,
)
ref_STO.calc = calc
ecf_sto = ExpCellFilter(ref_STO)
opt_sto = BFGS(ecf_sto, logfile="sto_relax.log")
opt_sto.run(fmax=0.01)
e_sto = ref_STO.get_potential_energy() / len(ref_STO)

print(f"\nPure BaTiO3 energy/atom: {e_bto:.4f} eV")
print(f"Pure SrTiO3 energy/atom: {e_sto:.4f} eV")

# Mixing energy on A-site sublattice
x_ba = n_ba / n_a_total
x_sr = n_sr / n_a_total
e_mix = e_sqs / len(sqs) - x_ba * e_bto - x_sr * e_sto
print(f"\nMixing energy: {e_mix * 1000:.1f} meV/atom")
print(f"  (per A-site atom: {e_mix * len(sqs) / n_a_total * 1000:.1f} meV)")

# Save structure
write("sqs_BaSrTiO3.cif", sqs)
write("sqs_BaSrTiO3.vasp", sqs, format="vasp")
print("\nSaved: sqs_BaSrTiO3.cif, sqs_BaSrTiO3.vasp")

# ============================================================
# Step 7: Visualize sublattice occupancy
# ============================================================
fig, ax = plt.subplots(figsize=(7, 7))
positions = sqs.get_positions()
symbols = sqs.get_chemical_symbols()

color_map = {"Ba": "#e41a1c", "Sr": "#377eb8", "Ti": "#4daf4a", "O": "#cccccc"}
size_map = {"Ba": 120, "Sr": 100, "Ti": 60, "O": 30}

for species in ["O", "Ti", "Sr", "Ba"]:  # plot order: back to front
    mask = [s == species for s in symbols]
    pos = positions[mask]
    if len(pos) > 0:
        ax.scatter(
            pos[:, 0], pos[:, 1],
            c=color_map[species], s=size_map[species],
            edgecolors="k", linewidth=0.3, label=species, zorder=2 if species in ["Ba", "Sr"] else 1,
        )

ax.set_xlabel("x (A)")
ax.set_ylabel("y (A)")
ax.set_title("(Ba,Sr)TiO$_3$ SQS (xy projection)")
ax.legend()
ax.set_aspect("equal")
plt.tight_layout()
plt.savefig("sqs_perovskite.png", dpi=150)
print("Saved: sqs_perovskite.png")
```

---

### Method I: SQS + LAMMPS for Large Cells

When you need SQS structures with hundreds or thousands of atoms -- beyond what is practical for DFT or even MACE -- use LAMMPS with classical potentials (EAM, MEAM, etc.). This is useful for:
- Converging SQS properties with cell size.
- Computing elastic constants, phonons, or thermal properties of random alloys.
- Screening many compositions before expensive DFT/MACE calculations.

```python
#!/usr/bin/env python3
"""
Large-cell SQS generation + LAMMPS energy minimization.
Generates a 500+ atom SQS for Cu-Au alloy and computes
mixing energy using EAM potential via LAMMPS (through ASE).

Requires: pip install icet lammps  (or LAMMPS accessible via ASE)
EAM potential file: CuAu.eam.alloy (or fetched from NIST repository)
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from ase.build import bulk, make_supercell
from ase.io import write, read
from ase.optimize import BFGS
from ase.constraints import ExpCellFilter
import os
import urllib.request

# ============================================================
# Step 1: Generate large SQS with icet
# ============================================================
from icet import ClusterSpace
from icet.tools.structure_generation import generate_sqs_from_supercells

parent = bulk("Cu", crystalstructure="fcc", a=3.80)

# Use pair-only cluster space for speed with large cells
cluster_space = ClusterSpace(
    structure=parent,
    cutoffs=[6.0],                   # pairs only (faster for large cells)
    chemical_symbols=[["Cu", "Au"]],
)
print(f"Cluster space: {cluster_space.number_of_orbits} orbits (pairs only)")

# Large supercells: target 500+ atoms
supercell_matrices = [
    [[5, 0, 0], [0, 5, 0], [0, 0, 5]],   # 500 atoms (FCC primitive -> 125*4)
    [[6, 0, 0], [0, 6, 0], [0, 0, 4]],   # 576 atoms
    [[4, 4, 0], [4, 0, 4], [0, 4, 4]],   # 512 atoms (more isotropic)
]

supercells = []
for sc_matrix in supercell_matrices:
    sc = make_supercell(parent, sc_matrix)
    supercells.append(sc)
    print(f"Supercell candidate: {len(sc)} atoms")

target_concentrations = {"Cu": 0.75, "Au": 0.25}

print("\nGenerating large SQS (this may take several minutes)...")
sqs = generate_sqs_from_supercells(
    cluster_space=cluster_space,
    supercells=supercells,
    target_concentrations=target_concentrations,
    n_steps=200000,      # more steps for larger cells
    random_seed=42,
)

n_cu = sum(1 for s in sqs.get_chemical_symbols() if s == "Cu")
n_au = sum(1 for s in sqs.get_chemical_symbols() if s == "Au")
print(f"\nLarge SQS: {len(sqs)} atoms, {n_cu} Cu + {n_au} Au")
print(f"Composition: Cu={n_cu/len(sqs):.3f}, Au={n_au/len(sqs):.3f}")

# Save the SQS
write("sqs_large_Cu75Au25.cif", sqs)
print("Saved: sqs_large_Cu75Au25.cif")

# ============================================================
# Step 2: Download or prepare EAM potential
# ============================================================
# Fetch Zhou et al. Cu-Au EAM potential from NIST if not present
eam_file = "CuAu.eam.alloy"
if not os.path.exists(eam_file):
    print(f"\nEAM potential file '{eam_file}' not found.")
    print("Attempting to download Zhou et al. Cu-Au EAM potential from NIST...")
    try:
        url = "https://www.ctcms.nist.gov/potentials/Download/2004--Zhou-X-W-Johnson-R-A-Wadley-H-N-G--Cu-Au/2/CuAu.eam.alloy"
        urllib.request.urlretrieve(url, eam_file)
        print(f"Downloaded: {eam_file}")
    except Exception as e:
        print(f"Download failed: {e}")
        print("Please manually download an EAM potential for Cu-Au and name it CuAu.eam.alloy")
        print("Continuing with MACE fallback...")
        eam_file = None

# ============================================================
# Step 3: Set up LAMMPS calculator via ASE
# ============================================================
use_lammps = False

if eam_file and os.path.exists(eam_file):
    try:
        from ase.calculators.lammpsrun import LAMMPS

        # LAMMPS parameters for EAM
        lammps_params = {
            "pair_style": "eam/alloy",
            "pair_coeff": [f"* * {os.path.abspath(eam_file)} Cu Au"],
        }

        calc_lammps = LAMMPS(
            parameters=lammps_params,
            files=[os.path.abspath(eam_file)],
            keep_tmp_files=False,
            specorder=["Cu", "Au"],
        )
        use_lammps = True
        print("\nLAMMPS calculator initialized with EAM potential")

    except ImportError:
        print("\nLAMMPS not available via ASE. Trying lammpslib...")
        try:
            from ase.calculators.lammpslib import LAMMPSlib

            lammps_cmds = [
                f"pair_style eam/alloy",
                f"pair_coeff * * {os.path.abspath(eam_file)} Cu Au",
            ]
            calc_lammps = LAMMPSlib(
                lmpcmds=lammps_cmds,
                atom_types={"Cu": 1, "Au": 2},
                log_file="lammps_sqs.log",
            )
            use_lammps = True
            print("\nLAMMPS calculator initialized via lammpslib")

        except ImportError:
            print("\nNo LAMMPS interface found. Falling back to MACE.")
            use_lammps = False

# ============================================================
# Step 4: Run energy minimization
# ============================================================
if use_lammps:
    print("\nRunning LAMMPS energy minimization...")
    calc = calc_lammps
else:
    print("\nFalling back to MACE for energy calculation...")
    from mace.calculators import mace_mp
    calc = mace_mp(model="medium", default_dtype="float64")

sqs.calc = calc

# Full cell + position relaxation
ecf = ExpCellFilter(sqs)
opt = BFGS(ecf, logfile="sqs_large_relax.log")
opt.run(fmax=0.01, steps=1000)

e_sqs = sqs.get_potential_energy()
print(f"\nRelaxed SQS energy: {e_sqs:.4f} eV ({len(sqs)} atoms)")
print(f"Energy/atom: {e_sqs / len(sqs):.6f} eV/atom")

# ============================================================
# Step 5: Compute mixing energy from LAMMPS
# ============================================================
# Pure Cu reference
cu_pure = bulk("Cu", crystalstructure="fcc", a=3.615)
cu_sc = make_supercell(cu_pure, [[3, 0, 0], [0, 3, 0], [0, 0, 3]])
cu_sc.calc = calc
ecf_cu = ExpCellFilter(cu_sc)
opt_cu = BFGS(ecf_cu, logfile="cu_pure_relax.log")
opt_cu.run(fmax=0.01)
e_cu = cu_sc.get_potential_energy() / len(cu_sc)

# Pure Au reference
au_pure = bulk("Au", crystalstructure="fcc", a=4.078)
au_sc = make_supercell(au_pure, [[3, 0, 0], [0, 3, 0], [0, 0, 3]])
au_sc.calc = calc
ecf_au = ExpCellFilter(au_sc)
opt_au = BFGS(ecf_au, logfile="au_pure_relax.log")
opt_au.run(fmax=0.01)
e_au = au_sc.get_potential_energy() / len(au_sc)

print(f"\nPure Cu energy/atom: {e_cu:.6f} eV")
print(f"Pure Au energy/atom: {e_au:.6f} eV")

x_cu = n_cu / len(sqs)
x_au = n_au / len(sqs)
e_mix = e_sqs / len(sqs) - x_cu * e_cu - x_au * e_au

print(f"\n== Mixing Energy Results ==")
print(f"Calculator: {'LAMMPS (EAM)' if use_lammps else 'MACE'}")
print(f"System: Cu{x_cu:.2f}Au{x_au:.2f}, {len(sqs)} atoms")
print(f"Mixing energy: {e_mix * 1000:.2f} meV/atom")
print(f"  Positive = endothermic (phase-separating)")
print(f"  Negative = exothermic (ordering tendency)")

# ============================================================
# Step 6: Convergence check -- compare with smaller SQS
# ============================================================
print("\n== Cell-Size Convergence ==")
print(f"{'N_atoms':>8} {'E_mix (meV/atom)':>18}")

# Also compute for a small SQS to check convergence
small_supercells = [make_supercell(parent, [[3, 0, 0], [0, 3, 0], [0, 0, 3]])]  # 108 atoms
small_sqs = generate_sqs_from_supercells(
    cluster_space=cluster_space,
    supercells=small_supercells,
    target_concentrations=target_concentrations,
    n_steps=50000,
    random_seed=123,
)
small_sqs.calc = calc
ecf_small = ExpCellFilter(small_sqs)
opt_small = BFGS(ecf_small, logfile="sqs_small_relax.log")
opt_small.run(fmax=0.01)

e_small = small_sqs.get_potential_energy() / len(small_sqs)
n_cu_s = sum(1 for s in small_sqs.get_chemical_symbols() if s == "Cu")
n_au_s = sum(1 for s in small_sqs.get_chemical_symbols() if s == "Au")
x_cu_s = n_cu_s / len(small_sqs)
x_au_s = n_au_s / len(small_sqs)
e_mix_small = e_small - x_cu_s * e_cu - x_au_s * e_au

print(f"{len(small_sqs):>8d} {e_mix_small * 1000:>18.2f}")
print(f"{len(sqs):>8d} {e_mix * 1000:>18.2f}")

diff = abs(e_mix - e_mix_small) * 1000
print(f"\nDifference: {diff:.2f} meV/atom")
if diff < 5.0:
    print("CONVERGED: Mixing energy is well-converged with cell size")
else:
    print("NOT CONVERGED: Consider an even larger cell or more MC steps")

# ============================================================
# Step 7: Save results and plot
# ============================================================
write("sqs_large_relaxed.cif", sqs)
print("\nSaved: sqs_large_relaxed.cif")

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# Left: Mixing energy convergence
cell_sizes = [len(small_sqs), len(sqs)]
mix_energies = [e_mix_small * 1000, e_mix * 1000]
axes[0].plot(cell_sizes, mix_energies, "o-", color="steelblue", markersize=10, linewidth=2)
axes[0].set_xlabel("Number of atoms in SQS")
axes[0].set_ylabel("Mixing energy (meV/atom)")
axes[0].set_title("SQS Cell-Size Convergence")
axes[0].axhline(0, color="k", linewidth=0.5)

# Right: Structure projection
positions = sqs.get_positions()
syms = sqs.get_chemical_symbols()
colors = ["#1f77b4" if s == "Cu" else "#d4af37" for s in syms]
sizes = [10 if s == "Cu" else 15 for s in syms]
axes[1].scatter(positions[:, 0], positions[:, 1], c=colors, s=sizes, edgecolors="k", linewidth=0.2, alpha=0.7)
axes[1].set_xlabel("x (A)")
axes[1].set_ylabel("y (A)")
axes[1].set_title(f"Large SQS: {len(sqs)} atoms Cu$_{{0.75}}$Au$_{{0.25}}$")
axes[1].set_aspect("equal")

from matplotlib.lines import Line2D
legend_elements = [
    Line2D([0], [0], marker="o", color="w", markerfacecolor="#1f77b4", markersize=6, label="Cu"),
    Line2D([0], [0], marker="o", color="w", markerfacecolor="#d4af37", markersize=7, label="Au"),
]
axes[1].legend(handles=legend_elements)

plt.tight_layout()
plt.savefig("sqs_large_results.png", dpi=150)
print("Saved: sqs_large_results.png")
print("\nDone.")
```

---

## SQS Quality Metrics

### How to Evaluate If a Generated SQS Is Good Enough

| Metric | How to Compute | Excellent | Acceptable | Poor |
|--------|---------------|-----------|------------|------|
| **Max pair correlation deviation** | max |Pi_pair(SQS) - Pi_pair(random)| across all pair shells | < 0.02 | < 0.05 | > 0.10 |
| **Total objective function** | sum of w_shell * delta^2 across all shells | < 0.001 | < 0.01 | > 0.05 |
| **Max triplet correlation deviation** | max |Pi_triplet(SQS) - Pi_triplet(random)| | < 0.05 | < 0.10 | > 0.20 |
| **Warren-Cowley SRO parameter** | alpha_1 = 1 - P_AB/(x_B), where P_AB is fraction of A-B nearest-neighbor pairs | |alpha_1| < 0.02 | |alpha_1| < 0.05 | |alpha_1| > 0.10 |
| **Composition deviation** | |x_actual - x_target| | 0 (exact) | < 0.02 | > 0.05 |

### Quality Evaluation Script

```python
#!/usr/bin/env python3
"""
Standalone SQS quality evaluation.
Reads a CIF file and reports all quality metrics.
"""

import numpy as np
from ase.io import read

def evaluate_sqs_quality(atoms, species_a, species_b, cutoffs=[6.0, 4.5]):
    """
    Comprehensive SQS quality report.

    Args:
        atoms: ASE Atoms object (the SQS structure)
        species_a, species_b: chemical symbols of the two species
        cutoffs: [pair_cutoff, triplet_cutoff] in Angstrom

    Returns:
        dict with all quality metrics
    """
    from icet import ClusterSpace
    from ase.build import bulk

    symbols = atoms.get_chemical_symbols()
    n_total = len(atoms)
    n_b = sum(1 for s in symbols if s == species_b)
    n_a = n_total - n_b
    x_b = n_b / n_total

    # Build cluster space from parent lattice
    # Detect lattice type from the structure
    a_eff = (atoms.get_volume() / n_total * 4) ** (1.0 / 3.0)
    parent = bulk(species_a, crystalstructure="fcc", a=a_eff)

    cs = ClusterSpace(
        structure=parent,
        cutoffs=cutoffs,
        chemical_symbols=[[species_a, species_b]],
    )

    # Get cluster vectors
    try:
        cv = cs.get_cluster_vector(atoms)
    except:
        from icet import ClusterVectorCalculator
        cv = ClusterVectorCalculator(cs).get_cluster_vector(atoms)

    # Target values
    sigma_avg = 2 * x_b - 1
    metrics = {
        "n_atoms": n_total,
        f"n_{species_a}": n_a,
        f"n_{species_b}": n_b,
        f"x_{species_b}": x_b,
        "n_orbits": cs.number_of_orbits,
        "pair_deltas": [],
        "triplet_deltas": [],
    }

    print(f"\n{'='*60}")
    print(f"SQS Quality Report: {species_a}{n_a}{species_b}{n_b}")
    print(f"{'='*60}")
    print(f"Composition: x({species_b}) = {x_b:.4f}")
    print(f"sigma_avg = {sigma_avg:.4f}")
    print(f"\n{'Orbit':>6} {'Order':>6} {'Radius':>8} {'SQS':>10} {'Target':>10} {'Delta':>10} {'Grade':>8}")
    print("-" * 60)

    for i, orb in enumerate(cs.orbit_data):
        idx = i + 1
        target = sigma_avg ** orb["order"]
        delta = cv[idx] - target
        abs_delta = abs(delta)

        if orb["order"] == 2:
            metrics["pair_deltas"].append(abs_delta)
            grade = "A+" if abs_delta < 0.02 else ("A" if abs_delta < 0.05 else ("B" if abs_delta < 0.10 else "F"))
        else:
            metrics["triplet_deltas"].append(abs_delta)
            grade = "A+" if abs_delta < 0.05 else ("A" if abs_delta < 0.10 else ("B" if abs_delta < 0.20 else "F"))

        print(f"{idx:>6d} {orb['order']:>6d} {orb['radius']:>8.3f} "
              f"{cv[idx]:>10.4f} {target:>10.4f} {delta:>10.4f} {grade:>8}")

    metrics["max_pair_delta"] = max(metrics["pair_deltas"]) if metrics["pair_deltas"] else 0
    metrics["max_triplet_delta"] = max(metrics["triplet_deltas"]) if metrics["triplet_deltas"] else 0
    metrics["total_objective"] = sum(d**2 for d in metrics["pair_deltas"] + metrics["triplet_deltas"])

    # Warren-Cowley SRO parameter for 1st NN shell
    distances = atoms.get_all_distances(mic=True)
    nn_dist = np.min(distances[distances > 0.1])
    nn_pairs_ab = 0
    nn_pairs_total = 0
    for i in range(n_total):
        for j in range(i+1, n_total):
            if distances[i,j] < nn_dist * 1.1:
                nn_pairs_total += 1
                if symbols[i] != symbols[j]:
                    nn_pairs_ab += 1
    p_ab = nn_pairs_ab / nn_pairs_total if nn_pairs_total > 0 else 0
    # For random alloy: p_ab_random = 2 * x_a * x_b
    p_ab_random = 2 * (1 - x_b) * x_b
    alpha_sro = 1 - p_ab / p_ab_random if p_ab_random > 0 else 0
    metrics["warren_cowley_alpha1"] = alpha_sro

    print(f"\n{'='*60}")
    print(f"Summary:")
    print(f"  Max pair delta:    {metrics['max_pair_delta']:.4f}  "
          f"({'Excellent' if metrics['max_pair_delta'] < 0.02 else 'Good' if metrics['max_pair_delta'] < 0.05 else 'Acceptable' if metrics['max_pair_delta'] < 0.10 else 'Poor'})")
    print(f"  Max triplet delta: {metrics['max_triplet_delta']:.4f}  "
          f"({'Excellent' if metrics['max_triplet_delta'] < 0.05 else 'Good' if metrics['max_triplet_delta'] < 0.10 else 'Acceptable' if metrics['max_triplet_delta'] < 0.20 else 'Poor'})")
    print(f"  Total objective:   {metrics['total_objective']:.6f}")
    print(f"  Warren-Cowley a1:  {alpha_sro:.4f}  "
          f"({'Excellent' if abs(alpha_sro) < 0.02 else 'Good' if abs(alpha_sro) < 0.05 else 'Acceptable' if abs(alpha_sro) < 0.10 else 'Poor'})")
    print(f"{'='*60}")

    return metrics


# Example usage
if __name__ == "__main__":
    atoms = read("sqs_Cu75Au25.cif")
    metrics = evaluate_sqs_quality(atoms, "Cu", "Au")
```

---

## Correlation Function Targets for Non-Equiatomic Compositions

For a binary A_{1-x}B_x alloy with pseudo-spin mapping sigma_A = -1, sigma_B = +1:

| Composition x(B) | sigma_avg | Pair target | Triplet target | Quadruplet target |
|-------------------|-----------|-------------|----------------|-------------------|
| 0.10 | -0.80 | 0.6400 | -0.5120 | 0.4096 |
| 0.20 | -0.60 | 0.3600 | -0.2160 | 0.1296 |
| 0.25 | -0.50 | 0.2500 | -0.1250 | 0.0625 |
| 0.30 | -0.40 | 0.1600 | -0.0640 | 0.0256 |
| 0.40 | -0.20 | 0.0400 | -0.0080 | 0.0016 |
| 0.50 |  0.00 | 0.0000 |  0.0000 | 0.0000 |
| 0.60 |  0.20 | 0.0400 |  0.0080 | 0.0016 |
| 0.70 |  0.40 | 0.1600 |  0.0640 | 0.0256 |
| 0.75 |  0.50 | 0.2500 |  0.1250 | 0.0625 |
| 0.80 |  0.60 | 0.3600 |  0.2160 | 0.1296 |
| 0.90 |  0.80 | 0.6400 |  0.5120 | 0.4096 |

Note: For **equiatomic** (x = 0.5), all targets are zero -- the SQS should have zero net correlation for all clusters. This is the easiest case to validate.

## Key Parameters

| Parameter | Typical Value | Effect |
|-----------|---------------|--------|
| **Supercell size** | 16--64 atoms | Larger = better random mimicry, but costlier to relax. 32 atoms is a common sweet spot. |
| **Pair cutoff** | 5--8 A | Include enough neighbor shells. At minimum, cover 3rd-nearest-neighbor pairs. |
| **Triplet cutoff** | 3--5 A | Triplet correlations matter for short-range order. Include at least 1st-nearest-neighbor triplets. |
| **MC optimization steps** | 10,000--100,000 | More steps = better convergence. 50,000 is usually sufficient for 32-atom cells. |
| **Target concentrations** | Depends on system | Must be compatible with supercell size (e.g., 25% of 32 = 8 atoms). |
| **Number of supercell shapes** | 3--10 | Try several aspect ratios. More isotropic shapes tend to give better SQS. |
| **Shell weights** (sqsgenerator) | [1, 1/2, 1/3, ...] | Default inverse-shell weighting. Increase weight on shell 1 for stronger NN constraint. |
| **N replicas** | 3--10 | Number of independent SQS per composition. More replicas = better statistical error bars on properties. |
| **Objective** (sqsgenerator) | 0.0 | Target value for objective function. 0 = perfectly random. |
| **Warren-Cowley alpha tolerance** | 0.05 | Maximum acceptable |alpha| for SRO validation. < 0.05 = excellent random character. |
| **SRO number of shells** | 4--8 | Number of neighbor shells to analyze for Warren-Cowley parameters. 6 shells is typical. |
| **SRO distance tolerance** | 0.1 A | Tolerance for grouping neighbor distances into shells. Increase for distorted lattices. |
| **Multi-sublattice chemical_symbols** | Per-site lists | For icet multi-sublattice: one list per site in parent cell. Fixed sites get single-element lists (e.g., `["Ti"]`). |
| **Multi-sublattice cutoffs** | [7.0, 5.0] A | For complex structures (perovskites, spinels), use larger cutoffs to capture A-A interactions across the lattice. |
| **LAMMPS pair_style** | eam/alloy | For large-cell SQS. Match the potential file format. Common: eam, eam/alloy, meam, tersoff. |
| **Large-cell MC steps** | 100,000--500,000 | Large cells need more MC steps to converge. Scale roughly linearly with atom count. |

## Choosing Supercell Size for Composition Sweeps

When sweeping over compositions, the supercell size constrains which concentrations are exactly representable:

| N_atoms | Representable x values (step) | Notes |
|---------|-------------------------------|-------|
| 16 | 0.0625 (1/16) | Sparse; only x = 0.125, 0.25, 0.375, 0.5, 0.625, 0.75, 0.875 |
| 27 | 0.0370 (1/27) | Good for x = 1/3, 2/3 |
| 32 | 0.03125 (1/32) | Good balance of resolution and cost |
| 36 | 0.0278 (1/36) | Good for x = 1/6, 1/3, 1/2 |
| 48 | 0.0208 (1/48) | Fine resolution; expensive for MACE |
| 64 | 0.0156 (1/64) | Very fine; borderline for MACE cost |

**Tip**: For a coarse sweep with x = 0.1, 0.2, ..., 0.9 on 32 atoms, the actual compositions will be rounded (e.g., x = 0.1 -> 3/32 = 0.09375). This is acceptable for most purposes. For exact x = 0.1, use a 10- or 20-atom cell, but the SQS quality will be worse.

## Interpreting Results

### Correlation Function Quality

- **Perfect SQS**: all correlation functions exactly match random-alloy target. In practice, deviations < 0.01 are excellent.
- **Pair correlations** are most important. If pairs match well but triplets deviate, the SQS is still usable for most thermodynamic properties.
- **Large deviations** in pair correlations (> 0.05) indicate the supercell is too small or needs more MC steps.

### Mixing Energy

- **Negative** mixing energy: alloy tends to order (exothermic mixing). Example: Cu-Au forms ordered L1_2 (Cu3Au).
- **Positive** mixing energy: alloy tends to phase-separate (endothermic mixing).
- **Magnitude**: typically 10--200 meV/atom for metallic alloys. Compare with literature values.

### Vegard's Law Deviation

- Plot the effective lattice parameter vs composition and compare with the linear Vegard's law prediction.
- **Positive deviation** (bowing up): atoms pack less efficiently in the alloy than a linear interpolation suggests.
- **Negative deviation** (bowing down): chemical bonding or size effects cause contraction.
- The bowing parameter b is defined as: a(x) = (1-x)*a_A + x*a_B + b*x*(1-x).

### Warren-Cowley SRO Interpretation

- **alpha = 0** for all shells: structure perfectly mimics a random alloy. This is the ideal SQS result.
- **alpha > 0** (positive): like-atom clustering. A-A and B-B pairs are favored over A-B pairs at this distance. The structure has segregation tendencies.
- **alpha < 0** (negative): unlike-atom ordering. A-B pairs are favored. The structure has ordering tendencies (e.g., L1_0, L1_2 superstructures).
- **alpha_1** (1st nearest-neighbor shell) is the most physically important. Higher shells converge to zero faster for finite-size effects.
- **Oscillating sign** across shells: characteristic of many real alloys. SQS should NOT show this -- it indicates the structure is not sufficiently random.
- **|alpha| > 0.10 for shell 1**: the SQS is of poor quality. Regenerate with more MC steps, a larger supercell, or a different supercell shape.
- When comparing SQS quality across methods (icet vs sqsgenerator vs manual), the Warren-Cowley alpha provides a method-independent validation metric.

### Multi-Sublattice Considerations

- For multi-sublattice systems, mixing energy should be referenced to endmembers that share the same crystal structure (e.g., BaTiO3 and SrTiO3 for (Ba,Sr)TiO3).
- Correlations are only meaningful on the disordered sublattice(s). Fixed sublattices contribute zero variance and are excluded from the SQS objective.
- If two sublattices are independently disordered (e.g., A-site and B-site in a double perovskite), icet treats each sublattice's correlations separately.
- Per-sublattice mixing energy: divide total mixing energy by the number of disordered sites (not total atoms) to get a per-disordered-site mixing energy.

### Structure Validation

- Check that the composition is exactly the target (rounding may cause slight deviations).
- Verify that the structure has no unphysical short bonds after relaxation.
- If mixing energy is anomalously large, the SQS may have converged to a locally unfavorable configuration -- regenerate with a different random seed.
- When using multiple replicas, the **standard deviation** of computed properties across replicas quantifies the statistical uncertainty from the finite SQS approximation. If std is large relative to the mean, increase N_replicas or supercell size.

## Common Issues

| Issue | Cause | Solution |
|-------|-------|----------|
| `icet` import fails | Not installed | `pip install icet` |
| `sqsgenerator` import fails | Not installed | `pip install sqsgenerator` or `conda install -c conda-forge sqsgenerator` |
| SQS composition does not match target | Supercell size incompatible with concentration | Choose a supercell size where `n_atoms * x` is an integer |
| MC optimization stuck | Too few steps or bad supercell shape | Increase `n_steps` to 100,000; try more isotropic supercell shapes |
| MACE relaxation diverges | Initial SQS has atoms too close | Check minimum interatomic distance; use a more conservative optimizer (FIRE instead of BFGS) |
| pymatgen SQS requires ATAT | `SQSTransformation` calls external `mcsqs` | Install ATAT or switch to icet (recommended) |
| Memory error for large cells | Too many atoms | Reduce supercell to 32--48 atoms for MACE; use QE for larger cells |
| Correlation function plot looks wrong | Mismatch between icet version conventions | Print raw cluster vectors and verify orbit orders manually |
| Large error bars across replicas | Supercell too small for this composition | Increase supercell size or N_replicas |
| Composition rounding gives wrong stoichiometry | Finite-size effect | Use `mole_fractions_to_composition()` helper to handle rounding consistently |
| sqsgenerator objective not reaching 0 | Finite cell cannot perfectly match random alloy | Acceptable if objective < 0.01; increase cell size or try different shape |
| Multi-concentration sweep takes too long | Too many compositions x replicas x MC steps | Reduce MC steps to 10,000 for initial screening; increase for publication quality |
| Warren-Cowley alpha is large for all shells | SQS generation used too few MC steps | Regenerate SQS with n_steps >= 100,000; verify with SRO analysis before relaxation |
| SRO shell detection fails or merges shells | Distance tolerance too large or too small for the lattice | Adjust `tol` parameter in `get_neighbor_shells()`: try 0.05 for compact lattices, 0.2 for distorted ones |
| Multi-sublattice SQS has wrong species on fixed sites | Parent structure or chemical_symbols list order mismatch | Verify that the chemical_symbols list matches the exact site order in the parent Atoms object |
| icet error: "structure not compatible with cluster space" | Supercell shape or atom count does not match parent lattice | Ensure supercells are built from the same parent structure used to define the ClusterSpace |
| LAMMPS calculator not found | ASE cannot find LAMMPS binary | Set `ASE_LAMMPS_RUN_COMMAND` environment variable or install lammps Python package: `pip install lammps` |
| EAM potential download fails | NIST server unreachable or URL changed | Download potential manually from https://www.ctcms.nist.gov/potentials/ and place in working directory |
| Large-cell SQS takes too long to generate | Triplet correlations scale poorly with cell size | Use pair-only cutoffs (`cutoffs=[6.0]`) for cells > 200 atoms; triplets are less critical for large cells |
| LAMMPS gives different mixing energy than MACE | Different interatomic potential accuracy | Expected; EAM is an approximation. Use LAMMPS for convergence studies and trends, MACE for final values |
| Multi-sublattice SQS quality is poor despite many MC steps | Too few disordered sites in the supercell | Increase supercell size to have at least 8--16 disordered sites per sublattice |
