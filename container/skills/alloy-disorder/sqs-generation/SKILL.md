# Special Quasirandom Structure (SQS) Generation

## When to Use

- You need to model a **random substitutional alloy** (e.g., Ti0.5Al0.5N, Cu0.75Au0.25) but DFT/MACE requires a periodic supercell.
- You want the smallest periodic cell whose **correlation functions** best match the ideal random alloy (all pair, triplet, ... correlations equal zero for equiatomic; or equal the appropriate analytical values for off-stoichiometry).
- You plan to compute mixing energies, lattice parameters, elastic constants, or electronic structure of a disordered alloy.

## Method Selection

| Method | Tool | Pros | Cons |
|--------|------|------|------|
| **icet + Monte Carlo** | `icet` | Rigorous cluster-correlation matching; fast MC optimization; handles arbitrary lattices | Requires `pip install icet` |
| **sqsgenerator** | `sqsgenerator` | Purpose-built SQS tool, parallel, can target specific shells | Requires `pip install sqsgenerator` |
| **pymatgen SQSTransformation** | `pymatgen` | No extra install; integrated with pymatgen Structures | Slower for large cells; fewer options |
| **Manual enumeration** | `pymatgen` + custom code | Full control | Impractical for large cells |

**Recommendation:** Use **icet** for production-quality SQS. It is well-documented, fast, and gives access to correlation functions for validation.

## Prerequisites

```bash
pip install icet
# Already available: ase, pymatgen, mace-torch, numpy, matplotlib
```

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

### Method B: SQS with pymatgen (No Extra Installs)

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

### Method C: Manual Enumeration (Educational)

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

## Key Parameters

| Parameter | Typical Value | Effect |
|-----------|---------------|--------|
| **Supercell size** | 16--64 atoms | Larger = better random mimicry, but costlier to relax. 32 atoms is a common sweet spot. |
| **Pair cutoff** | 5--8 A | Include enough neighbor shells. At minimum, cover 3rd-nearest-neighbor pairs. |
| **Triplet cutoff** | 3--5 A | Triplet correlations matter for short-range order. Include at least 1st-nearest-neighbor triplets. |
| **MC optimization steps** | 10,000--100,000 | More steps = better convergence. 50,000 is usually sufficient for 32-atom cells. |
| **Target concentrations** | Depends on system | Must be compatible with supercell size (e.g., 25% of 32 = 8 atoms). |
| **Number of supercell shapes** | 3--10 | Try several aspect ratios. More isotropic shapes tend to give better SQS. |

## Interpreting Results

### Correlation Function Quality

- **Perfect SQS**: all correlation functions exactly match random-alloy target. In practice, deviations < 0.01 are excellent.
- **Pair correlations** are most important. If pairs match well but triplets deviate, the SQS is still usable for most thermodynamic properties.
- **Large deviations** in pair correlations (> 0.05) indicate the supercell is too small or needs more MC steps.

### Mixing Energy

- **Negative** mixing energy: alloy tends to order (exothermic mixing). Example: Cu-Au forms ordered L1_2 (Cu3Au).
- **Positive** mixing energy: alloy tends to phase-separate (endothermic mixing).
- **Magnitude**: typically 10--200 meV/atom for metallic alloys. Compare with literature values.

### Structure Validation

- Check that the composition is exactly the target (rounding may cause slight deviations).
- Verify that the structure has no unphysical short bonds after relaxation.
- If mixing energy is anomalously large, the SQS may have converged to a locally unfavorable configuration -- regenerate with a different random seed.

## Common Issues

| Issue | Cause | Solution |
|-------|-------|----------|
| `icet` import fails | Not installed | `pip install icet` |
| SQS composition does not match target | Supercell size incompatible with concentration | Choose a supercell size where `n_atoms * x` is an integer |
| MC optimization stuck | Too few steps or bad supercell shape | Increase `n_steps` to 100,000; try more isotropic supercell shapes |
| MACE relaxation diverges | Initial SQS has atoms too close | Check minimum interatomic distance; use a more conservative optimizer (FIRE instead of BFGS) |
| pymatgen SQS requires ATAT | `SQSTransformation` calls external `mcsqs` | Install ATAT or switch to icet (recommended) |
| Memory error for large cells | Too many atoms | Reduce supercell to 32--48 atoms for MACE; use QE for larger cells |
| Correlation function plot looks wrong | Mismatch between icet version conventions | Print raw cluster vectors and verify orbit orders manually |
