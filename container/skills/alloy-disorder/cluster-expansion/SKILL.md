# Cluster Expansion for Alloy Thermodynamics

## When to Use

- You need to predict **alloy phase diagrams**, order-disorder transition temperatures, or ground-state structures across a composition range.
- Direct DFT/MACE enumeration of all configurations is impractical -- cluster expansion (CE) provides a fast lattice Hamiltonian fit to a training set of computed energies.
- You want to run **Monte Carlo simulations** on a lattice model to sample thermodynamic averages at finite temperature.

## Method Selection

| Approach | Tool | Pros | Cons |
|----------|------|------|------|
| **icet full workflow** | `icet` + `mchammer` | Integrated: cluster space, fitting, MC all in one package | Requires `pip install icet` |
| **Manual CE** | Custom Python | Full control over basis, fitting | Significant coding effort |

**Recommendation:** Use **icet** exclusively. It provides ClusterSpace, StructureContainer, cross-validation, and the mchammer Monte Carlo engine in a single coherent package.

## Prerequisites

```bash
pip install icet trainstation
# Already available: ase, pymatgen, mace-torch, numpy, scipy, matplotlib
```

## Theory Summary

A cluster expansion expresses the energy of a lattice configuration sigma as:

```
E(sigma) = sum_alpha  J_alpha * Pi_alpha(sigma)
```

where:
- **alpha** labels symmetry-distinct clusters (point, pair, triplet, ...) on the lattice
- **J_alpha** are the **effective cluster interactions** (ECIs) -- the parameters to fit
- **Pi_alpha(sigma)** are the **cluster correlation functions** of configuration sigma -- computed from the lattice occupation variables

The cluster correlations form a complete orthonormal basis on the configuration space. Truncating at a finite cluster size and range, and fitting the ECIs to DFT/MACE energies, gives a fast Hamiltonian that can predict the energy of any configuration on the same lattice.

## Detailed Steps

### Complete Workflow: Binary FCC Alloy (Cu-Au)

```python
#!/usr/bin/env python3
"""
Complete cluster expansion workflow for FCC Cu-Au:
1. Define cluster space
2. Generate training structures
3. Compute energies with MACE
4. Fit ECIs (with cross-validation)
5. Monte Carlo simulation for order-disorder temperature
6. Plot results
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

# ============================================================
# Step 1: Define the cluster space
# ============================================================
from ase.build import bulk
from icet import ClusterSpace

# FCC primitive cell as parent lattice
parent = bulk("Cu", crystalstructure="fcc", a=3.80)

# Cluster space with pair and triplet interactions
# Cutoffs: pairs up to 7.0 A (~4th neighbor), triplets up to 4.5 A
cluster_space = ClusterSpace(
    structure=parent,
    cutoffs=[7.0, 4.5],
    chemical_symbols=[["Cu", "Au"]],
)

print("=" * 60)
print("CLUSTER SPACE")
print("=" * 60)
print(cluster_space)
print(f"Number of parameters (ECIs): {len(cluster_space)}")

# ============================================================
# Step 2: Generate training structures
# ============================================================
from icet.tools import enumerate_structures

# Enumerate small supercells at various compositions
print("\n" + "=" * 60)
print("GENERATING TRAINING STRUCTURES")
print("=" * 60)

training_structures = []
# Enumerate structures with 1-8 atoms per cell
for n_atoms in range(1, 9):
    structs = enumerate_structures(
        cluster_space=cluster_space,
        sizes=list(range(1, n_atoms + 1)),
        chemical_symbols=[["Cu", "Au"]],
    )
    for s in structs:
        if len(s) <= 8:  # keep manageable size
            training_structures.append(s)

# Remove duplicates by checking compositions and sizes
unique_structures = []
seen = set()
for s in training_structures:
    key = (len(s), tuple(sorted(s.get_chemical_symbols())))
    if key not in seen:
        seen.add(key)
        unique_structures.append(s)

# If too many, subsample; if too few, add random structures
print(f"Enumerated {len(unique_structures)} unique structures")

# Also add some larger random structures for better sampling
from icet.tools.structure_generation import generate_sqs_from_supercells
from ase.build import make_supercell

rng = np.random.default_rng(42)
extra_structures = []
sc_matrix = [[3, 0, 0], [0, 3, 0], [0, 0, 1]]  # 9 atoms
supercell_template = make_supercell(parent, sc_matrix)

for x_au in [0.11, 0.22, 0.33, 0.44, 0.56, 0.67, 0.78, 0.89]:
    n_au = round(x_au * len(supercell_template))
    if n_au == 0 or n_au == len(supercell_template):
        continue
    sc = supercell_template.copy()
    symbols = ["Cu"] * len(sc)
    au_indices = rng.choice(len(sc), size=n_au, replace=False)
    for idx in au_indices:
        symbols[idx] = "Au"
    sc.set_chemical_symbols(symbols)
    extra_structures.append(sc)

all_structures = unique_structures + extra_structures
# Limit total to ~80 structures for efficiency
if len(all_structures) > 80:
    indices = rng.choice(len(all_structures), size=80, replace=False)
    all_structures = [all_structures[i] for i in sorted(indices)]

print(f"Total training structures: {len(all_structures)}")

# ============================================================
# Step 3: Compute energies with MACE
# ============================================================
from ase.optimize import BFGS
from ase.constraints import ExpCellFilter
from mace.calculators import mace_mp

print("\n" + "=" * 60)
print("COMPUTING ENERGIES WITH MACE")
print("=" * 60)

calc = mace_mp(model="medium", default_dtype="float64")

# Pure reference energies (for formation energy)
cu_ref = bulk("Cu", crystalstructure="fcc", a=3.615)
cu_ref.calc = calc
ecf = ExpCellFilter(cu_ref)
BFGS(ecf, logfile=None).run(fmax=0.02)
e_cu_ref = cu_ref.get_potential_energy() / len(cu_ref)

au_ref = bulk("Au", crystalstructure="fcc", a=4.078)
au_ref.calc = calc
ecf = ExpCellFilter(au_ref)
BFGS(ecf, logfile=None).run(fmax=0.02)
e_au_ref = au_ref.get_potential_energy() / len(au_ref)

print(f"Reference: E(Cu) = {e_cu_ref:.4f} eV/atom, E(Au) = {e_au_ref:.4f} eV/atom")

energies = []
valid_structures = []
compositions = []

for i, atoms in enumerate(all_structures):
    atoms_copy = atoms.copy()
    atoms_copy.calc = calc

    # Relax positions only (keep cell fixed for lattice model consistency)
    try:
        opt = BFGS(atoms_copy, logfile=None)
        opt.run(fmax=0.05, steps=100)
        e_total = atoms_copy.get_potential_energy()
        e_per_atom = e_total / len(atoms_copy)

        # Formation energy
        n_au = sum(1 for s in atoms_copy.get_chemical_symbols() if s == "Au")
        x_au = n_au / len(atoms_copy)
        e_form = e_per_atom - (1 - x_au) * e_cu_ref - x_au * e_au_ref

        energies.append(e_total)
        valid_structures.append(atoms_copy)
        compositions.append(x_au)

        if i % 10 == 0:
            print(f"  Structure {i+1}/{len(all_structures)}: "
                  f"{len(atoms_copy)} atoms, x_Au={x_au:.2f}, "
                  f"E_form={e_form*1000:.1f} meV/atom")
    except Exception as e:
        print(f"  Structure {i+1} failed: {e}")

print(f"\nSuccessfully computed {len(valid_structures)} structures")

# ============================================================
# Step 4: Build structure container and fit ECIs
# ============================================================
from icet import StructureContainer
from trainstation import CrossValidationEstimator

print("\n" + "=" * 60)
print("FITTING CLUSTER EXPANSION")
print("=" * 60)

# Build structure container (stores structures + energies + cluster vectors)
sc = StructureContainer(cluster_space=cluster_space)
for atoms, energy in zip(valid_structures, energies):
    try:
        sc.add_structure(
            structure=atoms,
            properties={"energy": energy / len(atoms)},  # energy per atom
        )
    except Exception as e:
        print(f"  Skipping structure: {e}")

print(f"Structure container: {len(sc)} structures")

# Fit with cross-validation using LASSO (L1 regularization)
# This automatically selects the regularization strength
A = sc.get_fit_data(key="energy")[0]  # cluster vector matrix
y = sc.get_fit_data(key="energy")[1]  # energy per atom vector

print(f"Fit matrix shape: {A.shape}")

cve = CrossValidationEstimator(
    fit_data=(A, y),
    fit_method="lasso",
    validation_method="k-fold",
    number_of_splits=10,
)
cve.validate()
cve.train()

print(f"\nCross-validation RMSE: {cve.rmse_validation * 1000:.1f} meV/atom")
print(f"Training RMSE:         {cve.rmse_train * 1000:.1f} meV/atom")
print(f"Number of nonzero ECIs: {np.count_nonzero(cve.parameters)}/{len(cve.parameters)}")

# ============================================================
# Step 5: Create ClusterExpansion object
# ============================================================
from icet import ClusterExpansion

ce = ClusterExpansion(
    cluster_space=cluster_space,
    parameters=cve.parameters,
)
print("\n" + str(ce))

# Save the CE for later use
ce.write("cluster_expansion.ce")
print("Saved: cluster_expansion.ce")

# ============================================================
# Step 6: Plot ECIs
# ============================================================
eci_values = cve.parameters
orbit_data = cluster_space.orbit_data

fig, ax = plt.subplots(figsize=(10, 5))

# Separate by cluster order
pair_ecis = []
pair_radii = []
triplet_ecis = []
triplet_indices = []

for i, orb in enumerate(orbit_data):
    eci_idx = i + 1  # skip zerolet
    if eci_idx >= len(eci_values):
        break
    if orb["order"] == 2:
        pair_ecis.append(eci_values[eci_idx])
        pair_radii.append(orb["radius"])
    elif orb["order"] == 3:
        triplet_ecis.append(eci_values[eci_idx])
        triplet_indices.append(eci_idx)

ax.bar(range(len(pair_ecis)), [e * 1000 for e in pair_ecis],
       color="steelblue", label=f"Pair ECIs ({len(pair_ecis)})")
offset = len(pair_ecis)
ax.bar(range(offset, offset + len(triplet_ecis)),
       [e * 1000 for e in triplet_ecis],
       color="salmon", label=f"Triplet ECIs ({len(triplet_ecis)})")

ax.set_xlabel("Orbit index")
ax.set_ylabel("ECI (meV)")
ax.set_title("Effective Cluster Interactions -- Cu-Au FCC")
ax.axhline(0, color="k", linewidth=0.5)
ax.legend()
plt.tight_layout()
plt.savefig("eci_values.png", dpi=150)
print("\nSaved: eci_values.png")

# ============================================================
# Step 7: Plot formation energy vs composition (CE vs MACE)
# ============================================================
fig2, ax2 = plt.subplots(figsize=(8, 5))

# MACE (training data) formation energies
form_energies_mace = []
for atoms, energy in zip(valid_structures, energies):
    n_au = sum(1 for s in atoms.get_chemical_symbols() if s == "Au")
    x_au = n_au / len(atoms)
    e_form = energy / len(atoms) - (1 - x_au) * e_cu_ref - x_au * e_au_ref
    form_energies_mace.append(e_form * 1000)  # meV

ax2.scatter(compositions, form_energies_mace, c="steelblue", s=30,
            alpha=0.7, label="MACE (training)", zorder=5)

# CE predictions
form_energies_ce = []
for atoms in valid_structures:
    try:
        e_ce = ce.predict(atoms) * len(atoms)  # total energy from CE
        e_ce_per_atom = e_ce / len(atoms)
        n_au = sum(1 for s in atoms.get_chemical_symbols() if s == "Au")
        x_au = n_au / len(atoms)
        e_form_ce = e_ce_per_atom - (1 - x_au) * e_cu_ref - x_au * e_au_ref
        form_energies_ce.append(e_form_ce * 1000)
    except Exception:
        form_energies_ce.append(np.nan)

ax2.scatter(compositions, form_energies_ce, c="salmon", s=30, marker="x",
            alpha=0.7, label="CE prediction", zorder=4)

ax2.set_xlabel("Au concentration x")
ax2.set_ylabel("Formation energy (meV/atom)")
ax2.set_title("Formation Energy: Cu$_{1-x}$Au$_x$ (FCC)")
ax2.legend()
ax2.axhline(0, color="k", linewidth=0.5)
ax2.set_xlim(-0.05, 1.05)
plt.tight_layout()
plt.savefig("formation_energy_vs_composition.png", dpi=150)
print("Saved: formation_energy_vs_composition.png")

# ============================================================
# Step 8: Monte Carlo simulation for order-disorder temperature
# ============================================================
from mchammer import CanonicalAnnealing
from mchammer.calculators import ClusterExpansionCalculator

print("\n" + "=" * 60)
print("MONTE CARLO: ORDER-DISORDER TRANSITION")
print("=" * 60)

# Build a large supercell for MC
mc_supercell = make_supercell(parent, [[8, 0, 0], [0, 8, 0], [0, 0, 8]])
n_mc = len(mc_supercell)
print(f"MC supercell: {n_mc} atoms")

# Set composition to Cu3Au (x_Au = 0.25)
n_au_mc = round(0.25 * n_mc)
symbols = ["Cu"] * n_mc
rng = np.random.default_rng(123)
au_indices = rng.choice(n_mc, size=n_au_mc, replace=False)
for idx in au_indices:
    symbols[idx] = "Au"
mc_supercell.set_chemical_symbols(symbols)

# Cluster expansion calculator for MC
ce_calc = ClusterExpansionCalculator(
    atoms=mc_supercell,
    cluster_expansion=ce,
)

# Simulated annealing: cool from high T to low T
mc = CanonicalAnnealing(
    atoms=mc_supercell,
    calculator=ce_calc,
    T_start=1200,       # start temperature (K)
    T_stop=50,          # end temperature (K)
    n_steps=200000,     # total MC steps
    cooling_function="exponential",
    random_seed=42,
)

print("Running simulated annealing (this may take a few minutes)...")
mc.run()
print("MC simulation complete.")

# Extract thermodynamic data
dc = mc.data_container
temperatures = dc.get("temperature")
energies_mc = dc.get("potential")  # energy per atom
print(f"MC data points: {len(temperatures)}")

# ============================================================
# Step 9: Compute heat capacity and find transition
# ============================================================
# Bin data by temperature and compute <E> and C_v = d<E>/dT
from scipy.ndimage import uniform_filter1d

# Sort by temperature
sort_idx = np.argsort(temperatures)
T_sorted = np.array(temperatures)[sort_idx]
E_sorted = np.array(energies_mc)[sort_idx]

# Bin into temperature windows
n_bins = 100
T_bins = np.linspace(T_sorted.min(), T_sorted.max(), n_bins + 1)
T_centers = 0.5 * (T_bins[:-1] + T_bins[1:])
E_means = np.zeros(n_bins)
E2_means = np.zeros(n_bins)

for i in range(n_bins):
    mask = (T_sorted >= T_bins[i]) & (T_sorted < T_bins[i + 1])
    if np.sum(mask) > 0:
        E_means[i] = np.mean(E_sorted[mask])
        E2_means[i] = np.mean(E_sorted[mask] ** 2)
    else:
        E_means[i] = np.nan
        E2_means[i] = np.nan

# Heat capacity from energy fluctuations: C_v = (<E^2> - <E>^2) / (kB * T^2)
kB = 8.617333e-5  # eV/K
valid = ~np.isnan(E_means) & (T_centers > 0)
Cv = (E2_means[valid] - E_means[valid] ** 2) / (kB * T_centers[valid] ** 2)

# Find peak (transition temperature)
Cv_smooth = uniform_filter1d(Cv, size=5)
T_transition_idx = np.argmax(Cv_smooth)
T_transition = T_centers[valid][T_transition_idx]
print(f"\nEstimated order-disorder temperature: {T_transition:.0f} K")
print(f"(Experimental Cu3Au: ~663 K)")

# ============================================================
# Step 10: Plot phase diagram / MC results
# ============================================================
fig3, (ax3a, ax3b) = plt.subplots(1, 2, figsize=(14, 5))

# Energy vs temperature
ax3a.scatter(T_sorted[::10], E_sorted[::10] * 1000, s=1, alpha=0.3, c="steelblue")
ax3a.plot(T_centers[valid], E_means[valid] * 1000, "r-", linewidth=2, label="<E>(T)")
ax3a.set_xlabel("Temperature (K)")
ax3a.set_ylabel("Energy (meV/atom)")
ax3a.set_title("MC Energy vs Temperature -- Cu$_3$Au")
ax3a.legend()
ax3a.axvline(T_transition, color="gray", linestyle="--", alpha=0.5,
             label=f"T_c ~ {T_transition:.0f} K")
ax3a.legend()

# Heat capacity vs temperature
ax3b.plot(T_centers[valid], Cv_smooth * 1000, "r-", linewidth=2)
ax3b.fill_between(T_centers[valid], 0, Cv_smooth * 1000, alpha=0.2, color="salmon")
ax3b.set_xlabel("Temperature (K)")
ax3b.set_ylabel("C$_v$ (meV/K/atom)")
ax3b.set_title("Heat Capacity -- Cu$_3$Au")
ax3b.axvline(T_transition, color="gray", linestyle="--",
             label=f"T_c ~ {T_transition:.0f} K")
ax3b.legend()

plt.tight_layout()
plt.savefig("mc_phase_diagram.png", dpi=150)
print("Saved: mc_phase_diagram.png")

# ============================================================
# Step 11: Schematic phase diagram (fixed compositions)
# ============================================================
# Run MC at several compositions to map out the phase boundary
print("\nMapping phase boundary at multiple compositions...")

compositions_mc = [0.10, 0.25, 0.50, 0.75, 0.90]
T_transitions_all = []

for x_au in compositions_mc:
    mc_cell = make_supercell(parent, [[6, 0, 0], [0, 6, 0], [0, 0, 6]])
    n = len(mc_cell)
    n_au_comp = round(x_au * n)
    if n_au_comp == 0 or n_au_comp == n:
        T_transitions_all.append(0)
        continue

    syms = ["Cu"] * n
    for idx in rng.choice(n, size=n_au_comp, replace=False):
        syms[idx] = "Au"
    mc_cell.set_chemical_symbols(syms)

    try:
        ce_calc_comp = ClusterExpansionCalculator(mc_cell, ce)
        mc_comp = CanonicalAnnealing(
            atoms=mc_cell,
            calculator=ce_calc_comp,
            T_start=1000,
            T_stop=50,
            n_steps=80000,
            cooling_function="exponential",
            random_seed=42,
        )
        mc_comp.run()

        dc_comp = mc_comp.data_container
        T_comp = np.array(dc_comp.get("temperature"))
        E_comp = np.array(dc_comp.get("potential"))

        sort_idx = np.argsort(T_comp)
        T_comp = T_comp[sort_idx]
        E_comp = E_comp[sort_idx]

        n_b = 50
        T_b = np.linspace(T_comp.min(), T_comp.max(), n_b + 1)
        T_c = 0.5 * (T_b[:-1] + T_b[1:])
        E_m = np.array([np.mean(E_comp[(T_comp >= T_b[i]) & (T_comp < T_b[i+1])])
                        if np.sum((T_comp >= T_b[i]) & (T_comp < T_b[i+1])) > 0
                        else np.nan
                        for i in range(n_b)])
        E2_m = np.array([np.mean(E_comp[(T_comp >= T_b[i]) & (T_comp < T_b[i+1])]**2)
                         if np.sum((T_comp >= T_b[i]) & (T_comp < T_b[i+1])) > 0
                         else np.nan
                         for i in range(n_b)])

        v = ~np.isnan(E_m) & (T_c > 0)
        Cv_comp = (E2_m[v] - E_m[v]**2) / (kB * T_c[v]**2)
        Cv_comp_smooth = uniform_filter1d(Cv_comp, size=3)
        T_c_comp = T_c[v][np.argmax(Cv_comp_smooth)]
        T_transitions_all.append(T_c_comp)
        print(f"  x_Au = {x_au:.2f}: T_c ~ {T_c_comp:.0f} K")
    except Exception as e:
        print(f"  x_Au = {x_au:.2f}: MC failed -- {e}")
        T_transitions_all.append(0)

fig4, ax4 = plt.subplots(figsize=(8, 6))
ax4.plot(compositions_mc, T_transitions_all, "ro-", markersize=8, linewidth=2)
ax4.fill_between(compositions_mc, 0, T_transitions_all, alpha=0.15, color="salmon")
ax4.set_xlabel("Au concentration x")
ax4.set_ylabel("Order-disorder temperature (K)")
ax4.set_title("Phase Diagram -- Cu$_{1-x}$Au$_x$ (from Cluster Expansion + MC)")
ax4.set_xlim(0, 1)
ax4.set_ylim(0, max(T_transitions_all) * 1.2 if max(T_transitions_all) > 0 else 1000)
ax4.text(0.25, max(T_transitions_all) * 0.5, "Ordered\n(L1$_2$, L1$_0$)",
         ha="center", fontsize=12, color="steelblue")
ax4.text(0.75, max(T_transitions_all) * 1.1, "Disordered\n(FCC solid solution)",
         ha="center", fontsize=12, color="gray")
plt.tight_layout()
plt.savefig("phase_diagram_CuAu.png", dpi=150)
print("Saved: phase_diagram_CuAu.png")

print("\n" + "=" * 60)
print("WORKFLOW COMPLETE")
print("=" * 60)
```

### Alternative: Using QE for Training Energies

If higher accuracy is needed, replace the MACE energy evaluation step with Quantum ESPRESSO SCF calculations:

```python
#!/usr/bin/env python3
"""
Replace MACE with QE pw.x for computing training structure energies.
Insert this in place of Step 3 above.
"""

import os
from ase.io import write as ase_write

def run_qe_scf(atoms, calc_dir, pseudopotentials, ecutwfc=60, kpoints_density=4.0):
    """
    Run a QE SCF calculation for the given atoms.
    Returns total energy in eV.
    """
    os.makedirs(calc_dir, exist_ok=True)

    # Determine k-point grid from density
    from ase.calculators.espresso import Espresso

    calc = Espresso(
        pseudopotentials=pseudopotentials,
        tstress=True,
        tprnfor=True,
        input_data={
            "control": {
                "calculation": "scf",
                "prefix": "pwscf",
                "outdir": os.path.join(calc_dir, "tmp"),
                "pseudo_dir": "/home/pseudo",
            },
            "system": {
                "ecutwfc": ecutwfc,
                "ecutrho": ecutwfc * 8,
                "occupations": "smearing",
                "smearing": "cold",
                "degauss": 0.02,
            },
            "electrons": {
                "conv_thr": 1.0e-6,
                "mixing_beta": 0.4,
            },
        },
        kpts=(kpoints_density, kpoints_density, kpoints_density),
        directory=calc_dir,
    )

    atoms.calc = calc
    energy = atoms.get_potential_energy()
    return energy

# Example usage:
# pseudopotentials = {"Cu": "Cu.pbe-dn-kjpaw_psl.1.0.0.UPF",
#                     "Au": "Au.pbe-dn-kjpaw_psl.1.0.1.UPF"}
# for i, atoms in enumerate(all_structures):
#     e = run_qe_scf(atoms, f"qe_calc_{i:04d}", pseudopotentials)
#     energies.append(e)
```

## Key Parameters

| Parameter | Typical Value | Effect |
|-----------|---------------|--------|
| **Pair cutoff** | 6--8 A | Controls range of pair interactions. Too short misses important interactions; too long overfits. |
| **Triplet cutoff** | 4--5 A | Usually shorter than pair cutoff. Triplets capture 3-body ordering tendencies. |
| **Quadruplet cutoff** | 3--4 A (optional) | Rarely needed. Include only if pair+triplet gives poor CV score. |
| **Number of training structures** | 30--100 | More is better but plateaus around 50-80 for binary alloys. Must exceed number of ECIs. |
| **Regularization method** | LASSO (L1) | Automatically zeroes out unimportant ECIs. Use `ardr` for Bayesian alternative. |
| **Cross-validation splits** | 5--10 | 10-fold is standard. Gives robust estimate of prediction error. |
| **MC supercell size** | 6x6x6 to 10x10x10 | Larger reduces finite-size effects. 8x8x8 FCC = 2048 atoms is typical. |
| **MC steps** | 100,000--500,000 | Must be enough for equilibration at each temperature. Check convergence of running averages. |
| **Temperature range** | 50--1500 K | Cover expected transition. Cu-Au orders below ~700 K. |

## Interpreting Results

### ECI Values

- **Zerolet** (J_0): sets the energy scale; includes the average energy contribution.
- **Point** (J_1): chemical potential-like term; reflects energy asymmetry between species.
- **Nearest-neighbor pair** (J_2,1): the dominant interaction. Negative = ordering (unlike atoms attract), positive = clustering (like atoms attract).
- **Longer-range pairs**: should decay with distance. If J at 4th neighbor is comparable to J at 1st neighbor, the CE may be poorly converged.
- **Triplets**: capture non-pairwise effects. Usually smaller than pair ECIs.

### Cross-Validation Score

- **RMSE < 5 meV/atom**: excellent CE, suitable for quantitative predictions.
- **RMSE 5-15 meV/atom**: good, usable for qualitative phase diagram features.
- **RMSE > 20 meV/atom**: poor fit. Add more training data, adjust cutoffs, or check for outliers.

### Phase Diagram

- **Order-disorder temperature**: location of the heat capacity peak. Compare with experiment.
- **Finite-size effects**: the transition will be broadened and slightly shifted in small MC cells. Use at least 6x6x6.
- **Hysteresis**: simulated annealing may supercool. For accurate T_c, run isothermal MC at several temperatures near the transition and check for discontinuity in order parameter.

### Ground-State Search

```python
# Use the CE to search for ground-state orderings
from icet.tools.ground_state_finder import GroundStateFinder

gsf = GroundStateFinder(ce, supercell_size=[4, 4, 4])
gs = gsf.get_ground_state(species_count={"Cu": 48, "Au": 16})  # Cu3Au in 4x4x4
print(f"Ground state energy: {ce.predict(gs):.4f} eV/atom")
```

## Common Issues

| Issue | Cause | Solution |
|-------|-------|----------|
| `icet` import error | Not installed | `pip install icet` |
| `trainstation` import error | Not installed | `pip install trainstation` |
| CV RMSE very large | Insufficient training data or outliers | Add more structures; check for unconverged MACE/QE calculations |
| More ECIs than training structures | Cluster space too large | Reduce cutoffs, or add more training structures |
| MC energy does not converge | Too few steps or supercell too small | Increase `n_steps`; increase supercell |
| Negative heat capacity | Statistical noise or insufficient sampling | Increase MC steps per temperature bin |
| Phase boundary looks wrong | CE inaccurate at some compositions | Add more training structures at undersampled compositions |
| `enumerate_structures` runs forever | Too many atoms requested | Limit to 8--10 atoms per cell; supplement with random structures |
| Ground state finder slow | Large supercell | Reduce to 3x3x3 or 4x4x4 for initial search |
| MACE gives unreasonable energies | Unusual composition or structure | Verify MACE model covers the element pair; compare a few structures with QE |
