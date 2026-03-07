# Alloy Builder

## When to Use

- You need to build a random substitutional alloy model (e.g., Cu0.5Zn0.5, Ti0.5Al0.5N).
- You want to generate a Special Quasi-random Structure (SQS) that best represents a random alloy with a finite supercell.
- You need multiple random alloy configurations for ensemble averaging.
- You want to study how alloy composition affects properties (band gap, lattice parameter, etc.).

## Method Selection

| Criterion | Random substitution (pymatgen) | SQS (icet/ATAT) | Enumeration (pymatgen) |
|---|---|---|---|
| Physical accuracy | Poor for small cells | Best representation of random alloy | All symmetry-distinct orderings |
| Speed | Instant | Seconds to minutes | Exponential with cell size |
| Correlations | Uncontrolled | Matches random alloy pair correlations | Exact for each ordering |
| Best for | Quick models, large cells | Publication, small-medium cells | Systematic studies of all orderings |

## Prerequisites

- pymatgen (pre-installed)
- numpy (pre-installed)
- Optional: `pip install icet` for SQS generation via Monte Carlo
- Optional: `pip install sqsgenerator` for alternative SQS approach

---

## Detailed Steps

### Method A: Random Substitutional Alloy (pymatgen)

```python
#!/usr/bin/env python3
"""
Build random substitutional alloy models using pymatgen.
Example: Cu-Zn brass alloy at various compositions.
"""

import numpy as np
import json
from pymatgen.core import Structure, Element, Lattice
from pymatgen.io.vasp import Poscar
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
from pymatgen.io.ase import AseAtomsAdaptor
from pathlib import Path

# ============================================================
# 1. Build the host structure
# ============================================================
# FCC Cu as the base structure
host = Structure.from_spacegroup(
    "Fm-3m",
    lattice=Lattice.cubic(3.615),
    species=["Cu"],
    coords=[[0, 0, 0]],
)
print(f"Host: {host.composition.reduced_formula}, {len(host)} atoms")

# Make supercell (need enough atoms for meaningful composition)
sc_size = [3, 3, 3]
host_sc = host.copy()
host_sc.make_supercell(sc_size)
n_atoms = len(host_sc)
print(f"Supercell: {sc_size}, {n_atoms} atoms")

# ============================================================
# 2. Random substitution at a given composition
# ============================================================
def random_substitution(structure, target_species, fraction, seed=42):
    """
    Randomly substitute a fraction of atoms with target_species.

    Parameters
    ----------
    structure : Structure
        Host structure (all same species).
    target_species : str
        Element to substitute (e.g., "Zn").
    fraction : float
        Fraction of atoms to substitute (0 to 1).
    seed : int
        Random seed for reproducibility.

    Returns
    -------
    Structure
        Alloy structure with random substitution.
    """
    rng = np.random.default_rng(seed)
    n_total = len(structure)
    n_substitute = int(round(fraction * n_total))

    alloy = structure.copy()
    indices = rng.choice(n_total, size=n_substitute, replace=False)

    for idx in sorted(indices, reverse=True):
        alloy.replace(idx, target_species)

    return alloy

# Generate alloy at x = 0.25 (Cu0.75Zn0.25)
alloy_025 = random_substitution(host_sc, "Zn", 0.25, seed=42)
print(f"\nCu0.75Zn0.25: {alloy_025.composition}")
alloy_025.to("alloy_Cu75Zn25.cif")
Poscar(alloy_025).write_file("POSCAR_Cu75Zn25")

# ============================================================
# 3. Composition scan with multiple random seeds
# ============================================================
print("\n=== Composition scan ===")

output_dir = Path("alloy_models")
output_dir.mkdir(exist_ok=True)

compositions = [0.1, 0.2, 0.25, 0.3, 0.4, 0.5]
n_configs = 3  # number of random configurations per composition

alloy_data = []

for x in compositions:
    print(f"\n  x(Zn) = {x:.2f}:")
    for seed in range(n_configs):
        alloy = random_substitution(host_sc, "Zn", x, seed=seed)

        # Actual composition (rounding to integers may differ slightly)
        n_zn = sum(1 for s in alloy if s.specie == Element("Zn"))
        actual_x = n_zn / len(alloy)

        filename = f"Cu{1-x:.2f}Zn{x:.2f}_seed{seed}"
        alloy.to(str(output_dir / f"{filename}.cif"))

        # Vegard's law lattice parameter estimate
        a_cu = 3.615
        a_zn_fcc = 3.95  # hypothetical FCC Zn
        a_vegard = (1 - actual_x) * a_cu + actual_x * a_zn_fcc

        print(f"    seed={seed}: n_Zn={n_zn}/{len(alloy)}, "
              f"actual x={actual_x:.3f}, "
              f"a(Vegard)={a_vegard:.3f} A")

        alloy_data.append({
            "x_target": x,
            "x_actual": actual_x,
            "seed": seed,
            "n_atoms": len(alloy),
            "n_Zn": n_zn,
            "filename": filename,
        })

with open(str(output_dir / "alloy_data.json"), "w") as f:
    json.dump(alloy_data, f, indent=2)

# ============================================================
# 4. Multi-component alloy (High-Entropy Alloy)
# ============================================================
print("\n=== High-Entropy Alloy (HEA) ===")

def random_multicomponent_alloy(structure, species_fractions, seed=42):
    """
    Create a multi-component random alloy.

    Parameters
    ----------
    structure : Structure
        Host supercell.
    species_fractions : dict
        {element: fraction}, e.g., {"Fe": 0.2, "Co": 0.2, "Ni": 0.2, "Cr": 0.2, "Mn": 0.2}
    seed : int
        Random seed.

    Returns
    -------
    Structure
    """
    rng = np.random.default_rng(seed)
    n_total = len(structure)
    alloy = structure.copy()

    # Assign species to each site
    species_list = []
    for elem, frac in species_fractions.items():
        n = int(round(frac * n_total))
        species_list.extend([elem] * n)

    # Pad or trim to match total
    while len(species_list) < n_total:
        species_list.append(list(species_fractions.keys())[0])
    species_list = species_list[:n_total]

    rng.shuffle(species_list)

    for i, sp in enumerate(species_list):
        alloy.replace(i, sp)

    return alloy

# Cantor alloy (equiatomic quinary)
hea_fractions = {"Fe": 0.2, "Co": 0.2, "Ni": 0.2, "Cr": 0.2, "Mn": 0.2}

# Need larger supercell for 5 components
host_hea = host.copy()
host_hea.make_supercell([4, 4, 4])  # 256 atoms for FCC

hea = random_multicomponent_alloy(host_hea, hea_fractions, seed=42)
print(f"  HEA composition: {hea.composition}")
hea.to(str(output_dir / "CantorAlloy.cif"))
Poscar(hea).write_file(str(output_dir / "POSCAR_CantorAlloy"))

# ============================================================
# 5. Relax alloy with MACE
# ============================================================
print("\n=== MACE relaxation of alloy ===")

import warnings
warnings.filterwarnings("ignore")
from mace.calculators import mace_mp
from ase.optimize import LBFGS
from ase.filters import FrechetCellFilter
from ase.io import write as ase_write

adaptor = AseAtomsAdaptor()
calc = mace_mp(model="medium", dispersion=False, default_dtype="float64")

atoms = adaptor.get_atoms(alloy_025)
atoms.calc = calc

ecf = FrechetCellFilter(atoms)
opt = LBFGS(ecf, logfile="alloy_relax.log")
opt.run(fmax=0.005, steps=300)

e_alloy = atoms.get_potential_energy()
print(f"  Relaxed energy: {e_alloy:.4f} eV ({len(atoms)} atoms)")
print(f"  E/atom: {e_alloy/len(atoms):.4f} eV")
print(f"  Relaxed cell: {np.round(atoms.cell.cellpar()[:3], 3)}")
ase_write("alloy_Cu75Zn25_relaxed.cif", atoms)

# ============================================================
# 6. Write QE input for alloy
# ============================================================
def write_qe_alloy_input(alloy, pseudo_dir="./pseudo", filename="alloy_scf.in"):
    """Write QE input for an alloy calculation."""
    cell = alloy.lattice.matrix
    elements = sorted(set(str(s.specie) for s in alloy))

    cell_lines = "\n".join(f"  {v[0]:.10f}  {v[1]:.10f}  {v[2]:.10f}" for v in cell)

    species_lines = []
    for el in elements:
        from pymatgen.core.periodic_table import Element as PmgElem
        mass = PmgElem(el).atomic_mass
        species_lines.append(f"  {el:4s} {mass:10.4f}  {el}.pbe-n-rrkjus_psl.1.0.0.UPF")

    pos_lines = []
    for site in alloy:
        fc = site.frac_coords
        pos_lines.append(f"  {str(site.specie):4s} {fc[0]:.10f} {fc[1]:.10f} {fc[2]:.10f}")

    lengths = alloy.lattice.abc
    k_grid = [max(1, int(np.ceil(30.0 / l))) for l in lengths]

    qe_input = f"""&CONTROL
    calculation = 'vc-relax'
    prefix      = 'alloy'
    outdir      = './tmp'
    pseudo_dir  = '{pseudo_dir}'
    tprnfor     = .true.
    tstress     = .true.
/
&SYSTEM
    ibrav       = 0
    nat         = {len(alloy)}
    ntyp        = {len(elements)}
    ecutwfc     = 50.0
    ecutrho     = 400.0
    occupations = 'smearing'
    smearing    = 'mv'
    degauss     = 0.02
/
&ELECTRONS
    conv_thr    = 1.0d-8
    mixing_beta = 0.3
/
&IONS
    ion_dynamics = 'bfgs'
/
&CELL
    cell_dynamics = 'bfgs'
/

CELL_PARAMETERS angstrom
{cell_lines}

ATOMIC_SPECIES
{chr(10).join(species_lines)}

ATOMIC_POSITIONS crystal
{chr(10).join(pos_lines)}

K_POINTS automatic
  {k_grid[0]} {k_grid[1]} {k_grid[2]}  0 0 0
"""
    with open(filename, "w") as f:
        f.write(qe_input)
    print(f"QE input written: {filename} ({len(alloy)} atoms)")

write_qe_alloy_input(alloy_025)
```

### Method B: SQS Generation

```python
#!/usr/bin/env python3
"""
Generate Special Quasi-random Structures (SQS) for alloys.
SQS structures best represent a random alloy with a finite supercell
by matching the pair correlation functions of the random alloy.

Method 1: Using icet (recommended)
Method 2: Using pymatgen enumeration + correlation matching
"""

import numpy as np
from pymatgen.core import Structure, Lattice
from pymatgen.io.vasp import Poscar
from pathlib import Path

output_dir = Path("sqs_models")
output_dir.mkdir(exist_ok=True)

# ============================================================
# Method 1: SQS using icet (Monte Carlo approach)
# ============================================================
print("=== SQS generation with icet ===")

try:
    from icet import ClusterSpace
    from icet.tools import enumerate_structures
    from icet.tools.structure_generation import generate_sqs_from_icet
    # Note: the exact API depends on icet version.
    # If generate_sqs_from_icet is not available, use the mchammer approach below.
    HAS_ICET = True
except ImportError:
    HAS_ICET = False
    print("  icet not installed. Install with: pip install icet")
    print("  Falling back to Method 2 (pymatgen).")

if HAS_ICET:
    from icet import ClusterSpace
    from ase.build import bulk
    from ase.io import write as ase_write

    # Build primitive cell
    prim = bulk("Cu", "fcc", a=3.615)

    # Define cluster space (pair correlations up to cutoff distance)
    cs = ClusterSpace(
        structure=prim,
        cutoffs=[8.0],          # pair cutoff in Angstrom
        chemical_symbols=["Cu", "Zn"],  # allowed species
    )
    print(f"  Cluster space: {len(cs)} orbits")

    # Target composition
    target_concentrations = {"Cu": 0.75, "Zn": 0.25}

    # Generate SQS using Monte Carlo simulated annealing
    from icet.tools.structure_generation import (
        generate_sqs,
    )

    supercell_size = [3, 3, 3]  # 3x3x3 supercell
    sc = prim.repeat(supercell_size)
    n_atoms = len(sc)
    n_Zn = int(round(0.25 * n_atoms))

    sqs = generate_sqs(
        cluster_space=cs,
        max_size=n_atoms,
        target_concentrations=target_concentrations,
        n_steps=50000,           # MC steps
        T_start=5.0,            # Starting temperature
        T_stop=0.001,           # Final temperature
    )

    print(f"  SQS generated: {len(sqs)} atoms")
    print(f"  Composition: {sqs.get_chemical_formula()}")
    ase_write(str(output_dir / "SQS_Cu75Zn25.cif"), sqs)
    print(f"  Saved: {output_dir}/SQS_Cu75Zn25.cif")

# ============================================================
# Method 2: Brute-force SQS via random search with correlation
# ============================================================
print("\n=== SQS via random search (no icet needed) ===")

from pymatgen.core import Structure, Element, Lattice

host = Structure.from_spacegroup(
    "Fm-3m",
    lattice=Lattice.cubic(3.615),
    species=["Cu"],
    coords=[[0, 0, 0]],
)
host.make_supercell([3, 3, 3])
n_atoms = len(host)
n_substitute = int(round(0.25 * n_atoms))  # 25% Zn

def compute_pair_correlation(structure, target_species="Zn",
                              cutoff_shells=None):
    """
    Compute pair correlation function for a binary alloy.
    Returns the Warren-Cowley short-range order parameter for each
    neighbor shell.

    alpha_i = 1 - P(AB at shell i) / x_B
    alpha = 0 for random alloy
    alpha > 0 for clustering (like atoms prefer each other)
    alpha < 0 for ordering (unlike atoms prefer each other)
    """
    from pymatgen.analysis.local_env import VoronoiNN

    x_B = sum(1 for s in structure if str(s.specie) == target_species) / len(structure)
    if x_B == 0 or x_B == 1:
        return []

    # Get distance matrix
    dist_matrix = structure.distance_matrix
    # Get unique neighbor distances (shells)
    all_dists = dist_matrix[np.triu_indices_from(dist_matrix, k=1)]
    all_dists = np.sort(all_dists)

    # Group into shells (tolerance 0.1 A)
    shells = []
    current_shell = [all_dists[0]]
    for d in all_dists[1:]:
        if d - current_shell[-1] < 0.1:
            current_shell.append(d)
        else:
            shells.append(np.mean(current_shell))
            current_shell = [d]
            if len(shells) >= 6:  # first 6 shells
                break
    if current_shell and len(shells) < 6:
        shells.append(np.mean(current_shell))

    # Compute alpha for each shell
    alphas = []
    for r_shell in shells:
        n_AB = 0
        n_total = 0
        for i in range(len(structure)):
            for j in range(i + 1, len(structure)):
                d = dist_matrix[i, j]
                if abs(d - r_shell) < 0.2:
                    n_total += 1
                    sp_i = str(structure[i].specie)
                    sp_j = str(structure[j].specie)
                    if (sp_i == target_species) != (sp_j == target_species):
                        n_AB += 1

        if n_total > 0:
            p_AB = n_AB / n_total
            alpha = 1 - p_AB / (2 * x_B * (1 - x_B))
            alphas.append((r_shell, alpha))

    return alphas

def generate_sqs_random_search(structure, target_species, fraction,
                                n_trials=1000, seed=42):
    """
    Generate SQS by random search: try many random configurations
    and pick the one with pair correlations closest to zero.
    """
    rng = np.random.default_rng(seed)
    n_total = len(structure)
    n_sub = int(round(fraction * n_total))

    best_structure = None
    best_score = float("inf")

    for trial in range(n_trials):
        candidate = structure.copy()
        indices = rng.choice(n_total, size=n_sub, replace=False)
        for idx in sorted(indices, reverse=True):
            candidate.replace(idx, target_species)

        # Compute pair correlations
        alphas = compute_pair_correlation(candidate, target_species)
        if not alphas:
            continue

        # Score: sum of |alpha| for first 3 shells (closer to 0 = more random)
        score = sum(abs(a) for _, a in alphas[:3])

        if score < best_score:
            best_score = score
            best_structure = candidate
            best_alphas = alphas

        if trial % 200 == 0 and trial > 0:
            print(f"    Trial {trial}/{n_trials}: best score = {best_score:.4f}")

    return best_structure, best_alphas, best_score

print(f"  Searching for SQS (Cu0.75Zn0.25, {n_atoms} atoms)...")
sqs, alphas, score = generate_sqs_random_search(
    host, "Zn", 0.25, n_trials=500, seed=42
)

print(f"  Best SQS score: {score:.4f}")
print(f"  Pair correlations (Warren-Cowley alpha):")
for r, alpha in alphas[:4]:
    print(f"    Shell at {r:.2f} A: alpha = {alpha:+.4f} (0 = random)")

sqs.to(str(output_dir / "SQS_random_search.cif"))
Poscar(sqs).write_file(str(output_dir / "POSCAR_SQS"))
print(f"  Saved: {output_dir}/SQS_random_search.cif")
```

### Method C: VASP (Future External Access)

```python
#!/usr/bin/env python3
"""
Generate VASP input files for alloy calculations.
Includes POSCAR, INCAR, and KPOINTS.
"""

import numpy as np
from pymatgen.core import Structure, Lattice
from pymatgen.io.vasp import Poscar, Incar, Kpoints

# Build alloy structure (same as Method A)
host = Structure.from_spacegroup(
    "Fm-3m",
    lattice=Lattice.cubic(3.615),
    species=["Cu"],
    coords=[[0, 0, 0]],
)
host.make_supercell([3, 3, 3])

# Random substitution
rng = np.random.default_rng(42)
n_sub = int(0.25 * len(host))
indices = rng.choice(len(host), size=n_sub, replace=False)
alloy = host.copy()
for idx in sorted(indices, reverse=True):
    alloy.replace(idx, "Zn")

# POSCAR
Poscar(alloy).write_file("POSCAR_alloy")

# KPOINTS (reduced for supercell)
kpts = Kpoints.gamma_automatic(kpts=(4, 4, 4), shift=(0, 0, 0))
kpts.write_file("KPOINTS_alloy")

# INCAR for alloy vc-relax
incar = Incar({
    "SYSTEM": "Cu-Zn alloy",
    "ENCUT": 520,
    "PREC": "Accurate",
    "EDIFF": 1e-6,
    "EDIFFG": -0.01,
    "IBRION": 2,
    "NSW": 100,
    "ISIF": 3,            # Relax cell + ions for alloy
    "ISMEAR": 1,
    "SIGMA": 0.1,
    "LWAVE": False,
    "LCHARG": False,
    "LREAL": "Auto",
})
incar.write_file("INCAR_alloy")

print(f"VASP alloy files: POSCAR_alloy ({len(alloy)} atoms), "
      f"KPOINTS_alloy, INCAR_alloy")
print("Note: Use ISIF=3 for full cell+ion relaxation of alloys.")
```

---

## Key Parameters

| Parameter | Typical Value | Notes |
|---|---|---|
| Supercell size | 3x3x3 to 4x4x4 | Larger cells better represent random alloy statistics |
| Number of configs | 3--10 | Average over multiple random configs for reliable properties |
| SQS cutoff (icet) | 6--10 A | Pair correlation cutoff; include at least 3 neighbor shells |
| SQS MC steps | 10000--100000 | More steps = better SQS; diminishing returns above 50000 |
| n_trials (random search) | 500--5000 | More trials = better chance of finding low-correlation structure |
| mixing_beta (QE) | 0.2--0.3 | Lower values for alloys (harder SCF convergence) |
| ISMEAR (VASP) | 1 (Methfessel-Paxton) | Metallic alloys need finite temperature smearing |

## Common Issues

| Problem | Solution |
|---|---|
| Composition cannot be exactly achieved | Integer rounding limits: 27-atom cell cannot have exactly 25% of any species. Use larger supercells. |
| SQS search takes too long | Reduce n_trials or MC steps. Use icet for efficient search. |
| SCF convergence fails for alloy | Reduce mixing_beta to 0.2. Try different smearing. Increase ecutrho. |
| Relaxed alloy distorts significantly | Normal for size-mismatched alloys. Use vc-relax (ISIF=3) to allow cell shape change. |
| Properties vary between random configs | This is physical disorder sampling. Average over multiple configurations. |
| icet installation fails | Use `pip install icet`. Requires a C++ compiler. Alternatively, use the random search method. |
