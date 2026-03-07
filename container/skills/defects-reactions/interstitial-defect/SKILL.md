# Interstitial Defect Modeling

## When to Use

- Find and characterize interstitial sites (tetrahedral, octahedral, other voids) in a crystal
- Calculate interstitial formation energies for self-interstitials or foreign interstitials
- Screen interstitial hydrogen, lithium, or oxygen insertion sites
- Determine preferred interstitial configurations before and after relaxation
- Evaluate interstitial solubility under different chemical conditions

## Method Selection

| Criterion | ASE + MACE | QE DFT | VASP DFT |
|---|---|---|---|
| Speed | Seconds to minutes | Hours | Hours |
| Interstitial site finding | Voronoi via pymatgen | Same initial structures | Same initial structures |
| Neutral interstitials | Good for trends | Publication quality | Publication quality |
| Charged interstitials | Not supported | Supported (tot_charge) | Supported (NELECT) |
| Electronic structure | Not available | Available | Available |
| Use case | Site screening, ranking | Publication, charge states | Publication, charge states |

**Decision flow:**

```
Find candidate interstitial sites?
  --> Voronoi analysis with pymatgen (all methods start here)

Quick screening of many sites?
  --> ASE + MACE (Method A): relax at each site, rank by energy

Publication-quality formation energy?
  --> QE DFT (Method B) or VASP DFT (Method C)

Charged interstitial (e.g., H^+, Li^+)?
  --> QE or VASP DFT required

Electronic structure at interstitial site?
  --> QE or VASP DFT required
```

## Prerequisites

- pymatgen (structure manipulation, interstitial site finding via Voronoi)
- ASE + mace-torch (Method A)
- Quantum ESPRESSO pw.x (Method B)
- SSSP pseudopotentials for QE
- VASP with PAW potentials (Method C -- future external access)
- Optional: `pip install pymatgen-analysis-defects` for InterstitialGenerator

## Detailed Steps

### Preliminary: Find Interstitial Sites via Voronoi Analysis

```python
#!/usr/bin/env python3
"""
Find candidate interstitial sites using Voronoi tessellation.
Identifies tetrahedral and octahedral voids in the crystal structure.

The Voronoi decomposition of the crystal finds the largest empty
spheres that can fit between atoms. These are the natural candidate
sites for interstitial atoms.
"""

import numpy as np
from pymatgen.core import Structure, Element
from pymatgen.analysis.defects.generators import InterstitialGenerator
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
import json

# ============================================================
# 1. Load or build the host structure
# ============================================================
# Example: MgO rocksalt
host = Structure.from_spacegroup(
    "Fm-3m",
    lattice=[[4.212, 0, 0], [0, 4.212, 0], [0, 0, 4.212]],
    species=["Mg", "O"],
    coords=[[0.0, 0.0, 0.0], [0.5, 0.5, 0.5]],
)
print(f"Host: {host.formula}, SG: {SpacegroupAnalyzer(host).get_space_group_symbol()}")

# ============================================================
# 2. Find interstitial sites using pymatgen InterstitialGenerator
# ============================================================
# InterstitialGenerator uses Voronoi decomposition internally
# to find symmetry-inequivalent interstitial positions
inter_gen = InterstitialGenerator()

# Find sites for different interstitial species
interstitial_species = ["O", "Mg", "Li", "H"]

all_sites = {}
for sp in interstitial_species:
    interstitials = list(inter_gen.generate(host, {sp: Element(sp)}))
    print(f"\n{sp} interstitial: {len(interstitials)} symmetry-inequivalent sites")

    sites_info = []
    for i, inter in enumerate(interstitials):
        frac = inter.site.frac_coords
        cart = host.lattice.get_cartesian_coords(frac)

        # Compute distance to nearest host atom
        min_dist = min(
            host.lattice.get_all_distances(frac.reshape(1, 3),
                                           host.frac_coords).flatten()
        )

        print(f"  Site {i}: frac={frac}, nearest host atom = {min_dist:.3f} A")
        sites_info.append({
            "index": i,
            "frac_coords": frac.tolist(),
            "cart_coords": cart.tolist(),
            "min_dist_to_host_A": float(min_dist),
        })

    all_sites[sp] = sites_info

with open("interstitial_sites.json", "w") as f:
    json.dump(all_sites, f, indent=2, default=str)
print("\nSite data saved to interstitial_sites.json")
```

#### Manual Voronoi Analysis (Without pymatgen-analysis-defects)

```python
#!/usr/bin/env python3
"""
Find interstitial sites using scipy Voronoi tessellation directly.
Useful when pymatgen-analysis-defects is not installed.

Identifies tetrahedral and octahedral voids by finding Voronoi vertices
in the periodic crystal structure.
"""

import numpy as np
from pymatgen.core import Structure, Element
from scipy.spatial import Voronoi
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
import json

# ============================================================
# Build host structure
# ============================================================
host = Structure.from_spacegroup(
    "Fm-3m",
    lattice=[[4.212, 0, 0], [0, 4.212, 0], [0, 0, 4.212]],
    species=["Mg", "O"],
    coords=[[0.0, 0.0, 0.0], [0.5, 0.5, 0.5]],
)

# ============================================================
# Generate periodic images for Voronoi (need surrounding atoms)
# ============================================================
# Create a 3x3x3 supercell to get periodic images
sc = host.copy()
sc.make_supercell([3, 3, 3])
cart_coords = np.array([s.coords for s in sc])

# ============================================================
# Compute Voronoi tessellation
# ============================================================
vor = Voronoi(cart_coords)
vertices = vor.vertices

# Filter vertices to those inside the central unit cell
lattice = host.lattice
central_offset = np.array([1, 1, 1])  # offset to central cell in 3x3x3
central_origin = lattice.get_cartesian_coords(central_offset)
central_end = lattice.get_cartesian_coords(central_offset + np.array([1, 1, 1]))

inside_mask = np.all(
    (vertices >= central_origin) & (vertices < central_end), axis=1
)
central_vertices = vertices[inside_mask]
print(f"Voronoi vertices inside central cell: {len(central_vertices)}")

# Convert to fractional coordinates of the primitive cell
frac_vertices = []
for v in central_vertices:
    frac = lattice.get_fractional_coords(v - central_origin)
    # Wrap to [0, 1)
    frac = frac % 1.0
    frac_vertices.append(frac)

frac_vertices = np.array(frac_vertices)

# ============================================================
# Remove duplicates (symmetry-equivalent vertices)
# ============================================================
unique_fracs = []
tol = 0.05  # fractional tolerance for merging

for frac in frac_vertices:
    is_dup = False
    for uf in unique_fracs:
        diff = np.abs(frac - uf)
        diff = np.minimum(diff, 1.0 - diff)  # periodic distance
        if np.all(diff < tol):
            is_dup = True
            break
    if not is_dup:
        unique_fracs.append(frac)

unique_fracs = np.array(unique_fracs)
print(f"Unique interstitial sites: {len(unique_fracs)}")

# Classify sites by distance to nearest host atom
for i, frac in enumerate(unique_fracs):
    cart = lattice.get_cartesian_coords(frac)
    dists = lattice.get_all_distances(frac.reshape(1, 3), host.frac_coords).flatten()
    min_dist = np.min(dists)
    nn_species = str(host[np.argmin(dists)].specie)

    # Classify: tetrahedral sites typically have 4 nearest neighbors at similar distance
    # octahedral sites have 6 nearest neighbors
    close_dists = dists[dists < min_dist * 1.3]
    n_near = len(close_dists)
    site_type = "tetrahedral" if n_near <= 4 else "octahedral"

    print(f"  Site {i}: frac={frac}, min_dist={min_dist:.3f} A, "
          f"nearest={nn_species}, ~{n_near} near neighbors ({site_type})")
```

### Method A: ASE + MACE (Neutral Interstitials)

#### Complete Workflow: Screen Interstitial Sites

```python
#!/usr/bin/env python3
"""
Screen interstitial defect formation energies at all candidate sites
using ASE + MACE.

Formation energy:
  E_f(X_i) = E_defect - E_bulk - mu_X

where X_i is species X at interstitial site i,
      mu_X is the chemical potential of the inserted species.
"""

import numpy as np
from pymatgen.core import Structure, Element
from pymatgen.analysis.defects.generators import InterstitialGenerator
from pymatgen.io.ase import AseAtomsAdaptor
from ase.optimize import BFGS
from ase.filters import FrechetCellFilter
from ase.build import molecule, bulk as ase_bulk
from mace.calculators import mace_mp
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import json

# ============================================================
# 1. Setup
# ============================================================
calc = mace_mp(model="medium", dispersion=False, default_dtype="float64")
adaptor = AseAtomsAdaptor()

# Host: MgO
host = Structure.from_spacegroup(
    "Fm-3m",
    lattice=[[4.212, 0, 0], [0, 4.212, 0], [0, 0, 4.212]],
    species=["Mg", "O"],
    coords=[[0.0, 0.0, 0.0], [0.5, 0.5, 0.5]],
)

# Relax host
atoms_host = adaptor.get_atoms(host)
atoms_host.calc = calc
ecf = FrechetCellFilter(atoms_host)
BFGS(ecf, logfile=None).run(fmax=0.005, steps=500)
host_relaxed = adaptor.get_structure(atoms_host)

# ============================================================
# 2. Compute reference energies
# ============================================================
# Li metal (bcc) reference for Li interstitial
li_atoms = ase_bulk("Li", "bcc", a=3.490)
li_atoms.calc = mace_mp(model="medium", dispersion=False, default_dtype="float64")
ecf_li = FrechetCellFilter(li_atoms)
BFGS(ecf_li, logfile=None).run(fmax=0.005, steps=300)
mu_Li = li_atoms.get_potential_energy() / len(li_atoms)
print(f"mu_Li (bcc ref) = {mu_Li:.6f} eV/atom")

# H2 molecule reference for H interstitial
h2 = molecule("H2")
h2.center(vacuum=10.0)
h2.calc = mace_mp(model="medium", dispersion=False, default_dtype="float64")
BFGS(h2, logfile=None).run(fmax=0.005, steps=200)
mu_H = h2.get_potential_energy() / 2
print(f"mu_H (H2/2 ref) = {mu_H:.6f} eV/atom")

# ============================================================
# 3. Build and relax bulk supercell
# ============================================================
SC_SIZE = 3
bulk_sc = host_relaxed.copy()
bulk_sc.make_supercell([SC_SIZE, SC_SIZE, SC_SIZE])

atoms_bulk = adaptor.get_atoms(bulk_sc)
atoms_bulk.calc = calc
BFGS(atoms_bulk, logfile="relax_bulk.log").run(fmax=0.005, steps=300)
e_bulk = atoms_bulk.get_potential_energy()
n_bulk = len(atoms_bulk)
print(f"\nBulk supercell: {n_bulk} atoms, E = {e_bulk:.6f} eV")

# ============================================================
# 4. Generate and screen interstitial sites
# ============================================================
INTERSTITIAL_SPECIES = "Li"
mu_interstitial = mu_Li  # change for different species

inter_gen = InterstitialGenerator()
interstitials = list(inter_gen.generate(host_relaxed, {INTERSTITIAL_SPECIES: Element(INTERSTITIAL_SPECIES)}))
print(f"\nFound {len(interstitials)} symmetry-inequivalent {INTERSTITIAL_SPECIES} interstitial sites")

sc_mat = np.eye(3, dtype=int) * SC_SIZE
results = []

for i, inter_defect in enumerate(interstitials):
    print(f"\n--- Interstitial site {i} ---")
    frac = inter_defect.site.frac_coords
    print(f"  Fractional coords: {frac}")

    # Generate supercell with interstitial
    defect_sc = inter_defect.get_supercell_structure(sc_mat=sc_mat)
    n_defect = len(defect_sc)
    print(f"  Supercell: {n_defect} atoms ({n_defect - n_bulk} interstitial added)")

    # Relax defect supercell
    atoms_defect = adaptor.get_atoms(defect_sc)
    atoms_defect.calc = mace_mp(model="medium", dispersion=False, default_dtype="float64")
    opt = BFGS(atoms_defect, logfile=f"relax_inter_{INTERSTITIAL_SPECIES}_{i}.log")
    opt.run(fmax=0.005, steps=500)
    e_defect = atoms_defect.get_potential_energy()
    print(f"  E_defect = {e_defect:.6f} eV (converged in {opt.nsteps} steps)")

    # Formation energy: E_f = E_defect - E_bulk - n_added * mu_X
    n_added = n_defect - n_bulk  # should be 1
    e_form = e_defect - e_bulk - n_added * mu_interstitial
    print(f"  E_form({INTERSTITIAL_SPECIES}_i site {i}) = {e_form:.4f} eV")

    # Check how far the interstitial moved during relaxation
    relaxed_defect = adaptor.get_structure(atoms_defect)
    # The interstitial is the last atom added
    inter_final_frac = relaxed_defect[-1].frac_coords
    displacement = host_relaxed.lattice.get_cartesian_coords(
        inter_final_frac - frac / SC_SIZE  # approximate
    )

    results.append({
        "site_index": i,
        "initial_frac_coords": frac.tolist(),
        "e_defect_eV": e_defect,
        "e_form_eV": e_form,
        "n_relax_steps": opt.nsteps,
    })

# ============================================================
# 5. Summary and ranking
# ============================================================
print("\n" + "=" * 60)
print(f"INTERSTITIAL FORMATION ENERGIES: {INTERSTITIAL_SPECIES} in {host.composition.reduced_formula}")
print("=" * 60)
print(f"{'Site':<8} {'E_form (eV)':<15} {'Steps':<8}")
print("-" * 35)
for r in sorted(results, key=lambda x: x["e_form_eV"]):
    marker = " <-- most stable" if r["e_form_eV"] == min(x["e_form_eV"] for x in results) else ""
    print(f"  {r['site_index']:<6} {r['e_form_eV']:<15.4f} {r['n_relax_steps']:<8}{marker}")

# Bar chart
if results:
    fig, ax = plt.subplots(figsize=(7, 5))
    sorted_results = sorted(results, key=lambda x: x["e_form_eV"])
    labels = [f"Site {r['site_index']}" for r in sorted_results]
    energies = [r["e_form_eV"] for r in sorted_results]
    colors = ["forestgreen" if e == min(energies) else "steelblue" for e in energies]

    ax.barh(labels, energies, color=colors, edgecolor="black", alpha=0.8)
    ax.set_xlabel("Formation energy (eV)", fontsize=12)
    ax.set_title(f"{INTERSTITIAL_SPECIES} interstitial sites in {host.composition.reduced_formula}",
                 fontsize=13)
    ax.axvline(x=0, color="black", linewidth=0.8)
    ax.grid(axis="x", alpha=0.3)
    fig.tight_layout()
    fig.savefig("interstitial_site_energies.png", dpi=150)
    print("\nPlot saved to interstitial_site_energies.png")

with open("interstitial_results.json", "w") as f:
    json.dump(results, f, indent=2, default=str)
print("Results saved to interstitial_results.json")
```

#### Self-Interstitial Analysis

```python
#!/usr/bin/env python3
"""
Self-interstitial formation energy.
A self-interstitial is when an extra atom of the host species
occupies an interstitial position.

Example: Mg self-interstitial in MgO.

Formation energy (using bulk per-atom reference):
  E_f(X_i) = E_defect - E_bulk - mu_X
  where mu_X = E_bulk / N_bulk (bulk per-atom energy)

This is equivalent to:
  E_f = E_defect - (N+1)/N * E_bulk
"""

import numpy as np
from pymatgen.core import Structure, Element
from pymatgen.analysis.defects.generators import InterstitialGenerator
from pymatgen.io.ase import AseAtomsAdaptor
from ase.optimize import BFGS
from mace.calculators import mace_mp
import json

calc = mace_mp(model="medium", dispersion=False, default_dtype="float64")
adaptor = AseAtomsAdaptor()

host = Structure.from_spacegroup(
    "Fm-3m",
    lattice=[[4.212, 0, 0], [0, 4.212, 0], [0, 0, 4.212]],
    species=["Mg", "O"],
    coords=[[0.0, 0.0, 0.0], [0.5, 0.5, 0.5]],
)

SC_SIZE = 3
SELF_INTERSTITIAL_SPECIES = "Mg"

# Relax bulk supercell
bulk_sc = host.copy()
bulk_sc.make_supercell([SC_SIZE, SC_SIZE, SC_SIZE])
atoms_bulk = adaptor.get_atoms(bulk_sc)
atoms_bulk.calc = calc
BFGS(atoms_bulk, logfile=None).run(fmax=0.005, steps=300)
e_bulk = atoms_bulk.get_potential_energy()
n_bulk = len(atoms_bulk)
mu_self = e_bulk / n_bulk
print(f"Bulk: {n_bulk} atoms, E/atom = {mu_self:.6f} eV")

# Generate self-interstitial sites
inter_gen = InterstitialGenerator()
interstitials = list(inter_gen.generate(host, {SELF_INTERSTITIAL_SPECIES: Element(SELF_INTERSTITIAL_SPECIES)}))

sc_mat = np.eye(3, dtype=int) * SC_SIZE
results = []

for i, inter in enumerate(interstitials):
    print(f"\n--- {SELF_INTERSTITIAL_SPECIES} self-interstitial site {i} ---")
    defect_sc = inter.get_supercell_structure(sc_mat=sc_mat)
    atoms_defect = adaptor.get_atoms(defect_sc)
    atoms_defect.calc = mace_mp(model="medium", dispersion=False, default_dtype="float64")
    BFGS(atoms_defect, logfile=None).run(fmax=0.005, steps=500)
    e_defect = atoms_defect.get_potential_energy()

    # Self-interstitial formation energy
    e_form = e_defect - e_bulk - mu_self
    # Equivalently: e_form = e_defect - (n_bulk + 1) / n_bulk * e_bulk
    print(f"  E_form = {e_form:.4f} eV")

    results.append({
        "site": i,
        "species": SELF_INTERSTITIAL_SPECIES,
        "e_form_eV": e_form,
    })

# Summary
print(f"\nSelf-interstitial energies for {SELF_INTERSTITIAL_SPECIES} in MgO:")
for r in sorted(results, key=lambda x: x["e_form_eV"]):
    print(f"  Site {r['site']}: {r['e_form_eV']:.4f} eV")
print("(Self-interstitials typically have higher formation energy than vacancies)")
```

### Method B: QE DFT (Neutral and Charged Interstitials)

#### Step 1: Generate QE Input Files

```python
#!/usr/bin/env python3
"""
Generate QE input files for interstitial defect calculations.
Uses the most stable site from MACE screening.

Example: Li interstitial in MgO (charged Li_i^{+1}).
"""

import numpy as np
from pymatgen.core import Structure, Element
from pymatgen.analysis.defects.generators import InterstitialGenerator
from pymatgen.io.pwscf import PWInput
from pathlib import Path

# ============================================================
# Configuration
# ============================================================
PSEUDO_DIR = "./pseudo"
ECUTWFC = 60.0
ECUTRHO = 480.0
SC_SIZE = 3

INTERSTITIAL_SPECIES = "Li"
CHARGE_STATES = [0, +1]  # Li_i^0 and Li_i^{+1}

pseudos = {
    "Mg": "Mg.pbe-spnl-kjpaw_psl.1.0.0.UPF",
    "O":  "O.pbe-n-kjpaw_psl.1.0.0.UPF",
    "Li": "Li.pbe-s-kjpaw_psl.1.0.0.UPF",
}

# ============================================================
# Build structures
# ============================================================
host = Structure.from_spacegroup(
    "Fm-3m",
    lattice=[[4.212, 0, 0], [0, 4.212, 0], [0, 0, 4.212]],
    species=["Mg", "O"],
    coords=[[0.0, 0.0, 0.0], [0.5, 0.5, 0.5]],
)

# Bulk supercell
bulk_sc = host.copy()
bulk_sc.make_supercell([SC_SIZE, SC_SIZE, SC_SIZE])

# Generate interstitial at the most stable site (from MACE screening)
inter_gen = InterstitialGenerator()
interstitials = list(inter_gen.generate(host, {INTERSTITIAL_SPECIES: Element(INTERSTITIAL_SPECIES)}))

# Use the first site (replace with the best site index from screening)
BEST_SITE_INDEX = 0
best_inter = interstitials[BEST_SITE_INDEX]
sc_mat = np.eye(3, dtype=int) * SC_SIZE
defect_sc = best_inter.get_supercell_structure(sc_mat=sc_mat)

print(f"Bulk: {len(bulk_sc)} atoms")
print(f"Defect: {len(defect_sc)} atoms (interstitial at site {BEST_SITE_INDEX})")

Path(PSEUDO_DIR).mkdir(exist_ok=True)

# Common parameters
control = {
    "calculation": "relax",
    "restart_mode": "from_scratch",
    "pseudo_dir": PSEUDO_DIR,
    "tprnfor": True,
    "tstress": True,
    "etot_conv_thr": 1.0e-6,
    "forc_conv_thr": 1.0e-4,
}

system_params = {
    "ecutwfc": ECUTWFC,
    "ecutrho": ECUTRHO,
    "occupations": "smearing",
    "smearing": "cold",
    "degauss": 0.01,
}

electrons = {
    "conv_thr": 1.0e-8,
    "mixing_beta": 0.3,
}

kpts = (2, 2, 2)

# ============================================================
# Write inputs
# ============================================================
# Bulk
pw_bulk = PWInput(
    bulk_sc, pseudo=pseudos,
    control=control | {"prefix": "bulk", "outdir": "./tmp_bulk"},
    system=system_params, electrons=electrons, kpoints_grid=kpts,
)
pw_bulk.write_file("pw_bulk.in")
print(f"Written pw_bulk.in")

# Interstitial (each charge state)
for q in CHARGE_STATES:
    qstr = f"q{q:+d}" if q != 0 else "q0"
    system_q = system_params.copy()
    if q != 0:
        system_q["tot_charge"] = float(q)

    pw_inter = PWInput(
        defect_sc, pseudo=pseudos,
        control=control | {"prefix": f"inter_{qstr}", "outdir": f"./tmp_inter_{qstr}"},
        system=system_q,
        electrons=electrons | ({"mixing_beta": 0.2} if q != 0 else {}),
        kpoints_grid=kpts,
    )
    pw_inter.write_file(f"pw_inter_{qstr}.in")
    print(f"Written pw_inter_{qstr}.in (q={q:+d})")

# Li bulk reference (bcc)
from pymatgen.core import Lattice
li_bcc = Structure.from_spacegroup(
    "Im-3m",
    lattice=Lattice.cubic(3.490),
    species=["Li"],
    coords=[[0.0, 0.0, 0.0]],
)
pw_li = PWInput(
    li_bcc, pseudo=pseudos,
    control=control | {"prefix": "Li_bcc", "outdir": "./tmp_Li"},
    system=system_params,
    electrons=electrons,
    kpoints_grid=(8, 8, 8),
)
pw_li.write_file("pw_Li_ref.in")
print(f"Written pw_Li_ref.in (Li bcc reference)")
```

#### Step 2: Run QE Calculations

```bash
#!/bin/bash
# Run interstitial defect QE calculations
NPROC=4

# Download pseudopotentials
mkdir -p pseudo tmp_bulk tmp_inter_q0 tmp_inter_q+1 tmp_Li
cd pseudo
wget -nc "https://pseudopotentials.quantum-espresso.org/upf_files/Mg.pbe-spnl-kjpaw_psl.1.0.0.UPF" 2>/dev/null
wget -nc "https://pseudopotentials.quantum-espresso.org/upf_files/O.pbe-n-kjpaw_psl.1.0.0.UPF" 2>/dev/null
wget -nc "https://pseudopotentials.quantum-espresso.org/upf_files/Li.pbe-s-kjpaw_psl.1.0.0.UPF" 2>/dev/null
cd ..

echo "=== Bulk supercell ==="
mpirun -np $NPROC pw.x -in pw_bulk.in > pw_bulk.out 2>&1
echo "Bulk: $(grep '!' pw_bulk.out | tail -1)"

echo "=== Li reference ==="
mpirun -np $NPROC pw.x -in pw_Li_ref.in > pw_Li_ref.out 2>&1
echo "Li: $(grep '!' pw_Li_ref.out | tail -1)"

echo "=== Neutral interstitial ==="
mpirun -np $NPROC pw.x -in pw_inter_q0.in > pw_inter_q0.out 2>&1
echo "Li_i^0: $(grep '!' pw_inter_q0.out | tail -1)"

echo "=== Charged interstitial q=+1 ==="
mpirun -np $NPROC pw.x -in pw_inter_q+1.in > pw_inter_q+1.out 2>&1
echo "Li_i^{+1}: $(grep '!' pw_inter_q+1.out | tail -1)"
```

#### Step 3: Compute Interstitial Formation Energy

```python
#!/usr/bin/env python3
"""
Parse QE outputs and compute interstitial formation energies.

Formation energy:
  E_f(X_i^q) = E_defect(q) - E_bulk - mu_X + q*(E_VBM + E_Fermi) + E_corr
"""

import re
import numpy as np
import json

def parse_qe_energy(filename):
    energy = None
    with open(filename) as f:
        for line in f:
            if "!" in line and "total energy" in line:
                match = re.search(r"=\s+([-\d.]+)\s+Ry", line)
                if match:
                    energy = float(match.group(1)) * 13.605693123
    return energy

def parse_qe_natoms(filename):
    with open(filename) as f:
        for line in f:
            if "number of atoms/cell" in line:
                return int(line.split("=")[1].strip())
    return None

def parse_qe_vbm(filename):
    vbm = None
    with open(filename) as f:
        for line in f:
            if "highest occupied, lowest unoccupied" in line:
                vals = re.findall(r"[-\d.]+", line)
                if vals:
                    vbm = float(vals[0])
            elif "highest occupied" in line:
                match = re.search(r"([-\d.]+)\s+eV", line)
                if match:
                    vbm = float(match.group(1))
    return vbm

# Parse
e_bulk = parse_qe_energy("pw_bulk.out")
n_bulk = parse_qe_natoms("pw_bulk.out")
e_vbm = parse_qe_vbm("pw_bulk.out")

e_li_ref = parse_qe_energy("pw_Li_ref.out")
n_li_ref = parse_qe_natoms("pw_Li_ref.out")
mu_Li = e_li_ref / n_li_ref

print(f"E_bulk = {e_bulk:.6f} eV ({n_bulk} atoms)")
print(f"mu_Li  = {mu_Li:.6f} eV/atom")
if e_vbm:
    print(f"E_VBM  = {e_vbm:.4f} eV")

# Finite-size correction parameters
dielectric = 9.8
L_sc = 4.212 * 3

charge_states = [0, 1]
results = {}

for q in charge_states:
    qstr = f"q{q:+d}" if q != 0 else "q0"
    e_defect = parse_qe_energy(f"pw_inter_{qstr}.out")

    if e_defect is None:
        print(f"\nWARNING: Could not parse pw_inter_{qstr}.out")
        continue

    # E_f = E_defect - E_bulk - mu_Li + q*(E_VBM + E_F) + E_corr
    e_form = e_defect - e_bulk - mu_Li

    e_corr = 0.0
    if q != 0:
        alpha_M = 2.8373
        e2 = 14.3996
        e_corr = (alpha_M * q**2 * e2) / (2.0 * dielectric * L_sc)
        e_form += e_corr
        if e_vbm:
            e_form += q * e_vbm

    print(f"\n--- Li_i^{{{q:+d}}} ---")
    print(f"  E_defect = {e_defect:.6f} eV")
    print(f"  E_form (at VBM) = {e_form:.4f} eV")
    if q != 0:
        print(f"  Freysoldt correction = {e_corr:.4f} eV")

    results[q] = {
        "charge": q,
        "e_defect_eV": e_defect,
        "e_form_at_VBM_eV": e_form,
        "e_corr_eV": e_corr,
    }

with open("interstitial_qe_results.json", "w") as f:
    json.dump(results, f, indent=2, default=float)
print("\nResults saved to interstitial_qe_results.json")
```

### Method C: VASP DFT (Neutral and Charged Interstitials)

#### Generate VASP Input Files

```python
#!/usr/bin/env python3
"""
Generate VASP input files for interstitial defect calculations.

Note: VASP execution will be available via future external access.
"""

import numpy as np
from pymatgen.core import Structure, Element, Lattice
from pymatgen.analysis.defects.generators import InterstitialGenerator
from pymatgen.io.vasp import Incar, Poscar, Kpoints
from pathlib import Path

# Build host
host = Structure.from_spacegroup(
    "Fm-3m",
    lattice=[[4.212, 0, 0], [0, 4.212, 0], [0, 0, 4.212]],
    species=["Mg", "O"],
    coords=[[0.0, 0.0, 0.0], [0.5, 0.5, 0.5]],
)

SC_SIZE = 3
INTERSTITIAL_SPECIES = "Li"
CHARGE_STATES = [0, +1]

# Bulk supercell
bulk_sc = host.copy()
bulk_sc.make_supercell([SC_SIZE, SC_SIZE, SC_SIZE])

# Interstitial supercell
inter_gen = InterstitialGenerator()
interstitials = list(inter_gen.generate(host, {INTERSTITIAL_SPECIES: Element(INTERSTITIAL_SPECIES)}))
sc_mat = np.eye(3, dtype=int) * SC_SIZE
defect_sc = interstitials[0].get_supercell_structure(sc_mat=sc_mat)

# ============================================================
# 1. Bulk
# ============================================================
bulk_dir = Path("vasp_bulk")
bulk_dir.mkdir(exist_ok=True)
Poscar(bulk_sc).write_file(str(bulk_dir / "POSCAR"))

incar_bulk = Incar({
    "SYSTEM": f"{host.composition.reduced_formula} bulk",
    "ENCUT": 520, "PREC": "Accurate",
    "EDIFF": 1e-6, "EDIFFG": -0.01,
    "IBRION": 2, "ISIF": 2, "NSW": 200,
    "ISMEAR": 0, "SIGMA": 0.05,
    "LREAL": "Auto", "LWAVE": False, "LCHARG": True,
    "LORBIT": 11, "ALGO": "Normal",
})
incar_bulk.write_file(str(bulk_dir / "INCAR"))
Kpoints.gamma_automatic(kpts=(2, 2, 2)).write_file(str(bulk_dir / "KPOINTS"))
print(f"Bulk: {bulk_dir}/ ({len(bulk_sc)} atoms)")

# ============================================================
# 2. Interstitial (each charge state)
# ============================================================
for q in CHARGE_STATES:
    qstr = f"q{q:+d}" if q != 0 else "q0"
    int_dir = Path(f"vasp_inter_{qstr}")
    int_dir.mkdir(exist_ok=True)

    Poscar(defect_sc).write_file(str(int_dir / "POSCAR"))

    incar_int = Incar({
        "SYSTEM": f"{INTERSTITIAL_SPECIES}_i^{{{q:+d}}} in {host.composition.reduced_formula}",
        "ENCUT": 520, "PREC": "Accurate",
        "EDIFF": 1e-6, "EDIFFG": -0.01,
        "IBRION": 2, "ISIF": 2, "NSW": 300,
        "ISMEAR": 0, "SIGMA": 0.05,
        "LREAL": "Auto", "LWAVE": False, "LCHARG": True,
        "LVHAR": True,
        "LORBIT": 11, "ALGO": "Normal", "NELM": 300,
    })

    if q != 0:
        incar_int["NELECT"] = "REPLACE_WITH_ZVAL_SUM_MINUS_Q"
        if q % 2 != 0:
            incar_int["ISPIN"] = 2

    incar_int.write_file(str(int_dir / "INCAR"))
    Kpoints.gamma_automatic(kpts=(2, 2, 2)).write_file(str(int_dir / "KPOINTS"))
    print(f"{INTERSTITIAL_SPECIES}_i^{{{q:+d}}}: {int_dir}/ ({len(defect_sc)} atoms)")

# Li reference
li_dir = Path("vasp_Li_ref")
li_dir.mkdir(exist_ok=True)
li_bcc = Structure.from_spacegroup(
    "Im-3m", lattice=Lattice.cubic(3.490),
    species=["Li"], coords=[[0.0, 0.0, 0.0]],
)
Poscar(li_bcc).write_file(str(li_dir / "POSCAR"))
Incar({
    "SYSTEM": "Li bcc ref", "ENCUT": 520, "PREC": "Accurate",
    "EDIFF": 1e-6, "EDIFFG": -0.01, "IBRION": 2, "ISIF": 3, "NSW": 100,
    "ISMEAR": 1, "SIGMA": 0.1, "LWAVE": False, "LCHARG": False,
}).write_file(str(li_dir / "INCAR"))
Kpoints.gamma_automatic(kpts=(12, 12, 12)).write_file(str(li_dir / "KPOINTS"))
print(f"Li reference: {li_dir}/")

print("""
=== VASP Interstitial Defect Workflow ===
1. Run bulk supercell
2. Run Li bcc reference
3. Run neutral interstitial; determine default NELECT
4. Adjust NELECT for charged states: NELECT = default - q
5. Run charged interstitials
6. Apply Freysoldt correction using LOCPOT files
""")
```

## Key Parameters

| Parameter | Recommended Value | Notes |
|---|---|---|
| Supercell size | >= 3x3x3 or min 10 A between interstitial images | Must accommodate local distortion around interstitial |
| MACE fmax | 0.005 eV/A | Tight for reliable energy ranking |
| QE ecutwfc | 50-80 Ry | Must accommodate the interstitial element pseudopotential |
| QE k-points | Gamma or 2x2x2 for large supercells | Converge independently |
| VASP ENCUT | 520 eV or 1.3x max ENMAX | Ensure convergence for both host and interstitial elements |
| Chemical potential ref | Elemental bulk or gas molecule | H: H2/2, O: O2/2, Li: Li bcc, etc. |
| Relaxation steps | 300-500 for defect | Interstitials may need many steps to find stable configuration |
| Spin polarization | Enable for magnetic interstitials | nspin=2 (QE) or ISPIN=2 (VASP) |
| Voronoi site tolerance | 0.05 fractional | For merging near-duplicate sites |

## Interpreting Results

1. **Site ranking**: The site with the lowest formation energy is the thermodynamically preferred interstitial position. Kinetics (migration barriers) also matter for which sites are actually occupied.

2. **Tetrahedral vs. octahedral**: In close-packed structures, octahedral sites are larger and often preferred for larger interstitials. Tetrahedral sites are preferred for small atoms like H.

3. **Relaxation displacement**: If the interstitial moves significantly from its initial Voronoi site during relaxation, the initial site is unstable. The final position is the true stable site. Multiple initial sites may relax to the same final configuration.

4. **Self-interstitial energies**: Self-interstitials typically have higher formation energies than vacancies because inserting an atom into a fully occupied lattice requires more distortion. The Frenkel pair energy (vacancy + self-interstitial) measures the intrinsic defect formation cost.

5. **Charged interstitials**: Small electropositive interstitials (Li, Na, H) often prefer positive charge states. The transition level determines where in the band gap the neutral-to-charged crossover occurs.

6. **MACE accuracy**: MACE is generally reliable for interstitial site screening (ranking sites by energy). Absolute formation energies may differ from DFT by 0.1-0.5 eV, but relative energies between sites are usually robust.

## Common Issues

| Issue | Solution |
|---|---|
| Interstitial relaxes to a host atom site (kick-out) | The interstitial displaces a host atom; this is a valid configuration (interstitialcy). Check the final structure carefully. |
| Multiple initial sites converge to same final position | Expected; report the unique final configurations and their energies |
| Interstitial causes very large local distortion | Use a larger supercell to accommodate the strain field |
| SCF not converging with interstitial | Reduce mixing_beta; use different mixing mode; start from larger NELM |
| Formation energy is very negative | Check chemical potential reference; interstitial may be highly favorable (e.g., H in many metals) |
| Voronoi analysis finds too many sites | Increase the symmetry reduction tolerance; manually inspect and merge equivalent sites |
| pymatgen-analysis-defects not installed | Use the manual Voronoi analysis script above, or `pip install pymatgen-analysis-defects` |
| VASP: wrong NELECT for charged interstitial | Run neutral first, grep NELECT, then NELECT = default - q |
| Interstitial element not in MACE training data | Use QE or VASP DFT directly; MACE coverage is limited for some elements |
