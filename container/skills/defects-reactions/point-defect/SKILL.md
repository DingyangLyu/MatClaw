# Point Defect Calculations

## When to Use

- Calculate vacancy or interstitial formation energies in crystalline materials
- Screen defect stability across different sites or charge states
- Determine which defects are thermodynamically favorable under given chemical conditions
- Compute defect concentrations at finite temperature
- Identify stable charge states and charge transition levels

## Method Selection

```
Neutral defect, quick screening?
  --> ASE + MACE (Method A): seconds to minutes, good for trends

Neutral defect, publication quality?
  --> QE DFT (Method B): hours, accurate energetics

Charged defect?
  --> QE DFT required (Method B): needs electrostatic potential for
      finite-size corrections (Freysoldt/Kumagai)

Need electronic structure at defect site?
  --> QE DFT required (Method B): MACE has no electronic degrees of freedom
```

## Prerequisites

- pymatgen (structure manipulation, defect generation)
- ASE + mace-torch (Method A)
- Quantum ESPRESSO pw.x (Method B)
- SSSP pseudopotentials for QE
- Optional: `pip install pymatgen-analysis-defects` for advanced defect workflows

## Detailed Steps

### Method A: ASE + MACE (Neutral Defects)

#### Complete Workflow: Vacancy Formation Energy

```python
#!/usr/bin/env python3
"""
Point defect formation energy calculation using ASE + MACE.
Example: oxygen vacancy in MgO.
"""

import numpy as np
from pymatgen.core import Structure, Element
from pymatgen.analysis.defects.generators import VacancyGenerator
from pymatgen.transformations.standard_transformations import SupercellTransformation
from pymatgen.io.ase import AseAtomsAdaptor
from ase.optimize import BFGS
from ase.filters import FrechetCellFilter
from mace.calculators import mace_mp
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
import json

# ============================================================
# 1. Build or load the primitive cell
# ============================================================
# Example: MgO rocksalt (Fm-3m)
primitive = Structure.from_spacegroup(
    "Fm-3m",
    lattice=[[4.212, 0, 0], [0, 4.212, 0], [0, 0, 4.212]],
    species=["Mg", "O"],
    coords=[[0.0, 0.0, 0.0], [0.5, 0.5, 0.5]],
)
print(f"Primitive cell: {primitive.formula}, {len(primitive)} atoms")

# ============================================================
# 2. Set up MACE calculator
# ============================================================
calc = mace_mp(model="medium", dispersion=False, default_dtype="float64")

# ============================================================
# 3. Relax the primitive cell (get E_bulk per atom reference)
# ============================================================
adaptor = AseAtomsAdaptor()
atoms_prim = adaptor.get_atoms(primitive)
atoms_prim.calc = calc

# Full cell + positions relaxation for the primitive
ecf = FrechetCellFilter(atoms_prim)
opt = BFGS(ecf, logfile="relax_primitive.log")
opt.run(fmax=0.005, steps=500)

relaxed_primitive = adaptor.get_structure(atoms_prim)
e_prim_per_atom = atoms_prim.get_potential_energy() / len(atoms_prim)
print(f"Relaxed primitive energy/atom: {e_prim_per_atom:.6f} eV")

# ============================================================
# 4. Generate supercells and test convergence
# ============================================================
supercell_sizes = [2, 3, 4]  # NxNxN supercells
vacancy_energies = {}

for n in supercell_sizes:
    print(f"\n--- Supercell {n}x{n}x{n} ---")

    # Build bulk supercell
    bulk_sc = relaxed_primitive.copy()
    bulk_sc.make_supercell([n, n, n])
    n_atoms_bulk = len(bulk_sc)
    print(f"  Bulk supercell: {n_atoms_bulk} atoms")

    # Relax bulk supercell (positions only, keep cell fixed)
    atoms_bulk = adaptor.get_atoms(bulk_sc)
    atoms_bulk.calc = calc
    opt_bulk = BFGS(atoms_bulk, logfile=f"relax_bulk_{n}x{n}x{n}.log")
    opt_bulk.run(fmax=0.005, steps=300)
    e_bulk = atoms_bulk.get_potential_energy()
    print(f"  E_bulk = {e_bulk:.6f} eV ({n_atoms_bulk} atoms)")

    # Create vacancy: remove one O atom (index 0 of the O sublattice)
    defect_sc = adaptor.get_structure(atoms_bulk).copy()
    # Find first O site
    o_indices = [i for i, site in enumerate(defect_sc) if site.specie == Element("O")]
    removed_idx = o_indices[0]
    removed_species = defect_sc[removed_idx].specie
    print(f"  Removing {removed_species} at site {removed_idx}")
    defect_sc.remove_sites([removed_idx])
    n_atoms_defect = len(defect_sc)

    # Relax defect supercell (positions only)
    atoms_defect = adaptor.get_atoms(defect_sc)
    atoms_defect.calc = calc
    opt_def = BFGS(atoms_defect, logfile=f"relax_vacancy_{n}x{n}x{n}.log")
    opt_def.run(fmax=0.005, steps=500)
    e_defect = atoms_defect.get_potential_energy()
    print(f"  E_defect = {e_defect:.6f} eV ({n_atoms_defect} atoms)")

    # Formation energy: E_f = E_defect - E_bulk + mu_O
    # For now, use bulk reference: mu_O = E_bulk / N_bulk * 1 (one O removed)
    # More precisely: mu_O is from a reference phase (see chemical potentials below)
    # Simple approximation: mu_O ~ e_prim_per_atom (per-atom energy of bulk)
    # Better: use O2 molecule energy / 2 for O-rich limit
    e_form = e_defect - e_bulk + e_bulk / n_atoms_bulk
    print(f"  E_formation(V_O) = {e_form:.4f} eV (bulk atom reference)")

    vacancy_energies[n] = {
        "n_atoms": n_atoms_bulk,
        "e_bulk": e_bulk,
        "e_defect": e_defect,
        "e_form": e_form,
    }

# ============================================================
# 5. Convergence plot
# ============================================================
sizes = [vacancy_energies[n]["n_atoms"] for n in supercell_sizes]
e_forms = [vacancy_energies[n]["e_form"] for n in supercell_sizes]

fig, ax = plt.subplots(figsize=(6, 4))
ax.plot(sizes, e_forms, "o-", color="steelblue", markersize=8, linewidth=2)
ax.set_xlabel("Number of atoms in supercell", fontsize=12)
ax.set_ylabel("Formation energy (eV)", fontsize=12)
ax.set_title("Vacancy formation energy convergence", fontsize=13)
ax.axhline(y=e_forms[-1], color="gray", linestyle="--", alpha=0.5)
ax.grid(True, alpha=0.3)
fig.tight_layout()
fig.savefig("vacancy_convergence.png", dpi=150)
print(f"\nConvergence plot saved to vacancy_convergence.png")

# ============================================================
# 6. Save results
# ============================================================
results = {
    "system": primitive.formula,
    "defect_type": "vacancy",
    "removed_species": str(removed_species),
    "supercell_convergence": {
        f"{n}x{n}x{n}": vacancy_energies[n] for n in supercell_sizes
    },
    "converged_formation_energy_eV": e_forms[-1],
}

with open("vacancy_results.json", "w") as f:
    json.dump(results, f, indent=2, default=str)
print("Results saved to vacancy_results.json")
```

#### Chemical Potential References

```python
#!/usr/bin/env python3
"""
Compute chemical potential references for defect formation energies.

Formation energy formula:
  E_f = E_defect - E_bulk + sum_i(n_i * mu_i) + q*(E_VBM + E_Fermi) + E_correction

where:
  n_i > 0 if species i is REMOVED (vacancy)
  n_i < 0 if species i is ADDED (interstitial)
  mu_i = chemical potential of species i

Chemical potential bounds for binary A_xB_y:
  mu_A + mu_B constrained by: x*mu_A + y*mu_B = Delta_H_f(A_xB_y)

  A-rich limit:  mu_A = mu_A(bulk),  mu_B = [Delta_H_f - x*mu_A(bulk)] / y
  B-rich limit:  mu_B = mu_B(bulk),  mu_A = [Delta_H_f - y*mu_B(bulk)] / x
"""

import numpy as np
from pymatgen.core import Structure, Element, Lattice
from pymatgen.io.ase import AseAtomsAdaptor
from ase.optimize import BFGS
from ase.filters import FrechetCellFilter
from ase.build import molecule
from mace.calculators import mace_mp
import json

calc = mace_mp(model="medium", dispersion=False, default_dtype="float64")
adaptor = AseAtomsAdaptor()

# --- Reference energies for elemental phases ---

# Mg: hcp metal
mg_hcp = Structure.from_spacegroup(
    "P6_3/mmc",
    lattice=Lattice.hexagonal(3.209, 5.211),
    species=["Mg", "Mg"],
    coords=[[1/3, 2/3, 0.25], [2/3, 1/3, 0.75]],
)
atoms_mg = adaptor.get_atoms(mg_hcp)
atoms_mg.calc = calc
ecf_mg = FrechetCellFilter(atoms_mg)
opt_mg = BFGS(ecf_mg, logfile="relax_Mg.log")
opt_mg.run(fmax=0.005, steps=300)
mu_Mg_bulk = atoms_mg.get_potential_energy() / len(atoms_mg)
print(f"mu_Mg(bulk) = {mu_Mg_bulk:.6f} eV/atom")

# O2 molecule (for O-rich limit)
o2 = molecule("O2")
o2.center(vacuum=10.0)
o2.calc = calc
opt_o2 = BFGS(o2, logfile="relax_O2.log")
opt_o2.run(fmax=0.005, steps=300)
mu_O_Orich = o2.get_potential_energy() / 2
print(f"mu_O(O-rich, from O2) = {mu_O_Orich:.6f} eV/atom")

# MgO bulk
mgo = Structure.from_spacegroup(
    "Fm-3m",
    lattice=[[4.212, 0, 0], [0, 4.212, 0], [0, 0, 4.212]],
    species=["Mg", "O"],
    coords=[[0.0, 0.0, 0.0], [0.5, 0.5, 0.5]],
)
atoms_mgo = adaptor.get_atoms(mgo)
atoms_mgo.calc = calc
ecf_mgo = FrechetCellFilter(atoms_mgo)
opt_mgo = BFGS(ecf_mgo, logfile="relax_MgO.log")
opt_mgo.run(fmax=0.005, steps=300)
e_MgO_per_fu = atoms_mgo.get_potential_energy() / (len(atoms_mgo) / 2)
print(f"E_MgO per formula unit = {e_MgO_per_fu:.6f} eV")

# Heat of formation: Delta_Hf = E_MgO - mu_Mg(bulk) - mu_O(O2/2)
delta_Hf = e_MgO_per_fu - mu_Mg_bulk - mu_O_Orich
print(f"\nDelta_Hf(MgO) = {delta_Hf:.4f} eV/f.u.")

# Chemical potential ranges
# O-rich limit: mu_O = mu_O(O2)/2, mu_Mg = Delta_Hf + mu_Mg(bulk) + mu_O_Orich - E_MgO
# Simplified: mu_Mg(O-rich) = mu_Mg(bulk) + Delta_Hf
mu_Mg_Orich = mu_Mg_bulk + delta_Hf
mu_O_Mgrich = mu_O_Orich + delta_Hf

print(f"\n--- Chemical Potential Bounds ---")
print(f"O-rich  limit: mu_Mg = {mu_Mg_Orich:.4f} eV, mu_O = {mu_O_Orich:.4f} eV")
print(f"Mg-rich limit: mu_Mg = {mu_Mg_bulk:.4f} eV,  mu_O = {mu_O_Mgrich:.4f} eV")

# Save references
refs = {
    "mu_Mg_bulk": mu_Mg_bulk,
    "mu_O_O2": mu_O_Orich,
    "e_MgO_per_fu": e_MgO_per_fu,
    "delta_Hf_MgO": delta_Hf,
    "O_rich": {"mu_Mg": mu_Mg_Orich, "mu_O": mu_O_Orich},
    "Mg_rich": {"mu_Mg": mu_Mg_bulk, "mu_O": mu_O_Mgrich},
}
with open("chemical_potentials.json", "w") as f:
    json.dump(refs, f, indent=2)
print("\nSaved to chemical_potentials.json")
```

#### Interstitial Defects

```python
#!/usr/bin/env python3
"""
Create and relax interstitial defects using pymatgen.
Example: O interstitial in MgO.
"""

import numpy as np
from pymatgen.core import Structure, Element, PeriodicSite
from pymatgen.analysis.defects.generators import InterstitialGenerator
from pymatgen.io.ase import AseAtomsAdaptor
from ase.optimize import BFGS
from mace.calculators import mace_mp

# Load relaxed bulk structure (or build it)
bulk = Structure.from_spacegroup(
    "Fm-3m",
    lattice=[[4.212, 0, 0], [0, 4.212, 0], [0, 0, 4.212]],
    species=["Mg", "O"],
    coords=[[0.0, 0.0, 0.0], [0.5, 0.5, 0.5]],
)

# Generate interstitial sites using Voronoi decomposition
# InterstitialGenerator finds symmetry-inequivalent interstitial sites
inter_gen = InterstitialGenerator()
interstitials = list(inter_gen.generate(bulk, {"O": Element("O")}))
print(f"Found {len(interstitials)} symmetry-distinct O interstitial sites")

calc = mace_mp(model="medium", dispersion=False, default_dtype="float64")
adaptor = AseAtomsAdaptor()

for i, interstitial_defect in enumerate(interstitials):
    print(f"\n--- Interstitial site {i} ---")
    # Generate supercell with the interstitial
    # The Defect object has a method to generate the supercell
    sc_struct = interstitial_defect.get_supercell_structure(
        sc_mat=np.eye(3) * 3,  # 3x3x3 supercell
    )
    print(f"  Supercell: {sc_struct.formula}, {len(sc_struct)} atoms")

    # Relax
    atoms = adaptor.get_atoms(sc_struct)
    atoms.calc = calc
    opt = BFGS(atoms, logfile=f"relax_interstitial_{i}.log")
    opt.run(fmax=0.01, steps=500)
    e_interstitial = atoms.get_potential_energy()
    print(f"  E_interstitial = {e_interstitial:.6f} eV")
    print(f"  Converged in {opt.nsteps} steps")
```

#### Visualize Defect Structure

```python
#!/usr/bin/env python3
"""Visualize a defect structure showing the vacancy site."""

import numpy as np
from pymatgen.core import Structure, Element
from pymatgen.io.ase import AseAtomsAdaptor
from ase.io import write as ase_write
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Build MgO supercell with vacancy
bulk = Structure.from_spacegroup(
    "Fm-3m",
    lattice=[[4.212, 0, 0], [0, 4.212, 0], [0, 0, 4.212]],
    species=["Mg", "O"],
    coords=[[0.0, 0.0, 0.0], [0.5, 0.5, 0.5]],
)
bulk.make_supercell([3, 3, 3])

# Record vacancy position before removal
o_indices = [i for i, site in enumerate(bulk) if site.specie == Element("O")]
vac_pos = bulk[o_indices[13]].coords  # pick a central O atom
bulk.remove_sites([o_indices[13]])

# 3D scatter plot
fig = plt.figure(figsize=(8, 8))
ax = fig.add_subplot(111, projection="3d")

for sp, color, sz in [(Element("Mg"), "blue", 40), (Element("O"), "red", 40)]:
    coords = np.array([s.coords for s in bulk if s.specie == sp])
    if len(coords) > 0:
        ax.scatter(coords[:, 0], coords[:, 1], coords[:, 2],
                   c=color, s=sz, alpha=0.6, label=str(sp))

# Mark vacancy
ax.scatter(*vac_pos, c="yellow", s=200, marker="X", edgecolors="black",
           linewidths=2, label="Vacancy", zorder=10)

ax.set_xlabel("x (A)")
ax.set_ylabel("y (A)")
ax.set_zlabel("z (A)")
ax.set_title("MgO 3x3x3 with O vacancy")
ax.legend()
fig.tight_layout()
fig.savefig("defect_structure.png", dpi=150)
print("Saved defect_structure.png")

# Also write CIF for external viewers
from pymatgen.io.cif import CifWriter
CifWriter(bulk).write_file("defect_supercell.cif")
print("Saved defect_supercell.cif")
```

### Method B: QE DFT

#### Step 1: Prepare QE Input for Bulk Supercell

```python
#!/usr/bin/env python3
"""
Generate QE input files for bulk and defect supercell SCF calculations.
"""

import numpy as np
from pymatgen.core import Structure, Element
from pymatgen.io.pwscf import PWInput
from pathlib import Path

# ============================================================
# Configuration
# ============================================================
PSEUDO_DIR = "./pseudo"
ECUTWFC = 60.0   # Ry, wavefunction cutoff (SSSP Efficiency)
ECUTRHO = 480.0  # Ry, charge density cutoff
KPOINTS_DENSITY = 3  # k-points per reciprocal lattice vector (for supercell)

# ============================================================
# Download pseudopotentials (SSSP Efficiency)
# ============================================================
Path(PSEUDO_DIR).mkdir(exist_ok=True)

# You would normally download these; provide filenames for the input
pseudos = {
    "Mg": "Mg.pbe-spnl-kjpaw_psl.1.0.0.UPF",
    "O":  "O.pbe-n-kjpaw_psl.1.0.0.UPF",
}

# ============================================================
# Build structures
# ============================================================
primitive = Structure.from_spacegroup(
    "Fm-3m",
    lattice=[[4.212, 0, 0], [0, 4.212, 0], [0, 0, 4.212]],
    species=["Mg", "O"],
    coords=[[0.0, 0.0, 0.0], [0.5, 0.5, 0.5]],
)

# Supercell
sc_size = 3
bulk_sc = primitive.copy()
bulk_sc.make_supercell([sc_size, sc_size, sc_size])

# Defect supercell (O vacancy)
defect_sc = bulk_sc.copy()
o_indices = [i for i, s in enumerate(defect_sc) if s.specie == Element("O")]
defect_sc.remove_sites([o_indices[0]])

# ============================================================
# Write QE inputs
# ============================================================
# Common control parameters
control_params = {
    "calculation": "relax",
    "restart_mode": "from_scratch",
    "pseudo_dir": PSEUDO_DIR,
    "outdir": "./tmp",
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

electrons_params = {
    "conv_thr": 1.0e-8,
    "mixing_beta": 0.4,
}

# Automatic k-points for supercell
kpts_sc = [max(1, KPOINTS_DENSITY) for _ in range(3)]

# Bulk supercell
pw_bulk = PWInput(
    bulk_sc,
    pseudo=pseudos,
    control=control_params | {"prefix": "bulk_sc"},
    system=system_params,
    electrons=electrons_params,
    kpoints_grid=tuple(kpts_sc),
)
pw_bulk.write_file("pw_bulk_sc.in")
print(f"Written pw_bulk_sc.in ({len(bulk_sc)} atoms, kpts={kpts_sc})")

# Defect supercell
pw_defect = PWInput(
    defect_sc,
    pseudo=pseudos,
    control=control_params | {"prefix": "defect_sc"},
    system=system_params,
    electrons=electrons_params,
    kpoints_grid=tuple(kpts_sc),
)
pw_defect.write_file("pw_defect_sc.in")
print(f"Written pw_defect_sc.in ({len(defect_sc)} atoms, kpts={kpts_sc})")
```

#### Step 2: Run QE Calculations

```bash
#!/bin/bash
# Run bulk and defect supercell calculations

NPROC=4  # adjust to your system

# Download pseudopotentials if needed
mkdir -p pseudo
cd pseudo
# SSSP Efficiency library - download specific pseudos
wget -nc "https://pseudopotentials.quantum-espresso.org/upf_files/Mg.pbe-spnl-kjpaw_psl.1.0.0.UPF" 2>/dev/null
wget -nc "https://pseudopotentials.quantum-espresso.org/upf_files/O.pbe-n-kjpaw_psl.1.0.0.UPF" 2>/dev/null
cd ..

# Bulk supercell relaxation
echo "=== Running bulk supercell ==="
mpirun -np $NPROC pw.x -in pw_bulk_sc.in > pw_bulk_sc.out 2>&1
echo "Bulk done: $(grep '!' pw_bulk_sc.out | tail -1)"

# Defect supercell relaxation
echo "=== Running defect supercell ==="
mpirun -np $NPROC pw.x -in pw_defect_sc.in > pw_defect_sc.out 2>&1
echo "Defect done: $(grep '!' pw_defect_sc.out | tail -1)"
```

#### Step 3: Extract Formation Energy from QE Output

```python
#!/usr/bin/env python3
"""
Parse QE output files and compute defect formation energy.
"""

import re
import json

def parse_qe_energy(filename):
    """Extract final total energy from QE output file."""
    energy = None
    with open(filename) as f:
        for line in f:
            if "!" in line and "total energy" in line:
                # Line format: "!    total energy              =    -1234.56789012 Ry"
                match = re.search(r"=\s+([-\d.]+)\s+Ry", line)
                if match:
                    energy = float(match.group(1)) * 13.605693123  # Ry to eV
    return energy

def parse_qe_natoms(filename):
    """Extract number of atoms from QE output."""
    with open(filename) as f:
        for line in f:
            if "number of atoms/cell" in line:
                return int(line.split("=")[1].strip())
    return None

# Parse outputs
e_bulk = parse_qe_energy("pw_bulk_sc.out")
e_defect = parse_qe_energy("pw_defect_sc.out")
n_bulk = parse_qe_natoms("pw_bulk_sc.out")
n_defect = parse_qe_natoms("pw_defect_sc.out")

print(f"E_bulk   = {e_bulk:.6f} eV  ({n_bulk} atoms)")
print(f"E_defect = {e_defect:.6f} eV ({n_defect} atoms)")

# Formation energy with elemental reference
# E_f = E_defect - E_bulk + n_removed * mu_species
# Using bulk per-atom energy as reference:
mu_O_approx = e_bulk / n_bulk  # crude approximation
e_form = e_defect - e_bulk + 1 * mu_O_approx
print(f"\nE_f(V_O) = {e_form:.4f} eV (bulk per-atom reference)")
print("Note: for accurate results, use proper chemical potential references.")
```

#### Charged Defect: QE Input Template

```python
#!/usr/bin/env python3
"""
Generate QE input for a CHARGED defect supercell.
Charged defects require:
  1. tot_charge in &SYSTEM
  2. Compensating background charge (automatic in QE)
  3. Post-hoc finite-size correction (Freysoldt or Kumagai)
"""

from pymatgen.core import Structure, Element
from pymatgen.io.pwscf import PWInput

# Build defect supercell (same as above)
primitive = Structure.from_spacegroup(
    "Fm-3m",
    lattice=[[4.212, 0, 0], [0, 4.212, 0], [0, 0, 4.212]],
    species=["Mg", "O"],
    coords=[[0.0, 0.0, 0.0], [0.5, 0.5, 0.5]],
)
defect_sc = primitive.copy()
defect_sc.make_supercell([3, 3, 3])
o_indices = [i for i, s in enumerate(defect_sc) if s.specie == Element("O")]
defect_sc.remove_sites([o_indices[0]])

charge_state = +2  # V_O^{2+}: removed neutral O, 2 electrons removed

pseudos = {
    "Mg": "Mg.pbe-spnl-kjpaw_psl.1.0.0.UPF",
    "O":  "O.pbe-n-kjpaw_psl.1.0.0.UPF",
}

pw_charged = PWInput(
    defect_sc,
    pseudo=pseudos,
    control={
        "calculation": "relax",
        "restart_mode": "from_scratch",
        "pseudo_dir": "./pseudo",
        "outdir": "./tmp",
        "tprnfor": True,
        "tstress": True,
    },
    system={
        "ecutwfc": 60.0,
        "ecutrho": 480.0,
        "occupations": "smearing",
        "smearing": "cold",
        "degauss": 0.01,
        "tot_charge": float(charge_state),  # +2 means 2 electrons removed
    },
    electrons={
        "conv_thr": 1.0e-8,
        "mixing_beta": 0.3,
    },
    kpoints_grid=(3, 3, 3),
)
pw_charged.write_file("pw_defect_charged.in")
print(f"Written pw_defect_charged.in (charge state q={charge_state})")
print("After running, apply Freysoldt correction (see below).")
```

#### Finite-Size Corrections for Charged Defects

```python
#!/usr/bin/env python3
"""
Freysoldt finite-size correction for charged defects.

The correction accounts for:
  1. Spurious electrostatic interaction between periodic images of the charged defect
  2. Interaction with the compensating jellium background
  3. Potential alignment between defect and bulk supercells

Requires: planar-averaged electrostatic potential from both bulk and defect calculations.

Formula:
  E_corr = E_Madelung(q, L, epsilon) + q * Delta_V

  E_Madelung ~ q^2 * alpha_M / (2 * epsilon * L)
  where alpha_M is the Madelung constant for the supercell shape,
        epsilon is the static dielectric constant,
        L is the supercell dimension.

For a cubic supercell:
  E_Madelung = -2.8373 * q^2 / (2 * epsilon * L)  [in eV, L in Angstrom]
  (the constant -2.8373 is the Madelung constant for SC lattice * e^2/(4*pi*eps0) in eV*A)
"""

import numpy as np

def freysoldt_correction_cubic(charge, dielectric_constant, lattice_parameter_A):
    """
    Simplified Freysoldt correction for a cubic supercell.

    Parameters
    ----------
    charge : int
        Charge state of the defect (e.g., +2 for V_O^{2+}).
    dielectric_constant : float
        Static (ionic + electronic) dielectric constant of the host.
    lattice_parameter_A : float
        Lattice parameter of the cubic supercell in Angstrom.

    Returns
    -------
    float
        Madelung energy correction in eV.
    """
    # Madelung constant for simple cubic * conversion factor
    # E = alpha * q^2 / (2 * eps * L)
    # alpha_sc = 2.8373 (Madelung constant for SC, in appropriate units)
    # With e^2/(4*pi*eps0) = 14.3996 eV*A
    alpha_madelung = 2.8373
    e2_4pieps0 = 14.3996  # eV * Angstrom

    E_madelung = (alpha_madelung * charge**2 * e2_4pieps0) / (
        2.0 * dielectric_constant * lattice_parameter_A
    )
    return E_madelung

# Example: V_O^{2+} in MgO
q = 2
eps_MgO = 9.8  # static dielectric constant of MgO
L_sc = 4.212 * 3  # 3x3x3 supercell of MgO (a=4.212 A)

E_corr = freysoldt_correction_cubic(q, eps_MgO, L_sc)
print(f"Freysoldt correction for V_O^{{2+}} in MgO:")
print(f"  Supercell: {L_sc:.2f} A")
print(f"  Dielectric constant: {eps_MgO}")
print(f"  Charge: {q}")
print(f"  E_Madelung correction: {E_corr:.4f} eV")

# For more accurate corrections, use pymatgen-analysis-defects:
print("\nFor full Freysoldt/Kumagai corrections with potential alignment:")
print("  pip install pymatgen-analysis-defects")
print("  from pymatgen.analysis.defects.corrections.freysoldt import get_freysoldt_correction")
print("  # Requires planar-averaged potentials from QE pp.x")

# Full formation energy for charged defect:
# E_f(q) = E_defect(q) - E_bulk + sum(n_i*mu_i) + q*(E_VBM + E_Fermi) + E_corr
print("\n--- Charged Defect Formation Energy Formula ---")
print("E_f(q) = E_defect(q) - E_bulk + sum(n_i * mu_i) + q*(E_VBM + E_Fermi) + E_corr")
print("  E_defect(q): total energy of charged defect supercell")
print("  E_bulk: total energy of perfect bulk supercell")
print("  n_i: number of atoms of species i removed (+) or added (-)")
print("  mu_i: chemical potential of species i")
print("  q: charge state")
print("  E_VBM: valence band maximum energy of bulk")
print("  E_Fermi: Fermi level referenced to VBM (0 to band gap)")
print("  E_corr: Freysoldt finite-size correction")
```

## Key Parameters

| Parameter | Recommended Value | Notes |
|---|---|---|
| Supercell size | >= 3x3x3 for cubic, or min ~10 A between defect images | Converge formation energy vs. supercell size |
| MACE fmax | 0.005-0.01 eV/A | Tighter for accurate energies |
| QE ecutwfc | 50-80 Ry | Depends on pseudopotential (check SSSP recommendations) |
| QE k-points | Gamma-only or 2x2x2 for large supercells | Scale inversely with supercell size |
| QE conv_thr | 1e-8 Ry | Tight convergence for energy differences |
| mixing_beta | 0.2-0.4 | Lower for metals or difficult convergence |
| Dielectric constant | Material-dependent | Needed for Freysoldt correction on charged defects |

## Interpreting Results

1. **Formation energy sign**: Positive means the defect costs energy to form (thermodynamically unfavorable at 0 K). Negative means spontaneous formation.
2. **Convergence with supercell size**: Formation energy should converge to within ~0.05 eV between the two largest supercells tested. If not, use larger supercells.
3. **Chemical potential dependence**: The formation energy depends on growth conditions through mu_i. Always report the chemical potential limits (e.g., "O-rich" vs. "metal-rich").
4. **Charged defect transition levels**: The Fermi level at which two charge states have equal formation energy is the charge transition level: epsilon(q1/q2) = (E_f(q1) - E_f(q2)) / (q2 - q1).
5. **MACE vs. QE**: MACE formation energies are typically within 0.1-0.5 eV of DFT for well-represented systems, but can deviate significantly for unusual chemistries. Always validate with DFT for publication.

## Common Issues

| Issue | Solution |
|---|---|
| SCF not converging for defect supercell | Reduce mixing_beta to 0.1-0.2; try different mixing modes; increase ecutrho |
| Large forces after vacancy creation | Normal; the neighboring atoms relax significantly. Use more relaxation steps. |
| Formation energy not converging with supercell | Use larger supercell; check k-point convergence independently |
| Charged defect gives wrong magnetic moment | Set `nspin=2` and `starting_magnetization` in QE input |
| Interstitial relaxes to wrong site | Try multiple starting positions; use smaller fmax; check with NEB if needed |
| MACE gives unreasonable energy for defect | System may be outside MACE training distribution; validate with QE DFT |
