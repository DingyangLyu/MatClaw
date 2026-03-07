# Vacancy Formation Energy Calculation

## When to Use

- Calculate the energy cost of creating a vacancy in a crystalline material
- Compare vacancy formation energies across different atomic sites or sublattices
- Study neutral and charged vacancy states (e.g., V_O^0, V_O^{1+}, V_O^{2+})
- Determine whether vacancies form spontaneously under specific chemical conditions
- Converge defect energetics with respect to supercell size
- Quickly screen vacancy stability using MACE before committing to expensive DFT

## Method Selection

| Criterion | ASE + MACE | QE DFT | VASP DFT |
|---|---|---|---|
| Speed | Seconds to minutes | Hours | Hours |
| Neutral vacancies | Good for trends | Publication quality | Publication quality |
| Charged vacancies | Not supported | Supported (tot_charge) | Supported (NELECT) |
| Finite-size corrections | N/A | Freysoldt via pp.x potential | Freysoldt via LOCPOT |
| Electronic structure at defect | Not available | Available | Available |
| Use case | Screening, quick estimates | Publication, charged defects | Publication, charged defects |

**Decision flow:**

```
Neutral vacancy, screening or trends?
  --> ASE + MACE (Method A): seconds, good for ranking sites

Neutral vacancy, publication quality?
  --> QE DFT (Method B) or VASP DFT (Method C)

Charged vacancy with transition levels?
  --> QE DFT (Method B) or VASP DFT (Method C): need electrostatic corrections

Quick MACE estimate followed by DFT validation?
  --> Method A first, then Method B or C for the most important configurations
```

## Prerequisites

- pymatgen (structure manipulation, supercell generation)
- ASE + mace-torch (Method A)
- Quantum ESPRESSO pw.x, pp.x (Method B)
- SSSP pseudopotentials for QE
- VASP with appropriate PAW potentials (Method C -- future external access)
- Optional: `pip install pymatgen-analysis-defects` for advanced defect workflows

## Detailed Steps

### Method A: ASE + MACE (Neutral Vacancies)

#### Complete Workflow: Vacancy Formation Energy with Supercell Convergence

```python
#!/usr/bin/env python3
"""
Vacancy formation energy calculation using ASE + MACE.
Computes E_vac = E_defect - (N-1)/N * E_bulk for neutral vacancies.
Tests convergence with supercell size.

Example: oxygen vacancy in MgO.
"""

import numpy as np
from pymatgen.core import Structure, Element
from pymatgen.io.ase import AseAtomsAdaptor
from ase.optimize import BFGS
from ase.filters import FrechetCellFilter
from mace.calculators import mace_mp
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import json
from pathlib import Path

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
# 2. Set up MACE calculator and relax primitive cell
# ============================================================
calc = mace_mp(model="medium", dispersion=False, default_dtype="float64")
adaptor = AseAtomsAdaptor()

atoms_prim = adaptor.get_atoms(primitive)
atoms_prim.calc = calc
ecf = FrechetCellFilter(atoms_prim)
opt = BFGS(ecf, logfile="relax_primitive.log")
opt.run(fmax=0.005, steps=500)

relaxed_primitive = adaptor.get_structure(atoms_prim)
e_prim = atoms_prim.get_potential_energy()
e_prim_per_atom = e_prim / len(atoms_prim)
print(f"Relaxed primitive energy: {e_prim:.6f} eV ({e_prim_per_atom:.6f} eV/atom)")

# ============================================================
# 3. Compute vacancy formation energy for each sublattice
# ============================================================
# Identify symmetry-inequivalent sites
unique_species = list(set(str(s.specie) for s in relaxed_primitive))
print(f"Unique species: {unique_species}")

sc_size = 3  # Use 3x3x3 supercell for production

results_by_species = {}

for sp_name in unique_species:
    print(f"\n{'='*50}")
    print(f"Vacancy on {sp_name} sublattice")
    print(f"{'='*50}")

    # Build bulk supercell
    bulk_sc = relaxed_primitive.copy()
    bulk_sc.make_supercell([sc_size, sc_size, sc_size])
    n_bulk = len(bulk_sc)

    # Relax bulk supercell (positions only, cell from primitive relaxation)
    atoms_bulk = adaptor.get_atoms(bulk_sc)
    atoms_bulk.calc = calc
    opt_bulk = BFGS(atoms_bulk, logfile=f"relax_bulk_{sc_size}x.log")
    opt_bulk.run(fmax=0.005, steps=300)
    e_bulk = atoms_bulk.get_potential_energy()
    print(f"  Bulk supercell: {n_bulk} atoms, E = {e_bulk:.6f} eV")

    # Find all sites of this species
    sp_indices = [i for i, s in enumerate(bulk_sc) if str(s.specie) == sp_name]

    # Create vacancy: remove the first atom of this species
    defect_sc = adaptor.get_structure(atoms_bulk).copy()
    removed_idx = sp_indices[0]
    print(f"  Removing {sp_name} at site {removed_idx}")
    defect_sc.remove_sites([removed_idx])
    n_defect = len(defect_sc)

    # Relax defect supercell (positions only)
    atoms_defect = adaptor.get_atoms(defect_sc)
    atoms_defect.calc = calc
    opt_def = BFGS(atoms_defect, logfile=f"relax_V_{sp_name}_{sc_size}x.log")
    opt_def.run(fmax=0.005, steps=500)
    e_defect = atoms_defect.get_potential_energy()
    print(f"  Defect supercell: {n_defect} atoms, E = {e_defect:.6f} eV")

    # Formation energy: E_vac = E_defect - (N-1)/N * E_bulk
    # This is equivalent to E_defect - E_bulk + E_bulk/N
    # and uses the bulk per-atom energy as the chemical potential reference.
    e_form = e_defect - (n_defect / n_bulk) * e_bulk
    print(f"  E_vac(V_{sp_name}) = {e_form:.4f} eV (bulk reference)")

    results_by_species[sp_name] = {
        "n_bulk": n_bulk,
        "e_bulk_eV": e_bulk,
        "e_defect_eV": e_defect,
        "e_form_eV": e_form,
        "supercell": f"{sc_size}x{sc_size}x{sc_size}",
    }

# ============================================================
# 4. Save results
# ============================================================
output = {
    "system": primitive.formula,
    "method": "MACE (medium)",
    "vacancies": results_by_species,
}
with open("vacancy_formation_results.json", "w") as f:
    json.dump(output, f, indent=2, default=str)
print("\nResults saved to vacancy_formation_results.json")
```

#### Supercell Size Convergence Study

```python
#!/usr/bin/env python3
"""
Converge vacancy formation energy with respect to supercell size.
Critical for ensuring the defect does not interact with its periodic images.
"""

import numpy as np
from pymatgen.core import Structure, Element
from pymatgen.io.ase import AseAtomsAdaptor
from ase.optimize import BFGS
from ase.filters import FrechetCellFilter
from mace.calculators import mace_mp
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import json

calc = mace_mp(model="medium", dispersion=False, default_dtype="float64")
adaptor = AseAtomsAdaptor()

# Build and relax primitive
primitive = Structure.from_spacegroup(
    "Fm-3m",
    lattice=[[4.212, 0, 0], [0, 4.212, 0], [0, 0, 4.212]],
    species=["Mg", "O"],
    coords=[[0.0, 0.0, 0.0], [0.5, 0.5, 0.5]],
)
atoms_prim = adaptor.get_atoms(primitive)
atoms_prim.calc = calc
ecf = FrechetCellFilter(atoms_prim)
BFGS(ecf, logfile=None).run(fmax=0.005, steps=500)
relaxed_prim = adaptor.get_structure(atoms_prim)

# Convergence test
supercell_sizes = [2, 3, 4, 5]
convergence = {}
vacancy_species = "O"

for n in supercell_sizes:
    print(f"\n--- {n}x{n}x{n} supercell ---")

    # Bulk supercell
    bulk_sc = relaxed_prim.copy()
    bulk_sc.make_supercell([n, n, n])
    n_bulk = len(bulk_sc)
    atoms_bulk = adaptor.get_atoms(bulk_sc)
    atoms_bulk.calc = calc
    BFGS(atoms_bulk, logfile=None).run(fmax=0.005, steps=300)
    e_bulk = atoms_bulk.get_potential_energy()

    # Defect supercell
    defect_sc = adaptor.get_structure(atoms_bulk).copy()
    sp_indices = [i for i, s in enumerate(defect_sc) if str(s.specie) == vacancy_species]
    defect_sc.remove_sites([sp_indices[0]])
    n_defect = len(defect_sc)
    atoms_defect = adaptor.get_atoms(defect_sc)
    atoms_defect.calc = calc
    BFGS(atoms_defect, logfile=None).run(fmax=0.005, steps=500)
    e_defect = atoms_defect.get_potential_energy()

    e_form = e_defect - (n_defect / n_bulk) * e_bulk

    # Minimum image distance (defect-defect separation)
    min_image_dist = min(bulk_sc.lattice.abc)
    print(f"  N_atoms={n_bulk}, image dist={min_image_dist:.2f} A, E_vac={e_form:.4f} eV")

    convergence[n] = {
        "n_atoms": n_bulk,
        "min_image_dist_A": min_image_dist,
        "e_form_eV": e_form,
    }

# Convergence plot
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

sizes = [convergence[n]["n_atoms"] for n in supercell_sizes]
e_forms = [convergence[n]["e_form_eV"] for n in supercell_sizes]
dists = [convergence[n]["min_image_dist_A"] for n in supercell_sizes]

# Plot vs. number of atoms
ax1.plot(sizes, e_forms, "o-", color="steelblue", markersize=8, linewidth=2)
ax1.set_xlabel("Number of atoms in supercell", fontsize=12)
ax1.set_ylabel("Formation energy (eV)", fontsize=12)
ax1.set_title(f"V_{vacancy_species} formation energy convergence", fontsize=13)
ax1.axhline(y=e_forms[-1], color="gray", linestyle="--", alpha=0.5)
ax1.grid(True, alpha=0.3)

# Plot vs. 1/L (finite-size scaling)
inv_L = [1.0 / d for d in dists]
ax2.plot(inv_L, e_forms, "s-", color="firebrick", markersize=8, linewidth=2)
ax2.set_xlabel("1 / L (1/A)", fontsize=12)
ax2.set_ylabel("Formation energy (eV)", fontsize=12)
ax2.set_title("Finite-size scaling", fontsize=13)
ax2.grid(True, alpha=0.3)

fig.tight_layout()
fig.savefig("vacancy_convergence.png", dpi=150)
print(f"\nConvergence plot saved to vacancy_convergence.png")

with open("vacancy_convergence.json", "w") as f:
    json.dump(convergence, f, indent=2, default=str)
print("Data saved to vacancy_convergence.json")
```

#### Multiple Vacancy Types on Same Sublattice

```python
#!/usr/bin/env python3
"""
Compare vacancy formation energies at all symmetry-inequivalent sites.
Uses pymatgen's VacancyGenerator for automatic site enumeration.
"""

import numpy as np
from pymatgen.core import Structure, Element
from pymatgen.analysis.defects.generators import VacancyGenerator
from pymatgen.io.ase import AseAtomsAdaptor
from ase.optimize import BFGS
from mace.calculators import mace_mp
import json

calc = mace_mp(model="medium", dispersion=False, default_dtype="float64")
adaptor = AseAtomsAdaptor()

# Example: SrTiO3 perovskite with multiple O sites
structure = Structure.from_spacegroup(
    "Pm-3m",
    lattice=[[3.905, 0, 0], [0, 3.905, 0], [0, 0, 3.905]],
    species=["Sr", "Ti", "O"],
    coords=[[0.0, 0.0, 0.0], [0.5, 0.5, 0.5], [0.5, 0.5, 0.0]],
)
print(f"Structure: {structure.formula}")

# Generate all symmetry-inequivalent vacancies
vac_gen = VacancyGenerator()
vacancies = list(vac_gen.generate(structure))
print(f"Found {len(vacancies)} symmetry-inequivalent vacancy types")

sc_mat = np.eye(3, dtype=int) * 3  # 3x3x3 supercell

# Relax bulk supercell once
bulk_sc = structure.copy()
bulk_sc.make_supercell([3, 3, 3])
atoms_bulk = adaptor.get_atoms(bulk_sc)
atoms_bulk.calc = calc
BFGS(atoms_bulk, logfile="relax_bulk_STO.log").run(fmax=0.005, steps=300)
e_bulk = atoms_bulk.get_potential_energy()
n_bulk = len(atoms_bulk)
print(f"Bulk supercell: {n_bulk} atoms, E = {e_bulk:.6f} eV")

# Compute formation energy for each vacancy
results = []
for i, vac in enumerate(vacancies):
    site_sp = str(vac.site.specie)
    print(f"\n--- Vacancy {i}: V_{site_sp} ---")

    # Generate defect supercell
    defect_sc = vac.get_supercell_structure(sc_mat=sc_mat)
    n_defect = len(defect_sc)

    atoms_defect = adaptor.get_atoms(defect_sc)
    atoms_defect.calc = calc
    BFGS(atoms_defect, logfile=f"relax_V_{site_sp}_{i}.log").run(fmax=0.005, steps=500)
    e_defect = atoms_defect.get_potential_energy()

    e_form = e_defect - (n_defect / n_bulk) * e_bulk
    print(f"  E_form = {e_form:.4f} eV")

    results.append({
        "vacancy_index": i,
        "removed_species": site_sp,
        "site_frac_coords": vac.site.frac_coords.tolist(),
        "n_defect_atoms": n_defect,
        "e_defect_eV": e_defect,
        "e_form_eV": e_form,
    })

# Summary
print("\n" + "=" * 60)
print("VACANCY FORMATION ENERGIES")
print("=" * 60)
print(f"{'Index':<8} {'Species':<10} {'E_form (eV)':<15}")
print("-" * 35)
for r in sorted(results, key=lambda x: x["e_form_eV"]):
    print(f"{r['vacancy_index']:<8} V_{r['removed_species']:<8} {r['e_form_eV']:<15.4f}")

with open("vacancy_types_results.json", "w") as f:
    json.dump(results, f, indent=2, default=str)
print("\nSaved to vacancy_types_results.json")
```

### Method B: QE DFT (Neutral and Charged Vacancies)

#### Step 1: Generate QE Input Files

```python
#!/usr/bin/env python3
"""
Generate QE input files for vacancy formation energy calculation.
Produces inputs for:
  1. Bulk supercell relaxation
  2. Neutral vacancy relaxation
  3. Charged vacancy relaxation (multiple charge states)
"""

import numpy as np
from pymatgen.core import Structure, Element
from pymatgen.io.pwscf import PWInput
from pathlib import Path

# ============================================================
# Configuration
# ============================================================
PSEUDO_DIR = "./pseudo"
ECUTWFC = 60.0   # Ry
ECUTRHO = 480.0  # Ry

pseudos = {
    "Mg": "Mg.pbe-spnl-kjpaw_psl.1.0.0.UPF",
    "O":  "O.pbe-n-kjpaw_psl.1.0.0.UPF",
}

SC_SIZE = 3
VACANCY_SPECIES = "O"
CHARGE_STATES = [0, +1, +2]  # V_O^0, V_O^{1+}, V_O^{2+}

# ============================================================
# Build structures
# ============================================================
primitive = Structure.from_spacegroup(
    "Fm-3m",
    lattice=[[4.212, 0, 0], [0, 4.212, 0], [0, 0, 4.212]],
    species=["Mg", "O"],
    coords=[[0.0, 0.0, 0.0], [0.5, 0.5, 0.5]],
)

bulk_sc = primitive.copy()
bulk_sc.make_supercell([SC_SIZE, SC_SIZE, SC_SIZE])

defect_sc = bulk_sc.copy()
o_indices = [i for i, s in enumerate(defect_sc) if s.specie == Element(VACANCY_SPECIES)]
defect_sc.remove_sites([o_indices[0]])

Path(PSEUDO_DIR).mkdir(exist_ok=True)

# ============================================================
# Common parameters
# ============================================================
control_common = {
    "calculation": "relax",
    "restart_mode": "from_scratch",
    "pseudo_dir": PSEUDO_DIR,
    "tprnfor": True,
    "tstress": True,
    "etot_conv_thr": 1.0e-6,
    "forc_conv_thr": 1.0e-4,
}

system_common = {
    "ecutwfc": ECUTWFC,
    "ecutrho": ECUTRHO,
    "occupations": "smearing",
    "smearing": "cold",
    "degauss": 0.01,
}

electrons_common = {
    "conv_thr": 1.0e-8,
    "mixing_beta": 0.3,
}

kpts = (2, 2, 2)  # Gamma-centered for large supercell

# ============================================================
# 1. Bulk supercell input
# ============================================================
pw_bulk = PWInput(
    bulk_sc,
    pseudo=pseudos,
    control=control_common | {"prefix": "bulk_sc", "outdir": "./tmp_bulk"},
    system=system_common,
    electrons=electrons_common,
    kpoints_grid=kpts,
)
pw_bulk.write_file("pw_bulk.in")
print(f"Written pw_bulk.in ({len(bulk_sc)} atoms)")

# ============================================================
# 2. Neutral vacancy input
# ============================================================
pw_neutral = PWInput(
    defect_sc,
    pseudo=pseudos,
    control=control_common | {"prefix": "vac_neutral", "outdir": "./tmp_vac_q0"},
    system=system_common,
    electrons=electrons_common,
    kpoints_grid=kpts,
)
pw_neutral.write_file("pw_vac_q0.in")
print(f"Written pw_vac_q0.in ({len(defect_sc)} atoms, q=0)")

# ============================================================
# 3. Charged vacancy inputs
# ============================================================
for q in CHARGE_STATES:
    if q == 0:
        continue  # already written above
    pw_charged = PWInput(
        defect_sc,
        pseudo=pseudos,
        control=control_common | {
            "prefix": f"vac_q{q:+d}",
            "outdir": f"./tmp_vac_q{q:+d}",
        },
        system=system_common | {
            "tot_charge": float(q),
        },
        electrons=electrons_common | {
            "mixing_beta": 0.2,  # tighter mixing for charged cells
        },
        kpoints_grid=kpts,
    )
    pw_charged.write_file(f"pw_vac_q{q:+d}.in")
    print(f"Written pw_vac_q{q:+d}.in ({len(defect_sc)} atoms, q={q:+d})")

print("\nAll QE input files generated.")
print("Download pseudopotentials, then run pw.x on each input file.")
```

#### Step 2: Run QE Calculations

```bash
#!/bin/bash
# Run vacancy formation energy calculations with QE
NPROC=4

# Download pseudopotentials
mkdir -p pseudo tmp_bulk tmp_vac_q0 tmp_vac_q+1 tmp_vac_q+2
cd pseudo
wget -nc "https://pseudopotentials.quantum-espresso.org/upf_files/Mg.pbe-spnl-kjpaw_psl.1.0.0.UPF" 2>/dev/null
wget -nc "https://pseudopotentials.quantum-espresso.org/upf_files/O.pbe-n-kjpaw_psl.1.0.0.UPF" 2>/dev/null
cd ..

# 1. Bulk supercell
echo "=== Bulk supercell ==="
mpirun -np $NPROC pw.x -in pw_bulk.in > pw_bulk.out 2>&1
echo "Bulk: $(grep '!' pw_bulk.out | tail -1)"

# 2. Neutral vacancy
echo "=== Neutral vacancy ==="
mpirun -np $NPROC pw.x -in pw_vac_q0.in > pw_vac_q0.out 2>&1
echo "V_O^0: $(grep '!' pw_vac_q0.out | tail -1)"

# 3. Charged vacancies
for Q in +1 +2; do
    echo "=== Charged vacancy q=$Q ==="
    mpirun -np $NPROC pw.x -in pw_vac_q${Q}.in > pw_vac_q${Q}.out 2>&1
    echo "V_O^{$Q}: $(grep '!' pw_vac_q${Q}.out | tail -1)"
done
```

#### Step 3: Extract Formation Energies and Apply Corrections

```python
#!/usr/bin/env python3
"""
Parse QE outputs and compute vacancy formation energies.
For charged defects, apply the Freysoldt finite-size correction.

Formation energy:
  E_f(V_X^q) = E_defect(q) - E_bulk + n_X * mu_X + q*(E_VBM + E_Fermi) + E_corr

where:
  E_defect(q) = total energy of charged defect supercell
  E_bulk = total energy of perfect bulk supercell
  n_X = number of atoms removed (positive for vacancy)
  mu_X = chemical potential of removed species
  q = charge state
  E_VBM = valence band maximum of the bulk
  E_Fermi = Fermi level referenced to VBM (variable, 0 to band gap)
  E_corr = Freysoldt finite-size correction for charged defects
"""

import re
import numpy as np
import json

# ============================================================
# Helper functions
# ============================================================
def parse_qe_energy(filename):
    """Extract final total energy in eV from QE output."""
    energy = None
    with open(filename) as f:
        for line in f:
            if "!" in line and "total energy" in line:
                match = re.search(r"=\s+([-\d.]+)\s+Ry", line)
                if match:
                    energy = float(match.group(1)) * 13.605693123  # Ry -> eV
    return energy

def parse_qe_vbm(filename):
    """Extract VBM energy from QE output (highest occupied state)."""
    vbm = None
    with open(filename) as f:
        for line in f:
            if "highest occupied" in line:
                match = re.search(r"([-\d.]+)\s+eV", line)
                if match:
                    vbm = float(match.group(1))
            elif "highest occupied, lowest unoccupied" in line:
                vals = re.findall(r"[-\d.]+", line)
                if len(vals) >= 1:
                    vbm = float(vals[0])
    return vbm

def parse_qe_natoms(filename):
    """Extract number of atoms from QE output."""
    with open(filename) as f:
        for line in f:
            if "number of atoms/cell" in line:
                return int(line.split("=")[1].strip())
    return None

def freysoldt_correction_cubic(charge, dielectric, L_angstrom):
    """
    Simplified Freysoldt (Makov-Payne) correction for cubic supercell.

    E_corr = alpha_M * q^2 * e^2 / (2 * epsilon * L)

    Parameters
    ----------
    charge : int
        Defect charge state.
    dielectric : float
        Static dielectric constant of the host material.
    L_angstrom : float
        Cubic supercell lattice parameter in Angstrom.

    Returns
    -------
    float
        Correction energy in eV.
    """
    alpha_madelung = 2.8373  # Madelung constant for simple cubic
    e2_4pieps0 = 14.3996     # eV * Angstrom
    return (alpha_madelung * charge**2 * e2_4pieps0) / (2.0 * dielectric * L_angstrom)

# ============================================================
# Parse all outputs
# ============================================================
e_bulk = parse_qe_energy("pw_bulk.out")
n_bulk = parse_qe_natoms("pw_bulk.out")
e_vbm = parse_qe_vbm("pw_bulk.out")

print(f"Bulk supercell: E = {e_bulk:.6f} eV, N = {n_bulk}")
if e_vbm is not None:
    print(f"VBM = {e_vbm:.4f} eV")

# Chemical potential reference (bulk per-atom as simple reference)
mu_O_bulk_ref = e_bulk / n_bulk
print(f"mu_O (bulk ref) = {mu_O_bulk_ref:.6f} eV/atom")

# Material parameters for Freysoldt correction
dielectric_MgO = 9.8      # static dielectric constant
L_supercell = 4.212 * 3   # cubic supercell lattice parameter (A)

# Parse each charge state
charge_states = [0, 1, 2]
results = {}

for q in charge_states:
    qstr = f"q{q:+d}" if q != 0 else "q0"
    outfile = f"pw_vac_{qstr}.out"

    e_defect = parse_qe_energy(outfile)
    n_defect = parse_qe_natoms(outfile)

    if e_defect is None:
        print(f"\nWARNING: Could not parse {outfile}")
        continue

    print(f"\n--- V_O^{{{q:+d}}} ---")
    print(f"  E_defect = {e_defect:.6f} eV, N = {n_defect}")

    # Formation energy (without Fermi level term for charged defects)
    # E_f = E_defect - E_bulk + 1 * mu_O
    # For charged: + q * (E_VBM + E_Fermi) + E_corr
    n_removed = n_bulk - n_defect  # should be 1
    e_form_base = e_defect - e_bulk + n_removed * mu_O_bulk_ref

    # Finite-size correction (only for charged defects)
    e_corr = 0.0
    if q != 0:
        e_corr = freysoldt_correction_cubic(q, dielectric_MgO, L_supercell)
        print(f"  Freysoldt correction: {e_corr:.4f} eV")

    # At E_Fermi = 0 (VBM):
    e_form_at_vbm = e_form_base + e_corr
    if q != 0 and e_vbm is not None:
        # The q*E_VBM term comes from referencing to VBM
        e_form_at_vbm = e_form_base + q * e_vbm + e_corr

    print(f"  E_form (at E_F=VBM) = {e_form_at_vbm:.4f} eV")

    results[q] = {
        "charge": q,
        "e_defect_eV": e_defect,
        "e_form_base_eV": e_form_base,
        "e_correction_eV": e_corr,
        "e_form_at_VBM_eV": e_form_at_vbm,
    }

# Save results
with open("qe_vacancy_results.json", "w") as f:
    json.dump(results, f, indent=2, default=float)
print("\nResults saved to qe_vacancy_results.json")
```

#### Step 4: Extract Electrostatic Potential for Full Freysoldt Correction

```python
#!/usr/bin/env python3
"""
Generate pp.x input to extract the planar-averaged electrostatic potential
from bulk and defect calculations. Required for the full Freysoldt correction
with potential alignment.
"""

# pp.x input for bulk
pp_bulk = """&INPUTPP
  prefix    = 'bulk_sc',
  outdir    = './tmp_bulk',
  filplot   = 'pot_bulk.dat',
  plot_num  = 11,
/

&PLOT
  iflag         = 3,
  output_format = 0,
  fileout       = 'avg_pot_bulk.dat',
  e1(1) = 0.0, e1(2) = 0.0, e1(3) = 1.0,
  x0(1) = 0.0, x0(2) = 0.0, x0(3) = 0.0,
  nx    = 500,
/
"""

# pp.x input for defect (each charge state)
for q in [0, 1, 2]:
    qstr = f"q{q:+d}" if q != 0 else "q0"
    pp_defect = f"""&INPUTPP
  prefix    = 'vac_{qstr}',
  outdir    = './tmp_vac_{qstr}',
  filplot   = 'pot_vac_{qstr}.dat',
  plot_num  = 11,
/

&PLOT
  iflag         = 3,
  output_format = 0,
  fileout       = 'avg_pot_vac_{qstr}.dat',
  e1(1) = 0.0, e1(2) = 0.0, e1(3) = 1.0,
  x0(1) = 0.0, x0(2) = 0.0, x0(3) = 0.0,
  nx    = 500,
/
"""
    with open(f"pp_vac_{qstr}.in", "w") as f:
        f.write(pp_defect)

with open("pp_bulk.in", "w") as f:
    f.write(pp_bulk)

print("Written pp.x input files.")
print("Run: mpirun -np 4 pp.x -in pp_bulk.in > pp_bulk.out")
print("     mpirun -np 4 pp.x -in pp_vac_q0.in > pp_vac_q0.out")
print("     etc.")
print("\nThen use pymatgen-analysis-defects for the full Freysoldt correction:")
print("  pip install pymatgen-analysis-defects")
print("  from pymatgen.analysis.defects.corrections.freysoldt import get_freysoldt_correction")
```

### Method C: VASP DFT (Neutral and Charged Vacancies)

When VASP is available via external access, vacancy calculations follow the same physics but use VASP-specific input files.

#### Step 1: Generate VASP Input Files

```python
#!/usr/bin/env python3
"""
Generate VASP input files for vacancy formation energy calculation.
Produces INCAR, POSCAR, KPOINTS for bulk and defect supercells.

Note: VASP execution will be available via future external access.
This script generates the input files for preparation.
"""

import numpy as np
from pymatgen.core import Structure, Element
from pymatgen.io.vasp import Incar, Poscar, Kpoints
from pathlib import Path

# ============================================================
# Build structures
# ============================================================
primitive = Structure.from_spacegroup(
    "Fm-3m",
    lattice=[[4.212, 0, 0], [0, 4.212, 0], [0, 0, 4.212]],
    species=["Mg", "O"],
    coords=[[0.0, 0.0, 0.0], [0.5, 0.5, 0.5]],
)

SC_SIZE = 3
VACANCY_SPECIES = "O"
CHARGE_STATES = [0, +1, +2]

bulk_sc = primitive.copy()
bulk_sc.make_supercell([SC_SIZE, SC_SIZE, SC_SIZE])

defect_sc = bulk_sc.copy()
o_indices = [i for i, s in enumerate(defect_sc) if s.specie == Element(VACANCY_SPECIES)]
defect_sc.remove_sites([o_indices[0]])

# ============================================================
# 1. Bulk supercell
# ============================================================
bulk_dir = Path("vasp_bulk")
bulk_dir.mkdir(exist_ok=True)

Poscar(bulk_sc).write_file(str(bulk_dir / "POSCAR"))

incar_bulk = Incar({
    "SYSTEM": f"{primitive.composition.reduced_formula} bulk {SC_SIZE}x{SC_SIZE}x{SC_SIZE}",
    "ENCUT": 520,
    "PREC": "Accurate",
    "EDIFF": 1e-6,
    "EDIFFG": -0.01,
    "IBRION": 2,
    "ISIF": 2,       # relax ions only, keep cell fixed
    "NSW": 200,
    "ISMEAR": 0,
    "SIGMA": 0.05,
    "LREAL": "Auto",  # for large supercells
    "LWAVE": False,
    "LCHARG": True,    # save CHGCAR for alignment
    "LORBIT": 11,
    "ALGO": "Normal",
    "NELM": 200,
})
incar_bulk.write_file(str(bulk_dir / "INCAR"))

kpoints_bulk = Kpoints.gamma_automatic(kpts=(2, 2, 2), shift=(0, 0, 0))
kpoints_bulk.write_file(str(bulk_dir / "KPOINTS"))

print(f"Bulk VASP inputs written to {bulk_dir}/ ({len(bulk_sc)} atoms)")

# ============================================================
# 2. Vacancy supercells (neutral + charged)
# ============================================================
for q in CHARGE_STATES:
    qstr = f"q{q:+d}" if q != 0 else "q0"
    vac_dir = Path(f"vasp_vac_{qstr}")
    vac_dir.mkdir(exist_ok=True)

    Poscar(defect_sc).write_file(str(vac_dir / "POSCAR"))

    # Determine NELECT adjustment for charged defect
    # For VASP: NELECT = default_NELECT - q
    # (removing electrons for positive charge, adding for negative)
    incar_vac = Incar({
        "SYSTEM": f"V_{VACANCY_SPECIES}^{{{q:+d}}} in {primitive.composition.reduced_formula}",
        "ENCUT": 520,
        "PREC": "Accurate",
        "EDIFF": 1e-6,
        "EDIFFG": -0.01,
        "IBRION": 2,
        "ISIF": 2,
        "NSW": 300,       # more steps for defect relaxation
        "ISMEAR": 0,
        "SIGMA": 0.05,
        "LREAL": "Auto",
        "LWAVE": False,
        "LCHARG": True,   # needed for potential alignment
        "LVHAR": True,     # write LOCPOT for Freysoldt correction
        "LORBIT": 11,
        "ALGO": "Normal",
        "NELM": 300,
    })

    if q != 0:
        # NOTE: User must determine default NELECT from a preliminary run
        # or calculate from pseudopotential valence electrons.
        # NELECT = sum(ZVAL) - q
        # Here we add a comment; the actual value depends on POTCAR.
        incar_vac["NELECT"] = "REPLACE_WITH_ZVAL_SUM_MINUS_Q"
        # For spin-polarized charged defects:
        if q % 2 != 0:
            incar_vac["ISPIN"] = 2
            incar_vac["LORBIT"] = 11

    incar_vac.write_file(str(vac_dir / "INCAR"))

    kpoints_vac = Kpoints.gamma_automatic(kpts=(2, 2, 2), shift=(0, 0, 0))
    kpoints_vac.write_file(str(vac_dir / "KPOINTS"))

    print(f"V_{VACANCY_SPECIES}^{{{q:+d}}} VASP inputs written to {vac_dir}/ ({len(defect_sc)} atoms)")

# ============================================================
# 3. Determine NELECT for charged defects
# ============================================================
print("""
=== IMPORTANT: Setting NELECT for charged defects in VASP ===

1. Run the neutral defect calculation first (or the bulk).
2. From the OUTCAR, find the default number of electrons:
     grep NELECT OUTCAR
3. For charge state q:
     NELECT = default_NELECT - q
   (positive q removes electrons, negative q adds electrons)
4. Replace REPLACE_WITH_ZVAL_SUM_MINUS_Q in the INCAR.

Example for V_O^{2+} in MgO (3x3x3 supercell):
  Default NELECT for 53-atom defect cell ~ 424 (if using standard PAW)
  NELECT for q=+2: 424 - 2 = 422

Also ensure LVHAR = .TRUE. to write LOCPOT for Freysoldt correction.
""")
```

#### Step 2: Parse VASP Results and Apply Freysoldt Correction

```python
#!/usr/bin/env python3
"""
Parse VASP vacancy calculation results and compute formation energies.
Apply Freysoldt correction for charged defects using LOCPOT files.

Note: This script is for when VASP is available via external access.
"""

import numpy as np
import json

# For VASP parsing
from pymatgen.io.vasp import Vasprun, Locpot, Outcar

def get_vasp_energy(vasprun_path):
    """Extract final energy from vasprun.xml."""
    vr = Vasprun(vasprun_path, parse_dos=False, parse_eigen=False)
    return vr.final_energy

def get_vasp_vbm(vasprun_path):
    """Extract VBM from vasprun.xml."""
    vr = Vasprun(vasprun_path)
    bs = vr.get_band_structure()
    return bs.get_vbm()["energy"]

def get_vasp_nelect(outcar_path):
    """Extract NELECT from OUTCAR."""
    outcar = Outcar(outcar_path)
    return outcar.nelect

# ============================================================
# Parse results (adapt paths to your directory structure)
# ============================================================
print("=== Parsing VASP vacancy results ===")
print("(Adapt file paths to your calculation directories)")
print()

# Example parsing code (uncomment when VASP outputs are available):
#
# e_bulk = get_vasp_energy("vasp_bulk/vasprun.xml")
# e_vbm = get_vasp_vbm("vasp_bulk/vasprun.xml")
# n_bulk = len(Vasprun("vasp_bulk/vasprun.xml").final_structure)
#
# for q in [0, 1, 2]:
#     qstr = f"q{q:+d}" if q != 0 else "q0"
#     e_defect = get_vasp_energy(f"vasp_vac_{qstr}/vasprun.xml")
#     n_defect = len(Vasprun(f"vasp_vac_{qstr}/vasprun.xml").final_structure)
#
#     mu_O = e_bulk / n_bulk  # simple bulk reference
#     e_form = e_defect - e_bulk + (n_bulk - n_defect) * mu_O
#
#     if q != 0:
#         # Apply Freysoldt correction using pymatgen
#         from pymatgen.analysis.defects.corrections.freysoldt import (
#             get_freysoldt_correction
#         )
#         bulk_locpot = Locpot.from_file("vasp_bulk/LOCPOT")
#         defect_locpot = Locpot.from_file(f"vasp_vac_{qstr}/LOCPOT")
#         correction = get_freysoldt_correction(
#             q, dielectric=9.8,
#             bulk_locpot=bulk_locpot,
#             defect_locpot=defect_locpot,
#         )
#         e_form += correction
#
#     print(f"V_O^{{{q:+d}}}: E_form = {e_form:.4f} eV (at E_F = VBM)")

print("Full Freysoldt correction requires:")
print("  1. LOCPOT from bulk and defect calculations (LVHAR=.TRUE.)")
print("  2. Dielectric constant of the host material")
print("  3. pip install pymatgen-analysis-defects")
```

## Key Parameters

| Parameter | Recommended Value | Notes |
|---|---|---|
| Supercell size | >= 3x3x3 or min 10 A between defect images | Converge E_form vs supercell size; larger for charged defects |
| MACE fmax | 0.005 eV/A | Tight convergence for accurate energy differences |
| QE ecutwfc | 50-80 Ry | Follow SSSP recommendations for your elements |
| QE k-points | Gamma or 2x2x2 for large supercells | Scale inversely with supercell size |
| QE conv_thr | 1e-8 Ry | Tight SCF convergence for energy differences |
| QE mixing_beta | 0.2-0.3 | Lower for charged defects or metals |
| VASP ENCUT | 520 eV (or 1.3x ENMAX) | Check POTCAR for ENMAX |
| VASP EDIFF | 1e-6 eV | Tight electronic convergence |
| VASP EDIFFG | -0.01 eV/A | Ionic force convergence |
| VASP LVHAR | .TRUE. | Required to write LOCPOT for Freysoldt correction |
| Dielectric constant | Material-dependent | Needed for charged defect corrections (DFT or experimental) |

## Interpreting Results

1. **Formation energy formula**: `E_vac = E_defect - (N-1)/N * E_bulk` uses the bulk per-atom energy as the chemical potential reference. This is the simplest form. For realistic conditions, use explicit chemical potential references (elemental phases, gas molecules).

2. **Positive vs. negative formation energy**: Positive means the vacancy costs energy to create. Under equilibrium conditions at temperature T, the concentration is `c ~ exp(-E_f / k_B T)`. Even a 1 eV formation energy gives significant concentrations at high T.

3. **Chemical potential dependence**: In a binary A_xB_y, the formation energy of V_A depends on mu_A, which ranges between A-rich and B-rich limits. Always report which limit is used.

4. **Supercell convergence**: Formation energy should converge within 0.05 eV between the two largest supercells. For charged defects, convergence is slower (1/L scaling) and requires Freysoldt corrections.

5. **Charged defect corrections**: The Freysoldt correction has two parts: (a) image-charge Madelung energy (scales as q^2/L) and (b) potential alignment (typically 0.01-0.2 eV). Always apply corrections for charged defects.

6. **MACE vs. DFT**: MACE typically reproduces neutral vacancy formation energies within 0.1-0.5 eV of DFT for well-represented systems. MACE cannot handle charged defects. Always validate with DFT for publication.

## Common Issues

| Issue | Solution |
|---|---|
| Formation energy not converging with supercell size | Use larger supercells; for charged defects, apply Freysoldt correction and check 1/L^3 extrapolation |
| SCF not converging for charged defect | Reduce mixing_beta to 0.1-0.2; start from neutral CHGCAR; increase NELM |
| Negative formation energy for a vacancy | Check chemical potential reference; may indicate the material is unstable under those conditions |
| Large relaxation around vacancy site | Expected; neighbors relax into or away from the vacancy. Use more ionic steps. |
| Odd spin state in charged vacancy | Set nspin=2 (QE) or ISPIN=2 (VASP); provide starting magnetization |
| MACE gives unreasonable vacancy energy | System outside MACE training data; validate with DFT |
| Freysoldt correction is very large (>1 eV) | Supercell is too small for the charge state; use a larger supercell |
| Different results with different pseudopotentials | Expected; use consistent pseudopotential library (SSSP for QE, standard PAW for VASP) |
| VASP: wrong NELECT for charged defect | Run neutral first, grep NELECT from OUTCAR, then adjust: NELECT = default - q |
