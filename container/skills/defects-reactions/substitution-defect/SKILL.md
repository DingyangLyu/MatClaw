# Substitutional Defect Energy Calculation

## When to Use

- Calculate the formation energy of substituting atom A with atom B in a crystal
- Screen dopant solubility and site preference (e.g., which site does Al prefer in SrTiO3?)
- Determine charge states of substitutional dopants
- Compute chemical potential-dependent formation energies across growth conditions
- Evaluate co-doping strategies (e.g., donor + acceptor compensation)

## Method Selection

| Criterion | ASE + MACE | QE DFT | VASP DFT |
|---|---|---|---|
| Speed | Seconds to minutes | Hours | Hours |
| Neutral substitution | Good for trends | Publication quality | Publication quality |
| Charged substitution | Not supported | Supported (tot_charge) | Supported (NELECT) |
| Donor/acceptor levels | Not available | From charge transition levels | From charge transition levels |
| Electronic structure | Not available | Available (PDOS at defect) | Available (PDOS at defect) |
| Use case | Screening dopant sites, trends | Publication, electronic analysis | Publication, electronic analysis |

**Decision flow:**

```
Quick screening of multiple dopants or sites?
  --> ASE + MACE (Method A): rank by formation energy

Neutral substitution, publication quality?
  --> QE DFT (Method B) or VASP DFT (Method C)

Charged dopant (donor/acceptor) with transition levels?
  --> QE DFT or VASP DFT required: need charge state analysis

Electronic structure at dopant site (PDOS, charge density)?
  --> QE DFT or VASP DFT required
```

## Prerequisites

- pymatgen (structure manipulation, defect generation)
- ASE + mace-torch (Method A)
- Quantum ESPRESSO pw.x (Method B)
- SSSP pseudopotentials for QE
- VASP with PAW potentials (Method C -- future external access)
- Optional: `pip install pymatgen-analysis-defects` for systematic defect generation

## Detailed Steps

### Method A: ASE + MACE (Neutral Substitutions)

#### Complete Workflow: Dopant Screening

```python
#!/usr/bin/env python3
"""
Substitutional defect formation energy using ASE + MACE.
Screen multiple dopants at multiple sites.

Formation energy:
  E_f(A_B) = E_defect - E_bulk - mu_A + mu_B

where A replaces B in the host.
  mu_A = chemical potential of the substituting atom
  mu_B = chemical potential of the replaced atom
  (using elemental bulk references in this example)

Example: dopants in MgO (replacing Mg or O)
"""

import numpy as np
from pymatgen.core import Structure, Element
from pymatgen.io.ase import AseAtomsAdaptor
from ase.optimize import BFGS
from ase.filters import FrechetCellFilter
from ase.build import bulk as ase_bulk
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

# Host material: MgO
host = Structure.from_spacegroup(
    "Fm-3m",
    lattice=[[4.212, 0, 0], [0, 4.212, 0], [0, 0, 4.212]],
    species=["Mg", "O"],
    coords=[[0.0, 0.0, 0.0], [0.5, 0.5, 0.5]],
)
print(f"Host: {host.formula}")

# ============================================================
# 2. Relax host primitive cell
# ============================================================
atoms_host = adaptor.get_atoms(host)
atoms_host.calc = calc
ecf = FrechetCellFilter(atoms_host)
BFGS(ecf, logfile=None).run(fmax=0.005, steps=500)
host_relaxed = adaptor.get_structure(atoms_host)
e_host_per_atom = atoms_host.get_potential_energy() / len(atoms_host)
print(f"Host relaxed: E/atom = {e_host_per_atom:.6f} eV")

# ============================================================
# 3. Compute elemental reference energies (chemical potentials)
# ============================================================
def get_elemental_energy(symbol):
    """
    Compute per-atom energy of an elemental reference phase using MACE.
    Uses common crystal structures for each element.
    """
    structures = {
        "Mg": ("hcp", 3.209),
        "O":  None,  # use O2 molecule
        "Al": ("fcc", 4.046),
        "Ca": ("fcc", 5.588),
        "Ti": ("hcp", 2.951),
        "Fe": ("bcc", 2.870),
        "Ni": ("fcc", 3.524),
        "Li": ("bcc", 3.490),
        "Na": ("bcc", 4.291),
        "N":  None,  # use N2 molecule
        "Si": ("diamond", 5.431),
    }

    if symbol not in structures:
        raise ValueError(f"No reference structure defined for {symbol}. Add it to the dictionary.")

    ref = structures[symbol]
    c = mace_mp(model="medium", dispersion=False, default_dtype="float64")

    if ref is None:
        # Diatomic molecule
        from ase.build import molecule
        mol = molecule(f"{symbol}2")
        mol.center(vacuum=10.0)
        mol.calc = c
        BFGS(mol, logfile=None).run(fmax=0.005, steps=200)
        return mol.get_potential_energy() / 2

    crystal_type, a = ref
    atoms = ase_bulk(symbol, crystal_type, a=a)
    atoms.calc = c
    ecf = FrechetCellFilter(atoms)
    BFGS(ecf, logfile=None).run(fmax=0.005, steps=300)
    return atoms.get_potential_energy() / len(atoms)

# Pre-compute references for host species
mu_ref = {}
for sp in ["Mg", "O"]:
    mu_ref[sp] = get_elemental_energy(sp)
    print(f"mu_{sp}(ref) = {mu_ref[sp]:.6f} eV/atom")

# ============================================================
# 4. Build bulk supercell and relax
# ============================================================
SC_SIZE = 3
bulk_sc = host_relaxed.copy()
bulk_sc.make_supercell([SC_SIZE, SC_SIZE, SC_SIZE])
n_bulk = len(bulk_sc)

atoms_bulk = adaptor.get_atoms(bulk_sc)
atoms_bulk.calc = calc
BFGS(atoms_bulk, logfile="relax_bulk_sc.log").run(fmax=0.005, steps=300)
e_bulk = atoms_bulk.get_potential_energy()
print(f"\nBulk supercell: {n_bulk} atoms, E = {e_bulk:.6f} eV")

# ============================================================
# 5. Screen substitutional dopants
# ============================================================
# Define substitutions: (dopant, site_to_replace)
substitutions = [
    ("Al", "Mg"),   # Al on Mg site (Al_Mg)
    ("Ca", "Mg"),   # Ca on Mg site (Ca_Mg)
    ("Li", "Mg"),   # Li on Mg site (Li_Mg)
    ("Ti", "Mg"),   # Ti on Mg site (Ti_Mg)
    ("N",  "O"),    # N on O site (N_O)
    ("Si", "Mg"),   # Si on Mg site (Si_Mg)
]

results = []

for dopant, replaced in substitutions:
    print(f"\n--- {dopant} on {replaced} site ({dopant}_{replaced}) ---")

    # Get dopant reference energy
    if dopant not in mu_ref:
        mu_ref[dopant] = get_elemental_energy(dopant)
        print(f"  mu_{dopant}(ref) = {mu_ref[dopant]:.6f} eV/atom")

    # Build defect supercell: replace one atom
    defect_sc = adaptor.get_structure(atoms_bulk).copy()
    target_indices = [i for i, s in enumerate(defect_sc) if str(s.specie) == replaced]

    if not target_indices:
        print(f"  ERROR: No {replaced} atoms found in supercell")
        continue

    # Replace the first atom of the target species
    replace_idx = target_indices[0]
    defect_sc.replace(replace_idx, Element(dopant))
    print(f"  Replaced {replaced} at site {replace_idx} with {dopant}")

    # Relax defect supercell
    atoms_defect = adaptor.get_atoms(defect_sc)
    atoms_defect.calc = mace_mp(model="medium", dispersion=False, default_dtype="float64")
    BFGS(atoms_defect, logfile=f"relax_{dopant}_{replaced}.log").run(fmax=0.005, steps=500)
    e_defect = atoms_defect.get_potential_energy()

    # Formation energy: E_f = E_defect - E_bulk - mu_dopant + mu_replaced
    e_form = e_defect - e_bulk - mu_ref[dopant] + mu_ref[replaced]
    print(f"  E_defect = {e_defect:.6f} eV")
    print(f"  E_form({dopant}_{replaced}) = {e_form:.4f} eV")

    results.append({
        "dopant": dopant,
        "replaced": replaced,
        "label": f"{dopant}_{replaced}",
        "e_defect_eV": e_defect,
        "e_form_eV": e_form,
    })

# ============================================================
# 6. Summary and visualization
# ============================================================
print("\n" + "=" * 60)
print("SUBSTITUTIONAL DEFECT FORMATION ENERGIES")
print("=" * 60)
print(f"{'Defect':<15} {'E_form (eV)':<15} {'Favorable?'}")
print("-" * 40)
for r in sorted(results, key=lambda x: x["e_form_eV"]):
    fav = "low cost" if r["e_form_eV"] < 2.0 else "high cost"
    if r["e_form_eV"] < 0:
        fav = "spontaneous"
    print(f"{r['label']:<15} {r['e_form_eV']:<15.4f} {fav}")

# Bar chart
fig, ax = plt.subplots(figsize=(8, 5))
labels = [r["label"] for r in sorted(results, key=lambda x: x["e_form_eV"])]
e_forms = [r["e_form_eV"] for r in sorted(results, key=lambda x: x["e_form_eV"])]
colors = ["green" if e < 0 else "steelblue" if e < 2 else "firebrick" for e in e_forms]

bars = ax.bar(labels, e_forms, color=colors, edgecolor="black", alpha=0.8)
ax.axhline(y=0, color="black", linewidth=0.8)
ax.set_ylabel("Formation energy (eV)", fontsize=12)
ax.set_title(f"Substitutional defects in {host.composition.reduced_formula}", fontsize=13)
ax.set_xlabel("Defect type", fontsize=12)
plt.xticks(rotation=45, ha="right")
ax.grid(axis="y", alpha=0.3)

for bar, val in zip(bars, e_forms):
    ypos = bar.get_height() + 0.05 if val >= 0 else bar.get_height() - 0.15
    ax.text(bar.get_x() + bar.get_width()/2, ypos,
            f"{val:.2f}", ha="center", va="bottom", fontsize=9)

fig.tight_layout()
fig.savefig("substitution_formation_energies.png", dpi=150)
print("\nPlot saved to substitution_formation_energies.png")

with open("substitution_results.json", "w") as f:
    json.dump(results, f, indent=2, default=str)
print("Results saved to substitution_results.json")
```

#### Site Preference Analysis

```python
#!/usr/bin/env python3
"""
Determine which sublattice a dopant prefers.
Example: Does Fe prefer the Mg site or the O site in MgO?
Uses pymatgen's SubstitutionGenerator for systematic enumeration.
"""

import numpy as np
from pymatgen.core import Structure, Element
from pymatgen.analysis.defects.generators import SubstitutionGenerator
from pymatgen.io.ase import AseAtomsAdaptor
from ase.optimize import BFGS
from mace.calculators import mace_mp
import json

calc = mace_mp(model="medium", dispersion=False, default_dtype="float64")
adaptor = AseAtomsAdaptor()

# Host: SrTiO3 perovskite (has 3 inequivalent sites)
host = Structure.from_spacegroup(
    "Pm-3m",
    lattice=[[3.905, 0, 0], [0, 3.905, 0], [0, 0, 3.905]],
    species=["Sr", "Ti", "O"],
    coords=[[0.0, 0.0, 0.0], [0.5, 0.5, 0.5], [0.5, 0.5, 0.0]],
)
print(f"Host: {host.formula}")

# Generate all substitution defects for a given dopant
DOPANT = "Fe"
sub_gen = SubstitutionGenerator()
subs = list(sub_gen.generate(host, {DOPANT: Element(DOPANT)}))
print(f"Found {len(subs)} symmetry-inequivalent substitution sites for {DOPANT}")

sc_mat = np.eye(3, dtype=int) * 3

# Relax bulk supercell
bulk_sc = host.copy()
bulk_sc.make_supercell([3, 3, 3])
atoms_bulk = adaptor.get_atoms(bulk_sc)
atoms_bulk.calc = calc
BFGS(atoms_bulk, logfile=None).run(fmax=0.005, steps=300)
e_bulk = atoms_bulk.get_potential_energy()
n_bulk = len(atoms_bulk)

results = []
for i, sub_defect in enumerate(subs):
    replaced_sp = str(sub_defect.site.specie)
    print(f"\n--- {DOPANT} on {replaced_sp} site (index {i}) ---")

    defect_sc = sub_defect.get_supercell_structure(sc_mat=sc_mat)
    atoms_defect = adaptor.get_atoms(defect_sc)
    atoms_defect.calc = mace_mp(model="medium", dispersion=False, default_dtype="float64")
    BFGS(atoms_defect, logfile=None).run(fmax=0.005, steps=500)
    e_defect = atoms_defect.get_potential_energy()

    # Simple formation energy (relative, for site comparison)
    e_form_relative = e_defect - e_bulk
    print(f"  E_defect - E_bulk = {e_form_relative:.4f} eV")

    results.append({
        "defect_index": i,
        "dopant": DOPANT,
        "replaced": replaced_sp,
        "label": f"{DOPANT}_{replaced_sp}",
        "e_relative_eV": e_form_relative,
        "site_frac_coords": sub_defect.site.frac_coords.tolist(),
    })

# Rank by energy
results_sorted = sorted(results, key=lambda x: x["e_relative_eV"])
print(f"\n=== Site Preference for {DOPANT} in {host.composition.reduced_formula} ===")
for r in results_sorted:
    delta = r["e_relative_eV"] - results_sorted[0]["e_relative_eV"]
    pref = " <-- preferred" if delta < 0.01 else ""
    print(f"  {r['label']}: E_rel = {r['e_relative_eV']:.4f} eV "
          f"(+{delta:.4f} eV vs best){pref}")

with open("site_preference_results.json", "w") as f:
    json.dump(results_sorted, f, indent=2, default=str)
print("\nSaved to site_preference_results.json")
```

### Method B: QE DFT (Neutral and Charged Substitutions)

#### Step 1: Generate QE Input Files

```python
#!/usr/bin/env python3
"""
Generate QE input files for substitutional defect calculations.
Creates inputs for bulk, neutral substitution, and charged substitutions.

Example: Al substituting Mg in MgO (Al_Mg).
Al_Mg can be neutral (Al_Mg^0) or charged (Al_Mg^{1+} as a donor).
"""

import numpy as np
from pymatgen.core import Structure, Element
from pymatgen.io.pwscf import PWInput
from pathlib import Path

# ============================================================
# Configuration
# ============================================================
PSEUDO_DIR = "./pseudo"
ECUTWFC = 60.0
ECUTRHO = 480.0
SC_SIZE = 3

DOPANT = "Al"
REPLACED = "Mg"
CHARGE_STATES = [0, +1]  # Al_Mg^0 and Al_Mg^{1+}

pseudos = {
    "Mg": "Mg.pbe-spnl-kjpaw_psl.1.0.0.UPF",
    "O":  "O.pbe-n-kjpaw_psl.1.0.0.UPF",
    "Al": "Al.pbe-n-kjpaw_psl.1.0.0.UPF",
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

bulk_sc = primitive.copy()
bulk_sc.make_supercell([SC_SIZE, SC_SIZE, SC_SIZE])

# Substitution supercell
sub_sc = bulk_sc.copy()
target_indices = [i for i, s in enumerate(sub_sc) if str(s.specie) == REPLACED]
sub_sc.replace(target_indices[0], Element(DOPANT))

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
# 1. Bulk supercell
# ============================================================
pw_bulk = PWInput(
    bulk_sc,
    pseudo=pseudos,
    control=control | {"prefix": "bulk", "outdir": "./tmp_bulk"},
    system=system_params,
    electrons=electrons,
    kpoints_grid=kpts,
)
pw_bulk.write_file("pw_bulk.in")
print(f"Written pw_bulk.in ({len(bulk_sc)} atoms)")

# ============================================================
# 2. Substitution (each charge state)
# ============================================================
for q in CHARGE_STATES:
    qstr = f"q{q:+d}" if q != 0 else "q0"

    system_q = system_params.copy()
    if q != 0:
        system_q["tot_charge"] = float(q)

    pw_sub = PWInput(
        sub_sc,
        pseudo=pseudos,
        control=control | {
            "prefix": f"sub_{qstr}",
            "outdir": f"./tmp_sub_{qstr}",
        },
        system=system_q,
        electrons=electrons | ({"mixing_beta": 0.2} if q != 0 else {}),
        kpoints_grid=kpts,
    )
    pw_sub.write_file(f"pw_sub_{qstr}.in")
    print(f"Written pw_sub_{qstr}.in ({len(sub_sc)} atoms, q={q:+d})")

# ============================================================
# 3. Elemental reference: dopant bulk
# ============================================================
# Al fcc reference
al_fcc = Structure.from_spacegroup(
    "Fm-3m",
    lattice=[[4.046, 0, 0], [0, 4.046, 0], [0, 0, 4.046]],
    species=["Al"],
    coords=[[0.0, 0.0, 0.0]],
)
pw_al = PWInput(
    al_fcc,
    pseudo=pseudos,
    control=control | {"prefix": "Al_bulk", "outdir": "./tmp_Al_bulk"},
    system=system_params,
    electrons=electrons,
    kpoints_grid=(8, 8, 8),  # dense k-mesh for small cell
)
pw_al.write_file("pw_Al_bulk.in")
print(f"Written pw_Al_bulk.in (Al fcc, {len(al_fcc)} atoms)")

print("\nAll QE inputs generated for substitutional defect calculation.")
```

#### Step 2: Run QE Calculations

```bash
#!/bin/bash
# Run substitutional defect QE calculations
NPROC=4

# Download pseudopotentials
mkdir -p pseudo tmp_bulk tmp_sub_q0 tmp_sub_q+1 tmp_Al_bulk
cd pseudo
wget -nc "https://pseudopotentials.quantum-espresso.org/upf_files/Mg.pbe-spnl-kjpaw_psl.1.0.0.UPF" 2>/dev/null
wget -nc "https://pseudopotentials.quantum-espresso.org/upf_files/O.pbe-n-kjpaw_psl.1.0.0.UPF" 2>/dev/null
wget -nc "https://pseudopotentials.quantum-espresso.org/upf_files/Al.pbe-n-kjpaw_psl.1.0.0.UPF" 2>/dev/null
cd ..

# 1. Bulk
echo "=== Bulk supercell ==="
mpirun -np $NPROC pw.x -in pw_bulk.in > pw_bulk.out 2>&1
echo "Bulk: $(grep '!' pw_bulk.out | tail -1)"

# 2. Al bulk reference
echo "=== Al bulk reference ==="
mpirun -np $NPROC pw.x -in pw_Al_bulk.in > pw_Al_bulk.out 2>&1
echo "Al bulk: $(grep '!' pw_Al_bulk.out | tail -1)"

# 3. Neutral substitution
echo "=== Neutral substitution ==="
mpirun -np $NPROC pw.x -in pw_sub_q0.in > pw_sub_q0.out 2>&1
echo "Sub q=0: $(grep '!' pw_sub_q0.out | tail -1)"

# 4. Charged substitution
echo "=== Charged substitution q=+1 ==="
mpirun -np $NPROC pw.x -in pw_sub_q+1.in > pw_sub_q+1.out 2>&1
echo "Sub q=+1: $(grep '!' pw_sub_q+1.out | tail -1)"
```

#### Step 3: Compute Substitution Formation Energy

```python
#!/usr/bin/env python3
"""
Parse QE outputs and compute substitutional defect formation energy.

Formation energy:
  E_f(A_B^q) = E_defect(q) - E_bulk + mu_B - mu_A + q*(E_VBM + E_Fermi) + E_corr

where:
  A = dopant (added), B = replaced atom (removed)
  mu_A, mu_B = chemical potentials of the dopant and replaced species
  q = charge state
  E_VBM = valence band maximum
  E_Fermi = Fermi level referenced to VBM
  E_corr = Freysoldt correction for charged defects
"""

import re
import numpy as np
import json

def parse_qe_energy(filename):
    """Extract final total energy in eV from QE output."""
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
            if "highest occupied" in line:
                match = re.search(r"([-\d.]+)\s+eV", line)
                if match:
                    vbm = float(match.group(1))
            elif "highest occupied, lowest unoccupied" in line:
                vals = re.findall(r"[-\d.]+", line)
                if vals:
                    vbm = float(vals[0])
    return vbm

# ============================================================
# Parse energies
# ============================================================
e_bulk = parse_qe_energy("pw_bulk.out")
e_vbm = parse_qe_vbm("pw_bulk.out")
n_bulk = parse_qe_natoms("pw_bulk.out")

e_al_ref = parse_qe_energy("pw_Al_bulk.out")
n_al_ref = parse_qe_natoms("pw_Al_bulk.out")
mu_Al = e_al_ref / n_al_ref  # per-atom energy of Al bulk

# Mg reference from bulk MgO per-atom (simplified)
# For accurate results, use the elemental Mg bulk energy
mu_Mg = e_bulk / n_bulk  # approximate; replace with proper Mg metal reference

print(f"E_bulk = {e_bulk:.6f} eV ({n_bulk} atoms)")
print(f"mu_Al  = {mu_Al:.6f} eV/atom (from Al fcc)")
print(f"mu_Mg  = {mu_Mg:.6f} eV/atom (bulk ref, approximate)")
if e_vbm:
    print(f"E_VBM  = {e_vbm:.4f} eV")

# ============================================================
# Compute formation energies
# ============================================================
charge_states = [0, 1]
dielectric = 9.8       # MgO static dielectric constant
L_sc = 4.212 * 3       # supercell lattice parameter (A)

results = {}
for q in charge_states:
    qstr = f"q{q:+d}" if q != 0 else "q0"
    e_defect = parse_qe_energy(f"pw_sub_{qstr}.out")

    if e_defect is None:
        print(f"\nWARNING: Could not parse pw_sub_{qstr}.out")
        continue

    # E_f = E_defect - E_bulk + mu_replaced - mu_dopant + q*(E_VBM + E_F) + E_corr
    # At E_Fermi = 0 (VBM):
    e_form = e_defect - e_bulk + mu_Mg - mu_Al

    e_corr = 0.0
    if q != 0:
        alpha_M = 2.8373
        e2 = 14.3996
        e_corr = (alpha_M * q**2 * e2) / (2.0 * dielectric * L_sc)
        e_form += q * e_vbm + e_corr if e_vbm else e_corr

    print(f"\n--- Al_Mg^{{{q:+d}}} ---")
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

with open("substitution_qe_results.json", "w") as f:
    json.dump(results, f, indent=2, default=float)
print("\nResults saved to substitution_qe_results.json")
```

### Method C: VASP DFT (Neutral and Charged Substitutions)

#### Generate VASP Input Files

```python
#!/usr/bin/env python3
"""
Generate VASP input files for substitutional defect calculations.

Note: VASP execution will be available via future external access.
This script generates the input files.
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
DOPANT = "Al"
REPLACED = "Mg"
CHARGE_STATES = [0, +1, -1]

bulk_sc = primitive.copy()
bulk_sc.make_supercell([SC_SIZE, SC_SIZE, SC_SIZE])

sub_sc = bulk_sc.copy()
target = [i for i, s in enumerate(sub_sc) if str(s.specie) == REPLACED]
sub_sc.replace(target[0], Element(DOPANT))

# ============================================================
# 1. Bulk supercell
# ============================================================
bulk_dir = Path("vasp_bulk")
bulk_dir.mkdir(exist_ok=True)

Poscar(bulk_sc).write_file(str(bulk_dir / "POSCAR"))

incar_bulk = Incar({
    "SYSTEM": f"{primitive.composition.reduced_formula} bulk",
    "ENCUT": 520,
    "PREC": "Accurate",
    "EDIFF": 1e-6,
    "EDIFFG": -0.01,
    "IBRION": 2,
    "ISIF": 2,
    "NSW": 200,
    "ISMEAR": 0,
    "SIGMA": 0.05,
    "LREAL": "Auto",
    "LWAVE": False,
    "LCHARG": True,
    "LORBIT": 11,
    "ALGO": "Normal",
})
incar_bulk.write_file(str(bulk_dir / "INCAR"))
Kpoints.gamma_automatic(kpts=(2, 2, 2)).write_file(str(bulk_dir / "KPOINTS"))
print(f"Bulk: {bulk_dir}/ ({len(bulk_sc)} atoms)")

# ============================================================
# 2. Substitution supercells
# ============================================================
for q in CHARGE_STATES:
    qstr = f"q{q:+d}" if q != 0 else "q0"
    sub_dir = Path(f"vasp_sub_{qstr}")
    sub_dir.mkdir(exist_ok=True)

    Poscar(sub_sc).write_file(str(sub_dir / "POSCAR"))

    incar_sub = Incar({
        "SYSTEM": f"{DOPANT}_{REPLACED}^{{{q:+d}}} in {primitive.composition.reduced_formula}",
        "ENCUT": 520,
        "PREC": "Accurate",
        "EDIFF": 1e-6,
        "EDIFFG": -0.01,
        "IBRION": 2,
        "ISIF": 2,
        "NSW": 300,
        "ISMEAR": 0,
        "SIGMA": 0.05,
        "LREAL": "Auto",
        "LWAVE": False,
        "LCHARG": True,
        "LVHAR": True,     # for Freysoldt correction
        "LORBIT": 11,
        "ALGO": "Normal",
        "NELM": 300,
    })

    if q != 0:
        incar_sub["NELECT"] = "REPLACE_WITH_ZVAL_SUM_MINUS_Q"
        if q % 2 != 0:
            incar_sub["ISPIN"] = 2

    incar_sub.write_file(str(sub_dir / "INCAR"))
    Kpoints.gamma_automatic(kpts=(2, 2, 2)).write_file(str(sub_dir / "KPOINTS"))
    print(f"{DOPANT}_{REPLACED}^{{{q:+d}}}: {sub_dir}/ ({len(sub_sc)} atoms)")

# ============================================================
# 3. Elemental references
# ============================================================
# Al fcc reference
al_dir = Path("vasp_Al_ref")
al_dir.mkdir(exist_ok=True)

al_fcc = Structure.from_spacegroup(
    "Fm-3m",
    lattice=[[4.046, 0, 0], [0, 4.046, 0], [0, 0, 4.046]],
    species=["Al"],
    coords=[[0.0, 0.0, 0.0]],
)
Poscar(al_fcc).write_file(str(al_dir / "POSCAR"))

incar_al = Incar({
    "SYSTEM": "Al fcc reference",
    "ENCUT": 520,
    "PREC": "Accurate",
    "EDIFF": 1e-6,
    "EDIFFG": -0.01,
    "IBRION": 2,
    "ISIF": 3,       # relax cell for reference
    "NSW": 100,
    "ISMEAR": 1,
    "SIGMA": 0.1,
    "LWAVE": False,
    "LCHARG": False,
})
incar_al.write_file(str(al_dir / "INCAR"))
Kpoints.gamma_automatic(kpts=(12, 12, 12)).write_file(str(al_dir / "KPOINTS"))
print(f"Al reference: {al_dir}/")

# Mg hcp reference
mg_dir = Path("vasp_Mg_ref")
mg_dir.mkdir(exist_ok=True)

from pymatgen.core import Lattice
mg_hcp = Structure.from_spacegroup(
    "P6_3/mmc",
    lattice=Lattice.hexagonal(3.209, 5.211),
    species=["Mg", "Mg"],
    coords=[[1/3, 2/3, 0.25], [2/3, 1/3, 0.75]],
)
Poscar(mg_hcp).write_file(str(mg_dir / "POSCAR"))

incar_mg = Incar({
    "SYSTEM": "Mg hcp reference",
    "ENCUT": 520,
    "PREC": "Accurate",
    "EDIFF": 1e-6,
    "EDIFFG": -0.01,
    "IBRION": 2,
    "ISIF": 3,
    "NSW": 100,
    "ISMEAR": 1,
    "SIGMA": 0.1,
    "LWAVE": False,
    "LCHARG": False,
})
incar_mg.write_file(str(mg_dir / "INCAR"))
Kpoints.gamma_automatic(kpts=(12, 12, 8)).write_file(str(mg_dir / "KPOINTS"))
print(f"Mg reference: {mg_dir}/")

print("""
=== VASP Substitutional Defect Workflow ===
1. Run bulk supercell: cd vasp_bulk && vasp_std
2. Run elemental references: cd vasp_Al_ref && vasp_std; cd vasp_Mg_ref && vasp_std
3. Determine NELECT from neutral run, adjust for charged defects
4. Run defect supercells
5. Parse with pymatgen Vasprun, apply Freysoldt correction for charged states
""")
```

## Key Parameters

| Parameter | Recommended Value | Notes |
|---|---|---|
| Supercell size | >= 3x3x3 or min 10 A between dopant images | Critical for dilute limit approximation |
| MACE fmax | 0.005 eV/A | Tight for accurate energy differences |
| QE ecutwfc | 50-80 Ry | Must accommodate both host and dopant elements |
| QE k-points | Gamma or 2x2x2 for large supercells | Converge independently |
| VASP ENCUT | 520 eV or 1.3x max ENMAX | Use consistent cutoff across all calculations |
| VASP EDIFF | 1e-6 eV | Tight electronic convergence |
| Chemical potential refs | Elemental phases (bulk metals, O2/N2 gas) | Must match the DFT functional and pseudopotentials used |
| Dielectric constant | Material-dependent | Required for charged defect Freysoldt correction |
| Spin polarization | Enable for transition metal dopants or odd-electron systems | nspin=2 (QE) or ISPIN=2 (VASP) |

## Interpreting Results

1. **Formation energy and solubility**: Lower (more negative) formation energy means easier incorporation. High formation energy (>3-4 eV) typically indicates very low dopant solubility.

2. **Chemical potential dependence**: The substitution energy depends on growth conditions. For A_B in a host: E_f depends on mu_A (higher under A-rich conditions means lower E_f) and mu_B (higher under B-poor conditions means lower E_f).

3. **Site preference**: When a dopant can substitute on multiple sublattices, the site with the lowest formation energy is preferred. The energy difference determines the site selectivity (>0.5 eV typically means strong preference).

4. **Donor vs. acceptor behavior**: A dopant that prefers a positive charge state acts as a donor (releases electrons). A dopant preferring negative charge is an acceptor (captures electrons). The charge transition level epsilon(q1/q2) determines where in the band gap the transition occurs.

5. **Compensation**: In a material with both donors and acceptors, the Fermi level adjusts to minimize total energy. High donor concentration can be compensated by native acceptor defects.

6. **MACE vs. DFT**: MACE is reliable for neutral substitution trends within well-represented chemistries. For charged states, electronic properties, or unusual dopant/host combinations, DFT is required.

## Common Issues

| Issue | Solution |
|---|---|
| SCF not converging with dopant | Reduce mixing_beta; use smearing; check pseudopotential compatibility |
| Dopant atom relaxes far from original site | May indicate the dopant does not fit at this site; try a larger supercell or different site |
| Formation energy depends strongly on supercell size | Increase supercell; for charged defects apply Freysoldt correction |
| Wrong magnetic state for transition metal dopant | Set nspin=2 (QE) or ISPIN=2 (VASP); try different starting_magnetization |
| Chemical potential reference gives negative formation energy | Check reference phase; may indicate phase decomposition is favorable |
| MACE gives wrong site preference | Validate with DFT; MACE may not capture subtle charge transfer effects |
| Dopant pseudopotential not in SSSP library | Download from QE pseudopotential library; verify cutoff compatibility |
| VASP: wrong NELECT for charged defect | Run neutral first, grep NELECT from OUTCAR, then NELECT = default - q |
| Large lattice distortion around dopant | Expected for size-mismatched dopants; ensure supercell is large enough to contain distortion |
