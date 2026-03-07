# Adsorption Energy Calculation

## When to Use

- Calculate the adsorption energy of a molecule or atom on a solid surface
- Screen adsorption sites (top, bridge, hollow) to find the preferred binding geometry
- Study coverage-dependent adsorption behavior (low coverage vs. monolayer)
- Include van der Waals (DFT-D3) corrections for physisorbed or weakly chemisorbed systems
- Compare adsorption strengths across different surface facets or materials
- Prepare initial data for reaction pathway or free energy diagram construction

## Method Selection

| Criterion | MACE (Method A) | QE DFT (Method B) | VASP (Method C) |
|---|---|---|---|
| Speed | Minutes per site | Hours per site | Hours per site |
| Accuracy | Good for trends | Publication quality | Publication quality |
| Dispersion (vdW) | `dispersion=True` in MACE | `vdw_corr='dft-d3'` | `IVDW=11` (DFT-D3) |
| Coverage study | Fast parametric sweep | Expensive but exact | Expensive but exact |
| Best for | Rapid site screening, coverage trends | Quantitative E_ads, charge analysis | VASP-ecosystem workflows |
| Limitations | ML potential coverage varies | Costly for large supercells | Requires VASP license |

**Recommended workflow**: Screen all adsorption sites with MACE (Method A), identify the 2-3 most favorable sites, then refine with QE DFT (Method B) or VASP (Method C) including dispersion corrections.

## Prerequisites

- pymatgen (SlabGenerator, AdsorbateSiteFinder, Molecule)
- ASE + mace-torch (Method A)
- Quantum ESPRESSO pw.x (Method B)
- SSSP pseudopotentials for QE
- numpy, matplotlib for analysis

## Detailed Steps

### Adsorption Energy Formula

```
E_ads = E(slab + adsorbate) - E(clean slab) - E(gas-phase molecule)
```

- `E(slab + adsorbate)` = relaxed total energy of the adsorbate-on-slab system
- `E(clean slab)` = relaxed total energy of the clean slab (same supercell and k-points)
- `E(gas-phase molecule)` = relaxed total energy of the isolated molecule in a large box

Negative E_ads means adsorption is exothermic (favorable). More negative = stronger binding.

### Method A: MACE -- Multi-Site Screening with Dispersion

#### Complete Workflow: CO2 Adsorption on MgO(100)

```python
#!/usr/bin/env python3
"""
Adsorption energy calculation: CO2 on MgO(100) surface.
Screens top, bridge, and hollow sites using ASE + MACE.
Includes DFT-D3 dispersion corrections via MACE's built-in dispersion.

E_ads = E(slab+CO2) - E(slab) - E(CO2_gas)
"""

import numpy as np
import warnings
warnings.filterwarnings("ignore")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from pymatgen.core import Structure, Lattice, Molecule
from pymatgen.core.surface import SlabGenerator
from pymatgen.analysis.adsorption import AdsorbateSiteFinder
from pymatgen.io.ase import AseAtomsAdaptor
from ase.optimize import BFGS
from ase.filters import FrechetCellFilter
from ase.constraints import FixAtoms
from ase.build import molecule as ase_molecule
from mace.calculators import mace_mp
import json

# ============================================================
# 1. Configuration
# ============================================================
USE_DISPERSION = True  # Enable D3 dispersion in MACE

def get_calc():
    """Return a fresh MACE calculator instance."""
    return mace_mp(
        model="medium",
        dispersion=USE_DISPERSION,
        default_dtype="float64",
    )

adaptor = AseAtomsAdaptor()

# ============================================================
# 2. Relax bulk MgO
# ============================================================
mgo_bulk = Structure.from_spacegroup(
    "Fm-3m",
    lattice=Lattice.cubic(4.212),
    species=["Mg", "O"],
    coords=[[0.0, 0.0, 0.0], [0.5, 0.5, 0.5]],
)

atoms_bulk = adaptor.get_atoms(mgo_bulk)
atoms_bulk.calc = get_calc()
ecf = FrechetCellFilter(atoms_bulk)
opt = BFGS(ecf, logfile="relax_bulk.log")
opt.run(fmax=0.003, steps=300)
mgo_relaxed = adaptor.get_structure(atoms_bulk)
print(f"Relaxed MgO: a = {mgo_relaxed.lattice.a:.4f} A")

# ============================================================
# 3. Generate MgO(100) slab
# ============================================================
slabgen = SlabGenerator(
    initial_structure=mgo_relaxed,
    miller_index=(1, 0, 0),
    min_slab_size=10.0,
    min_vacuum_size=18.0,
    center_slab=True,
    in_unit_planes=False,
    lll_reduce=True,
    primitive=True,
)

slabs = slabgen.get_slabs(symmetrize=False)
slab = slabs[0]
slab.make_supercell([2, 2, 1])
print(f"Slab: {slab.formula}, {len(slab)} atoms")

# Relax clean slab (fix bottom half)
atoms_slab = adaptor.get_atoms(slab)
atoms_slab.calc = get_calc()

z = atoms_slab.get_positions()[:, 2]
z_median = np.median(z)
fix_mask = z < z_median
n_fixed = int(np.sum(fix_mask))
n_free = len(atoms_slab) - n_fixed
print(f"Fixed {n_fixed} bottom atoms, {n_free} free atoms")

atoms_slab.set_constraint(FixAtoms(mask=fix_mask))
opt_slab = BFGS(atoms_slab, logfile="relax_slab_clean.log")
opt_slab.run(fmax=0.01, steps=300)
e_slab = atoms_slab.get_potential_energy()
print(f"E_slab (clean) = {e_slab:.6f} eV")

slab_relaxed = adaptor.get_structure(atoms_slab)

# ============================================================
# 4. Relax isolated CO2 molecule in vacuum box
# ============================================================
co2 = ase_molecule("CO2")
co2.center(vacuum=10.0)
co2.calc = get_calc()
opt_mol = BFGS(co2, logfile="relax_CO2.log")
opt_mol.run(fmax=0.003, steps=100)
e_mol = co2.get_potential_energy()
print(f"\nE_CO2 (gas) = {e_mol:.6f} eV")

# ============================================================
# 5. Find adsorption sites
# ============================================================
asf = AdsorbateSiteFinder(slab_relaxed)
ads_sites = asf.find_adsorption_sites(
    distance=2.5,
    symm_reduce=0.01,
    near_reduce=0.5,
    no_obtuse_hollow=True,
)

print(f"\nAdsorption sites found:")
for site_type in ["ontop", "bridge", "hollow"]:
    n_sites = len(ads_sites.get(site_type, []))
    print(f"  {site_type}: {n_sites} sites")

# ============================================================
# 6. Place CO2 and relax at each site type
# ============================================================
# CO2 molecule: O=C=O (linear), place C at the site, O-C-O along x
co2_pmg = Molecule(
    ["C", "O", "O"],
    [[0, 0, 0], [0, 0, 1.16], [0, 0, -1.16]],
)

results = {}

for site_type in ["ontop", "bridge", "hollow"]:
    sites = ads_sites.get(site_type, [])
    if not sites:
        continue

    print(f"\n--- {site_type} site ---")

    # Generate adslab structures
    adslabs = asf.generate_adsorption_structures(
        co2_pmg,
        repeat=[1, 1, 1],
        find_args={"distance": 2.5},
    )

    if not adslabs:
        print(f"  No adslab generated for {site_type}")
        continue

    # Use the structure corresponding to this site type
    # pymatgen generates one per unique site; we take the first available
    adslab = adslabs[0] if site_type == "ontop" else (
        adslabs[1] if len(adslabs) > 1 and site_type == "bridge" else (
            adslabs[min(2, len(adslabs) - 1)] if site_type == "hollow" else adslabs[0]
        )
    )

    atoms_ads = adaptor.get_atoms(adslab)
    atoms_ads.calc = get_calc()

    # Fix bottom atoms
    pos = atoms_ads.get_positions()
    z_a = pos[:, 2]
    z_med = np.median(z_a)
    fix_a = z_a < z_med
    atoms_ads.set_constraint(FixAtoms(mask=fix_a))

    opt_a = BFGS(atoms_ads, logfile=f"relax_adslab_{site_type}.log")
    opt_a.run(fmax=0.01, steps=500)

    e_adslab = atoms_ads.get_potential_energy()
    e_ads = e_adslab - e_slab - e_mol

    print(f"  E_adslab = {e_adslab:.6f} eV")
    print(f"  E_ads = {e_ads:.4f} eV")

    # Find adsorbate-surface distance (min distance from C to slab atoms)
    relaxed_adslab = adaptor.get_structure(atoms_ads)
    n_slab_atoms = len(slab_relaxed)
    ads_z = min(pos[n_slab_atoms:, 2])  # lowest adsorbate atom z
    surf_z = max(pos[:n_slab_atoms, 2])  # highest slab atom z
    d_ads_surf = ads_z - surf_z

    results[site_type] = {
        "e_adslab_eV": float(e_adslab),
        "e_ads_eV": float(e_ads),
        "ads_surface_distance_A": float(d_ads_surf),
    }

# ============================================================
# 7. Summary
# ============================================================
print("\n" + "=" * 55)
print("ADSORPTION ENERGY SUMMARY: CO2 on MgO(100)")
print("=" * 55)
print(f"E_slab (clean) = {e_slab:.6f} eV")
print(f"E_CO2 (gas)    = {e_mol:.6f} eV")
print(f"Dispersion:      {'D3 ON' if USE_DISPERSION else 'OFF'}")
print(f"\n{'Site':<10} {'E_ads (eV)':<14} {'d_surf (A)':<12} {'Type'}")
print("-" * 50)
for st in sorted(results, key=lambda x: results[x]["e_ads_eV"]):
    d = results[st]
    bind = "chemisorbed" if d["e_ads_eV"] < -0.5 else (
        "physisorbed" if d["e_ads_eV"] < -0.05 else "not bound")
    print(f"{st:<10} {d['e_ads_eV']:<14.4f} {d['ads_surface_distance_A']:<12.3f} {bind}")

# Plot
if results:
    fig, ax = plt.subplots(figsize=(7, 5))
    names = list(results.keys())
    evals = [results[n]["e_ads_eV"] for n in names]
    colors = ["green" if e < -0.5 else ("blue" if e < -0.05 else "red") for e in evals]

    bars = ax.bar(names, evals, color=colors, edgecolor="black", alpha=0.7)
    ax.axhline(y=0, color="black", linewidth=0.8)
    ax.set_ylabel("Adsorption energy (eV)", fontsize=12)
    ax.set_title("CO2 on MgO(100) -- Site Comparison", fontsize=13)
    ax.grid(axis="y", alpha=0.3)

    for bar, val in zip(bars, evals):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() - 0.05 if val < 0 else bar.get_height() + 0.02,
                f"{val:.3f}", ha="center",
                va="top" if val < 0 else "bottom", fontsize=10)

    fig.tight_layout()
    fig.savefig("adsorption_energy_sites.png", dpi=150)
    print("\nPlot saved to adsorption_energy_sites.png")

output = {
    "system": "CO2 on MgO(100)",
    "dispersion": USE_DISPERSION,
    "e_slab_eV": float(e_slab),
    "e_molecule_eV": float(e_mol),
    "sites": results,
}
with open("adsorption_results.json", "w") as f:
    json.dump(output, f, indent=2)
print("Results saved to adsorption_results.json")
```

#### Coverage Dependence Study

```python
#!/usr/bin/env python3
"""
Coverage-dependent adsorption energy using MACE.
Vary the surface supercell size (coverage = 1/N^2 ML) and track E_ads vs. coverage.

Example: CO on Pt(111) at various coverages.
"""

import numpy as np
import warnings
warnings.filterwarnings("ignore")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from pymatgen.core import Structure, Lattice, Molecule
from pymatgen.core.surface import SlabGenerator
from pymatgen.analysis.adsorption import AdsorbateSiteFinder
from pymatgen.io.ase import AseAtomsAdaptor
from ase.optimize import BFGS
from ase.filters import FrechetCellFilter
from ase.constraints import FixAtoms
from ase.build import molecule as ase_molecule
from mace.calculators import mace_mp
import json

adaptor = AseAtomsAdaptor()

def get_calc():
    return mace_mp(model="medium", dispersion=True, default_dtype="float64")

# Relax bulk Pt (fcc)
pt_bulk = Structure.from_spacegroup(
    "Fm-3m",
    lattice=Lattice.cubic(3.924),
    species=["Pt"],
    coords=[[0.0, 0.0, 0.0]],
)
atoms_pt = adaptor.get_atoms(pt_bulk)
atoms_pt.calc = get_calc()
ecf = FrechetCellFilter(atoms_pt)
BFGS(ecf, logfile=None).run(fmax=0.003, steps=200)
pt_relaxed = adaptor.get_structure(atoms_pt)

# Relax CO molecule
co = ase_molecule("CO")
co.center(vacuum=10.0)
co.calc = get_calc()
BFGS(co, logfile=None).run(fmax=0.003, steps=100)
e_co = co.get_potential_energy()
print(f"E_CO (gas) = {e_co:.6f} eV")

# Coverage study: 1x1 (1 ML), 2x2 (1/4 ML), 3x3 (1/9 ML), 4x4 (1/16 ML)
supercell_sizes = [1, 2, 3, 4]
coverage_results = []

co_pmg = Molecule(["C", "O"], [[0, 0, 0], [0, 0, 1.128]])

for n_sc in supercell_sizes:
    coverage = 1.0 / n_sc**2  # one adsorbate per n_sc x n_sc surface unit cell
    print(f"\n=== Coverage = 1/{n_sc**2} ML ({coverage:.4f} ML) ===")

    # Generate slab
    slabgen = SlabGenerator(
        pt_relaxed, (1, 1, 1),
        min_slab_size=10.0,
        min_vacuum_size=18.0,
        center_slab=True,
        primitive=True,
    )
    slab = slabgen.get_slabs()[0]
    slab.make_supercell([n_sc, n_sc, 1])
    n_slab = len(slab)

    # Relax clean slab
    atoms_slab = adaptor.get_atoms(slab)
    atoms_slab.calc = get_calc()
    z = atoms_slab.get_positions()[:, 2]
    fix = z < np.median(z)
    atoms_slab.set_constraint(FixAtoms(mask=fix))
    BFGS(atoms_slab, logfile=None).run(fmax=0.01, steps=300)
    e_slab = atoms_slab.get_potential_energy()

    # Add CO at top site (above surface atom)
    slab_relaxed = adaptor.get_structure(atoms_slab)
    asf = AdsorbateSiteFinder(slab_relaxed)
    adslabs = asf.generate_adsorption_structures(
        co_pmg,
        repeat=[1, 1, 1],
        find_args={"distance": 1.9},
    )

    if not adslabs:
        print(f"  No adslab generated for {n_sc}x{n_sc}")
        continue

    adslab = adslabs[0]
    atoms_ads = adaptor.get_atoms(adslab)
    atoms_ads.calc = get_calc()
    z_a = atoms_ads.get_positions()[:, 2]
    fix_a = z_a < np.median(z_a[:n_slab])
    fix_a[n_slab:] = False  # do not fix adsorbate
    atoms_ads.set_constraint(FixAtoms(mask=fix_a))
    BFGS(atoms_ads, logfile=None).run(fmax=0.01, steps=500)

    e_adslab = atoms_ads.get_potential_energy()
    e_ads = e_adslab - e_slab - e_co

    area = np.linalg.norm(np.cross(
        slab.lattice.matrix[0], slab.lattice.matrix[1]
    ))

    coverage_results.append({
        "supercell": f"{n_sc}x{n_sc}",
        "coverage_ML": float(coverage),
        "n_slab_atoms": n_slab,
        "area_A2": float(area),
        "e_ads_eV": float(e_ads),
    })
    print(f"  E_ads = {e_ads:.4f} eV, area = {area:.1f} A^2")

# Plot coverage dependence
if coverage_results:
    fig, ax = plt.subplots(figsize=(7, 5))
    coverages = [r["coverage_ML"] for r in coverage_results]
    eads = [r["e_ads_eV"] for r in coverage_results]
    sc_labels = [r["supercell"] for r in coverage_results]

    ax.plot(coverages, eads, "o-", color="steelblue", markersize=8, linewidth=2)
    for c, e, lab in zip(coverages, eads, sc_labels):
        ax.annotate(lab, (c, e), textcoords="offset points",
                    xytext=(10, 5), fontsize=9)

    ax.set_xlabel("Coverage (ML)", fontsize=12)
    ax.set_ylabel("E_ads (eV)", fontsize=12)
    ax.set_title("CO/Pt(111): Coverage-Dependent Adsorption Energy", fontsize=13)
    ax.invert_xaxis()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig("adsorption_coverage.png", dpi=150)
    print("\nPlot saved to adsorption_coverage.png")

with open("adsorption_coverage_results.json", "w") as f:
    json.dump(coverage_results, f, indent=2)
print("Results saved to adsorption_coverage_results.json")
```

### Method B: QE DFT with DFT-D3 Dispersion

#### Step 1: Generate QE Input Files

```python
#!/usr/bin/env python3
"""
Generate QE input files for adsorption energy with DFT-D3 dispersion.
Produces: pw_slab.in, pw_molecule.in, pw_adslab.in
All include vdw_corr='dft-d3' for dispersion corrections.

Example: CO2 on MgO(100).
"""

import numpy as np
from pymatgen.core import Structure, Lattice, Molecule
from pymatgen.core.surface import SlabGenerator
from pymatgen.analysis.adsorption import AdsorbateSiteFinder
from pymatgen.io.pwscf import PWInput
from pathlib import Path

# ============================================================
# Configuration
# ============================================================
PSEUDO_DIR = "./pseudo"
ECUTWFC = 60.0
ECUTRHO = 480.0
Path(PSEUDO_DIR).mkdir(exist_ok=True)

pseudos = {
    "Mg": "Mg.pbe-spnl-kjpaw_psl.1.0.0.UPF",
    "O":  "O.pbe-n-kjpaw_psl.1.0.0.UPF",
    "C":  "C.pbe-n-kjpaw_psl.1.0.0.UPF",
}

# Common system parameters with DFT-D3
system_params = {
    "ecutwfc": ECUTWFC,
    "ecutrho": ECUTRHO,
    "occupations": "smearing",
    "smearing": "cold",
    "degauss": 0.01,
    "vdw_corr": "dft-d3",     # <-- DFT-D3 dispersion correction
    "dftd3_version": 4,        # D3(BJ) with Becke-Johnson damping
}

electron_params = {"conv_thr": 1.0e-8, "mixing_beta": 0.3}

# ============================================================
# 1. Build MgO(100) slab
# ============================================================
mgo_bulk = Structure.from_spacegroup(
    "Fm-3m",
    lattice=Lattice.cubic(4.212),
    species=["Mg", "O"],
    coords=[[0.0, 0.0, 0.0], [0.5, 0.5, 0.5]],
)

slabgen = SlabGenerator(
    mgo_bulk, (1, 0, 0),
    min_slab_size=10.0,
    min_vacuum_size=18.0,
    center_slab=True,
    primitive=True,
)
slab = slabgen.get_slabs()[0]
slab.make_supercell([2, 2, 1])
kpts_slab = (4, 4, 1)

# ============================================================
# 2. Clean slab input
# ============================================================
pw_slab = PWInput(
    slab,
    pseudo=pseudos,
    control={
        "calculation": "relax",
        "restart_mode": "from_scratch",
        "pseudo_dir": PSEUDO_DIR,
        "outdir": "./tmp_slab",
        "prefix": "slab",
        "tprnfor": True,
        "tstress": True,
        "forc_conv_thr": 1.0e-4,
    },
    system=system_params,
    electrons=electron_params,
    kpoints_grid=kpts_slab,
)
pw_slab.write_file("pw_slab.in")
print(f"Written pw_slab.in ({len(slab)} atoms, kpts={kpts_slab})")

# ============================================================
# 3. Isolated CO2 molecule in box
# ============================================================
co2_box = Structure(
    lattice=Lattice.cubic(15.0),
    species=["C", "O", "O"],
    coords=[[0.5, 0.5, 0.5],
            [0.5, 0.5, 0.5 + 1.16/15.0],
            [0.5, 0.5, 0.5 - 1.16/15.0]],
)

pw_mol = PWInput(
    co2_box,
    pseudo=pseudos,
    control={
        "calculation": "relax",
        "restart_mode": "from_scratch",
        "pseudo_dir": PSEUDO_DIR,
        "outdir": "./tmp_mol",
        "prefix": "mol",
        "tprnfor": True,
    },
    system=system_params | {"degauss": 0.005, "smearing": "gaussian"},
    electrons=electron_params | {"mixing_beta": 0.4},
    kpoints_grid=(1, 1, 1),
)
pw_mol.write_file("pw_molecule.in")
print("Written pw_molecule.in (Gamma-only)")

# ============================================================
# 4. Adsorbate + slab
# ============================================================
asf = AdsorbateSiteFinder(slab)
co2_pmg = Molecule(
    ["C", "O", "O"],
    [[0, 0, 0], [0, 0, 1.16], [0, 0, -1.16]],
)

adslabs = asf.generate_adsorption_structures(
    co2_pmg,
    repeat=[1, 1, 1],
    find_args={"distance": 2.5},
)

if adslabs:
    adslab = adslabs[0]
    pw_adslab = PWInput(
        adslab,
        pseudo=pseudos,
        control={
            "calculation": "relax",
            "restart_mode": "from_scratch",
            "pseudo_dir": PSEUDO_DIR,
            "outdir": "./tmp_adslab",
            "prefix": "adslab",
            "tprnfor": True,
            "tstress": True,
            "forc_conv_thr": 1.0e-4,
        },
        system=system_params,
        electrons=electron_params,
        kpoints_grid=kpts_slab,
    )
    pw_adslab.write_file("pw_adslab.in")
    print(f"Written pw_adslab.in ({len(adslab)} atoms)")
else:
    print("ERROR: No adslab generated")

print("\nIMPORTANT: Add selective dynamics (if_pos flags) to fix bottom layers.")
print("See defects-reactions/surface-adsorption/ for the helper script.")
print("\nDFT-D3 settings included:")
print("  vdw_corr = 'dft-d3'")
print("  dftd3_version = 4  (Becke-Johnson damping)")
```

#### Step 2: Run QE Calculations

```bash
#!/bin/bash
# Run QE adsorption calculations with DFT-D3 dispersion

NPROC=4

# Download pseudopotentials
mkdir -p pseudo tmp_slab tmp_mol tmp_adslab
cd pseudo
wget -nc "https://pseudopotentials.quantum-espresso.org/upf_files/Mg.pbe-spnl-kjpaw_psl.1.0.0.UPF" 2>/dev/null
wget -nc "https://pseudopotentials.quantum-espresso.org/upf_files/O.pbe-n-kjpaw_psl.1.0.0.UPF" 2>/dev/null
wget -nc "https://pseudopotentials.quantum-espresso.org/upf_files/C.pbe-n-kjpaw_psl.1.0.0.UPF" 2>/dev/null
cd ..

# 1. Clean slab
echo "=== Relaxing clean slab ==="
mpirun -np $NPROC pw.x -in pw_slab.in > pw_slab.out 2>&1
echo "Slab: $(grep '!' pw_slab.out | tail -1)"

# 2. Isolated molecule
echo "=== Relaxing CO2 molecule ==="
mpirun -np $NPROC pw.x -in pw_molecule.in > pw_molecule.out 2>&1
echo "Molecule: $(grep '!' pw_molecule.out | tail -1)"

# 3. Adsorbate + slab
echo "=== Relaxing adsorbate + slab ==="
mpirun -np $NPROC pw.x -in pw_adslab.in > pw_adslab.out 2>&1
echo "Adslab: $(grep '!' pw_adslab.out | tail -1)"
```

#### Step 3: Parse and Compute Adsorption Energy

```python
#!/usr/bin/env python3
"""
Parse QE outputs and compute adsorption energy.
Extracts both total energy and the D3 dispersion contribution separately.
"""

import re
import json

def parse_qe_energy(filename):
    """Extract final total energy in eV."""
    energy = None
    with open(filename) as f:
        for line in f:
            if "!" in line and "total energy" in line:
                match = re.search(r"=\s+([-\d.]+)\s+Ry", line)
                if match:
                    energy = float(match.group(1)) * 13.605693123
    return energy

def parse_d3_energy(filename):
    """Extract DFT-D3 dispersion energy contribution in eV."""
    with open(filename) as f:
        for line in f:
            if "Dispersion Correction" in line or "DFT-D3" in line:
                match = re.search(r"=\s+([-\d.]+)\s+Ry", line)
                if match:
                    return float(match.group(1)) * 13.605693123
    return None

# Parse all three calculations
e_slab = parse_qe_energy("pw_slab.out")
e_mol = parse_qe_energy("pw_molecule.out")
e_adslab = parse_qe_energy("pw_adslab.out")

print(f"E_slab   = {e_slab:.6f} eV")
print(f"E_mol    = {e_mol:.6f} eV")
print(f"E_adslab = {e_adslab:.6f} eV")

# Adsorption energy
e_ads = e_adslab - e_slab - e_mol
print(f"\nE_ads = {e_ads:.4f} eV")

if e_ads < -0.5:
    print("Classification: chemisorption")
elif e_ads < -0.05:
    print("Classification: physisorption / weak chemisorption")
elif e_ads < 0:
    print("Classification: very weak physisorption")
else:
    print("Classification: not bound (endothermic)")

# D3 contributions
d3_slab = parse_d3_energy("pw_slab.out")
d3_mol = parse_d3_energy("pw_molecule.out")
d3_adslab = parse_d3_energy("pw_adslab.out")

if d3_slab is not None and d3_mol is not None and d3_adslab is not None:
    d3_contribution = d3_adslab - d3_slab - d3_mol
    print(f"\nD3 dispersion contribution to E_ads: {d3_contribution:.4f} eV")
    print(f"PBE-only E_ads (approx): {e_ads - d3_contribution:.4f} eV")

results = {
    "e_slab_eV": e_slab,
    "e_molecule_eV": e_mol,
    "e_adslab_eV": e_adslab,
    "e_ads_eV": e_ads,
    "d3_slab_eV": d3_slab,
    "d3_molecule_eV": d3_mol,
    "d3_adslab_eV": d3_adslab,
}
with open("qe_adsorption_results.json", "w") as f:
    json.dump(results, f, indent=2)
print("\nSaved to qe_adsorption_results.json")
```

### Method C: VASP Workflow (Future)

```
VASP adsorption energy workflow (requires VASP license):

1. Clean slab relaxation:
   IBRION = 2, ISIF = 2, NSW = 200
   ENCUT = 400 (or 1.3x ENMAX)
   PREC = Accurate
   Selective dynamics in POSCAR to fix bottom layers
   IVDW = 11          # <-- DFT-D3 with Becke-Johnson damping
   KPOINTS: Nx Ny 1

2. Gas-phase molecule:
   Same ENCUT, PREC
   Large box (15 A), Gamma-only k-point
   IVDW = 11
   IBRION = 2, ISIF = 2, NSW = 100

3. Adsorbate + slab:
   Same settings as clean slab
   IVDW = 11
   Selective dynamics: fix bottom layers, free adsorbate + top layers

4. Compute adsorption energy:
   E_ads = E(adslab) - E(slab) - E(molecule)
   All from OSZICAR or OUTCAR final energies

5. DFT-D3 flavors in VASP (IVDW tag):
   IVDW = 1   -> DFT-D2 (Grimme)
   IVDW = 11  -> DFT-D3 with BJ damping (recommended)
   IVDW = 12  -> DFT-D3 with zero damping
   IVDW = 4   -> Tkatchenko-Scheffler
   IVDW = 20  -> Tkatchenko-Scheffler with self-consistent screening (MBD)

6. Coverage study:
   Generate POSCAR files with different supercell sizes using pymatgen
   One adsorbate per supercell -> coverage = 1/(N_x * N_y) ML
   Run VASP for each, collect E_ads vs coverage

7. Site preference:
   Generate POSCAR for each adsorption site (top, bridge, hollow)
   using pymatgen AdsorbateSiteFinder
   Compare E_ads across sites
```

## Key Parameters

| Parameter | Recommended Value | Notes |
|---|---|---|
| Slab thickness | 4-6 layers (8-14 A) | Converge E_ads vs. layers |
| Vacuum | 15-20 A | Prevent periodic image interaction |
| Surface supercell | >= 2x2 | Ensure adsorbate-adsorbate distance > 8 A for low-coverage limit |
| Fixed bottom layers | 2-3 layers | Mimic bulk interior |
| K-points (slab) | 4x4x1 to 8x8x1 | Dense in-plane, 1 along vacuum |
| K-points (molecule) | Gamma only (1x1x1) | Isolated molecule in large box |
| MACE fmax | 0.01 eV/A | Tight for accurate E_ads |
| QE ecutwfc | 50-80 Ry | Match between slab and molecule calculations |
| Adsorbate-surface initial distance | 1.8-2.5 A | Optimization finds equilibrium; start reasonable |
| vdw_corr (QE) | `'dft-d3'` with `dftd3_version=4` | BJ damping recommended |
| IVDW (VASP) | 11 | DFT-D3(BJ), most widely validated |
| Dipole correction | Recommended for asymmetric slabs | QE: `dipfield=.true., edir=3`; VASP: `LDIPOL=.TRUE., IDIPOL=3` |
| Molecule box size | >= 15 A | Avoid periodic image interaction for isolated molecule |

## Interpreting Results

1. **E_ads sign**: Negative = exothermic (favorable). More negative = stronger binding.
2. **Binding classification**:
   - Physisorption: -0.01 to -0.3 eV (van der Waals dominated; D3 correction essential)
   - Weak chemisorption: -0.3 to -1.0 eV
   - Strong chemisorption: -1.0 to -5.0 eV
3. **Site preference**: The site with the most negative E_ads is the preferred binding site. For metals, CO typically prefers top (Pt) or hollow (fcc metals). For oxides, adsorption often occurs above cation sites.
4. **Coverage dependence**: E_ads typically becomes less negative (weaker) at higher coverage due to adsorbate-adsorbate repulsion. The low-coverage limit (large supercell) gives the intrinsic binding energy.
5. **Dispersion contribution**: For physisorbed systems (e.g., noble gases, saturated hydrocarbons on surfaces), D3 can contribute 0.1-0.5 eV to E_ads. Omitting D3 may change the sign of E_ads.
6. **MACE vs. DFT**: MACE E_ads can deviate 0.1-0.5 eV from DFT, especially for charge-transfer systems. Use DFT for quantitative catalysis predictions.

## Common Issues

| Issue | Solution |
|---|---|
| Adsorbate desorbs during relaxation | Start with shorter initial distance (1.5-1.8 A); use smaller BFGS step size; try different starting orientation |
| Adsorbate dissociates on surface | May be physical; if unwanted, constrain adsorbate internal bonds or use shorter optimization |
| DFT-D3 not available in QE build | Recompile QE with `-DD3` flag, or use `vdw_corr='grimme-d2'` as fallback |
| Coverage too high leads to artifacts | Use larger supercell (>= 3x3); ensure adsorbate images are separated by > 8 A |
| Different E_ads with/without dispersion | Expected. Report both values. D3 is essential for physisorption and recommended for chemisorption. |
| Wrong adsorption site after relaxation | Adsorbate migrated to a different site; this reveals the true preferred site. Recheck by starting at other sites. |
| SCF convergence fails for adslab | Reduce mixing_beta to 0.1-0.2; try `mixing_mode='local-TF'`; check initial geometry for atom overlap |
| Large supercell is too expensive for DFT | Use MACE for coverage study; validate the low-coverage limit with DFT |
| Molecule energy depends on box size | Increase box to 15-20 A; verify energy converges with box size |
| Inconsistent pseudopotentials between slab and molecule | Always use the same pseudopotential set and cutoffs for ALL calculations in the E_ads formula |
