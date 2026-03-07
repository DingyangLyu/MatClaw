# Surface Adsorption Calculations

## When to Use

- Calculate adsorption energies of molecules or atoms on solid surfaces
- Screen adsorption sites (atop, bridge, hollow) for catalytic activity
- Determine preferred surface terminations
- Compute work function changes upon adsorption
- Study surface reconstruction and adsorbate-induced geometry changes
- Generate binding energy trends across a series of surfaces or adsorbates

## Method Selection

```
Quick screening of adsorption sites / trends?
  --> ASE + MACE (Method A): minutes per configuration

Publication-quality adsorption energy?
  --> QE DFT (Method B): hours per slab configuration

Need electronic structure (charge transfer, PDOS at surface)?
  --> QE DFT required (Method B)

Work function calculation?
  --> QE DFT required (Method B): needs planar-averaged potential
```

## Prerequisites

- pymatgen (slab generation, adsorption site finding)
- ASE + mace-torch (Method A)
- Quantum ESPRESSO pw.x, pp.x (Method B)
- SSSP pseudopotentials for QE

## Detailed Steps

### Method A: ASE + MACE

#### Complete Workflow: CO Adsorption on Cu(111)

```python
#!/usr/bin/env python3
"""
Surface adsorption workflow using ASE + MACE.
Example: CO on Cu(111) -- a classic catalysis benchmark.

Follows the pattern from atomate2's AdsorptionMaker:
  1. Relax bulk
  2. Generate slab
  3. Relax clean slab
  4. Relax isolated molecule
  5. Find adsorption sites and relax adsorbate+slab
  6. Compute adsorption energies
"""

import numpy as np
from pymatgen.core import Structure, Lattice, Molecule, Element
from pymatgen.core.surface import SlabGenerator
from pymatgen.analysis.adsorption import AdsorbateSiteFinder
from pymatgen.io.ase import AseAtomsAdaptor
from ase.optimize import BFGS
from ase.filters import FrechetCellFilter
from ase.constraints import FixAtoms
from ase.build import molecule as ase_molecule
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

# ============================================================
# 2. Relax bulk Cu (fcc)
# ============================================================
cu_bulk = Structure.from_spacegroup(
    "Fm-3m",
    lattice=Lattice.cubic(3.615),  # experimental Cu lattice parameter
    species=["Cu"],
    coords=[[0.0, 0.0, 0.0]],
)

atoms_bulk = adaptor.get_atoms(cu_bulk)
atoms_bulk.calc = calc
ecf = FrechetCellFilter(atoms_bulk)
opt_bulk = BFGS(ecf, logfile="relax_bulk_Cu.log")
opt_bulk.run(fmax=0.005, steps=200)
cu_bulk_relaxed = adaptor.get_structure(atoms_bulk)
print(f"Relaxed Cu bulk: a = {cu_bulk_relaxed.lattice.a:.4f} A")

# ============================================================
# 3. Generate Cu(111) slab
# ============================================================
slabgen = SlabGenerator(
    initial_structure=cu_bulk_relaxed,
    miller_index=(1, 1, 1),
    min_slab_size=10.0,     # minimum slab thickness in Angstrom
    min_vacuum_size=18.0,   # vacuum thickness in Angstrom
    center_slab=True,       # center slab in the cell
    in_unit_planes=False,   # interpret sizes in Angstrom, not planes
    lll_reduce=True,
    primitive=True,
)

slabs = slabgen.get_slabs(
    symmetrize=False,       # asymmetric slab for adsorption on one side
)
print(f"Generated {len(slabs)} slab terminations for Cu(111)")

# Use first slab (should be the most common termination)
slab = slabs[0]
print(f"Slab: {slab.formula}, {len(slab)} atoms")
print(f"Slab thickness: {slab.lattice.c:.2f} A (including vacuum)")

# Make slab larger laterally if needed (2x2 surface supercell)
slab.make_supercell([2, 2, 1])
print(f"2x2 surface supercell: {slab.formula}, {len(slab)} atoms")

# ============================================================
# 4. Relax clean slab (fix bottom layers)
# ============================================================
atoms_slab = adaptor.get_atoms(slab)
atoms_slab.calc = calc

# Fix bottom half of atoms (those with z < median z)
positions = atoms_slab.get_positions()
z_coords = positions[:, 2]
z_median = np.median(z_coords)
fix_mask = z_coords < z_median  # fix bottom half
n_fixed = np.sum(fix_mask)
n_free = len(atoms_slab) - n_fixed
print(f"Fixed {n_fixed} bottom atoms, {n_free} free surface atoms")

atoms_slab.set_constraint(FixAtoms(mask=fix_mask))
opt_slab = BFGS(atoms_slab, logfile="relax_slab_clean.log")
opt_slab.run(fmax=0.01, steps=300)

e_slab = atoms_slab.get_potential_energy()
print(f"Clean slab energy: {e_slab:.6f} eV")

# Save relaxed slab
slab_relaxed = adaptor.get_structure(atoms_slab)

# ============================================================
# 5. Relax isolated CO molecule in a box
# ============================================================
co_mol = ase_molecule("CO")
co_mol.center(vacuum=10.0)
co_mol.calc = calc
opt_mol = BFGS(co_mol, logfile="relax_CO.log")
opt_mol.run(fmax=0.005, steps=100)

e_mol = co_mol.get_potential_energy()
co_bond_length = co_mol.get_distance(0, 1)
print(f"\nCO molecule energy: {e_mol:.6f} eV")
print(f"CO bond length: {co_bond_length:.4f} A")

# ============================================================
# 6. Find adsorption sites using pymatgen
# ============================================================
asf = AdsorbateSiteFinder(slab_relaxed)

# Get unique adsorption sites (atop, bridge, hollow)
ads_sites = asf.find_adsorption_sites(
    distance=2.0,           # initial adsorbate-surface distance in A
    symm_reduce=0.01,       # tolerance for symmetry reduction
    near_reduce=0.5,        # merge sites closer than this (A)
    no_obtuse_hollow=True,  # skip obtuse hollow sites
)

print(f"\nAdsorption sites found:")
for site_type in ["ontop", "bridge", "hollow"]:
    coords = ads_sites.get(site_type, [])
    print(f"  {site_type}: {len(coords)} sites")

# ============================================================
# 7. Place adsorbate and relax at each site type
# ============================================================
# Build pymatgen Molecule for CO (C down toward surface for Cu)
co_pmg = Molecule(["C", "O"], [[0, 0, 0], [0, 0, 1.128]])

results = {}

for site_type in ["ontop", "bridge", "hollow"]:
    sites = ads_sites.get(site_type, [])
    if not sites:
        continue

    # Use first site of each type
    site_coords = sites[0]
    print(f"\n--- {site_type} site at {site_coords} ---")

    # Generate adslab structure
    adslab_structs = asf.generate_adsorption_structures(
        co_pmg,
        repeat=[1, 1, 1],  # don't repeat (already 2x2)
        find_args={"distance": 2.0},
    )

    if not adslab_structs:
        print(f"  No adslab generated for {site_type}")
        continue

    # Use the first generated structure for this site type
    # (pymatgen generates one per unique site)
    adslab = adslab_structs[0]

    # Relax adslab with fixed bottom layers
    atoms_adslab = adaptor.get_atoms(adslab)
    atoms_adslab.calc = mace_mp(
        model="medium", dispersion=False, default_dtype="float64"
    )

    # Fix bottom atoms (same approach as clean slab)
    pos = atoms_adslab.get_positions()
    z = pos[:, 2]
    z_med = np.median(z)
    fix = z < z_med
    atoms_adslab.set_constraint(FixAtoms(mask=fix))

    opt_ads = BFGS(atoms_adslab, logfile=f"relax_adslab_{site_type}.log")
    opt_ads.run(fmax=0.01, steps=500)

    e_adslab = atoms_adslab.get_potential_energy()

    # Adsorption energy: E_ads = E_slab+mol - E_slab - E_mol
    e_ads = e_adslab - e_slab - e_mol
    print(f"  E_adslab = {e_adslab:.6f} eV")
    print(f"  E_ads = {e_ads:.4f} eV")

    results[site_type] = {
        "e_adslab": e_adslab,
        "e_ads": e_ads,
        "site_coords": site_coords.tolist() if hasattr(site_coords, "tolist") else list(site_coords),
    }

# ============================================================
# 8. Summary and visualization
# ============================================================
print("\n" + "=" * 50)
print("ADSORPTION ENERGY SUMMARY")
print("=" * 50)
print(f"E_slab (clean) = {e_slab:.6f} eV")
print(f"E_CO (gas)     = {e_mol:.6f} eV")
print(f"{'Site':<10} {'E_ads (eV)':<15} {'Favorable?'}")
print("-" * 40)
for site_type, data in sorted(results.items(), key=lambda x: x[1]["e_ads"]):
    fav = "YES" if data["e_ads"] < 0 else "NO"
    print(f"{site_type:<10} {data['e_ads']:<15.4f} {fav}")

# Bar chart of adsorption energies
if results:
    fig, ax = plt.subplots(figsize=(6, 4))
    site_names = list(results.keys())
    e_ads_values = [results[s]["e_ads"] for s in site_names]
    colors = ["green" if e < 0 else "red" for e in e_ads_values]

    bars = ax.bar(site_names, e_ads_values, color=colors, edgecolor="black", alpha=0.7)
    ax.axhline(y=0, color="black", linewidth=0.8)
    ax.set_ylabel("Adsorption energy (eV)", fontsize=12)
    ax.set_title("CO adsorption on Cu(111)", fontsize=13)
    ax.grid(axis="y", alpha=0.3)

    for bar, val in zip(bars, e_ads_values):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                f"{val:.3f}", ha="center", va="bottom", fontsize=10)

    fig.tight_layout()
    fig.savefig("adsorption_energies.png", dpi=150)
    print("\nPlot saved to adsorption_energies.png")

# Save results
output = {
    "system": "CO on Cu(111)",
    "slab_formula": slab.formula,
    "n_slab_atoms": len(slab),
    "e_slab_eV": e_slab,
    "e_molecule_eV": e_mol,
    "adsorption_sites": results,
}
with open("adsorption_results.json", "w") as f:
    json.dump(output, f, indent=2, default=float)
print("Results saved to adsorption_results.json")
```

#### Visualize Slab and Adsorption Sites

```python
#!/usr/bin/env python3
"""Visualize slab structure with adsorption sites marked."""

import numpy as np
from pymatgen.core import Structure, Lattice, Molecule
from pymatgen.core.surface import SlabGenerator
from pymatgen.analysis.adsorption import AdsorbateSiteFinder, plot_slab
from pymatgen.io.ase import AseAtomsAdaptor
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Build slab (use relaxed slab from previous step, or rebuild)
cu_bulk = Structure.from_spacegroup(
    "Fm-3m",
    lattice=Lattice.cubic(3.615),
    species=["Cu"],
    coords=[[0.0, 0.0, 0.0]],
)

slabgen = SlabGenerator(cu_bulk, (1, 1, 1), 10.0, 18.0,
                        center_slab=True, primitive=True)
slab = slabgen.get_slabs()[0]
slab.make_supercell([2, 2, 1])

# Find adsorption sites
asf = AdsorbateSiteFinder(slab)
ads_sites = asf.find_adsorption_sites(distance=2.0)

# Use pymatgen's built-in slab plotter
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111)

plot_slab(slab, ax, adsorption_sites=True, decay=0.1)
ax.set_title("Cu(111) 2x2 slab with adsorption sites", fontsize=13)

fig.tight_layout()
fig.savefig("slab_adsorption_sites.png", dpi=150)
print("Saved slab_adsorption_sites.png")

# Also save slab structure
from pymatgen.io.cif import CifWriter
CifWriter(slab).write_file("slab_Cu111.cif")
print("Saved slab_Cu111.cif")
```

#### Different Miller Indices and Surface Terminations

```python
#!/usr/bin/env python3
"""
Generate slabs for multiple Miller indices and compare surface energies.
"""

import numpy as np
from pymatgen.core import Structure, Lattice
from pymatgen.core.surface import SlabGenerator
from pymatgen.io.ase import AseAtomsAdaptor
from ase.optimize import BFGS
from ase.constraints import FixAtoms
from mace.calculators import mace_mp
import json

calc = mace_mp(model="medium", dispersion=False, default_dtype="float64")
adaptor = AseAtomsAdaptor()

# Relax bulk first
cu_bulk = Structure.from_spacegroup(
    "Fm-3m", lattice=Lattice.cubic(3.615),
    species=["Cu"], coords=[[0, 0, 0]],
)
atoms_bulk = adaptor.get_atoms(cu_bulk)
atoms_bulk.calc = calc
from ase.filters import FrechetCellFilter
opt = BFGS(FrechetCellFilter(atoms_bulk), logfile=None)
opt.run(fmax=0.005, steps=200)
cu_relaxed = adaptor.get_structure(atoms_bulk)
e_bulk_per_atom = atoms_bulk.get_potential_energy() / len(atoms_bulk)

miller_indices = [(1, 0, 0), (1, 1, 0), (1, 1, 1)]
surface_results = {}

for hkl in miller_indices:
    hkl_str = "".join(map(str, hkl))
    print(f"\n=== Cu({hkl_str}) ===")

    slabgen = SlabGenerator(
        cu_relaxed, hkl,
        min_slab_size=12.0,
        min_vacuum_size=18.0,
        center_slab=True,
        primitive=True,
        lll_reduce=True,
    )

    slabs = slabgen.get_slabs()
    if not slabs:
        print(f"  No slabs generated")
        continue

    slab = slabs[0]
    slab.make_supercell([2, 2, 1])
    n_atoms = len(slab)
    print(f"  {n_atoms} atoms, formula: {slab.formula}")

    # Relax with fixed bottom
    atoms_slab = adaptor.get_atoms(slab)
    atoms_slab.calc = mace_mp(model="medium", dispersion=False, default_dtype="float64")

    z = atoms_slab.get_positions()[:, 2]
    fix_mask = z < np.median(z)
    atoms_slab.set_constraint(FixAtoms(mask=fix_mask))

    opt_s = BFGS(atoms_slab, logfile=f"relax_slab_{hkl_str}.log")
    opt_s.run(fmax=0.01, steps=300)

    e_slab = atoms_slab.get_potential_energy()

    # Surface energy: gamma = (E_slab - N * e_bulk_per_atom) / (2 * A)
    # Factor of 2 because slab has two surfaces
    area = np.linalg.norm(np.cross(
        slab.lattice.matrix[0], slab.lattice.matrix[1]
    ))  # surface area in A^2
    gamma = (e_slab - n_atoms * e_bulk_per_atom) / (2 * area)
    gamma_J_m2 = gamma * 16.0218  # eV/A^2 to J/m^2

    print(f"  E_slab = {e_slab:.4f} eV")
    print(f"  Surface area = {area:.2f} A^2")
    print(f"  Surface energy = {gamma:.6f} eV/A^2 = {gamma_J_m2:.4f} J/m^2")

    surface_results[hkl_str] = {
        "miller_index": list(hkl),
        "n_atoms": n_atoms,
        "e_slab_eV": e_slab,
        "area_A2": area,
        "gamma_eV_per_A2": gamma,
        "gamma_J_per_m2": gamma_J_m2,
    }

# Summary
print("\n" + "=" * 50)
print("SURFACE ENERGY SUMMARY")
print("=" * 50)
print(f"{'Surface':<12} {'gamma (J/m^2)':<15}")
print("-" * 30)
for hkl_str, data in sorted(surface_results.items(),
                              key=lambda x: x[1]["gamma_J_per_m2"]):
    print(f"Cu({hkl_str}){'':<6} {data['gamma_J_per_m2']:<15.4f}")

with open("surface_energies.json", "w") as f:
    json.dump(surface_results, f, indent=2, default=float)
print("\nSaved to surface_energies.json")
```

### Method B: QE DFT

#### Step 1: Generate QE Input Files

```python
#!/usr/bin/env python3
"""
Generate QE input files for surface adsorption calculations.
Produces:
  1. pw_slab_clean.in  -- clean slab relaxation
  2. pw_molecule.in    -- isolated molecule in box
  3. pw_adslab.in      -- slab + adsorbate relaxation
"""

import numpy as np
from pymatgen.core import Structure, Lattice, Molecule, Element
from pymatgen.core.surface import SlabGenerator
from pymatgen.analysis.adsorption import AdsorbateSiteFinder
from pymatgen.io.pwscf import PWInput
from pathlib import Path

# ============================================================
# Configuration
# ============================================================
PSEUDO_DIR = "./pseudo"
ECUTWFC = 50.0
ECUTRHO = 400.0

pseudos = {
    "Cu": "Cu.pbe-dn-kjpaw_psl.1.0.0.UPF",
    "C":  "C.pbe-n-kjpaw_psl.1.0.0.UPF",
    "O":  "O.pbe-n-kjpaw_psl.1.0.0.UPF",
}

# ============================================================
# 1. Build slab
# ============================================================
cu_bulk = Structure.from_spacegroup(
    "Fm-3m", lattice=Lattice.cubic(3.615),
    species=["Cu"], coords=[[0, 0, 0]],
)

slabgen = SlabGenerator(
    cu_bulk, (1, 1, 1),
    min_slab_size=8.0,      # ~4 layers
    min_vacuum_size=18.0,
    center_slab=True,
    primitive=True,
)
slab = slabgen.get_slabs()[0]
slab.make_supercell([2, 2, 1])
n_slab = len(slab)

# Determine which atoms to fix (bottom 2 layers)
z_coords = [s.coords[2] for s in slab]
z_sorted = sorted(set(round(z, 1) for z in z_coords))
n_layers = len(z_sorted)
# Fix bottom 2 layers
fix_z_threshold = z_sorted[min(1, n_layers-1)] + 0.5
print(f"Slab: {slab.formula}, {n_slab} atoms, {n_layers} layers")

# ============================================================
# 2. Clean slab input
# ============================================================
# K-points: dense in-plane, single point out-of-plane
kpts_slab = (4, 4, 1)

pw_slab = PWInput(
    slab,
    pseudo=pseudos,
    control={
        "calculation": "relax",
        "restart_mode": "from_scratch",
        "pseudo_dir": PSEUDO_DIR,
        "outdir": "./tmp_slab",
        "tprnfor": True,
        "tstress": True,
        "etot_conv_thr": 1.0e-5,
        "forc_conv_thr": 1.0e-4,
    },
    system={
        "ecutwfc": ECUTWFC,
        "ecutrho": ECUTRHO,
        "occupations": "smearing",
        "smearing": "cold",
        "degauss": 0.02,
        "input_dft": "PBE",
    },
    electrons={
        "conv_thr": 1.0e-7,
        "mixing_beta": 0.3,
    },
    kpoints_grid=kpts_slab,
)
pw_slab.write_file("pw_slab_clean.in")
print(f"Written pw_slab_clean.in (kpts={kpts_slab})")

# NOTE: For fixed bottom layers in QE, you need to manually edit the
# ATOMIC_POSITIONS block to add "if_pos" flags (0 0 0 for fixed atoms).
# pymatgen's PWInput does not support selective dynamics directly.
# See the helper script below.

# ============================================================
# 3. Isolated molecule in box
# ============================================================
co_mol_struct = Structure(
    lattice=Lattice.cubic(15.0),  # large box
    species=["C", "O"],
    coords=[[0.5, 0.5, 0.5], [0.5, 0.5, 0.575]],  # fractional
)

pw_mol = PWInput(
    co_mol_struct,
    pseudo=pseudos,
    control={
        "calculation": "relax",
        "restart_mode": "from_scratch",
        "pseudo_dir": PSEUDO_DIR,
        "outdir": "./tmp_mol",
        "tprnfor": True,
    },
    system={
        "ecutwfc": ECUTWFC,
        "ecutrho": ECUTRHO,
        "occupations": "smearing",
        "smearing": "gaussian",
        "degauss": 0.005,
    },
    electrons={
        "conv_thr": 1.0e-7,
        "mixing_beta": 0.4,
    },
    kpoints_grid=(1, 1, 1),
)
pw_mol.write_file("pw_molecule.in")
print("Written pw_molecule.in (Gamma-only)")

# ============================================================
# 4. Adsorbate + slab
# ============================================================
asf = AdsorbateSiteFinder(slab)
co_pmg = Molecule(["C", "O"], [[0, 0, 0], [0, 0, 1.128]])

# Generate adsorption structures (ontop site, first one)
adslabs = asf.generate_adsorption_structures(
    co_pmg,
    repeat=[1, 1, 1],
    find_args={"distance": 1.9},
)

if adslabs:
    adslab = adslabs[0]  # first (typically ontop) configuration
    print(f"Adslab: {adslab.formula}, {len(adslab)} atoms")

    pw_adslab = PWInput(
        adslab,
        pseudo=pseudos,
        control={
            "calculation": "relax",
            "restart_mode": "from_scratch",
            "pseudo_dir": PSEUDO_DIR,
            "outdir": "./tmp_adslab",
            "tprnfor": True,
            "tstress": True,
            "etot_conv_thr": 1.0e-5,
            "forc_conv_thr": 1.0e-4,
        },
        system={
            "ecutwfc": ECUTWFC,
            "ecutrho": ECUTRHO,
            "occupations": "smearing",
            "smearing": "cold",
            "degauss": 0.02,
        },
        electrons={
            "conv_thr": 1.0e-7,
            "mixing_beta": 0.3,
        },
        kpoints_grid=kpts_slab,
    )
    pw_adslab.write_file("pw_adslab.in")
    print(f"Written pw_adslab.in (kpts={kpts_slab})")
else:
    print("ERROR: No adslab structures generated")
```

#### Helper: Add Selective Dynamics to QE Input

```python
#!/usr/bin/env python3
"""
Post-process QE input file to add selective dynamics (fix bottom layers).
QE uses if_pos flags: 0 0 0 = fixed, 1 1 1 = free.
"""

import numpy as np


def add_selective_dynamics_qe(input_file, output_file, fix_below_z_frac=0.45):
    """
    Read a QE input file and add if_pos flags to fix atoms below a z threshold.

    Parameters
    ----------
    input_file : str
        Path to QE input file.
    output_file : str
        Path to write modified file.
    fix_below_z_frac : float
        Fractional z coordinate below which atoms are fixed.
    """
    with open(input_file) as f:
        lines = f.readlines()

    new_lines = []
    in_atomic_positions = False
    coord_format = "crystal"  # default

    for line in lines:
        stripped = line.strip()

        if stripped.upper().startswith("ATOMIC_POSITIONS"):
            in_atomic_positions = True
            # Detect coordinate format
            if "crystal" in stripped.lower():
                coord_format = "crystal"
            elif "angstrom" in stripped.lower():
                coord_format = "angstrom"
            new_lines.append(line)
            continue

        if in_atomic_positions and stripped:
            parts = stripped.split()
            if len(parts) >= 4 and parts[0].isalpha():
                # This is an atom line: species x y z [if_pos_x if_pos_y if_pos_z]
                z_val = float(parts[3])
                if z_val < fix_below_z_frac:
                    # Fix this atom
                    base = f"  {parts[0]:4s} {parts[1]:>14s} {parts[2]:>14s} {parts[3]:>14s}  0 0 0\n"
                else:
                    # Free atom
                    base = f"  {parts[0]:4s} {parts[1]:>14s} {parts[2]:>14s} {parts[3]:>14s}  1 1 1\n"
                new_lines.append(base)
                continue
            else:
                in_atomic_positions = False

        new_lines.append(line)

    with open(output_file, "w") as f:
        f.writelines(new_lines)
    print(f"Written {output_file} with selective dynamics (fix z < {fix_below_z_frac})")


# Apply to slab and adslab inputs
add_selective_dynamics_qe("pw_slab_clean.in", "pw_slab_clean_fixed.in", fix_below_z_frac=0.45)
add_selective_dynamics_qe("pw_adslab.in", "pw_adslab_fixed.in", fix_below_z_frac=0.40)
```

#### Step 2: Run QE Calculations

```bash
#!/bin/bash
# Run surface adsorption QE calculations

NPROC=4

# Download pseudopotentials
mkdir -p pseudo tmp_slab tmp_mol tmp_adslab
cd pseudo
wget -nc "https://pseudopotentials.quantum-espresso.org/upf_files/Cu.pbe-dn-kjpaw_psl.1.0.0.UPF" 2>/dev/null
wget -nc "https://pseudopotentials.quantum-espresso.org/upf_files/C.pbe-n-kjpaw_psl.1.0.0.UPF" 2>/dev/null
wget -nc "https://pseudopotentials.quantum-espresso.org/upf_files/O.pbe-n-kjpaw_psl.1.0.0.UPF" 2>/dev/null
cd ..

# 1. Clean slab
echo "=== Relaxing clean slab ==="
mpirun -np $NPROC pw.x -in pw_slab_clean_fixed.in > pw_slab_clean.out 2>&1
echo "Slab: $(grep '!' pw_slab_clean.out | tail -1)"

# 2. Isolated molecule
echo "=== Relaxing CO molecule ==="
mpirun -np $NPROC pw.x -in pw_molecule.in > pw_molecule.out 2>&1
echo "Molecule: $(grep '!' pw_molecule.out | tail -1)"

# 3. Adsorbate + slab
echo "=== Relaxing adsorbate + slab ==="
mpirun -np $NPROC pw.x -in pw_adslab_fixed.in > pw_adslab.out 2>&1
echo "Adslab: $(grep '!' pw_adslab.out | tail -1)"
```

#### Step 3: Compute Adsorption Energy and Work Function

```python
#!/usr/bin/env python3
"""
Parse QE outputs and compute adsorption energy.
Also compute work function from the electrostatic potential.
"""

import re
import numpy as np
import json

def parse_qe_energy(filename):
    """Extract final total energy (eV) from QE output."""
    energy = None
    with open(filename) as f:
        for line in f:
            if "!" in line and "total energy" in line:
                match = re.search(r"=\s+([-\d.]+)\s+Ry", line)
                if match:
                    energy = float(match.group(1)) * 13.605693123
    return energy

def parse_fermi_energy(filename):
    """Extract Fermi energy (eV) from QE output."""
    with open(filename) as f:
        for line in f:
            if "the Fermi energy is" in line:
                match = re.search(r"is\s+([-\d.]+)\s+ev", line, re.IGNORECASE)
                if match:
                    return float(match.group(1))
    return None

# Parse energies
e_slab = parse_qe_energy("pw_slab_clean.out")
e_mol = parse_qe_energy("pw_molecule.out")
e_adslab = parse_qe_energy("pw_adslab.out")

print(f"E_slab  = {e_slab:.6f} eV")
print(f"E_mol   = {e_mol:.6f} eV")
print(f"E_adslab = {e_adslab:.6f} eV")

# Adsorption energy
e_ads = e_adslab - e_slab - e_mol
print(f"\nE_ads = {e_ads:.4f} eV")
if e_ads < 0:
    print("Adsorption is exothermic (favorable)")
else:
    print("Adsorption is endothermic (unfavorable)")

# Fermi level
e_fermi = parse_fermi_energy("pw_slab_clean.out")
if e_fermi is not None:
    print(f"\nFermi energy (clean slab): {e_fermi:.4f} eV")

results = {
    "e_slab_eV": e_slab,
    "e_molecule_eV": e_mol,
    "e_adslab_eV": e_adslab,
    "e_adsorption_eV": e_ads,
    "e_fermi_eV": e_fermi,
}

with open("qe_adsorption_results.json", "w") as f:
    json.dump(results, f, indent=2)
print("\nSaved to qe_adsorption_results.json")
```

#### Work Function Calculation

```python
#!/usr/bin/env python3
"""
Work function calculation from planar-averaged electrostatic potential.
Requires running pp.x after the slab SCF to extract the potential.

Work function: phi = V_vacuum - E_Fermi

Steps:
  1. Run SCF on clean slab (already done)
  2. Run pp.x to get planar-averaged potential along z
  3. Average the potential in the vacuum region
  4. phi = V_vacuum - E_Fermi
"""

# ============================================================
# Step 1: Generate pp.x input for planar-averaged potential
# ============================================================
pp_input = """&INPUTPP
  prefix   = 'pwscf',
  outdir   = './tmp_slab',
  filplot  = 'potential.dat',
  plot_num = 11,
/

&PLOT
  iflag  = 3,
  output_format = 0,
  fileout = 'avg_potential.dat',
  nfile = 1,
  e1(1) = 0.0, e1(2) = 0.0, e1(3) = 1.0,
  x0(1) = 0.0, x0(2) = 0.0, x0(3) = 0.0,
  nx = 1000,
/
"""

with open("pp_potential.in", "w") as f:
    f.write(pp_input)
print("Written pp_potential.in")

# Alternatively, use average.x for planar averaging:
avg_input = """1
potential.dat
1.0
1000
3
1.0
"""
with open("average.in", "w") as f:
    f.write(avg_input)
print("Written average.in")

# ============================================================
# Step 2: Parse and plot the potential (after running pp.x + average.x)
# ============================================================
print("""
Run these commands:
  mpirun -np 4 pp.x -in pp_potential.in > pp_potential.out
  average.x < average.in > average.out

Then parse average.out for the planar-averaged potential.
""")


def compute_work_function(avg_potential_file, e_fermi, slab_center_frac=0.5,
                          vacuum_threshold=0.8):
    """
    Compute work function from planar-averaged potential data.

    Parameters
    ----------
    avg_potential_file : str
        File with z (Bohr) and V(z) (Ry) columns from average.x.
    e_fermi : float
        Fermi energy in eV from the SCF calculation.
    slab_center_frac : float
        Approximate fractional position of the slab center.
    vacuum_threshold : float
        Fractional z above which to average the vacuum potential.

    Returns
    -------
    work_function : float
        Work function in eV.
    """
    import numpy as np

    data = np.loadtxt(avg_potential_file)
    z_bohr = data[:, 0]
    v_ry = data[:, 1]

    # Convert
    z_ang = z_bohr * 0.529177
    v_ev = v_ry * 13.605693123

    z_max = z_ang[-1]

    # Average potential in vacuum region
    vacuum_mask = z_ang > vacuum_threshold * z_max
    v_vacuum = np.mean(v_ev[vacuum_mask])

    work_function = v_vacuum - e_fermi

    return work_function, z_ang, v_ev


# Example usage (after running pp.x + average.x):
# phi, z, v = compute_work_function("avg_potential.dat", e_fermi)
# print(f"Work function: {phi:.3f} eV")
#
# import matplotlib.pyplot as plt
# fig, ax = plt.subplots(figsize=(8, 4))
# ax.plot(z, v, "b-")
# ax.axhline(y=e_fermi, color="r", linestyle="--", label=f"E_Fermi = {e_fermi:.2f} eV")
# ax.set_xlabel("z (A)")
# ax.set_ylabel("V(z) (eV)")
# ax.set_title("Planar-averaged potential")
# ax.legend()
# fig.savefig("work_function.png", dpi=150)
```

## Key Parameters

| Parameter | Recommended Value | Notes |
|---|---|---|
| Slab thickness | 4-6 layers (8-12 A) | Converge surface energy vs. number of layers |
| Vacuum thickness | 15-20 A | Must be large enough that surfaces do not interact |
| Surface supercell | 2x2 to 3x3 | Ensures adsorbate images are far apart (>8 A) |
| Fixed bottom layers | 2 layers for 4-layer slab | Bottom atoms mimic bulk; top layers relax |
| K-points (slab) | 4x4x1 to 6x6x1 | Dense in-plane, single point perpendicular to surface |
| K-points (molecule) | Gamma only (1x1x1) | Molecule in large box; no dispersion needed |
| MACE fmax | 0.01 eV/A | Tighter for accurate adsorption energies |
| QE ecutwfc | 50-80 Ry | Depends on pseudopotentials |
| Adsorbate-surface distance | 1.8-2.5 A initial | Optimization will find the equilibrium |
| Dipole correction | Recommended for asymmetric slabs | QE: `dipfield=.true.` in &CONTROL, `edir=3` |

## Interpreting Results

1. **Adsorption energy sign**: Negative E_ads means favorable (exothermic) adsorption. More negative = stronger binding.
2. **Site preference**: Compare E_ads across atop, bridge, and hollow sites. The most negative is the preferred binding site.
3. **Typical ranges**:
   - Physisorption: -0.01 to -0.3 eV (van der Waals; may need DFT-D correction)
   - Weak chemisorption: -0.3 to -1.0 eV
   - Strong chemisorption: -1.0 to -5.0 eV
4. **Work function change**: Delta_phi = phi(adslab) - phi(clean slab). Positive means the adsorbate increases the work function (electron-withdrawing).
5. **Surface energy**: Lower surface energy = more stable surface = more likely to appear in the Wulff shape.
6. **Convergence checks**: Always verify convergence with respect to slab thickness, vacuum size, k-points, and surface supercell size.
7. **MACE vs. QE**: MACE adsorption energies can deviate 0.1-0.5 eV from DFT, especially for molecules with charge transfer. Use DFT for quantitative catalysis predictions.

## Common Issues

| Issue | Solution |
|---|---|
| Slab not converged with respect to thickness | Add more layers; check surface energy vs. layers |
| Vacuum too thin (surface interaction) | Increase vacuum to 20+ A; check potential is flat in vacuum |
| Adsorbate desorbs during relaxation | Start with shorter adsorbate-surface distance; use gentler optimizer |
| Slab buckles or reconstructs | May be physical; check with experiment. Use more layers. |
| Dipole interaction between periodic images | Add dipole correction: `dipfield=.true., edir=3` in QE |
| SlabGenerator gives wrong termination | Inspect all terminations from `get_slabs()`; select manually |
| K-point convergence | Test 4x4x1, 6x6x1, 8x8x1; adsorption energy should converge within 0.05 eV |
| Molecule in box has wrong energy | Make box large enough (15 A); use Gamma-only k-points |
| MACE gives wrong binding site preference | Validate with QE DFT; MACE may not capture subtle site differences |
| SCF convergence problems for metallic slab | Use `smearing = 'cold'` or `'mv'`, increase `degauss` to 0.02-0.03 Ry |
