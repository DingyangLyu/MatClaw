# Surface Energy Calculation

## When to Use

- Calculate surface energies for different Miller indices (hkl) of a crystal
- Converge slab thickness and vacuum to obtain reliable surface energies
- Screen multiple facets rapidly to identify the most stable surfaces
- Construct a Wulff shape to predict the equilibrium crystal morphology
- Compare surface stability across different materials or compositions
- Prepare for subsequent adsorption or catalysis studies on the most relevant facets

## Method Selection

| Criterion | MACE (Method A) | QE DFT (Method B) | VASP (Method C) |
|---|---|---|---|
| Speed | Seconds per slab | Minutes to hours per slab | Minutes to hours per slab |
| Accuracy | Good for trends, screening | Publication quality | Publication quality |
| Wulff construction | Fast multi-facet screening | Validate key facets | Validate key facets |
| Best for | Rapid ranking of many hkl | Final numbers, polar surfaces | VASP-ecosystem workflows |
| Limitations | ML potential accuracy varies | Computationally expensive | Requires VASP license |

**Recommended workflow**: Screen many facets with MACE (Method A), then validate the 3-5 most important facets with QE DFT (Method B) or VASP (Method C).

## Prerequisites

- pymatgen (SlabGenerator, WulffShape)
- ASE + mace-torch (Method A)
- Quantum ESPRESSO pw.x (Method B)
- SSSP pseudopotentials for QE
- numpy, scipy, matplotlib for analysis and Wulff plotting

## Detailed Steps

### Surface Energy Formula

```
gamma = (E_slab - N * E_bulk_per_atom) / (2 * A)
```

where:
- `E_slab` = total energy of the relaxed slab supercell
- `N` = number of atoms in the slab
- `E_bulk_per_atom` = total energy per atom of the bulk crystal (same method)
- `A` = surface area of one face of the slab (|a x b|)
- Factor of 2 accounts for two exposed surfaces (top and bottom)

For asymmetric slabs (one side fixed, one side relaxed), the factor of 2 still applies if the fixed side mimics a bulk-like termination, but introduces a small systematic error. For best results, use symmetric slabs or converge the error away with thick slabs.

### Method A: MACE -- Multi-Facet Screening with Convergence

```python
#!/usr/bin/env python3
"""
Surface energy calculation for multiple Miller indices using ASE + MACE.
Includes slab thickness convergence, vacuum convergence, and Wulff construction.

Example: Al (fcc) -- a well-characterized system for surface energy benchmarks.
"""

import numpy as np
import warnings
warnings.filterwarnings("ignore")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from pymatgen.core import Structure, Lattice
from pymatgen.core.surface import SlabGenerator
from pymatgen.analysis.wulff import WulffShape
from pymatgen.io.ase import AseAtomsAdaptor
from ase.optimize import BFGS
from ase.filters import FrechetCellFilter
from ase.constraints import FixAtoms
from mace.calculators import mace_mp
import json

# ============================================================
# 1. Configuration
# ============================================================
MATERIAL_NAME = "Al"
# Build primitive cell (fcc Al)
primitive = Structure.from_spacegroup(
    "Fm-3m",
    lattice=Lattice.cubic(4.05),
    species=["Al"],
    coords=[[0.0, 0.0, 0.0]],
)

MILLER_INDICES = [
    (1, 0, 0),
    (1, 1, 0),
    (1, 1, 1),
    (2, 1, 0),
    (2, 1, 1),
    (3, 1, 0),
    (3, 1, 1),
    (2, 2, 1),
]
MIN_SLAB_THICKNESS = 14.0   # Angstrom
MIN_VACUUM = 16.0            # Angstrom
N_FIXED_LAYERS = 2           # bottom layers to freeze

# ============================================================
# 2. Set up MACE calculator and relax bulk
# ============================================================
calc = mace_mp(model="medium", dispersion=False, default_dtype="float64")
adaptor = AseAtomsAdaptor()

atoms_bulk = adaptor.get_atoms(primitive)
atoms_bulk.calc = calc
ecf = FrechetCellFilter(atoms_bulk)
opt = BFGS(ecf, logfile="relax_bulk.log")
opt.run(fmax=0.001, steps=300)

relaxed_bulk = adaptor.get_structure(atoms_bulk)
e_bulk_per_atom = atoms_bulk.get_potential_energy() / len(atoms_bulk)
a_relaxed = relaxed_bulk.lattice.a

print(f"Relaxed bulk {MATERIAL_NAME}: a = {a_relaxed:.4f} A")
print(f"E_bulk per atom = {e_bulk_per_atom:.6f} eV")

# ============================================================
# 3. Compute surface energies for all Miller indices
# ============================================================
surface_results = {}

print(f"\n{'Miller Index':<15} {'N_atoms':<10} {'Area (A^2)':<12} "
      f"{'gamma (eV/A^2)':<16} {'gamma (J/m^2)':<14}")
print("-" * 70)

for hkl in MILLER_INDICES:
    hkl_str = "".join(map(str, hkl))
    try:
        slabgen = SlabGenerator(
            initial_structure=relaxed_bulk,
            miller_index=hkl,
            min_slab_size=MIN_SLAB_THICKNESS,
            min_vacuum_size=MIN_VACUUM,
            center_slab=True,
            in_unit_planes=False,
            lll_reduce=True,
            reorient_lattice=True,
        )
        slabs = slabgen.get_slabs(symmetrize=False)
        if not slabs:
            print(f"({hkl_str}){'':<10} -- no slab generated --")
            continue

        slab = slabs[0]
        n_atoms = len(slab)

        # Convert to ASE, fix bottom layers, relax
        atoms_slab = adaptor.get_atoms(slab)
        z_coords = atoms_slab.get_positions()[:, 2]
        z_unique = np.sort(np.unique(np.round(z_coords, decimals=2)))
        if len(z_unique) >= N_FIXED_LAYERS:
            z_thr = z_unique[N_FIXED_LAYERS - 1] + 0.1
            fix_idx = [i for i, z in enumerate(z_coords) if z <= z_thr]
        else:
            fix_idx = []

        atoms_slab.set_constraint(FixAtoms(indices=fix_idx))
        atoms_slab.calc = mace_mp(
            model="medium", dispersion=False, default_dtype="float64"
        )
        opt_s = BFGS(atoms_slab, logfile=f"relax_slab_{hkl_str}.log")
        opt_s.run(fmax=0.005, steps=500)

        e_slab = atoms_slab.get_potential_energy()
        cell = atoms_slab.cell
        area = np.linalg.norm(np.cross(cell[0], cell[1]))

        gamma_eV = (e_slab - n_atoms * e_bulk_per_atom) / (2 * area)
        gamma_J = gamma_eV * 16.0218  # eV/A^2 -> J/m^2

        surface_results[hkl_str] = {
            "miller_index": list(hkl),
            "n_atoms": n_atoms,
            "area_A2": float(area),
            "gamma_eV_per_A2": float(gamma_eV),
            "gamma_J_per_m2": float(gamma_J),
        }

        print(f"({hkl_str}){'':<{12-len(hkl_str)}} {n_atoms:<10} "
              f"{area:<12.2f} {gamma_eV:<16.6f} {gamma_J:<14.4f}")

    except Exception as e:
        print(f"({hkl_str}){'':<10} ERROR: {e}")

# ============================================================
# 4. Convergence test: slab thickness for the (111) surface
# ============================================================
print("\n=== Slab Thickness Convergence for (111) ===")
thicknesses = [6.0, 8.0, 10.0, 12.0, 14.0, 16.0, 18.0, 20.0]
gamma_vs_thickness = []

for thk in thicknesses:
    slabgen = SlabGenerator(
        initial_structure=relaxed_bulk,
        miller_index=(1, 1, 1),
        min_slab_size=thk,
        min_vacuum_size=MIN_VACUUM,
        center_slab=True,
        in_unit_planes=False,
        lll_reduce=True,
    )
    slabs = slabgen.get_slabs(symmetrize=False)
    if not slabs:
        continue

    atoms_t = adaptor.get_atoms(slabs[0])
    n_t = len(atoms_t)

    z_t = atoms_t.get_positions()[:, 2]
    z_u = np.sort(np.unique(np.round(z_t, 2)))
    if len(z_u) >= N_FIXED_LAYERS:
        fix_t = [i for i, z in enumerate(z_t) if z <= z_u[N_FIXED_LAYERS - 1] + 0.1]
    else:
        fix_t = []

    atoms_t.set_constraint(FixAtoms(indices=fix_t))
    atoms_t.calc = calc
    BFGS(atoms_t, logfile=None).run(fmax=0.005, steps=300)

    e_t = atoms_t.get_potential_energy()
    area_t = np.linalg.norm(np.cross(atoms_t.cell[0], atoms_t.cell[1]))
    g_t = (e_t - n_t * e_bulk_per_atom) / (2 * area_t) * 16.0218

    gamma_vs_thickness.append((thk, n_t, g_t))
    print(f"  thickness >= {thk:5.1f} A ({n_t:3d} atoms): gamma = {g_t:.4f} J/m^2")

# ============================================================
# 5. Convergence test: vacuum for the (111) surface
# ============================================================
print("\n=== Vacuum Convergence for (111) ===")
vacuums = [8.0, 10.0, 12.0, 14.0, 16.0, 18.0, 20.0, 25.0]
gamma_vs_vacuum = []

for vac in vacuums:
    slabgen = SlabGenerator(
        initial_structure=relaxed_bulk,
        miller_index=(1, 1, 1),
        min_slab_size=MIN_SLAB_THICKNESS,
        min_vacuum_size=vac,
        center_slab=True,
        in_unit_planes=False,
        lll_reduce=True,
    )
    slabs = slabgen.get_slabs(symmetrize=False)
    if not slabs:
        continue

    atoms_v = adaptor.get_atoms(slabs[0])
    n_v = len(atoms_v)

    z_v = atoms_v.get_positions()[:, 2]
    z_uv = np.sort(np.unique(np.round(z_v, 2)))
    if len(z_uv) >= N_FIXED_LAYERS:
        fix_v = [i for i, z in enumerate(z_v) if z <= z_uv[N_FIXED_LAYERS - 1] + 0.1]
    else:
        fix_v = []

    atoms_v.set_constraint(FixAtoms(indices=fix_v))
    atoms_v.calc = calc
    BFGS(atoms_v, logfile=None).run(fmax=0.005, steps=300)

    e_v = atoms_v.get_potential_energy()
    area_v = np.linalg.norm(np.cross(atoms_v.cell[0], atoms_v.cell[1]))
    g_v = (e_v - n_v * e_bulk_per_atom) / (2 * area_v) * 16.0218

    gamma_vs_vacuum.append((vac, g_v))
    print(f"  vacuum = {vac:5.1f} A: gamma = {g_v:.4f} J/m^2")

# ============================================================
# 6. Plot: surface energy bar chart + convergence
# ============================================================
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# (a) Surface energy comparison across facets
if surface_results:
    labels = [f"({k})" for k in surface_results]
    gammas = [surface_results[k]["gamma_J_per_m2"] for k in surface_results]

    ax = axes[0]
    x = np.arange(len(labels))
    bars = ax.bar(x, gammas, color="steelblue", edgecolor="black", linewidth=0.5)
    min_idx = np.argmin(gammas)
    bars[min_idx].set_color("coral")

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=10, rotation=45, ha="right")
    ax.set_ylabel(r"$\gamma$ (J/m$^2$)", fontsize=12)
    ax.set_title(f"{MATERIAL_NAME} Surface Energies (MACE)", fontsize=13)
    ax.grid(axis="y", alpha=0.3)
    for i, (xi, gi) in enumerate(zip(x, gammas)):
        ax.text(xi, gi + 0.02, f"{gi:.3f}", ha="center", va="bottom", fontsize=8)

# (b) Thickness convergence
if gamma_vs_thickness:
    ax = axes[1]
    thks = [r[0] for r in gamma_vs_thickness]
    gams = [r[2] for r in gamma_vs_thickness]
    ax.plot(thks, gams, "o-", color="steelblue", markersize=6, linewidth=1.5)
    ax.set_xlabel("Min slab thickness (A)", fontsize=12)
    ax.set_ylabel(r"$\gamma$ (J/m$^2$)", fontsize=12)
    ax.set_title(f"{MATERIAL_NAME}(111) Thickness Convergence", fontsize=13)
    ax.grid(True, alpha=0.3)
    if len(gams) >= 3:
        ax.axhspan(gams[-1] - 0.05, gams[-1] + 0.05, alpha=0.1, color="green")

# (c) Vacuum convergence
if gamma_vs_vacuum:
    ax = axes[2]
    vacs = [r[0] for r in gamma_vs_vacuum]
    gams_v = [r[1] for r in gamma_vs_vacuum]
    ax.plot(vacs, gams_v, "s-", color="coral", markersize=6, linewidth=1.5)
    ax.set_xlabel("Vacuum thickness (A)", fontsize=12)
    ax.set_ylabel(r"$\gamma$ (J/m$^2$)", fontsize=12)
    ax.set_title(f"{MATERIAL_NAME}(111) Vacuum Convergence", fontsize=13)
    ax.grid(True, alpha=0.3)
    if len(gams_v) >= 3:
        ax.axhspan(gams_v[-1] - 0.05, gams_v[-1] + 0.05, alpha=0.1, color="green")

fig.tight_layout()
fig.savefig("surface_energy_analysis.png", dpi=150)
print("\nPlot saved to surface_energy_analysis.png")

# ============================================================
# 7. Save all results
# ============================================================
output = {
    "material": MATERIAL_NAME,
    "e_bulk_per_atom_eV": e_bulk_per_atom,
    "a_relaxed_A": a_relaxed,
    "surface_energies": surface_results,
    "convergence_thickness_111": [
        {"thickness_A": t, "n_atoms": n, "gamma_J_m2": g}
        for t, n, g in gamma_vs_thickness
    ],
    "convergence_vacuum_111": [
        {"vacuum_A": v, "gamma_J_m2": g}
        for v, g in gamma_vs_vacuum
    ],
}
with open("surface_energy_results.json", "w") as f:
    json.dump(output, f, indent=2, default=float)
print("Results saved to surface_energy_results.json")
```

#### Wulff Construction for Equilibrium Crystal Shape

```python
#!/usr/bin/env python3
"""
Build a Wulff construction from computed surface energies
to predict the equilibrium crystal shape.

Uses pymatgen.analysis.wulff.WulffShape.
"""

import numpy as np
import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from pymatgen.core import Lattice
from pymatgen.analysis.wulff import WulffShape

# ============================================================
# 1. Load surface energy results (or provide manually)
# ============================================================
# Option A: Load from previous calculation
try:
    with open("surface_energy_results.json") as f:
        data = json.load(f)
    surface_data = data["surface_energies"]
    lattice = Lattice.cubic(data["a_relaxed_A"])
    material_name = data["material"]
except FileNotFoundError:
    # Option B: Provide values manually (example: Al)
    surface_data = {
        "100": {"miller_index": [1, 0, 0], "gamma_J_per_m2": 0.98},
        "110": {"miller_index": [1, 1, 0], "gamma_J_per_m2": 1.09},
        "111": {"miller_index": [1, 1, 1], "gamma_J_per_m2": 0.85},
        "210": {"miller_index": [2, 1, 0], "gamma_J_per_m2": 1.12},
        "211": {"miller_index": [2, 1, 1], "gamma_J_per_m2": 1.02},
    }
    lattice = Lattice.cubic(4.05)
    material_name = "Al"

# ============================================================
# 2. Build Wulff shape
# ============================================================
miller_list = [tuple(v["miller_index"]) for v in surface_data.values()]
gamma_list = [v["gamma_J_per_m2"] for v in surface_data.values()]

wulff = WulffShape(
    lattice=lattice,
    miller_list=miller_list,
    e_surf_list=gamma_list,
)

print(f"=== Wulff Shape for {material_name} ===")
print(f"Weighted surface energy: {wulff.weighted_surface_energy:.4f} J/m^2")
print(f"Shape factor: {wulff.shape_factor:.4f}")
print(f"Effective radius: {wulff.effective_radius:.4f} A")
print(f"Surface area: {wulff.surface_area:.2f} A^2")
print(f"Volume: {wulff.volume:.2f} A^3")
print(f"Anisotropy: {wulff.anisotropy:.4f}")

print(f"\nFacet areas:")
for hkl, frac in zip(wulff.miller_list, wulff.area_fraction_dict.values()):
    print(f"  {hkl}: {frac*100:.1f}% of total area")

# ============================================================
# 3. Visualize the Wulff shape
# ============================================================
# pymatgen WulffShape has a built-in plotter
fig = wulff.get_plot()
fig.savefig("wulff_shape.png", dpi=150, bbox_inches="tight")
print("\nWulff shape saved to wulff_shape.png")

# Also make a simpler bar chart of area fractions
fig2, ax = plt.subplots(figsize=(8, 5))
hkl_labels = [str(hkl) for hkl in wulff.miller_list]
fractions = list(wulff.area_fraction_dict.values())

# Sort by fraction descending
sorted_pairs = sorted(zip(hkl_labels, fractions), key=lambda x: -x[1])
hkl_sorted = [p[0] for p in sorted_pairs]
frac_sorted = [p[1] * 100 for p in sorted_pairs]

colors = plt.cm.Set2(np.linspace(0, 1, len(hkl_sorted)))
bars = ax.barh(hkl_sorted, frac_sorted, color=colors, edgecolor="black", linewidth=0.5)
ax.set_xlabel("Area fraction (%)", fontsize=12)
ax.set_ylabel("Miller index", fontsize=12)
ax.set_title(f"{material_name} Wulff Shape -- Facet Distribution", fontsize=13)
ax.grid(axis="x", alpha=0.3)

for bar, pct in zip(bars, frac_sorted):
    if pct > 2:
        ax.text(bar.get_width() + 0.5, bar.get_y() + bar.get_height() / 2,
                f"{pct:.1f}%", va="center", fontsize=10)

fig2.tight_layout()
fig2.savefig("wulff_area_fractions.png", dpi=150)
print("Area fractions plot saved to wulff_area_fractions.png")

# Save Wulff data
wulff_output = {
    "material": material_name,
    "weighted_surface_energy_J_m2": float(wulff.weighted_surface_energy),
    "shape_factor": float(wulff.shape_factor),
    "anisotropy": float(wulff.anisotropy),
    "facets": {
        str(hkl): {"gamma_J_m2": g, "area_fraction": float(f)}
        for hkl, g, f in zip(
            wulff.miller_list, gamma_list,
            wulff.area_fraction_dict.values()
        )
    },
}
with open("wulff_results.json", "w") as f:
    json.dump(wulff_output, f, indent=2)
print("Wulff data saved to wulff_results.json")
```

### Method B: QE DFT Surface Energy

#### Step 1: Generate QE Inputs for Bulk and Slabs

```python
#!/usr/bin/env python3
"""
Generate QE input files for surface energy calculation (bulk + multiple slabs).
Example: Al (fcc) with (111), (100), (110) surfaces.
"""

import numpy as np
import os
from pymatgen.core import Structure, Lattice
from pymatgen.core.surface import SlabGenerator
from pymatgen.io.pwscf import PWInput
from pathlib import Path

# ============================================================
# Configuration
# ============================================================
PSEUDO_DIR = "./pseudo"
ECUTWFC = 40.0    # Ry (Al SSSP Efficiency)
ECUTRHO = 320.0   # Ry
Path(PSEUDO_DIR).mkdir(exist_ok=True)

pseudos = {"Al": "Al.pbe-n-kjpaw_psl.1.0.0.UPF"}
KPOINTS_BULK = (12, 12, 12)

# Build primitive Al fcc
primitive = Structure.from_spacegroup(
    "Fm-3m",
    lattice=Lattice.cubic(4.05),
    species=["Al"],
    coords=[[0.0, 0.0, 0.0]],
)

# ============================================================
# 1. Bulk SCF input (variable-cell relaxation for accurate reference)
# ============================================================
pw_bulk = PWInput(
    primitive,
    pseudo=pseudos,
    control={
        "calculation": "vc-relax",
        "restart_mode": "from_scratch",
        "pseudo_dir": PSEUDO_DIR,
        "outdir": "./tmp_bulk",
        "prefix": "Al_bulk",
        "tprnfor": True,
        "tstress": True,
        "etot_conv_thr": 1.0e-6,
        "forc_conv_thr": 1.0e-5,
    },
    system={
        "ecutwfc": ECUTWFC,
        "ecutrho": ECUTRHO,
        "occupations": "smearing",
        "smearing": "cold",
        "degauss": 0.02,
    },
    electrons={"conv_thr": 1.0e-8, "mixing_beta": 0.4},
    kpoints_grid=KPOINTS_BULK,
)
pw_bulk.write_file("pw_bulk.in")
print(f"Written pw_bulk.in ({len(primitive)} atoms, kpts={KPOINTS_BULK})")

# ============================================================
# 2. Slab inputs for multiple facets
# ============================================================
miller_indices = [(1, 0, 0), (1, 1, 0), (1, 1, 1)]
SLAB_THICKNESS = 14.0   # Angstrom
VACUUM = 16.0            # Angstrom

for hkl in miller_indices:
    hkl_str = "".join(map(str, hkl))
    print(f"\n--- Generating Al({hkl_str}) slab ---")

    slabgen = SlabGenerator(
        initial_structure=primitive,
        miller_index=hkl,
        min_slab_size=SLAB_THICKNESS,
        min_vacuum_size=VACUUM,
        center_slab=True,
        in_unit_planes=False,
        lll_reduce=True,
        reorient_lattice=True,
    )
    slabs = slabgen.get_slabs(symmetrize=False)
    if not slabs:
        print(f"  No slabs generated for ({hkl_str})")
        continue

    slab = slabs[0]
    n_atoms = len(slab)

    # Determine k-points: maintain similar in-plane density as bulk
    # bulk has 12x12x12 for a ~4 A cell -> density ~ 3/A
    a_slab = slab.lattice.a
    b_slab = slab.lattice.b
    kx = max(1, round(12 * primitive.lattice.a / a_slab))
    ky = max(1, round(12 * primitive.lattice.a / b_slab))
    kpts_slab = (kx, ky, 1)

    pw_slab = PWInput(
        slab,
        pseudo=pseudos,
        control={
            "calculation": "relax",
            "restart_mode": "from_scratch",
            "pseudo_dir": PSEUDO_DIR,
            "outdir": f"./tmp_slab_{hkl_str}",
            "prefix": f"Al_{hkl_str}",
            "tprnfor": True,
            "tstress": True,
            "etot_conv_thr": 1.0e-6,
            "forc_conv_thr": 1.0e-4,
        },
        system={
            "ecutwfc": ECUTWFC,
            "ecutrho": ECUTRHO,
            "occupations": "smearing",
            "smearing": "cold",
            "degauss": 0.02,
        },
        electrons={"conv_thr": 1.0e-8, "mixing_beta": 0.3},
        kpoints_grid=kpts_slab,
    )
    pw_slab.write_file(f"pw_slab_{hkl_str}.in")
    print(f"  Written pw_slab_{hkl_str}.in ({n_atoms} atoms, kpts={kpts_slab})")
    print(f"  Area = {np.linalg.norm(np.cross(slab.lattice.matrix[0], slab.lattice.matrix[1])):.2f} A^2")

print("\nNOTE: After generating inputs, add selective dynamics to fix bottom layers.")
print("See the helper script in surface-adsorption/ or use:")
print("  Add '0 0 0' (fixed) / '1 1 1' (free) flags after ATOMIC_POSITIONS lines.")
```

#### Step 2: Run QE Calculations

```bash
#!/bin/bash
# Run bulk and slab QE calculations for surface energy

NPROC=4

# Download pseudopotentials
mkdir -p pseudo tmp_bulk
cd pseudo
wget -nc "https://pseudopotentials.quantum-espresso.org/upf_files/Al.pbe-n-kjpaw_psl.1.0.0.UPF" 2>/dev/null
cd ..

# 1. Bulk vc-relax
echo "=== Running bulk vc-relax ==="
mpirun -np $NPROC pw.x -in pw_bulk.in > pw_bulk.out 2>&1
echo "Bulk: $(grep '!' pw_bulk.out | tail -1)"

# 2. Slab relaxations
for HKL in 100 110 111; do
    mkdir -p tmp_slab_$HKL
    echo "=== Running Al($HKL) slab ==="
    mpirun -np $NPROC pw.x -in pw_slab_$HKL.in > pw_slab_$HKL.out 2>&1
    echo "Slab $HKL: $(grep '!' pw_slab_$HKL.out | tail -1)"
done
```

#### Step 3: Parse Results and Compute Surface Energies

```python
#!/usr/bin/env python3
"""
Parse QE outputs for bulk and slab calculations.
Compute surface energies and plot comparison.
"""

import re
import numpy as np
import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from pymatgen.core import Lattice
from pymatgen.analysis.wulff import WulffShape


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
    """Extract number of atoms from QE output."""
    with open(filename) as f:
        for line in f:
            if "number of atoms/cell" in line:
                return int(line.split("=")[1].strip())
    return None


def parse_qe_cell(filename):
    """Extract cell parameters from QE output (last set)."""
    cell = []
    reading = False
    with open(filename) as f:
        for line in f:
            if "CELL_PARAMETERS" in line:
                cell = []
                reading = True
                continue
            if reading and line.strip():
                parts = line.split()
                if len(parts) == 3:
                    cell.append([float(x) for x in parts])
                    if len(cell) == 3:
                        reading = False
    return np.array(cell) if cell else None


# Parse bulk
e_bulk = parse_qe_energy("pw_bulk.out")
n_bulk = parse_qe_natoms("pw_bulk.out")
e_bulk_per_atom = e_bulk / n_bulk
print(f"Bulk: E = {e_bulk:.6f} eV, {n_bulk} atoms, E/atom = {e_bulk_per_atom:.6f} eV")

# Parse slabs
miller_indices = ["100", "110", "111"]
results = {}

for hkl in miller_indices:
    outfile = f"pw_slab_{hkl}.out"
    e_slab = parse_qe_energy(outfile)
    n_slab = parse_qe_natoms(outfile)

    if e_slab is None:
        print(f"({hkl}): could not parse energy from {outfile}")
        continue

    # Get cell for area calculation
    cell = parse_qe_cell(outfile)
    if cell is None:
        # Try to get from input file
        from pymatgen.io.pwscf import PWInput
        pw = PWInput.from_file(f"pw_slab_{hkl}.in")
        cell = pw.structure.lattice.matrix

    area = np.linalg.norm(np.cross(cell[0], cell[1]))
    gamma_eV = (e_slab - n_slab * e_bulk_per_atom) / (2 * area)
    gamma_J = gamma_eV * 16.0218

    results[hkl] = {
        "e_slab_eV": e_slab,
        "n_atoms": n_slab,
        "area_A2": float(area),
        "gamma_eV_per_A2": float(gamma_eV),
        "gamma_J_per_m2": float(gamma_J),
    }
    print(f"({hkl}): gamma = {gamma_J:.4f} J/m^2 "
          f"(E_slab = {e_slab:.4f} eV, {n_slab} atoms, A = {area:.2f} A^2)")

# Plot
if results:
    fig, ax = plt.subplots(figsize=(7, 5))
    labels = [f"({k})" for k in results]
    gammas = [results[k]["gamma_J_per_m2"] for k in results]

    bars = ax.bar(labels, gammas, color="steelblue", edgecolor="black")
    min_idx = np.argmin(gammas)
    bars[min_idx].set_color("coral")

    ax.set_ylabel(r"$\gamma$ (J/m$^2$)", fontsize=12)
    ax.set_title("Al Surface Energies (QE DFT)", fontsize=13)
    ax.grid(axis="y", alpha=0.3)
    for bar, g in zip(bars, gammas):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                f"{g:.3f}", ha="center", va="bottom", fontsize=10)

    fig.tight_layout()
    fig.savefig("surface_energy_dft.png", dpi=150)
    print("\nPlot saved to surface_energy_dft.png")

with open("surface_energy_dft_results.json", "w") as f:
    json.dump({"e_bulk_per_atom": e_bulk_per_atom, "slabs": results}, f, indent=2)
print("Results saved to surface_energy_dft_results.json")
```

### Method C: VASP Workflow (Future)

```
VASP surface energy workflow (requires VASP license):

1. Bulk reference:
   ISIF = 3 (full relaxation), IBRION = 2, NSW = 100
   PREC = Accurate, ENCUT = 400 (or 1.3x ENMAX)
   Dense k-mesh (e.g., 12x12x12 for fcc)

2. Slab calculation:
   ISIF = 2 (fix cell, relax ions), IBRION = 2, NSW = 200
   Same ENCUT, PREC as bulk
   K-points: Nx Ny 1 (only 1 k-point along vacuum)
   Use selective dynamics to fix bottom layers:
     In POSCAR: "Selective dynamics" header, then T/F flags per atom

3. Compute surface energy:
   gamma = (E_slab - N * E_bulk/atom) / (2 * A)
   A from CONTCAR: |a x b|

4. Multiple facets:
   Generate POSCAR files for each (hkl) with pymatgen SlabGenerator
   Run VASP for each slab
   Collect energies and compute gamma for each facet

5. Wulff construction:
   Use pymatgen WulffShape with the computed gamma values (same script as above)

6. Convergence:
   - Slab thickness: vary POSCAR slab layers, converge gamma to < 0.01 J/m^2
   - Vacuum: vary vacuum thickness, converge to < 0.01 J/m^2
   - K-points: test Nx Ny 1 grids until gamma converges
   - ENCUT: should already be converged from bulk

VASP-specific settings for polar surfaces:
   LDIPOL = .TRUE., IDIPOL = 3  (dipole correction along z)
   LREAL = Auto (for large slabs)
```

## Key Parameters

| Parameter | Recommended Value | Notes |
|---|---|---|
| Slab thickness | 10-20 A (5-10 layers) | Converge until gamma changes by < 0.01 J/m^2 between successive thicknesses |
| Vacuum thickness | 14-20 A | Prevents slab-slab interaction; 15 A is usually safe |
| Fixed bottom layers | 2-3 layers | Mimics bulk interior; more for thicker slabs |
| MACE fmax | 0.005 eV/A | Tight enough for accurate surface energies |
| QE ecutwfc | 40-80 Ry | Must match bulk calculation; use SSSP recommendations |
| K-points (slab) | Nx Ny 1 | Dense in-plane, single k-point along vacuum (c-axis) |
| K-points (bulk) | Dense (e.g., 12x12x12) | Must be well converged since E_bulk/atom appears in gamma |
| center_slab | True | Centers slab in vacuum for symmetric potential profile |
| symmetrize | False for adsorption; True for pure gamma | Symmetric slabs avoid dipole issues but double the cost |
| Dipole correction | Recommended for polar/asymmetric slabs | QE: `dipfield=.true.`; VASP: `LDIPOL=.TRUE., IDIPOL=3` |

## Interpreting Results

1. **Typical surface energies**: Metals 1-3 J/m^2 (close-packed lower, open higher). Oxides 0.5-2.5 J/m^2. Semiconductors 0.5-2 J/m^2.
2. **FCC metals ordering**: gamma(111) < gamma(100) < gamma(110) is typical. Close-packed surfaces are most stable.
3. **BCC metals ordering**: gamma(110) < gamma(100) < gamma(111). The (110) plane is close-packed for BCC.
4. **Wulff shape**: Facets with lowest gamma dominate the equilibrium shape. A facet that does not appear on the Wulff shape has gamma too high relative to neighboring orientations.
5. **Negative surface energy**: Always indicates an error. Common causes: inconsistent bulk/slab parameters (different pseudopotentials, ecutwfc, or k-point density), or the slab is too thin.
6. **Relaxation effect**: Relaxed gamma is always <= unrelaxed (cleavage) gamma. The difference quantifies surface reconstruction energy.
7. **MACE vs. DFT**: MACE values are typically within 10-30% of DFT for absolute gamma. Relative ordering of facets is usually preserved. Always validate with DFT for publication.
8. **Convergence criterion**: Surface energy should be converged to within 0.05 J/m^2 (preferably 0.01 J/m^2) with respect to slab thickness, vacuum, and k-points.

## Common Issues

| Issue | Solution |
|---|---|
| Negative surface energy | Ensure bulk and slab use identical pseudopotentials, ecutwfc, ecutrho. Re-check bulk E/atom. |
| Surface energy oscillates with slab thickness | Common for metals with quantum size effects. Use thick enough slabs (> 8 layers) or use linear fit of E_slab vs N_atoms (slope gives E_bulk/atom self-consistently). |
| Polar surface gives divergent energy | Use symmetric slab termination or apply dipole correction (`dipfield=.true.`). Consider non-stoichiometric slabs. |
| SlabGenerator gives wrong termination | Inspect all slabs from `get_slabs()`. Select the one with the desired surface chemistry. |
| SCF convergence issues for metallic slab | Use `smearing='cold'`, increase degauss to 0.02-0.03 Ry, reduce mixing_beta to 0.2-0.3. |
| Inconsistent k-point density bulk vs. slab | Maintain same k-point density per reciprocal length. For bulk 12x12x12 with a~4 A, use proportionally scaled in-plane grid for slab. |
| Wulff shape has only one facet | Check that all gamma values are correct. If one is anomalously low, it dominates. May indicate an error in that slab calculation. |
| MACE gives unphysical surface energy for unusual material | System outside MACE-MP-0 training set. Fall back to DFT. |
