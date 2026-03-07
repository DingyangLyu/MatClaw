# Wulff Construction for Equilibrium Crystal Shape

## When to Use

- You need the equilibrium crystal shape (Wulff shape) predicted from surface energies.
- You want to know which crystal facets dominate the equilibrium morphology.
- You are predicting nanoparticle shapes or crystal growth habits.
- You need facet area fractions for catalysis studies (which surfaces are exposed).
- You want to compute the shape factor or effective surface energy of a nanocrystal.

## Method Selection

| Input Source | When |
|---|---|
| MACE surface energies | Fast screening; qualitative shape prediction |
| QE DFT surface energies | Publication-quality Wulff shapes |
| Literature values | When surface energies are already known |

## Prerequisites

- Surface energies for multiple facets (compute using `surface-energy-calc/` skill or use literature values).
- Python: `pymatgen` (WulffShape class), `matplotlib`, `numpy`.

---

## Detailed Steps

### Step 1: Compute Surface Energies for Multiple Facets (MACE)

This step uses the surface energy calculation from the `surface-energy-calc/` skill. If you already have surface energies, skip to Step 2.

```python
#!/usr/bin/env python3
"""
Compute surface energies for multiple facets using MACE,
then build the Wulff shape with pymatgen.
"""
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from ase.build import bulk
from ase.optimize import LBFGS
from ase.filters import FrechetCellFilter
from ase.constraints import FixAtoms
from mace.calculators import mace_mp

from pymatgen.core import Structure, Lattice
from pymatgen.core.surface import SlabGenerator
from pymatgen.io.ase import AseAtomsAdaptor
from pymatgen.analysis.wulff import WulffShape

# ══════════════════════════════════════════════════════════════════════
# Configuration
# ══════════════════════════════════════════════════════════════════════
MATERIAL = "Cu"
STRUCTURE_TYPE = "fcc"
LATTICE_PARAM = 3.615

MILLER_INDICES = [
    (1, 0, 0),
    (1, 1, 0),
    (1, 1, 1),
    (2, 1, 0),
    (2, 1, 1),
    (3, 1, 0),
    (3, 1, 1),
]

SLAB_THICKNESS = 15.0  # Angstrom
VACUUM = 15.0
N_FIXED_LAYERS = 2

# ══════════════════════════════════════════════════════════════════════
# Step 1a: Bulk energy
# ══════════════════════════════════════════════════════════════════════
calc = mace_mp(model="medium", dispersion=False, device="cpu")
atoms_bulk = bulk(MATERIAL, STRUCTURE_TYPE, a=LATTICE_PARAM)
atoms_bulk.calc = calc
filtered = FrechetCellFilter(atoms_bulk)
opt = LBFGS(filtered, logfile="/dev/null")
opt.run(fmax=0.001)
e_bulk_per_atom = atoms_bulk.get_potential_energy() / len(atoms_bulk)

adaptor = AseAtomsAdaptor()
struct_bulk = adaptor.get_structure(atoms_bulk)

print(f"Material: {MATERIAL} ({STRUCTURE_TYPE})")
print(f"Bulk energy per atom: {e_bulk_per_atom:.6f} eV")
print(f"Relaxed lattice param: {atoms_bulk.cell.cellpar()[0]:.4f} Ang")

# ══════════════════════════════════════════════════════════════════════
# Step 1b: Surface energies for each facet
# ══════════════════════════════════════════════════════════════════════
print(f"\nComputing surface energies...")
print(f"{'Miller':>12s} | {'N_atoms':>7s} | {'Area':>10s} | {'gamma (J/m^2)':>14s}")
print("-" * 55)

surface_energies = {}  # {(h,k,l): gamma in J/m^2}

for hkl in MILLER_INDICES:
    try:
        sg = SlabGenerator(
            initial_structure=struct_bulk,
            miller_index=hkl,
            min_slab_size=SLAB_THICKNESS,
            min_vacuum_size=VACUUM,
            center_slab=True,
            in_unit_planes=False,
            lll_reduce=True,
            reorient_lattice=True,
        )
        slabs = sg.get_slabs(symmetrize=False)
        if len(slabs) == 0:
            print(f"  {str(hkl):>12s} | {'SKIP':>7s} | {'---':>10s} | {'N/A':>14s}")
            continue

        slab_atoms = adaptor.get_atoms(slabs[0])
        n_at = len(slab_atoms)

        # Fix bottom layers
        z = slab_atoms.positions[:, 2]
        z_unique = np.sort(np.unique(np.round(z, decimals=2)))
        if len(z_unique) >= N_FIXED_LAYERS:
            z_thr = z_unique[N_FIXED_LAYERS - 1] + 0.1
            fix_idx = [i for i, zi in enumerate(z) if zi <= z_thr]
        else:
            fix_idx = []

        slab_atoms.set_constraint(FixAtoms(indices=fix_idx))
        slab_atoms.calc = calc
        opt = LBFGS(slab_atoms, logfile="/dev/null")
        opt.run(fmax=0.005)

        e_slab = slab_atoms.get_potential_energy()
        cell = slab_atoms.cell
        area = np.linalg.norm(np.cross(cell[0], cell[1]))

        gamma = (e_slab - n_at * e_bulk_per_atom) / (2 * area) * 16.0217663
        surface_energies[hkl] = gamma

        print(f"  {str(hkl):>12s} | {n_at:>7d} | {area:>10.2f} | {gamma:>14.4f}")

    except Exception as e:
        print(f"  {str(hkl):>12s} | ERROR: {e}")

if len(surface_energies) == 0:
    raise RuntimeError("No surface energies computed! Check inputs.")

print(f"\nComputed {len(surface_energies)} surface energies.")
```

### Step 2: Build the Wulff Shape

```python
# ══════════════════════════════════════════════════════════════════════
# Step 2: Build the Wulff shape with pymatgen
# ══════════════════════════════════════════════════════════════════════

# pymatgen WulffShape requires:
#   lattice: the crystal lattice
#   miller_list: list of Miller index tuples
#   e_surf_list: list of surface energies (same order)

lattice = struct_bulk.lattice

miller_list = list(surface_energies.keys())
e_surf_list = [surface_energies[hkl] for hkl in miller_list]

print("\n" + "=" * 60)
print("Building Wulff Shape")
print("=" * 60)

wulff = WulffShape(lattice, miller_list, e_surf_list)

# ── Extract properties ──────────────────────────────────────────────
print(f"\nWulff Shape Properties:")
print(f"  Weighted surface energy:  {wulff.weighted_surface_energy:.4f} J/m^2")
print(f"  Shape factor:             {wulff.shape_factor:.4f}")
print(f"  Anisotropy:               {wulff.anisotropy:.4f}")
print(f"  Effective radius (arb.):  {wulff.effective_radius:.4f}")
print(f"  Total surface area:       {wulff.total_surface_area:.4f} (arb. units)")
print(f"  Volume:                   {wulff.volume:.4f} (arb. units)")

# ── Facet areas and fractions ───────────────────────────────────────
print(f"\nFacet Analysis:")
print(f"  {'Miller':>12s} | {'gamma (J/m^2)':>14s} | {'Area Fraction':>14s}")
print("  " + "-" * 50)

area_fractions = wulff.area_fraction_dict
for hkl in miller_list:
    gamma_val = surface_energies[hkl]
    # area_fraction_dict uses pymatgen Miller index tuple format
    frac = area_fractions.get(hkl, 0.0)
    marker = " <-- dominant" if frac > 0.25 else ""
    print(f"  {str(hkl):>12s} | {gamma_val:>14.4f} | {frac:>14.4f}{marker}")

# Identify dominant facets
dominant = [(hkl, area_fractions.get(hkl, 0.0)) for hkl in miller_list]
dominant.sort(key=lambda x: x[1], reverse=True)
print(f"\n  Dominant facet: {dominant[0][0]} ({dominant[0][1]*100:.1f}% of surface)")
if len(dominant) > 1 and dominant[1][1] > 0.01:
    print(f"  Second facet:   {dominant[1][0]} ({dominant[1][1]*100:.1f}% of surface)")
```

### Step 3: Plot the 3D Wulff Shape

```python
# ══════════════════════════════════════════════════════════════════════
# Step 3: Plot the 3D Wulff shape
# ══════════════════════════════════════════════════════════════════════

# Method A: Use pymatgen's built-in plotting (saves to file)
fig_wulff = wulff.get_plot()
fig_wulff.savefig("wulff_shape_3d.png", dpi=200, bbox_inches="tight")
print("\nSaved: wulff_shape_3d.png (3D Wulff shape)")
plt.close(fig_wulff)

# Method B: Custom multi-panel figure with additional analysis
fig = plt.figure(figsize=(16, 6))

# Panel 1: Bar chart of surface energies
ax1 = fig.add_subplot(131)
labels = [str(hkl) for hkl in miller_list]
gammas = [surface_energies[hkl] for hkl in miller_list]
colors = plt.cm.RdYlGn_r(np.linspace(0.2, 0.8, len(gammas)))
# Sort by energy for better visualization
sorted_idx = np.argsort(gammas)
sorted_labels = [labels[i] for i in sorted_idx]
sorted_gammas = [gammas[i] for i in sorted_idx]
sorted_colors = [colors[i] for i in sorted_idx]

bars = ax1.barh(range(len(sorted_labels)), sorted_gammas, color=sorted_colors,
                edgecolor="black", linewidth=0.5)
ax1.set_yticks(range(len(sorted_labels)))
ax1.set_yticklabels(sorted_labels, fontsize=11)
ax1.set_xlabel(r"$\gamma$ (J/m$^2$)", fontsize=12)
ax1.set_title("Surface Energies", fontsize=13)
ax1.grid(axis="x", alpha=0.3)

# Panel 2: Pie chart of area fractions
ax2 = fig.add_subplot(132)
fracs = [area_fractions.get(hkl, 0.0) for hkl in miller_list]
# Filter out zero-area facets
nonzero = [(hkl, f) for hkl, f in zip(miller_list, fracs) if f > 0.005]
if nonzero:
    pie_labels = [str(hkl) for hkl, _ in nonzero]
    pie_fracs = [f for _, f in nonzero]
    pie_colors = plt.cm.Set3(np.linspace(0, 1, len(pie_labels)))
    wedges, texts, autotexts = ax2.pie(
        pie_fracs, labels=pie_labels, autopct="%1.1f%%",
        colors=pie_colors, startangle=90,
        textprops={"fontsize": 10}
    )
    for autotext in autotexts:
        autotext.set_fontsize(9)
ax2.set_title("Facet Area Fractions", fontsize=13)

# Panel 3: Wulff shape (3D)
ax3 = fig.add_subplot(133, projection="3d")
# Use pymatgen's internal data to plot
# The WulffShape stores facet vertices which we can extract
for facet in wulff.facets:
    hkl = facet.miller_index
    # Get vertices of this facet
    vertices = np.array(facet.points)
    if len(vertices) < 3:
        continue

    # Close the polygon
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection
    # Sort vertices to form a proper polygon (convex hull on the facet plane)
    center = vertices.mean(axis=0)
    # Project to 2D for sorting
    v1 = vertices[1] - vertices[0]
    v1 = v1 / np.linalg.norm(v1)
    normal = np.cross(vertices[1] - vertices[0], vertices[2] - vertices[0])
    if np.linalg.norm(normal) < 1e-10:
        continue
    normal = normal / np.linalg.norm(normal)
    v2 = np.cross(normal, v1)

    angles = []
    for v in vertices:
        d = v - center
        angle = np.arctan2(np.dot(d, v2), np.dot(d, v1))
        angles.append(angle)
    order = np.argsort(angles)
    sorted_verts = vertices[order]

    # Color by surface energy
    gamma_hkl = surface_energies.get(hkl, 0)
    gamma_min = min(surface_energies.values())
    gamma_max = max(surface_energies.values())
    if gamma_max > gamma_min:
        norm_val = (gamma_hkl - gamma_min) / (gamma_max - gamma_min)
    else:
        norm_val = 0.5
    color = plt.cm.RdYlGn_r(norm_val * 0.6 + 0.2)

    poly = Poly3DCollection([sorted_verts], alpha=0.7)
    poly.set_facecolor(color)
    poly.set_edgecolor("black")
    poly.set_linewidth(0.5)
    ax3.add_collection3d(poly)

# Set axis limits
all_pts = []
for facet in wulff.facets:
    all_pts.extend(facet.points)
if all_pts:
    all_pts = np.array(all_pts)
    maxval = np.max(np.abs(all_pts)) * 1.1
    ax3.set_xlim(-maxval, maxval)
    ax3.set_ylim(-maxval, maxval)
    ax3.set_zlim(-maxval, maxval)

ax3.set_xlabel("x")
ax3.set_ylabel("y")
ax3.set_zlabel("z")
ax3.set_title("Wulff Shape", fontsize=13)

plt.suptitle(f"{MATERIAL} Equilibrium Crystal Shape (Wulff Construction)", fontsize=15, y=1.02)
plt.tight_layout()
plt.savefig("wulff_analysis.png", dpi=200, bbox_inches="tight")
print("Saved: wulff_analysis.png (multi-panel analysis)")
plt.close()
```

### Complete Single-Script Workflow

```python
#!/usr/bin/env python3
"""
Complete Wulff construction workflow:
  1. Compute surface energies with MACE
  2. Build Wulff shape with pymatgen
  3. Plot and analyze results
"""
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from ase.build import bulk
from ase.optimize import LBFGS
from ase.filters import FrechetCellFilter
from ase.constraints import FixAtoms
from mace.calculators import mace_mp

from pymatgen.core import Structure
from pymatgen.core.surface import SlabGenerator
from pymatgen.io.ase import AseAtomsAdaptor
from pymatgen.analysis.wulff import WulffShape

# ══════════════════════════════════════════════════════════════════════
# Configuration -- EDIT THESE FOR YOUR SYSTEM
# ══════════════════════════════════════════════════════════════════════
MATERIAL = "Cu"
STRUCTURE_TYPE = "fcc"
LATTICE_PARAM = 3.615
MILLER_INDICES = [(1,0,0), (1,1,0), (1,1,1), (2,1,0), (2,1,1), (3,1,0)]
SLAB_THICKNESS = 15.0
VACUUM = 15.0
N_FIXED_LAYERS = 2

# ══════════════════════════════════════════════════════════════════════
# Step 1: Bulk reference
# ══════════════════════════════════════════════════════════════════════
print(f"Wulff Construction for {MATERIAL}")
print("=" * 60)

calc = mace_mp(model="medium", dispersion=False, device="cpu")
atoms_bulk = bulk(MATERIAL, STRUCTURE_TYPE, a=LATTICE_PARAM)
atoms_bulk.calc = calc
opt = LBFGS(FrechetCellFilter(atoms_bulk), logfile="/dev/null")
opt.run(fmax=0.001)
e_bulk_per_atom = atoms_bulk.get_potential_energy() / len(atoms_bulk)
adaptor = AseAtomsAdaptor()
struct_bulk = adaptor.get_structure(atoms_bulk)

print(f"Bulk energy/atom: {e_bulk_per_atom:.6f} eV\n")

# ══════════════════════════════════════════════════════════════════════
# Step 2: Surface energies
# ══════════════════════════════════════════════════════════════════════
surface_energies = {}

for hkl in MILLER_INDICES:
    try:
        sg = SlabGenerator(
            initial_structure=struct_bulk,
            miller_index=hkl,
            min_slab_size=SLAB_THICKNESS,
            min_vacuum_size=VACUUM,
            center_slab=True,
            in_unit_planes=False,
            lll_reduce=True,
            reorient_lattice=True,
        )
        slabs = sg.get_slabs(symmetrize=False)
        if not slabs:
            continue

        slab_atoms = adaptor.get_atoms(slabs[0])
        n_at = len(slab_atoms)

        z = slab_atoms.positions[:, 2]
        z_unique = np.sort(np.unique(np.round(z, decimals=2)))
        if len(z_unique) >= N_FIXED_LAYERS:
            z_thr = z_unique[N_FIXED_LAYERS - 1] + 0.1
            fix_idx = [i for i, zi in enumerate(z) if zi <= z_thr]
        else:
            fix_idx = []

        slab_atoms.set_constraint(FixAtoms(indices=fix_idx))
        slab_atoms.calc = calc
        opt = LBFGS(slab_atoms, logfile="/dev/null")
        opt.run(fmax=0.005)

        e_slab = slab_atoms.get_potential_energy()
        area = np.linalg.norm(np.cross(slab_atoms.cell[0], slab_atoms.cell[1]))
        gamma = (e_slab - n_at * e_bulk_per_atom) / (2 * area) * 16.0217663

        surface_energies[hkl] = gamma
        print(f"  {str(hkl):>12s}: gamma = {gamma:.4f} J/m^2 ({n_at} atoms)")

    except Exception as e:
        print(f"  {str(hkl):>12s}: FAILED ({e})")

# ══════════════════════════════════════════════════════════════════════
# Step 3: Wulff shape
# ══════════════════════════════════════════════════════════════════════
miller_list = list(surface_energies.keys())
e_surf_list = [surface_energies[hkl] for hkl in miller_list]

wulff = WulffShape(struct_bulk.lattice, miller_list, e_surf_list)

print(f"\n{'='*60}")
print(f"Wulff Shape Results")
print(f"{'='*60}")
print(f"  Weighted surface energy: {wulff.weighted_surface_energy:.4f} J/m^2")
print(f"  Shape factor:            {wulff.shape_factor:.4f}")
print(f"  Anisotropy:              {wulff.anisotropy:.4f}")

area_fracs = wulff.area_fraction_dict
print(f"\n  Facet area fractions:")
for hkl in miller_list:
    frac = area_fracs.get(hkl, 0.0)
    bar = "#" * int(frac * 50)
    print(f"    {str(hkl):>12s}: {frac:6.3f}  {bar}")

# ══════════════════════════════════════════════════════════════════════
# Step 4: Plots
# ══════════════════════════════════════════════════════════════════════

# 4a: Wulff shape (pymatgen built-in)
fig_wulff = wulff.get_plot()
fig_wulff.savefig("wulff_shape.png", dpi=200, bbox_inches="tight")
plt.close(fig_wulff)
print(f"\nSaved: wulff_shape.png")

# 4b: Summary figure
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# Surface energy bar chart
ax1 = axes[0]
sorted_hkl = sorted(surface_energies.keys(), key=lambda h: surface_energies[h])
sorted_labels = [str(h) for h in sorted_hkl]
sorted_gammas = [surface_energies[h] for h in sorted_hkl]
colors = plt.cm.viridis(np.linspace(0.2, 0.9, len(sorted_gammas)))
ax1.barh(range(len(sorted_labels)), sorted_gammas, color=colors,
         edgecolor="black", linewidth=0.5)
ax1.set_yticks(range(len(sorted_labels)))
ax1.set_yticklabels(sorted_labels, fontsize=11)
ax1.set_xlabel(r"$\gamma$ (J/m$^2$)", fontsize=12)
ax1.set_title("Surface Energies", fontsize=13)
ax1.grid(axis="x", alpha=0.3)

# Area fraction pie chart
ax2 = axes[1]
nonzero_facets = [(str(hkl), area_fracs.get(hkl, 0.0))
                  for hkl in miller_list if area_fracs.get(hkl, 0.0) > 0.005]
if nonzero_facets:
    pie_labels, pie_values = zip(*nonzero_facets)
    pie_colors = plt.cm.Set2(np.linspace(0, 1, len(pie_labels)))
    ax2.pie(pie_values, labels=pie_labels, autopct="%1.1f%%",
            colors=pie_colors, startangle=90, textprops={"fontsize": 10})
ax2.set_title("Equilibrium Facet Areas", fontsize=13)

plt.suptitle(f"{MATERIAL} Wulff Construction (MACE)", fontsize=14)
plt.tight_layout()
plt.savefig("wulff_summary.png", dpi=200, bbox_inches="tight")
print("Saved: wulff_summary.png")
plt.close()

# ══════════════════════════════════════════════════════════════════════
# Step 5: Nanoparticle morphology analysis
# ══════════════════════════════════════════════════════════════════════
print(f"\n{'='*60}")
print(f"Nanoparticle Morphology Analysis")
print(f"{'='*60}")

# Effective surface energy for a nanoparticle of radius r:
# Total surface energy = gamma_eff * Surface_area
gamma_eff = wulff.weighted_surface_energy
print(f"  Effective (weighted) surface energy: {gamma_eff:.4f} J/m^2")
print(f"  Shape factor (sphere=1): {wulff.shape_factor:.4f}")
print(f"    (shape factor > 1 means the Wulff shape has higher")
print(f"     surface-to-volume ratio than a sphere)")

# For FCC metals, the typical Wulff shape is a truncated octahedron
# dominated by {111} and {100} facets.
dominant_facets = sorted(
    [(hkl, area_fracs.get(hkl, 0.0)) for hkl in miller_list],
    key=lambda x: x[1], reverse=True
)

print(f"\n  Dominant facets (>1% area):")
for hkl, frac in dominant_facets:
    if frac > 0.01:
        print(f"    {str(hkl):>12s}: {frac*100:5.1f}%  gamma = {surface_energies[hkl]:.4f} J/m^2")

# Estimate the nanoparticle regime where shape is Wulff-like
# Wulff shape is valid when surface energy dominates over edge/corner terms
# Roughly valid for particles > ~3 nm
print(f"\n  Applicability:")
print(f"    Wulff shape is thermodynamically valid for nanoparticles")
print(f"    larger than approximately 2-5 nm, where surface energy")
print(f"    dominates over edge and corner energy contributions.")
print(f"    For smaller clusters, atomistic calculations are needed.")
print(f"\n  Catalytic implications:")
for hkl, frac in dominant_facets[:3]:
    if frac > 0.01:
        hkl_str = "".join(str(i) for i in hkl)
        print(f"    {hkl}: {frac*100:.0f}% exposed -- consider {MATERIAL}({hkl_str}) surface reactivity")

print(f"\nDone.")
```

### Using Literature Surface Energies

If you already have surface energies (from DFT literature or your own calculations), you can skip the MACE computation and directly build the Wulff shape:

```python
#!/usr/bin/env python3
"""
Build Wulff shape from known surface energies (no MACE needed).
Example: Gold (Au) FCC with literature DFT-PBE values.
"""
import warnings
warnings.filterwarnings("ignore")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from pymatgen.core import Lattice
from pymatgen.analysis.wulff import WulffShape

# ── Define the lattice ──────────────────────────────────────────────
# Gold FCC, a = 4.08 Angstrom
lattice = Lattice.cubic(4.08)

# ── Surface energies from literature (J/m^2) ───────────────────────
# Example values (DFT-PBE) for Au:
# Reference: Vitos et al., Surf. Sci. 411, 186 (1998) or similar
surface_energies = {
    (1, 1, 1): 0.74,  # close-packed, lowest energy
    (1, 0, 0): 0.89,
    (1, 1, 0): 0.98,
    (2, 1, 0): 1.02,
    (2, 1, 1): 0.95,
    (3, 1, 0): 1.05,
    (3, 1, 1): 0.97,
    (3, 3, 1): 0.82,
}

miller_list = list(surface_energies.keys())
e_surf_list = [surface_energies[hkl] for hkl in miller_list]

# Build Wulff shape
wulff = WulffShape(lattice, miller_list, e_surf_list)

# Print results
print("Au Wulff Shape from Literature Values")
print("=" * 50)
print(f"Weighted surface energy: {wulff.weighted_surface_energy:.4f} J/m^2")
print(f"Shape factor: {wulff.shape_factor:.4f}")
print(f"Anisotropy: {wulff.anisotropy:.4f}")

area_fracs = wulff.area_fraction_dict
print(f"\nFacet area fractions:")
for hkl in miller_list:
    frac = area_fracs.get(hkl, 0.0)
    print(f"  {str(hkl):>12s}: {frac:.4f}  ({frac*100:.1f}%)")

# Plot
fig = wulff.get_plot()
fig.savefig("wulff_au_literature.png", dpi=200, bbox_inches="tight")
plt.close(fig)
print("\nSaved: wulff_au_literature.png")
```

---

## Key Parameters

| Parameter | Typical Value | Notes |
|---|---|---|
| Miller indices | {100}, {110}, {111} minimum | Include at least the 3 low-index facets. More facets = more accurate shape. |
| Surface energies | J/m^2 | Must all be computed with the same method and settings for consistency. |
| Lattice | Crystal lattice | Must match the structure used to compute surface energies. |
| Slab thickness | 15+ Ang for converged gamma | Converge before computing Wulff shape (see surface-energy-calc skill). |

### Which Facets to Include

| Crystal System | Essential Facets | Additional (recommended) |
|---|---|---|
| FCC | {111}, {100}, {110} | {210}, {211}, {310}, {311}, {221}, {331} |
| BCC | {110}, {100}, {111} | {210}, {211}, {310}, {321} |
| HCP | {0001}, {10-10}, {10-11} | {11-20}, {10-12}, {11-21} |
| Perovskite | {100}, {110}, {111} | {210}, {211} |

For FCC metals, {111} is almost always the lowest-energy surface (close-packed). The Wulff shape is typically a truncated octahedron.

## Interpreting Results

- **Shape factor**: Ratio of Wulff shape surface-to-volume ratio vs a sphere. A sphere has shape factor 1.0. Wulff shapes have shape factor >= 1.0. Values around 1.0-1.1 indicate a nearly spherical (low anisotropy) crystal.
- **Anisotropy**: (gamma_max - gamma_min) / gamma_min. Low anisotropy (< 0.3) means all facets have similar energies and the shape is nearly spherical. High anisotropy (> 0.5) means certain facets strongly dominate.
- **Weighted surface energy**: The average surface energy weighted by area fraction. This is the effective surface energy for thermodynamic models.
- **Dominant facets**: Facets with the lowest surface energy occupy the largest area. These are the surfaces most relevant for catalysis and adsorption studies.
- **Missing facets**: Facets with high surface energy may not appear at all in the Wulff shape (zero area fraction). They are thermodynamically unstable.
- **FCC pattern**: For most FCC metals, {111} dominates (60-80% area), followed by {100} (10-30%). {110} and higher-index facets have small or zero area.

## Common Issues

| Problem | Solution |
|---|---|
| WulffShape raises error | Ensure all Miller indices are valid for the lattice. Ensure all surface energies are positive. |
| All area goes to one facet | The surface energies may be too different. Check convergence of each surface energy. One value may be spuriously low. |
| Zero-area facets expected to appear | Their surface energy is too high relative to neighboring facets. Verify the calculation. |
| Different results from MACE vs DFT | MACE gives approximate surface energies. Use DFT for publication. Relative ordering should agree. |
| Plot looks wrong | The 3D plotting depends on correct facet vertices. Use `wulff.get_plot()` for reliable visualization. |
| HCP Miller indices | Use 3-index notation (hkl) not 4-index (hkil). pymatgen handles the conversion internally. |
| Need absolute nanoparticle size | Wulff construction gives only the shape (ratios). To get absolute dimensions, you need to specify the total number of atoms or volume. |
| Kinetic vs thermodynamic shape | Wulff shape is the thermodynamic equilibrium shape. Real nanoparticles may differ due to kinetic effects during growth. |
