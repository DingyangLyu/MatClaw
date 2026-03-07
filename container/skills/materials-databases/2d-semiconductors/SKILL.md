# 2D Semiconductor Databases

## When to Use

- Search for 2D semiconductor materials with specific band gaps
- Screen monolayer or few-layer materials for optoelectronic applications
- Retrieve 2D material structures from computational databases
- Compare properties across families of 2D materials (TMDs, MXenes, etc.)
- Find 2D materials with specific band edge alignments for photocatalysis or heterojunctions
- Corresponds to VASPKIT task 705 (2D semiconductor database)

## Method Selection

| Database | Access Method | Coverage |
|---|---|---|
| Materials Project (2D) | mp-api + dimensionality filter | Large, DFT-computed |
| C2DB (Computational 2D Database) | Web download / ASE database | ~4000 monolayers, PBE+HSE |
| 2DMatPedia | Web API / download | ~6000 2D materials |
| JARVIS-DFT 2D | jarvis-tools API | ~1000 2D materials, OptB88vdW |
| pymatgen + robocrys | Rule-based identification | Identify 2D materials from any database |

```
Need to screen many 2D materials by properties?
  --> Use Materials Project with dimensionality filter (Method A)
  --> Or use C2DB/2DMatPedia for pre-screened 2D materials (Method B)

Need accurate band gaps for 2D semiconductors?
  --> C2DB provides HSE06 band gaps (Method B)
  --> Or compute with QE/VASP yourself

Need to identify 2D materials from a general database?
  --> Use pymatgen dimensionality analysis (Method C)

Need 2D material structures for MACE/QE calculations?
  --> Download from any database, then run your calculation
```

## Prerequisites

- `mp-api` (Materials Project queries)
- `pymatgen` (structure manipulation, dimensionality analysis)
- `ase` (reading database files)
- `matplotlib` (plotting)
- `numpy` (numerical analysis)
- Optional: `jarvis-tools` (`pip install jarvis-tools`)

## Detailed Steps

### Method A: Materials Project -- 2D Material Search

```python
#!/usr/bin/env python3
"""
Search Materials Project for 2D materials using dimensionality analysis.

Strategy:
1. Query MP for layered materials (van der Waals bonded)
2. Filter by band gap range
3. Analyze dimensionality using pymatgen
4. Download structures
"""

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mp_api.client import MPRester
from pymatgen.core import Structure
from pymatgen.analysis.dimensionality import get_dimensionality_larsen
from pymatgen.io.cif import CifWriter
from pathlib import Path
import json

api_key = os.environ.get("MP_API_KEY")

output_dir = Path("2d_materials")
output_dir.mkdir(exist_ok=True)

with MPRester(api_key) as mpr:
    # ============================================================
    # 1. Search for known 2D material families
    # ============================================================
    # Transition Metal Dichalcogenides (TMDs)
    tmd_formulas = [
        "MoS2", "MoSe2", "MoTe2",
        "WS2", "WSe2", "WTe2",
        "NbS2", "NbSe2",
        "TaS2", "TaSe2",
        "TiS2", "TiSe2",
        "ZrS2", "ZrSe2",
        "HfS2", "HfSe2",
        "SnS2", "SnSe2",
        "PtS2", "PtSe2",
    ]

    # Other 2D materials
    other_2d = [
        "BN",       # hexagonal boron nitride
        "GaS",      # gallium sulfide
        "GaSe",     # gallium selenide
        "InSe",     # indium selenide
        "Bi2Se3",   # topological insulator
        "Bi2Te3",
        "MnBi2Te4", # magnetic TI
    ]

    all_formulas = tmd_formulas + other_2d

    results_list = []

    print(f"{'Formula':<12} {'MP ID':<14} {'Eg (eV)':<9} {'E_hull':<10} "
          f"{'n_sites':<8} {'SG'}")
    print("-" * 70)

    for formula in all_formulas:
        docs = mpr.materials.summary.search(
            formula=formula,
            fields=[
                "material_id", "formula_pretty", "structure",
                "band_gap", "energy_above_hull", "symmetry",
                "nsites",
            ],
        )

        if not docs:
            continue

        # Take most stable entry
        docs.sort(key=lambda x: x.energy_above_hull or 999)
        best = docs[0]

        sg = best.symmetry.symbol if best.symmetry else "N/A"
        print(f"{best.formula_pretty:<12} {best.material_id:<14} "
              f"{best.band_gap:<9.2f} {best.energy_above_hull:<10.4f} "
              f"{best.nsites:<8} {sg}")

        results_list.append({
            "formula": best.formula_pretty,
            "material_id": str(best.material_id),
            "band_gap_eV": best.band_gap,
            "energy_above_hull": best.energy_above_hull,
            "n_sites": best.nsites,
            "space_group": sg,
        })

        # Save structure
        struct = best.structure
        mp_id = str(best.material_id)
        CifWriter(struct).write_file(str(output_dir / f"{mp_id}_{formula}.cif"))

    # ============================================================
    # 2. Plot band gaps of 2D materials
    # ============================================================
    formulas = [r["formula"] for r in results_list]
    band_gaps = [r["band_gap_eV"] for r in results_list]

    fig, ax = plt.subplots(figsize=(14, 5))
    colors = plt.cm.viridis(np.linspace(0.2, 0.9, len(formulas)))

    bars = ax.bar(range(len(formulas)), band_gaps, color=colors,
                  edgecolor="black", alpha=0.8)

    # Label bars
    for bar, val in zip(bars, band_gaps):
        if val > 0.1:
            ax.text(bar.get_x() + bar.get_width()/2, val + 0.05,
                    f"{val:.1f}", ha="center", va="bottom", fontsize=7)

    ax.set_xticks(range(len(formulas)))
    ax.set_xticklabels(formulas, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("Band Gap (eV)", fontsize=12)
    ax.set_title("Band Gaps of 2D Materials (Materials Project, PBE)", fontsize=14)

    # Reference lines for application ranges
    ax.axhline(1.1, color="red", linestyle="--", alpha=0.5, label="Si (1.1 eV)")
    ax.axhline(1.8, color="orange", linestyle="--", alpha=0.5, label="Water splitting")
    ax.legend(fontsize=9)
    ax.grid(axis="y", alpha=0.3)

    fig.tight_layout()
    fig.savefig(str(output_dir / "2d_band_gaps.png"), dpi=150)
    print(f"\nSaved {output_dir}/2d_band_gaps.png")

    # Save manifest
    with open(str(output_dir / "2d_materials_manifest.json"), "w") as f:
        json.dump(results_list, f, indent=2)
    print(f"Saved {output_dir}/2d_materials_manifest.json")
```

### Method B: C2DB / 2DMatPedia Style Database Access

```python
#!/usr/bin/env python3
"""
Work with the Computational 2D Materials Database (C2DB) data.

C2DB provides pre-computed properties for ~4000 monolayer materials including:
  - PBE and HSE06 band gaps
  - Band edge positions (VBM, CBM)
  - Effective masses
  - Stability analysis (thermodynamic, dynamic, mechanical)

Since direct API access may not be available, this script demonstrates
how to:
1. Build a local 2D materials database from Materials Project
2. Screen by properties relevant to 2D semiconductors
3. Compute additional properties (exfoliation energy, etc.)
"""

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pymatgen.core import Structure
from pymatgen.analysis.dimensionality import get_dimensionality_larsen
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
import json


def analyze_2d_material(structure, name="material"):
    """
    Analyze a structure to determine if it is a 2D (layered) material.

    Returns a dict with dimensionality, layer thickness, van der Waals gap, etc.
    """
    result = {
        "name": name,
        "formula": structure.composition.reduced_formula,
        "n_atoms": len(structure),
    }

    # Dimensionality analysis
    try:
        dim = get_dimensionality_larsen(structure)
        result["dimensionality"] = dim
        result["is_2d"] = (dim == 2)
    except Exception:
        result["dimensionality"] = None
        result["is_2d"] = None

    # Layer analysis for 2D materials
    # Measure the extent along c-axis
    frac_coords = structure.frac_coords
    z_coords = frac_coords[:, 2]

    # Cartesian z-coordinates
    cart_z = np.array([s.coords[2] for s in structure])

    if len(cart_z) > 1:
        z_range = np.max(cart_z) - np.min(cart_z)
        c_length = structure.lattice.c

        # Layer thickness = range of atoms in z
        result["layer_thickness_A"] = float(z_range)
        result["c_parameter_A"] = float(c_length)
        result["vacuum_fraction"] = float(1 - z_range / c_length) if c_length > 0 else 0

    # Space group
    try:
        sga = SpacegroupAnalyzer(structure)
        result["space_group"] = sga.get_space_group_symbol()
        result["crystal_system"] = sga.get_crystal_system()
    except Exception:
        result["space_group"] = "N/A"
        result["crystal_system"] = "N/A"

    return result


def screen_2d_semiconductors(materials_data, band_gap_range=(0.5, 3.0)):
    """
    Screen a list of 2D materials for semiconductor applications.

    Parameters
    ----------
    materials_data : list of dict
        Each dict has: formula, band_gap_eV, structure, ...
    band_gap_range : tuple
        (min, max) band gap in eV.

    Returns
    -------
    list of dict: Filtered and ranked materials.
    """
    candidates = []
    for mat in materials_data:
        bg = mat.get("band_gap_eV", 0)
        if bg is None:
            continue
        if band_gap_range[0] <= bg <= band_gap_range[1]:
            candidates.append(mat)

    # Rank by band gap (closest to target for solar cell: ~1.3-1.5 eV)
    target_gap = 1.4
    candidates.sort(key=lambda x: abs(x.get("band_gap_eV", 0) - target_gap))

    return candidates


# ============================================================
# Example: Analyze a set of known 2D structures
# ============================================================
# Load structures from CIF files (downloaded from MP or other databases)
from pathlib import Path

struct_dir = Path("2d_materials")
cif_files = sorted(struct_dir.glob("*.cif")) if struct_dir.exists() else []

analysis_results = []

print("2D MATERIAL ANALYSIS")
print("=" * 70)
print(f"{'Formula':<12} {'Dim':<5} {'2D?':<5} {'Layer (A)':<10} "
      f"{'c (A)':<8} {'SG':<12}")
print("-" * 60)

for cif_file in cif_files:
    try:
        struct = Structure.from_file(str(cif_file))
        result = analyze_2d_material(struct, name=cif_file.stem)

        dim_str = str(result.get("dimensionality", "?"))
        is_2d = "YES" if result.get("is_2d") else "no"
        layer = f"{result.get('layer_thickness_A', 0):.2f}"
        c_param = f"{result.get('c_parameter_A', 0):.2f}"
        sg = result.get("space_group", "N/A")

        print(f"{result['formula']:<12} {dim_str:<5} {is_2d:<5} "
              f"{layer:<10} {c_param:<8} {sg:<12}")

        analysis_results.append(result)
    except Exception as e:
        print(f"  Error processing {cif_file.name}: {e}")

if analysis_results:
    with open("2d_analysis.json", "w") as f:
        json.dump(analysis_results, f, indent=2, default=float)
    print(f"\nSaved 2d_analysis.json ({len(analysis_results)} materials)")
```

### Method C: Identify 2D Materials Using Dimensionality Analysis

```python
#!/usr/bin/env python3
"""
Identify 2D (layered) materials from a set of structures using
pymatgen's dimensionality analysis.

Uses the Larsen algorithm which identifies connected components
based on bonding, then determines structural dimensionality.
"""

import numpy as np
from pymatgen.core import Structure
from pymatgen.analysis.dimensionality import get_dimensionality_larsen
from pymatgen.analysis.structure_matcher import StructureMatcher
from pymatgen.analysis.local_env import CrystalNN
import json


def identify_2d_materials(structures, labels=None):
    """
    Classify structures by dimensionality.

    Parameters
    ----------
    structures : list of pymatgen Structure
    labels : list of str, optional

    Returns
    -------
    dict mapping dimensionality (0, 1, 2, 3) to list of (label, structure)
    """
    if labels is None:
        labels = [f"struct_{i}" for i in range(len(structures))]

    classified = {0: [], 1: [], 2: [], 3: []}

    for struct, label in zip(structures, labels):
        try:
            dim = get_dimensionality_larsen(struct)
            classified[dim].append((label, struct))
        except Exception as e:
            print(f"  Could not analyze {label}: {e}")

    return classified


def get_exfoliation_energy_estimate(bulk_structure, monolayer_structure=None):
    """
    Estimate exfoliation energy from interlayer spacing.

    This is a rough estimate based on empirical correlation:
    E_exf ~ 20 meV/A^2 * A_cell (for typical vdW materials)

    For accurate values, compute:
    E_exf = (E_monolayer - E_bulk/n_layers) / A_cell

    Parameters
    ----------
    bulk_structure : Structure
        Bulk layered structure.

    Returns
    -------
    float : Estimated exfoliation energy in meV/A^2.
    """
    # Surface area of the ab-plane
    a_vec = bulk_structure.lattice.matrix[0]
    b_vec = bulk_structure.lattice.matrix[1]
    area = np.linalg.norm(np.cross(a_vec, b_vec))

    # Typical vdW exfoliation energies:
    # Graphite: ~20 meV/A^2
    # MoS2: ~25 meV/A^2
    # BN: ~22 meV/A^2
    # General vdW: 15-30 meV/A^2

    return 20.0  # meV/A^2 (rough estimate)


def create_monolayer(bulk_structure, vacuum=20.0):
    """
    Create a monolayer from a layered bulk structure by:
    1. Identifying layer boundaries
    2. Extracting one layer
    3. Adding vacuum

    Parameters
    ----------
    bulk_structure : Structure
        Bulk layered structure (should have layers along c-axis).
    vacuum : float
        Vacuum thickness in Angstrom.

    Returns
    -------
    Structure : Monolayer structure with vacuum.
    """
    # Get atomic positions in fractional coordinates
    frac_coords = bulk_structure.frac_coords.copy()
    species = [str(s.specie) for s in bulk_structure]

    # Sort by z-coordinate
    z_sorted = np.sort(frac_coords[:, 2])

    # Find the largest gap in z (interlayer gap)
    gaps = np.diff(z_sorted)
    max_gap_idx = np.argmax(gaps)
    gap_center = (z_sorted[max_gap_idx] + z_sorted[max_gap_idx + 1]) / 2

    # Take atoms below the gap (one layer)
    layer_mask = frac_coords[:, 2] < gap_center
    layer_species = [sp for sp, m in zip(species, layer_mask) if m]
    layer_coords = frac_coords[layer_mask]

    if len(layer_species) == 0:
        # Try the other layer
        layer_mask = frac_coords[:, 2] >= gap_center
        layer_species = [sp for sp, m in zip(species, layer_mask) if m]
        layer_coords = frac_coords[layer_mask]

    # Get layer thickness in Cartesian coordinates
    cart_z = bulk_structure.lattice.get_cartesian_coords(layer_coords)[:, 2]
    layer_thickness = np.max(cart_z) - np.min(cart_z)

    # New c parameter = layer thickness + vacuum
    new_c = layer_thickness + vacuum

    # New lattice: keep a, b; change c
    old_matrix = bulk_structure.lattice.matrix.copy()
    c_hat = old_matrix[2] / np.linalg.norm(old_matrix[2])
    new_matrix = old_matrix.copy()
    new_matrix[2] = c_hat * new_c

    # Rescale z-coordinates to new cell
    from pymatgen.core import Lattice
    new_lattice = Lattice(new_matrix)

    # Convert layer coords to Cartesian, then to new fractional
    old_cart = bulk_structure.lattice.get_cartesian_coords(layer_coords)
    # Center the layer in the new cell
    z_center = np.mean(old_cart[:, 2])
    old_cart[:, 2] -= z_center  # center at z=0
    old_cart[:, 2] += new_c / 2  # move to center of new cell

    new_frac = new_lattice.get_fractional_coords(old_cart)

    monolayer = Structure(new_lattice, layer_species, new_frac,
                          coords_are_cartesian=False)

    return monolayer


# ============================================================
# Example usage
# ============================================================
# Build example bulk MoS2
from pymatgen.core import Structure, Lattice

# 2H-MoS2 (hexagonal, P6_3/mmc)
mos2_bulk = Structure.from_spacegroup(
    "P6_3/mmc",
    Lattice.hexagonal(3.16, 12.30),
    species=["Mo", "S", "S"],
    coords=[[1/3, 2/3, 1/4], [1/3, 2/3, 0.621], [1/3, 2/3, 0.879]],
)

print("Bulk MoS2:")
print(f"  Formula: {mos2_bulk.composition.reduced_formula}")
print(f"  Atoms: {len(mos2_bulk)}")
print(f"  Lattice: a={mos2_bulk.lattice.a:.4f}, c={mos2_bulk.lattice.c:.4f}")

# Check dimensionality
dim = get_dimensionality_larsen(mos2_bulk)
print(f"  Dimensionality: {dim}")

# Create monolayer
monolayer = create_monolayer(mos2_bulk, vacuum=20.0)
print(f"\nMonolayer MoS2:")
print(f"  Atoms: {len(monolayer)}")
print(f"  c parameter: {monolayer.lattice.c:.4f} A")
monolayer.to(filename="MoS2_monolayer.cif")
print("  Saved: MoS2_monolayer.cif")
```

### JARVIS-DFT 2D Materials (Optional)

```python
#!/usr/bin/env python3
"""
Access the JARVIS-DFT 2D materials database.
Requires: pip install jarvis-tools
"""

try:
    from jarvis.db.figshare import data as jarvis_data
    from jarvis.core.atoms import Atoms as JarvisAtoms

    # Download 2D materials dataset
    d2 = jarvis_data("dft_2d")
    print(f"JARVIS 2D database: {len(d2)} entries")

    # Filter for semiconductors with band gap 1-3 eV
    semiconductors_2d = [
        entry for entry in d2
        if entry.get("optb88vdw_bandgap", 0) is not None
        and 1.0 <= (entry.get("optb88vdw_bandgap", 0) or 0) <= 3.0
    ]

    print(f"2D semiconductors (1-3 eV): {len(semiconductors_2d)}")

    # Display top entries
    print(f"\n{'Formula':<15} {'Eg (eV)':<10} {'Stable?':<8} {'JVASP ID'}")
    print("-" * 50)

    for entry in sorted(semiconductors_2d,
                        key=lambda x: abs((x.get("optb88vdw_bandgap", 0) or 0) - 1.5))[:20]:
        formula = entry.get("formula", "N/A")
        bg = entry.get("optb88vdw_bandgap", 0) or 0
        jid = entry.get("jid", "N/A")
        ehull = entry.get("ehull", None)
        stable = "YES" if ehull is not None and ehull < 0.1 else "?"
        print(f"{formula:<15} {bg:<10.3f} {stable:<8} {jid}")

except ImportError:
    print("jarvis-tools not installed. Install with: pip install jarvis-tools")
    print("This is optional -- Materials Project queries work without it.")
```

## Key Parameters

| Parameter | Notes |
|---|---|
| Band gap range | PBE underestimates; use (0.5, 3.0) for screening. HSE06 values are more accurate. |
| Dimensionality threshold | `get_dimensionality_larsen()` returns 0 (molecular), 1 (chain), 2 (layer), 3 (framework) |
| Vacuum thickness | 15-20 A for monolayer calculations to avoid interlayer interaction |
| Exfoliation energy | < 200 meV/atom suggests easy exfoliation. Compare to graphite (~50 meV/atom) |
| Interlayer distance | 3.0-3.5 A for vdW gap in TMDs. Use DFT-D3 or vdW-DF for accurate values |

## Interpreting Results

1. **Dimensionality = 2**: Material has layered structure with weak interlayer bonding. Good candidate for exfoliation.
2. **Band gap (PBE)**: Underestimated by ~30-50%. A PBE gap of 1.0 eV likely corresponds to an experimental gap of ~1.5-2.0 eV.
3. **Stability (E_hull)**: Materials on the convex hull (E_hull = 0) are thermodynamically stable. For 2D materials, also check dynamic stability (phonon band structure).
4. **TMD families**: MoS2/WS2 family has direct gaps in monolayer form (~1.8-2.0 eV experimental). Bulk forms have indirect gaps (~1.2-1.3 eV).
5. **Exfoliation energy**: Lower values indicate easier mechanical exfoliation. Graphite: ~20 meV/A^2, MoS2: ~25 meV/A^2.

## Common Issues

| Issue | Solution |
|---|---|
| `get_dimensionality_larsen` gives wrong result | Try adjusting bond length cutoffs. Some structures need manual inspection. |
| Band gap is 0 for a known semiconductor | PBE may close the gap. Use HSE06 or GW for accurate gaps. Check if the correct polymorph was retrieved. |
| Cannot create monolayer from bulk | Structure may not be properly oriented with layers along c-axis. Use `pymatgen.transformations` to reorient. |
| JARVIS data download is slow | The first download is large. Subsequent calls use cache. |
| Materials Project returns too many results | Use more specific filters (elements, band_gap range, is_stable). |
| 2D material relaxes to 3D structure | Add vacuum constraints. Fix in-plane lattice parameters during relaxation. |
