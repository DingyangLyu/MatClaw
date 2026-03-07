# Materials Project API Queries

## When to Use

- Retrieve crystal structures by formula, material ID, or composition
- Search for materials with specific properties (band gap, formation energy, magnetic ordering)
- Construct phase diagrams and convex hulls for a given chemical system
- Download relaxed DFT structures as starting points for further calculations
- Compare your computed results against the Materials Project database
- Get thermodynamic data (formation energies, decomposition products)
- Corresponds to VASPKIT task 702 (query online databases)

## Method Selection

| Task | API Method | Notes |
|---|---|---|
| Search by formula | `mpr.materials.summary.search(formula=...)` | Returns all polymorphs/entries for a formula |
| Search by MP ID | `mpr.materials.summary.get_data_by_id(...)` | Direct lookup by materials ID |
| Property screening | `mpr.materials.summary.search(band_gap=..., ...)` | Filter by property ranges |
| Phase diagram | `mpr.get_entries_in_chemsys(...)` | Get all entries in a chemical system |
| Structure download | `mpr.get_structure_by_material_id(...)` | Get pymatgen Structure object |
| Pourbaix diagram | `mpr.get_pourbaix_entries(...)` | Electrochemical stability |
| Phonon data | `mpr.materials.phonon.search(...)` | Pre-computed phonon band structures |

## Prerequisites

- `mp-api` package (pre-installed)
- `pymatgen` (pre-installed)
- Materials Project API key: set `MP_API_KEY` environment variable
  ```bash
  export MP_API_KEY="your_api_key_here"
  ```
- matplotlib for plotting

## Detailed Steps

### 1. Basic Structure Retrieval

```python
#!/usr/bin/env python3
"""
Query Materials Project for structures by formula or material ID.
Download structures in various formats (CIF, POSCAR, etc.).
"""

import os
from mp_api.client import MPRester
from pymatgen.core import Structure
from pymatgen.io.cif import CifWriter
import json

api_key = os.environ.get("MP_API_KEY")
if not api_key:
    raise ValueError("Set MP_API_KEY environment variable")

with MPRester(api_key) as mpr:
    # ============================================================
    # Search by chemical formula
    # ============================================================
    print("=== Search by Formula: TiO2 ===")
    results = mpr.materials.summary.search(
        formula="TiO2",
        fields=[
            "material_id", "formula_pretty", "structure",
            "energy_above_hull", "band_gap", "is_stable",
            "symmetry", "density",
        ],
    )

    print(f"Found {len(results)} entries for TiO2\n")

    # Sort by energy above hull (most stable first)
    results_sorted = sorted(results, key=lambda x: x.energy_above_hull or 999)

    print(f"{'MP ID':<14} {'Formula':<10} {'E_hull (eV/atom)':<18} "
          f"{'Band Gap (eV)':<15} {'Space Group':<15} {'Stable?'}")
    print("-" * 90)

    for r in results_sorted[:10]:  # show top 10
        sg = r.symmetry.symbol if r.symmetry else "N/A"
        print(f"{r.material_id:<14} {r.formula_pretty:<10} "
              f"{r.energy_above_hull:<18.4f} {r.band_gap:<15.2f} "
              f"{sg:<15} {'YES' if r.is_stable else 'no'}")

    # Download the most stable structure
    if results_sorted:
        best = results_sorted[0]
        struct = best.structure
        mp_id = str(best.material_id)

        # Save in multiple formats
        CifWriter(struct).write_file(f"{mp_id}_TiO2.cif")
        struct.to(filename=f"{mp_id}_TiO2.vasp", fmt="poscar")
        print(f"\nSaved structure: {mp_id}_TiO2.cif and .vasp")
        print(f"  Space group: {struct.get_space_group_info()}")
        print(f"  Lattice: a={struct.lattice.a:.4f}, b={struct.lattice.b:.4f}, "
              f"c={struct.lattice.c:.4f}")

    # ============================================================
    # Search by material ID
    # ============================================================
    print("\n=== Search by Material ID ===")
    mp_id_query = "mp-2657"  # rutile TiO2
    doc = mpr.materials.summary.get_data_by_id(mp_id_query)

    print(f"Material: {doc.formula_pretty} ({mp_id_query})")
    print(f"  Band gap: {doc.band_gap:.2f} eV")
    print(f"  Formation energy: {doc.formation_energy_per_atom:.4f} eV/atom")
    print(f"  Energy above hull: {doc.energy_above_hull:.4f} eV/atom")
    print(f"  Density: {doc.density:.4f} g/cm^3")
    print(f"  Is stable: {doc.is_stable}")

    # Get structure
    struct = doc.structure
    struct.to(filename=f"{mp_id_query}.cif")
    print(f"  Saved: {mp_id_query}.cif")
```

### 2. Property-Based Screening

```python
#!/usr/bin/env python3
"""
Screen materials by property ranges.
Example: Find stable semiconductors with band gap between 1.5-2.5 eV
in the (Ti, Zn, Cd)-(O, S, Se) chemical space.
"""

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mp_api.client import MPRester
import json

api_key = os.environ.get("MP_API_KEY")

with MPRester(api_key) as mpr:
    # ============================================================
    # Screen for semiconductors with specific band gap
    # ============================================================
    print("=== Screening: Semiconductors with 1.5-2.5 eV band gap ===")
    results = mpr.materials.summary.search(
        # Filter by band gap range
        band_gap=(1.5, 2.5),
        # Only thermodynamically stable materials
        is_stable=True,
        # Specify elements to include (optional)
        elements=["O"],
        # Fields to retrieve
        fields=[
            "material_id", "formula_pretty", "band_gap",
            "energy_above_hull", "formation_energy_per_atom",
            "nsites", "density", "symmetry",
            "is_metal",
        ],
    )

    # Filter out metals (band_gap > 0 should already exclude, but double-check)
    semiconductors = [r for r in results if not r.is_metal and r.band_gap > 0.1]

    print(f"Found {len(semiconductors)} stable oxide semiconductors "
          f"with 1.5-2.5 eV band gap\n")

    # Sort by formation energy (most stable first)
    semiconductors.sort(key=lambda x: x.formation_energy_per_atom or 0)

    print(f"{'MP ID':<14} {'Formula':<12} {'Eg (eV)':<9} "
          f"{'E_form (eV/at)':<16} {'n_sites':<8} {'SG'}")
    print("-" * 75)
    for r in semiconductors[:20]:
        sg = r.symmetry.symbol if r.symmetry else "N/A"
        print(f"{r.material_id:<14} {r.formula_pretty:<12} "
              f"{r.band_gap:<9.2f} {r.formation_energy_per_atom:<16.4f} "
              f"{r.nsites:<8} {sg}")

    # ============================================================
    # Plot band gap distribution
    # ============================================================
    band_gaps = [r.band_gap for r in semiconductors if r.band_gap is not None]
    form_energies = [r.formation_energy_per_atom for r in semiconductors
                     if r.formation_energy_per_atom is not None]

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Band gap histogram
    ax1 = axes[0]
    ax1.hist(band_gaps, bins=30, color="steelblue", edgecolor="black", alpha=0.7)
    ax1.set_xlabel("Band Gap (eV)", fontsize=12)
    ax1.set_ylabel("Count", fontsize=12)
    ax1.set_title("Band Gap Distribution", fontsize=13)
    ax1.grid(alpha=0.3)

    # Formation energy vs band gap scatter
    ax2 = axes[1]
    if len(band_gaps) == len(form_energies):
        ax2.scatter(band_gaps, form_energies, c="steelblue", alpha=0.5, s=30)
    ax2.set_xlabel("Band Gap (eV)", fontsize=12)
    ax2.set_ylabel("Formation Energy (eV/atom)", fontsize=12)
    ax2.set_title("Stability vs Band Gap", fontsize=13)
    ax2.grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig("materials_screening.png", dpi=150)
    print("\nSaved materials_screening.png")

    # Save results
    output = [
        {
            "material_id": str(r.material_id),
            "formula": r.formula_pretty,
            "band_gap_eV": r.band_gap,
            "formation_energy_eV_per_atom": r.formation_energy_per_atom,
            "nsites": r.nsites,
        }
        for r in semiconductors[:50]
    ]
    with open("screening_results.json", "w") as f:
        json.dump(output, f, indent=2)
    print("Saved screening_results.json")
```

### 3. Phase Diagram and Convex Hull

```python
#!/usr/bin/env python3
"""
Construct a phase diagram (convex hull) for a chemical system
using Materials Project data.

Example: Li-Fe-O system (relevant for battery cathodes).
"""

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mp_api.client import MPRester
from pymatgen.analysis.phase_diagram import PhaseDiagram, PDPlotter
from pymatgen.core import Composition
import json

api_key = os.environ.get("MP_API_KEY")

with MPRester(api_key) as mpr:
    # ============================================================
    # 1. Get all entries in the Li-Fe-O system
    # ============================================================
    system = ["Li", "Fe", "O"]
    system_str = "-".join(system)
    print(f"=== Phase Diagram: {system_str} ===")

    entries = mpr.get_entries_in_chemsys(
        system,
        additional_criteria={"thermo_types": ["GGA_GGA+U"]},
    )
    print(f"Retrieved {len(entries)} entries")

    # ============================================================
    # 2. Build phase diagram
    # ============================================================
    pd = PhaseDiagram(entries)

    # List stable phases
    stable_entries = pd.stable_entries
    print(f"\nStable phases ({len(stable_entries)}):")
    print(f"{'Formula':<20} {'E_form (eV/atom)':<18} {'MP ID'}")
    print("-" * 55)
    for entry in sorted(stable_entries,
                        key=lambda x: x.composition.reduced_formula):
        e_form = pd.get_form_energy_per_atom(entry)
        mp_id = entry.entry_id
        print(f"{entry.composition.reduced_formula:<20} "
              f"{e_form:<18.4f} {mp_id}")

    # ============================================================
    # 3. Check stability of a specific composition
    # ============================================================
    target_comp = Composition("LiFePO4")
    print(f"\n=== Stability of {target_comp.reduced_formula} ===")

    # Find the entry closest to this composition
    try:
        decomp, e_above_hull = pd.get_decomp_and_e_above_hull(
            pd.get_hull_entry(target_comp)
        )
        if e_above_hull < 0.001:
            print(f"  {target_comp.reduced_formula} is ON the convex hull (stable)")
        else:
            print(f"  {target_comp.reduced_formula} is {e_above_hull:.4f} eV/atom "
                  f"above the hull")
            print(f"  Decomposes into:")
            for phase, amount in decomp.items():
                print(f"    {phase.composition.reduced_formula}: {amount:.4f}")
    except Exception as e:
        print(f"  Could not analyze: {e}")
        print("  (Composition may not be in the chemical system)")

    # ============================================================
    # 4. Plot phase diagram
    # ============================================================
    # Binary phase diagram (if 2 elements)
    if len(system) == 2:
        plotter = PDPlotter(pd)
        fig = plotter.get_plot()
        fig.savefig(f"phase_diagram_{system_str}.png", dpi=150)
        print(f"\nSaved phase_diagram_{system_str}.png")

    # Ternary phase diagram (if 3 elements)
    elif len(system) == 3:
        plotter = PDPlotter(pd, ternary_style="2d")
        fig = plotter.get_plot()
        fig.savefig(f"phase_diagram_{system_str}.png", dpi=150)
        print(f"\nSaved phase_diagram_{system_str}.png")

    # For any system: plot formation energies
    fig, ax = plt.subplots(figsize=(10, 6))

    # All entries
    formulas = []
    e_hull_values = []
    for entry in entries:
        try:
            e_hull = pd.get_e_above_hull(entry)
            formulas.append(entry.composition.reduced_formula)
            e_hull_values.append(e_hull)
        except Exception:
            pass

    # Sort by e_above_hull
    sorted_indices = np.argsort(e_hull_values)

    # Plot stable (e_hull = 0) and unstable phases
    colors = ["green" if e < 0.001 else "red" for e in e_hull_values]

    ax.bar(range(min(30, len(formulas))),
           [e_hull_values[i] for i in sorted_indices[:30]],
           color=[colors[i] for i in sorted_indices[:30]],
           edgecolor="black", alpha=0.7)
    ax.set_xticks(range(min(30, len(formulas))))
    ax.set_xticklabels([formulas[i] for i in sorted_indices[:30]],
                        rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("Energy above hull (eV/atom)", fontsize=12)
    ax.set_title(f"Phase Stability: {system_str} System", fontsize=14)
    ax.grid(axis="y", alpha=0.3)

    fig.tight_layout()
    fig.savefig(f"stability_{system_str}.png", dpi=150)
    print(f"Saved stability_{system_str}.png")
```

### 4. Batch Structure Download

```python
#!/usr/bin/env python3
"""
Download multiple structures from Materials Project for a set of compositions.
Save in CIF and POSCAR format for subsequent calculations.
"""

import os
from mp_api.client import MPRester
from pymatgen.io.cif import CifWriter
from pathlib import Path
import json

api_key = os.environ.get("MP_API_KEY")

# List of materials to download (by formula)
target_formulas = [
    "MoS2",
    "WS2",
    "MoSe2",
    "WSe2",
    "BN",
    "graphite",
    "MoTe2",
]

output_dir = Path("mp_structures")
output_dir.mkdir(exist_ok=True)

manifest = []

with MPRester(api_key) as mpr:
    for formula in target_formulas:
        print(f"\n=== {formula} ===")

        results = mpr.materials.summary.search(
            formula=formula,
            is_stable=True,
            fields=["material_id", "formula_pretty", "structure",
                    "band_gap", "energy_above_hull", "symmetry"],
        )

        if not results:
            # Try without is_stable filter
            results = mpr.materials.summary.search(
                formula=formula,
                fields=["material_id", "formula_pretty", "structure",
                        "band_gap", "energy_above_hull", "symmetry"],
            )
            if results:
                results.sort(key=lambda x: x.energy_above_hull or 999)

        if not results:
            print(f"  No entries found for {formula}")
            continue

        # Take the most stable entry
        best = results[0]
        mp_id = str(best.material_id)
        struct = best.structure
        sg = best.symmetry.symbol if best.symmetry else "N/A"

        print(f"  MP ID: {mp_id}")
        print(f"  Space group: {sg}")
        print(f"  Band gap: {best.band_gap:.2f} eV")
        print(f"  E above hull: {best.energy_above_hull:.4f} eV/atom")
        print(f"  Sites: {len(struct)}")

        # Save files
        cif_path = output_dir / f"{mp_id}_{formula}.cif"
        poscar_path = output_dir / f"{mp_id}_{formula}.vasp"

        CifWriter(struct).write_file(str(cif_path))
        struct.to(filename=str(poscar_path), fmt="poscar")

        print(f"  Saved: {cif_path.name}, {poscar_path.name}")

        manifest.append({
            "formula": formula,
            "material_id": mp_id,
            "space_group": sg,
            "band_gap_eV": best.band_gap,
            "e_above_hull_eV_per_atom": best.energy_above_hull,
            "n_sites": len(struct),
            "cif_file": str(cif_path),
            "poscar_file": str(poscar_path),
        })

# Save manifest
with open(str(output_dir / "manifest.json"), "w") as f:
    json.dump(manifest, f, indent=2)
print(f"\nManifest saved to {output_dir}/manifest.json")
print(f"Downloaded {len(manifest)} structures")
```

### 5. Retrieve Elastic and Mechanical Properties

```python
#!/usr/bin/env python3
"""
Query Materials Project for elastic and mechanical properties.
"""

import os
from mp_api.client import MPRester
import json

api_key = os.environ.get("MP_API_KEY")

with MPRester(api_key) as mpr:
    # Search for materials with elastic data
    formula = "Si"
    print(f"=== Elastic Properties: {formula} ===")

    elastic_docs = mpr.materials.elasticity.search(
        formula=formula,
        fields=[
            "material_id", "formula_pretty",
            "bulk_modulus", "shear_modulus",
            "universal_anisotropy", "elastic_tensor",
        ],
    )

    if elastic_docs:
        for doc in elastic_docs[:5]:
            print(f"\n{doc.material_id} ({doc.formula_pretty}):")
            if doc.bulk_modulus:
                print(f"  Bulk modulus (Voigt):  {doc.bulk_modulus.voigt:.1f} GPa")
            if doc.shear_modulus:
                print(f"  Shear modulus (Voigt): {doc.shear_modulus.voigt:.1f} GPa")
            if doc.universal_anisotropy is not None:
                print(f"  Universal anisotropy:  {doc.universal_anisotropy:.4f}")
    else:
        print(f"  No elastic data found for {formula}")

    # ============================================================
    # Dielectric properties
    # ============================================================
    print(f"\n=== Dielectric Properties: {formula} ===")

    dielectric_docs = mpr.materials.dielectric.search(
        formula=formula,
        fields=["material_id", "formula_pretty", "e_total", "e_ionic", "e_electronic"],
    )

    if dielectric_docs:
        for doc in dielectric_docs[:5]:
            print(f"\n{doc.material_id} ({doc.formula_pretty}):")
            if doc.e_total is not None:
                print(f"  Total dielectric constant: {doc.e_total:.2f}")
            if doc.e_electronic is not None:
                print(f"  Electronic contribution:   {doc.e_electronic:.2f}")
            if doc.e_ionic is not None:
                print(f"  Ionic contribution:        {doc.e_ionic:.2f}")
    else:
        print(f"  No dielectric data found for {formula}")
```

## Key Parameters

| Parameter | Notes |
|---|---|
| `MP_API_KEY` | Required. Get from materialsproject.org/api |
| `formula` | Chemical formula (e.g., "TiO2", "LiFePO4") |
| `elements` | List of elements to include (e.g., ["Ti", "O"]) |
| `chemsys` | Chemical system string (e.g., "Li-Fe-O") |
| `band_gap` | Tuple (min, max) in eV for screening |
| `is_stable` | Boolean, filter for thermodynamically stable phases |
| `fields` | List of fields to retrieve (reduces query time) |
| `energy_above_hull` | Stability metric: 0 = on hull, > 0 = metastable |

## Interpreting Results

1. **Energy above hull**: 0 eV/atom means the phase is thermodynamically stable. < 0.025 eV/atom is considered near the hull and potentially synthesizable. > 0.1 eV/atom is unlikely to be stable.
2. **Band gap**: DFT (PBE/GGA) systematically underestimates band gaps by ~30-50%. Materials Project uses GGA or GGA+U. For accurate gaps, consider HSE06 or GW corrections.
3. **Formation energy**: Negative means the compound is stable with respect to its elemental constituents. More negative = more stable.
4. **Phase diagram**: Phases on the convex hull are thermodynamically stable. Phases above the hull will decompose into the hull phases.
5. **Multiple entries per formula**: Different polymorphs or magnetic orderings. Sort by energy_above_hull to find the ground state.

## Common Issues

| Issue | Solution |
|---|---|
| API key not set | `export MP_API_KEY="your_key"` or set in Python: `os.environ["MP_API_KEY"] = "..."` |
| Rate limiting (429 errors) | Add delays between queries. Use `fields` parameter to reduce data. |
| No results for a formula | Check formula spelling. Try searching by elements instead. |
| Old API syntax fails | Ensure `mp-api >= 0.33`. The new API uses `mpr.materials.summary.search()`. |
| Structure has wrong conventional cell | Use `SpacegroupAnalyzer(struct).get_conventional_standard_structure()` |
| Phase diagram fails with mixed functionals | Use `additional_criteria={"thermo_types": ["GGA_GGA+U"]}` for consistent energies |
| Cannot find specific property | Not all materials have all properties computed. Check available fields in docs. |
