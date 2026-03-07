# Materials Database Querying and Filtering

## When to Use

- You need to find materials with specific properties (band gap range, crystal system, stability, elastic constants) from the Materials Project database.
- You want to download structures for a chemical system (e.g., all Li-Mn-O compounds) for further computation.
- You need statistical overviews of property distributions across a chemical space.
- You want to export filtered structures as CIF files and tabulated data as CSV.

## Prerequisites

- `MP_API_KEY` environment variable set (obtain from https://next-gen.materialsproject.org/api).
- Python packages: `mp-api`, `pymatgen`, `pandas`, `numpy`, `matplotlib`.

## Detailed Steps

### Pattern 1 -- Fetch All Stable Compounds in a Chemical System

```python
#!/usr/bin/env python3
"""
Fetch all thermodynamically stable (or near-stable) compounds
in the Li-Mn-O chemical system from Materials Project.
"""

import os
import warnings
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from mp_api.client import MPRester
from pymatgen.core import Structure

warnings.filterwarnings("ignore")

MP_API_KEY = os.environ["MP_API_KEY"]

with MPRester(MP_API_KEY) as mpr:
    docs = mpr.materials.summary.search(
        chemsys="Li-Mn-O",
        fields=[
            "material_id",
            "formula_pretty",
            "structure",
            "energy_above_hull",
            "formation_energy_per_atom",
            "band_gap",
            "nsites",
            "symmetry",
            "volume",
            "density",
            "is_stable",
        ],
    )

print(f"Total compounds in Li-Mn-O: {len(docs)}")

# Build a DataFrame
rows = []
for doc in docs:
    rows.append({
        "mp_id": str(doc.material_id),
        "formula": doc.formula_pretty,
        "e_above_hull": doc.energy_above_hull,
        "form_energy": doc.formation_energy_per_atom,
        "band_gap": doc.band_gap,
        "nsites": doc.nsites,
        "spacegroup": doc.symmetry.symbol if doc.symmetry else None,
        "crystal_system": doc.symmetry.crystal_system if doc.symmetry else None,
        "volume": doc.volume,
        "density": doc.density,
        "is_stable": doc.is_stable,
    })

df = pd.DataFrame(rows)

# Filter: stable or near-stable (e_above_hull < 0.025 eV/atom)
df_stable = df[df["e_above_hull"] < 0.025].copy()
df_stable.sort_values("e_above_hull", inplace=True)
df_stable.reset_index(drop=True, inplace=True)

print(f"\nStable / near-stable compounds (e_above_hull < 25 meV/atom):")
print(df_stable[["mp_id", "formula", "e_above_hull", "band_gap",
                  "spacegroup"]].to_string(index=False))

# Save structures as CIF
os.makedirs("Li_Mn_O_structures", exist_ok=True)
for doc in docs:
    mp_id = str(doc.material_id)
    if mp_id in df_stable["mp_id"].values:
        doc.structure.to(
            fmt="cif",
            filename=f"Li_Mn_O_structures/{mp_id}_{doc.formula_pretty}.cif"
        )
print(f"\nSaved {len(df_stable)} CIF files to Li_Mn_O_structures/")

# Export full table
df.to_csv("Li_Mn_O_all.csv", index=False)
print("Saved Li_Mn_O_all.csv")

# --- Plot: e_above_hull distribution ---
fig, ax = plt.subplots(figsize=(7, 4))
ax.hist(df["e_above_hull"], bins=30, edgecolor="black", alpha=0.8)
ax.axvline(0.025, color="red", ls="--", label="25 meV/atom cutoff")
ax.set_xlabel("Energy above hull (eV/atom)")
ax.set_ylabel("Count")
ax.set_title("Li-Mn-O: stability distribution")
ax.legend()
plt.tight_layout()
plt.savefig("Li_Mn_O_ehull_distribution.png", dpi=150)
plt.close()
print("Saved Li_Mn_O_ehull_distribution.png")
```

### Pattern 2 -- Search by Band Gap Range

```python
#!/usr/bin/env python3
"""
Find all materials with band gap between 1.0 and 3.0 eV
that are thermodynamically stable.
Useful for photovoltaic or photocatalysis screening.
"""

import os
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mp_api.client import MPRester

MP_API_KEY = os.environ["MP_API_KEY"]

with MPRester(MP_API_KEY) as mpr:
    docs = mpr.materials.summary.search(
        band_gap=(1.0, 3.0),       # min, max in eV
        is_stable=True,             # only on the convex hull
        fields=[
            "material_id",
            "formula_pretty",
            "band_gap",
            "energy_above_hull",
            "formation_energy_per_atom",
            "nsites",
            "symmetry",
            "density",
        ],
    )

print(f"Found {len(docs)} stable materials with band gap in [1, 3] eV")

rows = []
for doc in docs:
    rows.append({
        "mp_id": str(doc.material_id),
        "formula": doc.formula_pretty,
        "band_gap": doc.band_gap,
        "e_above_hull": doc.energy_above_hull,
        "form_energy": doc.formation_energy_per_atom,
        "nsites": doc.nsites,
        "crystal_system": doc.symmetry.crystal_system if doc.symmetry else None,
        "density": doc.density,
    })

df = pd.DataFrame(rows)
df.sort_values("band_gap", inplace=True)

# Show distribution by crystal system
print("\nCount by crystal system:")
print(df["crystal_system"].value_counts().to_string())

# Export
df.to_csv("bandgap_1_3eV_stable.csv", index=False)
print(f"\nSaved bandgap_1_3eV_stable.csv ({len(df)} entries)")

# --- Plot: band gap histogram ---
fig, axes = plt.subplots(1, 2, figsize=(12, 4))

axes[0].hist(df["band_gap"], bins=40, edgecolor="black", color="C0")
axes[0].set_xlabel("Band gap (eV)")
axes[0].set_ylabel("Count")
axes[0].set_title("Band gap distribution (stable, 1-3 eV)")

# Scatter: band gap vs formation energy
axes[1].scatter(df["band_gap"], df["form_energy"], alpha=0.4, s=10)
axes[1].set_xlabel("Band gap (eV)")
axes[1].set_ylabel("Formation energy (eV/atom)")
axes[1].set_title("Band gap vs. formation energy")

plt.tight_layout()
plt.savefig("bandgap_analysis.png", dpi=150)
plt.close()
print("Saved bandgap_analysis.png")
```

### Pattern 3 -- Fetch Elastic Data for Specific Materials

```python
#!/usr/bin/env python3
"""
Retrieve elastic properties (bulk modulus, shear modulus, Poisson ratio)
for a list of materials from Materials Project.
"""

import os
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mp_api.client import MPRester

MP_API_KEY = os.environ["MP_API_KEY"]

# Target materials -- can be MP IDs or formulas
TARGET_CHEMSYS = "Si-O"

with MPRester(MP_API_KEY) as mpr:
    # Fetch elastic data -- uses the elasticity sub-API
    elastic_docs = mpr.elasticity.search(
        chemsys=TARGET_CHEMSYS,
        fields=[
            "material_id",
            "formula_pretty",
            "bulk_modulus",       # Voigt-Reuss-Hill averages
            "shear_modulus",
            "universal_anisotropy",
            "homogeneous_poisson",
        ],
    )

print(f"Found {len(elastic_docs)} entries with elastic data in {TARGET_CHEMSYS}")

rows = []
for doc in elastic_docs:
    bm = doc.bulk_modulus
    sm = doc.shear_modulus
    rows.append({
        "mp_id": str(doc.material_id),
        "formula": doc.formula_pretty,
        "K_vrh": bm.vrh if bm else None,
        "G_vrh": sm.vrh if sm else None,
        "poisson": doc.homogeneous_poisson,
        "anisotropy": doc.universal_anisotropy,
    })

df = pd.DataFrame(rows)
df.dropna(subset=["K_vrh", "G_vrh"], inplace=True)
df.sort_values("K_vrh", ascending=False, inplace=True)

print(df.to_string(index=False))
df.to_csv("elastic_SiO.csv", index=False)

# --- Plot: K vs G with Pugh ratio ---
fig, ax = plt.subplots(figsize=(7, 6))
sc = ax.scatter(df["K_vrh"], df["G_vrh"], c=df["poisson"],
                cmap="coolwarm", s=60, edgecolors="black", linewidth=0.5)
plt.colorbar(sc, label="Poisson ratio")

for _, row in df.iterrows():
    ax.annotate(row["formula"], (row["K_vrh"], row["G_vrh"]),
                fontsize=7, textcoords="offset points", xytext=(4, 4))

# Pugh ductility line: G/K = 0.571 => K/G = 1.75
k_range = [0, df["K_vrh"].max() * 1.1]
ax.plot(k_range, [k * 0.571 for k in k_range],
        "k--", alpha=0.5, label="Pugh line (G/K=0.571)")

ax.set_xlabel("Bulk modulus K (GPa)")
ax.set_ylabel("Shear modulus G (GPa)")
ax.set_title(f"Elastic properties: {TARGET_CHEMSYS}")
ax.legend()
plt.tight_layout()
plt.savefig("elastic_KvsG.png", dpi=150)
plt.close()
print("Saved elastic_KvsG.png")
```

### Pattern 4 -- Multi-Criteria Filtering with Statistical Overview

```python
#!/usr/bin/env python3
"""
Advanced multi-criteria filtering:
  - Chemical system: any oxide
  - Crystal system: cubic
  - Band gap: 0.5 - 2.0 eV (for thermoelectric screening)
  - Thermodynamically stable (e_above_hull = 0)
  - At most 20 atoms per unit cell

Produces a correlation matrix and pairwise scatter plots.
"""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mp_api.client import MPRester

MP_API_KEY = os.environ["MP_API_KEY"]

with MPRester(MP_API_KEY) as mpr:
    docs = mpr.materials.summary.search(
        elements=["O"],
        crystal_system="Cubic",
        band_gap=(0.5, 2.0),
        is_stable=True,
        num_sites=(1, 20),
        fields=[
            "material_id",
            "formula_pretty",
            "band_gap",
            "formation_energy_per_atom",
            "density",
            "volume",
            "nsites",
            "symmetry",
            "energy_above_hull",
        ],
    )

print(f"Found {len(docs)} cubic oxides with Eg in [0.5, 2.0] eV, stable")

rows = []
for doc in docs:
    rows.append({
        "mp_id": str(doc.material_id),
        "formula": doc.formula_pretty,
        "band_gap": doc.band_gap,
        "form_energy": doc.formation_energy_per_atom,
        "density": doc.density,
        "volume_per_atom": doc.volume / doc.nsites if doc.nsites else None,
        "nsites": doc.nsites,
        "spacegroup": doc.symmetry.symbol if doc.symmetry else None,
    })

df = pd.DataFrame(rows)
df.to_csv("cubic_oxides_filtered.csv", index=False)
print(f"Saved cubic_oxides_filtered.csv")
print(df.head(20).to_string(index=False))

# --- Statistical overview ---
numeric_cols = ["band_gap", "form_energy", "density", "volume_per_atom"]
desc = df[numeric_cols].describe()
print("\nDescriptive statistics:")
print(desc.to_string())

# Correlation matrix
corr = df[numeric_cols].corr()
print("\nCorrelation matrix:")
print(corr.to_string())

# --- Plot: correlation heatmap ---
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Heatmap
im = axes[0].imshow(corr.values, cmap="RdBu_r", vmin=-1, vmax=1)
axes[0].set_xticks(range(len(numeric_cols)))
axes[0].set_yticks(range(len(numeric_cols)))
axes[0].set_xticklabels(numeric_cols, rotation=45, ha="right")
axes[0].set_yticklabels(numeric_cols)
for i in range(len(numeric_cols)):
    for j in range(len(numeric_cols)):
        axes[0].text(j, i, f"{corr.values[i,j]:.2f}",
                     ha="center", va="center", fontsize=9)
plt.colorbar(im, ax=axes[0], shrink=0.8)
axes[0].set_title("Property correlations")

# Scatter: band gap vs density
axes[1].scatter(df["band_gap"], df["density"], alpha=0.6, edgecolors="black",
                linewidth=0.3)
axes[1].set_xlabel("Band gap (eV)")
axes[1].set_ylabel("Density (g/cm^3)")
axes[1].set_title("Band gap vs. density")

plt.tight_layout()
plt.savefig("cubic_oxides_analysis.png", dpi=150)
plt.close()
print("Saved cubic_oxides_analysis.png")

# --- Pairwise scatter matrix ---
fig, axes = plt.subplots(len(numeric_cols), len(numeric_cols),
                          figsize=(12, 12))
for i, col_i in enumerate(numeric_cols):
    for j, col_j in enumerate(numeric_cols):
        ax = axes[i][j]
        if i == j:
            ax.hist(df[col_i].dropna(), bins=15, edgecolor="black", alpha=0.7)
        else:
            ax.scatter(df[col_j], df[col_i], alpha=0.4, s=8)
        if i == len(numeric_cols) - 1:
            ax.set_xlabel(col_j, fontsize=7)
        if j == 0:
            ax.set_ylabel(col_i, fontsize=7)
        ax.tick_params(labelsize=6)

plt.suptitle("Pairwise scatter matrix: cubic stable oxides", fontsize=12)
plt.tight_layout()
plt.savefig("cubic_oxides_scatter_matrix.png", dpi=150)
plt.close()
print("Saved cubic_oxides_scatter_matrix.png")
```

### Saving Structures as CIF from Any Query

```python
#!/usr/bin/env python3
"""Utility: download structures from any query and save as CIF files."""

import os
from mp_api.client import MPRester

MP_API_KEY = os.environ["MP_API_KEY"]
OUTPUT_DIR = "downloaded_structures"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Example: all stable perovskites in Ca-Ti-O
with MPRester(MP_API_KEY) as mpr:
    docs = mpr.materials.summary.search(
        chemsys="Ca-Ti-O",
        is_stable=True,
        fields=["material_id", "formula_pretty", "structure"],
    )

for doc in docs:
    mp_id = str(doc.material_id)
    fname = f"{OUTPUT_DIR}/{mp_id}_{doc.formula_pretty}.cif"
    doc.structure.to(fmt="cif", filename=fname)
    print(f"Saved {fname}")

print(f"\nTotal: {len(docs)} structures saved to {OUTPUT_DIR}/")
```

## Key Parameters

### MPRester Search Fields

| Parameter | Type | Description |
|-----------|------|-------------|
| `chemsys` | str | Chemical system, e.g., `"Li-Fe-O"` |
| `elements` | list | Required elements, e.g., `["O"]` |
| `num_elements` | int or tuple | Number of elements, e.g., `2` or `(2, 4)` |
| `band_gap` | tuple | (min, max) in eV |
| `is_stable` | bool | Only on convex hull |
| `crystal_system` | str | `"Cubic"`, `"Hexagonal"`, etc. |
| `num_sites` | tuple | (min, max) atoms per cell |
| `fields` | list | Which data fields to return |

### Common Fields to Request

| Field | Description |
|-------|-------------|
| `material_id` | MP identifier (e.g., `mp-149`) |
| `formula_pretty` | Human-readable formula |
| `structure` | pymatgen Structure object |
| `energy_above_hull` | Thermodynamic stability (eV/atom) |
| `formation_energy_per_atom` | Formation energy (eV/atom) |
| `band_gap` | Electronic band gap (eV) |
| `symmetry` | Space group and crystal system |
| `density` | Density (g/cm^3) |
| `volume` | Unit cell volume (A^3) |

## Interpreting Results

- **`energy_above_hull = 0`** means the phase lies on the convex hull and is thermodynamically stable at 0 K. Values below ~25 meV/atom are often considered "nearly stable" and may be synthesizable.
- **Band gaps from MP** are computed with GGA (PBE), which systematically underestimates gaps. Expect real gaps to be 30-100% larger. For more accurate gaps, look for GW or HSE data, or run your own QE calculation.
- **Elastic data** is not available for all materials. The `elasticity.search()` endpoint only returns materials for which the full elastic tensor was computed.
- **Crystal system filter** uses the conventional cell setting. Some materials may appear in unexpected systems due to symmetry-lowering distortions.

## Common Issues

| Issue | Solution |
|-------|----------|
| `MPRestError: API key not supplied` | `export MP_API_KEY="your_key"` |
| Query returns 0 results | Broaden criteria; check that chemsys string uses the `"A-B-C"` format |
| Rate limiting (429 errors) | Add `time.sleep(1)` between queries; use batch queries |
| `structure` field is None | Some entries lack structures; filter with `if doc.structure is not None` |
| Large downloads are slow | Limit `fields` to only what you need; use `num_sites` to exclude huge cells |
| Elastic data missing | Not all materials have elastic tensors; use `mpr.elasticity.search()` separately |
| Band gap = 0 for a known semiconductor | PBE underestimates gaps; the material may be metallic in GGA. Check with HSE or experiment |
