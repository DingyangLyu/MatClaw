# Pourbaix Diagram (Electrochemical Stability)

## When to Use

- Determine electrochemical stability of a material as a function of pH and electrode potential.
- Identify corrosion, passivation, and immunity regions for metals and oxides.
- Assess stability of battery electrode materials in aqueous environments.
- Evaluate catalyst stability under operating conditions.
- Screen materials for aqueous electrochemistry applications (water splitting, CO2 reduction).

## Prerequisites

- `pymatgen` (PourbaixDiagram, PourbaixPlotter, PourbaixEntry)
- `mp-api` (fetching solid and ion data from Materials Project)
- `matplotlib` (plotting)
- A valid Materials Project API key (set as environment variable `MP_API_KEY`).

## Detailed Steps

### 1. Build a Pourbaix diagram from Materials Project data

```python
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mp_api.client import MPRester
from pymatgen.analysis.pourbaix_diagram import PourbaixDiagram, PourbaixPlotter

api_key = os.environ.get("MP_API_KEY")

# --- Fetch Pourbaix entries for a single element (e.g., Fe) ---
with MPRester(api_key) as mpr:
    pourbaix_entries = mpr.get_pourbaix_entries(["Fe"])

print(f"Fetched {len(pourbaix_entries)} Pourbaix entries for Fe")

# --- Build Pourbaix diagram ---
# Default ion concentration: 1e-6 M (dilute solution)
pbx = PourbaixDiagram(pourbaix_entries, comp_dict={"Fe": 1.0})

# --- Plot ---
plotter = PourbaixPlotter(pbx)
fig, ax = plt.subplots(figsize=(10, 7))
plotter.get_pourbaix_plot(ax=ax)
ax.set_title("Pourbaix Diagram for Fe", fontsize=14)
ax.set_xlabel("pH", fontsize=12)
ax.set_ylabel("Potential vs SHE (V)", fontsize=12)
plt.tight_layout()
plt.savefig("pourbaix_Fe.png", dpi=150)
plt.close()
print("Saved: pourbaix_Fe.png")
```

### 2. Multi-element Pourbaix diagram (e.g., Fe-Cr for stainless steel)

```python
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mp_api.client import MPRester
from pymatgen.analysis.pourbaix_diagram import PourbaixDiagram, PourbaixPlotter

api_key = os.environ.get("MP_API_KEY")

# --- Fetch entries for the Fe-Cr system ---
with MPRester(api_key) as mpr:
    pourbaix_entries = mpr.get_pourbaix_entries(["Fe", "Cr"])

print(f"Fetched {len(pourbaix_entries)} entries for Fe-Cr system")

# --- Build diagram with specified composition ---
# comp_dict gives the fractional composition of the metallic elements
# For an alloy Fe0.7Cr0.3:
pbx = PourbaixDiagram(
    pourbaix_entries,
    comp_dict={"Fe": 0.7, "Cr": 0.3},
)

# --- Plot ---
plotter = PourbaixPlotter(pbx)
fig, ax = plt.subplots(figsize=(10, 7))
plotter.get_pourbaix_plot(ax=ax)
ax.set_title("Pourbaix Diagram for Fe$_{0.7}$Cr$_{0.3}$", fontsize=14)
ax.set_xlabel("pH", fontsize=12)
ax.set_ylabel("Potential vs SHE (V)", fontsize=12)
plt.tight_layout()
plt.savefig("pourbaix_FeCr.png", dpi=150)
plt.close()
print("Saved: pourbaix_FeCr.png")
```

### 3. Query stable species at specific pH and potential

```python
import os
from mp_api.client import MPRester
from pymatgen.analysis.pourbaix_diagram import PourbaixDiagram

api_key = os.environ.get("MP_API_KEY")

with MPRester(api_key) as mpr:
    pourbaix_entries = mpr.get_pourbaix_entries(["Fe"])

pbx = PourbaixDiagram(pourbaix_entries, comp_dict={"Fe": 1.0})

# --- Check stability at specific conditions ---
conditions = [
    (7.0, 0.0),     # neutral pH, zero potential
    (0.0, -0.5),    # acidic, reducing
    (14.0, 0.8),    # alkaline, oxidizing
    (3.0, 0.5),     # mildly acidic, oxidizing
    (7.0, -0.8),    # neutral, strongly reducing
]

print(f"{'pH':>5s} {'E (V)':>7s}  {'Stable Species':>30s}  {'E_decomp (eV/atom)':>20s}")
print("-" * 70)

for ph, potential in conditions:
    # Get the stable entry at this pH and potential
    stable_entry = pbx.get_stable_entry(ph, potential)

    # Get the decomposition energy (how far from the stable region
    # the current conditions are for any given entry)
    e_decomp = pbx.get_decomposition_energy(
        pbx.all_entries[0], ph, potential
    )

    species_name = stable_entry.name if hasattr(stable_entry, 'name') else str(stable_entry)
    print(f"{ph:>5.1f} {potential:>7.2f}  {species_name:>30s}  {e_decomp:>20.4f}")
```

### 4. Assess stability of a specific material

```python
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mp_api.client import MPRester
from pymatgen.analysis.pourbaix_diagram import PourbaixDiagram, PourbaixPlotter

api_key = os.environ.get("MP_API_KEY")

# --- Example: assess stability of a battery cathode material ---
# LiMn2O4 (spinel) stability in aqueous solution
with MPRester(api_key) as mpr:
    pourbaix_entries = mpr.get_pourbaix_entries(["Li", "Mn"])

print(f"Fetched {len(pourbaix_entries)} entries")

pbx = PourbaixDiagram(
    pourbaix_entries,
    comp_dict={"Li": 1/3, "Mn": 2/3},  # Li:Mn = 1:2
)

# --- Map decomposition energy across pH-potential space ---
ph_range = np.linspace(-2, 16, 100)
v_range = np.linspace(-2, 3, 100)
PH, V = np.meshgrid(ph_range, v_range)

# Find the entry corresponding to a solid of interest
target_entry = None
for entry in pbx.all_entries:
    name = entry.name if hasattr(entry, 'name') else ""
    if "LiMn2O4" in name or "LiMn2(O4)" in name:
        target_entry = entry
        break

if target_entry is not None:
    print(f"Found target entry: {target_entry.name}")
    # Compute decomposition energy at each point
    E_decomp = np.zeros_like(PH)
    for i in range(PH.shape[0]):
        for j in range(PH.shape[1]):
            E_decomp[i, j] = pbx.get_decomposition_energy(
                target_entry, PH[i, j], V[i, j]
            )

    fig, ax = plt.subplots(figsize=(10, 7))
    im = ax.contourf(PH, V, E_decomp, levels=20, cmap="RdYlGn_r")
    plt.colorbar(im, ax=ax, label="Decomposition Energy (eV/atom)")
    ax.contour(PH, V, E_decomp, levels=[0], colors="black", linewidths=2)
    ax.set_xlabel("pH", fontsize=12)
    ax.set_ylabel("Potential vs SHE (V)", fontsize=12)
    ax.set_title("LiMn2O4 Stability Map", fontsize=14)
    plt.tight_layout()
    plt.savefig("stability_map_LiMn2O4.png", dpi=150)
    plt.close()
    print("Saved: stability_map_LiMn2O4.png")
else:
    print("Target entry not found. Plotting overall Pourbaix diagram instead.")
    plotter = PourbaixPlotter(pbx)
    fig, ax = plt.subplots(figsize=(10, 7))
    plotter.get_pourbaix_plot(ax=ax)
    plt.tight_layout()
    plt.savefig("pourbaix_LiMn.png", dpi=150)
    plt.close()
    print("Saved: pourbaix_LiMn.png")
```

### 5. Effect of ion concentration

```python
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mp_api.client import MPRester
from pymatgen.analysis.pourbaix_diagram import PourbaixDiagram, PourbaixPlotter

api_key = os.environ.get("MP_API_KEY")

with MPRester(api_key) as mpr:
    pourbaix_entries = mpr.get_pourbaix_entries(["Cu"])

# --- Compare diagrams at different ion concentrations ---
concentrations = [1e-8, 1e-6, 1e-4, 1e-2]

fig, axes = plt.subplots(2, 2, figsize=(16, 12))
axes = axes.flatten()

for idx, conc in enumerate(concentrations):
    # Filter and set concentration on ion entries
    entries_copy = []
    for entry in pourbaix_entries:
        entries_copy.append(entry)

    pbx = PourbaixDiagram(
        entries_copy,
        comp_dict={"Cu": 1.0},
        conc_dict={"Cu": conc},
    )

    plotter = PourbaixPlotter(pbx)
    plotter.get_pourbaix_plot(ax=axes[idx])
    axes[idx].set_title(f"[Cu] = {conc:.0e} M", fontsize=12)
    axes[idx].set_xlabel("pH")
    axes[idx].set_ylabel("E vs SHE (V)")

plt.suptitle("Cu Pourbaix Diagram - Effect of Ion Concentration", fontsize=14)
plt.tight_layout()
plt.savefig("pourbaix_Cu_concentrations.png", dpi=150)
plt.close()
print("Saved: pourbaix_Cu_concentrations.png")
```

### 6. Water stability window overlay

```python
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mp_api.client import MPRester
from pymatgen.analysis.pourbaix_diagram import PourbaixDiagram, PourbaixPlotter

api_key = os.environ.get("MP_API_KEY")

with MPRester(api_key) as mpr:
    pourbaix_entries = mpr.get_pourbaix_entries(["Ti"])

pbx = PourbaixDiagram(pourbaix_entries, comp_dict={"Ti": 1.0})

# --- Plot with water stability lines ---
plotter = PourbaixPlotter(pbx)
fig, ax = plt.subplots(figsize=(10, 7))
plotter.get_pourbaix_plot(ax=ax)

# Add water stability window
# O2/H2O line:  E = 1.229 - 0.0592 * pH  (at 25 C)
# H+/H2 line:   E = 0.000 - 0.0592 * pH
ph_line = np.linspace(-2, 16, 100)
e_o2 = 1.229 - 0.0592 * ph_line
e_h2 = 0.000 - 0.0592 * ph_line

ax.plot(ph_line, e_o2, "b--", linewidth=2, label="O$_2$/H$_2$O")
ax.plot(ph_line, e_h2, "b--", linewidth=2, label="H$^+$/H$_2$")
ax.fill_between(ph_line, e_h2, e_o2, alpha=0.05, color="blue")
ax.legend(loc="upper right")
ax.set_title("Ti Pourbaix Diagram with Water Stability Window", fontsize=14)
ax.set_xlabel("pH", fontsize=12)
ax.set_ylabel("Potential vs SHE (V)", fontsize=12)
plt.tight_layout()
plt.savefig("pourbaix_Ti_water.png", dpi=150)
plt.close()
print("Saved: pourbaix_Ti_water.png")
```

## Key Parameters

| Parameter | Description |
|-----------|-------------|
| `comp_dict` | Dictionary of element fractions, e.g., `{"Fe": 0.7, "Cr": 0.3}`. Fractions should sum to 1 for multi-element systems. |
| `conc_dict` | Dictionary of ion concentrations in mol/L, e.g., `{"Fe": 1e-6}`. Default is typically 1e-6 M. Lower concentrations expand the corrosion region. |
| `filter_solids` | If `True` (default), filters out metastable solid phases not on the convex hull. |
| pH range | Typical range: -2 to 16. Standard conditions: pH 7. |
| Potential range | Typical range: -2 to 3 V vs SHE. Negative = reducing, positive = oxidizing. |

## Interpreting Results

Pourbaix diagrams divide pH-potential space into regions where different species are thermodynamically stable:

- **Immunity region**: The bare metal is stable. No corrosion occurs. Found at low (reducing) potentials. The metal is thermodynamically protected.

- **Corrosion region**: Dissolved ionic species (e.g., Fe2+, Fe3+, HFeO2-) are stable. The metal dissolves. Active corrosion occurs in these regions. Represented by aqueous ions in the diagram.

- **Passivation region**: A solid oxide, hydroxide, or other compound forms on the surface. This may protect the metal from further corrosion if the film is dense and adherent. Represented by solid phases like Fe2O3, Fe3O4, Cr2O3.

**Water stability window**: The region between the O2/H2O and H+/H2 lines. Outside this window, water itself is unstable (either oxidized to O2 or reduced to H2). Relevant for aqueous electrochemistry.

**Application guidance**:

| Application | What to Look For |
|-------------|-----------------|
| Corrosion resistance | Large passivation regions at operating pH; immunity at achievable potentials |
| Battery electrodes | Stability of electrode material within the water window at operating pH |
| Electrocatalysis | Stability at the potential and pH of the reaction (e.g., OER at pH 14, E ~ 1.23 V) |
| Photocatalysis | Stability at the band edge potentials of the semiconductor |

## Common Issues

| Issue | Solution |
|-------|----------|
| `get_pourbaix_entries` fails | Check API key. Some elements have limited aqueous ion data in MP. |
| Diagram looks empty or wrong | Verify the `comp_dict` fractions sum to 1. Check that the elements match the fetched entries. |
| Ion concentration not taking effect | Use `conc_dict` parameter in `PourbaixDiagram`. The default is 1e-6 M. Ensure keys match element symbols. |
| Very slow for multi-element systems | Multi-element Pourbaix diagrams (3+ metals) require computing many competing phases. This can be slow. Reduce the system size if possible. |
| `KeyError` on element | The element may not have Pourbaix data in MP. Check `mpr.get_pourbaix_entries()` returns non-empty results. |
| Plot axes wrong or labels missing | Use `plotter.get_pourbaix_plot(ax=ax)` and then customize the axes labels/title on the returned axes object. |
| Temperature effects | Standard Pourbaix diagrams are at 25 C. For other temperatures, the Nernst slopes change (0.0592 V/pH at 25 C becomes RT/F * ln(10) at temperature T). pymatgen does not handle temperature-dependent Pourbaix diagrams natively. |
