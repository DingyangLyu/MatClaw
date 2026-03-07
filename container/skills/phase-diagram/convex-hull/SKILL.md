# Thermodynamic Convex Hull and Phase Stability

## When to Use

- Assess whether a material is thermodynamically stable or metastable.
- Compute formation energies and energy above the convex hull.
- Identify decomposition products for unstable phases.
- Build binary or ternary phase diagrams from Materials Project data and/or MACE-computed energies.
- Screen candidate structures for thermodynamic viability.

## Prerequisites

- `pymatgen` (PhaseDiagram, PDPlotter, ComputedEntry)
- `mp-api` (fetching known phases from Materials Project)
- `mace-torch` (computing energies for custom structures)
- `ase` (structure manipulation for MACE calculations)
- `matplotlib` (plotting)
- A valid Materials Project API key (set as environment variable `MP_API_KEY`) when using MP data. Not required for manual entry creation.

## Detailed Steps

### 1. Fetch entries from Materials Project and build a convex hull

```python
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mp_api.client import MPRester
from pymatgen.analysis.phase_diagram import PhaseDiagram, PDPlotter
from pymatgen.entries.computed_entries import ComputedEntry

# --- Fetch entries from Materials Project ---
api_key = os.environ.get("MP_API_KEY")
system = ["Li", "Fe", "O"]  # ternary system

with MPRester(api_key) as mpr:
    entries = mpr.get_entries_in_chemsys(system)

print(f"Fetched {len(entries)} entries for {'-'.join(system)}")

# --- Build the phase diagram ---
pd = PhaseDiagram(entries)

# --- List stable phases ---
print("\n--- Stable phases (on the hull) ---")
for entry in pd.stable_entries:
    print(f"  {entry.composition.reduced_formula:>20s}  "
          f"Ef = {pd.get_form_energy_per_atom(entry):+.4f} eV/atom")

# --- Plot the phase diagram ---
plotter = PDPlotter(pd)
fig = plotter.get_plot()
plt.title(f"{'-'.join(system)} Convex Hull")
plt.tight_layout()
plt.savefig("convex_hull_ternary.png", dpi=150)
plt.close()
print("\nSaved: convex_hull_ternary.png")
```

### 2. Binary phase diagram example

```python
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mp_api.client import MPRester
from pymatgen.analysis.phase_diagram import PhaseDiagram, PDPlotter

api_key = os.environ.get("MP_API_KEY")
system = ["Li", "Si"]  # binary system

with MPRester(api_key) as mpr:
    entries = mpr.get_entries_in_chemsys(system)

print(f"Fetched {len(entries)} entries for {'-'.join(system)}")

pd = PhaseDiagram(entries)

plotter = PDPlotter(pd)
fig = plotter.get_plot()
plt.title(f"{'-'.join(system)} Binary Phase Diagram")
plt.tight_layout()
plt.savefig("convex_hull_binary.png", dpi=150)
plt.close()
print("Saved: convex_hull_binary.png")
```

### 3. Compute energy above hull and decomposition products

```python
import os
from mp_api.client import MPRester
from pymatgen.analysis.phase_diagram import PhaseDiagram

api_key = os.environ.get("MP_API_KEY")

with MPRester(api_key) as mpr:
    entries = mpr.get_entries_in_chemsys(["Li", "Fe", "O"])

pd = PhaseDiagram(entries)

# --- Evaluate every entry ---
print(f"{'Formula':>20s} {'E_above_hull (eV/atom)':>24s}  {'Stable?':>8s}")
print("-" * 60)

for entry in sorted(entries, key=lambda e: pd.get_e_above_hull(e)):
    formula = entry.composition.reduced_formula
    e_above_hull = pd.get_e_above_hull(entry)
    is_stable = "YES" if e_above_hull < 1e-6 else "no"
    print(f"{formula:>20s} {e_above_hull:>24.4f}  {is_stable:>8s}")

    # Show decomposition for unstable phases
    if e_above_hull > 1e-6:
        decomp, e_decomp = pd.get_decomp_and_e_above_hull(entry)
        products = " + ".join(
            f"{v:.2f} {k.composition.reduced_formula}" for k, v in decomp.items()
        )
        print(f"{'':>20s}   -> decomposes to: {products}")
```

### 4. Add custom MACE-computed entries to the phase diagram

```python
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mp_api.client import MPRester
from pymatgen.analysis.phase_diagram import PhaseDiagram, PDPlotter
from pymatgen.entries.computed_entries import ComputedEntry
from pymatgen.core import Composition

# --- Compute energy of a custom structure with MACE ---
from ase.io import read as ase_read
from mace.calculators import mace_mp

calc = mace_mp(model="medium", default_dtype="float64")

# Load your custom structure (POSCAR, CIF, etc.)
atoms = ase_read("my_custom_LiFeO2.cif")
atoms.calc = calc
energy_total = atoms.get_potential_energy()  # eV for full cell
n_atoms = len(atoms)

print(f"MACE total energy: {energy_total:.4f} eV ({n_atoms} atoms)")
print(f"Energy per atom:   {energy_total / n_atoms:.4f} eV/atom")

# --- Fetch MP entries for the chemical system ---
api_key = os.environ.get("MP_API_KEY")
with MPRester(api_key) as mpr:
    entries = mpr.get_entries_in_chemsys(["Li", "Fe", "O"])

# --- Create a ComputedEntry for the custom structure ---
# The energy here is total energy (eV) for the full composition.
# pymatgen normalizes internally.
comp_dict = {}
symbols = atoms.get_chemical_symbols()
for s in symbols:
    comp_dict[s] = comp_dict.get(s, 0) + 1

custom_entry = ComputedEntry(
    composition=Composition(comp_dict),
    energy=energy_total,  # total energy in eV for the full cell
)

# Add to entries list
entries.append(custom_entry)

# --- Rebuild phase diagram with custom entry ---
pd = PhaseDiagram(entries)

e_hull = pd.get_e_above_hull(custom_entry)
print(f"\nCustom phase: {custom_entry.composition.reduced_formula}")
print(f"Energy above hull: {e_hull:.4f} eV/atom")

if e_hull < 1e-6:
    print("Status: STABLE (on the convex hull)")
else:
    print("Status: METASTABLE (above the hull)")
    decomp, _ = pd.get_decomp_and_e_above_hull(custom_entry)
    products = " + ".join(
        f"{v:.2f} {k.composition.reduced_formula}" for k, v in decomp.items()
    )
    print(f"Decomposes to: {products}")

# --- Plot ---
plotter = PDPlotter(pd)
fig = plotter.get_plot()
plt.title("Li-Fe-O Hull with Custom Phase")
plt.tight_layout()
plt.savefig("hull_with_custom.png", dpi=150)
plt.close()
print("\nSaved: hull_with_custom.png")
```

### 5. Without MP API: manual entry creation from computed energies

Use this approach when you have no API key or want full control over
the reference data.

```python
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pymatgen.analysis.phase_diagram import PhaseDiagram, PDPlotter
from pymatgen.entries.computed_entries import ComputedEntry
from pymatgen.core import Composition

# --- Define entries manually ---
# Each entry needs: composition and total energy (eV) for that composition.
# You MUST include the elemental reference states.
# Energies can come from MACE, DFT databases, or literature.

entries = [
    # Elemental references (energy per atom from your calculator)
    # These define the zero of formation energy.
    ComputedEntry("Li",  -1.9000),    # 1 atom of Li
    ComputedEntry("Si",  -5.4230),    # 1 atom of Si

    # Binary phases: total energy for the FULL formula unit
    # Li12Si7: 19 atoms total
    ComputedEntry("Li12Si7", 19 * (-3.8500)),
    # Li7Si3: 10 atoms total
    ComputedEntry("Li7Si3",  10 * (-3.7200)),
    # Li13Si4: 17 atoms total
    ComputedEntry("Li13Si4", 17 * (-3.5800)),
    # Li15Si4: 19 atoms total
    ComputedEntry("Li15Si4", 19 * (-3.4500)),
    # Li21Si5: 26 atoms total
    ComputedEntry("Li21Si5", 26 * (-3.2000)),
    # Li22Si5: 27 atoms total
    ComputedEntry("Li22Si5", 27 * (-3.1800)),
]

# --- Build phase diagram ---
pd = PhaseDiagram(entries)

print("--- Stable phases ---")
for entry in pd.stable_entries:
    formula = entry.composition.reduced_formula
    ef = pd.get_form_energy_per_atom(entry)
    print(f"  {formula:>15s}  Ef = {ef:+.4f} eV/atom")

# --- Evaluate a new candidate ---
candidate = ComputedEntry("Li3Si2", 5 * (-3.9000))
# Temporarily add to entries and rebuild
test_entries = entries + [candidate]
pd_test = PhaseDiagram(test_entries)
e_hull = pd_test.get_e_above_hull(candidate)
print(f"\nCandidate Li3Si2: e_above_hull = {e_hull:.4f} eV/atom")

# --- Plot ---
plotter = PDPlotter(pd)
fig = plotter.get_plot()
plt.title("Li-Si Phase Diagram (Manual Entries)")
plt.tight_layout()
plt.savefig("manual_hull.png", dpi=150)
plt.close()
print("\nSaved: manual_hull.png")
```

### 6. Batch stability screening with MACE

```python
import os
import glob
import numpy as np
from ase.io import read as ase_read
from mace.calculators import mace_mp
from mp_api.client import MPRester
from pymatgen.analysis.phase_diagram import PhaseDiagram
from pymatgen.entries.computed_entries import ComputedEntry
from pymatgen.core import Composition

# --- Set up MACE calculator ---
calc = mace_mp(model="medium", default_dtype="float64")

# --- Compute energies for all candidate structures ---
candidates = []
for cif_file in glob.glob("candidates/*.cif"):
    atoms = ase_read(cif_file)
    atoms.calc = calc
    energy = atoms.get_potential_energy()

    comp_dict = {}
    for s in atoms.get_chemical_symbols():
        comp_dict[s] = comp_dict.get(s, 0) + 1

    entry = ComputedEntry(Composition(comp_dict), energy)
    candidates.append((cif_file, entry))
    print(f"{cif_file}: {entry.composition.reduced_formula}, "
          f"E = {energy:.4f} eV")

# --- Determine the full chemical system ---
all_elements = set()
for _, entry in candidates:
    all_elements.update(entry.composition.get_el_amt_dict().keys())
system = sorted(all_elements)

# --- Fetch MP reference entries ---
api_key = os.environ.get("MP_API_KEY")
with MPRester(api_key) as mpr:
    mp_entries = mpr.get_entries_in_chemsys(system)

# --- Build combined phase diagram ---
all_entries = mp_entries + [e for _, e in candidates]
pd = PhaseDiagram(all_entries)

# --- Report stability ---
print(f"\n{'File':>40s} {'Formula':>15s} {'E_hull (eV/at)':>15s} {'Stable?':>8s}")
print("-" * 85)
for filename, entry in candidates:
    e_hull = pd.get_e_above_hull(entry)
    stable = "YES" if e_hull < 1e-6 else "no"
    print(f"{filename:>40s} {entry.composition.reduced_formula:>15s} "
          f"{e_hull:>15.4f} {stable:>8s}")
```

## Key Parameters

| Parameter | Description |
|-----------|-------------|
| Chemical system | List of elements, e.g., `["Li", "Fe", "O"]`. Must include all elements of interest. |
| `ComputedEntry(composition, energy)` | `composition`: formula string or Composition object. `energy`: **total** energy in eV for the full formula unit (not per atom). |
| `get_form_energy_per_atom(entry)` | Formation energy per atom relative to elemental references on the hull. |
| `get_e_above_hull(entry)` | Energy above the convex hull in eV/atom. Zero means the phase is on the hull. |
| `get_decomp_and_e_above_hull(entry)` | Returns both decomposition products (dict of entry -> fraction) and energy above hull. |

## Interpreting Results

- **e_above_hull = 0**: The phase lies exactly on the convex hull and is thermodynamically stable. It will not spontaneously decompose at 0 K.
- **e_above_hull > 0 but small (< ~0.025 eV/atom)**: Metastable but potentially synthesizable. Many known materials fall in this range. Kinetic barriers may prevent decomposition.
- **e_above_hull > 0.1 eV/atom**: Unlikely to be synthesizable under equilibrium conditions. May exist at high temperature/pressure or as a transient phase.
- **Formation energy < 0**: Compound is stable relative to its constituent elements.
- **Formation energy > 0**: Compound is unstable relative to elements (does not mean it cannot exist; there may be other competing phases).
- **Decomposition products**: The set of stable phases that the unstable compound would decompose into to reach the hull. The fractions indicate the molar amounts.

**Important caveats**:
- Mixing data sources (e.g., MACE energies with MP/DFT energies) introduces systematic errors. Elemental reference energies must be consistent. When possible, compute all entries with the same method.
- The convex hull is a 0 K, 0 pressure construction. Finite temperature effects (entropy, vibrations) are not included.
- MP entries use specific DFT settings (PBE+U for transition metals). MACE energies may not align perfectly with MP energies.

## Common Issues

| Issue | Solution |
|-------|----------|
| `ValueError: No terminal entries` | You are missing elemental reference entries for one or more elements. Ensure all elements in your compositions have at least one pure-element entry. |
| MP API key not set | Set `MP_API_KEY` environment variable: `os.environ["MP_API_KEY"] = "your_key"` or export in shell. |
| Mixing MACE and MP energies gives wrong hull | Use a consistent energy source for all entries, or apply energy corrections. Compute elemental references with the same calculator. |
| `ComputedEntry` energy units | Energy must be in eV (total, not per atom). pymatgen normalizes per atom internally. |
| Phase diagram too crowded to read | Use `PDPlotter(pd, show_unstable=0)` to show only stable phases, or `show_unstable=0.05` to limit to phases within 50 meV/atom of the hull. |
| Ternary plot axes unclear | pymatgen uses Gibbs triangle coordinates. The three corners correspond to the three elements. |
