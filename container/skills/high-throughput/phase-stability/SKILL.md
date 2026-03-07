# Phase Stability and Convex Hull Analysis

## When to Use

- Determine whether a candidate material is thermodynamically stable or metastable.
- Compute formation energies of competing phases in a chemical system.
- Build the convex hull and identify which compositions lie on it.
- Calculate the energy above hull for custom structures (from MACE or DFT).
- Find decomposition products and decomposition energy for unstable phases.
- Construct binary, ternary, or pseudo-binary phase diagram sections.
- Screen a batch of candidate structures for thermodynamic viability before synthesis.

## Method Selection

| Approach | Data Source | Accuracy | Use When |
|---|---|---|---|
| MP-only convex hull | Materials Project API | MP DFT (PBE/PBE+U) | Need reference hull for known systems |
| MACE energies + MP hull | MACE for candidates, MP for references | ~10-30 meV/atom systematic offset | Rapid screening of new candidates |
| MACE-only hull | MACE for everything | Internally consistent but approximate | No API key, or exploring hypothetical compositions |
| QE DFT + MP hull | QE for candidates, MP for references | Publication quality (with corrections) | Final validation of top candidates |
| Fully DFT hull | QE/VASP for all phases | Best internal consistency | Small systems, publication-quality phase diagrams |

## Prerequisites

- `pymatgen` (PhaseDiagram, PDPlotter, PDEntry, ComputedEntry).
- `mp-api` (for fetching reference phases from Materials Project).
- `mace-torch` + `ase` (for computing energies of custom structures).
- `matplotlib` (for plotting phase diagrams).
- `pandas`, `numpy` (for data management).
- `MP_API_KEY` environment variable when using Materials Project data.

## Detailed Steps

### 1. Complete formation energy and convex hull workflow

```python
#!/usr/bin/env python3
"""
Phase stability analysis: compute formation energies, build convex hull,
evaluate stability of candidate materials.

Pipeline:
  1. Fetch reference entries from Materials Project for the chemical system.
  2. Compute MACE energies for custom candidate structures.
  3. Build the convex hull with all entries.
  4. Evaluate energy above hull and decomposition for each candidate.
  5. Generate phase diagram plots and summary tables.
"""

import os
import glob
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from mp_api.client import MPRester
from pymatgen.core import Structure, Composition, Element
from pymatgen.analysis.phase_diagram import (
    PhaseDiagram, PDPlotter, PDEntry, GrandPotentialPhaseDiagram
)
from pymatgen.entries.computed_entries import ComputedEntry
from pymatgen.io.ase import AseAtomsAdaptor

from ase.io import read as ase_read
from ase.optimize import LBFGS
from ase.constraints import ExpCellFilter
from mace.calculators import mace_mp

# ================================================================== #
#  CONFIGURATION
# ================================================================== #
MP_API_KEY = os.environ.get("MP_API_KEY")
assert MP_API_KEY, "Set MP_API_KEY environment variable first."

# Chemical system of interest
SYSTEM_ELEMENTS = ["Li", "Mn", "O"]

# Directory containing candidate structures (CIF, POSCAR, etc.)
CANDIDATES_DIR = "candidates"

# MACE settings
MACE_MODEL = "medium"
FMAX = 0.01               # eV/Ang for relaxation
RELAX_CELL = True          # also relax cell parameters

# Output
OUTPUT_DIR = "phase_stability_results"
# ================================================================== #

os.makedirs(OUTPUT_DIR, exist_ok=True)
adaptor = AseAtomsAdaptor()
calc = mace_mp(model=MACE_MODEL, default_dtype="float64")


# ================================================================== #
#  STEP 1: Fetch reference entries from Materials Project
# ================================================================== #
print("Step 1: Fetching reference entries from Materials Project ...")

with MPRester(MP_API_KEY) as mpr:
    mp_entries = mpr.get_entries_in_chemsys(SYSTEM_ELEMENTS)

print(f"  Fetched {len(mp_entries)} entries for "
      f"{'-'.join(SYSTEM_ELEMENTS)} system.")

# Identify elements and their reference energies
pd_mp = PhaseDiagram(mp_entries)
print(f"  Stable phases on MP hull: {len(pd_mp.stable_entries)}")
for entry in pd_mp.stable_entries:
    ef = pd_mp.get_form_energy_per_atom(entry)
    print(f"    {entry.composition.reduced_formula:>15s}  "
          f"Ef = {ef:+.4f} eV/atom")


# ================================================================== #
#  STEP 2: Compute MACE energies for candidate structures
# ================================================================== #
print("\nStep 2: Computing MACE energies for candidates ...")

candidate_files = sorted(
    glob.glob(os.path.join(CANDIDATES_DIR, "*.cif"))
    + glob.glob(os.path.join(CANDIDATES_DIR, "*.vasp"))
    + glob.glob(os.path.join(CANDIDATES_DIR, "POSCAR*"))
)

if not candidate_files:
    print(f"  No structure files found in {CANDIDATES_DIR}/")
    print("  Place CIF or POSCAR files in that directory and re-run.")
    # Continue with MP-only hull for demonstration
    candidate_entries = []
else:
    candidate_entries = []
    for filepath in candidate_files:
        basename = os.path.basename(filepath)
        try:
            structure = Structure.from_file(filepath)
        except Exception as e:
            print(f"  ERROR reading {filepath}: {e}")
            continue

        # Convert to ASE and relax with MACE
        atoms = adaptor.get_atoms(structure)
        atoms.calc = calc

        try:
            if RELAX_CELL:
                ecf = ExpCellFilter(atoms, hydrostatic_strain=False)
                opt = LBFGS(ecf, logfile=None)
            else:
                opt = LBFGS(atoms, logfile=None)
            opt.run(fmax=FMAX, steps=300)
            energy_total = atoms.get_potential_energy()
        except Exception as e:
            print(f"  MACE failed for {basename}: {e}")
            continue

        # Build ComputedEntry
        comp_dict = {}
        for s in atoms.get_chemical_symbols():
            comp_dict[s] = comp_dict.get(s, 0) + 1

        entry = ComputedEntry(
            composition=Composition(comp_dict),
            energy=energy_total,
        )

        n = len(atoms)
        print(f"  {basename:>30s}  {entry.composition.reduced_formula:>12s}  "
              f"E = {energy_total:.4f} eV  ({energy_total/n:.4f} eV/atom)")

        candidate_entries.append({
            "filename": basename,
            "entry": entry,
            "structure": adaptor.get_structure(atoms),
            "energy_total": energy_total,
            "energy_per_atom": energy_total / n,
            "n_atoms": n,
        })


# ================================================================== #
#  STEP 3: Build combined convex hull
# ================================================================== #
print("\nStep 3: Building convex hull ...")

# Combine MP entries with candidate entries
all_entries = list(mp_entries) + [c["entry"] for c in candidate_entries]
pd_combined = PhaseDiagram(all_entries)

print(f"  Total entries: {len(all_entries)}")
print(f"  Stable phases: {len(pd_combined.stable_entries)}")


# ================================================================== #
#  STEP 4: Evaluate each candidate
# ================================================================== #
print("\nStep 4: Evaluating candidate stability ...")

results = []
for cand in candidate_entries:
    entry = cand["entry"]
    formula = entry.composition.reduced_formula

    e_hull = pd_combined.get_e_above_hull(entry)
    form_energy = pd_combined.get_form_energy_per_atom(entry)
    is_stable = e_hull < 1e-6

    # Get decomposition products
    decomp_str = ""
    if not is_stable:
        decomp, _ = pd_combined.get_decomp_and_e_above_hull(entry)
        decomp_str = " + ".join(
            f"{v:.3f} {k.composition.reduced_formula}"
            for k, v in decomp.items()
        )

    status = "STABLE" if is_stable else "METASTABLE"
    print(f"  {cand['filename']:>30s}  {formula:>12s}  "
          f"E_hull = {e_hull:.4f} eV/atom  "
          f"Ef = {form_energy:+.4f} eV/atom  [{status}]")
    if decomp_str:
        print(f"  {'':>30s}  Decomposes to: {decomp_str}")

    results.append({
        "filename": cand["filename"],
        "formula": formula,
        "n_atoms": cand["n_atoms"],
        "energy_total_eV": cand["energy_total"],
        "energy_per_atom_eV": cand["energy_per_atom"],
        "formation_energy_eV_atom": form_energy,
        "e_above_hull_eV_atom": e_hull,
        "is_stable": is_stable,
        "decomposition": decomp_str,
    })

# Export results
df = pd.DataFrame(results)
df.sort_values("e_above_hull_eV_atom", inplace=True)
df.reset_index(drop=True, inplace=True)

csv_path = os.path.join(OUTPUT_DIR, "phase_stability.csv")
df.to_csv(csv_path, index=False)
print(f"\nResults saved to {csv_path}")


# ================================================================== #
#  STEP 5: Phase diagram plots
# ================================================================== #
print("\nStep 5: Generating plots ...")

# --- Convex hull plot (pymatgen built-in) ---
plotter = PDPlotter(pd_combined, show_unstable=0.1)
fig = plotter.get_plot()
plt.title(f"{'-'.join(SYSTEM_ELEMENTS)} Phase Diagram "
          f"(MP + MACE candidates)")
plt.tight_layout()
hull_path = os.path.join(OUTPUT_DIR, "convex_hull.png")
plt.savefig(hull_path, dpi=150, bbox_inches="tight")
plt.close()
print(f"  Saved {hull_path}")

# --- Bar chart: energy above hull for all candidates ---
if len(results) > 0:
    fig, ax = plt.subplots(figsize=(max(8, len(results) * 0.8), 5))

    formulas = [r["formula"] for r in results]
    e_hulls = [r["e_above_hull_eV_atom"] for r in results]
    colors = ["green" if e < 1e-6 else "orange" if e < 0.025
              else "red" for e in e_hulls]

    bars = ax.bar(range(len(formulas)), e_hulls, color=colors,
                  edgecolor="black")
    ax.set_xticks(range(len(formulas)))
    ax.set_xticklabels(formulas, rotation=45, ha="right")
    ax.set_ylabel("Energy above hull (eV/atom)")
    ax.set_title("Candidate phase stability")

    # Reference lines
    ax.axhline(0, color="black", linewidth=0.5)
    ax.axhline(0.025, color="orange", linestyle="--", alpha=0.7,
               label="25 meV/atom (synthesizability threshold)")
    ax.legend()

    plt.tight_layout()
    bar_path = os.path.join(OUTPUT_DIR, "stability_bar_chart.png")
    plt.savefig(bar_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved {bar_path}")

# --- Formation energy vs composition (for binary subsystems) ---
if len(SYSTEM_ELEMENTS) == 2:
    fig, ax = plt.subplots(figsize=(8, 5))

    # Plot hull entries
    el_a, el_b = SYSTEM_ELEMENTS
    hull_x, hull_y = [], []
    for entry in pd_combined.stable_entries:
        comp = entry.composition.fractional_composition
        x = comp.get(Element(el_b), 0)
        y = pd_combined.get_form_energy_per_atom(entry)
        hull_x.append(x)
        hull_y.append(y)
        ax.annotate(entry.composition.reduced_formula, (x, y),
                    textcoords="offset points", xytext=(3, 5), fontsize=8)

    # Sort and plot hull line
    order = np.argsort(hull_x)
    ax.plot(np.array(hull_x)[order], np.array(hull_y)[order],
            "b-o", markersize=8, label="Convex hull")

    # Plot unstable entries
    for entry in all_entries:
        if entry not in pd_combined.stable_entries:
            comp = entry.composition.fractional_composition
            x = comp.get(Element(el_b), 0)
            y = pd_combined.get_form_energy_per_atom(entry)
            ax.scatter(x, y, c="gray", s=15, alpha=0.3, zorder=1)

    # Highlight candidates
    for cand in candidate_entries:
        entry = cand["entry"]
        comp = entry.composition.fractional_composition
        x = comp.get(Element(el_b), 0)
        y = pd_combined.get_form_energy_per_atom(entry)
        ax.scatter(x, y, c="red", s=60, zorder=5, edgecolors="black",
                   marker="*")
        ax.annotate(entry.composition.reduced_formula, (x, y),
                    textcoords="offset points", xytext=(5, 5),
                    fontsize=8, color="red")

    ax.set_xlabel(f"Fraction of {el_b}")
    ax.set_ylabel("Formation energy (eV/atom)")
    ax.set_title(f"{el_a}-{el_b} formation energy diagram")
    ax.legend()
    plt.tight_layout()
    fe_path = os.path.join(OUTPUT_DIR, "formation_energy_diagram.png")
    plt.savefig(fe_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved {fe_path}")

print(f"\nPhase stability analysis complete. Results in {OUTPUT_DIR}/")
```

### 2. Grand potential phase diagram (variable chemical potential)

Use this when you want to explore stability as a function of an element's chemical potential (e.g., oxygen partial pressure, lithium reservoir).

```python
#!/usr/bin/env python3
"""
Grand potential phase diagram: stability as a function of chemical potential.

Example: Li-Mn-O system with variable Li chemical potential
(relevant for battery electrode stability).
"""

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from mp_api.client import MPRester
from pymatgen.core import Element
from pymatgen.analysis.phase_diagram import (
    PhaseDiagram, GrandPotentialPhaseDiagram
)

MP_API_KEY = os.environ.get("MP_API_KEY")
SYSTEM = ["Li", "Mn", "O"]
OPEN_ELEMENT = "Li"          # element with variable chemical potential
MU_RANGE = (-4.0, 0.0)       # range of chemical potential (eV)
N_MU_POINTS = 50

os.makedirs("grand_potential_results", exist_ok=True)

# Fetch entries
with MPRester(MP_API_KEY) as mpr:
    entries = mpr.get_entries_in_chemsys(SYSTEM)

print(f"Fetched {len(entries)} entries for {'-'.join(SYSTEM)}")

# Build canonical phase diagram first
pd_canonical = PhaseDiagram(entries)

# Sweep over chemical potential values
mu_values = np.linspace(MU_RANGE[0], MU_RANGE[1], N_MU_POINTS)
stability_data = {}  # formula -> list of (mu, e_above_hull)

for mu in mu_values:
    chempots = {Element(OPEN_ELEMENT): mu}
    gpd = GrandPotentialPhaseDiagram(entries, chempots)

    for entry in gpd.stable_entries:
        formula = entry.original_entry.composition.reduced_formula
        if formula not in stability_data:
            stability_data[formula] = []
        stability_data[formula].append((mu, 0.0))

# --- Plot stability regions ---
fig, ax = plt.subplots(figsize=(10, 6))

colors = plt.cm.tab10(np.linspace(0, 1, len(stability_data)))
for idx, (formula, points) in enumerate(sorted(stability_data.items())):
    if len(points) < 2:
        continue
    mus = [p[0] for p in points]
    mu_min, mu_max = min(mus), max(mus)
    ax.barh(idx, mu_max - mu_min, left=mu_min, height=0.6,
            color=colors[idx], edgecolor="black", alpha=0.8,
            label=formula)

ax.set_xlabel(f"Chemical potential of {OPEN_ELEMENT} (eV)")
ax.set_ylabel("Stable phase")
ax.set_yticks(range(len(stability_data)))
ax.set_yticklabels(sorted(stability_data.keys()), fontsize=8)
ax.set_title(f"Grand potential stability: {'-'.join(SYSTEM)}, "
             f"open element = {OPEN_ELEMENT}")
ax.set_xlim(MU_RANGE)

plt.tight_layout()
plt.savefig("grand_potential_results/stability_regions.png",
            dpi=150, bbox_inches="tight")
plt.close()
print("Saved grand_potential_results/stability_regions.png")
```

### 3. Pseudo-binary phase diagram section

For ternary or higher systems, plot a phase diagram section between two endpoint compositions.

```python
#!/usr/bin/env python3
"""
Pseudo-binary phase diagram section.
Example: LiMnO2 -- Li2MnO3 join in the Li-Mn-O system.
"""

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from mp_api.client import MPRester
from pymatgen.core import Composition, Element
from pymatgen.analysis.phase_diagram import PhaseDiagram

MP_API_KEY = os.environ.get("MP_API_KEY")
SYSTEM = ["Li", "Mn", "O"]

# Endpoint compositions for the pseudo-binary section
COMP_A = Composition("LiMnO2")    # x = 0
COMP_B = Composition("Li2MnO3")   # x = 1

N_POINTS = 100   # interpolation points along the section

# Fetch all entries
with MPRester(MP_API_KEY) as mpr:
    entries = mpr.get_entries_in_chemsys(SYSTEM)

pd_full = PhaseDiagram(entries)

# --- Interpolate along the pseudo-binary join ---
x_values = np.linspace(0, 1, N_POINTS)
form_energies = []

for x in x_values:
    # Interpolate composition: (1-x)*COMP_A + x*COMP_B
    comp_dict = {}
    for el in set(list(COMP_A.get_el_amt_dict().keys())
                  + list(COMP_B.get_el_amt_dict().keys())):
        amt_a = COMP_A.get(Element(el), 0) * (1 - x)
        amt_b = COMP_B.get(Element(el), 0) * x
        comp_dict[el] = amt_a + amt_b

    comp = Composition(comp_dict)

    # Get the hull energy at this composition
    try:
        hull_energy = pd_full.get_hull_energy_per_atom(comp)
        form_energies.append(hull_energy)
    except Exception:
        form_energies.append(np.nan)

form_energies = np.array(form_energies)

# --- Find stable phases along the section ---
fig, ax = plt.subplots(figsize=(10, 6))

ax.plot(x_values, form_energies, "b-", linewidth=2, label="Hull energy")

# Mark stable phases that lie on or near the join
for entry in pd_full.stable_entries:
    comp = entry.composition.reduced_composition
    ef = pd_full.get_form_energy_per_atom(entry)

    # Check if this composition lies on the A-B join
    # by projecting onto the line parameterized by x
    try:
        # Simple check: is the composition a linear combination of A and B?
        comp_norm = comp.fractional_composition
        a_norm = COMP_A.fractional_composition
        b_norm = COMP_B.fractional_composition

        # Find x such that comp = (1-x)*A + x*B (element-wise)
        elements = list(set(list(a_norm.get_el_amt_dict().keys())
                           + list(b_norm.get_el_amt_dict().keys())))
        x_vals_fit = []
        for el in elements:
            a_frac = a_norm.get(Element(el), 0)
            b_frac = b_norm.get(Element(el), 0)
            c_frac = comp_norm.get(Element(el), 0)
            if abs(b_frac - a_frac) > 1e-10:
                x_est = (c_frac - a_frac) / (b_frac - a_frac)
                x_vals_fit.append(x_est)

        if x_vals_fit and np.std(x_vals_fit) < 0.05:
            x_proj = np.mean(x_vals_fit)
            if -0.05 <= x_proj <= 1.05:
                ax.scatter(x_proj, ef, c="red", s=80, zorder=5,
                           edgecolors="black")
                ax.annotate(comp.reduced_formula, (x_proj, ef),
                            textcoords="offset points", xytext=(5, 8),
                            fontsize=9)
    except Exception:
        pass

ax.set_xlabel(f"x in (1-x){COMP_A.reduced_formula} + "
              f"x {COMP_B.reduced_formula}")
ax.set_ylabel("Formation energy (eV/atom)")
ax.set_title(f"Pseudo-binary section: {COMP_A.reduced_formula} -- "
             f"{COMP_B.reduced_formula}")
ax.legend()
plt.tight_layout()
plt.savefig("pseudo_binary_section.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved pseudo_binary_section.png")
```

### 4. MACE-only convex hull (no API key needed)

Build a convex hull entirely from MACE-computed energies when no API key is available or when testing hypothetical compositions.

```python
#!/usr/bin/env python3
"""
Build a convex hull entirely from MACE-computed energies.
No Materials Project API key needed.

You must provide:
  - Elemental reference structures (one per element)
  - All candidate phase structures
  - All known competing phases you want on the hull
in the input directory.
"""

import os
import glob
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from ase.io import read as ase_read
from ase.optimize import LBFGS
from ase.constraints import ExpCellFilter
from mace.calculators import mace_mp

from pymatgen.core import Composition
from pymatgen.analysis.phase_diagram import PhaseDiagram, PDPlotter
from pymatgen.entries.computed_entries import ComputedEntry

# ================================================================== #
STRUCTURES_DIR = "all_phases"    # directory with ALL structures (elements + compounds)
MACE_MODEL = "medium"
FMAX = 0.01
OUTPUT_DIR = "mace_hull_results"
# ================================================================== #

os.makedirs(OUTPUT_DIR, exist_ok=True)
calc = mace_mp(model=MACE_MODEL, default_dtype="float64")

# --- Compute MACE energies for all structures ---
entries = []
structure_files = sorted(
    glob.glob(os.path.join(STRUCTURES_DIR, "*.cif"))
    + glob.glob(os.path.join(STRUCTURES_DIR, "*.vasp"))
    + glob.glob(os.path.join(STRUCTURES_DIR, "POSCAR*"))
)

print(f"Computing MACE energies for {len(structure_files)} structures ...")

for filepath in structure_files:
    basename = os.path.basename(filepath)
    atoms = ase_read(filepath)
    atoms.calc = calc

    try:
        ecf = ExpCellFilter(atoms, hydrostatic_strain=False)
        opt = LBFGS(ecf, logfile=None)
        opt.run(fmax=FMAX, steps=300)
        energy = atoms.get_potential_energy()
    except Exception as e:
        print(f"  FAILED: {basename}: {e}")
        continue

    comp_dict = {}
    for s in atoms.get_chemical_symbols():
        comp_dict[s] = comp_dict.get(s, 0) + 1

    entry = ComputedEntry(Composition(comp_dict), energy)
    entries.append(entry)

    n = len(atoms)
    print(f"  {basename:>30s}  {entry.composition.reduced_formula:>12s}  "
          f"E = {energy/n:.4f} eV/atom")

# --- Build hull ---
pd_mace = PhaseDiagram(entries)

print(f"\nConvex hull built with {len(entries)} entries.")
print(f"Stable phases: {len(pd_mace.stable_entries)}")

for entry in sorted(pd_mace.stable_entries,
                    key=lambda e: pd_mace.get_form_energy_per_atom(e)):
    ef = pd_mace.get_form_energy_per_atom(entry)
    print(f"  {entry.composition.reduced_formula:>15s}  "
          f"Ef = {ef:+.4f} eV/atom")

# --- Evaluate unstable entries ---
print(f"\nUnstable phases:")
for entry in entries:
    e_hull = pd_mace.get_e_above_hull(entry)
    if e_hull > 1e-6:
        decomp, _ = pd_mace.get_decomp_and_e_above_hull(entry)
        products = " + ".join(
            f"{v:.2f} {k.composition.reduced_formula}"
            for k, v in decomp.items()
        )
        print(f"  {entry.composition.reduced_formula:>15s}  "
              f"E_hull = {e_hull:.4f} eV/atom  -> {products}")

# --- Plot ---
plotter = PDPlotter(pd_mace, show_unstable=0.05)
fig = plotter.get_plot()
plt.title("MACE-Only Convex Hull")
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "mace_hull.png"),
            dpi=150, bbox_inches="tight")
plt.close()
print(f"\nSaved {OUTPUT_DIR}/mace_hull.png")
```

### 5. Batch stability screening of many candidates

```python
#!/usr/bin/env python3
"""
Batch screening: evaluate thermodynamic stability for a large number of
candidate structures against the Materials Project convex hull.

Outputs a ranked table with e_above_hull, formation energy, and
decomposition products for each candidate.
"""

import os
import glob
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from mp_api.client import MPRester
from pymatgen.core import Composition
from pymatgen.analysis.phase_diagram import PhaseDiagram
from pymatgen.entries.computed_entries import ComputedEntry
from pymatgen.io.ase import AseAtomsAdaptor

from ase.io import read as ase_read
from ase.optimize import LBFGS
from ase.constraints import ExpCellFilter
from mace.calculators import mace_mp

MP_API_KEY = os.environ.get("MP_API_KEY")
CANDIDATES_DIR = "candidates"
MACE_MODEL = "medium"
FMAX = 0.02

adaptor = AseAtomsAdaptor()
calc = mace_mp(model=MACE_MODEL, default_dtype="float64")

# --- Compute MACE energies ---
candidate_entries = []
candidate_files = sorted(
    glob.glob(os.path.join(CANDIDATES_DIR, "*.cif"))
    + glob.glob(os.path.join(CANDIDATES_DIR, "*.vasp"))
)

for filepath in candidate_files:
    basename = os.path.basename(filepath)
    atoms = ase_read(filepath)
    atoms.calc = calc

    try:
        ecf = ExpCellFilter(atoms, hydrostatic_strain=False)
        opt = LBFGS(ecf, logfile=None)
        opt.run(fmax=FMAX, steps=300)
        energy = atoms.get_potential_energy()
    except Exception as e:
        print(f"  SKIP {basename}: {e}")
        continue

    comp_dict = {}
    for s in atoms.get_chemical_symbols():
        comp_dict[s] = comp_dict.get(s, 0) + 1

    entry = ComputedEntry(Composition(comp_dict), energy)
    candidate_entries.append({"filename": basename, "entry": entry})
    print(f"  {basename}: {entry.composition.reduced_formula}, "
          f"E = {energy/len(atoms):.4f} eV/atom")

# --- Determine full chemical system ---
all_elements = set()
for c in candidate_entries:
    all_elements.update(c["entry"].composition.get_el_amt_dict().keys())
system = sorted(all_elements)

# --- Fetch MP references ---
with MPRester(MP_API_KEY) as mpr:
    mp_entries = mpr.get_entries_in_chemsys(system)

# --- Build hull and evaluate ---
all_entries = mp_entries + [c["entry"] for c in candidate_entries]
pd_hull = PhaseDiagram(all_entries)

rows = []
for c in candidate_entries:
    entry = c["entry"]
    e_hull = pd_hull.get_e_above_hull(entry)
    ef = pd_hull.get_form_energy_per_atom(entry)
    stable = e_hull < 1e-6

    decomp_str = ""
    if not stable:
        decomp, _ = pd_hull.get_decomp_and_e_above_hull(entry)
        decomp_str = " + ".join(
            f"{v:.2f} {k.composition.reduced_formula}"
            for k, v in decomp.items()
        )

    rows.append({
        "filename": c["filename"],
        "formula": entry.composition.reduced_formula,
        "e_above_hull": e_hull,
        "formation_energy": ef,
        "is_stable": stable,
        "decomposition": decomp_str,
    })

df = pd.DataFrame(rows)
df.sort_values("e_above_hull", inplace=True)
df.to_csv("batch_stability.csv", index=False)

print(f"\n{'Formula':>15s} {'E_hull (eV/at)':>15s} {'Ef (eV/at)':>12s} "
      f"{'Stable':>8s}")
print("-" * 55)
for _, row in df.iterrows():
    print(f"{row['formula']:>15s} {row['e_above_hull']:>15.4f} "
          f"{row['formation_energy']:>12.4f} "
          f"{'YES' if row['is_stable'] else 'no':>8s}")

print(f"\nSaved batch_stability.csv")
```

## Key Parameters

| Parameter | Description | Notes |
|---|---|---|
| `SYSTEM_ELEMENTS` | List of elements defining the chemical system | Must include all elements present in candidate structures |
| `ComputedEntry(composition, energy)` | Composition + total energy in eV | Energy is for the full formula unit, not per atom |
| `get_e_above_hull(entry)` | Energy above convex hull (eV/atom) | 0 = stable; > 0 = metastable |
| `get_form_energy_per_atom(entry)` | Formation energy relative to elemental references | Negative = exothermic formation |
| `get_decomp_and_e_above_hull(entry)` | Decomposition products + energy above hull | Products are the stable phases the material would decompose into |
| `show_unstable` (PDPlotter) | Max e_above_hull to show in plot | 0 = hull only; 0.05 = within 50 meV of hull |
| `GrandPotentialPhaseDiagram` | Phase diagram at fixed chemical potential | Used for open systems (e.g., electrode in contact with Li reservoir) |
| MACE `FMAX` | Force convergence for relaxation | 0.01 eV/Ang for accurate energies; 0.05 for quick screening |

## Interpreting Results

- **e_above_hull = 0**: The phase is thermodynamically stable at 0 K and 0 GPa. It lies exactly on the convex hull.
- **0 < e_above_hull < 0.025 eV/atom**: Metastable but potentially synthesizable. Many known metastable phases (e.g., diamond, anatase TiO2) fall in this range. Kinetic barriers may prevent decomposition at room temperature.
- **e_above_hull > 0.025 eV/atom**: Increasingly unlikely to be synthesizable under equilibrium conditions. May exist at high temperature, high pressure, or as epitaxially stabilized thin films.
- **e_above_hull > 0.1 eV/atom**: Very unlikely to be experimentally accessible. Reconsider the candidate.
- **Formation energy < 0**: The compound is stable relative to its constituent elements. This is necessary but not sufficient for thermodynamic stability (other competing phases may be lower in energy).
- **Decomposition products**: The set of stable phases that the unstable compound would decompose into. The fractions indicate molar ratios at the decomposition point on the hull.
- **Grand potential diagram**: Shows which phases are stable as a function of an element's chemical potential. Useful for battery electrodes (Li chemical potential maps to voltage) and gas-phase reactions (O chemical potential maps to oxygen partial pressure).

**Important caveats**:
- Mixing MACE energies with MP DFT energies introduces systematic offsets (~10-30 meV/atom). The relative ranking of candidates is more reliable than absolute e_above_hull values.
- MP uses PBE+U for transition metal oxides. MACE-MP-0 was trained on a mix of PBE and PBE+U data. For quantitative stability analysis, compute all entries with the same method.
- The convex hull is a 0 K, 0 GPa construction. Finite-temperature effects (vibrational entropy, configurational entropy) can stabilize phases that appear metastable at 0 K.

## Common Issues

| Problem | Cause | Solution |
|---|---|---|
| `ValueError: No terminal entries` | Missing elemental reference for one or more elements | Ensure all elements have at least one single-element entry in the hull |
| Candidate appears stable with MACE but not with DFT | Systematic energy offset between MACE and DFT | Recompute with consistent method (all MACE or all DFT) |
| Phase diagram too crowded | Many entries plotted | Use `PDPlotter(pd, show_unstable=0)` to show only stable phases |
| Ternary plot has wrong axis labels | pymatgen Gibbs triangle convention | The three corners correspond to the three elements in alphabetical order |
| `get_entries_in_chemsys` returns empty | Invalid system specification or API key | Check that elements are valid symbols; verify `MP_API_KEY` |
| Grand potential diagram crashes | Chemical potential outside valid range | Use `pd_canonical.get_transition_chempots(Element("Li"))` to find valid range |
| ComputedEntry energy units wrong | Used energy per atom instead of total energy | `ComputedEntry` expects total energy in eV for the full composition |
| Negative formation energy but positive e_above_hull | Competing phases are even more stable | This is physically correct -- the compound forms from elements but decomposes to other compounds |
