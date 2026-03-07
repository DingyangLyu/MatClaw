# Intercalation Voltage Calculation

## When to Use

- Calculate the average intercalation voltage of a cathode or anode material
- Construct step voltage profiles as a function of Li (or Na, Mg, ...) content
- Determine which Li concentrations are thermodynamically stable (convex hull)
- Compare voltage and capacity of candidate electrode materials
- Screen new cathode compositions before synthesis

## Method Selection

```
Quick screening of voltage trend?
  --> Method A: ASE + MACE (minutes, no pseudopotentials needed)

Publication-quality voltage for transition metal oxide cathode?
  --> Method B: QE DFT+U (hours, includes Hubbard U for TM d-electrons)

System outside MACE training data (novel compositions, exotic anions)?
  --> Method B: QE DFT+U required

Large enumeration of Li orderings (>50 configs)?
  --> Method A for energies, then refine hull vertices with Method B
```

## Prerequisites

- pymatgen (structure building, Li ordering enumeration, voltage analysis)
- ASE + mace-torch (Method A)
- Quantum ESPRESSO pw.x (Method B)
- SSSP pseudopotentials (Method B)
- matplotlib for plotting voltage profiles
- Optional: `mp-api` for fetching reference structures from Materials Project

## Theory

### Intercalation Voltage

The open-circuit voltage for the reaction:

    Li_x2 MO_z + (x2 - x1) Li  -->  Li_x1 MO_z

is given by the Nernst equation:

    V = -DeltaG / ((x2 - x1) * F)

where DeltaG is the Gibbs free energy change and F is Faraday's constant.

At 0 K and neglecting PV and entropy contributions (valid for solids):

    V = -(E[Li_x2 MO_z] - E[Li_x1 MO_z] - (x2 - x1) * E[Li_metal]) / ((x2 - x1) * e)

where E are total energies (per formula unit) and e is the elementary charge (so V comes out in volts when E is in eV).

### Average vs Step Voltage

- **Average voltage**: Uses fully lithiated (x=1) and fully delithiated (x=0) endpoints only. Quick but misses plateaus.
- **Step voltage**: Computed between consecutive stable compositions on the Li_x convex hull. Gives the staircase voltage profile seen experimentally.

### Convex Hull Construction

Plot formation energy per atom of Li_xMO_z vs x. Points on the lower convex hull are thermodynamically stable. Voltage steps correspond to two-phase regions between adjacent hull vertices.

## Detailed Steps

### Method A: ASE + MACE

#### Complete Workflow: Voltage Profile for LiCoO2

```python
#!/usr/bin/env python3
"""
Intercalation voltage profile for Li_xCoO2 (0 <= x <= 1) using ASE + MACE.

This script:
1. Builds the LiCoO2 layered structure (R-3m)
2. Creates a supercell for Li ordering enumeration
3. Enumerates symmetry-distinct Li/vacancy orderings at each composition
4. Relaxes each configuration with MACE
5. Constructs the convex hull and voltage profile
6. Plots results

Inspired by the atomate2 ElectrodeInsertionMaker workflow pattern:
  relax host -> enumerate insertions -> relax each -> collect energies -> voltage analysis
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

from pymatgen.core import Structure, Lattice, Element, Species
from pymatgen.transformations.standard_transformations import (
    OrderDisorderedStructureTransformation,
)
from pymatgen.analysis.phase_diagram import PhaseDiagram, PDPlotter
from pymatgen.entries.computed_entries import ComputedEntry
from pymatgen.io.ase import AseAtomsAdaptor

from ase.optimize import LBFGS
from ase.constraints import ExpCellFilter
from mace.calculators import mace_mp

import json
from pathlib import Path

# ============================================================
# 1. CONFIGURATION
# ============================================================
MACE_MODEL = "medium"       # "small", "medium", "large"
DEVICE = "cpu"              # "cpu" or "cuda"
FMAX_RELAX = 0.02           # eV/Angstrom for relaxation convergence
MAX_STEPS = 500             # max optimizer steps
SUPERCELL = [1, 1, 2]       # supercell dimensions (from R-3m hexagonal cell)
                            # [1,1,2] gives ~24 atoms -- good balance
N_ORDERINGS = 4             # max symmetry-distinct orderings per composition
OUTPUT_DIR = Path("voltage_profile")
OUTPUT_DIR.mkdir(exist_ok=True)

# ============================================================
# 2. SET UP MACE CALCULATOR
# ============================================================
print("Loading MACE calculator...")
calc = mace_mp(model=MACE_MODEL, dispersion=False, default_dtype="float64",
               device=DEVICE)
adaptor = AseAtomsAdaptor()

# ============================================================
# 3. BUILD LiCoO2 STRUCTURE (R-3m, layered oxide)
# ============================================================
# Hexagonal setting of R-3m (space group 166)
# Experimental: a = 2.816 A, c = 14.08 A
lco = Structure.from_spacegroup(
    "R-3m",
    Lattice.hexagonal(a=2.816, c=14.08),
    species=["Li", "Co", "O"],
    coords=[
        [0.0, 0.0, 0.5],    # Li at 3b Wyckoff
        [0.0, 0.0, 0.0],    # Co at 3a Wyckoff
        [0.0, 0.0, 0.2393], # O at 6c Wyckoff
    ],
)
print(f"Primitive LiCoO2: {lco.formula}, {len(lco)} atoms")

# Build supercell
supercell = lco.copy()
supercell.make_supercell(SUPERCELL)
n_total_atoms = len(supercell)
n_li_total = sum(1 for s in supercell.species if s == Element("Li"))
print(f"Supercell: {supercell.formula}, {n_total_atoms} atoms, {n_li_total} Li sites")

# ============================================================
# 4. RELAX LI METAL REFERENCE
# ============================================================
print("\nRelaxing Li metal (BCC)...")
li_metal = Structure.from_spacegroup(
    "Im-3m",
    Lattice.cubic(3.51),
    species=["Li"],
    coords=[[0.0, 0.0, 0.0]],
)
li_atoms = adaptor.get_atoms(li_metal)
li_atoms.calc = calc
ecf = ExpCellFilter(li_atoms, scalar_pressure=0.0)
opt = LBFGS(ecf, logfile=str(OUTPUT_DIR / "li_metal_relax.log"))
opt.run(fmax=FMAX_RELAX, steps=MAX_STEPS)
e_li_metal_per_atom = li_atoms.get_potential_energy() / len(li_atoms)
print(f"Li metal energy: {e_li_metal_per_atom:.6f} eV/atom")

# ============================================================
# 5. HELPER: Relax a structure with MACE
# ============================================================
def relax_structure(structure, label=""):
    """Relax structure with MACE. Returns (relaxed_structure, energy_per_atom)."""
    atoms = adaptor.get_atoms(structure)
    atoms.calc = calc
    ecf = ExpCellFilter(atoms, scalar_pressure=0.0)
    logfile = str(OUTPUT_DIR / f"relax_{label}.log") if label else None
    opt = LBFGS(ecf, logfile=logfile)
    try:
        opt.run(fmax=FMAX_RELAX, steps=MAX_STEPS)
    except Exception as e:
        print(f"  WARNING: relaxation failed for {label}: {e}")
        return None, None
    energy = atoms.get_potential_energy()
    relaxed = adaptor.get_structure(atoms)
    return relaxed, energy

# ============================================================
# 6. GENERATE COMPOSITIONS AND ENUMERATE ORDERINGS
# ============================================================
# We systematically remove Li atoms from the fully lithiated supercell.
# At each composition x in Li_xCoO2, we enumerate symmetry-distinct
# orderings of Li/vacancy on the Li sublattice.

compositions_to_test = np.linspace(0.0, 1.0, n_li_total + 1)
# e.g., for 6 Li sites: x = [0, 1/6, 2/6, 3/6, 4/6, 5/6, 1]

all_results = []  # list of (x, energy_per_fu, formula)

print(f"\nEnumerating Li orderings for {len(compositions_to_test)} compositions...")

for x_target in compositions_to_test:
    n_li_keep = round(x_target * n_li_total)
    n_li_remove = n_li_total - n_li_keep
    x_actual = n_li_keep / n_li_total

    print(f"\n--- x = {x_actual:.4f} ({n_li_keep}/{n_li_total} Li) ---")

    if n_li_remove == 0:
        # Fully lithiated -- just relax the supercell
        relaxed, energy = relax_structure(supercell.copy(), f"x{x_actual:.2f}_full")
        if energy is not None:
            # energy per formula unit of Li_xCoO2
            n_fu = sum(1 for s in supercell.species if s == Element("Co"))
            e_per_fu = energy / n_fu
            all_results.append((x_actual, e_per_fu, relaxed.formula))
            print(f"  E = {energy:.4f} eV total, {e_per_fu:.4f} eV/f.u.")
        continue

    if n_li_keep == 0:
        # Fully delithiated -- remove all Li
        delith = supercell.copy()
        li_indices = [i for i, s in enumerate(delith.species) if s == Element("Li")]
        delith.remove_sites(li_indices)
        relaxed, energy = relax_structure(delith, f"x{x_actual:.2f}_empty")
        if energy is not None:
            n_fu = sum(1 for s in delith.species if s == Element("Co"))
            e_per_fu = energy / n_fu
            all_results.append((x_actual, e_per_fu, relaxed.formula))
            print(f"  E = {energy:.4f} eV total, {e_per_fu:.4f} eV/f.u.")
        continue

    # Intermediate composition: create a disordered structure and enumerate orderings
    # Replace Li sites with partial occupancy Li/vacancy
    disordered = supercell.copy()
    li_indices = [i for i, s in enumerate(disordered.species) if s == Element("Li")]

    # Create a disordered structure with fractional Li occupancy
    from pymatgen.core import PeriodicSite
    species_map = {Element("Li"): {Element("Li"): x_actual, "Vacancy": 1 - x_actual}}
    disordered_struct = disordered.copy()
    for idx in li_indices:
        disordered_struct.replace(
            idx,
            species={Element("Li"): x_actual, "Vacancy": 1 - x_actual},
        )

    # Enumerate symmetry-distinct orderings
    try:
        trans = OrderDisorderedStructureTransformation(
            algo=2,                        # enumlib algorithm
            no_oxi_states=True,
            symmetrized_structures=False,
        )
        ordered_structs = trans.apply_transformation(
            disordered_struct, return_ranked_list=N_ORDERINGS
        )
    except Exception as e:
        print(f"  Enumeration failed: {e}")
        print("  Falling back to manual Li removal...")
        # Fallback: just remove Li atoms by index (first N)
        manual_struct = supercell.copy()
        li_idx = [i for i, s in enumerate(manual_struct.species) if s == Element("Li")]
        remove_idx = li_idx[:n_li_remove]
        manual_struct.remove_sites(remove_idx)
        ordered_structs = [{"structure": manual_struct}]

    print(f"  Generated {len(ordered_structs)} orderings")

    # Relax each ordering and keep the lowest energy
    best_energy = None
    best_struct = None
    for i, entry in enumerate(ordered_structs):
        struct = entry["structure"] if isinstance(entry, dict) else entry
        label = f"x{x_actual:.2f}_ord{i}"
        relaxed, energy = relax_structure(struct, label)
        if energy is not None:
            n_fu = sum(1 for s in struct.species if s == Element("Co"))
            e_per_fu = energy / n_fu
            print(f"  Ordering {i}: E = {e_per_fu:.4f} eV/f.u.")
            if best_energy is None or e_per_fu < best_energy:
                best_energy = e_per_fu
                best_struct = relaxed

    if best_energy is not None:
        all_results.append((x_actual, best_energy, best_struct.formula))

# ============================================================
# 7. CONSTRUCT CONVEX HULL AND COMPUTE VOLTAGES
# ============================================================
print("\n" + "=" * 60)
print("CONVEX HULL AND VOLTAGE PROFILE")
print("=" * 60)

# Sort by x
all_results.sort(key=lambda r: r[0])

# Print all energies
print("\nAll compositions (sorted by x):")
print(f"{'x':>8s} {'E (eV/f.u.)':>14s} {'Formula':>20s}")
for x, e, f in all_results:
    print(f"{x:8.4f} {e:14.6f} {f:>20s}")

# Formation energy relative to endpoints (x=0, x=1)
x_vals = np.array([r[0] for r in all_results])
e_vals = np.array([r[1] for r in all_results])

# Endpoints
e_x0 = e_vals[x_vals == 0.0][0] if any(x_vals == 0.0) else e_vals[0]
e_x1 = e_vals[x_vals == 1.0][0] if any(x_vals == 1.0) else e_vals[-1]

# Formation energy: E_f(x) = E(x) - [(1-x)*E(x=0) + x*(E(x=1))]
# But for voltage we also need the Li metal reference
# E_f(x) = E(Li_xCoO2) - E(CoO2) - x*E(Li_metal)
e_form = e_vals - e_x0 - x_vals * e_li_metal_per_atom
e_form_per_fu = e_form  # already per formula unit

print("\nFormation energies (vs CoO2 + x*Li_metal):")
for i, (x, ef) in enumerate(zip(x_vals, e_form_per_fu)):
    print(f"  x = {x:.4f}: E_f = {ef:.4f} eV/f.u.")

# Build convex hull in the x vs E_f space
# Find the lower convex hull
from scipy.spatial import ConvexHull

if len(x_vals) >= 3:
    points = np.column_stack([x_vals, e_form_per_fu])

    # Simple lower convex hull: walk from left to right keeping only
    # points that maintain a decreasing (or equal) slope
    def lower_convex_hull(x, y):
        """Return indices of points on the lower convex hull."""
        n = len(x)
        order = np.argsort(x)
        x_sorted = x[order]
        y_sorted = y[order]

        hull_idx = [0]  # always include leftmost
        for i in range(1, n):
            while len(hull_idx) >= 2:
                # Check if the last point is above the line from
                # hull_idx[-2] to current point i
                x1, y1 = x_sorted[hull_idx[-2]], y_sorted[hull_idx[-2]]
                x2, y2 = x_sorted[hull_idx[-1]], y_sorted[hull_idx[-1]]
                x3, y3 = x_sorted[i], y_sorted[i]
                # Cross product: if positive, the middle point is above the line
                cross = (x2 - x1) * (y3 - y1) - (y2 - y1) * (x3 - x1)
                if cross >= 0:
                    hull_idx.pop()
                else:
                    break
            hull_idx.append(i)
        return order[hull_idx]

    hull_indices = lower_convex_hull(x_vals, e_form_per_fu)
    hull_x = x_vals[hull_indices]
    hull_e = e_form_per_fu[hull_indices]

    print(f"\nConvex hull vertices: {len(hull_indices)} compositions")
    for hx, he in zip(hull_x, hull_e):
        print(f"  x = {hx:.4f}, E_f = {he:.4f} eV/f.u.")
else:
    hull_indices = np.arange(len(x_vals))
    hull_x = x_vals
    hull_e = e_form_per_fu

# ============================================================
# 8. COMPUTE STEP VOLTAGES FROM HULL VERTICES
# ============================================================
# Voltage between consecutive hull points:
# V_step = -(E(x2) - E(x1) - (x2-x1)*E_Li) / (x2 - x1)
# Since E_f already subtracts E(x=0) + x*E_Li:
# V_step = -(E_f(x2) - E_f(x1)) / (x2 - x1)  -- but wait, E_f definition matters

# Direct voltage from total energies per f.u.:
hull_e_total = e_vals[hull_indices]  # total energies per f.u. on the hull

print("\nStep voltages (between consecutive hull vertices):")
step_voltages = []
step_x_ranges = []
for i in range(len(hull_x) - 1):
    x1, x2 = hull_x[i], hull_x[i + 1]
    e1, e2 = hull_e_total[i], hull_e_total[i + 1]
    dx = x2 - x1
    if abs(dx) < 1e-10:
        continue
    # V = -(E(Li_x2 CoO2) - E(Li_x1 CoO2) - (x2-x1)*E_Li_metal) / (x2-x1)
    v_step = -(e2 - e1 - dx * e_li_metal_per_atom) / dx
    step_voltages.append(v_step)
    step_x_ranges.append((x1, x2))
    print(f"  x = [{x1:.4f}, {x2:.4f}]: V = {v_step:.3f} V")

# Average voltage (endpoints only)
if len(x_vals) >= 2:
    v_avg = -(e_x1 - e_x0 - 1.0 * e_li_metal_per_atom) / 1.0
    print(f"\nAverage voltage (x=0 to x=1): {v_avg:.3f} V")

# ============================================================
# 9. PLOT RESULTS
# ============================================================
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# (a) Formation energy convex hull
ax1 = axes[0]
ax1.scatter(x_vals, e_form_per_fu, color="blue", zorder=5, label="All configs")
ax1.plot(hull_x, hull_e, "r-o", linewidth=2, markersize=8,
         zorder=6, label="Convex hull")
ax1.set_xlabel("x in Li$_x$CoO$_2$", fontsize=13)
ax1.set_ylabel("Formation energy (eV/f.u.)", fontsize=13)
ax1.set_title("Li$_x$CoO$_2$ Convex Hull", fontsize=14)
ax1.legend(fontsize=11)
ax1.set_xlim(-0.05, 1.05)
ax1.axhline(0, color="gray", linestyle="--", alpha=0.5)
ax1.grid(True, alpha=0.3)

# (b) Voltage profile
ax2 = axes[1]
for v, (x1, x2) in zip(step_voltages, step_x_ranges):
    ax2.plot([x1, x2], [v, v], "b-", linewidth=2.5)
    # vertical connectors
for i in range(len(step_voltages) - 1):
    x_conn = step_x_ranges[i][1]
    ax2.plot([x_conn, x_conn], [step_voltages[i], step_voltages[i + 1]],
             "b-", linewidth=1, alpha=0.5)
if len(x_vals) >= 2:
    ax2.axhline(v_avg, color="red", linestyle="--", linewidth=1.5,
                label=f"Average V = {v_avg:.2f} V")
ax2.set_xlabel("x in Li$_x$CoO$_2$", fontsize=13)
ax2.set_ylabel("Voltage vs Li/Li$^+$ (V)", fontsize=13)
ax2.set_title("Voltage Profile", fontsize=14)
ax2.legend(fontsize=11)
ax2.set_xlim(-0.05, 1.05)
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(str(OUTPUT_DIR / "voltage_profile.png"), dpi=150, bbox_inches="tight")
plt.close()
print(f"\nSaved: {OUTPUT_DIR / 'voltage_profile.png'}")

# ============================================================
# 10. SAVE NUMERICAL RESULTS
# ============================================================
results_dict = {
    "system": "Li_xCoO2",
    "method": f"MACE ({MACE_MODEL})",
    "supercell": SUPERCELL,
    "e_li_metal_per_atom_eV": float(e_li_metal_per_atom),
    "compositions": [
        {"x": float(x), "energy_per_fu_eV": float(e), "formula": f}
        for x, e, f in all_results
    ],
    "hull_vertices": [
        {"x": float(hx), "formation_energy_eV": float(he)}
        for hx, he in zip(hull_x, hull_e)
    ],
    "step_voltages": [
        {"x_start": float(x1), "x_end": float(x2), "voltage_V": float(v)}
        for v, (x1, x2) in zip(step_voltages, step_x_ranges)
    ],
    "average_voltage_V": float(v_avg) if len(x_vals) >= 2 else None,
}
with open(OUTPUT_DIR / "voltage_results.json", "w") as f:
    json.dump(results_dict, f, indent=2)
print(f"Saved: {OUTPUT_DIR / 'voltage_results.json'}")
```

### Method B: QE DFT+U

For publication-quality results on transition metal oxide cathodes, use Quantum ESPRESSO with DFT+U. Below is a complete workflow for LiCoO2 / CoO2.

#### Step 1: Generate Structures and QE Input Files

```python
#!/usr/bin/env python3
"""
Generate QE input files for LiCoO2 voltage calculation with DFT+U.
Creates inputs for:
  1. Li metal (BCC) -- reference
  2. LiCoO2 (fully lithiated, R-3m)
  3. CoO2 (fully delithiated)
  4. Intermediate Li_0.5 CoO2
"""

import os
import subprocess
from pathlib import Path
from pymatgen.core import Structure, Lattice, Element

# ============================================================
# Download pseudopotentials (SSSP Efficiency 1.3)
# ============================================================
PSEUDO_DIR = Path("./pseudo")
PSEUDO_DIR.mkdir(exist_ok=True)

sssp_base = "https://pseudopotentials.quantum-espresso.org/upf_files"
pseudos = {
    "Li": "Li.pbe-s-kjpaw_psl.1.0.0.UPF",
    "Co": "Co.pbe-spn-kjpaw_psl.0.3.1.UPF",
    "O":  "o_pbe_v1.2.uspp.F.UPF",
}

for elem, fname in pseudos.items():
    fpath = PSEUDO_DIR / fname
    if not fpath.exists():
        url = f"{sssp_base}/{fname}"
        print(f"Downloading {fname}...")
        subprocess.run(["wget", "-q", "-O", str(fpath), url], check=False)
        # Fallback: try alternate naming
        if not fpath.exists() or fpath.stat().st_size < 1000:
            print(f"  Primary URL failed. Trying SSSP library...")
            subprocess.run([
                "wget", "-q", "-O", str(fpath),
                f"https://raw.githubusercontent.com/PseudoDojo/ONCVPSP-PBE-SR/"
                f"master/{fname}"
            ], check=False)

# ============================================================
# Common QE parameters
# ============================================================
ECUTWFC = 60.0   # Ry, wavefunction cutoff
ECUTRHO = 600.0  # Ry, charge density cutoff (10x for USPP/PAW)
K_GRID_BULK = "6 6 6"
K_GRID_LCO = "6 6 2"    # layered: fewer k-points along c
CONV_THR = 1.0e-8
FORC_CONV = 1.0e-4       # Ry/Bohr for relaxation

# Hubbard U for Co-3d (literature value for LiCoO2 with PBE)
# Wang, Maxisch, Ceder, PRB 73, 195107 (2006): U_Co = 3.32 eV
U_CO = 3.32

WORK = Path("qe_voltage")
WORK.mkdir(exist_ok=True)

# ============================================================
# 1. Li metal (BCC, Im-3m)
# ============================================================
li_dir = WORK / "li_metal"
li_dir.mkdir(exist_ok=True)

li_input = f"""&CONTROL
    calculation  = 'vc-relax'
    prefix       = 'li_metal'
    outdir       = './tmp'
    pseudo_dir   = '../pseudo'
    tprnfor      = .true.
    tstress      = .true.
    etot_conv_thr = 1.0d-6
    forc_conv_thr = {FORC_CONV}
/
&SYSTEM
    ibrav        = 3
    celldm(1)    = 6.63   ! a = 3.51 Angstrom in Bohr
    nat          = 1
    ntyp         = 1
    ecutwfc      = {ECUTWFC}
    ecutrho      = {ECUTRHO}
    occupations  = 'smearing'
    smearing     = 'mv'
    degauss      = 0.02
/
&ELECTRONS
    conv_thr     = {CONV_THR}
    mixing_beta  = 0.3
/
&IONS
    ion_dynamics = 'bfgs'
/
&CELL
    cell_dynamics = 'bfgs'
    press         = 0.0
/
ATOMIC_SPECIES
    Li  6.941  {pseudos["Li"]}
ATOMIC_POSITIONS crystal
    Li  0.0  0.0  0.0
K_POINTS automatic
    {K_GRID_BULK}  1 1 1
"""
with open(li_dir / "pw.in", "w") as f:
    f.write(li_input)
print(f"Written: {li_dir / 'pw.in'}")

# ============================================================
# 2. LiCoO2 (R-3m hexagonal, fully lithiated)
# ============================================================
lco_dir = WORK / "LiCoO2"
lco_dir.mkdir(exist_ok=True)

# Hexagonal cell: a = 2.816 A, c = 14.08 A
# Atomic positions in crystal coordinates (hexagonal axes)
lco_input = f"""&CONTROL
    calculation  = 'vc-relax'
    prefix       = 'licoo2'
    outdir       = './tmp'
    pseudo_dir   = '../pseudo'
    tprnfor      = .true.
    tstress      = .true.
    etot_conv_thr = 1.0d-6
    forc_conv_thr = {FORC_CONV}
/
&SYSTEM
    ibrav        = 0
    nat          = 12
    ntyp         = 3
    ecutwfc      = {ECUTWFC}
    ecutrho      = {ECUTRHO}
    occupations  = 'smearing'
    smearing     = 'mv'
    degauss      = 0.02
    nspin        = 2
    starting_magnetization(2) = 0.1
    ! DFT+U (QE 7.x ortho-atomic formulation)
    lda_plus_u   = .true.
    lda_plus_u_kind = 0
    Hubbard_U(2) = {U_CO}
/
&ELECTRONS
    conv_thr     = {CONV_THR}
    mixing_beta  = 0.2
/
&IONS
    ion_dynamics = 'bfgs'
/
&CELL
    cell_dynamics = 'bfgs'
    press         = 0.0
/
ATOMIC_SPECIES
    Li   6.941   {pseudos["Li"]}
    Co  58.933   {pseudos["Co"]}
    O   15.999   {pseudos["O"]}
CELL_PARAMETERS angstrom
    2.816000    0.000000    0.000000
   -1.408000    2.438614    0.000000
    0.000000    0.000000   14.080000
ATOMIC_POSITIONS crystal
    Co   0.000000   0.000000   0.000000
    Co   0.333333   0.666667   0.666667
    Co   0.666667   0.333333   0.333333
    Li   0.000000   0.000000   0.500000
    Li   0.333333   0.666667   0.166667
    Li   0.666667   0.333333   0.833333
    O    0.000000   0.000000   0.239300
    O    0.333333   0.666667   0.905967
    O    0.666667   0.333333   0.572633
    O    0.000000   0.000000   0.760700
    O    0.333333   0.666667   0.427367
    O    0.666667   0.333333   0.094033
K_POINTS automatic
    {K_GRID_LCO}  0 0 0
"""
with open(lco_dir / "pw.in", "w") as f:
    f.write(lco_input)
print(f"Written: {lco_dir / 'pw.in'}")

# ============================================================
# 3. CoO2 (fully delithiated, same R-3m framework without Li)
# ============================================================
coo2_dir = WORK / "CoO2"
coo2_dir.mkdir(exist_ok=True)

coo2_input = f"""&CONTROL
    calculation  = 'vc-relax'
    prefix       = 'coo2'
    outdir       = './tmp'
    pseudo_dir   = '../pseudo'
    tprnfor      = .true.
    tstress      = .true.
    etot_conv_thr = 1.0d-6
    forc_conv_thr = {FORC_CONV}
/
&SYSTEM
    ibrav        = 0
    nat          = 9
    ntyp         = 2
    ecutwfc      = {ECUTWFC}
    ecutrho      = {ECUTRHO}
    occupations  = 'smearing'
    smearing     = 'mv'
    degauss      = 0.02
    nspin        = 2
    starting_magnetization(1) = 0.5
    ! DFT+U for Co
    lda_plus_u   = .true.
    lda_plus_u_kind = 0
    Hubbard_U(1) = {U_CO}
/
&ELECTRONS
    conv_thr     = {CONV_THR}
    mixing_beta  = 0.2
/
&IONS
    ion_dynamics = 'bfgs'
/
&CELL
    cell_dynamics = 'bfgs'
    press         = 0.0
/
ATOMIC_SPECIES
    Co  58.933   {pseudos["Co"]}
    O   15.999   {pseudos["O"]}
CELL_PARAMETERS angstrom
    2.816000    0.000000    0.000000
   -1.408000    2.438614    0.000000
    0.000000    0.000000   14.080000
ATOMIC_POSITIONS crystal
    Co   0.000000   0.000000   0.000000
    Co   0.333333   0.666667   0.666667
    Co   0.666667   0.333333   0.333333
    O    0.000000   0.000000   0.239300
    O    0.333333   0.666667   0.905967
    O    0.666667   0.333333   0.572633
    O    0.000000   0.000000   0.760700
    O    0.333333   0.666667   0.427367
    O    0.666667   0.333333   0.094033
K_POINTS automatic
    {K_GRID_LCO}  0 0 0
"""
with open(coo2_dir / "pw.in", "w") as f:
    f.write(coo2_input)
print(f"Written: {coo2_dir / 'pw.in'}")

# ============================================================
# 4. Li_0.5 CoO2 (half lithiated, 1 Li removed from 3)
# ============================================================
half_dir = WORK / "Li0.5CoO2"
half_dir.mkdir(exist_ok=True)

# Remove the Li at (0,0,0.5) -- keep the other two Li
half_input = f"""&CONTROL
    calculation  = 'vc-relax'
    prefix       = 'li05coo2'
    outdir       = './tmp'
    pseudo_dir   = '../pseudo'
    tprnfor      = .true.
    tstress      = .true.
    etot_conv_thr = 1.0d-6
    forc_conv_thr = {FORC_CONV}
/
&SYSTEM
    ibrav        = 0
    nat          = 11
    ntyp         = 3
    ecutwfc      = {ECUTWFC}
    ecutrho      = {ECUTRHO}
    occupations  = 'smearing'
    smearing     = 'mv'
    degauss      = 0.02
    nspin        = 2
    starting_magnetization(2) = 0.3
    ! DFT+U for Co
    lda_plus_u   = .true.
    lda_plus_u_kind = 0
    Hubbard_U(2) = {U_CO}
/
&ELECTRONS
    conv_thr     = {CONV_THR}
    mixing_beta  = 0.2
/
&IONS
    ion_dynamics = 'bfgs'
/
&CELL
    cell_dynamics = 'bfgs'
    press         = 0.0
/
ATOMIC_SPECIES
    Li   6.941   {pseudos["Li"]}
    Co  58.933   {pseudos["Co"]}
    O   15.999   {pseudos["O"]}
CELL_PARAMETERS angstrom
    2.816000    0.000000    0.000000
   -1.408000    2.438614    0.000000
    0.000000    0.000000   14.080000
ATOMIC_POSITIONS crystal
    Co   0.000000   0.000000   0.000000
    Co   0.333333   0.666667   0.666667
    Co   0.666667   0.333333   0.333333
    Li   0.333333   0.666667   0.166667
    Li   0.666667   0.333333   0.833333
    O    0.000000   0.000000   0.239300
    O    0.333333   0.666667   0.905967
    O    0.666667   0.333333   0.572633
    O    0.000000   0.000000   0.760700
    O    0.333333   0.666667   0.427367
    O    0.666667   0.333333   0.094033
K_POINTS automatic
    {K_GRID_LCO}  0 0 0
"""
with open(half_dir / "pw.in", "w") as f:
    f.write(half_input)
print(f"Written: {half_dir / 'pw.in'}")

print("\nAll QE input files generated.")
print("Run each with: mpirun -np N pw.x < pw.in > pw.out")
```

#### Step 2: Run QE Calculations

```bash
#!/bin/bash
# run_all_voltage.sh -- Run all QE vc-relax calculations for voltage profile
# Usage: bash run_all_voltage.sh [NPROC]

NPROC=${1:-4}
BASE="qe_voltage"

for dir in "$BASE"/li_metal "$BASE"/LiCoO2 "$BASE"/CoO2 "$BASE"/Li0.5CoO2; do
    echo "========================================="
    echo "Running: $dir"
    echo "========================================="
    cd "$dir"
    mkdir -p tmp
    mpirun -np $NPROC pw.x < pw.in > pw.out 2>&1
    # Check convergence
    if grep -q "JOB DONE" pw.out; then
        echo "  CONVERGED"
        E=$(grep '!' pw.out | tail -1 | awk '{print $5}')
        echo "  Total energy: $E Ry"
    else
        echo "  WARNING: not converged, check pw.out"
    fi
    cd - > /dev/null
done

echo ""
echo "All calculations complete. Extract energies with the Python script below."
```

#### Step 3: Extract Energies and Compute Voltage

```python
#!/usr/bin/env python3
"""
Extract total energies from QE outputs and compute intercalation voltage.
Run after all pw.x calculations complete.
"""

import re
import json
import numpy as np
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BASE = Path("qe_voltage")
RY_TO_EV = 13.605693123  # 1 Ry = 13.6057 eV

def extract_energy(pw_out_path):
    """Extract final total energy (Ry) from pw.x output."""
    energy = None
    with open(pw_out_path) as f:
        for line in f:
            if line.strip().startswith("!"):
                # "!    total energy              =     -XXX.XXXXXXXX Ry"
                match = re.search(r"=\s+([-\d.]+)\s+Ry", line)
                if match:
                    energy = float(match.group(1))
    return energy

# ============================================================
# Extract energies
# ============================================================
calcs = {
    "Li_metal":   {"dir": "li_metal",   "n_fu": 1, "n_li": 1, "n_co": 0},
    "LiCoO2":     {"dir": "LiCoO2",     "n_fu": 3, "n_li": 3, "n_co": 3},
    "Li0.5CoO2":  {"dir": "Li0.5CoO2",  "n_fu": 3, "n_li": 2, "n_co": 3},
    "CoO2":       {"dir": "CoO2",       "n_fu": 3, "n_li": 0, "n_co": 3},
}

energies = {}
print("Extracted energies:")
print(f"{'System':>15s} {'E_total (Ry)':>15s} {'E_total (eV)':>15s} {'E/f.u. (eV)':>15s}")
for name, info in calcs.items():
    pw_out = BASE / info["dir"] / "pw.out"
    if not pw_out.exists():
        print(f"  {name}: pw.out not found -- skipping")
        continue
    e_ry = extract_energy(pw_out)
    if e_ry is None:
        print(f"  {name}: energy not found in output -- check convergence")
        continue
    e_ev = e_ry * RY_TO_EV
    e_per_fu = e_ev / info["n_fu"]
    energies[name] = {
        "e_ry": e_ry,
        "e_ev": e_ev,
        "e_per_fu": e_per_fu,
        "x": info["n_li"] / info["n_co"] if info["n_co"] > 0 else None,
    }
    print(f"{name:>15s} {e_ry:15.8f} {e_ev:15.6f} {e_per_fu:15.6f}")

# ============================================================
# Compute voltages
# ============================================================
if all(k in energies for k in ["Li_metal", "LiCoO2", "CoO2"]):
    e_li = energies["Li_metal"]["e_per_fu"]  # per atom
    e_lco = energies["LiCoO2"]["e_per_fu"]   # per LiCoO2 f.u.
    e_coo2 = energies["CoO2"]["e_per_fu"]    # per CoO2 f.u.

    # Average voltage: V = -(E_LiCoO2 - E_CoO2 - E_Li) / (1 * e)
    v_avg = -(e_lco - e_coo2 - 1.0 * e_li) / 1.0
    print(f"\nAverage voltage (x=0 to x=1): {v_avg:.3f} V vs Li/Li+")

    # Step voltages (if Li0.5 is available)
    if "Li0.5CoO2" in energies:
        e_half = energies["Li0.5CoO2"]["e_per_fu"]
        x_half = energies["Li0.5CoO2"]["x"]

        # V(x=0 to 2/3)
        v_low = -(e_half - e_coo2 - x_half * e_li) / x_half
        # V(x=2/3 to 1)
        v_high = -(e_lco - e_half - (1.0 - x_half) * e_li) / (1.0 - x_half)

        print(f"Step voltage (x=0.00 to x={x_half:.2f}): {v_low:.3f} V")
        print(f"Step voltage (x={x_half:.2f} to x=1.00): {v_high:.3f} V")

        # Plot
        fig, ax = plt.subplots(figsize=(7, 5))
        ax.plot([0, x_half], [v_low, v_low], "b-", linewidth=2.5)
        ax.plot([x_half, 1.0], [v_high, v_high], "b-", linewidth=2.5)
        ax.plot([x_half, x_half], [v_low, v_high], "b-", linewidth=1, alpha=0.5)
        ax.axhline(v_avg, color="red", linestyle="--", linewidth=1.5,
                   label=f"Avg V = {v_avg:.2f} V")
        ax.set_xlabel("x in Li$_x$CoO$_2$", fontsize=13)
        ax.set_ylabel("Voltage vs Li/Li$^+$ (V)", fontsize=13)
        ax.set_title("LiCoO$_2$ Voltage Profile (QE DFT+U)", fontsize=14)
        ax.legend(fontsize=11)
        ax.set_xlim(-0.05, 1.05)
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(str(BASE / "voltage_profile_qe.png"), dpi=150)
        plt.close()
        print(f"\nSaved: {BASE / 'voltage_profile_qe.png'}")
else:
    print("\nNot all calculations available. Run QE first.")

# Save results
with open(BASE / "voltage_results_qe.json", "w") as f:
    json.dump(energies, f, indent=2, default=str)
print(f"Saved: {BASE / 'voltage_results_qe.json'}")
```

#### Alternative: QE 7.x Hubbard Card Syntax

QE 7.x introduced a new `HUBBARD` input card that replaces the old `lda_plus_u` namelist keywords. If your QE build supports it, use this instead:

```
! Remove these from &SYSTEM:
!   lda_plus_u = .true.
!   lda_plus_u_kind = 0
!   Hubbard_U(2) = 3.32

! Add after ATOMIC_POSITIONS:
HUBBARD ortho-atomic
U Co-3d 3.32
```

This syntax allows more control (e.g., inter-site V parameters) and is the recommended approach for QE >= 7.1.

## Key Parameters

| Parameter | Recommended Value | Notes |
|-----------|-------------------|-------|
| **Supercell size** | >= 2x2x1 for layered, >= 2x2x2 for spinel/olivine | Larger = more Li orderings but slower |
| **N_ORDERINGS** | 4--8 per composition | More orderings = better hull but slower |
| **Hubbard U (Co)** | 3.3 eV | Wang, Maxisch, Ceder, PRB 2006; use hp.x for self-consistent U |
| **Hubbard U (Fe)** | 4.3 eV | Common for LiFePO4 |
| **Hubbard U (Mn)** | 3.9 eV | Common for LiMn2O4 |
| **Hubbard U (Ni)** | 6.4 eV | Common for LiNiO2 |
| **ecutwfc (QE)** | 60--80 Ry | Depends on pseudopotential |
| **k-mesh (QE)** | 6x6x2 for layered hexagonal | Dense enough for energy convergence |
| **FMAX (MACE)** | 0.01--0.02 eV/A | Tighter for accurate energies |
| **MACE model** | "medium" or "large" | "large" is more accurate but slower |

## Interpreting Results

### Voltage Values
- **LiCoO2**: Expect 3.7--4.2 V. MACE may give ~3.5--4.5 V (less accurate for absolute values but good for trends). QE DFT+U typically gives 3.8--4.1 V, close to experiment.
- **LiFePO4**: Expect ~3.4 V (flat plateau). This is a two-phase system (LiFePO4 / FePO4).
- **LiMn2O4**: Expect 3.9--4.1 V for the 4V plateau.

### Convex Hull
- Points **on the hull** are thermodynamically stable compositions.
- Points **above the hull** are metastable and will phase-separate into adjacent hull compositions.
- The **energy above hull** (in meV/atom) indicates metastability: < 25 meV/atom is often considered synthesizable.

### Capacity
Theoretical gravimetric capacity:

    Q = (n * F) / (3.6 * M)    [mAh/g]

where n = number of Li per formula unit, F = 96485 C/mol, M = molar mass in g/mol, and 3.6 converts C to mAh.

## Common Issues

| Issue | Cause | Solution |
|-------|-------|----------|
| Negative voltage | Wrong reference or unstable structure | Check Li metal energy; verify structure didn't collapse during relaxation |
| Voltage too high (> 6V) | MACE inaccurate for this chemistry | Use QE DFT+U; check if material is in MACE training set |
| Enumeration too slow | Too many Li sites in supercell | Reduce supercell size or use `algo=0` (slower but handles large cells) |
| QE not converging | Spin/U initialization | Try `mixing_beta = 0.1`, increase `electron_maxstep`, or use `mixing_mode = 'local-TF'` |
| Structures distort heavily on delithiation | Physical (layer collapse in CoO2) or numerical | Check with experiment; for QE, fix cell shape if unphysical |
| OrderDisorderedStructureTransformation fails | enumlib not installed or too many sites | `pip install enum34`; reduce supercell; or use manual Li removal fallback |
| MACE gives different energy scale than QE | Expected -- different methods | Do not mix MACE and QE energies on the same hull; use one method consistently |
