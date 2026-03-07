# Heterostructure / Interface Construction and Calculation

## When to Use

- You need to build a heterostructure interface between two different materials (e.g., MoS2/graphene, oxide/metal, semiconductor/semiconductor).
- You want to find lattice-matched surface orientations with minimal strain.
- You need to compute interface adhesion (work of adhesion) or separation energy.
- You are preparing structures for band-alignment or charge-transfer studies at interfaces.

## Prerequisites

- Python packages: `pymatgen`, `ase`, `mace-torch`, `numpy`, `matplotlib`.
- For QE validation: Quantum ESPRESSO 7.5 (`pw.x`).
- Key pymatgen modules: `SubstrateAnalyzer`, `ZSLGenerator` (Zur and McGill Superlattice generator).

## Detailed Steps

### Background

Building a heterostructure requires:

1. **Lattice matching** -- find surface supercells of the film and substrate that match within a strain tolerance. The ZSL (Zur and McGill Super Lattice) algorithm systematically enumerates coincident superlattice pairs.
2. **Slab construction** -- cut slabs from each material along the matched orientation.
3. **Interface assembly** -- stack film on substrate with appropriate interlayer spacing and vacuum.
4. **Relaxation** -- optimize the structure (often fixing the substrate) with MACE or QE.
5. **Adhesion energy** -- W_ad = (E_film + E_substrate - E_interface) / A.

### Step 1 -- Complete Heterostructure Workflow

```python
#!/usr/bin/env python3
"""
Heterostructure construction and adhesion energy calculation.

Example system: MoS2 monolayer on graphene.
Uses pymatgen's ZSLGenerator for lattice matching, then ASE + MACE
for relaxation and adhesion energy.
"""

import numpy as np
import warnings
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from pymatgen.core import Structure, Lattice
from pymatgen.analysis.interfaces.zsl import ZSLGenerator
from pymatgen.analysis.interfaces.substrate_analyzer import SubstrateAnalyzer
from pymatgen.core.surface import SlabGenerator
from pymatgen.io.ase import AseAtomsAdaptor

from ase import Atoms
from ase.build import make_supercell
from ase.optimize import FIRE, LBFGS
from ase.constraints import FixAtoms
from ase.io import write as ase_write

warnings.filterwarnings("ignore")

# ------------------------------------------------------------------ #
#  Configuration
# ------------------------------------------------------------------ #
MACE_MODEL = "medium"
FMAX = 0.03           # eV/Ang
MAX_STRAIN = 0.05     # 5% max strain tolerance
MAX_AREA = 200        # Ang^2 max supercell area
VACUUM = 20.0         # Ang vacuum above heterostructure
INTERLAYER_GAP = 3.3  # Ang initial interlayer spacing

adaptor = AseAtomsAdaptor()

# ------------------------------------------------------------------ #
#  Step 1: Define film and substrate bulk structures
# ------------------------------------------------------------------ #

# --- Graphene (substrate) ---
a_graphene = 2.46
graphene_bulk = Structure(
    Lattice.hexagonal(a_graphene, 10.0),
    ["C", "C"],
    [[1/3, 2/3, 0.5], [2/3, 1/3, 0.5]],
)

# --- MoS2 monolayer (film) ---
a_mos2 = 3.16
c_mos2 = 12.3   # large c for 2D separation
mos2_bulk = Structure(
    Lattice.hexagonal(a_mos2, c_mos2),
    ["Mo", "S", "S"],
    [[0, 0, 0.5],
     [1/3, 2/3, 0.5 + 1.58/c_mos2],
     [1/3, 2/3, 0.5 - 1.58/c_mos2]],
)

print("Substrate (graphene):", graphene_bulk.composition)
print("Film (MoS2):", mos2_bulk.composition)


# ------------------------------------------------------------------ #
#  Step 2: Find lattice matches with ZSLGenerator
# ------------------------------------------------------------------ #
print("\n=== Lattice Matching ===")

zsl = ZSLGenerator(
    max_area_ratio_tol=0.09,
    max_area=MAX_AREA,
    max_length_tol=0.03,
    max_angle_tol=0.01,
)

# For 2D-on-2D, we match the in-plane lattice vectors directly
# SubstrateAnalyzer wraps ZSL with strain analysis
sub_analyzer = SubstrateAnalyzer(zslgen=zsl)

matches = list(sub_analyzer.calculate(
    film=mos2_bulk,
    substrate=graphene_bulk,
    film_millers=[(0, 0, 1)],      # basal plane
    substrate_millers=[(0, 0, 1)],  # basal plane
))

print(f"Found {len(matches)} lattice matches")

# Collect match data for analysis
match_data = []
for i, match in enumerate(matches):
    strain = match.von_mises_strain
    film_sl = match.film_sl_vectors
    sub_sl = match.substrate_sl_vectors

    # Compute supercell area from substrate superlattice vectors
    area = abs(np.cross(sub_sl[0], sub_sl[1]))

    match_data.append({
        "index": i,
        "strain": strain,
        "area": area,
        "film_sl": film_sl,
        "sub_sl": sub_sl,
        "film_transformation": match.film_transformation,
        "substrate_transformation": match.substrate_transformation,
        "match_obj": match,
    })

    if i < 5:
        print(f"  Match {i}: strain={strain:.4f}, area={area:.1f} Ang^2")

# Sort by strain
match_data.sort(key=lambda m: m["strain"])
print(f"\nBest match: strain={match_data[0]['strain']:.4f}, "
      f"area={match_data[0]['area']:.1f} Ang^2")


# ------------------------------------------------------------------ #
#  Step 3: Build heterostructure from the best match
# ------------------------------------------------------------------ #
print("\n=== Building Heterostructure ===")

best = match_data[0]


def build_heterostructure(film_struct, sub_struct, match,
                          film_layers=1, sub_layers=1,
                          gap=INTERLAYER_GAP, vacuum=VACUUM):
    """
    Build a heterostructure slab from matched film and substrate.

    This constructs supercells of film and substrate according to
    the ZSL transformation matrices, then stacks them.
    """
    # Generate slabs
    film_slab_gen = SlabGenerator(
        film_struct,
        miller_index=(0, 0, 1),
        min_slab_size=1.0,      # Angstrom -- minimal for monolayer
        min_vacuum_size=0.1,
        in_unit_planes=True,
        center_slab=True,
    )
    sub_slab_gen = SlabGenerator(
        sub_struct,
        miller_index=(0, 0, 1),
        min_slab_size=1.0,
        min_vacuum_size=0.1,
        in_unit_planes=True,
        center_slab=True,
    )

    film_slabs = film_slab_gen.get_slabs()
    sub_slabs = sub_slab_gen.get_slabs()

    if not film_slabs or not sub_slabs:
        raise ValueError("Could not generate slabs. Check Miller indices.")

    film_slab = film_slabs[0]
    sub_slab = sub_slabs[0]

    # Apply supercell transformations from ZSL
    film_transform = match["film_transformation"]
    sub_transform = match["substrate_transformation"]

    # Build 3x3 transformation matrices (2D transform + identity in z)
    ft_3x3 = np.eye(3, dtype=int)
    ft_3x3[:2, :2] = np.array(film_transform)
    st_3x3 = np.eye(3, dtype=int)
    st_3x3[:2, :2] = np.array(sub_transform)

    film_sc = film_slab.make_supercell(ft_3x3)
    sub_sc = sub_slab.make_supercell(st_3x3)

    # Convert to ASE for easier manipulation
    film_atoms = adaptor.get_atoms(film_sc)
    sub_atoms = adaptor.get_atoms(sub_sc)

    # Strain the film to match the substrate lattice (substrate is reference)
    sub_cell = sub_atoms.get_cell()
    film_atoms.set_cell(
        [sub_cell[0], sub_cell[1], film_atoms.get_cell()[2]],
        scale_atoms=True
    )

    # Position atoms: substrate at bottom, film on top with gap
    sub_pos = sub_atoms.get_positions()
    film_pos = film_atoms.get_positions()

    sub_zmax = sub_pos[:, 2].max()
    film_zmin = film_pos[:, 2].min()

    # Shift film up
    film_pos[:, 2] += (sub_zmax - film_zmin + gap)
    film_atoms.set_positions(film_pos)

    # Combine into one Atoms object
    interface = sub_atoms + film_atoms

    # Set cell height to include vacuum
    all_z = interface.get_positions()[:, 2]
    z_height = all_z.max() - all_z.min() + vacuum
    new_cell = interface.get_cell().copy()
    new_cell[2, 2] = z_height

    # Center slab
    interface.set_cell(new_cell)
    positions = interface.get_positions()
    z_center = (positions[:, 2].max() + positions[:, 2].min()) / 2
    cell_center = new_cell[2, 2] / 2
    positions[:, 2] += (cell_center - z_center)
    interface.set_positions(positions)

    interface.pbc = [True, True, True]

    n_sub = len(sub_atoms)
    n_film = len(film_atoms)
    print(f"  Substrate atoms: {n_sub}")
    print(f"  Film atoms:      {n_film}")
    print(f"  Total atoms:     {len(interface)}")
    print(f"  Cell: {new_cell[0][0]:.2f} x {new_cell[1][1]:.2f} x "
          f"{new_cell[2][2]:.2f} Ang")

    return interface, n_sub, n_film


interface, n_sub, n_film = build_heterostructure(
    mos2_bulk, graphene_bulk, best,
    gap=INTERLAYER_GAP, vacuum=VACUUM,
)

ase_write("heterostructure_initial.xyz", interface)
print("Saved heterostructure_initial.xyz")


# ------------------------------------------------------------------ #
#  Step 4: Relax with MACE (fix substrate)
# ------------------------------------------------------------------ #
print("\n=== MACE Relaxation ===")

from mace.calculators import mace_mp

calc = mace_mp(model=MACE_MODEL, default_dtype="float64")


def relax_interface(atoms, n_substrate, fix_substrate=True):
    """Relax the interface, optionally fixing substrate atoms."""
    atoms = atoms.copy()
    atoms.calc = calc

    if fix_substrate:
        fix_idx = list(range(n_substrate))
        atoms.set_constraint(FixAtoms(indices=fix_idx))
        print(f"  Fixed {n_substrate} substrate atoms")

    opt = FIRE(atoms, logfile=None)
    opt.run(fmax=FMAX, steps=500)
    print(f"  Converged: {opt.converged()}, "
          f"steps: {opt.nsteps}")

    return atoms


interface_relaxed = relax_interface(interface, n_sub, fix_substrate=True)
ase_write("heterostructure_relaxed.xyz", interface_relaxed)
print("Saved heterostructure_relaxed.xyz")


# ------------------------------------------------------------------ #
#  Step 5: Compute adhesion energy
# ------------------------------------------------------------------ #
print("\n=== Adhesion Energy ===")


def compute_adhesion_energy(interface_atoms, n_sub, n_film, calculator):
    """
    W_ad = (E_film_isolated + E_sub_isolated - E_interface) / A

    Positive W_ad means it costs energy to separate the interface
    (favorable adhesion).
    """
    # Interface energy
    interface_atoms.calc = calculator
    e_interface = interface_atoms.get_potential_energy()

    # Isolate substrate: remove film atoms
    sub_atoms = interface_atoms[:n_sub].copy()
    sub_atoms.calc = calculator
    # Add vacuum where the film was
    sub_cell = sub_atoms.get_cell().copy()
    sub_atoms.set_cell(sub_cell)
    e_sub = sub_atoms.get_potential_energy()

    # Isolate film: remove substrate atoms
    film_atoms = interface_atoms[n_sub:].copy()
    film_atoms.calc = calculator
    film_cell = film_atoms.get_cell().copy()
    film_atoms.set_cell(film_cell)
    e_film = film_atoms.get_potential_energy()

    # Interface area
    cell = interface_atoms.get_cell()
    area = np.linalg.norm(np.cross(cell[0], cell[1]))

    w_ad = (e_film + e_sub - e_interface) / area  # eV/Ang^2
    w_ad_si = w_ad * 16.02176634  # J/m^2

    print(f"  E_interface = {e_interface:.4f} eV")
    print(f"  E_substrate = {e_sub:.4f} eV")
    print(f"  E_film      = {e_film:.4f} eV")
    print(f"  Area         = {area:.2f} Ang^2")
    print(f"  W_ad         = {w_ad:.6f} eV/Ang^2 = {w_ad_si:.4f} J/m^2")

    return w_ad_si, e_interface, e_sub, e_film, area


w_ad, e_int, e_sub_val, e_film_val, area = compute_adhesion_energy(
    interface_relaxed, n_sub, n_film, calc
)


# ------------------------------------------------------------------ #
#  Step 6: Scan interlayer distance for binding curve
# ------------------------------------------------------------------ #
print("\n=== Interlayer Distance Scan ===")


def scan_interlayer_distance(atoms, n_sub, calculator,
                              d_range=np.arange(2.5, 6.0, 0.2)):
    """
    Rigidly scan the film-substrate distance and compute E(d).
    """
    sub_atoms = atoms[:n_sub]
    film_atoms_ref = atoms[n_sub:]

    sub_zmax = sub_atoms.get_positions()[:, 2].max()
    film_pos_ref = film_atoms_ref.get_positions().copy()
    film_zmin_ref = film_pos_ref[:, 2].min()

    distances = []
    energies = []

    for d in d_range:
        atoms_d = atoms.copy()
        atoms_d.calc = calculator

        # Shift film to set interlayer distance = d
        new_pos = atoms_d.get_positions()
        shift = d - (film_zmin_ref - sub_zmax)
        new_pos[n_sub:, 2] = film_pos_ref[:, 2] + shift
        atoms_d.set_positions(new_pos)

        e = atoms_d.get_potential_energy()
        distances.append(d)
        energies.append(e)

    distances = np.array(distances)
    energies = np.array(energies)

    # Normalize to per-area
    cell = atoms.get_cell()
    area = np.linalg.norm(np.cross(cell[0], cell[1]))

    return distances, energies, area


d_vals, e_vals, scan_area = scan_interlayer_distance(
    interface_relaxed, n_sub, calc,
    d_range=np.arange(2.0, 7.0, 0.25)
)

# Reference: isolated slabs energy
e_isolated = e_sub_val + e_film_val
binding_curve = (e_vals - e_isolated) / scan_area * 16.02176634  # J/m^2


# ------------------------------------------------------------------ #
#  Step 7: Plots
# ------------------------------------------------------------------ #
print("\n=== Generating Plots ===")

# --- Plot 1: Binding energy curve ---
fig, ax = plt.subplots(figsize=(7, 5))
ax.plot(d_vals, binding_curve, "o-", markersize=5, color="C0")
ax.axhline(0, color="gray", ls="--", alpha=0.5)
ax.set_xlabel("Interlayer distance (Ang)", fontsize=12)
ax.set_ylabel("Binding energy (J/m$^2$)", fontsize=12)
ax.set_title("MoS$_2$/Graphene binding curve", fontsize=13)
ax.grid(True, alpha=0.3)

# Mark minimum
imin = np.argmin(binding_curve)
ax.plot(d_vals[imin], binding_curve[imin], "r*", markersize=15,
        label=f"Min: d={d_vals[imin]:.2f} Ang, "
              f"E={binding_curve[imin]:.4f} J/m$^2$")
ax.legend(fontsize=10)
plt.tight_layout()
plt.savefig("hetero_binding_curve.png", dpi=150)
plt.close()
print("Saved hetero_binding_curve.png")

# --- Plot 2: Strain vs supercell area for all matches ---
fig, ax = plt.subplots(figsize=(7, 5))
strains = [m["strain"] * 100 for m in match_data]
areas = [m["area"] for m in match_data]
ax.scatter(areas, strains, alpha=0.6, edgecolors="black", linewidth=0.3)
ax.set_xlabel("Supercell area (Ang$^2$)", fontsize=12)
ax.set_ylabel("Von Mises strain (%)", fontsize=12)
ax.set_title("Lattice match candidates: strain vs. area", fontsize=13)
ax.axhline(MAX_STRAIN * 100, color="red", ls="--",
           label=f"Max strain = {MAX_STRAIN*100:.0f}%")

# Highlight best match
ax.plot(areas[0], strains[0], "r*", markersize=15,
        label=f"Best: {strains[0]:.2f}%, {areas[0]:.0f} Ang$^2$")
ax.legend()
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("hetero_strain_vs_area.png", dpi=150)
plt.close()
print("Saved hetero_strain_vs_area.png")

# --- Plot 3: Interface structure side view ---
fig, ax = plt.subplots(figsize=(8, 6))
pos = interface_relaxed.get_positions()
symbols = interface_relaxed.get_chemical_symbols()

color_map = {"C": "black", "Mo": "C0", "S": "gold"}
for sym in set(symbols):
    mask = np.array([s == sym for s in symbols])
    ax.scatter(pos[mask, 0], pos[mask, 2], s=30, label=sym,
               c=color_map.get(sym, "gray"), edgecolors="black",
               linewidth=0.3, alpha=0.8)

ax.set_xlabel("x (Ang)", fontsize=11)
ax.set_ylabel("z (Ang)", fontsize=11)
ax.set_title("Heterostructure side view (relaxed)", fontsize=12)
ax.legend()
ax.set_aspect("equal")
plt.tight_layout()
plt.savefig("hetero_structure_sideview.png", dpi=150)
plt.close()
print("Saved hetero_structure_sideview.png")

# --- Summary ---
print(f"\n=== Summary ===")
print(f"System: MoS2 / Graphene (0001)")
print(f"Lattice strain: {best['strain']*100:.2f}%")
print(f"Supercell area: {best['area']:.1f} Ang^2")
print(f"Adhesion energy: {w_ad:.4f} J/m^2")
print(f"Equilibrium distance: {d_vals[imin]:.2f} Ang")
print(f"Total atoms: {len(interface_relaxed)}")

print("\nDone.")
```

### Step 2 -- Alternative: Using SubstrateAnalyzer for Bulk Film on Bulk Substrate

For 3D-on-3D heterostructures (e.g., oxide on metal), the `SubstrateAnalyzer` provides a more streamlined interface:

```python
#!/usr/bin/env python3
"""
Find lattice-matched interfaces between a film (e.g., TiO2 rutile)
and a substrate (e.g., Al2O3 corundum) using SubstrateAnalyzer.
"""

import numpy as np
from pymatgen.core import Structure
from pymatgen.analysis.interfaces.substrate_analyzer import SubstrateAnalyzer
from pymatgen.analysis.interfaces.zsl import ZSLGenerator
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
from mp_api.client import MPRester
import os

MP_API_KEY = os.environ["MP_API_KEY"]

# Fetch structures from Materials Project
with MPRester(MP_API_KEY) as mpr:
    # TiO2 rutile (mp-2657)
    tio2 = mpr.get_structure_by_material_id("mp-2657")
    # Al2O3 corundum (mp-1143)
    al2o3 = mpr.get_structure_by_material_id("mp-1143")

print(f"Film:      TiO2 rutile, SG={SpacegroupAnalyzer(tio2).get_space_group_symbol()}")
print(f"Substrate: Al2O3 corundum, SG={SpacegroupAnalyzer(al2o3).get_space_group_symbol()}")

# Configure ZSL generator
zsl = ZSLGenerator(
    max_area_ratio_tol=0.09,
    max_area=300,           # Ang^2
    max_length_tol=0.03,
    max_angle_tol=0.01,
)

sub_analyzer = SubstrateAnalyzer(zslgen=zsl)

# Scan common low-index surfaces
film_millers = [(0, 0, 1), (1, 0, 0), (1, 1, 0)]
sub_millers = [(0, 0, 1), (1, 0, 0), (1, 1, 0)]

all_matches = []
for fm in film_millers:
    for sm in sub_millers:
        matches = list(sub_analyzer.calculate(
            film=tio2,
            substrate=al2o3,
            film_millers=[fm],
            substrate_millers=[sm],
        ))
        for m in matches:
            all_matches.append({
                "film_miller": fm,
                "sub_miller": sm,
                "strain": m.von_mises_strain,
                "area": abs(np.cross(
                    m.substrate_sl_vectors[0],
                    m.substrate_sl_vectors[1]
                )),
                "match": m,
            })

all_matches.sort(key=lambda m: m["strain"])

print(f"\nFound {len(all_matches)} total matches across all Miller index pairs")
print("\nTop 10 matches by strain:")
print(f"{'Film':>10} {'Sub':>10} {'Strain%':>10} {'Area':>10}")
for m in all_matches[:10]:
    print(f"{str(m['film_miller']):>10} {str(m['sub_miller']):>10} "
          f"{m['strain']*100:>9.3f}% {m['area']:>9.1f}")
```

### Step 3 -- QE Band Alignment at the Interface

```python
#!/usr/bin/env python3
"""
Write QE input for a heterostructure to compute
the electrostatic potential profile for band alignment.
"""

from ase.io import read as ase_read
from ase.io.espresso import write_espresso_in
import numpy as np

# Read the relaxed heterostructure
atoms = ase_read("heterostructure_relaxed.xyz")

# Determine k-points
cell = atoms.get_cell()
a_len = np.linalg.norm(cell[0])
b_len = np.linalg.norm(cell[1])
c_len = np.linalg.norm(cell[2])
ka = max(1, int(round(25 / a_len)))
kb = max(1, int(round(25 / b_len)))
kc = 1  # slab geometry: only 1 k-point along vacuum direction

input_data = {
    "control": {
        "calculation": "scf",
        "prefix": "hetero",
        "outdir": "./tmp_hetero",
        "pseudo_dir": "/opt/pseudo",
        "tprnfor": True,
        "tstress": True,
    },
    "system": {
        "ecutwfc": 50,
        "ecutrho": 400,
        "occupations": "smearing",
        "smearing": "cold",
        "degauss": 0.01,
        "assume_isolated": "2D",     # 2D Coulomb cutoff for slabs
    },
    "electrons": {
        "conv_thr": 1.0e-6,
        "mixing_beta": 0.3,
    },
}

# Map elements to pseudopotentials
elements = sorted(set(atoms.get_chemical_symbols()))
pseudopotentials = {el: f"{el}.pbe-n-kjpaw_psl.1.0.0.UPF" for el in elements}

with open("hetero_scf.pwi", "w") as f:
    write_espresso_in(
        f, atoms,
        input_data=input_data,
        pseudopotentials=pseudopotentials,
        kpts=(ka, kb, kc),
    )
print(f"Wrote hetero_scf.pwi (k-points: {ka}x{kb}x{kc})")
print("Run with: mpirun -np 4 pw.x -in hetero_scf.pwi > hetero_scf.pwo")
print("\nFor planar-averaged potential, use pp.x with plot_num=11 (local potential).")
```

## Key Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `MAX_STRAIN` | 0.05 (5%) | Maximum allowed lattice mismatch. Lower values give more physical interfaces but fewer/larger supercells |
| `MAX_AREA` | 200 Ang^2 | Maximum supercell area. Larger allows more match options but bigger cells |
| `INTERLAYER_GAP` | 3.3 Ang | Initial film-substrate separation. Typical van der Waals gap is 3.0-3.5 Ang |
| `VACUUM` | 20 Ang | Vacuum thickness above slab. Must be large enough to decouple periodic images (>15 Ang) |
| `max_area_ratio_tol` | 0.09 | ZSL tolerance on area ratio between film and substrate supercells |
| `max_length_tol` | 0.03 | ZSL tolerance on vector length mismatch |
| `max_angle_tol` | 0.01 | ZSL tolerance on angle mismatch (radians) |
| `fix_substrate` | True | Whether to fix substrate atoms during relaxation |
| `FMAX` | 0.03 eV/A | Force convergence threshold |

## Interpreting Results

- **Adhesion energy (W_ad)**: Positive values mean adhesion is favorable. For van der Waals heterostructures (MoS2/graphene), expect W_ad ~ 0.02-0.1 J/m^2. For chemically bonded interfaces (oxide/metal), values of 1-5 J/m^2 are typical.
- **Binding curve**: The minimum in the E(d) plot gives the equilibrium interlayer spacing and the depth gives W_ad. The shape indicates the interaction type (shallow/broad = van der Waals, deep/narrow = chemical bonding).
- **Strain**: Keep below 3-5% for physically meaningful results. Higher strains introduce artifacts. If strain is too large, try different Miller index combinations or larger MAX_AREA.
- **Strain vs. area trade-off**: The scatter plot shows that lower strain generally requires larger supercells. Choose the Pareto-optimal point that balances computational cost and physical accuracy.
- **Band alignment**: Requires QE (DFT) -- extract the planar-averaged electrostatic potential from each side of the interface to determine band offsets. MACE does not provide electronic structure.

## Common Issues

| Issue | Solution |
|-------|----------|
| ZSLGenerator returns no matches | Increase `MAX_AREA`, relax `max_length_tol` and `max_angle_tol`, or try different Miller indices |
| Very large supercell (>500 atoms) | Reduce `MAX_AREA` or accept slightly higher strain with a smaller cell |
| Film atoms move into substrate during relaxation | Increase `INTERLAYER_GAP`; ensure fix constraint is applied correctly |
| Negative adhesion energy | Indicates repulsion at the chosen distance/configuration; check interlayer spacing and atomic overlap |
| Slab too thin -- surface effects | Increase slab thickness (more layers) for both film and substrate |
| Vacuum too thin | Increase `VACUUM` to >20 Ang; for charged systems use dipole correction in QE |
| Lattice vectors not orthogonal | Normal for hexagonal systems; pymatgen and ASE handle non-orthogonal cells correctly |
| Structure looks wrong in visualization | Check that `scale_atoms=True` was used when changing the cell; verify atom counts match expectations |
| QE calculation diverges | Reduce `mixing_beta` to 0.1-0.2 for metallic interfaces; use `assume_isolated='2D'` for slabs |
