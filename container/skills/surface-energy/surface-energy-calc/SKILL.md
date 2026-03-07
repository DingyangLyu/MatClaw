# Surface Energy Calculation

## When to Use

- You need the surface energy of a specific crystal facet (e.g., (100), (110), (111)).
- You are screening multiple facets to find the most stable surface.
- You need convergence-tested surface energies for Wulff construction (see `wulff-construction/` skill).
- You are studying catalytic surfaces, thin films, or surface reconstructions.
- You need to compare surface stability across different materials or compositions.

## Method Selection

| Criterion | ASE + MACE (Method A) | QE DFT (Method B) |
|---|---|---|
| Speed | Seconds per slab | Minutes to hours per slab |
| Accuracy | Good for trends, screening | Publication quality |
| Forces/relaxation | Fast, iterative | Expensive but exact |
| Best for | Rapid screening of many facets, convergence tests | Final publication numbers, validation |
| Limitations | ML potential accuracy depends on training data | Computationally expensive |

**Recommended workflow**: Screen with MACE first (Method A), then validate the most important facets with QE DFT (Method B).

## Prerequisites

- A bulk crystal structure (CIF, POSCAR, or built with pymatgen/ASE).
- **Method A**: ASE, MACE-torch, pymatgen (pre-installed).
- **Method B**: Quantum ESPRESSO (pw.x), pseudopotential files.
- Python: numpy, matplotlib for analysis and plotting.

---

## Detailed Steps

### Surface Energy Formula

The surface energy is:

```
gamma = (E_slab - N * E_bulk_per_atom) / (2 * A)
```

where:
- `E_slab` = total energy of the slab supercell
- `N` = number of atoms in the slab
- `E_bulk_per_atom` = energy per atom of the bulk crystal
- `A` = surface area of one face of the slab
- Factor of 2 accounts for two surfaces (top and bottom)

For a relaxed asymmetric slab (one side fixed, one side relaxed), the factor of 2 still applies if the fixed side is a bulk-like termination, but be aware this introduces a small error. For best results, use symmetric slabs or apply a correction.

### Method A: ASE + MACE (Fast Screening)

```python
#!/usr/bin/env python3
"""
Surface energy calculation using ASE + MACE.
Complete workflow: bulk energy -> slab generation -> slab relaxation -> surface energy.
Includes convergence tests vs slab thickness and vacuum size.
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
from ase.io import write
from mace.calculators import mace_mp

from pymatgen.core import Structure
from pymatgen.core.surface import SlabGenerator
from pymatgen.io.ase import AseAtomsAdaptor

# ══════════════════════════════════════════════════════════════════════
# Configuration
# ══════════════════════════════════════════════════════════════════════
MATERIAL = "Cu"           # element or compound
STRUCTURE_TYPE = "fcc"    # crystal structure type
LATTICE_PARAM = 3.615     # lattice parameter in Angstrom
MILLER_INDEX = (1, 1, 1)  # Miller index for the surface
MIN_SLAB_SIZE = 10.0      # minimum slab thickness in Angstrom
MIN_VACUUM = 15.0         # vacuum thickness in Angstrom
N_FIXED_LAYERS = 2        # number of bottom layers to fix during relaxation

# ══════════════════════════════════════════════════════════════════════
# Step 1: Bulk energy per atom
# ══════════════════════════════════════════════════════════════════════
print("=" * 60)
print(f"Surface Energy Calculation: {MATERIAL} {MILLER_INDEX}")
print("=" * 60)

# Build and relax bulk
atoms_bulk = bulk(MATERIAL, STRUCTURE_TYPE, a=LATTICE_PARAM)
calc = mace_mp(model="medium", dispersion=False, device="cpu")
atoms_bulk.calc = calc

# Relax bulk (cell + positions)
filtered = FrechetCellFilter(atoms_bulk)
opt = LBFGS(filtered, logfile="/dev/null")
opt.run(fmax=0.001)

e_bulk_per_atom = atoms_bulk.get_potential_energy() / len(atoms_bulk)
print(f"\n[Step 1] Bulk energy per atom: {e_bulk_per_atom:.6f} eV")
print(f"  Relaxed lattice parameter: {atoms_bulk.cell.cellpar()[0]:.4f} Ang")

# ══════════════════════════════════════════════════════════════════════
# Step 2: Generate slab with pymatgen
# ══════════════════════════════════════════════════════════════════════
print(f"\n[Step 2] Generating slab: {MILLER_INDEX}, thickness >= {MIN_SLAB_SIZE} Ang, vacuum = {MIN_VACUUM} Ang")

# Convert ASE bulk to pymatgen Structure
adaptor = AseAtomsAdaptor()
struct_bulk = adaptor.get_structure(atoms_bulk)

# Generate slabs
slabgen = SlabGenerator(
    initial_structure=struct_bulk,
    miller_index=MILLER_INDEX,
    min_slab_size=MIN_SLAB_SIZE,  # Angstrom
    min_vacuum_size=MIN_VACUUM,   # Angstrom
    center_slab=True,             # center slab in the vacuum
    in_unit_planes=False,         # use Angstrom for thickness
    lll_reduce=True,              # reduce cell vectors
    reorient_lattice=True,        # orient surface normal along c
)

slabs = slabgen.get_slabs(
    symmetrize=False,     # True for symmetric slabs (recommended for accurate gamma)
    ftol=0.1,             # tolerance for symmetry detection
)

if len(slabs) == 0:
    raise RuntimeError("No slabs generated! Check Miller index and structure.")

# Use the first slab (usually the most common termination)
slab_pmg = slabs[0]
print(f"  Generated {len(slabs)} slab termination(s)")
print(f"  Using termination 0: {len(slab_pmg)} atoms")
print(f"  Slab cell: {slab_pmg.lattice.abc}")

# Convert to ASE Atoms
slab_atoms = adaptor.get_atoms(slab_pmg)

# ══════════════════════════════════════════════════════════════════════
# Step 3: Fix bottom layers and relax slab
# ══════════════════════════════════════════════════════════════════════
print(f"\n[Step 3] Relaxing slab (fixing bottom {N_FIXED_LAYERS} layers)...")

# Identify layers by z-coordinate
z_coords = slab_atoms.positions[:, 2]
z_sorted = np.sort(np.unique(np.round(z_coords, decimals=2)))

# Determine which atoms are in the bottom N layers
if len(z_sorted) >= N_FIXED_LAYERS:
    z_threshold = z_sorted[N_FIXED_LAYERS - 1] + 0.1  # small tolerance
    fixed_indices = [i for i, z in enumerate(z_coords) if z <= z_threshold]
else:
    fixed_indices = []

print(f"  Total atoms: {len(slab_atoms)}")
print(f"  Fixed atoms: {len(fixed_indices)} (bottom {N_FIXED_LAYERS} layers)")
print(f"  Free atoms:  {len(slab_atoms) - len(fixed_indices)}")

# Apply constraints
slab_atoms.set_constraint(FixAtoms(indices=fixed_indices))
slab_atoms.calc = calc

# Relax (positions only, not cell -- slab cell should be fixed)
opt = LBFGS(slab_atoms, logfile="/dev/null")
opt.run(fmax=0.005)

e_slab = slab_atoms.get_potential_energy()
n_atoms = len(slab_atoms)
print(f"  Slab energy: {e_slab:.6f} eV")

# ══════════════════════════════════════════════════════════════════════
# Step 4: Compute surface energy
# ══════════════════════════════════════════════════════════════════════
# Surface area = |a x b| for the slab supercell (a, b are in-plane vectors)
cell = slab_atoms.cell
a_vec = cell[0]
b_vec = cell[1]
area = np.linalg.norm(np.cross(a_vec, b_vec))  # Angstrom^2

gamma_eV_per_A2 = (e_slab - n_atoms * e_bulk_per_atom) / (2 * area)
gamma_J_per_m2 = gamma_eV_per_A2 * 16.0217663  # 1 eV/Ang^2 = 16.0217663 J/m^2

print(f"\n[Step 4] Surface Energy for {MATERIAL}{MILLER_INDEX}:")
print(f"  E_slab          = {e_slab:.6f} eV ({n_atoms} atoms)")
print(f"  E_bulk/atom     = {e_bulk_per_atom:.6f} eV")
print(f"  Cleavage energy = {e_slab - n_atoms * e_bulk_per_atom:.6f} eV")
print(f"  Surface area    = {area:.4f} Ang^2")
print(f"  gamma           = {gamma_eV_per_A2:.6f} eV/Ang^2")
print(f"  gamma           = {gamma_J_per_m2:.4f} J/m^2")

write("slab_relaxed.cif", slab_atoms)
print(f"  Relaxed slab saved: slab_relaxed.cif")

# ══════════════════════════════════════════════════════════════════════
# Step 5: Convergence test vs slab thickness
# ══════════════════════════════════════════════════════════════════════
print(f"\n[Step 5] Convergence test: surface energy vs slab thickness")

slab_thicknesses = [6.0, 8.0, 10.0, 12.0, 15.0, 18.0, 20.0]
gamma_vs_thickness = []

for thickness in slab_thicknesses:
    sg = SlabGenerator(
        initial_structure=struct_bulk,
        miller_index=MILLER_INDEX,
        min_slab_size=thickness,
        min_vacuum_size=MIN_VACUUM,
        center_slab=True,
        in_unit_planes=False,
        lll_reduce=True,
        reorient_lattice=True,
    )
    slabs_t = sg.get_slabs(symmetrize=False)
    if len(slabs_t) == 0:
        continue

    slab_t = adaptor.get_atoms(slabs_t[0])
    n_at = len(slab_t)

    # Fix bottom layers
    z_t = slab_t.positions[:, 2]
    z_sorted_t = np.sort(np.unique(np.round(z_t, decimals=2)))
    if len(z_sorted_t) >= N_FIXED_LAYERS:
        z_thr = z_sorted_t[N_FIXED_LAYERS - 1] + 0.1
        fix_idx = [i for i, z in enumerate(z_t) if z <= z_thr]
    else:
        fix_idx = []

    slab_t.set_constraint(FixAtoms(indices=fix_idx))
    slab_t.calc = calc

    opt = LBFGS(slab_t, logfile="/dev/null")
    opt.run(fmax=0.005)

    e_slab_t = slab_t.get_potential_energy()
    cell_t = slab_t.cell
    area_t = np.linalg.norm(np.cross(cell_t[0], cell_t[1]))

    gamma_t = (e_slab_t - n_at * e_bulk_per_atom) / (2 * area_t) * 16.0217663
    gamma_vs_thickness.append((thickness, n_at, gamma_t))
    print(f"  thickness >= {thickness:5.1f} Ang ({n_at:3d} atoms): gamma = {gamma_t:.4f} J/m^2")

# ══════════════════════════════════════════════════════════════════════
# Step 6: Convergence test vs vacuum size
# ══════════════════════════════════════════════════════════════════════
print(f"\n[Step 6] Convergence test: surface energy vs vacuum size")

vacuum_sizes = [8.0, 10.0, 12.0, 15.0, 18.0, 20.0, 25.0]
gamma_vs_vacuum = []

for vac in vacuum_sizes:
    sg = SlabGenerator(
        initial_structure=struct_bulk,
        miller_index=MILLER_INDEX,
        min_slab_size=MIN_SLAB_SIZE,
        min_vacuum_size=vac,
        center_slab=True,
        in_unit_planes=False,
        lll_reduce=True,
        reorient_lattice=True,
    )
    slabs_v = sg.get_slabs(symmetrize=False)
    if len(slabs_v) == 0:
        continue

    slab_v = adaptor.get_atoms(slabs_v[0])
    n_at = len(slab_v)

    z_v = slab_v.positions[:, 2]
    z_sorted_v = np.sort(np.unique(np.round(z_v, decimals=2)))
    if len(z_sorted_v) >= N_FIXED_LAYERS:
        z_thr = z_sorted_v[N_FIXED_LAYERS - 1] + 0.1
        fix_idx = [i for i, z in enumerate(z_v) if z <= z_thr]
    else:
        fix_idx = []

    slab_v.set_constraint(FixAtoms(indices=fix_idx))
    slab_v.calc = calc

    opt = LBFGS(slab_v, logfile="/dev/null")
    opt.run(fmax=0.005)

    e_slab_v = slab_v.get_potential_energy()
    cell_v = slab_v.cell
    area_v = np.linalg.norm(np.cross(cell_v[0], cell_v[1]))

    gamma_v = (e_slab_v - n_at * e_bulk_per_atom) / (2 * area_v) * 16.0217663
    gamma_vs_vacuum.append((vac, gamma_v))
    print(f"  vacuum = {vac:5.1f} Ang: gamma = {gamma_v:.4f} J/m^2")

# ══════════════════════════════════════════════════════════════════════
# Step 7: Plot convergence
# ══════════════════════════════════════════════════════════════════════
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

# Thickness convergence
if gamma_vs_thickness:
    thk = [x[0] for x in gamma_vs_thickness]
    gam = [x[2] for x in gamma_vs_thickness]
    ax1.plot(thk, gam, "o-", color="steelblue", linewidth=1.5, markersize=6)
    ax1.set_xlabel("Minimum slab thickness (Ang)", fontsize=13)
    ax1.set_ylabel(r"$\gamma$ (J/m$^2$)", fontsize=13)
    ax1.set_title(f"{MATERIAL}{MILLER_INDEX} - Thickness Convergence", fontsize=14)
    ax1.grid(True, alpha=0.3)
    # Add converged region shading
    if len(gam) >= 3:
        converged = gam[-1]
        ax1.axhspan(converged - 0.05, converged + 0.05, alpha=0.1, color="green")

# Vacuum convergence
if gamma_vs_vacuum:
    vacs = [x[0] for x in gamma_vs_vacuum]
    gams = [x[1] for x in gamma_vs_vacuum]
    ax2.plot(vacs, gams, "s-", color="coral", linewidth=1.5, markersize=6)
    ax2.set_xlabel("Vacuum thickness (Ang)", fontsize=13)
    ax2.set_ylabel(r"$\gamma$ (J/m$^2$)", fontsize=13)
    ax2.set_title(f"{MATERIAL}{MILLER_INDEX} - Vacuum Convergence", fontsize=14)
    ax2.grid(True, alpha=0.3)
    if len(gams) >= 3:
        converged_v = gams[-1]
        ax2.axhspan(converged_v - 0.05, converged_v + 0.05, alpha=0.1, color="green")

plt.tight_layout()
plt.savefig("surface_energy_convergence.png", dpi=200, bbox_inches="tight")
print(f"\nSaved: surface_energy_convergence.png")
```

### Multiple Miller Indices Comparison (MACE)

```python
#!/usr/bin/env python3
"""
Compare surface energies across multiple Miller indices using MACE.
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

# ── Configuration ───────────────────────────────────────────────────
MATERIAL = "Cu"
STRUCTURE_TYPE = "fcc"
LATTICE_PARAM = 3.615
MILLER_INDICES = [
    (1, 0, 0),
    (1, 1, 0),
    (1, 1, 1),
    (2, 1, 0),
    (2, 1, 1),
    (2, 2, 1),
    (3, 1, 0),
    (3, 1, 1),
]
SLAB_THICKNESS = 15.0  # Angstrom
VACUUM = 15.0           # Angstrom
N_FIXED_LAYERS = 2

# ── Bulk energy ─────────────────────────────────────────────────────
calc = mace_mp(model="medium", dispersion=False, device="cpu")
atoms_bulk = bulk(MATERIAL, STRUCTURE_TYPE, a=LATTICE_PARAM)
atoms_bulk.calc = calc
filtered = FrechetCellFilter(atoms_bulk)
opt = LBFGS(filtered, logfile="/dev/null")
opt.run(fmax=0.001)
e_bulk_per_atom = atoms_bulk.get_potential_energy() / len(atoms_bulk)

adaptor = AseAtomsAdaptor()
struct_bulk = adaptor.get_structure(atoms_bulk)

print(f"Bulk energy per atom: {e_bulk_per_atom:.6f} eV")
print(f"{'Miller Index':>15s} | {'Atoms':>6s} | {'Area (Ang^2)':>12s} | {'gamma (J/m^2)':>14s}")
print("-" * 60)

results = []

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
            print(f"  {str(hkl):>15s} | {'SKIP':>6s} | {'no slab':>12s} | {'N/A':>14s}")
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
        results.append((hkl, n_at, area, gamma))
        print(f"  {str(hkl):>15s} | {n_at:>6d} | {area:>12.4f} | {gamma:>14.4f}")

    except Exception as e:
        print(f"  {str(hkl):>15s} | ERROR: {e}")

# ── Plot comparison ─────────────────────────────────────────────────
if results:
    labels = [str(r[0]) for r in results]
    gammas = [r[3] for r in results]

    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(len(labels))
    bars = ax.bar(x, gammas, color="steelblue", edgecolor="black", linewidth=0.5)

    # Highlight the lowest surface energy
    min_idx = np.argmin(gammas)
    bars[min_idx].set_color("coral")

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=11, rotation=45, ha="right")
    ax.set_ylabel(r"$\gamma$ (J/m$^2$)", fontsize=13)
    ax.set_title(f"{MATERIAL} Surface Energies (MACE)", fontsize=14)
    ax.grid(axis="y", alpha=0.3)

    # Annotate values
    for i, (xi, gi) in enumerate(zip(x, gammas)):
        ax.text(xi, gi + 0.02, f"{gi:.3f}", ha="center", va="bottom", fontsize=9)

    plt.tight_layout()
    plt.savefig("surface_energy_comparison.png", dpi=200, bbox_inches="tight")
    print(f"\nSaved: surface_energy_comparison.png")
    print(f"\nLowest surface energy: {labels[min_idx]} with gamma = {gammas[min_idx]:.4f} J/m^2")
```

### Method B: QE DFT Surface Energy

```python
#!/usr/bin/env python3
"""
Surface energy calculation using Quantum ESPRESSO DFT.
Workflow: bulk SCF -> slab SCF -> compute surface energy.
"""
import os
import subprocess
import re
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from pymatgen.core import Structure
from pymatgen.core.surface import SlabGenerator
from pymatgen.io.ase import AseAtomsAdaptor
from ase.build import bulk
from ase.data import chemical_symbols, atomic_masses

# ══════════════════════════════════════════════════════════════════════
# Configuration
# ══════════════════════════════════════════════════════════════════════
PSEUDO_DIR = os.path.abspath("./pseudo")
OUTDIR_BULK = os.path.abspath("./tmp_surf_bulk")
OUTDIR_SLAB = os.path.abspath("./tmp_surf_slab")
os.makedirs(PSEUDO_DIR, exist_ok=True)
os.makedirs(OUTDIR_BULK, exist_ok=True)
os.makedirs(OUTDIR_SLAB, exist_ok=True)

MATERIAL = "Cu"
STRUCTURE_TYPE = "fcc"
LATTICE_PARAM = 3.615
MILLER_INDEX = (1, 1, 1)
SLAB_THICKNESS = 12.0   # Angstrom
VACUUM = 15.0            # Angstrom
N_FIXED_LAYERS = 2
ECUTWFC = 50.0
ECUTRHO = 400.0
KPOINTS_BULK = "8 8 8"
KPOINTS_SLAB = "8 8 1"  # only 1 k-point along vacuum direction
NPROC = 4

# Pseudopotential filename (must match files in PSEUDO_DIR)
PP_FILES = {"Cu": "Cu.pbe-dn-rrkjus_psl.1.0.0.UPF"}

# Download pseudopotentials
for elem, ppname in PP_FILES.items():
    pp_path = os.path.join(PSEUDO_DIR, ppname)
    if not os.path.exists(pp_path):
        url = f"https://pseudopotentials.quantum-espresso.org/upf_files/{ppname}"
        subprocess.run(["wget", "-q", "-O", pp_path, url], check=True)
        print(f"Downloaded {ppname}")

# ══════════════════════════════════════════════════════════════════════
# Helper: generate QE input from pymatgen Structure
# ══════════════════════════════════════════════════════════════════════
def make_qe_input(structure, prefix, outdir, calculation="scf",
                   ecutwfc=50.0, ecutrho=400.0, kpoints="8 8 8",
                   pp_files=None, constraints=None, occupations="smearing",
                   smearing="cold", degauss=0.02):
    """
    Generate a QE pw.x input string from a pymatgen Structure.
    constraints: list of atom indices to fix (if_pos = 0 0 0)
    """
    if pp_files is None:
        pp_files = {}

    species = sorted(set(str(s) for s in structure.species))
    nat = len(structure)
    ntyp = len(species)

    # Cell parameters
    cell_lines = ""
    for row in structure.lattice.matrix:
        cell_lines += f"  {row[0]:16.10f} {row[1]:16.10f} {row[2]:16.10f}\n"

    # Atomic species
    species_lines = ""
    for sp in species:
        z = [i for i, s in enumerate(chemical_symbols) if s == sp][0]
        mass = atomic_masses[z]
        pp = pp_files.get(sp, f"{sp}.pbe-n-rrkjus_psl.1.0.0.UPF")
        species_lines += f"  {sp:4s} {mass:10.4f}  {pp}\n"

    # Atomic positions (with constraints)
    pos_lines = ""
    for i, site in enumerate(structure):
        sp = str(site.specie)
        fc = site.frac_coords
        line = f"  {sp:4s} {fc[0]:14.10f} {fc[1]:14.10f} {fc[2]:14.10f}"
        if constraints and i in constraints:
            line += "  0 0 0"
        pos_lines += line + "\n"

    inp = f"""&CONTROL
    calculation = '{calculation}'
    prefix      = '{prefix}'
    outdir      = '{outdir}'
    pseudo_dir  = '{PSEUDO_DIR}'
    tprnfor     = .true.
    tstress     = .true.
/

&SYSTEM
    ibrav       = 0
    nat         = {nat}
    ntyp        = {ntyp}
    ecutwfc     = {ecutwfc}
    ecutrho     = {ecutrho}
    occupations = '{occupations}'
    smearing    = '{smearing}'
    degauss     = {degauss}
/

&ELECTRONS
    conv_thr    = 1.0d-8
    mixing_beta = 0.4
/

CELL_PARAMETERS (angstrom)
{cell_lines}
ATOMIC_SPECIES
{species_lines}
ATOMIC_POSITIONS (crystal)
{pos_lines}
K_POINTS (automatic)
  {kpoints} 0 0 0
"""
    return inp


def run_qe(input_file, output_file, nproc=4, timeout=1200):
    """Run pw.x and return stdout."""
    r = subprocess.run(
        ["mpirun", "--allow-run-as-root", "-np", str(nproc), "pw.x", "-in", input_file],
        capture_output=True, text=True, timeout=timeout
    )
    with open(output_file, "w") as f:
        f.write(r.stdout)
    return r


def extract_total_energy(output_text):
    """Extract total energy from QE output (in eV)."""
    # QE reports energy in Ry. 1 Ry = 13.605693 eV
    for line in reversed(output_text.split("\n")):
        if "!" in line and "total energy" in line:
            m = re.search(r"=\s+([-\d.]+)\s+Ry", line)
            if m:
                return float(m.group(1)) * 13.605693  # convert to eV
    return None

# ══════════════════════════════════════════════════════════════════════
# Step 1: Bulk SCF
# ══════════════════════════════════════════════════════════════════════
print("[1/3] Running bulk SCF...")
atoms_bulk = bulk(MATERIAL, STRUCTURE_TYPE, a=LATTICE_PARAM)
adaptor = AseAtomsAdaptor()
struct_bulk = adaptor.get_structure(atoms_bulk)

bulk_input = make_qe_input(
    struct_bulk, prefix="bulk", outdir=OUTDIR_BULK,
    ecutwfc=ECUTWFC, ecutrho=ECUTRHO, kpoints=KPOINTS_BULK,
    pp_files=PP_FILES,
)
with open("bulk_scf.in", "w") as f:
    f.write(bulk_input)

r_bulk = run_qe("bulk_scf.in", "bulk_scf.out", nproc=NPROC)
e_bulk_total = extract_total_energy(r_bulk.stdout)
if e_bulk_total is None:
    raise RuntimeError("Failed to extract bulk energy! Check bulk_scf.out")

n_bulk = len(struct_bulk)
e_bulk_per_atom = e_bulk_total / n_bulk
print(f"  Bulk total energy: {e_bulk_total:.6f} eV ({n_bulk} atoms)")
print(f"  Bulk energy/atom:  {e_bulk_per_atom:.6f} eV")

# ══════════════════════════════════════════════════════════════════════
# Step 2: Generate slab and run slab SCF
# ══════════════════════════════════════════════════════════════════════
print(f"\n[2/3] Generating and computing {MILLER_INDEX} slab...")

sg = SlabGenerator(
    initial_structure=struct_bulk,
    miller_index=MILLER_INDEX,
    min_slab_size=SLAB_THICKNESS,
    min_vacuum_size=VACUUM,
    center_slab=True,
    in_unit_planes=False,
    lll_reduce=True,
    reorient_lattice=True,
)
slabs = sg.get_slabs(symmetrize=False)
slab_pmg = slabs[0]
print(f"  Slab: {len(slab_pmg)} atoms")

# Identify atoms to fix (bottom N_FIXED_LAYERS layers)
z_coords = np.array([site.frac_coords[2] for site in slab_pmg])
z_unique = np.sort(np.unique(np.round(z_coords, decimals=4)))
if len(z_unique) >= N_FIXED_LAYERS:
    z_thr = z_unique[N_FIXED_LAYERS - 1] + 0.001
    fixed_indices = [i for i, z in enumerate(z_coords) if z <= z_thr]
else:
    fixed_indices = []

print(f"  Fixed atoms (bottom {N_FIXED_LAYERS} layers): {len(fixed_indices)}")

# For DFT surface energy, we typically do a static (unrelaxed) calculation
# or a relax-ions calculation. Static is simpler and gives the unrelaxed
# surface energy. For relaxed gamma, use calculation='relax'.
slab_input = make_qe_input(
    slab_pmg, prefix="slab", outdir=OUTDIR_SLAB,
    calculation="scf",  # Change to 'relax' for relaxed surface energy
    ecutwfc=ECUTWFC, ecutrho=ECUTRHO,
    kpoints=KPOINTS_SLAB,
    pp_files=PP_FILES,
    constraints=fixed_indices,
)
with open("slab_scf.in", "w") as f:
    f.write(slab_input)

r_slab = run_qe("slab_scf.in", "slab_scf.out", nproc=NPROC, timeout=1800)
e_slab_total = extract_total_energy(r_slab.stdout)
if e_slab_total is None:
    raise RuntimeError("Failed to extract slab energy! Check slab_scf.out")

n_slab = len(slab_pmg)
print(f"  Slab total energy: {e_slab_total:.6f} eV ({n_slab} atoms)")

# ══════════════════════════════════════════════════════════════════════
# Step 3: Compute surface energy
# ══════════════════════════════════════════════════════════════════════
print(f"\n[3/3] Computing surface energy...")

# Surface area
a_vec = slab_pmg.lattice.matrix[0]
b_vec = slab_pmg.lattice.matrix[1]
area = np.linalg.norm(np.cross(a_vec, b_vec))

gamma_eV_A2 = (e_slab_total - n_slab * e_bulk_per_atom) / (2 * area)
gamma_J_m2 = gamma_eV_A2 * 16.0217663

print(f"\n{'='*50}")
print(f"  Surface Energy: {MATERIAL}{MILLER_INDEX}")
print(f"  E_slab       = {e_slab_total:.6f} eV")
print(f"  N_atoms      = {n_slab}")
print(f"  E_bulk/atom  = {e_bulk_per_atom:.6f} eV")
print(f"  Area         = {area:.4f} Ang^2")
print(f"  gamma        = {gamma_eV_A2:.6f} eV/Ang^2")
print(f"  gamma        = {gamma_J_m2:.4f} J/m^2")
print(f"{'='*50}")

# ── Convergence test: vary number of layers ─────────────────────────
print(f"\nConvergence test: surface energy vs slab thickness (DFT)")
thicknesses = [8.0, 10.0, 12.0, 15.0]
conv_results = []

for thk in thicknesses:
    sg_t = SlabGenerator(
        initial_structure=struct_bulk,
        miller_index=MILLER_INDEX,
        min_slab_size=thk,
        min_vacuum_size=VACUUM,
        center_slab=True,
        in_unit_planes=False,
        lll_reduce=True,
        reorient_lattice=True,
    )
    slabs_t = sg_t.get_slabs(symmetrize=False)
    if len(slabs_t) == 0:
        continue
    slab_t = slabs_t[0]
    n_at = len(slab_t)

    z_t = np.array([site.frac_coords[2] for site in slab_t])
    z_u = np.sort(np.unique(np.round(z_t, decimals=4)))
    z_thr = z_u[min(N_FIXED_LAYERS, len(z_u)) - 1] + 0.001 if len(z_u) >= N_FIXED_LAYERS else 0
    fix_t = [i for i, z in enumerate(z_t) if z <= z_thr]

    inp_t = make_qe_input(
        slab_t, prefix=f"slab_t{int(thk)}", outdir=OUTDIR_SLAB,
        ecutwfc=ECUTWFC, ecutrho=ECUTRHO,
        kpoints=KPOINTS_SLAB, pp_files=PP_FILES,
        constraints=fix_t,
    )
    inp_file = f"slab_t{int(thk)}_scf.in"
    out_file = f"slab_t{int(thk)}_scf.out"
    with open(inp_file, "w") as f:
        f.write(inp_t)

    r_t = run_qe(inp_file, out_file, nproc=NPROC, timeout=1800)
    e_t = extract_total_energy(r_t.stdout)
    if e_t is not None:
        area_t = np.linalg.norm(np.cross(slab_t.lattice.matrix[0], slab_t.lattice.matrix[1]))
        g_t = (e_t - n_at * e_bulk_per_atom) / (2 * area_t) * 16.0217663
        conv_results.append((thk, n_at, g_t))
        print(f"  thickness >= {thk:.0f} Ang ({n_at} atoms): gamma = {g_t:.4f} J/m^2")

# Plot convergence
if conv_results:
    fig, ax = plt.subplots(figsize=(7, 5))
    thks = [r[0] for r in conv_results]
    gams = [r[2] for r in conv_results]
    ax.plot(thks, gams, "o-", color="steelblue", linewidth=1.5, markersize=6)
    ax.set_xlabel("Minimum slab thickness (Ang)", fontsize=13)
    ax.set_ylabel(r"$\gamma$ (J/m$^2$)", fontsize=13)
    ax.set_title(f"{MATERIAL}{MILLER_INDEX} Surface Energy Convergence (QE DFT)", fontsize=14)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("surface_energy_convergence_dft.png", dpi=200, bbox_inches="tight")
    print("Saved: surface_energy_convergence_dft.png")
```

---

## Key Parameters

| Parameter | Typical Value | Notes |
|---|---|---|
| `min_slab_size` | 10-20 Ang (5-10 layers) | Must be converged. Thicker = more accurate but more expensive. |
| `min_vacuum_size` | 12-20 Ang | Must prevent slab-slab interaction across periodic boundary. 15 Ang is usually safe. |
| `N_FIXED_LAYERS` | 2-3 | Bottom layers fixed to bulk positions. Prevents artificial relaxation. |
| `ecutwfc` / `ecutrho` | 50 / 400 Ry (typical) | Must match bulk calculation. Use converged values. |
| `kpoints` (slab) | Nx Ny 1 | Only 1 k-point in the vacuum direction (c-axis). In-plane k-points same density as bulk. |
| `smearing` / `degauss` | cold / 0.02 Ry | For metals. For insulators, use fixed occupations or very small degauss. |

### Convergence Checklist

1. **Slab thickness**: Increase layers until gamma changes by < 0.01 J/m^2.
2. **Vacuum size**: Increase until gamma changes by < 0.01 J/m^2. Typically 15 Ang is sufficient.
3. **K-points**: Test in-plane k-grid density (e.g., 6x6x1 vs 8x8x1 vs 10x10x1).
4. **ecutwfc**: Should already be converged from bulk tests.
5. **Bulk energy**: Must use the SAME pseudopotential and ecutwfc as the slab.

## Interpreting Results

- **Typical surface energies**: Metals 1-3 J/m^2. Oxides 0.5-2 J/m^2. Semiconductors 0.5-2 J/m^2.
- **FCC metals**: gamma(111) < gamma(100) < gamma(110) is the typical ordering. Close-packed surfaces are most stable.
- **Negative surface energy**: Indicates an error. Common causes: inconsistent bulk/slab parameters, too-thin slab, or basis set mismatch.
- **MACE vs DFT**: MACE values are approximate. Expect 10-30% deviation from DFT for absolute values, but relative ordering of facets is usually preserved.
- **Relaxation effect**: Relaxed surface energy is always lower than or equal to unrelaxed (cleavage) energy. The difference indicates the magnitude of surface reconstruction.

## Common Issues

| Problem | Solution |
|---|---|
| Negative surface energy | Check that bulk and slab use identical pseudopotentials, ecutwfc, ecutrho. Ensure bulk energy/atom is correctly computed. |
| Surface energy not converging with thickness | May need > 10 layers for polar surfaces or surfaces with strong relaxation. Try symmetric slabs. |
| Slab SCF does not converge | Reduce `mixing_beta` to 0.2-0.3. Use `mixing_mode = 'local-TF'`. For metals, ensure adequate smearing. |
| Pymatgen generates weird slabs | Check that the initial structure is properly symmetrized. Try `symmetrize=True` in `get_slabs()`. |
| Different terminations give different energies | This is physical for polar surfaces. Report the range or the average. Use symmetric slabs to avoid dipole issues. |
| Dipole correction needed | For asymmetric slabs of polar surfaces, add `dipfield = .true.` and `tefield = .true.` in QE `&SYSTEM`. |
| ASE constraint error | Ensure fixed indices are within range [0, N-1]. Check that layer identification works for your slab geometry. |
