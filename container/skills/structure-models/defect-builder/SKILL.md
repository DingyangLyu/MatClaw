# Point Defect Builder

## When to Use

- You need to create a vacancy, substitution, or interstitial defect in a crystal supercell.
- You want to systematically generate all symmetry-inequivalent defect sites.
- You need multiple defect types (single vacancy, double vacancy, Frenkel pair) for a given host.
- You want to screen defect formation energies across different sites or species.
- You are preparing defect structures for DFT or MACE relaxation and formation energy calculations.

## Method Selection

| Criterion | pymatgen manual | pymatgen-analysis-defects | ASE manual |
|---|---|---|---|
| Vacancy generation | Simple (remove site) | Automatic, symmetry-aware | Simple (pop atom) |
| Substitution | Simple (replace species) | Automatic, all inequivalent sites | Simple (set symbol) |
| Interstitial | Manual coordinate | Voronoi-based site finding | Manual coordinate |
| Charge states | Not handled | Built-in charge state support | Not handled |
| Best for | Quick single defect | Systematic defect studies | ASE-based workflows |

## Prerequisites

- pymatgen (pre-installed)
- ASE (pre-installed)
- numpy, matplotlib (pre-installed)
- Optional: `pip install pymatgen-analysis-defects` for advanced defect workflows
- Optional: MACE for quick relaxation screening

---

## Detailed Steps

### Method A: pymatgen Defect Construction

```python
#!/usr/bin/env python3
"""
Build point defects (vacancy, substitution, interstitial) in crystal supercells.
Systematically generates all symmetry-inequivalent defect sites.
"""

import numpy as np
import json
import warnings
warnings.filterwarnings("ignore")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from pymatgen.core import Structure, Element, Lattice
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
from pymatgen.io.vasp import Poscar
from pymatgen.io.ase import AseAtomsAdaptor
from pathlib import Path

output_dir = Path("defect_models")
output_dir.mkdir(exist_ok=True)

# ============================================================
# 1. Build and relax the host structure
# ============================================================
# MgO rocksalt as example
host = Structure.from_spacegroup(
    "Fm-3m",
    lattice=Lattice.cubic(4.212),
    species=["Mg", "O"],
    coords=[[0.0, 0.0, 0.0], [0.5, 0.5, 0.5]],
)
print(f"Host: {host.composition.reduced_formula}")

sga = SpacegroupAnalyzer(host)
print(f"Space group: {sga.get_space_group_symbol()}")

# ============================================================
# 2. Build supercell
# ============================================================
SC_SIZE = [3, 3, 3]
bulk_sc = host.copy()
bulk_sc.make_supercell(SC_SIZE)
n_bulk = len(bulk_sc)
print(f"Supercell: {SC_SIZE}, {n_bulk} atoms")

bulk_sc.to(str(output_dir / "bulk_supercell.cif"))
Poscar(bulk_sc).write_file(str(output_dir / "POSCAR_bulk"))

# ============================================================
# 3. Find symmetry-inequivalent sites
# ============================================================
def get_inequivalent_sites(structure, element=None, symprec=0.01):
    """
    Find all symmetry-inequivalent sites for a given element.
    Returns a list of (site_index, wyckoff_label) tuples.
    """
    sga = SpacegroupAnalyzer(structure, symprec=symprec)
    sym_structure = sga.get_symmetrized_structure()

    inequiv_sites = []
    for equiv_sites in sym_structure.equivalent_sites:
        site = equiv_sites[0]
        if element is None or site.specie == Element(element):
            # Find this site's index in the original structure
            idx = None
            for i, s in enumerate(structure):
                if np.allclose(s.frac_coords, site.frac_coords, atol=symprec):
                    idx = i
                    break
            if idx is not None:
                inequiv_sites.append({
                    "index": idx,
                    "species": str(site.specie),
                    "frac_coords": site.frac_coords.tolist(),
                    "multiplicity": len(equiv_sites),
                })

    return inequiv_sites

# In MgO, all Mg sites are equivalent and all O sites are equivalent
mg_sites = get_inequivalent_sites(bulk_sc, "Mg")
o_sites = get_inequivalent_sites(bulk_sc, "O")

print(f"\nInequivalent Mg sites: {len(mg_sites)}")
for s in mg_sites:
    print(f"  Index {s['index']}: ({s['frac_coords'][0]:.3f}, "
          f"{s['frac_coords'][1]:.3f}, {s['frac_coords'][2]:.3f}), "
          f"multiplicity = {s['multiplicity']}")

print(f"Inequivalent O sites: {len(o_sites)}")
for s in o_sites:
    print(f"  Index {s['index']}: ({s['frac_coords'][0]:.3f}, "
          f"{s['frac_coords'][1]:.3f}, {s['frac_coords'][2]:.3f}), "
          f"multiplicity = {s['multiplicity']}")

# ============================================================
# 4. Create vacancy defects
# ============================================================
print("\n=== Vacancy Defects ===")

def create_vacancy(structure, site_index, label=None):
    """
    Create a vacancy by removing an atom at site_index.
    Returns the defect structure and info dict.
    """
    defect = structure.copy()
    removed_species = str(defect[site_index].specie)
    removed_coords = defect[site_index].frac_coords.copy()
    defect.remove_sites([site_index])

    info = {
        "type": "vacancy",
        "removed_species": removed_species,
        "removed_site_index": site_index,
        "removed_frac_coords": removed_coords.tolist(),
        "n_atoms": len(defect),
        "label": label or f"V_{removed_species}",
    }
    return defect, info

# O vacancy (most common in MgO)
vac_O, vac_O_info = create_vacancy(bulk_sc, o_sites[0]["index"], "V_O")
print(f"  {vac_O_info['label']}: removed {vac_O_info['removed_species']} "
      f"at {vac_O_info['removed_frac_coords']}, {vac_O_info['n_atoms']} atoms")
vac_O.to(str(output_dir / "vacancy_O.cif"))
Poscar(vac_O).write_file(str(output_dir / "POSCAR_V_O"))

# Mg vacancy
vac_Mg, vac_Mg_info = create_vacancy(bulk_sc, mg_sites[0]["index"], "V_Mg")
print(f"  {vac_Mg_info['label']}: removed {vac_Mg_info['removed_species']}, "
      f"{vac_Mg_info['n_atoms']} atoms")
vac_Mg.to(str(output_dir / "vacancy_Mg.cif"))
Poscar(vac_Mg).write_file(str(output_dir / "POSCAR_V_Mg"))

# ============================================================
# 5. Create substitutional defects
# ============================================================
print("\n=== Substitutional Defects ===")

def create_substitution(structure, site_index, new_species, label=None):
    """
    Create a substitutional defect by replacing an atom.
    """
    defect = structure.copy()
    original_species = str(defect[site_index].specie)
    defect.replace(site_index, new_species)

    info = {
        "type": "substitution",
        "original_species": original_species,
        "new_species": new_species,
        "site_index": site_index,
        "frac_coords": defect[site_index].frac_coords.tolist(),
        "n_atoms": len(defect),
        "label": label or f"{new_species}_{original_species}",
    }
    return defect, info

# Al substituting Mg (Al_Mg)
sub_Al, sub_Al_info = create_substitution(
    bulk_sc, mg_sites[0]["index"], "Al", "Al_Mg"
)
print(f"  {sub_Al_info['label']}: {sub_Al_info['original_species']} -> "
      f"{sub_Al_info['new_species']}, {sub_Al_info['n_atoms']} atoms")
sub_Al.to(str(output_dir / "sub_Al_Mg.cif"))
Poscar(sub_Al).write_file(str(output_dir / "POSCAR_Al_Mg"))

# Ca substituting Mg (Ca_Mg)
sub_Ca, sub_Ca_info = create_substitution(
    bulk_sc, mg_sites[0]["index"], "Ca", "Ca_Mg"
)
print(f"  {sub_Ca_info['label']}: {sub_Ca_info['original_species']} -> "
      f"{sub_Ca_info['new_species']}, {sub_Ca_info['n_atoms']} atoms")
sub_Ca.to(str(output_dir / "sub_Ca_Mg.cif"))

# N substituting O (N_O)
sub_N, sub_N_info = create_substitution(
    bulk_sc, o_sites[0]["index"], "N", "N_O"
)
print(f"  {sub_N_info['label']}: {sub_N_info['original_species']} -> "
      f"{sub_N_info['new_species']}, {sub_N_info['n_atoms']} atoms")
sub_N.to(str(output_dir / "sub_N_O.cif"))

# ============================================================
# 6. Create interstitial defects
# ============================================================
print("\n=== Interstitial Defects ===")

def find_interstitial_sites(structure, symprec=0.01):
    """
    Find interstitial sites using a simple Voronoi-like approach.
    Returns candidate sites at high-symmetry interstitial positions.
    """
    # Common interstitial positions for rocksalt:
    # Tetrahedral: (0.25, 0.25, 0.25) and equivalents
    # Octahedral: (0.5, 0.5, 0.5) -- but this is occupied in rocksalt
    # For other structures, use pymatgen-analysis-defects

    # Simple approach: generate a grid and find points far from all atoms
    from scipy.spatial import Voronoi

    cart_coords = structure.cart_coords
    cell = structure.lattice.matrix

    # Generate periodic images for Voronoi
    extended_coords = []
    for i in range(-1, 2):
        for j in range(-1, 2):
            for k in range(-1, 2):
                shift = i * cell[0] + j * cell[1] + k * cell[2]
                for c in cart_coords:
                    extended_coords.append(c + shift)
    extended_coords = np.array(extended_coords)

    # Voronoi decomposition
    vor = Voronoi(extended_coords)

    # Find Voronoi vertices inside the unit cell
    interstitial_candidates = []
    for vertex in vor.vertices:
        frac = structure.lattice.get_fractional_coords(vertex)
        if np.all(frac >= -0.01) and np.all(frac <= 1.01):
            # Check distance to nearest atom
            dists = structure.lattice.get_all_distances(
                [frac % 1.0],
                structure.frac_coords,
            )[0]
            min_dist = np.min(dists)
            if min_dist > 1.0:  # At least 1 A from nearest atom
                interstitial_candidates.append({
                    "frac_coords": (frac % 1.0).tolist(),
                    "min_distance_to_atom": float(min_dist),
                })

    # Remove duplicates (within tolerance)
    unique_sites = []
    for cand in interstitial_candidates:
        is_duplicate = False
        for existing in unique_sites:
            diff = np.array(cand["frac_coords"]) - np.array(existing["frac_coords"])
            diff -= np.round(diff)
            if np.linalg.norm(structure.lattice.get_cartesian_coords(diff)) < 0.5:
                is_duplicate = True
                break
        if not is_duplicate:
            unique_sites.append(cand)

    # Sort by distance (prefer sites far from existing atoms)
    unique_sites.sort(key=lambda x: -x["min_distance_to_atom"])

    return unique_sites[:10]  # Top 10 candidates

interstitial_sites = find_interstitial_sites(bulk_sc)
print(f"  Found {len(interstitial_sites)} candidate interstitial sites:")
for i, site in enumerate(interstitial_sites[:5]):
    fc = site["frac_coords"]
    print(f"    Site {i}: ({fc[0]:.3f}, {fc[1]:.3f}, {fc[2]:.3f}), "
          f"min_dist = {site['min_distance_to_atom']:.2f} A")

def create_interstitial(structure, frac_coords, species, label=None):
    """
    Create an interstitial defect by adding an atom at frac_coords.
    """
    defect = structure.copy()
    defect.append(species, frac_coords)

    info = {
        "type": "interstitial",
        "species": species,
        "frac_coords": list(frac_coords),
        "n_atoms": len(defect),
        "label": label or f"{species}_i",
    }
    return defect, info

# Create O interstitial at best Voronoi site
if interstitial_sites:
    best_site = interstitial_sites[0]["frac_coords"]
    inter_O, inter_O_info = create_interstitial(
        bulk_sc, best_site, "O", "O_i"
    )
    print(f"\n  {inter_O_info['label']}: added O at {best_site}, "
          f"{inter_O_info['n_atoms']} atoms")
    inter_O.to(str(output_dir / "interstitial_O.cif"))
    Poscar(inter_O).write_file(str(output_dir / "POSCAR_O_i"))

# ============================================================
# 7. Relaxation with MACE and formation energy
# ============================================================
print("\n=== MACE Relaxation and Formation Energy ===")

from mace.calculators import mace_mp
from ase.optimize import BFGS

adaptor = AseAtomsAdaptor()
calc = mace_mp(model="medium", dispersion=False, default_dtype="float64")

# Relax bulk supercell
atoms_bulk = adaptor.get_atoms(bulk_sc)
atoms_bulk.calc = calc
opt = BFGS(atoms_bulk, logfile=str(output_dir / "bulk_relax.log"))
opt.run(fmax=0.005, steps=300)
e_bulk = atoms_bulk.get_potential_energy()
e_per_atom = e_bulk / len(atoms_bulk)
print(f"  Bulk: E = {e_bulk:.4f} eV ({len(atoms_bulk)} atoms), "
      f"E/atom = {e_per_atom:.4f} eV")

# Relax vacancy
atoms_vac = adaptor.get_atoms(vac_O)
atoms_vac.calc = calc
opt = BFGS(atoms_vac, logfile=str(output_dir / "vac_relax.log"))
opt.run(fmax=0.005, steps=500)
e_vac = atoms_vac.get_potential_energy()
print(f"  V_O:  E = {e_vac:.4f} eV ({len(atoms_vac)} atoms)")

# Formation energy (simple bulk reference)
e_form_vac = e_vac - e_bulk + e_per_atom
print(f"  E_form(V_O) = {e_form_vac:.4f} eV (bulk atom reference)")

# Relax substitution
atoms_sub = adaptor.get_atoms(sub_Al)
atoms_sub.calc = calc
opt = BFGS(atoms_sub, logfile=str(output_dir / "sub_relax.log"))
opt.run(fmax=0.005, steps=500)
e_sub = atoms_sub.get_potential_energy()
print(f"  Al_Mg: E = {e_sub:.4f} eV ({len(atoms_sub)} atoms)")

# ============================================================
# 8. Generate all defects systematically
# ============================================================
print("\n=== Systematic Defect Generation ===")

defect_catalog = []

# All vacancies
for element in ["Mg", "O"]:
    sites = get_inequivalent_sites(bulk_sc, element)
    for i, site in enumerate(sites):
        defect, info = create_vacancy(bulk_sc, site["index"])
        defect.to(str(output_dir / f"V_{element}_{i}.cif"))
        defect_catalog.append(info)
        print(f"  V_{element} (site {i}): {info['n_atoms']} atoms")

# Substitutions
substitutions = [
    ("Mg", "Al"), ("Mg", "Ca"), ("Mg", "Fe"),
    ("O", "N"), ("O", "S"),
]
for orig, new in substitutions:
    sites = get_inequivalent_sites(bulk_sc, orig)
    for i, site in enumerate(sites):
        defect, info = create_substitution(bulk_sc, site["index"], new)
        defect.to(str(output_dir / f"{new}_{orig}_{i}.cif"))
        defect_catalog.append(info)
        print(f"  {new}_{orig} (site {i}): {info['n_atoms']} atoms")

# Save catalog
with open(str(output_dir / "defect_catalog.json"), "w") as f:
    json.dump(defect_catalog, f, indent=2, default=str)
print(f"\nDefect catalog saved: {output_dir}/defect_catalog.json")
print(f"Total defects: {len(defect_catalog)}")
```

### Method B: QE DFT Defect Calculation

```python
#!/usr/bin/env python3
"""
Generate QE input files for defect calculations.
Produces both bulk and defect supercell inputs.
"""

import numpy as np
from pymatgen.core import Structure, Element, Lattice
from pymatgen.io.vasp import Poscar
from pathlib import Path

output_dir = Path("defect_qe")
output_dir.mkdir(exist_ok=True)

# Build host and supercell
host = Structure.from_spacegroup(
    "Fm-3m",
    lattice=Lattice.cubic(4.212),
    species=["Mg", "O"],
    coords=[[0.0, 0.0, 0.0], [0.5, 0.5, 0.5]],
)

sc = host.copy()
sc.make_supercell([3, 3, 3])

# Create vacancy
defect_sc = sc.copy()
o_indices = [i for i, s in enumerate(defect_sc) if s.specie == Element("O")]
defect_sc.remove_sites([o_indices[0]])

# ============================================================
# Write QE inputs
# ============================================================
PSEUDO_DIR = "./pseudo"

def write_qe_defect_input(structure, prefix, calculation="relax",
                            pseudo_dir=PSEUDO_DIR, filename=None):
    """Write QE input for bulk or defect supercell."""
    cell = structure.lattice.matrix
    elements = sorted(set(str(s.specie) for s in structure))

    cell_lines = "\n".join(
        f"  {v[0]:.10f}  {v[1]:.10f}  {v[2]:.10f}" for v in cell
    )

    species_lines = []
    for el in elements:
        from pymatgen.core.periodic_table import Element as PmgElem
        mass = PmgElem(el).atomic_mass
        species_lines.append(
            f"  {el:4s} {mass:10.4f}  {el}.pbe-n-kjpaw_psl.1.0.0.UPF"
        )

    pos_lines = []
    for site in structure:
        fc = site.frac_coords
        pos_lines.append(
            f"  {str(site.specie):4s} {fc[0]:.10f} {fc[1]:.10f} {fc[2]:.10f}"
        )

    lengths = structure.lattice.abc
    k_grid = [max(1, int(np.ceil(25.0 / l))) for l in lengths]

    ions_block = ""
    if calculation == "relax":
        ions_block = """
&IONS
    ion_dynamics = 'bfgs'
/
"""

    qe_input = f"""&CONTROL
    calculation = '{calculation}'
    prefix      = '{prefix}'
    outdir      = './tmp'
    pseudo_dir  = '{pseudo_dir}'
    tprnfor     = .true.
    tstress     = .true.
    etot_conv_thr = 1.0d-6
    forc_conv_thr = 1.0d-4
/
&SYSTEM
    ibrav       = 0
    nat         = {len(structure)}
    ntyp        = {len(elements)}
    ecutwfc     = 60.0
    ecutrho     = 480.0
    occupations = 'smearing'
    smearing    = 'cold'
    degauss     = 0.01
/
&ELECTRONS
    conv_thr    = 1.0d-8
    mixing_beta = 0.4
/
{ions_block}
CELL_PARAMETERS angstrom
{cell_lines}

ATOMIC_SPECIES
{chr(10).join(species_lines)}

ATOMIC_POSITIONS crystal
{chr(10).join(pos_lines)}

K_POINTS automatic
  {k_grid[0]} {k_grid[1]} {k_grid[2]}  0 0 0
"""

    outfile = filename or str(output_dir / f"{prefix}.in")
    with open(outfile, "w") as f:
        f.write(qe_input)
    print(f"Written: {outfile} ({len(structure)} atoms, k={k_grid})")

# Bulk supercell
write_qe_defect_input(sc, "bulk_sc", calculation="relax")

# Defect supercell
write_qe_defect_input(defect_sc, "defect_V_O", calculation="relax")

# ============================================================
# Run script
# ============================================================
run_script = f"""#!/bin/bash
NPROC=$(nproc)

echo "=== Bulk supercell ==="
cd {output_dir}
mpirun --allow-run-as-root -np $NPROC pw.x -in bulk_sc.in > bulk_sc.out 2>&1
echo "Bulk done: $(grep '!' bulk_sc.out | tail -1)"

echo "=== Defect supercell (V_O) ==="
mpirun --allow-run-as-root -np $NPROC pw.x -in defect_V_O.in > defect_V_O.out 2>&1
echo "Defect done: $(grep '!' defect_V_O.out | tail -1)"
"""

with open(str(output_dir / "run_defects.sh"), "w") as f:
    f.write(run_script)

print(f"\nRun script: bash {output_dir}/run_defects.sh")

# ============================================================
# Post-processing script
# ============================================================
post_script = '''#!/usr/bin/env python3
"""Parse QE defect outputs and compute formation energy."""
import re

def parse_qe_energy(filename):
    energy = None
    with open(filename) as f:
        for line in f:
            if "!" in line and "total energy" in line:
                match = re.search(r"=\\s+([-\\d.]+)\\s+Ry", line)
                if match:
                    energy = float(match.group(1)) * 13.605693123  # Ry -> eV
    return energy

e_bulk = parse_qe_energy("bulk_sc.out")
e_defect = parse_qe_energy("defect_V_O.out")

n_bulk = 216  # 3x3x3 MgO = 216 atoms
e_per_atom = e_bulk / n_bulk

e_form = e_defect - e_bulk + e_per_atom
print(f"E_bulk   = {e_bulk:.6f} eV ({n_bulk} atoms)")
print(f"E_defect = {e_defect:.6f} eV ({n_bulk - 1} atoms)")
print(f"E_form(V_O) = {e_form:.4f} eV")
'''

with open(str(output_dir / "parse_defects.py"), "w") as f:
    f.write(post_script)
print(f"Post-processing: python {output_dir}/parse_defects.py")
```

### Method C: VASP (Future External Access)

```python
#!/usr/bin/env python3
"""
Generate VASP input files for defect calculations.
Includes POSCAR, INCAR, KPOINTS for bulk and defect supercells.
"""

import numpy as np
from pymatgen.core import Structure, Element, Lattice
from pymatgen.io.vasp import Poscar, Incar, Kpoints

# Build host and supercell
host = Structure.from_spacegroup(
    "Fm-3m",
    lattice=Lattice.cubic(4.212),
    species=["Mg", "O"],
    coords=[[0.0, 0.0, 0.0], [0.5, 0.5, 0.5]],
)

sc = host.copy()
sc.make_supercell([3, 3, 3])

# Create defects
vac_O = sc.copy()
o_idx = [i for i, s in enumerate(vac_O) if s.specie == Element("O")]
vac_O.remove_sites([o_idx[0]])

sub_Al = sc.copy()
mg_idx = [i for i, s in enumerate(sub_Al) if s.specie == Element("Mg")]
sub_Al.replace(mg_idx[0], "Al")

# POSCAR files
Poscar(sc).write_file("POSCAR_bulk_sc")
Poscar(vac_O).write_file("POSCAR_V_O")
Poscar(sub_Al).write_file("POSCAR_Al_Mg")

# KPOINTS (small for supercell)
Kpoints.gamma_automatic(kpts=(2, 2, 2)).write_file("KPOINTS_defect")

# INCAR for defect relaxation
incar_defect = Incar({
    "SYSTEM": "Defect calculation",
    "ENCUT": 520,
    "PREC": "Accurate",
    "EDIFF": 1e-6,
    "EDIFFG": -0.01,
    "IBRION": 2,
    "NSW": 200,
    "ISIF": 2,             # Relax ions only (fixed cell for defect)
    "ISMEAR": 0,
    "SIGMA": 0.05,
    "LWAVE": True,         # Need for charged defect corrections
    "LCHARG": True,
    "LORBIT": 11,
    "LREAL": "Auto",       # Real-space projection for large cells
    "ALGO": "Normal",
})
incar_defect.write_file("INCAR_defect")

# For charged defects, add NELECT
n_electrons_neutral = sum(Element(str(s.specie)).Z for s in vac_O)
# V_O^{2+}: remove 2 electrons
incar_charged = incar_defect.copy()
# NELECT would be set based on the neutral electron count minus charge
print(f"Neutral V_O electron count: {n_electrons_neutral}")
print("For V_O^{2+}: NELECT = NELECT_neutral - 2")

print("\nVASP defect files written:")
print("  POSCAR_bulk_sc, POSCAR_V_O, POSCAR_Al_Mg")
print("  KPOINTS_defect, INCAR_defect")
print("\nWorkflow:")
print("  1. Relax bulk supercell (ISIF=3)")
print("  2. Create defect POSCAR from relaxed bulk")
print("  3. Relax defect (ISIF=2, fixed cell)")
print("  4. Compute formation energy from total energies")
```

### Visualize Defect Structure

```python
#!/usr/bin/env python3
"""Visualize defect structures with vacancy site marked."""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from pymatgen.core import Structure, Element

# Load defect structure
# defect = Structure.from_file("defect_models/vacancy_O.cif")
# For demo, build inline:
from pymatgen.core import Lattice
host = Structure.from_spacegroup(
    "Fm-3m", lattice=Lattice.cubic(4.212),
    species=["Mg", "O"], coords=[[0, 0, 0], [0.5, 0.5, 0.5]],
)
host.make_supercell([3, 3, 3])

# Record vacancy position
o_indices = [i for i, s in enumerate(host) if s.specie == Element("O")]
vac_cart = host[o_indices[13]].coords.copy()
host.remove_sites([o_indices[13]])

# Plot
fig = plt.figure(figsize=(8, 8))
ax = fig.add_subplot(111, projection="3d")

for sp, color, sz in [(Element("Mg"), "blue", 30), (Element("O"), "red", 30)]:
    coords = np.array([s.coords for s in host if s.specie == sp])
    if len(coords) > 0:
        ax.scatter(coords[:, 0], coords[:, 1], coords[:, 2],
                   c=color, s=sz, alpha=0.5, label=str(sp))

ax.scatter(*vac_cart, c="yellow", s=200, marker="X", edgecolors="black",
           linewidths=2, label="Vacancy", zorder=10)

ax.set_xlabel("x (A)")
ax.set_ylabel("y (A)")
ax.set_zlabel("z (A)")
ax.set_title("MgO with O vacancy")
ax.legend()
plt.tight_layout()
plt.savefig("defect_visualization.png", dpi=150)
print("Saved: defect_visualization.png")
```

---

## Key Parameters

| Parameter | Typical Value | Notes |
|---|---|---|
| Supercell size | >= 3x3x3 or min ~10 A between images | Converge formation energy vs. size |
| MACE fmax | 0.005 eV/A | Tight for accurate energies |
| QE ecutwfc | 50--80 Ry | System-dependent; converge for your pseudopotentials |
| QE k-points | Gamma-only or 2x2x2 | Scale inversely with supercell size |
| QE conv_thr | 1e-8 Ry | Tight for energy differences |
| ISIF (VASP) | 2 (ions only) | Do not relax cell shape for defect supercells |
| mixing_beta | 0.2--0.4 | Lower for difficult SCF convergence |
| Interstitial site tolerance | >= 1.0 A from nearest atom | Closer sites are unphysical |

## Common Issues

| Problem | Solution |
|---|---|
| SCF does not converge for defect | Reduce mixing_beta to 0.2. Try `mixing_mode = 'local-TF'` in QE. Increase ecutrho. |
| Formation energy not converging with supercell | Use larger supercell (4x4x4). For charged defects, apply Freysoldt correction. |
| Interstitial relaxes to wrong position | Try multiple starting sites. Use smaller fmax. Visualize before and after relaxation. |
| Substitution causes large distortion | Expected for large size mismatch. Use more relaxation steps. |
| MACE gives unreasonable defect energy | System may be outside MACE training data. Validate with QE DFT. |
| Wrong charge state for defect | For VASP: set NELECT. For QE: set tot_charge. Always apply finite-size corrections. |
| Defect-defect interaction in small supercell | Increase supercell size until formation energy converges (< 0.05 eV change). |
