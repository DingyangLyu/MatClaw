# Interatomic Force Constant Analysis

## When to Use

- You want to quantify bond strengths in a material using interatomic force constants (IFCs) from phonon calculations.
- You need to compare force constants across different materials or different bonds within the same material (e.g., strong covalent vs weak van der Waals bonds).
- You are analyzing phonon mode contributions to identify which atomic pairs dominate specific vibrational modes.
- You need to identify weak bonds that may be relevant for thermal expansion, phase transitions, or mechanical failure.
- You want to visualize how IFC strength decays with interatomic distance (checking supercell convergence).
- You need the normalized trace Tr(Phi_ij)/3 of IFC tensors for rotationally invariant bond strength comparison.
- You are studying bonding anisotropy through off-diagonal IFC tensor components.
- Equivalent to VASPKIT task 788 (Normalized Trace of Interatomic Force Constant Tensors).

## Method Selection

| Criterion | Phonopy + QE | Phonopy + VASP |
|---|---|---|
| Force engine | QE pw.x (pre-installed) | VASP (external license) |
| IFC generation | phonopy finite displacements + QE forces | phonopy finite displacements + VASP forces |
| Force constant file | phonopy FORCE_CONSTANTS or phonopy_params.yaml | Same (phonopy is the post-processor) |
| Analysis tool | Python (numpy, matplotlib) | Same |
| MACE alternative | phonopy + MACE via ASE (no DFT needed) | N/A |
| Best for | Production DFT-quality IFCs | Re-analyzing existing VASP phonon data |

## Prerequisites

- `pip install phonopy seekpath` (phonopy is not pre-installed).
- A completed phonopy calculation with force constants available (`FORCE_CONSTANTS` file or `phonopy_params.yaml`).
- Alternatively, raw displaced supercell calculations (QE or VASP) ready for phonopy to parse.
- Python packages: `numpy`, `scipy`, `matplotlib`, `pymatgen` (pre-installed).
- For QE: `pw.x` in `/opt/qe/bin/`.
- For VASP: completed displacement calculations with vasprun.xml files.

## Background

### Interatomic Force Constants

The interatomic force constant (IFC) tensor Phi_ij is a 3x3 matrix relating the force on atom i to the displacement of atom j:

```
F_i,alpha = -sum_j,beta  Phi_ij,alpha,beta * u_j,beta
```

where alpha, beta are Cartesian indices (x, y, z), F is force, and u is displacement. The IFC tensor encodes the strength and directionality of the bonding interaction between atoms i and j.

### Normalized Trace

The trace of the IFC tensor provides a rotationally invariant scalar measure of bond stiffness:

```
Tr(Phi_ij) = Phi_ij,xx + Phi_ij,yy + Phi_ij,zz
```

The normalized trace divides by the number of spatial dimensions:

```
Tr(Phi_ij) / 3
```

This gives the average force constant per degree of freedom, making it comparable across materials with different symmetries.

### Frobenius Norm

An alternative scalar measure is the Frobenius norm:

```
|Phi_ij| = sqrt(sum_{alpha,beta} Phi_ij,alpha,beta^2)
```

The ratio Tr(Phi_ij) / |Phi_ij| indicates the isotropy of the interaction:
- Close to sqrt(3) ~ 1.73: isotropic (all diagonal, equal).
- Close to 0: highly anisotropic (large off-diagonal or unequal diagonal).

### Typical IFC Values

| Material | Bond | Tr(Phi)/3 (eV/A^2) | Character |
|---|---|---|---|
| Diamond (C) | C-C (1st NN) | ~30 | Very strong covalent |
| Si | Si-Si (1st NN) | ~8-10 | Strong covalent |
| NaCl | Na-Cl (1st NN) | ~3-5 | Ionic |
| MoS2 | Mo-S (in-plane) | ~10-15 | Strong |
| MoS2 | S-S (interlayer) | ~0.1-0.5 | Weak vdW |
| Graphite | C-C (in-plane) | ~25 | Strong sp2 |
| Graphite | C-C (interlayer) | ~0.05 | Very weak vdW |

---

## Detailed Steps

### Overview

```
Method A: Phonopy + QE
  Step 1: Generate phonopy displaced supercells and run QE forces
  Step 2: Parse forces and produce force constants
  Step 3: Read FORCE_CONSTANTS and compute normalized trace
  Step 4: Plot bond strength map and compare with bond lengths
  Step 5: Analyze IFC tensor anisotropy

Method B: Phonopy + VASP
  Step 1: Parse VASP forces and produce force constants
  Step 2: Compute normalized traces (same analysis as Method A)
  Reference VASPKIT 788
```

---

### Method A: Phonopy + QE

#### Step A1: Generate Displaced Supercells and Run QE

```python
#!/usr/bin/env python3
"""
Generate phonopy displaced supercells for QE force calculations.
This step produces the FORCE_CONSTANTS needed for IFC analysis.

If you already have force constants (from a previous phonon calculation),
skip to Step A3.
"""
import os
import subprocess
import numpy as np
import phonopy
from pymatgen.core.structure import Structure, Lattice
from pymatgen.io.phonopy import get_phonopy_structure

# ============================================================
# CONFIGURATION
# ============================================================

STRUCTURE_FILE = "POSCAR"       # or .cif, .xyz
PSEUDO_DIR = os.path.abspath("./pseudo")
WORK_DIR = os.path.abspath("./phonopy_ifc")
MIN_LENGTH = 20.0               # Minimum supercell dimension (Angstrom)
DISPLACEMENT = 0.01             # Displacement magnitude (Angstrom)
SYMPREC = 1e-5
ECUTWFC = 60.0                  # Ry
ECUTRHO = 480.0                 # Ry
KPPA = 2000                     # K-point density per atom
NPROC = 4                       # MPI processes

os.makedirs(WORK_DIR, exist_ok=True)
os.makedirs(PSEUDO_DIR, exist_ok=True)

# ============================================================
# LOAD STRUCTURE AND SETUP PHONOPY
# ============================================================

structure = Structure.from_file(STRUCTURE_FILE)
phonopy_structure = get_phonopy_structure(structure)

cell_lengths = np.array(structure.lattice.abc)
multiples = np.maximum(np.ceil(MIN_LENGTH / cell_lengths).astype(int), 1)
supercell_matrix = np.diag(multiples)

phonon = phonopy.Phonopy(
    phonopy_structure,
    supercell_matrix=supercell_matrix.tolist(),
    symprec=SYMPREC,
)
phonon.generate_displacements(distance=DISPLACEMENT)
supercells = phonon.supercells_with_displacements

print(f"Structure: {structure.composition.reduced_formula}")
print(f"Supercell: {multiples[0]}x{multiples[1]}x{multiples[2]}")
print(f"Supercell atoms: {len(phonon.supercell.symbols)}")
print(f"Displacements: {len(supercells)}")

# Save phonopy setup
phonon.save(os.path.join(WORK_DIR, "phonopy_disp.yaml"))

# ============================================================
# GENERATE QE INPUTS FOR EACH DISPLACEMENT
# ============================================================

unique_species = sorted(set(str(s) for s in structure.species))

for i_disp, sc in enumerate(supercells):
    disp_dir = os.path.join(WORK_DIR, f"disp-{i_disp+1:03d}")
    os.makedirs(disp_dir, exist_ok=True)

    sc_structure = Structure(
        lattice=Lattice(sc.cell),
        species=sc.symbols,
        coords=sc.scaled_positions,
    )

    # Auto k-mesh
    from pymatgen.io.vasp import Kpoints
    kpts = Kpoints.automatic_density(sc_structure, kppa=KPPA,
                                     force_gamma=True)
    kx, ky, kz = kpts.kpts[0]

    # Atomic positions block
    atom_lines = []
    for site in sc_structure:
        fc = site.frac_coords
        atom_lines.append(f"  {str(site.specie):>4s}  "
                          f"{fc[0]:14.10f}  {fc[1]:14.10f}  {fc[2]:14.10f}")

    # Atomic species block
    species_lines = []
    for sp in unique_species:
        from pymatgen.core import Element
        mass = Element(sp).atomic_mass
        pseudo = f"{sp}.pbe-n-kjpaw_psl.1.0.0.UPF"
        species_lines.append(f"  {sp}  {float(mass):.4f}  {pseudo}")

    # Cell parameters
    cell = sc_structure.lattice.matrix

    qe_input = f"""&CONTROL
    calculation   = 'scf'
    prefix        = 'phonopy_disp'
    pseudo_dir    = '{PSEUDO_DIR}'
    outdir        = '{disp_dir}/tmp'
    tprnfor       = .true.
    tstress       = .true.
/

&SYSTEM
    ibrav         = 0
    nat           = {len(sc_structure)}
    ntyp          = {len(unique_species)}
    ecutwfc       = {ECUTWFC}
    ecutrho       = {ECUTRHO}
    occupations   = 'smearing'
    smearing      = 'gaussian'
    degauss       = 0.005
/

&ELECTRONS
    conv_thr      = 1.0d-10
    mixing_beta   = 0.7
/

ATOMIC_SPECIES
{chr(10).join(species_lines)}

CELL_PARAMETERS {{angstrom}}
  {cell[0,0]:14.10f}  {cell[0,1]:14.10f}  {cell[0,2]:14.10f}
  {cell[1,0]:14.10f}  {cell[1,1]:14.10f}  {cell[1,2]:14.10f}
  {cell[2,0]:14.10f}  {cell[2,1]:14.10f}  {cell[2,2]:14.10f}

ATOMIC_POSITIONS {{crystal}}
{chr(10).join(atom_lines)}

K_POINTS {{automatic}}
  {kx} {ky} {kz}  0 0 0
"""

    input_file = os.path.join(disp_dir, "scf.in")
    with open(input_file, 'w') as f:
        f.write(qe_input)

    if (i_disp + 1) % 5 == 0 or (i_disp + 1) == len(supercells):
        print(f"  Written QE input: disp-{i_disp+1:03d}/scf.in")

print(f"\nAll QE inputs written to: {WORK_DIR}/disp-XXX/")
print(f"Run each with: mpirun -np {NPROC} pw.x -in scf.in > scf.out")

# ============================================================
# RUN ALL DISPLACEMENTS
# ============================================================

print("\nRunning QE force calculations ...")
for i_disp in range(len(supercells)):
    disp_dir = os.path.join(WORK_DIR, f"disp-{i_disp+1:03d}")
    input_file = os.path.join(disp_dir, "scf.in")
    output_file = os.path.join(disp_dir, "scf.out")

    if os.path.exists(output_file):
        print(f"  disp-{i_disp+1:03d}: already computed, skipping")
        continue

    os.makedirs(os.path.join(disp_dir, "tmp"), exist_ok=True)
    result = subprocess.run(
        ["mpirun", "-np", str(NPROC), "pw.x", "-in", "scf.in"],
        capture_output=True, text=True, timeout=7200,
        cwd=disp_dir,
    )

    with open(output_file, 'w') as f:
        f.write(result.stdout)

    if result.returncode == 0:
        print(f"  disp-{i_disp+1:03d}: completed")
    else:
        print(f"  disp-{i_disp+1:03d}: FAILED")
        print(f"    {result.stderr[-200:] if result.stderr else ''}")

print("Done.")
```

#### Step A2: Parse Forces and Produce Force Constants

```python
#!/usr/bin/env python3
"""
Parse QE forces from displacement calculations and produce phonopy
force constants. Saves FORCE_CONSTANTS and phonopy_params.yaml.
"""
import os
import re
import numpy as np
import phonopy

WORK_DIR = os.path.abspath("./phonopy_ifc")

# ============================================================
# LOAD PHONOPY DISPLACEMENT DATA
# ============================================================

phonon = phonopy.load(
    os.path.join(WORK_DIR, "phonopy_disp.yaml"),
    produce_fc=False,
)
n_disp = len(phonon.supercells_with_displacements)
n_atoms_super = len(phonon.supercell.symbols)

print(f"Loaded phonopy with {n_disp} displacements")
print(f"Supercell atoms: {n_atoms_super}")


def parse_qe_forces(filename):
    """
    Parse forces from QE pw.x output.

    Returns
    -------
    forces : ndarray (n_atoms, 3) in eV/Angstrom
    """
    RY_TO_EV = 13.605693122994
    BOHR_TO_ANG = 0.529177210903

    with open(filename, 'r') as f:
        text = f.read()

    # Find "Forces acting on atoms" section
    force_block = re.search(
        r'Forces acting on atoms.*?\n(.*?)(?:Total force|Writing)',
        text, re.DOTALL)

    if force_block is None:
        raise ValueError(f"No forces found in {filename}")

    forces = []
    for line in force_block.group(1).strip().split('\n'):
        m = re.search(
            r'force\s*=\s*([-\d.]+)\s+([-\d.]+)\s+([-\d.]+)', line)
        if m:
            # Forces in QE output are in Ry/bohr; convert to eV/Angstrom
            fx = float(m.group(1)) * RY_TO_EV / BOHR_TO_ANG
            fy = float(m.group(2)) * RY_TO_EV / BOHR_TO_ANG
            fz = float(m.group(3)) * RY_TO_EV / BOHR_TO_ANG
            forces.append([fx, fy, fz])

    return np.array(forces)


# ============================================================
# PARSE ALL FORCES
# ============================================================

forces_list = []
for i_disp in range(1, n_disp + 1):
    output_file = os.path.join(WORK_DIR, f"disp-{i_disp:03d}", "scf.out")

    if not os.path.exists(output_file):
        print(f"  WARNING: {output_file} not found!")
        forces_list.append(np.zeros((n_atoms_super, 3)))
        continue

    try:
        forces = parse_qe_forces(output_file)
        forces_list.append(forces)
        if i_disp % 5 == 0 or i_disp == n_disp:
            print(f"  Parsed forces: disp-{i_disp:03d} "
                  f"(max |F| = {np.max(np.abs(forces)):.4f} eV/A)")
    except Exception as e:
        print(f"  ERROR parsing disp-{i_disp:03d}: {e}")
        forces_list.append(np.zeros((n_atoms_super, 3)))

# ============================================================
# PRODUCE FORCE CONSTANTS
# ============================================================

phonon.forces = forces_list
phonon.produce_force_constants()

# Save in multiple formats
phonon.save(os.path.join(WORK_DIR, "phonopy_params.yaml"))
print(f"Saved: {WORK_DIR}/phonopy_params.yaml")

# Write FORCE_CONSTANTS file (readable format)
from phonopy.file_IO import write_FORCE_CONSTANTS
write_FORCE_CONSTANTS(phonon.force_constants,
                      filename=os.path.join(WORK_DIR, "FORCE_CONSTANTS"))
print(f"Saved: {WORK_DIR}/FORCE_CONSTANTS")

fc = phonon.force_constants
print(f"\nForce constants shape: {fc.shape}")
print(f"  (n_atoms_prim, n_atoms_super, 3, 3)")
```

#### Step A3: Compute Normalized Trace of IFC Tensors

```python
#!/usr/bin/env python3
"""
Compute the normalized trace Tr(Phi_ij)/3 for all atom pairs,
analyze bond strengths, and compare with bond lengths.

The normalized trace is a rotationally invariant scalar that
quantifies the average stiffness of the interaction between
atoms i and j.
"""
import os
import json
import numpy as np
import phonopy
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from collections import defaultdict

WORK_DIR = os.path.abspath("./phonopy_ifc")

# ============================================================
# LOAD FORCE CONSTANTS
# ============================================================

phonon = phonopy.load(os.path.join(WORK_DIR, "phonopy_params.yaml"))
fc = phonon.force_constants  # shape: (n_prim, n_super, 3, 3)
supercell = phonon.supercell
unitcell = phonon.unitcell

n_prim = len(unitcell.symbols)
n_super = len(supercell.symbols)

print(f"Force constants shape: {fc.shape}")
print(f"Primitive cell atoms: {n_prim} ({', '.join(unitcell.symbols)})")
print(f"Supercell atoms: {n_super}")

# ============================================================
# COMPUTE NORMALIZED TRACE FOR ALL PAIRS
# ============================================================

print("\n" + "=" * 70)
print("Normalized Trace of IFC Tensors: Tr(Phi_ij) / 3")
print("=" * 70)

ifc_data = []

for i in range(n_prim):
    pos_i_frac = unitcell.scaled_positions[i]
    cart_i = unitcell.cell @ pos_i_frac
    sym_i = unitcell.symbols[i]

    for j in range(n_super):
        phi = fc[i, j]  # 3x3 tensor (eV/A^2)

        # Scalar measures
        trace = np.trace(phi)
        frob_norm = np.linalg.norm(phi, 'fro')
        norm_trace_3 = trace / 3.0  # Normalized trace (divide by 3)

        if frob_norm < 1e-10:
            continue  # Skip zero interactions

        norm_trace_frob = trace / frob_norm  # Trace / Frobenius norm

        # Distance between atoms
        pos_j_frac = supercell.scaled_positions[j]
        cart_j = supercell.cell @ pos_j_frac
        dist = np.linalg.norm(cart_j - cart_i)

        sym_j = supercell.symbols[j]

        # Eigenvalues of the IFC tensor (principal stiffnesses)
        eigvals = np.linalg.eigvalsh(phi)

        ifc_data.append({
            'i': i, 'j': j,
            'sym_i': sym_i, 'sym_j': sym_j,
            'distance': float(dist),
            'trace': float(trace),
            'norm_trace_3': float(norm_trace_3),
            'frob_norm': float(frob_norm),
            'norm_trace_frob': float(norm_trace_frob),
            'eigvals': eigvals.tolist(),
            'tensor': phi.tolist(),
        })

# Sort by distance
ifc_data.sort(key=lambda x: x['distance'])

# ============================================================
# PRINT SUMMARY TABLE
# ============================================================

print(f"\n{'Pair':>10s}  {'Dist (A)':>9s}  {'Tr/3':>10s}  "
      f"{'|Phi|':>10s}  {'Tr/|Phi|':>9s}  {'Aniso':>7s}")
print("-" * 70)

seen_pairs = set()
for d in ifc_data:
    if d['distance'] < 0.1:
        continue  # Skip self-interaction
    pair_key = (d['sym_i'], d['sym_j'], round(d['distance'], 2))
    if pair_key in seen_pairs:
        continue
    seen_pairs.add(pair_key)

    # Anisotropy: ratio of max to min absolute eigenvalue
    eigvals = np.array(d['eigvals'])
    abs_eigvals = np.abs(eigvals)
    if min(abs_eigvals) > 1e-10:
        aniso = max(abs_eigvals) / min(abs_eigvals)
    else:
        aniso = float('inf')

    print(f"{d['sym_i']}-{d['sym_j']:>4s}  {d['distance']:9.4f}  "
          f"{d['norm_trace_3']:10.4f}  {d['frob_norm']:10.4f}  "
          f"{d['norm_trace_frob']:9.4f}  {aniso:7.2f}")

    if len(seen_pairs) >= 30:
        print("  ... (showing first 30 unique pairs)")
        break

# ============================================================
# GROUP BY BOND TYPE
# ============================================================

print("\n" + "=" * 70)
print("Bond Type Summary (grouped by element pair and neighbor shell)")
print("=" * 70)

bond_groups = defaultdict(list)
for d in ifc_data:
    if d['distance'] < 0.1:
        continue
    pair = tuple(sorted([d['sym_i'], d['sym_j']]))
    shell = round(d['distance'], 1)
    bond_groups[(pair, shell)].append(d)

print(f"\n{'Bond':>10s}  {'Shell (A)':>10s}  {'Count':>6s}  "
      f"{'Avg Tr/3':>10s}  {'Avg |Phi|':>10s}  {'Strength':>12s}")
print("-" * 70)

for (pair, shell), group in sorted(bond_groups.items(),
                                     key=lambda x: x[0][1]):
    avg_nt3 = np.mean([d['norm_trace_3'] for d in group])
    avg_frob = np.mean([d['frob_norm'] for d in group])

    if abs(avg_nt3) > 5:
        strength = "very strong"
    elif abs(avg_nt3) > 1:
        strength = "strong"
    elif abs(avg_nt3) > 0.1:
        strength = "moderate"
    elif abs(avg_nt3) > 0.01:
        strength = "weak"
    else:
        strength = "negligible"

    pair_str = f"{pair[0]}-{pair[1]}"
    print(f"{pair_str:>10s}  {shell:10.1f}  {len(group):6d}  "
          f"{avg_nt3:10.4f}  {avg_frob:10.4f}  {strength:>12s}")

# ============================================================
# SAVE RESULTS
# ============================================================

output = {
    'material': ', '.join(unitcell.symbols),
    'n_prim': n_prim,
    'n_super': n_super,
    'pairs': [{
        'sym_i': d['sym_i'], 'sym_j': d['sym_j'],
        'distance': d['distance'],
        'norm_trace_3': d['norm_trace_3'],
        'frob_norm': d['frob_norm'],
        'norm_trace_frob': d['norm_trace_frob'],
        'eigvals': d['eigvals'],
    } for d in ifc_data if d['distance'] > 0.1],
}

with open(os.path.join(WORK_DIR, 'ifc_analysis.json'), 'w') as f:
    json.dump(output, f, indent=2)
print(f"\nSaved: {WORK_DIR}/ifc_analysis.json")
```

#### Step A4: Plot Bond Strength Map

```python
#!/usr/bin/env python3
"""
Plot IFC bond strength analysis:
1. IFC magnitude vs distance (decay plot)
2. Normalized trace vs distance
3. Bond strength comparison bar chart
4. IFC tensor eigenvalue spectra
"""
import os
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from collections import defaultdict

WORK_DIR = os.path.abspath("./phonopy_ifc")

# ============================================================
# LOAD DATA
# ============================================================

with open(os.path.join(WORK_DIR, 'ifc_analysis.json'), 'r') as f:
    data = json.load(f)

pairs = data['pairs']
distances = np.array([p['distance'] for p in pairs])
norm_trace = np.array([p['norm_trace_3'] for p in pairs])
frob_norms = np.array([p['frob_norm'] for p in pairs])
norm_trace_frob = np.array([p['norm_trace_frob'] for p in pairs])

# Unique bond types
bond_types = defaultdict(list)
for p in pairs:
    bt = tuple(sorted([p['sym_i'], p['sym_j']]))
    bond_types[bt].append(p)

# Color map for bond types
unique_bonds = sorted(bond_types.keys())
colors = plt.cm.Set1(np.linspace(0, 1, max(len(unique_bonds), 3)))

# ============================================================
# PLOT 1: IFC Magnitude vs Distance (Decay Plot)
# ============================================================

fig, axes = plt.subplots(2, 2, figsize=(14, 12))

ax = axes[0, 0]
for i, bt in enumerate(unique_bonds):
    pts = bond_types[bt]
    d = [p['distance'] for p in pts]
    fn = [p['frob_norm'] for p in pts]
    label = f"{bt[0]}-{bt[1]}"
    ax.scatter(d, fn, s=15, alpha=0.6, color=colors[i], label=label)

ax.set_xlabel("Distance (A)", fontsize=11)
ax.set_ylabel("|Phi| (eV/A^2)", fontsize=11)
ax.set_title("IFC Frobenius Norm vs Distance", fontsize=12)
ax.set_yscale("log")
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)

# ============================================================
# PLOT 2: Normalized Trace / 3 vs Distance
# ============================================================

ax = axes[0, 1]
for i, bt in enumerate(unique_bonds):
    pts = bond_types[bt]
    d = [p['distance'] for p in pts]
    nt = [abs(p['norm_trace_3']) for p in pts]
    label = f"{bt[0]}-{bt[1]}"
    ax.scatter(d, nt, s=15, alpha=0.6, color=colors[i], label=label)

ax.set_xlabel("Distance (A)", fontsize=11)
ax.set_ylabel("|Tr(Phi)/3| (eV/A^2)", fontsize=11)
ax.set_title("Normalized Trace vs Distance", fontsize=12)
ax.set_yscale("log")
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)

# ============================================================
# PLOT 3: Bond Strength Bar Chart (1st neighbor shell)
# ============================================================

ax = axes[1, 0]

bar_data = []
for bt in unique_bonds:
    pts = bond_types[bt]
    # Find first neighbor shell (shortest distance)
    dists = sorted(set(round(p['distance'], 1) for p in pts))
    if len(dists) == 0:
        continue
    first_shell = dists[0]
    shell_pts = [p for p in pts
                 if abs(p['distance'] - first_shell) < 0.2]
    avg_nt3 = np.mean([abs(p['norm_trace_3']) for p in shell_pts])
    bar_data.append({
        'label': f"{bt[0]}-{bt[1]}\n({first_shell:.1f} A)",
        'value': avg_nt3,
    })

bar_data.sort(key=lambda x: x['value'], reverse=True)
x_pos = np.arange(len(bar_data))
bar_colors = plt.cm.RdYlGn_r(
    np.linspace(0.2, 0.8, len(bar_data)))

ax.bar(x_pos, [b['value'] for b in bar_data],
       color=bar_colors, edgecolor='black', linewidth=0.5)
ax.set_xticks(x_pos)
ax.set_xticklabels([b['label'] for b in bar_data], fontsize=9)
ax.set_ylabel("|Tr(Phi)/3| (eV/A^2)", fontsize=11)
ax.set_title("Bond Strength (1st Neighbor Shell)", fontsize=12)
ax.grid(True, alpha=0.3, axis='y')

# ============================================================
# PLOT 4: IFC Tensor Eigenvalue Spectrum
# ============================================================

ax = axes[1, 1]

for i, bt in enumerate(unique_bonds):
    pts = bond_types[bt]
    # First neighbor shell
    dists = sorted(set(round(p['distance'], 1) for p in pts))
    if len(dists) == 0:
        continue
    first_shell = dists[0]
    shell_pts = [p for p in pts
                 if abs(p['distance'] - first_shell) < 0.2]

    all_eigvals = []
    for p in shell_pts:
        all_eigvals.extend(p['eigvals'])

    label = f"{bt[0]}-{bt[1]} ({first_shell:.1f} A)"
    ax.hist(all_eigvals, bins=30, alpha=0.5, color=colors[i],
            label=label, density=True)

ax.set_xlabel("IFC Eigenvalue (eV/A^2)", fontsize=11)
ax.set_ylabel("Density", fontsize=11)
ax.set_title("IFC Tensor Eigenvalue Distribution (1st NN)", fontsize=12)
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)
ax.axvline(x=0, color='gray', linestyle='--', linewidth=0.5)

fig.suptitle("Interatomic Force Constant Analysis", fontsize=14, y=1.02)
fig.tight_layout()
fig.savefig(os.path.join(WORK_DIR, "ifc_bond_strength.png"),
            dpi=150, bbox_inches='tight')
plt.close()
print(f"Saved: {WORK_DIR}/ifc_bond_strength.png")
```

#### Step A5: Compare IFCs with Bond Lengths

```python
#!/usr/bin/env python3
"""
Correlate IFC normalized trace with bond lengths.
Strong bonds typically have shorter lengths and larger IFCs.
This analysis helps identify anomalous bonds (strong but long,
or weak but short).
"""
import os
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from collections import defaultdict
from scipy.stats import pearsonr

WORK_DIR = os.path.abspath("./phonopy_ifc")

# ============================================================
# LOAD DATA
# ============================================================

with open(os.path.join(WORK_DIR, 'ifc_analysis.json'), 'r') as f:
    data = json.load(f)

pairs = data['pairs']

# Group by bond type and neighbor shell
bond_shells = defaultdict(list)
for p in pairs:
    bt = tuple(sorted([p['sym_i'], p['sym_j']]))
    shell = round(p['distance'], 1)
    bond_shells[(bt, shell)].append(p)

# ============================================================
# BOND LENGTH vs IFC CORRELATION
# ============================================================

print("=" * 65)
print("Bond Length vs IFC Correlation")
print("=" * 65)

shell_distances = []
shell_ifcs = []
shell_labels = []

for (bt, shell), pts in sorted(bond_shells.items(),
                                  key=lambda x: x[0][1]):
    if len(pts) < 1:
        continue

    avg_dist = np.mean([p['distance'] for p in pts])
    avg_ifc = np.mean([abs(p['norm_trace_3']) for p in pts])

    if avg_ifc < 1e-4:
        continue  # Skip negligible interactions

    shell_distances.append(avg_dist)
    shell_ifcs.append(avg_ifc)
    label = f"{bt[0]}-{bt[1]} ({shell:.1f})"
    shell_labels.append(label)

    print(f"  {label:>20s}: d = {avg_dist:.3f} A, "
          f"|Tr/3| = {avg_ifc:.4f} eV/A^2")

shell_distances = np.array(shell_distances)
shell_ifcs = np.array(shell_ifcs)

# Compute correlation
if len(shell_distances) > 2:
    r_corr, p_val = pearsonr(shell_distances, np.log10(shell_ifcs))
    print(f"\n  Pearson r (d vs log10|Tr/3|): {r_corr:.4f} "
          f"(p = {p_val:.2e})")
    print(f"  {'Strong negative' if r_corr < -0.7 else 'Moderate' if r_corr < -0.3 else 'Weak'}"
          f" anticorrelation (expected: shorter bonds = stronger)")

# ============================================================
# PLOT: Bond Length vs IFC
# ============================================================

fig, ax = plt.subplots(1, 1, figsize=(8, 6))

ax.scatter(shell_distances, shell_ifcs, s=60, c='steelblue',
           edgecolors='black', linewidth=0.5, zorder=5)

# Label each point
for i, label in enumerate(shell_labels):
    ax.annotate(label, (shell_distances[i], shell_ifcs[i]),
                fontsize=8, ha='left', va='bottom',
                xytext=(5, 5), textcoords='offset points')

# Fit exponential decay: IFC ~ A * exp(-B * d)
if len(shell_distances) > 2:
    log_ifcs = np.log(shell_ifcs)
    valid = np.isfinite(log_ifcs)
    if np.sum(valid) > 2:
        coeffs = np.polyfit(shell_distances[valid], log_ifcs[valid], 1)
        d_fit = np.linspace(min(shell_distances) * 0.9,
                            max(shell_distances) * 1.1, 100)
        ifc_fit = np.exp(np.polyval(coeffs, d_fit))
        ax.plot(d_fit, ifc_fit, 'r--', linewidth=1.5,
                label=f'Exponential fit (decay rate = {abs(coeffs[0]):.2f} /A)')
        ax.legend(fontsize=10)

ax.set_xlabel("Bond Length (A)", fontsize=12)
ax.set_ylabel("|Tr(Phi)/3| (eV/A^2)", fontsize=12)
ax.set_title("IFC Strength vs Bond Length", fontsize=13)
ax.set_yscale("log")
ax.grid(True, alpha=0.3)

fig.tight_layout()
fig.savefig(os.path.join(WORK_DIR, "ifc_vs_bond_length.png"),
            dpi=150, bbox_inches='tight')
plt.close()
print(f"\nSaved: {WORK_DIR}/ifc_vs_bond_length.png")

# ============================================================
# IDENTIFY ANOMALOUS BONDS
# ============================================================

print("\n" + "=" * 65)
print("Anomalous Bonds (deviating from exponential trend)")
print("=" * 65)

if len(shell_distances) > 2 and np.sum(valid) > 2:
    predicted_log_ifc = np.polyval(coeffs, shell_distances)
    residuals = np.log(shell_ifcs) - predicted_log_ifc
    std_res = np.std(residuals)

    for i in range(len(shell_labels)):
        if abs(residuals[i]) > 1.5 * std_res:
            direction = "stronger" if residuals[i] > 0 else "weaker"
            ratio = np.exp(abs(residuals[i]))
            print(f"  {shell_labels[i]:>20s}: {ratio:.1f}x {direction} "
                  f"than expected for d = {shell_distances[i]:.2f} A")
else:
    print("  Not enough data for anomaly detection.")
```

---

### Method B: Phonopy + VASP

#### Step B1: Parse VASP Forces and Compute IFCs

```python
#!/usr/bin/env python3
"""
Parse VASP vasprun.xml forces from phonopy displacement calculations
and produce force constants for IFC analysis.

After running VASP on each phonopy-displaced POSCAR:
  phonopy --vasp -f disp-*/vasprun.xml

Or use this script for more control over the parsing.
Equivalent to VASPKIT task 788 post-processing.
"""
import os
import numpy as np
import phonopy
from pymatgen.io.vasp import Vasprun

# ============================================================
# CONFIGURATION
# ============================================================

WORK_DIR = "./phonopy_vasp"     # Directory with phonopy_disp.yaml and disp-XXX/
OUTPUT_DIR = "./ifc_vasp"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ============================================================
# LOAD PHONOPY AND PARSE VASP FORCES
# ============================================================

disp_yaml = os.path.join(WORK_DIR, "phonopy_disp.yaml")
if not os.path.exists(disp_yaml):
    print(f"ERROR: {disp_yaml} not found.")
    print("Generate it first with phonopy or the QE workflow (Method A Step 1).")
    raise FileNotFoundError(disp_yaml)

phonon = phonopy.load(disp_yaml, produce_fc=False)
n_disp = len(phonon.supercells_with_displacements)

print(f"Loading VASP forces for {n_disp} displacements ...")

forces_list = []
for i in range(1, n_disp + 1):
    vasprun_path = os.path.join(WORK_DIR, f"disp-{i:03d}", "vasprun.xml")

    if not os.path.exists(vasprun_path):
        print(f"  WARNING: {vasprun_path} not found!")
        continue

    vr = Vasprun(vasprun_path, parse_dos=False, parse_eigen=False)
    forces = np.array(vr.ionic_steps[-1]["forces"])
    forces_list.append(forces)

    if i % 5 == 0 or i == n_disp:
        print(f"  Parsed: {i}/{n_disp}")

# ============================================================
# PRODUCE FORCE CONSTANTS
# ============================================================

phonon.forces = forces_list
phonon.produce_force_constants()

phonon.save(os.path.join(OUTPUT_DIR, "phonopy_params.yaml"))
print(f"\nSaved: {OUTPUT_DIR}/phonopy_params.yaml")

from phonopy.file_IO import write_FORCE_CONSTANTS
write_FORCE_CONSTANTS(phonon.force_constants,
                      filename=os.path.join(OUTPUT_DIR, "FORCE_CONSTANTS"))
print(f"Saved: {OUTPUT_DIR}/FORCE_CONSTANTS")

# ============================================================
# RUN IFC ANALYSIS (same as Method A Step A3)
# ============================================================

fc = phonon.force_constants
supercell = phonon.supercell
unitcell = phonon.unitcell
n_prim = len(unitcell.symbols)
n_super = len(supercell.symbols)

print(f"\nForce constants shape: {fc.shape}")
print(f"Primitive atoms: {n_prim}")
print(f"Supercell atoms: {n_super}")

print(f"\n{'Pair':>10s}  {'Dist (A)':>9s}  {'Tr/3':>10s}  "
      f"{'|Phi|':>10s}  {'Strength':>12s}")
print("-" * 60)

seen_pairs = set()
for i in range(n_prim):
    cart_i = unitcell.cell @ unitcell.scaled_positions[i]
    sym_i = unitcell.symbols[i]

    for j in range(n_super):
        phi = fc[i, j]
        frob_norm = np.linalg.norm(phi, 'fro')
        if frob_norm < 1e-10:
            continue

        cart_j = supercell.cell @ supercell.scaled_positions[j]
        dist = np.linalg.norm(cart_j - cart_i)
        if dist < 0.1:
            continue

        trace = np.trace(phi)
        nt3 = trace / 3.0
        sym_j = supercell.symbols[j]

        pair_key = (sym_i, sym_j, round(dist, 2))
        if pair_key in seen_pairs:
            continue
        seen_pairs.add(pair_key)

        if abs(nt3) > 5:
            strength = "very strong"
        elif abs(nt3) > 1:
            strength = "strong"
        elif abs(nt3) > 0.1:
            strength = "moderate"
        elif abs(nt3) > 0.01:
            strength = "weak"
        else:
            strength = "negligible"

        print(f"{sym_i}-{sym_j:>4s}  {dist:9.4f}  {nt3:10.4f}  "
              f"{frob_norm:10.4f}  {strength:>12s}")

        if len(seen_pairs) >= 30:
            break
    if len(seen_pairs) >= 30:
        print("  ... (showing first 30 unique pairs)")
        break

print(f"\nFor full analysis plots, use the plotting scripts from Method A")
print(f"with data loaded from: {OUTPUT_DIR}/phonopy_params.yaml")
print(f"\nVASPKIT equivalent: vaspkit -task 788")
print("VASPKIT 788 computes the normalized trace directly from VASP data.")
```

---

## Key Parameters

| Parameter | Typical Value | Notes |
|---|---|---|
| displacement | 0.01 A | Standard for phonopy. Use 0.005 for noisy potentials, 0.03 for quick testing. |
| min_length | 20.0 A | Minimum supercell dimension. IFCs must decay to near zero at the supercell boundary. |
| symprec | 1e-5 | Symmetry tolerance. Tighter = more displacements but more accurate symmetry. |
| conv_thr (QE) | 1e-10 Ry | Tight SCF convergence for accurate forces. |
| EDIFF (VASP) | 1e-8 eV | Same purpose. |
| ecutwfc (QE) | 50-80 Ry | Must be converged for force accuracy. |
| ENCUT (VASP) | 400-600 eV | Same. |
| KPPA | 1000-4000 | K-point density per atom for supercell SCF. Auto-scales with cell size. |
| fmax for relaxation | 1e-4 eV/A | Structure must be well-relaxed before computing IFCs. |

### IFC Analysis Thresholds

| |Tr(Phi)/3| Range (eV/A^2) | Interpretation | Typical Bonds |
|---|---|---|
| > 10 | Very strong covalent | C-C (diamond), N-N (triple), Si-O |
| 1 -- 10 | Strong | Si-Si, Ge-Ge, metal-O (oxides) |
| 0.1 -- 1 | Moderate | Ionic bonds, hydrogen bonds (strong) |
| 0.01 -- 0.1 | Weak | Weak hydrogen bonds, halide interactions |
| < 0.01 | Negligible / vdW | Interlayer (graphite, MoS2), noble gas |

## Interpreting Results

### Normalized Trace

- **Large |Tr(Phi)/3|**: Strong bonding interaction. The atomic pair resists displacement in all directions.
- **Small |Tr(Phi)/3| with large |Phi|**: The bond is stiff in some directions but not others (anisotropic). Common for layered materials where in-plane bonding is strong but interlayer is weak.
- **Negative Tr(Phi)/3 for self-interaction (i=j)**: The diagonal IFC (self-term) is always negative due to the acoustic sum rule: Phi_ii = -sum_{j!=i} Phi_ij. Its magnitude reflects the total bonding environment.
- **Sign of off-diagonal Tr**: Positive trace for the off-diagonal (i!=j) interaction means attractive restoring forces; negative means the interaction opposes the displacement.

### IFC Decay with Distance

- **Exponential decay**: IFCs should decay roughly exponentially with distance for well-converged supercells. If they remain significant at the supercell boundary, increase the supercell size.
- **Oscillatory decay in metals**: Long-range Friedel oscillations can cause IFCs to oscillate in sign with distance. This is physical but requires larger supercells.
- **Boundary artifacts**: If the IFC magnitude at the largest distance is >1% of the nearest-neighbor IFC, the supercell is too small.

### Tensor Eigenvalue Analysis

- **All three eigenvalues similar**: Isotropic interaction (e.g., NaCl ionic bond).
- **One eigenvalue much larger**: Strongly directional bond (e.g., stretching vs bending stiffness in covalent bonds).
- **One eigenvalue near zero**: Bond has a soft direction (relevant for phase transitions, displacive instabilities).

### Bond Strength Correlation

- **Strong anticorrelation (r < -0.7)**: Normal behavior -- shorter bonds are stronger.
- **Outliers above the trend**: Bonds that are stronger than expected for their length. May indicate unusual bonding (double/triple bonds, strong hybridization).
- **Outliers below the trend**: Bonds weaker than expected. May indicate frustrated bonding, defect-related softening, or partial occupancy.

## Common Issues

| Problem | Solution |
|---|---|
| **IFCs do not decay to zero at boundary** | Supercell too small. Increase min_length to 25-30 A. IFCs at the boundary should be <1% of the nearest-neighbor value. |
| **Negative frequencies in phonon dispersion** | Structure not properly relaxed. Relax with tighter fmax (<1e-5 eV/A), then regenerate displacements. |
| **QE forces parsing fails** | Ensure `tprnfor = .true.` is set in the QE input. Check that all SCF calculations converged. |
| **VASP vasprun.xml missing** | VASP calculation did not complete. Check job logs. Ensure NSW=0 for single-point SCF. |
| **Very large self-IFCs** | Normal -- the self-term Phi_ii is the negative sum of all off-diagonal terms. It should be the largest magnitude entry. |
| **Acoustic sum rule violated** | Phonopy enforces this automatically. If using raw FORCE_CONSTANTS from another source, check that sum_j Phi_ij = 0 for each i. |
| **IFC analysis gives different results for symmetry-equivalent bonds** | Increase symprec in phonopy or symmetrize the structure before generating displacements. |
| **Memory error with large supercell** | Force constant matrix scales as O(N^2). For >500 atoms, consider using MACE instead of DFT for forces, or analyze a smaller supercell. |
| **Bond types not separated in plots** | Bonds at very similar distances may overlap. Adjust the shell rounding threshold (round to 0.05 A instead of 0.1 A). |
