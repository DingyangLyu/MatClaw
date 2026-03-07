# Phonopy Interface for Lattice Dynamics

## When to Use

- You want to use phonopy as the phonon post-processor with QE or VASP as the force engine.
- You need displaced supercell structures for QE `pw.x` or VASP force calculations.
- You have completed force calculations and need to import results back into phonopy.
- You want phonon band structures, DOS, thermal properties, or interatomic force constant (IFC) analysis.
- You need the normalized trace of IFC tensors for bonding analysis.
- You need to sort phonon bands for smooth dispersion curves.
- Equivalent to VASPKIT functions 788-789 (phonopy interface).

## Method Selection

| Source Code | Method | Notes |
|---|---|---|
| QE (pw.x) | Method A: phonopy + QE finite displacements | Generate displaced structures, run QE SCF on each, parse forces, feed to phonopy |
| VASP | Method B: phonopy + VASP | Generate POSCAR displacements, run VASP on each, parse vasprun.xml. Input generation only. |
| MACE (pre-installed) | Method C: phonopy + MACE via ASE | Direct force evaluation (no DFT needed). See `thermal-properties/phonon/` for full workflow. |
| Any ASE calculator | Method C variant | Any calculator providing forces via ASE interface |

## Prerequisites

```bash
# Install phonopy (not pre-installed)
pip install phonopy seekpath
```

- Pre-installed: `ase`, `pymatgen`, `spglib`, `numpy`, `scipy`, `matplotlib`, `mace-torch`.
- QE binaries: `pw.x` in `/opt/qe/bin/`.
- For VASP: input preparation only (VASP binary not in container).

---

## Detailed Steps

### Method A: Phonopy + QE Finite Displacements

#### Step 1: Generate Displaced Supercells for QE

```python
#!/usr/bin/env python3
"""
Generate phonopy displaced supercells and write QE pw.x input files
for each displacement. This replaces VASPKIT 788 for QE.
"""
import os
import numpy as np
import phonopy
from pymatgen.core.structure import Structure, Lattice
from pymatgen.io.ase import AseAtomsAdaptor
from pymatgen.io.phonopy import get_phonopy_structure

# ============================================================
# CONFIGURATION
# ============================================================

STRUCTURE_FILE = "structure.cif"    # Input structure (CIF, POSCAR, etc.)
WORK_DIR = "/tmp/phonopy_qe"
PSEUDO_DIR = os.path.join(WORK_DIR, "pseudo")
MIN_LENGTH = 20.0                   # Minimum supercell length (A)
DISPLACEMENT = 0.01                 # Displacement distance (A)
SYMPREC = 1e-5                      # Symmetry precision
ECUTWFC = 60.0                      # QE cutoff (Ry)
ECUTRHO = 480.0                     # QE charge density cutoff (Ry)
K_DENSITY = 40                      # k-point density (per reciprocal A)

os.makedirs(WORK_DIR, exist_ok=True)
os.makedirs(PSEUDO_DIR, exist_ok=True)

# ============================================================
# LOAD AND PREPARE STRUCTURE
# ============================================================

structure = Structure.from_file(STRUCTURE_FILE)
print(f"Structure: {structure.composition.reduced_formula}")
print(f"  Atoms: {len(structure)}")
print(f"  Lattice: {structure.lattice.abc}")

# Convert to phonopy format
phonopy_structure = get_phonopy_structure(structure)

# Determine supercell matrix
cell_lengths = np.array(structure.lattice.abc)
multiples = np.maximum(np.ceil(MIN_LENGTH / cell_lengths).astype(int), 1)
supercell_matrix = np.diag(multiples)
print(f"  Supercell: {multiples[0]}x{multiples[1]}x{multiples[2]}")

# ============================================================
# GENERATE DISPLACEMENTS
# ============================================================

phonon = phonopy.Phonopy(
    phonopy_structure,
    supercell_matrix=supercell_matrix.tolist(),
    symprec=SYMPREC,
)

phonon.generate_displacements(distance=DISPLACEMENT)
supercells = phonon.supercells_with_displacements
n_disp = len(supercells)
print(f"  Displacements: {n_disp}")
print(f"  Atoms per supercell: {len(supercells[0].symbols)}")

# Save phonopy settings for later
phonon.save("phonopy_disp.yaml")
print(f"  Saved: phonopy_disp.yaml")

# ============================================================
# WRITE QE INPUT FOR EACH DISPLACEMENT
# ============================================================

elements = sorted(set(structure.composition.get_el_amt_dict().keys()))
pseudo_map = {el: f"{el}.pbe-n-kjpaw_psl.1.0.0.UPF" for el in elements}

def write_qe_scf(sc, index, work_dir):
    """Write QE SCF input for a displaced supercell."""
    sc_dir = os.path.join(work_dir, f"disp-{index:03d}")
    os.makedirs(sc_dir, exist_ok=True)

    # Convert phonopy supercell to pymatgen
    sc_structure = Structure(
        lattice=Lattice(sc.cell),
        species=sc.symbols,
        coords=sc.scaled_positions,
    )

    # Auto k-grid for supercell
    recip = sc_structure.lattice.reciprocal_lattice.abc
    kgrid = tuple(max(1, int(round(K_DENSITY / (2 * np.pi) * r))) for r in recip)

    species_order = sorted(set(sc.symbols))
    lines = []
    lines.append("&CONTROL")
    lines.append(f"    calculation = 'scf'")
    lines.append(f"    prefix      = 'phonon'")
    lines.append(f"    outdir      = './tmp'")
    lines.append(f"    pseudo_dir  = '{PSEUDO_DIR}'")
    lines.append(f"    tprnfor     = .true.")
    lines.append(f"    tstress     = .true.")
    lines.append("/\n")

    lines.append("&SYSTEM")
    lines.append(f"    ibrav       = 0")
    lines.append(f"    nat         = {len(sc.symbols)}")
    lines.append(f"    ntyp        = {len(species_order)}")
    lines.append(f"    ecutwfc     = {ECUTWFC}")
    lines.append(f"    ecutrho     = {ECUTRHO}")
    lines.append(f"    occupations = 'smearing'")
    lines.append(f"    smearing    = 'cold'")
    lines.append(f"    degauss     = 0.01")
    lines.append("/\n")

    lines.append("&ELECTRONS")
    lines.append(f"    conv_thr    = 1.0d-10")
    lines.append(f"    mixing_beta = 0.7")
    lines.append("/\n")

    lines.append("ATOMIC_SPECIES")
    for el in species_order:
        from pymatgen.core.periodic_table import Element
        mass = Element(el).atomic_mass
        lines.append(f"  {el:4s} {float(mass):10.4f}  {pseudo_map.get(el, el + '.UPF')}")
    lines.append("")

    lines.append("CELL_PARAMETERS angstrom")
    for row in sc.cell:
        lines.append(f"  {row[0]:16.10f} {row[1]:16.10f} {row[2]:16.10f}")
    lines.append("")

    lines.append("ATOMIC_POSITIONS crystal")
    for sym, pos in zip(sc.symbols, sc.scaled_positions):
        lines.append(f"  {sym:4s} {pos[0]:16.10f} {pos[1]:16.10f} {pos[2]:16.10f}")
    lines.append("")

    lines.append("K_POINTS automatic")
    lines.append(f"  {kgrid[0]} {kgrid[1]} {kgrid[2]}  0 0 0")

    input_file = os.path.join(sc_dir, "scf.in")
    with open(input_file, "w") as f:
        f.write("\n".join(lines) + "\n")

    return sc_dir


for i, sc in enumerate(supercells):
    sc_dir = write_qe_scf(sc, i + 1, WORK_DIR)
    if (i + 1) % 5 == 0 or (i + 1) == n_disp:
        print(f"  Written QE input for displacement {i+1}/{n_disp}")

# ============================================================
# GENERATE RUN SCRIPT
# ============================================================

nprocs = os.cpu_count() or 4
run_script = f"""#!/bin/bash
# Run all displacement SCF calculations
set -e
cd {WORK_DIR}
NP={nprocs}

for d in disp-*/; do
    echo "Running $d ..."
    cd $d
    mpirun --allow-run-as-root -np $NP pw.x < scf.in > scf.out 2>&1
    cd ..
done

echo "All displacement calculations complete."
"""

run_path = os.path.join(WORK_DIR, "run_all.sh")
with open(run_path, "w") as f:
    f.write(run_script)
os.chmod(run_path, 0o755)
print(f"\nRun script: {run_path}")
print(f"Execute: bash {run_path}")
```

#### Step 2: Parse QE Forces and Build Force Constants

```python
#!/usr/bin/env python3
"""
Parse QE SCF outputs for forces, import into phonopy,
compute force constants, phonon bands, DOS, and thermal properties.
Equivalent to VASPKIT 789 post-processing.
"""
import os
import re
import numpy as np
import phonopy
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ============================================================
# CONFIGURATION
# ============================================================

WORK_DIR = "/tmp/phonopy_qe"
MESH = [20, 20, 20]                  # q-mesh for DOS
T_MIN, T_MAX, T_STEP = 0, 1000, 10  # Temperature range

# ============================================================
# PARSE QE FORCES
# ============================================================

def parse_qe_forces(output_file):
    """Parse forces from QE pw.x output file."""
    forces = []
    with open(output_file) as f:
        lines = f.readlines()

    in_forces = False
    for line in lines:
        if "Forces acting on atoms" in line:
            in_forces = True
            forces = []
            continue
        if in_forces:
            if "force =" in line:
                parts = line.split("force =")[1].split()
                fx, fy, fz = float(parts[0]), float(parts[1]), float(parts[2])
                # QE forces are in Ry/Bohr, convert to eV/A
                # 1 Ry/Bohr = 25.7112 eV/A
                forces.append([fx * 25.7112, fy * 25.7112, fz * 25.7112])
            elif len(forces) > 0 and "force =" not in line and line.strip():
                if "Total force" in line:
                    in_forces = False

    return np.array(forces)

# ============================================================
# LOAD PHONOPY AND IMPORT FORCES
# ============================================================

# Load phonopy from saved displacement data
phonon = phonopy.load("phonopy_disp.yaml", produce_fc=False)

n_disp = len(phonon.supercells_with_displacements)
print(f"Loading forces from {n_disp} displacement calculations...")

forces_list = []
for i in range(1, n_disp + 1):
    output_file = os.path.join(WORK_DIR, f"disp-{i:03d}", "scf.out")
    if not os.path.exists(output_file):
        print(f"  WARNING: {output_file} not found!")
        continue

    forces = parse_qe_forces(output_file)
    if len(forces) == 0:
        print(f"  WARNING: No forces parsed from disp-{i:03d}/scf.out")
        continue

    forces_list.append(forces)
    if i % 5 == 0 or i == n_disp:
        print(f"  Parsed forces for displacement {i}/{n_disp}")

if len(forces_list) != n_disp:
    print(f"ERROR: Expected {n_disp} force sets, got {len(forces_list)}")
    print("Some displacement calculations may have failed.")

# ============================================================
# BUILD FORCE CONSTANTS
# ============================================================

phonon.forces = forces_list
phonon.produce_force_constants()
print("\nForce constants built.")

# Save for reuse
phonon.save("phonopy_params.yaml")

# ============================================================
# PHONON BAND STRUCTURE
# ============================================================

print("\n=== Phonon Band Structure ===")
phonon.auto_band_structure(npoints=101, write_yaml=True)

fig, ax = plt.subplots(figsize=(8, 5))
phonon.plot_band_structure(ax=ax)
ax.set_ylabel("Frequency (THz)")
ax.set_title(f"Phonon Band Structure")
ax.axhline(y=0, color="gray", linestyle="--", linewidth=0.5)
fig.tight_layout()
fig.savefig(os.path.join(WORK_DIR, "phonon_bands.png"), dpi=150, bbox_inches="tight")
plt.close()
print("  Saved: phonon_bands.png")

# ============================================================
# PHONON DOS
# ============================================================

print("\n=== Phonon DOS ===")
phonon.run_mesh(MESH, with_eigenvectors=False, is_gamma_center=True)
phonon.run_total_dos()

fig, ax = plt.subplots(figsize=(6, 4))
phonon.plot_total_dos(ax=ax)
ax.set_xlabel("Frequency (THz)")
ax.set_ylabel("DOS")
fig.tight_layout()
fig.savefig(os.path.join(WORK_DIR, "phonon_dos.png"), dpi=150, bbox_inches="tight")
plt.close()
print("  Saved: phonon_dos.png")

# ============================================================
# THERMAL PROPERTIES
# ============================================================

print("\n=== Thermal Properties ===")
phonon.run_thermal_properties(t_min=T_MIN, t_max=T_MAX, t_step=T_STEP)
tp = phonon.get_thermal_properties_dict()

temperatures = tp["temperatures"]
free_energy = tp["free_energy"]
entropy = tp["entropy"]
heat_capacity = tp["heat_capacity"]

fig, axes = plt.subplots(1, 3, figsize=(14, 4))
axes[0].plot(temperatures, free_energy, "b-")
axes[0].set_xlabel("T (K)"); axes[0].set_ylabel("F (kJ/mol)"); axes[0].set_title("Free Energy")
axes[1].plot(temperatures, entropy, "r-")
axes[1].set_xlabel("T (K)"); axes[1].set_ylabel("S (J/K/mol)"); axes[1].set_title("Entropy")
axes[2].plot(temperatures, heat_capacity, "g-")
axes[2].set_xlabel("T (K)"); axes[2].set_ylabel("Cv (J/K/mol)"); axes[2].set_title("Heat Capacity")
for ax in axes:
    ax.grid(True, alpha=0.3)
fig.tight_layout()
fig.savefig(os.path.join(WORK_DIR, "thermal_properties.png"), dpi=150, bbox_inches="tight")
plt.close()
print("  Saved: thermal_properties.png")

# Print table
print(f"\n  {'T (K)':>8} {'F (kJ/mol)':>12} {'S (J/K/mol)':>13} {'Cv (J/K/mol)':>14}")
for i in range(0, len(temperatures), max(1, len(temperatures) // 10)):
    print(f"  {temperatures[i]:8.1f} {free_energy[i]:12.4f} {entropy[i]:13.4f} {heat_capacity[i]:14.4f}")

# ============================================================
# STABILITY CHECK
# ============================================================

print("\n=== Stability Check ===")
mesh_dict = phonon.get_mesh_dict()
freqs = mesh_dict["frequencies"]
min_freq = freqs.min()

if min_freq < -0.1:
    print(f"  WARNING: Imaginary frequencies detected! Min: {min_freq:.4f} THz")
    print(f"  Structure is dynamically UNSTABLE.")
elif min_freq < 0:
    print(f"  Small negative frequencies (min: {min_freq:.4f} THz). Likely numerical noise.")
else:
    print(f"  All positive (min: {min_freq:.4f} THz). Structure is STABLE.")
```

#### Step 3: IFC Analysis -- Normalized Trace of Force Constant Tensors

```python
#!/usr/bin/env python3
"""
Analyze interatomic force constants (IFCs) from phonopy.
Compute the normalized trace of IFC tensors for bonding analysis.
"""
import numpy as np
import phonopy
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pymatgen.core.structure import Structure

WORK_DIR = "/tmp/phonopy_qe"

# ============================================================
# LOAD FORCE CONSTANTS
# ============================================================

phonon = phonopy.load("phonopy_params.yaml")
fc = phonon.force_constants  # shape: (n_atoms, n_atoms_super, 3, 3)
supercell = phonon.supercell
unitcell = phonon.unitcell

n_atoms_prim = len(unitcell.symbols)
n_atoms_super = len(supercell.symbols)

print(f"Force constants shape: {fc.shape}")
print(f"  Primitive cell atoms: {n_atoms_prim}")
print(f"  Supercell atoms: {n_atoms_super}")

# ============================================================
# NORMALIZED TRACE OF IFC TENSORS
# ============================================================

print("\n=== Normalized Trace of IFC Tensors ===")
print("The trace of the 3x3 IFC tensor Phi(i,j) measures the")
print("overall stiffness of the interaction between atoms i and j.\n")

# For each pair (i in unit cell, j in supercell), compute:
#   Tr(Phi_ij) = Phi_xx + Phi_yy + Phi_zz
#   |Phi_ij| = sqrt(sum of all 9 elements squared) = Frobenius norm
#   Normalized trace = Tr(Phi_ij) / |Phi_ij|

ifc_data = []

for i in range(n_atoms_prim):
    pos_i = unitcell.scaled_positions[i]
    sym_i = unitcell.symbols[i]

    for j in range(n_atoms_super):
        phi = fc[i, j]  # 3x3 tensor
        trace = np.trace(phi)
        frob_norm = np.linalg.norm(phi, "fro")

        if frob_norm < 1e-10:
            continue  # skip zero interactions

        normalized_trace = trace / frob_norm

        # Compute distance
        pos_j = supercell.scaled_positions[j]
        # Convert to Cartesian for distance
        cart_i = unitcell.cell @ pos_i
        cart_j = supercell.cell @ pos_j
        dist = np.linalg.norm(cart_j - cart_i)

        sym_j = supercell.symbols[j]

        ifc_data.append({
            "i": i, "j": j,
            "sym_i": sym_i, "sym_j": sym_j,
            "distance": dist,
            "trace": trace,
            "frob_norm": frob_norm,
            "norm_trace": normalized_trace,
        })

# Sort by distance
ifc_data.sort(key=lambda x: x["distance"])

# Print top interactions
print(f"{'Pair':>10} {'Dist (A)':>10} {'Tr(Phi)':>12} {'|Phi|':>12} {'Tr/|Phi|':>10}")
print("-" * 60)

seen_pairs = set()
for d in ifc_data[:50]:
    pair_key = (d["sym_i"], d["sym_j"], round(d["distance"], 2))
    if pair_key in seen_pairs:
        continue
    seen_pairs.add(pair_key)
    print(f"{d['sym_i']}-{d['sym_j']:>6s} {d['distance']:10.4f} "
          f"{d['trace']:12.4f} {d['frob_norm']:12.4f} {d['norm_trace']:10.4f}")

# ============================================================
# PLOT: IFC MAGNITUDE VS DISTANCE
# ============================================================

distances = [d["distance"] for d in ifc_data if d["distance"] > 0.1]
traces = [abs(d["trace"]) for d in ifc_data if d["distance"] > 0.1]
frob_norms = [d["frob_norm"] for d in ifc_data if d["distance"] > 0.1]

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

axes[0].scatter(distances, frob_norms, s=10, alpha=0.5, color="steelblue")
axes[0].set_xlabel("Distance (A)", fontsize=12)
axes[0].set_ylabel("|Phi| (eV/A^2)", fontsize=12)
axes[0].set_title("IFC Frobenius Norm vs Distance")
axes[0].set_yscale("log")
axes[0].grid(True, alpha=0.3)

axes[1].scatter(distances, traces, s=10, alpha=0.5, color="coral")
axes[1].set_xlabel("Distance (A)", fontsize=12)
axes[1].set_ylabel("|Tr(Phi)| (eV/A^2)", fontsize=12)
axes[1].set_title("IFC Trace Magnitude vs Distance")
axes[1].set_yscale("log")
axes[1].grid(True, alpha=0.3)

fig.tight_layout()
fig.savefig(os.path.join(WORK_DIR, "ifc_analysis.png"), dpi=150, bbox_inches="tight")
plt.close()
print(f"\nSaved: {WORK_DIR}/ifc_analysis.png")
```

#### Step 4: Sort Phonon Bands for Continuity

```python
#!/usr/bin/env python3
"""
Sort phonon bands for smooth dispersion by tracking eigenvector continuity.
Phonopy bands can have band crossings that make plots discontinuous;
this script resolves them by matching eigenvectors between adjacent q-points.
"""
import numpy as np
import phonopy
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

WORK_DIR = "/tmp/phonopy_qe"

# ============================================================
# LOAD PHONOPY AND COMPUTE BANDS WITH EIGENVECTORS
# ============================================================

phonon = phonopy.load("phonopy_params.yaml")

# Get band structure with eigenvectors
phonon.auto_band_structure(npoints=101, with_eigenvectors=True, write_yaml=False)

band_dict = phonon.get_band_structure_dict()
distances = band_dict["distances"]       # list of arrays, one per segment
frequencies = band_dict["frequencies"]    # list of arrays (nq, nbands)
eigenvectors = band_dict["eigenvectors"] # list of arrays (nq, nbands, natoms, 3) complex

# ============================================================
# SORT BANDS BY EIGENVECTOR OVERLAP
# ============================================================

def sort_bands_by_eigenvectors(freqs, eigvecs):
    """
    Sort bands at each q-point to maximize continuity.
    Uses eigenvector overlap |<e_n(q)|e_m(q+dq)>|^2 to track bands.

    Parameters
    ----------
    freqs : array, shape (nq, nbands)
    eigvecs : array, shape (nq, nbands, natoms*3) complex

    Returns
    -------
    sorted_freqs : array, shape (nq, nbands)
    """
    nq, nbands = freqs.shape
    sorted_freqs = np.copy(freqs)
    sorted_eigvecs = np.copy(eigvecs)

    for iq in range(1, nq):
        # Compute overlap matrix between eigenvectors at q and q-1
        # overlap[m, n] = |<e_m(q-1) | e_n(q)>|^2
        prev = sorted_eigvecs[iq - 1]  # (nbands, natoms*3)
        curr = eigvecs[iq]              # (nbands, natoms*3)

        overlap = np.abs(np.dot(prev.conj(), curr.T))**2  # (nbands, nbands)

        # Hungarian algorithm or greedy matching
        # Use greedy for simplicity: assign each previous band to best matching current band
        used = set()
        perm = np.zeros(nbands, dtype=int)

        for m in range(nbands):
            # Find best match for band m (from previous q)
            scores = overlap[m].copy()
            for u in used:
                scores[u] = -1  # exclude already-assigned bands
            best = np.argmax(scores)
            perm[m] = best
            used.add(best)

        # Apply permutation
        sorted_freqs[iq] = freqs[iq, perm]
        sorted_eigvecs[iq] = eigvecs[iq, perm]

    return sorted_freqs


# Flatten eigenvectors for sorting
sorted_segments = []
for seg_idx in range(len(frequencies)):
    freqs = frequencies[seg_idx]      # (nq, nbands)
    eigvecs = eigenvectors[seg_idx]   # (nq, nbands, natoms, 3) complex

    nq, nbands = freqs.shape
    natoms = eigvecs.shape[2]

    # Reshape eigenvectors to (nq, nbands, natoms*3)
    eigvecs_flat = eigvecs.reshape(nq, nbands, natoms * 3)

    sorted_freqs = sort_bands_by_eigenvectors(freqs, eigvecs_flat)
    sorted_segments.append(sorted_freqs)

# ============================================================
# PLOT: ORIGINAL VS SORTED
# ============================================================

fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=True)

# Original
offset = 0
for seg_idx, (dists, freqs) in enumerate(zip(distances, frequencies)):
    x = dists + offset
    nbands = freqs.shape[1]
    for b in range(nbands):
        axes[0].plot(x, freqs[:, b], "b-", linewidth=0.7)
    offset = x[-1]

axes[0].set_ylabel("Frequency (THz)", fontsize=12)
axes[0].set_title("Original (unsorted)", fontsize=13)
axes[0].axhline(y=0, color="gray", linestyle="--", linewidth=0.5)
axes[0].grid(True, alpha=0.2)

# Sorted
offset = 0
for seg_idx, (dists, freqs) in enumerate(zip(distances, sorted_segments)):
    x = dists + offset
    nbands = freqs.shape[1]
    for b in range(nbands):
        axes[1].plot(x, freqs[:, b], "r-", linewidth=0.7)
    offset = x[-1]

axes[1].set_title("Sorted (eigenvector continuity)", fontsize=13)
axes[1].axhline(y=0, color="gray", linestyle="--", linewidth=0.5)
axes[1].grid(True, alpha=0.2)

for ax in axes:
    ax.set_xlabel("q-path", fontsize=12)

fig.tight_layout()
fig.savefig(os.path.join(WORK_DIR, "phonon_bands_sorted.png"), dpi=150, bbox_inches="tight")
plt.close()
print(f"Saved: {WORK_DIR}/phonon_bands_sorted.png")
```

### Method B: Phonopy + VASP (Input Generation)

Generate VASP POSCAR files for each phonopy displacement (VASPKIT 788 equivalent).

```python
#!/usr/bin/env python3
"""
Generate phonopy displaced POSCAR files for VASP force calculations.
After running VASP on each, parse forces with phonopy's VASP interface.
Equivalent to VASPKIT function 788.
"""
import os
import numpy as np
import phonopy
from pymatgen.core.structure import Structure, Lattice
from pymatgen.io.vasp import Poscar, Incar, Kpoints
from pymatgen.io.phonopy import get_phonopy_structure

# ============================================================
# CONFIGURATION
# ============================================================

STRUCTURE_FILE = "POSCAR"
WORK_DIR = "./phonopy_vasp"
MIN_LENGTH = 20.0
DISPLACEMENT = 0.01
SYMPREC = 1e-5
ENCUT = 520        # eV

os.makedirs(WORK_DIR, exist_ok=True)

# ============================================================
# SETUP
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
print(f"Displacements: {len(supercells)}")

# ============================================================
# WRITE VASP INPUTS FOR EACH DISPLACEMENT
# ============================================================

for i, sc in enumerate(supercells):
    sc_dir = os.path.join(WORK_DIR, f"disp-{i+1:03d}")
    os.makedirs(sc_dir, exist_ok=True)

    # Write POSCAR
    sc_structure = Structure(
        lattice=Lattice(sc.cell),
        species=sc.symbols,
        coords=sc.scaled_positions,
    )
    poscar = Poscar(sc_structure)
    poscar.write_file(os.path.join(sc_dir, "POSCAR"))

    # Write INCAR
    incar = Incar({
        "SYSTEM": f"Phonopy displacement {i+1}",
        "ENCUT": ENCUT,
        "PREC": "Accurate",
        "EDIFF": 1e-8,
        "ISMEAR": 0,
        "SIGMA": 0.05,
        "IBRION": -1,
        "NSW": 0,
        "LWAVE": False,
        "LCHARG": False,
        "LREAL": "Auto" if len(sc.symbols) > 20 else False,
        "NCORE": 4,
    })
    incar.write_file(os.path.join(sc_dir, "INCAR"))

    # Write KPOINTS
    kpoints = Kpoints.automatic_density(sc_structure, kppa=1000, force_gamma=True)
    kpoints.write_file(os.path.join(sc_dir, "KPOINTS"))

    if (i + 1) % 5 == 0 or (i + 1) == len(supercells):
        print(f"  Written VASP inputs for displacement {i+1}/{len(supercells)}")

# Save phonopy displacement data
phonon.save(os.path.join(WORK_DIR, "phonopy_disp.yaml"))

print(f"\nVASP inputs in: {WORK_DIR}/disp-XXX/")
print("After running VASP in each directory, parse with:")
print(f"  phonopy --vasp -f {WORK_DIR}/disp-*/vasprun.xml")
```

### Method B (continued): Parse VASP Forces and Compute Phonons

After running VASP on each displaced supercell:

```python
#!/usr/bin/env python3
"""
Parse VASP vasprun.xml forces and compute phonon properties.
Run this after all VASP displacement calculations are complete.
Equivalent to VASPKIT function 789 post-processing.
"""
import os
import numpy as np
import phonopy
from pymatgen.io.vasp import Vasprun

WORK_DIR = "./phonopy_vasp"

# Load phonopy displacement data
phonon = phonopy.load(os.path.join(WORK_DIR, "phonopy_disp.yaml"), produce_fc=False)
n_disp = len(phonon.supercells_with_displacements)

# Parse forces from each vasprun.xml
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
        print(f"  Parsed forces {i}/{n_disp}")

phonon.forces = forces_list
phonon.produce_force_constants()
phonon.save(os.path.join(WORK_DIR, "phonopy_params.yaml"))

# Phonon bands, DOS, thermal -- same as QE workflow above
phonon.auto_band_structure(npoints=101, write_yaml=True)
phonon.run_mesh([20, 20, 20])
phonon.run_total_dos()
phonon.run_thermal_properties(t_min=0, t_max=1000, t_step=10)

print("Phonon properties computed. Use phonopy_params.yaml for further analysis.")
```

## Key Parameters

| Parameter | Default | Notes |
|---|---|---|
| `displacement` | 0.01 A | Standard value. Use 0.005 for noisy potentials, 0.03 for testing. |
| `min_length` | 20.0 A | Minimum supercell dimension. 15 A minimum viable; 20 A recommended. |
| `symprec` | 1e-5 | Symmetry tolerance. Tighter = more displacements. |
| `fmax` for relaxation | 1e-4 eV/A | Must be tight for phonons. Residual forces bias force constants. |
| `conv_thr` (QE) | 1e-10 Ry | Tight SCF for accurate forces on displaced supercells. |
| `EDIFF` (VASP) | 1e-8 eV | Same purpose as conv_thr. |
| `MESH` | [20,20,20] | q-mesh for DOS and thermal properties. Denser = smoother. |
| K-grid density | 40 per recip. A | For supercell SCF. Auto-scales with supercell size. |

## Interpreting Results

### Phonon band structure
- All positive frequencies: dynamically stable.
- Imaginary frequencies (plotted negative): dynamical instability; structure is a saddle point.
- Small imaginary near Gamma only: usually numerical noise from insufficient supercell or loose force convergence.

### IFC analysis (normalized trace)
- Large |Tr(Phi)|: strong bonding interaction.
- Tr(Phi)/|Phi| near 1: isotropic interaction (all directions equally stiff).
- Tr(Phi)/|Phi| near 0: highly anisotropic interaction.
- IFCs should decay with distance. If they remain large at the supercell boundary, increase supercell size.

### Thermal properties
- Heat capacity Cv approaches 3NkB (Dulong-Petit) at high T.
- Entropy monotonically increases with T.
- Free energy decreases with T (from -TS term).
- These are harmonic values; for anharmonic corrections, see `gruneisen-qha/`.

### Band sorting
- Unsorted bands show apparent crossings and discontinuities.
- Eigenvector-sorted bands track the same mode across q-points, giving smooth curves.
- Sorting is important for identifying specific modes and for comparing with experiment.

## Common Issues

| Issue | Cause | Fix |
|---|---|---|
| Many imaginary frequencies | Structure not properly relaxed | Relax with tighter fmax (1e-5 eV/A), then regenerate displacements. |
| Small imaginary near Gamma | Supercell too small | Increase min_length to 25-30 A. |
| QE forces parsing fails | Output format mismatch | Check that `tprnfor = .true.` is set. Verify output completed. |
| VASP vasprun.xml not found | Calculation did not complete | Check VASP job status. Ensure NSW=0 for single-point SCF. |
| phonopy symmetry error | Input structure not matching | Symmetrize with spglib first: `SpacegroupAnalyzer(structure).get_refined_structure()`. |
| Band sorting introduces artifacts | Degenerate or near-degenerate bands | Increase npoints for finer q-sampling. Degeneracies at high-symmetry points are physical. |
| IFCs do not decay to zero | Supercell too small | IFCs at the supercell boundary should be negligible (<1% of max). Increase supercell. |
| Memory error with large supercell | Too many atoms | Normal for supercells > 500 atoms. Reduce min_length or use MACE instead of DFT. |
