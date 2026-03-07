# Phonon Calculations

## When to Use

- Computing phonon band structures and phonon density of states
- Checking dynamical stability of a crystal structure (imaginary frequencies = unstable)
- Obtaining thermodynamic properties: heat capacity (Cv), entropy (S), Helmholtz free energy (F) vs temperature
- Validating a relaxed structure before further calculations
- Comparing lattice dynamics from MLIP vs DFT

## Method Selection

```
Need phonon properties?
  Is speed important and DFT accuracy not required?
    YES --> Method A: ASE + MACE + phonopy (finite displacements)
    NO  --> Is the system a simple crystal with < ~20 atoms in the unit cell?
              YES --> Method B: QE DFPT (ph.x, most accurate for simple systems)
              NO  --> Method A with MACE, or Method B on a cluster
```

Key trade-offs:
- **ASE + MACE + phonopy**: Fast (seconds to minutes), no pseudopotential setup, good for screening. Accuracy depends on MACE model quality for your chemistry.
- **QE DFPT (ph.x)**: Full DFT accuracy, handles long-range dipole corrections (LO-TO splitting) natively, but much slower. Best for publication-quality results on small unit cells.

## Prerequisites

```bash
# phonopy and seekpath are required for Method A
pip install phonopy seekpath
```

Pre-installed: `ase`, `mace-torch`, `pymatgen`, `spglib`, `numpy`, `scipy`, `matplotlib`.

QE binaries: `pw.x`, `ph.x`, `matdyn.x`, `q2r.x` in `/opt/qe/bin/`.

## Detailed Steps

### Method A: ASE + MACE + phonopy (Finite Displacement)

This follows the atomate2 phonon workflow pattern: relax structure, determine supercell size (min_length >= 20 A), generate displaced supercells, compute forces, build force constants, then extract phonon properties.

```python
#!/usr/bin/env python3
"""
Phonon calculation using ASE + MACE + phonopy.
Complete runnable script for any crystal structure.
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from ase.io import read, write
from ase.optimize import LBFGS
from ase.constraints import ExpCellFilter

from mace.calculators import mace_mp

from pymatgen.core.structure import Structure
from pymatgen.io.ase import AseAtomsAdaptor
from pymatgen.io.phonopy import get_phonopy_structure
from pymatgen.symmetry.bandstructure import HighSymmKpath

import phonopy
from phonopy.phonon.band_structure import get_band_qpoints_and_path_connections

import warnings
warnings.filterwarnings("ignore")

# ============================================================
# 1. CONFIGURATION
# ============================================================

STRUCTURE_FILE = "structure.cif"       # Input structure (CIF, POSCAR, etc.)
MACE_MODEL = "medium"                  # "small", "medium", or "large"
DEVICE = "cpu"                         # "cpu" or "cuda"
MIN_LENGTH = 20.0                      # Minimum supercell length in Angstrom
DISPLACEMENT = 0.01                    # Displacement in Angstrom
SYMPREC = 1e-5                         # Symmetry precision
FMAX = 1e-4                            # Force convergence for relaxation (eV/A)
MESH = [20, 20, 20]                    # q-point mesh for DOS
T_MIN, T_MAX, T_STEP = 0, 1000, 10    # Temperature range for thermodynamics (K)

# ============================================================
# 2. SET UP CALCULATOR
# ============================================================

calc = mace_mp(model=MACE_MODEL, device=DEVICE, default_dtype="float64")

# ============================================================
# 3. READ AND RELAX STRUCTURE
# ============================================================

atoms = read(STRUCTURE_FILE)
atoms.calc = calc

print("=== Structure Relaxation ===")
print(f"  Formula: {atoms.get_chemical_formula()}")
print(f"  Initial energy: {atoms.get_potential_energy():.6f} eV")

# Relax both positions and cell
ecf = ExpCellFilter(atoms, hydrostatic_strain=False)
opt = LBFGS(ecf, logfile="relax.log")
opt.run(fmax=FMAX, steps=500)

print(f"  Final energy:   {atoms.get_potential_energy():.6f} eV")
print(f"  Relaxed cell:   {atoms.cell.lengths()}")
write("relaxed.cif", atoms)

# ============================================================
# 4. DETERMINE SUPERCELL MATRIX
# ============================================================

def get_supercell_matrix(atoms, min_length=20.0):
    """Determine supercell matrix so each lattice vector >= min_length."""
    cell_lengths = atoms.cell.lengths()
    multiples = np.ceil(min_length / cell_lengths).astype(int)
    multiples = np.maximum(multiples, 1)  # At least 1x1x1
    supercell_matrix = np.diag(multiples)
    print(f"  Cell lengths: {cell_lengths}")
    print(f"  Supercell: {multiples[0]}x{multiples[1]}x{multiples[2]}")
    return supercell_matrix

print("\n=== Supercell Determination ===")
supercell_matrix = get_supercell_matrix(atoms, min_length=MIN_LENGTH)

# ============================================================
# 5. SET UP PHONOPY AND GENERATE DISPLACEMENTS
# ============================================================

# Convert ASE atoms to pymatgen then to phonopy
pmg_structure = AseAtomsAdaptor.get_structure(atoms)
phonopy_structure = get_phonopy_structure(pmg_structure)

phonon = phonopy.Phonopy(
    phonopy_structure,
    supercell_matrix=supercell_matrix.tolist(),
    symprec=SYMPREC,
)

phonon.generate_displacements(distance=DISPLACEMENT)
supercells = phonon.supercells_with_displacements

n_disp = len(supercells)
print(f"\n=== Force Calculations ===")
print(f"  Number of displaced supercells: {n_disp}")

# ============================================================
# 6. CALCULATE FORCES ON DISPLACED SUPERCELLS
# ============================================================

adaptor = AseAtomsAdaptor()
forces_list = []

for i, sc in enumerate(supercells):
    # Convert phonopy supercell to ASE atoms
    sc_atoms = adaptor.get_atoms(
        Structure(
            lattice=sc.cell,
            species=sc.symbols,
            coords=sc.scaled_positions,
        )
    )
    sc_atoms.calc = calc
    forces = sc_atoms.get_forces()
    forces_list.append(forces)
    if (i + 1) % 5 == 0 or (i + 1) == n_disp:
        print(f"  Computed forces for supercell {i+1}/{n_disp}")

# ============================================================
# 7. BUILD FORCE CONSTANTS
# ============================================================

phonon.forces = forces_list
phonon.produce_force_constants()

print("\n=== Force Constants Built ===")

# Save force constants for later reuse
phonon.save("phonopy_params.yaml")

# ============================================================
# 8. PHONON BAND STRUCTURE
# ============================================================

print("\n=== Phonon Band Structure ===")

# Get high-symmetry k-path using seekpath via phonopy
phonon.auto_band_structure(
    npoints=101,
    with_eigenvectors=False,
    write_yaml=True,
)

# Plot band structure
fig, ax = plt.subplots(figsize=(8, 5))
phonon.plot_band_structure(ax=ax)
ax.set_ylabel("Frequency (THz)")
ax.set_title(f"Phonon Band Structure - {atoms.get_chemical_formula()}")
ax.axhline(y=0, color="gray", linestyle="--", linewidth=0.5)
fig.tight_layout()
fig.savefig("phonon_bands.png", dpi=150, bbox_inches="tight")
plt.close()
print("  Saved: phonon_bands.png")

# ============================================================
# 9. PHONON DOS
# ============================================================

print("\n=== Phonon DOS ===")

phonon.run_mesh(MESH, with_eigenvectors=False, is_gamma_center=True)
phonon.run_total_dos()

fig, ax = plt.subplots(figsize=(6, 4))
phonon.plot_total_dos(ax=ax)
ax.set_xlabel("Frequency (THz)")
ax.set_ylabel("DOS")
ax.set_title(f"Phonon DOS - {atoms.get_chemical_formula()}")
fig.tight_layout()
fig.savefig("phonon_dos.png", dpi=150, bbox_inches="tight")
plt.close()
print("  Saved: phonon_dos.png")

# ============================================================
# 10. THERMODYNAMIC PROPERTIES
# ============================================================

print("\n=== Thermodynamic Properties ===")

phonon.run_thermal_properties(
    t_min=T_MIN, t_max=T_MAX, t_step=T_STEP,
)
tp = phonon.get_thermal_properties_dict()

temperatures = tp["temperatures"]       # K
free_energy = tp["free_energy"]          # kJ/mol
entropy = tp["entropy"]                  # J/K/mol
heat_capacity = tp["heat_capacity"]      # J/K/mol

fig, axes = plt.subplots(1, 3, figsize=(14, 4))

axes[0].plot(temperatures, free_energy, "b-")
axes[0].set_xlabel("Temperature (K)")
axes[0].set_ylabel("Helmholtz Free Energy (kJ/mol)")
axes[0].set_title("Free Energy")

axes[1].plot(temperatures, entropy, "r-")
axes[1].set_xlabel("Temperature (K)")
axes[1].set_ylabel("Entropy (J/K/mol)")
axes[1].set_title("Entropy")

axes[2].plot(temperatures, heat_capacity, "g-")
axes[2].set_xlabel("Temperature (K)")
axes[2].set_ylabel("Cv (J/K/mol)")
axes[2].set_title("Heat Capacity")

for ax in axes:
    ax.grid(True, alpha=0.3)

fig.suptitle(f"Thermodynamic Properties - {atoms.get_chemical_formula()}", y=1.02)
fig.tight_layout()
fig.savefig("thermodynamic_properties.png", dpi=150, bbox_inches="tight")
plt.close()
print("  Saved: thermodynamic_properties.png")

# Print table of selected temperatures
print(f"\n  {'T (K)':>8} {'F (kJ/mol)':>12} {'S (J/K/mol)':>13} {'Cv (J/K/mol)':>14}")
for i in range(0, len(temperatures), max(1, len(temperatures) // 10)):
    print(f"  {temperatures[i]:8.1f} {free_energy[i]:12.4f} {entropy[i]:13.4f} {heat_capacity[i]:14.4f}")

# ============================================================
# 11. CHECK FOR IMAGINARY FREQUENCIES
# ============================================================

print("\n=== Stability Check ===")

mesh_dict = phonon.get_mesh_dict()
frequencies = mesh_dict["frequencies"]  # shape: (n_qpoints, n_bands)
min_freq = frequencies.min()

if min_freq < -0.1:  # threshold in THz
    print(f"  WARNING: Imaginary frequencies detected!")
    print(f"  Most negative frequency: {min_freq:.4f} THz")
    print(f"  The structure is dynamically UNSTABLE.")
    print(f"  This means the structure sits at a saddle point on the PES.")
    print(f"  Consider: (1) re-relaxing with tighter criteria,")
    print(f"            (2) distorting along the unstable mode,")
    print(f"            (3) using a larger supercell.")
elif min_freq < 0:
    print(f"  Small negative frequencies detected (min: {min_freq:.4f} THz).")
    print(f"  Likely numerical noise near Gamma. Structure is probably stable.")
    print(f"  Try a larger supercell or tighter force convergence if concerned.")
else:
    print(f"  All frequencies positive (min: {min_freq:.4f} THz).")
    print(f"  Structure is dynamically STABLE.")

print("\nDone.")
```

### Method B: QE DFPT (ph.x)

Density Functional Perturbation Theory provides exact DFT-level phonons without supercells.

#### Step 1: SCF calculation

```python
#!/usr/bin/env python3
"""
Generate QE input files for DFPT phonon calculation.
Handles pseudopotential download, SCF input, and ph.x input.
"""

import os
import subprocess
from pymatgen.core.structure import Structure
from pymatgen.io.pwscf import PWInput

# ============================================================
# CONFIGURATION
# ============================================================

STRUCTURE_FILE = "structure.cif"
WORK_DIR = "/tmp/qe_phonon"
PSEUDO_DIR = os.path.join(WORK_DIR, "pseudo")
ECUTWFC = 60.0       # Ry - plane-wave cutoff
ECUTRHO = 480.0      # Ry - charge density cutoff (8x for PAW)
K_GRID = (6, 6, 6)  # k-point grid for SCF
Q_GRID = (4, 4, 4)  # q-point grid for phonons

os.makedirs(PSEUDO_DIR, exist_ok=True)
os.makedirs(WORK_DIR, exist_ok=True)

# ============================================================
# LOAD STRUCTURE
# ============================================================

structure = Structure.from_file(STRUCTURE_FILE)
print(f"Structure: {structure.composition.reduced_formula}")
print(f"Space group: {structure.get_space_group_info()}")

# ============================================================
# DOWNLOAD PSEUDOPOTENTIALS
# ============================================================

# Map of element -> pseudopotential filename (PAW PBE, pslibrary)
PSEUDO_MAP = {
    "H": "H.pbe-kjpaw_psl.1.0.0.UPF",
    "Li": "Li.pbe-s-kjpaw_psl.1.0.0.UPF",
    "C": "C.pbe-n-kjpaw_psl.1.0.0.UPF",
    "N": "N.pbe-n-kjpaw_psl.1.0.0.UPF",
    "O": "O.pbe-n-kjpaw_psl.1.0.0.UPF",
    "Na": "Na.pbe-spn-kjpaw_psl.1.0.0.UPF",
    "Al": "Al.pbe-n-kjpaw_psl.1.0.0.UPF",
    "Si": "Si.pbe-n-kjpaw_psl.1.0.0.UPF",
    "K": "K.pbe-spn-kjpaw_psl.1.0.0.UPF",
    "Ca": "Ca.pbe-spn-kjpaw_psl.1.0.0.UPF",
    "Ti": "Ti.pbe-spn-kjpaw_psl.1.0.0.UPF",
    "Fe": "Fe.pbe-spn-kjpaw_psl.1.0.0.UPF",
    "Cu": "Cu.pbe-dn-kjpaw_psl.1.0.0.UPF",
    "Zn": "Zn.pbe-dn-kjpaw_psl.1.0.0.UPF",
    "Ba": "Ba.pbe-spn-kjpaw_psl.1.0.0.UPF",
    "Ni": "Ni.pbe-n-kjpaw_psl.1.0.0.UPF",
}

elements = set(str(sp) for sp in structure.species)
pseudo_dict = {}
base_url = "https://pseudopotentials.quantum-espresso.org/upf_files"

for el in elements:
    if el not in PSEUDO_MAP:
        raise ValueError(
            f"No pseudopotential mapping for {el}. "
            f"Add it to PSEUDO_MAP or download manually."
        )
    pp_file = PSEUDO_MAP[el]
    pp_path = os.path.join(PSEUDO_DIR, pp_file)
    if not os.path.exists(pp_path):
        print(f"Downloading pseudopotential for {el}...")
        subprocess.run(
            ["wget", "-q", f"{base_url}/{pp_file}", "-O", pp_path],
            check=True,
        )
    pseudo_dict[el] = pp_file

# ============================================================
# WRITE SCF INPUT
# ============================================================

scf_input = f"""&CONTROL
    calculation = 'scf'
    prefix      = 'phonon'
    outdir      = './tmp'
    pseudo_dir  = '{PSEUDO_DIR}'
    tprnfor     = .true.
    tstress     = .true.
    verbosity   = 'high'
/
&SYSTEM
    ibrav       = 0
    nat         = {len(structure)}
    ntyp        = {len(elements)}
    ecutwfc     = {ECUTWFC}
    ecutrho     = {ECUTRHO}
    occupations = 'smearing'
    smearing    = 'cold'
    degauss     = 0.01
/
&ELECTRONS
    conv_thr    = 1.0d-10
    mixing_beta = 0.7
/

ATOMIC_SPECIES
"""

for el in sorted(elements):
    # Get mass from pymatgen
    from pymatgen.core.periodic_table import Element
    mass = Element(el).atomic_mass
    scf_input += f"  {el:4s} {mass:10.4f}  {pseudo_dict[el]}\n"

scf_input += "\nCELL_PARAMETERS angstrom\n"
for vec in structure.lattice.matrix:
    scf_input += f"  {vec[0]:16.10f} {vec[1]:16.10f} {vec[2]:16.10f}\n"

scf_input += "\nATOMIC_POSITIONS crystal\n"
for site in structure:
    el = str(site.specie)
    scf_input += f"  {el:4s} {site.frac_coords[0]:16.10f} {site.frac_coords[1]:16.10f} {site.frac_coords[2]:16.10f}\n"

scf_input += f"\nK_POINTS automatic\n  {K_GRID[0]} {K_GRID[1]} {K_GRID[2]}  0 0 0\n"

scf_path = os.path.join(WORK_DIR, "scf.in")
with open(scf_path, "w") as f:
    f.write(scf_input)
print(f"Written: {scf_path}")

# ============================================================
# WRITE PH.X INPUT
# ============================================================

ph_input = f"""Phonon calculation on uniform q-grid
&INPUTPH
    prefix   = 'phonon'
    outdir   = './tmp'
    fildyn   = 'phonon.dyn'
    ldisp    = .true.
    nq1      = {Q_GRID[0]}
    nq2      = {Q_GRID[1]}
    nq3      = {Q_GRID[2]}
    tr2_ph   = 1.0d-14
    asr_type = 'crystal'
/
"""

ph_path = os.path.join(WORK_DIR, "ph.in")
with open(ph_path, "w") as f:
    f.write(ph_input)
print(f"Written: {ph_path}")

# ============================================================
# WRITE Q2R.X INPUT (IFCs in real space)
# ============================================================

q2r_input = f"""&INPUT
    fildyn = 'phonon.dyn'
    zasr   = 'crystal'
    flfrc  = 'phonon.fc'
/
"""

q2r_path = os.path.join(WORK_DIR, "q2r.in")
with open(q2r_path, "w") as f:
    f.write(q2r_input)
print(f"Written: {q2r_path}")

# ============================================================
# WRITE MATDYN.X INPUT (band structure interpolation)
# ============================================================

# Generate k-path from pymatgen
from pymatgen.symmetry.bandstructure import HighSymmKpath
kpath = HighSymmKpath(structure)
path_labels = kpath.kpath["path"]
kpoints_dict = kpath.kpath["kpoints"]

# Build the q-point path for matdyn.x
qpoints_lines = []
npoints_per_segment = 51
total_segments = sum(len(seg) - 1 for seg in path_labels)

for segment in path_labels:
    for i in range(len(segment) - 1):
        k1 = kpoints_dict[segment[i]]
        k2 = kpoints_dict[segment[i + 1]]
        label1 = segment[i]
        label2 = segment[i + 1]
        qpoints_lines.append(
            f"  {k1[0]:10.6f} {k1[1]:10.6f} {k1[2]:10.6f}  {npoints_per_segment}"
        )
    # Add last point of last segment
    k_last = kpoints_dict[segment[-1]]
    qpoints_lines.append(
        f"  {k_last[0]:10.6f} {k_last[1]:10.6f} {k_last[2]:10.6f}  0"
    )

matdyn_bands_input = f"""&INPUT
    asr   = 'crystal'
    flfrc = 'phonon.fc'
    flfrq = 'phonon_bands.freq'
    flvec = 'phonon_bands.modes'
    q_in_band_form = .true.
/
{len(qpoints_lines)}
"""
matdyn_bands_input += "\n".join(qpoints_lines) + "\n"

matdyn_bands_path = os.path.join(WORK_DIR, "matdyn_bands.in")
with open(matdyn_bands_path, "w") as f:
    f.write(matdyn_bands_input)
print(f"Written: {matdyn_bands_path}")

# ============================================================
# WRITE MATDYN.X INPUT (DOS)
# ============================================================

matdyn_dos_input = f"""&INPUT
    asr    = 'crystal'
    flfrc  = 'phonon.fc'
    flfrq  = 'phonon_dos.freq'
    fldos  = 'phonon.dos'
    dos    = .true.
    nk1    = 20
    nk2    = 20
    nk3    = 20
    ndos   = 500
/
"""

matdyn_dos_path = os.path.join(WORK_DIR, "matdyn_dos.in")
with open(matdyn_dos_path, "w") as f:
    f.write(matdyn_dos_input)
print(f"Written: {matdyn_dos_path}")

# ============================================================
# PRINT EXECUTION COMMANDS
# ============================================================

nprocs = os.cpu_count() or 4

print(f"\n=== Run these commands in {WORK_DIR} ===\n")
print(f"cd {WORK_DIR}")
print(f"# Step 1: SCF")
print(f"mpirun --allow-run-as-root -np {nprocs} pw.x < scf.in > scf.out 2>&1")
print(f"# Step 2: Phonon (this is the expensive step)")
print(f"mpirun --allow-run-as-root -np {nprocs} ph.x < ph.in > ph.out 2>&1")
print(f"# Step 3: Real-space force constants")
print(f"q2r.x < q2r.in > q2r.out 2>&1")
print(f"# Step 4a: Band structure interpolation")
print(f"matdyn.x < matdyn_bands.in > matdyn_bands.out 2>&1")
print(f"# Step 4b: DOS")
print(f"matdyn.x < matdyn_dos.in > matdyn_dos.out 2>&1")
```

#### Step 2: Run QE (shell commands)

```bash
cd /tmp/qe_phonon
NPROCS=$(nproc)

# SCF
mpirun --allow-run-as-root -np $NPROCS pw.x < scf.in > scf.out 2>&1

# Phonon calculation (may take hours for large systems)
mpirun --allow-run-as-root -np $NPROCS ph.x < ph.in > ph.out 2>&1

# Fourier transform dynamical matrices to real-space force constants
q2r.x < q2r.in > q2r.out 2>&1

# Interpolate phonon dispersion along high-symmetry path
matdyn.x < matdyn_bands.in > matdyn_bands.out 2>&1

# Compute phonon DOS on dense mesh
matdyn.x < matdyn_dos.in > matdyn_dos.out 2>&1
```

#### Step 3: Post-process and plot QE phonon results

```python
#!/usr/bin/env python3
"""
Post-process QE DFPT phonon results: parse and plot band structure + DOS.
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import os

WORK_DIR = "/tmp/qe_phonon"

# ============================================================
# PARSE PHONON BAND STRUCTURE
# ============================================================

def parse_matdyn_freq(filename):
    """Parse matdyn.x frequency output file."""
    with open(filename) as f:
        lines = f.readlines()

    qpoints = []
    frequencies = []
    current_freqs = []

    for line in lines:
        line = line.strip()
        if not line:
            continue
        if line.startswith("q =") or line.startswith("nbnd"):
            # skip header-type lines
            continue

        parts = line.split()
        # Detect q-point line: starts with q-point coordinates and nbnd info
        # Actually, matdyn output format has q-point lines and frequency lines
        # q-point lines have 3 floats after "q ="
        if "q =" in line:
            if current_freqs:
                frequencies.append(current_freqs)
                current_freqs = []
            coords = line.split("q =")[1].strip().strip("()").split()
            qpoints.append([float(x) for x in coords[:3]])
        else:
            # Frequency line
            try:
                freqs = [float(x) for x in parts]
                current_freqs.extend(freqs)
            except ValueError:
                pass

    if current_freqs:
        frequencies.append(current_freqs)

    return np.array(qpoints), np.array(frequencies)

def parse_phonon_dos(filename):
    """Parse matdyn.x DOS output."""
    data = np.loadtxt(filename, comments="#")
    # Columns: frequency (cm^-1), DOS
    return data[:, 0], data[:, 1]


# Plot band structure
freq_file = os.path.join(WORK_DIR, "phonon_bands.freq")
if os.path.exists(freq_file):
    qpoints, frequencies = parse_matdyn_freq(freq_file)

    # Convert cm^-1 to THz (1 cm^-1 = 0.02998 THz)
    frequencies_thz = frequencies * 0.02998

    n_qpoints = len(qpoints)
    x_axis = np.arange(n_qpoints)

    fig, ax = plt.subplots(figsize=(8, 5))
    for band_idx in range(frequencies_thz.shape[1]):
        ax.plot(x_axis, frequencies_thz[:, band_idx], "b-", linewidth=0.8)

    ax.axhline(y=0, color="gray", linestyle="--", linewidth=0.5)
    ax.set_xlabel("q-point index")
    ax.set_ylabel("Frequency (THz)")
    ax.set_title("QE DFPT Phonon Band Structure")
    ax.set_xlim(0, n_qpoints - 1)
    fig.tight_layout()
    fig.savefig(os.path.join(WORK_DIR, "qe_phonon_bands.png"), dpi=150)
    plt.close()
    print(f"Saved: {WORK_DIR}/qe_phonon_bands.png")

    # Check for imaginary modes
    min_freq = frequencies.min()
    if min_freq < -10:  # cm^-1
        print(f"WARNING: Imaginary frequencies! Most negative: {min_freq:.2f} cm^-1")
    else:
        print(f"All frequencies positive or near-zero. Min: {min_freq:.2f} cm^-1")

# Plot DOS
dos_file = os.path.join(WORK_DIR, "phonon.dos")
if os.path.exists(dos_file):
    freq_dos, dos = parse_phonon_dos(dos_file)
    freq_dos_thz = freq_dos * 0.02998

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(freq_dos_thz, dos, "r-")
    ax.axvline(x=0, color="gray", linestyle="--", linewidth=0.5)
    ax.set_xlabel("Frequency (THz)")
    ax.set_ylabel("Phonon DOS")
    ax.set_title("QE DFPT Phonon DOS")
    ax.set_xlim(left=min(freq_dos_thz) - 1)
    fig.tight_layout()
    fig.savefig(os.path.join(WORK_DIR, "qe_phonon_dos.png"), dpi=150)
    plt.close()
    print(f"Saved: {WORK_DIR}/qe_phonon_dos.png")
```

## Key Parameters

| Parameter | Default | Notes |
|---|---|---|
| `displacement` | 0.01 A | Finite displacement distance. 0.01 A is standard. Larger values (0.03) can help noisy potentials but introduce anharmonicity. |
| `min_length` | 20.0 A | Minimum supercell dimension. Ensures phonon interactions are captured. 15 A is minimum viable; 20 A recommended for accuracy. |
| `symprec` | 1e-5 | Symmetry detection tolerance. Tighter = fewer symmetry ops found = more displacements. |
| `fmax` | 1e-4 eV/A | Relaxation convergence. Must be tight for phonons; residual forces bias force constants. |
| `MESH` | [20,20,20] | q-mesh for DOS. Denser = smoother DOS. Adjust if cell is anisotropic. |
| `ecutwfc` | 60 Ry | QE wavefunction cutoff. System-dependent; converge for your system. |
| `nq1,nq2,nq3` | 4,4,4 | QE q-grid for DFPT. Finer = more accurate but much more expensive. |
| `tr2_ph` | 1e-14 | QE phonon self-consistency threshold. 1e-14 is tight; 1e-12 acceptable for screening. |
| `asr_type` / `zasr` | 'crystal' | Acoustic sum rule. 'crystal' enforces translational invariance for 3D periodics. Use 'simple' as fallback. |

## Interpreting Results

### Phonon band structure
- **All positive frequencies**: structure is dynamically stable at harmonic level.
- **Imaginary frequencies** (plotted as negative by convention): the structure is a saddle point. The eigenvector of the imaginary mode shows the instability direction.
  - At Gamma: usually indicates the structure wants to distort (ferroelectric, Jahn-Teller, etc.).
  - At zone boundary: indicates a supercell reconstruction (e.g., CDW, antiferrodistortive).
  - Small imaginary near Gamma only: often numerical artifact from insufficient supercell size or loose force convergence.

### Phonon DOS
- Peaks correspond to flat bands (van Hove singularities).
- Gap in DOS = phonon band gap (common in materials with large mass contrast like PbTe).
- High-frequency cutoff relates to the lightest atom and stiffest bond.

### Thermodynamic properties
- **Heat capacity** (Cv): approaches 3NkB (Dulong-Petit) at high T. If it plateaus below this, check for numerical issues.
- **Entropy**: always increases with T.
- **Free energy**: always decreases with T (from -TS contribution).
- These are harmonic values. For thermal expansion effects, see `gruneisen-qha/`.

## Common Issues

| Issue | Cause | Fix |
|---|---|---|
| Many imaginary frequencies | Structure not properly relaxed | Relax with tighter `fmax` (1e-5 eV/A or smaller), then rerun phonons |
| Small imaginary near Gamma | Supercell too small or numerical noise | Increase `min_length` to 25-30 A; use `displacement=0.005` |
| Phonopy error about symmetry | Input structure not matching detected symmetry | Symmetrize structure with `spglib` before phonon calc, or set `symprec=1e-3` |
| QE ph.x crashes | Insufficient memory, wrong q-grid | Reduce q-grid, increase `conv_thr`, check `ecutwfc` convergence |
| Bands look noisy (MACE) | MACE model not accurate for this chemistry | Try `model="large"`, or switch to QE DFPT for publication quality |
| Very long runtime (MACE) | Large supercell with many displacements | Normal for > 500 atoms; use symmetry (`sym_reduce=True`) |
| Acoustic modes not at zero at Gamma | Missing acoustic sum rule correction | phonopy applies ASR by default; for QE, use `asr_type='crystal'` |
