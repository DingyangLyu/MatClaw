# Anharmonicity Score (sigma^A)

## When to Use

- Quantifying how anharmonic a material is before choosing a thermodynamics method (QHA vs. MD)
- Screening materials for strong anharmonicity (thermoelectrics, phase-change materials, superionic conductors)
- Deciding whether the quasi-harmonic approximation (QHA) is valid for a given material at a target temperature
- Reproducing or comparing with the anharmonicity metric from Knoop et al., Phys. Rev. Materials 4, 083809 (2020)
- Pre-screening before expensive phonon-phonon scattering (phono3py) or AIMD calculations

## Method Selection

```
Need to know if QHA is reliable for your material at temperature T?
  YES --> Compute sigma^A (this skill)
    sigma^A < 0.2  --> QHA is safe, use gruneisen-qha/
    sigma^A 0.2-0.5 --> QHA is marginal; consider validating with MD
    sigma^A > 0.5  --> QHA breaks down; use molecular-dynamics/ for thermodynamics

Quick screening or large set of materials?
  YES --> Method A: ASE + MACE + phonopy (minutes per material)

Publication-quality single material?
  YES --> Method B: QE DFT + phonopy (hours to days)
```

The anharmonicity score sigma^A measures how much the true (DFT/MLIP) forces on a thermally displaced configuration deviate from the forces predicted by the harmonic model. It is defined as:

sigma^A = std(F_DFT - F_harmonic) / std(F_DFT)

where F_DFT are the forces computed on a one-shot thermally displaced supercell, and F_harmonic are the forces predicted from the harmonic force constants applied to the same displacements. A perfectly harmonic material gives sigma^A = 0; strongly anharmonic materials give sigma^A > 0.5.

## Prerequisites

```bash
pip install phonopy seekpath
```

Pre-installed: `ase`, `mace-torch`, `pymatgen`, `numpy`, `scipy`, `matplotlib`.

## Detailed Steps

### Method A: ASE + MACE + phonopy (fast screening)

The workflow:
1. Relax the structure at ground state
2. Build a phonon model: finite displacements + force constants via phonopy
3. Generate a one-shot thermally displaced supercell at target temperature T
4. Compute MACE forces on the displaced supercell (these are the "true" forces)
5. Compute harmonic forces from force constants times displacements
6. Calculate sigma^A and generate diagnostic plots

```python
#!/usr/bin/env python3
"""
Anharmonicity Score (sigma^A) using ASE + MACE + phonopy.
Measures how anharmonic a material is at a target temperature.

Reference: Knoop et al., Phys. Rev. Materials 4, 083809 (2020)
Inspired by atomate2's BaseAnharmonicityMaker workflow.

Complete runnable script.
"""

import os
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

import phonopy
from phonopy.structure.atoms import PhonopyAtoms

import warnings
warnings.filterwarnings("ignore")

# ============================================================
# 1. CONFIGURATION
# ============================================================

STRUCTURE_FILE = "structure.cif"       # Input structure (CIF, POSCAR, etc.)
MACE_MODEL = "medium"                  # "small", "medium", "large"
DEVICE = "cpu"                         # "cpu" or "cuda"

# Phonon settings
MIN_LENGTH = 15.0                      # Supercell min length (A) -- 15 A for screening, 20 A for accuracy
DISPLACEMENT = 0.01                    # Finite displacement distance (A) for force constants
SYMPREC = 1e-5                         # Symmetry precision
FMAX = 1e-4                            # Relaxation force convergence (eV/A)
MESH = [20, 20, 20]                    # q-mesh for phonon DOS (used in phonon model)

# Anharmonicity settings
TARGET_TEMPERATURE = 300.0             # Temperature (K) for thermal displacement sampling
N_SNAPSHOTS = 1                        # Number of displaced snapshots (1 is standard for sigma^A)

WORK_DIR = "/tmp/anharmonicity_calc"
os.makedirs(WORK_DIR, exist_ok=True)

# ============================================================
# 2. SET UP CALCULATOR AND RELAX STRUCTURE
# ============================================================

calc = mace_mp(model=MACE_MODEL, device=DEVICE, default_dtype="float64")

atoms_orig = read(STRUCTURE_FILE)
atoms_orig.calc = calc

print("=== Structure Relaxation ===")
print(f"  Formula: {atoms_orig.get_chemical_formula()}")
print(f"  Number of atoms: {len(atoms_orig)}")

ecf = ExpCellFilter(atoms_orig, hydrostatic_strain=False)
opt = LBFGS(ecf, logfile=os.path.join(WORK_DIR, "relax.log"))
opt.run(fmax=FMAX, steps=500)

V0 = atoms_orig.get_volume()
E0 = atoms_orig.get_potential_energy()
print(f"  Relaxed volume: {V0:.4f} A^3")
print(f"  Relaxed energy: {E0:.6f} eV")
write(os.path.join(WORK_DIR, "relaxed.cif"), atoms_orig)

# ============================================================
# 3. DETERMINE SUPERCELL MATRIX
# ============================================================

def get_supercell_matrix(atoms, min_length=15.0):
    """Build a diagonal supercell matrix ensuring each lattice direction >= min_length."""
    cell_lengths = atoms.cell.lengths()
    multiples = np.ceil(min_length / cell_lengths).astype(int)
    multiples = np.maximum(multiples, 1)
    return np.diag(multiples)

supercell_matrix = get_supercell_matrix(atoms_orig, min_length=MIN_LENGTH)
print(f"  Supercell matrix: diag({np.diag(supercell_matrix)})")

# ============================================================
# 4. COMPUTE HARMONIC FORCE CONSTANTS
# ============================================================

print("\n=== Harmonic Phonon Calculation ===")

adaptor = AseAtomsAdaptor()
pmg_struct = adaptor.get_structure(atoms_orig)
phonopy_struct = get_phonopy_structure(pmg_struct)

phonon = phonopy.Phonopy(
    phonopy_struct,
    supercell_matrix=supercell_matrix.tolist(),
    symprec=SYMPREC,
)
phonon.generate_displacements(distance=DISPLACEMENT)
supercells = phonon.supercells_with_displacements
n_disp = len(supercells)
print(f"  Number of finite displacements: {n_disp}")

# Compute forces for each displaced supercell
forces_list = []
for j, sc in enumerate(supercells):
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
    if (j + 1) % 5 == 0 or (j + 1) == n_disp:
        print(f"    Displacement {j + 1}/{n_disp} done")

phonon.forces = forces_list
phonon.produce_force_constants()

# Quick check: run mesh and report any imaginary modes
phonon.run_mesh(MESH, with_eigenvectors=False, is_gamma_center=True)
mesh_dict = phonon.get_mesh_dict()
min_freq = mesh_dict["frequencies"].min()
print(f"  Minimum phonon frequency: {min_freq:.4f} THz")
if min_freq < -0.5:
    print("  WARNING: Significant imaginary modes detected!")
    print("  The harmonic model may be unreliable. Check structure relaxation.")

# Save the phonon model
phonon.save(os.path.join(WORK_DIR, "phonopy_harmonic.yaml"))
print("  Saved harmonic phonon model: phonopy_harmonic.yaml")

# ============================================================
# 5. GENERATE THERMALLY DISPLACED SUPERCELL (ONE-SHOT)
# ============================================================

print(f"\n=== Thermal Displacement at T = {TARGET_TEMPERATURE} K ===")

# Use phonopy's thermal displacement generation
# This samples atomic positions from the phonon thermal distribution at temperature T
phonon.generate_displacements(
    temperature=TARGET_TEMPERATURE,
    number_of_snapshots=N_SNAPSHOTS,
    random_seed=42,
)

# The displaced supercells are stored in phonon.supercells_with_displacements
displaced_supercells = phonon.supercells_with_displacements
n_sc_atoms = len(displaced_supercells[0].positions)
print(f"  Generated {len(displaced_supercells)} thermally displaced snapshot(s)")
print(f"  Supercell has {n_sc_atoms} atoms")

# Get the equilibrium (undisplaced) supercell for reference
supercell_eq = phonon.supercell  # Equilibrium supercell (PhonopyAtoms)
positions_eq = supercell_eq.positions.copy()

# ============================================================
# 6. COMPUTE "TRUE" (MACE) FORCES ON DISPLACED CONFIGURATION
# ============================================================

print("\n=== Computing MACE Forces on Displaced Configuration ===")

all_sigma_A = []
all_forces_true = []
all_forces_harmonic = []
all_displacements = []

for i_snap, sc_disp in enumerate(displaced_supercells):
    print(f"\n--- Snapshot {i_snap + 1}/{len(displaced_supercells)} ---")

    # Convert displaced supercell to ASE Atoms
    sc_atoms_disp = adaptor.get_atoms(
        Structure(
            lattice=sc_disp.cell,
            species=sc_disp.symbols,
            coords=sc_disp.scaled_positions,
        )
    )
    sc_atoms_disp.calc = calc
    forces_true = sc_atoms_disp.get_forces()  # Shape: (n_sc_atoms, 3)

    # Compute displacements from equilibrium
    positions_disp = sc_disp.positions.copy()
    displacements = positions_disp - positions_eq  # Shape: (n_sc_atoms, 3)

    # Report displacement statistics
    disp_magnitudes = np.linalg.norm(displacements, axis=1)
    print(f"  Mean displacement: {np.mean(disp_magnitudes):.4f} A")
    print(f"  Max displacement:  {np.max(disp_magnitudes):.4f} A")

    # ============================================================
    # 7. COMPUTE HARMONIC FORCES FROM FORCE CONSTANTS
    # ============================================================

    # F_harmonic_i = -sum_j Phi_ij * u_j
    # where Phi_ij is the force constant matrix and u_j are displacements
    #
    # phonon.force_constants has shape (n_atoms_prim_or_super, n_atoms_super, 3, 3)
    # depending on whether compact or full format is used.

    fc = phonon.force_constants  # May be compact (n_prim, n_super, 3, 3) or full (n_super, n_super, 3, 3)

    # Ensure we have the full force constant matrix
    if fc.shape[0] != n_sc_atoms:
        # Force constants are in compact form; convert to full
        from phonopy.harmonic.force_constants import (
            compact_fc_to_full_fc,
        )
        p2s_map = phonon.primitive.p2s_map  # primitive to supercell atom mapping
        s2p_map = phonon.supercell.s2p_map if hasattr(phonon.supercell, 's2p_map') else None

        # Use phonopy's symmetry to distribute force constants
        fc_full = np.zeros((n_sc_atoms, n_sc_atoms, 3, 3), dtype=float)
        try:
            # phonopy >= 2.x has this utility
            from phonopy.harmonic.force_constants import distribute_force_constants_by_translations
            # Alternatively, set force constants type to full
            phonon.symmetrize_force_constants()
            fc_shape = phonon.force_constants.shape
            if fc_shape[0] == n_sc_atoms:
                fc_full = phonon.force_constants
            else:
                # Manual expansion: use compact_fc_to_full_fc if available
                fc_full = compact_fc_to_full_fc(phonon, phonon.force_constants)
        except (ImportError, TypeError):
            # Fallback: regenerate with full FC
            phonon.produce_force_constants(fc_calculator=None)
            # Try setting to full type
            try:
                phonon.force_constants = phonon.force_constants
                if phonon.force_constants.shape[0] == n_sc_atoms:
                    fc_full = phonon.force_constants
                else:
                    raise ValueError("Cannot convert to full FC")
            except Exception:
                print("  WARNING: Could not expand force constants to full matrix.")
                print("  Attempting direct matrix-vector product with compact FC.")
                fc_full = None
    else:
        fc_full = fc

    # Compute harmonic forces: F_i = -sum_j FC[i,j] . u_j
    if fc_full is not None and fc_full.shape[0] == n_sc_atoms:
        # fc_full shape: (N, N, 3, 3), displacements shape: (N, 3)
        # F_harm[i] = -sum_j fc_full[i, j] @ u[j]
        forces_harmonic = np.zeros_like(forces_true)
        for i in range(n_sc_atoms):
            for j in range(n_sc_atoms):
                forces_harmonic[i] -= fc_full[i, j] @ displacements[j]
    else:
        # Use compact FC with s2p mapping
        n_prim = fc.shape[0]
        s2p = phonon.primitive.s2p_map  # maps supercell atom index -> primitive atom index
        p2s = phonon.primitive.p2s_map  # maps primitive atom index -> supercell atom index

        forces_harmonic = np.zeros_like(forces_true)
        # For compact FC: fc[p, j, 3, 3] where p is primitive index
        # F_i = -sum_j FC[s2p[i], j] @ u[j]  (only valid if atom i maps to a primitive atom in p2s)
        # For atoms not directly in p2s, we use the s2p mapping
        for i in range(n_sc_atoms):
            p_i = s2p[i]  # Which primitive atom does supercell atom i correspond to?
            # Find the index in p2s list
            p_idx = np.where(p2s == p_i)[0]
            if len(p_idx) > 0:
                p_idx = p_idx[0]
                for j in range(n_sc_atoms):
                    forces_harmonic[i] -= fc[p_idx, j] @ displacements[j]

    # ============================================================
    # 8. COMPUTE ANHARMONICITY SCORE sigma^A
    # ============================================================

    force_diff = forces_true - forces_harmonic  # (n_sc_atoms, 3)

    # Flatten to 3N vector for std computation (as in Knoop et al.)
    f_true_flat = forces_true.flatten()
    f_harm_flat = forces_harmonic.flatten()
    f_diff_flat = force_diff.flatten()

    std_diff = np.std(f_diff_flat)
    std_true = np.std(f_true_flat)

    if std_true > 1e-12:
        sigma_A = std_diff / std_true
    else:
        sigma_A = 0.0
        print("  WARNING: std(F_true) is near zero -- forces are negligible.")

    print(f"\n  === Anharmonicity Score ===")
    print(f"  std(F_true):          {std_true:.6f} eV/A")
    print(f"  std(F_harmonic):      {np.std(f_harm_flat):.6f} eV/A")
    print(f"  std(F_true - F_harm): {std_diff:.6f} eV/A")
    print(f"  sigma^A = {sigma_A:.4f}")

    if sigma_A < 0.2:
        print(f"  --> Material is HARMONIC at {TARGET_TEMPERATURE} K (QHA is reliable)")
    elif sigma_A < 0.5:
        print(f"  --> Material is MODERATELY ANHARMONIC at {TARGET_TEMPERATURE} K (QHA marginal)")
    else:
        print(f"  --> Material is STRONGLY ANHARMONIC at {TARGET_TEMPERATURE} K (use MD)")

    all_sigma_A.append(sigma_A)
    all_forces_true.append(forces_true)
    all_forces_harmonic.append(forces_harmonic)
    all_displacements.append(displacements)

    # Per-atom anharmonicity
    per_atom_sigma = np.zeros(n_sc_atoms)
    for iatom in range(n_sc_atoms):
        f_true_atom = forces_true[iatom]
        f_diff_atom = force_diff[iatom]
        norm_true = np.linalg.norm(f_true_atom)
        norm_diff = np.linalg.norm(f_diff_atom)
        if norm_true > 1e-12:
            per_atom_sigma[iatom] = norm_diff / norm_true
        else:
            per_atom_sigma[iatom] = 0.0

# ============================================================
# 9. PLOTTING
# ============================================================

print("\n=== Generating Plots ===")

# Use the last snapshot's data for plots (or first if only one)
forces_true = all_forces_true[0]
forces_harmonic = all_forces_harmonic[0]
force_diff = forces_true - forces_harmonic
sigma_A = all_sigma_A[0]
displacements = all_displacements[0]

f_true_flat = forces_true.flatten()
f_harm_flat = forces_harmonic.flatten()

# --- Plot 1: Force parity plot (F_harmonic vs F_true) ---
fig, ax = plt.subplots(figsize=(6, 6))
ax.scatter(f_true_flat, f_harm_flat, s=3, alpha=0.4, c="steelblue", edgecolors="none")

# Perfect agreement line
fmin = min(f_true_flat.min(), f_harm_flat.min())
fmax_val = max(f_true_flat.max(), f_harm_flat.max())
margin = 0.05 * (fmax_val - fmin)
lims = [fmin - margin, fmax_val + margin]
ax.plot(lims, lims, "k--", linewidth=1.0, label="Perfect harmonic")

ax.set_xlabel("True (MACE) Force Component (eV/A)", fontsize=11)
ax.set_ylabel("Harmonic Force Component (eV/A)", fontsize=11)
ax.set_title(
    f"Force Parity Plot -- $\\sigma^A$ = {sigma_A:.3f} at {TARGET_TEMPERATURE} K",
    fontsize=12,
)
ax.set_xlim(lims)
ax.set_ylim(lims)
ax.set_aspect("equal")
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3)
fig.tight_layout()
fig.savefig(os.path.join(WORK_DIR, "force_parity.png"), dpi=150)
plt.close()
print("  Saved: force_parity.png")

# --- Plot 2: Per-atom sigma^A histogram ---
per_atom_sigma = np.zeros(n_sc_atoms)
for iatom in range(n_sc_atoms):
    f_true_atom = forces_true[iatom]
    f_diff_atom = (forces_true - forces_harmonic)[iatom]
    norm_true = np.linalg.norm(f_true_atom)
    norm_diff = np.linalg.norm(f_diff_atom)
    if norm_true > 1e-12:
        per_atom_sigma[iatom] = norm_diff / norm_true
    else:
        per_atom_sigma[iatom] = 0.0

# Color by species
species_list = list(displaced_supercells[0].symbols)
unique_species = sorted(set(species_list))
colors_map = plt.cm.Set1(np.linspace(0, 1, max(len(unique_species), 3)))

fig, ax = plt.subplots(figsize=(8, 4))
for idx_sp, sp in enumerate(unique_species):
    mask = np.array([s == sp for s in species_list])
    ax.hist(
        per_atom_sigma[mask],
        bins=30,
        alpha=0.6,
        label=sp,
        color=colors_map[idx_sp],
        edgecolor="black",
        linewidth=0.5,
    )

ax.axvline(
    x=sigma_A, color="red", linestyle="--", linewidth=1.5,
    label=f"Global $\\sigma^A$ = {sigma_A:.3f}",
)
ax.set_xlabel("Per-Atom Anharmonicity |F_diff| / |F_true|", fontsize=11)
ax.set_ylabel("Count", fontsize=11)
ax.set_title(
    f"Per-Atom Anharmonicity Distribution at {TARGET_TEMPERATURE} K",
    fontsize=12,
)
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)
fig.tight_layout()
fig.savefig(os.path.join(WORK_DIR, "per_atom_anharmonicity.png"), dpi=150)
plt.close()
print("  Saved: per_atom_anharmonicity.png")

# --- Plot 3: Force residual (F_true - F_harmonic) distribution ---
fig, ax = plt.subplots(figsize=(7, 4))
f_diff_flat = (forces_true - forces_harmonic).flatten()
ax.hist(f_diff_flat, bins=80, density=True, alpha=0.7, color="salmon", edgecolor="black", linewidth=0.3)
ax.set_xlabel("Force Residual F_true - F_harmonic (eV/A)", fontsize=11)
ax.set_ylabel("Probability Density", fontsize=11)
ax.set_title(
    f"Force Residual Distribution ($\\sigma^A$ = {sigma_A:.3f})",
    fontsize=12,
)
ax.axvline(x=0, color="black", linestyle="-", linewidth=0.8)

# Overlay Gaussian with the measured std
from scipy.stats import norm as gaussian_dist
x_range = np.linspace(f_diff_flat.min(), f_diff_flat.max(), 300)
ax.plot(
    x_range,
    gaussian_dist.pdf(x_range, loc=np.mean(f_diff_flat), scale=np.std(f_diff_flat)),
    "r-", linewidth=1.5, label=f"Gaussian fit (std={np.std(f_diff_flat):.4f})",
)
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)
fig.tight_layout()
fig.savefig(os.path.join(WORK_DIR, "force_residual_distribution.png"), dpi=150)
plt.close()
print("  Saved: force_residual_distribution.png")

# --- Plot 4: Displacement magnitude vs force residual magnitude (per atom) ---
fig, ax = plt.subplots(figsize=(6, 5))
disp_mag = np.linalg.norm(displacements, axis=1)
fdiff_mag = np.linalg.norm(forces_true - forces_harmonic, axis=1)

for idx_sp, sp in enumerate(unique_species):
    mask = np.array([s == sp for s in species_list])
    ax.scatter(
        disp_mag[mask], fdiff_mag[mask],
        s=15, alpha=0.6, label=sp, color=colors_map[idx_sp],
        edgecolors="black", linewidths=0.3,
    )

ax.set_xlabel("Displacement Magnitude (A)", fontsize=11)
ax.set_ylabel("|F_true - F_harmonic| (eV/A)", fontsize=11)
ax.set_title("Force Residual vs Displacement", fontsize=12)
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)
fig.tight_layout()
fig.savefig(os.path.join(WORK_DIR, "displacement_vs_residual.png"), dpi=150)
plt.close()
print("  Saved: displacement_vs_residual.png")

# ============================================================
# 10. TEMPERATURE SCAN (OPTIONAL)
# ============================================================

# Scan sigma^A across a range of temperatures to see how anharmonicity grows
TEMP_SCAN = [100, 200, 300, 500, 800]
print(f"\n=== Temperature Scan of sigma^A ===")
print(f"  Temperatures: {TEMP_SCAN} K")

sigma_vs_T = []

for T_scan in TEMP_SCAN:
    # Reload the harmonic phonon model to reset displacements
    phonon_scan = phonopy.load(os.path.join(WORK_DIR, "phonopy_harmonic.yaml"))

    phonon_scan.generate_displacements(
        temperature=T_scan,
        number_of_snapshots=1,
        random_seed=42,
    )
    sc_disp_scan = phonon_scan.supercells_with_displacements[0]
    sc_eq_scan = phonon_scan.supercell

    # MACE forces
    sc_atoms_scan = adaptor.get_atoms(
        Structure(
            lattice=sc_disp_scan.cell,
            species=sc_disp_scan.symbols,
            coords=sc_disp_scan.scaled_positions,
        )
    )
    sc_atoms_scan.calc = calc
    f_true_scan = sc_atoms_scan.get_forces()

    # Displacements and harmonic forces
    u_scan = sc_disp_scan.positions - sc_eq_scan.positions
    n_at = len(sc_eq_scan.positions)

    fc_scan = phonon_scan.force_constants
    if fc_scan.shape[0] == n_at:
        f_harm_scan = np.zeros_like(f_true_scan)
        for i in range(n_at):
            for j in range(n_at):
                f_harm_scan[i] -= fc_scan[i, j] @ u_scan[j]
    else:
        # Compact FC: expand using s2p mapping
        s2p_scan = phonon_scan.primitive.s2p_map
        p2s_scan = phonon_scan.primitive.p2s_map
        f_harm_scan = np.zeros_like(f_true_scan)
        for i in range(n_at):
            p_i = s2p_scan[i]
            p_idx = np.where(p2s_scan == p_i)[0]
            if len(p_idx) > 0:
                p_idx = p_idx[0]
                for j in range(n_at):
                    f_harm_scan[i] -= fc_scan[p_idx, j] @ u_scan[j]

    diff_scan = f_true_scan - f_harm_scan
    std_diff_scan = np.std(diff_scan.flatten())
    std_true_scan = np.std(f_true_scan.flatten())
    sigma_scan = std_diff_scan / std_true_scan if std_true_scan > 1e-12 else 0.0
    sigma_vs_T.append(sigma_scan)
    print(f"  T = {T_scan:6.0f} K : sigma^A = {sigma_scan:.4f}")

# --- Plot 5: sigma^A vs temperature ---
fig, ax = plt.subplots(figsize=(6, 4))
ax.plot(TEMP_SCAN, sigma_vs_T, "o-", color="darkred", markersize=8, linewidth=2)
ax.axhline(y=0.2, color="green", linestyle="--", alpha=0.7, label="Harmonic limit (0.2)")
ax.axhline(y=0.5, color="orange", linestyle="--", alpha=0.7, label="QHA breakdown (0.5)")
ax.fill_between(TEMP_SCAN, 0, 0.2, alpha=0.08, color="green")
ax.fill_between(TEMP_SCAN, 0.2, 0.5, alpha=0.08, color="orange")
ax.fill_between(TEMP_SCAN, 0.5, max(1.0, max(sigma_vs_T) + 0.1), alpha=0.08, color="red")
ax.set_xlabel("Temperature (K)", fontsize=11)
ax.set_ylabel("$\\sigma^A$", fontsize=13)
ax.set_title("Anharmonicity Score vs Temperature", fontsize=12)
ax.set_ylim(bottom=0)
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)
fig.tight_layout()
fig.savefig(os.path.join(WORK_DIR, "sigma_vs_temperature.png"), dpi=150)
plt.close()
print("  Saved: sigma_vs_temperature.png")

# ============================================================
# 11. SUMMARY
# ============================================================

print("\n" + "=" * 60)
print("  ANHARMONICITY ANALYSIS COMPLETE")
print("=" * 60)
print(f"  Material:      {atoms_orig.get_chemical_formula()}")
print(f"  Supercell:     diag({np.diag(supercell_matrix)}) ({n_sc_atoms} atoms)")
print(f"  Temperature:   {TARGET_TEMPERATURE} K")
print(f"  sigma^A:       {all_sigma_A[0]:.4f}")
print()
print("  Temperature scan:")
for T_val, sig_val in zip(TEMP_SCAN, sigma_vs_T):
    label = "HARMONIC" if sig_val < 0.2 else ("MODERATE" if sig_val < 0.5 else "ANHARMONIC")
    print(f"    {T_val:6.0f} K : sigma^A = {sig_val:.4f}  [{label}]")
print()
print(f"  Output directory: {WORK_DIR}")
print(f"  Plots: force_parity.png, per_atom_anharmonicity.png,")
print(f"         force_residual_distribution.png, displacement_vs_residual.png,")
print(f"         sigma_vs_temperature.png")
print("=" * 60)
```

### Method B: QE DFT (publication quality)

For publication-quality results, replace the MACE calculator with Quantum Espresso DFT.
The workflow is identical but uses QE for (a) structural relaxation, (b) harmonic force constants, and (c) forces on the thermally displaced supercell.

**Step 1: Relax the structure with QE**

```bash
#!/bin/bash
# relax.sh -- QE structural relaxation
# Adjust pseudopotentials and parameters for your system

cat > relax.in << 'EOF'
&CONTROL
  calculation = 'vc-relax'
  prefix      = 'struct'
  outdir      = './tmp'
  pseudo_dir  = '/opt/pseudopotentials/SSSP_1.3.0_PBEsol_efficiency/'
  forc_conv_thr = 1.0d-5
  etot_conv_thr = 1.0d-7
/
&SYSTEM
  ibrav       = 0
  nat         = 2
  ntyp        = 1
  ecutwfc     = 60.0
  ecutrho     = 480.0
  occupations = 'smearing'
  smearing    = 'cold'
  degauss     = 0.01
/
&ELECTRONS
  conv_thr    = 1.0d-10
  mixing_beta = 0.3
/
&IONS
  ion_dynamics = 'bfgs'
/
&CELL
  cell_dynamics = 'bfgs'
  press         = 0.0
/
ATOMIC_SPECIES
  Si  28.0855  Si.upf

CELL_PARAMETERS angstrom
  0.000  2.715  2.715
  2.715  0.000  2.715
  2.715  2.715  0.000

ATOMIC_POSITIONS crystal
  Si  0.00  0.00  0.00
  Si  0.25  0.25  0.25

K_POINTS automatic
  8 8 8  0 0 0
EOF

mpirun -np 4 pw.x -in relax.in > relax.out 2>&1
```

**Step 2: Full workflow (phonon force constants + thermal displacement + sigma^A)**

```python
#!/usr/bin/env python3
"""
Anharmonicity Score (sigma^A) with QE DFT + phonopy.
Publication-quality workflow.

Prerequisites:
  - QE pw.x in PATH
  - Relaxed structure from Step 1 (relax.out)
  - phonopy, pymatgen, ase installed

Reference: Knoop et al., Phys. Rev. Materials 4, 083809 (2020)
"""

import os
import subprocess
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from ase.io import read, write
from ase.io.espresso import read_espresso_out

from pymatgen.core.structure import Structure
from pymatgen.io.ase import AseAtomsAdaptor
from pymatgen.io.phonopy import get_phonopy_structure

import phonopy
from phonopy.interface.qe import read_pw_output

import warnings
warnings.filterwarnings("ignore")

# ============================================================
# 1. CONFIGURATION
# ============================================================

RELAXED_OUT = "relax.out"              # QE relaxation output file
PSEUDO_DIR = "/opt/pseudopotentials/SSSP_1.3.0_PBEsol_efficiency/"
PSEUDOS = {"Si": "Si.upf"}            # Element -> pseudopotential filename
ECUTWFC = 60.0                         # Plane-wave cutoff (Ry)
ECUTRHO = 480.0                        # Charge density cutoff (Ry)
KPOINTS = "4 4 4"                      # K-mesh for supercell SCF (reduced from unit cell)
NPROCS = 4                             # MPI processes for pw.x

MIN_LENGTH = 15.0                      # Supercell min length (A)
DISPLACEMENT = 0.01                    # Finite displacement for force constants (A)
SYMPREC = 1e-5
MESH = [20, 20, 20]                    # Phonon q-mesh

TARGET_TEMPERATURE = 300.0             # Temperature for thermal displacement (K)

WORK_DIR = "/tmp/anharmonicity_qe"
os.makedirs(WORK_DIR, exist_ok=True)

# ============================================================
# 2. READ RELAXED STRUCTURE
# ============================================================

print("=== Reading Relaxed Structure ===")
atoms_relaxed = read(RELAXED_OUT, format="espresso-out", index=-1)
print(f"  Formula: {atoms_relaxed.get_chemical_formula()}")
print(f"  Volume:  {atoms_relaxed.get_volume():.4f} A^3")

adaptor = AseAtomsAdaptor()
pmg_struct = adaptor.get_structure(atoms_relaxed)
phonopy_struct = get_phonopy_structure(pmg_struct)

# ============================================================
# 3. SUPERCELL AND PHONON DISPLACEMENTS
# ============================================================

def get_supercell_matrix(atoms, min_length=15.0):
    cell_lengths = atoms.cell.lengths()
    multiples = np.ceil(min_length / cell_lengths).astype(int)
    multiples = np.maximum(multiples, 1)
    return np.diag(multiples)

supercell_matrix = get_supercell_matrix(atoms_relaxed, min_length=MIN_LENGTH)
print(f"  Supercell matrix: diag({np.diag(supercell_matrix)})")

phonon = phonopy.Phonopy(
    phonopy_struct,
    supercell_matrix=supercell_matrix.tolist(),
    symprec=SYMPREC,
)
phonon.generate_displacements(distance=DISPLACEMENT)
supercells = phonon.supercells_with_displacements
n_disp = len(supercells)
print(f"  Number of displacements: {n_disp}")

# ============================================================
# 4. HELPER: RUN QE SCF AND EXTRACT FORCES
# ============================================================

def write_qe_input(atoms, filename, pseudo_dir, pseudos, ecutwfc, ecutrho, kpoints):
    """Write a QE pw.x SCF input file for force calculation."""
    species = sorted(set(atoms.get_chemical_symbols()))
    nat = len(atoms)
    ntyp = len(species)

    cell = atoms.cell
    positions = atoms.get_scaled_positions()
    symbols = atoms.get_chemical_symbols()

    with open(filename, "w") as f:
        f.write(f"""&CONTROL
  calculation = 'scf'
  prefix      = 'scf'
  outdir      = './tmp'
  pseudo_dir  = '{pseudo_dir}'
  tprnfor     = .true.
  tstress     = .false.
/
&SYSTEM
  ibrav       = 0
  nat         = {nat}
  ntyp        = {ntyp}
  ecutwfc     = {ecutwfc}
  ecutrho     = {ecutrho}
  occupations = 'smearing'
  smearing    = 'cold'
  degauss     = 0.01
/
&ELECTRONS
  conv_thr    = 1.0d-10
  mixing_beta = 0.3
/

ATOMIC_SPECIES
""")
        from ase.data import atomic_masses, chemical_symbols
        for sp in species:
            mass = atomic_masses[chemical_symbols.index(sp)]
            f.write(f"  {sp}  {mass:.4f}  {pseudos[sp]}\n")

        f.write("\nCELL_PARAMETERS angstrom\n")
        for v in cell:
            f.write(f"  {v[0]:.10f}  {v[1]:.10f}  {v[2]:.10f}\n")

        f.write("\nATOMIC_POSITIONS crystal\n")
        for i in range(nat):
            f.write(f"  {symbols[i]}  {positions[i][0]:.10f}  {positions[i][1]:.10f}  {positions[i][2]:.10f}\n")

        f.write(f"\nK_POINTS automatic\n  {kpoints}  0 0 0\n")


def run_qe_forces(atoms, label, work_dir):
    """Run QE SCF and return forces array (natoms, 3) in eV/A."""
    calc_dir = os.path.join(work_dir, label)
    os.makedirs(calc_dir, exist_ok=True)
    os.makedirs(os.path.join(calc_dir, "tmp"), exist_ok=True)

    input_file = os.path.join(calc_dir, "scf.in")
    output_file = os.path.join(calc_dir, "scf.out")

    write_qe_input(atoms, input_file, PSEUDO_DIR, PSEUDOS, ECUTWFC, ECUTRHO, KPOINTS)

    cmd = f"mpirun -np {NPROCS} pw.x -in scf.in > scf.out 2>&1"
    result = subprocess.run(cmd, shell=True, cwd=calc_dir, timeout=7200)

    if result.returncode != 0:
        raise RuntimeError(f"QE calculation failed for {label}. Check {output_file}")

    # Parse forces from QE output
    atoms_out = read(output_file, format="espresso-out", index=-1)
    # QE forces from ASE are in eV/A
    forces = atoms_out.get_forces()
    return forces

# ============================================================
# 5. COMPUTE FORCES FOR FINITE DISPLACEMENTS
# ============================================================

print("\n=== Computing QE Forces for Harmonic Force Constants ===")

forces_list = []
for j, sc in enumerate(supercells):
    print(f"  Displacement {j + 1}/{n_disp}...")
    sc_atoms = adaptor.get_atoms(
        Structure(
            lattice=sc.cell,
            species=sc.symbols,
            coords=sc.scaled_positions,
        )
    )
    forces = run_qe_forces(sc_atoms, f"disp_{j:03d}", WORK_DIR)
    forces_list.append(forces)

phonon.forces = forces_list
phonon.produce_force_constants()
phonon.save(os.path.join(WORK_DIR, "phonopy_harmonic.yaml"))
print("  Harmonic force constants computed and saved.")

# ============================================================
# 6. GENERATE THERMAL DISPLACEMENT AND COMPUTE sigma^A
# ============================================================

print(f"\n=== Thermal Displacement at T = {TARGET_TEMPERATURE} K ===")

phonon.generate_displacements(
    temperature=TARGET_TEMPERATURE,
    number_of_snapshots=1,
    random_seed=42,
)
sc_disp = phonon.supercells_with_displacements[0]
sc_eq = phonon.supercell
n_sc_atoms = len(sc_eq.positions)
print(f"  Supercell has {n_sc_atoms} atoms")

# Convert displaced supercell to ASE Atoms
sc_atoms_disp = adaptor.get_atoms(
    Structure(
        lattice=sc_disp.cell,
        species=sc_disp.symbols,
        coords=sc_disp.scaled_positions,
    )
)

# Compute QE forces on displaced configuration
print("  Computing QE forces on thermally displaced configuration...")
forces_true = run_qe_forces(sc_atoms_disp, "thermal_disp", WORK_DIR)

# Compute displacements
displacements = sc_disp.positions - sc_eq.positions

# Compute harmonic forces: F_i = -sum_j FC[i,j] . u_j
fc = phonon.force_constants
if fc.shape[0] == n_sc_atoms:
    fc_full = fc
else:
    from phonopy.harmonic.force_constants import compact_fc_to_full_fc
    fc_full = compact_fc_to_full_fc(phonon, fc)

forces_harmonic = np.zeros_like(forces_true)
for i in range(n_sc_atoms):
    for j in range(n_sc_atoms):
        forces_harmonic[i] -= fc_full[i, j] @ displacements[j]

# Compute sigma^A
force_diff = forces_true - forces_harmonic
std_diff = np.std(force_diff.flatten())
std_true = np.std(forces_true.flatten())
sigma_A = std_diff / std_true if std_true > 1e-12 else 0.0

print(f"\n  === Anharmonicity Score ===")
print(f"  std(F_DFT):           {std_true:.6f} eV/A")
print(f"  std(F_harmonic):      {np.std(forces_harmonic.flatten()):.6f} eV/A")
print(f"  std(F_DFT - F_harm):  {std_diff:.6f} eV/A")
print(f"  sigma^A = {sigma_A:.4f}")

if sigma_A < 0.2:
    print(f"  --> HARMONIC at {TARGET_TEMPERATURE} K (QHA is reliable)")
elif sigma_A < 0.5:
    print(f"  --> MODERATELY ANHARMONIC at {TARGET_TEMPERATURE} K")
else:
    print(f"  --> STRONGLY ANHARMONIC at {TARGET_TEMPERATURE} K (use MD)")

# ============================================================
# 7. PLOTTING
# ============================================================

print("\n=== Generating Plots ===")

f_true_flat = forces_true.flatten()
f_harm_flat = forces_harmonic.flatten()

# --- Force parity plot ---
fig, ax = plt.subplots(figsize=(6, 6))
ax.scatter(f_true_flat, f_harm_flat, s=3, alpha=0.4, c="steelblue", edgecolors="none")
fmin = min(f_true_flat.min(), f_harm_flat.min())
fmax_val = max(f_true_flat.max(), f_harm_flat.max())
margin = 0.05 * (fmax_val - fmin)
lims = [fmin - margin, fmax_val + margin]
ax.plot(lims, lims, "k--", linewidth=1.0, label="Perfect harmonic")
ax.set_xlabel("DFT Force Component (eV/A)", fontsize=11)
ax.set_ylabel("Harmonic Force Component (eV/A)", fontsize=11)
ax.set_title(f"Force Parity -- $\\sigma^A$ = {sigma_A:.3f} at {TARGET_TEMPERATURE} K", fontsize=12)
ax.set_xlim(lims)
ax.set_ylim(lims)
ax.set_aspect("equal")
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3)
fig.tight_layout()
fig.savefig(os.path.join(WORK_DIR, "force_parity.png"), dpi=150)
plt.close()
print("  Saved: force_parity.png")

# --- Per-atom anharmonicity histogram ---
species_list = list(sc_disp.symbols)
unique_species = sorted(set(species_list))
colors_map = plt.cm.Set1(np.linspace(0, 1, max(len(unique_species), 3)))

per_atom_sigma = np.zeros(n_sc_atoms)
for iatom in range(n_sc_atoms):
    norm_true = np.linalg.norm(forces_true[iatom])
    norm_diff = np.linalg.norm(force_diff[iatom])
    per_atom_sigma[iatom] = norm_diff / norm_true if norm_true > 1e-12 else 0.0

fig, ax = plt.subplots(figsize=(8, 4))
for idx_sp, sp in enumerate(unique_species):
    mask = np.array([s == sp for s in species_list])
    ax.hist(per_atom_sigma[mask], bins=30, alpha=0.6, label=sp,
            color=colors_map[idx_sp], edgecolor="black", linewidth=0.5)
ax.axvline(x=sigma_A, color="red", linestyle="--", linewidth=1.5,
           label=f"Global $\\sigma^A$ = {sigma_A:.3f}")
ax.set_xlabel("Per-Atom Anharmonicity", fontsize=11)
ax.set_ylabel("Count", fontsize=11)
ax.set_title(f"Per-Atom Anharmonicity Distribution at {TARGET_TEMPERATURE} K", fontsize=12)
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)
fig.tight_layout()
fig.savefig(os.path.join(WORK_DIR, "per_atom_anharmonicity.png"), dpi=150)
plt.close()
print("  Saved: per_atom_anharmonicity.png")

print(f"\nDone. All outputs in: {WORK_DIR}")
```

## Key Parameters

| Parameter | Default | Notes |
|---|---|---|
| `TARGET_TEMPERATURE` | 300 K | Temperature at which to evaluate anharmonicity. Higher T gives larger displacements and reveals more anharmonicity. Use the temperature relevant to your application. |
| `MIN_LENGTH` | 15 A | Minimum supercell length for phonon calculation. 15 A for screening; 20 A for publication. Larger supercells improve force constant accuracy. |
| `DISPLACEMENT` | 0.01 A | Finite displacement distance for harmonic force constants. Standard value; 0.01-0.03 A. |
| `MESH` | [20,20,20] | q-mesh for phonon DOS. Affects thermal displacement sampling quality. 20x20x20 is sufficient. |
| `N_SNAPSHOTS` | 1 | Number of thermally displaced snapshots. 1 is standard for sigma^A (as in Knoop et al.). More snapshots reduce statistical noise but are usually unnecessary. |
| `MACE_MODEL` | "medium" | MACE foundation model size. "medium" balances speed and accuracy. Use "large" for better accuracy. |
| `SYMPREC` | 1e-5 | Symmetry detection tolerance. Tighter values find more symmetry operations, reducing the number of displacements. |
| `random_seed` | 42 | Seed for thermal displacement sampling. Fix for reproducibility; vary to assess sensitivity. |

## Interpreting Results

1. **sigma^A < 0.2**: The material is essentially harmonic at this temperature. The quasi-harmonic approximation (QHA) is reliable. Phonon-based thermodynamics (free energy, heat capacity, thermal expansion via QHA) will be accurate. Examples: diamond, Si, Al2O3 at room temperature.

2. **sigma^A = 0.2 - 0.5**: Moderately anharmonic. QHA may still give qualitatively correct results, but quantitative accuracy degrades. Consider validating key predictions (e.g., thermal expansion) against MD. Examples: many metals at moderate temperatures, some perovskites.

3. **sigma^A > 0.5**: Strongly anharmonic. QHA breaks down. Must use molecular dynamics (MD) for thermodynamic properties. Phonon lifetimes are short; thermal conductivity requires explicit anharmonic methods (phono3py or Green-Kubo MD). Examples: PbTe, SnSe, CsPbBr3 near or above room temperature, superionic conductors.

4. **Temperature dependence**: sigma^A generally increases with temperature because larger thermal displacements probe more of the anharmonic potential energy surface. The temperature at which sigma^A crosses 0.5 indicates the approximate upper limit for QHA validity.

5. **Per-atom analysis**: The per-atom anharmonicity histogram reveals which sublattice or atomic species contributes most to anharmonicity. In halide perovskites, for instance, the halide sublattice is often more anharmonic than the metal sublattice.

6. **Force parity plot**: Points on the diagonal indicate harmonic behavior; scatter away from the diagonal indicates anharmonicity. Systematic curvature (banana shape) suggests a strong quartic anharmonic term.

7. **Force residual distribution**: A Gaussian distribution of residuals is expected for thermal fluctuations. Heavy tails or bimodality may indicate rattler modes, soft phonons, or proximity to a structural phase transition.

## Common Issues

| Issue | Cause | Fix |
|---|---|---|
| sigma^A is exactly 0.0 | Forces are zero or structure is not displaced | Check that `TARGET_TEMPERATURE` > 0 and the phonon model has no errors |
| sigma^A > 1.0 | Harmonic model is completely wrong; imaginary modes or wrong force constants | Check for imaginary phonon modes; ensure structure is fully relaxed; increase supercell size |
| Imaginary phonon modes | Structure not at a true minimum, or supercell too small | Re-relax with tighter convergence (`FMAX = 1e-5`); increase `MIN_LENGTH` to 20 A |
| Force constants in compact form | phonopy stores compact FC by default for large supercells | Script handles both compact and full FC; if issues persist, set `phonon.force_constants_type = 'full'` before `produce_force_constants()` |
| sigma^A varies with random seed | Statistical noise from single snapshot | Run 3-5 snapshots and average; for screening, single snapshot is usually sufficient |
| MACE gives different sigma^A than DFT | MLIP inaccuracy for this chemistry or configuration | Compare MACE forces with DFT on a few displaced configurations; if MACE error is comparable to anharmonic signal, use DFT |
| Very slow for large supercells | O(N^2) force constant matrix-vector product | For > 500 atoms, consider using sparse FC representation or phonopy's built-in routines |
| QE calculation fails on displaced supercell | Large displacements at high T cause SCF convergence issues | Reduce `mixing_beta`, increase `electron_maxstep`, or use `startingwfc = 'random'` |
