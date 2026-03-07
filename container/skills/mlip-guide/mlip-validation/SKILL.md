# MLIP Validation

Systematic methods to validate machine learning interatomic potentials (particularly MACE-MP-0) against DFT reference data. Covers energy/force/stress parity, equation of state, phonon frequencies, elastic constants, and surface/defect energies. Essential for establishing when MACE results can be trusted and when DFT fallback is needed.

## When to Use

- Before trusting MACE results on a new or unusual material system
- After fine-tuning or retraining a MACE model on custom data
- Benchmarking MLIP accuracy against DFT for a specific chemistry
- Deciding whether MACE accuracy is sufficient for a given property (EOS, phonons, elasticity, defects)
- Establishing error bars on MLIP predictions for publication
- Comparing multiple MLIP models (MACE small/medium/large, CHGNet, etc.)

## Method Selection

| Validation Type | What it Tests | When to Use |
|---|---|---|
| Energy/Force/Stress Parity (Method A) | Overall prediction accuracy on a test set | First validation step for any new system; provides MAE, RMSE, R-squared |
| Equation of State (Method B) | Accuracy of E-V curve, bulk modulus, equilibrium volume | Validating mechanical/thermodynamic properties; comparing curvature of potential energy surface |
| Phonon Frequencies (Method C) | Force constant accuracy, dynamical stability | Validating vibrational properties; checking for spurious imaginary modes |
| Elastic Constants (Method D) | Stress response to strain; anisotropic mechanical properties | Validating elastic tensor components; checking shear vs bulk accuracy |
| Surface/Defect Energies (Method E) | Accuracy for non-bulk configurations (surfaces, vacancies) | Validating MACE for surface science, catalysis, defect engineering |

## Prerequisites

- **MACE-MP-0**: Pre-installed (`mace-torch` package). No additional installation needed.
- **ASE**: Pre-installed. Used as the interface layer for MACE calculations.
- **pymatgen**: Pre-installed. Used for structure manipulation, symmetry analysis, and elastic tensor fitting.
- **phonopy**: `pip install phonopy seekpath` (needed for Method C).
- **numpy, scipy, matplotlib**: Pre-installed. Used for data analysis and plotting.
- **Quantum ESPRESSO 7.5**: `/opt/qe/bin/pw.x` and `/opt/qe/bin/ph.x` for generating DFT reference data. SSSP pseudopotentials required (see `electronic-structure/scf-relax/SKILL.md`).

## Detailed Steps

---

### Method A: Energy/Force/Stress Parity Plots

Compare MACE predictions against QE DFT on a set of perturbed structures. Generate parity plots (MACE vs DFT) for energies, forces, and stresses, and compute error metrics (MAE, RMSE, R-squared).

```python
#!/usr/bin/env python3
"""
MLIP Validation Method A: Energy/Force/Stress Parity Plots.

Workflow:
  1. Start from a relaxed structure.
  2. Generate perturbed configurations (rattled atoms + strained cells).
  3. Compute energies, forces, and stresses with both MACE and QE.
  4. Generate parity plots and compute error metrics.

Usage:
  - Set INPUT_FILE to your structure (CIF/POSCAR).
  - Set PSEUDO_DIR to your pseudopotential directory.
  - Adjust N_RATTLE and N_STRAIN for test set size.
"""

import os
import re
import json
import subprocess
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from ase.build import bulk
from ase.io import read as ase_read, write as ase_write
from ase.optimize import LBFGS
from ase.constraints import ExpCellFilter

from pymatgen.core.structure import Structure
from pymatgen.io.ase import AseAtomsAdaptor
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer

# --- CONFIGURATION -----------------------------------------------------------
INPUT_FILE = "structure.cif"          # Input structure
MACE_MODEL = "medium"                 # "small", "medium", or "large"
PSEUDO_DIR = "./pseudo"               # QE pseudopotential directory
N_RATTLE = 5                          # Number of rattled configurations
N_STRAIN = 5                          # Number of strained configurations
RATTLE_STD = 0.05                     # Rattle amplitude (Angstrom)
STRAIN_RANGE = 0.03                   # Max isotropic strain magnitude
SUPERCELL = (2, 2, 2)                 # Supercell for test structures
ECUTWFC = 60.0                        # QE plane-wave cutoff (Ry)
ECUTRHO = 480.0                       # QE charge density cutoff (Ry)
K_GRID = (4, 4, 4)                    # k-point grid for supercell
QE_NP = 4                             # Number of MPI processes
OUTPUT_DIR = "validation_parity"
# ------------------------------------------------------------------------------

os.makedirs(OUTPUT_DIR, exist_ok=True)

from mace.calculators import mace_mp
calc_mace = mace_mp(model=MACE_MODEL, device="cpu", default_dtype="float64")

adaptor = AseAtomsAdaptor()
structure = Structure.from_file(INPUT_FILE)
formula = structure.composition.reduced_formula
print(f"Loaded: {formula}")

# --- Step 1: Relax with MACE -------------------------------------------------
atoms_eq = adaptor.get_atoms(structure)
atoms_eq.calc = calc_mace
ecf = ExpCellFilter(atoms_eq, hydrostatic_strain=False)
opt = LBFGS(ecf, logfile=os.path.join(OUTPUT_DIR, "relax.log"))
opt.run(fmax=1e-4, steps=500)
relaxed = adaptor.get_structure(atoms_eq)
print(f"Relaxed: a={relaxed.lattice.a:.4f} A, V={relaxed.volume:.4f} A^3")

# --- Step 2: Generate test configurations ------------------------------------
test_configs = []

# 2a. Rattled configurations
sc = relaxed.copy()
sc.make_supercell(SUPERCELL)
for i in range(N_RATTLE):
    rattled = sc.copy()
    rattled.perturb(RATTLE_STD)
    test_configs.append(("rattle", i, rattled))

# 2b. Strained configurations
strains = np.linspace(-STRAIN_RANGE, STRAIN_RANGE, N_STRAIN)
for i, eps in enumerate(strains):
    strained = sc.copy()
    strained.apply_strain(eps)
    test_configs.append(("strain", i, strained))

print(f"Generated {len(test_configs)} test configurations")

# --- Step 3: MACE predictions ------------------------------------------------
mace_energies = []
mace_forces_all = []
mace_stresses = []

for tag, idx, struct in test_configs:
    atoms = adaptor.get_atoms(struct)
    atoms.calc = calc_mace
    e = atoms.get_potential_energy()
    f = atoms.get_forces()
    s = atoms.get_stress(voigt=True)  # eV/A^3, Voigt [xx,yy,zz,yz,xz,xy]
    mace_energies.append(e / len(atoms))
    mace_forces_all.append(f.flatten())
    mace_stresses.append(s * 160.21766)  # Convert to GPa
    print(f"  MACE {tag}_{idx}: E={e/len(atoms):.6f} eV/atom, "
          f"|F|_max={np.max(np.abs(f)):.4f} eV/A")

# --- Step 4: QE DFT reference calculations -----------------------------------
def run_qe_single(struct, work_dir, label):
    """Run QE SCF on a single structure. Returns (energy_eV, forces, stress_GPa)."""
    os.makedirs(work_dir, exist_ok=True)
    atoms = adaptor.get_atoms(struct)

    from ase.io.espresso import write_espresso_in
    pseudopotentials = {}
    for symbol in set(atoms.get_chemical_symbols()):
        # SSSP naming convention -- adjust if needed
        pseudopotentials[symbol] = f"{symbol}.UPF"

    input_data = {
        "control": {
            "calculation": "scf",
            "outdir": os.path.join(work_dir, "tmp"),
            "pseudo_dir": os.path.abspath(PSEUDO_DIR),
            "tprnfor": True,
            "tstress": True,
        },
        "system": {
            "ecutwfc": ECUTWFC,
            "ecutrho": ECUTRHO,
            "occupations": "smearing",
            "smearing": "cold",
            "degauss": 0.01,
        },
        "electrons": {
            "conv_thr": 1.0e-8,
        },
    }

    input_file = os.path.join(work_dir, "scf.in")
    with open(input_file, "w") as fh:
        write_espresso_in(fh, atoms, input_data=input_data,
                          pseudopotentials=pseudopotentials, kpts=K_GRID)

    output_file = os.path.join(work_dir, "scf.out")
    with open(output_file, "w") as fout:
        subprocess.run(
            ["mpirun", "--allow-run-as-root", "-np", str(QE_NP),
             "pw.x", "-in", input_file],
            stdout=fout, stderr=subprocess.STDOUT, timeout=7200
        )
    return parse_qe_output(output_file, len(atoms))


def parse_qe_output(output_file, n_atoms):
    """Parse energy (eV), forces (eV/A), and stress (GPa) from QE output."""
    energy_ry = None
    forces = []
    stress_kbar = []

    with open(output_file) as f:
        lines = f.readlines()

    for i, line in enumerate(lines):
        if line.strip().startswith("!") and "total energy" in line:
            energy_ry = float(re.search(r"=\s*([-\d.]+)", line).group(1))
        if "Forces acting on atoms" in line:
            forces = []
            for j in range(i + 2, i + 2 + n_atoms):
                parts = lines[j].split()
                fx, fy, fz = float(parts[-3]), float(parts[-2]), float(parts[-1])
                forces.append([fx, fy, fz])
        if "total   stress" in line:
            stress_kbar = []
            for j in range(1, 4):
                parts = lines[i + j].split()
                stress_kbar.append([float(parts[0]), float(parts[1]), float(parts[2])])

    if energy_ry is None:
        raise ValueError(f"Energy not found in {output_file}")

    energy_ev = energy_ry * 13.605693123
    # QE forces are in Ry/bohr; convert to eV/A
    forces_ev_a = np.array(forces) * 13.605693123 / 0.529177249
    # QE stress in kbar -> GPa
    stress_gpa_matrix = np.array(stress_kbar) * 0.1
    # Convert 3x3 to Voigt [xx, yy, zz, yz, xz, xy]
    stress_voigt = np.array([
        stress_gpa_matrix[0, 0], stress_gpa_matrix[1, 1], stress_gpa_matrix[2, 2],
        stress_gpa_matrix[1, 2], stress_gpa_matrix[0, 2], stress_gpa_matrix[0, 1]
    ])

    return energy_ev, forces_ev_a, stress_voigt


dft_energies = []
dft_forces_all = []
dft_stresses = []

for tag, idx, struct in test_configs:
    label = f"{tag}_{idx:03d}"
    work_dir = os.path.join(OUTPUT_DIR, f"qe_{label}")
    print(f"  Running QE {label}...")
    try:
        e_dft, f_dft, s_dft = run_qe_single(struct, work_dir, label)
        dft_energies.append(e_dft / len(struct))
        dft_forces_all.append(f_dft.flatten())
        dft_stresses.append(s_dft)
    except Exception as ex:
        print(f"    QE FAILED for {label}: {ex}")
        dft_energies.append(None)
        dft_forces_all.append(None)
        dft_stresses.append(None)

# --- Step 5: Compute metrics and plot ----------------------------------------
# Filter out failed QE calculations
valid = [i for i in range(len(dft_energies)) if dft_energies[i] is not None]
if len(valid) < 2:
    print("ERROR: Need at least 2 valid DFT results for parity analysis.")
else:
    e_dft = np.array([dft_energies[i] for i in valid])
    e_mace = np.array([mace_energies[i] for i in valid])
    f_dft = np.concatenate([dft_forces_all[i] for i in valid])
    f_mace = np.concatenate([mace_forces_all[i] for i in valid])
    s_dft = np.concatenate([dft_stresses[i] for i in valid])
    s_mace = np.concatenate([mace_stresses[i] for i in valid])

    def metrics(pred, ref):
        diff = pred - ref
        mae = np.mean(np.abs(diff))
        rmse = np.sqrt(np.mean(diff**2))
        ss_res = np.sum(diff**2)
        ss_tot = np.sum((ref - np.mean(ref))**2)
        r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0
        return mae, rmse, r2

    mae_e, rmse_e, r2_e = metrics(e_mace, e_dft)
    mae_f, rmse_f, r2_f = metrics(f_mace, f_dft)
    mae_s, rmse_s, r2_s = metrics(s_mace, s_dft)

    print(f"\n{'='*60}")
    print(f"VALIDATION METRICS (MACE {MACE_MODEL} vs QE DFT)")
    print(f"{'='*60}")
    print(f"  Energy:  MAE = {mae_e*1000:.2f} meV/atom, "
          f"RMSE = {rmse_e*1000:.2f} meV/atom, R^2 = {r2_e:.6f}")
    print(f"  Forces:  MAE = {mae_f*1000:.2f} meV/A,   "
          f"RMSE = {rmse_f*1000:.2f} meV/A,   R^2 = {r2_f:.6f}")
    print(f"  Stress:  MAE = {mae_s:.4f} GPa,      "
          f"RMSE = {rmse_s:.4f} GPa,      R^2 = {r2_s:.6f}")

    # --- Parity plots ---
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # Energy parity
    ax = axes[0]
    ax.scatter(e_dft, e_mace, c="steelblue", s=40, alpha=0.8, edgecolors="k", lw=0.5)
    lims = [min(e_dft.min(), e_mace.min()), max(e_dft.max(), e_mace.max())]
    margin = (lims[1] - lims[0]) * 0.05
    ax.plot([lims[0]-margin, lims[1]+margin], [lims[0]-margin, lims[1]+margin],
            "k--", lw=1, label="y = x")
    ax.set_xlabel("DFT Energy (eV/atom)", fontsize=12)
    ax.set_ylabel("MACE Energy (eV/atom)", fontsize=12)
    ax.set_title(f"Energy Parity\nMAE={mae_e*1000:.1f} meV/atom, R$^2$={r2_e:.4f}",
                 fontsize=11)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_aspect("equal")

    # Force parity
    ax = axes[1]
    subsample = np.random.choice(len(f_dft), min(5000, len(f_dft)), replace=False)
    ax.scatter(f_dft[subsample], f_mace[subsample], c="coral", s=5, alpha=0.3)
    f_lims = [min(f_dft.min(), f_mace.min()), max(f_dft.max(), f_mace.max())]
    f_margin = (f_lims[1] - f_lims[0]) * 0.05
    ax.plot([f_lims[0]-f_margin, f_lims[1]+f_margin],
            [f_lims[0]-f_margin, f_lims[1]+f_margin], "k--", lw=1)
    ax.set_xlabel("DFT Force Component (eV/A)", fontsize=12)
    ax.set_ylabel("MACE Force Component (eV/A)", fontsize=12)
    ax.set_title(f"Force Parity\nMAE={mae_f*1000:.1f} meV/A, R$^2$={r2_f:.4f}",
                 fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.set_aspect("equal")

    # Stress parity
    ax = axes[2]
    ax.scatter(s_dft, s_mace, c="forestgreen", s=40, alpha=0.8, edgecolors="k", lw=0.5)
    s_lims = [min(s_dft.min(), s_mace.min()), max(s_dft.max(), s_mace.max())]
    s_margin = (s_lims[1] - s_lims[0]) * 0.05
    ax.plot([s_lims[0]-s_margin, s_lims[1]+s_margin],
            [s_lims[0]-s_margin, s_lims[1]+s_margin], "k--", lw=1)
    ax.set_xlabel("DFT Stress Component (GPa)", fontsize=12)
    ax.set_ylabel("MACE Stress Component (GPa)", fontsize=12)
    ax.set_title(f"Stress Parity\nMAE={mae_s:.3f} GPa, R$^2$={r2_s:.4f}",
                 fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.set_aspect("equal")

    plt.suptitle(f"MACE-MP-0 ({MACE_MODEL}) vs QE DFT: {formula}", fontsize=14, y=1.02)
    plt.tight_layout()
    plot_file = os.path.join(OUTPUT_DIR, "parity_plots.png")
    plt.savefig(plot_file, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\nParity plots saved to {plot_file}")

    # Save metrics
    results = {
        "formula": formula,
        "mace_model": MACE_MODEL,
        "n_configs": len(valid),
        "energy_MAE_meV_per_atom": float(mae_e * 1000),
        "energy_RMSE_meV_per_atom": float(rmse_e * 1000),
        "energy_R2": float(r2_e),
        "force_MAE_meV_per_A": float(mae_f * 1000),
        "force_RMSE_meV_per_A": float(rmse_f * 1000),
        "force_R2": float(r2_f),
        "stress_MAE_GPa": float(mae_s),
        "stress_RMSE_GPa": float(rmse_s),
        "stress_R2": float(r2_s),
    }
    with open(os.path.join(OUTPUT_DIR, "parity_metrics.json"), "w") as f:
        json.dump(results, f, indent=2)
    print(f"Metrics saved to {OUTPUT_DIR}/parity_metrics.json")
```

---

### Method B: Equation of State Validation

Compare E-V curves and Birch-Murnaghan EOS parameters (V0, B0, B0') from MACE vs QE. This tests whether the MACE potential correctly reproduces the curvature of the potential energy surface.

```python
#!/usr/bin/env python3
"""
MLIP Validation Method B: Equation of State Comparison.

Workflow:
  1. Compute E-V curve with MACE at multiple volumes.
  2. Compute E-V curve with QE at the same volumes.
  3. Fit Birch-Murnaghan EOS to both.
  4. Compare V0, B0, B0' and overlay E-V curves.
"""

import os
import re
import json
import subprocess
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

from ase.optimize import LBFGS
from ase.constraints import ExpCellFilter
from ase.eos import EquationOfState as ASE_EOS
from ase.units import kJ

from pymatgen.core.structure import Structure
from pymatgen.io.ase import AseAtomsAdaptor

# --- CONFIGURATION -----------------------------------------------------------
INPUT_FILE = "structure.cif"
MACE_MODEL = "medium"
PSEUDO_DIR = "./pseudo"
N_POINTS = 9
LINEAR_STRAIN = (-0.05, 0.05)
ECUTWFC = 60.0
ECUTRHO = 480.0
K_GRID = (8, 8, 8)
QE_NP = 4
OUTPUT_DIR = "validation_eos"
# ------------------------------------------------------------------------------

os.makedirs(OUTPUT_DIR, exist_ok=True)

from mace.calculators import mace_mp
calc_mace = mace_mp(model=MACE_MODEL, device="cpu", default_dtype="float64")
adaptor = AseAtomsAdaptor()

structure = Structure.from_file(INPUT_FILE)
formula = structure.composition.reduced_formula
n_atoms = len(structure)

# --- Step 1: Relax with MACE -------------------------------------------------
atoms_eq = adaptor.get_atoms(structure)
atoms_eq.calc = calc_mace
ecf = ExpCellFilter(atoms_eq, hydrostatic_strain=False)
opt = LBFGS(ecf, logfile=os.path.join(OUTPUT_DIR, "relax.log"))
opt.run(fmax=1e-4, steps=500)
relaxed = adaptor.get_structure(atoms_eq)
V0 = relaxed.volume
print(f"Relaxed: {formula}, V0={V0:.4f} A^3")

# --- Step 2: MACE E-V curve --------------------------------------------------
strain_values = np.linspace(LINEAR_STRAIN[0], LINEAR_STRAIN[1], N_POINTS)
mace_volumes = []
mace_energies = []

for eps in strain_values:
    strained = relaxed.copy()
    strained.apply_strain(eps)
    atoms_s = adaptor.get_atoms(strained)
    atoms_s.calc = calc_mace
    opt_s = LBFGS(atoms_s, logfile=os.devnull)
    opt_s.run(fmax=1e-3, steps=300)
    V = atoms_s.get_volume()
    E = atoms_s.get_potential_energy()
    mace_volumes.append(V)
    mace_energies.append(E)

mace_volumes = np.array(mace_volumes)
mace_energies = np.array(mace_energies)
sort_idx = np.argsort(mace_volumes)
mace_volumes = mace_volumes[sort_idx]
mace_energies = mace_energies[sort_idx]

# Fit MACE BM EOS
eos_mace = ASE_EOS(mace_volumes, mace_energies, eos="birchmurnaghan")
v0_mace, e0_mace, B_mace = eos_mace.fit()
B0_mace_gpa = B_mace * 160.21766
print(f"MACE EOS: V0={v0_mace:.4f} A^3, B0={B0_mace_gpa:.2f} GPa")

# --- Step 3: QE E-V curve ----------------------------------------------------
def run_qe_scf_ev(struct, work_dir):
    """Run QE SCF and return (energy_eV, volume_A3)."""
    os.makedirs(work_dir, exist_ok=True)
    atoms = adaptor.get_atoms(struct)

    from ase.io.espresso import write_espresso_in
    pseudopotentials = {s: f"{s}.UPF" for s in set(atoms.get_chemical_symbols())}
    input_data = {
        "control": {
            "calculation": "scf", "outdir": os.path.join(work_dir, "tmp"),
            "pseudo_dir": os.path.abspath(PSEUDO_DIR),
            "tprnfor": True, "tstress": True,
        },
        "system": {
            "ecutwfc": ECUTWFC, "ecutrho": ECUTRHO,
            "occupations": "smearing", "smearing": "cold", "degauss": 0.01,
        },
        "electrons": {"conv_thr": 1.0e-8},
    }
    input_file = os.path.join(work_dir, "scf.in")
    with open(input_file, "w") as fh:
        write_espresso_in(fh, atoms, input_data=input_data,
                          pseudopotentials=pseudopotentials, kpts=K_GRID)
    output_file = os.path.join(work_dir, "scf.out")
    with open(output_file, "w") as fout:
        subprocess.run(
            ["mpirun", "--allow-run-as-root", "-np", str(QE_NP),
             "pw.x", "-in", input_file],
            stdout=fout, stderr=subprocess.STDOUT, timeout=7200
        )

    energy_ry = None
    with open(output_file) as f:
        for line in f:
            if line.strip().startswith("!") and "total energy" in line:
                energy_ry = float(re.search(r"=\s*([-\d.]+)", line).group(1))
    if energy_ry is None:
        raise ValueError(f"Energy not found in {output_file}")
    return energy_ry * 13.605693123, struct.volume


dft_volumes = []
dft_energies = []

for i, eps in enumerate(strain_values):
    strained = relaxed.copy()
    strained.apply_strain(eps)
    work_dir = os.path.join(OUTPUT_DIR, f"qe_vol_{i:03d}")
    print(f"  QE vol point {i+1}/{N_POINTS}: eps={eps:+.4f}...")
    try:
        e_dft, v_dft = run_qe_scf_ev(strained, work_dir)
        dft_volumes.append(v_dft)
        dft_energies.append(e_dft)
    except Exception as ex:
        print(f"    FAILED: {ex}")

dft_volumes = np.array(dft_volumes)
dft_energies = np.array(dft_energies)
sort_idx = np.argsort(dft_volumes)
dft_volumes = dft_volumes[sort_idx]
dft_energies = dft_energies[sort_idx]

# Fit DFT BM EOS
eos_dft = ASE_EOS(dft_volumes, dft_energies, eos="birchmurnaghan")
v0_dft, e0_dft, B_dft = eos_dft.fit()
B0_dft_gpa = B_dft * 160.21766
print(f"DFT EOS:  V0={v0_dft:.4f} A^3, B0={B0_dft_gpa:.2f} GPa")

# --- Step 4: Compare and plot ------------------------------------------------
v0_err = abs(v0_mace - v0_dft) / v0_dft * 100
b0_err = abs(B0_mace_gpa - B0_dft_gpa) / B0_dft_gpa * 100

print(f"\n{'='*60}")
print(f"EOS COMPARISON: {formula}")
print(f"{'='*60}")
print(f"{'Property':<20} {'MACE':>12} {'QE DFT':>12} {'Error (%)':>12}")
print(f"{'-'*60}")
print(f"{'V0 (A^3)':<20} {v0_mace:>12.4f} {v0_dft:>12.4f} {v0_err:>11.2f}%")
print(f"{'B0 (GPa)':<20} {B0_mace_gpa:>12.2f} {B0_dft_gpa:>12.2f} {b0_err:>11.1f}%")

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

# E-V comparison
e_mace_shifted = (mace_energies - mace_energies.min()) / n_atoms * 1000  # meV/atom
e_dft_shifted = (dft_energies - dft_energies.min()) / n_atoms * 1000
ax1.plot(mace_volumes, e_mace_shifted, "s-", color="steelblue", markersize=7,
         label=f"MACE {MACE_MODEL}")
ax1.plot(dft_volumes, e_dft_shifted, "o-", color="crimson", markersize=7,
         label="QE DFT")
ax1.set_xlabel("Volume ($\\AA^3$)", fontsize=12)
ax1.set_ylabel("$\\Delta E$ (meV/atom)", fontsize=12)
ax1.set_title(f"E-V Curves: {formula}", fontsize=13)
ax1.legend(fontsize=11)
ax1.grid(True, alpha=0.3)

# Energy difference plot
if len(mace_volumes) == len(dft_volumes):
    e_diff = (mace_energies / n_atoms - dft_energies / n_atoms) * 1000
    ax2.bar(range(len(e_diff)), e_diff, color="teal", alpha=0.7)
    ax2.axhline(0, color="k", lw=0.5)
    ax2.set_xlabel("Volume Point Index", fontsize=12)
    ax2.set_ylabel("MACE - DFT (meV/atom)", fontsize=12)
    ax2.set_title(f"Energy Residuals: {formula}", fontsize=13)
    ax2.grid(True, alpha=0.3)
else:
    ax2.text(0.5, 0.5, "Mismatched volume grids;\ncannot plot residuals",
             ha="center", va="center", transform=ax2.transAxes, fontsize=12)

plt.tight_layout()
plot_file = os.path.join(OUTPUT_DIR, "eos_comparison.png")
plt.savefig(plot_file, dpi=150, bbox_inches="tight")
plt.close()
print(f"\nPlot saved to {plot_file}")

results = {
    "formula": formula,
    "mace_model": MACE_MODEL,
    "mace_V0_A3": float(v0_mace), "dft_V0_A3": float(v0_dft),
    "V0_error_pct": float(v0_err),
    "mace_B0_GPa": float(B0_mace_gpa), "dft_B0_GPa": float(B0_dft_gpa),
    "B0_error_pct": float(b0_err),
}
with open(os.path.join(OUTPUT_DIR, "eos_comparison.json"), "w") as f:
    json.dump(results, f, indent=2)
print(f"Results saved to {OUTPUT_DIR}/eos_comparison.json")
```

---

### Method C: Phonon Frequency Comparison

Compare phonon band structures and density of states from MACE (finite displacements via phonopy) against QE DFPT. Detect spurious imaginary modes introduced by the MLIP.

```python
#!/usr/bin/env python3
"""
MLIP Validation Method C: Phonon Frequency Comparison.

Workflow:
  1. Compute phonon band structure with MACE + phonopy (finite displacements).
  2. Compute phonon band structure with QE DFPT (ph.x).
  3. Compare phonon dispersions and DOS.
  4. Detect imaginary modes and quantify frequency errors.

Prerequisites: pip install phonopy seekpath
"""

import os
import json
import subprocess
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from ase.optimize import LBFGS
from ase.constraints import ExpCellFilter

from pymatgen.core.structure import Structure
from pymatgen.io.ase import AseAtomsAdaptor
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer

import phonopy
from phonopy.structure.atoms import PhonopyAtoms

# --- CONFIGURATION -----------------------------------------------------------
INPUT_FILE = "structure.cif"
MACE_MODEL = "medium"
PSEUDO_DIR = "./pseudo"
SUPERCELL_DIM = [2, 2, 2]            # Phonon supercell dimensions
ECUTWFC = 60.0
ECUTRHO = 480.0
K_GRID_SCF = (8, 8, 8)               # k-grid for SCF
Q_GRID = (4, 4, 4)                   # q-grid for DFPT phonons
QE_NP = 4
OUTPUT_DIR = "validation_phonon"
# ------------------------------------------------------------------------------

os.makedirs(OUTPUT_DIR, exist_ok=True)

from mace.calculators import mace_mp
calc_mace = mace_mp(model=MACE_MODEL, device="cpu", default_dtype="float64")
adaptor = AseAtomsAdaptor()

structure = Structure.from_file(INPUT_FILE)
formula = structure.composition.reduced_formula

# --- Step 1: Relax with MACE -------------------------------------------------
atoms_eq = adaptor.get_atoms(structure)
atoms_eq.calc = calc_mace
ecf = ExpCellFilter(atoms_eq, hydrostatic_strain=False)
opt = LBFGS(ecf, logfile=os.path.join(OUTPUT_DIR, "relax.log"))
opt.run(fmax=1e-4, steps=500)
relaxed = adaptor.get_structure(atoms_eq)
print(f"Relaxed: {formula}, a={relaxed.lattice.a:.4f} A")

# --- Step 2: MACE phonon band structure (phonopy) ----------------------------
def structure_to_phonopy(struct):
    """Convert pymatgen Structure to PhonopyAtoms."""
    return PhonopyAtoms(
        symbols=[str(s) for s in struct.species],
        cell=struct.lattice.matrix,
        scaled_positions=struct.frac_coords,
    )

unitcell = structure_to_phonopy(relaxed)
ph = phonopy.Phonopy(unitcell, supercell_matrix=np.diag(SUPERCELL_DIM))
ph.generate_displacements(distance=0.01)

supercells = ph.supercells_with_displacements
print(f"MACE phonons: {len(supercells)} displaced supercells")

forces_sets = []
for i, sc in enumerate(supercells):
    atoms_sc = adaptor.get_atoms(Structure(
        lattice=sc.cell, species=sc.symbols,
        coords=sc.scaled_positions, coords_are_cartesian=False
    ))
    atoms_sc.calc = calc_mace
    forces = atoms_sc.get_forces()
    forces_sets.append(forces)

ph.forces = forces_sets
ph.produce_force_constants()

# Get high-symmetry k-path using seekpath
from seekpath import get_path
cell = (relaxed.lattice.matrix, relaxed.frac_coords,
        [s.Z for s in relaxed.species])
path_data = get_path(cell, with_time_reversal=True, symprec=0.01)
labels = []
band_segments = []
current_path = []
label_pairs = path_data["path"]
point_coords = path_data["point_coords"]

for seg_start, seg_end in label_pairs:
    q_start = point_coords[seg_start]
    q_end = point_coords[seg_end]
    band_segments.append([q_start, q_end])
    labels.append((seg_start, seg_end))

ph.run_band_structure(band_segments, with_eigenvectors=False, labels=None)
band_dict = ph.get_band_structure_dict()

mace_distances = []
mace_frequencies = []
for seg in band_dict["distances"]:
    mace_distances.append(np.array(seg))
for seg in band_dict["frequencies"]:
    mace_frequencies.append(np.array(seg))

# DOS
ph.run_mesh([20, 20, 20])
ph.run_total_dos()
dos_dict = ph.get_total_dos_dict()
mace_dos_freq = np.array(dos_dict["frequency_points"])
mace_dos_vals = np.array(dos_dict["total_dos"])

# Check for imaginary modes
min_freq_mace = min(f.min() for f in mace_frequencies)
has_imaginary_mace = min_freq_mace < -0.5  # THz threshold
print(f"MACE min frequency: {min_freq_mace:.3f} THz "
      f"({'IMAGINARY MODES DETECTED' if has_imaginary_mace else 'OK'})")

# --- Step 3: QE DFPT phonon band structure ------------------------------------
qe_dir = os.path.join(OUTPUT_DIR, "qe_phonon")
os.makedirs(qe_dir, exist_ok=True)

# 3a. SCF calculation
atoms_scf = adaptor.get_atoms(relaxed)
from ase.io.espresso import write_espresso_in
pseudopotentials = {s: f"{s}.UPF" for s in set(atoms_scf.get_chemical_symbols())}

scf_input_data = {
    "control": {
        "calculation": "scf", "outdir": os.path.join(qe_dir, "tmp"),
        "pseudo_dir": os.path.abspath(PSEUDO_DIR), "tprnfor": True,
    },
    "system": {
        "ecutwfc": ECUTWFC, "ecutrho": ECUTRHO,
        "occupations": "smearing", "smearing": "cold", "degauss": 0.01,
    },
    "electrons": {"conv_thr": 1.0e-10},
}

scf_in = os.path.join(qe_dir, "scf.in")
with open(scf_in, "w") as fh:
    write_espresso_in(fh, atoms_scf, input_data=scf_input_data,
                      pseudopotentials=pseudopotentials, kpts=K_GRID_SCF)

print("Running QE SCF...")
scf_out = os.path.join(qe_dir, "scf.out")
with open(scf_out, "w") as fout:
    subprocess.run(
        ["mpirun", "--allow-run-as-root", "-np", str(QE_NP),
         "pw.x", "-in", scf_in],
        stdout=fout, stderr=subprocess.STDOUT, timeout=7200
    )

# 3b. Phonon calculation (ph.x)
ph_input = f"""&INPUTPH
  outdir = '{os.path.join(qe_dir, "tmp")}'
  prefix = 'pwscf'
  fildyn = '{os.path.join(qe_dir, "dyn")}'
  ldisp = .true.
  nq1 = {Q_GRID[0]}, nq2 = {Q_GRID[1]}, nq3 = {Q_GRID[2]}
  tr2_ph = 1.0d-14
/
"""
ph_in = os.path.join(qe_dir, "ph.in")
with open(ph_in, "w") as f:
    f.write(ph_input)

print("Running QE ph.x (DFPT)...")
ph_out = os.path.join(qe_dir, "ph.out")
with open(ph_out, "w") as fout:
    subprocess.run(
        ["mpirun", "--allow-run-as-root", "-np", str(QE_NP),
         "ph.x", "-in", ph_in],
        stdout=fout, stderr=subprocess.STDOUT, timeout=36000,
        cwd=qe_dir
    )

# 3c. Fourier interpolation (q2r.x + matdyn.x)
q2r_input = f"""&INPUT
  fildyn = '{os.path.join(qe_dir, "dyn")}'
  flfrc  = '{os.path.join(qe_dir, "fc.dat")}'
/
"""
q2r_in = os.path.join(qe_dir, "q2r.in")
with open(q2r_in, "w") as f:
    f.write(q2r_input)
subprocess.run(["q2r.x", "-in", q2r_in],
               stdout=open(os.path.join(qe_dir, "q2r.out"), "w"),
               stderr=subprocess.STDOUT, timeout=600, cwd=qe_dir)

# Generate q-path for matdyn.x
qpoints_lines = []
for seg_start, seg_end in label_pairs:
    q_s = point_coords[seg_start]
    q_e = point_coords[seg_end]
    qpoints_lines.append(f"  {q_s[0]:.6f} {q_s[1]:.6f} {q_s[2]:.6f}  51")
qpoints_lines.append(f"  {point_coords[label_pairs[-1][1]][0]:.6f} "
                      f"{point_coords[label_pairs[-1][1]][1]:.6f} "
                      f"{point_coords[label_pairs[-1][1]][2]:.6f}  1")

matdyn_input = f"""&INPUT
  asr = 'crystal'
  flfrc = '{os.path.join(qe_dir, "fc.dat")}'
  flfrq = '{os.path.join(qe_dir, "freq.dat")}'
  q_in_band_form = .true.
/
{len(qpoints_lines)}
""" + "\n".join(qpoints_lines)

matdyn_in = os.path.join(qe_dir, "matdyn_band.in")
with open(matdyn_in, "w") as f:
    f.write(matdyn_input)
subprocess.run(["matdyn.x", "-in", matdyn_in],
               stdout=open(os.path.join(qe_dir, "matdyn_band.out"), "w"),
               stderr=subprocess.STDOUT, timeout=600, cwd=qe_dir)

# Parse DFT phonon frequencies
freq_file = os.path.join(qe_dir, "freq.dat")
dft_qpoints = []
dft_freqs_per_q = []

if os.path.exists(freq_file):
    with open(freq_file) as f:
        current_freqs = []
        for line in f:
            line = line.strip()
            if not line:
                if current_freqs:
                    dft_freqs_per_q.append(current_freqs)
                    current_freqs = []
                continue
            parts = line.split()
            if len(parts) >= 4 and parts[0] == "q":
                continue
            try:
                freqs = [float(x) for x in parts]
                current_freqs.extend(freqs)
            except ValueError:
                continue
        if current_freqs:
            dft_freqs_per_q.append(current_freqs)

    # Convert cm^-1 to THz (1 cm^-1 = 0.02998 THz)
    dft_freqs_thz = []
    for flist in dft_freqs_per_q:
        dft_freqs_thz.append([f * 0.02998 for f in flist])

    min_freq_dft = min(min(fl) for fl in dft_freqs_thz)
    has_imaginary_dft = min_freq_dft < -0.5
    print(f"DFT min frequency: {min_freq_dft:.3f} THz "
          f"{'IMAGINARY' if has_imaginary_dft else 'OK'}")

# --- Step 4: Comparison plot -------------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Phonon band structure comparison
ax = axes[0]
offset = 0
for seg_idx, seg_freqs in enumerate(mace_frequencies):
    dists = mace_distances[seg_idx]
    for branch_idx in range(seg_freqs.shape[1]):
        label_m = "MACE" if seg_idx == 0 and branch_idx == 0 else None
        ax.plot(dists + offset, seg_freqs[:, branch_idx], "-",
                color="steelblue", lw=1.5, label=label_m)
    offset += dists[-1]

ax.axhline(0, color="gray", ls=":", lw=0.5)
ax.set_xlabel("q-path", fontsize=12)
ax.set_ylabel("Frequency (THz)", fontsize=12)
ax.set_title(f"Phonon Dispersion: {formula}", fontsize=13)
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3)

# DOS comparison
ax = axes[1]
ax.plot(mace_dos_freq, mace_dos_vals, "-", color="steelblue", lw=1.5, label="MACE")
ax.axvline(0, color="gray", ls=":", lw=0.5)
ax.set_xlabel("Frequency (THz)", fontsize=12)
ax.set_ylabel("DOS (states/THz)", fontsize=12)
ax.set_title(f"Phonon DOS: {formula}", fontsize=13)
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3)

plt.suptitle(f"MACE-MP-0 ({MACE_MODEL}) Phonon Validation: {formula}",
             fontsize=14, y=1.02)
plt.tight_layout()
plot_file = os.path.join(OUTPUT_DIR, "phonon_comparison.png")
plt.savefig(plot_file, dpi=150, bbox_inches="tight")
plt.close()
print(f"\nPlot saved to {plot_file}")

results = {
    "formula": formula,
    "mace_model": MACE_MODEL,
    "supercell": SUPERCELL_DIM,
    "mace_min_freq_THz": float(min_freq_mace),
    "mace_has_imaginary": bool(has_imaginary_mace),
}
if dft_freqs_thz:
    results["dft_min_freq_THz"] = float(min_freq_dft)
    results["dft_has_imaginary"] = bool(has_imaginary_dft)

with open(os.path.join(OUTPUT_DIR, "phonon_comparison.json"), "w") as f:
    json.dump(results, f, indent=2)
print(f"Results saved to {OUTPUT_DIR}/phonon_comparison.json")
```

---

### Method D: Elastic Constants Comparison

Compare the full 6x6 elastic tensor and polycrystalline moduli from MACE vs QE. This tests the accuracy of second-derivative properties (stress response to strain).

```python
#!/usr/bin/env python3
"""
MLIP Validation Method D: Elastic Constants Comparison.

Workflow:
  1. Compute elastic tensor with MACE (stress-strain method).
  2. Compute elastic tensor with QE (stress-strain method).
  3. Compare C_ij components, bulk/shear moduli.
  4. Generate comparison table and heatmap.
"""

import os
import re
import json
import subprocess
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from ase.optimize import LBFGS
from ase.constraints import ExpCellFilter

from pymatgen.core.structure import Structure
from pymatgen.core.tensors import symmetry_reduce
from pymatgen.analysis.elasticity import Strain, Stress, Deformation, ElasticTensor
from pymatgen.io.ase import AseAtomsAdaptor
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer

# --- CONFIGURATION -----------------------------------------------------------
INPUT_FILE = "structure.cif"
MACE_MODEL = "medium"
PSEUDO_DIR = "./pseudo"
STRAIN_MAGNITUDES = [-0.01, -0.005, 0.005, 0.01]
ECUTWFC = 60.0
ECUTRHO = 480.0
K_GRID = (8, 8, 8)
QE_NP = 4
SYMPREC = 0.01
OUTPUT_DIR = "validation_elastic"
# ------------------------------------------------------------------------------

os.makedirs(OUTPUT_DIR, exist_ok=True)

from mace.calculators import mace_mp
calc_mace = mace_mp(model=MACE_MODEL, device="cpu", default_dtype="float64")
adaptor = AseAtomsAdaptor()

structure = Structure.from_file(INPUT_FILE)
formula = structure.composition.reduced_formula

# --- Step 1: Relax with MACE -------------------------------------------------
atoms_eq = adaptor.get_atoms(structure)
atoms_eq.calc = calc_mace
ecf = ExpCellFilter(atoms_eq, hydrostatic_strain=False)
opt = LBFGS(ecf, logfile=os.path.join(OUTPUT_DIR, "relax.log"))
opt.run(fmax=1e-4, steps=500)
relaxed = adaptor.get_structure(atoms_eq)

sga = SpacegroupAnalyzer(relaxed, symprec=SYMPREC)
print(f"Relaxed: {formula}, SG={sga.get_space_group_symbol()}")

# --- Step 2: Generate strain deformations ------------------------------------
strain_states = [
    (1,0,0,0,0,0), (0,1,0,0,0,0), (0,0,1,0,0,0),
    (0,0,0,2,0,0), (0,0,0,0,2,0), (0,0,0,0,0,2),
]
strains = []
for state in strain_states:
    for mag in STRAIN_MAGNITUDES:
        s = Strain.from_voigt(mag * np.array(state))
        if (np.abs(s) > 1e-10).any():
            strains.append(s)

strain_mapping = symmetry_reduce(strains, relaxed, symprec=SYMPREC)
strains = list(strain_mapping.keys())
deformations = [s.get_deformation_matrix() for s in strains]
print(f"Symmetry-reduced deformations: {len(deformations)}")

# --- Step 3: MACE elastic tensor ---------------------------------------------
mace_stresses = []
for idx, deformation in enumerate(deformations):
    deformed = relaxed.copy()
    deformed.apply_strain(Strain.from_deformation(Deformation(deformation)))
    atoms_d = adaptor.get_atoms(deformed)
    atoms_d.calc = calc_mace
    opt_d = LBFGS(atoms_d, logfile=os.devnull)
    opt_d.run(fmax=1e-3, steps=300)
    stress_3x3 = atoms_d.get_stress(voigt=False) * 160.21766  # eV/A^3 -> GPa
    mace_stresses.append(Stress(-stress_3x3))

eq_stress_mace = Stress(-np.array(atoms_eq.get_stress(voigt=False)) * 160.21766)
C_mace = ElasticTensor.from_independent_strains(
    mace_stresses, strains, eq_stress=eq_stress_mace
)

# Symmetrize
symmops = sga.get_symmetry_operations(cartesian=True)
C_mace_sym = np.zeros((6, 6))
for op in symmops:
    C_mace_sym += C_mace.transform(op.rotation_matrix).voigt
C_mace_sym /= len(symmops)
C_mace_tensor = ElasticTensor.from_voigt(C_mace_sym)

K_mace = C_mace_tensor.k_vrh
G_mace = C_mace_tensor.g_vrh
print(f"MACE: K_VRH={K_mace:.2f} GPa, G_VRH={G_mace:.2f} GPa")

# --- Step 4: QE elastic tensor -----------------------------------------------
def run_qe_stress(struct, work_dir):
    """Run QE SCF with ionic relaxation, return stress (3x3, GPa)."""
    os.makedirs(work_dir, exist_ok=True)
    atoms = adaptor.get_atoms(struct)

    from ase.io.espresso import write_espresso_in
    pseudopotentials = {s: f"{s}.UPF" for s in set(atoms.get_chemical_symbols())}
    input_data = {
        "control": {
            "calculation": "relax", "outdir": os.path.join(work_dir, "tmp"),
            "pseudo_dir": os.path.abspath(PSEUDO_DIR),
            "tprnfor": True, "tstress": True, "forc_conv_thr": 1.0e-5,
        },
        "system": {
            "ecutwfc": ECUTWFC, "ecutrho": ECUTRHO,
            "occupations": "smearing", "smearing": "cold", "degauss": 0.01,
        },
        "electrons": {"conv_thr": 1.0e-10},
        "ions": {"ion_dynamics": "bfgs"},
    }
    input_file = os.path.join(work_dir, "scf.in")
    with open(input_file, "w") as fh:
        write_espresso_in(fh, atoms, input_data=input_data,
                          pseudopotentials=pseudopotentials, kpts=K_GRID)
    output_file = os.path.join(work_dir, "scf.out")
    with open(output_file, "w") as fout:
        subprocess.run(
            ["mpirun", "--allow-run-as-root", "-np", str(QE_NP),
             "pw.x", "-in", input_file],
            stdout=fout, stderr=subprocess.STDOUT, timeout=7200
        )

    # Parse stress
    stress = None
    with open(output_file) as f:
        lines = f.readlines()
    for i, line in enumerate(lines):
        if "total   stress" in line:
            s = []
            for j in range(1, 4):
                parts = lines[i+j].split()
                s.append([float(parts[0]), float(parts[1]), float(parts[2])])
            stress = np.array(s) * 0.1  # kbar -> GPa
    if stress is None:
        raise ValueError(f"Stress not found in {output_file}")
    return stress


dft_stresses = []
for idx, deformation in enumerate(deformations):
    deformed = relaxed.copy()
    deformed.apply_strain(Strain.from_deformation(Deformation(deformation)))
    work_dir = os.path.join(OUTPUT_DIR, f"qe_deform_{idx:03d}")
    print(f"  QE deformation {idx+1}/{len(deformations)}...")
    try:
        stress_gpa = run_qe_stress(deformed, work_dir)
        dft_stresses.append(Stress(-stress_gpa))
    except Exception as ex:
        print(f"    FAILED: {ex}")
        dft_stresses.append(None)

# Filter valid results
valid_idx = [i for i, s in enumerate(dft_stresses) if s is not None]
valid_strains = [strains[i] for i in valid_idx]
valid_dft_stresses = [dft_stresses[i] for i in valid_idx]

if len(valid_strains) >= 6:
    eq_stress_dft = Stress(np.zeros((3, 3)))
    C_dft = ElasticTensor.from_independent_strains(
        valid_dft_stresses, valid_strains, eq_stress=eq_stress_dft
    )
    C_dft_sym = np.zeros((6, 6))
    for op in symmops:
        C_dft_sym += C_dft.transform(op.rotation_matrix).voigt
    C_dft_sym /= len(symmops)
    C_dft_tensor = ElasticTensor.from_voigt(C_dft_sym)

    K_dft = C_dft_tensor.k_vrh
    G_dft = C_dft_tensor.g_vrh
    print(f"DFT:  K_VRH={K_dft:.2f} GPa, G_VRH={G_dft:.2f} GPa")

    # --- Step 5: Comparison ---------------------------------------------------
    C_diff = C_mace_sym - C_dft_sym
    C_diff_pct = np.where(np.abs(C_dft_sym) > 1, C_diff / C_dft_sym * 100, 0)

    print(f"\n{'='*70}")
    print(f"ELASTIC TENSOR COMPARISON: {formula}")
    print(f"{'='*70}")
    print(f"{'Component':<12} {'MACE (GPa)':>12} {'DFT (GPa)':>12} "
          f"{'Diff (GPa)':>12} {'Error (%)':>10}")
    print(f"{'-'*70}")
    voigt_labels = ["C11","C12","C13","C14","C15","C16",
                    "C22","C23","C24","C25","C26",
                    "C33","C34","C35","C36",
                    "C44","C45","C46","C55","C56","C66"]
    idx_pairs = []
    for i in range(6):
        for j in range(i, 6):
            idx_pairs.append((i, j))

    for (i, j), lbl in zip(idx_pairs, voigt_labels):
        m_val = C_mace_sym[i, j]
        d_val = C_dft_sym[i, j]
        diff = m_val - d_val
        pct = diff / d_val * 100 if abs(d_val) > 1 else 0
        if abs(m_val) > 1 or abs(d_val) > 1:
            print(f"{lbl:<12} {m_val:>12.2f} {d_val:>12.2f} "
                  f"{diff:>12.2f} {pct:>9.1f}%")

    K_err = abs(K_mace - K_dft) / K_dft * 100 if K_dft > 0 else 0
    G_err = abs(G_mace - G_dft) / G_dft * 100 if G_dft > 0 else 0
    print(f"\n{'K_VRH (GPa)':<12} {K_mace:>12.2f} {K_dft:>12.2f} "
          f"{K_mace-K_dft:>12.2f} {K_err:>9.1f}%")
    print(f"{'G_VRH (GPa)':<12} {G_mace:>12.2f} {G_dft:>12.2f} "
          f"{G_mace-G_dft:>12.2f} {G_err:>9.1f}%")

    # --- Heatmap of differences ---
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    for ax, data, title in zip(axes,
        [C_mace_sym, C_dft_sym, C_diff],
        [f"MACE ({MACE_MODEL})", "QE DFT", "Difference (MACE - DFT)"]):
        im = ax.imshow(data, cmap="RdBu_r", aspect="equal",
                       vmin=-max(abs(data.min()), abs(data.max())),
                       vmax=max(abs(data.min()), abs(data.max())))
        ax.set_xticks(range(6))
        ax.set_yticks(range(6))
        labels_v = ["1","2","3","4","5","6"]
        ax.set_xticklabels(labels_v)
        ax.set_yticklabels(labels_v)
        ax.set_xlabel("j")
        ax.set_ylabel("i")
        ax.set_title(f"C$_{{ij}}$ (GPa): {title}", fontsize=11)
        for ii in range(6):
            for jj in range(6):
                ax.text(jj, ii, f"{data[ii,jj]:.1f}", ha="center", va="center",
                        fontsize=7, color="black" if abs(data[ii,jj]) < data.max()*0.5
                        else "white")
        plt.colorbar(im, ax=ax, shrink=0.8)

    plt.suptitle(f"Elastic Tensor Comparison: {formula}", fontsize=14, y=1.02)
    plt.tight_layout()
    plot_file = os.path.join(OUTPUT_DIR, "elastic_comparison.png")
    plt.savefig(plot_file, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\nPlot saved to {plot_file}")

    results = {
        "formula": formula,
        "mace_model": MACE_MODEL,
        "C_mace_GPa": C_mace_sym.tolist(),
        "C_dft_GPa": C_dft_sym.tolist(),
        "K_mace_GPa": float(K_mace), "K_dft_GPa": float(K_dft),
        "K_error_pct": float(K_err),
        "G_mace_GPa": float(G_mace), "G_dft_GPa": float(G_dft),
        "G_error_pct": float(G_err),
        "max_Cij_error_GPa": float(np.max(np.abs(C_diff))),
    }
    with open(os.path.join(OUTPUT_DIR, "elastic_comparison.json"), "w") as f:
        json.dump(results, f, indent=2)
    print(f"Results saved to {OUTPUT_DIR}/elastic_comparison.json")
else:
    print("ERROR: Insufficient valid QE deformation results for tensor fitting.")
```

---

### Method E: Surface/Defect Energy Validation

Compare surface energies and vacancy formation energies from MACE vs QE DFT. These test MACE accuracy for non-bulk configurations that may be outside the training distribution.

```python
#!/usr/bin/env python3
"""
MLIP Validation Method E: Surface and Defect Energy Comparison.

Workflow:
  Part 1 -- Surface Energy:
    1. Build slab models for common facets.
    2. Compute surface energy with MACE and QE.
    3. Compare results.
  Part 2 -- Vacancy Formation Energy:
    1. Create vacancy in a supercell.
    2. Compute vacancy formation energy with MACE and QE.
    3. Compare results.
"""

import os
import re
import json
import subprocess
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from ase.optimize import LBFGS
from ase.constraints import ExpCellFilter, FixAtoms

from pymatgen.core.structure import Structure
from pymatgen.core.surface import SlabGenerator
from pymatgen.io.ase import AseAtomsAdaptor

# --- CONFIGURATION -----------------------------------------------------------
INPUT_FILE = "structure.cif"
MACE_MODEL = "medium"
PSEUDO_DIR = "./pseudo"
MILLER_INDICES = [(1,0,0), (1,1,0), (1,1,1)]  # Surface facets to test
SLAB_THICKNESS = 10.0                           # Angstrom
VACUUM_THICKNESS = 15.0                         # Angstrom
SUPERCELL_DEFECT = (2, 2, 2)                    # Supercell for vacancy
ECUTWFC = 60.0
ECUTRHO = 480.0
K_GRID_BULK = (8, 8, 8)
K_GRID_SLAB = (4, 4, 1)                        # Reduced k_z for slab
K_GRID_DEFECT = (2, 2, 2)                      # For supercell
QE_NP = 4
OUTPUT_DIR = "validation_surface_defect"
# ------------------------------------------------------------------------------

os.makedirs(OUTPUT_DIR, exist_ok=True)

from mace.calculators import mace_mp
calc_mace = mace_mp(model=MACE_MODEL, device="cpu", default_dtype="float64")
adaptor = AseAtomsAdaptor()

structure = Structure.from_file(INPUT_FILE)
formula = structure.composition.reduced_formula

# --- Helper: QE SCF energy ---------------------------------------------------
def qe_energy(struct, work_dir, kpts):
    """Run QE SCF and return total energy in eV."""
    os.makedirs(work_dir, exist_ok=True)
    atoms = adaptor.get_atoms(struct)

    from ase.io.espresso import write_espresso_in
    pseudopotentials = {s: f"{s}.UPF" for s in set(atoms.get_chemical_symbols())}
    input_data = {
        "control": {
            "calculation": "scf", "outdir": os.path.join(work_dir, "tmp"),
            "pseudo_dir": os.path.abspath(PSEUDO_DIR),
            "tprnfor": True, "tstress": True,
        },
        "system": {
            "ecutwfc": ECUTWFC, "ecutrho": ECUTRHO,
            "occupations": "smearing", "smearing": "cold", "degauss": 0.01,
        },
        "electrons": {"conv_thr": 1.0e-8},
    }
    input_file = os.path.join(work_dir, "scf.in")
    with open(input_file, "w") as fh:
        write_espresso_in(fh, atoms, input_data=input_data,
                          pseudopotentials=pseudopotentials, kpts=kpts)
    output_file = os.path.join(work_dir, "scf.out")
    with open(output_file, "w") as fout:
        subprocess.run(
            ["mpirun", "--allow-run-as-root", "-np", str(QE_NP),
             "pw.x", "-in", input_file],
            stdout=fout, stderr=subprocess.STDOUT, timeout=7200
        )
    energy_ry = None
    with open(output_file) as f:
        for line in f:
            if line.strip().startswith("!") and "total energy" in line:
                energy_ry = float(re.search(r"=\s*([-\d.]+)", line).group(1))
    if energy_ry is None:
        raise ValueError(f"Energy not found in {output_file}")
    return energy_ry * 13.605693123

# --- Helper: MACE energy after relaxation ------------------------------------
def mace_energy(struct, relax_cell=False, fix_bottom=False):
    """Compute energy with MACE, optionally relaxing."""
    atoms = adaptor.get_atoms(struct)
    atoms.calc = calc_mace

    if fix_bottom:
        z_coords = atoms.positions[:, 2]
        z_min, z_max = z_coords.min(), z_coords.max()
        z_mid = (z_min + z_max) / 2
        fix_mask = z_coords < z_mid
        atoms.set_constraint(FixAtoms(mask=fix_mask))

    if relax_cell:
        ecf = ExpCellFilter(atoms, hydrostatic_strain=False)
        opt = LBFGS(ecf, logfile=os.devnull)
        opt.run(fmax=1e-3, steps=300)
    else:
        opt = LBFGS(atoms, logfile=os.devnull)
        opt.run(fmax=1e-3, steps=300)

    return atoms.get_potential_energy()


# === PART 1: Surface Energy ===================================================
print(f"\n{'='*60}")
print(f"PART 1: Surface Energy Validation")
print(f"{'='*60}")

# Bulk energy per atom (MACE)
atoms_bulk = adaptor.get_atoms(structure)
atoms_bulk.calc = calc_mace
ecf = ExpCellFilter(atoms_bulk, hydrostatic_strain=False)
opt = LBFGS(ecf, logfile=os.devnull)
opt.run(fmax=1e-4, steps=500)
relaxed = adaptor.get_structure(atoms_bulk)
e_bulk_mace = atoms_bulk.get_potential_energy() / len(atoms_bulk)
print(f"MACE bulk energy: {e_bulk_mace:.6f} eV/atom")

# Bulk energy per atom (QE)
try:
    e_bulk_total_dft = qe_energy(relaxed, os.path.join(OUTPUT_DIR, "qe_bulk"), K_GRID_BULK)
    e_bulk_dft = e_bulk_total_dft / len(relaxed)
    print(f"DFT  bulk energy: {e_bulk_dft:.6f} eV/atom")
except Exception as ex:
    print(f"DFT bulk energy FAILED: {ex}")
    e_bulk_dft = None

surface_results = []
for hkl in MILLER_INDICES:
    hkl_str = "".join(str(i) for i in hkl)
    print(f"\n--- Surface ({hkl_str}) ---")

    slabgen = SlabGenerator(
        relaxed, hkl, min_slab_size=SLAB_THICKNESS,
        min_vacuum_size=VACUUM_THICKNESS, center_slab=True,
        in_unit_planes=False, primitive=True, lll_reduce=True
    )
    slabs = slabgen.get_slabs(symmetrize=True)

    if not slabs:
        print(f"  No symmetric slab found for ({hkl_str})")
        continue

    slab = slabs[0]
    n_slab = len(slab)
    area = slab.surface_area  # A^2
    print(f"  Slab: {n_slab} atoms, area={area:.2f} A^2")

    # MACE surface energy
    e_slab_mace = mace_energy(slab, relax_cell=False, fix_bottom=False)
    gamma_mace = (e_slab_mace - n_slab * e_bulk_mace) / (2 * area)
    gamma_mace_j_m2 = gamma_mace * 16.0218  # eV/A^2 -> J/m^2
    print(f"  MACE surface energy: {gamma_mace_j_m2:.4f} J/m^2")

    # QE surface energy
    gamma_dft_j_m2 = None
    if e_bulk_dft is not None:
        try:
            e_slab_dft = qe_energy(slab,
                                   os.path.join(OUTPUT_DIR, f"qe_slab_{hkl_str}"),
                                   K_GRID_SLAB)
            gamma_dft = (e_slab_dft - n_slab * e_bulk_dft) / (2 * area)
            gamma_dft_j_m2 = gamma_dft * 16.0218
            print(f"  DFT  surface energy: {gamma_dft_j_m2:.4f} J/m^2")
        except Exception as ex:
            print(f"  DFT surface energy FAILED: {ex}")

    result = {
        "hkl": list(hkl),
        "n_atoms": n_slab,
        "area_A2": float(area),
        "gamma_mace_J_m2": float(gamma_mace_j_m2),
    }
    if gamma_dft_j_m2 is not None:
        result["gamma_dft_J_m2"] = float(gamma_dft_j_m2)
        result["error_J_m2"] = float(abs(gamma_mace_j_m2 - gamma_dft_j_m2))
        result["error_pct"] = float(abs(gamma_mace_j_m2 - gamma_dft_j_m2)
                                     / gamma_dft_j_m2 * 100) if gamma_dft_j_m2 != 0 else 0
    surface_results.append(result)


# === PART 2: Vacancy Formation Energy =========================================
print(f"\n{'='*60}")
print(f"PART 2: Vacancy Formation Energy Validation")
print(f"{'='*60}")

# Build supercell
sc = relaxed.copy()
sc.make_supercell(SUPERCELL_DEFECT)
n_sc = len(sc)
print(f"Supercell: {n_sc} atoms")

# MACE perfect supercell energy
e_perfect_mace = mace_energy(sc, relax_cell=True)
print(f"MACE perfect supercell: {e_perfect_mace:.6f} eV")

# QE perfect supercell energy
try:
    e_perfect_dft = qe_energy(sc, os.path.join(OUTPUT_DIR, "qe_perfect"), K_GRID_DEFECT)
    print(f"DFT  perfect supercell: {e_perfect_dft:.6f} eV")
except Exception as ex:
    print(f"DFT perfect supercell FAILED: {ex}")
    e_perfect_dft = None

# Create vacancy (remove first atom of each species type)
vacancy_results = []
unique_species = list(set(str(s) for s in sc.species))

for species in unique_species:
    # Find the first atom of this species
    for idx, site in enumerate(sc):
        if str(site.specie) == species:
            vac_idx = idx
            break

    vac_struct = sc.copy()
    vac_struct.remove_sites([vac_idx])
    print(f"\n--- Vacancy: {species} (removed site {vac_idx}) ---")
    print(f"  Defect supercell: {len(vac_struct)} atoms")

    # MACE vacancy formation energy
    e_vac_mace = mace_energy(vac_struct, relax_cell=False)
    ef_mace = e_vac_mace - (n_sc - 1) * e_bulk_mace
    print(f"  MACE vacancy formation energy: {ef_mace:.4f} eV")

    # QE vacancy formation energy
    ef_dft = None
    if e_bulk_dft is not None:
        try:
            e_vac_dft = qe_energy(vac_struct,
                                  os.path.join(OUTPUT_DIR, f"qe_vac_{species}"),
                                  K_GRID_DEFECT)
            ef_dft = e_vac_dft - (n_sc - 1) * e_bulk_dft
            print(f"  DFT  vacancy formation energy: {ef_dft:.4f} eV")
        except Exception as ex:
            print(f"  DFT vacancy formation energy FAILED: {ex}")

    result = {"species": species, "ef_mace_eV": float(ef_mace)}
    if ef_dft is not None:
        result["ef_dft_eV"] = float(ef_dft)
        result["error_eV"] = float(abs(ef_mace - ef_dft))
        result["error_pct"] = float(abs(ef_mace - ef_dft) / abs(ef_dft) * 100
                                     ) if ef_dft != 0 else 0
    vacancy_results.append(result)


# === Summary and Visualization ================================================
print(f"\n{'='*60}")
print(f"VALIDATION SUMMARY: {formula}")
print(f"{'='*60}")

if surface_results:
    print(f"\n{'Surface':<10} {'MACE (J/m2)':>12} {'DFT (J/m2)':>12} {'Error (%)':>10}")
    print(f"{'-'*50}")
    for r in surface_results:
        hkl_s = "(" + "".join(str(i) for i in r["hkl"]) + ")"
        if "gamma_dft_J_m2" in r:
            print(f"{hkl_s:<10} {r['gamma_mace_J_m2']:>12.4f} "
                  f"{r['gamma_dft_J_m2']:>12.4f} {r['error_pct']:>9.1f}%")
        else:
            print(f"{hkl_s:<10} {r['gamma_mace_J_m2']:>12.4f} {'N/A':>12} {'N/A':>10}")

if vacancy_results:
    print(f"\n{'Vacancy':<10} {'MACE (eV)':>12} {'DFT (eV)':>12} {'Error (%)':>10}")
    print(f"{'-'*50}")
    for r in vacancy_results:
        if "ef_dft_eV" in r:
            print(f"{r['species']:<10} {r['ef_mace_eV']:>12.4f} "
                  f"{r['ef_dft_eV']:>12.4f} {r['error_pct']:>9.1f}%")
        else:
            print(f"{r['species']:<10} {r['ef_mace_eV']:>12.4f} {'N/A':>12} {'N/A':>10}")

# Bar chart comparison
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Surface energies
ax = axes[0]
if surface_results and any("gamma_dft_J_m2" in r for r in surface_results):
    labels = ["(" + "".join(str(i) for i in r["hkl"]) + ")" for r in surface_results]
    mace_vals = [r["gamma_mace_J_m2"] for r in surface_results]
    dft_vals = [r.get("gamma_dft_J_m2", 0) for r in surface_results]
    x = np.arange(len(labels))
    w = 0.35
    ax.bar(x - w/2, mace_vals, w, color="steelblue", label="MACE")
    ax.bar(x + w/2, dft_vals, w, color="crimson", label="QE DFT")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=11)
    ax.set_ylabel("Surface Energy (J/m$^2$)", fontsize=12)
    ax.set_title(f"Surface Energies: {formula}", fontsize=13)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3, axis="y")
else:
    ax.text(0.5, 0.5, "No DFT surface data", ha="center", va="center",
            transform=ax.transAxes, fontsize=12)

# Vacancy formation energies
ax = axes[1]
if vacancy_results and any("ef_dft_eV" in r for r in vacancy_results):
    labels = [r["species"] for r in vacancy_results]
    mace_vals = [r["ef_mace_eV"] for r in vacancy_results]
    dft_vals = [r.get("ef_dft_eV", 0) for r in vacancy_results]
    x = np.arange(len(labels))
    w = 0.35
    ax.bar(x - w/2, mace_vals, w, color="steelblue", label="MACE")
    ax.bar(x + w/2, dft_vals, w, color="crimson", label="QE DFT")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=11)
    ax.set_ylabel("Formation Energy (eV)", fontsize=12)
    ax.set_title(f"Vacancy Formation Energies: {formula}", fontsize=13)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3, axis="y")
else:
    ax.text(0.5, 0.5, "No DFT vacancy data", ha="center", va="center",
            transform=ax.transAxes, fontsize=12)

plt.tight_layout()
plot_file = os.path.join(OUTPUT_DIR, "surface_defect_comparison.png")
plt.savefig(plot_file, dpi=150, bbox_inches="tight")
plt.close()
print(f"\nPlot saved to {plot_file}")

# Save all results
all_results = {
    "formula": formula,
    "mace_model": MACE_MODEL,
    "surface_results": surface_results,
    "vacancy_results": vacancy_results,
}
with open(os.path.join(OUTPUT_DIR, "surface_defect_comparison.json"), "w") as f:
    json.dump(all_results, f, indent=2)
print(f"Results saved to {OUTPUT_DIR}/surface_defect_comparison.json")
```

## Key Parameters

| Parameter | Acceptable Threshold | Warning Threshold | Notes |
|---|---|---|---|
| Energy MAE | < 5 meV/atom | 5--20 meV/atom | > 20 meV/atom means MACE is unreliable for this system |
| Energy RMSE | < 10 meV/atom | 10--30 meV/atom | RMSE penalizes outliers more than MAE |
| Force MAE | < 50 meV/A | 50--100 meV/A | > 100 meV/A means relaxation results are untrustworthy |
| Force RMSE | < 80 meV/A | 80--150 meV/A | Large RMSE indicates occasional large force errors |
| Stress MAE | < 0.5 GPa | 0.5--2.0 GPa | Stress errors propagate into elastic constant errors |
| Energy R-squared | > 0.999 | 0.99--0.999 | < 0.99 means poor correlation with DFT |
| Force R-squared | > 0.98 | 0.95--0.98 | Forces are harder to predict than energies |
| V0 error | < 1% | 1--3% | Equilibrium volume from EOS fit |
| B0 error | < 10% | 10--20% | Bulk modulus from EOS fit |
| Elastic C_ij error | < 15% | 15--30% | Individual elastic tensor components |
| K_VRH error | < 10% | 10--20% | Voigt-Reuss-Hill bulk modulus |
| G_VRH error | < 15% | 15--30% | Shear modulus is typically less accurate than bulk |
| Phonon frequency error | < 0.5 THz | 0.5--1.5 THz | At zone center and high-symmetry points |
| Imaginary modes (spurious) | None | Any spurious imaginary | If MACE shows imaginary modes that DFT does not, the potential is unreliable |
| Surface energy error | < 0.1 J/m^2 | 0.1--0.3 J/m^2 | Absolute error; relative error < 15% is acceptable |
| Vacancy formation energy error | < 0.3 eV | 0.3--0.5 eV | > 0.5 eV means MACE is unreliable for defect energetics |

## Interpreting Results

**When MACE is reliable (use with confidence):**
- Energy MAE < 5 meV/atom and R-squared > 0.999
- Force MAE < 50 meV/A and R-squared > 0.98
- EOS V0 and B0 errors < 5%
- No spurious imaginary phonon modes
- Surface/vacancy energy errors < 15%
- The material is well-represented in the Materials Project training data (common elements, standard oxidation states, not extreme conditions)

**When MACE needs caution (validate further):**
- Energy MAE 5--20 meV/atom
- Force MAE 50--100 meV/A
- B0 error 10--20%
- Some elastic constants show > 20% error while others are accurate
- Minor frequency shifts in phonon bands but correct topology
- The material contains less common elements or unusual bonding environments

**When DFT is needed (MACE is unreliable):**
- Energy MAE > 20 meV/atom or R-squared < 0.99
- Force MAE > 100 meV/A
- MACE predicts imaginary phonon modes that DFT does not (or vice versa)
- EOS V0 or B0 error > 20%
- Surface or vacancy energies differ by > 50% from DFT
- The material contains rare elements (fewer than ~50 structures in MP), exotic oxidation states, or is under extreme pressure/temperature conditions
- Electronic properties are needed (band gap, DOS, charge density) -- MLIPs cannot predict these

**Validation hierarchy (recommended order):**
1. Start with Method A (parity plots) -- cheapest, gives overall accuracy picture
2. If energies/forces pass, run Method B (EOS) -- tests curvature of PES
3. If EOS passes, run Method C (phonons) -- tests higher derivatives (force constants)
4. Methods D and E (elasticity, surfaces/defects) -- run for specific applications

## Common Issues

| Issue | Solution |
|---|---|
| QE calculations fail or do not converge | Check pseudopotentials exist in PSEUDO_DIR. Increase ecutwfc. Reduce mixing_beta for metals. Check that structure is reasonable (no overlapping atoms). |
| MACE gives NaN or extremely large energies | Structure may be unphysical (atoms too close). Increase rattle amplitude gradually. Use float64 precision. |
| Parity plot shows systematic offset in energies | This is expected -- MACE and QE use different energy references. Compare relative energies only (shift to common reference). |
| Force parity has large scatter but good energy parity | Forces are derivatives of energy and amplify noise. Use MACE "large" model for better force accuracy. |
| EOS fit fails or gives unreasonable B0 | Ensure at least 7 volume points spanning +/-5% linear strain. Check for unconverged QE calculations at extreme volumes. |
| Phonopy calculation gives all-zero frequencies | Force constants not computed properly. Check that displacements were applied and forces were collected for all supercells. |
| MACE phonons have small imaginary modes near Gamma | Often a numerical artifact from insufficient supercell size. Increase SUPERCELL_DIM. If DFT also shows them, the structure may be genuinely unstable. |
| Elastic tensor is not symmetric | Normal for raw data. The symmetrization step (averaging over space group operations) fixes this. |
| Surface energy is negative | Bulk energy reference is wrong (e.g., different calculator or insufficient convergence). Re-relax bulk with tighter fmax. |
| Vacancy formation energy differs wildly | Supercell may be too small (defect-defect interaction). Increase SUPERCELL_DEFECT to (3,3,3) or larger. |
| QE stress has wrong sign convention | QE uses compression-positive convention. The scripts negate QE stresses before passing to pymatgen. If results look inverted, check the sign convention. |
| Comparison is unfair because MACE and QE use different structures | Always use the same atomic positions for both. The scripts above use MACE-relaxed structures for both MACE and QE single-point calculations. |
