# Advanced MACE Usage

## When to Use

- You have custom DFT data and want to fine-tune MACE-MP-0 for improved accuracy on your specific system.
- You want to use multi-fidelity training (mix different levels of DFT data or experimental data).
- You need uncertainty quantification via committee models for reliability assessment.
- You are performing high-throughput screening of many structures and need efficient batch calculations.
- You want to compare foundation model predictions vs fine-tuned model predictions.
- You need to decide whether the universal MACE-MP-0 is sufficient or fine-tuning is warranted.

## Method Selection

| Task | Method | When |
|---|---|---|
| Use MACE-MP-0 as-is | Method A: Foundation model | First attempt on any new system. Often sufficient for common materials. |
| Fine-tune on custom data | Method B: Fine-tuning | When foundation model error is too large (> 50 meV/atom or > 200 meV/A). |
| Multi-fidelity training | Method C: Multi-fidelity | When you have a mix of cheap (GGA) and expensive (hybrid, CCSD(T)) data. |
| Uncertainty quantification | Method D: Committee models | When you need error bars on predictions or active learning. |
| High-throughput screening | Method E: Batch calculations | Screening 100+ structures rapidly. |

## Prerequisites

- MACE-MP-0: pre-installed (`mace-torch`).
- For fine-tuning: `mace-torch` with training capabilities.
- For committee models: multiple MACE models (train N models with different random seeds).
- DFT training data: energies, forces, and optionally stresses in ASE-compatible format (extxyz).
- Python: `numpy`, `scipy`, `matplotlib`, `ase`, `pymatgen`, `torch` (all pre-installed).

```bash
# Verify MACE installation
python3 -c "import mace; print(mace.__version__)"

# For fine-tuning, ensure mace_run_train is available
which mace_run_train || echo "mace_run_train not found -- install from source if needed"
```

---

## Detailed Steps

### Method A: Foundation Model Assessment

Before fine-tuning, always assess the foundation model on your system.

```python
#!/usr/bin/env python3
"""
Assess MACE-MP-0 foundation model on a target system.
Compute energies, forces, stress for a set of structures
and compare with DFT reference data.
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from ase.io import read
from ase.optimize import BFGS
from ase.constraints import ExpCellFilter
from mace.calculators import mace_mp
import warnings
warnings.filterwarnings("ignore")

# ============================================================
# CONFIGURATION
# ============================================================

STRUCTURES_FILE = "training_data.extxyz"  # Your DFT data with energies/forces
MACE_MODEL = "medium"                     # "small", "medium", "large"

# ============================================================
# LOAD MODEL AND DATA
# ============================================================

calc = mace_mp(model=MACE_MODEL, device="cpu", default_dtype="float64")

# Read structures with DFT reference data
# extxyz format should have energy in info dict and forces in arrays
try:
    structures = read(STRUCTURES_FILE, index=":")
    print(f"Loaded {len(structures)} structures from {STRUCTURES_FILE}")
except FileNotFoundError:
    # Generate example structures for demonstration
    from ase.build import bulk
    print("No training data found. Using generated Si structures for demo.")
    structures = []
    for a in np.linspace(5.2, 5.7, 11):
        atoms = bulk("Si", "diamond", a=a)
        # These would normally come with DFT energies/forces
        atoms.info["REF_energy"] = None  # placeholder
        structures.append(atoms)

# ============================================================
# EVALUATE MACE ON ALL STRUCTURES
# ============================================================

results = []
for i, atoms in enumerate(structures):
    atoms_copy = atoms.copy()
    atoms_copy.calc = calc

    mace_energy = atoms_copy.get_potential_energy()
    mace_forces = atoms_copy.get_forces()
    mace_stress = atoms_copy.get_stress()

    entry = {
        "idx": i,
        "n_atoms": len(atoms),
        "formula": atoms.get_chemical_formula(),
        "mace_energy": mace_energy / len(atoms),
        "mace_forces": mace_forces,
        "mace_stress": mace_stress,
    }

    # If DFT reference is available
    if "REF_energy" in atoms.info and atoms.info["REF_energy"] is not None:
        entry["dft_energy"] = atoms.info["REF_energy"] / len(atoms)
    if "REF_forces" in atoms.arrays:
        entry["dft_forces"] = atoms.arrays["REF_forces"]

    results.append(entry)
    if (i + 1) % 50 == 0:
        print(f"  Evaluated {i+1}/{len(structures)}")

print(f"  Evaluated {len(structures)} structures total.")

# ============================================================
# COMPUTE ERROR METRICS
# ============================================================

has_dft = all("dft_energy" in r for r in results)

if has_dft:
    dft_e = np.array([r["dft_energy"] for r in results])
    mace_e = np.array([r["mace_energy"] for r in results])

    e_mae = np.mean(np.abs(mace_e - dft_e)) * 1000  # meV/atom
    e_rmse = np.sqrt(np.mean((mace_e - dft_e)**2)) * 1000
    e_max = np.max(np.abs(mace_e - dft_e)) * 1000

    print(f"\n=== Energy Errors ===")
    print(f"  MAE:  {e_mae:.2f} meV/atom")
    print(f"  RMSE: {e_rmse:.2f} meV/atom")
    print(f"  MAX:  {e_max:.2f} meV/atom")

    has_forces = all("dft_forces" in r for r in results)
    if has_forces:
        all_dft_f = np.concatenate([r["dft_forces"].flatten() for r in results])
        all_mace_f = np.concatenate([r["mace_forces"].flatten() for r in results])

        f_mae = np.mean(np.abs(all_mace_f - all_dft_f)) * 1000
        f_rmse = np.sqrt(np.mean((all_mace_f - all_dft_f)**2)) * 1000

        print(f"\n=== Force Errors ===")
        print(f"  MAE:  {f_mae:.2f} meV/A")
        print(f"  RMSE: {f_rmse:.2f} meV/A")

    # Decision guidance
    print(f"\n=== Recommendation ===")
    if e_mae < 10 and (not has_forces or f_mae < 50):
        print(f"  Foundation model accuracy is GOOD for this system.")
        print(f"  Fine-tuning is unlikely to improve results significantly.")
    elif e_mae < 30 and (not has_forces or f_mae < 100):
        print(f"  Foundation model accuracy is ACCEPTABLE.")
        print(f"  Fine-tuning could improve results, especially for forces.")
    else:
        print(f"  Foundation model accuracy is INSUFFICIENT.")
        print(f"  Fine-tuning is RECOMMENDED for this system.")

    # Parity plot
    fig, axes = plt.subplots(1, 2 if has_forces else 1, figsize=(12 if has_forces else 6, 5))
    if not has_forces:
        axes = [axes]

    axes[0].scatter(dft_e, mace_e, s=15, alpha=0.7, color="steelblue")
    lims = [min(dft_e.min(), mace_e.min()), max(dft_e.max(), mace_e.max())]
    axes[0].plot(lims, lims, "k--", linewidth=1)
    axes[0].set_xlabel("DFT Energy (eV/atom)", fontsize=12)
    axes[0].set_ylabel("MACE Energy (eV/atom)", fontsize=12)
    axes[0].set_title(f"Energy Parity (MAE={e_mae:.1f} meV/atom)", fontsize=13)
    axes[0].set_aspect("equal")
    axes[0].grid(True, alpha=0.3)

    if has_forces:
        # Subsample forces for plotting
        n_plot = min(5000, len(all_dft_f))
        idx = np.random.choice(len(all_dft_f), n_plot, replace=False)
        axes[1].scatter(all_dft_f[idx], all_mace_f[idx], s=3, alpha=0.3, color="coral")
        flims = [all_dft_f.min(), all_dft_f.max()]
        axes[1].plot(flims, flims, "k--", linewidth=1)
        axes[1].set_xlabel("DFT Forces (eV/A)", fontsize=12)
        axes[1].set_ylabel("MACE Forces (eV/A)", fontsize=12)
        axes[1].set_title(f"Force Parity (MAE={f_mae:.1f} meV/A)", fontsize=13)
        axes[1].set_aspect("equal")
        axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("foundation_model_assessment.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\nSaved: foundation_model_assessment.png")
else:
    print("\nNo DFT reference data found. Cannot compute errors.")
    print("Provide extxyz with REF_energy in info and REF_forces in arrays.")
```

### Method B: Fine-Tuning MACE on Custom DFT Data

```python
#!/usr/bin/env python3
"""
Fine-tune MACE-MP-0 on custom DFT data.
Uses the MACE fine-tuning API to adapt the foundation model
to a specific chemistry or configuration space.
"""
import os
import subprocess
import numpy as np
from ase.io import read, write

# ============================================================
# CONFIGURATION
# ============================================================

TRAIN_FILE = "train.extxyz"      # Training data (80% of your DFT data)
VALID_FILE = "valid.extxyz"      # Validation data (20% of your DFT data)
FOUNDATION_MODEL = "medium"       # MACE-MP-0 model to fine-tune
OUTPUT_DIR = "/tmp/mace_finetune"
MODEL_NAME = "mace_finetuned"

# Training hyperparameters
MAX_NUM_EPOCHS = 200
LR = 0.001                        # Learning rate (lower than training from scratch)
BATCH_SIZE = 5
PATIENCE = 20                     # Early stopping patience
ENERGY_WEIGHT = 1.0
FORCES_WEIGHT = 100.0             # Force weight (higher = prioritize force accuracy)
STRESS_WEIGHT = 10.0

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ============================================================
# STEP 1: PREPARE TRAINING DATA
# ============================================================

def prepare_extxyz(dft_data_file, train_file, valid_file, train_frac=0.8):
    """
    Split DFT data into training and validation sets.

    The extxyz file must have:
    - energy in atoms.info['REF_energy'] (total energy in eV)
    - forces in atoms.arrays['REF_forces'] (eV/A)
    - stress in atoms.info['REF_stress'] (eV/A^3, Voigt) [optional]
    """
    structures = read(dft_data_file, index=":")
    np.random.shuffle(structures)

    n_train = int(len(structures) * train_frac)
    train_set = structures[:n_train]
    valid_set = structures[n_train:]

    # Convert to MACE format
    # MACE expects: energy, forces, stress as standard keys
    for atoms in train_set + valid_set:
        if "REF_energy" in atoms.info:
            atoms.info["energy"] = atoms.info["REF_energy"]
        if "REF_forces" in atoms.arrays:
            atoms.arrays["forces"] = atoms.arrays["REF_forces"]
        if "REF_stress" in atoms.info:
            atoms.info["stress"] = atoms.info["REF_stress"]

    write(train_file, train_set)
    write(valid_file, valid_set)

    print(f"Training set: {len(train_set)} structures -> {train_file}")
    print(f"Validation set: {len(valid_set)} structures -> {valid_file}")

    return len(train_set), len(valid_set)


# Check if training data exists
if os.path.exists("dft_data.extxyz"):
    n_train, n_valid = prepare_extxyz(
        "dft_data.extxyz",
        os.path.join(OUTPUT_DIR, TRAIN_FILE),
        os.path.join(OUTPUT_DIR, VALID_FILE),
    )
else:
    print("No training data (dft_data.extxyz) found.")
    print("Create extxyz with energy/forces from your DFT calculations.")
    print("Example format:")
    print('  atoms.info["REF_energy"] = total_energy_eV')
    print('  atoms.arrays["REF_forces"] = forces_array_eV_per_A')

# ============================================================
# STEP 2: FINE-TUNE MACE
# ============================================================

# MACE fine-tuning command
finetune_cmd = [
    "mace_run_train",
    "--name", MODEL_NAME,
    "--foundation_model", FOUNDATION_MODEL,
    "--train_file", os.path.join(OUTPUT_DIR, TRAIN_FILE),
    "--valid_file", os.path.join(OUTPUT_DIR, VALID_FILE),
    "--energy_key", "energy",
    "--forces_key", "forces",
    "--stress_key", "stress",
    "--energy_weight", str(ENERGY_WEIGHT),
    "--forces_weight", str(FORCES_WEIGHT),
    "--stress_weight", str(STRESS_WEIGHT),
    "--max_num_epochs", str(MAX_NUM_EPOCHS),
    "--lr", str(LR),
    "--batch_size", str(BATCH_SIZE),
    "--patience", str(PATIENCE),
    "--device", "cpu",
    "--default_dtype", "float64",
    "--model_dir", OUTPUT_DIR,
    "--results_dir", os.path.join(OUTPUT_DIR, "results"),
    "--log_dir", os.path.join(OUTPUT_DIR, "logs"),
    "--seed", "42",
]

print(f"\n=== Fine-tuning Command ===")
print(" ".join(finetune_cmd))
print(f"\nTo run fine-tuning:")
print(f"  cd {OUTPUT_DIR}")
print(f"  {' '.join(finetune_cmd)}")

# Uncomment to run directly:
# result = subprocess.run(finetune_cmd, capture_output=True, text=True)
# print(result.stdout)
# if result.returncode != 0:
#     print(f"ERROR: {result.stderr}")

# ============================================================
# STEP 3: USE FINE-TUNED MODEL
# ============================================================

print(f"""
=== After Fine-Tuning ===

Load the fine-tuned model:

    from mace.calculators import MACECalculator

    calc = MACECalculator(
        model_paths="{OUTPUT_DIR}/{MODEL_NAME}.model",
        device="cpu",
        default_dtype="float64",
    )

    atoms.calc = calc
    energy = atoms.get_potential_energy()
    forces = atoms.get_forces()
""")
```

### Method C: Multi-Fidelity Training

```python
#!/usr/bin/env python3
"""
Multi-fidelity training with MACE.
Combine data from different DFT levels (PBE, PBEsol, HSE06, etc.)
to train a more accurate model using transfer learning.
"""
import numpy as np
from ase.io import read, write

# ============================================================
# STRATEGY
# ============================================================

print("""
=== Multi-Fidelity Training Strategy ===

Level 0 (cheapest):  PBE with small basis / coarse k-grid  (many structures)
Level 1 (medium):    PBE with converged basis               (moderate count)
Level 2 (expensive): HSE06 / r2SCAN / CCSD(T)              (few structures)

Approach:
1. Train/fine-tune MACE on Level 0 data (many structures)
2. Fine-tune that model on Level 1 data (fewer but better)
3. Fine-tune again on Level 2 data (few but high accuracy)

Each fine-tuning step inherits knowledge from the previous level.
The final model is as accurate as Level 2 but benefits from
the broader coverage of Level 0.
""")

# ============================================================
# EXAMPLE: Tag data with fidelity level
# ============================================================

def tag_fidelity(extxyz_file, fidelity_level, output_file):
    """
    Tag structures with a fidelity level for multi-fidelity training.

    Parameters
    ----------
    extxyz_file : str
        Input extxyz file with DFT data.
    fidelity_level : int
        0 = low, 1 = medium, 2 = high fidelity.
    output_file : str
        Output file with fidelity tags.
    """
    structures = read(extxyz_file, index=":")
    for atoms in structures:
        atoms.info["fidelity"] = fidelity_level
        # Optionally weight by fidelity
        atoms.info["weight"] = {0: 0.1, 1: 1.0, 2: 10.0}[fidelity_level]
    write(output_file, structures)
    print(f"Tagged {len(structures)} structures as fidelity={fidelity_level}")


# Example workflow:
# tag_fidelity("pbe_coarse.extxyz", 0, "level0.extxyz")
# tag_fidelity("pbe_converged.extxyz", 1, "level1.extxyz")
# tag_fidelity("hse06.extxyz", 2, "level2.extxyz")

# Fine-tuning sequence:
# 1. mace_run_train --foundation_model medium --train_file level0.extxyz ...
# 2. mace_run_train --foundation_model level0_model.model --train_file level1.extxyz ...
# 3. mace_run_train --foundation_model level1_model.model --train_file level2.extxyz ...

print("""
=== Multi-Fidelity Fine-Tuning Commands ===

# Step 1: Fine-tune on cheap data (broad coverage)
mace_run_train --name level0_model \\
    --foundation_model medium \\
    --train_file level0_train.extxyz \\
    --valid_file level0_valid.extxyz \\
    --max_num_epochs 100 --lr 0.001 --device cpu

# Step 2: Fine-tune on medium data (transfer from step 1)
mace_run_train --name level1_model \\
    --foundation_model ./level0_model.model \\
    --train_file level1_train.extxyz \\
    --valid_file level1_valid.extxyz \\
    --max_num_epochs 100 --lr 0.0005 --device cpu

# Step 3: Fine-tune on expensive data (transfer from step 2)
mace_run_train --name level2_model \\
    --foundation_model ./level1_model.model \\
    --train_file level2_train.extxyz \\
    --valid_file level2_valid.extxyz \\
    --max_num_epochs 50 --lr 0.0001 --device cpu

The learning rate decreases at each level because the model
is already close to the target and only needs fine adjustment.
""")
```

### Method D: Committee Models for Uncertainty Quantification

```python
#!/usr/bin/env python3
"""
Committee of MACE models for uncertainty quantification.
Train N models with different random seeds, then use the
spread of predictions as an uncertainty estimate.
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from ase.io import read
from ase.build import bulk
from ase.optimize import BFGS
from ase.constraints import ExpCellFilter
from mace.calculators import mace_mp, MACECalculator
import warnings
warnings.filterwarnings("ignore")

# ============================================================
# OPTION 1: COMMITTEE FROM FOUNDATION MODELS (QUICK)
# ============================================================

print("=== Option 1: Committee from Different Model Sizes ===")
print("Uses small/medium/large as a rough committee.\n")

def committee_from_foundation(atoms):
    """
    Use MACE-MP-0 small/medium/large as a committee.
    The spread gives a rough uncertainty estimate.
    """
    models = {}
    for size in ["small", "medium", "large"]:
        calc = mace_mp(model=size, device="cpu", default_dtype="float64")
        atoms_copy = atoms.copy()
        atoms_copy.calc = calc
        e = atoms_copy.get_potential_energy() / len(atoms_copy)
        f = atoms_copy.get_forces()
        models[size] = {"energy": e, "forces": f}

    energies = [models[s]["energy"] for s in models]
    e_mean = np.mean(energies)
    e_std = np.std(energies)

    # Force uncertainty: RMS of force standard deviations
    forces_all = np.array([models[s]["forces"] for s in models])
    f_std = np.std(forces_all, axis=0)
    f_rms_std = np.sqrt(np.mean(f_std**2))

    return {
        "e_mean": e_mean,
        "e_std": e_std,
        "f_rms_std": f_rms_std,
        "models": models,
    }


# Example
atoms = bulk("Si", "diamond", a=5.43)
result = committee_from_foundation(atoms)
print(f"Si (diamond, a=5.43 A):")
print(f"  Energy: {result['e_mean']:.6f} +/- {result['e_std']*1000:.2f} meV/atom")
print(f"  Force uncertainty: {result['f_rms_std']*1000:.2f} meV/A")

# ============================================================
# OPTION 2: COMMITTEE FROM FINE-TUNED MODELS
# ============================================================

print("\n=== Option 2: Committee from Fine-Tuned Models ===")
print("Train N models with different random seeds.\n")

print("""
Training commands for committee of 4 models:

for seed in 42 123 456 789; do
    mace_run_train --name model_seed${seed} \\
        --foundation_model medium \\
        --train_file train.extxyz \\
        --valid_file valid.extxyz \\
        --max_num_epochs 200 \\
        --seed ${seed} \\
        --device cpu \\
        --model_dir ./committee/
done
""")


def committee_prediction(atoms, model_paths):
    """
    Predict with a committee of fine-tuned MACE models.

    Parameters
    ----------
    atoms : ASE Atoms
        Structure to evaluate.
    model_paths : list of str
        Paths to .model files.

    Returns
    -------
    dict with mean, std of energy and forces.
    """
    energies = []
    forces_all = []

    for path in model_paths:
        calc = MACECalculator(model_paths=path, device="cpu", default_dtype="float64")
        atoms_copy = atoms.copy()
        atoms_copy.calc = calc
        e = atoms_copy.get_potential_energy() / len(atoms_copy)
        f = atoms_copy.get_forces()
        energies.append(e)
        forces_all.append(f)

    energies = np.array(energies)
    forces_all = np.array(forces_all)

    return {
        "e_mean": np.mean(energies),
        "e_std": np.std(energies),
        "f_mean": np.mean(forces_all, axis=0),
        "f_std": np.std(forces_all, axis=0),
        "f_rms_std": np.sqrt(np.mean(np.std(forces_all, axis=0)**2)),
    }


# Usage example (uncomment when models are trained):
# model_paths = [f"./committee/model_seed{s}.model" for s in [42, 123, 456, 789]]
# result = committee_prediction(atoms, model_paths)
# print(f"Energy: {result['e_mean']:.6f} +/- {result['e_std']*1000:.2f} meV/atom")

# ============================================================
# OPTION 3: UNCERTAINTY-GUIDED SCREENING
# ============================================================

print("\n=== Uncertainty-Guided Screening ===")

def screen_with_uncertainty(structures, calc_list_or_sizes=["small", "medium", "large"],
                             e_threshold_meV=30, f_threshold_meV_A=100):
    """
    Screen structures and flag those with high uncertainty for DFT validation.

    Parameters
    ----------
    structures : list of ASE Atoms
        Structures to screen.
    calc_list_or_sizes : list
        Model sizes (foundation) or paths to fine-tuned models.
    e_threshold_meV : float
        Energy uncertainty threshold above which to flag for DFT.
    f_threshold_meV_A : float
        Force uncertainty threshold.

    Returns
    -------
    confident : list of Atoms with low uncertainty
    uncertain : list of Atoms needing DFT validation
    """
    calcs = []
    for item in calc_list_or_sizes:
        if item in ("small", "medium", "large"):
            calcs.append(mace_mp(model=item, device="cpu", default_dtype="float64"))
        else:
            calcs.append(MACECalculator(model_paths=item, device="cpu",
                                         default_dtype="float64"))

    confident = []
    uncertain = []

    for i, atoms in enumerate(structures):
        energies = []
        forces_all = []
        for calc in calcs:
            a = atoms.copy()
            a.calc = calc
            energies.append(a.get_potential_energy() / len(a))
            forces_all.append(a.get_forces())

        e_std = np.std(energies) * 1000  # meV/atom
        f_rms_std = np.sqrt(np.mean(np.std(forces_all, axis=0)**2)) * 1000  # meV/A

        atoms.info["mace_e_mean"] = np.mean(energies)
        atoms.info["mace_e_std_meV"] = e_std
        atoms.info["mace_f_rms_std_meV_A"] = f_rms_std

        if e_std > e_threshold_meV or f_rms_std > f_threshold_meV_A:
            uncertain.append(atoms)
        else:
            confident.append(atoms)

        if (i + 1) % 20 == 0:
            print(f"  Screened {i+1}/{len(structures)}: "
                  f"{len(confident)} confident, {len(uncertain)} uncertain")

    print(f"\n  Total: {len(confident)} confident, {len(uncertain)} uncertain")
    print(f"  Run DFT on the {len(uncertain)} uncertain structures for validation.")

    return confident, uncertain


# Example:
# structures = read("candidates.extxyz", index=":")
# confident, uncertain = screen_with_uncertainty(structures)
# write("confident.extxyz", confident)
# write("needs_dft.extxyz", uncertain)
```

### Method E: Batch Calculations with MACE

```python
#!/usr/bin/env python3
"""
High-throughput batch calculations with MACE.
Efficiently process many structures for screening.
"""
import numpy as np
import time
from ase.io import read, write
from ase.build import bulk
from ase.optimize import BFGS
from ase.constraints import ExpCellFilter
from mace.calculators import mace_mp
import warnings
warnings.filterwarnings("ignore")

# ============================================================
# BATCH ENERGY EVALUATION
# ============================================================

def batch_evaluate(structures, model="medium", properties=("energy", "forces")):
    """
    Evaluate MACE on a batch of structures.

    Parameters
    ----------
    structures : list of ASE Atoms
    model : str
        MACE model size.
    properties : tuple
        Properties to compute.

    Returns
    -------
    list of dicts with results.
    """
    calc = mace_mp(model=model, device="cpu", default_dtype="float64")
    results = []

    t_start = time.time()
    for i, atoms in enumerate(structures):
        atoms_copy = atoms.copy()
        atoms_copy.calc = calc

        entry = {
            "idx": i,
            "formula": atoms.get_chemical_formula(),
            "n_atoms": len(atoms),
            "volume": atoms.get_volume(),
        }

        if "energy" in properties:
            entry["energy"] = atoms_copy.get_potential_energy()
            entry["energy_per_atom"] = entry["energy"] / len(atoms)

        if "forces" in properties:
            forces = atoms_copy.get_forces()
            entry["max_force"] = np.sqrt((forces**2).sum(axis=1)).max()

        if "stress" in properties:
            entry["stress"] = atoms_copy.get_stress()
            entry["pressure_GPa"] = -np.trace(atoms_copy.get_stress(voigt=False)) / 3 * 160.2177

        results.append(entry)

        if (i + 1) % 100 == 0:
            elapsed = time.time() - t_start
            rate = (i + 1) / elapsed
            print(f"  {i+1}/{len(structures)} ({rate:.1f} structures/s)")

    elapsed = time.time() - t_start
    print(f"  Completed {len(structures)} structures in {elapsed:.1f}s "
          f"({len(structures)/elapsed:.1f} structures/s)")

    return results


# ============================================================
# BATCH RELAXATION
# ============================================================

def batch_relax(structures, model="medium", fmax=0.05, steps=200,
                relax_cell=True, logfile=None):
    """
    Relax a batch of structures with MACE.

    Parameters
    ----------
    structures : list of ASE Atoms
    model : str
    fmax : float
        Force convergence (eV/A).
    steps : int
        Max optimization steps.
    relax_cell : bool
        If True, relax both positions and cell.

    Returns
    -------
    relaxed : list of ASE Atoms (relaxed)
    results : list of dicts with energies and convergence info.
    """
    calc = mace_mp(model=model, device="cpu", default_dtype="float64")
    relaxed = []
    results = []

    t_start = time.time()
    for i, atoms in enumerate(structures):
        atoms_copy = atoms.copy()
        atoms_copy.calc = calc

        try:
            if relax_cell:
                ecf = ExpCellFilter(atoms_copy)
                opt = BFGS(ecf, logfile=logfile)
            else:
                opt = BFGS(atoms_copy, logfile=logfile)

            opt.run(fmax=fmax, steps=steps)
            converged = opt.converged()
        except Exception as e:
            converged = False
            print(f"  WARNING: Relaxation failed for structure {i}: {e}")

        relaxed.append(atoms_copy)
        results.append({
            "idx": i,
            "formula": atoms.get_chemical_formula(),
            "converged": converged,
            "energy_per_atom": atoms_copy.get_potential_energy() / len(atoms_copy),
            "max_force": np.sqrt((atoms_copy.get_forces()**2).sum(axis=1)).max(),
            "volume": atoms_copy.get_volume(),
            "a": atoms_copy.cell.cellpar()[0],
        })

        if (i + 1) % 50 == 0:
            elapsed = time.time() - t_start
            print(f"  Relaxed {i+1}/{len(structures)} ({elapsed:.0f}s)")

    elapsed = time.time() - t_start
    n_converged = sum(1 for r in results if r["converged"])
    print(f"  Completed: {n_converged}/{len(structures)} converged in {elapsed:.1f}s")

    return relaxed, results


# ============================================================
# EXAMPLE: SCREEN ALLOY COMPOSITIONS
# ============================================================

def screen_binary_alloy(elem1, elem2, crystal, a_range, n_compositions=5):
    """
    Screen binary alloy compositions for stability.
    """
    from ase.build import bulk as ase_bulk

    structures = []
    for a in np.linspace(a_range[0], a_range[1], 5):
        # Pure elements
        s1 = ase_bulk(elem1, crystal, a=a)
        s1.info["composition"] = f"{elem1}_100"
        structures.append(s1)

        s2 = ase_bulk(elem2, crystal, a=a)
        s2.info["composition"] = f"{elem2}_100"
        structures.append(s2)

    print(f"Generated {len(structures)} test structures")

    # Evaluate
    results = batch_evaluate(structures)

    # Print results
    print(f"\n{'Formula':>12} {'E/atom (eV)':>14} {'Max Force':>12}")
    print("-" * 42)
    for r in results:
        print(f"{r['formula']:>12} {r['energy_per_atom']:>14.6f} {r['max_force']:>12.6f}")

    return results


# Example usage:
print("=== Batch Evaluation Example ===")
test_structures = [bulk("Si", "diamond", a=a) for a in np.linspace(5.2, 5.7, 20)]
results = batch_evaluate(test_structures)

print(f"\n{'a (A)':>8} {'E/atom (eV)':>14}")
for i, r in enumerate(results):
    a = test_structures[i].cell.cellpar()[0]
    print(f"{a:>8.3f} {r['energy_per_atom']:>14.6f}")
```

### Method F: Foundation Model vs Fine-Tuned Comparison

```python
#!/usr/bin/env python3
"""
Compare foundation MACE-MP-0 vs fine-tuned model.
Systematic benchmark on lattice constant, elastic constants, phonons.
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from ase.build import bulk
from ase.optimize import BFGS
from ase.constraints import ExpCellFilter
from ase.eos import EquationOfState
from ase.units import kJ
from mace.calculators import mace_mp
import warnings
warnings.filterwarnings("ignore")

def benchmark_model(calc, label, element="Si", crystal="diamond", a_guess=5.43):
    """Run full benchmark on a MACE calculator."""
    print(f"\n--- {label} ---")

    # 1. Lattice constant
    atoms = bulk(element, crystal, a=a_guess)
    atoms.calc = calc
    ecf = ExpCellFilter(atoms)
    opt = BFGS(ecf, logfile=None)
    opt.run(fmax=0.001, steps=200)
    a_eq = atoms.cell.cellpar()[0]
    e_eq = atoms.get_potential_energy() / len(atoms)
    print(f"  Lattice constant: {a_eq:.4f} A")
    print(f"  Energy/atom: {e_eq:.6f} eV")

    # 2. Bulk modulus via E-V curve
    volumes, energies = [], []
    for s in np.linspace(0.94, 1.06, 13):
        a = bulk(element, crystal, a=a_eq * s)
        a.calc = calc
        volumes.append(a.get_volume())
        energies.append(a.get_potential_energy())

    eos = EquationOfState(volumes, energies, eos="birchmurnaghan")
    try:
        v0, e0, B = eos.fit()
        B_GPa = B / kJ * 1e24
        print(f"  Bulk modulus: {B_GPa:.1f} GPa")
    except Exception:
        B_GPa = None
        print(f"  Bulk modulus: fit failed")

    return {"label": label, "a": a_eq, "e": e_eq, "B": B_GPa,
            "volumes": volumes, "energies": energies}


# Benchmark foundation models
results = []
for size in ["small", "medium", "large"]:
    calc = mace_mp(model=size, device="cpu", default_dtype="float64")
    r = benchmark_model(calc, f"MACE-MP-0 ({size})")
    results.append(r)

# If fine-tuned model exists, benchmark it too
# from mace.calculators import MACECalculator
# calc_ft = MACECalculator(model_paths="./mace_finetuned.model",
#                           device="cpu", default_dtype="float64")
# r = benchmark_model(calc_ft, "Fine-tuned MACE")
# results.append(r)

# Print comparison table
print("\n" + "=" * 60)
print(f"{'Model':<25} {'a (A)':>10} {'B (GPa)':>10} {'E/atom (eV)':>14}")
print("-" * 60)
print(f"{'Experiment':<25} {'5.431':>10} {'99.2':>10} {'--':>14}")
for r in results:
    b_str = f"{r['B']:.1f}" if r["B"] else "N/A"
    print(f"{r['label']:<25} {r['a']:>10.4f} {b_str:>10} {r['e']:>14.6f}")
print("=" * 60)

# Plot E-V curves
fig, ax = plt.subplots(figsize=(8, 5))
colors = ["#1565C0", "#2E7D32", "#E65100", "#C62828"]
for i, r in enumerate(results):
    v = np.array(r["volumes"])
    e = np.array(r["energies"]) - min(r["energies"])
    ax.plot(v, e, "o-", color=colors[i % len(colors)], markersize=4,
            linewidth=1.5, label=r["label"])

ax.set_xlabel("Volume (A^3)", fontsize=12)
ax.set_ylabel("E - E_min (eV)", fontsize=12)
ax.set_title("E-V Curves: Foundation vs Fine-Tuned", fontsize=13)
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("model_comparison.png", dpi=150, bbox_inches="tight")
plt.close()
print(f"\nSaved: model_comparison.png")
```

## Key Parameters

| Parameter | Typical Value | Notes |
|---|---|---|
| Foundation model | `"medium"` | Start with medium. Use large for best accuracy. |
| `default_dtype` | `"float64"` | Always use float64 for relaxation, EOS, phonons. |
| Fine-tuning LR | 0.001 - 0.0001 | Lower than training from scratch. Decrease for higher fidelity levels. |
| `forces_weight` | 10 - 1000 | Higher = prioritize force accuracy. 100 is a good default. |
| `max_num_epochs` | 50 - 500 | Fine-tuning needs fewer epochs than training from scratch. |
| `patience` | 10 - 50 | Early stopping. Prevents overfitting. |
| Committee size | 3 - 5 models | More models = better uncertainty estimate, but linear cost. |
| Training set size | 50 - 5000 structures | Fine-tuning works with much less data than training from scratch. |
| Validation split | 10-20% | Must be held out from training for monitoring generalization. |
| `seed` | Vary per committee member | Different seeds give independent models for uncertainty. |

## Interpreting Results

### Foundation model assessment
- Energy MAE < 10 meV/atom: excellent, no fine-tuning needed.
- Energy MAE 10-30 meV/atom: acceptable for many applications.
- Energy MAE > 30 meV/atom: fine-tuning recommended.
- Force MAE < 50 meV/A: excellent for relaxation and MD.
- Force MAE 50-100 meV/A: acceptable for screening, may miss subtle phonon features.
- Force MAE > 100 meV/A: fine-tuning recommended.

### Fine-tuning convergence
- Monitor training and validation loss. If validation loss increases while training loss decreases, the model is overfitting.
- Use early stopping (patience parameter) to prevent overfitting.
- The validation MAE should decrease and plateau.

### Committee uncertainty
- Large energy spread (> 30 meV/atom) indicates the structure is outside the training distribution.
- Consistent predictions across committee members (< 5 meV/atom spread) indicate reliable extrapolation.
- Use uncertainty to guide active learning: add DFT data for high-uncertainty structures and retrain.

## Common Issues

| Issue | Cause | Fix |
|---|---|---|
| Fine-tuning overfits quickly | Too little training data or too many epochs | Add more data, reduce epochs, increase patience, use smaller LR. |
| Fine-tuned model worse than foundation | Training data inconsistent or too specialized | Ensure training data is clean. Use multi-fidelity approach. |
| `mace_run_train` not found | Not installed from source | `pip install mace-torch[train]` or install from GitHub source. |
| Committee models all agree but DFT disagrees | Systematic bias in MACE | Foundation model has systematic errors for your chemistry. Fine-tune. |
| Batch evaluation too slow | Large structures or many structures | Use "small" model for initial screening. GPU if available. |
| Memory error during training | Batch size too large | Reduce batch_size. Use gradient accumulation if supported. |
| Fine-tuned model gives NaN | Training diverged | Reduce learning rate. Check training data for outliers. |
| extxyz format errors | Wrong key names | Use `energy`, `forces`, `stress` as keys (or specify via CLI flags). |
