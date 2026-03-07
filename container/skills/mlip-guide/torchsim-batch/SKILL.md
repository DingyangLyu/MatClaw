# TorchSim GPU-Accelerated Batch Molecular Dynamics and Optimization

## When to Use

- Running MD simulations with MLIPs (MACE, etc.) and need 10-100x speedup over plain ASE on GPU
- Batching hundreds of structures for parallel energy/force evaluation or relaxation in a single GPU call
- Performing high-throughput screening where ASE's serial evaluation is the bottleneck
- Running NVE, NVT (Langevin), or NPT molecular dynamics natively in PyTorch on GPU
- Relaxing many structures simultaneously on GPU using FIRE optimizer
- Auto-batching structures of different sizes (different atom counts, cell dimensions) efficiently
- Need differentiable simulations (forces, energies backpropagable through the simulation trajectory)

## Method Selection

```
Need to run MD or relaxation with MLIPs?

Single structure, moderate system size (< 500 atoms), no GPU?
  --> Plain ASE + MACE calculator (see universal-mlip/ skill)
      Simplest approach, no extra dependencies.

Single structure, large system (> 1000 atoms), need LAMMPS-level scaling?
  --> LAMMPS with MACE or SevenNet pair_style
      Best for very large single simulations on CPU clusters.

Many structures (10-1000+) to relax or screen, GPU available?
  --> TorchSim batch relaxation (this skill, Workflow 2)
      Massive throughput gain from batching on GPU.

MD simulation, GPU available, want maximum speed?
  --> TorchSim MD (this skill, Workflow 1)
      Native PyTorch MD, 10-100x faster than ASE on GPU.

High-throughput property screening (energy, forces, stress) on many structures?
  --> TorchSim auto-batching (this skill, Workflow 3)
      Automatically bins structures by size for efficient GPU batching.

Need differentiable simulation (gradient through trajectory)?
  --> TorchSim (this skill)
      Only option with full PyTorch differentiability.

No GPU available?
  --> Plain ASE is usually sufficient.
      TorchSim can run on CPU but the speedup is minimal.
```

## Prerequisites

```bash
pip install torch-sim mace-torch
```

TorchSim requires PyTorch with CUDA support for GPU acceleration. Verify GPU availability:

```python
import torch
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Memory: {torch.cuda.get_total_memory(0) / 1e9:.1f} GB")
```

If no GPU is available, TorchSim still works on CPU but without the batching speedup advantage over ASE.

## Detailed Steps

---

### Workflow 1: GPU-Accelerated MD with MACE

Run NVT Langevin molecular dynamics on GPU using TorchSim with a MACE potential. This is 10-100x faster than equivalent ASE Langevin MD for MACE models.

```python
#!/usr/bin/env python3
"""
GPU-accelerated NVT Langevin MD with MACE using TorchSim.
Demonstrates single-structure MD on GPU with trajectory reporting.
"""
import torch
import torch_sim as ts
from torch_sim.integrators import langevin
from torch_sim.trajectory import TrajectoryReporter
from ase.build import bulk
from ase.io import read
import numpy as np
import time

# ============================================================
# CONFIGURATION
# ============================================================

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.float32  # float32 is fine for MD; float64 for precise relaxation
TEMPERATURE_K = 300.0
TIMESTEP_FS = 1.0
N_STEPS = 5000         # 5 ps at 1 fs timestep
FRICTION = 0.01        # Langevin friction coefficient (1/fs)
TRAJECTORY_FILE = "/tmp/torchsim_md_trajectory.extxyz"
LOG_INTERVAL = 100     # Save frame every 100 steps

print(f"Running on: {DEVICE}")

# ============================================================
# STEP 1: BUILD STRUCTURE
# ============================================================

# Build a supercell for MD (need enough atoms for meaningful dynamics)
atoms = bulk("Cu", "fcc", a=3.615) * (4, 4, 4)  # 256 atoms
print(f"System: {atoms.get_chemical_formula()} ({len(atoms)} atoms)")

# ============================================================
# STEP 2: LOAD MACE MODEL FOR TORCHSIM
# ============================================================

# Load MACE-MP-0 and wrap it for TorchSim
from mace.calculators import mace_mp

# Get the underlying MACE model for torch_sim
# torch_sim needs the raw model, not the ASE calculator
mace_calc = mace_mp(model="medium", device=DEVICE, default_dtype="float32")
model = mace_calc.models[0]  # Extract the underlying torch model

# ============================================================
# STEP 3: CONVERT STRUCTURE TO TORCHSIM STATE
# ============================================================

# Convert ASE Atoms to TorchSim SimState
state = ts.atoms_to_state(atoms, device=DEVICE, dtype=DTYPE)

# ============================================================
# STEP 4: SET UP TRAJECTORY REPORTER
# ============================================================

reporter = TrajectoryReporter(
    TRAJECTORY_FILE,
    state,
    report_interval=LOG_INTERVAL,
)

# ============================================================
# STEP 5: BUILD THE LANGEVIN INTEGRATOR
# ============================================================

# Create NVT Langevin integrator
# TorchSim uses a functional style: the integrator is a function
# that takes (state, model) and returns the updated state
langevin_init, langevin_step = langevin(
    state,
    model,
    dt=TIMESTEP_FS * 1e-3,  # TorchSim uses ps internally
    temperature=TEMPERATURE_K,
    friction=FRICTION,
)

# Initialize the integrator state
integrator_state = langevin_init(state)

# ============================================================
# STEP 6: RUN MD
# ============================================================

print(f"\nRunning {N_STEPS} steps of NVT Langevin MD at {TEMPERATURE_K} K...")
t_start = time.time()

for step in range(N_STEPS):
    state, integrator_state = langevin_step(state, integrator_state)

    # Report trajectory
    reporter.report(state, step)

    # Print progress
    if (step + 1) % 500 == 0:
        # Compute kinetic energy for temperature estimate
        ke = 0.5 * torch.sum(
            state.masses * torch.sum(state.velocities ** 2, dim=-1, keepdim=True)
        )
        n_dof = 3 * state.n_atoms - 3
        temp = (2 * ke / (n_dof * 8.617333e-5)).item()  # kB in eV/K
        elapsed = time.time() - t_start
        rate = (step + 1) / elapsed
        print(f"  Step {step+1:>6d}/{N_STEPS}: T = {temp:>7.1f} K "
              f"({rate:.0f} steps/s)")

elapsed = time.time() - t_start
print(f"\nMD completed in {elapsed:.1f} s ({N_STEPS/elapsed:.0f} steps/s)")
print(f"Trajectory saved to: {TRAJECTORY_FILE}")

# ============================================================
# STEP 7: ANALYZE TRAJECTORY
# ============================================================

# Read back trajectory with ASE for analysis
trajectory = read(TRAJECTORY_FILE, index=":")
print(f"\nTrajectory frames: {len(trajectory)}")

if len(trajectory) > 1:
    energies = [frame.get_potential_energy() for frame in trajectory
                if 'energy' in frame.info or hasattr(frame, '_calc')]
    positions = [frame.get_positions() for frame in trajectory]

    # Compute MSD (mean squared displacement)
    ref_pos = positions[0]
    msd = [np.mean(np.sum((pos - ref_pos)**2, axis=1)) for pos in positions]
    print(f"Final MSD: {msd[-1]:.4f} A^2")
```

---

### Workflow 2: Batch Structure Relaxation

Relax many structures simultaneously on GPU using TorchSim's FIRE optimizer and auto-batching. This is the primary use case for high-throughput screening.

```python
#!/usr/bin/env python3
"""
Batch GPU relaxation of many structures using TorchSim FIRE optimizer.
Demonstrates auto-batching structures of different sizes for efficient
parallel relaxation on a single GPU.
"""
import torch
import torch_sim as ts
from torch_sim.optimizers import fire
from torch_sim.autobatching import BinningAutoBatcher
from ase.build import bulk
from ase.io import read, write
from pymatgen.core import Structure, Lattice
from pymatgen.io.ase import AseAtomsAdaptor
import numpy as np
import time

# ============================================================
# CONFIGURATION
# ============================================================

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.float64   # float64 recommended for relaxation precision
FMAX = 0.05             # Force convergence threshold (eV/A)
MAX_STEPS = 500         # Maximum optimization steps
OUTPUT_FILE = "/tmp/torchsim_relaxed.extxyz"

print(f"Running on: {DEVICE}")

# ============================================================
# STEP 1: GENERATE OR LOAD STRUCTURES
# ============================================================

# Example: generate a set of structures with different compositions
# In practice, read from files: structures = read("candidates.extxyz", index=":")

structures = []

# Binary compounds with slightly distorted lattice constants
compounds = [
    ("Si", "diamond", 5.50),      # Si (exp: 5.431)
    ("Ge", "diamond", 5.72),      # Ge (exp: 5.658)
    ("C", "diamond", 3.60),       # C (exp: 3.567)
    ("Cu", "fcc", 3.65),          # Cu (exp: 3.615)
    ("Al", "fcc", 4.10),          # Al (exp: 4.050)
    ("Ag", "fcc", 4.15),          # Ag (exp: 4.086)
    ("Au", "fcc", 4.12),          # Au (exp: 4.078)
    ("Ni", "fcc", 3.55),          # Ni (exp: 3.524)
    ("Pd", "fcc", 3.95),          # Pd (exp: 3.890)
    ("Pt", "fcc", 3.97),          # Pt (exp: 3.924)
]

for elem, crystal, a in compounds:
    # Create supercells of different sizes to test auto-batching
    for repeat in [(1, 1, 1), (2, 2, 2)]:
        atoms = bulk(elem, crystal, a=a) * repeat
        atoms.info["source"] = f"{elem}_{crystal}_{'x'.join(map(str, repeat))}"
        structures.append(atoms)

print(f"Total structures to relax: {len(structures)}")
for i, s in enumerate(structures):
    print(f"  [{i}] {s.get_chemical_formula():>12s} ({len(s):>4d} atoms) "
          f"- {s.info.get('source', 'unknown')}")

# ============================================================
# STEP 2: LOAD MACE MODEL
# ============================================================

from mace.calculators import mace_mp

mace_calc = mace_mp(model="medium", device=DEVICE, default_dtype="float64")
model = mace_calc.models[0]

# ============================================================
# STEP 3: BATCH RELAXATION WITH AUTO-BATCHING
# ============================================================

print(f"\n=== Batch Relaxation with FIRE Optimizer ===")
print(f"Convergence: fmax < {FMAX} eV/A, max {MAX_STEPS} steps")

t_start = time.time()

# Convert all structures to TorchSim states
states = [ts.atoms_to_state(atoms, device=DEVICE, dtype=DTYPE)
          for atoms in structures]

# Use BinningAutoBatcher to handle different structure sizes
# It groups structures with similar atom counts for efficient batching
batcher = BinningAutoBatcher(
    states,
    memory_limit=2e9,   # 2 GB GPU memory limit for batching
)

relaxed_structures = []
results = []

# Process each batch
for batch_idx, (batched_state, batch_indices) in enumerate(batcher):
    n_in_batch = len(batch_indices)
    print(f"\n  Batch {batch_idx}: {n_in_batch} structures, "
          f"atoms/structure: {[len(structures[i]) for i in batch_indices]}")

    # Set up FIRE optimizer for this batch
    fire_init, fire_step = fire(
        batched_state,
        model,
        dt_max=0.1,     # Maximum timestep (ps)
        dt_start=0.01,  # Initial timestep (ps)
    )

    opt_state = fire_init(batched_state)

    # Run optimization loop
    for step in range(MAX_STEPS):
        batched_state, opt_state = fire_step(batched_state, opt_state)

        # Check convergence: max force on each structure in the batch
        forces = batched_state.forces
        max_forces = torch.sqrt(
            torch.sum(forces ** 2, dim=-1)
        ).max()

        if max_forces.item() < FMAX:
            print(f"    Converged at step {step + 1} "
                  f"(max force: {max_forces.item():.4f} eV/A)")
            break
    else:
        print(f"    Reached max steps ({MAX_STEPS}), "
              f"max force: {max_forces.item():.4f} eV/A")

    # Extract relaxed structures from the batch
    relaxed_atoms_list = ts.state_to_atoms(batched_state)

    for local_idx, global_idx in enumerate(batch_indices):
        relaxed_atoms = relaxed_atoms_list[local_idx]
        original = structures[global_idx]

        # Store results
        energy_per_atom = (relaxed_atoms.get_potential_energy()
                           / len(relaxed_atoms))
        max_force = np.sqrt(
            (relaxed_atoms.get_forces() ** 2).sum(axis=1)
        ).max()
        a_relaxed = relaxed_atoms.cell.cellpar()[0]

        relaxed_atoms.info["source"] = original.info.get("source", "")
        relaxed_atoms.info["energy_per_atom"] = energy_per_atom
        relaxed_atoms.info["max_force"] = max_force
        relaxed_structures.append(relaxed_atoms)

        results.append({
            "formula": original.get_chemical_formula(),
            "source": original.info.get("source", ""),
            "n_atoms": len(original),
            "a_initial": original.cell.cellpar()[0],
            "a_relaxed": a_relaxed,
            "energy_per_atom": energy_per_atom,
            "max_force": max_force,
        })

elapsed = time.time() - t_start
print(f"\n=== Relaxation Complete ===")
print(f"Total time: {elapsed:.1f} s ({len(structures)/elapsed:.1f} structures/s)")

# ============================================================
# STEP 4: SUMMARIZE AND SAVE RESULTS
# ============================================================

print(f"\n{'Formula':>12} {'Source':>20} {'a_init':>8} {'a_relax':>8} "
      f"{'E/atom':>10} {'F_max':>8}")
print("-" * 78)
for r in results:
    print(f"{r['formula']:>12} {r['source']:>20} {r['a_initial']:>8.3f} "
          f"{r['a_relaxed']:>8.4f} {r['energy_per_atom']:>10.4f} "
          f"{r['max_force']:>8.4f}")

# Save relaxed structures
write(OUTPUT_FILE, relaxed_structures)
print(f"\nRelaxed structures saved to: {OUTPUT_FILE}")
```

---

### Workflow 3: High-Throughput Property Screening

Screen a large set of structures for energetic stability, computing energy/forces/stress in batched GPU calls with auto-batching for different structure sizes.

```python
#!/usr/bin/env python3
"""
High-throughput property screening using TorchSim batch evaluation.
Efficiently compute energies, forces, and stresses for many structures
on GPU with automatic batching by structure size.
"""
import torch
import torch_sim as ts
from torch_sim.autobatching import BinningAutoBatcher
from ase.build import bulk
from ase.io import read, write
import numpy as np
import time
import json

# ============================================================
# CONFIGURATION
# ============================================================

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.float32   # float32 is sufficient for screening
GPU_MEMORY_LIMIT = 4e9  # 4 GB memory budget for batching
OUTPUT_JSON = "/tmp/torchsim_screening_results.json"

print(f"Running on: {DEVICE}")

# ============================================================
# STEP 1: LOAD OR GENERATE CANDIDATE STRUCTURES
# ============================================================

# In practice: structures = read("candidates.extxyz", index=":")
# Here we generate a test set: elemental solids at various lattice constants

candidates = []

elements_fcc = [("Cu", 3.615), ("Al", 4.050), ("Ag", 4.086),
                ("Au", 4.078), ("Ni", 3.524), ("Pd", 3.890),
                ("Pt", 3.924), ("Pb", 4.950), ("Ca", 5.588)]

elements_bcc = [("Fe", 2.870), ("W", 3.165), ("Mo", 3.147),
                ("V", 3.024), ("Cr", 2.884), ("Na", 4.291),
                ("K", 5.328), ("Li", 3.510), ("Ba", 5.020)]

elements_diamond = [("Si", 5.431), ("Ge", 5.658), ("C", 3.567)]

# Generate strained versions for E-V screening
for elem, a0 in elements_fcc + elements_bcc + elements_diamond:
    crystal = "fcc"
    if (elem, a0) in elements_bcc:
        crystal = "bcc"
    elif (elem, a0) in elements_diamond:
        crystal = "diamond"

    for strain in [0.96, 0.98, 1.00, 1.02, 1.04]:
        atoms = bulk(elem, crystal, a=a0 * strain)
        atoms.info["element"] = elem
        atoms.info["crystal"] = crystal
        atoms.info["a0_exp"] = a0
        atoms.info["strain"] = strain
        atoms.info["a_current"] = a0 * strain
        candidates.append(atoms)

print(f"Total candidates: {len(candidates)}")
print(f"Elements: {len(elements_fcc)} FCC, {len(elements_bcc)} BCC, "
      f"{len(elements_diamond)} diamond")

# ============================================================
# STEP 2: LOAD MACE MODEL
# ============================================================

from mace.calculators import mace_mp

mace_calc = mace_mp(model="medium", device=DEVICE, default_dtype="float32")
model = mace_calc.models[0]

# ============================================================
# STEP 3: BATCH EVALUATION WITH AUTO-BATCHING
# ============================================================

print(f"\n=== Batch Property Evaluation ===")
t_start = time.time()

# Convert all structures to TorchSim states
states = [ts.atoms_to_state(atoms, device=DEVICE, dtype=DTYPE)
          for atoms in candidates]

# Set up auto-batcher
batcher = BinningAutoBatcher(
    states,
    memory_limit=GPU_MEMORY_LIMIT,
)

# Store results aligned with candidates
all_results = [None] * len(candidates)

n_evaluated = 0
for batch_idx, (batched_state, batch_indices) in enumerate(batcher):
    # Evaluate the model on the batch
    # The model returns energy, forces, stress for all structures in the batch
    with torch.no_grad():
        result = model(batched_state)

    # Extract per-structure results
    energies = result["energy"]
    forces = result["forces"]
    stresses = result.get("stress", None)

    for local_idx, global_idx in enumerate(batch_indices):
        atoms = candidates[global_idx]
        n_atoms = len(atoms)

        energy = energies[local_idx].item()
        energy_per_atom = energy / n_atoms

        # Max force magnitude
        f = forces[local_idx] if forces.dim() == 3 else forces
        max_force = torch.sqrt(torch.sum(f ** 2, dim=-1)).max().item()

        # Pressure from stress (if available)
        pressure_gpa = None
        if stresses is not None:
            stress = stresses[local_idx]
            # Voigt: xx, yy, zz, yz, xz, xy
            pressure_gpa = (
                -torch.mean(stress[:3]).item() * 160.2177  # eV/A^3 -> GPa
            )

        all_results[global_idx] = {
            "index": global_idx,
            "formula": atoms.get_chemical_formula(),
            "element": atoms.info["element"],
            "crystal": atoms.info["crystal"],
            "strain": atoms.info["strain"],
            "a_current": atoms.info["a_current"],
            "a0_exp": atoms.info["a0_exp"],
            "n_atoms": n_atoms,
            "volume_per_atom": atoms.get_volume() / n_atoms,
            "energy_per_atom": energy_per_atom,
            "max_force": max_force,
            "pressure_GPa": pressure_gpa,
        }

    n_evaluated += len(batch_indices)
    if (batch_idx + 1) % 5 == 0 or n_evaluated == len(candidates):
        elapsed = time.time() - t_start
        print(f"  Evaluated {n_evaluated}/{len(candidates)} structures "
              f"({n_evaluated/elapsed:.0f} structures/s)")

elapsed = time.time() - t_start
print(f"\nScreening completed in {elapsed:.1f} s "
      f"({len(candidates)/elapsed:.0f} structures/s)")

# ============================================================
# STEP 4: ANALYZE RESULTS - FIND EQUILIBRIUM PROPERTIES
# ============================================================

print(f"\n=== Equilibrium Properties by Element ===")
print(f"{'Element':>8} {'Crystal':>8} {'a0_exp':>8} {'a0_MACE':>9} "
      f"{'Err (%)':>8} {'E_min/at':>10}")
print("-" * 62)

# Group by element and find minimum energy (approximate equilibrium)
from collections import defaultdict
by_element = defaultdict(list)
for r in all_results:
    if r is not None:
        key = (r["element"], r["crystal"])
        by_element[key].append(r)

for (elem, crystal), entries in sorted(by_element.items()):
    # Find strain with lowest energy
    entries_sorted = sorted(entries, key=lambda x: x["energy_per_atom"])
    best = entries_sorted[0]

    a_exp = best["a0_exp"]
    a_mace = best["a_current"]
    err = abs(a_mace - a_exp) / a_exp * 100

    print(f"{elem:>8} {crystal:>8} {a_exp:>8.3f} {a_mace:>9.4f} "
          f"{err:>7.2f}% {best['energy_per_atom']:>10.4f}")

# ============================================================
# STEP 5: SAVE RESULTS
# ============================================================

# Save as JSON for further analysis
with open(OUTPUT_JSON, "w") as f:
    json.dump([r for r in all_results if r is not None], f, indent=2)
print(f"\nResults saved to: {OUTPUT_JSON}")

# ============================================================
# STEP 6: FIND MOST STABLE STRUCTURES (RANKING)
# ============================================================

print(f"\n=== Stability Ranking (lowest energy per atom) ===")
# Filter to strain=1.0 (near equilibrium) for fair comparison
equilibrium = [r for r in all_results
               if r is not None and abs(r["strain"] - 1.0) < 0.001]
equilibrium.sort(key=lambda x: x["energy_per_atom"])

print(f"{'Rank':>5} {'Formula':>8} {'Crystal':>8} {'E/atom (eV)':>12} "
      f"{'F_max':>8} {'P (GPa)':>9}")
print("-" * 56)
for i, r in enumerate(equilibrium[:15]):
    p_str = f"{r['pressure_GPa']:.2f}" if r["pressure_GPa"] is not None else "N/A"
    print(f"{i+1:>5} {r['formula']:>8} {r['crystal']:>8} "
          f"{r['energy_per_atom']:>12.6f} {r['max_force']:>8.4f} {p_str:>9}")
```

## Key Parameters

### TorchSim Core

| Parameter | Default / Typical | Description |
|---|---|---|
| `device` | `"cuda"` | Device for computation. Use `"cuda"` for GPU acceleration, `"cpu"` as fallback. |
| `dtype` | `torch.float32` | Data type. Use `float32` for MD/screening, `float64` for precise relaxation. |

### Langevin Integrator (`ts.integrators.langevin`)

| Parameter | Typical Value | Description |
|---|---|---|
| `dt` | 0.001 (ps) = 1 fs | Integration timestep in picoseconds. |
| `temperature` | 300.0 | Target temperature in Kelvin. |
| `friction` | 0.01 | Langevin friction coefficient (1/fs). Higher = stronger thermostat coupling. |

### FIRE Optimizer (`ts.optimizers.fire`)

| Parameter | Typical Value | Description |
|---|---|---|
| `dt_max` | 0.1 (ps) | Maximum adaptive timestep. |
| `dt_start` | 0.01 (ps) | Initial timestep. |
| `fmax` (convergence) | 0.01-0.05 eV/A | Force convergence criterion (checked manually in loop). |

### Auto-Batcher (`ts.autobatching.BinningAutoBatcher`)

| Parameter | Typical Value | Description |
|---|---|---|
| `memory_limit` | `2e9` - `8e9` | GPU memory budget in bytes. Set to ~50-75% of total GPU memory. |

### TrajectoryReporter (`ts.trajectory.TrajectoryReporter`)

| Parameter | Typical Value | Description |
|---|---|---|
| `report_interval` | 10-100 | Save a frame every N steps. Lower = larger trajectory files. |

### Comparison with ASE

| Setting | ASE | TorchSim |
|---|---|---|
| Timestep | `1.0 * units.fs` | `0.001` (ps) |
| Temperature | `temperature_K=300` | `temperature=300.0` |
| Force threshold | `fmax=0.01` | Check manually: `forces.max() < 0.01` |
| Trajectory output | `ase.io.Trajectory` | `TrajectoryReporter` |
| Batching | Not supported (serial loop) | Native batch + auto-batch |

## Interpreting Results

### Speed benchmarks
- **Single structure MD**: Expect 10-50x speedup over ASE for MACE on GPU. The speedup comes from keeping everything on GPU (no CPU-GPU data transfer per step).
- **Batch relaxation**: Expect near-linear scaling with batch size up to GPU memory limits. 100 structures batched together can be 50-100x faster than relaxing them serially with ASE.
- **Throughput screening**: 100-1000+ structures per second for single-point energy evaluation, depending on structure size and GPU.

### Energy and forces
- Energies and forces from TorchSim are identical to ASE + MACE (same underlying model). The only difference is computational efficiency.
- When using `float32`, expect ~0.1 meV/atom numerical noise compared to `float64`. This is negligible for MD and screening but can matter for precise equation-of-state fitting.
- Always use `float64` for relaxation where force convergence below 0.01 eV/A is needed.

### Auto-batching behavior
- The `BinningAutoBatcher` groups structures by atom count into bins. Structures with similar sizes are batched together for maximum GPU efficiency.
- If structures have very different sizes (e.g., 2-atom unit cells mixed with 200-atom supercells), they will be processed in separate batches. This is expected and optimal.
- The `memory_limit` controls the maximum batch size. If you get CUDA out-of-memory errors, reduce this value.

### Trajectory output
- TorchSim trajectories are saved in extxyz format, readable by ASE: `read("trajectory.extxyz", index=":")`.
- Positions, cell, energies, and forces are stored per frame.
- For large trajectories, use `report_interval` to control file size.

## Common Issues

| Issue | Cause | Fix |
|---|---|---|
| `CUDA out of memory` | Batch too large for GPU | Reduce `memory_limit` in `BinningAutoBatcher`. Use a smaller model (MACE small). Reduce supercell sizes. |
| TorchSim not finding MACE model | Wrong model extraction | Use `mace_calc.models[0]` to get the raw PyTorch model from the ASE calculator wrapper. |
| `float32` relaxation not converging below fmax=0.01 | Numerical noise in float32 | Switch to `dtype=torch.float64` for tight relaxation convergence. |
| MD temperature drift | Timestep too large or friction too low | Reduce `dt` (try 0.5 fs). Increase `friction` (try 0.05). Check that masses are correct. |
| Slow performance on CPU | No GPU acceleration | TorchSim is designed for GPU. On CPU, use plain ASE instead (simpler, similar speed). |
| `ImportError: No module named 'torch_sim'` | Not installed | `pip install torch-sim`. Requires PyTorch >= 2.0. |
| Auto-batcher yields one structure per batch | Structures have very different sizes | Expected behavior. The batcher bins by atom count. Pad to similar sizes if needed, or accept per-structure batches. |
| NaN in energy/forces during MD | Atoms too close, timestep too large, or unstable structure | Check initial structure for overlapping atoms. Reduce timestep. Start with a short equilibration at lower temperature. |
| Trajectory file very large | Too many frames saved | Increase `report_interval`. For 100k-step MD, use interval of 100-1000. |
| Model gives different results than ASE | Dtype mismatch or model version | Ensure same model variant and dtype. TorchSim float32 vs ASE float64 can differ by ~0.1 meV/atom. |
