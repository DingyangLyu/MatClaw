# Free Energy Calculation

## When to Use

- You need the Helmholtz or Gibbs free energy of a crystalline solid or liquid at a given temperature and pressure.
- You want to determine which phase (e.g., FCC vs BCC, solid vs liquid) is thermodynamically more stable at specific conditions.
- You need to compute melting points by finding the temperature where solid and liquid free energies cross.
- You need free energy as a function of temperature (via reversible scaling / temperature integration).
- You want to compare the accuracy of two interatomic potentials by computing a free energy difference between them (alchemical transformation).

## Background

Free energy cannot be directly measured from a single MD simulation because it depends on the partition function. Special thermodynamic integration techniques are required:

- **Frenkel-Ladd (Einstein crystal) method** for **solids**: The system is reversibly transformed into a non-interacting Einstein crystal (harmonic springs tethering atoms to lattice sites) whose free energy is known analytically. The work done during this switching gives the free energy difference.
- **Uhlenbeck-Ford (UF) model method** for **liquids**: The system is reversibly transformed into a Uhlenbeck-Ford reference fluid whose free energy is known analytically.
- **Reversible scaling (temperature integration)**: Once the free energy is known at one temperature, the Gibbs-Helmholtz equation is integrated to obtain free energy over a temperature range.

These methods are implemented in the [calphy](https://doi.org/10.1103/PhysRevMaterials.5.103801) framework. The scripts below implement the same physics standalone, using LAMMPS directly via subprocess.

## Method Selection

| Criterion | LAMMPS + Classical Potential | LAMMPS + MACE (ML) | Direct DFT (QE) |
|---|---|---|---|
| Speed | Fast (minutes) | Moderate (hours) | Impractical |
| Accuracy | Depends on potential quality | Near-DFT accuracy | Exact (but cannot run long enough for TI) |
| System size | 4x4x4 supercell or larger (256+ atoms) | 3x3x3 to 4x4x4 (108-256 atoms) | Not feasible for free energy |
| Best for | Rapid screening, potentials from literature | Publication-quality with ML potentials | Not recommended for free energy |

**MACE cannot compute free energies on its own** -- it must be coupled to LAMMPS as a pair style so that the LAMMPS `fix adapt` and thermostat machinery can perform the thermodynamic integration switching.

## Prerequisites

- LAMMPS binary with MACE pair style support (`lmp` in PATH, or specify path). The container has LAMMPS with the ML-MACE package compiled in.
- For classical potentials: EAM/MEAM/Tersoff potential files.
- For MACE: a MACE model file (`.model` format). The foundation model is available via `mace_mp`.
- Python packages: `numpy`, `scipy`, `matplotlib`, `ase`, `pymatgen` (all pre-installed).
- A relaxed structure (use the `scf-relax` or `molecular-dynamics` skill first to equilibrate).

---

## Detailed Steps

### Method A: Free Energy of a Solid Phase (Frenkel-Ladd / Einstein Crystal)

This script performs a complete Frenkel-Ladd thermodynamic integration for a solid using LAMMPS. It:
1. Equilibrates the system at the target temperature
2. Determines optimal spring constants
3. Switches from the real system to an Einstein crystal (forward)
4. Switches back (backward) to estimate hysteresis error
5. Computes the total Helmholtz free energy

```python
#!/usr/bin/env python3
"""
Frenkel-Ladd (Einstein crystal) free energy calculation for a solid phase.
Uses LAMMPS via subprocess. No pyiron dependency.

Reference: Frenkel & Ladd, J. Chem. Phys. 81, 3188 (1984)
           Menon et al., Phys. Rev. Materials 5, 103801 (2021) [calphy]
"""

import os
import sys
import json
import subprocess
import tempfile
import shutil
import numpy as np
from scipy import integrate

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from ase.build import bulk
from ase.io import write, read

# ============================================================
# 1. CONFIGURATION
# ============================================================

ELEMENT = "Cu"
CRYSTAL_STRUCTURE = "fcc"
LATTICE_CONSTANT = 3.615       # Angstrom
SUPERCELL = (4, 4, 4)          # At least 4x4x4 for converged results
TEMPERATURE = 800.0            # K
PRESSURE = 0.0                 # bar (0 = NVT at zero pressure)
MASS = 63.546                  # amu

# LAMMPS potential specification
# Option 1: Classical EAM potential
PAIR_STYLE = "eam/alloy"
PAIR_COEFF = "* * Cu_mishin1.eam.alloy Cu"
POTENTIAL_FILES = ["Cu_mishin1.eam.alloy"]  # must exist in working dir

# Option 2: MACE ML potential (uncomment to use)
# PAIR_STYLE = "mace no_domain_decomposition"
# PAIR_COEFF = "* * /path/to/mace_model.model Cu"
# POTENTIAL_FILES = []

# MD parameters
TIMESTEP = 0.001               # ps
N_EQUIL = 15000                # equilibration steps
N_SWITCH = 25000               # switching steps (forward and backward)
N_PRINT = 100                  # print frequency during switching
N_THERMO = 100                 # thermo output frequency

# Spring constant for Einstein crystal (eV/A^2)
# If None, will be estimated from mean-square displacement
SPRING_CONSTANT = None

# LAMMPS executable
LAMMPS_CMD = "lmp"

WORKDIR = os.path.abspath("free_energy_solid")
os.makedirs(WORKDIR, exist_ok=True)

# ============================================================
# 2. BUILD AND WRITE STRUCTURE
# ============================================================

print("=" * 60)
print("Frenkel-Ladd Free Energy Calculation")
print("=" * 60)

atoms = bulk(ELEMENT, CRYSTAL_STRUCTURE, a=LATTICE_CONSTANT, cubic=True)
atoms = atoms.repeat(SUPERCELL)
n_atoms = len(atoms)

print(f"  Element: {ELEMENT}")
print(f"  Structure: {CRYSTAL_STRUCTURE}")
print(f"  Supercell: {SUPERCELL}, N_atoms = {n_atoms}")
print(f"  Temperature: {TEMPERATURE} K")
print(f"  Pressure: {PRESSURE} bar")

# Write LAMMPS data file
data_file = os.path.join(WORKDIR, "structure.data")
write(data_file, atoms, format="lammps-data")

# ============================================================
# 3. STEP 1: EQUILIBRATION AND SPRING CONSTANT ESTIMATION
# ============================================================

print("\n--- Step 1: Equilibration ---")

equil_script = f"""# Equilibration at target temperature
units          metal
atom_style     atomic
boundary       p p p

read_data      structure.data

mass           1 {MASS}

pair_style     {PAIR_STYLE}
pair_coeff     {PAIR_COEFF}

# Initial velocities
velocity       all create {2*TEMPERATURE} 12345 dist gaussian

# Equilibrate NPT to get correct density, then NVT
fix            1 all npt temp {TEMPERATURE} {TEMPERATURE} 0.1 iso {PRESSURE} {PRESSURE} 1.0
thermo_style   custom step temp pe ke etotal press vol lx ly lz
thermo         {N_THERMO}
run            {N_EQUIL}
unfix          1

# Switch to NVT for production
fix            2 all nvt temp {TEMPERATURE} {TEMPERATURE} 0.1
run            {N_EQUIL}

# Compute MSD for spring constant estimation
compute        msd_all all msd
fix            msd_avg all ave/time 1 100 100 c_msd_all[4] file msd.dat

# Store reference positions (lattice sites)
fix            ref all store/force

# Production run to measure MSD
run            5000

# Write equilibrated structure
write_data     equilibrated.data
"""

equil_file = os.path.join(WORKDIR, "equil.in")
with open(equil_file, "w") as f:
    f.write(equil_script)

result = subprocess.run(
    [LAMMPS_CMD, "-in", "equil.in"],
    cwd=WORKDIR,
    capture_output=True, text=True, timeout=1800,
)

with open(os.path.join(WORKDIR, "equil.log"), "w") as f:
    f.write(result.stdout)

if result.returncode != 0:
    print(f"ERROR: Equilibration failed!")
    print(result.stderr[-1000:] if result.stderr else "No stderr")
    sys.exit(1)

print("  Equilibration completed.")

# Estimate spring constant from MSD if not provided
if SPRING_CONSTANT is None:
    try:
        msd_data = np.loadtxt(os.path.join(WORKDIR, "msd.dat"), comments="#")
        mean_msd = np.mean(msd_data[-20:, 1])  # last 20 entries
        # k_B T = k * <u^2> / 3  =>  k = 3 k_B T / <u^2>
        kB = 8.617333262e-5  # eV/K
        if mean_msd > 1e-10:
            SPRING_CONSTANT = 3.0 * kB * TEMPERATURE / mean_msd
        else:
            SPRING_CONSTANT = 5.0  # default fallback
        print(f"  Estimated spring constant: {SPRING_CONSTANT:.4f} eV/A^2")
        print(f"  (from MSD = {mean_msd:.6f} A^2)")
    except Exception as e:
        SPRING_CONSTANT = 5.0
        print(f"  Using default spring constant: {SPRING_CONSTANT} eV/A^2 (MSD read failed: {e})")
else:
    print(f"  Using provided spring constant: {SPRING_CONSTANT} eV/A^2")

# ============================================================
# 4. STEP 2: THERMODYNAMIC INTEGRATION (SWITCHING)
# ============================================================

print("\n--- Step 2: Thermodynamic Integration ---")

# The switching is done by linearly interpolating between the real potential
# and the Einstein crystal: U(lambda) = lambda * U_real + (1 - lambda) * U_einstein
# We record dU/dlambda = U_real - U_einstein at each step.

switch_script = f"""# Frenkel-Ladd thermodynamic integration switching
units          metal
atom_style     atomic
boundary       p p p

read_data      equilibrated.data

mass           1 {MASS}

pair_style     {PAIR_STYLE}
pair_coeff     {PAIR_COEFF}

# Store reference (equilibrium) positions for Einstein crystal
# We use fix store/state to save positions, then fix spring/self

velocity       all create {TEMPERATURE} 23456 dist gaussian

# --- Forward switching: real -> Einstein crystal ---
# lambda goes from 0 to 1
# At lambda=0: pure real potential
# At lambda=1: pure Einstein crystal

# NVT thermostat
fix            thermostat all nvt temp {TEMPERATURE} {TEMPERATURE} 0.1

# Compute per-atom potential energy from real potential
compute        pe_real all pe/atom
compute        pe_total all reduce sum c_pe_real

# Apply Einstein crystal springs (tethered to initial positions)
# fix spring/self applies: E_spring = k/2 * (r - r0)^2 for each atom
fix            einstein all spring/self {SPRING_CONSTANT}
fix_modify     einstein energy yes

# Compute spring energy
compute        pe_spring all reduce sum f_einstein

# Now we need to switch: scale real potential by (1-lambda) and springs by lambda
# We use fix adapt to scale pair interactions
# lambda = 0 -> 1 over N_SWITCH steps

# Actually, for Frenkel-Ladd, we use a simpler approach:
# Run at full real potential + full springs, and record the energy difference.
# The integrand is: <U_real - U_spring> at each lambda
# We use the direct switching approach with fix adapt.

# For the direct approach: we run the full system and compute
# the difference dU = U_real - U_einstein at each lambda value

variable       lambda equal ramp(0,1)
variable       dlambda equal 1.0/{N_SWITCH}

# Forward: lambda 0->1 (switching ON the springs, switching OFF real potential)
variable       pe_diff equal c_pe_total-c_pe_spring

thermo_style   custom step v_lambda c_pe_total c_pe_spring v_pe_diff temp press
thermo         {N_PRINT}

# Use fix print to record the integrand
fix            fwd_print all print {N_PRINT} "${{lambda}} ${{pe_diff}} ${{pe_total}} ${{pe_spring}}" &
               file forward.dat screen no title "# lambda  pe_diff  pe_real  pe_spring"

# Run forward switching
# Use fix adapt to scale the pair potential
fix            scale_pair all adapt 1 pair {PAIR_STYLE.split()[0]} scale * * v_lambda
run            {N_SWITCH}
unfix          scale_pair
unfix          fwd_print

# --- Backward switching: Einstein crystal -> real ---
variable       lambda_bk equal ramp(1,0)

fix            bwd_print all print {N_PRINT} "${{lambda_bk}} ${{pe_diff}} ${{pe_total}} ${{pe_spring}}" &
               file backward.dat screen no title "# lambda  pe_diff  pe_real  pe_spring"

fix            scale_pair_bk all adapt 1 pair {PAIR_STYLE.split()[0]} scale * * v_lambda_bk
run            {N_SWITCH}
unfix          scale_pair_bk
unfix          bwd_print
"""

switch_file = os.path.join(WORKDIR, "switch.in")
with open(switch_file, "w") as f:
    f.write(switch_script)

result = subprocess.run(
    [LAMMPS_CMD, "-in", "switch.in"],
    cwd=WORKDIR,
    capture_output=True, text=True, timeout=3600,
)

with open(os.path.join(WORKDIR, "switch.log"), "w") as f:
    f.write(result.stdout)

if result.returncode != 0:
    print(f"ERROR: Switching failed!")
    print(result.stderr[-1000:] if result.stderr else "No stderr")
    sys.exit(1)

print("  Forward and backward switching completed.")

# ============================================================
# 5. STEP 3: COMPUTE FREE ENERGY
# ============================================================

print("\n--- Step 3: Computing Free Energy ---")

kB = 8.617333262e-5  # eV/K

# Read switching data
try:
    fwd_data = np.loadtxt(os.path.join(WORKDIR, "forward.dat"), comments="#")
    bwd_data = np.loadtxt(os.path.join(WORKDIR, "backward.dat"), comments="#")
except Exception as e:
    print(f"ERROR reading switching data: {e}")
    sys.exit(1)

fwd_lambda = fwd_data[:, 0]
fwd_dU = fwd_data[:, 1]  # pe_diff = U_real - U_spring

bwd_lambda = bwd_data[:, 0]
bwd_dU = bwd_data[:, 1]

# Sort by lambda
fwd_sort = np.argsort(fwd_lambda)
bwd_sort = np.argsort(bwd_lambda)

fwd_lambda = fwd_lambda[fwd_sort]
fwd_dU = fwd_dU[fwd_sort]
bwd_lambda = bwd_lambda[bwd_sort]
bwd_dU = bwd_dU[bwd_sort]

# Integrate: W = integral_0^1 <dU/dlambda> dlambda
# For the switching from real to Einstein crystal:
# dU/dlambda = U_spring - U_real = -pe_diff
W_forward = -integrate.trapezoid(fwd_dU / n_atoms, fwd_lambda)
W_backward = -integrate.trapezoid(bwd_dU / n_atoms, bwd_lambda)

W_avg = (W_forward + W_backward) / 2.0
W_err = abs(W_forward - W_backward) / 2.0

print(f"  Work (forward):  {W_forward:.6f} eV/atom")
print(f"  Work (backward): {W_backward:.6f} eV/atom")
print(f"  Work (average):  {W_avg:.6f} eV/atom")
print(f"  Hysteresis:      {W_err:.6f} eV/atom")

# Einstein crystal free energy (analytical, per atom)
# F_einstein = 3/2 * kB * T * ln(beta * hbar * omega)  [quantum]
# Classical:  F_einstein = 3 * kB * T * ln(sqrt(k / (2 * pi * kB * T)))
# In LAMMPS units (metal): energy = eV, mass = amu, distance = Angstrom
# omega = sqrt(k / m),  hbar = 6.5821e-16 eV*s,  m in kg

hbar = 6.582119569e-16   # eV*s
amu_to_kg = 1.66053906660e-27
omega = np.sqrt(SPRING_CONSTANT * 1.60218e-19 / (MASS * amu_to_kg)) / 1e-10  # rad/s
# omega = sqrt(k[eV/A^2] / m[kg]) with unit conversions

beta_hbar_omega = hbar * omega / (kB * TEMPERATURE)

if beta_hbar_omega > 0.1:
    # Quantum harmonic oscillator
    F_einstein = 3.0 * kB * TEMPERATURE * np.log(2 * np.sinh(beta_hbar_omega / 2.0))
    print(f"  Using quantum harmonic oscillator reference (beta*hbar*omega = {beta_hbar_omega:.3f})")
else:
    # Classical limit
    F_einstein = 3.0 * kB * TEMPERATURE * np.log(beta_hbar_omega)
    print(f"  Using classical harmonic oscillator reference (beta*hbar*omega = {beta_hbar_omega:.3f})")

# Center-of-mass correction (for finite system)
# F_cm = -kB*T * ln(N) * 3/2 / N  -- small for large N
F_cm = -1.5 * kB * TEMPERATURE * np.log(n_atoms) / n_atoms

# Total Helmholtz free energy per atom
F_total = F_einstein + W_avg + F_cm

print(f"\n  F_einstein:      {F_einstein:.6f} eV/atom")
print(f"  F_cm:            {F_cm:.6f} eV/atom")
print(f"  W_switching:     {W_avg:.6f} eV/atom")
print(f"  -----------------------------------------")
print(f"  F_total:         {F_total:.6f} eV/atom")
print(f"  F_total error:   {W_err:.6f} eV/atom")

# ============================================================
# 6. SAVE RESULTS AND PLOT
# ============================================================

results = {
    "element": ELEMENT,
    "crystal_structure": CRYSTAL_STRUCTURE,
    "temperature_K": TEMPERATURE,
    "pressure_bar": PRESSURE,
    "n_atoms": n_atoms,
    "spring_constant_eV_per_A2": SPRING_CONSTANT,
    "F_total_eV_per_atom": float(F_total),
    "F_total_error_eV_per_atom": float(W_err),
    "F_einstein_eV_per_atom": float(F_einstein),
    "W_switching_eV_per_atom": float(W_avg),
    "W_forward_eV_per_atom": float(W_forward),
    "W_backward_eV_per_atom": float(W_backward),
    "F_cm_eV_per_atom": float(F_cm),
}

with open(os.path.join(WORKDIR, "results.json"), "w") as f:
    json.dump(results, f, indent=2)

print(f"\nResults saved to {WORKDIR}/results.json")

# Plot the switching integrand
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

ax1 = axes[0]
ax1.plot(fwd_lambda, fwd_dU / n_atoms, "b-", alpha=0.5, linewidth=0.8, label="Forward")
ax1.plot(bwd_lambda, bwd_dU / n_atoms, "r-", alpha=0.5, linewidth=0.8, label="Backward")
ax1.set_xlabel(r"$\lambda$", fontsize=13)
ax1.set_ylabel(r"$\Delta U / N$ (eV/atom)", fontsize=13)
ax1.set_title("Switching Integrand", fontsize=14)
ax1.legend(fontsize=11)
ax1.grid(alpha=0.3)

# Cumulative integral
fwd_cumint = np.array([
    -integrate.trapezoid(fwd_dU[:i+1] / n_atoms, fwd_lambda[:i+1])
    for i in range(len(fwd_lambda))
])
bwd_cumint = np.array([
    -integrate.trapezoid(bwd_dU[:i+1] / n_atoms, bwd_lambda[:i+1])
    for i in range(len(bwd_lambda))
])

ax2 = axes[1]
ax2.plot(fwd_lambda, fwd_cumint, "b-", linewidth=1.5, label="Forward")
ax2.plot(bwd_lambda, bwd_cumint, "r--", linewidth=1.5, label="Backward")
ax2.set_xlabel(r"$\lambda$", fontsize=13)
ax2.set_ylabel("Cumulative Work (eV/atom)", fontsize=13)
ax2.set_title("Cumulative Switching Work", fontsize=14)
ax2.legend(fontsize=11)
ax2.grid(alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(WORKDIR, "switching_integrand.png"), dpi=200, bbox_inches="tight")
print(f"Plot saved to {WORKDIR}/switching_integrand.png")
```

### Method B: Free Energy of a Liquid Phase (Uhlenbeck-Ford Reference)

For liquids, the Frenkel-Ladd method does not work because atoms diffuse away from lattice sites. Instead, a Uhlenbeck-Ford (UF) reference fluid is used, or a two-step approach: (1) compute the free energy of the solid, (2) integrate along a reversible path (temperature integration) through the melting point.

The script below uses the more practical approach of computing the liquid free energy directly via thermodynamic integration with a soft reference potential.

```python
#!/usr/bin/env python3
"""
Liquid free energy calculation via thermodynamic integration.

Strategy: integrate from an ideal gas (known free energy) to the full
interacting liquid along a coupling parameter lambda.

U(lambda) = lambda * U_real
F_liquid = F_ideal_gas + integral_0^1 <U_real>_lambda dlambda

This requires running multiple short simulations at different lambda values
(discrete TI), which is more robust than continuous switching for liquids.

Reference: Frenkel & Smit, "Understanding Molecular Simulation", Ch. 7
"""

import os
import sys
import json
import subprocess
import numpy as np
from scipy import integrate

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from ase.build import bulk
from ase.io import write

# ============================================================
# 1. CONFIGURATION
# ============================================================

ELEMENT = "Cu"
LATTICE_CONSTANT = 3.615
SUPERCELL = (4, 4, 4)
TEMPERATURE = 1400.0          # K (above melting for Cu ~1358 K)
DENSITY = None                # atoms/A^3, None to use supercell volume
MASS = 63.546

# Potential
PAIR_STYLE = "eam/alloy"
PAIR_COEFF = "* * Cu_mishin1.eam.alloy Cu"
POTENTIAL_FILES = ["Cu_mishin1.eam.alloy"]

# MACE alternative:
# PAIR_STYLE = "mace no_domain_decomposition"
# PAIR_COEFF = "* * /path/to/mace_model.model Cu"
# POTENTIAL_FILES = []

TIMESTEP = 0.001
N_EQUIL = 20000               # equilibration per lambda point
N_SAMPLE = 10000              # sampling per lambda point
N_THERMO = 50

# Lambda grid for discrete TI (denser near 0 where the integrand changes rapidly)
LAMBDA_GRID = np.concatenate([
    np.linspace(0.0, 0.1, 6),
    np.linspace(0.15, 0.5, 8),
    np.linspace(0.6, 1.0, 5),
])
LAMBDA_GRID = np.unique(LAMBDA_GRID)

LAMMPS_CMD = "lmp"
WORKDIR = os.path.abspath("free_energy_liquid")
os.makedirs(WORKDIR, exist_ok=True)

# ============================================================
# 2. BUILD INITIAL (LIQUID) STRUCTURE
# ============================================================

print("=" * 60)
print("Liquid Free Energy Calculation (Discrete TI)")
print("=" * 60)

atoms = bulk(ELEMENT, "fcc", a=LATTICE_CONSTANT, cubic=True)
atoms = atoms.repeat(SUPERCELL)
n_atoms = len(atoms)
volume = atoms.get_volume()
density = n_atoms / volume

print(f"  Element: {ELEMENT}")
print(f"  N_atoms: {n_atoms}")
print(f"  Volume: {volume:.2f} A^3")
print(f"  Density: {density:.6f} atoms/A^3")
print(f"  Temperature: {TEMPERATURE} K")
print(f"  Lambda points: {len(LAMBDA_GRID)}")

data_file = os.path.join(WORKDIR, "structure.data")
write(data_file, atoms, format="lammps-data")

# ============================================================
# 3. MELT THE STRUCTURE FIRST
# ============================================================

print("\n--- Melting the structure ---")

melt_script = f"""# Melt the solid to create a liquid configuration
units          metal
atom_style     atomic
boundary       p p p

read_data      structure.data

mass           1 {MASS}

pair_style     {PAIR_STYLE}
pair_coeff     {PAIR_COEFF}

# Heat above melting point
velocity       all create {TEMPERATURE * 2} 99999 dist gaussian

fix            1 all npt temp {TEMPERATURE * 2} {TEMPERATURE * 2} 0.1 iso 0 0 1.0
thermo_style   custom step temp pe press vol density
thermo         500
run            20000
unfix          1

# Cool to target temperature
fix            2 all npt temp {TEMPERATURE * 2} {TEMPERATURE} 0.1 iso 0 0 1.0
run            10000
unfix          2

# Equilibrate at target T
fix            3 all npt temp {TEMPERATURE} {TEMPERATURE} 0.1 iso 0 0 1.0
run            20000
unfix          3

write_data     liquid.data
"""

melt_file = os.path.join(WORKDIR, "melt.in")
with open(melt_file, "w") as f:
    f.write(melt_script)

result = subprocess.run(
    [LAMMPS_CMD, "-in", "melt.in"],
    cwd=WORKDIR,
    capture_output=True, text=True, timeout=3600,
)

with open(os.path.join(WORKDIR, "melt.log"), "w") as f:
    f.write(result.stdout)

if result.returncode != 0:
    print(f"ERROR: Melting failed!")
    print(result.stderr[-1000:] if result.stderr else "")
    sys.exit(1)

print("  Liquid structure prepared.")

# ============================================================
# 4. DISCRETE TI: RUN AT EACH LAMBDA
# ============================================================

print("\n--- Discrete Thermodynamic Integration ---")

mean_pe = []
std_pe = []

for idx, lam in enumerate(LAMBDA_GRID):
    print(f"  Lambda = {lam:.4f} ({idx+1}/{len(LAMBDA_GRID)})", end=" ... ")

    if lam < 1e-10:
        # Ideal gas limit: <U> = 0
        mean_pe.append(0.0)
        std_pe.append(0.0)
        print("ideal gas (skipped)")
        continue

    # For lambda < 1, we use pair_style hybrid/scaled or fix adapt
    # Simpler approach: scale the potential via pair_coeff with a prefactor
    # Most general: use fix adapt to scale pair interactions

    ti_script = f"""# TI at lambda = {lam:.6f}
units          metal
atom_style     atomic
boundary       p p p

read_data      liquid.data

mass           1 {MASS}

pair_style     {PAIR_STYLE}
pair_coeff     {PAIR_COEFF}

# Scale pair interactions by lambda using fix adapt
variable       scale equal {lam}

velocity       all create {TEMPERATURE} {12345 + idx * 1000} dist gaussian

fix            thermostat all nvt temp {TEMPERATURE} {TEMPERATURE} 0.1

# fix adapt to scale the pair potential
fix            scale_pot all adapt 0 pair {PAIR_STYLE.split()[0]} scale * * v_scale

# Equilibration
thermo_style   custom step temp pe press
thermo         {N_THERMO}
run            {N_EQUIL}

# Sampling: record potential energy
# The PE reported by LAMMPS is already scaled by lambda,
# so the unscaled PE is pe/lambda (the integrand we need is <U_real>_lambda)
compute        pe_comp all pe
variable       pe_unscaled equal c_pe_comp/{lam if lam > 1e-10 else 1.0}

fix            pe_avg all ave/time 1 {N_SAMPLE} {N_SAMPLE} v_pe_unscaled file pe_lambda_{idx:03d}.dat

run            {N_SAMPLE}
"""

    ti_file = os.path.join(WORKDIR, f"ti_{idx:03d}.in")
    with open(ti_file, "w") as f:
        f.write(ti_script)

    result = subprocess.run(
        [LAMMPS_CMD, "-in", f"ti_{idx:03d}.in"],
        cwd=WORKDIR,
        capture_output=True, text=True, timeout=1800,
    )

    if result.returncode != 0:
        print(f"FAILED")
        print(result.stderr[-500:] if result.stderr else "")
        mean_pe.append(np.nan)
        std_pe.append(np.nan)
        continue

    # Parse PE from output
    try:
        pe_data = np.loadtxt(
            os.path.join(WORKDIR, f"pe_lambda_{idx:03d}.dat"), comments="#"
        )
        if pe_data.ndim == 0 or len(pe_data.shape) < 2:
            # Single value
            pe_val = float(pe_data) if pe_data.ndim == 0 else float(pe_data[-1])
            mean_pe.append(pe_val / n_atoms)
            std_pe.append(0.0)
        else:
            pe_vals = pe_data[:, 1] / n_atoms
            mean_pe.append(np.mean(pe_vals))
            std_pe.append(np.std(pe_vals) / np.sqrt(len(pe_vals)))
    except Exception as e:
        # Fallback: parse from thermo output
        pe_vals = []
        for line in result.stdout.split("\n"):
            parts = line.split()
            if len(parts) >= 3:
                try:
                    step = int(parts[0])
                    pe = float(parts[2])
                    if step > N_EQUIL:
                        pe_vals.append(pe / n_atoms)
                except ValueError:
                    continue
        if pe_vals:
            mean_pe.append(np.mean(pe_vals))
            std_pe.append(np.std(pe_vals) / np.sqrt(len(pe_vals)))
        else:
            mean_pe.append(np.nan)
            std_pe.append(np.nan)

    print(f"<U/N> = {mean_pe[-1]:.6f} eV/atom")

mean_pe = np.array(mean_pe)
std_pe = np.array(std_pe)

# ============================================================
# 5. INTEGRATE AND COMPUTE FREE ENERGY
# ============================================================

print("\n--- Computing Free Energy ---")

kB = 8.617333262e-5  # eV/K

# Remove NaN values
valid = ~np.isnan(mean_pe)
lam_valid = LAMBDA_GRID[valid]
pe_valid = mean_pe[valid]

# Integrate: F_excess = integral_0^1 <U_real>_lambda dlambda
F_excess = integrate.trapezoid(pe_valid, lam_valid)

# Ideal gas free energy (per atom)
# F_ideal = kB * T * [ln(rho * Lambda^3) - 1]
# Lambda = thermal de Broglie wavelength = h / sqrt(2 * pi * m * kB * T)
h = 4.135667696e-15          # eV*s
amu_to_kg = 1.66053906660e-27
Lambda_dB = h / np.sqrt(2 * np.pi * MASS * amu_to_kg * kB * TEMPERATURE * 1.60218e-19)
Lambda_dB_A = Lambda_dB * 1e10  # convert m to Angstrom

F_ideal = kB * TEMPERATURE * (np.log(density * Lambda_dB_A**3) - 1.0)

F_total = F_ideal + F_excess

print(f"  Thermal de Broglie wavelength: {Lambda_dB_A:.4f} A")
print(f"  F_ideal:    {F_ideal:.6f} eV/atom")
print(f"  F_excess:   {F_excess:.6f} eV/atom")
print(f"  -----------------------------------------")
print(f"  F_liquid:   {F_total:.6f} eV/atom")

# ============================================================
# 6. SAVE AND PLOT
# ============================================================

results = {
    "element": ELEMENT,
    "phase": "liquid",
    "temperature_K": TEMPERATURE,
    "n_atoms": n_atoms,
    "density_atoms_per_A3": density,
    "F_total_eV_per_atom": float(F_total),
    "F_ideal_eV_per_atom": float(F_ideal),
    "F_excess_eV_per_atom": float(F_excess),
    "lambda_grid": LAMBDA_GRID.tolist(),
    "mean_pe_per_atom": mean_pe.tolist(),
}

with open(os.path.join(WORKDIR, "results.json"), "w") as f:
    json.dump(results, f, indent=2)

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

ax1 = axes[0]
ax1.errorbar(lam_valid, pe_valid, yerr=std_pe[valid], fmt="o-", color="steelblue",
             markersize=5, capsize=3, label=r"$\langle U_{\mathrm{real}} \rangle_\lambda / N$")
ax1.set_xlabel(r"$\lambda$", fontsize=13)
ax1.set_ylabel("Potential Energy (eV/atom)", fontsize=13)
ax1.set_title("TI Integrand", fontsize=14)
ax1.legend(fontsize=11)
ax1.grid(alpha=0.3)

# Cumulative integral
cumint = np.array([
    integrate.trapezoid(pe_valid[:i+1], lam_valid[:i+1])
    for i in range(len(lam_valid))
])
ax2 = axes[1]
ax2.plot(lam_valid, cumint, "o-", color="darkorange", markersize=5)
ax2.axhline(F_excess, color="red", linestyle="--", alpha=0.7, label=f"$F_{{excess}}$ = {F_excess:.4f} eV/atom")
ax2.set_xlabel(r"$\lambda$", fontsize=13)
ax2.set_ylabel("Cumulative Integral (eV/atom)", fontsize=13)
ax2.set_title("Cumulative Excess Free Energy", fontsize=14)
ax2.legend(fontsize=11)
ax2.grid(alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(WORKDIR, "ti_integrand.png"), dpi=200, bbox_inches="tight")
print(f"\nPlot saved to {WORKDIR}/ti_integrand.png")
print(f"Results saved to {WORKDIR}/results.json")
```

### Method C: Reversible Scaling (Free Energy vs Temperature)

Once the free energy is known at a single temperature T0 (from Method A or B), reversible scaling extends it to a range of temperatures using the Gibbs-Helmholtz equation.

```python
#!/usr/bin/env python3
"""
Reversible scaling (temperature integration) to obtain free energy
as a function of temperature.

Given F(T0) from a Frenkel-Ladd or liquid TI calculation, this script
uses the Gibbs-Helmholtz equation:

  d(F/T)/d(1/T) = U(T)

to integrate F(T) over a temperature range.

In practice, LAMMPS runs an MD simulation where the temperature is
continuously swept from T0 to T1, recording <U> as a function of T.

Reference: de Koning & Antonelli, Phys. Rev. E 53, 465 (1996)
"""

import os
import sys
import json
import subprocess
import numpy as np
from scipy import integrate, interpolate

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from ase.build import bulk
from ase.io import write

# ============================================================
# 1. CONFIGURATION
# ============================================================

ELEMENT = "Cu"
CRYSTAL_STRUCTURE = "fcc"
LATTICE_CONSTANT = 3.615
SUPERCELL = (4, 4, 4)
MASS = 63.546

# Reference free energy (from Method A or B)
T_REF = 800.0                  # K
F_REF = -3.50                  # eV/atom (REPLACE with your Method A result)

# Temperature range for reversible scaling
T_START = 300.0                # K
T_END = 1300.0                 # K

# Phase: "solid" or "liquid"
PHASE = "solid"

# Potential
PAIR_STYLE = "eam/alloy"
PAIR_COEFF = "* * Cu_mishin1.eam.alloy Cu"

TIMESTEP = 0.001
N_EQUIL = 15000
N_SWEEP = 50000                # steps for the temperature sweep
N_PRINT = 50

LAMMPS_CMD = "lmp"
WORKDIR = os.path.abspath("free_energy_reversible_scaling")
os.makedirs(WORKDIR, exist_ok=True)

# ============================================================
# 2. BUILD STRUCTURE
# ============================================================

print("=" * 60)
print("Reversible Scaling (Temperature Integration)")
print("=" * 60)

atoms = bulk(ELEMENT, CRYSTAL_STRUCTURE, a=LATTICE_CONSTANT, cubic=True)
atoms = atoms.repeat(SUPERCELL)
n_atoms = len(atoms)

data_file = os.path.join(WORKDIR, "structure.data")
write(data_file, atoms, format="lammps-data")

print(f"  Element: {ELEMENT}, Phase: {PHASE}")
print(f"  N_atoms: {n_atoms}")
print(f"  T_ref: {T_REF} K, F_ref: {F_REF} eV/atom")
print(f"  T range: {T_START} -- {T_END} K")

# ============================================================
# 3. FORWARD TEMPERATURE SWEEP
# ============================================================

print("\n--- Forward Temperature Sweep ---")

sweep_script = f"""# Reversible scaling: temperature sweep
units          metal
atom_style     atomic
boundary       p p p

read_data      structure.data

mass           1 {MASS}

pair_style     {PAIR_STYLE}
pair_coeff     {PAIR_COEFF}

# Equilibrate at starting temperature
velocity       all create {T_START} 54321 dist gaussian

fix            eq all npt temp {T_START} {T_START} 0.1 iso 0 0 1.0
thermo         500
thermo_style   custom step temp pe ke etotal press vol
run            {N_EQUIL}
unfix          eq

# Forward sweep: T_START -> T_END
variable       T_current equal ramp({T_START},{T_END})

fix            sweep all npt temp {T_START} {T_END} 0.1 iso 0 0 1.0

# Record T, PE, KE, enthalpy, volume, pressure
compute        pe_comp all pe
compute        ke_comp all ke
variable       enthalpy equal (c_pe_comp+c_ke_comp+press*vol*0.0006242)

thermo_style   custom step v_T_current temp c_pe_comp c_ke_comp press vol
thermo         {N_PRINT}

fix            fwd_out all print {N_PRINT} "${{T_current}} ${{temp}} ${{pe_comp}} ${{press}} ${{vol}}" &
               file forward_sweep.dat screen no &
               title "# T_target  T_actual  PE  Press  Vol"

run            {N_SWEEP}
unfix          sweep
unfix          fwd_out

# Backward sweep: T_END -> T_START
fix            sweep_bk all npt temp {T_END} {T_START} 0.1 iso 0 0 1.0

fix            bwd_out all print {N_PRINT} "${{T_current}} ${{temp}} ${{pe_comp}} ${{press}} ${{vol}}" &
               file backward_sweep.dat screen no &
               title "# T_target  T_actual  PE  Press  Vol"

# For backward, redefine T_current
variable       T_bk equal ramp({T_END},{T_START})

run            {N_SWEEP}
"""

sweep_file = os.path.join(WORKDIR, "sweep.in")
with open(sweep_file, "w") as f:
    f.write(sweep_script)

result = subprocess.run(
    [LAMMPS_CMD, "-in", "sweep.in"],
    cwd=WORKDIR,
    capture_output=True, text=True, timeout=3600,
)

with open(os.path.join(WORKDIR, "sweep.log"), "w") as f:
    f.write(result.stdout)

if result.returncode != 0:
    print(f"ERROR: Sweep failed!")
    print(result.stderr[-1000:] if result.stderr else "")
    sys.exit(1)

print("  Temperature sweeps completed.")

# ============================================================
# 4. COMPUTE F(T) VIA GIBBS-HELMHOLTZ INTEGRATION
# ============================================================

print("\n--- Computing F(T) ---")

kB = 8.617333262e-5  # eV/K

# Read sweep data
try:
    fwd_data = np.loadtxt(os.path.join(WORKDIR, "forward_sweep.dat"), comments="#")
    bwd_data = np.loadtxt(os.path.join(WORKDIR, "backward_sweep.dat"), comments="#")
except Exception as e:
    print(f"ERROR reading sweep data: {e}")
    sys.exit(1)

# Columns: T_target, T_actual, PE, Press, Vol
T_fwd = fwd_data[:, 0]
PE_fwd = fwd_data[:, 2] / n_atoms  # per atom
T_bwd = bwd_data[:, 0]
PE_bwd = bwd_data[:, 2] / n_atoms

# Sort by temperature
fwd_sort = np.argsort(T_fwd)
T_fwd = T_fwd[fwd_sort]
PE_fwd = PE_fwd[fwd_sort]

bwd_sort = np.argsort(T_bwd)
T_bwd = T_bwd[bwd_sort]
PE_bwd = PE_bwd[bwd_sort]

# Average forward and backward
# Interpolate both onto a common temperature grid
T_common = np.linspace(max(T_fwd[0], T_bwd[0]), min(T_fwd[-1], T_bwd[-1]), 500)

interp_fwd = interpolate.interp1d(T_fwd, PE_fwd, kind="linear", fill_value="extrapolate")
interp_bwd = interpolate.interp1d(T_bwd, PE_bwd, kind="linear", fill_value="extrapolate")

PE_avg = (interp_fwd(T_common) + interp_bwd(T_common)) / 2.0

# Gibbs-Helmholtz integration:
# d(F/T) / d(1/T) = U(T)  =>  d(F/T) = U * d(1/T)
# F(T)/T = F(T0)/T0 + integral_{1/T0}^{1/T} U(T') d(1/T')

# We integrate in terms of beta = 1/(kB*T) for numerical stability
beta_common = 1.0 / (kB * T_common)
beta_ref = 1.0 / (kB * T_REF)

# Find index closest to T_REF
idx_ref = np.argmin(np.abs(T_common - T_REF))

# Integrate from T_REF outward
F_over_T = np.zeros_like(T_common)
F_over_T[idx_ref] = F_REF / T_REF

# Integrate forward (T_REF -> higher T, i.e., smaller beta)
for i in range(idx_ref + 1, len(T_common)):
    d_inv_T = 1.0 / T_common[i] - 1.0 / T_common[i - 1]
    U_avg = (PE_avg[i] + PE_avg[i - 1]) / 2.0
    F_over_T[i] = F_over_T[i - 1] + U_avg * d_inv_T

# Integrate backward (T_REF -> lower T, i.e., larger beta)
for i in range(idx_ref - 1, -1, -1):
    d_inv_T = 1.0 / T_common[i] - 1.0 / T_common[i + 1]
    U_avg = (PE_avg[i] + PE_avg[i + 1]) / 2.0
    F_over_T[i] = F_over_T[i + 1] + U_avg * d_inv_T

F_of_T = F_over_T * T_common

print(f"  F({T_START:.0f} K) = {F_of_T[0]:.6f} eV/atom")
print(f"  F({T_REF:.0f} K) = {F_of_T[idx_ref]:.6f} eV/atom (reference)")
print(f"  F({T_END:.0f} K) = {F_of_T[-1]:.6f} eV/atom")

# ============================================================
# 5. SAVE AND PLOT
# ============================================================

results = {
    "element": ELEMENT,
    "phase": PHASE,
    "T_ref_K": T_REF,
    "F_ref_eV_per_atom": F_REF,
    "T_range_K": [float(T_START), float(T_END)],
    "temperature_K": T_common.tolist(),
    "free_energy_eV_per_atom": F_of_T.tolist(),
    "potential_energy_eV_per_atom": PE_avg.tolist(),
}

with open(os.path.join(WORKDIR, "results.json"), "w") as f:
    json.dump(results, f, indent=2)

# Save tabulated data
np.savetxt(
    os.path.join(WORKDIR, "free_energy_vs_T.dat"),
    np.column_stack([T_common, F_of_T, PE_avg]),
    header="T(K)  F(eV/atom)  U(eV/atom)",
    fmt="%.4f  %.8f  %.8f",
)

fig, axes = plt.subplots(1, 3, figsize=(16, 5))

ax1 = axes[0]
ax1.plot(T_fwd, PE_fwd, "b-", alpha=0.3, linewidth=0.5, label="Forward")
ax1.plot(T_bwd, PE_bwd, "r-", alpha=0.3, linewidth=0.5, label="Backward")
ax1.plot(T_common, PE_avg, "k-", linewidth=1.5, label="Average")
ax1.set_xlabel("Temperature (K)", fontsize=13)
ax1.set_ylabel("U (eV/atom)", fontsize=13)
ax1.set_title("Potential Energy vs T", fontsize=14)
ax1.legend(fontsize=10)
ax1.grid(alpha=0.3)

ax2 = axes[1]
ax2.plot(T_common, F_of_T, "k-", linewidth=2)
ax2.axvline(T_REF, color="red", linestyle="--", alpha=0.5, label=f"$T_{{ref}}$ = {T_REF:.0f} K")
ax2.set_xlabel("Temperature (K)", fontsize=13)
ax2.set_ylabel("F (eV/atom)", fontsize=13)
ax2.set_title("Helmholtz Free Energy vs T", fontsize=14)
ax2.legend(fontsize=10)
ax2.grid(alpha=0.3)

ax3 = axes[2]
S_of_T = -(np.gradient(F_of_T, T_common))  # S = -dF/dT
S_kB = S_of_T / kB  # in units of kB
ax3.plot(T_common, S_kB, "g-", linewidth=2)
ax3.set_xlabel("Temperature (K)", fontsize=13)
ax3.set_ylabel(r"S / $k_B$ (per atom)", fontsize=13)
ax3.set_title("Entropy vs T", fontsize=14)
ax3.grid(alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(WORKDIR, "free_energy_vs_T.png"), dpi=200, bbox_inches="tight")
print(f"\nPlots saved to {WORKDIR}/free_energy_vs_T.png")
print(f"Results saved to {WORKDIR}/results.json")
print(f"Tabulated data saved to {WORKDIR}/free_energy_vs_T.dat")
```

### Method D: Phase Stability Comparison and Melting Point

This script takes the free energy results from solid (Method A) and liquid (Method B) calculations and determines the melting point and relative phase stability.

```python
#!/usr/bin/env python3
"""
Phase stability comparison: determine melting point from free energy curves.

Requires: free energy vs temperature for both solid and liquid phases
(from Methods A+C for solid, Methods B+C for liquid, or individual
point calculations at multiple temperatures).

The melting point is where F_solid(T_m) = F_liquid(T_m).
"""

import os
import json
import numpy as np
from scipy import interpolate, optimize

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ============================================================
# 1. LOAD FREE ENERGY DATA
# ============================================================

print("=" * 60)
print("Phase Stability Comparison")
print("=" * 60)

# Option 1: Load from reversible scaling results
# Uncomment and adjust paths as needed:
# with open("free_energy_reversible_scaling_solid/results.json") as f:
#     solid_data = json.load(f)
# with open("free_energy_reversible_scaling_liquid/results.json") as f:
#     liquid_data = json.load(f)
# T_solid = np.array(solid_data["temperature_K"])
# F_solid = np.array(solid_data["free_energy_eV_per_atom"])
# T_liquid = np.array(liquid_data["temperature_K"])
# F_liquid = np.array(liquid_data["free_energy_eV_per_atom"])

# Option 2: Manual input from individual calculations at different T
# Replace these with your actual computed values.
# Each row: (temperature_K, F_eV_per_atom)
solid_points = np.array([
    [300,  -3.480],
    [500,  -3.520],
    [700,  -3.580],
    [900,  -3.660],
    [1000, -3.710],
    [1100, -3.770],
    [1200, -3.840],
    [1300, -3.920],
    [1400, -4.010],
    [1500, -4.110],
])

liquid_points = np.array([
    [300,  -3.100],
    [500,  -3.200],
    [700,  -3.350],
    [900,  -3.530],
    [1000, -3.630],
    [1100, -3.730],
    [1200, -3.840],
    [1300, -3.960],
    [1400, -4.080],
    [1500, -4.210],
])

T_solid = solid_points[:, 0]
F_solid = solid_points[:, 1]
T_liquid = liquid_points[:, 0]
F_liquid = liquid_points[:, 1]

# ============================================================
# 2. INTERPOLATE AND FIND CROSSING
# ============================================================

# Common temperature range
T_min = max(T_solid.min(), T_liquid.min())
T_max = min(T_solid.max(), T_liquid.max())
T_common = np.linspace(T_min, T_max, 1000)

# Interpolate
f_solid_interp = interpolate.interp1d(T_solid, F_solid, kind="cubic", fill_value="extrapolate")
f_liquid_interp = interpolate.interp1d(T_liquid, F_liquid, kind="cubic", fill_value="extrapolate")

F_s = f_solid_interp(T_common)
F_l = f_liquid_interp(T_common)

# Find crossing point(s): F_solid = F_liquid
delta_F = F_s - F_l

# Find sign changes
crossings = []
for i in range(len(delta_F) - 1):
    if delta_F[i] * delta_F[i + 1] < 0:
        # Linear interpolation for the crossing temperature
        T_cross = T_common[i] - delta_F[i] * (T_common[i + 1] - T_common[i]) / (delta_F[i + 1] - delta_F[i])
        crossings.append(T_cross)

        # Refine with root finding
        try:
            diff_func = lambda T: float(f_solid_interp(T) - f_liquid_interp(T))
            T_refined = optimize.brentq(diff_func, T_common[i], T_common[i + 1])
            crossings[-1] = T_refined
        except Exception:
            pass

print(f"\n  Temperature range: {T_min:.0f} -- {T_max:.0f} K")

if crossings:
    for i, T_m in enumerate(crossings):
        F_at_Tm = float(f_solid_interp(T_m))
        print(f"\n  Crossing point {i+1}:")
        print(f"    Melting temperature: {T_m:.1f} K")
        print(f"    Free energy at T_m:  {F_at_Tm:.6f} eV/atom")
        print(f"    Delta_F at T_m:      {float(f_solid_interp(T_m) - f_liquid_interp(T_m)):.8f} eV/atom")
else:
    print("\n  No crossing found in the temperature range.")
    print(f"  Delta_F at {T_min:.0f} K: {delta_F[0]:.6f} eV/atom")
    print(f"  Delta_F at {T_max:.0f} K: {delta_F[-1]:.6f} eV/atom")
    if delta_F[0] < 0:
        print("  Solid is more stable throughout the range.")
    else:
        print("  Liquid is more stable throughout the range.")

# Phase stability at specific temperatures
print("\n  Phase stability summary:")
print(f"  {'T (K)':>8s}  {'F_solid':>12s}  {'F_liquid':>12s}  {'Delta_F':>12s}  {'Stable Phase':>14s}")
print(f"  {'':->8s}  {'':->12s}  {'':->12s}  {'':->12s}  {'':->14s}")
for T_check in [300, 500, 800, 1000, 1200, 1400]:
    if T_min <= T_check <= T_max:
        fs = float(f_solid_interp(T_check))
        fl = float(f_liquid_interp(T_check))
        df = fs - fl
        stable = "solid" if df < 0 else "liquid"
        print(f"  {T_check:8.0f}  {fs:12.6f}  {fl:12.6f}  {df:12.6f}  {stable:>14s}")

# ============================================================
# 3. PLOT
# ============================================================

fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))

ax1 = axes[0]
ax1.plot(T_common, F_s, "b-", linewidth=2, label="Solid (FCC)")
ax1.plot(T_common, F_l, "r-", linewidth=2, label="Liquid")
ax1.plot(T_solid, F_solid, "bs", markersize=7, zorder=5)
ax1.plot(T_liquid, F_liquid, "ro", markersize=7, zorder=5)
for T_m in crossings:
    F_m = float(f_solid_interp(T_m))
    ax1.axvline(T_m, color="green", linestyle="--", alpha=0.7)
    ax1.plot(T_m, F_m, "g*", markersize=15, zorder=10, label=f"$T_m$ = {T_m:.0f} K")
ax1.set_xlabel("Temperature (K)", fontsize=13)
ax1.set_ylabel("Helmholtz Free Energy (eV/atom)", fontsize=13)
ax1.set_title("Phase Free Energies", fontsize=14)
ax1.legend(fontsize=11)
ax1.grid(alpha=0.3)

ax2 = axes[1]
ax2.plot(T_common, delta_F * 1000, "k-", linewidth=2)  # in meV
ax2.axhline(0, color="gray", linestyle="-", linewidth=0.5)
for T_m in crossings:
    ax2.axvline(T_m, color="green", linestyle="--", alpha=0.7, label=f"$T_m$ = {T_m:.0f} K")
ax2.fill_between(T_common, delta_F * 1000, 0, where=(delta_F < 0), alpha=0.2, color="blue", label="Solid stable")
ax2.fill_between(T_common, delta_F * 1000, 0, where=(delta_F > 0), alpha=0.2, color="red", label="Liquid stable")
ax2.set_xlabel("Temperature (K)", fontsize=13)
ax2.set_ylabel(r"$\Delta F = F_{solid} - F_{liquid}$ (meV/atom)", fontsize=13)
ax2.set_title("Free Energy Difference", fontsize=14)
ax2.legend(fontsize=10)
ax2.grid(alpha=0.3)

plt.tight_layout()
plt.savefig("phase_stability.png", dpi=200, bbox_inches="tight")
print(f"\nPlot saved to phase_stability.png")

# Save results
results = {
    "melting_temperatures_K": [float(T) for T in crossings],
    "T_range_K": [float(T_min), float(T_max)],
}
with open("phase_stability.json", "w") as f:
    json.dump(results, f, indent=2)
print("Results saved to phase_stability.json")
```

### Method E: Using MACE as the LAMMPS Pair Style

To use a MACE machine-learning potential instead of a classical EAM/MEAM potential, modify the potential configuration in any of the scripts above. The container's LAMMPS is compiled with the ML-MACE package.

```python
#!/usr/bin/env python3
"""
Example: prepare a MACE model file for LAMMPS free energy calculations.

MACE foundation models can be downloaded and converted to LAMMPS-compatible
format. This script shows how to set up the potential configuration.
"""

import os
import warnings
warnings.filterwarnings("ignore")

# ============================================================
# Option 1: Use MACE foundation model (mace-mp-0)
# ============================================================

# The MACE-MP-0 model can be loaded from the mace package
from mace.calculators import mace_mp

# This downloads and caches the model
calc = mace_mp(model="medium", device="cpu", default_dtype="float64")

# Find the model file path
model_path = None
for attr in ["model_path", "model_paths"]:
    if hasattr(calc, attr):
        mp = getattr(calc, attr)
        if isinstance(mp, (list, tuple)):
            model_path = mp[0]
        elif isinstance(mp, str):
            model_path = mp
        break

if model_path is None:
    # The model is stored in the torch cache
    import torch
    cache_dir = os.path.join(torch.hub.get_dir(), "checkpoints")
    for f in os.listdir(cache_dir):
        if "mace" in f.lower() and f.endswith(".model"):
            model_path = os.path.join(cache_dir, f)
            break

if model_path:
    print(f"MACE model file: {model_path}")
else:
    print("Could not locate MACE model file automatically.")
    print("Download manually from: https://github.com/ACEsuit/mace-mp")
    model_path = "/path/to/your/mace_model.model"

# ============================================================
# LAMMPS Configuration for MACE
# ============================================================

# For any of the free energy scripts (Methods A, B, C), replace:
#
#   PAIR_STYLE = "eam/alloy"
#   PAIR_COEFF = "* * Cu_mishin1.eam.alloy Cu"
#
# with:
#
#   PAIR_STYLE = "mace no_domain_decomposition"
#   PAIR_COEFF = f"* * {model_path} Cu"
#
# Notes:
# - "no_domain_decomposition" is required for MACE in LAMMPS
# - List ALL element symbols in pair_coeff in the order matching your type map
# - For multi-element systems: PAIR_COEFF = f"* * {model_path} Cu Zn"
# - MACE is significantly slower than EAM, so use smaller supercells
#   (3x3x3 instead of 4x4x4) and fewer switching steps

print("\n--- LAMMPS input configuration for MACE ---")
print(f'PAIR_STYLE = "mace no_domain_decomposition"')
print(f'PAIR_COEFF = "* * {model_path} Cu"')
print()

# ============================================================
# Option 2: Custom-trained MACE model
# ============================================================

# If you have a custom MACE model trained on DFT data:
#   PAIR_STYLE = "mace no_domain_decomposition"
#   PAIR_COEFF = "* * /path/to/custom_mace.model Element1 Element2"

print("--- For custom MACE models ---")
print('PAIR_STYLE = "mace no_domain_decomposition"')
print('PAIR_COEFF = "* * /path/to/custom_mace.model Element1 Element2"')
print()

# ============================================================
# MACE-specific considerations for free energy calculations
# ============================================================

print("--- MACE-specific tips for free energy calculations ---")
print("1. Use smaller supercells (3x3x3) due to higher computational cost")
print("2. MACE does not support fix adapt for pair scaling -- use")
print("   hybrid/overlay with a soft potential for switching instead")
print("3. For Frenkel-Ladd: the Einstein crystal springs (fix spring/self)")
print("   work with any pair style including MACE")
print("4. For liquid TI: use the discrete TI approach (Method B) with")
print("   separate runs at each lambda value")
print("5. Typical wall times: ~10x slower than EAM for same system size")
print("6. Use GPU (device='cuda') if available for significant speedup")
```

---

## Key Parameters

| Parameter | Typical Value | Notes |
|---|---|---|
| `SUPERCELL` | (4,4,4) for EAM; (3,3,3) for MACE | Minimum ~100 atoms; larger reduces finite-size errors |
| `N_EQUIL` | 15000-25000 | Equilibration steps; increase for complex systems |
| `N_SWITCH` | 25000-50000 | Switching steps; more = less hysteresis but slower |
| `SPRING_CONSTANT` | 1-10 eV/A^2 | Auto-estimated from MSD; higher for stiff materials |
| `TIMESTEP` | 0.001 ps | Standard for metals; reduce for light elements (H, Li) |
| `TEMPERATURE` | Target T | Must be well below melting for solid, well above for liquid |
| `N_PRINT` | 50-200 | Frequency of integrand sampling; balance resolution vs file size |
| `LAMBDA_GRID` | 15-25 points | For discrete TI (liquid); denser near lambda=0 |
| `N_ITERATIONS` | 1-3 | Repeat switching to estimate statistical error |

## Interpreting Results

- **Free energy values** are reported in eV/atom. Multiply by the number of atoms per formula unit to get eV/formula unit.
- **Hysteresis** between forward and backward switching should be small (< 1-5 meV/atom). Large hysteresis indicates insufficient switching steps or equilibration.
- **The more negative free energy wins**: the phase with lower (more negative) F at a given T and P is thermodynamically stable.
- **Melting temperature** is where F_solid(T) = F_liquid(T). It is determined by the crossing of the two free energy curves.
- **Entropy** can be extracted as S = -dF/dT. A jump in entropy at T_m gives the latent heat: L = T_m * (S_liquid - S_solid).
- **Finite-size effects**: Free energy converges as ~1/N. Always test convergence with at least two supercell sizes.
- **Statistical error**: Run multiple switching iterations (N_ITERATIONS > 1) and report the standard error of the mean.
- **Classical vs quantum**: The Einstein crystal reference uses the quantum harmonic oscillator formula. For heavy elements at high T, classical and quantum results converge. For light elements (Li, H) or low T, the quantum correction matters.

## Common Issues

| Problem | Solution |
|---|---|
| Large hysteresis (forward != backward) | Increase `N_SWITCH` (switching steps). Typical: 50000-100000 for metals. |
| System melts during solid free energy calc | Reduce temperature or check that the structure is correctly equilibrated. Add a melting check (compare RDF to expected crystal). |
| Liquid freezes during liquid calculation | Increase temperature or start from a properly melted configuration. Use the melting script in Method B. |
| Spring constant too high/low | Auto-estimate from MSD (provided in Method A). Rule of thumb: k ~ 3*kB*T / <u^2> where <u^2> is the mean-square displacement. |
| LAMMPS crashes with MACE | Ensure `no_domain_decomposition` is specified. Check that the model file path is correct and the model supports your elements. |
| `fix adapt` not compatible with pair style | Not all pair styles support `fix adapt`. For MACE, use discrete TI (separate runs) instead of continuous switching. |
| Free energy not converging with system size | Increase supercell. For FCC metals, 4x4x4 (256 atoms) is usually sufficient; for complex structures, test 5x5x5. |
| Ideal gas reference diverges at lambda=0 | Skip lambda=0 (it is analytically zero). Start the lambda grid at lambda=0.001 or 0.01. |
| Pressure contribution not included | For Gibbs free energy at finite pressure: G = F + PV. The scripts compute Helmholtz F; add PV correction if P != 0. |
| Results differ from calphy | Ensure identical: potential, supercell size, switching steps, spring constant, and temperature. Small differences (~1 meV/atom) are expected from different random seeds. |
