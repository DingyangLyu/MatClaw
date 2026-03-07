# Amorphous Structure Generation and Analysis

## When to Use

- Generating amorphous / glassy structures from a crystalline starting point
- Modeling metallic glasses, oxide glasses, amorphous semiconductors
- Studying glass transition temperature (Tg)
- Analyzing short-range and medium-range order in disordered materials
- Preparing initial structures for further property calculations (e.g., mechanical, electronic)
- Comparing simulated amorphous structure against experimental X-ray/neutron scattering data
- Following the MPMorph workflow pattern: equilibrium volume search, high-T melt, stepwise quench, production run

## Method Selection

```
What do you need?

Generate a single amorphous structure for further calculations?
  --> Method A: ASE + MACE melt-quench (simple, works for any chemistry)

Generate amorphous structure for a metal or simple alloy?
  --> Method B: LAMMPS melt-quench (faster, larger systems, longer timescales)

Estimate glass transition temperature?
  --> Method A with volume-vs-temperature analysis during quench

Full structural characterization (RDF, coordination, bond angles, S(q))?
  --> Analysis section below (works with structures from either method)

Following atomate2 MPMorph pattern?
  --> Method A implements the equivalent workflow:
      1. (Optional) Equilibrium volume via EOS
      2. High-T equilibration
      3. Slow quench (stepwise temperature descent)
      4. (Optional) Fast quench to 0 K (relaxation)
```

## Prerequisites

Pre-installed: `ase`, `mace-torch`, `pymatgen`, `numpy`, `scipy`, `matplotlib`, `spglib`.

LAMMPS binary: `lmp` (with MANYBODY, KSPACE packages).

For LAMMPS potentials, download EAM/MEAM files from NIST Interatomic Potentials Repository.

## Detailed Steps

### Method A: ASE + MACE Melt-Quench Workflow

This implements the full MPMorph-inspired melt-quench protocol:
1. Relax crystalline structure
2. Build supercell
3. Heat to well above melting point
4. Equilibrate the liquid
5. Stepwise quench to target temperature (following atomate2 SlowQuenchMaker pattern)
6. Anneal at target temperature
7. (Optional) Fast quench to 0 K and relax

```python
#!/usr/bin/env python3
"""
Amorphous structure generation via melt-quench molecular dynamics.
Uses ASE + MACE following the MPMorph workflow pattern.

Protocol:
  1. Relax crystalline unit cell at 0 K
  2. Build supercell (>100 atoms, ideally >200)
  3. Heat to T_melt (well above experimental Tm)
  4. Equilibrate liquid at T_melt
  5. Stepwise quench: T_melt -> T_target in discrete steps
  6. Anneal at T_target
  7. (Optional) Relax to 0 K for final amorphous structure

Outputs:
  - Amorphous structure files (.cif, .xyz)
  - Volume vs T curve (for Tg estimation)
  - Thermodynamic data at each quench step
  - Trajectory file for post-processing
"""

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from ase.io import read, write, Trajectory
from ase.optimize import LBFGS
from ase.constraints import ExpCellFilter
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution, Stationary
from ase.md.langevin import Langevin
from ase.md.nptberendsen import NPTBerendsen
from ase import units

from mace.calculators import mace_mp

import warnings
warnings.filterwarnings("ignore")

# ============================================================
# 1. CONFIGURATION
# ============================================================

STRUCTURE_FILE = "structure.cif"       # Input crystalline structure
MACE_MODEL = "medium"                  # "small", "medium", "large"
DEVICE = "cpu"                         # "cpu" or "cuda"

# Supercell
SUPERCELL = (3, 3, 3)                  # Adjust so total atoms > 100

# Melt-quench temperatures (K)
T_MELT = 3000                          # Melting/liquid temperature (well above Tm)
T_TARGET = 300                         # Final target temperature
T_QUENCH_STEP = 200                    # Temperature step during quench (K)
# Quench rate ~ T_QUENCH_STEP / (N_QUENCH_PER_STEP * TIMESTEP)
# With defaults: 200 K / (2000 * 2 fs) = 200 / 4 ps = 5e13 K/s

# MD parameters
TIMESTEP = 2.0                         # fs (use 1.0 for light elements)
N_HEAT = 5000                          # Steps to heat to T_MELT
N_EQUIL_LIQUID = 10000                 # Steps to equilibrate liquid at T_MELT
N_QUENCH_PER_STEP = 2000              # Steps at each quench temperature
N_ANNEAL = 10000                       # Steps to anneal at T_TARGET
LOG_INTERVAL = 20                      # Thermo logging interval

# NPT settings
PRESSURE = 0.0                         # GPa

# Output
WORK_DIR = "/tmp/amorphous_gen"
os.makedirs(WORK_DIR, exist_ok=True)

# ============================================================
# 2. SET UP CALCULATOR AND STRUCTURE
# ============================================================

calc = mace_mp(model=MACE_MODEL, device=DEVICE, default_dtype="float64")

atoms = read(STRUCTURE_FILE)
formula = atoms.get_chemical_formula()
print(f"Input structure: {formula}")

# Relax unit cell at 0 K
atoms.calc = calc
ecf = ExpCellFilter(atoms)
opt = LBFGS(ecf, logfile=os.path.join(WORK_DIR, "relax.log"))
opt.run(fmax=0.01, steps=300)
e_crystal = atoms.get_potential_energy() / len(atoms)
v_crystal = atoms.get_volume() / len(atoms)
write(os.path.join(WORK_DIR, "relaxed_crystal.cif"), atoms)
print(f"Relaxed crystal: E = {e_crystal:.4f} eV/atom, V = {v_crystal:.3f} A^3/atom")

# Build supercell
atoms_sc = atoms.repeat(SUPERCELL)
atoms_sc.calc = calc
n_atoms = len(atoms_sc)
print(f"Supercell: {SUPERCELL}, {n_atoms} atoms")

if n_atoms < 100:
    print("WARNING: fewer than 100 atoms. Increase SUPERCELL for reliable amorphous generation.")

# ============================================================
# 3. HELPER: RUN NPT MD AT GIVEN TEMPERATURE
# ============================================================

def run_npt_md(atoms, temperature, n_steps, label="", collect_interval=None):
    """
    Run NPT MD at a fixed temperature.
    Returns dict with mean E, V, T and optional time series.

    This mirrors the atomate2 ForceFieldMDMaker.make() pattern:
    set temperature, n_steps, run, return output.
    """
    dt = TIMESTEP * units.fs
    pressure_au = PRESSURE * units.GPa

    dyn = NPTBerendsen(
        atoms,
        timestep=dt,
        temperature_K=temperature,
        pressure_au=pressure_au,
        taut=100 * units.fs,
        taup=1000 * units.fs,
        compressibility_au=4.57e-5 / units.bar,
    )

    energies = []
    volumes = []
    temps = []

    interval = collect_interval if collect_interval else LOG_INTERVAL

    def collect():
        energies.append(atoms.get_potential_energy() / n_atoms)
        volumes.append(atoms.get_volume() / n_atoms)
        temps.append(atoms.get_temperature())

    dyn.attach(collect, interval=interval)
    dyn.run(n_steps)

    # Clean up observer
    dyn.observers = []

    result = {
        "T_target": temperature,
        "T_actual": np.mean(temps[-len(temps)//2:]) if temps else temperature,
        "E_mean": np.mean(energies[-len(energies)//2:]) if energies else 0,
        "E_std": np.std(energies[-len(energies)//2:]) if energies else 0,
        "V_mean": np.mean(volumes[-len(volumes)//2:]) if volumes else 0,
        "V_std": np.std(volumes[-len(volumes)//2:]) if volumes else 0,
        "E_series": np.array(energies),
        "V_series": np.array(volumes),
        "T_series": np.array(temps),
    }

    if label:
        print(f"  [{label}] T_target={temperature:.0f} K, <T>={result['T_actual']:.0f} K, "
              f"<E>={result['E_mean']:.4f} eV/atom, <V>={result['V_mean']:.3f} A^3/atom")

    return result

# ============================================================
# 4. PHASE 1: HEAT TO LIQUID TEMPERATURE
# ============================================================

print(f"\n{'='*70}")
print(f"PHASE 1: Heating to {T_MELT} K")
print(f"{'='*70}")

# Initialize velocities at a moderate starting temperature
T_init = min(T_MELT, 600)
MaxwellBoltzmannDistribution(atoms_sc, temperature_K=T_init)
Stationary(atoms_sc)

# Ramp temperature using Langevin (more stable for large T jumps)
dt = TIMESTEP * units.fs
dyn_heat = Langevin(
    atoms_sc,
    timestep=dt,
    temperature_K=T_MELT,
    friction=0.01 / units.fs,
)

heat_data = {"T": [], "E": [], "V": []}

def collect_heat():
    heat_data["T"].append(atoms_sc.get_temperature())
    heat_data["E"].append(atoms_sc.get_potential_energy() / n_atoms)
    heat_data["V"].append(atoms_sc.get_volume() / n_atoms)

dyn_heat.attach(collect_heat, interval=LOG_INTERVAL)

print(f"  Heating {N_HEAT} steps ({N_HEAT * TIMESTEP / 1000:.1f} ps)...")
dyn_heat.run(N_HEAT)
print(f"  Final T = {atoms_sc.get_temperature():.0f} K")

# ============================================================
# 5. PHASE 2: EQUILIBRATE LIQUID
# ============================================================

print(f"\n{'='*70}")
print(f"PHASE 2: Equilibrating liquid at {T_MELT} K")
print(f"{'='*70}")

liquid_result = run_npt_md(
    atoms_sc, T_MELT, N_EQUIL_LIQUID,
    label="Liquid equilibration"
)

write(os.path.join(WORK_DIR, "equilibrated_liquid.xyz"), atoms_sc)
print(f"  Liquid equilibrated. Saved snapshot.")

# ============================================================
# 6. PHASE 3: STEPWISE QUENCH (MPMorph SlowQuench pattern)
# ============================================================
# This follows the atomate2 SlowQuenchMaker pattern:
# - Loop from quench_start_temperature to quench_end_temperature
# - At each step, run MD for quench_n_steps
# - Step size: quench_temperature_step

print(f"\n{'='*70}")
print(f"PHASE 3: Stepwise quench {T_MELT} K -> {T_TARGET} K "
      f"(step = {T_QUENCH_STEP} K)")
print(f"{'='*70}")

quench_temperatures = np.arange(T_MELT, T_TARGET - 1, -T_QUENCH_STEP)
# Ensure T_TARGET is included
if quench_temperatures[-1] != T_TARGET:
    quench_temperatures = np.append(quench_temperatures, T_TARGET)

quench_results = []
traj_quench = Trajectory(os.path.join(WORK_DIR, "quench.traj"), "w", atoms_sc)

for i, T_q in enumerate(quench_temperatures):
    # Rescale velocities to new target temperature
    current_T = atoms_sc.get_temperature()
    if current_T > 1.0:
        scale = np.sqrt(T_q / current_T)
        atoms_sc.set_momenta(atoms_sc.get_momenta() * scale)
    else:
        MaxwellBoltzmannDistribution(atoms_sc, temperature_K=T_q)
        Stationary(atoms_sc)

    result = run_npt_md(
        atoms_sc, T_q, N_QUENCH_PER_STEP,
        label=f"Quench step {i+1}/{len(quench_temperatures)}"
    )
    quench_results.append(result)
    traj_quench.write(atoms_sc)

traj_quench.close()

# Compute effective quench rate
total_quench_time_ps = len(quench_temperatures) * N_QUENCH_PER_STEP * TIMESTEP / 1000.0
delta_T = T_MELT - T_TARGET
quench_rate = delta_T / (total_quench_time_ps * 1e-12)  # K/s
print(f"\n  Effective quench rate: {quench_rate:.2e} K/s")
print(f"  Total quench time: {total_quench_time_ps:.1f} ps")

# ============================================================
# 7. PHASE 4: ANNEAL AT TARGET TEMPERATURE
# ============================================================

print(f"\n{'='*70}")
print(f"PHASE 4: Annealing at {T_TARGET} K")
print(f"{'='*70}")

anneal_result = run_npt_md(
    atoms_sc, T_TARGET, N_ANNEAL,
    label="Annealing"
)

write(os.path.join(WORK_DIR, "amorphous_annealed.xyz"), atoms_sc)
write(os.path.join(WORK_DIR, "amorphous_annealed.cif"), atoms_sc)
print(f"  Annealed amorphous structure saved.")

# ============================================================
# 8. (OPTIONAL) PHASE 5: FAST QUENCH TO 0 K
# ============================================================
# Following atomate2 FastQuenchMaker pattern: relax to 0 K

print(f"\n{'='*70}")
print(f"PHASE 5: Fast quench -- relax to 0 K")
print(f"{'='*70}")

atoms_0K = atoms_sc.copy()
atoms_0K.calc = calc
atoms_0K.set_momenta(np.zeros_like(atoms_0K.get_momenta()))

# Relax positions only (preserve amorphous cell volume)
opt_0K = LBFGS(atoms_0K, logfile=os.path.join(WORK_DIR, "relax_0K.log"))
opt_0K.run(fmax=0.05, steps=500)

e_amorphous = atoms_0K.get_potential_energy() / n_atoms
v_amorphous = atoms_0K.get_volume() / n_atoms

print(f"  0 K amorphous: E = {e_amorphous:.4f} eV/atom, V = {v_amorphous:.3f} A^3/atom")
print(f"  Crystal reference: E = {e_crystal:.4f} eV/atom, V = {v_crystal:.3f} A^3/atom")
print(f"  dE (amorphous - crystal) = {e_amorphous - e_crystal:.4f} eV/atom")
print(f"  dV/V = {(v_amorphous - v_crystal) / v_crystal * 100:.1f}%")

write(os.path.join(WORK_DIR, "amorphous_0K.xyz"), atoms_0K)
write(os.path.join(WORK_DIR, "amorphous_0K.cif"), atoms_0K)
print(f"  Final amorphous structure saved.")

# ============================================================
# 9. GLASS TRANSITION TEMPERATURE (Tg) ESTIMATION
# ============================================================

print(f"\n{'='*70}")
print(f"Estimating Glass Transition Temperature (Tg)")
print(f"{'='*70}")

# Tg is found from the change in slope of V(T) during quench
# Liquid: high dV/dT (high thermal expansion)
# Glass: lower dV/dT (similar to crystal)
# Tg is where the slope changes

quench_T = np.array([r["T_target"] for r in quench_results])
quench_V = np.array([r["V_mean"] for r in quench_results])

# Fit two lines: high-T (liquid) and low-T (glass) regions
# Use piecewise linear fit
from scipy.optimize import minimize_scalar

def piecewise_linear_residual(T_break):
    """Fit two lines separated at T_break and return total residual."""
    mask_high = quench_T >= T_break
    mask_low = quench_T < T_break

    if np.sum(mask_high) < 3 or np.sum(mask_low) < 3:
        return 1e10

    # Fit high-T line
    p_high = np.polyfit(quench_T[mask_high], quench_V[mask_high], 1)
    res_high = np.sum((np.polyval(p_high, quench_T[mask_high]) - quench_V[mask_high])**2)

    # Fit low-T line
    p_low = np.polyfit(quench_T[mask_low], quench_V[mask_low], 1)
    res_low = np.sum((np.polyval(p_low, quench_T[mask_low]) - quench_V[mask_low])**2)

    return res_high + res_low

# Search for optimal break point
T_candidates = quench_T[(quench_T > T_TARGET + 200) & (quench_T < T_MELT - 200)]
if len(T_candidates) > 2:
    residuals = [piecewise_linear_residual(Tc) for Tc in T_candidates]
    T_g = T_candidates[np.argmin(residuals)]

    # Fit the two segments for plotting
    mask_high = quench_T >= T_g
    mask_low = quench_T < T_g
    p_high = np.polyfit(quench_T[mask_high], quench_V[mask_high], 1)
    p_low = np.polyfit(quench_T[mask_low], quench_V[mask_low], 1)

    print(f"  Estimated Tg ~ {T_g:.0f} K")
    print(f"  Liquid slope (dV/dT): {p_high[0]:.6f} A^3/atom/K")
    print(f"  Glass slope (dV/dT):  {p_low[0]:.6f} A^3/atom/K")
    print(f"  Note: Simulated Tg depends on quench rate. Faster quench -> higher Tg.")
else:
    T_g = None
    print("  Insufficient data points to estimate Tg. Need more quench steps.")

# ============================================================
# 10. VISUALIZATION
# ============================================================

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# V(T) during quench with Tg
ax = axes[0, 0]
ax.plot(quench_T, quench_V, "o-", markersize=4, color="tab:blue", label="Quench data")
if T_g is not None:
    T_plot_high = np.linspace(T_g, T_MELT, 50)
    T_plot_low = np.linspace(T_TARGET, T_g, 50)
    ax.plot(T_plot_high, np.polyval(p_high, T_plot_high), "r--", label="Liquid fit")
    ax.plot(T_plot_low, np.polyval(p_low, T_plot_low), "g--", label="Glass fit")
    ax.axvline(T_g, color="orange", linestyle=":", linewidth=2,
               label=f"Tg ~ {T_g:.0f} K")
ax.set_xlabel("Temperature (K)")
ax.set_ylabel("Volume (A^3/atom)")
ax.set_title("Volume vs Temperature (Tg estimation)")
ax.legend()

# E(T) during quench
ax = axes[0, 1]
quench_E = np.array([r["E_mean"] for r in quench_results])
ax.plot(quench_T, quench_E, "s-", markersize=4, color="tab:red")
if T_g is not None:
    ax.axvline(T_g, color="orange", linestyle=":", linewidth=2, label=f"Tg ~ {T_g:.0f} K")
ax.set_xlabel("Temperature (K)")
ax.set_ylabel("Energy (eV/atom)")
ax.set_title("Energy vs Temperature During Quench")
ax.legend()

# Temperature trace during entire simulation
ax = axes[1, 0]
# Combine all temperature data
all_T = list(heat_data["T"])
all_T += list(liquid_result["T_series"])
for r in quench_results:
    all_T += list(r["T_series"])
all_T += list(anneal_result["T_series"])
time_ps = np.arange(len(all_T)) * LOG_INTERVAL * TIMESTEP / 1000.0
ax.plot(time_ps, all_T, "-", linewidth=0.3, color="tab:purple", alpha=0.5)
# Overlay target temperature profile
quench_T_profile = []
for r in quench_results:
    quench_T_profile.extend([r["T_target"]] * len(r["T_series"]))
heat_T_profile = [T_MELT] * len(heat_data["T"])
liquid_T_profile = [T_MELT] * len(liquid_result["T_series"])
anneal_T_profile = [T_TARGET] * len(anneal_result["T_series"])
target_profile = heat_T_profile + liquid_T_profile + quench_T_profile + anneal_T_profile
if len(target_profile) == len(all_T):
    ax.plot(time_ps, target_profile, "k--", linewidth=1, label="Target T")
ax.set_xlabel("Time (ps)")
ax.set_ylabel("Temperature (K)")
ax.set_title("Temperature Profile (Full Simulation)")
ax.legend()

# Energy trace during entire simulation
ax = axes[1, 1]
all_E = list(heat_data["E"])
all_E += list(liquid_result["E_series"])
for r in quench_results:
    all_E += list(r["E_series"])
all_E += list(anneal_result["E_series"])
time_ps_e = np.arange(len(all_E)) * LOG_INTERVAL * TIMESTEP / 1000.0
ax.plot(time_ps_e, all_E, "-", linewidth=0.3, color="tab:blue", alpha=0.5)
ax.set_xlabel("Time (ps)")
ax.set_ylabel("Energy (eV/atom)")
ax.set_title("Energy Trace (Full Simulation)")

fig.suptitle(f"Amorphous Generation: {formula}", fontsize=14)
fig.tight_layout()
fig.savefig(os.path.join(WORK_DIR, "melt_quench_overview.png"), dpi=150, bbox_inches="tight")
plt.close()
print(f"\nSaved: {WORK_DIR}/melt_quench_overview.png")

print(f"\n{'='*70}")
print(f"SUMMARY")
print(f"{'='*70}")
print(f"  Crystal:     E = {e_crystal:.4f} eV/atom, V = {v_crystal:.3f} A^3/atom")
print(f"  Amorphous:   E = {e_amorphous:.4f} eV/atom, V = {v_amorphous:.3f} A^3/atom")
print(f"  dE = {e_amorphous - e_crystal:.4f} eV/atom")
print(f"  Quench rate: {quench_rate:.2e} K/s")
if T_g:
    print(f"  Tg ~ {T_g:.0f} K")
print(f"  Output files in: {WORK_DIR}/")
```

### Method A Extension: Comprehensive Amorphous Structure Analysis

Run this after generating the amorphous structure to compute all structural descriptors.

```python
#!/usr/bin/env python3
"""
Comprehensive analysis of amorphous structure:
  - Radial distribution function (RDF) / pair correlation function
  - Partial RDFs for multi-component systems
  - Coordination number distribution
  - Bond angle distribution
  - Structure factor S(q)
  - Mean squared displacement check (confirm glassy, not crystalline)

Input: amorphous structure file (from melt-quench workflow)
"""

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from collections import defaultdict

from ase.io import read
from ase.neighborlist import neighbor_list
from ase.geometry import get_distances

import warnings
warnings.filterwarnings("ignore")

# ============================================================
# CONFIGURATION
# ============================================================

STRUCTURE_FILE = "/tmp/amorphous_gen/amorphous_0K.xyz"  # Or .cif
R_MAX = 10.0               # Max distance for RDF (Angstrom)
N_BINS_RDF = 500            # Number of bins for RDF
Q_MAX = 15.0                # Max q for structure factor (1/Angstrom)
N_Q = 300                   # Number of q points
CUTOFF_CN = 3.5             # Cutoff for coordination number (Angstrom)
                            # Adjust based on first minimum in RDF

WORK_DIR = "/tmp/amorphous_analysis"
os.makedirs(WORK_DIR, exist_ok=True)

# ============================================================
# 1. LOAD STRUCTURE
# ============================================================

atoms = read(STRUCTURE_FILE)
n_atoms = len(atoms)
volume = atoms.get_volume()
rho = n_atoms / volume  # number density
symbols = atoms.get_chemical_symbols()
unique_species = sorted(set(symbols))
print(f"Structure: {atoms.get_chemical_formula()}, {n_atoms} atoms")
print(f"Volume: {volume:.2f} A^3 ({volume/n_atoms:.3f} A^3/atom)")
print(f"Number density: {rho:.6f} A^-3")
print(f"Species: {unique_species}")

# ============================================================
# 2. TOTAL RDF
# ============================================================

def compute_total_rdf(atoms, r_max=R_MAX, n_bins=N_BINS_RDF):
    """Compute total radial distribution function g(r)."""
    n = len(atoms)
    vol = atoms.get_volume()
    rho_local = n / vol

    i_list, j_list, d_list = neighbor_list("ijd", atoms, cutoff=r_max)

    r_edges = np.linspace(0, r_max, n_bins + 1)
    r_centers = 0.5 * (r_edges[:-1] + r_edges[1:])
    dr = r_edges[1] - r_edges[0]

    hist, _ = np.histogram(d_list, bins=r_edges)
    shell_volumes = 4.0 * np.pi * r_centers**2 * dr
    g_r = hist / (n * shell_volumes * rho_local)

    return r_centers, g_r

r, g_r = compute_total_rdf(atoms)

# Find first peak and first minimum (for coordination number cutoff)
# Search for first peak after r > 1.0 A
mask = r > 1.0
r_search = r[mask]
g_search = g_r[mask]

idx_first_peak = np.argmax(g_search)
r_first_peak = r_search[idx_first_peak]
g_first_peak = g_search[idx_first_peak]

# Find first minimum after first peak
g_after_peak = g_search[idx_first_peak:]
r_after_peak = r_search[idx_first_peak:]
idx_first_min = np.argmin(g_after_peak)
r_first_min = r_after_peak[idx_first_min]

print(f"\nTotal RDF:")
print(f"  First peak: r = {r_first_peak:.3f} A, g(r) = {g_first_peak:.3f}")
print(f"  First minimum: r = {r_first_min:.3f} A")
print(f"  Suggested CN cutoff: {r_first_min:.2f} A")

# ============================================================
# 3. PARTIAL RDFs (for multi-component systems)
# ============================================================

def compute_partial_rdfs(atoms, r_max=R_MAX, n_bins=N_BINS_RDF):
    """Compute partial RDFs g_ab(r) for each pair of species."""
    n = len(atoms)
    vol = atoms.get_volume()
    species = atoms.get_chemical_symbols()
    unique = sorted(set(species))

    i_list, j_list, d_list = neighbor_list("ijd", atoms, cutoff=r_max)

    r_edges = np.linspace(0, r_max, n_bins + 1)
    r_centers = 0.5 * (r_edges[:-1] + r_edges[1:])
    dr = r_edges[1] - r_edges[0]

    partial_rdfs = {}

    for sp_a in unique:
        for sp_b in unique:
            # Select pairs where i is sp_a and j is sp_b
            mask = np.array([species[i] == sp_a and species[j] == sp_b
                            for i, j in zip(i_list, j_list)])
            dists_ab = d_list[mask]

            n_a = sum(1 for s in species if s == sp_a)
            n_b = sum(1 for s in species if s == sp_b)
            rho_b = n_b / vol

            hist, _ = np.histogram(dists_ab, bins=r_edges)
            shell_volumes = 4.0 * np.pi * r_centers**2 * dr

            if n_a > 0 and rho_b > 0:
                g_ab = hist / (n_a * shell_volumes * rho_b)
            else:
                g_ab = np.zeros_like(r_centers)

            pair_key = f"{sp_a}-{sp_b}"
            partial_rdfs[pair_key] = (r_centers, g_ab)

    return partial_rdfs

if len(unique_species) > 1:
    partial_rdfs = compute_partial_rdfs(atoms)
    print(f"\nPartial RDFs computed for {len(partial_rdfs)} pairs")
    for pair, (r_p, g_p) in partial_rdfs.items():
        mask_p = r_p > 1.0
        if np.any(g_p[mask_p] > 0.1):
            idx_pk = np.argmax(g_p[mask_p])
            print(f"  {pair}: first peak at r = {r_p[mask_p][idx_pk]:.3f} A")
else:
    partial_rdfs = {}

# ============================================================
# 4. COORDINATION NUMBER DISTRIBUTION
# ============================================================

def compute_coordination(atoms, cutoff=None):
    """Compute coordination number distribution."""
    if cutoff is None:
        cutoff = CUTOFF_CN

    species = atoms.get_chemical_symbols()
    unique = sorted(set(species))

    i_list, j_list, d_list = neighbor_list("ijd", atoms, cutoff=cutoff)

    # Total coordination
    cn_total = np.zeros(len(atoms), dtype=int)
    for i_idx in i_list:
        cn_total[i_idx] += 1

    # Per-species coordination
    cn_by_species = {}
    for sp in unique:
        mask_sp = np.array([species[i] == sp for i in range(len(atoms))])
        cn_sp = cn_total[mask_sp]
        cn_by_species[sp] = cn_sp

    return cn_total, cn_by_species

# Use first minimum as cutoff
cn_cutoff = r_first_min
cn_total, cn_by_species = compute_coordination(atoms, cutoff=cn_cutoff)

print(f"\nCoordination number (cutoff = {cn_cutoff:.2f} A):")
print(f"  Overall: mean = {np.mean(cn_total):.2f}, "
      f"std = {np.std(cn_total):.2f}, "
      f"range = [{np.min(cn_total)}, {np.max(cn_total)}]")

for sp, cn_sp in cn_by_species.items():
    print(f"  {sp}: mean = {np.mean(cn_sp):.2f}, std = {np.std(cn_sp):.2f}")

# ============================================================
# 5. BOND ANGLE DISTRIBUTION
# ============================================================

def compute_bond_angles(atoms, cutoff=None):
    """
    Compute distribution of bond angles theta_ijk where j is the central atom
    and i,k are neighbors within cutoff.
    """
    if cutoff is None:
        cutoff = CUTOFF_CN

    i_list, j_list, d_list, D_list = neighbor_list("ijdD", atoms, cutoff=cutoff)

    # Group neighbors by central atom j
    neighbors = defaultdict(list)
    for idx in range(len(i_list)):
        j = j_list[idx]        # central atom
        # Actually neighbor_list gives i->j pairs, so i is the central atom
        i = i_list[idx]
        neighbors[i].append(D_list[idx])  # displacement vector from i to j

    angles = []
    for atom_idx, vecs in neighbors.items():
        n_neigh = len(vecs)
        if n_neigh < 2:
            continue
        # Compute all pairs of angles
        for a in range(n_neigh):
            for b in range(a + 1, n_neigh):
                v1 = vecs[a]
                v2 = vecs[b]
                cos_theta = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
                cos_theta = np.clip(cos_theta, -1.0, 1.0)
                theta = np.degrees(np.arccos(cos_theta))
                angles.append(theta)

    return np.array(angles)

angles = compute_bond_angles(atoms, cutoff=cn_cutoff)
print(f"\nBond angle distribution:")
print(f"  Number of angles: {len(angles)}")
if len(angles) > 0:
    print(f"  Mean angle: {np.mean(angles):.1f} degrees")
    print(f"  Most probable angle: {np.histogram(angles, bins=180, range=(0, 180))[1][np.argmax(np.histogram(angles, bins=180, range=(0, 180))[0])]:.0f} degrees")

# ============================================================
# 6. STRUCTURE FACTOR S(q)
# ============================================================

def compute_structure_factor(r, g_r, rho_local, q_max=Q_MAX, n_q=N_Q):
    """Compute S(q) from g(r) via Fourier transform.

    S(q) = 1 + 4*pi*rho * integral_0^inf r*(g(r)-1)*sin(qr)/q dr
    """
    q = np.linspace(0.1, q_max, n_q)
    S_q = np.ones_like(q)
    dr = r[1] - r[0]

    for iq, qi in enumerate(q):
        integrand = r * (g_r - 1.0) * np.sin(qi * r) / qi
        S_q[iq] = 1.0 + 4.0 * np.pi * rho_local * np.trapz(integrand, dx=dr)

    return q, S_q

q, S_q = compute_structure_factor(r, g_r, rho)

# Find first sharp diffraction peak (FSDP)
mask_q = q > 0.5
q_search = q[mask_q]
S_search = S_q[mask_q]
idx_fsdp = np.argmax(S_search)
q_fsdp = q_search[idx_fsdp]
S_fsdp = S_search[idx_fsdp]

print(f"\nStructure factor S(q):")
print(f"  First sharp diffraction peak (FSDP): q = {q_fsdp:.3f} A^-1, S(q) = {S_fsdp:.3f}")
print(f"  Corresponding real-space distance: d = 2*pi/q = {2*np.pi/q_fsdp:.3f} A")

# ============================================================
# 7. SAVE NUMERICAL DATA
# ============================================================

np.savetxt(os.path.join(WORK_DIR, "rdf_total.dat"),
           np.column_stack([r, g_r]),
           header="r(A)  g(r)", fmt="%.6f")

np.savetxt(os.path.join(WORK_DIR, "structure_factor.dat"),
           np.column_stack([q, S_q]),
           header="q(1/A)  S(q)", fmt="%.6f")

if len(angles) > 0:
    angle_hist, angle_edges = np.histogram(angles, bins=180, range=(0, 180), density=True)
    angle_centers = 0.5 * (angle_edges[:-1] + angle_edges[1:])
    np.savetxt(os.path.join(WORK_DIR, "bond_angles.dat"),
               np.column_stack([angle_centers, angle_hist]),
               header="angle(deg)  P(angle)", fmt="%.6f")

cn_hist_vals, cn_edges = np.histogram(cn_total,
                                       bins=np.arange(0, np.max(cn_total) + 2) - 0.5,
                                       density=True)
cn_centers = 0.5 * (cn_edges[:-1] + cn_edges[1:])
np.savetxt(os.path.join(WORK_DIR, "coordination_numbers.dat"),
           np.column_stack([cn_centers, cn_hist_vals]),
           header="CN  fraction", fmt="%.1f %.6f")

print(f"\nNumerical data saved to {WORK_DIR}/")

# ============================================================
# 8. COMPREHENSIVE VISUALIZATION
# ============================================================

fig, axes = plt.subplots(2, 3, figsize=(18, 10))

# (a) Total RDF
ax = axes[0, 0]
ax.plot(r, g_r, "b-", linewidth=1)
ax.axhline(1.0, color="gray", linestyle="--", alpha=0.5)
ax.axvline(r_first_peak, color="red", linestyle=":", alpha=0.5,
           label=f"1st peak: {r_first_peak:.2f} A")
ax.axvline(r_first_min, color="green", linestyle=":", alpha=0.5,
           label=f"1st min: {r_first_min:.2f} A")
ax.set_xlabel("r (Angstrom)")
ax.set_ylabel("g(r)")
ax.set_title("Total RDF")
ax.set_xlim(0, R_MAX)
ax.legend(fontsize=8)

# (b) Partial RDFs (if multi-component)
ax = axes[0, 1]
if partial_rdfs:
    # Only plot unique pairs (A-B same as B-A for display)
    plotted = set()
    for pair, (r_p, g_p) in partial_rdfs.items():
        sp_a, sp_b = pair.split("-")
        pair_sorted = tuple(sorted([sp_a, sp_b]))
        if pair_sorted in plotted:
            continue
        plotted.add(pair_sorted)
        ax.plot(r_p, g_p, linewidth=1, label=f"g_{{{sp_a}-{sp_b}}}(r)")
    ax.axhline(1.0, color="gray", linestyle="--", alpha=0.5)
    ax.set_xlabel("r (Angstrom)")
    ax.set_ylabel("g_ab(r)")
    ax.set_title("Partial RDFs")
    ax.set_xlim(0, R_MAX)
    ax.legend(fontsize=8)
else:
    ax.plot(r, g_r, "b-", linewidth=1)
    ax.axhline(1.0, color="gray", linestyle="--", alpha=0.5)
    ax.set_xlabel("r (Angstrom)")
    ax.set_ylabel("g(r)")
    ax.set_title("RDF (single species)")
    ax.set_xlim(0, R_MAX)

# (c) Structure factor
ax = axes[0, 2]
ax.plot(q, S_q, "r-", linewidth=1)
ax.axhline(1.0, color="gray", linestyle="--", alpha=0.5)
ax.axvline(q_fsdp, color="blue", linestyle=":", alpha=0.5,
           label=f"FSDP: q={q_fsdp:.2f} A^-1")
ax.set_xlabel("q (1/Angstrom)")
ax.set_ylabel("S(q)")
ax.set_title("Structure Factor")
ax.legend(fontsize=8)

# (d) Coordination number distribution
ax = axes[1, 0]
cn_unique, cn_counts = np.unique(cn_total, return_counts=True)
ax.bar(cn_unique, cn_counts / n_atoms, width=0.8, color="tab:orange", alpha=0.8)
ax.set_xlabel("Coordination Number")
ax.set_ylabel("Fraction")
ax.set_title(f"CN Distribution (cutoff={cn_cutoff:.2f} A)")
ax.set_xticks(cn_unique)

# (e) Bond angle distribution
ax = axes[1, 1]
if len(angles) > 0:
    ax.hist(angles, bins=180, range=(0, 180), density=True,
            color="tab:green", alpha=0.7, edgecolor="none")
    ax.set_xlabel("Bond Angle (degrees)")
    ax.set_ylabel("Probability Density")
    ax.set_title(f"Bond Angle Distribution (cutoff={cn_cutoff:.2f} A)")
    # Mark reference angles
    for ref_angle, ref_label in [(60, "60 (tri)"), (90, "90 (oct)"),
                                  (109.5, "109.5 (tet)"), (120, "120 (tri-planar)")]:
        ax.axvline(ref_angle, color="gray", linestyle=":", alpha=0.4)
        ax.text(ref_angle, ax.get_ylim()[1]*0.95, ref_label,
                rotation=90, fontsize=6, va="top", ha="right")
else:
    ax.text(0.5, 0.5, "No angles computed", transform=ax.transAxes,
            ha="center", va="center")

# (f) Cumulative coordination number from RDF
ax = axes[1, 2]
# N(r) = 4*pi*rho * integral_0^r g(r')*r'^2 dr'
dr = r[1] - r[0]
N_r = 4.0 * np.pi * rho * np.cumsum(g_r * r**2 * dr)
ax.plot(r, N_r, "m-", linewidth=1)
ax.axvline(cn_cutoff, color="green", linestyle="--",
           label=f"CN cutoff: {cn_cutoff:.2f} A")
# Mark CN at cutoff
cn_at_cutoff = np.interp(cn_cutoff, r, N_r)
ax.axhline(cn_at_cutoff, color="gray", linestyle=":", alpha=0.5)
ax.text(0.5, cn_at_cutoff + 0.3, f"CN = {cn_at_cutoff:.1f}", fontsize=9)
ax.set_xlabel("r (Angstrom)")
ax.set_ylabel("N(r)")
ax.set_title("Running Coordination Number")
ax.set_xlim(0, R_MAX)
ax.legend(fontsize=8)

fig.suptitle(f"Amorphous Structure Analysis: {atoms.get_chemical_formula()}", fontsize=14)
fig.tight_layout()
fig.savefig(os.path.join(WORK_DIR, "amorphous_analysis.png"), dpi=150, bbox_inches="tight")
plt.close()
print(f"\nSaved: {WORK_DIR}/amorphous_analysis.png")

# ============================================================
# 9. VALIDATION: COMPARE WITH REFERENCE (if available)
# ============================================================

print(f"\n{'='*60}")
print("VALIDATION CHECKLIST")
print(f"{'='*60}")
print(f"1. RDF first peak position: {r_first_peak:.3f} A")
print(f"   (Compare with sum of covalent/metallic radii of constituent atoms)")
print(f"2. Mean coordination number: {np.mean(cn_total):.2f}")
print(f"   (Compare with expected CN: tetrahedral=4, octahedral=6, metallic~12)")
print(f"3. FSDP position: q = {q_fsdp:.3f} A^-1")
print(f"   (Compare with experimental X-ray/neutron scattering if available)")
print(f"4. Check g(r) -> 1 at large r (no residual long-range order)")
print(f"   g(r) at r={R_MAX:.0f} A: {g_r[-1]:.4f} (should be close to 1.0)")
print(f"5. Energy above crystal: compare with literature values for this glass")
```

### Method B: LAMMPS Melt-Quench (Metals with EAM)

#### Step 1: Prepare structure

```python
#!/usr/bin/env python3
"""Convert CIF to LAMMPS data file and build supercell."""
from ase.io import read, write

atoms = read("structure.cif")
atoms = atoms.repeat((4, 4, 4))  # Aim for >200 atoms
write("structure.lammps", atoms, format="lammps-data")
print(f"Wrote structure.lammps with {len(atoms)} atoms")
```

#### Step 2: LAMMPS input file for melt-quench

Save as `in.amorphous`:

```
# LAMMPS melt-quench protocol for amorphous structure generation
# Works with EAM/MEAM potentials for metals and alloys

units           metal
atom_style      atomic
boundary        p p p

read_data       structure.lammps

# Potential -- adjust for your system
pair_style      eam/alloy
pair_coeff      * * Cu_mishin1.eam.alloy Cu

neighbor        2.0 bin
neigh_modify    every 1 delay 0 check yes

# Initial minimization
minimize        1.0e-6 1.0e-8 1000 10000
reset_timestep  0

timestep        0.002  # 2 fs

# Thermodynamic output
thermo          200
thermo_style    custom step temp pe ke etotal vol press density

# ============================================================
# Phase 1: Heat to liquid temperature
# ============================================================
velocity        all create 300.0 54321 dist gaussian

# NPT equilibrate at 300 K first
fix             eq1 all npt temp 300.0 300.0 0.1 iso 0.0 0.0 1.0
run             5000
unfix           eq1

# Heat to 3000 K (adjust above your material's Tm)
fix             heat all npt temp 300.0 3000.0 0.1 iso 0.0 0.0 1.0
run             25000
unfix           heat

# ============================================================
# Phase 2: Equilibrate liquid at 3000 K
# ============================================================
fix             liq all npt temp 3000.0 3000.0 0.1 iso 0.0 0.0 1.0
run             50000
unfix           liq

write_data      liquid_equilibrated.data

# ============================================================
# Phase 3: Stepwise quench 3000 K -> 300 K
# ============================================================
# Quench in 200 K steps, 2000 steps (4 ps) per hold
# Total quench: (3000-300)/200 = 13.5 steps * 4 ps = 54 ps
# Quench rate: 2700 K / 54 ps = 5e13 K/s

variable        T_high  equal 3000.0
variable        T_low   equal 300.0
variable        T_step  equal 200.0
variable        n_hold  equal 2000

# Track V(T) for Tg estimation
variable        myvol equal vol/atoms
variable        mytemp equal temp
variable        mype equal pe/atoms
fix             quench_log all print 200 "${mytemp} ${myvol} ${mype}" &
                file quench_VT.dat screen no &
                title "# T(K) V(A3/atom) PE(eV/atom)"

# Quench loop: 3000 -> 2800 -> ... -> 300
label           quench_loop
variable        T_current equal ${T_high}

label           loop_start
if              "${T_current} < ${T_low}" then "jump SELF loop_end"

fix             qstep all npt temp ${T_current} ${T_current} 0.1 iso 0.0 0.0 1.0
run             ${n_hold}
unfix           qstep

variable        T_current equal ${T_current}-${T_step}
jump            SELF loop_start
label           loop_end

# ============================================================
# Phase 4: Anneal at target temperature
# ============================================================
fix             anneal all npt temp ${T_low} ${T_low} 0.1 iso 0.0 0.0 1.0
run             50000
unfix           anneal

# ============================================================
# Phase 5: Final minimization (fast quench to 0 K)
# ============================================================
minimize        1.0e-8 1.0e-10 5000 50000

# Save final amorphous structure
write_data      amorphous_final.data
write_dump      all custom amorphous_final.dump id type x y z
dump            snap all custom 1 amorphous_snapshot.dump id type x y z
run             0
undump          snap

print "Melt-quench complete. Amorphous structure saved."
```

#### Step 3: Run LAMMPS

```bash
lmp -in in.amorphous -log log.amorphous
```

#### Step 4: Analyze LAMMPS quench results

```python
#!/usr/bin/env python3
"""Analyze LAMMPS melt-quench: V(T) for Tg, convert to ASE for RDF analysis."""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.optimize import minimize_scalar

# Load V(T) data from quench
data = np.loadtxt("quench_VT.dat", comments="#")
temp = data[:, 0]
vol = data[:, 1]
pe = data[:, 2]

# Bin by temperature for cleaner V(T) curve
T_bins = np.arange(200, 3100, 100)
V_binned = []
T_binned = []
for T_lo, T_hi in zip(T_bins[:-1], T_bins[1:]):
    mask = (temp >= T_lo) & (temp < T_hi)
    if np.sum(mask) > 5:
        T_binned.append((T_lo + T_hi) / 2)
        V_binned.append(np.mean(vol[mask]))

T_binned = np.array(T_binned)
V_binned = np.array(V_binned)

# Estimate Tg from V(T) slope change
def piecewise_residual(T_break):
    mask_h = T_binned >= T_break
    mask_l = T_binned < T_break
    if np.sum(mask_h) < 3 or np.sum(mask_l) < 3:
        return 1e10
    p_h = np.polyfit(T_binned[mask_h], V_binned[mask_h], 1)
    p_l = np.polyfit(T_binned[mask_l], V_binned[mask_l], 1)
    res = (np.sum((np.polyval(p_h, T_binned[mask_h]) - V_binned[mask_h])**2) +
           np.sum((np.polyval(p_l, T_binned[mask_l]) - V_binned[mask_l])**2))
    return res

T_search = T_binned[(T_binned > 400) & (T_binned < 2500)]
if len(T_search) > 2:
    residuals = [piecewise_residual(Tc) for Tc in T_search]
    Tg = T_search[np.argmin(residuals)]
    print(f"Estimated Tg ~ {Tg:.0f} K")

    # Fit segments for plotting
    mask_h = T_binned >= Tg
    mask_l = T_binned < Tg
    p_h = np.polyfit(T_binned[mask_h], V_binned[mask_h], 1)
    p_l = np.polyfit(T_binned[mask_l], V_binned[mask_l], 1)

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(T_binned, V_binned, "ko-", markersize=4, label="Quench data")
    T_plot_h = np.linspace(Tg, max(T_binned), 50)
    T_plot_l = np.linspace(min(T_binned), Tg, 50)
    ax.plot(T_plot_h, np.polyval(p_h, T_plot_h), "r--", label="Liquid fit")
    ax.plot(T_plot_l, np.polyval(p_l, T_plot_l), "b--", label="Glass fit")
    ax.axvline(Tg, color="orange", linestyle=":", linewidth=2, label=f"Tg ~ {Tg:.0f} K")
    ax.set_xlabel("Temperature (K)")
    ax.set_ylabel("Volume (A^3/atom)")
    ax.set_title("Glass Transition from Melt-Quench")
    ax.legend()
    fig.tight_layout()
    fig.savefig("Tg_analysis.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("Saved: Tg_analysis.png")

# Convert LAMMPS dump to ASE for structural analysis
from ase.io import read as ase_read
atoms = ase_read("amorphous_final.dump", format="lammps-dump-text")
print(f"Loaded amorphous structure: {len(atoms)} atoms")
print("Use the analysis script from Method A for RDF, CN, bond angles, S(q).")
```

## Key Parameters

| Parameter | Typical Value | Notes |
|---|---|---|
| Supercell size | >100 atoms (min), >200 recommended | Smaller systems have surface-to-volume artifacts in PBC; RDF statistics are poor |
| Melt temperature | 1.5-2x experimental Tm | Must fully melt; check RDF shows liquid character |
| Liquid equilibration | 10-50 ps | Ensure diffusion has occurred (MSD check) |
| Quench rate | 10^12-10^13 K/s (simulation) | Faster quench -> higher Tg, more disordered. Experimental glasses are 10^0-10^6 K/s |
| Temperature step (quench) | 100-500 K | Finer steps give smoother V(T) for Tg but cost more |
| Hold time per step | 2-10 ps | Must allow partial equilibration at each T |
| Anneal time | 10-50 ps | Longer annealing at T_target relaxes the glass |
| Timestep | 1-2 fs | Use 0.5-1 fs for light elements (H, Li) |
| RDF cutoff | Half the shortest cell dimension | Ensure no self-interaction artifacts |
| CN cutoff | First minimum of g(r) | Typically 2.5-4.0 A depending on chemistry |
| MACE model | "medium" or "large" | "large" for publication quality |

### Quench Rate Considerations

| Quench Rate (K/s) | Context | Notes |
|---|---|---|
| 10^0-10^3 | Experimental bulk glass | Not accessible in MD |
| 10^6-10^9 | Experimental splat quench / melt spinning | Not accessible in standard MD |
| 10^10-10^11 | Slow MD quench | Very expensive; most realistic for simulation |
| 10^12-10^13 | Typical MD quench | Good balance of cost and quality |
| 10^14-10^15 | Very fast MD quench | May not properly form glass; more liquid-like |

## Interpreting Results

### RDF of Amorphous vs. Crystalline Structure
- **Crystal**: sharp, well-defined peaks at lattice neighbor distances; peaks persist to large r
- **Amorphous**: broadened first peak (short-range order preserved), split or merged second peak, g(r) -> 1 for r > 5-8 A (no long-range order)
- **Still crystalline after quench**: sharp peaks persist -- quench was too slow or Tm was not reached. Increase T_MELT or quench rate.

### Coordination Number
- Compare with crystal coordination: metallic glasses typically have CN ~12 (similar to FCC/HCP)
- Covalent glasses (SiO2): Si has CN ~4 (tetrahedral), O has CN ~2
- Broad CN distribution indicates more disorder
- Narrow CN distribution suggests strong directional bonding preference

### Bond Angle Distribution
- Tetrahedral networks (SiO2, a-Si): peak near 109.5 degrees
- Octahedral coordination: peaks near 90 and 180 degrees
- Metallic glasses: broad distribution reflecting close-packing distortions
- Multiple peaks may indicate mixed coordination environments

### Structure Factor S(q)
- First sharp diffraction peak (FSDP) indicates medium-range order
- Position of FSDP: 2*pi/q_FSDP gives characteristic length scale
- Compare peak positions with experimental X-ray or neutron scattering data
- Amorphous: broad peaks; crystalline: sharp Bragg peaks

### Glass Transition Temperature (Tg)
- Found from slope change in V(T) during quench
- Simulated Tg is typically higher than experimental Tg due to faster quench rate
- Empirical correction: Tg_sim ~ Tg_exp + C * log10(quench_rate_sim / quench_rate_exp)
- Multiple quench rates can be used to extrapolate to experimental Tg

### Energy Above Crystal
- Amorphous energy should be higher than crystalline ground state
- Typical: 0.01-0.2 eV/atom above crystal depending on material class
- Too high: structure may not be well-annealed; increase anneal time
- Too low: may still have crystalline domains; check RDF

## Common Issues

**Structure does not melt**: T_MELT is too low for this material. Increase to 2x or 3x the expected Tm. Check that MACE model gives reasonable energetics by comparing relaxed crystal energy with DFT.

**Structure re-crystallizes during quench**: Quench rate too slow, or system too small (small systems crystallize easily). Increase quench rate (reduce N_QUENCH_PER_STEP or increase T_QUENCH_STEP) or increase supercell size.

**RDF shows sharp peaks after quench (still crystalline)**: The liquid was not properly equilibrated, or the quench was too slow. Verify the liquid by checking that RDF at T_MELT shows liquid-like character (single broad first peak, no long-range peaks). Increase N_EQUIL_LIQUID.

**NPT cell becomes unstable / collapses**: Reduce barostat coupling (increase taup), or switch to NVT for the high-temperature equilibration and only use NPT for the quench. For very high temperatures, NVT is more stable.

**MACE gives NaN or crashes at high T**: The MACE foundation model may encounter atomic configurations far outside its training distribution. Try reducing T_MELT, or use a smaller timestep. If persistent, switch to LAMMPS with a classical potential for the high-T phase and only use MACE for analysis.

**Tg estimation fails or gives unreasonable value**: Need more temperature points during quench (smaller T_QUENCH_STEP). Also, the piecewise linear fit is approximate; for better accuracy, use multiple quench rates and extrapolate.

**Coordination number cutoff unclear**: Always base the CN cutoff on the first minimum of g(r), not a fixed value. The first minimum position varies by chemistry and can shift in the amorphous vs. crystalline state.

**Bond angle calculation is slow for large systems**: The O(N * CN^2) scaling means systems with high coordination and many atoms can be expensive. Subsample atoms if needed (randomly select a fraction for angle analysis).

**Volume vs. T curve is noisy**: Increase production time at each quench step (N_QUENCH_PER_STEP) so that averaged volumes are more converged. Also ensure LOG_INTERVAL captures enough samples.
