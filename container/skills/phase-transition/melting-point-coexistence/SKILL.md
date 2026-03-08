# Melting Point by Solid-Liquid Coexistence (Interface Method)

## When to Use

- Determining the thermodynamic melting temperature (Tm) without superheating bias
- You need an accurate melting point (within ~10-50 K) for a crystalline material
- Validating interatomic potentials or ML force fields against experimental melting data
- Studying the solid-liquid interface at equilibrium
- You have already identified an approximate Tm from a heating curve and need to refine it
- The material has a well-defined crystalline phase (FCC, BCC, HCP, diamond cubic, etc.)

## Method Selection

```
Which coexistence workflow should you use?

Any chemistry, no classical potential available?
  --> Method A: ASE + MACE coexistence
  Advantages: Works for any element/compound MACE supports, no potential fitting
  Limitations: Slower than LAMMPS, limited to ~1000 atoms practically

Metal with validated EAM/MEAM/ADP potential?
  --> Method B: LAMMPS coexistence (subprocess)
  Advantages: Very fast, handles >10,000 atoms, long production runs
  Limitations: Restricted to elements with available classical potentials

Metal/material with MACE potential + need for LAMMPS speed?
  --> Method C: LAMMPS + MACE-MP via pair_style mace
  Advantages: LAMMPS speed + MACE universality
  Limitations: Requires LAMMPS compiled with MACE support (ML-IAP package)

System size guidance:
  < 500 atoms:  Too small -- interface effects dominate, unreliable Tm
  500-2000:     Acceptable for initial bracketing
  2000-10000:   Good production size
  > 10000:      Excellent, minimal finite-size effects
```

## Prerequisites

Pre-installed: `ase`, `mace-torch`, `pymatgen`, `numpy`, `scipy`, `matplotlib`, `spglib`.

LAMMPS binary: `lmp` (with MANYBODY, KSPACE, and optionally ML-IAP packages).

For LAMMPS EAM potentials, download from [NIST Interatomic Potentials Repository](https://www.ctcms.nist.gov/potentials/) or use files bundled with LAMMPS in the potentials directory.

## Background: The Interface Method

The solid-liquid coexistence (interface) method determines the melting point by constructing a simulation cell containing both a solid slab and a liquid slab in direct contact. The system is then run under NPT molecular dynamics at a series of trial temperatures:

- **Below Tm**: The solid phase is thermodynamically favored. The solid-liquid interface migrates into the liquid region, and the system eventually solidifies completely. Potential energy drifts downward.
- **Above Tm**: The liquid phase is favored. The interface migrates into the solid, and the system melts entirely. Potential energy drifts upward.
- **At Tm**: Both phases coexist in dynamic equilibrium. The interface fluctuates but neither phase grows. Potential energy fluctuates around a constant value.

This avoids the superheating artifact of single-phase heating methods (which overshoot Tm by 100-300 K) because the pre-existing interface provides heterogeneous nucleation sites, eliminating the nucleation barrier.

The protocol, inspired by the pyiron interface method (Zhu & Janssen, 2020), involves:
1. Build a crystalline supercell and relax it
2. Equilibrate the solid at an estimated Tm via NPT
3. Melt one half of the slab (freeze the other half, heat to ~1.5*Tm)
4. Combine solid + liquid into a two-phase slab
5. Run NPT MD at trial temperatures and observe PE drift
6. Bracket Tm via bisection between solidifying and melting temperatures

---

## Detailed Steps

### Method A: ASE + MACE -- Solid-Liquid Coexistence

Complete standalone script. No pyiron dependency.

```python
#!/usr/bin/env python3
"""
Melting point determination via solid-liquid coexistence (interface method).

Uses ASE + MACE to:
  1. Relax crystalline unit cell
  2. Build elongated supercell (doubled along z for solid|liquid slab)
  3. Equilibrate solid half at estimated Tm
  4. Melt liquid half at high T while freezing solid half
  5. Combine and run NPT at trial temperatures
  6. Bracket Tm from PE drift direction
  7. Refine by bisection

No pyiron dependency -- standalone script.
"""

import os
import sys
import json
import time
import traceback
import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from ase.io import read, write
from ase.optimize import LBFGS
from ase.constraints import ExpCellFilter, FixAtoms
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution, Stationary
from ase.md.langevin import Langevin
from ase.md.nptberendsen import NPTBerendsen
from ase import units, Atoms

import warnings
warnings.filterwarnings("ignore")

# ============================================================
# 1. CONFIGURATION
# ============================================================

STRUCTURE_FILE = "structure.cif"       # Input crystalline structure
MACE_MODEL = "medium"                  # "small", "medium", "large"
DEVICE = "cpu"                         # "cpu" or "cuda"

# Supercell: base size for each half of the slab
# The final cell is doubled along SLAB_DIRECTION
SUPERCELL_BASE = (3, 3, 3)            # Adjust so each half has >250 atoms
SLAB_DIRECTION = 2                     # z-axis: interface normal

# Approximate melting point (from experiment, heating curve, or literature)
# Used for initial solid equilibration and to set trial temperature range
T_ESTIMATE = 1400                      # K -- adjust for your material

# Melting the liquid half
T_MELT = 2.0 * T_ESTIMATE             # High T to fully melt
N_MELT_STEPS = 5000                    # Steps to equilibrate liquid half

# Trial temperature scan for bracketing
# Start with coarse bracket, then refine
T_BRACKET_COARSE = [
    T_ESTIMATE - 300, T_ESTIMATE - 150,
    T_ESTIMATE, T_ESTIMATE + 150, T_ESTIMATE + 300,
]
# Fine bracket will be generated automatically after coarse pass

# MD parameters
TIMESTEP = 2.0                         # fs (use 1.0 for light elements)
N_EQUIL = 5000                         # Equilibration steps at each trial T
N_PROD = 15000                         # Production steps to observe interface
LOG_INTERVAL = 50                      # Logging interval (steps)

# NPT settings
PRESSURE = 0.0                         # GPa (ambient)

# Convergence
TM_TOLERANCE = 25.0                    # K -- stop bisection when bracket < this

WORK_DIR = "/tmp/coexistence_mace"
os.makedirs(WORK_DIR, exist_ok=True)

# ============================================================
# 2. INITIALIZE CALCULATOR AND RELAX UNIT CELL
# ============================================================

try:
    from mace.calculators import mace_mp
    calc = mace_mp(model=MACE_MODEL, device=DEVICE, default_dtype="float64")
except Exception as e:
    print(f"ERROR: Failed to load MACE calculator: {e}")
    sys.exit(1)

print("=" * 70)
print("MELTING POINT BY SOLID-LIQUID COEXISTENCE (ASE + MACE)")
print("=" * 70)

# Read and relax unit cell
try:
    atoms_unit = read(STRUCTURE_FILE)
except Exception as e:
    print(f"ERROR: Cannot read {STRUCTURE_FILE}: {e}")
    sys.exit(1)

atoms_unit.calc = calc
ecf = ExpCellFilter(atoms_unit)
opt = LBFGS(ecf, logfile=os.path.join(WORK_DIR, "relax.log"))
opt.run(fmax=0.01, steps=500)
a0 = atoms_unit.cell.cellpar()
write(os.path.join(WORK_DIR, "relaxed_unitcell.cif"), atoms_unit)
print(f"Relaxed unit cell: {atoms_unit.get_chemical_formula()}")
print(f"  Lattice: a={a0[0]:.3f} b={a0[1]:.3f} c={a0[2]:.3f} A")
print(f"  Energy: {atoms_unit.get_potential_energy():.4f} eV")

# ============================================================
# 3. BUILD TWO-PHASE SLAB STRUCTURE
# ============================================================

# Step 3a: Build solid half and equilibrate at estimated Tm
atoms_solid = atoms_unit.repeat(SUPERCELL_BASE)
atoms_solid.calc = calc
n_half = len(atoms_solid)
print(f"\nSolid half: {n_half} atoms (supercell {SUPERCELL_BASE})")

if n_half < 100:
    print("WARNING: Very small supercell. Increase SUPERCELL_BASE for reliable results.")

# Equilibrate solid at estimated Tm using short NPT run
print(f"Equilibrating solid at {T_ESTIMATE} K ...")
MaxwellBoltzmannDistribution(atoms_solid, temperature_K=T_ESTIMATE)
Stationary(atoms_solid)
dyn_solid = NPTBerendsen(
    atoms_solid,
    timestep=TIMESTEP * units.fs,
    temperature_K=T_ESTIMATE,
    pressure_au=PRESSURE * units.GPa,
    taut=100 * units.fs,
    taup=1000 * units.fs,
    compressibility_au=4.57e-5 / units.bar,
)
dyn_solid.run(3000)
write(os.path.join(WORK_DIR, "solid_half_equilibrated.xyz"), atoms_solid)
print(f"  Solid equilibrated. T_inst = {atoms_solid.get_temperature():.0f} K")

# Step 3b: Create liquid half by melting a copy
# Freeze bottom half (solid) using FixAtoms, melt top half at high T
atoms_liquid = atoms_solid.copy()
atoms_liquid.calc = calc

print(f"Melting liquid half at {T_MELT:.0f} K for {N_MELT_STEPS} steps ...")
MaxwellBoltzmannDistribution(atoms_liquid, temperature_K=T_MELT)
Stationary(atoms_liquid)
dyn_melt = Langevin(
    atoms_liquid,
    timestep=TIMESTEP * units.fs,
    temperature_K=T_MELT,
    friction=0.02 / units.fs,
)
dyn_melt.run(N_MELT_STEPS)

# Cool the liquid to near Tm so the combined system is not far from equilibrium
print(f"Cooling liquid to {T_ESTIMATE} K ...")
dyn_cool = Langevin(
    atoms_liquid,
    timestep=TIMESTEP * units.fs,
    temperature_K=T_ESTIMATE,
    friction=0.02 / units.fs,
)
dyn_cool.run(2000)
write(os.path.join(WORK_DIR, "liquid_half.xyz"), atoms_liquid)
print(f"  Liquid half prepared. T_inst = {atoms_liquid.get_temperature():.0f} K")

# Step 3c: Combine solid and liquid into two-phase slab
solid_cell = atoms_solid.get_cell().copy()
liquid_cell = atoms_liquid.get_cell().copy()

# New cell: doubled along slab direction
new_cell = solid_cell.copy()
new_cell[SLAB_DIRECTION] = solid_cell[SLAB_DIRECTION] + liquid_cell[SLAB_DIRECTION]

# Shift liquid positions along slab direction
liquid_positions = atoms_liquid.get_positions().copy()
shift = solid_cell[SLAB_DIRECTION, SLAB_DIRECTION]
liquid_positions[:, SLAB_DIRECTION] += shift

combined_positions = np.vstack([atoms_solid.get_positions(), liquid_positions])
combined_numbers = list(atoms_solid.get_atomic_numbers()) + \
                   list(atoms_liquid.get_atomic_numbers())

atoms_2phase = Atoms(
    numbers=combined_numbers,
    positions=combined_positions,
    cell=new_cell,
    pbc=True,
)
atoms_2phase.calc = calc
n_total = len(atoms_2phase)

write(os.path.join(WORK_DIR, "two_phase_initial.xyz"), atoms_2phase)
print(f"\nTwo-phase slab created: {n_total} atoms")
print(f"  Solid half: atoms 0-{n_half-1}")
print(f"  Liquid half: atoms {n_half}-{n_total-1}")
print(f"  Cell z-length: {new_cell[SLAB_DIRECTION, SLAB_DIRECTION]:.2f} A")

# ============================================================
# 4. HELPER: PE PROFILE AND ORDER PARAMETER
# ============================================================

def compute_pe_profile(atoms, direction, n_bins=40):
    """Compute per-atom PE averaged in spatial bins along a direction.
    Solid regions have lower PE than liquid regions.
    """
    try:
        energies_atom = atoms.get_potential_energies()
    except Exception:
        # Fallback: use total PE / N as uniform estimate
        return None, None, None

    positions = atoms.get_positions()
    cell = atoms.get_cell()
    L = cell[direction, direction]

    bin_edges = np.linspace(0, L, n_bins + 1)
    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])

    pe_profile = np.zeros(n_bins)
    counts = np.zeros(n_bins)

    for pos_z, e in zip(positions[:, direction], energies_atom):
        idx = min(int(pos_z / L * n_bins), n_bins - 1)
        if idx < 0:
            idx = 0
        pe_profile[idx] += e
        counts[idx] += 1

    mask = counts > 0
    pe_profile[mask] /= counts[mask]
    return bin_centers, pe_profile, counts


def compute_density_profile(atoms, direction, n_bins=40):
    """Compute number density profile along a direction."""
    positions = atoms.get_positions()
    cell = atoms.get_cell()
    L = cell[direction, direction]

    bin_edges = np.linspace(0, L, n_bins + 1)
    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    dz = bin_edges[1] - bin_edges[0]

    # Cross-sectional area
    dirs = [0, 1, 2]
    dirs.remove(direction)
    area = np.linalg.norm(np.cross(cell[dirs[0]], cell[dirs[1]]))

    hist, _ = np.histogram(positions[:, direction], bins=bin_edges)
    density = hist / (area * dz)  # atoms / A^3

    return bin_centers, density


def compute_steinhardt_q6(atoms, cutoff=None):
    """
    Compute Steinhardt Q6 bond-orientational order parameter per atom.
    Q6 ~ 0.57 for FCC, ~0.51 for BCC, ~0.48 for HCP, ~0.0 for liquid.
    Uses ASE neighbor list.

    Returns per-atom Q6 array.
    """
    from ase.neighborlist import neighbor_list
    from scipy.special import sph_harm

    if cutoff is None:
        # Estimate cutoff from average nearest-neighbor distance
        from ase.geometry import get_distances
        pos = atoms.get_positions()
        cell = atoms.get_cell()
        n_sample = min(20, len(atoms))
        _, d = get_distances(pos[:n_sample], pos, cell=cell, pbc=True)
        d_flat = d[d > 0.1]
        if len(d_flat) > 0:
            cutoff = 1.5 * np.min(d_flat)
        else:
            cutoff = 3.5  # fallback

    i_idx, j_idx, d_vec = neighbor_list("ijD", atoms, cutoff=cutoff)

    n_atoms = len(atoms)
    l = 6  # Q6
    q6_per_atom = np.zeros(n_atoms)

    for i_atom in range(n_atoms):
        mask = i_idx == i_atom
        if not np.any(mask):
            continue
        vecs = d_vec[mask]
        n_neigh = len(vecs)

        # Convert to spherical coordinates
        r = np.linalg.norm(vecs, axis=1)
        theta = np.arccos(np.clip(vecs[:, 2] / r, -1, 1))
        phi = np.arctan2(vecs[:, 1], vecs[:, 0])

        # Compute q_lm
        q_lm = np.zeros(2 * l + 1, dtype=complex)
        for m in range(-l, l + 1):
            for k in range(n_neigh):
                q_lm[m + l] += sph_harm(m, l, phi[k], theta[k])
            q_lm[m + l] /= n_neigh

        # Q6 = sqrt(4*pi/(2l+1) * sum|q_lm|^2)
        q6_per_atom[i_atom] = np.sqrt(
            4 * np.pi / (2 * l + 1) * np.sum(np.abs(q_lm) ** 2)
        )

    return q6_per_atom


def classify_phase_from_pe_drift(pe_timeseries, threshold=1e-6):
    """Determine phase outcome from PE time series.

    Returns: 'SOLIDIFYING', 'MELTING', or 'COEXISTING'
    """
    x = np.arange(len(pe_timeseries))
    slope, _ = np.polyfit(x, pe_timeseries, 1)

    if slope < -threshold:
        return "SOLIDIFYING", slope
    elif slope > threshold:
        return "MELTING", slope
    else:
        return "COEXISTING", slope

# ============================================================
# 5. RUN COARSE BRACKET
# ============================================================

def run_trial_temperature(atoms_2phase_ref, T_trial, calc, label=""):
    """Run NPT MD at a trial temperature and determine phase outcome.

    Returns dict with PE timeseries, status, slope, etc.
    """
    trial_atoms = atoms_2phase_ref.copy()
    trial_atoms.calc = calc

    MaxwellBoltzmannDistribution(trial_atoms, temperature_K=max(T_trial, 10))
    Stationary(trial_atoms)

    # NPT dynamics
    dyn = NPTBerendsen(
        trial_atoms,
        timestep=TIMESTEP * units.fs,
        temperature_K=max(T_trial, 10),
        pressure_au=PRESSURE * units.GPa,
        taut=100 * units.fs,
        taup=1000 * units.fs,
        compressibility_au=4.57e-5 / units.bar,
    )

    # Equilibrate
    dyn.run(N_EQUIL)

    # Production: collect PE and volume
    pe_list = []
    vol_list = []
    temp_list = []
    n_atoms_local = len(trial_atoms)

    def _collect():
        pe_list.append(trial_atoms.get_potential_energy() / n_atoms_local)
        vol_list.append(trial_atoms.get_volume() / n_atoms_local)
        temp_list.append(trial_atoms.get_temperature())

    dyn.attach(_collect, interval=LOG_INTERVAL)
    dyn.run(N_PROD)

    pe_arr = np.array(pe_list)
    vol_arr = np.array(vol_list)
    temp_arr = np.array(temp_list)

    status, slope = classify_phase_from_pe_drift(pe_arr)

    # Save snapshot
    snap_path = os.path.join(WORK_DIR, f"twophase_T{T_trial:.0f}K{label}.xyz")
    write(snap_path, trial_atoms)

    # PE profile along slab
    pe_profile_data = compute_pe_profile(trial_atoms, SLAB_DIRECTION)
    density_profile_data = compute_density_profile(trial_atoms, SLAB_DIRECTION)

    return {
        "T": T_trial,
        "pe_mean": np.mean(pe_arr),
        "pe_std": np.std(pe_arr),
        "vol_mean": np.mean(vol_arr),
        "vol_std": np.std(vol_arr),
        "temp_mean": np.mean(temp_arr),
        "pe_slope": slope,
        "status": status,
        "pe_timeseries": pe_arr.tolist(),
        "vol_timeseries": vol_arr.tolist(),
        "pe_profile": pe_profile_data,
        "density_profile": density_profile_data,
        "final_atoms": trial_atoms,
    }


print(f"\n{'='*70}")
print(f"PHASE 1: Coarse temperature bracket")
print(f"  Trial temperatures: {T_BRACKET_COARSE}")
print(f"{'='*70}")

all_results = {}

for T_trial in T_BRACKET_COARSE:
    if T_trial <= 0:
        print(f"  Skipping T = {T_trial} K (non-positive)")
        continue
    print(f"\n--- Trial T = {T_trial:.0f} K ---")
    t0 = time.time()
    try:
        result = run_trial_temperature(atoms_2phase, T_trial, calc, label="_coarse")
        all_results[T_trial] = result
        elapsed = time.time() - t0
        print(f"  Status: {result['status']}, PE slope = {result['pe_slope']:.2e}, "
              f"<PE> = {result['pe_mean']:.4f} eV/atom ({elapsed:.0f}s)")
    except Exception as e:
        print(f"  ERROR at T={T_trial}: {e}")
        traceback.print_exc()

# ============================================================
# 6. BISECTION REFINEMENT
# ============================================================

def find_bracket(results_dict):
    """Find the tightest bracket [T_low, T_high] where
    T_low is the highest solidifying T and T_high is the lowest melting T.
    """
    solidifying = sorted([T for T, r in results_dict.items()
                          if r["status"] == "SOLIDIFYING"])
    melting = sorted([T for T, r in results_dict.items()
                      if r["status"] == "MELTING"])
    coexisting = sorted([T for T, r in results_dict.items()
                         if r["status"] == "COEXISTING"])

    T_low = max(solidifying) if solidifying else None
    T_high = min(melting) if melting else None

    return T_low, T_high, coexisting


T_low, T_high, T_coexist = find_bracket(all_results)

if T_low is not None and T_high is not None:
    print(f"\nCoarse bracket: {T_low:.0f} K (solidifying) -- {T_high:.0f} K (melting)")
elif T_coexist:
    print(f"\nCoexisting temperatures found: {T_coexist}")
    print(f"Melting point is near: {np.mean(T_coexist):.0f} K")
else:
    print("\nWARNING: Could not establish bracket from coarse scan.")
    print("All trials show same behavior. Expand temperature range.")

# Bisection refinement
iteration = 0
max_bisections = 10

while (T_low is not None and T_high is not None
       and (T_high - T_low) > TM_TOLERANCE
       and iteration < max_bisections):
    T_mid = (T_low + T_high) / 2.0
    iteration += 1
    print(f"\n--- Bisection {iteration}: T_mid = {T_mid:.1f} K "
          f"(bracket [{T_low:.0f}, {T_high:.0f}] K, width {T_high-T_low:.0f} K) ---")

    t0 = time.time()
    try:
        result = run_trial_temperature(atoms_2phase, T_mid, calc,
                                       label=f"_bisect{iteration}")
        all_results[T_mid] = result
        elapsed = time.time() - t0
        print(f"  Status: {result['status']}, PE slope = {result['pe_slope']:.2e} "
              f"({elapsed:.0f}s)")

        if result["status"] == "SOLIDIFYING":
            T_low = T_mid
        elif result["status"] == "MELTING":
            T_high = T_mid
        else:
            # Coexisting -- could be right at Tm
            # Narrow from both sides
            T_high = T_mid + TM_TOLERANCE / 2
            T_low = T_mid - TM_TOLERANCE / 2
            print(f"  Coexisting at {T_mid:.1f} K -- likely very close to Tm!")
            break
    except Exception as e:
        print(f"  ERROR at T={T_mid:.1f}: {e}")
        break

# ============================================================
# 7. FINAL MELTING POINT ESTIMATE
# ============================================================

T_low_final, T_high_final, T_coexist_final = find_bracket(all_results)

if T_coexist_final:
    T_m = np.mean(T_coexist_final)
    print(f"\n{'='*70}")
    print(f"RESULT: Tm = {T_m:.0f} K (coexisting temperature)")
    print(f"{'='*70}")
elif T_low_final is not None and T_high_final is not None:
    T_m = (T_low_final + T_high_final) / 2.0
    uncertainty = (T_high_final - T_low_final) / 2.0
    print(f"\n{'='*70}")
    print(f"RESULT: Tm = {T_m:.0f} +/- {uncertainty:.0f} K")
    print(f"  Bracket: [{T_low_final:.0f}, {T_high_final:.0f}] K")
    print(f"{'='*70}")
else:
    T_m = T_ESTIMATE
    print(f"\nWARNING: Could not determine Tm. Using initial estimate {T_ESTIMATE} K.")

# ============================================================
# 8. SAVE RESULTS
# ============================================================

summary = {
    "melting_point_K": float(T_m),
    "method": "solid-liquid coexistence (ASE + MACE)",
    "model": MACE_MODEL,
    "n_atoms": n_total,
    "formula": atoms_2phase.get_chemical_formula(),
    "trials": {},
}

for T_trial, r in sorted(all_results.items()):
    summary["trials"][f"{T_trial:.1f}"] = {
        "status": r["status"],
        "pe_slope": float(r["pe_slope"]),
        "pe_mean_eV_per_atom": float(r["pe_mean"]),
        "vol_mean_A3_per_atom": float(r["vol_mean"]),
        "temp_mean_K": float(r["temp_mean"]),
    }

with open(os.path.join(WORK_DIR, "coexistence_results.json"), "w") as f:
    json.dump(summary, f, indent=2)
print(f"\nResults saved to {WORK_DIR}/coexistence_results.json")

# Save tabular data
rows = []
for T_trial in sorted(all_results.keys()):
    r = all_results[T_trial]
    status_code = {"SOLIDIFYING": -1, "COEXISTING": 0, "MELTING": 1}[r["status"]]
    rows.append([T_trial, r["pe_mean"], r["pe_std"], r["vol_mean"],
                 r["vol_std"], r["pe_slope"], status_code])
data_out = np.array(rows)
np.savetxt(
    os.path.join(WORK_DIR, "coexistence_data.dat"), data_out,
    header="T(K)  PE_mean(eV/atom)  PE_std  Vol_mean(A3/atom)  Vol_std  PE_slope  Status(-1=solid,0=coex,1=melt)",
    fmt="%10.2f %14.6f %12.6f %12.4f %10.4f %14.6e %4d",
)

# ============================================================
# 9. VISUALIZATION
# ============================================================

fig, axes = plt.subplots(2, 3, figsize=(20, 12))

# (a) PE time series at each trial T
ax = axes[0, 0]
sorted_trials = sorted(all_results.keys())
cmap = plt.cm.coolwarm(np.linspace(0, 1, len(sorted_trials)))
for idx, T_trial in enumerate(sorted_trials):
    r = all_results[T_trial]
    pe_ts = np.array(r["pe_timeseries"])
    time_ps = np.arange(len(pe_ts)) * LOG_INTERVAL * TIMESTEP / 1000.0
    label_short = f"{T_trial:.0f}K ({r['status'][:3]})"
    ax.plot(time_ps, pe_ts, color=cmap[idx], linewidth=0.8, label=label_short)
ax.set_xlabel("Time (ps)")
ax.set_ylabel("PE (eV/atom)")
ax.set_title("PE Time Series at Trial Temperatures")
ax.legend(fontsize=6, ncol=2, loc="best")

# (b) PE slope vs temperature (bracketing diagram)
ax = axes[0, 1]
trial_Ts = sorted(all_results.keys())
slopes = [all_results[T]["pe_slope"] for T in trial_Ts]
colors = []
for T in trial_Ts:
    s = all_results[T]["status"]
    if s == "SOLIDIFYING":
        colors.append("blue")
    elif s == "MELTING":
        colors.append("red")
    else:
        colors.append("green")
ax.bar(trial_Ts, slopes, width=max(15, (max(trial_Ts)-min(trial_Ts))/len(trial_Ts)*0.6),
       color=colors, alpha=0.7, edgecolor="black", linewidth=0.5)
ax.axhline(0, color="black", linestyle="-", linewidth=0.5)
if T_low_final and T_high_final:
    ax.axvline(T_m, color="green", linestyle="--", linewidth=2,
               label=f"Tm ~ {T_m:.0f} K")
ax.set_xlabel("Temperature (K)")
ax.set_ylabel("PE Slope (eV/atom/step)")
ax.set_title("Phase Outcome vs Temperature\n(blue=solidify, red=melt, green=coexist)")
ax.legend()

# (c) Mean PE and Volume vs T
ax = axes[0, 2]
pe_means = [all_results[T]["pe_mean"] for T in trial_Ts]
vol_means = [all_results[T]["vol_mean"] for T in trial_Ts]
ax.plot(trial_Ts, pe_means, "o-", color="tab:blue", label="PE")
ax.set_xlabel("Temperature (K)")
ax.set_ylabel("PE (eV/atom)", color="tab:blue")
ax.tick_params(axis="y", labelcolor="tab:blue")
ax2 = ax.twinx()
ax2.plot(trial_Ts, vol_means, "s-", color="tab:green", label="Vol")
ax2.set_ylabel("Volume (A^3/atom)", color="tab:green")
ax2.tick_params(axis="y", labelcolor="tab:green")
ax.set_title("Mean PE and Volume vs T")
if T_low_final and T_high_final:
    ax.axvline(T_m, color="red", linestyle="--", alpha=0.7)

# (d) PE profile along slab at selected temperatures
ax = axes[1, 0]
for idx, T_trial in enumerate(sorted_trials):
    r = all_results[T_trial]
    pe_prof = r.get("pe_profile")
    if pe_prof is not None and pe_prof[0] is not None:
        bz, pp, _ = pe_prof
        ax.plot(bz, pp, color=cmap[idx], label=f"{T_trial:.0f} K")
ax.set_xlabel(f"Position along {'xyz'[SLAB_DIRECTION]} (A)")
ax.set_ylabel("PE per atom (eV)")
ax.set_title("PE Profile Along Slab")
ax.legend(fontsize=6, ncol=2)

# (e) Density profile along slab
ax = axes[1, 1]
for idx, T_trial in enumerate(sorted_trials):
    r = all_results[T_trial]
    dens_prof = r.get("density_profile")
    if dens_prof is not None and dens_prof[0] is not None:
        bz, rho = dens_prof
        ax.plot(bz, rho, color=cmap[idx], label=f"{T_trial:.0f} K")
ax.set_xlabel(f"Position along {'xyz'[SLAB_DIRECTION]} (A)")
ax.set_ylabel("Number density (atoms/A^3)")
ax.set_title("Density Profile Along Slab")
ax.legend(fontsize=6, ncol=2)

# (f) Q6 order parameter histogram at first and last trial T
ax = axes[1, 2]
T_first = sorted_trials[0]
T_last = sorted_trials[-1]
for T_sel, col, ls in [(T_first, "blue", "-"), (T_last, "red", "--")]:
    r = all_results[T_sel]
    final_atoms = r.get("final_atoms")
    if final_atoms is not None:
        try:
            q6 = compute_steinhardt_q6(final_atoms)
            ax.hist(q6, bins=50, alpha=0.5, color=col, density=True,
                    label=f"T={T_sel:.0f} K")
        except Exception:
            pass
ax.set_xlabel("Q6 Order Parameter")
ax.set_ylabel("Probability Density")
ax.set_title("Q6 Distribution (Solid vs Liquid)")
ax.legend()

fig.suptitle(f"Solid-Liquid Coexistence Analysis | Tm ~ {T_m:.0f} K", fontsize=15)
fig.tight_layout()
fig.savefig(os.path.join(WORK_DIR, "coexistence_analysis.png"), dpi=150,
            bbox_inches="tight")
plt.close()
print(f"Saved: {WORK_DIR}/coexistence_analysis.png")

print(f"\n*** MELTING POINT: {T_m:.0f} K ***")
```

### Method B: LAMMPS Coexistence with Classical Potentials (subprocess)

For metals with validated EAM/MEAM potentials. Much faster, enabling larger systems and longer runs.

#### Step B1: Prepare structure and LAMMPS data file

```python
#!/usr/bin/env python3
"""Convert CIF to LAMMPS data file for coexistence simulation."""
import os
from ase.io import read, write

STRUCTURE_FILE = "structure.cif"
SUPERCELL = (4, 4, 4)  # Per half -- total will be doubled along z

atoms = read(STRUCTURE_FILE)
atoms_half = atoms.repeat(SUPERCELL)
n_half = len(atoms_half)

# Double along z for full two-phase cell
atoms_full = atoms_half.repeat((1, 1, 2))
write("coexist_initial.lammps", atoms_full, format="lammps-data")
print(f"Wrote coexist_initial.lammps: {len(atoms_full)} atoms "
      f"({n_half} per half, z-doubled)")
print(f"Cell: {atoms_full.cell.cellpar()[:3]}")
```

#### Step B2: LAMMPS input -- create two-phase structure and run coexistence

Save as `in.coexist`:

```
# ==============================================================
# Solid-liquid coexistence for melting point determination
# Uses EAM potential. Adjust pair_style/pair_coeff for your system.
# ==============================================================

units           metal
atom_style      atomic
boundary        p p p

# Read initial crystalline structure (doubled along z)
read_data       coexist_initial.lammps

# ---------- Potential ----------
pair_style      eam/alloy
pair_coeff      * * Cu_mishin1.eam.alloy Cu
# Adjust: pair_coeff * * YourPotential.eam.alloy Elem1 Elem2 ...

neighbor        2.0 bin
neigh_modify    every 1 delay 0 check yes

# ---------- Variables ----------
variable        T_est    equal 1350       # Estimated Tm (K)
variable        T_melt   equal 2500       # High T to melt liquid half
variable        zlo      equal bound(all,zmin)
variable        zhi      equal bound(all,zmax)
variable        zmid     equal (v_zlo+v_zhi)/2.0

# ============================================================
# STEP 1: Minimize
# ============================================================
minimize        1.0e-6 1.0e-8 2000 20000
reset_timestep  0

# ============================================================
# STEP 2: Equilibrate entire crystal at estimated Tm
# ============================================================
velocity        all create ${T_est} 12345 dist gaussian
fix             eq1 all npt temp ${T_est} ${T_est} 0.1 iso 0.0 0.0 1.0
timestep        0.002
thermo          500
thermo_style    custom step temp pe ke etotal vol press lz
run             20000
unfix           eq1

# ============================================================
# STEP 3: Melt the upper half (z > zmid)
# Freeze lower half (solid), heat upper half to T_melt
# ============================================================
# Define groups
variable        zmid_now equal (bound(all,zmin)+bound(all,zmax))/2.0
region          solid_region block INF INF INF INF INF ${zmid_now}
region          liquid_region block INF INF INF INF ${zmid_now} INF
group           solid_grp region solid_region
group           liquid_grp region liquid_region

# Fix solid atoms in place
fix             freeze solid_grp setforce 0.0 0.0 0.0
velocity        solid_grp set 0.0 0.0 0.0

# Heat liquid half
velocity        liquid_grp create ${T_melt} 54321 dist gaussian
fix             melt_npt liquid_grp nvt temp ${T_melt} ${T_melt} 0.1
timestep        0.002
run             30000
unfix           melt_npt

# Cool liquid half to near Tm
fix             cool_npt liquid_grp nvt temp ${T_est} ${T_est} 0.1
run             10000
unfix           cool_npt
unfix           freeze

write_data      coexist_prepared.lammps
write_dump      all custom coexist_prepared.dump id type x y z

# ============================================================
# STEP 4: Production runs at trial temperatures
# Use a loop or run separate input files at each T_trial
# ============================================================
# Reset for production
reset_timestep  0
velocity        all create ${T_est} 99999 dist gaussian

# Full NPT at estimated Tm -- observe if system solidifies or melts
fix             prod all npt temp ${T_est} ${T_est} 0.1 aniso 0.0 0.0 1.0 couple xy
timestep        0.002
thermo          200
thermo_style    custom step temp pe ke etotal vol press lz

# Track per-atom PE for order parameter
compute         pe_atom all pe/atom
# Track centro-symmetry parameter (order parameter for FCC/BCC)
compute         csp all centro/atom fcc
# For BCC use: compute csp all centro/atom bcc

variable        mype equal pe/atoms
variable        myvol equal vol/atoms
variable        mystep equal step
variable        mytemp equal temp
fix             thermo_log all print 200 "${mystep} ${mytemp} ${mype} ${myvol}" &
                file coexist_T${T_est}_thermo.dat screen no &
                title "# step temp(K) pe(eV/atom) vol(A3/atom)"

dump            prod_dump all custom 5000 coexist_T${T_est}.dump &
                id type x y z c_pe_atom c_csp
dump_modify     prod_dump sort id

run             200000

print "Coexistence run at T = ${T_est} K complete"
```

#### Step B3: Run at multiple trial temperatures

```python
#!/usr/bin/env python3
"""
Run LAMMPS coexistence simulations at multiple trial temperatures.
Generates separate LAMMPS input files from a template and runs them.
"""
import os
import subprocess
import sys

# Trial temperatures to test (K)
T_TRIALS = [1200, 1300, 1350, 1400, 1500]
LAMMPS_BIN = "lmp"  # or "lmp_mpi", etc.
NPROC = 4

# Read the template (in.coexist should have ${T_est} as variable)
TEMPLATE_FILE = "in.coexist"
if not os.path.exists(TEMPLATE_FILE):
    print(f"ERROR: Template {TEMPLATE_FILE} not found")
    sys.exit(1)

with open(TEMPLATE_FILE, "r") as f:
    template = f.read()

results = {}

for T in T_TRIALS:
    print(f"\n{'='*60}")
    print(f"Running coexistence at T = {T} K")
    print(f"{'='*60}")

    # Create modified input file
    input_text = template.replace(
        "variable        T_est    equal 1350",
        f"variable        T_est    equal {T}"
    )
    input_file = f"in.coexist_T{T}"
    with open(input_file, "w") as f:
        f.write(input_text)

    # Run LAMMPS
    log_file = f"log.coexist_T{T}"
    try:
        result = subprocess.run(
            ["mpirun", "-np", str(NPROC), LAMMPS_BIN, "-in", input_file],
            capture_output=True, text=True, timeout=3600
        )
        with open(log_file, "w") as f:
            f.write(result.stdout)

        if result.returncode != 0:
            print(f"  WARNING: LAMMPS returned exit code {result.returncode}")
            if result.stderr:
                print(f"  stderr: {result.stderr[-300:]}")
        else:
            print(f"  Completed successfully. Log: {log_file}")

        results[T] = {
            "returncode": result.returncode,
            "thermo_file": f"coexist_T{T}_thermo.dat",
        }
    except subprocess.TimeoutExpired:
        print(f"  TIMEOUT at T = {T} K")
        results[T] = {"returncode": -1, "error": "timeout"}
    except FileNotFoundError:
        print(f"  ERROR: LAMMPS binary '{LAMMPS_BIN}' not found")
        sys.exit(1)

print(f"\nAll trial runs complete. Thermo files ready for analysis.")
```

#### Step B4: Analyze LAMMPS coexistence results

```python
#!/usr/bin/env python3
"""
Analyze LAMMPS coexistence simulation results.
Reads thermo data from multiple trial temperatures, determines PE drift
direction, brackets Tm, and produces diagnostic plots.
"""
import os
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ============================================================
# Configuration
# ============================================================

T_TRIALS = [1200, 1300, 1350, 1400, 1500]  # Must match run script
THERMO_PATTERN = "coexist_T{T}_thermo.dat"
PE_SLOPE_THRESHOLD = 1e-7  # eV/atom/step

WORK_DIR = "."

# ============================================================
# Load and analyze thermo data
# ============================================================

results = {}

for T in T_TRIALS:
    fname = THERMO_PATTERN.format(T=T)
    if not os.path.exists(fname):
        print(f"WARNING: {fname} not found, skipping T={T} K")
        continue

    try:
        data = np.loadtxt(fname, comments="#")
        if data.ndim < 2 or data.shape[1] < 4:
            print(f"WARNING: {fname} has unexpected format, skipping")
            continue

        steps = data[:, 0]
        temp = data[:, 1]
        pe = data[:, 2]
        vol = data[:, 3]
    except Exception as e:
        print(f"WARNING: Error reading {fname}: {e}")
        continue

    # Discard initial equilibration transient (first 20%)
    n_start = len(pe) // 5
    pe_prod = pe[n_start:]
    vol_prod = vol[n_start:]
    steps_prod = steps[n_start:]
    temp_prod = temp[n_start:]

    # Fit linear trend to PE
    x = np.arange(len(pe_prod))
    slope, intercept = np.polyfit(x, pe_prod, 1)

    if slope < -PE_SLOPE_THRESHOLD:
        status = "SOLIDIFYING"
    elif slope > PE_SLOPE_THRESHOLD:
        status = "MELTING"
    else:
        status = "COEXISTING"

    results[T] = {
        "steps": steps,
        "temp": temp,
        "pe": pe,
        "vol": vol,
        "pe_prod": pe_prod,
        "pe_slope": slope,
        "pe_mean": np.mean(pe_prod),
        "pe_std": np.std(pe_prod),
        "vol_mean": np.mean(vol_prod),
        "vol_std": np.std(vol_prod),
        "temp_mean": np.mean(temp_prod),
        "status": status,
    }

    print(f"T = {T:6.0f} K | <PE> = {np.mean(pe_prod):.5f} eV/atom | "
          f"slope = {slope:+.2e} | {status}")

# ============================================================
# Bracket the melting point
# ============================================================

solidifying_Ts = sorted([T for T, r in results.items() if r["status"] == "SOLIDIFYING"])
melting_Ts = sorted([T for T, r in results.items() if r["status"] == "MELTING"])
coexisting_Ts = sorted([T for T, r in results.items() if r["status"] == "COEXISTING"])

T_m = None
T_low = None
T_high = None

if solidifying_Ts and melting_Ts:
    T_low = max(solidifying_Ts)
    T_high = min(melting_Ts)
    T_m = (T_low + T_high) / 2.0
    print(f"\n*** Tm = {T_m:.0f} K (bracket: [{T_low:.0f}, {T_high:.0f}] K) ***")
elif coexisting_Ts:
    T_m = np.mean(coexisting_Ts)
    print(f"\n*** Tm ~ {T_m:.0f} K (coexisting temperature(s): {coexisting_Ts}) ***")
else:
    print("\nWARNING: Cannot bracket Tm. All trials show the same behavior.")
    print("  Try a wider temperature range.")

# ============================================================
# Visualization
# ============================================================

fig, axes = plt.subplots(2, 2, figsize=(14, 12))

# (a) PE time series
ax = axes[0, 0]
cmap = plt.cm.coolwarm(np.linspace(0, 1, len(results)))
for idx, T in enumerate(sorted(results.keys())):
    r = results[T]
    timestep_ps = 0.002  # metal units, 2 fs
    time_ps = r["steps"] * timestep_ps / 1000.0  # convert to ps
    ax.plot(time_ps, r["pe"], color=cmap[idx], linewidth=0.5,
            label=f"{T:.0f} K ({r['status'][:3]})")
ax.set_xlabel("Time (ps)")
ax.set_ylabel("PE (eV/atom)")
ax.set_title("Potential Energy Time Series")
ax.legend(fontsize=7, ncol=2)

# (b) PE slope vs T
ax = axes[0, 1]
trial_Ts = sorted(results.keys())
slopes = [results[T]["pe_slope"] for T in trial_Ts]
colors_bar = []
for T in trial_Ts:
    s = results[T]["status"]
    if s == "SOLIDIFYING":
        colors_bar.append("blue")
    elif s == "MELTING":
        colors_bar.append("red")
    else:
        colors_bar.append("green")

bar_width = max(10, (max(trial_Ts) - min(trial_Ts)) / len(trial_Ts) * 0.6) \
    if len(trial_Ts) > 1 else 20
ax.bar(trial_Ts, slopes, width=bar_width, color=colors_bar, alpha=0.7,
       edgecolor="black", linewidth=0.5)
ax.axhline(0, color="black", linewidth=0.5)
if T_m:
    ax.axvline(T_m, color="green", linestyle="--", linewidth=2,
               label=f"Tm ~ {T_m:.0f} K")
ax.set_xlabel("Temperature (K)")
ax.set_ylabel("PE slope (eV/atom/step)")
ax.set_title("Phase Outcome\n(blue=solidify, red=melt, green=coexist)")
ax.legend()

# (c) Mean PE vs T
ax = axes[1, 0]
pe_means = [results[T]["pe_mean"] for T in trial_Ts]
pe_stds = [results[T]["pe_std"] for T in trial_Ts]
ax.errorbar(trial_Ts, pe_means, yerr=pe_stds, fmt="o-", capsize=3,
            color="tab:blue", markersize=5)
if T_m:
    ax.axvline(T_m, color="red", linestyle="--", alpha=0.7,
               label=f"Tm ~ {T_m:.0f} K")
ax.set_xlabel("Temperature (K)")
ax.set_ylabel("Mean PE (eV/atom)")
ax.set_title("Mean Potential Energy vs Temperature")
ax.legend()

# (d) Mean Volume vs T
ax = axes[1, 1]
vol_means = [results[T]["vol_mean"] for T in trial_Ts]
vol_stds = [results[T]["vol_std"] for T in trial_Ts]
ax.errorbar(trial_Ts, vol_means, yerr=vol_stds, fmt="s-", capsize=3,
            color="tab:green", markersize=5)
if T_m:
    ax.axvline(T_m, color="red", linestyle="--", alpha=0.7,
               label=f"Tm ~ {T_m:.0f} K")
ax.set_xlabel("Temperature (K)")
ax.set_ylabel("Mean Volume (A^3/atom)")
ax.set_title("Mean Volume vs Temperature")
ax.legend()

fig.suptitle(f"LAMMPS Coexistence Analysis | Tm ~ {T_m:.0f} K" if T_m
             else "LAMMPS Coexistence Analysis", fontsize=15)
fig.tight_layout()
fig.savefig("coexistence_lammps_analysis.png", dpi=150, bbox_inches="tight")
plt.close()
print(f"\nSaved: coexistence_lammps_analysis.png")
```

### Method C: LAMMPS + MACE-MP (ML Potential via pair_style mace)

For systems where you want LAMMPS efficiency with MACE universality. Requires LAMMPS compiled with the ML-IAP or MACE pair style.

#### LAMMPS input with MACE potential

Save as `in.coexist_mace`:

```
# ==============================================================
# Solid-liquid coexistence with MACE ML potential in LAMMPS
# Requires LAMMPS compiled with pair_style mace
# ==============================================================

units           metal
atom_style      atomic
boundary        p p p

read_data       coexist_initial.lammps

# ---------- MACE ML Potential ----------
# Option 1: pair_style mace (if compiled with MACE support)
pair_style      mace no_domain_decomposition
pair_coeff      * * /path/to/mace_model.model Cu
# Adjust element mapping for multi-component systems

# Option 2: pair_style mace/mp (MACE foundation model)
# pair_style    mace no_domain_decomposition
# pair_coeff    * * /path/to/2024-01-07-mace-128-L2_epoch-199.model Cu

neighbor        2.0 bin
neigh_modify    every 1 delay 0 check yes

# ---------- Variables ----------
variable        T_trial  equal 1350

# Minimize
minimize        1.0e-4 1.0e-6 1000 10000
reset_timestep  0

# Equilibrate at trial T
velocity        all create ${T_trial} 12345 dist gaussian
fix             eq all npt temp ${T_trial} ${T_trial} 0.1 aniso 0.0 0.0 1.0 couple xy
timestep        0.001    # 1 fs (conservative for ML potential)
thermo          200
thermo_style    custom step temp pe ke etotal vol press lz
run             10000
unfix           eq

# Production
fix             prod all npt temp ${T_trial} ${T_trial} 0.1 aniso 0.0 0.0 1.0 couple xy
timestep        0.001

variable        mype equal pe/atoms
variable        myvol equal vol/atoms
variable        mystep equal step
variable        mytemp equal temp
fix             thermo_log all print 200 "${mystep} ${mytemp} ${mype} ${myvol}" &
                file coexist_mace_T${T_trial}_thermo.dat screen no &
                title "# step temp(K) pe(eV/atom) vol(A3/atom)"

dump            1 all custom 5000 coexist_mace_T${T_trial}.dump id type x y z
dump_modify     1 sort id

run             100000

print "MACE coexistence at T = ${T_trial} K complete"
```

Use the same multi-temperature runner (Step B3) and analysis (Step B4) scripts, substituting file name patterns as needed.

---

## Key Parameters

| Parameter | Typical Value | Notes |
|---|---|---|
| System size | >500 atoms (>250 per half) | Minimum for reliable interface; 2000+ recommended |
| Slab direction | z (index 2) | Interface normal; cell must be orthorhombic along this axis |
| Supercell per half | 3x3x3 to 5x5x5 | Depends on unit cell; aim for >250 atoms per half |
| Timestep | 1-2 fs (ASE/MACE), 2 fs (LAMMPS EAM) | 0.5-1 fs if H or Li present |
| T_MELT (liquid prep) | 1.5-2.0 * T_estimate | Must fully disorder the liquid half |
| N_MELT_STEPS | 5000-10000 | Enough to equilibrate liquid; check RDF |
| N_EQUIL per trial T | 3000-10000 | Must stabilize T and P before production |
| N_PROD per trial T | 10000-50000 (ASE), 100000-500000 (LAMMPS) | Longer = clearer PE drift signal |
| Initial bracket width | ~600 K centered on T_estimate | E.g., T_est +/- 300 K |
| Bisection tolerance | 10-50 K | Finer = more iterations but better precision |
| NPT taut | 100 fs | Nose-Hoover thermostat coupling |
| NPT taup | 1000 fs | Barostat coupling; too tight causes oscillations |
| PE slope threshold | ~1e-6 to 1e-7 eV/atom/step | For classifying solidify/melt/coexist |
| MACE model | "medium" or "large" | "large" more accurate but 3-5x slower |
| LAMMPS aniso + couple xy | Required | Allows z-dimension to change independently (interface growth) |

## Interpreting Results

### PE Drift Direction
- **Downward PE drift**: System is solidifying. Temperature is below Tm. The solid-liquid interface migrates into the liquid region.
- **Upward PE drift**: System is melting. Temperature is above Tm. The interface migrates into the solid region.
- **Flat PE (fluctuations around constant)**: Coexistence. Temperature is at or very near Tm. This is the melting point.
- A clear linear trend should emerge within 10-20 ps (ASE) or 100-200 ps (LAMMPS).

### PE Profile Along Slab
- Solid region: lower PE per atom (more negative, more cohesive)
- Liquid region: higher PE per atom
- At Tm: the solid/liquid boundary is stable in position
- Below Tm: the solid region grows (boundary moves toward liquid)
- Above Tm: the liquid region grows (boundary moves toward solid)

### Density Profile
- Solid typically has higher number density (atoms more closely packed)
- Liquid has lower density (for most materials; exceptions: Si, Ge, H2O)
- Sharp density change at the interface
- Profile width of the interface region is typically 2-4 atomic layers

### Order Parameters
- **Q6 (Steinhardt)**: Q6 ~ 0.57 for FCC, ~0.51 for BCC, ~0.35 for HCP, near 0 for liquid. Bimodal histogram indicates coexistence.
- **Centro-symmetry parameter (LAMMPS)**: 0 for perfect crystal, large values for liquid. Available via `compute centro/atom`.
- **Common Neighbor Analysis**: Classifies atoms as FCC, BCC, HCP, or "other" (disordered/liquid). Track fraction of crystalline atoms over time.

### Finite-Size Effects
- Small systems (<500 atoms) can show ambiguous results because the interface region is a large fraction of the slab.
- The interface has an energy cost (interface free energy), which shifts the apparent Tm slightly in small systems.
- For production results, use >2000 atoms and verify that Tm converges with system size.

## Common Issues

**Liquid half re-crystallizes before production begins**: T_MELT is too low, or the cooling step brings it below Tm. Increase T_MELT to 2-3x T_estimate. Also increase N_MELT_STEPS. Verify with RDF that the liquid half is truly disordered before starting trial runs.

**Both halves melt at all trial temperatures**: All trial temperatures are above Tm. Lower the trial temperature range. Or the initial structure is not well-relaxed and is mechanically unstable.

**Both halves solidify at all trial temperatures**: All trial temperatures are below Tm. Raise the temperature range. Or the liquid half was not fully melted during preparation.

**PE shows no clear drift (noisy)**: Production run too short. Increase N_PROD. Or the system is very close to Tm (which is actually a good sign -- coexistence). Also increase system size to reduce noise.

**Cell becomes non-orthorhombic or collapses**: Use `couple xy` in LAMMPS to keep x and y dimensions equal while letting z change freely. In ASE, the Berendsen barostat is simpler and less prone to cell distortion than Parrinello-Rahman for this application.

**Interface drifts to cell boundary (periodic image)**: The slab is too thin. Increase the supercell along the slab direction so each half has at least 4-5 unit cell repeats.

**MACE gives unphysical energies at very high T**: The MACE foundation models are trained on near-equilibrium data. At T > 2*Tm, energetics may be unreliable. Use a less extreme T_MELT (1.5*Tm) and increase melt duration instead.

**LAMMPS with MACE is very slow**: The `no_domain_decomposition` flag limits parallelism. For large systems, consider using the MACE model exported to a format compatible with `pair_style allegro` or using GPU acceleration.

**Barostat oscillations in LAMMPS**: Increase `Pdamp` (pressure damping time) from 1.0 to 2.0 or higher. For anisotropic NPT, ensure `couple xy` is set so the interface direction (z) can evolve independently.
