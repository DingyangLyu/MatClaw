# Melting Point Determination and Liquid Structure

## When to Use

- Determining the melting temperature (Tm) of a crystalline material
- Studying solid-to-liquid phase transitions
- Computing liquid structure properties (RDF, structure factor, diffusion)
- Validating interatomic potentials against experimental melting points
- Screening thermal stability of new materials or alloys
- Generating equilibrated liquid structures for downstream workflows (e.g., amorphous generation)

## Method Selection

```
What approach should you use?

Quick estimate of melting point (within ~100-200 K)?
  --> Method A: ASE + MACE heating curve (discrete T scan)
  Advantages: Easy setup, works for any chemistry MACE supports
  Limitations: Superheating bias (single-phase melting overshoots Tm)

Accurate melting point (within ~50 K)?
  --> Method A variant: Two-phase coexistence (solid-liquid interface)
  Advantages: No superheating; gives thermodynamic Tm
  Limitations: Requires large supercell (>500 atoms), longer runs

Metal with validated EAM potential?
  --> Method B: LAMMPS heating simulation
  Advantages: Very fast, handles large cells and long runs
  Limitations: Restricted to elements/alloys with good EAM potentials

System size check:
  < 200 atoms: Too small for reliable melting, increase supercell
  200-500 atoms: Acceptable for heating curve screening
  500+ atoms: Good for two-phase coexistence
```

## Prerequisites

Pre-installed: `ase`, `mace-torch`, `pymatgen`, `numpy`, `scipy`, `matplotlib`, `spglib`.

LAMMPS binary: `lmp` (with MANYBODY, KSPACE packages for EAM/MEAM potentials).

For LAMMPS EAM potentials, download from NIST Interatomic Potentials Repository or use files bundled with LAMMPS in the potentials directory.

## Detailed Steps

### Method A: ASE + MACE -- Heating Curve (Discrete Temperature Scan)

This method heats a supercell at a series of discrete temperatures under NPT conditions. The melting point is identified from the discontinuity in potential energy and volume versus temperature (latent heat signature).

```python
#!/usr/bin/env python3
"""
Melting point determination via discrete temperature heating curve.
Uses ASE + MACE NPT-MD at multiple temperatures.
Detects melting from E(T) and V(T) discontinuities plus Lindemann criterion.

Workflow (inspired by atomate2 MPMorph):
  1. Relax unit cell at 0 K
  2. Build supercell
  3. NPT MD at each temperature (ascending)
  4. Collect <E>, <V>, Lindemann ratio, RDF at each T
  5. Plot E(T), V(T) to locate melting point
"""

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from ase.io import read, write
from ase.optimize import LBFGS
from ase.constraints import ExpCellFilter
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution, Stationary
from ase.md.langevin import Langevin
from ase.md.nptberendsen import NPTBerendsen
from ase import units

from mace.calculators import mace_mp
from pymatgen.core import Structure

import warnings
warnings.filterwarnings("ignore")

# ============================================================
# 1. CONFIGURATION
# ============================================================

STRUCTURE_FILE = "structure.cif"       # Input crystalline structure
MACE_MODEL = "medium"                  # "small", "medium", "large"
DEVICE = "cpu"                         # "cpu" or "cuda"

# Temperature scan parameters
T_START = 200                          # Starting temperature (K)
T_END = 3000                           # Maximum temperature (K)
T_STEP = 100                           # Temperature step (K)
# Refine around expected Tm with smaller step if desired

# MD parameters per temperature point
TIMESTEP = 2.0                         # fs (use 1.0 if light elements like Li, H)
N_EQUIL = 2000                         # Equilibration steps per temperature
N_PROD = 3000                          # Production steps per temperature
LOG_INTERVAL = 10                      # Logging interval (steps)

# NPT settings
PRESSURE = 0.0                         # GPa (ambient pressure)

# Supercell -- aim for >100 atoms, ideally >200
SUPERCELL = (3, 3, 3)                  # Adjust based on unit cell size

WORK_DIR = "/tmp/melting_curve"
os.makedirs(WORK_DIR, exist_ok=True)

# ============================================================
# 2. SET UP CALCULATOR AND STRUCTURE
# ============================================================

calc = mace_mp(model=MACE_MODEL, device=DEVICE, default_dtype="float64")

atoms = read(STRUCTURE_FILE)

# Relax unit cell at 0 K
atoms.calc = calc
ecf = ExpCellFilter(atoms)
opt = LBFGS(ecf, logfile=os.path.join(WORK_DIR, "relax.log"))
opt.run(fmax=0.01, steps=300)
write(os.path.join(WORK_DIR, "relaxed_unitcell.cif"), atoms)
print(f"Relaxed unit cell: {atoms.get_chemical_formula()}")
print(f"  Energy: {atoms.get_potential_energy():.4f} eV")

# Build supercell
atoms_sc = atoms.repeat(SUPERCELL)
atoms_sc.calc = calc
n_atoms = len(atoms_sc)
print(f"Supercell: {SUPERCELL}, {n_atoms} atoms")

if n_atoms < 100:
    print("WARNING: supercell has fewer than 100 atoms. "
          "Increase SUPERCELL for reliable melting detection.")

# Store reference positions for Lindemann criterion
ref_positions = atoms_sc.get_positions().copy()

# ============================================================
# 3. HELPER FUNCTIONS
# ============================================================

def compute_rdf(atoms, r_max=8.0, n_bins=200):
    """Compute radial distribution function g(r)."""
    from ase.geometry import get_distances
    cell = atoms.get_cell()
    positions = atoms.get_positions()
    n = len(atoms)

    # Use minimum image convention
    _, dist_matrix = get_distances(positions, cell=cell, pbc=True)

    r_edges = np.linspace(0, r_max, n_bins + 1)
    r_centers = 0.5 * (r_edges[:-1] + r_edges[1:])
    dr = r_edges[1] - r_edges[0]

    # Flatten upper triangle of distance matrix (exclude self)
    dists = dist_matrix[np.triu_indices(n, k=1)]

    hist, _ = np.histogram(dists, bins=r_edges)

    # Normalize: g(r) = hist / (n_pairs * 4*pi*r^2*dr * rho)
    volume = atoms.get_volume()
    rho = n / volume  # number density
    n_pairs = n * (n - 1) / 2
    shell_volumes = 4.0 * np.pi * r_centers**2 * dr
    # Each pair counted once in upper triangle
    g_r = hist / (n_pairs * shell_volumes * rho / n * 2)
    # Simpler normalization: ideal gas pair count in shell
    g_r = hist * volume / (n_pairs * shell_volumes * 2)

    return r_centers, g_r


def compute_rdf_fast(atoms, r_max=8.0, n_bins=200):
    """Compute RDF using neighbor list for efficiency."""
    from ase.neighborlist import neighbor_list
    n = len(atoms)
    volume = atoms.get_volume()
    rho = n / volume

    # Get all pair distances within r_max
    i_list, j_list, d_list = neighbor_list("ijd", atoms, cutoff=r_max)

    r_edges = np.linspace(0, r_max, n_bins + 1)
    r_centers = 0.5 * (r_edges[:-1] + r_edges[1:])
    dr = r_edges[1] - r_edges[0]

    hist, _ = np.histogram(d_list, bins=r_edges)

    # Each pair (i,j) appears twice in neighbor list (i->j and j->i)
    # g(r) = hist / (N * 4*pi*r^2*dr * rho) where hist counts each pair twice
    shell_volumes = 4.0 * np.pi * r_centers**2 * dr
    g_r = hist / (n * shell_volumes * rho)

    return r_centers, g_r


def compute_lindemann(atoms, ref_pos):
    """
    Compute Lindemann ratio: <u^2>^{1/2} / d_nn
    where u is displacement from reference and d_nn is nearest-neighbor distance.

    Lindemann criterion: melting when ratio > ~0.1 (typically 0.1-0.15).
    """
    from ase.geometry import get_distances

    # Mean squared displacement from reference positions
    displacements = atoms.get_positions() - ref_pos
    # Apply minimum image for displacements (handle PBC wrapping)
    cell = atoms.get_cell()
    # Simple approach: wrap displacements
    frac_disp = np.linalg.solve(cell.T, displacements.T).T
    frac_disp -= np.round(frac_disp)
    displacements = frac_disp @ cell

    msd = np.mean(np.sum(displacements**2, axis=1))
    rms_disp = np.sqrt(msd)

    # Nearest-neighbor distance from reference structure
    _, dists = get_distances(ref_pos[:1], ref_pos[1:], cell=cell, pbc=True)
    d_nn = np.min(dists[dists > 0.1])  # Exclude self

    lindemann = rms_disp / d_nn
    return lindemann


def compute_structure_factor(atoms, q_max=15.0, n_q=300):
    """Compute structure factor S(q) from RDF via Fourier transform."""
    r, g_r = compute_rdf_fast(atoms, r_max=10.0, n_bins=500)
    n = len(atoms)
    volume = atoms.get_volume()
    rho = n / volume

    q = np.linspace(0.1, q_max, n_q)
    S_q = np.ones_like(q)

    dr = r[1] - r[0]
    for iq, qi in enumerate(q):
        integrand = r * (g_r - 1.0) * np.sin(qi * r) / qi
        S_q[iq] = 1.0 + 4.0 * np.pi * rho * np.trapz(integrand, dx=dr)

    return q, S_q


# ============================================================
# 4. TEMPERATURE SCAN -- NPT MD AT EACH TEMPERATURE
# ============================================================

temperatures = np.arange(T_START, T_END + T_STEP, T_STEP)
results = {
    "T": [], "E_mean": [], "E_std": [], "V_mean": [], "V_std": [],
    "lindemann": [], "T_actual": [],
}
rdf_data = {}  # Store RDF at selected temperatures

print(f"\n{'='*70}")
print(f"Starting temperature scan: {T_START} K to {T_END} K, step {T_STEP} K")
print(f"{'='*70}")

# We reuse the atoms object and heat progressively (ascending T)
current_atoms = atoms_sc.copy()
current_atoms.calc = calc

for i_temp, T in enumerate(temperatures):
    print(f"\n--- T = {T:.0f} K ({i_temp+1}/{len(temperatures)}) ---")

    # Initialize or rescale velocities
    if i_temp == 0:
        MaxwellBoltzmannDistribution(current_atoms, temperature_K=T)
        Stationary(current_atoms)
    else:
        # Rescale velocities from previous temperature to new target
        current_T = current_atoms.get_temperature()
        if current_T > 1.0:
            scale = np.sqrt(T / current_T)
            current_atoms.set_momenta(current_atoms.get_momenta() * scale)
        else:
            MaxwellBoltzmannDistribution(current_atoms, temperature_K=T)
            Stationary(current_atoms)

    # Set up NPT dynamics (Berendsen)
    dt = TIMESTEP * units.fs
    pressure_au = PRESSURE * units.GPa

    dyn = NPTBerendsen(
        current_atoms,
        timestep=dt,
        temperature_K=T,
        pressure_au=pressure_au,
        taut=100 * units.fs,       # Thermostat coupling (100 fs)
        taup=1000 * units.fs,      # Barostat coupling (1000 fs)
        compressibility_au=4.57e-5 / units.bar,
    )

    # --- Equilibration ---
    dyn.run(N_EQUIL)

    # --- Production: collect averages ---
    energies = []
    volumes = []
    temp_inst = []

    def collect_data():
        energies.append(current_atoms.get_potential_energy())
        volumes.append(current_atoms.get_volume())
        temp_inst.append(current_atoms.get_temperature())

    dyn.attach(collect_data, interval=LOG_INTERVAL)
    dyn.run(N_PROD)

    # Remove the observer for next iteration
    dyn.observers = []

    # Compute averages
    E_mean = np.mean(energies)
    E_std = np.std(energies)
    V_mean = np.mean(volumes)
    V_std = np.std(volumes)
    T_actual = np.mean(temp_inst)
    linde = compute_lindemann(current_atoms, ref_positions)

    results["T"].append(T)
    results["E_mean"].append(E_mean / n_atoms)       # eV/atom
    results["E_std"].append(E_std / n_atoms)
    results["V_mean"].append(V_mean / n_atoms)        # A^3/atom
    results["V_std"].append(V_std / n_atoms)
    results["lindemann"].append(linde)
    results["T_actual"].append(T_actual)

    print(f"  <T> = {T_actual:.1f} K, <E> = {E_mean/n_atoms:.4f} eV/atom, "
          f"<V> = {V_mean/n_atoms:.3f} A^3/atom, Lindemann = {linde:.4f}")

    # Store RDF at selected temperatures
    if i_temp % max(1, len(temperatures) // 10) == 0 or linde > 0.08:
        r, g_r = compute_rdf_fast(current_atoms)
        rdf_data[T] = (r, g_r)

    # Save snapshot
    write(os.path.join(WORK_DIR, f"snapshot_T{T:.0f}K.xyz"), current_atoms)

    # Early stop if clearly melted (Lindemann > 0.3, well above threshold)
    # Continue a bit after melting to map out liquid region
    if linde > 0.3 and i_temp > 5:
        # Run a few more points in the liquid to map the curve
        remaining_liquid_points = 3
        if i_temp >= len(temperatures) - remaining_liquid_points:
            break

# ============================================================
# 5. DETECT MELTING POINT
# ============================================================

T_arr = np.array(results["T"])
E_arr = np.array(results["E_mean"])
V_arr = np.array(results["V_mean"])
L_arr = np.array(results["lindemann"])

# Method 1: Lindemann criterion (ratio > 0.1)
lindemann_threshold = 0.1
melted_mask = L_arr > lindemann_threshold
if np.any(melted_mask):
    T_lindemann = T_arr[melted_mask][0]
    print(f"\nLindemann criterion (>{lindemann_threshold}): T_m ~ {T_lindemann:.0f} K")
else:
    T_lindemann = None
    print("\nLindemann criterion: melting not detected in temperature range.")

# Method 2: Maximum dE/dT (latent heat causes spike)
if len(T_arr) > 3:
    dE_dT = np.gradient(E_arr, T_arr)
    idx_max_dEdT = np.argmax(dE_dT)
    T_energy_jump = T_arr[idx_max_dEdT]
    print(f"Max dE/dT at T ~ {T_energy_jump:.0f} K")

# Method 3: Maximum dV/dT
if len(T_arr) > 3:
    dV_dT = np.gradient(V_arr, T_arr)
    idx_max_dVdT = np.argmax(dV_dT)
    T_volume_jump = T_arr[idx_max_dVdT]
    print(f"Max dV/dT at T ~ {T_volume_jump:.0f} K")

# Best estimate (average of indicators)
estimates = [x for x in [T_lindemann, T_energy_jump, T_volume_jump] if x is not None]
if estimates:
    T_m_estimate = np.mean(estimates)
    print(f"\n*** Estimated melting point: {T_m_estimate:.0f} K ***")
    print(f"    (Note: single-phase heating overshoots by ~100-300 K due to superheating)")
    print(f"    (Use two-phase coexistence for more accurate Tm)")

# ============================================================
# 6. SAVE RESULTS
# ============================================================

data_out = np.column_stack([T_arr, E_arr, np.array(results["E_std"]),
                            V_arr, np.array(results["V_std"]), L_arr])
np.savetxt(
    os.path.join(WORK_DIR, "melting_curve.dat"), data_out,
    header="T(K)  E_mean(eV/atom)  E_std  V_mean(A3/atom)  V_std  Lindemann",
    fmt="%8.1f %14.6f %14.6f %12.4f %12.4f %10.5f",
)
print(f"\nData saved to {WORK_DIR}/melting_curve.dat")

# ============================================================
# 7. VISUALIZATION
# ============================================================

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# E(T) curve
ax = axes[0, 0]
ax.errorbar(T_arr, E_arr, yerr=np.array(results["E_std"]),
            fmt="o-", markersize=4, capsize=2, color="tab:blue")
if T_lindemann:
    ax.axvline(T_lindemann, color="red", linestyle="--", alpha=0.7,
               label=f"Lindemann Tm ~ {T_lindemann:.0f} K")
ax.set_xlabel("Temperature (K)")
ax.set_ylabel("Energy (eV/atom)")
ax.set_title("Energy vs Temperature")
ax.legend()

# V(T) curve
ax = axes[0, 1]
ax.errorbar(T_arr, V_arr, yerr=np.array(results["V_std"]),
            fmt="s-", markersize=4, capsize=2, color="tab:green")
if T_lindemann:
    ax.axvline(T_lindemann, color="red", linestyle="--", alpha=0.7,
               label=f"Lindemann Tm ~ {T_lindemann:.0f} K")
ax.set_xlabel("Temperature (K)")
ax.set_ylabel("Volume (A^3/atom)")
ax.set_title("Volume vs Temperature")
ax.legend()

# Lindemann ratio
ax = axes[1, 0]
ax.plot(T_arr, L_arr, "D-", markersize=4, color="tab:orange")
ax.axhline(lindemann_threshold, color="red", linestyle="--",
           label=f"Threshold = {lindemann_threshold}")
ax.set_xlabel("Temperature (K)")
ax.set_ylabel("Lindemann Ratio")
ax.set_title("Lindemann Criterion")
ax.legend()

# RDF comparison at selected temperatures
ax = axes[1, 1]
sorted_temps = sorted(rdf_data.keys())
colors = plt.cm.coolwarm(np.linspace(0, 1, len(sorted_temps)))
for idx, T_rdf in enumerate(sorted_temps):
    r, g_r = rdf_data[T_rdf]
    ax.plot(r, g_r + idx * 0.5, color=colors[idx], label=f"{T_rdf:.0f} K")
ax.set_xlabel("r (Angstrom)")
ax.set_ylabel("g(r) (offset)")
ax.set_title("RDF at Selected Temperatures")
ax.legend(fontsize=7, ncol=2)

fig.suptitle("Melting Point Analysis", fontsize=14)
fig.tight_layout()
fig.savefig(os.path.join(WORK_DIR, "melting_analysis.png"), dpi=150, bbox_inches="tight")
plt.close()
print(f"Saved: {WORK_DIR}/melting_analysis.png")
```

### Method A Variant: Two-Phase Coexistence Method

The two-phase coexistence method avoids superheating by creating a slab with half solid and half liquid. At the true melting point, the interface is stable; below Tm the whole system solidifies; above Tm it fully melts.

```python
#!/usr/bin/env python3
"""
Two-phase coexistence method for melting point determination.
Creates solid-liquid interface and runs NPT MD at trial temperatures.
The temperature at which the interface is stable gives the thermodynamic Tm.

Requires a larger supercell (>500 atoms recommended).
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
# CONFIGURATION
# ============================================================

STRUCTURE_FILE = "structure.cif"
MACE_MODEL = "medium"
DEVICE = "cpu"

# Two-phase setup: elongate in one direction for solid|liquid slab
SUPERCELL_BASE = (3, 3, 3)        # Base supercell for each half
SLAB_DIRECTION = 2                 # z-axis for the interface normal

# Melting protocol
T_MELT_INITIAL = 3000             # High T to melt half the slab (K)
N_MELT_STEPS = 5000               # Steps to melt the liquid half
T_TRIAL_TEMPS = [800, 1000, 1200, 1400, 1600]  # Trial temperatures (K)
# Adjust trial temperatures around expected Tm from heating curve

TIMESTEP = 2.0                     # fs
N_EQUIL = 3000                     # Equilibration steps at each trial T
N_PROD = 10000                     # Production steps to observe interface
LOG_INTERVAL = 50

PRESSURE = 0.0                     # GPa

WORK_DIR = "/tmp/two_phase_coexistence"
os.makedirs(WORK_DIR, exist_ok=True)

# ============================================================
# 1. BUILD TWO-PHASE STRUCTURE
# ============================================================

calc = mace_mp(model=MACE_MODEL, device=DEVICE, default_dtype="float64")

# Relax unit cell
atoms_unit = read(STRUCTURE_FILE)
atoms_unit.calc = calc
ecf = ExpCellFilter(atoms_unit)
opt = LBFGS(ecf, logfile=os.path.join(WORK_DIR, "relax.log"))
opt.run(fmax=0.01, steps=300)

# Build supercell and double along slab direction
sc = list(SUPERCELL_BASE)
sc[SLAB_DIRECTION] *= 2  # Double in slab direction for solid + liquid halves
atoms_2phase = atoms_unit.repeat(sc)
atoms_2phase.calc = calc
n_atoms = len(atoms_2phase)
print(f"Two-phase supercell: {sc}, {n_atoms} atoms")

# Identify atoms in each half
positions = atoms_2phase.get_positions()
cell = atoms_2phase.get_cell()
slab_length = cell[SLAB_DIRECTION, SLAB_DIRECTION]
midpoint = slab_length / 2.0

mask_liquid = positions[:, SLAB_DIRECTION] >= midpoint
mask_solid = ~mask_liquid
n_liquid = np.sum(mask_liquid)
n_solid = np.sum(mask_solid)
print(f"Solid half: {n_solid} atoms, Liquid half: {n_liquid} atoms")

# ============================================================
# 2. MELT THE LIQUID HALF
# ============================================================
# Strategy: Run high-T MD on the whole system but with constraints
# Alternative: melt a separate box and stitch together
# Here we use the simpler approach of melting the entire slab at high T
# then re-crystallizing one half by running at moderate T.

# Simpler approach: melt a separate copy, then combine
atoms_solid = atoms_unit.repeat(SUPERCELL_BASE)
atoms_solid.calc = calc
write(os.path.join(WORK_DIR, "solid_half.xyz"), atoms_solid)

# Create liquid by melting at high temperature
atoms_liquid = atoms_solid.copy()
atoms_liquid.calc = calc
MaxwellBoltzmannDistribution(atoms_liquid, temperature_K=T_MELT_INITIAL)
Stationary(atoms_liquid)

print(f"\nMelting liquid half at {T_MELT_INITIAL} K for {N_MELT_STEPS} steps...")
dyn_melt = Langevin(
    atoms_liquid,
    timestep=TIMESTEP * units.fs,
    temperature_K=T_MELT_INITIAL,
    friction=0.01 / units.fs,
)
dyn_melt.run(N_MELT_STEPS)
write(os.path.join(WORK_DIR, "liquid_half.xyz"), atoms_liquid)
print(f"Liquid half melted. Final T = {atoms_liquid.get_temperature():.0f} K")

# Combine solid and liquid into a two-phase slab
# Stack along SLAB_DIRECTION
from ase import Atoms

solid_cell = atoms_solid.get_cell()
liquid_cell = atoms_liquid.get_cell()

# New cell: doubled in slab direction
new_cell = solid_cell.copy()
new_cell[SLAB_DIRECTION] = solid_cell[SLAB_DIRECTION] + liquid_cell[SLAB_DIRECTION]

# Shift liquid positions
liquid_positions = atoms_liquid.get_positions().copy()
liquid_positions[:, SLAB_DIRECTION] += solid_cell[SLAB_DIRECTION, SLAB_DIRECTION]

combined_positions = np.vstack([atoms_solid.get_positions(), liquid_positions])
combined_numbers = list(atoms_solid.get_atomic_numbers()) + list(atoms_liquid.get_atomic_numbers())

atoms_2phase = Atoms(
    numbers=combined_numbers,
    positions=combined_positions,
    cell=new_cell,
    pbc=True,
)
atoms_2phase.calc = calc
n_atoms = len(atoms_2phase)

write(os.path.join(WORK_DIR, "two_phase_initial.xyz"), atoms_2phase)
print(f"\nTwo-phase structure created: {n_atoms} atoms")

# ============================================================
# 3. RUN TRIAL TEMPERATURES
# ============================================================

def compute_pe_profile(atoms, direction, n_bins=50):
    """Compute per-atom PE averaged in bins along a direction.
    Used to monitor solid vs. liquid regions.
    """
    positions = atoms.get_positions()
    energies_atom = atoms.get_potential_energies()
    cell = atoms.get_cell()
    L = cell[direction, direction]

    bin_edges = np.linspace(0, L, n_bins + 1)
    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])

    pe_profile = np.zeros(n_bins)
    counts = np.zeros(n_bins)

    for pos_z, e in zip(positions[:, direction], energies_atom):
        idx = int(pos_z / L * n_bins) % n_bins
        pe_profile[idx] += e
        counts[idx] += 1

    mask = counts > 0
    pe_profile[mask] /= counts[mask]
    return bin_centers, pe_profile, counts


print(f"\n{'='*70}")
print(f"Two-phase coexistence: testing temperatures {T_TRIAL_TEMPS}")
print(f"{'='*70}")

coexistence_results = {}

for T_trial in T_TRIAL_TEMPS:
    print(f"\n--- Trial T = {T_trial} K ---")

    trial_atoms = atoms_2phase.copy()
    trial_atoms.calc = calc

    MaxwellBoltzmannDistribution(trial_atoms, temperature_K=T_trial)
    Stationary(trial_atoms)

    # NPT dynamics
    dyn = NPTBerendsen(
        trial_atoms,
        timestep=TIMESTEP * units.fs,
        temperature_K=T_trial,
        pressure_au=PRESSURE * units.GPa,
        taut=100 * units.fs,
        taup=1000 * units.fs,
        compressibility_au=4.57e-5 / units.bar,
    )

    # Equilibrate
    print(f"  Equilibrating {N_EQUIL} steps...")
    dyn.run(N_EQUIL)

    # Production: monitor potential energy over time
    pe_timeseries = []
    vol_timeseries = []

    def collect():
        pe_timeseries.append(trial_atoms.get_potential_energy() / n_atoms)
        vol_timeseries.append(trial_atoms.get_volume() / n_atoms)

    dyn.attach(collect, interval=LOG_INTERVAL)
    print(f"  Production {N_PROD} steps...")
    dyn.run(N_PROD)

    pe_arr = np.array(pe_timeseries)
    vol_arr = np.array(vol_timeseries)

    # Check for drift: if PE drifts up --> melting; down --> solidifying
    # Fit linear trend to PE
    x = np.arange(len(pe_arr))
    slope, intercept = np.polyfit(x, pe_arr, 1)

    status = "COEXISTING"
    if slope > 1e-6:
        status = "MELTING (above Tm)"
    elif slope < -1e-6:
        status = "SOLIDIFYING (below Tm)"

    print(f"  <PE> = {np.mean(pe_arr):.4f} eV/atom, "
          f"<V> = {np.mean(vol_arr):.3f} A^3/atom, "
          f"PE slope = {slope:.2e}, Status: {status}")

    # Save PE profile along slab direction
    bin_z, pe_prof, counts = compute_pe_profile(trial_atoms, SLAB_DIRECTION)

    coexistence_results[T_trial] = {
        "pe_mean": np.mean(pe_arr),
        "vol_mean": np.mean(vol_arr),
        "pe_slope": slope,
        "status": status,
        "pe_timeseries": pe_arr,
        "pe_profile": (bin_z, pe_prof),
    }

    write(os.path.join(WORK_DIR, f"twophase_T{T_trial}K.xyz"), trial_atoms)

# ============================================================
# 4. DETERMINE Tm FROM COEXISTENCE
# ============================================================

# Tm is bracketed by the highest "SOLIDIFYING" T and lowest "MELTING" T
solidifying_temps = [T for T, r in coexistence_results.items()
                     if "SOLIDIFYING" in r["status"]]
melting_temps = [T for T, r in coexistence_results.items()
                 if "MELTING" in r["status"]]

if solidifying_temps and melting_temps:
    T_lower = max(solidifying_temps)
    T_upper = min(melting_temps)
    T_m_coexist = (T_lower + T_upper) / 2.0
    print(f"\n*** Two-phase Tm estimate: {T_m_coexist:.0f} K "
          f"(between {T_lower} and {T_upper} K) ***")
else:
    print("\nCould not bracket Tm. Try wider temperature range or finer steps.")
    coexist_temps = [T for T, r in coexistence_results.items()
                     if "COEXISTING" in r["status"]]
    if coexist_temps:
        print(f"Coexisting temperatures: {coexist_temps}")

# ============================================================
# 5. VISUALIZATION
# ============================================================

fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# PE time series at each trial T
ax = axes[0]
for T_trial in sorted(coexistence_results.keys()):
    r = coexistence_results[T_trial]
    time_ps = np.arange(len(r["pe_timeseries"])) * LOG_INTERVAL * TIMESTEP / 1000.0
    ax.plot(time_ps, r["pe_timeseries"], label=f"{T_trial} K ({r['status'][:4]})")
ax.set_xlabel("Time (ps)")
ax.set_ylabel("PE (eV/atom)")
ax.set_title("PE Time Series")
ax.legend(fontsize=7)

# PE profile along slab
ax = axes[1]
for T_trial in sorted(coexistence_results.keys()):
    r = coexistence_results[T_trial]
    bz, pp = r["pe_profile"]
    ax.plot(bz, pp, label=f"{T_trial} K")
ax.set_xlabel(f"Position along {'xyz'[SLAB_DIRECTION]} (Angstrom)")
ax.set_ylabel("PE per atom (eV)")
ax.set_title("PE Profile Along Slab")
ax.legend(fontsize=7)

# Volume vs T
ax = axes[2]
trial_T_sorted = sorted(coexistence_results.keys())
vol_means = [coexistence_results[T]["vol_mean"] for T in trial_T_sorted]
pe_slopes = [coexistence_results[T]["pe_slope"] for T in trial_T_sorted]
colors = ["blue" if s < -1e-6 else "red" if s > 1e-6 else "green" for s in pe_slopes]
ax.bar(trial_T_sorted, vol_means, width=T_STEP*0.6, color=colors, alpha=0.7)
ax.set_xlabel("Temperature (K)")
ax.set_ylabel("Volume (A^3/atom)")
ax.set_title("Volume at Trial Temperatures\n(blue=solidifying, red=melting, green=coexisting)")

fig.suptitle("Two-Phase Coexistence Analysis", fontsize=14)
fig.tight_layout()
fig.savefig(os.path.join(WORK_DIR, "two_phase_analysis.png"), dpi=150, bbox_inches="tight")
plt.close()
print(f"\nSaved: {WORK_DIR}/two_phase_analysis.png")
```

### Method B: LAMMPS Heating Simulation (Metals with EAM)

For metals with well-validated EAM potentials, LAMMPS is much faster and can handle larger systems and longer simulation times.

#### Step 1: Prepare structure file

```python
#!/usr/bin/env python3
"""Convert CIF to LAMMPS data file using ASE."""
from ase.io import read, write

atoms = read("structure.cif")
# Build supercell (aim for > 500 atoms for melting)
atoms = atoms.repeat((5, 5, 5))
write("structure.lammps", atoms, format="lammps-data")
print(f"Wrote structure.lammps with {len(atoms)} atoms")
```

#### Step 2: LAMMPS input file for heating ramp

Save as `in.melt`:

```
# LAMMPS melting point determination via continuous heating
# Works with EAM potentials for metals (Cu, Al, Fe, Ni, etc.)

units           metal
atom_style      atomic
boundary        p p p

# Read structure
read_data       structure.lammps

# EAM potential -- adjust path and element
# Download from: https://www.ctcms.nist.gov/potentials/
pair_style      eam/alloy
pair_coeff      * * Cu_mishin1.eam.alloy Cu
# For multi-element: pair_coeff * * file.eam.alloy Elem1 Elem2

# Settings
neighbor        2.0 bin
neigh_modify    every 1 delay 0 check yes

# Initial minimization
minimize        1.0e-6 1.0e-8 1000 10000
reset_timestep  0

# Thermodynamic output
thermo          100
thermo_style    custom step temp pe ke etotal vol press lx ly lz density

# Equilibrate at low temperature first
velocity        all create 300.0 12345 dist gaussian
fix             1 all npt temp 300.0 300.0 0.1 iso 0.0 0.0 1.0
timestep        0.002   # 2 fs
run             10000
unfix           1

# ============================================================
# Continuous heating ramp: 300 K -> 3000 K
# ============================================================
# Heating rate: (3000-300) K / (500000 * 0.002 ps) = 2.7e12 K/s
# Adjust run length and temperature range as needed

variable        T_start equal 300.0
variable        T_end   equal 3000.0

fix             2 all npt temp ${T_start} ${T_end} 0.1 iso 0.0 0.0 1.0
timestep        0.002

# Output trajectory for post-processing
dump            1 all custom 500 heating.dump id type x y z vx vy vz
dump_modify     1 sort id

# Output per-step thermo data for E(T) curve
variable        mytemp equal temp
variable        mype   equal pe/atoms
variable        myvol  equal vol/atoms
variable        mystep equal step
fix             thermo_out all print 100 "${mystep} ${mytemp} ${mype} ${myvol}" &
                file heating_thermo.dat screen no &
                title "# step temp(K) pe(eV/atom) vol(A3/atom)"

run             500000
unfix           2

print "Heating simulation complete"
```

#### Step 3: Run LAMMPS

```bash
# Run with available cores
lmp -in in.melt -log log.melt

# Or parallel
mpirun -np 4 lmp -in in.melt -log log.melt
```

#### Step 4: Analyze LAMMPS results

```python
#!/usr/bin/env python3
"""Analyze LAMMPS heating simulation to find melting point."""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Load thermo data
data = np.loadtxt("heating_thermo.dat", comments="#")
steps = data[:, 0]
temp = data[:, 1]
pe = data[:, 2]    # eV/atom
vol = data[:, 3]   # A^3/atom

# Smooth data with running average
def running_mean(x, N=50):
    return np.convolve(x, np.ones(N)/N, mode='valid')

temp_s = running_mean(temp)
pe_s = running_mean(pe)
vol_s = running_mean(vol)

# Find melting: max of d(PE)/dT
dpe_dT = np.gradient(pe_s, temp_s)
idx_melt = np.argmax(dpe_dT)
T_melt = temp_s[idx_melt]

dvol_dT = np.gradient(vol_s, temp_s)
idx_melt_v = np.argmax(dvol_dT)
T_melt_v = temp_s[idx_melt_v]

print(f"Melting point from max dE/dT: {T_melt:.0f} K")
print(f"Melting point from max dV/dT: {T_melt_v:.0f} K")
print(f"Note: continuous heating overestimates Tm due to superheating.")
print(f"True Tm is likely 100-300 K lower.")

# Plot
fig, axes = plt.subplots(2, 2, figsize=(12, 10))

axes[0,0].plot(temp_s, pe_s, 'b-', linewidth=0.5)
axes[0,0].axvline(T_melt, color='r', linestyle='--', label=f'Tm ~ {T_melt:.0f} K')
axes[0,0].set_xlabel("Temperature (K)")
axes[0,0].set_ylabel("PE (eV/atom)")
axes[0,0].set_title("Energy vs Temperature")
axes[0,0].legend()

axes[0,1].plot(temp_s, vol_s, 'g-', linewidth=0.5)
axes[0,1].axvline(T_melt_v, color='r', linestyle='--', label=f'Tm ~ {T_melt_v:.0f} K')
axes[0,1].set_xlabel("Temperature (K)")
axes[0,1].set_ylabel("Volume (A^3/atom)")
axes[0,1].set_title("Volume vs Temperature")
axes[0,1].legend()

axes[1,0].plot(temp_s, dpe_dT, 'm-', linewidth=0.5)
axes[1,0].axvline(T_melt, color='r', linestyle='--')
axes[1,0].set_xlabel("Temperature (K)")
axes[1,0].set_ylabel("dE/dT (eV/atom/K)")
axes[1,0].set_title("Heat Capacity (dE/dT)")

axes[1,1].plot(temp_s, dvol_dT, 'c-', linewidth=0.5)
axes[1,1].axvline(T_melt_v, color='r', linestyle='--')
axes[1,1].set_xlabel("Temperature (K)")
axes[1,1].set_ylabel("dV/dT (A^3/atom/K)")
axes[1,1].set_title("Thermal Expansion (dV/dT)")

fig.suptitle("LAMMPS Heating Simulation Analysis", fontsize=14)
fig.tight_layout()
fig.savefig("melting_analysis_lammps.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved: melting_analysis_lammps.png")
```

## Key Parameters

| Parameter | Typical Value | Notes |
|---|---|---|
| Supercell size | >100 atoms (heating curve), >500 (coexistence) | Too small causes artificial superheating and finite-size effects |
| Timestep | 1-2 fs | Use 0.5-1 fs if H or Li present |
| Equilibration | 2000-5000 steps per T point | Must fully equilibrate at each T |
| Production | 3000-10000 steps per T point | Longer for better statistics |
| Temperature step | 50-100 K (coarse), 25 K (refined) | Refine around expected Tm |
| Temperature range | 0.3*Tm_exp to 1.5*Tm_exp | Cover both solid and liquid clearly |
| NPT pressure coupling | 1000 fs (taup) | Too tight causes oscillations |
| NPT temperature coupling | 100 fs (taut) | Standard for Berendsen |
| Lindemann threshold | 0.1-0.15 | System-dependent; 0.1 is common |
| MACE model | "medium" or "large" | "large" more accurate but slower |
| Heating rate (LAMMPS) | 10^12-10^13 K/s | Lower is better but more expensive |

## Interpreting Results

### Energy vs. Temperature (E(T) curve)
- **Solid region**: E increases roughly linearly with T (slope ~ Cv)
- **Melting**: Sharp jump in E (latent heat of fusion)
- **Liquid region**: E increases linearly again, but with steeper slope (higher Cv)
- The midpoint of the energy jump gives an estimate of Tm

### Volume vs. Temperature (V(T) curve)
- Most materials expand upon melting (V_liquid > V_solid)
- Exception: water, Si, Ge, Bi have anomalous density changes
- V(T) jump location should agree with E(T) jump

### Lindemann Criterion
- Ratio of RMS atomic displacement to nearest-neighbor distance
- Solid: ratio < 0.1 (atoms oscillate around lattice sites)
- Liquid: ratio > 0.1-0.15 (atoms diffuse freely)
- Note: Lindemann ratio from single snapshots can be noisy; average over production run

### Radial Distribution Function (RDF)
- **Crystalline solid**: sharp peaks at lattice-neighbor distances
- **Near melting**: peaks broaden, second-neighbor peaks start merging
- **Liquid**: first peak survives (short-range order), long-range peaks disappear, g(r) -> 1 at large r

### Two-Phase Coexistence
- **Below Tm**: PE drifts downward (system solidifies); interface moves into liquid
- **Above Tm**: PE drifts upward (system melts); interface moves into solid
- **At Tm**: PE fluctuates without drift; both phases coexist stably

### Superheating Correction
Single-phase heating curve melting points typically overshoot by 100-300 K due to the nucleation barrier for melting in a perfect crystal. The two-phase coexistence method avoids this. If only a heating curve is available, subtract approximately 10-20% from the observed Tm as a rough correction.

## Common Issues

**System too small**: Fewer than 100 atoms gives large finite-size effects and artificial superheating. Increase supercell.

**Melting not detected**: Temperature range may not reach high enough, or the material has a very high melting point. Extend T_END. Also check if the potential (MACE or EAM) gives reasonable energetics for this material.

**Gradual rather than sharp transition**: Small systems show rounded transitions rather than sharp discontinuities. Increase supercell size. Also ensure equilibration is sufficient at each temperature.

**NPT barostat instability**: If the cell oscillates wildly or collapses, increase taup (pressure coupling time) or reduce timestep. The compressibility parameter also matters -- use a value appropriate for your material class.

**Lindemann ratio noisy**: Average over many snapshots in the production run rather than using a single snapshot. The implementation above uses instantaneous displacement from the reference, which can be affected by center-of-mass drift -- ensure Stationary() is applied.

**MACE model inaccuracy for high-T liquid**: The MACE foundation models are trained primarily on near-equilibrium structures. For very high temperatures (>2*Tm), energetics may become unreliable. Validate against known experimental Tm for a similar material.

**LAMMPS EAM potential not available**: Not all elements have reliable EAM potentials. For non-metals, alloys, or complex compositions, use ASE + MACE instead.

**Two-phase method: liquid half re-crystallizes too fast**: The initial melting temperature may not be high enough, or the liquid is not fully equilibrated. Increase T_MELT_INITIAL and N_MELT_STEPS. Also ensure the slab is thick enough (at least 3-4 unit cell lengths per half).
