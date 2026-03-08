# Metadynamics for Free Energy Surface Exploration

## When to Use

- Mapping free energy surfaces (FES) along collective variables (CVs) for phase transitions
- Escaping deep free energy minima that trap standard MD simulations
- Computing free energy barriers for solid-solid phase transitions (e.g., BCC to FCC)
- Studying reconstructive transitions, diffusion, or chemical reactions with rare-event character
- Determining minimum free energy paths between metastable phases
- Quantifying nucleation barriers and transition mechanisms
- Exploring conformational landscapes of surface adsorbates or molecular crystals

## Background

Metadynamics accelerates rare events by periodically depositing repulsive Gaussian bias potentials ("hills") along selected collective variables (CVs). Over time, the accumulated bias fills free energy basins, enabling the system to escape local minima and explore the full CV landscape. The free energy surface is reconstructed as the negative of the deposited bias.

**Well-tempered metadynamics** (Barducci et al., PRL 2008) improves convergence by progressively reducing hill heights according to:

    w(t) = w0 * exp(-V_bias(s, t) / (Delta_T * k_B))

where Delta_T is a bias temperature controlling the exploration-exploitation tradeoff. The bias factor gamma = (T + Delta_T) / T controls how aggressively barriers are filled. As t -> infinity, the bias converges to -(1 - 1/gamma) * F(s), giving a bounded, convergent estimate of the FES.

**Key concepts from pyiron_atomistics VaspMetadyn:**
- **Primitive constraints**: bond distance (R), angle (A), torsion (T), Cartesian positions (X, Y, Z) -- fundamental geometric CVs acting on specific atom indices
- **Complex constraints**: linear combinations (S), norms (C), coordination numbers (D) -- built from primitive constraints with coefficients
- **Biased vs. unbiased**: constraints can be passive (monitoring only, status=0) or active bias targets (status=5 or 7 for primitive/complex)
- **ICONST file**: defines the constraint topology; PENALTYPOT defines penalty/bias potential parameters
- **REPORT file**: logs CV values and bias at each step for post-processing

## Method Selection

```
What is your system and goal?

Small system (<500 atoms), MACE-compatible chemistry, moderate barriers?
  --> Method A: ASE + MACE metadynamics (pure Python, no external plugins)
  Advantages: Easy setup, any chemistry MACE supports, full control
  Limitations: Slower than LAMMPS, simple Gaussian hills implementation

Large system or long timescale needed?
  --> Method B: LAMMPS + PLUMED metadynamics
  Advantages: Fast, production-quality, full PLUMED CV library
  Limitations: Requires LAMMPS compiled with PLUMED plugin and a classical/ML potential in LAMMPS format

Need DFT accuracy for electronic-structure-sensitive transitions?
  --> Use VASP metadynamics (see VASP documentation for MDALGO=21, ICONST, PENALTYPOT)
  Limitations: Expensive, VASP license required; not scripted here

Decision on CVs:
  Phase transition (volume/density driven)?
    --> CV = cell volume, coordination number, or Steinhardt order parameters
  Bond breaking/formation?
    --> CV = interatomic distance or coordination number
  Surface diffusion?
    --> CV = Cartesian position (x, y) of the adsorbate
  Angle-dependent (e.g., octahedral tilting)?
    --> CV = bond angle or torsion angle
```

## Prerequisites

Pre-installed: `ase`, `mace-torch`, `numpy`, `scipy`, `matplotlib`, `pymatgen`.

For Method B: `lmp` binary compiled with PLUMED package (`lmp -h` should list PLUMED). PLUMED standalone CLI (`plumed info --root` should return a path).

---

## Detailed Steps

### Method A: ASE + MACE Well-Tempered Metadynamics

This method implements metadynamics entirely in Python using ASE dynamics and MACE as the energy/force engine. Gaussian hills are deposited along user-defined CVs, and forces are modified on-the-fly. No external plugin is required.

#### Script A1: Distance CV -- Dimer Dissociation / Recombination

```python
#!/usr/bin/env python3
"""
Well-tempered metadynamics with ASE + MACE.
CV: interatomic distance between two atoms in a bulk environment.
Example: vacancy-mediated diffusion hop in a metal (Al FCC).

The script:
  1. Builds a supercell with a vacancy
  2. Defines a distance CV between the migrating atom and its target site
  3. Runs well-tempered metadynamics depositing Gaussian hills on the distance CV
  4. Reconstructs the free energy surface F(d)
  5. Plots FES and CV trajectory

Standalone -- no pyiron dependency.
"""

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

from ase.build import bulk
from ase.optimize import LBFGS
from ase.md.langevin import Langevin
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution, Stationary
from ase.io import Trajectory
from ase import units

from mace.calculators import mace_mp

# ============================================================
# 1. CONFIGURATION
# ============================================================

MACE_MODEL = "medium"
DEVICE = "cpu"                         # "cpu" or "cuda"
SUPERCELL = (3, 3, 3)                  # 3x3x3 FCC Al = 108 atoms, remove 1 -> 107
TEMPERATURE = 600.0                    # K
TIMESTEP = 2.0                         # fs
N_STEPS = 20000                        # total MD steps
HILL_EVERY = 50                        # deposit a hill every N steps
HILL_HEIGHT_INIT = 0.05                # initial hill height (eV)
HILL_SIGMA = 0.15                      # Gaussian width in distance CV (Angstrom)
BIAS_FACTOR = 10.0                     # gamma for well-tempered metadynamics
CV_ATOM_1 = 0                          # index of migrating atom (set after vacancy creation)
CV_ATOM_2 = 1                          # index of nearest neighbor to vacancy site
FRICTION = 0.01                        # Langevin friction (1/fs)
OUTPUT_DIR = "metad_distance"

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ============================================================
# 2. BUILD SYSTEM -- Al FCC with one vacancy
# ============================================================

atoms = bulk("Al", "fcc", a=4.05, cubic=True) * SUPERCELL
n_atoms_perfect = len(atoms)

# Remove one atom to create a vacancy
vacancy_index = 13  # arbitrary interior atom
vacancy_pos = atoms.positions[vacancy_index].copy()
del atoms[vacancy_index]
print(f"Created {n_atoms_perfect-1}-atom Al supercell with vacancy at {vacancy_pos}")

# Identify the migrating atom: nearest neighbor to the vacancy
dists = np.linalg.norm(atoms.positions - vacancy_pos, axis=1)
CV_ATOM_1 = int(np.argmin(dists))  # closest atom to vacancy = migrant
# Target: second nearest as reference
sorted_idx = np.argsort(dists)
CV_ATOM_2 = int(sorted_idx[1])
print(f"CV atoms: migrant={CV_ATOM_1} (d={dists[CV_ATOM_1]:.2f} A), "
      f"reference={CV_ATOM_2} (d={dists[CV_ATOM_2]:.2f} A)")

# ============================================================
# 3. SET UP CALCULATOR AND RELAX
# ============================================================

calc = mace_mp(model=MACE_MODEL, dispersion=False, device=DEVICE, default_dtype="float64")
atoms.calc = calc

opt = LBFGS(atoms, logfile=os.path.join(OUTPUT_DIR, "relax.log"))
opt.run(fmax=0.01, steps=200)
print(f"Relaxed energy: {atoms.get_potential_energy():.4f} eV")

# ============================================================
# 4. METADYNAMICS ENGINE (pure Python)
# ============================================================

class WellTemperedMetadynamics:
    """
    Well-tempered metadynamics bias on a 1D collective variable.

    Deposits Gaussian hills periodically and modifies atomic forces
    to include the bias gradient. Tracks deposited hills for FES
    reconstruction.

    Reference: Barducci, Bussi, Parrinello, PRL 100, 020603 (2008).
    """

    def __init__(self, height, sigma, bias_factor, temperature, cv_func, cv_grad_func):
        """
        Args:
            height: initial hill height w0 (eV)
            sigma: Gaussian width in CV space
            bias_factor: gamma = (T + Delta_T) / T
            temperature: simulation temperature (K)
            cv_func: callable(atoms) -> float, returns CV value
            cv_grad_func: callable(atoms) -> array shape (N, 3), returns dCV/dr_i
        """
        self.w0 = height
        self.sigma = sigma
        self.gamma = bias_factor
        self.kT = temperature * units.kB  # eV
        self.delta_T_kT = (bias_factor - 1.0) * self.kT  # Delta_T * k_B
        self.cv_func = cv_func
        self.cv_grad_func = cv_grad_func
        # Storage: list of (cv_center, height) for each deposited hill
        self.hills = []
        self.cv_history = []
        self.bias_history = []

    def get_bias_energy(self, s):
        """Compute total bias V_bias(s) from all deposited hills."""
        V = 0.0
        for s_i, w_i in self.hills:
            V += w_i * np.exp(-0.5 * ((s - s_i) / self.sigma) ** 2)
        return V

    def get_bias_force_on_cv(self, s):
        """Compute -dV_bias/ds."""
        dVds = 0.0
        for s_i, w_i in self.hills:
            g = w_i * np.exp(-0.5 * ((s - s_i) / self.sigma) ** 2)
            dVds += g * (-(s - s_i) / self.sigma**2)
        return -dVds  # force = -gradient

    def deposit_hill(self, s):
        """Deposit a well-tempered Gaussian hill at CV value s."""
        V_current = self.get_bias_energy(s)
        # Well-tempered scaling: w = w0 * exp(-V_bias / (Delta_T * kB))
        w = self.w0 * np.exp(-V_current / self.delta_T_kT)
        self.hills.append((s, w))
        return w

    def get_bias_forces(self, atoms):
        """
        Compute the bias force on all atoms: F_bias_i = -(dV/ds) * (ds/dr_i).
        Returns array of shape (N, 3).
        """
        s = self.cv_func(atoms)
        # Force on CV
        f_cv = self.get_bias_force_on_cv(s)
        # Chain rule: F_i = f_cv * (ds/dr_i)
        grad_cv = self.cv_grad_func(atoms)  # (N, 3)
        F_bias = f_cv * grad_cv
        return F_bias, s

    def reconstruct_fes(self, s_min, s_max, n_bins=200):
        """
        Reconstruct FES from deposited hills.
        For well-tempered metadynamics: F(s) = -(gamma / (gamma - 1)) * V_bias(s)
        """
        s_grid = np.linspace(s_min, s_max, n_bins)
        V_bias = np.array([self.get_bias_energy(s) for s in s_grid])
        # Well-tempered correction
        F = -(self.gamma / (self.gamma - 1.0)) * V_bias
        # Shift minimum to zero
        F -= np.min(F)
        return s_grid, F


# ── Define CV: distance between two atoms ──────────────────────

def cv_distance(atoms, i=CV_ATOM_1, j=CV_ATOM_2):
    """Compute distance between atoms i and j (with MIC)."""
    vec = atoms.get_distance(i, j, mic=True, vector=True)
    return np.linalg.norm(vec)


def cv_distance_gradient(atoms, i=CV_ATOM_1, j=CV_ATOM_2):
    """
    Gradient of distance CV w.r.t. all atomic positions.
    d(|r_ij|)/dr_i = (r_i - r_j) / |r_ij|   for atom i
    d(|r_ij|)/dr_j = -(r_i - r_j) / |r_ij|  for atom j
    All others are zero.
    """
    vec = atoms.get_distance(i, j, mic=True, vector=True)
    d = np.linalg.norm(vec)
    if d < 1e-10:
        return np.zeros((len(atoms), 3))
    unit = vec / d
    grad = np.zeros((len(atoms), 3))
    grad[i] = -unit   # dr/dr_i: atom i moves, distance changes
    grad[j] = unit    # dr/dr_j: atom j moves, distance changes oppositely
    return grad


# ============================================================
# 5. RUN METADYNAMICS
# ============================================================

# Initialize velocities
MaxwellBoltzmannDistribution(atoms, temperature_K=TEMPERATURE)
Stationary(atoms)

# Langevin thermostat
dyn = Langevin(atoms, timestep=TIMESTEP * units.fs,
               temperature_K=TEMPERATURE, friction=FRICTION)

# Create metadynamics engine
metad = WellTemperedMetadynamics(
    height=HILL_HEIGHT_INIT,
    sigma=HILL_SIGMA,
    bias_factor=BIAS_FACTOR,
    temperature=TEMPERATURE,
    cv_func=cv_distance,
    cv_grad_func=cv_distance_gradient,
)

# Trajectory
traj = Trajectory(os.path.join(OUTPUT_DIR, "metad.traj"), "w", atoms)

# Storage for analysis
cv_values = []
bias_energies = []
step_log = []
hill_heights = []

print(f"\nStarting well-tempered metadynamics: {N_STEPS} steps, T={TEMPERATURE} K")
print(f"  Hill: w0={HILL_HEIGHT_INIT} eV, sigma={HILL_SIGMA} A, gamma={BIAS_FACTOR}")
print(f"  Deposit every {HILL_EVERY} steps")
print("-" * 70)

for step in range(N_STEPS):
    # Get unbiased forces from MACE
    forces_unbiased = atoms.get_forces()

    # Get bias forces from metadynamics
    bias_forces, s = metad.get_bias_forces(atoms)
    V_bias = metad.get_bias_energy(s)

    # Apply combined forces (override calculator forces)
    atoms.arrays["forces_orig"] = forces_unbiased.copy()
    total_forces = forces_unbiased + bias_forces

    # Manually set forces for the dynamics step
    atoms.calc.results["forces"] = total_forces

    # Deposit hill if scheduled
    if step > 0 and step % HILL_EVERY == 0:
        w = metad.deposit_hill(s)
        hill_heights.append(w)

    # Record
    cv_values.append(s)
    bias_energies.append(V_bias)
    step_log.append(step)

    # Dynamics step
    dyn.run(1)

    # Save trajectory periodically
    if step % 200 == 0:
        traj.write(atoms)

    # Progress
    if step % 2000 == 0:
        n_hills = len(metad.hills)
        print(f"  Step {step:6d}/{N_STEPS} | CV={s:.3f} A | "
              f"V_bias={V_bias:.4f} eV | hills={n_hills}")

traj.close()
print(f"\nMetadynamics complete. {len(metad.hills)} hills deposited.")

# ============================================================
# 6. RECONSTRUCT FREE ENERGY SURFACE
# ============================================================

cv_arr = np.array(cv_values)
s_min, s_max = cv_arr.min() - 0.5, cv_arr.max() + 0.5
s_grid, fes = metad.reconstruct_fes(s_min, s_max, n_bins=300)

# Save data
np.savetxt(os.path.join(OUTPUT_DIR, "fes.dat"),
           np.column_stack([s_grid, fes]),
           header="CV_distance(A)  FreeEnergy(eV)")
np.savetxt(os.path.join(OUTPUT_DIR, "cv_trajectory.dat"),
           np.column_stack([step_log, cv_values, bias_energies]),
           header="Step  CV_distance(A)  V_bias(eV)")
np.savetxt(os.path.join(OUTPUT_DIR, "hills.dat"),
           np.array(metad.hills),
           header="CV_center(A)  Height(eV)")

# ============================================================
# 7. PLOTTING
# ============================================================

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# (a) Free energy surface
ax = axes[0, 0]
ax.plot(s_grid, fes, "b-", linewidth=2)
ax.set_xlabel("Distance CV (A)", fontsize=12)
ax.set_ylabel("Free Energy (eV)", fontsize=12)
ax.set_title("Free Energy Surface", fontsize=13)
ax.grid(True, alpha=0.3)

# Mark minima
from scipy.signal import argrelmin
minima_idx = argrelmin(fes, order=10)[0]
for idx in minima_idx:
    ax.axvline(s_grid[idx], color="red", linestyle="--", alpha=0.5)
    ax.annotate(f"{fes[idx]:.3f} eV", (s_grid[idx], fes[idx]),
                textcoords="offset points", xytext=(5, 10), fontsize=9)

# (b) CV trajectory
ax = axes[0, 1]
time_ps = np.array(step_log) * TIMESTEP / 1000.0
ax.plot(time_ps, cv_values, "k-", linewidth=0.3, alpha=0.6)
ax.set_xlabel("Time (ps)", fontsize=12)
ax.set_ylabel("Distance CV (A)", fontsize=12)
ax.set_title("CV Trajectory", fontsize=13)
ax.grid(True, alpha=0.3)

# (c) Bias energy over time
ax = axes[1, 0]
ax.plot(time_ps, bias_energies, "r-", linewidth=0.5)
ax.set_xlabel("Time (ps)", fontsize=12)
ax.set_ylabel("V_bias (eV)", fontsize=12)
ax.set_title("Accumulated Bias Energy", fontsize=13)
ax.grid(True, alpha=0.3)

# (d) Hill heights over time (convergence check)
ax = axes[1, 1]
hill_steps = np.arange(1, len(hill_heights) + 1) * HILL_EVERY * TIMESTEP / 1000.0
ax.semilogy(hill_steps, hill_heights, "go-", markersize=2, linewidth=0.5)
ax.set_xlabel("Time (ps)", fontsize=12)
ax.set_ylabel("Hill Height (eV)", fontsize=12)
ax.set_title("Hill Height Decay (well-tempered)", fontsize=13)
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "metad_distance_results.png"), dpi=200, bbox_inches="tight")
print(f"Plots saved: {OUTPUT_DIR}/metad_distance_results.png")

# ── Report ──
print(f"\n{'='*60}")
print("FREE ENERGY SURFACE SUMMARY")
print(f"{'='*60}")
print(f"CV range explored: {cv_arr.min():.3f} -- {cv_arr.max():.3f} A")
print(f"Number of hills deposited: {len(metad.hills)}")
if len(hill_heights) > 10:
    print(f"Final hill height: {hill_heights[-1]:.6f} eV "
          f"(initial: {HILL_HEIGHT_INIT} eV, ratio: {hill_heights[-1]/HILL_HEIGHT_INIT:.4f})")
if len(minima_idx) >= 2:
    barrier = np.max(fes[minima_idx[0]:minima_idx[1]+1]) - fes[minima_idx[0]]
    print(f"Barrier between first two minima: {barrier:.4f} eV ({barrier*1000:.1f} meV)")
print(f"FES data: {OUTPUT_DIR}/fes.dat")
```

#### Script A2: Coordination Number CV -- Phase Transition

```python
#!/usr/bin/env python3
"""
Well-tempered metadynamics with coordination number CV.
Example: BCC-to-FCC phase transition in iron at high pressure.

Coordination number (CN) is a smooth CV that distinguishes crystal structures:
  BCC: CN ~ 8 (first shell), FCC: CN ~ 12 (first shell)

Uses a continuous switching function:
  CN_i = sum_j  1/(1 + (r_ij/r0)^n)  for j != i

Standalone -- no pyiron dependency.
"""

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

from ase.build import bulk
from ase.optimize import LBFGS
from ase.constraints import ExpCellFilter
from ase.md.langevin import Langevin
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution, Stationary
from ase.io import Trajectory
from ase import units

from mace.calculators import mace_mp

# ============================================================
# 1. CONFIGURATION
# ============================================================

MACE_MODEL = "medium"
DEVICE = "cpu"
ELEMENT = "Fe"
STRUCTURE = "bcc"                      # starting phase
A_LATTICE = 2.87                       # Fe BCC lattice constant (A)
SUPERCELL = (3, 3, 3)                  # 3x3x3 BCC -> 54 atoms
TEMPERATURE = 1000.0                   # K
TIMESTEP = 1.0                         # fs (shorter for metals at high T)
N_STEPS = 30000
HILL_EVERY = 100
HILL_HEIGHT_INIT = 0.03                # eV
HILL_SIGMA = 0.5                       # width in CN space
BIAS_FACTOR = 15.0                     # gamma
FRICTION = 0.005

# Coordination number parameters
CN_R0 = 3.0                            # switching function midpoint (A)
CN_NN = 6                              # exponent n in switching function
CN_MM = 12                             # exponent m in denominator (steeper cutoff)
CN_CUTOFF = 5.0                        # ignore pairs beyond this distance (A)
# Which atom's CN to bias (average over all for collective transition)
CN_MODE = "average"                    # "average" or "single"
CN_ATOM = 0                            # used if CN_MODE == "single"

OUTPUT_DIR = "metad_coordination"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ============================================================
# 2. BUILD SYSTEM
# ============================================================

atoms = bulk(ELEMENT, STRUCTURE, a=A_LATTICE, cubic=True) * SUPERCELL
print(f"Built {len(atoms)}-atom {ELEMENT} {STRUCTURE} supercell")

calc = mace_mp(model=MACE_MODEL, dispersion=False, device=DEVICE, default_dtype="float64")
atoms.calc = calc

# Relax
ecf = ExpCellFilter(atoms)
opt = LBFGS(ecf, logfile=os.path.join(OUTPUT_DIR, "relax.log"))
opt.run(fmax=0.02, steps=200)
print(f"Relaxed energy: {atoms.get_potential_energy():.4f} eV")

# ============================================================
# 3. COORDINATION NUMBER CV
# ============================================================

def switching_function(r, r0, n=6, m=12):
    """
    Continuous switching function for coordination number.
    s(r) = (1 - (r/r0)^n) / (1 - (r/r0)^m)
    Approaches 1 for r << r0 and 0 for r >> r0.
    """
    x = r / r0
    # Numerical stability for x close to 1
    xn = np.power(x, n)
    xm = np.power(x, m)
    num = 1.0 - xn
    den = 1.0 - xm
    # Avoid division by zero
    mask = np.abs(den) < 1e-12
    result = np.where(mask, n / (m * x), num / den)
    return np.clip(result, 0.0, 1.0)


def switching_function_deriv(r, r0, n=6, m=12):
    """
    Derivative ds/dr of the switching function.
    """
    x = r / r0
    xn = np.power(x, n)
    xm = np.power(x, m)
    num = 1.0 - xn
    den = 1.0 - xm
    mask = np.abs(den) < 1e-12
    # d/dr [(1-x^n)/(1-x^m)] via quotient rule
    dnum = -n * np.power(x, n - 1) / r0
    dden = -m * np.power(x, m - 1) / r0
    deriv = np.where(
        mask,
        np.zeros_like(r),
        (dnum * den - num * dden) / (den**2 + 1e-30)
    )
    return deriv


def compute_cn_average(atoms, r0=CN_R0, n=CN_NN, m=CN_MM, cutoff=CN_CUTOFF):
    """Compute average coordination number over all atoms."""
    from ase.neighborlist import neighbor_list
    i_list, j_list, d_list = neighbor_list("ijd", atoms, cutoff)
    N = len(atoms)
    cn = np.zeros(N)
    for idx in range(len(i_list)):
        cn[i_list[idx]] += switching_function(d_list[idx], r0, n, m)
    return np.mean(cn)


def compute_cn_gradient(atoms, r0=CN_R0, n=CN_NN, m=CN_MM, cutoff=CN_CUTOFF):
    """
    Gradient of average CN w.r.t. all atomic positions.
    For average CN: d<CN>/dr_k = (1/N) * sum over pairs involving k.
    """
    from ase.neighborlist import neighbor_list
    i_list, j_list, d_list, D_list = neighbor_list("ijdD", atoms, cutoff)
    N = len(atoms)
    grad = np.zeros((N, 3))

    for idx in range(len(i_list)):
        i, j = i_list[idx], j_list[idx]
        d = d_list[idx]
        D = D_list[idx]  # vector from i to j
        if d < 1e-10:
            continue
        ds_dr = switching_function_deriv(d, r0, n, m)
        # ds/dr_i = ds/d|r_ij| * d|r_ij|/dr_i = ds/d|r_ij| * (-D/d)
        direction = D / d
        grad[i] -= ds_dr * direction / N
        grad[j] += ds_dr * direction / N

    return grad


def cv_coordination(atoms):
    """CV function: average coordination number."""
    return compute_cn_average(atoms)


def cv_coordination_gradient(atoms):
    """Gradient of CV w.r.t. atomic positions."""
    return compute_cn_gradient(atoms)


# ============================================================
# 4. METADYNAMICS ENGINE (reuse from Script A1)
# ============================================================

class WellTemperedMetadynamics:
    """Well-tempered metadynamics on a 1D collective variable."""

    def __init__(self, height, sigma, bias_factor, temperature, cv_func, cv_grad_func):
        self.w0 = height
        self.sigma = sigma
        self.gamma = bias_factor
        self.kT = temperature * units.kB
        self.delta_T_kT = (bias_factor - 1.0) * self.kT
        self.cv_func = cv_func
        self.cv_grad_func = cv_grad_func
        self.hills = []
        self.cv_history = []
        self.bias_history = []

    def get_bias_energy(self, s):
        V = 0.0
        for s_i, w_i in self.hills:
            V += w_i * np.exp(-0.5 * ((s - s_i) / self.sigma) ** 2)
        return V

    def get_bias_force_on_cv(self, s):
        dVds = 0.0
        for s_i, w_i in self.hills:
            g = w_i * np.exp(-0.5 * ((s - s_i) / self.sigma) ** 2)
            dVds += g * (-(s - s_i) / self.sigma**2)
        return -dVds

    def deposit_hill(self, s):
        V_current = self.get_bias_energy(s)
        w = self.w0 * np.exp(-V_current / self.delta_T_kT)
        self.hills.append((s, w))
        return w

    def get_bias_forces(self, atoms):
        s = self.cv_func(atoms)
        f_cv = self.get_bias_force_on_cv(s)
        grad_cv = self.cv_grad_func(atoms)
        F_bias = f_cv * grad_cv
        return F_bias, s

    def reconstruct_fes(self, s_min, s_max, n_bins=200):
        s_grid = np.linspace(s_min, s_max, n_bins)
        V_bias = np.array([self.get_bias_energy(s) for s in s_grid])
        F = -(self.gamma / (self.gamma - 1.0)) * V_bias
        F -= np.min(F)
        return s_grid, F


# ============================================================
# 5. RUN METADYNAMICS
# ============================================================

MaxwellBoltzmannDistribution(atoms, temperature_K=TEMPERATURE)
Stationary(atoms)

dyn = Langevin(atoms, timestep=TIMESTEP * units.fs,
               temperature_K=TEMPERATURE, friction=FRICTION)

metad = WellTemperedMetadynamics(
    height=HILL_HEIGHT_INIT,
    sigma=HILL_SIGMA,
    bias_factor=BIAS_FACTOR,
    temperature=TEMPERATURE,
    cv_func=cv_coordination,
    cv_grad_func=cv_coordination_gradient,
)

traj = Trajectory(os.path.join(OUTPUT_DIR, "metad_cn.traj"), "w", atoms)

cv_values = []
bias_energies = []
step_log = []
hill_heights = []

# Measure initial CN
cn_init = cv_coordination(atoms)
print(f"\nInitial average CN: {cn_init:.2f}")
print(f"Expected: BCC ~ 8, FCC ~ 12, HCP ~ 12, liquid ~ 10-14")
print(f"\nStarting well-tempered metadynamics with CN CV...")
print("-" * 70)

for step in range(N_STEPS):
    forces_unbiased = atoms.get_forces()
    bias_forces, s = metad.get_bias_forces(atoms)
    V_bias = metad.get_bias_energy(s)

    total_forces = forces_unbiased + bias_forces
    atoms.calc.results["forces"] = total_forces

    if step > 0 and step % HILL_EVERY == 0:
        w = metad.deposit_hill(s)
        hill_heights.append(w)

    cv_values.append(s)
    bias_energies.append(V_bias)
    step_log.append(step)

    dyn.run(1)

    if step % 500 == 0:
        traj.write(atoms)

    if step % 3000 == 0:
        print(f"  Step {step:6d}/{N_STEPS} | CN={s:.2f} | "
              f"V_bias={V_bias:.4f} eV | hills={len(metad.hills)}")

traj.close()
print(f"\nMetadynamics complete. {len(metad.hills)} hills deposited.")

# ============================================================
# 6. RECONSTRUCT FES AND PLOT
# ============================================================

cv_arr = np.array(cv_values)
s_grid, fes = metad.reconstruct_fes(cv_arr.min() - 1.0, cv_arr.max() + 1.0, n_bins=300)

np.savetxt(os.path.join(OUTPUT_DIR, "fes_cn.dat"),
           np.column_stack([s_grid, fes]),
           header="CoordinationNumber  FreeEnergy(eV)")
np.savetxt(os.path.join(OUTPUT_DIR, "cv_cn_trajectory.dat"),
           np.column_stack([step_log, cv_values, bias_energies]),
           header="Step  CN  V_bias(eV)")

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# (a) FES
ax = axes[0, 0]
ax.plot(s_grid, fes, "b-", linewidth=2)
ax.axvline(8.0, color="orange", linestyle="--", alpha=0.7, label="BCC CN ~ 8")
ax.axvline(12.0, color="green", linestyle="--", alpha=0.7, label="FCC CN ~ 12")
ax.set_xlabel("Average Coordination Number", fontsize=12)
ax.set_ylabel("Free Energy (eV/atom)", fontsize=12)
ax.set_title("Free Energy Surface: BCC-FCC Transition", fontsize=13)
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3)

# (b) CN trajectory
ax = axes[0, 1]
time_ps = np.array(step_log) * TIMESTEP / 1000.0
ax.plot(time_ps, cv_values, "k-", linewidth=0.3, alpha=0.6)
ax.axhline(8.0, color="orange", linestyle="--", alpha=0.5)
ax.axhline(12.0, color="green", linestyle="--", alpha=0.5)
ax.set_xlabel("Time (ps)", fontsize=12)
ax.set_ylabel("Average CN", fontsize=12)
ax.set_title("CN Trajectory", fontsize=13)
ax.grid(True, alpha=0.3)

# (c) Bias energy
ax = axes[1, 0]
ax.plot(time_ps, bias_energies, "r-", linewidth=0.5)
ax.set_xlabel("Time (ps)", fontsize=12)
ax.set_ylabel("V_bias (eV)", fontsize=12)
ax.set_title("Accumulated Bias", fontsize=13)
ax.grid(True, alpha=0.3)

# (d) Hill height decay
ax = axes[1, 1]
if hill_heights:
    hill_times = np.arange(1, len(hill_heights)+1) * HILL_EVERY * TIMESTEP / 1000.0
    ax.semilogy(hill_times, hill_heights, "go-", markersize=2, linewidth=0.5)
ax.set_xlabel("Time (ps)", fontsize=12)
ax.set_ylabel("Hill Height (eV)", fontsize=12)
ax.set_title("Hill Height Decay", fontsize=13)
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "metad_cn_results.png"), dpi=200, bbox_inches="tight")
print(f"Plots saved: {OUTPUT_DIR}/metad_cn_results.png")

# ── Summary ──
print(f"\n{'='*60}")
print("METADYNAMICS SUMMARY (Coordination Number CV)")
print(f"{'='*60}")
print(f"Starting structure: {ELEMENT} {STRUCTURE}")
print(f"CN range explored: {cv_arr.min():.2f} -- {cv_arr.max():.2f}")
print(f"Hills deposited: {len(metad.hills)}")
if hill_heights:
    print(f"Hill height: {HILL_HEIGHT_INIT:.4f} -> {hill_heights[-1]:.6f} eV")
```

#### Script A3: Volume CV -- Pressure-Induced Phase Transition

```python
#!/usr/bin/env python3
"""
Well-tempered metadynamics with cell volume as collective variable.
Example: pressure-induced phase transition.

Volume CV is useful for transitions that involve significant density changes
(e.g., graphite -> diamond, BCC -> HCP under pressure).

This uses NPT dynamics where the volume is a natural dynamical variable,
and the bias is deposited directly in volume space.

Standalone -- no pyiron dependency.
"""

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

from ase.build import bulk
from ase.optimize import LBFGS
from ase.constraints import ExpCellFilter
from ase.md.nptberendsen import NPTBerendsen
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution, Stationary
from ase.io import Trajectory
from ase import units

from mace.calculators import mace_mp

# ============================================================
# 1. CONFIGURATION
# ============================================================

MACE_MODEL = "medium"
DEVICE = "cpu"
ELEMENT = "Si"
STRUCTURE = "diamond"
A_LATTICE = 5.43
SUPERCELL = (2, 2, 2)                  # 2x2x2 diamond = 64 atoms
TEMPERATURE = 800.0                    # K
PRESSURE = 10.0                        # GPa -- high pressure to drive transition
TIMESTEP = 2.0                         # fs
N_STEPS = 25000
HILL_EVERY = 80
HILL_HEIGHT_INIT = 0.10                # eV (larger for volume CV -- broader landscape)
HILL_SIGMA = 20.0                      # A^3 (volume units)
BIAS_FACTOR = 12.0

OUTPUT_DIR = "metad_volume"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ============================================================
# 2. BUILD AND RELAX
# ============================================================

atoms = bulk(ELEMENT, STRUCTURE, a=A_LATTICE) * SUPERCELL
print(f"Built {len(atoms)}-atom {ELEMENT} {STRUCTURE} supercell")

calc = mace_mp(model=MACE_MODEL, dispersion=False, device=DEVICE, default_dtype="float64")
atoms.calc = calc

ecf = ExpCellFilter(atoms, scalar_pressure=PRESSURE * units.GPa)
opt = LBFGS(ecf, logfile=os.path.join(OUTPUT_DIR, "relax.log"))
opt.run(fmax=0.05, steps=200)
V0 = atoms.get_volume()
print(f"Relaxed volume at {PRESSURE} GPa: {V0:.2f} A^3 ({V0/len(atoms):.3f} A^3/atom)")

# ============================================================
# 3. VOLUME CV
# ============================================================

def cv_volume(atoms):
    """CV: total cell volume in A^3."""
    return atoms.get_volume()


def cv_volume_gradient(atoms):
    """
    Gradient of volume w.r.t. atomic positions.

    For a periodic cell, volume depends on cell vectors, not directly on
    atomic positions in fractional coordinates. However, in ASE NPT dynamics,
    the cell vectors are dynamical variables. We couple the bias through
    a virtual stress rather than atomic forces.

    For simplicity in this pure-Python implementation, we apply the bias
    as an isotropic pressure correction: dV/dV = 1, and the force is
    distributed as a stress tensor correction to the NPT barostat.

    Returns a scalar (pressure correction in eV/A^3) rather than per-atom forces.
    """
    # Not used directly -- we apply bias via stress correction (see run loop)
    return None


# ============================================================
# 4. METADYNAMICS ENGINE (volume-adapted)
# ============================================================

class VolumeMetadynamics:
    """
    Well-tempered metadynamics for cell volume CV.
    Instead of atomic forces, the bias produces a pressure correction:
        P_bias = -dV_bias/dVolume
    which modifies the effective external pressure felt by the barostat.
    """

    def __init__(self, height, sigma, bias_factor, temperature):
        self.w0 = height
        self.sigma = sigma
        self.gamma = bias_factor
        self.kT = temperature * units.kB
        self.delta_T_kT = (bias_factor - 1.0) * self.kT
        self.hills = []

    def get_bias_energy(self, V):
        E = 0.0
        for V_i, w_i in self.hills:
            E += w_i * np.exp(-0.5 * ((V - V_i) / self.sigma) ** 2)
        return E

    def get_bias_pressure(self, V):
        """
        Compute bias pressure P_bias = -dV_bias/dVolume.
        This acts as an additional pressure term in the NPT barostat.
        Returned in units of eV/A^3.
        """
        dE_dV = 0.0
        for V_i, w_i in self.hills:
            g = w_i * np.exp(-0.5 * ((V - V_i) / self.sigma) ** 2)
            dE_dV += g * (-(V - V_i) / self.sigma**2)
        return -dE_dV  # P_bias = -dE/dV

    def deposit_hill(self, V):
        E_current = self.get_bias_energy(V)
        w = self.w0 * np.exp(-E_current / self.delta_T_kT)
        self.hills.append((V, w))
        return w

    def reconstruct_fes(self, V_min, V_max, n_bins=300):
        V_grid = np.linspace(V_min, V_max, n_bins)
        E_bias = np.array([self.get_bias_energy(V) for V in V_grid])
        F = -(self.gamma / (self.gamma - 1.0)) * E_bias
        F -= np.min(F)
        return V_grid, F


# ============================================================
# 5. RUN METADYNAMICS WITH NPT
# ============================================================

MaxwellBoltzmannDistribution(atoms, temperature_K=TEMPERATURE)
Stationary(atoms)

# NPT Berendsen barostat
dyn = NPTBerendsen(
    atoms,
    timestep=TIMESTEP * units.fs,
    temperature_K=TEMPERATURE,
    pressure_au=PRESSURE * units.GPa,
    taut=100 * units.fs,
    taup=500 * units.fs,
    compressibility_au=4.57e-5 / units.GPa,  # typical for metals/semiconductors
)

metad = VolumeMetadynamics(
    height=HILL_HEIGHT_INIT,
    sigma=HILL_SIGMA,
    bias_factor=BIAS_FACTOR,
    temperature=TEMPERATURE,
)

traj = Trajectory(os.path.join(OUTPUT_DIR, "metad_vol.traj"), "w", atoms)

vol_values = []
bias_energies = []
step_log = []
hill_heights = []
pressure_log = []

print(f"\nStarting volume-CV metadynamics at T={TEMPERATURE} K, P={PRESSURE} GPa")
print(f"  Initial volume: {V0:.2f} A^3")
print("-" * 70)

for step in range(N_STEPS):
    V = atoms.get_volume()

    # Compute bias pressure and add to stress tensor
    P_bias = metad.get_bias_pressure(V)
    V_bias = metad.get_bias_energy(V)

    # Modify the external pressure for this step
    # The barostat sees P_external + P_bias
    effective_pressure = PRESSURE * units.GPa + P_bias
    dyn.pressure = effective_pressure

    if step > 0 and step % HILL_EVERY == 0:
        w = metad.deposit_hill(V)
        hill_heights.append(w)

    vol_values.append(V)
    bias_energies.append(V_bias)
    step_log.append(step)
    pressure_log.append(P_bias / units.GPa)

    dyn.run(1)

    if step % 500 == 0:
        traj.write(atoms)

    if step % 3000 == 0:
        print(f"  Step {step:6d}/{N_STEPS} | V={V:.1f} A^3 "
              f"({V/len(atoms):.3f} A^3/at) | V_bias={V_bias:.4f} eV | "
              f"P_bias={P_bias/units.GPa:.3f} GPa")

traj.close()
print(f"\nComplete. {len(metad.hills)} hills deposited.")

# ============================================================
# 6. ANALYSIS AND PLOTTING
# ============================================================

vol_arr = np.array(vol_values)
V_min, V_max = vol_arr.min() - 30, vol_arr.max() + 30
V_grid, fes = metad.reconstruct_fes(V_min, V_max, n_bins=400)

np.savetxt(os.path.join(OUTPUT_DIR, "fes_volume.dat"),
           np.column_stack([V_grid, V_grid / len(atoms), fes]),
           header="Volume(A^3)  Volume_per_atom(A^3)  FreeEnergy(eV)")
np.savetxt(os.path.join(OUTPUT_DIR, "cv_volume_trajectory.dat"),
           np.column_stack([step_log, vol_values, bias_energies]),
           header="Step  Volume(A^3)  V_bias(eV)")

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# (a) FES vs volume per atom
ax = axes[0, 0]
n_at = len(atoms)
ax.plot(V_grid / n_at, fes, "b-", linewidth=2)
ax.set_xlabel("Volume per atom ($\\AA^3$)", fontsize=12)
ax.set_ylabel("Free Energy (eV)", fontsize=12)
ax.set_title(f"FES at T={TEMPERATURE} K, P={PRESSURE} GPa", fontsize=13)
ax.grid(True, alpha=0.3)

# (b) Volume trajectory
ax = axes[0, 1]
time_ps = np.array(step_log) * TIMESTEP / 1000.0
ax.plot(time_ps, np.array(vol_values) / n_at, "k-", linewidth=0.3, alpha=0.6)
ax.set_xlabel("Time (ps)", fontsize=12)
ax.set_ylabel("Volume per atom ($\\AA^3$)", fontsize=12)
ax.set_title("Volume Trajectory", fontsize=13)
ax.grid(True, alpha=0.3)

# (c) Bias pressure
ax = axes[1, 0]
ax.plot(time_ps, pressure_log, "r-", linewidth=0.5)
ax.set_xlabel("Time (ps)", fontsize=12)
ax.set_ylabel("Bias Pressure (GPa)", fontsize=12)
ax.set_title("Bias Pressure Correction", fontsize=13)
ax.grid(True, alpha=0.3)

# (d) Hill heights
ax = axes[1, 1]
if hill_heights:
    ht = np.arange(1, len(hill_heights)+1) * HILL_EVERY * TIMESTEP / 1000.0
    ax.semilogy(ht, hill_heights, "go-", markersize=2, linewidth=0.5)
ax.set_xlabel("Time (ps)", fontsize=12)
ax.set_ylabel("Hill Height (eV)", fontsize=12)
ax.set_title("Hill Height Decay", fontsize=13)
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "metad_volume_results.png"), dpi=200, bbox_inches="tight")
print(f"Plots saved: {OUTPUT_DIR}/metad_volume_results.png")
```

---

### Method B: LAMMPS + PLUMED Metadynamics

PLUMED is the standard engine for enhanced sampling in molecular simulations. When compiled as a LAMMPS plugin, it provides a vast library of collective variables, bias methods, and analysis tools. This method is recommended for large systems and production runs.

#### Script B1: LAMMPS + PLUMED Well-Tempered Metadynamics

```python
#!/usr/bin/env python3
"""
LAMMPS + PLUMED well-tempered metadynamics.
Example: diffusion of a vacancy in copper FCC.

Generates:
  1. LAMMPS input script with fix plumed
  2. PLUMED input file defining CVs and metadynamics parameters
  3. Runs LAMMPS
  4. Post-processes HILLS and COLVAR files to reconstruct FES

Prerequisites:
  - lmp binary compiled with PLUMED package
  - EAM potential file for Cu (or download from NIST)
  - plumed CLI tool (for sum_hills post-processing)

Standalone -- no pyiron dependency.
"""

import os
import subprocess
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ============================================================
# 1. CONFIGURATION
# ============================================================

LAMMPS_BIN = "lmp"                     # LAMMPS binary with PLUMED support
PLUMED_BIN = "plumed"                  # PLUMED command-line tool
POTENTIAL_FILE = "Cu_mishin1.eam.alloy" # EAM potential for Cu
NPROC = 4                              # MPI processes

TEMPERATURE = 600.0                    # K
PRESSURE = 0.0                         # bar
TIMESTEP = 2.0                         # fs
N_STEPS = 200000                       # total steps
DUMP_EVERY = 1000                      # trajectory output interval

# Metadynamics parameters (set in PLUMED input)
HILL_HEIGHT = 0.5                      # kJ/mol (PLUMED uses kJ/mol)
HILL_SIGMA = 0.02                      # nm (PLUMED uses nm)
PACE = 500                             # deposit hill every N steps
BIAS_FACTOR = 15.0                     # well-tempered gamma
GRID_MIN = 0.0                         # nm
GRID_MAX = 1.5                         # nm
GRID_BIN = 300

OUTPUT_DIR = "metad_lammps_plumed"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ============================================================
# 2. BUILD Cu FCC WITH VACANCY (LAMMPS data file)
# ============================================================

from ase.build import bulk
from ase.io import write as ase_write

atoms = bulk("Cu", "fcc", a=3.615, cubic=True) * (4, 4, 4)  # 256 atoms
# Remove atom to create vacancy
vacancy_idx = 32
vacancy_pos = atoms.positions[vacancy_idx].copy()
del atoms[vacancy_idx]

# Find migrating atom (nearest neighbor to vacancy)
dists = np.linalg.norm(atoms.positions - vacancy_pos, axis=1)
migrant_idx = int(np.argmin(dists)) + 1  # LAMMPS 1-indexed
reference_idx = int(np.argsort(dists)[5]) + 1  # a non-neighbor reference atom

print(f"Created {len(atoms)}-atom Cu FCC supercell with vacancy")
print(f"Migrant atom (LAMMPS id): {migrant_idx}")
print(f"Reference atom (LAMMPS id): {reference_idx}")

# Write LAMMPS data file
ase_write(os.path.join(OUTPUT_DIR, "cu_vacancy.data"), atoms, format="lammps-data")

# ============================================================
# 3. WRITE PLUMED INPUT
# ============================================================

plumed_input = f"""# PLUMED input for well-tempered metadynamics
# CV: distance between migrating atom and a reference atom
UNITS LENGTH=A ENERGY=kj/mol TIME=fs

# Define the distance CV between migrant and reference atoms
d1: DISTANCE ATOMS={migrant_idx},{reference_idx} NOPBC

# Well-tempered metadynamics
metad: METAD ...
  ARG=d1
  SIGMA={HILL_SIGMA * 10.0:.4f}
  HEIGHT={HILL_HEIGHT:.4f}
  PACE={PACE}
  BIASFACTOR={BIAS_FACTOR:.1f}
  TEMP={TEMPERATURE:.1f}
  GRID_MIN={GRID_MIN * 10.0:.2f}
  GRID_MAX={GRID_MAX * 10.0:.2f}
  GRID_BIN={GRID_BIN}
  FILE=HILLS
  CALC_RDB
...

# Print CV and bias to COLVAR file
PRINT ARG=d1,metad.bias STRIDE=100 FILE=COLVAR
"""

with open(os.path.join(OUTPUT_DIR, "plumed.dat"), "w") as f:
    f.write(plumed_input)
print("PLUMED input written: plumed.dat")

# ============================================================
# 4. WRITE LAMMPS INPUT SCRIPT
# ============================================================

lammps_input = f"""# LAMMPS input for metadynamics with PLUMED
# Cu FCC vacancy diffusion

units           metal
atom_style      atomic
boundary        p p p

read_data       cu_vacancy.data

# Interatomic potential
pair_style      eam/alloy
pair_coeff      * * {POTENTIAL_FILE} Cu

# Neighbor list
neighbor        2.0 bin
neigh_modify    every 1 delay 0 check yes

# Minimize first
minimize        1.0e-6 1.0e-8 1000 10000
reset_timestep  0

# Thermostat: NVT
velocity        all create {TEMPERATURE} 12345 dist gaussian
fix             1 all nvt temp {TEMPERATURE} {TEMPERATURE} $(100.0*dt)

# PLUMED fix -- this activates metadynamics
fix             plumed all plumed plumedfile plumed.dat outfile plumed.out

# Timestep
timestep        {TIMESTEP / 1000.0:.6f}

# Output
thermo          1000
thermo_style    custom step temp pe ke etotal press vol

dump            1 all custom {DUMP_EVERY} traj.lammpstrj id type x y z

# Run
run             {N_STEPS}

# Cleanup
unfix           1
unfix           plumed
"""

with open(os.path.join(OUTPUT_DIR, "in.metad"), "w") as f:
    f.write(lammps_input)
print("LAMMPS input written: in.metad")

# ============================================================
# 5. RUN LAMMPS
# ============================================================

print(f"\nRunning LAMMPS + PLUMED metadynamics ({N_STEPS} steps)...")
result = subprocess.run(
    ["mpirun", "-np", str(NPROC), LAMMPS_BIN, "-in", "in.metad"],
    capture_output=True, text=True, cwd=OUTPUT_DIR, timeout=3600,
)

with open(os.path.join(OUTPUT_DIR, "lammps.out"), "w") as f:
    f.write(result.stdout)
with open(os.path.join(OUTPUT_DIR, "lammps.err"), "w") as f:
    f.write(result.stderr)

if result.returncode == 0:
    print("LAMMPS completed successfully.")
else:
    print(f"LAMMPS exited with code {result.returncode}")
    print("Check lammps.out and lammps.err for details.")
    print(result.stderr[-500:] if result.stderr else "")

# ============================================================
# 6. RECONSTRUCT FES WITH PLUMED sum_hills
# ============================================================

print("\nReconstructing FES with plumed sum_hills...")
result_sh = subprocess.run(
    [PLUMED_BIN, "sum_hills", "--hills", "HILLS",
     "--outfile", "fes.dat", "--mintozero", "--kt", f"{TEMPERATURE * 8.314e-3:.4f}"],
    capture_output=True, text=True, cwd=OUTPUT_DIR, timeout=120,
)

if result_sh.returncode == 0:
    print("FES reconstruction complete: fes.dat")
else:
    print("sum_hills failed; attempting manual reconstruction from HILLS file...")

# ============================================================
# 7. PARSE AND PLOT RESULTS
# ============================================================

def parse_plumed_file(filepath, comment="#"):
    """Parse a PLUMED output file (COLVAR, HILLS, fes.dat)."""
    data = []
    headers = []
    with open(filepath, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if line.startswith(comment):
                if "FIELDS" in line:
                    headers = line.split()[2:]  # skip "#! FIELDS"
                continue
            data.append([float(x) for x in line.split()])
    return np.array(data), headers


# Parse COLVAR
colvar_path = os.path.join(OUTPUT_DIR, "COLVAR")
if os.path.exists(colvar_path):
    colvar, colvar_headers = parse_plumed_file(colvar_path)
    print(f"COLVAR columns: {colvar_headers}")
    print(f"COLVAR shape: {colvar.shape}")
else:
    print("WARNING: COLVAR file not found. LAMMPS+PLUMED may have failed.")
    colvar = None

# Parse FES
fes_path = os.path.join(OUTPUT_DIR, "fes.dat")
if os.path.exists(fes_path):
    fes_data, fes_headers = parse_plumed_file(fes_path)
else:
    fes_data = None

# Parse HILLS
hills_path = os.path.join(OUTPUT_DIR, "HILLS")
if os.path.exists(hills_path):
    hills_data, hills_headers = parse_plumed_file(hills_path)
else:
    hills_data = None

# ── Plot ────────────────────────────────────────────────────────
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# (a) Free energy surface
ax = axes[0, 0]
if fes_data is not None and fes_data.shape[1] >= 2:
    ax.plot(fes_data[:, 0], fes_data[:, 1], "b-", linewidth=2)
    ax.set_xlabel("Distance CV (A)", fontsize=12)
    ax.set_ylabel("Free Energy (kJ/mol)", fontsize=12)
    ax.set_title("Free Energy Surface (PLUMED sum_hills)", fontsize=13)
    ax.grid(True, alpha=0.3)
else:
    ax.text(0.5, 0.5, "FES data not available", transform=ax.transAxes,
            ha="center", fontsize=12)

# (b) CV trajectory
ax = axes[0, 1]
if colvar is not None and colvar.shape[1] >= 2:
    time_ps = colvar[:, 0] / 1000.0  # fs -> ps
    ax.plot(time_ps, colvar[:, 1], "k-", linewidth=0.3, alpha=0.6)
    ax.set_xlabel("Time (ps)", fontsize=12)
    ax.set_ylabel("Distance CV (A)", fontsize=12)
    ax.set_title("CV Trajectory", fontsize=13)
    ax.grid(True, alpha=0.3)

# (c) Bias over time
ax = axes[1, 0]
if colvar is not None and colvar.shape[1] >= 3:
    ax.plot(time_ps, colvar[:, 2], "r-", linewidth=0.5)
    ax.set_xlabel("Time (ps)", fontsize=12)
    ax.set_ylabel("Bias Energy (kJ/mol)", fontsize=12)
    ax.set_title("Bias Energy vs Time", fontsize=13)
    ax.grid(True, alpha=0.3)

# (d) Hill heights from HILLS file
ax = axes[1, 1]
if hills_data is not None and hills_data.shape[1] >= 4:
    hill_times_ps = hills_data[:, 0] / 1000.0
    hill_h = hills_data[:, 3] if hills_data.shape[1] > 3 else hills_data[:, 2]
    ax.semilogy(hill_times_ps, hill_h, "go-", markersize=1, linewidth=0.5)
    ax.set_xlabel("Time (ps)", fontsize=12)
    ax.set_ylabel("Hill Height (kJ/mol)", fontsize=12)
    ax.set_title("Hill Heights (well-tempered decay)", fontsize=13)
    ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "metad_lammps_results.png"), dpi=200, bbox_inches="tight")
print(f"\nPlots saved: {OUTPUT_DIR}/metad_lammps_results.png")
```

#### Script B2: LAMMPS + PLUMED with Steinhardt Order Parameters

```python
#!/usr/bin/env python3
"""
LAMMPS + PLUMED metadynamics using Steinhardt Q6 order parameter as CV.
Useful for crystallization/melting and solid-solid phase transitions.

Q6 distinguishes crystal structures:
  liquid ~ 0.0-0.1, BCC ~ 0.35-0.40, FCC ~ 0.57, HCP ~ 0.48

Generates LAMMPS + PLUMED input files, runs simulation, and plots FES.

Standalone -- no pyiron dependency.
"""

import os
import subprocess
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ============================================================
# 1. CONFIGURATION
# ============================================================

LAMMPS_BIN = "lmp"
PLUMED_BIN = "plumed"
POTENTIAL_FILE = "Al_mishin.eam.alloy"
NPROC = 4

ELEMENT = "Al"
A_LATTICE = 4.05
SUPERCELL = (3, 3, 3)                  # 108 atoms FCC
TEMPERATURE = 800.0                    # K
TIMESTEP = 2.0                         # fs
N_STEPS = 300000

# Metadynamics
HILL_HEIGHT = 1.0                      # kJ/mol
PACE = 500
BIAS_FACTOR = 20.0
Q6_SIGMA = 0.02                        # width in Q6 space

OUTPUT_DIR = "metad_lammps_q6"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ============================================================
# 2. BUILD STRUCTURE
# ============================================================

from ase.build import bulk
from ase.io import write as ase_write

atoms = bulk(ELEMENT, "fcc", a=A_LATTICE, cubic=True) * SUPERCELL
ase_write(os.path.join(OUTPUT_DIR, "structure.data"), atoms, format="lammps-data")
n_atoms = len(atoms)
print(f"Built {n_atoms}-atom {ELEMENT} FCC supercell")

# Build atom list string for PLUMED (all atoms)
atom_list = ",".join(str(i+1) for i in range(n_atoms))

# ============================================================
# 3. PLUMED INPUT -- Steinhardt Q6
# ============================================================

plumed_input = f"""# PLUMED input: well-tempered metadynamics with Q6 order parameter
UNITS LENGTH=A ENERGY=kj/mol TIME=fs

# Coordination-like environment for each atom (neighbors within cutoff)
coord: COORDINATIONNUMBER SPECIES=1-{n_atoms} SWITCH={{RATIONAL R_0=3.5 D_MAX=4.5 NN=6 MM=12}}

# Steinhardt Q6 averaged over all atoms
q6: Q6 SPECIES=1-{n_atoms} SWITCH={{RATIONAL R_0=3.5 D_MAX=4.5 NN=6 MM=12}} MEAN
# The MEAN component gives the average Q6 over all atoms

# Well-tempered metadynamics on mean Q6
metad: METAD ...
  ARG=q6.mean
  SIGMA={Q6_SIGMA}
  HEIGHT={HILL_HEIGHT}
  PACE={PACE}
  BIASFACTOR={BIAS_FACTOR}
  TEMP={TEMPERATURE}
  GRID_MIN=0.0
  GRID_MAX=0.7
  GRID_BIN=300
  FILE=HILLS
  CALC_RDB
...

# Print to COLVAR
PRINT ARG=q6.mean,metad.bias STRIDE=100 FILE=COLVAR
"""

with open(os.path.join(OUTPUT_DIR, "plumed.dat"), "w") as f:
    f.write(plumed_input)

# ============================================================
# 4. LAMMPS INPUT
# ============================================================

lammps_input = f"""# LAMMPS + PLUMED: metadynamics with Q6 order parameter
units           metal
atom_style      atomic
boundary        p p p

read_data       structure.data

pair_style      eam/alloy
pair_coeff      * * {POTENTIAL_FILE} {ELEMENT}

neighbor        2.0 bin
neigh_modify    every 1 delay 0 check yes

minimize        1.0e-6 1.0e-8 1000 10000
reset_timestep  0

velocity        all create {TEMPERATURE} 54321 dist gaussian

# NPT to allow volume changes during phase transition
fix             1 all npt temp {TEMPERATURE} {TEMPERATURE} $(100.0*dt) &
                iso 0.0 0.0 $(1000.0*dt)

fix             plumed all plumed plumedfile plumed.dat outfile plumed.out

timestep        {TIMESTEP / 1000.0:.6f}

thermo          2000
thermo_style    custom step temp pe ke etotal press vol

dump            1 all custom 2000 traj.lammpstrj id type x y z

run             {N_STEPS}

unfix           1
unfix           plumed
"""

with open(os.path.join(OUTPUT_DIR, "in.metad_q6"), "w") as f:
    f.write(lammps_input)

# ============================================================
# 5. RUN
# ============================================================

print(f"Running LAMMPS + PLUMED Q6 metadynamics ({N_STEPS} steps)...")
result = subprocess.run(
    ["mpirun", "-np", str(NPROC), LAMMPS_BIN, "-in", "in.metad_q6"],
    capture_output=True, text=True, cwd=OUTPUT_DIR, timeout=7200,
)

with open(os.path.join(OUTPUT_DIR, "lammps.out"), "w") as f:
    f.write(result.stdout)
if result.returncode == 0:
    print("Completed successfully.")
else:
    print(f"LAMMPS exited with code {result.returncode}")

# sum_hills
subprocess.run(
    [PLUMED_BIN, "sum_hills", "--hills", "HILLS",
     "--outfile", "fes_q6.dat", "--mintozero",
     "--kt", f"{TEMPERATURE * 8.314e-3:.4f}"],
    capture_output=True, text=True, cwd=OUTPUT_DIR, timeout=120,
)

# ============================================================
# 6. PLOT
# ============================================================

def parse_plumed_file(filepath):
    data, headers = [], []
    with open(filepath, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if line.startswith("#"):
                if "FIELDS" in line:
                    headers = line.split()[2:]
                continue
            data.append([float(x) for x in line.split()])
    return np.array(data) if data else None, headers

fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# FES
fes_path = os.path.join(OUTPUT_DIR, "fes_q6.dat")
if os.path.exists(fes_path):
    fes_data, _ = parse_plumed_file(fes_path)
    if fes_data is not None:
        ax = axes[0]
        ax.plot(fes_data[:, 0], fes_data[:, 1], "b-", linewidth=2)
        # Mark known Q6 values
        ax.axvline(0.05, color="gray", linestyle=":", alpha=0.5, label="Liquid")
        ax.axvline(0.37, color="orange", linestyle="--", alpha=0.7, label="BCC")
        ax.axvline(0.48, color="green", linestyle="--", alpha=0.7, label="HCP")
        ax.axvline(0.57, color="red", linestyle="--", alpha=0.7, label="FCC")
        ax.set_xlabel("Mean Q6", fontsize=12)
        ax.set_ylabel("Free Energy (kJ/mol)", fontsize=12)
        ax.set_title("FES: Steinhardt Q6", fontsize=13)
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)

# Q6 trajectory
colvar_path = os.path.join(OUTPUT_DIR, "COLVAR")
if os.path.exists(colvar_path):
    colvar, _ = parse_plumed_file(colvar_path)
    if colvar is not None:
        ax = axes[1]
        t_ps = colvar[:, 0] / 1000.0
        ax.plot(t_ps, colvar[:, 1], "k-", linewidth=0.3, alpha=0.6)
        ax.set_xlabel("Time (ps)", fontsize=12)
        ax.set_ylabel("Mean Q6", fontsize=12)
        ax.set_title("Q6 Trajectory", fontsize=13)
        ax.grid(True, alpha=0.3)

# Hill heights
hills_path = os.path.join(OUTPUT_DIR, "HILLS")
if os.path.exists(hills_path):
    hills, _ = parse_plumed_file(hills_path)
    if hills is not None:
        ax = axes[2]
        ht_ps = hills[:, 0] / 1000.0
        # height is typically the 4th column (time, cv, sigma, height, biasfactor)
        h_col = min(3, hills.shape[1] - 1)
        ax.semilogy(ht_ps, np.abs(hills[:, h_col]), "go-", markersize=1, linewidth=0.5)
        ax.set_xlabel("Time (ps)", fontsize=12)
        ax.set_ylabel("Hill Height (kJ/mol)", fontsize=12)
        ax.set_title("Hill Height Decay", fontsize=13)
        ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "metad_q6_results.png"), dpi=200, bbox_inches="tight")
print(f"Plots saved: {OUTPUT_DIR}/metad_q6_results.png")
```

---

### Analysis Scripts

#### Script C1: FES Convergence Analysis

```python
#!/usr/bin/env python3
"""
Convergence analysis for metadynamics simulations.
Checks whether the free energy surface has converged by:
  1. Comparing FES reconstructed from different time windows
  2. Monitoring hill height decay (well-tempered)
  3. Computing the running integral of deposited bias
  4. Checking recrossing statistics

Works with both Method A (Python hills) and Method B (PLUMED HILLS file).

Standalone -- no pyiron dependency.
"""

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.signal import argrelmin

# ============================================================
# CONFIGURATION
# ============================================================

# Choose one source:
# Option 1: PLUMED HILLS file
HILLS_FILE = "metad_lammps_plumed/HILLS"    # Set to None if using Python hills
# Option 2: Python hills file from Method A
PYTHON_HILLS_FILE = "metad_distance/hills.dat"  # Set to None if using PLUMED

OUTPUT_DIR = "metad_convergence"
os.makedirs(OUTPUT_DIR, exist_ok=True)

TEMPERATURE = 600.0                    # K (must match simulation)
BIAS_FACTOR = 15.0                     # gamma (must match simulation)
N_WINDOWS = 5                          # number of time windows for convergence check

# ============================================================
# PARSE HILLS
# ============================================================

def parse_plumed_hills(filepath):
    """Parse PLUMED HILLS file. Returns (time, cv_center, sigma, height)."""
    data = []
    with open(filepath, "r") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            data.append([float(x) for x in line.split()])
    data = np.array(data)
    # Typical columns: time, cv, sigma, height, biasfactor
    return data[:, 0], data[:, 1], data[:, 2], data[:, 3]


def parse_python_hills(filepath):
    """Parse hills.dat from Method A. Format: cv_center, height."""
    data = np.loadtxt(filepath)
    times = np.arange(len(data))  # no time info, use index
    return times, data[:, 0], None, data[:, 1]


# Load hills
if HILLS_FILE and os.path.exists(HILLS_FILE):
    times, centers, sigmas, heights = parse_plumed_hills(HILLS_FILE)
    if sigmas is not None:
        sigma = sigmas[0]  # assume constant sigma
    else:
        sigma = 0.2  # default
    source = "PLUMED"
    print(f"Loaded {len(times)} hills from PLUMED HILLS file")
elif PYTHON_HILLS_FILE and os.path.exists(PYTHON_HILLS_FILE):
    times, centers, sigmas, heights = parse_python_hills(PYTHON_HILLS_FILE)
    sigma = 0.15  # must match HILL_SIGMA from simulation
    source = "Python"
    print(f"Loaded {len(times)} hills from Python hills file")
else:
    raise FileNotFoundError("No HILLS file found. Run a metadynamics simulation first.")

# ============================================================
# 1. FES FROM DIFFERENT TIME WINDOWS
# ============================================================

def reconstruct_fes_from_hills(centers, heights, sigma, s_min, s_max, n_bins=300,
                                gamma=None):
    """Reconstruct FES from a set of deposited hills."""
    s_grid = np.linspace(s_min, s_max, n_bins)
    V_bias = np.zeros(n_bins)
    for c, h in zip(centers, heights):
        V_bias += h * np.exp(-0.5 * ((s_grid - c) / sigma) ** 2)
    if gamma is not None and gamma > 1.0:
        F = -(gamma / (gamma - 1.0)) * V_bias
    else:
        F = -V_bias
    F -= np.min(F)
    return s_grid, F


s_min = centers.min() - 3 * sigma
s_max = centers.max() + 3 * sigma
n_hills = len(centers)
window_size = n_hills // N_WINDOWS

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# (a) FES at different time windows
ax = axes[0, 0]
colors = plt.cm.viridis(np.linspace(0.2, 1.0, N_WINDOWS))
for i in range(N_WINDOWS):
    end_idx = (i + 1) * window_size
    s_grid, fes_i = reconstruct_fes_from_hills(
        centers[:end_idx], heights[:end_idx], sigma, s_min, s_max,
        gamma=BIAS_FACTOR
    )
    label = f"t={end_idx}/{n_hills} hills"
    ax.plot(s_grid, fes_i, color=colors[i], linewidth=1.5, label=label)

ax.set_xlabel("Collective Variable", fontsize=12)
ax.set_ylabel("Free Energy (eV or kJ/mol)", fontsize=12)
ax.set_title("FES Convergence: Time Windows", fontsize=13)
ax.legend(fontsize=8, loc="upper right")
ax.grid(True, alpha=0.3)

# (b) FES difference between successive windows
ax = axes[0, 1]
for i in range(1, N_WINDOWS):
    end_prev = i * window_size
    end_curr = (i + 1) * window_size
    _, fes_prev = reconstruct_fes_from_hills(
        centers[:end_prev], heights[:end_prev], sigma, s_min, s_max, gamma=BIAS_FACTOR
    )
    _, fes_curr = reconstruct_fes_from_hills(
        centers[:end_curr], heights[:end_curr], sigma, s_min, s_max, gamma=BIAS_FACTOR
    )
    diff = np.abs(fes_curr - fes_prev)
    ax.plot(s_grid, diff, color=colors[i], linewidth=1.0,
            label=f"Window {i} -> {i+1}")
    print(f"Max FES diff (window {i}->{i+1}): {np.max(diff):.4f}")

ax.set_xlabel("Collective Variable", fontsize=12)
ax.set_ylabel("|delta F|", fontsize=12)
ax.set_title("FES Difference Between Windows", fontsize=13)
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3)

# (c) Hill height evolution
ax = axes[1, 0]
ax.semilogy(np.arange(len(heights)), heights, "g-", linewidth=0.5)
ax.set_xlabel("Hill Index", fontsize=12)
ax.set_ylabel("Hill Height", fontsize=12)
ax.set_title("Hill Height vs Deposition Index", fontsize=13)
ax.grid(True, alpha=0.3)

# Fit exponential decay for well-tempered
if len(heights) > 20:
    # Running average
    window = min(50, len(heights) // 5)
    if window > 1:
        kernel = np.ones(window) / window
        smoothed = np.convolve(heights, kernel, mode="valid")
        ax.semilogy(np.arange(len(smoothed)) + window // 2, smoothed, "r-",
                    linewidth=2, label="Running average")
        ax.legend(fontsize=10)

# (d) Cumulative bias
ax = axes[1, 1]
cum_bias = np.cumsum(heights)
ax.plot(np.arange(len(cum_bias)), cum_bias, "b-", linewidth=1.5)
ax.set_xlabel("Hill Index", fontsize=12)
ax.set_ylabel("Cumulative Bias Deposited", fontsize=12)
ax.set_title("Total Bias Accumulation", fontsize=13)
ax.grid(True, alpha=0.3)
# A plateau indicates convergence
slope_late = (cum_bias[-1] - cum_bias[-len(cum_bias)//5]) / (len(cum_bias) // 5)
slope_early = (cum_bias[len(cum_bias)//5] - cum_bias[0]) / (len(cum_bias) // 5)
if slope_early > 0:
    ratio = slope_late / slope_early
    ax.text(0.05, 0.95, f"Late/early slope ratio: {ratio:.3f}\n(< 0.1 = good convergence)",
            transform=ax.transAxes, fontsize=10, va="top",
            bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5))

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "convergence_analysis.png"), dpi=200, bbox_inches="tight")
print(f"\nPlots saved: {OUTPUT_DIR}/convergence_analysis.png")

# ── Final FES with error estimate ──
s_grid_final, fes_final = reconstruct_fes_from_hills(
    centers, heights, sigma, s_min, s_max, gamma=BIAS_FACTOR
)

# Estimate error from window-to-window variation
fes_windows = []
for i in range(N_WINDOWS):
    start = i * window_size
    end = (i + 1) * window_size
    _, f_i = reconstruct_fes_from_hills(
        centers[start:end], heights[start:end], sigma, s_min, s_max, gamma=BIAS_FACTOR
    )
    fes_windows.append(f_i)
fes_std = np.std(fes_windows, axis=0)

np.savetxt(os.path.join(OUTPUT_DIR, "fes_with_error.dat"),
           np.column_stack([s_grid_final, fes_final, fes_std]),
           header="CV  FreeEnergy  StdError")

print(f"\nFES with error bars saved: {OUTPUT_DIR}/fes_with_error.dat")
print(f"Mean error across FES: {np.mean(fes_std):.4f}")
print(f"Max error: {np.max(fes_std):.4f}")
```

#### Script C2: Minimum Free Energy Path Extraction

```python
#!/usr/bin/env python3
"""
Extract the minimum free energy path (MFEP) from a 1D or 2D FES.
Also computes forward/reverse barriers and identifies transition states.

Works with FES data from any of the above scripts.

Standalone -- no pyiron dependency.
"""

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.signal import argrelmin, argrelextrema
from scipy.ndimage import gaussian_filter1d

# ============================================================
# CONFIGURATION
# ============================================================

FES_FILE = "metad_distance/fes.dat"    # 2-column: CV, FreeEnergy
OUTPUT_DIR = "metad_mfep"
os.makedirs(OUTPUT_DIR, exist_ok=True)

SMOOTH_SIGMA = 3.0                     # Gaussian smoothing width (grid points)

# ============================================================
# LOAD FES
# ============================================================

data = np.loadtxt(FES_FILE)
cv = data[:, 0]
fes_raw = data[:, 1]

# Smooth for robust extremum detection
fes = gaussian_filter1d(fes_raw, sigma=SMOOTH_SIGMA)

print(f"Loaded FES: {len(cv)} points, CV range [{cv.min():.3f}, {cv.max():.3f}]")

# ============================================================
# FIND MINIMA AND MAXIMA
# ============================================================

# Local minima
min_idx = argrelextrema(fes, np.less, order=10)[0]
# Local maxima (transition states)
max_idx = argrelextrema(fes, np.greater, order=10)[0]

print(f"\nLocal minima found: {len(min_idx)}")
for i, idx in enumerate(min_idx):
    print(f"  Minimum {i+1}: CV = {cv[idx]:.4f}, F = {fes[idx]:.4f}")

print(f"\nLocal maxima (TS candidates): {len(max_idx)}")
for i, idx in enumerate(max_idx):
    print(f"  Maximum {i+1}: CV = {cv[idx]:.4f}, F = {fes[idx]:.4f}")

# ============================================================
# COMPUTE BARRIERS
# ============================================================

barriers = []
if len(min_idx) >= 2:
    for i in range(len(min_idx) - 1):
        idx_a = min_idx[i]
        idx_b = min_idx[i + 1]
        # Find highest point between the two minima
        segment = fes[idx_a:idx_b + 1]
        ts_local = np.argmax(segment)
        ts_idx = idx_a + ts_local
        barrier_fwd = fes[ts_idx] - fes[idx_a]
        barrier_rev = fes[ts_idx] - fes[idx_b]
        delta_F = fes[idx_b] - fes[idx_a]
        barriers.append({
            "min_a": cv[idx_a],
            "min_b": cv[idx_b],
            "ts": cv[ts_idx],
            "F_a": fes[idx_a],
            "F_b": fes[idx_b],
            "F_ts": fes[ts_idx],
            "barrier_fwd": barrier_fwd,
            "barrier_rev": barrier_rev,
            "delta_F": delta_F,
        })
        print(f"\n--- Transition {i+1}: Min{i+1} -> Min{i+2} ---")
        print(f"  Min A: CV={cv[idx_a]:.4f}, F={fes[idx_a]:.4f}")
        print(f"  Min B: CV={cv[idx_b]:.4f}, F={fes[idx_b]:.4f}")
        print(f"  TS:    CV={cv[ts_idx]:.4f}, F={fes[ts_idx]:.4f}")
        print(f"  Forward barrier:  {barrier_fwd:.4f}")
        print(f"  Reverse barrier:  {barrier_rev:.4f}")
        print(f"  Free energy diff: {delta_F:.4f}")

# ============================================================
# TRANSITION RATE ESTIMATE (Kramers/TST)
# ============================================================

kB = 8.617e-5  # eV/K
temperatures = [300, 500, 800, 1000, 1500]
print(f"\n{'='*60}")
print("TRANSITION RATE ESTIMATES (harmonic TST approximation)")
print(f"{'='*60}")
print(f"  k ~ nu_0 * exp(-barrier / kT)")
print(f"  Assuming attempt frequency nu_0 ~ 10^13 Hz (typical phonon)")
nu0 = 1e13  # Hz

for b in barriers:
    print(f"\n  Transition: CV {b['min_a']:.3f} -> {b['min_b']:.3f}")
    for T in temperatures:
        kT = kB * T
        rate_fwd = nu0 * np.exp(-b["barrier_fwd"] / kT)
        rate_rev = nu0 * np.exp(-b["barrier_rev"] / kT)
        t_fwd = 1.0 / rate_fwd if rate_fwd > 0 else np.inf
        t_rev = 1.0 / rate_rev if rate_rev > 0 else np.inf
        print(f"    T={T:5d} K: k_fwd={rate_fwd:.2e} Hz (tau={t_fwd:.2e} s), "
              f"k_rev={rate_rev:.2e} Hz")

# ============================================================
# PLOT
# ============================================================

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# (a) FES with annotations
ax = axes[0]
ax.plot(cv, fes_raw, "b-", linewidth=1, alpha=0.3, label="Raw FES")
ax.plot(cv, fes, "b-", linewidth=2, label="Smoothed FES")

# Mark minima
for idx in min_idx:
    ax.plot(cv[idx], fes[idx], "gv", markersize=12, zorder=5)
    ax.annotate(f"Min\n{cv[idx]:.2f}", (cv[idx], fes[idx]),
                textcoords="offset points", xytext=(0, -25), ha="center", fontsize=9)

# Mark TS
for idx in max_idx:
    ax.plot(cv[idx], fes[idx], "r^", markersize=12, zorder=5)
    ax.annotate(f"TS\n{cv[idx]:.2f}", (cv[idx], fes[idx]),
                textcoords="offset points", xytext=(0, 15), ha="center", fontsize=9)

# Draw barriers
for b in barriers:
    ax.annotate("", xy=(b["ts"], b["F_ts"]),
                xytext=(b["min_a"], b["F_a"]),
                arrowprops=dict(arrowstyle="<->", color="red", lw=1.5))
    mid_cv = (b["min_a"] + b["ts"]) / 2
    mid_f = (b["F_a"] + b["F_ts"]) / 2
    ax.text(mid_cv, mid_f, f"{b['barrier_fwd']:.3f}", fontsize=10, color="red",
            ha="center", va="bottom")

ax.set_xlabel("Collective Variable", fontsize=12)
ax.set_ylabel("Free Energy", fontsize=12)
ax.set_title("Free Energy Surface with Barriers", fontsize=13)
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3)

# (b) Arrhenius plot for forward barrier
ax = axes[1]
if barriers:
    inv_T = 1000.0 / np.array(temperatures)  # 1000/T
    for i, b in enumerate(barriers):
        rates = [nu0 * np.exp(-b["barrier_fwd"] / (kB * T)) for T in temperatures]
        ax.semilogy(inv_T, rates, "o-", linewidth=2,
                    label=f"Transition {i+1} ({b['barrier_fwd']:.3f} eV)")
    ax.set_xlabel("1000/T (1/K)", fontsize=12)
    ax.set_ylabel("Rate (Hz)", fontsize=12)
    ax.set_title("Arrhenius Plot", fontsize=13)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "mfep_analysis.png"), dpi=200, bbox_inches="tight")
print(f"\nPlots saved: {OUTPUT_DIR}/mfep_analysis.png")

# Save barrier data
if barriers:
    with open(os.path.join(OUTPUT_DIR, "barriers.txt"), "w") as f:
        f.write("# Barriers extracted from FES\n")
        for i, b in enumerate(barriers):
            f.write(f"Transition {i+1}:\n")
            for key, val in b.items():
                f.write(f"  {key}: {val:.6f}\n")
    print(f"Barriers saved: {OUTPUT_DIR}/barriers.txt")
```

---

## Key Parameters

| Parameter | Typical Value | Notes |
|---|---|---|
| `HILL_HEIGHT` | 0.01--0.1 eV (MACE) or 0.5--5 kJ/mol (PLUMED) | Too large: poor resolution. Too small: slow exploration. |
| `HILL_SIGMA` | ~1/3 of expected CV range per basin | Must resolve distinct basins. Too narrow: noisy FES. Too wide: basins merge. |
| `BIAS_FACTOR` | 5--30 | Higher = more exploration, slower convergence. 10--15 is typical. |
| `PACE` / `HILL_EVERY` | 100--1000 steps | Must be longer than CV decorrelation time. |
| `TEMPERATURE` | System-dependent | Higher T helps exploration but may destabilize structures. |
| `N_STEPS` | 10k--1M | Must be enough for multiple basin recrossings. |
| `CN_R0` | Near first-neighbor distance | For coordination number CV; depends on element and structure. |
| `Q6 SWITCH R_0` | First-neighbor distance | For Steinhardt order parameters. |

## Collective Variable Selection Guide

| CV Type | Use Case | Advantages | Limitations |
|---|---|---|---|
| Distance | Bond breaking, vacancy hop, dissociation | Simple, exact gradient | Only 2-body; poor for collective transitions |
| Coordination Number | Phase transitions, melting, dissolution | Captures local environment changes | Requires careful switching function tuning |
| Steinhardt Q4/Q6 | Solid-solid transitions, crystallization | Distinguishes crystal symmetries | Expensive for large systems; averaged value may miss local nucleation |
| Volume | Pressure-driven transitions (graphite->diamond) | Natural NPT variable | Does not distinguish polymorphs at same density |
| Angle / Torsion | Octahedral tilting, molecular rotation | Direct geometric meaning | Limited to specific structural motifs |
| Linear Combination | Multiple competing processes | Captures coupled motions | Requires knowledge of important primitive CVs |
| Path CV (s, z) | Complex multi-step transitions | Projects onto known reaction coordinate | Requires initial/final state structures |

## Interpreting Results

- **Well-converged FES**: Hill heights decay to < 1% of initial value; FES from last 20% of simulation matches FES from last 40%.
- **Recrossing**: CV trajectory should revisit all basins multiple times after filling. No recrossing = insufficient sampling.
- **Barrier height**: Read from FES as the difference between a minimum and the highest saddle point on the path connecting it to another minimum.
- **Free energy difference**: Difference between two basin minima on the FES gives the thermodynamic driving force.
- **Well-tempered correction**: Raw bias gives -(gamma/(gamma-1)) * F(s). PLUMED sum_hills applies this automatically when --kt is provided.
- **Error bars**: Compare FES from independent time blocks. Standard deviation across blocks estimates statistical uncertainty.

## Common Issues

| Problem | Solution |
|---|---|
| FES does not converge (hills never shrink) | Increase `BIAS_FACTOR`, reduce `HILL_HEIGHT`, run longer. Check that CV actually varies during simulation. |
| CV stuck in one basin | Hill height too small, or sigma too wide (fills basin too slowly). Increase height or decrease sigma. |
| CV oscillates rapidly but never transitions | CV may not be the slow degree of freedom. Choose a better CV or add orthogonal CVs. |
| LAMMPS crashes with PLUMED | Ensure LAMMPS was compiled with PLUMED support (`lmp -h` should list PLUMED). Check unit consistency (PLUMED defaults: nm, kJ/mol, ps). |
| FES has unphysical negative regions | Artifacts from insufficient sampling at boundaries. Increase exploration or restrict CV range with PLUMED UPPER_WALLS/LOWER_WALLS. |
| Forces blow up | Sigma too small (sharp Gaussians create steep gradients). Increase sigma. For MACE, ensure atoms do not get too close (use UPPER_WALLS on distance CVs). |
| Volume CV metadynamics unstable | Volume changes too fast; reduce hill height and increase deposition interval. Use NPT with a stiff barostat (small taup). |
| Q6 computation slow | Steinhardt parameters require neighbor lists at every step. Reduce cutoff or use PLUMED's GRID acceleration. |
