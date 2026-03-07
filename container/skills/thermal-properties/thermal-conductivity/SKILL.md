# Lattice Thermal Conductivity

## When to Use

- Computing lattice thermal conductivity of crystalline solids
- Screening materials for thermoelectric performance (low kappa_L desired)
- Identifying thermal barrier coatings or thermal interface materials
- Comparing phonon-mediated heat transport across material families
- Estimating temperature-dependent thermal conductivity

## Method Selection

```
What level of accuracy and system complexity?

Quick screening of many materials?
  --> Method A: ASE + MACE equilibrium MD (Green-Kubo)
     Pro: Easy setup, any chemistry MACE supports
     Con: Large system + long run needed; convergence can be slow

Need directional (anisotropic) conductivity?
  --> Method B: LAMMPS NEMD (non-equilibrium MD)
     Pro: Direct method, clear signal, directional
     Con: Needs validated potential; finite-size effects

Publication-quality for simple crystals?
  --> Method C: phono3py (phonon Boltzmann transport equation)
     Pro: Most rigorous; temperature-dependent; no MD noise
     Con: Expensive (3rd-order force constants); limited to harmonic regime

Need conductivity above Debye temperature / for disordered systems?
  --> Use Green-Kubo (Method A) or NEMD (Method B)
     Phonon BTE breaks down for strongly anharmonic / disordered systems

System has > 50 atoms in unit cell?
  --> Method A or B preferred (phono3py becomes very expensive)
```

Key trade-offs:
- **Green-Kubo (MD)**: General, handles anharmonicity naturally, but needs long runs (1+ ns) and large cells. Statistical uncertainty is inherent.
- **NEMD**: Direct temperature gradient method. Clearer signal than Green-Kubo, but requires careful finite-size scaling.
- **phono3py (BTE)**: Most accurate for crystalline solids in the phonon picture. Computes 3-phonon scattering rates explicitly. Expensive but systematic.

## Prerequisites

```bash
# For Method C (phono3py)
pip install phono3py phonopy seekpath
```

Pre-installed: `ase`, `mace-torch`, `pymatgen`, `numpy`, `scipy`, `matplotlib`.

LAMMPS binary: `lmp` (for Method B).

## Detailed Steps

### Method A: Green-Kubo (Equilibrium MD) with ASE + MACE

The Green-Kubo relation computes thermal conductivity from the autocorrelation of the heat flux:

kappa = V / (kB * T^2) * integral_0^inf <J(0) . J(t)> dt

where J is the heat flux vector.

```python
#!/usr/bin/env python3
"""
Lattice thermal conductivity via the Green-Kubo method.
Uses ASE + MACE for MD simulation with heat flux calculation.
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import os

from ase.io import read, write, Trajectory
from ase.optimize import LBFGS
from ase.constraints import ExpCellFilter
from ase.md.langevin import Langevin
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution, Stationary
from ase.md.verlet import VelocityVerlet
from ase import units

from mace.calculators import mace_mp

import warnings
warnings.filterwarnings("ignore")

# ============================================================
# 1. CONFIGURATION
# ============================================================

STRUCTURE_FILE = "structure.cif"
MACE_MODEL = "medium"
DEVICE = "cpu"

TEMPERATURE = 300.0          # K
SUPERCELL = (4, 4, 4)        # Need large supercell for converged kappa
TIMESTEP = 1.0               # fs (use 0.5 for systems with H)

# Equilibration with thermostat (NVT)
N_EQUIL = 5000               # Equilibration steps

# Production with NVE (no thermostat -- essential for Green-Kubo)
N_PROD = 100000              # Production steps (need long runs!)
HEAT_FLUX_INTERVAL = 1       # Compute heat flux every N steps

# Green-Kubo settings
MAX_CORRELATION_TIME_PS = 10.0  # Max correlation time for HFACF
WINDOW_AVERAGE = True           # Average over multiple time origins

WORK_DIR = "/tmp/thermal_conductivity_gk"
os.makedirs(WORK_DIR, exist_ok=True)

# ============================================================
# 2. SET UP STRUCTURE
# ============================================================

calc = mace_mp(model=MACE_MODEL, device=DEVICE, default_dtype="float64")

atoms = read(STRUCTURE_FILE)
atoms.calc = calc

# Relax
print("=== Relaxing structure ===")
ecf = ExpCellFilter(atoms)
opt = LBFGS(ecf, logfile=os.path.join(WORK_DIR, "relax.log"))
opt.run(fmax=1e-4, steps=500)
print(f"  Formula: {atoms.get_chemical_formula()}")
print(f"  Relaxed energy: {atoms.get_potential_energy():.6f} eV")

# Build supercell
atoms = atoms.repeat(SUPERCELL)
atoms.calc = calc
n_atoms = len(atoms)
volume_A3 = atoms.get_volume()
print(f"  Supercell: {SUPERCELL}, {n_atoms} atoms, V = {volume_A3:.2f} A^3")

# ============================================================
# 3. COMPUTE HEAT FLUX
# ============================================================

def compute_heat_flux(atoms):
    """
    Compute the heat flux vector J for a system with a pair potential.

    J = (1/V) * [sum_i e_i * v_i + sum_i (f_ij . v_i) * r_ij]

    For practical MD with general potentials, we use the Hardy formulation:
    J = (1/V) * [sum_i (KE_i + PE_i) * v_i + sum_i S_i . v_i]

    where S_i is the per-atom stress tensor.

    For MACE/ASE, we compute the per-atom energy and stress contributions.
    Simplified approach: use the virial stress and kinetic energy.
    """
    velocities = atoms.get_velocities()  # A/fs (ASE internal)
    masses = atoms.get_masses()          # amu
    forces = atoms.get_forces()          # eV/A
    positions = atoms.get_positions()    # A
    volume = atoms.get_volume()          # A^3

    # Kinetic energy contribution: sum_i (0.5 * m_i * v_i^2) * v_i
    # Convert mass from amu to eV*fs^2/A^2: m(amu) * units._amu
    # In ASE units: KE = 0.5 * m * v^2 is in eV if m is in internal units and v in A/fs
    # Actually, ASE velocities and masses are in internal units.
    # KE_i = 0.5 * masses[i] * amu * sum(velocities[i]^2)

    from ase.units import _amu  # amu in eV*fs^2/A^2 -- this is amu converted to ASE mass units

    # Per-atom kinetic energy
    ke_per_atom = 0.5 * masses * _amu * np.sum(velocities ** 2, axis=1)  # eV

    # Per-atom potential energy (requires per-atom energy decomposition)
    # MACE supports per-atom energies via the calculator
    try:
        pe_per_atom = atoms.calc.results.get("energies", None)
        if pe_per_atom is None:
            # Fallback: distribute total energy equally (approximation)
            pe_total = atoms.get_potential_energy()
            pe_per_atom = np.full(n_atoms, pe_total / n_atoms)
    except Exception:
        pe_total = atoms.get_potential_energy()
        pe_per_atom = np.full(n_atoms, pe_total / n_atoms)

    # Convective heat flux: J_conv = (1/V) * sum_i (ke_i + pe_i) * v_i
    e_per_atom = ke_per_atom + pe_per_atom  # eV
    J_conv = np.sum(e_per_atom[:, np.newaxis] * velocities, axis=0) / volume  # eV*A/fs / A^3 = eV/(A^2*fs)

    # Virial contribution: J_virial = (1/V) * sum_i (F_i . v_i contribution via stress)
    # Use the per-atom stress * velocity approach
    # For pair potentials: J_virial = (1/V) * sum_{i<j} (r_ij outer f_ij) . (v_i + v_j)/2
    # Approximate using the total stress tensor:
    # sigma = -(1/V) * [sum_i m_i v_i outer v_i + sum_i r_i outer f_i]
    # The virial heat flux contribution is:
    # J_virial = -(1/V) * sum_i (S_i . v_i) where S_i is per-atom stress

    # Simplified: use force-velocity cross term
    # J_fv = -(1/V) * sum_i (r_i outer f_i) . v_i
    # This is an approximation for many-body potentials
    stress_contrib = np.zeros(3)
    for i in range(n_atoms):
        # r_i outer f_i gives a 3x3 matrix, dotted with v_i gives a 3-vector
        stress_contrib += np.dot(np.outer(positions[i], forces[i]), velocities[i])
    J_virial = -stress_contrib / volume

    J_total = J_conv + J_virial  # eV / (A^2 * fs)

    return J_total


# ============================================================
# 4. RUN MD AND COLLECT HEAT FLUX
# ============================================================

# Initialize velocities
MaxwellBoltzmannDistribution(atoms, temperature_K=TEMPERATURE)
Stationary(atoms)

# NVT equilibration
print(f"\n=== NVT Equilibration: {N_EQUIL} steps ===")
friction = 0.01 / units.fs
dyn_nvt = Langevin(atoms, timestep=TIMESTEP * units.fs, temperature_K=TEMPERATURE, friction=friction)
dyn_nvt.run(N_EQUIL)
print(f"  T after equilibration: {atoms.get_temperature():.1f} K")

# Switch to NVE for production (thermostat would corrupt heat flux autocorrelation)
print(f"\n=== NVE Production: {N_PROD} steps ({N_PROD * TIMESTEP / 1000:.1f} ps) ===")
dyn_nve = VelocityVerlet(atoms, timestep=TIMESTEP * units.fs)

heat_flux_data = []
thermo_data = []

def collect_heat_flux():
    """Collect heat flux at each step."""
    J = compute_heat_flux(atoms)
    heat_flux_data.append(J.copy())
    step = dyn_nve.nsteps
    if step % 10000 == 0:
        T = atoms.get_temperature()
        thermo_data.append((step, T))
        print(f"  Step {step:8d}: T = {T:.1f} K")

dyn_nve.attach(collect_heat_flux, interval=HEAT_FLUX_INTERVAL)
dyn_nve.run(N_PROD)

heat_flux = np.array(heat_flux_data)  # (n_samples, 3)
dt_ps = TIMESTEP * HEAT_FLUX_INTERVAL / 1000.0  # ps
print(f"\n  Heat flux samples: {len(heat_flux)}")
print(f"  Time resolution: {dt_ps:.4f} ps")

# ============================================================
# 5. GREEN-KUBO: HEAT FLUX AUTOCORRELATION
# ============================================================

print("\n=== Green-Kubo Analysis ===")

max_lag = int(MAX_CORRELATION_TIME_PS / dt_ps)
max_lag = min(max_lag, len(heat_flux) // 2)

def compute_hfacf(J, max_lag):
    """
    Compute heat flux autocorrelation function (HFACF).
    HFACF(t) = <J(0) . J(t)>
    Uses FFT for efficiency.
    """
    n = len(J)
    n_fft = 2 ** int(np.ceil(np.log2(2 * n)))

    hfacf = np.zeros(max_lag)

    for dim in range(3):
        j = J[:, dim]
        fft_j = np.fft.fft(j, n=n_fft)
        power = np.real(fft_j * np.conj(fft_j))
        acf = np.real(np.fft.ifft(power))[:n]
        # Normalize by number of overlapping pairs
        acf /= np.arange(n, 0, -1)
        hfacf += acf[:max_lag]

    return hfacf


hfacf = compute_hfacf(heat_flux, max_lag)
time_corr = np.arange(max_lag) * dt_ps  # ps

# ============================================================
# 6. COMPUTE THERMAL CONDUCTIVITY
# ============================================================

# kappa = V / (kB * T^2) * integral_0^inf HFACF(t) dt
#
# Units:
# HFACF has units of [J]^2 = (eV / (A^2 * fs))^2
# integral HFACF * dt: (eV / (A^2 * fs))^2 * ps = (eV / (A^2 * fs))^2 * 1000 fs
# V in A^3
# kB = 8.617e-5 eV/K
# T in K
#
# kappa = V(A^3) / (kB(eV/K) * T^2(K^2)) * integral
# Need to convert to W/(m*K):
# 1 eV = 1.602e-19 J, 1 A = 1e-10 m, 1 fs = 1e-15 s
# [eV/(A^2*fs)]^2 * fs * A^3 / (eV/K * K^2)
# = eV^2/(A^4*fs^2) * fs * A^3 / (eV * K)
# = eV / (A * fs * K)
# = 1.602e-19 / (1e-10 * 1e-15 * 1) J / (m * s * K)
# = 1.602e-19 / 1e-25 W/(m*K)
# = 1.602e6 W/(m*K)

from scipy.constants import Boltzmann, eV as eV_J

kB_eV = Boltzmann / eV_J  # eV/K

# Running integral of HFACF
running_integral = np.cumsum(hfacf) * dt_ps  # (eV/(A^2*fs))^2 * ps

# Convert units
# Factor: V / (kB * T^2)
# Then convert (eV/(A^2*fs))^2 * ps * A^3 / (eV/K * K^2)
# = eV * ps / (A * fs * K)  [simplifying eV^2/eV, A^3/A^4, ps/fs^2]
# ps / fs = 1000, so
# = 1000 * eV / (A * K * fs)  -- wait, let me redo this carefully.

# HFACF units: [eV / (A^2 * fs)]^2 = eV^2 / (A^4 * fs^2)
# integral * dt_ps: eV^2 / (A^4 * fs^2) * ps = eV^2 / (A^4 * fs^2) * 1e3 fs = 1e3 * eV^2 / (A^4 * fs)
# Prefactor V/(kB * T^2): A^3 / (eV/K * K^2) = A^3 * K / eV
# kappa = A^3 * K / eV * 1e3 * eV^2 / (A^4 * fs)
#       = 1e3 * eV * K / (A * fs)
# Convert to SI: eV -> 1.602e-19 J, A -> 1e-10 m, fs -> 1e-15 s, K -> K
# = 1e3 * 1.602e-19 / (1e-10 * 1e-15) W/m/K
# = 1e3 * 1.602e-19 / 1e-25 W/m/K
# = 1e3 * 1.602e6 W/m/K
# = 1.602e9 W/m/K  <-- this seems too large. Let me recheck.

# Actually, let's just use a direct SI conversion.
# Convert everything to SI first:
# J in eV/(A^2 * fs) -> SI: J * eV_J / (1e-10^2 * 1e-15) = J * eV_J / 1e-35 = J * eV_J * 1e35
#                        = J in W/m^2
# HFACF in [W/m^2]^2
# integral * dt in [W/m^2]^2 * s
# kappa = V(m^3) / (kB(J/K) * T^2) * integral

# Volume in m^3
V_m3 = volume_A3 * 1e-30

# Convert heat flux to W/m^2
# J_eV_A2_fs = J in eV/(A^2 * fs)
# J_SI = J_eV_A2_fs * eV_J / (1e-10)^2 / (1e-15)  -- No, heat flux is energy/(area*time)
# Wait: our J has units eV/(A^2 * fs) which is energy_flux/area... no.
# Let me reconsider. Our J = (1/V) * sum(e*v), so
# [J] = 1/A^3 * eV * A/fs = eV / (A^2 * fs)
# This is energy flux density: energy / (area * time) = power / area = W/m^2 equivalent.
# Convert: eV/(A^2 * fs) -> J/(m^2 * s)
# = eV_J / ((1e-10)^2 * (1e-15)) = eV_J / 1e-35 = eV_J * 1e35
conversion_J = eV_J * 1e35  # converts eV/(A^2*fs) to W/m^2

# HFACF in SI: (W/m^2)^2
hfacf_SI = hfacf * conversion_J ** 2

# dt in seconds
dt_s = dt_ps * 1e-12

# Running integral in SI: (W/m^2)^2 * s
running_integral_SI = np.cumsum(hfacf_SI) * dt_s

# kappa = V / (kB * T^2) * integral
kappa_running = V_m3 / (Boltzmann * TEMPERATURE ** 2) * running_integral_SI  # W/(m*K)

# Average over x, y, z is already done (we summed HFACF over 3 dimensions)
# Divide by 3 for the trace average
kappa_running /= 3.0

# The converged kappa is the plateau of kappa_running
# Find plateau by looking at the last 30% of the correlation window
plateau_start = int(0.5 * max_lag)
plateau_end = max_lag
kappa_avg = np.mean(kappa_running[plateau_start:plateau_end])
kappa_std = np.std(kappa_running[plateau_start:plateau_end])

print(f"  Thermal conductivity (Green-Kubo):")
print(f"    kappa = {kappa_avg:.2f} +/- {kappa_std:.2f} W/(m*K)")
print(f"    (Averaged over correlation times {time_corr[plateau_start]:.1f} - {time_corr[plateau_end-1]:.1f} ps)")

# ============================================================
# 7. PLOTS
# ============================================================

# HFACF
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

axes[0].plot(time_corr, hfacf / hfacf[0], "b-", linewidth=0.8)
axes[0].axhline(y=0, color="gray", linestyle="--", linewidth=0.5)
axes[0].set_xlabel("Correlation time (ps)", fontsize=12)
axes[0].set_ylabel("Normalized HFACF", fontsize=12)
axes[0].set_title("Heat Flux Autocorrelation", fontsize=13)
axes[0].grid(True, alpha=0.3)

axes[1].plot(time_corr, kappa_running, "r-", linewidth=1.2)
axes[1].axhline(y=kappa_avg, color="k", linestyle="--", linewidth=1,
                label=f"$\\kappa$ = {kappa_avg:.1f} W/(m*K)")
axes[1].axhspan(kappa_avg - kappa_std, kappa_avg + kappa_std, alpha=0.2, color="gray")
axes[1].set_xlabel("Correlation time (ps)", fontsize=12)
axes[1].set_ylabel("$\\kappa$ (W/(m*K))", fontsize=12)
axes[1].set_title("Running Thermal Conductivity", fontsize=13)
axes[1].legend(fontsize=10)
axes[1].grid(True, alpha=0.3)
axes[1].set_ylim(bottom=0)

fig.suptitle(f"Green-Kubo Thermal Conductivity at {TEMPERATURE} K", fontsize=14)
fig.tight_layout()
fig.savefig(os.path.join(WORK_DIR, "thermal_conductivity_gk.png"), dpi=150, bbox_inches="tight")
plt.close()
print(f"\n  Saved: {WORK_DIR}/thermal_conductivity_gk.png")

# Save data
np.savetxt(os.path.join(WORK_DIR, "hfacf.dat"),
           np.column_stack([time_corr, hfacf, hfacf / hfacf[0]]),
           header="time_ps  HFACF_raw  HFACF_normalized",
           fmt="%12.6f %18.8e %14.8f")
np.savetxt(os.path.join(WORK_DIR, "kappa_running.dat"),
           np.column_stack([time_corr, kappa_running]),
           header="time_ps  kappa_W_per_mK",
           fmt="%12.6f %14.6f")

print(f"\nAll outputs in: {WORK_DIR}/")
```

### Method B: Non-Equilibrium MD (NEMD) with LAMMPS

```python
#!/usr/bin/env python3
"""
Non-equilibrium MD (NEMD) thermal conductivity with LAMMPS.
Uses the Muller-Plathe reverse NEMD method or direct temperature gradient.
Generates LAMMPS input file for the NEMD simulation.
"""

import os
import numpy as np
from pymatgen.core.structure import Structure
from pymatgen.core.lattice import Lattice
from pymatgen.io.lammps.data import LammpsData

# ============================================================
# CONFIGURATION
# ============================================================

STRUCTURE_FILE = "structure.cif"
WORK_DIR = "/tmp/thermal_conductivity_nemd"
os.makedirs(WORK_DIR, exist_ok=True)

TEMPERATURE = 300.0        # K
TIMESTEP_PS = 0.001        # 1 fs in ps (LAMMPS metal units)
N_EQUIL = 50000            # Equilibration steps
N_PROD = 200000            # Production steps for NEMD

# Supercell: elongated in transport direction (z)
# Need long dimension for temperature gradient
SUPERCELL = [4, 4, 20]     # Short x,y, long z for z-direction transport

# NEMD swap parameters
SWAP_INTERVAL = 100         # Swap kinetic energy every N steps

# Potential (example: Cu EAM)
POTENTIAL_STYLE = "eam/alloy"
POTENTIAL_FILE = "Cu01.eam.alloy"
POTENTIAL_COEFF = "* * Cu01.eam.alloy Cu"

# ============================================================
# GENERATE STRUCTURE
# ============================================================

structure = Structure.from_file(STRUCTURE_FILE)
structure.make_supercell(SUPERCELL)
print(f"Structure: {structure.composition}, {len(structure)} atoms")
print(f"Cell dimensions: {structure.lattice.abc}")

lammps_data = LammpsData.from_structure(structure, atom_style="atomic")
data_file = os.path.join(WORK_DIR, "structure.data")
lammps_data.write_file(data_file)

# ============================================================
# WRITE LAMMPS INPUT FOR NEMD
# ============================================================

# Method: Muller-Plathe reverse NEMD
# - Divide box into slabs along z
# - Periodically swap velocities between hottest atom in cold slab and coldest in hot slab
# - Measure steady-state temperature gradient
# - kappa = Q / (A * dT/dz)

lammps_input = f"""# LAMMPS NEMD thermal conductivity - Muller-Plathe method
# Transport direction: z

# ============ Initialization ============
units           metal
dimension       3
boundary        p p p
atom_style      atomic

# ============ Read Structure ============
read_data       structure.data

# ============ Potential ============
pair_style      {POTENTIAL_STYLE}
pair_coeff      {POTENTIAL_COEFF}

# ============ Settings ============
neighbor        2.0 bin
neigh_modify    every 1 delay 0 check yes
timestep        {TIMESTEP_PS}

# ============ Minimization ============
minimize        1.0e-6 1.0e-8 1000 10000
reset_timestep  0

# ============ Initialize Velocities ============
velocity        all create {TEMPERATURE} 12345 dist gaussian

# ============ NVT Equilibration ============
thermo          1000
thermo_style    custom step temp pe ke etotal press

fix             equil all nvt temp {TEMPERATURE} {TEMPERATURE} $(100*dt)
run             {N_EQUIL}
unfix           equil

# ============ NEMD Production ============
reset_timestep  0

# Define hot and cold regions for Muller-Plathe
# The box is divided into 20 slabs along z
# Slab 0 (bottom) = cold sink, Slab 10 (middle) = hot source

# Muller-Plathe: swap kinetic energy between slabs
fix             nemd all thermal/conductivity {SWAP_INTERVAL} z 20

# Compute temperature profile
# Divide into 20 layers along z and compute T in each
compute         layers all chunk/atom bin/1d z lower 0.05
fix             temp_profile all ave/chunk 10 1000 10000 layers temp &
                file temperature_profile.dat

# Also track the cumulative energy swapped
# The fix thermal/conductivity outputs the swapped energy as a scalar

thermo          {N_PROD // 100}
thermo_style    custom step temp f_nemd

# Trajectory for verification
dump            nemd_traj all custom 10000 dump.nemd id type x y z vx vy vz
dump_modify     nemd_traj sort id

# NVE production (no thermostat during NEMD!)
fix             nve all nve
run             {N_PROD}

# ============ Output ============
print "=== NEMD Complete ==="
print "Total energy swapped (eV): $(f_nemd)"
"""

input_file = os.path.join(WORK_DIR, "in.nemd")
with open(input_file, "w") as f:
    f.write(lammps_input)
print(f"Written: {input_file}")

# ============================================================
# POST-PROCESSING SCRIPT
# ============================================================

postprocess_script = '''#!/usr/bin/env python3
"""
Post-process LAMMPS NEMD output to extract thermal conductivity.
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import os

WORK_DIR = "/tmp/thermal_conductivity_nemd"
TIMESTEP_PS = 0.001
N_PROD = 200000
SWAP_INTERVAL = 100
TEMPERATURE = 300.0

# Parse temperature profile
temp_profile_file = os.path.join(WORK_DIR, "temperature_profile.dat")

if os.path.exists(temp_profile_file):
    # LAMMPS ave/chunk output format
    # Read blocks of data (each block is one time average)
    blocks = []
    current_block = []
    with open(temp_profile_file, "r") as f:
        for line in f:
            if line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) == 2:
                # Header line: Nchunks, Timestep
                if current_block:
                    blocks.append(np.array(current_block))
                current_block = []
            elif len(parts) >= 3:
                current_block.append([float(x) for x in parts])
    if current_block:
        blocks.append(np.array(current_block))

    if blocks:
        # Use the last block (steady state)
        last_block = blocks[-1]
        # Columns: chunk_id, coord, N_count, T
        z_coord = last_block[:, 1]  # fractional z coordinate
        temp = last_block[:, -1]     # temperature

        # Plot temperature profile
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.plot(z_coord, temp, "ro-", markersize=6)
        ax.axhline(y=TEMPERATURE, color="gray", linestyle="--", linewidth=0.5)
        ax.set_xlabel("z (fractional)", fontsize=12)
        ax.set_ylabel("Temperature (K)", fontsize=12)
        ax.set_title("NEMD Temperature Profile (Steady State)", fontsize=13)
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(os.path.join(WORK_DIR, "temperature_profile.png"), dpi=150)
        plt.close()
        print(f"Saved: {WORK_DIR}/temperature_profile.png")

        # Extract temperature gradient
        # In Muller-Plathe, T profile is symmetric:
        # cold at edges (z=0, z=1), hot at center (z=0.5)
        n_slabs = len(z_coord)
        half = n_slabs // 2

        # Fit gradient in the first half (cold -> hot)
        z_half = z_coord[:half]
        T_half = temp[:half]
        if len(z_half) > 2:
            coeffs = np.polyfit(z_half, T_half, 1)
            dT_dz_frac = coeffs[0]  # K per fractional unit

            print(f"  Temperature gradient (first half): {dT_dz_frac:.2f} K/frac_unit")
            print(f"  This needs to be converted using the actual box length in z.")
            print(f"  kappa = Q / (2 * A * |dT/dz|)")
            print(f"  where Q = total swapped energy / time, A = cross-sectional area")
else:
    print("Temperature profile file not found. Run LAMMPS first.")

# Parse swapped energy from LAMMPS log
log_file = os.path.join(WORK_DIR, "lammps.out")
if os.path.exists(log_file):
    with open(log_file, "r") as f:
        for line in f:
            if "Total energy swapped" in line:
                parts = line.split(":")
                if len(parts) >= 2:
                    try:
                        Q_total = float(parts[-1].strip())
                        total_time_ps = N_PROD * TIMESTEP_PS
                        Q_rate = Q_total / total_time_ps  # eV/ps
                        print(f"  Total swapped energy: {Q_total:.6f} eV")
                        print(f"  Energy swap rate: {Q_rate:.6f} eV/ps")
                    except ValueError:
                        pass
'''

postprocess_file = os.path.join(WORK_DIR, "postprocess_nemd.py")
with open(postprocess_file, "w") as f:
    f.write(postprocess_script)

nprocs = os.cpu_count() or 4
print(f"\n=== Run Commands ===")
print(f"cd {WORK_DIR}")
print(f"mpirun --allow-run-as-root -np {nprocs} lmp -in in.nemd > lammps.out 2>&1")
print(f"python3 postprocess_nemd.py")
```

### Method C: Phonon Boltzmann Transport (phono3py)

```python
#!/usr/bin/env python3
"""
Lattice thermal conductivity via the phonon Boltzmann transport equation.
Uses phono3py with ASE + MACE for force calculations.
Computes 3rd-order force constants and solves the BTE.
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import os

from ase.io import read, write
from ase.optimize import LBFGS
from ase.constraints import ExpCellFilter

from mace.calculators import mace_mp

from pymatgen.core.structure import Structure
from pymatgen.io.ase import AseAtomsAdaptor
from pymatgen.io.phonopy import get_phonopy_structure

import warnings
warnings.filterwarnings("ignore")

# ============================================================
# 1. CONFIGURATION
# ============================================================

STRUCTURE_FILE = "structure.cif"
MACE_MODEL = "medium"
DEVICE = "cpu"

# Supercell for force constants
SUPERCELL_DIM = [2, 2, 2]        # 2x2x2 supercell (adjust based on unit cell size)
DISPLACEMENT = 0.03              # Displacement in A (phono3py default: 0.03)
SYMPREC = 1e-5

# BTE settings
MESH = [11, 11, 11]              # q-point mesh for BTE
TEMPERATURES = [200, 300, 400, 500, 600, 800, 1000]  # K

WORK_DIR = "/tmp/thermal_conductivity_bte"
os.makedirs(WORK_DIR, exist_ok=True)

# ============================================================
# 2. RELAX STRUCTURE
# ============================================================

calc = mace_mp(model=MACE_MODEL, device=DEVICE, default_dtype="float64")

atoms = read(STRUCTURE_FILE)
atoms.calc = calc

print("=== Relaxing structure ===")
ecf = ExpCellFilter(atoms, hydrostatic_strain=False)
opt = LBFGS(ecf, logfile=os.path.join(WORK_DIR, "relax.log"))
opt.run(fmax=1e-5, steps=500)
print(f"  Formula: {atoms.get_chemical_formula()}")
print(f"  Relaxed energy: {atoms.get_potential_energy():.6f} eV")
write(os.path.join(WORK_DIR, "relaxed.cif"), atoms)

# ============================================================
# 3. SET UP PHONO3PY
# ============================================================

try:
    import phono3py
    from phono3py import Phono3py
except ImportError:
    print("phono3py not installed. Install with: pip install phono3py")
    print("This is required for phonon Boltzmann transport calculations.")
    raise SystemExit(1)

adaptor = AseAtomsAdaptor()
pmg_structure = adaptor.get_structure(atoms)
phonopy_structure = get_phonopy_structure(pmg_structure)

ph3 = Phono3py(
    phonopy_structure,
    supercell_matrix=np.diag(SUPERCELL_DIM).tolist(),
    symprec=SYMPREC,
)

# Generate displacements for 2nd and 3rd order force constants
ph3.generate_displacements(distance=DISPLACEMENT)

supercells = ph3.supercells_with_displacements
n_disp = len(supercells)
print(f"\n=== Force Calculations ===")
print(f"  Number of displaced supercells: {n_disp}")
print(f"  (This includes both 2nd and 3rd order displacements)")

if n_disp > 500:
    print(f"  WARNING: {n_disp} displacements is very large.")
    print(f"  Consider using a smaller supercell or higher symmetry structure.")

# ============================================================
# 4. COMPUTE FORCES
# ============================================================

forces_list = []
for i, sc in enumerate(supercells):
    # Convert to ASE atoms
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

    if (i + 1) % 50 == 0 or (i + 1) == n_disp:
        print(f"  Computed forces: {i + 1}/{n_disp}")

# ============================================================
# 5. BUILD FORCE CONSTANTS AND SOLVE BTE
# ============================================================

print("\n=== Building Force Constants ===")
ph3.forces = forces_list
ph3.produce_fc2()
ph3.produce_fc3()
print("  2nd and 3rd order force constants computed.")

# Save for reuse
ph3.save(os.path.join(WORK_DIR, "phono3py_params.yaml"))
print(f"  Saved: {WORK_DIR}/phono3py_params.yaml")

# Run thermal conductivity calculation
print(f"\n=== Solving BTE on {MESH} mesh ===")

ph3.mesh_numbers = MESH
ph3.init_phph_interaction()

kappa_results = {}
for T in TEMPERATURES:
    print(f"  T = {T} K...", end=" ")
    ph3.run_thermal_conductivity(
        temperatures=[T],
        is_isotope=True,         # Include isotope scattering
        boundary_mfp=1e6,        # No boundary scattering (bulk)
    )
    kappa_tensor = ph3.thermal_conductivity  # (1, 6) or (1, 3, 3)

    # kappa is a tensor; extract diagonal elements
    if kappa_tensor.ndim == 2:
        # Shape: (n_temps, 6) in Voigt notation: xx, yy, zz, yz, xz, xy
        kappa_xx = kappa_tensor[0, 0]
        kappa_yy = kappa_tensor[0, 1]
        kappa_zz = kappa_tensor[0, 2]
    elif kappa_tensor.ndim == 3:
        kappa_xx = kappa_tensor[0, 0, 0]
        kappa_yy = kappa_tensor[0, 1, 1]
        kappa_zz = kappa_tensor[0, 2, 2]
    else:
        kappa_xx = kappa_yy = kappa_zz = float(kappa_tensor)

    kappa_avg = (kappa_xx + kappa_yy + kappa_zz) / 3.0
    kappa_results[T] = {
        "xx": kappa_xx, "yy": kappa_yy, "zz": kappa_zz, "avg": kappa_avg
    }
    print(f"kappa = {kappa_avg:.2f} W/(m*K) (xx={kappa_xx:.2f}, yy={kappa_yy:.2f}, zz={kappa_zz:.2f})")

# ============================================================
# 6. PLOT AND SAVE RESULTS
# ============================================================

temps = sorted(kappa_results.keys())
kappa_avg = [kappa_results[T]["avg"] for T in temps]
kappa_xx = [kappa_results[T]["xx"] for T in temps]
kappa_yy = [kappa_results[T]["yy"] for T in temps]
kappa_zz = [kappa_results[T]["zz"] for T in temps]

fig, ax = plt.subplots(figsize=(7, 5))
ax.plot(temps, kappa_avg, "ko-", linewidth=2, markersize=8, label="Average")
ax.plot(temps, kappa_xx, "r--", linewidth=1, marker="s", markersize=5, label="$\\kappa_{xx}$")
ax.plot(temps, kappa_yy, "g--", linewidth=1, marker="^", markersize=5, label="$\\kappa_{yy}$")
ax.plot(temps, kappa_zz, "b--", linewidth=1, marker="v", markersize=5, label="$\\kappa_{zz}$")

ax.set_xlabel("Temperature (K)", fontsize=12)
ax.set_ylabel("$\\kappa_L$ (W/(m*K))", fontsize=12)
ax.set_title(f"Lattice Thermal Conductivity - {atoms.get_chemical_formula()}", fontsize=13)
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3)
ax.set_ylim(bottom=0)

# Add 1/T reference line
if len(temps) > 1:
    T_ref = temps[len(temps) // 2]
    k_ref = kappa_avg[len(temps) // 2]
    T_arr = np.linspace(min(temps), max(temps), 100)
    ax.plot(T_arr, k_ref * T_ref / T_arr, ":", color="gray", linewidth=1, label="$\\sim 1/T$")
    ax.legend(fontsize=10)

fig.tight_layout()
fig.savefig(os.path.join(WORK_DIR, "thermal_conductivity_bte.png"), dpi=150, bbox_inches="tight")
plt.close()
print(f"\n  Saved: {WORK_DIR}/thermal_conductivity_bte.png")

# Save table
with open(os.path.join(WORK_DIR, "kappa_summary.txt"), "w") as f:
    f.write(f"# Lattice thermal conductivity: {atoms.get_chemical_formula()}\n")
    f.write(f"# Method: phono3py BTE, mesh: {MESH}\n")
    f.write(f"# MACE model: {MACE_MODEL}\n\n")
    f.write(f"{'T(K)':>8s} {'kappa_avg':>10s} {'kappa_xx':>10s} {'kappa_yy':>10s} {'kappa_zz':>10s}  (W/m/K)\n")
    f.write("-" * 55 + "\n")
    for T in temps:
        k = kappa_results[T]
        f.write(f"{T:8.0f} {k['avg']:10.2f} {k['xx']:10.2f} {k['yy']:10.2f} {k['zz']:10.2f}\n")

np.savetxt(os.path.join(WORK_DIR, "kappa_vs_T.dat"),
           np.column_stack([temps, kappa_avg, kappa_xx, kappa_yy, kappa_zz]),
           header="T(K)  kappa_avg  kappa_xx  kappa_yy  kappa_zz  (W/m/K)",
           fmt="%8.0f %10.4f %10.4f %10.4f %10.4f")

print(f"\nAll outputs in: {WORK_DIR}/")
```

## Key Parameters

| Parameter | Default | Notes |
|---|---|---|
| **Green-Kubo (Method A)** | | |
| `N_PROD` | 100000 steps | Need long NVE runs for converged HFACF. 100+ ps minimum; 1 ns preferred. |
| `SUPERCELL` | (4,4,4) | Large supercells needed. Minimum ~500 atoms; 1000+ preferred. |
| `MAX_CORRELATION_TIME_PS` | 10 ps | Max lag for HFACF. Should exceed phonon lifetime. |
| NVE vs NVT | NVE required | Thermostat corrupts heat flux correlations. Use NVE after NVT equilibration. |
| **NEMD (Method B)** | | |
| `SUPERCELL` z-direction | 20+ cells | Long z-dimension for establishing temperature gradient. |
| `SWAP_INTERVAL` | 100 | Energy swap frequency. Too frequent = too large gradient; too rare = noisy signal. |
| `N_PROD` | 200000 | Need steady state. Allow 50+ ps for equilibration of temperature profile. |
| **phono3py (Method C)** | | |
| `SUPERCELL_DIM` | [2,2,2] | Supercell for force constants. Larger = more accurate but exponentially more displacements. |
| `DISPLACEMENT` | 0.03 A | Phono3py default. Larger than harmonic (0.01) because 3rd order needs wider sampling. |
| `MESH` | [11,11,11] | q-mesh for BTE. Converge by increasing; 11x11x11 is a good starting point. |
| `is_isotope` | True | Include natural isotope scattering. Important for Ge, Ga, etc. |

### System size convergence

- **Green-Kubo**: kappa converges slowly with system size. Test at least 2 cell sizes.
- **NEMD**: kappa depends on simulation length (finite-size effect). Extrapolate 1/kappa vs 1/L to L=infinity.
- **phono3py**: Converge with mesh density and supercell size. 2x2x2 is minimum; 3x3x3 is better for accurate results.

## Interpreting Results

### Typical thermal conductivity values
- **Diamond**: ~2000 W/(m*K) -- highest known for bulk crystals
- **Silicon**: ~150 W/(m*K) at 300 K
- **Metals (Cu, Al)**: 200-400 W/(m*K) (but mostly electronic, not lattice)
- **Oxide ceramics (Al2O3)**: 30-40 W/(m*K)
- **Thermoelectrics (Bi2Te3)**: 1-2 W/(m*K)
- **Amorphous SiO2 (glass)**: ~1 W/(m*K)
- **Minimum thermal conductivity**: ~0.1-1 W/(m*K) (Cahill-Pohl limit)

### Temperature dependence
- **Crystalline solids**: kappa ~ 1/T at high T (Umklapp scattering dominates).
- **Disordered/amorphous**: kappa weakly T-dependent (already at minimum).
- **Below Debye temperature**: kappa increases with T (more phonons excited to carry heat).
- **Peak in kappa(T)**: Occurs near 0.1 * Debye temperature for pure crystals.

### Anisotropy
- Layered materials (graphite, MoS2): kappa_in-plane >> kappa_cross-plane.
- Chain structures: kappa_along-chain >> kappa_perpendicular.
- Cubic systems: kappa_xx = kappa_yy = kappa_zz (isotropic by symmetry).

### Green-Kubo convergence
- kappa_running should plateau. If it diverges, the run is too short.
- Multiple independent runs help establish error bars.
- HFACF should decay to zero within the correlation window.

## Common Issues

| Issue | Cause | Fix |
|---|---|---|
| Green-Kubo kappa does not converge | Run too short or system too small | Run 1+ ns; use 1000+ atoms; average over 5+ independent runs |
| Green-Kubo kappa negative | Noisy HFACF; statistical artifact | Longer runs; check NVE energy drift is small |
| NEMD temperature profile not linear | Not at steady state; boundary effects | Run longer equilibration; use larger system in transport direction |
| NEMD kappa depends on system length | Finite-size effect (expected) | Run at 3+ lengths; extrapolate 1/kappa vs 1/L to 1/L=0 |
| phono3py too many displacements | Large unit cell or low symmetry | Use smaller supercell (2x2x2); increase SYMPREC to find more symmetry |
| phono3py kappa too high | Supercell too small; insufficient q-mesh | Increase SUPERCELL_DIM; increase MESH; check force constant convergence |
| phono3py kappa too low | MACE forces inaccurate for 3rd-order FC | Use DFT forces for critical systems; try MACE "large" model |
| kappa does not follow 1/T | Strong anharmonicity or grain-boundary scattering | Expected for complex or defective materials; MD methods capture this naturally |
| Heat flux computation wrong | Per-atom energy decomposition approximate | For many-body potentials, the Hardy/Irving-Kirkwood decomposition is approximate. Use validated implementations. |
| LAMMPS thermal/conductivity errors | Incompatible fix combinations | Check LAMMPS documentation for fix compatibility; use NVE during NEMD |
