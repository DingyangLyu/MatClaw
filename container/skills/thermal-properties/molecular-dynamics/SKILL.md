# Molecular Dynamics Simulations

## When to Use

- Computing diffusion coefficients (MSD analysis)
- Studying melting, phase transitions, or amorphization
- Generating equilibrated liquid or disordered structures
- Computing radial distribution functions (RDF) and structural analysis
- Temperature-dependent property sampling beyond harmonic approximation
- Simulating annealing, quenching, or thermal cycling protocols
- Validating potential energy surface sampling with MLIP

## Method Selection

```
What is the system and goal?

Small system (<500 atoms) with MACE-compatible chemistry?
  YES --> Method A: ASE + MACE MD (easy setup, accurate PES)
  NO  --> Is a validated classical potential available (EAM, Tersoff, ReaxFF)?
            YES --> Method B: LAMMPS (fast, large systems, long timescales)
            NO  --> Use ASE + MACE (or train/use another MLIP)

Need > 1 ns timescale or > 10k atoms?
  --> LAMMPS with classical potential (Method B)

Need accurate energetics for complex chemistry (oxides, interfaces)?
  --> ASE + MACE (Method A)
```

## Prerequisites

Pre-installed: `ase`, `mace-torch`, `pymatgen`, `numpy`, `scipy`, `matplotlib`.

LAMMPS binary: `lmp` (with common packages: MANYBODY, KSPACE, MOLECULE, etc.).

For LAMMPS EAM potentials, download from NIST Interatomic Potentials Repository or use OpenKIM.

## Detailed Steps

### Method A: ASE + MACE MD

#### Complete NVT/NVE/NPT example

```python
#!/usr/bin/env python3
"""
Molecular dynamics using ASE + MACE.
Supports NVE, NVT (Langevin, Nose-Hoover), and NPT ensembles.
Complete runnable script with trajectory analysis.
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
from ase.md.verlet import VelocityVerlet
from ase.md.langevin import Langevin
from ase.md.nptberendsen import NPTBerendsen
from ase import units

from mace.calculators import mace_mp

import warnings
warnings.filterwarnings("ignore")

# ============================================================
# 1. CONFIGURATION
# ============================================================

STRUCTURE_FILE = "structure.cif"       # Input structure
MACE_MODEL = "medium"                  # "small", "medium", "large"
DEVICE = "cpu"                         # "cpu" or "cuda"

# MD settings
ENSEMBLE = "nvt"                       # "nve", "nvt", or "npt"
TEMPERATURE = 300.0                    # Kelvin
PRESSURE = 0.0                         # GPa (only for NPT)
TIMESTEP = 1.0                         # femtoseconds (use 0.5 if H present)
N_EQUIL = 500                          # Equilibration steps
N_PROD = 2000                          # Production steps
TRAJ_INTERVAL = 10                     # Save trajectory every N steps
LOG_INTERVAL = 10                      # Print thermodynamic info every N steps

# Supercell size for MD (need enough atoms for meaningful statistics)
SUPERCELL = (3, 3, 3)                  # Adjust based on unit cell size

WORK_DIR = "/tmp/ase_md"
os.makedirs(WORK_DIR, exist_ok=True)

# ============================================================
# 2. SET UP CALCULATOR AND STRUCTURE
# ============================================================

calc = mace_mp(model=MACE_MODEL, device=DEVICE, default_dtype="float64")

atoms = read(STRUCTURE_FILE)

# Quick relaxation of unit cell first
atoms.calc = calc
ecf = ExpCellFilter(atoms)
opt = LBFGS(ecf, logfile=os.path.join(WORK_DIR, "relax.log"))
opt.run(fmax=0.01, steps=200)
print(f"Relaxed unit cell: {atoms.get_chemical_formula()}")
print(f"  Energy: {atoms.get_potential_energy():.4f} eV")

# Build supercell
atoms = atoms.repeat(SUPERCELL)
atoms.calc = calc
n_atoms = len(atoms)
print(f"Supercell: {SUPERCELL}, {n_atoms} atoms")

# ============================================================
# 3. INITIALIZE VELOCITIES
# ============================================================

MaxwellBoltzmannDistribution(atoms, temperature_K=TEMPERATURE)
Stationary(atoms)  # Remove center-of-mass velocity

print(f"\nInitial temperature: {atoms.get_temperature():.1f} K (target: {TEMPERATURE} K)")
print(f"Initial kinetic energy: {atoms.get_kinetic_energy():.4f} eV")

# ============================================================
# 4. SET UP MD INTEGRATOR
# ============================================================

traj_file = os.path.join(WORK_DIR, "md.traj")
log_file = os.path.join(WORK_DIR, "md.log")

dt = TIMESTEP * units.fs

if ENSEMBLE == "nve":
    print("\n=== NVE Ensemble (Microcanonical) ===")
    dyn = VelocityVerlet(atoms, timestep=dt)

elif ENSEMBLE == "nvt":
    # Langevin thermostat
    # friction = 1/(relaxation_time). Typical: 0.01-0.1 fs^-1 -> 10-100 fs relaxation
    friction = 0.01 / units.fs  # 100 fs relaxation time
    print(f"\n=== NVT Ensemble (Langevin) ===")
    print(f"  Friction: {friction * units.fs:.4f} fs^-1 (relaxation: {1/(friction * units.fs):.0f} fs)")
    dyn = Langevin(
        atoms,
        timestep=dt,
        temperature_K=TEMPERATURE,
        friction=friction,
    )

elif ENSEMBLE == "npt":
    # Berendsen NPT
    pressure_au = PRESSURE * units.GPa  # Convert GPa to ASE internal units
    print(f"\n=== NPT Ensemble (Berendsen) ===")
    print(f"  Pressure: {PRESSURE} GPa")
    dyn = NPTBerendsen(
        atoms,
        timestep=dt,
        temperature_K=TEMPERATURE,
        pressure_au=pressure_au,
        taut=100 * units.fs,     # T coupling time
        taup=1000 * units.fs,    # P coupling time
        compressibility_au=4.57e-5 / units.bar,  # typical solid
    )
else:
    raise ValueError(f"Unknown ensemble: {ENSEMBLE}")

# ============================================================
# 5. ATTACH OBSERVERS
# ============================================================

# Trajectory writer
traj = Trajectory(traj_file, "w", atoms)
dyn.attach(traj.write, interval=TRAJ_INTERVAL)

# Thermodynamic logger
thermo_data = {"step": [], "time_ps": [], "T": [], "Epot": [], "Ekin": [], "Etot": []}

def log_thermo():
    step = dyn.nsteps
    time_ps = step * TIMESTEP / 1000.0
    T = atoms.get_temperature()
    Epot = atoms.get_potential_energy()
    Ekin = atoms.get_kinetic_energy()
    Etot = Epot + Ekin

    thermo_data["step"].append(step)
    thermo_data["time_ps"].append(time_ps)
    thermo_data["T"].append(T)
    thermo_data["Epot"].append(Epot)
    thermo_data["Ekin"].append(Ekin)
    thermo_data["Etot"].append(Etot)

    if step % (LOG_INTERVAL * 10) == 0:
        print(f"  Step {step:6d} | t={time_ps:.3f} ps | T={T:.1f} K | "
              f"Epot={Epot:.4f} eV | Etot={Etot:.4f} eV")

dyn.attach(log_thermo, interval=LOG_INTERVAL)

# ============================================================
# 6. RUN EQUILIBRATION
# ============================================================

print(f"\n--- Equilibration: {N_EQUIL} steps ({N_EQUIL * TIMESTEP / 1000:.2f} ps) ---")
dyn.run(N_EQUIL)

equil_T_mean = np.mean(thermo_data["T"][-N_EQUIL // LOG_INTERVAL:])
print(f"  Mean T during last half of equilibration: {equil_T_mean:.1f} K")

# Mark start of production
prod_start_idx = len(thermo_data["step"])

# ============================================================
# 7. RUN PRODUCTION
# ============================================================

print(f"\n--- Production: {N_PROD} steps ({N_PROD * TIMESTEP / 1000:.2f} ps) ---")
dyn.run(N_PROD)

traj.close()

# ============================================================
# 8. SAVE THERMODYNAMIC DATA
# ============================================================

thermo_arr = np.column_stack([
    thermo_data["step"], thermo_data["time_ps"],
    thermo_data["T"], thermo_data["Epot"],
    thermo_data["Ekin"], thermo_data["Etot"],
])
np.savetxt(
    os.path.join(WORK_DIR, "thermo.dat"), thermo_arr,
    header="step  time_ps  T(K)  Epot(eV)  Ekin(eV)  Etot(eV)",
    fmt="%10d %12.5f %10.2f %14.6f %14.6f %14.6f",
)
print(f"\nThermodynamic data saved to {WORK_DIR}/thermo.dat")

# ============================================================
# 9. PLOT EQUILIBRATION CHECK
# ============================================================

fig, axes = plt.subplots(2, 2, figsize=(12, 8))

steps = np.array(thermo_data["step"])
time_ps = np.array(thermo_data["time_ps"])

# Temperature
axes[0, 0].plot(time_ps, thermo_data["T"], "r-", alpha=0.7, linewidth=0.5)
axes[0, 0].axhline(y=TEMPERATURE, color="k", linestyle="--", label=f"Target: {TEMPERATURE} K")
axes[0, 0].axvline(x=N_EQUIL * TIMESTEP / 1000, color="gray", linestyle=":", label="Prod start")
axes[0, 0].set_xlabel("Time (ps)")
axes[0, 0].set_ylabel("Temperature (K)")
axes[0, 0].legend()
axes[0, 0].set_title("Temperature")

# Potential energy
axes[0, 1].plot(time_ps, thermo_data["Epot"], "b-", alpha=0.7, linewidth=0.5)
axes[0, 1].axvline(x=N_EQUIL * TIMESTEP / 1000, color="gray", linestyle=":")
axes[0, 1].set_xlabel("Time (ps)")
axes[0, 1].set_ylabel("Potential Energy (eV)")
axes[0, 1].set_title("Potential Energy")

# Total energy
axes[1, 0].plot(time_ps, thermo_data["Etot"], "g-", alpha=0.7, linewidth=0.5)
axes[1, 0].axvline(x=N_EQUIL * TIMESTEP / 1000, color="gray", linestyle=":")
axes[1, 0].set_xlabel("Time (ps)")
axes[1, 0].set_ylabel("Total Energy (eV)")
axes[1, 0].set_title("Total Energy")

# Total energy drift (important for NVE, should be flat for NVT)
etot = np.array(thermo_data["Etot"])
drift = (etot - etot[0]) / n_atoms  # eV/atom
axes[1, 1].plot(time_ps, drift * 1000, "m-", alpha=0.7, linewidth=0.5)
axes[1, 1].axvline(x=N_EQUIL * TIMESTEP / 1000, color="gray", linestyle=":")
axes[1, 1].set_xlabel("Time (ps)")
axes[1, 1].set_ylabel("Energy drift (meV/atom)")
axes[1, 1].set_title("Energy Drift")

fig.suptitle(f"MD Equilibration Check - {ENSEMBLE.upper()} at {TEMPERATURE} K")
fig.tight_layout()
fig.savefig(os.path.join(WORK_DIR, "equilibration_check.png"), dpi=150, bbox_inches="tight")
plt.close()
print(f"Saved: {WORK_DIR}/equilibration_check.png")

print(f"\n=== Production Run Statistics ===")
prod_T = thermo_data["T"][prod_start_idx:]
prod_Epot = thermo_data["Epot"][prod_start_idx:]
print(f"  Temperature: {np.mean(prod_T):.1f} +/- {np.std(prod_T):.1f} K")
print(f"  Epot:        {np.mean(prod_Epot):.4f} +/- {np.std(prod_Epot):.4f} eV")
print(f"  Trajectory:  {traj_file}")
```

#### Temperature ramping / annealing protocol

```python
#!/usr/bin/env python3
"""
Temperature annealing protocol with ASE + MACE.
Heat from T_start -> T_max, then cool back to T_start.
"""

import os
import numpy as np
from ase.io import read, Trajectory
from ase.md.langevin import Langevin
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution, Stationary
from ase import units
from mace.calculators import mace_mp

import warnings
warnings.filterwarnings("ignore")

STRUCTURE_FILE = "structure.cif"
WORK_DIR = "/tmp/ase_anneal"
os.makedirs(WORK_DIR, exist_ok=True)

# Annealing schedule: list of (temperature_K, n_steps)
SCHEDULE = [
    (300,  500),    # Equilibrate at 300 K
    (600,  500),    # Heat to 600 K
    (900,  500),    # Heat to 900 K
    (1200, 1000),   # Hold at 1200 K (high T dwell)
    (900,  500),    # Cool to 900 K
    (600,  500),    # Cool to 600 K
    (300,  1000),   # Cool to 300 K and equilibrate
]

TIMESTEP = 1.0  # fs
SUPERCELL = (3, 3, 3)

calc = mace_mp(model="medium", device="cpu", default_dtype="float64")

atoms = read(STRUCTURE_FILE)
atoms = atoms.repeat(SUPERCELL)
atoms.calc = calc

MaxwellBoltzmannDistribution(atoms, temperature_K=SCHEDULE[0][0])
Stationary(atoms)

friction = 0.01 / units.fs
dyn = Langevin(atoms, timestep=TIMESTEP * units.fs, temperature_K=SCHEDULE[0][0], friction=friction)

traj = Trajectory(os.path.join(WORK_DIR, "anneal.traj"), "w", atoms)
dyn.attach(traj.write, interval=20)

anneal_log = []

def record():
    anneal_log.append((dyn.nsteps, atoms.get_temperature(), atoms.get_potential_energy()))

dyn.attach(record, interval=10)

total_step = 0
for target_T, n_steps in SCHEDULE:
    print(f"  Ramping to {target_T} K for {n_steps} steps...")
    dyn.set_temperature(temperature_K=target_T)
    dyn.run(n_steps)
    total_step += n_steps
    print(f"    Step {total_step}: T={atoms.get_temperature():.1f} K, Epot={atoms.get_potential_energy():.4f} eV")

traj.close()

# Plot annealing profile
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

log = np.array(anneal_log)
fig, ax1 = plt.subplots(figsize=(8, 4))
ax1.plot(log[:, 0] * TIMESTEP / 1000, log[:, 1], "r-", alpha=0.5, label="Instantaneous T")
ax1.set_xlabel("Time (ps)")
ax1.set_ylabel("Temperature (K)", color="r")

ax2 = ax1.twinx()
ax2.plot(log[:, 0] * TIMESTEP / 1000, log[:, 2], "b-", alpha=0.5, label="Epot")
ax2.set_ylabel("Potential Energy (eV)", color="b")

fig.suptitle("Annealing Protocol")
fig.tight_layout()
fig.savefig(os.path.join(WORK_DIR, "anneal_profile.png"), dpi=150)
plt.close()
print(f"Saved: {WORK_DIR}/anneal_profile.png")
```

### Method B: LAMMPS MD with Classical Potentials

#### LAMMPS input file template (EAM example for Cu)

```python
#!/usr/bin/env python3
"""
Generate LAMMPS input files for MD simulation.
Example: EAM potential for FCC Cu.
"""

import os
import subprocess
import numpy as np
from pymatgen.core.structure import Structure
from pymatgen.core.lattice import Lattice

WORK_DIR = "/tmp/lammps_md"
os.makedirs(WORK_DIR, exist_ok=True)

# ============================================================
# 1. GENERATE STRUCTURE (FCC Cu as example)
# ============================================================

a = 3.615  # Cu lattice constant in Angstrom
lattice = Lattice.cubic(a)
structure = Structure(lattice, ["Cu"], [[0, 0, 0]])

# Replicate to desired supercell
structure.make_supercell([5, 5, 5])
print(f"Structure: {structure.composition}, {len(structure)} atoms")

# Write LAMMPS data file
from pymatgen.io.lammps.data import LammpsData

lammps_data = LammpsData.from_structure(structure, atom_style="atomic")
data_file = os.path.join(WORK_DIR, "structure.data")
lammps_data.write_file(data_file)
print(f"Written: {data_file}")

# ============================================================
# 2. DOWNLOAD EAM POTENTIAL
# ============================================================

# Mishin 2001 Cu EAM potential from NIST
pot_url = "https://www.ctcms.nist.gov/potentials/Download/2001--Mishin-Y-Mehl-M-J-Papaconstantopoulos-D-A--Cu/2/Cu01.eam.alloy"
pot_file = os.path.join(WORK_DIR, "Cu01.eam.alloy")

if not os.path.exists(pot_file):
    print("Downloading EAM potential...")
    subprocess.run(["wget", "-q", pot_url, "-O", pot_file], check=True)

# ============================================================
# 3. WRITE LAMMPS INPUT FILE
# ============================================================

TEMPERATURE = 300.0    # K
PRESSURE = 0.0         # bar
TIMESTEP = 2.0         # fs
N_EQUIL = 5000         # Equilibration steps
N_PROD = 20000         # Production steps
THERMO_INTERVAL = 100
DUMP_INTERVAL = 200

lammps_input = f"""# LAMMPS MD simulation - Cu EAM
# NVT equilibration followed by NPT production

# ============ Initialization ============
units           metal
dimension       3
boundary        p p p
atom_style      atomic

# ============ Read Structure ============
read_data       structure.data

# ============ Potential ============
pair_style      eam/alloy
pair_coeff      * * Cu01.eam.alloy Cu

# ============ Settings ============
neighbor        2.0 bin
neigh_modify    every 1 delay 0 check yes

timestep        {TIMESTEP / 1000.0}
# Note: LAMMPS 'metal' units use ps for timestep, so convert fs -> ps

# ============ Thermodynamic Output ============
thermo          {THERMO_INTERVAL}
thermo_style    custom step time temp pe ke etotal press vol density lx ly lz

# ============ Minimization (optional) ============
minimize        1.0e-6 1.0e-8 1000 10000
reset_timestep  0

# ============ Initialize Velocities ============
velocity        all create {TEMPERATURE} 12345 dist gaussian

# ============ NVT Equilibration ============
fix             eq_nvt all nvt temp {TEMPERATURE} {TEMPERATURE} $(100*dt)
run             {N_EQUIL}
unfix           eq_nvt

# ============ NPT Production ============
reset_timestep  0

# Trajectory dump
dump            prod all custom {DUMP_INTERVAL} dump.lammpstrj id type x y z vx vy vz fx fy fz
dump_modify     prod sort id

# Log per-atom data for MSD
compute         msd_all all msd com yes
fix             msd_out all ave/time 1 {THERMO_INTERVAL} {THERMO_INTERVAL} &
                c_msd_all[1] c_msd_all[2] c_msd_all[3] c_msd_all[4] &
                file msd.dat title1 "# Step MSD_x MSD_y MSD_z MSD_total"

# RDF computation
compute         rdf_all all rdf 200 1 1
fix             rdf_out all ave/time 100 {N_PROD // THERMO_INTERVAL} {N_PROD} &
                c_rdf_all[*] file rdf.dat mode vector title1 "# RDF: r g(r)"

# Production run: NPT
fix             prod_npt all npt temp {TEMPERATURE} {TEMPERATURE} $(100*dt) &
                iso {PRESSURE} {PRESSURE} $(1000*dt)

thermo_style    custom step time temp pe ke etotal press vol density

run             {N_PROD}

# ============ Final Output ============
write_data      final.data
write_dump      all custom final.lammpstrj id type x y z modify sort id

print "=== MD Complete ==="
"""

input_file = os.path.join(WORK_DIR, "in.md")
with open(input_file, "w") as f:
    f.write(lammps_input)
print(f"Written: {input_file}")

# ============================================================
# 4. PRINT RUN COMMAND
# ============================================================

nprocs = os.cpu_count() or 4
print(f"\n=== Run Command ===")
print(f"cd {WORK_DIR} && mpirun --allow-run-as-root -np {nprocs} lmp -in in.md > lammps.out 2>&1")
```

#### Run LAMMPS

```bash
cd /tmp/lammps_md
NPROCS=$(nproc)
mpirun --allow-run-as-root -np $NPROCS lmp -in in.md > lammps.out 2>&1
echo "Exit code: $?"
tail -5 lammps.out
```

### Post-Processing: RDF, MSD, Diffusion

```python
#!/usr/bin/env python3
"""
Post-process MD trajectory: RDF, MSD, diffusion coefficient.
Works with both ASE trajectories and LAMMPS dump files.
"""

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ============================================================
# Choose data source
# ============================================================

USE_ASE_TRAJ = True  # Set False to parse LAMMPS output files

ASE_TRAJ_FILE = "/tmp/ase_md/md.traj"
LAMMPS_MSD_FILE = "/tmp/lammps_md/msd.dat"
LAMMPS_RDF_FILE = "/tmp/lammps_md/rdf.dat"
LAMMPS_THERMO_FILE = "/tmp/lammps_md/lammps.out"

OUTPUT_DIR = "/tmp/md_analysis"
os.makedirs(OUTPUT_DIR, exist_ok=True)

TIMESTEP_FS = 1.0          # fs per MD step (for ASE)
TRAJ_INTERVAL = 10         # Steps between saved frames (for ASE)
N_EQUIL_FRAMES = 50        # Number of equilibration frames to skip
TEMPERATURE = 300.0        # K (for labeling)

if USE_ASE_TRAJ:
    # ============================================================
    # ASE Trajectory Analysis
    # ============================================================

    from ase.io import Trajectory

    traj = Trajectory(ASE_TRAJ_FILE, "r")
    all_frames = [atoms.copy() for atoms in traj]
    traj.close()

    print(f"Total frames: {len(all_frames)}")
    print(f"Atoms per frame: {len(all_frames[0])}")

    # Skip equilibration
    frames = all_frames[N_EQUIL_FRAMES:]
    n_frames = len(frames)
    n_atoms = len(frames[0])
    print(f"Production frames: {n_frames}")

    # ------ RDF ------
    print("\nComputing RDF...")

    def compute_rdf(frames, r_max=8.0, n_bins=200):
        """Compute pair RDF from ASE frames."""
        dr = r_max / n_bins
        r_edges = np.linspace(0, r_max, n_bins + 1)
        r_centers = 0.5 * (r_edges[:-1] + r_edges[1:])
        hist = np.zeros(n_bins)

        for atoms in frames:
            cell = atoms.get_cell()
            volume = atoms.get_volume()
            positions = atoms.get_positions()
            n = len(atoms)
            rho = n / volume

            # Use ASE neighbor list for efficiency
            from ase.neighborlist import neighbor_list
            i_list, j_list, d_list = neighbor_list("ijd", atoms, cutoff=r_max)

            h, _ = np.histogram(d_list, bins=r_edges)
            # Each pair counted once by neighbor_list, normalize
            shell_vol = 4 * np.pi * r_centers**2 * dr
            norm = len(frames) * n * rho * shell_vol
            hist += h

        g_r = hist / norm
        return r_centers, g_r

    r, g_r = compute_rdf(frames, r_max=8.0, n_bins=200)

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(r, g_r, "b-")
    ax.axhline(y=1.0, color="gray", linestyle="--", linewidth=0.5)
    ax.set_xlabel("r (A)")
    ax.set_ylabel("g(r)")
    ax.set_title(f"Radial Distribution Function - {TEMPERATURE} K")
    ax.set_xlim(0, 8)
    fig.tight_layout()
    fig.savefig(os.path.join(OUTPUT_DIR, "rdf.png"), dpi=150)
    plt.close()
    print(f"  Saved: {OUTPUT_DIR}/rdf.png")

    # ------ MSD and Diffusion ------
    print("\nComputing MSD...")

    def compute_msd(frames, max_lag=None):
        """
        Compute mean square displacement using the direct method.
        Returns time_lags (in frames) and MSD values (A^2).
        """
        n_frames = len(frames)
        if max_lag is None:
            max_lag = n_frames // 2
        max_lag = min(max_lag, n_frames - 1)

        # Extract all positions, unwrap PBC using displacement tracking
        # Simple approach: track displacement from initial frame
        pos_0 = frames[0].get_positions()
        cell_0 = frames[0].get_cell()
        n_atoms = len(frames[0])

        # For wrapped coordinates, compute MSD carefully
        # Method: use scaled coordinates and unwrap
        positions = np.zeros((n_frames, n_atoms, 3))
        positions[0] = pos_0.copy()

        for i in range(1, n_frames):
            pos_curr = frames[i].get_positions()
            pos_prev = frames[i - 1].get_positions()
            cell = frames[i].get_cell()

            # Compute displacements and apply minimum image convention
            disp = pos_curr - pos_prev
            # Minimum image convention
            scaled_disp = np.linalg.solve(cell.T, disp.T).T
            scaled_disp -= np.round(scaled_disp)
            real_disp = scaled_disp @ cell

            positions[i] = positions[i - 1] + real_disp

        # Compute MSD
        msd = np.zeros(max_lag)
        counts = np.zeros(max_lag)

        for lag in range(1, max_lag + 1):
            disp = positions[lag:] - positions[:-lag]  # (n_frames - lag, n_atoms, 3)
            sq_disp = np.sum(disp**2, axis=2)  # (n_frames - lag, n_atoms)
            msd[lag - 1] = np.mean(sq_disp)
            counts[lag - 1] = sq_disp.size

        time_lags = np.arange(1, max_lag + 1)
        return time_lags, msd

    time_lags, msd = compute_msd(frames, max_lag=n_frames // 2)
    time_ps = time_lags * TRAJ_INTERVAL * TIMESTEP_FS / 1000.0  # Convert to ps

    # Fit diffusion coefficient: MSD = 6*D*t (3D)
    # Use linear region (skip initial ballistic and final noisy parts)
    fit_start = len(time_ps) // 5
    fit_end = len(time_ps) * 4 // 5

    if fit_end > fit_start + 5:
        coeffs = np.polyfit(time_ps[fit_start:fit_end], msd[fit_start:fit_end], 1)
        D = coeffs[0] / 6.0  # A^2/ps
        D_cm2_s = D * 1e-8  # Convert A^2/ps to cm^2/s (* 1e-16 / 1e-12 = 1e-4... let me be precise)
        # 1 A^2/ps = 1e-20 m^2 / 1e-12 s = 1e-8 m^2/s = 1e-4 cm^2/s
        D_cm2_s = D * 1e-4
        print(f"  Diffusion coefficient: D = {D:.4f} A^2/ps = {D_cm2_s:.4e} cm^2/s")
    else:
        D = None
        print("  Not enough data points for diffusion fit.")

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(time_ps, msd, "b-", label="MSD")
    if D is not None:
        fit_line = 6 * D * time_ps
        ax.plot(time_ps, fit_line, "r--", label=f"Fit: D={D:.4f} A$^2$/ps")
    ax.set_xlabel("Time (ps)")
    ax.set_ylabel("MSD (A$^2$)")
    ax.set_title(f"Mean Square Displacement - {TEMPERATURE} K")
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(OUTPUT_DIR, "msd.png"), dpi=150)
    plt.close()
    print(f"  Saved: {OUTPUT_DIR}/msd.png")

else:
    # ============================================================
    # LAMMPS Output Analysis
    # ============================================================

    # ------ Parse LAMMPS MSD ------
    if os.path.exists(LAMMPS_MSD_FILE):
        print("Parsing LAMMPS MSD data...")
        data = np.loadtxt(LAMMPS_MSD_FILE, comments="#")
        # Columns: step, MSD_x, MSD_y, MSD_z, MSD_total
        steps = data[:, 0]
        msd_total = data[:, 4]  # Total MSD in A^2

        # Convert steps to time
        # LAMMPS metal units: timestep is in ps
        lammps_timestep_ps = 0.002  # 2 fs = 0.002 ps
        time_ps = steps * lammps_timestep_ps

        fig, ax = plt.subplots(figsize=(6, 4))
        ax.plot(time_ps, msd_total, "b-")
        ax.set_xlabel("Time (ps)")
        ax.set_ylabel("MSD (A$^2$)")
        ax.set_title("LAMMPS MSD")
        fig.tight_layout()
        fig.savefig(os.path.join(OUTPUT_DIR, "lammps_msd.png"), dpi=150)
        plt.close()
        print(f"  Saved: {OUTPUT_DIR}/lammps_msd.png")

        # Fit diffusion
        n = len(time_ps)
        fit_start = n // 5
        fit_end = n * 4 // 5
        if fit_end > fit_start + 5:
            coeffs = np.polyfit(time_ps[fit_start:fit_end], msd_total[fit_start:fit_end], 1)
            D = coeffs[0] / 6.0
            D_cm2_s = D * 1e-4
            print(f"  D = {D:.4f} A^2/ps = {D_cm2_s:.4e} cm^2/s")

    # ------ Parse LAMMPS RDF ------
    if os.path.exists(LAMMPS_RDF_FILE):
        print("\nParsing LAMMPS RDF data...")
        # LAMMPS rdf output format can vary; parse carefully
        data = np.loadtxt(LAMMPS_RDF_FILE, comments="#")
        if data.ndim == 2 and data.shape[1] >= 3:
            r = data[:, 1]
            g_r = data[:, 2]

            fig, ax = plt.subplots(figsize=(6, 4))
            ax.plot(r, g_r, "b-")
            ax.axhline(y=1.0, color="gray", linestyle="--")
            ax.set_xlabel("r (A)")
            ax.set_ylabel("g(r)")
            ax.set_title("LAMMPS RDF")
            fig.tight_layout()
            fig.savefig(os.path.join(OUTPUT_DIR, "lammps_rdf.png"), dpi=150)
            plt.close()
            print(f"  Saved: {OUTPUT_DIR}/lammps_rdf.png")

    # ------ Parse LAMMPS thermo output ------
    print("\nParsing LAMMPS thermo log...")

    def parse_lammps_log(filename):
        """Extract thermo data from LAMMPS output."""
        data_blocks = []
        current_block = []
        reading = False
        headers = None

        with open(filename) as f:
            for line in f:
                if line.startswith("Step ") or line.startswith("   Step "):
                    headers = line.split()
                    reading = True
                    current_block = []
                    continue
                if reading:
                    if line.startswith("Loop time") or line.startswith("ERROR"):
                        if current_block:
                            data_blocks.append((headers, np.array(current_block)))
                        reading = False
                        continue
                    try:
                        values = [float(x) for x in line.split()]
                        if len(values) == len(headers):
                            current_block.append(values)
                    except ValueError:
                        if current_block:
                            data_blocks.append((headers, np.array(current_block)))
                        reading = False

        return data_blocks

    if os.path.exists(LAMMPS_THERMO_FILE):
        blocks = parse_lammps_log(LAMMPS_THERMO_FILE)
        if blocks:
            # Use last block (production)
            headers, data = blocks[-1]
            print(f"  Found {len(blocks)} thermo blocks, using last one ({len(data)} rows)")
            print(f"  Columns: {headers}")

            if "Temp" in headers and "TotEng" in headers:
                t_idx = headers.index("Temp")
                e_idx = headers.index("TotEng")
                step_idx = headers.index("Step") if "Step" in headers else 0

                fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
                ax1.plot(data[:, step_idx], data[:, t_idx], "r-", alpha=0.7)
                ax1.set_xlabel("Step")
                ax1.set_ylabel("Temperature (K)")
                ax1.set_title("Temperature")

                ax2.plot(data[:, step_idx], data[:, e_idx], "b-", alpha=0.7)
                ax2.set_xlabel("Step")
                ax2.set_ylabel("Total Energy (eV)")
                ax2.set_title("Total Energy")

                fig.tight_layout()
                fig.savefig(os.path.join(OUTPUT_DIR, "lammps_thermo.png"), dpi=150)
                plt.close()
                print(f"  Saved: {OUTPUT_DIR}/lammps_thermo.png")

print("\n=== Analysis Complete ===")
```

## Key Parameters

| Parameter | Typical Value | Notes |
|---|---|---|
| `timestep` | 1-2 fs | Use 0.5-1 fs if H atoms present. LAMMPS metal units use ps (0.001-0.002). |
| `equilibration` | 500-5000 steps | Monitor temperature and energy to confirm equilibration. 1-10 ps typical. |
| `production` | 5000-100000 steps | Depends on property: diffusion needs long runs (100+ ps). RDF converges fast (~10 ps). |
| `temperature` | System-specific | Set via thermostat. Check instantaneous T matches target within fluctuations. |
| `friction` (Langevin) | 0.01 fs^-1 | Relaxation time = 1/friction. 50-200 fs typical. Too high = overdamped. |
| `taut` (Nose-Hoover) | 100*dt | Time constant for T coupling. 100-500 fs typical. |
| `taup` (NPT) | 1000*dt | Time constant for P coupling. Should be > taut. |
| `supercell size` | >500 atoms | Smaller cells have large finite-size effects, especially for diffusion. |
| `dump_interval` | 100-500 steps | Balance storage vs resolution. 0.1-1 ps between frames for RDF/MSD. |

### Timestep selection guidelines:
- No hydrogen: 2 fs is safe
- With hydrogen: 0.5-1 fs
- High temperature (>2000 K): reduce timestep by 25-50%
- Test with NVE: total energy drift < 1 meV/atom/ps

## Interpreting Results

### Temperature equilibration
- Instantaneous T fluctuates as sqrt(2/(3N))*T_target. For 500 atoms at 300 K, expect +/- ~20 K.
- If T drifts systematically, the thermostat coupling is wrong or the structure is transforming.

### RDF
- **Crystalline solid**: sharp peaks at neighbor shell distances. First peak position = nearest-neighbor distance.
- **Liquid**: broad peaks, g(r) -> 1 at large r. First minimum defines coordination shell.
- **Amorphous**: broad first peak, rapid decay.
- First peak position and height are key structural descriptors.

### MSD and diffusion
- **Solid**: MSD plateaus (atoms vibrate but do not migrate). D ~ 0.
- **Liquid**: MSD grows linearly with time. D = slope/(6) for 3D. Typical liquid metals: D ~ 1e-5 to 1e-4 cm^2/s.
- **Ballistic regime**: MSD ~ t^2 at very short times (< 0.1 ps). Do not include in diffusion fit.
- **Subdiffusive**: MSD ~ t^alpha with alpha < 1 indicates confined or glassy dynamics.

### Energy conservation (NVE check)
- Energy drift should be < 1 meV/atom/ps for a good integrator+potential combination.
- Large drift means timestep is too large or the potential has discontinuities.

## Common Issues

| Issue | Cause | Fix |
|---|---|---|
| Temperature explodes | Timestep too large, bad initial config, or unstable potential | Reduce timestep; minimize before MD; check potential validity |
| Atoms fly apart | Overlapping atoms in initial config | Run energy minimization first |
| MSD shows no diffusion in liquid | Run too short, or system froze | Increase run time to > 100 ps; check T is above melting point |
| LAMMPS error: lost atoms | Atoms moved too far in one step | Reduce timestep; check potential cutoff; ensure good initial structure |
| MACE MD very slow | Large system or large model | Use model="small" for screening; reduce system size; use "cuda" if available |
| RDF noisy | Not enough frames or atoms | Increase production run length or supercell size |
| NPT cell oscillates wildly | taup too small | Increase pressure coupling time constant |
| NVT temperature wrong | Friction too low or thermostat not connected | Increase friction for Langevin; check thermostat is properly attached |
| Trajectory file huge | Dumping too frequently | Increase dump interval; use binary trajectory format |
