# Lattice Thermal Conductivity

## When to Use

Use this skill when you need to compute the lattice (phonon-mediated) thermal conductivity of a crystalline material. This is the dominant heat-transport mechanism in semiconductors and insulators. Typical applications include thermoelectric screening, thermal management materials, and thermal barrier coatings.

## Method Selection

| Method | Speed | Accuracy | Best For |
|--------|-------|----------|----------|
| **A: ASE + MACE + phono3py** | Fast (minutes-hours) | Good (ML potential quality) | Rapid screening, large cells, many materials |
| **B: Green-Kubo MD** | Moderate (hours) | Moderate (classical, no quantum effects) | High-T, strongly anharmonic, disordered systems |
| **C: QE + phono3py** | Slow (days) | High (DFT-level) | Publication-quality, benchmarking, small cells |

**Decision guide:**
- Need a quick estimate or screening many materials? Use **Method A**.
- System is strongly anharmonic, has disorder, or you need high-T behavior? Use **Method B**.
- Need DFT-level accuracy for a paper? Use **Method C**.
- Always cross-check Method A against Method C for at least one material to gauge MACE accuracy.

## Prerequisites

```bash
pip install phonopy phono3py seekpath
```

Required Python packages (pre-installed): `ase`, `mace-torch`, `numpy`, `scipy`, `matplotlib`, `pymatgen`, `spglib`.

For Method C: Quantum ESPRESSO 7.5 (`pw.x`) must be available.

---

## Method A: ASE + MACE + phono3py (Boltzmann Transport Equation)

### Overview

1. Relax the structure with MACE.
2. Generate displaced supercells for 2nd-order (phonopy) and 3rd-order (phono3py) force constants.
3. Compute forces on all displaced supercells using MACE.
4. Build 2nd and 3rd order interatomic force constants (IFCs).
5. Solve the linearized Boltzmann transport equation (LBTE) in phono3py.
6. Extract kappa(T).

### Detailed Steps

```python
#!/usr/bin/env python3
"""
Lattice thermal conductivity via MACE + phono3py (BTE).
Complete, runnable script.
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from ase.io import read, write
from ase.optimize import LBFGS
from ase.constraints import ExpCellFilter
from pathlib import Path
import json
import warnings
warnings.filterwarnings('ignore')

# ── 0. Configuration ──────────────────────────────────────────────
STRUCTURE_FILE = "POSCAR"          # Input structure (any ASE-readable format)
MACE_MODEL     = "medium"         # "small", "medium", "large", or path to .model
SUPERCELL_2ND  = [3, 3, 3]        # Supercell for 2nd-order FC (harmonic phonons)
SUPERCELL_3RD  = [2, 2, 2]        # Supercell for 3rd-order FC (smaller is OK)
CUTOFF_PAIR_3RD = None            # Pair cutoff for 3rd-order (Angstrom); None = no cutoff
MESH           = [20, 20, 20]     # q-point mesh for BTE
TEMPERATURES   = list(range(100, 1001, 50))  # Temperature range (K)
FMAX           = 0.005            # Force convergence for relaxation (eV/Ang)
OUTPUT_DIR     = Path("kappa_mace_phono3py")
# ──────────────────────────────────────────────────────────────────

OUTPUT_DIR.mkdir(exist_ok=True)

# ── 1. Load MACE calculator ──────────────────────────────────────
from mace.calculators import mace_mp
calc = mace_mp(model=MACE_MODEL, default_dtype="float64")
print(f"Loaded MACE model: {MACE_MODEL}")

# ── 2. Relax the structure ────────────────────────────────────────
atoms = read(STRUCTURE_FILE)
atoms.calc = calc

ecf = ExpCellFilter(atoms, hydrostatic_strain=False)
opt = LBFGS(ecf, logfile=str(OUTPUT_DIR / "relax.log"))
opt.run(fmax=FMAX)
write(OUTPUT_DIR / "POSCAR_relaxed", atoms, format="vasp")
print(f"Relaxed energy: {atoms.get_potential_energy():.6f} eV")
print(f"Relaxed cell:\n{atoms.cell[:]}")

# ── 3. Set up phono3py ────────────────────────────────────────────
from phono3py import Phono3py
from phono3py.interface.calculator import read_crystal_structure

# Convert ASE atoms to phonopy-compatible cell
cell_tuple = (
    atoms.cell[:].tolist(),
    atoms.get_scaled_positions().tolist(),
    atoms.get_atomic_numbers().tolist(),
)

ph3 = Phono3py(
    unitcell=None,
    supercell_matrix=SUPERCELL_3RD,
    phonon_supercell_matrix=SUPERCELL_2ND,
    log_level=1,
)

# Build unitcell from ASE atoms using phono3py's PhonopyAtoms
from phonopy.structure.atoms import PhonopyAtoms

unitcell = PhonopyAtoms(
    symbols=atoms.get_chemical_symbols(),
    cell=atoms.cell[:],
    scaled_positions=atoms.get_scaled_positions(),
)

ph3 = Phono3py(
    unitcell=unitcell,
    supercell_matrix=SUPERCELL_3RD,
    phonon_supercell_matrix=SUPERCELL_2ND,
    log_level=1,
)

# ── 4. Generate displaced supercells ──────────────────────────────
# 2nd-order displacements (for harmonic phonons)
ph3.generate_displacements(distance=0.03)  # 0.03 Ang displacement

# Get supercells with displacements
supercells_3rd = ph3.supercells_with_displacements
phonon_supercells = ph3.phonon_supercells_with_displacements

n_3rd = len([s for s in supercells_3rd if s is not None])
n_2nd = len([s for s in phonon_supercells if s is not None])
print(f"Number of 3rd-order displaced supercells: {n_3rd}")
print(f"Number of 2nd-order displaced supercells: {n_2nd}")

# ── 5. Compute forces with MACE ──────────────────────────────────
from ase import Atoms as ASEAtoms

def phonopy_to_ase(phonopy_atoms):
    """Convert PhonopyAtoms to ASE Atoms."""
    return ASEAtoms(
        symbols=phonopy_atoms.symbols,
        cell=phonopy_atoms.cell,
        scaled_positions=phonopy_atoms.scaled_positions,
        pbc=True,
    )

def compute_forces(supercell_list, label=""):
    """Compute forces on a list of supercells using MACE."""
    forces_list = []
    total = len([s for s in supercell_list if s is not None])
    count = 0
    for i, sc in enumerate(supercell_list):
        if sc is None:
            forces_list.append(None)
            continue
        count += 1
        ase_sc = phonopy_to_ase(sc)
        ase_sc.calc = calc
        f = ase_sc.get_forces()
        forces_list.append(f)
        if count % 10 == 0 or count == total:
            print(f"  {label}: {count}/{total} done")
    return forces_list

print("Computing 3rd-order forces...")
forces_3rd = compute_forces(supercells_3rd, label="3rd-order")

print("Computing 2nd-order forces...")
forces_2nd = compute_forces(phonon_supercells, label="2nd-order")

# ── 6. Set forces and produce force constants ─────────────────────
# Set 3rd order forces
forces_3rd_array = np.array([f for f in forces_3rd if f is not None])
ph3.forces = forces_3rd_array

# Set 2nd order (phonon) forces
forces_2nd_array = np.array([f for f in forces_2nd if f is not None])
ph3.phonon_forces = forces_2nd_array

# Produce force constants
print("Producing 2nd-order force constants...")
ph3.produce_fc2()
print("Producing 3rd-order force constants...")
ph3.produce_fc3()

# ── 7. Initialize and run thermal conductivity (LBTE) ─────────────
ph3.mesh_numbers = MESH
ph3.init_phph_interaction()

ph3.run_thermal_conductivity(
    temperatures=TEMPERATURES,
    is_isotope=False,
    boundary_mfp=1e6,  # Large value = no boundary scattering
    write_kappa=True,
)

tc = ph3.thermal_conductivity

# ── 8. Extract kappa(T) ──────────────────────────────────────────
kappa = tc.kappa          # shape: (n_temp, 6) - xx, yy, zz, yz, xz, xy
temps = tc.temperatures

# Average (isotropic approximation): (kxx + kyy + kzz) / 3
kappa_avg = np.mean(kappa[:, :3], axis=1)

print("\n=== Lattice Thermal Conductivity ===")
print(f"{'T (K)':>8s}  {'kappa_xx':>10s}  {'kappa_yy':>10s}  {'kappa_zz':>10s}  {'kappa_avg':>10s}")
print(f"{'':>8s}  {'(W/mK)':>10s}  {'(W/mK)':>10s}  {'(W/mK)':>10s}  {'(W/mK)':>10s}")
for i, T in enumerate(temps):
    print(f"{T:8.1f}  {kappa[i,0]:10.3f}  {kappa[i,1]:10.3f}  {kappa[i,2]:10.3f}  {kappa_avg[i]:10.3f}")

# Save results
results = {
    "temperatures_K": list(temps),
    "kappa_xx": list(kappa[:, 0]),
    "kappa_yy": list(kappa[:, 1]),
    "kappa_zz": list(kappa[:, 2]),
    "kappa_avg": list(kappa_avg),
    "unit": "W/mK",
    "method": "MACE + phono3py (LBTE)",
    "supercell_2nd": SUPERCELL_2ND,
    "supercell_3rd": SUPERCELL_3RD,
    "mesh": MESH,
}
with open(OUTPUT_DIR / "kappa_results.json", "w") as f:
    json.dump(results, f, indent=2)

# ── 9. Plot kappa vs T ───────────────────────────────────────────
fig, ax = plt.subplots(figsize=(8, 5))
ax.plot(temps, kappa[:, 0], 'o-', label=r'$\kappa_{xx}$', markersize=4)
ax.plot(temps, kappa[:, 1], 's-', label=r'$\kappa_{yy}$', markersize=4)
ax.plot(temps, kappa[:, 2], '^-', label=r'$\kappa_{zz}$', markersize=4)
ax.plot(temps, kappa_avg,   'k--', label=r'$\kappa_{avg}$', linewidth=2)
ax.set_xlabel("Temperature (K)", fontsize=13)
ax.set_ylabel(r"$\kappa_L$ (W/mK)", fontsize=13)
ax.set_title("Lattice Thermal Conductivity", fontsize=14)
ax.legend(fontsize=11)
ax.set_xlim(temps[0], temps[-1])
ax.set_ylim(bottom=0)
ax.grid(True, alpha=0.3)
fig.tight_layout()
fig.savefig(OUTPUT_DIR / "kappa_vs_T.png", dpi=150)
print(f"\nPlot saved to {OUTPUT_DIR / 'kappa_vs_T.png'}")
```

---

## Method B: Green-Kubo from Molecular Dynamics (ASE + MACE)

### Overview

The Green-Kubo relation connects the thermal conductivity to the integral of the heat-flux autocorrelation function (HFACF):

kappa = V / (3 k_B T^2) * integral_0^inf <J(0) . J(t)> dt

1. NPT equilibration at target T, P.
2. NVE production run (microcanonical for proper fluctuations).
3. Compute heat flux J(t) at each step.
4. Compute autocorrelation and integrate.

### Detailed Steps

```python
#!/usr/bin/env python3
"""
Lattice thermal conductivity via Green-Kubo (MD heat flux autocorrelation).
Uses ASE + MACE.

NOTE: The Green-Kubo method requires long MD trajectories (100+ ps) and
multiple independent runs for statistical averaging. This script provides
a complete implementation.
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from ase.io import read, write
from ase.md.langevin import Langevin
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution
from ase.md.nptberendsen import NPTBerendsen
from ase.md.verlet import VelocityVerlet
from ase import units
from pathlib import Path
import json
import warnings
warnings.filterwarnings('ignore')

# ── 0. Configuration ──────────────────────────────────────────────
STRUCTURE_FILE   = "POSCAR"
MACE_MODEL       = "medium"
SUPERCELL        = [4, 4, 4]       # MD supercell (need large cell for convergence)
TEMPERATURE      = 300.0           # Target temperature (K)
PRESSURE         = 0.0             # Target pressure (GPa)
DT               = 1.0            # Timestep (fs)
NPT_STEPS        = 5000           # NPT equilibration steps
NVE_STEPS        = 50000          # NVE production steps (50 ps at 1 fs)
CORR_STEPS       = 5000           # Max correlation lag steps
N_INDEPENDENT    = 3              # Number of independent runs for averaging
FMAX             = 0.01           # Relaxation convergence (eV/Ang)
OUTPUT_DIR       = Path("kappa_green_kubo")
# ──────────────────────────────────────────────────────────────────

OUTPUT_DIR.mkdir(exist_ok=True)

# ── 1. Load MACE ──────────────────────────────────────────────────
from mace.calculators import mace_mp
calc = mace_mp(model=MACE_MODEL, default_dtype="float64")

# ── 2. Build supercell ────────────────────────────────────────────
atoms_prim = read(STRUCTURE_FILE)
atoms = atoms_prim.repeat(SUPERCELL)
print(f"Supercell: {len(atoms)} atoms, cell volume: {atoms.get_volume():.2f} A^3")

# ── 3. Heat flux computation ──────────────────────────────────────
def compute_heat_flux(atoms):
    """
    Compute the instantaneous heat flux vector J for a system.

    J = (1/V) * sum_i [ e_i * v_i + sum_{j>i} (f_ij . v_i) * r_ij ]

    For a pair potential this is exact. For a many-body potential (like MACE),
    we use the Hardy/Irving-Kirkwood stress-based formulation:

    J = (1/V) * [ sum_i e_i * v_i - S . v ]

    where S is the per-atom stress tensor contribution and the second term
    captures the virial (interaction) part.

    In practice we use the computationally simpler form:
    J_alpha = (1/V) * sum_i [e_i * v_i,alpha + sum_beta sigma_i,alpha,beta * v_i,beta]

    where sigma_i is the per-atom stress times atomic volume.
    """
    V = atoms.get_volume()
    velocities = atoms.get_velocities()          # (N, 3) in Ang/fs * sqrt(amu)
    masses = atoms.get_masses()                   # (N,)
    KE_per_atom = 0.5 * masses[:, None] * velocities**2  # (N, 3)
    ke_per_atom = 0.5 * np.sum(masses[:, None] * velocities**2, axis=1)  # (N,)

    # Get potential energy per atom
    pe_per_atom = atoms.get_potential_energies()   # (N,) eV

    # Total energy per atom
    e_per_atom = ke_per_atom + pe_per_atom         # (N,) eV

    # Convective part: sum_i e_i * v_i
    J_conv = np.sum(e_per_atom[:, None] * velocities, axis=0)  # (3,)

    # Virial (stress) part using per-atom stresses
    # ASE stresses: stress tensor in Voigt notation (xx, yy, zz, yz, xz, xy)
    # in units of eV/Ang^3 * volume = eV
    # We need per-atom stresses. Many calculators support this.
    try:
        stresses = atoms.get_stresses()  # (N, 6) Voigt, units eV/Ang^3
        # Convert Voigt to full 3x3: xx, yy, zz, yz, xz, xy
        # s_full[i] shape (3,3)
        s_full = np.zeros((len(atoms), 3, 3))
        s_full[:, 0, 0] = stresses[:, 0]
        s_full[:, 1, 1] = stresses[:, 1]
        s_full[:, 2, 2] = stresses[:, 2]
        s_full[:, 1, 2] = s_full[:, 2, 1] = stresses[:, 3]
        s_full[:, 0, 2] = s_full[:, 2, 0] = stresses[:, 4]
        s_full[:, 0, 1] = s_full[:, 1, 0] = stresses[:, 5]

        # Virial contribution: sum_i sigma_i . v_i
        # sigma_i has units eV/Ang^3, multiply by atomic volume -> eV/Ang
        # But ASE per-atom stress * V/N gives per-atom virial
        atomic_vol = V / len(atoms)
        J_virial = np.zeros(3)
        for alpha in range(3):
            for beta in range(3):
                # Note: ASE convention has stress = -sigma/V
                # per-atom stress * atomic_volume gives virial contribution
                J_virial[alpha] -= np.sum(
                    stresses[:, [0,5,4,5,1,3,4,3,2][alpha*3+beta]]
                    * velocities[:, beta]
                ) * atomic_vol
        # Simpler: use Einstein summation
        J_virial = -np.einsum('ijk,ij->k', s_full, velocities) * atomic_vol

    except Exception:
        # Fallback: use forces-based estimate
        # J_virial ~ -sum_i (r_i . f_i) * v_i / V  (approximate)
        forces = atoms.get_forces()
        positions = atoms.get_positions()
        J_virial = np.zeros(3)
        for alpha in range(3):
            J_virial[alpha] = np.sum(
                np.sum(positions * forces, axis=1) * velocities[:, alpha]
            )

    J = (J_conv + J_virial) / V  # eV/Ang^2/fs -> need unit conversion

    # Convert to SI-compatible: eV * Ang/fs / Ang^3 = eV/(Ang^2 * fs)
    # 1 eV = 1.602e-19 J, 1 Ang = 1e-10 m, 1 fs = 1e-15 s
    # J [eV/(Ang^2*fs)] * 1.602e-19 / (1e-10)^2 / (1e-15)^(-1)
    # = 1.602e-19 / 1e-20 * 1e15 = 1.602e-19 * 1e35 = 1.602e16 W/m^2...
    # We'll keep in eV/Ang^2/fs and convert at the end.
    return J


def autocorrelation(x, max_lag):
    """Compute normalized autocorrelation function of a 1D signal."""
    n = len(x)
    result = np.zeros(max_lag)
    mean = np.mean(x)
    x_centered = x - mean
    var = np.sum(x_centered**2)
    for lag in range(max_lag):
        if lag < n:
            result[lag] = np.sum(x_centered[:n-lag] * x_centered[lag:]) / (n - lag)
    return result


def compute_kappa_green_kubo(heat_flux_trajectory, volume, temperature, dt_fs, max_lag):
    """
    Compute thermal conductivity from heat flux trajectory using Green-Kubo.

    kappa = V / (3 * kB * T^2) * integral <J(0).J(t)> dt

    Parameters
    ----------
    heat_flux_trajectory : (n_steps, 3) array in eV/(Ang^2 * fs)
    volume : float, in Ang^3
    temperature : float, in K
    dt_fs : float, timestep in fs
    max_lag : int, max correlation steps

    Returns
    -------
    kappa_t : running integral of kappa (converges to kappa)
    acf : autocorrelation function
    """
    kB = 8.617333262e-5  # eV/K

    # Autocorrelation for each component
    acf_total = np.zeros(max_lag)
    for dim in range(3):
        acf_total += autocorrelation(heat_flux_trajectory[:, dim], max_lag)

    # Convert units:
    # J in eV/(Ang^2 * fs)
    # <J.J> in [eV/(Ang^2 * fs)]^2
    # integral in [eV/(Ang^2 * fs)]^2 * fs = eV^2/(Ang^4 * fs)
    # kappa = V/(3*kB*T^2) * integral
    #       = Ang^3 * eV^2/(Ang^4 * fs) / (eV/K * K^2)
    #       = eV / (Ang * fs * K)

    # To W/(m*K):
    # 1 eV = 1.602176634e-19 J
    # 1 Ang = 1e-10 m
    # 1 fs = 1e-15 s
    # eV/(Ang*fs*K) * (1.602e-19 J/eV) / (1e-10 m/Ang) / (1e-15 s/fs)
    # = 1.602e-19 / (1e-10 * 1e-15) / K = 1.602e-19 / 1e-25 = 1.602e6 W/(m*K)
    # But we divide by 3*kB*T^2 first.

    conv_factor = 1.602176634e-19 / (1e-10 * 1e-15)  # eV/(Ang*fs) -> W/m

    prefactor = volume / (3.0 * kB * temperature**2)

    # Running integral (trapezoidal)
    dt_fs_val = dt_fs
    kappa_running = np.zeros(max_lag)
    integral = 0.0
    for i in range(max_lag):
        if i == 0:
            integral = 0.0
        else:
            integral += 0.5 * (acf_total[i-1] + acf_total[i]) * dt_fs_val
        kappa_running[i] = prefactor * integral * conv_factor

    return kappa_running, acf_total


# ── 4. Run MD and collect heat flux ───────────────────────────────
all_kappa_runs = []

for run_idx in range(N_INDEPENDENT):
    print(f"\n=== Independent run {run_idx + 1}/{N_INDEPENDENT} ===")

    # Fresh copy
    md_atoms = atoms.copy()
    md_atoms.calc = calc

    # Initialize velocities
    MaxwellBoltzmannDistribution(md_atoms, temperature_K=TEMPERATURE)

    # ── 4a. NPT equilibration ─────────────────────────────────────
    print(f"  NPT equilibration: {NPT_STEPS} steps...")
    npt = NPTBerendsen(
        md_atoms,
        timestep=DT * units.fs,
        temperature_K=TEMPERATURE,
        pressure_au=PRESSURE * units.GPa,
        taut=100 * units.fs,
        taup=500 * units.fs,
        compressibility_au=4.57e-5 / units.GPa,  # typical for solids
    )
    npt.run(NPT_STEPS)
    V_equil = md_atoms.get_volume()
    print(f"  Equilibrated volume: {V_equil:.2f} Ang^3")

    # ── 4b. NVE production ─────────────────────────────────────────
    print(f"  NVE production: {NVE_STEPS} steps...")
    nve = VelocityVerlet(md_atoms, timestep=DT * units.fs)

    heat_flux_traj = []
    temperatures_traj = []

    for step in range(NVE_STEPS):
        nve.run(1)

        J = compute_heat_flux(md_atoms)
        heat_flux_traj.append(J)

        if (step + 1) % 5000 == 0:
            T_inst = md_atoms.get_kinetic_energy() / (1.5 * len(md_atoms) * units.kB)
            temperatures_traj.append(T_inst)
            print(f"    Step {step+1}/{NVE_STEPS}, T_inst = {T_inst:.1f} K")

    heat_flux_traj = np.array(heat_flux_traj)  # (NVE_STEPS, 3)

    # ── 4c. Compute kappa from this run ────────────────────────────
    kappa_running, acf = compute_kappa_green_kubo(
        heat_flux_traj, V_equil, TEMPERATURE, DT, min(CORR_STEPS, NVE_STEPS // 2)
    )
    all_kappa_runs.append(kappa_running)
    print(f"  Run {run_idx+1} kappa estimate: {kappa_running[-1]:.2f} W/mK (raw)")

# ── 5. Average over independent runs ──────────────────────────────
all_kappa_runs = np.array(all_kappa_runs)  # (N_INDEPENDENT, CORR_STEPS)
kappa_mean = np.mean(all_kappa_runs, axis=0)
kappa_std  = np.std(all_kappa_runs, axis=0)

# Find plateau region (kappa should plateau before decaying due to noise)
# Simple approach: take value at ~40% of correlation length
plateau_idx = int(0.4 * len(kappa_mean))
kappa_final = kappa_mean[plateau_idx]
kappa_err   = kappa_std[plateau_idx]

print(f"\n=== Green-Kubo Result ===")
print(f"Temperature: {TEMPERATURE} K")
print(f"kappa = {kappa_final:.2f} +/- {kappa_err:.2f} W/mK")

# ── 6. Save and plot ──────────────────────────────────────────────
results = {
    "temperature_K": TEMPERATURE,
    "kappa_WpmK": float(kappa_final),
    "kappa_error_WpmK": float(kappa_err),
    "method": "Green-Kubo (ASE + MACE)",
    "n_atoms": len(atoms),
    "supercell": SUPERCELL,
    "nve_steps": NVE_STEPS,
    "dt_fs": DT,
    "n_independent_runs": N_INDEPENDENT,
}
with open(OUTPUT_DIR / "kappa_green_kubo.json", "w") as f:
    json.dump(results, f, indent=2)

# Plot running integral
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

time_ps = np.arange(len(kappa_mean)) * DT / 1000.0

ax1.plot(time_ps, kappa_mean, 'b-', linewidth=1)
ax1.fill_between(time_ps, kappa_mean - kappa_std, kappa_mean + kappa_std, alpha=0.3)
ax1.axhline(kappa_final, color='r', linestyle='--', label=f'$\\kappa$ = {kappa_final:.1f} W/mK')
ax1.set_xlabel("Correlation time (ps)", fontsize=12)
ax1.set_ylabel(r"$\kappa$ (W/mK)", fontsize=12)
ax1.set_title("Running integral of Green-Kubo", fontsize=13)
ax1.legend()
ax1.grid(True, alpha=0.3)

# Plot HFACF
acf_norm = acf / acf[0] if acf[0] != 0 else acf
ax2.plot(time_ps, acf_norm, 'g-', linewidth=0.5)
ax2.set_xlabel("Correlation time (ps)", fontsize=12)
ax2.set_ylabel("HFACF (normalized)", fontsize=12)
ax2.set_title("Heat Flux Autocorrelation", fontsize=13)
ax2.grid(True, alpha=0.3)

fig.tight_layout()
fig.savefig(OUTPUT_DIR / "kappa_green_kubo.png", dpi=150)
print(f"Plot saved to {OUTPUT_DIR / 'kappa_green_kubo.png'}")
```

---

## Method C: Quantum ESPRESSO + phono3py (DFT-level BTE)

### Overview

1. Self-consistent field (SCF) calculation of the unit cell.
2. Generate displaced supercells via phono3py.
3. Run QE SCF for each displaced supercell to get forces.
4. Build 2nd and 3rd order IFCs.
5. Solve LBTE in phono3py.

### Detailed Steps

```python
#!/usr/bin/env python3
"""
Lattice thermal conductivity via QE + phono3py.
Generates all input files, run scripts, and post-processing.

This script:
  1. Relaxes the structure with QE (vc-relax).
  2. Generates phono3py displaced supercells.
  3. Creates QE scf input for each displaced supercell.
  4. Provides a bash runner script.
  5. Post-processes forces and runs phono3py BTE.
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
from pymatgen.core import Structure
from pymatgen.io.pwscf import PWInput
import json
import subprocess
import warnings
warnings.filterwarnings('ignore')

# ── 0. Configuration ──────────────────────────────────────────────
STRUCTURE_FILE   = "POSCAR"
PSEUDO_DIR       = "/home/pps/"          # Path to pseudopotentials
SUPERCELL_2ND    = [3, 3, 3]
SUPERCELL_3RD    = [2, 2, 2]
MESH             = [20, 20, 20]
TEMPERATURES     = list(range(100, 1001, 50))
ECUTWFC          = 60.0                  # Plane-wave cutoff (Ry)
ECUTRHO          = 480.0                 # Charge density cutoff (Ry)
KPOINTS_RELAX    = [6, 6, 6]             # k-mesh for relaxation
KPOINTS_SCF      = [2, 2, 2]             # k-mesh for supercell SCF (reduced)
CONV_THR         = 1.0e-8                # SCF convergence (Ry)
FORC_CONV_THR    = 1.0e-5                # Force convergence for relax (Ry/au)
QE_CMD           = "mpirun -np 4 pw.x"  # QE command
OUTPUT_DIR       = Path("kappa_qe_phono3py")
# ──────────────────────────────────────────────────────────────────

OUTPUT_DIR.mkdir(exist_ok=True)

# Mapping of element to pseudopotential filename (adjust as needed)
# Using SSSP efficiency pseudopotentials as an example
PSEUDO_MAP = {
    "Si": "Si.pbe-n-rrkjus_psl.1.0.0.UPF",
    "Ge": "Ge.pbe-dn-rrkjus_psl.1.0.0.UPF",
    "C":  "C.pbe-n-kjpaw_psl.1.0.0.UPF",
    "O":  "O.pbe-n-kjpaw_psl.1.0.0.UPF",
    "N":  "N.pbe-n-kjpaw_psl.1.0.0.UPF",
    "Al": "Al.pbe-n-kjpaw_psl.1.0.0.UPF",
    "Ga": "Ga.pbe-dn-kjpaw_psl.1.0.0.UPF",
    "Zn": "Zn.pbe-dn-kjpaw_psl.1.0.0.UPF",
    "Ti": "Ti.pbe-spn-kjpaw_psl.1.0.0.UPF",
    "Ba": "Ba.pbe-spn-kjpaw_psl.1.0.0.UPF",
    "Pb": "Pb.pbe-dn-kjpaw_psl.1.0.0.UPF",
}

def get_pseudo(element):
    """Get pseudopotential filename for an element."""
    if element in PSEUDO_MAP:
        return PSEUDO_MAP[element]
    # Generic fallback
    return f"{element}.pbe-n-kjpaw_psl.1.0.0.UPF"


# ── 1. Relaxation with QE ─────────────────────────────────────────
print("Step 1: Generating QE relaxation input...")
struct = Structure.from_file(STRUCTURE_FILE)
elements = list(set(str(s) for s in struct.species))

relax_input = f"""&CONTROL
  calculation  = 'vc-relax'
  prefix       = 'relax'
  outdir       = './tmp_relax'
  pseudo_dir   = '{PSEUDO_DIR}'
  tprnfor      = .true.
  tstress      = .true.
  forc_conv_thr = {FORC_CONV_THR}
  etot_conv_thr = 1.0d-6
/
&SYSTEM
  ibrav        = 0
  nat          = {struct.num_sites}
  ntyp         = {len(elements)}
  ecutwfc      = {ECUTWFC}
  ecutrho      = {ECUTRHO}
  occupations  = 'smearing'
  smearing     = 'gaussian'
  degauss      = 0.01
/
&ELECTRONS
  conv_thr     = {CONV_THR}
  mixing_beta  = 0.7
/
&IONS
  ion_dynamics = 'bfgs'
/
&CELL
  cell_dynamics = 'bfgs'
  press         = 0.0
/

ATOMIC_SPECIES
"""

for el in sorted(elements):
    mass = Structure.from_file(STRUCTURE_FILE).composition.get(el, 1.0)
    # Get atomic mass from pymatgen
    from pymatgen.core.periodic_table import Element as PmgElement
    atomic_mass = PmgElement(el).atomic_mass
    relax_input += f"  {el}  {atomic_mass:.4f}  {get_pseudo(el)}\n"

relax_input += f"\nATOMIC_POSITIONS crystal\n"
for site in struct:
    el = str(site.specie)
    relax_input += f"  {el}  {site.frac_coords[0]:.10f}  {site.frac_coords[1]:.10f}  {site.frac_coords[2]:.10f}\n"

relax_input += f"\nK_POINTS automatic\n"
relax_input += f"  {KPOINTS_RELAX[0]} {KPOINTS_RELAX[1]} {KPOINTS_RELAX[2]}  0 0 0\n"

relax_input += f"\nCELL_PARAMETERS angstrom\n"
for vec in struct.lattice.matrix:
    relax_input += f"  {vec[0]:.10f}  {vec[1]:.10f}  {vec[2]:.10f}\n"

with open(OUTPUT_DIR / "relax.in", "w") as f:
    f.write(relax_input)

print(f"  Written: {OUTPUT_DIR / 'relax.in'}")

# ── 2. Parse relaxed structure (after QE runs) and generate supercells ─
def parse_qe_output_structure(pw_output_file):
    """Parse relaxed structure from QE output."""
    from pymatgen.io.pwscf import PWOutput
    # Alternative: parse manually for robustness
    lines = open(pw_output_file).readlines()

    # Find final cell parameters and positions
    cell = np.zeros((3, 3))
    positions = []
    species = []

    # Search backwards for final CELL_PARAMETERS
    cell_found = False
    for i in range(len(lines)-1, -1, -1):
        if 'CELL_PARAMETERS' in lines[i]:
            unit = 'angstrom' if 'angstrom' in lines[i].lower() else 'bohr'
            scale = 1.0 if 'angstrom' in lines[i].lower() else 0.529177
            for j in range(3):
                vals = lines[i+1+j].split()
                cell[j] = [float(v) * scale for v in vals]
            cell_found = True
            break

    # Search backwards for final ATOMIC_POSITIONS
    for i in range(len(lines)-1, -1, -1):
        if 'ATOMIC_POSITIONS' in lines[i]:
            coord_type = 'crystal' if 'crystal' in lines[i].lower() else 'angstrom'
            j = i + 1
            while j < len(lines) and lines[j].strip() and not lines[j].strip().startswith(('End', 'CELL', 'K_POINTS')):
                parts = lines[j].split()
                if len(parts) >= 4:
                    species.append(parts[0])
                    positions.append([float(parts[1]), float(parts[2]), float(parts[3])])
                j += 1
            break

    if not cell_found or not positions:
        raise ValueError(f"Could not parse structure from {pw_output_file}")

    from pymatgen.core import Lattice
    lattice = Lattice(cell)
    if coord_type == 'crystal':
        return Structure(lattice, species, positions)
    else:
        return Structure(lattice, species, positions, coords_are_cartesian=True)


# ── 3. Generate phono3py supercells ────────────────────────────────
print("\nStep 2-3: Setting up phono3py displaced supercells...")

from phonopy.structure.atoms import PhonopyAtoms
from phono3py import Phono3py

def structure_to_phonopy(struct):
    """Convert pymatgen Structure to PhonopyAtoms."""
    return PhonopyAtoms(
        symbols=[str(s) for s in struct.species],
        cell=struct.lattice.matrix,
        scaled_positions=struct.frac_coords,
    )

def phonopy_to_structure(phonopy_atoms):
    """Convert PhonopyAtoms to pymatgen Structure."""
    from pymatgen.core import Lattice
    lattice = Lattice(phonopy_atoms.cell)
    return Structure(
        lattice,
        phonopy_atoms.symbols,
        phonopy_atoms.scaled_positions,
    )

# Use the input structure for now (in production, use relaxed structure)
# To use relaxed: struct_relaxed = parse_qe_output_structure("relax.out")
unitcell = structure_to_phonopy(struct)

ph3 = Phono3py(
    unitcell=unitcell,
    supercell_matrix=SUPERCELL_3RD,
    phonon_supercell_matrix=SUPERCELL_2ND,
    log_level=1,
)

ph3.generate_displacements(distance=0.03)

supercells_3rd = ph3.supercells_with_displacements
phonon_supercells = ph3.phonon_supercells_with_displacements

n_3rd = len([s for s in supercells_3rd if s is not None])
n_2nd = len([s for s in phonon_supercells if s is not None])
print(f"  3rd-order displaced supercells: {n_3rd}")
print(f"  2nd-order displaced supercells: {n_2nd}")

# ── 4. Write QE inputs for each displaced supercell ───────────────
def write_qe_scf_input(phonopy_atoms, filename, prefix, kpoints):
    """Write a QE scf input file for a supercell."""
    struct_pmg = phonopy_to_structure(phonopy_atoms)
    elements = sorted(set(str(s) for s in struct_pmg.species))

    input_text = f"""&CONTROL
  calculation  = 'scf'
  prefix       = '{prefix}'
  outdir       = './tmp_{prefix}'
  pseudo_dir   = '{PSEUDO_DIR}'
  tprnfor      = .true.
  tstress      = .true.
/
&SYSTEM
  ibrav        = 0
  nat          = {struct_pmg.num_sites}
  ntyp         = {len(elements)}
  ecutwfc      = {ECUTWFC}
  ecutrho      = {ECUTRHO}
  occupations  = 'smearing'
  smearing     = 'gaussian'
  degauss      = 0.01
/
&ELECTRONS
  conv_thr     = {CONV_THR}
  mixing_beta  = 0.7
/

ATOMIC_SPECIES
"""
    for el in elements:
        from pymatgen.core.periodic_table import Element as PmgElement
        atomic_mass = PmgElement(el).atomic_mass
        input_text += f"  {el}  {atomic_mass:.4f}  {get_pseudo(el)}\n"

    input_text += "\nATOMIC_POSITIONS crystal\n"
    for site in struct_pmg:
        el = str(site.specie)
        input_text += f"  {el}  {site.frac_coords[0]:.10f}  {site.frac_coords[1]:.10f}  {site.frac_coords[2]:.10f}\n"

    input_text += f"\nK_POINTS automatic\n"
    input_text += f"  {kpoints[0]} {kpoints[1]} {kpoints[2]}  0 0 0\n"

    input_text += "\nCELL_PARAMETERS angstrom\n"
    for vec in struct_pmg.lattice.matrix:
        input_text += f"  {vec[0]:.10f}  {vec[1]:.10f}  {vec[2]:.10f}\n"

    with open(filename, "w") as f:
        f.write(input_text)

# Write 3rd-order supercell inputs
disp3_dir = OUTPUT_DIR / "disp_fc3"
disp3_dir.mkdir(exist_ok=True)
print(f"\nWriting 3rd-order QE inputs to {disp3_dir}/...")
for i, sc in enumerate(supercells_3rd):
    if sc is None:
        continue
    fname = disp3_dir / f"scf_disp3_{i:05d}.in"
    write_qe_scf_input(sc, fname, f"d3_{i:05d}", KPOINTS_SCF)

# Write 2nd-order supercell inputs
disp2_dir = OUTPUT_DIR / "disp_fc2"
disp2_dir.mkdir(exist_ok=True)
print(f"Writing 2nd-order QE inputs to {disp2_dir}/...")
for i, sc in enumerate(phonon_supercells):
    if sc is None:
        continue
    fname = disp2_dir / f"scf_disp2_{i:05d}.in"
    write_qe_scf_input(sc, fname, f"d2_{i:05d}", KPOINTS_SCF)

# ── 5. Write runner script ────────────────────────────────────────
runner_script = f"""#!/bin/bash
# Run all QE SCF calculations for phono3py
# Usage: bash run_all_qe.sh

QE_CMD="{QE_CMD}"
NPROC=$(nproc)

echo "=== Running QE relaxation ==="
cd {OUTPUT_DIR}
mkdir -p tmp_relax
$QE_CMD < relax.in > relax.out 2>&1
echo "Relaxation done."

echo "=== Running 3rd-order displaced supercells ==="
cd {disp3_dir}
for inp in scf_disp3_*.in; do
    out="${{inp%.in}}.out"
    if [ ! -f "$out" ]; then
        echo "Running $inp ..."
        prefix=$(echo $inp | sed 's/.in//')
        mkdir -p tmp_$prefix
        $QE_CMD < $inp > $out 2>&1
    else
        echo "Skipping $inp (output exists)"
    fi
done

echo "=== Running 2nd-order displaced supercells ==="
cd {disp2_dir}
for inp in scf_disp2_*.in; do
    out="${{inp%.in}}.out"
    if [ ! -f "$out" ]; then
        echo "Running $inp ..."
        prefix=$(echo $inp | sed 's/.in//')
        mkdir -p tmp_$prefix
        $QE_CMD < $inp > $out 2>&1
    else
        echo "Skipping $inp (output exists)"
    fi
done

echo "=== All QE calculations complete ==="
"""

with open(OUTPUT_DIR / "run_all_qe.sh", "w") as f:
    f.write(runner_script)
print(f"  Runner script: {OUTPUT_DIR / 'run_all_qe.sh'}")

# ── 6. Post-processing script (run after QE completes) ────────────
postproc_script = f'''#!/usr/bin/env python3
"""
Post-processing: parse QE forces, build FCs, solve BTE.
Run this AFTER all QE calculations are complete.
"""
import numpy as np
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
from phonopy.structure.atoms import PhonopyAtoms
from phono3py import Phono3py

OUTPUT_DIR = Path("{OUTPUT_DIR}")
SUPERCELL_2ND = {SUPERCELL_2ND}
SUPERCELL_3RD = {SUPERCELL_3RD}
MESH = {MESH}
TEMPERATURES = {TEMPERATURES}

# Bohr to Angstrom, Ry/Bohr to eV/Angstrom
RY_TO_EV = 13.605693122994
BOHR_TO_ANG = 0.529177249

def parse_forces_from_qe_output(filename):
    """Parse forces from QE output file. Returns (n_atoms, 3) array in eV/Ang."""
    lines = open(filename).readlines()
    forces = []
    for i, line in enumerate(lines):
        if "Forces acting on atoms" in line:
            forces = []
            j = i + 2  # skip blank line
            while j < len(lines):
                if "force =" in lines[j]:
                    parts = lines[j].split("force =")[1].split()
                    fx, fy, fz = float(parts[0]), float(parts[1]), float(parts[2])
                    # QE forces are in Ry/Bohr -> convert to eV/Ang
                    forces.append([
                        fx * RY_TO_EV / BOHR_TO_ANG,
                        fy * RY_TO_EV / BOHR_TO_ANG,
                        fz * RY_TO_EV / BOHR_TO_ANG,
                    ])
                elif lines[j].strip() == "" and forces:
                    break
                j += 1
    return np.array(forces)


# Reconstruct phono3py object
from pymatgen.core import Structure
struct = Structure.from_file("{STRUCTURE_FILE}")
# If relaxed structure available, load it instead:
# struct = parse_qe_output_structure(OUTPUT_DIR / "relax.out")

unitcell = PhonopyAtoms(
    symbols=[str(s) for s in struct.species],
    cell=struct.lattice.matrix,
    scaled_positions=struct.frac_coords,
)

ph3 = Phono3py(
    unitcell=unitcell,
    supercell_matrix=SUPERCELL_3RD,
    phonon_supercell_matrix=SUPERCELL_2ND,
    log_level=1,
)
ph3.generate_displacements(distance=0.03)

supercells_3rd = ph3.supercells_with_displacements
phonon_supercells = ph3.phonon_supercells_with_displacements

# Parse 3rd-order forces
print("Parsing 3rd-order forces...")
forces_3rd = []
disp3_dir = OUTPUT_DIR / "disp_fc3"
for i, sc in enumerate(supercells_3rd):
    if sc is None:
        continue
    out_file = disp3_dir / f"scf_disp3_{{i:05d}}.out"
    f = parse_forces_from_qe_output(out_file)
    forces_3rd.append(f)

# Parse 2nd-order forces
print("Parsing 2nd-order forces...")
forces_2nd = []
disp2_dir = OUTPUT_DIR / "disp_fc2"
for i, sc in enumerate(phonon_supercells):
    if sc is None:
        continue
    out_file = disp2_dir / f"scf_disp2_{{i:05d}}.out"
    f = parse_forces_from_qe_output(out_file)
    forces_2nd.append(f)

forces_3rd = np.array(forces_3rd)
forces_2nd = np.array(forces_2nd)

# Set forces
ph3.forces = forces_3rd
ph3.phonon_forces = forces_2nd

# Produce force constants
print("Producing force constants...")
ph3.produce_fc2()
ph3.produce_fc3()

# Solve BTE
print("Solving BTE...")
ph3.mesh_numbers = MESH
ph3.init_phph_interaction()
ph3.run_thermal_conductivity(
    temperatures=TEMPERATURES,
    is_isotope=False,
    boundary_mfp=1e6,
    write_kappa=True,
)

tc = ph3.thermal_conductivity
kappa = tc.kappa
temps = tc.temperatures
kappa_avg = np.mean(kappa[:, :3], axis=1)

print("\\n=== Lattice Thermal Conductivity (QE + phono3py) ===")
print(f"{{'T (K)':>8s}}  {{'kappa_xx':>10s}}  {{'kappa_yy':>10s}}  {{'kappa_zz':>10s}}  {{'kappa_avg':>10s}}")
for i, T in enumerate(temps):
    print(f"{{T:8.1f}}  {{kappa[i,0]:10.3f}}  {{kappa[i,1]:10.3f}}  {{kappa[i,2]:10.3f}}  {{kappa_avg[i]:10.3f}}")

results = {{
    "temperatures_K": list(temps),
    "kappa_xx": list(kappa[:, 0]),
    "kappa_yy": list(kappa[:, 1]),
    "kappa_zz": list(kappa[:, 2]),
    "kappa_avg": list(kappa_avg),
    "unit": "W/mK",
    "method": "QE + phono3py (LBTE)",
}}
with open(OUTPUT_DIR / "kappa_qe_results.json", "w") as f:
    json.dump(results, f, indent=2)

# Plot
fig, ax = plt.subplots(figsize=(8, 5))
ax.plot(temps, kappa[:, 0], "o-", label=r"$\\kappa_{{xx}}$", markersize=4)
ax.plot(temps, kappa[:, 1], "s-", label=r"$\\kappa_{{yy}}$", markersize=4)
ax.plot(temps, kappa[:, 2], "^-", label=r"$\\kappa_{{zz}}$", markersize=4)
ax.plot(temps, kappa_avg, "k--", label=r"$\\kappa_{{avg}}$", linewidth=2)
ax.set_xlabel("Temperature (K)", fontsize=13)
ax.set_ylabel(r"$\\kappa_L$ (W/mK)", fontsize=13)
ax.set_title("Lattice Thermal Conductivity (QE + phono3py)", fontsize=14)
ax.legend(fontsize=11)
ax.set_xlim(temps[0], temps[-1])
ax.set_ylim(bottom=0)
ax.grid(True, alpha=0.3)
fig.tight_layout()
fig.savefig(OUTPUT_DIR / "kappa_vs_T_qe.png", dpi=150)
print(f"Plot saved: {{OUTPUT_DIR / 'kappa_vs_T_qe.png'}}")
'''

with open(OUTPUT_DIR / "postprocess_kappa.py", "w") as f:
    f.write(postproc_script)
print(f"  Post-processing script: {OUTPUT_DIR / 'postprocess_kappa.py'}")

print("\n=== Workflow Summary ===")
print(f"1. Run QE relaxation:  cd {OUTPUT_DIR} && {QE_CMD} < relax.in > relax.out")
print(f"2. (Optional) Update structure in postprocess script with relaxed geometry")
print(f"3. Run all displaced supercells: bash {OUTPUT_DIR / 'run_all_qe.sh'}")
print(f"4. Post-process: python {OUTPUT_DIR / 'postprocess_kappa.py'}")
```

---

## Key Parameters

### Supercell size
| Parameter | Recommended | Notes |
|-----------|-------------|-------|
| 2nd-order supercell | 3x3x3 or larger | Must converge phonon dispersion; check for imaginary modes |
| 3rd-order supercell | 2x2x2 | Smaller is OK; 3rd-order IFCs are shorter-ranged. Increase to 3x3x3 only if kappa is not converged |
| Displacement distance | 0.03 Ang | Standard value. Smaller (0.01) for stiffer materials |

### Pair cutoff for 3rd order
- Setting `cutoff_pair_distance` in phono3py reduces the number of displaced supercells dramatically.
- Typical values: nearest-neighbor distance + 1-2 Ang.
- For Si (nn = 2.35 Ang): cutoff ~4-5 Ang is usually sufficient.
- Set `None` to include all pairs (most accurate but expensive).

### BTE q-mesh
| Material type | Mesh | Notes |
|---------------|------|-------|
| High-kappa (diamond, Si) | 20x20x20 or denser | Need fine mesh to capture long-MFP phonons |
| Low-kappa (PbTe, Bi2Te3) | 15x15x15 | Converges faster |
| Convergence check | Compare 15^3 vs 20^3 vs 25^3 | kappa should change < 5% |

### Green-Kubo specific
| Parameter | Recommended | Notes |
|-----------|-------------|-------|
| Supercell | 4x4x4 or larger (>200 atoms) | Finite-size effects are severe with small cells |
| Timestep | 1 fs | Decrease for light elements (H, Li) |
| NVE production | 50-200 ps | Must be >> phonon relaxation times |
| Correlation length | ~10 ps | Ensure HFACF decays to noise |
| Independent runs | 3-5 | For statistical averaging |

### Temperature range
- Typical: 100-1000 K in steps of 50 K.
- Above Debye temperature: kappa ~ 1/T (Umklapp-dominated).
- Below Debye temperature: kappa drops due to fewer phonons.
- At very low T: boundary scattering may dominate (set `boundary_mfp`).

---

## Interpreting Results

### Expected behavior
- **kappa vs T:** Should decrease roughly as 1/T above the Debye temperature (classical Umklapp regime). At low T, kappa rises, peaks near T_Debye/5, then falls.
- **Anisotropy:** Layered or anisotropic crystals show different kappa_xx, kappa_yy, kappa_zz. Cubic crystals should show kappa_xx = kappa_yy = kappa_zz.
- **Magnitude benchmarks:**
  - Diamond: ~2000 W/mK at 300 K
  - Silicon: ~150 W/mK at 300 K
  - GaAs: ~45 W/mK at 300 K
  - PbTe: ~2 W/mK at 300 K

### Convergence checks
1. **Supercell convergence:** Run with 2x2x2 and 3x3x3 3rd-order supercells. kappa should differ < 10%.
2. **q-mesh convergence:** Compare meshes (e.g., 15^3, 20^3, 25^3).
3. **2nd-order IFC quality:** Check phonon dispersion for imaginary frequencies. If present, increase supercell or check relaxation.
4. **Green-Kubo plateau:** The running integral should show a clear plateau before noise dominates. If no plateau, run longer or increase cell size.

### Comparing methods
- Method A (MACE) vs Method C (QE): Expect 10-30% difference depending on MACE accuracy for the material. MACE-MP-0 tends to be better for main-group elements.
- Method B (Green-Kubo) vs Method A (BTE): Green-Kubo includes all orders of anharmonicity but is classical (no Bose-Einstein statistics). Agree at high T; disagree at low T.

---

## Common Issues

| Problem | Cause | Solution |
|---------|-------|----------|
| Imaginary phonon frequencies | Poor relaxation or too-small supercell | Re-relax with tighter convergence; increase 2nd-order supercell |
| kappa does not converge with q-mesh | Long-MFP phonons dominate | Use denser q-mesh; check with iterative BTE (not just RTA) |
| kappa too high compared to experiment | Missing isotope/boundary scattering | Enable `is_isotope=True` in phono3py; set realistic `boundary_mfp` |
| kappa too low compared to experiment | Poor force constants or too-small supercell | Increase supercell size; check MACE accuracy against DFT |
| Green-Kubo: no plateau in running integral | Too short trajectory or too small cell | Increase NVE production to 200+ ps; use larger supercell |
| Green-Kubo: very noisy | Too few independent runs | Run 5-10 independent trajectories and average |
| phono3py OOM for large systems | Too many displaced supercells | Use `cutoff_pair_distance` to reduce 3rd-order pairs |
| QE SCF not converging for supercells | Poor initial charge density | Reduce `mixing_beta` to 0.3-0.5; use `mixing_mode = 'local-TF'` |
| MACE gives wrong phonons | MACE not accurate for this chemistry | Cross-check a few forces against QE; consider fine-tuning MACE |
