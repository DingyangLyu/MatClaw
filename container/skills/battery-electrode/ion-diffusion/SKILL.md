# Ion Diffusion in Battery Materials

## When to Use

- Compute Li-ion (or Na, K, Mg) diffusion coefficients in cathodes, anodes, or solid electrolytes
- Determine activation energy for ionic transport
- Calculate migration barriers for specific diffusion pathways
- Compare rate capability of candidate electrode/electrolyte materials
- Predict room-temperature diffusivity from elevated-temperature simulations

## Method Selection

```
What do you need?

Single migration barrier for a known pathway?
  --> Method B: NEB (ASE + MACE or QE neb.x)
  Fast screening --> ASE + MACE NEB (minutes)
  Publication quality --> QE neb.x (hours to days)

Full diffusion coefficient D(T) and activation energy Ea?
  --> Method A: AIMD (ASE + MACE MD at multiple temperatures)
  Requires: 50+ ps per temperature, 4-6 temperatures, >100 atoms
  Gives: D(T), Ea from Arrhenius fit, diffusion mechanism

Both barrier and D(T) for validation?
  --> Run NEB first for barrier estimate, then AIMD to confirm and get prefactor

Comparison of methods:
  NEB: gives barrier height directly, path-specific, no thermal sampling
  AIMD: gives D(T), Ea, mechanism, includes thermal effects and correlations
  NEB barrier should be close to AIMD Ea for single-barrier mechanisms
```

## Prerequisites

- pymatgen (structure building, supercell creation)
- ASE + mace-torch (both methods)
- numpy, scipy (MSD analysis, Arrhenius fitting)
- matplotlib (plotting)
- Optional: `pip install pymatgen-analysis-diffusion` for advanced MSD analysis and probability density

## Detailed Steps

### Method A: ASE + MACE AIMD

#### Complete Workflow: Li Diffusion in Li3PO4 (Model Solid Electrolyte)

```python
#!/usr/bin/env python3
"""
Ab initio molecular dynamics (AIMD) with MACE for Li-ion diffusion.

Workflow:
1. Build supercell (>100 atoms for meaningful statistics)
2. NPT equilibration at each temperature
3. NVT production runs at 600K, 800K, 1000K, 1200K
4. MSD analysis -> D(T) at each temperature
5. Arrhenius fit: ln(D) vs 1/T -> activation energy Ea
6. Extrapolate to room temperature (300K)

System: Li3PO4 (gamma phase) -- model solid electrolyte
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

from pathlib import Path
import json
import time

from pymatgen.core import Structure, Lattice, Element
from pymatgen.io.ase import AseAtomsAdaptor

from ase.io import read, write, Trajectory
from ase.optimize import LBFGS
from ase.constraints import ExpCellFilter
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution, Stationary
from ase.md.langevin import Langevin
from ase.md.nptberendsen import NPTBerendsen
from ase import units

from mace.calculators import mace_mp
from scipy.stats import linregress

# ============================================================
# 1. CONFIGURATION
# ============================================================
MACE_MODEL = "medium"          # "small", "medium", "large"
DEVICE = "cpu"                 # "cpu" or "cuda"
SUPERCELL = [2, 2, 2]         # supercell dimensions
TEMPERATURES = [600, 800, 1000, 1200]  # K -- elevated temperatures
EQUILIBRATION_STEPS = 2000     # NPT equilibration (2 ps at 1 fs timestep)
PRODUCTION_STEPS = 50000       # NVT production (50 ps at 1 fs timestep)
TIMESTEP = 1.0                 # fs
DUMP_INTERVAL = 10             # save trajectory every N steps
FRICTION = 0.01                # Langevin friction (1/fs)
OUTPUT_DIR = Path("li_diffusion")
OUTPUT_DIR.mkdir(exist_ok=True)

# ============================================================
# 2. BUILD STRUCTURE
# ============================================================
print("Building Li3PO4 supercell...")

# Gamma-Li3PO4, Pnma (space group 62)
# Experimental: a=10.49, b=6.12, c=4.93 Angstrom
li3po4 = Structure.from_spacegroup(
    "Pnma",
    Lattice.orthorhombic(10.49, 6.12, 4.93),
    species=["Li", "Li", "P", "O", "O", "O"],
    coords=[
        [0.000, 0.000, 0.000],   # Li1 (4a)
        [0.164, 0.500, 0.190],   # Li2 (4c)
        [0.090, 0.250, 0.420],   # P (4c)
        [0.165, 0.250, 0.710],   # O1 (4c)
        [0.090, 0.040, 0.290],   # O2 (8d)
        [0.210, 0.250, 0.220],   # O3 (4c)
    ],
)
print(f"Primitive cell: {li3po4.formula}, {len(li3po4)} atoms")

# Make supercell
supercell = li3po4.copy()
supercell.make_supercell(SUPERCELL)
n_atoms = len(supercell)
n_li = sum(1 for s in supercell.species if s == Element("Li"))
print(f"Supercell: {supercell.formula}, {n_atoms} atoms, {n_li} Li atoms")

if n_atoms < 100:
    print(f"WARNING: {n_atoms} atoms is small. Consider larger supercell for "
          f"reliable diffusion statistics (recommend >100 atoms).")

# ============================================================
# 3. SET UP CALCULATOR
# ============================================================
print("Loading MACE calculator...")
calc = mace_mp(model=MACE_MODEL, dispersion=False, default_dtype="float64",
               device=DEVICE)
adaptor = AseAtomsAdaptor()

# ============================================================
# 4. INITIAL RELAXATION
# ============================================================
print("Relaxing initial structure...")
atoms = adaptor.get_atoms(supercell)
atoms.calc = calc
ecf = ExpCellFilter(atoms, scalar_pressure=0.0)
opt = LBFGS(ecf, logfile=str(OUTPUT_DIR / "initial_relax.log"))
opt.run(fmax=0.02, steps=300)

relaxed_struct = adaptor.get_structure(atoms)
relaxed_struct.to(str(OUTPUT_DIR / "relaxed_supercell.cif"), fmt="cif")
print(f"Relaxed. Volume = {relaxed_struct.volume:.2f} A^3")

# ============================================================
# 5. MD AT EACH TEMPERATURE
# ============================================================
def compute_msd(trajectory_file, species="Li", start_frac=0.2):
    """
    Compute mean square displacement (MSD) for a specific species.

    Uses the standard Einstein relation:
        MSD(t) = <|r(t) - r(0)|^2>

    The start_frac parameter skips the initial fraction of the trajectory
    to avoid transient effects.

    Returns: (time_ps, msd_angstrom2)
    """
    traj = read(trajectory_file, index=":")
    n_frames = len(traj)
    start_frame = int(n_frames * start_frac)
    traj = traj[start_frame:]
    n_frames = len(traj)

    if n_frames < 10:
        return None, None

    # Identify atoms of the target species
    species_indices = [i for i, s in enumerate(traj[0].get_chemical_symbols())
                       if s == species]
    n_species = len(species_indices)

    if n_species == 0:
        return None, None

    # Get positions (unwrapped -- ASE stores unwrapped positions)
    # For a proper MSD we need unwrapped coordinates.
    # ASE Langevin/NVT keeps track of unwrapped positions automatically
    # if we use atoms.get_positions() with the trajectory.
    # However, for periodic systems, we must unwrap manually.

    # Get all positions
    positions = np.array([frame.get_positions() for frame in traj])  # (n_frames, n_atoms, 3)
    species_pos = positions[:, species_indices, :]  # (n_frames, n_species, 3)

    # Unwrap positions: detect jumps > half the cell and correct
    cell_lengths = np.array(traj[0].get_cell().lengths())
    for t in range(1, n_frames):
        diff = species_pos[t] - species_pos[t - 1]
        # If any component jumps by more than half the cell, unwrap
        for dim in range(3):
            jumps = diff[:, dim]
            species_pos[t, jumps > cell_lengths[dim] / 2, dim] -= cell_lengths[dim]
            species_pos[t, jumps < -cell_lengths[dim] / 2, dim] += cell_lengths[dim]

    # Compute MSD using multiple time origins for better statistics
    max_lag = n_frames // 2
    msd = np.zeros(max_lag)
    counts = np.zeros(max_lag)

    for t0 in range(0, n_frames - max_lag, max(1, max_lag // 20)):
        for dt in range(1, max_lag):
            disp = species_pos[t0 + dt] - species_pos[t0]  # (n_species, 3)
            msd[dt] += np.mean(np.sum(disp ** 2, axis=1))  # average over species atoms
            counts[dt] += 1

    valid = counts > 0
    msd[valid] /= counts[valid]

    time_ps = np.arange(max_lag) * TIMESTEP * DUMP_INTERVAL / 1000.0  # convert fs to ps
    return time_ps, msd

diffusion_results = {}

for temp in TEMPERATURES:
    print(f"\n{'=' * 60}")
    print(f"TEMPERATURE: {temp} K")
    print(f"{'=' * 60}")

    temp_dir = OUTPUT_DIR / f"T{temp}K"
    temp_dir.mkdir(exist_ok=True)

    # Fresh atoms from relaxed structure
    atoms = adaptor.get_atoms(relaxed_struct)
    atoms.calc = calc

    # Initialize velocities
    MaxwellBoltzmannDistribution(atoms, temperature_K=temp)
    Stationary(atoms)  # zero total momentum

    # ----- NPT EQUILIBRATION -----
    print(f"  NPT equilibration ({EQUILIBRATION_STEPS} steps)...")
    traj_equil = Trajectory(str(temp_dir / "equil.traj"), "w", atoms)

    dyn_npt = NPTBerendsen(
        atoms,
        timestep=TIMESTEP * units.fs,
        temperature_K=temp,
        pressure_au=0.0,             # zero pressure (atmospheric ~ 0 for solids)
        taut=100 * units.fs,         # thermostat time constant
        taup=1000 * units.fs,        # barostat time constant
        compressibility_au=4.57e-5 / units.Pascal,  # typical solid
    )
    dyn_npt.attach(traj_equil.write, interval=DUMP_INTERVAL)

    t_start = time.time()
    dyn_npt.run(EQUILIBRATION_STEPS)
    traj_equil.close()
    t_equil = time.time() - t_start
    print(f"  Equilibration done in {t_equil:.1f} s")

    # Get equilibrated cell for NVT
    equil_cell = atoms.get_cell().copy()
    equil_volume = atoms.get_volume()
    print(f"  Equilibrated volume: {equil_volume:.2f} A^3")

    # ----- NVT PRODUCTION -----
    print(f"  NVT production ({PRODUCTION_STEPS} steps = "
          f"{PRODUCTION_STEPS * TIMESTEP / 1000:.1f} ps)...")
    traj_prod = Trajectory(str(temp_dir / "production.traj"), "w", atoms)

    dyn_nvt = Langevin(
        atoms,
        timestep=TIMESTEP * units.fs,
        temperature_K=temp,
        friction=FRICTION / units.fs,
    )
    dyn_nvt.attach(traj_prod.write, interval=DUMP_INTERVAL)

    # Energy logger
    energies_log = []
    def log_energy():
        epot = atoms.get_potential_energy() / len(atoms)
        ekin = atoms.get_kinetic_energy() / len(atoms)
        t_inst = ekin / (1.5 * units.kB)
        energies_log.append((epot, ekin, t_inst))

    dyn_nvt.attach(log_energy, interval=DUMP_INTERVAL)

    t_start = time.time()
    dyn_nvt.run(PRODUCTION_STEPS)
    traj_prod.close()
    t_prod = time.time() - t_start
    print(f"  Production done in {t_prod:.1f} s "
          f"({t_prod / PRODUCTION_STEPS * 1000:.2f} ms/step)")

    # Save energy log
    energies_arr = np.array(energies_log)
    np.savetxt(str(temp_dir / "energy.dat"), energies_arr,
               header="E_pot(eV/at) E_kin(eV/at) T_inst(K)", fmt="%.6f")

    # ----- MSD ANALYSIS -----
    print("  Computing MSD for Li...")
    time_ps, msd = compute_msd(str(temp_dir / "production.traj"), species="Li",
                                start_frac=0.2)

    if time_ps is not None and len(time_ps) > 10:
        # Fit the linear regime of MSD to get diffusion coefficient
        # D = MSD / (2 * d * t), where d = 3 for 3D diffusion
        # Fit the linear part (skip first 20% and last 20%)
        n = len(time_ps)
        fit_start = max(1, n // 5)
        fit_end = 4 * n // 5

        t_fit = time_ps[fit_start:fit_end]
        msd_fit = msd[fit_start:fit_end]

        if len(t_fit) > 5:
            slope, intercept, r_value, p_value, std_err = linregress(t_fit, msd_fit)

            # D in cm^2/s: MSD in A^2, time in ps
            # 1 A^2/ps = 1e-16 m^2 / 1e-12 s = 1e-4 m^2/s = 1 cm^2/s * 1e-4... wait
            # 1 A^2/ps = (1e-10 m)^2 / (1e-12 s) = 1e-20/1e-12 = 1e-8 m^2/s = 1e-4 cm^2/s
            D_cm2_s = slope / 6.0 * 1e-4  # MSD = 6*D*t for 3D, A^2/ps -> cm^2/s

            print(f"  MSD slope = {slope:.4f} A^2/ps (R^2 = {r_value**2:.4f})")
            print(f"  D({temp}K) = {D_cm2_s:.3e} cm^2/s")

            diffusion_results[temp] = {
                "D_cm2_s": D_cm2_s,
                "slope_A2_ps": slope,
                "R2": r_value ** 2,
            }

            # Plot MSD vs time
            fig, ax = plt.subplots(figsize=(7, 5))
            ax.plot(time_ps[1:], msd[1:], "b-", linewidth=1.5, label="MSD(t)")
            ax.plot(t_fit, slope * t_fit + intercept, "r--", linewidth=2,
                    label=f"Linear fit (D = {D_cm2_s:.2e} cm$^2$/s)")
            ax.set_xlabel("Time (ps)", fontsize=13)
            ax.set_ylabel("MSD ($\\AA^2$)", fontsize=13)
            ax.set_title(f"Li MSD at {temp} K", fontsize=14)
            ax.legend(fontsize=11)
            ax.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(str(temp_dir / f"msd_{temp}K.png"), dpi=150)
            plt.close()
            print(f"  Saved: {temp_dir / f'msd_{temp}K.png'}")
        else:
            print("  WARNING: not enough data points for MSD fit")
    else:
        print("  WARNING: MSD computation failed")

# ============================================================
# 6. ARRHENIUS FIT
# ============================================================
print(f"\n{'=' * 60}")
print("ARRHENIUS ANALYSIS")
print(f"{'=' * 60}")

if len(diffusion_results) >= 3:
    temps = sorted(diffusion_results.keys())
    T_arr = np.array(temps, dtype=float)
    D_arr = np.array([diffusion_results[t]["D_cm2_s"] for t in temps])

    # ln(D) = ln(D0) - Ea / (kB * T)
    inv_T = 1000.0 / T_arr  # 1000/T in 1/K
    ln_D = np.log(D_arr)

    slope_arr, intercept_arr, r_value_arr, _, std_err_arr = linregress(inv_T, ln_D)

    # Ea = -slope * kB (slope is d(ln(D))/d(1000/T))
    kB_eV = 8.617333e-5  # eV/K
    Ea_eV = -slope_arr * kB_eV * 1000.0  # eV
    Ea_meV = Ea_eV * 1000.0  # meV
    D0 = np.exp(intercept_arr)  # pre-exponential factor, cm^2/s

    print(f"\nArrhenius fit: ln(D) = {intercept_arr:.3f} - {-slope_arr:.1f} * (1000/T)")
    print(f"  Activation energy: Ea = {Ea_eV:.3f} eV ({Ea_meV:.0f} meV)")
    print(f"  Pre-exponential: D0 = {D0:.3e} cm^2/s")
    print(f"  R^2 = {r_value_arr**2:.4f}")

    # Extrapolate to room temperature (300 K)
    D_300K = D0 * np.exp(-Ea_eV / (kB_eV * 300.0))
    print(f"\n  Extrapolated D(300K) = {D_300K:.3e} cm^2/s")

    # Conductivity estimate: sigma = n * e^2 * D / (kB * T)  (Nernst-Einstein)
    # For Li3PO4: n_Li ~ 3 per formula unit
    # This is a rough estimate
    print(f"\n  (For reference, typical Li solid electrolytes: "
          f"D ~ 1e-8 to 1e-12 cm^2/s at 300K)")

    # Plot Arrhenius
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.scatter(inv_T, ln_D, color="blue", s=80, zorder=5, label="AIMD data")

    # Fit line extended to 300K
    inv_T_ext = np.linspace(1000.0 / 1300, 1000.0 / 280, 100)
    ln_D_fit = slope_arr * inv_T_ext + intercept_arr
    ax.plot(inv_T_ext, ln_D_fit, "r--", linewidth=1.5,
            label=f"Fit: Ea = {Ea_eV:.3f} eV")

    # Mark 300K extrapolation
    inv_T_300 = 1000.0 / 300
    ln_D_300 = np.log(D_300K)
    ax.scatter([inv_T_300], [ln_D_300], color="red", marker="*", s=200, zorder=6,
               label=f"D(300K) = {D_300K:.1e} cm$^2$/s")

    ax.set_xlabel("1000/T (1/K)", fontsize=13)
    ax.set_ylabel("ln(D) [D in cm$^2$/s]", fontsize=13)
    ax.set_title("Arrhenius Plot: Li Diffusion", fontsize=14)
    ax.legend(fontsize=10, loc="lower left")
    ax.grid(True, alpha=0.3)

    # Add temperature axis on top
    ax2 = ax.twiny()
    temp_ticks = [1200, 1000, 800, 600, 400, 300]
    ax2.set_xlim(ax.get_xlim())
    ax2.set_xticks([1000.0 / t for t in temp_ticks])
    ax2.set_xticklabels([f"{t}" for t in temp_ticks])
    ax2.set_xlabel("Temperature (K)", fontsize=12)

    plt.tight_layout()
    plt.savefig(str(OUTPUT_DIR / "arrhenius_plot.png"), dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\nSaved: {OUTPUT_DIR / 'arrhenius_plot.png'}")

    # Save all results
    results = {
        "system": "Li3PO4 (gamma)",
        "method": f"MACE AIMD ({MACE_MODEL})",
        "supercell": SUPERCELL,
        "n_atoms": n_atoms,
        "timestep_fs": TIMESTEP,
        "production_steps": PRODUCTION_STEPS,
        "production_time_ps": PRODUCTION_STEPS * TIMESTEP / 1000,
        "temperatures_K": temps,
        "diffusion_coefficients": {
            str(t): {
                "D_cm2_s": diffusion_results[t]["D_cm2_s"],
                "R2": diffusion_results[t]["R2"],
            }
            for t in temps
        },
        "arrhenius": {
            "Ea_eV": float(Ea_eV),
            "Ea_meV": float(Ea_meV),
            "D0_cm2_s": float(D0),
            "R2": float(r_value_arr ** 2),
        },
        "D_300K_cm2_s": float(D_300K),
    }
    with open(OUTPUT_DIR / "diffusion_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"Saved: {OUTPUT_DIR / 'diffusion_results.json'}")

else:
    print(f"Need >= 3 temperatures for Arrhenius fit, got {len(diffusion_results)}")
    print("Check MSD analysis above for issues.")
```

### Method B: NEB for Migration Barrier

#### Complete Workflow: Li Vacancy Migration in LiCoO2

```python
#!/usr/bin/env python3
"""
NEB calculation for Li vacancy migration in LiCoO2 using ASE + MACE.

Workflow:
1. Build LiCoO2 supercell
2. Create initial and final images (Li vacancy at two adjacent sites)
3. Interpolate intermediate images
4. Run NEB optimization
5. Extract migration barrier
6. Plot energy profile along MEP

The migration path in layered LiCoO2 is within the Li layer,
hopping through a tetrahedral intermediate site (TSH mechanism).
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

from pathlib import Path
import json

from pymatgen.core import Structure, Lattice, Element
from pymatgen.io.ase import AseAtomsAdaptor

from ase.optimize import FIRE, LBFGS
from ase.constraints import ExpCellFilter
from ase.mep.neb import NEB, NEBTools
from ase.mep.neb import idpp_interpolate
from mace.calculators import mace_mp

# ============================================================
# 1. CONFIGURATION
# ============================================================
MACE_MODEL = "medium"
DEVICE = "cpu"
N_IMAGES = 7           # intermediate images (not counting endpoints)
FMAX_RELAX = 0.02      # eV/A for endpoint relaxation
FMAX_NEB = 0.05        # eV/A for NEB convergence
MAX_NEB_STEPS = 300    # max NEB optimizer steps
SUPERCELL = [2, 2, 1]  # supercell for NEB
K_SPRING = 0.1         # NEB spring constant (eV/A^2)
OUTPUT_DIR = Path("neb_li_migration")
OUTPUT_DIR.mkdir(exist_ok=True)

# ============================================================
# 2. SET UP CALCULATOR
# ============================================================
print("Loading MACE calculator...")
calc = mace_mp(model=MACE_MODEL, dispersion=False, default_dtype="float64",
               device=DEVICE)
adaptor = AseAtomsAdaptor()

def get_calc():
    """Return a fresh calculator instance for each image."""
    return mace_mp(model=MACE_MODEL, dispersion=False, default_dtype="float64",
                   device=DEVICE)

# ============================================================
# 3. BUILD LiCoO2 SUPERCELL
# ============================================================
print("Building LiCoO2 supercell...")

lco = Structure.from_spacegroup(
    "R-3m",
    Lattice.hexagonal(a=2.816, c=14.08),
    species=["Li", "Co", "O"],
    coords=[
        [0.0, 0.0, 0.5],
        [0.0, 0.0, 0.0],
        [0.0, 0.0, 0.2393],
    ],
)

supercell = lco.copy()
supercell.make_supercell(SUPERCELL)
print(f"Supercell: {supercell.formula}, {len(supercell)} atoms")

# Relax the perfect supercell first
print("Relaxing perfect supercell...")
atoms_perfect = adaptor.get_atoms(supercell)
atoms_perfect.calc = calc
ecf = ExpCellFilter(atoms_perfect, scalar_pressure=0.0)
opt = LBFGS(ecf, logfile=str(OUTPUT_DIR / "perfect_relax.log"))
opt.run(fmax=FMAX_RELAX, steps=300)
relaxed_sc = adaptor.get_structure(atoms_perfect)
print(f"Relaxed. Volume = {relaxed_sc.volume:.2f} A^3")

# ============================================================
# 4. CREATE NEB ENDPOINTS (Li vacancy at two adjacent sites)
# ============================================================
# Find Li sites in the relaxed supercell
li_indices = [i for i, s in enumerate(relaxed_sc.species) if s == Element("Li")]
print(f"Found {len(li_indices)} Li sites")

# Find the two closest Li atoms (these define the migration path)
li_coords = np.array([relaxed_sc[i].frac_coords for i in li_indices])
li_cart = np.array([relaxed_sc[i].coords for i in li_indices])

# Compute pairwise distances between Li sites
min_dist = float("inf")
pair = (0, 1)
for i in range(len(li_indices)):
    for j in range(i + 1, len(li_indices)):
        d = relaxed_sc.get_distance(li_indices[i], li_indices[j])
        if d < min_dist:
            min_dist = d
            pair = (i, j)

li_idx_start = li_indices[pair[0]]
li_idx_end = li_indices[pair[1]]
hop_dist = min_dist
print(f"Migration path: Li site {li_idx_start} -> {li_idx_end}")
print(f"Hop distance: {hop_dist:.3f} A")

# Create initial endpoint: vacancy at li_idx_start (Li removed)
# The migrating Li is at li_idx_end and will hop to the vacancy
endpoint_initial = relaxed_sc.copy()
endpoint_initial.remove_sites([li_idx_start])
print(f"Initial endpoint: {endpoint_initial.formula} (vacancy at site {li_idx_start})")

# Create final endpoint: vacancy at li_idx_end
endpoint_final = relaxed_sc.copy()
endpoint_final.remove_sites([li_idx_end])
print(f"Final endpoint: {endpoint_final.formula} (vacancy at site {li_idx_end})")

# ============================================================
# 5. RELAX ENDPOINTS WITH FIXED CELL
# ============================================================
print("\nRelaxing initial endpoint...")
atoms_init = adaptor.get_atoms(endpoint_initial)
atoms_init.calc = get_calc()
opt = LBFGS(atoms_init, logfile=str(OUTPUT_DIR / "endpoint_init_relax.log"))
opt.run(fmax=FMAX_RELAX, steps=300)
e_init = atoms_init.get_potential_energy()
print(f"  E_initial = {e_init:.6f} eV")

print("Relaxing final endpoint...")
atoms_final = adaptor.get_atoms(endpoint_final)
atoms_final.calc = get_calc()
opt = LBFGS(atoms_final, logfile=str(OUTPUT_DIR / "endpoint_final_relax.log"))
opt.run(fmax=FMAX_RELAX, steps=300)
e_final = atoms_final.get_potential_energy()
print(f"  E_final = {e_final:.6f} eV")

# ============================================================
# 6. SET UP NEB IMAGES
# ============================================================
print(f"\nSetting up NEB with {N_IMAGES} intermediate images...")

# Create copies for interpolation
images = [atoms_init.copy()]
for i in range(N_IMAGES):
    image = atoms_init.copy()
    image.calc = get_calc()
    images.append(image)
images.append(atoms_final.copy())

# Set up NEB band
neb = NEB(images, k=K_SPRING, climb=False)

# IDPP interpolation for initial path
print("Performing IDPP interpolation...")
try:
    idpp_interpolate(neb, traj=str(OUTPUT_DIR / "idpp_path.traj"))
    print("  IDPP interpolation successful")
except Exception as e:
    print(f"  IDPP failed ({e}), using linear interpolation")
    neb.interpolate()

# ============================================================
# 7. RUN NEB OPTIMIZATION (without climbing image first)
# ============================================================
print(f"\nRunning NEB optimization (fmax = {FMAX_NEB} eV/A)...")
optimizer = FIRE(neb, logfile=str(OUTPUT_DIR / "neb.log"))

# Track convergence
neb_energies_history = []
def save_neb_energies():
    energies = [img.get_potential_energy() for img in images]
    neb_energies_history.append(energies.copy())

optimizer.attach(save_neb_energies, interval=10)

converged = optimizer.run(fmax=FMAX_NEB, steps=MAX_NEB_STEPS)
if converged:
    print("  NEB converged!")
else:
    print(f"  NEB reached max steps ({MAX_NEB_STEPS})")

# ============================================================
# 8. CLIMBING IMAGE NEB (refine the saddle point)
# ============================================================
print("\nSwitching to climbing image NEB...")
neb_ci = NEB(images, k=K_SPRING, climb=True)
optimizer_ci = FIRE(neb_ci, logfile=str(OUTPUT_DIR / "neb_ci.log"))
optimizer_ci.run(fmax=FMAX_NEB, steps=MAX_NEB_STEPS)

# ============================================================
# 9. EXTRACT RESULTS
# ============================================================
# Get energies along the path
image_energies = [img.get_potential_energy() for img in images]
e_ref = image_energies[0]  # reference to initial endpoint
relative_energies = [e - e_ref for e in image_energies]

# Compute distances along the path (reaction coordinate)
nebtools = NEBTools(images)
barrier_fwd = nebtools.get_barrier()[0]  # forward barrier
barrier_rev = nebtools.get_barrier()[1]  # reverse barrier

print(f"\n{'=' * 50}")
print("NEB RESULTS")
print(f"{'=' * 50}")
print(f"Forward barrier:  {barrier_fwd:.4f} eV ({barrier_fwd * 1000:.1f} meV)")
print(f"Reverse barrier:  {barrier_rev:.4f} eV ({barrier_rev * 1000:.1f} meV)")
print(f"Reaction energy:  {relative_energies[-1]:.4f} eV")
print(f"Hop distance:     {hop_dist:.3f} A")

# Estimate diffusivity from barrier (transition state theory)
kB = 8.617333e-5  # eV/K
nu0 = 1e13        # attempt frequency (typical phonon frequency, Hz)
T_range = [300, 400, 600, 800]
print(f"\nEstimated diffusivity D = a^2 * nu0 * exp(-Ea/kB*T):")
print(f"  (nu0 = {nu0:.0e} Hz, a = {hop_dist:.2f} A)")
for T in T_range:
    D_est = (hop_dist * 1e-8) ** 2 * nu0 * np.exp(-barrier_fwd / (kB * T))
    print(f"  D({T}K) = {D_est:.3e} cm^2/s")

# ============================================================
# 10. PLOT ENERGY PROFILE
# ============================================================
# Get the MEP from NEBTools
s, E, Sfit, Efit = nebtools.get_fit()

fig, ax = plt.subplots(figsize=(8, 5))

# Fitted spline
ax.plot(Sfit, Efit - Efit[0], "b-", linewidth=2, label="Spline fit")

# Individual images
ax.scatter(s, E - E[0], color="red", s=80, zorder=5, label="NEB images")

# Mark the barrier
barrier_idx = np.argmax(E)
ax.annotate(
    f"Barrier = {barrier_fwd:.3f} eV",
    xy=(s[barrier_idx], E[barrier_idx] - E[0]),
    xytext=(s[barrier_idx] + 0.3, E[barrier_idx] - E[0] + 0.02),
    fontsize=12,
    arrowprops=dict(arrowstyle="->", color="black"),
)

ax.set_xlabel("Reaction coordinate ($\\AA$)", fontsize=13)
ax.set_ylabel("Energy (eV)", fontsize=13)
ax.set_title(f"Li Migration Barrier in LiCoO$_2$ ({MACE_MODEL} MACE)", fontsize=14)
ax.legend(fontsize=11)
ax.grid(True, alpha=0.3)
ax.axhline(0, color="gray", linestyle="--", alpha=0.3)

plt.tight_layout()
plt.savefig(str(OUTPUT_DIR / "neb_profile.png"), dpi=150, bbox_inches="tight")
plt.close()
print(f"\nSaved: {OUTPUT_DIR / 'neb_profile.png'}")

# Save numerical results
results = {
    "system": "LiCoO2 (Li vacancy migration)",
    "method": f"MACE NEB ({MACE_MODEL})",
    "supercell": SUPERCELL,
    "n_images": N_IMAGES,
    "climbing_image": True,
    "hop_distance_A": float(hop_dist),
    "forward_barrier_eV": float(barrier_fwd),
    "reverse_barrier_eV": float(barrier_rev),
    "reaction_energy_eV": float(relative_energies[-1]),
    "image_energies_eV": [float(e) for e in relative_energies],
    "image_positions_A": [float(si) for si in s],
}
with open(OUTPUT_DIR / "neb_results.json", "w") as f:
    json.dump(results, f, indent=2)
print(f"Saved: {OUTPUT_DIR / 'neb_results.json'}")

# Save trajectory
from ase.io import write as ase_write
ase_write(str(OUTPUT_DIR / "neb_path.xyz"), images)
print(f"Saved: {OUTPUT_DIR / 'neb_path.xyz'}")
```

### Method B (QE): NEB with neb.x

For DFT-level NEB barriers, use QE's `neb.x`. Below is the input file structure.

#### Generate QE NEB Input

```python
#!/usr/bin/env python3
"""
Generate QE NEB input for Li vacancy migration in LiCoO2.
Uses neb.x (not pw.x) with DFT+U.

Prerequisite: Run the ASE+MACE NEB above first to get relaxed
endpoint structures, then refine with QE NEB for publication quality.
"""

from pathlib import Path

PSEUDO_DIR = "../pseudo"
NEB_DIR = Path("qe_neb_li_migration")
NEB_DIR.mkdir(exist_ok=True)

# NEB input for neb.x
# Note: neb.x uses a different input format than pw.x
# The first and last ATOMIC_POSITIONS blocks define the endpoints

neb_input = """BEGIN
BEGIN_PATH_INPUT
&PATH
    string_method   = 'neb'
    nstep_path      = 50
    ds              = 1.0
    opt_scheme      = 'broyden'
    num_of_images   = 7
    CI_scheme       = 'auto'     ! climbing image after initial convergence
    path_thr        = 0.05       ! eV/A convergence threshold
/
END_PATH_INPUT
BEGIN_ENGINE_INPUT
&CONTROL
    prefix       = 'neb_li'
    outdir       = './tmp'
    pseudo_dir   = '{pseudo_dir}'
/
&SYSTEM
    ibrav        = 0
    nat          = 23
    ntyp         = 3
    ecutwfc      = 60.0
    ecutrho      = 600.0
    occupations  = 'smearing'
    smearing     = 'mv'
    degauss      = 0.02
    nspin        = 2
    starting_magnetization(2) = 0.3
    lda_plus_u   = .true.
    lda_plus_u_kind = 0
    Hubbard_U(2) = 3.32
/
&ELECTRONS
    conv_thr     = 1.0d-7
    mixing_beta  = 0.2
/
ATOMIC_SPECIES
    Li   6.941   Li.pbe-s-kjpaw_psl.1.0.0.UPF
    Co  58.933   Co.pbe-spn-kjpaw_psl.0.3.1.UPF
    O   15.999   o_pbe_v1.2.uspp.F.UPF
CELL_PARAMETERS angstrom
    5.632000    0.000000    0.000000
   -2.816000    4.877228    0.000000
    0.000000    0.000000   14.080000
K_POINTS automatic
    3 3 2  0 0 0
BEGIN_POSITIONS
FIRST_IMAGE
ATOMIC_POSITIONS crystal
! === INITIAL IMAGE: vacancy at site A, Li at site B ===
! (Replace these with your actual relaxed endpoint coordinates)
! Co atoms (8 sites in 2x2x1 supercell)
    Co   0.000000   0.000000   0.000000
    Co   0.500000   0.000000   0.000000
    Co   0.000000   0.500000   0.000000
    Co   0.500000   0.500000   0.000000
    Co   0.166667   0.333333   0.666667
    Co   0.666667   0.333333   0.666667
    Co   0.166667   0.833333   0.666667
    Co   0.666667   0.833333   0.666667
! Li atoms (3 remain after creating 1 vacancy from 4)
    Li   0.500000   0.000000   0.500000
    Li   0.000000   0.500000   0.500000
    Li   0.500000   0.500000   0.500000
! O atoms (12 sites)
    O    0.000000   0.000000   0.239300
    O    0.500000   0.000000   0.239300
    O    0.000000   0.500000   0.239300
    O    0.500000   0.500000   0.239300
    O    0.166667   0.333333   0.905967
    O    0.666667   0.333333   0.905967
    O    0.166667   0.833333   0.905967
    O    0.666667   0.833333   0.905967
    O    0.000000   0.000000   0.760700
    O    0.500000   0.000000   0.760700
    O    0.000000   0.500000   0.760700
    O    0.500000   0.500000   0.760700
LAST_IMAGE
ATOMIC_POSITIONS crystal
! === FINAL IMAGE: vacancy at site B, Li at site A ===
! (Same structure but the migrating Li has hopped)
! Co atoms (unchanged)
    Co   0.000000   0.000000   0.000000
    Co   0.500000   0.000000   0.000000
    Co   0.000000   0.500000   0.000000
    Co   0.500000   0.500000   0.000000
    Co   0.166667   0.333333   0.666667
    Co   0.666667   0.333333   0.666667
    Co   0.166667   0.833333   0.666667
    Co   0.666667   0.833333   0.666667
! Li atoms (the migrating Li is now at the previously vacant site)
    Li   0.000000   0.000000   0.500000
    Li   0.000000   0.500000   0.500000
    Li   0.500000   0.500000   0.500000
! O atoms (unchanged)
    O    0.000000   0.000000   0.239300
    O    0.500000   0.000000   0.239300
    O    0.000000   0.500000   0.239300
    O    0.500000   0.500000   0.239300
    O    0.166667   0.333333   0.905967
    O    0.666667   0.333333   0.905967
    O    0.166667   0.833333   0.905967
    O    0.666667   0.833333   0.905967
    O    0.000000   0.000000   0.760700
    O    0.500000   0.000000   0.760700
    O    0.000000   0.500000   0.760700
    O    0.500000   0.500000   0.760700
END_POSITIONS
END_ENGINE_INPUT
END
""".format(pseudo_dir=PSEUDO_DIR)

with open(NEB_DIR / "neb.in", "w") as f:
    f.write(neb_input)

print(f"Written: {NEB_DIR / 'neb.in'}")
print()
print("Run with:")
print(f"  cd {NEB_DIR}")
print(f"  mkdir -p tmp")
print(f"  mpirun -np 8 neb.x -inp neb.in > neb.out 2>&1")
print()
print("IMPORTANT: Replace the ATOMIC_POSITIONS above with your actual")
print("relaxed endpoint coordinates from the MACE NEB or from separate")
print("QE vc-relax calculations of each endpoint.")
print()
print("After completion, extract the barrier with:")
print("  grep 'activation energy' neb.out")
```

#### Extract QE NEB Barrier

```bash
#!/bin/bash
# extract_neb_barrier.sh -- Parse neb.x output for barrier information
# Usage: bash extract_neb_barrier.sh qe_neb_li_migration/neb.out

NEB_OUT="${1:-qe_neb_li_migration/neb.out}"

echo "=== NEB Results from $NEB_OUT ==="
echo ""

# Activation energy
echo "--- Activation Energy ---"
grep "activation energy" "$NEB_OUT"

# Path length
echo ""
echo "--- Path Length ---"
grep "path length" "$NEB_OUT"

# Image energies at final iteration
echo ""
echo "--- Final Image Energies ---"
grep -A 20 "climbing image" "$NEB_OUT" | tail -20
```

## Key Parameters

| Parameter | Recommended Value | Notes |
|-----------|-------------------|-------|
| **Supercell size** | >= 3x3x3 for cubic, >= 2x2x2 for orthorhombic | AIMD: need >100 atoms; NEB: >50 atoms to avoid vacancy-vacancy interaction |
| **AIMD temperatures** | 600, 800, 1000, 1200 K | Need 4+ temperatures for reliable Arrhenius fit |
| **AIMD production time** | >= 50 ps per temperature | Short runs give noisy MSD; 100+ ps preferred |
| **AIMD timestep** | 1--2 fs | 1 fs for light Li; 2 fs acceptable with constraints |
| **MSD fit region** | 20--80% of trajectory | Skip initial equilibration and noisy tail |
| **NEB images** | 5--9 intermediate | 7 is typical; more for complex paths |
| **NEB spring constant** | 0.05--0.1 eV/A^2 | Too stiff = poor saddle resolution; too soft = corner cutting |
| **Climbing image** | Always enable for barrier | Start without CI, then enable for refinement |
| **MACE model** | "medium" or "large" | "large" recommended for barriers |
| **Langevin friction** | 0.01--0.02 1/fs | Lower = less perturbation but slower equilibration |

## Interpreting Results

### AIMD Diffusion Coefficients

- **MSD should be linear** in the diffusive regime. If MSD is flat, the ion is trapped (too low T or too short simulation).
- **Subdiffusive** (MSD ~ t^alpha, alpha < 1): indicates correlated motion or confinement. Run longer.
- **Ballistic** (MSD ~ t^2): simulation too short, still in the ballistic regime. Need more production time.
- **R^2 of Arrhenius fit** should be > 0.95. If not, check individual MSD fits.
- **Typical Li diffusivities** in oxides at 300K: 1e-8 to 1e-14 cm^2/s depending on material.

### NEB Barriers

- **LiCoO2 Li migration**: Expect 0.3--0.8 eV depending on path (in-plane TSH vs OSH).
- **LiFePO4 Li migration**: Expect 0.2--0.4 eV along 1D channels ([010] direction).
- **LLZO (Li7La3Zr2O12)**: Expect 0.2--0.35 eV (excellent solid electrolyte).
- **Li3PO4**: Expect 0.5--0.7 eV.
- NEB barrier should approximate AIMD Ea for simple single-barrier mechanisms.

### AIMD vs NEB Comparison

| Aspect | AIMD | NEB |
|--------|------|-----|
| What it gives | D(T), Ea, diffusion mechanism | Barrier height, MEP geometry |
| Thermal effects | Included (finite T sampling) | Not included (0K path) |
| Correlated motion | Captured (concerted migration) | Only the specified path |
| Computational cost | High (50+ ps per T, 4+ temperatures) | Moderate (7-9 images, single optimization) |
| Statistical error | Large (need long runs) | Small (deterministic) |
| Best for | Quantitative D and Ea | Identifying rate-limiting step |

## Common Issues

| Issue | Cause | Solution |
|-------|-------|----------|
| MSD is flat | T too low or time too short | Increase temperature or extend simulation to >100 ps |
| MSD is noisy | Too few diffusing atoms | Use larger supercell (>150 atoms) |
| Arrhenius plot is curved | Phase transition or mechanism change at high T | Exclude outlier temperatures; check structure stability |
| NEB not converging | Poor initial path or stiff springs | Use IDPP interpolation; reduce k_spring; try FIRE optimizer |
| NEB finds wrong saddle | Path crosses periodic boundary incorrectly | Visualize the path; try different endpoint pairing |
| D(300K) is unreasonably high | Ea too low or D0 too large | Check that MACE is reliable for your system; compare with literature |
| Different barriers for forward/reverse | Asymmetric path (inequivalent sites) | Expected for some structures; report both |
| QE NEB crashes | Memory or parallelization issue | Reduce images or k-points; use `-nimage N` flag for image parallelism |
| MACE gives wrong barrier | Chemistry outside training data | Validate against single-point QE; use QE NEB for publication |
