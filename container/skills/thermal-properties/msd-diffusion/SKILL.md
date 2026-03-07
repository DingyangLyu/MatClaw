# Mean Squared Displacement (MSD) and Diffusion Coefficient

## When to Use

- Computing diffusion coefficients from MD trajectories (ASE, LAMMPS, VASP XDATCAR)
- Analyzing ionic conductivity in solid electrolytes, battery materials, or molten salts
- Determining whether a system is solid (MSD plateaus) vs liquid (MSD grows linearly)
- Computing element-resolved MSD in multi-component systems (e.g., Li diffusion in LLZO)
- Estimating ionic mobility and conductivity from the Nernst-Einstein relation
- Equivalent to VASPKIT functions 721 (MSD), 722 (diffusion coefficient), 723 (ionic conductivity)

## Method Selection

```
What trajectory format do you have?

ASE .traj file?
  --> Method A: Parse with ASE Trajectory reader

LAMMPS dump file (lammpstrj)?
  --> Method B: Parse with ASE read() or custom parser

VASP XDATCAR?
  --> Method C: Parse with pymatgen or ASE VASP reader

Need fast MSD for very long trajectories (>100k frames)?
  --> Use FFT method (all methods below support this)

Multi-element system, need per-element diffusion?
  --> All methods support element-projected MSD
```

## Prerequisites

Pre-installed: `ase`, `pymatgen`, `numpy`, `scipy`, `matplotlib`.

No additional packages required.

## Detailed Steps

### Complete MSD and Diffusion Analysis Script

This single script handles all trajectory formats, computes MSD using the efficient FFT method, performs element-resolved analysis, and extracts diffusion coefficients via the Einstein relation.

```python
#!/usr/bin/env python3
"""
Mean Squared Displacement (MSD) and Diffusion Coefficient Calculator.
Supports ASE .traj, LAMMPS dump, and VASP XDATCAR trajectories.
Uses the FFT method for O(N log N) MSD computation.
Computes total and element-resolved MSD, diffusion coefficient, and ionic conductivity.
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

# ============================================================
# 1. CONFIGURATION
# ============================================================

# --- Trajectory source (uncomment the one you need) ---
TRAJ_FORMAT = "ase"          # "ase", "lammps", or "xdatcar"
TRAJ_FILE = "md.traj"        # Path to trajectory file

# For LAMMPS:
# TRAJ_FORMAT = "lammps"
# TRAJ_FILE = "dump.lammpstrj"

# For VASP XDATCAR:
# TRAJ_FORMAT = "xdatcar"
# TRAJ_FILE = "XDATCAR"

# --- Time parameters ---
TIMESTEP_FS = 1.0            # MD timestep in femtoseconds
TRAJ_INTERVAL = 10           # Frames saved every N MD steps
N_EQUIL_FRAMES = 50          # Number of equilibration frames to skip

# --- MSD settings ---
MAX_LAG_FRACTION = 0.5       # Use first half of trajectory for MSD lags
FIT_START_FRAC = 0.2         # Start linear fit at 20% of lag range
FIT_END_FRAC = 0.8           # End linear fit at 80% of lag range

# --- Physical parameters for ionic conductivity ---
TEMPERATURE = 300.0          # K (for Nernst-Einstein)
CHARGE_DICT = {              # Formal charges for ionic conductivity
    "Li": 1, "Na": 1, "K": 1,
    "O": -2, "S": -2, "F": -1, "Cl": -1,
    "Mg": 2, "Ca": 2, "Zn": 2,
    "Al": 3, "La": 3, "Zr": 4, "Ti": 4,
}

OUTPUT_DIR = "msd_analysis"

# ============================================================
# 2. LOAD TRAJECTORY
# ============================================================

import os
os.makedirs(OUTPUT_DIR, exist_ok=True)

def load_trajectory(traj_format, traj_file):
    """Load trajectory from various formats. Returns list of ASE Atoms."""
    if traj_format == "ase":
        from ase.io import Trajectory
        traj = Trajectory(traj_file, "r")
        frames = [atoms.copy() for atoms in traj]
        traj.close()

    elif traj_format == "lammps":
        from ase.io import read
        frames = read(traj_file, index=":", format="lammps-dump-text")

    elif traj_format == "xdatcar":
        from pymatgen.io.vasp import Xdatcar
        xdatcar = Xdatcar(traj_file)
        structures = xdatcar.structures
        # Convert pymatgen structures to ASE atoms
        from pymatgen.io.ase import AseAtomsAdaptor
        adaptor = AseAtomsAdaptor()
        frames = [adaptor.get_atoms(s) for s in structures]

    else:
        raise ValueError(f"Unknown format: {traj_format}")

    return frames


print(f"Loading trajectory: {TRAJ_FILE} (format: {TRAJ_FORMAT})")
all_frames = load_trajectory(TRAJ_FORMAT, TRAJ_FILE)
print(f"  Total frames loaded: {len(all_frames)}")
print(f"  Atoms per frame: {len(all_frames[0])}")
print(f"  Chemical formula: {all_frames[0].get_chemical_formula()}")

# Skip equilibration
frames = all_frames[N_EQUIL_FRAMES:]
n_frames = len(frames)
n_atoms = len(frames[0])
print(f"  Production frames (after skipping {N_EQUIL_FRAMES}): {n_frames}")

# Time between saved frames in ps
dt_ps = TIMESTEP_FS * TRAJ_INTERVAL / 1000.0
print(f"  Time between frames: {dt_ps:.4f} ps")
print(f"  Total production time: {n_frames * dt_ps:.2f} ps")

# ============================================================
# 3. UNWRAP PBC-WRAPPED POSITIONS
# ============================================================

def unwrap_positions(frames):
    """
    Unwrap PBC-wrapped trajectories using the minimum image convention.
    Returns array of shape (n_frames, n_atoms, 3) with unwrapped Cartesian coords.
    """
    n_frames = len(frames)
    n_atoms = len(frames[0])
    positions = np.zeros((n_frames, n_atoms, 3))
    positions[0] = frames[0].get_positions()

    for i in range(1, n_frames):
        pos_curr = frames[i].get_positions()
        pos_prev = frames[i - 1].get_positions()
        cell = frames[i].get_cell()

        # Displacement in Cartesian
        disp = pos_curr - pos_prev

        # Apply minimum image convention via fractional coordinates
        cell_inv = np.linalg.inv(cell.T)  # maps Cartesian -> fractional
        scaled_disp = disp @ cell_inv.T
        scaled_disp -= np.round(scaled_disp)
        real_disp = scaled_disp @ cell

        positions[i] = positions[i - 1] + real_disp

    return positions


print("\nUnwrapping PBC-wrapped positions...")
positions = unwrap_positions(frames)
print(f"  Unwrapped positions shape: {positions.shape}")

# ============================================================
# 4. MSD USING FFT METHOD (FAST)
# ============================================================

def msd_fft_1d(x):
    """
    Compute MSD for a 1D signal using the FFT method.
    Based on: Calandrini et al., Ecole thematique de la SFN (2011).
    Algorithm: MSD(m) = <|r(t+m) - r(t)|^2>_t
    Uses the identity: MSD(m) = S2(m) - 2*S1(m)
    where S2(m) = <r(t+m)^2 + r(t)^2>_t and S1(m) = <r(t+m)*r(t)>_t (autocorrelation).
    The autocorrelation is computed via FFT.
    Complexity: O(N log N) instead of O(N^2).
    """
    N = len(x)
    # Zero-pad for FFT-based autocorrelation
    fft_x = np.fft.fft(x, n=2 * N)
    acf = np.real(np.fft.ifft(fft_x * np.conj(fft_x)))[:N]
    # Normalize by number of contributing pairs
    acf /= np.arange(N, 0, -1)

    # Compute S2 term: cumulative sum of x^2
    x2 = x ** 2
    s2 = 2 * np.sum(x2)
    # Build the S2(m) array using the recursive relation
    s2_arr = np.zeros(N)
    s2_arr[0] = s2 / N
    for m in range(1, N):
        s2_arr[m] = s2_arr[m - 1] - (x2[m - 1] + x2[N - m]) / (N - m) * (N - m + 1) / (N - m)

    # Actually, let's use the clean formulation
    # MSD(m) = (1/(N-m)) * sum_{t=0}^{N-m-1} |r(t+m) - r(t)|^2
    # = (1/(N-m)) * [sum_t r(t+m)^2 + sum_t r(t)^2 - 2*sum_t r(t+m)*r(t)]
    # The cross term is the autocorrelation (already computed via FFT).
    # For the self terms, use cumulative sums.

    cumsum_x2 = np.zeros(N + 1)
    cumsum_x2[1:] = np.cumsum(x2)

    msd = np.zeros(N)
    for m in range(N):
        n_pairs = N - m
        if n_pairs <= 0:
            break
        # sum of x[m:]^2 + sum of x[:N-m]^2
        sum_sq = (cumsum_x2[N] - cumsum_x2[m]) + (cumsum_x2[N - m] - cumsum_x2[0])
        msd[m] = sum_sq / n_pairs - 2.0 * acf[m]

    return msd


def compute_msd_fft(positions, max_lag=None):
    """
    Compute MSD for all atoms using the FFT method.
    positions: (n_frames, n_atoms, 3)
    Returns: time_lags (frames), msd (A^2)
    """
    n_frames, n_atoms, _ = positions.shape
    if max_lag is None:
        max_lag = n_frames // 2
    max_lag = min(max_lag, n_frames - 1)

    msd_total = np.zeros(n_frames)

    for atom_idx in range(n_atoms):
        for dim in range(3):
            msd_1d = msd_fft_1d(positions[:, atom_idx, dim])
            msd_total += msd_1d

    # Average over atoms
    msd_total /= n_atoms

    return np.arange(max_lag), msd_total[:max_lag]


def compute_msd_fft_by_element(positions, symbols, max_lag=None):
    """
    Compute element-resolved MSD using the FFT method.
    Returns: dict of {element: (time_lags, msd)}
    """
    n_frames, n_atoms, _ = positions.shape
    if max_lag is None:
        max_lag = n_frames // 2
    max_lag = min(max_lag, n_frames - 1)

    # Group atoms by element
    unique_elements = sorted(set(symbols))
    element_indices = {}
    for el in unique_elements:
        element_indices[el] = [i for i, s in enumerate(symbols) if s == el]

    results = {}
    for el in unique_elements:
        indices = element_indices[el]
        msd_el = np.zeros(n_frames)
        for atom_idx in indices:
            for dim in range(3):
                msd_1d = msd_fft_1d(positions[:, atom_idx, dim])
                msd_el += msd_1d
        msd_el /= len(indices)
        results[el] = (np.arange(max_lag), msd_el[:max_lag])

    return results


print("\nComputing MSD (FFT method)...")

max_lag = int(n_frames * MAX_LAG_FRACTION)
symbols = frames[0].get_chemical_symbols()

# Total MSD
time_lags, msd_total = compute_msd_fft(positions, max_lag=max_lag)
time_ps = time_lags * dt_ps
print(f"  Total MSD computed ({max_lag} lag points)")

# Element-resolved MSD
msd_by_element = compute_msd_fft_by_element(positions, symbols, max_lag=max_lag)
unique_elements = sorted(set(symbols))
print(f"  Element-resolved MSD for: {unique_elements}")

# ============================================================
# 5. DIFFUSION COEFFICIENT (EINSTEIN RELATION)
# ============================================================

def fit_diffusion(time_ps, msd, fit_start_frac=0.2, fit_end_frac=0.8):
    """
    Fit diffusion coefficient from MSD vs time using Einstein relation:
      MSD = 2*d*D*t + c   (d=3 for 3D diffusion)
    Returns D in A^2/ps and fit parameters.
    """
    n = len(time_ps)
    i_start = int(n * fit_start_frac)
    i_end = int(n * fit_end_frac)

    if i_end - i_start < 5:
        return None, None, None, None

    t_fit = time_ps[i_start:i_end]
    msd_fit = msd[i_start:i_end]

    # Linear fit: MSD = slope * t + intercept
    coeffs = np.polyfit(t_fit, msd_fit, 1)
    slope = coeffs[0]       # A^2 / ps
    intercept = coeffs[1]   # A^2

    # D = slope / (2 * d) for d dimensions
    D_A2_ps = slope / 6.0   # 3D diffusion

    # Convert to cm^2/s: 1 A^2/ps = 1e-20 m^2 / 1e-12 s = 1e-8 m^2/s = 1e-4 cm^2/s
    D_cm2_s = D_A2_ps * 1e-4

    # R^2 of fit
    msd_pred = np.polyval(coeffs, t_fit)
    ss_res = np.sum((msd_fit - msd_pred) ** 2)
    ss_tot = np.sum((msd_fit - np.mean(msd_fit)) ** 2)
    R2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0

    return D_A2_ps, D_cm2_s, R2, coeffs


print("\n=== Diffusion Coefficients (Einstein Relation) ===")
print(f"  Fit range: {FIT_START_FRAC*100:.0f}% - {FIT_END_FRAC*100:.0f}% of lag window")

# Total diffusion
D_total, D_total_cm2s, R2_total, coeffs_total = fit_diffusion(
    time_ps, msd_total, FIT_START_FRAC, FIT_END_FRAC
)

if D_total is not None:
    print(f"\n  Total (all atoms):")
    print(f"    D = {D_total:.6f} A^2/ps = {D_total_cm2s:.4e} cm^2/s")
    print(f"    R^2 = {R2_total:.6f}")
else:
    print("  WARNING: Not enough data for total diffusion fit.")

# Per-element diffusion
element_D = {}
for el in unique_elements:
    lags_el, msd_el = msd_by_element[el]
    t_el = lags_el * dt_ps
    D_el, D_el_cm2s, R2_el, coeffs_el = fit_diffusion(
        t_el, msd_el, FIT_START_FRAC, FIT_END_FRAC
    )
    element_D[el] = (D_el, D_el_cm2s, R2_el, coeffs_el)
    if D_el is not None:
        print(f"\n  {el}:")
        print(f"    D = {D_el:.6f} A^2/ps = {D_el_cm2s:.4e} cm^2/s")
        print(f"    R^2 = {R2_el:.6f}")
        n_el = len([s for s in symbols if s == el])
        print(f"    Number of {el} atoms: {n_el}")

# ============================================================
# 6. IONIC CONDUCTIVITY (NERNST-EINSTEIN)
# ============================================================

print(f"\n=== Ionic Conductivity (Nernst-Einstein, T = {TEMPERATURE} K) ===")

# sigma = (n * q^2 * D) / (k_B * T)
# where n = number density (1/A^3), q = charge (C), D = diffusion coefficient (A^2/ps)
# k_B = 8.617333e-5 eV/K = 1.380649e-23 J/K

from scipy.constants import elementary_charge, Boltzmann

volume_A3 = frames[0].get_volume()

for el in unique_elements:
    if el not in CHARGE_DICT:
        continue
    D_el, D_el_cm2s, R2_el, _ = element_D[el]
    if D_el is None or D_el <= 0:
        continue

    z = CHARGE_DICT[el]
    n_el = len([s for s in symbols if s == el])
    # Number density in 1/m^3
    n_density = n_el / (volume_A3 * 1e-30)  # 1 A^3 = 1e-30 m^3

    # D in m^2/s
    D_m2s = D_el * 1e-20 / 1e-12  # A^2/ps -> m^2/s: * 1e-20 / 1e-12 = * 1e-8

    # sigma = n * (z*e)^2 * D / (k_B * T)
    sigma = n_density * (z * elementary_charge) ** 2 * D_m2s / (Boltzmann * TEMPERATURE)
    # Convert S/m to mS/cm: 1 S/m = 0.1 S/cm = 100 mS/cm... wait
    # 1 S/m = 0.01 S/cm = 10 mS/cm
    sigma_mS_cm = sigma * 10.0

    print(f"  {el} (charge {z:+d}):")
    print(f"    Number density: {n_density:.4e} m^-3")
    print(f"    Conductivity: {sigma:.4e} S/m = {sigma_mS_cm:.4f} mS/cm")

    # Activation energy note
    print(f"    (For activation energy, compute D at multiple temperatures and fit Arrhenius)")

# ============================================================
# 7. SAVE NUMERICAL DATA
# ============================================================

# Save total MSD
msd_data = np.column_stack([time_lags, time_ps, msd_total])
header = "lag_frame  time_ps  MSD_total_A2"
np.savetxt(os.path.join(OUTPUT_DIR, "msd_total.dat"), msd_data,
           header=header, fmt="%8d %12.5f %14.6f")

# Save per-element MSD
for el in unique_elements:
    lags_el, msd_el = msd_by_element[el]
    t_el = lags_el * dt_ps
    data_el = np.column_stack([lags_el, t_el, msd_el])
    np.savetxt(os.path.join(OUTPUT_DIR, f"msd_{el}.dat"), data_el,
               header=f"lag_frame  time_ps  MSD_{el}_A2",
               fmt="%8d %12.5f %14.6f")

# Save diffusion summary
with open(os.path.join(OUTPUT_DIR, "diffusion_summary.txt"), "w") as f:
    f.write(f"# Diffusion coefficient summary\n")
    f.write(f"# Temperature: {TEMPERATURE} K\n")
    f.write(f"# Trajectory: {TRAJ_FILE}\n")
    f.write(f"# Production frames: {n_frames}, dt = {dt_ps} ps\n")
    f.write(f"# Fit range: {FIT_START_FRAC*100:.0f}%-{FIT_END_FRAC*100:.0f}% of lag window\n\n")
    f.write(f"{'Element':>8s} {'D(A^2/ps)':>12s} {'D(cm^2/s)':>14s} {'R^2':>10s} {'N_atoms':>8s}\n")
    f.write("-" * 60 + "\n")
    f.write(f"{'Total':>8s} {D_total:12.6f} {D_total_cm2s:14.4e} {R2_total:10.6f} {n_atoms:8d}\n")
    for el in unique_elements:
        D_el, D_el_cm2s, R2_el, _ = element_D[el]
        n_el = len([s for s in symbols if s == el])
        if D_el is not None:
            f.write(f"{el:>8s} {D_el:12.6f} {D_el_cm2s:14.4e} {R2_el:10.6f} {n_el:8d}\n")

print(f"\nData saved to {OUTPUT_DIR}/")

# ============================================================
# 8. PLOT MSD vs TIME WITH LINEAR FIT
# ============================================================

print("\nGenerating plots...")

# --- Plot 1: Total MSD with fit ---
fig, ax = plt.subplots(figsize=(7, 5))
ax.plot(time_ps, msd_total, "b-", linewidth=1.5, label="MSD (total)")

if coeffs_total is not None:
    fit_line = np.polyval(coeffs_total, time_ps)
    ax.plot(time_ps, fit_line, "r--", linewidth=1.2,
            label=f"Linear fit: D = {D_total_cm2s:.2e} cm$^2$/s")

    # Mark fit region
    i_start = int(len(time_ps) * FIT_START_FRAC)
    i_end = int(len(time_ps) * FIT_END_FRAC)
    ax.axvspan(time_ps[i_start], time_ps[i_end], alpha=0.1, color="red",
               label="Fit region")

ax.set_xlabel("Time (ps)", fontsize=12)
ax.set_ylabel("MSD ($\\AA^2$)", fontsize=12)
ax.set_title(f"Mean Squared Displacement - {TEMPERATURE:.0f} K", fontsize=13)
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3)
ax.set_xlim(0, time_ps[-1])
ax.set_ylim(bottom=0)
fig.tight_layout()
fig.savefig(os.path.join(OUTPUT_DIR, "msd_total.png"), dpi=150, bbox_inches="tight")
plt.close()
print(f"  Saved: {OUTPUT_DIR}/msd_total.png")

# --- Plot 2: Element-resolved MSD ---
colors = plt.cm.tab10(np.linspace(0, 1, len(unique_elements)))
fig, ax = plt.subplots(figsize=(7, 5))

for idx, el in enumerate(unique_elements):
    lags_el, msd_el = msd_by_element[el]
    t_el = lags_el * dt_ps
    ax.plot(t_el, msd_el, "-", color=colors[idx], linewidth=1.5, label=el)

    D_el, D_el_cm2s, _, coeffs_el = element_D[el]
    if coeffs_el is not None:
        fit_line_el = np.polyval(coeffs_el, t_el)
        ax.plot(t_el, fit_line_el, "--", color=colors[idx], linewidth=0.8, alpha=0.7)

ax.set_xlabel("Time (ps)", fontsize=12)
ax.set_ylabel("MSD ($\\AA^2$)", fontsize=12)
ax.set_title(f"Element-Resolved MSD - {TEMPERATURE:.0f} K", fontsize=13)
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3)
ax.set_xlim(0, time_ps[-1])
ax.set_ylim(bottom=0)
fig.tight_layout()
fig.savefig(os.path.join(OUTPUT_DIR, "msd_by_element.png"), dpi=150, bbox_inches="tight")
plt.close()
print(f"  Saved: {OUTPUT_DIR}/msd_by_element.png")

# --- Plot 3: Log-log MSD to check diffusive regime ---
fig, ax = plt.subplots(figsize=(7, 5))

# Avoid log(0)
valid = (time_ps > 0) & (msd_total > 0)
ax.loglog(time_ps[valid], msd_total[valid], "b-", linewidth=1.5, label="MSD")

# Reference slopes
t_ref = time_ps[valid]
# Ballistic: MSD ~ t^2
ax.loglog(t_ref, 0.5 * msd_total[valid][0] / t_ref[0]**2 * t_ref**2,
          "g:", linewidth=1, alpha=0.6, label="$\\sim t^2$ (ballistic)")
# Diffusive: MSD ~ t^1
ax.loglog(t_ref, 0.5 * msd_total[valid][len(valid)//2] / t_ref[len(valid)//2] * t_ref,
          "r:", linewidth=1, alpha=0.6, label="$\\sim t^1$ (diffusive)")

ax.set_xlabel("Time (ps)", fontsize=12)
ax.set_ylabel("MSD ($\\AA^2$)", fontsize=12)
ax.set_title("Log-Log MSD (Check Diffusive Regime)", fontsize=13)
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3, which="both")
fig.tight_layout()
fig.savefig(os.path.join(OUTPUT_DIR, "msd_loglog.png"), dpi=150, bbox_inches="tight")
plt.close()
print(f"  Saved: {OUTPUT_DIR}/msd_loglog.png")

# --- Plot 4: MSD components (x, y, z) for anisotropy check ---
print("\nComputing directional MSD...")

msd_xyz = np.zeros((3, max_lag))
for atom_idx in range(n_atoms):
    for dim in range(3):
        msd_1d = msd_fft_1d(positions[:, atom_idx, dim])
        msd_xyz[dim] += msd_1d[:max_lag]
msd_xyz /= n_atoms

fig, ax = plt.subplots(figsize=(7, 5))
labels = ["x", "y", "z"]
for dim in range(3):
    ax.plot(time_ps, msd_xyz[dim], "-", linewidth=1.2, label=f"MSD$_{{{labels[dim]}}}$")
ax.plot(time_ps, msd_total, "k-", linewidth=1.5, label="MSD$_{total}$")

ax.set_xlabel("Time (ps)", fontsize=12)
ax.set_ylabel("MSD ($\\AA^2$)", fontsize=12)
ax.set_title("Directional MSD (Anisotropy Check)", fontsize=13)
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3)
ax.set_xlim(0, time_ps[-1])
ax.set_ylim(bottom=0)
fig.tight_layout()
fig.savefig(os.path.join(OUTPUT_DIR, "msd_directional.png"), dpi=150, bbox_inches="tight")
plt.close()
print(f"  Saved: {OUTPUT_DIR}/msd_directional.png")

print("\n=== MSD Analysis Complete ===")
print(f"All outputs saved in: {OUTPUT_DIR}/")
```

### Arrhenius Plot for Activation Energy

When you have diffusion coefficients at multiple temperatures, use this script to extract the activation energy.

```python
#!/usr/bin/env python3
"""
Arrhenius plot for diffusion activation energy.
Input: diffusion coefficients at multiple temperatures.
Output: activation energy from ln(D) vs 1/T fit.
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.constants import Boltzmann, elementary_charge

# ============================================================
# INPUT: Diffusion coefficients at different temperatures
# Replace with your computed values
# ============================================================

temperatures_K = np.array([600, 700, 800, 900, 1000])  # Kelvin
D_cm2_s = np.array([1.2e-8, 5.1e-8, 1.8e-7, 4.9e-7, 1.1e-6])  # cm^2/s

# ============================================================
# ARRHENIUS FIT: D = D0 * exp(-Ea / kB T)
# ln(D) = ln(D0) - Ea / (kB * T)
# ============================================================

inv_T = 1000.0 / temperatures_K   # 1000/T for plotting convenience
ln_D = np.log(D_cm2_s)

# Linear fit: ln(D) = a * (1/T) + b
# where a = -Ea/kB and b = ln(D0)
# Using 1/T in 1/K (not 1000/T)
coeffs = np.polyfit(1.0 / temperatures_K, ln_D, 1)
slope = coeffs[0]      # -Ea / kB  (units: K)
intercept = coeffs[1]  # ln(D0)

# Activation energy
Ea_J = -slope * Boltzmann            # Joules
Ea_eV = Ea_J / elementary_charge     # eV
Ea_kJ_mol = Ea_J * 6.022e23 / 1000  # kJ/mol

D0 = np.exp(intercept)

print(f"=== Arrhenius Analysis ===")
print(f"  Activation energy: {Ea_eV:.4f} eV = {Ea_kJ_mol:.2f} kJ/mol")
print(f"  Pre-exponential D0: {D0:.4e} cm^2/s")
print(f"  R^2: ", end="")

# R^2
ln_D_pred = np.polyval(coeffs, 1.0 / temperatures_K)
ss_res = np.sum((ln_D - ln_D_pred) ** 2)
ss_tot = np.sum((ln_D - np.mean(ln_D)) ** 2)
R2 = 1 - ss_res / ss_tot
print(f"{R2:.6f}")

# ============================================================
# PLOT
# ============================================================

fig, ax = plt.subplots(figsize=(6, 5))

ax.semilogy(inv_T, D_cm2_s, "ro", markersize=8, label="Data")

# Fit line
inv_T_fit = np.linspace(inv_T.min() * 0.95, inv_T.max() * 1.05, 100)
D_fit = np.exp(np.polyval(coeffs, inv_T_fit / 1000.0))
ax.semilogy(inv_T_fit, D_fit, "b-", linewidth=1.5,
            label=f"Fit: $E_a$ = {Ea_eV:.3f} eV ({Ea_kJ_mol:.1f} kJ/mol)")

ax.set_xlabel("1000/T (K$^{-1}$)", fontsize=12)
ax.set_ylabel("D (cm$^2$/s)", fontsize=12)
ax.set_title("Arrhenius Plot - Diffusion Coefficient", fontsize=13)
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3, which="both")

# Add secondary x-axis for temperature
ax2 = ax.twiny()
temp_ticks = temperatures_K
ax2.set_xlim(ax.get_xlim())
ax2.set_xticks(1000.0 / temp_ticks)
ax2.set_xticklabels([f"{T:.0f}" for T in temp_ticks])
ax2.set_xlabel("Temperature (K)", fontsize=12)

fig.tight_layout()
fig.savefig("arrhenius_plot.png", dpi=150, bbox_inches="tight")
plt.close()
print(f"  Saved: arrhenius_plot.png")
```

## Key Parameters

| Parameter | Default | Notes |
|---|---|---|
| `TIMESTEP_FS` | 1.0 fs | MD integration timestep. Must match your simulation. |
| `TRAJ_INTERVAL` | 10 | Frames saved every N steps. Determines time resolution. |
| `N_EQUIL_FRAMES` | 50 | Skip equilibration. Check energy/temperature convergence first. |
| `MAX_LAG_FRACTION` | 0.5 | Use first half of trajectory for MSD lags. Using >50% leads to poor statistics at large lags. |
| `FIT_START_FRAC` | 0.2 | Start of linear fit region (fraction of lag window). Skip ballistic regime. |
| `FIT_END_FRAC` | 0.8 | End of linear fit region. Skip noisy large-lag region. |
| `TEMPERATURE` | 300 K | For Nernst-Einstein conductivity. Must match MD simulation. |
| FFT method threshold | N/A | FFT method is always used. O(N log N) vs O(N^2) for direct method. |

### Trajectory length guidelines for diffusion

- Solid-state diffusion (D ~ 1e-8 cm^2/s): need 500+ ps, ideally 1+ ns
- Liquid diffusion (D ~ 1e-5 cm^2/s): 50-100 ps sufficient
- Superionic conductors (D ~ 1e-6 cm^2/s): 100-500 ps
- Rule of thumb: MSD should reach at least 10 A^2 in the linear regime

## Interpreting Results

### MSD shape
- **Linear MSD (slope = const)**: Normal (Fickian) diffusion. Einstein relation valid.
- **MSD plateau**: Solid-like behavior. Atoms vibrate but do not migrate. D ~ 0.
- **MSD ~ t^2 at short times**: Ballistic regime (< 0.1 ps). Do not include in fit.
- **MSD ~ t^alpha with alpha < 1**: Subdiffusive. Confined motion, glassy dynamics, or cage effect. Einstein relation does not apply directly.
- **MSD ~ t^alpha with alpha > 1**: Superdiffusive. Rare; check for simulation artifacts.

### Diffusion coefficient values
- **Liquids**: D ~ 1e-5 to 1e-4 cm^2/s
- **Superionic conductors (LLZO, LGPS)**: D_Li ~ 1e-7 to 1e-6 cm^2/s at 300 K
- **Normal solids at 300 K**: D ~ 1e-12 to 1e-8 cm^2/s (very slow, may not see diffusion in short MD)
- **D < 0**: Unphysical. Indicates MSD is not in the linear regime or trajectory is too short.

### Anisotropy
- If MSD_x, MSD_y, MSD_z differ significantly, diffusion is anisotropic.
- Common in layered materials (e.g., Li diffusion in graphite is 2D).
- For 2D diffusion: D = slope / (4) instead of slope / (6).
- For 1D diffusion: D = slope / (2).

### Ionic conductivity
- Nernst-Einstein gives an upper bound (neglects correlation / Haven ratio).
- True conductivity = sigma_NE / H_R, where Haven ratio H_R is typically 0.3-1.0.
- For correlated ion motion (concerted migration), compute the collective MSD instead.

## Common Issues

| Issue | Cause | Fix |
|---|---|---|
| MSD shows no diffusion (flat) | System is solid at this temperature, or run too short | Increase T above superionic transition; run for 1+ ns |
| Negative diffusion coefficient | MSD decreasing at large lags (poor statistics) | Reduce `MAX_LAG_FRACTION` to 0.3; increase trajectory length |
| Large R^2 but wrong D | Fitting ballistic or sub-diffusive region | Check log-log plot; adjust `FIT_START_FRAC` to skip non-linear regime |
| MSD noisy at large lag times | Too few independent samples at large lags | Normal behavior. Use smaller `FIT_END_FRAC` (e.g., 0.6) |
| Element-resolved MSD has huge variance | Too few atoms of that element | Increase supercell size; average over multiple trajectories |
| PBC unwrapping artifacts | Atoms jumped more than half a cell in one step | Reduce trajectory frame interval; check timestep is correct |
| Conductivity too high | Nernst-Einstein overestimate; Haven ratio ignored | Apply Haven ratio correction (H_R ~ 0.3-0.5 for most systems) |
| XDATCAR has wrong number of frames | VASP restarted or XDATCAR concatenated incorrectly | Check XDATCAR consistency; use `pymatgen.io.vasp.Xdatcar` parser |
