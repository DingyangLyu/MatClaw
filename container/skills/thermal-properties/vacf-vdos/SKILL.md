# Velocity Autocorrelation Function (VACF) and Vibrational Density of States (VDOS)

## When to Use

- Computing the vibrational density of states (phonon DOS) from MD trajectories
- Obtaining element-projected VDOS to identify which atoms contribute to which frequency range
- Comparing MD-derived VDOS with harmonic phonon DOS from lattice dynamics
- Analyzing anharmonic effects on vibrational spectra (peak broadening, shifts)
- Computing the velocity autocorrelation function (VACF) to study single-particle dynamics
- Extracting the diffusion coefficient from the VACF integral (Green-Kubo relation)
- Equivalent to VASPKIT functions 727 (VACF) and 728 (VDOS)

## Method Selection

```
What do you need?

VDOS (phonon DOS from MD)?
  --> Section A: Compute VACF, then FFT to get VDOS

Element-projected VDOS?
  --> Section B: Compute VACF per element, then FFT

Compare with harmonic phonon DOS?
  --> Section C: Overlay VDOS with phonopy phonon DOS

Diffusion coefficient from VACF?
  --> Section D: Integrate VACF (Green-Kubo)

What trajectory format?
  Need velocities! Check:
  - ASE .traj with velocities stored  --> Direct read
  - LAMMPS dump with vx vy vz columns --> Parse velocity columns
  - VASP XDATCAR (positions only)     --> Compute velocities via finite differences
```

## Prerequisites

Pre-installed: `ase`, `pymatgen`, `numpy`, `scipy`, `matplotlib`.

Optional: `phonopy` (for harmonic DOS comparison).

## Detailed Steps

### Complete VACF and VDOS Analysis Script

```python
#!/usr/bin/env python3
"""
Velocity Autocorrelation Function (VACF) and Vibrational Density of States (VDOS)
from MD trajectories.
Supports ASE .traj, LAMMPS dump (with velocities), and VASP XDATCAR.
For XDATCAR (no velocities stored), velocities are computed from finite differences.
Includes element-projected VDOS and comparison with harmonic phonon DOS.
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import os

# ============================================================
# 1. CONFIGURATION
# ============================================================

# --- Trajectory source ---
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
TRAJ_INTERVAL = 1            # Frames saved every N MD steps
                              # NOTE: For VDOS, save EVERY step (TRAJ_INTERVAL=1)
                              # to capture high-frequency modes correctly.

N_EQUIL_FRAMES = 100         # Skip equilibration frames

# --- VACF/VDOS settings ---
MAX_LAG_FRACTION = 0.25      # Fraction of trajectory for VACF lag
WINDOW_FUNCTION = "hann"     # FFT window: "hann", "blackman", "none"
FREQ_UNIT = "THz"            # "THz", "cm-1", or "meV"
ZERO_PAD_FACTOR = 4          # Zero-padding for FFT (improves frequency resolution)

# --- Comparison with harmonic DOS ---
PHONOPY_DOS_FILE = ""        # Path to phonopy total_dos.dat (leave empty to skip)

OUTPUT_DIR = "vacf_vdos_analysis"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ============================================================
# 2. LOAD TRAJECTORY AND EXTRACT VELOCITIES
# ============================================================

def load_velocities(traj_format, traj_file, timestep_fs, traj_interval):
    """
    Load velocities from trajectory.
    Returns:
        velocities: (n_frames, n_atoms, 3) in A/fs
        symbols: list of element symbols
        masses: array of atomic masses (amu)
    """
    if traj_format == "ase":
        from ase.io import Trajectory
        traj = Trajectory(traj_file, "r")
        frames = [atoms.copy() for atoms in traj]
        traj.close()

        # ASE stores velocities in A/fs (internal units) if set during MD
        symbols = frames[0].get_chemical_symbols()
        masses = frames[0].get_masses()

        velocities = []
        has_velocities = True
        for atoms in frames:
            try:
                v = atoms.get_velocities()
                if v is None or np.all(v == 0):
                    has_velocities = False
                    break
                velocities.append(v)
            except Exception:
                has_velocities = False
                break

        if has_velocities:
            print("  Velocities loaded directly from ASE trajectory.")
            return np.array(velocities), symbols, masses

        # Fall back to finite differences
        print("  No velocities in trajectory. Computing from finite differences...")
        dt = timestep_fs * traj_interval  # fs between frames
        positions = np.array([f.get_positions() for f in frames])
        velocities = _velocities_from_positions(positions, frames, dt)
        return velocities, symbols, masses

    elif traj_format == "lammps":
        from ase.io import read
        frames = read(traj_file, index=":", format="lammps-dump-text")
        symbols = frames[0].get_chemical_symbols()
        masses = frames[0].get_masses()

        # Try to read velocities
        has_vel = True
        velocities = []
        for atoms in frames:
            try:
                v = atoms.get_velocities()
                if v is None or np.all(v == 0):
                    has_vel = False
                    break
                velocities.append(v)
            except Exception:
                has_vel = False
                break

        if has_vel:
            print("  Velocities loaded from LAMMPS dump.")
            return np.array(velocities), symbols, masses

        print("  No velocities in LAMMPS dump. Computing from finite differences...")
        dt = timestep_fs * traj_interval
        positions = np.array([f.get_positions() for f in frames])
        velocities = _velocities_from_positions(positions, frames, dt)
        return velocities, symbols, masses

    elif traj_format == "xdatcar":
        from pymatgen.io.vasp import Xdatcar
        from pymatgen.io.ase import AseAtomsAdaptor
        xdatcar = Xdatcar(traj_file)
        adaptor = AseAtomsAdaptor()
        frames = [adaptor.get_atoms(s) for s in xdatcar.structures]

        symbols = frames[0].get_chemical_symbols()
        masses = frames[0].get_masses()

        print("  XDATCAR: computing velocities from finite differences...")
        dt = timestep_fs * traj_interval
        positions = np.array([f.get_positions() for f in frames])
        velocities = _velocities_from_positions(positions, frames, dt)
        return velocities, symbols, masses

    else:
        raise ValueError(f"Unknown format: {traj_format}")


def _velocities_from_positions(positions, frames, dt_fs):
    """
    Compute velocities from positions using central finite differences.
    Handles PBC wrapping via minimum image convention.
    positions: (n_frames, n_atoms, 3) - Cartesian, possibly wrapped
    dt_fs: time between frames in femtoseconds
    Returns: velocities (n_frames-2, n_atoms, 3) in A/fs
    """
    n_frames = len(positions)
    n_atoms = positions.shape[1]
    velocities = np.zeros((n_frames - 2, n_atoms, 3))

    for i in range(1, n_frames - 1):
        cell = frames[i].get_cell()

        # Forward and backward displacements
        disp_fwd = positions[i + 1] - positions[i]
        disp_bwd = positions[i] - positions[i - 1]

        # Minimum image convention
        for disp in [disp_fwd, disp_bwd]:
            scaled = np.linalg.solve(cell.T, disp.T).T
            scaled -= np.round(scaled)
            disp[:] = scaled @ cell

        # Central difference: v(t) = [r(t+dt) - r(t-dt)] / (2*dt)
        total_disp = disp_fwd + disp_bwd  # r(t+1) - r(t) + r(t) - r(t-1) = r(t+1) - r(t-1)
        # Wait, we need: r(t+dt) - r(t-dt) = disp_fwd + disp_bwd... no.
        # disp_fwd = r(t+1) - r(t) (unwrapped)
        # disp_bwd = r(t) - r(t-1) (unwrapped)
        # r(t+1) - r(t-1) = disp_fwd + disp_bwd
        velocities[i - 1] = (disp_fwd + disp_bwd) / (2.0 * dt_fs)

    return velocities


print(f"Loading trajectory: {TRAJ_FILE} (format: {TRAJ_FORMAT})")
all_velocities, symbols, masses = load_velocities(
    TRAJ_FORMAT, TRAJ_FILE, TIMESTEP_FS, TRAJ_INTERVAL
)
print(f"  Total velocity frames: {all_velocities.shape[0]}")
print(f"  Atoms: {all_velocities.shape[1]}")
print(f"  Elements: {sorted(set(symbols))}")

# Skip equilibration
velocities = all_velocities[N_EQUIL_FRAMES:]
n_frames, n_atoms, _ = velocities.shape
dt_ps = TIMESTEP_FS * TRAJ_INTERVAL / 1000.0  # ps between frames
print(f"  Production frames: {n_frames}")
print(f"  Time step: {dt_ps:.6f} ps ({TIMESTEP_FS * TRAJ_INTERVAL:.2f} fs)")
print(f"  Total time: {n_frames * dt_ps:.2f} ps")

# ============================================================
# 3. SECTION A: VELOCITY AUTOCORRELATION FUNCTION (VACF)
# ============================================================

print("\n=== Section A: Velocity Autocorrelation Function ===")


def compute_vacf_fft(velocities, max_lag=None):
    """
    Compute the mass-weighted VACF using FFT for efficiency.
    VACF(t) = <v(t0) . v(t0 + t)>_{t0, atoms}
    Uses the Wiener-Khinchin theorem: autocorrelation = IFFT(|FFT(v)|^2).
    Returns normalized VACF (VACF(0) = 1).
    """
    n_frames, n_atoms, _ = velocities.shape
    if max_lag is None:
        max_lag = n_frames // 4

    # Pad to next power of 2 for FFT efficiency
    n_fft = 2 ** int(np.ceil(np.log2(2 * n_frames)))

    vacf = np.zeros(n_fft)

    for atom_idx in range(n_atoms):
        for dim in range(3):
            v = velocities[:, atom_idx, dim]
            # FFT-based autocorrelation
            fft_v = np.fft.fft(v, n=n_fft)
            power = np.real(fft_v * np.conj(fft_v))
            acf = np.real(np.fft.ifft(power))
            # Normalize by number of overlapping pairs
            norm = np.arange(n_frames, n_frames - n_fft, -1)
            norm = np.maximum(norm, 1)
            # Only first n_frames points are meaningful
            acf_normalized = acf[:n_frames] / np.arange(n_frames, 0, -1)
            vacf[:n_frames] += acf_normalized

    vacf = vacf[:n_frames]
    vacf /= n_atoms  # Average over atoms (already summed over 3 dimensions)

    # Normalize: VACF(0) = 1
    vacf_normalized = vacf / vacf[0] if vacf[0] != 0 else vacf

    return vacf_normalized[:max_lag], vacf[:max_lag]


def compute_vacf_by_element(velocities, symbols, max_lag=None):
    """Compute element-resolved VACF."""
    n_frames, n_atoms, _ = velocities.shape
    if max_lag is None:
        max_lag = n_frames // 4

    unique_elements = sorted(set(symbols))
    n_fft = 2 ** int(np.ceil(np.log2(2 * n_frames)))

    results = {}
    for el in unique_elements:
        indices = [i for i, s in enumerate(symbols) if s == el]
        vacf_el = np.zeros(n_frames)

        for atom_idx in indices:
            for dim in range(3):
                v = velocities[:, atom_idx, dim]
                fft_v = np.fft.fft(v, n=n_fft)
                power = np.real(fft_v * np.conj(fft_v))
                acf = np.real(np.fft.ifft(power))[:n_frames]
                acf /= np.arange(n_frames, 0, -1)
                vacf_el += acf

        vacf_el /= len(indices)
        vacf_norm = vacf_el / vacf_el[0] if vacf_el[0] != 0 else vacf_el
        results[el] = (vacf_norm[:max_lag], vacf_el[:max_lag])

    return results


max_lag = int(n_frames * MAX_LAG_FRACTION)
vacf_norm, vacf_raw = compute_vacf_fft(velocities, max_lag=max_lag)
time_vacf = np.arange(max_lag) * dt_ps  # ps

print(f"  VACF computed ({max_lag} lag points, {time_vacf[-1]:.2f} ps)")
print(f"  VACF(0) = {vacf_norm[0]:.6f} (should be 1.0)")

# Zero-crossing time (characteristic of vibrational period)
zero_crossings = np.where(np.diff(np.sign(vacf_norm)))[0]
if len(zero_crossings) > 0:
    t_cross = time_vacf[zero_crossings[0]]
    print(f"  First zero-crossing: {t_cross:.4f} ps ({t_cross * 1000:.2f} fs)")

# Element-resolved VACF
vacf_by_element = compute_vacf_by_element(velocities, symbols, max_lag=max_lag)

# Save VACF
np.savetxt(os.path.join(OUTPUT_DIR, "vacf_total.dat"),
           np.column_stack([time_vacf, vacf_norm, vacf_raw]),
           header="time_ps  VACF_normalized  VACF_raw",
           fmt="%12.6f %14.8f %14.8e")

# Plot VACF
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# Full VACF
axes[0].plot(time_vacf, vacf_norm, "b-", linewidth=1.2)
axes[0].axhline(y=0, color="gray", linestyle="--", linewidth=0.5)
axes[0].set_xlabel("Time (ps)", fontsize=12)
axes[0].set_ylabel("VACF (normalized)", fontsize=12)
axes[0].set_title("Velocity Autocorrelation Function", fontsize=13)
axes[0].grid(True, alpha=0.3)

# Short-time VACF (first 10% of lag)
short_cut = max(max_lag // 10, 20)
axes[1].plot(time_vacf[:short_cut], vacf_norm[:short_cut], "b-", linewidth=1.5)
axes[1].axhline(y=0, color="gray", linestyle="--", linewidth=0.5)
axes[1].set_xlabel("Time (ps)", fontsize=12)
axes[1].set_ylabel("VACF (normalized)", fontsize=12)
axes[1].set_title("VACF (short time)", fontsize=13)
axes[1].grid(True, alpha=0.3)

fig.tight_layout()
fig.savefig(os.path.join(OUTPUT_DIR, "vacf.png"), dpi=150, bbox_inches="tight")
plt.close()
print(f"  Saved: {OUTPUT_DIR}/vacf.png")

# Element-resolved VACF plot
fig, ax = plt.subplots(figsize=(7, 5))
colors = plt.cm.tab10(np.linspace(0, 1, len(vacf_by_element)))
for idx, (el, (vacf_el_norm, _)) in enumerate(vacf_by_element.items()):
    ax.plot(time_vacf, vacf_el_norm, "-", color=colors[idx], linewidth=1.2, label=el)
ax.axhline(y=0, color="gray", linestyle="--", linewidth=0.5)
ax.set_xlabel("Time (ps)", fontsize=12)
ax.set_ylabel("VACF (normalized)", fontsize=12)
ax.set_title("Element-Resolved VACF", fontsize=13)
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3)
fig.tight_layout()
fig.savefig(os.path.join(OUTPUT_DIR, "vacf_by_element.png"), dpi=150, bbox_inches="tight")
plt.close()
print(f"  Saved: {OUTPUT_DIR}/vacf_by_element.png")

# ============================================================
# 4. SECTION B: VIBRATIONAL DENSITY OF STATES (VDOS)
# ============================================================

print("\n=== Section B: Vibrational Density of States (VDOS) ===")


def vacf_to_vdos(vacf_raw, dt_ps, window="hann", zero_pad_factor=4, freq_unit="THz"):
    """
    Compute VDOS from VACF via Fourier transform.
    VDOS(omega) = integral_0^inf VACF(t) * cos(omega*t) dt
                = Re[FT(VACF)]
    """
    n = len(vacf_raw)

    # Apply window function to reduce spectral leakage
    if window == "hann":
        w = np.hanning(2 * n)[n:]
    elif window == "blackman":
        w = np.blackman(2 * n)[n:]
    elif window == "none":
        w = np.ones(n)
    else:
        w = np.ones(n)

    vacf_windowed = vacf_raw * w

    # Zero-pad for better frequency resolution
    n_fft = n * zero_pad_factor
    n_fft = 2 ** int(np.ceil(np.log2(n_fft)))

    # Compute FFT (real part = cosine transform for real, even-like signal)
    fft_result = np.fft.rfft(vacf_windowed, n=n_fft)
    vdos = np.real(fft_result)

    # Frequency axis
    freq_ps = np.fft.rfftfreq(n_fft, d=dt_ps)  # 1/ps = THz

    # Convert frequency units
    if freq_unit == "THz":
        freq = freq_ps  # Already in THz (1/ps)
        freq_label = "Frequency (THz)"
    elif freq_unit == "cm-1":
        freq = freq_ps * 33.3564  # 1 THz = 33.3564 cm^-1
        freq_label = "Frequency (cm$^{-1}$)"
    elif freq_unit == "meV":
        freq = freq_ps * 4.13567  # 1 THz = 4.13567 meV
        freq_label = "Frequency (meV)"
    else:
        freq = freq_ps
        freq_label = "Frequency (THz)"

    # Normalize VDOS so integral = 1 (or 3N for total DOS)
    df = freq[1] - freq[0] if len(freq) > 1 else 1.0
    vdos_pos = np.maximum(vdos, 0)  # Physical VDOS is non-negative
    integral = np.trapz(vdos_pos, freq)
    if integral > 0:
        vdos_normalized = vdos_pos / integral
    else:
        vdos_normalized = vdos_pos

    return freq, vdos_normalized, freq_label


# Total VDOS
freq_total, vdos_total, freq_label = vacf_to_vdos(
    vacf_raw, dt_ps, window=WINDOW_FUNCTION,
    zero_pad_factor=ZERO_PAD_FACTOR, freq_unit=FREQ_UNIT
)

# Nyquist frequency check
nyquist_thz = 1.0 / (2.0 * dt_ps)
print(f"  Nyquist frequency: {nyquist_thz:.1f} THz")
print(f"  (Modes above this are aliased. For high-freq modes, reduce TRAJ_INTERVAL.)")

# Find VDOS peaks
from scipy.signal import find_peaks
vdos_peaks, vdos_props = find_peaks(vdos_total, height=np.max(vdos_total) * 0.05,
                                      distance=len(freq_total) // 100, prominence=0.001)

print(f"  VDOS peaks:")
for p in vdos_peaks[:10]:
    print(f"    {freq_total[p]:.2f} {FREQ_UNIT}")

# Save VDOS
np.savetxt(os.path.join(OUTPUT_DIR, "vdos_total.dat"),
           np.column_stack([freq_total, vdos_total]),
           header=f"frequency({FREQ_UNIT})  VDOS",
           fmt="%12.4f %14.8f")

# Element-projected VDOS
unique_elements = sorted(set(symbols))
vdos_by_element = {}
for el, (_, vacf_el_raw) in vacf_by_element.items():
    freq_el, vdos_el, _ = vacf_to_vdos(
        vacf_el_raw, dt_ps, window=WINDOW_FUNCTION,
        zero_pad_factor=ZERO_PAD_FACTOR, freq_unit=FREQ_UNIT
    )
    # Weight by number of atoms of this element
    n_el = len([s for s in symbols if s == el])
    vdos_by_element[el] = (freq_el, vdos_el, n_el)

    np.savetxt(os.path.join(OUTPUT_DIR, f"vdos_{el}.dat"),
               np.column_stack([freq_el, vdos_el]),
               header=f"frequency({FREQ_UNIT})  VDOS_{el}",
               fmt="%12.4f %14.8f")

# ============================================================
# 5. PLOT VDOS
# ============================================================

# --- Plot 1: Total VDOS ---
fig, ax = plt.subplots(figsize=(8, 5))
ax.fill_between(freq_total, vdos_total, alpha=0.3, color="steelblue")
ax.plot(freq_total, vdos_total, "b-", linewidth=1.5, label="Total VDOS (from MD)")

# Label peaks
for p in vdos_peaks[:6]:
    ax.annotate(f"{freq_total[p]:.1f}",
                xy=(freq_total[p], vdos_total[p]),
                xytext=(freq_total[p], vdos_total[p] * 1.1),
                fontsize=9, color="red", ha="center")

ax.set_xlabel(freq_label, fontsize=12)
ax.set_ylabel("VDOS (arb. units)", fontsize=12)
ax.set_title("Vibrational Density of States (from MD)", fontsize=13)
ax.set_xlim(0, min(freq_total[-1], nyquist_thz * {"THz": 1, "cm-1": 33.3564, "meV": 4.13567}[FREQ_UNIT]))
ax.set_ylim(bottom=0)
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3)
fig.tight_layout()
fig.savefig(os.path.join(OUTPUT_DIR, "vdos_total.png"), dpi=150, bbox_inches="tight")
plt.close()
print(f"  Saved: {OUTPUT_DIR}/vdos_total.png")

# --- Plot 2: Element-projected VDOS (stacked) ---
fig, ax = plt.subplots(figsize=(8, 5))
colors_el = plt.cm.Set2(np.linspace(0, 1, len(unique_elements)))
bottom = np.zeros_like(freq_total)

for idx, el in enumerate(unique_elements):
    freq_el, vdos_el, n_el = vdos_by_element[el]
    # Interpolate to common frequency grid
    vdos_interp = np.interp(freq_total, freq_el, vdos_el)
    # Weight by fraction of atoms
    weight = n_el / n_atoms
    vdos_weighted = vdos_interp * weight

    ax.fill_between(freq_total, bottom, bottom + vdos_weighted,
                    alpha=0.5, color=colors_el[idx], label=f"{el} (n={n_el})")
    bottom += vdos_weighted

ax.set_xlabel(freq_label, fontsize=12)
ax.set_ylabel("VDOS (arb. units)", fontsize=12)
ax.set_title("Element-Projected VDOS (Stacked)", fontsize=13)
ax.set_xlim(0, min(freq_total[-1], nyquist_thz * {"THz": 1, "cm-1": 33.3564, "meV": 4.13567}[FREQ_UNIT]))
ax.set_ylim(bottom=0)
ax.legend(fontsize=9, loc="upper right")
ax.grid(True, alpha=0.3)
fig.tight_layout()
fig.savefig(os.path.join(OUTPUT_DIR, "vdos_element_stacked.png"), dpi=150, bbox_inches="tight")
plt.close()
print(f"  Saved: {OUTPUT_DIR}/vdos_element_stacked.png")

# --- Plot 3: Element VDOS overlay ---
fig, ax = plt.subplots(figsize=(8, 5))
for idx, el in enumerate(unique_elements):
    freq_el, vdos_el, n_el = vdos_by_element[el]
    ax.plot(freq_el, vdos_el, "-", color=colors_el[idx], linewidth=1.5, label=f"{el} (n={n_el})")

ax.set_xlabel(freq_label, fontsize=12)
ax.set_ylabel("VDOS (arb. units)", fontsize=12)
ax.set_title("Element-Projected VDOS (Overlay)", fontsize=13)
ax.set_xlim(0, min(freq_total[-1], nyquist_thz * {"THz": 1, "cm-1": 33.3564, "meV": 4.13567}[FREQ_UNIT]))
ax.set_ylim(bottom=0)
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3)
fig.tight_layout()
fig.savefig(os.path.join(OUTPUT_DIR, "vdos_element_overlay.png"), dpi=150, bbox_inches="tight")
plt.close()
print(f"  Saved: {OUTPUT_DIR}/vdos_element_overlay.png")

# ============================================================
# 6. SECTION C: COMPARE WITH HARMONIC PHONON DOS
# ============================================================

print("\n=== Section C: Comparison with Harmonic Phonon DOS ===")

if PHONOPY_DOS_FILE and os.path.exists(PHONOPY_DOS_FILE):
    # Parse phonopy total_dos.dat
    phonopy_data = np.loadtxt(PHONOPY_DOS_FILE, comments="#")
    freq_harmonic = phonopy_data[:, 0]  # THz
    dos_harmonic = phonopy_data[:, 1]

    # Convert to same frequency unit
    if FREQ_UNIT == "cm-1":
        freq_harmonic *= 33.3564
    elif FREQ_UNIT == "meV":
        freq_harmonic *= 4.13567

    # Normalize for comparison
    integral_h = np.trapz(dos_harmonic, freq_harmonic)
    if integral_h > 0:
        dos_harmonic /= integral_h

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.fill_between(freq_total, vdos_total, alpha=0.3, color="steelblue")
    ax.plot(freq_total, vdos_total, "b-", linewidth=1.5, label="VDOS (MD, anharmonic)")
    ax.plot(freq_harmonic, dos_harmonic, "r-", linewidth=1.5, label="Phonon DOS (harmonic)")

    ax.set_xlabel(freq_label, fontsize=12)
    ax.set_ylabel("DOS (normalized)", fontsize=12)
    ax.set_title("VDOS vs Harmonic Phonon DOS", fontsize=13)
    ax.legend(fontsize=10)
    ax.set_xlim(0, min(freq_total[-1], nyquist_thz * {"THz": 1, "cm-1": 33.3564, "meV": 4.13567}[FREQ_UNIT]))
    ax.set_ylim(bottom=0)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(OUTPUT_DIR, "vdos_vs_harmonic.png"), dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {OUTPUT_DIR}/vdos_vs_harmonic.png")
else:
    print("  No phonopy DOS file provided. Skipping comparison.")
    print("  To generate one, run the phonon/ skill and use the total_dos.dat output.")

# ============================================================
# 7. SECTION D: DIFFUSION COEFFICIENT FROM VACF (GREEN-KUBO)
# ============================================================

print("\n=== Section D: Diffusion Coefficient (Green-Kubo) ===")

# Green-Kubo: D = (1/3) * integral_0^inf <v(0).v(t)> dt
# Using the unnormalized VACF (vacf_raw has units of (A/fs)^2 if velocities are in A/fs)

# The raw VACF from compute_vacf_fft is the sum over 3 dimensions, averaged over atoms.
# <v(0).v(t)> = sum_{dim} <v_dim(0) * v_dim(t)>
# D = (1/3) * integral <v(0).v(t)> dt

# Integrate using trapezoidal rule
D_gk_A2_fs = np.trapz(vacf_raw, time_vacf * 1000.0) / 3.0  # integral in A^2/fs (time in fs)
# Convert: 1 A^2/fs = 1e-20 m^2 / 1e-15 s = 1e-5 m^2/s = 1e-1 cm^2/s
# Wait, let's be careful. vacf_raw is in (A/fs)^2 averaged over atoms, summed over dims.
# Actually the units depend on how velocities were stored. ASE uses A/(ase_time_unit).
# If velocities are in A/fs: [VACF] = A^2/fs^2
# integral VACF dt (with dt in fs) = A^2/fs
# That gives nonsensical D. Let's convert properly.

# If VACF is in (A/ps)^2 (ASE internal velocities converted):
# integral VACF*dt (ps) = A^2/ps
# D = integral/3 in A^2/ps
# D (cm^2/s) = D (A^2/ps) * 1e-4

# Let's just use the normalized VACF and compute D from VACF(0):
# <v^2> = 3*kB*T/m  (equipartition)
# D = (kB*T)/(m) * integral_0^inf VACF_norm(t) dt

# Simpler approach: compute from the raw VACF integral
# The raw VACF already has proper units if we know the velocity units.
# ASE velocities are in A/fs by default. Let's assume this.

# vacf_raw = <v(0).v(t)> summed over 3 dims, averaged over atoms
# Units if v in A/fs: vacf_raw in A^2/fs^2
# D = (1/3) * integral vacf_raw dt
# dt_ps, so we need to convert time to same unit:
# integral in A^2/fs^2 * ps = A^2/fs^2 * 1000 fs = 1000 * A^2/fs

# Let's use time in ps directly and handle units:
# vacf_raw is dimensionless in our code because we normalize differently.
# Instead, let's use the known relation:

# From the normalized VACF:
# D = <v^2>/3 * integral_0^inf VACF_norm(t) dt
# <v^2> = 3*kB*T/m (per atom, 3D)
# So D = kB*T/m * integral VACF_norm(t) dt

# For a multi-element system, compute effective D:
from scipy.constants import Boltzmann, atomic_mass as amu

avg_mass_kg = np.mean(masses) * amu  # kg
kBT = Boltzmann * 300.0  # J (use actual temperature from MD)

# Integral of normalized VACF in ps
integral_vacf = np.trapz(vacf_norm, time_vacf)  # ps

# D = kBT / m * integral  [m^2/s if we convert ps to s]
# kBT/m = J / kg = m^2/s^2
# integral in ps = integral * 1e-12 s
D_gk_m2s = (kBT / avg_mass_kg) * integral_vacf * 1e-12
D_gk_cm2s = D_gk_m2s * 1e4
D_gk_A2ps = D_gk_m2s * 1e20 / 1e12  # m^2/s -> A^2/ps: * 1e20 / 1e12 = *1e8... no
# 1 m^2/s = 1e20 A^2/s = 1e20 * 1e-12 A^2/ps = 1e8 A^2/ps
D_gk_A2ps = D_gk_m2s * 1e8

print(f"  Green-Kubo diffusion coefficient:")
print(f"    D = {D_gk_A2ps:.6f} A^2/ps = {D_gk_cm2s:.4e} cm^2/s")
print(f"    VACF integral: {integral_vacf:.6f} ps")
print(f"    Average mass: {np.mean(masses):.2f} amu")
print(f"  Note: Compare with Einstein relation (MSD) result for consistency.")

# Save summary
with open(os.path.join(OUTPUT_DIR, "vacf_vdos_summary.txt"), "w") as f:
    f.write(f"# VACF/VDOS Analysis Summary\n")
    f.write(f"# Trajectory: {TRAJ_FILE}\n")
    f.write(f"# Production frames: {n_frames}\n")
    f.write(f"# Time step: {dt_ps} ps, total time: {n_frames * dt_ps:.2f} ps\n")
    f.write(f"# Nyquist frequency: {nyquist_thz:.1f} THz\n\n")
    f.write(f"VACF integral: {integral_vacf:.6f} ps\n")
    f.write(f"Green-Kubo D: {D_gk_A2ps:.6f} A^2/ps = {D_gk_cm2s:.4e} cm^2/s\n\n")
    f.write(f"VDOS peaks ({FREQ_UNIT}):\n")
    for p in vdos_peaks[:10]:
        f.write(f"  {freq_total[p]:.2f}\n")

print(f"\n=== VACF/VDOS Analysis Complete ===")
print(f"All outputs in: {OUTPUT_DIR}/")
```

## Key Parameters

| Parameter | Default | Notes |
|---|---|---|
| `TIMESTEP_FS` | 1.0 fs | MD integration timestep. Must match simulation. |
| `TRAJ_INTERVAL` | 1 | Save every N steps. For VDOS, use 1 (every step) to capture high frequencies. |
| `N_EQUIL_FRAMES` | 100 | Equilibration frames to skip. |
| `MAX_LAG_FRACTION` | 0.25 | Fraction of trajectory for VACF lags. 0.25 gives good statistics. |
| `WINDOW_FUNCTION` | "hann" | FFT window to reduce spectral leakage. "hann" is a good default. "blackman" has less leakage but broader peaks. "none" for no windowing. |
| `ZERO_PAD_FACTOR` | 4 | Zero-padding multiplier for FFT. Higher = smoother VDOS (interpolated, not more resolution). |
| `FREQ_UNIT` | "THz" | Output frequency unit. "THz", "cm-1" (1 THz = 33.36 cm^-1), or "meV" (1 THz = 4.136 meV). |

### Trajectory requirements for VDOS

- **Frame interval**: Ideally save every MD step (`TRAJ_INTERVAL=1`). The Nyquist frequency is f_Nyq = 1/(2*dt), so:
  - dt = 1 fs: f_Nyq = 500 THz (~16,700 cm^-1) -- captures everything
  - dt = 10 fs: f_Nyq = 50 THz (~1,670 cm^-1) -- misses C-H, O-H stretches
  - dt = 100 fs: f_Nyq = 5 THz (~167 cm^-1) -- only low-frequency modes
- **Total trajectory**: Frequency resolution is df = 1/T_total. For 10 ps: df = 0.1 THz. For 1 ps: df = 1 THz (too coarse).
- **Minimum recommended**: 10+ ps with every-step output for reasonable VDOS.

## Interpreting Results

### VACF shape
- **VACF decays to zero with oscillations**: Normal solid behavior. Oscillation period ~ 1/(vibrational frequency).
- **VACF decays to zero monotonically**: Liquid or overdamped system.
- **VACF stays positive (no zero crossing)**: Very short trajectory or purely diffusive motion.
- **First zero-crossing time**: Related to the Einstein frequency. For solids, typically 0.05-0.5 ps.

### VDOS interpretation
- **Peaks**: Correspond to van Hove singularities (flat phonon bands). Peak positions should match harmonic phonon DOS peak positions.
- **Peak broadening (vs harmonic)**: Due to anharmonicity and finite temperature. Larger broadening = stronger anharmonicity.
- **Peak shifts (vs harmonic)**: Frequency softening (red shift) at high T is common. Hardening (blue shift) is rarer.
- **High-frequency cutoff**: Determined by the lightest atoms and stiffest bonds.
- **Low-frequency region**: Debye-like (proportional to omega^2 in 3D). Deviation indicates disorder or low-dimensionality.

### Element-projected VDOS
- **Heavy atoms**: Dominate low-frequency region (acoustic modes).
- **Light atoms**: Dominate high-frequency region (optical modes).
- Useful for identifying which atoms contribute to specific peaks.
- In oxides: O VDOS typically extends to higher frequencies than cation VDOS.

### Green-Kubo diffusion coefficient
- Should agree with Einstein relation (MSD slope) within statistical error.
- Disagreement indicates: insufficient trajectory length, poor VACF convergence, or non-diffusive regime.
- Green-Kubo converges faster than Einstein for liquids but is noisier.

## Common Issues

| Issue | Cause | Fix |
|---|---|---|
| VDOS cuts off at low frequency | Trajectory too short | Run longer MD (> 10 ps); total time determines frequency resolution |
| VDOS missing high frequencies | Frame interval too large (Nyquist limit) | Save every MD step (TRAJ_INTERVAL=1); check dt vs max vibrational frequency |
| VDOS has negative values | Spectral leakage from FFT | Use Hann or Blackman window; increase zero-padding |
| VDOS peaks very broad | High temperature or strong anharmonicity | Expected physical behavior; compare with harmonic DOS at same chemistry |
| No velocities in trajectory | ASE/LAMMPS not saving velocities | For ASE: velocities stored by default in .traj. For LAMMPS: add vx vy vz to dump columns. For XDATCAR: finite differences used (noisier). |
| Green-Kubo D disagrees with Einstein | Insufficient sampling or non-ergodic | Run longer trajectory; check both methods converge independently |
| Element VDOS does not sum to total | Different normalization | Re-normalize: weight each element VDOS by (n_element/n_total) before summing |
| Artifacts at zero frequency | DC offset in velocity signal | Subtract mean velocity (remove COM drift) before VACF computation |
| VACF very noisy | Too few atoms or too short trajectory | Increase system size; run longer trajectory; average over multiple independent runs |
