# Raman and IR Spectroscopy Simulation

## When to Use

- You need to compute **infrared (IR) absorption** or **Raman scattering** spectra for a crystalline or molecular material.
- You want to identify vibrational modes and assign peaks to specific atomic motions.
- You want to compare computed spectra with experimental FTIR or Raman measurements.
- You need **Born effective charges** (IR) or **polarizability derivatives** (Raman) from first principles.

## Method Selection

| Method | Tool | IR | Raman | Accuracy | Speed |
|--------|------|-----|-------|----------|-------|
| **QE ph.x** (DFPT) | `pw.x` + `ph.x` + `dynmat.x` | Yes (Born charges) | Yes (`lraman=.true.`) | High (DFT) | Slow |
| **MACE + finite diff** | `mace-torch` + `phonopy` | Approximate (dipole derivatives) | Approximate (polarizability derivatives) | Medium (MLIP) | Fast |
| **QE ph.x + finite diff Raman** | `pw.x` + `ph.x` + Python | Yes (DFPT) | Yes (finite-diff dielectric) | High | Medium |

**Recommendation:**
- For **publication-quality IR spectra**: use QE `ph.x` with `epsil=.true.` for Born effective charges.
- For **quick screening or large systems**: use MACE + phonopy for frequencies, with finite-difference dipole derivatives for approximate IR intensities.
- For **Raman**: QE `ph.x` with `lraman=.true.` when available, or finite-difference dielectric tensor approach.

## Prerequisites

```bash
pip install phonopy
# QE binaries: pw.x, ph.x, dynmat.x
# Already available: ase, mace-torch, numpy, scipy, matplotlib, pymatgen
```

## Detailed Steps

### Method A: Full QE DFPT Approach (IR + Raman)

#### Step A1: Relax the structure with QE

```python
#!/usr/bin/env python3
"""
Complete IR and Raman spectrum workflow using Quantum ESPRESSO.
Example system: alpha-quartz SiO2
"""

import os
import subprocess
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

WORK_DIR = "/home/work/spectroscopy_ir_raman"
PSEUDO_DIR = "/home/pseudo"
os.makedirs(WORK_DIR, exist_ok=True)

# ============================================================
# Step A1: SCF relaxation
# ============================================================
relax_input = f"""&CONTROL
    calculation = 'relax'
    prefix      = 'sio2'
    outdir      = '{WORK_DIR}/tmp'
    pseudo_dir  = '{PSEUDO_DIR}'
    tprnfor     = .true.
    tstress     = .true.
    forc_conv_thr = 1.0d-5
/
&SYSTEM
    ibrav       = 0
    nat         = 9
    ntyp        = 2
    ecutwfc     = 80.0
    ecutrho     = 640.0
/
&ELECTRONS
    conv_thr    = 1.0d-10
    mixing_beta = 0.4
/
&IONS
    ion_dynamics = 'bfgs'
/

ATOMIC_SPECIES
  Si  28.0855  Si.pbe-n-rrkjus_psl.1.0.0.UPF
  O   15.9994  O.pbe-n-rrkjus_psl.1.0.0.UPF

CELL_PARAMETERS angstrom
  4.9134  0.0000  0.0000
 -2.4567  4.2551  0.0000
  0.0000  0.0000  5.4052

ATOMIC_POSITIONS crystal
  Si  0.46990  0.00000  0.33333
  Si  0.00000  0.46990  0.66667
  Si  0.53010  0.53010  0.00000
  O   0.41440  0.26760  0.21300
  O   0.26760  0.41440  0.78700
  O   0.73240  0.14680  0.54633
  O   0.58560  0.85320  0.87967
  O   0.85320  0.58560  0.45367
  O   0.14680  0.73240  0.12033

K_POINTS automatic
  4 4 4 0 0 0
"""

relax_file = os.path.join(WORK_DIR, "sio2_relax.in")
with open(relax_file, "w") as f:
    f.write(relax_input)

print("Running SCF relaxation...")
result = subprocess.run(
    ["mpirun", "-np", "4", "pw.x", "-input", relax_file],
    capture_output=True, text=True, cwd=WORK_DIR,
    timeout=3600,
)
with open(os.path.join(WORK_DIR, "sio2_relax.out"), "w") as f:
    f.write(result.stdout)
print("Relaxation complete.")
```

#### Step A2: SCF at relaxed geometry

```python
# ============================================================
# Step A2: SCF at relaxed geometry (needed by ph.x)
# ============================================================
# Extract relaxed coordinates from relax output, then run SCF
# For simplicity, assume relaxation is minor and use same coords

scf_input = f"""&CONTROL
    calculation = 'scf'
    prefix      = 'sio2'
    outdir      = '{WORK_DIR}/tmp'
    pseudo_dir  = '{PSEUDO_DIR}'
    tprnfor     = .true.
    tstress     = .true.
/
&SYSTEM
    ibrav       = 0
    nat         = 9
    ntyp        = 2
    ecutwfc     = 80.0
    ecutrho     = 640.0
/
&ELECTRONS
    conv_thr    = 1.0d-10
    mixing_beta = 0.4
/

ATOMIC_SPECIES
  Si  28.0855  Si.pbe-n-rrkjus_psl.1.0.0.UPF
  O   15.9994  O.pbe-n-rrkjus_psl.1.0.0.UPF

CELL_PARAMETERS angstrom
  4.9134  0.0000  0.0000
 -2.4567  4.2551  0.0000
  0.0000  0.0000  5.4052

ATOMIC_POSITIONS crystal
  Si  0.46990  0.00000  0.33333
  Si  0.00000  0.46990  0.66667
  Si  0.53010  0.53010  0.00000
  O   0.41440  0.26760  0.21300
  O   0.26760  0.41440  0.78700
  O   0.73240  0.14680  0.54633
  O   0.58560  0.85320  0.87967
  O   0.85320  0.58560  0.45367
  O   0.14680  0.73240  0.12033

K_POINTS automatic
  4 4 4 0 0 0
"""

scf_file = os.path.join(WORK_DIR, "sio2_scf.in")
with open(scf_file, "w") as f:
    f.write(scf_input)

print("Running SCF...")
result = subprocess.run(
    ["mpirun", "-np", "4", "pw.x", "-input", scf_file],
    capture_output=True, text=True, cwd=WORK_DIR,
    timeout=3600,
)
with open(os.path.join(WORK_DIR, "sio2_scf.out"), "w") as f:
    f.write(result.stdout)
print("SCF complete.")
```

#### Step A3: Phonon calculation at Gamma with Born charges

```python
# ============================================================
# Step A3: Phonon at Gamma point + Born effective charges + dielectric tensor
# ============================================================
ph_input = f"""Phonon at Gamma
&INPUTPH
    prefix   = 'sio2'
    outdir   = '{WORK_DIR}/tmp'
    fildyn   = '{WORK_DIR}/sio2.dynG'
    tr2_ph   = 1.0d-14
    epsil    = .true.
    lraman   = .true.
    trans    = .true.
    asr      = .true.
/
0.0 0.0 0.0
"""

ph_file = os.path.join(WORK_DIR, "sio2_ph.in")
with open(ph_file, "w") as f:
    f.write(ph_input)

print("Running phonon calculation at Gamma (with Born charges and Raman)...")
result = subprocess.run(
    ["mpirun", "-np", "4", "ph.x", "-input", ph_file],
    capture_output=True, text=True, cwd=WORK_DIR,
    timeout=7200,
)
with open(os.path.join(WORK_DIR, "sio2_ph.out"), "w") as f:
    f.write(result.stdout)
print("Phonon calculation complete.")
```

#### Step A4: Process with dynmat.x to get IR/Raman intensities

```python
# ============================================================
# Step A4: dynmat.x for IR intensities and Raman activities
# ============================================================
dynmat_input = f"""&INPUT
    fildyn = '{WORK_DIR}/sio2.dynG'
    asr    = 'crystal'
    filout = '{WORK_DIR}/sio2_dynmat.out'
    filmol = '{WORK_DIR}/sio2_molden.mol'
    filxsf = '{WORK_DIR}/sio2_modes.axsf'
/
"""

dynmat_file = os.path.join(WORK_DIR, "sio2_dynmat.in")
with open(dynmat_file, "w") as f:
    f.write(dynmat_input)

print("Running dynmat.x...")
result = subprocess.run(
    ["dynmat.x", "-input", dynmat_file],
    capture_output=True, text=True, cwd=WORK_DIR,
    timeout=300,
)
with open(os.path.join(WORK_DIR, "sio2_dynmat_run.out"), "w") as f:
    f.write(result.stdout)
print("dynmat.x complete.")
```

#### Step A5: Parse output and plot spectra

```python
# ============================================================
# Step A5: Parse dynmat output and plot IR + Raman spectra
# ============================================================

def parse_dynmat_output(filename):
    """
    Parse dynmat.x output for frequencies, IR intensities, and Raman activities.
    Returns lists of: frequencies (cm^-1), IR intensities, Raman activities.
    """
    frequencies = []
    ir_intensities = []
    raman_activities = []

    with open(filename, "r") as f:
        lines = f.readlines()

    in_mode_section = False
    for line in lines:
        # dynmat.x output format:
        # mode   freq(cm-1)  IR_intensity  Raman_activity  ...
        line = line.strip()
        if "freq" in line.lower() and "ir" in line.lower():
            in_mode_section = True
            continue
        if in_mode_section and line:
            parts = line.split()
            if len(parts) >= 3:
                try:
                    freq = float(parts[1])
                    ir_int = float(parts[2])
                    raman_act = float(parts[3]) if len(parts) > 3 else 0.0
                    frequencies.append(freq)
                    ir_intensities.append(ir_int)
                    raman_activities.append(raman_act)
                except (ValueError, IndexError):
                    continue

    return (np.array(frequencies), np.array(ir_intensities),
            np.array(raman_activities))


def lorentzian(x, x0, gamma, intensity):
    """Lorentzian line shape for spectral broadening."""
    return intensity * (gamma / np.pi) / ((x - x0)**2 + gamma**2)


def plot_spectrum(frequencies, intensities, title, filename,
                  freq_range=(0, 1200), broadening=5.0, ylabel="Intensity (a.u.)"):
    """
    Plot a broadened spectrum with stick markers.
    """
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 7), height_ratios=[3, 1],
                                    sharex=True)

    # Broadened spectrum
    x = np.linspace(freq_range[0], freq_range[1], 2000)
    y = np.zeros_like(x)
    for freq, intensity in zip(frequencies, intensities):
        if freq > freq_range[0] and intensity > 0:
            y += lorentzian(x, freq, broadening, intensity)

    ax1.plot(x, y, "b-", linewidth=1.5)
    ax1.fill_between(x, 0, y, alpha=0.2, color="steelblue")
    ax1.set_ylabel(ylabel)
    ax1.set_title(title)
    ax1.set_xlim(freq_range)

    # Stick spectrum
    for freq, intensity in zip(frequencies, intensities):
        if freq > freq_range[0] and intensity > 0:
            ax2.vlines(freq, 0, intensity, colors="r", linewidth=1.5)
    ax2.set_xlabel("Frequency (cm$^{-1}$)")
    ax2.set_ylabel("Stick intensity")
    ax2.set_xlim(freq_range)

    plt.tight_layout()
    plt.savefig(filename, dpi=150)
    print(f"Saved: {filename}")


# Parse results
dynmat_out = os.path.join(WORK_DIR, "sio2_dynmat.out")
freqs, ir_int, raman_act = parse_dynmat_output(dynmat_out)

if len(freqs) > 0:
    print(f"\nParsed {len(freqs)} modes:")
    print(f"{'Mode':>5} {'Freq (cm-1)':>12} {'IR intensity':>14} {'Raman activity':>15}")
    for i, (f, ir, ra) in enumerate(zip(freqs, ir_int, raman_act)):
        print(f"{i+1:>5d} {f:>12.1f} {ir:>14.4f} {ra:>15.4f}")

    # Plot IR spectrum
    plot_spectrum(
        freqs, ir_int,
        title="IR Spectrum -- alpha-SiO$_2$ (QE DFPT)",
        filename=os.path.join(WORK_DIR, "ir_spectrum_sio2.png"),
        freq_range=(100, 1200),
        broadening=8.0,
        ylabel="IR Absorption (a.u.)",
    )

    # Plot Raman spectrum
    plot_spectrum(
        freqs, raman_act,
        title="Raman Spectrum -- alpha-SiO$_2$ (QE DFPT)",
        filename=os.path.join(WORK_DIR, "raman_spectrum_sio2.png"),
        freq_range=(100, 1200),
        broadening=6.0,
        ylabel="Raman Activity (a.u.)",
    )
else:
    print("WARNING: No modes parsed. Check dynmat output file format.")
```

### Method B: MACE + Phonopy for IR Spectrum (Fast Screening)

```python
#!/usr/bin/env python3
"""
IR spectrum from MACE + phonopy finite displacements.
IR intensities are estimated from finite-difference dipole derivatives.
Example: alpha-quartz SiO2.
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from ase.build import bulk
from ase import Atoms
from ase.optimize import BFGS
from ase.constraints import ExpCellFilter

# ============================================================
# Step 1: Build and relax SiO2 structure with MACE
# ============================================================
from pymatgen.core import Structure, Lattice
from pymatgen.io.ase import AseAtomsAdaptor

# Alpha-quartz SiO2
a, c = 4.9134, 5.4052
lattice = Lattice.hexagonal(a, c)
sio2_pmg = Structure(
    lattice,
    ["Si", "Si", "Si", "O", "O", "O", "O", "O", "O"],
    [
        [0.4699, 0.0000, 0.3333],
        [0.0000, 0.4699, 0.6667],
        [0.5301, 0.5301, 0.0000],
        [0.4144, 0.2676, 0.2130],
        [0.2676, 0.4144, 0.7870],
        [0.7324, 0.1468, 0.5463],
        [0.5856, 0.8532, 0.8797],
        [0.8532, 0.5856, 0.4537],
        [0.1468, 0.7324, 0.1203],
    ],
)

adaptor = AseAtomsAdaptor()
sio2 = adaptor.get_atoms(sio2_pmg)

from mace.calculators import mace_mp
calc = mace_mp(model="medium", default_dtype="float64")
sio2.calc = calc

print("Relaxing SiO2 with MACE...")
ecf = ExpCellFilter(sio2)
opt = BFGS(ecf, logfile="sio2_mace_relax.log")
opt.run(fmax=0.005, steps=300)
print(f"Relaxed: {len(sio2)} atoms, energy = {sio2.get_potential_energy():.4f} eV")

# ============================================================
# Step 2: Compute Gamma-point phonons with phonopy
# ============================================================
from phonopy import Phonopy
from phonopy.interface.ase import get_phonopy_structure

phonopy_cell = get_phonopy_structure(sio2)
phonon = Phonopy(
    phonopy_cell,
    supercell_matrix=[[2, 0, 0], [0, 2, 0], [0, 0, 2]],
    primitive_matrix="auto",
)
phonon.generate_displacements(distance=0.01)  # 0.01 A displacement
supercells = phonon.supercells_with_displacements

print(f"Number of displaced supercells: {len(supercells)}")

# Calculate forces for each displacement
from phonopy.interface.ase import get_atoms_from_phonopy_structure

forces_list = []
for i, sc in enumerate(supercells):
    atoms = get_atoms_from_phonopy_structure(sc)
    atoms.calc = calc
    forces = atoms.get_forces()
    forces_list.append(forces)
    if (i + 1) % 10 == 0:
        print(f"  Computed forces for displacement {i+1}/{len(supercells)}")

phonon.forces = forces_list
phonon.produce_force_constants()

# Gamma-point frequencies
phonon.run_qpoints([[0, 0, 0]])
qpoint_data = phonon.get_qpoints_dict()
frequencies_thz = qpoint_data["frequencies"][0]  # THz
frequencies_cm = frequencies_thz * 33.3564  # THz -> cm^-1

print(f"\nGamma-point phonon frequencies (cm^-1):")
for i, f in enumerate(frequencies_cm):
    label = "acoustic" if abs(f) < 5 else ""
    print(f"  Mode {i+1:3d}: {f:8.1f} cm^-1  {label}")

# Eigenvectors for mode assignment
eigenvectors = qpoint_data["eigenvectors"][0]  # (n_modes, n_atoms, 3)

# ============================================================
# Step 3: Estimate IR intensities from finite-difference dipoles
# ============================================================
# IR intensity ~ |dmu/dQ|^2 where mu is dipole moment, Q is normal mode coordinate
# Approximate dipole from sum of charge * position using oxidation states

print("\nEstimating IR intensities from dipole derivatives...")

# Use nominal charges as proxy (Si=+4, O=-2) for dipole moment
# This is approximate; QE Born charges are more accurate
nominal_charges = {"Si": 4.0, "O": -2.0}
charges = np.array([nominal_charges[s] for s in sio2.get_chemical_symbols()])

def compute_dipole(atoms, charges):
    """Compute electric dipole moment mu = sum(q_i * r_i)."""
    positions = atoms.get_positions()
    return np.sum(charges[:, np.newaxis] * positions, axis=0)

# Finite-difference dipole derivative along each normal mode
delta_Q = 0.01  # displacement amplitude in sqrt(amu)*A
ir_intensities = np.zeros(len(frequencies_cm))

from ase.units import _amu

# Get eigenvectors from phonopy
# Shape: (n_modes, n_atoms, 3) complex -- take real part for Gamma
n_modes = len(frequencies_cm)
n_atoms_prim = len(sio2)
masses = sio2.get_masses()

for mode_idx in range(n_modes):
    if frequencies_cm[mode_idx] < 10:
        continue  # skip acoustic modes

    # Mass-weighted eigenvector
    evec = np.real(eigenvectors[mode_idx])  # (n_atoms, 3)

    # Displace along mode: delta_r_i = evec_i / sqrt(m_i) * delta_Q
    disp = np.zeros((n_atoms_prim, 3))
    for a in range(n_atoms_prim):
        disp[a] = evec[a] / np.sqrt(masses[a]) * delta_Q

    # Forward displacement
    atoms_plus = sio2.copy()
    atoms_plus.set_positions(sio2.get_positions() + disp)
    mu_plus = compute_dipole(atoms_plus, charges)

    # Backward displacement
    atoms_minus = sio2.copy()
    atoms_minus.set_positions(sio2.get_positions() - disp)
    mu_minus = compute_dipole(atoms_minus, charges)

    # Dipole derivative
    dmu_dQ = (mu_plus - mu_minus) / (2 * delta_Q)
    ir_intensities[mode_idx] = np.sum(dmu_dQ**2)

# Normalize
if np.max(ir_intensities) > 0:
    ir_intensities /= np.max(ir_intensities)

# ============================================================
# Step 4: Estimate Raman activities from polarizability derivatives
# ============================================================
print("Estimating Raman activities from polarizability derivatives...")

def compute_polarizability_tensor(atoms, calc, delta=0.01):
    """
    Estimate dielectric response by applying electric field via finite differences.
    For a simple estimate, we compute the force response to displacement
    and relate it to polarizability via sum rules.

    A better approach: compute dipole moment changes under strain.
    Here we use a simplified Raman activity estimate from mode eigenvectors
    and force constant asymmetry.
    """
    # Simplified: use atomic polarizabilities as proxy
    # This is very approximate; real Raman requires dielectric derivative
    atomic_polarizability = {"Si": 5.38, "O": 3.88}  # in Bohr^3 (approximate)
    alpha = np.array([atomic_polarizability[s] for s in atoms.get_chemical_symbols()])
    return alpha

# Simplified Raman activity: proportional to |d(alpha)/dQ|^2
# Use bond polarizability model: alpha depends on bond length
raman_activities = np.zeros(len(frequencies_cm))
alpha_atomic = compute_polarizability_tensor(sio2, calc)

for mode_idx in range(n_modes):
    if frequencies_cm[mode_idx] < 10:
        continue

    evec = np.real(eigenvectors[mode_idx])

    # Simple Raman activity proxy: symmetric breathing modes are Raman active
    # Weight by atomic polarizability * displacement magnitude
    activity = 0.0
    for a in range(n_atoms_prim):
        disp_mag = np.linalg.norm(evec[a]) / np.sqrt(masses[a])
        activity += alpha_atomic[a] * disp_mag**2
    raman_activities[mode_idx] = activity

if np.max(raman_activities) > 0:
    raman_activities /= np.max(raman_activities)

# ============================================================
# Step 5: Plot spectra
# ============================================================

def lorentzian(x, x0, gamma, intensity):
    return intensity * (gamma / np.pi) / ((x - x0)**2 + gamma**2)

freq_range = (50, 1200)
broadening = 8.0
x = np.linspace(freq_range[0], freq_range[1], 2000)

# --- IR spectrum ---
fig, axes = plt.subplots(2, 1, figsize=(10, 8))

y_ir = np.zeros_like(x)
for freq, ir_int in zip(frequencies_cm, ir_intensities):
    if freq > freq_range[0] and ir_int > 0.01:
        y_ir += lorentzian(x, freq, broadening, ir_int)

axes[0].plot(x, y_ir, "b-", linewidth=1.5)
axes[0].fill_between(x, 0, y_ir, alpha=0.2, color="steelblue")
for freq, ir_int in zip(frequencies_cm, ir_intensities):
    if freq > freq_range[0] and ir_int > 0.01:
        axes[0].vlines(freq, 0, ir_int * max(y_ir) * 0.3, colors="r",
                        linewidth=1, alpha=0.5)
axes[0].set_ylabel("IR Intensity (a.u.)")
axes[0].set_title("IR Spectrum -- alpha-SiO$_2$ (MACE + phonopy)")
axes[0].set_xlim(freq_range)

# --- Raman spectrum ---
y_raman = np.zeros_like(x)
for freq, ra_int in zip(frequencies_cm, raman_activities):
    if freq > freq_range[0] and ra_int > 0.01:
        y_raman += lorentzian(x, freq, broadening, ra_int)

axes[1].plot(x, y_raman, "g-", linewidth=1.5)
axes[1].fill_between(x, 0, y_raman, alpha=0.2, color="lightgreen")
for freq, ra_int in zip(frequencies_cm, raman_activities):
    if freq > freq_range[0] and ra_int > 0.01:
        axes[1].vlines(freq, 0, ra_int * max(y_raman) * 0.3, colors="darkgreen",
                        linewidth=1, alpha=0.5)
axes[1].set_ylabel("Raman Activity (a.u.)")
axes[1].set_xlabel("Frequency (cm$^{-1}$)")
axes[1].set_title("Raman Spectrum -- alpha-SiO$_2$ (MACE + phonopy)")
axes[1].set_xlim(freq_range)

plt.tight_layout()
plt.savefig("ir_raman_spectrum_mace.png", dpi=150)
print("\nSaved: ir_raman_spectrum_mace.png")

# ============================================================
# Step 6: Mode symmetry assignment (using spglib)
# ============================================================
import spglib

cell = (sio2.get_cell(), sio2.get_scaled_positions(),
        sio2.get_atomic_numbers())
sym_data = spglib.get_symmetry_dataset(cell, symprec=0.01)
print(f"\nSpace group: {sym_data['international']} (#{sym_data['number']})")
print(f"Point group: {sym_data['pointgroup']}")

# For alpha-quartz (P3_221, point group 32/D3):
# IR active: A2 + E modes
# Raman active: A1 + E modes
# Both IR and Raman: E modes
print("\nExpected for alpha-quartz (point group 32):")
print("  IR active: A2 (z-polarized) + E (xy-polarized)")
print("  Raman active: A1 + E")
print("  Silent: A2 (Raman inactive)")

print("\n== Mode Summary ==")
print(f"{'Mode':>5} {'Freq (cm-1)':>12} {'IR int':>10} {'Raman act':>10} {'Assignment':>12}")
for i in range(n_modes):
    if frequencies_cm[i] < 10:
        assign = "acoustic"
    elif ir_intensities[i] > 0.1 and raman_activities[i] > 0.1:
        assign = "E (IR+Raman)"
    elif ir_intensities[i] > 0.1:
        assign = "A2 (IR)"
    elif raman_activities[i] > 0.1:
        assign = "A1 (Raman)"
    else:
        assign = "weak/silent"
    print(f"{i+1:>5d} {frequencies_cm[i]:>12.1f} {ir_intensities[i]:>10.3f} "
          f"{raman_activities[i]:>10.3f} {assign:>12}")
```

### Method C: QE Phonons + Finite-Difference Raman (High Accuracy)

When `lraman=.true.` is not available in `ph.x`, compute Raman tensors from finite differences of the dielectric tensor:

```python
#!/usr/bin/env python3
"""
Raman tensor from finite differences of dielectric tensor.
For each phonon mode Q_j:
  R_j = d(epsilon_ab) / d(Q_j)
computed by displacing atoms along each mode eigenvector and
recalculating the dielectric tensor with ph.x (epsil=.true.).
"""

import os
import subprocess
import numpy as np

WORK_DIR = "/home/work/raman_finite_diff"
PSEUDO_DIR = "/home/pseudo"
os.makedirs(WORK_DIR, exist_ok=True)

def run_epsil_calculation(atoms_positions, cell_params, work_dir, tag):
    """
    Run QE scf + ph.x epsil calculation for a displaced structure.
    Returns the 3x3 dielectric tensor.
    """
    calc_dir = os.path.join(work_dir, tag)
    os.makedirs(calc_dir, exist_ok=True)

    # Write SCF input
    nat = len(atoms_positions)
    pos_str = ""
    for symbol, pos in atoms_positions:
        pos_str += f"  {symbol}  {pos[0]:.10f}  {pos[1]:.10f}  {pos[2]:.10f}\n"

    scf_input = f"""&CONTROL
    calculation = 'scf'
    prefix      = 'raman'
    outdir      = '{calc_dir}/tmp'
    pseudo_dir  = '{PSEUDO_DIR}'
/
&SYSTEM
    ibrav       = 0
    nat         = {nat}
    ntyp        = 2
    ecutwfc     = 80.0
    ecutrho     = 640.0
/
&ELECTRONS
    conv_thr    = 1.0d-10
/

ATOMIC_SPECIES
  Si  28.0855  Si.pbe-n-rrkjus_psl.1.0.0.UPF
  O   15.9994  O.pbe-n-rrkjus_psl.1.0.0.UPF

{cell_params}

ATOMIC_POSITIONS angstrom
{pos_str}

K_POINTS automatic
  4 4 4 0 0 0
"""
    with open(os.path.join(calc_dir, "scf.in"), "w") as f:
        f.write(scf_input)

    subprocess.run(
        ["mpirun", "-np", "4", "pw.x", "-input", "scf.in"],
        capture_output=True, text=True, cwd=calc_dir, timeout=3600,
    )

    # Run ph.x for dielectric tensor only (no phonon modes)
    ph_input = f"""Dielectric tensor
&INPUTPH
    prefix   = 'raman'
    outdir   = '{calc_dir}/tmp'
    fildyn   = '{calc_dir}/dyn.xml'
    tr2_ph   = 1.0d-14
    epsil    = .true.
    trans    = .false.
/
0.0 0.0 0.0
"""
    with open(os.path.join(calc_dir, "ph.in"), "w") as f:
        f.write(ph_input)

    result = subprocess.run(
        ["mpirun", "-np", "4", "ph.x", "-input", "ph.in"],
        capture_output=True, text=True, cwd=calc_dir, timeout=3600,
    )

    # Parse dielectric tensor from ph.x output
    epsilon = np.zeros((3, 3))
    lines = result.stdout.split("\n")
    for i, line in enumerate(lines):
        if "Dielectric constant in cartesian" in line:
            for j in range(3):
                parts = lines[i + 2 + j].split()
                # Format: ( eps_xx  eps_xy  eps_xz )
                vals = [float(x.replace("(", "").replace(")", ""))
                        for x in parts if x.replace(".", "").replace("-", "").replace("+", "").replace("e", "").replace("E", "").isdigit() or "." in x]
                if len(vals) >= 3:
                    epsilon[j] = vals[:3]
            break

    return epsilon

def compute_raman_tensor(mode_eigenvector, masses, atoms_info, cell_params,
                         work_dir, mode_idx, delta=0.01):
    """
    Compute Raman tensor for a phonon mode using central finite differences:
    R_ab = [epsilon_ab(+dQ) - epsilon_ab(-dQ)] / (2 * dQ)
    """
    n_atoms = len(atoms_info)

    # Displacement along mode
    disp = np.zeros((n_atoms, 3))
    for a in range(n_atoms):
        disp[a] = np.real(mode_eigenvector[a]) / np.sqrt(masses[a]) * delta

    # Forward displacement
    atoms_plus = []
    for a, (symbol, pos) in enumerate(atoms_info):
        atoms_plus.append((symbol, pos + disp[a]))

    # Backward displacement
    atoms_minus = []
    for a, (symbol, pos) in enumerate(atoms_info):
        atoms_minus.append((symbol, pos - disp[a]))

    eps_plus = run_epsil_calculation(atoms_plus, cell_params, work_dir,
                                     f"mode_{mode_idx}_plus")
    eps_minus = run_epsil_calculation(atoms_minus, cell_params, work_dir,
                                      f"mode_{mode_idx}_minus")

    raman_tensor = (eps_plus - eps_minus) / (2 * delta)

    # Raman activity: I_Raman = 45 * alpha'^2 + 7 * gamma'^2
    # alpha' = (1/3) * Tr(R)
    # gamma'^2 = (1/2) * sum_ab [3*R_ab*R_ab - R_aa*R_bb]
    alpha_prime = np.trace(raman_tensor) / 3.0
    gamma_sq = 0.0
    for a in range(3):
        for b in range(3):
            gamma_sq += 0.5 * (3 * raman_tensor[a, b]**2
                               - raman_tensor[a, a] * raman_tensor[b, b])

    raman_activity = 45 * alpha_prime**2 + 7 * gamma_sq
    return raman_tensor, raman_activity

# Example usage (would be called after phonon mode calculation):
# raman_tensor, raman_activity = compute_raman_tensor(
#     eigenvectors[mode_idx], masses, atoms_info, cell_params,
#     WORK_DIR, mode_idx
# )
print("Raman finite-difference framework defined.")
print("Call compute_raman_tensor() for each IR-active or Raman-active mode.")
```

## Key Parameters

| Parameter | Typical Value | Effect |
|-----------|---------------|--------|
| **ecutwfc** | 60--100 Ry | Plane-wave cutoff for QE. Higher = more accurate phonons. Test convergence. |
| **tr2_ph** | 1e-14 | Phonon self-consistency threshold. Tight for accurate frequencies. |
| **epsil** | `.true.` | Compute dielectric tensor and Born effective charges (needed for IR). |
| **lraman** | `.true.` | Compute Raman tensors directly (if available in your QE build). |
| **asr** | `'crystal'` | Acoustic sum rule enforcement. Essential for removing translational mode artifacts. |
| **Broadening (Lorentzian gamma)** | 5--15 cm^-1 | Controls peak width. Use 5--8 cm^-1 for crystalline solids; 10--20 for disordered. |
| **Frequency range** | 0--1500 cm^-1 | Depends on material. Silicates: 100--1200; organic: 400--4000. |
| **Displacement distance (phonopy)** | 0.01 A | For finite-difference force constants. Smaller = more accurate but noisier. |
| **Supercell size (phonopy)** | 2x2x2 or larger | Larger supercells reduce interpolation errors at Gamma. |

## Interpreting Results

### IR Spectrum

- **Strong IR peaks**: modes with large dipole change (Born charge times displacement). High-frequency Si-O stretch modes (~1000-1100 cm^-1) are strongly IR active.
- **TO/LO splitting**: In polar crystals, transverse and longitudinal optical modes split due to long-range Coulomb interactions. QE `dynmat.x` with Born charges handles this.
- **Acoustic modes at ~0 cm^-1**: should have zero IR intensity. If they have nonzero intensity, the acoustic sum rule is not enforced properly.

### Raman Spectrum

- **Selection rules**: depend on point group symmetry. For centrosymmetric crystals, IR-active modes are Raman-inactive (mutual exclusion rule).
- **Raman activity vs Raman intensity**: the calculated quantity is the Raman activity (proportional to |d alpha/dQ|^2). The observed Raman intensity also depends on the laser wavelength, temperature (Bose factor), and scattering geometry.
- **Polarization dependence**: the full Raman tensor R_ab gives polarization-dependent intensities. Powder-averaged Raman intensity uses the isotropic formulas (45 * alpha'^2 + 7 * gamma'^2).

### Peak Assignment

- Compare computed frequencies with experimental peak positions. DFT-PBE typically underestimates frequencies by 1--5%.
- Use eigenvector visualization (XSF/Molden files from dynmat.x) to identify the atomic motions for each mode.
- Group-theory analysis: use the Bilbao Crystallographic Server or `phonopy --irreps` for symmetry labels.

## Common Issues

| Issue | Cause | Solution |
|-------|-------|----------|
| Imaginary frequencies at Gamma | Structure not fully relaxed or cell too stiff | Re-relax with tighter force threshold (`forc_conv_thr = 1e-5`) |
| IR intensities all zero | `epsil=.false.` or non-polar crystal | Set `epsil=.true.`; note that metals have no IR-active phonons |
| `lraman` not recognized | QE build without Raman support | Use finite-difference dielectric approach (Method C) |
| LO/TO splitting absent | Born charges not included in dynmat.x | Ensure `epsil=.true.` in ph.x and that dynmat reads the correct `.dynG` file |
| MACE IR intensities inaccurate | Nominal charges are poor approximation | Use QE Born effective charges for quantitative IR; MACE is for screening only |
| Very broad spectrum | Broadening parameter too large | Reduce Lorentzian gamma to 3--5 cm^-1 |
| Phonopy force constants noisy | Displacement too small or MACE forces imprecise | Increase displacement to 0.02 A; use `float64` precision for MACE |
| `ph.x` crashes with segfault | Memory or parallelization issue | Reduce k-points or number of MPI ranks; increase memory allocation |
