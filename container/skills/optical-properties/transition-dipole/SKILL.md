# Transition Dipole Moment Calculation

## When to Use

- You need optical transition dipole matrix elements between specific bands.
- You want to determine oscillator strengths for interband transitions.
- You need to evaluate optical selection rules for a material.
- You are analyzing polarization-dependent absorption (e.g., for polarized photoluminescence).
- You want to identify which transitions dominate the absorption spectrum.
- You need the WAVEDER file analysis from VASP (momentum matrix elements).
- You are computing radiative recombination rates or spontaneous emission lifetimes.

## Method Selection

| Criterion | QE (epsilon.x / projwfc.x) | VASP (WAVEDER) | VASPKIT (713--714) |
|---|---|---|---|
| Transition dipole moments | From epsilon.x matrix elements or projwfc.x projections | Direct from WAVEDER file | Post-processing of WAVEDER |
| Oscillator strengths | Derived from epsilon.x eps2 output | Computed from WAVEDER | Automated via VASPKIT 713 |
| Band-resolved transitions | Via energy-filtered analysis of eps2 | Full k-resolved matrix elements | VASPKIT 714 for band-pair analysis |
| Selection rules | Inferred from symmetry + eps2 components | Explicit from matrix element magnitudes | Automated symmetry analysis |

**MACE cannot compute transition dipole moments.** It has no electronic wavefunctions. Always use QE or VASP.

## Prerequisites

- A relaxed crystal structure.
- For QE: Norm-conserving pseudopotentials in `./pseudo/` (required for epsilon.x). Completed SCF and NSCF calculations.
- QE executables: `pw.x`, `epsilon.x`.
- For VASP (future): VASP compiled with WAVEDER support, INCAR with `LOPTICS = .TRUE.`.
- Python packages: `numpy`, `scipy`, `matplotlib`.

---

## Detailed Steps

### Background: Transition Dipole Moments

The transition dipole moment between valence band v and conduction band c at k-point k is:

```
p_cv(k) = <psi_ck| p |psi_vk>
```

where p is the momentum operator. The oscillator strength is:

```
f_cv(k) = (2 / m_e * omega_cv) * |p_cv(k)|^2
```

The imaginary part of the dielectric function is directly related to these matrix elements:

```
eps2(omega) = (4*pi^2*e^2) / (m_e^2 * omega^2 * V) *
              sum_{v,c,k} |p_cv(k)|^2 * delta(E_ck - E_vk - hbar*omega)
```

### Method A: QE -- Transition Dipole from epsilon.x

epsilon.x computes the full frequency-dependent dielectric function including the matrix elements internally. We extract effective transition dipole information by analyzing the energy-resolved contributions to eps2.

#### Step A1: SCF Calculation

```python
#!/usr/bin/env python3
"""
Step 1: SCF calculation for transition dipole analysis.
Example: GaAs (direct gap, strong dipole-allowed transitions).
"""
import os
import subprocess

PSEUDO_DIR = os.path.abspath("./pseudo")
OUTDIR = os.path.abspath("./tmp_dipole")
os.makedirs(OUTDIR, exist_ok=True)
os.makedirs(PSEUDO_DIR, exist_ok=True)
PREFIX = "gaas"

scf_input = f"""&CONTROL
    calculation  = 'scf'
    prefix       = '{PREFIX}'
    outdir       = '{OUTDIR}'
    pseudo_dir   = '{PSEUDO_DIR}'
/

&SYSTEM
    ibrav        = 2
    celldm(1)    = 10.6829
    nat          = 2
    ntyp         = 2
    ecutwfc      = 60.0
    ecutrho      = 240.0
    occupations  = 'fixed'
/

&ELECTRONS
    conv_thr = 1.0d-10
/

ATOMIC_SPECIES
  Ga  69.723   Ga_ONCV_PBE-1.2.upf
  As  74.922   As_ONCV_PBE-1.2.upf

ATOMIC_POSITIONS (crystal)
  Ga  0.00  0.00  0.00
  As  0.25  0.25  0.25

K_POINTS (automatic)
  16 16 16  0 0 0
"""

with open(f"{PREFIX}_scf.in", "w") as f:
    f.write(scf_input)

print("Running SCF...")
result = subprocess.run(
    ["mpirun", "-np", "4", "pw.x", "-in", f"{PREFIX}_scf.in"],
    capture_output=True, text=True, timeout=600
)
with open(f"{PREFIX}_scf.out", "w") as f:
    f.write(result.stdout)

assert "convergence has been achieved" in result.stdout, "SCF did not converge!"
print("SCF converged.")
```

#### Step A2: NSCF with Many Bands

```python
#!/usr/bin/env python3
"""
Step 2: NSCF calculation with many bands for transition dipole analysis.
Need enough conduction bands to cover the energy range of interest.
"""
import os
import subprocess

PSEUDO_DIR = os.path.abspath("./pseudo")
OUTDIR = os.path.abspath("./tmp_dipole")
PREFIX = "gaas"
NBND = 60  # GaAs: 4 VB, need many CB for transitions up to ~15 eV

nscf_input = f"""&CONTROL
    calculation  = 'nscf'
    prefix       = '{PREFIX}'
    outdir       = '{OUTDIR}'
    pseudo_dir   = '{PSEUDO_DIR}'
/

&SYSTEM
    ibrav        = 2
    celldm(1)    = 10.6829
    nat          = 2
    ntyp         = 2
    ecutwfc      = 60.0
    ecutrho      = 240.0
    occupations  = 'fixed'
    nbnd         = {NBND}
    nosym        = .true.
    noinv        = .true.
/

&ELECTRONS
    conv_thr = 1.0d-10
/

ATOMIC_SPECIES
  Ga  69.723   Ga_ONCV_PBE-1.2.upf
  As  74.922   As_ONCV_PBE-1.2.upf

ATOMIC_POSITIONS (crystal)
  Ga  0.00  0.00  0.00
  As  0.25  0.25  0.25

K_POINTS (automatic)
  16 16 16  0 0 0
"""

with open(f"{PREFIX}_nscf.in", "w") as f:
    f.write(nscf_input)

print(f"Running NSCF with nbnd={NBND}...")
result = subprocess.run(
    ["mpirun", "-np", "4", "pw.x", "-in", f"{PREFIX}_nscf.in"],
    capture_output=True, text=True, timeout=1800
)
with open(f"{PREFIX}_nscf.out", "w") as f:
    f.write(result.stdout)

assert result.returncode == 0, "NSCF failed!"
print("NSCF completed.")
```

#### Step A3: epsilon.x with Fine Energy Resolution

```python
#!/usr/bin/env python3
"""
Step 3: Run epsilon.x with fine energy resolution to resolve individual transitions.
Use small broadening to see individual transition contributions.
"""
import os
import subprocess

OUTDIR = os.path.abspath("./tmp_dipole")
PREFIX = "gaas"

epsilon_input = f"""&INPUTPP
    prefix       = '{PREFIX}'
    outdir       = '{OUTDIR}'
    calculation  = 'eps'
/

&ENERGY_GRID
    smeartype    = 'gauss'
    intersmear   = 0.05
    wmin         = 0.0
    wmax         = 15.0
    nw           = 3000
    shift        = 0.0
/
"""
# NOTE: intersmear = 0.05 eV gives finer resolution than default 0.1
# This helps resolve individual transition peaks

with open(f"{PREFIX}_epsilon.in", "w") as f:
    f.write(epsilon_input)

print("Running epsilon.x with fine resolution...")
result = subprocess.run(
    ["epsilon.x", "-in", f"{PREFIX}_epsilon.in"],
    capture_output=True, text=True, timeout=600
)
with open(f"{PREFIX}_epsilon.out", "w") as f:
    f.write(result.stdout)

assert result.returncode == 0, f"epsilon.x failed!\n{result.stderr}"
print("epsilon.x completed.")
```

#### Step A4: Extract Transition Dipole Information and Oscillator Strengths

```python
#!/usr/bin/env python3
"""
Step 4: Extract effective transition dipole moments and oscillator strengths
from epsilon.x output.

The imaginary part of the dielectric function eps2(omega) is directly
proportional to |p_cv|^2 summed over transitions at that energy.
We decompose eps2 into contributions and compute oscillator strengths.
"""
import numpy as np
import glob
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.signal import find_peaks

# ── Physical constants ───────────────────────────────────────────────
HBAR_EV_S = 6.582119569e-16   # hbar in eV*s
M_E_KG = 9.1093837015e-31     # electron mass in kg
E_CHARGE = 1.602176634e-19    # elementary charge in C
EPS0 = 8.8541878128e-12       # vacuum permittivity in F/m
BOHR_TO_M = 5.29177210903e-11 # Bohr radius in m
EV_TO_J = 1.602176634e-19

# ── Load dielectric function data ───────────────────────────────────
def load_eps_dat(filename):
    """Load epsilon.x output .dat file."""
    data = []
    with open(filename) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            try:
                vals = [float(x) for x in parts]
                if len(vals) >= 4:
                    data.append(vals[:4])
                elif len(vals) >= 2:
                    data.append([vals[0], vals[1], vals[1], vals[1]])
            except ValueError:
                continue
    return np.array(data)

# Find files
real_files = sorted(glob.glob("epsr*.dat"))
imag_files = sorted(glob.glob("epsi*.dat"))

if not real_files or not imag_files:
    raise FileNotFoundError("Cannot find epsr/epsi .dat files. Run epsilon.x first.")

dr = load_eps_dat(real_files[0])
di = load_eps_dat(imag_files[0])

energy = dr[:, 0]  # eV
eps1_xx, eps1_yy, eps1_zz = dr[:, 1], dr[:, 2], dr[:, 3]
eps2_xx, eps2_yy, eps2_zz = di[:, 1], di[:, 2], di[:, 3]

# Average for cubic systems
eps2_avg = (eps2_xx + eps2_yy + eps2_zz) / 3.0

print(f"Loaded {len(energy)} frequency points")
print(f"Energy range: {energy[0]:.3f} -- {energy[-1]:.3f} eV")

# ── Compute effective transition dipole squared ──────────────────────
# From eps2(omega) = (4*pi^2*e^2)/(m_e^2 * omega^2 * V * Omega_BZ) *
#                    sum_{v,c,k} |p_cv(k)|^2 * delta(E_ck - E_vk - hbar*omega)
#
# In practice, eps2 * omega^2 is proportional to |p_cv|^2 * JDOS.
# We define an effective squared transition dipole:
#   |d_eff(omega)|^2 ~ eps2(omega) * omega^2
# This removes the 1/omega^2 factor and isolates the matrix element contribution.

omega = energy.copy()
omega[omega < 1e-6] = 1e-6  # avoid division by zero

# Effective |p|^2 (proportional): remove the 1/omega^2 from eps2
p_squared_eff = eps2_avg * omega**2

# Oscillator strength density (proportional to f(omega)):
# f(omega) = (2/(m_e * omega)) * |p|^2 * JDOS(omega)
# Since JDOS ~ eps2 * omega^2 / |p|^2, and f ~ |p|^2 / omega:
# f_eff(omega) ~ eps2(omega) * omega
f_eff = eps2_avg * omega

# ── Identify prominent transitions ──────────────────────────────────
# Find peaks in eps2 -- these correspond to critical points (Van Hove singularities)
peaks, properties = find_peaks(
    eps2_avg,
    height=0.1 * np.max(eps2_avg),
    distance=30,  # minimum separation in data points
    prominence=0.05 * np.max(eps2_avg)
)

print(f"\n=== Transition Analysis ===")
print(f"{'Peak':>4s}  {'Energy (eV)':>12s}  {'eps2':>10s}  {'|p|^2_eff':>12s}  {'f_eff':>10s}")
print("-" * 60)
for i, p in enumerate(peaks):
    print(f"{i+1:4d}  {energy[p]:12.3f}  {eps2_avg[p]:10.4f}  "
          f"{p_squared_eff[p]:12.4f}  {f_eff[p]:10.4f}")

# ── Polarization analysis (selection rules) ──────────────────────────
# For non-cubic systems, different polarizations probe different transitions
print(f"\n=== Polarization-Resolved Transitions ===")
print(f"{'Energy (eV)':>12s}  {'eps2_xx':>10s}  {'eps2_yy':>10s}  {'eps2_zz':>10s}  {'Anisotropy':>12s}")
print("-" * 65)

peaks_xx, _ = find_peaks(eps2_xx, height=0.1 * np.max(eps2_xx), distance=30)
all_peak_indices = sorted(set(list(peaks) + list(peaks_xx)))

for p in all_peak_indices[:15]:  # show top 15
    aniso = np.std([eps2_xx[p], eps2_yy[p], eps2_zz[p]]) / (np.mean([eps2_xx[p], eps2_yy[p], eps2_zz[p]]) + 1e-10)
    print(f"{energy[p]:12.3f}  {eps2_xx[p]:10.4f}  {eps2_yy[p]:10.4f}  {eps2_zz[p]:10.4f}  {aniso:12.4f}")

# ── Selection rule assessment ────────────────────────────────────────
# For cubic zincblende (GaAs): all xx=yy=zz transitions are dipole-allowed
# For hexagonal/tetragonal: xx=yy != zz indicates polarization selection rules
# Anisotropy ratio > 0.1 indicates strong polarization dependence
print(f"\n=== Selection Rule Summary ===")
for i, p in enumerate(peaks[:5]):
    ratio_xy = eps2_xx[p] / (eps2_zz[p] + 1e-10)
    if abs(ratio_xy - 1.0) < 0.05:
        rule = "Isotropic (cubic allowed)"
    elif eps2_zz[p] < 0.01 * eps2_xx[p]:
        rule = "E perp c only (z-forbidden)"
    elif eps2_xx[p] < 0.01 * eps2_zz[p]:
        rule = "E || c only (xy-forbidden)"
    else:
        rule = f"Anisotropic (xx/zz = {ratio_xy:.2f})"
    print(f"  Peak {i+1} at {energy[p]:.2f} eV: {rule}")

# ── Compute integrated oscillator strength ───────────────────────────
# Sum rule: integral of eps2*omega over all frequencies gives the total
# oscillator strength, related to the effective number of electrons (N_eff)
# N_eff(E) = (2*m_e*V) / (pi*e^2*hbar^2) * integral_0^E eps2(w)*w dw

dE = energy[1] - energy[0] if len(energy) > 1 else 0.01
cumulative_f = np.cumsum(eps2_avg * omega * dE)
# Normalize: for the f-sum rule, integral should approach N_electrons
# The normalization constant depends on the cell volume and units
print(f"\n=== Cumulative Oscillator Strength (arbitrary units) ===")
for E_check in [2.0, 5.0, 8.0, 10.0, 15.0]:
    idx = np.argmin(np.abs(energy - E_check))
    if idx < len(cumulative_f):
        print(f"  Integrated f up to {E_check:.0f} eV: {cumulative_f[idx]:.4f}")

# ── Plotting ─────────────────────────────────────────────────────────

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Panel (a): eps2 with transition peaks marked
ax = axes[0, 0]
ax.plot(energy, eps2_avg, "r-", linewidth=1.2, label=r"$\varepsilon_2$ (average)")
ax.fill_between(energy, 0, eps2_avg, alpha=0.1, color="red")
for i, p in enumerate(peaks[:8]):
    ax.annotate(f"T{i+1}\n{energy[p]:.1f} eV",
                xy=(energy[p], eps2_avg[p]),
                xytext=(0, 15), textcoords="offset points",
                ha="center", fontsize=8,
                arrowprops=dict(arrowstyle="->", color="gray", lw=0.8))
ax.set_xlabel("Energy (eV)")
ax.set_ylabel(r"$\varepsilon_2(\omega)$")
ax.set_title("(a) Imaginary Dielectric Function with Transitions")
ax.set_xlim(0, 12)
ax.grid(alpha=0.3)
ax.legend(fontsize=10)

# Panel (b): Effective |p|^2 (transition dipole squared)
ax = axes[0, 1]
ax.plot(energy, p_squared_eff, "b-", linewidth=1.2)
ax.set_xlabel("Energy (eV)")
ax.set_ylabel(r"$|\mathbf{p}_{cv}|^2_{\mathrm{eff}}$ (arb. units)")
ax.set_title(r"(b) Effective $|p_{cv}|^2$ vs Energy")
ax.set_xlim(0, 12)
ax.grid(alpha=0.3)

# Panel (c): Oscillator strength density
ax = axes[1, 0]
ax.plot(energy, f_eff, "g-", linewidth=1.2, label=r"$f_{\mathrm{eff}}(\omega)$")
ax.plot(energy, cumulative_f / (cumulative_f[-1] + 1e-10) * np.max(f_eff),
        "k--", linewidth=1.0, label="Cumulative (normalized)")
ax.set_xlabel("Energy (eV)")
ax.set_ylabel("Oscillator Strength Density (arb. units)")
ax.set_title("(c) Oscillator Strength Density")
ax.set_xlim(0, 12)
ax.legend(fontsize=10)
ax.grid(alpha=0.3)

# Panel (d): Polarization-resolved eps2
ax = axes[1, 1]
ax.plot(energy, eps2_xx, linewidth=1.0, label=r"$\varepsilon_2^{xx}$")
ax.plot(energy, eps2_yy, linewidth=1.0, linestyle="--", label=r"$\varepsilon_2^{yy}$")
ax.plot(energy, eps2_zz, linewidth=1.0, linestyle="-.", label=r"$\varepsilon_2^{zz}$")
ax.set_xlabel("Energy (eV)")
ax.set_ylabel(r"$\varepsilon_2(\omega)$")
ax.set_title("(d) Polarization-Resolved Transitions")
ax.set_xlim(0, 12)
ax.legend(fontsize=10)
ax.grid(alpha=0.3)

plt.suptitle("Transition Dipole Analysis -- GaAs (PBE, RPA)", fontsize=15, y=1.01)
plt.tight_layout()
plt.savefig("transition_dipole_analysis.png", dpi=200, bbox_inches="tight")
print("\nSaved: transition_dipole_analysis.png")
```

### Method B: VASP -- WAVEDER File Parsing (Future)

When VASP is available, the WAVEDER file contains explicit momentum matrix elements between all band pairs at each k-point. This provides the most detailed transition dipole information.

#### Step B1: VASP INCAR for Optical Matrix Elements

```
# INCAR for optical transition dipole calculation
SYSTEM = GaAs transition dipole
PREC   = Accurate
ENCUT  = 500
EDIFF  = 1E-8
ISMEAR = 0
SIGMA  = 0.05

# Optical properties
LOPTICS = .TRUE.
CSHIFT  = 0.1
NEDOS   = 5000
NBANDS  = 120

# Write WAVEDER file with momentum matrix elements
LWAVE  = .TRUE.
LREAL  = .FALSE.
```

#### Step B2: Parse WAVEDER and Compute Transition Dipoles (VASPKIT 713--714)

```python
#!/usr/bin/env python3
"""
Parse VASP WAVEDER file to extract transition dipole matrix elements.
WAVEDER contains <psi_nk|nabla|psi_mk> for all band pairs (n,m) at each k-point.

Equivalent to VASPKIT options:
  713 - Transition Dipole Moment
  714 - Band-Decomposed Transition Dipole Moment

NOTE: Requires VASP output files (WAVEDER, EIGENVAL, OUTCAR).
This script will work when VASP becomes available in the container.
"""
import numpy as np
import os
import struct
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ── Physical constants ───────────────────────────────────────────────
HBAR = 1.054571817e-34       # J*s
M_E = 9.1093837015e-31       # kg
EV_TO_J = 1.602176634e-19
BOHR = 5.29177210903e-11     # m
HARTREE_TO_EV = 27.211386245988

def read_waveder(filename="WAVEDER"):
    """
    Read VASP WAVEDER file (binary format).

    WAVEDER contains the matrix elements of the nabla operator
    between all pairs of bands at each k-point:
        <psi_nk| d/dr_alpha |psi_mk>
    where alpha = x, y, z.

    Returns:
        nbands: number of bands
        nkpts: number of k-points
        cder: complex array of shape (nkpts, nbands, nbands, 3)
              containing the derivative matrix elements
    """
    if not os.path.exists(filename):
        raise FileNotFoundError(
            f"{filename} not found. Run VASP with LOPTICS=.TRUE. first."
        )

    with open(filename, "rb") as f:
        # Read header
        nbands, nbands2, nkpts = struct.unpack("iii", f.read(12))
        # Skip padding
        _ = struct.unpack("i", f.read(4))

        print(f"WAVEDER: nbands={nbands}, nkpts={nkpts}")

        # Read matrix elements
        # Format: for each k-point, for each direction (x,y,z),
        #         nbands x nbands complex matrix
        cder = np.zeros((nkpts, nbands, nbands, 3), dtype=complex)

        for ik in range(nkpts):
            for idir in range(3):
                for ib2 in range(nbands):
                    for ib1 in range(nbands):
                        re, im = struct.unpack("dd", f.read(16))
                        cder[ik, ib1, ib2, idir] = complex(re, im)

    return nbands, nkpts, cder


def read_eigenval(filename="EIGENVAL"):
    """
    Read VASP EIGENVAL file to get band energies.

    Returns:
        nkpts: number of k-points
        nbands: number of bands
        kpoints: array of k-point coordinates (nkpts, 3)
        eigenvalues: array of eigenvalues (nkpts, nbands) in eV
        occupations: array of occupations (nkpts, nbands)
    """
    if not os.path.exists(filename):
        raise FileNotFoundError(f"{filename} not found.")

    with open(filename) as f:
        lines = f.readlines()

    # Line 6 has: NELECT, NKPTS, NBANDS
    header = lines[5].split()
    nelect = int(header[0])
    nkpts = int(header[1])
    nbands = int(header[2])

    kpoints = np.zeros((nkpts, 3))
    eigenvalues = np.zeros((nkpts, nbands))
    occupations = np.zeros((nkpts, nbands))

    line_idx = 7  # data starts after header
    for ik in range(nkpts):
        # k-point line
        kdata = lines[line_idx].split()
        kpoints[ik] = [float(kdata[0]), float(kdata[1]), float(kdata[2])]
        line_idx += 1

        for ib in range(nbands):
            bdata = lines[line_idx].split()
            eigenvalues[ik, ib] = float(bdata[1])
            if len(bdata) > 2:
                occupations[ik, ib] = float(bdata[2])
            line_idx += 1
        line_idx += 1  # blank line between k-points

    return nkpts, nbands, kpoints, eigenvalues, occupations


def compute_transition_dipoles(cder, eigenvalues, occupations,
                               n_valence=None, energy_range=(0, 15)):
    """
    Compute transition dipole moments and oscillator strengths
    from WAVEDER matrix elements and eigenvalues.

    Parameters:
        cder: complex matrix elements (nkpts, nbands, nbands, 3)
        eigenvalues: band energies (nkpts, nbands) in eV
        occupations: band occupations (nkpts, nbands)
        n_valence: number of valence bands (auto-detected if None)
        energy_range: (Emin, Emax) for transitions in eV

    Returns:
        Dictionary with transition analysis results
    """
    nkpts, nbands = eigenvalues.shape

    # Auto-detect valence band count from occupations
    if n_valence is None:
        avg_occ = np.mean(occupations, axis=0)
        n_valence = np.sum(avg_occ > 0.5)
        print(f"Auto-detected {n_valence} valence bands")

    # Compute transition dipole moments for all VB->CB pairs
    transitions = []

    for ik in range(nkpts):
        for iv in range(n_valence):
            for ic in range(n_valence, nbands):
                dE = eigenvalues[ik, ic] - eigenvalues[ik, iv]

                if dE < energy_range[0] or dE > energy_range[1]:
                    continue
                if dE < 0.01:  # skip degenerate
                    continue

                # Transition dipole vector (momentum matrix element)
                p_cv = cder[ik, iv, ic, :]  # complex 3-vector

                # |p_cv|^2 = |p_x|^2 + |p_y|^2 + |p_z|^2
                p_squared = np.sum(np.abs(p_cv)**2)

                # Oscillator strength: f = 2|p_cv|^2 / (m_e * dE)
                # In atomic units: f = 2|p_cv|^2 / dE  (dE in Hartree, p in a.u.)
                # We store in eV and a.u. of momentum
                f_osc = 2.0 * p_squared / (dE / HARTREE_TO_EV) if dE > 0.01 else 0.0

                transitions.append({
                    "ik": ik,
                    "iv": iv + 1,  # 1-based band index
                    "ic": ic + 1,
                    "dE_eV": dE,
                    "px": p_cv[0],
                    "py": p_cv[1],
                    "pz": p_cv[2],
                    "p_squared": p_squared,
                    "f_osc": f_osc,
                })

    # Sort by oscillator strength
    transitions.sort(key=lambda t: t["f_osc"], reverse=True)

    return transitions


def analyze_and_plot(transitions, eigenvalues, n_valence,
                     energy_bins=500, energy_range=(0, 15)):
    """
    Analyze transitions and create plots.
    """
    nkpts = eigenvalues.shape[0]

    # ── Transition dipole moment vs energy histogram ────────────────
    energies_t = np.array([t["dE_eV"] for t in transitions])
    p_sq_t = np.array([t["p_squared"] for t in transitions])
    f_osc_t = np.array([t["f_osc"] for t in transitions])

    # Bin transitions by energy
    E_edges = np.linspace(energy_range[0], energy_range[1], energy_bins + 1)
    E_centers = 0.5 * (E_edges[:-1] + E_edges[1:])
    dE_bin = E_edges[1] - E_edges[0]

    p_sq_binned = np.zeros(energy_bins)
    f_binned = np.zeros(energy_bins)
    count_binned = np.zeros(energy_bins)

    for t in transitions:
        idx = int((t["dE_eV"] - energy_range[0]) / dE_bin)
        if 0 <= idx < energy_bins:
            p_sq_binned[idx] += t["p_squared"]
            f_binned[idx] += t["f_osc"]
            count_binned[idx] += 1

    # Normalize per k-point
    p_sq_binned /= nkpts
    f_binned /= nkpts

    # ── Print top transitions ───────────────────────────────────────
    print(f"\n=== Top 20 Transitions by Oscillator Strength ===")
    print(f"{'Rank':>4s}  {'k-pt':>5s}  {'VB':>3s}  {'CB':>3s}  "
          f"{'dE(eV)':>8s}  {'|p|^2':>10s}  {'f_osc':>10s}")
    print("-" * 55)
    for i, t in enumerate(transitions[:20]):
        print(f"{i+1:4d}  {t['ik']+1:5d}  {t['iv']:3d}  {t['ic']:3d}  "
              f"{t['dE_eV']:8.3f}  {t['p_squared']:10.4f}  {t['f_osc']:10.4f}")

    # ── Band-pair analysis ──────────────────────────────────────────
    # Aggregate by (VB, CB) pair across all k-points
    pair_data = {}
    for t in transitions:
        key = (t["iv"], t["ic"])
        if key not in pair_data:
            pair_data[key] = {"f_sum": 0, "p_sq_sum": 0, "count": 0,
                              "E_min": t["dE_eV"], "E_max": t["dE_eV"]}
        pair_data[key]["f_sum"] += t["f_osc"]
        pair_data[key]["p_sq_sum"] += t["p_squared"]
        pair_data[key]["count"] += 1
        pair_data[key]["E_min"] = min(pair_data[key]["E_min"], t["dE_eV"])
        pair_data[key]["E_max"] = max(pair_data[key]["E_max"], t["dE_eV"])

    # Sort by total f
    sorted_pairs = sorted(pair_data.items(), key=lambda x: x[1]["f_sum"], reverse=True)

    print(f"\n=== Top Band Pairs by Total Oscillator Strength ===")
    print(f"{'VB':>3s} -> {'CB':>3s}  {'f_total':>10s}  {'E_range (eV)':>15s}  {'N_kpts':>6s}")
    print("-" * 50)
    for (iv, ic), data in sorted_pairs[:10]:
        print(f"{iv:3d} -> {ic:3d}  {data['f_sum']/nkpts:10.4f}  "
              f"{data['E_min']:6.2f} -- {data['E_max']:6.2f}  {data['count']:6d}")

    # ── Plotting ────────────────────────────────────────────────────
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # (a) |p_cv|^2 vs energy
    ax = axes[0, 0]
    ax.bar(E_centers, p_sq_binned, width=dE_bin * 0.9, color="steelblue", alpha=0.7)
    ax.set_xlabel("Transition Energy (eV)")
    ax.set_ylabel(r"$\sum_{cv} |\mathbf{p}_{cv}|^2$ (arb. units / k-pt)")
    ax.set_title(r"(a) Transition Dipole $|p_{cv}|^2$ vs Energy")
    ax.set_xlim(*energy_range)
    ax.grid(alpha=0.3)

    # (b) Oscillator strength density
    ax = axes[0, 1]
    ax.bar(E_centers, f_binned, width=dE_bin * 0.9, color="darkred", alpha=0.7)
    ax.set_xlabel("Transition Energy (eV)")
    ax.set_ylabel(r"$\sum_{cv} f_{cv}$ (per k-pt)")
    ax.set_title("(b) Oscillator Strength Density")
    ax.set_xlim(*energy_range)
    ax.grid(alpha=0.3)

    # (c) Band gap region transitions scatter plot
    ax = axes[1, 0]
    # Show individual transitions near the gap
    near_gap = [t for t in transitions if t["dE_eV"] < 5.0]
    if near_gap:
        e_vals = [t["dE_eV"] for t in near_gap[:500]]
        f_vals = [t["f_osc"] for t in near_gap[:500]]
        ax.scatter(e_vals, f_vals, s=10, alpha=0.5, c="navy")
    ax.set_xlabel("Transition Energy (eV)")
    ax.set_ylabel("Oscillator Strength $f_{cv}$")
    ax.set_title("(c) Near-Gap Transitions (individual)")
    ax.set_xlim(0, 5)
    ax.grid(alpha=0.3)

    # (d) Cumulative oscillator strength (f-sum rule check)
    ax = axes[1, 1]
    cumul_f = np.cumsum(f_binned * dE_bin)
    ax.plot(E_centers, cumul_f, "k-", linewidth=1.5)
    ax.set_xlabel("Energy (eV)")
    ax.set_ylabel("Cumulative Oscillator Strength")
    ax.set_title("(d) Cumulative f (f-sum rule)")
    ax.set_xlim(*energy_range)
    ax.grid(alpha=0.3)

    plt.suptitle("Transition Dipole Moment Analysis", fontsize=15, y=1.01)
    plt.tight_layout()
    plt.savefig("transition_dipole_waveder.png", dpi=200, bbox_inches="tight")
    print("\nSaved: transition_dipole_waveder.png")


# ── Main execution ───────────────────────────────────────────────────
if __name__ == "__main__":
    try:
        nbands, nkpts, cder = read_waveder("WAVEDER")
        nkpts_e, nbands_e, kpoints, eigenvalues, occupations = read_eigenval("EIGENVAL")

        assert nkpts == nkpts_e, f"k-point mismatch: WAVEDER={nkpts}, EIGENVAL={nkpts_e}"
        assert nbands == nbands_e, f"band mismatch: WAVEDER={nbands}, EIGENVAL={nbands_e}"

        transitions = compute_transition_dipoles(
            cder, eigenvalues, occupations,
            energy_range=(0, 15)
        )
        n_val = np.sum(np.mean(occupations, axis=0) > 0.5)
        analyze_and_plot(transitions, eigenvalues, n_val)

    except FileNotFoundError as e:
        print(f"VASP files not available: {e}")
        print("Use Method A (QE epsilon.x) for transition dipole analysis,")
        print("or run VASP with LOPTICS=.TRUE. to generate WAVEDER.")
```

### Complete Single-Script Workflow (QE)

```python
#!/usr/bin/env python3
"""
Complete transition dipole moment workflow using QE epsilon.x.
SCF -> NSCF -> epsilon.x -> transition dipole analysis + oscillator strengths.

Example: GaAs (zincblende, direct gap)
"""
import os
import subprocess
import glob
import numpy as np
from scipy.signal import find_peaks
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ── Configuration ────────────────────────────────────────────────────
PSEUDO_DIR = os.path.abspath("./pseudo")
OUTDIR = os.path.abspath("./tmp_dipole")
os.makedirs(OUTDIR, exist_ok=True)
os.makedirs(PSEUDO_DIR, exist_ok=True)
PREFIX = "gaas"
ECUTWFC = 60.0
ECUTRHO = 240.0
NBND = 60
NPROC = 4
KGRID = "16 16 16"

# ── Step 1: SCF ──────────────────────────────────────────────────────
scf_in = f"""&CONTROL
    calculation = 'scf', prefix = '{PREFIX}',
    outdir = '{OUTDIR}', pseudo_dir = '{PSEUDO_DIR}',
/
&SYSTEM
    ibrav = 2, celldm(1) = 10.6829,
    nat = 2, ntyp = 2,
    ecutwfc = {ECUTWFC}, ecutrho = {ECUTRHO},
    occupations = 'fixed',
/
&ELECTRONS
    conv_thr = 1.0d-10,
/
ATOMIC_SPECIES
  Ga 69.723 Ga_ONCV_PBE-1.2.upf
  As 74.922 As_ONCV_PBE-1.2.upf
ATOMIC_POSITIONS (crystal)
  Ga 0.00 0.00 0.00
  As 0.25 0.25 0.25
K_POINTS (automatic)
  {KGRID} 0 0 0
"""
with open(f"{PREFIX}_scf.in", "w") as f:
    f.write(scf_in)

print("[1/4] SCF...")
r = subprocess.run(["mpirun", "-np", str(NPROC), "pw.x", "-in", f"{PREFIX}_scf.in"],
                    capture_output=True, text=True, timeout=600)
with open(f"{PREFIX}_scf.out", "w") as f:
    f.write(r.stdout)
assert "convergence has been achieved" in r.stdout, "SCF failed!"
print("      Done.")

# ── Step 2: NSCF ────────────────────────────────────────────────────
nscf_in = f"""&CONTROL
    calculation = 'nscf', prefix = '{PREFIX}',
    outdir = '{OUTDIR}', pseudo_dir = '{PSEUDO_DIR}',
/
&SYSTEM
    ibrav = 2, celldm(1) = 10.6829,
    nat = 2, ntyp = 2,
    ecutwfc = {ECUTWFC}, ecutrho = {ECUTRHO},
    occupations = 'fixed', nbnd = {NBND},
    nosym = .true., noinv = .true.,
/
&ELECTRONS
    conv_thr = 1.0d-10,
/
ATOMIC_SPECIES
  Ga 69.723 Ga_ONCV_PBE-1.2.upf
  As 74.922 As_ONCV_PBE-1.2.upf
ATOMIC_POSITIONS (crystal)
  Ga 0.00 0.00 0.00
  As 0.25 0.25 0.25
K_POINTS (automatic)
  {KGRID} 0 0 0
"""
with open(f"{PREFIX}_nscf.in", "w") as f:
    f.write(nscf_in)

print(f"[2/4] NSCF (nbnd={NBND})...")
r = subprocess.run(["mpirun", "-np", str(NPROC), "pw.x", "-in", f"{PREFIX}_nscf.in"],
                    capture_output=True, text=True, timeout=1800)
with open(f"{PREFIX}_nscf.out", "w") as f:
    f.write(r.stdout)
assert r.returncode == 0, "NSCF failed!"
print("      Done.")

# ── Step 3: epsilon.x ───────────────────────────────────────────────
eps_in = f"""&INPUTPP
    prefix = '{PREFIX}', outdir = '{OUTDIR}',
    calculation = 'eps',
/
&ENERGY_GRID
    smeartype = 'gauss', intersmear = 0.05,
    wmin = 0.0, wmax = 15.0, nw = 3000,
    shift = 0.0,
/
"""
with open(f"{PREFIX}_epsilon.in", "w") as f:
    f.write(eps_in)

print("[3/4] epsilon.x...")
r = subprocess.run(["epsilon.x", "-in", f"{PREFIX}_epsilon.in"],
                    capture_output=True, text=True, timeout=600)
with open(f"{PREFIX}_epsilon.out", "w") as f:
    f.write(r.stdout)
assert r.returncode == 0, f"epsilon.x failed!"
print("      Done.")

# ── Step 4: Transition dipole analysis ───────────────────────────────
print("[4/4] Analyzing transition dipoles...")

def load_eps(fn):
    data = []
    with open(fn) as fh:
        for line in fh:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            try:
                v = [float(x) for x in parts]
                if len(v) >= 4:
                    data.append(v[:4])
            except ValueError:
                continue
    return np.array(data)

imf = sorted(glob.glob("epsi*.dat"))
assert imf, "No epsilon output files found!"
di = load_eps(imf[0])
energy = di[:, 0]
eps2_xx, eps2_yy, eps2_zz = di[:, 1], di[:, 2], di[:, 3]
eps2_avg = (eps2_xx + eps2_yy + eps2_zz) / 3.0

omega = np.maximum(energy, 1e-6)
p_squared_eff = eps2_avg * omega**2
f_eff = eps2_avg * omega

# Find transitions
peaks, _ = find_peaks(eps2_avg, height=0.1 * np.max(eps2_avg),
                      distance=30, prominence=0.05 * np.max(eps2_avg))

print(f"\n=== Key Transitions ===")
for i, p in enumerate(peaks[:10]):
    print(f"  T{i+1}: E = {energy[p]:.2f} eV, eps2 = {eps2_avg[p]:.3f}, "
          f"|p|^2_eff = {p_squared_eff[p]:.3f}")

# Polarization analysis
print(f"\n=== Polarization Selection Rules ===")
for i, p in enumerate(peaks[:5]):
    aniso = np.std([eps2_xx[p], eps2_yy[p], eps2_zz[p]]) / (eps2_avg[p] + 1e-10)
    if aniso < 0.05:
        rule = "Isotropic (allowed for all polarizations)"
    elif eps2_zz[p] < 0.1 * eps2_xx[p]:
        rule = "E perp c preferred"
    elif eps2_xx[p] < 0.1 * eps2_zz[p]:
        rule = "E || c preferred"
    else:
        rule = f"Weakly anisotropic (ratio: {aniso:.3f})"
    print(f"  T{i+1} ({energy[p]:.2f} eV): {rule}")

# Plot
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

axes[0, 0].plot(energy, eps2_avg, "r-", lw=1.2)
axes[0, 0].fill_between(energy, 0, eps2_avg, alpha=0.1, color="red")
for i, p in enumerate(peaks[:6]):
    axes[0, 0].annotate(f"T{i+1}", xy=(energy[p], eps2_avg[p]),
                        xytext=(0, 10), textcoords="offset points",
                        ha="center", fontsize=9)
axes[0, 0].set(xlabel="Energy (eV)", ylabel=r"$\varepsilon_2$",
               title="(a) Transitions in eps2", xlim=(0, 12))
axes[0, 0].grid(alpha=0.3)

axes[0, 1].plot(energy, p_squared_eff, "b-", lw=1.2)
axes[0, 1].set(xlabel="Energy (eV)", ylabel=r"$|p_{cv}|^2_{eff}$",
               title=r"(b) Effective $|p_{cv}|^2$", xlim=(0, 12))
axes[0, 1].grid(alpha=0.3)

dE = energy[1] - energy[0]
cumul = np.cumsum(f_eff * dE)
axes[1, 0].plot(energy, f_eff, "g-", lw=1.2, label="f(E)")
ax2 = axes[1, 0].twinx()
ax2.plot(energy, cumul, "k--", lw=1.0, label="Cumulative")
ax2.set_ylabel("Cumulative f")
axes[1, 0].set(xlabel="Energy (eV)", ylabel="f_eff",
               title="(c) Oscillator Strength", xlim=(0, 12))
axes[1, 0].grid(alpha=0.3)

axes[1, 1].plot(energy, eps2_xx, lw=1.0, label="xx")
axes[1, 1].plot(energy, eps2_yy, "--", lw=1.0, label="yy")
axes[1, 1].plot(energy, eps2_zz, "-.", lw=1.0, label="zz")
axes[1, 1].set(xlabel="Energy (eV)", ylabel=r"$\varepsilon_2$",
               title="(d) Polarization Components", xlim=(0, 12))
axes[1, 1].legend()
axes[1, 1].grid(alpha=0.3)

plt.suptitle("Transition Dipole Analysis -- GaAs (PBE, RPA)", fontsize=15, y=1.01)
plt.tight_layout()
plt.savefig("transition_dipole_complete.png", dpi=200, bbox_inches="tight")
print(f"\nSaved: transition_dipole_complete.png")
```

---

## Key Parameters

| Parameter | Where | Typical Value | Impact |
|---|---|---|---|
| `K_POINTS` | SCF/NSCF | 16x16x16+ | Critical for sampling all transitions. Sparse grids miss important k-points |
| `nbnd` | NSCF | 3--5x occupied | Must include enough conduction bands for the energy range of interest |
| `intersmear` | epsilon.x | 0.03--0.1 eV | Smaller values resolve individual transitions better but require denser k-grids |
| `nw` | epsilon.x | 2000--5000 | More points give finer energy resolution for identifying transitions |
| `LOPTICS` | VASP INCAR | `.TRUE.` | Required to generate WAVEDER file |
| `NBANDS` | VASP INCAR | 2--4x occupied | Same role as nbnd in QE |
| `nosym/noinv` | NSCF | `.true.` | Required for epsilon.x; generates full k-mesh |
| `shift` | epsilon.x | 0.0--1.5 eV | Scissors correction shifts transition energies |

## Interpreting Results

- **Transition dipole magnitude**: Larger |p_cv|^2 means stronger optical coupling. Transitions with |p_cv|^2 near zero are dipole-forbidden (but may be quadrupole-allowed).

- **Oscillator strength**: f_cv is dimensionless. Values > 0.1 per band pair per k-point indicate strong transitions. The f-sum rule states that the total integrated oscillator strength equals the number of electrons.

- **Selection rules**: For cubic crystals (GaAs, Si), eps2_xx = eps2_yy = eps2_zz, so all polarizations are equivalent. For hexagonal crystals (GaN, ZnO), in-plane (xx, yy) and out-of-plane (zz) transitions differ, revealing polarization selection rules.

- **Band-pair decomposition**: Identifying which (VB, CB) pairs dominate the oscillator strength helps understand the optical response. The highest VB to lowest CB transition is often (but not always) the strongest.

- **Comparison with experiment**: PBE transition energies are red-shifted due to band gap underestimation. Apply scissors correction or use hybrid functionals. Oscillator strength ratios between transitions are more reliable than absolute values.

- **Radiative lifetime**: The spontaneous emission rate is proportional to omega^3 * |d_cv|^2 * n_refr, where d_cv is the transition dipole in length gauge (d = p / (m * omega)).

## Common Issues

| Problem | Solution |
|---|---|
| Cannot resolve individual transitions in eps2 | Reduce `intersmear` to 0.03--0.05 eV and increase k-grid density |
| All eps2 components are identical for non-cubic system | Check that the structure has correct symmetry. Ensure `nosym=.true.` in NSCF |
| Oscillator strength sum does not converge | Include more bands (increase `nbnd`). The f-sum rule requires all bands |
| WAVEDER file not generated by VASP | Ensure `LOPTICS = .TRUE.` and `LWAVE = .TRUE.` in INCAR |
| WAVEDER is very large | Normal for many bands/k-points. File size scales as nkpts * nbands^2 * 48 bytes |
| Transitions appear at wrong energies | Apply scissors correction. PBE systematically underestimates gaps |
| Need transition dipoles between specific bands only | Use the band-pair analysis code to filter by (VB, CB) indices |
| eps2 is zero at the gap energy for indirect-gap material | Indirect transitions are phonon-assisted and not captured by epsilon.x or LOPTICS |
