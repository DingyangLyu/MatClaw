# Optical Absorption Spectrum

## When to Use

- You need the optical absorption coefficient alpha(omega) of a crystal.
- You want to compute the reflectivity R(omega) or refractive index n(omega).
- You need the joint density of states (JDOS) to understand transition origins.
- You are comparing theoretical absorption spectra with UV-Vis or ellipsometry experiments.
- You want to assess a material's suitability for photovoltaic or optoelectronic applications.

## Method Selection

| Criterion | ASE + MACE | QE DFT (epsilon.x) |
|---|---|---|
| Absorption spectrum | Not available | From Im(epsilon) via RPA |
| Reflectivity / refractive index | Not available | Derived from complex epsilon |
| Excitonic effects | Not available | Not included in RPA; need BSE (not in standard QE) |

**MACE cannot compute optical absorption.** Always use QE. The workflow is: SCF -> NSCF (many bands) -> epsilon.x -> post-process to get alpha, R, n, k.

## Prerequisites

- A relaxed crystal structure.
- Norm-conserving pseudopotentials in `./pseudo/` (recommended for epsilon.x).
- QE executables: `pw.x`, `epsilon.x`.
- Python packages: `numpy`, `scipy`, `matplotlib`.
- Completed SCF and NSCF calculations (see the `dielectric-function` skill for setup).

---

## Detailed Steps

### Background: Optical Properties from the Dielectric Function

All optical properties derive from the complex dielectric function:

```
epsilon(omega) = eps1(omega) + i * eps2(omega)
```

The key relationships are:

```
Complex refractive index:  N = n + ik = sqrt(epsilon)
  n(omega) = sqrt[ (sqrt(eps1^2 + eps2^2) + eps1) / 2 ]
  k(omega) = sqrt[ (sqrt(eps1^2 + eps2^2) - eps1) / 2 ]

Absorption coefficient:  alpha(omega) = 2 * omega * k / c
  = (2 * omega / c) * sqrt[ (sqrt(eps1^2 + eps2^2) - eps1) / 2 ]
  In practical units: alpha (cm^-1) = omega(eV) * k / (hbar * c)

Normal-incidence reflectivity:  R = |(N-1)/(N+1)|^2
  = [(n-1)^2 + k^2] / [(n+1)^2 + k^2]
```

### Step 1: SCF Calculation

```python
#!/usr/bin/env python3
"""
Step 1: SCF calculation for optical absorption.
Dense k-grid is critical for smooth absorption spectra.
"""
import os
import subprocess

PSEUDO_DIR = os.path.abspath("./pseudo")
OUTDIR = os.path.abspath("./tmp_optical")
os.makedirs(OUTDIR, exist_ok=True)
os.makedirs(PSEUDO_DIR, exist_ok=True)
PREFIX = "si"

scf_input = f"""&CONTROL
    calculation  = 'scf'
    prefix       = '{PREFIX}'
    outdir       = '{OUTDIR}'
    pseudo_dir   = '{PSEUDO_DIR}'
/

&SYSTEM
    ibrav        = 2
    celldm(1)    = 10.26
    nat          = 2
    ntyp         = 1
    ecutwfc      = 60.0
    ecutrho      = 240.0
    occupations  = 'fixed'
/

&ELECTRONS
    conv_thr = 1.0d-10
/

ATOMIC_SPECIES
  Si  28.0855  Si_ONCV_PBE-1.2.upf

ATOMIC_POSITIONS (crystal)
  Si  0.00  0.00  0.00
  Si  0.25  0.25  0.25

K_POINTS (automatic)
  16 16 16  0 0 0
"""
# NOTE: 16x16x16 gives smoother spectra than 12x12x12.
# For publication quality, test convergence with 20x20x20.

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

### Step 2: NSCF with Many Bands

```python
#!/usr/bin/env python3
"""
Step 2: NSCF calculation with many bands.
For absorption up to ~10 eV, need enough empty bands to cover that range.
"""
import os
import subprocess

PSEUDO_DIR = os.path.abspath("./pseudo")
OUTDIR = os.path.abspath("./tmp_optical")
PREFIX = "si"

# Si: 4 occupied bands, want transitions up to ~15 eV
# Rule: nbnd ~ occupied + (Emax / avg_band_spacing)
# For Si with 16^3 k-grid, ~60 bands covers transitions to ~30 eV
NBND = 60

nscf_input = f"""&CONTROL
    calculation  = 'nscf'
    prefix       = '{PREFIX}'
    outdir       = '{OUTDIR}'
    pseudo_dir   = '{PSEUDO_DIR}'
/

&SYSTEM
    ibrav        = 2
    celldm(1)    = 10.26
    nat          = 2
    ntyp         = 1
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
  Si  28.0855  Si_ONCV_PBE-1.2.upf

ATOMIC_POSITIONS (crystal)
  Si  0.00  0.00  0.00
  Si  0.25  0.25  0.25

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

### Step 3: epsilon.x

```python
#!/usr/bin/env python3
"""
Step 3: Run epsilon.x for absorption spectrum.
Use smaller broadening for more resolved features.
"""
import os
import subprocess

OUTDIR = os.path.abspath("./tmp_optical")
PREFIX = "si"

epsilon_input = f"""&INPUTPP
    prefix       = '{PREFIX}'
    outdir       = '{OUTDIR}'
    calculation  = 'eps'
/

&ENERGY_GRID
    smeartype    = 'gauss'
    intersmear   = 0.1
    wmin         = 0.0
    wmax         = 20.0
    nw           = 2000
    shift        = 0.0
/
"""
# For scissors correction: set shift = (experimental_gap - DFT_gap)
# Si: experimental direct gap ~3.4 eV, PBE ~2.5 eV, so shift ~ 0.9

with open(f"{PREFIX}_epsilon.in", "w") as f:
    f.write(epsilon_input)

print("Running epsilon.x...")
result = subprocess.run(
    ["epsilon.x", "-in", f"{PREFIX}_epsilon.in"],
    capture_output=True, text=True, timeout=600
)
with open(f"{PREFIX}_epsilon.out", "w") as f:
    f.write(result.stdout)

assert result.returncode == 0, f"epsilon.x failed!\n{result.stderr}"
print("epsilon.x completed.")
```

### Step 4: Compute Absorption Coefficient, Reflectivity, Refractive Index

```python
#!/usr/bin/env python3
"""
Step 4: Convert dielectric function to all optical properties.

From eps1(omega) and eps2(omega), compute:
  - Complex refractive index: n(omega), k(omega)
  - Absorption coefficient: alpha(omega)
  - Normal-incidence reflectivity: R(omega)
  - Energy loss function: -Im(1/epsilon)
  - Joint density of states (proportional to eps2*omega^2)
"""
import numpy as np
import glob
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ── Physical constants ───────────────────────────────────────────────
HBAR_EV_S = 6.582119569e-16   # hbar in eV*s
C_CM_S = 2.99792458e10        # speed of light in cm/s
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
eps1 = (dr[:, 1] + dr[:, 2] + dr[:, 3]) / 3.0  # average for cubic
eps2 = (di[:, 1] + di[:, 2] + di[:, 3]) / 3.0

print(f"Loaded {len(energy)} frequency points from {real_files[0]} and {imag_files[0]}")
print(f"Energy range: {energy[0]:.3f} -- {energy[-1]:.3f} eV")

# ── Compute optical properties ───────────────────────────────────────

# Complex refractive index: N = n + ik = sqrt(eps1 + i*eps2)
eps_abs = np.sqrt(eps1**2 + eps2**2)
n = np.sqrt((eps_abs + eps1) / 2.0)    # refractive index
k = np.sqrt((eps_abs - eps1) / 2.0)    # extinction coefficient

# Absorption coefficient: alpha = 2*omega*k/c
# omega = E/hbar, so alpha = 2*E*k / (hbar*c)
# In cm^-1: alpha = 2 * E(eV) * k / (hbar(eV*s) * c(cm/s))
# Avoid division by zero at E=0
alpha = np.zeros_like(energy)
mask = energy > 0
alpha[mask] = 2.0 * energy[mask] * k[mask] / (HBAR_EV_S * C_CM_S)

# Normal-incidence reflectivity: R = |(n-1+ik)/(n+1+ik)|^2
#   = [(n-1)^2 + k^2] / [(n+1)^2 + k^2]
R = ((n - 1)**2 + k**2) / ((n + 1)**2 + k**2)

# Energy loss function: -Im(1/epsilon) = eps2 / (eps1^2 + eps2^2)
eels = eps2 / (eps1**2 + eps2**2)

# Joint density of states (proportional): JDOS ~ eps2 * omega^2 / |p_cv|^2
# Simplified (assuming constant matrix element): JDOS ~ eps2 * E^2
jdos = eps2 * energy**2

# ── Print key values ─────────────────────────────────────────────────
print(f"\n=== Optical Properties Summary ===")
print(f"Static refractive index n(0):     {n[0]:.4f}")
print(f"  (Experimental Si: ~3.42)")
print(f"Static dielectric constant eps(0): {eps1[0]:.4f}")

# Absorption onset
abs_threshold = 100  # cm^-1
onset_idx = np.where(alpha > abs_threshold)[0]
if len(onset_idx) > 0:
    print(f"Absorption onset (alpha > {abs_threshold} cm^-1): {energy[onset_idx[0]]:.2f} eV")

# Peak absorption
peak_idx = np.argmax(alpha)
print(f"Peak absorption coefficient: {alpha[peak_idx]:.2e} cm^-1 at {energy[peak_idx]:.2f} eV")

# Maximum reflectivity
max_R_idx = np.argmax(R[energy > 1.0]) + np.argmin(np.abs(energy - 1.0))
print(f"Maximum reflectivity: {R[max_R_idx]*100:.1f}% at {energy[max_R_idx]:.2f} eV")

# Plasma frequency (where eps1 crosses zero)
zero_crossings = np.where(np.diff(np.sign(eps1)))[0]
for zc in zero_crossings:
    if eps2[zc] > 0.5:  # meaningful crossing
        print(f"eps1 zero crossing at {energy[zc]:.2f} eV")

# ── Convert energy to wavelength for practical use ───────────────────
wavelength_nm = 1239.84 / energy[energy > 0.1]  # E(eV) = 1239.84/lambda(nm)
alpha_vs_wl = alpha[energy > 0.1]

# ── Plotting ─────────────────────────────────────────────────────────

# Figure 1: Absorption coefficient
fig1, ax1 = plt.subplots(figsize=(10, 6))
ax1.semilogy(energy, alpha, color="darkred", linewidth=1.5)
ax1.set_xlabel("Photon Energy (eV)", fontsize=14)
ax1.set_ylabel(r"Absorption Coefficient $\alpha$ (cm$^{-1}$)", fontsize=14)
ax1.set_title("Optical Absorption Coefficient -- Si (PBE, RPA)", fontsize=15)
ax1.set_xlim(0, 12)
ax1.set_ylim(1e2, 1e8)
ax1.grid(True, alpha=0.3)
ax1.axvline(3.4, color="green", linestyle="--", alpha=0.5, label="Expt. direct gap (3.4 eV)")
ax1.legend(fontsize=11)
plt.tight_layout()
plt.savefig("absorption_coefficient.png", dpi=200, bbox_inches="tight")
print("\nSaved: absorption_coefficient.png")

# Figure 2: Reflectivity
fig2, ax2 = plt.subplots(figsize=(10, 6))
ax2.plot(energy, R * 100, color="navy", linewidth=1.5)
ax2.set_xlabel("Photon Energy (eV)", fontsize=14)
ax2.set_ylabel("Reflectivity R (%)", fontsize=14)
ax2.set_title("Normal-Incidence Reflectivity -- Si (PBE, RPA)", fontsize=15)
ax2.set_xlim(0, 15)
ax2.set_ylim(0, 80)
ax2.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("reflectivity.png", dpi=200, bbox_inches="tight")
print("Saved: reflectivity.png")

# Figure 3: Refractive index and extinction coefficient
fig3, ax3 = plt.subplots(figsize=(10, 6))
ax3.plot(energy, n, color="steelblue", linewidth=1.5, label="n (refractive index)")
ax3.plot(energy, k, color="crimson", linewidth=1.5, label="k (extinction coefficient)")
ax3.set_xlabel("Photon Energy (eV)", fontsize=14)
ax3.set_ylabel("n, k", fontsize=14)
ax3.set_title("Complex Refractive Index -- Si (PBE, RPA)", fontsize=15)
ax3.set_xlim(0, 15)
ax3.legend(fontsize=12)
ax3.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("refractive_index.png", dpi=200, bbox_inches="tight")
print("Saved: refractive_index.png")

# Figure 4: Combined panel plot (publication-quality)
fig4, axes = plt.subplots(2, 2, figsize=(14, 10))

# Panel (a): Dielectric function
axes[0, 0].plot(energy, eps1, "b-", linewidth=1.2, label=r"$\varepsilon_1$")
axes[0, 0].plot(energy, eps2, "r-", linewidth=1.2, label=r"$\varepsilon_2$")
axes[0, 0].axhline(0, color="gray", linestyle="--", linewidth=0.5)
axes[0, 0].set_xlabel("Energy (eV)")
axes[0, 0].set_ylabel(r"$\varepsilon(\omega)$")
axes[0, 0].set_title("(a) Dielectric Function")
axes[0, 0].legend()
axes[0, 0].set_xlim(0, 12)
axes[0, 0].grid(alpha=0.3)

# Panel (b): Absorption coefficient
axes[0, 1].semilogy(energy, alpha, "darkred", linewidth=1.2)
axes[0, 1].set_xlabel("Energy (eV)")
axes[0, 1].set_ylabel(r"$\alpha$ (cm$^{-1}$)")
axes[0, 1].set_title("(b) Absorption Coefficient")
axes[0, 1].set_xlim(0, 12)
axes[0, 1].set_ylim(1e2, 1e8)
axes[0, 1].grid(alpha=0.3)

# Panel (c): Reflectivity
axes[1, 0].plot(energy, R * 100, "navy", linewidth=1.2)
axes[1, 0].set_xlabel("Energy (eV)")
axes[1, 0].set_ylabel("R (%)")
axes[1, 0].set_title("(c) Reflectivity")
axes[1, 0].set_xlim(0, 12)
axes[1, 0].grid(alpha=0.3)

# Panel (d): Refractive index
axes[1, 1].plot(energy, n, "steelblue", linewidth=1.2, label="n")
axes[1, 1].plot(energy, k, "crimson", linewidth=1.2, label="k")
axes[1, 1].set_xlabel("Energy (eV)")
axes[1, 1].set_ylabel("n, k")
axes[1, 1].set_title("(d) Refractive Index")
axes[1, 1].legend()
axes[1, 1].set_xlim(0, 12)
axes[1, 1].grid(alpha=0.3)

plt.suptitle("Optical Properties of Si (PBE, RPA)", fontsize=16, y=1.01)
plt.tight_layout()
plt.savefig("optical_properties_panel.png", dpi=200, bbox_inches="tight")
print("Saved: optical_properties_panel.png")

# Figure 5: Energy loss function
fig5, ax5 = plt.subplots(figsize=(10, 6))
ax5.plot(energy, eels, color="darkgreen", linewidth=1.5)
ax5.set_xlabel("Energy (eV)", fontsize=14)
ax5.set_ylabel(r"$-\mathrm{Im}[1/\varepsilon(\omega)]$", fontsize=14)
ax5.set_title("Electron Energy Loss Function -- Si (PBE, RPA)", fontsize=15)
ax5.set_xlim(0, 25)
ax5.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("energy_loss_function.png", dpi=200, bbox_inches="tight")
print("Saved: energy_loss_function.png")

# Figure 6: Absorption vs wavelength (for experimentalists)
fig6, ax6 = plt.subplots(figsize=(10, 6))
ax6.semilogy(wavelength_nm, alpha_vs_wl, color="darkred", linewidth=1.5)
ax6.set_xlabel("Wavelength (nm)", fontsize=14)
ax6.set_ylabel(r"$\alpha$ (cm$^{-1}$)", fontsize=14)
ax6.set_title("Absorption Coefficient vs Wavelength -- Si", fontsize=15)
ax6.set_xlim(100, 1200)
ax6.invert_xaxis()
ax6.grid(True, alpha=0.3)
# Mark visible range
ax6.axvspan(380, 780, alpha=0.08, color="yellow", label="Visible range")
ax6.legend(fontsize=11)
plt.tight_layout()
plt.savefig("absorption_vs_wavelength.png", dpi=200, bbox_inches="tight")
print("Saved: absorption_vs_wavelength.png")
```

### Complete Single-Script Workflow

```python
#!/usr/bin/env python3
"""
Complete absorption spectrum workflow in one script.
SCF -> NSCF -> epsilon.x -> absorption/reflectivity/refractive index plots

Example: Silicon (diamond structure)
"""
import os
import subprocess
import glob
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ── Configuration ────────────────────────────────────────────────────
PSEUDO_DIR = os.path.abspath("./pseudo")
OUTDIR = os.path.abspath("./tmp_optical")
os.makedirs(OUTDIR, exist_ok=True)
os.makedirs(PSEUDO_DIR, exist_ok=True)
PREFIX = "si"
ECUTWFC = 60.0
ECUTRHO = 240.0
NBND = 60
NPROC = 4
KGRID = "16 16 16"
PP_FILE = "Si_ONCV_PBE-1.2.upf"
HBAR_EV_S = 6.582119569e-16
C_CM_S = 2.99792458e10

# ── Step 1: SCF ──────────────────────────────────────────────────────
scf_in = f"""&CONTROL
    calculation = 'scf', prefix = '{PREFIX}',
    outdir = '{OUTDIR}', pseudo_dir = '{PSEUDO_DIR}',
/
&SYSTEM
    ibrav = 2, celldm(1) = 10.26,
    nat = 2, ntyp = 1,
    ecutwfc = {ECUTWFC}, ecutrho = {ECUTRHO},
    occupations = 'fixed',
/
&ELECTRONS
    conv_thr = 1.0d-10,
/
ATOMIC_SPECIES
  Si 28.0855 {PP_FILE}
ATOMIC_POSITIONS (crystal)
  Si 0.00 0.00 0.00
  Si 0.25 0.25 0.25
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
    ibrav = 2, celldm(1) = 10.26,
    nat = 2, ntyp = 1,
    ecutwfc = {ECUTWFC}, ecutrho = {ECUTRHO},
    occupations = 'fixed', nbnd = {NBND},
    nosym = .true., noinv = .true.,
/
&ELECTRONS
    conv_thr = 1.0d-10,
/
ATOMIC_SPECIES
  Si 28.0855 {PP_FILE}
ATOMIC_POSITIONS (crystal)
  Si 0.00 0.00 0.00
  Si 0.25 0.25 0.25
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
    smeartype = 'gauss', intersmear = 0.1,
    wmin = 0.0, wmax = 20.0, nw = 2000,
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

# ── Step 4: Post-process and plot ────────────────────────────────────
print("[4/4] Post-processing...")

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
                elif len(v) >= 2:
                    data.append([v[0], v[1], v[1], v[1]])
            except ValueError:
                continue
    return np.array(data)

rf = sorted(glob.glob("epsr*.dat"))
imf = sorted(glob.glob("epsi*.dat"))
assert rf and imf, "No epsilon output files found!"

dr = load_eps(rf[0])
di = load_eps(imf[0])
energy = dr[:, 0]
eps1 = (dr[:, 1] + dr[:, 2] + dr[:, 3]) / 3.0
eps2 = (di[:, 1] + di[:, 2] + di[:, 3]) / 3.0

# Compute optical properties
eps_abs = np.sqrt(eps1**2 + eps2**2)
n = np.sqrt((eps_abs + eps1) / 2.0)
k = np.sqrt((eps_abs - eps1) / 2.0)
alpha = np.zeros_like(energy)
mask = energy > 0
alpha[mask] = 2.0 * energy[mask] * k[mask] / (HBAR_EV_S * C_CM_S)
R = ((n - 1)**2 + k**2) / ((n + 1)**2 + k**2)
eels = np.where(eps_abs > 1e-10, eps2 / eps_abs**2, 0)

# Print summary
print(f"\n=== Optical Properties Summary ===")
print(f"n(0) = {n[0]:.3f}  (expt Si: 3.42)")
print(f"eps1(0) = {eps1[0]:.3f}  (expt Si: 11.7)")
abs_onset = energy[np.where(alpha > 100)[0][0]] if np.any(alpha > 100) else float("nan")
print(f"Absorption onset: {abs_onset:.2f} eV")
print(f"Peak alpha: {np.max(alpha):.2e} cm^-1 at {energy[np.argmax(alpha)]:.2f} eV")

# Four-panel plot
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

axes[0, 0].plot(energy, eps1, "b-", lw=1.3, label=r"$\varepsilon_1$")
axes[0, 0].plot(energy, eps2, "r-", lw=1.3, label=r"$\varepsilon_2$")
axes[0, 0].axhline(0, color="gray", ls="--", lw=0.5)
axes[0, 0].set(xlabel="Energy (eV)", ylabel=r"$\varepsilon$", title="(a) Dielectric Function", xlim=(0, 12))
axes[0, 0].legend()
axes[0, 0].grid(alpha=0.3)

axes[0, 1].semilogy(energy, alpha, "darkred", lw=1.3)
axes[0, 1].set(xlabel="Energy (eV)", ylabel=r"$\alpha$ (cm$^{-1}$)",
               title="(b) Absorption Coefficient", xlim=(0, 12), ylim=(1e2, 1e8))
axes[0, 1].grid(alpha=0.3)

axes[1, 0].plot(energy, R * 100, "navy", lw=1.3)
axes[1, 0].set(xlabel="Energy (eV)", ylabel="R (%)",
               title="(c) Reflectivity", xlim=(0, 12))
axes[1, 0].grid(alpha=0.3)

axes[1, 1].plot(energy, n, "steelblue", lw=1.3, label="n")
axes[1, 1].plot(energy, k, "crimson", lw=1.3, label="k")
axes[1, 1].set(xlabel="Energy (eV)", ylabel="n, k",
               title="(d) Refractive Index", xlim=(0, 12))
axes[1, 1].legend()
axes[1, 1].grid(alpha=0.3)

plt.suptitle("Optical Properties -- Si (PBE, RPA)", fontsize=16, y=1.01)
plt.tight_layout()
plt.savefig("optical_properties.png", dpi=200, bbox_inches="tight")
print(f"\nSaved: optical_properties.png")
```

---

## Key Parameters

| Parameter | Where | Typical Value | Impact |
|---|---|---|---|
| `K_POINTS` | SCF/NSCF | 16x16x16+ (bulk) | Most critical for smooth spectra. Insufficient k-points cause jagged, noisy absorption curves |
| `nbnd` | NSCF | 2--4x occupied | Determines maximum transition energy. Too few = spectrum truncated at low energy |
| `intersmear` | epsilon.x | 0.05--0.3 eV | Broadening. Smaller = more features but noisier. Larger = smoother but may obscure real features |
| `nw` | epsilon.x | 1000--2000 | Frequency resolution. More points = smoother interpolation |
| `wmax` | epsilon.x | 15--40 eV | Maximum photon energy. Set below the range covered by nbnd |
| `shift` | epsilon.x | 0.0--1.5 eV | Scissors correction. Apply if PBE gap is known to underestimate the experimental gap |
| `nosym` / `noinv` | NSCF | `.true.` | Required for epsilon.x; disables symmetry to get full k-mesh |
| `ecutwfc` | All | 60--80 Ry (NC) | Follow pseudopotential recommendation. Higher for NC vs US |

### K-Point Convergence for Optical Properties

Optical properties converge much more slowly with k-points than total energies. Recommended convergence test:

```python
# Test k-grids: 8x8x8, 12x12x12, 16x16x16, 20x20x20
# Compare: peak height in eps2, absorption onset, reflectivity maximum
# Typically 16x16x16 is adequate for bulk Si; some materials need 24x24x24+
```

## Interpreting Results

- **Absorption coefficient units**: alpha in cm^-1. Typical values range from 10^2 (near gap) to 10^6 (strong absorption).

- **Absorption onset**: Identifies the optical (direct) gap. Note PBE underestimates this. For indirect-gap materials (Si, Ge), the indirect gap is not captured by epsilon.x (which only computes direct, vertical transitions).

- **Peaks in eps2 / absorption**: Correspond to Van Hove singularities in the joint density of states -- critical points where valence and conduction bands are parallel.

- **Reflectivity plateau**: High reflectivity (~30-60%) in the region where eps2 is large and eps1 is negative is typical for semiconductors (reststrahlen band).

- **Comparison with experiment**: PBE-RPA systematically:
  - Underestimates the band gap (red-shifts the absorption edge)
  - Overestimates eps2 peak heights (no excitonic redistribution)
  - Gets qualitatively correct peak positions and spectral shape
  - Apply scissors correction (`shift`) to improve onset position

- **Wavelength conversion**: E(eV) = 1239.84 / lambda(nm). Visible range is 380--780 nm (1.59--3.26 eV).

- **For photovoltaic assessment**: Look at absorption in the visible/near-IR range. Good absorbers have alpha > 10^4 cm^-1 across the solar spectrum (1.1--3.5 eV).

## Common Issues

| Problem | Solution |
|---|---|
| Absorption spectrum is very jagged/noisy | Increase k-point density (most common fix). 16x16x16 minimum for bulk |
| Absorption onset at wrong energy | Apply scissors correction via `shift` in epsilon.x. Or use hybrid functionals for SCF |
| alpha is zero below the gap | Expected for direct-gap materials. For indirect gaps, phonon-assisted absorption is not captured |
| Reflectivity > 100% or negative | Numerical issue. Check that eps1 and eps2 are physically reasonable. Increase k-points |
| Spectrum stops abruptly at some energy | `nbnd` too small. Increase to include more empty states |
| No output .dat files | Check epsilon.x stdout for errors. Common: prefix/outdir mismatch with NSCF |
| Very different xx/yy/zz components for cubic | Symmetry might not be well converged. Check structure and k-grid |
| Need to compare with EELS experiment | Use the energy loss function -Im(1/epsilon), not eps2 directly |
| Memory errors in NSCF | Reduce nbnd or use fewer k-points. Or increase MPI ranks |
