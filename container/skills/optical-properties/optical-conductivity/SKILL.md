# Optical Conductivity

## When to Use

- You need the frequency-dependent optical conductivity sigma(omega) of a material.
- You want to characterize the charge dynamics and carrier response in metals or doped semiconductors.
- You need the Drude parameters (plasma frequency, scattering rate) for a metal.
- You are comparing with infrared/THz spectroscopy or optical conductivity measurements.
- You want to verify the f-sum rule to assess the quality of your calculation.
- You need the DC conductivity limit or the spectral weight of interband transitions.

## Method Selection

| Criterion | ASE + MACE | QE DFT (epsilon.x) |
|---|---|---|
| Optical conductivity | Not available | Derived from the dielectric function |
| Drude weight (metals) | Not available | From plasma frequency via epsilon.x or from Fermi surface |
| DC conductivity | Not available | Extrapolation from sigma(omega->0) for metals |

**MACE cannot compute optical conductivity.** It is a force field without electronic states. Always use QE.

## Prerequisites

- A relaxed crystal structure.
- Norm-conserving pseudopotentials in `./pseudo/`.
- QE executables: `pw.x`, `epsilon.x`.
- Python packages: `numpy`, `scipy`, `matplotlib`.
- Completed SCF + NSCF + epsilon.x workflow (see the `dielectric-function` skill).

---

## Detailed Steps

### Background: Optical Conductivity and the Dielectric Function

The complex optical conductivity sigma(omega) is related to the complex dielectric function epsilon(omega) by:

```
sigma(omega) = -i * omega * epsilon_0 * (epsilon(omega) - 1)

where epsilon(omega) = eps1(omega) + i * eps2(omega)

Separating real and imaginary parts:
  sigma_1(omega) = omega * epsilon_0 * eps2(omega)          [real part]
  sigma_2(omega) = -omega * epsilon_0 * (eps1(omega) - 1)   [imaginary part]
```

In Gaussian units (common in condensed matter):
```
  sigma_1(omega) = omega * eps2(omega) / (4*pi)
  sigma_2(omega) = -omega * (eps1(omega) - 1) / (4*pi)
```

In practical SI units with omega in eV:
```
  sigma_1 (Ohm^-1 cm^-1) = omega(eV) * eps2 * epsilon_0 / hbar
                          = omega(s^-1) * eps2 * epsilon_0
```

The **f-sum rule** provides a powerful consistency check:
```
  integral_0^inf sigma_1(omega) d(omega) = pi * e^2 * n_e / (2 * m_e)
                                         = omega_p^2 * epsilon_0 / 2

where omega_p is the plasma frequency and n_e is the electron density.
```

For **metals**, the dielectric function has an additional **intraband (Drude)** contribution:
```
  epsilon_Drude(omega) = 1 - omega_p^2 / (omega^2 + i*omega*gamma)

  where omega_p = plasma frequency, gamma = scattering rate (1/tau)

  This gives:
    eps1_Drude = 1 - omega_p^2 / (omega^2 + gamma^2)
    eps2_Drude = omega_p^2 * gamma / (omega * (omega^2 + gamma^2))

  And the Drude conductivity:
    sigma1_Drude = epsilon_0 * omega_p^2 * gamma / (omega^2 + gamma^2)
    sigma1_Drude(0) = epsilon_0 * omega_p^2 / gamma = sigma_DC
```

### Step 1: SCF + NSCF + epsilon.x (Same as Dielectric Function Skill)

```python
#!/usr/bin/env python3
"""
Steps 1-3: Run the full QE workflow (SCF -> NSCF -> epsilon.x).
This is identical to the dielectric-function skill.
For metals, use 'smearing' occupation and a denser k-grid.
"""
import os
import subprocess

PSEUDO_DIR = os.path.abspath("./pseudo")
OUTDIR = os.path.abspath("./tmp_sigma")
os.makedirs(OUTDIR, exist_ok=True)
os.makedirs(PSEUDO_DIR, exist_ok=True)
PREFIX = "al"   # Example: Aluminum (metal)
ECUTWFC = 60.0
ECUTRHO = 240.0
NBND = 40
NPROC = 4

# ── SCF for a metal (Al, FCC) ───────────────────────────────────────
scf_input = f"""&CONTROL
    calculation  = 'scf'
    prefix       = '{PREFIX}'
    outdir       = '{OUTDIR}'
    pseudo_dir   = '{PSEUDO_DIR}'
/
&SYSTEM
    ibrav        = 2
    celldm(1)    = 7.63
    nat          = 1
    ntyp         = 1
    ecutwfc      = {ECUTWFC}
    ecutrho      = {ECUTRHO}
    occupations  = 'smearing'
    smearing     = 'mp'
    degauss      = 0.02
/
&ELECTRONS
    conv_thr = 1.0d-10
/
ATOMIC_SPECIES
  Al  26.9815  Al.pbe-n-kjpaw_psl.1.0.0.UPF
ATOMIC_POSITIONS (crystal)
  Al  0.00  0.00  0.00
K_POINTS (automatic)
  20 20 20  0 0 0
"""
# NOTE: For metals, a very dense k-grid (20x20x20+) is essential
# to resolve the Fermi surface and intraband transitions.

with open(f"{PREFIX}_scf.in", "w") as f:
    f.write(scf_input)

print("[1/3] Running SCF for metal...")
r = subprocess.run(["mpirun", "-np", str(NPROC), "pw.x", "-in", f"{PREFIX}_scf.in"],
                    capture_output=True, text=True, timeout=600)
with open(f"{PREFIX}_scf.out", "w") as f:
    f.write(r.stdout)
assert "convergence has been achieved" in r.stdout, "SCF did not converge!"
print("      SCF converged.")

# ── NSCF ─────────────────────────────────────────────────────────────
nscf_input = f"""&CONTROL
    calculation  = 'nscf'
    prefix       = '{PREFIX}'
    outdir       = '{OUTDIR}'
    pseudo_dir   = '{PSEUDO_DIR}'
/
&SYSTEM
    ibrav        = 2
    celldm(1)    = 7.63
    nat          = 1
    ntyp         = 1
    ecutwfc      = {ECUTWFC}
    ecutrho      = {ECUTRHO}
    occupations  = 'smearing'
    smearing     = 'mp'
    degauss      = 0.02
    nbnd         = {NBND}
    nosym        = .true.
    noinv        = .true.
/
&ELECTRONS
    conv_thr = 1.0d-10
/
ATOMIC_SPECIES
  Al  26.9815  Al.pbe-n-kjpaw_psl.1.0.0.UPF
ATOMIC_POSITIONS (crystal)
  Al  0.00  0.00  0.00
K_POINTS (automatic)
  20 20 20  0 0 0
"""

with open(f"{PREFIX}_nscf.in", "w") as f:
    f.write(nscf_input)

print(f"[2/3] Running NSCF (nbnd={NBND})...")
r = subprocess.run(["mpirun", "-np", str(NPROC), "pw.x", "-in", f"{PREFIX}_nscf.in"],
                    capture_output=True, text=True, timeout=1200)
with open(f"{PREFIX}_nscf.out", "w") as f:
    f.write(r.stdout)
assert r.returncode == 0, "NSCF failed!"
print("      NSCF completed.")

# ── epsilon.x ────────────────────────────────────────────────────────
# For metals, epsilon.x computes only the INTERBAND contribution.
# The intraband (Drude) part must be added separately in post-processing.
epsilon_input = f"""&INPUTPP
    prefix       = '{PREFIX}'
    outdir       = '{OUTDIR}'
    calculation  = 'eps'
/
&ENERGY_GRID
    smeartype    = 'gauss'
    intersmear   = 0.15
    wmin         = 0.0
    wmax         = 30.0
    nw           = 1500
    shift        = 0.0
/
"""

with open(f"{PREFIX}_epsilon.in", "w") as f:
    f.write(epsilon_input)

print("[3/3] Running epsilon.x...")
r = subprocess.run(["epsilon.x", "-in", f"{PREFIX}_epsilon.in"],
                    capture_output=True, text=True, timeout=600)
with open(f"{PREFIX}_epsilon.out", "w") as f:
    f.write(r.stdout)
assert r.returncode == 0, f"epsilon.x failed!"
print("      epsilon.x completed.")
```

### Step 2: Compute Optical Conductivity from Dielectric Function

```python
#!/usr/bin/env python3
"""
Compute optical conductivity from epsilon.x output.
Handles both semiconductors/insulators and metals.
Includes Drude model fitting for metals.
"""
import numpy as np
import glob
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.integrate import trapezoid

# ── Physical constants ───────────────────────────────────────────────
EPSILON_0 = 8.854187817e-12    # F/m (vacuum permittivity)
HBAR_EV_S = 6.582119569e-16   # eV*s
HBAR_SI = 1.054571817e-34     # J*s
E_CHARGE = 1.602176634e-19    # C
M_ELECTRON = 9.1093837015e-31 # kg
C_LIGHT = 2.99792458e8        # m/s

# Conversion: 1 eV = 1.602e-19 J
# omega(rad/s) = E(eV) / hbar(eV*s)
# sigma (S/m) = omega(rad/s) * eps2 * epsilon_0

# ── Load dielectric function ────────────────────────────────────────
def load_eps_dat(filename):
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

real_files = sorted(glob.glob("epsr*.dat"))
imag_files = sorted(glob.glob("epsi*.dat"))
if not real_files or not imag_files:
    raise FileNotFoundError("No epsr/epsi .dat files found. Run epsilon.x first.")

dr = load_eps_dat(real_files[0])
di = load_eps_dat(imag_files[0])

energy_eV = dr[:, 0]  # eV
eps1 = (dr[:, 1] + dr[:, 2] + dr[:, 3]) / 3.0  # average for cubic
eps2 = (di[:, 1] + di[:, 2] + di[:, 3]) / 3.0

# Convert energy to angular frequency
omega_rad = energy_eV / HBAR_EV_S  # rad/s

print(f"Loaded {len(energy_eV)} frequency points")
print(f"Energy range: {energy_eV[0]:.3f} -- {energy_eV[-1]:.3f} eV")

# ── Compute optical conductivity (SI units: S/m = Ohm^-1 m^-1) ─────
# sigma_1 = omega * epsilon_0 * eps2    (real part, absorptive)
# sigma_2 = -omega * epsilon_0 * (eps1 - 1)  (imaginary part, reactive)

sigma1_SI = np.zeros_like(energy_eV)
sigma2_SI = np.zeros_like(energy_eV)

mask = energy_eV > 0  # avoid omega=0
sigma1_SI[mask] = omega_rad[mask] * EPSILON_0 * eps2[mask]
sigma2_SI[mask] = -omega_rad[mask] * EPSILON_0 * (eps1[mask] - 1.0)

# Convert to more common units: Ohm^-1 cm^-1
sigma1_cgs = sigma1_SI / 100.0  # 1 S/m = 0.01 S/cm = 0.01 Ohm^-1 cm^-1
sigma2_cgs = sigma2_SI / 100.0

print(f"\n=== Optical Conductivity (interband) ===")
print(f"sigma1 range: {np.min(sigma1_cgs[mask]):.1f} -- {np.max(sigma1_cgs[mask]):.1f} Ohm^-1 cm^-1")
print(f"sigma1(lowest E): {sigma1_cgs[np.argmin(np.abs(energy_eV - 0.1))]:.1f} Ohm^-1 cm^-1")

# Peak in sigma1
peak_idx = np.argmax(sigma1_cgs)
print(f"Peak sigma1: {sigma1_cgs[peak_idx]:.1f} Ohm^-1 cm^-1 at {energy_eV[peak_idx]:.2f} eV")

# ── Drude model fitting (for metals) ────────────────────────────────
def drude_eps1(omega_eV, omega_p_eV, gamma_eV):
    """Drude model: real part of dielectric function."""
    return 1.0 - omega_p_eV**2 / (omega_eV**2 + gamma_eV**2)

def drude_eps2(omega_eV, omega_p_eV, gamma_eV):
    """Drude model: imaginary part of dielectric function."""
    return omega_p_eV**2 * gamma_eV / (omega_eV * (omega_eV**2 + gamma_eV**2))

def drude_sigma1(omega_eV, omega_p_eV, gamma_eV):
    """Drude model: real part of optical conductivity (in eV units)."""
    # sigma1 = epsilon_0 * omega_p^2 * gamma / (omega^2 + gamma^2)
    # Convert to S/m: multiply by e/hbar for unit conversion
    omega_p_rad = omega_p_eV / HBAR_EV_S
    gamma_rad = gamma_eV / HBAR_EV_S
    omega_rad_local = omega_eV / HBAR_EV_S
    return EPSILON_0 * omega_p_rad**2 * gamma_rad / (omega_rad_local**2 + gamma_rad**2)

# Attempt Drude fit to low-energy sigma1 (only meaningful for metals)
# Fit in the range 0.1 -- 2.0 eV where Drude dominates
fit_mask = (energy_eV > 0.05) & (energy_eV < 2.0) & (sigma1_SI > 0)
is_metal = False

if np.sum(fit_mask) > 10:
    try:
        popt, pcov = curve_fit(
            drude_sigma1,
            energy_eV[fit_mask],
            sigma1_SI[fit_mask],
            p0=[10.0, 0.5],  # initial guess: omega_p=10 eV, gamma=0.5 eV
            bounds=([0.1, 0.001], [30.0, 5.0]),
            maxfev=10000
        )
        omega_p_fit, gamma_fit = popt
        perr = np.sqrt(np.diag(pcov))

        # Check if Drude-like (sigma1 should decrease with omega for Drude)
        if sigma1_SI[fit_mask][0] > sigma1_SI[fit_mask][-1] * 1.5:
            is_metal = True
            tau_fs = HBAR_EV_S / gamma_fit * 1e15  # relaxation time in fs
            sigma_DC = EPSILON_0 * (omega_p_fit / HBAR_EV_S)**2 * HBAR_EV_S / gamma_fit
            sigma_DC_cgs = sigma_DC / 100.0

            print(f"\n=== Drude Fit (Metal) ===")
            print(f"Plasma frequency:  omega_p = {omega_p_fit:.2f} +/- {perr[0]:.2f} eV")
            print(f"                          = {omega_p_fit * 241.799:.0f} THz")
            print(f"Scattering rate:   gamma   = {gamma_fit:.3f} +/- {perr[1]:.3f} eV")
            print(f"Relaxation time:   tau     = {tau_fs:.1f} fs")
            print(f"DC conductivity:   sigma_0 = {sigma_DC_cgs:.0f} Ohm^-1 cm^-1")
            print(f"                          = {sigma_DC:.2e} S/m")
        else:
            print("\nLow-energy sigma1 does not show Drude behavior (likely a semiconductor).")
    except (RuntimeError, ValueError) as e:
        print(f"\nDrude fit failed: {e}")
        print("Material is likely not metallic, or fit range is inappropriate.")

# ── f-sum rule check ─────────────────────────────────────────────────
# Partial f-sum rule:
#   integral_0^E sigma1(omega) d(omega) = pi*e^2*N_eff(E) / (2*m_e*V)
# where N_eff is the effective number of electrons participating in
# transitions up to energy E.
#
# In practice: N_eff(E) = (2*m_e*V) / (pi*e^2) * integral_0^E sigma1 d(omega)
# We compute the integral of sigma1 (in S/m) over omega (in rad/s)

# For the sum rule, integrate sigma1 * d(omega) where omega is in rad/s
# N_eff * e^2 / (2 * m * V) = (1/pi) * integral sigma1 d(omega)
# Or simply: spectral weight W(E) = integral_0^E sigma1(omega) d(omega)

spectral_weight = np.zeros_like(energy_eV)
for i in range(1, len(energy_eV)):
    # Integrate sigma1 in SI units over omega in rad/s
    spectral_weight[i] = trapezoid(
        sigma1_SI[:i+1],
        omega_rad[:i+1]
    )

# Normalized spectral weight (effective electrons per unit cell)
# For a cubic cell with volume V:
# N_eff = 2*m*V / (pi*e^2) * W
# We'll report the raw spectral weight and N_eff if volume is known

print(f"\n=== Spectral Weight (f-sum rule) ===")
total_SW = spectral_weight[-1]
print(f"Total spectral weight (0 to {energy_eV[-1]:.1f} eV): {total_SW:.4e} S rad / m s")

# For reference, for Al (1 atom, 3 valence electrons):
# V_cell = a^3/4 (FCC), a = 4.05 A = 4.05e-10 m
# V = (4.05e-10)^3 / 4 = 1.66e-29 m^3
# N_eff_total should approach 3 for Al (3 valence electrons)
# This is a sanity check for your calculation

# ── Plotting ─────────────────────────────────────────────────────────

# Figure 1: Optical conductivity (real and imaginary parts)
fig1, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

ax1.plot(energy_eV, sigma1_cgs, color="steelblue", linewidth=1.5,
         label=r"$\sigma_1(\omega)$ (interband)")
if is_metal:
    # Overlay Drude fit
    e_fit = np.linspace(0.01, 5.0, 500)
    sigma1_drude = drude_sigma1(e_fit, omega_p_fit, gamma_fit) / 100.0
    ax1.plot(e_fit, sigma1_drude, "r--", linewidth=1.2,
             label=f"Drude fit ($\\omega_p$={omega_p_fit:.1f} eV, $\\gamma$={gamma_fit:.2f} eV)")
ax1.set_ylabel(r"$\sigma_1(\omega)$ ($\Omega^{-1}$ cm$^{-1}$)", fontsize=13)
ax1.set_title("Optical Conductivity (Real Part)", fontsize=14)
ax1.legend(fontsize=10)
ax1.grid(alpha=0.3)
ax1.set_xlim(0, 15)

ax2.plot(energy_eV, sigma2_cgs, color="crimson", linewidth=1.5,
         label=r"$\sigma_2(\omega)$")
ax2.axhline(0, color="gray", linestyle="--", linewidth=0.5)
ax2.set_xlabel("Photon Energy (eV)", fontsize=13)
ax2.set_ylabel(r"$\sigma_2(\omega)$ ($\Omega^{-1}$ cm$^{-1}$)", fontsize=13)
ax2.set_title("Optical Conductivity (Imaginary Part)", fontsize=14)
ax2.legend(fontsize=10)
ax2.grid(alpha=0.3)
ax2.set_xlim(0, 15)

plt.tight_layout()
plt.savefig("optical_conductivity.png", dpi=200, bbox_inches="tight")
print("\nSaved: optical_conductivity.png")

# Figure 2: Spectral weight / f-sum rule
fig2, ax_sw = plt.subplots(figsize=(10, 6))
ax_sw.plot(energy_eV, spectral_weight, color="darkgreen", linewidth=1.5)
ax_sw.set_xlabel("Energy (eV)", fontsize=14)
ax_sw.set_ylabel("Spectral Weight (S rad / m s)", fontsize=14)
ax_sw.set_title("Cumulative Spectral Weight (f-sum rule)", fontsize=15)
ax_sw.grid(alpha=0.3)
ax_sw.set_xlim(0, energy_eV[-1])
plt.tight_layout()
plt.savefig("spectral_weight.png", dpi=200, bbox_inches="tight")
print("Saved: spectral_weight.png")

# Figure 3: sigma1 on log scale (useful for metals)
fig3, ax_log = plt.subplots(figsize=(10, 6))
pos_mask = (energy_eV > 0.01) & (sigma1_cgs > 0)
ax_log.loglog(energy_eV[pos_mask], sigma1_cgs[pos_mask], color="steelblue",
              linewidth=1.5, label=r"$\sigma_1(\omega)$")
if is_metal:
    e_fit = np.logspace(-2, 1, 500)
    sigma1_d = drude_sigma1(e_fit, omega_p_fit, gamma_fit) / 100.0
    ax_log.loglog(e_fit, sigma1_d, "r--", linewidth=1.2,
                  label=f"Drude ($\\omega_p$={omega_p_fit:.1f}, $\\gamma$={gamma_fit:.2f} eV)")
ax_log.set_xlabel("Photon Energy (eV)", fontsize=14)
ax_log.set_ylabel(r"$\sigma_1$ ($\Omega^{-1}$ cm$^{-1}$)", fontsize=14)
ax_log.set_title("Optical Conductivity (log-log)", fontsize=15)
ax_log.legend(fontsize=11)
ax_log.grid(True, which="both", alpha=0.3)
plt.tight_layout()
plt.savefig("optical_conductivity_loglog.png", dpi=200, bbox_inches="tight")
print("Saved: optical_conductivity_loglog.png")
```

### Step 3: Semiconductor Example (Silicon)

For semiconductors, the optical conductivity onset corresponds to the band gap, and there is no Drude contribution.

```python
#!/usr/bin/env python3
"""
Optical conductivity for a semiconductor (Si).
No Drude term; sigma1 is zero below the gap.
"""
import numpy as np
import glob
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.integrate import trapezoid

EPSILON_0 = 8.854187817e-12
HBAR_EV_S = 6.582119569e-16

def load_eps(fn):
    data = []
    with open(fn) as f:
        for line in f:
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
assert rf and imf, "No epsilon data files found!"

dr = load_eps(rf[0])
di = load_eps(imf[0])
energy = dr[:, 0]
eps1 = (dr[:, 1] + dr[:, 2] + dr[:, 3]) / 3.0
eps2 = (di[:, 1] + di[:, 2] + di[:, 3]) / 3.0

# Optical conductivity
omega_rad = energy / HBAR_EV_S
sigma1 = np.zeros_like(energy)
sigma2 = np.zeros_like(energy)
m = energy > 0
sigma1[m] = omega_rad[m] * EPSILON_0 * eps2[m] / 100.0       # Ohm^-1 cm^-1
sigma2[m] = -omega_rad[m] * EPSILON_0 * (eps1[m] - 1.0) / 100.0

# For semiconductors: sigma1 is zero below the gap
gap_onset = energy[np.where(sigma1 > 10)[0][0]] if np.any(sigma1 > 10) else float("nan")
print(f"Optical conductivity onset: {gap_onset:.2f} eV")
print(f"  This corresponds to the direct optical gap.")

# Interband spectral weight
sw = np.zeros_like(energy)
for i in range(1, len(energy)):
    sw[i] = trapezoid(sigma1[:i+1] * 100.0, omega_rad[:i+1])  # back to S/m for integration

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

ax1.plot(energy, sigma1, "steelblue", lw=1.5, label=r"$\sigma_1$")
ax1.fill_between(energy, 0, sigma1, alpha=0.1, color="steelblue")
ax1.set(xlabel="Energy (eV)", ylabel=r"$\sigma_1$ ($\Omega^{-1}$ cm$^{-1}$)",
        title="Optical Conductivity -- Si", xlim=(0, 12))
ax1.legend()
ax1.grid(alpha=0.3)
ax1.axvline(gap_onset, color="green", ls="--", alpha=0.5, label=f"Onset: {gap_onset:.1f} eV")
ax1.legend()

ax2.plot(energy, sw, "darkgreen", lw=1.5)
ax2.set(xlabel="Energy (eV)", ylabel="Spectral Weight",
        title="Cumulative Spectral Weight -- Si", xlim=(0, 15))
ax2.grid(alpha=0.3)

plt.tight_layout()
plt.savefig("optical_conductivity_semiconductor.png", dpi=200, bbox_inches="tight")
print("Saved: optical_conductivity_semiconductor.png")
```

### Complete Single-Script Workflow (Metal: Aluminum)

```python
#!/usr/bin/env python3
"""
Complete optical conductivity workflow for a metal (Aluminum).
SCF -> NSCF -> epsilon.x -> optical conductivity + Drude fit.
"""
import os
import subprocess
import glob
import re
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.integrate import trapezoid

# ── Constants ────────────────────────────────────────────────────────
EPSILON_0 = 8.854187817e-12
HBAR_EV_S = 6.582119569e-16
HBAR_SI = 1.054571817e-34
E_CHARGE = 1.602176634e-19
M_ELECTRON = 9.1093837015e-31

# ── Configuration ────────────────────────────────────────────────────
PSEUDO_DIR = os.path.abspath("./pseudo")
OUTDIR = os.path.abspath("./tmp_sigma")
os.makedirs(OUTDIR, exist_ok=True)
os.makedirs(PSEUDO_DIR, exist_ok=True)
PREFIX = "al"
ECUTWFC = 60.0
ECUTRHO = 240.0
NBND = 40
NPROC = 4
KGRID = "20 20 20"
PP_FILE = "Al.pbe-n-kjpaw_psl.1.0.0.UPF"

# ── Step 1: SCF ──────────────────────────────────────────────────────
scf_in = f"""&CONTROL
    calculation = 'scf', prefix = '{PREFIX}',
    outdir = '{OUTDIR}', pseudo_dir = '{PSEUDO_DIR}',
/
&SYSTEM
    ibrav = 2, celldm(1) = 7.63,
    nat = 1, ntyp = 1,
    ecutwfc = {ECUTWFC}, ecutrho = {ECUTRHO},
    occupations = 'smearing', smearing = 'mp', degauss = 0.02,
/
&ELECTRONS
    conv_thr = 1.0d-10,
/
ATOMIC_SPECIES
  Al 26.9815 {PP_FILE}
ATOMIC_POSITIONS (crystal)
  Al 0.00 0.00 0.00
K_POINTS (automatic)
  {KGRID} 0 0 0
"""
with open(f"{PREFIX}_scf.in", "w") as f:
    f.write(scf_in)

print("[1/4] SCF (metal)...")
r = subprocess.run(["mpirun", "-np", str(NPROC), "pw.x", "-in", f"{PREFIX}_scf.in"],
                    capture_output=True, text=True, timeout=600)
with open(f"{PREFIX}_scf.out", "w") as f:
    f.write(r.stdout)
assert "convergence has been achieved" in r.stdout, "SCF failed!"
print("      Done.")

# Extract Fermi energy
e_fermi = 0.0
for line in r.stdout.split("\n"):
    if "the Fermi energy is" in line:
        m = re.search(r"is\s+([-\d.]+)", line)
        if m:
            e_fermi = float(m.group(1))
print(f"      Fermi energy: {e_fermi:.4f} eV")

# ── Step 2: NSCF ────────────────────────────────────────────────────
nscf_in = f"""&CONTROL
    calculation = 'nscf', prefix = '{PREFIX}',
    outdir = '{OUTDIR}', pseudo_dir = '{PSEUDO_DIR}',
/
&SYSTEM
    ibrav = 2, celldm(1) = 7.63,
    nat = 1, ntyp = 1,
    ecutwfc = {ECUTWFC}, ecutrho = {ECUTRHO},
    occupations = 'smearing', smearing = 'mp', degauss = 0.02,
    nbnd = {NBND}, nosym = .true., noinv = .true.,
/
&ELECTRONS
    conv_thr = 1.0d-10,
/
ATOMIC_SPECIES
  Al 26.9815 {PP_FILE}
ATOMIC_POSITIONS (crystal)
  Al 0.00 0.00 0.00
K_POINTS (automatic)
  {KGRID} 0 0 0
"""
with open(f"{PREFIX}_nscf.in", "w") as f:
    f.write(nscf_in)

print(f"[2/4] NSCF (nbnd={NBND})...")
r = subprocess.run(["mpirun", "-np", str(NPROC), "pw.x", "-in", f"{PREFIX}_nscf.in"],
                    capture_output=True, text=True, timeout=1200)
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
    smeartype = 'gauss', intersmear = 0.15,
    wmin = 0.0, wmax = 30.0, nw = 1500,
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

# ── Step 4: Post-process ────────────────────────────────────────────
print("[4/4] Computing optical conductivity...")

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
assert rf and imf, "No epsilon output files!"

dr = load_eps(rf[0])
di = load_eps(imf[0])
energy = dr[:, 0]
eps1 = (dr[:, 1] + dr[:, 2] + dr[:, 3]) / 3.0
eps2 = (di[:, 1] + di[:, 2] + di[:, 3]) / 3.0

omega_rad = energy / HBAR_EV_S
sigma1_SI = np.zeros_like(energy)
sigma2_SI = np.zeros_like(energy)
m = energy > 0
sigma1_SI[m] = omega_rad[m] * EPSILON_0 * eps2[m]
sigma2_SI[m] = -omega_rad[m] * EPSILON_0 * (eps1[m] - 1.0)
sigma1_cgs = sigma1_SI / 100.0
sigma2_cgs = sigma2_SI / 100.0

# Drude fit
def drude_s1(omega_eV, wp, gam):
    w = omega_eV / HBAR_EV_S
    wp_r = wp / HBAR_EV_S
    gam_r = gam / HBAR_EV_S
    return EPSILON_0 * wp_r**2 * gam_r / (w**2 + gam_r**2)

fit_m = (energy > 0.05) & (energy < 2.0) & (sigma1_SI > 0)
try:
    popt, pcov = curve_fit(drude_s1, energy[fit_m], sigma1_SI[fit_m],
                           p0=[12.0, 0.5], bounds=([1, 0.001], [30, 5]), maxfev=10000)
    wp_fit, gam_fit = popt
    perr = np.sqrt(np.diag(pcov))
    tau_fs = HBAR_EV_S / gam_fit * 1e15
    sigma_dc = EPSILON_0 * (wp_fit / HBAR_EV_S)**2 * HBAR_EV_S / gam_fit / 100.0

    print(f"\n=== Drude Parameters ===")
    print(f"Plasma frequency: {wp_fit:.2f} +/- {perr[0]:.2f} eV ({wp_fit*241.8:.0f} THz)")
    print(f"Scattering rate:  {gam_fit:.3f} +/- {perr[1]:.3f} eV")
    print(f"Relaxation time:  {tau_fs:.1f} fs")
    print(f"DC conductivity:  {sigma_dc:.0f} Ohm^-1 cm^-1")
    has_drude = True
except Exception as e:
    print(f"Drude fit failed: {e}")
    has_drude = False

# Spectral weight
sw = np.zeros_like(energy)
for i in range(1, len(energy)):
    sw[i] = trapezoid(sigma1_SI[:i+1], omega_rad[:i+1])

# Plotting
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# (a) sigma1
axes[0, 0].plot(energy, sigma1_cgs, "steelblue", lw=1.5, label=r"$\sigma_1$ (interband)")
if has_drude:
    ef = np.linspace(0.01, 8, 500)
    axes[0, 0].plot(ef, drude_s1(ef, wp_fit, gam_fit) / 100, "r--", lw=1.2,
                    label=f"Drude ($\\omega_p$={wp_fit:.1f}, $\\gamma$={gam_fit:.2f} eV)")
axes[0, 0].set(xlabel="Energy (eV)", ylabel=r"$\sigma_1$ ($\Omega^{-1}$ cm$^{-1}$)",
               title="(a) Re[$\\sigma(\\omega)$]", xlim=(0, 15))
axes[0, 0].legend(fontsize=9)
axes[0, 0].grid(alpha=0.3)

# (b) sigma2
axes[0, 1].plot(energy, sigma2_cgs, "crimson", lw=1.5, label=r"$\sigma_2$")
axes[0, 1].axhline(0, color="gray", ls="--", lw=0.5)
axes[0, 1].set(xlabel="Energy (eV)", ylabel=r"$\sigma_2$ ($\Omega^{-1}$ cm$^{-1}$)",
               title="(b) Im[$\\sigma(\\omega)$]", xlim=(0, 15))
axes[0, 1].legend(fontsize=9)
axes[0, 1].grid(alpha=0.3)

# (c) log-log sigma1
pos = (energy > 0.01) & (sigma1_cgs > 0)
axes[1, 0].loglog(energy[pos], sigma1_cgs[pos], "steelblue", lw=1.5)
if has_drude:
    ef2 = np.logspace(-2, 1.5, 500)
    axes[1, 0].loglog(ef2, drude_s1(ef2, wp_fit, gam_fit) / 100, "r--", lw=1.2)
axes[1, 0].set(xlabel="Energy (eV)", ylabel=r"$\sigma_1$ ($\Omega^{-1}$ cm$^{-1}$)",
               title="(c) $\\sigma_1$ (log-log)", xlim=(0.01, 30))
axes[1, 0].grid(True, which="both", alpha=0.3)

# (d) spectral weight
axes[1, 1].plot(energy, sw, "darkgreen", lw=1.5)
axes[1, 1].set(xlabel="Energy (eV)", ylabel="Spectral Weight (S rad / m s)",
               title="(d) Cumulative Spectral Weight", xlim=(0, 20))
axes[1, 1].grid(alpha=0.3)

plt.suptitle(f"Optical Conductivity -- Al (PBE, RPA)", fontsize=16, y=1.01)
plt.tight_layout()
plt.savefig("optical_conductivity_Al.png", dpi=200, bbox_inches="tight")
print(f"\nSaved: optical_conductivity_Al.png")
```

---

## Key Parameters

| Parameter | Where | Typical Value | Notes |
|---|---|---|---|
| `K_POINTS` | SCF/NSCF | 20x20x20+ (metals) | Metals need very dense k-grids to resolve the Fermi surface |
| `nbnd` | NSCF | 2--4x occupied | Determines highest interband transition energy |
| `occupations` | SCF/NSCF | 'smearing' (metals) | Use Methfessel-Paxton or cold smearing for metals |
| `degauss` | SCF/NSCF | 0.01--0.03 Ry | Smearing width. Smaller = more accurate but needs denser k-grid |
| `intersmear` | epsilon.x | 0.1--0.3 eV | Broadening. For metals, may need larger broadening at low energy |
| `nosym` / `noinv` | NSCF | `.true.` | Required for epsilon.x |
| Drude fit range | Post-processing | 0.05--2.0 eV | Low-energy range where Drude model dominates (metals only) |

## Interpreting Results

### For Metals
- **sigma1 at low energy**: Should rise steeply following the Drude 1/omega^2 form. Departure from Drude behavior at higher energies indicates interband transitions.
- **Plasma frequency**: Where eps1 crosses zero corresponds to the screened plasma frequency. The unscreened plasma frequency from the Drude fit is larger.
- **DC conductivity**: sigma1(omega->0) = sigma_DC. Compare with experimental resistivity: rho = 1/sigma_DC.
- **Interband onset**: The energy where sigma1 deviates upward from the Drude curve marks the onset of interband transitions.

### For Semiconductors/Insulators
- **sigma1 onset**: Corresponds to the direct optical gap. sigma1 is zero below the gap.
- **No Drude term**: sigma1(0) = 0 for ideal semiconductors (no free carriers).
- **Peak structure**: Peaks in sigma1 correspond to Van Hove singularities (same as eps2 peaks, weighted by omega).

### Sum Rules
- **f-sum rule**: The integral of sigma1 over all frequencies equals pi*n*e^2/(2*m), where n is the total electron density. Partial sum rule: the integral up to energy E gives the effective number of electrons participating in transitions below E.
- **Converging sum rule**: If the spectral weight saturates well before wmax, the calculation has captured all important transitions. If it is still rising at wmax, increase nbnd.

## Common Issues

| Problem | Solution |
|---|---|
| Drude fit fails or gives unphysical parameters | Check that the material is truly metallic. Adjust fit range. Ensure k-grid is dense enough to resolve the Fermi surface |
| sigma1 is negative at some frequencies | Numerical issue. Increase k-points and ecutwfc. Should not happen for physical sigma1 |
| sigma1 does not go to zero below the gap (semiconductor) | Residual from broadening (`intersmear`). Reduce intersmear or accept as artificial |
| Spectral weight does not saturate | Increase nbnd to include more high-energy transitions |
| DC conductivity too high/low vs experiment | epsilon.x computes only interband; Drude fit extrapolation depends on fit quality and k-density |
| Anisotropic conductivity needed | Use the individual xx/yy/zz components from epsilon.x instead of the average |
| Need intraband Drude contribution in epsilon | epsilon.x does not include it; add the Drude term manually in post-processing using the fitted parameters |
| Results differ from experiment significantly | RPA does not include many-body effects. Vertex corrections, electron-phonon coupling, and impurity scattering all affect experimental optical conductivity |
