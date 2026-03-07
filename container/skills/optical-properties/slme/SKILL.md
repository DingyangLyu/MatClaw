# Spectroscopic Limited Maximum Efficiency (SLME)

## When to Use

- You are screening materials for photovoltaic (solar cell) applications.
- You want to compute the theoretical maximum solar cell efficiency beyond the Shockley-Queisser (SQ) limit.
- You need to account for the actual absorption spectrum shape (not just the band gap) in efficiency estimates.
- You want to compare different materials or polymorphs for solar energy harvesting.
- You are optimizing absorber layer thickness for thin-film solar cells.
- You want to quantify losses due to non-ideal absorption onset (indirect gap, weak absorption).

## Method Selection

| Criterion | QE (epsilon.x + post-process) | VASP (LOPTICS + post-process) | VASPKIT (719) |
|---|---|---|---|
| Absorption spectrum | From epsilon.x via RPA | From VASP LOPTICS | From VASP dielectric function |
| Band gap | From NSCF eigenvalues | From EIGENVAL/DOSCAR | VASPKIT auto-detects |
| SLME calculation | Python post-processing | Python post-processing | VASPKIT 719 automated |
| Accuracy | PBE (underestimates gap); apply scissors or use HSE | Same | Same |
| AM1.5G spectrum | User provides or downloads | User provides | Built-in |

**MACE cannot compute SLME.** It requires optical absorption from electronic structure calculations. Use QE or VASP for the absorption spectrum, then compute SLME from the absorption coefficient.

## Prerequisites

- A relaxed crystal structure.
- Computed absorption coefficient alpha(omega) from the `absorption-spectrum` skill, OR a dielectric function from the `dielectric-function` skill.
- The band gap (direct and fundamental) of the material.
- For QE: norm-conserving pseudopotentials, `pw.x`, `epsilon.x`.
- Python packages: `numpy`, `scipy`, `matplotlib`.
- AM1.5G solar spectrum data (ASTM G173-03). The code below includes a built-in parametrization.

---

## Detailed Steps

### Background: SLME Theory

The Spectroscopic Limited Maximum Efficiency (SLME), proposed by Yu and Zunger (PRL 108, 068701, 2012), improves upon the Shockley-Queisser (SQ) limit by incorporating:

1. **The actual absorption spectrum** alpha(E) instead of the step-function approximation.
2. **The distinction between direct and indirect gaps**, which affects absorption onset.
3. **Film thickness L** as a parameter, since thin films may not absorb all above-gap photons.

The absorptivity is:

```
a(E) = 1 - exp(-2 * alpha(E) * L)
```

The short-circuit current:

```
J_sc = e * integral_0^inf a(E) * I_sun(E) / E dE
```

The radiative recombination current (detailed balance):

```
J_0 = e * pi * integral_0^inf a(E) * (2*E^2) / (h^3*c^2) * 1/(exp(E/kT)-1) dE
```

For non-radiative losses (indirect gap materials):

```
J_0_nr = J_0 * exp((E_g_direct - E_g_fundamental) / kT)
```

The efficiency:

```
eta = max_V [ V * (J_sc - J_0_nr * (exp(eV/kT) - 1)) ] / P_sun
```

### Step 1: Compute Absorption Spectrum (QE)

If you already have the absorption spectrum from the `absorption-spectrum` skill, skip to Step 3.

```python
#!/usr/bin/env python3
"""
Step 1: Complete QE workflow to compute the absorption coefficient.
SCF -> NSCF -> epsilon.x -> alpha(E).

Example: CdTe (zincblende, direct gap ~1.5 eV, good solar absorber)
"""
import os
import subprocess
import glob
import numpy as np

PSEUDO_DIR = os.path.abspath("./pseudo")
OUTDIR = os.path.abspath("./tmp_slme")
os.makedirs(OUTDIR, exist_ok=True)
os.makedirs(PSEUDO_DIR, exist_ok=True)
PREFIX = "cdte"
ECUTWFC = 60.0
ECUTRHO = 240.0
NBND = 60
NPROC = 4
KGRID = "16 16 16"

# ── Physical constants ───────────────────────────────────────────────
HBAR_EV_S = 6.582119569e-16
C_CM_S = 2.99792458e10

# ── SCF ──────────────────────────────────────────────────────────────
scf_in = f"""&CONTROL
    calculation = 'scf', prefix = '{PREFIX}',
    outdir = '{OUTDIR}', pseudo_dir = '{PSEUDO_DIR}',
/
&SYSTEM
    ibrav = 2, celldm(1) = 12.2485,
    nat = 2, ntyp = 2,
    ecutwfc = {ECUTWFC}, ecutrho = {ECUTRHO},
    occupations = 'fixed',
/
&ELECTRONS
    conv_thr = 1.0d-10,
/
ATOMIC_SPECIES
  Cd 112.411 Cd_ONCV_PBE-1.2.upf
  Te 127.600 Te_ONCV_PBE-1.2.upf
ATOMIC_POSITIONS (crystal)
  Cd 0.00 0.00 0.00
  Te 0.25 0.25 0.25
K_POINTS (automatic)
  {KGRID} 0 0 0
"""
with open(f"{PREFIX}_scf.in", "w") as f:
    f.write(scf_in)

print("[1/3] SCF...")
r = subprocess.run(["mpirun", "-np", str(NPROC), "pw.x", "-in", f"{PREFIX}_scf.in"],
                    capture_output=True, text=True, timeout=600)
with open(f"{PREFIX}_scf.out", "w") as f:
    f.write(r.stdout)
assert "convergence has been achieved" in r.stdout, "SCF failed!"

# ── NSCF ─────────────────────────────────────────────────────────────
nscf_in = f"""&CONTROL
    calculation = 'nscf', prefix = '{PREFIX}',
    outdir = '{OUTDIR}', pseudo_dir = '{PSEUDO_DIR}',
/
&SYSTEM
    ibrav = 2, celldm(1) = 12.2485,
    nat = 2, ntyp = 2,
    ecutwfc = {ECUTWFC}, ecutrho = {ECUTRHO},
    occupations = 'fixed', nbnd = {NBND},
    nosym = .true., noinv = .true.,
/
&ELECTRONS
    conv_thr = 1.0d-10,
/
ATOMIC_SPECIES
  Cd 112.411 Cd_ONCV_PBE-1.2.upf
  Te 127.600 Te_ONCV_PBE-1.2.upf
ATOMIC_POSITIONS (crystal)
  Cd 0.00 0.00 0.00
  Te 0.25 0.25 0.25
K_POINTS (automatic)
  {KGRID} 0 0 0
"""
with open(f"{PREFIX}_nscf.in", "w") as f:
    f.write(nscf_in)

print("[2/3] NSCF...")
r = subprocess.run(["mpirun", "-np", str(NPROC), "pw.x", "-in", f"{PREFIX}_nscf.in"],
                    capture_output=True, text=True, timeout=1800)
with open(f"{PREFIX}_nscf.out", "w") as f:
    f.write(r.stdout)
assert r.returncode == 0, "NSCF failed!"

# ── epsilon.x ────────────────────────────────────────────────────────
eps_in = f"""&INPUTPP
    prefix = '{PREFIX}', outdir = '{OUTDIR}',
    calculation = 'eps',
/
&ENERGY_GRID
    smeartype = 'gauss', intersmear = 0.1,
    wmin = 0.0, wmax = 10.0, nw = 2000,
    shift = 0.0,
/
"""
with open(f"{PREFIX}_epsilon.in", "w") as f:
    f.write(eps_in)

print("[3/3] epsilon.x...")
r = subprocess.run(["epsilon.x", "-in", f"{PREFIX}_epsilon.in"],
                    capture_output=True, text=True, timeout=600)
with open(f"{PREFIX}_epsilon.out", "w") as f:
    f.write(r.stdout)
assert r.returncode == 0, "epsilon.x failed!"

# ── Extract absorption coefficient ───────────────────────────────────
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

rf = sorted(glob.glob("epsr*.dat"))
imf = sorted(glob.glob("epsi*.dat"))
assert rf and imf, "No epsilon output files found!"

dr = load_eps(rf[0])
di = load_eps(imf[0])
energy_eV = dr[:, 0]
eps1 = (dr[:, 1] + dr[:, 2] + dr[:, 3]) / 3.0
eps2 = (di[:, 1] + di[:, 2] + di[:, 3]) / 3.0

# Compute absorption coefficient
eps_abs = np.sqrt(eps1**2 + eps2**2)
k_ext = np.sqrt(np.maximum((eps_abs - eps1) / 2.0, 0.0))
alpha_cm = np.zeros_like(energy_eV)
mask = energy_eV > 0
alpha_cm[mask] = 2.0 * energy_eV[mask] * k_ext[mask] / (HBAR_EV_S * C_CM_S)

# Save for SLME calculation
np.savetxt("absorption_for_slme.dat",
           np.column_stack([energy_eV, alpha_cm]),
           header="Energy(eV)  alpha(cm^-1)", fmt="%.6f  %.6e")
print("Absorption coefficient saved to absorption_for_slme.dat")
print(f"Energy range: {energy_eV[0]:.2f} -- {energy_eV[-1]:.2f} eV")
print(f"Max alpha: {np.max(alpha_cm):.2e} cm^-1")
```

### Step 2: AM1.5G Solar Spectrum

```python
#!/usr/bin/env python3
"""
Step 2: AM1.5G solar spectrum.
Provides the reference solar spectrum for SLME calculation.

The AM1.5G spectrum (ASTM G173-03) is parametrized as a blackbody
(T=5778 K) modified by atmospheric absorption.
A built-in parametrization is provided; for maximum accuracy, download
the NREL data.
"""
import numpy as np
import os

def am15g_spectrum_blackbody(energy_eV, T_sun=5778.0):
    """
    Approximate AM1.5G solar spectrum using a 5778 K blackbody
    scaled to match the standard AM1.5G total irradiance (1000 W/m^2).

    Parameters:
        energy_eV: array of photon energies in eV
        T_sun: effective solar temperature in K

    Returns:
        I_sun: spectral irradiance in W/(m^2 * eV)
        phi_sun: spectral photon flux in photons/(m^2 * s * eV)
    """
    kB = 8.617333262e-5   # Boltzmann constant in eV/K
    h = 4.135667696e-15   # Planck constant in eV*s
    c = 2.99792458e8      # speed of light in m/s

    # Planck distribution for photon flux
    # phi(E) = (2*pi / (h^3 * c^2)) * E^2 / (exp(E/kT) - 1)
    # This is the blackbody photon flux per unit energy per unit solid angle
    # Multiply by pi * (R_sun/D_sun)^2 for disk-averaged flux at Earth

    # Geometric factor: (R_sun / D_sun)^2 = (6.96e8 / 1.496e11)^2
    geo_factor = (6.96e8 / 1.496e11)**2

    kT = kB * T_sun

    # Avoid overflow for large E/kT
    exp_arg = energy_eV / kT
    exp_arg = np.minimum(exp_arg, 500)  # prevent overflow

    # Photon flux: photons/(m^2 * s * eV)
    phi_sun = (2.0 * np.pi * geo_factor / (h**3 * c**2)) * \
              energy_eV**2 / (np.exp(exp_arg) - 1.0)

    # Spectral irradiance: W/(m^2 * eV)
    I_sun = phi_sun * energy_eV * 1.602176634e-19  # convert eV to J

    return I_sun, phi_sun


def download_am15g_nrel(output_file="am15g_nrel.dat"):
    """
    Download the NREL AM1.5G reference spectrum.
    Returns energy(eV) and spectral irradiance W/(m^2*eV).
    """
    import subprocess

    url = "https://www.nrel.gov/grid/solar-resource/assets/data/astmg173.csv"
    if not os.path.exists("astmg173.csv"):
        print("Downloading NREL AM1.5G spectrum...")
        result = subprocess.run(["wget", "-q", "-O", "astmg173.csv", url],
                                capture_output=True)
        if result.returncode != 0:
            print("Download failed. Using blackbody approximation.")
            return None

    # Parse NREL data
    # Format: wavelength(nm), ETR(W/m^2/nm), Global Tilt(W/m^2/nm), Direct+Circumsolar
    data = []
    with open("astmg173.csv") as f:
        for line in f:
            if line.startswith("#") or not line.strip():
                continue
            parts = line.strip().split(",")
            if len(parts) >= 3:
                try:
                    wl_nm = float(parts[0])
                    irr_wm2nm = float(parts[2])  # Global tilt = AM1.5G
                    if wl_nm > 0:
                        data.append([wl_nm, irr_wm2nm])
                except ValueError:
                    continue

    if not data:
        return None

    data = np.array(data)
    wl_nm = data[:, 0]
    irr_wm2nm = data[:, 1]

    # Convert to energy: E(eV) = 1239.84 / lambda(nm)
    energy_eV = 1239.84 / wl_nm
    # Convert irradiance: W/(m^2*nm) -> W/(m^2*eV)
    # dE/dlambda = -1239.84 / lambda^2, so |dE| = 1239.84/lambda^2 * |dlambda|
    # I(E) = I(lambda) * |dlambda/dE| = I(lambda) * lambda^2 / 1239.84
    irr_wm2eV = irr_wm2nm * wl_nm**2 / 1239.84

    # Sort by energy (ascending)
    sort_idx = np.argsort(energy_eV)
    energy_eV = energy_eV[sort_idx]
    irr_wm2eV = irr_wm2eV[sort_idx]

    np.savetxt(output_file,
               np.column_stack([energy_eV, irr_wm2eV]),
               header="Energy(eV)  Irradiance(W/m^2/eV)", fmt="%.6f  %.6e")
    print(f"AM1.5G spectrum saved to {output_file}")
    return energy_eV, irr_wm2eV


# Generate spectrum
energy_grid = np.linspace(0.3, 6.0, 5000)
I_sun, phi_sun = am15g_spectrum_blackbody(energy_grid)
total_power = np.trapz(I_sun, energy_grid)
print(f"Total solar power (blackbody approx): {total_power:.1f} W/m^2")
print(f"  (Standard AM1.5G: 1000 W/m^2)")

# Scale to match 1000 W/m^2
scale = 1000.0 / total_power
I_sun *= scale
phi_sun *= scale
print(f"After scaling: {np.trapz(I_sun, energy_grid):.1f} W/m^2")

np.savetxt("am15g_spectrum.dat",
           np.column_stack([energy_grid, I_sun, phi_sun]),
           header="Energy(eV)  I_sun(W/m^2/eV)  phi_sun(photons/m^2/s/eV)",
           fmt="%.6f  %.6e  %.6e")
print("Saved: am15g_spectrum.dat")
```

### Step 3: SLME Calculation

```python
#!/usr/bin/env python3
"""
Step 3: Compute the Spectroscopic Limited Maximum Efficiency (SLME).

This implements the Yu-Zunger method (PRL 108, 068701, 2012):
  1. Load absorption coefficient alpha(E) and solar spectrum I_sun(E).
  2. Compute absorptivity a(E,L) = 1 - exp(-2*alpha*L) for thickness L.
  3. Compute J_sc from absorbed solar photon flux.
  4. Compute J_0 from detailed balance (blackbody emission at 300K).
  5. Maximize V*(J_sc - J_0*(exp(eV/kT)-1)) over voltage V.
  6. Compute efficiency eta = P_max / P_sun.
  7. Repeat for multiple thicknesses.
"""
import numpy as np
from scipy.optimize import minimize_scalar
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ── Physical constants ───────────────────────────────────────────────
kB_eV = 8.617333262e-5     # Boltzmann constant in eV/K
h_eVs = 4.135667696e-15    # Planck constant in eV*s
c_ms = 2.99792458e8        # speed of light in m/s
e_C = 1.602176634e-19      # elementary charge in C
T = 300.0                  # temperature in K
kT = kB_eV * T             # ~0.02585 eV at 300K
P_sun = 1000.0             # AM1.5G total irradiance W/m^2

# ── Load absorption spectrum ────────────────────────────────────────
def load_absorption(filename="absorption_for_slme.dat"):
    """Load absorption coefficient: columns are energy(eV), alpha(cm^-1)."""
    data = np.loadtxt(filename, comments="#")
    return data[:, 0], data[:, 1]  # energy_eV, alpha_cm


def am15g_blackbody(energy_eV, T_sun=5778.0):
    """AM1.5G approximation as scaled blackbody."""
    geo = (6.96e8 / 1.496e11)**2
    kT_sun = kB_eV * T_sun
    exp_arg = np.minimum(energy_eV / kT_sun, 500)
    phi = (2.0 * np.pi * geo / (h_eVs**3 * c_ms**2)) * \
          energy_eV**2 / (np.exp(exp_arg) - 1.0)
    I = phi * energy_eV * e_C  # W/(m^2*eV)
    # Scale to 1000 W/m^2
    total = np.trapz(I, energy_eV)
    if total > 0:
        scale = P_sun / total
        I *= scale
        phi *= scale
    return I, phi


def blackbody_emission(energy_eV, T=300.0):
    """
    Planck blackbody photon flux at temperature T.
    Returns photon flux in photons/(m^2 * s * eV * sr).
    Multiply by pi for hemispherical emission.
    """
    exp_arg = np.minimum(energy_eV / (kB_eV * T), 500)
    phi = (2.0 / (h_eVs**3 * c_ms**2)) * \
          energy_eV**2 / (np.exp(exp_arg) - 1.0)
    return phi


def compute_slme(energy_eV, alpha_cm, Eg_direct, Eg_fundamental=None,
                 thickness_um=1.0, T=300.0):
    """
    Compute SLME for a given absorber.

    Parameters:
        energy_eV: photon energy array (eV)
        alpha_cm: absorption coefficient array (cm^-1)
        Eg_direct: direct band gap (eV)
        Eg_fundamental: fundamental (possibly indirect) band gap (eV)
                        If None, assumes direct gap material.
        thickness_um: absorber layer thickness (micrometers)
        T: temperature (K)

    Returns:
        eta: maximum efficiency (fraction, not percent)
        V_opt: optimal operating voltage (V)
        J_sc: short-circuit current density (A/m^2)
        J_0: reverse saturation current density (A/m^2)
        details: dictionary with intermediate results
    """
    if Eg_fundamental is None:
        Eg_fundamental = Eg_direct

    kT_val = kB_eV * T
    L_cm = thickness_um * 1e-4  # um to cm

    # Absorptivity (double-pass for back-reflected thin film)
    absorptivity = 1.0 - np.exp(-2.0 * alpha_cm * L_cm)

    # Solar photon flux
    I_sun, phi_sun = am15g_blackbody(energy_eV)

    # Short-circuit current: J_sc = e * integral[a(E) * phi_sun(E) dE]
    # Only count photons above the fundamental gap
    integrand_sc = absorptivity * phi_sun
    integrand_sc[energy_eV < Eg_fundamental] = 0
    J_sc = e_C * np.trapz(integrand_sc, energy_eV)  # A/m^2

    # Reverse saturation current (radiative recombination)
    # J_0_rad = e * pi * integral[a(E) * phi_bb(E, T) dE]
    phi_bb = blackbody_emission(energy_eV, T)
    integrand_j0 = absorptivity * phi_bb
    integrand_j0[energy_eV < Eg_fundamental] = 0
    J_0_rad = e_C * np.pi * np.trapz(integrand_j0, energy_eV)

    # Non-radiative loss factor for indirect gap materials
    # f_nr = exp((Eg_direct - Eg_fundamental) / kT)
    delta_Eg = Eg_direct - Eg_fundamental
    if delta_Eg > 0:
        f_nr = np.exp(delta_Eg / kT_val)
    else:
        f_nr = 1.0

    J_0 = J_0_rad * f_nr

    # Find optimal voltage by maximizing power P(V) = V * [J_sc - J_0*(exp(eV/kT)-1)]
    def neg_power(V):
        if V <= 0:
            return 0.0
        J = J_sc - J_0 * (np.exp(V / kT_val) - 1.0)
        P = V * J
        return -P if J > 0 else 0.0

    # Search for V_opt in [0, Eg_direct]
    result = minimize_scalar(neg_power, bounds=(0.01, Eg_direct),
                             method="bounded")
    V_opt = result.x
    P_max = -result.fun  # W/m^2

    eta = P_max / P_sun

    # Shockley-Queisser for comparison (step function absorption)
    absorptivity_sq = np.where(energy_eV >= Eg_direct, 1.0, 0.0)
    J_sc_sq = e_C * np.trapz(absorptivity_sq * phi_sun, energy_eV)
    J_0_sq = e_C * np.pi * np.trapz(absorptivity_sq * phi_bb, energy_eV)

    def neg_power_sq(V):
        if V <= 0:
            return 0.0
        J = J_sc_sq - J_0_sq * (np.exp(V / kT_val) - 1.0)
        P = V * J
        return -P if J > 0 else 0.0

    result_sq = minimize_scalar(neg_power_sq, bounds=(0.01, Eg_direct),
                                method="bounded")
    eta_sq = -result_sq.fun / P_sun

    details = {
        "J_sc_A_m2": J_sc,
        "J_sc_mA_cm2": J_sc * 0.1,  # A/m^2 to mA/cm^2
        "J_0_A_m2": J_0,
        "J_0_rad_A_m2": J_0_rad,
        "V_opt_V": V_opt,
        "P_max_W_m2": P_max,
        "eta_SLME": eta,
        "eta_SQ": eta_sq,
        "f_nr": f_nr,
        "Eg_direct": Eg_direct,
        "Eg_fundamental": Eg_fundamental,
        "thickness_um": thickness_um,
    }

    return eta, V_opt, J_sc, J_0, details


# ── Load data ────────────────────────────────────────────────────────
try:
    energy_eV, alpha_cm = load_absorption("absorption_for_slme.dat")
    print(f"Loaded absorption: {len(energy_eV)} points, "
          f"E = {energy_eV[0]:.2f}--{energy_eV[-1]:.2f} eV")
except FileNotFoundError:
    # Demo with a model absorption coefficient
    print("No absorption file found. Using model CdTe absorption.")
    energy_eV = np.linspace(0.1, 6.0, 5000)
    Eg_model = 1.5  # CdTe direct gap
    # Model: alpha = A * sqrt(E - Eg) for E > Eg
    alpha_cm = np.where(energy_eV > Eg_model,
                        1.0e5 * np.sqrt(energy_eV - Eg_model),
                        0.0)

# ── Material parameters ──────────────────────────────────────────────
# For CdTe: direct gap ~ 1.5 eV (PBE underestimates; expt ~1.5 eV)
# Adjust these for your material:
Eg_direct = 1.5          # Direct band gap in eV
Eg_fundamental = 1.5     # Fundamental gap (= direct for CdTe)
# For indirect gap materials like Si: Eg_fundamental = 1.1, Eg_direct = 3.4

# ── Single thickness SLME ───────────────────────────────────────────
L = 1.0  # um
eta, V_opt, J_sc, J_0, details = compute_slme(
    energy_eV, alpha_cm, Eg_direct, Eg_fundamental, thickness_um=L
)

print(f"\n{'='*60}")
print(f"SLME Results: Eg_direct={Eg_direct} eV, L={L} um")
print(f"{'='*60}")
print(f"  SLME:               {eta*100:.2f}%")
print(f"  SQ limit:           {details['eta_SQ']*100:.2f}%")
print(f"  SLME / SQ:          {eta/details['eta_SQ']*100:.1f}%")
print(f"  J_sc:               {details['J_sc_mA_cm2']:.2f} mA/cm^2")
print(f"  J_0 (radiative):    {details['J_0_rad_A_m2']:.4e} A/m^2")
print(f"  J_0 (total):        {details['J_0_A_m2']:.4e} A/m^2")
print(f"  V_opt:              {V_opt:.4f} V")
print(f"  P_max:              {details['P_max_W_m2']:.2f} W/m^2")
print(f"  Non-radiative factor: {details['f_nr']:.4f}")

# ── SLME vs thickness ────────────────────────────────────────────────
thicknesses = np.logspace(-2, 2, 50)  # 0.01 um to 100 um
slme_values = []
jsc_values = []
vopt_values = []

for L_val in thicknesses:
    eta_val, V_val, Jsc_val, _, _ = compute_slme(
        energy_eV, alpha_cm, Eg_direct, Eg_fundamental, thickness_um=L_val
    )
    slme_values.append(eta_val * 100)
    jsc_values.append(Jsc_val * 0.1)  # mA/cm^2
    vopt_values.append(V_val)

slme_values = np.array(slme_values)
jsc_values = np.array(jsc_values)
vopt_values = np.array(vopt_values)

# ── SLME vs band gap (SQ-like plot) ──────────────────────────────────
# Compute SLME for different hypothetical direct gaps
# at a fixed thickness, using step-function absorption as SQ reference
gap_range = np.linspace(0.5, 3.0, 100)
eta_sq_curve = []
eta_slme_1um = []

for Eg in gap_range:
    # SQ limit (step function)
    alpha_sq = np.where(energy_eV >= Eg, 1e6, 0.0)  # very large alpha = perfect absorber
    eta_sq_val, _, _, _, _ = compute_slme(
        energy_eV, alpha_sq, Eg, Eg, thickness_um=1000.0  # thick = SQ limit
    )
    eta_sq_curve.append(eta_sq_val * 100)

    # SLME at 1 um with model absorption
    alpha_model = np.where(energy_eV > Eg,
                           1.0e5 * np.sqrt(energy_eV - Eg), 0.0)
    eta_slme_val, _, _, _, _ = compute_slme(
        energy_eV, alpha_model, Eg, Eg, thickness_um=1.0
    )
    eta_slme_1um.append(eta_slme_val * 100)

# ── Plotting ─────────────────────────────────────────────────────────

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# (a) SLME vs thickness
ax = axes[0, 0]
ax.semilogx(thicknesses, slme_values, "b-", linewidth=2)
ax.axhline(details['eta_SQ'] * 100, color="r", linestyle="--",
           label=f"SQ limit ({details['eta_SQ']*100:.1f}%)")
ax.set_xlabel("Absorber Thickness (um)", fontsize=12)
ax.set_ylabel("SLME (%)", fontsize=12)
ax.set_title(f"(a) SLME vs Thickness (Eg={Eg_direct} eV)", fontsize=13)
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3)

# (b) J_sc and V_opt vs thickness
ax = axes[0, 1]
ax.semilogx(thicknesses, jsc_values, "b-", linewidth=1.5, label="J_sc")
ax.set_xlabel("Thickness (um)")
ax.set_ylabel("J_sc (mA/cm^2)", color="b")
ax.tick_params(axis="y", labelcolor="b")
ax2 = ax.twinx()
ax2.semilogx(thicknesses, vopt_values, "r--", linewidth=1.5, label="V_opt")
ax2.set_ylabel("V_opt (V)", color="r")
ax2.tick_params(axis="y", labelcolor="r")
ax.set_title("(b) J_sc and V_opt vs Thickness", fontsize=13)
ax.grid(True, alpha=0.3)

# (c) Absorptivity at different thicknesses
ax = axes[1, 0]
for L_plot in [0.1, 0.5, 1.0, 5.0, 10.0]:
    absorptivity = 1.0 - np.exp(-2.0 * alpha_cm * L_plot * 1e-4)
    ax.plot(energy_eV, absorptivity, linewidth=1.0, label=f"L={L_plot} um")
ax.set_xlabel("Energy (eV)")
ax.set_ylabel("Absorptivity a(E)")
ax.set_title("(c) Absorptivity vs Energy", fontsize=13)
ax.set_xlim(0, 5)
ax.legend(fontsize=9)
ax.grid(alpha=0.3)

# (d) SLME vs band gap (SQ comparison)
ax = axes[1, 1]
ax.plot(gap_range, eta_sq_curve, "r-", linewidth=1.5, label="SQ limit (thick)")
ax.plot(gap_range, eta_slme_1um, "b--", linewidth=1.5, label="SLME (1 um, model)")
ax.axvline(Eg_direct, color="gray", linestyle=":", alpha=0.5,
           label=f"This material ({Eg_direct} eV)")
ax.set_xlabel("Band Gap (eV)")
ax.set_ylabel("Efficiency (%)")
ax.set_title("(d) Efficiency vs Band Gap", fontsize=13)
ax.legend(fontsize=10)
ax.set_xlim(0.5, 3.0)
ax.set_ylim(0, 35)
ax.grid(alpha=0.3)

plt.suptitle("Spectroscopic Limited Maximum Efficiency (SLME)", fontsize=15, y=1.01)
plt.tight_layout()
plt.savefig("slme_analysis.png", dpi=200, bbox_inches="tight")
print("\nSaved: slme_analysis.png")

# ── Save results ─────────────────────────────────────────────────────
import json
results = {
    "material_example": "CdTe",
    "Eg_direct_eV": Eg_direct,
    "Eg_fundamental_eV": Eg_fundamental,
    "results_at_1um": details,
    "thickness_um": thicknesses.tolist(),
    "SLME_percent": slme_values.tolist(),
    "Jsc_mA_cm2": jsc_values.tolist(),
}
with open("slme_results.json", "w") as f:
    json.dump(results, f, indent=2, default=str)
print("Saved: slme_results.json")
```

### Step 4: VASP Workflow (VASPKIT 719 Equivalent)

```python
#!/usr/bin/env python3
"""
Step 4: SLME from VASP output, equivalent to VASPKIT option 719.

When VASP is available, use the dielectric function from OUTCAR
or the WAVEDER-based optical properties.

NOTE: Requires VASP output files. This script will work when VASP
becomes available in the container.
"""
import numpy as np
import os
import re
from scipy.optimize import minimize_scalar
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Physical constants (same as above)
kB_eV = 8.617333262e-5
h_eVs = 4.135667696e-15
c_ms = 2.99792458e8
e_C = 1.602176634e-19
HBAR_EV_S = 6.582119569e-16
C_CM_S = 2.99792458e10
T = 300.0
kT = kB_eV * T
P_sun = 1000.0


def parse_vasp_dielectric(outcar_file="OUTCAR"):
    """
    Parse the frequency-dependent dielectric function from VASP OUTCAR.
    VASP prints eps1 and eps2 after 'REAL DIELECTRIC FUNCTION' and
    'IMAGINARY DIELECTRIC FUNCTION' headers.

    Returns: energy_eV, eps1_avg, eps2_avg
    """
    if not os.path.exists(outcar_file):
        raise FileNotFoundError(f"{outcar_file} not found.")

    energy_r, eps1_data = [], []
    energy_i, eps2_data = [], []
    reading_real = False
    reading_imag = False

    with open(outcar_file) as f:
        for line in f:
            if "REAL DIELECTRIC FUNCTION" in line:
                reading_real = True
                reading_imag = False
                continue
            elif "IMAGINARY DIELECTRIC FUNCTION" in line:
                reading_real = False
                reading_imag = True
                continue

            if reading_real or reading_imag:
                parts = line.split()
                if len(parts) >= 7:
                    try:
                        e = float(parts[0])
                        xx = float(parts[1])
                        yy = float(parts[2])
                        zz = float(parts[3])
                        if reading_real:
                            energy_r.append(e)
                            eps1_data.append((xx + yy + zz) / 3.0)
                        else:
                            energy_i.append(e)
                            eps2_data.append((xx + yy + zz) / 3.0)
                    except ValueError:
                        reading_real = False
                        reading_imag = False

    return (np.array(energy_r), np.array(eps1_data),
            np.array(energy_i), np.array(eps2_data))


def parse_vasp_bandgap(outcar_file="OUTCAR"):
    """Extract band gap from VASP OUTCAR."""
    Eg = None
    with open(outcar_file) as f:
        for line in f:
            if "direct band gap" in line.lower():
                nums = re.findall(r"[\d.]+", line)
                if nums:
                    Eg = float(nums[-1])
    return Eg


def vasp_slme_workflow():
    """Complete VASP -> SLME workflow."""
    # Parse dielectric function
    energy_r, eps1, energy_i, eps2 = parse_vasp_dielectric("OUTCAR")

    # Compute absorption coefficient
    energy_eV = energy_i
    eps_abs = np.sqrt(eps1[:len(eps2)]**2 + eps2**2)
    k_ext = np.sqrt(np.maximum((eps_abs - eps1[:len(eps2)]) / 2.0, 0.0))
    alpha_cm = np.zeros_like(energy_eV)
    mask = energy_eV > 0
    alpha_cm[mask] = 2.0 * energy_eV[mask] * k_ext[mask] / (HBAR_EV_S * C_CM_S)

    # Get band gap
    Eg = parse_vasp_bandgap("OUTCAR")
    if Eg is None:
        # Estimate from absorption onset
        onset_idx = np.where(alpha_cm > 100)[0]
        Eg = energy_eV[onset_idx[0]] if len(onset_idx) > 0 else 1.0
        print(f"Estimated band gap from absorption onset: {Eg:.2f} eV")
    else:
        print(f"Band gap from OUTCAR: {Eg:.2f} eV")

    # Compute SLME vs thickness
    # (Uses the same compute_slme function from Step 3)
    print(f"\nSLME calculation with Eg = {Eg:.2f} eV")
    print(f"{'Thickness (um)':>15s}  {'SLME (%)':>10s}  {'J_sc (mA/cm2)':>15s}")
    print("-" * 45)
    for L in [0.1, 0.5, 1.0, 2.0, 5.0, 10.0]:
        # Inline SLME (same algorithm as compute_slme above)
        absorptivity = 1.0 - np.exp(-2.0 * alpha_cm * L * 1e-4)

        geo = (6.96e8 / 1.496e11)**2
        kT_sun = kB_eV * 5778
        exp_sun = np.minimum(energy_eV / kT_sun, 500)
        phi_sun = (2*np.pi*geo / (h_eVs**3 * c_ms**2)) * energy_eV**2 / (np.exp(exp_sun) - 1)
        I_sun = phi_sun * energy_eV * e_C
        total = np.trapz(I_sun, energy_eV)
        if total > 0:
            phi_sun *= P_sun / total

        J_sc = e_C * np.trapz(absorptivity * phi_sun * (energy_eV >= Eg), energy_eV)

        exp_bb = np.minimum(energy_eV / kT, 500)
        phi_bb = (2 / (h_eVs**3 * c_ms**2)) * energy_eV**2 / (np.exp(exp_bb) - 1)
        J_0 = e_C * np.pi * np.trapz(absorptivity * phi_bb * (energy_eV >= Eg), energy_eV)

        def neg_p(V):
            J = J_sc - J_0 * (np.exp(V / kT) - 1)
            return -(V * J) if J > 0 else 0.0

        res = minimize_scalar(neg_p, bounds=(0.01, Eg), method="bounded")
        eta = -res.fun / P_sun * 100

        print(f"{L:15.1f}  {eta:10.2f}  {J_sc*0.1:15.2f}")


if __name__ == "__main__":
    try:
        vasp_slme_workflow()
    except FileNotFoundError as e:
        print(f"VASP files not available: {e}")
        print("Use the QE workflow (Steps 1-3) instead.")
```

---

## Key Parameters

| Parameter | Where | Typical Value | Impact |
|---|---|---|---|
| `Eg_direct` | SLME input | Material-specific | The direct gap determines the SQ baseline. PBE underestimates; use HSE or scissored values |
| `Eg_fundamental` | SLME input | Material-specific | If indirect < direct, non-radiative losses increase J_0 exponentially |
| `thickness_um` | SLME input | 0.1--100 um | Thin films: lower SLME but less material. Thick: approaches SQ limit |
| `T` | SLME calculation | 300 K | Cell operating temperature. Higher T reduces V_oc and efficiency |
| `alpha(E)` | From DFT | Material-specific | The absorption spectrum shape is the key SLME advantage over SQ |
| `K_POINTS` | QE SCF/NSCF | 16x16x16+ | Smooth absorption spectrum needed for accurate SLME integration |
| `intersmear` | epsilon.x | 0.05--0.1 eV | Moderate broadening. Too much smears the absorption onset |
| `shift` (scissors) | epsilon.x | 0--1.5 eV | Corrects PBE gap underestimation. Critical for SLME accuracy |

### Scissors Correction

Since PBE systematically underestimates band gaps, applying a scissors correction is essential for quantitative SLME predictions:

```python
# Method 1: epsilon.x shift parameter
shift = Eg_experimental - Eg_PBE  # e.g., 0.9 eV for Si

# Method 2: Post-processing shift of the absorption spectrum
# Shift the energy axis of alpha(E) by the scissors correction
energy_shifted = energy_eV + scissors_shift
# Then interpolate alpha back to the original grid
```

## Interpreting Results

- **SLME vs SQ limit**: The SLME is always less than or equal to the SQ limit. The ratio SLME/SQ quantifies how close the real material approaches the thermodynamic limit. Values > 80% indicate excellent absorbers.

- **Optimal band gap**: The SQ limit peaks at Eg ~ 1.34 eV (33.7%). Good solar cell materials have direct gaps of 1.0--1.7 eV. SLME penalizes indirect-gap or weakly absorbing materials.

- **Thickness dependence**: SLME increases monotonically with thickness (more absorption). For strong absorbers (CdTe, GaAs, perovskites), SLME saturates by ~1 um. For weak absorbers (Si, due to indirect gap), much thicker films are needed.

- **J_sc values**: Typical maximum J_sc for single-junction cells is ~45 mA/cm^2 (all above-gap photons absorbed). Real cells achieve 30--42 mA/cm^2.

- **V_opt**: The optimal voltage is typically 70--90% of the band gap. Lower V_opt indicates higher recombination losses.

- **Non-radiative factor (f_nr)**: For direct-gap materials, f_nr = 1. For Si (indirect gap, Eg_direct - Eg_fundamental ~ 2.3 eV), f_nr ~ 10^39, which severely limits efficiency at small thicknesses.

- **Material comparison**: SLME enables ranking materials by their photovoltaic potential, accounting for absorption strength. It is more physically meaningful than comparing band gaps alone.

## Common Issues

| Problem | Solution |
|---|---|
| SLME is much lower than SQ limit | Check absorption coefficient magnitude. Weak absorption (alpha < 10^4 cm^-1 near gap) gives low SLME at thin films |
| SLME exceeds SQ limit | Bug in the calculation. Check that absorptivity <= 1 and J_0 is computed correctly |
| Band gap is wrong (PBE underestimate) | Apply scissors correction. Use HSE or GW for the gap value, PBE for the absorption shape |
| Negative SLME or negative J_sc | Check that alpha(E) is non-negative. Verify energy units are consistent (eV everywhere) |
| SLME does not saturate with thickness | Material has very weak absorption. Increase thickness range or check alpha values |
| AM1.5G total power is not 1000 W/m^2 | The blackbody approximation may need scaling. Use the NREL reference data for accuracy |
| Indirect-gap material shows very low SLME | Expected: f_nr penalizes indirect gaps heavily. Report both SLME and SQ for comparison |
| Numerical oscillations in SLME vs thickness | Smooth the absorption coefficient before SLME integration. Use more energy grid points |
