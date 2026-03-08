# Quasi-Harmonic Debye Model

## When to Use

- You need finite-temperature thermodynamic properties (Gibbs free energy, thermal expansion, heat capacity Cp, bulk modulus vs T) but want to avoid expensive phonon calculations at every volume.
- You have an E(V) curve from static calculations and want to extract temperature-dependent properties via the Debye approximation.
- Screening many materials or compositions quickly for thermal properties before committing to full phonon QHA.
- Estimating the Debye temperature and its volume dependence from equation-of-state data.
- Computing the thermal equation of state P(V,T) or G(T,P) at arbitrary pressures.

## Method Selection

```
Need finite-T thermodynamic properties?
  YES --> Is phonon accuracy required (anisotropic, soft modes, negative thermal expansion)?
    YES --> Use full phonon QHA (gruneisen-qha skill with phonopy)
    NO  --> Quasi-Harmonic Debye (this skill) -- much faster
      Is MACE accurate for your system's E(V)?
        YES --> Method A: ASE + MACE (fast, minutes)
        NO  --> Method B: QE pw.x SCF at each volume (hours)
        Alternative --> Method C: LAMMPS with classical potential (seconds)
```

The Debye model replaces the full phonon density of states with a single parameter -- the Debye temperature Theta_D -- derived from the bulk modulus and average atomic mass. The vibrational free energy is then:

```
F_vib(V,T) = n*k_B*T * [ (9/8)*(Theta_D/T) + 3*ln(1 - exp(-Theta_D/T)) - D(Theta_D/T) ]
```

where D(x) is the Debye function and n is the number of atoms. The Debye temperature depends on volume through B(V) from the equation of state:

```
Theta_D(V) = (hbar/k_B) * (6*pi^2*V^(1/2)*n)^(1/3) * sqrt(B_S / M)
```

with B_S the adiabatic bulk modulus and M the average atomic mass. This captures the essential physics of thermal expansion (the Gruneisen mechanism) without computing any phonon dispersions.

**Advantages over full phonon QHA:**
- Only needs E(V) -- no force-constant matrices, no supercells, no displacements.
- 10-100x faster: one static calculation per volume point vs. dozens of displacement calculations.
- Sufficient for cubic metals, simple semiconductors, and alloy screening.

**Limitations:**
- Single Debye cutoff misses optical/acoustic mode structure.
- Cannot capture negative thermal expansion or van Hove singularities.
- Less accurate for layered, molecular, or low-symmetry crystals with complex phonon DOS.
- Assumes isotropic elastic response.

## Prerequisites

```bash
pip install scipy
```

Pre-installed: `ase`, `mace-torch`, `numpy`, `matplotlib`.

No phonopy dependency. No pyiron dependency. All scripts are fully standalone.

## Detailed Steps

### Method A: ASE + MACE (Complete Standalone Script)

The workflow:
1. Relax the structure at ground state to get equilibrium volume V0
2. Compute E(V) at 11-15 volumes spanning +/- 5-10% around V0
3. Fit 3rd-order Birch-Murnaghan EOS to get E0, V0, B0, B0'
4. Derive the Debye temperature Theta_D(V) from B(V) and atomic mass
5. At each (V, T), compute vibrational Helmholtz free energy F_vib(V,T)
6. Minimize G(V,T,P) = E(V) + F_vib(V,T) + PV over V at each T to get equilibrium V(T)
7. Extract G(T,P), alpha(T), B(T), Cp(T)

```python
#!/usr/bin/env python3
"""
Quasi-Harmonic Debye Model: finite-temperature thermodynamics from E(V).
Uses ASE + MACE for the volume scan. No phonon calculations required.

Computes: Gibbs free energy G(T,P), thermal expansion alpha(T),
          bulk modulus B(T), heat capacity Cp(T), Debye temperature Theta_D(V).

Complete standalone script -- no pyiron, no phonopy.
"""

import os
import numpy as np
from scipy.optimize import minimize_scalar, curve_fit
from scipy.integrate import quad

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from ase.io import read, write
from ase.optimize import LBFGS
from ase.constraints import ExpCellFilter

from mace.calculators import mace_mp

import warnings
warnings.filterwarnings("ignore")

# ============================================================
# 1. CONFIGURATION
# ============================================================

STRUCTURE_FILE = "structure.cif"       # Input structure (CIF, POSCAR, etc.)
MACE_MODEL = "medium"                  # "small", "medium", "large"
DEVICE = "cpu"                         # "cpu" or "cuda"

# Volume scan: number of points and fractional range around V0
N_VOL_POINTS = 11                      # Odd number recommended (includes V0)
VOL_RANGE = 0.06                       # +/- 6% volume variation

# Temperature and pressure range
T_MIN = 0                              # K
T_MAX = 1500                           # K
T_STEP = 5                             # K
PRESSURE = 0.0                         # GPa (0 = zero pressure)

# Relaxation
FMAX = 1e-4                            # eV/A

# Output directory
WORK_DIR = "/tmp/qh_debye"
os.makedirs(WORK_DIR, exist_ok=True)

# Physical constants (SI)
kB_SI = 1.380649e-23                   # J/K
hbar_SI = 1.054571817e-34              # J*s
NA = 6.02214076e23                     # 1/mol
eV_to_J = 1.602176634e-19             # J/eV
A3_to_m3 = 1e-30                       # m^3/A^3
GPa_to_Pa = 1e9                        # Pa/GPa
amu_to_kg = 1.66053906660e-27          # kg/amu

# Boltzmann constant in eV/K
kB_eV = 8.617333262e-5

# ============================================================
# 2. RELAX GROUND STATE
# ============================================================

calc = mace_mp(model=MACE_MODEL, device=DEVICE, default_dtype="float64")

atoms_orig = read(STRUCTURE_FILE)
atoms_orig.calc = calc

print("=" * 60)
print("Quasi-Harmonic Debye Model")
print("=" * 60)
print(f"\nFormula: {atoms_orig.get_chemical_formula()}")
print(f"Number of atoms: {len(atoms_orig)}")

ecf = ExpCellFilter(atoms_orig, hydrostatic_strain=False)
opt = LBFGS(ecf, logfile=os.path.join(WORK_DIR, "relax.log"))
opt.run(fmax=FMAX, steps=500)

V0_init = atoms_orig.get_volume()
E0_init = atoms_orig.get_potential_energy()
print(f"Relaxed volume: {V0_init:.4f} A^3")
print(f"Relaxed energy: {E0_init:.6f} eV")
write(os.path.join(WORK_DIR, "relaxed.cif"), atoms_orig)

# ============================================================
# 3. VOLUME SCAN: E(V)
# ============================================================

print(f"\n--- Volume Scan: {N_VOL_POINTS} points, +/- {VOL_RANGE*100:.1f}% ---")

vol_fractions = np.linspace(1.0 - VOL_RANGE, 1.0 + VOL_RANGE, N_VOL_POINTS)
volumes = []
energies = []

for i, vf in enumerate(vol_fractions):
    atoms_v = atoms_orig.copy()
    atoms_v.calc = calc
    # Scale cell isotropically: V' = V0 * vf, linear factor = vf^(1/3)
    scale = vf ** (1.0 / 3.0)
    atoms_v.set_cell(atoms_orig.cell * scale, scale_atoms=True)
    V = atoms_v.get_volume()
    E = atoms_v.get_potential_energy()
    volumes.append(V)
    energies.append(E)
    print(f"  [{i+1:2d}/{N_VOL_POINTS}] V = {V:.4f} A^3 ({vf*100:.1f}%), E = {E:.6f} eV")

volumes = np.array(volumes)
energies = np.array(energies)
n_atoms = len(atoms_orig)

# ============================================================
# 4. BIRCH-MURNAGHAN EOS FIT
# ============================================================

def birch_murnaghan_energy(V, E0, V0, B0, B0p):
    """3rd-order Birch-Murnaghan equation of state: E(V)."""
    eta = (V0 / V) ** (2.0 / 3.0)
    E = E0 + (9.0 * V0 * B0 / 16.0) * (
        (eta - 1.0) ** 3 * B0p
        + (eta - 1.0) ** 2 * (6.0 - 4.0 * eta)
    )
    return E

def birch_murnaghan_pressure(V, V0, B0, B0p):
    """Pressure P(V) from 3rd-order Birch-Murnaghan EOS, in GPa."""
    eta = (V0 / V) ** (2.0 / 3.0)
    P = (3.0 * B0 / 2.0) * (eta ** (7.0 / 2.0) - eta ** (5.0 / 2.0)) * (
        1.0 + 0.75 * (B0p - 4.0) * (eta - 1.0)
    )
    return P

def birch_murnaghan_bulk_modulus(V, V0, B0, B0p):
    """Isothermal bulk modulus B(V) from 3rd-order Birch-Murnaghan EOS, in GPa."""
    eta = (V0 / V) ** (2.0 / 3.0)
    B = (B0 / 2.0) * (
        eta ** (7.0 / 2.0) * (7.0 - 7.0 * eta + 9.0 * eta * B0p
                                - 9.0 * B0p + 9.0 * eta**2 * (1.0 - B0p))
        # Simplified: use numerical derivative for robustness
    )
    # More robust: compute B = -V * dP/dV numerically
    dV = V * 1e-5
    P_plus = birch_murnaghan_pressure(V + dV, V0, B0, B0p)
    P_minus = birch_murnaghan_pressure(V - dV, V0, B0, B0p)
    B_numerical = -V * (P_plus - P_minus) / (2.0 * dV)
    return B_numerical

print("\n--- Birch-Murnaghan EOS Fit ---")

# Initial guess
idx_min = np.argmin(energies)
E0_guess = energies[idx_min]
V0_guess = volumes[idx_min]
# Estimate B0 from parabolic fit
c2 = np.polyfit(volumes, energies, 2)[0]
B0_guess = 2.0 * c2 * V0_guess * 160.2176634  # eV/A^3 -> GPa
B0p_guess = 4.0

try:
    popt, pcov = curve_fit(
        birch_murnaghan_energy,
        volumes, energies,
        p0=[E0_guess, V0_guess, B0_guess / 160.2176634, B0p_guess],
        maxfev=10000,
    )
    E0_fit, V0_fit, B0_fit_eVA3, B0p_fit = popt
    B0_fit = B0_fit_eVA3 * 160.2176634  # eV/A^3 -> GPa

    # Compute residuals
    E_fitted = birch_murnaghan_energy(volumes, *popt)
    residual_rms = np.sqrt(np.mean((energies - E_fitted) ** 2))

    print(f"  E0  = {E0_fit:.6f} eV")
    print(f"  V0  = {V0_fit:.4f} A^3")
    print(f"  B0  = {B0_fit:.2f} GPa")
    print(f"  B0' = {B0p_fit:.2f}")
    print(f"  RMS residual = {residual_rms:.2e} eV")

except Exception as e:
    print(f"  ERROR: EOS fit failed: {e}")
    print("  Falling back to parabolic fit.")
    coeffs = np.polyfit(volumes, energies, 2)
    V0_fit = -coeffs[1] / (2 * coeffs[0])
    E0_fit = np.polyval(coeffs, V0_fit)
    B0_fit = 2.0 * coeffs[0] * V0_fit * 160.2176634
    B0p_fit = 4.0
    B0_fit_eVA3 = B0_fit / 160.2176634
    popt = [E0_fit, V0_fit, B0_fit_eVA3, B0p_fit]

# ============================================================
# 5. DEBYE TEMPERATURE FROM EOS
# ============================================================

def debye_temperature(V, B_GPa, n_atoms, avg_mass_amu):
    """
    Compute Debye temperature from volume and bulk modulus.

    Theta_D = (hbar/kB) * (6*pi^2 * n/V)^(1/3) * sqrt(B_S / rho)

    where rho = n*M/V is the mass density and B_S is the adiabatic bulk modulus.
    For the Debye model at T << Theta_D, B_S ~ B_T (isothermal).

    Simplified Debye-Slater formula:
      Theta_D = (hbar/kB) * [6*pi^2 * V^(1/2) * n]^(1/3) * sqrt(B / M)
    corrected with proper dimensional analysis:

      Theta_D = A * (V/n)^(1/6) * sqrt(B/M)

    where A is a numerical constant involving 6*pi^2, hbar, kB.

    More precisely:
      Theta_D = (hbar/kB) * (6*pi^2)^(1/3) * (n/V)^(1/3) * v_m

    with v_m = mean sound velocity derived from B and density rho = n*M/V:
      v_m ~ (B/rho)^(1/2)   (Debye-Slater approximation, ignoring shear modulus)

    In practice we use:
      Theta_D = (hbar/kB) * (6*pi^2)^(1/3) * (n/V)^(1/6) * sqrt(B/(M_avg))

    with all quantities in SI.
    """
    V_SI = V * A3_to_m3                          # m^3
    B_SI = B_GPa * GPa_to_Pa                     # Pa
    M_SI = avg_mass_amu * amu_to_kg               # kg
    rho = n_atoms * M_SI / V_SI                   # kg/m^3

    # Mean sound velocity (Debye-Slater: assumes Poisson ratio ~ 0.25)
    # v_m = (B/rho)^(1/2) * f(nu), where f(0.25) ~ 0.617
    # A commonly used scaling with Poisson ratio nu:
    #   v_m = [ (1/3) * (2/v_t^3 + 1/v_l^3) ]^(-1/3)
    # For Debye-Slater with Poisson ratio 0.25:
    #   v_l = sqrt(B*(1-nu)/rho/(1+nu)/(1-2*nu)) ~ sqrt(1.2*B/rho)
    #   v_t = sqrt(B*(1-2*nu)/2/rho/(1+nu))      ~ sqrt(0.4*B/rho)
    # => v_m ~ 0.617 * sqrt(B/rho)
    poisson = 0.25
    factor_l = np.sqrt((1.0 - poisson) / ((1.0 + poisson) * (1.0 - 2.0 * poisson)))
    factor_t = np.sqrt((1.0 - 2.0 * poisson) / (2.0 * (1.0 + poisson)))
    v_l = factor_l * np.sqrt(B_SI / rho)
    v_t = factor_t * np.sqrt(B_SI / rho)
    v_m = ((1.0 / 3.0) * (2.0 / v_t**3 + 1.0 / v_l**3)) ** (-1.0 / 3.0)

    theta = (hbar_SI / kB_SI) * (6.0 * np.pi**2 * n_atoms / V_SI) ** (1.0 / 3.0) * v_m
    return theta

# Average atomic mass
from ase.data import atomic_masses, atomic_numbers
symbols = atoms_orig.get_chemical_symbols()
masses = np.array([atomic_masses[atomic_numbers[s]] for s in symbols])
avg_mass = np.mean(masses)
print(f"\nAverage atomic mass: {avg_mass:.4f} amu")

# Debye temperature at equilibrium
B0_at_V0 = B0_fit
theta_D_0 = debye_temperature(V0_fit, B0_at_V0, n_atoms, avg_mass)
print(f"Debye temperature at V0: {theta_D_0:.1f} K")

# ============================================================
# 6. DEBYE VIBRATIONAL FREE ENERGY
# ============================================================

def debye_function_3(x):
    """
    Debye function D_3(x) = (3/x^3) * integral_0^x t^3/(e^t - 1) dt.
    """
    if x < 1e-12:
        return 1.0
    if x > 150:
        return 0.0
    integrand = lambda t: t**3 / (np.exp(t) - 1.0) if t > 0 else 0.0
    result, _ = quad(integrand, 0, x, limit=200)
    return 3.0 / x**3 * result

def vibrational_free_energy(T, theta_D, n_atoms):
    """
    Helmholtz vibrational free energy per unit cell from the Debye model (eV).

    F_vib = n * kB * T * [ (9/8)*(theta/T) + 3*ln(1 - exp(-theta/T)) - D_3(theta/T) ]

    At T=0: F_vib = n * (9/8) * kB * theta  (zero-point energy).
    """
    if T < 1e-6:
        # Zero-point energy
        return n_atoms * (9.0 / 8.0) * kB_eV * theta_D

    x = theta_D / T
    if x > 500:
        # Very low T limit
        return n_atoms * (9.0 / 8.0) * kB_eV * theta_D

    D3 = debye_function_3(x)
    exp_term = np.exp(-x)
    if exp_term > 1.0 - 1e-15:
        log_term = np.log(1e-15)
    else:
        log_term = np.log(1.0 - exp_term)

    F = n_atoms * kB_eV * T * (
        (9.0 / 8.0) * x + 3.0 * log_term - D3
    )
    return F

def vibrational_entropy(T, theta_D, n_atoms):
    """
    Vibrational entropy per unit cell from the Debye model (eV/K).
    S = n * kB * [ 4*D_3(theta/T) - 3*ln(1 - exp(-theta/T)) ]
    """
    if T < 1e-6:
        return 0.0
    x = theta_D / T
    if x > 500:
        return 0.0
    D3 = debye_function_3(x)
    exp_term = np.exp(-x)
    if exp_term > 1.0 - 1e-15:
        log_term = np.log(1e-15)
    else:
        log_term = np.log(1.0 - exp_term)
    S = n_atoms * kB_eV * (4.0 * D3 - 3.0 * log_term)
    return S

def vibrational_cv(T, theta_D, n_atoms):
    """
    Vibrational heat capacity Cv per unit cell from the Debye model (eV/K).
    Cv = 3*n*kB * [4*D_3(x) - 3*x/(e^x - 1)]  where x = theta/T.
    """
    if T < 1e-6:
        return 0.0
    x = theta_D / T
    if x > 500:
        return 0.0
    D3 = debye_function_3(x)
    ex = np.exp(x)
    Cv = 3.0 * n_atoms * kB_eV * (4.0 * D3 - 3.0 * x / (ex - 1.0))
    return Cv

# ============================================================
# 7. GIBBS FREE ENERGY MINIMIZATION: G(T,P)
# ============================================================

print(f"\n--- Computing Thermodynamic Properties ---")
print(f"  T range: {T_MIN} to {T_MAX} K, step {T_STEP} K")
print(f"  Pressure: {PRESSURE} GPa")

temperatures = np.arange(T_MIN, T_MAX + T_STEP, T_STEP, dtype=float)
n_T = len(temperatures)

# For each temperature, minimize G(V) = E(V) + F_vib(V,T) + P*V over V
# E(V) from the BM fit, F_vib from the Debye model with Theta_D(V)

V_min_scan = volumes.min() * 0.98
V_max_scan = volumes.max() * 1.02

# Store results
V_of_T = np.zeros(n_T)
G_of_T = np.zeros(n_T)
F_of_T = np.zeros(n_T)
S_of_T = np.zeros(n_T)
Cv_of_T = np.zeros(n_T)
B_of_T = np.zeros(n_T)
theta_of_T = np.zeros(n_T)

# PV term: P in GPa, V in A^3 -> PV in GPa*A^3
# 1 GPa*A^3 = 1e9 Pa * 1e-30 m^3 = 1e-21 J = 1e-21/1.602e-19 eV = 6.242e-3 eV
GPa_A3_to_eV = 6.2415091e-3

for i_T, T in enumerate(temperatures):
    def gibbs_at_V(V):
        # Static energy from BM EOS
        E_static = birch_murnaghan_energy(V, *popt)
        # Bulk modulus at this volume
        B_V = birch_murnaghan_bulk_modulus(V, V0_fit, B0_fit, B0p_fit)
        if B_V < 0.1:
            B_V = 0.1  # Prevent unphysical negative B
        # Debye temperature at this volume
        theta = debye_temperature(V, B_V, n_atoms, avg_mass)
        # Vibrational free energy
        F_vib = vibrational_free_energy(T, theta, n_atoms)
        # PV term
        PV = PRESSURE * V * GPa_A3_to_eV
        return E_static + F_vib + PV

    result = minimize_scalar(gibbs_at_V, bounds=(V_min_scan, V_max_scan), method="bounded")
    V_eq = result.x
    G_eq = result.fun

    # Compute properties at equilibrium volume
    B_eq = birch_murnaghan_bulk_modulus(V_eq, V0_fit, B0_fit, B0p_fit)
    if B_eq < 0.1:
        B_eq = 0.1
    theta_eq = debye_temperature(V_eq, B_eq, n_atoms, avg_mass)
    S_eq = vibrational_entropy(T, theta_eq, n_atoms)
    Cv_eq = vibrational_cv(T, theta_eq, n_atoms)

    V_of_T[i_T] = V_eq
    G_of_T[i_T] = G_eq
    F_of_T[i_T] = G_eq - PRESSURE * V_eq * GPa_A3_to_eV  # Helmholtz
    S_of_T[i_T] = S_eq
    Cv_of_T[i_T] = Cv_eq
    B_of_T[i_T] = B_eq
    theta_of_T[i_T] = theta_eq

print("  Minimization complete.")

# ============================================================
# 8. DERIVED PROPERTIES
# ============================================================

# Thermal expansion coefficient: alpha = (1/V) * dV/dT
# Compute by numerical differentiation
alpha_of_T = np.zeros(n_T)
for i in range(1, n_T - 1):
    dT = temperatures[i + 1] - temperatures[i - 1]
    if dT > 0:
        alpha_of_T[i] = (V_of_T[i + 1] - V_of_T[i - 1]) / (dT * V_of_T[i])
# End points: forward/backward difference
if n_T > 1:
    dT_fwd = temperatures[1] - temperatures[0]
    if dT_fwd > 0:
        alpha_of_T[0] = (V_of_T[1] - V_of_T[0]) / (dT_fwd * V_of_T[0])
    dT_bwd = temperatures[-1] - temperatures[-2]
    if dT_bwd > 0:
        alpha_of_T[-1] = (V_of_T[-1] - V_of_T[-2]) / (dT_bwd * V_of_T[-1])

# Heat capacity at constant pressure: Cp = Cv + alpha^2 * B * V * T
# Units: alpha (1/K), B (GPa), V (A^3), T (K)
# alpha^2 * B * V * T in (1/K^2) * GPa * A^3 * K = GPa*A^3/K
# Convert to eV/K: multiply by GPa_A3_to_eV
Cp_of_T = np.zeros(n_T)
for i in range(n_T):
    T = temperatures[i]
    if T > 0:
        Cp_of_T[i] = Cv_of_T[i] + alpha_of_T[i]**2 * B_of_T[i] * V_of_T[i] * T * GPa_A3_to_eV
    else:
        Cp_of_T[i] = Cv_of_T[i]

# Convert Cv and Cp to J/(K*mol_atoms) for more intuitive units
# 1 eV/K per cell = eV_to_J / n_atoms * NA  J/(K*mol_atoms)
eVK_to_JKmol = eV_to_J * NA  # per mole of unit cells
# Per mole of atoms:
eVK_to_JKmol_atom = eV_to_J * NA / n_atoms

Cv_JKmol = Cv_of_T * eVK_to_JKmol_atom
Cp_JKmol = Cp_of_T * eVK_to_JKmol_atom

# Gruneisen parameter: gamma = alpha * B * V / Cv
gamma_of_T = np.zeros(n_T)
for i in range(n_T):
    if Cv_of_T[i] > 1e-20:
        gamma_of_T[i] = alpha_of_T[i] * B_of_T[i] * V_of_T[i] * GPa_A3_to_eV / Cv_of_T[i]
    else:
        gamma_of_T[i] = np.nan

# ============================================================
# 9. SAVE NUMERICAL DATA
# ============================================================

data_file = os.path.join(WORK_DIR, "qh_debye_results.dat")
header = (
    "# Quasi-Harmonic Debye Model Results\n"
    f"# Formula: {atoms_orig.get_chemical_formula()}, n_atoms = {n_atoms}\n"
    f"# E0 = {E0_fit:.6f} eV, V0 = {V0_fit:.4f} A^3, B0 = {B0_fit:.2f} GPa, B0' = {B0p_fit:.2f}\n"
    f"# Theta_D(V0) = {theta_D_0:.1f} K\n"
    f"# Pressure = {PRESSURE:.4f} GPa\n"
    "# Columns: T(K) V(A^3) G(eV) Cv(J/K/mol_atom) Cp(J/K/mol_atom) "
    "alpha(1/K) B(GPa) Theta_D(K) gamma\n"
)
data = np.column_stack([
    temperatures, V_of_T, G_of_T, Cv_JKmol, Cp_JKmol,
    alpha_of_T, B_of_T, theta_of_T, gamma_of_T
])
np.savetxt(data_file, data, header=header, fmt="%.6e",
           comments="")
print(f"\n  Data saved: {data_file}")

# Also save E(V) data
ev_file = os.path.join(WORK_DIR, "e_v_data.dat")
np.savetxt(ev_file, np.column_stack([volumes, energies]),
           header="V(A^3) E(eV)", fmt="%.6e")
print(f"  E(V) data saved: {ev_file}")

# ============================================================
# 10. PLOTTING
# ============================================================

print("\n--- Generating Plots ---")

# --- Plot 1: E(V) curve with BM fit ---
fig, ax = plt.subplots(figsize=(7, 5))
V_fine = np.linspace(volumes.min() * 0.99, volumes.max() * 1.01, 200)
E_fine = birch_murnaghan_energy(V_fine, *popt)
ax.plot(V_fine, E_fine, "b-", linewidth=1.5, label="Birch-Murnaghan fit")
ax.plot(volumes, energies, "ko", markersize=6, label="Calculated (MACE)")
ax.axvline(V0_fit, color="gray", linestyle="--", alpha=0.5, label=f"$V_0$ = {V0_fit:.2f} $\\AA^3$")
ax.set_xlabel("Volume ($\\AA^3$)", fontsize=13)
ax.set_ylabel("Energy (eV)", fontsize=13)
ax.set_title("Energy-Volume Curve", fontsize=14)
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3)
fig.tight_layout()
fig.savefig(os.path.join(WORK_DIR, "e_v_curve.png"), dpi=150)
plt.close()
print("  Saved: e_v_curve.png")

# --- Plot 2: Gibbs free energy G(T) ---
fig, ax = plt.subplots(figsize=(7, 5))
mask_T = temperatures > 0
ax.plot(temperatures[mask_T], G_of_T[mask_T], "b-", linewidth=1.5)
ax.set_xlabel("Temperature (K)", fontsize=13)
ax.set_ylabel("Gibbs Free Energy (eV/cell)", fontsize=13)
ax.set_title(f"G(T) at P = {PRESSURE} GPa", fontsize=14)
ax.grid(True, alpha=0.3)
fig.tight_layout()
fig.savefig(os.path.join(WORK_DIR, "gibbs_free_energy.png"), dpi=150)
plt.close()
print("  Saved: gibbs_free_energy.png")

# --- Plot 3: Thermal expansion coefficient alpha(T) ---
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

mask_alpha = temperatures > 10
ax1.plot(temperatures[mask_alpha], V_of_T[mask_alpha], "b-", linewidth=1.5)
ax1.set_xlabel("Temperature (K)", fontsize=13)
ax1.set_ylabel("Volume ($\\AA^3$)", fontsize=13)
ax1.set_title("V(T)", fontsize=14)
ax1.grid(True, alpha=0.3)

ax2.plot(temperatures[mask_alpha], alpha_of_T[mask_alpha] * 1e6, "r-", linewidth=1.5)
ax2.set_xlabel("Temperature (K)", fontsize=13)
ax2.set_ylabel("$\\alpha$ ($10^{-6}$ K$^{-1}$)", fontsize=13)
ax2.set_title("Thermal Expansion Coefficient", fontsize=14)
ax2.grid(True, alpha=0.3)

fig.tight_layout()
fig.savefig(os.path.join(WORK_DIR, "thermal_expansion.png"), dpi=150)
plt.close()
print("  Saved: thermal_expansion.png")

# --- Plot 4: Bulk modulus B(T) ---
fig, ax = plt.subplots(figsize=(7, 5))
ax.plot(temperatures[mask_T], B_of_T[mask_T], "g-", linewidth=1.5)
ax.set_xlabel("Temperature (K)", fontsize=13)
ax.set_ylabel("Bulk Modulus (GPa)", fontsize=13)
ax.set_title("B(T)", fontsize=14)
ax.grid(True, alpha=0.3)
fig.tight_layout()
fig.savefig(os.path.join(WORK_DIR, "bulk_modulus_T.png"), dpi=150)
plt.close()
print("  Saved: bulk_modulus_T.png")

# --- Plot 5: Heat capacities Cv(T) and Cp(T) ---
fig, ax = plt.subplots(figsize=(7, 5))
ax.plot(temperatures[mask_T], Cv_JKmol[mask_T], "b-", linewidth=1.5, label="$C_v$")
ax.plot(temperatures[mask_T], Cp_JKmol[mask_T], "r-", linewidth=1.5, label="$C_p$")
ax.axhline(3.0 * 8.314, color="gray", linestyle="--", alpha=0.5, label="Dulong-Petit (3R)")
ax.set_xlabel("Temperature (K)", fontsize=13)
ax.set_ylabel("Heat Capacity (J K$^{-1}$ mol$^{-1}$)", fontsize=13)
ax.set_title("Heat Capacity (per mol of atoms)", fontsize=14)
ax.legend(fontsize=11)
ax.grid(True, alpha=0.3)
fig.tight_layout()
fig.savefig(os.path.join(WORK_DIR, "heat_capacity.png"), dpi=150)
plt.close()
print("  Saved: heat_capacity.png")

# --- Plot 6: Debye temperature vs T ---
fig, ax = plt.subplots(figsize=(7, 5))
ax.plot(temperatures[mask_T], theta_of_T[mask_T], "m-", linewidth=1.5)
ax.set_xlabel("Temperature (K)", fontsize=13)
ax.set_ylabel("$\\Theta_D$ (K)", fontsize=13)
ax.set_title("Debye Temperature vs T", fontsize=14)
ax.grid(True, alpha=0.3)
fig.tight_layout()
fig.savefig(os.path.join(WORK_DIR, "debye_temperature.png"), dpi=150)
plt.close()
print("  Saved: debye_temperature.png")

# --- Plot 7: Gruneisen parameter vs T ---
fig, ax = plt.subplots(figsize=(7, 5))
mask_g = (temperatures > 20) & np.isfinite(gamma_of_T)
if np.any(mask_g):
    ax.plot(temperatures[mask_g], gamma_of_T[mask_g], "darkorange", linewidth=1.5)
ax.set_xlabel("Temperature (K)", fontsize=13)
ax.set_ylabel("Gruneisen Parameter $\\gamma$", fontsize=13)
ax.set_title("Thermodynamic Gruneisen Parameter", fontsize=14)
ax.grid(True, alpha=0.3)
fig.tight_layout()
fig.savefig(os.path.join(WORK_DIR, "gruneisen_parameter.png"), dpi=150)
plt.close()
print("  Saved: gruneisen_parameter.png")

# --- Plot 8: Multi-pressure G(T) comparison (if P=0, also show P=5,10 GPa) ---
if PRESSURE < 0.01:
    fig, ax = plt.subplots(figsize=(7, 5))
    for P_plot in [0.0, 5.0, 10.0, 20.0]:
        G_P = np.zeros(n_T)
        for i_T, T in enumerate(temperatures):
            def gibbs_P(V):
                E_s = birch_murnaghan_energy(V, *popt)
                B_V = birch_murnaghan_bulk_modulus(V, V0_fit, B0_fit, B0p_fit)
                if B_V < 0.1:
                    B_V = 0.1
                th = debye_temperature(V, B_V, n_atoms, avg_mass)
                Fv = vibrational_free_energy(T, th, n_atoms)
                return E_s + Fv + P_plot * V * GPa_A3_to_eV
            res = minimize_scalar(gibbs_P, bounds=(V_min_scan, V_max_scan), method="bounded")
            G_P[i_T] = res.fun
        ax.plot(temperatures[mask_T], G_P[mask_T], linewidth=1.5, label=f"P = {P_plot:.0f} GPa")
    ax.set_xlabel("Temperature (K)", fontsize=13)
    ax.set_ylabel("Gibbs Free Energy (eV/cell)", fontsize=13)
    ax.set_title("G(T) at Different Pressures", fontsize=14)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(WORK_DIR, "gibbs_multi_pressure.png"), dpi=150)
    plt.close()
    print("  Saved: gibbs_multi_pressure.png")

# ============================================================
# 11. SUMMARY
# ============================================================

print("\n" + "=" * 60)
print("RESULTS SUMMARY")
print("=" * 60)
print(f"  Formula:         {atoms_orig.get_chemical_formula()}")
print(f"  Atoms/cell:      {n_atoms}")
print(f"  Average mass:    {avg_mass:.2f} amu")
print(f"  V0 (EOS):        {V0_fit:.4f} A^3")
print(f"  B0 (EOS):        {B0_fit:.2f} GPa")
print(f"  B0' (EOS):       {B0p_fit:.2f}")
print(f"  Theta_D(V0):     {theta_D_0:.1f} K")

for target_T in [100, 300, 500, 800, 1000]:
    if target_T > T_MAX:
        break
    idx = np.argmin(np.abs(temperatures - target_T))
    T_actual = temperatures[idx]
    print(f"\n  At T = {T_actual:.0f} K:")
    print(f"    V(T)      = {V_of_T[idx]:.4f} A^3  ({(V_of_T[idx]/V0_fit - 1)*100:+.3f}%)")
    print(f"    G(T)      = {G_of_T[idx]:.6f} eV")
    if alpha_of_T[idx] > 0:
        print(f"    alpha(T)  = {alpha_of_T[idx]*1e6:.2f} x 10^-6 K^-1")
    print(f"    B(T)      = {B_of_T[idx]:.2f} GPa")
    print(f"    Cv(T)     = {Cv_JKmol[idx]:.2f} J/(K*mol)")
    print(f"    Cp(T)     = {Cp_JKmol[idx]:.2f} J/(K*mol)")
    print(f"    Theta_D   = {theta_of_T[idx]:.1f} K")
    if np.isfinite(gamma_of_T[idx]):
        print(f"    gamma     = {gamma_of_T[idx]:.3f}")

print(f"\nAll outputs saved to: {WORK_DIR}")
print("Done.")
```

### Method B: QE pw.x for DFT-Accurate E(V)

For higher accuracy (or when MACE is not trained for your chemistry), compute the E(V) curve using QE SCF calculations at each volume. The Debye analysis is identical.

```python
#!/usr/bin/env python3
"""
Quasi-Harmonic Debye Model with QE pw.x for E(V).
Runs SCF at multiple volumes, fits EOS, applies Debye model.
"""

import os
import re
import subprocess
import numpy as np
from scipy.optimize import minimize_scalar, curve_fit
from scipy.integrate import quad

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from ase.io import read
from pymatgen.core.structure import Structure
from pymatgen.io.ase import AseAtomsAdaptor

import warnings
warnings.filterwarnings("ignore")

# ============================================================
# CONFIGURATION
# ============================================================

STRUCTURE_FILE = "relaxed.cif"         # Pre-relaxed structure (use scf-relax skill first)
PSEUDO_DIR = os.path.abspath("./pseudo")
OUTDIR_BASE = os.path.abspath("./tmp_qh_debye_qe")
os.makedirs(OUTDIR_BASE, exist_ok=True)

ECUTWFC = 60.0                         # Ry
ECUTRHO = 480.0                        # Ry
KPOINTS = "6 6 6"                      # k-mesh (adjust for your cell)
NPROC = 4                              # MPI processes
PREFIX = "qhd"

N_VOL_POINTS = 11
VOL_RANGE = 0.06                       # +/- 6%

T_MIN = 0
T_MAX = 1500
T_STEP = 5
PRESSURE = 0.0                         # GPa

WORK_DIR = "/tmp/qh_debye_qe"
os.makedirs(WORK_DIR, exist_ok=True)

# Physical constants
kB_SI = 1.380649e-23
hbar_SI = 1.054571817e-34
NA = 6.02214076e23
eV_to_J = 1.602176634e-19
Ry_to_eV = 13.605693122994
A3_to_m3 = 1e-30
GPa_to_Pa = 1e9
amu_to_kg = 1.66053906660e-27
kB_eV = 8.617333262e-5
GPa_A3_to_eV = 6.2415091e-3
bohr_to_A = 0.529177249

# ============================================================
# LOAD STRUCTURE AND BUILD QE INPUTS
# ============================================================

atoms = read(STRUCTURE_FILE)
n_atoms = len(atoms)
formula = atoms.get_chemical_formula()
print(f"Formula: {formula}, n_atoms: {n_atoms}")

pmg_struct = AseAtomsAdaptor.get_structure(atoms)
V0 = atoms.get_volume()

# Get unique species and write ATOMIC_SPECIES
from collections import OrderedDict
from ase.data import atomic_masses, atomic_numbers
species_info = OrderedDict()
for sym in atoms.get_chemical_symbols():
    if sym not in species_info:
        mass = atomic_masses[atomic_numbers[sym]]
        # Pseudopotential naming convention -- adjust as needed
        pp_file = f"{sym}.pbe-n-rrkjus_psl.1.0.0.UPF"
        species_info[sym] = (mass, pp_file)

species_card = "\n".join(
    f"  {sym}  {mass:.4f}  {pp}" for sym, (mass, pp) in species_info.items()
)
ntyp = len(species_info)

# ============================================================
# RUN SCF AT EACH VOLUME
# ============================================================

vol_fractions = np.linspace(1.0 - VOL_RANGE, 1.0 + VOL_RANGE, N_VOL_POINTS)
volumes = []
energies_eV = []

print(f"\n--- SCF at {N_VOL_POINTS} volumes ---")

for i_v, vf in enumerate(vol_fractions):
    scale = vf ** (1.0 / 3.0)
    atoms_v = atoms.copy()
    atoms_v.set_cell(atoms.cell * scale, scale_atoms=True)
    V = atoms_v.get_volume()

    # Build CELL_PARAMETERS and ATOMIC_POSITIONS
    cell = atoms_v.cell[:]
    cell_card = "\n".join(f"  {row[0]:.10f}  {row[1]:.10f}  {row[2]:.10f}" for row in cell)
    positions = atoms_v.get_scaled_positions()
    symbols = atoms_v.get_chemical_symbols()
    pos_card = "\n".join(
        f"  {sym}  {p[0]:.10f}  {p[1]:.10f}  {p[2]:.10f}"
        for sym, p in zip(symbols, positions)
    )

    out_dir_v = os.path.join(OUTDIR_BASE, f"vol_{i_v:02d}")
    os.makedirs(out_dir_v, exist_ok=True)

    scf_input = f"""&CONTROL
    calculation = 'scf'
    prefix      = '{PREFIX}'
    outdir      = '{out_dir_v}'
    pseudo_dir  = '{PSEUDO_DIR}'
    tprnfor     = .true.
    tstress     = .true.
/
&SYSTEM
    ibrav       = 0
    nat         = {n_atoms}
    ntyp        = {ntyp}
    ecutwfc     = {ECUTWFC}
    ecutrho     = {ECUTRHO}
    occupations = 'smearing'
    smearing    = 'cold'
    degauss     = 0.01
/
&ELECTRONS
    conv_thr = 1.0d-8
/

CELL_PARAMETERS (angstrom)
{cell_card}

ATOMIC_SPECIES
{species_card}

ATOMIC_POSITIONS (crystal)
{pos_card}

K_POINTS (automatic)
  {KPOINTS}  0 0 0
"""
    in_file = os.path.join(WORK_DIR, f"scf_vol_{i_v:02d}.in")
    out_file = os.path.join(WORK_DIR, f"scf_vol_{i_v:02d}.out")
    with open(in_file, "w") as f:
        f.write(scf_input)

    print(f"  [{i_v+1:2d}/{N_VOL_POINTS}] V = {V:.4f} A^3 ({vf*100:.1f}%) ... ", end="", flush=True)

    result = subprocess.run(
        ["mpirun", "-np", str(NPROC), "pw.x", "-in", in_file],
        capture_output=True, text=True, timeout=3600,
    )
    with open(out_file, "w") as f:
        f.write(result.stdout)

    # Extract total energy (Ry) from output
    energy_Ry = None
    for line in result.stdout.split("\n"):
        if "!" in line and "total energy" in line:
            m = re.search(r"=\s+([-\d.]+)\s+Ry", line)
            if m:
                energy_Ry = float(m.group(1))

    if energy_Ry is not None:
        E_eV = energy_Ry * Ry_to_eV
        volumes.append(V)
        energies_eV.append(E_eV)
        print(f"E = {E_eV:.6f} eV")
    else:
        print("FAILED -- check output")

volumes = np.array(volumes)
energies_eV = np.array(energies_eV)

print(f"\nSuccessfully computed {len(volumes)}/{N_VOL_POINTS} volume points.")

# ============================================================
# EOS FIT + DEBYE ANALYSIS
# (identical to Method A from here -- same functions apply)
# ============================================================

# ... (paste the Birch-Murnaghan fit, Debye functions, Gibbs
#      minimization, and plotting sections from Method A above)

print("Proceed with EOS fit and Debye analysis using the functions from Method A.")
print("Copy sections 4-11 from Method A, replacing 'volumes' and 'energies' with the QE values.")
```

### Method C: LAMMPS for Classical Potential E(V)

For ultra-fast screening with classical interatomic potentials (EAM, MEAM, Tersoff, etc.), use LAMMPS to compute the E(V) curve, then apply the same Debye model analysis.

```python
#!/usr/bin/env python3
"""
Quasi-Harmonic Debye with LAMMPS E(V) scan.
Uses ASE's LAMMPS calculator interface.
"""

import os
import numpy as np
from scipy.optimize import minimize_scalar, curve_fit
from scipy.integrate import quad

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from ase.io import read, write
from ase.optimize import LBFGS
from ase.constraints import ExpCellFilter
from ase.calculators.lammpsrun import LAMMPS

import warnings
warnings.filterwarnings("ignore")

# ============================================================
# CONFIGURATION
# ============================================================

STRUCTURE_FILE = "structure.cif"

# LAMMPS potential setup -- adjust for your system
LAMMPS_PARAMS = {
    "pair_style": "eam/alloy",
    "pair_coeff": ["* * Cu_mishin1.eam.alloy Cu"],
}
# For other potentials:
#   Tersoff: pair_style tersoff, pair_coeff * * SiC.tersoff Si C
#   MEAM:    pair_style meam, pair_coeff * * library.meam Si Si.meam Si

LAMMPS_FILES = ["Cu_mishin1.eam.alloy"]  # Potential files to ship

N_VOL_POINTS = 15
VOL_RANGE = 0.08
T_MAX = 1500
T_STEP = 5
PRESSURE = 0.0

WORK_DIR = "/tmp/qh_debye_lammps"
os.makedirs(WORK_DIR, exist_ok=True)

# ============================================================
# VOLUME SCAN
# ============================================================

atoms = read(STRUCTURE_FILE)
n_atoms = len(atoms)

calc = LAMMPS(
    parameters=LAMMPS_PARAMS,
    files=LAMMPS_FILES,
    tmp_dir=os.path.join(WORK_DIR, "lammps_tmp"),
    keep_tmp_files=False,
)

# Relax first
atoms.calc = calc
ecf = ExpCellFilter(atoms, hydrostatic_strain=False)
opt = LBFGS(ecf, logfile=os.path.join(WORK_DIR, "relax.log"))
opt.run(fmax=1e-4, steps=500)
V0 = atoms.get_volume()

vol_fractions = np.linspace(1.0 - VOL_RANGE, 1.0 + VOL_RANGE, N_VOL_POINTS)
volumes = []
energies = []

for i, vf in enumerate(vol_fractions):
    atoms_v = atoms.copy()
    atoms_v.calc = calc
    scale = vf ** (1.0 / 3.0)
    atoms_v.set_cell(atoms.cell * scale, scale_atoms=True)
    V = atoms_v.get_volume()
    E = atoms_v.get_potential_energy()
    volumes.append(V)
    energies.append(E)
    print(f"  [{i+1:2d}/{N_VOL_POINTS}] V = {V:.4f} A^3 ({vf*100:.1f}%), E = {E:.6f} eV")

volumes = np.array(volumes)
energies = np.array(energies)

# Proceed with EOS fit and Debye analysis (sections 4-11 from Method A)
print("E(V) scan complete. Apply Birch-Murnaghan fit and Debye analysis from Method A.")
```

## Key Parameters

| Parameter | Default | Notes |
|---|---|---|
| `N_VOL_POINTS` | 11 | Number of volumes. Minimum 7 for a robust 3rd-order BM fit. 11-15 is recommended. |
| `VOL_RANGE` | 0.06 | Fractional volume range (+/- 6%). Too narrow gives poor B0'; too wide may leave the harmonic regime. |
| `T_MAX` | 1500 K | Upper temperature. Should be below melting. Debye model is most accurate below Theta_D. |
| `PRESSURE` | 0.0 GPa | Applied pressure for G(T,P). Nonzero values shift the equilibrium volume. |
| `Poisson ratio` | 0.25 | Assumed in the Debye-Slater formula. 0.25 is a reasonable default for metals and ceramics. For specific materials, set to the known value. |
| `FMAX` | 1e-4 eV/A | Relaxation convergence. Tighter than needed for E(V) but ensures clean V0. |
| `MACE_MODEL` | "medium" | MACE foundation model size. "large" is more accurate but 3x slower. |

### Choosing the volume range

- **Hard metals (W, Mo, Fe)**: +/- 4-6% is safe. These are stiff and remain stable over wide ranges.
- **Soft metals (Al, Cu, Au)**: +/- 5-8% works well.
- **Semiconductors (Si, GaAs)**: +/- 4-6%. Larger strains can trigger phase transitions.
- **Ionic compounds (MgO, NaCl)**: +/- 5-8% is fine for rocksalt structures.
- **Molecular crystals / soft materials**: Use +/- 3-4%. These are fragile under compression.
- **If the EOS fit is poor (large residuals, unphysical B0')**: Narrow the range or add more points.

### Choosing the Poisson ratio

The Debye-Slater model requires an assumed Poisson ratio to convert B into sound velocities. Common values:

| Material class | Typical Poisson ratio |
|---|---|
| Most metals | 0.25-0.35 |
| Ceramics/oxides | 0.20-0.28 |
| Covalent semiconductors | 0.20-0.25 |
| Cauchy relation (ionic) | 0.25 |
| Nearly incompressible | 0.45-0.50 |

If elastic constants are known, compute shear modulus G and use nu = (3B - 2G) / (2*(3B + G)).

## Interpreting Results

### Debye temperature (Theta_D)

- **Typical values**: Diamond 2230 K, Fe ~470 K, Al ~428 K, Cu ~343 K, Au ~165 K, Pb ~105 K.
- Higher Theta_D = stiffer material with lighter atoms.
- Theta_D decreases with increasing T (volume expansion softens the material).
- The Debye model is most accurate when the actual phonon DOS resembles a Debye spectrum (parabolic at low omega, sharp cutoff). This is best for simple cubic metals and worst for layered or molecular crystals.

### Gibbs free energy G(T,P)

- G decreases with T due to the -TS entropy contribution.
- At constant T, G increases with P due to the PV term.
- The equilibrium phase at given (T,P) has the lowest G. Compare G(T,P) of two phases to predict phase boundaries.

### Thermal expansion coefficient alpha(T)

- **Typical metals**: 10-30 x 10^-6 K^-1 (Cu ~17, Al ~23, Fe ~12 at 300 K).
- **Ceramics**: 5-15 x 10^-6 K^-1 (MgO ~10, Al2O3 ~8).
- **Diamond/SiC**: 1-5 x 10^-6 K^-1.
- alpha increases from 0 at T=0, rises steeply near Theta_D/5, and plateaus at high T.
- The Debye model cannot produce negative thermal expansion (that requires specific phonon mode behavior).

### Heat capacity Cp(T)

- Cv follows the Debye T^3 law at low T, saturates to 3NkB (Dulong-Petit) at high T.
- Cp > Cv always. The difference Cp - Cv = alpha^2 * B * V * T grows with T.
- For metals at 300 K, Cp is typically 1-5% larger than Cv.
- The Debye model gives Cv accurately for monatomic solids. For complex crystals with optical branches, it may underestimate Cv below Theta_D.

### Bulk modulus B(T)

- Decreases monotonically with T (materials soften as they expand).
- Typical decrease: 5-15% from 0 K to 1000 K for metals.
- If B(T) shows non-monotonic behavior, check the EOS fit quality.

### Gruneisen parameter gamma

- gamma = alpha * B * V / Cv.
- **Typical values**: 1.0-3.0 for most solids (diamond ~1.0, Cu ~1.96, Al ~2.17, NaCl ~1.57).
- Nearly constant with T for simple solids.
- gamma > 3 suggests strong anharmonicity where the Debye model becomes questionable.

## Common Issues

| Issue | Cause | Fix |
|---|---|---|
| EOS fit fails or gives B0' < 0 or > 10 | Noisy E(V) data or too few points | Increase N_VOL_POINTS; tighten FMAX; try narrower VOL_RANGE |
| Theta_D unrealistically high (> 5000 K) | Bulk modulus too large (common in MACE for some chemistries) | Check B0 against experiment. Try a different MACE model or use QE. |
| Theta_D unrealistically low (< 50 K) | Bulk modulus too small or wrong average mass | Verify the structure is correct and fully relaxed. |
| alpha(T) oscillates or is noisy | Numerical differentiation of V(T) amplifies noise | Use more temperature points (smaller T_STEP); smooth V(T) before differentiating |
| Cp diverges at high T | alpha^2 * B * V * T term blows up if alpha is noisy | Smooth alpha(T) or reduce T_MAX |
| B0' ~ 4.0 always | BM fit insensitive to B0' with narrow volume range | Widen VOL_RANGE to +/- 8% or use 13+ volume points |
| Results differ from full phonon QHA | Debye model approximates the full phonon DOS | Expected. Debye model is a screening tool. Use phonon QHA (gruneisen-qha skill) for publication accuracy. |
| G(T) has kinks or discontinuities | Volume minimization finds different local minima at adjacent T | Increase V scan density; use the previous T's V_eq as initial guess |
| MACE gives wrong E(V) curvature | MACE not accurate for this chemistry | Compare with QE at 3-5 volumes. If disagreement > 5% in B0, use QE for all points. |
| Zero-point energy shifts V0 upward | Physical effect: ZPE expands the lattice | This is correct behavior. The Debye ZPE = (9/8)*n*kB*Theta_D is included automatically. |

## Comparison: Debye Model vs Full Phonon QHA

| Aspect | Quasi-Harmonic Debye (this skill) | Full Phonon QHA (gruneisen-qha skill) |
|---|---|---|
| Input | E(V) only | E(V) + phonon spectra at each V |
| Speed | Minutes (MACE), hours (QE) | Hours (MACE), days (QE) |
| Accuracy | Good for simple cubic metals | High for all crystals |
| Cp(T) | From Debye model | From full phonon DOS |
| Negative alpha | Cannot capture | Can capture (if phonon modes soften) |
| Optical modes | Approximated by single Theta_D | Fully resolved |
| Best for | Screening, alloy design, trends | Publication-quality results |
| Dependencies | scipy only | phonopy required |
