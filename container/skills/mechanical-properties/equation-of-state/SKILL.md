# Equation of State (EOS) Calculation

## When to Use

- You need the equilibrium volume, bulk modulus, or pressure derivative of the bulk modulus.
- You want the energy-volume (E-V) curve of a material.
- You are studying pressure-induced phase transitions (compare E-V curves of competing phases).
- You need to validate a relaxed structure by checking that it sits at the E-V minimum.
- You want to compare MACE predictions against DFT for a specific material.
- You need enthalpy-pressure (H-P) curves to identify phase transition pressures.
- You want to compare multiple EOS models (Birch-Murnaghan, Vinet, Murnaghan, etc.) and assess fit quality.
- You need cohesive energy from the E-V minimum referenced to isolated atoms.

## Method Selection (MACE vs QE vs LAMMPS)

| Criterion | MACE (ASE) | QE (DFT) | LAMMPS (classical) |
|---|---|---|---|
| Speed | Seconds (7--11 points in ~10s) | Hours (7--11 SCF calculations) | Milliseconds to seconds |
| Accuracy | Good for systems within MACE training domain | Systematically improvable, publication quality | Depends on potential quality |
| Use when | Screening, rapid estimation, comparing many structures | Publication results, unusual chemistry, validating MACE | Large cells, classical potentials, high-throughput |

## Prerequisites

- A crystal structure (CIF, POSCAR, or pymatgen Structure). Ideally pre-relaxed, but the workflow includes an initial relaxation step.
- For QE: pseudopotential files (SSSP recommended). See `electronic-structure/scf-relax/SKILL.md`.
- For LAMMPS: interatomic potential file (EAM, MEAM, Tersoff, etc.).
- Python packages: `pymatgen`, `ase`, `mace-torch`, `numpy`, `scipy`, `matplotlib` (pre-installed).

## Detailed Steps

### Method A: ASE + MACE (Complete Murnaghan-style Workflow)

This method performs a systematic volume scan inspired by the pyiron Murnaghan workflow: (1) relax the equilibrium structure, (2) generate strained structures over a configurable volume range, (3) relax ions at each fixed volume, (4) collect E(V) data, (5) fit to five EOS models (3rd-order Birch-Murnaghan, 4th-order Birch-Murnaghan, Vinet, Murnaghan, Poirier-Tarantola), (6) run convergence checks, (7) extract V0, B0, B0', cohesive energy, (8) generate publication-quality plots including E-V, P-V, and H-P curves.

```python
#!/usr/bin/env python3
"""
Equation of State calculation using ASE + MACE.

Murnaghan-style workflow (following pyiron_atomistics pattern):
  1. Relax the structure (full cell + ions).
  2. Define strain range (default: -5% to +5% volume strain, 11 points).
  3. Generate strained structures via isotropic lattice scaling.
  4. At each strained volume, relax ionic positions (fixed cell).
  5. Collect E(V), P(V) data.
  6. Fit 5 EOS models: BM3, BM4, Vinet, Murnaghan, Poirier-Tarantola.
  7. Convergence diagnostics: check volume range and point density.
  8. Extract V0, E0, B0, B0', cohesive energy.
  9. Publication-quality plots: E-V, P-V, H-P, model comparison.

No pyiron dependency -- standalone script using ASE, scipy, pymatgen.
"""

import json
import os
import warnings
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

from ase.io import read as ase_read
from ase.optimize import LBFGS
from ase.constraints import ExpCellFilter
from ase.units import kJ

from pymatgen.core.structure import Structure
from pymatgen.io.ase import AseAtomsAdaptor
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer

from scipy.optimize import curve_fit, least_squares
from scipy.interpolate import UnivariateSpline

# ─── CONFIGURATION ───────────────────────────────────────────────────────────
INPUT_FILE = "structure.cif"           # Input structure (CIF, POSCAR, etc.)
MACE_MODEL = "medium"                  # MACE model: "small", "medium", "large"
VOL_RANGE = 0.05                       # Fractional volume strain range (+/-)
N_POINTS = 11                          # Number of volume points (odd centers on V0)
STRAINS = None                         # Override: explicit list of volume strains
                                       # e.g. [-0.06, -0.04, ..., 0.04, 0.06]
AXES = ("x", "y", "z")                # Axes along which strain is applied
FMAX_BULK = 1e-4                       # Force convergence for initial relaxation (eV/A)
FMAX_EOS = 1e-3                        # Force convergence for strained relaxations
FIT_ORDER_POLY = 3                     # Order for polynomial fit (fallback)
OUTPUT_DIR = "eos_results"
# ─────────────────────────────────────────────────────────────────────────────

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ─── EOS MODEL DEFINITIONS ──────────────────────────────────────────────────
# All models: E(V; E0, V0, B0, B0p) or E(V; E0, V0, B0, B0p, B0pp)
# B0 in eV/A^3, convert to GPa via * 160.21766208

def birch_murnaghan_3rd(V, E0, V0, B0, B0p):
    """3rd-order Birch-Murnaghan EOS: E(V)."""
    eta = (V0 / V) ** (2.0 / 3.0)
    return E0 + (9.0 * V0 * B0 / 16.0) * (
        (eta - 1.0)**3 * B0p + (eta - 1.0)**2 * (6.0 - 4.0 * eta)
    )

def birch_murnaghan_4th(V, E0, V0, B0, B0p, B0pp):
    """4th-order Birch-Murnaghan EOS: E(V).
    B0pp is the second pressure derivative of the bulk modulus (in GPa^-1)."""
    f = 0.5 * ((V0 / V) ** (2.0 / 3.0) - 1.0)
    # H = B0*B0pp + B0p**2 - 7*B0p + 143/9
    H = B0 * B0pp + B0p**2 - 7.0 * B0p + 143.0 / 9.0
    return E0 + (9.0 / 6.0) * V0 * B0 * f**2 * (
        1.0 + (B0p - 4.0) * f + H * f**2
    )

def vinet_eos(V, E0, V0, B0, B0p):
    """Vinet (Rose) universal EOS: E(V)."""
    x = (V / V0) ** (1.0 / 3.0)
    eta = 1.5 * (B0p - 1.0) * (1.0 - x)
    return E0 + (2.0 * B0 * V0 / (B0p - 1.0)**2) * (
        2.0 - (5.0 + 3.0 * B0p * (x - 1.0) - 3.0 * x) * np.exp(eta)
    )

def murnaghan_eos(V, E0, V0, B0, B0p):
    """Murnaghan EOS: E(V)."""
    return E0 + B0 * V / B0p * (
        ((V0 / V)**B0p) / (B0p - 1.0) + 1.0
    ) - V0 * B0 / (B0p - 1.0)

def poirier_tarantola(V, E0, V0, B0, B0p):
    """Poirier-Tarantola (logarithmic) EOS: E(V)."""
    x = np.log(V / V0) / 3.0
    return E0 + (9.0 / 2.0) * V0 * B0 * x**2 * (
        1.0 + (B0p - 2.0) * x
    )

def birch_murnaghan_pressure(V, V0, B0, B0p):
    """3rd-order Birch-Murnaghan: P(V) in same units as B0."""
    eta = (V0 / V) ** (2.0 / 3.0)
    return (3.0 * B0 / 2.0) * (eta**(7.0/2.0) - eta**(5.0/2.0)) * (
        1.0 + 0.75 * (B0p - 4.0) * (eta - 1.0)
    )

def vinet_pressure(V, V0, B0, B0p):
    """Vinet: P(V) in same units as B0."""
    x = (V / V0) ** (1.0 / 3.0)
    return 3.0 * B0 * (1.0 - x) / x**2 * np.exp(
        1.5 * (B0p - 1.0) * (1.0 - x)
    )

EV_PER_A3_TO_GPA = 160.21766208

EOS_MODELS = {
    "birch_murnaghan_3rd": {
        "func": birch_murnaghan_3rd,
        "nparams": 4,
        "param_names": ["E0", "V0", "B0", "B0p"],
        "label": "Birch-Murnaghan (3rd)",
    },
    "birch_murnaghan_4th": {
        "func": birch_murnaghan_4th,
        "nparams": 5,
        "param_names": ["E0", "V0", "B0", "B0p", "B0pp"],
        "label": "Birch-Murnaghan (4th)",
    },
    "vinet": {
        "func": vinet_eos,
        "nparams": 4,
        "param_names": ["E0", "V0", "B0", "B0p"],
        "label": "Vinet (Rose)",
    },
    "murnaghan": {
        "func": murnaghan_eos,
        "nparams": 4,
        "param_names": ["E0", "V0", "B0", "B0p"],
        "label": "Murnaghan",
    },
    "poirier_tarantola": {
        "func": poirier_tarantola,
        "nparams": 4,
        "param_names": ["E0", "V0", "B0", "B0p"],
        "label": "Poirier-Tarantola",
    },
}

PRESSURE_MODELS = {
    "birch_murnaghan_3rd": birch_murnaghan_pressure,
    "vinet": vinet_pressure,
}

# ─── HELPER: Strain axes (following pyiron _strain_axes pattern) ─────────────

def strain_axes(structure, axes, volume_strain):
    """
    Apply isotropic volume strain along specified axes.
    volume_strain: fractional change in volume (e.g. 0.05 = +5%).
    Returns a new pymatgen Structure.
    """
    n_axes = len(axes)
    linear_scale = (1.0 + volume_strain) ** (1.0 / n_axes)
    new_lattice = structure.lattice.matrix.copy()
    axis_map = {"x": 0, "y": 1, "z": 2}
    for ax in axes:
        new_lattice[axis_map[ax]] *= linear_scale
    strained = structure.copy()
    strained.lattice = new_lattice
    return strained


# ─── STEP 1: Load structure and set up MACE calculator ───────────────────────
from mace.calculators import mace_mp
calc = mace_mp(model=MACE_MODEL, default_dtype="float64")

adaptor = AseAtomsAdaptor()

structure = Structure.from_file(INPUT_FILE)
formula = structure.composition.reduced_formula
n_atoms = len(structure)
print(f"Loaded: {formula} ({n_atoms} atoms)")

sga = SpacegroupAnalyzer(structure, symprec=0.01)
print(f"Space group: {sga.get_space_group_symbol()}")

# ─── STEP 2: Relax the equilibrium structure ────────────────────────────────
atoms_eq = adaptor.get_atoms(structure)
atoms_eq.calc = calc

ecf = ExpCellFilter(atoms_eq, hydrostatic_strain=False)
opt = LBFGS(ecf, logfile=os.path.join(OUTPUT_DIR, "eq_relax.log"))
opt.run(fmax=FMAX_BULK, steps=500)

relaxed_structure = adaptor.get_structure(atoms_eq)
V0_relaxed = relaxed_structure.volume
E0_relaxed = atoms_eq.get_potential_energy()

relaxed_structure.to(os.path.join(OUTPUT_DIR, "relaxed_structure.cif"))
print(f"Equilibrium: V = {V0_relaxed:.4f} A^3, E = {E0_relaxed:.6f} eV")

# ─── STEP 3: Generate volume-strained structures (pyiron Murnaghan pattern) ──
# Either use explicit strains or generate uniform grid over [-vol_range, +vol_range]
if STRAINS is not None:
    strain_values = np.array(STRAINS)
else:
    strain_values = np.linspace(-VOL_RANGE, VOL_RANGE, N_POINTS)

print(f"\nVolume strain values: {strain_values}")
print(f"Volume ratios: {1.0 + strain_values}")

# ─── STEP 4: Relax at each strained volume ──────────────────────────────────
volumes = []
energies = []
pressures = []

# Include the equilibrium point
volumes.append(V0_relaxed)
energies.append(E0_relaxed)
eq_stress = atoms_eq.get_stress(voigt=True)  # eV/A^3
eq_pressure = -np.mean(eq_stress[:3]) * EV_PER_A3_TO_GPA  # GPa
pressures.append(eq_pressure)
print(f"\nEquilibrium: V={V0_relaxed:.4f} A^3, E={E0_relaxed:.6f} eV, "
      f"P={eq_pressure:.4f} GPa")

for i, vol_strain in enumerate(strain_values):
    # Apply isotropic volume strain along specified axes
    strained = strain_axes(relaxed_structure, AXES, vol_strain)

    # Convert to ASE and relax ions only (fixed cell)
    atoms_s = adaptor.get_atoms(strained)
    atoms_s.calc = calc

    opt = LBFGS(atoms_s, logfile=os.devnull)
    opt.run(fmax=FMAX_EOS, steps=300)

    V = atoms_s.get_volume()
    E = atoms_s.get_potential_energy()
    stress = atoms_s.get_stress(voigt=True)
    P = -np.mean(stress[:3]) * EV_PER_A3_TO_GPA  # GPa

    volumes.append(V)
    energies.append(E)
    pressures.append(P)

    print(f"  Point {i+1}/{len(strain_values)}: strain={vol_strain:+.4f}, "
          f"V={V:.4f} A^3 ({V/V0_relaxed*100:.1f}%), E={E:.6f} eV, P={P:.3f} GPa")

# Sort by volume
sort_idx = np.argsort(volumes)
volumes = np.array(volumes)[sort_idx]
energies = np.array(energies)[sort_idx]
pressures = np.array(pressures)[sort_idx]
energies_per_atom = energies / n_atoms

# ─── STEP 5: Fit all EOS models ─────────────────────────────────────────────
print("\n" + "=" * 70)
print("EOS FITTING RESULTS (5 models)")
print("=" * 70)

fit_results = {}

# Initial guesses
E0_guess = np.min(energies)
V0_guess = volumes[np.argmin(energies)]
# Estimate B0 from parabolic fit near minimum
poly_coeffs = np.polyfit(volumes, energies, 2)
B0_guess = 2.0 * poly_coeffs[0] * V0_guess  # d^2E/dV^2 * V
if B0_guess <= 0:
    B0_guess = 0.5  # eV/A^3 fallback
B0p_guess = 4.0
B0pp_guess = -0.01  # GPa^-1, for 4th-order BM

for model_name, model_info in EOS_MODELS.items():
    try:
        func = model_info["func"]
        nparams = model_info["nparams"]

        if nparams == 4:
            p0 = [E0_guess, V0_guess, B0_guess, B0p_guess]
            bounds = (
                [-np.inf, volumes.min() * 0.5, 0.0, 0.0],
                [np.inf, volumes.max() * 2.0, np.inf, 20.0],
            )
        else:  # 5 params (4th-order BM)
            p0 = [E0_guess, V0_guess, B0_guess, B0p_guess, B0pp_guess]
            bounds = (
                [-np.inf, volumes.min() * 0.5, 0.0, 0.0, -1.0],
                [np.inf, volumes.max() * 2.0, np.inf, 20.0, 1.0],
            )

        popt, pcov = curve_fit(func, volumes, energies, p0=p0, maxfev=50000,
                               bounds=bounds)
        perr = np.sqrt(np.diag(pcov))

        # Compute residuals and R^2
        E_pred = func(volumes, *popt)
        ss_res = np.sum((energies - E_pred) ** 2)
        ss_tot = np.sum((energies - np.mean(energies)) ** 2)
        r_squared = 1.0 - ss_res / ss_tot
        rmse = np.sqrt(ss_res / len(energies))

        E0_fit, V0_fit, B0_fit, B0p_fit = popt[:4]
        B0_gpa = B0_fit * EV_PER_A3_TO_GPA
        B0_err_gpa = perr[2] * EV_PER_A3_TO_GPA

        result = {
            "E0_eV": float(E0_fit),
            "V0_A3": float(V0_fit),
            "B0_GPa": float(B0_gpa),
            "B0_prime": float(B0p_fit),
            "E0_err_eV": float(perr[0]),
            "V0_err_A3": float(perr[1]),
            "B0_err_GPa": float(B0_err_gpa),
            "B0_prime_err": float(perr[3]),
            "R_squared": float(r_squared),
            "RMSE_eV": float(rmse),
            "popt": [float(x) for x in popt],
        }
        if nparams == 5:
            result["B0_double_prime_GPa_inv"] = float(popt[4])
            result["B0_double_prime_err"] = float(perr[4])

        fit_results[model_name] = result

        print(f"\n  {model_info['label']}:")
        print(f"    V0 = {V0_fit:.4f} +/- {perr[1]:.4f} A^3")
        print(f"    E0 = {E0_fit:.6f} +/- {perr[0]:.6f} eV "
              f"({E0_fit/n_atoms:.6f} eV/atom)")
        print(f"    B0 = {B0_gpa:.2f} +/- {B0_err_gpa:.2f} GPa")
        print(f"    B0' = {B0p_fit:.2f} +/- {perr[3]:.2f}")
        if nparams == 5:
            print(f"    B0'' = {popt[4]:.4f} +/- {perr[4]:.4f} GPa^-1")
        print(f"    R^2 = {r_squared:.10f}, RMSE = {rmse:.2e} eV")

    except Exception as e:
        print(f"\n  {model_info['label']}: FAILED -- {e}")

# --- Polynomial fit (fallback, following pyiron) ---
try:
    poly_fit = np.polyfit(volumes, energies, FIT_ORDER_POLY)
    p_poly = np.poly1d(poly_fit)
    E_poly_pred = p_poly(volumes)
    ss_res_poly = np.sum((energies - E_poly_pred) ** 2)
    ss_tot_poly = np.sum((energies - np.mean(energies)) ** 2)
    r2_poly = 1.0 - ss_res_poly / ss_tot_poly

    # Extract V0 from polynomial minimum
    p_deriv = p_poly.deriv()
    roots = np.roots(p_deriv)
    real_roots = roots[np.isreal(roots)].real
    valid_roots = real_roots[
        (real_roots > volumes.min()) & (real_roots < volumes.max())
    ]
    if len(valid_roots) > 0:
        V0_poly = valid_roots[np.argmin(p_poly(valid_roots))]
        E0_poly = p_poly(V0_poly)
        # B0 from second derivative: B0 = V * d^2E/dV^2
        p_deriv2 = p_poly.deriv(2)
        B0_poly = V0_poly * p_deriv2(V0_poly) * EV_PER_A3_TO_GPA

        fit_results["polynomial"] = {
            "E0_eV": float(E0_poly),
            "V0_A3": float(V0_poly),
            "B0_GPa": float(B0_poly),
            "poly_coeffs": [float(c) for c in poly_fit],
            "fit_order": FIT_ORDER_POLY,
            "R_squared": float(r2_poly),
        }
        print(f"\n  Polynomial (order {FIT_ORDER_POLY}):")
        print(f"    V0 = {V0_poly:.4f} A^3, E0 = {E0_poly:.6f} eV, "
              f"B0 = {B0_poly:.2f} GPa, R^2 = {r2_poly:.10f}")
except Exception as e:
    print(f"\n  Polynomial fit: FAILED -- {e}")

# ─── STEP 6: Cohesive energy ────────────────────────────────────────────────
# Compute isolated atom energies for cohesive energy
print("\n" + "-" * 50)
print("COHESIVE ENERGY")
print("-" * 50)
cohesive_energy = None
try:
    from ase import Atoms as ASE_Atoms

    # Get unique species
    species_counts = {}
    for site in relaxed_structure:
        sym = site.specie.symbol
        species_counts[sym] = species_counts.get(sym, 0) + 1

    isolated_energies = {}
    for symbol in species_counts:
        atom = ASE_Atoms(symbol, positions=[[0, 0, 0]],
                         cell=[15, 15, 15], pbc=False)
        atom.calc = calc
        e_iso = atom.get_potential_energy()
        isolated_energies[symbol] = e_iso
        print(f"  E_isolated({symbol}) = {e_iso:.6f} eV")

    # Cohesive energy per atom = (sum of isolated - E_bulk) / n_atoms
    E_iso_total = sum(isolated_energies[sym] * count
                      for sym, count in species_counts.items())
    # Use best-fit E0 (prefer BM3)
    E_bulk = fit_results.get("birch_murnaghan_3rd", {}).get("E0_eV", E0_relaxed)
    cohesive_energy = (E_iso_total - E_bulk) / n_atoms
    print(f"  Cohesive energy = {cohesive_energy:.4f} eV/atom "
          f"({cohesive_energy * 96.485:.1f} kJ/mol)")
except Exception as e:
    print(f"  Cohesive energy calculation: FAILED -- {e}")

# ─── STEP 7: Convergence checks ─────────────────────────────────────────────
print("\n" + "-" * 50)
print("CONVERGENCE DIAGNOSTICS")
print("-" * 50)

convergence_ok = True
warnings_list = []

# Check 1: Does the E-V curve have a clear minimum inside the range?
E_min_idx = np.argmin(energies)
if E_min_idx == 0 or E_min_idx == len(energies) - 1:
    msg = ("WARNING: E-V minimum is at the boundary of the volume range. "
           "The equilibrium volume may lie outside. Expand the volume range.")
    warnings_list.append(msg)
    convergence_ok = False
    print(f"  [FAIL] {msg}")
else:
    print("  [OK] E-V minimum is interior to the scanned volume range.")

# Check 2: Are there enough points?
if len(volumes) < 7:
    msg = ("WARNING: Only {} data points. Use at least 7 for reliable 4-parameter "
           "fits. Recommended: 11.".format(len(volumes)))
    warnings_list.append(msg)
    print(f"  [WARN] {msg}")
else:
    print(f"  [OK] {len(volumes)} data points (>= 7).")

# Check 3: Do different EOS models agree on V0 and B0?
if len(fit_results) >= 2:
    V0_vals = [r["V0_A3"] for r in fit_results.values() if "V0_A3" in r]
    B0_vals = [r["B0_GPa"] for r in fit_results.values() if "B0_GPa" in r]
    if V0_vals:
        V0_spread = (max(V0_vals) - min(V0_vals)) / np.mean(V0_vals) * 100
        if V0_spread > 2.0:
            msg = (f"WARNING: V0 spread across models = {V0_spread:.1f}% "
                   f"(>2%). Data may be noisy or range insufficient.")
            warnings_list.append(msg)
            convergence_ok = False
            print(f"  [WARN] {msg}")
        else:
            print(f"  [OK] V0 spread = {V0_spread:.2f}% across models.")
    if B0_vals:
        B0_spread = (max(B0_vals) - min(B0_vals)) / np.mean(B0_vals) * 100
        if B0_spread > 5.0:
            msg = (f"WARNING: B0 spread across models = {B0_spread:.1f}% "
                   f"(>5%). Consider wider range or more points.")
            warnings_list.append(msg)
            print(f"  [WARN] {msg}")
        else:
            print(f"  [OK] B0 spread = {B0_spread:.2f}% across models.")

# Check 4: B0' sanity
bm3 = fit_results.get("birch_murnaghan_3rd", {})
if bm3:
    B0p_val = bm3.get("B0_prime", 4.0)
    if B0p_val < 2.0 or B0p_val > 8.0:
        msg = (f"WARNING: B0' = {B0p_val:.2f} is outside typical range [2, 8]. "
               f"Check data quality or expand volume range.")
        warnings_list.append(msg)
        print(f"  [WARN] {msg}")
    else:
        print(f"  [OK] B0' = {B0p_val:.2f} within normal range [2, 8].")

# Check 5: Is the volume range wide enough for the E-V curvature?
E_range = energies.max() - energies.min()
if E_range < 0.05 * n_atoms:  # Less than 50 meV/atom span
    msg = (f"WARNING: E-V span = {E_range/n_atoms*1000:.1f} meV/atom "
           f"(<50 meV/atom). Range may be too narrow for reliable fits.")
    warnings_list.append(msg)
    print(f"  [WARN] {msg}")
else:
    print(f"  [OK] E-V span = {E_range/n_atoms*1000:.1f} meV/atom (>= 50 meV/atom).")

# Check 6: Equilibrium volume from fit is close to relaxed volume
if bm3:
    v0_shift = abs(bm3["V0_A3"] - V0_relaxed) / V0_relaxed * 100
    if v0_shift > 1.0:
        msg = (f"WARNING: Fitted V0 differs from relaxed V0 by {v0_shift:.2f}%. "
               f"Initial relaxation may not have converged.")
        warnings_list.append(msg)
        print(f"  [WARN] {msg}")
    else:
        print(f"  [OK] Fitted V0 within {v0_shift:.3f}% of relaxed V0.")

convergence_status = "CONVERGED" if convergence_ok and len(warnings_list) == 0 else (
    "CONVERGED_WITH_WARNINGS" if convergence_ok else "NOT_CONVERGED"
)
print(f"\n  Overall: {convergence_status}")

# ─── STEP 8: Enthalpy-Pressure curves for phase transition analysis ─────────
# H = E + PV  (enthalpy per atom)
enthalpies_per_atom = (energies + pressures / EV_PER_A3_TO_GPA * volumes) / n_atoms

# ─── STEP 9: Publication-quality plots ───────────────────────────────────────
fig = plt.figure(figsize=(18, 14))
gs = GridSpec(2, 2, figure=fig, hspace=0.30, wspace=0.28)

colors = {
    "birch_murnaghan_3rd": "#e41a1c",
    "birch_murnaghan_4th": "#ff7f00",
    "vinet": "#4daf4a",
    "murnaghan": "#377eb8",
    "poirier_tarantola": "#984ea3",
    "polynomial": "#a65628",
}

V_fine = np.linspace(volumes.min() * 0.98, volumes.max() * 1.02, 300)

# --- Panel 1: E-V curve with all models ---
ax1 = fig.add_subplot(gs[0, 0])
ax1.plot(volumes / n_atoms, energies_per_atom, "ko", markersize=7,
         label="Calculated", zorder=5)

for model_name, result in fit_results.items():
    if model_name == "polynomial":
        p_poly = np.poly1d(result["poly_coeffs"])
        ax1.plot(V_fine / n_atoms, p_poly(V_fine) / n_atoms, "--",
                 color=colors.get(model_name, "gray"), linewidth=1.2,
                 label=f"Poly (order {result['fit_order']})", alpha=0.7)
    elif model_name in EOS_MODELS:
        func = EOS_MODELS[model_name]["func"]
        popt = result["popt"]
        E_fit = func(V_fine, *popt)
        ax1.plot(V_fine / n_atoms, E_fit / n_atoms, "-",
                 color=colors.get(model_name, "gray"), linewidth=1.5,
                 label=EOS_MODELS[model_name]["label"])

# Mark equilibrium
if "birch_murnaghan_3rd" in fit_results:
    bm3_v0 = fit_results["birch_murnaghan_3rd"]["V0_A3"]
    bm3_e0 = fit_results["birch_murnaghan_3rd"]["E0_eV"]
    ax1.axvline(bm3_v0 / n_atoms, color="gray", linestyle=":", alpha=0.4)
    ax1.plot(bm3_v0 / n_atoms, bm3_e0 / n_atoms, "r*", markersize=14,
             zorder=6, label=f"V$_0$={bm3_v0:.2f} $\\AA^3$")

ax1.set_xlabel("Volume per atom ($\\AA^3$/atom)", fontsize=12)
ax1.set_ylabel("Energy per atom (eV/atom)", fontsize=12)
ax1.set_title(f"E-V Curve: {formula}", fontsize=13, fontweight="bold")
ax1.legend(fontsize=8, loc="upper right")
ax1.grid(True, alpha=0.3)

# --- Panel 2: P-V curve ---
ax2 = fig.add_subplot(gs[0, 1])
ax2.plot(volumes / n_atoms, pressures, "ko", markersize=7,
         label="Calculated (stress)", zorder=5)

for model_name in ["birch_murnaghan_3rd", "vinet"]:
    if model_name in fit_results and model_name in PRESSURE_MODELS:
        r = fit_results[model_name]
        P_fit = PRESSURE_MODELS[model_name](
            V_fine, r["V0_A3"], r["B0_GPa"] / EV_PER_A3_TO_GPA, r["B0_prime"]
        ) * EV_PER_A3_TO_GPA
        ax2.plot(V_fine / n_atoms, P_fit, "-",
                 color=colors.get(model_name, "gray"), linewidth=1.5,
                 label=EOS_MODELS[model_name]["label"])

ax2.axhline(0, color="gray", linestyle=":", alpha=0.4)
ax2.set_xlabel("Volume per atom ($\\AA^3$/atom)", fontsize=12)
ax2.set_ylabel("Pressure (GPa)", fontsize=12)
ax2.set_title(f"P-V Curve: {formula}", fontsize=13, fontweight="bold")
ax2.legend(fontsize=9)
ax2.grid(True, alpha=0.3)

# --- Panel 3: H-P curve (enthalpy vs pressure, for phase transitions) ---
ax3 = fig.add_subplot(gs[1, 0])
ax3.plot(pressures, enthalpies_per_atom, "ko-", markersize=7,
         linewidth=1.0, label=formula, zorder=5)

ax3.set_xlabel("Pressure (GPa)", fontsize=12)
ax3.set_ylabel("Enthalpy per atom (eV/atom)", fontsize=12)
ax3.set_title(f"H-P Curve: {formula}", fontsize=13, fontweight="bold")
ax3.legend(fontsize=10)
ax3.grid(True, alpha=0.3)

# --- Panel 4: Residuals & model comparison ---
ax4 = fig.add_subplot(gs[1, 1])

for model_name in ["birch_murnaghan_3rd", "vinet", "murnaghan", "poirier_tarantola"]:
    if model_name in fit_results and model_name in EOS_MODELS:
        func = EOS_MODELS[model_name]["func"]
        popt = fit_results[model_name]["popt"]
        E_pred = func(volumes, *popt)
        residuals_meV = (energies - E_pred) * 1000.0  # meV
        ax4.plot(volumes / n_atoms, residuals_meV / n_atoms, "o-",
                 color=colors.get(model_name, "gray"), markersize=5,
                 linewidth=1.0, label=EOS_MODELS[model_name]["label"])

ax4.axhline(0, color="gray", linestyle=":", alpha=0.4)
ax4.set_xlabel("Volume per atom ($\\AA^3$/atom)", fontsize=12)
ax4.set_ylabel("Residual (meV/atom)", fontsize=12)
ax4.set_title("EOS Fit Residuals", fontsize=13, fontweight="bold")
ax4.legend(fontsize=8)
ax4.grid(True, alpha=0.3)

plt.savefig(os.path.join(OUTPUT_DIR, "eos_curves.png"), dpi=200, bbox_inches="tight")
print(f"\nPlot saved to {OUTPUT_DIR}/eos_curves.png")

# ─── Summary table ───────────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("SUMMARY TABLE")
print("=" * 70)
print(f"{'Model':<25} {'V0 (A^3)':<14} {'B0 (GPa)':<14} {'B0\\'':<10} {'R^2':<14}")
print("-" * 70)
for model_name, result in fit_results.items():
    label = EOS_MODELS.get(model_name, {}).get("label", model_name)
    v0 = result.get("V0_A3", float("nan"))
    b0 = result.get("B0_GPa", float("nan"))
    b0p = result.get("B0_prime", float("nan"))
    r2 = result.get("R_squared", float("nan"))
    print(f"  {label:<23} {v0:<14.4f} {b0:<14.2f} {b0p:<10.2f} {r2:<14.10f}")

# ─── Save all results ───────────────────────────────────────────────────────
all_results = {
    "formula": formula,
    "n_atoms": n_atoms,
    "method": f"MACE-MP-0 ({MACE_MODEL})",
    "vol_range": VOL_RANGE,
    "n_points": N_POINTS,
    "axes": list(AXES),
    "volumes_A3": volumes.tolist(),
    "energies_eV": energies.tolist(),
    "energies_per_atom_eV": energies_per_atom.tolist(),
    "pressures_GPa": pressures.tolist(),
    "enthalpies_per_atom_eV": enthalpies_per_atom.tolist(),
    "eos_fits": fit_results,
    "cohesive_energy_eV_per_atom": cohesive_energy,
    "convergence": {
        "status": convergence_status,
        "warnings": warnings_list,
    },
}
with open(os.path.join(OUTPUT_DIR, "eos_results.json"), "w") as f:
    json.dump(all_results, f, indent=2, default=str)
print(f"\nResults saved to {OUTPUT_DIR}/eos_results.json")
```

### Method B: LAMMPS (Classical Potentials)

This method uses LAMMPS via ASE's LAMMPSlib interface for fast EOS calculations with classical interatomic potentials. Useful for large cells, metals with established EAM/MEAM potentials, or high-throughput screening.

```python
#!/usr/bin/env python3
"""
EOS calculation using LAMMPS via ASE.

Uses the same Murnaghan-style volume scan as Method A, but with a
classical interatomic potential instead of MACE.
"""

import json
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from ase.io import read as ase_read
from ase.optimize import LBFGS
from ase.constraints import ExpCellFilter
from ase.calculators.lammpslib import LAMMPSlib

from pymatgen.core.structure import Structure
from pymatgen.io.ase import AseAtomsAdaptor
from scipy.optimize import curve_fit

# ─── CONFIGURATION ───────────────────────────────────────────────────────────
INPUT_FILE = "structure.cif"
POTENTIAL_FILE = "Cu.eam.alloy"        # Path to potential file
ELEMENT_LIST = ["Cu"]                   # Element(s) in order
PAIR_STYLE = "eam/alloy"               # LAMMPS pair_style
VOL_RANGE = 0.05
N_POINTS = 11
FMAX_BULK = 1e-5
FMAX_EOS = 1e-4
OUTPUT_DIR = "eos_results_lammps"
# ─────────────────────────────────────────────────────────────────────────────

os.makedirs(OUTPUT_DIR, exist_ok=True)

# Set up LAMMPS calculator
lammps_cmds = [
    f"pair_style {PAIR_STYLE}",
    f"pair_coeff * * {os.path.abspath(POTENTIAL_FILE)} {' '.join(ELEMENT_LIST)}",
]
calc = LAMMPSlib(
    lmpcmds=lammps_cmds,
    atom_types={el: i + 1 for i, el in enumerate(ELEMENT_LIST)},
    log_file=os.path.join(OUTPUT_DIR, "lammps.log"),
    keep_alive=True,
)

adaptor = AseAtomsAdaptor()
structure = Structure.from_file(INPUT_FILE)
formula = structure.composition.reduced_formula
n_atoms = len(structure)

# Relax equilibrium
atoms_eq = adaptor.get_atoms(structure)
atoms_eq.calc = calc
ecf = ExpCellFilter(atoms_eq, hydrostatic_strain=False)
opt = LBFGS(ecf, logfile=os.path.join(OUTPUT_DIR, "eq_relax.log"))
opt.run(fmax=FMAX_BULK, steps=500)

relaxed_structure = adaptor.get_structure(atoms_eq)
V0_relaxed = relaxed_structure.volume
E0_relaxed = atoms_eq.get_potential_energy()
print(f"Equilibrium: V = {V0_relaxed:.4f} A^3, E = {E0_relaxed:.6f} eV")

# Volume scan
strain_values = np.linspace(-VOL_RANGE, VOL_RANGE, N_POINTS)
volumes, energies, pressures = [V0_relaxed], [E0_relaxed], []
eq_stress = atoms_eq.get_stress(voigt=True)
pressures.append(-np.mean(eq_stress[:3]) * 160.21766208)

for i, vol_strain in enumerate(strain_values):
    n_axes = 3
    linear_scale = (1.0 + vol_strain) ** (1.0 / n_axes)
    strained = relaxed_structure.copy()
    strained.lattice = strained.lattice.matrix * linear_scale

    atoms_s = adaptor.get_atoms(strained)
    atoms_s.calc = calc
    opt = LBFGS(atoms_s, logfile=os.devnull)
    opt.run(fmax=FMAX_EOS, steps=300)

    V = atoms_s.get_volume()
    E = atoms_s.get_potential_energy()
    stress = atoms_s.get_stress(voigt=True)
    P = -np.mean(stress[:3]) * 160.21766208

    volumes.append(V)
    energies.append(E)
    pressures.append(P)
    print(f"  Point {i+1}/{N_POINTS}: V={V:.4f} A^3, E={E:.6f} eV, P={P:.3f} GPa")

sort_idx = np.argsort(volumes)
volumes = np.array(volumes)[sort_idx]
energies = np.array(energies)[sort_idx]
pressures = np.array(pressures)[sort_idx]

# Fit Birch-Murnaghan 3rd order
def bm3(V, E0, V0, B0, B0p):
    eta = (V0 / V) ** (2.0 / 3.0)
    return E0 + (9.0 * V0 * B0 / 16.0) * (
        (eta - 1.0)**3 * B0p + (eta - 1.0)**2 * (6.0 - 4.0 * eta))

popt, pcov = curve_fit(bm3, volumes, energies,
    p0=[np.min(energies), volumes[np.argmin(energies)], 0.5, 4.0], maxfev=50000)
perr = np.sqrt(np.diag(pcov))
print(f"\nBirch-Murnaghan 3rd-order fit:")
print(f"  V0 = {popt[1]:.4f} A^3, B0 = {popt[2]*160.21766208:.2f} GPa, "
      f"B0' = {popt[3]:.2f}")

results = {
    "formula": formula, "method": f"LAMMPS ({PAIR_STYLE})",
    "V0_A3": float(popt[1]), "B0_GPa": float(popt[2] * 160.21766208),
    "B0_prime": float(popt[3]), "E0_eV": float(popt[0]),
    "volumes_A3": volumes.tolist(), "energies_eV": energies.tolist(),
    "pressures_GPa": pressures.tolist(),
}
with open(os.path.join(OUTPUT_DIR, "eos_results.json"), "w") as f:
    json.dump(results, f, indent=2, default=str)
print(f"Results saved to {OUTPUT_DIR}/eos_results.json")
```

### Method C: QE DFT

This method uses Quantum ESPRESSO `pw.x` to compute total energies at multiple volumes. The workflow mirrors atomate2: relax the equilibrium cell, then perform fixed-volume relaxations (or SCF calculations) at strained volumes.

#### Step C1: Relax the equilibrium structure

Use the same `vc-relax` input as in the elastic constants workflow.

```
&CONTROL
    calculation  = 'vc-relax'
    prefix       = 'bulk'
    outdir       = './tmp'
    pseudo_dir   = './pseudo'
    tprnfor      = .true.
    tstress      = .true.
    forc_conv_thr = 1.0d-5
    etot_conv_thr = 1.0d-7
/
&SYSTEM
    ibrav        = 0
    nat          = 2
    ntyp         = 1
    ecutwfc      = 60.0
    ecutrho      = 480.0
    occupations  = 'smearing'
    smearing     = 'mv'
    degauss      = 0.02
/
&ELECTRONS
    conv_thr     = 1.0d-10
    mixing_beta  = 0.7
/
&IONS
    ion_dynamics = 'bfgs'
/
&CELL
    cell_dynamics = 'bfgs'
    press_conv_thr = 0.1
/
ATOMIC_SPECIES
  Si  28.085  Si.pbe-n-rrkjus_psl.1.0.0.UPF
ATOMIC_POSITIONS crystal
  Si  0.000  0.000  0.000
  Si  0.250  0.250  0.250
CELL_PARAMETERS angstrom
  0.000  2.715  2.715
  2.715  0.000  2.715
  2.715  2.715  0.000
K_POINTS automatic
  8 8 8  0 0 0
```

Run: `pw.x < relax.in > relax.out`

#### Step C2: Generate volume-strained QE inputs

```python
#!/usr/bin/env python3
"""
Generate QE inputs for EOS calculation: fixed-volume ionic relaxation
at multiple isotropic strains.
"""

import os
import json
import numpy as np
from pymatgen.core.structure import Structure
from pymatgen.io.pwscf import PWInput

# ─── CONFIGURATION ───────────────────────────────────────────────────────────
RELAXED_FILE = "relaxed_structure.cif"
PSEUDO_DIR = "./pseudo"
ECUTWFC = 60.0
ECUTRHO = 480.0
K_GRID = [8, 8, 8]
VOL_RANGE = 0.05                       # +/-5% volume strain
N_POINTS = 11                          # 11 points (pyiron default)
WORK_DIR = "eos_qe"
# ─────────────────────────────────────────────────────────────────────────────

os.makedirs(WORK_DIR, exist_ok=True)

structure = Structure.from_file(RELAXED_FILE)
formula = structure.composition.reduced_formula
V0 = structure.volume
print(f"Relaxed structure: {formula}, V0 = {V0:.4f} A^3")

# Generate strain values (uniform grid, following pyiron Murnaghan)
strain_values = np.linspace(-VOL_RANGE, VOL_RANGE, N_POINTS)

# Pseudopotential map
pseudo_map = {}
for el in structure.composition.elements:
    symbol = el.symbol
    pseudo_map[symbol] = f"{symbol}.pbe-n-rrkjus_psl.1.0.0.UPF"

eos_info = []
for idx, vol_strain in enumerate(strain_values):
    deform_dir = os.path.join(WORK_DIR, f"vol_{idx:03d}")
    os.makedirs(deform_dir, exist_ok=True)

    # Isotropic volume strain: scale all lattice vectors uniformly
    linear_scale = (1.0 + vol_strain) ** (1.0 / 3.0)
    strained = structure.copy()
    new_lattice = strained.lattice.matrix * linear_scale
    strained.lattice = new_lattice
    V_strained = strained.volume

    # QE input: relax ions at fixed cell
    input_params = {
        "CONTROL": {
            "calculation": "relax",
            "prefix": f"vol_{idx:03d}",
            "outdir": "./tmp",
            "pseudo_dir": os.path.abspath(PSEUDO_DIR),
            "tprnfor": True,
            "tstress": True,
            "forc_conv_thr": 1.0e-5,
            "etot_conv_thr": 1.0e-8,
        },
        "SYSTEM": {
            "ecutwfc": ECUTWFC,
            "ecutrho": ECUTRHO,
            "occupations": "smearing",
            "smearing": "mv",
            "degauss": 0.02,
        },
        "ELECTRONS": {
            "conv_thr": 1.0e-10,
            "mixing_beta": 0.7,
        },
        "IONS": {
            "ion_dynamics": "bfgs",
        },
    }

    pw_input = PWInput(
        strained,
        pseudo=pseudo_map,
        control=input_params["CONTROL"],
        system=input_params["SYSTEM"],
        electrons=input_params["ELECTRONS"],
        ions=input_params["IONS"],
        kpoints_grid=tuple(K_GRID),
    )
    input_file = os.path.join(deform_dir, "scf.in")
    pw_input.write_file(input_file)

    eos_info.append({
        "index": idx,
        "volume_strain": float(vol_strain),
        "volume_ratio": float(1.0 + vol_strain),
        "volume_A3": float(V_strained),
        "directory": deform_dir,
    })
    print(f"  Point {idx}: strain={vol_strain:+.4f}, V={V_strained:.4f} A^3 "
          f"({V_strained/V0*100:.1f}%)")

with open(os.path.join(WORK_DIR, "eos_info.json"), "w") as f:
    json.dump(eos_info, f, indent=2)

# Generate run script
run_script = "#!/bin/bash\n"
run_script += f"# EOS: {N_POINTS} volume points for {formula}\n\n"
for idx in range(N_POINTS):
    run_script += f"echo 'Running volume point {idx+1}/{N_POINTS}'\n"
    run_script += f"cd {WORK_DIR}/vol_{idx:03d}\n"
    run_script += f"pw.x < scf.in > scf.out 2>&1\n"
    run_script += f"cd ../..\n\n"

with open(os.path.join(WORK_DIR, "run_all.sh"), "w") as f:
    f.write(run_script)
os.chmod(os.path.join(WORK_DIR, "run_all.sh"), 0o755)

print(f"\nGenerated {N_POINTS} QE input files in {WORK_DIR}/")
print(f"Run: bash {WORK_DIR}/run_all.sh")
```

#### Step C3: Post-process, fit all 5 EOS models, and generate publication plots

```python
#!/usr/bin/env python3
"""
Post-process QE EOS outputs: extract E(V) data, fit 5 EOS models,
run convergence checks, produce publication-quality plots.
No pyiron dependency.
"""

import os
import json
import re
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

from pymatgen.core.structure import Structure
from scipy.optimize import curve_fit

# ─── CONFIGURATION ───────────────────────────────────────────────────────────
RELAXED_FILE = "relaxed_structure.cif"
WORK_DIR = "eos_qe"
OUTPUT_DIR = "eos_results_qe"
# ─────────────────────────────────────────────────────────────────────────────

os.makedirs(OUTPUT_DIR, exist_ok=True)

structure = Structure.from_file(RELAXED_FILE)
formula = structure.composition.reduced_formula
n_atoms = len(structure)

EV_PER_A3_TO_GPA = 160.21766208

# ─── EOS model definitions (same as Method A) ───────────────────────────────

def birch_murnaghan_3rd(V, E0, V0, B0, B0p):
    eta = (V0 / V) ** (2.0 / 3.0)
    return E0 + (9.0 * V0 * B0 / 16.0) * (
        (eta - 1.0)**3 * B0p + (eta - 1.0)**2 * (6.0 - 4.0 * eta))

def birch_murnaghan_4th(V, E0, V0, B0, B0p, B0pp):
    f = 0.5 * ((V0 / V) ** (2.0 / 3.0) - 1.0)
    H = B0 * B0pp + B0p**2 - 7.0 * B0p + 143.0 / 9.0
    return E0 + (9.0 / 6.0) * V0 * B0 * f**2 * (
        1.0 + (B0p - 4.0) * f + H * f**2)

def vinet_eos(V, E0, V0, B0, B0p):
    x = (V / V0) ** (1.0 / 3.0)
    eta = 1.5 * (B0p - 1.0) * (1.0 - x)
    return E0 + (2.0 * B0 * V0 / (B0p - 1.0)**2) * (
        2.0 - (5.0 + 3.0 * B0p * (x - 1.0) - 3.0 * x) * np.exp(eta))

def murnaghan_eos(V, E0, V0, B0, B0p):
    return E0 + B0 * V / B0p * (
        ((V0 / V)**B0p) / (B0p - 1.0) + 1.0) - V0 * B0 / (B0p - 1.0)

def poirier_tarantola(V, E0, V0, B0, B0p):
    x = np.log(V / V0) / 3.0
    return E0 + (9.0 / 2.0) * V0 * B0 * x**2 * (1.0 + (B0p - 2.0) * x)

def bm3_pressure(V, V0, B0, B0p):
    eta = (V0 / V) ** (2.0 / 3.0)
    return (3.0 * B0 / 2.0) * (eta**(7.0/2.0) - eta**(5.0/2.0)) * (
        1.0 + 0.75 * (B0p - 4.0) * (eta - 1.0))

def vinet_pressure(V, V0, B0, B0p):
    x = (V / V0) ** (1.0 / 3.0)
    return 3.0 * B0 * (1.0 - x) / x**2 * np.exp(
        1.5 * (B0p - 1.0) * (1.0 - x))

EOS_MODELS = {
    "birch_murnaghan_3rd": {"func": birch_murnaghan_3rd, "nparams": 4,
        "label": "Birch-Murnaghan (3rd)"},
    "birch_murnaghan_4th": {"func": birch_murnaghan_4th, "nparams": 5,
        "label": "Birch-Murnaghan (4th)"},
    "vinet": {"func": vinet_eos, "nparams": 4, "label": "Vinet (Rose)"},
    "murnaghan": {"func": murnaghan_eos, "nparams": 4, "label": "Murnaghan"},
    "poirier_tarantola": {"func": poirier_tarantola, "nparams": 4,
        "label": "Poirier-Tarantola"},
}

# ─── Parse QE outputs ───────────────────────────────────────────────────────

def parse_qe_energy_volume(output_file):
    """Extract total energy (eV) and cell volume (A^3) from QE pw.x output."""
    energy_ry = None
    volume = None
    with open(output_file) as f:
        for line in f:
            if line.strip().startswith("!"):
                match = re.search(r"total energy\s*=\s*([-\d.]+)\s*Ry", line)
                if match:
                    energy_ry = float(match.group(1))
            if "unit-cell volume" in line:
                match = re.search(r"=\s*([\d.]+)\s*\(a\.u\.\)\^3", line)
                if match:
                    volume = float(match.group(1)) * 0.14818471147
    if energy_ry is None or volume is None:
        raise ValueError(f"Could not parse energy/volume from {output_file}")
    return energy_ry * 13.605693123, volume

def parse_qe_pressure(output_file):
    """Extract pressure (GPa) from QE output."""
    pressure_kbar = None
    with open(output_file) as f:
        for line in f:
            if "P=" in line and "total   stress" in line:
                match = re.search(r"P=\s*([-\d.]+)", line)
                if match:
                    pressure_kbar = float(match.group(1))
    return pressure_kbar * 0.1 if pressure_kbar is not None else None

# Load metadata and parse outputs
with open(os.path.join(WORK_DIR, "eos_info.json")) as f:
    eos_info = json.load(f)

volumes, energies, pressures = [], [], []
for info in eos_info:
    output_file = os.path.join(info["directory"], "scf.out")
    if not os.path.exists(output_file):
        print(f"  WARNING: {output_file} not found, skipping")
        continue
    try:
        E, V = parse_qe_energy_volume(output_file)
        P = parse_qe_pressure(output_file)
        volumes.append(V)
        energies.append(E)
        pressures.append(P if P is not None else 0.0)
        print(f"  Point {info['index']}: V={V:.4f} A^3, E={E:.6f} eV"
              + (f", P={P:.3f} GPa" if P is not None else ""))
    except ValueError as e:
        print(f"  WARNING: {e}")

print(f"\nSuccessfully parsed {len(volumes)}/{len(eos_info)} points")
if len(volumes) < 5:
    raise RuntimeError("Need at least 5 data points for reliable EOS fit.")

sort_idx = np.argsort(volumes)
volumes = np.array(volumes)[sort_idx]
energies = np.array(energies)[sort_idx]
pressures = np.array(pressures)[sort_idx]
energies_per_atom = energies / n_atoms

# ─── Fit all 5 EOS models ───────────────────────────────────────────────────
print("\n" + "=" * 70)
print("EOS FITTING RESULTS (QE DFT, 5 models)")
print("=" * 70)

fit_results = {}
E0_guess = np.min(energies)
V0_guess = volumes[np.argmin(energies)]
poly_c = np.polyfit(volumes, energies, 2)
B0_guess = max(2.0 * poly_c[0] * V0_guess, 0.01)
B0p_guess = 4.0

for model_name, model_info in EOS_MODELS.items():
    try:
        func = model_info["func"]
        nparams = model_info["nparams"]
        if nparams == 4:
            p0 = [E0_guess, V0_guess, B0_guess, B0p_guess]
            bounds = ([-np.inf, volumes.min()*0.5, 0, 0],
                      [np.inf, volumes.max()*2, np.inf, 20])
        else:
            p0 = [E0_guess, V0_guess, B0_guess, B0p_guess, -0.01]
            bounds = ([-np.inf, volumes.min()*0.5, 0, 0, -1],
                      [np.inf, volumes.max()*2, np.inf, 20, 1])

        popt, pcov = curve_fit(func, volumes, energies, p0=p0, maxfev=50000,
                               bounds=bounds)
        perr = np.sqrt(np.diag(pcov))
        E_pred = func(volumes, *popt)
        ss_res = np.sum((energies - E_pred)**2)
        ss_tot = np.sum((energies - np.mean(energies))**2)
        r2 = 1.0 - ss_res / ss_tot

        result = {
            "E0_eV": float(popt[0]), "V0_A3": float(popt[1]),
            "B0_GPa": float(popt[2] * EV_PER_A3_TO_GPA),
            "B0_prime": float(popt[3]),
            "V0_err_A3": float(perr[1]),
            "B0_err_GPa": float(perr[2] * EV_PER_A3_TO_GPA),
            "B0_prime_err": float(perr[3]),
            "R_squared": float(r2),
            "RMSE_eV": float(np.sqrt(ss_res / len(energies))),
            "popt": [float(x) for x in popt],
        }
        if nparams == 5:
            result["B0_double_prime_GPa_inv"] = float(popt[4])
        fit_results[model_name] = result

        print(f"\n  {model_info['label']}:")
        print(f"    V0 = {popt[1]:.4f} +/- {perr[1]:.4f} A^3")
        print(f"    E0 = {popt[0]:.6f} eV ({popt[0]/n_atoms:.6f} eV/atom)")
        print(f"    B0 = {popt[2]*EV_PER_A3_TO_GPA:.2f} +/- "
              f"{perr[2]*EV_PER_A3_TO_GPA:.2f} GPa")
        print(f"    B0' = {popt[3]:.2f} +/- {perr[3]:.2f}")
        print(f"    R^2 = {r2:.10f}")
    except Exception as e:
        print(f"\n  {model_info['label']}: FAILED -- {e}")

# ─── Convergence checks ─────────────────────────────────────────────────────
print("\n" + "-" * 50)
print("CONVERGENCE DIAGNOSTICS")
print("-" * 50)
warnings_list = []

E_min_idx = np.argmin(energies)
if E_min_idx == 0 or E_min_idx == len(energies) - 1:
    msg = "E-V minimum at boundary -- expand volume range."
    warnings_list.append(msg)
    print(f"  [FAIL] {msg}")
else:
    print("  [OK] E-V minimum is interior.")

if len(fit_results) >= 2:
    V0s = [r["V0_A3"] for r in fit_results.values()]
    B0s = [r["B0_GPa"] for r in fit_results.values()]
    v_spread = (max(V0s) - min(V0s)) / np.mean(V0s) * 100
    b_spread = (max(B0s) - min(B0s)) / np.mean(B0s) * 100
    print(f"  V0 spread: {v_spread:.2f}%, B0 spread: {b_spread:.2f}%")
    if v_spread > 2:
        warnings_list.append(f"V0 spread {v_spread:.1f}% > 2%")
    if b_spread > 5:
        warnings_list.append(f"B0 spread {b_spread:.1f}% > 5%")

# ─── Enthalpy-Pressure data ─────────────────────────────────────────────────
enthalpies_per_atom = (energies + pressures / EV_PER_A3_TO_GPA * volumes) / n_atoms

# ─── Publication-quality plots ───────────────────────────────────────────────
fig = plt.figure(figsize=(18, 14))
gs = GridSpec(2, 2, figure=fig, hspace=0.30, wspace=0.28)
colors = {
    "birch_murnaghan_3rd": "#e41a1c", "birch_murnaghan_4th": "#ff7f00",
    "vinet": "#4daf4a", "murnaghan": "#377eb8",
    "poirier_tarantola": "#984ea3",
}
V_fine = np.linspace(volumes.min() * 0.98, volumes.max() * 1.02, 300)

# E-V
ax1 = fig.add_subplot(gs[0, 0])
ax1.plot(volumes / n_atoms, energies_per_atom, "ko", ms=7, label="QE DFT", zorder=5)
for mn, r in fit_results.items():
    if mn in EOS_MODELS:
        E_f = EOS_MODELS[mn]["func"](V_fine, *r["popt"])
        ax1.plot(V_fine / n_atoms, E_f / n_atoms, "-",
                 color=colors.get(mn, "gray"), lw=1.5,
                 label=EOS_MODELS[mn]["label"])
ax1.set_xlabel("Volume per atom ($\\AA^3$/atom)", fontsize=12)
ax1.set_ylabel("Energy per atom (eV/atom)", fontsize=12)
ax1.set_title(f"E-V: {formula} (QE DFT)", fontsize=13, fontweight="bold")
ax1.legend(fontsize=8); ax1.grid(True, alpha=0.3)

# P-V
ax2 = fig.add_subplot(gs[0, 1])
ax2.plot(volumes / n_atoms, pressures, "ko", ms=7, label="QE DFT", zorder=5)
for mn, pfunc in [("birch_murnaghan_3rd", bm3_pressure), ("vinet", vinet_pressure)]:
    if mn in fit_results:
        r = fit_results[mn]
        P_f = pfunc(V_fine, r["V0_A3"], r["B0_GPa"]/EV_PER_A3_TO_GPA,
                     r["B0_prime"]) * EV_PER_A3_TO_GPA
        ax2.plot(V_fine / n_atoms, P_f, "-", color=colors.get(mn, "gray"),
                 lw=1.5, label=EOS_MODELS[mn]["label"])
ax2.axhline(0, color="gray", ls=":", alpha=0.4)
ax2.set_xlabel("Volume per atom ($\\AA^3$/atom)", fontsize=12)
ax2.set_ylabel("Pressure (GPa)", fontsize=12)
ax2.set_title(f"P-V: {formula} (QE DFT)", fontsize=13, fontweight="bold")
ax2.legend(fontsize=9); ax2.grid(True, alpha=0.3)

# H-P
ax3 = fig.add_subplot(gs[1, 0])
ax3.plot(pressures, enthalpies_per_atom, "ko-", ms=7, lw=1.0, label=formula, zorder=5)
ax3.set_xlabel("Pressure (GPa)", fontsize=12)
ax3.set_ylabel("Enthalpy per atom (eV/atom)", fontsize=12)
ax3.set_title(f"H-P: {formula} (QE DFT)", fontsize=13, fontweight="bold")
ax3.legend(fontsize=10); ax3.grid(True, alpha=0.3)

# Residuals
ax4 = fig.add_subplot(gs[1, 1])
for mn in ["birch_murnaghan_3rd", "vinet", "murnaghan", "poirier_tarantola"]:
    if mn in fit_results and mn in EOS_MODELS:
        E_p = EOS_MODELS[mn]["func"](volumes, *fit_results[mn]["popt"])
        res_meV = (energies - E_p) * 1000 / n_atoms
        ax4.plot(volumes / n_atoms, res_meV, "o-", color=colors.get(mn, "gray"),
                 ms=5, lw=1, label=EOS_MODELS[mn]["label"])
ax4.axhline(0, color="gray", ls=":", alpha=0.4)
ax4.set_xlabel("Volume per atom ($\\AA^3$/atom)", fontsize=12)
ax4.set_ylabel("Residual (meV/atom)", fontsize=12)
ax4.set_title("EOS Fit Residuals", fontsize=13, fontweight="bold")
ax4.legend(fontsize=8); ax4.grid(True, alpha=0.3)

plt.savefig(os.path.join(OUTPUT_DIR, "eos_curves_qe.png"), dpi=200, bbox_inches="tight")
print(f"\nPlot saved to {OUTPUT_DIR}/eos_curves_qe.png")

# ─── Summary table ───────────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("SUMMARY TABLE")
print("=" * 70)
print(f"{'Model':<25} {'V0 (A^3)':<14} {'B0 (GPa)':<14} {'B0\\'':<10} {'R^2':<14}")
print("-" * 70)
for mn, r in fit_results.items():
    label = EOS_MODELS.get(mn, {}).get("label", mn)
    print(f"  {label:<23} {r['V0_A3']:<14.4f} {r['B0_GPa']:<14.2f} "
          f"{r['B0_prime']:<10.2f} {r['R_squared']:<14.10f}")

# ─── Save results ───────────────────────────────────────────────────────────
all_results = {
    "formula": formula, "n_atoms": n_atoms, "method": "QE PBE",
    "volumes_A3": volumes.tolist(), "energies_eV": energies.tolist(),
    "energies_per_atom_eV": energies_per_atom.tolist(),
    "pressures_GPa": pressures.tolist(),
    "enthalpies_per_atom_eV": enthalpies_per_atom.tolist(),
    "eos_fits": fit_results,
    "convergence_warnings": warnings_list,
}
with open(os.path.join(OUTPUT_DIR, "eos_results.json"), "w") as f:
    json.dump(all_results, f, indent=2, default=str)
print(f"Results saved to {OUTPUT_DIR}/eos_results.json")
```

### Method D: Phase Transition Analysis (Multi-phase H-P Comparison)

To identify pressure-driven phase transitions, compute EOS for competing phases and compare their enthalpy-pressure curves. The transition pressure is where H(P) curves cross.

```python
#!/usr/bin/env python3
"""
Phase transition analysis: compare H-P curves of competing phases.
Run Method A or C for each phase first, then use this script to overlay.
"""

import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from scipy.optimize import brentq
from scipy.interpolate import UnivariateSpline

# ─── CONFIGURATION ───────────────────────────────────────────────────────────
PHASE_FILES = {
    "B1 (rocksalt)": "eos_results_B1/eos_results.json",
    "B2 (cesium chloride)": "eos_results_B2/eos_results.json",
}
OUTPUT_DIR = "phase_transition_results"
REFERENCE_PHASE = "B1 (rocksalt)"      # Subtract this phase's H for clarity
# ─────────────────────────────────────────────────────────────────────────────

import os
os.makedirs(OUTPUT_DIR, exist_ok=True)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
colors = ["#e41a1c", "#377eb8", "#4daf4a", "#984ea3"]

phase_data = {}
for i, (label, filepath) in enumerate(PHASE_FILES.items()):
    with open(filepath) as f:
        data = json.load(f)

    P = np.array(data["pressures_GPa"])
    H = np.array(data["enthalpies_per_atom_eV"])
    sort = np.argsort(P)
    P, H = P[sort], H[sort]
    phase_data[label] = {"P": P, "H": H}

    ax1.plot(P, H, "o-", color=colors[i % len(colors)], ms=6, lw=1.5,
             label=label)

ax1.set_xlabel("Pressure (GPa)", fontsize=12)
ax1.set_ylabel("Enthalpy per atom (eV/atom)", fontsize=12)
ax1.set_title("H-P Curves: Phase Comparison", fontsize=13, fontweight="bold")
ax1.legend(fontsize=10)
ax1.grid(True, alpha=0.3)

# Enthalpy difference relative to reference phase
if REFERENCE_PHASE in phase_data:
    ref = phase_data[REFERENCE_PHASE]
    P_ref_spline = UnivariateSpline(ref["P"], ref["H"], s=0)

    transition_pressures = []
    for label, pdata in phase_data.items():
        if label == REFERENCE_PHASE:
            continue
        # Interpolate both to common pressure grid
        P_common = np.linspace(
            max(pdata["P"].min(), ref["P"].min()),
            min(pdata["P"].max(), ref["P"].max()),
            200
        )
        H_ref_interp = P_ref_spline(P_common)
        H_phase_spline = UnivariateSpline(pdata["P"], pdata["H"], s=0)
        H_phase_interp = H_phase_spline(P_common)
        dH = H_phase_interp - H_ref_interp

        ax2.plot(P_common, dH * 1000, "-", lw=2,
                 label=f"{label} - {REFERENCE_PHASE}")

        # Find crossing (transition pressure)
        sign_changes = np.where(np.diff(np.sign(dH)))[0]
        for sc in sign_changes:
            try:
                def dH_func(p):
                    return float(H_phase_spline(p) - P_ref_spline(p))
                P_trans = brentq(dH_func, P_common[sc], P_common[sc + 1])
                transition_pressures.append({
                    "phases": f"{label} <-> {REFERENCE_PHASE}",
                    "P_transition_GPa": float(P_trans),
                })
                ax2.axvline(P_trans, color="gray", ls="--", alpha=0.5)
                ax2.annotate(f"P$_t$ = {P_trans:.1f} GPa",
                             xy=(P_trans, 0), xytext=(10, 20),
                             textcoords="offset points", fontsize=10,
                             arrowprops=dict(arrowstyle="->"))
                print(f"  Transition: {label} <-> {REFERENCE_PHASE} "
                      f"at P = {P_trans:.2f} GPa")
            except Exception:
                pass

ax2.axhline(0, color="black", ls="-", lw=0.5)
ax2.set_xlabel("Pressure (GPa)", fontsize=12)
ax2.set_ylabel(f"$\\Delta$H (meV/atom) rel. to {REFERENCE_PHASE}", fontsize=12)
ax2.set_title("Enthalpy Difference", fontsize=13, fontweight="bold")
ax2.legend(fontsize=10)
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "phase_transition.png"), dpi=200,
            bbox_inches="tight")
print(f"\nPlot saved to {OUTPUT_DIR}/phase_transition.png")

with open(os.path.join(OUTPUT_DIR, "transition_pressures.json"), "w") as f:
    json.dump(transition_pressures, f, indent=2)
```

### Method E: Multi-Phase EOS Comparison (Standalone)

This method computes E-V curves for multiple competing crystal phases of the same material (e.g., BCC, FCC, HCP iron) in a single standalone script, overlays them on one plot, identifies the ground-state phase, and determines pressure-induced phase transitions via enthalpy-pressure (H-P) analysis with common-tangent construction. Inspired by pyiron's `Murnaghan` job class multi-phase workflow.

```python
#!/usr/bin/env python3
"""
Multi-Phase Equation of State Comparison using ASE + MACE.

Workflow (inspired by pyiron Murnaghan multi-phase):
  1. Build competing crystal phases for a given element.
  2. Relax each phase independently with MACE.
  3. Compute E-V curves for all phases.
  4. Fit Birch-Murnaghan EOS to each phase.
  5. Overlay E-V curves to identify the ground-state phase.
  6. Compute enthalpy H(P) = E + PV for each phase.
  7. Determine transition pressures from H-P crossings.
  8. Plot E-V and H-P diagrams with phase stability regions.

No pyiron dependency -- standalone script using ASE, scipy, numpy.
"""

import json
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from ase.build import bulk
from ase.optimize import LBFGS
from ase.constraints import ExpCellFilter
from scipy.optimize import curve_fit, brentq

# ─── CONFIGURATION ───────────────────────────────────────────────────────────
ELEMENT = "Fe"                         # Element to study
PHASES = {                             # Phase name -> ASE bulk kwargs
    "BCC": {"name": "Fe", "crystalstructure": "bcc", "a": 2.87},
    "FCC": {"name": "Fe", "crystalstructure": "fcc", "a": 3.60},
    "HCP": {"name": "Fe", "crystalstructure": "hcp", "a": 2.50, "c": 4.07},
}
MACE_MODEL = "medium"                  # MACE model size
VOL_RANGE = 0.06                       # +/-6% volume strain (wider for transitions)
N_POINTS = 11                          # Points per phase
FMAX_BULK = 1e-4                       # Force convergence for bulk relaxation (eV/A)
FMAX_EOS = 1e-3                        # Force convergence for strained structures (eV/A)
PRESSURE_RANGE = (0, 150)              # GPa, range for H-P analysis
OUTPUT_DIR = "eos_multiphase"
# ─────────────────────────────────────────────────────────────────────────────

os.makedirs(OUTPUT_DIR, exist_ok=True)

from mace.calculators import mace_mp
calc = mace_mp(model=MACE_MODEL, default_dtype="float64")

EV_PER_A3_TO_GPA = 160.21766208

# ─── Birch-Murnaghan EOS functions ───────────────────────────────────────────

def bm_energy(V, E0, V0, B0, B0p):
    """3rd-order Birch-Murnaghan energy E(V)."""
    eta = (V0 / V) ** (2.0 / 3.0)
    return E0 + (9.0 * V0 * B0 / 16.0) * (
        (eta - 1.0)**3 * B0p + (eta - 1.0)**2 * (6.0 - 4.0 * eta)
    )

def bm_pressure(V, V0, B0, B0p):
    """3rd-order Birch-Murnaghan pressure P(V) in eV/A^3."""
    eta = (V0 / V) ** (2.0 / 3.0)
    return (3.0 * B0 / 2.0) * (eta**(7.0/2.0) - eta**(5.0/2.0)) * (
        1.0 + 0.75 * (B0p - 4.0) * (eta - 1.0)
    )

def bm_enthalpy(P_gpa, E0, V0, B0_evA3, B0p):
    """
    Compute enthalpy H = E + PV at pressure P for a BM EOS.
    P_gpa: pressure in GPa.
    Returns H in eV (per the unit cell used in fitting).
    """
    P_evA3 = P_gpa / EV_PER_A3_TO_GPA  # GPa -> eV/A^3
    V_min = V0 * 0.5
    V_max = V0 * 1.5
    # P(V) is monotonically decreasing with V, so solve P(V) - P_target = 0
    try:
        V_at_P = brentq(lambda V: bm_pressure(V, V0, B0_evA3, B0p) - P_evA3,
                         V_min, V_max, xtol=1e-10)
    except ValueError:
        return np.nan
    E_at_P = bm_energy(V_at_P, E0, V0, B0_evA3, B0p)
    H = E_at_P + P_evA3 * V_at_P
    return H

# ─── STEP 1: Build, relax, and compute E-V for each phase ────────────────────

phase_results = {}

for phase_name, phase_kwargs in PHASES.items():
    print(f"\n{'='*60}")
    print(f"Phase: {phase_name}")
    print(f"{'='*60}")

    # Build the structure
    atoms = bulk(**phase_kwargs)
    n_atoms = len(atoms)
    atoms.calc = calc

    # Full relaxation
    ecf = ExpCellFilter(atoms, hydrostatic_strain=False)
    opt = LBFGS(ecf, logfile=os.path.join(OUTPUT_DIR, f"relax_{phase_name}.log"))
    opt.run(fmax=FMAX_BULK, steps=500)

    V0_relaxed = atoms.get_volume()
    E0_relaxed = atoms.get_potential_energy()
    print(f"  Relaxed: V = {V0_relaxed:.4f} A^3, E = {E0_relaxed:.6f} eV "
          f"({E0_relaxed/n_atoms:.6f} eV/atom)")

    # E-V curve: volume strain grid
    strain_values = np.linspace(-VOL_RANGE, VOL_RANGE, N_POINTS)

    volumes = [V0_relaxed]
    energies = [E0_relaxed]

    cell_relaxed = atoms.get_cell().copy()

    for i, vol_strain in enumerate(strain_values):
        linear_scale = (1.0 + vol_strain) ** (1.0 / 3.0)
        strained = atoms.copy()
        strained.set_cell(cell_relaxed * linear_scale, scale_atoms=True)
        strained.calc = calc

        opt = LBFGS(strained, logfile=os.devnull)
        opt.run(fmax=FMAX_EOS, steps=300)

        V = strained.get_volume()
        E = strained.get_potential_energy()
        volumes.append(V)
        energies.append(E)
        print(f"    Point {i+1}/{N_POINTS}: strain={vol_strain:+.4f}, "
              f"V={V:.4f} A^3, E/atom={E/n_atoms:.6f} eV")

    # Sort by volume
    sort_idx = np.argsort(volumes)
    volumes = np.array(volumes)[sort_idx]
    energies = np.array(energies)[sort_idx]
    energies_per_atom = energies / n_atoms

    # Fit BM EOS
    try:
        popt, pcov = curve_fit(
            bm_energy, volumes, energies,
            p0=[np.min(energies), volumes[np.argmin(energies)], 0.5, 4.0],
            maxfev=10000
        )
        E0_fit, V0_fit, B0_fit, B0p_fit = popt
        perr = np.sqrt(np.diag(pcov))
        B0_gpa = B0_fit * EV_PER_A3_TO_GPA

        print(f"  BM Fit: V0={V0_fit:.4f} A^3, E0={E0_fit:.6f} eV, "
              f"B0={B0_gpa:.1f} GPa, B0'={B0p_fit:.2f}")

        phase_results[phase_name] = {
            "n_atoms": n_atoms,
            "volumes": volumes.tolist(),
            "energies": energies.tolist(),
            "energies_per_atom": energies_per_atom.tolist(),
            "fit": {
                "E0": float(E0_fit), "V0": float(V0_fit),
                "B0_evA3": float(B0_fit), "B0_GPa": float(B0_gpa),
                "B0_prime": float(B0p_fit),
            },
        }
    except Exception as e:
        print(f"  BM fit FAILED: {e}")

# ─── STEP 2: Identify ground-state phase at P = 0 ────────────────────────────

print(f"\n{'='*60}")
print("PHASE COMPARISON")
print(f"{'='*60}")

for name, res in phase_results.items():
    fit = res["fit"]
    e_per_atom = fit["E0"] / res["n_atoms"]
    print(f"  {name:6s}: E0/atom = {e_per_atom:.6f} eV, "
          f"V0/atom = {fit['V0']/res['n_atoms']:.4f} A^3/atom, "
          f"B0 = {fit['B0_GPa']:.1f} GPa")

ground_state = min(phase_results.keys(),
                   key=lambda k: phase_results[k]["fit"]["E0"] / phase_results[k]["n_atoms"])
print(f"\n  Ground-state phase at P=0: {ground_state}")

# ─── STEP 3: Compute H(P) for each phase and find transition pressures ───────

P_array = np.linspace(PRESSURE_RANGE[0], PRESSURE_RANGE[1], 500)

enthalpy_data = {}
for name, res in phase_results.items():
    fit = res["fit"]
    n_at = res["n_atoms"]
    H_values = []
    for P in P_array:
        H = bm_enthalpy(P, fit["E0"], fit["V0"], fit["B0_evA3"], fit["B0_prime"])
        H_values.append(H / n_at if not np.isnan(H) else np.nan)
    enthalpy_data[name] = np.array(H_values)

# Find transition pressures (crossings of H-P curves)
phase_names = list(phase_results.keys())
transitions = []
for i in range(len(phase_names)):
    for j in range(i + 1, len(phase_names)):
        name_i, name_j = phase_names[i], phase_names[j]
        H_i, H_j = enthalpy_data[name_i], enthalpy_data[name_j]
        diff = H_i - H_j
        valid = ~(np.isnan(diff))
        diff_valid = diff[valid]
        P_valid = P_array[valid]
        sign_changes = np.where(np.diff(np.sign(diff_valid)))[0]
        for sc in sign_changes:
            # Linear interpolation for transition pressure
            P_trans = P_valid[sc] + (P_valid[sc+1] - P_valid[sc]) * (
                -diff_valid[sc] / (diff_valid[sc+1] - diff_valid[sc])
            )
            transitions.append({
                "phase_1": name_i, "phase_2": name_j,
                "P_transition_GPa": float(P_trans),
            })
            print(f"  Transition: {name_i} -> {name_j} at P = {P_trans:.1f} GPa")

if not transitions:
    print("  No phase transitions found in the pressure range.")

# ─── STEP 4: Visualization ───────────────────────────────────────────────────

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

colors_phase = {"BCC": "#377eb8", "FCC": "#e41a1c", "HCP": "#4daf4a",
                "SC": "#984ea3", "diamond": "#ff7f00"}

# --- Panel 1: E-V curves for all phases ---
for name, res in phase_results.items():
    fit = res["fit"]
    n_at = res["n_atoms"]
    color = colors_phase.get(name, "gray")

    # Data points
    ax1.plot(np.array(res["volumes"]) / n_at,
             res["energies_per_atom"],
             "o", color=color, markersize=6, label=f"{name} (data)", zorder=5)

    # Fitted curve
    V_fine = np.linspace(min(res["volumes"]) * 0.98,
                         max(res["volumes"]) * 1.02, 200)
    E_fine = bm_energy(V_fine, fit["E0"], fit["V0"], fit["B0_evA3"], fit["B0_prime"])
    ax1.plot(V_fine / n_at, E_fine / n_at, "-", color=color, linewidth=1.5,
             label=f"{name} (BM fit)")

ax1.set_xlabel("Volume per atom ($\\AA^3$/atom)", fontsize=12)
ax1.set_ylabel("Energy per atom (eV/atom)", fontsize=12)
ax1.set_title(f"E-V: {ELEMENT} phases", fontsize=13, fontweight="bold")
ax1.legend(fontsize=9)
ax1.grid(True, alpha=0.3)

# --- Panel 2: H-P diagram (enthalpy difference relative to ground state) ---
H_ref = enthalpy_data[ground_state]
for name in phase_results:
    color = colors_phase.get(name, "gray")
    dH = (enthalpy_data[name] - H_ref) * 1000  # meV/atom
    ax2.plot(P_array, dH, "-", color=color, linewidth=2, label=name)

# Mark transitions
for t in transitions:
    ax2.axvline(t["P_transition_GPa"], color="gray", linestyle="--", alpha=0.5)
    ax2.annotate(f"{t['phase_1']}/{t['phase_2']}\n{t['P_transition_GPa']:.0f} GPa",
                 xy=(t["P_transition_GPa"], 0),
                 xytext=(5, 15), textcoords="offset points",
                 fontsize=9, color="gray",
                 arrowprops=dict(arrowstyle="->", color="gray"))

ax2.axhline(0, color="black", linewidth=0.5)
ax2.set_xlabel("Pressure (GPa)", fontsize=12)
ax2.set_ylabel(f"$\\Delta H$ relative to {ground_state} (meV/atom)", fontsize=12)
ax2.set_title(f"H-P: {ELEMENT} phase transitions", fontsize=13, fontweight="bold")
ax2.legend(fontsize=10)
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "multiphase_eos.png"), dpi=200, bbox_inches="tight")
print(f"\nPlot saved to {OUTPUT_DIR}/multiphase_eos.png")

# ─── Save results ────────────────────────────────────────────────────────────
all_results = {
    "element": ELEMENT,
    "method": f"MACE-MP-0 ({MACE_MODEL})",
    "ground_state_phase": ground_state,
    "transitions": transitions,
    "phases": {},
}
for name, res in phase_results.items():
    all_results["phases"][name] = {
        "n_atoms": res["n_atoms"],
        "fit": res["fit"],
        "E0_per_atom_eV": res["fit"]["E0"] / res["n_atoms"],
        "V0_per_atom_A3": res["fit"]["V0"] / res["n_atoms"],
    }

with open(os.path.join(OUTPUT_DIR, "multiphase_results.json"), "w") as f:
    json.dump(all_results, f, indent=2, default=str)
print(f"Results saved to {OUTPUT_DIR}/multiphase_results.json")
```

### Method F: EOS + Debye Model for Thermal Properties

This method combines the Birch-Murnaghan EOS with the Debye-Gruneisen quasi-harmonic model to estimate finite-temperature thermodynamic properties from zero-temperature E-V data. Inspired by pyiron's `DebyeModel` integration with the `Murnaghan` job class. The Debye model approximates the phonon density of states as a single Debye cutoff frequency, which is sufficient for estimating thermal expansion, heat capacity, and the Gruneisen parameter from EOS data alone (no phonon calculation needed).

```python
#!/usr/bin/env python3
"""
EOS + Debye-Gruneisen Model for Thermal Properties.

Workflow (inspired by pyiron DebyeModel):
  1. Compute E-V data using MACE.
  2. Fit 3rd-order Birch-Murnaghan EOS.
  3. Estimate the Debye temperature from EOS parameters.
  4. Use the Debye-Gruneisen model to compute:
     - Debye temperature Theta_D(V)
     - Helmholtz free energy F(V,T)
     - Thermal expansion alpha(T)
     - Heat capacity Cv(T) and Cp(T)
     - Gruneisen parameter gamma(V)
  5. Plot thermal properties as functions of temperature.

No pyiron dependency -- standalone script using ASE, scipy, numpy.
"""

import json
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from ase.build import bulk
from ase.optimize import LBFGS
from ase.constraints import ExpCellFilter
from scipy.optimize import curve_fit, minimize_scalar
from scipy.integrate import quad

# ─── CONFIGURATION ───────────────────────────────────────────────────────────
STRUCTURE_NAME = "Cu"                  # Element or compound
CRYSTAL_STRUCTURE = "fcc"              # Crystal structure for ase.build.bulk
LATTICE_A = 3.61                       # Initial lattice parameter (A)
MACE_MODEL = "medium"                  # MACE model size
VOL_RANGE = 0.06                       # +/-6% volume strain
N_POINTS = 11                          # Number of volume points
FMAX_BULK = 1e-4                       # Force convergence for relaxation (eV/A)
FMAX_EOS = 1e-3                        # Force convergence for strained relaxation (eV/A)
TEMPERATURE_RANGE = (10, 1200)         # K, temperature range for thermal properties
N_TEMP = 100                           # Number of temperature points
MASS_AMU = 63.546                      # Atomic mass in AMU (Cu)
POISSON_RATIO = 0.34                   # Poisson ratio estimate (for Debye temp)
OUTPUT_DIR = "eos_debye"
# ─────────────────────────────────────────────────────────────────────────────

os.makedirs(OUTPUT_DIR, exist_ok=True)

from mace.calculators import mace_mp
calc = mace_mp(model=MACE_MODEL, default_dtype="float64")

# Physical constants
kB = 8.617333262e-5       # eV/K (Boltzmann constant)
kB_J = 1.380649e-23       # J/K
hbar = 1.054571817e-34    # J*s
NA = 6.02214076e23        # Avogadro
eV_to_J = 1.602176634e-19 # J/eV
EV_PER_A3_TO_GPA = 160.21766208

# ─── STEP 1: Build and relax structure ────────────────────────────────────────

atoms = bulk(STRUCTURE_NAME, crystalstructure=CRYSTAL_STRUCTURE, a=LATTICE_A)
n_atoms = len(atoms)
atoms.calc = calc

ecf = ExpCellFilter(atoms, hydrostatic_strain=False)
opt = LBFGS(ecf, logfile=os.path.join(OUTPUT_DIR, "relax.log"))
opt.run(fmax=FMAX_BULK, steps=500)

V0_relaxed = atoms.get_volume()
E0_relaxed = atoms.get_potential_energy()
print(f"Relaxed: V = {V0_relaxed:.4f} A^3, E = {E0_relaxed:.6f} eV")

# ─── STEP 2: Compute E-V data ────────────────────────────────────────────────

strain_values = np.linspace(-VOL_RANGE, VOL_RANGE, N_POINTS)

volumes = [V0_relaxed]
energies = [E0_relaxed]

cell_relaxed = atoms.get_cell().copy()

for i, vol_strain in enumerate(strain_values):
    linear_scale = (1.0 + vol_strain) ** (1.0 / 3.0)
    strained = atoms.copy()
    strained.set_cell(cell_relaxed * linear_scale, scale_atoms=True)
    strained.calc = calc
    opt = LBFGS(strained, logfile=os.devnull)
    opt.run(fmax=FMAX_EOS, steps=300)
    V = strained.get_volume()
    E = strained.get_potential_energy()
    volumes.append(V)
    energies.append(E)
    print(f"  Point {i+1}/{N_POINTS}: strain={vol_strain:+.4f}, V={V:.4f}, "
          f"E/atom={E/n_atoms:.6f}")

sort_idx = np.argsort(volumes)
volumes = np.array(volumes)[sort_idx]
energies = np.array(energies)[sort_idx]

# ─── STEP 3: Fit Birch-Murnaghan EOS ─────────────────────────────────────────

def bm_energy(V, E0, V0, B0, B0p):
    """3rd-order Birch-Murnaghan energy."""
    eta = (V0 / V) ** (2.0 / 3.0)
    return E0 + (9.0 * V0 * B0 / 16.0) * (
        (eta - 1.0)**3 * B0p + (eta - 1.0)**2 * (6.0 - 4.0 * eta)
    )

def bm_pressure(V, V0, B0, B0p):
    """3rd-order Birch-Murnaghan pressure in eV/A^3."""
    eta = (V0 / V) ** (2.0 / 3.0)
    return (3.0 * B0 / 2.0) * (eta**(7.0/2.0) - eta**(5.0/2.0)) * (
        1.0 + 0.75 * (B0p - 4.0) * (eta - 1.0)
    )

popt, pcov = curve_fit(
    bm_energy, volumes, energies,
    p0=[np.min(energies), volumes[np.argmin(energies)], 0.5, 4.0],
    maxfev=10000
)
E0_fit, V0_fit, B0_fit, B0p_fit = popt
B0_gpa = B0_fit * EV_PER_A3_TO_GPA

print(f"\nBM Fit: V0={V0_fit:.4f} A^3, E0={E0_fit:.6f} eV, "
      f"B0={B0_gpa:.1f} GPa, B0'={B0p_fit:.2f}")

# ─── STEP 4: Debye-Gruneisen model ───────────────────────────────────────────

def debye_temperature_from_eos(V, V0, B0_evA3, B0p, mass_amu, n_atoms, nu):
    """
    Estimate Debye temperature using the Moruzzi-Janak-Schwarz approach.

    Theta_D is derived from the mean sound velocity, which is estimated
    from the bulk modulus and Poisson ratio.

    V, V0 in A^3; B0 in eV/A^3; mass_amu per atom; nu = Poisson ratio.
    """
    # Scale B to volume V using approximate power-law scaling
    B_at_V = B0_evA3 * (V0 / V) ** (B0p)

    # Convert to Pa
    B_Pa = B_at_V * eV_to_J / 1e-30  # eV/A^3 -> J/m^3 = Pa

    # Density
    total_mass_kg = n_atoms * mass_amu * 1.66054e-27  # kg
    V_m3 = V * 1e-30  # A^3 -> m^3
    rho = total_mass_kg / V_m3  # kg/m^3

    # Shear modulus from Poisson ratio: G = 3B(1-2nu) / (2(1+nu))
    G = 3.0 * B_Pa * (1.0 - 2.0 * nu) / (2.0 * (1.0 + nu))

    # Sound velocities
    v_l = np.sqrt((B_Pa + 4.0 * G / 3.0) / rho)  # longitudinal
    v_t = np.sqrt(G / rho)                         # transverse

    # Mean sound velocity: 1/v_m^3 = (1/3)(1/v_l^3 + 2/v_t^3)
    v_m = (1.0 / 3.0 * (1.0 / v_l**3 + 2.0 / v_t**3)) ** (-1.0 / 3.0)

    # Debye temperature
    n_density = n_atoms / V_m3  # atoms per m^3
    theta_D = (hbar / kB_J) * (6.0 * np.pi**2 * n_density) ** (1.0 / 3.0) * v_m

    return theta_D


def gruneisen_parameter(V, V0, B0p):
    """
    Gruneisen parameter from Slater approximation:
    gamma_0 = (B0' - 1) / 2 - 1/6

    Volume-dependent: gamma(V) = gamma_0 * (V0/V)^q with q ~ 1.
    """
    gamma_0 = (B0p - 1.0) / 2.0 - 1.0 / 6.0
    q = 1.0  # Slater approximation
    return gamma_0 * (V0 / V) ** q


def debye_function_3(x):
    """
    Debye function D_3(x) = (3/x^3) * integral_0^x [t^3 / (e^t - 1)] dt.
    """
    if x < 1e-6:
        return 1.0  # D_3(0) = 1
    integrand = lambda t: t**3 / (np.exp(t) - 1.0) if t > 0 else 0.0
    result, _ = quad(integrand, 1e-12, x, limit=200)
    return 3.0 / x**3 * result


def debye_internal_energy(T, theta_D, n_atoms):
    """Internal vibrational energy E_vib(T) in eV per cell."""
    if T < 1e-3:
        return (9.0 / 8.0) * n_atoms * kB * theta_D  # zero-point energy
    x = theta_D / T
    D3 = debye_function_3(x)
    return 3.0 * n_atoms * kB * T * D3 + (9.0 / 8.0) * n_atoms * kB * theta_D


def debye_entropy(T, theta_D, n_atoms):
    """Vibrational entropy S(T) in eV/K per cell."""
    if T < 1e-3:
        return 0.0
    x = theta_D / T
    D3 = debye_function_3(x)
    return n_atoms * kB * (-3.0 * np.log(1.0 - np.exp(-x)) + 4.0 * D3)


def debye_cv(T, theta_D, n_atoms):
    """Heat capacity Cv(T) in eV/K per cell."""
    if T < 1e-3:
        return 0.0
    x = theta_D / T
    D3 = debye_function_3(x)
    return 3.0 * n_atoms * kB * (4.0 * D3 - 3.0 * x / (np.exp(x) - 1.0))


def helmholtz_free_energy(V, T, E0, V0, B0, B0p, mass_amu, n_atoms, nu):
    """
    Total Helmholtz free energy F(V, T) = E_cold(V) + F_vib(V, T).
    E_cold from BM EOS, F_vib from Debye model.
    """
    E_cold = bm_energy(V, E0, V0, B0, B0p)
    theta_D = debye_temperature_from_eos(V, V0, B0, B0p, mass_amu, n_atoms, nu)
    E_vib = debye_internal_energy(T, theta_D, n_atoms)
    S_vib = debye_entropy(T, theta_D, n_atoms)
    F_vib = E_vib - T * S_vib
    return E_cold + F_vib


# ─── STEP 5: Compute thermal properties ──────────────────────────────────────

temperatures = np.linspace(TEMPERATURE_RANGE[0], TEMPERATURE_RANGE[1], N_TEMP)

# Debye temperature and Gruneisen parameter at V0
theta_D_0 = debye_temperature_from_eos(
    V0_fit, V0_fit, B0_fit, B0p_fit, MASS_AMU, n_atoms, POISSON_RATIO
)
gamma_0 = gruneisen_parameter(V0_fit, V0_fit, B0p_fit)
print(f"\nDebye temperature at V0: {theta_D_0:.1f} K")
print(f"Gruneisen parameter at V0: {gamma_0:.3f}")

# For each temperature, find the equilibrium volume by minimizing F(V, T)
V_eq_T = []
Cv_T = []
theta_D_T = []

for T in temperatures:
    result = minimize_scalar(
        lambda V: helmholtz_free_energy(
            V, T, E0_fit, V0_fit, B0_fit, B0p_fit, MASS_AMU, n_atoms, POISSON_RATIO
        ),
        bounds=(V0_fit * 0.85, V0_fit * 1.15),
        method="bounded"
    )
    V_eq = result.x
    V_eq_T.append(V_eq)

    theta_D = debye_temperature_from_eos(
        V_eq, V0_fit, B0_fit, B0p_fit, MASS_AMU, n_atoms, POISSON_RATIO
    )
    theta_D_T.append(theta_D)

    Cv = debye_cv(T, theta_D, n_atoms)
    Cv_T.append(Cv)

V_eq_T = np.array(V_eq_T)
Cv_T = np.array(Cv_T)
theta_D_T = np.array(theta_D_T)

# Thermal expansion: alpha = (1/V)(dV/dT)
dV_dT = np.gradient(V_eq_T, temperatures)
alpha_T = dV_dT / V_eq_T  # 1/K

# Cp = Cv + alpha^2 * B * V * T
# Approximate bulk modulus at each T from numerical derivative
B_T = np.array([
    -(bm_pressure(V + 0.01, V0_fit, B0_fit, B0p_fit) -
      bm_pressure(V - 0.01, V0_fit, B0_fit, B0p_fit)) / (0.02 / V)
    for V in V_eq_T
])  # B = -V * dP/dV in eV/A^3
Cp_T = Cv_T + alpha_T**2 * B_T * V_eq_T * temperatures

# Convert to J/(mol*K) for standard units
Cv_JmolK = Cv_T * eV_to_J * NA / n_atoms  # per mol of atoms
Cp_JmolK = Cp_T * eV_to_J * NA / n_atoms

# Dulong-Petit limit
DP_limit = 3.0 * kB_J * NA  # ~24.94 J/(mol*K)

print(f"\nThermal properties computed for {N_TEMP} temperatures from "
      f"{TEMPERATURE_RANGE[0]} to {TEMPERATURE_RANGE[1]} K")

# ─── STEP 6: Visualization ───────────────────────────────────────────────────

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Panel 1: E-V with BM fit
ax = axes[0, 0]
ax.plot(volumes / n_atoms, energies / n_atoms, "ko", markersize=6, label="MACE data")
V_fine = np.linspace(volumes.min() * 0.98, volumes.max() * 1.02, 200)
E_fine = bm_energy(V_fine, E0_fit, V0_fit, B0_fit, B0p_fit)
ax.plot(V_fine / n_atoms, E_fine / n_atoms, "r-", linewidth=1.5, label="BM fit")
ax.set_xlabel("Volume per atom ($\\AA^3$/atom)", fontsize=11)
ax.set_ylabel("Energy per atom (eV/atom)", fontsize=11)
ax.set_title(f"E-V: {STRUCTURE_NAME} ({CRYSTAL_STRUCTURE})", fontsize=12,
             fontweight="bold")
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3)

# Panel 2: Debye temperature and Gruneisen parameter vs T
ax = axes[0, 1]
ax.plot(temperatures, theta_D_T, "b-", linewidth=2, label="$\\Theta_D(T)$")
ax.set_xlabel("Temperature (K)", fontsize=11)
ax.set_ylabel("Debye temperature (K)", fontsize=11, color="blue")
ax.tick_params(axis="y", labelcolor="blue")

ax2 = ax.twinx()
gamma_T = np.array([gruneisen_parameter(V, V0_fit, B0p_fit) for V in V_eq_T])
ax2.plot(temperatures, gamma_T, "r--", linewidth=2, label="$\\gamma(T)$")
ax2.set_ylabel("Gruneisen parameter", fontsize=11, color="red")
ax2.tick_params(axis="y", labelcolor="red")

ax.set_title("Debye temperature & Gruneisen parameter", fontsize=12,
             fontweight="bold")
lines1, labels1 = ax.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax.legend(lines1 + lines2, labels1 + labels2, fontsize=10, loc="center right")
ax.grid(True, alpha=0.3)

# Panel 3: Heat capacity Cv and Cp
ax = axes[1, 0]
ax.plot(temperatures, Cv_JmolK, "b-", linewidth=2, label="$C_v$")
ax.plot(temperatures, Cp_JmolK, "r-", linewidth=2, label="$C_p$")
ax.axhline(DP_limit, color="gray", linestyle="--", alpha=0.5, label="Dulong-Petit")
ax.set_xlabel("Temperature (K)", fontsize=11)
ax.set_ylabel("Heat capacity (J/mol/K)", fontsize=11)
ax.set_title("Heat capacity", fontsize=12, fontweight="bold")
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3)
ax.set_ylim(0, max(DP_limit * 1.3, np.nanmax(Cp_JmolK) * 1.1))

# Panel 4: Thermal expansion
ax = axes[1, 1]
ax.plot(temperatures, alpha_T * 1e6, "g-", linewidth=2)
ax.set_xlabel("Temperature (K)", fontsize=11)
ax.set_ylabel("Thermal expansion ($\\times 10^{-6}$ K$^{-1}$)", fontsize=11)
ax.set_title("Thermal expansion coefficient", fontsize=12, fontweight="bold")
ax.grid(True, alpha=0.3)

plt.suptitle(f"Thermal Properties: {STRUCTURE_NAME} ({CRYSTAL_STRUCTURE}) -- "
             f"Debye-Gruneisen Model", fontsize=14, y=1.02)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "debye_thermal.png"), dpi=200, bbox_inches="tight")
print(f"\nPlot saved to {OUTPUT_DIR}/debye_thermal.png")

# ─── Save results ────────────────────────────────────────────────────────────
all_results = {
    "material": STRUCTURE_NAME,
    "crystal_structure": CRYSTAL_STRUCTURE,
    "method": f"MACE-MP-0 ({MACE_MODEL}) + Debye-Gruneisen",
    "n_atoms": n_atoms,
    "eos_fit": {
        "V0_A3": float(V0_fit),
        "E0_eV": float(E0_fit),
        "B0_GPa": float(B0_gpa),
        "B0_prime": float(B0p_fit),
    },
    "debye": {
        "theta_D_at_V0_K": float(theta_D_0),
        "gruneisen_at_V0": float(gamma_0),
        "poisson_ratio_input": POISSON_RATIO,
    },
    "thermal": {
        "temperatures_K": temperatures.tolist(),
        "V_eq_A3": V_eq_T.tolist(),
        "theta_D_K": theta_D_T.tolist(),
        "Cv_JmolK": Cv_JmolK.tolist(),
        "Cp_JmolK": Cp_JmolK.tolist(),
        "alpha_1perK": alpha_T.tolist(),
    },
}
with open(os.path.join(OUTPUT_DIR, "debye_results.json"), "w") as f:
    json.dump(all_results, f, indent=2, default=str)
print(f"Results saved to {OUTPUT_DIR}/debye_results.json")
```

### Method G: LAMMPS vs MACE Comparison for Large Systems

This method runs LAMMPS with an EAM/MEAM potential and MACE on the same material, overlaying both E-V curves on a single plot to validate MACE predictions against well-established classical potentials. Useful for studying system-size effects and for benchmarking MACE in domains where classical potentials are well-tested.

```python
#!/usr/bin/env python3
"""
EOS comparison: LAMMPS (EAM potential) vs MACE on the same material.

Workflow:
  1. Build a supercell.
  2. Compute E-V with LAMMPS (EAM potential) on the supercell.
  3. Compute E-V with MACE on the primitive cell (for speed).
  4. Fit Birch-Murnaghan EOS to both.
  5. Compare V0, B0, B0' between LAMMPS and MACE.
  6. Overlay E-V curves and plot residuals.

No pyiron dependency -- standalone script using ASE, scipy, numpy.
"""

import json
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from ase.build import bulk
from ase.optimize import LBFGS
from ase.constraints import ExpCellFilter
from scipy.optimize import curve_fit

# ─── CONFIGURATION ───────────────────────────────────────────────────────────
ELEMENT = "Cu"
CRYSTAL_STRUCTURE = "fcc"
LATTICE_A = 3.615                      # Initial lattice parameter (A)
SUPERCELL = (3, 3, 3)                  # Supercell for LAMMPS (3x3x3 = 108 atoms)
MACE_MODEL = "medium"                  # MACE model size

# LAMMPS EAM potential configuration
# Download EAM file from NIST: https://www.ctcms.nist.gov/potentials/
LAMMPS_EAM_FILE = "Cu_mishin.eam.alloy"  # Path to EAM potential file
LAMMPS_PAIR_STYLE = "eam/alloy"
LAMMPS_ELEMENT_LIST = ["Cu"]

VOL_RANGE = 0.05                       # +/-5% volume strain
N_POINTS = 11
FMAX_BULK = 1e-4
FMAX_EOS = 1e-3
OUTPUT_DIR = "eos_lammps_mace_comparison"
# ─────────────────────────────────────────────────────────────────────────────

os.makedirs(OUTPUT_DIR, exist_ok=True)

EV_PER_A3_TO_GPA = 160.21766208

# ─── Birch-Murnaghan function ────────────────────────────────────────────────

def bm_energy(V, E0, V0, B0, B0p):
    eta = (V0 / V) ** (2.0 / 3.0)
    return E0 + (9.0 * V0 * B0 / 16.0) * (
        (eta - 1.0)**3 * B0p + (eta - 1.0)**2 * (6.0 - 4.0 * eta)
    )

# ─── Helper: compute E-V for any calculator ──────────────────────────────────

def compute_ev_curve(atoms_template, calc, label, n_points=N_POINTS,
                     vol_range=VOL_RANGE, fmax_bulk=FMAX_BULK,
                     fmax_eos=FMAX_EOS):
    """
    Full EOS workflow: relax + volume scan + collect E(V).
    Returns (volumes, energies, n_atoms, fit_params).
    """
    atoms = atoms_template.copy()
    atoms.calc = calc

    # Full relaxation
    ecf = ExpCellFilter(atoms, hydrostatic_strain=False)
    opt = LBFGS(ecf, logfile=os.path.join(OUTPUT_DIR, f"relax_{label}.log"))
    opt.run(fmax=fmax_bulk, steps=500)

    V0 = atoms.get_volume()
    E0 = atoms.get_potential_energy()
    n_atoms = len(atoms)
    print(f"  {label} relaxed: V={V0:.4f} A^3, E/atom={E0/n_atoms:.6f} eV")

    # Volume scan
    strain_values = np.linspace(-vol_range, vol_range, n_points)

    volumes = [V0]
    energies = [E0]
    cell_relaxed = atoms.get_cell().copy()

    for i, vol_strain in enumerate(strain_values):
        linear_scale = (1.0 + vol_strain) ** (1.0 / 3.0)
        strained = atoms.copy()
        strained.set_cell(cell_relaxed * linear_scale, scale_atoms=True)
        strained.calc = calc
        opt = LBFGS(strained, logfile=os.devnull)
        opt.run(fmax=fmax_eos, steps=300)
        V = strained.get_volume()
        E = strained.get_potential_energy()
        volumes.append(V)
        energies.append(E)

    sort_idx = np.argsort(volumes)
    volumes = np.array(volumes)[sort_idx]
    energies = np.array(energies)[sort_idx]

    # Fit BM EOS
    try:
        popt, pcov = curve_fit(
            bm_energy, volumes, energies,
            p0=[np.min(energies), volumes[np.argmin(energies)], 0.5, 4.0],
            maxfev=10000
        )
        E0_fit, V0_fit, B0_fit, B0p_fit = popt
        B0_gpa = B0_fit * EV_PER_A3_TO_GPA
        fit_params = {
            "E0": float(E0_fit), "V0": float(V0_fit),
            "B0_evA3": float(B0_fit), "B0_GPa": float(B0_gpa),
            "B0_prime": float(B0p_fit),
        }
        print(f"  {label} BM fit: V0={V0_fit:.4f}, B0={B0_gpa:.1f} GPa, "
              f"B0'={B0p_fit:.2f}")
    except Exception as e:
        print(f"  {label} BM fit FAILED: {e}")
        fit_params = None

    return volumes, energies, n_atoms, fit_params

# ─── STEP 1: LAMMPS calculation ──────────────────────────────────────────────

print("="*60)
print("LAMMPS EAM calculation")
print("="*60)

atoms_lammps = bulk(ELEMENT, crystalstructure=CRYSTAL_STRUCTURE, a=LATTICE_A)
atoms_lammps = atoms_lammps.repeat(SUPERCELL)
print(f"Supercell: {SUPERCELL}, {len(atoms_lammps)} atoms")

lammps_available = True
try:
    from ase.calculators.lammpslib import LAMMPSlib

    lammps_cmds = [
        f"pair_style {LAMMPS_PAIR_STYLE}",
        f"pair_coeff * * {os.path.abspath(LAMMPS_EAM_FILE)} "
        f"{' '.join(LAMMPS_ELEMENT_LIST)}",
    ]

    calc_lammps = LAMMPSlib(
        lmpcmds=lammps_cmds,
        atom_types={el: i + 1 for i, el in enumerate(LAMMPS_ELEMENT_LIST)},
        log_file=os.path.join(OUTPUT_DIR, "lammps.log"),
        keep_alive=True,
    )

    V_lammps, E_lammps, n_atoms_lammps, fit_lammps = compute_ev_curve(
        atoms_lammps, calc_lammps, "LAMMPS_EAM"
    )
except ImportError:
    print("WARNING: LAMMPSlib not available. Install lammps Python bindings.")
    print("  pip install lammps   (or build from source)")
    lammps_available = False
    V_lammps = E_lammps = None
    n_atoms_lammps = len(atoms_lammps)
    fit_lammps = None
except Exception as e:
    print(f"WARNING: LAMMPS calculation failed: {e}")
    print("  Make sure the EAM potential file exists and is valid.")
    lammps_available = False
    V_lammps = E_lammps = None
    n_atoms_lammps = len(atoms_lammps)
    fit_lammps = None

# ─── STEP 2: MACE calculation (primitive cell) ───────────────────────────────

print(f"\n{'='*60}")
print("MACE calculation")
print("="*60)

from mace.calculators import mace_mp
calc_mace = mace_mp(model=MACE_MODEL, default_dtype="float64")

atoms_mace = bulk(ELEMENT, crystalstructure=CRYSTAL_STRUCTURE, a=LATTICE_A)
print(f"Primitive cell: {len(atoms_mace)} atoms")

V_mace, E_mace, n_atoms_mace, fit_mace = compute_ev_curve(
    atoms_mace, calc_mace, "MACE"
)

# ─── STEP 3: Comparison table ────────────────────────────────────────────────

print(f"\n{'='*60}")
print("COMPARISON: LAMMPS EAM vs MACE")
print("="*60)
print(f"{'Property':25s} {'LAMMPS EAM':>15s} {'MACE':>15s} {'Diff':>10s}")
print("-"*65)

if fit_lammps and fit_mace:
    for prop, key in [("V0/atom (A^3)", "V0"), ("B0 (GPa)", "B0_GPa"),
                      ("B0'", "B0_prime")]:
        val_l = fit_lammps[key] / n_atoms_lammps if key == "V0" else fit_lammps[key]
        val_m = fit_mace[key] / n_atoms_mace if key == "V0" else fit_mace[key]
        diff = val_m - val_l
        print(f"  {prop:23s} {val_l:15.4f} {val_m:15.4f} {diff:+10.4f}")
elif fit_mace:
    print("  LAMMPS results not available. Showing MACE only.")
    for prop, key in [("V0/atom (A^3)", "V0"), ("B0 (GPa)", "B0_GPa"),
                      ("B0'", "B0_prime")]:
        val_m = fit_mace[key] / n_atoms_mace if key == "V0" else fit_mace[key]
        print(f"  {prop:23s} {'N/A':>15s} {val_m:15.4f}")

# ─── STEP 4: Visualization ───────────────────────────────────────────────────

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

# --- Panel 1: E-V comparison (per atom, shifted to common reference) ---
if V_lammps is not None:
    E_lammps_shifted = E_lammps / n_atoms_lammps
    E_lammps_shifted -= np.min(E_lammps_shifted)
    ax1.plot(V_lammps / n_atoms_lammps, E_lammps_shifted * 1000,
             "bs", markersize=6, label="LAMMPS EAM (data)", zorder=5)

    if fit_lammps:
        V_fine_l = np.linspace(V_lammps.min() * 0.98, V_lammps.max() * 1.02, 200)
        E_fine_l = bm_energy(V_fine_l, fit_lammps["E0"], fit_lammps["V0"],
                              fit_lammps["B0_evA3"], fit_lammps["B0_prime"])
        E_fine_l_shifted = E_fine_l / n_atoms_lammps
        E_fine_l_shifted -= np.min(E_fine_l_shifted)
        ax1.plot(V_fine_l / n_atoms_lammps, E_fine_l_shifted * 1000,
                 "b-", linewidth=1.5, label="LAMMPS EAM (BM fit)")

E_mace_shifted = E_mace / n_atoms_mace
E_mace_shifted -= np.min(E_mace_shifted)
ax1.plot(V_mace / n_atoms_mace, E_mace_shifted * 1000,
         "ro", markersize=6, label="MACE (data)", zorder=5)

if fit_mace:
    V_fine_m = np.linspace(V_mace.min() * 0.98, V_mace.max() * 1.02, 200)
    E_fine_m = bm_energy(V_fine_m, fit_mace["E0"], fit_mace["V0"],
                          fit_mace["B0_evA3"], fit_mace["B0_prime"])
    E_fine_m_shifted = E_fine_m / n_atoms_mace
    E_fine_m_shifted -= np.min(E_fine_m_shifted)
    ax1.plot(V_fine_m / n_atoms_mace, E_fine_m_shifted * 1000,
             "r-", linewidth=1.5, label="MACE (BM fit)")

ax1.set_xlabel("Volume per atom ($\\AA^3$/atom)", fontsize=12)
ax1.set_ylabel("$\\Delta E$ per atom (meV/atom)", fontsize=12)
ax1.set_title(f"E-V: {ELEMENT} ({CRYSTAL_STRUCTURE}) -- LAMMPS vs MACE",
              fontsize=13, fontweight="bold")
ax1.legend(fontsize=10)
ax1.grid(True, alpha=0.3)

# --- Panel 2: Residuals (MACE - LAMMPS) if both available ---
if fit_lammps and fit_mace:
    V_per_atom = np.linspace(
        max(V_lammps.min()/n_atoms_lammps, V_mace.min()/n_atoms_mace) * 1.01,
        min(V_lammps.max()/n_atoms_lammps, V_mace.max()/n_atoms_mace) * 0.99,
        200
    )
    E_l_fit = bm_energy(V_per_atom * n_atoms_lammps, fit_lammps["E0"],
                         fit_lammps["V0"], fit_lammps["B0_evA3"],
                         fit_lammps["B0_prime"]) / n_atoms_lammps
    E_m_fit = bm_energy(V_per_atom * n_atoms_mace, fit_mace["E0"],
                         fit_mace["V0"], fit_mace["B0_evA3"],
                         fit_mace["B0_prime"]) / n_atoms_mace

    # Align at minimum (different E0 references)
    E_l_fit -= np.min(E_l_fit)
    E_m_fit -= np.min(E_m_fit)
    residual = (E_m_fit - E_l_fit) * 1000  # meV/atom

    ax2.plot(V_per_atom, residual, "k-", linewidth=2)
    ax2.axhline(0, color="gray", linestyle=":", alpha=0.5)
    ax2.set_xlabel("Volume per atom ($\\AA^3$/atom)", fontsize=12)
    ax2.set_ylabel("$\\Delta E$ MACE - LAMMPS (meV/atom)", fontsize=12)
    ax2.set_title("E-V residual (relative to minima)", fontsize=13,
                  fontweight="bold")
    ax2.grid(True, alpha=0.3)
else:
    ax2.text(0.5, 0.5, "LAMMPS data not available\nfor comparison",
             ha="center", va="center", transform=ax2.transAxes, fontsize=14)
    ax2.set_title("Residual (not available)", fontsize=13, fontweight="bold")

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "lammps_mace_comparison.png"),
            dpi=200, bbox_inches="tight")
print(f"\nPlot saved to {OUTPUT_DIR}/lammps_mace_comparison.png")

# ─── Save results ────────────────────────────────────────────────────────────
all_results = {
    "element": ELEMENT,
    "crystal_structure": CRYSTAL_STRUCTURE,
    "supercell": list(SUPERCELL),
    "lammps": {
        "potential": LAMMPS_PAIR_STYLE,
        "n_atoms": n_atoms_lammps,
        "available": lammps_available,
        "fit": fit_lammps,
        "V0_per_atom": fit_lammps["V0"] / n_atoms_lammps if fit_lammps else None,
    },
    "mace": {
        "model": MACE_MODEL,
        "n_atoms": n_atoms_mace,
        "fit": fit_mace,
        "V0_per_atom": fit_mace["V0"] / n_atoms_mace if fit_mace else None,
    },
}
with open(os.path.join(OUTPUT_DIR, "lammps_mace_comparison.json"), "w") as f:
    json.dump(all_results, f, indent=2, default=str)
print(f"Results saved to {OUTPUT_DIR}/lammps_mace_comparison.json")
```

## Key Parameters

| Parameter | Typical Value | Notes |
|---|---|---|
| Volume range (`VOL_RANGE`) | 0.05 (+/-5% volume) | Wider for high-pressure studies (0.10). Narrower for soft materials (0.03). Corresponds to ~+/-1.6% linear strain per axis. |
| Number of points (`N_POINTS`) | 11 (pyiron default) | More points improve fit quality. Minimum 5 for 4-parameter fits. 11 recommended for 5-parameter BM4. Odd number centers on V0. |
| Axes (`AXES`) | ("x", "y", "z") | For isotropic strain on all axes. Can restrict to ("x",) for uniaxial studies. |
| ecutwfc (QE) | 60--80 Ry | Must be converged. Same cutoff for all volume points. |
| k-grid (QE) | Dense, consistent | Same k-grid for all points to avoid systematic errors. |
| conv_thr (QE) | 1e-10 Ry | Tight convergence: E-V differences can be in the meV range. |
| EOS model (primary) | Birch-Murnaghan (3rd) | Most widely used. Vinet better for very compressed/expanded volumes. BM4 adds B0'' for extreme ranges. Murnaghan less accurate at large compressions. |
| FMAX (MACE/LAMMPS) | 1e-3 to 1e-4 eV/A | Tighter for equilibrium (1e-4), can be looser for strained (1e-3). |
| Competing phases (Method E) | All plausible crystal structures | Include BCC, FCC, HCP at minimum for metals. Add SC, diamond-cubic, etc. if relevant. Use experimental knowledge to select candidate phases. |
| Pressure range (Method E) | 0--150 GPa | Extend for ultra-high-pressure studies. Common-tangent construction requires H(P) to be well-defined across this range. |
| Poisson ratio (Method F) | 0.25--0.35 | Used to estimate Debye temperature from bulk modulus. Use experimental value if available. Metals: ~0.30--0.35; ceramics: ~0.20--0.25. |
| Temperature range (Method F) | 10--1200 K | Debye model is most accurate below the Debye temperature. At very high T, anharmonic effects dominate and the model becomes unreliable. |
| Gruneisen exponent q (Method F) | 1.0 | Exponent in gamma(V) = gamma_0 * (V0/V)^q. q = 1 (Slater), q = 2/3 (Dugdale-MacDonald), q = 1/2 (free-volume). |
| Atomic mass (Method F) | Element-specific AMU | Required for Debye temperature calculation. Use average atomic mass for multi-component systems. |
| LAMMPS potential (Method G) | EAM/alloy or MEAM | Must match the element. Download from NIST Interatomic Potentials Repository. Validate against known experimental properties. |
| Supercell size (Method G) | 3x3x3 to 5x5x5 | Larger supercells reduce finite-size effects but increase LAMMPS runtime. For EOS, 3x3x3 is usually sufficient. |

## EOS Models Reference

| Model | Parameters | Best for | Notes |
|---|---|---|---|
| Birch-Murnaghan (3rd) | E0, V0, B0, B0' | General use, moderate compression | Most widely cited, derived from finite strain theory |
| Birch-Murnaghan (4th) | E0, V0, B0, B0', B0'' | Wide volume ranges | Adds curvature correction; needs >= 7 data points |
| Vinet (Rose) | E0, V0, B0, B0' | Extreme compression, metals | Universal form; often preferred for high-pressure work |
| Murnaghan | E0, V0, B0, B0' | Small compressions only | Simplest form; diverges from BM/Vinet at large strain |
| Poirier-Tarantola | E0, V0, B0, B0' | Logarithmic strain regime | Based on natural strain; useful comparison |
| Polynomial (order 3) | Coefficients | Quick estimate, fallback | No physical parameters; used for initial V0 guess |

## Interpreting Results

**Equilibrium volume (V0):** The volume at the E-V minimum. Compare with experiment. PBE typically overestimates by 1--3%.

**Bulk modulus (B0):** Resistance to uniform compression. Units: GPa. Reference values:
- Metals: 30--400 GPa (Na ~7, Fe ~170, W ~310, diamond ~440)
- Semiconductors: 50--100 GPa (Si ~98, GaAs ~75)
- If B0 differs significantly between MACE and QE, trust QE.

**Pressure derivative (B0'):** Dimensionless, typically 3.5--6 for most solids. B0' ~ 4 is a common default. Values outside 2--8 may indicate fitting issues.

**Cohesive energy:** Energy per atom gained by forming the solid from isolated atoms. Compare with experimental sublimation enthalpy (adding zero-point energy corrections). PBE overbinds by ~0.1-0.5 eV/atom for many materials.

**Comparing EOS models:** If BM3, Vinet, and Murnaghan give consistent V0 and B0 (within 1--2%), the fit is robust. Large disagreements indicate the data range may be too wide or too narrow, or the data has noise. The residuals plot (Panel 4) reveals systematic deviations.

**E-V curve shape:** Should be a smooth parabola-like curve with a clear minimum. Kinks or scatter indicate convergence problems in individual calculations.

**Phase transitions (H-P curves):** The phase with the lowest enthalpy H = E + PV at a given pressure is thermodynamically stable. Where H-P curves of two phases cross defines the transition pressure P_t. At P < P_t the phase with lower H is stable; at P > P_t the other phase becomes stable. For multi-phase comparisons (Method E), always normalize to per-atom quantities before comparing phases with different numbers of atoms in the unit cell.

**Ground-state phase identification:** The phase with the lowest E0/atom at P = 0 is the ground-state phase. For iron, MACE typically predicts BCC as the ground state, consistent with experiment. If MACE predicts the wrong ground state, this indicates the model is unreliable for phase stability of that element.

**Debye temperature (Method F):** The Debye temperature Theta_D characterizes the phonon spectrum. Reference values: Cu ~343 K, Al ~428 K, Fe ~470 K, Si ~645 K, diamond ~2230 K. Debye temperatures estimated from the EOS-based Moruzzi-Janak-Schwarz approach are approximate (typically within 10--20% of experimental values). The main source of error is the assumed Poisson ratio and the single-frequency approximation.

**Gruneisen parameter (Method F):** The dimensionless Gruneisen parameter gamma describes how the phonon frequencies scale with volume. Typical range: 1.0--2.5 for most solids. The Slater approximation gamma = (B0' - 1)/2 - 1/6 gives a simple estimate from EOS data. Values outside 0.5--3.0 may indicate fitting issues.

**Thermal expansion (Method F):** The Debye-Gruneisen model typically overestimates thermal expansion at high temperatures due to neglect of anharmonic effects. Compare with experimental data below the Debye temperature for best accuracy. The model predicts linear thermal expansion in the high-T limit.

**Heat capacity (Method F):** Cv should approach the Dulong-Petit limit (3R ~ 24.94 J/mol/K) at high temperatures. Cp > Cv due to the thermal expansion contribution. If Cv exceeds the Dulong-Petit limit significantly, check the Debye temperature estimate.

**LAMMPS vs MACE comparison (Method G):** Agreement within 5--10% on B0 and 1--2% on V0 is typical for well-parameterized EAM potentials vs MACE. Large disagreements suggest either the EAM potential is poorly fitted for EOS properties or MACE has limitations for that material. The E-V residual plot reveals whether discrepancies are systematic (e.g., different curvature) or random.

## Convergence Checklist

The scripts include automated convergence checks. Here is what they verify:

1. **Interior minimum**: The E-V minimum must not be at the boundary of the volume range. If it is, the equilibrium may lie outside -- expand `VOL_RANGE`.
2. **Point density**: At least 7 data points for 4-parameter fits, 9+ for BM4 (5-parameter). The default 11 points is sufficient.
3. **Model agreement**: V0 should agree within 2% across models; B0 within 5%. Larger spreads indicate data quality issues.
4. **B0' range**: Values of B0' outside [2, 8] suggest fitting problems or insufficient data.
5. **Energy span**: The E-V curve should span at least 50 meV/atom to provide enough curvature for fitting.
6. **V0 consistency**: The fitted V0 should be within ~1% of the relaxed structure volume.

## Common Issues

| Problem | Cause | Solution |
|---|---|---|
| E-V curve has kinks or scatter | Inconsistent convergence across volume points | Ensure identical `ecutwfc`, `ecutrho`, k-grid, and `conv_thr` for all points. Check that all calculations converged. |
| B0 unreasonably high or low | Volume range too narrow or data noise | Expand `VOL_RANGE` to 0.07--0.10. Add more points. |
| B0' outside 2--8 range | Insufficient data points or too-narrow range | Use at least 9 points. Expand volume range. Check for unconverged calculations. |
| Fit does not converge | Poor initial guess or bad data | Remove obvious outlier points. Ensure energies span >= 0.1 eV total. Try polynomial fit first to get V0 guess. |
| BM3 and Vinet disagree on B0 | Non-parabolic E-V near extremes | Use only the central 80% of data points, or reduce `VOL_RANGE`. BM3 is most reliable for moderate compressions. |
| BM4 fit gives unreasonable B0'' | Too few data points for 5-parameter fit | Need >= 9 well-converged points. Fall back to BM3. |
| MACE gives different B0 than QE | MACE model limitations | Expected for materials outside MACE training domain. Use QE as ground truth. |
| QE crashes at large compression | Basis set or pseudopotential breakdown | Reduce `VOL_RANGE`. Increase `ecutwfc`. Check pseudopotential pressure validity. |
| Volume points not evenly spaced in V | Strain applied as volume fraction | Correct behavior: strain_values define fractional volume change, so V = V0*(1+strain). EOS fitting handles non-uniform spacing. |
| Cohesive energy too large | MACE isolated atom energy inaccurate | Use large vacuum box (15+ A). Compare with known experimental values. For DFT, use spin-polarized calculations for magnetic atoms. |
| H-P crossing not found | Phases too similar or pressure range too narrow | Extend the pressure range by using larger `VOL_RANGE` (0.10--0.15). Check that both phases are well-converged. |
| Multi-phase: wrong ground state predicted | MACE energy ordering inaccurate | Compare with DFT. Some MACE models struggle with magnetic phases (e.g., Fe). Try a larger MACE model ("large") or use DFT for definitive phase ordering. |
| Multi-phase: BM fit fails for one phase | Phase is mechanically unstable at some volumes | Reduce `VOL_RANGE` for that phase. Check if the phase spontaneously transforms during relaxation (e.g., FCC Fe at large expansion). |
| Debye temperature far from experiment | Poisson ratio estimate is poor | Use an experimentally measured Poisson ratio. The Moruzzi-Janak-Schwarz method is approximate; consider computing phonons for more accurate Debye temperatures. |
| Thermal expansion is negative | Numerical noise in F(V,T) minimization | Increase `N_TEMP` for smoother temperature sampling. Ensure the BM fit is well-converged (R^2 > 0.9999). Use a wider volume search range in `minimize_scalar`. |
| Cp diverges at high T | alpha^2 * B * V * T term dominates | This is a known limitation of the quasi-harmonic Debye model at T >> Theta_D. Restrict the temperature range to T < 1.5 * Theta_D. |
| LAMMPS LAMMPSlib import fails | LAMMPS Python bindings not installed | Install via `pip install lammps` or build from source with shared libraries. Ensure `liblammps.so` is on `LD_LIBRARY_PATH`. |
| LAMMPS EAM potential not found | Incorrect path to potential file | Use `os.path.abspath()` for the potential file path. Download from NIST Interatomic Potentials Repository. |
| LAMMPS and MACE E-V curves have very different curvature | Fundamentally different potential energy surfaces | This is expected -- EAM potentials and DFT-trained MLIPs model different physics. Compare both against experimental B0 to determine which is more accurate. |
| LAMMPS gives different V0 than MACE | Different equilibrium lattice constants | Expected behavior. Compare both against experimental lattice constant. EAM potentials are often fitted to experimental a0, while MACE inherits PBE's ~1-3% overestimate. |
