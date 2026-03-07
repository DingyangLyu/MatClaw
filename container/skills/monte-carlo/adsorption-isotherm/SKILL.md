# Adsorption Isotherm Calculation with RASPA3

## When to Use

- Computing single-component or multi-component adsorption isotherms (loading vs pressure) in porous materials
- Fitting experimental or simulated isotherms to Langmuir, dual-site Langmuir, BET, or Freundlich models
- Predicting multi-component adsorption from single-component data using Ideal Adsorbed Solution Theory (IAST)
- Generating pressure-composition (P-x-y) diagrams for binary gas mixtures
- Calculating Henry coefficients from the zero-pressure slope of isotherms
- Determining isosteric heat of adsorption from isotherms at multiple temperatures or from Widom insertion
- Screening MOFs, zeolites, or carbon nanotubes for gas storage capacity

## Method Selection

| Goal | Method | Notes |
|------|--------|-------|
| Single-component isotherm | GCMC at multiple pressures | Loop over 10-20 pressure points; most common approach |
| Multi-component isotherm | Multi-component GCMC | Specify mole fractions; computationally expensive |
| Multi-component prediction from singles | IAST fitting | Fit single-component isotherms, then apply pyIAST; much faster |
| Henry coefficient (infinite dilution) | Widom insertion | More efficient than low-pressure GCMC; gives K_H and Q_st simultaneously |
| Heat of adsorption at zero loading | Widom insertion | Returns enthalpy of adsorption directly |
| Heat of adsorption at finite loading | Fluctuation method in GCMC | RASPA3 reports Q_st from energy-loading fluctuations |
| Isosteric heat vs loading | Clausius-Clapeyron from isotherms at 2+ temperatures | Fit isotherms at T1, T2; compute Q_st = -R d(ln P)/d(1/T) at constant loading |
| BET surface area | N2 isotherm at 77 K + BET fit | Use GCMC isotherm in P/P0 = 0.05-0.30 range |

## Prerequisites

- RASPA3 binary (`raspa3`) -- pre-installed in the container
- A framework CIF file (MOF from CoRE-MOF, zeolite from IZA database, or user-provided)
- Official examples at `/usr/share/raspa3/examples/` -- always start from these
- Python packages: `numpy`, `scipy`, `matplotlib` (pre-installed); optionally `pyiast` (`pip install pyiast`)

## Detailed Steps

### Step 1: Prepare framework structure

```python
#!/usr/bin/env python3
"""Prepare a framework CIF for RASPA3 isotherm calculations."""
from pymatgen.core import Structure
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
import os

work_dir = "/tmp/isotherm_calc"
os.makedirs(work_dir, exist_ok=True)

# Load framework CIF (from file, Materials Project, or CoRE-MOF)
structure = Structure.from_file("my_framework.cif")

# Convert to P1 symmetry (RASPA3 handles symmetry expansion, but P1 is safest)
sga = SpacegroupAnalyzer(structure)
conventional = sga.get_conventional_standard_structure()
conventional.to(filename=os.path.join(work_dir, "framework.cif"))

print(f"Framework: {conventional.composition.reduced_formula}")
print(f"Space group: {sga.get_space_group_symbol()}")
print(f"Cell volume: {conventional.volume:.2f} A^3")
print(f"Number of atoms: {len(conventional)}")

# Check if charges are present (important for polar molecules)
if "charge" in conventional.site_properties:
    charges = conventional.site_properties["charge"]
    print(f"Charges found: min={min(charges):.3f}, max={max(charges):.3f}")
else:
    print("WARNING: No partial charges found in CIF.")
    print("  Charges are needed for polar adsorbates (CO2, H2O, NH3).")
    print("  Non-polar adsorbates (CH4, H2, noble gases) can proceed without charges.")
```

### Step 2: Single-component isotherm -- complete workflow

```python
#!/usr/bin/env python3
"""
Complete single-component adsorption isotherm workflow.
Example: CH4 in IRMOF-1 at 298 K.
"""
import json
import subprocess
import os
import re
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

# ===================== CONFIGURATION =====================
FRAMEWORK = "IRMOF-1"          # Name must match CIF filename (without .cif)
GUEST = "methane"               # RASPA3 molecule name
TEMPERATURE = 298.0             # Kelvin
FORCE_FIELD = "GenericMOFs"     # or "GenericZeolites", "UFF"
MOLECULE_DEF = "TraPPE"         # Guest molecule model
CUTOFF = 12.0                   # Angstrom
N_CYCLES = 25000                # Production MC cycles
N_INIT = 10000                  # Equilibration cycles

# Pressure grid (Pascals): logarithmically spaced from 0.01 bar to 100 bar
PRESSURES_PA = np.logspace(3, 7, 15)  # 1e3 to 1e7 Pa

BASE_DIR = "/tmp/ch4_isotherm"
# ==========================================================


def build_gcmc_input(framework, guest, temperature, pressure,
                     force_field, mol_def, cutoff, n_cycles, n_init):
    """Build RASPA3 simulation.json for single-component GCMC."""
    component = {
        "Name": guest,
        "Type": "Adsorbate",
        "MoleculeDefinition": mol_def,
        "TranslationProbability": 0.5,
        "ReinsertionProbability": 0.5,
        "SwapProbability": 1.0,
        "CreateNumberOfMolecules": 0
    }
    # Add rotation for multi-site molecules
    if guest.lower() not in ["methane", "helium", "neon", "argon", "krypton", "xenon"]:
        component["RotationProbability"] = 0.5

    return {
        "SimulationType": "MonteCarlo",
        "NumberOfCycles": n_cycles,
        "NumberOfInitializationCycles": n_init,
        "PrintEvery": max(n_cycles // 10, 100),
        "Systems": [
            {
                "Type": "Framework",
                "Name": framework,
                "ExternalTemperature": temperature,
                "ExternalPressure": pressure,
                "ChargeMethod": "Ewald",
                "ForceField": force_field,
                "CutOff": cutoff,
                "Components": [component]
            }
        ]
    }


def run_single_pressure(pressure, sim_index):
    """Run GCMC at one pressure point and return parsed results."""
    sim_dir = os.path.join(BASE_DIR, f"P{sim_index:02d}_{pressure:.2e}")
    os.makedirs(sim_dir, exist_ok=True)

    sim_input = build_gcmc_input(
        FRAMEWORK, GUEST, TEMPERATURE, pressure,
        FORCE_FIELD, MOLECULE_DEF, CUTOFF, N_CYCLES, N_INIT
    )

    with open(os.path.join(sim_dir, "simulation.json"), "w") as f:
        json.dump(sim_input, f, indent=2)

    # Copy framework CIF into simulation directory if not already there
    cif_src = os.path.join(BASE_DIR, f"{FRAMEWORK}.cif")
    cif_dst = os.path.join(sim_dir, f"{FRAMEWORK}.cif")
    if os.path.exists(cif_src) and not os.path.exists(cif_dst):
        os.system(f"cp '{cif_src}' '{cif_dst}'")

    result = subprocess.run(
        ["raspa3"], cwd=sim_dir,
        capture_output=True, text=True, timeout=7200
    )

    log_path = os.path.join(sim_dir, "output.log")
    with open(log_path, "w") as f:
        f.write(result.stdout)
        if result.stderr:
            f.write("\n--- STDERR ---\n")
            f.write(result.stderr)

    return parse_raspa3_loading(log_path)


def parse_raspa3_loading(log_path):
    """Parse loading and heat of adsorption from RASPA3 output."""
    with open(log_path) as f:
        text = f.read()

    result = {"loading_mol_kg": 0.0, "loading_err": 0.0,
              "loading_molec_uc": 0.0, "loading_molec_uc_err": 0.0,
              "qst_kJ_mol": None}

    patterns = {
        "mol_kg": r"Average loading absolute\s*\[mol/kg framework\]\s*[:\s]*([\d.eE+-]+)\s*\+/-\s*([\d.eE+-]+)",
        "molec_uc": r"Average loading absolute\s*\[molecules/unit cell\]\s*[:\s]*([\d.eE+-]+)\s*\+/-\s*([\d.eE+-]+)",
    }
    for key, pat in patterns.items():
        m = re.search(pat, text)
        if m:
            if key == "mol_kg":
                result["loading_mol_kg"] = float(m.group(1))
                result["loading_err"] = float(m.group(2))
            elif key == "molec_uc":
                result["loading_molec_uc"] = float(m.group(1))
                result["loading_molec_uc_err"] = float(m.group(2))

    # Heat of adsorption
    qst_match = re.search(r"[Ee]nthalpy of [Aa]dsorption\s*[:\s]*([-\d.eE+]+)", text)
    if not qst_match:
        qst_match = re.search(r"[Hh]eat of [Aa]dsorption\s*[:\s]*([-\d.eE+]+)", text)
    if qst_match:
        result["qst_kJ_mol"] = float(qst_match.group(1))

    return result


# ======================== MAIN ========================
os.makedirs(BASE_DIR, exist_ok=True)

# Run GCMC at each pressure
all_results = []
for i, P in enumerate(PRESSURES_PA):
    print(f"[{i+1}/{len(PRESSURES_PA)}] P = {P/1e5:.4f} bar ({P:.2e} Pa)")
    res = run_single_pressure(P, i)
    res["pressure_Pa"] = P
    all_results.append(res)
    print(f"  Loading = {res['loading_mol_kg']:.4f} +/- {res['loading_err']:.4f} mol/kg")

# Collect arrays
P_bar = np.array([r["pressure_Pa"] for r in all_results]) / 1e5
loading = np.array([r["loading_mol_kg"] for r in all_results])
loading_err = np.array([r["loading_err"] for r in all_results])

# Save raw data
data = np.column_stack([P_bar, loading, loading_err])
np.savetxt(os.path.join(BASE_DIR, "isotherm_data.csv"), data,
           header="Pressure(bar) Loading(mol/kg) Error(mol/kg)",
           fmt="%.6e", delimiter="  ")
print(f"\nData saved to {BASE_DIR}/isotherm_data.csv")
```

### Step 3: Langmuir and dual-site Langmuir fitting

```python
#!/usr/bin/env python3
"""
Fit adsorption isotherms to analytical models.
Requires isotherm data from Step 2.
"""
import numpy as np
from scipy.optimize import curve_fit
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Load simulated isotherm data
data = np.loadtxt("/tmp/ch4_isotherm/isotherm_data.csv")
P_bar = data[:, 0]
loading = data[:, 1]
loading_err = data[:, 2]

P_Pa = P_bar * 1e5  # Convert to Pa for fitting


# ---- Langmuir model ----
def langmuir(P, q_sat, K_L):
    """Single-site Langmuir: q = q_sat * K_L * P / (1 + K_L * P)"""
    return q_sat * K_L * P / (1.0 + K_L * P)


# ---- Dual-site Langmuir model ----
def dual_site_langmuir(P, q1, K1, q2, K2):
    """q = q1*K1*P/(1+K1*P) + q2*K2*P/(1+K2*P)"""
    return q1 * K1 * P / (1.0 + K1 * P) + q2 * K2 * P / (1.0 + K2 * P)


# ---- Freundlich model ----
def freundlich(P, K_F, n):
    """q = K_F * P^(1/n)"""
    return K_F * np.power(P, 1.0 / n)


# ---- BET model (for N2 at 77 K) ----
def bet_transform(P_rel, v_m, C):
    """BET linearized: 1/[v(P0/P - 1)] = (C-1)/(v_m*C) * (P/P0) + 1/(v_m*C)
       Here we return v(P) = v_m * C * P_rel / ((1 - P_rel) * (1 + (C-1)*P_rel))
    """
    return v_m * C * P_rel / ((1.0 - P_rel) * (1.0 + (C - 1.0) * P_rel))


# ========== Fit Langmuir ==========
try:
    popt_L, pcov_L = curve_fit(langmuir, P_Pa, loading,
                               p0=[max(loading)*1.5, 1e-5],
                               sigma=loading_err if np.all(loading_err > 0) else None,
                               bounds=([0, 0], [np.inf, np.inf]),
                               maxfev=10000)
    q_sat_L, K_L = popt_L
    perr_L = np.sqrt(np.diag(pcov_L))
    print("=== Langmuir Fit ===")
    print(f"  q_sat = {q_sat_L:.4f} +/- {perr_L[0]:.4f} mol/kg")
    print(f"  K_L   = {K_L:.4e} +/- {perr_L[1]:.4e} 1/Pa")
    print(f"  Henry coeff K_H = q_sat * K_L = {q_sat_L * K_L:.4e} mol/(kg*Pa)")

    # R-squared
    ss_res = np.sum((loading - langmuir(P_Pa, *popt_L))**2)
    ss_tot = np.sum((loading - np.mean(loading))**2)
    R2_L = 1 - ss_res / ss_tot
    print(f"  R^2 = {R2_L:.6f}")
except RuntimeError as e:
    print(f"Langmuir fit failed: {e}")
    popt_L = None

# ========== Fit dual-site Langmuir ==========
try:
    popt_DSL, pcov_DSL = curve_fit(
        dual_site_langmuir, P_Pa, loading,
        p0=[max(loading)*0.6, 1e-4, max(loading)*0.4, 1e-6],
        sigma=loading_err if np.all(loading_err > 0) else None,
        bounds=([0, 0, 0, 0], [np.inf, np.inf, np.inf, np.inf]),
        maxfev=50000
    )
    perr_DSL = np.sqrt(np.diag(pcov_DSL))
    print("\n=== Dual-Site Langmuir Fit ===")
    print(f"  Site 1: q1={popt_DSL[0]:.4f}, K1={popt_DSL[1]:.4e}")
    print(f"  Site 2: q2={popt_DSL[2]:.4f}, K2={popt_DSL[3]:.4e}")

    ss_res = np.sum((loading - dual_site_langmuir(P_Pa, *popt_DSL))**2)
    R2_DSL = 1 - ss_res / ss_tot
    print(f"  R^2 = {R2_DSL:.6f}")
except RuntimeError as e:
    print(f"Dual-site Langmuir fit failed: {e}")
    popt_DSL = None

# ========== Plot ==========
P_fit = np.logspace(np.log10(P_Pa.min()), np.log10(P_Pa.max()), 200)

fig, ax = plt.subplots(figsize=(9, 6))
ax.errorbar(P_bar, loading, yerr=loading_err,
            fmt="ko", capsize=4, markersize=7, label="GCMC data", zorder=5)

if popt_L is not None:
    ax.plot(P_fit/1e5, langmuir(P_fit, *popt_L),
            "b-", linewidth=2, label=f"Langmuir (R$^2$={R2_L:.4f})")

if popt_DSL is not None:
    ax.plot(P_fit/1e5, dual_site_langmuir(P_fit, *popt_DSL),
            "r--", linewidth=2, label=f"DSL (R$^2$={R2_DSL:.4f})")

ax.set_xlabel("Pressure (bar)", fontsize=14)
ax.set_ylabel("Loading (mol/kg)", fontsize=14)
ax.set_title(f"CH$_4$ adsorption isotherm at {298} K", fontsize=16)
ax.set_xscale("log")
ax.legend(fontsize=12)
ax.grid(True, alpha=0.3)
ax.tick_params(labelsize=12)
plt.tight_layout()
plt.savefig("/tmp/ch4_isotherm/isotherm_fitted.png", dpi=150, bbox_inches="tight")
plt.close()
print("\nPlot saved to /tmp/ch4_isotherm/isotherm_fitted.png")
```

### Step 4: BET surface area from N2 isotherm at 77 K

```python
#!/usr/bin/env python3
"""
BET surface area from simulated N2 isotherm at 77 K.
Run GCMC at relative pressures P/P0 = 0.01 to 0.35
(P0 for N2 at 77 K ~ 101325 Pa = 1 atm).
"""
import json
import subprocess
import os
import re
import numpy as np
from scipy.optimize import curve_fit
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

FRAMEWORK = "IRMOF-1"
TEMPERATURE = 77.0           # K (liquid nitrogen temperature)
P0 = 101325.0                # Saturation pressure of N2 at 77 K (Pa)
FORCE_FIELD = "GenericMOFs"
CUTOFF = 12.0
N_CYCLES = 30000
N_INIT = 15000
BASE_DIR = "/tmp/bet_analysis"

# Relative pressures for BET analysis
P_rel_values = [0.01, 0.02, 0.04, 0.06, 0.08, 0.10, 0.12,
                0.15, 0.18, 0.20, 0.25, 0.30, 0.35]

os.makedirs(BASE_DIR, exist_ok=True)

# --- Run GCMC at each relative pressure ---
loadings = []
for i, P_rel in enumerate(P_rel_values):
    P_abs = P_rel * P0
    sim_dir = os.path.join(BASE_DIR, f"Prel_{P_rel:.3f}")
    os.makedirs(sim_dir, exist_ok=True)

    sim_input = {
        "SimulationType": "MonteCarlo",
        "NumberOfCycles": N_CYCLES,
        "NumberOfInitializationCycles": N_INIT,
        "PrintEvery": N_CYCLES // 10,
        "Systems": [{
            "Type": "Framework",
            "Name": FRAMEWORK,
            "ExternalTemperature": TEMPERATURE,
            "ExternalPressure": P_abs,
            "ChargeMethod": "Ewald",
            "ForceField": FORCE_FIELD,
            "CutOff": CUTOFF,
            "Components": [{
                "Name": "N2",
                "Type": "Adsorbate",
                "MoleculeDefinition": "TraPPE",
                "TranslationProbability": 0.5,
                "RotationProbability": 0.5,
                "ReinsertionProbability": 0.5,
                "SwapProbability": 1.0,
                "CreateNumberOfMolecules": 0
            }]
        }]
    }

    with open(os.path.join(sim_dir, "simulation.json"), "w") as f:
        json.dump(sim_input, f, indent=2)

    result = subprocess.run(["raspa3"], cwd=sim_dir,
                            capture_output=True, text=True, timeout=7200)

    log_path = os.path.join(sim_dir, "output.log")
    with open(log_path, "w") as f:
        f.write(result.stdout)

    # Parse loading in cm^3(STP)/g -- needed for BET
    # Also accept mol/kg and convert: 1 mol/kg = 22414 cm^3(STP)/1000 g = 22.414 cm^3(STP)/g
    text = result.stdout
    match_cc = re.search(
        r"Average loading absolute\s*\[cm\^3.*?/g\]\s*[:\s]*([\d.eE+-]+)", text)
    match_mol = re.search(
        r"Average loading absolute\s*\[mol/kg framework\]\s*[:\s]*([\d.eE+-]+)", text)

    if match_cc:
        v_ads = float(match_cc.group(1))
    elif match_mol:
        v_ads = float(match_mol.group(1)) * 22.414  # mol/kg to cm^3(STP)/g
    else:
        v_ads = 0.0

    loadings.append(v_ads)
    print(f"P/P0 = {P_rel:.3f}, P = {P_abs:.1f} Pa, V_ads = {v_ads:.2f} cm^3(STP)/g")

P_rel = np.array(P_rel_values)
V_ads = np.array(loadings)

# --- BET analysis (use P/P0 = 0.05 to 0.30 range) ---
mask = (P_rel >= 0.05) & (P_rel <= 0.30)
P_bet = P_rel[mask]
V_bet = V_ads[mask]

# BET linearization: P/P0 / [V*(1 - P/P0)] = 1/(V_m*C) + (C-1)/(V_m*C) * P/P0
Y_bet = P_bet / (V_bet * (1 - P_bet))

# Linear fit: Y = intercept + slope * (P/P0)
coeffs = np.polyfit(P_bet, Y_bet, 1)
slope = coeffs[0]      # (C-1) / (V_m * C)
intercept = coeffs[1]  # 1 / (V_m * C)

V_m = 1.0 / (slope + intercept)
C = slope / intercept + 1.0

# BET surface area: S_BET = V_m * N_A * sigma_N2 / (22414)
# sigma_N2 = 0.162 nm^2 = 16.2e-20 m^2
N_A = 6.022e23
sigma_N2 = 16.2e-20  # m^2 per molecule
S_BET = V_m * N_A * sigma_N2 / 22414.0  # m^2/g

print(f"\n=== BET Analysis ===")
print(f"V_m (monolayer capacity) = {V_m:.2f} cm^3(STP)/g")
print(f"C (BET constant) = {C:.1f}")
print(f"BET surface area = {S_BET:.1f} m^2/g")

# R-squared of linearization
Y_pred = np.polyval(coeffs, P_bet)
ss_res = np.sum((Y_bet - Y_pred)**2)
ss_tot = np.sum((Y_bet - np.mean(Y_bet))**2)
R2 = 1 - ss_res / ss_tot
print(f"R^2 (BET linearization) = {R2:.6f}")

# Consistency check: C should be positive; recommended 50 < C < 200 for N2
if C < 0:
    print("WARNING: Negative C indicates poor BET applicability in this P/P0 range.")
elif C < 10:
    print("WARNING: Low C value; BET may not be reliable. Try adjusting the P/P0 range.")

# --- Plot ---
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

# Left: full isotherm
ax1.plot(P_rel, V_ads, "bo-", markersize=7, linewidth=1.5)
ax1.axvspan(0.05, 0.30, alpha=0.15, color="green", label="BET fitting range")
ax1.set_xlabel("P/P$_0$", fontsize=14)
ax1.set_ylabel("V$_{ads}$ (cm$^3$ STP/g)", fontsize=14)
ax1.set_title("N$_2$ adsorption isotherm at 77 K", fontsize=15)
ax1.legend(fontsize=11)
ax1.grid(True, alpha=0.3)

# Right: BET plot
ax2.plot(P_bet, Y_bet, "rs", markersize=8, label="Data")
P_fit_line = np.linspace(P_bet.min(), P_bet.max(), 50)
ax2.plot(P_fit_line, np.polyval(coeffs, P_fit_line), "k-", linewidth=2,
         label=f"Fit (R$^2$={R2:.4f})")
ax2.set_xlabel("P/P$_0$", fontsize=14)
ax2.set_ylabel("(P/P$_0$) / [V(1 - P/P$_0$)]", fontsize=14)
ax2.set_title(f"BET plot  --  S$_{{BET}}$ = {S_BET:.0f} m$^2$/g", fontsize=15)
ax2.legend(fontsize=11)
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(BASE_DIR, "bet_analysis.png"), dpi=150, bbox_inches="tight")
plt.close()
print(f"Plot saved to {BASE_DIR}/bet_analysis.png")
```

### Step 5: Henry coefficient and heat of adsorption from Widom insertion

```python
#!/usr/bin/env python3
"""
Henry coefficient and heat of adsorption at infinite dilution
via Widom insertion in RASPA3.
"""
import json
import subprocess
import os
import re
import numpy as np

BASE_DIR = "/tmp/henry_calc"
FRAMEWORK = "IRMOF-1"
GUESTS = ["methane", "CO2", "N2", "H2"]  # Compare multiple gases
TEMPERATURE = 298.0
N_CYCLES = 100000       # Widom needs many cycles for statistics
N_INIT = 10000
CUTOFF = 12.0

os.makedirs(BASE_DIR, exist_ok=True)

results_table = []

for guest in GUESTS:
    sim_dir = os.path.join(BASE_DIR, guest)
    os.makedirs(sim_dir, exist_ok=True)

    component = {
        "Name": guest,
        "Type": "Adsorbate",
        "MoleculeDefinition": "TraPPE",
        "WidomProbability": 1.0,
        "CreateNumberOfMolecules": 0
    }
    # Add rotation for multi-site molecules
    if guest.lower() not in ["methane", "helium", "argon", "krypton", "xenon"]:
        component["RotationProbability"] = 0.5

    sim_input = {
        "SimulationType": "MonteCarlo",
        "NumberOfCycles": N_CYCLES,
        "NumberOfInitializationCycles": N_INIT,
        "PrintEvery": N_CYCLES // 5,
        "Systems": [{
            "Type": "Framework",
            "Name": FRAMEWORK,
            "ExternalTemperature": TEMPERATURE,
            "ChargeMethod": "Ewald",
            "ForceField": "GenericMOFs",
            "CutOff": CUTOFF,
            "Components": [component]
        }]
    }

    with open(os.path.join(sim_dir, "simulation.json"), "w") as f:
        json.dump(sim_input, f, indent=2)

    result = subprocess.run(["raspa3"], cwd=sim_dir,
                            capture_output=True, text=True, timeout=3600)

    log_path = os.path.join(sim_dir, "output.log")
    with open(log_path, "w") as f:
        f.write(result.stdout)

    # Parse Henry coefficient and heat of adsorption
    text = result.stdout
    K_H = None
    Q_st = None

    kh_match = re.search(r"Henry coefficient\s*[:\s]*([\d.eE+-]+)", text)
    if kh_match:
        K_H = float(kh_match.group(1))

    qst_match = re.search(r"[Hh]eat of [Aa]dsorption\s*[:\s]*([-\d.eE+]+)", text)
    if not qst_match:
        qst_match = re.search(r"[Ee]nthalpy of [Aa]dsorption\s*[:\s]*([-\d.eE+]+)", text)
    if qst_match:
        Q_st = float(qst_match.group(1))

    results_table.append({
        "guest": guest,
        "K_H_mol_kg_Pa": K_H,
        "Q_st_kJ_mol": Q_st
    })
    print(f"{guest:>10s}: K_H = {K_H}, Q_st = {Q_st} kJ/mol")

# Print summary
print(f"\n{'='*60}")
print(f"{'Guest':>10s} {'K_H (mol/kg/Pa)':>18s} {'Q_st (kJ/mol)':>16s}")
print(f"{'='*60}")
for r in results_table:
    kh_str = f"{r['K_H_mol_kg_Pa']:.4e}" if r["K_H_mol_kg_Pa"] else "N/A"
    qst_str = f"{r['Q_st_kJ_mol']:.2f}" if r["Q_st_kJ_mol"] else "N/A"
    print(f"{r['guest']:>10s} {kh_str:>18s} {qst_str:>16s}")
```

### Step 6: IAST prediction of multi-component adsorption

```python
#!/usr/bin/env python3
"""
Ideal Adsorbed Solution Theory (IAST) prediction of binary mixture
adsorption from single-component isotherms.

Install pyiast if not available: pip install pyiast
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Try pyiast; if not available, use manual IAST implementation
try:
    import pyiast
    HAS_PYIAST = True
    print("Using pyIAST library")
except ImportError:
    HAS_PYIAST = False
    print("pyIAST not found. Using built-in Langmuir-based IAST.")
    print("Install with: pip install pyiast")

from scipy.optimize import fsolve, curve_fit


# ============ Langmuir model for IAST ============
def langmuir(P, q_sat, K):
    return q_sat * K * P / (1.0 + K * P)


def langmuir_spreading_pressure(P, q_sat, K):
    """Integral of q/P dP from 0 to P for Langmuir model.
    pi*A/(RT) = integral = q_sat * ln(1 + K*P)"""
    return q_sat * np.log(1.0 + K * P)


def iast_binary_langmuir(P_total, y, params_A, params_B):
    """
    IAST for binary Langmuir isotherms.
    y = [y_A, y_B] gas-phase mole fractions
    params_A = (q_sat_A, K_A), params_B = (q_sat_B, K_B)
    Returns: q_A, q_B (adsorbed loadings, mol/kg)
    """
    q_sat_A, K_A = params_A
    q_sat_B, K_B = params_B

    def equations(x_A):
        x_B = 1.0 - x_A
        # P0_i: pressure at which pure component i has same spreading pressure
        # For Langmuir: pi = q_sat * ln(1 + K * P0)
        # Equal spreading pressure: q_sat_A * ln(1+K_A*P0_A) = q_sat_B * ln(1+K_B*P0_B)
        # Raoult's law: P_total * y_i = P0_i * x_i
        P0_A = P_total * y[0] / max(x_A, 1e-30)
        P0_B = P_total * y[1] / max(x_B, 1e-30)

        pi_A = langmuir_spreading_pressure(P0_A, q_sat_A, K_A)
        pi_B = langmuir_spreading_pressure(P0_B, q_sat_B, K_B)

        return pi_A - pi_B

    # Solve for x_A
    x_A_sol = fsolve(equations, 0.5, full_output=False)[0]
    x_A_sol = np.clip(x_A_sol, 1e-10, 1.0 - 1e-10)
    x_B_sol = 1.0 - x_A_sol

    P0_A = P_total * y[0] / x_A_sol
    P0_B = P_total * y[1] / x_B_sol

    q0_A = langmuir(P0_A, q_sat_A, K_A)
    q0_B = langmuir(P0_B, q_sat_B, K_B)

    # Total loading: 1/q_total = sum(x_i / q0_i)
    q_total = 1.0 / (x_A_sol / q0_A + x_B_sol / q0_B)
    q_A = q_total * x_A_sol
    q_B = q_total * x_B_sol

    return q_A, q_B


# ============ Example: CO2/N2 separation ============
# Langmuir parameters (fit from single-component GCMC isotherms)
# Replace with actual fitted values from Step 3
params_CO2 = (8.0, 5e-5)    # (q_sat mol/kg, K_L 1/Pa) -- example values
params_N2  = (6.0, 5e-6)    # (q_sat mol/kg, K_L 1/Pa) -- example values

# Flue gas conditions: 15% CO2, 85% N2, total 1 bar
P_total = 1e5  # Pa
y_CO2 = 0.15
y_N2 = 0.85

q_CO2, q_N2 = iast_binary_langmuir(P_total, [y_CO2, y_N2], params_CO2, params_N2)
S = (q_CO2 / q_N2) / (y_CO2 / y_N2)

print(f"IAST prediction at {P_total/1e5:.0f} bar ({y_CO2*100:.0f}% CO2 / {y_N2*100:.0f}% N2):")
print(f"  CO2 loading: {q_CO2:.4f} mol/kg")
print(f"  N2  loading: {q_N2:.4f} mol/kg")
print(f"  Selectivity S(CO2/N2) = {S:.1f}")

# --- Pressure-composition diagram ---
P_range = np.logspace(3, 7, 30)  # 0.01 to 100 bar
q_CO2_arr = []
q_N2_arr = []

for P in P_range:
    qA, qB = iast_binary_langmuir(P, [y_CO2, y_N2], params_CO2, params_N2)
    q_CO2_arr.append(qA)
    q_N2_arr.append(qB)

q_CO2_arr = np.array(q_CO2_arr)
q_N2_arr = np.array(q_N2_arr)
selectivity = (q_CO2_arr / q_N2_arr) / (y_CO2 / y_N2)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

ax1.plot(P_range/1e5, q_CO2_arr, "r-o", markersize=4, linewidth=2, label="CO$_2$")
ax1.plot(P_range/1e5, q_N2_arr, "b-s", markersize=4, linewidth=2, label="N$_2$")
ax1.set_xlabel("Total Pressure (bar)", fontsize=14)
ax1.set_ylabel("Loading (mol/kg)", fontsize=14)
ax1.set_title("IAST binary adsorption", fontsize=15)
ax1.set_xscale("log")
ax1.legend(fontsize=12)
ax1.grid(True, alpha=0.3)

ax2.plot(P_range/1e5, selectivity, "g-^", markersize=5, linewidth=2)
ax2.set_xlabel("Total Pressure (bar)", fontsize=14)
ax2.set_ylabel("Selectivity S(CO$_2$/N$_2$)", fontsize=14)
ax2.set_title("IAST selectivity vs pressure", fontsize=15)
ax2.set_xscale("log")
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("/tmp/ch4_isotherm/iast_prediction.png", dpi=150, bbox_inches="tight")
plt.close()
print("Plot saved to /tmp/ch4_isotherm/iast_prediction.png")
```

### Step 7: Isosteric heat of adsorption from isotherms at two temperatures

```python
#!/usr/bin/env python3
"""
Clausius-Clapeyron method: compute isosteric heat Q_st as a function
of loading from isotherms at two (or more) temperatures.

Q_st = -R * d(ln P) / d(1/T)  at constant loading
"""
import numpy as np
from scipy.interpolate import interp1d
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

R = 8.314e-3  # kJ/(mol*K)

# Load isotherms at two temperatures (from Step 2 runs at T1 and T2)
# Format: pressure (bar), loading (mol/kg), error (mol/kg)
data_T1 = np.loadtxt("/tmp/isotherm_T1/isotherm_data.csv")  # e.g., 273 K
data_T2 = np.loadtxt("/tmp/isotherm_T2/isotherm_data.csv")  # e.g., 298 K

T1, T2 = 273.0, 298.0  # Kelvin

P1, q1 = data_T1[:, 0], data_T1[:, 1]
P2, q2 = data_T2[:, 0], data_T2[:, 1]

# Interpolate: P as a function of loading (invert the isotherm)
# Need monotonic loading -- use only the ascending part
f_P1 = interp1d(q1, np.log(P1), kind="linear", fill_value="extrapolate")
f_P2 = interp1d(q2, np.log(P2), kind="linear", fill_value="extrapolate")

# Compute Q_st at a range of loadings
q_min = max(q1.min(), q2.min())
q_max = min(q1.max(), q2.max())
q_range = np.linspace(q_min * 1.1, q_max * 0.9, 30)

Q_st = np.zeros_like(q_range)
for i, q in enumerate(q_range):
    ln_P1 = f_P1(q)
    ln_P2 = f_P2(q)
    # Q_st = -R * (ln P2 - ln P1) / (1/T2 - 1/T1)
    Q_st[i] = -R * (ln_P2 - ln_P1) / (1.0/T2 - 1.0/T1)

print(f"Isosteric heat of adsorption (Clausius-Clapeyron):")
print(f"{'Loading (mol/kg)':>18s} {'Q_st (kJ/mol)':>16s}")
for q, qst in zip(q_range[::3], Q_st[::3]):
    print(f"{q:>18.4f} {qst:>16.2f}")

# Plot
fig, ax = plt.subplots(figsize=(8, 5))
ax.plot(q_range, Q_st, "r-o", markersize=5, linewidth=2)
ax.set_xlabel("Loading (mol/kg)", fontsize=14)
ax.set_ylabel("Q$_{st}$ (kJ/mol)", fontsize=14)
ax.set_title("Isosteric heat of adsorption", fontsize=15)
ax.grid(True, alpha=0.3)
ax.tick_params(labelsize=12)
plt.tight_layout()
plt.savefig("/tmp/isotherm_Qst/clausius_clapeyron.png", dpi=150, bbox_inches="tight")
plt.close()
print("Plot saved.")
```

### Step 8: Multi-component isotherm with pressure-composition diagram

```python
#!/usr/bin/env python3
"""
Multi-component GCMC isotherm: vary total pressure at fixed composition.
Generates a pressure-composition (P-x) diagram.
"""
import json
import subprocess
import os
import re
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

FRAMEWORK = "IRMOF-1"
TEMPERATURE = 298.0
FORCE_FIELD = "GenericMOFs"
CUTOFF = 12.0
N_CYCLES = 30000
N_INIT = 15000
BASE_DIR = "/tmp/multicomp_isotherm"

# Binary mixture: CO2 (15%) + N2 (85%)
COMPONENTS = [
    {"Name": "CO2",  "MoleFraction": 0.15, "MoleculeDefinition": "TraPPE"},
    {"Name": "N2",   "MoleFraction": 0.85, "MoleculeDefinition": "TraPPE"},
]

PRESSURES_PA = np.logspace(3, 7, 12)

os.makedirs(BASE_DIR, exist_ok=True)

all_data = []

for idx, P in enumerate(PRESSURES_PA):
    sim_dir = os.path.join(BASE_DIR, f"P{idx:02d}_{P:.2e}")
    os.makedirs(sim_dir, exist_ok=True)

    components_json = []
    for comp in COMPONENTS:
        c = {
            "Name": comp["Name"],
            "Type": "Adsorbate",
            "MoleculeDefinition": comp["MoleculeDefinition"],
            "MoleFraction": comp["MoleFraction"],
            "TranslationProbability": 0.5,
            "RotationProbability": 0.5,
            "ReinsertionProbability": 0.5,
            "SwapProbability": 1.0,
            "CreateNumberOfMolecules": 0
        }
        components_json.append(c)

    sim_input = {
        "SimulationType": "MonteCarlo",
        "NumberOfCycles": N_CYCLES,
        "NumberOfInitializationCycles": N_INIT,
        "PrintEvery": N_CYCLES // 10,
        "Systems": [{
            "Type": "Framework",
            "Name": FRAMEWORK,
            "ExternalTemperature": TEMPERATURE,
            "ExternalPressure": P,
            "ChargeMethod": "Ewald",
            "ForceField": FORCE_FIELD,
            "CutOff": CUTOFF,
            "Components": components_json
        }]
    }

    with open(os.path.join(sim_dir, "simulation.json"), "w") as f:
        json.dump(sim_input, f, indent=2)

    result = subprocess.run(["raspa3"], cwd=sim_dir,
                            capture_output=True, text=True, timeout=7200)

    log_path = os.path.join(sim_dir, "output.log")
    with open(log_path, "w") as f:
        f.write(result.stdout)

    # Parse per-component loading
    text = result.stdout
    row = {"P_Pa": P}
    for comp in COMPONENTS:
        name = comp["Name"]
        # Look for component-specific loading
        pat = rf"Component.*{name}.*?Average loading absolute\s*\[mol/kg framework\]\s*[:\s]*([\d.eE+-]+)"
        m = re.search(pat, text, re.DOTALL)
        if m:
            row[f"q_{name}"] = float(m.group(1))
        else:
            row[f"q_{name}"] = 0.0
    all_data.append(row)
    print(f"P = {P/1e5:.4f} bar: " + ", ".join(
        f"{k}={v:.4f}" for k, v in row.items() if k.startswith("q_")))

# Plot pressure-composition diagram
P_bar = np.array([d["P_Pa"] for d in all_data]) / 1e5
q_CO2 = np.array([d.get("q_CO2", 0) for d in all_data])
q_N2 = np.array([d.get("q_N2", 0) for d in all_data])
q_total = q_CO2 + q_N2
x_CO2 = np.where(q_total > 0, q_CO2 / q_total, 0)  # adsorbed phase mole fraction

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

ax1.plot(P_bar, q_CO2, "r-o", label="CO$_2$", markersize=5, linewidth=2)
ax1.plot(P_bar, q_N2, "b-s", label="N$_2$", markersize=5, linewidth=2)
ax1.plot(P_bar, q_total, "k--", label="Total", linewidth=1.5)
ax1.set_xlabel("Total Pressure (bar)", fontsize=14)
ax1.set_ylabel("Loading (mol/kg)", fontsize=14)
ax1.set_title("Multi-component adsorption", fontsize=15)
ax1.set_xscale("log")
ax1.legend(fontsize=12)
ax1.grid(True, alpha=0.3)

ax2.plot(P_bar, x_CO2, "g-^", markersize=6, linewidth=2)
ax2.axhline(y=0.15, color="gray", linestyle=":", label="Gas phase y$_{CO_2}$")
ax2.set_xlabel("Total Pressure (bar)", fontsize=14)
ax2.set_ylabel("Adsorbed phase x$_{CO_2}$", fontsize=14)
ax2.set_title("P-x diagram (CO$_2$ enrichment)", fontsize=15)
ax2.set_xscale("log")
ax2.legend(fontsize=12)
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(BASE_DIR, "multicomp_isotherm.png"), dpi=150, bbox_inches="tight")
plt.close()
print(f"Plot saved to {BASE_DIR}/multicomp_isotherm.png")
```

### Step 9: Framework examples -- zeolite and carbon nanotube

```python
#!/usr/bin/env python3
"""
Setup GCMC isotherms for non-MOF frameworks: zeolites and carbon nanotubes.
"""
import json
import os

# ============ Zeolite Example: CO2 in MFI (silicalite-1) ============
def zeolite_gcmc_input(framework="MFI", guest="CO2", T=300.0, P=1e5):
    """GCMC input for zeolite framework."""
    return {
        "SimulationType": "MonteCarlo",
        "NumberOfCycles": 25000,
        "NumberOfInitializationCycles": 10000,
        "PrintEvery": 2500,
        "Systems": [{
            "Type": "Framework",
            "Name": framework,
            "ExternalTemperature": T,
            "ExternalPressure": P,
            "ChargeMethod": "Ewald",
            "ForceField": "GenericZeolites",   # Zeolite-specific force field
            "CutOff": 12.0,
            "Components": [{
                "Name": guest,
                "Type": "Adsorbate",
                "MoleculeDefinition": "TraPPE",
                "TranslationProbability": 0.5,
                "RotationProbability": 0.5,
                "ReinsertionProbability": 0.5,
                "SwapProbability": 1.0,
                "CreateNumberOfMolecules": 0
            }]
        }]
    }


# ============ Carbon Nanotube Example ============
def cnt_gcmc_input(framework="CNT_10_10", guest="H2", T=77.0, P=1e5):
    """
    GCMC for gas adsorption in carbon nanotubes.
    The CNT must be provided as a CIF with periodic boundary conditions.
    Use UFF or custom Lennard-Jones parameters for carbon.
    """
    return {
        "SimulationType": "MonteCarlo",
        "NumberOfCycles": 30000,
        "NumberOfInitializationCycles": 15000,
        "PrintEvery": 3000,
        "Systems": [{
            "Type": "Framework",
            "Name": framework,
            "ExternalTemperature": T,
            "ExternalPressure": P,
            "ChargeMethod": "None",         # CNTs are non-polar
            "ForceField": "UFF",
            "CutOff": 12.0,
            "Components": [{
                "Name": guest,
                "Type": "Adsorbate",
                "MoleculeDefinition": "TraPPE",
                "TranslationProbability": 0.5,
                "ReinsertionProbability": 0.5,
                "SwapProbability": 1.0,
                "CreateNumberOfMolecules": 0
            }]
        }]
    }


# Generate CNT CIF using ASE
def generate_cnt_cif(n=10, m=10, length=4, vacuum=15.0, output="CNT_10_10.cif"):
    """Generate a periodic carbon nanotube CIF using ASE."""
    from ase.build import nanotube
    from ase.io import write

    cnt = nanotube(n, m, length=length, vacuum=vacuum)
    # Make it periodic along the tube axis (z)
    cnt.pbc = [True, True, True]
    write(output, cnt)
    print(f"CNT ({n},{m}) CIF saved to {output}")
    print(f"  Atoms: {len(cnt)}, Cell: {cnt.cell.lengths()}")
    return output


# Write example inputs
for name, inp_func, kwargs in [
    ("zeolite_CO2", zeolite_gcmc_input, {"framework": "MFI", "guest": "CO2"}),
    ("cnt_H2", cnt_gcmc_input, {"framework": "CNT_10_10", "guest": "H2", "T": 77.0}),
]:
    out_dir = f"/tmp/framework_examples/{name}"
    os.makedirs(out_dir, exist_ok=True)
    sim_input = inp_func(**kwargs)
    with open(os.path.join(out_dir, "simulation.json"), "w") as f:
        json.dump(sim_input, f, indent=2)
    print(f"Created {out_dir}/simulation.json")

# Generate CNT structure
generate_cnt_cif(n=10, m=10, length=4, vacuum=15.0,
                 output="/tmp/framework_examples/cnt_H2/CNT_10_10.cif")
```

## Key Parameters

| Parameter | Description | Typical Value |
|-----------|-------------|---------------|
| `NumberOfCycles` | Production MC cycles | 20000-100000; increase for tighter error bars |
| `NumberOfInitializationCycles` | Equilibration (not sampled) | 10000-50000 |
| `ExternalTemperature` | Temperature (K) | 77 (BET), 273-298 (ambient), 313 (post-combustion) |
| `ExternalPressure` | Pressure (Pa) | 1e3-1e7 for isotherms; 1e5 = 1 bar |
| `CutOff` | LJ cutoff (Angstrom) | 12.0; must be < half smallest box dimension |
| `ForceField` | Framework force field | `GenericMOFs`, `GenericZeolites`, `UFF` |
| `MoleculeDefinition` | Guest model | `TraPPE` (hydrocarbons, CO2, N2); check RASPA3 library |
| `SwapProbability` | GCMC insert/delete | 1.0 (essential for GCMC) |
| `WidomProbability` | Widom test insertion | 1.0 (Henry coefficient only; no SwapProbability) |
| `MoleFraction` | Component fraction in mixture | Must sum to 1.0 across all components |
| Pressure points | Number of isotherm points | 10-20 logarithmically spaced for good resolution |
| BET P/P0 range | Relative pressure for BET fit | 0.05-0.30 (N2 at 77 K, P0 = 101325 Pa) |
| Langmuir `q_sat` | Saturation capacity (mol/kg) | Material-dependent; 1-30 mol/kg for typical MOFs |
| Langmuir `K_L` | Affinity constant (1/Pa) | 1e-7 to 1e-3; higher = stronger binding |

## Interpreting Results

### Loading units
- **mol/kg**: moles of gas per kg of framework (gravimetric) -- most common for comparisons
- **molecules/unit cell**: absolute count per unit cell
- **cm^3(STP)/cm^3**: volumetric uptake at standard T/P
- **cm^3(STP)/g**: volumetric per gram (used in BET analysis)
- **wt%**: weight percent = 100 * (mass_gas / (mass_gas + mass_framework))

### Isotherm shape classification
- **Type I** (Langmuir): monotonic rise to plateau; microporous materials (most MOFs, zeolites)
- **Type II**: sigmoidal; non-porous or macroporous (unrestricted multilayer)
- **Type IV**: hysteresis loop; mesoporous materials (check for pore condensation)
- **Type V**: weak adsorbent with mesopores; see water isotherms in hydrophobic MOFs

### Goodness of fit
- R^2 > 0.99 for Langmuir/DSL fit indicates good single-site or two-site behavior
- If Langmuir R^2 < 0.95, try dual-site Langmuir or Freundlich
- Dual-site Langmuir is often needed when the framework has two distinct adsorption sites

### Henry coefficient
- K_H in mol/(kg*Pa): higher = stronger affinity at low pressure
- For ranking materials, compare K_H at the same temperature
- K_H relates to the initial isotherm slope: q = K_H * P at low coverage

### BET surface area benchmarks
- IRMOF-1 (MOF-5): ~3000-3500 m^2/g
- HKUST-1 (MOF-199): ~1500-1800 m^2/g
- ZIF-8: ~1200-1600 m^2/g
- MFI zeolite: ~400-500 m^2/g
- Activated carbon: ~1000-2000 m^2/g
- If your BET value differs by >30% from literature, check charges, force field, and CIF quality

### Heat of adsorption
- Q_st from Widom: valid only at zero loading (infinite dilution)
- Q_st from Clausius-Clapeyron: valid at finite loading; requires isotherms at 2+ temperatures
- Typical ranges: CH4 in MOFs: 15-25 kJ/mol; CO2 in MOFs: 20-50 kJ/mol; H2: 4-12 kJ/mol

## Common Issues

| Problem | Symptom | Solution |
|---------|---------|----------|
| Box too small for cutoff | RASPA3 error about cutoff > half box | Add `"NumberOfUnitCells": [2,2,2]` to system definition; or reduce `CutOff` to 10.0 |
| No charges on framework | CO2/N2 selectivity unrealistically low | Add partial charges to CIF (`_atom_site_charge`); use DDEC or EQeq charges |
| Poor convergence | Large error bars (>20% of loading) | Increase `NumberOfCycles`; ensure `ReinsertionProbability` is set; check move acceptance rates |
| Zero loading at all pressures | Framework may be non-porous or CIF malformed | Verify CIF has voids; check with pore-analysis skill; try `"NumberOfUnitCells"` |
| Swap acceptance too low | Acceptance < 0.1% for large molecules | Add `CBMCProbability` for configurational-bias moves; increase `ReinsertionProbability` |
| BET gives negative C | C < 0 from BET linearization | Adjust P/P0 range; ensure isotherm is Type I/II in BET region; check for simulation artifacts |
| BET surface area too low | S_BET << expected literature value | Check if N2 can access all pores; narrow channels may need more equilibration cycles |
| Langmuir fit poor | R^2 < 0.95 | Try dual-site Langmuir, Toth, or Sips model; material may have heterogeneous sites |
| IAST fails for strongly non-ideal mixtures | pyIAST convergence error | Use direct multi-component GCMC instead; IAST assumes ideal mixing in adsorbed phase |
| Isosteric heat noisy | Q_st oscillates with loading | Use more pressure points in isotherms; ensure isotherms at T1 and T2 are well-converged; use spline interpolation |
| Guest molecule not found | RASPA3 error about molecule definition | Check `/usr/share/raspa3/` for available molecule definitions; verify spelling matches exactly |
