# Gas Mixture Separation Analysis in Porous Materials

## When to Use

- Evaluating porous materials (MOFs, zeolites, COFs) for gas mixture separation
- Computing adsorption selectivity from multi-component GCMC simulations
- Predicting mixture behavior from single-component isotherms using IAST
- Analyzing specific separations: CO2/N2 (post-combustion capture), CO2/CH4 (natural gas purification), H2/CH4 (hydrogen purification), C2H4/C2H6 (olefin-paraffin), N2/CH4 (nitrogen rejection)
- Simulating breakthrough curves for fixed-bed adsorption columns
- Computing performance metrics: working capacity, selectivity, sorbent selection parameter, regenerability
- Screening materials for separation performance at industrially relevant conditions

## Method Selection

| Goal | Method | Notes |
|------|--------|-------|
| Direct mixture selectivity at one condition | Multi-component GCMC | Most accurate; computationally expensive |
| Selectivity over a range of conditions | IAST from single-component isotherms | Fast; requires good single-component fits; assumes ideal mixing |
| Selectivity screening across many materials | IAST + Henry coefficients | Fastest screening; valid only at low loading |
| Column breakthrough behavior | Breakthrough simulation (RASPA3) or 1D column model | Predicts dynamic performance |
| Working capacity (PSA/VSA/TSA) | Two-point GCMC or IAST | Compute loading at adsorption and desorption conditions |
| Non-ideal mixture (polar/polar) | Direct multi-component GCMC | IAST fails for strongly non-ideal systems (e.g., CO2/H2O) |

## Prerequisites

- RASPA3 binary (`raspa3`) -- pre-installed in the container
- Framework CIF file with partial charges (essential for polar gas separations)
- Single-component isotherm data or fitted parameters (for IAST approach)
- Python packages: `numpy`, `scipy`, `matplotlib` (pre-installed); optionally `pyiast` (`pip install pyiast`)

## Detailed Steps

### Step 1: Multi-component GCMC -- CO2/N2 separation

```python
#!/usr/bin/env python3
"""
Multi-component GCMC for CO2/N2 separation in a MOF.
Typical post-combustion flue gas: 15% CO2, 85% N2, T=313 K, P=1 bar.
"""
import json
import subprocess
import os
import re
import numpy as np

BASE_DIR = "/tmp/gas_separation"
FRAMEWORK = "IRMOF-1"    # Replace with your framework
TEMPERATURE = 313.0       # Post-combustion temperature
FORCE_FIELD = "GenericMOFs"
CUTOFF = 12.0
N_CYCLES = 40000
N_INIT = 20000

os.makedirs(BASE_DIR, exist_ok=True)


def build_mixture_gcmc(framework, T, P_total, components, force_field, cutoff,
                       n_cycles, n_init):
    """
    Build RASPA3 input for multi-component GCMC.

    components: list of dicts with keys: name, mol_frac, mol_def
    """
    comp_list = []
    for c in components:
        comp_dict = {
            "Name": c["name"],
            "Type": "Adsorbate",
            "MoleculeDefinition": c.get("mol_def", "TraPPE"),
            "MoleFraction": c["mol_frac"],
            "TranslationProbability": 0.5,
            "RotationProbability": 0.5,
            "ReinsertionProbability": 0.5,
            "SwapProbability": 1.0,
            "CreateNumberOfMolecules": 0
        }
        comp_list.append(comp_dict)

    return {
        "SimulationType": "MonteCarlo",
        "NumberOfCycles": n_cycles,
        "NumberOfInitializationCycles": n_init,
        "PrintEvery": max(n_cycles // 10, 500),
        "Systems": [{
            "Type": "Framework",
            "Name": framework,
            "ExternalTemperature": T,
            "ExternalPressure": P_total,
            "ChargeMethod": "Ewald",
            "ForceField": force_field,
            "CutOff": cutoff,
            "Components": comp_list
        }]
    }


def parse_mixture_output(log_path, component_names):
    """Parse per-component loading from RASPA3 multi-component output."""
    with open(log_path) as f:
        text = f.read()

    results = {}
    for name in component_names:
        # RASPA3 prints per-component loading; look for the component section
        # Pattern: Component N [name] ... Average loading absolute [mol/kg]
        pattern = (
            rf"Component\s+\d+\s*\[{re.escape(name)}\]"
            r".*?Average loading absolute\s*\[mol/kg framework\]\s*[:\s]*([\d.eE+-]+)\s*\+/-\s*([\d.eE+-]+)"
        )
        m = re.search(pattern, text, re.DOTALL)
        if m:
            results[name] = {"loading": float(m.group(1)), "error": float(m.group(2))}
        else:
            # Fallback: try a simpler sequential search
            results[name] = {"loading": 0.0, "error": 0.0}

    return results


def compute_selectivity(loadings, gas_fracs, comp_A, comp_B):
    """
    Compute adsorption selectivity S(A/B).
    S = (x_A/x_B) / (y_A/y_B)
    x = adsorbed phase mole fraction, y = gas phase mole fraction
    """
    q_A = loadings[comp_A]["loading"]
    q_B = loadings[comp_B]["loading"]
    y_A = gas_fracs[comp_A]
    y_B = gas_fracs[comp_B]

    if q_B > 0 and y_B > 0:
        S = (q_A / q_B) / (y_A / y_B)
    else:
        S = float("inf")
    return S


# === Run CO2/N2 separation at flue gas conditions ===
components = [
    {"name": "CO2", "mol_frac": 0.15, "mol_def": "TraPPE"},
    {"name": "N2",  "mol_frac": 0.85, "mol_def": "TraPPE"},
]

P_total = 1e5  # 1 bar
gas_fracs = {"CO2": 0.15, "N2": 0.85}

sim_dir = os.path.join(BASE_DIR, "co2_n2_1bar")
os.makedirs(sim_dir, exist_ok=True)

sim_input = build_mixture_gcmc(
    FRAMEWORK, TEMPERATURE, P_total, components,
    FORCE_FIELD, CUTOFF, N_CYCLES, N_INIT
)

with open(os.path.join(sim_dir, "simulation.json"), "w") as f:
    json.dump(sim_input, f, indent=2)

result = subprocess.run(["raspa3"], cwd=sim_dir,
                        capture_output=True, text=True, timeout=7200)
log_path = os.path.join(sim_dir, "output.log")
with open(log_path, "w") as f:
    f.write(result.stdout)

loadings = parse_mixture_output(log_path, ["CO2", "N2"])
S = compute_selectivity(loadings, gas_fracs, "CO2", "N2")

print(f"\n{'='*60}")
print(f"CO2/N2 Separation in {FRAMEWORK} at {TEMPERATURE} K, {P_total/1e5} bar")
print(f"{'='*60}")
print(f"CO2 loading: {loadings['CO2']['loading']:.4f} +/- {loadings['CO2']['error']:.4f} mol/kg")
print(f"N2  loading: {loadings['N2']['loading']:.4f} +/- {loadings['N2']['error']:.4f} mol/kg")
print(f"Selectivity S(CO2/N2) = {S:.1f}")
print(f"{'='*60}")
```

### Step 2: Selectivity over a range of pressures

```python
#!/usr/bin/env python3
"""
Compute selectivity as a function of total pressure.
Useful for determining optimal operating pressure for PSA/VSA.
"""
import json
import subprocess
import os
import re
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BASE_DIR = "/tmp/selectivity_vs_pressure"
FRAMEWORK = "IRMOF-1"
TEMPERATURE = 313.0
FORCE_FIELD = "GenericMOFs"
CUTOFF = 12.0
N_CYCLES = 40000
N_INIT = 20000

# Components and gas-phase composition
COMP_A = "CO2"
COMP_B = "N2"
y_A = 0.15
y_B = 0.85

# Pressure range
PRESSURES_PA = np.logspace(3, 6, 10)  # 0.01 bar to 10 bar

os.makedirs(BASE_DIR, exist_ok=True)

results_all = []

for i, P in enumerate(PRESSURES_PA):
    sim_dir = os.path.join(BASE_DIR, f"P{i:02d}_{P:.2e}")
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
            "ExternalPressure": float(P),
            "ChargeMethod": "Ewald",
            "ForceField": FORCE_FIELD,
            "CutOff": CUTOFF,
            "Components": [
                {
                    "Name": COMP_A, "Type": "Adsorbate",
                    "MoleculeDefinition": "TraPPE",
                    "MoleFraction": y_A,
                    "TranslationProbability": 0.5, "RotationProbability": 0.5,
                    "ReinsertionProbability": 0.5, "SwapProbability": 1.0,
                    "CreateNumberOfMolecules": 0
                },
                {
                    "Name": COMP_B, "Type": "Adsorbate",
                    "MoleculeDefinition": "TraPPE",
                    "MoleFraction": y_B,
                    "TranslationProbability": 0.5, "RotationProbability": 0.5,
                    "ReinsertionProbability": 0.5, "SwapProbability": 1.0,
                    "CreateNumberOfMolecules": 0
                }
            ]
        }]
    }

    with open(os.path.join(sim_dir, "simulation.json"), "w") as f:
        json.dump(sim_input, f, indent=2)

    result = subprocess.run(["raspa3"], cwd=sim_dir,
                            capture_output=True, text=True, timeout=7200)
    log_path = os.path.join(sim_dir, "output.log")
    with open(log_path, "w") as f:
        f.write(result.stdout)

    text = result.stdout
    q = {}
    for name in [COMP_A, COMP_B]:
        pat = (rf"Component\s+\d+\s*\[{re.escape(name)}\]"
               r".*?Average loading absolute\s*\[mol/kg framework\]\s*[:\s]*([\d.eE+-]+)")
        m = re.search(pat, text, re.DOTALL)
        q[name] = float(m.group(1)) if m else 0.0

    S = (q[COMP_A] / q[COMP_B]) / (y_A / y_B) if q[COMP_B] > 0 else 0
    results_all.append({"P_bar": P / 1e5, "q_A": q[COMP_A], "q_B": q[COMP_B], "S": S})
    print(f"P={P/1e5:.4f} bar: q_{COMP_A}={q[COMP_A]:.4f}, q_{COMP_B}={q[COMP_B]:.4f}, S={S:.1f}")

# Plot
P_arr = [r["P_bar"] for r in results_all]
q_A_arr = [r["q_A"] for r in results_all]
q_B_arr = [r["q_B"] for r in results_all]
S_arr = [r["S"] for r in results_all]

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

ax1.plot(P_arr, q_A_arr, "ro-", label=COMP_A, markersize=6, linewidth=2)
ax1.plot(P_arr, q_B_arr, "bs-", label=COMP_B, markersize=6, linewidth=2)
ax1.set_xlabel("Total Pressure (bar)", fontsize=14)
ax1.set_ylabel("Loading (mol/kg)", fontsize=14)
ax1.set_title(f"Mixture adsorption ({y_A*100:.0f}% {COMP_A}/{y_B*100:.0f}% {COMP_B})", fontsize=15)
ax1.set_xscale("log")
ax1.legend(fontsize=12)
ax1.grid(True, alpha=0.3)

ax2.plot(P_arr, S_arr, "g^-", markersize=7, linewidth=2)
ax2.set_xlabel("Total Pressure (bar)", fontsize=14)
ax2.set_ylabel(f"Selectivity S({COMP_A}/{COMP_B})", fontsize=14)
ax2.set_title("Adsorption selectivity vs pressure", fontsize=15)
ax2.set_xscale("log")
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(BASE_DIR, "selectivity_vs_pressure.png"),
            dpi=150, bbox_inches="tight")
plt.close()
print(f"\nPlot saved to {BASE_DIR}/selectivity_vs_pressure.png")
```

### Step 3: IAST predictions from single-component isotherms

```python
#!/usr/bin/env python3
"""
Ideal Adsorbed Solution Theory (IAST) for mixture separation prediction
from single-component isotherm data.

This is much faster than direct multi-component GCMC and is the standard
approach for initial material screening.
"""
import numpy as np
from scipy.optimize import fsolve, curve_fit
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ============ Isotherm Models ============

def langmuir(P, q_sat, K):
    """Single-site Langmuir."""
    return q_sat * K * P / (1.0 + K * P)

def dual_site_langmuir(P, q1, K1, q2, K2):
    """Dual-site Langmuir."""
    return q1*K1*P/(1+K1*P) + q2*K2*P/(1+K2*P)

def langmuir_spreading(P, q_sat, K):
    """Spreading pressure integral for Langmuir: pi*A/(RT) = q_sat*ln(1+K*P)."""
    return q_sat * np.log(1.0 + K * P)

def dsl_spreading(P, q1, K1, q2, K2):
    """Spreading pressure integral for dual-site Langmuir."""
    return q1*np.log(1+K1*P) + q2*np.log(1+K2*P)


# ============ IAST Solver ============

def iast_binary(P_total, y, isotherm_A, spreading_A, params_A,
                isotherm_B, spreading_B, params_B):
    """
    Solve IAST for a binary mixture.

    Parameters
    ----------
    P_total : float
        Total pressure (Pa)
    y : array [y_A, y_B]
        Gas phase mole fractions
    isotherm_X : callable
        Single-component isotherm function q(P, *params)
    spreading_X : callable
        Spreading pressure integral function pi(P, *params)
    params_X : tuple
        Isotherm model parameters

    Returns
    -------
    q_A, q_B : float
        Component loadings (mol/kg)
    x_A : float
        Adsorbed phase mole fraction of A
    """
    def equations(x_A):
        x_B = 1.0 - x_A
        # Raoult's law: P_total * y_i = P0_i * x_i
        P0_A = P_total * y[0] / max(x_A, 1e-30)
        P0_B = P_total * y[1] / max(x_B, 1e-30)

        # Equal spreading pressure
        pi_A = spreading_A(P0_A, *params_A)
        pi_B = spreading_B(P0_B, *params_B)
        return pi_A - pi_B

    x_A_sol = fsolve(equations, 0.5, full_output=False)[0]
    x_A_sol = np.clip(x_A_sol, 1e-12, 1.0 - 1e-12)
    x_B_sol = 1.0 - x_A_sol

    P0_A = P_total * y[0] / x_A_sol
    P0_B = P_total * y[1] / x_B_sol

    q0_A = isotherm_A(P0_A, *params_A)
    q0_B = isotherm_B(P0_B, *params_B)

    q_total = 1.0 / (x_A_sol / q0_A + x_B_sol / q0_B)
    q_A = q_total * x_A_sol
    q_B = q_total * x_B_sol

    return q_A, q_B, x_A_sol


# ============ Example: CO2/N2 IAST at multiple pressures ============

# Langmuir parameters from single-component GCMC isotherm fitting
# Replace with actual fitted values from your simulations
params_CO2 = (8.5, 4.0e-5)    # (q_sat, K_L) for CO2 at 313 K
params_N2  = (6.0, 3.0e-6)    # (q_sat, K_L) for N2 at 313 K

y_CO2 = 0.15
y_N2 = 0.85
y = [y_CO2, y_N2]

pressures = np.logspace(3, 7, 50)  # 0.01 to 100 bar
q_CO2_arr, q_N2_arr, x_CO2_arr = [], [], []

for P in pressures:
    q_A, q_B, x_A = iast_binary(
        P, y,
        langmuir, langmuir_spreading, params_CO2,
        langmuir, langmuir_spreading, params_N2
    )
    q_CO2_arr.append(q_A)
    q_N2_arr.append(q_B)
    x_CO2_arr.append(x_A)

q_CO2_arr = np.array(q_CO2_arr)
q_N2_arr = np.array(q_N2_arr)
x_CO2_arr = np.array(x_CO2_arr)
selectivity = (q_CO2_arr / q_N2_arr) / (y_CO2 / y_N2)
P_bar = pressures / 1e5

# ============ Multi-separation comparison ============
separations = {
    "CO2/N2 (flue gas)":   {"y": [0.15, 0.85], "A": (8.5, 4e-5), "B": (6.0, 3e-6)},
    "CO2/CH4 (nat. gas)":  {"y": [0.10, 0.90], "A": (8.5, 4e-5), "B": (10.0, 2e-5)},
    "H2/CH4 (H2 purif.)":  {"y": [0.50, 0.50], "A": (3.0, 1e-6), "B": (10.0, 2e-5)},
}

fig, axes = plt.subplots(1, 3, figsize=(18, 5))

for ax, (sep_name, sep) in zip(axes, separations.items()):
    S_vals = []
    for P in pressures:
        qA, qB, _ = iast_binary(
            P, sep["y"],
            langmuir, langmuir_spreading, sep["A"],
            langmuir, langmuir_spreading, sep["B"]
        )
        S = (qA / qB) / (sep["y"][0] / sep["y"][1]) if qB > 0 else 0
        S_vals.append(S)

    ax.plot(P_bar, S_vals, linewidth=2.5)
    ax.set_xlabel("Total Pressure (bar)", fontsize=13)
    ax.set_ylabel("Selectivity", fontsize=13)
    ax.set_title(sep_name, fontsize=14)
    ax.set_xscale("log")
    ax.grid(True, alpha=0.3)
    ax.tick_params(labelsize=11)

plt.tight_layout()
plt.savefig("/tmp/gas_separation/iast_multi_separation.png",
            dpi=150, bbox_inches="tight")
plt.close()
print("Multi-separation IAST plot saved.")
```

### Step 4: Working capacity and sorbent selection parameter

```python
#!/usr/bin/env python3
"""
Compute key performance metrics for pressure-swing adsorption (PSA):
1. Working capacity: delta_q = q(P_ads) - q(P_des)
2. Selectivity at adsorption conditions
3. Sorbent Selection Parameter (SSP): SSP = S^2 * delta_q
4. Regenerability: R = delta_q / q(P_ads)
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.optimize import fsolve


def langmuir(P, q_sat, K):
    return q_sat * K * P / (1.0 + K * P)

def langmuir_spreading(P, q_sat, K):
    return q_sat * np.log(1.0 + K * P)


def iast_binary(P_total, y, params_A, params_B):
    """Simplified IAST for two Langmuir components."""
    def equations(x_A):
        x_B = 1.0 - x_A
        P0_A = P_total * y[0] / max(x_A, 1e-30)
        P0_B = P_total * y[1] / max(x_B, 1e-30)
        return langmuir_spreading(P0_A, *params_A) - langmuir_spreading(P0_B, *params_B)

    x_A = np.clip(fsolve(equations, 0.5)[0], 1e-12, 1-1e-12)
    x_B = 1.0 - x_A
    P0_A = P_total * y[0] / x_A
    P0_B = P_total * y[1] / x_B
    q0_A = langmuir(P0_A, *params_A)
    q0_B = langmuir(P0_B, *params_B)
    q_t = 1.0 / (x_A/q0_A + x_B/q0_B)
    return q_t * x_A, q_t * x_B


# ============ PSA Analysis for CO2/N2 ============
# Adsorption: P_ads = 1 bar, T = 313 K
# Desorption: P_des = 0.1 bar, T = 313 K (VSA)

# Example Langmuir parameters (replace with your fitted values)
materials = {
    "MOF-A": {"CO2": (10.0, 6e-5), "N2": (7.0, 5e-6)},
    "MOF-B": {"CO2": (6.0, 2e-4),  "N2": (4.0, 8e-6)},
    "MOF-C": {"CO2": (15.0, 1e-5), "N2": (12.0, 3e-6)},
    "Zeolite-13X": {"CO2": (5.5, 5e-4), "N2": (3.0, 1e-5)},
}

y = [0.15, 0.85]  # Flue gas
P_ads = 1e5        # 1 bar
P_des = 1e4        # 0.1 bar (VSA desorption)

print(f"{'Material':<15s} {'q_CO2_ads':>10s} {'q_CO2_des':>10s} {'WC':>8s} "
      f"{'S(CO2/N2)':>10s} {'SSP':>10s} {'Regen%':>8s}")
print("-" * 75)

metrics = []
for name, params in materials.items():
    # Adsorption conditions
    q_CO2_ads, q_N2_ads = iast_binary(P_ads, y, params["CO2"], params["N2"])
    # Desorption conditions
    q_CO2_des, q_N2_des = iast_binary(P_des, y, params["CO2"], params["N2"])

    # Working capacity
    WC = q_CO2_ads - q_CO2_des  # mol/kg

    # Selectivity at adsorption conditions
    S = (q_CO2_ads / q_N2_ads) / (y[0] / y[1]) if q_N2_ads > 0 else 0

    # Sorbent Selection Parameter
    SSP = S**2 * WC

    # Regenerability
    R = WC / q_CO2_ads * 100 if q_CO2_ads > 0 else 0

    print(f"{name:<15s} {q_CO2_ads:>10.3f} {q_CO2_des:>10.3f} {WC:>8.3f} "
          f"{S:>10.1f} {SSP:>10.1f} {R:>7.1f}%")

    metrics.append({
        "name": name, "WC": WC, "S": S, "SSP": SSP, "R": R,
        "q_ads": q_CO2_ads, "q_des": q_CO2_des
    })

# === Plot: Selectivity vs Working Capacity (Pareto front) ===
fig, ax = plt.subplots(figsize=(8, 6))
for m in metrics:
    ax.scatter(m["WC"], m["S"], s=m["SSP"]*2+50, zorder=5)
    ax.annotate(m["name"], (m["WC"], m["S"]),
                textcoords="offset points", xytext=(10, 5), fontsize=10)

ax.set_xlabel("Working Capacity (mol/kg)", fontsize=14)
ax.set_ylabel("Selectivity S(CO$_2$/N$_2$)", fontsize=14)
ax.set_title("PSA Performance: Selectivity vs Working Capacity\n(bubble size = SSP)",
             fontsize=14)
ax.grid(True, alpha=0.3)
ax.tick_params(labelsize=12)
plt.tight_layout()
plt.savefig("/tmp/gas_separation/psa_performance.png", dpi=150, bbox_inches="tight")
plt.close()
print("\nPerformance plot saved to /tmp/gas_separation/psa_performance.png")
```

### Step 5: Breakthrough curve simulation

```python
#!/usr/bin/env python3
"""
Simple 1D breakthrough curve model for fixed-bed adsorption column.
Uses the Thomas model (analytical) and a numerical finite-difference model.

For RASPA3 built-in breakthrough simulations, check:
  ls /usr/share/raspa3/examples/breakthrough/
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.special import erfc


# ============ Thomas Model (Analytical) ============
def thomas_model(t, K_TH, q_0, m, C_0, Q):
    """
    Thomas model for breakthrough curve.

    C/C0 = 1 / (1 + exp(K_TH * (q_0*m/Q - C_0*t)))

    Parameters
    ----------
    t : array
        Time (minutes)
    K_TH : float
        Thomas rate constant (L/(min*mg))
    q_0 : float
        Maximum adsorption capacity (mg/g)
    m : float
        Mass of adsorbent (g)
    C_0 : float
        Feed concentration (mg/L)
    Q : float
        Volumetric flow rate (L/min)
    """
    exponent = K_TH * (q_0 * m / Q - C_0 * t)
    # Clip to avoid overflow
    exponent = np.clip(exponent, -500, 500)
    return 1.0 / (1.0 + np.exp(exponent))


# ============ 1D Column Model (Finite Difference) ============
def breakthrough_1d(params):
    """
    1D advection-dispersion with Langmuir adsorption.

    Returns time and exit concentration profile.
    """
    # Unpack parameters
    L = params["column_length_cm"]       # cm
    v = params["velocity_cm_min"]        # cm/min
    D_L = params["dispersion_cm2_min"]   # axial dispersion cm^2/min
    epsilon = params["void_fraction"]     # bed void fraction
    rho_b = params["bulk_density_g_cm3"] # bulk density g/cm^3
    q_sat = params["q_sat_mg_g"]         # Langmuir q_sat mg/g
    K_L = params["K_L_L_mg"]            # Langmuir K (L/mg)
    C_0 = params["feed_conc_mg_L"]      # feed concentration mg/L
    t_max = params["t_max_min"]          # simulation time minutes

    # Discretization
    nz = 200
    dz = L / nz
    dt = 0.5 * min(dz / v, dz**2 / (2 * D_L)) if D_L > 0 else 0.5 * dz / v
    nt = int(t_max / dt) + 1

    C = np.zeros(nz)   # gas phase concentration
    q = np.zeros(nz)   # adsorbed phase concentration (mg/g)

    t_out = []
    C_out = []

    for n in range(nt):
        t = n * dt

        # Save exit concentration periodically
        if n % max(nt // 500, 1) == 0:
            t_out.append(t)
            C_out.append(C[-1] / C_0 if C_0 > 0 else 0)

        # Compute equilibrium loading (Langmuir)
        q_eq = q_sat * K_L * C / (1.0 + K_L * C)

        # Mass transfer (linear driving force)
        k_LDF = params.get("k_LDF_1_min", 0.5)  # 1/min
        dqdt = k_LDF * (q_eq - q)

        # Advection + dispersion (upwind + central difference)
        C_new = C.copy()
        for i in range(1, nz):
            advection = -v * (C[i] - C[i-1]) / dz
            if D_L > 0 and i < nz - 1:
                dispersion = D_L * (C[i+1] - 2*C[i] + C[i-1]) / dz**2
            else:
                dispersion = 0.0
            adsorption = -rho_b / epsilon * dqdt[i]
            C_new[i] = C[i] + dt * (advection + dispersion + adsorption)

        # Boundary condition: inlet
        C_new[0] = C_0

        # Update
        C = np.maximum(C_new, 0)
        q += dt * dqdt

    return np.array(t_out), np.array(C_out)


# ============ Example: CO2 breakthrough in MOF column ============
# Thomas model
t = np.linspace(0, 120, 500)  # minutes
C_C0_thomas = thomas_model(
    t, K_TH=0.05, q_0=80.0, m=10.0, C_0=50.0, Q=0.1
)

# 1D numerical model
column_params = {
    "column_length_cm": 20.0,
    "velocity_cm_min": 5.0,
    "dispersion_cm2_min": 0.5,
    "void_fraction": 0.4,
    "bulk_density_g_cm3": 0.5,
    "q_sat_mg_g": 120.0,
    "K_L_L_mg": 0.02,
    "feed_conc_mg_L": 50.0,
    "t_max_min": 120.0,
    "k_LDF_1_min": 0.3,
}

t_1d, C_1d = breakthrough_1d(column_params)

# Plot
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

ax1.plot(t, C_C0_thomas, "b-", linewidth=2, label="Thomas model")
ax1.axhline(y=0.05, color="gray", linestyle="--", alpha=0.5, label="5% breakthrough")
ax1.axhline(y=0.95, color="gray", linestyle=":", alpha=0.5, label="95% saturation")
ax1.set_xlabel("Time (min)", fontsize=14)
ax1.set_ylabel("C/C$_0$", fontsize=14)
ax1.set_title("Thomas model breakthrough curve", fontsize=15)
ax1.legend(fontsize=11)
ax1.grid(True, alpha=0.3)
ax1.set_ylim(-0.05, 1.05)

ax2.plot(t_1d, C_1d, "r-", linewidth=2, label="1D column model")
ax2.axhline(y=0.05, color="gray", linestyle="--", alpha=0.5, label="5% breakthrough")
ax2.set_xlabel("Time (min)", fontsize=14)
ax2.set_ylabel("C/C$_0$", fontsize=14)
ax2.set_title("1D numerical breakthrough", fontsize=15)
ax2.legend(fontsize=11)
ax2.grid(True, alpha=0.3)
ax2.set_ylim(-0.05, 1.05)

plt.tight_layout()
plt.savefig("/tmp/gas_separation/breakthrough_curves.png", dpi=150, bbox_inches="tight")
plt.close()
print("Breakthrough curve plots saved.")

# Compute breakthrough time (C/C0 = 0.05)
t_break_thomas = t[np.argmax(C_C0_thomas > 0.05)]
t_break_1d = t_1d[np.argmax(C_1d > 0.05)] if np.any(C_1d > 0.05) else t_1d[-1]
print(f"\nBreakthrough time (5%):")
print(f"  Thomas model: {t_break_thomas:.1f} min")
print(f"  1D numerical: {t_break_1d:.1f} min")
```

### Step 6: Common gas separation pairs -- complete comparison

```python
#!/usr/bin/env python3
"""
Evaluate a single material for multiple industrially relevant
gas separation pairs using IAST.
"""
import numpy as np
from scipy.optimize import fsolve
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def langmuir(P, q_sat, K):
    return q_sat * K * P / (1.0 + K * P)

def langmuir_spreading(P, q_sat, K):
    return q_sat * np.log(1.0 + K * P)

def iast_selectivity(P_total, y, params_A, params_B):
    def eq(x_A):
        x_B = 1.0 - x_A
        P0_A = P_total * y[0] / max(x_A, 1e-30)
        P0_B = P_total * y[1] / max(x_B, 1e-30)
        return langmuir_spreading(P0_A, *params_A) - langmuir_spreading(P0_B, *params_B)

    x_A = np.clip(fsolve(eq, 0.5)[0], 1e-12, 1-1e-12)
    x_B = 1.0 - x_A
    P0_A = P_total * y[0] / x_A
    P0_B = P_total * y[1] / x_B
    q0_A = langmuir(P0_A, *params_A)
    q0_B = langmuir(P0_B, *params_B)
    q_t = 1.0 / (x_A/q0_A + x_B/q0_B)
    q_A, q_B = q_t * x_A, q_t * x_B
    S = (q_A / q_B) / (y[0] / y[1]) if q_B > 0 else float("inf")
    return q_A, q_B, S


# Single-component Langmuir parameters at 298 K for hypothetical MOF
# Replace with actual fitted parameters from your GCMC isotherms
gas_params = {
    "CO2":  (8.0, 5e-5),
    "N2":   (6.0, 4e-6),
    "CH4":  (10.0, 2e-5),
    "H2":   (3.0, 8e-7),
    "C2H4": (7.0, 8e-5),
    "C2H6": (7.5, 6e-5),
    "H2O":  (12.0, 3e-4),  # Note: IAST less reliable for H2O
}

# Industrial separation pairs
separation_pairs = [
    {"name": "CO2/N2 (flue gas)",         "A": "CO2",  "B": "N2",   "y_A": 0.15, "P_bar": 1.0},
    {"name": "CO2/CH4 (natural gas)",     "A": "CO2",  "B": "CH4",  "y_A": 0.10, "P_bar": 5.0},
    {"name": "H2/CH4 (H2 purification)",  "A": "CH4",  "B": "H2",   "y_A": 0.50, "P_bar": 30.0},
    {"name": "C2H4/C2H6 (olefin)",        "A": "C2H4", "B": "C2H6", "y_A": 0.50, "P_bar": 1.0},
    {"name": "N2/CH4 (N2 rejection)",     "A": "N2",   "B": "CH4",  "y_A": 0.10, "P_bar": 10.0},
    {"name": "CO2/H2 (pre-combustion)",   "A": "CO2",  "B": "H2",   "y_A": 0.40, "P_bar": 30.0},
]

print(f"\n{'Separation':<30s} {'S(A/B)':>8s} {'q_A (mol/kg)':>13s} {'q_B (mol/kg)':>13s}")
print("=" * 68)

for sep in separation_pairs:
    y = [sep["y_A"], 1.0 - sep["y_A"]]
    P_total = sep["P_bar"] * 1e5
    pA = gas_params[sep["A"]]
    pB = gas_params[sep["B"]]
    q_A, q_B, S = iast_selectivity(P_total, y, pA, pB)
    print(f"{sep['name']:<30s} {S:>8.1f} {q_A:>13.3f} {q_B:>13.3f}")

    # Quality assessment
    if S > 100:
        quality = "Excellent"
    elif S > 20:
        quality = "Good"
    elif S > 5:
        quality = "Moderate"
    else:
        quality = "Poor"
    print(f"{'':>30s} Assessment: {quality}")
```

### Step 7: Using RASPA3 breakthrough simulation

```bash
# RASPA3 has built-in breakthrough simulation capabilities.
# Check official examples:
ls /usr/share/raspa3/examples/breakthrough/ 2>/dev/null || echo "Check RASPA3 examples directory"

# Copy and modify a breakthrough example:
# cp -r /usr/share/raspa3/examples/breakthrough/example1 /tmp/breakthrough_sim
# cd /tmp/breakthrough_sim
# Edit simulation.json to configure column parameters
# raspa3
```

```python
#!/usr/bin/env python3
"""
Setup RASPA3 breakthrough simulation input.
RASPA3 can simulate fixed-bed breakthrough directly.
"""
import json
import os

breakthrough_input = {
    "SimulationType": "Breakthrough",
    "NumberOfTimeSteps": 10000,
    "PrintEvery": 100,
    "Systems": [{
        "Type": "Framework",
        "Name": "IRMOF-1",
        "ExternalTemperature": 298.0,
        "TotalPressure": 1e5,        # 1 bar inlet
        "ColumnLength": 0.30,         # meters
        "ColumnVoidFraction": 0.4,
        "ParticleDensity": 1000.0,    # kg/m^3
        "ChargeMethod": "Ewald",
        "ForceField": "GenericMOFs",
        "CutOff": 12.0,
        "Components": [
            {
                "Name": "CO2",
                "Type": "Adsorbate",
                "MoleculeDefinition": "TraPPE",
                "MoleFraction": 0.15,
                "IsothermModel": "Langmuir",
                "IsothermParameters": {
                    "q_sat": 8.5,      # mol/kg
                    "K": 4e-5          # 1/Pa
                }
            },
            {
                "Name": "N2",
                "Type": "Adsorbate",
                "MoleculeDefinition": "TraPPE",
                "MoleFraction": 0.85,
                "IsothermModel": "Langmuir",
                "IsothermParameters": {
                    "q_sat": 6.0,
                    "K": 3e-6
                }
            }
        ]
    }]
}

out_dir = "/tmp/breakthrough_raspa3"
os.makedirs(out_dir, exist_ok=True)
with open(os.path.join(out_dir, "simulation.json"), "w") as f:
    json.dump(breakthrough_input, f, indent=2)
print(f"RASPA3 breakthrough input written to {out_dir}/simulation.json")
print("Note: Verify exact field names against RASPA3 documentation and examples.")
print("The breakthrough simulation mode may differ between RASPA3 versions.")
```

## Key Parameters

| Parameter | Description | Typical Value |
|-----------|-------------|---------------|
| `MoleFraction` | Gas-phase composition of each component | Must sum to 1.0; e.g., 0.15 CO2 + 0.85 N2 |
| `ExternalPressure` | Total pressure (Pa) | 1e5 (ambient), 3e6 (natural gas), 3e7 (H2 purification) |
| `ExternalTemperature` | Temperature (K) | 298 (ambient), 313 (flue gas), 77 (cryogenic) |
| `NumberOfCycles` | Production MC cycles | 30000-100000; mixtures need more than single-component |
| Langmuir `q_sat` | Saturation capacity per component (mol/kg) | 1-30; fit from single-component isotherms |
| Langmuir `K_L` | Affinity constant (1/Pa) | 1e-7 to 1e-3; determines Henry region slope |
| PSA `P_ads` | Adsorption pressure | Typically 1-30 bar |
| PSA `P_des` | Desorption pressure (VSA) | 0.01-0.1 bar for VSA; 1 bar for PSA with purge |
| TSA `T_des` | Desorption temperature | 373-473 K for TSA regeneration |
| Column void fraction | Bed porosity | 0.35-0.45 for packed beds |
| Particle density | Adsorbent density (kg/m^3) | 500-1500 depending on material |

## Interpreting Results

### Selectivity benchmarks
| Separation | S (Poor) | S (Good) | S (Excellent) | Best known materials |
|------------|----------|----------|---------------|---------------------|
| CO2/N2 | < 10 | 20-100 | > 100 | Mg-MOF-74, SIFSIX, zeolite 13X |
| CO2/CH4 | < 5 | 10-50 | > 50 | ZIF-8, MOF-508, UTSA-16 |
| H2/CH4 | < 5 | 10-50 | > 50 | CuBTC, ZIF-7, zeolite 5A |
| C2H4/C2H6 | < 2 | 2-10 | > 10 | Fe-MOF-74, NOTT-300 |
| N2/CH4 | < 2 | 3-10 | > 10 | MIL-120, Ba-ETS-4 |

### Sorbent Selection Parameter (SSP)
- SSP = S^2 * delta_q (combines selectivity and capacity)
- Higher SSP = better overall PSA performance
- Useful for comparing materials that trade off selectivity vs capacity
- Materials on the Pareto front of S vs delta_q are optimal candidates

### Breakthrough curve metrics
- **Breakthrough time**: time until outlet C/C0 = 0.05 (5% of feed)
- **Saturation time**: time until C/C0 = 0.95
- **Stoichiometric time**: area above the breakthrough curve = total capacity used
- **Mass transfer zone length**: column length between 5% and 95% breakthrough
- Sharper breakthrough = better mass transfer kinetics

### IAST validity
- IAST is exact for ideal adsorbed solutions (analogous to Raoult's law)
- Works well for: non-polar/non-polar mixtures, similar-sized molecules
- Less reliable for: polar/non-polar mixtures, strong sorbent-sorbate interactions, very different molecular sizes
- Validate IAST predictions against direct multi-component GCMC for key conditions

## Common Issues

| Problem | Symptom | Solution |
|---------|---------|----------|
| IAST does not converge | fsolve returns nonsense or oscillates | Provide better initial guess; check that single-component isotherms are well-fit; try bisection method |
| IAST overpredicts selectivity | S_IAST >> S_GCMC | System is non-ideal (competitive adsorption); use direct multi-component GCMC |
| Low selectivity for CO2/N2 | S < 10 despite good CO2 capacity | Framework likely lacks strong CO2 binding sites; add open metal sites or amine functionalization |
| Working capacity near zero | q_ads ~ q_des | Operating pressures are both in saturated region; widen P_ads/P_des ratio or use TSA |
| Multi-component GCMC does not converge | Loadings fluctuate wildly | Increase cycles to 50000+; check mole fractions sum to 1.0; increase equilibration |
| Wrong selectivity sign | S < 1 when A should be preferred | Check which component is A vs B in selectivity formula; verify gas-phase mole fractions |
| Breakthrough too fast | Very short breakthrough time | Increase column length, reduce flow rate, or use material with higher capacity |
| Negative working capacity | q(P_des) > q(P_ads) | Likely a fitting error; check isotherm parameters; ensure desorption P < adsorption P |
| Framework charges missing | CO2/N2 selectivity unrealistically close to 1 | CO2 selectivity is dominated by electrostatics; add DDEC/EQeq charges to CIF |
| Component loading not parsed | Parser returns 0 for one component | RASPA3 multi-component output format may differ; examine raw output log manually |
