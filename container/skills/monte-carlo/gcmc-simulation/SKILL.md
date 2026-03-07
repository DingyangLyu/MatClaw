# Grand Canonical Monte Carlo (GCMC) Simulation with RASPA3

## When to Use

- Setting up and running GCMC simulations for gas adsorption in porous materials
- Configuring RASPA3 input files: `simulation.json`, force field parameters, molecule definitions
- Understanding and tuning Lennard-Jones and Coulombic interaction parameters
- Configuring Ewald summation for long-range electrostatics
- Running simulations with flexible or rigid framework options
- Analyzing GCMC output: loading vs pressure, energy components, acceptance ratios
- Troubleshooting simulation convergence, force field selection, and performance

## Method Selection

| Goal | Approach | Notes |
|------|----------|-------|
| Quick screening (rigid framework) | Standard GCMC with rigid CIF | Fastest; suitable for most screening studies |
| Accurate adsorption in flexible MOFs | GCMC with flexible framework | Much slower; needed for breathing MOFs (MIL-53) or gate-opening |
| Non-polar guest (CH4, H2, noble gases) | LJ-only interactions | Skip `ChargeMethod`; 3-5x faster |
| Polar guest (CO2, H2O, NH3) | LJ + Coulomb with Ewald | Requires framework charges in CIF |
| Custom force field parameters | User-defined `force_field.json` | Override default LJ sigma/epsilon for specific atom types |
| High-throughput screening | Python scripting over many structures | Automate input generation and output parsing |
| Confined molecules (tight pores) | Add CBMC moves | Configurational-bias MC for efficient insertion |

## Prerequisites

- RASPA3 binary (`raspa3`) -- pre-installed in the container
- Framework CIF file with proper symmetry (P1 recommended)
- Knowledge of guest molecules, temperature, pressure, and desired ensemble
- Official examples at `/usr/share/raspa3/examples/` -- always start from these
- For charged systems: CIF with `_atom_site_charge` column (DDEC, EQeq, or REPEAT charges)

## Detailed Steps

### Step 0: Explore RASPA3 examples and directory structure

```bash
# List all official examples
ls /usr/share/raspa3/examples/

# Typical structure:
# basic/           -- simple MC in box, LJ fluid
# adsorption/      -- GCMC in frameworks
# breakthrough/    -- breakthrough simulations
# each example has simulation.json and possibly force_field/ and molecules/ dirs

# Copy an example as starting point
cp -r /usr/share/raspa3/examples/basic/1_mc_methane_in_box /tmp/my_gcmc
cd /tmp/my_gcmc && cat simulation.json
```

### Step 1: Complete simulation.json anatomy

```python
#!/usr/bin/env python3
"""
Annotated RASPA3 simulation.json builder with all key options.
"""
import json
import os

def build_gcmc_simulation(
    framework_name="IRMOF-1",
    temperature=298.0,
    pressure=1e5,
    guests=None,
    force_field="GenericMOFs",
    cutoff=12.0,
    charge_method="Ewald",
    n_cycles=25000,
    n_init=10000,
    n_unit_cells=None,
    use_tabulated_energies=False
):
    """
    Build a complete RASPA3 simulation.json for GCMC.

    Parameters
    ----------
    framework_name : str
        Name matching the CIF file (without .cif extension)
    temperature : float
        Temperature in Kelvin
    pressure : float
        Total pressure in Pascals
    guests : list of dict
        Each dict: {"name": str, "mol_def": str, "mol_frac": float}
    force_field : str
        Force field name (GenericMOFs, GenericZeolites, UFF)
    cutoff : float
        LJ cutoff in Angstrom
    charge_method : str
        "Ewald" for charged systems, "None" for non-polar guests
    n_cycles : int
        Production MC cycles
    n_init : int
        Equilibration cycles
    n_unit_cells : list of 3 int or None
        Supercell replication [nx, ny, nz]; None = auto
    use_tabulated_energies : bool
        Pre-tabulate energies on a grid (speeds up large systems)
    """
    if guests is None:
        guests = [{"name": "methane", "mol_def": "TraPPE", "mol_frac": 1.0}]

    components = []
    for g in guests:
        comp = {
            "Name": g["name"],
            "Type": "Adsorbate",
            "MoleculeDefinition": g.get("mol_def", "TraPPE"),
            "TranslationProbability": 0.5,
            "ReinsertionProbability": 0.5,
            "SwapProbability": 1.0,
            "CreateNumberOfMolecules": 0
        }
        # Multi-site molecules need rotation
        single_site = ["methane", "helium", "neon", "argon", "krypton", "xenon"]
        if g["name"].lower() not in single_site:
            comp["RotationProbability"] = 0.5

        # For mixtures, add mole fraction
        if len(guests) > 1:
            comp["MoleFraction"] = g.get("mol_frac", 1.0 / len(guests))

        components.append(comp)

    system = {
        "Type": "Framework",
        "Name": framework_name,
        "ExternalTemperature": temperature,
        "ExternalPressure": pressure,
        "ChargeMethod": charge_method,
        "ForceField": force_field,
        "CutOff": cutoff,
        "Components": components
    }

    if n_unit_cells is not None:
        system["NumberOfUnitCells"] = n_unit_cells

    if use_tabulated_energies:
        system["UseTabularGrid"] = True

    sim = {
        "SimulationType": "MonteCarlo",
        "NumberOfCycles": n_cycles,
        "NumberOfInitializationCycles": n_init,
        "PrintEvery": max(n_cycles // 10, 100),
        "Systems": [system]
    }

    return sim


# === Example 1: Methane in MOF (simple, non-polar) ===
sim1 = build_gcmc_simulation(
    framework_name="IRMOF-1",
    temperature=298.0,
    pressure=1e6,
    guests=[{"name": "methane", "mol_def": "TraPPE"}],
    force_field="GenericMOFs",
    charge_method="None",    # CH4 is non-polar, skip Ewald
    n_cycles=25000,
    n_init=10000
)

# === Example 2: CO2/N2 mixture in MOF (polar, needs charges) ===
sim2 = build_gcmc_simulation(
    framework_name="Mg-MOF-74",
    temperature=313.0,         # Post-combustion temperature
    pressure=1e5,
    guests=[
        {"name": "CO2", "mol_def": "TraPPE", "mol_frac": 0.15},
        {"name": "N2",  "mol_def": "TraPPE", "mol_frac": 0.85},
    ],
    force_field="GenericMOFs",
    charge_method="Ewald",
    n_cycles=40000,
    n_init=20000,
    n_unit_cells=[1, 1, 2]    # Elongate along channel direction
)

# === Example 3: H2 in zeolite at cryogenic T ===
sim3 = build_gcmc_simulation(
    framework_name="MFI",
    temperature=77.0,
    pressure=1e5,
    guests=[{"name": "H2", "mol_def": "TraPPE"}],
    force_field="GenericZeolites",
    charge_method="Ewald",
    n_cycles=30000,
    n_init=15000
)

# Write examples
for name, sim in [("mof_ch4", sim1), ("mof_co2n2", sim2), ("zeolite_h2", sim3)]:
    out_dir = f"/tmp/gcmc_examples/{name}"
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "simulation.json"), "w") as f:
        json.dump(sim, f, indent=2)
    print(f"Written: {out_dir}/simulation.json")
```

### Step 2: Force field configuration

```python
#!/usr/bin/env python3
"""
Configure custom force field parameters for RASPA3.

RASPA3 looks for force field definitions in:
1. Built-in force fields (GenericMOFs, GenericZeolites, UFF)
2. Custom force_field.json in the simulation directory

Lennard-Jones potential: V(r) = 4*eps*[(sigma/r)^12 - (sigma/r)^6]
Lorentz-Berthelot mixing rules (default):
  sigma_ij = (sigma_i + sigma_j) / 2
  eps_ij = sqrt(eps_i * eps_j)
"""
import json
import os

# =============== Custom Force Field Definition ===============
# This overrides or supplements the built-in force field.
# Define atom-type-specific LJ parameters.

custom_ff = {
    "ForceFieldName": "CustomMOF",
    "CutOff": 12.0,
    "MixingRule": "Lorentz-Berthelot",  # or "Jorgensen" (geometric mean for sigma too)
    "TailCorrections": True,            # Long-range LJ tail corrections
    "ShiftPotential": False,            # If True, shift LJ to zero at cutoff
    "Atoms": {
        # Framework atom types
        "Zn": {"epsilon_over_kB": 62.399,  "sigma": 2.462},   # K, Angstrom
        "O_mof": {"epsilon_over_kB": 48.158, "sigma": 3.033},
        "C_mof": {"epsilon_over_kB": 47.856, "sigma": 3.473},
        "H_mof": {"epsilon_over_kB": 7.649,  "sigma": 2.846},
        # Guest atom types (from TraPPE)
        "CH4_sp3": {"epsilon_over_kB": 148.0, "sigma": 3.73},
        "C_co2": {"epsilon_over_kB": 27.0,  "sigma": 2.80},
        "O_co2": {"epsilon_over_kB": 79.0,  "sigma": 3.05},
        "N_n2":  {"epsilon_over_kB": 36.0,  "sigma": 3.31},
    },
    # Optional: explicit pair interactions (override mixing rules)
    "PairInteractions": {
        # "Zn-O_co2": {"epsilon_over_kB": 65.0, "sigma": 2.75}
    }
}


def write_force_field(ff_dict, output_dir):
    """Write force field JSON file for RASPA3."""
    ff_path = os.path.join(output_dir, "force_field.json")
    with open(ff_path, "w") as f:
        json.dump(ff_dict, f, indent=2)
    print(f"Force field written to {ff_path}")

    # Print summary
    print(f"\nForce field: {ff_dict['ForceFieldName']}")
    print(f"Mixing rule: {ff_dict['MixingRule']}")
    print(f"Tail corrections: {ff_dict['TailCorrections']}")
    print(f"\n{'Atom Type':<12s} {'eps/kB (K)':<12s} {'sigma (A)':<10s}")
    print("-" * 34)
    for atom, params in ff_dict["Atoms"].items():
        print(f"{atom:<12s} {params['epsilon_over_kB']:<12.3f} {params['sigma']:<10.3f}")


output_dir = "/tmp/gcmc_examples/custom_ff"
os.makedirs(output_dir, exist_ok=True)
write_force_field(custom_ff, output_dir)
```

### Step 3: Lennard-Jones and Coulombic interaction details

```python
#!/usr/bin/env python3
"""
Understanding and computing LJ and Coulombic interactions
used in GCMC simulations.
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ============ Lennard-Jones Potential ============
def lj_potential(r, epsilon, sigma):
    """V_LJ(r) = 4*epsilon*[(sigma/r)^12 - (sigma/r)^6]"""
    sr6 = (sigma / r)**6
    return 4.0 * epsilon * (sr6**2 - sr6)


def lj_force(r, epsilon, sigma):
    """F_LJ(r) = 24*epsilon/r * [2*(sigma/r)^12 - (sigma/r)^6]"""
    sr6 = (sigma / r)**6
    return 24.0 * epsilon / r * (2.0 * sr6**2 - sr6)


# ============ Lorentz-Berthelot Mixing Rules ============
def lorentz_berthelot(eps_i, sig_i, eps_j, sig_j):
    """Compute mixed LJ parameters."""
    eps_ij = np.sqrt(eps_i * eps_j)
    sig_ij = (sig_i + sig_j) / 2.0
    return eps_ij, sig_ij


# ============ Coulombic Interaction ============
def coulomb_potential(r, q_i, q_j):
    """V_coul(r) = k_e * q_i * q_j / r  (in SI, returns Joules)
    k_e = 1/(4*pi*eps_0) = 8.9875e9 N*m^2/C^2
    For RASPA3: charges in units of electron charge (e)
    """
    k_e = 8.9875e9  # N*m^2/C^2
    e = 1.602e-19   # Coulombs
    r_m = r * 1e-10  # Angstrom to meters
    V_J = k_e * q_i * q_j * e**2 / r_m
    V_kJ_mol = V_J * 6.022e23 / 1000.0
    return V_kJ_mol


# ============ Example: CO2 interacting with framework Zn ============
# TraPPE CO2: C (q=+0.70, eps/kB=27.0 K, sigma=2.80 A)
#             O (q=-0.35, eps/kB=79.0 K, sigma=3.05 A)
# Framework Zn (eps/kB=62.4 K, sigma=2.46 A, charge depends on method)

eps_Zn = 62.4 * 1.380649e-23 / 1000 * 6.022e23  # K -> kJ/mol
sig_Zn = 2.462

eps_O_co2 = 79.0 * 1.380649e-23 / 1000 * 6.022e23
sig_O_co2 = 3.05

eps_mix, sig_mix = lorentz_berthelot(eps_Zn, sig_Zn, eps_O_co2, sig_O_co2)

# Plot LJ potential for the Zn-O_CO2 interaction
r_range = np.linspace(2.5, 10.0, 200)
V_lj = np.array([lj_potential(r, eps_mix, sig_mix) for r in r_range])

fig, ax = plt.subplots(figsize=(8, 5))
ax.plot(r_range, V_lj, "b-", linewidth=2, label=f"Zn--O$_{{CO2}}$ (LJ)")
ax.axhline(y=0, color="k", linewidth=0.5)
ax.set_xlabel("r (Angstrom)", fontsize=14)
ax.set_ylabel("V (kJ/mol)", fontsize=14)
ax.set_title("Lennard-Jones interaction", fontsize=15)
ax.set_ylim(-2*abs(eps_mix), 3*abs(eps_mix))
ax.legend(fontsize=12)
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("/tmp/gcmc_examples/lj_potential.png", dpi=150, bbox_inches="tight")
plt.close()
print("LJ potential plot saved.")

print(f"\nMixed parameters (Lorentz-Berthelot):")
print(f"  eps_mix = {eps_mix:.4f} kJ/mol")
print(f"  sig_mix = {sig_mix:.3f} A")
```

### Step 4: Ewald summation configuration

```python
#!/usr/bin/env python3
"""
Ewald summation parameters for long-range electrostatics in RASPA3.

The Ewald method splits the Coulombic sum into:
1. Real-space sum (short-range, decays with erfc)
2. Reciprocal-space sum (long-range, converges in Fourier space)
3. Self-energy correction

RASPA3 handles Ewald automatically when ChargeMethod="Ewald" is set.
This script explains the key tuning parameters.
"""
import numpy as np

def estimate_ewald_params(box_lengths, cutoff, precision=1e-6):
    """
    Estimate Ewald parameters for a given box and cutoff.

    Parameters
    ----------
    box_lengths : array of 3 floats
        Cell dimensions [a, b, c] in Angstrom
    cutoff : float
        Real-space cutoff in Angstrom
    precision : float
        Target precision (default 1e-6)

    Returns
    -------
    alpha : float
        Ewald splitting parameter (1/Angstrom)
    k_max : array of 3 int
        Maximum k-vectors in each direction
    """
    # Alpha: controls the split between real and reciprocal space
    # Higher alpha = faster real-space convergence but slower k-space
    alpha = np.sqrt(-np.log(precision)) / cutoff

    # Maximum k-vectors: ensure reciprocal sum converges to same precision
    k_max = []
    for L in box_lengths:
        k = int(np.ceil(alpha * L / np.pi * np.sqrt(-np.log(precision)))) + 1
        k_max.append(k)

    return alpha, k_max


# Example: IRMOF-1 (cubic, a ~ 25.83 A)
box = [25.83, 25.83, 25.83]
cutoff = 12.0

alpha, k_max = estimate_ewald_params(box, cutoff)

print("=== Ewald Summation Parameters ===")
print(f"Box dimensions: {box} A")
print(f"Real-space cutoff: {cutoff} A")
print(f"Alpha (splitting parameter): {alpha:.4f} 1/A")
print(f"k_max vectors: {k_max}")
print(f"\nRASPA3 computes these automatically. Manual override is rarely needed.")
print(f"If the box is very elongated (e.g., nanotube), consider adjusting k_max.")

# ============ When to skip Ewald ============
print(f"""
=== When to use which ChargeMethod ===

"Ewald"   - Framework has partial charges AND guest is polar (CO2, H2O, NH3)
            Required for quantitative selectivity in separation studies
            Cost: O(N^1.5) per cycle

"None"    - Guest is non-polar (CH4, H2, noble gases)
            OR quick screening where charge effects are secondary
            Cost: much faster (LJ only)

"Wolf"    - Alternative to Ewald for very large systems (>10000 atoms)
            Approximate but much faster
            Use with caution; validate against Ewald for your system

Note: RASPA3 reads charges from the CIF file (_atom_site_charge column).
If charges are absent, Ewald will still run but all charges are zero (no effect).
""")
```

### Step 5: Framework flexibility options

```python
#!/usr/bin/env python3
"""
Framework flexibility in GCMC with RASPA3.

Most GCMC studies use rigid frameworks (atoms fixed at CIF positions).
Flexible framework GCMC is needed for:
- Breathing MOFs (MIL-53, DUT-49)
- Gate-opening MOFs (ZIF-8 at high pressure)
- Adsorption-induced structural transitions
"""
import json
import os

# ============ Rigid Framework (default, most common) ============
rigid_input = {
    "SimulationType": "MonteCarlo",
    "NumberOfCycles": 25000,
    "NumberOfInitializationCycles": 10000,
    "PrintEvery": 2500,
    "Systems": [{
        "Type": "Framework",
        "Name": "IRMOF-1",
        "ExternalTemperature": 298.0,
        "ExternalPressure": 1e5,
        "FrameworkModel": "Rigid",     # Explicit rigid (also the default)
        "ChargeMethod": "Ewald",
        "ForceField": "GenericMOFs",
        "CutOff": 12.0,
        "Components": [{
            "Name": "CO2",
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

# ============ Flexible Framework ============
# Framework atoms can move during the simulation.
# Requires intra-framework force field (bonds, angles, dihedrals, impropers).
flexible_input = {
    "SimulationType": "MonteCarlo",
    "NumberOfCycles": 50000,         # More cycles needed
    "NumberOfInitializationCycles": 25000,
    "PrintEvery": 5000,
    "Systems": [{
        "Type": "Framework",
        "Name": "MIL-53",
        "ExternalTemperature": 298.0,
        "ExternalPressure": 1e5,
        "FrameworkModel": "Flexible",
        "ChargeMethod": "Ewald",
        "ForceField": "GenericMOFs",
        "CutOff": 12.0,
        # Framework MC moves
        "FrameworkTranslationProbability": 0.5,  # Move framework atoms
        "VolumeChangeProbability": 0.01,          # NPT-like volume moves (for breathing)
        "Components": [{
            "Name": "CO2",
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

for name, inp in [("rigid", rigid_input), ("flexible", flexible_input)]:
    out_dir = f"/tmp/gcmc_examples/framework_{name}"
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "simulation.json"), "w") as f:
        json.dump(inp, f, indent=2)
    print(f"Written: {out_dir}/simulation.json")

print("""
=== Rigid vs Flexible Framework Guidelines ===

Use RIGID when:
  - Framework is known to be rigid (IRMOF-1, UiO-66, most zeolites)
  - Screening many structures (speed matters)
  - Guest does not induce structural changes

Use FLEXIBLE when:
  - Framework undergoes breathing transitions (MIL-53, DUT-49)
  - Gate-opening behavior is expected (ZIF-8 at high loading)
  - Computing adsorption-induced stress or strain
  - Validating rigid approximation

Performance impact:
  - Flexible GCMC is 10-100x slower than rigid
  - Requires well-parameterized intra-framework force field
  - May need longer equilibration (more initialization cycles)
""")
```

### Step 6: Run GCMC and analyze output

```python
#!/usr/bin/env python3
"""
Complete GCMC workflow: run simulation, parse output, analyze results.
"""
import json
import subprocess
import os
import re
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def run_gcmc(sim_dir, timeout=7200):
    """Run RASPA3 GCMC in the given directory."""
    result = subprocess.run(
        ["raspa3"], cwd=sim_dir,
        capture_output=True, text=True, timeout=timeout
    )

    log_path = os.path.join(sim_dir, "output.log")
    with open(log_path, "w") as f:
        f.write(result.stdout)
        if result.stderr:
            f.write("\n--- STDERR ---\n")
            f.write(result.stderr)

    return result.returncode, log_path


def parse_gcmc_output(log_path):
    """
    Comprehensive parser for RASPA3 GCMC output.
    Extracts loading, energy, acceptance ratios, and thermodynamic properties.
    """
    with open(log_path) as f:
        text = f.read()

    results = {}

    # --- Loading ---
    loading_units = {
        "mol/kg framework": r"Average loading absolute\s*\[mol/kg framework\]\s*[:\s]*([\d.eE+-]+)\s*\+/-\s*([\d.eE+-]+)",
        "molecules/unit cell": r"Average loading absolute\s*\[molecules/unit cell\]\s*[:\s]*([\d.eE+-]+)\s*\+/-\s*([\d.eE+-]+)",
        "cm3(STP)/cm3": r"Average loading absolute\s*\[cm\^3.*?/cm\^3.*?\]\s*[:\s]*([\d.eE+-]+)\s*\+/-\s*([\d.eE+-]+)",
        "mg/g": r"Average loading absolute\s*\[mg/g.*?\]\s*[:\s]*([\d.eE+-]+)\s*\+/-\s*([\d.eE+-]+)",
    }

    results["loading"] = {}
    for unit, pat in loading_units.items():
        m = re.search(pat, text)
        if m:
            results["loading"][unit] = {
                "mean": float(m.group(1)),
                "std": float(m.group(2))
            }

    # --- Excess loading ---
    excess_pat = r"Average loading excess\s*\[mol/kg framework\]\s*[:\s]*([\d.eE+-]+)\s*\+/-\s*([\d.eE+-]+)"
    m = re.search(excess_pat, text)
    if m:
        results["excess_loading_mol_kg"] = {
            "mean": float(m.group(1)), "std": float(m.group(2))
        }

    # --- Energy ---
    energy_pats = {
        "total_energy_kJ_mol": r"Total energy.*?[:\s]*([-\d.eE+]+)\s*\+/-\s*([\d.eE+-]+)",
        "host_guest_vdw": r"Host-Guest.*?Van der Waals.*?[:\s]*([-\d.eE+]+)\s*\+/-\s*([\d.eE+-]+)",
        "host_guest_coulomb": r"Host-Guest.*?Coulomb.*?[:\s]*([-\d.eE+]+)\s*\+/-\s*([\d.eE+-]+)",
        "guest_guest_vdw": r"Guest-Guest.*?Van der Waals.*?[:\s]*([-\d.eE+]+)\s*\+/-\s*([\d.eE+-]+)",
    }
    results["energy"] = {}
    for key, pat in energy_pats.items():
        m = re.search(pat, text)
        if m:
            results["energy"][key] = {
                "mean": float(m.group(1)), "std": float(m.group(2))
            }

    # --- Heat of adsorption ---
    for pat in [r"[Hh]eat of [Aa]dsorption\s*[:\s]*([-\d.eE+]+)",
                r"[Ee]nthalpy of [Aa]dsorption\s*[:\s]*([-\d.eE+]+)"]:
        m = re.search(pat, text)
        if m:
            results["Qst_kJ_mol"] = float(m.group(1))
            break

    # --- Henry coefficient ---
    m = re.search(r"Henry coefficient\s*[:\s]*([\d.eE+-]+)", text)
    if m:
        results["henry_coeff"] = float(m.group(1))

    # --- Acceptance ratios ---
    acceptance_pats = {
        "translation": r"Translation.*?accepted:\s*([\d.]+)%",
        "rotation": r"Rotation.*?accepted:\s*([\d.]+)%",
        "swap_insertion": r"Swap.*?insertion.*?accepted:\s*([\d.]+)%",
        "swap_deletion": r"Swap.*?deletion.*?accepted:\s*([\d.]+)%",
        "reinsertion": r"Reinsertion.*?accepted:\s*([\d.]+)%",
    }
    results["acceptance"] = {}
    for key, pat in acceptance_pats.items():
        m = re.search(pat, text, re.IGNORECASE)
        if m:
            results["acceptance"][key] = float(m.group(1))

    return results


def print_gcmc_summary(results, pressure_Pa=None, temperature=None):
    """Print a formatted summary of GCMC results."""
    print("\n" + "=" * 60)
    if pressure_Pa and temperature:
        print(f"GCMC Results: T={temperature} K, P={pressure_Pa:.2e} Pa ({pressure_Pa/1e5:.3f} bar)")
    print("=" * 60)

    if results["loading"]:
        print("\n--- Loading ---")
        for unit, val in results["loading"].items():
            print(f"  {unit}: {val['mean']:.4f} +/- {val['std']:.4f}")

    if results.get("Qst_kJ_mol"):
        print(f"\n--- Heat of adsorption: {results['Qst_kJ_mol']:.2f} kJ/mol ---")

    if results["energy"]:
        print("\n--- Energy components ---")
        for key, val in results["energy"].items():
            print(f"  {key}: {val['mean']:.2f} +/- {val['std']:.2f} kJ/mol")

    if results["acceptance"]:
        print("\n--- Move acceptance ratios ---")
        for move, pct in results["acceptance"].items():
            status = "OK" if 0.1 < pct < 90 else "CHECK"
            print(f"  {move}: {pct:.1f}% [{status}]")

    print("=" * 60)


# === Run a GCMC simulation and analyze ===
sim_dir = "/tmp/gcmc_analysis"
os.makedirs(sim_dir, exist_ok=True)

# Write simulation input
sim_input = {
    "SimulationType": "MonteCarlo",
    "NumberOfCycles": 25000,
    "NumberOfInitializationCycles": 10000,
    "PrintEvery": 2500,
    "Systems": [{
        "Type": "Framework",
        "Name": "IRMOF-1",
        "ExternalTemperature": 298.0,
        "ExternalPressure": 1e6,
        "ChargeMethod": "Ewald",
        "ForceField": "GenericMOFs",
        "CutOff": 12.0,
        "Components": [{
            "Name": "methane",
            "Type": "Adsorbate",
            "MoleculeDefinition": "TraPPE",
            "TranslationProbability": 0.5,
            "ReinsertionProbability": 0.5,
            "SwapProbability": 1.0,
            "CreateNumberOfMolecules": 0
        }]
    }]
}

with open(os.path.join(sim_dir, "simulation.json"), "w") as f:
    json.dump(sim_input, f, indent=2)

retcode, log_path = run_gcmc(sim_dir)
results = parse_gcmc_output(log_path)
print_gcmc_summary(results, pressure_Pa=1e6, temperature=298.0)
```

### Step 7: Loading vs pressure analysis with convergence checks

```python
#!/usr/bin/env python3
"""
Run GCMC at multiple pressures and analyze loading vs pressure
with convergence diagnostics.
"""
import json
import subprocess
import os
import re
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BASE_DIR = "/tmp/gcmc_isotherm"
FRAMEWORK = "IRMOF-1"
GUEST = "methane"
TEMPERATURE = 298.0
PRESSURES_PA = np.logspace(3, 7, 12)  # 0.01 to 100 bar

os.makedirs(BASE_DIR, exist_ok=True)

data = {"P_Pa": [], "P_bar": [], "loading": [], "error": [],
        "swap_accept": [], "Qst": []}

for i, P in enumerate(PRESSURES_PA):
    sim_dir = os.path.join(BASE_DIR, f"run_{i:02d}")
    os.makedirs(sim_dir, exist_ok=True)

    sim_input = {
        "SimulationType": "MonteCarlo",
        "NumberOfCycles": 25000,
        "NumberOfInitializationCycles": 10000,
        "PrintEvery": 2500,
        "Systems": [{
            "Type": "Framework",
            "Name": FRAMEWORK,
            "ExternalTemperature": TEMPERATURE,
            "ExternalPressure": float(P),
            "ChargeMethod": "None",
            "ForceField": "GenericMOFs",
            "CutOff": 12.0,
            "Components": [{
                "Name": GUEST,
                "Type": "Adsorbate",
                "MoleculeDefinition": "TraPPE",
                "TranslationProbability": 0.5,
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

    # Parse
    text = result.stdout
    m = re.search(r"Average loading absolute\s*\[mol/kg framework\]\s*[:\s]*([\d.eE+-]+)\s*\+/-\s*([\d.eE+-]+)", text)
    load = float(m.group(1)) if m else 0.0
    err = float(m.group(2)) if m else 0.0

    m_swap = re.search(r"Swap.*?insertion.*?accepted:\s*([\d.]+)%", text, re.IGNORECASE)
    swap_pct = float(m_swap.group(1)) if m_swap else -1

    m_qst = re.search(r"[Hh]eat of [Aa]dsorption\s*[:\s]*([-\d.eE+]+)", text)
    qst = float(m_qst.group(1)) if m_qst else None

    data["P_Pa"].append(P)
    data["P_bar"].append(P / 1e5)
    data["loading"].append(load)
    data["error"].append(err)
    data["swap_accept"].append(swap_pct)
    data["Qst"].append(qst)

    rel_err = (err / load * 100) if load > 0 else 0
    print(f"P={P/1e5:.4f} bar: q={load:.4f}+/-{err:.4f} mol/kg "
          f"(rel_err={rel_err:.1f}%), swap_accept={swap_pct:.1f}%")

# Save data
np.savetxt(os.path.join(BASE_DIR, "loading_vs_pressure.csv"),
           np.column_stack([data["P_bar"], data["loading"], data["error"]]),
           header="Pressure(bar)  Loading(mol/kg)  Error(mol/kg)",
           fmt="%.6e")

# === Convergence diagnostics ===
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# (a) Isotherm
ax = axes[0, 0]
ax.errorbar(data["P_bar"], data["loading"], yerr=data["error"],
            fmt="bo-", capsize=4, markersize=6, linewidth=1.5)
ax.set_xlabel("Pressure (bar)")
ax.set_ylabel("Loading (mol/kg)")
ax.set_title(f"{GUEST} in {FRAMEWORK} at {TEMPERATURE} K")
ax.set_xscale("log")
ax.grid(True, alpha=0.3)

# (b) Relative error
ax = axes[0, 1]
rel_errors = [e/l*100 if l > 0 else 0 for l, e in zip(data["loading"], data["error"])]
colors = ["green" if r < 10 else "orange" if r < 20 else "red" for r in rel_errors]
ax.bar(range(len(rel_errors)), rel_errors, color=colors)
ax.axhline(y=10, color="green", linestyle="--", label="10% threshold")
ax.set_xlabel("Pressure point index")
ax.set_ylabel("Relative error (%)")
ax.set_title("Convergence check")
ax.legend()

# (c) Swap acceptance
ax = axes[1, 0]
ax.plot(data["P_bar"], data["swap_accept"], "rs-", markersize=6)
ax.axhspan(0.1, 50, alpha=0.1, color="green", label="Healthy range")
ax.set_xlabel("Pressure (bar)")
ax.set_ylabel("Swap acceptance (%)")
ax.set_title("MC swap move acceptance")
ax.set_xscale("log")
ax.legend()
ax.grid(True, alpha=0.3)

# (d) Heat of adsorption
ax = axes[1, 1]
valid_qst = [(p, q) for p, q in zip(data["P_bar"], data["Qst"]) if q is not None]
if valid_qst:
    ax.plot([x[0] for x in valid_qst], [x[1] for x in valid_qst],
            "g^-", markersize=7, linewidth=1.5)
ax.set_xlabel("Pressure (bar)")
ax.set_ylabel("Q$_{st}$ (kJ/mol)")
ax.set_title("Heat of adsorption")
ax.set_xscale("log")
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(BASE_DIR, "gcmc_diagnostics.png"), dpi=150, bbox_inches="tight")
plt.close()
print(f"\nDiagnostics plot saved to {BASE_DIR}/gcmc_diagnostics.png")
```

### Step 8: Molecule definition files

```python
#!/usr/bin/env python3
"""
Create custom molecule definition files for RASPA3.

RASPA3 includes standard molecule definitions (TraPPE, ExampleDefinition).
Custom molecules can be defined in JSON format in the simulation directory.
"""
import json
import os

# ============ TraPPE CO2 (3-site linear model) ============
co2_definition = {
    "Name": "CO2",
    "NumberOfAtoms": 3,
    "NumberOfBonds": 2,
    "CriticalTemperature": 304.13,    # K
    "CriticalPressure": 7.375e6,      # Pa
    "AcentricFactor": 0.2236,
    "Atoms": [
        {"Type": "C_co2",  "Position": [0.0, 0.0, 0.0],     "Charge": 0.6512},
        {"Type": "O_co2",  "Position": [0.0, 0.0, 1.16],    "Charge": -0.3256},
        {"Type": "O_co2",  "Position": [0.0, 0.0, -1.16],   "Charge": -0.3256}
    ],
    "Bonds": [
        {"Atoms": [0, 1], "Type": "Rigid", "Length": 1.16},
        {"Atoms": [0, 2], "Type": "Rigid", "Length": 1.16}
    ]
}

# ============ TraPPE N2 (2-site + COM charge) ============
n2_definition = {
    "Name": "N2",
    "NumberOfAtoms": 3,
    "NumberOfBonds": 2,
    "CriticalTemperature": 126.2,
    "CriticalPressure": 3.4e6,
    "AcentricFactor": 0.0377,
    "Atoms": [
        {"Type": "N_n2",   "Position": [0.0, 0.0, 0.549],   "Charge": -0.482},
        {"Type": "N_n2",   "Position": [0.0, 0.0, -0.549],   "Charge": -0.482},
        {"Type": "N_com",  "Position": [0.0, 0.0, 0.0],      "Charge": 0.964}
    ],
    "Bonds": [
        {"Atoms": [0, 2], "Type": "Rigid", "Length": 0.549},
        {"Atoms": [1, 2], "Type": "Rigid", "Length": 0.549}
    ]
}

# ============ United-atom methane (single site) ============
ch4_definition = {
    "Name": "methane",
    "NumberOfAtoms": 1,
    "NumberOfBonds": 0,
    "CriticalTemperature": 190.56,
    "CriticalPressure": 4.599e6,
    "AcentricFactor": 0.0115,
    "Atoms": [
        {"Type": "CH4_sp3", "Position": [0.0, 0.0, 0.0], "Charge": 0.0}
    ],
    "Bonds": []
}

# ============ H2 (2-site model) ============
h2_definition = {
    "Name": "H2",
    "NumberOfAtoms": 2,
    "NumberOfBonds": 1,
    "CriticalTemperature": 33.19,
    "CriticalPressure": 1.315e6,
    "AcentricFactor": -0.216,
    "Atoms": [
        {"Type": "H_h2", "Position": [0.0, 0.0, 0.37],  "Charge": 0.0},
        {"Type": "H_h2", "Position": [0.0, 0.0, -0.37], "Charge": 0.0}
    ],
    "Bonds": [
        {"Atoms": [0, 1], "Type": "Rigid", "Length": 0.74}
    ]
}

# Write molecule definitions
mol_dir = "/tmp/gcmc_examples/molecules"
os.makedirs(mol_dir, exist_ok=True)

for mol in [co2_definition, n2_definition, ch4_definition, h2_definition]:
    filepath = os.path.join(mol_dir, f"{mol['Name']}.json")
    with open(filepath, "w") as f:
        json.dump(mol, f, indent=2)
    print(f"Written: {filepath}")

print(f"""
=== Using custom molecule definitions ===

Place molecule JSON files in the simulation directory or a molecules/ subdirectory.
In simulation.json, reference them:

  "MoleculeDefinition": "Local"     -- looks in simulation directory
  "MoleculeDefinition": "TraPPE"    -- uses built-in TraPPE library

Custom definitions override built-in ones when placed locally.
""")
```

## Key Parameters

| Parameter | Description | Typical Value |
|-----------|-------------|---------------|
| `SimulationType` | Ensemble type | `"MonteCarlo"` for GCMC |
| `NumberOfCycles` | Production MC cycles | 20000-100000 |
| `NumberOfInitializationCycles` | Equilibration cycles | 10000-50000 |
| `PrintEvery` | Output frequency | Every 1000-5000 cycles |
| `ExternalTemperature` | Temperature (K) | 77-400 K depending on application |
| `ExternalPressure` | Pressure (Pa) | 1e3-1e7; 1e5 = 1 bar |
| `CutOff` | LJ cutoff (Angstrom) | 12.0 standard; must be < L_min/2 |
| `ChargeMethod` | Electrostatics | `"Ewald"`, `"Wolf"`, `"None"` |
| `ForceField` | LJ parameters source | `"GenericMOFs"`, `"GenericZeolites"`, `"UFF"` |
| `MoleculeDefinition` | Guest model library | `"TraPPE"`, `"ExampleDefinition"`, `"Local"` |
| `SwapProbability` | Insert/delete moves | 1.0 (essential for GCMC) |
| `TranslationProbability` | Translation moves | 0.5 |
| `RotationProbability` | Rotation moves | 0.5 (multi-site molecules) |
| `ReinsertionProbability` | Random reinsertion | 0.5 (improves sampling) |
| `WidomProbability` | Test insertion (Henry coeff) | 1.0 (Widom only; no swap) |
| `NumberOfUnitCells` | Supercell size | `[2,2,2]` if unit cell is small |
| `FrameworkModel` | Rigid or flexible | `"Rigid"` (default) or `"Flexible"` |
| `MoleFraction` | Mixture composition | Sum = 1.0 across all components |
| `UseTabularGrid` | Pre-tabulate energies | `true` for large systems (speeds up) |
| `TailCorrections` | LJ tail corrections | `true` (default in most force fields) |

## Interpreting Results

### Loading
- Primary output of GCMC: average number of adsorbed molecules
- Reported in multiple units: mol/kg, molecules/uc, cm^3(STP)/cm^3, mg/g
- Absolute loading includes gas-phase contribution; excess loading subtracts bulk gas density
- At low pressures absolute and excess loading are essentially identical

### Energy decomposition
- Host-guest VdW: framework-adsorbate Lennard-Jones interaction (should be negative for adsorption)
- Host-guest Coulomb: framework-adsorbate electrostatic interaction
- Guest-guest VdW: adsorbate-adsorbate LJ (becomes significant at high loading)
- Total energy: sum of all contributions

### Acceptance ratios
- Translation: 30-60% is normal; adjust maximum displacement if outside range
- Rotation: 30-60% is normal for rigid molecules
- Swap (insertion): 0.1-50% is typical; drops at high loading due to excluded volume
- Swap (deletion): usually higher than insertion
- If swap acceptance < 0.01%, sampling is poor; consider CBMC moves or larger cutoff

### Convergence assessment
- Relative error < 10%: well converged
- Relative error 10-20%: marginal; consider doubling `NumberOfCycles`
- Relative error > 20%: poorly converged; significantly increase cycles or check simulation setup
- Run block averaging to check for drift

## Common Issues

| Problem | Symptom | Solution |
|---------|---------|----------|
| CutOff > half box dimension | RASPA3 error at startup | Add `"NumberOfUnitCells": [2,2,2]` or reduce `CutOff` |
| Missing framework CIF | File not found error | Place CIF in simulation directory; name must match `"Name"` field |
| Guest molecule not found | Molecule definition error | Check spelling; verify molecule exists in chosen `MoleculeDefinition` library |
| Zero loading everywhere | No adsorption detected | Check CIF has accessible pores; verify force field assigns non-zero LJ to framework atoms |
| Very low swap acceptance | < 0.01% insertion | System is fully loaded or molecule too large; add `"CBMCProbability"` for chain molecules |
| Ewald divergence | NaN energies or crashes | Reduce `CutOff`; check cell is not too small; verify charges sum to zero |
| Charges not read from CIF | All charges zero despite `"Ewald"` | CIF must have `_atom_site_charge` column; check with `grep _atom_site_charge file.cif` |
| Wrong force field atom types | LJ parameters not assigned | Atom type names in CIF must match force field definitions; check RASPA3 log for warnings |
| Flexible framework unstable | Framework distorts unrealistically | Ensure proper bond/angle/dihedral parameters; reduce temperature step; increase equilibration |
| Simulation extremely slow | Hours per 1000 cycles | Use `"ChargeMethod": "None"` for non-polar guests; reduce system size; pre-tabulate energies |
| Negative heat of adsorption printed as positive | Sign convention confusion | RASPA3 may report Q_st as positive (exothermic convention); check output header for sign convention |
| Mixture mole fractions do not sum to 1 | RASPA3 warning or error | Ensure `"MoleFraction"` values across all components sum exactly to 1.0 |
