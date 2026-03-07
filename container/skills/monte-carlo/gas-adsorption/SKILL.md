# Gas Adsorption Monte Carlo with RASPA3

## When to Use

- Computing adsorption isotherms of gases (CH4, CO2, N2, H2, H2O, etc.) in porous frameworks (MOFs, zeolites, COFs)
- Calculating Henry coefficients and heats of adsorption at infinite dilution
- Evaluating mixture selectivity (e.g., CO2/N2 separation)
- Screening porous materials for gas storage or separation applications

## Method Selection

```
Single gas loading at one (T, P)?
  --> Single-point GCMC

Full adsorption isotherm (loading vs pressure)?
  --> Loop GCMC over multiple pressures

Henry coefficient / heat of adsorption at zero loading?
  --> Widom insertion method (more efficient than low-pressure GCMC)

Binary or ternary mixture?
  --> Multi-component GCMC with partial pressures
```

## Prerequisites

- RASPA3 binary (`raspa3`) -- pre-installed in the container
- A framework CIF file (from CoRE-MOF database, CSD, or user-provided)
- Knowledge of guest molecule(s), temperature, and pressure range
- Official examples at `/usr/share/raspa3/examples/` -- always start from these

## Detailed Steps

### Step 0: Explore official examples to find a suitable starting point

```bash
ls /usr/share/raspa3/examples/
# Typical directories: basic/, adsorption/, breakthrough/, etc.
# Copy a relevant example and modify it:
cp -r /usr/share/raspa3/examples/basic/1_mc_methane_in_box /tmp/my_sim
```

### Step 1: Prepare the framework CIF file

The framework CIF must be placed where RASPA3 can find it. RASPA3 searches for framework files in the current working directory or paths specified in the simulation input.

```python
# Option A: Download from Materials Project or CoRE-MOF via pymatgen
from pymatgen.core import Structure

# Example: create a simple test structure or load a CIF
# For real MOFs, use a CIF from CoRE-MOF or your own source
structure = Structure.from_file("my_framework.cif")
# Ensure the CIF has fractional coordinates and proper symmetry
structure.to(filename="/tmp/raspa_sim/my_framework.cif")
```

```bash
# Option B: Use a CIF file you already have
cp my_framework.cif /tmp/raspa_sim/
```

### Step 2: Single-point GCMC -- Methane in a MOF

Create the simulation input file `simulation.json`. RASPA3 uses JSON format exclusively.

**Complete `simulation.json` for methane adsorption in IRMOF-1 at 300 K, 1 bar:**

```json
{
  "SimulationType": "MonteCarlo",
  "NumberOfCycles": 10000,
  "NumberOfInitializationCycles": 5000,
  "PrintEvery": 1000,
  "Systems": [
    {
      "Type": "Framework",
      "Name": "IRMOF-1",
      "ExternalTemperature": 300.0,
      "ExternalPressure": 1e5,
      "ChargeMethod": "Ewald",
      "ForceField": "GenericMOFs",
      "CutOff": 12.0,
      "Components": [
        {
          "Name": "methane",
          "Type": "Adsorbate",
          "MoleculeDefinition": "TraPPE",
          "TranslationProbability": 0.5,
          "ReinsertionProbability": 0.5,
          "SwapProbability": 1.0,
          "CreateNumberOfMolecules": 0
        }
      ]
    }
  ]
}
```

Run the simulation:

```bash
cd /tmp/raspa_sim && raspa3
```

RASPA3 reads `simulation.json` from the current directory by default.

### Step 3: Adsorption isotherm -- multiple pressures

To compute a full isotherm, run GCMC at multiple pressures. This can be scripted in Python:

```python
import json
import subprocess
import os
import numpy as np

# Pressure points for the isotherm (in Pascals)
pressures = [1e3, 5e3, 1e4, 5e4, 1e5, 5e5, 1e6, 5e6, 1e7]
temperature = 298.0
framework_name = "IRMOF-1"

results = []

for P in pressures:
    # Create a directory for each pressure point
    sim_dir = f"/tmp/isotherm/P_{P:.0e}"
    os.makedirs(sim_dir, exist_ok=True)

    # Copy framework CIF if needed
    # os.system(f"cp /path/to/IRMOF-1.cif {sim_dir}/")

    sim_input = {
        "SimulationType": "MonteCarlo",
        "NumberOfCycles": 20000,
        "NumberOfInitializationCycles": 10000,
        "PrintEvery": 2000,
        "Systems": [
            {
                "Type": "Framework",
                "Name": framework_name,
                "ExternalTemperature": temperature,
                "ExternalPressure": P,
                "ChargeMethod": "Ewald",
                "ForceField": "GenericMOFs",
                "CutOff": 12.0,
                "Components": [
                    {
                        "Name": "methane",
                        "Type": "Adsorbate",
                        "MoleculeDefinition": "TraPPE",
                        "TranslationProbability": 0.5,
                        "ReinsertionProbability": 0.5,
                        "SwapProbability": 1.0,
                        "CreateNumberOfMolecules": 0
                    }
                ]
            }
        ]
    }

    with open(os.path.join(sim_dir, "simulation.json"), "w") as f:
        json.dump(sim_input, f, indent=2)

    # Run RASPA3
    result = subprocess.run(
        ["raspa3"],
        cwd=sim_dir,
        capture_output=True,
        text=True,
        timeout=3600
    )

    # Store stdout for later parsing
    with open(os.path.join(sim_dir, "output.log"), "w") as f:
        f.write(result.stdout)
        f.write(result.stderr)

    print(f"Completed P = {P:.2e} Pa")
```

### Step 4: Henry coefficient calculation (Widom insertion)

The Widom insertion method calculates the Henry coefficient at infinite dilution, which is more efficient than running GCMC at very low pressures.

```json
{
  "SimulationType": "MonteCarlo",
  "NumberOfCycles": 50000,
  "NumberOfInitializationCycles": 10000,
  "PrintEvery": 5000,
  "Systems": [
    {
      "Type": "Framework",
      "Name": "IRMOF-1",
      "ExternalTemperature": 298.0,
      "ChargeMethod": "Ewald",
      "ForceField": "GenericMOFs",
      "CutOff": 12.0,
      "Components": [
        {
          "Name": "methane",
          "Type": "Adsorbate",
          "MoleculeDefinition": "TraPPE",
          "WidomProbability": 1.0,
          "CreateNumberOfMolecules": 0
        }
      ]
    }
  ]
}
```

Note: For Widom insertion, set `WidomProbability` to 1.0 and remove `SwapProbability`. No molecules are inserted/deleted; instead, test insertions compute the excess chemical potential.

### Step 5: CO2/N2 mixture selectivity

Multi-component GCMC for evaluating separation performance. Specify partial pressures for each component (e.g., flue gas: ~15% CO2, ~85% N2 at 1 bar total).

```json
{
  "SimulationType": "MonteCarlo",
  "NumberOfCycles": 30000,
  "NumberOfInitializationCycles": 15000,
  "PrintEvery": 3000,
  "Systems": [
    {
      "Type": "Framework",
      "Name": "IRMOF-1",
      "ExternalTemperature": 298.0,
      "ExternalPressure": 1e5,
      "ChargeMethod": "Ewald",
      "ForceField": "GenericMOFs",
      "CutOff": 12.0,
      "Components": [
        {
          "Name": "CO2",
          "Type": "Adsorbate",
          "MoleculeDefinition": "TraPPE",
          "MoleFraction": 0.15,
          "TranslationProbability": 0.5,
          "RotationProbability": 0.5,
          "ReinsertionProbability": 0.5,
          "SwapProbability": 1.0,
          "CreateNumberOfMolecules": 0
        },
        {
          "Name": "N2",
          "Type": "Adsorbate",
          "MoleculeDefinition": "TraPPE",
          "MoleFraction": 0.85,
          "TranslationProbability": 0.5,
          "RotationProbability": 0.5,
          "ReinsertionProbability": 0.5,
          "SwapProbability": 1.0,
          "CreateNumberOfMolecules": 0
        }
      ]
    }
  ]
}
```

The selectivity is computed as:

```
S_CO2/N2 = (x_CO2 / x_N2) / (y_CO2 / y_N2)
```

where `x` is the adsorbed phase mole fraction and `y` is the bulk gas mole fraction.

### Step 6: Reading RASPA3 output

RASPA3 writes results to stdout and to output files in the simulation directory. Key quantities to extract:

```python
import re
import os

def parse_raspa3_output(output_dir):
    """Parse RASPA3 output to extract average loading and energy."""
    # RASPA3 prints results to stdout -- read from saved log
    log_file = os.path.join(output_dir, "output.log")

    if not os.path.exists(log_file):
        # Try reading from any .data files RASPA3 may produce
        for f in os.listdir(output_dir):
            if f.endswith(".data") or f.endswith(".txt"):
                log_file = os.path.join(output_dir, f)
                break

    with open(log_file, "r") as f:
        text = f.read()

    results = {}

    # Look for average loading lines
    # Pattern varies by RASPA3 version; common patterns:
    #   "Average loading absolute [molecules/unit cell]"
    #   "Average loading excess [mol/kg framework]"
    loading_patterns = [
        r"Average loading absolute\s*\[molecules/unit cell\]\s*[:\s]*([\d.eE+-]+)\s*\+/-\s*([\d.eE+-]+)",
        r"Average loading absolute\s*\[mol/kg framework\]\s*[:\s]*([\d.eE+-]+)\s*\+/-\s*([\d.eE+-]+)",
        r"Average loading absolute\s*\[cm\^3 \(STP\)/cm\^3 framework\]\s*[:\s]*([\d.eE+-]+)\s*\+/-\s*([\d.eE+-]+)",
    ]

    for pattern in loading_patterns:
        match = re.search(pattern, text)
        if match:
            key = pattern.split("[")[1].split("]")[0]
            results[f"loading ({key})"] = {
                "mean": float(match.group(1)),
                "std": float(match.group(2))
            }

    # Look for Henry coefficient
    henry_match = re.search(
        r"Henry coefficient\s*[:\s]*([\d.eE+-]+)",
        text
    )
    if henry_match:
        results["henry_coefficient"] = float(henry_match.group(1))

    # Look for heat of adsorption
    heat_match = re.search(
        r"[Hh]eat of [Aa]dsorption\s*[:\s]*([\d.eE+-]+)",
        text
    )
    if heat_match:
        results["heat_of_adsorption_kJ_per_mol"] = float(heat_match.group(1))

    # Look for enthalpy of adsorption
    enthalpy_match = re.search(
        r"[Ee]nthalpy of [Aa]dsorption\s*[:\s]*([\d.eE+-]+)",
        text
    )
    if enthalpy_match:
        results["enthalpy_of_adsorption_kJ_per_mol"] = float(enthalpy_match.group(1))

    return results
```

Alternatively, check the output files that RASPA3 produces in its output directory:

```bash
# RASPA3 typically writes output to:
ls /tmp/raspa_sim/Output/
# or
ls /tmp/raspa_sim/output/
# Look for files like:
#   System_0/  -- contains per-system output
#   System_0/output_IRMOF-1_298.000000_1e+05/  -- T and P labeled
```

### Step 7: Post-processing -- plot adsorption isotherm

```python
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import os
import json

def collect_isotherm_data(base_dir, pressures):
    """Collect loading vs pressure from multiple RASPA3 runs."""
    loadings_mean = []
    loadings_std = []

    for P in pressures:
        sim_dir = os.path.join(base_dir, f"P_{P:.0e}")
        data = parse_raspa3_output(sim_dir)

        # Get loading in mol/kg (preferred unit) or molecules/uc
        loading_key = None
        for key in data:
            if "loading" in key.lower():
                loading_key = key
                break

        if loading_key:
            loadings_mean.append(data[loading_key]["mean"])
            loadings_std.append(data[loading_key]["std"])
        else:
            loadings_mean.append(0.0)
            loadings_std.append(0.0)

    return np.array(loadings_mean), np.array(loadings_std)


def plot_isotherm(pressures, loadings, errors, output_file="isotherm.png",
                  temperature=298.0, guest="CH4", framework="IRMOF-1"):
    """Plot adsorption isotherm with error bars."""
    pressures_bar = np.array(pressures) / 1e5  # Pa to bar

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.errorbar(pressures_bar, loadings, yerr=errors,
                fmt="o-", capsize=4, linewidth=2, markersize=8,
                color="#2196F3", ecolor="#F44336")

    ax.set_xlabel("Pressure (bar)", fontsize=14)
    ax.set_ylabel("Loading (mol/kg)", fontsize=14)
    ax.set_title(f"{guest} adsorption in {framework} at {temperature} K", fontsize=16)
    ax.set_xscale("log")
    ax.grid(True, alpha=0.3)
    ax.tick_params(labelsize=12)

    plt.tight_layout()
    plt.savefig(output_file, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Isotherm plot saved to {output_file}")


# Usage:
pressures = [1e3, 5e3, 1e4, 5e4, 1e5, 5e5, 1e6, 5e6, 1e7]
loadings, errors = collect_isotherm_data("/tmp/isotherm", pressures)
plot_isotherm(pressures, loadings, errors,
              output_file="/tmp/isotherm/methane_isotherm.png")
```

### Full end-to-end script: methane isotherm in a MOF

```python
#!/usr/bin/env python3
"""
Complete RASPA3 GCMC workflow: methane adsorption isotherm in IRMOF-1.
"""
import json
import subprocess
import os
import re
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def create_simulation_input(framework, guest, temperature, pressure,
                            n_cycles=20000, n_init=10000):
    """Create RASPA3 simulation.json for single-component GCMC."""
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
                "ForceField": "GenericMOFs",
                "CutOff": 12.0,
                "Components": [
                    {
                        "Name": guest,
                        "Type": "Adsorbate",
                        "MoleculeDefinition": "TraPPE",
                        "TranslationProbability": 0.5,
                        "ReinsertionProbability": 0.5,
                        "SwapProbability": 1.0,
                        "CreateNumberOfMolecules": 0
                    }
                ]
            }
        ]
    }


def run_raspa3(sim_dir, sim_input):
    """Write input and run RASPA3 in the given directory."""
    os.makedirs(sim_dir, exist_ok=True)

    with open(os.path.join(sim_dir, "simulation.json"), "w") as f:
        json.dump(sim_input, f, indent=2)

    result = subprocess.run(
        ["raspa3"],
        cwd=sim_dir,
        capture_output=True,
        text=True,
        timeout=7200
    )

    log_path = os.path.join(sim_dir, "output.log")
    with open(log_path, "w") as f:
        f.write(result.stdout)
        if result.stderr:
            f.write("\n--- STDERR ---\n")
            f.write(result.stderr)

    return result.returncode, log_path


def parse_loading(log_path):
    """Extract average absolute loading from RASPA3 output."""
    with open(log_path, "r") as f:
        text = f.read()

    # Try multiple patterns to be robust across RASPA3 output variants
    patterns = [
        r"Average loading absolute\s*\[mol/kg framework\]\s*[:\s]*([\d.eE+-]+)\s*\+/-\s*([\d.eE+-]+)",
        r"Average loading absolute\s*\[molecules/unit cell\]\s*[:\s]*([\d.eE+-]+)\s*\+/-\s*([\d.eE+-]+)",
        r"Average\s+loading\s+absolute.*?:\s*([\d.eE+-]+)\s*\+/-\s*([\d.eE+-]+)",
    ]
    for pat in patterns:
        match = re.search(pat, text)
        if match:
            return float(match.group(1)), float(match.group(2))

    # Fallback: look for any "loading" number
    match = re.search(r"loading.*?([\d.]+)\s*\+/-\s*([\d.]+)", text, re.IGNORECASE)
    if match:
        return float(match.group(1)), float(match.group(2))

    print(f"WARNING: Could not parse loading from {log_path}")
    return 0.0, 0.0


def main():
    base_dir = "/tmp/methane_isotherm"
    framework = "IRMOF-1"
    guest = "methane"
    temperature = 298.0

    # Pressure range: 0.01 bar to 100 bar
    pressures_Pa = [1e3, 5e3, 1e4, 5e4, 1e5, 5e5, 1e6, 5e6, 1e7]

    loadings = []
    errors = []

    for P in pressures_Pa:
        print(f"\n{'='*60}")
        print(f"Running GCMC at T={temperature} K, P={P:.2e} Pa ({P/1e5:.3f} bar)")
        print(f"{'='*60}")

        sim_dir = os.path.join(base_dir, f"P_{P:.0e}")
        sim_input = create_simulation_input(
            framework, guest, temperature, P,
            n_cycles=20000, n_init=10000
        )

        retcode, log_path = run_raspa3(sim_dir, sim_input)

        if retcode != 0:
            print(f"  RASPA3 returned exit code {retcode}")

        loading, error = parse_loading(log_path)
        loadings.append(loading)
        errors.append(error)
        print(f"  Loading = {loading:.4f} +/- {error:.4f}")

    # Convert to arrays
    pressures_bar = np.array(pressures_Pa) / 1e5
    loadings = np.array(loadings)
    errors = np.array(errors)

    # Save data
    data_file = os.path.join(base_dir, "isotherm_data.csv")
    np.savetxt(
        data_file,
        np.column_stack([pressures_bar, loadings, errors]),
        header="Pressure(bar)  Loading(mol/kg)  Error(mol/kg)",
        fmt="%.6e"
    )
    print(f"\nIsotherm data saved to {data_file}")

    # Plot
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.errorbar(pressures_bar, loadings, yerr=errors,
                fmt="s-", capsize=4, linewidth=2, markersize=8,
                color="#1565C0", ecolor="#E53935", markerfacecolor="#42A5F5")
    ax.set_xlabel("Pressure (bar)", fontsize=14)
    ax.set_ylabel("Uptake (mol/kg)", fontsize=14)
    ax.set_title(f"CH$_4$ adsorption in {framework} at {temperature} K", fontsize=16)
    ax.set_xscale("log")
    ax.grid(True, alpha=0.3)
    ax.tick_params(labelsize=12)
    plt.tight_layout()

    plot_file = os.path.join(base_dir, "methane_isotherm.png")
    plt.savefig(plot_file, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Isotherm plot saved to {plot_file}")


if __name__ == "__main__":
    main()
```

## Key Parameters

| Parameter | Description | Typical Value |
|---|---|---|
| `NumberOfCycles` | Production MC cycles | 10000--100000 (more for accurate statistics) |
| `NumberOfInitializationCycles` | Equilibration cycles (not sampled) | 5000--50000 (at least half of production) |
| `ExternalTemperature` | Temperature in Kelvin | 77 (N2 BET), 298 (ambient), 313 (flue gas) |
| `ExternalPressure` | Pressure in Pascals | 1e5 = 1 bar; range 1e2--1e7 for isotherms |
| `CutOff` | LJ cutoff in Angstrom | 12.0 (standard); must be < half the smallest box dimension |
| `ChargeMethod` | Electrostatic method | `"Ewald"` for charged systems, omit for non-polar guests |
| `ForceField` | Force field name | `"GenericMOFs"`, `"GenericZeolites"`, `"UFF"` |
| `MoleculeDefinition` | Guest molecule model | `"TraPPE"` (recommended), `"ExampleDefinition"` |
| `SwapProbability` | GCMC insert/delete probability | 1.0 (essential for GCMC) |
| `TranslationProbability` | MC translation move probability | 0.5 |
| `RotationProbability` | MC rotation move probability | 0.5 (for multi-site molecules like CO2) |
| `ReinsertionProbability` | Random reinsertion probability | 0.5 (helps sampling) |
| `WidomProbability` | Widom test insertion probability | 1.0 (for Henry coefficient only) |
| `MoleFraction` | Mole fraction for mixtures | Sum of all components must equal 1.0 |

## Interpreting Results

### Loading (uptake)
- **molecules/unit cell**: number of guest molecules per framework unit cell
- **mol/kg framework**: moles of guest per kg of framework (gravimetric)
- **cm^3 (STP)/cm^3 framework**: volumetric uptake at standard conditions
- **mg/g**: milligrams guest per gram framework

### Henry coefficient
- Units: mol/(kg Pa) -- higher value means stronger affinity
- Slope of the isotherm at zero pressure
- Useful for ranking materials at low coverage

### Heat of adsorption
- Typically reported as isosteric heat Q_st in kJ/mol
- More negative = stronger binding
- CH4 in typical MOFs: -15 to -25 kJ/mol
- CO2 in typical MOFs: -20 to -50 kJ/mol

### Selectivity (mixtures)
- Adsorption selectivity: S = (x_A/x_B) / (y_A/y_B)
- x = adsorbed phase composition, y = gas phase composition
- S > 1 means component A is preferred
- For CO2/N2 in good sorbents: S > 20

### Convergence checks
- Run block averages: loading should stabilize during production
- Increase `NumberOfCycles` if the error bar is too large (>10% of mean)
- Ensure acceptance ratios for swap moves are reasonable (typically 0.1-50%)

## Common Issues

### Framework charges

**Problem**: Polar molecules (CO2, H2O, NH3) require framework partial charges for accurate results. Non-polar molecules (CH4, noble gases) are less sensitive.

**Solutions**:
1. Use a CIF file that already contains `_atom_site_charge` columns (e.g., from CoRE-MOF with DDEC or EQeq charges)
2. Compute charges externally (EQeq, REPEAT, DDEC6) and add to the CIF
3. For quick screening, use charge-equilibration methods built into some workflows

```python
# Check if a CIF has charges using pymatgen
from pymatgen.core import Structure

s = Structure.from_file("framework.cif")
# If the CIF has _atom_site_charge, pymatgen may store it in site properties
if hasattr(s[0], "charge") or "charge" in s.site_properties:
    print("Framework has charges assigned")
else:
    print("WARNING: No charges found. Results for polar molecules will be inaccurate.")
```

### Force field selection

| Force Field | Best For | Notes |
|---|---|---|
| `GenericMOFs` | MOF structures | Dreiding/UFF-based, reasonable for screening |
| `GenericZeolites` | Zeolite structures | Hill-Sauer or similar; tuned for SiO2 frameworks |
| `UFF` | General use | Universal, but less accurate than specialized FFs |
| `TraPPE` | Guest molecules | United-atom model; good for hydrocarbons, CO2, N2 |

### CutOff too large for box

**Problem**: `CutOff` must be less than half the smallest box dimension. For small unit cells, either reduce the cutoff or replicate the unit cell.

```json
{
  "NumberOfUnitCells": [2, 2, 2]
}
```

Add `"NumberOfUnitCells"` inside the system definition to replicate the framework. RASPA3 usually handles this automatically, but specify it explicitly if the box is too small.

### Simulation not converging

- Increase `NumberOfInitializationCycles` to ensure equilibration
- For high-pressure or high-loading systems, increase total cycles
- Check that `SwapProbability` is set (without swaps, GCMC cannot exchange molecules with the reservoir)
- For large/branched molecules, add `ReinsertionProbability` and `CBMCProbability` (configurational-bias MC)

### Common molecule names in RASPA3

Molecule names must match definitions available in the chosen `MoleculeDefinition` library:
- `methane`, `ethane`, `propane`, `butane`, `isobutane`
- `CO2`, `N2`, `O2`, `H2`, `Ar`, `He`, `Kr`, `Xe`
- `H2O` (requires special handling -- flexible models)

### CIF file issues

- Remove disorder/partial occupancy before use (RASPA3 may not handle them)
- Ensure the CIF uses P1 symmetry or that RASPA3 can expand the symmetry
- Solvent molecules in as-synthesized CIFs must be removed (use `pymatgen` or `remove_solvent` tools)

```python
# Clean a CIF: remove solvent, convert to P1
from pymatgen.core import Structure

s = Structure.from_file("raw_mof.cif")
# Remove any sites that look like solvent (optional manual step)
# Convert to P1 (primitive cell with no symmetry operations)
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
sga = SpacegroupAnalyzer(s)
primitive = sga.get_primitive_standard_structure()
primitive.to(filename="clean_framework.cif")
```
