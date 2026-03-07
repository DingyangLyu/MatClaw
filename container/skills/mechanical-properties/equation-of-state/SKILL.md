# Equation of State (EOS) Calculation

## When to Use

- You need the equilibrium volume, bulk modulus, or pressure derivative of the bulk modulus.
- You want the energy-volume (E-V) curve of a material.
- You are studying pressure-induced phase transitions (compare E-V curves of competing phases).
- You need to validate a relaxed structure by checking that it sits at the E-V minimum.
- You want to compare MACE predictions against DFT for a specific material.

## Method Selection (MACE vs QE)

| Criterion | MACE (ASE) | QE (DFT) |
|---|---|---|
| Speed | Seconds (7--11 points in ~10s) | Hours (7--11 SCF calculations) |
| Accuracy | Good for systems within MACE training domain | Systematically improvable, publication quality |
| Use when | Screening, rapid estimation, comparing many structures | Publication results, unusual chemistry, validating MACE |

## Prerequisites

- A crystal structure (CIF, POSCAR, or pymatgen Structure). Ideally pre-relaxed, but the workflow includes an initial relaxation step.
- For QE: pseudopotential files (SSSP recommended). See `electronic-structure/scf-relax/SKILL.md`.
- Python packages: `pymatgen`, `ase`, `mace-torch`, `numpy`, `scipy`, `matplotlib` (pre-installed).

## Detailed Steps

### Method A: ASE + MACE

This method relaxes the structure at multiple volumes using the MACE calculator and fits E-V data to several EOS models. The workflow follows atomate2's `CommonEosMaker` pattern: (1) relax the equilibrium structure, (2) apply isotropic volume strains to generate deformed cells, (3) relax ions at each fixed volume, (4) collect energies and volumes, (5) fit EOS models.

```python
#!/usr/bin/env python3
"""
Equation of State calculation using ASE + MACE.

Workflow (following atomate2 CommonEosMaker pattern):
  1. Relax the structure (full cell + ions).
  2. Apply linear strain from -5% to +5% (isotropic volume deformations).
  3. At each deformed volume, relax ionic positions (fixed cell).
  4. Collect E(V) data.
  5. Fit to Birch-Murnaghan, Murnaghan, Vinet EOS.
  6. Extract V0, E0, B0, B0'.
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
from ase.eos import EquationOfState as ASE_EOS
from ase.units import kJ

from pymatgen.core.structure import Structure
from pymatgen.analysis.eos import EOS as PMG_EOS
from pymatgen.io.ase import AseAtomsAdaptor
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer

# ─── CONFIGURATION ───────────────────────────────────────────────────────────
INPUT_FILE = "structure.cif"           # Input structure (CIF, POSCAR, etc.)
MACE_MODEL = "medium"                  # MACE model: "small", "medium", "large"
LINEAR_STRAIN = (-0.05, 0.05)          # Linear strain range (maps to ~-14% to +16% volume)
N_POINTS = 9                           # Number of volume points (odd number centers on V0)
FMAX_BULK = 1e-4                       # Force convergence for initial relaxation (eV/A)
FMAX_EOS = 1e-3                        # Force convergence for strained structures (eV/A)
OUTPUT_DIR = "eos_results"
# ─────────────────────────────────────────────────────────────────────────────

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ─── STEP 1: Load structure and set up MACE calculator ───────────────────────
from mace.calculators import mace_mp
calc = mace_mp(model=MACE_MODEL, default_dtype="float64")

adaptor = AseAtomsAdaptor()

structure = Structure.from_file(INPUT_FILE)
formula = structure.composition.reduced_formula
print(f"Loaded: {formula}")

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

# ─── STEP 3: Generate volume-strained structures ────────────────────────────
# Following atomate2: apply isotropic linear strain.
# Linear strain eps means each lattice vector is scaled by (1 + eps),
# so volume scales as (1 + eps)^3.

strain_values = np.linspace(LINEAR_STRAIN[0], LINEAR_STRAIN[1], N_POINTS)

# If an initial relaxation was done, perturb the zero-strain point slightly
# (following atomate2 logic) since the equilibrium is already computed.
zero_mask = np.abs(strain_values) < 1e-15
if np.any(zero_mask):
    _, strain_delta = np.linspace(
        LINEAR_STRAIN[0], LINEAR_STRAIN[1], N_POINTS, retstep=True
    )
    nzs = np.sum(zero_mask)
    shift = strain_delta / (nzs + 1.0) * np.linspace(-1.0, 1.0, int(nzs))
    strain_values[zero_mask] += shift

print(f"\nLinear strain values: {strain_values}")
print(f"Corresponding volume ratios: {(1 + strain_values)**3}")

# ─── STEP 4: Relax at each strained volume ──────────────────────────────────
volumes = []
energies = []
pressures = []

# Include the equilibrium point
volumes.append(V0_relaxed)
energies.append(E0_relaxed)
eq_stress = atoms_eq.get_stress(voigt=True)  # eV/A^3
eq_pressure = -np.mean(eq_stress[:3]) * 160.21766  # GPa, positive = compressive
pressures.append(eq_pressure)
print(f"\nEquilibrium: V={V0_relaxed:.4f} A^3, E={E0_relaxed:.6f} eV, P={eq_pressure:.4f} GPa")

for i, eps in enumerate(strain_values):
    # Create isotropic deformation matrix: diag(1+eps, 1+eps, 1+eps)
    deform = np.eye(3) * (1.0 + eps)

    # Apply to relaxed structure
    strained = relaxed_structure.copy()
    new_lattice = np.dot(deform, strained.lattice.matrix)
    strained.lattice = new_lattice

    # Convert to ASE and relax ions only (fixed cell)
    atoms_s = adaptor.get_atoms(strained)
    atoms_s.calc = calc

    opt = LBFGS(atoms_s, logfile=os.devnull)
    opt.run(fmax=FMAX_EOS, steps=300)

    V = atoms_s.get_volume()
    E = atoms_s.get_potential_energy()
    stress = atoms_s.get_stress(voigt=True)
    P = -np.mean(stress[:3]) * 160.21766  # GPa

    volumes.append(V)
    energies.append(E)
    pressures.append(P)

    print(f"  Point {i+1}/{len(strain_values)}: eps={eps:+.4f}, "
          f"V={V:.4f} A^3 ({V/V0_relaxed*100:.1f}%), E={E:.6f} eV, P={P:.3f} GPa")

# Sort by volume
sort_idx = np.argsort(volumes)
volumes = np.array(volumes)[sort_idx]
energies = np.array(energies)[sort_idx]
pressures = np.array(pressures)[sort_idx]

# Normalize energies per atom for cleaner comparison
n_atoms = len(relaxed_structure)
energies_per_atom = energies / n_atoms

# ─── STEP 5: Fit EOS models ─────────────────────────────────────────────────

# --- Method 5a: ASE EquationOfState ---
print("\n" + "="*60)
print("EOS FITTING RESULTS")
print("="*60)

ase_eos_results = {}
for eos_name in ["birchmurnaghan", "murnaghan", "vinet"]:
    try:
        eos = ASE_EOS(volumes, energies, eos=eos_name)
        v0, e0, B0 = eos.fit()
        # B0 from ASE is in eV/A^3 -- convert to GPa
        B0_gpa = B0 * 160.21766
        ase_eos_results[eos_name] = {"V0": v0, "E0": e0, "B0_GPa": B0_gpa}
        print(f"\n  ASE {eos_name}:")
        print(f"    V0 = {v0:.4f} A^3, E0 = {e0:.6f} eV, B0 = {B0_gpa:.2f} GPa")
    except Exception as e:
        print(f"\n  ASE {eos_name}: FAILED -- {e}")

# --- Method 5b: pymatgen EOS (gives B0' too) ---
pmg_eos_results = {}
eos_model_names = ["birch_murnaghan", "murnaghan", "vinet"]

for eos_name in eos_model_names:
    try:
        eos = PMG_EOS(eos_name)
        eos_fit = eos.fit(volumes, energies)

        V0 = eos_fit.v0
        E0 = eos_fit.e0
        B0_gpa = eos_fit.b0_GPa

        # Extract B0' (pressure derivative of bulk modulus)
        # For Birch-Murnaghan: B0' is the 2nd parameter
        # pymatgen stores fit results in eos_fit.results
        results_dict = eos_fit.results
        B0_prime = results_dict.get("b1", None)

        pmg_eos_results[eos_name] = {
            "V0": float(V0),
            "E0": float(E0),
            "B0_GPa": float(B0_gpa),
            "B0_prime": float(B0_prime) if B0_prime is not None else None,
        }

        print(f"\n  pymatgen {eos_name}:")
        print(f"    V0 = {V0:.4f} A^3")
        print(f"    E0 = {E0:.6f} eV")
        print(f"    B0 = {B0_gpa:.2f} GPa")
        if B0_prime is not None:
            print(f"    B0' = {B0_prime:.2f}")
    except Exception as e:
        print(f"\n  pymatgen {eos_name}: FAILED -- {e}")

# --- Method 5c: Manual Birch-Murnaghan fitting (for full control) ---
from scipy.optimize import curve_fit

def birch_murnaghan_energy(V, E0, V0, B0, B0p):
    """3rd-order Birch-Murnaghan equation of state: E(V)."""
    eta = (V0 / V) ** (2.0 / 3.0)
    return E0 + (9.0 * V0 * B0 / 16.0) * (
        (eta - 1.0)**3 * B0p + (eta - 1.0)**2 * (6.0 - 4.0 * eta)
    )

def birch_murnaghan_pressure(V, V0, B0, B0p):
    """3rd-order Birch-Murnaghan pressure: P(V) in same units as B0."""
    eta = (V0 / V) ** (2.0 / 3.0)
    return (3.0 * B0 / 2.0) * (eta**(7.0/2.0) - eta**(5.0/2.0)) * (
        1.0 + 0.75 * (B0p - 4.0) * (eta - 1.0)
    )

try:
    # Initial guesses
    E0_guess = np.min(energies)
    V0_guess = volumes[np.argmin(energies)]
    B0_guess_ev = 0.5  # eV/A^3 ~ 80 GPa
    B0p_guess = 4.0

    popt, pcov = curve_fit(
        birch_murnaghan_energy,
        volumes, energies,
        p0=[E0_guess, V0_guess, B0_guess_ev, B0p_guess],
        maxfev=10000
    )
    E0_fit, V0_fit, B0_fit, B0p_fit = popt
    B0_fit_gpa = B0_fit * 160.21766

    perr = np.sqrt(np.diag(pcov))
    B0_err_gpa = perr[2] * 160.21766

    print(f"\n  Manual 3rd-order Birch-Murnaghan fit:")
    print(f"    V0 = {V0_fit:.4f} +/- {perr[1]:.4f} A^3")
    print(f"    E0 = {E0_fit:.6f} +/- {perr[0]:.6f} eV")
    print(f"    B0 = {B0_fit_gpa:.2f} +/- {B0_err_gpa:.2f} GPa")
    print(f"    B0' = {B0p_fit:.2f} +/- {perr[3]:.2f}")

    manual_fit = {
        "V0": float(V0_fit), "E0": float(E0_fit),
        "B0_GPa": float(B0_fit_gpa), "B0_prime": float(B0p_fit),
        "V0_err": float(perr[1]), "B0_err_GPa": float(B0_err_gpa),
        "B0_prime_err": float(perr[3]),
    }
except Exception as e:
    print(f"\n  Manual BM fit: FAILED -- {e}")
    manual_fit = None

# ─── STEP 6: Visualization ──────────────────────────────────────────────────

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

# --- Panel 1: E-V curve ---
ax1.plot(volumes, energies_per_atom, "ko", markersize=8, label="Calculated", zorder=5)

# Plot fitted curves
V_fine = np.linspace(volumes.min() * 0.98, volumes.max() * 1.02, 200)

colors = {"birch_murnaghan": "red", "murnaghan": "blue", "vinet": "green"}
for eos_name in eos_model_names:
    try:
        eos = PMG_EOS(eos_name)
        eos_fit = eos.fit(volumes, energies)
        E_fit = [eos_fit.func(v) for v in V_fine]
        ax1.plot(V_fine, np.array(E_fit) / n_atoms, "-",
                 color=colors.get(eos_name, "gray"), linewidth=1.5,
                 label=eos_name.replace("_", "-").title())
    except Exception:
        pass

# Mark equilibrium
if manual_fit:
    ax1.axvline(manual_fit["V0"], color="gray", linestyle=":", alpha=0.5)
    ax1.annotate(f"V$_0$={manual_fit['V0']:.2f} A$^3$",
                 xy=(manual_fit["V0"], min(energies_per_atom)),
                 xytext=(10, 20), textcoords="offset points",
                 arrowprops=dict(arrowstyle="->", color="gray"),
                 fontsize=10, color="gray")

ax1.set_xlabel("Volume ($\\AA^3$)", fontsize=12)
ax1.set_ylabel("Energy per atom (eV/atom)", fontsize=12)
ax1.set_title(f"E-V Curve: {formula}", fontsize=13)
ax1.legend(fontsize=10)
ax1.grid(True, alpha=0.3)

# --- Panel 2: P-V curve ---
ax2.plot(volumes, pressures, "ko", markersize=8, label="Calculated", zorder=5)

# Plot BM pressure curve
if manual_fit:
    P_fit = birch_murnaghan_pressure(
        V_fine, manual_fit["V0"],
        manual_fit["B0_GPa"] / 160.21766,  # back to eV/A^3
        manual_fit["B0_prime"]
    ) * 160.21766  # to GPa
    ax2.plot(V_fine, P_fit, "r-", linewidth=1.5, label="Birch-Murnaghan fit")

ax2.axhline(0, color="gray", linestyle=":", alpha=0.5)
ax2.set_xlabel("Volume ($\\AA^3$)", fontsize=12)
ax2.set_ylabel("Pressure (GPa)", fontsize=12)
ax2.set_title(f"P-V Curve: {formula}", fontsize=13)
ax2.legend(fontsize=10)
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "eos_curves.png"), dpi=150, bbox_inches="tight")
print(f"\nPlot saved to {OUTPUT_DIR}/eos_curves.png")

# ─── Save all results ───────────────────────────────────────────────────────
all_results = {
    "formula": formula,
    "n_atoms": n_atoms,
    "method": f"MACE-MP-0 ({MACE_MODEL})",
    "linear_strain_range": list(LINEAR_STRAIN),
    "n_points": N_POINTS,
    "volumes_A3": volumes.tolist(),
    "energies_eV": energies.tolist(),
    "energies_per_atom_eV": energies_per_atom.tolist(),
    "pressures_GPa": pressures.tolist(),
    "eos_fits": {
        "ase": ase_eos_results,
        "pymatgen": pmg_eos_results,
        "manual_birch_murnaghan": manual_fit,
    },
}
with open(os.path.join(OUTPUT_DIR, "eos_results.json"), "w") as f:
    json.dump(all_results, f, indent=2, default=str)
print(f"Results saved to {OUTPUT_DIR}/eos_results.json")
```

### Method B: QE DFT

This method uses Quantum ESPRESSO `pw.x` to compute total energies at multiple volumes. The workflow mirrors atomate2: relax the equilibrium cell, then perform fixed-volume relaxations (or SCF calculations) at strained volumes.

#### Step B1: Relax the equilibrium structure

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

#### Step B2: Generate volume-strained QE inputs

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
LINEAR_STRAIN = (-0.05, 0.05)
N_POINTS = 9
WORK_DIR = "eos_qe"
# ─────────────────────────────────────────────────────────────────────────────

os.makedirs(WORK_DIR, exist_ok=True)

structure = Structure.from_file(RELAXED_FILE)
formula = structure.composition.reduced_formula
V0 = structure.volume
print(f"Relaxed structure: {formula}, V0 = {V0:.4f} A^3")

# Generate strain values (with zero-strain shifted, following atomate2)
strain_values, strain_delta = np.linspace(
    LINEAR_STRAIN[0], LINEAR_STRAIN[1], N_POINTS, retstep=True
)
zero_mask = np.abs(strain_values) < 1e-15
if np.any(zero_mask):
    nzs = int(np.sum(zero_mask))
    shift = strain_delta / (nzs + 1.0) * np.linspace(-1.0, 1.0, nzs)
    strain_values[zero_mask] += shift

# Pseudopotential map
pseudo_map = {}
for el in structure.composition.elements:
    symbol = el.symbol
    pseudo_map[symbol] = f"{symbol}.pbe-n-rrkjus_psl.1.0.0.UPF"

eos_info = []
for idx, eps in enumerate(strain_values):
    deform_dir = os.path.join(WORK_DIR, f"vol_{idx:03d}")
    os.makedirs(deform_dir, exist_ok=True)

    # Isotropic strain: scale all lattice vectors by (1 + eps)
    strained = structure.copy()
    scale = 1.0 + eps
    new_lattice = strained.lattice.matrix * scale
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
        "linear_strain": float(eps),
        "volume_ratio": float(scale**3),
        "volume_A3": float(V_strained),
        "directory": deform_dir,
    })
    print(f"  Point {idx}: eps={eps:+.4f}, V={V_strained:.4f} A^3 ({V_strained/V0*100:.1f}%)")

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

#### Step B3: Post-process and fit EOS

```python
#!/usr/bin/env python3
"""
Post-process QE EOS outputs: extract E(V) data and fit multiple EOS models.
Run after all volume-point calculations complete.
"""

import os
import json
import re
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from pymatgen.core.structure import Structure
from pymatgen.analysis.eos import EOS as PMG_EOS
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


def parse_qe_energy_volume(output_file):
    """
    Extract total energy (Ry) and cell volume (A^3) from a QE pw.x output.
    Returns (energy_eV, volume_A3).
    """
    energy_ry = None
    volume = None
    with open(output_file) as f:
        for line in f:
            # Look for final total energy (last "!" line)
            if line.strip().startswith("!"):
                match = re.search(r"total energy\s*=\s*([-\d.]+)\s*Ry", line)
                if match:
                    energy_ry = float(match.group(1))
            # Look for unit-cell volume
            if "unit-cell volume" in line:
                match = re.search(r"=\s*([\d.]+)\s*\(a\.u\.\)\^3", line)
                if match:
                    volume_bohr3 = float(match.group(1))
                    volume = volume_bohr3 * 0.14818471147  # bohr^3 to A^3

    if energy_ry is None or volume is None:
        raise ValueError(f"Could not parse energy/volume from {output_file}")

    energy_ev = energy_ry * 13.605693123  # Ry to eV
    return energy_ev, volume


def parse_qe_pressure(output_file):
    """Extract pressure (kbar) from QE output. Returns pressure in GPa."""
    pressure_kbar = None
    with open(output_file) as f:
        for line in f:
            if "P=" in line and "total   stress" in line:
                match = re.search(r"P=\s*([-\d.]+)", line)
                if match:
                    pressure_kbar = float(match.group(1))
    if pressure_kbar is not None:
        return pressure_kbar * 0.1  # kbar to GPa
    return None


# Load EOS metadata
with open(os.path.join(WORK_DIR, "eos_info.json")) as f:
    eos_info = json.load(f)

volumes = []
energies = []
pressures = []

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

if len(volumes) < 4:
    raise RuntimeError("Need at least 4 data points for EOS fit.")

# Sort by volume
sort_idx = np.argsort(volumes)
volumes = np.array(volumes)[sort_idx]
energies = np.array(energies)[sort_idx]
pressures = np.array(pressures)[sort_idx]
energies_per_atom = energies / n_atoms

# ─── Fit EOS models (pymatgen) ───────────────────────────────────────────────
print("\n" + "="*60)
print("EOS FITTING RESULTS (QE DFT)")
print("="*60)

eos_results = {}
eos_model_names = ["birch_murnaghan", "murnaghan", "vinet"]

for eos_name in eos_model_names:
    try:
        eos = PMG_EOS(eos_name)
        eos_fit = eos.fit(volumes, energies)
        results_dict = eos_fit.results
        B0_prime = results_dict.get("b1", None)

        eos_results[eos_name] = {
            "V0": float(eos_fit.v0),
            "E0": float(eos_fit.e0),
            "B0_GPa": float(eos_fit.b0_GPa),
            "B0_prime": float(B0_prime) if B0_prime is not None else None,
        }
        print(f"\n  {eos_name}:")
        print(f"    V0 = {eos_fit.v0:.4f} A^3")
        print(f"    E0 = {eos_fit.e0:.6f} eV ({eos_fit.e0/n_atoms:.6f} eV/atom)")
        print(f"    B0 = {eos_fit.b0_GPa:.2f} GPa")
        if B0_prime is not None:
            print(f"    B0' = {B0_prime:.2f}")
    except Exception as e:
        print(f"\n  {eos_name}: FAILED -- {e}")

# ─── Manual 3rd-order Birch-Murnaghan fit ────────────────────────────────────
def birch_murnaghan_energy(V, E0, V0, B0, B0p):
    eta = (V0 / V) ** (2.0 / 3.0)
    return E0 + (9.0 * V0 * B0 / 16.0) * (
        (eta - 1.0)**3 * B0p + (eta - 1.0)**2 * (6.0 - 4.0 * eta)
    )

def birch_murnaghan_pressure(V, V0, B0, B0p):
    eta = (V0 / V) ** (2.0 / 3.0)
    return (3.0 * B0 / 2.0) * (eta**(7.0/2.0) - eta**(5.0/2.0)) * (
        1.0 + 0.75 * (B0p - 4.0) * (eta - 1.0)
    )

try:
    E0g = np.min(energies)
    V0g = volumes[np.argmin(energies)]
    B0g = 0.5  # eV/A^3
    B0pg = 4.0

    popt, pcov = curve_fit(
        birch_murnaghan_energy, volumes, energies,
        p0=[E0g, V0g, B0g, B0pg], maxfev=10000
    )
    E0_fit, V0_fit, B0_fit, B0p_fit = popt
    perr = np.sqrt(np.diag(pcov))

    manual_fit = {
        "V0": float(V0_fit),
        "E0": float(E0_fit),
        "B0_GPa": float(B0_fit * 160.21766),
        "B0_prime": float(B0p_fit),
        "V0_err": float(perr[1]),
        "B0_err_GPa": float(perr[2] * 160.21766),
        "B0_prime_err": float(perr[3]),
    }
    print(f"\n  Manual 3rd-order Birch-Murnaghan:")
    print(f"    V0 = {V0_fit:.4f} +/- {perr[1]:.4f} A^3")
    print(f"    E0 = {E0_fit:.6f} +/- {perr[0]:.6f} eV")
    print(f"    B0 = {B0_fit*160.21766:.2f} +/- {perr[2]*160.21766:.2f} GPa")
    print(f"    B0' = {B0p_fit:.2f} +/- {perr[3]:.2f}")
except Exception as e:
    print(f"\n  Manual BM fit: FAILED -- {e}")
    manual_fit = None

# ─── Visualization ───────────────────────────────────────────────────────────
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

# Panel 1: E-V
ax1.plot(volumes, energies_per_atom, "ko", markersize=8, label="QE DFT", zorder=5)

V_fine = np.linspace(volumes.min() * 0.98, volumes.max() * 1.02, 200)
colors = {"birch_murnaghan": "red", "murnaghan": "blue", "vinet": "green"}
for eos_name in eos_model_names:
    try:
        eos = PMG_EOS(eos_name)
        eos_fit = eos.fit(volumes, energies)
        E_fit = [eos_fit.func(v) for v in V_fine]
        ax1.plot(V_fine, np.array(E_fit)/n_atoms, "-",
                 color=colors.get(eos_name, "gray"), linewidth=1.5,
                 label=eos_name.replace("_", "-").title())
    except Exception:
        pass

if manual_fit:
    ax1.axvline(manual_fit["V0"], color="gray", ls=":", alpha=0.5)

ax1.set_xlabel("Volume ($\\AA^3$)", fontsize=12)
ax1.set_ylabel("Energy (eV/atom)", fontsize=12)
ax1.set_title(f"E-V: {formula} (QE DFT)", fontsize=13)
ax1.legend(fontsize=10)
ax1.grid(True, alpha=0.3)

# Panel 2: P-V
ax2.plot(volumes, pressures, "ko", markersize=8, label="QE DFT", zorder=5)

if manual_fit:
    P_fit = birch_murnaghan_pressure(
        V_fine, manual_fit["V0"],
        manual_fit["B0_GPa"]/160.21766,
        manual_fit["B0_prime"]
    ) * 160.21766
    ax2.plot(V_fine, P_fit, "r-", lw=1.5, label="BM fit")

ax2.axhline(0, color="gray", ls=":", alpha=0.5)
ax2.set_xlabel("Volume ($\\AA^3$)", fontsize=12)
ax2.set_ylabel("Pressure (GPa)", fontsize=12)
ax2.set_title(f"P-V: {formula} (QE DFT)", fontsize=13)
ax2.legend(fontsize=10)
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "eos_curves_qe.png"), dpi=150, bbox_inches="tight")
print(f"\nPlot saved to {OUTPUT_DIR}/eos_curves_qe.png")

# ─── Save results ────────────────────────────────────────────────────────────
all_results = {
    "formula": formula,
    "n_atoms": n_atoms,
    "method": "QE PBE",
    "volumes_A3": volumes.tolist(),
    "energies_eV": energies.tolist(),
    "energies_per_atom_eV": energies_per_atom.tolist(),
    "pressures_GPa": pressures.tolist(),
    "eos_fits": {
        "pymatgen": eos_results,
        "manual_birch_murnaghan": manual_fit,
    },
}
with open(os.path.join(OUTPUT_DIR, "eos_results.json"), "w") as f:
    json.dump(all_results, f, indent=2, default=str)
print(f"Results saved to {OUTPUT_DIR}/eos_results.json")
```

## Key Parameters

| Parameter | Typical Value | Notes |
|---|---|---|
| Volume range | +/-5% linear strain (+/-14% to +16% volume) | Wider range for high-pressure studies. For soft materials, +/-3% may suffice. |
| Number of points | 7--11 (odd preferred) | More points improve fit quality. Minimum 4 for a 3-parameter fit, 5 for 4-parameter (3rd-order BM). |
| ecutwfc (QE) | 60--80 Ry | Must be converged. Use the same cutoff for all volume points. |
| k-grid (QE) | Dense, consistent | Use the same k-grid for all points to avoid systematic errors. |
| conv_thr (QE) | 1e-10 Ry | Tight convergence needed since E-V differences can be small (meV range). |
| EOS model | Birch-Murnaghan (3rd order) | Most widely used. Vinet works better for very compressed or expanded volumes. Murnaghan is simpler but less accurate at large compressions. |

## Interpreting Results

**Equilibrium volume (V0):** The volume at the E-V minimum. Compare with experiment. PBE typically overestimates by 1--3%.

**Bulk modulus (B0):** Resistance to uniform compression. Units: GPa. Compare:
- Metals: 30--400 GPa (Na ~7, Fe ~170, W ~310, diamond ~440)
- Semiconductors: 50--100 GPa (Si ~98, GaAs ~75)
- If B0 differs significantly between MACE and QE, trust QE.

**Pressure derivative (B0'):** Dimensionless, typically 3.5--6 for most solids. B0' ~ 4 is a common default. Values outside 2--8 may indicate fitting issues.

**Comparing EOS models:** If Birch-Murnaghan, Murnaghan, and Vinet give consistent V0 and B0 (within 1--2%), the fit is robust. Large disagreements indicate the data range may be too wide or too narrow, or the data has noise.

**E-V curve shape:** Should be a smooth parabola-like curve with a clear minimum. Kinks or scatter indicate convergence problems in individual calculations.

## Common Issues

| Problem | Cause | Solution |
|---|---|---|
| E-V curve has kinks or scatter | Inconsistent convergence across volume points | Ensure identical `ecutwfc`, `ecutrho`, k-grid, and `conv_thr` for all points. Check that all calculations converged. |
| B0 unreasonably high or low | Volume range too narrow or data noise | Expand the strain range to +/-7%. Add more data points. |
| B0' outside 2--8 range | Insufficient data points or too-narrow range | Use at least 7 points. Expand volume range. Check for unconverged calculations. |
| Fit does not converge | Poor initial guess or bad data | Remove obvious outlier points. Try different EOS models. Check that energies span at least ~0.1 eV range. |
| Different EOS models give different B0 | Non-parabolic E-V near extremes | Use only the central 80% of data points, or reduce volume range. Birch-Murnaghan is most reliable for moderate compressions. |
| MACE gives different B0 than QE | MACE model limitations | Expected for materials outside MACE training domain. Use QE as ground truth. |
| QE calculation crashes at large compression | Basis set or pseudopotential breakdown | Reduce compression range. Increase `ecutwfc`. Check pseudopotential validity at high pressure. |
| Volume points not evenly spaced | Strain applied to lattice vectors, not volume | This is correct -- linear strain produces non-uniform volume spacing because V ~ (1+eps)^3. The EOS fitting handles this. |
