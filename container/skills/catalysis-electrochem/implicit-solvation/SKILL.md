# Implicit Solvation for Electrochemistry

## When to Use

- You need to model a solid-liquid interface (electrode in contact with electrolyte) without explicit water molecules.
- You want solvation free energies for molecules or ions from first principles.
- You are computing electrochemical reaction energetics and need solvent stabilization of charged intermediates.
- You need to estimate pKa values from DFT solvation energies.
- You want to model surface charge and the electrical double layer at an electrode-electrolyte interface.
- You are studying corrosion, electrocatalysis (HER, OER, ORR, CO2RR), or battery electrode-electrolyte reactions.
- Explicit solvation is too expensive or you need a quick screening approach.

## Method Selection

| Criterion | VASPsol (VASP + solvation) | QE + Environ | ASE Solvation Corrections |
|---|---|---|---|
| Accuracy | High (self-consistent cavity + PB) | High (self-consistent SCCS/PB) | Approximate (empirical or Born model) |
| Theory | Linearized Poisson-Boltzmann | SCCS or linearized PB | Born/GB/PCM-like analytical |
| Requires DFT code | VASP (commercial license) | Quantum ESPRESSO + Environ plugin | No DFT needed |
| Self-consistent | Yes | Yes | No (post-hoc correction) |
| Electrolyte ions | Yes (Debye screening) | Yes (PB electrolyte) | No |
| Speed | DFT cost + ~10-20% overhead | DFT cost + ~10-20% overhead | Seconds (analytical) |
| Best for | Publication-quality solvation at metal surfaces | Publication-quality, open-source DFT solvation | Rapid screening, trends, non-DFT workflows |

```
Need self-consistent solvation with DFT accuracy?
  Have VASP license?
    YES --> Method A: VASPsol (linearized PB, dielectric cavity)
  Have QE + Environ compiled?
    YES --> Method B: QE + Environ (SCCS or PB model)
  Neither available?
    --> Method C: ASE-based solvation corrections (Born, PCM-like, empirical)

Need quick solvation energy estimates for screening?
  --> Method C: analytical solvation models (seconds per structure)

Need double-layer / surface charge modeling?
  --> Method A or B (self-consistent electrostatics required)
```

## Prerequisites

- **Method A (VASPsol)**: VASP compiled with VASPsol patch, ASE for structure I/O
- **Method B (QE + Environ)**: Quantum ESPRESSO pw.x compiled with Environ module, pseudopotentials in `./pseudo/`
- **Method C (ASE corrections)**: ASE, numpy, scipy, matplotlib (no DFT code needed)
- Python packages: `numpy`, `scipy`, `matplotlib`, `ase`, `pymatgen`

## Background: Implicit Solvation Theory

Implicit solvation replaces explicit solvent molecules with a continuous dielectric medium. The key equation solved is the (linearized) Poisson-Boltzmann equation:

```
-nabla . [epsilon(r) nabla phi(r)] = 4*pi * rho_solute(r) + lambda_D^2 * epsilon_bulk * phi(r)
```

where:
- `epsilon(r)` is the position-dependent dielectric function (1 inside solute, ~80 in water)
- `phi(r)` is the electrostatic potential
- `rho_solute(r)` is the solute charge density (from DFT)
- `lambda_D` is the inverse Debye screening length (from electrolyte concentration)

The dielectric cavity is defined by the electron density of the solute:

```
epsilon(r) = 1 + (epsilon_bulk - 1) * S(rho_el(r))
```

where `S` is a switching function that transitions from 0 (inside the solute, high electron density) to 1 (in the solvent, low electron density). VASPsol uses `NC_K` and `SIGMA_K` to define this transition; Environ uses `rhomax` and `rhomin`.

The solvation energy is:

```
Delta G_solv = E_total(in solvent) - E_total(in vacuum)
```

---

## Detailed Steps

### Method A: VASPsol -- VASP with Implicit Solvation

VASPsol adds implicit solvation to VASP via the linearized Poisson-Boltzmann model. The key INCAR parameters are `LSOL = .TRUE.` to activate solvation, `EB_K` for the bulk dielectric constant, `TAU` for the cavity surface tension, and optionally `LAMBDA_D_K` for Debye screening (electrolyte).

#### Step A1: Generate VASPsol Input Files

```python
#!/usr/bin/env python3
"""
Generate VASP input files with VASPsol implicit solvation.
Demonstrates the key parameters for solvation in electrochemistry.

Example: Pt(111) slab in aqueous electrolyte.
"""
import warnings
warnings.filterwarnings("ignore")

import numpy as np
from ase.build import fcc111
from ase.io import write
from ase.constraints import FixAtoms

# ============================================================
# 1. Build slab model
# ============================================================
slab = fcc111("Pt", size=(3, 3, 4), vacuum=15.0)
n_slab = len(slab)

# Fix bottom 2 layers
z_coords = slab.get_positions()[:, 2]
z_vals = sorted(set(np.round(z_coords, 1)))
fix_z = z_vals[1] + 0.5  # fix atoms in bottom 2 layers
fix_mask = z_coords < fix_z
slab.set_constraint(FixAtoms(mask=fix_mask))

write("POSCAR", slab, format="vasp")
print(f"Slab: {n_slab} Pt atoms, 3x3x4 layers, 15 A vacuum")

# ============================================================
# 2. Write INCAR for vacuum calculation (reference)
# ============================================================
incar_vacuum = """# Pt(111) slab -- VACUUM reference
SYSTEM  = Pt111_vacuum
ENCUT   = 450
EDIFF   = 1E-5
EDIFFG  = -0.02
IBRION  = 2
NSW     = 100
ISIF    = 2
ISMEAR  = 1
SIGMA   = 0.1
LREAL   = Auto
ALGO    = Normal
PREC    = Accurate
LORBIT  = 11
LWAVE   = .FALSE.
LCHARG  = .FALSE.

# NO solvation -- vacuum reference
# LSOL = .FALSE. (default)
"""

with open("INCAR_vacuum", "w") as f:
    f.write(incar_vacuum)

# ============================================================
# 3. Write INCAR for VASPsol solvation calculation
# ============================================================
incar_solvated = """# Pt(111) slab -- VASPsol implicit solvation
SYSTEM  = Pt111_solvated
ENCUT   = 450
EDIFF   = 1E-5
EDIFFG  = -0.02
IBRION  = 2
NSW     = 100
ISIF    = 2
ISMEAR  = 1
SIGMA   = 0.1
LREAL   = Auto
ALGO    = Normal
PREC    = Accurate
LORBIT  = 11
LWAVE   = .FALSE.
LCHARG  = .FALSE.

# ── VASPsol solvation parameters ──────────────────────────
LSOL      = .TRUE.          # Activate implicit solvation
EB_K      = 78.4            # Bulk dielectric constant (water at 298 K)
TAU       = 0.000525        # Cavity surface tension (eV/A^2)
                            # Fitted to reproduce solvation energies of small molecules
                            # Default: 0.000525 eV/A^2

# ── Optional: electrolyte (Debye-Huckel screening) ────────
# LAMBDA_D_K = 3.0          # Debye screening length in Angstrom
                            # lambda_D = sqrt(eps_0 * eps_r * kT / (2 * c * e^2))
                            # For 1 M NaCl at 298K: lambda_D ~ 3.0 A
                            # For 0.1 M: lambda_D ~ 9.6 A
                            # Uncomment to include electrolyte ions

# ── Advanced cavity parameters (usually keep defaults) ─────
# NC_K      = 0.0025        # Critical electron density for cavity (e/A^3)
                            # Defines where epsilon transitions from 1 to eps_bulk
# SIGMA_K   = 0.6           # Width of the dielectric cavity transition (A)
"""

with open("INCAR_solvated", "w") as f:
    f.write(incar_solvated)

# ============================================================
# 4. Write KPOINTS
# ============================================================
kpoints = """Automatic k-mesh
0
Gamma
4 4 1
0 0 0
"""
with open("KPOINTS", "w") as f:
    f.write(kpoints)

print("Generated: POSCAR, INCAR_vacuum, INCAR_solvated, KPOINTS")
print()
print("Workflow:")
print("  1. Run VASP with INCAR_vacuum -> get E_vacuum")
print("  2. Run VASP with INCAR_solvated -> get E_solvated")
print("  3. Delta_G_solv = E_solvated - E_vacuum")
print()
print("For electrochemistry with surface charge:")
print("  Add NELECT to control the number of electrons")
print("  NELECT = N_default + delta_q  (negative delta_q = add electrons = negative charge)")
```

#### Step A2: Parse VASPsol Results and Compute Solvation Energy

```python
#!/usr/bin/env python3
"""
Parse VASP/VASPsol OUTCAR files and compute solvation energy.
Also demonstrates surface charge analysis for electrochemistry.
"""
import re
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import json


def parse_vasp_energy(outcar_file):
    """Extract the final total energy from VASP OUTCAR."""
    energy = None
    with open(outcar_file) as f:
        for line in f:
            # "free  energy   TOTEN  =   -xxx.xxxxxxxx eV"
            if "free  energy   TOTEN" in line:
                m = re.search(r"=\s+([-\d.]+)", line)
                if m:
                    energy = float(m.group(1))
    return energy


def parse_vaspsol_components(outcar_file):
    """
    Parse VASPsol energy components from OUTCAR.
    VASPsol prints additional lines for the solvation contributions.
    """
    e_sol_el = None    # electrostatic solvation energy
    e_sol_cav = None   # cavitation energy
    with open(outcar_file) as f:
        for line in f:
            if "Solvation energy (Eelec)" in line or "electrostatic" in line.lower():
                m = re.search(r"=\s+([-\d.]+)", line)
                if m:
                    e_sol_el = float(m.group(1))
            if "cavitation energy" in line.lower() or "Ecav" in line:
                m = re.search(r"=\s+([-\d.]+)", line)
                if m:
                    e_sol_cav = float(m.group(1))
    return {"electrostatic": e_sol_el, "cavitation": e_sol_cav}


# ============================================================
# Example: Compute solvation energy from two OUTCAR files
# ============================================================
# Replace these with actual file paths after running VASP
vacuum_outcar = "vacuum/OUTCAR"
solvated_outcar = "solvated/OUTCAR"

# --- Demo with placeholder values (replace with actual parsing) ---
# In a real calculation:
# E_vacuum = parse_vasp_energy(vacuum_outcar)
# E_solvated = parse_vasp_energy(solvated_outcar)
# sol_components = parse_vaspsol_components(solvated_outcar)

# Placeholder values for a Pt(111) slab (typical results)
E_vacuum = -216.384       # eV (example)
E_solvated = -216.512     # eV (example)

delta_G_solv = E_solvated - E_vacuum
print(f"E(vacuum)   = {E_vacuum:.4f} eV")
print(f"E(solvated) = {E_solvated:.4f} eV")
print(f"Delta G_solv = {delta_G_solv:.4f} eV ({delta_G_solv * 23.061:.2f} kcal/mol)")

# ============================================================
# Surface charge scan for capacitance / PZC determination
# ============================================================
# By varying NELECT, we can compute the energy at different surface charges
# and determine the potential of zero charge (PZC) and capacitance.

# Example data: (delta_q in electrons, E_total in eV)
# delta_q > 0 means extra electrons (negative surface charge)
# delta_q < 0 means fewer electrons (positive surface charge)
charge_data = {
    "delta_q": [-0.5, -0.25, 0.0, 0.25, 0.5],
    "E_solvated": [-216.210, -216.398, -216.512, -216.558, -216.541],
    "E_vacuum": [-215.900, -216.180, -216.384, -216.520, -216.590],
}

delta_q = np.array(charge_data["delta_q"])
E_solv = np.array(charge_data["E_solvated"])
E_vac = np.array(charge_data["E_vacuum"])

# The solvation energy as a function of charge
dG_solv_q = E_solv - E_vac

# Fit a parabola to E_solvated(q) to find PZC
# Grand potential: Omega(q) = E(q) - q * phi_ref
# d(Omega)/dq = 0 at PZC
coeffs = np.polyfit(delta_q, E_solv, 2)
# PZC: dE/dq = 0 => 2*a*q + b = 0 => q_PZC = -b/(2a)
q_PZC = -coeffs[1] / (2 * coeffs[0])
# Capacitance: C = d^2E/dq^2 = 2*a (in eV/e^2)
# Convert: C (uF/cm^2) = 2*a * e / A_surface (needs surface area)
capacitance_eV = 2 * coeffs[0]

print(f"\nCharge of zero charge (q_PZC) = {q_PZC:.3f} e")
print(f"Capacitance = {capacitance_eV:.3f} eV/e^2")

# ============================================================
# Plot solvation energy vs charge
# ============================================================
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

# Panel 1: Total energies
ax1.plot(delta_q, E_solv, "o-", color="steelblue", label="Solvated", linewidth=2)
ax1.plot(delta_q, E_vac, "s--", color="coral", label="Vacuum", linewidth=2)
q_fit = np.linspace(delta_q[0], delta_q[-1], 50)
ax1.plot(q_fit, np.polyval(coeffs, q_fit), "-", color="steelblue", alpha=0.3, linewidth=3)
ax1.axvline(q_PZC, color="gray", linestyle=":", label=f"PZC (q={q_PZC:.2f} e)")
ax1.set_xlabel(r"$\Delta q$ (electrons)", fontsize=13)
ax1.set_ylabel("Total energy (eV)", fontsize=13)
ax1.set_title("Energy vs Surface Charge", fontsize=14)
ax1.legend(fontsize=11)
ax1.grid(alpha=0.3)

# Panel 2: Solvation energy vs charge
ax2.plot(delta_q, dG_solv_q, "o-", color="green", linewidth=2, markersize=8)
ax2.axhline(0, color="gray", linestyle="-", linewidth=0.5)
ax2.set_xlabel(r"$\Delta q$ (electrons)", fontsize=13)
ax2.set_ylabel(r"$\Delta G_{solv}$ (eV)", fontsize=13)
ax2.set_title("Solvation Energy vs Surface Charge", fontsize=14)
ax2.grid(alpha=0.3)

plt.tight_layout()
plt.savefig("solvation_vs_charge.png", dpi=200, bbox_inches="tight")
print("Saved: solvation_vs_charge.png")

# ============================================================
# Save results
# ============================================================
results = {
    "E_vacuum_eV": E_vacuum,
    "E_solvated_eV": E_solvated,
    "delta_G_solv_eV": delta_G_solv,
    "delta_G_solv_kcal_mol": delta_G_solv * 23.061,
    "charge_scan": {
        "delta_q_electrons": charge_data["delta_q"],
        "E_solvated_eV": charge_data["E_solvated"],
        "E_vacuum_eV": charge_data["E_vacuum"],
        "delta_G_solv_eV": dG_solv_q.tolist(),
        "q_PZC_electrons": float(q_PZC),
        "capacitance_eV_per_e2": float(capacitance_eV),
    },
}
with open("vaspsol_results.json", "w") as f:
    json.dump(results, f, indent=2)
print("Saved: vaspsol_results.json")
```

### Method B: Quantum ESPRESSO + Environ Module

The Environ module adds self-consistent continuum solvation to QE. It supports the Self-Consistent Continuum Solvation (SCCS) model and the linearized Poisson-Boltzmann model. Environ is compiled as a plugin to QE.

#### Step B1: SCF in Vacuum (Reference)

```python
#!/usr/bin/env python3
"""
Run QE SCF for a slab in vacuum (reference for solvation energy).
Example: Pt(111) slab.
"""
import os
import subprocess
import re
import numpy as np
from ase.build import fcc111
from ase.io import write

# ============================================================
# 1. Build slab
# ============================================================
slab = fcc111("Pt", size=(2, 2, 4), vacuum=15.0)
n_atoms = len(slab)
cell = slab.get_cell()
symbols = slab.get_chemical_symbols()
positions = slab.get_positions()

# Fix bottom 2 layers
z_coords = positions[:, 2]
z_vals = sorted(set(np.round(z_coords, 1)))
fix_z = z_vals[1] + 0.5

PSEUDO_DIR = os.path.abspath("./pseudo")
OUTDIR = os.path.abspath("./tmp_solv_vacuum")
os.makedirs(OUTDIR, exist_ok=True)

# ============================================================
# 2. Write QE input -- vacuum
# ============================================================
cell_lines = "\n".join(
    f"  {v[0]:.10f}  {v[1]:.10f}  {v[2]:.10f}" for v in cell
)

pos_lines = []
for i, (sym, pos) in enumerate(zip(symbols, positions)):
    if_pos = "0 0 0" if z_coords[i] < fix_z else "1 1 1"
    pos_lines.append(f"  {sym}  {pos[0]:.10f}  {pos[1]:.10f}  {pos[2]:.10f}  {if_pos}")
pos_card = "\n".join(pos_lines)

qe_input = f"""&CONTROL
    calculation = 'scf'
    prefix      = 'pt_slab'
    outdir      = '{OUTDIR}'
    pseudo_dir  = '{PSEUDO_DIR}'
    tprnfor     = .true.
/
&SYSTEM
    ibrav       = 0
    nat         = {n_atoms}
    ntyp        = 1
    ecutwfc     = 50.0
    ecutrho     = 400.0
    occupations = 'smearing'
    smearing    = 'mv'
    degauss     = 0.02
/
&ELECTRONS
    conv_thr    = 1.0d-6
    mixing_beta = 0.3
/

CELL_PARAMETERS angstrom
{cell_lines}

ATOMIC_SPECIES
  Pt  195.078  Pt.pbe-n-rrkjus_psl.1.0.0.UPF

ATOMIC_POSITIONS angstrom
{pos_card}

K_POINTS automatic
  4 4 1  0 0 0
"""

with open("pt_vacuum.in", "w") as f:
    f.write(qe_input)

print(f"Slab: {n_atoms} Pt atoms")
print("Running vacuum SCF...")

result = subprocess.run(
    ["mpirun", "--allow-run-as-root", "-np", "4", "pw.x", "-in", "pt_vacuum.in"],
    capture_output=True, text=True, timeout=1200
)
with open("pt_vacuum.out", "w") as f:
    f.write(result.stdout)

# Parse energy
e_vacuum = None
for line in result.stdout.split("\n"):
    if "!" in line and "total energy" in line:
        m = re.search(r"=\s+([-\d.]+)\s+Ry", line)
        if m:
            e_vacuum = float(m.group(1)) * 13.6057  # Ry to eV
if e_vacuum:
    print(f"E(vacuum) = {e_vacuum:.6f} eV")
else:
    print("WARNING: Could not parse energy. Check pt_vacuum.out")
```

#### Step B2: SCF with Environ Solvation

```python
#!/usr/bin/env python3
"""
Run QE SCF with Environ implicit solvation.
Requires QE compiled with the Environ module.

The Environ input is specified via a separate 'environ.in' file
or via additional namelists in the QE input.

Two solvation models are available:
  1. SCCS (Self-Consistent Continuum Solvation) -- recommended for neutral surfaces
  2. Linearized Poisson-Boltzmann -- for charged surfaces / electrolyte

Example: Pt(111) slab in water.
"""
import os
import subprocess
import re
import numpy as np
from ase.build import fcc111

# ============================================================
# 1. Build slab (same as vacuum calculation)
# ============================================================
slab = fcc111("Pt", size=(2, 2, 4), vacuum=15.0)
n_atoms = len(slab)
cell = slab.get_cell()
symbols = slab.get_chemical_symbols()
positions = slab.get_positions()

z_coords = positions[:, 2]
z_vals = sorted(set(np.round(z_coords, 1)))
fix_z = z_vals[1] + 0.5

PSEUDO_DIR = os.path.abspath("./pseudo")
OUTDIR = os.path.abspath("./tmp_solv_environ")
os.makedirs(OUTDIR, exist_ok=True)

cell_lines = "\n".join(
    f"  {v[0]:.10f}  {v[1]:.10f}  {v[2]:.10f}" for v in cell
)

pos_lines = []
for i, (sym, pos) in enumerate(zip(symbols, positions)):
    if_pos = "0 0 0" if z_coords[i] < fix_z else "1 1 1"
    pos_lines.append(f"  {sym}  {pos[0]:.10f}  {pos[1]:.10f}  {pos[2]:.10f}  {if_pos}")
pos_card = "\n".join(pos_lines)

# ============================================================
# 2. Write QE + Environ input
# ============================================================
# NOTE: Environ is activated via the 'environ.in' file in the
# working directory. QE must be compiled with Environ support.
# When Environ is active, add environ_thr to &ELECTRONS.

qe_input = f"""&CONTROL
    calculation = 'scf'
    prefix      = 'pt_slab'
    outdir      = '{OUTDIR}'
    pseudo_dir  = '{PSEUDO_DIR}'
    tprnfor     = .true.
/
&SYSTEM
    ibrav       = 0
    nat         = {n_atoms}
    ntyp        = 1
    ecutwfc     = 50.0
    ecutrho     = 400.0
    occupations = 'smearing'
    smearing    = 'mv'
    degauss     = 0.02
/
&ELECTRONS
    conv_thr    = 1.0d-6
    mixing_beta = 0.3
/

CELL_PARAMETERS angstrom
{cell_lines}

ATOMIC_SPECIES
  Pt  195.078  Pt.pbe-n-rrkjus_psl.1.0.0.UPF

ATOMIC_POSITIONS angstrom
{pos_card}

K_POINTS automatic
  4 4 1  0 0 0
"""

with open("pt_solvated.in", "w") as f:
    f.write(qe_input)

# ============================================================
# 3. Write Environ input file (environ.in)
# ============================================================
# Environ reads its parameters from environ.in in the run directory.
# Below is the SCCS model (Andreussi, Dabo, Marzari, JCP 2012).

environ_sccs = """&ENVIRON
    verbose         = 1          ! 0=silent, 1=summary, 2=debug
    environ_thr     = 1.0d-1     ! threshold for Environ convergence (Ry)
    environ_type    = 'vacuum'   ! overridden by solvent settings below
    env_static_permittivity = 78.3  ! bulk dielectric constant of water
    env_surface_tension = 50.0d-3   ! surface tension in Ry/bohr^2
                                     ! (~72 dyn/cm for water, converted)
    env_pressure    = 0.0        ! external pressure (usually 0)
/
&BOUNDARY
    solvent_mode    = 'electronic'  ! cavity from electron density
    ! --- SCCS cavity definition ---
    rhomax          = 0.005      ! max density for full solvation (e/bohr^3)
    rhomin          = 0.0001     ! min density -- no solvation above this (e/bohr^3)
    ! rhomax and rhomin define the smooth dielectric transition region.
    ! Inside (rho > rhomax): epsilon = 1 (vacuum)
    ! Outside (rho < rhomin): epsilon = eps_bulk (water)
    ! Between: smooth interpolation
/
&ELECTROSTATIC
    pbc_correction  = 'parabolic'  ! correct for periodic boundary conditions
    pbc_dim         = 2            ! 2D slab geometry
    pbc_axis        = 3            ! slab normal along z
    tol             = 1.0d-11      ! solver tolerance
/
"""

with open("environ.in", "w") as f:
    f.write(environ_sccs)

print("Generated: pt_solvated.in, environ.in (SCCS model)")

# ============================================================
# 4. Alternative: Environ with Linearized Poisson-Boltzmann
# ============================================================
# For electrolyte with ionic screening (e.g., 1 M NaCl)

environ_lpb = """&ENVIRON
    verbose         = 1
    environ_thr     = 1.0d-1
    env_static_permittivity = 78.3
    env_surface_tension = 50.0d-3
    env_electrolyte_ntyp = 1       ! 1 salt type (e.g., NaCl -> 1:1 electrolyte)
    ! electrolyte_concentration in mol/L
    ! Multiple ion types can be specified
/
&BOUNDARY
    solvent_mode    = 'electronic'
    rhomax          = 0.005
    rhomin          = 0.0001
/
&ELECTROSTATIC
    pbc_correction  = 'parabolic'
    pbc_dim         = 2
    pbc_axis        = 3
    tol             = 1.0d-11
    problem         = 'linpb'        ! linearized Poisson-Boltzmann
    ! For full PB: problem = 'pb'
/
&ELECTROLYTE
    cion(1)     = 1.0        ! concentration of ion type 1 (mol/L)
    zion(1)     = 1          ! valence of cation
    ! For a 1:1 electrolyte (NaCl), the code automatically
    ! creates the counter-ion with opposite charge
    ! Debye length: lambda_D = sqrt(eps_0*eps_r*kT / (2*c*e^2))
    ! At 1 M, lambda_D ~ 3.0 A
/
"""

with open("environ_lpb.in", "w") as f:
    f.write(environ_lpb)

print("Generated: environ_lpb.in (linearized PB model with electrolyte)")
print()
print("To run with SCCS: cp environ.in environ.in && mpirun pw.x -in pt_solvated.in")
print("To run with LPB:  cp environ_lpb.in environ.in && mpirun pw.x -in pt_solvated.in")

# ============================================================
# 5. Run the solvated calculation (SCCS)
# ============================================================
print("\nRunning solvated SCF with Environ (SCCS)...")

result = subprocess.run(
    ["mpirun", "--allow-run-as-root", "-np", "4", "pw.x", "-in", "pt_solvated.in"],
    capture_output=True, text=True, timeout=1200
)
with open("pt_solvated.out", "w") as f:
    f.write(result.stdout)

# Parse total energy
e_solvated = None
for line in result.stdout.split("\n"):
    if "!" in line and "total energy" in line:
        m = re.search(r"=\s+([-\d.]+)\s+Ry", line)
        if m:
            e_solvated = float(m.group(1)) * 13.6057  # Ry to eV

if e_solvated:
    print(f"E(solvated) = {e_solvated:.6f} eV")
    # Read vacuum energy from previous step if available
    if os.path.exists("pt_vacuum.out"):
        with open("pt_vacuum.out") as f:
            for line in f:
                if "!" in line and "total energy" in line:
                    m = re.search(r"=\s+([-\d.]+)\s+Ry", line)
                    if m:
                        e_vacuum = float(m.group(1)) * 13.6057
                        dG = e_solvated - e_vacuum
                        print(f"E(vacuum)   = {e_vacuum:.6f} eV")
                        print(f"Delta G_solv = {dG:.4f} eV ({dG * 23.061:.2f} kcal/mol)")
else:
    print("Check pt_solvated.out for errors. QE may need Environ plugin compiled in.")
```

#### Step B3: Parse Environ Output and Analyze Solvation Components

```python
#!/usr/bin/env python3
"""
Parse Environ output to extract solvation energy components:
  - Electrostatic solvation (dielectric screening)
  - Cavitation energy (cost of creating the solvent cavity)
  - PV work (pressure-volume, usually negligible)

Also demonstrates computing the solvation energy for a molecule
(e.g., for pKa or solvation free energy benchmarks).
"""
import re
import numpy as np
import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def parse_environ_output(qe_output_file):
    """
    Parse QE output with Environ to extract solvation components.
    Environ prints a summary block at the end of the SCF.
    """
    results = {
        "total_energy_Ry": None,
        "total_energy_eV": None,
        "electrostatic_solv_Ry": None,
        "cavitation_Ry": None,
        "pv_work_Ry": None,
        "total_solv_Ry": None,
        "converged": False,
    }

    with open(qe_output_file) as f:
        text = f.read()

    # Total energy
    for m in re.finditer(r"!\s+total energy\s+=\s+([-\d.]+)\s+Ry", text):
        results["total_energy_Ry"] = float(m.group(1))
        results["total_energy_eV"] = float(m.group(1)) * 13.6057

    # Environ solvation components (look for Environ summary block)
    # Typical Environ output:
    #   Environ Module Summary
    #     electrostatic_energy =   -X.XXXXX Ry
    #     cavitation_energy    =    X.XXXXX Ry
    #     PV_energy            =    X.XXXXX Ry
    for m in re.finditer(r"electrostatic.*?=\s+([-\d.]+)\s*Ry", text, re.IGNORECASE):
        results["electrostatic_solv_Ry"] = float(m.group(1))
    for m in re.finditer(r"cavitation.*?=\s+([-\d.]+)\s*Ry", text, re.IGNORECASE):
        results["cavitation_Ry"] = float(m.group(1))
    for m in re.finditer(r"PV.*?=\s+([-\d.]+)\s*Ry", text, re.IGNORECASE):
        results["pv_work_Ry"] = float(m.group(1))

    if "convergence has been achieved" in text:
        results["converged"] = True

    # Compute total solvation contribution
    solv_parts = [results["electrostatic_solv_Ry"],
                  results["cavitation_Ry"],
                  results["pv_work_Ry"]]
    known_parts = [x for x in solv_parts if x is not None]
    if known_parts:
        results["total_solv_Ry"] = sum(known_parts)

    return results


# ============================================================
# Example: Compare solvated vs vacuum for a set of molecules
# ============================================================
# In practice, you would run vacuum and solvated calculations for each molecule.
# Here we demonstrate with placeholder data for benchmarking.

# Experimental solvation free energies (kcal/mol) -- Minnesota Solvation Database
experimental = {
    "H2O":      -6.31,
    "CH3OH":    -5.10,
    "HCOOH":    -7.71,
    "CH3COOH":  -6.70,
    "NH3":      -4.31,
    "H2S":      -0.70,
    "HCl":      -6.10,
    "acetone":  -3.85,
    "benzene":  -0.87,
    "phenol":   -6.62,
}

# Example computed solvation free energies (kcal/mol)
# (Replace with your actual VASPsol or Environ results)
computed = {
    "H2O":      -7.02,
    "CH3OH":    -5.45,
    "HCOOH":    -8.20,
    "CH3COOH":  -7.15,
    "NH3":      -4.80,
    "H2S":      -1.10,
    "HCl":      -6.70,
    "acetone":  -4.20,
    "benzene":  -1.30,
    "phenol":   -7.10,
}

molecules = list(experimental.keys())
exp_vals = [experimental[m] for m in molecules]
comp_vals = [computed[m] for m in molecules]

# ============================================================
# Parity plot: computed vs experimental solvation energies
# ============================================================
fig, ax = plt.subplots(figsize=(7, 7))

ax.scatter(exp_vals, comp_vals, s=80, color="steelblue", edgecolors="navy",
           linewidths=1.2, zorder=3)

for i, mol in enumerate(molecules):
    ax.annotate(mol, (exp_vals[i], comp_vals[i]),
                textcoords="offset points", xytext=(8, 5), fontsize=9)

# Parity line
lims = [min(min(exp_vals), min(comp_vals)) - 1,
        max(max(exp_vals), max(comp_vals)) + 1]
ax.plot(lims, lims, "k--", linewidth=1, alpha=0.5, label="Parity")
ax.set_xlim(lims)
ax.set_ylim(lims)

# Statistics
errors = np.array(comp_vals) - np.array(exp_vals)
mae = np.mean(np.abs(errors))
rmse = np.sqrt(np.mean(errors ** 2))
mse = np.mean(errors)

ax.set_xlabel(r"Experimental $\Delta G_{solv}$ (kcal/mol)", fontsize=13)
ax.set_ylabel(r"Computed $\Delta G_{solv}$ (kcal/mol)", fontsize=13)
ax.set_title("Implicit Solvation: Computed vs Experiment", fontsize=14)
ax.text(0.05, 0.95, f"MAE = {mae:.2f} kcal/mol\nRMSE = {rmse:.2f}\nMSE = {mse:.2f}",
        transform=ax.transAxes, fontsize=11, verticalalignment="top",
        bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5))
ax.legend(fontsize=11)
ax.grid(alpha=0.3)
ax.set_aspect("equal")

plt.tight_layout()
plt.savefig("solvation_parity.png", dpi=200, bbox_inches="tight")
print("Saved: solvation_parity.png")
print(f"MAE  = {mae:.2f} kcal/mol")
print(f"RMSE = {rmse:.2f} kcal/mol")
print(f"MSE  = {mse:.2f} kcal/mol")
```

### Method C: ASE-Based Solvation Corrections (No DFT Required)

When DFT implicit solvation codes are not available, analytical solvation models provide fast, approximate corrections. These are useful for screening, trend analysis, and adding solvation effects to MACE or force-field calculations.

#### Step C1: Born Solvation Model for Ions

```python
#!/usr/bin/env python3
"""
Born solvation model for ionic solvation free energies.

The Born model treats the ion as a charged sphere in a dielectric continuum:
  Delta G_solv = -(z*e)^2 / (8*pi*eps_0*R) * (1 - 1/eps_r)

This gives the electrostatic contribution to solvation.
Accuracy: ~10-20% for monovalent ions, worse for polyvalent.

Useful for:
  - Quick solvation energy estimates for ions
  - pKa estimation (solvation difference between acid and conjugate base)
  - Screening electrode-electrolyte interface stability
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import json

# ============================================================
# Physical constants
# ============================================================
e_charge = 1.602176634e-19    # C
eps_0 = 8.8541878128e-12      # F/m
k_B = 1.380649e-23            # J/K
N_A = 6.02214076e23           # /mol
cal_to_J = 4.184
eV_to_J = 1.602176634e-19
bohr_to_m = 5.29177210903e-11
angstrom_to_m = 1e-10


def born_solvation_energy(z, R_angstrom, eps_r=78.4):
    """
    Compute Born solvation free energy.

    Parameters
    ----------
    z : int
        Formal charge of the ion.
    R_angstrom : float
        Effective ionic radius in Angstrom.
        Use Shannon crystal radii + ~0.85 A (Born radius correction).
    eps_r : float
        Relative dielectric constant of the solvent (78.4 for water at 298K).

    Returns
    -------
    dG_solv_eV : float
        Solvation free energy in eV.
    dG_solv_kcal : float
        Solvation free energy in kcal/mol.
    """
    R_m = R_angstrom * angstrom_to_m
    # Born equation: dG = -(z*e)^2 / (8*pi*eps_0*R) * (1 - 1/eps_r)
    dG_J = -(z * e_charge) ** 2 / (8 * np.pi * eps_0 * R_m) * (1 - 1 / eps_r)
    dG_J_per_mol = dG_J * N_A
    dG_eV = dG_J / eV_to_J
    dG_kcal = dG_J_per_mol / (cal_to_J * 1000)
    return dG_eV, dG_kcal


# ============================================================
# Shannon crystal radii + Born correction (~0.85 A for cations, ~0.1 A for anions)
# Born radii calibrated to reproduce experimental hydration free energies
# ============================================================
ions = {
    "Li+":  {"z": 1,  "R_shannon": 0.76, "R_born": 1.64, "dG_exp_kcal": -113.5},
    "Na+":  {"z": 1,  "R_shannon": 1.02, "R_born": 1.87, "dG_exp_kcal": -87.2},
    "K+":   {"z": 1,  "R_shannon": 1.38, "R_born": 2.21, "dG_exp_kcal": -70.5},
    "Rb+":  {"z": 1,  "R_shannon": 1.52, "R_born": 2.36, "dG_exp_kcal": -65.7},
    "Cs+":  {"z": 1,  "R_shannon": 1.67, "R_born": 2.53, "dG_exp_kcal": -59.8},
    "Mg2+": {"z": 2,  "R_shannon": 0.72, "R_born": 1.50, "dG_exp_kcal": -437.4},
    "Ca2+": {"z": 2,  "R_shannon": 1.00, "R_born": 1.81, "dG_exp_kcal": -359.6},
    "Zn2+": {"z": 2,  "R_shannon": 0.74, "R_born": 1.54, "dG_exp_kcal": -467.3},
    "F-":   {"z": -1, "R_shannon": 1.33, "R_born": 1.44, "dG_exp_kcal": -111.1},
    "Cl-":  {"z": -1, "R_shannon": 1.81, "R_born": 1.94, "dG_exp_kcal": -81.3},
    "Br-":  {"z": -1, "R_shannon": 1.96, "R_born": 2.10, "dG_exp_kcal": -75.3},
    "I-":   {"z": -1, "R_shannon": 2.20, "R_born": 2.36, "dG_exp_kcal": -65.7},
    "OH-":  {"z": -1, "R_shannon": 1.37, "R_born": 1.50, "dG_exp_kcal": -105.0},
}

print(f"{'Ion':<8} {'z':<4} {'R_Born(A)':<10} {'dG_Born':<14} {'dG_exp':<14} {'Error':<10}")
print(f"{'':>8} {'':>4} {'':>10} {'(kcal/mol)':<14} {'(kcal/mol)':<14} {'(kcal/mol)':<10}")
print("-" * 65)

exp_vals = []
born_vals = []
ion_names = []

for ion_name, data in ions.items():
    dG_eV, dG_kcal = born_solvation_energy(data["z"], data["R_born"])
    error = dG_kcal - data["dG_exp_kcal"]
    print(f"{ion_name:<8} {data['z']:<4} {data['R_born']:<10.2f} {dG_kcal:<14.1f} "
          f"{data['dG_exp_kcal']:<14.1f} {error:<10.1f}")
    exp_vals.append(data["dG_exp_kcal"])
    born_vals.append(dG_kcal)
    ion_names.append(ion_name)

# ============================================================
# Plot: Born model vs experimental solvation energies
# ============================================================
fig, ax = plt.subplots(figsize=(7, 7))

# Color by charge
colors = ["steelblue" if ions[n]["z"] > 0 else "coral" for n in ion_names]
ax.scatter(exp_vals, born_vals, c=colors, s=80, edgecolors="black", linewidths=0.8, zorder=3)

for i, name in enumerate(ion_names):
    ax.annotate(name, (exp_vals[i], born_vals[i]),
                textcoords="offset points", xytext=(6, 4), fontsize=9)

lims = [min(min(exp_vals), min(born_vals)) * 1.1,
        max(max(exp_vals), max(born_vals)) * 0.9]
ax.plot(lims, lims, "k--", linewidth=1, alpha=0.5, label="Parity")
ax.set_xlim(lims)
ax.set_ylim(lims)

mae = np.mean(np.abs(np.array(born_vals) - np.array(exp_vals)))
ax.set_xlabel(r"Experimental $\Delta G_{solv}$ (kcal/mol)", fontsize=13)
ax.set_ylabel(r"Born model $\Delta G_{solv}$ (kcal/mol)", fontsize=13)
ax.set_title(f"Born Solvation Model (MAE = {mae:.1f} kcal/mol)", fontsize=14)
ax.legend(fontsize=11)
ax.grid(alpha=0.3)
ax.set_aspect("equal")

# Legend for colors
from matplotlib.lines import Line2D
legend_elements = [
    Line2D([0], [0], marker="o", color="w", markerfacecolor="steelblue",
           markersize=10, label="Cations"),
    Line2D([0], [0], marker="o", color="w", markerfacecolor="coral",
           markersize=10, label="Anions"),
    Line2D([0], [0], linestyle="--", color="black", label="Parity"),
]
ax.legend(handles=legend_elements, fontsize=11)

plt.tight_layout()
plt.savefig("born_solvation.png", dpi=200, bbox_inches="tight")
print(f"\nSaved: born_solvation.png")
print(f"Overall MAE = {mae:.1f} kcal/mol")
```

#### Step C2: Generalized Born / Surface Area (GBSA) Solvation Corrections

```python
#!/usr/bin/env python3
"""
Generalized Born / Surface Area (GBSA) solvation correction.

Approximates the PCM (Polarizable Continuum Model) by:
  1. Electrostatic part: Generalized Born formula (pairwise screened Coulomb)
  2. Non-electrostatic part: proportional to solvent-accessible surface area (SASA)

Delta G_solv = Delta G_elec (GB) + gamma * SASA

This can be applied as a post-hoc correction to any gas-phase energy
(DFT, MACE, force field).

Useful for:
  - Adding solvation corrections to MACE/FF calculations
  - Screening solvation effects without running DFT solvation
  - Estimating solvation stabilization of reaction intermediates
"""
import numpy as np
from scipy.spatial.distance import pdist, squareform
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import json

# ============================================================
# Physical constants (atomic units -> eV conversions)
# ============================================================
ANGSTROM_TO_BOHR = 1.8897259886
HARTREE_TO_EV = 27.211386245988
HARTREE_TO_KCAL = 627.509474
eV_TO_KCAL = 23.0609

# Dielectric constant
EPS_WATER = 78.4

# Surface tension parameter for SASA term (kcal/mol/A^2)
# Typical values: 0.005-0.010 kcal/mol/A^2
GAMMA_SASA = 0.005  # kcal/mol per A^2

# Van der Waals radii (Angstrom) -- Bondi radii
VDW_RADII = {
    "H": 1.20, "He": 1.40, "C": 1.70, "N": 1.55, "O": 1.52,
    "F": 1.47, "Si": 2.10, "P": 1.80, "S": 1.80, "Cl": 1.75,
    "Br": 1.85, "I": 1.98, "Li": 1.82, "Na": 2.27, "K": 2.75,
    "Cu": 1.40, "Zn": 1.39, "Pt": 1.75, "Pd": 1.63, "Au": 1.66,
    "Fe": 1.56, "Co": 1.52, "Ni": 1.63, "Mn": 1.61, "Ti": 1.60,
}

# Probe radius for SASA (water)
R_PROBE = 1.4  # Angstrom


def compute_sasa(positions, symbols, probe_radius=R_PROBE, n_points=960):
    """
    Compute solvent-accessible surface area using Shrake-Rupley algorithm.

    Parameters
    ----------
    positions : ndarray (N, 3)
        Atomic positions in Angstrom.
    symbols : list of str
        Element symbols.
    probe_radius : float
        Probe radius (water = 1.4 A).
    n_points : int
        Number of test points per atom sphere.

    Returns
    -------
    total_sasa : float
        Total SASA in A^2.
    atom_sasa : ndarray
        SASA per atom in A^2.
    """
    n_atoms = len(positions)

    # Generate uniform points on unit sphere (Fibonacci lattice)
    indices = np.arange(n_points, dtype=float)
    phi = np.arccos(1 - 2 * (indices + 0.5) / n_points)
    theta = np.pi * (1 + np.sqrt(5)) * indices
    unit_points = np.column_stack([
        np.sin(phi) * np.cos(theta),
        np.sin(phi) * np.sin(theta),
        np.cos(phi),
    ])

    radii = np.array([VDW_RADII.get(s, 1.70) + probe_radius for s in symbols])
    atom_sasa = np.zeros(n_atoms)

    for i in range(n_atoms):
        r_i = radii[i]
        test_points = positions[i] + r_i * unit_points  # shape (n_points, 3)

        # Check which test points are buried inside other atoms
        is_exposed = np.ones(n_points, dtype=bool)
        for j in range(n_atoms):
            if j == i:
                continue
            dists = np.linalg.norm(test_points - positions[j], axis=1)
            is_exposed &= (dists > radii[j])

        # SASA for this atom = fraction of exposed points * sphere surface area
        fraction_exposed = np.sum(is_exposed) / n_points
        atom_sasa[i] = 4 * np.pi * r_i ** 2 * fraction_exposed

    return np.sum(atom_sasa), atom_sasa


def generalized_born_energy(positions, charges, symbols, eps_r=EPS_WATER):
    """
    Compute the generalized Born electrostatic solvation energy.

    Delta G_elec = -0.5 * (1 - 1/eps_r) * sum_ij q_i * q_j / f_GB(r_ij)

    where f_GB = sqrt(r_ij^2 + a_i*a_j * exp(-r_ij^2/(4*a_i*a_j)))
    and a_i are the effective Born radii.

    Parameters
    ----------
    positions : ndarray (N, 3)
        Atomic positions in Angstrom.
    charges : ndarray (N,)
        Partial atomic charges in elementary charge units.
    symbols : list of str
        Element symbols.
    eps_r : float
        Dielectric constant of the solvent.

    Returns
    -------
    dG_elec_kcal : float
        GB electrostatic solvation energy in kcal/mol.
    """
    n_atoms = len(positions)

    # Effective Born radii (approximate: use VDW radii scaled)
    # More accurate methods (e.g., ALPB, GBn2) compute these from the molecular surface.
    # Here we use a simple approximation: a_i = VDW_radius_i * scale
    scale = 0.85  # typical scaling factor
    born_radii = np.array([VDW_RADII.get(s, 1.70) * scale for s in symbols])

    # Pairwise distances
    dist_matrix = squareform(pdist(positions))

    # Compute GB energy
    # Using the Still formula:
    # f_GB(i,j) = sqrt(r_ij^2 + a_i*a_j*exp(-r_ij^2/(4*a_i*a_j)))
    # For i=j: f_GB(i,i) = a_i (self term)

    dG_elec = 0.0
    factor = -0.5 * (1.0 - 1.0 / eps_r)

    # Conversion: 332.063714 kcal*A/(mol*e^2) is the Coulomb constant in kcal/mol units
    COULOMB_CONST = 332.063714  # kcal*A/(mol*e^2)

    for i in range(n_atoms):
        for j in range(i, n_atoms):
            ai = born_radii[i]
            aj = born_radii[j]
            qi = charges[i]
            qj = charges[j]

            if i == j:
                f_gb = ai
                weight = 0.5  # self term
            else:
                rij = dist_matrix[i, j]
                f_gb = np.sqrt(rij ** 2 + ai * aj * np.exp(-rij ** 2 / (4 * ai * aj)))
                weight = 1.0

            dG_elec += weight * factor * COULOMB_CONST * qi * qj / f_gb

    return dG_elec


def gbsa_solvation_energy(positions, charges, symbols, eps_r=EPS_WATER,
                          gamma=GAMMA_SASA):
    """
    Full GBSA solvation energy = GB electrostatic + SASA non-polar.

    Returns
    -------
    results : dict
        Contains dG_total, dG_elec, dG_nonpolar, SASA.
    """
    dG_elec = generalized_born_energy(positions, charges, symbols, eps_r)
    sasa_total, sasa_per_atom = compute_sasa(positions, symbols)
    dG_nonpolar = gamma * sasa_total  # kcal/mol

    return {
        "dG_total_kcal": dG_elec + dG_nonpolar,
        "dG_elec_kcal": dG_elec,
        "dG_nonpolar_kcal": dG_nonpolar,
        "SASA_A2": sasa_total,
        "SASA_per_atom_A2": sasa_per_atom.tolist(),
    }


# ============================================================
# Example: Solvation correction for small molecules
# ============================================================
from ase.build import molecule as ase_molecule

test_molecules = {
    "H2O": {"charge_method": "tip3p"},
    "CH3OH": {"charge_method": "gasteiger"},
    "NH3": {"charge_method": "manual"},
}

# For demonstration, use simple partial charges
# In practice, use Mulliken/Lowdin charges from DFT or empirical methods
partial_charges = {
    "H2O": {"charges": [-0.834, 0.417, 0.417]},
    "CH3OH": {"charges": [-0.25, 0.12, 0.12, 0.12, -0.68, 0.42, -0.04]},
    # Adjusted for correct atom count -- actual charges depend on structure
    "NH3": {"charges": [-1.02, 0.34, 0.34, 0.34]},
}

print(f"{'Molecule':<12} {'dG_elec':<12} {'dG_nonpol':<12} {'dG_total':<12} {'SASA':<10}")
print(f"{'':>12} {'(kcal/mol)':<12} {'(kcal/mol)':<12} {'(kcal/mol)':<12} {'(A^2)':<10}")
print("-" * 58)

all_results = {}

for mol_name in ["H2O", "NH3"]:
    atoms = ase_molecule(mol_name)
    atoms.center(vacuum=0)  # no vacuum box needed for GBSA
    positions = atoms.get_positions()
    symbols = list(atoms.get_chemical_symbols())

    charges = np.array(partial_charges[mol_name]["charges"])

    # Ensure charge array matches atoms
    if len(charges) != len(atoms):
        print(f"  {mol_name}: charge array mismatch, skipping")
        continue

    result = gbsa_solvation_energy(positions, charges, symbols)

    print(f"{mol_name:<12} {result['dG_elec_kcal']:<12.2f} "
          f"{result['dG_nonpolar_kcal']:<12.2f} "
          f"{result['dG_total_kcal']:<12.2f} {result['SASA_A2']:<10.1f}")

    all_results[mol_name] = result

with open("gbsa_results.json", "w") as f:
    json.dump(all_results, f, indent=2)
print("\nSaved: gbsa_results.json")
```

#### Step C3: Solvation-Corrected Electrochemical Reaction Energetics

```python
#!/usr/bin/env python3
"""
Apply solvation corrections to electrochemical reaction energetics.

Combines:
  1. Gas-phase DFT or MACE energies
  2. Implicit solvation correction (Born model for ions, GBSA for neutrals)
  3. Electrode potential effect via the computational hydrogen electrode (CHE)

Example: CO2 reduction reaction (CO2RR) on Cu(111)
  CO2 + 2H+ + 2e- -> CO + H2O
  CO2 + 2H+ + 2e- -> HCOOH

Also demonstrates pKa estimation from solvation energies.
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import json


# ============================================================
# 1. Solvation correction database
# ============================================================
# Empirical solvation corrections for common electrochemical intermediates.
# These are derived from implicit solvation calculations on metal surfaces
# (Norskov group, RPBE + VASPsol).
# Units: eV

SOLVATION_CORRECTIONS = {
    # Adsorbate solvation corrections (stabilization relative to vacuum DFT)
    # From: Gauthier et al., ACS Catal. 2019, 9, 920
    "*OH":     -0.50,    # OH* stabilized by ~0.5 eV through H-bonding with water
    "*OOH":    -0.40,    # OOH* similarly stabilized
    "*O":      -0.10,    # O* has weak interaction with water
    "*H":       0.00,    # H* negligible solvation correction
    "*CO":     -0.10,    # CO* weak dipole-water interaction
    "*COOH":   -0.25,    # COOH* moderate H-bonding
    "*CHO":    -0.10,    # CHO* weak interaction
    "*COH":    -0.25,    # COH* moderate H-bonding
    "*OCHO":   -0.20,    # formate
    "*CO2":    -0.10,    # physisorbed CO2
    "*H2O":    -0.20,    # adsorbed water
    # Gas-phase molecules: solvation correction is the full solvation energy
    "H2O(l)":  -0.27,   # liquid water correction (gas -> liquid at 298K, 1 atm)
    "CO2(g)":   0.00,   # gas-phase reference
    "CO(g)":    0.00,
    "H2(g)":    0.00,
    "HCOOH(aq)": -0.35, # formic acid in solution
}

# ============================================================
# 2. Computational Hydrogen Electrode (CHE) framework
# ============================================================
# At the SHE:  G(H+ + e-) = 0.5 * G(H2) at U=0 V
# At potential U:  G(H+ + e-) = 0.5 * G(H2) - eU
# pH correction: G(H+ + e-) = 0.5 * G(H2) - eU + kT*ln(10)*pH

kT_298 = 0.02569  # eV at 298 K


def che_correction(n_electrons, U_SHE, pH=0):
    """
    Computational hydrogen electrode correction.

    Parameters
    ----------
    n_electrons : int
        Number of (H+ + e-) transfers in the step.
    U_SHE : float
        Electrode potential vs SHE in V.
    pH : float
        pH of the solution.

    Returns
    -------
    dG_correction : float
        Free energy correction in eV.
    """
    return n_electrons * (-U_SHE + kT_298 * np.log(10) * pH)


# ============================================================
# 3. CO2RR free energy diagram on Cu(111)
# ============================================================
# Pathway: CO2 -> *COOH -> *CO -> CO(g)
# Each step: CO2 + H+ + e- -> *COOH (proton-coupled electron transfer)
#            *COOH + H+ + e- -> *CO + H2O
#            *CO -> CO(g) + *

# DFT adsorption energies (eV) -- example values for Cu(111)
# These would come from your MACE or QE calculations
dE_COOH = 0.55     # E(*COOH) - E(*) - E(CO2) - 0.5*E(H2) in vacuum
dE_CO = -0.35      # E(*CO) - E(*) - E(CO2) + E(H2O) - E(H2) in vacuum

# Apply solvation corrections
dG_COOH_solv = dE_COOH + SOLVATION_CORRECTIONS["*COOH"]
dG_CO_solv = dE_CO + SOLVATION_CORRECTIONS["*CO"]

# ZPE + thermal corrections (from thermal-corrections skill)
dG_ZPE_COOH = 0.60     # ZPE + TS correction for *COOH formation
dG_ZPE_CO = 0.35       # ZPE + TS correction for *CO formation

# Total free energies at U=0
dG_COOH_total = dE_COOH + SOLVATION_CORRECTIONS["*COOH"] + dG_ZPE_COOH
dG_CO_total = dE_CO + SOLVATION_CORRECTIONS["*CO"] + dG_ZPE_CO

print("CO2RR on Cu(111): CO2 -> *COOH -> *CO -> CO")
print(f"  dE(*COOH)_vac     = {dE_COOH:.3f} eV")
print(f"  + solvation corr  = {SOLVATION_CORRECTIONS['*COOH']:.3f} eV")
print(f"  + ZPE/thermal     = {dG_ZPE_COOH:.3f} eV")
print(f"  dG(*COOH)_total   = {dG_COOH_total:.3f} eV")
print()
print(f"  dE(*CO)_vac       = {dE_CO:.3f} eV")
print(f"  + solvation corr  = {SOLVATION_CORRECTIONS['*CO']:.3f} eV")
print(f"  + ZPE/thermal     = {dG_ZPE_CO:.3f} eV")
print(f"  dG(*CO)_total     = {dG_CO_total:.3f} eV")

# ============================================================
# 4. Free energy diagram at different potentials
# ============================================================
# Step free energies:
# Step 1: CO2 + H+ + e- -> *COOH
#   dG1 = dG_COOH_total + che_correction(1, U)
# Step 2: *COOH + H+ + e- -> *CO + H2O
#   dG2 = dG_CO_total - dG_COOH_total + che_correction(1, U) + G(H2O_corr)
# Step 3: *CO -> CO(g)
#   dG3 = -dG_CO_total (desorption, no electron transfer)

G_H2O_corr = SOLVATION_CORRECTIONS["H2O(l)"]  # gas -> liquid correction

fig, ax = plt.subplots(figsize=(10, 6))

potentials = [0.0, -0.5, -1.0, -1.17]  # V vs SHE
colors = ["#e74c3c", "#f39c12", "#3498db", "#2ecc71"]

state_labels = [
    r"CO$_2$(g) + 2(H$^+$+e$^-$)",
    r"*COOH + (H$^+$+e$^-$)",
    r"*CO + H$_2$O",
    r"CO(g) + H$_2$O + *",
]

for U, color in zip(potentials, colors):
    dG1 = dG_COOH_total + che_correction(1, U)
    dG2 = (dG_CO_total - dG_COOH_total) + che_correction(1, U) + G_H2O_corr
    dG3 = -dG_CO_total  # desorption, no e- transfer

    levels = [0.0]
    for dg in [dG1, dG2, dG3]:
        levels.append(levels[-1] + dg)

    x_pos = np.arange(len(levels))
    w = 0.3
    for i in range(len(levels)):
        ax.plot([x_pos[i] - w, x_pos[i] + w], [levels[i], levels[i]],
                color=color, linewidth=2.5)
        if i < len(levels) - 1:
            ax.plot([x_pos[i] + w, x_pos[i + 1] - w],
                    [levels[i], levels[i + 1]],
                    color=color, linewidth=1, linestyle="--", alpha=0.4)
    ax.plot([], [], color=color, linewidth=2.5, label=f"U = {U:.2f} V vs SHE")

ax.set_xticks(x_pos)
ax.set_xticklabels(state_labels, fontsize=10, rotation=10, ha="right")
ax.set_ylabel("Free Energy (eV)", fontsize=13)
ax.set_title(r"CO$_2$RR Free Energy Diagram on Cu(111) (with solvation)", fontsize=14)
ax.axhline(0, color="gray", linewidth=0.5)
ax.legend(fontsize=10, loc="upper right")
ax.grid(axis="y", alpha=0.3)

plt.tight_layout()
plt.savefig("co2rr_free_energy.png", dpi=200, bbox_inches="tight")
print("\nSaved: co2rr_free_energy.png")

# Limiting potential: the most positive U at which all steps are downhill
# Step 1 is potential-determining: dG1 = dG_COOH_total - eU = 0
U_lim = -dG_COOH_total
print(f"\nLimiting potential = {U_lim:.3f} V vs SHE")
print(f"Overpotential (vs -0.11 V thermodynamic) = {abs(U_lim) - 0.11:.3f} V")

# ============================================================
# 5. pKa estimation from solvation energies
# ============================================================
print("\n" + "=" * 60)
print("pKa Estimation from Solvation Energies")
print("=" * 60)

# pKa = dG_deprot / (kT * ln(10))
# dG_deprot = [G(A-,aq) + G(H+,aq)] - G(HA,aq)
# G(X,aq) = G(X,gas) + dG_solv(X)

# Experimental: G_solv(H+) = -265.9 kcal/mol (absolute, Tissandier convention)
# Or use: dG_solv(H+) = -11.53 eV (absolute)
dG_solv_Hplus = -11.53  # eV (absolute proton solvation free energy)

# Example: acetic acid (CH3COOH -> CH3COO- + H+)
# Gas-phase deprotonation energy (from DFT or experiment)
dG_deprot_gas = 15.12    # eV (gas-phase, including ZPE)
dG_solv_HA = -0.29       # eV (solvation of CH3COOH -- from implicit solv calc)
dG_solv_Aminus = -3.25   # eV (solvation of CH3COO- -- from implicit solv calc)

dG_deprot_aq = dG_deprot_gas + dG_solv_Aminus + dG_solv_Hplus - dG_solv_HA
pKa_computed = dG_deprot_aq / (kT_298 * np.log(10))

print(f"Acetic acid:")
print(f"  dG_deprot(gas) = {dG_deprot_gas:.2f} eV")
print(f"  dG_solv(HA)    = {dG_solv_HA:.2f} eV")
print(f"  dG_solv(A-)    = {dG_solv_Aminus:.2f} eV")
print(f"  dG_solv(H+)    = {dG_solv_Hplus:.2f} eV")
print(f"  dG_deprot(aq)  = {dG_deprot_aq:.4f} eV")
print(f"  pKa (computed) = {pKa_computed:.1f}")
print(f"  pKa (expt)     = 4.76")

# ============================================================
# 6. Save all results
# ============================================================
results = {
    "co2rr_cu111": {
        "dG_COOH_vac_eV": dE_COOH,
        "dG_CO_vac_eV": dE_CO,
        "solvation_corrections_eV": {
            k: v for k, v in SOLVATION_CORRECTIONS.items()
        },
        "dG_COOH_total_eV": dG_COOH_total,
        "dG_CO_total_eV": dG_CO_total,
        "limiting_potential_V": U_lim,
    },
    "pka_acetic_acid": {
        "dG_deprot_gas_eV": dG_deprot_gas,
        "dG_solv_HA_eV": dG_solv_HA,
        "dG_solv_Aminus_eV": dG_solv_Aminus,
        "pKa_computed": pKa_computed,
        "pKa_experimental": 4.76,
    },
}

with open("solvation_electrochem_results.json", "w") as f:
    json.dump(results, f, indent=2)
print("\nSaved: solvation_electrochem_results.json")
```

#### Step C4: Complete Solvation Workflow -- MACE Energy + Solvation Correction

```python
#!/usr/bin/env python3
"""
Complete workflow: compute gas-phase energy with MACE,
then add solvation correction using analytical models.

This is a fast screening approach that avoids DFT entirely.
Suitable for:
  - Catalyst screening across many surfaces
  - Approximate solvation trends
  - Pre-screening before expensive DFT+solvation calculations

Example: Compute adsorption energies of OH* on several metal surfaces
with and without solvation correction.
"""
import warnings
warnings.filterwarnings("ignore")

import numpy as np
from ase.build import fcc111, add_adsorbate, molecule
from ase.optimize import BFGS
from ase.constraints import FixAtoms
from mace.calculators import mace_mp
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import json

# ============================================================
# 1. Configuration
# ============================================================
calc = mace_mp(model="medium", dispersion=False, default_dtype="float64")

metals = ["Cu", "Ag", "Au", "Pd", "Pt", "Ni"]
lattice_constants = {
    "Cu": 3.615, "Ag": 4.085, "Au": 4.078,
    "Pd": 3.890, "Pt": 3.924, "Ni": 3.524,
}

# Solvation correction for OH* (eV)
# From Norskov group benchmarks: OH* is stabilized by ~0.50 eV in water
SOLV_CORR_OH = -0.50
SOLV_CORR_SLAB = 0.00  # clean slab has negligible solvation effect

results = {}

# ============================================================
# 2. Reference energies
# ============================================================
# H2O gas phase
h2o = molecule("H2O")
h2o.center(vacuum=10.0)
h2o.calc = calc
opt = BFGS(h2o, logfile="/dev/null")
opt.run(fmax=0.01, steps=100)
E_H2O = h2o.get_potential_energy()

# H2 gas phase
h2 = molecule("H2")
h2.center(vacuum=10.0)
h2.calc = calc
opt = BFGS(h2, logfile="/dev/null")
opt.run(fmax=0.01, steps=100)
E_H2 = h2.get_potential_energy()

print(f"Reference energies (MACE):")
print(f"  E(H2O) = {E_H2O:.4f} eV")
print(f"  E(H2)  = {E_H2:.4f} eV")

# ============================================================
# 3. Compute OH* adsorption energy on each metal
# ============================================================
print(f"\n{'Metal':<8} {'E_slab':<12} {'E_OH+slab':<12} {'dE_OH_vac':<12} "
      f"{'dE_OH_solv':<12} {'diff':<8}")
print("-" * 65)

for metal in metals:
    a = lattice_constants[metal]

    # Build clean slab
    slab = fcc111(metal, size=(3, 3, 4), a=a, vacuum=15.0)
    n_slab = len(slab)

    # Fix bottom 2 layers
    z = slab.get_positions()[:, 2]
    z_layers = sorted(set(np.round(z, 1)))
    fix_z = z_layers[1] + 0.5
    slab.set_constraint(FixAtoms(mask=z < fix_z))

    slab.calc = calc
    opt = BFGS(slab, logfile="/dev/null")
    opt.run(fmax=0.02, steps=200)
    E_slab = slab.get_potential_energy()

    # Add OH adsorbate
    oh_slab = slab.copy()
    add_adsorbate(oh_slab, "O", height=2.0, position="fcc")
    # Add H to the O
    o_idx = len(oh_slab) - 1
    o_pos = oh_slab.get_positions()[o_idx]
    from ase import Atoms
    h_atom = Atoms("H", positions=[o_pos + [0, 0, 0.97]])
    oh_slab = oh_slab + h_atom

    # Fix bottom 2 layers (same as clean slab)
    z2 = oh_slab.get_positions()[:, 2]
    fix_mask2 = np.zeros(len(oh_slab), dtype=bool)
    fix_mask2[:n_slab] = z2[:n_slab] < fix_z
    oh_slab.set_constraint(FixAtoms(mask=fix_mask2))

    oh_slab.calc = calc
    opt = BFGS(oh_slab, logfile="/dev/null")
    opt.run(fmax=0.02, steps=200)
    E_OH_slab = oh_slab.get_potential_energy()

    # Adsorption energy: dE_OH = E(OH+slab) - E(slab) - E(H2O) + 0.5*E(H2)
    dE_OH_vac = E_OH_slab - E_slab - E_H2O + 0.5 * E_H2
    dE_OH_solv = dE_OH_vac + SOLV_CORR_OH - SOLV_CORR_SLAB
    diff = SOLV_CORR_OH

    print(f"{metal:<8} {E_slab:<12.4f} {E_OH_slab:<12.4f} {dE_OH_vac:<12.4f} "
          f"{dE_OH_solv:<12.4f} {diff:<8.2f}")

    results[metal] = {
        "E_slab_eV": float(E_slab),
        "E_OH_slab_eV": float(E_OH_slab),
        "dE_OH_vac_eV": float(dE_OH_vac),
        "dE_OH_solv_eV": float(dE_OH_solv),
        "solvation_correction_eV": float(SOLV_CORR_OH),
    }

# ============================================================
# 4. Volcano plot: activity vs OH* binding energy
# ============================================================
fig, ax = plt.subplots(figsize=(9, 6))

dE_vac = [results[m]["dE_OH_vac_eV"] for m in metals]
dE_solv = [results[m]["dE_OH_solv_eV"] for m in metals]

# Activity descriptor: limiting potential (simplified)
# For ORR: eta ~ |dG_OH - 1.23| (Sabatier principle)
optimal_dG_OH = 0.0  # ideal binding for ORR

eta_vac = [abs(dE - optimal_dG_OH) for dE in dE_vac]
eta_solv = [abs(dE - optimal_dG_OH) for dE in dE_solv]

ax.scatter(dE_vac, eta_vac, s=100, color="coral", edgecolors="darkred",
           linewidths=1, zorder=3, label="Vacuum")
ax.scatter(dE_solv, eta_solv, s=100, color="steelblue", edgecolors="navy",
           linewidths=1, zorder=3, label="With solvation")

for i, metal in enumerate(metals):
    ax.annotate(metal, (dE_vac[i], eta_vac[i]),
                textcoords="offset points", xytext=(8, 5), fontsize=11,
                color="darkred")
    ax.annotate(metal, (dE_solv[i], eta_solv[i]),
                textcoords="offset points", xytext=(8, -12), fontsize=11,
                color="navy")

# Draw arrows showing the solvation shift
for i in range(len(metals)):
    ax.annotate("", xy=(dE_solv[i], eta_solv[i]),
                xytext=(dE_vac[i], eta_vac[i]),
                arrowprops=dict(arrowstyle="->", color="gray", alpha=0.5, lw=1))

ax.set_xlabel(r"$\Delta E_{OH*}$ (eV)", fontsize=13)
ax.set_ylabel(r"$|\Delta G_{OH*}|$ ~ Overpotential (eV)", fontsize=13)
ax.set_title("Effect of Solvation on ORR Activity Descriptor", fontsize=14)
ax.legend(fontsize=11)
ax.grid(alpha=0.3)
ax.axvline(optimal_dG_OH, color="green", linestyle=":", alpha=0.5, label="Optimal binding")

plt.tight_layout()
plt.savefig("solvation_volcano.png", dpi=200, bbox_inches="tight")
print(f"\nSaved: solvation_volcano.png")

with open("mace_solvation_screening.json", "w") as f:
    json.dump(results, f, indent=2)
print("Saved: mace_solvation_screening.json")
```

---

## Method Comparison Table

| Feature | VASPsol | QE + Environ | ASE Born/GBSA | Empirical Correction |
|---|---|---|---|---|
| **Theory** | Linearized PB + cavity | SCCS or PB + cavity | Born sphere / GB pairwise | Tabulated values |
| **Self-consistent** | Yes | Yes | No | No |
| **Dielectric cavity** | Electron-density based | Electron-density based | VDW radii | N/A |
| **Electrolyte (ions)** | Yes (Debye screening) | Yes (PB electrolyte) | Born model only | No |
| **Surface charge** | Yes (NELECT) | Yes (tot_charge) | No | No |
| **Accuracy (neutrals)** | MAE ~1-2 kcal/mol | MAE ~1-2 kcal/mol | MAE ~3-5 kcal/mol | MAE ~0.1-0.3 eV |
| **Accuracy (ions)** | MAE ~3-5 kcal/mol | MAE ~3-5 kcal/mol | MAE ~10-20 kcal/mol | N/A |
| **Cost** | DFT + 10-20% | DFT + 10-20% | Seconds | Zero |
| **License** | Commercial (VASP) | Open source | Open source | Open source |
| **Best for** | Metal surfaces, electrochemistry | Metal surfaces, electrochemistry | Ion solvation screening | Reaction energetics |

## Key Parameters

### VASPsol Parameters

| Parameter | Default | Description |
|---|---|---|
| `LSOL` | `.FALSE.` | Activate implicit solvation |
| `EB_K` | 78.4 | Bulk dielectric constant (78.4 for water at 298K) |
| `TAU` | 0.000525 | Cavity surface tension (eV/A^2) |
| `LAMBDA_D_K` | -- | Debye screening length (A). Set for electrolyte. |
| `NC_K` | 0.0025 | Critical electron density for cavity (e/A^3) |
| `SIGMA_K` | 0.6 | Width of dielectric transition (A) |

### Environ Parameters

| Parameter | Default | Description |
|---|---|---|
| `env_static_permittivity` | 78.3 | Bulk dielectric constant |
| `env_surface_tension` | 50.0d-3 | Surface tension (Ry/bohr^2) |
| `rhomax` | 0.005 | Max density for full solvation (e/bohr^3) |
| `rhomin` | 0.0001 | Min density for no solvation (e/bohr^3) |
| `problem` | `generalized` | `linpb` for linearized PB, `pb` for full PB |
| `cion(i)` | -- | Electrolyte concentration (mol/L) |
| `zion(i)` | -- | Ion valence |
| `pbc_correction` | `parabolic` | PBC correction for slabs |

### Empirical Solvation Corrections for Common Adsorbates

| Adsorbate | Solvation Correction (eV) | Source |
|---|---|---|
| *OH | -0.50 | Norskov group, VASPsol benchmarks |
| *OOH | -0.40 | Norskov group |
| *O | -0.10 | Norskov group |
| *H | 0.00 | Negligible |
| *CO | -0.10 | Weak dipole interaction |
| *COOH | -0.25 | Moderate H-bonding |
| *CHO | -0.10 | Weak |

## Interpreting Results

1. **Solvation energy sign**: Negative means stabilization (favorable solvation). Most species are stabilized in water.
2. **Magnitude**: Neutral molecules: -0.1 to -0.5 eV. Ions: -1 to -10 eV. The larger the charge or dipole, the stronger the solvation.
3. **Effect on catalysis**: Solvation preferentially stabilizes polar intermediates (*OH, *OOH), shifting the volcano plot and changing the predicted optimal catalyst.
4. **Surface charge**: In electrochemistry, the electrode is at a controlled potential, meaning the surface has a net charge. Implicit solvation naturally handles the counter-charge in the electrolyte (Debye screening).
5. **pKa estimation**: Requires accurate solvation of both neutral acid and charged conjugate base. Born/GBSA is often too crude; use VASPsol or Environ for quantitative pKa.
6. **Double layer**: The linearized PB model gives the diffuse double layer profile. For the compact (Helmholtz) layer, explicit water molecules are needed.

## Common Issues

| Issue | Solution |
|---|---|
| Solvation energy too large / unphysical | Check cavity parameters. Too small `NC_K` or `rhomin` can create cavities inside the slab. |
| SCF convergence issues with solvation | Reduce `mixing_beta` (QE) or use `ALGO=Normal` (VASP). Solvation adds nonlinearity. |
| Asymmetric slab gives spurious dipole | Use dipole correction: `LDIPOL=.TRUE., IDIPOL=3` (VASP) or `pbc_correction=parabolic` (Environ). |
| Electrolyte concentration has no effect | Check Debye length. At very low concentration, screening is negligible. At very high concentration, the linearized PB may break down. |
| SASA-only model gives poor ionic solvation | SASA captures non-polar solvation only. For ions, the electrostatic (Born/GB) term dominates. Always include both. |
| VASPsol not recognized by VASP | VASP must be recompiled with the VASPsol patch. Check that `LSOL` appears in OUTCAR header. |
| Environ not active in QE | QE must be compiled with `--with-environ`. Check for Environ banner in QE output. |
| Born model overestimates multivalent ion solvation | Born radii need careful calibration. Consider using Shannon radii + empirical offset. |
| Solvation correction changes sign of adsorption energy | This is physical: some weakly bound species may desorb in solvent. Verify by comparing vacuum and solvated binding energies. |
| Different solvation codes give different results | Expected: SCCS (Environ) and VASPsol use different cavity definitions. Compare on the same test set to quantify systematic offsets. Typical difference: 0.05-0.15 eV for neutrals. |
