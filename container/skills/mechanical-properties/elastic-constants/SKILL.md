# Elastic Constants Calculation

## When to Use

- You need the full 6x6 elastic stiffness tensor (C_ij) of a crystalline material.
- You want polycrystalline aggregate moduli: bulk modulus (K), shear modulus (G), Young's modulus (E), Poisson's ratio (nu).
- You need to check mechanical (Born) stability of a crystal structure.
- You want stress-strain curves under specific deformation modes.

## Method Selection (MACE vs QE)

| Criterion | MACE (ASE) | QE (DFT) |
|---|---|---|
| Speed | Seconds | Hours to days |
| Accuracy | Good for structures similar to MACE training data | Systematically improvable, publication quality |
| Use when | Screening, rapid prototyping, large-scale studies | Publication, unusual chemistry, validating MACE |

Use MACE first for a quick estimate. If the material is exotic or the elastic constants look suspicious, follow up with QE.

## Prerequisites

- A relaxed crystal structure (CIF, POSCAR, or pymatgen Structure). The structure must be at or near equilibrium (zero stress) before applying strain deformations.
- For QE: pseudopotential files (SSSP recommended). See `electronic-structure/scf-relax/SKILL.md` for setup.
- Python packages: `pymatgen`, `ase`, `mace-torch`, `numpy`, `scipy`, `matplotlib` (pre-installed).

## Detailed Steps

### Method A: ASE + MACE

This method applies a set of symmetry-adapted strain deformations to the relaxed structure, computes the stress tensor at each deformed configuration using the MACE calculator, and fits the full elastic tensor. The workflow mirrors atomate2's `ElasticMaker`: (1) relax bulk, (2) generate deformations from strain states with symmetry reduction, (3) apply each deformation and relax ions at fixed cell, (4) collect stresses, (5) fit the elastic tensor.

```python
#!/usr/bin/env python3
"""
Elastic constants calculation using ASE + MACE.

Workflow (following atomate2 ElasticMaker pattern):
  1. Relax the structure (full cell + ions).
  2. Generate symmetry-reduced strain deformations.
  3. For each deformation: apply strain, relax ions (fixed cell), record stress.
  4. Fit the 6x6 elastic tensor from stress-strain data.
  5. Compute Voigt-Reuss-Hill polycrystalline moduli.
  6. Check Born stability criteria.
"""

import json
import warnings
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from ase.io import read as ase_read, write as ase_write
from ase.optimize import LBFGS
from ase.constraints import ExpCellFilter

from pymatgen.core.structure import Structure
from pymatgen.core.tensors import symmetry_reduce
from pymatgen.analysis.elasticity import (
    Strain, Stress, Deformation, ElasticTensor
)
from pymatgen.io.ase import AseAtomsAdaptor
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer

# ─── CONFIGURATION ───────────────────────────────────────────────────────────
INPUT_FILE = "structure.cif"           # Input structure (CIF, POSCAR, etc.)
MACE_MODEL = "medium"                  # MACE model: "small", "medium", "large"
STRAIN_MAGNITUDES = [-0.01, -0.005, 0.005, 0.01]  # Strain amplitudes
SYM_REDUCE = True                      # Use symmetry to reduce deformation count
SYMPREC = 0.01                         # Symmetry precision (Angstrom)
FMAX_BULK = 1e-4                       # Force convergence for bulk relaxation (eV/A)
FMAX_DEFORM = 1e-3                     # Force convergence for deformed cells (eV/A)
OUTPUT_DIR = "elastic_results"
# ─────────────────────────────────────────────────────────────────────────────

import os
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ─── STEP 1: Load structure and set up MACE calculator ───────────────────────
from mace.calculators import mace_mp
calc = mace_mp(model=MACE_MODEL, default_dtype="float64")

adaptor = AseAtomsAdaptor()

# Load structure via pymatgen (handles CIF, POSCAR, etc.)
structure = Structure.from_file(INPUT_FILE)
print(f"Loaded structure: {structure.composition.reduced_formula}")
print(f"Space group: {SpacegroupAnalyzer(structure, symprec=SYMPREC).get_space_group_symbol()}")

# ─── STEP 2: Relax the bulk structure (cell + ions) ─────────────────────────
atoms = adaptor.get_atoms(structure)
atoms.calc = calc

ecf = ExpCellFilter(atoms, hydrostatic_strain=False)
opt = LBFGS(ecf, logfile=os.path.join(OUTPUT_DIR, "bulk_relax.log"))
opt.run(fmax=FMAX_BULK, steps=500)

relaxed_structure = adaptor.get_structure(atoms)
relaxed_structure.to(os.path.join(OUTPUT_DIR, "relaxed_structure.cif"))
print(f"Relaxed volume: {relaxed_structure.volume:.4f} A^3")

# Get equilibrium stress (should be near zero after relaxation)
equilibrium_stress_voigt = atoms.get_stress(voigt=True)  # eV/A^3, Voigt order
# Convert from eV/A^3 to kBar for reference (1 eV/A^3 = 1602.1766 kBar)
print(f"Equilibrium stress (eV/A^3, Voigt): {equilibrium_stress_voigt}")

# ─── STEP 3: Generate strain deformations ────────────────────────────────────
# Default 2nd-order strain states in Voigt notation (same as atomate2):
#   (1,0,0,0,0,0) = e_xx
#   (0,1,0,0,0,0) = e_yy
#   (0,0,1,0,0,0) = e_zz
#   (0,0,0,2,0,0) = e_yz (factor 2 for engineering shear strain)
#   (0,0,0,0,2,0) = e_xz
#   (0,0,0,0,0,2) = e_xy

strain_states = [
    (1, 0, 0, 0, 0, 0),
    (0, 1, 0, 0, 0, 0),
    (0, 0, 1, 0, 0, 0),
    (0, 0, 0, 2, 0, 0),
    (0, 0, 0, 0, 2, 0),
    (0, 0, 0, 0, 0, 2),
]

strains = []
for state in strain_states:
    for mag in STRAIN_MAGNITUDES:
        strains.append(Strain.from_voigt(mag * np.array(state)))

# Remove any zero strains
strains = [s for s in strains if (np.abs(s) > 1e-10).any()]

print(f"Total strain deformations before symmetry reduction: {len(strains)}")

# Check rank sufficiency
if np.linalg.matrix_rank([s.voigt for s in strains]) < 6:
    raise ValueError("Strain list is insufficient to fit an elastic tensor.")

# Apply symmetry reduction (following atomate2 pattern)
if SYM_REDUCE:
    strain_mapping = symmetry_reduce(strains, relaxed_structure, symprec=SYMPREC)
    strains = list(strain_mapping.keys())
    print(f"After symmetry reduction: {len(strains)} independent deformations")

deformations = [s.get_deformation_matrix() for s in strains]

# ─── STEP 4: Apply deformations and compute stresses ────────────────────────
stresses_list = []
deformations_list = []
strain_data_for_plot = []  # For visualization

for idx, deformation in enumerate(deformations):
    deform_matrix = np.array(deformation)

    # Apply deformation to lattice
    deformed_structure = relaxed_structure.copy()
    deformed_structure.apply_strain(
        Strain.from_deformation(Deformation(deform_matrix))
    )

    # Convert to ASE and relax ions only (fixed cell)
    deformed_atoms = adaptor.get_atoms(deformed_structure)
    deformed_atoms.calc = calc

    opt = LBFGS(deformed_atoms, logfile=os.devnull)
    opt.run(fmax=FMAX_DEFORM, steps=300)

    # Get stress: ASE returns stress in eV/A^3, Voigt convention [xx,yy,zz,yz,xz,xy]
    # pymatgen Stress expects 3x3 in kBar. Convert.
    stress_voigt_ev = deformed_atoms.get_stress(voigt=False)  # 3x3, eV/A^3
    # Convert eV/A^3 to GPa (1 eV/A^3 = 160.21766 GPa)
    stress_3x3_gpa = stress_voigt_ev * 160.21766
    # ASE uses the physics sign convention: positive = tensile.
    # pymatgen Stress expects the same sign convention for ElasticTensor.from_stress_dict.
    # However, for the fit we need to be careful: ASE stress has opposite sign from VASP.
    # ASE: sigma_ij = (1/V) * dE/d(epsilon_ij) with compression = negative
    # We negate to match the convention used by pymatgen ElasticTensor fitting
    # (which expects stress in the standard mechanics convention).
    stress_obj = Stress(-stress_3x3_gpa)

    stresses_list.append(stress_obj)
    deformations_list.append(Deformation(deform_matrix))

    strain_voigt = Strain.from_deformation(Deformation(deform_matrix)).voigt
    strain_data_for_plot.append({
        "strain_voigt": strain_voigt.tolist(),
        "stress_voigt_GPa": stress_obj.voigt.tolist(),
        "deformation": deform_matrix.tolist(),
    })

    print(f"  Deformation {idx+1}/{len(deformations)}: "
          f"strain_norm={np.linalg.norm(strain_voigt):.5f}, "
          f"max|stress|={np.max(np.abs(stress_obj.voigt)):.4f} GPa")

# ─── STEP 5: Fit the elastic tensor ─────────────────────────────────────────
# Subtract equilibrium stress if nonzero
eq_stress_3x3_gpa = Stress(
    -np.array(atoms.get_stress(voigt=False)) * 160.21766
)

elastic_tensor = ElasticTensor.from_independent_strains(
    stresses_list, strains, eq_stress=eq_stress_3x3_gpa
)

# Symmetrize the elastic tensor using the crystal symmetry
sga = SpacegroupAnalyzer(relaxed_structure, symprec=SYMPREC)
symmops = sga.get_symmetry_operations(cartesian=True)

# Average the tensor over symmetry operations for cleaner results
C_raw = elastic_tensor.voigt
C_symmetrized = np.zeros((6, 6))
for op in symmops:
    rotated = elastic_tensor.transform(op.rotation_matrix)
    C_symmetrized += rotated.voigt
C_symmetrized /= len(symmops)
C_symmetrized = ElasticTensor.from_voigt(C_symmetrized)

print("\n" + "="*60)
print("ELASTIC TENSOR (Voigt notation, GPa):")
print("="*60)
C_matrix = C_symmetrized.voigt
for i in range(6):
    row = "  ".join(f"{C_matrix[i,j]:8.2f}" for j in range(6))
    print(f"  [{row}]")

# ─── STEP 6: Voigt-Reuss-Hill polycrystalline moduli ────────────────────────
K_voigt = C_symmetrized.k_voigt
K_reuss = C_symmetrized.k_reuss
K_vrh = C_symmetrized.k_vrh

G_voigt = C_symmetrized.g_voigt
G_reuss = C_symmetrized.g_reuss
G_vrh = C_symmetrized.g_vrh

# Young's modulus and Poisson's ratio from VRH averages
E_vrh = 9 * K_vrh * G_vrh / (3 * K_vrh + G_vrh) if (3*K_vrh + G_vrh) > 0 else 0.0
nu_vrh = (3 * K_vrh - 2 * G_vrh) / (6 * K_vrh + 2 * G_vrh) if (6*K_vrh + 2*G_vrh) > 0 else 0.0

print(f"\nPolycrystalline Moduli (GPa):")
print(f"  Bulk  modulus: K_Voigt={K_voigt:.2f}, K_Reuss={K_reuss:.2f}, K_VRH={K_vrh:.2f}")
print(f"  Shear modulus: G_Voigt={G_voigt:.2f}, G_Reuss={G_reuss:.2f}, G_VRH={G_vrh:.2f}")
print(f"  Young's modulus (VRH): E = {E_vrh:.2f} GPa")
print(f"  Poisson's ratio (VRH): nu = {nu_vrh:.4f}")
print(f"  Pugh's ratio: K/G = {K_vrh/G_vrh:.3f}" if G_vrh > 0 else "  Pugh's ratio: undefined (G=0)")

# ─── STEP 7: Born mechanical stability check ────────────────────────────────
# For a general crystal, the elastic tensor must be positive definite.
eigenvalues = np.linalg.eigvalsh(C_matrix)
is_stable = all(ev > 0 for ev in eigenvalues)

print(f"\nBorn Stability Check:")
print(f"  Eigenvalues of C_ij: {', '.join(f'{ev:.2f}' for ev in eigenvalues)}")
print(f"  Mechanically stable: {'YES' if is_stable else 'NO -- UNSTABLE'}")

# Additional cubic stability criteria (if applicable)
sga_info = sga.get_crystal_system()
print(f"  Crystal system: {sga_info}")
if sga_info == "cubic":
    c11, c12, c44 = C_matrix[0, 0], C_matrix[0, 1], C_matrix[3, 3]
    cubic_stable = (c11 > 0) and (c44 > 0) and (c11 > abs(c12)) and (c11 + 2*c12 > 0)
    print(f"  Cubic criteria: C11={c11:.1f}>0, C44={c44:.1f}>0, "
          f"C11>|C12|: {c11:.1f}>{abs(c12):.1f}, C11+2C12={c11+2*c12:.1f}>0 --> "
          f"{'PASS' if cubic_stable else 'FAIL'}")
    # Zener anisotropy ratio
    A_zener = 2 * c44 / (c11 - c12) if (c11 - c12) != 0 else float('inf')
    print(f"  Zener anisotropy ratio: A = {A_zener:.3f} (1.0 = isotropic)")

# ─── STEP 8: Visualization ──────────────────────────────────────────────────
fig, axes = plt.subplots(2, 3, figsize=(15, 10))
voigt_labels = ["xx", "yy", "zz", "yz", "xz", "xy"]

for i, ax in enumerate(axes.flat):
    strains_i = []
    stresses_i = []
    for data in strain_data_for_plot:
        sv = data["strain_voigt"]
        # Find deformations where this strain component is dominant
        dominant_idx = np.argmax(np.abs(sv))
        if dominant_idx == i:
            strains_i.append(sv[i])
            stresses_i.append(data["stress_voigt_GPa"][i])

    if strains_i:
        order = np.argsort(strains_i)
        strains_sorted = np.array(strains_i)[order]
        stresses_sorted = np.array(stresses_i)[order]
        ax.plot(strains_sorted, stresses_sorted, "o-", color="steelblue", markersize=6)

        # Linear fit
        if len(strains_sorted) >= 2:
            coeffs = np.polyfit(strains_sorted, stresses_sorted, 1)
            fit_x = np.linspace(strains_sorted.min(), strains_sorted.max(), 50)
            ax.plot(fit_x, np.polyval(coeffs, fit_x), "--", color="red", alpha=0.7,
                    label=f"slope={coeffs[0]:.1f} GPa")
            ax.legend(fontsize=9)

    ax.set_xlabel(f"Strain $\\epsilon_{{{voigt_labels[i]}}}$")
    ax.set_ylabel(f"Stress $\\sigma_{{{voigt_labels[i]}}}$ (GPa)")
    ax.set_title(f"Component {voigt_labels[i]}")
    ax.grid(True, alpha=0.3)

fig.suptitle(f"Stress-Strain Curves: {relaxed_structure.composition.reduced_formula}",
             fontsize=14)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "stress_strain_curves.png"), dpi=150, bbox_inches="tight")
print(f"\nStress-strain plot saved to {OUTPUT_DIR}/stress_strain_curves.png")

# ─── Save results to JSON ───────────────────────────────────────────────────
results = {
    "formula": str(relaxed_structure.composition.reduced_formula),
    "space_group": sga.get_space_group_symbol(),
    "crystal_system": sga_info,
    "volume_A3": relaxed_structure.volume,
    "elastic_tensor_GPa": C_matrix.tolist(),
    "K_Voigt_GPa": K_voigt,
    "K_Reuss_GPa": K_reuss,
    "K_VRH_GPa": K_vrh,
    "G_Voigt_GPa": G_voigt,
    "G_Reuss_GPa": G_reuss,
    "G_VRH_GPa": G_vrh,
    "E_VRH_GPa": E_vrh,
    "Poisson_ratio_VRH": nu_vrh,
    "eigenvalues_C": eigenvalues.tolist(),
    "is_mechanically_stable": bool(is_stable),
    "method": f"MACE-MP-0 ({MACE_MODEL})",
    "strain_magnitudes": STRAIN_MAGNITUDES,
    "n_deformations": len(deformations),
}
with open(os.path.join(OUTPUT_DIR, "elastic_results.json"), "w") as f:
    json.dump(results, f, indent=2)
print(f"Results saved to {OUTPUT_DIR}/elastic_results.json")
```

### Method B: QE DFT

This method uses Quantum ESPRESSO `pw.x` to compute stresses under applied strain deformations. The workflow is: (1) relax the structure with `vc-relax`, (2) apply strain deformations, (3) run `scf` at each deformed geometry (ions relaxed or fixed), (4) extract stresses, (5) fit the elastic tensor.

#### Step B1: Relax the bulk structure

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

#### Step B2: Generate deformed inputs and run calculations

```python
#!/usr/bin/env python3
"""
Generate QE input files for elastic constant calculation (stress-strain method).
Run after vc-relax to get the equilibrium structure.
"""

import os
import json
import re
import numpy as np
from pymatgen.core.structure import Structure
from pymatgen.io.pwscf import PWInput
from pymatgen.analysis.elasticity import Strain, Stress, Deformation, ElasticTensor
from pymatgen.core.tensors import symmetry_reduce
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer

# ─── CONFIGURATION ───────────────────────────────────────────────────────────
RELAXED_FILE = "relaxed_structure.cif"  # Extracted from vc-relax output
PSEUDO_DIR = "./pseudo"
ECUTWFC = 60.0
ECUTRHO = 480.0
K_GRID = [8, 8, 8]
STRAIN_MAGNITUDES = [-0.01, -0.005, 0.005, 0.01]
SYM_REDUCE = True
SYMPREC = 0.01
WORK_DIR = "elastic_qe"
# ─────────────────────────────────────────────────────────────────────────────

os.makedirs(WORK_DIR, exist_ok=True)

# Load relaxed structure
structure = Structure.from_file(RELAXED_FILE)
print(f"Structure: {structure.composition.reduced_formula}")
print(f"Volume: {structure.volume:.4f} A^3")

sga = SpacegroupAnalyzer(structure, symprec=SYMPREC)
print(f"Space group: {sga.get_space_group_symbol()}")

# Generate strain deformations (same as atomate2)
strain_states = [
    (1, 0, 0, 0, 0, 0),
    (0, 1, 0, 0, 0, 0),
    (0, 0, 1, 0, 0, 0),
    (0, 0, 0, 2, 0, 0),
    (0, 0, 0, 0, 2, 0),
    (0, 0, 0, 0, 0, 2),
]

strains = []
for state in strain_states:
    for mag in STRAIN_MAGNITUDES:
        strains.append(Strain.from_voigt(mag * np.array(state)))
strains = [s for s in strains if (np.abs(s) > 1e-10).any()]

if SYM_REDUCE:
    strain_mapping = symmetry_reduce(strains, structure, symprec=SYMPREC)
    strains = list(strain_mapping.keys())
    print(f"Symmetry-reduced deformations: {len(strains)}")

deformations = [s.get_deformation_matrix() for s in strains]

# Pseudopotential map -- adjust for your elements
pseudo_map = {}
for el in structure.composition.elements:
    symbol = el.symbol
    # Common SSSP naming convention
    pseudo_map[symbol] = f"{symbol}.pbe-n-rrkjus_psl.1.0.0.UPF"

# Generate QE input for each deformation
deformation_info = []
for idx, deformation in enumerate(deformations):
    deform_dir = os.path.join(WORK_DIR, f"deform_{idx:03d}")
    os.makedirs(deform_dir, exist_ok=True)

    # Apply deformation
    deformed = structure.copy()
    deformed.apply_strain(Strain.from_deformation(Deformation(deformation)))

    # Write QE input -- relax ions at fixed cell shape
    input_params = {
        "CONTROL": {
            "calculation": "relax",
            "prefix": f"deform_{idx:03d}",
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
        deformed,
        pseudo=pseudo_map,
        control=input_params["CONTROL"],
        system=input_params["SYSTEM"],
        electrons=input_params["ELECTRONS"],
        ions=input_params["IONS"],
        kpoints_grid=tuple(K_GRID),
    )
    input_file = os.path.join(deform_dir, "scf.in")
    pw_input.write_file(input_file)

    deformation_info.append({
        "index": idx,
        "deformation": np.array(deformation).tolist(),
        "strain_voigt": Strain.from_deformation(Deformation(deformation)).voigt.tolist(),
        "directory": deform_dir,
    })

# Save deformation metadata
with open(os.path.join(WORK_DIR, "deformation_info.json"), "w") as f:
    json.dump(deformation_info, f, indent=2)

# Generate run script
run_script = "#!/bin/bash\n"
run_script += "# Run all deformation calculations\n"
run_script += f"# Total deformations: {len(deformations)}\n\n"
for idx in range(len(deformations)):
    run_script += f"echo 'Running deformation {idx+1}/{len(deformations)}'\n"
    run_script += f"cd {WORK_DIR}/deform_{idx:03d}\n"
    run_script += f"pw.x < scf.in > scf.out 2>&1\n"
    run_script += f"cd ../..\n\n"

with open(os.path.join(WORK_DIR, "run_all.sh"), "w") as f:
    f.write(run_script)
os.chmod(os.path.join(WORK_DIR, "run_all.sh"), 0o755)

print(f"\nGenerated {len(deformations)} QE input files in {WORK_DIR}/")
print(f"Run: bash {WORK_DIR}/run_all.sh")
```

#### Step B3: Extract stresses and fit elastic tensor

```python
#!/usr/bin/env python3
"""
Post-process QE outputs to extract stresses and fit the elastic tensor.
Run after all deformation calculations complete.
"""

import os
import json
import re
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from pymatgen.core.structure import Structure
from pymatgen.analysis.elasticity import Strain, Stress, Deformation, ElasticTensor
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer

# ─── CONFIGURATION ───────────────────────────────────────────────────────────
RELAXED_FILE = "relaxed_structure.cif"
WORK_DIR = "elastic_qe"
SYMPREC = 0.01
OUTPUT_DIR = "elastic_results_qe"
# ─────────────────────────────────────────────────────────────────────────────

os.makedirs(OUTPUT_DIR, exist_ok=True)

structure = Structure.from_file(RELAXED_FILE)


def parse_qe_stress(output_file):
    """
    Extract the stress tensor from a QE pw.x output file.
    QE prints stress in kbar. Returns 3x3 array in GPa.
    """
    stress = None
    with open(output_file) as f:
        lines = f.readlines()

    for i, line in enumerate(lines):
        if "total   stress" in line:
            # Next 3 lines contain the stress tensor (in kbar)
            s = []
            for j in range(1, 4):
                parts = lines[i + j].split()
                # QE output: sigma(1 1) sigma(1 2) sigma(1 3)   P= ...
                # First 3 numbers are the stress tensor row in kbar
                s.append([float(parts[0]), float(parts[1]), float(parts[2])])
            stress = np.array(s)  # kbar

    if stress is None:
        raise ValueError(f"Could not parse stress from {output_file}")

    # Convert kbar to GPa (1 kbar = 0.1 GPa)
    return stress * 0.1


# Also get the equilibrium stress from the relaxed structure
# (should be ~0 after vc-relax)
eq_stress_file = os.path.join(WORK_DIR, "..", "relax.out")
try:
    eq_stress_gpa = parse_qe_stress(eq_stress_file)
    eq_stress = Stress(eq_stress_gpa)
    print(f"Equilibrium stress (GPa):\n{eq_stress_gpa}")
except Exception:
    eq_stress = Stress(np.zeros((3, 3)))
    print("No equilibrium stress found, assuming zero.")

# Load deformation metadata
with open(os.path.join(WORK_DIR, "deformation_info.json")) as f:
    deformation_info = json.load(f)

stresses_list = []
deformations_list = []
strain_data = []

for info in deformation_info:
    idx = info["index"]
    deform_dir = info["directory"]
    output_file = os.path.join(deform_dir, "scf.out")

    if not os.path.exists(output_file):
        print(f"  WARNING: {output_file} not found, skipping deformation {idx}")
        continue

    try:
        stress_gpa = parse_qe_stress(output_file)
    except ValueError as e:
        print(f"  WARNING: {e}, skipping deformation {idx}")
        continue

    # QE stress convention: positive = compressive (like VASP).
    # For pymatgen ElasticTensor, we need sigma_ij with the mechanics convention.
    # QE: stress = -d(E)/d(strain)/V, with compression positive.
    # pymatgen expects: stress with tension positive.
    # So negate the QE stress:
    stress_obj = Stress(-stress_gpa)

    deform_matrix = np.array(info["deformation"])
    deformation = Deformation(deform_matrix)
    strain_voigt = Strain.from_deformation(deformation).voigt

    stresses_list.append(stress_obj)
    deformations_list.append(deformation)
    strain_data.append({
        "strain_voigt": strain_voigt.tolist(),
        "stress_voigt_GPa": stress_obj.voigt.tolist(),
    })

    print(f"  Deformation {idx}: strain_norm={np.linalg.norm(strain_voigt):.5f}, "
          f"max|stress|={np.max(np.abs(stress_obj.voigt)):.4f} GPa")

print(f"\nSuccessfully parsed {len(stresses_list)}/{len(deformation_info)} deformations")

# Reconstruct strains
strains = [Strain.from_deformation(d) for d in deformations_list]

# Fit elastic tensor
elastic_tensor = ElasticTensor.from_independent_strains(
    stresses_list, strains, eq_stress=eq_stress
)

# Symmetrize
sga = SpacegroupAnalyzer(structure, symprec=SYMPREC)
symmops = sga.get_symmetry_operations(cartesian=True)
C_raw = elastic_tensor.voigt
C_sym = np.zeros((6, 6))
for op in symmops:
    rotated = elastic_tensor.transform(op.rotation_matrix)
    C_sym += rotated.voigt
C_sym /= len(symmops)
C_sym_tensor = ElasticTensor.from_voigt(C_sym)

print("\n" + "="*60)
print("ELASTIC TENSOR (Voigt notation, GPa):")
print("="*60)
for i in range(6):
    row = "  ".join(f"{C_sym[i,j]:8.2f}" for j in range(6))
    print(f"  [{row}]")

# Polycrystalline moduli
K_vrh = C_sym_tensor.k_vrh
G_vrh = C_sym_tensor.g_vrh
E_vrh = 9*K_vrh*G_vrh/(3*K_vrh+G_vrh) if (3*K_vrh+G_vrh) > 0 else 0
nu_vrh = (3*K_vrh-2*G_vrh)/(6*K_vrh+2*G_vrh) if (6*K_vrh+2*G_vrh) > 0 else 0

print(f"\nPolycrystalline Moduli (VRH, GPa):")
print(f"  Bulk modulus  K = {K_vrh:.2f}")
print(f"  Shear modulus G = {G_vrh:.2f}")
print(f"  Young's modulus E = {E_vrh:.2f}")
print(f"  Poisson's ratio nu = {nu_vrh:.4f}")

# Born stability
eigenvalues = np.linalg.eigvalsh(C_sym)
is_stable = all(ev > 0 for ev in eigenvalues)
print(f"\nBorn Stability: eigenvalues = {[f'{ev:.2f}' for ev in eigenvalues]}")
print(f"  Mechanically stable: {'YES' if is_stable else 'NO'}")

# Save results
results = {
    "formula": str(structure.composition.reduced_formula),
    "elastic_tensor_GPa": C_sym.tolist(),
    "K_VRH_GPa": K_vrh,
    "G_VRH_GPa": G_vrh,
    "E_VRH_GPa": E_vrh,
    "Poisson_ratio_VRH": nu_vrh,
    "eigenvalues_C": eigenvalues.tolist(),
    "is_mechanically_stable": bool(is_stable),
    "method": "QE PBE (stress-strain)",
    "n_deformations": len(stresses_list),
}
with open(os.path.join(OUTPUT_DIR, "elastic_results.json"), "w") as f:
    json.dump(results, f, indent=2)
print(f"\nResults saved to {OUTPUT_DIR}/elastic_results.json")

# ─── Visualization ───────────────────────────────────────────────────────────
fig, axes = plt.subplots(2, 3, figsize=(15, 10))
voigt_labels = ["xx", "yy", "zz", "yz", "xz", "xy"]

for i, ax in enumerate(axes.flat):
    strains_i, stresses_i = [], []
    for data in strain_data:
        sv = data["strain_voigt"]
        dominant_idx = np.argmax(np.abs(sv))
        if dominant_idx == i:
            strains_i.append(sv[i])
            stresses_i.append(data["stress_voigt_GPa"][i])

    if strains_i:
        order = np.argsort(strains_i)
        xs = np.array(strains_i)[order]
        ys = np.array(stresses_i)[order]
        ax.plot(xs, ys, "o-", color="steelblue", markersize=6)
        if len(xs) >= 2:
            c = np.polyfit(xs, ys, 1)
            fx = np.linspace(xs.min(), xs.max(), 50)
            ax.plot(fx, np.polyval(c, fx), "--r", alpha=0.7, label=f"slope={c[0]:.1f} GPa")
            ax.legend(fontsize=9)

    ax.set_xlabel(f"$\\epsilon_{{{voigt_labels[i]}}}$")
    ax.set_ylabel(f"$\\sigma_{{{voigt_labels[i]}}}$ (GPa)")
    ax.set_title(f"Component {voigt_labels[i]}")
    ax.grid(True, alpha=0.3)

fig.suptitle(f"Stress-Strain (QE): {structure.composition.reduced_formula}", fontsize=14)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "stress_strain_curves_qe.png"), dpi=150, bbox_inches="tight")
print(f"Plot saved to {OUTPUT_DIR}/stress_strain_curves_qe.png")
```

## Key Parameters

| Parameter | Typical Value | Notes |
|---|---|---|
| Strain magnitude | 0.005 -- 0.01 | Larger strains improve signal-to-noise but risk nonlinearity. Use 0.005 for stiff materials, 0.01 for soft. |
| Number of strain magnitudes | 4--5 per strain state | atomate2 default: `np.linspace(-0.01, 0.01, 5)`. More points give better linear fit. |
| Strain states | 6 (2nd order) | The 6 independent Voigt directions: 3 normal + 3 shear. |
| Symmetry reduction | Enabled | Reduces deformations significantly for high-symmetry structures (e.g., cubic: 24 -> ~2--4 independent). |
| FMAX (MACE relaxation) | 1e-4 eV/A (bulk), 1e-3 eV/A (deformed) | Tighter bulk relaxation ensures near-zero equilibrium stress. |
| ecutwfc (QE) | 60--80 Ry | Must be converged for accurate stresses. Stresses converge slower than energies. |
| k-grid (QE) | Dense (e.g., 8x8x8 for FCC) | Stresses are sensitive to k-point sampling. |
| conv_thr (QE) | 1e-10 Ry | Tight SCF convergence needed for reliable stress tensors. |

## Interpreting Results

**Elastic tensor (C_ij):**
- Diagonal elements C_11, C_22, C_33 represent resistance to uniaxial strain.
- C_44, C_55, C_66 represent resistance to shear.
- Off-diagonal C_12, C_13, C_23 describe coupling between normal strains.
- For cubic: only C_11, C_12, C_44 are independent.
- For hexagonal: C_11, C_12, C_13, C_33, C_44 are independent.

**Polycrystalline moduli:**
- K (bulk modulus): resistance to uniform compression. Voigt = upper bound, Reuss = lower bound.
- G (shear modulus): resistance to shape change.
- E (Young's modulus): uniaxial stiffness. E = 9KG/(3K+G).
- nu (Poisson's ratio): lateral contraction. Typically 0.2--0.4 for metals, 0.1--0.3 for ceramics.
- Pugh's ratio K/G > 1.75 suggests ductile behavior, < 1.75 suggests brittle.

**Born stability criteria:**
- All eigenvalues of C_ij must be positive for mechanical stability.
- Negative eigenvalues indicate the structure is a saddle point, not a true minimum.

**Typical accuracy:**
- MACE: within 10--20% of DFT for well-represented chemistries. Shear constants often less accurate than bulk modulus.
- QE PBE: within 5--15% of experiment. GGA typically underestimates elastic constants slightly.

## Common Issues

| Problem | Cause | Solution |
|---|---|---|
| Negative eigenvalues of C_ij | Structure not fully relaxed, or genuinely unstable phase | Re-relax with tighter convergence. If still unstable, the phase may be mechanically unstable at 0 K. |
| Large off-diagonal noise | Insufficient symmetry reduction or poor stress convergence | Enable `SYM_REDUCE=True`. For QE, tighten `conv_thr` and increase `ecutwfc`. |
| C_ij not symmetric | Numerical noise, asymmetric strain set | Symmetrize: `C = (C + C.T) / 2`. The `ElasticTensor` class does this internally. |
| Stress-strain curve is nonlinear | Strain magnitude too large | Reduce strain magnitude to 0.005 or smaller. |
| Very different Voigt vs Reuss bounds | Strong elastic anisotropy | This is physical for anisotropic materials (e.g., layered structures). Report VRH average. |
| QE stress oscillating | Poor SCF convergence or insufficient k-points | Increase k-grid density. Set `conv_thr = 1e-10`. Use `mixing_beta = 0.3` for metals. |
| MACE gives unreasonable values | Material outside MACE training domain | Fall back to QE DFT. Check by comparing MACE-relaxed structure with known lattice parameters. |
| "Strain list insufficient" error | Symmetry reduction removed too many strains | Disable symmetry reduction or add more strain magnitudes. |
