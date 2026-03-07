# Energy-Strain Method for Elastic Constants

## When to Use

- You need elastic constants computed from the curvature of the energy-strain curve E(epsilon) rather than from stress tensors (see `stress-strain-method/` for the alternative).
- The DFT code or force-field does not provide reliable stress tensors, but total energies are well converged.
- You want a more robust approach for codes where stress implementation is incomplete or numerically noisy (e.g., some all-electron codes, older pseudopotential codes, certain MLIP backends).
- You need to cross-validate elastic constants obtained by the stress-strain method using an independent fitting approach.
- You are working with VASP and want to use the VASPKIT elastic constant workflow (menu 02, tasks 204--207) which is based on the energy-strain formalism.
- You want to compute second-order elastic constants from the parabolic relation E(epsilon) = E0 + (1/2) V0 C_eff epsilon^2, where C_eff is a linear combination of C_ij determined by the strain pattern.

## Method Selection

| Criterion | ASE + MACE | QE DFT | VASP |
|---|---|---|---|
| Speed | Seconds | Hours to days | Hours to days |
| Accuracy | Good for MACE training domain | Publication quality (PBE/PBEsol) | Publication quality (PBE/PBEsol/SCAN) |
| Energy output | `atoms.get_potential_energy()` | `!    total energy` in pw.x output | `TOTEN` / `energy without entropy` in OUTCAR |
| Use when | Screening, rapid prototyping | Standard DFT, no VASP license | VASP license available; VASPKIT automates 204--207 |
| Advantages | Near-instant; energy is the most fundamental quantity | Energies converge faster than stresses with basis set | VASPKIT automates strain generation, OUTCAR parsing, and fitting |
| Disadvantages | Accuracy limited to MACE training domain | Requires more strain points than stress-strain (parabolic fit) | Requires VASP license and POTCAR library |
| Stress tensor needed? | No -- uses only total energies | No -- uses only total energies | No -- uses only total energies |

## Prerequisites

- A relaxed crystal structure (CIF, POSCAR, or pymatgen Structure) at zero stress (equilibrium). The structure **must** be fully relaxed before applying strain deformations -- residual stress introduces linear terms in E(epsilon) that corrupt the parabolic fit.
- For QE: pseudopotential files (SSSP recommended). See `electronic-structure/scf-relax/SKILL.md` for setup.
- For VASP: POTCAR files and a valid VASP license. VASPKIT >= 1.3 for automated elastic workflows.
- Python packages: `pymatgen`, `ase`, `mace-torch`, `numpy`, `scipy`, `matplotlib` (pre-installed).
- Understanding of Lagrangian strain: the method applies finite Lagrangian strain tensors eta_ij and fits the resulting energy change Delta_E = (V0/2) * sum_{ijkl} C_{ijkl} eta_{ij} eta_{kl}. For each symmetry-adapted strain pattern, only a specific linear combination C_eff of elastic constants contributes.

## Detailed Steps

### Method A: ASE + MACE

This method applies symmetry-adapted Lagrangian strain patterns at multiple magnitudes, computes the total energy at each strained configuration using the MACE calculator, and fits parabolas E(epsilon) to extract each independent elastic constant combination. The workflow: (1) relax bulk, (2) identify crystal system and select symmetry-adapted strain patterns, (3) apply each pattern at multiple magnitudes and compute energy, (4) fit E(epsilon) = E0 + (1/2) V0 C_eff epsilon^2 for each pattern, (5) solve for the independent C_ij, (6) compute polycrystalline moduli.

```python
#!/usr/bin/env python3
"""
Elastic constants via the energy-strain method using ASE + MACE.

Theory:
  For a crystal under Lagrangian strain eta, the elastic energy is:
    Delta_E = (V0 / 2) * sum_{ijkl} C_{ijkl} * eta_{ij} * eta_{kl}

  By choosing specific strain patterns (e.g., uniaxial, shear, volume-
  conserving), each pattern yields a parabola:
    E(eps) = E0 + (1/2) * V0 * C_eff * eps^2

  where C_eff is a known linear combination of C_ij. Fitting the parabola
  gives C_eff, and solving the system of equations gives all independent C_ij.

Workflow:
  1. Relax the structure (full cell + ions).
  2. Identify crystal system and select strain patterns.
  3. For each strain pattern at multiple magnitudes: apply strain, relax ions
     (fixed cell), record total energy.
  4. Fit E(eps) parabolas to extract C_eff for each pattern.
  5. Solve for independent elastic constants C_ij.
  6. Compute Voigt-Reuss-Hill polycrystalline moduli.
  7. Check Born stability criteria.
  8. Plot energy-strain curves with fits.
"""

import json
import os
import warnings
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from ase.io import read as ase_read
from ase.optimize import LBFGS
from ase.constraints import ExpCellFilter

from pymatgen.core.structure import Structure
from pymatgen.analysis.elasticity import ElasticTensor
from pymatgen.io.ase import AseAtomsAdaptor
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
from scipy.optimize import curve_fit

# ---- CONFIGURATION --------------------------------------------------------
INPUT_FILE = "structure.cif"           # Input structure (CIF, POSCAR, etc.)
MACE_MODEL = "medium"                  # MACE model: "small", "medium", "large"
STRAIN_MAGNITUDES = np.array([-0.020, -0.015, -0.010, -0.005,
                               0.005,  0.010,  0.015,  0.020])
FMAX_BULK = 1e-4                       # Force convergence for bulk relaxation (eV/A)
FMAX_DEFORM = 1e-3                     # Force convergence for strained cells (eV/A)
OUTPUT_DIR = "energy_strain_results"
SYMPREC = 0.01                         # Symmetry precision (Angstrom)
# ---------------------------------------------------------------------------

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ---- Lagrangian strain patterns per crystal system -----------------------
# Each pattern is a 6-component Voigt vector (eta_xx, eta_yy, eta_zz,
# 2*eta_yz, 2*eta_xz, 2*eta_xy). The associated C_eff is the linear
# combination of C_ij that appears in the energy expansion.
#
# Reference: Ravindran et al., J. Appl. Phys. 84, 4891 (1998);
#            Mouhat & Coudert, Phys. Rev. B 90, 224104 (2014).

STRAIN_PATTERNS = {
    "cubic": {
        # 3 independent: C11, C12, C44
        "patterns": [
            {"name": "D1_volume",   "voigt": [1, 1, 1, 0, 0, 0],
             "c_eff_expr": "C11 + 2*C12",
             "c_eff_coeffs": {"C11": 1, "C12": 2}},
            {"name": "D2_tetra",    "voigt": [1, -1, 0, 0, 0, 0],
             "c_eff_expr": "C11 - C12",
             "c_eff_coeffs": {"C11": 1, "C12": -1}},
            {"name": "D3_shear",    "voigt": [0, 0, 0, 2, 0, 0],
             "c_eff_expr": "C44",
             "c_eff_coeffs": {"C44": 1}},
        ],
        "independent": ["C11", "C12", "C44"],
    },
    "hexagonal": {
        # 5 independent: C11, C12, C13, C33, C44
        "patterns": [
            {"name": "D1_xx",       "voigt": [1, 0, 0, 0, 0, 0],
             "c_eff_expr": "C11",
             "c_eff_coeffs": {"C11": 1}},
            {"name": "D2_zz",       "voigt": [0, 0, 1, 0, 0, 0],
             "c_eff_expr": "C33",
             "c_eff_coeffs": {"C33": 1}},
            {"name": "D3_biaxial",  "voigt": [1, 1, 0, 0, 0, 0],
             "c_eff_expr": "C11 + C12",
             "c_eff_coeffs": {"C11": 1, "C12": 1}},
            {"name": "D4_xz",      "voigt": [1, 0, 1, 0, 0, 0],
             "c_eff_expr": "C11 + C33 + 2*C13",
             "c_eff_coeffs": {"C11": 1, "C33": 1, "C13": 2}},
            {"name": "D5_shear",    "voigt": [0, 0, 0, 2, 0, 0],
             "c_eff_expr": "C44",
             "c_eff_coeffs": {"C44": 1}},
        ],
        "independent": ["C11", "C12", "C13", "C33", "C44"],
    },
    "trigonal_high": {
        # 6 independent: C11, C12, C13, C14, C33, C44  (point groups 32, -3m, 3m)
        "patterns": [
            {"name": "D1_xx",       "voigt": [1, 0, 0, 0, 0, 0],
             "c_eff_expr": "C11",
             "c_eff_coeffs": {"C11": 1}},
            {"name": "D2_zz",       "voigt": [0, 0, 1, 0, 0, 0],
             "c_eff_expr": "C33",
             "c_eff_coeffs": {"C33": 1}},
            {"name": "D3_biaxial",  "voigt": [1, 1, 0, 0, 0, 0],
             "c_eff_expr": "C11 + C12",
             "c_eff_coeffs": {"C11": 1, "C12": 1}},
            {"name": "D4_xz",      "voigt": [1, 0, 1, 0, 0, 0],
             "c_eff_expr": "C11 + C33 + 2*C13",
             "c_eff_coeffs": {"C11": 1, "C33": 1, "C13": 2}},
            {"name": "D5_yz_shear", "voigt": [0, 0, 0, 2, 0, 0],
             "c_eff_expr": "C44",
             "c_eff_coeffs": {"C44": 1}},
            {"name": "D6_coupled",  "voigt": [0, 0, 0, 2, 2, 0],
             "c_eff_expr": "2*C44 + 2*C14",
             "c_eff_coeffs": {"C44": 2, "C14": 2}},
        ],
        "independent": ["C11", "C12", "C13", "C14", "C33", "C44"],
    },
    "tetragonal_high": {
        # 6 independent: C11, C12, C13, C33, C44, C66  (point groups 4/mmm, 422, -42m, 4mm)
        "patterns": [
            {"name": "D1_xx",       "voigt": [1, 0, 0, 0, 0, 0],
             "c_eff_expr": "C11",
             "c_eff_coeffs": {"C11": 1}},
            {"name": "D2_zz",       "voigt": [0, 0, 1, 0, 0, 0],
             "c_eff_expr": "C33",
             "c_eff_coeffs": {"C33": 1}},
            {"name": "D3_biaxial",  "voigt": [1, 1, 0, 0, 0, 0],
             "c_eff_expr": "C11 + C12",
             "c_eff_coeffs": {"C11": 1, "C12": 1}},
            {"name": "D4_xz",      "voigt": [1, 0, 1, 0, 0, 0],
             "c_eff_expr": "C11 + C33 + 2*C13",
             "c_eff_coeffs": {"C11": 1, "C33": 1, "C13": 2}},
            {"name": "D5_yz_shear", "voigt": [0, 0, 0, 2, 0, 0],
             "c_eff_expr": "C44",
             "c_eff_coeffs": {"C44": 1}},
            {"name": "D6_xy_shear", "voigt": [0, 0, 0, 0, 0, 2],
             "c_eff_expr": "C66",
             "c_eff_coeffs": {"C66": 1}},
        ],
        "independent": ["C11", "C12", "C13", "C33", "C44", "C66"],
    },
    "orthorhombic": {
        # 9 independent: C11, C12, C13, C22, C23, C33, C44, C55, C66
        "patterns": [
            {"name": "D1_xx",       "voigt": [1, 0, 0, 0, 0, 0],
             "c_eff_expr": "C11",
             "c_eff_coeffs": {"C11": 1}},
            {"name": "D2_yy",       "voigt": [0, 1, 0, 0, 0, 0],
             "c_eff_expr": "C22",
             "c_eff_coeffs": {"C22": 1}},
            {"name": "D3_zz",       "voigt": [0, 0, 1, 0, 0, 0],
             "c_eff_expr": "C33",
             "c_eff_coeffs": {"C33": 1}},
            {"name": "D4_xy",       "voigt": [1, 1, 0, 0, 0, 0],
             "c_eff_expr": "C11 + C22 + 2*C12",
             "c_eff_coeffs": {"C11": 1, "C22": 1, "C12": 2}},
            {"name": "D5_xz",       "voigt": [1, 0, 1, 0, 0, 0],
             "c_eff_expr": "C11 + C33 + 2*C13",
             "c_eff_coeffs": {"C11": 1, "C33": 1, "C13": 2}},
            {"name": "D6_yz",       "voigt": [0, 1, 1, 0, 0, 0],
             "c_eff_expr": "C22 + C33 + 2*C23",
             "c_eff_coeffs": {"C22": 1, "C33": 1, "C23": 2}},
            {"name": "D7_shear_yz", "voigt": [0, 0, 0, 2, 0, 0],
             "c_eff_expr": "C44",
             "c_eff_coeffs": {"C44": 1}},
            {"name": "D8_shear_xz", "voigt": [0, 0, 0, 0, 2, 0],
             "c_eff_expr": "C55",
             "c_eff_coeffs": {"C55": 1}},
            {"name": "D9_shear_xy", "voigt": [0, 0, 0, 0, 0, 2],
             "c_eff_expr": "C66",
             "c_eff_coeffs": {"C66": 1}},
        ],
        "independent": ["C11", "C12", "C13", "C22", "C23", "C33",
                         "C44", "C55", "C66"],
    },
    "monoclinic": {
        # 13 independent -- use all 6 uniaxial + 3 shear + 4 coupled patterns
        "patterns": [
            {"name": "D01_xx",  "voigt": [1, 0, 0, 0, 0, 0], "c_eff_coeffs": {"C11": 1}},
            {"name": "D02_yy",  "voigt": [0, 1, 0, 0, 0, 0], "c_eff_coeffs": {"C22": 1}},
            {"name": "D03_zz",  "voigt": [0, 0, 1, 0, 0, 0], "c_eff_coeffs": {"C33": 1}},
            {"name": "D04_yz",  "voigt": [0, 0, 0, 2, 0, 0], "c_eff_coeffs": {"C44": 1}},
            {"name": "D05_xz",  "voigt": [0, 0, 0, 0, 2, 0], "c_eff_coeffs": {"C55": 1}},
            {"name": "D06_xy",  "voigt": [0, 0, 0, 0, 0, 2], "c_eff_coeffs": {"C66": 1}},
            {"name": "D07_xxyy", "voigt": [1, 1, 0, 0, 0, 0], "c_eff_coeffs": {"C11": 1, "C22": 1, "C12": 2}},
            {"name": "D08_xxzz", "voigt": [1, 0, 1, 0, 0, 0], "c_eff_coeffs": {"C11": 1, "C33": 1, "C13": 2}},
            {"name": "D09_yyzz", "voigt": [0, 1, 1, 0, 0, 0], "c_eff_coeffs": {"C22": 1, "C33": 1, "C23": 2}},
            {"name": "D10_yz_xy", "voigt": [0, 0, 0, 2, 0, 2], "c_eff_coeffs": {"C44": 1, "C66": 1, "C46": 2}},
            {"name": "D11_xx_xz", "voigt": [1, 0, 0, 0, 2, 0], "c_eff_coeffs": {"C11": 1, "C55": 1, "C15": 2}},
            {"name": "D12_yy_xz", "voigt": [0, 1, 0, 0, 2, 0], "c_eff_coeffs": {"C22": 1, "C55": 1, "C25": 2}},
            {"name": "D13_zz_xz", "voigt": [0, 0, 1, 0, 2, 0], "c_eff_coeffs": {"C33": 1, "C55": 1, "C35": 2}},
        ],
        "independent": ["C11", "C12", "C13", "C15", "C22", "C23", "C25",
                         "C33", "C35", "C44", "C46", "C55", "C66"],
    },
    "triclinic": {
        # 21 independent -- use full set of 21 strain patterns
        "patterns": [
            {"name": "D01_xx",     "voigt": [1, 0, 0, 0, 0, 0], "c_eff_coeffs": {"C11": 1}},
            {"name": "D02_yy",     "voigt": [0, 1, 0, 0, 0, 0], "c_eff_coeffs": {"C22": 1}},
            {"name": "D03_zz",     "voigt": [0, 0, 1, 0, 0, 0], "c_eff_coeffs": {"C33": 1}},
            {"name": "D04_yz",     "voigt": [0, 0, 0, 2, 0, 0], "c_eff_coeffs": {"C44": 1}},
            {"name": "D05_xz",     "voigt": [0, 0, 0, 0, 2, 0], "c_eff_coeffs": {"C55": 1}},
            {"name": "D06_xy",     "voigt": [0, 0, 0, 0, 0, 2], "c_eff_coeffs": {"C66": 1}},
            {"name": "D07_xxyy",   "voigt": [1, 1, 0, 0, 0, 0], "c_eff_coeffs": {"C11": 1, "C22": 1, "C12": 2}},
            {"name": "D08_xxzz",   "voigt": [1, 0, 1, 0, 0, 0], "c_eff_coeffs": {"C11": 1, "C33": 1, "C13": 2}},
            {"name": "D09_yyzz",   "voigt": [0, 1, 1, 0, 0, 0], "c_eff_coeffs": {"C22": 1, "C33": 1, "C23": 2}},
            {"name": "D10_xxyz",   "voigt": [1, 0, 0, 2, 0, 0], "c_eff_coeffs": {"C11": 1, "C44": 1, "C14": 2}},
            {"name": "D11_xxxz",   "voigt": [1, 0, 0, 0, 2, 0], "c_eff_coeffs": {"C11": 1, "C55": 1, "C15": 2}},
            {"name": "D12_xxxy",   "voigt": [1, 0, 0, 0, 0, 2], "c_eff_coeffs": {"C11": 1, "C66": 1, "C16": 2}},
            {"name": "D13_yyyz",   "voigt": [0, 1, 0, 2, 0, 0], "c_eff_coeffs": {"C22": 1, "C44": 1, "C24": 2}},
            {"name": "D14_yyxz",   "voigt": [0, 1, 0, 0, 2, 0], "c_eff_coeffs": {"C22": 1, "C55": 1, "C25": 2}},
            {"name": "D15_yyxy",   "voigt": [0, 1, 0, 0, 0, 2], "c_eff_coeffs": {"C22": 1, "C66": 1, "C26": 2}},
            {"name": "D16_zzyz",   "voigt": [0, 0, 1, 2, 0, 0], "c_eff_coeffs": {"C33": 1, "C44": 1, "C34": 2}},
            {"name": "D17_zzxz",   "voigt": [0, 0, 1, 0, 2, 0], "c_eff_coeffs": {"C33": 1, "C55": 1, "C35": 2}},
            {"name": "D18_zzxy",   "voigt": [0, 0, 1, 0, 0, 2], "c_eff_coeffs": {"C33": 1, "C66": 1, "C36": 2}},
            {"name": "D19_yzxz",   "voigt": [0, 0, 0, 2, 2, 0], "c_eff_coeffs": {"C44": 1, "C55": 1, "C45": 2}},
            {"name": "D20_yzxy",   "voigt": [0, 0, 0, 2, 0, 2], "c_eff_coeffs": {"C44": 1, "C66": 1, "C46": 2}},
            {"name": "D21_xzxy",   "voigt": [0, 0, 0, 0, 2, 2], "c_eff_coeffs": {"C55": 1, "C66": 1, "C56": 2}},
        ],
        "independent": ["C11", "C12", "C13", "C14", "C15", "C16",
                         "C22", "C23", "C24", "C25", "C26",
                         "C33", "C34", "C35", "C36",
                         "C44", "C45", "C46", "C55", "C56", "C66"],
    },
}


def voigt_strain_to_matrix(voigt):
    """Convert 6-component Voigt strain to 3x3 Lagrangian strain matrix.
    Voigt convention: (eta_xx, eta_yy, eta_zz, 2*eta_yz, 2*eta_xz, 2*eta_xy).
    """
    eta = np.array(voigt, dtype=float)
    return np.array([
        [eta[0],       eta[5] / 2.0, eta[4] / 2.0],
        [eta[5] / 2.0, eta[1],       eta[3] / 2.0],
        [eta[4] / 2.0, eta[3] / 2.0, eta[2]],
    ])


def lagrangian_strain_to_deformation(eta_matrix):
    """Convert Lagrangian strain matrix eta to deformation gradient F.
    F^T F = I + 2*eta  =>  F = sqrtm(I + 2*eta).
    For small strains, F ~ I + eta (but we use the exact formula).
    """
    I = np.eye(3)
    C_green = I + 2.0 * eta_matrix
    # Eigendecompose for stable square root
    eigvals, eigvecs = np.linalg.eigh(C_green)
    if np.any(eigvals < 0):
        raise ValueError("Green deformation tensor C is not positive definite. "
                         "Strain magnitude may be too large.")
    sqrt_eigvals = np.sqrt(eigvals)
    F = eigvecs @ np.diag(sqrt_eigvals) @ eigvecs.T
    return F


def map_crystal_system(crystal_system):
    """Map pymatgen crystal system name to our strain pattern key."""
    cs = crystal_system.lower()
    if cs == "cubic":
        return "cubic"
    elif cs == "hexagonal":
        return "hexagonal"
    elif cs == "trigonal":
        return "trigonal_high"
    elif cs == "tetragonal":
        return "tetragonal_high"
    elif cs == "orthorhombic":
        return "orthorhombic"
    elif cs == "monoclinic":
        return "monoclinic"
    elif cs == "triclinic":
        return "triclinic"
    else:
        warnings.warn(f"Unknown crystal system '{cs}', using triclinic (21 constants).")
        return "triclinic"


def solve_elastic_constants(c_eff_values, pattern_defs, independent_names):
    """Solve the linear system: for each pattern, C_eff = sum(coeff_k * C_k).
    Returns a dict {name: value_GPa}.
    """
    n_patterns = len(c_eff_values)
    n_unknowns = len(independent_names)

    # Build the coefficient matrix A and RHS b:  A @ x = b
    A = np.zeros((n_patterns, n_unknowns))
    b = np.array(c_eff_values)

    name_to_idx = {name: i for i, name in enumerate(independent_names)}
    for p_idx, pdef in enumerate(pattern_defs):
        for cname, coeff in pdef["c_eff_coeffs"].items():
            if cname in name_to_idx:
                A[p_idx, name_to_idx[cname]] = coeff

    # Solve via least squares (handles overdetermined systems)
    x, residuals, rank, sv = np.linalg.lstsq(A, b, rcond=None)

    return {name: x[i] for i, name in enumerate(independent_names)}


def build_voigt_matrix(cij_dict):
    """Build the 6x6 Voigt elastic tensor from a dict of independent C_ij.
    Missing entries default to zero; symmetry C_ij = C_ji is enforced.
    """
    C = np.zeros((6, 6))
    voigt_map = {
        "C11": (0,0), "C12": (0,1), "C13": (0,2), "C14": (0,3), "C15": (0,4), "C16": (0,5),
        "C22": (1,1), "C23": (1,2), "C24": (1,3), "C25": (1,4), "C26": (1,5),
        "C33": (2,2), "C34": (2,3), "C35": (2,4), "C36": (2,5),
        "C44": (3,3), "C45": (3,4), "C46": (3,5),
        "C55": (4,4), "C56": (4,5),
        "C66": (5,5),
    }
    for name, val in cij_dict.items():
        if name in voigt_map:
            i, j = voigt_map[name]
            C[i, j] = val
            C[j, i] = val  # symmetry
    return C


# ---- STEP 1: Load structure and set up MACE calculator --------------------
from mace.calculators import mace_mp
calc = mace_mp(model=MACE_MODEL, default_dtype="float64")

adaptor = AseAtomsAdaptor()

structure = Structure.from_file(INPUT_FILE)
formula = structure.composition.reduced_formula
sga = SpacegroupAnalyzer(structure, symprec=SYMPREC)
crystal_system = sga.get_crystal_system()
space_group = sga.get_space_group_symbol()

print(f"Loaded structure: {formula}")
print(f"Space group: {space_group}")
print(f"Crystal system: {crystal_system}")

# ---- STEP 2: Relax the bulk structure (cell + ions) -----------------------
atoms = adaptor.get_atoms(structure)
atoms.calc = calc

ecf = ExpCellFilter(atoms, hydrostatic_strain=False)
opt = LBFGS(ecf, logfile=os.path.join(OUTPUT_DIR, "bulk_relax.log"))
opt.run(fmax=FMAX_BULK, steps=500)

relaxed_structure = adaptor.get_structure(atoms)
relaxed_structure.to(os.path.join(OUTPUT_DIR, "relaxed_structure.cif"))

V0 = relaxed_structure.volume
E0 = atoms.get_potential_energy()
print(f"Relaxed volume: {V0:.4f} A^3")
print(f"Equilibrium energy: {E0:.6f} eV")

# ---- STEP 3: Select strain patterns for crystal system --------------------
cs_key = map_crystal_system(crystal_system)
sp_data = STRAIN_PATTERNS[cs_key]
patterns = sp_data["patterns"]
independent_names = sp_data["independent"]

print(f"\nUsing strain patterns for: {cs_key}")
print(f"Number of independent elastic constants: {len(independent_names)}")
print(f"Number of strain patterns: {len(patterns)}")
print(f"Strain magnitudes: {STRAIN_MAGNITUDES}")
print(f"Total single-point calculations: {len(patterns) * len(STRAIN_MAGNITUDES)}")

# ---- STEP 4: Apply strains, compute energies, fit parabolas ---------------
c_eff_values = []
fit_data = []

for p_idx, pattern in enumerate(patterns):
    voigt_pattern = np.array(pattern["voigt"], dtype=float)
    pattern_name = pattern["name"]

    epsilons = []
    energies = []

    for eps_mag in STRAIN_MAGNITUDES:
        # Scale the Voigt strain pattern by the magnitude
        voigt_scaled = voigt_pattern * eps_mag

        # Convert to Lagrangian strain matrix
        eta_matrix = voigt_strain_to_matrix(voigt_scaled)

        # Convert to deformation gradient F
        F = lagrangian_strain_to_deformation(eta_matrix)

        # Apply deformation to the relaxed structure
        deformed = relaxed_structure.copy()
        new_lattice = np.dot(F, deformed.lattice.matrix)
        deformed.lattice = new_lattice

        # Convert to ASE and relax ions only (fixed cell)
        deformed_atoms = adaptor.get_atoms(deformed)
        deformed_atoms.calc = calc

        opt_d = LBFGS(deformed_atoms, logfile=os.devnull)
        opt_d.run(fmax=FMAX_DEFORM, steps=300)

        E = deformed_atoms.get_potential_energy()
        epsilons.append(eps_mag)
        energies.append(E)

    epsilons = np.array(epsilons)
    energies = np.array(energies)
    delta_E = energies - E0  # Energy change relative to equilibrium

    # Fit parabola: Delta_E = a2 * eps^2 + a1 * eps + a0
    # The a1*eps term should be ~0 if the structure is well relaxed.
    # The a2 coefficient gives: C_eff = 2 * a2 / V0
    coeffs = np.polyfit(epsilons, delta_E, 2)
    a2 = coeffs[0]  # coefficient of eps^2
    a1 = coeffs[1]  # linear term (should be ~0)
    a0 = coeffs[2]  # constant (should be ~0)

    # C_eff = 2 * a2 / V0   (in eV/A^3)
    # Convert to GPa: 1 eV/A^3 = 160.21766 GPa
    C_eff_eV_A3 = 2.0 * a2 / V0
    C_eff_GPa = C_eff_eV_A3 * 160.21766

    c_eff_values.append(C_eff_GPa)

    # Compute R^2 for the parabolic fit
    delta_E_fit = np.polyval(coeffs, epsilons)
    ss_res = np.sum((delta_E - delta_E_fit) ** 2)
    ss_tot = np.sum((delta_E - np.mean(delta_E)) ** 2)
    R2 = 1.0 - ss_res / ss_tot if ss_tot > 1e-30 else 1.0

    fit_data.append({
        "pattern_name": pattern_name,
        "c_eff_expr": pattern.get("c_eff_expr", ""),
        "epsilons": epsilons.tolist(),
        "delta_E_eV": delta_E.tolist(),
        "a2": float(a2),
        "a1": float(a1),
        "a0": float(a0),
        "C_eff_GPa": float(C_eff_GPa),
        "R_squared": float(R2),
    })

    print(f"  Pattern {p_idx+1}/{len(patterns)} ({pattern_name}): "
          f"C_eff = {C_eff_GPa:.2f} GPa, R^2 = {R2:.6f}, "
          f"|a1/a2| = {abs(a1/a2):.4f}" if abs(a2) > 1e-20 else "")

# ---- STEP 5: Solve for independent elastic constants -----------------------
cij_dict = solve_elastic_constants(c_eff_values, patterns, independent_names)

print("\n" + "=" * 60)
print("INDEPENDENT ELASTIC CONSTANTS (GPa):")
print("=" * 60)
for name in independent_names:
    print(f"  {name} = {cij_dict[name]:.2f}")

# Build the full 6x6 Voigt matrix
C_matrix = build_voigt_matrix(cij_dict)

print("\nFULL ELASTIC TENSOR (Voigt notation, GPa):")
for i in range(6):
    row = "  ".join(f"{C_matrix[i, j]:8.2f}" for j in range(6))
    print(f"  [{row}]")

# ---- STEP 6: Voigt-Reuss-Hill polycrystalline moduli ----------------------
C_tensor = ElasticTensor.from_voigt(C_matrix)

K_voigt = C_tensor.k_voigt
K_reuss = C_tensor.k_reuss
K_vrh = C_tensor.k_vrh
G_voigt = C_tensor.g_voigt
G_reuss = C_tensor.g_reuss
G_vrh = C_tensor.g_vrh
E_vrh = 9 * K_vrh * G_vrh / (3 * K_vrh + G_vrh) if (3 * K_vrh + G_vrh) > 0 else 0.0
nu_vrh = (3 * K_vrh - 2 * G_vrh) / (6 * K_vrh + 2 * G_vrh) if (6 * K_vrh + 2 * G_vrh) > 0 else 0.0

print(f"\nPolycrystalline Moduli (GPa):")
print(f"  Bulk  modulus: K_Voigt={K_voigt:.2f}, K_Reuss={K_reuss:.2f}, K_VRH={K_vrh:.2f}")
print(f"  Shear modulus: G_Voigt={G_voigt:.2f}, G_Reuss={G_reuss:.2f}, G_VRH={G_vrh:.2f}")
print(f"  Young's modulus (VRH): E = {E_vrh:.2f} GPa")
print(f"  Poisson's ratio (VRH): nu = {nu_vrh:.4f}")
print(f"  Pugh's ratio: K/G = {K_vrh / G_vrh:.3f}" if G_vrh > 0 else "  Pugh's ratio: undefined")

# ---- STEP 7: Born mechanical stability check ------------------------------
eigenvalues = np.linalg.eigvalsh(C_matrix)
is_stable = all(ev > 0 for ev in eigenvalues)

print(f"\nBorn Stability Check:")
print(f"  Eigenvalues of C_ij: {', '.join(f'{ev:.2f}' for ev in eigenvalues)}")
print(f"  Mechanically stable: {'YES' if is_stable else 'NO -- UNSTABLE'}")

if cs_key == "cubic":
    c11, c12, c44 = cij_dict["C11"], cij_dict["C12"], cij_dict["C44"]
    cubic_ok = (c11 > 0) and (c44 > 0) and (c11 > abs(c12)) and (c11 + 2 * c12 > 0)
    A_zener = 2 * c44 / (c11 - c12) if (c11 - c12) != 0 else float('inf')
    print(f"  Cubic criteria: C11={c11:.1f}>0, C44={c44:.1f}>0, "
          f"C11>|C12|: {c11:.1f}>{abs(c12):.1f}, C11+2C12={c11 + 2 * c12:.1f}>0 "
          f"--> {'PASS' if cubic_ok else 'FAIL'}")
    print(f"  Zener anisotropy ratio: A = {A_zener:.3f} (1.0 = isotropic)")

# ---- STEP 8: Visualization ------------------------------------------------
n_patterns = len(fit_data)
ncols = min(3, n_patterns)
nrows = int(np.ceil(n_patterns / ncols))
fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4 * nrows), squeeze=False)

for p_idx, fdata in enumerate(fit_data):
    ax = axes[p_idx // ncols][p_idx % ncols]

    eps_arr = np.array(fdata["epsilons"])
    dE_arr = np.array(fdata["delta_E_eV"]) * 1000  # Convert to meV

    # Plot data points
    ax.plot(eps_arr * 100, dE_arr, "ko", markersize=6, zorder=5)

    # Plot parabolic fit
    eps_fine = np.linspace(eps_arr.min(), eps_arr.max(), 100)
    dE_fit = (fdata["a2"] * eps_fine**2 + fdata["a1"] * eps_fine + fdata["a0"]) * 1000
    ax.plot(eps_fine * 100, dE_fit, "r-", linewidth=1.5,
            label=f"$C_{{eff}}$ = {fdata['C_eff_GPa']:.1f} GPa\n$R^2$ = {fdata['R_squared']:.5f}")

    ax.set_xlabel("Strain (%)")
    ax.set_ylabel("$\\Delta E$ (meV)")
    ax.set_title(fdata["pattern_name"], fontsize=10)
    ax.legend(fontsize=8, loc="upper center")
    ax.grid(True, alpha=0.3)

# Hide unused subplots
for p_idx in range(n_patterns, nrows * ncols):
    axes[p_idx // ncols][p_idx % ncols].set_visible(False)

fig.suptitle(f"Energy-Strain Curves: {formula} ({space_group})", fontsize=13)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "energy_strain_curves.png"), dpi=150, bbox_inches="tight")
print(f"\nEnergy-strain plot saved to {OUTPUT_DIR}/energy_strain_curves.png")

# ---- Save results to JSON -------------------------------------------------
results = {
    "formula": formula,
    "space_group": space_group,
    "crystal_system": crystal_system,
    "volume_A3": V0,
    "E0_eV": E0,
    "independent_elastic_constants_GPa": cij_dict,
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
    "method": f"MACE-MP-0 ({MACE_MODEL}) energy-strain",
    "strain_magnitudes": STRAIN_MAGNITUDES.tolist(),
    "n_patterns": len(patterns),
    "fit_details": fit_data,
}
with open(os.path.join(OUTPUT_DIR, "elastic_energy_strain.json"), "w") as f:
    json.dump(results, f, indent=2)
print(f"Results saved to {OUTPUT_DIR}/elastic_energy_strain.json")
```

### Method B: QE DFT

This method uses Quantum ESPRESSO `pw.x` to compute total energies at each strained configuration. Unlike the stress-strain method, only the total energy from each SCF calculation is needed -- `tstress=.true.` is optional. The workflow: (1) relax with `vc-relax`, (2) generate strained inputs using symmetry-adapted Lagrangian strain patterns, (3) run `scf` at each deformed geometry, (4) extract total energies, (5) fit E(epsilon) parabolas, (6) solve for C_ij.

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

#### Step B2: Generate strained QE inputs

```python
#!/usr/bin/env python3
"""
Generate QE input files for elastic constant calculation (energy-strain method).
Run after vc-relax to get the equilibrium structure.

Each strain pattern is applied at multiple magnitudes. Only SCF total
energies are needed -- stress tensors are not required.
"""

import os
import json
import numpy as np
from pymatgen.core.structure import Structure
from pymatgen.io.pwscf import PWInput
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer

# ---- CONFIGURATION --------------------------------------------------------
RELAXED_FILE = "relaxed_structure.cif"  # From vc-relax output
PSEUDO_DIR = "./pseudo"
ECUTWFC = 60.0
ECUTRHO = 480.0
K_GRID = [8, 8, 8]
STRAIN_MAGNITUDES = [-0.020, -0.015, -0.010, -0.005, 0.005, 0.010, 0.015, 0.020]
WORK_DIR = "elastic_energy_qe"
SYMPREC = 0.01
# ---------------------------------------------------------------------------

os.makedirs(WORK_DIR, exist_ok=True)

structure = Structure.from_file(RELAXED_FILE)
formula = structure.composition.reduced_formula
V0 = structure.volume

sga = SpacegroupAnalyzer(structure, symprec=SYMPREC)
crystal_system = sga.get_crystal_system().lower()
print(f"Structure: {formula}, V0 = {V0:.4f} A^3")
print(f"Crystal system: {crystal_system}")

# ---- Strain patterns (same definitions as Method A) -----------------------
# For the full set of patterns per crystal system, see Method A above.
# Here we show common systems; extend for monoclinic/triclinic as needed.

STRAIN_PATTERNS_ALL = {
    "cubic": [
        {"name": "D1_volume", "voigt": [1, 1, 1, 0, 0, 0]},
        {"name": "D2_tetra",  "voigt": [1, -1, 0, 0, 0, 0]},
        {"name": "D3_shear",  "voigt": [0, 0, 0, 2, 0, 0]},
    ],
    "hexagonal": [
        {"name": "D1_xx",      "voigt": [1, 0, 0, 0, 0, 0]},
        {"name": "D2_zz",      "voigt": [0, 0, 1, 0, 0, 0]},
        {"name": "D3_biaxial", "voigt": [1, 1, 0, 0, 0, 0]},
        {"name": "D4_xz",      "voigt": [1, 0, 1, 0, 0, 0]},
        {"name": "D5_shear",   "voigt": [0, 0, 0, 2, 0, 0]},
    ],
    "orthorhombic": [
        {"name": "D1_xx", "voigt": [1, 0, 0, 0, 0, 0]},
        {"name": "D2_yy", "voigt": [0, 1, 0, 0, 0, 0]},
        {"name": "D3_zz", "voigt": [0, 0, 1, 0, 0, 0]},
        {"name": "D4_xy", "voigt": [1, 1, 0, 0, 0, 0]},
        {"name": "D5_xz", "voigt": [1, 0, 1, 0, 0, 0]},
        {"name": "D6_yz", "voigt": [0, 1, 1, 0, 0, 0]},
        {"name": "D7_shear_yz", "voigt": [0, 0, 0, 2, 0, 0]},
        {"name": "D8_shear_xz", "voigt": [0, 0, 0, 0, 2, 0]},
        {"name": "D9_shear_xy", "voigt": [0, 0, 0, 0, 0, 2]},
    ],
}

# Fall back to orthorhombic (9 patterns) for systems not explicitly listed
if crystal_system in STRAIN_PATTERNS_ALL:
    patterns = STRAIN_PATTERNS_ALL[crystal_system]
else:
    print(f"No predefined patterns for {crystal_system}, using orthorhombic (9 patterns)")
    patterns = STRAIN_PATTERNS_ALL["orthorhombic"]


def voigt_strain_to_matrix(voigt):
    eta = np.array(voigt, dtype=float)
    return np.array([
        [eta[0],       eta[5]/2.0, eta[4]/2.0],
        [eta[5]/2.0,   eta[1],     eta[3]/2.0],
        [eta[4]/2.0,   eta[3]/2.0, eta[2]],
    ])


def lagrangian_strain_to_deformation(eta_matrix):
    I = np.eye(3)
    C_green = I + 2.0 * eta_matrix
    eigvals, eigvecs = np.linalg.eigh(C_green)
    sqrt_eigvals = np.sqrt(np.maximum(eigvals, 0.0))
    F = eigvecs @ np.diag(sqrt_eigvals) @ eigvecs.T
    return F


# Pseudopotential map -- adjust for your elements
pseudo_map = {}
for el in structure.composition.elements:
    symbol = el.symbol
    pseudo_map[symbol] = f"{symbol}.pbe-n-rrkjus_psl.1.0.0.UPF"

# Generate QE inputs
deformation_info = []
calc_index = 0

for p_idx, pattern in enumerate(patterns):
    voigt_pattern = np.array(pattern["voigt"], dtype=float)
    pattern_name = pattern["name"]

    for eps_mag in STRAIN_MAGNITUDES:
        voigt_scaled = voigt_pattern * eps_mag
        eta_matrix = voigt_strain_to_matrix(voigt_scaled)
        F = lagrangian_strain_to_deformation(eta_matrix)

        deformed = structure.copy()
        new_lattice = np.dot(F, deformed.lattice.matrix)
        deformed.lattice = new_lattice

        calc_dir = os.path.join(WORK_DIR, f"strain_{calc_index:04d}")
        os.makedirs(calc_dir, exist_ok=True)

        # SCF only (no ionic relaxation for simplicity; use 'relax' if
        # internal degrees of freedom are not fixed by symmetry)
        input_params = {
            "CONTROL": {
                "calculation": "scf",
                "prefix": f"strain_{calc_index:04d}",
                "outdir": "./tmp",
                "pseudo_dir": os.path.abspath(PSEUDO_DIR),
                "tprnfor": False,
                "tstress": False,
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
        }

        pw_input = PWInput(
            deformed,
            pseudo=pseudo_map,
            control=input_params["CONTROL"],
            system=input_params["SYSTEM"],
            electrons=input_params["ELECTRONS"],
            kpoints_grid=tuple(K_GRID),
        )
        input_file = os.path.join(calc_dir, "scf.in")
        pw_input.write_file(input_file)

        deformation_info.append({
            "index": calc_index,
            "pattern_name": pattern_name,
            "pattern_index": p_idx,
            "epsilon": eps_mag,
            "voigt_strain": voigt_scaled.tolist(),
            "directory": calc_dir,
        })
        calc_index += 1

# Save metadata
with open(os.path.join(WORK_DIR, "deformation_info.json"), "w") as f:
    json.dump(deformation_info, f, indent=2)

# Generate run script
run_script = "#!/bin/bash\n"
run_script += f"# Energy-strain elastic constants for {formula}\n"
run_script += f"# Total calculations: {calc_index}\n\n"
for idx in range(calc_index):
    run_script += f"echo 'Running calculation {idx+1}/{calc_index}'\n"
    run_script += f"cd {WORK_DIR}/strain_{idx:04d}\n"
    run_script += f"pw.x < scf.in > scf.out 2>&1\n"
    run_script += f"cd ../..\n\n"

with open(os.path.join(WORK_DIR, "run_all.sh"), "w") as f:
    f.write(run_script)
os.chmod(os.path.join(WORK_DIR, "run_all.sh"), 0o755)

print(f"\nGenerated {calc_index} QE input files in {WORK_DIR}/")
print(f"Run: bash {WORK_DIR}/run_all.sh")
```

#### Step B3: Extract energies, fit parabolas, solve for C_ij

```python
#!/usr/bin/env python3
"""
Post-process QE outputs for energy-strain elastic constants.
Extracts total energy from each SCF output, fits E(eps) parabolas,
and solves for the independent elastic constants.
"""

import os
import json
import re
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from pymatgen.core.structure import Structure
from pymatgen.analysis.elasticity import ElasticTensor
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer

# ---- CONFIGURATION --------------------------------------------------------
RELAXED_FILE = "relaxed_structure.cif"
WORK_DIR = "elastic_energy_qe"
OUTPUT_DIR = "elastic_energy_results_qe"
SYMPREC = 0.01
# ---------------------------------------------------------------------------

os.makedirs(OUTPUT_DIR, exist_ok=True)

structure = Structure.from_file(RELAXED_FILE)
formula = structure.composition.reduced_formula
V0 = structure.volume

sga = SpacegroupAnalyzer(structure, symprec=SYMPREC)
crystal_system = sga.get_crystal_system().lower()
print(f"Structure: {formula}, V0 = {V0:.4f} A^3, crystal system: {crystal_system}")

# ---- C_eff coefficient definitions per crystal system --------------------
# Must match the patterns used in Step B2.
C_EFF_DEFS = {
    "cubic": {
        "patterns": [
            {"name": "D1_volume", "c_eff_coeffs": {"C11": 1, "C12": 2}},
            {"name": "D2_tetra",  "c_eff_coeffs": {"C11": 1, "C12": -1}},
            {"name": "D3_shear",  "c_eff_coeffs": {"C44": 1}},
        ],
        "independent": ["C11", "C12", "C44"],
    },
    "hexagonal": {
        "patterns": [
            {"name": "D1_xx",      "c_eff_coeffs": {"C11": 1}},
            {"name": "D2_zz",      "c_eff_coeffs": {"C33": 1}},
            {"name": "D3_biaxial", "c_eff_coeffs": {"C11": 1, "C12": 1}},
            {"name": "D4_xz",      "c_eff_coeffs": {"C11": 1, "C33": 1, "C13": 2}},
            {"name": "D5_shear",   "c_eff_coeffs": {"C44": 1}},
        ],
        "independent": ["C11", "C12", "C13", "C33", "C44"],
    },
    "orthorhombic": {
        "patterns": [
            {"name": "D1_xx", "c_eff_coeffs": {"C11": 1}},
            {"name": "D2_yy", "c_eff_coeffs": {"C22": 1}},
            {"name": "D3_zz", "c_eff_coeffs": {"C33": 1}},
            {"name": "D4_xy", "c_eff_coeffs": {"C11": 1, "C22": 1, "C12": 2}},
            {"name": "D5_xz", "c_eff_coeffs": {"C11": 1, "C33": 1, "C13": 2}},
            {"name": "D6_yz", "c_eff_coeffs": {"C22": 1, "C33": 1, "C23": 2}},
            {"name": "D7_shear_yz", "c_eff_coeffs": {"C44": 1}},
            {"name": "D8_shear_xz", "c_eff_coeffs": {"C55": 1}},
            {"name": "D9_shear_xy", "c_eff_coeffs": {"C66": 1}},
        ],
        "independent": ["C11", "C12", "C13", "C22", "C23", "C33",
                         "C44", "C55", "C66"],
    },
}


def parse_qe_energy(output_file):
    """Extract total energy (eV) from a QE pw.x output file."""
    energy_ry = None
    with open(output_file) as f:
        for line in f:
            if line.strip().startswith("!"):
                match = re.search(r"total energy\s*=\s*([-\d.]+)\s*Ry", line)
                if match:
                    energy_ry = float(match.group(1))
    if energy_ry is None:
        raise ValueError(f"Could not parse energy from {output_file}")
    return energy_ry * 13.605693123  # Ry to eV


def solve_elastic_constants(c_eff_values, pattern_defs, independent_names):
    n_patterns = len(c_eff_values)
    n_unknowns = len(independent_names)
    A = np.zeros((n_patterns, n_unknowns))
    b = np.array(c_eff_values)
    name_to_idx = {name: i for i, name in enumerate(independent_names)}
    for p_idx, pdef in enumerate(pattern_defs):
        for cname, coeff in pdef["c_eff_coeffs"].items():
            if cname in name_to_idx:
                A[p_idx, name_to_idx[cname]] = coeff
    x, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
    return {name: x[i] for i, name in enumerate(independent_names)}


def build_voigt_matrix(cij_dict):
    C = np.zeros((6, 6))
    voigt_map = {
        "C11": (0,0), "C12": (0,1), "C13": (0,2), "C14": (0,3), "C15": (0,4), "C16": (0,5),
        "C22": (1,1), "C23": (1,2), "C24": (1,3), "C25": (1,4), "C26": (1,5),
        "C33": (2,2), "C34": (2,3), "C35": (2,4), "C36": (2,5),
        "C44": (3,3), "C45": (3,4), "C46": (3,5),
        "C55": (4,4), "C56": (4,5), "C66": (5,5),
    }
    for name, val in cij_dict.items():
        if name in voigt_map:
            i, j = voigt_map[name]
            C[i, j] = val
            C[j, i] = val
    return C


# ---- Get equilibrium energy from the relaxed calculation -------------------
eq_out = os.path.join(os.path.dirname(WORK_DIR), "relax.out")
try:
    E0 = parse_qe_energy(eq_out)
    print(f"Equilibrium energy: {E0:.6f} eV")
except Exception:
    # Estimate from the minimum energy in the data set
    E0 = None
    print("WARNING: Could not find equilibrium energy; will estimate from data.")

# ---- Load deformation metadata and extract energies -----------------------
with open(os.path.join(WORK_DIR, "deformation_info.json")) as f:
    deformation_info = json.load(f)

# Group by pattern
pattern_data = {}
for info in deformation_info:
    pname = info["pattern_name"]
    if pname not in pattern_data:
        pattern_data[pname] = {"epsilons": [], "energies": []}

    output_file = os.path.join(info["directory"], "scf.out")
    if not os.path.exists(output_file):
        print(f"  WARNING: {output_file} not found, skipping")
        continue

    try:
        E = parse_qe_energy(output_file)
        pattern_data[pname]["epsilons"].append(info["epsilon"])
        pattern_data[pname]["energies"].append(E)
    except ValueError as e:
        print(f"  WARNING: {e}")

# Estimate E0 if not available
if E0 is None:
    all_energies = []
    for pdata in pattern_data.values():
        all_energies.extend(pdata["energies"])
    E0 = min(all_energies) if all_energies else 0.0
    print(f"Estimated E0 = {E0:.6f} eV (minimum across all strains)")

# ---- Fit parabolas and extract C_eff for each pattern ---------------------
cs_key = crystal_system if crystal_system in C_EFF_DEFS else "orthorhombic"
ceff_defs = C_EFF_DEFS[cs_key]
pattern_list = ceff_defs["patterns"]
independent_names = ceff_defs["independent"]

c_eff_values = []
fit_data = []

for pdef in pattern_list:
    pname = pdef["name"]
    if pname not in pattern_data or len(pattern_data[pname]["epsilons"]) < 3:
        print(f"  ERROR: Insufficient data for pattern {pname}")
        c_eff_values.append(0.0)
        continue

    eps_arr = np.array(pattern_data[pname]["epsilons"])
    E_arr = np.array(pattern_data[pname]["energies"])
    dE_arr = E_arr - E0

    # Sort by strain
    sort_idx = np.argsort(eps_arr)
    eps_arr = eps_arr[sort_idx]
    dE_arr = dE_arr[sort_idx]

    # Fit: dE = a2*eps^2 + a1*eps + a0
    coeffs = np.polyfit(eps_arr, dE_arr, 2)
    a2, a1, a0 = coeffs

    C_eff_GPa = (2.0 * a2 / V0) * 160.21766

    dE_fit = np.polyval(coeffs, eps_arr)
    ss_res = np.sum((dE_arr - dE_fit)**2)
    ss_tot = np.sum((dE_arr - np.mean(dE_arr))**2)
    R2 = 1.0 - ss_res / ss_tot if ss_tot > 1e-30 else 1.0

    c_eff_values.append(C_eff_GPa)
    fit_data.append({
        "pattern_name": pname,
        "epsilons": eps_arr.tolist(),
        "delta_E_eV": dE_arr.tolist(),
        "C_eff_GPa": C_eff_GPa,
        "R_squared": R2,
        "a2": float(a2), "a1": float(a1), "a0": float(a0),
    })

    print(f"  {pname}: C_eff = {C_eff_GPa:.2f} GPa, R^2 = {R2:.6f}")

# ---- Solve for independent C_ij ------------------------------------------
cij_dict = solve_elastic_constants(c_eff_values, pattern_list, independent_names)
C_matrix = build_voigt_matrix(cij_dict)

print("\n" + "=" * 60)
print("ELASTIC TENSOR (Voigt notation, GPa) -- QE energy-strain:")
print("=" * 60)
for i in range(6):
    row = "  ".join(f"{C_matrix[i, j]:8.2f}" for j in range(6))
    print(f"  [{row}]")

for name in independent_names:
    print(f"  {name} = {cij_dict[name]:.2f} GPa")

# ---- Polycrystalline moduli -----------------------------------------------
C_tensor = ElasticTensor.from_voigt(C_matrix)
K_vrh = C_tensor.k_vrh
G_vrh = C_tensor.g_vrh
E_vrh = 9*K_vrh*G_vrh/(3*K_vrh+G_vrh) if (3*K_vrh+G_vrh) > 0 else 0
nu_vrh = (3*K_vrh-2*G_vrh)/(6*K_vrh+2*G_vrh) if (6*K_vrh+2*G_vrh) > 0 else 0

print(f"\nPolycrystalline Moduli (VRH, GPa):")
print(f"  Bulk modulus  K = {K_vrh:.2f}")
print(f"  Shear modulus G = {G_vrh:.2f}")
print(f"  Young's modulus E = {E_vrh:.2f}")
print(f"  Poisson's ratio nu = {nu_vrh:.4f}")

# Born stability
eigenvalues = np.linalg.eigvalsh(C_matrix)
is_stable = all(ev > 0 for ev in eigenvalues)
print(f"\nBorn Stability: eigenvalues = {[f'{ev:.2f}' for ev in eigenvalues]}")
print(f"  Mechanically stable: {'YES' if is_stable else 'NO'}")

# ---- Visualization -------------------------------------------------------
n_patterns_plot = len(fit_data)
ncols = min(3, n_patterns_plot)
nrows = int(np.ceil(n_patterns_plot / ncols))
fig, axes = plt.subplots(nrows, ncols, figsize=(5*ncols, 4*nrows), squeeze=False)

for p_idx, fdata in enumerate(fit_data):
    ax = axes[p_idx // ncols][p_idx % ncols]
    eps_arr = np.array(fdata["epsilons"])
    dE_arr = np.array(fdata["delta_E_eV"]) * 1000  # meV

    ax.plot(eps_arr * 100, dE_arr, "ko", markersize=6, zorder=5)
    eps_fine = np.linspace(eps_arr.min(), eps_arr.max(), 100)
    dE_fit = (fdata["a2"]*eps_fine**2 + fdata["a1"]*eps_fine + fdata["a0"]) * 1000
    ax.plot(eps_fine * 100, dE_fit, "r-", lw=1.5,
            label=f"$C_{{eff}}$={fdata['C_eff_GPa']:.1f} GPa\n$R^2$={fdata['R_squared']:.5f}")
    ax.set_xlabel("Strain (%)")
    ax.set_ylabel("$\\Delta E$ (meV)")
    ax.set_title(fdata["pattern_name"], fontsize=10)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

for p_idx in range(n_patterns_plot, nrows * ncols):
    axes[p_idx // ncols][p_idx % ncols].set_visible(False)

fig.suptitle(f"Energy-Strain (QE): {formula}", fontsize=13)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "energy_strain_qe.png"), dpi=150, bbox_inches="tight")
print(f"\nPlot saved to {OUTPUT_DIR}/energy_strain_qe.png")

# ---- Save results ---------------------------------------------------------
results = {
    "formula": formula,
    "crystal_system": crystal_system,
    "volume_A3": V0,
    "E0_eV": E0,
    "independent_elastic_constants_GPa": cij_dict,
    "elastic_tensor_GPa": C_matrix.tolist(),
    "K_VRH_GPa": K_vrh,
    "G_VRH_GPa": G_vrh,
    "E_VRH_GPa": E_vrh,
    "Poisson_ratio_VRH": nu_vrh,
    "eigenvalues_C": eigenvalues.tolist(),
    "is_mechanically_stable": bool(is_stable),
    "method": "QE PBE (energy-strain)",
    "fit_details": fit_data,
}
with open(os.path.join(OUTPUT_DIR, "elastic_energy_strain.json"), "w") as f:
    json.dump(results, f, indent=2)
print(f"Results saved to {OUTPUT_DIR}/elastic_energy_strain.json")
```

### Method C: VASP

This method uses VASP to compute total energies at each strained configuration. VASPKIT (menu 02, tasks 204--207) automates the entire workflow. Alternatively, the workflow below generates strained POSCARs, runs single-point VASP calculations, parses OUTCAR energies, and fits parabolas -- useful when VASPKIT is not available or for custom strain patterns.

#### VASPKIT Automated Workflow (Tasks 204--207)

VASPKIT provides a fully automated energy-strain elastic constant workflow:

```bash
# Step 1: Prepare equilibrium POSCAR (must be fully relaxed, ISIF=3)
# Step 2: Launch VASPKIT
vaspkit

# In the VASPKIT menu:
#   2) Elastic Properties
#     204) Elastic Constants from Energy-Strain Approach (2D)
#     205) Elastic Constants from Energy-Strain Approach (3D)
#     206) Generate Strained Structures
#     207) Post-process Energy-Strain Data

# For 3D elastic constants:
#   Select 205 -> follow prompts for strain range and number of points
#   VASPKIT generates strained POSCAR directories
#   Run VASP in each directory
#   Return to VASPKIT 207 to extract and fit
```

#### Manual VASP Workflow

```python
#!/usr/bin/env python3
"""
Energy-strain elastic constants with VASP (manual workflow).

Generates strained POSCARs, VASP input sets (INCAR, KPOINTS, POTCAR via
pymatgen), runs single-point calculations, parses total energies from OUTCAR,
fits E(eps) parabolas, and solves for C_ij.

Requires: pymatgen, numpy, scipy, matplotlib.
VASP must be in PATH as 'vasp_std' (or modify RUN_COMMAND).
"""

import os
import json
import subprocess
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from pymatgen.core.structure import Structure
from pymatgen.io.vasp import Poscar, Incar, Kpoints
from pymatgen.analysis.elasticity import ElasticTensor
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer

# ---- CONFIGURATION --------------------------------------------------------
INPUT_FILE = "POSCAR"                  # Fully relaxed equilibrium structure
STRAIN_MAGNITUDES = np.array([-0.020, -0.015, -0.010, -0.005,
                               0.005,  0.010,  0.015,  0.020])
WORK_DIR = "elastic_energy_vasp"
OUTPUT_DIR = "elastic_energy_results_vasp"
RUN_COMMAND = "vasp_std"               # VASP executable
SYMPREC = 0.01
ENCUT = 520                            # Plane-wave cutoff (eV)
# ---------------------------------------------------------------------------

os.makedirs(WORK_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ---- Load relaxed structure -----------------------------------------------
structure = Structure.from_file(INPUT_FILE)
formula = structure.composition.reduced_formula
V0 = structure.volume

sga = SpacegroupAnalyzer(structure, symprec=SYMPREC)
crystal_system = sga.get_crystal_system().lower()
print(f"Structure: {formula}, V0 = {V0:.4f} A^3")
print(f"Crystal system: {crystal_system}")

# ---- Strain patterns (select by crystal system) ---------------------------
STRAIN_PATTERNS_ALL = {
    "cubic": [
        {"name": "D1_volume", "voigt": [1, 1, 1, 0, 0, 0],
         "c_eff_coeffs": {"C11": 1, "C12": 2}},
        {"name": "D2_tetra",  "voigt": [1, -1, 0, 0, 0, 0],
         "c_eff_coeffs": {"C11": 1, "C12": -1}},
        {"name": "D3_shear",  "voigt": [0, 0, 0, 2, 0, 0],
         "c_eff_coeffs": {"C44": 1}},
    ],
    "hexagonal": [
        {"name": "D1_xx",      "voigt": [1, 0, 0, 0, 0, 0],
         "c_eff_coeffs": {"C11": 1}},
        {"name": "D2_zz",      "voigt": [0, 0, 1, 0, 0, 0],
         "c_eff_coeffs": {"C33": 1}},
        {"name": "D3_biaxial", "voigt": [1, 1, 0, 0, 0, 0],
         "c_eff_coeffs": {"C11": 1, "C12": 1}},
        {"name": "D4_xz",      "voigt": [1, 0, 1, 0, 0, 0],
         "c_eff_coeffs": {"C11": 1, "C33": 1, "C13": 2}},
        {"name": "D5_shear",   "voigt": [0, 0, 0, 2, 0, 0],
         "c_eff_coeffs": {"C44": 1}},
    ],
    "orthorhombic": [
        {"name": "D1_xx", "voigt": [1,0,0,0,0,0], "c_eff_coeffs": {"C11": 1}},
        {"name": "D2_yy", "voigt": [0,1,0,0,0,0], "c_eff_coeffs": {"C22": 1}},
        {"name": "D3_zz", "voigt": [0,0,1,0,0,0], "c_eff_coeffs": {"C33": 1}},
        {"name": "D4_xy", "voigt": [1,1,0,0,0,0],
         "c_eff_coeffs": {"C11": 1, "C22": 1, "C12": 2}},
        {"name": "D5_xz", "voigt": [1,0,1,0,0,0],
         "c_eff_coeffs": {"C11": 1, "C33": 1, "C13": 2}},
        {"name": "D6_yz", "voigt": [0,1,1,0,0,0],
         "c_eff_coeffs": {"C22": 1, "C33": 1, "C23": 2}},
        {"name": "D7_shear_yz", "voigt": [0,0,0,2,0,0], "c_eff_coeffs": {"C44": 1}},
        {"name": "D8_shear_xz", "voigt": [0,0,0,0,2,0], "c_eff_coeffs": {"C55": 1}},
        {"name": "D9_shear_xy", "voigt": [0,0,0,0,0,2], "c_eff_coeffs": {"C66": 1}},
    ],
}

cs_key = crystal_system if crystal_system in STRAIN_PATTERNS_ALL else "orthorhombic"
patterns = STRAIN_PATTERNS_ALL[cs_key]
independent_names = list(set(
    k for p in patterns for k in p["c_eff_coeffs"].keys()
))
independent_names.sort()


def voigt_strain_to_matrix(voigt):
    eta = np.array(voigt, dtype=float)
    return np.array([
        [eta[0],       eta[5]/2.0, eta[4]/2.0],
        [eta[5]/2.0,   eta[1],     eta[3]/2.0],
        [eta[4]/2.0,   eta[3]/2.0, eta[2]],
    ])


def lagrangian_strain_to_deformation(eta_matrix):
    I = np.eye(3)
    C_green = I + 2.0 * eta_matrix
    eigvals, eigvecs = np.linalg.eigh(C_green)
    sqrt_eigvals = np.sqrt(np.maximum(eigvals, 0.0))
    return eigvecs @ np.diag(sqrt_eigvals) @ eigvecs.T


# ---- INCAR for single-point energy calculation ----------------------------
incar_dict = {
    "SYSTEM": formula,
    "PREC": "Accurate",
    "ENCUT": ENCUT,
    "EDIFF": 1e-7,           # Tight SCF convergence for accurate energies
    "ISMEAR": 1,             # Methfessel-Paxton for metals; use 0 for insulators
    "SIGMA": 0.1,
    "ISIF": 2,               # Calculate stress and forces, fixed cell shape
    "NSW": 0,                # Single-point (no ionic relaxation)
    "IBRION": -1,            # No ionic update
    "LREAL": "Auto",
    "LWAVE": False,
    "LCHARG": False,
    "NELM": 200,
}

# ---- Generate strained structures and VASP inputs -------------------------
calc_info = []
calc_index = 0

# Also run equilibrium (zero-strain) for reference energy
eq_dir = os.path.join(WORK_DIR, "eq_0000")
os.makedirs(eq_dir, exist_ok=True)
Poscar(structure).write_file(os.path.join(eq_dir, "POSCAR"))
Incar(incar_dict).write_file(os.path.join(eq_dir, "INCAR"))
Kpoints.automatic_density(structure, kppa=5000).write_file(
    os.path.join(eq_dir, "KPOINTS"))
calc_info.append({
    "index": -1, "pattern_name": "equilibrium",
    "epsilon": 0.0, "directory": eq_dir
})

for p_idx, pattern in enumerate(patterns):
    voigt_pattern = np.array(pattern["voigt"], dtype=float)
    for eps_mag in STRAIN_MAGNITUDES:
        voigt_scaled = voigt_pattern * eps_mag
        eta_matrix = voigt_strain_to_matrix(voigt_scaled)
        F = lagrangian_strain_to_deformation(eta_matrix)

        deformed = structure.copy()
        new_lattice = np.dot(F, deformed.lattice.matrix)
        deformed.lattice = new_lattice

        calc_dir = os.path.join(WORK_DIR, f"strain_{calc_index:04d}")
        os.makedirs(calc_dir, exist_ok=True)

        Poscar(deformed).write_file(os.path.join(calc_dir, "POSCAR"))
        Incar(incar_dict).write_file(os.path.join(calc_dir, "INCAR"))
        Kpoints.automatic_density(deformed, kppa=5000).write_file(
            os.path.join(calc_dir, "KPOINTS"))
        # NOTE: POTCAR must be generated separately (pymatgen needs
        # VASP_PSP_DIR set), or copy from a template.

        calc_info.append({
            "index": calc_index,
            "pattern_name": pattern["name"],
            "pattern_index": p_idx,
            "epsilon": float(eps_mag),
            "directory": calc_dir,
        })
        calc_index += 1

with open(os.path.join(WORK_DIR, "calc_info.json"), "w") as f:
    json.dump(calc_info, f, indent=2)

# Run script
run_script = "#!/bin/bash\n"
run_script += f"# Energy-strain for {formula}: {calc_index} strained + 1 equilibrium\n\n"
run_script += f"echo 'Running equilibrium'\ncd {eq_dir}\n{RUN_COMMAND}\ncd ../..\n\n"
for idx in range(calc_index):
    run_script += f"echo 'Running strain {idx+1}/{calc_index}'\n"
    run_script += f"cd {WORK_DIR}/strain_{idx:04d}\n{RUN_COMMAND}\ncd ../..\n\n"

with open(os.path.join(WORK_DIR, "run_all.sh"), "w") as f:
    f.write(run_script)
os.chmod(os.path.join(WORK_DIR, "run_all.sh"), 0o755)

print(f"Generated {calc_index + 1} VASP calculations in {WORK_DIR}/")
print(f"Generate POTCARs, then run: bash {WORK_DIR}/run_all.sh")


# ===========================================================================
# POST-PROCESSING (run after all VASP calculations finish)
# ===========================================================================

def parse_outcar_energy(outcar_path):
    """Extract 'energy without entropy' (sigma->0) from OUTCAR."""
    energy = None
    with open(outcar_path) as f:
        for line in f:
            if "energy  without entropy" in line:
                parts = line.split()
                energy = float(parts[-1])  # Last value on the line (eV)
    if energy is None:
        raise ValueError(f"Could not parse energy from {outcar_path}")
    return energy


def postprocess_vasp():
    """Parse OUTCAR files, fit parabolas, and compute elastic tensor."""
    with open(os.path.join(WORK_DIR, "calc_info.json")) as f:
        calc_info_loaded = json.load(f)

    # Get equilibrium energy
    E0_vasp = None
    for info in calc_info_loaded:
        if info["pattern_name"] == "equilibrium":
            outcar_path = os.path.join(info["directory"], "OUTCAR")
            E0_vasp = parse_outcar_energy(outcar_path)
            print(f"Equilibrium energy: {E0_vasp:.6f} eV")
            break

    if E0_vasp is None:
        raise RuntimeError("Equilibrium OUTCAR not found.")

    # Group by pattern
    pattern_data_vasp = {}
    for info in calc_info_loaded:
        if info["pattern_name"] == "equilibrium":
            continue
        pname = info["pattern_name"]
        if pname not in pattern_data_vasp:
            pattern_data_vasp[pname] = {"epsilons": [], "energies": []}
        outcar_path = os.path.join(info["directory"], "OUTCAR")
        try:
            E = parse_outcar_energy(outcar_path)
            pattern_data_vasp[pname]["epsilons"].append(info["epsilon"])
            pattern_data_vasp[pname]["energies"].append(E)
        except ValueError as e:
            print(f"  WARNING: {e}")

    # Fit parabolas
    c_eff_values_vasp = []
    fit_data_vasp = []

    for pdef in patterns:
        pname = pdef["name"]
        pdata = pattern_data_vasp.get(pname, {"epsilons": [], "energies": []})
        if len(pdata["epsilons"]) < 3:
            print(f"  ERROR: Insufficient data for {pname}")
            c_eff_values_vasp.append(0.0)
            continue

        eps_arr = np.array(pdata["epsilons"])
        E_arr = np.array(pdata["energies"])
        dE_arr = E_arr - E0_vasp

        sort_idx = np.argsort(eps_arr)
        eps_arr = eps_arr[sort_idx]
        dE_arr = dE_arr[sort_idx]

        coeffs = np.polyfit(eps_arr, dE_arr, 2)
        a2, a1, a0 = coeffs
        C_eff_GPa = (2.0 * a2 / V0) * 160.21766

        dE_fit = np.polyval(coeffs, eps_arr)
        ss_res = np.sum((dE_arr - dE_fit)**2)
        ss_tot = np.sum((dE_arr - np.mean(dE_arr))**2)
        R2 = 1.0 - ss_res / ss_tot if ss_tot > 1e-30 else 1.0

        c_eff_values_vasp.append(C_eff_GPa)
        fit_data_vasp.append({
            "pattern_name": pname,
            "epsilons": eps_arr.tolist(),
            "delta_E_eV": dE_arr.tolist(),
            "C_eff_GPa": C_eff_GPa,
            "R_squared": R2,
        })
        print(f"  {pname}: C_eff = {C_eff_GPa:.2f} GPa, R^2 = {R2:.6f}")

    # Solve for C_ij
    def solve_cij(c_eff_vals, pat_defs, ind_names):
        n_p = len(c_eff_vals)
        n_u = len(ind_names)
        A_mat = np.zeros((n_p, n_u))
        name_idx = {n: i for i, n in enumerate(ind_names)}
        for pi, pd in enumerate(pat_defs):
            for cn, co in pd["c_eff_coeffs"].items():
                if cn in name_idx:
                    A_mat[pi, name_idx[cn]] = co
        x, _, _, _ = np.linalg.lstsq(A_mat, np.array(c_eff_vals), rcond=None)
        return {n: x[i] for i, n in enumerate(ind_names)}

    cij_vasp = solve_cij(c_eff_values_vasp, patterns, independent_names)
    C_mat_vasp = build_voigt_matrix(cij_vasp)

    print("\n" + "=" * 60)
    print("ELASTIC TENSOR (GPa) -- VASP energy-strain:")
    print("=" * 60)
    for i in range(6):
        row = "  ".join(f"{C_mat_vasp[i,j]:8.2f}" for j in range(6))
        print(f"  [{row}]")

    # Moduli
    C_t = ElasticTensor.from_voigt(C_mat_vasp)
    K_v = C_t.k_vrh
    G_v = C_t.g_vrh
    E_v = 9*K_v*G_v/(3*K_v+G_v) if (3*K_v+G_v) > 0 else 0
    nu_v = (3*K_v-2*G_v)/(6*K_v+2*G_v) if (6*K_v+2*G_v) > 0 else 0

    print(f"\nPolycrystalline: K={K_v:.2f}, G={G_v:.2f}, E={E_v:.2f} GPa, nu={nu_v:.4f}")

    # Visualization
    n_p = len(fit_data_vasp)
    nc = min(3, n_p)
    nr = int(np.ceil(n_p / nc))
    fig, axes = plt.subplots(nr, nc, figsize=(5*nc, 4*nr), squeeze=False)

    for pi, fd in enumerate(fit_data_vasp):
        ax = axes[pi // nc][pi % nc]
        ea = np.array(fd["epsilons"])
        da = np.array(fd["delta_E_eV"]) * 1000
        ax.plot(ea * 100, da, "ko", markersize=6, zorder=5)
        ef = np.linspace(ea.min(), ea.max(), 100)
        cf = np.polyfit(ea, np.array(fd["delta_E_eV"]), 2)
        df = np.polyval(cf, ef) * 1000
        ax.plot(ef * 100, df, "r-", lw=1.5,
                label=f"$C_{{eff}}$={fd['C_eff_GPa']:.1f} GPa\n$R^2$={fd['R_squared']:.5f}")
        ax.set_xlabel("Strain (%)")
        ax.set_ylabel("$\\Delta E$ (meV)")
        ax.set_title(fd["pattern_name"], fontsize=10)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    for pi in range(n_p, nr * nc):
        axes[pi // nc][pi % nc].set_visible(False)

    fig.suptitle(f"Energy-Strain (VASP): {formula}", fontsize=13)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "energy_strain_vasp.png"), dpi=150, bbox_inches="tight")
    print(f"Plot saved to {OUTPUT_DIR}/energy_strain_vasp.png")

    # Save
    out = {
        "formula": formula, "crystal_system": crystal_system,
        "volume_A3": V0, "E0_eV": E0_vasp,
        "elastic_tensor_GPa": C_mat_vasp.tolist(),
        "independent_Cij_GPa": cij_vasp,
        "K_VRH_GPa": K_v, "G_VRH_GPa": G_v,
        "E_VRH_GPa": E_v, "Poisson_ratio_VRH": nu_v,
        "method": "VASP PBE (energy-strain)",
        "fit_details": fit_data_vasp,
    }
    with open(os.path.join(OUTPUT_DIR, "elastic_energy_strain_vasp.json"), "w") as f:
        json.dump(out, f, indent=2)
    print(f"Results saved to {OUTPUT_DIR}/elastic_energy_strain_vasp.json")


# Uncomment the line below after all VASP calculations are complete:
# postprocess_vasp()
```

## Key Parameters

| Parameter | Value | Notes |
|---|---|---|
| Strain magnitude range | -0.02 to +0.02 (i.e., +/-2%) | Must be small enough for the parabolic approximation to hold but large enough for the energy differences to exceed numerical noise. |
| Number of strain magnitudes | 6--8 per pattern | More points improve the parabolic fit and allow detection of nonlinearity. Minimum 5 recommended. |
| Strain type | Lagrangian (finite) | The deformation gradient F is computed exactly from eta via F = sqrt(I + 2*eta). Infinitesimal strain is only valid for eps << 1%. |
| Strain patterns per crystal system | Cubic: 3, Hexagonal: 5, Tetragonal: 6, Orthorhombic: 9, Monoclinic: 13, Triclinic: 21 | Each pattern probes a specific linear combination of C_ij. More patterns than unknowns enables least-squares fitting. |
| FMAX (MACE ionic relaxation) | 1e-4 eV/A (bulk), 1e-3 eV/A (strained) | Tight bulk relaxation ensures E0 is at the true minimum; strained calculations can use looser tolerance. |
| ecutwfc (QE) | 60--80 Ry | Energies converge faster than stresses, so moderate cutoffs are often sufficient. Still must be converged. |
| ENCUT (VASP) | 1.3x ENMAX or 520 eV | Use a consistent value across all strain calculations. |
| conv_thr / EDIFF | QE: 1e-10 Ry; VASP: 1e-7 eV | Energy differences between strained structures can be meV-scale; tight convergence is essential. |
| k-grid | Dense, identical for all strains | k-point sampling must be consistent across all calculations to avoid systematic errors. |
| Polynomial order | 2 (parabolic) | Quadratic fit captures the leading elastic energy term. Include a linear term to check for residual stress. A cubic term can be added for anharmonicity diagnostics. |
| ISIF (VASP) | 2 (with NSW=0) | Single-point calculation; ISIF=2 computes stress and forces but does not update the cell. |

## Interpreting Results

**Parabolic fit quality (R-squared):**
- R^2 > 0.9999 indicates an excellent fit with negligible noise.
- R^2 < 0.999 may indicate unconverged calculations, too-large strains, or residual stress.
- A nonzero linear coefficient a1 signals that the structure was not fully relaxed at zero strain. The ratio |a1/a2| should be < 0.01.

**Comparing energy-strain with stress-strain:**
- Both methods should yield the same C_ij within numerical precision (typically < 1% for MACE, < 0.1% for well-converged DFT).
- The energy-strain method requires more strain magnitudes (6--8 per pattern for a good parabolic fit) compared to stress-strain (which only needs 2--4 per direction for a linear fit).
- Energy-strain is more robust when stress tensor computation is problematic (e.g., some all-electron codes, ultrasoft pseudopotentials at low cutoff).
- Stress-strain provides the full 6x6 tensor from a single set of 6 deformation types; energy-strain requires pattern-specific deformations matched to the crystal system.

**Elastic tensor validation:**
- The tensor must be symmetric: C_ij = C_ji. Asymmetry beyond ~0.1 GPa indicates fitting issues.
- Eigenvalues of C_ij must all be positive for mechanical stability (Born criteria).
- Compare with known values from the Materials Project or literature.

**Polycrystalline moduli (Voigt-Reuss-Hill):**
- Voigt = upper bound, Reuss = lower bound, VRH = arithmetic average of both.
- A large gap between Voigt and Reuss indicates strong elastic anisotropy.
- Pugh's ratio K/G > 1.75 suggests ductile behavior.

## Common Issues

| Issue | Solution |
|---|---|
| Parabola has a significant linear term (a1 >> 0) | The equilibrium structure is not fully relaxed. Re-relax with tighter force and stress convergence before applying strains. |
| R^2 of the parabolic fit is poor (< 0.999) | Increase the number of strain magnitudes. Check that all SCF calculations converged. Reduce strain range if nonlinearity is the cause. |
| Energy differences are in the noise floor | Strain magnitudes are too small for the given SCF convergence. Either increase strain to +/-2% or tighten conv_thr / EDIFF. |
| Negative eigenvalues of C_ij | Structure may be mechanically unstable, or the fit is corrupted by poorly converged points. Remove outlier points and re-fit. If still unstable, the phase is genuinely unstable at 0 K. |
| Off-diagonal C_ij not resolved (e.g., C12 noisy) | Off-diagonal constants are obtained by subtraction (e.g., C12 = C_eff(D1) - C11). Error amplification occurs. Use more strain magnitudes and tighter convergence. |
| C_ij differ significantly between MACE and DFT | Expected for materials outside the MACE training domain. Trust DFT and use MACE only for screening. |
| VASPKIT 205 gives different results than manual fit | VASPKIT may use a different strain convention or polynomial order. Verify the strain patterns and ensure the same number of points. Check VASPKIT version >= 1.3. |
| Parabola opens downward (negative curvature) | The energy minimum is not at zero strain -- the structure is not at equilibrium. Or the strain pattern has the wrong sign convention. Re-relax the structure. |
| Too many calculations for low-symmetry systems | Triclinic requires 21 patterns x 8 magnitudes = 168 calculations. Use MACE for initial screening, then run DFT only for the few patterns that are most uncertain. |
| Inconsistent k-point grids across strained cells | Strained cells have different shapes, so automatic k-mesh generation may give different grids. Use a fixed k-spacing (KSPACING in VASP, or explicit K_POINTS in QE) to ensure consistency. |
