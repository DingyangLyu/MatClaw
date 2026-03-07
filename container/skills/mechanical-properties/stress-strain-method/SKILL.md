# Stress-Strain Method for Elastic Constants

## When to Use

- You need the full 6x6 elastic stiffness tensor C_ij by applying finite strains and measuring the resulting stress response.
- You want polycrystalline moduli (bulk K, shear G, Young's E, Poisson's ratio nu) via Voigt-Reuss-Hill averaging.
- You need to verify mechanical (Born) stability of a crystal structure at 0 K.
- You want to screen elastic properties rapidly with MACE before committing to expensive DFT.
- You want VASPKIT-compatible workflows (menu 02, tasks 200-203) for stress-strain post-processing.
- You prefer the stress-strain approach over the energy-strain method because it requires fewer strain points (linear fit vs. polynomial).

## Method Selection

| Criterion | ASE+MACE | QE DFT | VASP |
|---|---|---|---|
| Speed | Seconds | Hours to days | Hours to days |
| Accuracy | Good within MACE training domain | Publication quality (PBE/PBEsol) | Publication quality (PBE/PBEsol/SCAN) |
| Stress output | `atoms.get_stress()` (analytic) | `tstress=.true.` in pw.x | OUTCAR (default with ISIF>=1) |
| Best for | Screening, rapid prototyping, large-scale studies | Standard DFT without VASP license | VASP users, VASPKIT 200-203 automation |
| Post-processing | pymatgen ElasticTensor | pymatgen ElasticTensor | VASPKIT task 200 or pymatgen |
| Limitations | Accuracy bounded by training data | Stress converges slower than energy | Requires VASP license and POTCAR library |

Use MACE first for a quick estimate. If the material is exotic or elastic constants look suspicious, follow up with QE or VASP DFT.

## Prerequisites

- A relaxed crystal structure (CIF, POSCAR, or pymatgen Structure) at zero stress (equilibrium).
- For QE: pseudopotential files (SSSP recommended). See `electronic-structure/scf-relax/SKILL.md`.
- For VASP: POTCAR library, VASP executable, optionally VASPKIT for tasks 200-203.
- Python packages: `pymatgen`, `ase`, `numpy`, `scipy`, `matplotlib` (pre-installed).
- For MACE: `mace-torch` (pre-installed).

## Detailed Steps

### Method A: ASE + MACE

Applies symmetry-reduced strain deformations to a relaxed structure, computes the stress tensor at each deformed configuration using MACE-MP-0, and fits the full elastic tensor by least squares. Includes stress-strain curve plotting.

```python
#!/usr/bin/env python3
"""
Elastic constants via stress-strain method using ASE + MACE-MP-0.

Workflow:
  1. Relax structure (cell + ions) with MACE.
  2. Generate 6 independent Voigt strain modes at +/-0.5% and +/-1.0%.
  3. Apply symmetry reduction to minimize deformation count.
  4. For each deformation: apply strain, relax ions (fixed cell), record stress.
  5. Fit 6x6 elastic tensor C_ij by least squares.
  6. Compute Voigt-Reuss-Hill moduli and check Born stability.
  7. Plot stress-strain curves for each Voigt component.
"""

import os, json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from ase.optimize import LBFGS
from ase.constraints import ExpCellFilter
from pymatgen.core.structure import Structure
from pymatgen.core.tensors import symmetry_reduce
from pymatgen.analysis.elasticity import Strain, Stress, Deformation, ElasticTensor
from pymatgen.io.ase import AseAtomsAdaptor
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
from mace.calculators import mace_mp

# ─── CONFIGURATION ───────────────────────────────────────────────────────────
INPUT_FILE = "structure.cif"           # Input structure (CIF, POSCAR, etc.)
MACE_MODEL = "medium"                  # MACE-MP-0: "small", "medium", "large"
STRAIN_MAGNITUDES = [-0.01, -0.005, 0.005, 0.01]  # +/-0.5%, +/-1.0%
SYM_REDUCE = True                      # Reduce deformations by crystal symmetry
SYMPREC = 0.01                         # Symmetry precision (Angstrom)
FMAX_BULK = 1e-4                       # Force convergence for bulk relaxation (eV/A)
FMAX_DEFORM = 1e-3                     # Force convergence for deformed cells (eV/A)
OUTPUT_DIR = "elastic_mace"
# ─────────────────────────────────────────────────────────────────────────────

os.makedirs(OUTPUT_DIR, exist_ok=True)
calc = mace_mp(model=MACE_MODEL, default_dtype="float64")
adaptor = AseAtomsAdaptor()

structure = Structure.from_file(INPUT_FILE)
sga = SpacegroupAnalyzer(structure, symprec=SYMPREC)
print(f"Input: {structure.composition.reduced_formula}, "
      f"SG={sga.get_space_group_symbol()}, system={sga.get_crystal_system()}")

# ─── STEP 1: Relax bulk ─────────────────────────────────────────────────────
atoms = adaptor.get_atoms(structure)
atoms.calc = calc
ecf = ExpCellFilter(atoms, hydrostatic_strain=False)
opt = LBFGS(ecf, logfile=os.path.join(OUTPUT_DIR, "bulk_relax.log"))
opt.run(fmax=FMAX_BULK, steps=500)

relaxed = adaptor.get_structure(atoms)
relaxed.to(os.path.join(OUTPUT_DIR, "relaxed.cif"))
print(f"Relaxed volume: {relaxed.volume:.4f} A^3")

# ─── STEP 2: Generate strain deformations ───────────────────────────────────
# 6 independent Voigt strain directions (factor 2 on shear for engineering strain)
strain_states = [
    (1,0,0,0,0,0), (0,1,0,0,0,0), (0,0,1,0,0,0),  # normal
    (0,0,0,2,0,0), (0,0,0,0,2,0), (0,0,0,0,0,2),   # shear
]

strains = []
for state in strain_states:
    for mag in STRAIN_MAGNITUDES:
        s = Strain.from_voigt(mag * np.array(state))
        if (np.abs(s) > 1e-10).any():
            strains.append(s)

if np.linalg.matrix_rank([s.voigt for s in strains]) < 6:
    raise ValueError("Strain set has insufficient rank to fit C_ij.")

if SYM_REDUCE:
    strain_mapping = symmetry_reduce(strains, relaxed, symprec=SYMPREC)
    strains = list(strain_mapping.keys())
    print(f"Symmetry-reduced: {len(strains)} independent deformations")

deformations = [s.get_deformation_matrix() for s in strains]

# ─── STEP 3: Apply deformations and compute stresses ────────────────────────
stresses_list, strain_plot_data = [], []

for idx, deformation in enumerate(deformations):
    deform_matrix = np.array(deformation)
    deformed = relaxed.copy()
    deformed.apply_strain(Strain.from_deformation(Deformation(deform_matrix)))

    atoms_d = adaptor.get_atoms(deformed)
    atoms_d.calc = calc
    opt = LBFGS(atoms_d, logfile=os.devnull)
    opt.run(fmax=FMAX_DEFORM, steps=300)

    # ASE stress: eV/A^3. Convert to GPa (1 eV/A^3 = 160.21766 GPa), negate for pymatgen.
    stress_3x3_gpa = atoms_d.get_stress(voigt=False) * 160.21766
    stress_obj = Stress(-stress_3x3_gpa)
    stresses_list.append(stress_obj)

    strain_voigt = Strain.from_deformation(Deformation(deform_matrix)).voigt
    strain_plot_data.append({
        "strain_voigt": strain_voigt.tolist(),
        "stress_voigt_GPa": stress_obj.voigt.tolist(),
    })
    print(f"  {idx+1}/{len(deformations)}: norm={np.linalg.norm(strain_voigt):.5f}, "
          f"max|stress|={np.max(np.abs(stress_obj.voigt)):.4f} GPa")

# ─── STEP 4: Fit elastic tensor ─────────────────────────────────────────────
eq_stress = Stress(-np.array(atoms.get_stress(voigt=False)) * 160.21766)
et = ElasticTensor.from_independent_strains(stresses_list, strains, eq_stress=eq_stress)

# Symmetrize by averaging over crystal symmetry operations
sga = SpacegroupAnalyzer(relaxed, symprec=SYMPREC)
symmops = sga.get_symmetry_operations(cartesian=True)
C_sym = np.zeros((6, 6))
for op in symmops:
    C_sym += et.transform(op.rotation_matrix).voigt
C_sym /= len(symmops)
et_sym = ElasticTensor.from_voigt(C_sym)
C = et_sym.voigt

print("\n" + "=" * 60)
print("ELASTIC TENSOR C_ij (GPa):")
for i in range(6):
    print("  [" + "  ".join(f"{C[i,j]:8.2f}" for j in range(6)) + "]")

# ─── STEP 5: Voigt-Reuss-Hill moduli ────────────────────────────────────────
K_V, K_R, K_H = et_sym.k_voigt, et_sym.k_reuss, et_sym.k_vrh
G_V, G_R, G_H = et_sym.g_voigt, et_sym.g_reuss, et_sym.g_vrh
E_H = 9 * K_H * G_H / (3 * K_H + G_H) if (3 * K_H + G_H) > 0 else 0.0
nu_H = (3 * K_H - 2 * G_H) / (6 * K_H + 2 * G_H) if (6 * K_H + 2 * G_H) > 0 else 0.0

print(f"\nPolycrystalline Moduli (GPa):")
print(f"  K: Voigt={K_V:.2f}, Reuss={K_R:.2f}, VRH={K_H:.2f}")
print(f"  G: Voigt={G_V:.2f}, Reuss={G_R:.2f}, VRH={G_H:.2f}")
print(f"  E (VRH) = {E_H:.2f},  nu (VRH) = {nu_H:.4f}")
print(f"  Pugh ratio K/G = {K_H / G_H:.3f}" if G_H > 0 else "")

# ─── STEP 6: Born stability ─────────────────────────────────────────────────
eigenvalues = np.linalg.eigvalsh(C)
is_stable = all(ev > 0 for ev in eigenvalues)
crystal_sys = sga.get_crystal_system()

print(f"\nBorn Stability ({crystal_sys}):")
print(f"  Eigenvalues: {', '.join(f'{ev:.2f}' for ev in eigenvalues)}")
print(f"  Mechanically stable: {'YES' if is_stable else 'NO -- UNSTABLE'}")

if crystal_sys == "cubic":
    c11, c12, c44 = C[0, 0], C[0, 1], C[3, 3]
    print(f"  Cubic: C11={c11:.1f}>0, C44={c44:.1f}>0, "
          f"C11>|C12|: {c11:.1f}>{abs(c12):.1f}, C11+2C12={c11+2*c12:.1f}>0")
    A_z = 2 * c44 / (c11 - c12) if (c11 - c12) != 0 else float("inf")
    print(f"  Zener anisotropy: A = {A_z:.3f} (1.0 = isotropic)")

# ─── STEP 7: Stress-strain curve plots ──────────────────────────────────────
fig, axes = plt.subplots(2, 3, figsize=(15, 10))
labels = ["xx", "yy", "zz", "yz", "xz", "xy"]

for i, ax in enumerate(axes.flat):
    si, ti = [], []
    for d in strain_plot_data:
        sv = d["strain_voigt"]
        if np.argmax(np.abs(sv)) == i:
            si.append(sv[i])
            ti.append(d["stress_voigt_GPa"][i])
    if si:
        order = np.argsort(si)
        xs, ys = np.array(si)[order], np.array(ti)[order]
        ax.plot(xs, ys, "o-", color="steelblue", markersize=6)
        if len(xs) >= 2:
            c_fit = np.polyfit(xs, ys, 1)
            fx = np.linspace(xs.min(), xs.max(), 50)
            ax.plot(fx, np.polyval(c_fit, fx), "--r", alpha=0.7,
                    label=f"slope={c_fit[0]:.1f} GPa")
            ax.legend(fontsize=9)
    ax.set_xlabel(f"$\\epsilon_{{{labels[i]}}}$")
    ax.set_ylabel(f"$\\sigma_{{{labels[i]}}}$ (GPa)")
    ax.set_title(f"Component {labels[i]}")
    ax.grid(True, alpha=0.3)

fig.suptitle(f"Stress-Strain: {relaxed.composition.reduced_formula} (MACE)", fontsize=14)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "stress_strain_curves.png"), dpi=150, bbox_inches="tight")
plt.close()
print(f"\nPlot saved to {OUTPUT_DIR}/stress_strain_curves.png")

# ─── Save JSON ───────────────────────────────────────────────────────────────
results = {
    "formula": str(relaxed.composition.reduced_formula),
    "space_group": sga.get_space_group_symbol(), "crystal_system": crystal_sys,
    "volume_A3": relaxed.volume, "elastic_tensor_GPa": C.tolist(),
    "K_Voigt_GPa": K_V, "K_Reuss_GPa": K_R, "K_VRH_GPa": K_H,
    "G_Voigt_GPa": G_V, "G_Reuss_GPa": G_R, "G_VRH_GPa": G_H,
    "E_VRH_GPa": E_H, "Poisson_ratio_VRH": nu_H,
    "eigenvalues_C": eigenvalues.tolist(),
    "is_mechanically_stable": bool(is_stable),
    "method": f"MACE-MP-0 ({MACE_MODEL})", "strain_magnitudes": STRAIN_MAGNITUDES,
    "n_deformations": len(deformations),
}
with open(os.path.join(OUTPUT_DIR, "elastic_results.json"), "w") as f:
    json.dump(results, f, indent=2)
print(f"Results saved to {OUTPUT_DIR}/elastic_results.json")
```

### Method B: QE DFT

Uses Quantum ESPRESSO pw.x to compute stresses under applied strain. Three-step workflow: (1) relax with vc-relax, (2) generate strained structures and QE inputs, (3) extract stresses and fit C_ij.

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
    ecutwfc      = 80.0
    ecutrho      = 640.0
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
  10 10 10  0 0 0
```

Run: `pw.x < relax.in > relax.out`

#### Step B2: Generate strained structures and QE inputs

```python
#!/usr/bin/env python3
"""Generate symmetry-reduced strained structures and QE input files."""

import os, json
import numpy as np
from pymatgen.core.structure import Structure
from pymatgen.io.pwscf import PWInput
from pymatgen.analysis.elasticity import Strain, Deformation
from pymatgen.core.tensors import symmetry_reduce
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer

RELAXED_FILE = "relaxed_structure.cif"
PSEUDO_DIR = "./pseudo"
ECUTWFC, ECUTRHO = 80.0, 640.0
K_GRID = [10, 10, 10]
STRAIN_MAGNITUDES = [-0.01, -0.005, 0.005, 0.01]
SYM_REDUCE, SYMPREC = True, 0.01
WORK_DIR = "stress_strain_qe"

os.makedirs(WORK_DIR, exist_ok=True)
structure = Structure.from_file(RELAXED_FILE)
sga = SpacegroupAnalyzer(structure, symprec=SYMPREC)
print(f"{structure.composition.reduced_formula}, SG={sga.get_space_group_symbol()}, "
      f"system={sga.get_crystal_system()}")

strain_states = [(1,0,0,0,0,0),(0,1,0,0,0,0),(0,0,1,0,0,0),
                 (0,0,0,2,0,0),(0,0,0,0,2,0),(0,0,0,0,0,2)]

all_strains = []
for state in strain_states:
    for mag in STRAIN_MAGNITUDES:
        s = Strain.from_voigt(mag * np.array(state, dtype=float))
        if (np.abs(s) > 1e-10).any():
            all_strains.append(s)

if SYM_REDUCE:
    strains = list(symmetry_reduce(all_strains, structure, symprec=SYMPREC).keys())
    print(f"Symmetry-reduced: {len(strains)} deformations (from {len(all_strains)})")
else:
    strains = all_strains

pseudo_map = {el.symbol: f"{el.symbol}.pbe-n-rrkjus_psl.1.0.0.UPF"
              for el in structure.composition.elements}

deformation_info = []
for idx, strain in enumerate(strains):
    deform_dir = os.path.join(WORK_DIR, f"deform_{idx:03d}")
    os.makedirs(deform_dir, exist_ok=True)
    deformation = strain.get_deformation_matrix()
    deformed = structure.copy()
    deformed.apply_strain(Strain.from_deformation(Deformation(deformation)))

    pw_input = PWInput(
        deformed, pseudo=pseudo_map,
        control={"calculation": "relax", "prefix": f"deform_{idx:03d}",
                 "outdir": "./tmp", "pseudo_dir": os.path.abspath(PSEUDO_DIR),
                 "tprnfor": True, "tstress": True,
                 "forc_conv_thr": 1.0e-5, "etot_conv_thr": 1.0e-8},
        system={"ecutwfc": ECUTWFC, "ecutrho": ECUTRHO,
                "occupations": "smearing", "smearing": "mv", "degauss": 0.02},
        electrons={"conv_thr": 1.0e-10, "mixing_beta": 0.7},
        ions={"ion_dynamics": "bfgs"},
        kpoints_grid=tuple(K_GRID),
    )
    pw_input.write_file(os.path.join(deform_dir, "scf.in"))
    deformation_info.append({"index": idx, "deformation": np.array(deformation).tolist(),
                             "strain_voigt": strain.voigt.tolist(), "directory": deform_dir})

with open(os.path.join(WORK_DIR, "deformation_info.json"), "w") as f:
    json.dump(deformation_info, f, indent=2)

with open(os.path.join(WORK_DIR, "run_all.sh"), "w") as f:
    f.write("#!/bin/bash\n")
    for idx in range(len(strains)):
        f.write(f"echo '{idx+1}/{len(strains)}' && "
                f"cd {WORK_DIR}/deform_{idx:03d} && pw.x < scf.in > scf.out 2>&1 && cd ../..\n")
os.chmod(os.path.join(WORK_DIR, "run_all.sh"), 0o755)
print(f"Run: bash {WORK_DIR}/run_all.sh")
```

#### Step B3: Extract stresses and fit elastic tensor

```python
#!/usr/bin/env python3
"""Post-process QE outputs: extract stresses, fit C_ij by least squares."""

import os, json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pymatgen.core.structure import Structure
from pymatgen.analysis.elasticity import ElasticTensor
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer

RELAXED_FILE = "relaxed_structure.cif"
WORK_DIR = "stress_strain_qe"
SYMPREC = 0.01
OUTPUT_DIR = "stress_strain_results_qe"
os.makedirs(OUTPUT_DIR, exist_ok=True)
structure = Structure.from_file(RELAXED_FILE)

def parse_qe_stress(output_file):
    """Extract final stress tensor from QE output. Returns 3x3 in GPa."""
    stress = None
    with open(output_file) as f:
        lines = f.readlines()
    for i, line in enumerate(lines):
        if "total   stress" in line:
            s = [[float(x) for x in lines[i+j].split()[:3]] for j in range(1, 4)]
            stress = np.array(s)
    if stress is None:
        raise ValueError(f"No stress in {output_file}")
    return stress * 0.1  # kbar -> GPa

eq_stress_gpa = np.zeros((3, 3))
if os.path.exists("relax.out"):
    try: eq_stress_gpa = parse_qe_stress("relax.out")
    except ValueError: pass

with open(os.path.join(WORK_DIR, "deformation_info.json")) as f:
    deformation_info = json.load(f)

strains_voigt, stresses_voigt = [], []
for info in deformation_info:
    output_file = os.path.join(info["directory"], "scf.out")
    if not os.path.exists(output_file):
        print(f"  WARNING: {output_file} not found"); continue
    try: stress_gpa = parse_qe_stress(output_file)
    except ValueError as e:
        print(f"  WARNING: {e}"); continue

    # QE: positive = compressive. Negate for tension-positive, subtract equilibrium.
    sc = -stress_gpa - (-eq_stress_gpa)
    stresses_voigt.append([sc[0,0], sc[1,1], sc[2,2], sc[1,2], sc[0,2], sc[0,1]])
    strains_voigt.append(np.array(info["strain_voigt"]))

print(f"Parsed {len(strains_voigt)}/{len(deformation_info)} deformations")
if len(strains_voigt) < 6:
    raise RuntimeError("Need >= 6 deformations to fit C_ij.")

# Least-squares fit: sigma_i = sum_j C_ij * epsilon_j
eps, sig = np.array(strains_voigt), np.array(stresses_voigt)
C_fit = np.zeros((6, 6))
for i in range(6):
    C_fit[i, :] = np.linalg.lstsq(eps, sig[:, i], rcond=None)[0]
C_fit = 0.5 * (C_fit + C_fit.T)

# Symmetrize by crystal symmetry
sga = SpacegroupAnalyzer(structure, symprec=SYMPREC)
et = ElasticTensor.from_voigt(C_fit)
symmops = sga.get_symmetry_operations(cartesian=True)
C_sym = sum(et.transform(op.rotation_matrix).voigt for op in symmops) / len(symmops)
C_sym = 0.5 * (C_sym + C_sym.T)
et_sym = ElasticTensor.from_voigt(C_sym)

print(f"\nElastic Tensor C_ij (GPa):")
for i in range(6):
    print("  [" + "  ".join(f"{C_sym[i,j]:8.2f}" for j in range(6)) + "]")

K_H, G_H = et_sym.k_vrh, et_sym.g_vrh
E_H = 9*K_H*G_H/(3*K_H+G_H) if (3*K_H+G_H) > 0 else 0
nu_H = (3*K_H-2*G_H)/(6*K_H+2*G_H) if (6*K_H+2*G_H) > 0 else 0
print(f"VRH: K={K_H:.2f}, G={G_H:.2f}, E={E_H:.2f} GPa, nu={nu_H:.4f}")

eigenvalues = np.linalg.eigvalsh(C_sym)
print(f"Born stability: {', '.join(f'{ev:.2f}' for ev in eigenvalues)}")
print(f"Stable: {'YES' if all(ev > 0 for ev in eigenvalues) else 'NO'}")

# Fit quality (R^2 per stress component)
sig_pred = eps @ C_fit.T
vlabels = ["xx", "yy", "zz", "yz", "xz", "xy"]
for i in range(6):
    ss_res = np.sum((sig[:, i] - sig_pred[:, i])**2)
    ss_tot = np.sum((sig[:, i] - np.mean(sig[:, i]))**2)
    print(f"  sigma_{vlabels[i]}: R^2 = {1-ss_res/ss_tot:.6f}" if ss_tot > 1e-20 else "")

# Stress-strain plot
fig, axes = plt.subplots(2, 3, figsize=(15, 10))
for i, ax in enumerate(axes.flat):
    si, ti = [], []
    for j in range(len(strains_voigt)):
        if np.argmax(np.abs(strains_voigt[j])) == i:
            si.append(strains_voigt[j][i]); ti.append(stresses_voigt[j][i])
    if si:
        order = np.argsort(si)
        xs, ys = np.array(si)[order], np.array(ti)[order]
        ax.plot(xs, ys, "o", color="steelblue", markersize=7, label="DFT")
        if len(xs) >= 2:
            fx = np.linspace(xs.min()*1.1, xs.max()*1.1, 50)
            ax.plot(fx, C_sym[i,i]*fx, "--r", alpha=0.7,
                    label=f"C$_{{{i+1}{i+1}}}$={C_sym[i,i]:.1f}")
            ax.legend(fontsize=9)
    ax.set_xlabel(f"$\\epsilon_{{{vlabels[i]}}}$"); ax.set_ylabel(f"$\\sigma_{{{vlabels[i]}}}$ (GPa)")
    ax.set_title(f"{vlabels[i]}"); ax.grid(True, alpha=0.3)
fig.suptitle(f"Stress-Strain (QE): {structure.composition.reduced_formula}", fontsize=14)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "stress_strain_qe.png"), dpi=150, bbox_inches="tight")
plt.close()

results = {
    "formula": str(structure.composition.reduced_formula),
    "elastic_tensor_GPa": C_sym.tolist(),
    "K_VRH_GPa": float(K_H), "G_VRH_GPa": float(G_H),
    "E_VRH_GPa": float(E_H), "Poisson_ratio_VRH": float(nu_H),
    "eigenvalues_C": eigenvalues.tolist(),
    "is_mechanically_stable": bool(all(ev > 0 for ev in eigenvalues)),
    "method": "QE PBE (stress-strain)", "n_deformations": len(strains_voigt),
}
with open(os.path.join(OUTPUT_DIR, "elastic_results.json"), "w") as f:
    json.dump(results, f, indent=2)
print(f"Results saved to {OUTPUT_DIR}/")
```

### Method C: VASP

Generates strained POSCARs, runs VASP with ISIF=2, and parses stresses from OUTCAR. Compatible with VASPKIT tasks 200-203 for automated post-processing.

#### Step C1: Relax the bulk structure

```
# INCAR for full relaxation (ISIF=3)
SYSTEM  = bulk_relax
PREC    = Accurate
ENCUT   = 520
EDIFF   = 1E-7
EDIFFG  = -1E-3
IBRION  = 2
ISIF    = 3
NSW     = 100
ISMEAR  = 1
SIGMA   = 0.2
LWAVE   = .FALSE.
LCHARG  = .FALSE.
LREAL   = .FALSE.
```

Run: `mpirun -np $NPROC vasp_std`. After convergence, copy CONTCAR to POSCAR.

**VASPKIT shortcut:** `vaspkit -task 200` auto-generates strained structures from CONTCAR.

#### Step C2: Generate strained POSCARs and VASP inputs

```python
#!/usr/bin/env python3
"""
Generate strained VASP inputs. VASPKIT reference:
  Task 200: Elastic constants (stress-strain, auto-generates strained structures)
  Task 201: Elastic constants (energy-strain)
  Task 202: Extract elastic tensor from OUTCAR (IBRION=6)
  Task 203: Mechanical properties from elastic tensor
"""

import os, json
import numpy as np
from pymatgen.core.structure import Structure
from pymatgen.analysis.elasticity import Strain, Deformation
from pymatgen.core.tensors import symmetry_reduce
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
from pymatgen.io.vasp.sets import MPStaticSet

RELAXED_FILE = "CONTCAR"
STRAIN_MAGNITUDES = [-0.01, -0.005, 0.005, 0.01]
SYM_REDUCE, SYMPREC = True, 0.01
WORK_DIR = "stress_strain_vasp"
ENCUT = 520

os.makedirs(WORK_DIR, exist_ok=True)
structure = Structure.from_file(RELAXED_FILE)
sga = SpacegroupAnalyzer(structure, symprec=SYMPREC)
print(f"{structure.composition.reduced_formula}, SG={sga.get_space_group_symbol()}")

strain_states = [(1,0,0,0,0,0),(0,1,0,0,0,0),(0,0,1,0,0,0),
                 (0,0,0,2,0,0),(0,0,0,0,2,0),(0,0,0,0,0,2)]

all_strains = [Strain.from_voigt(mag * np.array(st, dtype=float))
               for st in strain_states for mag in STRAIN_MAGNITUDES
               if (np.abs(Strain.from_voigt(mag * np.array(st, dtype=float))) > 1e-10).any()]

if SYM_REDUCE:
    strains = list(symmetry_reduce(all_strains, structure, symprec=SYMPREC).keys())
    print(f"Symmetry-reduced: {len(strains)} deformations (from {len(all_strains)})")
else:
    strains = all_strains

deformation_info = []
for idx, strain in enumerate(strains):
    deform_dir = os.path.join(WORK_DIR, f"deform_{idx:03d}")
    os.makedirs(deform_dir, exist_ok=True)
    deformation = strain.get_deformation_matrix()
    deformed = structure.copy()
    deformed.apply_strain(Strain.from_deformation(Deformation(deformation)))

    # ISIF=2: relax ions, compute stress, keep cell shape fixed
    vasp_set = MPStaticSet(deformed, user_incar_settings={
        "ENCUT": ENCUT, "EDIFF": 1e-7, "EDIFFG": -1e-4,
        "IBRION": 2, "ISIF": 2, "NSW": 50,
        "ISMEAR": 1, "SIGMA": 0.2,
        "PREC": "Accurate", "LWAVE": False, "LCHARG": False, "LREAL": False,
    })
    vasp_set.write_input(deform_dir)
    deformation_info.append({"index": idx, "deformation": np.array(deformation).tolist(),
                             "strain_voigt": strain.voigt.tolist(), "directory": deform_dir})

with open(os.path.join(WORK_DIR, "deformation_info.json"), "w") as f:
    json.dump(deformation_info, f, indent=2)

with open(os.path.join(WORK_DIR, "run_all.sh"), "w") as f:
    f.write("#!/bin/bash\n")
    for idx in range(len(strains)):
        f.write(f"echo '{idx+1}/{len(strains)}' && "
                f"cd {WORK_DIR}/deform_{idx:03d} && mpirun -np $NPROC vasp_std && cd ../..\n")
os.chmod(os.path.join(WORK_DIR, "run_all.sh"), 0o755)
print(f"Generated {len(strains)} VASP inputs. Run: bash {WORK_DIR}/run_all.sh")
```

#### Step C3: Parse OUTCAR stresses and fit C_ij

```python
#!/usr/bin/env python3
"""Parse VASP OUTCAR stresses and fit C_ij. Alt: vaspkit -task 203."""

import os, json
import numpy as np
from pymatgen.core.structure import Structure
from pymatgen.io.vasp.outputs import Outcar
from pymatgen.analysis.elasticity import ElasticTensor
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer

RELAXED_FILE, WORK_DIR, SYMPREC = "CONTCAR", "stress_strain_vasp", 0.01
OUTPUT_DIR = "stress_strain_results_vasp"
os.makedirs(OUTPUT_DIR, exist_ok=True)
structure = Structure.from_file(RELAXED_FILE)

# Equilibrium stress
eq_stress = np.zeros((3, 3))
if os.path.exists("OUTCAR"):
    try:
        oc = Outcar("OUTCAR")
        if oc.stress is not None: eq_stress = np.array(oc.stress) * 0.1
    except Exception: pass

with open(os.path.join(WORK_DIR, "deformation_info.json")) as f:
    deformation_info = json.load(f)

strains_voigt, stresses_voigt = [], []
for info in deformation_info:
    outcar_path = os.path.join(info["directory"], "OUTCAR")
    if not os.path.exists(outcar_path):
        print(f"  WARNING: {outcar_path} not found"); continue
    try:
        stress_gpa = np.array(Outcar(outcar_path).stress) * 0.1
    except Exception as e:
        print(f"  WARNING: {e}"); continue

    # VASP: positive = compressive. Negate, subtract equilibrium.
    sc = -stress_gpa - (-eq_stress)
    stresses_voigt.append([sc[0,0], sc[1,1], sc[2,2], sc[1,2], sc[0,2], sc[0,1]])
    strains_voigt.append(np.array(info["strain_voigt"]))

print(f"Parsed {len(strains_voigt)}/{len(deformation_info)} deformations")

eps, sig = np.array(strains_voigt), np.array(stresses_voigt)
C_fit = np.zeros((6, 6))
for i in range(6):
    C_fit[i, :] = np.linalg.lstsq(eps, sig[:, i], rcond=None)[0]
C_fit = 0.5 * (C_fit + C_fit.T)

sga = SpacegroupAnalyzer(structure, symprec=SYMPREC)
et = ElasticTensor.from_voigt(C_fit)
symmops = sga.get_symmetry_operations(cartesian=True)
C_sym = sum(et.transform(op.rotation_matrix).voigt for op in symmops) / len(symmops)
C_sym = 0.5 * (C_sym + C_sym.T)
et_sym = ElasticTensor.from_voigt(C_sym)

print(f"\nElastic Tensor C_ij (GPa) -- VASP:")
for i in range(6):
    print("  [" + "  ".join(f"{C_sym[i,j]:8.2f}" for j in range(6)) + "]")

K_H, G_H = et_sym.k_vrh, et_sym.g_vrh
E_H = 9*K_H*G_H/(3*K_H+G_H) if (3*K_H+G_H) > 0 else 0
nu_H = (3*K_H-2*G_H)/(6*K_H+2*G_H) if (6*K_H+2*G_H) > 0 else 0
eigenvalues = np.linalg.eigvalsh(C_sym)

print(f"VRH: K={K_H:.2f}, G={G_H:.2f}, E={E_H:.2f} GPa, nu={nu_H:.4f}")
print(f"Born stability: {[f'{ev:.2f}' for ev in eigenvalues]}")
print(f"Stable: {'YES' if all(ev > 0 for ev in eigenvalues) else 'NO'}")

results = {
    "formula": str(structure.composition.reduced_formula),
    "elastic_tensor_GPa": C_sym.tolist(),
    "K_VRH_GPa": float(K_H), "G_VRH_GPa": float(G_H),
    "E_VRH_GPa": float(E_H), "Poisson_ratio_VRH": float(nu_H),
    "eigenvalues_C": eigenvalues.tolist(),
    "is_mechanically_stable": bool(all(ev > 0 for ev in eigenvalues)),
    "method": "VASP PBE (stress-strain)", "n_deformations": len(strains_voigt),
}
with open(os.path.join(OUTPUT_DIR, "elastic_results.json"), "w") as f:
    json.dump(results, f, indent=2)
print(f"Results saved to {OUTPUT_DIR}/")
print("Tip: run 'vaspkit -task 203' to compute mechanical properties from the elastic tensor.")
```

## Key Parameters

| Parameter | Value | Notes |
|---|---|---|
| Strain magnitude | 0.005 -- 0.01 | Use 0.005 for stiff materials (diamond, carbides), 0.01 for softer. Larger strains risk nonlinearity. |
| Strain points per mode | 4 (two positive, two negative) | Minimum 2 for linear fit; 4--5 gives better regression and residual diagnostics. |
| Strain states | 6 (Voigt basis) | 3 normal (e_xx, e_yy, e_zz) + 3 shear (e_yz, e_xz, e_xy). Symmetry reduces the independent count. |
| Symmetry reduction | Enabled | Cubic: 24 -> 2--4 deformations. Hexagonal: 24 -> 4--6. Triclinic: no reduction (all 24). |
| ecutwfc (QE) | 80 Ry | Stresses converge slower than energies. Increase by 10 Ry to test convergence. |
| ENCUT (VASP) | 520 eV | PREC=Accurate recommended. Must exceed ENMAX from POTCAR by at least 1.3x. |
| k-grid | Dense (10x10x10 for FCC) | Stress is very sensitive to k-point density. Denser than for energy convergence alone. |
| conv_thr / EDIFF | 1e-10 Ry / 1e-7 eV | Tight SCF convergence essential for reliable stress tensors. |
| ISIF (VASP) | 2 (strained calcs) | ISIF=2: relax ions, compute stress, fixed cell. ISIF=3: full relaxation (equilibrium only). |
| FMAX (MACE) | 1e-4 eV/A (bulk), 1e-3 eV/A (deformed) | Tighter bulk relaxation ensures near-zero equilibrium stress. |

## Interpreting Results

**Elastic tensor (C_ij):**
- Diagonal C_11, C_22, C_33: resistance to uniaxial strain along x, y, z.
- Diagonal C_44, C_55, C_66: resistance to shear deformation.
- Off-diagonal C_12, C_13, C_23: coupling between normal strain components.
- Independent constants by crystal system -- cubic: 3 (C_11, C_12, C_44), hexagonal: 5, orthorhombic: 9, monoclinic: 13, triclinic: 21.

**Voigt-Reuss-Hill averages:**
- Voigt (uniform strain) = upper bound. Reuss (uniform stress) = lower bound. Hill (VRH) = arithmetic mean.
- K (bulk modulus): resistance to compression. G (shear modulus): resistance to shape change.
- E = 9KG/(3K+G): Young's modulus. nu = (3K-2G)/(6K+2G): Poisson's ratio.
- Pugh's ratio K/G > 1.75 suggests ductile behavior, < 1.75 suggests brittle.

**Born stability criteria:**
- All eigenvalues of the 6x6 C_ij matrix must be positive for mechanical stability.
- Negative eigenvalues indicate the structure is at a saddle point, not a true energy minimum.
- Crystal-specific: cubic requires C_11 > 0, C_44 > 0, C_11 > |C_12|, C_11 + 2C_12 > 0.

**Typical accuracy:**
- MACE: within 10--20% of DFT for well-represented chemistries. Shear constants less accurate than bulk modulus.
- QE/VASP PBE: within 5--15% of experiment. GGA typically underestimates slightly.
- Stress-strain vs. energy-strain (VASPKIT 200 vs. 201): should agree within 1--2% when well converged.

## Common Issues

| Issue | Solution |
|---|---|
| Negative eigenvalues of C_ij | Re-relax with tighter convergence. If persistent, the phase may be genuinely unstable at 0 K. |
| Asymmetric C_ij (C_12 != C_21) | Symmetrize via `C = (C + C.T) / 2` and crystal symmetry averaging. |
| Stress-strain curve is nonlinear | Reduce strain magnitude to 0.005 or 0.003. |
| Large off-diagonal noise | Enable symmetry reduction. Tighten conv_thr/EDIFF and increase ecutwfc/ENCUT. |
| Very different Voigt vs Reuss | Physical for anisotropic materials (layered structures). Report VRH average. |
| QE/VASP stress oscillates | Tighten SCF (conv_thr=1e-12, EDIFF=1e-8). Increase k-grid. Reduce mixing_beta for metals. |
| MACE gives unreasonable values | Material outside training domain. Fall back to DFT. Compare lattice params with experiment. |
| "Rank < 6" error | Symmetry reduction removed too many strains. Disable SYM_REDUCE or add more magnitudes. |
| VASPKIT task 200 fails | Ensure CONTCAR exists and POTCAR is consistent. Run ISIF=3 relaxation first. |
| Poor R^2 for one component | That elastic constant may be near zero or strain set ill-conditioned. Add more strain points. |
