# Molecular Quantum Chemistry Workflows

## When to Use

- You need to optimize the geometry of an isolated molecule and compute vibrational frequencies, IR/Raman intensities, or thermochemistry (ZPE, enthalpy, Gibbs free energy).
- You need the solvation free energy of a molecule using implicit solvent models (PCM, CPCM, SMD).
- You need to compute reaction energies, enthalpies, or Gibbs free energies for a chemical reaction (A + B -> C).
- You need UV-Vis absorption spectra or excited-state energies via TD-DFT.
- You need NMR chemical shifts or other molecular response properties.
- Your system is a molecule or molecular cluster (not a periodic solid). For periodic solids, use the `electronic-structure/` skills instead.

## Method Selection

```
How large is your molecule?

< 15 heavy atoms:
  Geometry/Freq:  B3LYP/def2-TZVP  (reliable, moderate cost)
  Single-point:   CCSD(T)/cc-pVTZ  (gold standard, expensive)
  Excited states: TD-B3LYP/def2-TZVP or EOM-CCSD/cc-pVDZ

15-50 heavy atoms:
  Geometry/Freq:  B3LYP/def2-SVP  (fast) then single-point at B3LYP/def2-TZVP
  Single-point:   MP2/cc-pVTZ  or  DLPNO-CCSD(T) (if available)
  Excited states: TD-B3LYP/def2-SVP

50-200 heavy atoms:
  Geometry/Freq:  B3LYP/def2-SVP  (may need density fitting: .density_fit())
  Single-point:   B3LYP-D3/def2-TZVP
  Excited states: TD-PBE0/def2-SVP (TD-DFT only feasible option)

> 200 heavy atoms:
  Consider ML potentials (ANI, MACE) or semi-empirical methods (xTB)
  QChem DFT becomes prohibitively expensive

Basis set guide:
  def2-SVP      -- double-zeta, fast, good for optimization
  def2-TZVP     -- triple-zeta, good for energetics
  cc-pVDZ/TZ/QZ -- Dunning correlation-consistent, best for post-HF (MP2, CCSD(T))
  aug-cc-pVXZ   -- diffuse functions, needed for anions, weak interactions, excited states

Functional guide:
  B3LYP         -- general-purpose hybrid, good geometries and frequencies
  PBE0          -- hybrid GGA, slightly better for transition metals
  wB97X-D       -- range-separated + dispersion, excellent for non-covalent interactions
  M06-2X        -- good for main-group thermochemistry and kinetics
  PBE           -- GGA, fast but less accurate for barrier heights
```

## Prerequisites

```bash
# Core requirements
pip install pyscf geometric

# Optional: for building molecules from SMILES
pip install rdkit

# Alternative engine (Psi4, if preferred over PySCF):
# conda install -c conda-forge psi4
```

Pre-installed in the MatClaw container: `ase`, `numpy`, `scipy`, `matplotlib`.

---

## Detailed Steps

### Workflow 1: Geometry Optimization + Frequency (PySCF)

Complete workflow: build a molecule from XYZ coordinates or SMILES, optimize with B3LYP/def2-SVP, compute harmonic frequencies, and extract thermochemistry (ZPE, H, G at 298.15 K).

```python
#!/usr/bin/env python3
"""
Molecular geometry optimization + vibrational frequency analysis using PySCF.
Computes optimized geometry, harmonic frequencies, IR intensities,
zero-point energy, thermal corrections, and Gibbs free energy at 298.15 K.
"""

import numpy as np
from pyscf import gto, dft
from pyscf.geomopt.geometric_solver import optimize
from pyscf.hessian import thermo

# ============================================================
# 1. BUILD THE MOLECULE
# ============================================================

# Option A: From XYZ coordinates (water molecule example)
mol = gto.Mole()
mol.atom = """
    O   0.000000   0.000000   0.117370
    H   0.000000   0.756950  -0.469483
    H   0.000000  -0.756950  -0.469483
"""
mol.basis = "def2-svp"
mol.charge = 0
mol.spin = 0  # 2S, not 2S+1: 0 for singlet, 1 for doublet, etc.
mol.verbose = 4
mol.build()

# Option B: From SMILES (requires rdkit)
# from rdkit import Chem
# from rdkit.Chem import AllChem
#
# smiles = "CCO"  # ethanol
# rdmol = Chem.MolFromSmiles(smiles)
# rdmol = Chem.AddHs(rdmol)
# AllChem.EmbedMolecule(rdmol, randomSeed=42)
# AllChem.MMFFOptimizeMolecule(rdmol)
#
# xyz_lines = []
# conf = rdmol.GetConformer()
# for i, atom in enumerate(rdmol.GetAtoms()):
#     pos = conf.GetAtomPosition(i)
#     xyz_lines.append(f"{atom.GetSymbol()}  {pos.x:.6f}  {pos.y:.6f}  {pos.z:.6f}")
#
# mol = gto.Mole()
# mol.atom = "\n".join(xyz_lines)
# mol.basis = "def2-svp"
# mol.charge = 0
# mol.spin = 0
# mol.verbose = 4
# mol.build()

print(f"Molecule: {mol.natm} atoms, {mol.nelectron} electrons")
print(f"Basis: {mol.basis}, {mol.nao_nr()} AOs")

# ============================================================
# 2. SET UP DFT CALCULATOR (B3LYP)
# ============================================================

mf = dft.RKS(mol)
mf.xc = "b3lyp"
mf.grids.level = 4  # integration grid fineness (default 3, 4-5 for tight)
mf.conv_tol = 1e-10  # tight SCF convergence for frequencies
mf.kernel()

print(f"\nInitial SCF energy: {mf.e_tot:.10f} Hartree")

# ============================================================
# 3. GEOMETRY OPTIMIZATION (geomeTRIC)
# ============================================================

print("\n=== Geometry Optimization ===")
mol_eq = optimize(mf, maxsteps=100, constraints=None)

# Update the mf object with the optimized geometry
mf_opt = dft.RKS(mol_eq)
mf_opt.xc = "b3lyp"
mf_opt.grids.level = 4
mf_opt.conv_tol = 1e-10
mf_opt.kernel()

print(f"Optimized SCF energy: {mf_opt.e_tot:.10f} Hartree")

# Print optimized coordinates
print("\nOptimized geometry (Angstrom):")
coords_bohr = mol_eq.atom_coords()  # in Bohr
coords_ang = coords_bohr * 0.529177  # convert to Angstrom
for i in range(mol_eq.natm):
    sym = mol_eq.atom_symbol(i)
    print(f"  {sym:2s}  {coords_ang[i,0]:12.6f}  {coords_ang[i,1]:12.6f}  {coords_ang[i,2]:12.6f}")

# Save optimized geometry as XYZ file
with open("optimized.xyz", "w") as f:
    f.write(f"{mol_eq.natm}\n")
    f.write(f"Optimized at B3LYP/def2-SVP, E = {mf_opt.e_tot:.10f} Hartree\n")
    for i in range(mol_eq.natm):
        sym = mol_eq.atom_symbol(i)
        f.write(f"{sym}  {coords_ang[i,0]:.6f}  {coords_ang[i,1]:.6f}  {coords_ang[i,2]:.6f}\n")
print("Saved: optimized.xyz")

# ============================================================
# 4. HESSIAN (FREQUENCY) CALCULATION
# ============================================================

print("\n=== Frequency Calculation ===")
hess_calc = mf_opt.Hessian()
hessian_matrix = hess_calc.kernel()

# ============================================================
# 5. THERMOCHEMISTRY FROM FREQUENCIES
# ============================================================

# thermo.harmonic_analysis returns frequencies, normal modes, etc.
freq_info = thermo.harmonic_analysis(mol_eq, hessian_matrix)

# freq_info is a dict with keys:
#   'freq_au'      : frequencies in atomic units
#   'freq_wavenumber': frequencies in cm^-1
#   'norm_mode'    : normal mode vectors
#   'reduced_mass' : reduced masses in amu

freqs_cm = freq_info["freq_wavenumber"]

print(f"\nVibrational frequencies (cm^-1):")
n_vib = 0
for i, f in enumerate(freqs_cm):
    kind = ""
    if abs(f) < 10:
        kind = " (translation/rotation)"
    elif f < 0:
        kind = " *** IMAGINARY ***"
        n_vib += 1
    else:
        n_vib += 1
    print(f"  Mode {i+1:3d}: {f:10.2f}{kind}")

# Compute thermodynamic quantities at 298.15 K, 1 atm
thermo_results = thermo.thermo(
    mf_opt,
    freq_info["freq_au"],
    298.15,  # temperature in K
    101325,  # pressure in Pa (1 atm)
)

# thermo_results is a dict with:
#   'E_elec'   : electronic energy (Hartree)
#   'E_ZPE'    : zero-point energy (Hartree)
#   'E_tot'    : E_elec + E_ZPE + thermal corrections (Hartree)
#   'H_tot'    : enthalpy (Hartree)
#   'G_tot'    : Gibbs free energy (Hartree)
#   'S_tot'    : total entropy (Hartree/K)
#   'Cv_tot'   : heat capacity at constant volume (Hartree/K)

ha_to_kcal = 627.509474
ha_to_kj = 2625.4996

E_elec = thermo_results["E_elec"][0]
E_zpe  = thermo_results["E_ZPE"][0]
H_tot  = thermo_results["H_tot"][0]
G_tot  = thermo_results["G_tot"][0]
S_tot  = thermo_results["S_tot"][0]

print(f"\n=== Thermochemistry at 298.15 K, 1 atm ===")
print(f"  Electronic energy:     {E_elec:16.10f} Hartree  ({E_elec * ha_to_kcal:12.4f} kcal/mol)")
print(f"  Zero-point energy:     {E_zpe:16.10f} Hartree  ({E_zpe * ha_to_kcal:12.4f} kcal/mol)")
print(f"  Enthalpy (H):          {H_tot:16.10f} Hartree  ({H_tot * ha_to_kcal:12.4f} kcal/mol)")
print(f"  Gibbs free energy (G): {G_tot:16.10f} Hartree  ({G_tot * ha_to_kcal:12.4f} kcal/mol)")
print(f"  Entropy (S):           {S_tot:16.10e} Hartree/K ({S_tot * ha_to_kcal * 1000:10.4f} cal/mol/K)")
print(f"  ZPE correction:        {(E_zpe - E_elec) * ha_to_kcal:12.4f} kcal/mol")
print(f"  Thermal corr to H:     {(H_tot - E_elec) * ha_to_kcal:12.4f} kcal/mol")
print(f"  Thermal corr to G:     {(G_tot - E_elec) * ha_to_kcal:12.4f} kcal/mol")

if any(f < -10 for f in freqs_cm):
    print("\n  WARNING: Imaginary frequencies detected! This is NOT a true minimum.")
    print("  The structure may be a transition state or saddle point.")
    print("  Re-optimize with tighter convergence or perturb along the imaginary mode.")
```

---

### Workflow 2: Solvation Free Energy

Compute the solvation free energy of a molecule by comparing the gas-phase and solvated (implicit solvent PCM/CPCM) energies. Both geometries are optimized independently.

```python
#!/usr/bin/env python3
"""
Solvation free energy calculation using PySCF with implicit solvent (ddCOSMO/PCM).
Compares gas-phase and solution-phase energies at B3LYP/def2-SVP.

The solvation free energy is:
    dG_solv = G_solution - G_gas
"""

import numpy as np
from pyscf import gto, dft, solvent
from pyscf.geomopt.geometric_solver import optimize
from pyscf.hessian import thermo

# ============================================================
# 1. BUILD THE MOLECULE (acetic acid example)
# ============================================================

atom_str = """
    C   0.000000   0.000000   0.000000
    C   1.520000   0.000000   0.000000
    O   2.100000   1.140000   0.000000
    O   2.080000  -1.070000   0.000000
    H  -0.390000   0.000000   1.020000
    H  -0.390000   0.890000  -0.510000
    H  -0.390000  -0.890000  -0.510000
    H   3.040000  -1.020000   0.000000
"""

TEMPERATURE = 298.15  # K
PRESSURE = 101325     # Pa (1 atm)
ha_to_kcal = 627.509474

def run_gas_phase(atom_str, basis="def2-svp", xc="b3lyp"):
    """Optimize in gas phase and compute thermochemistry."""
    mol = gto.Mole()
    mol.atom = atom_str
    mol.basis = basis
    mol.charge = 0
    mol.spin = 0
    mol.verbose = 3
    mol.build()

    # DFT single point then optimize
    mf = dft.RKS(mol)
    mf.xc = xc
    mf.grids.level = 4
    mf.conv_tol = 1e-10
    mf.kernel()

    # Geometry optimization
    mol_opt = optimize(mf, maxsteps=100)

    # Recompute at optimized geometry
    mf_opt = dft.RKS(mol_opt)
    mf_opt.xc = xc
    mf_opt.grids.level = 4
    mf_opt.conv_tol = 1e-10
    mf_opt.kernel()

    # Frequencies and thermochemistry
    hess = mf_opt.Hessian().kernel()
    freq_info = thermo.harmonic_analysis(mol_opt, hess)
    thermo_data = thermo.thermo(mf_opt, freq_info["freq_au"], TEMPERATURE, PRESSURE)

    return {
        "mol": mol_opt,
        "mf": mf_opt,
        "E_elec": mf_opt.e_tot,
        "G_tot": thermo_data["G_tot"][0],
        "H_tot": thermo_data["H_tot"][0],
        "freqs_cm": freq_info["freq_wavenumber"],
    }


def run_solvated(atom_str, basis="def2-svp", xc="b3lyp", solvent_eps=78.39):
    """Optimize with implicit solvent (ddCOSMO) and compute thermochemistry.

    solvent_eps: dielectric constant of solvent
        Water:        78.39
        DMSO:         46.7
        Acetonitrile: 35.7
        Methanol:     32.7
        Ethanol:      24.3
        THF:          7.58
        Toluene:      2.38
        Hexane:       1.88
    """
    mol = gto.Mole()
    mol.atom = atom_str
    mol.basis = basis
    mol.charge = 0
    mol.spin = 0
    mol.verbose = 3
    mol.build()

    # DFT with ddCOSMO implicit solvent
    mf = dft.RKS(mol)
    mf.xc = xc
    mf.grids.level = 4
    mf.conv_tol = 1e-10
    mf = solvent.ddCOSMO(mf)
    mf.with_solvent.eps = solvent_eps
    mf.kernel()

    # Geometry optimization in solvent
    mol_opt = optimize(mf, maxsteps=100)

    # Recompute at optimized geometry
    mf_opt = dft.RKS(mol_opt)
    mf_opt.xc = xc
    mf_opt.grids.level = 4
    mf_opt.conv_tol = 1e-10
    mf_opt = solvent.ddCOSMO(mf_opt)
    mf_opt.with_solvent.eps = solvent_eps
    mf_opt.kernel()

    # Frequencies and thermochemistry in solvent
    hess = mf_opt.Hessian().kernel()
    freq_info = thermo.harmonic_analysis(mol_opt, hess)
    thermo_data = thermo.thermo(mf_opt, freq_info["freq_au"], TEMPERATURE, PRESSURE)

    return {
        "mol": mol_opt,
        "mf": mf_opt,
        "E_elec": mf_opt.e_tot,
        "G_tot": thermo_data["G_tot"][0],
        "H_tot": thermo_data["H_tot"][0],
        "freqs_cm": freq_info["freq_wavenumber"],
    }


# ============================================================
# 2. RUN BOTH PHASES
# ============================================================

print("=" * 60)
print("GAS PHASE CALCULATION")
print("=" * 60)
gas = run_gas_phase(atom_str)

print("\n" + "=" * 60)
print("SOLUTION PHASE CALCULATION (water, eps=78.39)")
print("=" * 60)
sol = run_solvated(atom_str, solvent_eps=78.39)

# ============================================================
# 3. COMPUTE SOLVATION FREE ENERGY
# ============================================================

dG_solv_hartree = sol["G_tot"] - gas["G_tot"]
dE_solv_hartree = sol["E_elec"] - gas["E_elec"]

print("\n" + "=" * 60)
print("SOLVATION RESULTS")
print("=" * 60)
print(f"  Gas-phase electronic energy:      {gas['E_elec']:16.10f} Hartree")
print(f"  Solution-phase electronic energy: {sol['E_elec']:16.10f} Hartree")
print(f"  Gas-phase Gibbs free energy:      {gas['G_tot']:16.10f} Hartree")
print(f"  Solution-phase Gibbs free energy: {sol['G_tot']:16.10f} Hartree")
print(f"")
print(f"  dE(solvation) = {dE_solv_hartree:12.8f} Hartree = {dE_solv_hartree * ha_to_kcal:8.3f} kcal/mol")
print(f"  dG(solvation) = {dG_solv_hartree:12.8f} Hartree = {dG_solv_hartree * ha_to_kcal:8.3f} kcal/mol")

if dG_solv_hartree < 0:
    print(f"\n  The molecule is stabilized by solvation (favorable).")
else:
    print(f"\n  The molecule is destabilized by solvation (unfavorable).")
    print(f"  This is unusual for a polar molecule -- check the geometry and solvent parameters.")

# ============================================================
# 4. SAVE RESULTS
# ============================================================

with open("solvation_results.txt", "w") as f:
    f.write("Solvation Free Energy Calculation\n")
    f.write(f"Method: B3LYP/def2-SVP, ddCOSMO (eps=78.39, water)\n")
    f.write(f"Temperature: {TEMPERATURE} K, Pressure: {PRESSURE} Pa\n\n")
    f.write(f"Gas-phase E_elec:  {gas['E_elec']:.10f} Hartree\n")
    f.write(f"Solution   E_elec: {sol['E_elec']:.10f} Hartree\n")
    f.write(f"Gas-phase G:       {gas['G_tot']:.10f} Hartree\n")
    f.write(f"Solution  G:       {sol['G_tot']:.10f} Hartree\n\n")
    f.write(f"dE(solvation) = {dE_solv_hartree * ha_to_kcal:.3f} kcal/mol\n")
    f.write(f"dG(solvation) = {dG_solv_hartree * ha_to_kcal:.3f} kcal/mol\n")
print("Saved: solvation_results.txt")
```

---

### Workflow 3: Reaction Energy Profile

Compute the reaction energy for a chemical reaction A + B -> C with thermodynamic corrections. Each species is independently optimized and frequencies computed.

```python
#!/usr/bin/env python3
"""
Reaction energy calculation using PySCF.
Example reaction: H2 + F2 -> 2 HF

Computes:
  dE  = E(products) - E(reactants)          (electronic)
  dH  = H(products) - H(reactants)          (enthalpy at 298 K)
  dG  = G(products) - G(reactants)          (Gibbs free energy at 298 K)
"""

import numpy as np
from pyscf import gto, dft
from pyscf.geomopt.geometric_solver import optimize
from pyscf.hessian import thermo

TEMPERATURE = 298.15  # K
PRESSURE = 101325     # Pa
ha_to_kcal = 627.509474
ha_to_kj = 2625.4996

# ============================================================
# DEFINE SPECIES
# ============================================================

species = {
    "H2": {
        "atoms": "H 0 0 0; H 0 0 0.74",
        "charge": 0,
        "spin": 0,
        "stoich_reactant": 1,  # coefficient in reaction (positive = reactant)
        "stoich_product": 0,
    },
    "F2": {
        "atoms": "F 0 0 0; F 0 0 1.42",
        "charge": 0,
        "spin": 0,
        "stoich_reactant": 1,
        "stoich_product": 0,
    },
    "HF": {
        "atoms": "H 0 0 0; F 0 0 0.92",
        "charge": 0,
        "spin": 0,
        "stoich_reactant": 0,
        "stoich_product": 2,
    },
}

# ============================================================
# COMPUTE EACH SPECIES
# ============================================================

def compute_species(name, atom_str, charge, spin, basis="def2-tzvp", xc="b3lyp"):
    """Optimize geometry, compute frequencies, and return thermochemistry."""
    print(f"\n{'='*50}")
    print(f"Computing: {name}")
    print(f"{'='*50}")

    mol = gto.Mole()
    mol.atom = atom_str
    mol.basis = basis
    mol.charge = charge
    mol.spin = spin
    mol.verbose = 3
    mol.build()

    # DFT calculation
    if spin == 0:
        mf = dft.RKS(mol)
    else:
        mf = dft.UKS(mol)
    mf.xc = xc
    mf.grids.level = 4
    mf.conv_tol = 1e-10
    mf.kernel()

    # Optimize
    mol_opt = optimize(mf, maxsteps=100)

    # Recompute at optimized geometry
    if spin == 0:
        mf_opt = dft.RKS(mol_opt)
    else:
        mf_opt = dft.UKS(mol_opt)
    mf_opt.xc = xc
    mf_opt.grids.level = 4
    mf_opt.conv_tol = 1e-10
    mf_opt.kernel()

    # Frequencies
    hess = mf_opt.Hessian().kernel()
    freq_info = thermo.harmonic_analysis(mol_opt, hess)
    thermo_data = thermo.thermo(mf_opt, freq_info["freq_au"], TEMPERATURE, PRESSURE)

    result = {
        "name": name,
        "E_elec": mf_opt.e_tot,
        "E_ZPE": thermo_data["E_ZPE"][0],
        "H_tot": thermo_data["H_tot"][0],
        "G_tot": thermo_data["G_tot"][0],
        "freqs_cm": freq_info["freq_wavenumber"],
    }

    print(f"  E_elec = {result['E_elec']:.10f} Hartree")
    print(f"  H_tot  = {result['H_tot']:.10f} Hartree")
    print(f"  G_tot  = {result['G_tot']:.10f} Hartree")

    real_freqs = [f for f in freq_info["freq_wavenumber"] if abs(f) > 10]
    print(f"  Frequencies (cm^-1): {', '.join(f'{f:.1f}' for f in real_freqs)}")

    return result


results = {}
for name, spec in species.items():
    results[name] = compute_species(
        name,
        spec["atoms"],
        spec["charge"],
        spec["spin"],
    )

# ============================================================
# COMPUTE REACTION ENERGETICS
# ============================================================

# Reaction: H2 + F2 -> 2 HF
# dX = sum(stoich_product * X_product) - sum(stoich_reactant * X_reactant)

dE = 0.0
dH = 0.0
dG = 0.0

print(f"\n{'='*50}")
print(f"Reaction: H2 + F2 -> 2 HF")
print(f"{'='*50}\n")

print("Reactants:")
for name, spec in species.items():
    n = spec["stoich_reactant"]
    if n > 0:
        dE -= n * results[name]["E_elec"]
        dH -= n * results[name]["H_tot"]
        dG -= n * results[name]["G_tot"]
        print(f"  {n} x {name}: E = {results[name]['E_elec']:.10f}, "
              f"H = {results[name]['H_tot']:.10f}, G = {results[name]['G_tot']:.10f}")

print("Products:")
for name, spec in species.items():
    n = spec["stoich_product"]
    if n > 0:
        dE += n * results[name]["E_elec"]
        dH += n * results[name]["H_tot"]
        dG += n * results[name]["G_tot"]
        print(f"  {n} x {name}: E = {results[name]['E_elec']:.10f}, "
              f"H = {results[name]['H_tot']:.10f}, G = {results[name]['G_tot']:.10f}")

print(f"\nReaction energetics:")
print(f"  dE  = {dE:12.8f} Hartree = {dE * ha_to_kcal:8.2f} kcal/mol = {dE * ha_to_kj:8.2f} kJ/mol")
print(f"  dH  = {dH:12.8f} Hartree = {dH * ha_to_kcal:8.2f} kcal/mol = {dH * ha_to_kj:8.2f} kJ/mol")
print(f"  dG  = {dG:12.8f} Hartree = {dG * ha_to_kcal:8.2f} kcal/mol = {dG * ha_to_kj:8.2f} kJ/mol")

if dG < 0:
    print(f"\n  Reaction is thermodynamically FAVORABLE (exergonic) at {TEMPERATURE} K.")
else:
    print(f"\n  Reaction is thermodynamically UNFAVORABLE (endergonic) at {TEMPERATURE} K.")

if dH < 0:
    print(f"  Reaction is EXOTHERMIC.")
else:
    print(f"  Reaction is ENDOTHERMIC.")

# Equilibrium constant from dG
R = 8.314462618e-3  # kJ/mol/K
dG_kJ = dG * ha_to_kj
K_eq = np.exp(-dG_kJ / (R * TEMPERATURE))
print(f"\n  Equilibrium constant K = {K_eq:.4e}")
print(f"  (K > 1 means products are favored)")

# Save results
with open("reaction_energy.txt", "w") as f:
    f.write(f"Reaction: H2 + F2 -> 2 HF\n")
    f.write(f"Method: B3LYP/def2-TZVP\n")
    f.write(f"Temperature: {TEMPERATURE} K\n\n")
    for name, r in results.items():
        f.write(f"{name}: E = {r['E_elec']:.10f}, H = {r['H_tot']:.10f}, G = {r['G_tot']:.10f} Hartree\n")
    f.write(f"\ndE = {dE * ha_to_kcal:.2f} kcal/mol\n")
    f.write(f"dH = {dH * ha_to_kcal:.2f} kcal/mol\n")
    f.write(f"dG = {dG * ha_to_kcal:.2f} kcal/mol\n")
    f.write(f"K_eq = {K_eq:.4e}\n")
print("Saved: reaction_energy.txt")
```

---

### Workflow 4: Excited States (TD-DFT)

Compute the UV-Vis absorption spectrum of a molecule using time-dependent DFT (TD-DFT). Extracts excitation energies, oscillator strengths, and dominant orbital transitions.

```python
#!/usr/bin/env python3
"""
TD-DFT excited-state calculation using PySCF.
Computes UV-Vis absorption spectrum: excitation energies, oscillator strengths,
and dominant orbital contributions.

Example: formaldehyde (H2CO) -- classic n->pi* and pi->pi* transitions.
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pyscf import gto, dft, tdscf
from pyscf.geomopt.geometric_solver import optimize

# ============================================================
# 1. BUILD AND OPTIMIZE THE MOLECULE
# ============================================================

mol = gto.Mole()
mol.atom = """
    C   0.000000   0.000000   0.000000
    O   0.000000   0.000000   1.203000
    H   0.000000   0.935000  -0.587000
    H   0.000000  -0.935000  -0.587000
"""
mol.basis = "def2-tzvp"  # triple-zeta for better excited states
mol.charge = 0
mol.spin = 0
mol.verbose = 4
mol.build()

print(f"Molecule: {mol.natm} atoms, {mol.nelectron} electrons")

# Ground-state DFT
mf = dft.RKS(mol)
mf.xc = "pbe0"  # PBE0 hybrid, good for excited states
mf.grids.level = 4
mf.conv_tol = 1e-10
mf.kernel()

# Optimize geometry
print("\n=== Geometry Optimization ===")
mol_opt = optimize(mf, maxsteps=100)

# Recompute at optimized geometry
mf_opt = dft.RKS(mol_opt)
mf_opt.xc = "pbe0"
mf_opt.grids.level = 4
mf_opt.conv_tol = 1e-10
mf_opt.kernel()

print(f"Ground-state energy: {mf_opt.e_tot:.10f} Hartree")

# ============================================================
# 2. TD-DFT CALCULATION
# ============================================================

print("\n=== TD-DFT Excited States ===")

# Compute the lowest N excited singlet states
N_STATES = 10

td = tdscf.TDA(mf_opt)  # Tamm-Dancoff approximation (faster, usually accurate)
# For full TD-DFT (RPA), use: td = tdscf.TDDFT(mf_opt)
td.nstates = N_STATES
td.kernel()

# ============================================================
# 3. EXTRACT AND PRINT RESULTS
# ============================================================

# Excitation energies in Hartree
excitation_energies_ha = td.e

# Convert to eV and nm
ha_to_ev = 27.211386245988
excitation_energies_ev = excitation_energies_ha * ha_to_ev
excitation_energies_nm = 1239.84193 / excitation_energies_ev  # eV to nm

# Oscillator strengths
oscillator_strengths = td.oscillator_strength()

print(f"\n{'State':>5s}  {'E (eV)':>8s}  {'E (nm)':>8s}  {'f (osc)':>8s}  {'Character':>20s}")
print("-" * 60)

for i in range(N_STATES):
    E_ev = excitation_energies_ev[i]
    E_nm = excitation_energies_nm[i]
    f_osc = oscillator_strengths[i]

    # Classify the transition
    if f_osc < 0.001:
        character = "forbidden/dark"
    elif E_ev < 4.0:
        character = "low-energy"
    elif E_ev < 6.0:
        character = "UV"
    else:
        character = "deep UV"

    print(f"  S{i+1:2d}   {E_ev:8.4f}   {E_nm:8.1f}   {f_osc:8.5f}   {character:>20s}")

# ============================================================
# 4. ANALYZE DOMINANT ORBITAL TRANSITIONS
# ============================================================

print(f"\n=== Dominant Orbital Transitions ===")

nocc = mol_opt.nelectron // 2  # number of occupied orbitals

for i in range(min(N_STATES, 5)):  # analyze first 5 states in detail
    print(f"\nState S{i+1}: {excitation_energies_ev[i]:.4f} eV, f = {oscillator_strengths[i]:.5f}")

    # The transition amplitudes are in td.xy[i]
    # For TDA: td.xy[i] is a tuple (X,) where X has shape (nocc*nvirt,)
    # For full TD-DFT: td.xy[i] is (X, Y) each with shape (nocc*nvirt,)
    X = td.xy[i][0]  # transition amplitudes

    nvirt = mf_opt.mo_coeff.shape[1] - nocc

    # Reshape to (nocc, nvirt)
    X_reshaped = X.reshape(nocc, nvirt)

    # Find dominant contributions (|c|^2 > 0.1)
    contributions = []
    for occ_idx in range(nocc):
        for virt_idx in range(nvirt):
            coeff = X_reshaped[occ_idx, virt_idx]
            weight = coeff ** 2
            if weight > 0.05:
                homo_label = f"HOMO{'-' + str(nocc - 1 - occ_idx) if occ_idx < nocc - 1 else ''}"
                lumo_label = f"LUMO{'+' + str(virt_idx) if virt_idx > 0 else ''}"
                contributions.append((weight, homo_label, lumo_label, coeff))

    contributions.sort(reverse=True)
    for weight, occ_label, virt_label, coeff in contributions[:3]:
        print(f"    {occ_label:>8s} -> {virt_label:<8s}  weight = {weight:.3f}  (c = {coeff:+.4f})")

# ============================================================
# 5. SIMULATED UV-VIS SPECTRUM
# ============================================================

print("\n=== Simulated UV-Vis Spectrum ===")

def gaussian_broadening(x, center, strength, sigma=0.3):
    """Gaussian broadening for spectral lines. sigma in eV."""
    return strength / (sigma * np.sqrt(2 * np.pi)) * np.exp(-0.5 * ((x - center) / sigma) ** 2)

# Energy axis in eV
E_axis = np.linspace(0, 12, 2000)
spectrum = np.zeros_like(E_axis)

for i in range(N_STATES):
    spectrum += gaussian_broadening(
        E_axis,
        excitation_energies_ev[i],
        oscillator_strengths[i],
        sigma=0.3,  # broadening width in eV
    )

# Wavelength axis
nm_axis = np.linspace(100, 800, 2000)
spectrum_nm = np.zeros_like(nm_axis)
for i in range(N_STATES):
    if excitation_energies_nm[i] > 100 and excitation_energies_nm[i] < 800:
        spectrum_nm += gaussian_broadening(
            nm_axis,
            excitation_energies_nm[i],
            oscillator_strengths[i],
            sigma=15,  # broadening width in nm
        )

# Plot
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

# Energy axis plot
ax1.plot(E_axis, spectrum, "b-", linewidth=1.5)
for i in range(N_STATES):
    ax1.axvline(excitation_energies_ev[i], color="red", alpha=0.3, linewidth=0.5)
    if oscillator_strengths[i] > 0.01:
        ax1.plot(excitation_energies_ev[i], oscillator_strengths[i], "rv", markersize=8)
ax1.set_xlabel("Energy (eV)", fontsize=13)
ax1.set_ylabel("Oscillator Strength / Absorption (arb. units)", fontsize=12)
ax1.set_title("UV-Vis Absorption Spectrum (Energy)", fontsize=14)
ax1.set_xlim(0, 12)
ax1.grid(True, alpha=0.3)

# Wavelength axis plot
ax2.plot(nm_axis, spectrum_nm, "b-", linewidth=1.5)
for i in range(N_STATES):
    lam = excitation_energies_nm[i]
    if 100 < lam < 800 and oscillator_strengths[i] > 0.01:
        ax2.plot(lam, oscillator_strengths[i], "rv", markersize=8)
        ax2.annotate(f"S{i+1}\n{lam:.0f} nm",
                     (lam, oscillator_strengths[i]),
                     textcoords="offset points", xytext=(10, 5), fontsize=9)
ax2.set_xlabel("Wavelength (nm)", fontsize=13)
ax2.set_ylabel("Absorption (arb. units)", fontsize=12)
ax2.set_title("UV-Vis Absorption Spectrum (Wavelength)", fontsize=14)
ax2.set_xlim(100, 800)
ax2.invert_xaxis()
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("uv_vis_spectrum.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved: uv_vis_spectrum.png")

# Save stick spectrum data
with open("excitation_data.csv", "w") as f:
    f.write("State,Energy_eV,Wavelength_nm,Oscillator_Strength\n")
    for i in range(N_STATES):
        f.write(f"S{i+1},{excitation_energies_ev[i]:.6f},"
                f"{excitation_energies_nm[i]:.2f},{oscillator_strengths[i]:.6f}\n")
print("Saved: excitation_data.csv")
```

---

## Key Parameters

| Parameter | Typical Value | Notes |
|---|---|---|
| `mol.basis` | `"def2-svp"` (optimization), `"def2-tzvp"` (energies) | Double-zeta for geometry, triple-zeta for energetics. Use `"cc-pvtz"` for correlated methods. |
| `mol.spin` | 0 (singlet), 1 (doublet), 2 (triplet) | PySCF uses 2S (number of unpaired electrons), NOT 2S+1. |
| `mf.xc` | `"b3lyp"`, `"pbe0"`, `"wb97x-d"`, `"m06-2x"` | B3LYP is general-purpose; PBE0 for excited states; wB97X-D for non-covalent. |
| `mf.grids.level` | 3 (default), 4 (tight), 5 (very tight) | Higher grid = better numerical integration = slower. Use 4+ for frequencies. |
| `mf.conv_tol` | `1e-10` | SCF energy convergence in Hartree. Must be tight (1e-10 or smaller) for frequency calculations. |
| `optimize(maxsteps=)` | 100 | Maximum geometry optimization steps. Increase for floppy molecules. |
| `solvent_eps` | 78.39 (water), 46.7 (DMSO), 7.58 (THF) | Dielectric constant for ddCOSMO implicit solvent model. |
| `td.nstates` | 5--20 | Number of excited states. More states = slower. 10 is typical for UV-Vis. |
| `TEMPERATURE` | 298.15 K | Temperature for thermochemistry. Standard conditions. |
| Gaussian broadening `sigma` | 0.3 eV or 15 nm | Spectral broadening width. Adjust to match experimental resolution. |

## Interpreting Results

### Geometry optimization
- **Converged in N steps**: good. If it fails to converge, try a better initial guess (e.g., from a force field or SMILES+MMFF).
- Check that all frequencies are real (positive). Imaginary frequencies mean you found a saddle point, not a minimum.

### Thermochemistry
- **ZPE**: zero-point energy correction, always positive. Typically 5--50 kcal/mol depending on molecule size.
- **H(298K)**: enthalpy includes electronic energy + ZPE + thermal vibrational/rotational/translational corrections + RT (PV work).
- **G(298K)**: Gibbs free energy = H - TS. The entropy term typically makes G lower than H by 10--30 kcal/mol for molecules with many atoms.
- **Accuracy**: B3LYP/def2-SVP typically gives bond lengths within 0.01--0.03 A and frequencies within 3--5% of experiment (after ~0.96 scaling factor).

### Solvation
- **dG(solv) for water**: typical values are -2 to -15 kcal/mol for neutral organic molecules, -50 to -110 kcal/mol for ions.
- ddCOSMO is a good approximation but does not capture specific hydrogen bonding. For accurate solvation of ions or H-bonded species, consider adding explicit solvent molecules.

### Excited states
- **Oscillator strength f > 0.01**: optically bright (allowed) transition. f < 0.001: dark (forbidden) transition.
- **TD-DFT accuracy**: typically within 0.2--0.5 eV of experiment for valence excitations. Rydberg and charge-transfer states may have larger errors.
- **Charge-transfer states**: use range-separated functionals (wB97X-D, CAM-B3LYP) instead of B3LYP for better accuracy.

### Reaction energies
- **dG < 0**: thermodynamically favorable (exergonic). dG > 0: unfavorable (endergonic).
- **dH vs dG**: the difference is the -TdS term. Reactions that produce more gas-phase molecules typically have favorable entropy.
- **Accuracy**: B3LYP/def2-TZVP typically gives reaction energies within 3--5 kcal/mol. For chemical accuracy (1 kcal/mol), use CCSD(T)/CBS extrapolation.

## Common Issues

| Problem | Cause | Solution |
|---|---|---|
| SCF not converging | Poor initial guess, near-degenerate orbitals | Try `mf.init_guess = '1e'` or `'atom'`; use `mf.level_shift = 0.2`; try `mf = mf.newton()` for second-order SCF |
| Geometry optimization fails | Bad initial geometry, floppy molecule | Use RDKit MMFF pre-optimization for SMILES input; increase `maxsteps`; try a smaller basis for initial opt |
| Imaginary frequencies at "minimum" | Not fully converged, numerical noise | Tighten `conv_tol` to `1e-12`; increase `grids.level` to 5; re-optimize with tighter criteria |
| Wrong spin state | Incorrect `mol.spin` | Remember PySCF uses 2S (number of unpaired electrons), not multiplicity 2S+1. Singlet = 0, doublet = 1, triplet = 2. |
| Open-shell system gives wrong energy | Using RKS for open-shell | Use `dft.UKS(mol)` for open-shell systems; check `<S^2>` for spin contamination |
| Basis set linear dependency | Very large basis or diffuse functions | Remove diffuse functions, or set `mol.linear_dep_threshold = 1e-7` |
| TD-DFT charge-transfer error | B3LYP underestimates CT excitations | Switch to range-separated functional: `mf.xc = "wb97x-d"` or `"cam-b3lyp"` |
| Memory error for large molecule | Basis set too large | Use density fitting: `mf = mf.density_fit()`; reduce basis to def2-SVP; consider RI-MP2 |
| Solvent model crashes | Hessian not implemented for solvent | For ddCOSMO Hessian issues, compute Hessian numerically or use gas-phase Hessian as approximation for thermochemistry |
| Frequencies differ from experiment | Harmonic approximation, basis set | Apply scaling factor (~0.96 for B3LYP, ~0.95 for HF); use anharmonic corrections for high accuracy |
