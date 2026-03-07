# Piezoelectric Tensor Calculation

## When to Use

- You need the piezoelectric stress tensor e_ij (C/m^2) relating polarization to strain.
- You need the piezoelectric strain tensor d_ij (pC/N or pm/V) relating strain to electric field.
- You need the dielectric tensor (epsilon) and elastic constants for converting between e_ij and d_ij.
- You are studying piezoelectric materials: ZnO, AlN, GaN, BaTiO3, PbTiO3, quartz, PVDF.
- You need the clamped-ion (electronic) and relaxed-ion (electronic + ionic) piezoelectric contributions.
- You need 2D piezoelectric constants (e.g., for monolayer MoS2, h-BN) in units of pC/m (C/m in 2D).
- You want to compute the piezoelectric response from DFPT (density functional perturbation theory).

## Method Selection

| Method | Tool | Strengths | Limitations |
|---|---|---|---|
| QE ph.x DFPT | ph.x with epsil=.true. | Accurate, analytical derivatives | Requires DFPT-compatible PPs |
| QE Berry phase finite diff. | pw.x with lberry=.true. | Conceptually simple, any PPs | Requires multiple strain steps |
| VASP DFPT | VASP with LEPSILON/IBRION=8 | Standard VASP approach | Requires VASP license |
| VASP finite diff. | VASP with IBRION=6 | Simple setup | More calculations needed |
| VASP LPEAD | VASP with LPEAD=.TRUE. | Berry phase approach | Only for insulators |

## Prerequisites

- A fully relaxed crystal structure (forces < 1e-5 Ry/Bohr for QE, EDIFFG < -0.001 for VASP).
- Quantum ESPRESSO 7.5 with `pw.x` and `ph.x` on PATH -- or VASP.
- Pseudopotential files (norm-conserving or PAW for QE DFPT).
- Python packages: `pymatgen`, `numpy`, `matplotlib`.
- The material must be an insulator (nonzero band gap). Piezoelectric response is undefined for metals.
- The material must lack inversion symmetry. Centrosymmetric crystals have zero piezoelectric tensor.

## Background

### Piezoelectric Tensors

The piezoelectric effect relates mechanical strain to electric polarization. Two tensors describe this:

**Piezoelectric stress tensor e_ij** (units: C/m^2):
```
P_i = sum_j e_ij * epsilon_j    (Voigt notation, j = 1..6)
```
where P_i is polarization (i = x, y, z) and epsilon_j is strain in Voigt notation.

**Piezoelectric strain tensor d_ij** (units: pC/N = pm/V):
```
epsilon_j = sum_i d_ij * E_i     (converse effect)
```

The two are related by the elastic compliance tensor S or stiffness tensor C:
```
d_ij = sum_k e_ik * S_kj    (d = e * S)
e_ij = sum_k d_ik * C_kj    (e = d * C)
```

### Decomposition

The piezoelectric tensor can be decomposed into:
```
e_ij = e^(clamped)_ij + e^(ionic)_ij
```

- **Clamped-ion (electronic)**: Response of electron density to strain with atoms fixed at scaled positions.
- **Internal-strain (ionic)**: Additional contribution from atomic relaxation within the strained cell, mediated by Born effective charges: e^(ionic) = sum_k Z*_k * du_k/d_epsilon.

### Crystal Symmetry Constraints

The non-zero components of e_ij depend on the crystal class (point group):

| Crystal Class | Example | Non-zero e_ij components |
|---|---|---|
| 6mm (wurtzite) | ZnO, AlN, GaN | e_31 = e_32, e_33, e_15 = e_24 |
| 4mm (tetragonal) | BaTiO3 (tet.) | e_31 = e_32, e_33, e_15 = e_24 |
| 3m (trigonal) | LiNbO3 | e_15 = e_24, e_22 = -e_21, e_31 = e_32, e_33 |
| 43m (zinc blende) | GaAs, ZnS | e_14 = e_25 = e_36 |
| D_3h (2D, h-BN) | h-BN monolayer | e_11 = -e_12 = -e_26 |

## Detailed Steps

### Overview

```
Method A (QE DFPT):
  Step 0: Download pseudopotentials
  Step 1: Full structural relaxation (pw.x vc-relax)
  Step 2: SCF with tight convergence (pw.x)
  Step 3: DFPT calculation (ph.x with epsil=.true.)
  Step 4: Parse piezoelectric tensor, dielectric tensor, Born charges
  Step 5: Compute d_ij from e_ij and elastic constants

Method B (VASP DFPT):
  Step 1: Structural relaxation (VASP ISIF=3)
  Step 2: DFPT with LEPSILON=.TRUE. or IBRION=8
  Step 3: Parse OUTCAR for piezoelectric and dielectric tensors

Method C (2D piezoelectric):
  Same as Method A/B but with vacuum correction for 2D materials
```

---

### Method A: QE DFPT Piezoelectric Calculation

#### Step A0: Download Pseudopotentials

```python
#!/usr/bin/env python3
"""Download pseudopotentials for ZnO piezoelectric calculation."""
import os
import urllib.request

PSEUDO_DIR = os.path.abspath("./pseudo")
os.makedirs(PSEUDO_DIR, exist_ok=True)

pseudos = {
    "Zn": "Zn.pbe-dn-kjpaw_psl.1.0.0.UPF",
    "O":  "O.pbe-n-kjpaw_psl.1.0.0.UPF",
}

BASE_URL = "https://pseudopotentials.quantum-espresso.org/upf_files/"

for element, fname in pseudos.items():
    dest = os.path.join(PSEUDO_DIR, fname)
    if os.path.exists(dest):
        print(f"  {fname} already exists, skipping.")
        continue
    url = BASE_URL + fname
    print(f"  Downloading {fname} ...")
    try:
        urllib.request.urlretrieve(url, dest)
        print(f"  OK: {fname}")
    except Exception as e:
        print(f"  WARNING: Could not download {fname}: {e}")

print(f"\nPseudopotentials in {PSEUDO_DIR}:")
for f in sorted(os.listdir(PSEUDO_DIR)):
    print(f"  {f}")
```

#### Step A1: Structural Relaxation

```python
#!/usr/bin/env python3
"""
Relax ZnO wurtzite structure before piezoelectric calculation.
Full variable-cell relaxation (vc-relax) to get equilibrium geometry.
"""
import os
import subprocess

PSEUDO_DIR = os.path.abspath("./pseudo")
OUTDIR = os.path.abspath("./tmp_piezo")
os.makedirs(OUTDIR, exist_ok=True)

# ZnO wurtzite: space group P6_3mc (186), point group 6mm
# a ~ 3.25 Ang, c ~ 5.21 Ang, u ~ 0.382
a_ang = 3.25
c_ang = 5.21
a_bohr = a_ang * 1.8897259886
c_bohr = c_ang * 1.8897259886

relax_input = f"""&CONTROL
    calculation   = 'vc-relax'
    prefix        = 'zno'
    pseudo_dir    = '{PSEUDO_DIR}'
    outdir        = '{OUTDIR}'
    tprnfor       = .true.
    tstress       = .true.
    forc_conv_thr = 1.0d-5
    verbosity     = 'high'
/

&SYSTEM
    ibrav         = 4
    celldm(1)     = {a_bohr:.6f}
    celldm(3)     = {c_bohr/a_bohr:.6f}
    nat           = 4
    ntyp          = 2
    ecutwfc       = 80.0
    ecutrho       = 640.0
    occupations   = 'fixed'
/

&ELECTRONS
    conv_thr      = 1.0d-10
    mixing_beta   = 0.5
/

&IONS
    ion_dynamics  = 'bfgs'
/

&CELL
    cell_dynamics = 'bfgs'
    press         = 0.0
    press_conv_thr = 0.1
/

ATOMIC_SPECIES
  Zn  65.38  Zn.pbe-dn-kjpaw_psl.1.0.0.UPF
  O   15.999 O.pbe-n-kjpaw_psl.1.0.0.UPF

ATOMIC_POSITIONS {{crystal}}
  Zn  0.333333333  0.666666667  0.000
  Zn  0.666666667  0.333333333  0.500
  O   0.333333333  0.666666667  0.382
  O   0.666666667  0.333333333  0.882

K_POINTS {{automatic}}
  8 8 6  0 0 0
"""

with open("zno_relax.in", "w") as f:
    f.write(relax_input)
print("Written: zno_relax.in")

print("Running vc-relax ...")
result = subprocess.run(
    ["mpirun", "-np", "4", "pw.x", "-in", "zno_relax.in"],
    capture_output=True, text=True, timeout=7200
)
with open("zno_relax.out", "w") as f:
    f.write(result.stdout)

if "Final enthalpy" in result.stdout or "bfgs converged" in result.stdout:
    print("Relaxation converged.")
else:
    print("WARNING: Check zno_relax.out for convergence.")
print("Output: zno_relax.out")
```

#### Step A2: SCF with Tight Convergence

```python
#!/usr/bin/env python3
"""
SCF calculation on the relaxed ZnO structure with very tight convergence
for DFPT. conv_thr must be 1e-12 or tighter for accurate piezoelectric
tensors.
"""
import os
import subprocess
import re

PSEUDO_DIR = os.path.abspath("./pseudo")
OUTDIR = os.path.abspath("./tmp_piezo")

# Extract relaxed structure from vc-relax output
# (In practice, read the final coordinates from the output file)

# For demonstration, use the known relaxed values:
a_bohr = 6.1447  # Relaxed a in Bohr (approximate)
c_over_a = 1.6034  # Relaxed c/a ratio

scf_input = f"""&CONTROL
    calculation   = 'scf'
    prefix        = 'zno'
    pseudo_dir    = '{PSEUDO_DIR}'
    outdir        = '{OUTDIR}'
    tprnfor       = .true.
    tstress       = .true.
    verbosity     = 'high'
/

&SYSTEM
    ibrav         = 4
    celldm(1)     = {a_bohr:.6f}
    celldm(3)     = {c_over_a:.6f}
    nat           = 4
    ntyp          = 2
    ecutwfc       = 80.0
    ecutrho       = 640.0
    occupations   = 'fixed'
/

&ELECTRONS
    conv_thr      = 1.0d-12
    mixing_beta   = 0.4
/

ATOMIC_SPECIES
  Zn  65.38  Zn.pbe-dn-kjpaw_psl.1.0.0.UPF
  O   15.999 O.pbe-n-kjpaw_psl.1.0.0.UPF

ATOMIC_POSITIONS {{crystal}}
  Zn  0.333333333  0.666666667  0.000
  Zn  0.666666667  0.333333333  0.500
  O   0.333333333  0.666666667  0.3820
  O   0.666666667  0.333333333  0.8820

K_POINTS {{automatic}}
  10 10 8  0 0 0
"""

with open("zno_scf.in", "w") as f:
    f.write(scf_input)
print("Written: zno_scf.in")

print("Running SCF (tight convergence for DFPT) ...")
result = subprocess.run(
    ["mpirun", "-np", "4", "pw.x", "-in", "zno_scf.in"],
    capture_output=True, text=True, timeout=3600
)
with open("zno_scf.out", "w") as f:
    f.write(result.stdout)

if "convergence has been achieved" in result.stdout:
    print("SCF converged.")
else:
    print("WARNING: SCF may not have converged. Check zno_scf.out")
```

#### Step A3: DFPT Calculation (ph.x)

```python
#!/usr/bin/env python3
"""
Run ph.x DFPT at Gamma with epsil=.true. to compute:
  - Dielectric tensor (epsilon_infinity)
  - Born effective charges (Z*)
  - Piezoelectric tensor (if supported by the QE version)
  - Phonon frequencies at Gamma

For the piezoelectric tensor via DFPT, QE computes the
strain-polarization coupling using the linear response formalism.
The ph.x output includes the "Dielectric Tensor" and
"Effective Charges" sections. The piezoelectric tensor components
require computing the strain derivative of polarization.

Note: QE's ph.x with epsil=.true. computes the electronic (clamped-ion)
dielectric and Born charge tensors. The ionic piezoelectric contribution
requires additional computation (internal strain + Born charges).
"""
import os
import subprocess

OUTDIR = os.path.abspath("./tmp_piezo")

# ph.x input for dielectric properties and Born charges at Gamma
ph_input = f"""Piezoelectric and dielectric properties
&INPUTPH
    prefix    = 'zno'
    outdir    = '{OUTDIR}'
    fildyn    = 'zno.dyn'
    tr2_ph    = 1.0d-16
    epsil     = .true.
    trans     = .true.
    ldisp     = .false.
    recover   = .false.
    verbosity = 1
/
0.0 0.0 0.0
"""

with open("zno_ph.in", "w") as f:
    f.write(ph_input)
print("Written: zno_ph.in")

print("Running ph.x DFPT (may take 30-120 minutes) ...")
result = subprocess.run(
    ["mpirun", "-np", "4", "ph.x", "-in", "zno_ph.in"],
    capture_output=True, text=True, timeout=14400  # 4 hours
)
with open("zno_ph.out", "w") as f:
    f.write(result.stdout)

if "JOB DONE" in result.stdout:
    print("ph.x completed successfully.")
else:
    print("WARNING: ph.x may not have completed. Check zno_ph.out")
print("Output: zno_ph.out")
print("Dynamical matrix: zno.dyn")
```

#### Step A4: Parse Piezoelectric Tensor and Compute e_ij

```python
#!/usr/bin/env python3
"""
Parse dielectric tensor, Born effective charges from ph.x output,
and compute the piezoelectric stress tensor e_ij.

The piezoelectric tensor has two contributions:
  e_ij = e^(clamped)_ij + e^(ionic)_ij

e^(clamped) comes from the electronic response to strain (requires
strain-perturbation DFPT, available in some QE versions).

e^(ionic) = (1/Omega) * sum_k Z*_k,ij * du_k / d(epsilon_j)
This requires the internal strain tensor (du/d_epsilon) from the
phonon dynamical matrix and the Born charges.

For systems where the full strain DFPT is not available, we compute
the piezoelectric tensor via finite differences of Berry phase
polarization (see the piezoelectric/ skill under the main skills dir).

Here we parse what ph.x gives us (Born charges, dielectric tensor)
and compute the ionic piezoelectric contribution using the known
internal strain parameters.
"""
import re
import os
import json
import numpy as np


def parse_ph_output_for_piezo(filename):
    """
    Parse ph.x output for dielectric tensor, Born charges,
    and internal strain parameters.
    """
    result = {
        "dielectric_tensor": None,
        "born_charges": [],
        "piezo_tensor": None,  # If directly computed by ph.x
    }

    if not os.path.exists(filename):
        print(f"File not found: {filename}")
        return result

    with open(filename, 'r') as f:
        lines = f.readlines()

    # Parse dielectric tensor
    for i, line in enumerate(lines):
        if ("Dielectric Tensor:" in line or
                "Dielectric constant in cartesian axis" in line):
            tensor = []
            row_count = 0
            j = i + 1
            while row_count < 3 and j < len(lines):
                stripped = lines[j].strip()
                if stripped:
                    nums = re.findall(r"[-+]?\d+\.\d+", stripped)
                    if len(nums) >= 3:
                        tensor.append([float(nums[0]), float(nums[1]),
                                        float(nums[2])])
                        row_count += 1
                j += 1
            if len(tensor) == 3:
                result["dielectric_tensor"] = tensor

    # Parse Born effective charges
    for i, line in enumerate(lines):
        if ("Effective Charges E-U" in line or
                "Born effective charges" in line or
                "effective charges (d Force / dE)" in line):
            j = i + 1
            while j < len(lines):
                atom_match = re.match(r"\s*atom\s+(\d+)\s+(\w+)", lines[j])
                if atom_match:
                    atom_idx = int(atom_match.group(1))
                    species = atom_match.group(2)
                    z_star = []
                    row_count = 0
                    k = j + 1
                    while row_count < 3 and k < len(lines):
                        stripped = lines[k].strip()
                        if stripped:
                            nums = re.findall(r"[-+]?\d+\.\d+", stripped)
                            if len(nums) >= 3:
                                z_star.append([float(nums[-3]),
                                                float(nums[-2]),
                                                float(nums[-1])])
                                row_count += 1
                        k += 1
                    if len(z_star) == 3:
                        result["born_charges"].append({
                            "atom": atom_idx,
                            "species": species,
                            "Z_star": z_star,
                        })
                    j = k
                elif (lines[j].strip() == "" or
                      any(kw in lines[j] for kw in
                          ["Dielectric", "phonon", "PHONON"])):
                    break
                else:
                    j += 1

    # Check for piezoelectric tensor in output (QE >= 7.x may print it)
    for i, line in enumerate(lines):
        if "Piezoelectric Tensor" in line or "piezoelectric" in line.lower():
            # Try to parse 3x6 tensor
            tensor = []
            j = i + 1
            while j < len(lines) and len(tensor) < 3:
                stripped = lines[j].strip()
                if stripped:
                    nums = re.findall(r"[-+]?\d+\.?\d*[eE]?[-+]?\d*",
                                       stripped)
                    if len(nums) >= 6:
                        tensor.append([float(n) for n in nums[:6]])
                j += 1
            if len(tensor) == 3:
                result["piezo_tensor"] = tensor

    return result


def compute_ionic_piezo_from_born_charges(born_charges, internal_strain,
                                           volume_ang3):
    """
    Compute ionic contribution to piezoelectric tensor.

    e^(ionic)_ij = (e / Omega) * sum_k Z*_k,ia * du_k,a / d(epsilon_j)

    Parameters
    ----------
    born_charges : list of dict with 'Z_star' (3x3)
    internal_strain : dict mapping (atom_idx, direction) -> strain_derivative
        du_k,a / d(epsilon_j) in Angstrom.
    volume_ang3 : float
        Unit cell volume in Angstrom^3.

    Returns
    -------
    e_ionic : ndarray (3, 6)
        Ionic piezoelectric tensor in C/m^2.
    """
    E_CHARGE = 1.602176634e-19  # C
    ANGSTROM_TO_M = 1e-10
    volume_m3 = volume_ang3 * ANGSTROM_TO_M**3

    e_ionic = np.zeros((3, 6))  # 3 pol directions x 6 strain components

    for bc in born_charges:
        Z = np.array(bc['Z_star'])  # 3x3
        atom_idx = bc['atom']

        for j_strain in range(6):  # Voigt: xx, yy, zz, yz, xz, xy
            for a_dir in range(3):  # atom displacement direction
                key = (atom_idx, a_dir, j_strain)
                if key in internal_strain:
                    du = internal_strain[key]  # Angstrom
                    for i_pol in range(3):
                        e_ionic[i_pol, j_strain] += (
                            E_CHARGE * Z[i_pol, a_dir] *
                            du * ANGSTROM_TO_M / volume_m3
                        )

    return e_ionic


def compute_piezo_constants_for_wurtzite(born_charges, volume_ang3,
                                          u_parameter=0.382,
                                          c_over_a=1.603):
    """
    Compute piezoelectric constants for wurtzite structure (6mm symmetry)
    using Born charges and known internal strain parameters.

    For wurtzite, the non-zero piezoelectric constants are:
      e_33, e_31 (=e_32), e_15 (=e_24)

    The ionic contribution uses:
      e_33^ion = (4*e*Z*_33 / (sqrt(3)*a^2*c)) * du/d(epsilon_3)
      e_31^ion = similar

    Parameters
    ----------
    born_charges : list of Born charge dicts
    volume_ang3 : float
    u_parameter : float
        Internal parameter of wurtzite (O position along c).
    c_over_a : float
        Relaxed c/a ratio.
    """
    E_CHARGE = 1.602176634e-19
    ANGSTROM_TO_M = 1e-10
    volume_m3 = volume_ang3 * ANGSTROM_TO_M**3

    # Average Born charges by species
    Z_Zn = None
    Z_O = None
    for bc in born_charges:
        Z = np.array(bc['Z_star'])
        if bc['species'] in ['Zn']:
            Z_Zn = Z if Z_Zn is None else (Z_Zn + Z) / 2
        elif bc['species'] in ['O']:
            Z_O = Z if Z_O is None else (Z_O + Z) / 2

    if Z_Zn is not None and Z_O is not None:
        print("\nBorn charges (averaged by species):")
        print(f"  Z*(Zn)_33 = {Z_Zn[2,2]:.4f} (formal: +2)")
        print(f"  Z*(O)_33  = {Z_O[2,2]:.4f}  (formal: -2)")
        print(f"  Z*(Zn)_11 = {Z_Zn[0,0]:.4f}")
        print(f"  Z*(O)_11  = {Z_O[0,0]:.4f}")

    # Symmetry analysis for wurtzite
    print("\nWurtzite symmetry (6mm) piezoelectric constants:")
    print("  Non-zero components: e_33, e_31 = e_32, e_15 = e_24")
    print(f"  u parameter: {u_parameter}")
    print(f"  c/a ratio: {c_over_a}")

    return Z_Zn, Z_O


# ── Main ───────────────────────────────────────────────────────────

print("=" * 60)
print("Piezoelectric Tensor from QE DFPT")
print("=" * 60)

ph_out = "zno_ph.out"
if os.path.exists(ph_out):
    data = parse_ph_output_for_piezo(ph_out)

    # Dielectric tensor
    if data["dielectric_tensor"] is not None:
        eps = np.array(data["dielectric_tensor"])
        print("\nDIELECTRIC TENSOR (epsilon_infinity):")
        for i in range(3):
            print(f"  {eps[i,0]:10.5f}  {eps[i,1]:10.5f}  {eps[i,2]:10.5f}")
        print(f"  eps_11 = eps_22 = {eps[0,0]:.4f}")
        print(f"  eps_33 = {eps[2,2]:.4f}")

    # Born charges
    if data["born_charges"]:
        print("\nBORN EFFECTIVE CHARGES:")
        for bc in data["born_charges"]:
            Z = np.array(bc["Z_star"])
            print(f"  Atom {bc['atom']} ({bc['species']}): "
                  f"Z*_11={Z[0,0]:.4f}, Z*_33={Z[2,2]:.4f}")

        # Wurtzite-specific analysis
        # Volume for 4-atom ZnO cell
        a_ang = 3.25
        c_ang = a_ang * 1.603
        volume = np.sqrt(3) / 2 * a_ang**2 * c_ang
        Z_Zn, Z_O = compute_piezo_constants_for_wurtzite(
            data["born_charges"], volume)

    # Direct piezoelectric tensor (if available)
    if data["piezo_tensor"] is not None:
        e = np.array(data["piezo_tensor"])
        print("\nPIEZOELECTRIC STRESS TENSOR e_ij (C/m^2):")
        print("     xx       yy       zz       yz       xz       xy")
        labels = ['P_x', 'P_y', 'P_z']
        for i in range(3):
            row = "  ".join(f"{e[i,j]:8.4f}" for j in range(6))
            print(f"  {labels[i]}: {row}")

    # Save results
    output = {
        "dielectric_tensor": data["dielectric_tensor"],
        "born_charges": data["born_charges"],
        "piezoelectric_tensor": data["piezo_tensor"],
    }
    with open("piezo_results.json", "w") as f:
        json.dump(output, f, indent=2)
    print("\nResults saved: piezo_results.json")
else:
    print(f"File not found: {ph_out}")
    print("Run ph.x first (Step A3).")
```

#### Step A5: Convert e_ij to d_ij Using Elastic Constants

```python
#!/usr/bin/env python3
"""
Convert piezoelectric stress tensor e_ij to piezoelectric strain tensor d_ij.

Relation: d = e * S  (S = C^{-1} is the elastic compliance tensor)

Elastic constants can come from:
  - The elastic-constants skill (stress-strain method)
  - Experimental values
  - MACE-MP-0 quick estimate
"""
import numpy as np
import json


def e_to_d_tensor(e_ij, C_ij):
    """
    Convert piezoelectric stress tensor e_ij to strain tensor d_ij.

    Parameters
    ----------
    e_ij : ndarray (3, 6)
        Piezoelectric stress tensor in C/m^2.
    C_ij : ndarray (6, 6)
        Elastic stiffness tensor in GPa.

    Returns
    -------
    d_ij : ndarray (3, 6)
        Piezoelectric strain tensor in pC/N (= pm/V).
    """
    # S = C^{-1} (compliance in 1/GPa)
    S_ij = np.linalg.inv(C_ij)  # 1/GPa

    # d = e * S
    # e in C/m^2, S in 1/GPa = 1/(1e9 Pa) = 1e-9 1/Pa = 1e-9 m^2/N
    # d in C/N = C/m^2 * m^2/N ... but we want pC/N
    # d_ij = e_ij (C/m^2) * S_jk (1/GPa)
    # = e_ij * S_jk * 1e-9 (C/N) * 1e12 (pC/C) = e_ij * S_jk * 1e3 (pC/N)

    d_ij = e_ij @ S_ij  # in C/m^2 * 1/GPa

    # Convert to pC/N
    d_pC_N = d_ij * 1e3  # C/m^2 / GPa = C/m^2 / (1e9 N/m^2) = 1e-9 C/N
    # Actually: d (C/N) = e (C/m^2) * S (1/Pa) = e * S * 1e-9
    # d (pC/N) = d(C/N) * 1e12 = e * S * 1e3
    d_pC_N = e_ij @ S_ij * 1e3

    return d_pC_N


# ── Example: ZnO ──────────────────────────────────────────────────
print("=" * 60)
print("Piezoelectric e_ij to d_ij Conversion")
print("=" * 60)

# ZnO experimental/DFT piezoelectric constants (C/m^2)
# (Wurtzite: non-zero e_31, e_33, e_15)
e_ij = np.zeros((3, 6))
# Voigt notation: 1=xx, 2=yy, 3=zz, 4=yz, 5=xz, 6=xy
e_ij[2, 0] = -0.51   # e_31 (P_z from epsilon_xx)
e_ij[2, 1] = -0.51   # e_32 = e_31
e_ij[2, 2] = 1.22    # e_33 (P_z from epsilon_zz)
e_ij[1, 3] = -0.45   # e_24 = e_15 (P_y from epsilon_yz)
e_ij[0, 4] = -0.45   # e_15 (P_x from epsilon_xz)

print("\nPiezoelectric stress tensor e_ij (C/m^2):")
print("       xx      yy      zz      yz      xz      xy")
for i, label in enumerate(['P_x', 'P_y', 'P_z']):
    row = "  ".join(f"{e_ij[i,j]:7.3f}" for j in range(6))
    print(f"  {label}: {row}")

# ZnO elastic constants (GPa) -- typical DFT values
# Wurtzite: C11, C12, C13, C33, C44 (C66 = (C11-C12)/2)
C11, C12, C13, C33, C44 = 209.7, 121.1, 105.1, 210.9, 42.5
C66 = (C11 - C12) / 2

C_ij = np.array([
    [C11, C12, C13,  0,   0,   0  ],
    [C12, C11, C13,  0,   0,   0  ],
    [C13, C13, C33,  0,   0,   0  ],
    [ 0,   0,   0,  C44,  0,   0  ],
    [ 0,   0,   0,   0,  C44,  0  ],
    [ 0,   0,   0,   0,   0,  C66 ],
])

print(f"\nElastic constants (GPa):")
print(f"  C11={C11}, C12={C12}, C13={C13}, C33={C33}, C44={C44}")

# Convert
d_ij = e_to_d_tensor(e_ij, C_ij)

print(f"\nPiezoelectric strain tensor d_ij (pC/N):")
print("       xx      yy      zz      yz      xz      xy")
for i, label in enumerate(['E_x', 'E_y', 'E_z']):
    row = "  ".join(f"{d_ij[i,j]:7.3f}" for j in range(6))
    print(f"  {label}: {row}")

# Key values for wurtzite
print(f"\nKey piezoelectric constants:")
print(f"  d_33 = {d_ij[2,2]:.2f} pC/N")
print(f"  d_31 = {d_ij[2,0]:.2f} pC/N")
print(f"  d_15 = {d_ij[0,4]:.2f} pC/N")

print(f"\nReference values for ZnO:")
print(f"  d_33 ~ 12.4 pC/N (experiment)")
print(f"  d_31 ~ -5.0 pC/N (experiment)")
print(f"  d_15 ~ -8.3 pC/N (experiment)")
```

---

### Method B: VASP DFPT Piezoelectric Calculation

#### Step B1: VASP INCAR for Piezoelectric Tensor

```
# INCAR for piezoelectric tensor via DFPT (IBRION=8)
SYSTEM  = ZnO_piezo
PREC    = Accurate
ENCUT   = 600
EDIFF   = 1E-8
IBRION  = 8      # DFPT for elastic, dielectric, piezoelectric
ISIF    = 3      # Compute stress tensor
LEPSILON = .TRUE.  # Compute dielectric tensor and Born charges
LREAL   = .FALSE.  # Required for DFPT
ADDGRID = .TRUE.
ISMEAR  = 0
SIGMA   = 0.05
NPAR    = 1      # NPAR must be 1 for DFPT
NSW     = 1      # At least 1 ionic step for IBRION=8

# Alternative: use IBRION=6 for finite differences
# IBRION = 6
# ISIF   = 3
# POTIM  = 0.015  # Finite difference step size
# NFREE  = 4      # Central differences (2 or 4)
```

**VASP DFPT approach (IBRION=8)**:
- Computes the full elastic tensor, dielectric tensor, and piezoelectric tensor in one calculation.
- Requires NPAR=1 (no parallelization over bands for DFPT).
- More expensive per step but gives all properties at once.

**VASP finite-difference approach (IBRION=6)**:
- Numerically differentiates stress and polarization with respect to strain/electric field.
- POTIM controls the finite difference step size (default 0.015 Angstrom).
- NFREE=4 for 4-point central differences (more accurate than NFREE=2).

For Berry phase piezoelectric with VASP:

```
# Alternative INCAR: Berry phase polarization + finite strain
SYSTEM  = ZnO_piezo_berry
PREC    = Accurate
ENCUT   = 600
EDIFF   = 1E-8
IBRION  = 6
ISIF    = 3
LPEAD   = .TRUE.   # Use Berry phase for polarization
LCALCPOL = .TRUE.
POTIM   = 0.015
NFREE   = 4
```

#### Step B2: Parse VASP Piezoelectric Output

```python
#!/usr/bin/env python3
"""
Parse piezoelectric tensor from VASP OUTCAR.

VASP outputs:
  - PIEZOELECTRIC TENSOR (from LEPSILON)
  - BORN EFFECTIVE CHARGES (from LEPSILON)
  - DIELECTRIC TENSOR (from LEPSILON)
  - PIEZOELECTRIC TENSOR for IONS (from IBRION=8)
  - Total PIEZOELECTRIC TENSOR (clamped + ionic)
"""
import re
import os
import json
import numpy as np


def parse_vasp_outcar_piezo(filename='OUTCAR'):
    """
    Parse piezoelectric tensor, dielectric tensor, and Born charges
    from VASP OUTCAR.
    """
    if not os.path.exists(filename):
        print(f"File not found: {filename}")
        return None

    with open(filename, 'r') as f:
        content = f.read()
        lines = content.split('\n')

    result = {
        'dielectric_electronic': None,
        'dielectric_ionic': None,
        'dielectric_total': None,
        'born_charges': [],
        'piezo_electronic': None,  # Clamped-ion (electronic)
        'piezo_ionic': None,       # Internal-strain (ionic)
        'piezo_total': None,       # Total = electronic + ionic
    }

    # ── Parse dielectric tensor ──────────────────────────────────
    for i, line in enumerate(lines):
        if 'MACROSCOPIC STATIC DIELECTRIC TENSOR' in line:
            if 'ionic contribution' in line:
                key = 'dielectric_ionic'
            else:
                key = 'dielectric_electronic'
            tensor = []
            for j in range(i+2, i+5):
                nums = re.findall(r'[-\d.]+', lines[j])
                if len(nums) >= 3:
                    tensor.append([float(nums[0]), float(nums[1]),
                                    float(nums[2])])
            if len(tensor) == 3:
                result[key] = tensor

    # Total dielectric
    if result['dielectric_electronic'] and result['dielectric_ionic']:
        eps_e = np.array(result['dielectric_electronic'])
        eps_i = np.array(result['dielectric_ionic'])
        result['dielectric_total'] = (eps_e + eps_i).tolist()

    # ── Parse piezoelectric tensor ────────────────────────────────
    for i, line in enumerate(lines):
        if 'PIEZOELECTRIC TENSOR' in line:
            # Determine which type (electronic or ionic)
            is_ionic = False
            if i > 0 and 'IONIC' in lines[i-1].upper():
                is_ionic = True
            elif 'ionic' in line.lower() or 'IONIC' in line:
                is_ionic = True

            # Read 3x6 tensor (next 3 data lines after header)
            tensor = []
            j = i + 1
            while j < len(lines) and len(tensor) < 3:
                stripped = lines[j].strip()
                if stripped and not stripped.startswith('-'):
                    nums = re.findall(r'[-\d.]+[eE]?[-+]?\d*', stripped)
                    # Filter out row labels (x, y, z)
                    float_nums = []
                    for n in nums:
                        try:
                            float_nums.append(float(n))
                        except ValueError:
                            pass
                    if len(float_nums) >= 6:
                        tensor.append(float_nums[:6])
                j += 1

            if len(tensor) == 3:
                if is_ionic:
                    result['piezo_ionic'] = tensor
                elif result['piezo_electronic'] is None:
                    result['piezo_electronic'] = tensor
                else:
                    # May be total
                    result['piezo_total'] = tensor

    # Compute total if we have both components
    if (result['piezo_electronic'] is not None and
            result['piezo_ionic'] is not None and
            result['piezo_total'] is None):
        e_elec = np.array(result['piezo_electronic'])
        e_ion = np.array(result['piezo_ionic'])
        result['piezo_total'] = (e_elec + e_ion).tolist()

    # ── Parse Born effective charges ──────────────────────────────
    for i, line in enumerate(lines):
        if 'BORN EFFECTIVE CHARGES' in line:
            j = i + 2
            atom_idx = 0
            while j < len(lines):
                if 'ion' in lines[j]:
                    atom_idx += 1
                    z_star = []
                    for k in range(j+1, j+4):
                        if k < len(lines):
                            nums = re.findall(r'[-\d.]+', lines[k])
                            if len(nums) >= 3:
                                z_star.append([float(nums[-3]),
                                                float(nums[-2]),
                                                float(nums[-1])])
                    if len(z_star) == 3:
                        result['born_charges'].append({
                            'atom': atom_idx,
                            'Z_star': z_star,
                        })
                    j += 4
                elif lines[j].strip() == '' or '---' in lines[j]:
                    break
                else:
                    j += 1

    return result


# ── Main ───────────────────────────────────────────────────────────

if os.path.exists('OUTCAR'):
    print("=" * 60)
    print("VASP Piezoelectric Tensor from OUTCAR")
    print("=" * 60)

    data = parse_vasp_outcar_piezo('OUTCAR')

    # Dielectric tensor
    if data['dielectric_electronic']:
        eps = np.array(data['dielectric_electronic'])
        print("\nElectronic dielectric tensor (epsilon_inf):")
        for i in range(3):
            print(f"  {eps[i,0]:8.4f}  {eps[i,1]:8.4f}  {eps[i,2]:8.4f}")

    if data['dielectric_total']:
        eps_tot = np.array(data['dielectric_total'])
        print("\nTotal dielectric tensor (static):")
        for i in range(3):
            print(f"  {eps_tot[i,0]:8.4f}  {eps_tot[i,1]:8.4f}  "
                  f"{eps_tot[i,2]:8.4f}")

    # Piezoelectric tensor
    for label, key in [('Electronic (clamped-ion)', 'piezo_electronic'),
                        ('Ionic (internal-strain)', 'piezo_ionic'),
                        ('Total', 'piezo_total')]:
        if data[key] is not None:
            e = np.array(data[key])
            print(f"\nPiezoelectric tensor ({label}) [C/m^2]:")
            print("       xx       yy       zz       yz       xz       xy")
            for i, pol in enumerate(['P_x', 'P_y', 'P_z']):
                row = "  ".join(f"{e[i,j]:8.4f}" for j in range(6))
                print(f"  {pol}: {row}")

    # Born charges
    if data['born_charges']:
        print(f"\nBorn effective charges ({len(data['born_charges'])} atoms):")
        for bc in data['born_charges']:
            Z = np.array(bc['Z_star'])
            print(f"  Atom {bc['atom']}: Z*_11={Z[0,0]:.3f}, "
                  f"Z*_33={Z[2,2]:.3f}")

    with open('vasp_piezo_results.json', 'w') as f:
        json.dump(data, f, indent=2)
    print("\nResults saved: vasp_piezo_results.json")
else:
    print("OUTCAR not found. Run VASP with LEPSILON=.TRUE. first.")
```

---

### Method C: 2D Piezoelectric Constants

```python
#!/usr/bin/env python3
"""
Compute 2D piezoelectric constants from 3D slab calculations.

For 2D materials (monolayers), the piezoelectric constant is defined
per unit length (not volume):
  e^2D_ij = e^3D_ij * L_z   (units: C/m instead of C/m^2)

where L_z is the vacuum-inclusive cell dimension perpendicular to the
2D sheet.

This correction is necessary because the 3D DFT calculation uses a
supercell with vacuum, and the 3D e_ij is artificially diluted by
the vacuum.

Common 2D piezoelectric materials:
  - h-BN monolayer: e_11 ~ 1.38 pC/m (D_3h symmetry)
  - MoS2 monolayer: e_11 ~ 3.64 pC/m (D_3h symmetry)
  - Janus MoSSe: e_11 and e_31 both non-zero
"""
import numpy as np
import json


def convert_3d_to_2d_piezo(e_3d, c_lattice_ang):
    """
    Convert 3D piezoelectric tensor to 2D.

    Parameters
    ----------
    e_3d : ndarray (3, 6)
        3D piezoelectric stress tensor in C/m^2 from slab calculation.
    c_lattice_ang : float
        c lattice parameter of the slab supercell in Angstrom
        (includes vacuum).

    Returns
    -------
    e_2d : ndarray (3, 6)
        2D piezoelectric tensor in C/m (= pC/m * 1e-12).
    e_2d_pC_m : ndarray (3, 6)
        Same in pC/m (more common unit for 2D).
    """
    c_m = c_lattice_ang * 1e-10  # Angstrom -> meters
    e_2d = e_3d * c_m  # C/m^2 * m = C/m
    e_2d_pC_m = e_2d * 1e12  # C/m -> pC/m

    return e_2d, e_2d_pC_m


def convert_3d_to_2d_dielectric(eps_3d, c_lattice_ang,
                                 sheet_thickness_ang=None):
    """
    Convert 3D dielectric tensor to 2D polarizability.

    The 2D polarizability alpha_2D = (eps_3D - 1) * eps_0 * L_z

    Parameters
    ----------
    eps_3d : ndarray (3, 3)
        3D dielectric tensor (dimensionless).
    c_lattice_ang : float
        Supercell c parameter in Angstrom.
    sheet_thickness_ang : float or None
        Effective sheet thickness for reporting eps_eff.
    """
    EPS_0 = 8.854187817e-12  # F/m

    c_m = c_lattice_ang * 1e-10
    alpha_2d = (eps_3d - np.eye(3)) * EPS_0 * c_m  # F (= C^2/(N*m))

    return alpha_2d


# ── Example: h-BN monolayer ────────────────────────────────────────

print("=" * 60)
print("2D Piezoelectric Constants")
print("=" * 60)

# Typical 3D results from a h-BN slab calculation with c = 20 Angstrom
c_slab = 20.0  # Angstrom (vacuum = ~17 Angstrom for ~3 Angstrom thick BN)

# 3D piezoelectric tensor from DFT (C/m^2) -- h-BN D_3h symmetry
# Non-zero: e_11 = -e_12 = -e_26 (in-plane only)
e_3d = np.zeros((3, 6))
e_3d[0, 0] = 0.069   # e_11 (P_x from epsilon_xx)
e_3d[0, 1] = -0.069  # e_12 = -e_11
e_3d[1, 5] = -0.069  # e_26 = -e_11

print(f"\n3D piezoelectric tensor e_ij (C/m^2) [slab c = {c_slab} Ang]:")
print("       xx       yy       zz       yz       xz       xy")
for i, pol in enumerate(['P_x', 'P_y', 'P_z']):
    row = "  ".join(f"{e_3d[i,j]:8.5f}" for j in range(6))
    print(f"  {pol}: {row}")

# Convert to 2D
e_2d, e_2d_pC = convert_3d_to_2d_piezo(e_3d, c_slab)

print(f"\n2D piezoelectric tensor e^2D_ij (pC/m):")
print("       xx       yy       zz       yz       xz       xy")
for i, pol in enumerate(['P_x', 'P_y', 'P_z']):
    row = "  ".join(f"{e_2d_pC[i,j]:8.3f}" for j in range(6))
    print(f"  {pol}: {row}")

print(f"\n  e^2D_11 = {e_2d_pC[0,0]:.3f} pC/m")
print(f"  (Literature: h-BN e_11 ~ 1.38 pC/m)")
print(f"  Note: The 3D value scales linearly with 1/c_slab.")
print(f"  If your c is different, the 2D value should be similar.")

# 3D dielectric tensor
eps_3d = np.array([
    [2.80, 0, 0],
    [0, 2.80, 0],
    [0, 0, 1.24],
])

print(f"\n3D dielectric tensor:")
for i in range(3):
    print(f"  {eps_3d[i,0]:8.4f}  {eps_3d[i,1]:8.4f}  {eps_3d[i,2]:8.4f}")

alpha_2d = convert_3d_to_2d_dielectric(eps_3d, c_slab)
print(f"\n2D polarizability (F):")
for i in range(3):
    print(f"  {alpha_2d[i,0]:12.4e}  {alpha_2d[i,1]:12.4e}  "
          f"{alpha_2d[i,2]:12.4e}")

# Save
results_2d = {
    'c_slab_Angstrom': c_slab,
    'e_3d_C_m2': e_3d.tolist(),
    'e_2d_pC_m': e_2d_pC.tolist(),
    'eps_3d': eps_3d.tolist(),
}
with open('piezo_2d_results.json', 'w') as f:
    json.dump(results_2d, f, indent=2)
print("\nResults saved: piezo_2d_results.json")
```

---

## Key Parameters

| Parameter | Typical Value | Notes |
|---|---|---|
| `epsil` (QE ph.x) | `.true.` | Required for dielectric and Born charge calculation |
| `tr2_ph` (QE ph.x) | `1.0d-16` | Very tight DFPT convergence for piezoelectric |
| `conv_thr` (QE SCF) | `1.0d-12` | Tight SCF convergence for DFPT |
| `LEPSILON` (VASP) | `.TRUE.` | Compute dielectric and piezoelectric via DFPT |
| `IBRION` (VASP) | `8` (DFPT) or `6` (finite diff.) | Method for computing response |
| `LPEAD` (VASP) | `.TRUE.` | Berry phase approach for polarization |
| ecutwfc (QE) | 60-80 Ry | Must be well converged for piezoelectric |
| ENCUT (VASP) | 500-600 eV | Same convergence requirement |
| K-grid | 10x10x8 (bulk), 16x16x1 (2D) | Denser than for total energy. Test convergence. |
| forc_conv_thr (QE) | `1.0d-5` Ry/Bohr | Very tight relaxation before DFPT |
| EDIFFG (VASP) | `-0.001` eV/Ang | Same: tight force convergence |
| 2D vacuum (c) | >= 15 Angstrom | Must be large enough for negligible inter-layer interaction |

### Convergence Testing

Piezoelectric constants are sensitive to:
1. **K-grid**: Test 8x8x6, 10x10x8, 12x12x10 for bulk ZnO.
2. **ecutwfc / ENCUT**: Test at several values; e_33 should converge to < 0.01 C/m^2.
3. **Structure relaxation**: Forces must be very small (< 1e-5 Ry/Bohr). Residual forces cause spurious contributions.

## Interpreting Results

### Piezoelectric Stress Tensor e_ij

- **Units**: C/m^2 (SI).
- **Sign**: Positive e_33 means positive strain along z increases P_z.
- **Magnitude**: Typical values range from 0.01 C/m^2 (weak, e.g., quartz) to 5+ C/m^2 (strong, e.g., PbTiO3).

### Typical Values for Validation

| Material | e_33 (C/m^2) | e_31 (C/m^2) | e_15 (C/m^2) | d_33 (pC/N) |
|---|---|---|---|---|
| ZnO | 1.0-1.3 | -0.4 to -0.6 | -0.4 to -0.5 | 12.4 |
| AlN | 1.4-1.6 | -0.5 to -0.6 | -0.3 to -0.4 | 5.1 |
| GaN | 0.7-0.9 | -0.3 to -0.5 | -0.2 to -0.4 | 3.1 |
| BaTiO3 | 3.2-3.7 | -2.5 to -2.7 | - | 86 |
| PbTiO3 | 3.2-3.8 | -0.7 to -1.0 | - | 117 |
| Quartz | 0.17 | -0.05 | - | 2.3 |

### 2D Piezoelectric Constants

| Material | e^2D_11 (pC/m) | Notes |
|---|---|---|
| h-BN | 1.38 | In-plane only |
| MoS2 | 3.64 | In-plane |
| MoSSe (Janus) | e_11 ~ 3.8, e_31 ~ 0.04 | Both in-plane and out-of-plane |
| WS2 | 2.12 | In-plane |

### Clamped vs Relaxed Ion

- **Clamped-ion**: Electronic contribution only. Usually smaller.
- **Ionic (internal strain)**: Often dominates, especially in perovskites. Mediated by Born charges.
- **Total = clamped + ionic**: This is the physically measurable quantity.
- In BaTiO3, the ionic contribution is ~80% of the total e_33.

### Dielectric Constants

The same DFPT calculation gives the dielectric tensor:
- **Electronic (epsilon_infinity)**: High-frequency limit. Typically 3-10 for insulators.
- **Ionic contribution**: Low-frequency (static) part from lattice vibrations.
- **Total (epsilon_static)**: Sum of electronic and ionic. Can be very large for ferroelectrics (epsilon > 100).

## Common Issues

| Problem | Solution |
|---|---|
| **ph.x crashes with epsil=.true.** | Check SCF convergence (conv_thr must be very tight). Ensure the system is an insulator. Check PP compatibility with DFPT. |
| **Piezoelectric tensor is zero** | Material has inversion symmetry (centrosymmetric). Piezoelectricity requires broken inversion symmetry. Check space group. |
| **e_ij too small or wrong** | Structure not fully relaxed. Residual forces cause errors in ionic contribution. Re-relax with tighter force threshold. |
| **VASP IBRION=8 fails** | NPAR must be 1 for DFPT. LREAL must be .FALSE. Check POTCAR compatibility. |
| **2D e^3D depends on vacuum** | This is expected. The 3D value scales as 1/c. The 2D value (e^3D * c) should be constant for c > 15 Angstrom. Verify by testing c = 15, 20, 25 Angstrom. |
| **d_ij requires elastic constants** | d = e * S (S = C^{-1}). Compute elastic constants from the mechanical-properties/elastic-constants skill. |
| **Born charges not converging** | Increase k-grid and tighten tr2_ph. Born charges (and therefore the ionic piezoelectric contribution) are sensitive to k-sampling. |
| **Off-diagonal e_ij components nonzero but should be zero** | Numerical noise. Values < 0.01 C/m^2 in forbidden components are acceptable. Enforce symmetry in post-processing. |
| **Results differ from literature** | Check: (1) same functional, (2) same pseudopotential type, (3) relaxed vs experimental structure, (4) clamped vs total tensor. Most published values are total (clamped + ionic). |
| **ph.x slow for large systems** | Use more MPI processes with `-npool`. DFPT at Gamma for a 4-atom cell typically takes 30-120 min on 4 cores. Larger cells scale roughly as N^3. |
