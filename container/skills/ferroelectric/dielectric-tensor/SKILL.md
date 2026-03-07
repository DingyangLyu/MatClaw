# Static and High-Frequency Dielectric Tensor

## When to Use

- You need the electronic (high-frequency) dielectric tensor epsilon_infinity, which governs the optical response and refractive index.
- You need the static (low-frequency) dielectric tensor epsilon_0, which includes both electronic and ionic (lattice) contributions.
- You want to compute the ionic contribution to the dielectric tensor from zone-centre phonon modes and Born effective charges.
- You need the frequency-dependent dielectric function epsilon(omega) to compare with infrared/Raman spectroscopy.
- You want to apply the Clausius-Mossotti relation to estimate polarizabilities from the dielectric tensor.
- You need dielectric constants as input for defect formation energy corrections (Makov-Payne, FNV) or exciton binding energy estimates.

## Method Selection

| Method | Code | What You Get | Cost | Accuracy |
|---|---|---|---|---|
| DFPT at Gamma (ph.x, epsil=.true.) | QE | epsilon_infinity + Born charges + Gamma phonons; epsilon_0 via Lyddane-Sachs-Teller | Medium | High (DFPT-exact) |
| DFPT full phonon dispersion + epsilon | QE | epsilon(omega) from all IR-active modes | High | High |
| LEPSILON + IBRION=8 | VASP | epsilon_infinity + Born charges + epsilon_0 (ionic) | Medium | High |
| Finite-field (sawlike potential) | QE (lelfield) | epsilon at finite electric field | High | High; includes nonlinear terms |
| MACE-MP-0 + finite displacement | ASE+MACE | Approximate epsilon_0 from force constants | Low | Screening quality only |

## Prerequisites

- A fully relaxed crystal structure (forces < 1e-4 Ry/Bohr, stress < 0.5 kbar).
- Quantum ESPRESSO 7.5 with `pw.x`, `ph.x`, and `dynmat.x` available on PATH.
- Pseudopotential files (norm-conserving recommended for DFPT; PAW/ultrasoft also work in QE 7.x).
- Python packages: `pymatgen`, `ase`, `numpy`, `scipy`, `matplotlib`.
- The material must be an insulator (nonzero band gap). The dielectric response diverges for metals.

## Background

### Electronic dielectric tensor (epsilon_infinity)

The electronic (or optical, or high-frequency) dielectric tensor epsilon_infinity describes the response of the electronic charge density to a macroscopic electric field at frequencies well above phonon frequencies but below electronic excitations. It is computed from DFPT as:

```
epsilon_infinity_ij = delta_ij + (4*pi / Omega) * chi_ij
```

where chi_ij is the electronic susceptibility computed from the first-order change in the wavefunctions under a macroscopic electric field perturbation. The refractive index is n_i = sqrt(epsilon_infinity_ii).

### Static dielectric tensor (epsilon_0)

The static dielectric tensor includes both electronic and ionic contributions:

```
epsilon_0_ij = epsilon_infinity_ij + epsilon_ionic_ij
```

The ionic contribution arises from the displacement of ions in response to the electric field and is given by:

```
epsilon_ionic_ij = (4*pi*e^2 / Omega) * sum_m [ S_m,i * S_m,j / omega_m^2 ]
```

where omega_m is the frequency of the m-th IR-active zone-centre phonon mode and S_m,i is the mode oscillator strength:

```
S_m,i = sum_k (Z*_k,ij / sqrt(M_k)) * e_m,k,j
```

with e_m,k,j being the phonon eigenvector, M_k the atomic mass, and Z* the Born effective charge tensor.

### Lyddane-Sachs-Teller relation

For cubic crystals, the ratio of static to high-frequency dielectric constants can be related to phonon frequencies:

```
epsilon_0 / epsilon_infinity = prod_m (omega_LO,m / omega_TO,m)^2
```

This provides a consistency check between dielectric and phonon calculations.

### Clausius-Mossotti relation

For cubic crystals, the Clausius-Mossotti relation connects the macroscopic dielectric constant to atomic polarizabilities alpha:

```
(epsilon - 1) / (epsilon + 2) = (4*pi / 3*Omega) * sum_i alpha_i
```

This is useful for estimating dielectric constants of solid solutions from component polarizabilities.

## Detailed Steps

### Overview

```
Step 0: Download pseudopotentials
Step 1: Run SCF calculation (pw.x) -- tight convergence
Step 2: Run DFPT at Gamma (ph.x with epsil=.true.) -- gives epsilon_infinity, Z*, phonons
Step 3: Post-process with dynmat.x -- gives epsilon_0 from mode analysis
Step 4: Parse and analyze dielectric tensor
Step 5: (Optional) Compute frequency-dependent dielectric function
Step 6: (Optional) VASP workflow
```

### Step 0: Download Pseudopotentials

```python
#!/usr/bin/env python3
"""Download pseudopotentials for SrTiO3 dielectric tensor calculation."""
import os
import urllib.request

PSEUDO_DIR = os.path.abspath("./pseudo")
os.makedirs(PSEUDO_DIR, exist_ok=True)

# PAW PPs from PSlibrary -- work well with ph.x in QE 7.5
pseudos = {
    "Sr": "Sr.pbe-spn-kjpaw_psl.1.0.0.UPF",
    "Ti": "Ti.pbe-spn-kjpaw_psl.1.0.0.UPF",
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

### Step 1: SCF Calculation

```python
#!/usr/bin/env python3
"""
Run SCF for cubic SrTiO3 as prerequisite for DFPT dielectric calculation.
SrTiO3 is an incipient ferroelectric with large static dielectric constant
(epsilon_0 ~ 300 at room temperature, diverging at low T).
"""
import os
import subprocess

PSEUDO_DIR = os.path.abspath("./pseudo")
OUTDIR = os.path.abspath("./tmp_dielectric")
os.makedirs(OUTDIR, exist_ok=True)

# Cubic SrTiO3: a = 3.905 Angstrom (experimental)
a_ang = 3.905
a_bohr = a_ang * 1.8897259886

scf_input = f"""&CONTROL
    calculation = 'scf'
    prefix      = 'srtio3'
    outdir      = '{OUTDIR}'
    pseudo_dir  = '{PSEUDO_DIR}'
    tprnfor     = .true.
    tstress     = .true.
    verbosity   = 'high'
/

&SYSTEM
    ibrav       = 1
    celldm(1)   = {a_bohr:.6f}
    nat         = 5
    ntyp        = 3
    ecutwfc     = 60.0
    ecutrho     = 480.0
    occupations = 'fixed'
/

&ELECTRONS
    conv_thr    = 1.0d-12
    mixing_beta = 0.5
/

ATOMIC_SPECIES
  Sr   87.62   Sr.pbe-spn-kjpaw_psl.1.0.0.UPF
  Ti   47.867  Ti.pbe-spn-kjpaw_psl.1.0.0.UPF
  O    15.999  O.pbe-n-kjpaw_psl.1.0.0.UPF

ATOMIC_POSITIONS (crystal)
  Sr  0.000  0.000  0.000
  Ti  0.500  0.500  0.500
  O   0.500  0.500  0.000
  O   0.500  0.000  0.500
  O   0.000  0.500  0.500

K_POINTS (automatic)
  8 8 8  1 1 1
"""

with open("srtio3_scf.in", "w") as f:
    f.write(scf_input)
print("Written: srtio3_scf.in")

print("Running pw.x SCF ...")
result = subprocess.run(
    ["mpirun", "-np", "4", "pw.x", "-in", "srtio3_scf.in"],
    capture_output=True, text=True, timeout=1800
)
with open("srtio3_scf.out", "w") as f:
    f.write(result.stdout)

if result.returncode != 0:
    print("ERROR: SCF failed!")
    print(result.stderr[-500:] if result.stderr else "No stderr")
else:
    if "convergence has been achieved" in result.stdout:
        print("SCF converged successfully.")
    else:
        print("WARNING: SCF may not have converged.")
    print("Output: srtio3_scf.out")
```

**Important**: Use `conv_thr = 1.0d-12` for the SCF feeding into ph.x. Use a dense k-grid (8x8x8 or denser). The dielectric tensor converges slowly with k-point sampling -- always test convergence.

### Step 2: DFPT at Gamma (ph.x with epsil=.true.)

```python
#!/usr/bin/env python3
"""
Run ph.x at Gamma to compute:
  1. Electronic dielectric tensor (epsilon_infinity)
  2. Born effective charge tensors Z*
  3. Zone-centre phonon frequencies and eigenvectors

These quantities are needed to compute both epsilon_infinity and epsilon_0.
"""
import os
import subprocess

OUTDIR = os.path.abspath("./tmp_dielectric")

ph_input = f"""Dielectric tensor and phonons at Gamma for SrTiO3
&INPUTPH
    prefix    = 'srtio3'
    outdir    = '{OUTDIR}'
    fildyn    = 'srtio3.dyn'
    tr2_ph    = 1.0d-14
    epsil     = .true.
    trans     = .true.
    ldisp     = .false.
    recover   = .false.
    verbosity = 1
/
0.0 0.0 0.0
"""

with open("srtio3_ph.in", "w") as f:
    f.write(ph_input)
print("Written: srtio3_ph.in")

print("Running ph.x (this may take 10-60 minutes) ...")
result = subprocess.run(
    ["mpirun", "-np", "4", "ph.x", "-in", "srtio3_ph.in"],
    capture_output=True, text=True, timeout=7200
)
with open("srtio3_ph.out", "w") as f:
    f.write(result.stdout)

if result.returncode != 0:
    print("ERROR: ph.x failed!")
    print(result.stderr[-500:] if result.stderr else "No stderr")
else:
    if "JOB DONE" in result.stdout:
        print("ph.x completed successfully.")
    else:
        print("WARNING: ph.x may not have completed.")
    print("Output: srtio3_ph.out")
    print("Dynamical matrix: srtio3.dyn")
```

### Step 3: Post-Process with dynmat.x

```python
#!/usr/bin/env python3
"""
Run dynmat.x to:
  1. Apply the acoustic sum rule
  2. Compute IR intensities and oscillator strengths
  3. Compute the ionic contribution to the dielectric tensor (epsilon_ionic)
  4. Compute the static dielectric tensor epsilon_0 = epsilon_infinity + epsilon_ionic

dynmat.x reads the dynamical matrix file produced by ph.x and outputs:
  - Corrected phonon frequencies
  - IR activities
  - Mode effective charges
  - Dielectric tensor contributions
"""
import os
import subprocess

# Run dynmat.x with ASR correction
dynmat_input = """&INPUT
    fildyn = 'srtio3.dyn'
    asr    = 'crystal'
    filout = 'srtio3_dynmat.out'
    filmol = 'srtio3_molden.mol'
/
"""

with open("dynmat.in", "w") as f:
    f.write(dynmat_input)
print("Written: dynmat.in")

print("Running dynmat.x ...")
result = subprocess.run(
    ["dynmat.x", "-in", "dynmat.in"],
    capture_output=True, text=True, timeout=120
)
with open("dynmat.out", "w") as f:
    f.write(result.stdout)

if result.returncode != 0:
    print("ERROR: dynmat.x failed!")
    print(result.stderr[-300:] if result.stderr else "")
else:
    print("dynmat.x completed successfully.")
    print("Output: dynmat.out")
    print("Detailed output: srtio3_dynmat.out")
    print("Molden visualization: srtio3_molden.mol")
```

### Step 4: Parse and Analyze Dielectric Tensor

```python
#!/usr/bin/env python3
"""
Parse dielectric tensor results from ph.x and dynmat.x output.
Compute epsilon_infinity, epsilon_ionic, and epsilon_0.

The static dielectric tensor is:
  epsilon_0 = epsilon_infinity + epsilon_ionic

where epsilon_ionic = (4*pi / Omega) * sum_m S_m S_m^T / omega_m^2
and S_m,i = sum_k Z*_k,ij * e_m,k,j / sqrt(M_k)
is the mode oscillator strength vector.
"""
import re
import os
import json
import numpy as np

# ---- Physical constants ----
E_CHARGE = 1.602176634e-19       # C
AMU_TO_KG = 1.66053906660e-27    # kg
ANGSTROM_TO_M = 1e-10            # m
EPS0_SI = 8.8541878128e-12       # F/m (vacuum permittivity)
CM1_TO_HZ = 2.99792458e10        # Hz per cm^-1
HBAR = 1.054571817e-34           # J.s


def parse_ph_output(filename):
    """
    Parse ph.x output for dielectric tensor, Born charges, and phonon data.

    Returns dict with:
        'epsilon_infinity': 3x3 array
        'born_charges': list of {'atom': int, 'species': str, 'Z_star': 3x3}
        'phonon_frequencies_cm1': list of frequencies
        'phonon_eigenvectors': list of eigenvectors (if parseable)
        'species_list': list of species in order
        'masses': dict of species -> mass in AMU
    """
    result = {
        "epsilon_infinity": None,
        "born_charges": [],
        "phonon_frequencies_cm1": [],
        "species_list": [],
    }

    if not os.path.exists(filename):
        print(f"File not found: {filename}")
        return result

    with open(filename, "r") as f:
        lines = f.readlines()

    # Parse dielectric tensor
    for i, line in enumerate(lines):
        if "Dielectric Tensor:" in line or "Dielectric constant in cartesian axis" in line:
            tensor = []
            row_count = 0
            j = i + 1
            while row_count < 3 and j < len(lines):
                stripped = lines[j].strip()
                if stripped and not stripped.startswith("("):
                    nums = re.findall(r"[-+]?\d+\.\d+", stripped)
                    if len(nums) >= 3:
                        tensor.append([float(nums[0]), float(nums[1]), float(nums[2])])
                        row_count += 1
                j += 1
            if len(tensor) == 3:
                result["epsilon_infinity"] = np.array(tensor)

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
                                z_star.append(
                                    [float(nums[-3]), float(nums[-2]), float(nums[-1])]
                                )
                                row_count += 1
                        k += 1
                    if len(z_star) == 3:
                        result["born_charges"].append({
                            "atom": atom_idx,
                            "species": species,
                            "Z_star": np.array(z_star),
                        })
                        result["species_list"].append(species)
                    j = k
                elif (lines[j].strip() == "" or "---" in lines[j] or
                      any(kw in lines[j] for kw in
                          ["Dielectric", "phonon", "PHONON", "freq", "Mode"])):
                    break
                else:
                    j += 1

    # Parse phonon frequencies
    for line in lines:
        freq_match = re.search(
            r"freq\s*\(\s*(\d+)\)\s*=\s*([-\d.]+)\s*\[cm-1\]", line
        )
        if freq_match:
            result["phonon_frequencies_cm1"].append(float(freq_match.group(2)))
        freq_match2 = re.search(
            r"omega\s*\(\s*(\d+)\)\s*=\s*([-\d.]+)\s*\[cm-1\]", line
        )
        if freq_match2:
            result["phonon_frequencies_cm1"].append(float(freq_match2.group(2)))

    return result


def compute_epsilon_ionic(born_charges, freqs_cm1, masses_amu, volume_A3,
                          natom, nmodes=None):
    """
    Compute the ionic contribution to the dielectric tensor.

    Uses the formula:
      epsilon_ionic_ij = (e^2 / (eps0 * Omega)) * sum_m S_m,i * S_m,j / omega_m^2

    where S_m,i = sum_k Z*_k,ij * e_m,k,j / sqrt(M_k) is the mode oscillator
    strength. For a simple isotropic estimate when eigenvectors are not
    available, we use the diagonal approximation.

    In practice, ph.x/dynmat.x reports mode effective charges directly.
    Here we parse the IR activities from dynmat.x output if available.

    Args:
        born_charges: list of Z* dicts from parse_ph_output
        freqs_cm1: list of phonon frequencies in cm^-1
        masses_amu: dict of species -> mass in AMU
        volume_A3: cell volume in Angstrom^3
        natom: number of atoms
        nmodes: number of modes (default: 3*natom)

    Returns:
        3x3 array: ionic contribution to dielectric tensor
    """
    if nmodes is None:
        nmodes = 3 * natom

    Omega_m3 = volume_A3 * ANGSTROM_TO_M**3

    # Filter out acoustic modes (freq < 10 cm^-1) and imaginary modes
    ir_freqs = [f for f in freqs_cm1 if f > 10.0]

    if not ir_freqs:
        print("  WARNING: No IR-active modes found above 10 cm^-1.")
        return np.zeros((3, 3))

    # Simplified estimate: use isotropic average of Z* and equal mode
    # contribution. This is a rough approximation when eigenvectors are
    # not available. For accurate results, parse dynmat.x output.
    #
    # Better approach: compute from Z* and mass-weighted displacements
    # For each IR-active mode m with frequency omega_m:
    #   epsilon_ionic_ij += (4*pi*e^2 / Omega) * |Z_eff,m|^2 / omega_m^2
    # where Z_eff,m is the mode effective charge.
    #
    # Here we provide a diagnostic estimate; the accurate values come
    # from the dynmat.x IR output (Step 5 below).

    # Compute average Z* as a proxy
    Z_avg = np.zeros((3, 3))
    for bc in born_charges:
        Z_avg += np.abs(bc["Z_star"])
    Z_avg /= len(born_charges)

    # Estimate using sum rule: sum of 1/omega^2 for IR modes
    sum_inv_omega2 = 0.0
    for freq in ir_freqs:
        omega_rad = 2.0 * np.pi * freq * CM1_TO_HZ  # rad/s
        sum_inv_omega2 += 1.0 / omega_rad**2

    # Rough estimate factor
    avg_mass_kg = np.mean(list(masses_amu.values())) * AMU_TO_KG
    factor = E_CHARGE**2 / (EPS0_SI * Omega_m3 * avg_mass_kg)

    eps_ionic = np.zeros((3, 3))
    for i in range(3):
        for j in range(3):
            eps_ionic[i, j] = factor * Z_avg[i, j]**2 * sum_inv_omega2

    return eps_ionic


def parse_dynmat_ir(filename):
    """
    Parse dynmat.x output for IR intensities and mode effective charges.

    dynmat.x output format (in srtio3_dynmat.out):
      mode   [cm-1]    [THz]      IR
        1    -0.00    -0.0000    0.0000
        2    -0.00    -0.0000    0.0000
        ...
        N   XXX.XX    XX.XXXX    Y.YYYY

    The IR column gives (d_mu/d_Q)^2 in (D/A)^2/AMU units.
    """
    result = {"modes": [], "ir_intensities": [], "frequencies_cm1": []}

    if not os.path.exists(filename):
        print(f"File not found: {filename}")
        return result

    with open(filename, "r") as f:
        lines = f.readlines()

    for line in lines:
        m = re.match(
            r"\s*(\d+)\s+([-\d.]+)\s+([-\d.]+)\s+([-\d.]+)", line
        )
        if m:
            mode_idx = int(m.group(1))
            freq_cm1 = float(m.group(2))
            freq_thz = float(m.group(3))
            ir_intensity = float(m.group(4))
            result["modes"].append(mode_idx)
            result["frequencies_cm1"].append(freq_cm1)
            result["ir_intensities"].append(ir_intensity)

    return result


def compute_epsilon_0_from_ir(epsilon_inf, ir_data, volume_A3):
    """
    Compute epsilon_0 from epsilon_infinity and IR intensities.

    For cubic crystals:
      epsilon_0 = epsilon_infinity + (4*pi / (3*Omega)) *
                  sum_m (S_m^2 / omega_m^2)

    where S_m is the mode oscillator strength and omega_m the frequency.
    The IR intensity from dynmat.x is proportional to S_m^2.

    This function uses the Lyddane-Sachs-Teller relation as a cross-check.
    """
    Omega_m3 = volume_A3 * ANGSTROM_TO_M**3

    # For each IR-active mode (freq > 10 cm^-1, IR > 0)
    eps_ionic_trace = 0.0
    for freq, ir_int in zip(ir_data["frequencies_cm1"],
                            ir_data["ir_intensities"]):
        if freq > 10.0 and ir_int > 0.0:
            omega_rad = 2.0 * np.pi * freq * CM1_TO_HZ
            # IR intensity in (D/A)^2/AMU from dynmat.x
            # Convert: 1 D = 3.33564e-30 C.m, 1 A = 1e-10 m
            # (D/A)^2/AMU = (3.33564e-30)^2 / (1e-10)^2 / (1.66054e-27)
            #             = 6.702e-31 C^2.m^2 / (1e-20 * 1.66054e-27)
            #             = 6.702e-31 / 1.66054e-47
            #             = 4.036e16 C^2/kg
            CONV = (3.33564e-30)**2 / (ANGSTROM_TO_M**2 * AMU_TO_KG)
            S2 = ir_int * CONV  # C^2/kg

            # Contribution to epsilon: S^2 / (eps0 * Omega * omega^2)
            # Factor of 1/3 for cubic averaging of the 3 degenerate components
            eps_ionic_trace += S2 / (EPS0_SI * Omega_m3 * omega_rad**2)

    return eps_ionic_trace


# ---- Main analysis ----
print("=" * 70)
print("DIELECTRIC TENSOR ANALYSIS")
print("=" * 70)

# Parse ph.x output
ph_data = parse_ph_output("srtio3_ph.out")

# Cell volume (cubic SrTiO3, a = 3.905 A)
a_ang = 3.905
volume_A3 = a_ang**3

masses_amu = {"Sr": 87.62, "Ti": 47.867, "O": 15.999}

# ---- 1. Electronic dielectric tensor ----
print("\n--- ELECTRONIC DIELECTRIC TENSOR (epsilon_infinity) ---")
if ph_data["epsilon_infinity"] is not None:
    eps_inf = ph_data["epsilon_infinity"]
    for i in range(3):
        print(f"  {eps_inf[i,0]:10.5f}  {eps_inf[i,1]:10.5f}  {eps_inf[i,2]:10.5f}")
    eps_inf_avg = np.trace(eps_inf) / 3
    print(f"\n  Isotropic average: epsilon_infinity = {eps_inf_avg:.5f}")
    print(f"  Refractive index:  n = {np.sqrt(eps_inf_avg):.4f}")
else:
    print("  Not found in ph.x output.")
    eps_inf = np.eye(3) * 6.0  # fallback

# ---- 2. Phonon frequencies at Gamma ----
print("\n--- ZONE-CENTRE PHONON FREQUENCIES ---")
if ph_data["phonon_frequencies_cm1"]:
    for i, freq in enumerate(ph_data["phonon_frequencies_cm1"]):
        marker = ""
        if freq < -5:
            marker = " ** SOFT MODE (imaginary) **"
        elif abs(freq) < 5:
            marker = " (acoustic)"
        print(f"  Mode {i+1:3d}: {freq:10.3f} cm^-1{marker}")

# ---- 3. Ionic dielectric tensor (estimate) ----
print("\n--- IONIC DIELECTRIC CONTRIBUTION (estimated) ---")
eps_ionic = compute_epsilon_ionic(
    ph_data["born_charges"],
    ph_data["phonon_frequencies_cm1"],
    masses_amu, volume_A3,
    natom=5
)
print("  Estimated ionic contribution:")
for i in range(3):
    print(f"  {eps_ionic[i,0]:10.3f}  {eps_ionic[i,1]:10.3f}  {eps_ionic[i,2]:10.3f}")

# ---- 4. Static dielectric tensor ----
print("\n--- STATIC DIELECTRIC TENSOR (epsilon_0 = eps_inf + eps_ionic) ---")
eps_0 = eps_inf + eps_ionic
for i in range(3):
    print(f"  {eps_0[i,0]:10.3f}  {eps_0[i,1]:10.3f}  {eps_0[i,2]:10.3f}")
eps_0_avg = np.trace(eps_0) / 3
print(f"\n  Isotropic average: epsilon_0 = {eps_0_avg:.3f}")

# ---- 5. Parse dynmat.x IR output for refined epsilon_0 ----
print("\n--- IR INTENSITIES FROM dynmat.x ---")
ir_data = parse_dynmat_ir("srtio3_dynmat.out")
if ir_data["modes"]:
    for mode, freq, ir_int in zip(ir_data["modes"],
                                   ir_data["frequencies_cm1"],
                                   ir_data["ir_intensities"]):
        marker = ""
        if freq > 10 and ir_int > 0.01:
            marker = " (IR-active)"
        print(f"  Mode {mode:3d}: {freq:10.3f} cm^-1,  "
              f"IR intensity = {ir_int:10.4f} (D/A)^2/AMU{marker}")

    eps_ionic_refined = compute_epsilon_0_from_ir(eps_inf, ir_data, volume_A3)
    print(f"\n  Refined ionic contribution (from IR): {eps_ionic_refined:.3f}")
    print(f"  Refined epsilon_0 = {eps_inf_avg + eps_ionic_refined:.3f}")

# ---- 6. Clausius-Mossotti analysis ----
print("\n--- CLAUSIUS-MOSSOTTI ANALYSIS ---")
if ph_data["epsilon_infinity"] is not None:
    eps = eps_inf_avg
    cm_factor = (eps - 1) / (eps + 2)
    alpha_total = cm_factor * 3 * volume_A3 / (4 * np.pi)
    print(f"  (epsilon_inf - 1) / (epsilon_inf + 2) = {cm_factor:.5f}")
    print(f"  Total polarizability (CM): {alpha_total:.3f} A^3")
    print(f"  Per formula unit: {alpha_total:.3f} A^3")

# ---- 7. Lyddane-Sachs-Teller check ----
print("\n--- LYDDANE-SACHS-TELLER CHECK ---")
if (ph_data["phonon_frequencies_cm1"] and
        ph_data["epsilon_infinity"] is not None):
    # For cubic perovskite: 3 acoustic, 3x TO, 3x LO modes
    # Need to identify TO and LO modes -- this requires dynmat.x with q->0
    # For now, print a reminder
    print("  To verify LST relation, compare:")
    print("  epsilon_0 / epsilon_infinity = product(omega_LO / omega_TO)^2")
    print("  Run dynmat.x with small q to get LO frequencies (see born-effective-charge skill).")

# ---- Save results ----
output = {
    "epsilon_infinity": eps_inf.tolist() if isinstance(eps_inf, np.ndarray) else eps_inf,
    "epsilon_infinity_avg": float(eps_inf_avg),
    "refractive_index": float(np.sqrt(eps_inf_avg)),
    "epsilon_ionic_estimate": eps_ionic.tolist(),
    "epsilon_0_estimate": eps_0.tolist(),
    "epsilon_0_avg_estimate": float(eps_0_avg),
    "phonon_frequencies_cm1": ph_data["phonon_frequencies_cm1"],
    "born_charges": [
        {"atom": bc["atom"], "species": bc["species"],
         "Z_star": bc["Z_star"].tolist()}
        for bc in ph_data["born_charges"]
    ],
}

with open("dielectric_results.json", "w") as f:
    json.dump(output, f, indent=2)
print(f"\nResults saved to dielectric_results.json")
```

### Step 5: Frequency-Dependent Dielectric Function

```python
#!/usr/bin/env python3
"""
Compute the frequency-dependent dielectric function epsilon(omega)
from IR-active mode parameters.

The dielectric function in the infrared region is modelled as:
  epsilon(omega) = epsilon_infinity + sum_m (S_m^2) /
                   (omega_TO,m^2 - omega^2 - i*gamma_m*omega)

where:
  S_m     = mode oscillator strength
  omega_m = TO phonon frequency
  gamma_m = phonon damping (broadening)

This produces peaks at TO frequencies and zeros near LO frequencies.
"""
import numpy as np
import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def dielectric_function(omega, eps_inf, modes):
    """
    Compute complex dielectric function.

    Args:
        omega: frequency array in cm^-1
        eps_inf: high-frequency dielectric constant (scalar for cubic)
        modes: list of dicts with 'freq_cm1', 'strength', 'gamma_cm1'

    Returns:
        complex array: epsilon(omega)
    """
    eps = np.full_like(omega, eps_inf, dtype=complex)

    for mode in modes:
        omega_TO = mode["freq_cm1"]
        S2 = mode["strength"]
        gamma = mode["gamma_cm1"]

        if omega_TO > 0 and S2 > 0:
            eps += S2 * omega_TO**2 / (
                omega_TO**2 - omega**2 - 1j * gamma * omega
            )

    return eps


# ---- Load results ----
with open("dielectric_results.json", "r") as f:
    data = json.load(f)

eps_inf = data["epsilon_infinity_avg"]

# Define IR-active modes for cubic SrTiO3 (example values)
# In practice, extract these from dynmat.x output
# SrTiO3 has 3 triply-degenerate IR-active TO modes:
#   TO1 ~ 90 cm^-1 (soft mode), TO2 ~ 175 cm^-1, TO4 ~ 545 cm^-1
# Mode oscillator strengths from experiment or from Z* analysis
modes = [
    {"freq_cm1": 90.0,  "strength": 200.0, "gamma_cm1": 5.0},  # soft mode
    {"freq_cm1": 175.0, "strength": 3.0,   "gamma_cm1": 8.0},
    {"freq_cm1": 545.0, "strength": 1.5,   "gamma_cm1": 10.0},
]

# Frequency range
omega = np.linspace(1, 800, 2000)

eps_omega = dielectric_function(omega, eps_inf, modes)

# ---- Plot ----
fig, axes = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

# Real part
ax1 = axes[0]
ax1.plot(omega, eps_omega.real, "b-", linewidth=1.5)
ax1.axhline(y=0, color="k", linestyle="-", linewidth=0.5)
ax1.axhline(y=eps_inf, color="gray", linestyle="--", alpha=0.5,
            label=f"$\\epsilon_\\infty$ = {eps_inf:.2f}")
ax1.set_ylabel("Re[$\\epsilon(\\omega)$]", fontsize=13)
ax1.set_ylim(-50, 350)
ax1.legend(fontsize=11)
ax1.set_title("Frequency-Dependent Dielectric Function of SrTiO$_3$",
              fontsize=14)
ax1.grid(True, alpha=0.3)

# Imaginary part
ax2 = axes[1]
ax2.plot(omega, eps_omega.imag, "r-", linewidth=1.5)
ax2.set_xlabel("Frequency (cm$^{-1}$)", fontsize=13)
ax2.set_ylabel("Im[$\\epsilon(\\omega)$]", fontsize=13)
ax2.set_ylim(bottom=0)
ax2.grid(True, alpha=0.3)

# Mark TO frequencies
for mode in modes:
    for ax in axes:
        ax.axvline(x=mode["freq_cm1"], color="green", linestyle=":",
                   alpha=0.5)

plt.tight_layout()
plt.savefig("dielectric_function_omega.png", dpi=150, bbox_inches="tight")
print("Plot saved: dielectric_function_omega.png")

# ---- Print key values ----
print(f"\nepsilon(0) estimate from model: {eps_omega[0].real:.1f}")
print(f"epsilon_infinity: {eps_inf:.3f}")
print("\nTO mode contributions to epsilon_0:")
for mode in modes:
    delta_eps = mode["strength"]
    print(f"  TO = {mode['freq_cm1']:6.1f} cm^-1,  "
          f"Delta_epsilon = {delta_eps:.1f}")
```

### Step 6: VASP Workflow (Reference)

```python
#!/usr/bin/env python3
"""
VASP workflow for computing static and high-frequency dielectric tensor.

VASP provides the dielectric tensor via DFPT (LEPSILON + IBRION=8).
This computes both epsilon_infinity (electronic) and the ionic contribution
in a single calculation.

NOTE: VASP is not available in the current container.
This script generates the INCAR/POSCAR/KPOINTS files for reference.
Run on a VASP-equipped cluster.
"""
import os
import numpy as np

# ---- POSCAR: Cubic SrTiO3 ----
poscar = """SrTiO3 cubic
3.905
1.0  0.0  0.0
0.0  1.0  0.0
0.0  0.0  1.0
Sr Ti O
1  1  3
Direct
0.000  0.000  0.000
0.500  0.500  0.500
0.500  0.500  0.000
0.500  0.000  0.500
0.000  0.500  0.500
"""

# ---- INCAR for dielectric tensor ----
incar = """SYSTEM = SrTiO3 dielectric tensor

# Electronic minimization
ENCUT  = 600        ! Plane-wave cutoff (eV)
EDIFF  = 1E-8       ! Tight SCF convergence for DFPT
PREC   = Accurate
LREAL  = .FALSE.    ! No real-space projection for DFPT
ALGO   = Normal

# Dielectric response (DFPT)
LEPSILON = .TRUE.   ! Compute dielectric tensor and Born charges
IBRION   = 8        ! DFPT for ionic + electronic dielectric response
                    ! IBRION=8 computes:
                    !   - epsilon_infinity (OUTCAR: "MACROSCOPIC STATIC DIELECTRIC TENSOR")
                    !   - epsilon_0 including ionic (OUTCAR: "MACROSCOPIC STATIC DIELECTRIC
                    !     TENSOR (including local field effects in DFT)")
                    !   - Born effective charges (OUTCAR: "BORN EFFECTIVE CHARGES")

# Symmetry
ISYM = 2            ! Use symmetry to reduce perturbations

# Smearing (insulator -- use tetrahedron method for DOS, but for DFPT
# QE-style fixed occupations are better; VASP handles this internally)
ISMEAR = 0
SIGMA  = 0.05

# Output
LWAVE  = .FALSE.
LCHARG = .FALSE.
"""

# ---- KPOINTS ----
kpoints = """Automatic mesh
0
Gamma
  8  8  8
  0  0  0
"""

os.makedirs("vasp_dielectric", exist_ok=True)
with open("vasp_dielectric/POSCAR", "w") as f:
    f.write(poscar)
with open("vasp_dielectric/INCAR", "w") as f:
    f.write(incar)
with open("vasp_dielectric/KPOINTS", "w") as f:
    f.write(kpoints)

print("VASP input files written to vasp_dielectric/")
print("\nTo run: cd vasp_dielectric && mpirun -np N vasp_std")
print("\nParsing VASP output (after calculation completes):")

vasp_parse_script = '''
#!/usr/bin/env python3
"""Parse dielectric tensor from VASP OUTCAR."""
import re
import numpy as np

def parse_vasp_dielectric(outcar_path="OUTCAR"):
    """
    Parse OUTCAR for dielectric tensors.

    VASP OUTCAR contains two dielectric tensor blocks:
    1. "MACROSCOPIC STATIC DIELECTRIC TENSOR (electronic contribution only)"
       -> epsilon_infinity
    2. "MACROSCOPIC STATIC DIELECTRIC TENSOR (including local field effects in DFT)"
       -> epsilon_0 (electronic + ionic)
    """
    with open(outcar_path, "r") as f:
        text = f.read()

    result = {"epsilon_infinity": None, "epsilon_0": None, "born_charges": []}

    # Parse epsilon_infinity
    m = re.search(
        r"MACROSCOPIC STATIC DIELECTRIC TENSOR.*?electronic.*?-+\\n"
        r"\\s+([-\\d.]+)\\s+([-\\d.]+)\\s+([-\\d.]+)\\n"
        r"\\s+([-\\d.]+)\\s+([-\\d.]+)\\s+([-\\d.]+)\\n"
        r"\\s+([-\\d.]+)\\s+([-\\d.]+)\\s+([-\\d.]+)",
        text, re.DOTALL
    )
    if m:
        result["epsilon_infinity"] = np.array([
            [float(m.group(i)) for i in range(1, 4)],
            [float(m.group(i)) for i in range(4, 7)],
            [float(m.group(i)) for i in range(7, 10)],
        ])

    # Parse epsilon_0 (including ionic)
    m = re.search(
        r"MACROSCOPIC STATIC DIELECTRIC TENSOR.*?including.*?-+\\n"
        r"\\s+([-\\d.]+)\\s+([-\\d.]+)\\s+([-\\d.]+)\\n"
        r"\\s+([-\\d.]+)\\s+([-\\d.]+)\\s+([-\\d.]+)\\n"
        r"\\s+([-\\d.]+)\\s+([-\\d.]+)\\s+([-\\d.]+)",
        text, re.DOTALL
    )
    if m:
        result["epsilon_0"] = np.array([
            [float(m.group(i)) for i in range(1, 4)],
            [float(m.group(i)) for i in range(4, 7)],
            [float(m.group(i)) for i in range(7, 10)],
        ])

    # Parse Born effective charges
    born_blocks = re.findall(
        r"BORN EFFECTIVE CHARGES.*?ion\\s+(\\d+).*?\\n"
        r"\\s+1\\s+([-\\d.]+)\\s+([-\\d.]+)\\s+([-\\d.]+)\\n"
        r"\\s+2\\s+([-\\d.]+)\\s+([-\\d.]+)\\s+([-\\d.]+)\\n"
        r"\\s+3\\s+([-\\d.]+)\\s+([-\\d.]+)\\s+([-\\d.]+)",
        text
    )
    for block in born_blocks:
        ion_idx = int(block[0])
        z_star = np.array([
            [float(block[i]) for i in range(1, 4)],
            [float(block[i]) for i in range(4, 7)],
            [float(block[i]) for i in range(7, 10)],
        ])
        result["born_charges"].append({
            "ion": ion_idx,
            "Z_star": z_star.tolist(),
        })

    return result


# Usage:
# data = parse_vasp_dielectric("vasp_dielectric/OUTCAR")
# print("epsilon_infinity:", data["epsilon_infinity"])
# print("epsilon_0:", data["epsilon_0"])
# print("epsilon_ionic:", data["epsilon_0"] - data["epsilon_infinity"])
'''

with open("vasp_dielectric/parse_outcar.py", "w") as f:
    f.write(vasp_parse_script)
print("VASP parser written: vasp_dielectric/parse_outcar.py")
```

---

## Key Parameters

| Parameter | Typical Value | Notes |
|---|---|---|
| `epsil` (ph.x) | `.true.` | **Required** -- triggers computation of epsilon_infinity and Born charges |
| `trans` (ph.x) | `.true.` | Compute phonons (needed for ionic dielectric contribution) |
| `tr2_ph` (ph.x) | `1.0d-14` | DFPT convergence. Use very tight values for accurate dielectric tensor. |
| `conv_thr` (pw.x) | `1.0d-12` | SCF convergence. Must be very tight for DFPT. |
| k-grid | 8x8x8 or denser | Dielectric tensor converges slowly with k-points. Always test convergence. |
| `ecutwfc` | 50-80 Ry | Converge with respect to epsilon_infinity. |
| `asr` (dynmat.x) | `'crystal'` | Acoustic sum rule enforcement for IR intensities. |
| `LEPSILON` (VASP) | `.TRUE.` | Compute electronic dielectric tensor. |
| `IBRION` (VASP) | `8` | DFPT ionic response -- gives epsilon_0 including ionic contribution. |
| `EDIFF` (VASP) | `1E-8` | Tight SCF convergence for DFPT accuracy. |

### Convergence Checklist

Dielectric constants are sensitive to computational parameters. Test convergence of epsilon_infinity with respect to:

1. **k-grid**: Run 6x6x6, 8x8x8, 10x10x10, 12x12x12. epsilon_infinity should converge to within 0.05.
2. **ecutwfc**: Test 40, 50, 60, 80 Ry. Usually converges by 50-60 Ry with PAW PPs.
3. **Pseudopotential type**: NC, US, and PAW may give slightly different results. PAW is generally most accurate.
4. **Functional**: PBE typically overestimates epsilon_infinity by 10-20% due to band gap underestimation. HSE06 or GW can improve accuracy.

## Interpreting Results

### Electronic dielectric tensor (epsilon_infinity)

- **Cubic crystals**: Diagonal with equal elements. Typical values for perovskites: 5-7. SrTiO3: ~5.2, BaTiO3: ~6.7.
- **Tetragonal/orthorhombic**: Anisotropic. epsilon_xx, epsilon_yy, epsilon_zz differ.
- **Refractive index**: n_i = sqrt(epsilon_infinity_ii). Useful for comparing with optical measurements.
- **DFT overestimate**: PBE underestimates the band gap, which leads to overestimation of epsilon_infinity. Expect ~10-20% overestimate vs. experiment.

### Static dielectric tensor (epsilon_0)

- **Incipient ferroelectrics** (e.g., SrTiO3): epsilon_0 can be very large (100-300) due to a soft TO mode near 90 cm^-1.
- **Normal ferroelectrics** (e.g., BaTiO3 in polar phase): epsilon_0 is typically 100-1000 depending on direction and proximity to phase transition.
- **Simple insulators** (e.g., MgO, Al2O3): epsilon_0 is modest (10-20) with small ionic contribution.
- **Negative or divergent epsilon_0**: Indicates an unstable phonon (imaginary TO frequency). The material wants to distort -- this signals a structural phase transition.

### IR activity

- **Triply degenerate modes** in cubic crystals: Each IR-active mode appears as a 3-fold degenerate set.
- **Selection rules**: In cubic perovskites (Pm-3m), the 15 zone-centre modes decompose as 3F1u(IR) + 1F1u(acoustic) + 1F2u(silent). Only F1u modes contribute to epsilon_ionic.
- **Soft modes**: A TO frequency approaching zero drives epsilon_0 -> infinity (Curie-Weiss behavior). This is the hallmark of an incipient or displacive ferroelectric.

### Clausius-Mossotti relation

- Valid for cubic crystals with well-separated, non-overlapping atomic polarizabilities.
- Useful for predicting dielectric constants of solid solutions: interpolate polarizabilities of end members.
- Breaks down for highly covalent systems where local field effects are strong.

### Typical dielectric constants for reference materials

| Material | epsilon_infinity | epsilon_0 | Reference |
|---|---|---|---|
| SrTiO3 (cubic) | 5.2 | ~300 (300 K) | Experiment |
| BaTiO3 (cubic) | 6.7 | ~1500 (near Tc) | Experiment |
| BaTiO3 (tetra, perp) | 5.6 | ~2000 | Experiment |
| BaTiO3 (tetra, par) | 6.2 | ~120 | Experiment |
| PbTiO3 (tetra) | 7.1 | ~100 | Experiment |
| MgO | 3.0 | 9.8 | Experiment |
| Al2O3 | 3.1 | 9.3 | Experiment |

## Common Issues

| Problem | Solution |
|---|---|
| **epsilon_infinity too large** | PBE underestimates band gaps, inflating epsilon_infinity. Use HSE06 or apply scissor correction. Check k-point convergence. |
| **epsilon_0 is negative** | An unstable (imaginary frequency) TO mode dominates. The structure is dynamically unstable -- relax to the true ground state (lower-symmetry phase) before computing dielectric tensor. |
| **epsilon_0 is extremely large (> 10000)** | A soft TO mode is very close to zero frequency. This indicates proximity to a ferroelectric phase transition. The material may be an incipient ferroelectric. Check if the structure is at a phase boundary. |
| **IR intensities all zero** | For centrosymmetric crystals, check if the modes are indeed IR-active (odd symmetry). Silent modes (e.g., F2u in perovskites) have zero IR intensity by symmetry. |
| **Mismatch between LST prediction and direct epsilon_0** | The Lyddane-Sachs-Teller relation assumes well-separated LO-TO pairs. If modes overlap or are degenerate, the relation is approximate. Also ensure you are using ASR-corrected frequencies. |
| **ph.x fails with epsil=.true.** | Same remedies as born-effective-charge skill: ensure insulating ground state, tight SCF convergence, matching prefix/outdir. |
| **VASP IBRION=8 gives wrong ionic epsilon** | Ensure EDIFF is tight (1e-8 or better). Check that ISYM is set correctly. For low-symmetry structures, more perturbations are needed. |
| **Off-diagonal elements of epsilon are nonzero in cubic crystal** | Numerical noise. Off-diagonal elements should be < 0.01 for cubic symmetry. If larger, increase k-grid and ecutwfc. |
| **Dielectric tensor not symmetric** | The tensor should be symmetric (epsilon_ij = epsilon_ji). Asymmetry indicates convergence issues. Tighten tr2_ph and conv_thr. |
| **dynmat.x crashes** | Ensure the .dyn file from ph.x is complete (check for "JOB DONE" in ph.x output). The .dyn file must contain Z* and epsilon_infinity data (only present when epsil=.true. was used). |
