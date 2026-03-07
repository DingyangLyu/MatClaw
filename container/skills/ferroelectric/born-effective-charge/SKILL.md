# Born Effective Charges and Dielectric Tensor

## When to Use

- You need the Born effective charge tensors Z* for each atom in a crystal (the dynamical charge that couples atomic displacement to macroscopic electric field).
- You need the high-frequency (electronic) dielectric tensor (epsilon_infinity).
- You want to identify anomalous Born charges as an indicator of ferroelectric instability.
- You need LO-TO splitting correction for phonon dispersion near Gamma.
- You want to compute the piezoelectric response or mode effective charges.
- You need input quantities for computing the spontaneous polarization via the linearized approximation: P_s ~ (1/Omega) * sum_i Z*_i . delta_r_i.

## Prerequisites

- A fully relaxed crystal structure (forces < 1e-4 Ry/Bohr).
- Quantum ESPRESSO 7.5 with `pw.x` and `ph.x` available on PATH.
- Pseudopotential files (norm-conserving recommended for DFPT; PAW/ultrasoft also work in QE 7.x).
- Python packages: `pymatgen`, `ase`, `numpy`.
- The material must be an insulator (nonzero band gap). Born charges are undefined for metals.

## Background

The Born effective charge tensor Z*_kij relates the macroscopic polarization change in direction i to a displacement of atom k in direction j:

```
Z*_k,ij = Omega * (dP_i / du_k,j)
```

where Omega is the unit cell volume and u_k,j is the displacement of atom k in direction j. In a purely ionic crystal, Z* equals the formal ionic charge. **Anomalous Born charges** -- significantly larger than formal charges -- indicate strong hybridization and are a hallmark of ferroelectric instability. For example, in BaTiO3:

| Atom | Formal charge | Z*_33 (DFT) |
|---|---|---|
| Ba | +2 | +2.75 |
| Ti | +4 | +7.25 |
| O_parallel | -2 | -5.71 |
| O_perpendicular | -2 | -2.15 |

The large Z*(Ti) ~ +7.25 (vs. formal +4) is a signature of the Ti-O hybridization driving the ferroelectric instability.

The high-frequency dielectric tensor epsilon_infinity is the electronic contribution to the dielectric response (excluding ionic/lattice contributions). It is computed from the same DFPT calculation.

## Detailed Steps

### Overview

```
Step 0: Download pseudopotentials
Step 1: Run SCF calculation (pw.x)
Step 2: Run DFPT at Gamma (ph.x with epsil=.true.)
Step 3: Parse Born effective charges and dielectric tensor
Step 4: Analyze results
```

### Step 0: Download Pseudopotentials

```python
#!/usr/bin/env python3
"""Download pseudopotentials for BaTiO3 Born effective charge calculation."""
import os
import urllib.request

PSEUDO_DIR = os.path.abspath("./pseudo")
os.makedirs(PSEUDO_DIR, exist_ok=True)

# Norm-conserving PPs recommended for DFPT (ph.x).
# PAW/ultrasoft also work in QE >= 6.x but NC are more robust.
# Here we use PAW PPs from PSlibrary which work well with ph.x in QE 7.5.
pseudos = {
    "Ba": "Ba.pbe-spn-kjpaw_psl.1.0.0.UPF",
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
Run SCF for BaTiO3 (cubic or tetragonal phase) as prerequisite for ph.x.
The SCF must converge tightly (conv_thr ~ 1e-12) for accurate DFPT.
"""
import os
import subprocess

PSEUDO_DIR = os.path.abspath("./pseudo")
OUTDIR = os.path.abspath("./tmp_born")
os.makedirs(OUTDIR, exist_ok=True)

# Cubic BaTiO3 (centrosymmetric reference)
# For the polar phase, change coordinates/cell accordingly
a_ang = 4.00
a_bohr = a_ang * 1.8897259886

scf_input = f"""&CONTROL
    calculation = 'scf'
    prefix      = 'batio3'
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
  Ba  137.327  Ba.pbe-spn-kjpaw_psl.1.0.0.UPF
  Ti   47.867  Ti.pbe-spn-kjpaw_psl.1.0.0.UPF
  O    15.999  O.pbe-n-kjpaw_psl.1.0.0.UPF

ATOMIC_POSITIONS (crystal)
  Ba  0.000  0.000  0.000
  Ti  0.500  0.500  0.500
  O   0.500  0.500  0.000
  O   0.500  0.000  0.500
  O   0.000  0.500  0.500

K_POINTS (automatic)
  8 8 8  1 1 1
"""

with open("batio3_scf.in", "w") as f:
    f.write(scf_input)
print("Written: batio3_scf.in")

print("Running pw.x SCF ...")
result = subprocess.run(
    ["mpirun", "-np", "4", "pw.x", "-in", "batio3_scf.in"],
    capture_output=True, text=True, timeout=1800
)
with open("batio3_scf.out", "w") as f:
    f.write(result.stdout)

if result.returncode != 0:
    print("ERROR: SCF failed!")
    print(result.stderr[-500:] if result.stderr else "No stderr")
else:
    if "convergence has been achieved" in result.stdout:
        print("SCF converged successfully.")
    else:
        print("WARNING: SCF may not have converged.")
    print("Output: batio3_scf.out")
```

**Important**: Use `conv_thr = 1.0d-12` (very tight) for the SCF that feeds into ph.x. The DFPT linear response requires a well-converged ground state. Also use a dense k-grid (8x8x8 or denser) since Born charges can be sensitive to k-point sampling.

### Step 2: DFPT Calculation at Gamma (ph.x)

```python
#!/usr/bin/env python3
"""
Run ph.x at Gamma to compute Born effective charges and dielectric tensor.

Key settings:
  - epsil = .true.  : compute dielectric tensor and Born charges
  - q = 0 0 0       : Gamma point only
  - trans = .true.   : compute phonons (needed for Born charges)
"""
import os
import subprocess

OUTDIR = os.path.abspath("./tmp_born")

ph_input = f"""Born effective charges and dielectric tensor
&INPUTPH
    prefix    = 'batio3'
    outdir    = '{OUTDIR}'
    fildyn    = 'batio3.dyn'
    tr2_ph    = 1.0d-14
    epsil     = .true.
    trans     = .true.
    ldisp     = .false.
    recover   = .false.
    verbosity = 1
/
0.0 0.0 0.0
"""
# The last line "0.0 0.0 0.0" specifies the q-point (Gamma).
# When ldisp=.false. and epsil=.true., ph.x computes at Gamma only.

with open("batio3_ph.in", "w") as f:
    f.write(ph_input)
print("Written: batio3_ph.in")

print("Running ph.x (this may take 10-60 minutes) ...")
result = subprocess.run(
    ["mpirun", "-np", "4", "ph.x", "-in", "batio3_ph.in"],
    capture_output=True, text=True, timeout=7200  # 2 hour timeout
)
with open("batio3_ph.out", "w") as f:
    f.write(result.stdout)

if result.returncode != 0:
    print("ERROR: ph.x failed!")
    print(result.stderr[-500:] if result.stderr else "No stderr")
else:
    if "JOB DONE" in result.stdout:
        print("ph.x completed successfully.")
    else:
        print("WARNING: ph.x may not have completed.")
    print("Output: batio3_ph.out")
    print("Dynamical matrix: batio3.dyn")
```

**Important ph.x parameters:**

| Parameter | Value | Purpose |
|---|---|---|
| `epsil` | `.true.` | Compute dielectric tensor and Born effective charges |
| `trans` | `.true.` | Compute phonon modes (required for Born charges via DFPT) |
| `tr2_ph` | `1.0d-14` | Convergence threshold for DFPT self-consistency. Use very tight values. |
| `ldisp` | `.false.` | Single q-point mode (Gamma only for Born charges) |
| `fildyn` | filename | Output file for dynamical matrix |

### Step 3: Parse Born Effective Charges and Dielectric Tensor

```python
#!/usr/bin/env python3
"""
Parse Born effective charges Z* and dielectric tensor epsilon_inf
from ph.x output.

QE ph.x output format:
  - Dielectric tensor block starts with "Dielectric Tensor:"
  - Born charges block starts with "Effective Charges E-U"
  - Each atom has a 3x3 tensor printed as three rows
"""
import re
import os
import json
import numpy as np


def parse_ph_output(filename):
    """
    Parse ph.x output for Born effective charges and dielectric tensor.

    Returns:
        dict with keys:
            'dielectric_tensor': 3x3 list (epsilon_infinity)
            'born_charges': list of dicts, each with 'atom', 'species', 'Z_star' (3x3)
            'phonon_frequencies_cm1': list of phonon frequencies at Gamma in cm^-1
            'phonon_modes': list of mode symmetries
    """
    result = {
        "dielectric_tensor": None,
        "born_charges": [],
        "phonon_frequencies_cm1": [],
        "phonon_modes": [],
    }

    if not os.path.exists(filename):
        print(f"File not found: {filename}")
        return result

    with open(filename, "r") as f:
        lines = f.readlines()

    # ── Parse dielectric tensor ───────────────────────────────────
    for i, line in enumerate(lines):
        if "Dielectric Tensor:" in line or "Dielectric constant in cartesian axis" in line:
            # Next 3 non-blank lines contain the 3x3 tensor
            tensor = []
            row_count = 0
            j = i + 1
            while row_count < 3 and j < len(lines):
                stripped = lines[j].strip()
                if stripped and not stripped.startswith("("):
                    # Parse numbers from the line
                    # QE format: "  1.23456   0.00000   0.00000"
                    # or with parenthetical labels
                    nums = re.findall(r"[-+]?\d+\.\d+", stripped)
                    if len(nums) >= 3:
                        tensor.append([float(nums[0]), float(nums[1]), float(nums[2])])
                        row_count += 1
                j += 1
            if len(tensor) == 3:
                result["dielectric_tensor"] = tensor

    # ── Parse Born effective charges ──────────────────────────────
    for i, line in enumerate(lines):
        if ("Effective Charges E-U" in line or
                "Born effective charges" in line or
                "effective charges (d Force / dE)" in line):
            # Parse each atom's Z* tensor
            j = i + 1
            while j < len(lines):
                # Look for atom header: "atom  N  species  XX"
                atom_match = re.match(
                    r"\s*atom\s+(\d+)\s+(\w+)", lines[j]
                )
                if atom_match:
                    atom_idx = int(atom_match.group(1))
                    species = atom_match.group(2)

                    # Read the 3x3 Z* tensor from next 3 lines
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
                            "Z_star": z_star,
                        })
                    j = k
                elif (lines[j].strip() == "" or
                      "---" in lines[j] or
                      any(kw in lines[j] for kw in
                          ["Dielectric", "phonon", "PHONON", "freq", "Mode"])):
                    # End of Born charges section
                    break
                else:
                    j += 1

    # ── Parse phonon frequencies at Gamma ─────────────────────────
    for i, line in enumerate(lines):
        freq_match = re.search(
            r"freq\s*\(\s*(\d+)\)\s*=\s*([-\d.]+)\s*\[cm-1\]", line
        )
        if freq_match:
            result["phonon_frequencies_cm1"].append(float(freq_match.group(2)))

        # Alternative format: "omega( N) = X.XXXX [cm-1]"
        freq_match2 = re.search(
            r"omega\s*\(\s*(\d+)\)\s*=\s*([-\d.]+)\s*\[cm-1\]", line
        )
        if freq_match2:
            result["phonon_frequencies_cm1"].append(float(freq_match2.group(2)))

    return result


# ── Parse the ph.x output ─────────────────────────────────────────
results = parse_ph_output("batio3_ph.out")

# ── Print dielectric tensor ───────────────────────────────────────
print("=" * 60)
print("HIGH-FREQUENCY DIELECTRIC TENSOR (epsilon_infinity)")
print("=" * 60)
if results["dielectric_tensor"] is not None:
    eps = np.array(results["dielectric_tensor"])
    for i in range(3):
        print(f"  {eps[i,0]:10.5f}  {eps[i,1]:10.5f}  {eps[i,2]:10.5f}")
    print(f"\n  Trace/3 = {np.trace(eps)/3:.5f}")
    print(f"  (Isotropic average of epsilon_infinity)")
else:
    print("  Not found in output.")

# ── Print Born effective charges ──────────────────────────────────
print("\n" + "=" * 60)
print("BORN EFFECTIVE CHARGE TENSORS Z*")
print("=" * 60)

formal_charges = {"Ba": 2.0, "Ti": 4.0, "O": -2.0}

for bc in results["born_charges"]:
    Z = np.array(bc["Z_star"])
    sp = bc["species"]
    formal = formal_charges.get(sp, "?")
    print(f"\n  Atom {bc['atom']:2d} ({sp}), formal charge = {formal}")
    print(f"  Z* tensor:")
    for i in range(3):
        print(f"    {Z[i,0]:10.5f}  {Z[i,1]:10.5f}  {Z[i,2]:10.5f}")
    print(f"  Diagonal: Z*_xx={Z[0,0]:.4f}, Z*_yy={Z[1,1]:.4f}, Z*_zz={Z[2,2]:.4f}")
    print(f"  Isotropic Z* = {np.trace(Z)/3:.4f} (formal = {formal})")

    if isinstance(formal, (int, float)):
        anomaly = np.trace(Z) / 3 - formal
        if abs(anomaly) > 1.0:
            print(f"  ** ANOMALOUS: Z* deviates from formal charge by {anomaly:+.2f} **")

# ── Check acoustic sum rule ───────────────────────────────────────
print("\n" + "=" * 60)
print("ACOUSTIC SUM RULE CHECK")
print("=" * 60)
if results["born_charges"]:
    Z_sum = np.zeros((3, 3))
    for bc in results["born_charges"]:
        Z_sum += np.array(bc["Z_star"])
    print("  Sum of all Z* tensors (should be zero for charge neutrality):")
    for i in range(3):
        print(f"    {Z_sum[i,0]:10.5f}  {Z_sum[i,1]:10.5f}  {Z_sum[i,2]:10.5f}")
    max_violation = np.max(np.abs(Z_sum))
    print(f"  Max violation: {max_violation:.6f}")
    if max_violation < 0.1:
        print("  OK: Acoustic sum rule approximately satisfied.")
    else:
        print("  WARNING: Significant violation. Check convergence.")

# ── Print phonon frequencies ──────────────────────────────────────
if results["phonon_frequencies_cm1"]:
    print("\n" + "=" * 60)
    print("PHONON FREQUENCIES AT GAMMA (cm^-1)")
    print("=" * 60)
    for i, freq in enumerate(results["phonon_frequencies_cm1"]):
        marker = ""
        if freq < -5:
            marker = " ** UNSTABLE (imaginary) **"
        elif freq < 5:
            marker = " (acoustic)"
        print(f"  Mode {i+1:3d}: {freq:10.3f} cm^-1{marker}")

# ── Save results ──────────────────────────────────────────────────
# Convert numpy arrays to lists for JSON serialization
output = {
    "dielectric_tensor": results["dielectric_tensor"],
    "born_charges": results["born_charges"],
    "phonon_frequencies_cm1": results["phonon_frequencies_cm1"],
}
with open("born_charges_results.json", "w") as f:
    json.dump(output, f, indent=2)
print(f"\nResults saved to born_charges_results.json")
```

### Step 4: LO-TO Splitting Analysis

```python
#!/usr/bin/env python3
"""
Analyze LO-TO splitting from Born effective charges and dielectric tensor.

The non-analytic correction to phonon frequencies at Gamma:
  Delta_omega^2 = (4*pi*e^2 / Omega) * |sum_k Z*_k . e_k|^2 / (q_hat . eps_inf . q_hat)

where e_k are the phonon eigenvectors and q_hat is the phonon wavevector direction.
This splits LO and TO modes at Gamma.

QE's dynmat.x code computes this automatically when given Z* and eps_inf.
"""
import os
import subprocess
import json
import numpy as np

# ── Run dynmat.x to apply LO-TO splitting ─────────────────────────
dynmat_input = """&INPUT
    fildyn = 'batio3.dyn'
    asr    = 'crystal'
    q(1)   = 0.0
    q(2)   = 0.0
    q(3)   = 0.0001
/
"""
# q is set to a small value along z to probe the LO-TO direction.
# asr = 'crystal' applies the acoustic sum rule.

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

    # Parse LO-TO corrected frequencies
    import re
    freqs_corrected = []
    for line in result.stdout.split("\n"):
        m = re.search(r"freq\s*\(\s*\d+\)\s*=\s*([-\d.]+)\s*\[cm-1\]", line)
        if m:
            freqs_corrected.append(float(m.group(1)))

    if freqs_corrected:
        print("\nLO-TO corrected frequencies (q -> 0 along z):")
        for i, freq in enumerate(freqs_corrected):
            print(f"  Mode {i+1:3d}: {freq:10.3f} cm^-1")
```

### Step 5: Estimate Polarization from Born Charges (Linear Approximation)

```python
#!/usr/bin/env python3
"""
Estimate spontaneous polarization from Born effective charges.

Linear approximation (valid for small distortions):
  P_i = (e / Omega) * sum_k Z*_k,ij * delta_u_k,j

where delta_u_k is the displacement of atom k from the centrosymmetric
reference position. This is faster than the full Berry phase calculation
but only approximate.
"""
import json
import numpy as np
from pymatgen.core import Structure, Lattice

# ── Load Born charges ─────────────────────────────────────────────
with open("born_charges_results.json", "r") as f:
    bc_data = json.load(f)

# ── Define structures ─────────────────────────────────────────────
# Centrosymmetric reference (cubic BaTiO3)
a_cubic = 4.00
ref = Structure(
    Lattice.cubic(a_cubic),
    species=["Ba", "Ti", "O", "O", "O"],
    coords=[
        [0.000, 0.000, 0.000],
        [0.500, 0.500, 0.500],
        [0.500, 0.500, 0.000],
        [0.500, 0.000, 0.500],
        [0.000, 0.500, 0.500],
    ],
    coords_are_cartesian=False,
)

# Polar phase (tetragonal BaTiO3)
polar = Structure(
    Lattice.tetragonal(3.994, 4.034),
    species=["Ba", "Ti", "O", "O", "O"],
    coords=[
        [0.000, 0.000, 0.000],
        [0.500, 0.500, 0.520],
        [0.500, 0.500, -0.020],
        [0.500, 0.000, 0.520],
        [0.000, 0.500, 0.520],
    ],
    coords_are_cartesian=False,
)

# ── Compute displacements in Cartesian coordinates ────────────────
# Use the polar lattice for converting fractional to Cartesian
E_CHARGE = 1.602176634e-19  # C
ANGSTROM_TO_M = 1e-10

# Fractional displacements
delta_frac = polar.frac_coords - ref.frac_coords
# Wrap to [-0.5, 0.5]
delta_frac = delta_frac - np.round(delta_frac)
# Convert to Cartesian (Angstrom)
delta_cart = delta_frac @ polar.lattice.matrix  # natom x 3, in Angstrom

print("Atomic displacements from centrosymmetric reference:")
for i, site in enumerate(polar):
    print(f"  {str(site.specie):3s}: "
          f"dx={delta_cart[i,0]:8.4f}, "
          f"dy={delta_cart[i,1]:8.4f}, "
          f"dz={delta_cart[i,2]:8.4f} Angstrom")

# ── Compute polarization ──────────────────────────────────────────
Omega_m3 = polar.volume * ANGSTROM_TO_M**3

P = np.zeros(3)  # C/m^2
for i, bc in enumerate(bc_data["born_charges"]):
    Z = np.array(bc["Z_star"])  # 3x3
    du = delta_cart[i] * ANGSTROM_TO_M  # displacement in meters
    P += E_CHARGE * Z @ du / Omega_m3

print(f"\nEstimated polarization (linear Born charge approximation):")
print(f"  P_x = {P[0]:.6f} C/m^2")
print(f"  P_y = {P[1]:.6f} C/m^2")
print(f"  P_z = {P[2]:.6f} C/m^2")
print(f"  |P| = {np.linalg.norm(P):.6f} C/m^2")
print(f"  |P| = {np.linalg.norm(P)*100:.4f} uC/cm^2")
print(f"\nNote: This linear estimate is approximate. For accurate values,")
print(f"use the full Berry phase calculation (see polarization/ sub-skill).")
```

---

## Key Parameters

| Parameter | Typical Value | Notes |
|---|---|---|
| `epsil` | `.true.` | **Required** in ph.x to compute dielectric tensor and Born charges |
| `trans` | `.true.` | Compute phonon perturbations (required for Born charges) |
| `tr2_ph` | `1.0d-14` | DFPT convergence threshold. Very tight values needed for accurate Z*. Use at least 1e-14. |
| `conv_thr` (scf) | `1.0d-12` | SCF convergence for pw.x. Must be very tight for DFPT. |
| k-grid | 8x8x8 or denser | Born charges are sensitive to k-point sampling. Test convergence. |
| `ecutwfc` | 50--80 Ry | Same as for the SCF step. Converge with respect to Z*. |
| `asr` (dynmat.x) | `'crystal'` | Acoustic sum rule enforcement in post-processing |

### Convergence Testing for Born Charges

Born charges are more sensitive to computational parameters than total energies. Test convergence with respect to:

1. **k-grid**: Run with 6x6x6, 8x8x8, 10x10x10, 12x12x12 and check Z* convergence.
2. **ecutwfc**: Test with 40, 50, 60, 80 Ry.
3. **tr2_ph**: Usually 1e-14 is sufficient. Try 1e-16 if Z* is noisy.

Z* is converged when the diagonal elements change by less than 0.05 between successive parameter values.

## Interpreting Results

### Dielectric Tensor

- **Cubic crystals**: epsilon_infinity is diagonal with equal elements. Typical values: 5--7 for perovskites.
- **Tetragonal**: epsilon_xx = epsilon_yy != epsilon_zz (uniaxial).
- **Comparison**: Experimental optical dielectric constants from refractive index: epsilon = n^2. BaTiO3 cubic: epsilon_infinity ~ 6.7.

### Born Effective Charges

- **Formal vs. effective**: If Z* significantly exceeds the formal ionic charge, the bonding has strong covalent/hybridization character.
- **Anomalous charges**: Z*(Ti) ~ +7 in BaTiO3 (formal +4) is a classic indicator of ferroelectric instability. The anomalous contribution arises from Ti 3d - O 2p hybridization.
- **Acoustic sum rule**: sum of all Z* tensors should be zero (charge neutrality). Violations > 0.1 indicate convergence issues.
- **Anisotropy**: In tetragonal BaTiO3, Z*(O) differs for oxygen atoms parallel vs. perpendicular to the polar axis. This anisotropy reflects the directional nature of the polar instability.

### Connection to Piezoelectric Response

The piezoelectric stress tensor e_ij can be decomposed into:

```
e_ij = e^clamped_ij + sum_k Z*_k,ij * du_k / d_strain_j
```

where the first term is the electronic (clamped-ion) contribution and the second is the ionic (internal-strain) contribution mediated by Born charges.

### Typical Z* Values for Common Ferroelectrics

| Material | Atom | Formal | Z*_xx | Z*_zz | Reference |
|---|---|---|---|---|---|
| BaTiO3 | Ba | +2 | +2.75 | +2.75 | Ghosez et al. 1998 |
| BaTiO3 | Ti | +4 | +7.25 | +7.25 | Ghosez et al. 1998 |
| BaTiO3 | O_perp | -2 | -2.15 | -5.71 | Ghosez et al. 1998 |
| PbTiO3 | Pb | +2 | +3.90 | +3.90 | Zhong et al. 1994 |
| PbTiO3 | Ti | +4 | +7.06 | +7.06 | Zhong et al. 1994 |

## Common Issues

| Problem | Solution |
|---|---|
| **ph.x crashes immediately** | Check that the SCF prefix and outdir match between pw.x and ph.x. The ph.x reads wavefunctions from the SCF output directory. |
| **ph.x does not converge** | Reduce `tr2_ph` gradually. If alpha_mix is oscillating, try `alpha_mix(1) = 0.3` in the ph.x input. Also ensure the SCF was tightly converged (conv_thr < 1e-12). |
| **Acoustic sum rule violation > 0.5** | Increase k-grid density and ecutwfc. Check that the structure is fully relaxed (forces < 1e-4 Ry/Bohr). |
| **Imaginary phonon frequencies** | Structure is dynamically unstable at this configuration. For cubic perovskites above T_c, this is expected (soft modes). For the polar phase, ensure the structure is fully relaxed. |
| **Z* matches formal charge (no anomaly)** | The material may not be ferroelectric, or the bonding is predominantly ionic. Check if this is physically expected. |
| **epsil=.true. fails with metals** | Born charges and epsilon_infinity are only defined for insulators. If the band gap closes with your PP/functional, try a Hubbard U correction or hybrid functional. |
| **Very slow ph.x calculation** | DFPT at Gamma for 5-atom cells typically takes 10-60 minutes on 4 cores. For larger cells, increase parallelism with `-nk` or `-npool`. Reduce ecutwfc if possible. |
| **Negative dielectric constant** | Unphysical. Usually indicates convergence failure. Tighten conv_thr and tr2_ph, increase k-grid. |
| **PAW vs NC pseudopotentials** | QE 7.x supports DFPT with PAW and ultrasoft PPs. However, if you encounter issues, try norm-conserving PPs (e.g., from PseudoDojo or SG15). |
| **LO-TO splitting not applied** | The raw ph.x output gives TO frequencies. Run dynmat.x with Born charges and epsilon_infinity to get LO-TO corrected frequencies. |
