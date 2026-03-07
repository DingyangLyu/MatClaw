# Z2 Topological Invariant Calculation

## When to Use

- Determining whether a material is a topological insulator (TI) protected by time-reversal symmetry
- Classifying materials as strong TI, weak TI, or trivial insulator
- Screening candidate TI materials (e.g., Bi2Se3, Bi2Te3, Sb2Te3, HgTe)
- Any system with spin-orbit coupling and time-reversal symmetry where band inversion is suspected

## Prerequisites

- Quantum ESPRESSO 7.5 (`pw.x`, `pw2wannier90.x`)
- Wannier90 (bundled or standalone)
- Python packages: `z2pack` (`pip install z2pack`), `numpy`, `matplotlib`
- Fully relativistic pseudopotentials with SOC support (FR-ONCV or PSlibrary `_rel`)
- Crystal structure of the candidate material

## Theory

The Z2 invariant classifies time-reversal invariant (TRI) topological insulators. In 3D,
four Z2 indices (nu0; nu1 nu2 nu3) exist:
- nu0 = 1: strong topological insulator (robust surface states on all surfaces)
- nu0 = 0, any nu_i = 1: weak topological insulator
- All zero: trivial insulator

**Methods to compute Z2:**

1. **Wilson loop / hybrid Wannier charge centers (HWCC):** Track the evolution of Wannier
   centers across the BZ. If the HWCC winds an odd number of times, Z2 = 1. This is
   gauge-invariant and works for any crystal. Implemented in `z2pack`.

2. **Parity analysis (Fu-Kane formula):** For inversion-symmetric crystals, Z2 is determined
   by the parity eigenvalues of occupied bands at TRIM points:
   (-1)^nu0 = product over TRIM_i of delta_i, where delta_i = product_m xi_{2m}(TRIM_i)
   and xi_{2m} is the parity eigenvalue of the 2m-th occupied band (Kramers pairs counted once).

## Detailed Steps

### Method 1: QE + Wannier90 + z2pack (General Method)

#### Step 1: Prepare Crystal Structure and QE Input (Bi2Se3 Example)

```python
#!/usr/bin/env python3
"""
Complete Z2 invariant workflow for Bi2Se3 using QE + z2pack.
Bi2Se3 is a known strong topological insulator with Z2 = (1;000).
"""
import os
import subprocess
import numpy as np

WORK_DIR = os.path.abspath("z2_bi2se3")
os.makedirs(WORK_DIR, exist_ok=True)

# ------------------------------------------------------------------
# Bi2Se3 rhombohedral structure (R-3m, #166)
# Experimental lattice parameters: a = 4.138 A, c = 28.64 A (hexagonal)
# We use the rhombohedral cell for efficiency.
# ------------------------------------------------------------------

# Rhombohedral lattice parameters
a_rhomb = 9.841  # Bohr (5.208 Angstrom)
alpha = 24.304   # degrees - rhombohedral angle

# Convert to Cartesian (QE ibrav=5 convention)
# For ibrav=5: celldm(1) = a in Bohr, celldm(4) = cos(alpha)
cos_alpha = np.cos(np.radians(alpha))

# Atomic positions in crystal coordinates for rhombohedral cell
# Bi at (mu, mu, mu) and (-mu, -mu, -mu), mu ~ 0.4
# Se at (nu, nu, nu), (-nu, -nu, -nu), and (0,0,0), nu ~ 0.206
bi_mu = 0.4006
se_nu = 0.2060

# --- Pseudopotential names ---
# Using fully relativistic PSlibrary pseudopotentials for SOC
PP_BI = "Bi.rel-pbe-dn-kjpaw_psl.1.0.0.UPF"
PP_SE = "Se.rel-pbe-dn-kjpaw_psl.1.0.0.UPF"
PSEUDO_DIR = os.environ.get("PSEUDO_DIR", "./pseudo")

# ------------------------------------------------------------------
# SCF calculation with SOC
# ------------------------------------------------------------------
scf_input = f"""&CONTROL
  calculation = 'scf'
  prefix = 'bi2se3'
  outdir = './tmp'
  pseudo_dir = '{PSEUDO_DIR}'
  verbosity = 'high'
  tprnfor = .true.
/
&SYSTEM
  ibrav = 5
  celldm(1) = {a_rhomb}
  celldm(4) = {cos_alpha:.6f}
  nat = 5
  ntyp = 2
  ecutwfc = 50.0
  ecutrho = 400.0
  occupations = 'smearing'
  smearing = 'cold'
  degauss = 0.005
  noncolin = .true.
  lspinorb = .true.
/
&ELECTRONS
  conv_thr = 1.0d-10
  mixing_beta = 0.3
/
ATOMIC_SPECIES
  Bi  208.98040  {PP_BI}
  Se   78.96000  {PP_SE}
ATOMIC_POSITIONS {{crystal}}
  Bi  {bi_mu:.6f}  {bi_mu:.6f}  {bi_mu:.6f}
  Bi  {1-bi_mu:.6f}  {1-bi_mu:.6f}  {1-bi_mu:.6f}
  Se  {se_nu:.6f}  {se_nu:.6f}  {se_nu:.6f}
  Se  {1-se_nu:.6f}  {1-se_nu:.6f}  {1-se_nu:.6f}
  Se  0.000000  0.000000  0.000000
K_POINTS {{automatic}}
  8 8 8 0 0 0
"""

with open(os.path.join(WORK_DIR, "scf.in"), "w") as f:
    f.write(scf_input)

print("SCF input written to", os.path.join(WORK_DIR, "scf.in"))
```

#### Step 2: Run SCF

```bash
cd z2_bi2se3
# Run SCF -- adjust mpirun as needed
mpirun -np 4 pw.x -in scf.in > scf.out 2>&1
# Verify convergence
grep "convergence has been achieved" scf.out
grep "highest occupied" scf.out
```

#### Step 3: Set Up z2pack Calculation

```python
#!/usr/bin/env python3
"""
z2pack workflow: compute Z2 invariant for Bi2Se3.
z2pack drives QE NSCF + Wannier90 iteratively to compute Wilson loops.
"""
import os
import subprocess
import z2pack
import matplotlib.pyplot as plt
import numpy as np

WORK_DIR = os.path.abspath("z2_bi2se3")

# ------------------------------------------------------------------
# Pseudopotential names (must match SCF)
# ------------------------------------------------------------------
PP_BI = "Bi.rel-pbe-dn-kjpaw_psl.1.0.0.UPF"
PP_SE = "Se.rel-pbe-dn-kjpaw_psl.1.0.0.UPF"
PSEUDO_DIR = os.environ.get("PSEUDO_DIR", "./pseudo")

a_rhomb = 9.841
cos_alpha = np.cos(np.radians(24.304))
bi_mu = 0.4006
se_nu = 0.2060

# ------------------------------------------------------------------
# NSCF input template for z2pack
# z2pack replaces K_POINTS block automatically
# ------------------------------------------------------------------
nscf_input = f"""&CONTROL
  calculation = 'nscf'
  prefix = 'bi2se3'
  outdir = './tmp'
  pseudo_dir = '{PSEUDO_DIR}'
  verbosity = 'high'
/
&SYSTEM
  ibrav = 5
  celldm(1) = {a_rhomb}
  celldm(4) = {cos_alpha:.6f}
  nat = 5
  ntyp = 2
  ecutwfc = 50.0
  ecutrho = 400.0
  occupations = 'smearing'
  smearing = 'cold'
  degauss = 0.005
  noncolin = .true.
  lspinorb = .true.
  nosym = .true.
  nbnd = 36
/
&ELECTRONS
  conv_thr = 1.0d-10
  mixing_beta = 0.3
  diago_full_acc = .true.
/
ATOMIC_SPECIES
  Bi  208.98040  {PP_BI}
  Se   78.96000  {PP_SE}
ATOMIC_POSITIONS {{crystal}}
  Bi  {bi_mu:.6f}  {bi_mu:.6f}  {bi_mu:.6f}
  Bi  {1-bi_mu:.6f}  {1-bi_mu:.6f}  {1-bi_mu:.6f}
  Se  {se_nu:.6f}  {se_nu:.6f}  {se_nu:.6f}
  Se  {1-se_nu:.6f}  {1-se_nu:.6f}  {1-se_nu:.6f}
  Se  0.000000  0.000000  0.000000
"""

with open(os.path.join(WORK_DIR, "nscf.in"), "w") as f:
    f.write(nscf_input)

# ------------------------------------------------------------------
# Wannier90 input (bi2se3.win)
# Bi2Se3 has 5 atoms, with SOC -> 10 spinor bands from p-orbitals
# We wannierize the occupied + low-lying conduction bands
# num_wann must be even (spinor) and cover the occupied manifold
# ------------------------------------------------------------------
# 5 atoms x 2 (spin) x 3 (p orbitals) = 30 possible,
# but we only need enough to span the gap.
# Typical: 14 Wannier functions (covering valence + lower conduction)
num_wann = 14
num_bands = 36  # total bands in NSCF

wannier_input = f"""num_wann = {num_wann}
num_bands = {num_bands}

! Exclude bands far from Fermi level
! (adjust based on actual band structure)
exclude_bands = 1-8

! Projections: Bi-p and Se-p orbitals
begin projections
Bi: p
Se: p
end projections

! Write overlap matrices (needed by z2pack)
write_mmn = .true.
write_amn = .true.
write_eig = .true.

! Spinor projections for SOC
spinors = .true.

! Do not perform wannierization -- z2pack only needs overlaps
! z2pack controls the k-points
"""

with open(os.path.join(WORK_DIR, "bi2se3.win"), "w") as f:
    f.write(wannier_input)

# ------------------------------------------------------------------
# Build the z2pack system
# ------------------------------------------------------------------

# z2pack needs to call: pw.x (nscf), wannier90.x -pp, pw2wannier90.x
# We create a z2pack.qe system

# Build input files for pw2wannier90
pw2wan_input = f"""&inputpp
  outdir = './tmp'
  prefix = 'bi2se3'
  seedname = 'bi2se3'
  spin_component = 'none'
  write_mmn = .true.
  write_amn = .true.
  write_unk = .false.
/
"""

with open(os.path.join(WORK_DIR, "pw2wan.in"), "w") as f:
    f.write(pw2wan_input)

# ------------------------------------------------------------------
# z2pack system definition
# ------------------------------------------------------------------
# z2pack calls a build function that runs the QE pipeline for each k-string

qe_system = z2pack.fp.System(
    input_files=[
        os.path.join(WORK_DIR, "nscf.in"),
        os.path.join(WORK_DIR, "bi2se3.win"),
        os.path.join(WORK_DIR, "pw2wan.in"),
    ],
    kpt_fct=z2pack.fp.kpoint.qe_explicit,
    kpt_path="nscf.in",  # file where K_POINTS are replaced
    command=(
        "cd {work_dir} && "
        "mpirun -np 4 pw.x -in nscf.in > nscf.out 2>&1 && "
        "wannier90.x -pp bi2se3 > wan_pp.out 2>&1 && "
        "mpirun -np 4 pw2wannier90.x -in pw2wan.in > pw2wan.out 2>&1"
    ),
    executable="",  # command already includes executables
    mmn_path="bi2se3.mmn",
    build_folder=os.path.join(WORK_DIR, "z2pack_build"),
)

# ------------------------------------------------------------------
# Compute Z2 on the six TRI planes
# For 3D Z2 = (nu0; nu1 nu2 nu3):
#   nu0 = (z2_k0 + z2_k1) mod 2   for each pair of TRI planes
# TRI planes in rhombohedral BZ:
#   k1=0, k1=0.5; k2=0, k2=0.5; k3=0, k3=0.5
# ------------------------------------------------------------------

results = {}
surfaces = {
    "k3_0":   lambda s, t: [s, t, 0.0],
    "k3_0.5": lambda s, t: [s, t, 0.5],
    "k2_0":   lambda s, t: [s, 0.0, t],
    "k2_0.5": lambda s, t: [s, 0.5, t],
    "k1_0":   lambda s, t: [0.0, s, t],
    "k1_0.5": lambda s, t: [0.5, s, t],
}

for name, surface in surfaces.items():
    print(f"\n{'='*60}")
    print(f"Computing Wilson loop on surface: {name}")
    print(f"{'='*60}")

    result = z2pack.surface.run(
        system=qe_system,
        surface=surface,
        num_lines=11,        # number of k-lines (odd for better convergence)
        min_neighbour_dist=0.01,
        iterator=range(8, 27, 2),  # adaptive refinement
        pos_tol=0.01,
        gap_tol=0.3,
        move_tol=0.3,
        save_file=os.path.join(WORK_DIR, f"result_{name}.json"),
    )
    results[name] = result
    z2_val = z2pack.invariant.z2(result)
    print(f"Z2 invariant on {name}: {z2_val}")

# ------------------------------------------------------------------
# Compute the full 3D Z2 invariant
# ------------------------------------------------------------------
z2_k1_0   = z2pack.invariant.z2(results["k1_0"])
z2_k1_05  = z2pack.invariant.z2(results["k1_0.5"])
z2_k2_0   = z2pack.invariant.z2(results["k2_0"])
z2_k2_05  = z2pack.invariant.z2(results["k2_0.5"])
z2_k3_0   = z2pack.invariant.z2(results["k3_0"])
z2_k3_05  = z2pack.invariant.z2(results["k3_0.5"])

nu1 = z2_k1_0
nu2 = z2_k2_0
nu3 = z2_k3_0
nu0 = (z2_k1_0 + z2_k1_05) % 2  # strong invariant

print(f"\n{'='*60}")
print(f"Z2 Topological Invariant: ({nu0}; {nu1}{nu2}{nu3})")
print(f"{'='*60}")
if nu0 == 1:
    print("Result: STRONG topological insulator")
elif nu1 or nu2 or nu3:
    print("Result: WEAK topological insulator")
else:
    print("Result: Trivial insulator")

# ------------------------------------------------------------------
# Plot Wilson loop (HWCC) evolution for the k3=0 plane
# ------------------------------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

for ax, (name, result) in zip(axes, [("k3_0", results["k3_0"]),
                                       ("k3_0.5", results["k3_0.5"])]):
    z2pack.plot.wcc(result, axis=ax)
    z2_val = z2pack.invariant.z2(result)
    ax.set_title(f"{name} plane, Z2 = {z2_val}")
    ax.set_xlabel("$k_1$")
    ax.set_ylabel("WCC ($\\bar{x}$)")

fig.suptitle("Bi$_2$Se$_3$ Wilson Loop (Hybrid Wannier Centers)")
plt.tight_layout()
plt.savefig(os.path.join(WORK_DIR, "wilson_loop_bi2se3.png"), dpi=150)
plt.close()
print(f"\nWilson loop plot saved to {os.path.join(WORK_DIR, 'wilson_loop_bi2se3.png')}")
```

### Method 2: Parity Analysis at TRIM Points (Fu-Kane Formula)

This method is simpler but only works for crystals with inversion symmetry.

```python
#!/usr/bin/env python3
"""
Z2 from parity eigenvalues at TRIM points (Fu-Kane formula).
For inversion-symmetric crystals only.
Works by parsing QE output at TRIM k-points.
"""
import os
import re
import subprocess
import numpy as np

WORK_DIR = os.path.abspath("z2_bi2se3_parity")
os.makedirs(WORK_DIR, exist_ok=True)

# ------------------------------------------------------------------
# Step 1: NSCF at TRIM points only
# For rhombohedral BZ, the 8 TRIM points are:
#   Gamma(0,0,0), Z(0.5,0.5,0.5),
#   3 L points, 3 F points (related by C3)
# In fractional coords of reciprocal lattice:
# ------------------------------------------------------------------
TRIM_POINTS = {
    "Gamma": [0.0, 0.0, 0.0],
    "Z":     [0.5, 0.5, 0.5],
    "L1":    [0.5, 0.0, 0.0],
    "L2":    [0.0, 0.5, 0.0],
    "L3":    [0.0, 0.0, 0.5],
    "F1":    [0.5, 0.5, 0.0],
    "F2":    [0.0, 0.5, 0.5],
    "F3":    [0.5, 0.0, 0.5],
}


def parse_parity_eigenvalues(output_file, n_occupied_pairs):
    """
    Parse parity eigenvalues from QE output.

    For SOC calculations, bands come in Kramers pairs.
    We need parity eigenvalue for each Kramers pair.
    QE prints symmetry representations at high-symmetry points
    when verbosity='high'.

    In practice, we read the eigenvalues and their parity
    labels from the output.

    Parameters
    ----------
    output_file : str
        Path to QE output file
    n_occupied_pairs : int
        Number of occupied Kramers pairs

    Returns
    -------
    dict : mapping TRIM label -> list of parity eigenvalues (+1 or -1)
    """
    parities = {}

    with open(output_file, "r") as f:
        content = f.read()

    # QE prints symmetry info like:
    #   "e(  1) = ... eV  ...  (rep  1, +-)"
    # where +- indicates parity.
    # The exact format depends on QE version and symmetry group.

    # Alternative approach: use <psi|P|psi> from QE's projections
    # or use the band2parity.x utility if available.

    # Here we demonstrate a robust approach: compute overlaps
    # <psi_nk | P | psi_nk> numerically from wavefunctions.

    # For a practical parser, look for lines like:
    # "Band symmetry, C_3v ..." and the symmetry labels
    pattern = re.compile(
        r"k\s*=\s*([\d\.\-]+)\s+([\d\.\-]+)\s+([\d\.\-]+).*?"
        r"Band symmetry.*?\n((?:\s+e\(.*\n)*)",
        re.DOTALL,
    )

    matches = pattern.findall(content)

    for match in matches:
        kx, ky, kz = float(match[0]), float(match[1]), float(match[2])
        band_lines = match[3]

        # Identify which TRIM point this is
        k_vec = np.array([kx, ky, kz])
        trim_label = None
        for label, trim_k in TRIM_POINTS.items():
            if np.allclose(k_vec, trim_k, atol=1e-4):
                trim_label = label
                break
        if trim_label is None:
            continue

        # Parse parity from symmetry labels
        # Even representations (Ag, A1g, etc.) have parity +1
        # Odd representations (Au, A1u, etc.) have parity -1
        parity_list = []
        for line in band_lines.strip().split("\n"):
            if "g" in line.lower() and "u" not in line.lower():
                parity_list.append(+1)
            elif "u" in line.lower():
                parity_list.append(-1)

        parities[trim_label] = parity_list[:n_occupied_pairs]

    return parities


def fu_kane_z2(parities):
    """
    Compute Z2 invariant from parity eigenvalues using Fu-Kane formula.

    (-1)^nu0 = product over all TRIM points of delta_i
    where delta_i = product over occupied Kramers pairs of xi_{2m}(TRIM_i)

    Parameters
    ----------
    parities : dict
        Mapping TRIM label -> list of parity eigenvalues for occupied Kramers pairs

    Returns
    -------
    tuple : (nu0, nu1, nu2, nu3)
    """
    # Compute delta_i for each TRIM
    deltas = {}
    for label, parity_list in parities.items():
        delta = 1
        for p in parity_list:
            delta *= p
        deltas[label] = delta
        print(f"  TRIM {label:6s}: delta = {delta:+d}  "
              f"(parities: {[f'{p:+d}' for p in parity_list]})")

    # Strong Z2 index: product over ALL 8 TRIM points
    product_all = 1
    for delta in deltas.values():
        product_all *= delta
    nu0 = 0 if product_all == 1 else 1

    # Weak Z2 indices: product over 4 TRIM points in each plane
    # nu1: TRIM with k1 = 0.5 -> L1, F1, F3, Z
    product_nu1 = deltas.get("L1", 1) * deltas.get("F1", 1) * \
                  deltas.get("F3", 1) * deltas.get("Z", 1)
    nu1 = 0 if product_nu1 == 1 else 1

    # nu2: TRIM with k2 = 0.5 -> L2, F1, F2, Z
    product_nu2 = deltas.get("L2", 1) * deltas.get("F1", 1) * \
                  deltas.get("F2", 1) * deltas.get("Z", 1)
    nu2 = 0 if product_nu2 == 1 else 1

    # nu3: TRIM with k3 = 0.5 -> L3, F2, F3, Z
    product_nu3 = deltas.get("L3", 1) * deltas.get("F2", 1) * \
                  deltas.get("F3", 1) * deltas.get("Z", 1)
    nu3 = 0 if product_nu3 == 1 else 1

    return (nu0, nu1, nu2, nu3)


# ------------------------------------------------------------------
# Example: Bi2Se3 parity analysis (with known results)
# In Bi2Se3, band inversion occurs at Gamma:
#   Gamma: one Kramers pair switches parity -> delta(Gamma) = -1
#   All other TRIM: delta = +1
# This gives Z2 = (1; 000) -> strong TI
# ------------------------------------------------------------------

# If we already have QE output with symmetry labels:
print("="*60)
print("Fu-Kane Parity Analysis for Bi2Se3")
print("="*60)

# Known parity results for Bi2Se3 (from literature/DFT):
# At Gamma, the inverted band flips one parity eigenvalue
# Occupied Kramers pairs have parities:
known_parities = {
    "Gamma": [+1, +1, +1, -1, +1, -1, +1],  # one inverted pair
    "Z":     [+1, +1, +1, +1, +1, -1, +1],
    "L1":    [+1, +1, +1, +1, +1, -1, +1],
    "L2":    [+1, +1, +1, +1, +1, -1, +1],
    "L3":    [+1, +1, +1, +1, +1, -1, +1],
    "F1":    [+1, +1, +1, +1, +1, -1, +1],
    "F2":    [+1, +1, +1, +1, +1, -1, +1],
    "F3":    [+1, +1, +1, +1, +1, -1, +1],
}

z2 = fu_kane_z2(known_parities)
print(f"\nZ2 = ({z2[0]}; {z2[1]}{z2[2]}{z2[3]})")
if z2[0] == 1:
    print("Strong topological insulator confirmed!")


# ------------------------------------------------------------------
# Generate NSCF input for TRIM-point calculation
# ------------------------------------------------------------------
def write_trim_nscf(work_dir, trim_points, pseudo_dir="./pseudo"):
    """Write QE NSCF input targeting TRIM points only."""

    nk = len(trim_points)
    kpt_block = f"K_POINTS {{crystal}}\n{nk}\n"
    for label, kpt in trim_points.items():
        kpt_block += f"  {kpt[0]:.8f} {kpt[1]:.8f} {kpt[2]:.8f}  1.0  ! {label}\n"

    cos_alpha = np.cos(np.radians(24.304))
    bi_mu = 0.4006
    se_nu = 0.2060

    PP_BI = "Bi.rel-pbe-dn-kjpaw_psl.1.0.0.UPF"
    PP_SE = "Se.rel-pbe-dn-kjpaw_psl.1.0.0.UPF"

    nscf_in = f"""&CONTROL
  calculation = 'nscf'
  prefix = 'bi2se3'
  outdir = './tmp'
  pseudo_dir = '{pseudo_dir}'
  verbosity = 'high'
/
&SYSTEM
  ibrav = 5
  celldm(1) = 9.841
  celldm(4) = {cos_alpha:.6f}
  nat = 5
  ntyp = 2
  ecutwfc = 50.0
  ecutrho = 400.0
  occupations = 'smearing'
  smearing = 'cold'
  degauss = 0.005
  noncolin = .true.
  lspinorb = .true.
  nbnd = 36
/
&ELECTRONS
  conv_thr = 1.0d-10
  diago_full_acc = .true.
/
ATOMIC_SPECIES
  Bi  208.98040  {PP_BI}
  Se   78.96000  {PP_SE}
ATOMIC_POSITIONS {{crystal}}
  Bi  {bi_mu:.6f}  {bi_mu:.6f}  {bi_mu:.6f}
  Bi  {1-bi_mu:.6f}  {1-bi_mu:.6f}  {1-bi_mu:.6f}
  Se  {se_nu:.6f}  {se_nu:.6f}  {se_nu:.6f}
  Se  {1-se_nu:.6f}  {1-se_nu:.6f}  {1-se_nu:.6f}
  Se  0.000000  0.000000  0.000000
{kpt_block}"""

    filepath = os.path.join(work_dir, "nscf_trim.in")
    with open(filepath, "w") as f:
        f.write(nscf_in)
    print(f"NSCF input for TRIM points written to {filepath}")
    return filepath


write_trim_nscf(WORK_DIR, TRIM_POINTS)
```

### Method 3: Simplified z2pack with Tight-Binding Model (Testing/Validation)

```python
#!/usr/bin/env python3
"""
Quick Z2 test using z2pack with a tight-binding model (no DFT needed).
Useful for validating the z2pack workflow before expensive DFT runs.
Uses the BHZ (Bernevig-Hughes-Zhang) model for HgTe/CdTe quantum wells.
"""
import numpy as np
import z2pack
import matplotlib.pyplot as plt


def bhz_hamiltonian(k, A=0.3645, B=-0.686, C=0.0, D=-0.512, M=-0.01):
    """
    BHZ model Hamiltonian for HgTe/CdTe quantum well.
    4x4 matrix in basis |E1+>, |H1+>, |E1->, |H1->

    Parameters from Bernevig, Hughes, Zhang, Science 314, 1757 (2006).
    M < 0 -> topological (inverted), M > 0 -> trivial
    """
    kx, ky = k[0], k[1]
    k_sq = kx**2 + ky**2

    epsilon_k = C - D * k_sq
    d_z = M - B * k_sq
    d_plus = A * (kx + 1j * ky)   # A * k_+
    d_minus = A * (kx - 1j * ky)  # A * k_-

    # Upper block (spin up): h(k)
    h_upper = np.array([
        [epsilon_k + d_z,  d_plus],
        [d_minus,          epsilon_k - d_z],
    ])

    # Lower block (spin down): h*(-k) (time-reversal partner)
    h_lower = np.array([
        [epsilon_k + d_z,  -d_minus],
        [-d_plus,          epsilon_k - d_z],
    ])

    H = np.zeros((4, 4), dtype=complex)
    H[:2, :2] = h_upper
    H[2:, 2:] = h_lower

    return H


# ------------------------------------------------------------------
# Create z2pack tight-binding system
# ------------------------------------------------------------------
# For 2D Z2, we compute Wilson loop on the half-BZ

# Topological case: M = -0.01 (inverted)
print("="*60)
print("BHZ Model - Topological Phase (M < 0)")
print("="*60)

system_topo = z2pack.hm.System(
    lambda k: bhz_hamiltonian([k[0] * 2 * np.pi, k[1] * 2 * np.pi], M=-0.01),
    dim=2,
    bands=2,  # number of occupied bands
)

result_topo = z2pack.surface.run(
    system=system_topo,
    surface=lambda s, t: [s, t],
    num_lines=51,
    pos_tol=None,
    gap_tol=None,
    move_tol=None,
)

z2_topo = z2pack.invariant.z2(result_topo)
print(f"Z2 = {z2_topo}")

# Trivial case: M = +0.01 (normal)
print("\n" + "="*60)
print("BHZ Model - Trivial Phase (M > 0)")
print("="*60)

system_trivial = z2pack.hm.System(
    lambda k: bhz_hamiltonian([k[0] * 2 * np.pi, k[1] * 2 * np.pi], M=+0.01),
    dim=2,
    bands=2,
)

result_trivial = z2pack.surface.run(
    system=system_trivial,
    surface=lambda s, t: [s, t],
    num_lines=51,
    pos_tol=None,
    gap_tol=None,
    move_tol=None,
)

z2_trivial = z2pack.invariant.z2(result_trivial)
print(f"Z2 = {z2_trivial}")

# ------------------------------------------------------------------
# Plot Wilson loops
# ------------------------------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

z2pack.plot.wcc(result_topo, axis=axes[0])
axes[0].set_title(f"Topological (M<0), Z2={z2_topo}")
axes[0].set_xlabel("$k_y / (2\\pi)$")
axes[0].set_ylabel("WCC ($\\bar{x}$)")

z2pack.plot.wcc(result_trivial, axis=axes[1])
axes[1].set_title(f"Trivial (M>0), Z2={z2_trivial}")
axes[1].set_xlabel("$k_y / (2\\pi)$")
axes[1].set_ylabel("WCC ($\\bar{x}$)")

plt.tight_layout()
plt.savefig("bhz_wilson_loops.png", dpi=150)
plt.close()
print("\nWilson loop plot saved to bhz_wilson_loops.png")
print(f"\nExpected: Topological Z2=1, Trivial Z2=0")
print(f"Got:      Topological Z2={z2_topo}, Trivial Z2={z2_trivial}")
```

## Key Parameters

| Parameter | Typical Value | Notes |
|-----------|--------------|-------|
| `ecutwfc` | 50-80 Ry | Higher for heavier elements; test convergence |
| `ecutrho` | 400-640 Ry | 8x ecutwfc for PAW/US |
| k-grid (SCF) | 8x8x8 | Must be well-converged |
| `noncolin` | `.true.` | Required for SOC |
| `lspinorb` | `.true.` | Required for SOC |
| `nosym` | `.true.` | Required in NSCF for z2pack/Wannier90 |
| `num_wann` | System-dependent | Must be even (spinor); cover occupied manifold |
| z2pack `num_lines` | 11-51 | Odd values; more lines = better convergence |
| z2pack `pos_tol` | 0.01 | Tolerance for WCC positions |
| z2pack `gap_tol` | 0.3 | Tolerance for gap detection in WCC spectrum |
| Pseudopotentials | FR (fully relativistic) | `_rel` suffix in PSlibrary; `_FR` in ONCVPSP |

## Interpreting Results

**Z2 = (1; 000):** Strong topological insulator. Robust metallic surface states on ALL
surfaces. Example: Bi2Se3, Bi2Te3. Cannot be removed by disorder (only by closing the
bulk gap).

**Z2 = (0; nu1 nu2 nu3) with some nu_i=1:** Weak topological insulator. Surface states
exist only on certain surfaces. Less robust -- can be localized by disorder. Example:
stacked 2D TI layers.

**Z2 = (0; 000):** Trivial insulator. No topologically protected surface states.

**Wilson loop interpretation:**
- Plot WCC (Wannier charge centers) vs. k along one direction
- If a horizontal reference line at any y-value crosses the WCC lines an ODD number of
  times -> Z2 = 1 (topological)
- If all reference lines cross an EVEN number of times -> Z2 = 0 (trivial)
- The WCC lines should show clear "winding" pattern for Z2 = 1

## Common Issues

1. **SCF does not converge with SOC:**
   - Reduce `mixing_beta` to 0.1-0.2
   - Start from a converged non-SOC calculation (use `startingpot = 'file'`)
   - Increase `electron_maxstep`

2. **z2pack convergence issues:**
   - Increase `num_lines` (try 21, 31, 51)
   - Decrease `pos_tol` and `gap_tol`
   - Check that the gap does not close on the integration surface
   - Verify Wannier90 convergence (check spread and centers)

3. **Wrong Z2 value:**
   - Ensure fully relativistic pseudopotentials are used (not scalar-relativistic)
   - Check that `noncolin = .true.` AND `lspinorb = .true.` are both set
   - Verify that the band gap is correct (compare with experiment)
   - Check Wannier90 disentanglement windows carefully

4. **Parity method gives wrong result:**
   - Only valid for centrosymmetric crystals
   - Make sure the inversion center is correctly identified
   - Band ordering at TRIM points must be correctly tracked
   - Some bands may be nearly degenerate -- check carefully

5. **Pseudopotential errors with SOC:**
   - Must use fully relativistic (FR) pseudopotentials, not scalar relativistic
   - PSlibrary: use `_rel` suffix (e.g., `Bi.rel-pbe-dn-kjpaw_psl.1.0.0.UPF`)
   - ONCVPSP: use `_FR` variant
   - Ensure consistency: all PPs must be FR when `lspinorb = .true.`

6. **Memory issues:**
   - SOC doubles the number of bands (spinor wavefunctions)
   - Reduce `nbnd` if possible
   - Use k-point parallelization (`-nk` flag in `mpirun`)
