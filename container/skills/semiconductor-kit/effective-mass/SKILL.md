# Carrier Effective Mass

## When to Use

- You need the carrier effective mass (m*/m_e) of electrons or holes near band extrema (CBM or VBM).
- You want to understand transport properties at a basic level (mobility ~ 1/m*).
- You need the effective mass tensor for anisotropic materials (different m* along different crystal directions).
- You are studying molecular crystals and need HOMO/LUMO effective masses.
- You want to classify bands as light-hole, heavy-hole, or split-off bands.
- You need input parameters for device simulation (Schrodinger-Poisson, drift-diffusion models).

## Method Selection

| Method | Tool | Strengths | Limitations |
|---|---|---|---|
| QE parabolic fit | pw.x + Python | Available in container, flexible | Requires dense k-mesh near extrema |
| VASP parabolic fit | VASP + Python | Standard VASP workflow | Requires VASP license |
| BoltzTraP2 transport mass | BoltzTraP2 | Accounts for non-parabolic bands | Gives transport average, not directional |
| Finite difference | Python | No fitting needed, works with any DFT | Sensitive to k-point spacing |
| sumo effective mass | sumo + VASP/QE | Automated, publication-quality | Needs `pip install sumo` |

## Prerequisites

- A completed band structure calculation along high-symmetry k-paths with dense sampling near band extrema.
- Alternatively, a dense uniform k-grid NSCF for tensor extraction.
- Python packages: `numpy`, `scipy`, `matplotlib`.
- For QE: pw.x output or bands.x output.
- For VASP: EIGENVAL or vasprun.xml.

## Background

### Effective Mass Definition

The effective mass is defined from the band curvature:

```
1/m*_ij = (1/hbar^2) * d^2E/dk_i dk_j
```

For a parabolic band near an extremum at k_0:

```
E(k) = E_0 + (hbar^2 / 2) * sum_ij (k_i - k0_i) * (1/m*_ij) * (k_j - k0_j)
```

The effective mass tensor is a 3x3 symmetric matrix. Its eigenvalues give the principal effective masses, and its eigenvectors give the principal directions.

### Typical Values

| Material | Carrier | m*/m_e | Notes |
|---|---|---|---|
| Si | electron (longitudinal) | 0.98 | Along Delta direction |
| Si | electron (transverse) | 0.19 | Perpendicular to Delta |
| Si | heavy hole | 0.49 | At Gamma |
| Si | light hole | 0.16 | At Gamma |
| GaAs | electron | 0.067 | At Gamma (very light) |
| GaAs | heavy hole | 0.50 | At Gamma |
| ZnO | electron | 0.28 | At Gamma |
| Graphene | Dirac fermion | ~0 | Linear dispersion, m*=0 |

## Detailed Steps

### Overview

```
Method A: QE parabolic fit along k-path
  Step 1: Dense band calculation near extrema
  Step 2: Parabolic fit to extract m*
  Step 3: Multi-direction fit for tensor

Method B: VASP parabolic fit
  Step 1: Dense band calculation
  Step 2: Parse EIGENVAL and fit

Method C: Full effective mass tensor from dense k-grid
  Step 1: Dense NSCF near extremum
  Step 2: Finite-difference second derivatives
  Step 3: Diagonalize tensor

Method D: HOMO/LUMO effective mass for molecular crystals
  Same approach but applied to HOMO/LUMO bands
```

---

### Method A: QE Parabolic Fit Along K-Path

#### Step A1: Dense Band Calculation Near Extrema

```python
#!/usr/bin/env python3
"""
Generate a dense k-path near band extrema for effective mass calculation.
Uses seekpath to find high-symmetry points, then adds extra k-points
around the VBM and CBM.
"""
import os
import subprocess
import numpy as np

# ── Generate dense k-path near extrema ────────────────────────────

def generate_dense_kpath_near_extremum(k_extremum, direction_vectors,
                                        n_points=21, dk_range=0.05):
    """
    Generate a dense k-point set around a band extremum for fitting.

    Parameters
    ----------
    k_extremum : array (3,)
        K-point of the extremum in crystal coordinates.
    direction_vectors : list of array (3,)
        Directions along which to sample. Each should be a unit vector
        in crystal coordinates.
    n_points : int
        Number of k-points per direction (odd number centered on extremum).
    dk_range : float
        Range in crystal coordinates on each side of extremum.

    Returns
    -------
    kpoints : list of (kx, ky, kz) in crystal coordinates
    labels : list of direction labels
    """
    kpoints = []
    labels = []

    for iv, direction in enumerate(direction_vectors):
        # Normalize direction
        d = np.array(direction, dtype=float)
        d_norm = d / np.linalg.norm(d)

        # Generate points along this direction
        dk_values = np.linspace(-dk_range, dk_range, n_points)
        for dk in dk_values:
            k = np.array(k_extremum) + dk * d_norm
            kpoints.append(k)
            labels.append(f'dir{iv}')

    return kpoints, labels


def write_qe_kpoints_crystal(kpoints, filename='kpoints_mstar.txt'):
    """Write k-points in QE crystal format for NSCF."""
    nk = len(kpoints)
    lines = [f"K_POINTS {{crystal}}", f"{nk}"]
    w = 1.0 / nk
    for k in kpoints:
        lines.append(f"  {k[0]:14.10f}  {k[1]:14.10f}  "
                      f"{k[2]:14.10f}  {w:.10f}")
    with open(filename, 'w') as f:
        f.write("\n".join(lines) + "\n")
    print(f"Written: {filename} ({nk} k-points)")
    return filename


# ── Example: Si CBM near X point ──────────────────────────────────
# Si CBM is at ~0.85 * (1,0,0) in crystal coordinates of the
# conventional cell. In the primitive cell (ibrav=2), the mapping
# is different. For demonstration, we use Gamma-point:

# For Si: VBM at Gamma, CBM at ~0.85 along Gamma-X
# VBM (Gamma point) -- sample along [100], [110], [111]
vbm_k = [0.0, 0.0, 0.0]
vbm_directions = [
    [1, 0, 0],   # Gamma-X
    [0, 1, 0],   # Gamma-Y
    [0, 0, 1],   # Gamma-Z
    [1, 1, 0],   # Gamma-M
    [1, 1, 1],   # Gamma-R
]

kpts_vbm, _ = generate_dense_kpath_near_extremum(
    vbm_k, vbm_directions, n_points=21, dk_range=0.04)

# CBM (near X point for Si)
cbm_k = [0.425, 0.425, 0.0]  # Approximate CBM in primitive cell
cbm_directions = [
    [1, 1, 0],    # Longitudinal (along Delta)
    [1, -1, 0],   # Transverse 1
    [0, 0, 1],    # Transverse 2
]

kpts_cbm, _ = generate_dense_kpath_near_extremum(
    cbm_k, cbm_directions, n_points=21, dk_range=0.04)

# Combine
all_kpts = kpts_vbm + kpts_cbm
write_qe_kpoints_crystal(all_kpts, 'kpoints_mstar.txt')

print(f"\nGenerated {len(kpts_vbm)} k-points near VBM (Gamma)")
print(f"Generated {len(kpts_cbm)} k-points near CBM")
print(f"Total: {len(all_kpts)} k-points")
print("\nPaste the K_POINTS block into your QE NSCF input file.")
```

#### Step A2: Run QE NSCF for Effective Mass

```python
#!/usr/bin/env python3
"""
Run QE NSCF calculation with dense k-points near band extrema
for effective mass extraction.
"""
import os
import subprocess

PSEUDO_DIR = os.path.abspath("./pseudo")
OUTDIR = os.path.abspath("./tmp_mstar")
os.makedirs(OUTDIR, exist_ok=True)

# Read the k-points file generated in Step A1
with open("kpoints_mstar.txt", "r") as f:
    kpoints_block = f.read()

nscf_input = f"""&CONTROL
    calculation   = 'nscf'
    prefix        = 'si'
    pseudo_dir    = '{PSEUDO_DIR}'
    outdir        = '{OUTDIR}'
    verbosity     = 'high'
/

&SYSTEM
    ibrav         = 2
    celldm(1)     = 10.2631
    nat           = 2
    ntyp          = 1
    ecutwfc       = 60.0
    ecutrho       = 480.0
    occupations   = 'smearing'
    smearing      = 'gaussian'
    degauss       = 0.005
    nbnd          = 16
/

&ELECTRONS
    conv_thr      = 1.0d-10
/

ATOMIC_SPECIES
  Si  28.0855  Si.pbe-n-kjpaw_psl.1.0.0.UPF

ATOMIC_POSITIONS {{crystal}}
  Si  0.00  0.00  0.00
  Si  0.25  0.25  0.25

{kpoints_block}
"""

with open("nscf_mstar.in", "w") as f:
    f.write(nscf_input)
print("Written: nscf_mstar.in")

# NOTE: SCF must be completed first in OUTDIR with same prefix.
# Copy charge density if needed:
# cp -r ./tmp/si.save ./tmp_mstar/si.save

print("Running NSCF for effective mass ...")
result = subprocess.run(
    ["mpirun", "-np", "4", "pw.x", "-in", "nscf_mstar.in"],
    capture_output=True, text=True, timeout=3600
)
with open("nscf_mstar.out", "w") as f:
    f.write(result.stdout)

if result.returncode != 0:
    print("ERROR: NSCF failed!")
    print(result.stderr[-500:] if result.stderr else "")
else:
    print("NSCF completed. Output: nscf_mstar.out")
```

#### Step A3: Extract Effective Mass via Parabolic Fit

```python
#!/usr/bin/env python3
"""
Extract effective mass from QE NSCF eigenvalues via parabolic fitting.

Workflow:
1. Parse eigenvalues from pw.x output
2. Identify VBM/CBM bands
3. Fit E(k) = E0 + hbar^2 * k^2 / (2 * m*) along each direction
4. Report m*/m_e for each direction
"""
import re
import os
import json
import numpy as np
from scipy.optimize import curve_fit
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


# ── Physical constants ─────────────────────────────────────────────
HBAR_SI = 1.054571817e-34       # J*s
M_E = 9.1093837015e-31          # kg
EV_TO_J = 1.602176634e-19       # J/eV
BOHR_TO_M = 5.29177210903e-11   # m
ANGSTROM_TO_M = 1e-10


def parse_nscf_eigenvalues(filename):
    """
    Parse k-points and eigenvalues from QE pw.x NSCF output.

    Returns
    -------
    kpoints : list of ndarray (3,) in crystal coordinates
    eigenvalues : ndarray (nk, nbnd) in eV
    fermi_energy : float in eV
    """
    with open(filename, 'r') as f:
        lines = f.readlines()

    fermi_energy = None
    for line in lines:
        if 'Fermi energy' in line:
            m = re.search(r'Fermi energy\s+is\s+([-\d.]+)', line)
            if m:
                fermi_energy = float(m.group(1))
        if 'highest occupied' in line:
            nums = re.findall(r'[-\d.]+', line)
            if nums:
                fermi_energy = float(nums[0])

    kpoints = []
    eigenvalues = []
    i = 0
    while i < len(lines):
        if 'k =' in lines[i]:
            m = re.search(
                r'k\s*=\s*([-\d.]+)\s+([-\d.]+)\s+([-\d.]+)', lines[i])
            if m:
                kpt = np.array([float(m.group(1)), float(m.group(2)),
                                float(m.group(3))])
                kpoints.append(kpt)

                j = i + 1
                while j < len(lines) and lines[j].strip() == '':
                    j += 1

                eigs = []
                while j < len(lines) and lines[j].strip() != '':
                    nums = re.findall(r'[-\d.]+', lines[j])
                    eigs.extend([float(x) for x in nums])
                    j += 1
                    if j < len(lines) and 'k =' in lines[j]:
                        break

                eigenvalues.append(eigs)
        i += 1

    nbnd = max(len(e) for e in eigenvalues)
    for i in range(len(eigenvalues)):
        while len(eigenvalues[i]) < nbnd:
            eigenvalues[i].append(np.nan)

    return kpoints, np.array(eigenvalues), fermi_energy


def parabolic(k_dist_inv_m, E0_eV, mstar_me):
    """
    Parabolic band: E(k) = E0 + hbar^2 * k^2 / (2 * m* * m_e)
    k_dist_inv_m in 1/m, returns E in eV.
    """
    return E0_eV + (HBAR_SI**2 * k_dist_inv_m**2) / (
        2 * mstar_me * M_E * EV_TO_J)


def fit_effective_mass(kpoints_cryst, eigenvalues, band_index,
                        k_extremum_index, direction_indices,
                        lattice_matrix_ang, is_valence=False):
    """
    Fit effective mass along a direction through a band extremum.

    Parameters
    ----------
    kpoints_cryst : list of ndarray (3,)
        K-points in crystal (fractional) coordinates.
    eigenvalues : ndarray (nk, nbnd)
        Eigenvalues in eV.
    band_index : int
        Band index (0-based).
    k_extremum_index : int
        Index of the extremum k-point.
    direction_indices : list of int
        Indices of k-points along the fitting direction.
    lattice_matrix_ang : ndarray (3,3)
        Real-space lattice matrix in Angstrom (rows are lattice vectors).
    is_valence : bool
        If True, the band is a valence band (maximum), and the
        curvature should be negative (m* is reported as positive).

    Returns
    -------
    mstar_me : float
        Effective mass in units of free electron mass.
    fit_quality : float
        R^2 of the parabolic fit.
    """
    # Reciprocal lattice vectors (in 1/Angstrom, without 2*pi factor)
    # b_i = 2*pi * (a_j x a_k) / V
    V = np.abs(np.linalg.det(lattice_matrix_ang))
    recip_matrix = 2 * np.pi * np.linalg.inv(lattice_matrix_ang).T

    # Convert k-points to Cartesian (1/Angstrom)
    k_cart = np.array([recip_matrix @ k for k in kpoints_cryst])

    # K-point distances from extremum in 1/m
    k0_cart = k_cart[k_extremum_index]
    k_sel = k_cart[direction_indices]
    dk_cart = k_sel - k0_cart  # (n, 3) in 1/Angstrom
    dk_m = dk_cart * 1e10  # Convert to 1/m

    k_dist = np.array([np.linalg.norm(dk) for dk in dk_m])

    # Assign sign based on projection along the direction vector
    # (to get a proper parabola, we need signed distances)
    if len(dk_cart) > 0:
        direction = dk_cart[-1] - dk_cart[0]
        direction_norm = direction / np.linalg.norm(direction)
        k_signed = np.array([np.dot(dk, direction_norm) * 1e10
                              for dk in dk_cart])
    else:
        k_signed = k_dist

    E_sel = eigenvalues[direction_indices, band_index]

    # Remove NaNs
    valid = ~np.isnan(E_sel)
    k_signed = k_signed[valid]
    E_sel = E_sel[valid]

    if len(k_signed) < 3:
        print("  Not enough points for parabolic fit.")
        return None, 0.0

    # Fit parabola: E = a * k^2 + b * k + c
    coeffs = np.polyfit(k_signed, E_sel, 2)
    a = coeffs[0]  # eV * m^2

    if abs(a) < 1e-30:
        print("  Flat band or zero curvature.")
        return None, 0.0

    # m* = hbar^2 / (2 * a * eV_to_J)
    a_SI = a * EV_TO_J  # J * m^2
    mstar_kg = HBAR_SI**2 / (2 * a_SI)
    mstar_me = mstar_kg / M_E

    # For valence band, curvature is negative, so m* comes out negative
    # Report as positive
    if is_valence:
        mstar_me = abs(mstar_me)

    # R^2
    E_fit = np.polyval(coeffs, k_signed)
    ss_res = np.sum((E_sel - E_fit)**2)
    ss_tot = np.sum((E_sel - np.mean(E_sel))**2)
    r_squared = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0

    return abs(mstar_me), r_squared


def compute_effective_mass_all_directions(kpoints_cryst, eigenvalues,
                                           band_index, k_extremum,
                                           directions, lattice_matrix_ang,
                                           is_valence=False,
                                           n_points_per_dir=21):
    """
    Compute effective mass along multiple directions.

    Parameters
    ----------
    k_extremum : array (3,)
        Extremum k-point in crystal coordinates.
    directions : list of (name, direction_vector)
        Named directions for fitting.
    n_points_per_dir : int
        Number of k-points per direction.
    """
    results = []

    # Group k-points by direction
    nk = len(kpoints_cryst)
    n_dirs = len(directions)

    for idir, (dir_name, dir_vec) in enumerate(directions):
        start = idir * n_points_per_dir
        end = start + n_points_per_dir
        if end > nk:
            print(f"  Direction {dir_name}: not enough k-points")
            continue

        indices = list(range(start, end))
        center_idx = start + n_points_per_dir // 2

        mstar, r2 = fit_effective_mass(
            kpoints_cryst, eigenvalues, band_index,
            center_idx, indices, lattice_matrix_ang, is_valence)

        if mstar is not None:
            results.append({
                'direction': dir_name,
                'mstar_me': mstar,
                'r_squared': r2,
            })
            quality = "GOOD" if r2 > 0.99 else ("OK" if r2 > 0.95 else "POOR")
            print(f"  {dir_name:20s}: m* = {mstar:.4f} m_e  "
                  f"(R^2 = {r2:.6f}, {quality})")

    return results


# ── Main workflow ──────────────────────────────────────────────────

print("=" * 60)
print("Effective Mass Calculation from Band Curvature")
print("=" * 60)

output_file = "nscf_mstar.out"
if not os.path.exists(output_file):
    output_file = "nscf.out"

if os.path.exists(output_file):
    kpoints, eigenvalues, fermi_e = parse_nscf_eigenvalues(output_file)
    print(f"\nParsed: {len(kpoints)} k-points, "
          f"{eigenvalues.shape[1]} bands")
    if fermi_e is not None:
        print(f"Fermi energy: {fermi_e:.4f} eV")

    # Determine occupied bands
    n_occ = 4  # Si: 4 occupied bands per spin (8 electrons / 2)

    # Si lattice (ibrav=2 primitive cell)
    a_si = 5.431  # Angstrom
    lattice = np.array([
        [-a_si/2, 0, a_si/2],
        [0, a_si/2, a_si/2],
        [-a_si/2, a_si/2, 0],
    ])

    # VBM directions (from Step A1)
    vbm_directions = [
        ("[100]", [1, 0, 0]),
        ("[010]", [0, 1, 0]),
        ("[001]", [0, 0, 1]),
        ("[110]", [1, 1, 0]),
        ("[111]", [1, 1, 1]),
    ]
    n_per_dir = 21  # Must match Step A1

    print(f"\n--- VBM Effective Mass (band {n_occ - 1}) ---")
    vbm_results = compute_effective_mass_all_directions(
        kpoints, eigenvalues,
        band_index=n_occ - 1,
        k_extremum=[0, 0, 0],
        directions=vbm_directions,
        lattice_matrix_ang=lattice,
        is_valence=True,
        n_points_per_dir=n_per_dir,
    )

    # CBM directions
    n_vbm_kpts = len(vbm_directions) * n_per_dir
    cbm_kpoints = kpoints[n_vbm_kpts:]
    cbm_eigenvalues = eigenvalues[n_vbm_kpts:]

    cbm_directions = [
        ("longitudinal [110]", [1, 1, 0]),
        ("transverse [1-10]", [1, -1, 0]),
        ("transverse [001]", [0, 0, 1]),
    ]

    print(f"\n--- CBM Effective Mass (band {n_occ}) ---")
    cbm_results = compute_effective_mass_all_directions(
        cbm_kpoints, cbm_eigenvalues,
        band_index=n_occ,
        k_extremum=[0.425, 0.425, 0.0],
        directions=cbm_directions,
        lattice_matrix_ang=lattice,
        is_valence=False,
        n_points_per_dir=n_per_dir,
    )

    # Save results
    all_results = {
        'vbm': vbm_results,
        'cbm': cbm_results,
    }
    with open('effective_mass_results.json', 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved: effective_mass_results.json")

else:
    print(f"Output file not found: {output_file}")
    print("Run the NSCF calculation first (Step A2).")
```

#### Step A4: Effective Mass Tensor

```python
#!/usr/bin/env python3
"""
Compute the full 3x3 effective mass tensor from band curvature.
Uses finite differences on a dense k-grid near the band extremum.
"""
import numpy as np
import json


HBAR_SI = 1.054571817e-34
M_E = 9.1093837015e-31
EV_TO_J = 1.602176634e-19


def effective_mass_tensor(kpoints_cart_inv_m, energies_eV, k0_index):
    """
    Compute 3x3 effective mass tensor via finite-difference
    second derivatives of E(k).

    Parameters
    ----------
    kpoints_cart_inv_m : ndarray (nk, 3)
        K-points in Cartesian coordinates, units 1/m.
    energies_eV : ndarray (nk,)
        Band energies at each k-point, in eV.
    k0_index : int
        Index of the extremum k-point.

    Returns
    -------
    mstar_tensor : ndarray (3,3)
        Effective mass tensor in units of m_e.
    principal_masses : ndarray (3,)
        Eigenvalues (principal effective masses) in m_e.
    principal_directions : ndarray (3,3)
        Eigenvectors (principal directions).
    """
    k0 = kpoints_cart_inv_m[k0_index]
    E0 = energies_eV[k0_index]

    # Build inverse mass tensor (1/m*)_ij = (1/hbar^2) * d^2E/dk_i dk_j
    # Use least-squares fit: E(k) = E0 + sum_ij A_ij * dk_i * dk_j
    # where A_ij = (hbar^2 / 2) * (1/m*)_ij  (in eV * m^2)

    dk = kpoints_cart_inv_m - k0  # (nk, 3) in 1/m
    dE = (energies_eV - E0) * EV_TO_J  # in Joules

    # Build design matrix for quadratic fit
    # E = E0 + A_xx*dx^2 + A_yy*dy^2 + A_zz*dz^2
    #       + 2*A_xy*dx*dy + 2*A_xz*dx*dz + 2*A_yz*dy*dz
    nk = len(dk)
    X = np.zeros((nk, 6))
    X[:, 0] = dk[:, 0]**2           # A_xx
    X[:, 1] = dk[:, 1]**2           # A_yy
    X[:, 2] = dk[:, 2]**2           # A_zz
    X[:, 3] = 2 * dk[:, 0] * dk[:, 1]  # 2*A_xy
    X[:, 4] = 2 * dk[:, 0] * dk[:, 2]  # 2*A_xz
    X[:, 5] = 2 * dk[:, 1] * dk[:, 2]  # 2*A_yz

    # Exclude the extremum point (dE = 0, dk = 0)
    mask = np.linalg.norm(dk, axis=1) > 1e-10
    X_fit = X[mask]
    dE_fit = dE[mask]

    if len(dE_fit) < 6:
        print("  Need at least 6 k-points (excluding extremum) for tensor fit.")
        return None, None, None

    # Least-squares fit
    A_flat, residuals, rank, sv = np.linalg.lstsq(X_fit, dE_fit, rcond=None)

    # Reconstruct A tensor (J * m^2)
    A_tensor = np.array([
        [A_flat[0], A_flat[3], A_flat[4]],
        [A_flat[3], A_flat[1], A_flat[5]],
        [A_flat[4], A_flat[5], A_flat[2]],
    ])

    # Inverse mass tensor: (1/m*)_ij = 2 * A_ij / hbar^2
    inv_mstar_tensor = 2 * A_tensor / HBAR_SI**2  # in 1/kg

    # Mass tensor: m*_ij in kg
    # Diagonalize inverse mass tensor
    eigenvalues, eigenvectors = np.linalg.eigh(inv_mstar_tensor)

    # Principal masses in m_e
    principal_masses = 1.0 / (eigenvalues * M_E)

    # Full tensor in m_e
    mstar_tensor = np.linalg.inv(inv_mstar_tensor) / M_E

    return mstar_tensor, np.abs(principal_masses), eigenvectors


# ── Example usage ──────────────────────────────────────────────────
print("=" * 60)
print("Effective Mass Tensor Calculation")
print("=" * 60)
print()
print("This requires k-points sampled in multiple directions")
print("around the band extremum. Use the k-grid from Step A1")
print("or a 3D grid centered on the extremum.")
print()
print("After running NSCF with those k-points, parse the")
print("eigenvalues and call effective_mass_tensor() above.")
print()

# Demo with synthetic data (parabolic band at Gamma)
np.random.seed(42)
a_lat = 5.431e-10  # m
recip_a = 2 * np.pi / a_lat

# Generate k-points in a small sphere around Gamma
n_test = 50
dk_cryst = 0.03 * (np.random.rand(n_test, 3) - 0.5)
dk_cart = dk_cryst * recip_a  # 1/m

# Synthetic band: E = 0.5 * (k_x^2/m_x + k_y^2/m_y + k_z^2/m_z)
# with m_x = 0.2 m_e, m_y = 0.2 m_e, m_z = 0.9 m_e (anisotropic)
m_true = np.array([0.2, 0.2, 0.9])
E_test = np.zeros(n_test)
for i in range(n_test):
    for j in range(3):
        E_test[i] += HBAR_SI**2 * dk_cart[i, j]**2 / (
            2 * m_true[j] * M_E) / EV_TO_J

# Add Gamma point
dk_cart_with_gamma = np.vstack([[0, 0, 0], dk_cart])
E_with_gamma = np.concatenate([[0], E_test])

tensor, principals, directions = effective_mass_tensor(
    dk_cart_with_gamma, E_with_gamma, k0_index=0)

if tensor is not None:
    print("Effective mass tensor (m_e):")
    for i in range(3):
        print(f"  {tensor[i,0]:8.4f}  {tensor[i,1]:8.4f}  {tensor[i,2]:8.4f}")

    print(f"\nPrincipal effective masses: "
          f"{principals[0]:.4f}, {principals[1]:.4f}, {principals[2]:.4f} m_e")
    print(f"(True values: {m_true[0]}, {m_true[1]}, {m_true[2]})")

    print("\nPrincipal directions:")
    for i in range(3):
        d = directions[:, i]
        print(f"  m* = {principals[i]:.4f}: "
              f"({d[0]:.4f}, {d[1]:.4f}, {d[2]:.4f})")

    # DOS effective mass
    m_dos = (principals[0] * principals[1] * principals[2])**(1.0/3.0)
    print(f"\nDOS effective mass: m*_DOS = {m_dos:.4f} m_e")

    # Conductivity effective mass
    m_cond = 3.0 / (1.0/principals[0] + 1.0/principals[1] + 1.0/principals[2])
    print(f"Conductivity effective mass: m*_cond = {m_cond:.4f} m_e")
```

---

### Method B: VASP Parabolic Fit

```python
#!/usr/bin/env python3
"""
Extract effective mass from VASP band structure via parabolic fitting.
Parses EIGENVAL and KPOINTS files.
"""
import os
import numpy as np
from scipy.optimize import curve_fit

HBAR_SI = 1.054571817e-34
M_E = 9.1093837015e-31
EV_TO_J = 1.602176634e-19


def parse_vasp_eigenval(filename='EIGENVAL'):
    """Parse VASP EIGENVAL. Returns kpoints, eigenvalues, n_electrons."""
    with open(filename, 'r') as f:
        lines = f.readlines()

    header = lines[5].split()
    n_electrons = int(float(header[0]))
    nkpt = int(header[1])
    nbands = int(header[2])

    kpoints = []
    eigenvalues = []

    idx = 7
    for ik in range(nkpt):
        kpt_line = lines[idx].split()
        kpoints.append([float(kpt_line[0]), float(kpt_line[1]),
                         float(kpt_line[2])])
        idx += 1

        eigs = []
        for ib in range(nbands):
            vals = lines[idx].split()
            eigs.append(float(vals[1]))
            idx += 1

        eigenvalues.append(eigs)
        idx += 1

    return np.array(kpoints), np.array(eigenvalues), n_electrons


def vasp_effective_mass(eigenval_file='EIGENVAL', poscar_file='POSCAR',
                         band_type='electron'):
    """
    Compute effective mass from VASP band structure.

    Parameters
    ----------
    band_type : str
        'electron' for CBM, 'hole' for VBM.
    """
    from pymatgen.core import Structure

    kpoints, eigenvalues, n_elec = parse_vasp_eigenval(eigenval_file)
    n_occ = int(n_elec / 2)

    if band_type == 'electron':
        band_idx = n_occ  # First unoccupied
        is_valence = False
    else:
        band_idx = n_occ - 1  # Last occupied
        is_valence = True

    # Get reciprocal lattice
    struct = Structure.from_file(poscar_file)
    recip = struct.lattice.reciprocal_lattice.matrix  # 1/Angstrom

    # Convert k-points to Cartesian
    k_cart_angstrom = kpoints @ recip  # (nk, 3) in 1/Angstrom
    k_cart_m = k_cart_angstrom * 1e10  # 1/m

    # Find extremum
    band = eigenvalues[:, band_idx]
    if is_valence:
        ext_idx = np.argmax(band)
    else:
        ext_idx = np.argmin(band)

    print(f"\n{band_type.capitalize()} band (index {band_idx}):")
    print(f"  Extremum at k-index {ext_idx}: "
          f"k = ({kpoints[ext_idx][0]:.4f}, {kpoints[ext_idx][1]:.4f}, "
          f"{kpoints[ext_idx][2]:.4f})")
    print(f"  Energy: {band[ext_idx]:.4f} eV")

    # Fit parabola using nearby k-points
    k0 = k_cart_m[ext_idx]
    dk = np.linalg.norm(k_cart_m - k0, axis=1)

    # Select points within a small radius
    radius = 0.1 * np.max(dk)
    nearby = np.where((dk < radius) & (dk > 0))[0]

    if len(nearby) < 3:
        print("  Not enough nearby k-points for fit.")
        return None

    dk_nearby = dk[nearby]
    E_nearby = band[nearby]

    # Fit: E = E0 + alpha * dk^2
    E0 = band[ext_idx]
    dE = E_nearby - E0

    alpha = np.polyfit(dk_nearby, dE, 2)[0]  # eV * m^2
    alpha_SI = alpha * EV_TO_J

    mstar = HBAR_SI**2 / (2 * alpha_SI * M_E)
    mstar = abs(mstar)

    print(f"  m* = {mstar:.4f} m_e (isotropic average near extremum)")

    return mstar


# ── Example ────────────────────────────────────────────────────────
if os.path.exists('EIGENVAL') and os.path.exists('POSCAR'):
    for bt in ['electron', 'hole']:
        vasp_effective_mass('EIGENVAL', 'POSCAR', band_type=bt)
```

---

### Method D: HOMO/LUMO Effective Mass for Molecular Crystals

```python
#!/usr/bin/env python3
"""
Compute HOMO/LUMO effective masses for molecular crystals.
Same approach as for inorganic semiconductors, but applied to
the HOMO (highest occupied molecular orbital band) and LUMO
(lowest unoccupied molecular orbital band).

Molecular crystals often have very flat bands (large m*), indicating
narrow bandwidth and poor transport. Typical values: m* = 1-10 m_e.
"""
import numpy as np
import json

# The physics is identical to Method A. The key differences are:
# 1. HOMO = VBM, LUMO = CBM in molecular crystal terminology
# 2. Bands are often very flat (m* >> 1)
# 3. The band structure may have many bands close in energy

# Use the same parabolic fitting code from Method A.
# Simply identify the correct band indices:

def molecular_crystal_masses(eigenvalues, kpoints, lattice_matrix,
                              n_homo, directions):
    """
    Compute HOMO and LUMO effective masses.

    Parameters
    ----------
    eigenvalues : ndarray (nk, nbnd) in eV
    kpoints : list of k-points in crystal coords
    lattice_matrix : ndarray (3,3) in Angstrom
    n_homo : int
        Index of HOMO band (0-based). LUMO = n_homo + 1.
    directions : list of (name, direction_vector)
    """
    print("=" * 50)
    print("Molecular Crystal Effective Masses")
    print("=" * 50)

    # HOMO effective mass (hole mass)
    homo_band = eigenvalues[:, n_homo]
    homo_max_k = np.argmax(homo_band)
    print(f"\nHOMO band (index {n_homo}):")
    print(f"  Maximum at k-index {homo_max_k}: {homo_band[homo_max_k]:.4f} eV")

    # LUMO effective mass (electron mass)
    lumo_band = eigenvalues[:, n_homo + 1]
    lumo_min_k = np.argmin(lumo_band)
    print(f"\nLUMO band (index {n_homo + 1}):")
    print(f"  Minimum at k-index {lumo_min_k}: {lumo_band[lumo_min_k]:.4f} eV")

    # HOMO-LUMO gap
    gap = np.min(lumo_band) - np.max(homo_band)
    print(f"\nHOMO-LUMO gap: {gap:.4f} eV")

    # Bandwidth (indicator of mobility)
    homo_bw = np.max(homo_band) - np.min(homo_band)
    lumo_bw = np.max(lumo_band) - np.min(lumo_band)
    print(f"HOMO bandwidth: {homo_bw:.4f} eV "
          f"({'narrow' if homo_bw < 0.5 else 'moderate'})")
    print(f"LUMO bandwidth: {lumo_bw:.4f} eV "
          f"({'narrow' if lumo_bw < 0.5 else 'moderate'})")

    # For molecular crystals with very flat bands,
    # the effective mass is large (m* >> m_e)
    # and the mobility is low (mu ~ 1/m*)
    print("\nNote: Molecular crystals with bandwidth < 0.1 eV")
    print("have m* > 5 m_e and very low band transport mobility.")
    print("In this regime, hopping transport may dominate.")


# Usage follows the same pattern as Method A with appropriate
# band indices for HOMO and LUMO.
print("Use Method A workflow with HOMO/LUMO band indices.")
print("Set is_valence=True for HOMO, False for LUMO.")
```

---

## Key Parameters

| Parameter | Typical Value | Notes |
|---|---|---|
| dk_range | 0.02-0.05 (crystal coords) | K-point range for fitting. Too large = non-parabolic. Too small = noise. |
| n_points_per_direction | 11-21 | More points = better fit, but more expensive |
| Fitting range | 5-10% of BZ | Should be small enough for parabolic approximation |
| Band index | VBM = n_occ-1, CBM = n_occ | 0-indexed. Verify with Fermi energy. |
| ecutwfc (QE) | 40-80 Ry | Must be converged for band curvature |
| ENCUT (VASP) | 300-500 eV | Same convergence requirement |
| K-grid for tensor | >= 20 points in 3D | Need points in multiple directions for 3x3 tensor |

### Effective Mass Types

| Type | Definition | Use |
|---|---|---|
| Band effective mass | m* from d^2E/dk^2 at extremum | Single-direction transport |
| DOS effective mass | (m_1 * m_2 * m_3)^(1/3) | Carrier concentration, Seebeck |
| Conductivity effective mass | 3 / (1/m_1 + 1/m_2 + 1/m_3) | Electrical conductivity |
| Transport effective mass | from BoltzTraP2 | Temperature-dependent transport |

## Interpreting Results

### Fit Quality

- **R^2 > 0.99**: Excellent parabolic fit. Band is truly parabolic in the fitted range.
- **R^2 > 0.95**: Acceptable fit. Some non-parabolicity.
- **R^2 < 0.90**: Poor fit. Band is significantly non-parabolic. Try reducing the fitting range (dk_range) or use BoltzTraP2 instead.

### Physical Interpretation

- **m* < 0.1 m_e**: Very light carriers (e.g., GaAs electrons at Gamma). High mobility.
- **m* ~ 0.1-0.5 m_e**: Light carriers (e.g., Si transverse electrons).
- **m* ~ 0.5-2.0 m_e**: Moderate mass (e.g., Si heavy holes, ZnO electrons).
- **m* > 2 m_e**: Heavy carriers. Low band mobility. May indicate flat band or narrow bandwidth.
- **m* >> 5 m_e**: Extremely heavy. Hopping transport likely dominates over band transport.

### Anisotropy

- **Isotropic**: m* same in all directions (cubic semiconductors at Gamma).
- **Anisotropic**: Different m* along different axes (e.g., Si CBM: m_l = 0.98, m_t = 0.19).
- **Strongly anisotropic**: Layered materials with very different in-plane vs out-of-plane masses.

## Common Issues

| Problem | Solution |
|---|---|
| **m* is unreasonable (too large or negative)** | Reduce dk_range -- fitting range extends into non-parabolic region. Or the wrong band is being fit. |
| **Poor R^2** | Band is non-parabolic. Reduce dk_range. For heavily non-parabolic bands, use BoltzTraP2 transport mass instead. |
| **m* changes with dk_range** | Intrinsic non-parabolicity. Report m* with the range used. For transport, BoltzTraP2 is more appropriate. |
| **Cannot find extremum** | VBM/CBM may be between sampled k-points. Use denser k-mesh or interpolate. |
| **Degenerate bands at VBM** | Common in cubic semiconductors (heavy + light holes). Fit each band separately. |
| **Spin-orbit splitting** | SOC splits degenerate bands. Include `lspinorb = .true.` in QE for heavy elements. |
| **HOMO/LUMO too flat** | Molecular crystal with narrow bandwidth. m* is large by definition. Consider hopping transport model. |
| **Different m* from different methods** | Parabolic fit, BoltzTraP2, and sumo may give different values due to different averaging. Document the method used. |
