# Angular-Dependent Effective Mass and Fermi Velocity

## When to Use

- You are studying anisotropic semiconductors where the effective mass varies significantly with crystallographic direction (e.g., Si, Ge, black phosphorus, SnSe).
- You need valley-dependent transport properties: different carrier pockets at inequivalent k-points have different angular mass profiles.
- You are characterizing Dirac or Weyl materials with tilted cones, where the Fermi velocity is direction-dependent.
- You need direction-dependent carrier mobility for device simulation (mobility ~ 1/m*(theta)).
- You want to visualize the angular anisotropy of effective mass or Fermi velocity as a polar plot.
- You need to extract the full effective mass tensor eigenvalues and principal directions from angular sampling.
- You are comparing in-plane vs out-of-plane transport in layered materials (e.g., MoS2, graphite, Bi2Se3).
- Equivalent to VASPKIT task 914 (Angular-Dependent Effective-Mass and Fermi-Velocity).

## Method Selection

| Criterion | QE DFT | VASP |
|---|---|---|
| Availability | pw.x pre-installed in container | Requires external VASP license |
| Band sampling | NSCF with custom k-point ring | EIGENVAL from band calculation |
| Fitting control | Full Python, arbitrary angular resolution | Post-processing of existing data |
| Fermi velocity | Computed from band slope dE/dk(theta) | Parsed from EIGENVAL, same analysis |
| Polar plot | Matplotlib polar (included) | Same |
| Tensor extraction | Least-squares fit of m*(theta) to tensor | Same |
| Best for | Production runs, full control | Re-analyzing existing VASP data |

## Prerequisites

- A completed SCF calculation (charge density available) for the system.
- Knowledge of the band extremum location in k-space (VBM/CBM k-point).
- For metallic systems: knowledge of where the band crosses the Fermi level.
- Python packages: `numpy`, `scipy`, `matplotlib` (pre-installed).
- For QE: `pw.x` in `/opt/qe/bin/`.
- For VASP: EIGENVAL, POSCAR, and KPOINTS files from a completed calculation.

## Background

### Angular Effective Mass

The effective mass tensor is a 3x3 symmetric matrix:

```
(1/m*)_ij = (1/hbar^2) * d^2E / dk_i dk_j
```

In general, this tensor is anisotropic. The effective mass along an arbitrary direction n_hat is:

```
1/m*(n_hat) = sum_ij n_i * (1/m*)_ij * n_j
```

where n_hat = (cos(theta)*sin(phi), sin(theta)*sin(phi), cos(phi)) in spherical coordinates. For a 2D analysis in a specific plane, we parameterize the direction by a single angle theta.

### Angular Fermi Velocity

For metallic or semimetallic bands crossing the Fermi level, the Fermi velocity along direction theta is:

```
v_F(theta) = (1/hbar) * |dE/dk|_{theta}
```

This is important for Dirac/Weyl materials where the cone may be tilted or anisotropic, leading to direction-dependent transport.

### Typical Anisotropies

| Material | Feature | Anisotropy Ratio |
|---|---|---|
| Si CBM (Delta valley) | m_l / m_t | ~5.2 (0.98/0.19) |
| Black phosphorus | m_armchair / m_zigzag | ~6-10 |
| SnSe (valence band) | m_a / m_b | ~3-5 |
| Bi2Se3 surface state | v_F anisotropy | ~1.0 (nearly isotropic) |
| Type-II Weyl semimetal | v_F(theta) | Diverges along tilt direction |
| Graphene Dirac cone | v_F anisotropy | 1.0 (isotropic) |

---

## Detailed Steps

### Overview

```
Method A: QE DFT -- full angular m*(theta) and v_F(theta)
  Step 1: SCF ground state
  Step 2: Generate k-points on circles at various angles around extremum
  Step 3: Run NSCF with angular k-points
  Step 4: Extract E(k) on each angular ring, fit m*(theta)
  Step 5: Polar plot of 1/m*(theta)
  Step 6: Extract effective mass tensor eigenvalues from angular data
  Step 7: Compute angle-dependent Fermi velocity v_F(theta) for metallic bands

Method B: VASP -- post-process EIGENVAL for angular m* and v_F
  Step 1: Parse EIGENVAL for k-resolved bands
  Step 2: Compute angular m*(theta) and v_F(theta)
  Step 3: Reference VASPKIT 914
```

---

### Method A: QE DFT

#### Step A1: SCF Ground State

```python
#!/usr/bin/env python3
"""
Run SCF ground state calculation for angular effective mass analysis.
The charge density from this step is used in subsequent NSCF calculations.
"""
import os
import subprocess

PSEUDO_DIR = os.path.abspath("./pseudo")
OUTDIR = os.path.abspath("./tmp_angular")
os.makedirs(OUTDIR, exist_ok=True)
os.makedirs(PSEUDO_DIR, exist_ok=True)

# ── Download pseudopotential if needed ─────────────────────────
pseudo_file = os.path.join(PSEUDO_DIR, "Si.pbe-n-kjpaw_psl.1.0.0.UPF")
if not os.path.exists(pseudo_file):
    url = ("https://pseudopotentials.quantum-espresso.org/upf_files/"
           "Si.pbe-n-kjpaw_psl.1.0.0.UPF")
    subprocess.run(["wget", "-q", "-O", pseudo_file, url], check=True)
    print(f"Downloaded: {pseudo_file}")

# ── SCF input ──────────────────────────────────────────────────
scf_input = f"""&CONTROL
    calculation   = 'scf'
    prefix        = 'si'
    pseudo_dir    = '{PSEUDO_DIR}'
    outdir        = '{OUTDIR}'
    tprnfor       = .true.
    tstress       = .true.
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
/

&ELECTRONS
    conv_thr      = 1.0d-10
    mixing_beta   = 0.7
/

ATOMIC_SPECIES
  Si  28.0855  Si.pbe-n-kjpaw_psl.1.0.0.UPF

ATOMIC_POSITIONS {{crystal}}
  Si  0.00  0.00  0.00
  Si  0.25  0.25  0.25

K_POINTS {{automatic}}
  12 12 12  0 0 0
"""

with open("scf_angular.in", "w") as f:
    f.write(scf_input)
print("Written: scf_angular.in")

print("Running SCF ...")
result = subprocess.run(
    ["mpirun", "-np", "4", "pw.x", "-in", "scf_angular.in"],
    capture_output=True, text=True, timeout=3600
)
with open("scf_angular.out", "w") as f:
    f.write(result.stdout)

if result.returncode != 0:
    print("ERROR: SCF failed!")
    print(result.stderr[-500:] if result.stderr else "")
else:
    # Check convergence
    converged = "convergence has been achieved" in result.stdout
    print(f"SCF {'converged' if converged else 'FAILED to converge'}.")
    print("Output: scf_angular.out")
```

#### Step A2: Generate K-Points on Angular Rings

```python
#!/usr/bin/env python3
"""
Generate k-points arranged in circles (rings) at various angles theta
around a band extremum. Each ring samples the energy surface E(k) at
a fixed |dk| but varying angle, enabling extraction of m*(theta).

The k-points lie in a chosen crystallographic plane (e.g., kx-ky, kx-kz)
around the extremum.
"""
import numpy as np
import os


def generate_angular_kpoints(k_extremum, plane_vectors, n_angles=72,
                              dk_radii=None, n_radial=5, dk_max=0.04):
    """
    Generate k-points on concentric rings around a band extremum.

    Parameters
    ----------
    k_extremum : array (3,)
        K-point of the extremum in crystal coordinates.
    plane_vectors : tuple of two arrays (3,)
        Two orthogonal direction vectors spanning the sampling plane
        in crystal coordinates. These define the angular reference:
        theta=0 is along plane_vectors[0].
    n_angles : int
        Number of angular samples per ring (angular resolution = 360/n_angles degrees).
    dk_radii : list of float or None
        Radii of the rings in crystal coordinate units.
        If None, uses n_radial equally spaced radii up to dk_max.
    n_radial : int
        Number of radial rings (used if dk_radii is None).
    dk_max : float
        Maximum ring radius in crystal coordinates.

    Returns
    -------
    kpoints : ndarray (N, 3)
        K-points in crystal coordinates.
    angles : ndarray (N,)
        Angle theta in radians for each k-point.
    radii : ndarray (N,)
        Radius |dk| for each k-point.
    ring_indices : list of lists
        Indices grouping k-points by ring.
    """
    if dk_radii is None:
        dk_radii = np.linspace(dk_max / n_radial, dk_max, n_radial)

    # Normalize plane vectors
    v1 = np.array(plane_vectors[0], dtype=float)
    v2 = np.array(plane_vectors[1], dtype=float)
    v1 = v1 / np.linalg.norm(v1)
    v2 = v2 / np.linalg.norm(v2)

    # Ensure orthogonality (Gram-Schmidt)
    v2 = v2 - np.dot(v2, v1) * v1
    v2 = v2 / np.linalg.norm(v2)

    k0 = np.array(k_extremum, dtype=float)
    theta_values = np.linspace(0, 2 * np.pi, n_angles, endpoint=False)

    kpoints = []
    angles = []
    radii = []
    ring_indices = []

    # Add the extremum point itself
    kpoints.append(k0.copy())
    angles.append(0.0)
    radii.append(0.0)

    for ir, r in enumerate(dk_radii):
        ring_idx = []
        for theta in theta_values:
            dk = r * (np.cos(theta) * v1 + np.sin(theta) * v2)
            k = k0 + dk
            kpoints.append(k)
            angles.append(theta)
            radii.append(r)
            ring_idx.append(len(kpoints) - 1)
        ring_indices.append(ring_idx)

    return (np.array(kpoints), np.array(angles),
            np.array(radii), ring_indices)


def write_qe_kpoints_angular(kpoints, filename='kpoints_angular.txt'):
    """Write k-points in QE crystal format for NSCF."""
    nk = len(kpoints)
    lines = ["K_POINTS {crystal}", f"{nk}"]
    w = 1.0 / nk
    for k in kpoints:
        lines.append(f"  {k[0]:14.10f}  {k[1]:14.10f}  "
                     f"{k[2]:14.10f}  {w:.10f}")
    with open(filename, 'w') as f:
        f.write("\n".join(lines) + "\n")
    print(f"Written: {filename} ({nk} k-points)")
    return filename


# ── Example: Si VBM at Gamma, sampling in kx-ky plane ─────────
# VBM at Gamma, sample in the (100)-(010) plane
k_extremum_vbm = [0.0, 0.0, 0.0]
plane_v1 = [1.0, 0.0, 0.0]  # kx direction
plane_v2 = [0.0, 1.0, 0.0]  # ky direction

kpoints_vbm, angles_vbm, radii_vbm, rings_vbm = generate_angular_kpoints(
    k_extremum_vbm,
    plane_vectors=(plane_v1, plane_v2),
    n_angles=72,        # 5-degree resolution
    n_radial=5,          # 5 concentric rings
    dk_max=0.04,         # max radius in crystal coords
)

# CBM near X for Si -- sample in the kx-ky plane around the valley
k_extremum_cbm = [0.425, 0.425, 0.0]
cbm_v1 = [1.0, 1.0, 0.0]    # longitudinal (along Delta)
cbm_v2 = [1.0, -1.0, 0.0]   # transverse

kpoints_cbm, angles_cbm, radii_cbm, rings_cbm = generate_angular_kpoints(
    k_extremum_cbm,
    plane_vectors=(cbm_v1, cbm_v2),
    n_angles=72,
    n_radial=5,
    dk_max=0.03,
)

# Combine all k-points
all_kpoints = np.vstack([kpoints_vbm, kpoints_cbm])
write_qe_kpoints_angular(all_kpoints, 'kpoints_angular.txt')

n_vbm = len(kpoints_vbm)
n_cbm = len(kpoints_cbm)
print(f"\nVBM angular k-points: {n_vbm} "
      f"({len(rings_vbm)} rings x 72 angles + 1 center)")
print(f"CBM angular k-points: {n_cbm}")
print(f"Total: {len(all_kpoints)} k-points")

# Save metadata for later parsing
np.savez('angular_kpoints_meta.npz',
         angles_vbm=angles_vbm, radii_vbm=radii_vbm,
         angles_cbm=angles_cbm, radii_cbm=radii_cbm,
         n_vbm=n_vbm, n_cbm=n_cbm,
         k_extremum_vbm=k_extremum_vbm,
         k_extremum_cbm=k_extremum_cbm)
print("Saved: angular_kpoints_meta.npz")
```

#### Step A3: Run NSCF with Angular K-Points

```python
#!/usr/bin/env python3
"""
Run QE NSCF calculation with angular k-points for m*(theta) extraction.
Requires SCF charge density from Step A1.
"""
import os
import subprocess
import shutil

PSEUDO_DIR = os.path.abspath("./pseudo")
SCF_OUTDIR = os.path.abspath("./tmp_angular")
NSCF_OUTDIR = os.path.abspath("./tmp_angular_nscf")

# Copy charge density from SCF
if os.path.exists(os.path.join(SCF_OUTDIR, "si.save")):
    os.makedirs(NSCF_OUTDIR, exist_ok=True)
    save_src = os.path.join(SCF_OUTDIR, "si.save")
    save_dst = os.path.join(NSCF_OUTDIR, "si.save")
    if not os.path.exists(save_dst):
        shutil.copytree(save_src, save_dst)
        print(f"Copied charge density: {save_src} -> {save_dst}")
else:
    print("ERROR: SCF save directory not found. Run Step A1 first.")
    raise FileNotFoundError("SCF not completed")

# Read angular k-points
with open("kpoints_angular.txt", "r") as f:
    kpoints_block = f.read()

nscf_input = f"""&CONTROL
    calculation   = 'nscf'
    prefix        = 'si'
    pseudo_dir    = '{PSEUDO_DIR}'
    outdir        = '{NSCF_OUTDIR}'
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

with open("nscf_angular.in", "w") as f:
    f.write(nscf_input)
print("Written: nscf_angular.in")

print("Running NSCF with angular k-points ...")
result = subprocess.run(
    ["mpirun", "-np", "4", "pw.x", "-in", "nscf_angular.in"],
    capture_output=True, text=True, timeout=7200
)
with open("nscf_angular.out", "w") as f:
    f.write(result.stdout)

if result.returncode != 0:
    print("ERROR: NSCF failed!")
    print(result.stderr[-500:] if result.stderr else "")
else:
    print("NSCF completed. Output: nscf_angular.out")
```

#### Step A4: Extract Angular Effective Mass m*(theta)

```python
#!/usr/bin/env python3
"""
Extract angular-dependent effective mass m*(theta) from QE NSCF output.

Workflow:
1. Parse eigenvalues from NSCF output
2. For each angular ring, extract E(theta) at fixed |dk|
3. Fit parabolic dispersion along radial direction for each angle
4. Compute m*(theta) = hbar^2 / (d^2E/dk^2) for each direction
5. Generate polar plot of 1/m*(theta)
"""
import re
import os
import json
import numpy as np
from scipy.optimize import curve_fit
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# ── Physical constants ─────────────────────────────────────────
HBAR_SI = 1.054571817e-34       # J*s
M_E = 9.1093837015e-31          # kg
EV_TO_J = 1.602176634e-19       # J/eV
BOHR_TO_M = 5.29177210903e-11   # m


def parse_nscf_eigenvalues(filename):
    """
    Parse k-points and eigenvalues from QE pw.x NSCF output.

    Returns
    -------
    kpoints : list of ndarray (3,) in crystal coordinates
    eigenvalues : ndarray (nk, nbnd) in eV
    fermi_energy : float in eV (or None)
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


def compute_angular_effective_mass(kpoints_cryst, eigenvalues,
                                    band_index, k_extremum,
                                    plane_vectors, angles, radii,
                                    ring_indices, lattice_matrix_ang,
                                    is_valence=False, n_angles=72,
                                    offset=0):
    """
    Compute angular-dependent effective mass m*(theta).

    For each angle theta, collect the energies at different radii along
    that direction and fit a parabola E(k) = E0 + hbar^2*k^2/(2*m*).

    Parameters
    ----------
    kpoints_cryst : ndarray (nk, 3)
        K-points in crystal coordinates.
    eigenvalues : ndarray (nk, nbnd)
        Eigenvalues in eV.
    band_index : int
        Band index (0-based).
    k_extremum : array (3,)
        Extremum k-point in crystal coordinates.
    plane_vectors : tuple of two arrays
        Plane vectors defining the angular sampling plane.
    angles : ndarray
        Angle for each k-point (radians).
    radii : ndarray
        Radius for each k-point (crystal coords).
    ring_indices : list of lists
        Indices for each ring.
    lattice_matrix_ang : ndarray (3,3)
        Real-space lattice matrix in Angstrom.
    is_valence : bool
        True for valence band (maximum).
    n_angles : int
        Number of angular samples per ring.
    offset : int
        Index offset into kpoints_cryst for this set of angular k-points.

    Returns
    -------
    theta_values : ndarray (n_angles,)
        Angles in radians.
    mstar_theta : ndarray (n_angles,)
        Effective mass m*/m_e for each angle.
    r_squared : ndarray (n_angles,)
        R^2 of parabolic fit for each angle.
    """
    # Reciprocal lattice vectors (1/Angstrom, with 2*pi)
    recip_matrix = 2 * np.pi * np.linalg.inv(lattice_matrix_ang).T

    # Convert all k-points to Cartesian (1/m)
    k_cart = np.array([recip_matrix @ k for k in kpoints_cryst])
    k_cart_m = k_cart * 1e10  # 1/Angstrom -> 1/m

    # Energy at the extremum (center point, index = offset)
    E0 = eigenvalues[offset, band_index]
    k0_cart = k_cart_m[offset]

    # Number of radial rings
    n_rings = len(ring_indices)

    theta_values = np.linspace(0, 2 * np.pi, n_angles, endpoint=False)
    mstar_theta = np.full(n_angles, np.nan)
    r_squared_arr = np.full(n_angles, np.nan)

    for i_angle in range(n_angles):
        # Collect radial data for this angle
        dk_list = [0.0]  # center point at dk=0
        E_list = [E0]

        for i_ring, ring_idx in enumerate(ring_indices):
            # Each ring has n_angles points; index i_angle within ring
            global_idx = ring_idx[i_angle]
            k_this = k_cart_m[global_idx]
            dk_mag = np.linalg.norm(k_this - k0_cart)
            E_this = eigenvalues[global_idx, band_index]

            if not np.isnan(E_this):
                dk_list.append(dk_mag)
                E_list.append(E_this)

        dk_arr = np.array(dk_list)
        E_arr = np.array(E_list)

        if len(dk_arr) < 3:
            continue

        # Fit parabola: E = E0 + alpha * dk^2
        # Using least squares: dE = alpha * dk^2
        dE = E_arr - E0
        dk2 = dk_arr ** 2

        # Exclude zero point for fitting (it's exactly zero)
        mask = dk_arr > 0
        if np.sum(mask) < 2:
            continue

        # Weighted least squares fit: dE = alpha * dk^2
        A = dk2[mask].reshape(-1, 1)
        b = dE[mask]
        alpha, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
        alpha = alpha[0]  # eV * m^2

        if abs(alpha) < 1e-40:
            continue

        # m* = hbar^2 / (2 * alpha * eV_to_J)
        alpha_SI = alpha * EV_TO_J  # J * m^2
        mstar_kg = HBAR_SI ** 2 / (2 * alpha_SI)
        mstar_me = mstar_kg / M_E

        if is_valence:
            mstar_me = abs(mstar_me)

        # R^2
        dE_fit = alpha * dk2
        ss_res = np.sum((dE - dE_fit) ** 2)
        ss_tot = np.sum((dE - np.mean(dE)) ** 2)
        r2 = 1 - ss_res / ss_tot if ss_tot > 1e-30 else 0.0

        mstar_theta[i_angle] = abs(mstar_me)
        r_squared_arr[i_angle] = r2

    return theta_values, mstar_theta, r_squared_arr


# ── Main workflow ──────────────────────────────────────────────
print("=" * 65)
print("Angular-Dependent Effective Mass from Band Curvature")
print("=" * 65)

output_file = "nscf_angular.out"
meta_file = "angular_kpoints_meta.npz"

if not os.path.exists(output_file):
    print(f"Output file not found: {output_file}")
    print("Run Step A3 first.")
elif not os.path.exists(meta_file):
    print(f"Metadata file not found: {meta_file}")
    print("Run Step A2 first.")
else:
    kpoints, eigenvalues, fermi_e = parse_nscf_eigenvalues(output_file)
    meta = np.load(meta_file, allow_pickle=True)

    print(f"\nParsed: {len(kpoints)} k-points, "
          f"{eigenvalues.shape[1]} bands")
    if fermi_e is not None:
        print(f"Fermi energy: {fermi_e:.4f} eV")

    n_vbm = int(meta['n_vbm'])
    n_cbm = int(meta['n_cbm'])
    n_angles = 72

    # Si lattice (ibrav=2 primitive cell)
    a_si = 5.431  # Angstrom
    lattice = np.array([
        [-a_si / 2, 0, a_si / 2],
        [0, a_si / 2, a_si / 2],
        [-a_si / 2, a_si / 2, 0],
    ])

    n_occ = 4  # Si: 4 valence bands

    # ── VBM angular effective mass ──────────────────────────────
    # Reconstruct ring_indices for VBM
    n_radial = 5
    vbm_ring_indices = []
    for ir in range(n_radial):
        start = 1 + ir * n_angles  # skip center point
        vbm_ring_indices.append(list(range(start, start + n_angles)))

    angles_vbm = meta['angles_vbm']
    radii_vbm = meta['radii_vbm']

    kpoints_arr = np.array([k for k in kpoints])

    print(f"\n--- VBM Angular Effective Mass (band {n_occ - 1}) ---")
    theta_vbm, mstar_vbm, r2_vbm = compute_angular_effective_mass(
        kpoints_arr[:n_vbm], eigenvalues[:n_vbm],
        band_index=n_occ - 1,
        k_extremum=[0, 0, 0],
        plane_vectors=([1, 0, 0], [0, 1, 0]),
        angles=angles_vbm, radii=radii_vbm,
        ring_indices=vbm_ring_indices,
        lattice_matrix_ang=lattice,
        is_valence=True,
        n_angles=n_angles,
        offset=0,
    )

    valid = ~np.isnan(mstar_vbm)
    print(f"  Valid angles: {np.sum(valid)}/{n_angles}")
    if np.any(valid):
        print(f"  m* range: {np.nanmin(mstar_vbm):.4f} -- "
              f"{np.nanmax(mstar_vbm):.4f} m_e")
        print(f"  m* mean: {np.nanmean(mstar_vbm):.4f} m_e")
        print(f"  Anisotropy ratio: "
              f"{np.nanmax(mstar_vbm)/np.nanmin(mstar_vbm):.2f}")

    # ── CBM angular effective mass ──────────────────────────────
    cbm_ring_indices = []
    for ir in range(n_radial):
        start = 1 + ir * n_angles
        cbm_ring_indices.append(list(range(start, start + n_angles)))

    angles_cbm = meta['angles_cbm']
    radii_cbm = meta['radii_cbm']

    print(f"\n--- CBM Angular Effective Mass (band {n_occ}) ---")
    theta_cbm, mstar_cbm, r2_cbm = compute_angular_effective_mass(
        kpoints_arr[n_vbm:n_vbm + n_cbm],
        eigenvalues[n_vbm:n_vbm + n_cbm],
        band_index=n_occ,
        k_extremum=[0.425, 0.425, 0.0],
        plane_vectors=([1, 1, 0], [1, -1, 0]),
        angles=angles_cbm, radii=radii_cbm,
        ring_indices=cbm_ring_indices,
        lattice_matrix_ang=lattice,
        is_valence=False,
        n_angles=n_angles,
        offset=0,
    )

    valid_cbm = ~np.isnan(mstar_cbm)
    if np.any(valid_cbm):
        print(f"  Valid angles: {np.sum(valid_cbm)}/{n_angles}")
        print(f"  m* range: {np.nanmin(mstar_cbm):.4f} -- "
              f"{np.nanmax(mstar_cbm):.4f} m_e")
        print(f"  Anisotropy ratio: "
              f"{np.nanmax(mstar_cbm)/np.nanmin(mstar_cbm):.2f}")

    # Save results
    results = {
        'theta_vbm_deg': (theta_vbm * 180 / np.pi).tolist(),
        'mstar_vbm': [float(x) if not np.isnan(x) else None
                      for x in mstar_vbm],
        'r2_vbm': [float(x) if not np.isnan(x) else None
                   for x in r2_vbm],
        'theta_cbm_deg': (theta_cbm * 180 / np.pi).tolist(),
        'mstar_cbm': [float(x) if not np.isnan(x) else None
                      for x in mstar_cbm],
        'r2_cbm': [float(x) if not np.isnan(x) else None
                   for x in r2_cbm],
    }
    with open('angular_mstar_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    print("\nResults saved: angular_mstar_results.json")
```

#### Step A5: Polar Plot of 1/m*(theta)

```python
#!/usr/bin/env python3
"""
Generate polar plots of the angular-dependent effective mass.
Plots 1/m*(theta) so that lighter carriers appear as larger lobes,
making the anisotropy visually intuitive.
"""
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def polar_plot_effective_mass(theta, mstar, title, filename,
                               plot_inverse=True):
    """
    Create a polar plot of angular effective mass.

    Parameters
    ----------
    theta : ndarray (n,)
        Angles in radians.
    mstar : ndarray (n,)
        Effective mass m*/m_e at each angle.
    title : str
        Plot title.
    filename : str
        Output filename.
    plot_inverse : bool
        If True, plot 1/m* (lighter = larger lobe). If False, plot m*.
    """
    valid = ~np.isnan(mstar) & (mstar > 0)
    theta_v = theta[valid]
    mstar_v = mstar[valid]

    if len(theta_v) == 0:
        print(f"  No valid data for {title}")
        return

    # Close the polar curve
    theta_plot = np.append(theta_v, theta_v[0])
    if plot_inverse:
        r_plot = np.append(1.0 / mstar_v, 1.0 / mstar_v[0])
        r_label = "1/m* (1/m_e)"
    else:
        r_plot = np.append(mstar_v, mstar_v[0])
        r_label = "m* (m_e)"

    fig, ax = plt.subplots(1, 1, figsize=(7, 7),
                           subplot_kw={'projection': 'polar'})

    ax.plot(theta_plot, r_plot, 'b-', linewidth=2.0)
    ax.fill(theta_plot, r_plot, alpha=0.15, color='steelblue')

    # Mark cardinal directions
    for angle_deg, label in [(0, '0 deg'), (90, '90 deg'),
                              (180, '180 deg'), (270, '270 deg')]:
        angle_rad = np.deg2rad(angle_deg)
        idx = np.argmin(np.abs(theta_v - angle_rad))
        val = 1.0 / mstar_v[idx] if plot_inverse else mstar_v[idx]
        ax.plot(angle_rad, val, 'ro', markersize=6)
        ax.annotate(f'm*={mstar_v[idx]:.3f}',
                    xy=(angle_rad, val),
                    fontsize=8,
                    textcoords='offset points',
                    xytext=(10, 5))

    ax.set_title(title, fontsize=14, pad=20)
    ax.set_rlabel_position(45)

    fig.tight_layout()
    fig.savefig(filename, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {filename}")


def polar_plot_comparison(theta_list, mstar_list, labels, title,
                           filename):
    """
    Overlay multiple angular effective mass curves on one polar plot.
    """
    fig, ax = plt.subplots(1, 1, figsize=(8, 8),
                           subplot_kw={'projection': 'polar'})

    colors = ['steelblue', 'coral', 'seagreen', 'darkorange', 'purple']

    for i, (theta, mstar, label) in enumerate(zip(theta_list,
                                                    mstar_list, labels)):
        valid = ~np.isnan(mstar) & (mstar > 0)
        if not np.any(valid):
            continue

        theta_v = theta[valid]
        inv_mstar = 1.0 / mstar[valid]

        theta_plot = np.append(theta_v, theta_v[0])
        r_plot = np.append(inv_mstar, inv_mstar[0])

        color = colors[i % len(colors)]
        ax.plot(theta_plot, r_plot, '-', linewidth=2.0,
                color=color, label=label)
        ax.fill(theta_plot, r_plot, alpha=0.1, color=color)

    ax.set_title(title, fontsize=14, pad=20)
    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))
    ax.set_rlabel_position(45)

    fig.tight_layout()
    fig.savefig(filename, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {filename}")


# ── Main: load results and plot ────────────────────────────────
print("=" * 60)
print("Polar Plots of Angular Effective Mass")
print("=" * 60)

results_file = 'angular_mstar_results.json'
if not os.path.exists(results_file):
    print(f"Results file not found: {results_file}")
    print("Run Step A4 first.")
else:
    with open(results_file, 'r') as f:
        results = json.load(f)

    theta_vbm = np.deg2rad(results['theta_vbm_deg'])
    mstar_vbm = np.array([x if x is not None else np.nan
                           for x in results['mstar_vbm']])

    theta_cbm = np.deg2rad(results['theta_cbm_deg'])
    mstar_cbm = np.array([x if x is not None else np.nan
                           for x in results['mstar_cbm']])

    # Individual polar plots
    polar_plot_effective_mass(
        theta_vbm, mstar_vbm,
        title="VBM: 1/m*(theta) in kx-ky plane",
        filename="polar_mstar_vbm.png"
    )

    polar_plot_effective_mass(
        theta_cbm, mstar_cbm,
        title="CBM: 1/m*(theta) in longitudinal-transverse plane",
        filename="polar_mstar_cbm.png"
    )

    # Comparison plot
    polar_plot_comparison(
        [theta_vbm, theta_cbm],
        [mstar_vbm, mstar_cbm],
        ['VBM (hole)', 'CBM (electron)'],
        title="Angular Effective Mass Comparison",
        filename="polar_mstar_comparison.png"
    )

    # Report key anisotropy metrics
    print("\n--- Anisotropy Summary ---")
    for name, theta, mstar in [('VBM', theta_vbm, mstar_vbm),
                                 ('CBM', theta_cbm, mstar_cbm)]:
        valid = ~np.isnan(mstar) & (mstar > 0)
        if not np.any(valid):
            continue
        m_min = np.nanmin(mstar)
        m_max = np.nanmax(mstar)
        idx_min = np.nanargmin(mstar)
        idx_max = np.nanargmax(mstar)
        print(f"\n  {name}:")
        print(f"    Lightest: m* = {m_min:.4f} m_e at "
              f"theta = {np.rad2deg(theta[idx_min]):.1f} deg")
        print(f"    Heaviest: m* = {m_max:.4f} m_e at "
              f"theta = {np.rad2deg(theta[idx_max]):.1f} deg")
        print(f"    Anisotropy ratio: {m_max / m_min:.2f}")
```

#### Step A6: Extract Effective Mass Tensor from Angular Data

```python
#!/usr/bin/env python3
"""
Extract the full 2D effective mass tensor from angular m*(theta) data.

The inverse mass in direction theta is:
  1/m*(theta) = cos^2(theta)/m_1 + sin^2(theta)/m_2
              + 2*cos(theta)*sin(theta)*(1/m*)_12

where m_1, m_2 are the principal masses and (1/m*)_12 is the off-diagonal
element. We fit this formula to m*(theta) data to extract the tensor.
"""
import json
import numpy as np

HBAR_SI = 1.054571817e-34
M_E = 9.1093837015e-31
EV_TO_J = 1.602176634e-19


def fit_mass_tensor_2d(theta, mstar_me):
    """
    Fit the 2D inverse effective mass tensor from angular data.

    The model is:
      1/m*(theta) = A*cos^2(theta) + B*sin^2(theta) + C*sin(2*theta)

    where A = (1/m*)_xx, B = (1/m*)_yy, C = (1/m*)_xy.

    Parameters
    ----------
    theta : ndarray (n,)
        Angles in radians.
    mstar_me : ndarray (n,)
        Effective mass m*/m_e at each angle.

    Returns
    -------
    inv_mass_tensor : ndarray (2,2)
        Inverse effective mass tensor in units of 1/m_e.
    principal_masses : ndarray (2,)
        Principal effective masses in m_e.
    principal_angles : ndarray (2,)
        Angles (radians) of the principal directions.
    fit_r_squared : float
        R^2 of the fit.
    """
    valid = ~np.isnan(mstar_me) & (mstar_me > 0)
    theta_v = theta[valid]
    inv_mstar = 1.0 / mstar_me[valid]

    if len(theta_v) < 3:
        print("  Not enough valid data for tensor fit.")
        return None, None, None, 0.0

    # Design matrix: 1/m* = A*cos^2 + B*sin^2 + C*sin(2*theta)
    X = np.column_stack([
        np.cos(theta_v) ** 2,
        np.sin(theta_v) ** 2,
        np.sin(2 * theta_v),
    ])

    # Least-squares fit
    coeffs, _, _, _ = np.linalg.lstsq(X, inv_mstar, rcond=None)
    A, B, C = coeffs

    # Inverse mass tensor (1/m_e units)
    inv_mass_tensor = np.array([
        [A, C],
        [C, B],
    ])

    # Diagonalize to get principal masses
    eigenvalues, eigenvectors = np.linalg.eigh(inv_mass_tensor)
    principal_masses = 1.0 / eigenvalues  # in m_e

    # Principal directions (angles)
    principal_angles = np.array([
        np.arctan2(eigenvectors[1, i], eigenvectors[0, i])
        for i in range(2)
    ])

    # R^2
    inv_mstar_fit = X @ coeffs
    ss_res = np.sum((inv_mstar - inv_mstar_fit) ** 2)
    ss_tot = np.sum((inv_mstar - np.mean(inv_mstar)) ** 2)
    r2 = 1 - ss_res / ss_tot if ss_tot > 1e-30 else 0.0

    return inv_mass_tensor, np.abs(principal_masses), principal_angles, r2


# ── Main ───────────────────────────────────────────────────────
print("=" * 60)
print("Effective Mass Tensor from Angular Data")
print("=" * 60)

results_file = 'angular_mstar_results.json'
if not os.path.exists(results_file):
    print(f"Results file not found: {results_file}")
else:
    with open(results_file, 'r') as f:
        results = json.load(f)

    for name, theta_key, mstar_key in [
        ('VBM', 'theta_vbm_deg', 'mstar_vbm'),
        ('CBM', 'theta_cbm_deg', 'mstar_cbm'),
    ]:
        theta = np.deg2rad(results[theta_key])
        mstar = np.array([x if x is not None else np.nan
                          for x in results[mstar_key]])

        print(f"\n--- {name} ---")
        inv_tensor, principals, angles, r2 = fit_mass_tensor_2d(
            theta, mstar)

        if inv_tensor is not None:
            print(f"  Inverse mass tensor (1/m_e):")
            print(f"    [{inv_tensor[0, 0]:8.4f}  {inv_tensor[0, 1]:8.4f}]")
            print(f"    [{inv_tensor[1, 0]:8.4f}  {inv_tensor[1, 1]:8.4f}]")

            print(f"\n  Principal effective masses:")
            for i in range(2):
                angle_deg = np.rad2deg(angles[i])
                print(f"    m*_{i + 1} = {principals[i]:.4f} m_e "
                      f"(direction: {angle_deg:.1f} deg)")

            print(f"\n  Anisotropy ratio: "
                  f"{max(principals) / min(principals):.2f}")
            print(f"  Fit R^2: {r2:.6f}")

            # Derived quantities
            m_dos_2d = np.sqrt(principals[0] * principals[1])
            m_cond_2d = 2.0 / (1.0 / principals[0] + 1.0 / principals[1])
            print(f"\n  2D DOS effective mass: {m_dos_2d:.4f} m_e")
            print(f"  2D conductivity mass: {m_cond_2d:.4f} m_e")

    # Save tensor results
    tensor_results = {}
    for name, theta_key, mstar_key in [
        ('VBM', 'theta_vbm_deg', 'mstar_vbm'),
        ('CBM', 'theta_cbm_deg', 'mstar_cbm'),
    ]:
        theta = np.deg2rad(results[theta_key])
        mstar = np.array([x if x is not None else np.nan
                          for x in results[mstar_key]])
        inv_tensor, principals, angles, r2 = fit_mass_tensor_2d(
            theta, mstar)
        if inv_tensor is not None:
            tensor_results[name] = {
                'inv_mass_tensor': inv_tensor.tolist(),
                'principal_masses_me': principals.tolist(),
                'principal_angles_deg': np.rad2deg(angles).tolist(),
                'fit_r_squared': float(r2),
            }

    with open('angular_mass_tensor.json', 'w') as f:
        json.dump(tensor_results, f, indent=2)
    print("\nTensor results saved: angular_mass_tensor.json")
```

#### Step A7: Angle-Dependent Fermi Velocity v_F(theta)

```python
#!/usr/bin/env python3
"""
Compute angle-dependent Fermi velocity v_F(theta) for metallic bands.

For bands crossing the Fermi level, the Fermi velocity along direction
theta is v_F(theta) = (1/hbar) * |dE/dk|_{E=E_F} evaluated along that
direction.

This is relevant for metals, semimetals, Dirac/Weyl materials, and
topological surface states.
"""
import re
import os
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

HBAR_SI = 1.054571817e-34       # J*s
M_E = 9.1093837015e-31          # kg
EV_TO_J = 1.602176634e-19       # J/eV
C_LIGHT = 2.998e8               # m/s


def parse_nscf_eigenvalues(filename):
    """Parse QE NSCF output for k-points and eigenvalues."""
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


def compute_angular_fermi_velocity(kpoints_cryst, eigenvalues,
                                    band_index, k_center,
                                    plane_vectors,
                                    ring_indices, n_angles,
                                    lattice_matrix_ang):
    """
    Compute angle-dependent Fermi velocity v_F(theta).

    For each angle theta, compute the band slope dE/dk along the
    radial direction at that angle using finite differences.

    Parameters
    ----------
    kpoints_cryst : ndarray (nk, 3)
        K-points in crystal coordinates.
    eigenvalues : ndarray (nk, nbnd)
        Eigenvalues in eV.
    band_index : int
        Band index crossing the Fermi level.
    k_center : array (3,)
        Center k-point (Dirac point or Fermi crossing).
    plane_vectors : tuple of two arrays
        Plane vectors defining the angular reference.
    ring_indices : list of lists
        Indices grouping k-points by ring.
    n_angles : int
        Number of angular samples per ring.
    lattice_matrix_ang : ndarray (3,3)
        Real-space lattice vectors in Angstrom.

    Returns
    -------
    theta_values : ndarray (n_angles,)
        Angles in radians.
    vf_theta : ndarray (n_angles,)
        Fermi velocity v_F(theta) in m/s.
    """
    recip_matrix = 2 * np.pi * np.linalg.inv(lattice_matrix_ang).T
    k_cart = np.array([recip_matrix @ k for k in kpoints_cryst])
    k_cart_m = k_cart * 1e10  # 1/m

    k0_cart = recip_matrix @ np.array(k_center) * 1e10

    theta_values = np.linspace(0, 2 * np.pi, n_angles, endpoint=False)
    vf_theta = np.full(n_angles, np.nan)

    # Center point energy
    E0 = eigenvalues[0, band_index]  # index 0 is the center

    for i_angle in range(n_angles):
        # Collect (dk, E) pairs along this angle
        dk_list = []
        E_list = []

        for ring_idx in ring_indices:
            global_idx = ring_idx[i_angle]
            k_this = k_cart_m[global_idx]
            dk_mag = np.linalg.norm(k_this - k0_cart)
            E_this = eigenvalues[global_idx, band_index]

            if not np.isnan(E_this) and dk_mag > 0:
                dk_list.append(dk_mag)
                E_list.append(E_this)

        if len(dk_list) < 2:
            continue

        dk_arr = np.array(dk_list)
        E_arr = np.array(E_list)

        # Fit linear E(k) = E0 + v * dk near the center
        # For Dirac cones, this is exact. For parabolic bands near E_F,
        # it gives the group velocity at the nearest ring.
        # Use the innermost ring for the most accurate slope.
        dE = E_arr - E0
        sort_idx = np.argsort(dk_arr)
        dk_sorted = dk_arr[sort_idx]
        dE_sorted = dE[sort_idx]

        # Linear fit through innermost points
        n_fit = min(3, len(dk_sorted))
        coeffs = np.polyfit(dk_sorted[:n_fit], dE_sorted[:n_fit], 1)
        slope_eV_per_m = coeffs[0]  # eV / (1/m) = eV*m

        # v_F = (1/hbar) * |dE/dk| in m/s
        v_F = abs(slope_eV_per_m * EV_TO_J) / HBAR_SI
        vf_theta[i_angle] = v_F

    return theta_values, vf_theta


def polar_plot_fermi_velocity(theta, vf, title, filename):
    """Create polar plot of Fermi velocity v_F(theta)."""
    valid = ~np.isnan(vf) & (vf > 0)
    if not np.any(valid):
        print(f"  No valid data for {title}")
        return

    theta_v = theta[valid]
    vf_v = vf[valid] / 1e5  # Convert to 10^5 m/s for readability

    # Close the curve
    theta_plot = np.append(theta_v, theta_v[0])
    vf_plot = np.append(vf_v, vf_v[0])

    fig, ax = plt.subplots(1, 1, figsize=(7, 7),
                           subplot_kw={'projection': 'polar'})

    ax.plot(theta_plot, vf_plot, 'r-', linewidth=2.0)
    ax.fill(theta_plot, vf_plot, alpha=0.15, color='coral')

    ax.set_title(title, fontsize=13, pad=20)
    ax.set_rlabel_position(45)

    # Add v_F/c annotation
    vf_max = np.nanmax(vf)
    vf_over_c = vf_max / C_LIGHT
    ax.annotate(f'v_F,max/c = {vf_over_c:.4f}',
                xy=(0.05, 0.95), xycoords='figure fraction',
                fontsize=10, ha='left')

    fig.tight_layout()
    fig.savefig(filename, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {filename}")


# ── Main: Fermi velocity analysis ──────────────────────────────
print("=" * 60)
print("Angular-Dependent Fermi Velocity")
print("=" * 60)

# Example for graphene (Dirac material)
# For a real calculation, replace with actual NSCF output.
print("\nFor metallic/Dirac systems:")
print("1. Run SCF + NSCF with angular k-points around the Fermi crossing")
print("   (e.g., Dirac point K in graphene)")
print("2. Parse eigenvalues and call compute_angular_fermi_velocity()")
print("3. Plot with polar_plot_fermi_velocity()")

# Demo with synthetic Dirac cone data
print("\n--- Demo: Synthetic anisotropic Dirac cone ---")
n_angles = 72
n_radial = 5
dk_max = 0.03

# Anisotropic Dirac cone: v_F depends on angle
# v_Fx = 8.0e5 m/s, v_Fy = 6.5e5 m/s
vFx = 8.0e5  # m/s
vFy = 6.5e5  # m/s

theta_demo = np.linspace(0, 2 * np.pi, n_angles, endpoint=False)
vf_demo = np.sqrt((vFx * np.cos(theta_demo)) ** 2 +
                   (vFy * np.sin(theta_demo)) ** 2)

print(f"  v_F range: {np.min(vf_demo)/1e5:.2f} -- "
      f"{np.max(vf_demo)/1e5:.2f} x 10^5 m/s")
print(f"  v_F/c range: {np.min(vf_demo)/C_LIGHT:.4f} -- "
      f"{np.max(vf_demo)/C_LIGHT:.4f}")
print(f"  Anisotropy ratio: {np.max(vf_demo)/np.min(vf_demo):.2f}")

polar_plot_fermi_velocity(
    theta_demo, vf_demo,
    title="Demo: Anisotropic Dirac Cone v_F(theta)",
    filename="polar_vf_demo.png"
)

# Save demo results
demo_results = {
    'theta_deg': (theta_demo * 180 / np.pi).tolist(),
    'vf_m_per_s': vf_demo.tolist(),
    'vf_over_c': (vf_demo / C_LIGHT).tolist(),
    'anisotropy_ratio': float(np.max(vf_demo) / np.min(vf_demo)),
}
with open('angular_vf_demo.json', 'w') as f:
    json.dump(demo_results, f, indent=2)
print("Saved: angular_vf_demo.json")
```

---

### Method B: VASP

#### Step B1: Parse EIGENVAL for Angular Effective Mass and Fermi Velocity

```python
#!/usr/bin/env python3
"""
Extract angular-dependent effective mass m*(theta) and Fermi velocity
v_F(theta) from VASP EIGENVAL data.

Requires a VASP band structure calculation with k-points sampled on
angular rings around the band extremum (or Fermi crossing).
This is the VASP equivalent of VASPKIT task 914.

If using VASPKIT directly:
  vaspkit -task 914
will prompt for the extremum k-point and plane, then generate the
angular m* and v_F analysis automatically.
"""
import os
import numpy as np
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

HBAR_SI = 1.054571817e-34
M_E = 9.1093837015e-31
EV_TO_J = 1.602176634e-19
C_LIGHT = 2.998e8


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
        idx += 1  # skip blank line

    return np.array(kpoints), np.array(eigenvalues), n_electrons


def vasp_angular_effective_mass(eigenval_file='EIGENVAL',
                                 poscar_file='POSCAR',
                                 band_index=None,
                                 k_extremum=None,
                                 n_angles=72,
                                 n_radial=5):
    """
    Compute angular effective mass from VASP EIGENVAL.

    Assumes k-points are arranged as:
      - 1 center point
      - n_radial rings of n_angles points each

    Parameters
    ----------
    eigenval_file : str
        Path to EIGENVAL.
    poscar_file : str
        Path to POSCAR.
    band_index : int or None
        Band index (0-based). If None, uses VBM.
    k_extremum : array or None
        Expected extremum location. If None, auto-detect.
    n_angles : int
        Number of angular samples per ring.
    n_radial : int
        Number of radial rings.

    Returns
    -------
    theta : ndarray (n_angles,)
    mstar : ndarray (n_angles,)
    vf : ndarray (n_angles,)
    """
    from pymatgen.core import Structure

    kpoints, eigenvalues, n_elec = parse_vasp_eigenval(eigenval_file)
    struct = Structure.from_file(poscar_file)
    recip = struct.lattice.reciprocal_lattice.matrix  # 1/Angstrom

    n_occ = int(n_elec / 2)
    if band_index is None:
        band_index = n_occ - 1  # VBM

    # Convert k-points to Cartesian (1/m)
    k_cart = kpoints @ recip * 1e10  # 1/m

    # Center point is index 0
    k0 = k_cart[0]
    E0 = eigenvalues[0, band_index]

    theta_values = np.linspace(0, 2 * np.pi, n_angles, endpoint=False)
    mstar_theta = np.full(n_angles, np.nan)
    vf_theta = np.full(n_angles, np.nan)

    for i_angle in range(n_angles):
        dk_list = []
        E_list = []

        for i_ring in range(n_radial):
            idx = 1 + i_ring * n_angles + i_angle
            if idx >= len(kpoints):
                break

            k_this = k_cart[idx]
            dk_mag = np.linalg.norm(k_this - k0)
            E_this = eigenvalues[idx, band_index]

            if not np.isnan(E_this) and dk_mag > 0:
                dk_list.append(dk_mag)
                E_list.append(E_this)

        if len(dk_list) < 2:
            continue

        dk_arr = np.array(dk_list)
        E_arr = np.array(E_list)
        dE = E_arr - E0

        # Parabolic fit for effective mass
        dk2 = dk_arr ** 2
        alpha = np.linalg.lstsq(
            dk2.reshape(-1, 1), dE, rcond=None)[0][0]

        if abs(alpha) > 1e-40:
            alpha_SI = alpha * EV_TO_J
            mstar_kg = HBAR_SI ** 2 / (2 * alpha_SI)
            mstar_theta[i_angle] = abs(mstar_kg / M_E)

        # Linear fit for Fermi velocity
        slope = np.linalg.lstsq(
            dk_arr[:2].reshape(-1, 1), dE[:2], rcond=None)[0][0]
        vf = abs(slope * EV_TO_J) / HBAR_SI
        vf_theta[i_angle] = vf

    return theta_values, mstar_theta, vf_theta


# ── Main ───────────────────────────────────────────────────────
print("=" * 60)
print("VASP Angular Effective Mass and Fermi Velocity")
print("=" * 60)

if os.path.exists('EIGENVAL') and os.path.exists('POSCAR'):
    theta, mstar, vf = vasp_angular_effective_mass(
        'EIGENVAL', 'POSCAR')

    valid_m = ~np.isnan(mstar)
    valid_v = ~np.isnan(vf)

    if np.any(valid_m):
        print(f"\n  m* range: {np.nanmin(mstar):.4f} -- "
              f"{np.nanmax(mstar):.4f} m_e")
        print(f"  Anisotropy ratio: "
              f"{np.nanmax(mstar) / np.nanmin(mstar):.2f}")

    if np.any(valid_v):
        print(f"\n  v_F range: {np.nanmin(vf)/1e5:.2f} -- "
              f"{np.nanmax(vf)/1e5:.2f} x 10^5 m/s")

    # Save
    output = {
        'theta_deg': (theta * 180 / np.pi).tolist(),
        'mstar_me': [float(x) if not np.isnan(x) else None
                     for x in mstar],
        'vf_m_per_s': [float(x) if not np.isnan(x) else None
                       for x in vf],
    }
    with open('vasp_angular_results.json', 'w') as f:
        json.dump(output, f, indent=2)
    print("\nResults saved: vasp_angular_results.json")

else:
    print("\nVASP files not found (EIGENVAL, POSCAR).")
    print("To use VASPKIT directly: vaspkit -task 914")
    print("VASPKIT 914 generates:")
    print("  - EFFECTIVE_MASS_ANGULAR.dat")
    print("  - FERMI_VELOCITY_ANGULAR.dat")
    print("  - Polar plots of m*(theta) and v_F(theta)")
```

---

## Key Parameters

| Parameter | Typical Value | Notes |
|---|---|---|
| n_angles | 36-72 | Angular resolution per ring. 72 = 5-degree steps. Higher = smoother polar plot. |
| n_radial | 3-7 | Number of concentric rings. More rings improve parabolic fit but increase computation. |
| dk_max | 0.02-0.05 (crystal coords) | Maximum ring radius. Too large = non-parabolic. Too small = numerical noise. |
| band_index | VBM or CBM | 0-indexed. For Dirac cones, use the band crossing the Dirac point. |
| ecutwfc (QE) | 40-80 Ry | Must be converged for accurate band curvature. |
| ENCUT (VASP) | 300-500 eV | Same convergence requirement. |
| plane_vectors | Crystal-specific | Define the 2D plane for angular sampling. Must be orthogonal. |
| n_fit (v_F) | 2-3 | Number of innermost rings used for Fermi velocity slope. |

### Angular Sampling Guidelines

| Symmetry | Recommended n_angles | Notes |
|---|---|---|
| Cubic (Gamma) | 36 | Fourfold symmetry in kx-ky plane; fewer points needed. |
| Hexagonal (K point) | 72 | Threefold symmetry; fine sampling captures details. |
| Orthorhombic | 72 | Two distinct principal axes; full angular scan. |
| Monoclinic | 90-120 | Low symmetry; high resolution needed. |

## Interpreting Results

### Angular Effective Mass

- **Nearly circular polar plot of 1/m*(theta)**: Isotropic mass. Typical for cubic semiconductors at Gamma.
- **Elliptical polar plot**: Anisotropic mass with two principal values. Common for Si CBM (longitudinal vs transverse).
- **Non-elliptical (higher-order angular variation)**: Warped energy surface. Occurs for heavy-hole bands and Dirac cones with trigonal warping.
- **Very large anisotropy ratio (>5)**: Strongly anisotropic transport. Common in layered materials (e.g., black phosphorus m_armchair/m_zigzag ~ 6-10).

### Angular Fermi Velocity

- **Circular v_F(theta) plot**: Isotropic Dirac cone (graphene).
- **Elliptical v_F(theta)**: Anisotropic Dirac/Weyl cone.
- **v_F/c > 0.001**: Relativistic-like quasiparticles. Graphene: v_F/c ~ 0.003.
- **v_F(theta) varies rapidly with angle**: Tilted Dirac cone or type-II Weyl point. The tilt direction has enhanced velocity.

### Principal Mass Extraction

- **R^2 > 0.99 for tensor fit**: Angular data is well described by elliptic model. The tensor eigenvalues are reliable.
- **R^2 < 0.95**: Significant warping beyond elliptic. Higher-order terms needed (trigonal warping in graphene, cubic warping in heavy-hole bands).
- **DOS effective mass (2D)**: m_DOS = sqrt(m_1 * m_2). Use for carrier concentration.
- **Conductivity effective mass (2D)**: m_cond = 2/(1/m_1 + 1/m_2). Use for transport.

## Common Issues

| Problem | Solution |
|---|---|
| **Polar plot is noisy** | Increase n_radial (more rings for better parabolic fit) or increase dk_max slightly. Check that NSCF converged tightly (conv_thr < 1e-10). |
| **m*(theta) is negative at some angles** | Fitting range is too large (non-parabolic region). Reduce dk_max. Or band is not a simple extremum (saddle point). |
| **Fermi velocity is zero at some angles** | Band is flat in that direction. Could indicate a van Hove singularity or a band touching. |
| **Anisotropy ratio is unreasonably large (>100)** | One direction has nearly flat dispersion. Check that the correct band is being analyzed. Consider whether this is a real very flat direction. |
| **Tensor fit R^2 is low** | Energy surface has warping beyond elliptic. Report the raw angular data instead of tensor eigenvalues. Consider fitting with higher harmonics: 1/m*(theta) = A + B*cos(2*theta) + C*sin(2*theta) + D*cos(4*theta) + ... |
| **QE NSCF fails with angular k-points** | Too many k-points for available memory. Reduce n_angles or n_radial. Or split into multiple NSCF runs and concatenate results. |
| **VBM/CBM not at expected k-point** | The extremum may shift under different computational parameters. Run a preliminary band structure to locate the true extremum before angular sampling. |
| **Degenerate bands at extremum** | Heavy-hole and light-hole at Gamma are degenerate. Fit each band separately. Spin-orbit coupling lifts some degeneracies. |
| **v_F varies with ring radius** | Band dispersion is not linear (not a true Dirac cone). Use only the innermost ring for v_F. Or report v_F at the Fermi level k-point explicitly. |
