# Berry Curvature and Anomalous Hall Conductivity

## When to Use

- Computing the Berry curvature distribution across the Brillouin zone
- Calculating Chern numbers for quantum anomalous Hall systems
- Computing the anomalous Hall conductivity (AHC) from the Kubo formula
- Identifying topological features: Weyl points, nodal lines, band crossings
- Analyzing intrinsic contributions to the Hall effect in magnetic materials

## Prerequisites

- Quantum ESPRESSO 7.5 (`pw.x`, `pw2wannier90.x`)
- Wannier90 (for interpolation approach)
- Python packages: `numpy`, `scipy`, `matplotlib`
- For magnetic systems: collinear or noncollinear magnetism in QE
- For SOC: fully relativistic pseudopotentials

## Theory

### Berry Curvature

The Berry curvature of band n is:

```
Omega_n(k) = -2 * Im * sum_{m != n} <n,k|dH/dk_x|m,k> <m,k|dH/dk_y|n,k> / (E_m(k) - E_n(k))^2
```

It acts as an effective magnetic field in k-space. Singularities in Berry curvature indicate
topological features (band crossings, Weyl points).

### Chern Number

The Chern number of a 2D band is the integral of Berry curvature over the full BZ:

```
C_n = (1 / 2*pi) * integral_BZ Omega_n(k) dk_x dk_y
```

C_n is an integer for gapped bands. Nonzero Chern number implies quantized Hall conductance.

### Anomalous Hall Conductivity (AHC)

The intrinsic AHC from the Kubo formula:

```
sigma_xy = -(e^2 / hbar) * (1 / (2*pi)^d) * sum_n integral_BZ f_n(k) * Omega_n(k) d^d k
```

where f_n(k) is the Fermi-Dirac occupation. In 2D, sigma_xy has units of e^2/h.
In 3D, sigma_xy has units of (e^2/h) / length = Ohm^-1 cm^-1.

### Fukui-Hatsugai-Suzuki (FHS) Method

Discretized Berry curvature on a lattice of k-points. The lattice Berry curvature
for a plaquette is:

```
F_12(k) = Im * ln[ U_1(k) * U_2(k + dk_1) * U_1^*(k + dk_2) * U_2^*(k) ]
```

where U_mu(k) = <u_k | u_{k+dk_mu}> / |<u_k | u_{k+dk_mu}>| is the link variable.

This method is gauge-invariant and always gives integer Chern numbers
(within numerical precision) when summed over the full BZ.

## Detailed Steps

### Method 1: Wannier90 Interpolation (Accurate, Production)

#### Step 1: QE SCF + NSCF + Wannier90

```python
#!/usr/bin/env python3
"""
Berry curvature via Wannier90 interpolation.
Example: Fe bcc (magnetic, known AHC ~ 750 Ohm^-1 cm^-1).
"""
import os
import subprocess
import numpy as np

WORK_DIR = os.path.abspath("berry_fe")
os.makedirs(WORK_DIR, exist_ok=True)

# Fe bcc lattice constant
a_bohr = 5.42  # Bohr (2.87 Angstrom)

PP_FE = "Fe.rel-pbe-spn-kjpaw_psl.1.0.0.UPF"
PSEUDO_DIR = os.environ.get("PSEUDO_DIR", "./pseudo")

# ------------------------------------------------------------------
# SCF: Magnetic Fe with SOC
# ------------------------------------------------------------------
scf_input = f"""&CONTROL
  calculation = 'scf'
  prefix = 'fe'
  outdir = './tmp'
  pseudo_dir = '{PSEUDO_DIR}'
  verbosity = 'high'
/
&SYSTEM
  ibrav = 3
  celldm(1) = {a_bohr}
  nat = 1
  ntyp = 1
  ecutwfc = 80.0
  ecutrho = 640.0
  occupations = 'smearing'
  smearing = 'cold'
  degauss = 0.01
  noncolin = .true.
  lspinorb = .true.
  starting_magnetization(1) = 0.5
  angle1(1) = 0.0
  angle2(1) = 0.0
/
&ELECTRONS
  conv_thr = 1.0d-10
  mixing_beta = 0.3
/
ATOMIC_SPECIES
  Fe  55.845  {PP_FE}
ATOMIC_POSITIONS {{crystal}}
  Fe  0.0  0.0  0.0
K_POINTS {{automatic}}
  12 12 12 0 0 0
"""

with open(os.path.join(WORK_DIR, "scf.in"), "w") as f:
    f.write(scf_input)

# ------------------------------------------------------------------
# NSCF on uniform grid for Wannier90
# ------------------------------------------------------------------
nscf_input = f"""&CONTROL
  calculation = 'nscf'
  prefix = 'fe'
  outdir = './tmp'
  pseudo_dir = '{PSEUDO_DIR}'
  verbosity = 'high'
/
&SYSTEM
  ibrav = 3
  celldm(1) = {a_bohr}
  nat = 1
  ntyp = 1
  ecutwfc = 80.0
  ecutrho = 640.0
  occupations = 'smearing'
  smearing = 'cold'
  degauss = 0.01
  noncolin = .true.
  lspinorb = .true.
  nosym = .true.
  nbnd = 36
  starting_magnetization(1) = 0.5
  angle1(1) = 0.0
  angle2(1) = 0.0
/
&ELECTRONS
  conv_thr = 1.0d-10
  diago_full_acc = .true.
/
ATOMIC_SPECIES
  Fe  55.845  {PP_FE}
ATOMIC_POSITIONS {{crystal}}
  Fe  0.0  0.0  0.0
K_POINTS {{automatic}}
  8 8 8 0 0 0
"""

with open(os.path.join(WORK_DIR, "nscf.in"), "w") as f:
    f.write(nscf_input)

# ------------------------------------------------------------------
# Wannier90 input for Berry curvature
# ------------------------------------------------------------------
wannier_input = """num_wann = 18
num_bands = 36

! Fe: 3d + 4s orbitals with spinors
begin projections
Fe: s;d
end projections

spinors = .true.

! Disentanglement
dis_win_min  = -10.0
dis_win_max  =  40.0
dis_froz_min = -10.0
dis_froz_max =   5.0
dis_num_iter = 200
dis_mix_ratio = 0.5

! Wannierization
num_iter = 200
conv_tol = 1.0e-10

! Berry curvature and AHC calculation
berry = true
berry_task = ahc
berry_kmesh = 50 50 50

! Fermi energy (from SCF output, adjust after running SCF)
fermi_energy = 12.0

! Output
write_hr = .true.
"""

with open(os.path.join(WORK_DIR, "fe.win"), "w") as f:
    f.write(wannier_input)

# ------------------------------------------------------------------
# pw2wannier90 input
# ------------------------------------------------------------------
pw2wan_input = """&inputpp
  outdir = './tmp'
  prefix = 'fe'
  seedname = 'fe'
  spin_component = 'none'
  write_mmn = .true.
  write_amn = .true.
  write_unk = .false.
/
"""

with open(os.path.join(WORK_DIR, "pw2wan.in"), "w") as f:
    f.write(pw2wan_input)

print("All input files written to", WORK_DIR)
print("\nExecution order:")
print("  1. mpirun -np 4 pw.x -in scf.in > scf.out")
print("  2. mpirun -np 4 pw.x -in nscf.in > nscf.out")
print("  3. wannier90.x -pp fe")
print("  4. mpirun -np 4 pw2wannier90.x -in pw2wan.in > pw2wan.out")
print("  5. wannier90.x fe  (runs Wannier90 + Berry curvature)")
print("\nAHC results will appear in fe-ahc-fermiscan.dat")
```

```bash
# Run the full pipeline
cd berry_fe
mpirun -np 4 pw.x -in scf.in > scf.out 2>&1
mpirun -np 4 pw.x -in nscf.in > nscf.out 2>&1
wannier90.x -pp fe > wan_pp.out 2>&1
mpirun -np 4 pw2wannier90.x -in pw2wan.in > pw2wan.out 2>&1
wannier90.x fe > wan.out 2>&1
```

#### Step 2: Parse and Plot Wannier90 AHC Output

```python
#!/usr/bin/env python3
"""
Parse Wannier90 AHC output and create plots.
"""
import os
import numpy as np
import matplotlib.pyplot as plt

WORK_DIR = os.path.abspath("berry_fe")


def parse_ahc_fermiscan(filepath):
    """
    Parse the fe-ahc-fermiscan.dat file from Wannier90.
    Format: E(eV)  sigma_xy  sigma_xz  sigma_yz  (Ohm^-1 cm^-1)
    """
    data = np.loadtxt(filepath, comments="#")
    return {
        "energy": data[:, 0],
        "sigma_xy": data[:, 1],
        "sigma_xz": data[:, 2],
        "sigma_yz": data[:, 3],
    }


# Parse results
ahc_file = os.path.join(WORK_DIR, "fe-ahc-fermiscan.dat")
if os.path.exists(ahc_file):
    ahc = parse_ahc_fermiscan(ahc_file)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(ahc["energy"], ahc["sigma_xy"], "b-", linewidth=2, label=r"$\sigma_{xy}$")
    ax.plot(ahc["energy"], ahc["sigma_xz"], "r--", linewidth=1.5, label=r"$\sigma_{xz}$")
    ax.plot(ahc["energy"], ahc["sigma_yz"], "g:", linewidth=1.5, label=r"$\sigma_{yz}$")

    ax.axhline(y=0, color="gray", linewidth=0.5)
    ax.axvline(x=0, color="gray", linewidth=0.5, linestyle="--", label="$E_F$")

    ax.set_xlabel("Energy (eV)", fontsize=12)
    ax.set_ylabel(r"AHC ($\Omega^{-1}$ cm$^{-1}$)", fontsize=12)
    ax.set_title("Anomalous Hall Conductivity - bcc Fe", fontsize=14)
    ax.legend(fontsize=11)
    ax.set_xlim(-5, 5)

    plt.tight_layout()
    plt.savefig(os.path.join(WORK_DIR, "ahc_fe.png"), dpi=150)
    plt.close()
    print(f"AHC plot saved to {os.path.join(WORK_DIR, 'ahc_fe.png')}")

    # Print value at Fermi level
    idx_ef = np.argmin(np.abs(ahc["energy"]))
    print(f"AHC at E_F: sigma_xy = {ahc['sigma_xy'][idx_ef]:.1f} Ohm^-1 cm^-1")
    print(f"Expected for bcc Fe: ~750 Ohm^-1 cm^-1")
else:
    print(f"AHC file not found: {ahc_file}")
    print("Run the Wannier90 calculation first.")
```

### Method 2: Pure Python Numerical Berry Curvature (FHS Method)

```python
#!/usr/bin/env python3
"""
Compute Berry curvature using the Fukui-Hatsugai-Suzuki (FHS) method.
Discretized, gauge-invariant approach on a k-grid.

This standalone script works with a model Hamiltonian for demonstration,
then provides the function to read QE eigenvalues/wavefunctions.
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm


# ===================================================================
# Part 1: FHS Berry curvature on a model (Haldane model)
# ===================================================================

def haldane_hamiltonian(kx, ky, t1=1.0, t2=0.1, phi=np.pi/2, M=0.0):
    """
    Haldane model on honeycomb lattice.
    t1: nearest-neighbor hopping
    t2: next-nearest-neighbor hopping (complex, breaks TRS)
    phi: phase of t2 (phi != 0, pi breaks TRS)
    M: sublattice mass (breaks inversion)

    Topological phase: |M/t2| < 3*sqrt(3)*|sin(phi)|
    """
    # Honeycomb lattice vectors
    a1 = np.array([1.0, 0.0])
    a2 = np.array([0.5, np.sqrt(3)/2])
    # Nearest-neighbor vectors (A -> B)
    d1 = np.array([0.0, 1.0/np.sqrt(3)])
    d2 = np.array([0.5, -1.0/(2*np.sqrt(3))])
    d3 = np.array([-0.5, -1.0/(2*np.sqrt(3))])
    # Next-nearest-neighbor vectors
    b1 = a1
    b2 = a2
    b3 = a2 - a1

    k = np.array([kx, ky])

    # Off-diagonal: nearest-neighbor hopping
    f_k = t1 * (np.exp(1j * k.dot(d1)) + np.exp(1j * k.dot(d2)) + np.exp(1j * k.dot(d3)))

    # Diagonal: NNN hopping (breaks TRS)
    g_k = 2 * t2 * (np.cos(k.dot(b1) + phi) + np.cos(k.dot(b2) + phi) +
                     np.cos(k.dot(b3) + phi))

    g_k_minus = 2 * t2 * (np.cos(k.dot(b1) - phi) + np.cos(k.dot(b2) - phi) +
                           np.cos(k.dot(b3) - phi))

    H = np.array([
        [M + g_k,        f_k],
        [np.conj(f_k),  -M + g_k_minus],
    ])

    return H


def compute_berry_curvature_fhs(hamiltonian_func, nkx, nky, kx_range, ky_range,
                                  n_occ=1):
    """
    Compute Berry curvature using the Fukui-Hatsugai-Suzuki method.

    Parameters
    ----------
    hamiltonian_func : callable
        H(kx, ky) -> (n_bands, n_bands) complex array
    nkx, nky : int
        Number of k-points in each direction
    kx_range, ky_range : tuple
        (kmin, kmax) for each direction
    n_occ : int
        Number of occupied bands

    Returns
    -------
    kx_grid : (nkx, nky) array of kx values (plaquette centers)
    ky_grid : (nkx, nky) array of ky values
    berry_curv : (nkx, nky) array of Berry curvature
    chern : float, Chern number
    """
    kx_vals = np.linspace(kx_range[0], kx_range[1], nkx, endpoint=False)
    ky_vals = np.linspace(ky_range[0], ky_range[1], nky, endpoint=False)
    dkx = kx_vals[1] - kx_vals[0]
    dky = ky_vals[1] - ky_vals[0]

    # Compute eigenstates on the grid (including one extra row/column for links)
    eigenstates = np.zeros((nkx + 1, nky + 1, n_occ, None.__class__), dtype=object)

    # First pass: determine band dimension
    H0 = hamiltonian_func(kx_vals[0], ky_vals[0])
    n_bands = H0.shape[0]

    # Store occupied eigenstates at each k-point
    states = np.zeros((nkx + 1, nky + 1, n_bands, n_occ), dtype=complex)

    for i in range(nkx + 1):
        for j in range(nky + 1):
            kx = kx_vals[i % nkx]
            ky = ky_vals[j % nky]
            H = hamiltonian_func(kx, ky)
            eigenvalues, eigenvectors = np.linalg.eigh(H)
            # Take the n_occ lowest eigenstates
            states[i, j] = eigenvectors[:, :n_occ]

    # Compute Berry curvature on each plaquette
    berry_curv = np.zeros((nkx, nky))

    for i in range(nkx):
        for j in range(nky):
            # Four corners of the plaquette
            # (i,j) -> (i+1,j) -> (i+1,j+1) -> (i,j+1) -> (i,j)
            u00 = states[i,   j]      # (n_bands, n_occ)
            u10 = states[i+1, j]
            u11 = states[i+1, j+1]
            u01 = states[i,   j+1]

            # Link variables: U_mu = det(<u_k | u_{k+dk_mu}>)
            # For multi-band case, compute determinant of overlap matrix
            U1 = np.linalg.det(u00.conj().T @ u10)  # link along kx at bottom
            U2 = np.linalg.det(u10.conj().T @ u11)  # link along ky at right
            U3 = np.linalg.det(u11.conj().T @ u01)  # link along -kx at top
            U4 = np.linalg.det(u01.conj().T @ u00)  # link along -ky at left

            # Berry phase around plaquette
            F = np.log(U1 * U2 * U3 * U4)
            berry_curv[i, j] = F.imag

    # Chern number = (1/2pi) * sum of Berry curvatures
    chern = np.sum(berry_curv) / (2 * np.pi)

    # Create meshgrid for plaquette centers
    kx_centers = kx_vals + dkx / 2
    ky_centers = ky_vals + dky / 2
    kx_grid, ky_grid = np.meshgrid(kx_centers, ky_centers, indexing="ij")

    # Convert from Berry phase per plaquette to curvature density
    berry_curv_density = berry_curv / (dkx * dky)

    return kx_grid, ky_grid, berry_curv_density, chern


# ------------------------------------------------------------------
# Compute Berry curvature for the Haldane model
# ------------------------------------------------------------------

# Reciprocal lattice vectors for honeycomb
b1 = 2 * np.pi * np.array([1.0, -1.0/np.sqrt(3)])
b2 = 2 * np.pi * np.array([0.0, 2.0/np.sqrt(3)])

# BZ boundaries (use rectangular region covering the BZ)
kx_range = (-2*np.pi, 2*np.pi)
ky_range = (-2*np.pi/np.sqrt(3), 4*np.pi/np.sqrt(3))

nk = 100  # k-points per direction

print("="*60)
print("Haldane Model Berry Curvature Calculation")
print("="*60)

# Topological phase: phi = pi/2, M = 0
print("\n--- Topological phase (phi=pi/2, M=0) ---")
kx_g, ky_g, omega_topo, chern_topo = compute_berry_curvature_fhs(
    lambda kx, ky: haldane_hamiltonian(kx, ky, t2=0.1, phi=np.pi/2, M=0.0),
    nk, nk, kx_range, ky_range, n_occ=1,
)
print(f"Chern number: {chern_topo:.4f} (expected: +1 or -1)")

# Trivial phase: phi = 0
print("\n--- Trivial phase (phi=0, M=0.5) ---")
kx_g2, ky_g2, omega_triv, chern_triv = compute_berry_curvature_fhs(
    lambda kx, ky: haldane_hamiltonian(kx, ky, t2=0.1, phi=0.0, M=0.5),
    nk, nk, kx_range, ky_range, n_occ=1,
)
print(f"Chern number: {chern_triv:.4f} (expected: 0)")

# ------------------------------------------------------------------
# Plot Berry curvature heatmaps
# ------------------------------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Topological
vmax = np.percentile(np.abs(omega_topo), 99)
norm1 = TwoSlopeNorm(vmin=-vmax, vcenter=0, vmax=vmax)
im1 = axes[0].pcolormesh(kx_g, ky_g, omega_topo, cmap="RdBu_r", norm=norm1,
                          shading="auto")
axes[0].set_xlabel(r"$k_x$", fontsize=12)
axes[0].set_ylabel(r"$k_y$", fontsize=12)
axes[0].set_title(f"Topological (C={chern_topo:.2f})", fontsize=13)
axes[0].set_aspect("equal")
plt.colorbar(im1, ax=axes[0], label=r"$\Omega(k)$")

# Trivial
vmax2 = max(np.percentile(np.abs(omega_triv), 99), 1e-6)
norm2 = TwoSlopeNorm(vmin=-vmax2, vcenter=0, vmax=vmax2)
im2 = axes[1].pcolormesh(kx_g2, ky_g2, omega_triv, cmap="RdBu_r", norm=norm2,
                          shading="auto")
axes[1].set_xlabel(r"$k_x$", fontsize=12)
axes[1].set_ylabel(r"$k_y$", fontsize=12)
axes[1].set_title(f"Trivial (C={chern_triv:.2f})", fontsize=13)
axes[1].set_aspect("equal")
plt.colorbar(im2, ax=axes[1], label=r"$\Omega(k)$")

plt.suptitle("Berry Curvature - Haldane Model", fontsize=14, y=1.02)
plt.tight_layout()
plt.savefig("berry_curvature_haldane.png", dpi=150, bbox_inches="tight")
plt.close()
print(f"\nBerry curvature plot saved to berry_curvature_haldane.png")


# ===================================================================
# Part 2: Berry curvature from QE wavefunctions
# ===================================================================

def read_qe_wavefunctions(prefix, outdir, ik, nbnd):
    """
    Read QE wavefunctions from UNK files (produced by pw2wannier90).
    UNK files contain u_nk(r) on the real-space grid.

    Alternative: read from the binary wfc files using the
    QE Python tools or parse the XML output.

    Parameters
    ----------
    prefix : str
        QE calculation prefix
    outdir : str
        QE output directory
    ik : int
        k-point index (1-based)
    nbnd : int
        Number of bands to read

    Returns
    -------
    eigenvectors : (ngrid, nbnd) complex array
    """
    # UNK file format: UNK{ik:05d}.{ispin}
    # Line 1: ngx ngy ngz ik nbnd
    # Then: nbnd blocks of ngx*ngy*ngz complex numbers

    unk_file = os.path.join(outdir, f"UNK{ik:05d}.1")
    if not os.path.exists(unk_file):
        raise FileNotFoundError(f"UNK file not found: {unk_file}")

    with open(unk_file, "r") as f:
        header = f.readline().split()
        ngx, ngy, ngz = int(header[0]), int(header[1]), int(header[2])
        ngrid = ngx * ngy * ngz

        wavefunctions = np.zeros((ngrid, nbnd), dtype=complex)
        for ib in range(nbnd):
            for ig in range(ngrid):
                line = f.readline().split()
                wavefunctions[ig, ib] = float(line[0]) + 1j * float(line[1])

    return wavefunctions


def berry_curvature_from_qe(prefix, outdir, kpoints, n_occ, nbnd):
    """
    Compute Berry curvature from QE wavefunctions using FHS method.

    Parameters
    ----------
    prefix : str
        QE prefix
    outdir : str
        QE output directory
    kpoints : (nkx*nky, 3) array
        k-points in crystal coordinates (must form a regular 2D grid)
    n_occ : int
        Number of occupied bands
    nbnd : int
        Total bands in UNK files

    Returns
    -------
    berry_curv : (nkx, nky) array
    chern : float
    """
    nk = len(kpoints)
    nkx = int(np.sqrt(nk))  # assumes square grid
    nky = nkx

    # Read all wavefunctions
    print(f"Reading wavefunctions for {nk} k-points...")
    wfcs = []
    for ik in range(1, nk + 1):
        wfc = read_qe_wavefunctions(prefix, outdir, ik, nbnd)
        wfcs.append(wfc[:, :n_occ])  # keep only occupied

    # Reshape to 2D grid
    wfcs = np.array(wfcs).reshape(nkx, nky, -1, n_occ)

    # Compute Berry curvature using FHS
    berry_curv = np.zeros((nkx, nky))

    for i in range(nkx):
        for j in range(nky):
            ip1 = (i + 1) % nkx
            jp1 = (j + 1) % nky

            u00 = wfcs[i, j]
            u10 = wfcs[ip1, j]
            u11 = wfcs[ip1, jp1]
            u01 = wfcs[i, jp1]

            # Overlap matrices
            U1 = np.linalg.det(u00.conj().T @ u10)
            U2 = np.linalg.det(u10.conj().T @ u11)
            U3 = np.linalg.det(u11.conj().T @ u01)
            U4 = np.linalg.det(u01.conj().T @ u00)

            berry_curv[i, j] = np.imag(np.log(U1 * U2 * U3 * U4))

    chern = np.sum(berry_curv) / (2 * np.pi)
    return berry_curv, chern


# ===================================================================
# Part 3: Berry curvature from Wannier90 Hamiltonian (_hr.dat)
# ===================================================================

def read_wannier90_hr(filename):
    """
    Read Wannier90 Hamiltonian from *_hr.dat file.

    Returns
    -------
    n_wann : int
    n_rpts : int
    degeneracies : (n_rpts,) int array
    rvecs : (n_rpts, 3) int array
    hr : (n_rpts, n_wann, n_wann) complex array
    """
    with open(filename, "r") as f:
        f.readline()  # header
        n_wann = int(f.readline().strip())
        n_rpts = int(f.readline().strip())

        # Degeneracies (may span multiple lines, 15 per line)
        degeneracies = []
        while len(degeneracies) < n_rpts:
            line = f.readline().split()
            degeneracies.extend([int(x) for x in line])
        degeneracies = np.array(degeneracies)

        # Hamiltonian matrix elements
        rvecs = np.zeros((n_rpts, 3), dtype=int)
        hr = np.zeros((n_rpts, n_wann, n_wann), dtype=complex)

        ir = 0
        im = 0
        in_ = 0
        for line in f:
            parts = line.split()
            if len(parts) < 7:
                continue
            r1, r2, r3 = int(parts[0]), int(parts[1]), int(parts[2])
            m, n = int(parts[3]) - 1, int(parts[4]) - 1  # 0-indexed
            re_val, im_val = float(parts[5]), float(parts[6])

            # Determine R-vector index
            rvec = np.array([r1, r2, r3])
            # Find or assign index
            if m == 0 and n == 0:
                rvecs[ir] = rvec
                current_ir = ir
                ir += 1
            else:
                current_ir = ir - 1

            hr[current_ir, m, n] = re_val + 1j * im_val

    return n_wann, n_rpts, degeneracies, rvecs, hr


def wannier_berry_curvature_2d(hr_file, kx_array, ky_array, kz=0.0,
                                 n_occ=None, ef=None):
    """
    Compute Berry curvature on a 2D k-slice from Wannier90 Hamiltonian.

    Uses the Kubo formula (sum over states) approach:
    Omega_xy(k) = -2 Im sum_{n occ, m unocc}
                  <n|dH/dkx|m><m|dH/dky|n> / (Em - En)^2

    Parameters
    ----------
    hr_file : str
        Path to *_hr.dat file
    kx_array, ky_array : 1D arrays
        k-points for the 2D grid (in fractional coords)
    kz : float
        Fixed kz value for the 2D slice
    n_occ : int or None
        Number of occupied bands (if None, use ef)
    ef : float or None
        Fermi energy in eV (used if n_occ is None)

    Returns
    -------
    omega : (nkx, nky) array, Berry curvature
    """
    n_wann, n_rpts, degen, rvecs, hr = read_wannier90_hr(hr_file)
    nkx = len(kx_array)
    nky = len(ky_array)

    omega = np.zeros((nkx, nky))

    for i, kx in enumerate(kx_array):
        for j, ky in enumerate(ky_array):
            k = np.array([kx, ky, kz])

            # Fourier transform: H(k) = sum_R H(R) * exp(i k.R) / deg(R)
            Hk = np.zeros((n_wann, n_wann), dtype=complex)
            dHk_dkx = np.zeros((n_wann, n_wann), dtype=complex)
            dHk_dky = np.zeros((n_wann, n_wann), dtype=complex)

            for ir in range(n_rpts):
                R = rvecs[ir]
                phase = np.exp(2j * np.pi * k.dot(R))
                weight = phase / degen[ir]

                Hk += hr[ir] * weight
                # dH/dk_alpha = i * R_alpha * H(R) * exp(i k.R) / deg
                dHk_dkx += 1j * (2 * np.pi * R[0]) * hr[ir] * weight
                dHk_dky += 1j * (2 * np.pi * R[1]) * hr[ir] * weight

            # Diagonalize
            eigenvalues, eigenvectors = np.linalg.eigh(Hk)
            # eigenvectors[:, n] is the n-th eigenvector

            # Determine occupied bands
            if n_occ is not None:
                occ_idx = list(range(n_occ))
                unocc_idx = list(range(n_occ, n_wann))
            else:
                occ_idx = [n for n in range(n_wann) if eigenvalues[n] <= ef]
                unocc_idx = [n for n in range(n_wann) if eigenvalues[n] > ef]

            # Kubo formula
            for n in occ_idx:
                for m in unocc_idx:
                    dE = eigenvalues[m] - eigenvalues[n]
                    if abs(dE) < 1e-6:
                        continue  # skip degenerate pairs

                    # Matrix elements in eigenbasis
                    vx_nm = eigenvectors[:, n].conj() @ dHk_dkx @ eigenvectors[:, m]
                    vy_mn = eigenvectors[:, m].conj() @ dHk_dky @ eigenvectors[:, n]

                    omega[i, j] -= 2.0 * np.imag(vx_nm * vy_mn) / dE**2

    return omega


def compute_ahc_from_omega(omega, kx_array, ky_array, cell_volume_ang3):
    """
    Integrate Berry curvature to get AHC.

    sigma_xy = -e^2/hbar * 1/V_cell * 1/(2pi)^2 * integral Omega dk

    Parameters
    ----------
    omega : (nkx, nky) array
        Berry curvature in reciprocal lattice units
    kx_array, ky_array : 1D arrays
        k-points (fractional)
    cell_volume_ang3 : float
        Unit cell volume in Angstrom^3

    Returns
    -------
    sigma_xy : float, in Ohm^-1 cm^-1
    """
    dkx = kx_array[1] - kx_array[0] if len(kx_array) > 1 else 1.0
    dky = ky_array[1] - ky_array[0] if len(ky_array) > 1 else 1.0

    # Integral in fractional coordinates
    integral = np.sum(omega) * dkx * dky

    # Convert to SI:
    # sigma_xy = -e^2/(hbar * V_cell) * integral / (2pi)
    # e = 1.602e-19 C, hbar = 1.055e-34 J.s
    # V_cell in m^3 = cell_volume_ang3 * 1e-30
    e = 1.602176634e-19
    hbar = 1.054571817e-34
    V_cell_m3 = cell_volume_ang3 * 1e-30

    sigma_xy_SI = -(e**2 / hbar) * integral / (V_cell_m3 * (2 * np.pi))

    # Convert to Ohm^-1 cm^-1 (1 S/m = 0.01 Ohm^-1 cm^-1)
    sigma_xy_cgs = sigma_xy_SI * 0.01

    return sigma_xy_cgs


# ------------------------------------------------------------------
# Example: Wannier90 Berry curvature for Fe (if _hr.dat exists)
# ------------------------------------------------------------------
hr_file = os.path.join(WORK_DIR, "fe_hr.dat") if 'WORK_DIR' in dir() else "fe_hr.dat"

if os.path.exists(hr_file):
    print("\n" + "="*60)
    print("Berry curvature from Wannier90 Hamiltonian")
    print("="*60)

    nk_dense = 50
    kx_arr = np.linspace(0, 1, nk_dense, endpoint=False)
    ky_arr = np.linspace(0, 1, nk_dense, endpoint=False)

    omega_w90 = wannier_berry_curvature_2d(
        hr_file, kx_arr, ky_arr, kz=0.0, n_occ=9
    )

    kx_g, ky_g = np.meshgrid(kx_arr, ky_arr, indexing="ij")
    vmax = np.percentile(np.abs(omega_w90), 98)
    norm = TwoSlopeNorm(vmin=-vmax, vcenter=0, vmax=vmax)

    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.pcolormesh(kx_g, ky_g, omega_w90, cmap="RdBu_r", norm=norm,
                        shading="auto")
    ax.set_xlabel(r"$k_x$ (frac.)", fontsize=12)
    ax.set_ylabel(r"$k_y$ (frac.)", fontsize=12)
    ax.set_title(r"Berry curvature $\Omega_{xy}(k_x, k_y, k_z=0)$ - bcc Fe",
                 fontsize=13)
    ax.set_aspect("equal")
    plt.colorbar(im, ax=ax, label=r"$\Omega_{xy}$ (a.u.)")
    plt.tight_layout()
    plt.savefig("berry_curvature_fe_w90.png", dpi=150)
    plt.close()
    print("Plot saved to berry_curvature_fe_w90.png")
else:
    print(f"\nWannier90 HR file not found ({hr_file}).")
    print("Run the full QE + Wannier90 pipeline first.")
    print("The Haldane model demonstration above runs without DFT.")
```

### Complete Workflow Script: Berry Curvature on 2D Slice

```python
#!/usr/bin/env python3
"""
End-to-end Berry curvature computation on a 2D BZ slice.
Generates QE inputs, runs the calculation, and produces plots.

Assumes SCF has already been run and charge density exists.
"""
import os
import subprocess
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm


def generate_2d_kgrid_nscf(prefix, nkx, nky, kz_fixed=0.0, outfile="nscf_2d.in",
                            template_file="nscf.in"):
    """
    Generate QE NSCF input with a 2D k-grid at fixed kz.

    Parameters
    ----------
    prefix : str
        QE prefix
    nkx, nky : int
        k-grid dimensions
    kz_fixed : float
        Fixed kz in crystal coordinates
    outfile : str
        Output filename
    template_file : str
        Template NSCF input (everything except K_POINTS)
    """
    # Read template
    with open(template_file, "r") as f:
        lines = f.readlines()

    # Remove existing K_POINTS block
    new_lines = []
    skip = False
    for line in lines:
        if "K_POINTS" in line:
            skip = True
            continue
        if skip and line.strip() and not line.strip()[0].isdigit():
            skip = False
        if skip:
            continue
        new_lines.append(line)

    # Add new K_POINTS block
    nk_total = nkx * nky
    new_lines.append(f"K_POINTS {{crystal}}\n")
    new_lines.append(f"{nk_total}\n")

    for i in range(nkx):
        for j in range(nky):
            kx = i / nkx
            ky = j / nky
            weight = 1.0 / nk_total
            new_lines.append(f"  {kx:.10f}  {ky:.10f}  {kz_fixed:.10f}  {weight:.10f}\n")

    with open(outfile, "w") as f:
        f.writelines(new_lines)

    print(f"NSCF input with {nkx}x{nky} 2D grid written to {outfile}")
    return nk_total


def run_nscf_and_extract(nscf_input, nproc=4):
    """Run NSCF calculation."""
    cmd = f"mpirun -np {nproc} pw.x -in {nscf_input} > {nscf_input.replace('.in', '.out')} 2>&1"
    print(f"Running: {cmd}")
    subprocess.run(cmd, shell=True, check=True)
    print("NSCF completed.")


def parse_eigenvalues_from_xml(prefix, outdir="./tmp"):
    """
    Parse eigenvalues from QE data-file-schema.xml.

    Returns
    -------
    eigenvalues : (nk, nbnd) array in eV
    kpoints : (nk, 3) array in crystal coordinates
    """
    import xml.etree.ElementTree as ET

    xml_file = os.path.join(outdir, f"{prefix}.save", "data-file-schema.xml")
    tree = ET.parse(xml_file)
    root = tree.getroot()

    # Namespace handling
    ns = {"qes": "http://www.quantum-espresso.org/ns/qes/qes-1.0"}

    band_structure = root.find(".//qes:band_structure", ns)
    if band_structure is None:
        # Try without namespace
        band_structure = root.find(".//band_structure")

    nk = int(band_structure.find("nks").text)
    nbnd = int(band_structure.find("nbnd").text)

    eigenvalues = np.zeros((nk, nbnd))
    kpoints = np.zeros((nk, 3))

    ks_energies = band_structure.findall("ks_energies")
    for ik, ks in enumerate(ks_energies):
        k_point = ks.find("k_point")
        kpoints[ik] = [float(x) for x in k_point.text.split()]

        eig_text = ks.find("eigenvalues").text
        eigenvalues[ik] = [float(x) for x in eig_text.split()]

    # Convert from Hartree to eV
    Ha_to_eV = 27.211386245988
    eigenvalues *= Ha_to_eV

    return eigenvalues, kpoints


# ------------------------------------------------------------------
# Main workflow (call after SCF is done)
# ------------------------------------------------------------------
def berry_curvature_workflow(work_dir, prefix, nkx=40, nky=40, kz=0.0,
                              n_occ=None, ef=None, nproc=4):
    """
    Full Berry curvature calculation workflow.

    Parameters
    ----------
    work_dir : str
    prefix : str
    nkx, nky : int
    kz : float
    n_occ : int or None
    ef : float or None (eV)
    nproc : int
    """
    os.chdir(work_dir)

    # Step 1: Generate NSCF input
    nk_total = generate_2d_kgrid_nscf(
        prefix, nkx, nky, kz, "nscf_berry.in", "nscf.in"
    )

    # Step 2: Ensure nosym and write_unk are set
    # (User should have these in the template)

    # Step 3: Run NSCF
    run_nscf_and_extract("nscf_berry.in", nproc)

    # Step 4: Parse eigenvalues
    eigenvalues, kpoints = parse_eigenvalues_from_xml(prefix)
    print(f"Parsed {eigenvalues.shape[0]} k-points, {eigenvalues.shape[1]} bands")

    # Step 5: Determine occupation
    if n_occ is None and ef is not None:
        # Count occupied bands at each k-point
        n_occ = int(np.median(np.sum(eigenvalues < ef, axis=1)))
        print(f"Occupied bands (from E_F={ef} eV): {n_occ}")

    print(f"\nBerry curvature calculation requires wavefunctions.")
    print(f"Use pw2wannier90 with write_unk=.true. to extract UNK files,")
    print(f"then call berry_curvature_from_qe() from this module.")

    return eigenvalues, kpoints


print("\nWorkflow functions defined. See individual function docstrings for usage.")
```

## Key Parameters

| Parameter | Typical Value | Notes |
|-----------|--------------|-------|
| k-grid for Berry curvature | 50x50x50 to 200x200x200 | Very dense grid needed; Berry curvature can be highly peaked |
| `berry_kmesh` in Wannier90 | 50 50 50 minimum | Production: 100 100 100 or denser |
| Band selection (`n_occ`) | Count occupied bands | Must be correct; check band structure |
| SOC | Required for AHC in most systems | `noncolin`, `lspinorb` |
| Wannier spread convergence | < 1 Angstrom^2 | Poor Wannier functions give noisy Berry curvature |
| Disentanglement windows | Must include all relevant bands | Frozen window should cover occupied manifold |
| `nosym = .true.` | Always for Wannier90 | Symmetry must be off in NSCF |
| Smearing for AHC integral | 0.01-0.1 eV (adaptive) | Too large smears features; too small needs denser k-grid |

## Interpreting Results

**Berry curvature heatmap:**
- Hot spots indicate band crossings, avoided crossings, or Weyl points
- For a Chern insulator, Berry curvature integrates to an integer (Chern number)
- Singular points correspond to band degeneracies (Berry curvature monopoles)

**Chern number:**
- Integer value (within numerical precision ~0.01)
- C = 0: trivial (but may still have nonzero AHC if time-reversal is broken)
- C != 0: quantum anomalous Hall insulator
- Quantized Hall conductance: sigma_xy = C * e^2/h

**Anomalous Hall conductivity:**
- Typical values for 3d ferromagnets: 100-1000 Ohm^-1 cm^-1
- bcc Fe: ~750 Ohm^-1 cm^-1 (experiment: 750 +/- 100)
- fcc Ni: ~-2100 Ohm^-1 cm^-1
- hcp Co: ~480 Ohm^-1 cm^-1
- Sign depends on magnetization direction convention

**Convergence check:**
- Increase k-grid density until AHC changes by < 5%
- Berry curvature distribution should be smooth (not spiky from under-sampling)
- Wannier interpolation converges much faster than direct k-grid integration

## Common Issues

1. **Berry curvature is extremely noisy:**
   - k-grid too coarse (most common issue)
   - Wannier functions poorly converged (check spread)
   - Band crossings near Fermi level cause sharp features -- use denser grid near those k-points

2. **Chern number is not an integer:**
   - k-grid too coarse for the FHS method
   - Gap closes somewhere in the BZ (system is metallic on that k-slice)
   - Band selection is wrong (crossing occupied/unoccupied boundary)

3. **AHC value disagrees with experiment:**
   - Intrinsic AHC only; experiment includes extrinsic (skew-scattering, side-jump)
   - Exchange splitting (hence magnetization) must be well-converged
   - SOC strength sensitive to pseudopotential choice
   - k-grid convergence is critical

4. **Wannier90 berry calculation fails:**
   - Check that `berry = true` and `berry_task = ahc` are set
   - Verify `fermi_energy` is correct (get from SCF output)
   - `berry_kmesh` must be specified

5. **Memory issues with dense k-grids:**
   - Use Wannier interpolation (O(N_wann^3) per k-point) instead of direct DFT
   - Wannier90's internal AHC module is memory-efficient
   - For custom Python code, process k-points in chunks
