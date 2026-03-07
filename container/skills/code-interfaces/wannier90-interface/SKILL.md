# Wannier90 Interface for Maximally Localized Wannier Functions

## When to Use

- You need maximally localized Wannier functions (MLWFs) from QE or VASP DFT calculations.
- You want Wannier-interpolated band structures (smooth, dense bands from coarse DFT grid).
- You need a tight-binding Hamiltonian in the Wannier basis for model building or transport.
- You want to compute Berry phase or the Z2 topological invariant.
- You are preparing input for WannierTools (topological analysis) or EPW (electron-phonon coupling).
- QE: uses `pw2wannier90.x` interface. VASP: uses `LWANNIER90 = .TRUE.`.

## Method Selection

| Approach | When to Use | Code Interface |
|---|---|---|
| Isolated bands (no disentanglement) | Target bands are separated by gaps at all k-points (e.g., valence bands of insulator) | QE or VASP |
| Entangled bands (disentanglement) | Target bands overlap with other bands (e.g., d-bands in metals) | QE or VASP |
| Berry phase / polarization | Computing electric polarization via geometric phase | QE (Berry phase module) or Wannier90 |
| Z2 topological invariant | Classifying topological insulators | Wannier90 + WannierTools |

## Prerequisites

- QE binaries: `pw.x`, `pw2wannier90.x` in `/opt/qe/bin/`.
- Wannier90: `wannier90.x` binary. If not available, install via conda:
  ```bash
  conda install -c conda-forge wannier90
  ```
- For VASP: input preparation only (VASP not in container). VASP must be compiled with Wannier90 support (`-DVASP_WANNIER90`).
- Python: `numpy`, `matplotlib`, `pymatgen` (pre-installed).

---

## Detailed Steps

### Part 1: QE + Wannier90 Workflow

The complete workflow:
```
SCF (pw.x) -> NSCF uniform grid (pw.x) -> wannier90.x -pp -> pw2wannier90.x -> wannier90.x
```

#### Step 1: Generate All Input Files

```python
#!/usr/bin/env python3
"""
Generate complete QE + Wannier90 input files for MLWF calculation.
Example: Silicon with sp3 Wannier functions.
"""
import os
import numpy as np
from pymatgen.core.structure import Structure, Lattice
from pymatgen.symmetry.bandstructure import HighSymmKpath

# ============================================================
# CONFIGURATION
# ============================================================

WORK_DIR = "/tmp/wannier90"
PSEUDO_DIR = os.path.join(WORK_DIR, "pseudo")
os.makedirs(PSEUDO_DIR, exist_ok=True)

# Structure: Silicon diamond
structure = Structure(
    lattice=Lattice.from_parameters(5.431, 5.431, 5.431, 90, 90, 90),
    species=["Si"] * 8,
    coords=[
        [0.00, 0.00, 0.00], [0.50, 0.50, 0.00],
        [0.50, 0.00, 0.50], [0.00, 0.50, 0.50],
        [0.25, 0.25, 0.25], [0.75, 0.75, 0.25],
        [0.75, 0.25, 0.75], [0.25, 0.75, 0.75],
    ],
)

PREFIX = "silicon"
ECUTWFC = 60.0
ECUTRHO = 480.0
K_SCF = (8, 8, 8)
K_NSCF = (6, 6, 6)        # Uniform grid for Wannier90 (must match win file)
NBND = 16                  # Include enough bands

# Wannier90 parameters
NUM_WANN = 8               # Number of Wannier functions (= number of target bands)
NUM_BANDS = 16             # Total bands in DFT for disentanglement window
# For isolated bands (e.g., 4 valence bands of Si):
# NUM_WANN = 4, no disentanglement needed
# For entangled bands (e.g., sp3 of Si including conduction):
# NUM_WANN = 8, use dis_win_min/max

# Initial projections
PROJECTIONS = [
    "Si: sp3",             # sp3 hybrid orbitals centered on Si atoms
]

# Energy windows for disentanglement (eV, relative to Fermi)
DIS_WIN_MIN = -100.0       # Include all bands below this
DIS_WIN_MAX = 17.0         # Outer window upper bound
DIS_FROZ_MIN = -100.0      # Frozen window lower bound
DIS_FROZ_MAX = 6.5         # Frozen window upper bound (bands inside are kept exactly)

# ============================================================
# WRITE SCF INPUT
# ============================================================

elements = sorted(set(str(sp) for sp in structure.species))
pseudo_map = {el: f"{el}.pbe-n-kjpaw_psl.1.0.0.UPF" for el in elements}

def write_qe(filename, calc, kgrid, nosym=False):
    lines = []
    lines.append("&CONTROL")
    lines.append(f"    calculation = '{calc}'")
    lines.append(f"    prefix      = '{PREFIX}'")
    lines.append(f"    outdir      = './tmp'")
    lines.append(f"    pseudo_dir  = '{PSEUDO_DIR}'")
    lines.append(f"    tprnfor     = .true.")
    lines.append(f"    tstress     = .true.")
    lines.append("/\n")
    lines.append("&SYSTEM")
    lines.append(f"    ibrav       = 0")
    lines.append(f"    nat         = {len(structure)}")
    lines.append(f"    ntyp        = {len(elements)}")
    lines.append(f"    ecutwfc     = {ECUTWFC}")
    lines.append(f"    ecutrho     = {ECUTRHO}")
    lines.append(f"    occupations = 'smearing'")
    lines.append(f"    smearing    = 'cold'")
    lines.append(f"    degauss     = 0.01")
    lines.append(f"    nbnd        = {NBND}")
    if nosym:
        lines.append(f"    nosym       = .true.")
        lines.append(f"    noinv       = .true.")
    lines.append("/\n")
    lines.append("&ELECTRONS")
    lines.append(f"    conv_thr    = 1.0d-10")
    lines.append("/\n")
    lines.append("ATOMIC_SPECIES")
    for el in elements:
        from pymatgen.core.periodic_table import Element as PTE
        mass = PTE(el).atomic_mass
        lines.append(f"  {el:4s} {float(mass):10.4f}  {pseudo_map[el]}")
    lines.append("")
    lines.append("CELL_PARAMETERS angstrom")
    for row in structure.lattice.matrix:
        lines.append(f"  {row[0]:16.10f} {row[1]:16.10f} {row[2]:16.10f}")
    lines.append("")
    lines.append("ATOMIC_POSITIONS crystal")
    for site in structure:
        fc = site.frac_coords
        lines.append(f"  {str(site.specie):4s} {fc[0]:16.10f} {fc[1]:16.10f} {fc[2]:16.10f}")
    lines.append("")
    lines.append("K_POINTS automatic")
    lines.append(f"  {kgrid[0]} {kgrid[1]} {kgrid[2]}  0 0 0")

    filepath = os.path.join(WORK_DIR, filename)
    with open(filepath, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"Written: {filepath}")

write_qe("scf.in", "scf", K_SCF, nosym=False)
write_qe("nscf.in", "nscf", K_NSCF, nosym=True)

# ============================================================
# WRITE WANNIER90 INPUT (.win file)
# ============================================================

# Get high-symmetry k-path
kpath = HighSymmKpath(structure)
path = kpath.kpath["path"]
kpoints_dict = kpath.kpath["kpoints"]

# Build k-path for Wannier90 band interpolation
kpath_lines = []
for segment in path:
    for i in range(len(segment) - 1):
        k1 = kpoints_dict[segment[i]]
        k2 = kpoints_dict[segment[i + 1]]
        l1 = segment[i].replace("\\Gamma", "G").replace("GAMMA", "G")
        l2 = segment[i + 1].replace("\\Gamma", "G").replace("GAMMA", "G")
        kpath_lines.append(
            f"  {k1[0]:8.5f} {k1[1]:8.5f} {k1[2]:8.5f}  "
            f"{k2[0]:8.5f} {k2[1]:8.5f} {k2[2]:8.5f}   {l1}  {l2}"
        )

# Generate uniform k-point grid for Wannier90 (must match NSCF)
nk1, nk2, nk3 = K_NSCF
kpoints_list = []
for i in range(nk1):
    for j in range(nk2):
        for k in range(nk3):
            kpoints_list.append(
                f"  {i/nk1:12.8f} {j/nk2:12.8f} {k/nk3:12.8f}"
            )

win_content = f"""! Wannier90 input for {structure.composition.reduced_formula}
! Generated by MatClaw

num_wann   = {NUM_WANN}
num_bands  = {NUM_BANDS}

! Disentanglement windows (eV)
dis_win_min   = {DIS_WIN_MIN}
dis_win_max   = {DIS_WIN_MAX}
dis_froz_min  = {DIS_FROZ_MIN}
dis_froz_max  = {DIS_FROZ_MAX}
dis_num_iter  = 200
dis_conv_tol  = 1.0e-10

! Wannierization
num_iter      = 500
conv_tol      = 1.0e-10
conv_window   = 5

! Band interpolation
bands_plot       = .true.
bands_num_points = 100

begin kpoint_path
{chr(10).join(kpath_lines)}
end kpoint_path

! Initial projections
begin projections
{chr(10).join(PROJECTIONS)}
end projections

! Output
write_hr       = .true.
write_xyz      = .true.

! Unit cell
begin unit_cell_cart
Ang
  {structure.lattice.matrix[0][0]:16.10f} {structure.lattice.matrix[0][1]:16.10f} {structure.lattice.matrix[0][2]:16.10f}
  {structure.lattice.matrix[1][0]:16.10f} {structure.lattice.matrix[1][1]:16.10f} {structure.lattice.matrix[1][2]:16.10f}
  {structure.lattice.matrix[2][0]:16.10f} {structure.lattice.matrix[2][1]:16.10f} {structure.lattice.matrix[2][2]:16.10f}
end unit_cell_cart

begin atoms_frac
"""

for site in structure:
    fc = site.frac_coords
    win_content += f"  {str(site.specie):4s} {fc[0]:16.10f} {fc[1]:16.10f} {fc[2]:16.10f}\n"

win_content += f"""end atoms_frac

! K-point mesh
mp_grid = {nk1} {nk2} {nk3}

begin kpoints
{chr(10).join(kpoints_list)}
end kpoints
"""

win_path = os.path.join(WORK_DIR, f"{PREFIX}.win")
with open(win_path, "w") as f:
    f.write(win_content)
print(f"Written: {win_path}")

# ============================================================
# WRITE PW2WANNIER90 INPUT
# ============================================================

pw2wan_content = f"""&INPUTPP
    outdir    = './tmp'
    prefix    = '{PREFIX}'
    seedname  = '{PREFIX}'
    write_mmn = .true.
    write_amn = .true.
    write_unk = .false.
/
"""

pw2wan_path = os.path.join(WORK_DIR, "pw2wan.in")
with open(pw2wan_path, "w") as f:
    f.write(pw2wan_content)
print(f"Written: {pw2wan_path}")

# ============================================================
# PRINT RUN COMMANDS
# ============================================================

nprocs = os.cpu_count() or 4
print(f"\n=== Run Commands (in {WORK_DIR}) ===")
print(f"cd {WORK_DIR}")
print(f"")
print(f"# Step 1: SCF")
print(f"mpirun --allow-run-as-root -np {nprocs} pw.x < scf.in > scf.out 2>&1")
print(f"")
print(f"# Step 2: NSCF on uniform grid (nosym)")
print(f"mpirun --allow-run-as-root -np {nprocs} pw.x < nscf.in > nscf.out 2>&1")
print(f"")
print(f"# Step 3: Wannier90 preprocessing (generates .nnkp)")
print(f"wannier90.x -pp {PREFIX}")
print(f"")
print(f"# Step 4: pw2wannier90 (generates .mmn and .amn)")
print(f"mpirun --allow-run-as-root -np {nprocs} pw2wannier90.x < pw2wan.in > pw2wan.out 2>&1")
print(f"")
print(f"# Step 5: Wannier90 (Wannierization + band interpolation)")
print(f"wannier90.x {PREFIX}")
```

#### Step 2: Run the Workflow

```bash
#!/bin/bash
# wannier90_workflow.sh -- Complete QE + Wannier90 workflow
set -e

WORK_DIR=/tmp/wannier90
cd $WORK_DIR
NP=$(nproc)
PREFIX=silicon

# Step 1: SCF
echo "--- SCF ---"
mpirun --allow-run-as-root -np $NP pw.x < scf.in > scf.out 2>&1
grep "convergence has been achieved" scf.out && echo "SCF converged."

# Step 2: NSCF
echo "--- NSCF ---"
mpirun --allow-run-as-root -np $NP pw.x < nscf.in > nscf.out 2>&1
echo "NSCF completed."

# Step 3: Wannier90 preprocessing
echo "--- Wannier90 -pp ---"
wannier90.x -pp $PREFIX
echo "Generated ${PREFIX}.nnkp"

# Step 4: pw2wannier90
echo "--- pw2wannier90 ---"
mpirun --allow-run-as-root -np $NP pw2wannier90.x < pw2wan.in > pw2wan.out 2>&1
echo "Generated .mmn and .amn files"

# Step 5: Wannierization
echo "--- Wannier90 ---"
wannier90.x $PREFIX
echo "Wannierization complete."
echo "Outputs: ${PREFIX}_band.dat, ${PREFIX}_hr.dat, ${PREFIX}.wout"
```

#### Step 3: Post-Process Wannier90 Output

```python
#!/usr/bin/env python3
"""
Post-process Wannier90 output: parse and plot Wannier-interpolated bands,
compare with DFT bands, analyze spread, extract hopping parameters.
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

WORK_DIR = "/tmp/wannier90"
PREFIX = "silicon"

# ============================================================
# PARSE WANNIER90 BAND STRUCTURE
# ============================================================

band_file = os.path.join(WORK_DIR, f"{PREFIX}_band.dat")
if not os.path.exists(band_file):
    print(f"ERROR: {band_file} not found. Run wannier90.x first.")
    exit(1)

# Parse band data (format: k-distance  energy)
data = np.loadtxt(band_file)
# Each band is separated by an empty line -- split at NaN or reload differently

# Alternative: parse segment by segment
with open(band_file) as f:
    lines = f.readlines()

kpoints_w = []
energies_w = []
current_k = []
current_e = []

for line in lines:
    line = line.strip()
    if not line:
        if current_k:
            kpoints_w.append(np.array(current_k))
            energies_w.append(np.array(current_e))
            current_k = []
            current_e = []
        continue
    parts = line.split()
    current_k.append(float(parts[0]))
    current_e.append(float(parts[1]))

if current_k:
    kpoints_w.append(np.array(current_k))
    energies_w.append(np.array(current_e))

n_bands = len(kpoints_w)
print(f"Wannier bands: {n_bands} bands")

# ============================================================
# PARSE WANNIER90 SPREAD
# ============================================================

wout_file = os.path.join(WORK_DIR, f"{PREFIX}.wout")
if os.path.exists(wout_file):
    with open(wout_file) as f:
        wout = f.read()

    # Find final spread
    spreads = []
    centers = []
    for line in wout.split("\n"):
        if "Final State" in line and "WF centre" in line:
            # Parse: WF centre = (x, y, z), Spread = X.XXX
            parts = line.split("Spread =")
            if len(parts) == 2:
                spread = float(parts[1].strip().split()[0])
                spreads.append(spread)
                # Parse center
                center_str = line.split("(")[1].split(")")[0]
                center = [float(x.strip(",")) for x in center_str.split()]
                centers.append(center)

    if spreads:
        print(f"\nWannier Function Spreads:")
        print(f"  {'WF':>4} {'Center (A)':>30} {'Spread (A^2)':>14}")
        for i, (c, s) in enumerate(zip(centers, spreads)):
            print(f"  {i+1:>4} ({c[0]:8.4f}, {c[1]:8.4f}, {c[2]:8.4f}) {s:14.6f}")
        print(f"  Total spread: {sum(spreads):.6f} A^2")
        print(f"  Average spread: {np.mean(spreads):.6f} A^2")

# ============================================================
# PLOT WANNIER-INTERPOLATED BAND STRUCTURE
# ============================================================

fig, ax = plt.subplots(figsize=(8, 6))

for i in range(n_bands):
    ax.plot(kpoints_w[i], energies_w[i], "b-", linewidth=1.0,
            label="Wannier" if i == 0 else None)

ax.set_xlabel("k-path", fontsize=13)
ax.set_ylabel("Energy (eV)", fontsize=13)
ax.set_title("Wannier-Interpolated Band Structure", fontsize=14)
ax.axhline(y=0, color="gray", linestyle="--", linewidth=0.5, label="E_F")
ax.legend(fontsize=10)
ax.grid(True, alpha=0.2)
plt.tight_layout()
plt.savefig(os.path.join(WORK_DIR, "wannier_bands.png"), dpi=150, bbox_inches="tight")
plt.close()
print(f"\nSaved: {WORK_DIR}/wannier_bands.png")

# ============================================================
# PARSE HAMILTONIAN (hr.dat) FOR HOPPING PARAMETERS
# ============================================================

hr_file = os.path.join(WORK_DIR, f"{PREFIX}_hr.dat")
if os.path.exists(hr_file):
    with open(hr_file) as f:
        hr_lines = f.readlines()

    # Line 1: comment
    # Line 2: num_wann
    # Line 3: num_rpts
    # Next lines: degeneracy weights
    # Then: R1 R2 R3 m n Re(H) Im(H)

    num_wann = int(hr_lines[1].strip())
    num_rpts = int(hr_lines[2].strip())

    print(f"\nHamiltonian (hr.dat):")
    print(f"  num_wann = {num_wann}")
    print(f"  num_rpts = {num_rpts}")

    # Parse degeneracy weights
    n_deg_lines = (num_rpts + 14) // 15  # 15 values per line
    deg_weights = []
    for i in range(3, 3 + n_deg_lines):
        deg_weights.extend([int(x) for x in hr_lines[i].split()])

    # Parse hopping parameters
    data_start = 3 + n_deg_lines
    hoppings = []
    for line in hr_lines[data_start:]:
        parts = line.split()
        if len(parts) == 7:
            R1, R2, R3 = int(parts[0]), int(parts[1]), int(parts[2])
            m, n = int(parts[3]), int(parts[4])
            re_h, im_h = float(parts[5]), float(parts[6])
            hoppings.append({
                "R": (R1, R2, R3), "m": m, "n": n,
                "H": complex(re_h, im_h)
            })

    # Print on-site energies (R = 0,0,0)
    print("\n  On-site energies (R=0,0,0, diagonal):")
    for h in hoppings:
        if h["R"] == (0, 0, 0) and h["m"] == h["n"]:
            print(f"    WF {h['m']}: {h['H'].real:.6f} eV")

    # Print nearest-neighbor hoppings
    print("\n  Nearest-neighbor hoppings (largest |H|, R != (0,0,0)):")
    nn_hops = [h for h in hoppings if h["R"] != (0, 0, 0)]
    nn_hops.sort(key=lambda x: abs(x["H"]), reverse=True)
    for h in nn_hops[:10]:
        print(f"    R={h['R']}, {h['m']}->{h['n']}: "
              f"Re={h['H'].real:.6f}, Im={h['H'].imag:.6f} eV, "
              f"|H|={abs(h['H']):.6f} eV")
```

### Part 2: Berry Phase and Z2 Topological Invariant

#### Berry Phase Calculation

```python
#!/usr/bin/env python3
"""
Compute Berry phase using Wannier90 output.
The Berry phase along a closed loop in k-space is related to
the electric polarization (modern theory of polarization).
"""
import numpy as np
import os

WORK_DIR = "/tmp/wannier90"
PREFIX = "silicon"

# ============================================================
# METHOD 1: Berry phase from QE (berry_phase.x)
# ============================================================

# QE has a built-in Berry phase module. Generate input:
berry_input = f"""&INPUTPP
    outdir    = './tmp'
    prefix    = '{PREFIX}'
    nppstr    = 7
    lberry    = .true.
    gdir      = 3
/
"""

berry_path = os.path.join(WORK_DIR, "berry.in")
with open(berry_path, "w") as f:
    f.write(berry_input)

print("Berry phase via QE:")
print(f"  Input: {berry_path}")
print(f"  Run: pw.x < berry.in > berry.out")
print(f"  Note: This uses the SCF calculation with lberry=.true.")
print(f"  Alternatively, use the separate 'bp' module of QE.")

# ============================================================
# METHOD 2: Berry phase from Wannier90 hr.dat
# ============================================================

print("\nBerry phase from Wannier90:")
print("  Wannier90 writes the tight-binding Hamiltonian H(R) in hr.dat.")
print("  The Berry phase can be computed by integrating the Berry connection:")
print("    phi_n = -Im integral <u_nk | nabla_k | u_nk> dk")
print("  In the Wannier basis, this reduces to computing the Wilson loop.")

# Wilson loop calculation
def compute_wilson_loop(hr_file, num_wann, nk=100, direction=2):
    """
    Compute the Wilson loop eigenvalues along a specified direction.
    The phases of these eigenvalues give the Wannier charge centers.

    Parameters
    ----------
    hr_file : str
        Path to wannier90_hr.dat.
    num_wann : int
        Number of Wannier functions.
    nk : int
        Number of k-points along the loop.
    direction : int
        Direction for the Wilson loop (0=x, 1=y, 2=z).
    """
    # Parse hr.dat
    with open(hr_file) as f:
        lines = f.readlines()

    num_wann_file = int(lines[1].strip())
    num_rpts = int(lines[2].strip())
    n_deg_lines = (num_rpts + 14) // 15
    deg_weights = []
    for i in range(3, 3 + n_deg_lines):
        deg_weights.extend([int(x) for x in lines[i].split()])

    # Build H(R) dictionary
    H_R = {}
    data_start = 3 + n_deg_lines
    for line in lines[data_start:]:
        parts = line.split()
        if len(parts) == 7:
            R = (int(parts[0]), int(parts[1]), int(parts[2]))
            m, n = int(parts[3]) - 1, int(parts[4]) - 1  # 0-indexed
            h = complex(float(parts[5]), float(parts[6]))
            if R not in H_R:
                H_R[R] = np.zeros((num_wann_file, num_wann_file), dtype=complex)
            H_R[R][m, n] = h

    # Compute H(k) and eigenstates along the loop
    k_loop = np.zeros((nk, 3))
    k_loop[:, direction] = np.linspace(0, 1, nk, endpoint=False)

    # Wilson loop = product of overlap matrices
    W = np.eye(num_wann_file, dtype=complex)

    eigvecs_prev = None
    for ik in range(nk):
        k = k_loop[ik]

        # Fourier transform: H(k) = sum_R H(R) * exp(2pi*i*k.R)
        Hk = np.zeros((num_wann_file, num_wann_file), dtype=complex)
        for R, HR in H_R.items():
            phase = np.exp(2j * np.pi * np.dot(k, R))
            Hk += HR * phase

        # Diagonalize
        eigvals, eigvecs = np.linalg.eigh(Hk)

        if eigvecs_prev is not None:
            # Overlap matrix
            S = eigvecs_prev.conj().T @ eigvecs
            W = W @ S

        eigvecs_prev = eigvecs

    # Close the loop
    k_0 = k_loop[0]
    Hk0 = np.zeros((num_wann_file, num_wann_file), dtype=complex)
    for R, HR in H_R.items():
        phase = np.exp(2j * np.pi * np.dot(k_0, R))
        Hk0 += HR * phase
    _, eigvecs_0 = np.linalg.eigh(Hk0)
    S = eigvecs_prev.conj().T @ eigvecs_0
    W = W @ S

    # Wilson loop eigenvalues
    wl_eigvals = np.linalg.eigvals(W)
    phases = np.angle(wl_eigvals) / (2 * np.pi)  # Wannier charge centers

    return phases, wl_eigvals


hr_file = os.path.join(WORK_DIR, f"{PREFIX}_hr.dat")
if os.path.exists(hr_file):
    phases, _ = compute_wilson_loop(hr_file, 8, nk=100, direction=2)
    print(f"\nWilson loop phases (Wannier charge centers):")
    for i, p in enumerate(sorted(phases)):
        print(f"  WCC {i+1}: {p:.6f}")
    print(f"  Sum of phases: {sum(phases):.6f}")
    print(f"  Berry phase: {sum(phases) * 2 * np.pi:.6f} rad")
```

#### Z2 Topological Invariant

```python
#!/usr/bin/env python3
"""
Compute Z2 topological invariant from Wannier charge centers (WCC).
Uses the Wilson loop approach with Wannier90 output.
"""
import numpy as np
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

WORK_DIR = "/tmp/wannier90"
PREFIX = "silicon"

def compute_z2_from_wcc(hr_file, num_wann, nk_loop=50, nk_perp=50, direction=2):
    """
    Compute Z2 invariant by tracking WCC evolution across half the BZ.

    The Z2 invariant is determined by counting the number of times
    the WCC lines cross an arbitrary reference line as k_perp goes
    from 0 to pi/a (half the BZ).

    Parameters
    ----------
    hr_file : str
        Path to wannier90_hr.dat.
    num_wann : int
        Number of Wannier functions.
    nk_loop : int
        Number of k-points along the Wilson loop direction.
    nk_perp : int
        Number of k-points perpendicular to the loop (0 to 0.5).
    direction : int
        Wilson loop direction (0=x, 1=y, 2=z).

    Returns
    -------
    z2 : int
        Z2 invariant (0 = trivial, 1 = topological).
    wcc_evolution : array
        WCC phases at each k_perp value.
    """
    # Parse hr.dat
    with open(hr_file) as f:
        lines = f.readlines()

    num_wann_file = int(lines[1].strip())
    num_rpts = int(lines[2].strip())
    n_deg_lines = (num_rpts + 14) // 15
    H_R = {}
    data_start = 3 + n_deg_lines
    for line in lines[data_start:]:
        parts = line.split()
        if len(parts) == 7:
            R = (int(parts[0]), int(parts[1]), int(parts[2]))
            m, n = int(parts[3]) - 1, int(parts[4]) - 1
            h = complex(float(parts[5]), float(parts[6]))
            if R not in H_R:
                H_R[R] = np.zeros((num_wann_file, num_wann_file), dtype=complex)
            H_R[R][m, n] = h

    # Perpendicular direction
    perp_dir = (direction + 1) % 3

    # Compute WCC at each k_perp
    k_perp_values = np.linspace(0, 0.5, nk_perp)
    wcc_evolution = []

    for k_perp in k_perp_values:
        # Build k-points along the loop
        k_loop = np.zeros((nk_loop, 3))
        k_loop[:, direction] = np.linspace(0, 1, nk_loop, endpoint=False)
        k_loop[:, perp_dir] = k_perp

        # Compute Wilson loop
        W = np.eye(num_wann_file, dtype=complex)
        eigvecs_prev = None

        for ik in range(nk_loop):
            k = k_loop[ik]
            Hk = np.zeros((num_wann_file, num_wann_file), dtype=complex)
            for R, HR in H_R.items():
                phase = np.exp(2j * np.pi * np.dot(k, R))
                Hk += HR * phase

            eigvals, eigvecs = np.linalg.eigh(Hk)
            # Take occupied bands only (num_wann // 2 for time-reversal pairs)
            n_occ = num_wann_file // 2
            eigvecs_occ = eigvecs[:, :n_occ]

            if eigvecs_prev is not None:
                S = eigvecs_prev.conj().T @ eigvecs_occ
                W_occ = W[:n_occ, :n_occ] @ S
                W = np.eye(num_wann_file, dtype=complex)
                W[:n_occ, :n_occ] = W_occ

            eigvecs_prev = eigvecs_occ

        # Close the loop
        k0 = k_loop[0]
        Hk0 = np.zeros((num_wann_file, num_wann_file), dtype=complex)
        for R, HR in H_R.items():
            Hk0 += HR * np.exp(2j * np.pi * np.dot(k0, R))
        _, ev0 = np.linalg.eigh(Hk0)
        S = eigvecs_prev.conj().T @ ev0[:, :n_occ]
        W_final = W[:n_occ, :n_occ] @ S

        wl_eigvals = np.linalg.eigvals(W_final)
        phases = np.sort(np.angle(wl_eigvals) / (2 * np.pi)) % 1.0
        wcc_evolution.append(phases)

    wcc_evolution = np.array(wcc_evolution)

    # Count crossings of a reference line (at 0.5)
    ref = 0.5
    crossings = 0
    for i in range(len(k_perp_values) - 1):
        for band in range(wcc_evolution.shape[1]):
            w1 = wcc_evolution[i, band]
            w2 = wcc_evolution[i + 1, band]
            if (w1 < ref and w2 >= ref) or (w1 >= ref and w2 < ref):
                crossings += 1

    z2 = crossings % 2

    return z2, wcc_evolution, k_perp_values


hr_file = os.path.join(WORK_DIR, f"{PREFIX}_hr.dat")
if os.path.exists(hr_file):
    z2, wcc_evo, k_perp = compute_z2_from_wcc(hr_file, 8, nk_loop=50, nk_perp=30)

    print(f"\nZ2 topological invariant: {z2}")
    print(f"  (0 = trivial insulator, 1 = topological insulator)")

    # Plot WCC evolution
    fig, ax = plt.subplots(figsize=(6, 5))
    for band in range(wcc_evo.shape[1]):
        ax.plot(k_perp, wcc_evo[:, band], "bo", markersize=3)
    ax.axhline(y=0.5, color="r", linestyle="--", linewidth=1, label="Reference line")
    ax.set_xlabel(r"$k_\perp$ (2$\pi$/a)", fontsize=13)
    ax.set_ylabel("WCC phase", fontsize=13)
    ax.set_title(f"Wannier Charge Center Evolution (Z2 = {z2})", fontsize=14)
    ax.set_xlim(0, 0.5)
    ax.set_ylim(0, 1)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(WORK_DIR, "z2_wcc.png"), dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {WORK_DIR}/z2_wcc.png")
else:
    print(f"  hr.dat not found. Run Wannier90 first.")
```

### Part 3: VASP + Wannier90 (Input Generation)

```python
#!/usr/bin/env python3
"""
Generate VASP input files for Wannier90 interface (LWANNIER90 = .TRUE.).
VASP must be compiled with Wannier90 support.
"""
import os
from pymatgen.core.structure import Structure
from pymatgen.io.vasp import Poscar, Incar, Kpoints
from pymatgen.symmetry.bandstructure import HighSymmKpath

STRUCTURE_FILE = "POSCAR"
OUTPUT_DIR = "./vasp_wannier"
os.makedirs(OUTPUT_DIR, exist_ok=True)

structure = Structure.from_file(STRUCTURE_FILE)

# ============================================================
# Step 1: SCF INCAR
# ============================================================

scf_dir = os.path.join(OUTPUT_DIR, "scf")
os.makedirs(scf_dir, exist_ok=True)

scf_incar = Incar({
    "SYSTEM": f"{structure.composition.reduced_formula} SCF",
    "ENCUT": 520,
    "PREC": "Accurate",
    "EDIFF": 1e-8,
    "ISMEAR": 0,
    "SIGMA": 0.05,
    "IBRION": -1,
    "NSW": 0,
    "LWAVE": True,
    "LCHARG": True,
    "NCORE": 4,
})
scf_incar.write_file(os.path.join(scf_dir, "INCAR"))
Kpoints.automatic_density(structure, kppa=2000).write_file(os.path.join(scf_dir, "KPOINTS"))
Poscar(structure).write_file(os.path.join(scf_dir, "POSCAR"))
print(f"SCF: {scf_dir}/")

# ============================================================
# Step 2: NSCF + LWANNIER90 INCAR
# ============================================================

wan_dir = os.path.join(OUTPUT_DIR, "wannier")
os.makedirs(wan_dir, exist_ok=True)

num_wann = len(structure) * 2  # example: 2 bands per atom

wan_incar = Incar({
    "SYSTEM": f"{structure.composition.reduced_formula} Wannier90",
    "ENCUT": 520,
    "PREC": "Accurate",
    "EDIFF": 1e-8,
    "ISMEAR": 0,
    "SIGMA": 0.05,
    "IBRION": -1,
    "NSW": 0,
    "ICHARG": 11,             # Read CHGCAR
    "NBANDS": num_wann * 2,   # Enough bands
    "LWANNIER90": True,        # Enable Wannier90 interface
    "LWRITE_MMN_AMN": True,    # Write .mmn and .amn files
    "NUM_WANN": num_wann,
    "ISYM": -1,                # No symmetry
    "NCORE": 4,
})
wan_incar.write_file(os.path.join(wan_dir, "INCAR"))
Kpoints.gamma_automatic((6, 6, 6)).write_file(os.path.join(wan_dir, "KPOINTS"))
Poscar(structure).write_file(os.path.join(wan_dir, "POSCAR"))

print(f"Wannier: {wan_dir}/")
print(f"  LWANNIER90 = .TRUE.")
print(f"  NUM_WANN = {num_wann}")
print(f"\nVASP workflow:")
print(f"  1. Run SCF in {scf_dir}/")
print(f"  2. Copy CHGCAR to {wan_dir}/")
print(f"  3. Create wannier90.win in {wan_dir}/ (see QE example for format)")
print(f"  4. Run VASP in {wan_dir}/ -- it calls wannier90 internally")
print(f"  5. Post-process with wannier90.x if needed for band interpolation")
```

## Key Parameters

| Parameter | Typical Value | Notes |
|---|---|---|
| `num_wann` | System-dependent | Number of target Wannier functions. Must match band count for isolated bands. |
| `num_bands` | >= num_wann | Total DFT bands. More bands allows wider disentanglement window. |
| `dis_win_min/max` | Energy window (eV) | Outer disentanglement window. Include all relevant bands. |
| `dis_froz_min/max` | Subset of outer window | Frozen window: bands inside are kept exactly. Key for accuracy. |
| `num_iter` | 500-2000 | Wannierization iterations. Monitor spread convergence. |
| `conv_tol` | 1e-10 | Spread convergence tolerance. |
| K-grid (NSCF) | 6x6x6 to 12x12x12 | Must be uniform, Gamma-centered. Denser = better interpolation. |
| `nosym`/`noinv` (QE) | .true. | MANDATORY for Wannier90. Full BZ without symmetry reduction. |
| `ISYM` (VASP) | -1 | Same as above for VASP. |
| `projections` | Orbital projections | Initial guess for Wannier functions. Crucial for convergence. |

## Interpreting Results

### Wannier spread
- Total spread = Omega_I (gauge-invariant) + Omega_tilde (gauge-dependent).
- Omega_I is fixed by band structure; Omega_tilde should be minimized.
- Well-localized WFs have spread comparable to bond length squared (~1-5 A^2).
- Very large spread indicates convergence problem or wrong num_wann.

### Band interpolation
- Wannier-interpolated bands should match DFT bands exactly within the frozen window.
- Outside the frozen window (disentangled region), agreement is approximate.
- Poor agreement indicates: wrong projections, insufficient k-grid, or wrong energy windows.

### Berry phase
- Total Berry phase = sum of Wannier charge centers (modulo lattice vector).
- Related to electric polarization: P = (e/V) * sum(WCC).
- Berry phase is defined modulo 2*pi.

### Z2 invariant
- Z2 = 0: trivial insulator. Z2 = 1: topological insulator (Z2 TI).
- Determined by counting WCC crossings of a reference line.
- For 3D: four Z2 indices (nu0; nu1 nu2 nu3). nu0 = "strong" TI index.

## Common Issues

| Issue | Cause | Fix |
|---|---|---|
| Wannierization does not converge | Wrong initial projections | Try different orbital projections. Use `random` projections as test. |
| Disentanglement fails | Energy windows wrong | Visualize DFT bands and set dis_win/dis_froz to bracket target bands. |
| `.nnkp` file not generated | `wannier90.x -pp` not run | Must run preprocessing step before pw2wannier90.x. |
| pw2wannier90 crashes | K-grid mismatch | NSCF k-grid must exactly match the `mp_grid` in .win file. Use nosym. |
| Interpolated bands wrong | K-grid too coarse | Increase NSCF k-grid (8x8x8 or 12x12x12). |
| Large spread that does not decrease | Too many / too few Wannier functions | num_wann must match the number of target bands. Adjust energy windows. |
| Z2 calculation gives inconsistent results | Insufficient k-points in Wilson loop | Increase nk_loop and nk_perp. Ensure SOC is included for real TI. |
| VASP LWANNIER90 error | VASP not compiled with Wannier90 | Need `-DVASP_WANNIER90` at compile time. Use QE workflow instead. |
