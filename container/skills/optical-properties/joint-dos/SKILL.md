# Joint Density of States (JDOS) Calculation

## When to Use

- You need the joint density of states to understand which energy transitions are available.
- You want to decompose optical absorption into contributions from specific valence-conduction band pairs.
- You are analyzing Van Hove singularities (critical points in the band structure).
- You want to compare the JDOS with eps2(omega) to separate matrix-element effects from density-of-states effects.
- You need partial JDOS between selected VB/CB pairs (e.g., VBM to CBM only).
- You are studying the onset of optical absorption and want to determine if it is JDOS-limited or matrix-element-limited.

## Method Selection

| Criterion | QE (from eigenvalues) | VASP (from EIGENVAL) | VASPKIT (716--717) |
|---|---|---|---|
| Total JDOS | Compute from NSCF eigenvalues on dense k-mesh | Compute from EIGENVAL | VASPKIT 716 automated |
| Partial JDOS (band-resolved) | Select specific (VB, CB) pairs from NSCF output | Select from EIGENVAL bands | VASPKIT 717 automated |
| Accuracy | Depends on k-mesh density and number of bands | Same | Same (post-processing) |
| Speed | Post-processing is fast; NSCF is the bottleneck | Post-processing is fast | Fast |

**MACE cannot compute JDOS.** It has no band structure. Always use QE or VASP.

## Prerequisites

- A relaxed crystal structure.
- For QE: Completed SCF and NSCF calculations with a dense k-grid and enough bands.
- QE executables: `pw.x`.
- For VASP (future): Converged calculation with EIGENVAL file.
- Python packages: `numpy`, `scipy`, `matplotlib`.

---

## Detailed Steps

### Background: Joint Density of States

The JDOS counts the number of vertical (direct) transitions available at each photon energy:

```
J(E) = (2 / (2*pi)^3) * integral_BZ sum_{v,c} delta(E_c(k) - E_v(k) - E) dk
```

Key properties:
- JDOS onset corresponds to the direct (optical) band gap.
- Peaks in JDOS correspond to Van Hove singularities where valence and conduction bands are parallel (dE_cv/dk = 0).
- eps2(omega) = C * |M_cv|^2 * JDOS(omega) / omega^2, so comparing eps2 with JDOS reveals the role of matrix elements |M_cv|^2.

The partial JDOS restricts the sum to specific band pairs:

```
J_{v,c}(E) = (2 / (2*pi)^3) * integral_BZ delta(E_c(k) - E_v(k) - E) dk
```

### Method A: QE -- JDOS from NSCF Eigenvalues

#### Step A1: SCF Calculation

```python
#!/usr/bin/env python3
"""
Step 1: SCF calculation for JDOS analysis.
Example: Silicon (diamond structure).
"""
import os
import subprocess

PSEUDO_DIR = os.path.abspath("./pseudo")
OUTDIR = os.path.abspath("./tmp_jdos")
os.makedirs(OUTDIR, exist_ok=True)
os.makedirs(PSEUDO_DIR, exist_ok=True)
PREFIX = "si"

scf_input = f"""&CONTROL
    calculation  = 'scf'
    prefix       = '{PREFIX}'
    outdir       = '{OUTDIR}'
    pseudo_dir   = '{PSEUDO_DIR}'
/

&SYSTEM
    ibrav        = 2
    celldm(1)    = 10.26
    nat          = 2
    ntyp         = 1
    ecutwfc      = 60.0
    ecutrho      = 240.0
    occupations  = 'fixed'
/

&ELECTRONS
    conv_thr = 1.0d-10
/

ATOMIC_SPECIES
  Si  28.0855  Si_ONCV_PBE-1.2.upf

ATOMIC_POSITIONS (crystal)
  Si  0.00  0.00  0.00
  Si  0.25  0.25  0.25

K_POINTS (automatic)
  20 20 20  0 0 0
"""
# NOTE: Dense k-grid (20x20x20) is critical for smooth JDOS.
# JDOS converges slower than total energy with k-points.

with open(f"{PREFIX}_scf.in", "w") as f:
    f.write(scf_input)

print("Running SCF...")
result = subprocess.run(
    ["mpirun", "-np", "4", "pw.x", "-in", f"{PREFIX}_scf.in"],
    capture_output=True, text=True, timeout=600
)
with open(f"{PREFIX}_scf.out", "w") as f:
    f.write(result.stdout)

assert "convergence has been achieved" in result.stdout, "SCF did not converge!"
print("SCF converged.")
```

#### Step A2: NSCF with Dense K-Grid

```python
#!/usr/bin/env python3
"""
Step 2: NSCF calculation with dense k-grid for JDOS.
The k-grid density is the most critical parameter for smooth JDOS.
"""
import os
import subprocess

PSEUDO_DIR = os.path.abspath("./pseudo")
OUTDIR = os.path.abspath("./tmp_jdos")
PREFIX = "si"
NBND = 20  # Si: 4 VB + enough CB for transitions up to ~10 eV

nscf_input = f"""&CONTROL
    calculation  = 'nscf'
    prefix       = '{PREFIX}'
    outdir       = '{OUTDIR}'
    pseudo_dir   = '{PSEUDO_DIR}'
/

&SYSTEM
    ibrav        = 2
    celldm(1)    = 10.26
    nat          = 2
    ntyp         = 1
    ecutwfc      = 60.0
    ecutrho      = 240.0
    occupations  = 'fixed'
    nbnd         = {NBND}
/

&ELECTRONS
    conv_thr = 1.0d-10
/

ATOMIC_SPECIES
  Si  28.0855  Si_ONCV_PBE-1.2.upf

ATOMIC_POSITIONS (crystal)
  Si  0.00  0.00  0.00
  Si  0.25  0.25  0.25

K_POINTS (automatic)
  20 20 20  0 0 0
"""
# NOTE: nosym/noinv not needed here since we are only reading eigenvalues,
# not running epsilon.x. Symmetry reduction speeds up the calculation.

with open(f"{PREFIX}_nscf.in", "w") as f:
    f.write(nscf_input)

print(f"Running NSCF with nbnd={NBND} on 20x20x20 k-grid...")
result = subprocess.run(
    ["mpirun", "-np", "4", "pw.x", "-in", f"{PREFIX}_nscf.in"],
    capture_output=True, text=True, timeout=1800
)
with open(f"{PREFIX}_nscf.out", "w") as f:
    f.write(result.stdout)

assert result.returncode == 0, "NSCF failed!"
print("NSCF completed.")
```

#### Step A3: Parse Eigenvalues and Compute JDOS

```python
#!/usr/bin/env python3
"""
Step 3: Parse QE NSCF output to extract eigenvalues, then compute
total and partial JDOS.
"""
import numpy as np
import re
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

PREFIX = "si"
OUTPUT_FILE = f"{PREFIX}_nscf.out"

# ── Parse eigenvalues from QE output ────────────────────────────────
def parse_qe_eigenvalues(filename):
    """
    Parse eigenvalues from QE NSCF output.

    Returns:
        kpoints: list of k-point coordinates (fractional)
        weights: list of k-point weights
        eigenvalues: array of shape (nkpts, nbands) in eV
        efermi: Fermi energy in eV
    """
    kpoints = []
    weights = []
    eigenvalues = []
    efermi = None

    with open(filename) as f:
        lines = f.readlines()

    # Find Fermi energy
    for line in lines:
        if "highest occupied" in line or "Fermi energy" in line:
            # Extract the energy value
            nums = re.findall(r"[-+]?\d*\.?\d+", line)
            if nums:
                efermi = float(nums[-1])
                break
        if "highest occupied, lowest unoccupied" in line:
            nums = re.findall(r"[-+]?\d*\.?\d+", line)
            if len(nums) >= 2:
                efermi = (float(nums[-2]) + float(nums[-1])) / 2.0
                break

    # Parse k-points and eigenvalues
    i = 0
    while i < len(lines):
        line = lines[i]

        # Match k-point line
        if "k =" in line and "bands" in line:
            # Extract k-point coordinates
            kline = line.split("k =")[1].split("(")[0].strip()
            kcoords = [float(x) for x in kline.split()]
            kpoints.append(kcoords)

            # Try to extract weight
            if "wk =" in line:
                wk = float(line.split("wk =")[1].strip())
            else:
                wk = 1.0
            weights.append(wk)

            # Skip blank line(s) and read eigenvalues
            i += 1
            while i < len(lines) and lines[i].strip() == "":
                i += 1

            bands = []
            while i < len(lines) and lines[i].strip() != "":
                vals = lines[i].split()
                try:
                    for v in vals:
                        bands.append(float(v))
                except ValueError:
                    break
                i += 1

            eigenvalues.append(bands)
        else:
            i += 1

    # Convert to numpy arrays
    nbands = min(len(b) for b in eigenvalues) if eigenvalues else 0
    eigenvalues = np.array([b[:nbands] for b in eigenvalues])
    weights = np.array(weights)

    # Normalize weights
    if weights.sum() > 0:
        weights /= weights.sum()

    print(f"Parsed {len(kpoints)} k-points, {nbands} bands")
    if efermi is not None:
        print(f"Fermi energy: {efermi:.4f} eV")

    return np.array(kpoints), weights, eigenvalues, efermi


def compute_jdos(eigenvalues, weights, n_valence, energy_range=(0, 15),
                 n_energy=1000, sigma=0.05):
    """
    Compute the total Joint Density of States.

    JDOS(E) = sum_k w_k * sum_{v=1}^{Nv} sum_{c=Nv+1}^{Nb}
              gaussian(E_c(k) - E_v(k) - E, sigma)

    Parameters:
        eigenvalues: (nkpts, nbands) array in eV
        weights: (nkpts,) k-point weights
        n_valence: number of occupied bands
        energy_range: (Emin, Emax) in eV
        n_energy: number of energy grid points
        sigma: Gaussian broadening in eV

    Returns:
        energy_grid: (n_energy,) array in eV
        jdos: (n_energy,) JDOS values
    """
    nkpts, nbands = eigenvalues.shape
    energy_grid = np.linspace(energy_range[0], energy_range[1], n_energy)
    jdos = np.zeros(n_energy)

    # Gaussian broadening function
    inv_sqrt2pi_sigma = 1.0 / (sigma * np.sqrt(2.0 * np.pi))
    inv_2sigma2 = 1.0 / (2.0 * sigma**2)

    for ik in range(nkpts):
        wk = weights[ik]
        for iv in range(n_valence):
            for ic in range(n_valence, nbands):
                dE = eigenvalues[ik, ic] - eigenvalues[ik, iv]
                if dE < energy_range[0] - 5 * sigma or dE > energy_range[1] + 5 * sigma:
                    continue
                # Add Gaussian contribution
                jdos += wk * inv_sqrt2pi_sigma * np.exp(
                    -(energy_grid - dE)**2 * inv_2sigma2
                )

    return energy_grid, jdos


def compute_partial_jdos(eigenvalues, weights, vb_indices, cb_indices,
                         energy_range=(0, 15), n_energy=1000, sigma=0.05):
    """
    Compute partial JDOS between specific VB and CB indices.

    Parameters:
        vb_indices: list of valence band indices (0-based)
        cb_indices: list of conduction band indices (0-based)

    Returns:
        energy_grid, partial_jdos
    """
    nkpts, nbands = eigenvalues.shape
    energy_grid = np.linspace(energy_range[0], energy_range[1], n_energy)
    pjdos = np.zeros(n_energy)

    inv_sqrt2pi_sigma = 1.0 / (sigma * np.sqrt(2.0 * np.pi))
    inv_2sigma2 = 1.0 / (2.0 * sigma**2)

    for ik in range(nkpts):
        wk = weights[ik]
        for iv in vb_indices:
            for ic in cb_indices:
                if iv >= nbands or ic >= nbands:
                    continue
                dE = eigenvalues[ik, ic] - eigenvalues[ik, iv]
                if dE < energy_range[0] - 5 * sigma or dE > energy_range[1] + 5 * sigma:
                    continue
                pjdos += wk * inv_sqrt2pi_sigma * np.exp(
                    -(energy_grid - dE)**2 * inv_2sigma2
                )

    return energy_grid, pjdos


# ── Main computation ────────────────────────────────────────────────
kpoints, weights, eigenvalues, efermi = parse_qe_eigenvalues(OUTPUT_FILE)
nkpts, nbands = eigenvalues.shape

# Determine number of valence bands
# For Si: 8 electrons / 2 (spin) = 4 valence bands
# Auto-detect from Fermi energy
if efermi is not None:
    avg_energies = np.mean(eigenvalues, axis=0)
    n_valence = np.sum(avg_energies < efermi + 0.1)
else:
    n_valence = 4  # Si default
print(f"Number of valence bands: {n_valence}")

# Shift eigenvalues so VBM = 0
vbm = np.max(eigenvalues[:, :n_valence])
cbm = np.min(eigenvalues[:, n_valence:])
band_gap = cbm - vbm
eigenvalues_shifted = eigenvalues - vbm
print(f"VBM = {vbm:.4f} eV, CBM = {cbm:.4f} eV")
print(f"Direct/indirect band gap: {band_gap:.4f} eV")

# Find direct gap
direct_gaps = np.min(eigenvalues[:, n_valence:] - np.max(eigenvalues[:, :n_valence], axis=1, keepdims=True), axis=1)
direct_gap = np.min(np.max(eigenvalues[:, n_valence:], axis=1) - np.max(eigenvalues[:, :n_valence], axis=1))
# More precisely:
direct_gap_per_k = eigenvalues[:, n_valence] - eigenvalues[:, n_valence - 1]
min_direct_gap = np.min(direct_gap_per_k)
print(f"Minimum direct gap (VBM band to CBM band): {min_direct_gap:.4f} eV")

# ── Compute total JDOS ──────────────────────────────────────────────
ENERGY_RANGE = (0, 12)
SIGMA = 0.05  # eV broadening
N_ENERGY = 2000

energy_grid, jdos_total = compute_jdos(
    eigenvalues_shifted, weights, n_valence,
    energy_range=ENERGY_RANGE, n_energy=N_ENERGY, sigma=SIGMA
)

print(f"\n=== Total JDOS ===")
# JDOS onset
onset_threshold = 0.01 * np.max(jdos_total)
onset_idx = np.where(jdos_total > onset_threshold)[0]
if len(onset_idx) > 0:
    print(f"JDOS onset: {energy_grid[onset_idx[0]]:.3f} eV")
print(f"JDOS peak: {energy_grid[np.argmax(jdos_total)]:.3f} eV")
print(f"JDOS peak value: {np.max(jdos_total):.4f}")

# ── Compute partial JDOS for specific band pairs ────────────────────
# VBM (band n_valence-1) -> CBM (band n_valence)
energy_grid_p, pjdos_vbm_cbm = compute_partial_jdos(
    eigenvalues_shifted, weights,
    vb_indices=[n_valence - 1], cb_indices=[n_valence],
    energy_range=ENERGY_RANGE, n_energy=N_ENERGY, sigma=SIGMA
)

# VBM-1 (band n_valence-2) -> CBM
_, pjdos_vbm1_cbm = compute_partial_jdos(
    eigenvalues_shifted, weights,
    vb_indices=[n_valence - 2], cb_indices=[n_valence],
    energy_range=ENERGY_RANGE, n_energy=N_ENERGY, sigma=SIGMA
)

# VBM -> CBM+1 (band n_valence+1)
_, pjdos_vbm_cbm1 = compute_partial_jdos(
    eigenvalues_shifted, weights,
    vb_indices=[n_valence - 1], cb_indices=[n_valence + 1],
    energy_range=ENERGY_RANGE, n_energy=N_ENERGY, sigma=SIGMA
)

# All VB -> CBM only
_, pjdos_all_cbm = compute_partial_jdos(
    eigenvalues_shifted, weights,
    vb_indices=list(range(n_valence)),
    cb_indices=[n_valence],
    energy_range=ENERGY_RANGE, n_energy=N_ENERGY, sigma=SIGMA
)

# VBM -> all CB
_, pjdos_vbm_all = compute_partial_jdos(
    eigenvalues_shifted, weights,
    vb_indices=[n_valence - 1],
    cb_indices=list(range(n_valence, nbands)),
    energy_range=ENERGY_RANGE, n_energy=N_ENERGY, sigma=SIGMA
)

print(f"\n=== Partial JDOS Analysis ===")
print(f"VBM->CBM onset: {energy_grid_p[np.where(pjdos_vbm_cbm > 0.01*np.max(pjdos_vbm_cbm))[0][0]]:.3f} eV" if np.max(pjdos_vbm_cbm) > 0 else "VBM->CBM: no transitions found")

# ── Van Hove singularity analysis ────────────────────────────────────
from scipy.signal import find_peaks

peaks, properties = find_peaks(jdos_total, height=0.1 * np.max(jdos_total),
                               distance=20, prominence=0.02 * np.max(jdos_total))

print(f"\n=== Van Hove Singularities (JDOS peaks) ===")
print(f"{'Peak':>4s}  {'Energy (eV)':>12s}  {'JDOS value':>12s}  {'Type':>10s}")
print("-" * 45)
for i, p in enumerate(peaks[:10]):
    # Classify: check derivative
    if p > 0 and p < len(jdos_total) - 1:
        d2 = jdos_total[p+1] + jdos_total[p-1] - 2*jdos_total[p]
        if d2 < -0.001:
            vhs_type = "M0/M1"
        elif d2 > 0.001:
            vhs_type = "M2/M3"
        else:
            vhs_type = "saddle"
    else:
        vhs_type = "edge"
    print(f"{i+1:4d}  {energy_grid[p]:12.3f}  {jdos_total[p]:12.4f}  {vhs_type:>10s}")

# ── Plotting ─────────────────────────────────────────────────────────

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Panel (a): Total JDOS
ax = axes[0, 0]
ax.plot(energy_grid, jdos_total, "b-", linewidth=1.5, label="Total JDOS")
ax.fill_between(energy_grid, 0, jdos_total, alpha=0.1, color="blue")
for i, p in enumerate(peaks[:5]):
    ax.annotate(f"VH{i+1}\n{energy_grid[p]:.1f}",
                xy=(energy_grid[p], jdos_total[p]),
                xytext=(0, 12), textcoords="offset points",
                ha="center", fontsize=8,
                arrowprops=dict(arrowstyle="->", color="gray", lw=0.8))
ax.set_xlabel("Energy (eV)")
ax.set_ylabel("JDOS (states/eV)")
ax.set_title("(a) Total Joint Density of States")
ax.set_xlim(*ENERGY_RANGE)
ax.grid(alpha=0.3)
ax.legend(fontsize=10)

# Panel (b): Partial JDOS (band-resolved)
ax = axes[0, 1]
ax.plot(energy_grid_p, pjdos_vbm_cbm, linewidth=1.2,
        label=f"VB{n_valence}->CB{n_valence+1}")
ax.plot(energy_grid_p, pjdos_vbm1_cbm, linewidth=1.2, linestyle="--",
        label=f"VB{n_valence-1}->CB{n_valence+1}")
ax.plot(energy_grid_p, pjdos_vbm_cbm1, linewidth=1.2, linestyle="-.",
        label=f"VB{n_valence}->CB{n_valence+2}")
ax.set_xlabel("Energy (eV)")
ax.set_ylabel("Partial JDOS (states/eV)")
ax.set_title("(b) Partial JDOS (band-pair resolved)")
ax.set_xlim(*ENERGY_RANGE)
ax.legend(fontsize=9)
ax.grid(alpha=0.3)

# Panel (c): Decomposition -- all-VB->CBM vs VBM->all-CB
ax = axes[1, 0]
ax.plot(energy_grid_p, pjdos_all_cbm, linewidth=1.2, label="All VB -> CBM")
ax.plot(energy_grid_p, pjdos_vbm_all, linewidth=1.2, linestyle="--",
        label="VBM -> All CB")
ax.plot(energy_grid, jdos_total, linewidth=1.0, linestyle=":", color="gray",
        label="Total JDOS")
ax.set_xlabel("Energy (eV)")
ax.set_ylabel("JDOS (states/eV)")
ax.set_title("(c) Band-Group Decomposition")
ax.set_xlim(*ENERGY_RANGE)
ax.legend(fontsize=9)
ax.grid(alpha=0.3)

# Panel (d): JDOS convergence with broadening
ax = axes[1, 1]
for sig in [0.02, 0.05, 0.1, 0.2]:
    _, jdos_sig = compute_jdos(eigenvalues_shifted, weights, n_valence,
                               energy_range=ENERGY_RANGE, n_energy=N_ENERGY, sigma=sig)
    ax.plot(energy_grid, jdos_sig, linewidth=1.0, label=f"sigma={sig} eV")
ax.set_xlabel("Energy (eV)")
ax.set_ylabel("JDOS (states/eV)")
ax.set_title("(d) Broadening Dependence")
ax.set_xlim(*ENERGY_RANGE)
ax.legend(fontsize=9)
ax.grid(alpha=0.3)

plt.suptitle("Joint Density of States -- Si (PBE)", fontsize=15, y=1.01)
plt.tight_layout()
plt.savefig("jdos_analysis.png", dpi=200, bbox_inches="tight")
print("\nSaved: jdos_analysis.png")
```

### Method B: VASP -- JDOS from EIGENVAL (VASPKIT 716--717)

When VASP is available, the EIGENVAL file provides eigenvalues on the k-grid used in the calculation.

#### Step B1: Parse EIGENVAL and Compute JDOS

```python
#!/usr/bin/env python3
"""
Compute JDOS from VASP EIGENVAL file.

Equivalent to VASPKIT options:
  716 - Joint Density of States
  717 - Partial (band-decomposed) Joint Density of States

NOTE: Requires VASP EIGENVAL file. This script will work when VASP
becomes available in the container.
"""
import numpy as np
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.signal import find_peaks


def read_eigenval(filename="EIGENVAL"):
    """
    Read VASP EIGENVAL file.

    Returns:
        nkpts, nbands, kpoints, eigenvalues, occupations, weights
    """
    if not os.path.exists(filename):
        raise FileNotFoundError(f"{filename} not found. Run VASP first.")

    with open(filename) as f:
        lines = f.readlines()

    # Line 1: number of ions, ions, NBLOCK*KBLOCK, ISPIN
    # Line 6: NELECT, NKPTS, NBANDS
    header = lines[5].split()
    nelect = int(float(header[0]))
    nkpts = int(header[1])
    nbands = int(header[2])

    kpoints = np.zeros((nkpts, 3))
    weights = np.zeros(nkpts)
    eigenvalues = np.zeros((nkpts, nbands))
    occupations = np.zeros((nkpts, nbands))

    line_idx = 7
    for ik in range(nkpts):
        kdata = lines[line_idx].split()
        kpoints[ik] = [float(kdata[0]), float(kdata[1]), float(kdata[2])]
        weights[ik] = float(kdata[3])
        line_idx += 1

        for ib in range(nbands):
            bdata = lines[line_idx].split()
            eigenvalues[ik, ib] = float(bdata[1])
            if len(bdata) > 2:
                occupations[ik, ib] = float(bdata[2])
            line_idx += 1
        line_idx += 1  # blank line

    # Normalize weights
    if weights.sum() > 0:
        weights /= weights.sum()

    return nkpts, nbands, kpoints, eigenvalues, occupations, weights


def compute_jdos_vasp(eigenvalues, weights, n_valence,
                      energy_range=(0, 15), n_energy=2000, sigma=0.05):
    """
    Compute total JDOS from VASP eigenvalues.
    Same algorithm as the QE version.
    """
    nkpts, nbands = eigenvalues.shape
    energy_grid = np.linspace(energy_range[0], energy_range[1], n_energy)
    jdos = np.zeros(n_energy)

    inv_sqrt2pi_sigma = 1.0 / (sigma * np.sqrt(2.0 * np.pi))
    inv_2sigma2 = 1.0 / (2.0 * sigma**2)

    for ik in range(nkpts):
        wk = weights[ik]
        for iv in range(n_valence):
            for ic in range(n_valence, nbands):
                dE = eigenvalues[ik, ic] - eigenvalues[ik, iv]
                if dE < energy_range[0] - 5*sigma or dE > energy_range[1] + 5*sigma:
                    continue
                jdos += wk * inv_sqrt2pi_sigma * np.exp(
                    -(energy_grid - dE)**2 * inv_2sigma2
                )

    return energy_grid, jdos


def compute_partial_jdos_vasp(eigenvalues, weights, vb_list, cb_list,
                              energy_range=(0, 15), n_energy=2000, sigma=0.05):
    """
    Compute partial JDOS for specified band pairs.
    vb_list and cb_list are 0-based band indices.
    """
    nkpts, nbands = eigenvalues.shape
    energy_grid = np.linspace(energy_range[0], energy_range[1], n_energy)
    pjdos = np.zeros(n_energy)

    inv_sqrt2pi_sigma = 1.0 / (sigma * np.sqrt(2.0 * np.pi))
    inv_2sigma2 = 1.0 / (2.0 * sigma**2)

    for ik in range(nkpts):
        wk = weights[ik]
        for iv in vb_list:
            for ic in cb_list:
                if iv >= nbands or ic >= nbands:
                    continue
                dE = eigenvalues[ik, ic] - eigenvalues[ik, iv]
                if dE < energy_range[0] - 5*sigma or dE > energy_range[1] + 5*sigma:
                    continue
                pjdos += wk * inv_sqrt2pi_sigma * np.exp(
                    -(energy_grid - dE)**2 * inv_2sigma2
                )

    return energy_grid, pjdos


# ── Main ──────────────────────────────────────────────────────────────
if __name__ == "__main__":
    try:
        nkpts, nbands, kpoints, eigenvalues, occupations, weights = read_eigenval()

        # Determine n_valence from occupations
        avg_occ = np.mean(occupations, axis=0)
        n_valence = np.sum(avg_occ > 0.5)
        print(f"EIGENVAL: {nkpts} k-points, {nbands} bands, {n_valence} valence bands")

        # Shift so VBM = 0
        vbm = np.max(eigenvalues[:, :n_valence])
        eigenvalues -= vbm

        # Total JDOS
        E, jdos = compute_jdos_vasp(eigenvalues, weights, n_valence,
                                    energy_range=(0, 12), sigma=0.05)

        # Partial JDOS: VBM -> CBM
        _, pjdos_1 = compute_partial_jdos_vasp(
            eigenvalues, weights,
            [n_valence - 1], [n_valence],
            energy_range=(0, 12), sigma=0.05
        )

        # Plot
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

        ax1.plot(E, jdos, "b-", lw=1.5, label="Total JDOS")
        ax1.fill_between(E, 0, jdos, alpha=0.1)
        ax1.set(xlabel="Energy (eV)", ylabel="JDOS",
                title="Total JDOS (VASP)", xlim=(0, 12))
        ax1.legend()
        ax1.grid(alpha=0.3)

        ax2.plot(E, pjdos_1, "r-", lw=1.5, label="VBM -> CBM")
        ax2.set(xlabel="Energy (eV)", ylabel="Partial JDOS",
                title="Partial JDOS (VASP)", xlim=(0, 12))
        ax2.legend()
        ax2.grid(alpha=0.3)

        plt.tight_layout()
        plt.savefig("jdos_vasp.png", dpi=200, bbox_inches="tight")
        print("Saved: jdos_vasp.png")

    except FileNotFoundError as e:
        print(f"VASP file not available: {e}")
        print("Use Method A (QE eigenvalues) to compute JDOS.")
```

### Complete Single-Script Workflow (QE)

```python
#!/usr/bin/env python3
"""
Complete JDOS workflow using QE.
SCF -> NSCF (dense k-grid) -> parse eigenvalues -> total + partial JDOS.

Example: Silicon (diamond structure)
"""
import os
import re
import subprocess
import numpy as np
from scipy.signal import find_peaks
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ── Configuration ────────────────────────────────────────────────────
PSEUDO_DIR = os.path.abspath("./pseudo")
OUTDIR = os.path.abspath("./tmp_jdos")
os.makedirs(OUTDIR, exist_ok=True)
os.makedirs(PSEUDO_DIR, exist_ok=True)
PREFIX = "si"
ECUTWFC = 60.0
ECUTRHO = 240.0
NBND = 20
NPROC = 4
KGRID = "20 20 20"
N_VALENCE = 4  # Si: 4 valence bands
SIGMA = 0.05   # Broadening in eV
ENERGY_MAX = 12.0

# ── Step 1: SCF ──────────────────────────────────────────────────────
scf_in = f"""&CONTROL
    calculation = 'scf', prefix = '{PREFIX}',
    outdir = '{OUTDIR}', pseudo_dir = '{PSEUDO_DIR}',
/
&SYSTEM
    ibrav = 2, celldm(1) = 10.26,
    nat = 2, ntyp = 1,
    ecutwfc = {ECUTWFC}, ecutrho = {ECUTRHO},
    occupations = 'fixed',
/
&ELECTRONS
    conv_thr = 1.0d-10,
/
ATOMIC_SPECIES
  Si 28.0855 Si_ONCV_PBE-1.2.upf
ATOMIC_POSITIONS (crystal)
  Si 0.00 0.00 0.00
  Si 0.25 0.25 0.25
K_POINTS (automatic)
  {KGRID} 0 0 0
"""
with open(f"{PREFIX}_scf.in", "w") as f:
    f.write(scf_in)

print("[1/3] SCF...")
r = subprocess.run(["mpirun", "-np", str(NPROC), "pw.x", "-in", f"{PREFIX}_scf.in"],
                    capture_output=True, text=True, timeout=600)
with open(f"{PREFIX}_scf.out", "w") as f:
    f.write(r.stdout)
assert "convergence has been achieved" in r.stdout, "SCF failed!"
print("      Done.")

# ── Step 2: NSCF ────────────────────────────────────────────────────
nscf_in = f"""&CONTROL
    calculation = 'nscf', prefix = '{PREFIX}',
    outdir = '{OUTDIR}', pseudo_dir = '{PSEUDO_DIR}',
/
&SYSTEM
    ibrav = 2, celldm(1) = 10.26,
    nat = 2, ntyp = 1,
    ecutwfc = {ECUTWFC}, ecutrho = {ECUTRHO},
    occupations = 'fixed', nbnd = {NBND},
/
&ELECTRONS
    conv_thr = 1.0d-10,
/
ATOMIC_SPECIES
  Si 28.0855 Si_ONCV_PBE-1.2.upf
ATOMIC_POSITIONS (crystal)
  Si 0.00 0.00 0.00
  Si 0.25 0.25 0.25
K_POINTS (automatic)
  {KGRID} 0 0 0
"""
with open(f"{PREFIX}_nscf.in", "w") as f:
    f.write(nscf_in)

print(f"[2/3] NSCF (nbnd={NBND}, k={KGRID})...")
r = subprocess.run(["mpirun", "-np", str(NPROC), "pw.x", "-in", f"{PREFIX}_nscf.in"],
                    capture_output=True, text=True, timeout=1800)
with open(f"{PREFIX}_nscf.out", "w") as f:
    f.write(r.stdout)
assert r.returncode == 0, "NSCF failed!"
print("      Done.")

# ── Step 3: Parse and compute JDOS ──────────────────────────────────
print("[3/3] Computing JDOS...")

# Parse eigenvalues
kpoints, weights_arr, eigenvalues = [], [], []
efermi = None

with open(f"{PREFIX}_nscf.out") as f:
    lines = f.readlines()

for line in lines:
    if "highest occupied" in line or "Fermi energy" in line:
        nums = re.findall(r"[-+]?\d*\.?\d+", line)
        if nums:
            efermi = float(nums[-1])
    if "highest occupied, lowest unoccupied" in line:
        nums = re.findall(r"[-+]?\d*\.?\d+", line)
        if len(nums) >= 2:
            efermi = (float(nums[-2]) + float(nums[-1])) / 2.0

i = 0
while i < len(lines):
    if "k =" in lines[i] and "bands" in lines[i]:
        kline = lines[i].split("k =")[1].split("(")[0].strip()
        kcoords = [float(x) for x in kline.split()]
        kpoints.append(kcoords)
        wk = float(lines[i].split("wk =")[1].strip()) if "wk =" in lines[i] else 1.0
        weights_arr.append(wk)
        i += 1
        while i < len(lines) and lines[i].strip() == "":
            i += 1
        bands = []
        while i < len(lines) and lines[i].strip() != "":
            try:
                bands.extend([float(x) for x in lines[i].split()])
            except ValueError:
                break
            i += 1
        eigenvalues.append(bands)
    else:
        i += 1

nb = min(len(b) for b in eigenvalues)
eigenvalues = np.array([b[:nb] for b in eigenvalues])
weights_arr = np.array(weights_arr)
if weights_arr.sum() > 0:
    weights_arr /= weights_arr.sum()

nkpts = len(eigenvalues)
print(f"Parsed {nkpts} k-points, {nb} bands")

# Shift VBM to 0
vbm = np.max(eigenvalues[:, :N_VALENCE])
cbm = np.min(eigenvalues[:, N_VALENCE:])
print(f"Band gap: {cbm - vbm:.4f} eV")
eigenvalues -= vbm

# Compute JDOS
def jdos_calc(eigs, wts, nv, erange, ne, sig):
    E = np.linspace(erange[0], erange[1], ne)
    J = np.zeros(ne)
    c1 = 1.0 / (sig * np.sqrt(2 * np.pi))
    c2 = 1.0 / (2 * sig**2)
    for ik in range(len(eigs)):
        w = wts[ik]
        for iv in range(nv):
            for ic in range(nv, eigs.shape[1]):
                dE = eigs[ik, ic] - eigs[ik, iv]
                if erange[0] - 5*sig < dE < erange[1] + 5*sig:
                    J += w * c1 * np.exp(-(E - dE)**2 * c2)
    return E, J

E_grid, J_total = jdos_calc(eigenvalues, weights_arr, N_VALENCE,
                            (0, ENERGY_MAX), 2000, SIGMA)

# Partial JDOS
def pjdos_calc(eigs, wts, vb_list, cb_list, erange, ne, sig):
    E = np.linspace(erange[0], erange[1], ne)
    J = np.zeros(ne)
    c1 = 1.0 / (sig * np.sqrt(2 * np.pi))
    c2 = 1.0 / (2 * sig**2)
    for ik in range(len(eigs)):
        w = wts[ik]
        for iv in vb_list:
            for ic in cb_list:
                if ic >= eigs.shape[1]:
                    continue
                dE = eigs[ik, ic] - eigs[ik, iv]
                if erange[0] - 5*sig < dE < erange[1] + 5*sig:
                    J += w * c1 * np.exp(-(E - dE)**2 * c2)
    return E, J

_, J_vbm_cbm = pjdos_calc(eigenvalues, weights_arr,
                          [N_VALENCE-1], [N_VALENCE],
                          (0, ENERGY_MAX), 2000, SIGMA)
_, J_vbm1_cbm = pjdos_calc(eigenvalues, weights_arr,
                           [N_VALENCE-2], [N_VALENCE],
                           (0, ENERGY_MAX), 2000, SIGMA)
_, J_vbm_cbm1 = pjdos_calc(eigenvalues, weights_arr,
                           [N_VALENCE-1], [N_VALENCE+1],
                           (0, ENERGY_MAX), 2000, SIGMA)

# Print summary
peaks, _ = find_peaks(J_total, height=0.1*np.max(J_total), distance=20)
print(f"\n=== JDOS Summary ===")
onset_idx = np.where(J_total > 0.01*np.max(J_total))[0]
if len(onset_idx) > 0:
    print(f"JDOS onset: {E_grid[onset_idx[0]]:.3f} eV")
for i, p in enumerate(peaks[:5]):
    print(f"Peak {i+1}: {E_grid[p]:.3f} eV (JDOS = {J_total[p]:.4f})")

# Plot
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

axes[0,0].plot(E_grid, J_total, "b-", lw=1.5)
axes[0,0].fill_between(E_grid, 0, J_total, alpha=0.1)
axes[0,0].set(xlabel="Energy (eV)", ylabel="JDOS (states/eV)",
              title="(a) Total JDOS", xlim=(0, ENERGY_MAX))
axes[0,0].grid(alpha=0.3)

axes[0,1].plot(E_grid, J_vbm_cbm, lw=1.2, label=f"VB{N_VALENCE}->CB{N_VALENCE+1}")
axes[0,1].plot(E_grid, J_vbm1_cbm, "--", lw=1.2, label=f"VB{N_VALENCE-1}->CB{N_VALENCE+1}")
axes[0,1].plot(E_grid, J_vbm_cbm1, "-.", lw=1.2, label=f"VB{N_VALENCE}->CB{N_VALENCE+2}")
axes[0,1].set(xlabel="Energy (eV)", ylabel="Partial JDOS",
              title="(b) Partial JDOS", xlim=(0, ENERGY_MAX))
axes[0,1].legend(fontsize=9)
axes[0,1].grid(alpha=0.3)

# Stacked partial contributions
sum_partial = J_vbm_cbm + J_vbm1_cbm + J_vbm_cbm1
axes[1,0].fill_between(E_grid, 0, J_vbm_cbm, alpha=0.4, label=f"VB{N_VALENCE}->CB{N_VALENCE+1}")
axes[1,0].fill_between(E_grid, J_vbm_cbm, J_vbm_cbm+J_vbm1_cbm, alpha=0.4,
                       label=f"VB{N_VALENCE-1}->CB{N_VALENCE+1}")
axes[1,0].fill_between(E_grid, J_vbm_cbm+J_vbm1_cbm, sum_partial, alpha=0.4,
                       label=f"VB{N_VALENCE}->CB{N_VALENCE+2}")
axes[1,0].plot(E_grid, J_total, "k-", lw=1.0, label="Total")
axes[1,0].set(xlabel="Energy (eV)", ylabel="JDOS",
              title="(c) Decomposition", xlim=(0, ENERGY_MAX))
axes[1,0].legend(fontsize=8)
axes[1,0].grid(alpha=0.3)

# Broadening comparison
for sig in [0.02, 0.05, 0.1, 0.2]:
    _, Js = jdos_calc(eigenvalues, weights_arr, N_VALENCE,
                      (0, ENERGY_MAX), 2000, sig)
    axes[1,1].plot(E_grid, Js, lw=1.0, label=f"sigma={sig}")
axes[1,1].set(xlabel="Energy (eV)", ylabel="JDOS",
              title="(d) Broadening Convergence", xlim=(0, ENERGY_MAX))
axes[1,1].legend(fontsize=9)
axes[1,1].grid(alpha=0.3)

plt.suptitle("Joint Density of States -- Si (PBE)", fontsize=15, y=1.01)
plt.tight_layout()
plt.savefig("jdos_complete.png", dpi=200, bbox_inches="tight")
print(f"\nSaved: jdos_complete.png")
```

---

## Key Parameters

| Parameter | Where | Typical Value | Impact |
|---|---|---|---|
| `K_POINTS` | SCF/NSCF | 20x20x20+ | Most critical for smooth JDOS. Insufficient k-points cause jagged, step-like JDOS |
| `nbnd` | NSCF | 2--4x occupied | Determines maximum transition energy. More bands = higher energy range |
| `sigma` | Post-processing | 0.02--0.1 eV | Gaussian broadening. Smaller = sharper features but noisier. Must be converged |
| `n_valence` | Post-processing | From Fermi energy | Number of occupied bands. Critical for correct VB/CB separation |
| `NBANDS` | VASP INCAR | 2--4x occupied | Same role as nbnd in QE |
| `ISMEAR` | VASP INCAR | 0 (Gaussian) | For semiconductors/insulators. Use -5 (tetrahedron) for metals |

### K-Grid Convergence for JDOS

JDOS converges more slowly than total energies with respect to k-grid density:

```python
# Test k-grids: 12x12x12, 16x16x16, 20x20x20, 24x24x24
# Compare: onset energy, peak positions, peak heights
# Typically 20x20x20 is adequate for bulk Si; layered materials may need more
```

## Interpreting Results

- **JDOS onset**: The onset energy of the JDOS corresponds to the minimum direct band gap. For indirect-gap materials (Si, Ge), the JDOS onset may be larger than the fundamental gap because JDOS only counts vertical transitions.

- **Van Hove singularities**: Peaks in the JDOS correspond to critical points where bands are parallel. In 3D, there are four types: M0 (band edge, square-root onset), M1 and M2 (saddle points), M3 (band maximum/minimum). These produce characteristic shapes in the JDOS.

- **JDOS vs eps2**: If JDOS peaks align with eps2 peaks, the optical response is JDOS-dominated. If eps2 peaks are shifted or suppressed relative to JDOS peaks, matrix element effects (selection rules) are important.

- **Partial JDOS**: Decomposing the JDOS by band pairs identifies which transitions contribute at each energy. The VBM->CBM pair dominates near the gap edge; higher transitions become important at higher energies.

- **Spin-orbit effects**: For materials with strong SOC (heavy elements), the JDOS may split features due to spin-orbit splitting of bands. Run with SOC enabled for accurate JDOS.

## Common Issues

| Problem | Solution |
|---|---|
| JDOS is very jagged/step-like | Increase k-grid density. 20x20x20 minimum for bulk. Increase broadening sigma as a temporary fix |
| JDOS onset at wrong energy | Check n_valence is correct. Verify VBM alignment. PBE underestimates gaps |
| Partial JDOS does not sum to total | Expected if you did not include all band pairs. The total includes all VB->CB combinations |
| JDOS is zero everywhere | Check that n_valence is correct and eigenvalues are properly parsed |
| Very different JDOS for different broadening | JDOS is not converged with k-points. Use denser k-grid so results are insensitive to broadening |
| Need JDOS for specific k-path | Modify the code to filter k-points along a specific direction. Useful for band-structure-resolved JDOS |
| QE eigenvalue parsing fails | Check the output format. Different QE versions may have slightly different formatting. Adjust the regex patterns |
| Memory error with large k-grid | Reduce nbnd or process k-points in batches. The JDOS computation is O(nkpts * n_valence * n_conduction * n_energy) |
