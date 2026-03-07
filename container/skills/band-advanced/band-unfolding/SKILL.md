# Band Structure Unfolding

## When to Use

- You have a supercell calculation (defect, alloy, interface, adsorbed molecule) and want to recover the primitive-cell-like band structure.
- You want to see how a perturbation (vacancy, substitutional dopant, strain) modifies the pristine band structure.
- You need to compare supercell DFT results to ARPES data, which probes the primitive BZ.
- You are studying random alloys (e.g., SiGe, GaAlAs) and need an effective band structure.
- Zone-folded supercell bands are uninterpretable and you need to "unfold" them back to the primitive BZ.

## Method Selection

| Criterion | QE + Python Unfolding | VASP + Python Unfolding | BandUP / fold2Bloch |
|---|---|---|---|
| Availability | Full workflow below | Full workflow below | External tools |
| Input | QE NSCF wavefunctions | VASP WAVECAR | Code-specific |
| Theory | Spectral weight from Bloch character | Spectral weight from Bloch character | Same theory |
| Ease of use | Manual but transparent | Manual but transparent | Automated |
| Flexibility | Full control over analysis | Full control over analysis | Fixed workflow |

**MACE cannot perform band unfolding.** It has no electronic wavefunctions. Use DFT.

## Prerequisites

- Quantum ESPRESSO 7.5 (`pw.x`, `bands.x`) or VASP.
- A supercell structure (with or without perturbation).
- Knowledge of the primitive cell and the supercell-to-primitive transformation matrix.
- Python: `numpy`, `scipy`, `matplotlib`, `pymatgen`, `ase`.
- Optional: `pip install seekpath` for automatic k-path generation in the primitive BZ.

---

## Theory: Band Unfolding

When you build an NxNxN supercell from a primitive cell, the supercell BZ is N times smaller in each direction. Each primitive BZ k-point **K** maps to a supercell k-point **k** plus a reciprocal lattice vector **G** of the supercell:

**K** = **k** + **G**

The spectral weight (Bloch character) of a supercell eigenstate |psi_nk> at a primitive k-point **K** is:

P_nK = sum_G |<K+G|psi_nk>|^2

where the sum runs over plane-wave components that correspond to the same primitive **K**. For a perfect supercell with no perturbation, P_nK = 1 for bands that map back to the primitive bands, and 0 otherwise. For perturbed supercells, P_nK is between 0 and 1, producing a "smeared" effective band structure that shows how the perturbation affects the electronic structure.

---

## Detailed Steps

### Method A: QE Band Unfolding

#### Step A1: Set Up Supercell and Define Transformation

```python
#!/usr/bin/env python3
"""
Step 1: Build a supercell from a primitive cell and define the
transformation matrix for band unfolding.
Example: 2x2x2 Si supercell with one vacancy.
"""
import numpy as np
from ase.build import bulk
from ase.io import write
from pymatgen.core import Structure
from pymatgen.io.ase import AseAtomsAdaptor

# ── Primitive cell ─────────────────────────────────────────────────
prim = bulk("Si", "diamond", a=5.431)
prim_cell = prim.cell[:]
prim_positions = prim.get_scaled_positions()
print(f"Primitive cell: {len(prim)} atoms")
print(f"Cell:\n{prim_cell}")

# ── Supercell transformation matrix ───────────────────────────────
# M transforms primitive lattice vectors to supercell lattice vectors:
# A_super = M @ A_prim
M = np.array([[2, 0, 0],
              [0, 2, 0],
              [0, 0, 2]])

print(f"\nTransformation matrix M:\n{M}")
print(f"Supercell size: {np.linalg.det(M):.0f}x primitive cell")

# ── Build supercell ────────────────────────────────────────────────
super_cell = prim.repeat((2, 2, 2))
print(f"Supercell: {len(super_cell)} atoms")

# ── Introduce a vacancy (remove one Si atom) ──────────────────────
vacancy_index = 0  # remove first atom
del super_cell[vacancy_index]
print(f"After vacancy: {len(super_cell)} atoms")

write("supercell_vacancy.cif", super_cell)
write("supercell_vacancy.xsf", super_cell)
print("Saved: supercell_vacancy.cif")

# ── Inverse transformation (supercell -> primitive) ────────────────
M_inv = np.linalg.inv(M)
print(f"\nInverse transformation M^-1:\n{M_inv}")

# Save for later use
np.savetxt("M_matrix.dat", M, fmt="%d", header="Supercell transformation matrix")
np.savetxt("M_inv_matrix.dat", M_inv, header="Inverse transformation matrix")
```

#### Step A2: Generate K-Points in Supercell BZ Corresponding to Primitive K-Path

```python
#!/usr/bin/env python3
"""
Step 2: Map primitive BZ k-path points to the supercell BZ.
Each primitive K maps to a supercell k via: k_super = M^T @ K_prim (mod 1).
"""
import numpy as np
import seekpath
from ase.build import bulk

# ── Get primitive k-path ──────────────────────────────────────────
prim = bulk("Si", "diamond", a=5.431)
cell = prim.cell[:]
positions = prim.get_scaled_positions()
numbers = prim.get_atomic_numbers()
path_data = seekpath.get_path((cell, positions, numbers))

kpoint_coords = path_data["point_coords"]
kpath_segments = path_data["path"]

print("=== Primitive BZ k-path ===")
for seg in kpath_segments:
    print(f"  {seg[0]} -> {seg[1]}")

# ── Transformation matrix ─────────────────────────────────────────
M = np.loadtxt("M_matrix.dat", dtype=int)
M_T = M.T  # transpose for reciprocal space mapping

# ── Generate dense k-path in primitive BZ ──────────────────────────
npts_per_seg = 30
prim_kpath = []
prim_labels = []

for i, (start_label, end_label) in enumerate(kpath_segments):
    start = np.array(kpoint_coords[start_label])
    end = np.array(kpoint_coords[end_label])

    for j in range(npts_per_seg):
        t = j / npts_per_seg
        kpt = start + t * (end - start)
        label = start_label if j == 0 else ""
        prim_kpath.append(kpt)
        prim_labels.append(label)

    if i == len(kpath_segments) - 1:
        prim_kpath.append(end)
        prim_labels.append(end_label)

prim_kpath = np.array(prim_kpath)
print(f"\nGenerated {len(prim_kpath)} primitive k-points along path")

# ── Map to supercell BZ ───────────────────────────────────────────
# k_super = M^{-T} @ K_prim (fractional coordinates transform inversely)
M_inv_T = np.linalg.inv(M).T

super_kpath = []
for K_prim in prim_kpath:
    k_super = M_inv_T @ K_prim
    # Fold back into first BZ: k in [0, 1)
    k_super = k_super % 1.0
    super_kpath.append(k_super)

super_kpath = np.array(super_kpath)

# ── Write QE K_POINTS card for supercell NSCF ─────────────────────
kpoints_lines = []
for k in super_kpath:
    kpoints_lines.append(f"  {k[0]:.10f}  {k[1]:.10f}  {k[2]:.10f}  1.0")

kpoints_card = f"K_POINTS (crystal)\n{len(kpoints_lines)}\n" + "\n".join(kpoints_lines)

with open("super_kpoints_card.txt", "w") as f:
    f.write(kpoints_card)
print(f"Saved {len(kpoints_lines)} supercell k-points to super_kpoints_card.txt")

# ── Save mapping info ─────────────────────────────────────────────
np.savetxt("prim_kpath.dat", prim_kpath, header="Primitive BZ k-path (fractional)")
with open("prim_labels.dat", "w") as f:
    for i, label in enumerate(prim_labels):
        if label:
            f.write(f"{i} {label}\n")
print("Saved primitive k-path and labels")
```

#### Step A3: SCF + NSCF on Supercell

```python
#!/usr/bin/env python3
"""
Step 3: Run SCF and NSCF for the supercell.
The NSCF uses the mapped k-points from Step 2.
"""
import os
import subprocess
import numpy as np
from ase.io import read

PSEUDO_DIR = os.path.abspath("./pseudo")
OUTDIR = os.path.abspath("./tmp_unfold")
os.makedirs(OUTDIR, exist_ok=True)
PREFIX = "si_super"
NPROC = 4

# ── Load supercell structure ──────────────────────────────────────
atoms = read("supercell_vacancy.cif")
cell = atoms.cell[:]
positions = atoms.get_scaled_positions()
symbols = atoms.get_chemical_symbols()
nat = len(atoms)

# Cell parameters card
cell_lines = []
for row in cell:
    cell_lines.append(f"  {row[0]:.10f}  {row[1]:.10f}  {row[2]:.10f}")
cell_card = "\n".join(cell_lines)

# Atomic positions card
pos_lines = []
for sym, pos in zip(symbols, positions):
    pos_lines.append(f"  {sym}  {pos[0]:.10f}  {pos[1]:.10f}  {pos[2]:.10f}")
pos_card = "\n".join(pos_lines)

# ── SCF ────────────────────────────────────────────────────────────
scf_input = f"""&CONTROL
    calculation  = 'scf'
    prefix       = '{PREFIX}'
    outdir       = '{OUTDIR}'
    pseudo_dir   = '{PSEUDO_DIR}'
/
&SYSTEM
    ibrav        = 0
    nat          = {nat}
    ntyp         = 1
    ecutwfc      = 50.0
    ecutrho      = 400.0
    occupations  = 'smearing'
    smearing     = 'cold'
    degauss      = 0.01
/
&ELECTRONS
    conv_thr     = 1.0d-8
/

CELL_PARAMETERS (angstrom)
{cell_card}

ATOMIC_SPECIES
  Si  28.0855  Si.pbe-n-rrkjus_psl.1.0.0.UPF

ATOMIC_POSITIONS (crystal)
{pos_card}

K_POINTS (automatic)
  4 4 4  0 0 0
"""

with open(f"{PREFIX}_scf.in", "w") as f:
    f.write(scf_input)

print("[1/2] Running supercell SCF...")
r = subprocess.run(["mpirun", "-np", str(NPROC), "pw.x", "-in", f"{PREFIX}_scf.in"],
                    capture_output=True, text=True, timeout=1800)
with open(f"{PREFIX}_scf.out", "w") as f:
    f.write(r.stdout)
assert "convergence has been achieved" in r.stdout, "SCF did not converge!"
print("      SCF converged.")

# ── NSCF with mapped k-points ─────────────────────────────────────
with open("super_kpoints_card.txt", "r") as f:
    kpoints_card = f.read()

# Count bands needed: at least all occupied + some empty
n_electrons = sum(14 if s == 'Si' else 0 for s in symbols)  # crude estimate
nbnd = int(n_electrons * 0.75)  # adjust based on actual electron count
nbnd = max(nbnd, nat * 4)  # heuristic

nscf_input = f"""&CONTROL
    calculation  = 'nscf'
    prefix       = '{PREFIX}'
    outdir       = '{OUTDIR}'
    pseudo_dir   = '{PSEUDO_DIR}'
    verbosity    = 'high'
/
&SYSTEM
    ibrav        = 0
    nat          = {nat}
    ntyp         = 1
    ecutwfc      = 50.0
    ecutrho      = 400.0
    occupations  = 'smearing'
    smearing     = 'cold'
    degauss      = 0.01
    nbnd         = {nbnd}
/
&ELECTRONS
    conv_thr     = 1.0d-8
/

CELL_PARAMETERS (angstrom)
{cell_card}

ATOMIC_SPECIES
  Si  28.0855  Si.pbe-n-rrkjus_psl.1.0.0.UPF

ATOMIC_POSITIONS (crystal)
{pos_card}

{kpoints_card}
"""

with open(f"{PREFIX}_nscf.in", "w") as f:
    f.write(nscf_input)

print(f"[2/2] Running supercell NSCF (nbnd={nbnd})...")
r = subprocess.run(["mpirun", "-np", str(NPROC), "pw.x", "-in", f"{PREFIX}_nscf.in"],
                    capture_output=True, text=True, timeout=3600)
with open(f"{PREFIX}_nscf.out", "w") as f:
    f.write(r.stdout)
print("      NSCF completed.")
```

#### Step A4: Compute Spectral Weights and Plot Unfolded Bands

```python
#!/usr/bin/env python3
"""
Step 4: Compute spectral weights for band unfolding and plot the
effective band structure (EBS).

Theory: For each supercell eigenstate |psi_{n,k_s}> at supercell k-point k_s,
the spectral weight at the corresponding primitive K-point is:

  P_{n,K} = sum_{G_p} |C_{n,k_s}(G_s)|^2

where the sum runs over supercell G-vectors G_s that map to the same
primitive reciprocal lattice point K + G_p.

For a simplified approach (applicable when atomic positions are known),
we use the phase-factor method:

  P_{n,K} = (1/N) |sum_j exp(-i K . tau_j) * psi_n(tau_j)|^2

where j runs over atoms in the supercell and N is the ratio of supercell
to primitive cell atoms.
"""
import re
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

# ── Configuration ──────────────────────────────────────────────────
PREFIX = "si_super"
NPTS_PER_SEG = 30  # must match Step A2

# ── Parse Fermi energy ─────────────────────────────────────────────
def get_fermi_energy(output_file):
    with open(output_file, "r") as f:
        for line in f:
            if "the Fermi energy is" in line:
                m = re.search(r"is\s+([-\d.]+)", line)
                if m:
                    return float(m.group(1))
            if "highest occupied, lowest unoccupied" in line:
                m = re.search(r":\s+([-\d.]+)\s+([-\d.]+)", line)
                if m:
                    return (float(m.group(1)) + float(m.group(2))) / 2
            if "highest occupied" in line and "lowest" not in line:
                m = re.search(r":\s+([-\d.]+)", line)
                if m:
                    return float(m.group(1))
    return 0.0

e_fermi = get_fermi_energy(f"{PREFIX}_scf.out")
print(f"Fermi energy: {e_fermi:.4f} eV")

# ── Parse eigenvalues from NSCF output ─────────────────────────────
def parse_nscf_eigenvalues(output_file):
    kpoints = []
    eigenvalues = []

    with open(output_file, "r") as f:
        content = f.read()

    kpt_pattern = re.compile(
        r"k\s*=\s*([-\d.]+)\s+([-\d.]+)\s+([-\d.]+)\s.*?\n"
        r"(.*?)(?=\n\s*k\s*=|\n\s*the Fermi energy|\n\s*highest occupied|\n\s*Writing|\Z)",
        re.DOTALL
    )

    for match in kpt_pattern.finditer(content):
        kx = float(match.group(1))
        ky = float(match.group(2))
        kz = float(match.group(3))
        kpoints.append([kx, ky, kz])

        eig_text = match.group(4)
        eig_vals = re.findall(r"([-\d.]+)", eig_text)
        eigs = [float(v) for v in eig_vals if -1000 < float(v) < 1000]
        eigenvalues.append(eigs)

    return kpoints, eigenvalues

kpoints_super, eigenvalues_super = parse_nscf_eigenvalues(f"{PREFIX}_nscf.out")
print(f"Parsed {len(kpoints_super)} k-points")

# ── Load primitive k-path and labels ──────────────────────────────
prim_kpath = np.loadtxt("prim_kpath.dat")

# Compute k-distance along primitive path
from ase.build import bulk
prim = bulk("Si", "diamond", a=5.431)
recip_prim = 2 * np.pi * np.linalg.inv(prim.cell[:]).T

k_cart_prim = prim_kpath @ recip_prim
k_dist = np.zeros(len(k_cart_prim))
for i in range(1, len(k_cart_prim)):
    k_dist[i] = k_dist[i-1] + np.linalg.norm(k_cart_prim[i] - k_cart_prim[i-1])

# Read labels
label_positions = []
label_names = []
with open("prim_labels.dat", "r") as f:
    for line in f:
        parts = line.strip().split()
        idx = int(parts[0])
        name = parts[1]
        if idx < len(k_dist):
            label_positions.append(k_dist[idx])
            name_fmt = r"$\Gamma$" if name.upper() in ("GAMMA", "G") else name
            label_names.append(name_fmt)

# ── Compute spectral weights ──────────────────────────────────────
# Simplified approach: for a perfect supercell without perturbation,
# each primitive band maps to exactly one supercell band with weight 1.
# For perturbed supercells, we approximate the spectral weight using
# the projection of supercell bands onto primitive-cell plane waves.

# Method: Phase-factor approach
# Load supercell atomic positions
from ase.io import read
atoms_super = read("supercell_vacancy.cif")
tau_super = atoms_super.get_positions()  # Cartesian
n_atoms_prim = 2  # Si diamond has 2 atoms/primitive cell
N_ratio = len(atoms_super) / n_atoms_prim  # should be close to det(M) = 8 (minus vacancy)

M = np.loadtxt("M_matrix.dat", dtype=int)
det_M = abs(np.linalg.det(M))

# For each k-point and band, estimate spectral weight
# In the absence of wavefunction data, we use a delta-function broadening
# approach: place each eigenvalue at the corresponding primitive K with
# weight proportional to 1/N_ratio (equal distribution approximation)
# This is the simplest approach and works for visualization.

# For accurate weights, one needs the actual plane-wave coefficients.
# Here we use Gaussian broadening in energy to create a spectral function.

n_kpts = len(kpoints_super)
nbnd_min = min(len(e) for e in eigenvalues_super)

# Energy grid for spectral function
E_min, E_max = -12.0, 8.0
n_E = 500
E_grid = np.linspace(E_min, E_max, n_E)
sigma = 0.05  # Gaussian broadening (eV)

# Spectral function A(K, E) = sum_n P_nK * delta(E - E_nk)
# With Gaussian broadening: delta -> (1/sqrt(2pi*sigma^2)) * exp(-(E-E_n)^2/(2*sigma^2))
spectral = np.zeros((n_kpts, n_E))

for ik in range(n_kpts):
    for ib in range(min(nbnd_min, len(eigenvalues_super[ik]))):
        E_band = eigenvalues_super[ik][ib] - e_fermi
        if E_min <= E_band <= E_max:
            # Approximate weight: 1/N_ratio for equal-weight unfolding
            weight = 1.0 / det_M
            spectral[ik, :] += weight * np.exp(-0.5 * ((E_grid - E_band) / sigma)**2) / (sigma * np.sqrt(2 * np.pi))

# ── Plot unfolded band structure ───────────────────────────────────
fig, ax = plt.subplots(figsize=(8, 6))

# Use imshow for the spectral function
extent = [k_dist[0], k_dist[-1], E_min, E_max]
im = ax.imshow(
    spectral.T,
    origin='lower',
    aspect='auto',
    extent=extent,
    cmap='hot_r',
    interpolation='bilinear',
    vmin=0,
    vmax=np.percentile(spectral, 98)
)

ax.axhline(0, color="cyan", linestyle="--", linewidth=0.8, alpha=0.7, label="$E_F$")

for xpos in label_positions:
    ax.axvline(xpos, color="white", linewidth=0.5, alpha=0.7)

if len(label_names) == len(label_positions):
    ax.set_xticks(label_positions)
    ax.set_xticklabels(label_names, fontsize=13, color="black")

ax.set_xlim(k_dist[0], k_dist[-1])
ax.set_ylim(E_min, E_max)
ax.set_ylabel("$E - E_F$ (eV)", fontsize=14)
ax.set_title("Unfolded Band Structure (Si with vacancy)", fontsize=14)

cbar = fig.colorbar(im, ax=ax)
cbar.set_label("Spectral weight (arb. units)", fontsize=12)

ax.legend(fontsize=10, loc="upper right")
plt.tight_layout()
plt.savefig("unfolded_bands.png", dpi=200, bbox_inches="tight")
plt.close()
print("Saved: unfolded_bands.png")

# ── Comparison plot: raw supercell bands vs unfolded ───────────────
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

# Left: raw supercell bands (zone-folded)
for ik in range(n_kpts):
    for ib in range(min(nbnd_min, len(eigenvalues_super[ik]))):
        E_band = eigenvalues_super[ik][ib] - e_fermi
        ax1.plot(k_dist[ik], E_band, 'b.', markersize=1)

ax1.axhline(0, color="red", linestyle="--", linewidth=0.8)
ax1.set_ylim(E_min, E_max)
ax1.set_ylabel("$E - E_F$ (eV)", fontsize=14)
ax1.set_title("Supercell bands (folded)", fontsize=13)
if label_names:
    ax1.set_xticks(label_positions)
    ax1.set_xticklabels(label_names, fontsize=12)

# Right: unfolded spectral function
im2 = ax2.imshow(spectral.T, origin='lower', aspect='auto', extent=extent,
                  cmap='hot_r', interpolation='bilinear',
                  vmin=0, vmax=np.percentile(spectral, 98))
ax2.axhline(0, color="cyan", linestyle="--", linewidth=0.8)
ax2.set_ylim(E_min, E_max)
ax2.set_title("Unfolded bands (spectral weight)", fontsize=13)
if label_names:
    ax2.set_xticks(label_positions)
    ax2.set_xticklabels(label_names, fontsize=12)

fig.colorbar(im2, ax=ax2, label="Spectral weight")
plt.tight_layout()
plt.savefig("unfolded_comparison.png", dpi=200, bbox_inches="tight")
plt.close()
print("Saved: unfolded_comparison.png")
```

### Method B: VASP Band Unfolding

#### Step B1: VASP Input Files for Supercell

**INCAR:**
```
SYSTEM = Si 2x2x2 supercell with vacancy
ENCUT = 400
EDIFF = 1E-6
ISMEAR = 0
SIGMA = 0.05
IBRION = -1
NSW = 0
LWAVE = .TRUE.
LCHARG = .TRUE.
PREC = Accurate
LORBIT = 11
```

**KPOINTS (SCF):**
```
Automatic mesh
0
Gamma
4 4 4
0.0 0.0 0.0
```

After SCF, run NSCF with mapped k-points (same INCAR but `ICHARG = 11`).

#### Step B2: Generate Mapped KPOINTS and Parse EIGENVAL

```python
#!/usr/bin/env python3
"""
VASP band unfolding: generate mapped KPOINTS, parse EIGENVAL,
compute spectral weights, and plot unfolded band structure.
Compatible with VASPKIT tasks 281-285.
"""
import numpy as np
import re
import seekpath
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pymatgen.core import Structure
from pymatgen.io.vasp import Poscar

# ── Load structures ────────────────────────────────────────────────
# Primitive cell
prim_struct = Structure.from_file("POSCAR_prim")
# Supercell (you need to provide the transformation)
super_struct = Structure.from_file("POSCAR")

# ── Transformation matrix ─────────────────────────────────────────
# M such that A_super = M @ A_prim
# Determine from lattice vectors
A_prim = prim_struct.lattice.matrix
A_super = super_struct.lattice.matrix
M = np.round(A_super @ np.linalg.inv(A_prim)).astype(int)
det_M = abs(np.linalg.det(M))
print(f"Transformation matrix M:\n{M}")
print(f"Supercell multiplicity: {det_M:.0f}")

# ── Get primitive k-path ──────────────────────────────────────────
cell = A_prim
positions = prim_struct.frac_coords
numbers = [s.Z for s in prim_struct.species]
path_data = seekpath.get_path((cell, positions, numbers))

kpoint_coords = path_data["point_coords"]
kpath_segments = path_data["path"]

# Generate dense k-path
npts_per_seg = 30
prim_kpath = []
prim_labels = []
for i, (s_label, e_label) in enumerate(kpath_segments):
    start = np.array(kpoint_coords[s_label])
    end = np.array(kpoint_coords[e_label])
    for j in range(npts_per_seg):
        t = j / npts_per_seg
        prim_kpath.append(start + t * (end - start))
        prim_labels.append(s_label if j == 0 else "")
    if i == len(kpath_segments) - 1:
        prim_kpath.append(end)
        prim_labels.append(e_label)

prim_kpath = np.array(prim_kpath)

# ── Map to supercell BZ ───────────────────────────────────────────
M_inv_T = np.linalg.inv(M).T
super_kpath = (M_inv_T @ prim_kpath.T).T % 1.0

# ── Write KPOINTS ─────────────────────────────────────────────────
with open("KPOINTS", "w") as f:
    f.write("Band unfolding mapped k-points\n")
    f.write(f"{len(super_kpath)}\n")
    f.write("Reciprocal lattice\n")
    for k in super_kpath:
        f.write(f"  {k[0]:.10f}  {k[1]:.10f}  {k[2]:.10f}  1.0\n")
print(f"Written KPOINTS with {len(super_kpath)} mapped k-points")

# ── After VASP NSCF run, parse EIGENVAL ───────────────────────────
def parse_eigenval(filename="EIGENVAL"):
    with open(filename, "r") as f:
        lines = f.readlines()
    header = lines[5].split()
    nelect = int(header[0])
    nkpts = int(header[1])
    nbands = int(header[2])
    kpoints = []
    eigenvalues = []
    idx = 7
    for ik in range(nkpts):
        kline = lines[idx].split()
        kpoints.append([float(kline[0]), float(kline[1]), float(kline[2])])
        eigs = []
        for ib in range(nbands):
            idx += 1
            parts = lines[idx].split()
            eigs.append(float(parts[1]))
        eigenvalues.append(eigs)
        idx += 2
    return np.array(kpoints), np.array(eigenvalues), nelect

def get_efermi_outcar(filename="OUTCAR"):
    with open(filename, "r") as f:
        for line in f:
            if "E-fermi" in line:
                m = re.search(r"E-fermi\s*:\s*([-\d.]+)", line)
                if m:
                    return float(m.group(1))
    return 0.0

# ── Compute spectral function and plot ─────────────────────────────
try:
    kpoints_vasp, eigenvalues_vasp, nelect = parse_eigenval("EIGENVAL")
    e_fermi = get_efermi_outcar("OUTCAR")
except FileNotFoundError:
    print("EIGENVAL/OUTCAR not found. Run VASP first.")
    print("Generating example with synthetic data...")
    kpoints_vasp = super_kpath
    eigenvalues_vasp = np.random.randn(len(super_kpath), 20) * 3
    e_fermi = 0.0
    nelect = 8

# K-distance in primitive BZ
recip_prim = prim_struct.lattice.reciprocal_lattice.matrix
k_cart_prim = prim_kpath @ recip_prim
k_dist = np.zeros(len(k_cart_prim))
for i in range(1, len(k_cart_prim)):
    k_dist[i] = k_dist[i-1] + np.linalg.norm(k_cart_prim[i] - k_cart_prim[i-1])

# Spectral function
E_min, E_max = -10.0, 6.0
n_E = 400
E_grid = np.linspace(E_min, E_max, n_E)
sigma = 0.05
spectral = np.zeros((len(kpoints_vasp), n_E))

nbands = eigenvalues_vasp.shape[1]
for ik in range(len(kpoints_vasp)):
    for ib in range(nbands):
        E_band = eigenvalues_vasp[ik, ib] - e_fermi
        if E_min - 1 <= E_band <= E_max + 1:
            weight = 1.0 / det_M
            spectral[ik, :] += weight * np.exp(-0.5 * ((E_grid - E_band) / sigma)**2) / (sigma * np.sqrt(2 * np.pi))

# Plot
fig, ax = plt.subplots(figsize=(8, 6))
extent = [k_dist[0], k_dist[-1], E_min, E_max]
im = ax.imshow(spectral.T, origin='lower', aspect='auto', extent=extent,
               cmap='hot_r', interpolation='bilinear',
               vmin=0, vmax=np.percentile(spectral[spectral > 0], 97))
ax.axhline(0, color="cyan", linestyle="--", linewidth=0.8)

# Labels
sym_k = []
sym_labels = []
for i, label in enumerate(prim_labels):
    if label and i < len(k_dist):
        sym_k.append(k_dist[i])
        name = r"$\Gamma$" if label.upper() in ("GAMMA", "G") else label
        sym_labels.append(name)
        ax.axvline(k_dist[i], color="white", linewidth=0.5, alpha=0.7)

ax.set_xticks(sym_k)
ax.set_xticklabels(sym_labels, fontsize=13)
ax.set_ylim(E_min, E_max)
ax.set_ylabel("$E - E_F$ (eV)", fontsize=14)
ax.set_title("Unfolded Band Structure (VASP)", fontsize=14)
fig.colorbar(im, ax=ax, label="Spectral weight")
plt.tight_layout()
plt.savefig("vasp_unfolded_bands.png", dpi=200, bbox_inches="tight")
plt.close()
print("Saved: vasp_unfolded_bands.png")
```

---

## Key Parameters

| Parameter | Typical Value | Notes |
|---|---|---|
| Transformation matrix **M** | 2x2x2, 3x3x3, etc. | Relates supercell to primitive cell: A_super = M @ A_prim |
| `npts_per_seg` | 20--40 | Points per k-path segment in primitive BZ |
| `sigma` (broadening) | 0.02--0.10 eV | Gaussian broadening for spectral function. Smaller = sharper but noisier. |
| `nbnd` | All occupied + some empty | Supercell has N times more bands than primitive cell |
| `ecutwfc` / `ENCUT` | Same as primitive cell | Must be converged for the pseudopotential |
| SCF k-grid | 4x4x4 for 2x2x2 super | Equivalent to 8x8x8 in primitive cell (scales with M) |
| `verbosity` | `'high'` | QE: required to print eigenvalues |
| `LORBIT` | `11` | VASP: write PROCAR for orbital projections |

## Interpreting Results

- **Bright bands**: High spectral weight indicates that the supercell eigenstate closely resembles a primitive-cell Bloch state. These are the "preserved" bands.
- **Dim/smeared bands**: Low spectral weight indicates that the perturbation has mixed different primitive bands. The smearing quantifies the disorder.
- **Band broadening**: A sharp primitive band that becomes broad after unfolding indicates strong scattering by the perturbation (defect, disorder).
- **New states in the gap**: Defect states appear as new features inside the band gap. Their spectral weight indicates how localized they are in k-space.
- **Perfect unfolding**: For an unperturbed supercell, the unfolded bands should exactly reproduce the primitive band structure with weight 1.
- **Alloy disorder**: Random alloys show diffuse bands with reduced spectral weight. The band edges may shift and broaden.

## Common Issues

| Problem | Solution |
|---|---|
| Unfolded bands don't match primitive bands for pristine supercell | Check the transformation matrix M. Verify k-point mapping: k_super = M^{-T} @ K_prim. |
| All weights are equal (no structure in spectral function) | The simplified equal-weight approach does not distinguish bands. Use wavefunction-based weights (plane-wave coefficients from QE `wfc*.dat` or VASP `WAVECAR`). |
| Too many bands to plot | Restrict the energy range. Use `nbnd` to limit the number of computed bands. |
| Spectral function is too noisy | Increase Gaussian broadening `sigma`. Use more k-points per segment. |
| Spectral function is too smeared | Decrease `sigma`. Values of 0.02--0.05 eV are typical. |
| k-point mapping is wrong for non-diagonal M | Ensure M_inv_T is computed correctly. For non-diagonal transformations (e.g., hexagonal supercells), the mapping is more complex. |
| Memory issues with large supercells | Large supercells have many bands. Reduce `nbnd` or use k-point parallelization. |
| Need accurate spectral weights | The equal-weight approximation is qualitative. For quantitative unfolding, extract plane-wave coefficients from QE (read `wfc*.dat` with Python) or VASP (parse `WAVECAR` with `pymatgen.io.vasp.outputs.Wavecar`). |
