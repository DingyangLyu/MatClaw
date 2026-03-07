# Wavefunction Parity Analysis for Topological Classification

## When to Use

- Determine parity eigenvalues of Bloch states at time-reversal invariant momentum (TRIM) points.
- Compute the Z2 topological invariant using the Fu-Kane parity criterion.
- Classify materials as topological insulators (TI) or trivial insulators.
- The system must have both time-reversal symmetry and inversion symmetry.
- Corresponds to VASPKIT tasks 515--516.

## Method Selection

| Criterion | QE (projwfc.x + symmetry) | VASP (OUTCAR/WAVECAR) | z2pack (automated) |
|---|---|---|---|
| Parity eigenvalues | Parse from pw.x verbose output | Parse from OUTCAR | Computed via Wilson loops (does not need parity) |
| Requires inversion symmetry | Yes | Yes | No (works for any TRS system) |
| SOC support | noncolin + lspinorb | LSORBIT | Via QE or VASP backend |
| Automation | Moderate | Moderate | High |

## Prerequisites

- Quantum ESPRESSO 7.5 (`pw.x`) with SOC support
- Fully relativistic pseudopotentials (FR-ONCV or PSlibrary `_rel` PPs)
- `noncolin = .true.`, `lspinorb = .true.` in QE input
- Python packages: `numpy`, `matplotlib`, `spglib`, `pymatgen`
- The material must have inversion symmetry (check space group first)
- For VASP: OUTCAR from SOC calculation with ISYM enabled

---

## Detailed Steps

### Background: Fu-Kane Parity Criterion

For a 3D time-reversal invariant insulator with inversion symmetry, the Z2
topological invariants (nu_0; nu_1 nu_2 nu_3) are determined by the parity
eigenvalues of the occupied Kramers pairs at the 8 TRIM points:

    (-1)^{nu_0} = product_{i=1}^{8} delta_i

where delta_i = product_{m=1}^{N} xi_{2m}(Gamma_i), xi_{2m} is the parity
eigenvalue of the 2m-th occupied band at TRIM point Gamma_i, and N is the
number of occupied Kramers pairs.

For 2D systems, there are 4 TRIM points and one Z2 invariant.

### Step 1: Verify Inversion Symmetry

```python
#!/usr/bin/env python3
"""
Check if a structure has inversion symmetry using spglib.
The Fu-Kane parity criterion only applies to centrosymmetric systems.
"""
import numpy as np
import spglib
from pymatgen.core import Structure
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer

# Load structure
# structure = Structure.from_file("POSCAR")
# For demo: Bi2Se3 (rhombohedral, space group R-3m, #166 -- has inversion)
from pymatgen.core import Lattice
a = 4.138
c = 28.64
lattice = Lattice.hexagonal(a, c)
structure = Structure(
    lattice,
    ["Bi", "Bi", "Se", "Se", "Se",
     "Bi", "Bi", "Se", "Se", "Se",
     "Bi", "Bi", "Se", "Se", "Se"],
    [[0.0, 0.0, 0.4008], [0.0, 0.0, 0.5992],
     [0.0, 0.0, 0.0000], [0.0, 0.0, 0.2117], [0.0, 0.0, 0.7883],
     [1/3, 2/3, 0.7341], [1/3, 2/3, 0.9326],
     [1/3, 2/3, 0.3333], [1/3, 2/3, 0.5451], [1/3, 2/3, 0.1216],
     [2/3, 1/3, 0.0674], [2/3, 1/3, 0.2659],
     [2/3, 1/3, 0.6667], [2/3, 1/3, 0.8784], [2/3, 1/3, 0.4549]],
)

sga = SpacegroupAnalyzer(structure, symprec=0.01)
spg = sga.get_space_group_symbol()
spg_number = sga.get_space_group_number()

print(f"Space group: {spg} (#{spg_number})")

# Check for inversion symmetry
has_inversion = False
sym_ops = sga.get_symmetry_operations()
for op in sym_ops:
    rot = op.rotation_matrix
    if np.allclose(rot, -np.eye(3)):
        has_inversion = True
        break

print(f"Has inversion symmetry: {has_inversion}")

if not has_inversion:
    print("WARNING: Fu-Kane parity criterion requires inversion symmetry.")
    print("Use Wilson loop method (z2pack) instead.")
```

### Step 2: Identify TRIM Points

```python
#!/usr/bin/env python3
"""
Identify the 8 TRIM (Time-Reversal Invariant Momentum) points for a 3D BZ.
TRIM points satisfy: k = -k + G (i.e., 2k is a reciprocal lattice vector).
In fractional coordinates, each component is 0 or 0.5.
"""
import numpy as np
from itertools import product


def get_trim_points_3d():
    """
    Return the 8 TRIM points for a 3D Brillouin zone.
    Each TRIM has fractional coordinates with components in {0, 0.5}.
    """
    trims = []
    labels = []
    for n1, n2, n3 in product([0.0, 0.5], repeat=3):
        trims.append(np.array([n1, n2, n3]))
        label = "Gamma" if (n1 == 0 and n2 == 0 and n3 == 0) else \
                f"({n1:.1f},{n2:.1f},{n3:.1f})"
        labels.append(label)
    return trims, labels


def get_trim_points_2d():
    """
    Return the 4 TRIM points for a 2D Brillouin zone.
    Components in {0, 0.5} for the two in-plane directions.
    """
    trims = []
    labels = []
    for n1, n2 in product([0.0, 0.5], repeat=2):
        trims.append(np.array([n1, n2, 0.0]))  # kz = 0 for 2D
        label = "Gamma" if (n1 == 0 and n2 == 0) else f"({n1:.1f},{n2:.1f})"
        labels.append(label)
    return trims, labels


trims_3d, labels_3d = get_trim_points_3d()
print("=== 3D TRIM Points ===")
for trim, label in zip(trims_3d, labels_3d):
    print(f"  {label:20s}: ({trim[0]:.1f}, {trim[1]:.1f}, {trim[2]:.1f})")

trims_2d, labels_2d = get_trim_points_2d()
print("\n=== 2D TRIM Points ===")
for trim, label in zip(trims_2d, labels_2d):
    print(f"  {label:20s}: ({trim[0]:.1f}, {trim[1]:.1f})")
```

### Step 3: QE Calculation at TRIM Points with SOC

```python
#!/usr/bin/env python3
"""
Run QE SCF + NSCF at TRIM points with SOC to extract parity eigenvalues.
The key is verbosity='high' which prints the symmetry representation
of each eigenstate, including parity (+ or -) under inversion.
"""
import os
import subprocess
import numpy as np
from itertools import product

PSEUDO_DIR = os.path.abspath("./pseudo")
OUTDIR = os.path.abspath("./tmp_parity")
os.makedirs(OUTDIR, exist_ok=True)
os.makedirs(PSEUDO_DIR, exist_ok=True)

PREFIX = "bi2se3"

# ── Step 1: SCF with SOC (dense k-grid for charge density) ───────
scf_input = f"""&CONTROL
    calculation   = 'scf'
    prefix        = '{PREFIX}'
    outdir        = '{OUTDIR}'
    pseudo_dir    = '{PSEUDO_DIR}'
    verbosity     = 'high'
/
&SYSTEM
    ibrav         = 0
    nat           = 15
    ntyp          = 2
    ecutwfc       = 40.0
    ecutrho       = 320.0
    occupations   = 'smearing'
    smearing      = 'mv'
    degauss       = 0.01
    noncolin      = .true.
    lspinorb      = .true.
/
&ELECTRONS
    conv_thr      = 1.0d-8
    mixing_beta   = 0.3
/

CELL_PARAMETERS angstrom
  4.138000   0.000000   0.000000
 -2.069000   3.583426   0.000000
  0.000000   0.000000  28.640000

ATOMIC_SPECIES
  Bi  208.980  Bi.rel-pbe-dn-kjpaw_psl.1.0.0.UPF
  Se   78.960  Se.rel-pbe-dn-kjpaw_psl.1.0.0.UPF

ATOMIC_POSITIONS crystal
  Bi  0.0000  0.0000  0.4008
  Bi  0.0000  0.0000  0.5992
  Se  0.0000  0.0000  0.0000
  Se  0.0000  0.0000  0.2117
  Se  0.0000  0.0000  0.7883
  Bi  0.3333  0.6667  0.7341
  Bi  0.3333  0.6667  0.9326
  Se  0.3333  0.6667  0.3333
  Se  0.3333  0.6667  0.5451
  Se  0.3333  0.6667  0.1216
  Bi  0.6667  0.3333  0.0674
  Bi  0.6667  0.3333  0.2659
  Se  0.6667  0.3333  0.6667
  Se  0.6667  0.3333  0.8784
  Se  0.6667  0.3333  0.4549

K_POINTS automatic
  6 6 2  0 0 0
"""

with open(f"{PREFIX}_scf.in", "w") as f:
    f.write(scf_input)

print("[1/2] Running SCF with SOC...")
r = subprocess.run(
    ["mpirun", "-np", "4", "pw.x", "-in", f"{PREFIX}_scf.in"],
    capture_output=True, text=True, timeout=3600
)
with open(f"{PREFIX}_scf.out", "w") as f:
    f.write(r.stdout)

if "convergence has been achieved" not in r.stdout:
    print("WARNING: SCF may not have converged!")

# ── Step 2: NSCF at TRIM points only ─────────────────────────────
trim_points = []
for n1, n2, n3 in product([0.0, 0.5], repeat=3):
    trim_points.append((n1, n2, n3))

kpoints_card = f"K_POINTS crystal\n{len(trim_points)}\n"
for kp in trim_points:
    kpoints_card += f"  {kp[0]:.4f}  {kp[1]:.4f}  {kp[2]:.4f}  {1.0/len(trim_points):.6f}\n"

nscf_input = f"""&CONTROL
    calculation   = 'nscf'
    prefix        = '{PREFIX}'
    outdir        = '{OUTDIR}'
    pseudo_dir    = '{PSEUDO_DIR}'
    verbosity     = 'high'
/
&SYSTEM
    ibrav         = 0
    nat           = 15
    ntyp          = 2
    ecutwfc       = 40.0
    ecutrho       = 320.0
    occupations   = 'smearing'
    smearing      = 'mv'
    degauss       = 0.01
    noncolin      = .true.
    lspinorb      = .true.
    nosym         = .false.
    nbnd          = 120
/
&ELECTRONS
    conv_thr      = 1.0d-8
/

CELL_PARAMETERS angstrom
  4.138000   0.000000   0.000000
 -2.069000   3.583426   0.000000
  0.000000   0.000000  28.640000

ATOMIC_SPECIES
  Bi  208.980  Bi.rel-pbe-dn-kjpaw_psl.1.0.0.UPF
  Se   78.960  Se.rel-pbe-dn-kjpaw_psl.1.0.0.UPF

ATOMIC_POSITIONS crystal
  Bi  0.0000  0.0000  0.4008
  Bi  0.0000  0.0000  0.5992
  Se  0.0000  0.0000  0.0000
  Se  0.0000  0.0000  0.2117
  Se  0.0000  0.0000  0.7883
  Bi  0.3333  0.6667  0.7341
  Bi  0.3333  0.6667  0.9326
  Se  0.3333  0.6667  0.3333
  Se  0.3333  0.6667  0.5451
  Se  0.3333  0.6667  0.1216
  Bi  0.6667  0.3333  0.0674
  Bi  0.6667  0.3333  0.2659
  Se  0.6667  0.3333  0.6667
  Se  0.6667  0.3333  0.8784
  Se  0.6667  0.3333  0.4549

{kpoints_card}
"""

with open(f"{PREFIX}_nscf_trim.in", "w") as f:
    f.write(nscf_input)

print("[2/2] Running NSCF at TRIM points...")
r = subprocess.run(
    ["mpirun", "-np", "4", "pw.x", "-in", f"{PREFIX}_nscf_trim.in"],
    capture_output=True, text=True, timeout=3600
)
with open(f"{PREFIX}_nscf_trim.out", "w") as f:
    f.write(r.stdout)

print("NSCF completed. Parse the output for parity eigenvalues.")
```

### Step 4: Parse Parity Eigenvalues from QE Output

```python
#!/usr/bin/env python3
"""
Parse parity eigenvalues from QE pw.x output with verbosity='high'.

When inversion symmetry is present, QE labels each eigenstate with its
irreducible representation. At TRIM points with inversion, the representations
include parity labels (e.g., G_7+ and G_7- for even/odd parity in double groups).

For SOC (double group): representations with '+' superscript are even parity,
'-' superscript are odd parity.
"""
import re
import numpy as np
from itertools import product


def parse_parity_from_qe(nscf_output, n_occupied_pairs):
    """
    Parse parity eigenvalues from QE NSCF output at TRIM points.

    In SOC calculations with double group symmetry, QE prints lines like:
        e(  1) =    -8.1234   eV   (rep  1,  G_7+)
    The '+' or '-' after the representation label indicates parity.

    Parameters
    ----------
    nscf_output : str
        Path to the NSCF output file.
    n_occupied_pairs : int
        Number of occupied Kramers pairs (= N_electrons / 2 for insulators with SOC).

    Returns
    -------
    parity_data : dict
        Keys are TRIM point labels, values are lists of parity eigenvalues (+1 or -1)
        for each occupied Kramers pair.
    delta : dict
        delta_i = product of parity eigenvalues at each TRIM point.
    """
    with open(nscf_output, "r") as f:
        content = f.read()

    # Find k-point blocks
    # Pattern: k = x.xxxx y.yyyy z.zzzz (  Nnn PWs)   bands (ev):
    kpt_blocks = re.split(r"\n\s*k\s*=", content)

    parity_data = {}
    trim_labels = []

    for block in kpt_blocks[1:]:
        # Extract k-point coordinates
        m = re.match(r"\s*([-\d.]+)\s+([-\d.]+)\s+([-\d.]+)", block)
        if not m:
            continue
        kx, ky, kz = float(m.group(1)), float(m.group(2)), float(m.group(3))

        # Check if this is a TRIM point (components are 0 or 0.5, mod 1)
        def is_trim_component(x):
            x_mod = x % 1.0
            return abs(x_mod) < 0.01 or abs(x_mod - 0.5) < 0.01

        if not (is_trim_component(kx) and is_trim_component(ky) and is_trim_component(kz)):
            continue

        label = f"({kx % 1:.1f},{ky % 1:.1f},{kz % 1:.1f})"
        if label == "(0.0,0.0,0.0)":
            label = "Gamma"

        # Parse eigenvalues and their representations
        # Pattern: e(  N) =   E.EEEE   eV   (rep  R,  LABEL+)  or  LABEL-)
        pattern = r"e\(\s*(\d+)\)\s*=\s*([-\d.]+)\s*eV.*?\((.*?)\)"
        matches = re.findall(pattern, block)

        parities = []
        for band_idx_str, energy_str, rep_str in matches:
            band_idx = int(band_idx_str)
            # Extract parity from representation label
            if "+" in rep_str:
                parity = +1
            elif "-" in rep_str:
                parity = -1
            else:
                parity = 0  # Unknown parity

            parities.append({
                "band": band_idx,
                "energy": float(energy_str),
                "parity": parity,
                "rep": rep_str.strip(),
            })

        # For Kramers pairs: bands come in degenerate pairs (2n-1, 2n)
        # Take the parity of one state from each pair
        kramers_parities = []
        for i in range(0, min(2 * n_occupied_pairs, len(parities)), 2):
            if i < len(parities):
                kramers_parities.append(parities[i]["parity"])

        parity_data[label] = {
            "kpoint": (kx, ky, kz),
            "parities": kramers_parities[:n_occupied_pairs],
            "all_bands": parities,
        }
        trim_labels.append(label)

    # Compute delta_i for each TRIM point
    delta = {}
    for label, data in parity_data.items():
        p = data["parities"]
        if p and all(x != 0 for x in p):
            delta[label] = int(np.prod(p))
        else:
            delta[label] = None
            print(f"WARNING: Cannot determine parity at {label}")

    return parity_data, delta


def compute_z2_from_parity(delta):
    """
    Compute Z2 invariants from delta values at TRIM points.

    For 3D: (nu_0; nu_1, nu_2, nu_3)
      (-1)^{nu_0} = product of all 8 delta_i
      (-1)^{nu_k} = product of 4 delta_i in the k-th plane

    Returns dict with Z2 invariants.
    """
    # 3D TRIM points in order
    trim_3d = []
    for n1, n2, n3 in product([0.0, 0.5], repeat=3):
        label = "Gamma" if (n1 == 0 and n2 == 0 and n3 == 0) else \
                f"({n1:.1f},{n2:.1f},{n3:.1f})"
        trim_3d.append((label, n1, n2, n3))

    # Strong invariant nu_0: product of all 8
    all_deltas = []
    for label, n1, n2, n3 in trim_3d:
        if label in delta and delta[label] is not None:
            all_deltas.append(delta[label])
        else:
            print(f"Missing delta for {label}, cannot compute Z2")
            return None

    prod_all = int(np.prod(all_deltas))
    nu_0 = 0 if prod_all == 1 else 1

    # Weak invariants nu_1, nu_2, nu_3
    # nu_k: product of delta_i for TRIM points in the k_k = pi/a_k plane
    weak = []
    for axis in range(3):
        plane_deltas = []
        for label, n1, n2, n3 in trim_3d:
            ns = [n1, n2, n3]
            if abs(ns[axis] - 0.5) < 0.01:
                if label in delta and delta[label] is not None:
                    plane_deltas.append(delta[label])
        prod_plane = int(np.prod(plane_deltas)) if plane_deltas else 1
        nu_k = 0 if prod_plane == 1 else 1
        weak.append(nu_k)

    result = {
        "nu_0": nu_0,
        "nu_1": weak[0],
        "nu_2": weak[1],
        "nu_3": weak[2],
        "is_topological": nu_0 == 1,
    }

    print(f"\n=== Z2 Topological Invariants ===")
    print(f"(nu_0; nu_1, nu_2, nu_3) = ({nu_0}; {weak[0]}, {weak[1]}, {weak[2]})")
    if nu_0 == 1:
        print("=> STRONG topological insulator")
    elif any(w == 1 for w in weak):
        print("=> WEAK topological insulator")
    else:
        print("=> Trivial insulator")

    return result


# ── Example usage ─────────────────────────────────────────────────
import os

nscf_file = "bi2se3_nscf_trim.out"
if os.path.exists(nscf_file):
    # Bi2Se3: 5 atoms per formula unit, 3 f.u. per cell = 15 atoms
    # Total electrons with SOC: ~ 138 valence electrons => 69 Kramers pairs
    N_OCCUPIED_PAIRS = 69  # Adjust for your system

    parity_data, delta = parse_parity_from_qe(nscf_file, N_OCCUPIED_PAIRS)

    print("=== Parity at TRIM Points ===")
    for label, d in sorted(parity_data.items()):
        parities = d["parities"]
        delta_val = delta.get(label, "?")
        print(f"  {label:20s}: delta = {delta_val}, "
              f"parities = {parities[:5]}...")

    z2 = compute_z2_from_parity(delta)
else:
    print(f"Output file '{nscf_file}' not found.")
    print("Run the QE NSCF calculation at TRIM points first.")
    print("\nDemo with Bi2Se3 expected result:")
    print("(nu_0; nu_1, nu_2, nu_3) = (1; 0, 0, 0) => Strong TI")
```

### Method B: VASP Parity Analysis

```python
#!/usr/bin/env python3
"""
Parse parity eigenvalues from VASP OUTCAR at TRIM points.

VASP with ISYM > 0 and SOC (LSORBIT=.TRUE.) prints the irreducible
representations of each eigenstate. With inversion symmetry, these include
parity labels (g = even, u = odd, or + and -).

INCAR settings needed:
  LSORBIT = .TRUE.
  ISYM    = 2        (use symmetry)
  LWAVE   = .TRUE.   (write WAVECAR for analysis)
  PREC    = Accurate

KPOINTS file should contain only the TRIM points in explicit list mode.
"""
import re
import numpy as np
from itertools import product


def parse_vasp_parity(outcar_file, n_occupied_pairs):
    """
    Parse parity from VASP OUTCAR at TRIM points.

    VASP prints representations like:
      band   1 # energy  -10.1234 # occ. 1.00000 # rep:  DG7+ (1)
    where DG7+ means double group representation 7 with even (+) parity.
    """
    with open(outcar_file, "r") as f:
        content = f.read()

    # Find k-point blocks
    blocks = re.split(r"k-point\s+(\d+)\s*:", content)

    parity_data = {}

    for i in range(1, len(blocks), 2):
        kpt_idx = int(blocks[i])
        block = blocks[i + 1]

        # Extract k-point coordinates
        m = re.search(r"([-\d.]+)\s+([-\d.]+)\s+([-\d.]+)", block)
        if not m:
            continue
        kx, ky, kz = float(m.group(1)), float(m.group(2)), float(m.group(3))

        # Check if TRIM
        def is_trim(x):
            x_mod = x % 1.0
            return abs(x_mod) < 0.01 or abs(x_mod - 0.5) < 0.01 or abs(x_mod - 1.0) < 0.01

        if not (is_trim(kx) and is_trim(ky) and is_trim(kz)):
            continue

        label = f"({kx%1:.1f},{ky%1:.1f},{kz%1:.1f})"
        if label == "(0.0,0.0,0.0)":
            label = "Gamma"

        # Parse bands with representations
        # Pattern: band  N # energy  E # occ. O # rep:  LABEL
        band_pattern = r"band\s+(\d+)\s*#\s*energy\s+([-\d.]+)\s*#\s*occ\.\s+([\d.]+)\s*#\s*rep:\s*(\S+)"
        matches = re.findall(band_pattern, block)

        parities = []
        for band_str, energy_str, occ_str, rep_label in matches:
            if "+" in rep_label:
                parity = +1
            elif "-" in rep_label:
                parity = -1
            else:
                parity = 0  # undetermined

            parities.append({
                "band": int(band_str),
                "energy": float(energy_str),
                "occ": float(occ_str),
                "parity": parity,
                "rep": rep_label,
            })

        # Extract Kramers pair parities (every other band)
        kramers_parities = []
        for j in range(0, min(2 * n_occupied_pairs, len(parities)), 2):
            kramers_parities.append(parities[j]["parity"])

        parity_data[label] = {
            "kpoint": (kx, ky, kz),
            "parities": kramers_parities[:n_occupied_pairs],
        }

    return parity_data


# Usage:
# parity_data = parse_vasp_parity("OUTCAR", n_occupied_pairs=69)
# Then use compute_z2_from_parity() from Method A (Step 4) to get Z2 invariants.
```

### Step 5: Visualization of Parity Table

```python
#!/usr/bin/env python3
"""
Create a publication-quality table and figure showing parity eigenvalues
at TRIM points and the resulting Z2 invariant.
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from itertools import product


def plot_parity_table(parity_data, delta, z2_result, output_png="parity_table.png"):
    """
    Create a visual summary of parity analysis.
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5),
                              gridspec_kw={"width_ratios": [2, 1]})

    # Left panel: parity table
    ax = axes[0]
    ax.axis("off")

    trim_labels = sorted(parity_data.keys())
    n_trims = len(trim_labels)

    # Table header
    col_labels = trim_labels
    row_labels = ["delta_i"]

    cell_text = []
    cell_colors = []
    delta_row = []
    delta_colors = []
    for label in trim_labels:
        d = delta.get(label, "?")
        delta_row.append(f"{d:+d}" if isinstance(d, int) else str(d))
        if d == +1:
            delta_colors.append("#90EE90")  # light green
        elif d == -1:
            delta_colors.append("#FFB6C1")  # light red
        else:
            delta_colors.append("#FFFFFF")
    cell_text.append(delta_row)
    cell_colors.append(delta_colors)

    table = ax.table(
        cellText=cell_text,
        cellColours=cell_colors,
        colLabels=col_labels,
        rowLabels=row_labels,
        loc="center",
        cellLoc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1.0, 2.0)
    ax.set_title("Parity Products at TRIM Points", fontsize=14, pad=20)

    # Right panel: Z2 result
    ax2 = axes[1]
    ax2.axis("off")

    if z2_result:
        nu = f"({z2_result['nu_0']}; {z2_result['nu_1']}, {z2_result['nu_2']}, {z2_result['nu_3']})"
        classification = "Strong TI" if z2_result["nu_0"] == 1 else \
                         "Weak TI" if any(z2_result[f"nu_{i}"] == 1 for i in range(1, 4)) else \
                         "Trivial"
        color = "green" if z2_result["is_topological"] else "gray"
    else:
        nu = "N/A"
        classification = "Unknown"
        color = "gray"

    ax2.text(0.5, 0.6, f"Z2 = {nu}", fontsize=20, ha="center", va="center",
             fontweight="bold", transform=ax2.transAxes)
    ax2.text(0.5, 0.35, classification, fontsize=16, ha="center", va="center",
             color=color, fontweight="bold", transform=ax2.transAxes)

    plt.tight_layout()
    plt.savefig(output_png, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"Saved: {output_png}")


# Demo with Bi2Se3 expected data
demo_delta = {
    "Gamma": -1,
    "(0.5,0.0,0.0)": +1,
    "(0.0,0.5,0.0)": +1,
    "(0.0,0.0,0.5)": +1,
    "(0.5,0.5,0.0)": +1,
    "(0.5,0.0,0.5)": +1,
    "(0.0,0.5,0.5)": +1,
    "(0.5,0.5,0.5)": +1,
}

demo_z2 = {
    "nu_0": 1, "nu_1": 0, "nu_2": 0, "nu_3": 0,
    "is_topological": True,
}

demo_parity = {k: {"parities": []} for k in demo_delta}

plot_parity_table(demo_parity, demo_delta, demo_z2, "parity_bi2se3_demo.png")
```

---

## Key Parameters

| Parameter | Typical Value | Notes |
|---|---|---|
| `noncolin` | `.true.` | Required for SOC |
| `lspinorb` | `.true.` | Enables spin-orbit coupling |
| `verbosity` | `'high'` | Required to print symmetry representations |
| `nosym` | `.false.` | Symmetry must be enabled for parity labels |
| Pseudopotentials | Fully relativistic (`_rel` or `_fr`) | Scalar-relativistic PPs cannot do SOC |
| `nbnd` | 1.5-2x occupied bands | Include enough empty bands for complete analysis |
| K-points | Only TRIM points for NSCF | 8 points for 3D, 4 for 2D |

## Interpreting Results

- **delta_i = +1**: Even number of parity inversions among occupied Kramers pairs at this TRIM.
- **delta_i = -1**: Odd number of parity inversions (band inversion occurred at this TRIM).
- **nu_0 = 1**: Strong topological insulator. Robust surface states on every surface.
  Example: Bi2Se3 has (1; 0,0,0).
- **nu_0 = 0 with weak indices nonzero**: Weak topological insulator. Surface states only
  on certain surfaces. Example: Bi bilayer stacks.
- **All nu = 0**: Trivial insulator. No topologically protected surface states.
- **Bi2Se3**: The band inversion occurs at Gamma, giving delta(Gamma) = -1, all other
  delta = +1, hence nu_0 = 1.
- **If parity labels are not printed**: QE may not recognize inversion symmetry for the
  given structure. Use `verbosity='high'` and ensure the structure is properly symmetrized.

## Common Issues

| Problem | Solution |
|---|---|
| QE does not print symmetry representations | Ensure `nosym=.false.` and `verbosity='high'`. Structure must be properly symmetrized (use `spglib` to standardize) |
| Parity labels are ambiguous | For double groups with SOC, look for `+` (even/gerade) and `-` (odd/ungerade) in the representation label |
| Wrong number of Kramers pairs | With SOC, each Kramers pair is doubly degenerate. Count total valence electrons / 2 |
| Band inversion not captured | Ensure pseudopotentials include the correct valence states (e.g., Bi 6p for Bi2Se3) |
| System lacks inversion symmetry | Fu-Kane criterion does not apply. Use Wilson loop method (z2pack) instead |
| SOC calculation does not converge | Reduce `mixing_beta` to 0.2. SOC calculations need tighter convergence |
| Different convention for TRIM points | VASP uses Cartesian k-points internally. Convert to fractional before checking TRIM |
