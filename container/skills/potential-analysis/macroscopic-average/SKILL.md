# Macroscopic Averaging for Interface Analysis

## When to Use

- Determine band offsets (valence band offset, conduction band offset) at heterojunctions.
- Calculate interface dipoles at metal/semiconductor or semiconductor/semiconductor interfaces.
- Obtain smooth potential profiles for Schottky barrier estimation.
- Apply the double-averaging technique (Baldereschi, Baroni, Resta) to remove atomic-scale oscillations.
- Corresponds to VASPKIT tasks 425--427 (macroscopic average of potential/charge).

## Method Selection

| Criterion | QE (average.x) | Python (custom) | VASP (LOCPOT + Python) |
|---|---|---|---|
| Built-in tool | average.x with `awin > 0` | Full control | Python post-processing |
| Double averaging | Set `awin` to interlayer spacing | Explicit convolution | Explicit convolution |
| Flexibility | Fixed window | Any window shape | Any window shape |
| Automation | Moderate | High | High |

## Prerequisites

- Quantum ESPRESSO 7.5 (`pw.x`, `pp.x`, `average.x`)
- Python packages: `numpy`, `scipy`, `matplotlib`, `ase`
- Converged slab or heterostructure calculation
- Knowledge of interlayer spacings for the averaging window
- For VASP: LOCPOT file from the interface calculation

---

## Detailed Steps

### Background: The Macroscopic Averaging Technique

The planar average of the potential still shows atomic-scale oscillations:

    V_planar(z) = (1/A) integral V(x,y,z) dx dy

The macroscopic average smooths these by convolving with a window function:

    V_macro(z) = (1/L) integral_{z-L/2}^{z+L/2} V_planar(z') dz'

where L is the period of the bulk material (one interlayer spacing along z).

For a heterostructure A/B, a double average may be needed: first average with period L_A,
then average again with period L_B. This removes oscillations from both materials.

### Method A: QE -- average.x with Macroscopic Averaging

#### Step A1: Extract Potential and Run average.x

```python
#!/usr/bin/env python3
"""
Extract electrostatic potential from a heterostructure SCF calculation
and compute the macroscopic average using average.x.

Example: Si/Ge(001) interface.
"""
import os
import subprocess
import numpy as np

OUTDIR = os.path.abspath("./tmp")
PREFIX = "interface"

# ── Step 1: pp.x to extract electrostatic potential ───────────────
pp_input = f"""&INPUTPP
    prefix  = '{PREFIX}'
    outdir  = '{OUTDIR}'
    filplot = 'interface_potential.dat'
    plot_num = 11
/
"""

with open("pp_interface.in", "w") as f:
    f.write(pp_input)

print("Running pp.x to extract electrostatic potential...")
result = subprocess.run(
    ["pp.x", "-in", "pp_interface.in"],
    capture_output=True, text=True, timeout=300
)
with open("pp_interface.out", "w") as f:
    f.write(result.stdout)

# ── Step 2: average.x with macroscopic averaging ─────────────────
# For a Si/Ge interface along z:
# Si interlayer spacing d_Si = a_Si / 4 * sqrt(3) ~ 1.36 Ang for (111)
# or d_Si = a_Si / 2 ~ 2.72 Ang for (001)
# The averaging window should be one period of the bulk oscillation.
#
# In average.x, awin is in units of the cell parameter along idir.
# If cell parameter along z is c_total, and the interlayer spacing is d,
# then awin = d / c_total.

# For demonstration, use a fraction of the cell:
C_TOTAL = 40.0  # total cell height in Angstrom (example)
D_LAYER = 2.72  # interlayer spacing in Angstrom (Si (001))
AWIN = D_LAYER / C_TOTAL  # fraction of cell parameter

avg_stdin = f"""1
interface_potential.dat
1.0
500
3
{AWIN:.6f}
"""

print(f"Running average.x with awin = {AWIN:.6f} (d = {D_LAYER:.2f} Ang)...")
result = subprocess.run(
    ["average.x"],
    input=avg_stdin,
    capture_output=True, text=True, timeout=120
)
with open("average_interface.out", "w") as f:
    f.write(result.stdout)

if result.returncode == 0:
    print("average.x completed.")
else:
    print("ERROR in average.x!")
```

#### Step A2: Double Macroscopic Average in Python

```python
#!/usr/bin/env python3
"""
Compute single and double macroscopic averages of the planar-averaged potential.
The double average is needed for heterostructures with two different periodicities.

Double averaging technique (Baldereschi, Baroni, Resta, PRL 1988):
  1) First convolve V_planar(z) with a window of period L_A.
  2) Then convolve the result with a window of period L_B.
This removes oscillations from both materials A and B.
"""
import numpy as np
from scipy.signal import fftconvolve
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def macroscopic_average(z, v, window_length):
    """
    Compute the macroscopic average of V(z) using a rectangular window.

    Parameters
    ----------
    z : np.ndarray
        Position array (uniformly spaced, in Angstrom).
    v : np.ndarray
        Planar-averaged values.
    window_length : float
        Averaging window in Angstrom (one bulk period along z).

    Returns
    -------
    v_macro : np.ndarray
        Macroscopic-averaged values (same length as v).
    """
    dz = z[1] - z[0]
    n_window = int(round(window_length / dz))
    if n_window < 1:
        n_window = 1

    # Rectangular window, normalized
    window = np.ones(n_window) / n_window

    # Convolve with periodic boundary conditions
    # Extend the signal periodically
    v_extended = np.concatenate([v, v, v])
    v_conv = fftconvolve(v_extended, window, mode='same')

    # Take the central portion
    n = len(v)
    v_macro = v_conv[n:2*n]

    return v_macro


def double_macroscopic_average(z, v, window_a, window_b):
    """
    Double macroscopic average: first average with period L_A, then with period L_B.

    Parameters
    ----------
    z : np.ndarray
        Position array in Angstrom.
    v : np.ndarray
        Planar-averaged values.
    window_a : float
        First averaging window in Angstrom (period of material A).
    window_b : float
        Second averaging window in Angstrom (period of material B).

    Returns
    -------
    v_double : np.ndarray
        Double macroscopic-averaged values.
    """
    v_single = macroscopic_average(z, v, window_a)
    v_double = macroscopic_average(z, v_single, window_b)
    return v_double


# ── Parse planar average data ─────────────────────────────────────
def parse_average_output(filename):
    """Parse average.x output (z_bohr, V_ry columns)."""
    z_bohr = []
    v_ry = []
    with open(filename, "r") as f:
        for line in f:
            parts = line.strip().split()
            try:
                if len(parts) >= 2:
                    z_bohr.append(float(parts[0]))
                    v_ry.append(float(parts[1]))
            except ValueError:
                continue

    BOHR_TO_ANG = 0.529177
    RY_TO_EV = 13.6057
    z = np.array(z_bohr) * BOHR_TO_ANG
    v = np.array(v_ry) * RY_TO_EV
    return z, v


def planar_average_from_cube(cube_file, axis=2):
    """Compute planar average from a cube file."""
    from ase.io.cube import read_cube_data

    data, atoms = read_cube_data(cube_file)
    cell = atoms.get_cell()
    length = np.linalg.norm(cell[axis])

    avg_axes = tuple(i for i in range(3) if i != axis)
    v_avg = np.mean(data, axis=avg_axes)
    z = np.linspace(0, length, len(v_avg), endpoint=False)

    RY_TO_EV = 13.6057
    v_ev = v_avg * RY_TO_EV
    return z, v_ev


# ── Example: Si/Ge (001) heterojunction ──────────────────────────

# Interlayer spacings along [001]:
# Si: d_Si = a_Si/4 = 5.43/4 = 1.358 Ang (per atomic layer)
#     or a_Si/2 = 2.715 Ang (per double layer = one period of oscillation)
# Ge: d_Ge = a_Ge/4 = 5.66/4 = 1.415 Ang
#     or a_Ge/2 = 2.830 Ang
D_SI = 2.715  # Si (001) period in Angstrom
D_GE = 2.830  # Ge (001) period in Angstrom

# Load data (from average.x output or cube file)
try:
    z, v_planar = parse_average_output("average_interface.out")
    print(f"Loaded average.x output: {len(z)} points")
except Exception:
    try:
        z, v_planar = planar_average_from_cube("interface_potential.cube", axis=2)
        print(f"Loaded cube file: {len(z)} points")
    except Exception:
        # Generate synthetic demo data
        print("No data files found. Using synthetic demo data.")
        z = np.linspace(0, 40, 1000)
        # Simulate potential with different oscillation periods on each side
        v_planar = np.zeros_like(z)
        # Si side (z < 20)
        mask_si = z < 20
        v_planar[mask_si] = -8.0 + 3.0 * np.sin(2 * np.pi * z[mask_si] / D_SI)
        # Ge side (z >= 20)
        mask_ge = z >= 20
        v_planar[mask_ge] = -7.5 + 2.5 * np.sin(2 * np.pi * z[mask_ge] / D_GE)
        # Smooth transition at interface
        transition = 0.5 * (1 + np.tanh((z - 20) / 1.0))
        v_planar = v_planar * (1 - 0.1 * transition)

# Compute averages
v_single_si = macroscopic_average(z, v_planar, D_SI)
v_single_ge = macroscopic_average(z, v_planar, D_GE)
v_double = double_macroscopic_average(z, v_planar, D_SI, D_GE)

# ── Band offset from double average ──────────────────────────────
# The double-averaged potential should show a step at the interface.
# The valence band offset (VBO) is:
#   VBO = (E_v^Ge - V_macro^Ge) - (E_v^Si - V_macro^Si) + (V_macro^Ge - V_macro^Si)
# where E_v is the bulk VBM relative to the average potential.
#
# In practice:
#   1) Compute bulk band alignment for each material separately
#   2) Read the potential step from the interface double average
#   3) Combine

# Identify regions far from the interface
n = len(z)
si_region = slice(n // 10, 2 * n // 10)
ge_region = slice(7 * n // 10, 9 * n // 10)

v_macro_si = np.mean(v_double[si_region])
v_macro_ge = np.mean(v_double[ge_region])
delta_v = v_macro_ge - v_macro_si

print(f"\n=== Interface Potential Step ===")
print(f"V_macro (Si region):  {v_macro_si:.4f} eV")
print(f"V_macro (Ge region):  {v_macro_ge:.4f} eV")
print(f"Delta V (interface):  {delta_v:.4f} eV")
print(f"\nTo get the VBO, combine this with bulk band alignment:")
print(f"VBO = (E_VBM^Ge - V_macro^Ge)_bulk - (E_VBM^Si - V_macro^Si)_bulk + Delta_V")

# ── Plot ──────────────────────────────────────────────────────────
fig, axes = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

# Panel 1: planar and single averages
ax1 = axes[0]
ax1.plot(z, v_planar, "b-", linewidth=0.5, alpha=0.5, label="Planar average")
ax1.plot(z, v_single_si, "r-", linewidth=1.5, label=f"Macro avg ($L_{{Si}}$ = {D_SI:.2f} Ang)")
ax1.set_ylabel("Potential (eV)", fontsize=12)
ax1.legend(fontsize=10)
ax1.grid(True, alpha=0.3)
ax1.set_title("Macroscopic Averaging at Si/Ge Interface", fontsize=14)

# Panel 2: double average
ax2 = axes[1]
ax2.plot(z, v_planar, "b-", linewidth=0.5, alpha=0.3, label="Planar average")
ax2.plot(z, v_double, "k-", linewidth=2.0, label="Double macro average")
ax2.axhline(v_macro_si, color="green", linestyle="--", alpha=0.7,
            label=f"$\\bar{{V}}_{{Si}}$ = {v_macro_si:.2f} eV")
ax2.axhline(v_macro_ge, color="orange", linestyle="--", alpha=0.7,
            label=f"$\\bar{{V}}_{{Ge}}$ = {v_macro_ge:.2f} eV")

# Annotate step
z_mid = z[n // 2]
ax2.annotate("", xy=(z_mid, v_macro_si), xytext=(z_mid, v_macro_ge),
             arrowprops=dict(arrowstyle="<->", color="red", lw=1.5))
ax2.text(z_mid + 0.5, (v_macro_si + v_macro_ge) / 2,
         f"$\\Delta V$ = {delta_v:.3f} eV", fontsize=11, color="red")

ax2.set_xlabel(r"z ($\mathrm{\AA}$)", fontsize=13)
ax2.set_ylabel("Potential (eV)", fontsize=12)
ax2.legend(fontsize=10)
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("macroscopic_average_interface.png", dpi=200, bbox_inches="tight")
print("\nSaved: macroscopic_average_interface.png")
```

### Method B: VASP -- LOCPOT Double Average

```python
#!/usr/bin/env python3
"""
Compute macroscopic and double macroscopic averages from VASP LOCPOT.
"""
import numpy as np
from scipy.signal import fftconvolve
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import re


def parse_locpot_planar(locpot_file, axis=2):
    """
    Parse VASP LOCPOT and return planar average along specified axis.
    """
    with open(locpot_file, "r") as f:
        lines = f.readlines()

    scale = float(lines[1].strip())
    cell = np.zeros((3, 3))
    for i in range(3):
        cell[i] = [float(x) for x in lines[2 + i].split()]
    cell *= scale

    counts = [int(x) for x in lines[6].split()]
    n_atoms = sum(counts)

    coord_line = lines[7].strip()
    coord_start = 9 if coord_line[0] in ("S", "s") else 8
    data_start = coord_start + n_atoms
    while data_start < len(lines) and lines[data_start].strip() == "":
        data_start += 1

    grid = [int(x) for x in lines[data_start].split()]
    nx, ny, nz = grid
    data_start += 1

    values = []
    for i in range(data_start, len(lines)):
        line = lines[i].strip()
        if line == "" and len(values) >= nx * ny * nz:
            break
        try:
            values.extend([float(x) for x in line.split()])
        except ValueError:
            break
        if len(values) >= nx * ny * nz:
            break

    potential = np.array(values[:nx * ny * nz]).reshape((nx, ny, nz), order='F')

    avg_axes = tuple(i for i in range(3) if i != axis)
    v_planar = np.mean(potential, axis=avg_axes)
    length = np.linalg.norm(cell[axis])
    z = np.linspace(0, length, len(v_planar), endpoint=False)

    return z, v_planar


def macroscopic_average(z, v, window_ang):
    """Macroscopic average with window in Angstrom."""
    dz = z[1] - z[0]
    n_win = max(int(round(window_ang / dz)), 1)
    window = np.ones(n_win) / n_win
    v_ext = np.concatenate([v, v, v])
    v_conv = fftconvolve(v_ext, window, mode='same')
    n = len(v)
    return v_conv[n:2*n]


def band_offset_from_locpot(locpot_file, d_a, d_b, axis=2,
                             region_a=(0.1, 0.3), region_b=(0.7, 0.9),
                             output_png="band_offset_vasp.png"):
    """
    Compute interface potential step from VASP LOCPOT using double averaging.

    Parameters
    ----------
    locpot_file : str
        Path to LOCPOT.
    d_a : float
        Interlayer period of material A in Angstrom.
    d_b : float
        Interlayer period of material B in Angstrom.
    axis : int
        Interface normal direction (0, 1, or 2).
    region_a : tuple
        Fractional range along z for material A bulk-like region.
    region_b : tuple
        Fractional range along z for material B bulk-like region.
    output_png : str
        Output figure filename.

    Returns
    -------
    delta_v : float
        Potential step across interface in eV.
    """
    z, v_planar = parse_locpot_planar(locpot_file, axis=axis)
    v_double = macroscopic_average(z, macroscopic_average(z, v_planar, d_a), d_b)

    n = len(z)
    idx_a = slice(int(region_a[0] * n), int(region_a[1] * n))
    idx_b = slice(int(region_b[0] * n), int(region_b[1] * n))

    v_a = np.mean(v_double[idx_a])
    v_b = np.mean(v_double[idx_b])
    delta_v = v_b - v_a

    print(f"V_macro (material A): {v_a:.4f} eV")
    print(f"V_macro (material B): {v_b:.4f} eV")
    print(f"Delta V = {delta_v:.4f} eV")

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(z, v_planar, "b-", linewidth=0.5, alpha=0.4, label="Planar avg")
    ax.plot(z, v_double, "k-", linewidth=2, label="Double macro avg")
    ax.axhline(v_a, color="green", linestyle="--", label=f"$V_A$ = {v_a:.2f} eV")
    ax.axhline(v_b, color="orange", linestyle="--", label=f"$V_B$ = {v_b:.2f} eV")
    ax.set_xlabel(r"z ($\mathrm{\AA}$)", fontsize=13)
    ax.set_ylabel("Potential (eV)", fontsize=13)
    ax.set_title("Double Macroscopic Average (VASP LOCPOT)", fontsize=14)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_png, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"Saved: {output_png}")

    return delta_v


# Usage:
# delta_v = band_offset_from_locpot("LOCPOT", d_a=2.715, d_b=2.830)
```

### Method C: Complete Band Alignment Workflow

```python
#!/usr/bin/env python3
"""
Complete band alignment workflow for a heterojunction.

Three calculations are needed:
  1) Bulk material A: SCF to get E_VBM and V_macro (from the bulk potential).
  2) Bulk material B: SCF to get E_VBM and V_macro (from the bulk potential).
  3) Interface supercell: SCF to get the potential step Delta_V.

The valence band offset (VBO) is:
  VBO = (E_VBM^B - V_macro^B)_bulk - (E_VBM^A - V_macro^A)_bulk + Delta_V

The conduction band offset (CBO) is:
  CBO = VBO + E_gap^A - E_gap^B
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def compute_band_alignment(e_vbm_a, v_macro_bulk_a, e_gap_a,
                           e_vbm_b, v_macro_bulk_b, e_gap_b,
                           delta_v_interface):
    """
    Compute band offsets at a heterojunction.

    Parameters
    ----------
    e_vbm_a : float
        VBM of material A from bulk calculation (eV, absolute).
    v_macro_bulk_a : float
        Macroscopic-averaged potential of bulk A (eV).
    e_gap_a : float
        Band gap of material A (eV).
    e_vbm_b : float
        VBM of material B from bulk calculation (eV, absolute).
    v_macro_bulk_b : float
        Macroscopic-averaged potential of bulk B (eV).
    e_gap_b : float
        Band gap of material B (eV).
    delta_v_interface : float
        V_macro(B) - V_macro(A) from the interface calculation (eV).

    Returns
    -------
    dict with VBO, CBO, and alignment type.
    """
    # VBM position relative to macroscopic potential in each bulk
    e_vbm_rel_a = e_vbm_a - v_macro_bulk_a
    e_vbm_rel_b = e_vbm_b - v_macro_bulk_b

    # Valence band offset
    vbo = e_vbm_rel_b - e_vbm_rel_a + delta_v_interface

    # Conduction band offset
    cbo = vbo + e_gap_a - e_gap_b

    # Determine alignment type
    if vbo > 0 and cbo > 0:
        alignment = "Type I (straddling)"
    elif vbo > 0 and cbo < 0:
        alignment = "Type II (staggered)"
    elif vbo < 0 and cbo > 0:
        alignment = "Type II (staggered)"
    else:
        alignment = "Type III (broken gap)"

    results = {
        "VBO": vbo,
        "CBO": cbo,
        "alignment": alignment,
        "E_VBM_rel_A": e_vbm_rel_a,
        "E_VBM_rel_B": e_vbm_rel_b,
    }

    print(f"=== Band Alignment Results ===")
    print(f"E_VBM(A) - V_macro(A) = {e_vbm_rel_a:.4f} eV")
    print(f"E_VBM(B) - V_macro(B) = {e_vbm_rel_b:.4f} eV")
    print(f"Delta V (interface)   = {delta_v_interface:.4f} eV")
    print(f"VBO = {vbo:.4f} eV")
    print(f"CBO = {cbo:.4f} eV")
    print(f"Alignment type: {alignment}")

    return results


def plot_band_alignment(e_gap_a, e_gap_b, vbo, cbo, label_a="Material A",
                        label_b="Material B", output_png="band_alignment.png"):
    """
    Plot band alignment diagram.
    """
    fig, ax = plt.subplots(figsize=(6, 6))

    # Material A: VBM at 0, CBM at gap_A
    ax.bar(0.5, e_gap_a, bottom=0, width=0.8, color="steelblue", alpha=0.5,
           edgecolor="steelblue", linewidth=2)
    ax.text(0.5, e_gap_a / 2, label_a, ha="center", va="center", fontsize=12,
            fontweight="bold")

    # Material B: VBM at vbo, CBM at vbo + gap_B
    ax.bar(1.8, e_gap_b, bottom=vbo, width=0.8, color="tomato", alpha=0.5,
           edgecolor="tomato", linewidth=2)
    ax.text(1.8, vbo + e_gap_b / 2, label_b, ha="center", va="center", fontsize=12,
            fontweight="bold")

    # VBO annotation
    ax.annotate("", xy=(1.3, 0), xytext=(1.3, vbo),
                arrowprops=dict(arrowstyle="<->", color="black", lw=1.5))
    ax.text(1.35, vbo / 2, f"VBO = {vbo:.3f} eV", fontsize=10, va="center")

    # CBO annotation
    ax.annotate("", xy=(1.3, e_gap_a), xytext=(1.3, vbo + e_gap_b),
                arrowprops=dict(arrowstyle="<->", color="red", lw=1.5))
    ax.text(1.35, (e_gap_a + vbo + e_gap_b) / 2, f"CBO = {cbo:.3f} eV",
            fontsize=10, va="center", color="red")

    # Labels
    ax.axhline(0, color="steelblue", linestyle=":", alpha=0.5)
    ax.axhline(e_gap_a, color="steelblue", linestyle=":", alpha=0.5)
    ax.axhline(vbo, color="tomato", linestyle=":", alpha=0.5)
    ax.axhline(vbo + e_gap_b, color="tomato", linestyle=":", alpha=0.5)

    ax.text(0.05, -0.15, "VBM(A)", fontsize=9, color="steelblue")
    ax.text(0.05, e_gap_a + 0.05, "CBM(A)", fontsize=9, color="steelblue")
    ax.text(2.25, vbo - 0.15, "VBM(B)", fontsize=9, color="tomato")
    ax.text(2.25, vbo + e_gap_b + 0.05, "CBM(B)", fontsize=9, color="tomato")

    ax.set_xlim(-0.2, 2.8)
    y_min = min(0, vbo) - 0.5
    y_max = max(e_gap_a, vbo + e_gap_b) + 0.5
    ax.set_ylim(y_min, y_max)
    ax.set_ylabel("Energy (eV)", fontsize=13)
    ax.set_xticks([0.5, 1.8])
    ax.set_xticklabels([label_a, label_b], fontsize=12)
    ax.set_title("Band Alignment Diagram", fontsize=14)
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_png, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"Saved: {output_png}")


# ── Example: Si/Ge band alignment ────────────────────────────────
# Values from DFT (PBE, illustrative):
results = compute_band_alignment(
    e_vbm_a=-5.0,         # Si VBM (absolute, from bulk SCF)
    v_macro_bulk_a=-10.0,  # Si macroscopic potential (bulk)
    e_gap_a=0.6,           # Si PBE gap
    e_vbm_b=-4.8,          # Ge VBM (absolute, from bulk SCF)
    v_macro_bulk_b=-9.5,   # Ge macroscopic potential (bulk)
    e_gap_b=0.35,          # Ge PBE gap
    delta_v_interface=0.2,  # from interface supercell double average
)

plot_band_alignment(
    e_gap_a=0.6, e_gap_b=0.35,
    vbo=results["VBO"], cbo=results["CBO"],
    label_a="Si", label_b="Ge",
    output_png="si_ge_band_alignment.png"
)
```

---

## Key Parameters

| Parameter | Typical Value | Notes |
|---|---|---|
| Window L_A, L_B | Interlayer spacing of each material along z | For FCC(001): a/2. For FCC(111): a/sqrt(3). Must match the bulk oscillation period |
| Vacuum thickness | 0 for interface supercells | No vacuum needed for A/B superlattice calculations |
| Interface supercell | 8--12 layers of each material | Must be thick enough for bulk-like potential in the interior |
| `awin` in average.x | L / c_total (fractional) | Fraction of total cell parameter, not Angstrom |
| K-points (interface) | Dense in-plane, depends on z | Scale k-grid inversely with supercell size along z |

## Interpreting Results

- **Potential step (Delta V)**: The difference in macroscopic-averaged potential between
  the two bulk-like regions of the interface supercell. Positive means material B is at
  higher potential.
- **VBO > 0**: VBM of B is above VBM of A.
- **Type I alignment**: Both VBM and CBM offsets have the same sign. The smaller-gap material
  is "straddled" by the larger-gap material.
- **Type II alignment**: VBM and CBM offsets have opposite signs. Electrons and holes are
  separated into different materials.
- **Type III alignment**: Broken-gap configuration (rare).
- **PBE band gaps are underestimated**: Use HSE06 or GW band gaps for more accurate CBO.
  The VBO from the potential lineup is less sensitive to the gap error.

## Common Issues

| Problem | Solution |
|---|---|
| Double average still shows oscillations | The averaging window does not match the true period. Check interlayer spacing carefully |
| Potential does not flatten in bulk regions | Interface supercell is too thin. Increase to 10+ layers per material |
| Band offset depends on supercell size | Not converged; increase slab thickness until Delta V converges to < 0.01 eV |
| Lattice mismatch causes strain | Use the average in-plane lattice parameter or explicitly account for strain effects on band edges |
| average.x awin interpretation | awin is a fraction of the cell parameter (0 to 1), NOT in Angstrom. Convert: awin = d_angstrom / c_total_angstrom |
| Interface dipole effects | For polar interfaces, dipole correction may be needed. Check that the potential is symmetric in the bulk regions |
