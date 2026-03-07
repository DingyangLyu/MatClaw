# Planar-Average of Potential and Charge Density

## When to Use

- Compute the planar average of the electrostatic potential along a slab normal direction.
- Compute the planar average of the charge density along an interface direction.
- Generate 1D line profiles of potential or charge density along specified paths.
- Analyze potential profiles across heterojunctions, surfaces, or grain boundaries.
- Corresponds to VASPKIT tasks 422, 425--427.

## Method Selection

| Criterion | QE (pp.x + average.x) | VASP (LOCPOT/CHGCAR) | Python (cube/xsf) |
|---|---|---|---|
| Electrostatic potential | `plot_num=11` or `plot_num=1` | LOCPOT file | Parse cube file |
| Charge density | `plot_num=0` | CHGCAR file | Parse cube file |
| Spin density | `plot_num=6` (nspin=2) | CHGCAR spin channel | Parse cube file |
| Automation | average.x or Python | Python | Python |

## Prerequisites

- Quantum ESPRESSO 7.5 (`pw.x`, `pp.x`, `average.x`)
- Python packages: `numpy`, `scipy`, `matplotlib`, `ase`
- Converged SCF calculation (slab or bulk) with output files
- For VASP: LOCPOT and/or CHGCAR files

---

## Detailed Steps

### Method A: QE -- pp.x + average.x

#### Step A1: Extract Quantity with pp.x

```python
#!/usr/bin/env python3
"""
Extract electrostatic potential or charge density from QE using pp.x.
Outputs both a raw data file (for average.x) and a cube file (for Python).
"""
import os
import subprocess

OUTDIR = os.path.abspath("./tmp")
PREFIX = "slab"

# ── Choose what to extract ────────────────────────────────────────
# plot_num = 0  : total charge density
# plot_num = 1  : total potential (V_bare + V_H + V_xc)
# plot_num = 6  : spin polarization (rho_up - rho_down), requires nspin=2
# plot_num = 11 : V_bare + V_Hartree (electrostatic potential, no xc)

PLOT_NUM = 0  # Change as needed
LABEL = {0: "charge", 1: "total_potential", 6: "spin_density", 11: "elec_potential"}
label = LABEL.get(PLOT_NUM, f"plot{PLOT_NUM}")

# pp.x input: extract raw data for average.x
pp_raw_input = f"""&INPUTPP
    prefix  = '{PREFIX}'
    outdir  = '{OUTDIR}'
    filplot = '{label}_raw.dat'
    plot_num = {PLOT_NUM}
/
"""

with open(f"pp_{label}_raw.in", "w") as f:
    f.write(pp_raw_input)

print(f"Running pp.x to extract {label} (raw)...")
result = subprocess.run(
    ["pp.x", "-in", f"pp_{label}_raw.in"],
    capture_output=True, text=True, timeout=300
)
with open(f"pp_{label}_raw.out", "w") as f:
    f.write(result.stdout)

# pp.x input: also produce a cube file for Python visualization
pp_cube_input = f"""&INPUTPP
    prefix  = '{PREFIX}'
    outdir  = '{OUTDIR}'
    filplot = '{label}_raw.dat'
    plot_num = {PLOT_NUM}
/
&PLOT
    nfile       = 1
    filepp(1)   = '{label}_raw.dat'
    weight(1)   = 1.0
    iflag       = 3
    output_format = 6
    fileout     = '{label}.cube'
/
"""

with open(f"pp_{label}_cube.in", "w") as f:
    f.write(pp_cube_input)

print(f"Running pp.x to produce {label}.cube...")
result = subprocess.run(
    ["pp.x", "-in", f"pp_{label}_cube.in"],
    capture_output=True, text=True, timeout=300
)
with open(f"pp_{label}_cube.out", "w") as f:
    f.write(result.stdout)

if result.returncode == 0:
    print(f"Output files: {label}_raw.dat, {label}.cube")
else:
    print("ERROR in pp.x!")
    print(result.stderr[-500:] if result.stderr else "")
```

#### Step A2: Planar Average with average.x

```python
#!/usr/bin/env python3
"""
Run QE average.x to compute planar average along a chosen direction.

average.x reads the raw pp.x output file and computes:
  1) Planar average: V_avg(z) = (1/A) integral V(x,y,z) dx dy
  2) Optionally, macroscopic average: convolution with a window function.

Input to average.x (via stdin):
  nfile         - number of input files (usually 1)
  filename      - path to pp.x raw output
  weight        - weight for this file (1.0)
  npt           - number of points for the output
  idir          - direction (1=x, 2=y, 3=z)
  awin          - macroscopic averaging window in units of cell parameter
                  (set to 0.0 for planar average only)
"""
import subprocess
import numpy as np

LABEL = "charge"       # or "elec_potential", etc.
RAW_FILE = f"{LABEL}_raw.dat"
IDIR = 3               # averaging direction (3 = z for slabs)
NPT = 500              # number of output points
AWIN = 0.0             # macroscopic averaging window (0 = none)

avg_stdin = f"""1
{RAW_FILE}
1.0
{NPT}
{IDIR}
{AWIN}
"""

print(f"Running average.x (idir={IDIR}, awin={AWIN})...")
result = subprocess.run(
    ["average.x"],
    input=avg_stdin,
    capture_output=True, text=True, timeout=120
)
with open(f"average_{LABEL}.out", "w") as f:
    f.write(result.stdout)

if result.returncode == 0:
    print(f"average.x completed. Output: average_{LABEL}.out")
else:
    print("ERROR in average.x!")
    print(result.stderr[-500:] if result.stderr else "")

# ── Parse average.x output ────────────────────────────────────────
def parse_average_output(filename):
    """
    Parse the average.x stdout. It contains columns:
      z (bohr)    V_planar (Ry)    [V_macro (Ry)]
    """
    z_bohr = []
    v_planar = []
    v_macro = []

    with open(filename, "r") as f:
        for line in f:
            parts = line.strip().split()
            try:
                if len(parts) >= 2:
                    z_bohr.append(float(parts[0]))
                    v_planar.append(float(parts[1]))
                    if len(parts) >= 3:
                        v_macro.append(float(parts[2]))
            except ValueError:
                continue

    BOHR_TO_ANG = 0.529177
    RY_TO_EV = 13.6057

    z_ang = np.array(z_bohr) * BOHR_TO_ANG
    v_planar_ev = np.array(v_planar) * RY_TO_EV
    v_macro_ev = np.array(v_macro) * RY_TO_EV if v_macro else None

    return z_ang, v_planar_ev, v_macro_ev

z, v_plan, v_mac = parse_average_output(f"average_{LABEL}.out")
print(f"Parsed {len(z)} data points")
print(f"z range: {z[0]:.2f} -- {z[-1]:.2f} Angstrom")
```

#### Step A3: Python-Based Planar Average from Cube File

```python
#!/usr/bin/env python3
"""
Compute planar average and line profile directly from a cube file
(alternative to average.x, fully in Python).
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from ase.io.cube import read_cube_data

def planar_average(cube_file, axis=2, convert_ry_to_ev=True):
    """
    Compute planar average of a cube file quantity along the given axis.

    Parameters
    ----------
    cube_file : str
        Path to the Gaussian cube file.
    axis : int
        Direction along which to compute the planar average (0, 1, or 2).
    convert_ry_to_ev : bool
        If True, multiply values by 13.6057 (Ry -> eV conversion).

    Returns
    -------
    z : np.ndarray
        Positions along the averaging axis in Angstrom.
    v_avg : np.ndarray
        Planar-averaged values (eV if converted, else cube-file units).
    """
    data, atoms = read_cube_data(cube_file)
    cell = atoms.get_cell()
    length = np.linalg.norm(cell[axis])

    avg_axes = tuple(i for i in range(3) if i != axis)
    v_avg = np.mean(data, axis=avg_axes)

    n_pts = v_avg.shape[0]
    z = np.linspace(0, length, n_pts, endpoint=False)

    if convert_ry_to_ev:
        v_avg = v_avg * 13.6057

    return z, v_avg


def line_profile(cube_file, start_frac, end_frac, n_points=200, convert_ry_to_ev=True):
    """
    Extract a 1D line profile between two fractional coordinates.

    Parameters
    ----------
    cube_file : str
        Path to the Gaussian cube file.
    start_frac : array-like
        Starting point in fractional coordinates [a, b, c].
    end_frac : array-like
        Ending point in fractional coordinates [a, b, c].
    n_points : int
        Number of interpolation points.
    convert_ry_to_ev : bool
        Convert Ry to eV.

    Returns
    -------
    dist : np.ndarray
        Distance along the line in Angstrom.
    values : np.ndarray
        Interpolated values along the line.
    """
    from scipy.ndimage import map_coordinates

    data, atoms = read_cube_data(cube_file)
    cell = atoms.get_cell()
    nx, ny, nz = data.shape

    start = np.array(start_frac)
    end = np.array(end_frac)

    # Parametric line in fractional coordinates
    t = np.linspace(0, 1, n_points)
    frac_points = start[None, :] + t[:, None] * (end - start)[None, :]

    # Convert fractional to grid indices
    grid_points = frac_points * np.array([nx, ny, nz])[None, :]

    # Interpolate
    values = map_coordinates(data, grid_points.T, order=3, mode='wrap')

    if convert_ry_to_ev:
        values = values * 13.6057

    # Compute real-space distance
    cart_start = cell.T @ start
    cart_end = cell.T @ end
    total_dist = np.linalg.norm(cart_end - cart_start)
    dist = t * total_dist

    return dist, values


# ── Example: planar average of electrostatic potential ────────────
z, v_plan = planar_average("elec_potential.cube", axis=2, convert_ry_to_ev=True)

fig, ax = plt.subplots(figsize=(10, 5))
ax.plot(z, v_plan, "b-", linewidth=1.2)
ax.set_xlabel(r"z ($\mathrm{\AA}$)", fontsize=13)
ax.set_ylabel("Potential (eV)", fontsize=13)
ax.set_title("Planar-Averaged Electrostatic Potential", fontsize=14)
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("planar_avg_potential.png", dpi=200, bbox_inches="tight")
print("Saved: planar_avg_potential.png")

# ── Example: planar average of charge density ─────────────────────
z_rho, rho_plan = planar_average("charge.cube", axis=2, convert_ry_to_ev=False)

fig, ax = plt.subplots(figsize=(10, 5))
ax.plot(z_rho, rho_plan, "r-", linewidth=1.2)
ax.set_xlabel(r"z ($\mathrm{\AA}$)", fontsize=13)
ax.set_ylabel(r"Charge density (e/bohr$^3$)", fontsize=13)
ax.set_title("Planar-Averaged Charge Density", fontsize=14)
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("planar_avg_charge.png", dpi=200, bbox_inches="tight")
print("Saved: planar_avg_charge.png")

# ── Example: line profile along [111] direction ──────────────────
dist, vals = line_profile(
    "elec_potential.cube",
    start_frac=[0.0, 0.0, 0.0],
    end_frac=[0.5, 0.5, 0.5],
    n_points=300,
    convert_ry_to_ev=True,
)

fig, ax = plt.subplots(figsize=(8, 4))
ax.plot(dist, vals, "g-", linewidth=1.2)
ax.set_xlabel(r"Distance ($\mathrm{\AA}$)", fontsize=13)
ax.set_ylabel("Potential (eV)", fontsize=13)
ax.set_title("Potential Line Profile along [111]", fontsize=14)
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("line_profile_111.png", dpi=200, bbox_inches="tight")
print("Saved: line_profile_111.png")
```

### Method B: VASP -- LOCPOT / CHGCAR Parsing

```python
#!/usr/bin/env python3
"""
Compute planar average and line profiles from VASP LOCPOT or CHGCAR files.
"""
import numpy as np
from scipy.ndimage import map_coordinates
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def parse_vasp_volumetric(filename):
    """
    Parse a VASP volumetric data file (LOCPOT, CHGCAR, etc.).

    Returns
    -------
    cell : np.ndarray (3, 3)
        Cell vectors in Angstrom.
    data : np.ndarray (nx, ny, nz)
        Volumetric data.
    atoms_frac : np.ndarray (n_atoms, 3)
        Fractional coordinates of atoms.
    species : list of str
        Atomic species labels.
    """
    with open(filename, "r") as f:
        lines = f.readlines()

    scale = float(lines[1].strip())
    cell = np.zeros((3, 3))
    for i in range(3):
        cell[i] = [float(x) for x in lines[2 + i].split()]
    cell *= scale

    species = lines[5].split()
    counts = [int(x) for x in lines[6].split()]
    n_atoms = sum(counts)

    # Determine coordinate type
    coord_line = lines[7].strip()
    if coord_line[0] in ("S", "s"):
        # Selective dynamics line; skip it
        coord_start = 9
    else:
        coord_start = 8

    atoms_frac = np.zeros((n_atoms, 3))
    for i in range(n_atoms):
        parts = lines[coord_start + i].split()
        atoms_frac[i] = [float(parts[j]) for j in range(3)]

    # Find the grid dimensions line
    data_start = coord_start + n_atoms
    while data_start < len(lines) and lines[data_start].strip() == "":
        data_start += 1

    grid_dims = [int(x) for x in lines[data_start].split()]
    nx, ny, nz = grid_dims
    data_start += 1

    values = []
    for i in range(data_start, len(lines)):
        line = lines[i].strip()
        if line == "":
            if len(values) >= nx * ny * nz:
                break
            continue
        try:
            values.extend([float(x) for x in line.split()])
        except ValueError:
            break
        if len(values) >= nx * ny * nz:
            break

    data = np.array(values[:nx * ny * nz]).reshape((nx, ny, nz), order='F')

    # CHGCAR stores rho * V_cell; LOCPOT stores V directly
    # For CHGCAR, divide by volume to get density per bohr^3
    # For LOCPOT, data is already in eV

    return cell, data, atoms_frac, species


def planar_average_vasp(filename, axis=2, is_chgcar=False):
    """
    Compute planar average of VASP volumetric data along specified axis.

    Parameters
    ----------
    filename : str
        Path to LOCPOT or CHGCAR.
    axis : int
        Averaging direction (0, 1, or 2).
    is_chgcar : bool
        If True, divide by cell volume (CHGCAR stores rho*V).

    Returns
    -------
    z : np.ndarray
        Position along axis in Angstrom.
    avg : np.ndarray
        Planar-averaged values.
    """
    cell, data, _, _ = parse_vasp_volumetric(filename)

    if is_chgcar:
        vol = np.abs(np.dot(cell[0], np.cross(cell[1], cell[2])))
        nx, ny, nz = data.shape
        data = data / vol  # now in e/Angstrom^3

    avg_axes = tuple(i for i in range(3) if i != axis)
    avg = np.mean(data, axis=avg_axes)

    length = np.linalg.norm(cell[axis])
    z = np.linspace(0, length, len(avg), endpoint=False)

    return z, avg


def line_profile_vasp(filename, start_frac, end_frac, n_points=200, is_chgcar=False):
    """
    Extract a 1D line profile from a VASP volumetric file.
    """
    cell, data, _, _ = parse_vasp_volumetric(filename)

    if is_chgcar:
        vol = np.abs(np.dot(cell[0], np.cross(cell[1], cell[2])))
        data = data / vol

    nx, ny, nz = data.shape
    start = np.array(start_frac)
    end = np.array(end_frac)

    t = np.linspace(0, 1, n_points)
    frac_pts = start[None, :] + t[:, None] * (end - start)[None, :]

    grid_pts = frac_pts * np.array([nx, ny, nz])[None, :]
    values = map_coordinates(data, grid_pts.T, order=3, mode='wrap')

    cart_start = cell.T @ start
    cart_end = cell.T @ end
    total_dist = np.linalg.norm(cart_end - cart_start)
    dist = t * total_dist

    return dist, values


# ── Example usage ─────────────────────────────────────────────────

# Planar average of LOCPOT
# z, v_avg = planar_average_vasp("LOCPOT", axis=2, is_chgcar=False)

# Planar average of CHGCAR
# z, rho_avg = planar_average_vasp("CHGCAR", axis=2, is_chgcar=True)

# Line profile
# dist, vals = line_profile_vasp("LOCPOT", [0,0,0], [0,0,1], n_points=300)

# ── Demo with synthetic data ─────────────────────────────────────
z_demo = np.linspace(0, 30, 500)
# Simulate a slab potential profile
v_demo = 5.0 * np.ones_like(z_demo)
slab_center = 15.0
slab_half = 5.0
v_demo[np.abs(z_demo - slab_center) < slab_half] = -10.0 + 5.0 * np.sin(
    2 * np.pi * (z_demo[np.abs(z_demo - slab_center) < slab_half] - 10.0) / 2.5
)

fig, ax = plt.subplots(figsize=(10, 5))
ax.plot(z_demo, v_demo, "b-", linewidth=1.2, label="Planar average")
ax.set_xlabel(r"z ($\mathrm{\AA}$)", fontsize=13)
ax.set_ylabel("Potential (eV)", fontsize=13)
ax.set_title("Planar-Averaged Potential (demo)", fontsize=14)
ax.legend(fontsize=11)
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("planar_avg_demo.png", dpi=200, bbox_inches="tight")
print("Saved: planar_avg_demo.png")
```

### Method C: Multi-Panel Plot (Potential + Charge + Atoms)

```python
#!/usr/bin/env python3
"""
Create a publication-quality multi-panel figure showing:
  Top:    Atomic structure (z positions)
  Middle: Planar-averaged charge density
  Bottom: Planar-averaged electrostatic potential
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from ase.io.cube import read_cube_data
from ase.io import read


def multi_panel_planar_average(charge_cube, potential_cube, structure_file=None,
                                axis=2, output_png="planar_avg_multi.png"):
    """
    Multi-panel plot of planar-averaged charge and potential.
    """
    # Parse cube files
    rho_data, rho_atoms = read_cube_data(charge_cube)
    v_data, v_atoms = read_cube_data(potential_cube)

    cell = rho_atoms.get_cell()
    c_length = np.linalg.norm(cell[axis])

    # Planar averages
    avg_ax = tuple(i for i in range(3) if i != axis)
    rho_avg = np.mean(rho_data, axis=avg_ax)
    v_avg = np.mean(v_data, axis=avg_ax) * 13.6057  # Ry -> eV

    n_rho = len(rho_avg)
    n_v = len(v_avg)
    z_rho = np.linspace(0, c_length, n_rho, endpoint=False)
    z_v = np.linspace(0, c_length, n_v, endpoint=False)

    # Atom positions along z
    if structure_file:
        atoms = read(structure_file)
    else:
        atoms = rho_atoms

    z_atoms = atoms.positions[:, axis]
    symbols = atoms.get_chemical_symbols()

    # Unique colors per element
    unique_elements = sorted(set(symbols))
    color_map = {}
    cmap_colors = plt.cm.Set1(np.linspace(0, 1, max(len(unique_elements), 3)))
    for i, elem in enumerate(unique_elements):
        color_map[elem] = cmap_colors[i]

    # ── Plot ──────────────────────────────────────────────────────
    fig, axes = plt.subplots(3, 1, figsize=(10, 10), sharex=True,
                              gridspec_kw={"height_ratios": [1, 2, 2]})

    # Panel 1: Atomic positions
    ax0 = axes[0]
    for za, sym in zip(z_atoms, symbols):
        ax0.axvline(za, color=color_map[sym], linewidth=1.5, alpha=0.7)
    # Legend
    for elem in unique_elements:
        ax0.axvline(-999, color=color_map[elem], linewidth=2, label=elem)
    ax0.set_ylabel("Atoms")
    ax0.set_yticks([])
    ax0.legend(loc="upper right", fontsize=10)
    ax0.set_title("Planar-Averaged Profiles Along Slab Normal", fontsize=14)

    # Panel 2: Charge density
    ax1 = axes[1]
    ax1.plot(z_rho, rho_avg, "r-", linewidth=1.2)
    ax1.set_ylabel(r"$\rho$ (e/bohr$^3$)", fontsize=12)
    ax1.grid(True, alpha=0.3)

    # Panel 3: Potential
    ax2 = axes[2]
    ax2.plot(z_v, v_avg, "b-", linewidth=1.2)
    ax2.set_ylabel("$V$ (eV)", fontsize=12)
    ax2.set_xlabel(r"z ($\mathrm{\AA}$)", fontsize=13)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_png, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"Saved: {output_png}")


# Usage:
# multi_panel_planar_average("charge.cube", "elec_potential.cube")
```

---

## Key Parameters

| Parameter | Typical Value | Notes |
|---|---|---|
| `plot_num` (pp.x) | 0 (charge), 11 (V_es), 1 (V_tot), 6 (spin) | Choose based on the quantity of interest |
| `iflag` (pp.x) | 3 (3D cube), 1 (1D line) | Use 3 for cube files, 1 for direct line profiles |
| `idir` (average.x) | 3 (z-axis for slabs) | Direction perpendicular to the surface/interface |
| `npt` (average.x) | 200--1000 | Number of output points; higher for smoother curves |
| `awin` (average.x) | 0.0 (planar only) | Set nonzero for macroscopic averaging (see macroscopic-average skill) |

## Interpreting Results

- **Electrostatic potential**: Should oscillate within the slab (peaks at nuclei) and flatten in vacuum.
  The flat vacuum region gives E_vac for work function calculations.
- **Charge density**: Peaks at atomic positions. In a metal slab, Friedel oscillations may appear
  near the surface. In vacuum, the charge density should decay to zero.
- **Spin density**: Positive/negative peaks at magnetic atom positions show spin-up/spin-down
  character. Should be zero in vacuum and at non-magnetic sites.
- **Line profiles**: Useful for examining bonding character along specific directions
  (e.g., along a bond, through an interface).

## Common Issues

| Problem | Solution |
|---|---|
| average.x hangs or gives empty output | Check input format: each field on a separate line. Ensure raw pp.x file exists |
| Cube file has wrong units | QE cube files use atomic units (bohr, Ry). Multiply potential by 13.6057 for eV |
| CHGCAR values seem huge | CHGCAR stores rho * V_cell (integrated charge). Divide by cell volume for density |
| Line profile shows discontinuities | Increase `n_points`. Ensure `mode='wrap'` for periodic interpolation |
| pp.x output is empty/zero | Verify SCF completed and `prefix`/`outdir` match the SCF calculation |
| Planar average axis is wrong | For slabs, use the axis perpendicular to the surface (usually z=2) |
