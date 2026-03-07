# Charge Density Analysis

Compute and visualize charge density from Quantum ESPRESSO using pp.x and Python.

## When to Use

- Visualize spatial distribution of electrons in a material.
- Compute charge difference (bonded system minus isolated atoms) to see bonding charge redistribution.
- Compute deformation density (self-consistent charge minus superposition of atomic charges).
- Generate 2D slice plots or 1D line profiles through bonds.
- Quantify charge transfer between atoms or layers.

## Prerequisites

- Quantum ESPRESSO 7.5 (`pw.x`, `pp.x`)
- Python: `numpy`, `matplotlib`, `ase` (for reading cube files)
- Pseudopotential files for your elements
- Converged SCF calculation

## Detailed Steps

### Step 1: SCF Calculation

Run a standard self-consistent calculation with `pw.x`. Save the charge density to disk.

**File: `scf.in`**

```
&CONTROL
    calculation  = 'scf'
    prefix       = 'silicon'
    outdir       = './tmp'
    pseudo_dir   = './pseudo'
    tprnfor      = .true.
    tstress      = .true.
/
&SYSTEM
    ibrav        = 2
    celldm(1)    = 10.26
    nat          = 2
    ntyp         = 1
    ecutwfc      = 60.0
    ecutrho      = 480.0
/
&ELECTRONS
    conv_thr     = 1.0d-8
    mixing_beta  = 0.7
/
ATOMIC_SPECIES
  Si  28.0855  Si.pbe-n-rrkjus_psl.1.0.0.UPF

ATOMIC_POSITIONS {crystal}
  Si  0.00  0.00  0.00
  Si  0.25  0.25  0.25

K_POINTS {automatic}
  8 8 8  0 0 0
```

**Run:**
```bash
pw.x < scf.in > scf.out
```

### Step 2: Extract Charge Density with pp.x

#### 2a: Total Charge Density (3D Cube File)

**File: `pp_charge_cube.in`**

```
&INPUTPP
    prefix  = 'silicon'
    outdir  = './tmp'
    filplot = 'charge_total.dat'
    plot_num = 0
/
&PLOT
    nfile       = 1
    filepp(1)   = 'charge_total.dat'
    weight(1)   = 1.0
    iflag       = 3
    output_format = 6
    fileout     = 'charge_total.cube'
/
```

**Run:**
```bash
pp.x < pp_charge_cube.in > pp_charge_cube.out
```

#### 2b: Total Charge Density (2D Slice)

To produce a 2D planar average or slice through a specific plane:

**File: `pp_charge_2d.in`**

```
&INPUTPP
    prefix  = 'silicon'
    outdir  = './tmp'
    filplot = 'charge_total.dat'
    plot_num = 0
/
&PLOT
    nfile       = 1
    filepp(1)   = 'charge_total.dat'
    weight(1)   = 1.0
    iflag       = 2
    output_format = 7
    fileout     = 'charge_2d.dat'
    e1(1)       = 1.0, e1(2) = 0.0, e1(3) = 0.0
    e2(1)       = 0.0, e2(2) = 1.0, e2(3) = 0.0
    x0(1)       = 0.0, x0(2) = 0.0, x0(3) = 0.0
    nx           = 100
    ny           = 100
/
```

`iflag=2` produces a 2D contour plot on a plane defined by origin `x0` and spanning vectors `e1`, `e2`.

**Run:**
```bash
pp.x < pp_charge_2d.in > pp_charge_2d.out
```

#### 2c: 1D Line Profile

**File: `pp_charge_1d.in`**

```
&INPUTPP
    prefix  = 'silicon'
    outdir  = './tmp'
    filplot = 'charge_total.dat'
    plot_num = 0
/
&PLOT
    nfile       = 1
    filepp(1)   = 'charge_total.dat'
    weight(1)   = 1.0
    iflag       = 1
    output_format = 0
    fileout     = 'charge_1d.dat'
    e1(1)       = 1.0, e1(2) = 1.0, e1(3) = 1.0
    x0(1)       = 0.0, x0(2) = 0.0, x0(3) = 0.0
    nx           = 200
/
```

`iflag=1` gives a 1D cut along direction `e1` starting at `x0`.

**Run:**
```bash
pp.x < pp_charge_1d.in > pp_charge_1d.out
```

#### 2d: Charge Difference / Deformation Density

For charge difference you need multiple pp.x runs: one for the full system and one for each isolated atom/fragment, then subtract using the `weight` mechanism or Python.

**Method A -- Using pp.x weight subtraction:**

First, extract charge for the full system and each fragment (isolated atom SCF in same cell). Then combine:

**File: `pp_diff.in`**

```
&INPUTPP
/
&PLOT
    nfile       = 3
    filepp(1)   = 'charge_full.dat'
    weight(1)   = 1.0
    filepp(2)   = 'charge_atom1.dat'
    weight(2)   = -1.0
    filepp(3)   = 'charge_atom2.dat'
    weight(3)   = -1.0
    iflag       = 3
    output_format = 6
    fileout     = 'charge_diff.cube'
/
```

**Method B -- Python subtraction (more flexible):**

See Python code below.

### Step 3: Python Visualization

#### 3a: Read and Plot a Cube File (2D Slice)

```python
#!/usr/bin/env python3
"""
Read a Gaussian cube file from pp.x and plot a 2D charge density slice.
"""
import numpy as np
import matplotlib.pyplot as plt
from ase.io.cube import read_cube_data


def plot_cube_slice(cube_file, axis=2, slice_index=None, output_png='charge_slice.png'):
    """
    Plot a 2D slice of charge density from a cube file.

    Parameters
    ----------
    cube_file : str
        Path to the .cube file produced by pp.x (output_format=6).
    axis : int
        Axis perpendicular to the slice plane (0=x, 1=y, 2=z).
    slice_index : int or None
        Grid index along `axis`. If None, uses the midpoint.
    output_png : str
        Output image filename.
    """
    data, atoms = read_cube_data(cube_file)
    # data shape: (nx, ny, nz), values in e/bohr^3

    if slice_index is None:
        slice_index = data.shape[axis] // 2

    if axis == 0:
        slice_2d = data[slice_index, :, :]
    elif axis == 1:
        slice_2d = data[:, slice_index, :]
    else:
        slice_2d = data[:, :, slice_index]

    cell = atoms.get_cell()
    # Determine the two in-plane axes
    axes = [i for i in range(3) if i != axis]
    extent = [0, np.linalg.norm(cell[axes[0]]), 0, np.linalg.norm(cell[axes[1]])]

    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.contourf(
        np.linspace(extent[0], extent[1], slice_2d.shape[0]),
        np.linspace(extent[2], extent[3], slice_2d.shape[1]),
        slice_2d.T,
        levels=50,
        cmap='RdYlBu_r'
    )
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label(r'Charge density (e/bohr$^3$)')
    ax.set_xlabel(r'Distance ($\mathrm{\AA}$)')
    ax.set_ylabel(r'Distance ($\mathrm{\AA}$)')
    ax.set_title(f'Charge density slice at axis={axis}, index={slice_index}')
    ax.set_aspect('equal')
    plt.tight_layout()
    plt.savefig(output_png, dpi=200)
    plt.close()
    print(f"Saved: {output_png}")


# ----- Run -----
plot_cube_slice('charge_total.cube', axis=2, slice_index=None)
```

#### 3b: Charge Difference in Python

```python
#!/usr/bin/env python3
"""
Compute and plot charge difference: rho_full - rho_atom1 - rho_atom2.
All cube files must be on the same grid (same cell, same FFT grid).
"""
import numpy as np
import matplotlib.pyplot as plt
from ase.io.cube import read_cube_data, write_cube


def charge_difference(cube_full, cube_fragments, output_cube='charge_diff.cube',
                      output_png='charge_diff.png', axis=2, slice_index=None):
    """
    Compute charge difference and save as cube + 2D slice image.

    Parameters
    ----------
    cube_full : str
        Cube file for the full (bonded) system.
    cube_fragments : list of str
        List of cube files for isolated fragments (atoms).
    output_cube : str
        Output cube file with the difference density.
    output_png : str
        Output PNG image.
    axis : int
        Slice axis.
    slice_index : int or None
        Slice grid index.
    """
    data_full, atoms_full = read_cube_data(cube_full)
    diff = data_full.copy()

    for frag_cube in cube_fragments:
        data_frag, _ = read_cube_data(frag_cube)
        diff -= data_frag

    # Write difference cube file
    from ase.io.cube import write_cube as _wc
    with open(output_cube, 'w') as f:
        _wc(f, atoms_full, diff)
    print(f"Wrote difference cube: {output_cube}")

    # Plot 2D slice
    if slice_index is None:
        slice_index = diff.shape[axis] // 2

    if axis == 0:
        sl = diff[slice_index, :, :]
    elif axis == 1:
        sl = diff[:, slice_index, :]
    else:
        sl = diff[:, :, slice_index]

    vmax = np.max(np.abs(sl)) * 0.8  # symmetric color scale

    fig, ax = plt.subplots(figsize=(8, 6))
    cell = atoms_full.get_cell()
    axes_2d = [i for i in range(3) if i != axis]
    ex = [0, np.linalg.norm(cell[axes_2d[0]]), 0, np.linalg.norm(cell[axes_2d[1]])]

    im = ax.contourf(
        np.linspace(ex[0], ex[1], sl.shape[0]),
        np.linspace(ex[2], ex[3], sl.shape[1]),
        sl.T,
        levels=np.linspace(-vmax, vmax, 51),
        cmap='bwr'
    )
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label(r'$\Delta\rho$ (e/bohr$^3$)')
    ax.set_xlabel(r'Distance ($\mathrm{\AA}$)')
    ax.set_ylabel(r'Distance ($\mathrm{\AA}$)')
    ax.set_title('Charge density difference')
    ax.set_aspect('equal')
    plt.tight_layout()
    plt.savefig(output_png, dpi=200)
    plt.close()
    print(f"Saved: {output_png}")


# ----- Run -----
charge_difference(
    cube_full='charge_full.cube',
    cube_fragments=['charge_atom1.cube', 'charge_atom2.cube']
)
```

#### 3c: Plot pp.x 1D Line Profile Output

```python
#!/usr/bin/env python3
"""
Plot 1D charge density profile from pp.x (iflag=1, output_format=0).
The file format is two columns: distance (bohr), charge density (e/bohr^3).
"""
import numpy as np
import matplotlib.pyplot as plt

BOHR_TO_ANG = 0.529177

def plot_1d_profile(data_file, output_png='charge_1d.png'):
    """
    Plot 1D charge density along a line.
    """
    data = np.loadtxt(data_file)
    dist_ang = data[:, 0] * BOHR_TO_ANG
    rho = data[:, 1]

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(dist_ang, rho, 'b-', linewidth=1.5)
    ax.set_xlabel(r'Distance ($\mathrm{\AA}$)')
    ax.set_ylabel(r'Charge density (e/bohr$^3$)')
    ax.set_title('1D Charge Density Profile')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_png, dpi=200)
    plt.close()
    print(f"Saved: {output_png}")


plot_1d_profile('charge_1d.dat')
```

#### 3d: Plot pp.x 2D Data (output_format=7, Gnuplot Format)

```python
#!/usr/bin/env python3
"""
Plot 2D charge density from pp.x output_format=7 (gnuplot format).
The file has blocks separated by blank lines: x y rho
"""
import numpy as np
import matplotlib.pyplot as plt

BOHR_TO_ANG = 0.529177


def plot_2d_gnuplot(data_file, output_png='charge_2d.png'):
    """
    Read pp.x 2D output (format 7) and create a contour plot.
    """
    # Read data, skipping comment lines
    raw = []
    with open(data_file, 'r') as f:
        for line in f:
            line = line.strip()
            if line.startswith('#') or len(line) == 0:
                continue
            parts = line.split()
            if len(parts) >= 3:
                raw.append([float(parts[0]), float(parts[1]), float(parts[2])])

    raw = np.array(raw)
    x = raw[:, 0] * BOHR_TO_ANG
    y = raw[:, 1] * BOHR_TO_ANG
    z = raw[:, 2]

    # Reconstruct 2D grid
    x_unique = np.unique(x)
    y_unique = np.unique(y)
    nx, ny = len(x_unique), len(y_unique)
    Z = z.reshape(nx, ny)

    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.contourf(x_unique, y_unique, Z.T, levels=50, cmap='RdYlBu_r')
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label(r'Charge density (e/bohr$^3$)')
    ax.set_xlabel(r'Distance ($\mathrm{\AA}$)')
    ax.set_ylabel(r'Distance ($\mathrm{\AA}$)')
    ax.set_title('2D Charge Density')
    ax.set_aspect('equal')
    plt.tight_layout()
    plt.savefig(output_png, dpi=200)
    plt.close()
    print(f"Saved: {output_png}")


plot_2d_gnuplot('charge_2d.dat')
```

#### 3e: Layer-Resolved Charge Transfer (Planar Average)

```python
#!/usr/bin/env python3
"""
Compute planar-averaged charge density along z and integrate
to find charge transfer between layers.
Works with cube files.
"""
import numpy as np
import matplotlib.pyplot as plt
from ase.io.cube import read_cube_data


def planar_average_and_transfer(cube_file, axis=2, output_png='planar_avg.png'):
    """
    Compute planar average of charge density along a given axis
    and estimate charge transfer between top/bottom halves.

    Parameters
    ----------
    cube_file : str
        Cube file from pp.x.
    axis : int
        Axis along which to average (default z=2).
    output_png : str
        Output image.
    """
    data, atoms = read_cube_data(cube_file)
    cell = atoms.get_cell()

    # Average over the two in-plane axes
    avg_axes = tuple(i for i in range(3) if i != axis)
    planar_avg = np.mean(data, axis=avg_axes)

    # Distance along the chosen axis in Angstrom
    n_pts = planar_avg.shape[0]
    length = np.linalg.norm(cell[axis])
    z = np.linspace(0, length, n_pts, endpoint=False)

    # Integrate: total electrons = sum(rho) * dV
    # dV = cell_volume / (nx * ny * nz)
    cell_vol = atoms.get_volume()  # Angstrom^3
    total_grid = data.shape[0] * data.shape[1] * data.shape[2]
    dV = cell_vol / total_grid
    total_electrons = np.sum(data) * dV
    print(f"Total electrons in cell: {total_electrons:.4f}")

    # Split at midpoint for layer analysis
    mid = n_pts // 2
    dz = length / n_pts
    n_inplane = data.shape[avg_axes[0]] * data.shape[avg_axes[1]]
    # Charge in bottom half
    q_bottom = np.sum(planar_avg[:mid]) * dz * n_inplane * dV / (length / n_pts * n_inplane)
    # More accurately: integrate planar_avg * area_element
    # planar_avg is mean over in-plane; multiply by in-plane area
    in_plane_area = cell_vol / length
    q_bottom_acc = np.sum(planar_avg[:mid]) * dz * in_plane_area
    q_top_acc = np.sum(planar_avg[mid:]) * dz * in_plane_area
    print(f"Charge bottom half: {q_bottom_acc:.4f} e")
    print(f"Charge top half:    {q_top_acc:.4f} e")
    print(f"Charge transfer (top - bottom): {q_top_acc - q_bottom_acc:.4f} e")

    # Plot
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(z, planar_avg, 'b-', linewidth=1.5)
    ax.axvline(length / 2, color='gray', linestyle='--', alpha=0.5, label='Midpoint')
    ax.set_xlabel(r'z ($\mathrm{\AA}$)')
    ax.set_ylabel(r'Planar average $\rho$ (e/bohr$^3$)')
    ax.set_title('Planar-Averaged Charge Density')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_png, dpi=200)
    plt.close()
    print(f"Saved: {output_png}")


planar_average_and_transfer('charge_total.cube', axis=2)
```

## Key Parameters

### pp.x `plot_num` Values

| `plot_num` | Quantity |
|------------|----------|
| 0 | Total charge density |
| 1 | Total potential (V_bare + V_H + V_xc) |
| 2 | Local ionic potential (V_bare) |
| 6 | Spin polarization (rho_up - rho_down) |
| 8 | Electron Localization Function (ELF) |
| 9 | Reduced density gradient |
| 10 | Integrated local density of states up to E_Fermi |
| 17 | All-electron charge density (PAW only) |

### pp.x `iflag` Values

| `iflag` | Description |
|---------|-------------|
| 0 | 1D spherical average |
| 1 | 1D along a line |
| 2 | 2D on a plane |
| 3 | 3D grid |
| 4 | 2D polar on a plane |

### pp.x `output_format` Values

| `output_format` | Format |
|------------------|--------|
| 0 | pp.x internal (1D gnuplot) |
| 2 | Plotrho format (obsolete) |
| 3 | XCrySDen (xsf) |
| 5 | XCrySDen (xsf, 3D) |
| 6 | Gaussian cube |
| 7 | Gnuplot 2D format |

### SCF Parameters That Affect Charge Density Quality

| Parameter | Guidance |
|-----------|----------|
| `ecutwfc` | Must be well converged; test convergence |
| `ecutrho` | For USPP/PAW: 8-12x ecutwfc; for NC: 4x ecutwfc |
| `conv_thr` | Use 1.0d-8 or tighter for accurate charge |
| K-points | Dense enough for metallic systems |

## Interpreting Results

- **Total charge density**: Peaks at atomic positions. Shared electron density between atoms indicates bonding.
- **Charge difference (delta-rho)**: Positive regions (red in `bwr` colormap) = charge accumulation upon bonding. Negative regions (blue) = charge depletion. Accumulation between atoms signals covalent bonding.
- **Deformation density**: Similar to charge difference but uses superposed atomic densities as reference. Shows how the electron density rearranges from spherical atomic distributions.
- **Planar average**: Useful for heterostructures and surfaces. Charge transfer across an interface appears as asymmetric integrated charge.

## Common Issues

1. **Cube files have wrong grid**: Ensure the SCF uses the same cell for the bonded system and each isolated fragment. Use the same `ecutwfc`/`ecutrho` so the FFT grids match.
2. **Units confusion**: pp.x outputs charge in e/bohr^3 by default. ASE `read_cube_data` returns the values as-is from the cube file (typically e/bohr^3 for QE).
3. **2D plot plane misoriented**: The vectors `e1` and `e2` in pp.x define the plotting plane in crystal (alat) coordinates. Check they span the desired plane.
4. **Negative charge density**: For charge difference plots, negative values are expected (charge depletion). For total charge, negative values indicate an error.
5. **Memory for large grids**: For `iflag=3` (3D output), cube files can be large. Use `iflag=2` for 2D slices if only a cross-section is needed.
6. **pp.x crash with `No output file`**: Ensure `filplot` is set in `&INPUTPP` and the corresponding file exists when referenced in `&PLOT`.
