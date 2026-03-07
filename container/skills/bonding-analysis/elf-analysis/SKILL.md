# Electron Localization Function (ELF) Analysis

Compute and visualize the Electron Localization Function using Quantum ESPRESSO pp.x to characterize bond types.

## When to Use

- Determine whether a bond is covalent, ionic, or metallic.
- Visualize lone pairs and bonding electron pairs.
- Compare bonding character across different materials or polymorphs.
- Identify regions of electron localization vs delocalization.

## Prerequisites

- Quantum ESPRESSO 7.5 (`pw.x`, `pp.x`)
- Python: `numpy`, `matplotlib`, `ase`
- Converged SCF calculation

## Detailed Steps

### Step 1: SCF Calculation

Run a well-converged SCF calculation. The ELF is derived from the Kohn-Sham orbitals, so the calculation must be fully converged.

**File: `scf.in`** (example: NaCl -- ionic bonding)

```
&CONTROL
    calculation  = 'scf'
    prefix       = 'nacl'
    outdir       = './tmp'
    pseudo_dir   = './pseudo'
    tprnfor      = .true.
/
&SYSTEM
    ibrav        = 2
    celldm(1)    = 10.68
    nat          = 2
    ntyp         = 2
    ecutwfc      = 60.0
    ecutrho      = 480.0
/
&ELECTRONS
    conv_thr     = 1.0d-8
    mixing_beta  = 0.7
/
ATOMIC_SPECIES
  Na  22.9898  Na.pbe-spnl-rrkjus_psl.1.0.0.UPF
  Cl  35.4530  Cl.pbe-nl-rrkjus_psl.1.0.0.UPF

ATOMIC_POSITIONS {crystal}
  Na  0.00  0.00  0.00
  Cl  0.50  0.50  0.50

K_POINTS {automatic}
  8 8 8  0 0 0
```

**Run:**
```bash
pw.x < scf.in > scf.out
```

### Step 2: Compute ELF with pp.x

The ELF is extracted using `plot_num = 8`.

#### 2a: ELF as 3D Cube File

**File: `pp_elf_cube.in`**

```
&INPUTPP
    prefix   = 'nacl'
    outdir   = './tmp'
    filplot  = 'elf.dat'
    plot_num = 8
/
&PLOT
    nfile       = 1
    filepp(1)   = 'elf.dat'
    weight(1)   = 1.0
    iflag       = 3
    output_format = 6
    fileout     = 'elf.cube'
/
```

**Run:**
```bash
pp.x < pp_elf_cube.in > pp_elf_cube.out
```

#### 2b: ELF as 2D Slice (Direct)

To get a 2D slice directly from pp.x (avoids large cube files):

**File: `pp_elf_2d.in`**

```
&INPUTPP
    prefix   = 'nacl'
    outdir   = './tmp'
    filplot  = 'elf.dat'
    plot_num = 8
/
&PLOT
    nfile       = 1
    filepp(1)   = 'elf.dat'
    weight(1)   = 1.0
    iflag       = 2
    output_format = 7
    fileout     = 'elf_2d.dat'
    e1(1) = 1.0, e1(2) = 0.0, e1(3) = 0.0
    e2(1) = 0.0, e2(2) = 1.0, e2(3) = 0.0
    x0(1) = 0.0, x0(2) = 0.0, x0(3) = 0.0
    nx    = 150
    ny    = 150
/
```

The plane is defined by:
- `x0`: origin point (in crystal/alat coordinates)
- `e1`, `e2`: two vectors spanning the plane
- `nx`, `ny`: number of grid points along each direction

**Run:**
```bash
pp.x < pp_elf_2d.in > pp_elf_2d.out
```

#### 2c: ELF Along a Bond (1D Profile)

**File: `pp_elf_1d.in`**

```
&INPUTPP
    prefix   = 'nacl'
    outdir   = './tmp'
    filplot  = 'elf.dat'
    plot_num = 8
/
&PLOT
    nfile       = 1
    filepp(1)   = 'elf.dat'
    weight(1)   = 1.0
    iflag       = 1
    output_format = 0
    fileout     = 'elf_1d.dat'
    e1(1) = 1.0, e1(2) = 1.0, e1(3) = 1.0
    x0(1) = 0.0, x0(2) = 0.0, x0(3) = 0.0
    nx    = 200
/
```

This extracts the ELF along the [111] direction (Na-Cl bond direction in rock salt).

**Run:**
```bash
pp.x < pp_elf_1d.in > pp_elf_1d.out
```

### Step 3: Python Visualization

#### 3a: 2D ELF Slice from Cube File

```python
#!/usr/bin/env python3
"""
Read ELF cube file and plot 2D slices with appropriate contour levels
for bond character analysis.
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from ase.io.cube import read_cube_data


def plot_elf_slice(cube_file, axis=2, slice_index=None, output_png='elf_slice.png',
                   show_atoms=True):
    """
    Plot a 2D slice of the ELF from a cube file.

    Parameters
    ----------
    cube_file : str
        Path to ELF cube file from pp.x.
    axis : int
        Axis perpendicular to the slice (0=x, 1=y, 2=z).
    slice_index : int or None
        Grid index along axis. None = midpoint.
    output_png : str
        Output image path.
    show_atoms : bool
        Overlay atom positions on the slice.
    """
    data, atoms = read_cube_data(cube_file)

    # ELF values range from 0 to 1
    data = np.clip(data, 0.0, 1.0)

    if slice_index is None:
        slice_index = data.shape[axis] // 2

    if axis == 0:
        sl = data[slice_index, :, :]
    elif axis == 1:
        sl = data[:, slice_index, :]
    else:
        sl = data[:, :, slice_index]

    cell = atoms.get_cell()
    axes_2d = [i for i in range(3) if i != axis]
    L1 = np.linalg.norm(cell[axes_2d[0]])
    L2 = np.linalg.norm(cell[axes_2d[1]])

    x = np.linspace(0, L1, sl.shape[0])
    y = np.linspace(0, L2, sl.shape[1])

    # Define contour levels meaningful for ELF interpretation
    levels = np.linspace(0, 1.0, 51)

    fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.contourf(x, y, sl.T, levels=levels, cmap='RdYlBu_r', norm=Normalize(0, 1))
    # Add contour lines at key ELF values
    contour_lines = [0.2, 0.5, 0.7, 0.85, 0.95]
    cs = ax.contour(x, y, sl.T, levels=contour_lines, colors='black',
                    linewidths=0.5, alpha=0.5)
    ax.clabel(cs, inline=True, fontsize=8, fmt='%.2f')

    cbar = plt.colorbar(im, ax=ax, ticks=[0, 0.2, 0.5, 0.7, 0.85, 1.0])
    cbar.set_label('ELF')
    cbar.ax.set_yticklabels(['0.0\n(delocalized)', '0.2', '0.5\n(free electron gas)',
                              '0.7', '0.85', '1.0\n(localized)'])

    # Overlay atom positions projected onto the slice plane
    if show_atoms:
        positions = atoms.get_positions()
        frac = atoms.get_scaled_positions()
        slice_frac = slice_index / data.shape[axis]
        tol = 0.1  # fractional tolerance for atoms near the slice
        for i, pos in enumerate(positions):
            f = frac[i]
            if abs(f[axis] - slice_frac) < tol or abs(f[axis] - slice_frac - 1) < tol:
                ax.plot(pos[axes_2d[0]], pos[axes_2d[1]], 'ko', markersize=10)
                ax.annotate(atoms.get_chemical_symbols()[i],
                           (pos[axes_2d[0]], pos[axes_2d[1]]),
                           textcoords="offset points", xytext=(5, 5),
                           fontsize=12, fontweight='bold')

    ax.set_xlabel(r'Distance ($\mathrm{\AA}$)')
    ax.set_ylabel(r'Distance ($\mathrm{\AA}$)')
    ax.set_title('Electron Localization Function (ELF)')
    ax.set_aspect('equal')
    plt.tight_layout()
    plt.savefig(output_png, dpi=200)
    plt.close()
    print(f"Saved: {output_png}")


# ----- Run -----
plot_elf_slice('elf.cube', axis=2, slice_index=None)
```

#### 3b: 1D ELF Profile Along a Bond

```python
#!/usr/bin/env python3
"""
Plot 1D ELF along a bond direction with interpretive annotations.
"""
import numpy as np
import matplotlib.pyplot as plt

BOHR_TO_ANG = 0.529177


def plot_elf_1d(data_file, output_png='elf_1d.png',
                atom_positions_ang=None, atom_labels=None):
    """
    Plot ELF along a line with bond-character annotations.

    Parameters
    ----------
    data_file : str
        pp.x output (iflag=1, output_format=0): two columns (dist_bohr, ELF).
    output_png : str
        Output image.
    atom_positions_ang : list of float or None
        Positions of atoms along the line in Angstrom.
    atom_labels : list of str or None
        Labels for atoms.
    """
    data = np.loadtxt(data_file)
    dist = data[:, 0] * BOHR_TO_ANG
    elf = data[:, 1]
    elf = np.clip(elf, 0, 1)

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(dist, elf, 'b-', linewidth=2)

    # Reference lines for ELF interpretation
    ax.axhline(y=1.0, color='red', linestyle=':', alpha=0.5, label='Perfectly localized (1.0)')
    ax.axhline(y=0.5, color='green', linestyle=':', alpha=0.5, label='Free electron gas (0.5)')
    ax.axhline(y=0.0, color='gray', linestyle=':', alpha=0.5, label='Fully delocalized (0.0)')

    # Shade interpretation regions
    ax.axhspan(0.85, 1.0, alpha=0.05, color='red', label='Covalent / lone pair')
    ax.axhspan(0.4, 0.6, alpha=0.05, color='green', label='Metallic-like')

    # Mark atom positions
    if atom_positions_ang is not None:
        for i, pos in enumerate(atom_positions_ang):
            label = atom_labels[i] if atom_labels else f'Atom {i}'
            ax.axvline(x=pos, color='black', linestyle='--', alpha=0.3)
            ax.annotate(label, (pos, 0.95), fontsize=12, ha='center', fontweight='bold')

    ax.set_xlabel(r'Distance along bond ($\mathrm{\AA}$)')
    ax.set_ylabel('ELF')
    ax.set_ylim(-0.05, 1.05)
    ax.set_title('ELF Along Bond Direction')
    ax.legend(loc='lower right', fontsize=9)
    ax.grid(True, alpha=0.2)
    plt.tight_layout()
    plt.savefig(output_png, dpi=200)
    plt.close()
    print(f"Saved: {output_png}")


# ----- Run -----
# For NaCl along [111]: Na at origin, Cl at a*sqrt(3)/2 ~ 3.88 Angstrom
plot_elf_1d(
    'elf_1d.dat',
    atom_positions_ang=[0.0, 3.88],
    atom_labels=['Na', 'Cl']
)
```

#### 3c: ELF from Cube -- Bond Characterization Summary

```python
#!/usr/bin/env python3
"""
Analyze ELF cube file to characterize bonding.
Extract ELF at bond midpoints between nearest neighbors.
"""
import numpy as np
from ase.io.cube import read_cube_data
from ase.neighborlist import neighbor_list


def characterize_bonds(cube_file, cutoff=3.5):
    """
    For each nearest-neighbor pair, evaluate ELF at the bond midpoint
    and classify the bond type.

    Parameters
    ----------
    cube_file : str
        ELF cube file.
    cutoff : float
        Neighbor cutoff in Angstrom.
    """
    data, atoms = read_cube_data(cube_file)
    data = np.clip(data, 0.0, 1.0)
    cell = atoms.get_cell()
    grid_shape = np.array(data.shape)

    # Find nearest neighbors
    idx_i, idx_j, dists = neighbor_list('ijd', atoms, cutoff)

    print(f"{'Bond':<12} {'Distance (A)':>12} {'ELF at midpoint':>16} {'Character':<15}")
    print("-" * 58)

    seen = set()
    for i, j, d in zip(idx_i, idx_j, dists):
        pair = (min(i, j), max(i, j))
        if pair in seen:
            continue
        seen.add(pair)

        symbols = atoms.get_chemical_symbols()
        pos_i = atoms.get_positions()[i]
        pos_j = atoms.get_positions()[j]
        midpoint = (pos_i + pos_j) / 2.0

        # Convert midpoint to fractional coordinates, then to grid index
        frac = np.linalg.solve(cell.T, midpoint)
        frac = frac % 1.0  # wrap into cell
        grid_idx = np.round(frac * grid_shape).astype(int) % grid_shape

        elf_val = data[grid_idx[0], grid_idx[1], grid_idx[2]]

        # Classify
        if elf_val > 0.85:
            character = "Covalent"
        elif elf_val > 0.65:
            character = "Polar covalent"
        elif elf_val > 0.45:
            character = "Metallic"
        elif elf_val > 0.25:
            character = "Ionic"
        else:
            character = "Very ionic/vdW"

        bond_label = f"{symbols[i]}{i}-{symbols[j]}{j}"
        print(f"{bond_label:<12} {d:>12.4f} {elf_val:>16.4f} {character:<15}")


# ----- Run -----
characterize_bonds('elf.cube', cutoff=3.5)
```

#### 3d: Compare ELF of Multiple Systems

```python
#!/usr/bin/env python3
"""
Compare ELF 1D profiles for multiple systems on one plot.
Useful for comparing bond character across polymorphs or substitutions.
"""
import numpy as np
import matplotlib.pyplot as plt

BOHR_TO_ANG = 0.529177


def compare_elf_profiles(file_list, labels, output_png='elf_compare.png'):
    """
    Overlay 1D ELF profiles from multiple systems.

    Parameters
    ----------
    file_list : list of str
        List of pp.x 1D output files.
    labels : list of str
        Legend labels for each system.
    output_png : str
        Output image.
    """
    fig, ax = plt.subplots(figsize=(10, 5))
    colors = plt.cm.tab10(np.linspace(0, 1, len(file_list)))

    for fname, label, color in zip(file_list, labels, colors):
        data = np.loadtxt(fname)
        dist = data[:, 0] * BOHR_TO_ANG
        elf = np.clip(data[:, 1], 0, 1)
        # Normalize distance to [0, 1] for comparison of different cell sizes
        dist_norm = dist / dist.max()
        ax.plot(dist_norm, elf, '-', color=color, linewidth=1.5, label=label)

    ax.axhline(y=0.5, color='gray', linestyle=':', alpha=0.4)
    ax.set_xlabel('Normalized distance along bond')
    ax.set_ylabel('ELF')
    ax.set_ylim(-0.05, 1.05)
    ax.set_title('ELF Comparison')
    ax.legend()
    ax.grid(True, alpha=0.2)
    plt.tight_layout()
    plt.savefig(output_png, dpi=200)
    plt.close()
    print(f"Saved: {output_png}")


# ----- Run -----
compare_elf_profiles(
    ['elf_1d_nacl.dat', 'elf_1d_si.dat', 'elf_1d_cu.dat'],
    ['NaCl (ionic)', 'Si (covalent)', 'Cu (metallic)']
)
```

## Key Parameters

### pp.x Settings for ELF

| Parameter | Value | Notes |
|-----------|-------|-------|
| `plot_num` | `8` | ELF |
| `iflag` | `1` / `2` / `3` | 1D / 2D / 3D output |
| `output_format` | `6` | Cube file (for iflag=3) |
| `output_format` | `7` | Gnuplot format (for iflag=2) |
| `output_format` | `0` | Text columns (for iflag=1) |

### SCF Convergence

ELF depends on the kinetic energy density of occupied orbitals. Requirements:

| Parameter | Recommendation |
|-----------|---------------|
| `ecutwfc` | Well converged (test with ELF convergence) |
| `ecutrho` | 8-12x ecutwfc for USPP/PAW |
| `conv_thr` | 1.0d-8 or tighter |
| K-points | Dense mesh, especially for metals |

## Interpreting Results

### ELF Value Ranges

| ELF Value | Interpretation |
|-----------|---------------|
| 0.9 -- 1.0 | Strongly localized: covalent bond, lone pair, core electrons |
| 0.7 -- 0.9 | Partially localized: polar covalent bonds |
| 0.45 -- 0.55 | Free-electron-gas-like: metallic bonding |
| 0.2 -- 0.45 | Delocalized region between ions: ionic character |
| 0.0 -- 0.2 | Strongly delocalized: between closed-shell ions, vacuum |

### Bond Type Signatures

- **Covalent bond (e.g., Si-Si, C-C)**: ELF maximum (near 1.0) at bond midpoint. The isosurface shows a basin connecting the two atoms.
- **Ionic bond (e.g., Na-Cl)**: ELF is high around anions (localized electrons) and low between atoms. No ELF basin at the bond midpoint.
- **Metallic bond (e.g., Cu-Cu)**: ELF approximately 0.5 throughout the interstitial region. No distinct bonding basins; electron-gas-like behavior.
- **Lone pairs (e.g., N in NH3)**: ELF maximum away from the bond axis, in the direction of the lone pair. Appears as an isosurface lobe.
- **Core electrons**: ELF near 1.0 in spherical shells around nuclei -- these are not related to bonding.

### Visual Features

- **Saddle points in ELF** between two atoms with ELF > 0.7: covalent bond.
- **ELF isosurface at 0.8** shows covalent bond basins; at 0.5 shows the metallic electron gas.
- **Ring-shaped ELF maxima** around atoms: core electron shells.

## Common Issues

1. **ELF values > 1 or < 0**: This can happen with numerical noise. Clip to [0, 1] in post-processing.
2. **ELF looks noisy**: Increase `ecutwfc` and `ecutrho` for a smoother ELF. Also check k-point convergence.
3. **ELF is 0 everywhere in vacuum**: This is correct -- ELF is undefined in regions of zero density, and pp.x sets it to 0.
4. **ELF comparison between systems**: ELF is already normalized (0 to 1), so direct comparison between different materials is meaningful.
5. **Spin-polarized systems**: For spin-polarized calculations, pp.x computes ELF separately for spin-up and spin-down. You may need to set `spin_component` in `&INPUTPP`.
6. **PAW vs USPP vs NC**: ELF from pp.x uses the pseudo-wavefunctions. PAW reconstructed all-electron ELF is not directly available from pp.x. The pseudo-ELF is usually qualitatively correct for bonding analysis.
7. **Plane definition**: When using `iflag=2`, the vectors `e1` and `e2` are in units of the lattice parameter (alat). For non-orthogonal cells, choose the plane carefully to pass through bond midpoints.
