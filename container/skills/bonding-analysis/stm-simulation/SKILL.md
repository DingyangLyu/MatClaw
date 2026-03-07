# STM Simulation

Simulate scanning tunneling microscope (STM) images based on the Tersoff-Hamann approximation. Generate constant-current and constant-height STM images from DFT calculations using Quantum ESPRESSO or VASP.

## When to Use

- Predict STM images to compare with experimental measurements.
- Identify surface features: adatoms, vacancies, reconstructions, molecular adsorbates.
- Determine which orbitals contribute to STM contrast at different bias voltages.
- Distinguish between different adsorption sites or surface configurations.
- Simulate bias-dependent STM images (filled vs. empty states).

## Method Selection

| Method | Code | Best For | Notes |
|--------|------|----------|-------|
| pp.x plot_num=5 (local DOS) | QE | Integrated LDOS at Fermi level | Tersoff-Hamann: STM ~ LDOS(r, E_F) |
| pp.x plot_num=10 (integrated LDOS) | QE | Energy-integrated LDOS to specified energy | For finite bias voltage |
| PARCHG (partial charge) | VASP | Partial charge in energy window near E_F | Use EINT or IBAND in INCAR |
| VASPKIT 325 | VASP | Quick STM image generation | Automated from PARCHG |
| Python from cube file | QE/VASP | Full control over height, current mode | Post-processing flexibility |

## Prerequisites

- Quantum ESPRESSO 7.5 (`pw.x`, `pp.x`) for QE workflows
- VASP (future) for VASP workflows
- Python: `numpy`, `matplotlib`, `scipy`, `ase`
- Well-converged SCF calculation for a slab system
- Sufficient vacuum (> 12 Angstrom) above the surface
- Dense k-point sampling in the surface plane (important for metallic surfaces)

## Detailed Steps

### QE Workflow: STM from Local DOS

#### Step 1: SCF Calculation for a Surface

**File: `scf_surface.in`** (example: Si(111) 7x7 simplified as Si(001) 2x1 for demonstration)

```
&CONTROL
    calculation  = 'scf'
    prefix       = 'si_surface'
    outdir       = './tmp'
    pseudo_dir   = './pseudo'
    tprnfor      = .true.
/
&SYSTEM
    ibrav        = 0
    nat          = 8
    ntyp         = 1
    ecutwfc      = 40.0
    ecutrho      = 320.0
    occupations  = 'smearing'
    smearing     = 'cold'
    degauss      = 0.005
    nbnd         = 30
/
&ELECTRONS
    conv_thr     = 1.0d-8
    mixing_beta  = 0.3
/
ATOMIC_SPECIES
  Si  28.086   Si.pbe-n-rrkjus_psl.1.0.0.UPF

CELL_PARAMETERS {angstrom}
  7.68   0.00   0.00
  0.00   5.43   0.00
  0.00   0.00  25.00

ATOMIC_POSITIONS {angstrom}
  Si   0.000  0.000  8.000
  Si   1.920  1.358  9.358
  Si   3.840  0.000  8.000
  Si   5.760  1.358  9.358
  Si   0.000  2.715  8.000
  Si   1.920  4.073  9.358
  Si   3.840  2.715  8.000
  Si   5.760  4.073  9.358

K_POINTS {automatic}
  4 6 1  0 0 0
```

```bash
pw.x < scf_surface.in > scf_surface.out
```

#### Step 2: Extract Local DOS at the Fermi Level with pp.x

**Tersoff-Hamann approximation**: The STM current is proportional to the local density of states (LDOS) at the tip position at the Fermi energy:

  I ~ LDOS(r_tip, E_F)

pp.x with `plot_num=5` computes the LDOS at a specified energy.

**File: `pp_stm.in`** (LDOS at E_Fermi)

```
&INPUTPP
    prefix   = 'si_surface'
    outdir   = './tmp'
    filplot  = 'stm_ldos.dat'
    plot_num = 5
    sample_bias = 0.0
/
&PLOT
    nfile       = 1
    filepp(1)   = 'stm_ldos.dat'
    weight(1)   = 1.0
    iflag       = 3
    output_format = 6
    fileout     = 'stm_ldos.cube'
/
```

The `sample_bias` parameter (in Ry) defines the energy window relative to the Fermi level:
- `sample_bias = 0.0` -- LDOS at E_Fermi (zero-bias limit)
- `sample_bias < 0` -- integrated LDOS from E_F+sample_bias to E_F (filled states, negative bias = typical STM imaging of occupied states)
- `sample_bias > 0` -- integrated LDOS from E_F to E_F+sample_bias (empty states)

```bash
pp.x < pp_stm.in > pp_stm.out
```

#### Step 2b: Filled-States STM (Negative Bias)

**File: `pp_stm_filled.in`** (simulate -1.0 V bias)

```
&INPUTPP
    prefix   = 'si_surface'
    outdir   = './tmp'
    filplot  = 'stm_filled.dat'
    plot_num = 5
    sample_bias = -0.0735
/
&PLOT
    nfile       = 1
    filepp(1)   = 'stm_filled.dat'
    weight(1)   = 1.0
    iflag       = 3
    output_format = 6
    fileout     = 'stm_filled.cube'
/
```

Note: `sample_bias` is in Ry. Convert from V: bias_Ry = bias_V / 13.6057.

**File: `pp_stm_empty.in`** (simulate +1.0 V bias)

```
&INPUTPP
    prefix   = 'si_surface'
    outdir   = './tmp'
    filplot  = 'stm_empty.dat'
    plot_num = 5
    sample_bias = 0.0735
/
&PLOT
    nfile       = 1
    filepp(1)   = 'stm_empty.dat'
    weight(1)   = 1.0
    iflag       = 3
    output_format = 6
    fileout     = 'stm_empty.cube'
/
```

```bash
pp.x < pp_stm_filled.in > pp_stm_filled.out
pp.x < pp_stm_empty.in > pp_stm_empty.out
```

#### Step 3: Generate STM Images

##### 3a: Constant-Height STM Image

```python
#!/usr/bin/env python3
"""
Generate a constant-height STM image from a cube file.
A constant-height image shows LDOS(x, y) at a fixed z above the surface.
"""
import numpy as np
import matplotlib.pyplot as plt
from ase.io.cube import read_cube_data
from scipy.interpolate import RegularGridInterpolator


def constant_height_stm(cube_file, height_angstrom=3.0,
                         output_png='stm_constant_height.png',
                         cmap='hot', log_scale=False):
    """
    Generate constant-height STM image.

    Parameters
    ----------
    cube_file : str
        LDOS cube file from pp.x (plot_num=5).
    height_angstrom : float
        Height above the topmost atom in Angstrom.
    output_png : str
        Output image.
    cmap : str
        Colormap (use 'hot' or 'copper' for STM-like appearance).
    log_scale : bool
        If True, plot log10(LDOS) for better dynamic range.
    """
    data, atoms = read_cube_data(cube_file)
    cell = atoms.get_cell()
    nx, ny, nz = data.shape

    # Find the topmost atom z-coordinate
    z_positions = atoms.get_positions()[:, 2]
    z_top = np.max(z_positions)
    z_slice = z_top + height_angstrom

    # Convert to fractional coordinate along z
    c_length = np.linalg.norm(cell[2])
    z_frac = z_slice / c_length

    if z_frac >= 1.0 or z_frac < 0.0:
        print(f"WARNING: STM height {z_slice:.2f} A is outside the cell "
              f"(c = {c_length:.2f} A). Wrapping.")
        z_frac = z_frac % 1.0

    z_idx = int(z_frac * nz)
    z_idx = min(z_idx, nz - 1)

    # Extract the 2D slice
    stm_image = data[:, :, z_idx]

    # Apply log scale if requested
    if log_scale:
        stm_image = np.log10(np.maximum(stm_image, 1e-10))

    # Plot dimensions
    a_length = np.linalg.norm(cell[0])
    b_length = np.linalg.norm(cell[1])

    fig, ax = plt.subplots(figsize=(8, 8 * b_length / a_length))
    x = np.linspace(0, a_length, nx)
    y = np.linspace(0, b_length, ny)

    im = ax.pcolormesh(x, y, stm_image.T, cmap=cmap, shading='gouraud')
    cbar = plt.colorbar(im, ax=ax, shrink=0.8)
    if log_scale:
        cbar.set_label(r'log$_{10}$(LDOS) (arb. units)')
    else:
        cbar.set_label('LDOS (arb. units)')

    ax.set_xlabel(r'x ($\mathrm{\AA}$)')
    ax.set_ylabel(r'y ($\mathrm{\AA}$)')
    ax.set_title(f'Constant-Height STM (h = {height_angstrom:.1f} A above surface)')
    ax.set_aspect('equal')
    plt.tight_layout()
    plt.savefig(output_png, dpi=300)
    plt.close()
    print(f"Saved: {output_png}")
    print(f"STM slice at z = {z_slice:.2f} A (grid index {z_idx}/{nz})")
    print(f"LDOS range: {np.min(stm_image):.2e} to {np.max(stm_image):.2e}")


# ----- Run -----
constant_height_stm('stm_ldos.cube', height_angstrom=3.0,
                     output_png='stm_constant_height.png')
```

##### 3b: Constant-Current STM Image

```python
#!/usr/bin/env python3
"""
Generate a constant-current STM image from a cube file.
A constant-current image shows the height z(x, y) at which
the LDOS equals a specified isosurface value.
This simulates the topographic mode of STM.
"""
import numpy as np
import matplotlib.pyplot as plt
from ase.io.cube import read_cube_data
from scipy.interpolate import RegularGridInterpolator


def constant_current_stm(cube_file, isoval=None, output_png='stm_constant_current.png',
                           cmap='copper', search_from_top=True):
    """
    Generate constant-current STM image by finding the isosurface height.

    Parameters
    ----------
    cube_file : str
        LDOS cube file from pp.x.
    isoval : float or None
        LDOS isovalue for the "current" threshold. If None, uses
        a value at the 10th percentile of non-zero LDOS in the
        vacuum region (a reasonable default).
    output_png : str
        Output image.
    cmap : str
        Colormap.
    search_from_top : bool
        If True, search downward from the top of the cell (vacuum side)
        to find the isosurface. This avoids catching the slab interior.
    """
    data, atoms = read_cube_data(cube_file)
    cell = atoms.get_cell()
    nx, ny, nz = data.shape
    c_length = np.linalg.norm(cell[2])

    # Auto-determine isoval if not specified
    if isoval is None:
        # Use LDOS values in the vacuum/tip region
        z_top_atom = np.max(atoms.get_positions()[:, 2])
        z_frac_top = z_top_atom / c_length
        z_idx_top = int(z_frac_top * nz)
        # Sample LDOS 2-5 A above the surface
        z_start = int((z_top_atom + 2.0) / c_length * nz)
        z_end = int((z_top_atom + 5.0) / c_length * nz)
        z_start = max(0, min(z_start, nz - 1))
        z_end = max(0, min(z_end, nz - 1))
        if z_start < z_end:
            vacuum_ldos = data[:, :, z_start:z_end]
            positive = vacuum_ldos[vacuum_ldos > 0]
            if len(positive) > 0:
                isoval = np.percentile(positive, 50)
            else:
                isoval = 1e-4
        else:
            isoval = 1e-4
        print(f"Auto-determined isovalue: {isoval:.2e}")

    a_length = np.linalg.norm(cell[0])
    b_length = np.linalg.norm(cell[1])

    # For each (x,y) pixel, find the z where LDOS = isoval
    # Search from the top (vacuum side) downward
    height_map = np.zeros((nx, ny))
    z_values = np.linspace(0, c_length, nz, endpoint=False)

    for ix in range(nx):
        for iy in range(ny):
            profile = data[ix, iy, :]
            if search_from_top:
                # Search from top of cell downward
                for iz in range(nz - 1, 0, -1):
                    if profile[iz] >= isoval:
                        # Linear interpolation between grid points
                        if iz < nz - 1 and profile[iz + 1] < isoval:
                            frac = (isoval - profile[iz]) / (profile[iz + 1] - profile[iz] + 1e-30)
                            height_map[ix, iy] = z_values[iz] + frac * (c_length / nz)
                        else:
                            height_map[ix, iy] = z_values[iz]
                        break
            else:
                for iz in range(nz):
                    if profile[iz] >= isoval:
                        height_map[ix, iy] = z_values[iz]
                        break

    # Remove background (subtract minimum to show corrugation)
    h_min = np.min(height_map[height_map > 0]) if np.any(height_map > 0) else 0
    corrugation = height_map - h_min

    fig, axes = plt.subplots(1, 2, figsize=(16, 7))

    # Topographic image
    ax1 = axes[0]
    x = np.linspace(0, a_length, nx)
    y = np.linspace(0, b_length, ny)
    im1 = ax1.pcolormesh(x, y, height_map.T, cmap=cmap, shading='gouraud')
    cbar1 = plt.colorbar(im1, ax=ax1, shrink=0.8)
    cbar1.set_label(r'Height ($\mathrm{\AA}$)')
    ax1.set_xlabel(r'x ($\mathrm{\AA}$)')
    ax1.set_ylabel(r'y ($\mathrm{\AA}$)')
    ax1.set_title(f'Constant-Current STM (isoval = {isoval:.2e})')
    ax1.set_aspect('equal')

    # Corrugation image
    ax2 = axes[1]
    im2 = ax2.pcolormesh(x, y, corrugation.T * 100,  # convert to pm
                          cmap='RdYlBu_r', shading='gouraud')
    cbar2 = plt.colorbar(im2, ax=ax2, shrink=0.8)
    cbar2.set_label('Corrugation (pm)')
    ax2.set_xlabel(r'x ($\mathrm{\AA}$)')
    ax2.set_ylabel(r'y ($\mathrm{\AA}$)')
    ax2.set_title('Surface Corrugation')
    ax2.set_aspect('equal')

    plt.tight_layout()
    plt.savefig(output_png, dpi=300)
    plt.close()

    max_corr = np.max(corrugation) * 100  # pm
    print(f"Saved: {output_png}")
    print(f"Maximum corrugation: {max_corr:.1f} pm")
    print(f"Height range: {np.min(height_map):.2f} -- {np.max(height_map):.2f} A")


# ----- Run -----
constant_current_stm('stm_ldos.cube', isoval=None,
                      output_png='stm_constant_current.png')
```

##### 3c: Bias-Dependent STM Images (Multi-Panel)

```python
#!/usr/bin/env python3
"""
Generate multi-panel bias-dependent STM images.
Requires separate pp.x runs at different sample_bias values.
"""
import numpy as np
import matplotlib.pyplot as plt
from ase.io.cube import read_cube_data


def bias_dependent_stm(cube_files, bias_voltages, height_angstrom=3.0,
                        output_png='stm_bias_dependent.png'):
    """
    Plot constant-height STM images at multiple bias voltages side by side.

    Parameters
    ----------
    cube_files : list of str
        LDOS cube files at different biases.
    bias_voltages : list of float
        Corresponding bias voltages in V (for labeling).
    height_angstrom : float
        Height above topmost atom.
    output_png : str
        Output image.
    """
    n_panels = len(cube_files)
    fig, axes = plt.subplots(1, n_panels, figsize=(5 * n_panels, 5))
    if n_panels == 1:
        axes = [axes]

    for ax, cube_file, bias in zip(axes, cube_files, bias_voltages):
        data, atoms = read_cube_data(cube_file)
        cell = atoms.get_cell()
        nx, ny, nz = data.shape

        z_top = np.max(atoms.get_positions()[:, 2])
        c_length = np.linalg.norm(cell[2])
        z_slice = z_top + height_angstrom
        z_idx = int((z_slice / c_length) * nz)
        z_idx = min(z_idx, nz - 1)

        stm_image = data[:, :, z_idx]

        a_length = np.linalg.norm(cell[0])
        b_length = np.linalg.norm(cell[1])
        x = np.linspace(0, a_length, nx)
        y = np.linspace(0, b_length, ny)

        im = ax.pcolormesh(x, y, stm_image.T, cmap='hot', shading='gouraud')
        plt.colorbar(im, ax=ax, shrink=0.7)
        ax.set_xlabel(r'x ($\mathrm{\AA}$)')
        ax.set_ylabel(r'y ($\mathrm{\AA}$)')
        sign = '+' if bias >= 0 else ''
        state_type = 'empty states' if bias > 0 else 'filled states' if bias < 0 else 'E_F'
        ax.set_title(f'V = {sign}{bias:.1f} V\n({state_type})')
        ax.set_aspect('equal')

    plt.suptitle(f'Bias-Dependent STM (h = {height_angstrom:.1f} A)', fontsize=14)
    plt.tight_layout()
    plt.savefig(output_png, dpi=300)
    plt.close()
    print(f"Saved: {output_png}")


# ----- Run -----
bias_dependent_stm(
    cube_files=['stm_filled.cube', 'stm_ldos.cube', 'stm_empty.cube'],
    bias_voltages=[-1.0, 0.0, 1.0],
    height_angstrom=3.0,
    output_png='stm_bias_dependent.png'
)
```

##### 3d: STM Image with Supercell Tiling

```python
#!/usr/bin/env python3
"""
Tile the STM image into a supercell for better visualization
of periodic surface features.
"""
import numpy as np
import matplotlib.pyplot as plt
from ase.io.cube import read_cube_data


def tiled_stm(cube_file, height_angstrom=3.0, tile=(3, 3),
               output_png='stm_tiled.png', cmap='hot'):
    """
    Generate a tiled (supercell) STM image.

    Parameters
    ----------
    cube_file : str
        LDOS cube file.
    height_angstrom : float
        Height above surface.
    tile : tuple of int
        (nx, ny) tiling factors.
    output_png : str
        Output image.
    """
    data, atoms = read_cube_data(cube_file)
    cell = atoms.get_cell()
    nx, ny, nz = data.shape

    z_top = np.max(atoms.get_positions()[:, 2])
    c_length = np.linalg.norm(cell[2])
    z_idx = int((z_top + height_angstrom) / c_length * nz)
    z_idx = min(z_idx, nz - 1)

    stm_unit = data[:, :, z_idx]

    # Tile
    stm_tiled = np.tile(stm_unit, tile)

    a_length = np.linalg.norm(cell[0]) * tile[0]
    b_length = np.linalg.norm(cell[1]) * tile[1]

    fig, ax = plt.subplots(figsize=(8, 8 * b_length / a_length))
    x = np.linspace(0, a_length, stm_tiled.shape[0])
    y = np.linspace(0, b_length, stm_tiled.shape[1])
    im = ax.pcolormesh(x, y, stm_tiled.T, cmap=cmap, shading='gouraud')
    cbar = plt.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label('LDOS (arb. units)')

    # Draw unit cell boundaries
    a_unit = np.linalg.norm(cell[0])
    b_unit = np.linalg.norm(cell[1])
    for i in range(1, tile[0]):
        ax.axvline(x=i * a_unit, color='white', linestyle='--',
                   linewidth=0.5, alpha=0.5)
    for j in range(1, tile[1]):
        ax.axhline(y=j * b_unit, color='white', linestyle='--',
                   linewidth=0.5, alpha=0.5)

    ax.set_xlabel(r'x ($\mathrm{\AA}$)')
    ax.set_ylabel(r'y ($\mathrm{\AA}$)')
    ax.set_title(f'STM Image ({tile[0]}x{tile[1]} supercell)')
    ax.set_aspect('equal')
    plt.tight_layout()
    plt.savefig(output_png, dpi=300)
    plt.close()
    print(f"Saved: {output_png}")


# ----- Run -----
tiled_stm('stm_ldos.cube', height_angstrom=3.0, tile=(3, 3),
           output_png='stm_tiled.png')
```

### VASP Workflow: STM from PARCHG

#### Step 1: Generate PARCHG Near Fermi Level

Add to INCAR for the SCF or a follow-up non-SCF calculation:

```
# INCAR additions for STM simulation
LPARD  = .TRUE.      # Write partial charge density
LSEPB  = .FALSE.     # Do not separate bands
LSEPK  = .FALSE.     # Do not separate k-points

# Energy window near Fermi level (eV)
# For filled states (negative bias, e.g., -1 V):
EINT   = -1.0  0.0   # Integrate LDOS from E_F-1 to E_F

# Alternative: for empty states (positive bias):
# EINT   = 0.0  1.0   # Integrate from E_F to E_F+1
```

Run VASP. It produces `PARCHG` containing the partial charge density in the specified energy window.

#### Step 2: VASPKIT 325

```bash
vaspkit -task 325
# Reads PARCHG and generates STM image data
# Follow prompts for height and imaging mode
```

#### Step 3: Parse PARCHG with pymatgen

```python
#!/usr/bin/env python3
"""
Generate STM image from VASP PARCHG using pymatgen.
"""
import numpy as np
import matplotlib.pyplot as plt
from pymatgen.io.vasp import Chgcar


def vasp_stm_from_parchg(parchg_file='PARCHG', height_angstrom=3.0,
                           mode='constant_height',
                           output_png='stm_vasp.png'):
    """
    Generate STM image from VASP PARCHG.

    Parameters
    ----------
    parchg_file : str
        Path to PARCHG file.
    height_angstrom : float
        Height above topmost atom (for constant-height mode).
    mode : str
        'constant_height' or 'constant_current'.
    output_png : str
        Output image.
    """
    parchg = Chgcar.from_file(parchg_file)
    structure = parchg.structure
    data = parchg.data['total']
    vol = structure.volume

    # Normalize
    ldos = data / vol

    lattice = structure.lattice
    nx, ny, nz = ldos.shape

    if mode == 'constant_height':
        z_top = max(site.coords[2] for site in structure)
        z_slice = z_top + height_angstrom
        z_frac = z_slice / lattice.c
        z_idx = int(z_frac * nz) % nz

        stm_image = ldos[:, :, z_idx]

        fig, ax = plt.subplots(figsize=(8, 8 * lattice.b / lattice.a))
        x = np.linspace(0, lattice.a, nx)
        y = np.linspace(0, lattice.b, ny)
        im = ax.pcolormesh(x, y, stm_image.T, cmap='hot', shading='gouraud')
        cbar = plt.colorbar(im, ax=ax, shrink=0.8)
        cbar.set_label('LDOS (arb. units)')
        ax.set_xlabel(r'x ($\mathrm{\AA}$)')
        ax.set_ylabel(r'y ($\mathrm{\AA}$)')
        ax.set_title(f'Constant-Height STM (h = {height_angstrom:.1f} A)')
        ax.set_aspect('equal')

    elif mode == 'constant_current':
        z_values = np.linspace(0, lattice.c, nz, endpoint=False)
        # Auto-determine isosurface value
        z_top = max(site.coords[2] for site in structure)
        z_start = int((z_top + 2.0) / lattice.c * nz)
        z_end = int((z_top + 5.0) / lattice.c * nz)
        z_start = max(0, min(z_start, nz - 1))
        z_end = max(z_start + 1, min(z_end, nz - 1))
        vac_data = ldos[:, :, z_start:z_end]
        positive = vac_data[vac_data > 0]
        isoval = np.percentile(positive, 50) if len(positive) > 0 else 1e-5

        height_map = np.zeros((nx, ny))
        for ix in range(nx):
            for iy in range(ny):
                profile = ldos[ix, iy, :]
                for iz in range(nz - 1, 0, -1):
                    if profile[iz] >= isoval:
                        height_map[ix, iy] = z_values[iz]
                        break

        fig, ax = plt.subplots(figsize=(8, 8 * lattice.b / lattice.a))
        x = np.linspace(0, lattice.a, nx)
        y = np.linspace(0, lattice.b, ny)
        im = ax.pcolormesh(x, y, height_map.T, cmap='copper', shading='gouraud')
        cbar = plt.colorbar(im, ax=ax, shrink=0.8)
        cbar.set_label(r'Height ($\mathrm{\AA}$)')
        ax.set_xlabel(r'x ($\mathrm{\AA}$)')
        ax.set_ylabel(r'y ($\mathrm{\AA}$)')
        ax.set_title(f'Constant-Current STM (isoval = {isoval:.2e})')
        ax.set_aspect('equal')

    plt.tight_layout()
    plt.savefig(output_png, dpi=300)
    plt.close()
    print(f"Saved: {output_png}")


# ----- Run -----
# vasp_stm_from_parchg('PARCHG', height_angstrom=3.0,
#                        mode='constant_height',
#                        output_png='stm_vasp_height.png')
# vasp_stm_from_parchg('PARCHG', height_angstrom=3.0,
#                        mode='constant_current',
#                        output_png='stm_vasp_current.png')
```

## Key Parameters

### QE pp.x Settings for STM

| Parameter | Value | Purpose |
|-----------|-------|---------|
| `plot_num` | `5` | Local DOS (Tersoff-Hamann) |
| `sample_bias` | e.g., `-0.0735` | Bias in Ry (-1 V = -0.0735 Ry) |
| `plot_num` | `10` | Integrated LDOS up to E_F |
| `iflag` | `3` | 3D grid output |
| `output_format` | `6` | Gaussian cube format |

### Bias Voltage Conversion

| Bias (V) | sample_bias (Ry) | States Probed |
|----------|-------------------|---------------|
| -2.0 | -0.1470 | Occupied states (E_F - 2 eV to E_F) |
| -1.0 | -0.0735 | Occupied states (E_F - 1 eV to E_F) |
| -0.5 | -0.0368 | Occupied states near E_F |
| 0.0 | 0.0 | LDOS at E_F only |
| +0.5 | +0.0368 | Unoccupied states near E_F |
| +1.0 | +0.0735 | Unoccupied states (E_F to E_F + 1 eV) |
| +2.0 | +0.1470 | Unoccupied states (E_F to E_F + 2 eV) |

### VASP INCAR Settings for PARCHG

| Parameter | Value | Purpose |
|-----------|-------|---------|
| `LPARD` | `.TRUE.` | Generate partial charge density |
| `EINT` | e.g., `-1.0 0.0` | Energy window [E_F + EINT(1), E_F + EINT(2)] |
| `IBAND` | e.g., `20 21 22` | Alternatively, select specific bands |
| `LSEPB` | `.FALSE.` | Sum over bands (do not separate) |
| `LSEPK` | `.FALSE.` | Sum over k-points |
| `NBMOD` | `-3` | Alternative: states in range [E_F+EINT(1), E_F+EINT(2)] |

### STM Height and Resolution

| Parameter | Typical Value | Notes |
|-----------|---------------|-------|
| STM tip height | 2 -- 5 Angstrom | Above topmost atom; 3 A is typical |
| Lateral resolution | Surface unit cell / (nx, ny) | Determined by FFT grid |
| Isosurface value | 1e-5 -- 1e-3 e/bohr^3 | For constant-current mode |
| K-point density | Dense in-plane | Metallic surfaces need > 8x8 |

## Interpreting Results

### Constant-Height Images

- **Bright spots**: High LDOS at tip height -- atoms or features contributing to tunneling.
- **Filled-state image** (V < 0): Shows occupied states; typically reflects covalent bonds and core electron density.
- **Empty-state image** (V > 0): Shows unoccupied states; may reveal different symmetry (e.g., dangling bonds, antibonding states).
- **Bias dependence**: Different features appear at different biases because different orbitals contribute.

### Constant-Current Images

- **Topographic information**: Height map shows apparent height of surface features.
- **Corrugation amplitude**: Peak-to-valley height difference. Typical: 10 -- 200 pm for clean metal surfaces, up to several Angstrom for molecular adsorbates.
- **Adatoms appear as protrusions**: Higher LDOS at adatom site raises the isosurface height.
- **Vacancies appear as depressions**: Reduced LDOS at vacancy site.

### Comparison with Experiment

- **Qualitative agreement**: Shape and symmetry of bright features should match experiment.
- **Quantitative corrugation**: DFT typically overestimates or underestimates corrugation by 20-50% due to finite tip effects not captured by Tersoff-Hamann.
- **Tip convolution**: Real STM tips have finite size; simulations assume an ideal s-wave tip.

### Physical Features

| Feature | STM Signature |
|---------|---------------|
| Adatom | Bright protrusion at single lattice site |
| Vacancy | Dark depression; may show enhanced contrast at neighbors |
| Step edge | Line of bright or dark contrast |
| Surface reconstruction | Periodic pattern different from bulk termination |
| Molecular adsorbate | Complex pattern reflecting molecular orbital symmetry |
| Standing waves (Friedel oscillations) | Concentric rings near defects on metal surfaces |

## Common Issues

| Issue | Cause | Solution |
|-------|-------|----------|
| STM image is uniformly dark | Height too far from surface or LDOS decayed | Reduce height or increase sample_bias window |
| Image shows only noise | SCF not converged or too few k-points | Tighten conv_thr; increase k-point density |
| Bright spots do not correspond to atom positions | STM images orbitals, not atoms | This is correct; compare with PDOS to identify contributing orbitals |
| Asymmetric image for symmetric surface | Insufficient k-points or symmetry breaking | Increase k-mesh; check surface relaxation |
| plot_num=5 gives zero everywhere | sample_bias too small or no states at E_F | Increase sample_bias window; check if system is insulating |
| VASP PARCHG is empty | EINT window contains no states | Adjust EINT range; check band structure near E_F |
| Constant-current fails to find isosurface | Isosurface value too high or wrong search direction | Lower isovalue or verify search_from_top direction |
| STM image periodic features do not match experiment | Wrong surface reconstruction or unit cell | Verify surface structure; may need larger supercell |
