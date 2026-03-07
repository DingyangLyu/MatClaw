# Charge Density Difference

Compute charge density difference (CDD) to visualize charge redistribution upon bonding, adsorption, or interface formation. Calculates rho_AB - rho_A - rho_B using Quantum ESPRESSO pp.x or VASP CHGCAR subtraction.

## When to Use

- Visualize charge redistribution when a molecule adsorbs on a surface.
- Analyze bonding at heterostructure interfaces (e.g., metal/oxide, 2D/2D).
- Understand charge transfer in molecular complexes or doped systems.
- Compute spin density difference (rho_up - rho_down) for magnetic systems.
- Compare bonding character across different adsorption sites or configurations.

## Method Selection

| Method | Code | Best For | Notes |
|--------|------|----------|-------|
| pp.x weight subtraction | QE | Simple two-fragment systems | Built-in, no external tools needed |
| pp.x + Python subtraction | QE | Multi-fragment, flexible analysis | More control over grid handling |
| CHGCAR subtraction (pymatgen) | VASP | VASP users, large systems | Requires matching FFT grids |
| VASPKIT 314 | VASP | Quick interactive CDD | Automated CHGCAR subtraction |
| Spin density difference | QE/VASP | Magnetic systems | plot_num=6 in QE; CHGCAR spin channel in VASP |

## Prerequisites

- Quantum ESPRESSO 7.5 (`pw.x`, `pp.x`) for QE workflows
- VASP (future) for VASP workflows
- Python: `numpy`, `matplotlib`, `ase`, `pymatgen`
- All fragment calculations must use the **same supercell** and **same FFT grid** (same `ecutwfc`/`ecutrho` and lattice vectors)
- Pseudopotential files for your elements

## Detailed Steps

### QE Workflow: Charge Density Difference for Molecule on Surface

This example computes the CDD for CO adsorbed on a Cu(111) surface.

#### Step 1: Three SCF Calculations

You need three separate SCF calculations with the **same cell**, **same cutoffs**, and **same k-point grid**:

1. **Full system** (surface + adsorbate)
2. **Fragment A** (bare surface, adsorbate removed but cell kept)
3. **Fragment B** (isolated adsorbate in same cell, surface removed)

**File: `scf_full.in`** (CO on Cu(111) slab)

```
&CONTROL
    calculation  = 'scf'
    prefix       = 'cu_co'
    outdir       = './tmp_full'
    pseudo_dir   = './pseudo'
    tprnfor      = .true.
/
&SYSTEM
    ibrav        = 0
    nat          = 5
    ntyp         = 3
    ecutwfc      = 50.0
    ecutrho      = 400.0
    occupations  = 'smearing'
    smearing     = 'cold'
    degauss      = 0.01
/
&ELECTRONS
    conv_thr     = 1.0d-8
    mixing_beta  = 0.3
/
ATOMIC_SPECIES
  Cu  63.546   Cu.pbe-dn-rrkjus_psl.1.0.0.UPF
  C   12.011   C.pbe-n-rrkjus_psl.1.0.0.UPF
  O   15.999   O.pbe-n-rrkjus_psl.1.0.0.UPF

CELL_PARAMETERS {angstrom}
  2.556  0.000  0.000
  1.278  2.213  0.000
  0.000  0.000  20.000

ATOMIC_POSITIONS {angstrom}
  Cu  0.000  0.000  0.000
  Cu  1.278  0.738  2.087
  Cu  2.556  1.475  4.174
  C   0.000  0.000  6.100
  O   0.000  0.000  7.243

K_POINTS {automatic}
  6 6 1  0 0 0
```

**File: `scf_slab.in`** (bare Cu(111) -- remove CO, keep cell)

```
&CONTROL
    calculation  = 'scf'
    prefix       = 'cu_slab'
    outdir       = './tmp_slab'
    pseudo_dir   = './pseudo'
    tprnfor      = .true.
/
&SYSTEM
    ibrav        = 0
    nat          = 3
    ntyp         = 1
    ecutwfc      = 50.0
    ecutrho      = 400.0
    occupations  = 'smearing'
    smearing     = 'cold'
    degauss      = 0.01
/
&ELECTRONS
    conv_thr     = 1.0d-8
    mixing_beta  = 0.3
/
ATOMIC_SPECIES
  Cu  63.546   Cu.pbe-dn-rrkjus_psl.1.0.0.UPF

CELL_PARAMETERS {angstrom}
  2.556  0.000  0.000
  1.278  2.213  0.000
  0.000  0.000  20.000

ATOMIC_POSITIONS {angstrom}
  Cu  0.000  0.000  0.000
  Cu  1.278  0.738  2.087
  Cu  2.556  1.475  4.174

K_POINTS {automatic}
  6 6 1  0 0 0
```

**File: `scf_co.in`** (isolated CO in same cell)

```
&CONTROL
    calculation  = 'scf'
    prefix       = 'co_mol'
    outdir       = './tmp_co'
    pseudo_dir   = './pseudo'
    tprnfor      = .true.
/
&SYSTEM
    ibrav        = 0
    nat          = 2
    ntyp         = 2
    ecutwfc      = 50.0
    ecutrho      = 400.0
    occupations  = 'smearing'
    smearing     = 'cold'
    degauss      = 0.01
/
&ELECTRONS
    conv_thr     = 1.0d-8
    mixing_beta  = 0.3
/
ATOMIC_SPECIES
  C   12.011   C.pbe-n-rrkjus_psl.1.0.0.UPF
  O   15.999   O.pbe-n-rrkjus_psl.1.0.0.UPF

CELL_PARAMETERS {angstrom}
  2.556  0.000  0.000
  1.278  2.213  0.000
  0.000  0.000  20.000

ATOMIC_POSITIONS {angstrom}
  C   0.000  0.000  6.100
  O   0.000  0.000  7.243

K_POINTS {automatic}
  6 6 1  0 0 0
```

**Run all three:**
```bash
pw.x < scf_full.in > scf_full.out
pw.x < scf_slab.in > scf_slab.out
pw.x < scf_co.in > scf_co.out
```

#### Step 2: Extract Charge Densities with pp.x

Extract 3D charge density as cube files from all three calculations.

**File: `pp_full.in`**

```
&INPUTPP
    prefix   = 'cu_co'
    outdir   = './tmp_full'
    filplot  = 'charge_full.dat'
    plot_num = 0
/
&PLOT
    nfile       = 1
    filepp(1)   = 'charge_full.dat'
    weight(1)   = 1.0
    iflag       = 3
    output_format = 6
    fileout     = 'charge_full.cube'
/
```

**File: `pp_slab.in`**

```
&INPUTPP
    prefix   = 'cu_slab'
    outdir   = './tmp_slab'
    filplot  = 'charge_slab.dat'
    plot_num = 0
/
&PLOT
    nfile       = 1
    filepp(1)   = 'charge_slab.dat'
    weight(1)   = 1.0
    iflag       = 3
    output_format = 6
    fileout     = 'charge_slab.cube'
/
```

**File: `pp_co.in`**

```
&INPUTPP
    prefix   = 'co_mol'
    outdir   = './tmp_co'
    filplot  = 'charge_co.dat'
    plot_num = 0
/
&PLOT
    nfile       = 1
    filepp(1)   = 'charge_co.dat'
    weight(1)   = 1.0
    iflag       = 3
    output_format = 6
    fileout     = 'charge_co.cube'
/
```

**Run:**
```bash
pp.x < pp_full.in > pp_full.out
pp.x < pp_slab.in > pp_slab.out
pp.x < pp_co.in > pp_co.out
```

#### Step 2b: Alternative -- pp.x Built-in Subtraction

pp.x can combine multiple charge density files with weights directly:

**File: `pp_diff_direct.in`**

```
&INPUTPP
/
&PLOT
    nfile       = 3
    filepp(1)   = 'charge_full.dat'
    weight(1)   = 1.0
    filepp(2)   = 'charge_slab.dat'
    weight(2)   = -1.0
    filepp(3)   = 'charge_co.dat'
    weight(3)   = -1.0
    iflag       = 3
    output_format = 6
    fileout     = 'charge_diff.cube'
/
```

```bash
pp.x < pp_diff_direct.in > pp_diff_direct.out
```

This directly produces the CDD cube file without Python.

#### Step 3: Python -- Compute CDD and Visualize

```python
#!/usr/bin/env python3
"""
Compute charge density difference: rho_AB - rho_A - rho_B
from Gaussian cube files, then generate 2D slices and isosurface data.
"""
import numpy as np
import matplotlib.pyplot as plt
from ase.io.cube import read_cube_data, write_cube


def compute_cdd(cube_full, cube_fragments, output_cube='cdd.cube'):
    """
    Compute charge density difference and write result as cube file.

    Parameters
    ----------
    cube_full : str
        Cube file for the full (combined) system.
    cube_fragments : list of str
        Cube files for isolated fragments (surface, molecule, etc.).
    output_cube : str
        Output cube file with delta-rho.

    Returns
    -------
    diff : np.ndarray
        3D difference charge density array.
    atoms : ase.Atoms
        Atoms object from the full system.
    """
    data_full, atoms_full = read_cube_data(cube_full)
    diff = data_full.copy()

    for frag_cube in cube_fragments:
        data_frag, _ = read_cube_data(frag_cube)
        if data_frag.shape != diff.shape:
            raise ValueError(
                f"Grid mismatch: {cube_full} has shape {diff.shape}, "
                f"but {frag_cube} has shape {data_frag.shape}. "
                f"Ensure same ecutwfc/ecutrho and cell in all calculations."
            )
        diff -= data_frag

    with open(output_cube, 'w') as f:
        write_cube(f, atoms_full, diff)
    print(f"Wrote CDD cube file: {output_cube}")

    # Print statistics
    cell_vol = atoms_full.get_volume()  # Angstrom^3
    total_grid = diff.shape[0] * diff.shape[1] * diff.shape[2]
    dV = cell_vol / total_grid
    q_gain = np.sum(diff[diff > 0]) * dV
    q_loss = np.sum(diff[diff < 0]) * dV
    print(f"Charge accumulation (positive):  {q_gain:+.4f} e")
    print(f"Charge depletion (negative):     {q_loss:+.4f} e")
    print(f"Net charge (should be ~0):       {q_gain + q_loss:+.6f} e")

    return diff, atoms_full


def plot_cdd_2d_slice(diff, atoms, axis=2, slice_frac=None,
                      output_png='cdd_slice.png', isoval=None):
    """
    Plot 2D slice of the charge density difference.

    Parameters
    ----------
    diff : np.ndarray
        3D CDD array.
    atoms : ase.Atoms
        Atoms from the full system.
    axis : int
        Axis perpendicular to slice (0=x, 1=y, 2=z).
    slice_frac : float or None
        Fractional position along axis (0 to 1). None = midpoint.
    output_png : str
        Output image filename.
    isoval : float or None
        If set, draw +/- isoval contour lines.
    """
    cell = atoms.get_cell()

    if slice_frac is None:
        slice_index = diff.shape[axis] // 2
    else:
        slice_index = int(slice_frac * diff.shape[axis])

    if axis == 0:
        sl = diff[slice_index, :, :]
    elif axis == 1:
        sl = diff[:, slice_index, :]
    else:
        sl = diff[:, :, slice_index]

    axes_2d = [i for i in range(3) if i != axis]
    L1 = np.linalg.norm(cell[axes_2d[0]])
    L2 = np.linalg.norm(cell[axes_2d[1]])

    x = np.linspace(0, L1, sl.shape[0])
    y = np.linspace(0, L2, sl.shape[1])

    vmax = np.max(np.abs(sl)) * 0.8
    if vmax < 1e-10:
        vmax = 0.01

    fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.contourf(x, y, sl.T,
                      levels=np.linspace(-vmax, vmax, 51),
                      cmap='bwr', extend='both')
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label(r'$\Delta\rho$ (e/bohr$^3$)')

    if isoval is not None:
        ax.contour(x, y, sl.T, levels=[-isoval], colors='blue',
                   linewidths=1.0, linestyles='--')
        ax.contour(x, y, sl.T, levels=[isoval], colors='red',
                   linewidths=1.0, linestyles='-')

    # Overlay atom positions near the slice plane
    positions = atoms.get_positions()
    frac_coords = atoms.get_scaled_positions()
    slice_frac_val = slice_index / diff.shape[axis]
    tol = 0.05
    symbols = atoms.get_chemical_symbols()
    for i, pos in enumerate(positions):
        f = frac_coords[i]
        if abs(f[axis] - slice_frac_val) < tol or abs(f[axis] - slice_frac_val - 1) < tol:
            ax.plot(pos[axes_2d[0]], pos[axes_2d[1]], 'ko', markersize=8)
            ax.annotate(symbols[i], (pos[axes_2d[0]], pos[axes_2d[1]]),
                        textcoords="offset points", xytext=(5, 5),
                        fontsize=11, fontweight='bold')

    ax.set_xlabel(r'Distance ($\mathrm{\AA}$)')
    ax.set_ylabel(r'Distance ($\mathrm{\AA}$)')
    ax.set_title(r'Charge Density Difference ($\Delta\rho$)')
    ax.set_aspect('equal')
    plt.tight_layout()
    plt.savefig(output_png, dpi=200)
    plt.close()
    print(f"Saved: {output_png}")


def plot_cdd_planar_average(diff, atoms, axis=2,
                             output_png='cdd_planar_avg.png'):
    """
    Plot the planar-averaged CDD along a specified axis.
    Useful for slab/interface systems to see where charge accumulates or depletes.

    Parameters
    ----------
    diff : np.ndarray
        3D CDD array.
    atoms : ase.Atoms
        Atoms from the full system.
    axis : int
        Averaging direction (usually 2 for z-axis in slab).
    output_png : str
        Output image.
    """
    cell = atoms.get_cell()
    length = np.linalg.norm(cell[axis])
    avg_axes = tuple(i for i in range(3) if i != axis)
    planar_avg = np.mean(diff, axis=avg_axes)
    z = np.linspace(0, length, len(planar_avg), endpoint=False)

    # Cumulative integral (charge transfer function)
    dz = length / len(planar_avg)
    cell_vol = atoms.get_volume()
    in_plane_area = cell_vol / length
    # Convert from e/bohr^3 to e/Angstrom by multiplying by in_plane_area
    delta_q = np.cumsum(planar_avg) * dz * in_plane_area

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

    # Planar average
    ax1.plot(z, planar_avg, 'b-', linewidth=1.5)
    ax1.fill_between(z, 0, planar_avg,
                      where=planar_avg > 0, alpha=0.2, color='red',
                      label='Accumulation')
    ax1.fill_between(z, 0, planar_avg,
                      where=planar_avg < 0, alpha=0.2, color='blue',
                      label='Depletion')
    ax1.axhline(y=0, color='black', linewidth=0.5)
    ax1.set_ylabel(r'$\langle\Delta\rho\rangle$ (e/bohr$^3$)')
    ax1.set_title('Planar-Averaged Charge Density Difference')
    ax1.legend(loc='upper right')
    ax1.grid(True, alpha=0.2)

    # Mark atom positions along z
    positions = atoms.get_positions()
    symbols = atoms.get_chemical_symbols()
    for i, pos in enumerate(positions):
        z_pos = pos[axis]
        ax1.axvline(x=z_pos, color='gray', linestyle=':', alpha=0.4)
        ax1.annotate(symbols[i], (z_pos, ax1.get_ylim()[1] * 0.9),
                     fontsize=9, ha='center', color='gray')

    # Cumulative charge transfer
    ax2.plot(z, delta_q, 'r-', linewidth=1.5)
    ax2.axhline(y=0, color='black', linewidth=0.5)
    ax2.set_xlabel(r'z ($\mathrm{\AA}$)')
    ax2.set_ylabel(r'$\Delta Q(z)$ (e)')
    ax2.set_title('Cumulative Charge Transfer')
    ax2.grid(True, alpha=0.2)

    for i, pos in enumerate(positions):
        z_pos = pos[axis]
        ax2.axvline(x=z_pos, color='gray', linestyle=':', alpha=0.4)

    plt.tight_layout()
    plt.savefig(output_png, dpi=200)
    plt.close()
    print(f"Saved: {output_png}")


# ----- Run -----
diff, atoms = compute_cdd(
    cube_full='charge_full.cube',
    cube_fragments=['charge_slab.cube', 'charge_co.cube'],
    output_cube='cdd.cube'
)

plot_cdd_2d_slice(diff, atoms, axis=1, slice_frac=0.5,
                  output_png='cdd_slice_y.png', isoval=0.005)

plot_cdd_planar_average(diff, atoms, axis=2,
                         output_png='cdd_planar_avg.png')
```

#### Step 4: Spin Density Difference (QE)

For magnetic systems, extract the spin density (rho_up - rho_down) using `plot_num=6`.

**File: `pp_spin.in`**

```
&INPUTPP
    prefix   = 'cu_co'
    outdir   = './tmp_full'
    filplot  = 'spin_density.dat'
    plot_num = 6
/
&PLOT
    nfile       = 1
    filepp(1)   = 'spin_density.dat'
    weight(1)   = 1.0
    iflag       = 3
    output_format = 6
    fileout     = 'spin_density.cube'
/
```

```bash
pp.x < pp_spin.in > pp_spin.out
```

For spin density *difference* (how magnetization changes upon adsorption), compute spin densities for all three systems and subtract:

```python
#!/usr/bin/env python3
"""
Compute spin density difference:
  delta_m = m_AB - m_A - m_B
where m = rho_up - rho_down.
"""
import numpy as np
from ase.io.cube import read_cube_data, write_cube


def spin_density_difference(spin_full, spin_fragments,
                             output_cube='spin_diff.cube'):
    """
    Compute spin density difference.

    Parameters
    ----------
    spin_full : str
        Cube file of spin density (plot_num=6) for full system.
    spin_fragments : list of str
        Cube files of spin density for isolated fragments.
    output_cube : str
        Output cube file.
    """
    data_full, atoms_full = read_cube_data(spin_full)
    diff = data_full.copy()

    for frag in spin_fragments:
        data_frag, _ = read_cube_data(frag)
        diff -= data_frag

    with open(output_cube, 'w') as f:
        write_cube(f, atoms_full, diff)

    cell_vol = atoms_full.get_volume()
    total_grid = diff.shape[0] * diff.shape[1] * diff.shape[2]
    dV = cell_vol / total_grid

    delta_m_total = np.sum(diff) * dV
    print(f"Total spin density difference: {delta_m_total:+.4f} mu_B")
    print(f"Wrote: {output_cube}")


# ----- Run -----
spin_density_difference(
    'spin_full.cube',
    ['spin_slab.cube', 'spin_co.cube'],
    'spin_diff.cube'
)
```

### VASP Workflow: Charge Density Difference

#### Step 1: Three VASP Calculations

Run three VASP SCF calculations with the **same ENCUT, same cell, same k-points**:

1. Full system (surface + adsorbate) -- produces `CHGCAR`
2. Bare surface -- produces `CHGCAR`
3. Isolated adsorbate in same cell -- produces `CHGCAR`

Rename the output CHGCAR files:
```bash
cp full_system/CHGCAR  CHGCAR_full
cp bare_surface/CHGCAR CHGCAR_slab
cp adsorbate/CHGCAR    CHGCAR_mol
```

#### Step 2a: VASPKIT 314 (Interactive)

```bash
vaspkit -task 314
# Follow prompts to select CHGCAR files for subtraction
```

#### Step 2b: CHGCAR Subtraction with pymatgen

```python
#!/usr/bin/env python3
"""
Compute charge density difference from VASP CHGCAR files using pymatgen.
rho_diff = rho_AB - rho_A - rho_B
"""
import numpy as np
import matplotlib.pyplot as plt
from pymatgen.io.vasp import Chgcar
from pymatgen.core import Structure


def chgcar_difference(chgcar_full, chgcar_fragments,
                       output_chgcar='CHGCAR_diff',
                       output_png='cdd_vasp.png'):
    """
    Subtract fragment charge densities from full system.

    Parameters
    ----------
    chgcar_full : str
        Path to CHGCAR of the full system.
    chgcar_fragments : list of str
        Paths to CHGCARs of isolated fragments.
    output_chgcar : str
        Output CHGCAR with difference density.
    output_png : str
        2D slice visualization.
    """
    chg_full = Chgcar.from_file(chgcar_full)
    structure = chg_full.structure
    data_full = chg_full.data['total'].copy()

    for frag_path in chgcar_fragments:
        chg_frag = Chgcar.from_file(frag_path)
        data_frag = chg_frag.data['total']
        if data_frag.shape != data_full.shape:
            raise ValueError(
                f"Grid mismatch: {chgcar_full} has {data_full.shape}, "
                f"{frag_path} has {data_frag.shape}. "
                f"Use same ENCUT and cell for all calculations."
            )
        data_full -= data_frag

    # VASP stores charge * volume; normalize to e/Angstrom^3
    vol = structure.volume
    diff_density = data_full / vol  # e/Angstrom^3

    # Write output CHGCAR
    chg_diff = chg_full.copy()
    chg_diff.data['total'] = data_full
    chg_diff.write_file(output_chgcar)
    print(f"Wrote: {output_chgcar}")

    # Statistics
    ngrid = data_full.size
    dV = vol / ngrid
    diff_norm = data_full / vol  # e/Angstrom^3
    q_gain = np.sum(diff_norm[diff_norm > 0]) * dV
    q_loss = np.sum(diff_norm[diff_norm < 0]) * dV
    print(f"Charge accumulation: {q_gain:+.4f} e")
    print(f"Charge depletion:    {q_loss:+.4f} e")
    print(f"Net:                 {q_gain + q_loss:+.6f} e")

    # Plot 2D slice along z at adsorbate height
    lattice = structure.lattice
    nz = diff_density.shape[2]

    # Find z-index near the adsorbate (use midpoint of slab-vacuum region)
    z_frac = 0.5
    z_idx = int(z_frac * nz)

    sl = diff_density[:, :, z_idx]
    nx, ny = sl.shape
    x = np.linspace(0, lattice.a, nx)
    y = np.linspace(0, lattice.b, ny)

    vmax = np.max(np.abs(sl)) * 0.8
    if vmax < 1e-10:
        vmax = 0.01

    fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.contourf(x, y, sl.T,
                      levels=np.linspace(-vmax, vmax, 51),
                      cmap='bwr', extend='both')
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label(r'$\Delta\rho$ (e/$\mathrm{\AA}^3$)')
    ax.set_xlabel(r'x ($\mathrm{\AA}$)')
    ax.set_ylabel(r'y ($\mathrm{\AA}$)')
    ax.set_title(r'VASP Charge Density Difference ($\Delta\rho$)')
    ax.set_aspect('equal')
    plt.tight_layout()
    plt.savefig(output_png, dpi=200)
    plt.close()
    print(f"Saved: {output_png}")


# ----- Run -----
chgcar_difference(
    chgcar_full='CHGCAR_full',
    chgcar_fragments=['CHGCAR_slab', 'CHGCAR_mol'],
    output_chgcar='CHGCAR_diff',
    output_png='cdd_vasp.png'
)
```

#### Step 3: VASP Planar-Averaged CDD

```python
#!/usr/bin/env python3
"""
Planar-averaged charge density difference from VASP CHGCAR.
Averages delta-rho over x-y planes and plots along z.
"""
import numpy as np
import matplotlib.pyplot as plt
from pymatgen.io.vasp import Chgcar


def vasp_cdd_planar_average(chgcar_full, chgcar_fragments,
                              output_png='cdd_vasp_planar.png'):
    """
    Compute and plot planar-averaged CDD from VASP CHGCARs.
    """
    chg_full = Chgcar.from_file(chgcar_full)
    data = chg_full.data['total'].copy()
    vol = chg_full.structure.volume

    for frag_path in chgcar_fragments:
        chg_frag = Chgcar.from_file(frag_path)
        data -= chg_frag.data['total']

    # Normalize to e/Angstrom^3
    diff = data / vol

    # Planar average along z
    planar_avg = np.mean(diff, axis=(0, 1))
    c_length = chg_full.structure.lattice.c
    z = np.linspace(0, c_length, len(planar_avg), endpoint=False)

    # Cumulative charge transfer
    dz = c_length / len(planar_avg)
    area = vol / c_length
    delta_q = np.cumsum(planar_avg) * dz * area

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

    ax1.plot(z, planar_avg, 'b-', linewidth=1.5)
    ax1.fill_between(z, 0, planar_avg,
                      where=planar_avg > 0, alpha=0.2, color='red',
                      label='Accumulation')
    ax1.fill_between(z, 0, planar_avg,
                      where=planar_avg < 0, alpha=0.2, color='blue',
                      label='Depletion')
    ax1.axhline(y=0, color='black', linewidth=0.5)
    ax1.set_ylabel(r'$\langle\Delta\rho\rangle$ (e/$\mathrm{\AA}^3$)')
    ax1.set_title('Planar-Averaged CDD (VASP)')
    ax1.legend()
    ax1.grid(True, alpha=0.2)

    ax2.plot(z, delta_q, 'r-', linewidth=1.5)
    ax2.axhline(y=0, color='black', linewidth=0.5)
    ax2.set_xlabel(r'z ($\mathrm{\AA}$)')
    ax2.set_ylabel(r'$\Delta Q(z)$ (e)')
    ax2.set_title('Cumulative Charge Transfer')
    ax2.grid(True, alpha=0.2)

    plt.tight_layout()
    plt.savefig(output_png, dpi=200)
    plt.close()
    print(f"Saved: {output_png}")


# ----- Run -----
vasp_cdd_planar_average(
    'CHGCAR_full',
    ['CHGCAR_slab', 'CHGCAR_mol'],
    'cdd_vasp_planar.png'
)
```

#### Step 4: 3D Isosurface Data (VESTA-Ready)

```python
#!/usr/bin/env python3
"""
Convert CDD cube file to VESTA-compatible format and print
suggested isosurface values.
"""
import numpy as np
from ase.io.cube import read_cube_data


def suggest_isosurface(cube_file):
    """
    Analyze CDD cube file and suggest isosurface values for visualization.
    """
    data, atoms = read_cube_data(cube_file)
    data_abs = np.abs(data)

    percentiles = [90, 95, 99, 99.5]
    print("Isosurface value suggestions (e/bohr^3):")
    print("-" * 45)
    for p in percentiles:
        val = np.percentile(data_abs, p)
        print(f"  {p:5.1f}th percentile: {val:.6f}")

    print(f"\n  Max positive:  {np.max(data):+.6f}")
    print(f"  Max negative:  {np.min(data):+.6f}")
    print(f"  Mean absolute: {np.mean(data_abs):.6f}")

    # Common choice: 0.002 -- 0.01 e/bohr^3 for adsorption CDD
    suggested = np.percentile(data_abs, 97)
    print(f"\n  Suggested isosurface: +/- {suggested:.5f} e/bohr^3")
    print("  (Use positive for accumulation [red], negative for depletion [blue])")
    print(f"\n  Cube file '{cube_file}' can be opened directly in VESTA.")


# ----- Run -----
suggest_isosurface('cdd.cube')
```

## Key Parameters

### QE pp.x Settings

| Parameter | Value | Purpose |
|-----------|-------|---------|
| `plot_num` | `0` | Total charge density |
| `plot_num` | `6` | Spin polarization (rho_up - rho_down) |
| `iflag` | `3` | 3D output (for cube files) |
| `output_format` | `6` | Gaussian cube format |
| `weight` | `+1.0` / `-1.0` | Add/subtract charge density files |

### Critical Consistency Requirements

| Parameter | Requirement |
|-----------|-------------|
| Cell (lattice vectors) | **Identical** in all three calculations |
| `ecutwfc` | **Identical** -- determines FFT grid |
| `ecutrho` | **Identical** -- determines charge density grid |
| K-points | **Identical** mesh for all calculations |
| Pseudopotentials | Same PP files for same elements |
| Atom positions | Fragment atoms at **same positions** as in the full system |

### VASP Settings

| Parameter | Value | Purpose |
|-----------|-------|---------|
| `ENCUT` | Same for all three calculations | Determines FFT grid |
| `LCHARG` | `.TRUE.` | Write CHGCAR |
| `NGX, NGY, NGZ` | Same (or let VASP set them consistently via ENCUT) | Grid dimensions |
| `PREC` | `Accurate` | Consistent grid across runs |

### Isosurface Value Guidelines

| System Type | Typical Isosurface (e/bohr^3) | Notes |
|-------------|-------------------------------|-------|
| Strong chemisorption | 0.005 -- 0.02 | Metal-adsorbate bonds |
| Weak physisorption | 0.0005 -- 0.002 | vdW interactions |
| Interface charge transfer | 0.002 -- 0.01 | Heterostructures |
| Spin density | 0.001 -- 0.01 | Magnetic redistribution |

## Interpreting Results

### 2D Slice (bwr colormap)

- **Red regions (positive delta-rho)**: Charge accumulation upon bonding. Electrons flow into these regions when fragments combine.
- **Blue regions (negative delta-rho)**: Charge depletion. Electrons leave these regions upon bonding.
- **Accumulation between two atoms**: Covalent bond formation.
- **Accumulation on one atom, depletion on the other**: Ionic/charge-transfer character.
- **Accumulation at the interface with symmetric depletion on both sides**: Typical interface dipole.

### Planar Average

- **Peaks in planar-averaged CDD**: Indicate z-positions where charge accumulates or depletes the most.
- **Cumulative charge transfer Delta-Q(z)**: Shows net charge transferred from one side to the other. The maximum of Delta-Q indicates the magnitude of charge transfer.
- **Sign change in Delta-Q**: Locates the effective charge-transfer boundary.

### Spin Density Difference

- **Positive spin density difference**: Induced spin-up polarization.
- **Negative spin density difference**: Induced spin-down polarization or quenched magnetism.
- Compare to total spin density to distinguish induced from quenched magnetization.

### Quantitative Metrics

- **Integrated charge transfer**: Integrate the positive part of delta-rho to get the total charge rearranged. Typical values: 0.1--0.5 e for chemisorption, <0.05 e for physisorption.
- **Dipole moment from CDD**: p = integral(z * delta-rho * dV). Relates to work function changes.

## Common Issues

| Issue | Cause | Solution |
|-------|-------|----------|
| Grid shape mismatch between cube files | Different `ecutwfc`/`ecutrho` or cell | Use **identical** cell, cutoffs, and k-points in all calculations |
| CDD shows huge values at atom cores | PAW/USPP pseudocharge artifacts | Use `plot_num=21` (PAW all-electron) or focus on interstitial regions |
| Net charge not zero after subtraction | Numerical noise from different SCF convergence | Tighten `conv_thr` to 1.0d-10; net should be < 0.01 e |
| CDD looks noisy | Insufficient k-points or cutoff | Converge k-points and ecutwfc/ecutrho for the CDD quantity specifically |
| Fragment calculation does not converge | Isolated molecule in a large cell with metallic smearing | Use `occupations='fixed'` for molecules if no partial occupations needed |
| pp.x crash with "grids not compatible" | Different outdir or prefix not matching | Verify prefix and outdir in each pp.x input match the corresponding SCF |
| VASP CHGCAR files have different NGX/NGY/NGZ | Different ENCUT or PREC | Set explicit NGX, NGY, NGZ or use same ENCUT and PREC=Accurate |
| Artifacts at cell boundaries | Periodic boundary conditions wrap around | Use a large enough vacuum region (> 12 Angstrom for slab calculations) |
