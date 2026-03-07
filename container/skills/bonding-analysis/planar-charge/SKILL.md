# Planar-Average and Linear-Average Charge Density

Compute planar-averaged and macroscopic-averaged charge density, potential, and charge redistribution profiles along a specified direction. Essential for slab, surface, and interface analysis.

## When to Use

- Analyze charge redistribution at heterostructure interfaces (e.g., metal/oxide, semiconductor junctions).
- Compute work functions from the electrostatic potential profile across a slab.
- Determine band alignment and interface dipoles in heterostructures.
- Visualize charge density or potential variation along z in slab calculations.
- Extract line profiles of charge density through specific bonds or structural features.
- Compute macroscopic averages to smooth out atomic-scale oscillations.

## Method Selection

| Method | Code | Best For | Notes |
|--------|------|----------|-------|
| pp.x + average.x | QE | Planar/macroscopic average of any pp.x quantity | Native QE tool chain |
| pp.x iflag=1 | QE | Line profiles through specific paths | Direct 1D extraction |
| Cube file + Python | QE | Flexible custom averaging and plotting | Full control over averaging windows |
| CHGCAR parsing (pymatgen) | VASP | VASP slab/interface analysis | Reads CHGCAR or LOCPOT |
| VASPKIT 315 | VASP | Quick planar-average charge density | Automated parsing |
| VASPKIT 316 | VASP | Macroscopic-average charge density | Applies convolution filter |
| VASPKIT 317 | VASP | Planar-average potential | From LOCPOT |
| VASPKIT 318 | VASP | Macroscopic-average potential | For band alignment |

## Prerequisites

- Quantum ESPRESSO 7.5 (`pw.x`, `pp.x`, `average.x`)
- VASP (future) for VASP workflows
- Python: `numpy`, `scipy`, `matplotlib`, `ase`, `pymatgen`
- Converged SCF calculation for a slab or interface system
- Sufficient vacuum region (> 12 Angstrom) for slab calculations

## Detailed Steps

### QE Workflow: Planar Average Using average.x

#### Step 1: SCF Calculation for a Slab

**File: `scf_slab.in`** (example: 5-layer Al(111) slab)

```
&CONTROL
    calculation  = 'scf'
    prefix       = 'al_slab'
    outdir       = './tmp'
    pseudo_dir   = './pseudo'
    tprnfor      = .true.
/
&SYSTEM
    ibrav        = 0
    nat          = 5
    ntyp         = 1
    ecutwfc      = 40.0
    ecutrho      = 320.0
    occupations  = 'smearing'
    smearing     = 'cold'
    degauss      = 0.01
/
&ELECTRONS
    conv_thr     = 1.0d-8
    mixing_beta  = 0.3
/
ATOMIC_SPECIES
  Al  26.982   Al.pbe-nl-rrkjus_psl.1.0.0.UPF

CELL_PARAMETERS {angstrom}
  2.863  0.000  0.000
  1.432  2.480  0.000
  0.000  0.000  25.000

ATOMIC_POSITIONS {angstrom}
  Al  0.000  0.000  8.000
  Al  1.432  0.827  10.338
  Al  2.863  1.653  12.676
  Al  1.432  0.827  15.014
  Al  0.000  0.000  17.352

K_POINTS {automatic}
  8 8 1  0 0 0
```

```bash
pw.x < scf_slab.in > scf_slab.out
```

#### Step 2: Extract Charge Density / Potential with pp.x

**File: `pp_charge.in`** (total charge density)

```
&INPUTPP
    prefix   = 'al_slab'
    outdir   = './tmp'
    filplot  = 'charge.dat'
    plot_num = 0
/
&PLOT
    nfile       = 1
    filepp(1)   = 'charge.dat'
    weight(1)   = 1.0
    iflag       = 3
    output_format = 6
    fileout     = 'charge.cube'
/
```

**File: `pp_potential.in`** (total potential -- for work function)

```
&INPUTPP
    prefix   = 'al_slab'
    outdir   = './tmp'
    filplot  = 'potential.dat'
    plot_num = 1
/
&PLOT
    nfile       = 1
    filepp(1)   = 'potential.dat'
    weight(1)   = 1.0
    iflag       = 3
    output_format = 6
    fileout     = 'potential.cube'
/
```

```bash
pp.x < pp_charge.in > pp_charge.out
pp.x < pp_potential.in > pp_potential.out
```

#### Step 3: Planar Average with average.x

`average.x` reads the pp.x intermediate file (`filplot`) and computes the planar average along a chosen direction. It also computes the macroscopic average using a specified window length.

**File: `average.in`**

```
1
charge.dat
1.0d0
3
10.0
```

Line-by-line explanation:
- `1` -- number of files to process
- `charge.dat` -- the pp.x `filplot` output file
- `1.0d0` -- weight for this file
- `3` -- direction for averaging (1=x, 2=y, 3=z)
- `10.0` -- window length for macroscopic average in Bohr (0.0 = no macroscopic averaging). Use one interlayer spacing for a single period; use two nested windows for heterostructures.

```bash
average.x < average.in > average.out
```

**Output file:** `avg.dat` -- columns are: z (Bohr), planar average, macroscopic average.

#### Step 3b: Macroscopic Average with Two Windows (Heterostructures)

For heterostructures with two different periodicities, `average.x` supports a double convolution:

**File: `average_hetero.in`**

```
1
charge.dat
1.0d0
3
8.5
7.2
```

The last two lines specify two macroscopic averaging windows (in Bohr), corresponding to the two interlayer spacings of the two materials in the heterostructure.

```bash
average.x < average_hetero.in > average_hetero.out
```

#### Step 4: Plot average.x Output

```python
#!/usr/bin/env python3
"""
Plot planar-average and macroscopic-average output from QE average.x.
Output file format: z(bohr)  planar_avg  macro_avg
"""
import numpy as np
import matplotlib.pyplot as plt

BOHR_TO_ANG = 0.529177
RY_TO_EV = 13.6057


def plot_average_x_output(avg_file='avg.dat', quantity='charge',
                           output_png='planar_avg.png',
                           e_fermi=None):
    """
    Plot planar and macroscopic averages from average.x output.

    Parameters
    ----------
    avg_file : str
        Output from average.x.
    quantity : str
        'charge' or 'potential'. Affects units and labels.
    output_png : str
        Output image.
    e_fermi : float or None
        Fermi energy in eV (for potential plots, to compute work function).
    """
    data = np.loadtxt(avg_file)

    z_bohr = data[:, 0]
    z_ang = z_bohr * BOHR_TO_ANG
    planar = data[:, 1]
    macro = data[:, 2] if data.shape[1] > 2 else None

    fig, ax = plt.subplots(figsize=(10, 5))

    if quantity == 'charge':
        ax.plot(z_ang, planar, 'b-', linewidth=1.0, alpha=0.6,
                label='Planar average')
        if macro is not None:
            ax.plot(z_ang, macro, 'r-', linewidth=2.0,
                    label='Macroscopic average')
        ax.set_ylabel(r'Charge density (e/bohr$^3$)')
        ax.set_title('Planar-Averaged Charge Density')
    elif quantity == 'potential':
        planar_ev = planar * RY_TO_EV
        ax.plot(z_ang, planar_ev, 'b-', linewidth=1.0, alpha=0.6,
                label='Planar average')
        if macro is not None:
            macro_ev = macro * RY_TO_EV
            ax.plot(z_ang, macro_ev, 'r-', linewidth=2.0,
                    label='Macroscopic average')
            # Work function from vacuum level
            vac_level = np.max(macro_ev)
            ax.axhline(y=vac_level, color='gray', linestyle='--', alpha=0.5)
            ax.annotate(f'Vacuum: {vac_level:.2f} eV',
                        (z_ang[-1] * 0.7, vac_level),
                        fontsize=10, color='gray')
            if e_fermi is not None:
                work_func = vac_level - e_fermi
                ax.axhline(y=e_fermi, color='green', linestyle='--', alpha=0.5)
                ax.annotate(f'E_F: {e_fermi:.2f} eV',
                            (z_ang[-1] * 0.7, e_fermi),
                            fontsize=10, color='green')
                ax.annotate(f'Work function: {work_func:.2f} eV',
                            (z_ang[-1] * 0.3, (vac_level + e_fermi) / 2),
                            fontsize=12, fontweight='bold',
                            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        ax.set_ylabel('Potential (eV)')
        ax.set_title('Planar-Averaged Electrostatic Potential')

    ax.set_xlabel(r'z ($\mathrm{\AA}$)')
    ax.legend(loc='best')
    ax.grid(True, alpha=0.2)
    plt.tight_layout()
    plt.savefig(output_png, dpi=200)
    plt.close()
    print(f"Saved: {output_png}")


# ----- Run -----
plot_average_x_output('avg.dat', quantity='charge',
                       output_png='planar_charge.png')

# For potential / work function:
# plot_average_x_output('avg_potential.dat', quantity='potential',
#                        output_png='planar_potential.png',
#                        e_fermi=4.52)
```

### QE Workflow: Line Profile with pp.x

Extract a 1D charge density profile along any arbitrary path through the structure.

#### Step 5: Line Profile Along a Bond

**File: `pp_line.in`** (line from atom 1 to atom 2)

```
&INPUTPP
    prefix   = 'al_slab'
    outdir   = './tmp'
    filplot  = 'charge.dat'
    plot_num = 0
/
&PLOT
    nfile       = 1
    filepp(1)   = 'charge.dat'
    weight(1)   = 1.0
    iflag       = 1
    output_format = 0
    fileout     = 'charge_line.dat'
    e1(1) = 0.0, e1(2) = 0.0, e1(3) = 1.0
    x0(1) = 0.0, x0(2) = 0.0, x0(3) = 0.0
    nx    = 500
/
```

```bash
pp.x < pp_line.in > pp_line.out
```

#### Step 6: Line Profile Through Arbitrary Points in Python

```python
#!/usr/bin/env python3
"""
Extract a line profile from a 3D cube file along any path
defined by start and end points. More flexible than pp.x iflag=1.
"""
import numpy as np
import matplotlib.pyplot as plt
from ase.io.cube import read_cube_data
from scipy.interpolate import RegularGridInterpolator


def line_profile_from_cube(cube_file, start_frac, end_frac, npoints=500,
                            output_png='line_profile.png'):
    """
    Extract charge density along a line between two fractional coordinates.

    Parameters
    ----------
    cube_file : str
        Cube file from pp.x.
    start_frac : array-like, shape (3,)
        Start point in fractional coordinates.
    end_frac : array-like, shape (3,)
        End point in fractional coordinates.
    npoints : int
        Number of sample points along the line.
    output_png : str
        Output image.
    """
    data, atoms = read_cube_data(cube_file)
    cell = atoms.get_cell()

    # Create grid coordinates in fractional space
    nx, ny, nz = data.shape
    fx = np.linspace(0, 1, nx, endpoint=False)
    fy = np.linspace(0, 1, ny, endpoint=False)
    fz = np.linspace(0, 1, nz, endpoint=False)

    # Build interpolator on fractional grid
    interpolator = RegularGridInterpolator(
        (fx, fy, fz), data, method='linear', bounds_error=False,
        fill_value=None  # extrapolate by nearest
    )

    # Sample points along the line
    start = np.array(start_frac)
    end = np.array(end_frac)
    t = np.linspace(0, 1, npoints)
    frac_points = start[None, :] + t[:, None] * (end - start)[None, :]

    # Wrap into [0, 1)
    frac_points = frac_points % 1.0

    # Interpolate
    values = interpolator(frac_points)

    # Compute real-space distance
    cart_start = start @ cell
    cart_end = end @ cell
    total_dist = np.linalg.norm(cart_end - cart_start)
    distances = t * total_dist

    # Plot
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(distances, values, 'b-', linewidth=1.5)
    ax.set_xlabel(r'Distance ($\mathrm{\AA}$)')
    ax.set_ylabel(r'Charge density (e/bohr$^3$)')
    ax.set_title('Line Profile Through Structure')
    ax.grid(True, alpha=0.2)

    # Annotate start and end
    ax.annotate(f'Start: ({start[0]:.2f}, {start[1]:.2f}, {start[2]:.2f})',
                (distances[0], values[0]), fontsize=9,
                xytext=(10, 10), textcoords='offset points')
    ax.annotate(f'End: ({end[0]:.2f}, {end[1]:.2f}, {end[2]:.2f})',
                (distances[-1], values[-1]), fontsize=9,
                xytext=(-100, 10), textcoords='offset points')

    plt.tight_layout()
    plt.savefig(output_png, dpi=200)
    plt.close()
    print(f"Saved: {output_png}")

    return distances, values


# ----- Run -----
# Profile along z through the slab
line_profile_from_cube(
    'charge.cube',
    start_frac=[0.0, 0.0, 0.0],
    end_frac=[0.0, 0.0, 1.0],
    npoints=500,
    output_png='charge_line_z.png'
)
```

### Python Workflow: Flexible Planar Average from Cube File

```python
#!/usr/bin/env python3
"""
Compute planar-average and macroscopic-average charge density
from a cube file. Equivalent to QE average.x but in pure Python.
Supports charge density, CDD, potential, or any 3D scalar field.
"""
import numpy as np
import matplotlib.pyplot as plt
from ase.io.cube import read_cube_data
from scipy.ndimage import uniform_filter1d


def planar_and_macro_average(cube_file, axis=2, window_angstrom=None,
                              output_png='planar_macro.png',
                              quantity_label='Charge density',
                              unit_label=r'e/bohr$^3$'):
    """
    Compute planar average along a given axis, optionally apply
    macroscopic averaging with a sliding window.

    Parameters
    ----------
    cube_file : str
        Any cube file (charge, CDD, potential, ELF, ...).
    axis : int
        Direction for planar average (0=x, 1=y, 2=z).
    window_angstrom : float or None
        Macroscopic averaging window in Angstrom. None = no macroscopic avg.
        Use one interlayer spacing for bulk, or lattice constant for slabs.
    output_png : str
        Output image.
    quantity_label : str
        Label for the quantity being plotted.
    unit_label : str
        Unit string for the y-axis.
    """
    data, atoms = read_cube_data(cube_file)
    cell = atoms.get_cell()
    length = np.linalg.norm(cell[axis])

    # Planar average: average over the two perpendicular axes
    avg_axes = tuple(i for i in range(3) if i != axis)
    planar_avg = np.mean(data, axis=avg_axes)
    n_pts = len(planar_avg)
    z = np.linspace(0, length, n_pts, endpoint=False)

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(z, planar_avg, 'b-', linewidth=1.0, alpha=0.6,
            label='Planar average')

    if window_angstrom is not None and window_angstrom > 0:
        # Macroscopic average: uniform filter with periodic wrapping
        dz = length / n_pts
        window_pts = int(round(window_angstrom / dz))
        if window_pts < 1:
            window_pts = 1
        macro_avg = uniform_filter1d(planar_avg, size=window_pts,
                                      mode='wrap')
        ax.plot(z, macro_avg, 'r-', linewidth=2.0,
                label=f'Macroscopic avg (window={window_angstrom:.1f} A)')

    # Mark atom positions
    positions = atoms.get_positions()
    symbols = atoms.get_chemical_symbols()
    for i, pos in enumerate(positions):
        ax.axvline(x=pos[axis], color='gray', linestyle=':', alpha=0.3)

    # Add atom labels at top (unique positions only to avoid clutter)
    labeled = set()
    for i, pos in enumerate(positions):
        z_pos = round(pos[axis], 1)
        if z_pos not in labeled:
            labeled.add(z_pos)
            ax.annotate(symbols[i], (pos[axis], ax.get_ylim()[1] if ax.get_ylim()[1] != 0 else 1),
                        fontsize=8, ha='center', color='gray',
                        xytext=(0, 5), textcoords='offset points')

    ax.set_xlabel(r'Distance ($\mathrm{\AA}$)')
    ax.set_ylabel(f'{quantity_label} ({unit_label})')
    ax.set_title(f'Planar-Averaged {quantity_label}')
    ax.legend(loc='best')
    ax.grid(True, alpha=0.2)
    plt.tight_layout()
    plt.savefig(output_png, dpi=200)
    plt.close()
    print(f"Saved: {output_png}")

    return z, planar_avg


def double_macro_average(cube_file, axis=2, window1_angstrom=None,
                          window2_angstrom=None,
                          output_png='double_macro.png'):
    """
    Double macroscopic average for heterostructures with two different
    periodicities. Equivalent to average.x with two window lengths.

    Parameters
    ----------
    window1_angstrom : float
        First averaging window (material 1 period).
    window2_angstrom : float
        Second averaging window (material 2 period).
    """
    data, atoms = read_cube_data(cube_file)
    cell = atoms.get_cell()
    length = np.linalg.norm(cell[axis])
    avg_axes = tuple(i for i in range(3) if i != axis)
    planar_avg = np.mean(data, axis=avg_axes)
    n_pts = len(planar_avg)
    z = np.linspace(0, length, n_pts, endpoint=False)
    dz = length / n_pts

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(z, planar_avg, 'b-', linewidth=0.8, alpha=0.4,
            label='Planar average')

    # First macroscopic average
    w1 = int(round(window1_angstrom / dz)) if window1_angstrom else 1
    macro1 = uniform_filter1d(planar_avg, size=w1, mode='wrap')
    ax.plot(z, macro1, 'g-', linewidth=1.5, alpha=0.7,
            label=f'Macro avg 1 ({window1_angstrom:.1f} A)')

    # Second macroscopic average (applied on top of the first)
    w2 = int(round(window2_angstrom / dz)) if window2_angstrom else 1
    macro2 = uniform_filter1d(macro1, size=w2, mode='wrap')
    ax.plot(z, macro2, 'r-', linewidth=2.0,
            label=f'Double macro avg ({window2_angstrom:.1f} A)')

    ax.set_xlabel(r'Distance ($\mathrm{\AA}$)')
    ax.set_ylabel(r'Charge density (e/bohr$^3$)')
    ax.set_title('Double Macroscopic Average (Heterostructure)')
    ax.legend(loc='best')
    ax.grid(True, alpha=0.2)
    plt.tight_layout()
    plt.savefig(output_png, dpi=200)
    plt.close()
    print(f"Saved: {output_png}")


# ----- Run -----
# Single macroscopic average for a slab
planar_and_macro_average(
    'charge.cube', axis=2, window_angstrom=2.34,
    output_png='planar_macro_charge.png'
)

# Double macroscopic average for a heterostructure
# double_macro_average(
#     'charge_interface.cube', axis=2,
#     window1_angstrom=4.0, window2_angstrom=3.5,
#     output_png='double_macro_interface.png'
# )
```

### VASP Workflow: Planar Average from CHGCAR / LOCPOT

```python
#!/usr/bin/env python3
"""
Planar-averaged charge density and potential from VASP output.
Reads CHGCAR (charge) or LOCPOT (potential) using pymatgen.
Equivalent to VASPKIT tasks 315-318.
"""
import numpy as np
import matplotlib.pyplot as plt
from pymatgen.io.vasp import Chgcar, Locpot
from scipy.ndimage import uniform_filter1d


def vasp_planar_average(vasp_file, file_type='chgcar', axis=2,
                         window_angstrom=None,
                         output_png='vasp_planar.png'):
    """
    Compute planar average from VASP CHGCAR or LOCPOT.

    Parameters
    ----------
    vasp_file : str
        Path to CHGCAR or LOCPOT.
    file_type : str
        'chgcar' or 'locpot'.
    axis : int
        Averaging direction (0=a, 1=b, 2=c).
    window_angstrom : float or None
        Macroscopic averaging window.
    output_png : str
        Output image.
    """
    if file_type == 'chgcar':
        vasp_data = Chgcar.from_file(vasp_file)
        raw_data = vasp_data.data['total']
        vol = vasp_data.structure.volume
        data = raw_data / vol  # Normalize to e/Angstrom^3
        ylabel = r'Charge density (e/$\mathrm{\AA}^3$)'
        title = 'Planar-Averaged Charge Density (VASP)'
    elif file_type == 'locpot':
        vasp_data = Locpot.from_file(vasp_file)
        data = vasp_data.data['total']  # Already in eV
        ylabel = 'Potential (eV)'
        title = 'Planar-Averaged Potential (VASP)'
    else:
        raise ValueError(f"Unknown file_type: {file_type}")

    structure = vasp_data.structure
    lattice = structure.lattice
    lengths = [lattice.a, lattice.b, lattice.c]
    length = lengths[axis]

    # Average over perpendicular axes
    avg_axes = tuple(i for i in range(3) if i != axis)
    planar_avg = np.mean(data, axis=avg_axes)
    n_pts = len(planar_avg)
    z = np.linspace(0, length, n_pts, endpoint=False)

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(z, planar_avg, 'b-', linewidth=1.0, alpha=0.6,
            label='Planar average')

    if window_angstrom is not None:
        dz = length / n_pts
        w = int(round(window_angstrom / dz))
        macro = uniform_filter1d(planar_avg, size=max(w, 1), mode='wrap')
        ax.plot(z, macro, 'r-', linewidth=2.0,
                label=f'Macroscopic avg ({window_angstrom:.1f} A)')

        if file_type == 'locpot':
            vac_level = np.max(macro)
            ax.axhline(y=vac_level, color='gray', linestyle='--', alpha=0.5)
            ax.annotate(f'V_vac = {vac_level:.3f} eV',
                        (z[-1] * 0.6, vac_level),
                        fontsize=10, color='gray')

    # Mark atom z-positions
    for site in structure:
        z_pos = site.coords[axis]
        ax.axvline(x=z_pos, color='gray', linestyle=':', alpha=0.3)

    ax.set_xlabel(r'Distance ($\mathrm{\AA}$)')
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend(loc='best')
    ax.grid(True, alpha=0.2)
    plt.tight_layout()
    plt.savefig(output_png, dpi=200)
    plt.close()
    print(f"Saved: {output_png}")

    return z, planar_avg


def vasp_work_function(locpot_file, e_fermi, axis=2,
                        output_png='work_function.png'):
    """
    Compute work function from VASP LOCPOT.

    Work function = V_vacuum - E_Fermi

    Parameters
    ----------
    locpot_file : str
        Path to LOCPOT file.
    e_fermi : float
        Fermi energy from OUTCAR (in eV).
    axis : int
        Surface normal direction.
    output_png : str
        Output image.
    """
    locpot = Locpot.from_file(locpot_file)
    data = locpot.data['total']
    structure = locpot.structure
    lattice = structure.lattice
    lengths = [lattice.a, lattice.b, lattice.c]
    length = lengths[axis]

    avg_axes = tuple(i for i in range(3) if i != axis)
    planar_avg = np.mean(data, axis=avg_axes)
    n_pts = len(planar_avg)
    z = np.linspace(0, length, n_pts, endpoint=False)

    # Vacuum level: maximum of planar average in vacuum region
    # Typically the flat region far from the slab
    vac_level = np.max(planar_avg)
    work_func = vac_level - e_fermi

    print(f"Vacuum level: {vac_level:.4f} eV")
    print(f"Fermi energy: {e_fermi:.4f} eV")
    print(f"Work function: {work_func:.4f} eV")

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(z, planar_avg, 'b-', linewidth=1.5)
    ax.axhline(y=vac_level, color='gray', linestyle='--',
               label=f'V_vacuum = {vac_level:.2f} eV')
    ax.axhline(y=e_fermi, color='green', linestyle='--',
               label=f'E_Fermi = {e_fermi:.2f} eV')

    # Annotate work function
    mid_z = z[len(z) // 4]
    ax.annotate('', xy=(mid_z, e_fermi), xytext=(mid_z, vac_level),
                arrowprops=dict(arrowstyle='<->', color='red', lw=2))
    ax.text(mid_z + 0.5, (vac_level + e_fermi) / 2,
            f'$\\Phi$ = {work_func:.2f} eV',
            fontsize=14, fontweight='bold', color='red',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

    ax.set_xlabel(r'z ($\mathrm{\AA}$)')
    ax.set_ylabel('Potential (eV)')
    ax.set_title('Work Function from LOCPOT')
    ax.legend(loc='lower right')
    ax.grid(True, alpha=0.2)
    plt.tight_layout()
    plt.savefig(output_png, dpi=200)
    plt.close()
    print(f"Saved: {output_png}")


# ----- Run -----
# Charge density
# vasp_planar_average('CHGCAR', file_type='chgcar', axis=2,
#                      window_angstrom=2.34, output_png='vasp_charge_planar.png')

# Potential and work function
# vasp_planar_average('LOCPOT', file_type='locpot', axis=2,
#                      window_angstrom=2.34, output_png='vasp_potential_planar.png')
# vasp_work_function('LOCPOT', e_fermi=4.52, axis=2,
#                     output_png='work_function.png')
```

### Interface Charge Redistribution Analysis

```python
#!/usr/bin/env python3
"""
Analyze charge redistribution at an interface by comparing
planar-averaged charge density of the interface system with
the sum of isolated components.
"""
import numpy as np
import matplotlib.pyplot as plt
from ase.io.cube import read_cube_data
from scipy.ndimage import uniform_filter1d


def interface_charge_redistribution(cube_interface, cube_components,
                                      axis=2, window_angstrom=None,
                                      output_png='interface_charge.png'):
    """
    Plot the charge redistribution at an interface:
    delta_rho(z) = rho_interface(z) - sum(rho_components(z))

    Parameters
    ----------
    cube_interface : str
        Cube file for the combined interface system.
    cube_components : list of str
        Cube files for isolated components (same cell).
    axis : int
        Interface normal direction.
    window_angstrom : float or None
        Macroscopic averaging window.
    output_png : str
        Output image.
    """
    data_full, atoms = read_cube_data(cube_interface)
    diff = data_full.copy()

    for comp_file in cube_components:
        data_comp, _ = read_cube_data(comp_file)
        diff -= data_comp

    cell = atoms.get_cell()
    length = np.linalg.norm(cell[axis])
    avg_axes = tuple(i for i in range(3) if i != axis)

    planar_full = np.mean(data_full, axis=avg_axes)
    planar_diff = np.mean(diff, axis=avg_axes)
    n_pts = len(planar_diff)
    z = np.linspace(0, length, n_pts, endpoint=False)
    dz = length / n_pts

    # Cumulative charge transfer
    cell_vol = atoms.get_volume()
    in_plane_area = cell_vol / length
    delta_q = np.cumsum(planar_diff) * dz * in_plane_area

    fig, axes_arr = plt.subplots(3, 1, figsize=(10, 12), sharex=True)

    # Panel 1: Total charge profile
    ax1 = axes_arr[0]
    ax1.plot(z, planar_full, 'k-', linewidth=1.0)
    ax1.set_ylabel(r'$\langle\rho\rangle$ (e/bohr$^3$)')
    ax1.set_title('Planar-Averaged Total Charge Density')
    ax1.grid(True, alpha=0.2)

    # Panel 2: Charge redistribution
    ax2 = axes_arr[1]
    ax2.plot(z, planar_diff, 'b-', linewidth=1.0, alpha=0.6,
             label='Planar avg')
    ax2.fill_between(z, 0, planar_diff,
                      where=planar_diff > 0, alpha=0.2, color='red')
    ax2.fill_between(z, 0, planar_diff,
                      where=planar_diff < 0, alpha=0.2, color='blue')

    if window_angstrom is not None:
        w = int(round(window_angstrom / dz))
        macro_diff = uniform_filter1d(planar_diff, size=max(w, 1), mode='wrap')
        ax2.plot(z, macro_diff, 'r-', linewidth=2.0,
                 label=f'Macroscopic avg ({window_angstrom:.1f} A)')

    ax2.axhline(y=0, color='black', linewidth=0.5)
    ax2.set_ylabel(r'$\langle\Delta\rho\rangle$ (e/bohr$^3$)')
    ax2.set_title('Charge Redistribution at Interface')
    ax2.legend()
    ax2.grid(True, alpha=0.2)

    # Panel 3: Cumulative charge transfer
    ax3 = axes_arr[2]
    ax3.plot(z, delta_q, 'r-', linewidth=2.0)
    ax3.axhline(y=0, color='black', linewidth=0.5)
    ax3.set_xlabel(r'z ($\mathrm{\AA}$)')
    ax3.set_ylabel(r'$\Delta Q(z)$ (e)')
    ax3.set_title('Cumulative Charge Transfer')
    ax3.grid(True, alpha=0.2)

    # Transfer amount at the interface
    max_dq = np.max(np.abs(delta_q))
    ax3.annotate(f'Max |Delta Q| = {max_dq:.4f} e',
                 (z[np.argmax(np.abs(delta_q))], delta_q[np.argmax(np.abs(delta_q))]),
                 fontsize=11, fontweight='bold',
                 xytext=(20, 20), textcoords='offset points',
                 arrowprops=dict(arrowstyle='->', color='red'),
                 bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

    # Mark atom positions on all panels
    positions = atoms.get_positions()
    symbols = atoms.get_chemical_symbols()
    for ax_panel in axes_arr:
        for i, pos in enumerate(positions):
            ax_panel.axvline(x=pos[axis], color='gray', linestyle=':',
                             alpha=0.2)

    plt.tight_layout()
    plt.savefig(output_png, dpi=200)
    plt.close()
    print(f"Saved: {output_png}")


# ----- Run -----
# interface_charge_redistribution(
#     'charge_interface.cube',
#     ['charge_slab_A.cube', 'charge_slab_B.cube'],
#     axis=2, window_angstrom=3.0,
#     output_png='interface_redistribution.png'
# )
```

## Key Parameters

### QE average.x Input

| Line | Description | Typical Value |
|------|-------------|---------------|
| 1 | Number of input files | `1` |
| 2 | pp.x filplot filename | `charge.dat` or `potential.dat` |
| 3 | Weight | `1.0d0` |
| 4 | Averaging direction | `3` (z-axis) |
| 5 | Macroscopic average window (Bohr) | interlayer spacing; `0.0` for none |
| 6 (optional) | Second window (Bohr) | for heterostructures |

### pp.x Settings for Planar/Line Profiles

| Parameter | Value | Purpose |
|-----------|-------|---------|
| `plot_num = 0` | Total charge density | Most common |
| `plot_num = 1` | Total potential | For work function |
| `plot_num = 11` | V_bare + V_H (no xc) | For band alignment |
| `iflag = 1` | 1D line profile | Along arbitrary direction |
| `iflag = 2` | 2D planar slice | Contour plot in a plane |
| `iflag = 3` | 3D full grid | For Python post-processing |

### Macroscopic Averaging Windows

| System | Window (Angstrom) | Notes |
|--------|-------------------|-------|
| FCC metal slab | a/sqrt(3) = interlayer spacing | e.g., Al: 2.34 A |
| BCC metal slab | a/2 | e.g., Fe: 1.43 A |
| Si, Ge | a/4 (one bilayer) | e.g., Si: 1.36 A |
| Oxide (perovskite) | a (one unit cell) | e.g., SrTiO3: 3.9 A |
| Heterostructure | Use two windows | One per material period |

### VASPKIT Tasks

| Task | Description | Input | Output |
|------|-------------|-------|--------|
| 315 | Planar-average charge density | CHGCAR | PLANAR_AVERAGE.dat |
| 316 | Macroscopic-average charge density | CHGCAR | MACRO_AVERAGE.dat |
| 317 | Planar-average potential | LOCPOT | PLANAR_AVERAGE.dat |
| 318 | Macroscopic-average potential | LOCPOT | MACRO_AVERAGE.dat |

## Interpreting Results

### Charge Density Profiles

- **Peaks at atom positions**: High charge density at atomic sites (core + valence).
- **Interstitial minima**: Lower charge density between atomic layers. Depth of minima indicates bond ionicity.
- **Asymmetry between surfaces**: Different surface terminations or relaxation.
- **Smooth profile in vacuum**: Should decay to near zero. Non-zero values in vacuum indicate insufficient vacuum or convergence issues.

### Macroscopic Average

- **Removes atomic-scale oscillations**: Reveals the bulk-like step in potential or charge.
- **Flat plateau in bulk-like region**: Material is thick enough to converge.
- **Step at interface**: Band offset or charge transfer.
- **Non-flat in vacuum**: Insufficient vacuum or dipole correction needed.

### Work Function

- **Work function = V_vacuum - E_Fermi**: Read V_vacuum from the flat region of the macroscopic-averaged potential in vacuum.
- **Typical values**: Al ~ 4.2 eV, Cu ~ 4.7 eV, Au ~ 5.1 eV, Pt ~ 5.7 eV.
- **Asymmetric slab**: Two surfaces may have different work functions. Use dipole correction (`dipfield=.true.` in QE or `LDIPOL=.TRUE.` in VASP).

### Interface Charge Transfer

- **Positive Delta-Q**: Net charge transfer from left to right (electrons move right to left).
- **Magnitude of max Delta-Q**: Quantifies the interface charge transfer in electrons per unit cell area.
- **Position of Delta-Q extremum**: Effective position of the charge-transfer plane.

## Common Issues

| Issue | Cause | Solution |
|-------|-------|----------|
| Potential does not converge to flat in vacuum | Vacuum too thin or dipole interaction | Increase vacuum to > 15 A; enable dipole correction |
| Macroscopic average oscillates | Wrong averaging window | Window must equal one complete period of the atomic oscillation |
| average.x reads wrong file | `filplot` name mismatch | Ensure pp.x `filplot` matches the filename in average.x input |
| Work function differs between two surfaces | Slab is polar or has different terminations | Use dipole correction; or use symmetric slab |
| Line profile shows artifacts at cell boundary | Periodic boundary wrapping | Ensure the line does not cross the vacuum-slab boundary unexpectedly |
| VASP CHGCAR has different grid for spin | Spin-polarized CHGCAR has two datasets | Use `data['total']` for total charge; `data['diff']` for spin |
| Planar average units wrong | VASP stores charge*volume in CHGCAR | Divide by cell volume to get e/Angstrom^3 |
| average.x window in wrong units | average.x expects Bohr, not Angstrom | Convert: window_bohr = window_angstrom / 0.529177 |
