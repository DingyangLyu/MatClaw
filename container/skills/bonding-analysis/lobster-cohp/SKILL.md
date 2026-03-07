# LOBSTER COHP Analysis

Crystal Orbital Hamilton Population (COHP) and Integrated COHP (ICOHP) analysis for quantitative chemical bonding characterization using LOBSTER. Projects plane-wave DFT results onto local atomic orbitals to extract bonding and antibonding contributions for each atom pair.

## When to Use

- Quantify bond strength between specific atom pairs (ICOHP as a bond strength indicator).
- Determine whether electronic states are bonding, antibonding, or non-bonding at specific energies.
- Compare bond strengths across different materials, polymorphs, or configurations.
- Analyze bond character changes upon adsorption, doping, or phase transition.
- Identify which orbitals contribute to bonding (orbital-resolved COHP).
- Complement PDOS analysis with pairwise bonding information.

## Method Selection

| Method | Code | Best For | Notes |
|--------|------|----------|-------|
| LOBSTER + QE | QE | Plane-wave DFT from QE output | Requires projectable PAW or NC pseudopotentials |
| LOBSTER + VASP | VASP | Standard VASP PAW output | Most mature LOBSTER pathway |
| LOBSTER + ABINIT | ABINIT | ABINIT users | Also supported |
| lobsterpy (Python) | Any | Automated plotting and analysis | High-level Python API |

## Prerequisites

- LOBSTER program (version 4.1+ recommended)
  - Download from http://www.cohp.de
  - Or install Python interface: `pip install lobsterpy`
- Quantum ESPRESSO 7.5 (`pw.x`) -- for QE workflow
- VASP -- for VASP workflow (future)
- Python: `numpy`, `matplotlib`, `pymatgen`
- For QE: PAW pseudopotentials compatible with LOBSTER (projectable: pslibrary recommended)
- Dense k-point mesh for accurate COHP (denser than for SCF convergence alone)

## Detailed Steps

### Installing LOBSTER

```bash
# Option A: Download binary from cohp.de
cd /opt
wget https://www.cohp.de/lobster/LOBSTER-4.1.0.tar.gz
tar xzf LOBSTER-4.1.0.tar.gz
cp lobster-4.1.0/lobster /usr/local/bin/
chmod +x /usr/local/bin/lobster

# Verify
lobster --version

# Option B: Install Python analysis tools (does not include LOBSTER binary)
pip install lobsterpy

# Option C: Install from conda (if available)
# conda install -c conda-forge lobster
```

### QE Workflow: COHP from Quantum ESPRESSO

#### Step 1: SCF Calculation

Use PAW pseudopotentials from pslibrary (these are projectable for LOBSTER).

**File: `scf.in`** (example: TiO2 rutile)

```
&CONTROL
    calculation  = 'scf'
    prefix       = 'tio2'
    outdir       = './tmp'
    pseudo_dir   = './pseudo'
    tprnfor      = .true.
    disk_io      = 'low'
/
&SYSTEM
    ibrav        = 6
    celldm(1)    = 8.6806
    celldm(3)    = 0.6441
    nat          = 6
    ntyp         = 2
    ecutwfc      = 60.0
    ecutrho      = 600.0
    occupations  = 'smearing'
    smearing     = 'cold'
    degauss      = 0.005
/
&ELECTRONS
    conv_thr     = 1.0d-8
    mixing_beta  = 0.4
/
ATOMIC_SPECIES
  Ti  47.867   Ti.pbe-spn-kjpaw_psl.1.0.0.UPF
  O   15.999   O.pbe-n-kjpaw_psl.1.0.0.UPF

ATOMIC_POSITIONS {crystal}
  Ti  0.0000  0.0000  0.0000
  Ti  0.5000  0.5000  0.5000
  O   0.3053  0.3053  0.0000
  O   0.6947  0.6947  0.0000
  O   0.1947  0.8053  0.5000
  O   0.8053  0.1947  0.5000

K_POINTS {automatic}
  8 8 12  0 0 0
```

```bash
pw.x < scf.in > scf.out
```

#### Step 2: NSCF Calculation with Dense k-mesh and Wavefunction Output

LOBSTER needs the wavefunctions at a dense k-mesh. Run an NSCF calculation that saves all wavefunctions.

**File: `nscf.in`**

```
&CONTROL
    calculation  = 'nscf'
    prefix       = 'tio2'
    outdir       = './tmp'
    pseudo_dir   = './pseudo'
    wf_collect   = .true.
/
&SYSTEM
    ibrav        = 6
    celldm(1)    = 8.6806
    celldm(3)    = 0.6441
    nat          = 6
    ntyp         = 2
    ecutwfc      = 60.0
    ecutrho      = 600.0
    occupations  = 'smearing'
    smearing     = 'cold'
    degauss      = 0.005
    nosym        = .true.
    noinv        = .true.
    nbnd         = 40
/
&ELECTRONS
    conv_thr     = 1.0d-8
/
ATOMIC_SPECIES
  Ti  47.867   Ti.pbe-spn-kjpaw_psl.1.0.0.UPF
  O   15.999   O.pbe-n-kjpaw_psl.1.0.0.UPF

ATOMIC_POSITIONS {crystal}
  Ti  0.0000  0.0000  0.0000
  Ti  0.5000  0.5000  0.5000
  O   0.3053  0.3053  0.0000
  O   0.6947  0.6947  0.0000
  O   0.1947  0.8053  0.5000
  O   0.8053  0.1947  0.5000

K_POINTS {automatic}
  8 8 12  0 0 0
```

Important settings:
- `nosym = .true.` and `noinv = .true.`: LOBSTER needs the full (non-symmetry-reduced) k-mesh.
- `wf_collect = .true.`: Ensures wavefunctions are written in a format LOBSTER can read.
- `nbnd`: Include enough bands (occupied + some unoccupied).

```bash
pw.x < nscf.in > nscf.out
```

#### Step 3: Prepare LOBSTER Input

**File: `lobsterin`**

```
! LOBSTER input for QE
COHPstartEnergy  -25.0
COHPendEnergy     10.0
COHPSteps         3001

! Basis functions for projection
! Must match the PAW data sets; check LOBSTER documentation for your PPs
basisfunctions Ti 3s 3p 3d 4s
basisfunctions O  2s 2p

! Use QE output
useQEaliases

! Optional: specify which bonds to analyze
! cohpbetween atom 1 atom 3   ! Ti1-O3 bond
! cohpbetween atom 1 atom 4   ! Ti1-O4 bond

! Automatically find bonds within cutoff
cohpGenerator from 0.0 to 3.0 type Ti type O

! Save all COHP data
saveProjectionToFile
```

#### Step 4: Run LOBSTER

```bash
# LOBSTER reads from the QE output directory
# It needs access to the tmp/ directory with wavefunctions
lobster
```

LOBSTER produces several output files:
- `COHPCAR.lobster` -- COHP data for all specified bonds
- `ICOHPLIST.lobster` -- Integrated COHP values (bond strengths)
- `COOPCAR.lobster` -- Crystal Orbital Overlap Population
- `CHARGE.lobster` -- Mulliken and Loewdin charges
- `DOSCAR.lobster` -- Projected DOS from LOBSTER basis
- `GROSSPOP.lobster` -- Gross population analysis
- `lobsterout` -- Log file with quality metrics

#### Step 5: Parse and Plot COHP

```python
#!/usr/bin/env python3
"""
Parse and plot COHP from LOBSTER output.
COHPCAR.lobster format:
  - Header with bond information
  - Columns: Energy, average-COHP, COHP(bond1), COHP(bond2), ...
Convention: negative COHP = bonding, positive COHP = antibonding.
Plots show -COHP so bonding appears on the right (positive) side.
"""
import numpy as np
import matplotlib.pyplot as plt
import re


def parse_cohpcar(cohpcar_file='COHPCAR.lobster'):
    """
    Parse COHPCAR.lobster file.

    Returns
    -------
    energy : np.ndarray
        Energy values (eV).
    cohp_data : dict
        {bond_label: np.ndarray of COHP values}
    e_fermi : float
        Fermi energy (if found in file).
    """
    with open(cohpcar_file, 'r') as f:
        lines = f.readlines()

    # First line: number of bonds and Fermi energy
    header_parts = lines[0].split()
    # Format varies; typically first line has number of data points
    # Second line has labels

    # Find the start of data (after header lines)
    data_start = 0
    bond_labels = []

    for i, line in enumerate(lines):
        if line.strip().startswith('No.'):
            # Parse bond labels from header
            # Format: "No.1:Ti1->O3(2.00A)" etc.
            labels_line = line.strip()
            label_pattern = r'No\.\d+:(\S+)'
            bond_labels = re.findall(label_pattern, labels_line)
            data_start = i + 1
            break

    if not bond_labels:
        # Alternative: try to parse from the first few lines
        for i, line in enumerate(lines):
            parts = line.split()
            try:
                float(parts[0])
                data_start = i
                break
            except (ValueError, IndexError):
                continue

    # Read numeric data
    data_lines = []
    for line in lines[data_start:]:
        parts = line.split()
        try:
            vals = [float(p) for p in parts]
            data_lines.append(vals)
        except ValueError:
            continue

    data = np.array(data_lines)
    energy = data[:, 0]

    cohp_data = {}
    # Column 1 is average COHP (if present)
    if data.shape[1] > 2:
        cohp_data['average'] = data[:, 1]
        for i, label in enumerate(bond_labels):
            col_idx = 2 + i * 2 if data.shape[1] > 2 + i * 2 else 2 + i
            if col_idx < data.shape[1]:
                cohp_data[label] = data[:, col_idx]
    elif data.shape[1] == 2:
        cohp_data['COHP'] = data[:, 1]

    return energy, cohp_data


def parse_icohplist(icohplist_file='ICOHPLIST.lobster'):
    """
    Parse ICOHPLIST.lobster for integrated COHP values.

    Returns
    -------
    bonds : list of dict
        Each dict: {'label': str, 'distance': float, 'icohp': float,
                    'atom1': str, 'atom2': str}
    """
    bonds = []
    with open(icohplist_file, 'r') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('No.') or line.startswith('label'):
                continue
            parts = line.split()
            if len(parts) >= 7:
                try:
                    bonds.append({
                        'index': int(parts[0]),
                        'atom1': parts[1],
                        'atom2': parts[2],
                        'distance': float(parts[3]),
                        'label': f"{parts[1]}-{parts[2]}",
                        'icohp': float(parts[-1]),  # Last column is ICOHP
                    })
                except (ValueError, IndexError):
                    continue
    return bonds


def plot_cohp(cohpcar_file='COHPCAR.lobster', e_fermi=0.0,
              bond_indices=None, output_png='cohp.png',
              emin=-15, emax=10):
    """
    Plot -COHP (bonding diagram) for selected bonds.

    Parameters
    ----------
    cohpcar_file : str
        COHPCAR.lobster file.
    e_fermi : float
        Fermi energy in eV (to shift energy axis).
    bond_indices : list of int or None
        Which bonds to plot (0-indexed). None = all.
    output_png : str
        Output image.
    emin, emax : float
        Energy range relative to E_F.
    """
    energy, cohp_data = parse_cohpcar(cohpcar_file)
    energy_shifted = energy - e_fermi

    if bond_indices is not None:
        keys = list(cohp_data.keys())
        selected = {keys[i]: cohp_data[keys[i]] for i in bond_indices
                    if i < len(keys)}
    else:
        selected = cohp_data

    n_bonds = len(selected)
    if n_bonds == 0:
        print("No COHP data to plot.")
        return

    fig, ax = plt.subplots(figsize=(6, 8))
    colors = plt.cm.Set1(np.linspace(0, 1, max(n_bonds, 1)))

    for (label, cohp), color in zip(selected.items(), colors):
        # Plot -COHP: positive = bonding, negative = antibonding
        neg_cohp = -cohp
        ax.plot(neg_cohp, energy_shifted, '-', color=color, linewidth=1.5,
                label=label)
        ax.fill_betweenx(energy_shifted, 0, neg_cohp,
                          where=neg_cohp > 0, alpha=0.1, color=color)
        ax.fill_betweenx(energy_shifted, 0, neg_cohp,
                          where=neg_cohp < 0, alpha=0.1, color='gray')

    ax.axvline(x=0, color='black', linewidth=0.5)
    ax.axhline(y=0, color='gray', linestyle='--', linewidth=0.5)
    ax.set_xlabel(r'$-$COHP (eV)')
    ax.set_ylabel(r'$E - E_F$ (eV)')
    ax.set_ylim(emin, emax)
    ax.set_title('Crystal Orbital Hamilton Population')
    ax.legend(loc='upper right', fontsize=9)
    ax.grid(True, alpha=0.2)

    # Add bonding/antibonding labels
    xlim = ax.get_xlim()
    ax.text(xlim[1] * 0.7, emax * 0.9, 'Bonding',
            fontsize=11, color='green', ha='center', alpha=0.7)
    ax.text(xlim[0] * 0.7, emax * 0.9, 'Antibonding',
            fontsize=11, color='red', ha='center', alpha=0.7)

    plt.tight_layout()
    plt.savefig(output_png, dpi=200)
    plt.close()
    print(f"Saved: {output_png}")


def plot_icohp_bar(icohplist_file='ICOHPLIST.lobster',
                    output_png='icohp_bar.png', max_bonds=20):
    """
    Bar chart of ICOHP values for all bonds (bond strength comparison).

    Parameters
    ----------
    icohplist_file : str
        ICOHPLIST.lobster file.
    output_png : str
        Output image.
    max_bonds : int
        Maximum number of bonds to display.
    """
    bonds = parse_icohplist(icohplist_file)

    if not bonds:
        print("No bonds found in ICOHPLIST.")
        return

    # Sort by ICOHP (most negative = strongest bonding)
    bonds.sort(key=lambda b: b['icohp'])

    # Limit number of bonds
    if len(bonds) > max_bonds:
        bonds = bonds[:max_bonds]

    labels = [f"{b['label']} ({b['distance']:.2f} A)" for b in bonds]
    icohp_values = [b['icohp'] for b in bonds]

    fig, ax = plt.subplots(figsize=(10, max(5, len(bonds) * 0.4)))
    colors = ['#2ca02c' if v < 0 else '#d62728' for v in icohp_values]

    y_pos = range(len(bonds))
    bars = ax.barh(y_pos, [-v for v in icohp_values], color=colors,
                    edgecolor='black', linewidth=0.3)

    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels, fontsize=9)
    ax.set_xlabel(r'$-$ICOHP (eV per bond)')
    ax.set_title('Bond Strengths from ICOHP Analysis')
    ax.axvline(x=0, color='black', linewidth=0.5)
    ax.grid(axis='x', alpha=0.3)

    # Add value labels
    for bar, val in zip(bars, icohp_values):
        w = bar.get_width()
        ax.text(w + 0.02, bar.get_y() + bar.get_height() / 2,
                f'{val:.3f}', va='center', fontsize=8)

    plt.tight_layout()
    plt.savefig(output_png, dpi=200)
    plt.close()
    print(f"Saved: {output_png}")


# ----- Run -----
# Get Fermi energy from QE output
import re
e_fermi = 0.0
try:
    with open('scf.out', 'r') as f:
        for line in f:
            if 'Fermi energy' in line or 'highest occupied' in line:
                match = re.search(r'([\d.-]+)\s*ev', line, re.IGNORECASE)
                if match:
                    e_fermi = float(match.group(1))
                    break
except FileNotFoundError:
    pass

plot_cohp('COHPCAR.lobster', e_fermi=e_fermi,
          output_png='cohp.png', emin=-15, emax=10)

plot_icohp_bar('ICOHPLIST.lobster', output_png='icohp_bar.png')
```

#### Step 6: Orbital-Resolved COHP

```python
#!/usr/bin/env python3
"""
Parse orbital-resolved COHP from LOBSTER output.
When LOBSTER is run with orbital resolution, it produces
separate COHP data for each orbital pair (e.g., Ti-3d / O-2p).
"""
import numpy as np
import matplotlib.pyplot as plt
import glob
import re


def plot_orbital_cohp(cohpcar_file='COHPCAR.lobster', e_fermi=0.0,
                       orb_pairs=None, output_png='cohp_orbital.png',
                       emin=-15, emax=10):
    """
    Plot orbital-resolved COHP.

    Parameters
    ----------
    cohpcar_file : str
        COHPCAR.lobster with orbital resolution.
    e_fermi : float
        Fermi energy.
    orb_pairs : list of str or None
        Orbital pair labels to plot, e.g., ['Ti1(3d)-O3(2p)'].
        None = plot all available.
    output_png : str
        Output image.
    """
    with open(cohpcar_file, 'r') as f:
        lines = f.readlines()

    # Parse header to find bond/orbital labels
    bond_labels = []
    data_start = 0
    for i, line in enumerate(lines):
        if 'No.' in line:
            label_pattern = r'No\.\d+:(\S+)'
            bond_labels = re.findall(label_pattern, line)
            data_start = i + 1
            break
        parts = line.split()
        try:
            float(parts[0])
            data_start = i
            break
        except (ValueError, IndexError):
            continue

    # Read data
    data_lines = []
    for line in lines[data_start:]:
        parts = line.split()
        try:
            data_lines.append([float(p) for p in parts])
        except ValueError:
            continue

    data = np.array(data_lines)
    energy = data[:, 0] - e_fermi

    fig, ax = plt.subplots(figsize=(6, 8))

    # If orbital pairs specified, find matching columns
    n_cols = data.shape[1] - 1  # exclude energy column
    if bond_labels:
        for i, label in enumerate(bond_labels):
            if orb_pairs is not None and label not in orb_pairs:
                continue
            col = i + 1  # +1 for energy column
            if col >= data.shape[1]:
                continue
            neg_cohp = -data[:, col]
            ax.plot(neg_cohp, energy, '-', linewidth=1.2, label=label)
    else:
        # No labels found; plot all columns
        for col in range(1, min(data.shape[1], 6)):
            neg_cohp = -data[:, col]
            ax.plot(neg_cohp, energy, '-', linewidth=1.2,
                    label=f'Column {col}')

    ax.axvline(x=0, color='black', linewidth=0.5)
    ax.axhline(y=0, color='gray', linestyle='--', linewidth=0.5)
    ax.set_xlabel(r'$-$COHP (eV)')
    ax.set_ylabel(r'$E - E_F$ (eV)')
    ax.set_ylim(emin, emax)
    ax.set_title('Orbital-Resolved COHP')
    ax.legend(loc='upper right', fontsize=8)
    ax.grid(True, alpha=0.2)

    plt.tight_layout()
    plt.savefig(output_png, dpi=200)
    plt.close()
    print(f"Saved: {output_png}")


# ----- Run -----
# plot_orbital_cohp('COHPCAR.lobster', e_fermi=e_fermi,
#                    output_png='cohp_orbital.png')
```

### VASP Workflow: COHP from VASP

#### Step 1: VASP Calculation with LOBSTER-Compatible Settings

INCAR additions for LOBSTER compatibility:

```
! INCAR for LOBSTER-compatible VASP run
ISYM   = -1        ! Disable symmetry (LOBSTER needs full k-mesh)
NBANDS = 40        ! Include enough bands
LWAVE  = .TRUE.    ! Write WAVECAR (required by LOBSTER)
PREC   = Accurate  ! High precision
ISPIN  = 1         ! or 2 for spin-polarized
ISMEAR = 0         ! Gaussian smearing
SIGMA  = 0.05      ! Small smearing
```

#### Step 2: LOBSTER Input for VASP

**File: `lobsterin`**

```
! LOBSTER input for VASP
COHPstartEnergy  -25.0
COHPendEnergy     10.0
COHPSteps         3001

! Basis functions (must match VASP PAW datasets)
basisfunctions Ti 3s 3p 3d 4s
basisfunctions O  2s 2p

! Bond analysis
cohpGenerator from 0.0 to 3.0 type Ti type O

! Optional: specific bonds
! cohpbetween atom 1 atom 3

saveProjectionToFile
```

#### Step 3: Run and Analyze

```bash
lobster
```

Then use the same Python scripts from Step 5-6 above to parse and plot the results.

### Using lobsterpy for Automated Analysis

```python
#!/usr/bin/env python3
"""
Use lobsterpy for automated LOBSTER output analysis.
lobsterpy provides high-level functions for parsing and plotting.
"""
try:
    from lobsterpy.cohp.analyze import Analysis
    from lobsterpy.cohp.describe import Description
    from lobsterpy.plotting import PlainCohpPlotter, IcohpDistancePlotter
    LOBSTERPY_AVAILABLE = True
except ImportError:
    LOBSTERPY_AVAILABLE = False
    print("lobsterpy not installed. Install with: pip install lobsterpy")

import matplotlib.pyplot as plt


def lobsterpy_analysis(path='.', output_png='cohp_lobsterpy.png'):
    """
    Automated COHP analysis using lobsterpy.

    Parameters
    ----------
    path : str
        Directory containing LOBSTER output files.
    output_png : str
        Output image.
    """
    if not LOBSTERPY_AVAILABLE:
        print("lobsterpy not available. Falling back to manual parsing.")
        return

    # Automated bonding analysis
    analysis = Analysis(
        path_to_poscar=f"{path}/POSCAR",
        path_to_icohplist=f"{path}/ICOHPLIST.lobster",
        path_to_cohpcar=f"{path}/COHPCAR.lobster",
        path_to_charge=f"{path}/CHARGE.lobster",
        which_bonds="cation-anion",  # or "all"
    )

    # Print summary
    description = Description(analysis)
    print("LOBSTER Bonding Analysis Summary")
    print("=" * 60)
    print(description.text)

    # Plot COHP
    plotter = PlainCohpPlotter()
    plotter.add_cohp_dict(analysis.chemenv.cohp_dict)
    fig = plotter.get_plot()
    fig.savefig(output_png, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"\nSaved: {output_png}")


def icohp_vs_distance(icohplist_file='ICOHPLIST.lobster',
                       output_png='icohp_distance.png'):
    """
    Plot ICOHP vs bond distance to identify trends.
    """
    # Parse ICOHPLIST
    bonds = []
    with open(icohplist_file, 'r') as f:
        for line in f:
            parts = line.split()
            if len(parts) >= 7:
                try:
                    bonds.append({
                        'atom1': parts[1],
                        'atom2': parts[2],
                        'distance': float(parts[3]),
                        'icohp': float(parts[-1]),
                        'label': f"{parts[1]}-{parts[2]}",
                    })
                except (ValueError, IndexError):
                    continue

    if not bonds:
        print("No bond data found.")
        return

    # Group by bond type
    bond_types = {}
    for b in bonds:
        btype = f"{b['atom1'].rstrip('0123456789')}-{b['atom2'].rstrip('0123456789')}"
        if btype not in bond_types:
            bond_types[btype] = {'distances': [], 'icohps': []}
        bond_types[btype]['distances'].append(b['distance'])
        bond_types[btype]['icohps'].append(b['icohp'])

    fig, ax = plt.subplots(figsize=(8, 6))
    colors = plt.cm.Set1(range(len(bond_types)))

    for (btype, data), color in zip(bond_types.items(), colors):
        ax.scatter(data['distances'], [-v for v in data['icohps']],
                   c=[color], s=80, edgecolors='black', linewidths=0.5,
                   label=btype, alpha=0.8)

    ax.set_xlabel(r'Bond distance ($\mathrm{\AA}$)')
    ax.set_ylabel(r'$-$ICOHP (eV per bond)')
    ax.set_title('ICOHP vs Bond Distance')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_png, dpi=200)
    plt.close()
    print(f"Saved: {output_png}")


# ----- Run -----
# lobsterpy_analysis('.', 'cohp_lobsterpy.png')
icohp_vs_distance('ICOHPLIST.lobster', 'icohp_distance.png')
```

### COHP Comparison Across Systems

```python
#!/usr/bin/env python3
"""
Compare COHP of the same bond type across different systems
(e.g., TiO2 rutile vs anatase, or pristine vs doped).
"""
import numpy as np
import matplotlib.pyplot as plt
import re


def compare_cohp(cohp_files, labels, e_fermis, bond_column=1,
                  output_png='cohp_compare.png', emin=-15, emax=10):
    """
    Overlay COHP from multiple systems for comparison.

    Parameters
    ----------
    cohp_files : list of str
        COHPCAR.lobster files from different systems.
    labels : list of str
        Labels for each system.
    e_fermis : list of float
        Fermi energy for each system.
    bond_column : int
        Column index (1-based, excluding energy) for the bond to compare.
    output_png : str
        Output image.
    """
    fig, ax = plt.subplots(figsize=(6, 8))
    colors = plt.cm.tab10(np.linspace(0, 1, len(cohp_files)))

    for cfile, label, ef, color in zip(cohp_files, labels, e_fermis, colors):
        # Parse data
        data_lines = []
        with open(cfile, 'r') as f:
            for line in f:
                parts = line.split()
                try:
                    vals = [float(p) for p in parts]
                    data_lines.append(vals)
                except ValueError:
                    continue

        data = np.array(data_lines)
        energy = data[:, 0] - ef
        col = min(bond_column, data.shape[1] - 1)
        neg_cohp = -data[:, col]

        ax.plot(neg_cohp, energy, '-', color=color, linewidth=1.5,
                label=label)

    ax.axvline(x=0, color='black', linewidth=0.5)
    ax.axhline(y=0, color='gray', linestyle='--', linewidth=0.5)
    ax.set_xlabel(r'$-$COHP (eV)')
    ax.set_ylabel(r'$E - E_F$ (eV)')
    ax.set_ylim(emin, emax)
    ax.set_title('COHP Comparison')
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.2)

    plt.tight_layout()
    plt.savefig(output_png, dpi=200)
    plt.close()
    print(f"Saved: {output_png}")


# ----- Run -----
# compare_cohp(
#     ['rutile/COHPCAR.lobster', 'anatase/COHPCAR.lobster'],
#     ['TiO2 rutile', 'TiO2 anatase'],
#     [6.5, 6.3],  # Fermi energies
#     bond_column=1,
#     output_png='cohp_compare.png'
# )
```

### Quality Assessment

```python
#!/usr/bin/env python3
"""
Parse LOBSTER quality metrics from lobsterout.
"""
import re


def check_lobster_quality(lobsterout_file='lobsterout'):
    """
    Extract and report quality metrics from LOBSTER output.

    Key metrics:
    - Charge spilling: should be < 2%
    - Total charge vs reference: should match
    - Band structure energy vs DFT: should be close
    """
    with open(lobsterout_file, 'r') as f:
        content = f.read()

    print("LOBSTER Quality Assessment")
    print("=" * 50)

    # Charge spilling
    spilling = re.search(r'spillings?\s*.*?(\d+\.\d+)\s*%', content, re.IGNORECASE)
    if spilling:
        spill_val = float(spilling.group(1))
        quality = "GOOD" if spill_val < 2.0 else "WARNING" if spill_val < 5.0 else "POOR"
        print(f"Charge spilling: {spill_val:.2f}% [{quality}]")
        if spill_val > 5.0:
            print("  -> High spilling indicates insufficient basis set.")
            print("     Add more basis functions in lobsterin.")

    # Absolute charge
    abs_charge = re.search(r'abs.*charge\s*.*?(\d+\.\d+)', content, re.IGNORECASE)
    if abs_charge:
        print(f"Absolute total charge: {abs_charge.group(1)}")

    # Total integrated DOS
    int_dos = re.search(r'total\s+integrated\s+DOS\s*.*?(\d+\.\d+)', content, re.IGNORECASE)
    if int_dos:
        print(f"Total integrated DOS: {int_dos.group(1)}")

    # Check for warnings
    warnings = re.findall(r'WARNING.*', content, re.IGNORECASE)
    if warnings:
        print(f"\nWarnings ({len(warnings)}):")
        for w in warnings[:5]:
            print(f"  {w.strip()}")

    # Check for errors
    errors = re.findall(r'ERROR.*', content, re.IGNORECASE)
    if errors:
        print(f"\nErrors ({len(errors)}):")
        for e in errors:
            print(f"  {e.strip()}")

    if not spilling and not abs_charge:
        print("Could not parse quality metrics. Check lobsterout format.")


# ----- Run -----
check_lobster_quality('lobsterout')
```

## Key Parameters

### LOBSTER Input (lobsterin)

| Parameter | Description | Example |
|-----------|-------------|---------|
| `COHPstartEnergy` | Lower energy bound (eV) | `-25.0` |
| `COHPendEnergy` | Upper energy bound (eV) | `10.0` |
| `COHPSteps` | Number of energy points | `3001` |
| `basisfunctions` | Atomic orbitals for projection | `Ti 3s 3p 3d 4s` |
| `cohpGenerator` | Auto-generate bonds by type/distance | `from 0.0 to 3.0 type Ti type O` |
| `cohpbetween` | Specific bond between two atoms | `atom 1 atom 3` |
| `saveProjectionToFile` | Save projection data | (flag, no value) |
| `useQEaliases` | Enable QE compatibility mode | (flag, no value) |

### Basis Set Selection

| Element | Common Basis | Notes |
|---------|-------------|-------|
| s-block (Na, K, Ca) | `ns (n-1)p` | e.g., Na: `2s 2p 3s` |
| p-block (O, N, C) | `ns np` | e.g., O: `2s 2p` |
| 3d metals (Ti, Fe, Co) | `3s 3p 3d 4s` | Include semi-core 3s 3p |
| 4d metals (Zr, Mo) | `4s 4p 4d 5s` | Include semi-core |
| 5d metals (W, Pt) | `5s 5p 5d 6s` | Include semi-core |
| Rare earths (La, Ce) | `5s 5p 4f 5d 6s` | f-electrons important |

### QE Requirements for LOBSTER

| Setting | Requirement | Reason |
|---------|-------------|--------|
| `nosym = .true.` | Mandatory | LOBSTER needs full k-mesh |
| `noinv = .true.` | Mandatory | No time-reversal reduction |
| `wf_collect = .true.` | Mandatory | Save wavefunctions to disk |
| K-points | Dense mesh (>= 8x8x8 for bulk) | COHP resolution |
| `nbnd` | Include unoccupied bands | Need antibonding states |
| Pseudopotentials | PAW from pslibrary | Must be projectable |

### VASP Requirements for LOBSTER

| Setting | Requirement | Reason |
|---------|-------------|--------|
| `ISYM = -1` | Mandatory | Disable symmetry |
| `LWAVE = .TRUE.` | Mandatory | Write WAVECAR |
| `NBANDS` | Include unoccupied | Need antibonding states |
| PAW datasets | Standard VASP PAW | Must be LOBSTER-compatible |

## Interpreting Results

### COHP Plot (-COHP vs Energy)

- **Positive -COHP (right side)**: Bonding states. Electrons in these states stabilize the bond.
- **Negative -COHP (left side)**: Antibonding states. Electrons here weaken the bond.
- **Zero -COHP**: Non-bonding states.
- **Area under -COHP curve below E_F**: Proportional to occupied bonding (or antibonding) character.

### ICOHP Values (Bond Strength Indicator)

| ICOHP Range (eV/bond) | Bond Character | Examples |
|------------------------|---------------|----------|
| < -5.0 | Very strong covalent | C-C in diamond, N-N triple bond |
| -3.0 to -5.0 | Strong covalent | Si-O, Ti-O in oxides |
| -1.0 to -3.0 | Moderate bond | Metal-O in perovskites |
| -0.5 to -1.0 | Weak bond | Hydrogen bonds, vdW (at DFT level) |
| > -0.5 | Very weak / non-bonding | Long-range interactions |

**Important**: ICOHP is negative for bonding interactions. More negative = stronger bond. Convention: plot -ICOHP (positive = stronger) for bar charts.

### Bond Strength Trends

- **ICOHP vs distance**: Shorter bonds typically have more negative ICOHP (stronger).
- **ICOHP across a series**: Compare the same bond type (e.g., Ti-O) in different compounds to rank bond strengths.
- **Orbital contributions**: If Ti-d / O-p COHP dominates, the bond has strong pd-covalent character.

### Quality Indicators

- **Charge spilling < 2%**: Projection is reliable.
- **Charge spilling 2-5%**: Results qualitatively correct but quantitatively less reliable.
- **Charge spilling > 5%**: Basis set is too small; add more orbitals.
- **Total charge consistency**: Integrated charge from LOBSTER should match total electrons from DFT.

### COHP vs COOP vs COBI

| Quantity | Measures | Convention |
|----------|----------|------------|
| COHP | Hamilton population (energy-weighted) | Negative = bonding |
| COOP | Overlap population (electron-weighted) | Positive = bonding |
| COBI | Bond index (normalized) | Positive = bonding |

COHP is most commonly used because it directly relates to bond energy contributions.

## Common Issues

| Issue | Cause | Solution |
|-------|-------|----------|
| LOBSTER crashes with "basis set too small" | Missing basis functions for an element | Add more orbitals in `basisfunctions` (include semi-core states) |
| High charge spilling (> 5%) | Insufficient local basis to represent plane-wave states | Add more orbitals; check that PAW datasets are LOBSTER-compatible |
| LOBSTER cannot read QE wavefunctions | `wf_collect` not set or wrong outdir | Set `wf_collect=.true.`; ensure LOBSTER can find the tmp/ directory |
| COHP is zero everywhere | Wrong bond specification or no bonds in cutoff range | Check `cohpGenerator` distance range; verify atom indices |
| ICOHP does not match expected bond strength | Wrong energy integration window or E_Fermi | Verify `COHPstartEnergy` covers all valence bands; check Fermi energy |
| LOBSTER fails with symmetry error | Symmetry was enabled in DFT | Set `nosym=.true.` and `noinv=.true.` in QE; `ISYM=-1` in VASP |
| Antibonding states not captured | Not enough empty bands | Increase `nbnd` in QE or `NBANDS` in VASP |
| LOBSTER binary not found | Not installed or not in PATH | Download from cohp.de; add to PATH or use full path |
| Column assignment wrong in COHPCAR | Header parsing differs by LOBSTER version | Print header line to identify column mapping; adjust column indices |
| Results differ from literature | Different pseudopotentials or basis sets | Use same PP family (pslibrary for QE, standard PAW for VASP) and document basis |
