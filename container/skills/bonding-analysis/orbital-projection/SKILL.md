# Orbital-Projected Analysis

Projected density of states (PDOS), fat bands, crystal field splitting, and orbital hybridization analysis using Quantum ESPRESSO `projwfc.x`.

## When to Use

- Determine which atomic orbitals contribute to states near the Fermi level.
- Identify orbital hybridization (e.g., sp3, pd mixing).
- Visualize crystal field splitting of d-orbitals in transition metal compounds.
- Create fat-band plots showing orbital character of each band.
- Understand bonding/antibonding character of electronic states.

## Prerequisites

- Quantum ESPRESSO 7.5 (`pw.x`, `projwfc.x`)
- Python: `numpy`, `matplotlib`, `scipy`
- Pseudopotentials with appropriate valence orbitals
- For fat bands: need an nscf calculation along a k-path

## Detailed Steps

### Step 1: SCF Calculation

**File: `scf.in`** (example: SrTiO3 -- perovskite with d-orbital physics)

```
&CONTROL
    calculation  = 'scf'
    prefix       = 'srtio3'
    outdir       = './tmp'
    pseudo_dir   = './pseudo'
    tprnfor      = .true.
/
&SYSTEM
    ibrav        = 1
    celldm(1)    = 7.38
    nat          = 5
    ntyp         = 3
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
  Sr  87.620   Sr.pbe-spn-rrkjus_psl.1.0.0.UPF
  Ti  47.867   Ti.pbe-spn-rrkjus_psl.1.0.0.UPF
  O   15.999   O.pbe-n-rrkjus_psl.1.0.0.UPF

ATOMIC_POSITIONS {crystal}
  Sr  0.0  0.0  0.0
  Ti  0.5  0.5  0.5
  O   0.5  0.5  0.0
  O   0.5  0.0  0.5
  O   0.0  0.5  0.5

K_POINTS {automatic}
  8 8 8  0 0 0
```

**Run:**
```bash
pw.x < scf.in > scf.out
```

### Step 2: NSCF Calculation for PDOS (Dense k-grid)

For accurate PDOS, use a denser k-grid in a non-self-consistent calculation.

**File: `nscf_pdos.in`**

```
&CONTROL
    calculation  = 'nscf'
    prefix       = 'srtio3'
    outdir       = './tmp'
    pseudo_dir   = './pseudo'
/
&SYSTEM
    ibrav        = 1
    celldm(1)    = 7.38
    nat          = 5
    ntyp         = 3
    ecutwfc      = 60.0
    ecutrho      = 600.0
    occupations  = 'tetrahedra'
    nosym        = .true.
/
&ELECTRONS
    conv_thr     = 1.0d-8
/
ATOMIC_SPECIES
  Sr  87.620   Sr.pbe-spn-rrkjus_psl.1.0.0.UPF
  Ti  47.867   Ti.pbe-spn-rrkjus_psl.1.0.0.UPF
  O   15.999   O.pbe-n-rrkjus_psl.1.0.0.UPF

ATOMIC_POSITIONS {crystal}
  Sr  0.0  0.0  0.0
  Ti  0.5  0.5  0.5
  O   0.5  0.5  0.0
  O   0.5  0.0  0.5
  O   0.0  0.5  0.5

K_POINTS {automatic}
  12 12 12  0 0 0
```

**Run:**
```bash
pw.x < nscf_pdos.in > nscf_pdos.out
```

### Step 3: Run projwfc.x for PDOS

**File: `projwfc_pdos.in`**

```
&PROJWFC
    prefix  = 'srtio3'
    outdir  = './tmp'
    filpdos = 'pdos'
    Emin    = -15.0
    Emax    = 10.0
    DeltaE  = 0.05
    ngauss  = 0
    degauss = 0.02
/
```

**Run:**
```bash
projwfc.x < projwfc_pdos.in > projwfc_pdos.out
```

This produces output files:
- `pdos.pdos_tot` -- total DOS
- `pdos.pdos_atm#N(SYMBOL)_wfc#M(ORBITAL)` -- projected DOS per atom per orbital

Example filenames:
```
pdos.pdos_atm#1(Sr)_wfc#1(s)
pdos.pdos_atm#1(Sr)_wfc#2(p)
pdos.pdos_atm#2(Ti)_wfc#1(s)
pdos.pdos_atm#2(Ti)_wfc#2(p)
pdos.pdos_atm#2(Ti)_wfc#3(d)
pdos.pdos_atm#3(O)_wfc#1(s)
pdos.pdos_atm#3(O)_wfc#2(p)
```

### Step 4: Parse and Plot PDOS

#### 4a: Basic PDOS Parsing and Plotting

```python
#!/usr/bin/env python3
"""
Parse and plot projected DOS from projwfc.x output files.
"""
import numpy as np
import matplotlib.pyplot as plt
import glob
import os
import re


def parse_pdos_file(filename):
    """
    Parse a single projwfc.x PDOS file.

    Returns
    -------
    energy : np.ndarray
        Energy values (eV).
    pdos : np.ndarray
        PDOS values (states/eV). For spin-unpolarized: single column.
        For spin-polarized: two columns (up, down).
    """
    data = np.loadtxt(filename, comments='#')
    # Column 0: energy, Column 1: ldos, Column 2: pdos
    # (or for spin-pol: energy, ldos_up, ldos_down, pdos_up, pdos_down)
    energy = data[:, 0]
    if data.shape[1] == 3:
        # Non-spin-polarized: E, ldos, pdos
        pdos = data[:, 2]
    elif data.shape[1] >= 5:
        # Spin-polarized: E, ldos_up, ldos_down, pdos_up, pdos_down
        pdos = data[:, 3]  # spin-up pdos; data[:,4] for spin-down
    else:
        pdos = data[:, 1]
    return energy, pdos


def parse_pdos_filename(filename):
    """
    Extract atom index, symbol, wfc index, and orbital from filename.

    Example: pdos.pdos_atm#2(Ti)_wfc#3(d) -> (2, 'Ti', 3, 'd')
    """
    basename = os.path.basename(filename)
    match = re.search(r'atm#(\d+)\((\w+)\)_wfc#(\d+)\((\w+)\)', basename)
    if match:
        return int(match.group(1)), match.group(2), int(match.group(3)), match.group(4)
    return None, None, None, None


def collect_pdos(pdos_prefix='pdos'):
    """
    Collect all PDOS files and organize by atom and orbital.

    Returns
    -------
    dict
        {(atom_index, symbol, orbital): (energy, pdos)}
    """
    pattern = f"{pdos_prefix}.pdos_atm#*"
    files = sorted(glob.glob(pattern))
    pdos_data = {}

    for f in files:
        atom_idx, symbol, wfc_idx, orbital = parse_pdos_filename(f)
        if atom_idx is None:
            continue
        energy, pdos = parse_pdos_file(f)
        key = (atom_idx, symbol, orbital)
        if key in pdos_data:
            # Same atom, same orbital type but different wfc index: sum them
            pdos_data[key] = (energy, pdos_data[key][1] + pdos)
        else:
            pdos_data[key] = (energy, pdos)

    return pdos_data


def get_fermi_energy(scf_output='scf.out'):
    """Extract Fermi energy from pw.x output."""
    with open(scf_output, 'r') as f:
        for line in f:
            if 'Fermi energy' in line or 'highest occupied' in line:
                # "the Fermi energy is    XX.XXXX ev"
                match = re.search(r'([\d.-]+)\s*ev', line, re.IGNORECASE)
                if match:
                    return float(match.group(1))
    return 0.0


def plot_pdos_by_element(pdos_prefix='pdos', scf_output='scf.out',
                         output_png='pdos_element.png'):
    """
    Plot PDOS summed by element and orbital type.
    """
    pdos_data = collect_pdos(pdos_prefix)
    e_fermi = get_fermi_energy(scf_output)

    # Group by element and orbital
    element_orbitals = {}
    for (atom_idx, symbol, orbital), (energy, pdos) in pdos_data.items():
        key = (symbol, orbital)
        if key in element_orbitals:
            element_orbitals[key] = (energy, element_orbitals[key][1] + pdos)
        else:
            element_orbitals[key] = (energy, pdos.copy())

    # Get unique elements preserving order
    elements = []
    for (atom_idx, symbol, orbital) in pdos_data:
        if symbol not in elements:
            elements.append(symbol)

    orbital_colors = {'s': '#1f77b4', 'p': '#ff7f0e', 'd': '#2ca02c',
                      'f': '#d62728', 'sp': '#9467bd'}

    n_elements = len(elements)
    fig, axes = plt.subplots(n_elements, 1, figsize=(10, 3 * n_elements),
                             sharex=True)
    if n_elements == 1:
        axes = [axes]

    for ax, element in zip(axes, elements):
        for (sym, orb), (energy, pdos) in sorted(element_orbitals.items()):
            if sym != element:
                continue
            color = orbital_colors.get(orb, 'gray')
            ax.plot(energy - e_fermi, pdos, '-', color=color, linewidth=1.2,
                    label=f'{orb}')
            ax.fill_between(energy - e_fermi, 0, pdos, alpha=0.15, color=color)

        ax.axvline(x=0, color='black', linestyle='--', linewidth=0.5, alpha=0.5)
        ax.set_ylabel(f'{element}\nPDOS (states/eV)')
        ax.legend(loc='upper right', fontsize=9)
        ax.set_xlim(-15, 10)
        ax.grid(True, alpha=0.2)

    axes[-1].set_xlabel(r'$E - E_F$ (eV)')
    fig.suptitle('Projected Density of States', fontsize=14, y=1.01)
    plt.tight_layout()
    plt.savefig(output_png, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_png}")


# ----- Run -----
plot_pdos_by_element('pdos', 'scf.out', 'pdos_element.png')
```

#### 4b: PDOS per Atom (Useful for Inequivalent Sites)

```python
#!/usr/bin/env python3
"""
Plot PDOS for each individual atom, useful when atoms of the same element
occupy inequivalent sites (e.g., surface vs bulk, different coordination).
"""
import numpy as np
import matplotlib.pyplot as plt
import glob
import os
import re


def parse_pdos_file(filename):
    data = np.loadtxt(filename, comments='#')
    energy = data[:, 0]
    pdos = data[:, 2] if data.shape[1] >= 3 else data[:, 1]
    return energy, pdos


def parse_pdos_filename(filename):
    basename = os.path.basename(filename)
    match = re.search(r'atm#(\d+)\((\w+)\)_wfc#(\d+)\((\w+)\)', basename)
    if match:
        return int(match.group(1)), match.group(2), int(match.group(3)), match.group(4)
    return None, None, None, None


def plot_pdos_per_atom(pdos_prefix='pdos', scf_output='scf.out',
                       atom_indices=None, output_png='pdos_per_atom.png'):
    """
    Plot PDOS for specific atoms, broken down by orbital.

    Parameters
    ----------
    atom_indices : list of int or None
        Atom indices to plot (1-based). None = all atoms.
    """
    pattern = f"{pdos_prefix}.pdos_atm#*"
    files = sorted(glob.glob(pattern))

    # Organize by atom
    atom_data = {}
    for f in files:
        atom_idx, symbol, wfc_idx, orbital = parse_pdos_filename(f)
        if atom_idx is None:
            continue
        if atom_indices is not None and atom_idx not in atom_indices:
            continue
        energy, pdos = parse_pdos_file(f)
        if atom_idx not in atom_data:
            atom_data[atom_idx] = {'symbol': symbol, 'orbitals': {}}
        if orbital in atom_data[atom_idx]['orbitals']:
            atom_data[atom_idx]['orbitals'][orbital] = (
                energy, atom_data[atom_idx]['orbitals'][orbital][1] + pdos)
        else:
            atom_data[atom_idx]['orbitals'][orbital] = (energy, pdos)

    # Get Fermi energy
    e_fermi = 0.0
    try:
        with open(scf_output, 'r') as fh:
            for line in fh:
                if 'Fermi energy' in line or 'highest occupied' in line:
                    match = re.search(r'([\d.-]+)\s*ev', line, re.IGNORECASE)
                    if match:
                        e_fermi = float(match.group(1))
                        break
    except FileNotFoundError:
        pass

    orbital_colors = {'s': '#1f77b4', 'p': '#ff7f0e', 'd': '#2ca02c',
                      'f': '#d62728'}

    n_atoms = len(atom_data)
    fig, axes = plt.subplots(n_atoms, 1, figsize=(10, 2.5 * n_atoms), sharex=True)
    if n_atoms == 1:
        axes = [axes]

    for ax, (atom_idx, ainfo) in zip(axes, sorted(atom_data.items())):
        for orbital, (energy, pdos) in sorted(ainfo['orbitals'].items()):
            color = orbital_colors.get(orbital, 'gray')
            ax.plot(energy - e_fermi, pdos, '-', color=color, linewidth=1.0,
                    label=f'{orbital}')
            ax.fill_between(energy - e_fermi, 0, pdos, alpha=0.1, color=color)
        ax.axvline(x=0, color='k', linestyle='--', linewidth=0.5, alpha=0.5)
        ax.set_ylabel(f"#{atom_idx} {ainfo['symbol']}\n(states/eV)")
        ax.legend(loc='upper right', fontsize=8)
        ax.grid(True, alpha=0.2)

    axes[-1].set_xlabel(r'$E - E_F$ (eV)')
    fig.suptitle('Atom-Resolved PDOS', fontsize=14, y=1.01)
    plt.tight_layout()
    plt.savefig(output_png, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_png}")


# ----- Run -----
plot_pdos_per_atom('pdos', 'scf.out', atom_indices=[2, 3], output_png='pdos_Ti_O.png')
```

### Step 5: Fat Bands (Orbital-Projected Band Structure)

#### 5a: NSCF on k-path

**File: `nscf_bands.in`** (k-path for cubic perovskite)

```
&CONTROL
    calculation  = 'bands'
    prefix       = 'srtio3'
    outdir       = './tmp'
    pseudo_dir   = './pseudo'
/
&SYSTEM
    ibrav        = 1
    celldm(1)    = 7.38
    nat          = 5
    ntyp         = 3
    ecutwfc      = 60.0
    ecutrho      = 600.0
    nbnd         = 30
/
&ELECTRONS
    conv_thr     = 1.0d-8
/
ATOMIC_SPECIES
  Sr  87.620   Sr.pbe-spn-rrkjus_psl.1.0.0.UPF
  Ti  47.867   Ti.pbe-spn-rrkjus_psl.1.0.0.UPF
  O   15.999   O.pbe-n-rrkjus_psl.1.0.0.UPF

ATOMIC_POSITIONS {crystal}
  Sr  0.0  0.0  0.0
  Ti  0.5  0.5  0.5
  O   0.5  0.5  0.0
  O   0.5  0.0  0.5
  O   0.0  0.5  0.5

K_POINTS {crystal_b}
5
  0.0  0.0  0.0  40  ! Gamma
  0.5  0.0  0.0  40  ! X
  0.5  0.5  0.0  40  ! M
  0.5  0.5  0.5  40  ! R
  0.0  0.0  0.0  1   ! Gamma
```

**Run:**
```bash
pw.x < nscf_bands.in > nscf_bands.out
```

#### 5b: projwfc.x on the Band Structure k-path

**File: `projwfc_bands.in`**

```
&PROJWFC
    prefix  = 'srtio3'
    outdir  = './tmp'
    filpdos = 'band_pdos'
    lsym    = .false.
    kresolveddos = .true.
/
```

**Run:**
```bash
projwfc.x < projwfc_bands.in > projwfc_bands.out
```

This produces projections for each k-point and band. The output file `projwfc_bands.out` contains the projection data, and `filproj` (if set) stores binary projection data. The PDOS files also contain k-resolved information when `kresolveddos = .true.`.

#### 5c: Parse projwfc.x Output for Fat Bands

```python
#!/usr/bin/env python3
"""
Parse projwfc.x output to extract orbital projections for fat band plots.
projwfc.x prints projections in the stdout when run on a bands calculation.

The format in projwfc.out:
    k =   0.0000  0.0000  0.0000
      ==== e(   1) =   -XX.XXXX eV ====
       psi = 0.XXX*[#  1: l=0 m= 1 (s  )] + 0.XXX*[#  2: l=1 m= 1 (p  )] + ...
      |psi|^2 = 0.XXXX
"""
import numpy as np
import matplotlib.pyplot as plt
import re
from collections import defaultdict


def parse_projwfc_bands(projwfc_out='projwfc_bands.out'):
    """
    Parse fat-band projections from projwfc.x output.

    Returns
    -------
    kpoints : list of np.ndarray
        k-point coordinates.
    eigenvalues : dict
        {band_index: list of eigenvalues at each k-point}
    projections : dict
        {(atom_idx, orbital): {band_index: list of |projection|^2 at each k-point}}
    """
    with open(projwfc_out, 'r') as f:
        lines = f.readlines()

    kpoints = []
    eigenvalues = defaultdict(list)
    projections = defaultdict(lambda: defaultdict(list))

    current_k = None
    current_band = None
    current_energy = None

    # Regex patterns
    k_pattern = re.compile(r'k\s*=\s*([\d.-]+)\s+([\d.-]+)\s+([\d.-]+)')
    band_pattern = re.compile(r'==== e\(\s*(\d+)\)\s*=\s*([\d.-]+)\s*eV\s*====')
    proj_pattern = re.compile(
        r'([\d.]+)\*\[#\s*(\d+):\s*l=\s*(\d+)\s*m=\s*(\d+)\s*\((\w+)\s*\)\]')

    # Map l quantum number to orbital label
    l_to_orb = {0: 's', 1: 'p', 2: 'd', 3: 'f'}

    for line in lines:
        # Check for k-point
        k_match = k_pattern.search(line)
        if k_match and 'k =' in line:
            current_k = np.array([float(k_match.group(i)) for i in range(1, 4)])
            kpoints.append(current_k)
            continue

        # Check for band eigenvalue
        band_match = band_pattern.search(line)
        if band_match:
            current_band = int(band_match.group(1))
            current_energy = float(band_match.group(2))
            eigenvalues[current_band].append(current_energy)
            continue

        # Check for projections
        if current_band is not None and 'psi' in line and '|psi|^2' not in line:
            proj_matches = proj_pattern.findall(line)
            # Accumulate projections for this band at this k-point
            atom_orbital_proj = defaultdict(float)
            for weight, atom_idx, l_val, m_val, orb_label in proj_matches:
                w = float(weight)
                a_idx = int(atom_idx)
                orb = l_to_orb.get(int(l_val), orb_label.strip())
                atom_orbital_proj[(a_idx, orb)] += w

            for (a_idx, orb), w in atom_orbital_proj.items():
                projections[(a_idx, orb)][current_band].append(w)

        if '|psi|^2' in line:
            current_band = None  # Reset for next band

    return kpoints, dict(eigenvalues), dict(projections)


def compute_k_distances(kpoints, reciprocal_lattice=None):
    """
    Compute cumulative k-point distances for band structure x-axis.
    """
    kpoints = np.array(kpoints)
    if reciprocal_lattice is not None:
        # Convert from crystal to Cartesian
        kpoints_cart = kpoints @ reciprocal_lattice
    else:
        kpoints_cart = kpoints

    distances = [0.0]
    for i in range(1, len(kpoints_cart)):
        dk = np.linalg.norm(kpoints_cart[i] - kpoints_cart[i - 1])
        distances.append(distances[-1] + dk)
    return np.array(distances)


def plot_fat_bands(projwfc_out='projwfc_bands.out', scf_output='scf.out',
                   atom_orbital_groups=None, output_png='fat_bands.png',
                   emin=-8, emax=8):
    """
    Plot fat bands with orbital character shown as line thickness/color.

    Parameters
    ----------
    atom_orbital_groups : dict or None
        {label: [(atom_idx, orbital), ...]}
        Groups of (atom, orbital) pairs to sum and display.
        Example: {'Ti-d': [(2, 'd')], 'O-p': [(3, 'p'), (4, 'p'), (5, 'p')]}
        If None, auto-detect from data.
    emin, emax : float
        Energy range relative to Fermi level.
    """
    kpoints, eigenvalues, projections = parse_projwfc_bands(projwfc_out)

    if not kpoints:
        print("ERROR: No k-points found. Check projwfc.x output format.")
        return

    # Get Fermi energy
    e_fermi = 0.0
    try:
        with open(scf_output, 'r') as f:
            for line in f:
                if 'Fermi energy' in line or 'highest occupied' in line:
                    match = re.search(r'([\d.-]+)\s*ev', line, re.IGNORECASE)
                    if match:
                        e_fermi = float(match.group(1))
                        break
    except FileNotFoundError:
        pass

    k_dist = compute_k_distances(kpoints)
    n_kpts = len(kpoints)

    # Auto-detect groups if not specified
    if atom_orbital_groups is None:
        all_keys = set()
        for key in projections:
            all_keys.add(key)
        # Group by orbital type
        atom_orbital_groups = {}
        for (a_idx, orb) in sorted(all_keys):
            label = f'Atom{a_idx}-{orb}'
            atom_orbital_groups[label] = [(a_idx, orb)]

    # Assign colors
    group_colors = plt.cm.Set1(np.linspace(0, 1, len(atom_orbital_groups)))

    fig, ax = plt.subplots(figsize=(10, 7))

    # First, plot thin black lines for all bands
    for band_idx in sorted(eigenvalues.keys()):
        energies = np.array(eigenvalues[band_idx]) - e_fermi
        if len(energies) != n_kpts:
            continue
        ax.plot(k_dist, energies, 'k-', linewidth=0.3, alpha=0.3)

    # Overlay fat bands
    for (label, group), color in zip(atom_orbital_groups.items(), group_colors):
        for band_idx in sorted(eigenvalues.keys()):
            energies = np.array(eigenvalues[band_idx]) - e_fermi
            if len(energies) != n_kpts:
                continue

            # Sum projections for this group
            weights = np.zeros(n_kpts)
            for (a_idx, orb) in group:
                if (a_idx, orb) in projections and band_idx in projections[(a_idx, orb)]:
                    proj_vals = projections[(a_idx, orb)][band_idx]
                    if len(proj_vals) == n_kpts:
                        weights += np.array(proj_vals)

            # Only plot if there is significant projection
            if np.max(weights) < 0.01:
                continue

            # Plot as scatter with size proportional to weight
            mask = (energies >= emin) & (energies <= emax)
            ax.scatter(k_dist[mask], energies[mask],
                      s=weights[mask] * 80,  # scale factor for visibility
                      c=[color], alpha=0.6, edgecolors='none',
                      label=label if band_idx == sorted(eigenvalues.keys())[0] else '')

    # Remove duplicate legend entries
    handles, labels_leg = ax.get_legend_handles_labels()
    seen = set()
    unique_handles, unique_labels = [], []
    for h, l in zip(handles, labels_leg):
        if l not in seen:
            seen.add(l)
            unique_handles.append(h)
            unique_labels.append(l)
    ax.legend(unique_handles, unique_labels, loc='upper right', fontsize=9,
              markerscale=2)

    ax.axhline(y=0, color='gray', linestyle='--', linewidth=0.5)
    ax.set_xlim(k_dist[0], k_dist[-1])
    ax.set_ylim(emin, emax)
    ax.set_ylabel(r'$E - E_F$ (eV)')
    ax.set_title('Fat Band Structure')

    # Add high-symmetry labels
    # Detect high-symmetry points (where k-distance jumps are zero or very small)
    hsp_labels = [r'$\Gamma$', 'X', 'M', 'R', r'$\Gamma$']
    # Simple heuristic: place labels at start, end, and points where direction changes
    n_segments = len(hsp_labels) - 1
    segment_size = n_kpts // n_segments
    hsp_positions = [k_dist[min(i * segment_size, n_kpts - 1)]
                     for i in range(n_segments)] + [k_dist[-1]]

    ax.set_xticks(hsp_positions)
    ax.set_xticklabels(hsp_labels)
    for pos in hsp_positions:
        ax.axvline(x=pos, color='gray', linewidth=0.5, alpha=0.5)

    plt.tight_layout()
    plt.savefig(output_png, dpi=200)
    plt.close()
    print(f"Saved: {output_png}")


# ----- Run -----
plot_fat_bands(
    projwfc_out='projwfc_bands.out',
    scf_output='scf.out',
    atom_orbital_groups={
        'Ti-d': [(2, 'd')],
        'O-p': [(3, 'p'), (4, 'p'), (5, 'p')],
        'Sr-d': [(1, 'd')],
    },
    output_png='fat_bands.png',
    emin=-8,
    emax=8
)
```

### Step 6: Crystal Field Splitting Analysis

```python
#!/usr/bin/env python3
"""
Analyze crystal field splitting of d-orbitals from PDOS.

In an octahedral field (e.g., TiO6 in SrTiO3):
  - t2g: dxy, dxz, dyz  (lower energy)
  - eg:  dz2, dx2-y2    (higher energy)

projwfc.x with l=2 gives m = -2, -1, 0, 1, 2:
  m=-2: dxy
  m=-1: dyz
  m= 0: dz2
  m= 1: dxz
  m= 2: dx2-y2
"""
import numpy as np
import matplotlib.pyplot as plt
import re
import glob


def parse_d_orbital_pdos(pdos_prefix='pdos', atom_idx=2, atom_symbol='Ti'):
    """
    Parse individual m-resolved d-orbital PDOS.

    projwfc.x generates files like:
      pdos.pdos_atm#2(Ti)_wfc#3(d)
    But for m-resolved, need to check if the file contains multiple columns
    or if there are separate files per m.

    For QE 7.x, a single d-wfc file may contain columns for each m value:
      E, ldos, pdos(m=-2), pdos(m=-1), pdos(m=0), pdos(m=1), pdos(m=2)

    The header comment line tells the column layout.
    """
    d_file_pattern = f"{pdos_prefix}.pdos_atm#{atom_idx}({atom_symbol})_wfc#*"
    d_files = glob.glob(d_file_pattern)

    # Find the d orbital file
    d_file = None
    for f in d_files:
        if '(d)' in f:
            d_file = f
            break

    if d_file is None:
        print(f"ERROR: No d-orbital PDOS file found for atom #{atom_idx} ({atom_symbol})")
        return None

    # Read header to determine column layout
    with open(d_file, 'r') as fh:
        header = fh.readline()

    data = np.loadtxt(d_file, comments='#')
    energy = data[:, 0]

    # Check number of columns
    n_cols = data.shape[1]

    if n_cols >= 7:
        # Columns: E, ldos, pdos_m-2, pdos_m-1, pdos_m0, pdos_m1, pdos_m2
        d_orbitals = {
            'dxy':      data[:, 2],   # m = -2
            'dyz':      data[:, 3],   # m = -1
            'dz2':      data[:, 4],   # m =  0
            'dxz':      data[:, 5],   # m =  1
            'dx2-y2':   data[:, 6],   # m =  2
        }
    elif n_cols == 3:
        # Only total d PDOS, no m-resolution
        print("WARNING: Only total d-PDOS available (no m-resolution).")
        print("Re-run projwfc.x with lsym=.false. for m-resolved output.")
        d_orbitals = {'d_total': data[:, 2]}
    else:
        print(f"WARNING: Unexpected number of columns ({n_cols}). Reading total d-PDOS.")
        d_orbitals = {'d_total': data[:, 1]}

    return energy, d_orbitals


def plot_crystal_field(pdos_prefix='pdos', atom_idx=2, atom_symbol='Ti',
                       scf_output='scf.out', output_png='crystal_field.png'):
    """
    Plot d-orbital PDOS to visualize crystal field splitting.
    """
    result = parse_d_orbital_pdos(pdos_prefix, atom_idx, atom_symbol)
    if result is None:
        return
    energy, d_orbitals = result

    # Get Fermi energy
    e_fermi = 0.0
    try:
        with open(scf_output, 'r') as f:
            for line in f:
                if 'Fermi energy' in line or 'highest occupied' in line:
                    match = re.search(r'([\d.-]+)\s*ev', line, re.IGNORECASE)
                    if match:
                        e_fermi = float(match.group(1))
                        break
    except FileNotFoundError:
        pass

    e_shifted = energy - e_fermi

    # Define t2g and eg groups
    t2g_orbs = ['dxy', 'dyz', 'dxz']
    eg_orbs = ['dz2', 'dx2-y2']

    fig, axes = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

    # Panel 1: Individual d-orbitals
    ax1 = axes[0]
    colors = {'dxy': '#1f77b4', 'dyz': '#ff7f0e', 'dz2': '#2ca02c',
              'dxz': '#d62728', 'dx2-y2': '#9467bd', 'd_total': 'black'}
    for orb_name, pdos in d_orbitals.items():
        c = colors.get(orb_name, 'gray')
        ax1.plot(e_shifted, pdos, '-', color=c, linewidth=1.2, label=orb_name)
    ax1.axvline(x=0, color='k', linestyle='--', linewidth=0.5, alpha=0.5)
    ax1.set_ylabel(f'{atom_symbol} d-PDOS (states/eV)')
    ax1.set_title(f'Crystal Field Splitting: {atom_symbol} d-orbitals')
    ax1.legend(loc='upper right', fontsize=9)
    ax1.grid(True, alpha=0.2)

    # Panel 2: t2g vs eg
    ax2 = axes[1]
    if all(k in d_orbitals for k in t2g_orbs):
        t2g_sum = sum(d_orbitals[k] for k in t2g_orbs)
        eg_sum = sum(d_orbitals[k] for k in eg_orbs)

        ax2.plot(e_shifted, t2g_sum, 'b-', linewidth=1.5, label=r'$t_{2g}$ (dxy + dyz + dxz)')
        ax2.fill_between(e_shifted, 0, t2g_sum, alpha=0.15, color='blue')
        ax2.plot(e_shifted, eg_sum, 'r-', linewidth=1.5, label=r'$e_g$ ($d_{z^2}$ + $d_{x^2-y^2}$)')
        ax2.fill_between(e_shifted, 0, eg_sum, alpha=0.15, color='red')

        # Estimate crystal field splitting (10Dq)
        # Find peak positions of t2g and eg
        mask = e_shifted > -2  # look at unoccupied states for d0 metals
        if np.any(mask):
            t2g_masked = t2g_sum[mask]
            eg_masked = eg_sum[mask]
            e_masked = e_shifted[mask]
            if np.max(t2g_masked) > 0.01 and np.max(eg_masked) > 0.01:
                t2g_peak_e = e_masked[np.argmax(t2g_masked)]
                eg_peak_e = e_masked[np.argmax(eg_masked)]
                splitting = abs(eg_peak_e - t2g_peak_e)
                ax2.annotate(f'10Dq ~ {splitting:.2f} eV',
                           xy=(0.05, 0.9), xycoords='axes fraction',
                           fontsize=12, fontweight='bold',
                           bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    else:
        # Only total d available
        for orb_name, pdos in d_orbitals.items():
            ax2.plot(e_shifted, pdos, 'k-', linewidth=1.5, label=orb_name)

    ax2.axvline(x=0, color='k', linestyle='--', linewidth=0.5, alpha=0.5)
    ax2.set_xlabel(r'$E - E_F$ (eV)')
    ax2.set_ylabel(f'{atom_symbol} d-PDOS (states/eV)')
    ax2.legend(loc='upper right', fontsize=9)
    ax2.grid(True, alpha=0.2)

    plt.tight_layout()
    plt.savefig(output_png, dpi=200)
    plt.close()
    print(f"Saved: {output_png}")


# ----- Run -----
plot_crystal_field(
    pdos_prefix='pdos',
    atom_idx=2,
    atom_symbol='Ti',
    scf_output='scf.out',
    output_png='crystal_field_Ti.png'
)
```

### Step 7: Orbital Hybridization Analysis

```python
#!/usr/bin/env python3
"""
Identify orbital hybridization by comparing PDOS of different atoms/orbitals.
Hybridization is indicated when orbitals of different atoms show coincident
peaks in the PDOS at the same energy.
"""
import numpy as np
import matplotlib.pyplot as plt
import re
import glob


def parse_pdos_file(filename):
    data = np.loadtxt(filename, comments='#')
    energy = data[:, 0]
    pdos = data[:, 2] if data.shape[1] >= 3 else data[:, 1]
    return energy, pdos


def parse_pdos_filename(filename):
    import os
    basename = os.path.basename(filename)
    match = re.search(r'atm#(\d+)\((\w+)\)_wfc#(\d+)\((\w+)\)', basename)
    if match:
        return int(match.group(1)), match.group(2), int(match.group(3)), match.group(4)
    return None, None, None, None


def hybridization_analysis(pdos_prefix='pdos', scf_output='scf.out',
                           pairs=None, output_png='hybridization.png'):
    """
    Analyze orbital hybridization between specified orbital pairs.

    Parameters
    ----------
    pairs : list of tuple
        Each tuple: ((atom_idx_1, orbital_1), (atom_idx_2, orbital_2), label)
        Example: [((2, 'd'), (3, 'p'), 'Ti-d / O-p hybridization')]
    """
    if pairs is None:
        print("Specify orbital pairs for hybridization analysis.")
        return

    # Collect PDOS data
    pattern = f"{pdos_prefix}.pdos_atm#*"
    files = sorted(glob.glob(pattern))

    pdos_data = {}
    for f in files:
        atom_idx, symbol, wfc_idx, orbital = parse_pdos_filename(f)
        if atom_idx is None:
            continue
        energy, pdos = parse_pdos_file(f)
        key = (atom_idx, orbital)
        if key in pdos_data:
            pdos_data[key] = (energy, pdos_data[key][1] + pdos)
        else:
            pdos_data[key] = (energy, pdos)

    # Fermi energy
    e_fermi = 0.0
    try:
        with open(scf_output, 'r') as f:
            for line in f:
                if 'Fermi energy' in line or 'highest occupied' in line:
                    match = re.search(r'([\d.-]+)\s*ev', line, re.IGNORECASE)
                    if match:
                        e_fermi = float(match.group(1))
                        break
    except FileNotFoundError:
        pass

    n_pairs = len(pairs)
    fig, axes = plt.subplots(n_pairs, 1, figsize=(10, 4 * n_pairs), sharex=True)
    if n_pairs == 1:
        axes = [axes]

    for ax, (key1, key2, label) in zip(axes, pairs):
        if key1 not in pdos_data:
            print(f"WARNING: {key1} not found in PDOS data")
            continue
        if key2 not in pdos_data:
            print(f"WARNING: {key2} not found in PDOS data")
            continue

        e1, p1 = pdos_data[key1]
        e2, p2 = pdos_data[key2]

        # Normalize for comparison
        p1_norm = p1 / (np.max(p1) + 1e-10)
        p2_norm = p2 / (np.max(p2) + 1e-10)

        e1_shifted = e1 - e_fermi
        e2_shifted = e2 - e_fermi

        ax.plot(e1_shifted, p1_norm, 'b-', linewidth=1.2,
                label=f'Atom#{key1[0]}-{key1[1]}')
        ax.fill_between(e1_shifted, 0, p1_norm, alpha=0.1, color='blue')
        ax.plot(e2_shifted, p2_norm, 'r-', linewidth=1.2,
                label=f'Atom#{key2[0]}-{key2[1]}')
        ax.fill_between(e2_shifted, 0, p2_norm, alpha=0.1, color='red')

        # Compute overlap (hybridization indicator)
        # Interpolate both onto the same energy grid
        e_common = np.linspace(max(e1_shifted.min(), e2_shifted.min()),
                               min(e1_shifted.max(), e2_shifted.max()), 1000)
        p1_interp = np.interp(e_common, e1_shifted, p1_norm)
        p2_interp = np.interp(e_common, e2_shifted, p2_norm)
        overlap = np.minimum(p1_interp, p2_interp)
        overlap_integral = np.trapz(overlap, e_common)

        ax.fill_between(e_common, 0, overlap * np.max([p1_norm.max(), p2_norm.max()]),
                        alpha=0.2, color='purple', label=f'Overlap (integral={overlap_integral:.2f})')

        ax.axvline(x=0, color='k', linestyle='--', linewidth=0.5, alpha=0.5)
        ax.set_ylabel('Normalized PDOS')
        ax.set_title(label)
        ax.legend(loc='upper right', fontsize=9)
        ax.grid(True, alpha=0.2)

    axes[-1].set_xlabel(r'$E - E_F$ (eV)')
    fig.suptitle('Orbital Hybridization Analysis', fontsize=14, y=1.01)
    plt.tight_layout()
    plt.savefig(output_png, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_png}")

    # Print hybridization summary
    print("\nHybridization Summary:")
    print("=" * 50)
    print("Coincident peaks at the same energy indicate hybridization.")
    print("Larger overlap integral = stronger hybridization.")


# ----- Run -----
hybridization_analysis(
    pdos_prefix='pdos',
    scf_output='scf.out',
    pairs=[
        ((2, 'd'), (3, 'p'), 'Ti-3d / O-2p hybridization'),
        ((1, 'd'), (3, 'p'), 'Sr-4d / O-2p hybridization'),
    ],
    output_png='hybridization.png'
)
```

### Step 8: Generate k-path with pymatgen (Helper)

```python
#!/usr/bin/env python3
"""
Generate a k-path for band structure calculations using pymatgen.
Produces the K_POINTS block for pw.x input.
"""
from pymatgen.core import Structure
from pymatgen.symmetry.bandstructure import HighSymmKpath


def generate_kpath(structure_file='POSCAR', npoints=40,
                   output_file='kpath.txt'):
    """
    Generate k-point path for QE bands calculation.

    Parameters
    ----------
    structure_file : str
        Input structure (POSCAR, CIF, etc.)
    npoints : int
        Number of k-points per segment.
    output_file : str
        Output file with K_POINTS block.
    """
    struct = Structure.from_file(structure_file)
    kpath = HighSymmKpath(struct)
    kpts = kpath.kpath

    labels = kpts['kpoints']
    path = kpts['path']

    print("High-symmetry points:")
    for label, coords in labels.items():
        print(f"  {label}: {coords}")

    print(f"\nPath: {path}")

    # Write K_POINTS block
    # Count total segments
    all_points = []
    for segment in path:
        for label in segment:
            coords = labels[label]
            all_points.append((label, coords))

    with open(output_file, 'w') as f:
        f.write(f"K_POINTS {{crystal_b}}\n")
        f.write(f"{len(all_points)}\n")
        for i, (label, coords) in enumerate(all_points):
            nk = npoints if i < len(all_points) - 1 else 1
            label_clean = label.replace('\\Gamma', 'Gamma').replace('$', '')
            f.write(f"  {coords[0]:.6f}  {coords[1]:.6f}  {coords[2]:.6f}  "
                    f"{nk}  ! {label_clean}\n")

    print(f"\nWrote: {output_file}")


# ----- Run -----
# generate_kpath('structure.cif', npoints=40, output_file='kpath.txt')
```

## Key Parameters

### projwfc.x Parameters

| Parameter | Description |
|-----------|-------------|
| `prefix` | Must match the pw.x calculation |
| `outdir` | Must match the pw.x calculation |
| `filpdos` | Prefix for PDOS output files |
| `Emin`, `Emax` | Energy window (eV) for PDOS |
| `DeltaE` | Energy grid spacing (eV); 0.05 typical |
| `ngauss` | 0 = Gaussian, 1 = Methfessel-Paxton, -1 = Fermi-Dirac |
| `degauss` | Broadening (Ry); 0.01--0.05 typical |
| `lsym` | `.true.` = symmetrize; `.false.` = preserve m-resolution |
| `kresolveddos` | `.true.` = k-resolved DOS output (for fat bands) |
| `filproj` | Output file for binary projections (optional) |

### Band Structure Calculation

| Parameter | Notes |
|-----------|-------|
| `calculation = 'bands'` | Non-SCF on a k-path |
| `nbnd` | Number of bands; include enough empty bands |
| `K_POINTS {crystal_b}` | k-path in crystal coordinates with # of points per segment |
| `nosym = .true.` | Sometimes needed for PDOS on k-path |

### Orbital Quantum Numbers

| l | Orbital | m values | QE m-index mapping |
|---|---------|----------|-------------------|
| 0 | s | 0 | m=1: s |
| 1 | p | -1, 0, 1 | m=1: pz, m=2: px, m=3: py |
| 2 | d | -2, -1, 0, 1, 2 | m=1: dz2, m=2: dxz, m=3: dyz, m=4: dx2-y2, m=5: dxy |
| 3 | f | -3, ..., 3 | see QE documentation |

Note: The m-index mapping depends on whether `lsym=.true.` or `.false.` and the QE version. Check the header of the PDOS file for the column layout.

## Interpreting Results

### PDOS Features

- **Peaks at same energy in two orbitals** = hybridization between those orbitals.
- **Sharp narrow peaks** = localized states (e.g., f-electrons, defect levels).
- **Broad features** = delocalized / itinerant states.
- **States at Fermi level** = metallic character from those orbitals.
- **Gap in PDOS** = band gap for that orbital character.

### Crystal Field Splitting

- **Octahedral (O_h)**: d splits into t2g (lower) and eg (higher). 10Dq = splitting.
- **Tetrahedral (T_d)**: d splits into e (lower) and t2 (higher). Splitting ~ 4/9 of octahedral.
- **Square planar (D_4h)**: d splits into four levels.
- **Typical 10Dq values**: 1--3 eV for 3d metals in oxides.

### Hybridization Indicators

- **sp hybridization**: s and p PDOS overlap at same energies.
- **pd hybridization** (e.g., TiO2): Ti-d and O-p show matching peaks, indicating covalent Ti-O bonds.
- **Bonding vs antibonding**: Lower-energy coincident peaks = bonding states; higher-energy = antibonding. Bonding states have same-sign overlap; antibonding have opposite phase (can be inferred from COOP, but PDOS overlap is a simpler indicator).

### Fat Bands

- **Large circles** at a given (k, E) point = strong orbital character of that band.
- Bands changing color along k-path = orbital character changes (band mixing).
- Flat bands with strong d-character = localized d-states.

## Common Issues

1. **projwfc.x crashes with segfault**: Usually due to memory. Reduce `nbnd` or k-points, or increase memory allocation.

2. **No PDOS files produced**: Check that `filpdos` is writable. Check for errors in `projwfc.out` (e.g., "cannot read wfc" means the wavefunctions were not saved -- re-run nscf with `wf_collect=.true.` or ensure `outdir` matches).

3. **m-resolved PDOS not available**: Set `lsym = .false.` in projwfc.x input. With `lsym = .true.`, projwfc.x symmetrizes and may combine m-values.

4. **Fat band projections sum to less than 1**: The spilling parameter indicates how much of the wavefunction is NOT captured by the atomic projections. For plane-wave basis with PAW/USPP pseudopotentials, some spilling is normal (typically < 5%).

5. **Energy reference mismatch**: PDOS energies are absolute (eV). Always shift by E_Fermi from the SCF output. For band structure, use the same Fermi energy from the SCF (not the bands) calculation.

6. **Inconsistent atom ordering**: The atom indices in projwfc.x output follow the order in the pw.x input file. Verify by checking the atom coordinates in `projwfc.out`.

7. **tetrahedra vs smearing for PDOS**: Use `occupations = 'tetrahedra'` in the nscf calculation for PDOS (gives sharper features). Use smearing for the SCF. The projwfc.x broadening (`degauss`) is separate and applied on top.

8. **Band structure k-path**: The `calculation = 'bands'` in pw.x must use the same prefix/outdir as the SCF. The k-path should follow the high-symmetry path for the crystal's Brillouin zone. Use pymatgen's `HighSymmKpath` to determine the correct path.
