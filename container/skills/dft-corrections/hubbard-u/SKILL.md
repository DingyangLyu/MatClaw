# Hubbard U Correction (DFT+U)

## When to Use

Apply DFT+U when your system has **strongly correlated electrons** -- typically partially filled d or f shells where standard GGA/LDA fails qualitatively:

- **Transition metal oxides**: NiO, Fe2O3, FeO, CoO, MnO, TiO2, VO2
- **f-electron systems**: lanthanide/actinide oxides (CeO2, UO2)
- **Mott insulators**: systems that GGA predicts as metallic but are experimentally insulating
- **Mixed-valence compounds**: Fe3O4, LiFePO4 (battery cathodes)
- **Perovskites with TM**: LaFeO3, LaMnO3, SrTiO3

**When NOT to use:**
- Simple metals (Al, Cu, Au) -- no localized electrons
- sp-bonded semiconductors (Si, GaAs) -- no correlation issue
- Molecular systems without TM centers
- When U value is unknown and you cannot compute it self-consistently

## Prerequisites

- Quantum ESPRESSO 7.5 with `pw.x` and `hp.x`
- SSSP or PseudoDojo pseudopotentials (PAW or USPP)
- Understanding of which atomic species need U correction

## Common U Values for 3d Transition Metals (eV)

These are typical literature values with PBE. **Always prefer self-consistent U from hp.x for publication-quality work.**

| Element | Typical U (eV) | Common Systems | Notes |
|---------|----------------|----------------|-------|
| Ti (3d) | 2.0 -- 4.0 | TiO2, SrTiO3 | U ~ 3.0 for TiO2 anatase |
| V (3d)  | 2.0 -- 4.0 | VO2, V2O5 | U ~ 3.25 for VO2 |
| Cr (3d) | 2.5 -- 4.0 | Cr2O3 | U ~ 3.5 typical |
| Mn (3d) | 3.0 -- 5.0 | MnO, LaMnO3 | U ~ 3.9 for MnO |
| Fe (3d) | 3.0 -- 5.5 | Fe2O3, FeO, LiFePO4 | U ~ 4.3 for Fe2O3 |
| Co (3d) | 3.0 -- 5.5 | CoO, LiCoO2 | U ~ 3.3 for CoO |
| Ni (3d) | 5.0 -- 7.0 | NiO, LiNiO2 | U ~ 6.4 for NiO |
| Cu (3d) | 4.0 -- 8.0 | CuO, cuprates | U ~ 5.0 -- 7.0 |

## Detailed Steps

### Example System: NiO (antiferromagnetic Mott insulator)

NiO is a classic test case: GGA incorrectly predicts it as a metal or small-gap semiconductor, while experimentally it is an insulator with a gap of ~4.3 eV.

### Step 1: Generate the Structure

```python
#!/usr/bin/env python3
"""Generate NiO structure files for DFT+U calculations."""

from pymatgen.core import Structure, Lattice
import numpy as np

# NiO rocksalt structure (conventional cell, AFM-II ordering)
# Experimental lattice parameter: a = 4.177 Angstrom
# For AFM-II, we use a rhombohedral cell with 4 atoms (2 Ni + 2 O)
# or a conventional cubic cell with magnetic ordering along [111]

# Use the conventional cubic cell (4 atoms)
a = 4.177  # Angstrom, experimental

lattice = Lattice.cubic(a)
species = ["Ni", "Ni", "O", "O"]
coords = [
    [0.0, 0.0, 0.0],  # Ni1 (spin up)
    [0.5, 0.5, 0.5],  # Ni2 (spin down, but we handle this in QE input)
    [0.5, 0.0, 0.0],  # O
    [0.0, 0.5, 0.0],  # O  (note: in rocksalt all O are equivalent)
]

# Actually rocksalt has Ni at (0,0,0) and O at (0.5,0,0) etc.
# Full rocksalt conventional cell: 4 Ni + 4 O
species_full = ["Ni", "Ni", "Ni", "Ni", "O", "O", "O", "O"]
coords_full = [
    [0.0, 0.0, 0.0],
    [0.5, 0.5, 0.0],
    [0.5, 0.0, 0.5],
    [0.0, 0.5, 0.5],
    [0.5, 0.0, 0.0],
    [0.0, 0.5, 0.0],
    [0.0, 0.0, 0.5],
    [0.5, 0.5, 0.5],
]

structure = Structure(lattice, species_full, coords_full)
print(f"Structure: {structure}")
print(f"Space group: rocksalt (Fm-3m)")
print(f"Number of atoms: {len(structure)}")
print(f"Lattice parameter: {a} Angstrom")
```

### Step 2: SCF Calculation WITHOUT U (reference)

Save as `nio_no_u.in`:

```
&CONTROL
    calculation   = 'scf'
    prefix        = 'nio_no_u'
    outdir        = './tmp_no_u/'
    pseudo_dir    = './pseudo/'
    tprnfor       = .true.
    tstress       = .true.
    verbosity     = 'high'
/

&SYSTEM
    ibrav         = 0
    nat           = 8
    ntyp          = 3
    ecutwfc       = 70.0
    ecutrho       = 560.0
    occupations   = 'smearing'
    smearing      = 'mv'
    degauss       = 0.01
    nspin         = 2
    starting_magnetization(1) =  0.8
    starting_magnetization(2) = -0.8
    starting_magnetization(3) =  0.0
/

&ELECTRONS
    conv_thr      = 1.0d-8
    mixing_beta   = 0.3
    mixing_mode   = 'plain'
    electron_maxstep = 200
/

ATOMIC_SPECIES
  Ni1  58.6934  Ni.pbe-spn-kjpaw_psl.1.0.0.UPF
  Ni2  58.6934  Ni.pbe-spn-kjpaw_psl.1.0.0.UPF
  O    15.9994  O.pbe-n-kjpaw_psl.1.0.0.UPF

CELL_PARAMETERS {angstrom}
  4.177  0.000  0.000
  0.000  4.177  0.000
  0.000  0.000  4.177

ATOMIC_POSITIONS {crystal}
  Ni1  0.000  0.000  0.000
  Ni2  0.500  0.500  0.000
  Ni1  0.500  0.000  0.500
  Ni2  0.000  0.500  0.500
  O    0.500  0.000  0.000
  O    0.000  0.500  0.000
  O    0.000  0.000  0.500
  O    0.500  0.500  0.500

K_POINTS {automatic}
  6 6 6 0 0 0
```

Note: We use two Ni types (Ni1, Ni2) with opposite starting magnetization to initialize the AFM-II order. They share the same pseudopotential file.

Run:
```bash
pw.x -npool 2 < nio_no_u.in > nio_no_u.out
```

### Step 3: SCF Calculation WITH U

Save as `nio_with_u.in`:

```
&CONTROL
    calculation   = 'scf'
    prefix        = 'nio_with_u'
    outdir        = './tmp_with_u/'
    pseudo_dir    = './pseudo/'
    tprnfor       = .true.
    tstress       = .true.
    verbosity     = 'high'
/

&SYSTEM
    ibrav         = 0
    nat           = 8
    ntyp          = 3
    ecutwfc       = 70.0
    ecutrho       = 560.0
    occupations   = 'smearing'
    smearing      = 'mv'
    degauss       = 0.01
    nspin         = 2
    starting_magnetization(1) =  0.8
    starting_magnetization(2) = -0.8
    starting_magnetization(3) =  0.0
    ! --- Hubbard U parameters ---
    lda_plus_u    = .true.
    lda_plus_u_kind = 0
    Hubbard_U(1)  = 6.4
    Hubbard_U(2)  = 6.4
/

&ELECTRONS
    conv_thr      = 1.0d-8
    mixing_beta   = 0.3
    mixing_mode   = 'plain'
    electron_maxstep = 200
/

ATOMIC_SPECIES
  Ni1  58.6934  Ni.pbe-spn-kjpaw_psl.1.0.0.UPF
  Ni2  58.6934  Ni.pbe-spn-kjpaw_psl.1.0.0.UPF
  O    15.9994  O.pbe-n-kjpaw_psl.1.0.0.UPF

CELL_PARAMETERS {angstrom}
  4.177  0.000  0.000
  0.000  4.177  0.000
  0.000  0.000  4.177

ATOMIC_POSITIONS {crystal}
  Ni1  0.000  0.000  0.000
  Ni2  0.500  0.500  0.000
  Ni1  0.500  0.000  0.500
  Ni2  0.000  0.500  0.500
  O    0.500  0.000  0.000
  O    0.000  0.500  0.000
  O    0.000  0.000  0.500
  O    0.500  0.500  0.500

K_POINTS {automatic}
  6 6 6 0 0 0
```

Key parameters explained:
- `lda_plus_u = .true.` -- Activates the DFT+U correction
- `lda_plus_u_kind = 0` -- Simplified (rotationally invariant) Dudarev formulation: E_U = (U-J)/2 * sum[Tr(n) - Tr(n*n)]. Only (U-J) matters; J is set to 0 by default.
- `Hubbard_U(1) = 6.4` -- U value in eV for species type 1 (Ni1). Applied to the 3d orbitals.
- `Hubbard_U(2) = 6.4` -- Same U for Ni2 (same element, different magnetic sublattice)

Run:
```bash
pw.x -npool 2 < nio_with_u.in > nio_with_u.out
```

### Step 3b: QE 7.x New-Style Hubbard Card (Alternative Syntax)

QE 7.x introduced a new `HUBBARD` card that replaces the old `lda_plus_u` keywords. This is the **preferred** syntax for QE >= 7.0 and is **required** for `hp.x`:

```
&SYSTEM
    ibrav         = 0
    nat           = 8
    ntyp          = 3
    ecutwfc       = 70.0
    ecutrho       = 560.0
    occupations   = 'smearing'
    smearing      = 'mv'
    degauss       = 0.01
    nspin         = 2
    starting_magnetization(1) =  0.8
    starting_magnetization(2) = -0.8
    starting_magnetization(3) =  0.0
/

... (other namelists same as above) ...

HUBBARD {ortho-atomic}
U Ni1-3d 6.4
U Ni2-3d 6.4
```

Projection types:
- `ortho-atomic` -- Lowdin-orthogonalized atomic orbitals (recommended, default in new syntax)
- `atomic` -- Non-orthogonalized atomic orbitals (legacy default)
- `norm-atomic` -- Normalized atomic orbitals

### Step 4: Self-Consistent U from hp.x (Linear Response)

The gold standard for DFT+U is to compute U self-consistently using density-functional perturbation theory via `hp.x`. This avoids empirical U values.

**Workflow: SCF with initial U guess --> hp.x --> extract new U --> SCF with new U --> repeat until converged.**

#### Step 4a: Initial SCF with a guess U (using new HUBBARD card)

Save as `nio_scf_hp.in`:

```
&CONTROL
    calculation   = 'scf'
    prefix        = 'nio'
    outdir        = './tmp_hp/'
    pseudo_dir    = './pseudo/'
    verbosity     = 'high'
/

&SYSTEM
    ibrav         = 0
    nat           = 8
    ntyp          = 3
    ecutwfc       = 70.0
    ecutrho       = 560.0
    occupations   = 'smearing'
    smearing      = 'mv'
    degauss       = 0.01
    nspin         = 2
    starting_magnetization(1) =  0.8
    starting_magnetization(2) = -0.8
    starting_magnetization(3) =  0.0
/

&ELECTRONS
    conv_thr      = 1.0d-8
    mixing_beta   = 0.3
    electron_maxstep = 200
/

ATOMIC_SPECIES
  Ni1  58.6934  Ni.pbe-spn-kjpaw_psl.1.0.0.UPF
  Ni2  58.6934  Ni.pbe-spn-kjpaw_psl.1.0.0.UPF
  O    15.9994  O.pbe-n-kjpaw_psl.1.0.0.UPF

CELL_PARAMETERS {angstrom}
  4.177  0.000  0.000
  0.000  4.177  0.000
  0.000  0.000  4.177

ATOMIC_POSITIONS {crystal}
  Ni1  0.000  0.000  0.000
  Ni2  0.500  0.500  0.000
  Ni1  0.500  0.000  0.500
  Ni2  0.000  0.500  0.500
  O    0.500  0.000  0.000
  O    0.000  0.500  0.000
  O    0.000  0.000  0.500
  O    0.500  0.500  0.500

K_POINTS {automatic}
  6 6 6 0 0 0

HUBBARD {ortho-atomic}
U Ni1-3d 5.0
U Ni2-3d 5.0
```

```bash
pw.x -npool 2 < nio_scf_hp.in > nio_scf_hp.out
```

#### Step 4b: Run hp.x to compute U

Save as `nio_hp.in`:

```
&inputhp
    prefix       = 'nio'
    outdir       = './tmp_hp/'
    nq1          = 2
    nq2          = 2
    nq3          = 2
    conv_thr_chi = 1.0d-5
    iverbosity   = 2
/
```

Key hp.x parameters:
- `nq1, nq2, nq3` -- q-point grid for the perturbation (2x2x2 is a reasonable starting point; use 3x3x3 or 4x4x4 for convergence tests)
- `conv_thr_chi` -- Convergence threshold for the response function
- `iverbosity` -- Higher values give more output detail

```bash
hp.x -npool 2 < nio_hp.in > nio_hp.out
```

hp.x will produce a file called `./tmp_hp/HP/nio.Hubbard_parameters.dat` containing the computed U values.

#### Step 4c: Parse hp.x Output and Iterate

```python
#!/usr/bin/env python3
"""Parse hp.x output and check self-consistency of U values."""

import re
import subprocess
import os

def parse_hp_output(hp_out_file):
    """Extract computed Hubbard U values from hp.x output."""
    u_values = {}
    with open(hp_out_file, 'r') as f:
        content = f.read()

    # Look for the Hubbard parameters summary
    # hp.x prints a table like:
    # site n.   type surface   Hubbard U (eV)
    #   1       Ni1             X.XXX
    pattern = r'(\d+)\s+(\w+)\s+[\d.]+\s+([\d.]+)'
    in_table = False
    for line in content.split('\n'):
        if 'Hubbard U (eV)' in line:
            in_table = True
            continue
        if in_table and line.strip():
            match = re.match(r'\s*(\d+)\s+(\S+)\s+\S+\s+([\d.]+)', line)
            if match:
                site = int(match.group(1))
                species = match.group(2)
                u_val = float(match.group(3))
                u_values[site] = {'species': species, 'U': u_val}
            elif '---' in line or line.strip() == '':
                continue
            else:
                in_table = False

    return u_values

def check_convergence(u_old, u_new, threshold=0.05):
    """Check if U values are converged within threshold (eV)."""
    converged = True
    for site in u_new:
        species = u_new[site]['species']
        new_u = u_new[site]['U']
        old_u = u_old.get(site, {}).get('U', 0.0)
        diff = abs(new_u - old_u)
        print(f"  Site {site} ({species}): U_old = {old_u:.3f}, U_new = {new_u:.3f}, "
              f"diff = {diff:.3f} eV")
        if diff > threshold:
            converged = False
    return converged

# Example self-consistent loop (pseudocode -- adapt paths as needed)
def self_consistent_u_loop(max_iter=5, threshold=0.05):
    """
    Iterate SCF + hp.x until U values are self-consistent.

    In practice, 2-4 iterations are usually sufficient.
    """
    u_current = {'Ni1': 5.0, 'Ni2': 5.0}  # Initial guess

    for iteration in range(max_iter):
        print(f"\n=== Iteration {iteration + 1} ===")
        print(f"Current U values: {u_current}")

        # 1. Write pw.x input with current U values
        write_pw_input('nio_scf_hp.in', u_current)

        # 2. Run SCF
        os.system('pw.x -npool 2 < nio_scf_hp.in > nio_scf_hp.out')

        # 3. Run hp.x
        os.system('hp.x -npool 2 < nio_hp.in > nio_hp.out')

        # 4. Parse new U values
        u_new = parse_hp_output('nio_hp.out')

        # 5. Check convergence
        u_old_dict = {1: {'species': 'Ni1', 'U': u_current['Ni1']},
                      2: {'species': 'Ni2', 'U': u_current['Ni2']}}
        if check_convergence(u_old_dict, u_new, threshold):
            print(f"\nU values converged after {iteration + 1} iterations!")
            return u_new
        else:
            # Update U values for next iteration
            for site in u_new:
                sp = u_new[site]['species']
                u_current[sp] = u_new[site]['U']

    print(f"\nWarning: U not converged after {max_iter} iterations.")
    return u_new

def write_pw_input(filename, u_values):
    """Write pw.x input file with given U values."""
    template = f"""&CONTROL
    calculation   = 'scf'
    prefix        = 'nio'
    outdir        = './tmp_hp/'
    pseudo_dir    = './pseudo/'
    verbosity     = 'high'
/

&SYSTEM
    ibrav         = 0
    nat           = 8
    ntyp          = 3
    ecutwfc       = 70.0
    ecutrho       = 560.0
    occupations   = 'smearing'
    smearing      = 'mv'
    degauss       = 0.01
    nspin         = 2
    starting_magnetization(1) =  0.8
    starting_magnetization(2) = -0.8
    starting_magnetization(3) =  0.0
/

&ELECTRONS
    conv_thr      = 1.0d-8
    mixing_beta   = 0.3
    electron_maxstep = 200
/

ATOMIC_SPECIES
  Ni1  58.6934  Ni.pbe-spn-kjpaw_psl.1.0.0.UPF
  Ni2  58.6934  Ni.pbe-spn-kjpaw_psl.1.0.0.UPF
  O    15.9994  O.pbe-n-kjpaw_psl.1.0.0.UPF

CELL_PARAMETERS {{angstrom}}
  4.177  0.000  0.000
  0.000  4.177  0.000
  0.000  0.000  4.177

ATOMIC_POSITIONS {{crystal}}
  Ni1  0.000  0.000  0.000
  Ni2  0.500  0.500  0.000
  Ni1  0.500  0.000  0.500
  Ni2  0.000  0.500  0.500
  O    0.500  0.000  0.000
  O    0.000  0.500  0.000
  O    0.000  0.000  0.500
  O    0.500  0.500  0.500

K_POINTS {{automatic}}
  6 6 6 0 0 0

HUBBARD {{ortho-atomic}}
U Ni1-3d {u_values['Ni1']:.4f}
U Ni2-3d {u_values['Ni2']:.4f}
"""
    with open(filename, 'w') as f:
        f.write(template)
```

### Step 5: DOS Calculation and Post-Processing

After SCF converges (both with and without U), compute the projected DOS:

#### NSCF for DOS (save as `nio_nscf_u.in`):

```
&CONTROL
    calculation   = 'nscf'
    prefix        = 'nio_with_u'
    outdir        = './tmp_with_u/'
    pseudo_dir    = './pseudo/'
    verbosity     = 'high'
/

&SYSTEM
    ibrav         = 0
    nat           = 8
    ntyp          = 3
    ecutwfc       = 70.0
    ecutrho       = 560.0
    occupations   = 'tetrahedra'
    nspin         = 2
    lda_plus_u    = .true.
    lda_plus_u_kind = 0
    Hubbard_U(1)  = 6.4
    Hubbard_U(2)  = 6.4
/

&ELECTRONS
    conv_thr      = 1.0d-8
/

ATOMIC_SPECIES
  Ni1  58.6934  Ni.pbe-spn-kjpaw_psl.1.0.0.UPF
  Ni2  58.6934  Ni.pbe-spn-kjpaw_psl.1.0.0.UPF
  O    15.9994  O.pbe-n-kjpaw_psl.1.0.0.UPF

CELL_PARAMETERS {angstrom}
  4.177  0.000  0.000
  0.000  4.177  0.000
  0.000  0.000  4.177

ATOMIC_POSITIONS {crystal}
  Ni1  0.000  0.000  0.000
  Ni2  0.500  0.500  0.000
  Ni1  0.500  0.000  0.500
  Ni2  0.000  0.500  0.500
  O    0.500  0.000  0.000
  O    0.000  0.500  0.000
  O    0.000  0.000  0.500
  O    0.500  0.500  0.500

K_POINTS {automatic}
  12 12 12 0 0 0
```

#### Projected DOS input (save as `nio_pdos_u.in`):

```
&PROJWFC
    prefix  = 'nio_with_u'
    outdir  = './tmp_with_u/'
    filpdos = 'nio_with_u_pdos'
    Emin    = -15.0
    Emax    = 10.0
    DeltaE  = 0.05
/
```

Run:
```bash
pw.x -npool 2 < nio_nscf_u.in > nio_nscf_u.out
projwfc.x < nio_pdos_u.in > nio_pdos_u.out
```

Repeat analogously for the no-U calculation (`nio_no_u` prefix, `tmp_no_u` directory, no Hubbard keywords).

### Step 6: Python Post-Processing -- Compare DOS With and Without U

```python
#!/usr/bin/env python3
"""
Compare DOS of NiO with and without Hubbard U correction.
Reads QE projwfc.x output files and generates publication-quality plots.
"""

import numpy as np
import matplotlib.pyplot as plt
import glob
import os


def read_dos_file(filename):
    """
    Read a QE DOS/PDOS file.
    Returns energy array and DOS columns (up, down if spin-polarized).
    """
    data = np.loadtxt(filename, comments='#')
    energy = data[:, 0]
    # For spin-polarized: col 1 = DOS up, col 2 = DOS down
    # For total DOS: col 1 = DOS, col 2 = integrated DOS
    return energy, data[:, 1:]


def read_pdos_files(prefix, outdir='.'):
    """
    Read projected DOS files from projwfc.x output.
    Returns dict with species/orbital-resolved PDOS.
    """
    pdos_data = {}
    pattern = os.path.join(outdir, f'{prefix}*')
    files = sorted(glob.glob(pattern))

    for f in files:
        basename = os.path.basename(f)
        if 'tot' in basename:
            continue  # Skip total DOS file for now

        try:
            data = np.loadtxt(f, comments='#')
            energy = data[:, 0]
            # Columns: E, ldos_up, ldos_down, pdos_up, pdos_down
            pdos_data[basename] = {
                'energy': energy,
                'ldos_up': data[:, 1] if data.shape[1] > 1 else None,
                'ldos_down': data[:, 2] if data.shape[1] > 2 else None,
            }
        except Exception as e:
            print(f"Warning: could not read {f}: {e}")

    return pdos_data


def aggregate_pdos_by_element(prefix, element_label, orbital='d', outdir='.'):
    """
    Sum PDOS over all atoms of a given element and orbital character.

    Parameters
    ----------
    prefix : str
        PDOS file prefix from projwfc.x
    element_label : str
        Element label (e.g., 'Ni1')
    orbital : str
        Orbital character to select: 's', 'p', 'd', 'f'
    outdir : str
        Directory containing PDOS files

    Returns
    -------
    energy, pdos_up, pdos_down : numpy arrays
    """
    orbital_l = {'s': 0, 'p': 1, 'd': 2, 'f': 3}
    l_val = orbital_l[orbital]

    pattern = os.path.join(outdir, f'{prefix}.pdos_atm#*({element_label})_wfc#{l_val}*')
    files = sorted(glob.glob(pattern))

    if not files:
        # Try alternative naming pattern
        pattern = os.path.join(outdir, f'{prefix}*{element_label}*')
        files = sorted(glob.glob(pattern))
        files = [f for f in files if f'{orbital}' in f.lower()]

    if not files:
        print(f"Warning: no PDOS files found for {element_label} {orbital}")
        return None, None, None

    pdos_up_total = None
    pdos_down_total = None
    energy = None

    for f in files:
        data = np.loadtxt(f, comments='#')
        if energy is None:
            energy = data[:, 0]
            pdos_up_total = np.zeros_like(energy)
            pdos_down_total = np.zeros_like(energy)

        # Sum all orbital components (columns after energy and ldos)
        # Typically: E, ldos_up, ldos_down, pdos1_up, pdos1_down, ...
        n_cols = data.shape[1]
        for col in range(3, n_cols, 2):  # pdos columns (up)
            pdos_up_total += data[:, col]
        for col in range(4, n_cols, 2):  # pdos columns (down)
            pdos_down_total += data[:, col]

    return energy, pdos_up_total, pdos_down_total


def plot_dos_comparison():
    """
    Create a publication-quality comparison of NiO DOS with and without U.
    """
    fig, axes = plt.subplots(2, 1, figsize=(8, 10), sharex=True)

    # ---- Panel (a): Without U ----
    ax = axes[0]
    ax.set_title('NiO: PBE (no Hubbard U)', fontsize=14, fontweight='bold')

    # Read total DOS
    try:
        e_no_u, dos_no_u = read_dos_file('nio_no_u_pdos.pdos_tot')
        ax.fill_between(e_no_u, dos_no_u[:, 0], alpha=0.2, color='gray', label='Total (up)')
        ax.fill_between(e_no_u, -dos_no_u[:, 1], alpha=0.2, color='gray', label='Total (down)')
    except FileNotFoundError:
        print("Total DOS file not found for no-U calculation. Using synthetic data for demo.")
        e_no_u = np.linspace(-15, 10, 500)
        # Synthetic: metallic (no gap)
        dos_up = np.exp(-(e_no_u + 2)**2 / 2) + 0.8 * np.exp(-(e_no_u - 0.5)**2 / 0.5)
        dos_down = np.exp(-(e_no_u + 1.5)**2 / 2) + 0.7 * np.exp(-(e_no_u + 0.2)**2 / 0.5)
        ax.fill_between(e_no_u, dos_up, alpha=0.2, color='gray', label='Total (up)')
        ax.fill_between(e_no_u, -dos_down, alpha=0.2, color='gray', label='Total (down)')

        # Ni-3d PDOS (no gap -- overlapping with Fermi level)
        ni_d_up = 0.9 * np.exp(-(e_no_u + 2)**2 / 1.5) + 0.7 * np.exp(-(e_no_u - 0.3)**2 / 0.3)
        ni_d_down = 0.8 * np.exp(-(e_no_u + 1.5)**2 / 1.5) + 0.6 * np.exp(-(e_no_u + 0.1)**2 / 0.3)
        ax.plot(e_no_u, ni_d_up, 'b-', linewidth=1.5, label='Ni 3d (up)')
        ax.plot(e_no_u, -ni_d_down, 'r-', linewidth=1.5, label='Ni 3d (down)')

        # O-2p PDOS
        o_p_up = 0.5 * np.exp(-(e_no_u + 5)**2 / 3)
        o_p_down = 0.5 * np.exp(-(e_no_u + 4.5)**2 / 3)
        ax.plot(e_no_u, o_p_up, 'g--', linewidth=1.5, label='O 2p (up)')
        ax.plot(e_no_u, -o_p_down, 'g--', linewidth=1.5, alpha=0.7)

    ax.axvline(x=0, color='k', linestyle='--', linewidth=0.8, label='$E_F$')
    ax.axhline(y=0, color='k', linewidth=0.5)
    ax.set_ylabel('DOS (states/eV)', fontsize=12)
    ax.legend(loc='upper right', fontsize=9)
    ax.set_xlim(-15, 10)
    ax.text(0.02, 0.95, '(a)', transform=ax.transAxes, fontsize=14,
            fontweight='bold', va='top')
    ax.text(1, ax.get_ylim()[1] * 0.8, 'No gap!\nIncorrect', fontsize=11,
            color='red', ha='center', style='italic')

    # ---- Panel (b): With U ----
    ax = axes[1]
    ax.set_title('NiO: PBE+U (U = 6.4 eV on Ni-3d)', fontsize=14, fontweight='bold')

    try:
        e_u, dos_u = read_dos_file('nio_with_u_pdos.pdos_tot')
        ax.fill_between(e_u, dos_u[:, 0], alpha=0.2, color='gray', label='Total (up)')
        ax.fill_between(e_u, -dos_u[:, 1], alpha=0.2, color='gray', label='Total (down)')
    except FileNotFoundError:
        print("Total DOS file not found for +U calculation. Using synthetic data for demo.")
        e_u = np.linspace(-15, 10, 500)
        # Synthetic: insulating (gap ~ 4 eV)
        dos_up_u = np.exp(-(e_u + 3)**2 / 2) + 0.6 * np.exp(-(e_u - 3.5)**2 / 1)
        dos_down_u = np.exp(-(e_u + 2.5)**2 / 2) + 0.5 * np.exp(-(e_u - 4)**2 / 1)
        ax.fill_between(e_u, dos_up_u, alpha=0.2, color='gray', label='Total (up)')
        ax.fill_between(e_u, -dos_down_u, alpha=0.2, color='gray', label='Total (down)')

        # Ni-3d PDOS (split by U, gap opens)
        ni_d_up_u = 0.8 * np.exp(-(e_u + 3)**2 / 1) + 0.5 * np.exp(-(e_u - 4)**2 / 0.8)
        ni_d_down_u = 0.7 * np.exp(-(e_u + 2.5)**2 / 1) + 0.4 * np.exp(-(e_u - 4.5)**2 / 0.8)
        ax.plot(e_u, ni_d_up_u, 'b-', linewidth=1.5, label='Ni 3d (up)')
        ax.plot(e_u, -ni_d_down_u, 'r-', linewidth=1.5, label='Ni 3d (down)')

        # O-2p PDOS
        o_p_up_u = 0.5 * np.exp(-(e_u + 6)**2 / 3)
        o_p_down_u = 0.5 * np.exp(-(e_u + 5.5)**2 / 3)
        ax.plot(e_u, o_p_up_u, 'g--', linewidth=1.5, label='O 2p (up)')
        ax.plot(e_u, -o_p_down_u, 'g--', linewidth=1.5, alpha=0.7)

        # Annotate the gap
        ax.annotate('', xy=(0.3, 0.05), xytext=(2.5, 0.05),
                     arrowprops=dict(arrowstyle='<->', color='purple', lw=2))
        ax.text(1.4, 0.15, 'Gap ~ 4 eV', fontsize=11, color='purple',
                ha='center', fontweight='bold')

    ax.axvline(x=0, color='k', linestyle='--', linewidth=0.8, label='$E_F$')
    ax.axhline(y=0, color='k', linewidth=0.5)
    ax.set_xlabel('Energy (eV)', fontsize=12)
    ax.set_ylabel('DOS (states/eV)', fontsize=12)
    ax.legend(loc='upper right', fontsize=9)
    ax.set_xlim(-15, 10)
    ax.text(0.02, 0.95, '(b)', transform=ax.transAxes, fontsize=14,
            fontweight='bold', va='top')

    plt.tight_layout()
    plt.savefig('nio_dos_comparison_U.png', dpi=300, bbox_inches='tight')
    plt.savefig('nio_dos_comparison_U.pdf', bbox_inches='tight')
    print("Saved: nio_dos_comparison_U.png, nio_dos_comparison_U.pdf")
    plt.show()


def extract_results_from_output(filename):
    """
    Extract key results (energy, magnetic moments, band gap) from pw.x output.
    """
    results = {}
    with open(filename, 'r') as f:
        content = f.read()

    # Total energy
    match = re.search(r'!\s+total energy\s+=\s+([-\d.]+)\s+Ry', content)
    if match:
        results['total_energy_ry'] = float(match.group(1))
        results['total_energy_ev'] = float(match.group(1)) * 13.6057

    # Magnetic moments
    mag_moments = re.findall(r'atom:\s+\d+\s+charge:\s+[\d.]+\s+magn:\s+([-\d.]+)', content)
    if mag_moments:
        results['magnetic_moments'] = [float(m) for m in mag_moments]

    # Total magnetization
    match = re.search(r'total magnetization\s+=\s+([-\d.]+)', content)
    if match:
        results['total_magnetization'] = float(match.group(1))

    # Absolute magnetization
    match = re.search(r'absolute magnetization\s+=\s+([-\d.]+)', content)
    if match:
        results['absolute_magnetization'] = float(match.group(1))

    # Highest occupied, lowest unoccupied
    match = re.search(r'highest occupied, lowest unoccupied level \(ev\):\s+([-\d.]+)\s+([-\d.]+)',
                       content)
    if match:
        results['homo'] = float(match.group(1))
        results['lumo'] = float(match.group(2))
        results['band_gap'] = results['lumo'] - results['homo']

    return results


if __name__ == '__main__':
    import re

    print("=" * 60)
    print("NiO DFT+U Post-Processing")
    print("=" * 60)

    # Try to extract results from actual QE output
    for label, fname in [('PBE (no U)', 'nio_no_u.out'),
                          ('PBE+U (U=6.4)', 'nio_with_u.out')]:
        print(f"\n--- {label} ---")
        if os.path.exists(fname):
            results = extract_results_from_output(fname)
            for k, v in results.items():
                if isinstance(v, list):
                    print(f"  {k}: {[f'{x:.3f}' for x in v]}")
                elif isinstance(v, float):
                    print(f"  {k}: {v:.4f}")
                else:
                    print(f"  {k}: {v}")
        else:
            print(f"  Output file {fname} not found. Run QE first.")
            print(f"  Expected results for NiO:")
            if 'no U' in label:
                print(f"    Band gap: ~ 0.0-1.0 eV (severely underestimated)")
                print(f"    Ni magnetic moment: ~ 1.0-1.3 muB (underestimated)")
            else:
                print(f"    Band gap: ~ 3.5-4.5 eV (close to experimental 4.3 eV)")
                print(f"    Ni magnetic moment: ~ 1.6-1.9 muB (close to experimental 1.9)")

    # Generate comparison plot
    print("\nGenerating DOS comparison plot...")
    plot_dos_comparison()
```

### Step 7: Structural Relaxation With U

For a full DFT+U structural optimization (e.g., to get the equilibrium lattice parameter):

Save as `nio_relax_u.in`:

```
&CONTROL
    calculation   = 'vc-relax'
    prefix        = 'nio_relax_u'
    outdir        = './tmp_relax_u/'
    pseudo_dir    = './pseudo/'
    forc_conv_thr = 1.0d-4
    etot_conv_thr = 1.0d-6
/

&SYSTEM
    ibrav         = 0
    nat           = 8
    ntyp          = 3
    ecutwfc       = 70.0
    ecutrho       = 560.0
    occupations   = 'smearing'
    smearing      = 'mv'
    degauss       = 0.01
    nspin         = 2
    starting_magnetization(1) =  0.8
    starting_magnetization(2) = -0.8
    starting_magnetization(3) =  0.0
    lda_plus_u    = .true.
    lda_plus_u_kind = 0
    Hubbard_U(1)  = 6.4
    Hubbard_U(2)  = 6.4
/

&ELECTRONS
    conv_thr      = 1.0d-8
    mixing_beta   = 0.3
/

&IONS
    ion_dynamics  = 'bfgs'
/

&CELL
    cell_dynamics = 'bfgs'
    press         = 0.0
    press_conv_thr = 0.1
/

ATOMIC_SPECIES
  Ni1  58.6934  Ni.pbe-spn-kjpaw_psl.1.0.0.UPF
  Ni2  58.6934  Ni.pbe-spn-kjpaw_psl.1.0.0.UPF
  O    15.9994  O.pbe-n-kjpaw_psl.1.0.0.UPF

CELL_PARAMETERS {angstrom}
  4.177  0.000  0.000
  0.000  4.177  0.000
  0.000  0.000  4.177

ATOMIC_POSITIONS {crystal}
  Ni1  0.000  0.000  0.000
  Ni2  0.500  0.500  0.000
  Ni1  0.500  0.000  0.500
  Ni2  0.000  0.500  0.500
  O    0.500  0.000  0.000
  O    0.000  0.500  0.000
  O    0.000  0.000  0.500
  O    0.500  0.500  0.500

K_POINTS {automatic}
  6 6 6 0 0 0
```

```bash
pw.x -npool 2 < nio_relax_u.in > nio_relax_u.out
```

## Key Parameters

| Parameter | Description | Recommended |
|-----------|-------------|-------------|
| `lda_plus_u` | Enable DFT+U | `.true.` |
| `lda_plus_u_kind` | 0 = Dudarev (simplified), 1 = Liechtenstein (full) | 0 (most common) |
| `Hubbard_U(i)` | U value in eV for species i | From hp.x or literature |
| `Hubbard_J(i,j)` | J value (only for kind=1) | Usually 0 for kind=0 |
| Projection type | `ortho-atomic` (recommended), `atomic` (legacy) | `ortho-atomic` |
| `HUBBARD` card | New QE 7.x syntax, required for hp.x | Use for new calculations |
| hp.x `nq1,nq2,nq3` | q-grid for linear response | Start with 2x2x2, converge |
| hp.x `conv_thr_chi` | Convergence of response function | 1.0d-5 |

## Interpreting Results

### Effect of U on Physical Properties

| Property | Without U (PBE) | With U (PBE+U) | Experiment |
|----------|----------------|-----------------|------------|
| NiO band gap | 0.0 -- 1.0 eV | 3.5 -- 4.5 eV | 4.3 eV |
| NiO Ni moment | 1.0 -- 1.3 muB | 1.6 -- 1.9 muB | 1.9 muB |
| NiO lattice param | ~4.10 A | ~4.18 A | 4.177 A |
| Electronic character | Metal/small gap | Charge-transfer insulator | Insulator |

### What to Check

1. **Band gap**: DFT+U should open the gap in Mott insulators. Compare with experiment.
2. **Magnetic moments**: Should increase toward experimental values with appropriate U.
3. **Lattice parameter**: Usually improves slightly with U.
4. **Orbital occupations**: Check the occupation matrix printed in the output. Diagonal elements should be close to integer values (0 or 1) for strongly localized systems.
5. **Self-consistency of U**: If using hp.x, the computed U should not change by more than ~0.05 eV between iterations.

### Reading the Occupation Matrix

QE prints the occupation matrix for each Hubbard site. For NiO with Ni2+ (d8):

```
atom    1   Tr[ns(na)] (up, down, total) =   4.700  3.300  8.000
   spin  1
    eigenvalues:
      0.940  0.940  0.940  0.940  0.940
   spin  2
    eigenvalues:
      0.440  0.440  0.440  0.940  0.940
```

The eigenvalues of the occupation matrix should be close to 0 or 1 for a well-localized system. Values far from 0/1 suggest delocalization.

## Common Issues

1. **Wrong magnetic ground state**: Always test multiple initial spin configurations. AFM-II is the ground state of NiO; initializing FM may trap you in a metastable state.

2. **U too large**: Overly large U pushes d-states too far, creating an unphysical electronic structure. Always validate against experiment or use self-consistent U from hp.x.

3. **Convergence problems**: DFT+U can cause SCF convergence issues.
   - Reduce `mixing_beta` to 0.1 -- 0.2
   - Use `mixing_mode = 'local-TF'`
   - Start from a converged non-U calculation using `startingpot = 'file'`

4. **Multiple local minima**: DFT+U has a more rugged energy landscape. Different initial occupations can converge to different solutions. Use `starting_ns_eigenvalue` to control the initial occupation matrix if needed.

5. **Incompatible syntax**: Do not mix old-style (`lda_plus_u`, `Hubbard_U`) with new-style (`HUBBARD` card) in the same input. Pick one.

6. **hp.x requires HUBBARD card**: The `hp.x` code in QE 7.x only works with the new `HUBBARD` card syntax, not the old `lda_plus_u` keywords.

7. **U depends on the functional and pseudopotential**: A U value computed for PBE with one pseudopotential set is not transferable to PBEsol or a different PP. Always recompute.

8. **U on oxygen/ligands**: For charge-transfer insulators like NiO, some studies also apply a small U on O-2p (U_O ~ 0 -- 2 eV). This is system-dependent and not always necessary.
