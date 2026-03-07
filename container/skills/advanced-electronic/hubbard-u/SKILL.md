# DFT+U Calculations for Strongly Correlated Systems

## When to Use

Apply DFT+U when your system contains **strongly correlated electrons** -- partially filled d or f shells where standard GGA/LDA fails qualitatively:

- **Transition metal oxides**: NiO, Fe2O3, FeO, CoO, MnO, TiO2, VO2, LiFePO4
- **f-electron systems**: lanthanide/actinide oxides (CeO2, UO2, PuO2)
- **Mott insulators**: systems that GGA predicts as metallic but are experimentally insulating
- **Mixed-valence compounds**: Fe3O4, LiFePO4, LiCoO2 (battery cathodes)
- **Perovskites with TM ions**: LaFeO3, LaMnO3, SrTiO3, BiFeO3
- **Orbital ordering**: Jahn-Teller distorted systems (LaMnO3, KCuF3)

**When NOT to use:**
- Simple metals (Al, Cu, Au) -- no localized electrons
- sp-bonded semiconductors (Si, GaAs) -- no correlation issue
- Molecular systems without transition metal centers
- When U value is unknown and you cannot compute it self-consistently

## Method Selection

| Approach | QE Implementation | VASP Implementation | Accuracy | Cost |
|---|---|---|---|---|
| Dudarev (simplified) | `lda_plus_u_kind=0` or `HUBBARD` card | `LDAUTYPE=2` (default) | Good | Same as DFT |
| Liechtenstein (full) | `lda_plus_u_kind=1` | `LDAUTYPE=1` | Better for anisotropic | Same as DFT |
| Self-consistent U (hp.x) | `hp.x` linear response | N/A (use ACBN0 approach) | Best | 5-10x DFT |
| Literature U values | Manual specification | Manual specification | Acceptable | Same as DFT |

## Prerequisites

- Quantum ESPRESSO 7.5 with `pw.x` and `hp.x`
- SSSP or PseudoDojo pseudopotentials (PAW or USPP)
- Understanding of which atomic species need U correction
- For VASP: PAW potentials, INCAR with LDAU flags

## Common U Values for 3d Transition Metals (eV)

These are typical literature values with PBE. **Always prefer self-consistent U from hp.x for publication-quality work.**

| Element | Typical U (eV) | Common Systems | Notes |
|---------|----------------|----------------|-------|
| Ti (3d) | 2.0 -- 4.0 | TiO2, SrTiO3, BaTiO3 | U ~ 3.0 for TiO2 anatase |
| V (3d)  | 2.0 -- 4.0 | VO2, V2O5, LaVO3 | U ~ 3.25 for VO2 |
| Cr (3d) | 2.5 -- 4.0 | Cr2O3, CrI3 | U ~ 3.5 typical |
| Mn (3d) | 3.0 -- 5.0 | MnO, LaMnO3, MnO2 | U ~ 3.9 for MnO |
| Fe (3d) | 3.0 -- 5.5 | Fe2O3, FeO, LiFePO4 | U ~ 4.3 for Fe2O3 |
| Co (3d) | 3.0 -- 5.5 | CoO, LiCoO2, LaCoO3 | U ~ 3.3 for CoO |
| Ni (3d) | 5.0 -- 7.0 | NiO, LiNiO2, NiPS3 | U ~ 6.4 for NiO |
| Cu (3d) | 4.0 -- 8.0 | CuO, cuprate superconductors | U ~ 5.0 -- 7.0 |
| Ce (4f) | 3.0 -- 6.0 | CeO2, CeAlO3 | U ~ 4.5 for CeO2 |
| U (5f)  | 2.0 -- 5.0 | UO2, UN | U ~ 3.5 for UO2 |

## Detailed Steps

### Example System: NiO (Antiferromagnetic Mott Insulator)

NiO is a classic test case: GGA incorrectly predicts it as a metal or small-gap semiconductor, while experimentally it is an insulator with a gap of ~4.3 eV.

### Step 1: Generate the Structure

```python
#!/usr/bin/env python3
"""Generate NiO structure files for DFT+U calculations."""

from pymatgen.core import Structure, Lattice
import numpy as np

# NiO rocksalt structure (conventional cell)
# Experimental lattice parameter: a = 4.177 Angstrom
# For AFM-II ordering: use two Ni types with opposite spins
a = 4.177  # Angstrom, experimental

lattice = Lattice.cubic(a)

# Full rocksalt conventional cell: 4 Ni + 4 O
# Split Ni into two sublattices for AFM-II ordering
species_full = ["Ni", "Ni", "Ni", "Ni", "O", "O", "O", "O"]
coords_full = [
    [0.0, 0.0, 0.0],    # Ni sublattice 1 (spin up)
    [0.5, 0.5, 0.0],    # Ni sublattice 2 (spin down)
    [0.5, 0.0, 0.5],    # Ni sublattice 1 (spin up)
    [0.0, 0.5, 0.5],    # Ni sublattice 2 (spin down)
    [0.5, 0.0, 0.0],    # O
    [0.0, 0.5, 0.0],    # O
    [0.0, 0.0, 0.5],    # O
    [0.5, 0.5, 0.5],    # O
]

structure = Structure(lattice, species_full, coords_full)
print(f"Structure: {structure}")
print(f"Space group: rocksalt (Fm-3m)")
print(f"Number of atoms: {len(structure)}")
print(f"Lattice parameter: {a} Angstrom")

# Write CIF file
structure.to(filename="NiO_cubic.cif")
print("Saved NiO_cubic.cif")
```

### Step 2: Download Pseudopotentials

```python
#!/usr/bin/env python3
"""Download SSSP pseudopotentials for NiO."""

import subprocess
import os

os.makedirs("pseudo", exist_ok=True)

# SSSP Efficiency 1.3.0 PBE pseudopotentials
pp_urls = {
    "Ni.pbe-spn-kjpaw_psl.1.0.0.UPF":
        "https://pseudopotentials.quantum-espresso.org/upf_files/Ni.pbe-spn-kjpaw_psl.1.0.0.UPF",
    "O.pbe-n-kjpaw_psl.1.0.0.UPF":
        "https://pseudopotentials.quantum-espresso.org/upf_files/O.pbe-n-kjpaw_psl.1.0.0.UPF",
}

for fname, url in pp_urls.items():
    target = os.path.join("pseudo", fname)
    if not os.path.exists(target):
        print(f"Downloading {fname}...")
        subprocess.run(["wget", "-q", "-O", target, url], check=True)
        print(f"  Saved to {target}")
    else:
        print(f"  {fname} already exists.")
```

### Step 3: SCF Without U (Reference Calculation)

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

Two Ni types (Ni1, Ni2) with opposite starting magnetization initialize AFM-II order. They share the same pseudopotential file.

```bash
pw.x -npool 2 < nio_no_u.in > nio_no_u.out
```

### Step 4: SCF With Hubbard U (Old-Style Syntax)

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

Key parameters:
- `lda_plus_u = .true.` -- Activates the DFT+U correction
- `lda_plus_u_kind = 0` -- Dudarev (rotationally invariant) formulation: E_U = (U-J)/2 * sum[Tr(n) - Tr(n*n)]. Only the effective U_eff = U-J matters; J is set to 0 by default.
- `Hubbard_U(1) = 6.4` -- U value in eV for species type 1 (Ni1), applied to 3d orbitals
- `Hubbard_U(2) = 6.4` -- Same U for Ni2 (same element, different magnetic sublattice)

```bash
pw.x -npool 2 < nio_with_u.in > nio_with_u.out
```

### Step 4b: QE 7.x New-Style HUBBARD Card (Preferred Syntax)

QE 7.x introduced a `HUBBARD` card that replaces the old `lda_plus_u` keywords. This is the **preferred** syntax for QE >= 7.0 and is **required** for `hp.x`:

Replace the `lda_plus_u` lines in `&SYSTEM` and add after the `K_POINTS` block:

```
HUBBARD {ortho-atomic}
U Ni1-3d 6.4
U Ni2-3d 6.4
```

Projection types:
- `ortho-atomic` -- Lowdin-orthogonalized atomic orbitals (recommended)
- `atomic` -- Non-orthogonalized atomic orbitals (legacy default)
- `norm-atomic` -- Normalized atomic orbitals

### Step 4c: VASP INCAR for DFT+U

For VASP users, add to INCAR:

```
# DFT+U settings (Dudarev method, LDAUTYPE=2)
LDAU    = .TRUE.
LDAUTYPE = 2
LDAUL   = 2 -1       ! l quantum number: 2=d for Ni, -1=off for O
LDAUU   = 6.4 0.0    ! U values: 6.4 eV for Ni, 0 for O
LDAUJ   = 0.0 0.0    ! J values: 0 (Dudarev: only U-J matters)
LDAUPRINT = 2         ! Print occupation matrices

# Spin-polarized (for AFM)
ISPIN   = 2
MAGMOM  = 4*2.0 4*-2.0 4*0.0   ! Initial moments: Ni1 up, Ni2 down, O zero

# For Liechtenstein formulation (LDAUTYPE=1):
! LDAUTYPE = 1
! LDAUU   = 8.0 0.0   ! U parameter
! LDAUJ   = 1.0 0.0   ! J parameter (U_eff = U - J = 7.0)
```

VASP notes:
- Species order in LDAUL/LDAUU/LDAUJ must match the order in POSCAR/POTCAR
- `LDAUTYPE=2` (Dudarev) is most common; only U-J matters
- `LDAUTYPE=1` (Liechtenstein) treats U and J independently
- Set `LMAXMIX = 4` for d-electrons or `LMAXMIX = 6` for f-electrons to properly mix the on-site density matrix

### Step 5: Self-Consistent U from hp.x (Linear Response)

The gold standard for DFT+U is computing U self-consistently using density-functional perturbation theory via `hp.x`. This eliminates empirical U values.

**Workflow: SCF with initial U guess --> hp.x --> extract new U --> SCF with new U --> repeat until converged.**

#### Step 5a: Initial SCF (must use HUBBARD card)

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

#### Step 5b: Run hp.x

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
- `nq1, nq2, nq3` -- q-point grid for the perturbation (start with 2x2x2; test 3x3x3 or 4x4x4 for convergence)
- `conv_thr_chi` -- convergence threshold for the response function
- `iverbosity` -- higher values give more output detail

```bash
hp.x -npool 2 < nio_hp.in > nio_hp.out
```

hp.x produces `./tmp_hp/HP/nio.Hubbard_parameters.dat` with the computed U values.

#### Step 5c: Self-Consistent U Iteration Script

```python
#!/usr/bin/env python3
"""
Self-consistent Hubbard U loop: iterate SCF + hp.x until U converges.
Typically 2-4 iterations are sufficient.
"""

import re
import os
import subprocess


def parse_hp_output(hp_out_file):
    """Extract computed Hubbard U values from hp.x output."""
    u_values = {}
    with open(hp_out_file, 'r') as f:
        content = f.read()

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


def self_consistent_u_loop(max_iter=5, threshold=0.05):
    """Iterate SCF + hp.x until U values converge within threshold (eV)."""
    u_current = {'Ni1': 5.0, 'Ni2': 5.0}  # Initial guess

    for iteration in range(max_iter):
        print(f"\n=== Iteration {iteration + 1} ===")
        print(f"Current U values: {u_current}")

        # 1. Write pw.x input with current U
        write_pw_input('nio_scf_hp.in', u_current)

        # 2. Run SCF
        subprocess.run('pw.x -npool 2 < nio_scf_hp.in > nio_scf_hp.out',
                        shell=True, check=True)

        # 3. Run hp.x
        subprocess.run('hp.x -npool 2 < nio_hp.in > nio_hp.out',
                        shell=True, check=True)

        # 4. Parse new U values
        u_new = parse_hp_output('nio_hp.out')

        # 5. Check convergence
        converged = True
        for site in u_new:
            sp = u_new[site]['species']
            new_u = u_new[site]['U']
            old_u = u_current.get(sp, 0.0)
            diff = abs(new_u - old_u)
            print(f"  Site {site} ({sp}): U_old = {old_u:.3f}, "
                  f"U_new = {new_u:.3f}, diff = {diff:.3f} eV")
            if diff > threshold:
                converged = False

        if converged:
            print(f"\nU values converged after {iteration + 1} iterations!")
            return u_new

        # Update for next iteration
        for site in u_new:
            sp = u_new[site]['species']
            u_current[sp] = u_new[site]['U']

    print(f"\nWarning: U not converged after {max_iter} iterations.")
    return u_new


if __name__ == '__main__':
    self_consistent_u_loop()
```

### Step 6: Projected DOS Comparison

After SCF converges with and without U, compute projected DOS to visualize the effect.

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

```bash
pw.x -npool 2 < nio_nscf_u.in > nio_nscf_u.out
projwfc.x < nio_pdos_u.in > nio_pdos_u.out
```

Repeat for no-U calculation with `nio_no_u` prefix and `tmp_no_u` directory (remove Hubbard keywords).

### Step 7: Post-Processing -- Compare DOS With and Without U

```python
#!/usr/bin/env python3
"""
Compare DOS of NiO with and without Hubbard U correction.
Reads QE projwfc.x output or generates demonstration plot with typical values.
"""

import numpy as np
import matplotlib.pyplot as plt
import glob
import os
import re


def read_dos_file(filename):
    """Read a QE DOS/PDOS file. Returns energy and DOS columns."""
    data = np.loadtxt(filename, comments='#')
    return data[:, 0], data[:, 1:]


def extract_results(filename):
    """Extract key results from pw.x output."""
    results = {}
    if not os.path.exists(filename):
        return results
    with open(filename, 'r') as f:
        content = f.read()

    match = re.search(r'!\s+total energy\s+=\s+([-\d.]+)\s+Ry', content)
    if match:
        results['total_energy_ry'] = float(match.group(1))

    mag_moments = re.findall(
        r'atom:\s+\d+\s+charge:\s+[\d.]+\s+magn:\s+([-\d.]+)', content)
    if mag_moments:
        results['magnetic_moments'] = [float(m) for m in mag_moments]

    match = re.search(
        r'highest occupied, lowest unoccupied level \(ev\):\s+([-\d.]+)\s+([-\d.]+)',
        content)
    if match:
        results['band_gap'] = float(match.group(2)) - float(match.group(1))

    return results


def plot_dos_comparison():
    """Compare NiO DOS with and without U. Uses real data if available."""
    fig, axes = plt.subplots(2, 1, figsize=(8, 10), sharex=True)

    for idx, (label, prefix) in enumerate([
        ('PBE (no Hubbard U)', 'nio_no_u_pdos'),
        ('PBE+U (U = 6.4 eV on Ni-3d)', 'nio_with_u_pdos'),
    ]):
        ax = axes[idx]
        ax.set_title(f'NiO: {label}', fontsize=14, fontweight='bold')

        tot_file = f'{prefix}.pdos_tot'
        if os.path.exists(tot_file):
            e, dos = read_dos_file(tot_file)
            ax.fill_between(e, dos[:, 0], alpha=0.2, color='gray',
                            label='Total (up)')
            ax.fill_between(e, -dos[:, 1], alpha=0.2, color='gray',
                            label='Total (down)')
        else:
            print(f"  {tot_file} not found -- using synthetic data for demo.")
            e = np.linspace(-15, 10, 500)
            if idx == 0:  # No U: metallic
                dos_up = (np.exp(-(e + 2)**2 / 2)
                          + 0.8 * np.exp(-(e - 0.5)**2 / 0.5))
                dos_dn = (np.exp(-(e + 1.5)**2 / 2)
                          + 0.7 * np.exp(-(e + 0.2)**2 / 0.5))
            else:  # With U: insulating
                dos_up = (np.exp(-(e + 3)**2 / 2)
                          + 0.6 * np.exp(-(e - 3.5)**2 / 1))
                dos_dn = (np.exp(-(e + 2.5)**2 / 2)
                          + 0.5 * np.exp(-(e - 4)**2 / 1))
            ax.fill_between(e, dos_up, alpha=0.2, color='gray')
            ax.fill_between(e, -dos_dn, alpha=0.2, color='gray')

        ax.axvline(x=0, color='k', linestyle='--', linewidth=0.8, label='$E_F$')
        ax.axhline(y=0, color='k', linewidth=0.5)
        ax.set_ylabel('DOS (states/eV)', fontsize=12)
        ax.legend(loc='upper right', fontsize=9)
        ax.set_xlim(-15, 10)
        panel = '(a)' if idx == 0 else '(b)'
        ax.text(0.02, 0.95, panel, transform=ax.transAxes, fontsize=14,
                fontweight='bold', va='top')

    axes[1].set_xlabel('Energy (eV)', fontsize=12)
    plt.tight_layout()
    plt.savefig('nio_dos_comparison_U.png', dpi=300, bbox_inches='tight')
    plt.savefig('nio_dos_comparison_U.pdf', bbox_inches='tight')
    print("Saved: nio_dos_comparison_U.png, nio_dos_comparison_U.pdf")


if __name__ == '__main__':
    print("=" * 60)
    print("NiO DFT+U Post-Processing")
    print("=" * 60)

    for label, fname in [('PBE (no U)', 'nio_no_u.out'),
                          ('PBE+U (U=6.4)', 'nio_with_u.out')]:
        print(f"\n--- {label} ---")
        res = extract_results(fname)
        if res:
            for k, v in res.items():
                if isinstance(v, list):
                    print(f"  {k}: {[f'{x:.3f}' for x in v]}")
                elif isinstance(v, float):
                    print(f"  {k}: {v:.4f}")
        else:
            print(f"  Output file {fname} not found.")
            if 'no U' in label:
                print(f"    Expected: band gap ~ 0.0-1.0 eV, Ni moment ~ 1.0-1.3 muB")
            else:
                print(f"    Expected: band gap ~ 3.5-4.5 eV, Ni moment ~ 1.6-1.9 muB")

    plot_dos_comparison()
```

### Step 8: Structural Relaxation With U

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
| `lda_plus_u` | Enable DFT+U (old syntax) | `.true.` |
| `lda_plus_u_kind` | 0 = Dudarev, 1 = Liechtenstein | 0 (most common) |
| `Hubbard_U(i)` | U value in eV for species i (old syntax) | From hp.x or literature |
| `Hubbard_J(i,j)` | J value (only for kind=1) | Usually 0 for kind=0 |
| `HUBBARD` card | New QE 7.x syntax, required for hp.x | Preferred for new work |
| Projection type | `ortho-atomic`, `atomic`, `norm-atomic` | `ortho-atomic` |
| hp.x `nq1,nq2,nq3` | q-grid for linear response | Start 2x2x2, converge |
| hp.x `conv_thr_chi` | Convergence of response function | 1.0d-5 |
| VASP `LDAU` | Enable DFT+U | `.TRUE.` |
| VASP `LDAUTYPE` | 1 = Liechtenstein, 2 = Dudarev | 2 |
| VASP `LDAUL` | l quantum number per species | 2 for d, 3 for f, -1 for off |
| VASP `LDAUU` | U values per species (eV) | Literature or linear response |
| VASP `LDAUJ` | J values per species (eV) | 0 for LDAUTYPE=2 |
| VASP `LMAXMIX` | Max l for on-site density matrix | 4 (d), 6 (f) |

## Interpreting Results

### Effect of U on Physical Properties

| Property | Without U (PBE) | With U (PBE+U) | Experiment |
|----------|----------------|-----------------|------------|
| NiO band gap | 0.0 -- 1.0 eV | 3.5 -- 4.5 eV | 4.3 eV |
| NiO Ni moment | 1.0 -- 1.3 muB | 1.6 -- 1.9 muB | 1.9 muB |
| NiO lattice param | ~4.10 A | ~4.18 A | 4.177 A |
| Electronic character | Metal/small gap | Charge-transfer insulator | Insulator |
| Fe2O3 band gap | ~0.5 eV | ~2.0 eV | 2.2 eV |
| MnO Mn moment | ~3.5 muB | ~4.5 muB | 4.58 muB |

### What to Check

1. **Band gap**: DFT+U should open the gap in Mott insulators. Compare with experiment.
2. **Magnetic moments**: Should increase toward experimental values with appropriate U.
3. **Lattice parameter**: Usually improves slightly with U.
4. **Orbital occupations**: Check the occupation matrix printed in QE output. Diagonal elements should approach 0 or 1 for strongly localized systems.
5. **Self-consistency of U**: If using hp.x, computed U should not change by more than ~0.05 eV between iterations.

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

Eigenvalues near 0 or 1 indicate a well-localized system. Values far from 0/1 suggest delocalization or wrong U.

## Common Issues

| Issue | Symptom | Solution |
|-------|---------|----------|
| Wrong magnetic ground state | Energy higher than expected; wrong moment pattern | Test multiple initial spin configurations (FM, AFM-I, AFM-II) |
| U too large | Unphysical gap, d-states pushed too far | Validate against experiment or use self-consistent U from hp.x |
| SCF convergence failure | oscillating total energy, exceeds max steps | Reduce `mixing_beta` to 0.1--0.2; use `mixing_mode = 'local-TF'`; start from converged non-U wavefunction |
| Multiple local minima | Different runs give different energies/occupations | Use `starting_ns_eigenvalue` to control initial occupation matrix |
| Incompatible syntax | QE error on input parsing | Do not mix old-style (`lda_plus_u`) with new-style (`HUBBARD` card) |
| hp.x fails | hp.x requires HUBBARD card | Rewrite input using `HUBBARD {ortho-atomic}` syntax |
| Non-transferable U | Different U for different PPs or functionals | Recompute U for each functional/pseudopotential combination |
| Charge-transfer gap wrong | Gap character incorrect for charge-transfer insulators | Consider adding small U on O-2p (U_O ~ 0--2 eV) for systems like NiO |
| VASP LMAXMIX too low | Incorrect occupation matrix mixing | Set `LMAXMIX = 4` for d-electrons, `LMAXMIX = 6` for f-electrons |
