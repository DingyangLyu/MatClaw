# Bader Charge Analysis

Quantitative charge partitioning using Bader's atoms-in-molecules (AIM) method with Quantum ESPRESSO charge densities, plus Lowdin charge analysis as a simpler alternative.

## When to Use

- Quantify charge transfer between atoms (e.g., how much charge moves from metal to ligand).
- Estimate oxidation states from first principles.
- Analyze charge redistribution upon adsorption, doping, or defect formation.
- Compare ionic character across a series of compounds.

## Prerequisites

- Quantum ESPRESSO 7.5 (`pw.x`, `pp.x`, `projwfc.x`)
- Python: `numpy`, `matplotlib`, `ase`
- Bader executable from the Henkelman group (installable, see below)
- For PAW calculations: use `plot_num=21` (all-electron density) for accurate Bader analysis

## Detailed Steps

### Method 1: Bader Analysis (Full Workflow)

#### Step 1a: Install the Bader Program

The Henkelman group provides a standalone `bader` binary. Install it inside the container:

```bash
# Option A: Download precompiled binary
cd /opt
wget http://theory.cm.utexas.edu/henkelman/code/bader/download/bader_lnx_64.tar.gz
tar xzf bader_lnx_64.tar.gz
cp bader /usr/local/bin/
chmod +x /usr/local/bin/bader

# Option B: pip install (if a Python wrapper is available)
pip install pybader

# Option C: Build from source
wget http://theory.cm.utexas.edu/henkelman/code/bader/download/bader.tar.gz
tar xzf bader.tar.gz
cd bader
make -f Makefile.lnx_ifort  # or Makefile.lnx_gfort
cp bader /usr/local/bin/
```

Verify installation:
```bash
bader --version  # or just: bader
```

#### Step 1b: SCF Calculation

**File: `scf.in`** (example: TiO2 rutile)

```
&CONTROL
    calculation  = 'scf'
    prefix       = 'tio2'
    outdir       = './tmp'
    pseudo_dir   = './pseudo'
    tprnfor      = .true.
/
&SYSTEM
    ibrav        = 6
    celldm(1)    = 8.6806
    celldm(3)    = 0.6441
    nat          = 6
    ntyp         = 2
    ecutwfc      = 60.0
    ecutrho      = 600.0
/
&ELECTRONS
    conv_thr     = 1.0d-8
    mixing_beta  = 0.4
/
ATOMIC_SPECIES
  Ti  47.867   Ti.pbe-spn-rrkjus_psl.1.0.0.UPF
  O   15.999   O.pbe-n-rrkjus_psl.1.0.0.UPF

ATOMIC_POSITIONS {crystal}
  Ti  0.0000  0.0000  0.0000
  Ti  0.5000  0.5000  0.5000
  O   0.3053  0.3053  0.0000
  O   0.6947  0.6947  0.0000
  O   0.1947  0.8053  0.5000
  O   0.8053  0.1947  0.5000

K_POINTS {automatic}
  6 6 8  0 0 0
```

**Run:**
```bash
pw.x < scf.in > scf.out
```

#### Step 1c: Extract Charge Density as Cube File

For Bader analysis, you need the total valence charge density (and optionally the all-electron charge for PAW).

**File: `pp_charge.in`** (valence charge)

```
&INPUTPP
    prefix   = 'tio2'
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
    fileout     = 'charge_valence.cube'
/
```

For PAW pseudopotentials, also extract the all-electron density (more accurate for Bader):

**File: `pp_charge_ae.in`** (all-electron charge, PAW only)

```
&INPUTPP
    prefix   = 'tio2'
    outdir   = './tmp'
    filplot  = 'charge_ae.dat'
    plot_num = 21
/
&PLOT
    nfile       = 1
    filepp(1)   = 'charge_ae.dat'
    weight(1)   = 1.0
    iflag       = 3
    output_format = 6
    fileout     = 'charge_ae.cube'
/
```

**Run:**
```bash
pp.x < pp_charge.in > pp_charge.out
# For PAW:
pp.x < pp_charge_ae.in > pp_charge_ae.out
```

#### Step 1d: Run Bader Analysis

```bash
# Basic Bader analysis on valence charge
bader charge_valence.cube

# For PAW: use all-electron density as reference, valence as partitioning grid
bader charge_valence.cube -ref charge_ae.cube

# Output files produced:
#   ACF.dat   - Atomic Charge File (main results)
#   BCF.dat   - Bader Charge File (basin info)
#   AVF.dat   - Atomic Volume File
```

#### Step 1e: Parse and Visualize Bader Results

```python
#!/usr/bin/env python3
"""
Parse Bader analysis output (ACF.dat) and compute charge transfer.
"""
import numpy as np
import matplotlib.pyplot as plt


def parse_acf(acf_file='ACF.dat'):
    """
    Parse ACF.dat from the Henkelman Bader program.

    Returns
    -------
    list of dict
        Each dict has keys: 'index', 'x', 'y', 'z', 'charge', 'min_dist', 'volume'
    """
    atoms = []
    with open(acf_file, 'r') as f:
        lines = f.readlines()

    # ACF.dat format:
    #  #   X       Y       Z       CHARGE     MIN DIST   ATOMIC VOL
    # ----------------------------------------------------------------
    #  1  x.xxx   y.yyy   z.zzz   cc.cccc    d.dddd     v.vvvv
    # ...
    # ----------------------------------------------------------------
    #  VACUUM CHARGE:   v.vvvv
    #  VACUUM VOLUME:   v.vvvv
    #  NUMBER OF ELECTRONS:  nn.nnnn

    for line in lines:
        line = line.strip()
        if line.startswith('#') or line.startswith('-') or len(line) == 0:
            continue
        if line.startswith('VACUUM') or line.startswith('NUMBER'):
            continue
        parts = line.split()
        if len(parts) >= 7:
            try:
                atoms.append({
                    'index': int(parts[0]),
                    'x': float(parts[1]),
                    'y': float(parts[2]),
                    'z': float(parts[3]),
                    'charge': float(parts[4]),
                    'min_dist': float(parts[5]),
                    'volume': float(parts[6]),
                })
            except ValueError:
                continue

    return atoms


def compute_charge_transfer(acf_file='ACF.dat', atom_symbols=None,
                            valence_electrons=None, output_png='bader_charges.png'):
    """
    Compute charge transfer from Bader analysis.

    Parameters
    ----------
    acf_file : str
        Path to ACF.dat.
    atom_symbols : list of str
        Chemical symbols in the same order as atoms in the SCF input.
    valence_electrons : dict
        {element: number_of_valence_electrons} for the pseudopotentials used.
        e.g., {'Ti': 12, 'O': 6} for typical USPP/PAW.
    output_png : str
        Bar chart output.
    """
    atoms_data = parse_acf(acf_file)
    n_atoms = len(atoms_data)

    if atom_symbols is None:
        atom_symbols = [f'Atom{i+1}' for i in range(n_atoms)]
    if valence_electrons is None:
        print("WARNING: valence_electrons not provided. Cannot compute charge transfer.")
        print("Provide the number of valence electrons from your pseudopotential.")
        return

    print(f"\n{'Atom':<8} {'Symbol':<6} {'Bader Charge (e)':<18} "
          f"{'Valence e-':<12} {'Transfer (e)':<14} {'Oxidation':<10}")
    print("-" * 70)

    transfers = []
    labels = []
    for i, ad in enumerate(atoms_data):
        sym = atom_symbols[i]
        val_e = valence_electrons.get(sym, 0)
        bader_q = ad['charge']
        transfer = val_e - bader_q  # positive = lost electrons (cation)
        oxidation = round(transfer)

        sign = '+' if transfer > 0 else '-' if transfer < 0 else ''
        print(f"  {i+1:<6} {sym:<6} {bader_q:<18.4f} {val_e:<12} "
              f"{transfer:<14.4f} {sign}{abs(oxidation)}")

        transfers.append(transfer)
        labels.append(f"{sym}{i+1}")

    # Verify charge neutrality
    total_transfer = sum(transfers)
    print(f"\nTotal charge transfer (should be ~0): {total_transfer:.4f} e")

    # Bar chart
    fig, ax = plt.subplots(figsize=(10, 5))
    colors = ['#d62728' if t > 0.1 else '#2ca02c' if t < -0.1 else '#7f7f7f'
              for t in transfers]
    bars = ax.bar(range(len(transfers)), transfers, color=colors, edgecolor='black',
                  linewidth=0.5)
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha='right')
    ax.set_ylabel('Charge transfer (e)\n(+ = electron loss, - = electron gain)')
    ax.set_title('Bader Charge Transfer Analysis')
    ax.axhline(y=0, color='black', linewidth=0.5)
    ax.grid(axis='y', alpha=0.3)

    # Add value labels on bars
    for bar, val in zip(bars, transfers):
        y_pos = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, y_pos,
                f'{val:+.2f}', ha='center',
                va='bottom' if val >= 0 else 'top', fontsize=9)

    plt.tight_layout()
    plt.savefig(output_png, dpi=200)
    plt.close()
    print(f"\nSaved: {output_png}")


# ----- Run -----
compute_charge_transfer(
    acf_file='ACF.dat',
    atom_symbols=['Ti', 'Ti', 'O', 'O', 'O', 'O'],
    valence_electrons={'Ti': 12, 'O': 6},
    output_png='bader_charges.png'
)
```

#### Step 1f: Bader Analysis with pymatgen (Alternative Parser)

```python
#!/usr/bin/env python3
"""
Alternative: use pymatgen to analyze Bader charges from cube files.
Requires running the bader executable first to produce ACF.dat.
"""
import numpy as np
from pymatgen.core import Structure
from pymatgen.io.ase import AseAtomsAdaptor
from ase.io.cube import read_cube_data


def bader_summary_pymatgen(cube_file, acf_file='ACF.dat',
                           valence_electrons=None):
    """
    Combine pymatgen structure info with Bader ACF output.
    """
    data, atoms = read_cube_data(cube_file)
    structure = AseAtomsAdaptor.get_structure(atoms)

    # Parse ACF
    charges = []
    with open(acf_file, 'r') as f:
        for line in f:
            parts = line.split()
            if len(parts) >= 5:
                try:
                    int(parts[0])
                    charges.append(float(parts[4]))
                except ValueError:
                    pass

    if valence_electrons is None:
        valence_electrons = {}

    print("\nBader Charge Summary")
    print("=" * 60)
    for i, site in enumerate(structure):
        sym = site.specie.symbol
        val_e = valence_electrons.get(sym, 0)
        bader_q = charges[i] if i < len(charges) else 0
        transfer = val_e - bader_q
        print(f"  {sym:>3} ({site.frac_coords[0]:.3f}, {site.frac_coords[1]:.3f}, "
              f"{site.frac_coords[2]:.3f})  "
              f"Bader: {bader_q:.3f} e  Transfer: {transfer:+.3f} e")


bader_summary_pymatgen(
    'charge_valence.cube',
    'ACF.dat',
    valence_electrons={'Ti': 12, 'O': 6}
)
```

---

### Method 2: Lowdin Charge Analysis with projwfc.x (Simpler Fallback)

Lowdin population analysis is built into Quantum ESPRESSO via `projwfc.x` and requires no external tools. It projects wavefunctions onto atomic orbitals and sums occupations per atom.

#### Step 2a: SCF Calculation

Same as Step 1b above.

#### Step 2b: Run projwfc.x

**File: `projwfc.in`**

```
&PROJWFC
    prefix  = 'tio2'
    outdir  = './tmp'
    filpdos = 'pdos'
    Emin    = -30.0
    Emax    = 20.0
    DeltaE  = 0.1
    ngauss  = 0
    degauss = 0.01
/
```

**Run:**
```bash
projwfc.x < projwfc.in > projwfc.out
```

The Lowdin charges are printed directly in `projwfc.out`.

#### Step 2c: Parse Lowdin Charges from projwfc.x Output

```python
#!/usr/bin/env python3
"""
Parse Lowdin charges from projwfc.x output.
Lowdin charges appear near the end of the output file.
"""
import re
import numpy as np
import matplotlib.pyplot as plt


def parse_lowdin_charges(projwfc_out='projwfc.out'):
    """
    Parse Lowdin charges from projwfc.x output.

    The output contains lines like:
        Atom #   1: total charge =  10.1234, s =  1.234, p =  5.678, d =  3.211
        ...
        Spilling parameter: ...

    Returns
    -------
    list of dict
        Each dict: {'atom_index': int, 'total': float, 's': float, 'p': float, 'd': float, ...}
    """
    with open(projwfc_out, 'r') as f:
        content = f.read()

    # Find the Lowdin charges section
    # Pattern: "Atom #  N: total charge = XX.XXXX, s = XX.XXXX, p = XX.XXXX, ..."
    pattern = r'Atom\s*#\s*(\d+):\s*total charge\s*=\s*([\d.]+)((?:,\s*\w+\s*=\s*[\d.]+)*)'
    matches = re.findall(pattern, content)

    atoms = []
    for match in matches:
        atom_idx = int(match[0])
        total = float(match[1])
        orbitals = {}
        if match[2]:
            orb_pattern = r'(\w+)\s*=\s*([\d.]+)'
            orb_matches = re.findall(orb_pattern, match[2])
            for orb_name, orb_val in orb_matches:
                orbitals[orb_name] = float(orb_val)
        atoms.append({
            'atom_index': atom_idx,
            'total': total,
            **orbitals
        })

    return atoms


def lowdin_charge_transfer(projwfc_out='projwfc.out', atom_symbols=None,
                           valence_electrons=None, output_png='lowdin_charges.png'):
    """
    Compute and plot Lowdin charge transfer.
    """
    atoms = parse_lowdin_charges(projwfc_out)

    if not atoms:
        print("ERROR: No Lowdin charges found in output file.")
        print("Make sure projwfc.x ran successfully.")
        return

    if atom_symbols is None:
        atom_symbols = [f'Atom{a["atom_index"]}' for a in atoms]

    print(f"\n{'Atom':<8} {'Symbol':<6} {'Lowdin Charge':<15}", end='')
    orb_keys = [k for k in atoms[0].keys() if k not in ('atom_index', 'total')]
    for ok in orb_keys:
        print(f" {ok:>6}", end='')
    if valence_electrons:
        print(f"  {'Transfer':>10}", end='')
    print()
    print("-" * (40 + 7 * len(orb_keys) + (12 if valence_electrons else 0)))

    transfers = []
    labels = []
    for i, a in enumerate(atoms):
        sym = atom_symbols[i] if i < len(atom_symbols) else f'?{i}'
        line = f"  {a['atom_index']:<6} {sym:<6} {a['total']:<15.4f}"
        for ok in orb_keys:
            line += f" {a.get(ok, 0):>6.3f}"
        if valence_electrons:
            val_e = valence_electrons.get(sym, 0)
            transfer = val_e - a['total']
            line += f"  {transfer:>+10.4f}"
            transfers.append(transfer)
        labels.append(f"{sym}{a['atom_index']}")
        print(line)

    # Also extract spilling parameter
    with open(projwfc_out, 'r') as f:
        for line in f:
            if 'Spilling' in line:
                print(f"\n{line.strip()}")
                break

    if transfers:
        fig, ax = plt.subplots(figsize=(10, 5))
        colors = ['#d62728' if t > 0.1 else '#2ca02c' if t < -0.1 else '#7f7f7f'
                  for t in transfers]
        bars = ax.bar(range(len(transfers)), transfers, color=colors,
                      edgecolor='black', linewidth=0.5)
        ax.set_xticks(range(len(labels)))
        ax.set_xticklabels(labels, rotation=45, ha='right')
        ax.set_ylabel('Charge transfer (e)\n(+ = electron loss)')
        ax.set_title(u'L\u00f6wdin Charge Transfer')
        ax.axhline(y=0, color='black', linewidth=0.5)
        ax.grid(axis='y', alpha=0.3)
        for bar, val in zip(bars, transfers):
            y_pos = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2, y_pos,
                    f'{val:+.2f}', ha='center',
                    va='bottom' if val >= 0 else 'top', fontsize=9)
        plt.tight_layout()
        plt.savefig(output_png, dpi=200)
        plt.close()
        print(f"\nSaved: {output_png}")


# ----- Run -----
lowdin_charge_transfer(
    projwfc_out='projwfc.out',
    atom_symbols=['Ti', 'Ti', 'O', 'O', 'O', 'O'],
    valence_electrons={'Ti': 12, 'O': 6},
    output_png='lowdin_charges.png'
)
```

#### Step 2d: Orbital-Resolved Lowdin Charges (Stacked Bar)

```python
#!/usr/bin/env python3
"""
Plot orbital-resolved Lowdin charges as stacked bar chart.
"""
import numpy as np
import matplotlib.pyplot as plt


def parse_lowdin_charges(projwfc_out='projwfc.out'):
    """Parse Lowdin charges (same as above)."""
    import re
    with open(projwfc_out, 'r') as f:
        content = f.read()
    pattern = r'Atom\s*#\s*(\d+):\s*total charge\s*=\s*([\d.]+)((?:,\s*\w+\s*=\s*[\d.]+)*)'
    matches = re.findall(pattern, content)
    atoms = []
    for match in matches:
        atom_idx = int(match[0])
        total = float(match[1])
        orbitals = {}
        if match[2]:
            orb_pattern = r'(\w+)\s*=\s*([\d.]+)'
            orb_matches = re.findall(orb_pattern, match[2])
            for orb_name, orb_val in orb_matches:
                orbitals[orb_name] = float(orb_val)
        atoms.append({'atom_index': atom_idx, 'total': total, **orbitals})
    return atoms


def plot_orbital_breakdown(projwfc_out='projwfc.out', atom_symbols=None,
                           output_png='lowdin_orbital_breakdown.png'):
    """
    Stacked bar chart of Lowdin charges broken down by orbital type.
    """
    atoms = parse_lowdin_charges(projwfc_out)
    if not atoms:
        print("No data found.")
        return

    if atom_symbols is None:
        atom_symbols = [f'Atom{a["atom_index"]}' for a in atoms]

    # Collect orbital types
    orb_types = []
    for a in atoms:
        for k in a:
            if k not in ('atom_index', 'total') and k not in orb_types:
                orb_types.append(k)

    orb_colors = {'s': '#1f77b4', 'p': '#ff7f0e', 'd': '#2ca02c',
                  'f': '#d62728', 'sp': '#9467bd'}

    labels = [f"{atom_symbols[i]}{atoms[i]['atom_index']}" for i in range(len(atoms))]
    x = np.arange(len(atoms))
    width = 0.6

    fig, ax = plt.subplots(figsize=(10, 6))
    bottom = np.zeros(len(atoms))

    for orb in orb_types:
        vals = [a.get(orb, 0) for a in atoms]
        color = orb_colors.get(orb, '#8c564b')
        ax.bar(x, vals, width, bottom=bottom, label=orb, color=color,
               edgecolor='black', linewidth=0.3)
        bottom += np.array(vals)

    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha='right')
    ax.set_ylabel('Lowdin charge (e)')
    ax.set_title('Orbital-Resolved Lowdin Charges')
    ax.legend(title='Orbital')
    ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_png, dpi=200)
    plt.close()
    print(f"Saved: {output_png}")


# ----- Run -----
plot_orbital_breakdown(
    'projwfc.out',
    atom_symbols=['Ti', 'Ti', 'O', 'O', 'O', 'O']
)
```

## Key Parameters

### Bader-Specific

| Parameter | Notes |
|-----------|-------|
| `plot_num = 0` | Valence charge (default for most analyses) |
| `plot_num = 21` | All-electron charge (PAW only, more accurate for Bader) |
| `output_format = 6` | Cube file format (required by bader program) |
| `iflag = 3` | 3D grid (required for cube output) |

### Bader Command-Line Options

| Option | Description |
|--------|-------------|
| `bader CHGCAR` | Basic Bader analysis |
| `bader CHG -ref CHGCAR_sum` | Use reference charge for partitioning |
| `-vac off` | Disable vacuum detection (for periodic systems) |
| `-b weight` | Use weight method for charge assignment (more accurate for grids) |
| `-p all_atom` | Output all Bader atoms |

### Lowdin / projwfc.x Parameters

| Parameter | Value | Notes |
|-----------|-------|-------|
| `Emin` | e.g., -30.0 | Energy window lower bound (eV relative to Fermi) |
| `Emax` | e.g., 20.0 | Energy window upper bound |
| `DeltaE` | 0.05 -- 0.1 | Energy grid spacing for PDOS |
| `ngauss` | 0 | Gaussian broadening type (0 = simple Gaussian) |
| `degauss` | 0.01 -- 0.05 | Broadening width (Ry) |

### Pseudopotential Valence Electrons (Common Values)

You MUST know how many valence electrons your pseudopotential includes to compute charge transfer. Check the PP header or QE output.

| Element | Typical Valence e- | Notes |
|---------|-------------------|-------|
| Li | 1 or 3 | 1 (sv: 3) |
| O | 6 | 2s2 2p4 |
| Ti | 4 or 12 | 4 (3d2 4s2) or 12 (3s2 3p6 3d2 4s2) |
| Fe | 8 or 16 | 8 (3d6 4s2) or 16 (3s2 3p6 3d6 4s2) |
| Si | 4 | 3s2 3p2 |
| Na | 1 or 9 | 1 (3s1) or 9 (2s2 2p6 3s1) |

## Interpreting Results

### Bader Charges

- **Charge transfer** = (valence electrons from PP) - (Bader charge).
- Positive transfer means the atom lost electrons (cation).
- Negative transfer means the atom gained electrons (anion).
- Typical ionic compounds: transfer is 60--90% of formal oxidation state (never exactly integer due to covalent mixing).
- Example TiO2: Ti shows transfer of ~+2.2 to +2.6 e (formal Ti4+), O shows ~-1.1 to -1.3 e (formal O2-).

### Lowdin vs Bader

| Feature | Bader | Lowdin |
|---------|-------|--------|
| Basis | Real-space charge density | Atomic orbital projection |
| Accuracy | Better for charge transfer | Depends on pseudopotential basis |
| Oxidation states | More reliable | Qualitatively correct |
| Ease of use | Needs external tool | Built into QE |
| Spilling | N/A | Should be < 1% |

### Oxidation State Estimation

From Bader charges, the oxidation state is approximated as the nearest integer to the charge transfer. For transition metals, the actual transfer is typically 50-70% of the formal oxidation state due to covalent character.

## Common Issues

1. **Wrong number of valence electrons**: The charge transfer calculation is only meaningful if you know the correct number of valence electrons from your pseudopotential. Check `grep 'valence' scf.out` or the PP file header.

2. **Bader finds wrong number of atoms**: This happens with coarse FFT grids. Increase `ecutrho` to get a finer grid, or use `bader -vac off` for bulk systems.

3. **Bader charges do not sum to total electrons**: Small discrepancies (< 0.01 e) are normal. Larger discrepancies suggest grid issues.

4. **Lowdin spilling parameter > 1%**: The projection onto atomic orbitals is incomplete. This is common for plane-wave calculations and means the Lowdin charges are less reliable. Bader is preferred in this case.

5. **PAW vs USPP for Bader**: PAW pseudopotentials allow reconstruction of the all-electron density (`plot_num=21`), giving more accurate Bader charges. USPP only provides pseudo-charge, which underestimates charges on atoms with core electrons.

6. **Cube file too large**: For large supercells, the cube file can be huge. Consider using a coarser `ecutrho` for the Bader-specific run (but keep it fine enough for Bader to resolve atomic basins -- at least 100 grid points per lattice vector).

7. **bader executable not found**: Make sure it is in your PATH. Try `which bader` or provide the full path.

8. **Charge neutrality check**: Sum of all charge transfers should be approximately zero. If not, there may be a vacuum charge contribution or a parsing error.
