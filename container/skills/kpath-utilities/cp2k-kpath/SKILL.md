# K-Path Generation for CP2K

## When to Use

- You are preparing a band structure calculation in CP2K and need the `&BAND` section with high-symmetry k-paths.
- You want to convert a crystal structure (CIF, POSCAR, or any pymatgen-readable format) into a CP2K-compatible band structure input.
- You need to convert an existing VASP KPOINTS file to CP2K `&BAND` format.
- You want seekpath-standardized k-paths with proper labels for your crystal's Bravais lattice.
- You are comparing band structures computed with CP2K and other codes (QE, VASP) and need the same k-path in CP2K format.
- Equivalent to VASPKIT task 306 (K-Path for CP2K Code).

## Method Selection

| Criterion | From Structure File | From VASP KPOINTS |
|---|---|---|
| Input | CIF, POSCAR, or any pymatgen-readable structure | VASP KPOINTS file (line mode) |
| K-path source | seekpath / pymatgen HighSymmKpath | Parsed from existing KPOINTS |
| Convention | Hinuma (seekpath) or Setyawan-Curtarolo | Whatever convention was used in KPOINTS |
| Best for | New calculations, standardized paths | Reproducing VASP band structures in CP2K |
| Tool | seekpath + pymatgen + Python | Python parser |
| Label format | CP2K `SPECIAL_POINT` keyword | Converted from VASP labels |

## Prerequisites

- `pip install seekpath` (seekpath is not pre-installed; pymatgen and spglib are available).
- Python packages: `numpy`, `pymatgen`, `spglib` (pre-installed).
- A crystal structure file (CIF, POSCAR, or similar) or a VASP KPOINTS file.
- Basic knowledge of your system's space group (helps verify the k-path).

## Background

### CP2K Band Structure Input Format

CP2K uses a `&BAND` section within `&DFT > &PRINT > &BAND_STRUCTURE` (or `&BAND` in the `&PROPERTIES` section, depending on the CP2K version). The key elements are:

```
&BAND
  NPOINTS 50
  &KPOINT_SET
    SPECIAL_POINT  GAMMA  0.0000  0.0000  0.0000
    SPECIAL_POINT  X      0.5000  0.0000  0.5000
    NPOINTS 50
    UNITS  B_VECTOR
  &END KPOINT_SET
  &KPOINT_SET
    SPECIAL_POINT  X      0.5000  0.0000  0.5000
    SPECIAL_POINT  W      0.5000  0.2500  0.7500
    NPOINTS 50
    UNITS  B_VECTOR
  &END KPOINT_SET
  ADDED_MOS 10
  &END BAND
```

Key differences from VASP KPOINTS:
- Each segment is a separate `&KPOINT_SET` block.
- Points are labeled with `SPECIAL_POINT label kx ky kz`.
- `NPOINTS` is per segment (like VASP line-mode).
- `UNITS` can be `B_VECTOR` (fractional reciprocal) or `CART_ANGSTROM`.
- `ADDED_MOS` specifies extra empty bands to compute (like VASP NBANDS).

### seekpath Convention

seekpath (Hinuma et al., Comp. Mat. Sci. 128, 2017) provides standardized k-paths based on the Bravais lattice type. It first standardizes the structure to the conventional cell, then returns the recommended path with proper labels. This is the same convention used internally by pymatgen's `HighSymmKpath(mode="hinuma")`.

---

## Detailed Steps

### Method A: From Crystal Structure (seekpath + pymatgen)

#### Step 1: Generate Standardized K-Path

```python
#!/usr/bin/env python3
"""
Generate a CP2K band structure k-path from a crystal structure file.
Uses seekpath for standardized high-symmetry paths, then formats
the output as a CP2K &BAND section.

Usage:
  - Place your structure file (CIF, POSCAR, etc.) in the working directory.
  - Adjust STRUCTURE_FILE, NPOINTS, and ADDED_MOS below.
  - Run this script to generate the CP2K &BAND input block.
"""
import os
import numpy as np

try:
    import seekpath
    HAS_SEEKPATH = True
except ImportError:
    HAS_SEEKPATH = False
    print("seekpath not installed. Install with: pip install seekpath")

from pymatgen.core import Structure
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
from pymatgen.symmetry.bandstructure import HighSymmKpath

# ============================================================
# CONFIGURATION
# ============================================================

STRUCTURE_FILE = "POSCAR"       # Input structure file
NPOINTS = 50                    # K-points per segment
ADDED_MOS = 10                  # Extra empty bands for conduction
OUTPUT_FILE = "cp2k_band.inc"   # Output file
KPATH_MODE = "seekpath"         # "seekpath" or "pymatgen"
UNITS = "B_VECTOR"              # "B_VECTOR" (fractional) or "CART_ANGSTROM"

# ============================================================
# LOAD STRUCTURE
# ============================================================

structure = Structure.from_file(STRUCTURE_FILE)
sga = SpacegroupAnalyzer(structure, symprec=1e-5)
conv_structure = sga.get_conventional_standard_structure()
prim_structure = sga.get_primitive_standard_structure()

spacegroup = sga.get_space_group_symbol()
sg_number = sga.get_space_group_number()
bravais = sga.get_lattice_type()

print(f"Structure: {structure.composition.reduced_formula}")
print(f"Space group: {spacegroup} (#{sg_number})")
print(f"Bravais lattice: {bravais}")
print(f"Primitive cell atoms: {len(prim_structure)}")
print(f"Conventional cell atoms: {len(conv_structure)}")

# ============================================================
# GENERATE K-PATH
# ============================================================

if KPATH_MODE == "seekpath" and HAS_SEEKPATH:
    # Use seekpath for standardized path
    cell = (prim_structure.lattice.matrix,
            prim_structure.frac_coords,
            [s.Z for s in prim_structure.species])

    kpath_data = seekpath.get_path(cell, with_time_reversal=True,
                                    symprec=1e-5)

    # Extract path segments
    path_segments = kpath_data['path']
    special_points = kpath_data['point_coords']

    print(f"\nseekpath k-path ({len(path_segments)} segments):")
    for seg in path_segments:
        k1 = special_points[seg[0]]
        k2 = special_points[seg[1]]
        print(f"  {seg[0]:>10s} ({k1[0]:7.4f} {k1[1]:7.4f} {k1[2]:7.4f})"
              f"  -->  {seg[1]:>10s} ({k2[0]:7.4f} {k2[1]:7.4f} {k2[2]:7.4f})")

else:
    # Fall back to pymatgen HighSymmKpath
    kpath = HighSymmKpath(structure)
    path_segments_raw = kpath.kpath['path']
    special_points = kpath.kpath['kpoints']

    # Convert to list of (label1, label2) tuples
    path_segments = []
    for segment in path_segments_raw:
        for i in range(len(segment) - 1):
            path_segments.append((segment[i], segment[i + 1]))

    print(f"\npymatgen k-path ({len(path_segments)} segments):")
    for seg in path_segments:
        k1 = special_points[seg[0]]
        k2 = special_points[seg[1]]
        print(f"  {seg[0]:>10s} ({k1[0]:7.4f} {k1[1]:7.4f} {k1[2]:7.4f})"
              f"  -->  {seg[1]:>10s} ({k2[0]:7.4f} {k2[1]:7.4f} {k2[2]:7.4f})")


# ============================================================
# FORMAT AS CP2K &BAND SECTION
# ============================================================

def format_cp2k_label(label):
    """
    Convert k-point label to CP2K-compatible format.
    CP2K uses plain text labels (no LaTeX).

    Mappings:
      'GAMMA' or '\\Gamma' -> 'GAMMA'
      'SIGMA' or '\\Sigma' -> 'SIGMA'
      'DELTA' or '\\Delta' -> 'DELTA'
      'LAMBDA' or '\\Lambda' -> 'LAMBDA'
      Anything with '_' subscripts is kept as-is.
    """
    label_map = {
        '\\Gamma': 'GAMMA',
        'GAMMA': 'GAMMA',
        'G': 'GAMMA',
        '\\Sigma': 'SIGMA',
        '\\Sigma_0': 'SIGMA_0',
        '\\Delta': 'DELTA',
        '\\Lambda': 'LAMBDA',
    }
    return label_map.get(label, label.replace('\\', '').upper())


def generate_cp2k_band_section(path_segments, special_points,
                                 npoints=50, added_mos=10,
                                 units='B_VECTOR'):
    """
    Generate CP2K &BAND section as a string.

    Parameters
    ----------
    path_segments : list of (label1, label2)
        K-path segments.
    special_points : dict
        Label -> (kx, ky, kz) mapping.
    npoints : int
        Number of k-points per segment.
    added_mos : int
        Number of extra empty bands (for conduction band).
    units : str
        'B_VECTOR' for fractional reciprocal or 'CART_ANGSTROM'.

    Returns
    -------
    band_section : str
    """
    lines = []
    lines.append("    &BAND")
    lines.append(f"      ADDED_MOS  {added_mos}")

    for seg in path_segments:
        label1, label2 = seg
        k1 = special_points[label1]
        k2 = special_points[label2]

        cp2k_label1 = format_cp2k_label(label1)
        cp2k_label2 = format_cp2k_label(label2)

        lines.append("      &KPOINT_SET")
        lines.append(f"        NPOINTS  {npoints}")
        lines.append(f"        UNITS    {units}")
        lines.append(f"        SPECIAL_POINT  {cp2k_label1:>10s}  "
                     f"{k1[0]:10.6f}  {k1[1]:10.6f}  {k1[2]:10.6f}")
        lines.append(f"        SPECIAL_POINT  {cp2k_label2:>10s}  "
                     f"{k2[0]:10.6f}  {k2[1]:10.6f}  {k2[2]:10.6f}")
        lines.append("      &END KPOINT_SET")

    lines.append("    &END BAND")

    return "\n".join(lines)


band_section = generate_cp2k_band_section(
    path_segments, special_points,
    npoints=NPOINTS, added_mos=ADDED_MOS, units=UNITS)

print(f"\n{'='*60}")
print("CP2K &BAND Section")
print(f"{'='*60}")
print(band_section)

# ============================================================
# WRITE OUTPUT FILE
# ============================================================

with open(OUTPUT_FILE, 'w') as f:
    f.write(band_section + "\n")
print(f"\nWritten: {OUTPUT_FILE}")
print(f"Include this in your CP2K input under &DFT > &PRINT > &BAND_STRUCTURE")
```

#### Step 2: Complete CP2K Band Structure Input Template

```python
#!/usr/bin/env python3
"""
Generate a complete CP2K input file for band structure calculation.
Includes the &BAND section from Step 1 and a full DFT setup.

This template uses the DZVP-MOLOPT-SR-GTH basis set and PBE functional.
Adjust basis set, functional, and parameters for your system.
"""
import os
import numpy as np
from pymatgen.core import Structure
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer

# ============================================================
# CONFIGURATION
# ============================================================

STRUCTURE_FILE = "POSCAR"
PROJECT_NAME = "band_structure"
BASIS_SET = "DZVP-MOLOPT-SR-GTH"
POTENTIAL = "GTH-PBE"
CUTOFF = 600          # Ry (plane-wave cutoff for the grid)
REL_CUTOFF = 60       # Ry (relative cutoff for Gaussian mapping)
NGRIDS = 5
KPOINTS_MESH = [6, 6, 6]   # SCF k-mesh
ADDED_MOS = 10
NPOINTS_BAND = 50

# ============================================================
# LOAD STRUCTURE
# ============================================================

structure = Structure.from_file(STRUCTURE_FILE)

# Get lattice vectors in Angstrom
lattice = structure.lattice.matrix  # (3, 3)

# ============================================================
# READ &BAND SECTION FROM STEP 1
# ============================================================

band_inc_file = "cp2k_band.inc"
if os.path.exists(band_inc_file):
    with open(band_inc_file, 'r') as f:
        band_section = f.read()
else:
    print(f"WARNING: {band_inc_file} not found. Run Step 1 first.")
    band_section = """    &BAND
      ADDED_MOS 10
      &KPOINT_SET
        NPOINTS 50
        UNITS B_VECTOR
        SPECIAL_POINT  GAMMA   0.000000   0.000000   0.000000
        SPECIAL_POINT  X       0.500000   0.000000   0.500000
      &END KPOINT_SET
    &END BAND"""

# ============================================================
# ELEMENT-SPECIFIC SETTINGS
# ============================================================

unique_elements = sorted(set(str(s) for s in structure.species))

# Basis set and potential mapping
# Adjust for your elements. Common choices:
element_settings = {}
for elem in unique_elements:
    element_settings[elem] = {
        'basis': BASIS_SET,
        'potential': POTENTIAL,
    }

# ============================================================
# GENERATE COORDINATE BLOCK
# ============================================================

coord_lines = []
for site in structure:
    elem = str(site.specie)
    x, y, z = site.coords  # Cartesian in Angstrom
    coord_lines.append(f"      {elem:>4s}  {x:14.8f}  {y:14.8f}  {z:14.8f}")

# ============================================================
# GENERATE CP2K INPUT
# ============================================================

# Kind blocks
kind_blocks = []
for elem in unique_elements:
    kind_blocks.append(f"""    &KIND {elem}
      BASIS_SET {element_settings[elem]['basis']}
      POTENTIAL {element_settings[elem]['potential']}
    &END KIND""")

kind_section = "\n".join(kind_blocks)

cp2k_input = f"""&GLOBAL
  PROJECT  {PROJECT_NAME}
  RUN_TYPE ENERGY
  PRINT_LEVEL MEDIUM
&END GLOBAL

&FORCE_EVAL
  METHOD Quickstep

  &SUBSYS
    &CELL
      A  {lattice[0,0]:14.8f}  {lattice[0,1]:14.8f}  {lattice[0,2]:14.8f}
      B  {lattice[1,0]:14.8f}  {lattice[1,1]:14.8f}  {lattice[1,2]:14.8f}
      C  {lattice[2,0]:14.8f}  {lattice[2,1]:14.8f}  {lattice[2,2]:14.8f}
      PERIODIC XYZ
    &END CELL

    &COORD
{chr(10).join(coord_lines)}
    &END COORD

{kind_section}

  &END SUBSYS

  &DFT
    BASIS_SET_FILE_NAME  BASIS_MOLOPT
    POTENTIAL_FILE_NAME  GTH_POTENTIALS

    &MGRID
      CUTOFF      {CUTOFF}
      REL_CUTOFF  {REL_CUTOFF}
      NGRIDS      {NGRIDS}
    &END MGRID

    &QS
      EPS_DEFAULT 1.0E-12
      EXTRAPOLATION ASPC
      EXTRAPOLATION_ORDER 3
    &END QS

    &SCF
      SCF_GUESS ATOMIC
      EPS_SCF   1.0E-7
      MAX_SCF   200
      ADDED_MOS {ADDED_MOS}

      &DIAGONALIZATION
        ALGORITHM STANDARD
      &END DIAGONALIZATION

      &SMEAR ON
        METHOD FERMI_DIRAC
        ELECTRONIC_TEMPERATURE [K] 300
      &END SMEAR

      &MIXING
        METHOD BROYDEN_MIXING
        ALPHA   0.4
        NBUFFER 8
      &END MIXING

      &PRINT
        &RESTART
          &EACH
            QS_SCF 0
          &END EACH
          ADD_LAST NUMERIC
        &END RESTART
      &END PRINT
    &END SCF

    &XC
      &XC_FUNCTIONAL PBE
      &END XC_FUNCTIONAL
    &END XC

    &KPOINTS
      SCHEME MONKHORST-PACK {KPOINTS_MESH[0]} {KPOINTS_MESH[1]} {KPOINTS_MESH[2]}
      SYMMETRY ON
      FULL_GRID OFF
    &END KPOINTS

    &PRINT
      &BAND_STRUCTURE
{band_section}
      &END BAND_STRUCTURE
    &END PRINT

  &END DFT
&END FORCE_EVAL
"""

output_file = f"{PROJECT_NAME}.inp"
with open(output_file, 'w') as f:
    f.write(cp2k_input)

print(f"Written: {output_file}")
print(f"\nStructure: {structure.composition.reduced_formula}")
print(f"Elements: {', '.join(unique_elements)}")
print(f"K-mesh: {KPOINTS_MESH}")
print(f"Band k-points: {NPOINTS_BAND} per segment")
print(f"ADDED_MOS: {ADDED_MOS}")
print(f"\nTo run: cp2k.popt -i {output_file} -o {PROJECT_NAME}.out")
```

#### Step 3: Parse CP2K Band Structure Output

```python
#!/usr/bin/env python3
"""
Parse CP2K band structure output and plot.
CP2K writes band energies to files named:
  {PROJECT}-BAND_S{spin}-{set}.bs

Each file contains k-point coordinates and eigenvalues for one segment.
"""
import os
import glob
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def parse_cp2k_band_file(filename):
    """
    Parse a single CP2K band structure file (.bs).

    The format is:
      # Set X: ...
      # Point X Spin X: ...
      #   Special K-Point: LABEL  kx ky kz
      #   Band    Energy [eV]    Occupation
           1    -12.3456        2.000000
           ...
      # Point X Spin X: ...
      ...

    Returns
    -------
    kpoints : list of dict with 'kpt', 'label' (or None), 'bands'
    """
    with open(filename, 'r') as f:
        lines = f.readlines()

    kpoints_data = []
    current_kpt = None
    current_label = None
    current_bands = []

    for line in lines:
        line_stripped = line.strip()

        if line_stripped.startswith('# Point'):
            # Save previous k-point
            if current_kpt is not None:
                kpoints_data.append({
                    'kpt': current_kpt,
                    'label': current_label,
                    'energies': np.array(current_bands),
                })

            current_kpt = None
            current_label = None
            current_bands = []

        elif line_stripped.startswith('#   Special K-Point:'):
            parts = line_stripped.split(':')[1].strip().split()
            current_label = parts[0]
            current_kpt = [float(parts[1]), float(parts[2]),
                           float(parts[3])]

        elif line_stripped.startswith('#') and 'Nr.' in line_stripped:
            # Header line with k-point coordinates but no special label
            # Format: # Nr.  X  Spin X  K-point:  kx  ky  kz
            parts = line_stripped.split('K-point:')
            if len(parts) > 1:
                coords = parts[1].strip().split()
                current_kpt = [float(coords[0]), float(coords[1]),
                               float(coords[2])]

        elif not line_stripped.startswith('#') and len(line_stripped) > 0:
            # Band data line: band_index energy occupation
            parts = line_stripped.split()
            if len(parts) >= 2:
                try:
                    energy = float(parts[1])
                    current_bands.append(energy)
                except ValueError:
                    pass

    # Save last k-point
    if current_kpt is not None and len(current_bands) > 0:
        kpoints_data.append({
            'kpt': current_kpt,
            'label': current_label,
            'energies': np.array(current_bands),
        })

    return kpoints_data


def plot_cp2k_bands(band_files, fermi_energy=None, output_file='bands_cp2k.png'):
    """
    Plot CP2K band structure from multiple .bs files.

    Parameters
    ----------
    band_files : list of str
        Paths to .bs files (one per k-path segment).
    fermi_energy : float or None
        Fermi energy in eV. If provided, bands are shifted.
    output_file : str
        Output plot filename.
    """
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))

    all_distances = []
    all_energies = []
    tick_positions = []
    tick_labels = []
    offset = 0.0

    for bs_file in sorted(band_files):
        kpt_data = parse_cp2k_band_file(bs_file)
        if len(kpt_data) == 0:
            continue

        # Compute k-path distances
        distances = [0.0]
        for i in range(1, len(kpt_data)):
            dk = np.array(kpt_data[i]['kpt']) - np.array(kpt_data[i-1]['kpt'])
            distances.append(distances[-1] + np.linalg.norm(dk))

        distances = np.array(distances) + offset
        nbands = len(kpt_data[0]['energies'])

        # Collect tick marks from special points
        for i, kd in enumerate(kpt_data):
            if kd['label'] is not None:
                tick_positions.append(distances[i])
                label = kd['label']
                if label == 'GAMMA':
                    label = r'$\Gamma$'
                tick_labels.append(label)

        # Plot bands
        for ib in range(nbands):
            band_energies = np.array([kd['energies'][ib]
                                      for kd in kpt_data])
            if fermi_energy is not None:
                band_energies -= fermi_energy
            ax.plot(distances, band_energies, 'b-', linewidth=0.8)

        offset = distances[-1]

    # Formatting
    if fermi_energy is not None:
        ax.axhline(y=0, color='gray', linestyle='--', linewidth=0.5)
        ax.set_ylabel('E - E_F (eV)', fontsize=12)
    else:
        ax.set_ylabel('Energy (eV)', fontsize=12)

    ax.set_xlabel('K-path', fontsize=12)
    ax.set_title('CP2K Band Structure', fontsize=14)

    # Add tick marks
    if tick_positions:
        ax.set_xticks(tick_positions)
        ax.set_xticklabels(tick_labels, fontsize=10)
        for tp in tick_positions:
            ax.axvline(x=tp, color='gray', linestyle='-',
                       linewidth=0.3, alpha=0.5)

    ax.set_xlim(0, offset)
    ax.grid(True, alpha=0.2)
    fig.tight_layout()
    fig.savefig(output_file, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_file}")


# ── Main ───────────────────────────────────────────────────────
print("=" * 60)
print("CP2K Band Structure Parser and Plotter")
print("=" * 60)

# Find band structure files
bs_files = sorted(glob.glob("*BAND*.bs"))
if not bs_files:
    bs_files = sorted(glob.glob("*-BAND_S*.bs"))

if bs_files:
    print(f"Found {len(bs_files)} band structure file(s):")
    for f in bs_files:
        print(f"  {f}")

    # Try to extract Fermi energy from CP2K output
    fermi_e = None
    out_files = glob.glob("*.out") + glob.glob("*.log")
    for out_file in out_files:
        with open(out_file, 'r') as f:
            for line in f:
                if 'Fermi energy' in line or 'FERMI' in line.upper():
                    parts = line.split()
                    for i, p in enumerate(parts):
                        try:
                            val = float(p)
                            fermi_e = val
                        except ValueError:
                            pass

    if fermi_e is not None:
        print(f"Fermi energy: {fermi_e:.4f} eV")

    plot_cp2k_bands(bs_files, fermi_energy=fermi_e,
                    output_file='bands_cp2k.png')
else:
    print("No CP2K band structure files found.")
    print("Run CP2K first, then re-run this script.")
    print("Expected files: {PROJECT}-BAND_S1-*.bs")
```

---

### Method B: Convert VASP KPOINTS to CP2K

```python
#!/usr/bin/env python3
"""
Convert a VASP KPOINTS file (line mode) to CP2K &BAND section format.
This is useful when you have an existing VASP band structure setup
and want to reproduce the same k-path in CP2K.

Equivalent to VASPKIT task 306 applied to an existing KPOINTS file.
"""
import os
import re
import numpy as np


def parse_vasp_kpoints_linemode(filename='KPOINTS'):
    """
    Parse VASP KPOINTS file in line mode.

    Line mode format:
      Comment line
      N  (number of k-points per segment)
      Line-mode | line_mode | L
      Reciprocal | Cart
      kx1  ky1  kz1  ! label1
      kx2  ky2  kz2  ! label2
      <blank line>
      kx3  ky3  kz3  ! label3
      kx4  ky4  kz4  ! label4
      ...

    Returns
    -------
    segments : list of (label1, k1, label2, k2)
    npoints : int
    is_reciprocal : bool
    """
    with open(filename, 'r') as f:
        lines = f.readlines()

    comment = lines[0].strip()
    npoints = int(lines[1].strip().split()[0])

    mode_line = lines[2].strip().lower()
    if mode_line.startswith('l') or 'line' in mode_line:
        line_mode = True
    else:
        line_mode = False

    coord_line = lines[3].strip().lower()
    is_reciprocal = coord_line.startswith('r') or coord_line.startswith('k')

    # Parse k-point pairs
    segments = []
    i = 4
    while i < len(lines) - 1:
        # Skip blank lines
        if lines[i].strip() == '':
            i += 1
            continue

        # Parse first point of segment
        parts1 = lines[i].strip().split()
        k1 = [float(parts1[0]), float(parts1[1]), float(parts1[2])]
        label1 = ''
        # Extract label after ! or from 4th element
        rest1 = ' '.join(parts1[3:])
        if '!' in lines[i]:
            label1 = lines[i].split('!')[1].strip()
        elif len(parts1) > 3:
            label1 = parts1[3].strip('!')
        i += 1

        # Parse second point
        if i >= len(lines):
            break
        parts2 = lines[i].strip().split()
        if len(parts2) < 3:
            i += 1
            continue

        k2 = [float(parts2[0]), float(parts2[1]), float(parts2[2])]
        label2 = ''
        if '!' in lines[i]:
            label2 = lines[i].split('!')[1].strip()
        elif len(parts2) > 3:
            label2 = parts2[3].strip('!')
        i += 1

        # Clean labels
        label1 = label1.replace('\\', '').strip()
        label2 = label2.replace('\\', '').strip()
        if not label1:
            label1 = f'K{len(segments)*2+1}'
        if not label2:
            label2 = f'K{len(segments)*2+2}'

        segments.append((label1, k1, label2, k2))

    return segments, npoints, is_reciprocal


def vasp_kpoints_to_cp2k_band(kpoints_file='KPOINTS',
                                added_mos=10,
                                output_file='cp2k_band_from_vasp.inc'):
    """
    Convert VASP KPOINTS (line mode) to CP2K &BAND section.

    Parameters
    ----------
    kpoints_file : str
        Path to VASP KPOINTS file.
    added_mos : int
        Number of extra empty bands for CP2K.
    output_file : str
        Output filename for CP2K &BAND section.
    """
    segments, npoints, is_reciprocal = parse_vasp_kpoints_linemode(
        kpoints_file)

    print(f"Parsed {len(segments)} segments from {kpoints_file}")
    print(f"Points per segment: {npoints}")
    print(f"Coordinates: {'reciprocal' if is_reciprocal else 'Cartesian'}")

    units = 'B_VECTOR' if is_reciprocal else 'CART_ANGSTROM'

    # Label conversion for CP2K
    def convert_label(label):
        label_map = {
            'Gamma': 'GAMMA', 'GAMMA': 'GAMMA', 'G': 'GAMMA',
            'Sigma': 'SIGMA', 'Delta': 'DELTA', 'Lambda': 'LAMBDA',
        }
        cleaned = label.replace('\\', '').strip()
        return label_map.get(cleaned, cleaned.upper())

    lines = []
    lines.append("    &BAND")
    lines.append(f"      ADDED_MOS  {added_mos}")

    for label1, k1, label2, k2 in segments:
        cp2k_label1 = convert_label(label1)
        cp2k_label2 = convert_label(label2)

        lines.append("      &KPOINT_SET")
        lines.append(f"        NPOINTS  {npoints}")
        lines.append(f"        UNITS    {units}")
        lines.append(f"        SPECIAL_POINT  {cp2k_label1:>10s}  "
                     f"{k1[0]:10.6f}  {k1[1]:10.6f}  {k1[2]:10.6f}")
        lines.append(f"        SPECIAL_POINT  {cp2k_label2:>10s}  "
                     f"{k2[0]:10.6f}  {k2[1]:10.6f}  {k2[2]:10.6f}")
        lines.append("      &END KPOINT_SET")

    lines.append("    &END BAND")

    band_text = "\n".join(lines)

    with open(output_file, 'w') as f:
        f.write(band_text + "\n")

    print(f"\nWritten: {output_file}")
    print(f"\n{'='*60}")
    print(band_text)
    print(f"{'='*60}")

    # Print segment summary
    print(f"\nSegment summary:")
    for i, (l1, k1, l2, k2) in enumerate(segments):
        print(f"  {i+1}. {convert_label(l1):>10s} "
              f"({k1[0]:7.4f} {k1[1]:7.4f} {k1[2]:7.4f}) --> "
              f"{convert_label(l2):>10s} "
              f"({k2[0]:7.4f} {k2[1]:7.4f} {k2[2]:7.4f})")

    return band_text


# ── Main ───────────────────────────────────────────────────────
print("=" * 60)
print("VASP KPOINTS to CP2K &BAND Converter")
print("=" * 60)

kpoints_file = 'KPOINTS'
if os.path.exists(kpoints_file):
    vasp_kpoints_to_cp2k_band(kpoints_file, added_mos=10)
else:
    print(f"\n{kpoints_file} not found.")
    print("Provide a VASP KPOINTS file in line mode, or use Method A")
    print("to generate a k-path from a structure file.")
    print("\nIf using VASPKIT directly: vaspkit -task 306")
    print("VASPKIT 306 generates the CP2K &BAND section automatically.")
```

---

## Key Parameters

| Parameter | Typical Value | Notes |
|---|---|---|
| NPOINTS | 30-100 | K-points per segment. 50 is a good default for smooth bands. More for complex band crossings. |
| ADDED_MOS | 5-20 | Extra empty bands beyond occupied. 10 is typical. Increase for wide conduction band range. |
| UNITS | B_VECTOR | Fractional reciprocal coordinates. Use CART_ANGSTROM only if your structure is not periodic in all directions. |
| CUTOFF (CP2K) | 400-800 Ry | Grid cutoff for Gaussian mapping. Converge for your system. |
| REL_CUTOFF | 40-80 Ry | Relative cutoff. 60 Ry is typical. |
| KPOINTS mesh | 6x6x6 to 12x12x12 | SCF k-mesh for the ground state. Must be converged separately. |
| symprec | 1e-5 | Symmetry tolerance for seekpath. Tighter may give more segments. |
| KPATH_MODE | seekpath | Use "seekpath" for standardized paths (recommended) or "pymatgen" as fallback. |

### CP2K-Specific Notes

| Setting | CP2K | VASP Equivalent |
|---|---|---|
| K-path definition | `&KPOINT_SET` blocks | KPOINTS file (line mode) |
| Segment points | `NPOINTS` per segment | First line of KPOINTS |
| Extra bands | `ADDED_MOS` in `&BAND` and `&SCF` | `NBANDS` in INCAR |
| K-point coordinates | `SPECIAL_POINT label kx ky kz` | `kx ky kz ! label` |
| Grid density | `&KPOINTS SCHEME MONKHORST-PACK` | KPOINTS file (automatic mode) |

## Interpreting Results

### K-Path Validation

- **Compare labels with literature**: The seekpath labels follow the Hinuma convention. Verify that the path covers the expected high-symmetry points for your space group.
- **Check segment connectivity**: Consecutive segments should share endpoints (e.g., Gamma-X, X-W, not Gamma-X, Y-W). Gaps in the path create discontinuities in the band plot.
- **Verify coordinates**: Cross-check a few k-points against the Bilbao Crystallographic Server or literature values for your space group.

### CP2K Band Output

- **File naming**: CP2K writes `{PROJECT}-BAND_S{spin}-{set}.bs` files. Spin 1 and 2 for spin-polarized calculations.
- **Energy reference**: CP2K eigenvalues are absolute (not shifted to Fermi level). Shift manually using the Fermi energy from the SCF output.
- **Convergence**: Verify that the SCF converged before trusting band energies. Unconverged SCF gives meaningless bands.

### Common Bravais Lattice Paths

| Lattice | seekpath Path | Key Points |
|---|---|---|
| Cubic (FCC) | GAMMA-X-W-K-GAMMA-L-U-W-L-K | X=(0.5,0,0.5), L=(0.5,0.5,0.5) |
| Cubic (BCC) | GAMMA-H-N-GAMMA-P-H | H=(0.5,-0.5,0.5), N=(0,0,0.5) |
| Hexagonal | GAMMA-M-K-GAMMA-A-L-H-A | M=(0.5,0,0), K=(1/3,1/3,0) |
| Tetragonal | GAMMA-X-M-GAMMA-Z-R-A-Z | X=(0.5,0,0), M=(0.5,0.5,0) |

## Common Issues

| Problem | Solution |
|---|---|
| **seekpath not installed** | Run `pip install seekpath`. It is not pre-installed in the container. |
| **Labels appear as numbers** | The KPOINTS file may not have labels after `!`. Add labels manually or use Method A to regenerate. |
| **Wrong k-path for space group** | Verify that the input structure is properly symmetrized. Use `SpacegroupAnalyzer(structure).get_refined_structure()` before generating the path. |
| **CP2K crashes on band structure** | Check that `ADDED_MOS` is set in both `&SCF` and `&BAND`. CP2K needs extra MOs allocated during SCF to compute unoccupied bands. |
| **Band plot has gaps between segments** | Segments must share endpoints. Check that the last point of segment N matches the first point of segment N+1. |
| **Coordinates are in wrong basis** | Ensure `UNITS B_VECTOR` matches fractional reciprocal coordinates. If your KPOINTS file uses Cartesian, set `UNITS CART_ANGSTROM`. |
| **Too few bands plotted** | Increase `ADDED_MOS` to include more conduction bands. Default 10 may not be enough for wide-gap insulators. |
| **CP2K and VASP band structures differ** | Different k-paths or different conventions. Use the same seekpath-generated path for both. Also check that CP2K uses the same primitive cell as VASP. |
| **VASPKIT 306 output format** | VASPKIT 306 writes a ready-to-use CP2K `&BAND` section. Copy-paste into your CP2K input file under `&DFT > &PRINT > &BAND_STRUCTURE`. |
