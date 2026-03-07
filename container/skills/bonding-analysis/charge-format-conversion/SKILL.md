# Charge Density Format Conversion

Convert charge density and volumetric data files between common computational chemistry formats. Build supercells of charge density data. Supports VASP CHGCAR, Gaussian .cube, XCrySDen .xsf, and NumPy arrays.

## When to Use

- Convert VASP CHGCAR to Gaussian .cube format for visualization in VESTA, VMD, or other tools.
- Convert QE cube files to XCrySDen .xsf format.
- Convert between any pair of CHGCAR, .cube, and .xsf formats.
- Build a supercell of volumetric data (charge density, ELF, potential) for visualization.
- Merge or split spin-polarized charge density files.
- Prepare charge density input for cross-code workflows (e.g., QE output to VASP post-processing tools).

## Method Selection

| Conversion | Tool | Notes |
|------------|------|-------|
| CHGCAR to .cube | pymatgen + ase | Via intermediate data array |
| CHGCAR to .xsf | pymatgen + ase | Via intermediate data array |
| .cube to CHGCAR | ase + pymatgen | Must set correct VASP normalization |
| .cube to .xsf | ase I/O | Direct conversion |
| .xsf to .cube | ase I/O | Direct conversion |
| .xsf to CHGCAR | ase + pymatgen | Via intermediate data array |
| Supercell of CHGCAR | VASPKIT 320 / pymatgen | Tile charge density in real space |
| Supercell of .cube | Python (numpy) | Tile 3D data array |
| VASPKIT 328 | VASP | CHGCAR to .cube conversion |
| VASPKIT 329 | VASP | CHGCAR to .xsf conversion |

## Prerequisites

- Python: `numpy`, `ase`, `pymatgen`
- No DFT codes needed -- this is purely post-processing
- Input volumetric data file in any supported format

## Detailed Steps

### Conversion: CHGCAR to Gaussian .cube

```python
#!/usr/bin/env python3
"""
Convert VASP CHGCAR to Gaussian cube format.
Handles normalization differences between formats.
"""
import numpy as np
from pymatgen.io.vasp import Chgcar
from ase.io.cube import write_cube
from ase.atoms import Atoms
from ase import units


def chgcar_to_cube(chgcar_file='CHGCAR', output_cube='output.cube',
                    data_key='total'):
    """
    Convert VASP CHGCAR to Gaussian .cube file.

    Parameters
    ----------
    chgcar_file : str
        Path to CHGCAR (or PARCHG, ELFCAR, LOCPOT).
    output_cube : str
        Output cube file path.
    data_key : str
        Which data to convert: 'total' or 'diff' (for spin-polarized).
        For ELFCAR/LOCPOT, use 'total'.
    """
    chgcar = Chgcar.from_file(chgcar_file)
    structure = chgcar.structure
    data = chgcar.data[data_key].copy()

    # VASP stores charge as rho * V_cell (total electrons in grid)
    # Cube format stores rho in e/bohr^3
    vol_angstrom = structure.volume
    vol_bohr = vol_angstrom / (units.Bohr ** 3)

    # Normalize: CHGCAR value / V_cell -> e/Angstrom^3
    #            then e/Angstrom^3 * Angstrom^3/bohr^3 -> e/bohr^3
    # Or equivalently: data / V_cell_bohr^3 / N_grid -> e/bohr^3
    # But cube write_cube in ASE expects e/bohr^3 on the grid
    data_density = data / vol_bohr  # e/bohr^3 * N_grid_total
    ngrid = data.shape[0] * data.shape[1] * data.shape[2]
    data_per_point = data_density / ngrid  # wrong -- let us be careful

    # Actually: CHGCAR stores rho(r) * V_cell
    # So data / V_cell = rho in e/Angstrom^3
    # Convert to e/bohr^3: rho * (Angstrom/bohr)^3
    BOHR = units.Bohr  # 0.529177 Angstrom
    rho_e_per_ang3 = data / vol_angstrom
    rho_e_per_bohr3 = rho_e_per_ang3 * (BOHR ** 3)

    # Build ASE Atoms from pymatgen Structure
    atoms = Atoms(
        symbols=[str(s.specie) for s in structure],
        positions=[s.coords for s in structure],
        cell=structure.lattice.matrix,
        pbc=True
    )

    with open(output_cube, 'w') as f:
        write_cube(f, atoms, rho_e_per_bohr3)

    total_e = np.sum(rho_e_per_ang3) * vol_angstrom / ngrid
    print(f"Converted: {chgcar_file} -> {output_cube}")
    print(f"Grid: {data.shape}")
    print(f"Total electrons: {total_e:.4f}")
    print(f"Data range: {rho_e_per_bohr3.min():.6e} to {rho_e_per_bohr3.max():.6e} e/bohr^3")


# ----- Run -----
chgcar_to_cube('CHGCAR', 'charge.cube', data_key='total')
```

### Conversion: Gaussian .cube to CHGCAR

```python
#!/usr/bin/env python3
"""
Convert Gaussian cube file (e.g., from QE pp.x) to VASP CHGCAR format.
"""
import numpy as np
from ase.io.cube import read_cube_data
from ase import units
from pymatgen.io.vasp import Chgcar
from pymatgen.core import Structure, Lattice


def cube_to_chgcar(cube_file='input.cube', output_chgcar='CHGCAR_converted'):
    """
    Convert Gaussian .cube to VASP CHGCAR format.

    Parameters
    ----------
    cube_file : str
        Input cube file.
    output_chgcar : str
        Output CHGCAR file.
    """
    data, atoms = read_cube_data(cube_file)

    # Cube data is in e/bohr^3; CHGCAR stores rho * V_cell
    BOHR = units.Bohr  # 0.529177 Angstrom
    cell = atoms.get_cell()
    vol_angstrom = atoms.get_volume()

    # Convert e/bohr^3 to e/Angstrom^3
    rho_ang = data / (BOHR ** 3)

    # CHGCAR format: data = rho * V_cell
    chgcar_data = rho_ang * vol_angstrom

    # Build pymatgen Structure
    lattice = Lattice(cell)
    species = atoms.get_chemical_symbols()
    coords = atoms.get_positions()
    structure = Structure(lattice, species, coords, coords_are_cartesian=True)

    # Create Chgcar object
    chgcar = Chgcar(structure, data={'total': chgcar_data})
    chgcar.write_file(output_chgcar)

    ngrid = data.shape[0] * data.shape[1] * data.shape[2]
    total_e = np.sum(rho_ang) * vol_angstrom / ngrid
    print(f"Converted: {cube_file} -> {output_chgcar}")
    print(f"Grid: {data.shape}")
    print(f"Total electrons: {total_e:.4f}")


# ----- Run -----
cube_to_chgcar('charge_total.cube', 'CHGCAR_from_qe')
```

### Conversion: CHGCAR to XCrySDen .xsf

```python
#!/usr/bin/env python3
"""
Convert VASP CHGCAR to XCrySDen .xsf format.
XSF is widely used for visualization (XCrySDen, VESTA).
"""
import numpy as np
from pymatgen.io.vasp import Chgcar
from ase.atoms import Atoms
from ase.io import write as ase_write
from ase import units


def chgcar_to_xsf(chgcar_file='CHGCAR', output_xsf='output.xsf',
                    data_key='total'):
    """
    Convert VASP CHGCAR to XCrySDen .xsf format.

    Parameters
    ----------
    chgcar_file : str
        Input CHGCAR file.
    output_xsf : str
        Output .xsf file.
    data_key : str
        'total' or 'diff'.
    """
    chgcar = Chgcar.from_file(chgcar_file)
    structure = chgcar.structure
    data = chgcar.data[data_key].copy()

    vol_angstrom = structure.volume

    # CHGCAR stores rho * V_cell; XSF expects density at grid points
    # XSF data grid format: values on a regular grid, units vary
    # Convention: store in e/Angstrom^3 for xsf
    rho = data / vol_angstrom

    # Build ASE Atoms
    atoms = Atoms(
        symbols=[str(s.specie) for s in structure],
        positions=[s.coords for s in structure],
        cell=structure.lattice.matrix,
        pbc=True
    )

    # Write XSF with data grid
    # ASE's write for xsf with data uses a specific API
    nx, ny, nz = rho.shape

    with open(output_xsf, 'w') as f:
        f.write("CRYSTAL\n")
        f.write("PRIMVEC\n")
        cell = structure.lattice.matrix
        for vec in cell:
            f.write(f"  {vec[0]:16.10f}  {vec[1]:16.10f}  {vec[2]:16.10f}\n")
        f.write("PRIMCOORD\n")
        f.write(f"  {len(structure)}  1\n")
        for site in structure:
            sym = str(site.specie)
            x, y, z = site.coords
            f.write(f"  {sym:>3s}  {x:16.10f}  {y:16.10f}  {z:16.10f}\n")
        f.write("\nBEGIN_BLOCK_DATAGRID_3D\n")
        f.write("  charge_density\n")
        f.write("  BEGIN_DATAGRID_3D_charge\n")
        f.write(f"    {nx}  {ny}  {nz}\n")
        f.write(f"    0.0  0.0  0.0\n")
        for vec in cell:
            f.write(f"    {vec[0]:16.10f}  {vec[1]:16.10f}  {vec[2]:16.10f}\n")

        # Write data in Fortran order (z fastest)
        count = 0
        for ix in range(nx):
            for iy in range(ny):
                for iz in range(nz):
                    f.write(f"  {rho[ix, iy, iz]:16.8e}")
                    count += 1
                    if count % 6 == 0:
                        f.write("\n")
        if count % 6 != 0:
            f.write("\n")

        f.write("  END_DATAGRID_3D_charge\n")
        f.write("END_BLOCK_DATAGRID_3D\n")

    print(f"Converted: {chgcar_file} -> {output_xsf}")
    print(f"Grid: {rho.shape}")


# ----- Run -----
# chgcar_to_xsf('CHGCAR', 'charge.xsf')
```

### Conversion: .cube to .xsf

```python
#!/usr/bin/env python3
"""
Convert Gaussian cube to XCrySDen .xsf format using ASE.
"""
import numpy as np
from ase.io.cube import read_cube_data


def cube_to_xsf(cube_file, output_xsf='output.xsf'):
    """
    Convert .cube file to .xsf format.

    Parameters
    ----------
    cube_file : str
        Input Gaussian cube file.
    output_xsf : str
        Output XCrySDen .xsf file.
    """
    data, atoms = read_cube_data(cube_file)
    cell = atoms.get_cell()
    nx, ny, nz = data.shape

    with open(output_xsf, 'w') as f:
        f.write("CRYSTAL\n")
        f.write("PRIMVEC\n")
        for vec in cell:
            f.write(f"  {vec[0]:16.10f}  {vec[1]:16.10f}  {vec[2]:16.10f}\n")
        f.write("PRIMCOORD\n")
        f.write(f"  {len(atoms)}  1\n")
        symbols = atoms.get_chemical_symbols()
        positions = atoms.get_positions()
        for sym, pos in zip(symbols, positions):
            f.write(f"  {sym:>3s}  {pos[0]:16.10f}  {pos[1]:16.10f}  {pos[2]:16.10f}\n")
        f.write("\nBEGIN_BLOCK_DATAGRID_3D\n")
        f.write("  data_from_cube\n")
        f.write("  BEGIN_DATAGRID_3D_data\n")
        f.write(f"    {nx}  {ny}  {nz}\n")
        f.write(f"    0.0  0.0  0.0\n")
        for vec in cell:
            f.write(f"    {vec[0]:16.10f}  {vec[1]:16.10f}  {vec[2]:16.10f}\n")
        count = 0
        for ix in range(nx):
            for iy in range(ny):
                for iz in range(nz):
                    f.write(f"  {data[ix, iy, iz]:16.8e}")
                    count += 1
                    if count % 6 == 0:
                        f.write("\n")
        if count % 6 != 0:
            f.write("\n")
        f.write("  END_DATAGRID_3D_data\n")
        f.write("END_BLOCK_DATAGRID_3D\n")

    print(f"Converted: {cube_file} -> {output_xsf}")
    print(f"Grid: {data.shape}")


# ----- Run -----
cube_to_xsf('charge_total.cube', 'charge.xsf')
```

### Conversion: .xsf to .cube

```python
#!/usr/bin/env python3
"""
Convert XCrySDen .xsf to Gaussian cube format.
"""
import numpy as np
from ase.io.cube import write_cube
from ase.atoms import Atoms
import re


def parse_xsf(xsf_file):
    """
    Parse an XCrySDen .xsf file with 3D datagrid.

    Returns
    -------
    data : np.ndarray
        3D volumetric data.
    atoms : ase.Atoms
        Atoms object.
    """
    with open(xsf_file, 'r') as f:
        content = f.read()

    # Parse cell vectors
    cell_match = re.search(r'PRIMVEC\s*\n(.*?\n.*?\n.*?\n)', content)
    cell = []
    if cell_match:
        for line in cell_match.group(1).strip().split('\n'):
            cell.append([float(x) for x in line.split()])
    cell = np.array(cell)

    # Parse atomic positions
    coord_match = re.search(r'PRIMCOORD\s*\n\s*(\d+)\s+\d+\s*\n(.*?)(?:\n\s*\n|\nBEGIN)',
                             content, re.DOTALL)
    symbols = []
    positions = []
    if coord_match:
        n_atoms = int(coord_match.group(1))
        lines = coord_match.group(2).strip().split('\n')
        for line in lines[:n_atoms]:
            parts = line.split()
            sym = parts[0]
            # Handle case where symbol is an atomic number
            try:
                z = int(sym)
                from ase.data import chemical_symbols
                sym = chemical_symbols[z]
            except ValueError:
                pass
            symbols.append(sym)
            positions.append([float(parts[1]), float(parts[2]), float(parts[3])])

    atoms = Atoms(symbols=symbols, positions=positions, cell=cell, pbc=True)

    # Parse 3D data grid
    grid_match = re.search(
        r'BEGIN_DATAGRID_3D_\w+\s*\n\s*(\d+)\s+(\d+)\s+(\d+)\s*\n'
        r'\s*[\d.eE+-]+\s+[\d.eE+-]+\s+[\d.eE+-]+\s*\n'
        r'.*?\n.*?\n.*?\n'
        r'(.*?)'
        r'END_DATAGRID_3D',
        content, re.DOTALL
    )

    if grid_match:
        nx = int(grid_match.group(1))
        ny = int(grid_match.group(2))
        nz = int(grid_match.group(3))
        values_str = grid_match.group(4)
        values = [float(x) for x in values_str.split()]
        data = np.array(values[:nx * ny * nz]).reshape(nx, ny, nz)
    else:
        raise ValueError(f"No 3D datagrid found in {xsf_file}")

    return data, atoms


def xsf_to_cube(xsf_file, output_cube='output.cube'):
    """
    Convert .xsf to .cube format.
    """
    data, atoms = parse_xsf(xsf_file)

    with open(output_cube, 'w') as f:
        write_cube(f, atoms, data)

    print(f"Converted: {xsf_file} -> {output_cube}")
    print(f"Grid: {data.shape}")


# ----- Run -----
# xsf_to_cube('charge.xsf', 'charge_from_xsf.cube')
```

### Supercell of Charge Density

#### From Cube File (Python)

```python
#!/usr/bin/env python3
"""
Build a supercell of volumetric data (charge density, ELF, potential).
Tiles the 3D data array and scales the cell accordingly.
"""
import numpy as np
from ase.io.cube import read_cube_data, write_cube
from ase.build import make_supercell


def supercell_cube(cube_file, supercell_matrix, output_cube='supercell.cube'):
    """
    Create a supercell of a cube file by tiling the data grid.

    Parameters
    ----------
    cube_file : str
        Input cube file.
    supercell_matrix : tuple or list
        (nx, ny, nz) tiling factors for a diagonal supercell,
        or a 3x3 matrix for general supercells (diagonal only
        supported for volumetric data tiling).
    output_cube : str
        Output cube file.
    """
    data, atoms = read_cube_data(cube_file)

    if isinstance(supercell_matrix, (list, tuple)) and len(supercell_matrix) == 3:
        sx, sy, sz = supercell_matrix
    else:
        raise ValueError("Only diagonal supercells (nx, ny, nz) are supported "
                         "for volumetric data tiling.")

    # Tile the data
    data_super = np.tile(data, (sx, sy, sz))

    # Build the supercell atoms
    atoms_super = atoms.repeat((sx, sy, sz))

    with open(output_cube, 'w') as f:
        write_cube(f, atoms_super, data_super)

    print(f"Created supercell: {sx}x{sy}x{sz}")
    print(f"Original grid: {data.shape} -> Supercell grid: {data_super.shape}")
    print(f"Original atoms: {len(atoms)} -> Supercell atoms: {len(atoms_super)}")
    print(f"Wrote: {output_cube}")


# ----- Run -----
supercell_cube('charge_total.cube', (2, 2, 1), 'charge_2x2x1.cube')
```

#### From VASP CHGCAR (pymatgen)

```python
#!/usr/bin/env python3
"""
Build a supercell of VASP CHGCAR. Equivalent to VASPKIT task 320.
"""
import numpy as np
from pymatgen.io.vasp import Chgcar


def supercell_chgcar(chgcar_file='CHGCAR', supercell=(2, 2, 1),
                      output_file='CHGCAR_super'):
    """
    Create a supercell of a CHGCAR file.

    Parameters
    ----------
    chgcar_file : str
        Input CHGCAR.
    supercell : tuple of int
        (nx, ny, nz) repetitions.
    output_file : str
        Output CHGCAR.
    """
    chgcar = Chgcar.from_file(chgcar_file)
    structure = chgcar.structure
    data = chgcar.data['total']
    sx, sy, sz = supercell

    # Tile data
    data_super = np.tile(data, (sx, sy, sz))

    # The CHGCAR stores rho * V_cell, and V_cell scales with supercell
    # So the tiled data is already correct (rho is periodic, V scales)
    # But we need to multiply by the scaling factor because
    # CHGCAR normalization is sum(data)/N_grid = N_electrons
    # After tiling: N_grid_new = N_grid * sx*sy*sz
    # sum(data_tiled) = sum(data) * sx*sy*sz = N_electrons * sx*sy*sz (correct)

    # Build supercell structure
    structure_super = structure.copy()
    structure_super.make_supercell([
        [sx, 0, 0],
        [0, sy, 0],
        [0, 0, sz]
    ])

    # Handle spin if present
    data_dict = {'total': data_super}
    if 'diff' in chgcar.data:
        data_dict['diff'] = np.tile(chgcar.data['diff'], (sx, sy, sz))

    chgcar_super = Chgcar(structure_super, data=data_dict)
    chgcar_super.write_file(output_file)

    print(f"Created CHGCAR supercell: {sx}x{sy}x{sz}")
    print(f"Original grid: {data.shape} -> Supercell grid: {data_super.shape}")
    print(f"Original atoms: {len(structure)} -> Supercell atoms: {len(structure_super)}")
    print(f"Wrote: {output_file}")


# ----- Run -----
# supercell_chgcar('CHGCAR', (2, 2, 1), 'CHGCAR_2x2x1')
```

### Batch Conversion Utility

```python
#!/usr/bin/env python3
"""
Batch convert charge density files between formats.
Auto-detects input format and converts to the requested output format.
"""
import os
import numpy as np
from ase.io.cube import read_cube_data, write_cube
from ase.atoms import Atoms
from ase import units


def detect_format(filepath):
    """
    Auto-detect the format of a volumetric data file.

    Returns
    -------
    str
        One of: 'cube', 'xsf', 'chgcar', 'unknown'
    """
    basename = os.path.basename(filepath).upper()

    if filepath.endswith('.cube') or filepath.endswith('.cub'):
        return 'cube'
    elif filepath.endswith('.xsf') or filepath.endswith('.XSF'):
        return 'xsf'
    elif 'CHGCAR' in basename or 'PARCHG' in basename or 'ELFCAR' in basename or 'LOCPOT' in basename:
        return 'chgcar'
    else:
        # Try to detect from content
        with open(filepath, 'r') as f:
            first_lines = f.read(500)
        if 'CRYSTAL' in first_lines or 'BEGIN_BLOCK_DATAGRID' in first_lines:
            return 'xsf'
        elif 'Cube' in first_lines or first_lines.strip().startswith('  '):
            return 'cube'
        else:
            return 'unknown'


def convert_volumetric(input_file, output_file, input_format=None,
                        output_format=None):
    """
    Convert between volumetric data formats.

    Parameters
    ----------
    input_file : str
        Input file path.
    output_file : str
        Output file path.
    input_format : str or None
        'cube', 'xsf', or 'chgcar'. Auto-detected if None.
    output_format : str or None
        Target format. Auto-detected from output_file extension if None.
    """
    # Detect formats
    if input_format is None:
        input_format = detect_format(input_file)
    if output_format is None:
        if output_file.endswith('.cube') or output_file.endswith('.cub'):
            output_format = 'cube'
        elif output_file.endswith('.xsf'):
            output_format = 'xsf'
        elif 'CHGCAR' in output_file.upper():
            output_format = 'chgcar'
        else:
            raise ValueError(f"Cannot determine output format from: {output_file}")

    print(f"Converting: {input_file} ({input_format}) -> {output_file} ({output_format})")

    # Read input
    if input_format == 'cube':
        data, atoms = read_cube_data(input_file)
        # data in e/bohr^3
        data_unit = 'e/bohr3'
    elif input_format == 'chgcar':
        from pymatgen.io.vasp import Chgcar
        chgcar = Chgcar.from_file(input_file)
        structure = chgcar.structure
        raw_data = chgcar.data['total']
        vol = structure.volume
        BOHR = units.Bohr
        data = (raw_data / vol) * (BOHR ** 3)  # -> e/bohr^3
        atoms = Atoms(
            symbols=[str(s.specie) for s in structure],
            positions=[s.coords for s in structure],
            cell=structure.lattice.matrix,
            pbc=True
        )
        data_unit = 'e/bohr3'
    elif input_format == 'xsf':
        # Use custom parser
        from io import StringIO
        # Simplified: read via the xsf parser defined earlier
        # For production, import the parse_xsf function
        raise NotImplementedError(
            "XSF reading requires the parse_xsf function. "
            "Use the xsf_to_cube script directly."
        )
    else:
        raise ValueError(f"Unknown input format: {input_format}")

    # Write output
    if output_format == 'cube':
        with open(output_file, 'w') as f:
            write_cube(f, atoms, data)
    elif output_format == 'chgcar':
        from pymatgen.io.vasp import Chgcar as ChgcarWriter
        from pymatgen.core import Structure, Lattice
        BOHR = units.Bohr
        vol = atoms.get_volume()
        rho_ang = data / (BOHR ** 3)
        chgcar_data = rho_ang * vol
        lattice = Lattice(atoms.get_cell())
        species = atoms.get_chemical_symbols()
        coords = atoms.get_positions()
        structure = Structure(lattice, species, coords, coords_are_cartesian=True)
        chg = ChgcarWriter(structure, data={'total': chgcar_data})
        chg.write_file(output_file)
    elif output_format == 'xsf':
        cell = atoms.get_cell()
        nx, ny, nz = data.shape
        with open(output_file, 'w') as f:
            f.write("CRYSTAL\nPRIMVEC\n")
            for vec in cell:
                f.write(f"  {vec[0]:16.10f}  {vec[1]:16.10f}  {vec[2]:16.10f}\n")
            f.write("PRIMCOORD\n")
            f.write(f"  {len(atoms)}  1\n")
            syms = atoms.get_chemical_symbols()
            pos = atoms.get_positions()
            for s, p in zip(syms, pos):
                f.write(f"  {s:>3s}  {p[0]:16.10f}  {p[1]:16.10f}  {p[2]:16.10f}\n")
            f.write("\nBEGIN_BLOCK_DATAGRID_3D\n  converted_data\n")
            f.write("  BEGIN_DATAGRID_3D_data\n")
            f.write(f"    {nx}  {ny}  {nz}\n")
            f.write("    0.0  0.0  0.0\n")
            for vec in cell:
                f.write(f"    {vec[0]:16.10f}  {vec[1]:16.10f}  {vec[2]:16.10f}\n")
            count = 0
            for ix in range(nx):
                for iy in range(ny):
                    for iz in range(nz):
                        f.write(f"  {data[ix, iy, iz]:16.8e}")
                        count += 1
                        if count % 6 == 0:
                            f.write("\n")
            if count % 6 != 0:
                f.write("\n")
            f.write("  END_DATAGRID_3D_data\nEND_BLOCK_DATAGRID_3D\n")

    print(f"Done. Grid: {data.shape}")


# ----- Run -----
# convert_volumetric('CHGCAR', 'charge.cube')
# convert_volumetric('charge.cube', 'charge.xsf')
# convert_volumetric('CHGCAR', 'charge.xsf')
```

### Spin-Polarized CHGCAR Handling

```python
#!/usr/bin/env python3
"""
Split or merge spin-polarized CHGCAR files.
VASP stores spin-polarized CHGCAR with two data blocks:
  - total = rho_up + rho_down
  - diff  = rho_up - rho_down
"""
import numpy as np
from pymatgen.io.vasp import Chgcar


def split_spin_chgcar(chgcar_file='CHGCAR',
                       output_up='CHGCAR_up',
                       output_down='CHGCAR_down',
                       output_spin='CHGCAR_spin'):
    """
    Split spin-polarized CHGCAR into spin-up, spin-down, and spin density.

    Parameters
    ----------
    chgcar_file : str
        Spin-polarized CHGCAR (has both 'total' and 'diff').
    output_up, output_down, output_spin : str
        Output file paths.
    """
    chgcar = Chgcar.from_file(chgcar_file)

    if 'diff' not in chgcar.data:
        print("Not a spin-polarized CHGCAR (no 'diff' data). Nothing to split.")
        return

    total = chgcar.data['total']
    diff = chgcar.data['diff']

    rho_up = (total + diff) / 2.0
    rho_down = (total - diff) / 2.0

    structure = chgcar.structure

    # Write spin-up
    chg_up = Chgcar(structure, data={'total': rho_up})
    chg_up.write_file(output_up)
    print(f"Wrote spin-up: {output_up}")

    # Write spin-down
    chg_down = Chgcar(structure, data={'total': rho_down})
    chg_down.write_file(output_down)
    print(f"Wrote spin-down: {output_down}")

    # Write spin density (magnetization density)
    chg_spin = Chgcar(structure, data={'total': diff})
    chg_spin.write_file(output_spin)
    print(f"Wrote spin density: {output_spin}")

    vol = structure.volume
    ngrid = total.size
    dV = vol / ngrid
    n_up = np.sum(rho_up) / vol
    n_down = np.sum(rho_down) / vol
    mag = np.sum(diff) / vol
    print(f"\nElectrons up:   {n_up:.4f}")
    print(f"Electrons down: {n_down:.4f}")
    print(f"Net magnetization: {mag:.4f} mu_B")


def merge_spin_chgcar(chgcar_up='CHGCAR_up', chgcar_down='CHGCAR_down',
                       output='CHGCAR_spin_merged'):
    """
    Merge separate spin-up and spin-down CHGCARs into a single
    spin-polarized CHGCAR.
    """
    chg_up = Chgcar.from_file(chgcar_up)
    chg_down = Chgcar.from_file(chgcar_down)

    total = chg_up.data['total'] + chg_down.data['total']
    diff = chg_up.data['total'] - chg_down.data['total']

    chg_merged = Chgcar(chg_up.structure, data={'total': total, 'diff': diff})
    chg_merged.write_file(output)
    print(f"Merged: {chgcar_up} + {chgcar_down} -> {output}")


# ----- Run -----
# split_spin_chgcar('CHGCAR')
# merge_spin_chgcar('CHGCAR_up', 'CHGCAR_down', 'CHGCAR_merged')
```

## Key Parameters

### Format Specifications

| Format | Extension | Code Origin | Units | Grid Order |
|--------|-----------|-------------|-------|------------|
| Gaussian cube | `.cube` | QE pp.x, Gaussian | e/bohr^3 | x-fast (C order) |
| VASP CHGCAR | `CHGCAR` | VASP | rho * V_cell | x-fast (Fortran-like) |
| XCrySDen XSF | `.xsf` | Various | User-defined | x-fast |
| VASP LOCPOT | `LOCPOT` | VASP | eV (potential) | Same as CHGCAR |
| VASP ELFCAR | `ELFCAR` | VASP | Dimensionless (0-1) | Same as CHGCAR |
| VASP PARCHG | `PARCHG` | VASP | Same as CHGCAR | Same as CHGCAR |

### Unit Conversions

| From | To | Multiply By |
|------|----|-------------|
| e/bohr^3 | e/Angstrom^3 | 1 / (0.529177)^3 = 6.7483 |
| e/Angstrom^3 | e/bohr^3 | (0.529177)^3 = 0.14818 |
| CHGCAR value | e/Angstrom^3 | 1 / V_cell (Angstrom^3) |
| Ry | eV | 13.6057 |
| Bohr | Angstrom | 0.529177 |

### VASPKIT Tasks

| Task | Description |
|------|-------------|
| 320 | Build supercell of CHGCAR |
| 328 | Convert CHGCAR to Gaussian cube |
| 329 | Convert CHGCAR to XCrySDen xsf |

## Interpreting Results

### Verifying Conversion Accuracy

- **Total electron count**: Integrate the charge density over the cell. Must be the same before and after conversion (within numerical precision).
- **Grid dimensions**: Should remain the same unless explicitly resampled.
- **Visual comparison**: Open both files in VESTA or another viewer and compare isosurfaces.

### Common Normalization Pitfalls

- **CHGCAR normalization**: VASP stores `rho * V_cell`, not `rho`. Forgetting this factor leads to values off by a factor of V_cell.
- **Cube file convention**: QE pp.x writes density in e/bohr^3. Some other codes may use e/Angstrom^3 or atomic units. Check the header.
- **XSF convention**: No universal unit convention. Typically matches whatever was computed.

### When to Use Each Format

| Format | Best For |
|--------|----------|
| .cube | VESTA, VMD, most visualization tools; cross-code compatibility |
| CHGCAR | VASP ecosystem (VASPKIT, pymatgen VASP tools, Bader analysis) |
| .xsf | XCrySDen visualization; compact with structure info |

## Common Issues

| Issue | Cause | Solution |
|-------|-------|----------|
| Converted file has wrong total electrons | Normalization error | Verify unit conversion (CHGCAR stores rho*V_cell; cube stores e/bohr^3) |
| VESTA shows garbage isosurface | Grid order mismatch (C vs Fortran) | Check array axis ordering; may need transpose |
| Cube file from CHGCAR looks inside-out | Missing volume normalization | Divide CHGCAR data by V_cell before converting |
| Supercell has visible seam lines in visualization | Floating-point precision at tile boundaries | Normal for visualization; not a real discontinuity |
| XSF file not readable by XCrySDen | Incorrect format syntax | Verify PRIMVEC, PRIMCOORD, BEGIN_BLOCK_DATAGRID_3D sections |
| Spin-polarized CHGCAR only has one block | Non-spin-polarized calculation | Re-run VASP with ISPIN=2 for spin-polarized CHGCAR |
| File size doubles unexpectedly | Converted to text format from binary | Cube and XSF are text; CHGCAR is text; all are large for fine grids |
| pymatgen Chgcar.from_file fails | CHGCAR is corrupt or truncated | Re-copy from the VASP run directory; check disk space during calculation |
