# Structure Format Conversion

## When to Use

- You need to convert a crystal structure between different file formats (CIF, POSCAR, XYZ, PDB, QE input, LAMMPS data, extxyz).
- You received a structure in one format and need it in another for a specific code.
- You need to batch-convert multiple structure files.
- You want to convert between periodic (crystal) and molecular (cluster) representations.
- You need to read a structure from one code's output and write it as input for another code (VASPKIT 419 equivalent).

## Method Selection

| Criterion | pymatgen | ASE | Direct conversion |
|---|---|---|---|
| CIF <-> POSCAR | Yes | Yes | Best with pymatgen (preserves symmetry info) |
| CIF <-> QE input | Yes (PWInput) | Yes (espresso module) | pymatgen preferred |
| CIF <-> XYZ | Yes | Yes | Either works |
| CIF <-> PDB | Yes | Yes | Either works |
| POSCAR <-> LAMMPS data | Yes (LammpsData) | Yes | pymatgen preferred |
| Any <-> extxyz | Limited | Yes (native format) | ASE preferred |
| Batch conversion | Both | Both | Script provided below |
| Available now | Yes | Yes | Yes |

## Prerequisites

- A structure file in any supported format.
- Python packages: `pymatgen`, `ase`, `numpy` (pre-installed).
- For QE input generation: pseudopotential files (optional, for complete input).

---

## Detailed Steps

### Method A: pymatgen -- Format Conversion

#### Step A1: Universal format converter

```python
#!/usr/bin/env python3
"""
Universal structure format converter using pymatgen.
Converts between CIF, POSCAR, XYZ, PDB, and JSON formats.
Equivalent to VASPKIT function 419.
"""
import os
import sys
from pymatgen.core.structure import Structure, Molecule
from pymatgen.io.vasp import Poscar
from pymatgen.io.cif import CifWriter
from pymatgen.io.xyz import XYZ

def convert_structure(input_file, output_file, output_format=None):
    """
    Convert a crystal structure between formats.

    Supported input formats: CIF, POSCAR/CONTCAR, XDATCAR, JSON, YAML
    Supported output formats: cif, poscar, xyz, pdb, json, yaml

    Parameters
    ----------
    input_file : str
        Path to input structure file.
    output_file : str
        Path to output file.
    output_format : str or None
        Output format. If None, inferred from file extension.
    """
    # Determine input format and load
    input_basename = os.path.basename(input_file).upper()

    if input_basename in ("POSCAR", "CONTCAR") or input_file.endswith(".vasp"):
        structure = Structure.from_file(input_file)
    elif input_file.endswith(".cif"):
        structure = Structure.from_file(input_file)
    elif input_file.endswith(".json"):
        structure = Structure.from_file(input_file)
    elif input_file.endswith(".yaml") or input_file.endswith(".yml"):
        structure = Structure.from_file(input_file)
    elif input_file.endswith(".xsf"):
        structure = Structure.from_file(input_file)
    else:
        # Try generic loading
        structure = Structure.from_file(input_file)

    print(f"Loaded: {input_file}")
    print(f"  Formula: {structure.composition.reduced_formula}")
    print(f"  Atoms: {len(structure)}")
    print(f"  Volume: {structure.volume:.4f} A^3")

    # Determine output format
    if output_format is None:
        ext = os.path.splitext(output_file)[1].lower()
        format_map = {
            ".cif": "cif",
            ".vasp": "poscar",
            ".xyz": "xyz",
            ".pdb": "pdb",
            ".json": "json",
            ".yaml": "yaml",
            ".yml": "yaml",
            ".xsf": "xsf",
        }
        output_format = format_map.get(ext, "cif")
        # Handle POSCAR/CONTCAR without extension
        if os.path.basename(output_file).upper() in ("POSCAR", "CONTCAR"):
            output_format = "poscar"

    # Write output
    if output_format == "cif":
        writer = CifWriter(structure, symprec=0.01)
        writer.write_file(output_file)
    elif output_format == "poscar":
        poscar = Poscar(structure,
                        comment=f"{structure.composition.reduced_formula}")
        poscar.write_file(output_file)
    elif output_format == "xyz":
        # XYZ is a molecular format -- include lattice as comment
        structure.to(output_file, fmt="xyz")
    elif output_format == "pdb":
        structure.to(output_file, fmt="pdb")
    elif output_format == "json":
        structure.to(output_file, fmt="json")
    elif output_format == "yaml":
        structure.to(output_file, fmt="yaml")
    elif output_format == "xsf":
        structure.to(output_file, fmt="xsf")
    else:
        structure.to(output_file, fmt=output_format)

    print(f"Written: {output_file} (format: {output_format})")
    return structure

# ── Example conversions ──────────────────────────────────────────
# CIF -> POSCAR
# convert_structure("structure.cif", "POSCAR")

# POSCAR -> CIF
# convert_structure("POSCAR", "structure_from_poscar.cif")

# CIF -> XYZ
# convert_structure("structure.cif", "structure.xyz")

# Any -> JSON (pymatgen native, lossless)
# convert_structure("structure.cif", "structure.json")

# ── Interactive usage ─────────────────────────────────────────────
input_file = "structure.cif"    # <-- Change this
output_file = "POSCAR"          # <-- Change this
structure = convert_structure(input_file, output_file)
```

#### Step A2: CIF to POSCAR (detailed)

```python
#!/usr/bin/env python3
"""
Convert CIF to VASP POSCAR with various options:
- Primitive vs conventional cell
- Direct vs Cartesian coordinates
- With or without selective dynamics
- With symmetry information preserved in CIF
"""
from pymatgen.core.structure import Structure
from pymatgen.io.vasp import Poscar
from pymatgen.io.cif import CifWriter, CifParser
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer

# ── Load CIF ──────────────────────────────────────────────────────
cif_file = "structure.cif"

# Method 1: Simple loading (uses occupancy-based site matching)
structure = Structure.from_file(cif_file)

# Method 2: CifParser with options (handles partial occupancies, etc.)
# parser = CifParser(cif_file, occupancy_tolerance=1.0)
# structure = parser.parse_structures()[0]

print(f"Loaded CIF: {structure.composition.reduced_formula}")
print(f"  Atoms: {len(structure)}")

sga = SpacegroupAnalyzer(structure, symprec=0.01)
print(f"  Space group: {sga.get_space_group_symbol()}")

# ── Option 1: Direct conversion (as-is) ──────────────────────────
poscar = Poscar(structure, comment=f"{structure.composition.reduced_formula}")
poscar.write_file("POSCAR")
print(f"\nPOSCAR written: {len(structure)} atoms")

# ── Option 2: Convert to primitive cell first ─────────────────────
primitive = sga.get_primitive_standard_structure()
poscar_prim = Poscar(primitive, comment=f"{primitive.composition.reduced_formula} primitive")
poscar_prim.write_file("POSCAR_primitive")
print(f"POSCAR_primitive: {len(primitive)} atoms")

# ── Option 3: Convert to conventional cell ────────────────────────
conventional = sga.get_conventional_standard_structure()
poscar_conv = Poscar(conventional,
                     comment=f"{conventional.composition.reduced_formula} conventional")
poscar_conv.write_file("POSCAR_conventional")
print(f"POSCAR_conventional: {len(conventional)} atoms")

# ── Option 4: Cartesian coordinates ──────────────────────────────
poscar_cart = Poscar(structure, direct=False,
                     comment=f"{structure.composition.reduced_formula} Cartesian")
poscar_cart.write_file("POSCAR_cartesian")
print(f"POSCAR_cartesian: Cartesian coordinates")

# ── Reverse: POSCAR to CIF with symmetry ─────────────────────────
structure_back = Structure.from_file("POSCAR")
# Write CIF with detected symmetry
cif_writer = CifWriter(structure_back, symprec=0.01)
cif_writer.write_file("structure_from_poscar.cif")
print(f"\nCIF from POSCAR: structure_from_poscar.cif")
# Write CIF without symmetry (P1)
cif_writer_p1 = CifWriter(structure_back, symprec=None)
cif_writer_p1.write_file("structure_P1.cif")
print(f"CIF (P1, no symmetry): structure_P1.cif")
```

#### Step A3: Convert to/from QE input format

```python
#!/usr/bin/env python3
"""
Convert between CIF/POSCAR and Quantum ESPRESSO pw.x input format.
Handles both reading QE output and writing QE input.
"""
import os
import numpy as np
from pymatgen.core.structure import Structure
from pymatgen.io.pwscf import PWInput, PWOutput

# ═══════════════════════════════════════════════════════════════════
# Part 1: CIF/POSCAR -> QE Input
# ═══════════════════════════════════════════════════════════════════

def structure_to_qe_input(structure, output_file="pw.in",
                          pseudo_dir="./pseudo", pseudo_map=None,
                          calculation="scf", ecutwfc=50.0, ecutrho=400.0,
                          kgrid=None):
    """
    Convert a pymatgen Structure to a QE pw.x input file.

    Parameters
    ----------
    structure : pymatgen Structure
    output_file : str
    pseudo_dir : str
    pseudo_map : dict or None
        Element -> pseudopotential filename mapping.
        If None, uses default naming convention.
    calculation : str
        "scf", "relax", "vc-relax", "nscf", "bands"
    ecutwfc, ecutrho : float
        Cutoffs in Ry.
    kgrid : tuple or None
        K-point grid (nk1, nk2, nk3). If None, auto-determined.
    """
    # Build pseudo map if not provided
    if pseudo_map is None:
        pseudo_map = {}
        for el in structure.composition.elements:
            symbol = str(el)
            # Try to find in pseudo_dir
            if os.path.exists(pseudo_dir):
                for fname in os.listdir(pseudo_dir):
                    if fname.endswith(".UPF") and fname.lower().startswith(symbol.lower()):
                        pseudo_map[symbol] = fname
                        break
            if symbol not in pseudo_map:
                pseudo_map[symbol] = f"{symbol}.UPF"

    # Auto k-grid
    if kgrid is None:
        recip = structure.lattice.reciprocal_lattice.abc
        kgrid = tuple(max(1, int(round(40 / (2 * np.pi) * rl))) for rl in recip)

    # Build QE input
    control = {
        "calculation": calculation,
        "prefix": structure.composition.reduced_formula.replace(" ", ""),
        "outdir": "./tmp",
        "pseudo_dir": os.path.abspath(pseudo_dir),
        "tprnfor": True,
        "tstress": True,
    }

    system_params = {
        "ecutwfc": ecutwfc,
        "ecutrho": ecutrho,
        "occupations": "smearing",
        "smearing": "cold",
        "degauss": 0.01,
    }

    electrons = {"conv_thr": 1.0e-8, "mixing_beta": 0.7}

    ions = {"ion_dynamics": "bfgs"} if calculation in ("relax", "vc-relax") else None
    cell = {"cell_dynamics": "bfgs", "press": 0.0} if calculation == "vc-relax" else None

    pw_input = PWInput(
        structure,
        pseudo=pseudo_map,
        control=control,
        system=system_params,
        electrons=electrons,
        ions=ions,
        cell=cell,
        kpoints_grid=kgrid,
    )
    pw_input.write_file(output_file)
    print(f"QE input written to: {output_file}")
    print(f"  K-grid: {kgrid[0]}x{kgrid[1]}x{kgrid[2]}")
    return pw_input

# Example: CIF -> QE
structure = Structure.from_file("structure.cif")
structure_to_qe_input(structure, "pw_from_cif.in")

# Example: POSCAR -> QE
# structure = Structure.from_file("POSCAR")
# structure_to_qe_input(structure, "pw_from_poscar.in")


# ═══════════════════════════════════════════════════════════════════
# Part 2: QE Output -> CIF/POSCAR
# ═══════════════════════════════════════════════════════════════════

def qe_output_to_structure(qe_output_file):
    """
    Extract the final structure from a QE pw.x output file.
    Works for scf, relax, and vc-relax calculations.
    """
    from ase.io.espresso import read_espresso_out
    from pymatgen.io.ase import AseAtomsAdaptor

    atoms = read_espresso_out(qe_output_file)
    structure = AseAtomsAdaptor.get_structure(atoms)
    return structure

# Example: QE output -> CIF and POSCAR
# structure = qe_output_to_structure("pw.out")
# structure.to("from_qe.cif", fmt="cif")
# structure.to("CONTCAR", fmt="poscar")
# print(f"Converted QE output: {structure.composition.reduced_formula}")


# ═══════════════════════════════════════════════════════════════════
# Part 3: QE Input -> Structure (read existing QE input)
# ═══════════════════════════════════════════════════════════════════

def read_qe_input(qe_input_file):
    """Read a QE pw.x input file and extract the structure."""
    pw_input = PWInput.from_file(qe_input_file)
    return pw_input.structure

# Example: Read QE input
# structure = read_qe_input("pw.in")
# print(f"From QE input: {structure.composition.reduced_formula}, {len(structure)} atoms")
# structure.to("from_qe_input.cif", fmt="cif")
```

#### Step A4: Convert to/from LAMMPS data format

```python
#!/usr/bin/env python3
"""
Convert between CIF/POSCAR and LAMMPS data format.
Handles atom types, charges, and box dimensions.
"""
import numpy as np
from pymatgen.core.structure import Structure
from pymatgen.io.lammps.data import LammpsData
from pymatgen.io.vasp import Poscar

# ═══════════════════════════════════════════════════════════════════
# Part 1: CIF/POSCAR -> LAMMPS Data
# ═══════════════════════════════════════════════════════════════════

def structure_to_lammps(structure, output_file="structure.lammps",
                        atom_style="atomic"):
    """
    Convert a pymatgen Structure to LAMMPS data file.

    Parameters
    ----------
    structure : pymatgen Structure
    output_file : str
    atom_style : str
        "atomic" (no charge), "charge" (with charges), "full" (charge + mol ID)
    """
    lammps_data = LammpsData.from_structure(structure, atom_style=atom_style)
    lammps_data.write_file(output_file)
    print(f"LAMMPS data written to: {output_file}")
    print(f"  Atom style: {atom_style}")
    print(f"  Atoms: {len(structure)}")
    print(f"  Atom types: {len(set(str(sp) for sp in structure.species))}")

    # Show box dimensions
    box = structure.lattice.matrix
    print(f"  Box (a): [{box[0][0]:.4f}, {box[0][1]:.4f}, {box[0][2]:.4f}]")
    print(f"  Box (b): [{box[1][0]:.4f}, {box[1][1]:.4f}, {box[1][2]:.4f}]")
    print(f"  Box (c): [{box[2][0]:.4f}, {box[2][1]:.4f}, {box[2][2]:.4f}]")

    return lammps_data

# Example: CIF -> LAMMPS
structure = Structure.from_file("structure.cif")
structure_to_lammps(structure, "structure.lammps", atom_style="atomic")

# For charged systems (e.g., ionic materials):
# structure_to_lammps(structure, "structure_charged.lammps", atom_style="charge")


# ═══════════════════════════════════════════════════════════════════
# Part 2: LAMMPS Data -> CIF/POSCAR
# ═══════════════════════════════════════════════════════════════════

def lammps_to_structure(lammps_file, atom_style="atomic", element_map=None):
    """
    Read a LAMMPS data file and convert to pymatgen Structure.

    Parameters
    ----------
    lammps_file : str
    atom_style : str
    element_map : dict or None
        Mapping from LAMMPS type ID to element symbol.
        e.g., {1: "Si", 2: "O"}. Required if types are numeric.
    """
    lammps_data = LammpsData.from_file(lammps_file, atom_style=atom_style)
    structure = lammps_data.structure
    print(f"Read LAMMPS data: {structure.composition.reduced_formula}")
    print(f"  Atoms: {len(structure)}")
    return structure

# Example: LAMMPS -> CIF
# structure = lammps_to_structure("structure.lammps")
# structure.to("from_lammps.cif", fmt="cif")
# Poscar(structure).write_file("POSCAR_from_lammps")
```

#### Step A5: Convert to/from PDB format

```python
#!/usr/bin/env python3
"""
Convert between CIF/POSCAR and PDB format.
PDB is common in biological and soft-matter simulations.
"""
from pymatgen.core.structure import Structure, Molecule
from pymatgen.io.vasp import Poscar

# ═══════════════════════════════════════════════════════════════════
# Part 1: CIF/POSCAR -> PDB
# ═══════════════════════════════════════════════════════════════════

structure = Structure.from_file("structure.cif")
structure.to("structure.pdb", fmt="pdb")
print(f"PDB written: structure.pdb ({len(structure)} atoms)")

# ═══════════════════════════════════════════════════════════════════
# Part 2: PDB -> CIF/POSCAR
# ═══════════════════════════════════════════════════════════════════

# PDB can be periodic or non-periodic
# For periodic structures:
# structure = Structure.from_file("structure.pdb")
# structure.to("from_pdb.cif", fmt="cif")
# Poscar(structure).write_file("POSCAR_from_pdb")

# For molecules (non-periodic):
# molecule = Molecule.from_file("molecule.pdb")
# molecule.to("molecule.xyz", fmt="xyz")
```

### Method A (continued): ASE-based format conversion

#### Step A6: ASE universal converter

```python
#!/usr/bin/env python3
"""
Universal structure format converter using ASE.
ASE supports many formats that pymatgen does not (e.g., extxyz, TURBOMOLE, Gaussian).
"""
from ase.io import read, write, formats
from pymatgen.io.ase import AseAtomsAdaptor

# ── List supported ASE formats ────────────────────────────────────
print("ASE supported formats (selection):")
useful_formats = {
    "vasp": "VASP POSCAR/CONTCAR",
    "cif": "Crystallographic Information File",
    "xyz": "XYZ (simple molecular)",
    "extxyz": "Extended XYZ (with cell, properties)",
    "pdb": "Protein Data Bank",
    "espresso-in": "Quantum ESPRESSO pw.x input",
    "espresso-out": "Quantum ESPRESSO pw.x output",
    "lammps-data": "LAMMPS data file",
    "lammps-dump-text": "LAMMPS dump file",
    "gen": "DFTB+ GEN format",
    "gaussian-in": "Gaussian input",
    "turbomole": "TURBOMOLE coord file",
    "aims": "FHI-aims geometry.in",
    "elk": "Elk GEOMETRY.OUT",
    "xsf": "XCrySDen structure file",
    "proteindatabank": "PDB (ASE native)",
}
for fmt, desc in useful_formats.items():
    print(f"  {fmt:20s} - {desc}")

# ── Universal conversion function ────────────────────────────────
def ase_convert(input_file, output_file, input_format=None, output_format=None):
    """
    Convert structure between any ASE-supported formats.

    Parameters
    ----------
    input_file, output_file : str
        File paths.
    input_format, output_format : str or None
        ASE format strings. If None, inferred from filename.
    """
    # Map common filenames to formats
    filename_map = {
        "POSCAR": "vasp", "CONTCAR": "vasp",
        "XDATCAR": "vasp-xdatcar",
    }

    if input_format is None:
        basename = input_file.split("/")[-1].upper()
        input_format = filename_map.get(basename, None)

    if output_format is None:
        basename = output_file.split("/")[-1].upper()
        output_format = filename_map.get(basename, None)

    atoms = read(input_file, format=input_format)
    print(f"Read: {input_file} -> {atoms.get_chemical_formula()} ({len(atoms)} atoms)")

    write(output_file, atoms, format=output_format)
    print(f"Written: {output_file}")
    return atoms

# ── Example conversions ──────────────────────────────────────────
# CIF -> extxyz (preserves cell, forces, energy if available)
# ase_convert("structure.cif", "structure.extxyz", output_format="extxyz")

# POSCAR -> extended XYZ
# ase_convert("POSCAR", "structure.extxyz", input_format="vasp", output_format="extxyz")

# QE output -> CIF
# ase_convert("pw.out", "from_qe.cif", input_format="espresso-out", output_format="cif")

# LAMMPS dump -> POSCAR
# ase_convert("dump.lammpstrj", "POSCAR", input_format="lammps-dump-text", output_format="vasp")

# ── Convert with pymatgen interop ─────────────────────────────────
# Read with ASE, convert to pymatgen for further processing
atoms = read("structure.cif")
structure = AseAtomsAdaptor.get_structure(atoms)
print(f"\nConverted to pymatgen: {structure.composition.reduced_formula}")
# Now use any pymatgen feature
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
sga = SpacegroupAnalyzer(structure, symprec=0.01)
print(f"Space group: {sga.get_space_group_symbol()}")
```

#### Step A7: Batch conversion

```python
#!/usr/bin/env python3
"""
Batch convert multiple structure files between formats.
"""
import os
import glob
from pymatgen.core.structure import Structure
from pymatgen.io.vasp import Poscar
from pymatgen.io.cif import CifWriter

def batch_convert(input_pattern, output_dir, output_format="poscar",
                  use_primitive=False, symprec=0.01):
    """
    Batch convert structure files.

    Parameters
    ----------
    input_pattern : str
        Glob pattern for input files (e.g., "*.cif", "structures/*.vasp")
    output_dir : str
        Directory for output files.
    output_format : str
        "poscar", "cif", "xyz", "json", "extxyz"
    use_primitive : bool
        Convert to primitive cell before writing.
    symprec : float
        Symmetry tolerance for primitive cell detection.
    """
    os.makedirs(output_dir, exist_ok=True)
    input_files = sorted(glob.glob(input_pattern))

    if not input_files:
        print(f"No files matching: {input_pattern}")
        return

    print(f"Converting {len(input_files)} files to {output_format}...")

    ext_map = {
        "poscar": ".vasp",
        "cif": ".cif",
        "xyz": ".xyz",
        "json": ".json",
        "extxyz": ".extxyz",
        "pdb": ".pdb",
        "xsf": ".xsf",
    }
    ext = ext_map.get(output_format, ".out")

    results = []
    for input_file in input_files:
        basename = os.path.splitext(os.path.basename(input_file))[0]
        output_file = os.path.join(output_dir, basename + ext)

        try:
            structure = Structure.from_file(input_file)

            if use_primitive:
                from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
                sga = SpacegroupAnalyzer(structure, symprec=symprec)
                structure = sga.get_primitive_standard_structure()

            if output_format == "poscar":
                poscar = Poscar(structure, comment=basename)
                poscar.write_file(output_file)
            elif output_format == "cif":
                writer = CifWriter(structure, symprec=symprec)
                writer.write_file(output_file)
            else:
                structure.to(output_file, fmt=output_format)

            results.append({
                "input": input_file,
                "output": output_file,
                "formula": str(structure.composition.reduced_formula),
                "atoms": len(structure),
                "status": "OK",
            })
            print(f"  OK: {input_file} -> {output_file} "
                  f"({structure.composition.reduced_formula}, {len(structure)} atoms)")

        except Exception as e:
            results.append({
                "input": input_file,
                "output": output_file,
                "status": f"ERROR: {e}",
            })
            print(f"  ERROR: {input_file} -> {e}")

    print(f"\nConverted: {sum(1 for r in results if r['status'] == 'OK')}/{len(results)}")
    return results

# ── Examples ──────────────────────────────────────────────────────
# Convert all CIFs to POSCAR
# batch_convert("structures/*.cif", "poscars/", output_format="poscar")

# Convert all POSCARs to CIF (with symmetry)
# batch_convert("poscars/*.vasp", "cifs/", output_format="cif")

# Convert to primitive cells
# batch_convert("*.cif", "primitive/", output_format="poscar", use_primitive=True)
```

#### Step A8: LAMMPS dump trajectory conversion

```python
#!/usr/bin/env python3
"""
Convert LAMMPS dump trajectory to other formats.
Extracts individual frames or converts entire trajectory.
"""
from ase.io import read, write
from pymatgen.io.ase import AseAtomsAdaptor

def convert_lammps_trajectory(dump_file, output_prefix="frame",
                              output_format="vasp", every_n=1):
    """
    Read a LAMMPS dump trajectory and convert frames.

    Parameters
    ----------
    dump_file : str
        LAMMPS dump file.
    output_prefix : str
        Prefix for output files.
    output_format : str
        ASE output format.
    every_n : int
        Write every N-th frame (for large trajectories).
    """
    # Read all frames
    frames = read(dump_file, index=":", format="lammps-dump-text")
    print(f"Read {len(frames)} frames from {dump_file}")

    for i, atoms in enumerate(frames):
        if i % every_n != 0:
            continue

        ext_map = {"vasp": ".vasp", "cif": ".cif", "xyz": ".xyz", "extxyz": ".extxyz"}
        ext = ext_map.get(output_format, f".{output_format}")
        output_file = f"{output_prefix}_{i:06d}{ext}"
        write(output_file, atoms, format=output_format)

    n_written = len(range(0, len(frames), every_n))
    print(f"Written {n_written} frames (every {every_n})")

    # Write entire trajectory as extxyz (single file, multiple frames)
    write(f"{output_prefix}_trajectory.extxyz", frames, format="extxyz")
    print(f"Full trajectory: {output_prefix}_trajectory.extxyz")

# Example:
# convert_lammps_trajectory("dump.lammpstrj", "frame", "vasp", every_n=10)
```

---

### Method B: QE-specific format operations

```python
#!/usr/bin/env python3
"""
QE-specific format conversions.
Extract structures from QE output files and convert to other formats.
"""
import re
import numpy as np
from ase.io import read as ase_read, write as ase_write
from ase.io.espresso import read_espresso_out, read_espresso_in
from pymatgen.io.ase import AseAtomsAdaptor
from pymatgen.io.vasp import Poscar

# ── QE output -> CIF / POSCAR ────────────────────────────────────
def qe_out_to_formats(qe_output, output_prefix="converted"):
    """Extract final structure from QE output and save in multiple formats."""
    # Read with ASE (handles scf, relax, vc-relax)
    atoms = read_espresso_out(qe_output)
    structure = AseAtomsAdaptor.get_structure(atoms)

    # Write multiple formats
    outputs = {}

    # CIF
    cif_file = f"{output_prefix}.cif"
    structure.to(cif_file, fmt="cif")
    outputs["cif"] = cif_file

    # POSCAR
    poscar_file = f"{output_prefix}.vasp"
    Poscar(structure).write_file(poscar_file)
    outputs["poscar"] = poscar_file

    # Extended XYZ (includes cell, energy, forces if available)
    xyz_file = f"{output_prefix}.extxyz"
    ase_write(xyz_file, atoms, format="extxyz")
    outputs["extxyz"] = xyz_file

    # XYZ (simple molecular format)
    xyz_simple = f"{output_prefix}.xyz"
    ase_write(xyz_simple, atoms, format="xyz")
    outputs["xyz"] = xyz_simple

    # JSON (pymatgen native, lossless)
    json_file = f"{output_prefix}.json"
    structure.to(json_file, fmt="json")
    outputs["json"] = json_file

    print(f"Converted QE output to:")
    for fmt, path in outputs.items():
        print(f"  {fmt:8s}: {path}")

    return structure, outputs

# Example:
# structure, files = qe_out_to_formats("pw.out", "relaxed_structure")


# ── QE output trajectory (relax/vc-relax) -> extxyz ──────────────
def qe_trajectory_to_extxyz(qe_output, output_file="trajectory.extxyz"):
    """Extract all ionic steps from QE output as a trajectory."""
    # ASE can read all frames from QE output
    frames = ase_read(qe_output, index=":", format="espresso-out")
    ase_write(output_file, frames, format="extxyz")
    print(f"Trajectory: {len(frames)} frames -> {output_file}")
    return frames

# Example:
# frames = qe_trajectory_to_extxyz("pw_vcrelax.out")
```

---

### Method C: VASP-specific format operations

```python
#!/usr/bin/env python3
"""
VASP-specific format conversions.
Convert VASP output files (CONTCAR, XDATCAR, OUTCAR) to other formats.
"""
import os
from pymatgen.core.structure import Structure
from pymatgen.io.vasp import Poscar, Xdatcar
from pymatgen.io.cif import CifWriter

# ── CONTCAR -> CIF (relaxed structure) ───────────────────────────
def contcar_to_cif(contcar_file="CONTCAR", output_file="relaxed.cif", symprec=0.01):
    """Convert VASP CONTCAR to CIF with symmetry."""
    structure = Structure.from_file(contcar_file)
    writer = CifWriter(structure, symprec=symprec)
    writer.write_file(output_file)
    print(f"CONTCAR -> CIF: {output_file}")
    print(f"  Formula: {structure.composition.reduced_formula}")
    print(f"  Atoms: {len(structure)}")
    return structure

# Example:
# contcar_to_cif("CONTCAR", "relaxed.cif")


# ── XDATCAR -> trajectory (MD or relaxation) ─────────────────────
def xdatcar_to_trajectory(xdatcar_file="XDATCAR", output_file="trajectory.extxyz",
                          every_n=1):
    """Convert VASP XDATCAR (MD trajectory) to extxyz or individual frames."""
    xdatcar = Xdatcar(xdatcar_file)
    structures = xdatcar.structures

    print(f"XDATCAR: {len(structures)} frames")

    # Write individual frames
    from ase.io import write as ase_write
    from pymatgen.io.ase import AseAtomsAdaptor

    frames = []
    for i, struct in enumerate(structures):
        if i % every_n != 0:
            continue
        atoms = AseAtomsAdaptor.get_atoms(struct)
        frames.append(atoms)

    ase_write(output_file, frames, format="extxyz")
    print(f"Trajectory: {len(frames)} frames -> {output_file}")

    # Also save first and last frames as POSCAR
    structures[0].to("POSCAR_first", fmt="poscar")
    structures[-1].to("POSCAR_last", fmt="poscar")
    print(f"First frame: POSCAR_first")
    print(f"Last frame:  POSCAR_last")

    return structures

# Example:
# xdatcar_to_trajectory("XDATCAR", "md_trajectory.extxyz", every_n=10)


# ── POSCAR -> All formats (one-shot conversion) ──────────────────
def poscar_to_all(poscar_file="POSCAR", output_dir="converted"):
    """Convert a POSCAR to all common formats at once."""
    os.makedirs(output_dir, exist_ok=True)
    structure = Structure.from_file(poscar_file)
    formula = structure.composition.reduced_formula

    outputs = {
        "cif": os.path.join(output_dir, f"{formula}.cif"),
        "xyz": os.path.join(output_dir, f"{formula}.xyz"),
        "pdb": os.path.join(output_dir, f"{formula}.pdb"),
        "json": os.path.join(output_dir, f"{formula}.json"),
        "xsf": os.path.join(output_dir, f"{formula}.xsf"),
    }

    for fmt, path in outputs.items():
        try:
            if fmt == "cif":
                CifWriter(structure, symprec=0.01).write_file(path)
            else:
                structure.to(path, fmt=fmt)
            print(f"  {fmt:6s}: {path}")
        except Exception as e:
            print(f"  {fmt:6s}: FAILED ({e})")

    return structure

# Example:
# poscar_to_all("POSCAR", "all_formats")
```

---

## Key Parameters

| Parameter | Default | Notes |
|---|---|---|
| `symprec` | 0.01 A | Symmetry tolerance for CIF writing. `None` for P1 (no symmetry). |
| `direct` (Poscar) | `True` | `True` = fractional coordinates, `False` = Cartesian. |
| `atom_style` (LAMMPS) | `"atomic"` | Use `"charge"` for ionic systems, `"full"` for molecular systems. |
| `occupancy_tolerance` (CIF) | 1.0 | Tolerance for partial occupancies in CIF. Increase for disordered structures. |
| Input format (ASE) | Auto-detected | Specify explicitly if auto-detection fails (e.g., `format="vasp"`). |

## Interpreting Results

- **CIF with symmetry**: Contains space group, Wyckoff positions, and symmetry operations. Fewer atom entries than the full structure (symmetry-equivalent atoms are generated by the space group operations).
- **CIF without symmetry (P1)**: All atoms listed explicitly. Larger file but unambiguous.
- **POSCAR direct vs Cartesian**: Both represent the same structure. Direct (fractional) coordinates are relative to lattice vectors; Cartesian are in Angstroms.
- **Extended XYZ**: Contains lattice vectors, atomic positions, and optional per-atom properties (forces, energy, charge). Native ASE format, good for ML training data.
- **LAMMPS data**: Atom types are numbered (1, 2, ...), not element symbols. Maintain a type-to-element mapping.

## Common Issues

| Problem | Solution |
|---|---|
| CIF has partial occupancies | Use `CifParser(file, occupancy_tolerance=1.0)`. For disordered structures, generate an ordered supercell. |
| POSCAR species order wrong | pymatgen sorts by electronegativity by default. Use `get_sorted_structure()` or specify order manually. |
| LAMMPS atom types are numbers | Maintain element mapping. Use `LammpsData.from_structure()` which preserves type info. |
| XYZ loses periodicity info | Use extxyz format instead, which stores the unit cell. |
| QE output read fails | Use ASE's `read_espresso_out()` which handles most QE output variations. |
| PDB atom names truncated | PDB format has a 4-character limit for atom names. This is a format limitation. |
| Large CIF file won't load | Try `CifParser(file, primitive=True)` to get the smallest unit cell. |
| Coordinate precision lost | Use JSON format for lossless round-tripping. CIF and POSCAR have limited decimal places. |
| Different number of atoms after conversion | Check if primitive vs conventional cell was used. Some formats trigger automatic symmetry expansion. |
