# MD Trajectory Manipulation Tools

## When to Use

- Extracting a subset of atoms from a trajectory (e.g., only Li atoms from an LLZO MD run)
- Converting between trajectory formats (XDATCAR, extxyz, PDB, LAMMPS dump, ASE .traj)
- Unwrapping PBC-wrapped trajectories for proper MSD / displacement analysis
- Animating trajectories as GIF or frame sequences for visualization
- Subsampling (thinning) long trajectories to reduce file size
- Splitting or merging trajectory files
- Equivalent to VASPKIT functions 736 (extract atoms) and 737 (convert format)

## Method Selection

```
What do you need?

Extract specific atoms from trajectory?
  --> Section A: Filter by element or atom indices

Convert trajectory format?
  --> Section B: Format conversion (any-to-any)

Unwrap PBC-wrapped positions?
  --> Section C: PBC unwrapping

Animate trajectory?
  --> Section D: Generate GIF animation or frame images

Subsample or split trajectory?
  --> Section E: Thin, slice, or merge trajectories
```

## Prerequisites

Pre-installed: `ase`, `pymatgen`, `numpy`, `matplotlib`.

Optional: `nglview` (Jupyter visualization), `imageio` (GIF creation -- install with `pip install imageio`).

## Detailed Steps

### Section A: Extract Subset of Atoms

```python
#!/usr/bin/env python3
"""
Extract trajectories of selected atoms from a full MD trajectory.
Filter by element symbol, atom index, or spatial region.
"""

import numpy as np
from ase.io import read, write, Trajectory
import os

# ============================================================
# CONFIGURATION
# ============================================================

INPUT_FILE = "md.traj"           # Input trajectory
INPUT_FORMAT = "ase"             # "ase", "lammps-dump-text", "vasp-xdatcar"
OUTPUT_FILE = "extracted.extxyz" # Output trajectory
OUTPUT_FORMAT = "extxyz"         # "extxyz", "vasp-xdatcar", "lammps-dump-text", etc.

# --- Filter mode (choose one) ---
FILTER_MODE = "element"          # "element", "index", or "region"

# For element filter:
ELEMENTS_TO_KEEP = ["Li"]        # Keep only these elements

# For index filter:
ATOM_INDICES = [0, 1, 2, 5, 10]  # Keep atoms at these indices

# For region filter (keep atoms within a box):
REGION_MIN = [0.0, 0.0, 0.0]    # Minimum Cartesian coordinates (A)
REGION_MAX = [5.0, 5.0, 5.0]    # Maximum Cartesian coordinates (A)

FRAME_START = 0                  # First frame to process
FRAME_END = None                 # Last frame (None = all)
FRAME_STEP = 1                   # Process every Nth frame

# ============================================================
# LOAD TRAJECTORY
# ============================================================

def load_traj(input_file, input_format):
    """Load trajectory from any supported format."""
    if input_format == "ase":
        traj = Trajectory(input_file, "r")
        frames = [atoms.copy() for atoms in traj]
        traj.close()
    elif input_format == "vasp-xdatcar":
        from pymatgen.io.vasp import Xdatcar
        from pymatgen.io.ase import AseAtomsAdaptor
        xdatcar = Xdatcar(input_file)
        adaptor = AseAtomsAdaptor()
        frames = [adaptor.get_atoms(s) for s in xdatcar.structures]
    else:
        frames = read(input_file, index=":", format=input_format)
    return frames


print(f"Loading: {INPUT_FILE}")
frames = load_traj(INPUT_FILE, INPUT_FORMAT)
print(f"  Total frames: {len(frames)}")
print(f"  Atoms per frame: {len(frames[0])}")
print(f"  Formula: {frames[0].get_chemical_formula()}")

# Select frame range
frames = frames[FRAME_START:FRAME_END:FRAME_STEP]
print(f"  Selected frames: {len(frames)}")

# ============================================================
# FILTER ATOMS
# ============================================================

def filter_atoms_by_element(atoms, elements):
    """Keep only atoms of specified elements."""
    indices = [i for i, s in enumerate(atoms.get_chemical_symbols()) if s in elements]
    return atoms[indices]


def filter_atoms_by_index(atoms, indices):
    """Keep atoms at specified indices."""
    valid = [i for i in indices if i < len(atoms)]
    return atoms[valid]


def filter_atoms_by_region(atoms, region_min, region_max):
    """Keep atoms within a Cartesian box."""
    positions = atoms.get_positions()
    mask = np.all((positions >= region_min) & (positions <= region_max), axis=1)
    indices = np.where(mask)[0]
    return atoms[indices]


extracted_frames = []
for frame in frames:
    if FILTER_MODE == "element":
        filtered = filter_atoms_by_element(frame, ELEMENTS_TO_KEEP)
    elif FILTER_MODE == "index":
        filtered = filter_atoms_by_index(frame, ATOM_INDICES)
    elif FILTER_MODE == "region":
        filtered = filter_atoms_by_region(frame, REGION_MIN, REGION_MAX)
    else:
        filtered = frame.copy()
    extracted_frames.append(filtered)

n_extracted = len(extracted_frames[0]) if extracted_frames else 0
print(f"\n  Extracted atoms per frame: {n_extracted}")
print(f"  Formula: {extracted_frames[0].get_chemical_formula() if n_extracted > 0 else 'empty'}")

# ============================================================
# WRITE OUTPUT
# ============================================================

if n_extracted > 0:
    write(OUTPUT_FILE, extracted_frames, format=OUTPUT_FORMAT)
    print(f"\nSaved: {OUTPUT_FILE} ({len(extracted_frames)} frames, format: {OUTPUT_FORMAT})")
else:
    print("\nWARNING: No atoms match the filter criteria.")

print("Done.")
```

### Section B: Format Conversion

```python
#!/usr/bin/env python3
"""
Convert between MD trajectory formats.
Supports: ASE .traj, extxyz, VASP XDATCAR, LAMMPS dump, PDB, XYZ, CIF sequence.
"""

import numpy as np
from ase.io import read, write, Trajectory
import os

# ============================================================
# CONFIGURATION
# ============================================================

INPUT_FILE = "md.traj"
INPUT_FORMAT = "ase"            # See format table below

OUTPUT_FILE = "converted.extxyz"
OUTPUT_FORMAT = "extxyz"        # See format table below

# Frame selection
FRAME_START = 0
FRAME_END = None
FRAME_STEP = 1

# Format table:
# ASE format string      | File type
# ----------------------- | ----------------------------
# None (auto-detect)      | Detected from extension
# "ase"                   | ASE binary .traj
# "extxyz"                | Extended XYZ (recommended for portability)
# "vasp-xdatcar"          | VASP XDATCAR
# "lammps-dump-text"      | LAMMPS dump file
# "proteindatabank"       | PDB format
# "xyz"                   | Simple XYZ (no cell info)
# "cif"                   | CIF (single frame only)
# "vasp"                  | VASP POSCAR/CONTCAR (single frame)

# ============================================================
# LOAD
# ============================================================

print(f"Converting: {INPUT_FILE} ({INPUT_FORMAT}) -> {OUTPUT_FILE} ({OUTPUT_FORMAT})")

if INPUT_FORMAT == "ase":
    traj = Trajectory(INPUT_FILE, "r")
    frames = [atoms.copy() for atoms in traj]
    traj.close()
elif INPUT_FORMAT == "vasp-xdatcar":
    from pymatgen.io.vasp import Xdatcar
    from pymatgen.io.ase import AseAtomsAdaptor
    xdatcar = Xdatcar(INPUT_FILE)
    adaptor = AseAtomsAdaptor()
    frames = [adaptor.get_atoms(s) for s in xdatcar.structures]
else:
    frames = read(INPUT_FILE, index=":", format=INPUT_FORMAT)

# Select frames
frames = frames[FRAME_START:FRAME_END:FRAME_STEP]
print(f"  Frames: {len(frames)}")
print(f"  Atoms: {len(frames[0])}")

# ============================================================
# WRITE
# ============================================================

if OUTPUT_FORMAT == "ase":
    traj_out = Trajectory(OUTPUT_FILE, "w")
    for f in frames:
        traj_out.write(f)
    traj_out.close()
elif OUTPUT_FORMAT == "vasp-xdatcar":
    # Write XDATCAR using pymatgen
    from pymatgen.io.ase import AseAtomsAdaptor
    from pymatgen.io.vasp import Xdatcar
    adaptor = AseAtomsAdaptor()
    structures = [adaptor.get_structure(f) for f in frames]
    # Manual XDATCAR write
    with open(OUTPUT_FILE, "w") as fout:
        # Header from first structure
        s0 = structures[0]
        fout.write(f"{s0.composition.reduced_formula}\n")
        fout.write("1.0\n")
        for vec in s0.lattice.matrix:
            fout.write(f"  {vec[0]:16.10f}  {vec[1]:16.10f}  {vec[2]:16.10f}\n")
        # Element names and counts
        elements = []
        counts = []
        for el, count in s0.composition.element_composition.items():
            elements.append(str(el))
            counts.append(int(count))
        fout.write("  " + "  ".join(elements) + "\n")
        fout.write("  " + "  ".join(str(c) for c in counts) + "\n")
        # Frames
        for i, s in enumerate(structures):
            fout.write(f"Direct configuration=     {i + 1}\n")
            for site in s:
                fc = site.frac_coords
                fout.write(f"  {fc[0]:16.10f}  {fc[1]:16.10f}  {fc[2]:16.10f}\n")
else:
    write(OUTPUT_FILE, frames, format=OUTPUT_FORMAT)

file_size = os.path.getsize(OUTPUT_FILE) / (1024 * 1024)
print(f"\nSaved: {OUTPUT_FILE} ({file_size:.2f} MB)")
print("Done.")
```

### Section C: PBC Unwrapping

```python
#!/usr/bin/env python3
"""
Unwrap PBC-wrapped trajectories.
Produces a trajectory where atoms move continuously (no jumps at box boundaries).
Essential for correct MSD and displacement calculations.
"""

import numpy as np
from ase.io import read, write, Trajectory
import os

# ============================================================
# CONFIGURATION
# ============================================================

INPUT_FILE = "md.traj"
INPUT_FORMAT = "ase"             # "ase", "lammps-dump-text", "vasp-xdatcar"
OUTPUT_FILE = "unwrapped.extxyz"
OUTPUT_FORMAT = "extxyz"

# ============================================================
# LOAD TRAJECTORY
# ============================================================

if INPUT_FORMAT == "ase":
    traj = Trajectory(INPUT_FILE, "r")
    frames = [atoms.copy() for atoms in traj]
    traj.close()
elif INPUT_FORMAT == "vasp-xdatcar":
    from pymatgen.io.vasp import Xdatcar
    from pymatgen.io.ase import AseAtomsAdaptor
    xdatcar = Xdatcar(INPUT_FILE)
    adaptor = AseAtomsAdaptor()
    frames = [adaptor.get_atoms(s) for s in xdatcar.structures]
else:
    frames = read(INPUT_FILE, index=":", format=INPUT_FORMAT)

n_frames = len(frames)
n_atoms = len(frames[0])
print(f"Loaded {n_frames} frames, {n_atoms} atoms")

# ============================================================
# UNWRAP POSITIONS
# ============================================================

def unwrap_trajectory(frames):
    """
    Unwrap PBC-wrapped positions using minimum image convention.
    Returns list of ASE Atoms with unwrapped positions.
    The cell is preserved but positions may extend outside [0, 1] in fractional coords.
    """
    unwrapped_frames = []

    # First frame: keep as-is
    first = frames[0].copy()
    unwrapped_frames.append(first)
    prev_pos = first.get_positions().copy()
    cumulative_pos = prev_pos.copy()

    for i in range(1, len(frames)):
        curr = frames[i]
        curr_pos = curr.get_positions()
        cell = curr.get_cell()

        # Displacement in Cartesian
        disp = curr_pos - prev_pos

        # Minimum image convention
        cell_inv = np.linalg.inv(cell.T)
        scaled_disp = disp @ cell_inv.T
        scaled_disp -= np.round(scaled_disp)
        real_disp = scaled_disp @ cell

        # Accumulate unwrapped positions
        cumulative_pos = cumulative_pos + real_disp

        # Create new Atoms with unwrapped positions
        new_atoms = curr.copy()
        new_atoms.set_positions(cumulative_pos)
        # Optionally keep velocities if available
        try:
            v = curr.get_velocities()
            if v is not None:
                new_atoms.set_velocities(v)
        except Exception:
            pass

        unwrapped_frames.append(new_atoms)
        prev_pos = curr_pos.copy()

    return unwrapped_frames


print("Unwrapping PBC...")
unwrapped = unwrap_trajectory(frames)

# Verify: compute displacement of first atom
pos_first = unwrapped[0].get_positions()[0]
pos_last = unwrapped[-1].get_positions()[0]
total_disp = np.linalg.norm(pos_last - pos_first)
print(f"  First atom total displacement: {total_disp:.3f} A")
print(f"  (If this is very large, atom is diffusing; if small, it is oscillating in place)")

# ============================================================
# SAVE
# ============================================================

write(OUTPUT_FILE, unwrapped, format=OUTPUT_FORMAT)
file_size = os.path.getsize(OUTPUT_FILE) / (1024 * 1024)
print(f"\nSaved: {OUTPUT_FILE} ({file_size:.2f} MB, {len(unwrapped)} frames)")

# Also compute and save per-atom total displacement
displacements = np.linalg.norm(
    unwrapped[-1].get_positions() - unwrapped[0].get_positions(), axis=1
)
symbols = unwrapped[0].get_chemical_symbols()

print(f"\nDisplacement statistics:")
for el in sorted(set(symbols)):
    indices = [i for i, s in enumerate(symbols) if s == el]
    d_el = displacements[indices]
    print(f"  {el}: mean = {np.mean(d_el):.3f} A, max = {np.max(d_el):.3f} A, "
          f"min = {np.min(d_el):.3f} A")

print("Done.")
```

### Section D: Trajectory Animation

```python
#!/usr/bin/env python3
"""
Animate MD trajectory as a GIF or series of PNG frames.
Uses matplotlib for rendering (no OpenGL required).
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from ase.io import read, Trajectory
import os

# ============================================================
# CONFIGURATION
# ============================================================

INPUT_FILE = "md.traj"
INPUT_FORMAT = "ase"

OUTPUT_GIF = "trajectory.gif"
OUTPUT_FRAMES_DIR = "frames"     # Directory for individual PNG frames

FRAME_START = 0
FRAME_END = None
FRAME_STEP = 5                   # Use every 5th frame for animation

# Visualization settings
VIEW_ELEV = 20                   # Elevation angle (degrees)
VIEW_AZIM = 45                   # Azimuth angle (degrees)
ATOM_SIZE = 50                   # Marker size
FIG_SIZE = (8, 6)
DPI = 100

# Element colors (CPK-like)
ELEMENT_COLORS = {
    "H": "#FFFFFF", "He": "#D9FFFF",
    "Li": "#CC80FF", "Be": "#C2FF00", "B": "#FFB5B5", "C": "#909090",
    "N": "#3050F8", "O": "#FF0D0D", "F": "#90E050", "Ne": "#B3E3F5",
    "Na": "#AB5CF2", "Mg": "#8AFF00", "Al": "#BFA6A6", "Si": "#F0C8A0",
    "P": "#FF8000", "S": "#FFFF30", "Cl": "#1FF01F", "Ar": "#80D1E3",
    "K": "#8F40D4", "Ca": "#3DFF00", "Ti": "#BFC2C7", "V": "#A6A6AB",
    "Cr": "#8A99C7", "Mn": "#9C7AC7", "Fe": "#E06633", "Co": "#F090A0",
    "Ni": "#50D050", "Cu": "#C88033", "Zn": "#7D80B0", "Ga": "#C28F8F",
    "Ge": "#668F8F", "As": "#BD80E3", "Se": "#FFA100", "Br": "#A62929",
    "Sr": "#00FF00", "Y": "#94FFFF", "Zr": "#94E0E0", "Nb": "#73C2C9",
    "Mo": "#54B5B5", "Ag": "#C0C0C0", "Sn": "#668080", "Ba": "#00C900",
    "La": "#70D4FF", "Ce": "#FFFFC7", "W": "#2194D6", "Pt": "#D0D0E0",
    "Au": "#FFD123", "Pb": "#575961", "Bi": "#9E4FB5",
}
DEFAULT_COLOR = "#808080"

# ============================================================
# LOAD TRAJECTORY
# ============================================================

if INPUT_FORMAT == "ase":
    traj = Trajectory(INPUT_FILE, "r")
    frames = [atoms.copy() for atoms in traj]
    traj.close()
elif INPUT_FORMAT == "vasp-xdatcar":
    from pymatgen.io.vasp import Xdatcar
    from pymatgen.io.ase import AseAtomsAdaptor
    xdatcar = Xdatcar(INPUT_FILE)
    adaptor = AseAtomsAdaptor()
    frames = [adaptor.get_atoms(s) for s in xdatcar.structures]
else:
    frames = read(INPUT_FILE, index=":", format=INPUT_FORMAT)

frames = frames[FRAME_START:FRAME_END:FRAME_STEP]
n_frames = len(frames)
print(f"Loaded {n_frames} frames for animation")

# ============================================================
# RENDER FRAMES
# ============================================================

os.makedirs(OUTPUT_FRAMES_DIR, exist_ok=True)

# Determine global position bounds for consistent axes
all_pos = np.concatenate([f.get_positions() for f in frames])
pos_min = all_pos.min(axis=0) - 1
pos_max = all_pos.max(axis=0) + 1

frame_files = []

for i, atoms in enumerate(frames):
    fig = plt.figure(figsize=FIG_SIZE, dpi=DPI)
    ax = fig.add_subplot(111, projection="3d")

    positions = atoms.get_positions()
    syms = atoms.get_chemical_symbols()

    # Color by element
    for el in sorted(set(syms)):
        idx = [j for j, s in enumerate(syms) if s == el]
        pos_el = positions[idx]
        color = ELEMENT_COLORS.get(el, DEFAULT_COLOR)
        ax.scatter(pos_el[:, 0], pos_el[:, 1], pos_el[:, 2],
                   c=color, s=ATOM_SIZE, label=el, alpha=0.8,
                   edgecolors="k", linewidths=0.3)

    # Draw cell edges
    cell = atoms.get_cell()
    origin = np.zeros(3)
    for j in range(3):
        for start in [origin, cell[j], cell[(j+1)%3], cell[j] + cell[(j+1)%3]]:
            end = start + cell[(j+2)%3] if not np.allclose(start, origin + cell[j] + cell[(j+1)%3]) else start
        # Simpler: draw 12 edges of the parallelepiped
    corners = np.array([
        [0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1],
        [1, 1, 0], [1, 0, 1], [0, 1, 1], [1, 1, 1]
    ]) @ cell
    edges = [
        (0, 1), (0, 2), (0, 3), (1, 4), (1, 5), (2, 4),
        (2, 6), (3, 5), (3, 6), (4, 7), (5, 7), (6, 7)
    ]
    for e0, e1 in edges:
        ax.plot3D(*zip(corners[e0], corners[e1]), color="gray", linewidth=0.5, alpha=0.5)

    ax.set_xlim(pos_min[0], pos_max[0])
    ax.set_ylim(pos_min[1], pos_max[1])
    ax.set_zlim(pos_min[2], pos_max[2])
    ax.view_init(elev=VIEW_ELEV, azim=VIEW_AZIM)
    ax.set_xlabel("x ($\\AA$)")
    ax.set_ylabel("y ($\\AA$)")
    ax.set_zlabel("z ($\\AA$)")
    ax.set_title(f"Frame {i * FRAME_STEP + FRAME_START}")

    if i == 0:
        ax.legend(fontsize=8, loc="upper left", markerscale=0.8)

    frame_path = os.path.join(OUTPUT_FRAMES_DIR, f"frame_{i:05d}.png")
    fig.savefig(frame_path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    frame_files.append(frame_path)

    if (i + 1) % 20 == 0 or i == n_frames - 1:
        print(f"  Rendered frame {i + 1}/{n_frames}")

# ============================================================
# CREATE GIF
# ============================================================

try:
    import imageio.v2 as imageio

    images = [imageio.imread(f) for f in frame_files]
    imageio.mimsave(OUTPUT_GIF, images, duration=0.1, loop=0)
    gif_size = os.path.getsize(OUTPUT_GIF) / (1024 * 1024)
    print(f"\nGIF saved: {OUTPUT_GIF} ({gif_size:.2f} MB, {n_frames} frames)")

except ImportError:
    print("\nimageio not installed. Install with: pip install imageio")
    print(f"Individual frames saved in: {OUTPUT_FRAMES_DIR}/")
    print("To create GIF manually:")
    print(f"  pip install imageio && python -c \"")
    print(f"  import imageio; import glob")
    print(f"  files = sorted(glob.glob('{OUTPUT_FRAMES_DIR}/frame_*.png'))")
    print(f"  images = [imageio.imread(f) for f in files]")
    print(f"  imageio.mimsave('{OUTPUT_GIF}', images, duration=0.1)\"")

print("Done.")
```

### Section E: Subsample, Slice, and Merge Trajectories

```python
#!/usr/bin/env python3
"""
Subsample (thin), slice, or merge MD trajectories.
Useful for reducing file size or combining multiple runs.
"""

import numpy as np
from ase.io import read, write, Trajectory
import os

# ============================================================
# MODE: "subsample", "slice", or "merge"
# ============================================================

MODE = "subsample"

# --- Subsample: keep every Nth frame ---
if MODE == "subsample":
    INPUT_FILE = "md.traj"
    INPUT_FORMAT = "ase"
    OUTPUT_FILE = "thinned.extxyz"
    OUTPUT_FORMAT = "extxyz"
    KEEP_EVERY = 10              # Keep every 10th frame

    if INPUT_FORMAT == "ase":
        traj = Trajectory(INPUT_FILE, "r")
        frames = [atoms.copy() for atoms in traj]
        traj.close()
    else:
        frames = read(INPUT_FILE, index=":", format=INPUT_FORMAT)

    thinned = frames[::KEEP_EVERY]
    write(OUTPUT_FILE, thinned, format=OUTPUT_FORMAT)
    print(f"Subsampled: {len(frames)} -> {len(thinned)} frames")
    print(f"Saved: {OUTPUT_FILE}")

# --- Slice: extract a specific frame range ---
elif MODE == "slice":
    INPUT_FILE = "md.traj"
    INPUT_FORMAT = "ase"
    OUTPUT_FILE = "sliced.extxyz"
    OUTPUT_FORMAT = "extxyz"
    START_FRAME = 100
    END_FRAME = 500

    if INPUT_FORMAT == "ase":
        traj = Trajectory(INPUT_FILE, "r")
        frames = [atoms.copy() for atoms in traj]
        traj.close()
    else:
        frames = read(INPUT_FILE, index=":", format=INPUT_FORMAT)

    sliced = frames[START_FRAME:END_FRAME]
    write(OUTPUT_FILE, sliced, format=OUTPUT_FORMAT)
    print(f"Sliced: frames {START_FRAME}-{END_FRAME} ({len(sliced)} frames)")
    print(f"Saved: {OUTPUT_FILE}")

# --- Merge: concatenate multiple trajectory files ---
elif MODE == "merge":
    INPUT_FILES = [
        ("run1.traj", "ase"),
        ("run2.traj", "ase"),
        ("run3.extxyz", "extxyz"),
    ]
    OUTPUT_FILE = "merged.extxyz"
    OUTPUT_FORMAT = "extxyz"

    all_frames = []
    for fname, fmt in INPUT_FILES:
        if fmt == "ase":
            traj = Trajectory(fname, "r")
            frames = [atoms.copy() for atoms in traj]
            traj.close()
        elif fmt == "vasp-xdatcar":
            from pymatgen.io.vasp import Xdatcar
            from pymatgen.io.ase import AseAtomsAdaptor
            xdatcar = Xdatcar(fname)
            adaptor = AseAtomsAdaptor()
            frames = [adaptor.get_atoms(s) for s in xdatcar.structures]
        else:
            frames = read(fname, index=":", format=fmt)

        print(f"  {fname}: {len(frames)} frames")
        all_frames.extend(frames)

    write(OUTPUT_FILE, all_frames, format=OUTPUT_FORMAT)
    print(f"\nMerged: {len(all_frames)} total frames")
    print(f"Saved: {OUTPUT_FILE}")

print("Done.")
```

## Key Parameters

| Parameter | Default | Notes |
|---|---|---|
| `INPUT_FORMAT` | "ase" | Source format. Supported: "ase", "extxyz", "lammps-dump-text", "vasp-xdatcar", "proteindatabank", "xyz". |
| `OUTPUT_FORMAT` | "extxyz" | Target format. "extxyz" recommended for portability (stores cell, species, positions, and custom arrays). |
| `FRAME_STEP` | 1-10 | Subsampling factor. Higher = smaller output. Balance storage vs time resolution. |
| `ELEMENTS_TO_KEEP` | [] | Elements to extract. Empty = keep all. |
| `ATOM_INDICES` | [] | Specific atom indices to extract. |
| `VIEW_ELEV/AZIM` | 20/45 | 3D view angles for animation. Adjust for best visibility of your structure. |
| `ATOM_SIZE` | 50 | Marker size for 3D scatter plot. Scale with number of atoms. |

### Format comparison

| Format | Cell info | Velocities | Multi-frame | File size | Portability |
|---|---|---|---|---|---|
| ASE .traj | Yes | Yes | Yes | Binary, compact | ASE only |
| extxyz | Yes | Optional | Yes | Text, moderate | Universal |
| XDATCAR | Yes | No | Yes | Text, moderate | VASP ecosystem |
| LAMMPS dump | Yes | Optional | Yes | Text, large | LAMMPS ecosystem |
| PDB | Partial | No | Yes | Text, moderate | Bio tools |
| XYZ | No | No | Yes | Text, small | Universal |
| CIF | Yes | No | Single | Text, small | Crystallography |

## Interpreting Results

### Atom extraction
- Extracted trajectories lose the full cell context. The cell is preserved but contains only the selected atoms.
- Useful for visualizing mobile species (e.g., Li in solid electrolytes) without framework atoms cluttering the view.
- Per-element extraction enables separate MSD analysis for each species.

### Format conversion
- extxyz is the recommended interchange format: human-readable, stores all information, and is supported by most tools.
- XDATCAR to extxyz conversion is common after VASP AIMD runs.
- LAMMPS dump conversion may require element-to-type mapping if the dump only has numeric type IDs.

### PBC unwrapping
- Wrapped trajectories show atoms jumping across cell boundaries. This breaks MSD calculations.
- Unwrapped trajectories show continuous motion. Positions may extend far outside the original cell for diffusing atoms.
- Always unwrap before computing MSD or tracking diffusion paths.
- The unwrapping algorithm assumes atoms move less than half a cell length per frame. If this is violated (timestep too large, frame interval too large), unwrapping will be incorrect.

### Animation
- GIF animations provide quick visual assessment of structural evolution.
- Look for: melting/disordering, phase transitions, preferential diffusion paths, surface reconstruction.
- For publication-quality visualization, export extxyz and use OVITO, VESTA, or VMD.

## Common Issues

| Issue | Cause | Fix |
|---|---|---|
| ASE cannot read LAMMPS dump | Missing atom type mapping | Specify `species` in ASE read, or use `ase.io.lammps` with type mapping |
| XDATCAR has inconsistent frames | VASP restarted mid-run | Check frame count; split at restart boundaries |
| Unwrapping produces jumps | Frame interval too large (atom moved > L/2) | Reduce frame dump interval in MD; or interpolate missing frames |
| extxyz file very large | Too many frames or atoms | Subsample with FRAME_STEP; use binary .traj for storage |
| GIF animation too large | Too many frames or high DPI | Increase FRAME_STEP; reduce DPI; reduce FIG_SIZE |
| Element colors wrong | Element not in color dictionary | Add to ELEMENT_COLORS dict |
| Memory error loading trajectory | Trajectory too large | Process in chunks: load N frames at a time |
| Velocities lost in conversion | Target format does not support velocities | Use extxyz (supports velocities) or ASE .traj |
