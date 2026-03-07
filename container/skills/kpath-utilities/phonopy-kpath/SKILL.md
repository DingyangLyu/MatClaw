# Phonopy K-Path Generation

## When to Use

- You are running a phonopy phonon band structure calculation and need the k-path settings.
- You want to generate `band.conf` or `KPATH.in` for phonopy.
- You need seekpath-consistent k-paths for phonon dispersion plots.
- You want to compare phonon band structures computed with different codes (QE matdyn.x, phonopy, VASP DFPT) using the same k-path.

## Method Selection

| Criterion | phonopy auto_band_structure | seekpath + manual band.conf | pymatgen HighSymmKpath |
|---|---|---|---|
| Ease of use | Simplest (one function call) | Moderate | Moderate |
| Convention | seekpath (Hinuma) internally | seekpath (Hinuma) | Setyawan-Curtarolo or Hinuma |
| Customization | Limited | Full control | Full control |
| Best for | Quick phonon bands | Custom paths, QE matdyn.x | Integration with pymatgen workflows |

## Prerequisites

- `pip install phonopy seekpath` (phonopy uses seekpath internally for auto_band_structure)
- pymatgen, spglib, numpy, matplotlib (pre-installed)

---

## Detailed Steps

### Method A: phonopy auto_band_structure (Simplest)

```python
#!/usr/bin/env python3
"""
Use phonopy's built-in auto_band_structure with seekpath integration.
This is the simplest approach -- phonopy handles everything internally.
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from pymatgen.core import Structure
from pymatgen.io.ase import AseAtomsAdaptor
from pymatgen.io.phonopy import get_phonopy_structure

import phonopy

# ============================================================
# 1. Load structure and set up phonopy
# ============================================================
# structure = Structure.from_file("POSCAR")
structure = Structure.from_spacegroup(
    "Fd-3m",
    lattice=[[5.431, 0, 0], [0, 5.431, 0], [0, 0, 5.431]],
    species=["Si"],
    coords=[[0.0, 0.0, 0.0]],
)

phonopy_struct = get_phonopy_structure(structure)
supercell_matrix = np.diag([4, 4, 4])

phonon = phonopy.Phonopy(
    phonopy_struct,
    supercell_matrix=supercell_matrix.tolist(),
    symprec=1e-5,
)

# ============================================================
# 2. (Assume force constants are already computed)
# ============================================================
# Option A: Load from a previous calculation
# phonon.load("phonopy_params.yaml")

# Option B: Generate forces with MACE and build force constants
from mace.calculators import mace_mp
from ase.io import read
import warnings
warnings.filterwarnings("ignore")

calc = mace_mp(model="medium", device="cpu", default_dtype="float64")
adaptor = AseAtomsAdaptor()

phonon.generate_displacements(distance=0.01)
supercells = phonon.supercells_with_displacements

forces_list = []
for i, sc in enumerate(supercells):
    sc_struct = Structure(
        lattice=sc.cell,
        species=sc.symbols,
        coords=sc.scaled_positions,
    )
    sc_atoms = adaptor.get_atoms(sc_struct)
    sc_atoms.calc = calc
    forces = sc_atoms.get_forces()
    forces_list.append(forces)
    if (i + 1) % 5 == 0 or (i + 1) == len(supercells):
        print(f"  Forces: {i+1}/{len(supercells)}")

phonon.forces = forces_list
phonon.produce_force_constants()

# ============================================================
# 3. Auto band structure (seekpath-based, one call)
# ============================================================
phonon.auto_band_structure(
    npoints=101,             # points per segment
    with_eigenvectors=False,
    write_yaml=True,         # writes band.yaml
)

# Plot
fig, ax = plt.subplots(figsize=(8, 5))
phonon.plot_band_structure(ax=ax)
ax.set_ylabel("Frequency (THz)")
ax.set_title(f"Phonon Band Structure - {structure.composition.reduced_formula}")
ax.axhline(y=0, color="gray", linestyle="--", linewidth=0.5)
fig.tight_layout()
fig.savefig("phonon_bands_auto.png", dpi=150)
print("Saved: phonon_bands_auto.png")
print("Saved: band.yaml (phonopy auto_band_structure output)")

# ============================================================
# 4. Extract the k-path that phonopy/seekpath used
# ============================================================
band_dict = phonon.get_band_structure_dict()
distances = band_dict["distances"]       # list of arrays (one per segment)
frequencies = band_dict["frequencies"]   # list of arrays
labels = band_dict.get("labels", None)   # high-symmetry point labels

if labels:
    print("\n=== K-path used by phonopy ===")
    # labels is a list of (label_start, label_end) for each segment
    for i, (seg_labels) in enumerate(labels):
        print(f"  Segment {i}: {seg_labels}")
```

### Method B: Generate band.conf for phonopy CLI

```python
#!/usr/bin/env python3
"""
Generate band.conf and KPATH.in files for phonopy command-line usage.
Uses seekpath to determine the standardized k-path.
"""

import numpy as np
import seekpath
from pymatgen.core import Structure
from pymatgen.io.phonopy import get_phonopy_structure

# ============================================================
# 1. Load structure
# ============================================================
# structure = Structure.from_file("POSCAR")
structure = Structure.from_spacegroup(
    "Fd-3m",
    lattice=[[5.431, 0, 0], [0, 5.431, 0], [0, 0, 5.431]],
    species=["Si"],
    coords=[[0.0, 0.0, 0.0]],
)

# ============================================================
# 2. Get k-path from seekpath
# ============================================================
cell = structure.lattice.matrix
positions = structure.frac_coords
numbers = [site.specie.Z for site in structure]

path_data = seekpath.get_path((cell, positions, numbers), symprec=1e-5)

kpoint_coords = path_data["point_coords"]
path_segments = path_data["path"]
bravais = path_data["bravais_lattice"]

print(f"Structure: {structure.composition.reduced_formula}")
print(f"Bravais lattice: {bravais}")
print(f"Path segments: {len(path_segments)}")

# ============================================================
# 3. Generate band.conf
# ============================================================
def generate_band_conf(path_data, npoints=101, filename="band.conf"):
    """
    Generate phonopy band.conf file from seekpath path data.

    phonopy BAND format:
      BAND = k1x k1y k1z  k2x k2y k2z  k3x k3y k3z, k4x k4y k4z  ...
    Comma separates disconnected segments. Spaces separate points within a
    connected segment.
    """
    segments = path_data["path"]
    coords = path_data["point_coords"]

    # Group connected segments
    connected_groups = []
    current_group = [segments[0][0]]

    for i, (start, end) in enumerate(segments):
        if i > 0 and start != segments[i-1][1]:
            # Disconnect: save current group, start new
            connected_groups.append(current_group)
            current_group = [start]
        current_group.append(end)

    if current_group:
        connected_groups.append(current_group)

    # Build BAND string
    group_strings = []
    for group in connected_groups:
        point_strings = []
        for label in group:
            c = coords[label]
            point_strings.append(f"{c[0]:.6f} {c[1]:.6f} {c[2]:.6f}")
        group_strings.append("  ".join(point_strings))

    band_line = "BAND = " + ", ".join(group_strings)

    # Build BAND_LABELS
    all_labels = []
    for i, group in enumerate(connected_groups):
        for j, label in enumerate(group):
            display = label.replace("GAMMA", "$\\Gamma$")
            if i > 0 and j == 0:
                # This is a disconnect point -- merge with previous end label
                all_labels[-1] = all_labels[-1] + "|" + display
            else:
                all_labels.append(display)

    label_line = "BAND_LABELS = " + " ".join(all_labels)

    content = f"""# Phonopy band.conf generated from seekpath
# Structure: {structure.composition.reduced_formula}
# Bravais lattice: {path_data['bravais_lattice']}

DIM = 4 4 4
PRIMITIVE_AXES = AUTO

{band_line}
{label_line}
BAND_POINTS = {npoints}
BAND_CONNECTION = .TRUE.

# Optional: force constants file
# FORCE_CONSTANTS = READ
"""

    with open(filename, "w") as f:
        f.write(content)
    print(f"\nband.conf written to {filename}")
    print(f"  BAND = {band_line[:80]}...")
    print(f"  BAND_LABELS = {label_line}")
    return content

band_conf = generate_band_conf(path_data, npoints=101)

# ============================================================
# 4. Generate KPATH.in (explicit k-point list for phonopy)
# ============================================================
def generate_kpath_in(path_data, npoints_per_segment=51, filename="KPATH.in"):
    """
    Generate explicit k-point list file (KPATH.in) for phonopy.
    Format: one k-point per line with fractional coordinates.
    Points are linearly interpolated between high-symmetry points.
    """
    segments = path_data["path"]
    coords = path_data["point_coords"]

    kpoints = []
    labels = []

    for seg_idx, (start, end) in enumerate(segments):
        sc = np.array(coords[start])
        ec = np.array(coords[end])

        for i in range(npoints_per_segment):
            t = i / (npoints_per_segment - 1)
            k = sc + t * (ec - sc)
            kpoints.append(k)

            # Mark start and end points
            if i == 0:
                labels.append(start)
            elif i == npoints_per_segment - 1:
                labels.append(end)
            else:
                labels.append("")

    # Write
    lines = [f"{len(kpoints)}"]
    for k, label in zip(kpoints, labels):
        comment = f"  # {label}" if label else ""
        lines.append(f"  {k[0]:12.8f}  {k[1]:12.8f}  {k[2]:12.8f}{comment}")

    content = "\n".join(lines) + "\n"
    with open(filename, "w") as f:
        f.write(content)
    print(f"KPATH.in written to {filename} ({len(kpoints)} k-points)")
    return content

generate_kpath_in(path_data, npoints_per_segment=51)

# ============================================================
# 5. Generate QE matdyn.x input with the same k-path
# ============================================================
def generate_matdyn_input(path_data, npoints_per_segment=51,
                           filename="matdyn_bands.in"):
    """
    Generate matdyn.x input file for QE phonon band interpolation.
    Uses the same seekpath k-path for consistency with phonopy.
    """
    segments = path_data["path"]
    coords = path_data["point_coords"]

    qpoints_lines = []

    for i, (start, end) in enumerate(segments):
        sc = coords[start]
        ec = coords[end]

        # Add start point
        if i == 0 or start != segments[i-1][1]:
            qpoints_lines.append(
                f"  {sc[0]:10.6f} {sc[1]:10.6f} {sc[2]:10.6f}  {npoints_per_segment}"
            )

        # Add end point
        end_npts = npoints_per_segment if i < len(segments) - 1 else 0
        if i < len(segments) - 1 and segments[i+1][0] != end:
            end_npts = 0  # disconnect
        qpoints_lines.append(
            f"  {ec[0]:10.6f} {ec[1]:10.6f} {ec[2]:10.6f}  {end_npts}"
        )

    content = f"""&INPUT
    asr   = 'crystal'
    flfrc = 'phonon.fc'
    flfrq = 'phonon_bands.freq'
    flvec = 'phonon_bands.modes'
    q_in_band_form = .true.
/
{len(qpoints_lines)}
"""
    content += "\n".join(qpoints_lines) + "\n"

    with open(filename, "w") as f:
        f.write(content)
    print(f"matdyn.x input written to {filename}")
    return content

generate_matdyn_input(path_data, npoints_per_segment=51)

# ============================================================
# 6. Print summary of all k-path outputs
# ============================================================
print("\n" + "=" * 60)
print("GENERATED FILES")
print("=" * 60)
print("  band.conf          -- phonopy band.conf (CLI usage)")
print("  KPATH.in           -- explicit k-point list")
print("  matdyn_bands.in    -- QE matdyn.x input (same k-path)")
```

### Method C: Custom K-Path for phonopy

```python
#!/usr/bin/env python3
"""
Manually specify a custom k-path for phonopy.
Useful when you want a non-standard path or specific points.
"""

import numpy as np
import phonopy
from pymatgen.core import Structure
from pymatgen.io.phonopy import get_phonopy_structure

# ============================================================
# 1. Set up phonopy with pre-computed force constants
# ============================================================
# structure = Structure.from_file("POSCAR")
structure = Structure.from_spacegroup(
    "Fd-3m",
    lattice=[[5.431, 0, 0], [0, 5.431, 0], [0, 0, 5.431]],
    species=["Si"],
    coords=[[0.0, 0.0, 0.0]],
)

phonopy_struct = get_phonopy_structure(structure)
phonon = phonopy.Phonopy(
    phonopy_struct,
    supercell_matrix=[[4, 0, 0], [0, 4, 0], [0, 0, 4]],
    symprec=1e-5,
)

# Load pre-computed force constants
# phonon.load("phonopy_params.yaml")
# Or compute with MACE (see phonon SKILL.md for details)

# ============================================================
# 2. Define custom k-path
# ============================================================
# FCC Si: custom path focusing on specific regions
custom_bands = [
    # Each sub-list is a connected path segment
    # Format: [[k1], [k2], [k3], ...] in fractional reciprocal coords
    [[0.0, 0.0, 0.0],    # GAMMA
     [0.5, 0.0, 0.5],    # X
     [0.5, 0.25, 0.75],  # W
     [0.375, 0.375, 0.75]],  # K

    [[0.0, 0.0, 0.0],    # GAMMA (disconnected)
     [0.5, 0.5, 0.5]],   # L
]

custom_labels = [
    [r"$\Gamma$", "X", "W", "K"],
    [r"$\Gamma$", "L"],
]

# ============================================================
# 3. Run phonopy with custom path
# ============================================================
# phonon.run_band_structure(
#     custom_bands,
#     labels=custom_labels,
#     with_eigenvectors=False,
#     is_band_connection=True,
# )

# ============================================================
# 4. Write custom band.conf
# ============================================================
def write_custom_band_conf(bands, labels, npoints=101, dim="4 4 4",
                            filename="band_custom.conf"):
    """Write a band.conf with a custom k-path."""
    band_strings = []
    for segment in bands:
        point_strings = [f"{k[0]:.6f} {k[1]:.6f} {k[2]:.6f}" for k in segment]
        band_strings.append("  ".join(point_strings))

    band_line = "BAND = " + ", ".join(band_strings)

    flat_labels = []
    for seg_labels in labels:
        flat_labels.extend(seg_labels)

    label_line = "BAND_LABELS = " + " ".join(flat_labels)

    content = f"""# Custom phonopy band.conf
DIM = {dim}
PRIMITIVE_AXES = AUTO

{band_line}
{label_line}
BAND_POINTS = {npoints}
BAND_CONNECTION = .TRUE.
"""
    with open(filename, "w") as f:
        f.write(content)
    print(f"Custom band.conf written to {filename}")
    print(content)

write_custom_band_conf(custom_bands, custom_labels)

# ============================================================
# 5. Generate mesh.conf for phonon DOS
# ============================================================
mesh_conf = """# Phonopy mesh.conf for phonon DOS
DIM = 4 4 4
PRIMITIVE_AXES = AUTO
MP = 20 20 20
GAMMA_CENTER = .TRUE.
DOS = .TRUE.
"""

with open("mesh.conf", "w") as f:
    f.write(mesh_conf)
print("\nmesh.conf written for phonon DOS")

# ============================================================
# 6. Phonopy CLI usage
# ============================================================
print("\n=== Phonopy CLI commands ===")
print("# After computing force constants (FORCE_CONSTANTS or phonopy_params.yaml):")
print("phonopy -p band.conf          # Plot phonon band structure")
print("phonopy -p mesh.conf          # Plot phonon DOS")
print("phonopy -t mesh.conf          # Thermal properties")
print("phonopy --band-auto POSCAR    # Auto k-path from POSCAR")
print("phonopy -p -s band.conf       # Plot and save to PDF")
```

### Method D: VASP-phonopy Integration (Future External Access)

```python
#!/usr/bin/env python3
"""
Generate phonopy configuration files for VASP-based phonon calculations.
Phonopy uses VASP as a force calculator.
"""

from pymatgen.core import Structure
import seekpath
import numpy as np

# Load structure
structure = Structure.from_spacegroup(
    "Fd-3m",
    lattice=[[5.431, 0, 0], [0, 5.431, 0], [0, 0, 5.431]],
    species=["Si"],
    coords=[[0.0, 0.0, 0.0]],
)

# Get seekpath k-path
cell = structure.lattice.matrix
positions = structure.frac_coords
numbers = [site.specie.Z for site in structure]
path_data = seekpath.get_path((cell, positions, numbers))

# ============================================================
# 1. phonopy command to generate displaced POSCARs
# ============================================================
print("=== Step 1: Generate displacements ===")
print("phonopy -d --dim 4 4 4 -c POSCAR")
print("# Creates SPOSCAR (supercell) and POSCAR-{001,002,...} (displaced)")

# ============================================================
# 2. VASP INCAR for force calculations
# ============================================================
incar_content = """SYSTEM = Phonon force calculation
PREC = Accurate
ENCUT = 520
EDIFF = 1.0E-8
IBRION = -1
NSW = 0
ISMEAR = 0
SIGMA = 0.05
LWAVE = .FALSE.
LCHARG = .FALSE.
LREAL = .FALSE.
ADDGRID = .TRUE.
"""
with open("INCAR_phonon", "w") as f:
    f.write(incar_content)
print("\n=== Step 2: INCAR for VASP force calculations ===")
print("Written: INCAR_phonon")

# ============================================================
# 3. KPOINTS for supercell SCF
# ============================================================
kpoints_content = """Gamma-centered mesh for phonon supercell
0
Gamma
  2 2 2
  0 0 0
"""
with open("KPOINTS_phonon", "w") as f:
    f.write(kpoints_content)
print("Written: KPOINTS_phonon (2x2x2 for 4x4x4 supercell)")

# ============================================================
# 4. band.conf with seekpath k-path
# ============================================================
segments = path_data["path"]
coords = path_data["point_coords"]

connected_groups = []
current_group = [segments[0][0]]
for i, (start, end) in enumerate(segments):
    if i > 0 and start != segments[i-1][1]:
        connected_groups.append(current_group)
        current_group = [start]
    current_group.append(end)
if current_group:
    connected_groups.append(current_group)

group_strings = []
for group in connected_groups:
    pts = [f"{coords[l][0]:.6f} {coords[l][1]:.6f} {coords[l][2]:.6f}" for l in group]
    group_strings.append("  ".join(pts))

all_labels = []
for i, group in enumerate(connected_groups):
    for j, label in enumerate(group):
        display = label.replace("GAMMA", "$\\Gamma$")
        if i > 0 and j == 0:
            all_labels[-1] = all_labels[-1] + "|" + display
        else:
            all_labels.append(display)

band_conf = f"""# band.conf for phonopy with VASP
DIM = 4 4 4
PRIMITIVE_AXES = AUTO
BAND = {', '.join(group_strings)}
BAND_LABELS = {' '.join(all_labels)}
BAND_POINTS = 101
BAND_CONNECTION = .TRUE.
FORCE_CONSTANTS = READ
"""
with open("band_vasp.conf", "w") as f:
    f.write(band_conf)
print("\n=== Step 4: band.conf ===")
print("Written: band_vasp.conf")

# ============================================================
# 5. Workflow summary
# ============================================================
print("\n=== Complete VASP-phonopy workflow ===")
print("1. phonopy -d --dim 4 4 4 -c POSCAR")
print("2. For each POSCAR-{NNN}:")
print("   mkdir disp-{NNN} && cp POSCAR-{NNN} disp-{NNN}/POSCAR")
print("   cp INCAR_phonon disp-{NNN}/INCAR")
print("   cp KPOINTS_phonon disp-{NNN}/KPOINTS")
print("   cp POTCAR disp-{NNN}/POTCAR")
print("   cd disp-{NNN} && vasp_std && cd ..")
print("3. phonopy -f disp-{001..NNN}/vasprun.xml")
print("4. phonopy -p band_vasp.conf")
```

---

## Key Parameters

| Parameter | Typical Value | Notes |
|---|---|---|
| `BAND_POINTS` | 51--101 | Number of q-points per path segment in phonopy |
| `DIM` | `4 4 4` or `3 3 3` | Supercell dimensions; must match force calculation |
| `PRIMITIVE_AXES` | `AUTO` | Let phonopy determine the primitive cell |
| `BAND_CONNECTION` | `.TRUE.` | Connect bands at crossings for smoother plots |
| npoints (seekpath/phonopy) | 101 | Points per segment for smooth dispersion curves |
| `asr` (matdyn.x) | `'crystal'` | Acoustic sum rule for 3D periodics |

## Common Issues

| Problem | Solution |
|---|---|
| phonopy band.conf gives wrong path | Ensure BAND coordinates match the primitive cell reciprocal lattice, not the conventional cell. Use `PRIMITIVE_AXES = AUTO`. |
| Labels are misaligned on phonopy plot | Labels in BAND_LABELS must match the number of distinct points in BAND. Check for comma-separated segments. |
| seekpath and phonopy give different paths | They should agree since phonopy uses seekpath internally. Ensure both use the same structure and symprec. |
| matdyn.x and phonopy paths don't match | Make sure both use the same k-point coordinates (fractional of the SAME reciprocal lattice). |
| phonopy `--band-auto` ignores custom path | `--band-auto` overrides BAND settings. Use `phonopy -p band.conf` instead for custom paths. |
| Discontinuous phonon bands | Normal at path segment breaks. Set `BAND_CONNECTION = .TRUE.` for connected plotting. |
