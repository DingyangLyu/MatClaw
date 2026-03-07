# Bulk K-Path Generation for 3D Crystals

## When to Use

- You need the high-symmetry k-point path through the Brillouin zone for a 3D crystal.
- You are preparing a band structure calculation with QE, VASP, or Wannier90.
- You need to visualize the first Brillouin zone with labeled high-symmetry points.
- You want a standardized, convention-consistent k-path (Hinuma et al. or Setyawan-Curtarolo).
- You need to convert a k-path between different code formats (QE crystal_b, VASP KPOINTS, Wannier90 kpath, phonopy band.conf).

## Method Selection

| Criterion | seekpath (Hinuma) | pymatgen HighSymmKpath (Setyawan-Curtarolo) | spglib (manual) |
|---|---|---|---|
| Convention | Hinuma et al. 2017 | Setyawan-Curtarolo 2010 | Raw symmetry data |
| Standardization | Auto-standardizes cell to conventional form | Uses input cell as-is | Returns standardized cell |
| Coverage | All 14 Bravais lattices, all space groups | All crystal systems | No built-in path generation |
| Output | Path segments, point coordinates, primitive cell | Path segments, kpoints dict | Space group, Wyckoff, reciprocal lattice |
| Best for | Reproducible paths, QE integration | Quick path generation, VASP integration | Custom BZ analysis |
| Installation | `pip install seekpath` | Pre-installed | Pre-installed |

**Recommendation**: Use seekpath for reproducible, publication-quality k-paths. Use pymatgen HighSymmKpath for quick path generation when the input structure is already in standard form.

## Prerequisites

- `pip install seekpath` (if not already installed)
- pymatgen, spglib, numpy, matplotlib (pre-installed)
- A crystal structure file (CIF, POSCAR, or built with pymatgen/ASE)

---

## Detailed Steps

### Method A: seekpath (Recommended)

```python
#!/usr/bin/env python3
"""
Generate high-symmetry k-paths for 3D bulk crystals using seekpath.
Outputs k-paths for QE, VASP, Wannier90, and phonopy.
Includes Brillouin zone visualization.

seekpath automatically standardizes the cell and determines the correct
Bravais lattice type, ensuring the k-path follows the Hinuma et al.
convention (Comp. Mat. Sci. 128, 2017).
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

# ============================================================
# 1. Load or build the structure
# ============================================================
from pymatgen.core import Structure
from pymatgen.io.ase import AseAtomsAdaptor

# Option 1: Load from file
# structure = Structure.from_file("POSCAR")
# structure = Structure.from_file("structure.cif")

# Option 2: Build a test structure (Si diamond)
structure = Structure.from_spacegroup(
    "Fd-3m",
    lattice=[[5.431, 0, 0], [0, 5.431, 0], [0, 0, 5.431]],
    species=["Si"],
    coords=[[0.0, 0.0, 0.0]],
)
print(f"Input structure: {structure.composition.reduced_formula}")
print(f"Space group: {structure.get_space_group_info()}")

# ============================================================
# 2. Run seekpath to get the standardized k-path
# ============================================================
import seekpath

# seekpath needs: (cell, scaled_positions, atomic_numbers)
cell = structure.lattice.matrix
positions = structure.frac_coords
numbers = [site.specie.Z for site in structure]

# get_path returns the standardized primitive cell + k-path
path_data = seekpath.get_path(
    (cell, positions, numbers),
    with_time_reversal=True,  # use time-reversal symmetry
    symprec=1e-5,             # symmetry detection tolerance
    angle_tolerance=-1.0,     # auto
    threshold=1e-7,           # numerical threshold
)

# Extract results
kpoint_coords = path_data["point_coords"]    # dict: label -> [kx, ky, kz]
path_segments = path_data["path"]            # list of (start, end) label pairs
prim_lattice = path_data["primitive_lattice"]
prim_positions = path_data["primitive_positions"]
prim_types = path_data["primitive_types"]
bravais_lattice = path_data["bravais_lattice"]
reciprocal_lattice = path_data["reciprocal_primitive_lattice"]

print(f"\nBravais lattice: {bravais_lattice}")
print(f"Primitive cell: {len(prim_types)} atoms")
print(f"Reciprocal lattice vectors:")
for i, vec in enumerate(reciprocal_lattice):
    print(f"  b{i+1} = ({vec[0]:.6f}, {vec[1]:.6f}, {vec[2]:.6f})")

print(f"\n=== High-symmetry points ===")
for label in sorted(kpoint_coords.keys()):
    coords = kpoint_coords[label]
    print(f"  {label:10s}: ({coords[0]:8.4f}, {coords[1]:8.4f}, {coords[2]:8.4f})")

print(f"\n=== Path segments ({len(path_segments)}) ===")
for s, e in path_segments:
    print(f"  {s} --> {e}")

# ============================================================
# 3. Output: QE K_POINTS {crystal_b} card
# ============================================================
def write_qe_kpath(path_data, npoints_per_segment=30, filename="kpath_qe.txt"):
    """
    Generate K_POINTS {crystal_b} card for QE bands calculation.
    """
    segments = path_data["path"]
    coords = path_data["point_coords"]

    klines = []
    labels_order = []

    for i, (start, end) in enumerate(segments):
        sc = coords[start]
        ec = coords[end]

        if i == 0:
            klines.append(f"  {sc[0]:.10f}  {sc[1]:.10f}  {sc[2]:.10f}  {npoints_per_segment}  ! {start}")
            labels_order.append(start)
        elif start != segments[i - 1][1]:
            # Discontinuity in path
            klines.append(f"  {sc[0]:.10f}  {sc[1]:.10f}  {sc[2]:.10f}  {npoints_per_segment}  ! {start}")
            labels_order[-1] = labels_order[-1] + "|" + start
        # else: continuous, start already added as previous end

        end_npts = npoints_per_segment if i < len(segments) - 1 else 0
        klines.append(f"  {ec[0]:.10f}  {ec[1]:.10f}  {ec[2]:.10f}  {end_npts}  ! {end}")
        labels_order.append(end)

    header = f"K_POINTS {{crystal_b}}\n{len(klines)}"
    content = header + "\n" + "\n".join(klines) + "\n"

    with open(filename, "w") as f:
        f.write(content)
    print(f"\nQE K_POINTS card written to {filename}")
    print(content)
    return content, labels_order

qe_kpath, labels_order = write_qe_kpath(path_data, npoints_per_segment=30)

# ============================================================
# 4. Output: VASP KPOINTS file
# ============================================================
def write_vasp_kpoints(path_data, npoints_per_segment=40, filename="KPOINTS"):
    """
    Generate VASP KPOINTS file in line mode.
    """
    segments = path_data["path"]
    coords = path_data["point_coords"]

    lines = ["K-path generated by seekpath (Hinuma et al.)",
             f"{npoints_per_segment}",
             "Line-mode",
             "Reciprocal"]

    for start, end in segments:
        sc = coords[start]
        ec = coords[end]
        label_s = start.replace("GAMMA", "\\Gamma")
        label_e = end.replace("GAMMA", "\\Gamma")
        lines.append(f"  {sc[0]:.10f}  {sc[1]:.10f}  {sc[2]:.10f}  ! {label_s}")
        lines.append(f"  {ec[0]:.10f}  {ec[1]:.10f}  {ec[2]:.10f}  ! {label_e}")
        lines.append("")  # blank line between segments

    content = "\n".join(lines) + "\n"
    with open(filename, "w") as f:
        f.write(content)
    print(f"VASP KPOINTS written to {filename}")
    return content

vasp_kpath = write_vasp_kpoints(path_data, npoints_per_segment=40)

# ============================================================
# 5. Output: Wannier90 kpath block
# ============================================================
def write_wannier90_kpath(path_data, filename="wannier90_kpath.txt"):
    """
    Generate the kpoint_path block for Wannier90 .win file.
    Format:
      begin kpoint_path
      G  0.000 0.000 0.000  X  0.500 0.000 0.500
      ...
      end kpoint_path
    """
    segments = path_data["path"]
    coords = path_data["point_coords"]

    lines = ["begin kpoint_path"]
    for start, end in segments:
        sc = coords[start]
        ec = coords[end]
        lines.append(
            f"  {start:6s} {sc[0]:8.5f} {sc[1]:8.5f} {sc[2]:8.5f}  "
            f"{end:6s} {ec[0]:8.5f} {ec[1]:8.5f} {ec[2]:8.5f}"
        )
    lines.append("end kpoint_path")

    content = "\n".join(lines) + "\n"
    with open(filename, "w") as f:
        f.write(content)
    print(f"Wannier90 kpath written to {filename}")
    return content

w90_kpath = write_wannier90_kpath(path_data)

# ============================================================
# 6. Output: phonopy band.conf format
# ============================================================
def write_phonopy_bandconf(path_data, filename="band.conf"):
    """
    Generate the BAND setting for phonopy band.conf file.
    Format: BAND = 0.0 0.0 0.0  0.5 0.0 0.5, 0.5 0.25 0.75  ...
    Comma separates discontinuous segments.
    """
    segments = path_data["path"]
    coords = path_data["point_coords"]

    band_strings = []
    current_segment_points = []

    for i, (start, end) in enumerate(segments):
        sc = coords[start]
        ec = coords[end]

        if i == 0:
            current_segment_points.append(f"{sc[0]:.5f} {sc[1]:.5f} {sc[2]:.5f}")
        elif start != segments[i - 1][1]:
            # Discontinuity: flush current segment, start new one
            band_strings.append("  ".join(current_segment_points))
            current_segment_points = [f"{sc[0]:.5f} {sc[1]:.5f} {sc[2]:.5f}"]

        current_segment_points.append(f"{ec[0]:.5f} {ec[1]:.5f} {ec[2]:.5f}")

    if current_segment_points:
        band_strings.append("  ".join(current_segment_points))

    band_line = "BAND = " + ", ".join(band_strings)

    # Build labels
    labels = []
    for i, (start, end) in enumerate(segments):
        if i == 0:
            labels.append(start.replace("GAMMA", "$\\Gamma$"))
        elif start != segments[i - 1][1]:
            labels.append(start.replace("GAMMA", "$\\Gamma$"))
        labels.append(end.replace("GAMMA", "$\\Gamma$"))

    label_line = "BAND_LABELS = " + " ".join(labels)

    content = f"{band_line}\n{label_line}\nBAND_POINTS = 101\n"
    with open(filename, "w") as f:
        f.write(content)
    print(f"Phonopy band.conf written to {filename}")
    print(content)
    return content

phonopy_conf = write_phonopy_bandconf(path_data)

# ============================================================
# 7. Write standardized primitive cell (POSCAR and CIF)
# ============================================================
from ase.data import chemical_symbols
from pymatgen.core import Lattice

prim_species = [chemical_symbols[z] for z in prim_types]
prim_struct = Structure(
    lattice=Lattice(prim_lattice),
    species=prim_species,
    coords=prim_positions,
)
prim_struct.to("seekpath_primitive.cif")
prim_struct.to("seekpath_primitive_POSCAR.vasp", fmt="poscar")
print(f"\nStandardized primitive cell saved:")
print(f"  seekpath_primitive.cif")
print(f"  seekpath_primitive_POSCAR.vasp")
print(f"  {prim_struct.composition.reduced_formula}, {len(prim_struct)} atoms")

# ============================================================
# 8. Visualize the Brillouin zone with high-symmetry points
# ============================================================
def plot_brillouin_zone(reciprocal_lattice, kpoint_coords, path_segments,
                        filename="brillouin_zone.png"):
    """
    Plot the first Brillouin zone with high-symmetry points and paths.
    Uses the Wigner-Seitz construction via scipy Voronoi.
    """
    from scipy.spatial import Voronoi

    # Generate reciprocal lattice points for Voronoi construction
    b = np.array(reciprocal_lattice)
    grid_points = []
    for i in range(-1, 2):
        for j in range(-1, 2):
            for k in range(-1, 2):
                grid_points.append(i * b[0] + j * b[1] + k * b[2])
    grid_points = np.array(grid_points)

    # Voronoi decomposition centered on Gamma
    vor = Voronoi(grid_points)

    # Find Gamma index (closest to origin)
    gamma_idx = np.argmin(np.linalg.norm(grid_points, axis=1))

    # Find BZ faces (Voronoi region of Gamma)
    bz_region = vor.regions[vor.point_region[gamma_idx]]
    if -1 in bz_region:
        bz_region = [v for v in bz_region if v != -1]

    bz_vertices = vor.vertices[bz_region]

    # Find BZ faces from ridge_points that include gamma_idx
    faces = []
    for ridge_idx, (p1, p2) in enumerate(vor.ridge_points):
        if p1 == gamma_idx or p2 == gamma_idx:
            face_verts = vor.ridge_vertices[ridge_idx]
            if -1 not in face_verts:
                verts = vor.vertices[face_verts]
                # Order vertices by angle around face normal
                center = verts.mean(axis=0)
                normal = grid_points[p2 if p1 == gamma_idx else p1]
                # Project onto plane perpendicular to normal
                v0 = verts[0] - center
                ref = v0 - np.dot(v0, normal) / np.dot(normal, normal) * normal
                if np.linalg.norm(ref) < 1e-10:
                    ref = np.cross(normal, [1, 0, 0])
                    if np.linalg.norm(ref) < 1e-10:
                        ref = np.cross(normal, [0, 1, 0])
                ref = ref / np.linalg.norm(ref)
                perp = np.cross(normal, ref)
                perp = perp / np.linalg.norm(perp)

                angles = []
                for v in verts:
                    dv = v - center
                    angles.append(np.arctan2(np.dot(dv, perp), np.dot(dv, ref)))
                order = np.argsort(angles)
                faces.append(verts[order])

    # Plot
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")

    # Draw BZ faces
    for face in faces:
        poly = Poly3DCollection([face], alpha=0.08, facecolor="steelblue",
                                 edgecolor="gray", linewidth=0.5)
        ax.add_collection3d(poly)

    # Plot high-symmetry points
    # Convert from crystal reciprocal to Cartesian
    b_matrix = np.array(reciprocal_lattice)
    for label, frac_coords in kpoint_coords.items():
        cart = np.dot(frac_coords, b_matrix)
        ax.scatter(*cart, s=50, c="red", zorder=5)
        display_label = label.replace("GAMMA", r"$\Gamma$")
        ax.text(cart[0], cart[1], cart[2], f"  {display_label}",
                fontsize=9, fontweight="bold")

    # Draw path segments
    for start, end in path_segments:
        sc = np.dot(kpoint_coords[start], b_matrix)
        ec = np.dot(kpoint_coords[end], b_matrix)
        ax.plot([sc[0], ec[0]], [sc[1], ec[1]], [sc[2], ec[2]],
                "r-", linewidth=2, alpha=0.7)

    # Draw reciprocal lattice vectors
    for i in range(3):
        ax.quiver(0, 0, 0, b[i, 0], b[i, 1], b[i, 2],
                  color="black", arrow_length_ratio=0.1, linewidth=1.5)
        ax.text(b[i, 0] * 1.1, b[i, 1] * 1.1, b[i, 2] * 1.1,
                f"$b_{i+1}$", fontsize=11)

    # Formatting
    max_range = max(np.abs(bz_vertices).max(), np.linalg.norm(b, axis=1).max()) * 1.2
    ax.set_xlim(-max_range, max_range)
    ax.set_ylim(-max_range, max_range)
    ax.set_zlim(-max_range, max_range)
    ax.set_xlabel("$k_x$ (1/A)")
    ax.set_ylabel("$k_y$ (1/A)")
    ax.set_zlabel("$k_z$ (1/A)")
    ax.set_title("First Brillouin Zone with High-Symmetry Path")

    plt.tight_layout()
    plt.savefig(filename, dpi=200, bbox_inches="tight")
    print(f"\nBrillouin zone plot saved: {filename}")

plot_brillouin_zone(reciprocal_lattice, kpoint_coords, path_segments)

# ============================================================
# 9. Summary of all outputs
# ============================================================
print("\n" + "=" * 60)
print("SUMMARY OF GENERATED FILES")
print("=" * 60)
print("  kpath_qe.txt              -- QE K_POINTS {crystal_b} card")
print("  KPOINTS                   -- VASP KPOINTS (line mode)")
print("  wannier90_kpath.txt       -- Wannier90 kpoint_path block")
print("  band.conf                 -- phonopy BAND settings")
print("  seekpath_primitive.cif    -- seekpath standardized primitive cell")
print("  seekpath_primitive_POSCAR.vasp -- same in POSCAR format")
print("  brillouin_zone.png        -- BZ visualization")
```

### Method B: pymatgen HighSymmKpath

```python
#!/usr/bin/env python3
"""
Generate high-symmetry k-paths using pymatgen HighSymmKpath.
Uses the Setyawan-Curtarolo convention (Comp. Mat. Sci. 49, 2010).
Also supports Hinuma (via seekpath backend) and Latimer-Munro conventions.
"""

import numpy as np
from pymatgen.core import Structure
from pymatgen.symmetry.bandstructure import HighSymmKpath
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer

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

sga = SpacegroupAnalyzer(structure)
prim = sga.get_primitive_standard_structure()
print(f"Structure: {structure.composition.reduced_formula}")
print(f"Space group: {sga.get_space_group_symbol()} ({sga.get_space_group_number()})")
print(f"Crystal system: {sga.get_crystal_system()}")
print(f"Primitive cell: {len(prim)} atoms")

# ============================================================
# 2. Generate k-path (three convention options)
# ============================================================
# Convention options:
#   "sc"       -- Setyawan-Curtarolo 2010 (default, original pymatgen)
#   "hinuma"   -- Hinuma et al. 2017 (same as seekpath)
#   "lm"       -- Latimer-Munro 2020 (ensures connected paths)

kpath = HighSymmKpath(structure, path_type="sc", symprec=0.01)
# kpath = HighSymmKpath(structure, path_type="hinuma")
# kpath = HighSymmKpath(structure, path_type="lm")

kpts_dict = kpath.kpath
print(f"\nConvention: Setyawan-Curtarolo")
print(f"\n=== High-symmetry points ===")
for label, coords in sorted(kpts_dict["kpoints"].items()):
    print(f"  {label:10s}: ({coords[0]:.4f}, {coords[1]:.4f}, {coords[2]:.4f})")

print(f"\n=== Path ===")
for segment in kpts_dict["path"]:
    print(f"  {' -> '.join(segment)}")

# ============================================================
# 3. Get explicit k-points along the path
# ============================================================
kpoints_list, labels_list = kpath.get_kpoints(
    line_density=40,  # k-points per reciprocal Angstrom
    coords_are_cartesian=False,
)

print(f"\nTotal k-points along path: {len(kpoints_list)}")

# ============================================================
# 4. Write QE K_POINTS card
# ============================================================
def write_qe_from_pymatgen(kpath, npoints=30, filename="kpath_qe_pmg.txt"):
    """Convert pymatgen HighSymmKpath to QE K_POINTS {crystal_b}."""
    kpts = kpath.kpath["kpoints"]
    path = kpath.kpath["path"]

    klines = []
    for seg_idx, segment in enumerate(path):
        for i, label in enumerate(segment):
            coords = kpts[label]
            if seg_idx == 0 and i == 0:
                npts = npoints
            elif i == 0:
                npts = npoints
            elif i == len(segment) - 1 and seg_idx == len(path) - 1:
                npts = 0
            else:
                npts = npoints
            klines.append(
                f"  {coords[0]:.10f}  {coords[1]:.10f}  {coords[2]:.10f}  {npts}  ! {label}"
            )

    content = f"K_POINTS {{crystal_b}}\n{len(klines)}\n" + "\n".join(klines) + "\n"
    with open(filename, "w") as f:
        f.write(content)
    print(f"QE K_POINTS written to {filename}")
    return content

qe_card = write_qe_from_pymatgen(kpath)
print(qe_card)

# ============================================================
# 5. Write VASP KPOINTS file
# ============================================================
def write_vasp_from_pymatgen(kpath, npoints=40, filename="KPOINTS_pmg"):
    """Convert pymatgen HighSymmKpath to VASP KPOINTS (line mode)."""
    kpts = kpath.kpath["kpoints"]
    path = kpath.kpath["path"]

    lines = [
        "K-path generated by pymatgen HighSymmKpath",
        f"{npoints}",
        "Line-mode",
        "Reciprocal",
    ]

    for segment in path:
        for i in range(len(segment) - 1):
            s_label = segment[i]
            e_label = segment[i + 1]
            sc = kpts[s_label]
            ec = kpts[e_label]
            label_s = s_label.replace("\\Gamma", "G")
            label_e = e_label.replace("\\Gamma", "G")
            lines.append(f"  {sc[0]:.10f}  {sc[1]:.10f}  {sc[2]:.10f}  ! {label_s}")
            lines.append(f"  {ec[0]:.10f}  {ec[1]:.10f}  {ec[2]:.10f}  ! {label_e}")
            lines.append("")

    content = "\n".join(lines) + "\n"
    with open(filename, "w") as f:
        f.write(content)
    print(f"VASP KPOINTS written to {filename}")
    return content

vasp_card = write_vasp_from_pymatgen(kpath)

# ============================================================
# 6. Compare all three conventions
# ============================================================
print("\n=== Convention Comparison ===")
for convention in ["sc", "hinuma", "lm"]:
    try:
        kp = HighSymmKpath(structure, path_type=convention, symprec=0.01)
        path_str = " | ".join(
            " -> ".join(seg) for seg in kp.kpath["path"]
        )
        n_points = len(kp.kpath["kpoints"])
        print(f"  {convention:8s}: {n_points} high-sym points, path: {path_str}")
    except Exception as e:
        print(f"  {convention:8s}: ERROR - {e}")
```

### Method C: VASP (Future External Access)

When VASP is available via external access, k-paths are provided through the KPOINTS file. The KPOINTS file generated above (Method A or B) is directly usable.

```python
#!/usr/bin/env python3
"""
Generate VASP KPOINTS file and POSCAR for band structure calculation.
Also generates INCAR settings for a VASP bands calculation.

Note: This generates the input files. VASP execution will be
available via future external access.
"""

import numpy as np
from pymatgen.core import Structure
from pymatgen.symmetry.bandstructure import HighSymmKpath
from pymatgen.io.vasp import Kpoints, Poscar, Incar

# Load structure
# structure = Structure.from_file("POSCAR")
structure = Structure.from_spacegroup(
    "Fd-3m",
    lattice=[[5.431, 0, 0], [0, 5.431, 0], [0, 0, 5.431]],
    species=["Si"],
    coords=[[0.0, 0.0, 0.0]],
)

# ============================================================
# 1. Generate KPOINTS using pymatgen automatic line mode
# ============================================================
kpath = HighSymmKpath(structure)

# Method 1: Use pymatgen's built-in Kpoints generation
kpoints = Kpoints.automatic_linemode(
    divisions=40,
    ibz=kpath,
)
kpoints.write_file("KPOINTS_bands")
print("VASP KPOINTS_bands written (line mode)")

# ============================================================
# 2. Write POSCAR for the primitive cell
# ============================================================
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
sga = SpacegroupAnalyzer(structure)
prim = sga.get_primitive_standard_structure()
Poscar(prim).write_file("POSCAR_prim")
print(f"POSCAR_prim written ({len(prim)} atoms)")

# ============================================================
# 3. Generate INCAR for bands calculation
# ============================================================
# Step 1: SCF INCAR
incar_scf = Incar({
    "SYSTEM": structure.composition.reduced_formula,
    "ENCUT": 520,
    "PREC": "Accurate",
    "EDIFF": 1e-6,
    "ISMEAR": 0,
    "SIGMA": 0.05,
    "IBRION": -1,
    "NSW": 0,
    "LWAVE": True,
    "LCHARG": True,
    "LORBIT": 11,
    "NEDOS": 3001,
})
incar_scf.write_file("INCAR_scf")
print("INCAR_scf written")

# Step 2: Bands INCAR (non-SCF, reads CHGCAR)
incar_bands = Incar({
    "SYSTEM": f"{structure.composition.reduced_formula} bands",
    "ENCUT": 520,
    "PREC": "Accurate",
    "EDIFF": 1e-6,
    "ISMEAR": 0,
    "SIGMA": 0.05,
    "IBRION": -1,
    "NSW": 0,
    "ICHARG": 11,         # Read charge density, non-SCF
    "LORBIT": 11,         # Projected band character
    "LWAVE": False,
    "LCHARG": False,
    "NBANDS": 24,         # Adjust: ~1.5x occupied bands
})
incar_bands.write_file("INCAR_bands")
print("INCAR_bands written (non-SCF, ICHARG=11)")

print("\n=== VASP band structure workflow ===")
print("1. SCF: cp INCAR_scf INCAR && cp KPOINTS_scf KPOINTS && vasp_std")
print("2. Bands: cp INCAR_bands INCAR && cp KPOINTS_bands KPOINTS && vasp_std")
print("3. Parse: use pymatgen Vasprun to read vasprun.xml")
```

### Method D: spglib (Low-Level BZ Queries)

```python
#!/usr/bin/env python3
"""
Use spglib for low-level symmetry analysis and custom BZ queries.
spglib provides the raw symmetry operations, Bravais lattice type,
and standardized cell, but does not generate k-paths directly.
Combine with manual k-point specification.
"""

import numpy as np
import spglib
from pymatgen.core import Structure
from pymatgen.io.ase import AseAtomsAdaptor

# Load structure
structure = Structure.from_spacegroup(
    "Fd-3m",
    lattice=[[5.431, 0, 0], [0, 5.431, 0], [0, 0, 5.431]],
    species=["Si"],
    coords=[[0.0, 0.0, 0.0]],
)

# Convert to spglib format
cell = (
    structure.lattice.matrix,
    structure.frac_coords,
    [site.specie.Z for site in structure],
)

# ============================================================
# 1. Space group analysis
# ============================================================
sym_data = spglib.get_symmetry_dataset(cell, symprec=1e-5)
print(f"Space group: {sym_data['international']} ({sym_data['number']})")
print(f"Hall symbol: {sym_data['hall']}")
print(f"Point group: {sym_data['pointgroup']}")
print(f"Number of symmetry operations: {len(sym_data['rotations'])}")

# ============================================================
# 2. Get standardized cell
# ============================================================
std_cell = spglib.standardize_cell(cell, to_primitive=True, symprec=1e-5)
if std_cell is not None:
    std_lattice, std_positions, std_numbers = std_cell
    print(f"\nStandardized primitive cell:")
    print(f"  Lattice vectors:")
    for i, vec in enumerate(std_lattice):
        print(f"    a{i+1} = ({vec[0]:.6f}, {vec[1]:.6f}, {vec[2]:.6f})")
    print(f"  Number of atoms: {len(std_numbers)}")

# ============================================================
# 3. Reciprocal lattice
# ============================================================
recip_lattice = np.linalg.inv(std_lattice).T * 2 * np.pi
print(f"\nReciprocal lattice vectors:")
for i, vec in enumerate(recip_lattice):
    print(f"  b{i+1} = ({vec[0]:.6f}, {vec[1]:.6f}, {vec[2]:.6f})")

# ============================================================
# 4. Get IR k-points (for sampling, not band paths)
# ============================================================
mesh = [8, 8, 8]
mapping, grid = spglib.get_ir_reciprocal_mesh(
    mesh, cell, is_shift=[0, 0, 0]
)
ir_indices = np.unique(mapping)
print(f"\nIrreducible k-points for {mesh} mesh: {len(ir_indices)} / {np.prod(mesh)}")

# ============================================================
# 5. Manual k-path for FCC (example)
# ============================================================
# For an FCC Bravais lattice, the standard high-symmetry points are:
fcc_kpoints = {
    "GAMMA": [0.0, 0.0, 0.0],
    "X":     [0.5, 0.0, 0.5],
    "W":     [0.5, 0.25, 0.75],
    "K":     [0.375, 0.375, 0.75],
    "U":     [0.625, 0.25, 0.625],
    "L":     [0.5, 0.5, 0.5],
}

fcc_path = [
    ("GAMMA", "X"), ("X", "W"), ("W", "K"),
    ("K", "GAMMA"), ("GAMMA", "L"), ("L", "U"),
    ("U", "W"),
]

print(f"\nManual FCC k-path:")
for s, e in fcc_path:
    sc = fcc_kpoints[s]
    ec = fcc_kpoints[e]
    print(f"  {s:6s} ({sc[0]:.3f} {sc[1]:.3f} {sc[2]:.3f}) --> "
          f"{e:6s} ({ec[0]:.3f} {ec[1]:.3f} {ec[2]:.3f})")
```

---

## Key Parameters

| Parameter | Typical Value | Notes |
|---|---|---|
| `npoints_per_segment` | 20--40 (QE), 40 (VASP) | More points = smoother bands but larger output |
| `symprec` | 1e-5 | Symmetry detection tolerance. Loosen to 1e-3 for slightly distorted cells |
| `path_type` (pymatgen) | `"sc"`, `"hinuma"`, `"lm"` | SC = Setyawan-Curtarolo, Hinuma = seekpath convention, LM = Latimer-Munro |
| `with_time_reversal` (seekpath) | True | Exploits time reversal to reduce the k-path |
| `line_density` (pymatgen) | 40 | K-points per reciprocal Angstrom along the path |
| `angle_tolerance` (seekpath) | -1.0 (auto) | Set positive for monoclinic systems with angles near 90 degrees |

## Common Issues

| Problem | Solution |
|---|---|
| seekpath returns a different cell than expected | seekpath standardizes to the Hinuma convention primitive cell. Use the returned `primitive_lattice` and `primitive_positions` for your calculation. |
| K-path labels differ between seekpath and pymatgen | Different conventions use different labeling schemes. Stick to one convention throughout a project. |
| BZ plot looks wrong for monoclinic/triclinic | These have complex BZ shapes. Ensure `angle_tolerance` is set correctly in seekpath. |
| VASP KPOINTS file gives wrong band structure | Ensure the POSCAR matches the convention used for KPOINTS. If using seekpath, use the standardized primitive cell it returns. |
| QE bands calculation gives scrambled bands | Verify that `K_POINTS {crystal_b}` uses fractional coordinates of the reciprocal lattice matching the `CELL_PARAMETERS` in the input. |
| Path misses some high-symmetry points | Some conventions include fewer points. Use `"lm"` (Latimer-Munro) in pymatgen for maximally connected paths. |
| Wrong BZ for hexagonal with c/a < 1 | seekpath handles this correctly. pymatgen `"sc"` may not distinguish variants. Use seekpath or `"hinuma"` convention. |
