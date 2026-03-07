# Structure Comparison and Matching

## When to Use

- Verify that a relaxation (MACE or DFT) did not change the crystal structure type.
- Compare two polymorphs or candidate structures for equivalence.
- Compute RMSD between atomic positions of two structures.
- Find the atomic site mapping between two representations of the same structure.
- Compare symmetry (space group, point group) before and after a transformation.
- Deduplicate a set of candidate structures from a structure search.

## Prerequisites

- `pymatgen` (StructureMatcher, SpacegroupAnalyzer, Structure)
- `spglib` (symmetry analysis)
- `numpy` (numerical operations)
- `matplotlib` (visualization)

## Detailed Steps

### 1. Basic structure matching with pymatgen

```python
from pymatgen.core import Structure
from pymatgen.analysis.structure_matcher import StructureMatcher

# --- Load two structures ---
struct1 = Structure.from_file("structure_before.cif")
struct2 = Structure.from_file("structure_after.cif")

print(f"Structure 1: {struct1.composition.reduced_formula}, "
      f"{len(struct1)} sites, V = {struct1.volume:.2f} A^3")
print(f"Structure 2: {struct2.composition.reduced_formula}, "
      f"{len(struct2)} sites, V = {struct2.volume:.2f} A^3")

# --- Create StructureMatcher ---
# Default tolerances:
#   ltol=0.2     (fractional length tolerance)
#   stol=0.3     (site tolerance in fractional coords)
#   angle_tol=5  (angle tolerance in degrees)
matcher = StructureMatcher(ltol=0.2, stol=0.3, angle_tol=5)

# --- Check if structures match ---
is_match = matcher.fit(struct1, struct2)
print(f"\nStructures match: {is_match}")

# --- Get RMS displacement ---
# Returns None if structures do not match
rms = matcher.get_rms_dist(struct1, struct2)
if rms is not None:
    rms_norm, rms_max = rms
    print(f"RMS displacement (normalized): {rms_norm:.6f}")
    print(f"Max displacement:              {rms_max:.6f}")
else:
    print("Structures do not match - cannot compute RMS displacement")
```

### 2. Adjusting tolerances for different use cases

```python
from pymatgen.analysis.structure_matcher import StructureMatcher

# --- Strict matching (e.g., checking if relaxation preserved structure) ---
strict_matcher = StructureMatcher(
    ltol=0.05,      # 5% length tolerance
    stol=0.1,       # tight site tolerance
    angle_tol=2,    # 2 degree angle tolerance
)

# --- Loose matching (e.g., comparing across different methods) ---
loose_matcher = StructureMatcher(
    ltol=0.5,       # 50% length tolerance
    stol=0.5,       # loose site tolerance
    angle_tol=10,   # 10 degree angle tolerance
)

# --- Matching ignoring site ordering (useful for disordered structures) ---
from pymatgen.analysis.structure_matcher import OrderDisorderElementComparator
disorder_matcher = StructureMatcher(
    comparator=OrderDisorderElementComparator(),
)

# --- Matching with species-agnostic comparison ---
from pymatgen.analysis.structure_matcher import FrameworkComparator
# Only compares the framework (ignores which species is on which site)
framework_matcher = StructureMatcher(
    comparator=FrameworkComparator(),
)
```

### 3. Find site mapping between two structures

```python
import numpy as np
from pymatgen.core import Structure
from pymatgen.analysis.structure_matcher import StructureMatcher

struct1 = Structure.from_file("structure_1.cif")
struct2 = Structure.from_file("structure_2.cif")

matcher = StructureMatcher(ltol=0.2, stol=0.3, angle_tol=5)

# --- Get the mapping ---
# Returns a list mapping each site in struct1 to the corresponding
# site index in struct2. Returns None if no match.
mapping = matcher.get_mapping(struct1, struct2)

if mapping is not None:
    print("Site mapping (struct1 -> struct2):")
    print(f"{'Site1':>6s} {'Species1':>10s} {'Site2':>6s} {'Species2':>10s}")
    print("-" * 36)
    for idx1, idx2 in enumerate(mapping):
        sp1 = str(struct1[idx1].specie)
        sp2 = str(struct2[idx2].specie)
        print(f"{idx1:>6d} {sp1:>10s} {idx2:>6d} {sp2:>10s}")

    # --- Compute per-site displacements ---
    # Get the transformed struct2 that aligns with struct1
    s2_matched = matcher.get_s2_like_s1(struct1, struct2)
    if s2_matched is not None:
        print("\nPer-site displacements:")
        displacements = []
        for i in range(len(struct1)):
            # Use fractional coordinates distance with periodic boundaries
            frac1 = struct1[i].frac_coords
            frac2 = s2_matched[i].frac_coords
            # Cartesian distance
            cart_disp = np.linalg.norm(
                struct1.lattice.get_cartesian_coords(frac1)
                - s2_matched.lattice.get_cartesian_coords(frac2)
            )
            displacements.append(cart_disp)
            print(f"  Site {i:>3d} ({str(struct1[i].specie):>4s}): "
                  f"{cart_disp:.4f} A")

        displacements = np.array(displacements)
        print(f"\nMean displacement: {displacements.mean():.4f} A")
        print(f"Max displacement:  {displacements.max():.4f} A")
        print(f"RMS displacement:  {np.sqrt(np.mean(displacements**2)):.4f} A")
else:
    print("No mapping found - structures are too different")
```

### 4. Symmetry comparison with spglib

```python
import spglib
import numpy as np
from pymatgen.core import Structure
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer

struct1 = Structure.from_file("structure_before.cif")
struct2 = Structure.from_file("structure_after.cif")

# --- Using pymatgen SpacegroupAnalyzer ---
tolerances = [0.01, 0.05, 0.1, 0.5]  # Angstrom

print("=== Symmetry Analysis ===\n")
for label, struct in [("Before", struct1), ("After", struct2)]:
    print(f"--- {label}: {struct.composition.reduced_formula} ---")
    for tol in tolerances:
        sga = SpacegroupAnalyzer(struct, symprec=tol)
        sg_symbol = sga.get_space_group_symbol()
        sg_number = sga.get_space_group_number()
        pg = sga.get_point_group_symbol()
        crystal_system = sga.get_crystal_system()
        n_ops = len(sga.get_symmetry_operations())
        print(f"  symprec={tol:.2f}: {sg_symbol} (#{sg_number}), "
              f"PG={pg}, {crystal_system}, {n_ops} sym ops")
    print()

# --- Direct spglib analysis ---
def structure_to_spglib(structure):
    """Convert pymatgen Structure to spglib cell tuple."""
    lattice = structure.lattice.matrix
    positions = structure.frac_coords
    numbers = [site.specie.Z for site in structure]
    return (lattice, positions, numbers)

for label, struct in [("Before", struct1), ("After", struct2)]:
    cell = structure_to_spglib(struct)
    dataset = spglib.get_symmetry_dataset(cell, symprec=0.1)
    if dataset is not None:
        print(f"{label}:")
        print(f"  International: {dataset['international']}")
        print(f"  Hall:          {dataset['hall']}")
        print(f"  Number:        {dataset['number']}")
        print(f"  Wyckoff:       {''.join(dataset['wyckoffs'])}")
        print(f"  Equivalent atoms: {dataset['equivalent_atoms']}")
        print()
```

### 5. Check if relaxation preserved structure type

```python
from pymatgen.core import Structure
from pymatgen.analysis.structure_matcher import StructureMatcher
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer

initial = Structure.from_file("POSCAR_initial")
relaxed = Structure.from_file("POSCAR_relaxed")

print("=== Relaxation Fidelity Check ===\n")

# --- 1. Volume change ---
dv = (relaxed.volume - initial.volume) / initial.volume * 100
print(f"Volume change: {dv:+.2f}%")

# --- 2. Lattice parameter changes ---
a1, b1, c1 = initial.lattice.abc
a2, b2, c2 = relaxed.lattice.abc
alpha1, beta1, gamma1 = initial.lattice.angles
alpha2, beta2, gamma2 = relaxed.lattice.angles

print(f"\nLattice parameters:")
print(f"  a: {a1:.4f} -> {a2:.4f} ({(a2-a1)/a1*100:+.2f}%)")
print(f"  b: {b1:.4f} -> {b2:.4f} ({(b2-b1)/b1*100:+.2f}%)")
print(f"  c: {c1:.4f} -> {c2:.4f} ({(c2-c1)/c1*100:+.2f}%)")
print(f"  alpha: {alpha1:.2f} -> {alpha2:.2f} ({alpha2-alpha1:+.2f} deg)")
print(f"  beta:  {beta1:.2f} -> {beta2:.2f} ({beta2-beta1:+.2f} deg)")
print(f"  gamma: {gamma1:.2f} -> {gamma2:.2f} ({gamma2-gamma1:+.2f} deg)")

# --- 3. Structure matching ---
matcher = StructureMatcher(ltol=0.2, stol=0.3, angle_tol=5)
match = matcher.fit(initial, relaxed)
print(f"\nStructure match (default tol): {match}")

rms = matcher.get_rms_dist(initial, relaxed)
if rms:
    print(f"RMS displacement: {rms[0]:.4f} (normalized), {rms[1]:.4f} (max)")

# --- 4. Symmetry comparison ---
sga_init = SpacegroupAnalyzer(initial, symprec=0.1)
sga_relax = SpacegroupAnalyzer(relaxed, symprec=0.1)

sg_init = sga_init.get_space_group_symbol()
sg_relax = sga_relax.get_space_group_symbol()

print(f"\nSpace group: {sg_init} -> {sg_relax}")

if sg_init == sg_relax and match:
    print("\nVERDICT: Relaxation preserved the structure type.")
elif match and sg_init != sg_relax:
    print("\nVERDICT: Structure framework preserved but symmetry changed "
          "(possible distortion).")
else:
    print("\nWARNING: Relaxation may have changed the structure type!")
```

### 6. Deduplicate a set of structures

```python
import glob
from pymatgen.core import Structure
from pymatgen.analysis.structure_matcher import StructureMatcher

# --- Load all candidate structures ---
files = sorted(glob.glob("candidates/*.cif"))
structures = []
for f in files:
    try:
        s = Structure.from_file(f)
        structures.append((f, s))
    except Exception as e:
        print(f"Error loading {f}: {e}")

print(f"Loaded {len(structures)} structures")

# --- Group equivalent structures ---
matcher = StructureMatcher(ltol=0.2, stol=0.3, angle_tol=5)

groups = []  # list of lists of (filename, structure)
assigned = [False] * len(structures)

for i, (f1, s1) in enumerate(structures):
    if assigned[i]:
        continue
    group = [(f1, s1)]
    assigned[i] = True
    for j, (f2, s2) in enumerate(structures):
        if j <= i or assigned[j]:
            continue
        if matcher.fit(s1, s2):
            group.append((f2, s2))
            assigned[j] = True
    groups.append(group)

print(f"\nFound {len(groups)} unique structures from {len(structures)} candidates:\n")
for gi, group in enumerate(groups):
    print(f"Group {gi + 1}: {len(group)} structure(s)")
    for f, s in group:
        print(f"  {f} ({s.composition.reduced_formula})")
```

### 7. Visualize structure overlay

```python
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from pymatgen.core import Structure
from pymatgen.analysis.structure_matcher import StructureMatcher

struct1 = Structure.from_file("structure_1.cif")
struct2 = Structure.from_file("structure_2.cif")

# --- Align struct2 to struct1 ---
matcher = StructureMatcher(ltol=0.3, stol=0.5, angle_tol=5)
s2_aligned = matcher.get_s2_like_s1(struct1, struct2)

if s2_aligned is None:
    print("Cannot align structures - they do not match")
else:
    # --- Get Cartesian coordinates ---
    coords1 = np.array([site.coords for site in struct1])
    coords2 = np.array([site.coords for site in s2_aligned])

    elements1 = [str(site.specie) for site in struct1]
    elements2 = [str(site.specie) for site in s2_aligned]

    # --- Color map by element ---
    unique_elements = sorted(set(elements1 + elements2))
    color_map = {}
    cmap = plt.cm.Set1
    for i, el in enumerate(unique_elements):
        color_map[el] = cmap(i / max(len(unique_elements), 1))

    # --- 3D scatter plot ---
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")

    # Plot struct1 as solid circles
    for i, (coord, el) in enumerate(zip(coords1, elements1)):
        ax.scatter(*coord, c=[color_map[el]], s=100, marker="o",
                   edgecolors="black", linewidths=0.5, alpha=0.8,
                   label=f"{el} (struct1)" if i == elements1.index(el) else "")

    # Plot struct2 as open circles
    for i, (coord, el) in enumerate(zip(coords2, elements2)):
        ax.scatter(*coord, c=[color_map[el]], s=100, marker="^",
                   edgecolors="black", linewidths=0.5, alpha=0.5,
                   label=f"{el} (struct2)" if i == elements2.index(el) else "")

    # Draw displacement vectors
    for c1, c2 in zip(coords1, coords2):
        ax.plot([c1[0], c2[0]], [c1[1], c2[1]], [c1[2], c2[2]],
                "k-", alpha=0.3, linewidth=0.5)

    # Draw unit cell edges
    lattice = struct1.lattice.matrix
    origin = np.array([0, 0, 0])
    for i in range(3):
        for j in range(3):
            if i != j:
                for k in [0, 1]:
                    start = k * lattice[3 - i - j]
                    ax.plot(
                        [start[0] + origin[0],
                         start[0] + lattice[i][0] + origin[0]],
                        [start[1] + origin[1],
                         start[1] + lattice[i][1] + origin[1]],
                        [start[2] + origin[2],
                         start[2] + lattice[i][2] + origin[2]],
                        "gray", alpha=0.3, linewidth=0.5,
                    )

    ax.set_xlabel("x (A)")
    ax.set_ylabel("y (A)")
    ax.set_zlabel("z (A)")
    ax.set_title("Structure Overlay (circles=struct1, triangles=struct2)")

    # Remove duplicate legend entries
    handles, labels = ax.get_legend_handles_labels()
    unique = dict(zip(labels, handles))
    ax.legend(unique.values(), unique.keys(), loc="upper right")

    plt.tight_layout()
    plt.savefig("structure_overlay.png", dpi=150)
    plt.close()
    print("Saved: structure_overlay.png")
```

### 8. Compute RMSD between two structures (standalone)

```python
import numpy as np
from pymatgen.core import Structure


def compute_rmsd(struct1, struct2):
    """
    Compute RMSD between two structures with the same number of sites
    and composition. Handles periodic boundary conditions.

    Returns:
        rmsd: root mean square displacement in Angstrom
        per_site_disp: array of per-site displacements
    """
    if len(struct1) != len(struct2):
        raise ValueError(
            f"Different number of sites: {len(struct1)} vs {len(struct2)}"
        )

    if struct1.composition != struct2.composition:
        raise ValueError(
            f"Different compositions: {struct1.composition} vs {struct2.composition}"
        )

    displacements = []
    for i in range(len(struct1)):
        # Fractional coordinate difference with minimum image convention
        frac_diff = struct1[i].frac_coords - struct2[i].frac_coords
        # Apply minimum image convention
        frac_diff = frac_diff - np.round(frac_diff)
        # Convert to Cartesian
        cart_diff = struct1.lattice.get_cartesian_coords(frac_diff)
        disp = np.linalg.norm(cart_diff)
        displacements.append(disp)

    displacements = np.array(displacements)
    rmsd = np.sqrt(np.mean(displacements**2))

    return rmsd, displacements


# --- Usage ---
struct1 = Structure.from_file("structure_1.cif")
struct2 = Structure.from_file("structure_2.cif")

rmsd, per_site = compute_rmsd(struct1, struct2)
print(f"RMSD: {rmsd:.4f} A")
print(f"Max displacement: {per_site.max():.4f} A")
print(f"Min displacement: {per_site.min():.4f} A")
```

## Key Parameters

| Parameter | Description |
|-----------|-------------|
| `ltol` | Fractional length tolerance for lattice matching. Default 0.2 (20%). Reduce for stricter matching. |
| `stol` | Site tolerance in fractional coordinates. Default 0.3. Lower = stricter. |
| `angle_tol` | Angle tolerance in degrees. Default 5. Lower = stricter. |
| `primitive_cell` | If `True` (default), reduce to primitive cell before comparison. Set `False` to compare supercells directly. |
| `scale` | If `True` (default), normalize volumes before comparing. Set `False` to require same volume. |
| `comparator` | Controls how species are compared. Default: `ElementComparator()`. Use `FrameworkComparator()` for topology-only comparison. |
| `symprec` (spglib) | Symmetry precision in Angstrom. Lower = stricter symmetry detection. Typical: 0.01-0.1. |

## Interpreting Results

- **`fit()` returns True**: The two structures are considered equivalent within the specified tolerances. They have the same structure type, possibly with different lattice parameters or slight atomic displacements.
- **`fit()` returns False**: The structures are different. Try loosening tolerances if you expect a match, or the structures genuinely represent different phases.
- **RMS displacement**: Quantifies how much atoms moved. < 0.1 A typically means negligible change. > 0.5 A suggests significant distortion.
- **Space group change**: If the space group changes after relaxation, a phase transition or symmetry-breaking distortion occurred. Check at multiple symprec values to confirm.
- **Wyckoff positions**: Changes in Wyckoff letters indicate atoms moved to different symmetry sites.

**Typical thresholds**:

| RMSD range | Interpretation |
|-----------|----------------|
| < 0.01 A | Essentially identical |
| 0.01 - 0.1 A | Minor relaxation, same structure |
| 0.1 - 0.5 A | Moderate distortion, likely same structure type |
| 0.5 - 1.0 A | Significant distortion, may be different phase |
| > 1.0 A | Different structure |

## Common Issues

| Issue | Solution |
|-------|----------|
| `fit()` returns False for obviously similar structures | Loosen tolerances (`ltol`, `stol`, `angle_tol`). Check that both structures have the same composition. |
| `get_rms_dist()` returns None | Structures do not match. Loosen tolerances or verify they are the same phase. |
| Different number of atoms | StructureMatcher can handle supercells if `primitive_cell=True`. Ensure both structures reduce to the same primitive cell. |
| Matching very distorted structures | Use `FrameworkComparator()` to ignore species and compare only topology. |
| spglib finds wrong space group | Adjust `symprec`. Too tight may give P1; too loose may give artificially high symmetry. Try a range (0.01 to 0.5). |
| Structures have different species | Use `FrameworkComparator()` for topology matching, or `OrderDisorderElementComparator()` for partial occupancy. |
| Slow for large sets of structures | StructureMatcher is O(N) per comparison. For M structures, grouping is O(M^2). For very large sets (>1000), consider fingerprint-based pre-screening. |
| Site mapping order seems wrong | `get_s2_like_s1` reorders struct2 to match struct1. Use this before computing per-site displacements. |
