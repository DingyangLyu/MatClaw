# Pore Geometry Analysis for MOFs and Zeolites

## When to Use

- Computing pore size distribution (PSD) of a framework structure
- Calculating geometric surface area and comparing with BET surface area
- Determining pore volume (geometric and helium-accessible)
- Finding largest cavity diameter (LCD), pore limiting diameter (PLD), and largest free sphere
- Computing helium void fraction using RASPA3 Widom insertion
- Analyzing channel dimensionality (1D channels, 2D layers, 3D networks)
- Screening materials for accessibility to specific gas molecules
- Validating framework structures before running expensive GCMC simulations

## Method Selection

| Property | Method | Tool |
|----------|--------|------|
| Pore size distribution | Grid-based sampling with distance-to-nearest-atom | Python (numpy) |
| Geometric surface area | Accessible surface area via probe rolling | Python (numpy/scipy) |
| BET surface area | N2 isotherm at 77 K + BET fit | RASPA3 GCMC (see adsorption-isotherm skill) |
| Pore volume | Geometric void volume from grid | Python (numpy) |
| Helium void fraction | Widom insertion of helium | RASPA3 |
| Largest cavity diameter (LCD) | Voronoi decomposition or grid search | Python |
| Pore limiting diameter (PLD) | Largest probe that can percolate | Python (grid + path finding) |
| Channel dimensionality | Percolation analysis along crystal axes | Python |
| Accessible volume for specific probe | Grid + probe radius exclusion | Python |

## Prerequisites

- RASPA3 binary (`raspa3`) -- for helium void fraction via Widom insertion
- Python packages: `pymatgen`, `numpy`, `scipy`, `matplotlib`, `ase`, `spglib` (all pre-installed)
- Framework CIF file (MOF, zeolite, COF, or any periodic porous crystal)
- For Zeo++-like analysis: implemented here in pure Python (no external Zeo++ binary required)

## Detailed Steps

### Step 1: Load and inspect framework structure

```python
#!/usr/bin/env python3
"""
Load a framework CIF and inspect its basic properties.
First step before any pore analysis.
"""
from pymatgen.core import Structure
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
import numpy as np

# Load framework
structure = Structure.from_file("framework.cif")

# Basic properties
print("=== Framework Summary ===")
print(f"Formula: {structure.composition.reduced_formula}")
print(f"Number of atoms: {len(structure)}")
print(f"Cell parameters:")
lattice = structure.lattice
print(f"  a = {lattice.a:.4f} A")
print(f"  b = {lattice.b:.4f} A")
print(f"  c = {lattice.c:.4f} A")
print(f"  alpha = {lattice.alpha:.2f} deg")
print(f"  beta  = {lattice.beta:.2f} deg")
print(f"  gamma = {lattice.gamma:.2f} deg")
print(f"Volume: {structure.volume:.2f} A^3")

# Density
mass_amu = sum(site.specie.atomic_mass for site in structure)
density = mass_amu * 1.66054e-24 / (structure.volume * 1e-24)  # g/cm^3
print(f"Framework density: {density:.4f} g/cm^3")

# Space group
sga = SpacegroupAnalyzer(structure, symprec=0.1)
print(f"Space group: {sga.get_space_group_symbol()} ({sga.get_space_group_number()})")

# Element types and their van der Waals radii (Angstrom)
vdw_radii = {
    "H": 1.20, "He": 1.40, "Li": 1.82, "Be": 1.53, "B": 1.92,
    "C": 1.70, "N": 1.55, "O": 1.52, "F": 1.47, "Ne": 1.54,
    "Na": 2.27, "Mg": 1.73, "Al": 1.84, "Si": 2.10, "P": 1.80,
    "S": 1.80, "Cl": 1.75, "Ar": 1.88, "K": 2.75, "Ca": 2.31,
    "Ti": 2.11, "V": 2.07, "Cr": 2.06, "Mn": 2.05, "Fe": 2.04,
    "Co": 2.00, "Ni": 1.97, "Cu": 1.96, "Zn": 2.01, "Ga": 1.87,
    "Ge": 2.11, "As": 1.85, "Se": 1.90, "Br": 1.85, "Kr": 2.02,
    "Zr": 2.23, "Mo": 2.17, "Pd": 2.02, "Ag": 2.03, "Cd": 2.18,
    "In": 1.93, "Sn": 2.17, "Sb": 2.06, "Te": 2.06, "I": 1.98,
    "Ba": 2.68, "La": 2.43, "Ce": 2.42, "Hf": 2.23, "W": 2.18,
    "Pt": 2.13, "Au": 2.14, "Pb": 2.02, "Bi": 2.07, "U": 2.41,
}

elements = set(str(site.specie) for site in structure)
print(f"\nElements: {', '.join(sorted(elements))}")
for el in sorted(elements):
    r = vdw_radii.get(el, 2.0)
    print(f"  {el}: vdW radius = {r:.2f} A")
```

### Step 2: Pore size distribution (PSD)

```python
#!/usr/bin/env python3
"""
Compute pore size distribution using grid-based sampling.

Method: Place a fine grid inside the unit cell. At each grid point,
compute the distance to the nearest framework atom (minus its vdW radius).
This gives the radius of the largest sphere that fits at each point.
The histogram of these radii is the pore size distribution.
"""
from pymatgen.core import Structure
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Van der Waals radii (Angstrom)
VDW_RADII = {
    "H": 1.20, "C": 1.70, "N": 1.55, "O": 1.52, "F": 1.47,
    "Si": 2.10, "P": 1.80, "S": 1.80, "Cl": 1.75,
    "Zn": 2.01, "Cu": 1.96, "Fe": 2.04, "Co": 2.00, "Ni": 1.97,
    "Al": 1.84, "Mg": 1.73, "Ca": 2.31, "Ti": 2.11, "Zr": 2.23,
    "Cr": 2.06, "Mn": 2.05, "V": 2.07, "Ba": 2.68, "K": 2.75,
    "Na": 2.27, "Li": 1.82,
}


def compute_psd(structure, grid_spacing=0.2, probe_radius=0.0):
    """
    Compute pore size distribution.

    Parameters
    ----------
    structure : pymatgen Structure
        Framework structure
    grid_spacing : float
        Grid spacing in Angstrom (smaller = more accurate but slower)
    probe_radius : float
        Probe radius (0 = geometric; 1.4 = N2-like; 1.3 = He-like)

    Returns
    -------
    radii : array
        Pore radius at each accessible grid point (Angstrom)
    grid_frac : array (N, 3)
        Fractional coordinates of grid points
    """
    lattice = structure.lattice

    # Create grid in fractional coordinates
    na = max(int(lattice.a / grid_spacing), 5)
    nb = max(int(lattice.b / grid_spacing), 5)
    nc = max(int(lattice.c / grid_spacing), 5)

    fa = np.linspace(0, 1, na, endpoint=False) + 0.5 / na
    fb = np.linspace(0, 1, nb, endpoint=False) + 0.5 / nb
    fc = np.linspace(0, 1, nc, endpoint=False) + 0.5 / nc

    grid_frac = np.array(np.meshgrid(fa, fb, fc, indexing="ij")).reshape(3, -1).T
    grid_cart = lattice.get_cartesian_coords(grid_frac)

    n_points = len(grid_cart)
    print(f"Grid: {na}x{nb}x{nc} = {n_points} points (spacing ~ {grid_spacing} A)")

    # Get framework atom positions and radii
    atom_frac = np.array([site.frac_coords for site in structure])
    atom_cart = lattice.get_cartesian_coords(atom_frac)
    atom_radii = np.array([VDW_RADII.get(str(site.specie), 2.0) for site in structure])

    # For each grid point, find distance to nearest atom surface
    # Use periodic boundary conditions
    max_pore_radius = np.zeros(n_points)

    # Process in batches to manage memory
    batch_size = 5000
    for start in range(0, n_points, batch_size):
        end = min(start + batch_size, n_points)
        batch_frac = grid_frac[start:end]

        min_dist_surface = np.full(end - start, np.inf)

        for j, site in enumerate(structure):
            # Distance from each grid point to this atom (with PBC)
            diffs_frac = batch_frac - site.frac_coords[np.newaxis, :]
            # Apply minimum image convention
            diffs_frac -= np.round(diffs_frac)
            diffs_cart = lattice.get_cartesian_coords(diffs_frac)
            distances = np.linalg.norm(diffs_cart, axis=1)

            # Distance to atom surface
            dist_surface = distances - atom_radii[j]
            min_dist_surface = np.minimum(min_dist_surface, dist_surface)

        max_pore_radius[start:end] = min_dist_surface

    # Apply probe radius
    pore_radii = max_pore_radius - probe_radius

    return pore_radii, grid_frac


def plot_psd(pore_radii, output_file, title="Pore Size Distribution"):
    """Plot pore size distribution histogram."""
    # Only include accessible pores (positive radius)
    accessible = pore_radii[pore_radii > 0]

    if len(accessible) == 0:
        print("No accessible pore space found.")
        return

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Histogram
    bins = np.linspace(0, max(accessible), 60)
    ax1.hist(accessible, bins=bins, density=True, alpha=0.7,
             color="#1565C0", edgecolor="white", linewidth=0.5)
    ax1.set_xlabel("Pore radius (Angstrom)", fontsize=14)
    ax1.set_ylabel("Probability density", fontsize=14)
    ax1.set_title(title, fontsize=15)
    ax1.grid(True, alpha=0.3)

    # Cumulative distribution
    sorted_r = np.sort(accessible)
    cdf = np.arange(1, len(sorted_r)+1) / len(sorted_r)
    ax2.plot(sorted_r, cdf, "r-", linewidth=2)
    ax2.set_xlabel("Pore radius (Angstrom)", fontsize=14)
    ax2.set_ylabel("Cumulative fraction", fontsize=14)
    ax2.set_title("Cumulative PSD", fontsize=15)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_file, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"PSD plot saved to {output_file}")


# === Run PSD analysis ===
structure = Structure.from_file("framework.cif")
pore_radii, grid_frac = compute_psd(structure, grid_spacing=0.25)

print(f"\n=== PSD Summary ===")
accessible = pore_radii[pore_radii > 0]
print(f"Fraction of void space: {len(accessible)/len(pore_radii)*100:.1f}%")
print(f"Max pore radius: {np.max(pore_radii):.2f} A")
print(f"Mean pore radius (accessible): {np.mean(accessible):.2f} A")
print(f"Median pore radius: {np.median(accessible):.2f} A")

plot_psd(pore_radii, "/tmp/pore_analysis/psd.png",
         title=f"PSD of {structure.composition.reduced_formula}")
```

### Step 3: Surface area (geometric accessible surface area)

```python
#!/usr/bin/env python3
"""
Compute geometric accessible surface area using probe-rolling method.

Method: Roll a probe sphere (radius = probe_r, typically 1.86 A for N2)
over the framework surface. Points where the probe center can sit without
overlapping framework atoms define the accessible surface.
The surface area is estimated from the number of grid points at the
boundary of accessible/inaccessible regions.
"""
from pymatgen.core import Structure
import numpy as np

VDW_RADII = {
    "H": 1.20, "C": 1.70, "N": 1.55, "O": 1.52, "F": 1.47,
    "Si": 2.10, "P": 1.80, "S": 1.80, "Cl": 1.75,
    "Zn": 2.01, "Cu": 1.96, "Fe": 2.04, "Co": 2.00, "Ni": 1.97,
    "Al": 1.84, "Mg": 1.73, "Ca": 2.31, "Ti": 2.11, "Zr": 2.23,
    "Cr": 2.06, "Mn": 2.05, "Ba": 2.68, "Na": 2.27,
}


def compute_surface_area(structure, probe_radius=1.86, grid_spacing=0.15):
    """
    Compute geometric accessible surface area.

    Parameters
    ----------
    structure : pymatgen Structure
    probe_radius : float
        Probe molecule radius (N2=1.86 A, He=1.30 A, Ar=1.88 A)
    grid_spacing : float
        Grid spacing in Angstrom

    Returns
    -------
    sa_m2_g : float
        Surface area in m^2/g
    sa_m2_cm3 : float
        Volumetric surface area in m^2/cm^3
    """
    lattice = structure.lattice

    na = max(int(lattice.a / grid_spacing), 5)
    nb = max(int(lattice.b / grid_spacing), 5)
    nc = max(int(lattice.c / grid_spacing), 5)

    dx = lattice.a / na
    dy = lattice.b / nb
    dz = lattice.c / nc

    fa = np.linspace(0, 1, na, endpoint=False) + 0.5 / na
    fb = np.linspace(0, 1, nb, endpoint=False) + 0.5 / nb
    fc = np.linspace(0, 1, nc, endpoint=False) + 0.5 / nc

    grid_frac = np.array(np.meshgrid(fa, fb, fc, indexing="ij")).reshape(3, -1).T
    n_points = len(grid_frac)

    # Classify each point as accessible or not
    accessible = np.ones(n_points, dtype=bool)

    for site in structure:
        r_atom = VDW_RADII.get(str(site.specie), 2.0)
        r_exclude = r_atom + probe_radius

        diffs = grid_frac - site.frac_coords[np.newaxis, :]
        diffs -= np.round(diffs)
        dists = np.linalg.norm(lattice.get_cartesian_coords(diffs), axis=1)

        accessible &= (dists > r_exclude)

    # Reshape to 3D grid for neighbor counting
    acc_3d = accessible.reshape(na, nb, nc)

    # Surface points: accessible points with at least one inaccessible neighbor
    surface_count = 0
    for di in [-1, 0, 1]:
        for dj in [-1, 0, 1]:
            for dk in [-1, 0, 1]:
                if di == 0 and dj == 0 and dk == 0:
                    continue
                shifted = np.roll(np.roll(np.roll(acc_3d, di, axis=0), dj, axis=1), dk, axis=2)
                # A point is on the surface if it is accessible but a neighbor is not
                surface_count += np.sum(acc_3d & ~shifted)

    # Each surface voxel face has area ~ grid_spacing^2
    # Average grid face area in Cartesian space
    vol_per_voxel = structure.volume / (na * nb * nc)
    face_area = vol_per_voxel**(2.0/3.0)  # Approximate

    # More accurate: surface area = surface_count * face_area / 6 (average over 26 neighbors)
    # But simpler: each surface-detected pair contributes one face
    total_surface_area_A2 = surface_count * face_area / 26.0  # Rough normalization

    # Convert to m^2/g
    mass_g_per_cell = sum(site.specie.atomic_mass for site in structure) * 1.66054e-24
    sa_m2_g = total_surface_area_A2 * 1e-20 / mass_g_per_cell  # A^2 -> m^2
    sa_m2_cm3 = total_surface_area_A2 * 1e-20 / (structure.volume * 1e-24)

    # Void fraction
    void_fraction = np.sum(accessible) / n_points

    print(f"=== Geometric Surface Area ===")
    print(f"Probe radius: {probe_radius} A")
    print(f"Grid: {na}x{nb}x{nc} ({n_points} points)")
    print(f"Void fraction: {void_fraction*100:.1f}%")
    print(f"Surface area: {sa_m2_g:.0f} m^2/g")
    print(f"Volumetric SA: {sa_m2_cm3:.0f} m^2/cm^3")

    return sa_m2_g, sa_m2_cm3, void_fraction


structure = Structure.from_file("framework.cif")
sa, sa_vol, vf = compute_surface_area(structure, probe_radius=1.86, grid_spacing=0.15)
```

### Step 4: Pore volume calculation

```python
#!/usr/bin/env python3
"""
Compute geometric pore volume and probe-accessible pore volume.
"""
from pymatgen.core import Structure
import numpy as np

VDW_RADII = {
    "H": 1.20, "C": 1.70, "N": 1.55, "O": 1.52, "F": 1.47,
    "Si": 2.10, "P": 1.80, "S": 1.80, "Cl": 1.75,
    "Zn": 2.01, "Cu": 1.96, "Fe": 2.04, "Co": 2.00, "Ni": 1.97,
    "Al": 1.84, "Mg": 1.73, "Ca": 2.31, "Ti": 2.11, "Zr": 2.23,
    "Cr": 2.06, "Mn": 2.05, "Ba": 2.68, "Na": 2.27,
}


def compute_pore_volume(structure, probe_radius=0.0, grid_spacing=0.2):
    """
    Compute pore volume.

    probe_radius = 0: geometric void volume (any point not inside vdW sphere)
    probe_radius = 1.30: He-accessible volume
    probe_radius = 1.86: N2-accessible volume

    Returns pore volume in cm^3/g and A^3/unit cell.
    """
    lattice = structure.lattice

    na = max(int(lattice.a / grid_spacing), 5)
    nb = max(int(lattice.b / grid_spacing), 5)
    nc = max(int(lattice.c / grid_spacing), 5)

    fa = np.linspace(0, 1, na, endpoint=False) + 0.5 / na
    fb = np.linspace(0, 1, nb, endpoint=False) + 0.5 / nb
    fc = np.linspace(0, 1, nc, endpoint=False) + 0.5 / nc

    grid_frac = np.array(np.meshgrid(fa, fb, fc, indexing="ij")).reshape(3, -1).T
    n_total = len(grid_frac)

    accessible = np.ones(n_total, dtype=bool)

    for site in structure:
        r_atom = VDW_RADII.get(str(site.specie), 2.0)
        r_exclude = r_atom + probe_radius

        diffs = grid_frac - site.frac_coords[np.newaxis, :]
        diffs -= np.round(diffs)
        dists = np.linalg.norm(lattice.get_cartesian_coords(diffs), axis=1)
        accessible &= (dists > r_exclude)

    void_fraction = np.sum(accessible) / n_total
    pore_vol_A3 = void_fraction * structure.volume  # A^3 per unit cell

    # Convert to cm^3/g
    mass_g = sum(site.specie.atomic_mass for site in structure) * 1.66054e-24
    pore_vol_cm3_g = pore_vol_A3 * 1e-24 / mass_g

    print(f"=== Pore Volume (probe_r = {probe_radius} A) ===")
    print(f"Grid: {na}x{nb}x{nc} = {n_total} points")
    print(f"Void fraction: {void_fraction:.4f} ({void_fraction*100:.1f}%)")
    print(f"Pore volume: {pore_vol_A3:.1f} A^3/uc")
    print(f"Pore volume: {pore_vol_cm3_g:.4f} cm^3/g")

    return pore_vol_cm3_g, void_fraction


structure = Structure.from_file("framework.cif")

# Geometric void volume (no probe)
pv_geom, vf_geom = compute_pore_volume(structure, probe_radius=0.0)

# Helium-accessible volume
pv_he, vf_he = compute_pore_volume(structure, probe_radius=1.30)

# N2-accessible volume
pv_n2, vf_n2 = compute_pore_volume(structure, probe_radius=1.86)

print(f"\n{'Probe':>12s} {'V_pore (cm^3/g)':>16s} {'Void frac':>10s}")
print(f"{'Geometric':>12s} {pv_geom:>16.4f} {vf_geom:>10.4f}")
print(f"{'He (1.30 A)':>12s} {pv_he:>16.4f} {vf_he:>10.4f}")
print(f"{'N2 (1.86 A)':>12s} {pv_n2:>16.4f} {vf_n2:>10.4f}")
```

### Step 5: Largest cavity diameter (LCD) and pore limiting diameter (PLD)

```python
#!/usr/bin/env python3
"""
Compute largest cavity diameter (LCD) and pore limiting diameter (PLD).

LCD: diameter of the largest sphere that fits inside the pore space
PLD: diameter of the largest sphere that can pass through the narrowest
     window connecting the pores (percolation-limited diameter)

These are key descriptors for predicting molecular sieving behavior.
"""
from pymatgen.core import Structure
import numpy as np
from scipy.ndimage import label, binary_dilation, binary_erosion
from collections import deque

VDW_RADII = {
    "H": 1.20, "C": 1.70, "N": 1.55, "O": 1.52, "F": 1.47,
    "Si": 2.10, "P": 1.80, "S": 1.80, "Cl": 1.75,
    "Zn": 2.01, "Cu": 1.96, "Fe": 2.04, "Co": 2.00, "Ni": 1.97,
    "Al": 1.84, "Mg": 1.73, "Ca": 2.31, "Ti": 2.11, "Zr": 2.23,
    "Cr": 2.06, "Mn": 2.05, "Ba": 2.68, "Na": 2.27,
}


def compute_distance_grid(structure, grid_spacing=0.15):
    """
    Compute the distance-to-nearest-atom-surface at each grid point.
    Returns the 3D grid and axis information.
    """
    lattice = structure.lattice

    na = max(int(lattice.a / grid_spacing), 5)
    nb = max(int(lattice.b / grid_spacing), 5)
    nc = max(int(lattice.c / grid_spacing), 5)

    fa = np.linspace(0, 1, na, endpoint=False) + 0.5 / na
    fb = np.linspace(0, 1, nb, endpoint=False) + 0.5 / nb
    fc = np.linspace(0, 1, nc, endpoint=False) + 0.5 / nc

    grid_frac = np.array(np.meshgrid(fa, fb, fc, indexing="ij")).reshape(3, -1).T

    # Distance to nearest atom surface
    dist_surface = np.full(len(grid_frac), np.inf)

    for site in structure:
        r_atom = VDW_RADII.get(str(site.specie), 2.0)
        diffs = grid_frac - site.frac_coords[np.newaxis, :]
        diffs -= np.round(diffs)
        dists = np.linalg.norm(lattice.get_cartesian_coords(diffs), axis=1)
        dist_surface = np.minimum(dist_surface, dists - r_atom)

    return dist_surface.reshape(na, nb, nc), (na, nb, nc), lattice


def compute_lcd(dist_grid):
    """
    Largest Cavity Diameter: twice the maximum value in the distance grid.
    """
    max_radius = np.max(dist_grid)
    lcd = 2.0 * max_radius
    # Location of LCD
    idx = np.unravel_index(np.argmax(dist_grid), dist_grid.shape)
    return lcd, idx


def compute_pld(dist_grid, grid_shape, lattice, grid_spacing=0.15):
    """
    Pore Limiting Diameter: largest probe that can percolate through
    the structure (connected path from one face to the opposite face).

    Uses binary search on probe radius + connectivity check with PBC.
    """
    na, nb, nc = grid_shape

    def is_percolating(probe_radius):
        """Check if pores accessible to this probe radius percolate in any direction."""
        accessible = dist_grid > probe_radius

        if not np.any(accessible):
            return False

        # Check percolation along each axis using periodic boundary conditions
        # Label connected components with PBC wrapping
        for axis in range(3):
            # Check if first and last slice along this axis are connected
            # through the accessible region

            # Simple check: label without PBC, then see if any label
            # appears on both the first and last slices
            labeled, n_labels = label(accessible)

            if axis == 0:
                labels_first = set(labeled[0, :, :].flatten()) - {0}
                labels_last = set(labeled[-1, :, :].flatten()) - {0}
            elif axis == 1:
                labels_first = set(labeled[:, 0, :].flatten()) - {0}
                labels_last = set(labeled[:, -1, :].flatten()) - {0}
            else:
                labels_first = set(labeled[:, :, 0].flatten()) - {0}
                labels_last = set(labeled[:, :, -1].flatten()) - {0}

            if labels_first & labels_last:
                return True

        return False

    # Binary search for PLD
    r_min = 0.0
    r_max = float(np.max(dist_grid))

    if not is_percolating(0.0):
        return 0.0  # No percolation even for point probe

    while (r_max - r_min) > 0.01:  # 0.02 A resolution
        r_mid = (r_min + r_max) / 2.0
        if is_percolating(r_mid):
            r_min = r_mid
        else:
            r_max = r_mid

    pld = 2.0 * r_min
    return pld


# === Compute LCD and PLD ===
structure = Structure.from_file("framework.cif")
spacing = 0.15  # Angstrom

print("Computing distance grid...")
dist_grid, grid_shape, lattice = compute_distance_grid(structure, grid_spacing=spacing)

lcd, lcd_idx = compute_lcd(dist_grid)
print(f"\nLargest Cavity Diameter (LCD) = {lcd:.2f} A")

print("Computing PLD (this may take a minute for large structures)...")
pld = compute_pld(dist_grid, grid_shape, lattice, grid_spacing=spacing)
print(f"Pore Limiting Diameter (PLD) = {pld:.2f} A")

# Molecular diameters for reference
mol_diameters = {
    "He":   2.60, "H2":   2.89, "CO2":  3.30,
    "N2":   3.64, "CH4":  3.80, "O2":   3.46,
    "Ar":   3.40, "C2H4": 3.90, "C2H6": 4.44,
    "C3H8": 4.30, "SF6":  5.50, "H2O":  2.65,
}

print(f"\n{'Molecule':>10s} {'Diameter (A)':>13s} {'Fits LCD?':>10s} {'Can percolate?':>15s}")
print("-" * 52)
for mol, d in sorted(mol_diameters.items(), key=lambda x: x[1]):
    fits = "Yes" if d < lcd else "No"
    percolates = "Yes" if d < pld else "No"
    print(f"{mol:>10s} {d:>13.2f} {fits:>10s} {percolates:>15s}")
```

### Step 6: Helium void fraction with RASPA3 Widom insertion

```python
#!/usr/bin/env python3
"""
Compute helium void fraction using RASPA3 Widom insertion.

This is the standard method for determining the accessible pore volume
and void fraction in porous materials. Helium is used because it has
negligible adsorption (purely repulsive at high T).
"""
import json
import subprocess
import os
import re

BASE_DIR = "/tmp/void_fraction"
FRAMEWORK = "IRMOF-1"
os.makedirs(BASE_DIR, exist_ok=True)

# Widom insertion of helium at high temperature (to avoid adsorption)
sim_input = {
    "SimulationType": "MonteCarlo",
    "NumberOfCycles": 100000,         # Many cycles for accurate statistics
    "NumberOfInitializationCycles": 0, # No equilibration needed for Widom
    "PrintEvery": 20000,
    "Systems": [{
        "Type": "Framework",
        "Name": FRAMEWORK,
        "ExternalTemperature": 298.0,  # Room temperature
        "ChargeMethod": "None",        # He is non-polar
        "ForceField": "GenericMOFs",
        "CutOff": 12.0,
        "HeliumVoidFraction": True,    # Tell RASPA3 to compute void fraction
        "Components": [{
            "Name": "helium",
            "Type": "Adsorbate",
            "MoleculeDefinition": "TraPPE",
            "WidomProbability": 1.0,
            "CreateNumberOfMolecules": 0
        }]
    }]
}

with open(os.path.join(BASE_DIR, "simulation.json"), "w") as f:
    json.dump(sim_input, f, indent=2)

print("Running RASPA3 Widom insertion for helium void fraction...")
result = subprocess.run(["raspa3"], cwd=BASE_DIR,
                        capture_output=True, text=True, timeout=3600)

log_path = os.path.join(BASE_DIR, "output.log")
with open(log_path, "w") as f:
    f.write(result.stdout)

# Parse void fraction
text = result.stdout
void_match = re.search(r"[Hh]elium void fraction\s*[:\s]*([\d.eE+-]+)", text)
if void_match:
    void_frac = float(void_match.group(1))
    print(f"\nHelium void fraction = {void_frac:.4f} ({void_frac*100:.1f}%)")
else:
    # Try alternative patterns
    void_match = re.search(r"[Vv]oid [Ff]raction\s*[:\s]*([\d.eE+-]+)", text)
    if void_match:
        void_frac = float(void_match.group(1))
        print(f"\nVoid fraction = {void_frac:.4f} ({void_frac*100:.1f}%)")
    else:
        print("Could not parse void fraction from output.")
        print("Check the output log manually.")

# Also extract Rosenbluth weight (related to Henry coefficient)
rosen_match = re.search(r"[Rr]osenbluth.*?[:\s]*([\d.eE+-]+)", text)
if rosen_match:
    print(f"Rosenbluth weight = {float(rosen_match.group(1)):.6e}")

# Estimate pore volume from void fraction
henry_match = re.search(r"Henry coefficient\s*[:\s]*([\d.eE+-]+)", text)
if henry_match:
    K_H = float(henry_match.group(1))
    print(f"Henry coefficient (He) = {K_H:.4e} mol/(kg*Pa)")
```

### Step 7: Channel dimensionality analysis

```python
#!/usr/bin/env python3
"""
Determine channel dimensionality: does the pore system form
1D channels, 2D layers, or a 3D interconnected network?

Method: Check if accessible pore space percolates along each
crystallographic direction independently.
"""
from pymatgen.core import Structure
import numpy as np
from scipy.ndimage import label

VDW_RADII = {
    "H": 1.20, "C": 1.70, "N": 1.55, "O": 1.52, "F": 1.47,
    "Si": 2.10, "Zn": 2.01, "Cu": 1.96, "Fe": 2.04, "O": 1.52,
    "N": 1.55, "Al": 1.84, "Zr": 2.23, "Co": 2.00, "Ni": 1.97,
}


def channel_analysis(structure, probe_radius=1.86, grid_spacing=0.2):
    """
    Analyze channel dimensionality.

    Returns dict with percolation info for each axis.
    """
    lattice = structure.lattice

    na = max(int(lattice.a / grid_spacing), 5)
    nb = max(int(lattice.b / grid_spacing), 5)
    nc = max(int(lattice.c / grid_spacing), 5)

    fa = np.linspace(0, 1, na, endpoint=False) + 0.5 / na
    fb = np.linspace(0, 1, nb, endpoint=False) + 0.5 / nb
    fc = np.linspace(0, 1, nc, endpoint=False) + 0.5 / nc

    grid_frac = np.array(np.meshgrid(fa, fb, fc, indexing="ij")).reshape(3, -1).T

    accessible = np.ones(len(grid_frac), dtype=bool)
    for site in structure:
        r_atom = VDW_RADII.get(str(site.specie), 2.0)
        r_exclude = r_atom + probe_radius
        diffs = grid_frac - site.frac_coords[np.newaxis, :]
        diffs -= np.round(diffs)
        dists = np.linalg.norm(lattice.get_cartesian_coords(diffs), axis=1)
        accessible &= (dists > r_exclude)

    acc_3d = accessible.reshape(na, nb, nc)
    void_frac = np.sum(accessible) / len(accessible)

    # Check percolation along each axis
    labeled, n_labels = label(acc_3d)
    axis_names = ["a", "b", "c"]
    percolates = {}

    for ax_idx, ax_name in enumerate(axis_names):
        if ax_idx == 0:
            labels_first = set(labeled[0, :, :].flatten()) - {0}
            labels_last = set(labeled[-1, :, :].flatten()) - {0}
        elif ax_idx == 1:
            labels_first = set(labeled[:, 0, :].flatten()) - {0}
            labels_last = set(labeled[:, -1, :].flatten()) - {0}
        else:
            labels_first = set(labeled[:, :, 0].flatten()) - {0}
            labels_last = set(labeled[:, :, -1].flatten()) - {0}

        percolates[ax_name] = bool(labels_first & labels_last)

    # Determine dimensionality
    n_percolating = sum(percolates.values())
    if n_percolating == 0:
        dim_label = "0D (isolated cavities)"
    elif n_percolating == 1:
        dim_label = "1D (channels)"
    elif n_percolating == 2:
        dim_label = "2D (layered)"
    else:
        dim_label = "3D (interconnected network)"

    print(f"\n=== Channel Dimensionality Analysis ===")
    print(f"Probe radius: {probe_radius} A")
    print(f"Void fraction: {void_frac*100:.1f}%")
    print(f"Connected components: {n_labels}")
    print(f"\nPercolation:")
    for ax_name, perc in percolates.items():
        status = "PERCOLATES" if perc else "blocked"
        print(f"  Along {ax_name}-axis: {status}")
    print(f"\nChannel dimensionality: {dim_label}")

    return percolates, n_percolating, dim_label, void_frac


structure = Structure.from_file("framework.cif")
percolates, n_dim, dim_label, vf = channel_analysis(structure, probe_radius=1.86)
```

### Step 8: Complete pore analysis report

```python
#!/usr/bin/env python3
"""
Generate a comprehensive pore analysis report for a framework structure.
Combines all analysis methods into a single workflow.
"""
from pymatgen.core import Structure
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
import numpy as np
import json
import subprocess
import os
import re
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

VDW_RADII = {
    "H": 1.20, "C": 1.70, "N": 1.55, "O": 1.52, "F": 1.47,
    "Si": 2.10, "P": 1.80, "S": 1.80, "Cl": 1.75,
    "Zn": 2.01, "Cu": 1.96, "Fe": 2.04, "Co": 2.00, "Ni": 1.97,
    "Al": 1.84, "Mg": 1.73, "Ca": 2.31, "Ti": 2.11, "Zr": 2.23,
    "Cr": 2.06, "Mn": 2.05, "Ba": 2.68, "Na": 2.27,
}

REPORT_DIR = "/tmp/pore_report"
os.makedirs(REPORT_DIR, exist_ok=True)


def full_pore_analysis(cif_path, report_dir=REPORT_DIR, grid_spacing=0.2):
    """Run complete pore analysis pipeline."""
    structure = Structure.from_file(cif_path)
    lattice = structure.lattice

    report = {}
    report["formula"] = structure.composition.reduced_formula
    report["n_atoms"] = len(structure)
    report["volume_A3"] = structure.volume
    report["a"] = lattice.a
    report["b"] = lattice.b
    report["c"] = lattice.c

    sga = SpacegroupAnalyzer(structure, symprec=0.1)
    report["space_group"] = sga.get_space_group_symbol()

    mass_amu = sum(site.specie.atomic_mass for site in structure)
    report["density_g_cm3"] = mass_amu * 1.66054e-24 / (structure.volume * 1e-24)
    mass_g = mass_amu * 1.66054e-24

    # Build distance grid
    na = max(int(lattice.a / grid_spacing), 5)
    nb = max(int(lattice.b / grid_spacing), 5)
    nc = max(int(lattice.c / grid_spacing), 5)

    fa = np.linspace(0, 1, na, endpoint=False) + 0.5 / na
    fb = np.linspace(0, 1, nb, endpoint=False) + 0.5 / nb
    fc = np.linspace(0, 1, nc, endpoint=False) + 0.5 / nc
    grid_frac = np.array(np.meshgrid(fa, fb, fc, indexing="ij")).reshape(3, -1).T

    dist_surface = np.full(len(grid_frac), np.inf)
    for site in structure:
        r_atom = VDW_RADII.get(str(site.specie), 2.0)
        diffs = grid_frac - site.frac_coords[np.newaxis, :]
        diffs -= np.round(diffs)
        dists = np.linalg.norm(lattice.get_cartesian_coords(diffs), axis=1)
        dist_surface = np.minimum(dist_surface, dists - r_atom)

    dist_3d = dist_surface.reshape(na, nb, nc)

    # LCD
    lcd = 2.0 * np.max(dist_3d)
    report["LCD_A"] = lcd

    # PLD (binary search)
    from scipy.ndimage import label as scipy_label

    def percolates_at_r(r):
        acc = dist_3d > r
        if not np.any(acc):
            return False
        labeled, _ = scipy_label(acc)
        for ax in range(3):
            if ax == 0:
                s1, s2 = set(labeled[0].flat)-{0}, set(labeled[-1].flat)-{0}
            elif ax == 1:
                s1, s2 = set(labeled[:,0].flat)-{0}, set(labeled[:,-1].flat)-{0}
            else:
                s1, s2 = set(labeled[:,:,0].flat)-{0}, set(labeled[:,:,-1].flat)-{0}
            if s1 & s2:
                return True
        return False

    r_lo, r_hi = 0.0, float(np.max(dist_3d))
    if percolates_at_r(0):
        while r_hi - r_lo > 0.02:
            r_mid = (r_lo + r_hi) / 2
            if percolates_at_r(r_mid):
                r_lo = r_mid
            else:
                r_hi = r_mid
        pld = 2.0 * r_lo
    else:
        pld = 0.0
    report["PLD_A"] = pld

    # Void fractions and pore volumes
    for probe_r, label_name in [(0.0, "geometric"), (1.30, "He"), (1.86, "N2")]:
        acc = dist_surface > probe_r
        vf = np.sum(acc) / len(acc)
        pv = vf * structure.volume * 1e-24 / mass_g  # cm^3/g
        report[f"void_frac_{label_name}"] = vf
        report[f"pore_vol_{label_name}_cm3_g"] = pv

    # PSD
    accessible_radii = dist_surface[dist_surface > 0]
    report["mean_pore_radius_A"] = float(np.mean(accessible_radii)) if len(accessible_radii) > 0 else 0
    report["median_pore_radius_A"] = float(np.median(accessible_radii)) if len(accessible_radii) > 0 else 0

    # Channel dimensionality (N2 probe)
    acc_n2 = (dist_3d > 1.86)
    labeled, _ = scipy_label(acc_n2)
    n_perc = 0
    perc_axes = []
    for ax in range(3):
        if ax == 0:
            s1, s2 = set(labeled[0].flat)-{0}, set(labeled[-1].flat)-{0}
        elif ax == 1:
            s1, s2 = set(labeled[:,0].flat)-{0}, set(labeled[:,-1].flat)-{0}
        else:
            s1, s2 = set(labeled[:,:,0].flat)-{0}, set(labeled[:,:,-1].flat)-{0}
        if s1 & s2:
            n_perc += 1
            perc_axes.append(["a","b","c"][ax])
    report["channel_dim"] = n_perc
    report["percolating_axes"] = perc_axes

    # === Print report ===
    print("\n" + "=" * 60)
    print(f"PORE ANALYSIS REPORT: {report['formula']}")
    print("=" * 60)
    print(f"Space group: {report['space_group']}")
    print(f"Cell: a={report['a']:.2f}, b={report['b']:.2f}, c={report['c']:.2f} A")
    print(f"Volume: {report['volume_A3']:.1f} A^3")
    print(f"Density: {report['density_g_cm3']:.4f} g/cm^3")
    print(f"Atoms: {report['n_atoms']}")
    print()
    print(f"LCD (largest cavity diameter): {report['LCD_A']:.2f} A")
    print(f"PLD (pore limiting diameter):  {report['PLD_A']:.2f} A")
    print()
    print(f"{'Probe':>12s} {'Void frac':>10s} {'V_pore (cm^3/g)':>16s}")
    for label_name in ["geometric", "He", "N2"]:
        vf = report[f"void_frac_{label_name}"]
        pv = report[f"pore_vol_{label_name}_cm3_g"]
        print(f"{label_name:>12s} {vf:>10.4f} {pv:>16.4f}")
    print()
    print(f"Mean pore radius: {report['mean_pore_radius_A']:.2f} A")
    print(f"Median pore radius: {report['median_pore_radius_A']:.2f} A")
    dim_labels = {0: "0D (isolated cavities)", 1: "1D (channels)",
                  2: "2D (layered)", 3: "3D (interconnected)"}
    print(f"Channel dimensionality: {dim_labels[report['channel_dim']]}")
    if report['percolating_axes']:
        print(f"Percolating along: {', '.join(report['percolating_axes'])}")
    print("=" * 60)

    # === Generate summary figure ===
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # PSD histogram
    if len(accessible_radii) > 0:
        axes[0].hist(accessible_radii, bins=50, density=True,
                     color="#1565C0", alpha=0.7, edgecolor="white")
        axes[0].axvline(x=lcd/2, color="red", linestyle="--",
                        label=f"LCD/2 = {lcd/2:.1f} A")
        axes[0].axvline(x=pld/2, color="green", linestyle="--",
                        label=f"PLD/2 = {pld/2:.1f} A")
    axes[0].set_xlabel("Pore radius (A)", fontsize=13)
    axes[0].set_ylabel("Probability density", fontsize=13)
    axes[0].set_title("Pore Size Distribution", fontsize=14)
    axes[0].legend(fontsize=10)
    axes[0].grid(True, alpha=0.3)

    # Void fraction vs probe size
    probes = np.linspace(0, lcd/2, 30)
    vfs = [np.sum(dist_surface > r) / len(dist_surface) for r in probes]
    axes[1].plot(probes * 2, vfs, "b-", linewidth=2)
    axes[1].axvline(x=2.60, color="gray", linestyle=":", alpha=0.7, label="He (2.6 A)")
    axes[1].axvline(x=3.64, color="orange", linestyle=":", alpha=0.7, label="N$_2$ (3.64 A)")
    axes[1].axvline(x=3.80, color="green", linestyle=":", alpha=0.7, label="CH$_4$ (3.8 A)")
    axes[1].set_xlabel("Probe diameter (A)", fontsize=13)
    axes[1].set_ylabel("Accessible void fraction", fontsize=13)
    axes[1].set_title("Void fraction vs probe size", fontsize=14)
    axes[1].legend(fontsize=9)
    axes[1].grid(True, alpha=0.3)

    # Bar chart of key metrics
    metrics = ["LCD", "PLD", "V$_{pore}$\n(He)", "V$_{pore}$\n(N$_2$)"]
    values = [report["LCD_A"], report["PLD_A"],
              report["pore_vol_He_cm3_g"]*10,  # Scale for visibility
              report["pore_vol_N2_cm3_g"]*10]
    colors = ["#E53935", "#43A047", "#1E88E5", "#FB8C00"]
    bars = axes[2].bar(metrics, values, color=colors, alpha=0.8, edgecolor="white")
    axes[2].set_ylabel("Value (A for LCD/PLD, cm$^3$/g x10 for V$_p$)", fontsize=11)
    axes[2].set_title("Key pore descriptors", fontsize=14)
    axes[2].grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    fig_path = os.path.join(report_dir, "pore_report.png")
    plt.savefig(fig_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\nReport figure saved to {fig_path}")

    # Save JSON report
    json_path = os.path.join(report_dir, "pore_report.json")
    with open(json_path, "w") as f:
        json.dump(report, f, indent=2, default=str)
    print(f"JSON report saved to {json_path}")

    return report


# === Run full analysis ===
report = full_pore_analysis("framework.cif")
```

## Key Parameters

| Parameter | Description | Typical Value |
|-----------|-------------|---------------|
| `grid_spacing` | Grid resolution (Angstrom) | 0.1-0.3; smaller = more accurate but slower |
| `probe_radius` (He) | Helium probe radius | 1.30 A |
| `probe_radius` (N2) | Nitrogen probe radius | 1.86 A |
| `probe_radius` (Ar) | Argon probe radius | 1.88 A |
| vdW radii | Per-element van der Waals radii | UFF or Bondi radii (see code) |
| RASPA3 Widom cycles | For helium void fraction | 50000-200000 for accurate statistics |
| PLD search resolution | Binary search tolerance | 0.02 A |
| CIF symmetry | Framework symmetry handling | Convert to P1 for most reliable results |

## Interpreting Results

### Largest Cavity Diameter (LCD)
- Diameter of the biggest sphere that fits inside a pore
- Determines the largest molecule that could be accommodated (static)
- IRMOF-1: ~11.0 A; ZIF-8: ~11.6 A; MFI zeolite: ~6.4 A
- If LCD < kinetic diameter of adsorbate, the molecule cannot be adsorbed

### Pore Limiting Diameter (PLD)
- Diameter of the largest sphere that can traverse the pore network through the narrowest window
- Determines molecular sieving capability (dynamic)
- If PLD < kinetic diameter of molecule, that molecule cannot diffuse through the material
- PLD < LCD always; the ratio PLD/LCD indicates how uniform the pore network is
- Materials with PLD between two molecular diameters are sieving candidates

### Kinetic diameters of common gases (reference)
| Molecule | Kinetic diameter (A) |
|----------|---------------------|
| He | 2.60 |
| H2 | 2.89 |
| H2O | 2.65 |
| CO2 | 3.30 |
| O2 | 3.46 |
| Ar | 3.40 |
| N2 | 3.64 |
| CH4 | 3.80 |
| C2H4 | 3.90 |
| C3H8 | 4.30 |
| C2H6 | 4.44 |
| SF6 | 5.50 |

### Void fraction
- Geometric: fraction of space not occupied by vdW spheres
- He void fraction: standard reference (from RASPA3 Widom or grid-based with He probe)
- N2 void fraction: relevant for gas adsorption accessibility
- Typical MOFs: 0.4-0.9; zeolites: 0.2-0.5; dense materials: < 0.1

### Pore volume
- Good MOFs for gas storage: > 0.5 cm^3/g (N2 accessible)
- Ultrahigh porosity (NU-110, DUT-60): > 3 cm^3/g
- Zeolites: typically 0.1-0.3 cm^3/g
- If geometric pore volume >> He-accessible volume, pores have narrow windows

### Channel dimensionality
- **0D** (isolated cavities): molecule can enter but not diffuse; surface-only adsorption
- **1D** (channels): transport along one direction; may have diffusion limitations
- **2D** (layered): transport in a plane; less common
- **3D** (interconnected): best for fast mass transport; ideal for adsorption

### Surface area comparison
- Geometric SA from grid analysis is approximate; use BET from N2 isotherm for publications
- If geometric SA >> BET SA, some pores may be inaccessible to N2 at 77 K
- If BET SA is unreasonably high, check for CIF errors (overlapping atoms, partial occupancy)

## Common Issues

| Problem | Symptom | Solution |
|---------|---------|----------|
| Grid too coarse | LCD/PLD values jump with different grid_spacing | Reduce `grid_spacing` to 0.15 or 0.10 A; check convergence |
| Wrong vdW radii | Void fraction does not match literature | Verify vdW radii for all elements in the framework; use UFF or Bondi radii consistently |
| CIF has partial occupancy | Artificially large pores or wrong atom positions | Remove disordered atoms; use only fully occupied sites |
| CIF has solvent molecules | Pore space appears blocked | Remove solvent/guest molecules from CIF before analysis |
| P1 expansion issues | Space group expansion gives wrong structure | Use `SpacegroupAnalyzer.get_conventional_standard_structure()` then convert to P1 |
| PLD = 0 despite visible pores | Pores are not connected across PBC | Check if unit cell needs replication; some structures have pores only within the unit cell |
| He void fraction from RASPA3 differs from grid | Different methods use different LJ parameters | RASPA3 uses LJ He parameters, not hard-sphere; small differences (5-10%) are expected |
| Very slow for large unit cells | > 100 atoms, small grid spacing | Increase `grid_spacing` to 0.3 A for screening; use 0.15 A only for final analysis |
| LCD unrealistically large | > 20 A in small unit cell | Check CIF for missing atoms; ensure all framework atoms are present |
| Surface area calculation inaccurate | Grid-based SA is a rough estimate | Use Monte Carlo surface area (random ray casting) or BET from simulated isotherm for accuracy |
