# Layer Manipulation for 2D Materials

## When to Use

- Center atomic layers within the simulation cell (VASPKIT 920)
- Move or shift layers to a specific position along the c-axis
- Standardize 2D cell orientation (layers in the ab-plane, vacuum along c)
- Extract a monolayer from a bulk layered material (VASPKIT 921)
- Split a structure into individual layers
- Build multilayer or heterostructure systems (VASPKIT 923)
- Reorient a structure so the layered direction is along c

## Method Selection

| Task | Tool | Notes |
|---|---|---|
| Center layers in cell | pymatgen / ASE (Method A) | Adjust fractional coords so layer is centered |
| Extract monolayer | pymatgen (Method A) | Find interlayer gap, extract one layer |
| Build heterostructure | pymatgen (Method B) | Stack two monolayers with controlled spacing |
| Reorient cell | pymatgen + spglib (Method A) | Ensure c is perpendicular to layers |
| Standardize 2D cell | pymatgen (Method A) | Conventional cell with layers in ab-plane |

```
Need to center a 2D material in its cell?
  --> Method A: shift_sites or translate_sites

Need to extract a monolayer from bulk?
  --> Method A: layer identification + extraction

Need to build a 2D heterostructure?
  --> Method B: lattice matching + stacking

Need to reorient the cell?
  --> Method A: rotation + cell redefinition
```

## Prerequisites

- pymatgen (Structure manipulation)
- ASE (I/O and visualization)
- numpy (coordinate operations)
- spglib (symmetry, optional)

## Detailed Steps

### Method A: Layer Centering, Extraction, and Cell Standardization

```python
#!/usr/bin/env python3
"""
Layer manipulation tools for 2D materials.
Provides functions for:
  - Centering layers in the cell
  - Extracting monolayer from bulk
  - Reorienting cell so layers are in the ab-plane
  - Splitting multilayer into individual layers
  - Adding/adjusting vacuum

Corresponds to VASPKIT 920-923.
"""

import numpy as np
from pymatgen.core import Structure, Lattice
from pymatgen.io.cif import CifWriter
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
import json
from pathlib import Path

# ============================================================
# 1. Center layers in the simulation cell
# ============================================================
def center_slab(structure, axis=2):
    """
    Center the atomic layers along the specified axis.

    For 2D materials, centers the layer(s) in the middle of the cell
    along the vacuum direction (default: c-axis = axis 2).

    Parameters
    ----------
    structure : Structure
        Input structure.
    axis : int
        Axis along which to center (0=a, 1=b, 2=c).

    Returns
    -------
    Structure
        Centered structure.
    """
    struct = structure.copy()
    frac_coords = struct.frac_coords.copy()

    # Current center of mass in fractional coordinates
    z_mean = np.mean(frac_coords[:, axis])

    # Shift to center (0.5 in fractional coords)
    shift = 0.5 - z_mean
    frac_coords[:, axis] += shift

    # Wrap back into [0, 1)
    frac_coords[:, axis] = frac_coords[:, axis] % 1.0

    # Check if wrapping caused a split -- if so, fix
    z_vals = frac_coords[:, axis]
    if np.max(z_vals) - np.min(z_vals) > 0.5:
        # Atoms are split across the periodic boundary
        # Shift atoms with z > 0.8 down by 1, then re-center
        high_mask = z_vals > 0.75
        frac_coords[high_mask, axis] -= 1.0
        z_mean = np.mean(frac_coords[:, axis])
        shift = 0.5 - z_mean
        frac_coords[:, axis] += shift
        frac_coords[:, axis] = frac_coords[:, axis] % 1.0

    # Create new structure
    species = [str(s.specie) for s in struct]
    centered = Structure(struct.lattice, species, frac_coords)

    return centered


# ============================================================
# 2. Extract monolayer from bulk layered material
# ============================================================
def extract_monolayer(structure, layer_index=0, vacuum=20.0, axis=2):
    """
    Extract a single layer from a layered bulk material.

    Identifies layers by finding gaps in the atomic z-coordinates.

    Parameters
    ----------
    structure : Structure
        Bulk layered structure.
    layer_index : int
        Which layer to extract (0 = first/bottom, 1 = second, etc.).
    vacuum : float
        Vacuum thickness to add (Angstrom).
    axis : int
        Layer direction (default: 2 = c-axis).

    Returns
    -------
    Structure
        Monolayer structure with vacuum.
    """
    # Get Cartesian coordinates along the layering axis
    cart_coords = np.array([s.coords for s in structure])
    z_cart = cart_coords[:, axis]

    # Sort atoms by z-coordinate
    sorted_indices = np.argsort(z_cart)
    sorted_z = z_cart[sorted_indices]

    # Find gaps between layers
    dz = np.diff(sorted_z)
    # Interlayer gap is typically the largest gap
    gap_threshold = np.mean(dz) + 2 * np.std(dz)

    # Identify layer boundaries
    gap_positions = np.where(dz > gap_threshold)[0]

    if len(gap_positions) == 0:
        print("WARNING: No clear interlayer gap found. Returning original structure.")
        return structure

    # Define layer boundaries
    boundaries = [-1] + list(gap_positions) + [len(sorted_z) - 1]
    n_layers = len(boundaries) - 1
    print(f"Found {n_layers} layers")

    if layer_index >= n_layers:
        print(f"WARNING: layer_index {layer_index} >= n_layers {n_layers}. Using 0.")
        layer_index = 0

    # Get atom indices for the selected layer
    start_idx = boundaries[layer_index] + 1
    end_idx = boundaries[layer_index + 1] + 1
    layer_atom_indices = sorted_indices[start_idx:end_idx]

    print(f"Extracting layer {layer_index}: atoms {start_idx}-{end_idx-1} "
          f"({len(layer_atom_indices)} atoms)")

    # Extract layer atoms
    layer_species = [str(structure[i].specie) for i in layer_atom_indices]
    layer_cart_coords = cart_coords[layer_atom_indices]

    # Get layer thickness
    layer_z = layer_cart_coords[:, axis]
    layer_thickness = np.max(layer_z) - np.min(layer_z)
    print(f"Layer thickness: {layer_thickness:.3f} A")

    # Build new cell
    old_matrix = structure.lattice.matrix.copy()

    # New c-axis length = layer_thickness + vacuum
    new_c_length = layer_thickness + vacuum
    c_hat = old_matrix[axis] / np.linalg.norm(old_matrix[axis])
    new_matrix = old_matrix.copy()
    new_matrix[axis] = c_hat * new_c_length

    # Recenter layer in new cell
    layer_cart_coords[:, axis] -= np.min(layer_z)  # shift to z=0
    layer_cart_coords[:, axis] += vacuum / 2  # center in vacuum

    # Create monolayer structure
    new_lattice = Lattice(new_matrix)
    monolayer = Structure(
        new_lattice,
        layer_species,
        layer_cart_coords,
        coords_are_cartesian=True,
    )

    return monolayer


# ============================================================
# 3. Split structure into individual layers
# ============================================================
def split_layers(structure, axis=2, gap_factor=2.0):
    """
    Split a multilayer structure into individual layers.

    Parameters
    ----------
    structure : Structure
        Multilayer structure.
    axis : int
        Layer direction.
    gap_factor : float
        Factor above mean spacing to identify interlayer gaps.

    Returns
    -------
    list of list of int : Atom indices for each layer.
    """
    cart_z = np.array([s.coords[axis] for s in structure])
    sorted_indices = np.argsort(cart_z)
    sorted_z = cart_z[sorted_indices]

    dz = np.diff(sorted_z)
    if len(dz) == 0:
        return [list(range(len(structure)))]

    threshold = np.mean(dz) + gap_factor * np.std(dz)
    gap_positions = np.where(dz > threshold)[0]

    layers = []
    start = 0
    for gap in gap_positions:
        layer_indices = sorted_indices[start:gap + 1].tolist()
        layers.append(layer_indices)
        start = gap + 1
    # Last layer
    layers.append(sorted_indices[start:].tolist())

    return layers


# ============================================================
# 4. Reorient cell (ensure layers are in ab-plane)
# ============================================================
def reorient_to_c_axis(structure):
    """
    Reorient structure so the layered (vacuum) direction is along c.

    Identifies the axis with the largest lattice parameter (likely the
    vacuum direction) and rotates the cell so it becomes c.

    Parameters
    ----------
    structure : Structure
        Input structure.

    Returns
    -------
    Structure
        Reoriented structure with vacuum along c.
    """
    lengths = structure.lattice.lengths
    max_axis = np.argmax(lengths)

    if max_axis == 2:
        # Already along c
        print("Layers already along c-axis. No reorientation needed.")
        return structure.copy()

    # Permute axes: move the longest axis to c
    matrix = structure.lattice.matrix.copy()
    axes = [0, 1, 2]
    # Swap max_axis with 2
    axes[max_axis], axes[2] = axes[2], axes[max_axis]

    new_matrix = matrix[axes]
    new_lattice = Lattice(new_matrix)

    # Permute fractional coordinates accordingly
    frac_coords = structure.frac_coords.copy()
    new_frac = frac_coords[:, axes]

    species = [str(s.specie) for s in structure]
    reoriented = Structure(new_lattice, species, new_frac)

    print(f"Reoriented: axis {max_axis} -> c (vacuum direction)")
    return reoriented


# ============================================================
# 5. Example: Complete workflow
# ============================================================
# Build bulk MoS2 (2H phase)
mos2_bulk = Structure.from_spacegroup(
    "P6_3/mmc",
    Lattice.hexagonal(3.16, 12.30),
    species=["Mo", "S", "S"],
    coords=[[1/3, 2/3, 1/4], [1/3, 2/3, 0.621], [1/3, 2/3, 0.879]],
)

print("=" * 50)
print("LAYER MANIPULATION: MoS2")
print("=" * 50)

print(f"\nBulk structure:")
print(f"  Formula: {mos2_bulk.composition.reduced_formula}")
print(f"  Atoms: {len(mos2_bulk)}")
print(f"  Lattice: a={mos2_bulk.lattice.a:.4f}, b={mos2_bulk.lattice.b:.4f}, "
      f"c={mos2_bulk.lattice.c:.4f}")

# Split into layers
layers = split_layers(mos2_bulk)
print(f"  Layers found: {len(layers)}")
for i, layer in enumerate(layers):
    species = [str(mos2_bulk[j].specie) for j in layer]
    print(f"    Layer {i}: {len(layer)} atoms ({', '.join(species)})")

# Extract monolayer
monolayer = extract_monolayer(mos2_bulk, layer_index=0, vacuum=20.0)
print(f"\nMonolayer:")
print(f"  Atoms: {len(monolayer)}")
print(f"  Lattice: a={monolayer.lattice.a:.4f}, c={monolayer.lattice.c:.4f}")

# Center the monolayer
centered = center_slab(monolayer, axis=2)
print(f"\nCentered monolayer:")
z_frac = centered.frac_coords[:, 2]
print(f"  z range: {np.min(z_frac):.4f} to {np.max(z_frac):.4f}")
print(f"  z center: {np.mean(z_frac):.4f}")

# Save structures
mos2_bulk.to(filename="MoS2_bulk.cif")
monolayer.to(filename="MoS2_monolayer.cif")
centered.to(filename="MoS2_monolayer_centered.cif")
monolayer.to(filename="MoS2_monolayer.vasp", fmt="poscar")

print("\nSaved files:")
print("  MoS2_bulk.cif")
print("  MoS2_monolayer.cif")
print("  MoS2_monolayer_centered.cif")
print("  MoS2_monolayer.vasp (POSCAR)")
```

### Method B: Build 2D Heterostructures

```python
#!/usr/bin/env python3
"""
Build a 2D heterostructure by stacking two monolayers.

Handles:
  - Lattice matching (find commensurate supercells)
  - Setting interlayer distance
  - Adding vacuum
  - Creating different stacking configurations

Example: MoS2 / WS2 vertical heterostructure.
"""

import numpy as np
from pymatgen.core import Structure, Lattice
from pymatgen.analysis.interfaces.substrate_analyzer import SubstrateAnalyzer
from pymatgen.io.cif import CifWriter
import json


def build_heterostructure(layer1, layer2, interlayer_distance=3.5,
                          vacuum=20.0, axis=2):
    """
    Stack two monolayers into a heterostructure.

    Parameters
    ----------
    layer1, layer2 : Structure
        Monolayer structures. Must have similar in-plane lattice parameters.
    interlayer_distance : float
        Distance between top of layer1 and bottom of layer2 (Angstrom).
    vacuum : float
        Total vacuum thickness (Angstrom).
    axis : int
        Stacking axis (default: 2 = c).

    Returns
    -------
    Structure
        Heterostructure with both layers and vacuum.
    """
    # Get Cartesian coordinates
    cart1 = np.array([s.coords for s in layer1])
    cart2 = np.array([s.coords for s in layer2])

    species1 = [str(s.specie) for s in layer1]
    species2 = [str(s.specie) for s in layer2]

    # Layer thicknesses
    z1 = cart1[:, axis]
    z2 = cart2[:, axis]
    t1 = np.max(z1) - np.min(z1)
    t2 = np.max(z2) - np.min(z2)

    # Total thickness = t1 + interlayer + t2 + vacuum
    total_c = t1 + interlayer_distance + t2 + vacuum

    # Position layer 1 starting at vacuum/2
    z_start_1 = vacuum / 2
    cart1[:, axis] -= np.min(z1)
    cart1[:, axis] += z_start_1

    # Position layer 2 above layer 1
    z_start_2 = z_start_1 + t1 + interlayer_distance
    cart2[:, axis] -= np.min(z2)
    cart2[:, axis] += z_start_2

    # Build new lattice
    # Use layer1's in-plane parameters (average could also work)
    matrix = layer1.lattice.matrix.copy()
    c_hat = matrix[axis] / np.linalg.norm(matrix[axis])
    matrix[axis] = c_hat * total_c
    new_lattice = Lattice(matrix)

    # Combine species and coordinates
    all_species = species1 + species2
    all_cart = np.vstack([cart1, cart2])

    heterostructure = Structure(
        new_lattice,
        all_species,
        all_cart,
        coords_are_cartesian=True,
    )

    return heterostructure


def find_lattice_match(struct1, struct2, max_strain=0.05, max_area_ratio=4):
    """
    Find commensurate supercells for two structures with minimal strain.

    Parameters
    ----------
    struct1, struct2 : Structure
        Two monolayer structures.
    max_strain : float
        Maximum allowed strain (fraction).
    max_area_ratio : float
        Maximum ratio of supercell area to primitive area.

    Returns
    -------
    list of dict : Possible matching configurations.
    """
    a1 = struct1.lattice.a
    b1 = struct1.lattice.b
    a2 = struct2.lattice.a
    b2 = struct2.lattice.b

    matches = []

    for na1 in range(1, int(max_area_ratio) + 1):
        for nb1 in range(1, int(max_area_ratio) + 1):
            for na2 in range(1, int(max_area_ratio) + 1):
                for nb2 in range(1, int(max_area_ratio) + 1):
                    # Effective lattice parameters after supercell
                    eff_a1 = a1 * na1
                    eff_b1 = b1 * nb1
                    eff_a2 = a2 * na2
                    eff_b2 = b2 * nb2

                    # Strain
                    strain_a = abs(eff_a1 - eff_a2) / min(eff_a1, eff_a2)
                    strain_b = abs(eff_b1 - eff_b2) / min(eff_b1, eff_b2)

                    if strain_a <= max_strain and strain_b <= max_strain:
                        total_atoms = len(struct1) * na1 * nb1 + len(struct2) * na2 * nb2
                        matches.append({
                            "supercell_1": (na1, nb1),
                            "supercell_2": (na2, nb2),
                            "strain_a": strain_a,
                            "strain_b": strain_b,
                            "max_strain": max(strain_a, strain_b),
                            "total_atoms": total_atoms,
                        })

    # Sort by strain (lower is better), then by atom count
    matches.sort(key=lambda x: (x["max_strain"], x["total_atoms"]))

    return matches


# ============================================================
# Example: MoS2 / WS2 heterostructure
# ============================================================
# Build monolayers
mos2_mono = Structure(
    Lattice.hexagonal(3.16, 25.0),
    ["Mo", "S", "S"],
    [[1/3, 2/3, 0.5], [1/3, 2/3, 0.562], [1/3, 2/3, 0.438]],
)

ws2_mono = Structure(
    Lattice.hexagonal(3.15, 25.0),
    ["W", "S", "S"],
    [[1/3, 2/3, 0.5], [1/3, 2/3, 0.562], [1/3, 2/3, 0.438]],
)

print("=== Building MoS2/WS2 Heterostructure ===")
print(f"MoS2: a = {mos2_mono.lattice.a:.4f} A")
print(f"WS2:  a = {ws2_mono.lattice.a:.4f} A")
print(f"Lattice mismatch: {abs(mos2_mono.lattice.a - ws2_mono.lattice.a)/mos2_mono.lattice.a*100:.2f}%")

# Find lattice matches
matches = find_lattice_match(mos2_mono, ws2_mono, max_strain=0.02)
print(f"\nLattice matches (strain < 2%): {len(matches)}")
if matches:
    best = matches[0]
    print(f"Best match: MoS2 {best['supercell_1']}, WS2 {best['supercell_2']}")
    print(f"  Strain: {best['max_strain']*100:.3f}%")
    print(f"  Total atoms: {best['total_atoms']}")

# Build heterostructure (1x1 for simplicity since lattices are similar)
# Average the lattice parameter
a_avg = (mos2_mono.lattice.a + ws2_mono.lattice.a) / 2

# Rebuild monolayers with matched lattice
mos2_matched = Structure(
    Lattice.hexagonal(a_avg, 25.0),
    ["Mo", "S", "S"],
    [[1/3, 2/3, 0.5], [1/3, 2/3, 0.562], [1/3, 2/3, 0.438]],
)

ws2_matched = Structure(
    Lattice.hexagonal(a_avg, 25.0),
    ["W", "S", "S"],
    [[1/3, 2/3, 0.5], [1/3, 2/3, 0.562], [1/3, 2/3, 0.438]],
)

hetero = build_heterostructure(
    mos2_matched, ws2_matched,
    interlayer_distance=3.3,
    vacuum=20.0,
)

print(f"\nHeterostructure:")
print(f"  Atoms: {len(hetero)}")
print(f"  Lattice: a={hetero.lattice.a:.4f}, c={hetero.lattice.c:.4f}")
print(f"  Species: {hetero.composition.reduced_formula}")

hetero.to(filename="MoS2_WS2_hetero.cif")
hetero.to(filename="MoS2_WS2_hetero.vasp", fmt="poscar")
print("  Saved: MoS2_WS2_hetero.cif and .vasp")
```

## Key Parameters

| Parameter | Recommended Value | Notes |
|---|---|---|
| Vacuum thickness | 15-20 A | Must converge total energy vs vacuum. 20 A is safe for most properties. |
| Interlayer distance | 3.0-3.5 A for TMDs | Depends on the material. Use DFT-D3 or MACE with dispersion for optimization. |
| Lattice mismatch | < 5% | Smaller is better. > 5% introduces significant strain artifacts. |
| Layer centering | z_center = 0.5 | Center layer at half the c-axis for symmetric vacuum on both sides. |
| Axis convention | Layers in ab-plane, vacuum along c | VASP, QE, and most codes expect this orientation. |

## Interpreting Results

1. **Layer thickness**: Monolayer MoS2 is ~3.1 A (S-Mo-S sandwich). If extracted thickness differs significantly, check the gap detection.
2. **Vacuum sufficiency**: Compute total energy vs vacuum thickness. It should converge within 1 meV/atom by ~15 A vacuum.
3. **Heterostructure strain**: Applied strain changes band gaps and band alignment. Report the strain used.
4. **Number of layers**: The gap detection algorithm finds distinct layers based on atomic z-spacing. For materials with buckled layers (e.g., phosphorene), the gap threshold may need adjustment.

## Common Issues

| Issue | Solution |
|---|---|
| Gap detection fails | Adjust gap_factor in split_layers(). For closely spaced layers, use a smaller threshold. |
| Monolayer has wrong stoichiometry | Check that all atoms belonging to the layer were captured. Some complex structures need manual atom selection. |
| Heterostructure has wrong interlayer distance | Set interlayer_distance explicitly. Do not rely on automatic placement. |
| Cell is not oriented correctly | Use reorient_to_c_axis() to ensure vacuum is along c. |
| Lattice mismatch too large | Use larger supercells for commensurate matching, or accept small strain (< 3%). |
| Atoms near cell boundary cause issues | Center the slab first, then add vacuum. Use center_slab() before other operations. |
