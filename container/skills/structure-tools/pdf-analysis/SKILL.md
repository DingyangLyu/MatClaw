# Pair Distribution Function (PDF) Analysis

## When to Use

- Characterize local atomic structure (nearest-neighbor distances, coordination environments).
- Compare crystalline and amorphous/disordered structures.
- Compute element-specific partial pair distribution functions.
- Analyze MD trajectories for time-averaged structural information.
- Validate MACE-relaxed structures against known bond lengths.
- Identify short-range order in glasses, liquids, or nanoparticles.

## Prerequisites

- `pymatgen` (Structure, loading CIF/POSCAR)
- `ase` (for MD trajectory handling)
- `numpy` (numerical computation)
- `matplotlib` (plotting)
- Optionally: `scipy` for smoothing

## Detailed Steps

### 1. Compute the total pair distribution function g(r)

```python
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pymatgen.core import Structure

# --- Load structure ---
structure = Structure.from_file("my_structure.cif")

# --- Parameters ---
r_max = 10.0       # maximum r in Angstrom
r_min = 0.0        # minimum r in Angstrom
bin_width = 0.05   # bin width in Angstrom
n_bins = int((r_max - r_min) / bin_width)

# --- Compute all pairwise distances ---
# Use pymatgen's get_all_neighbors for periodic boundary conditions
r_edges = np.linspace(r_min, r_max, n_bins + 1)
r_centers = 0.5 * (r_edges[:-1] + r_edges[1:])
hist = np.zeros(n_bins)

all_neighbors = structure.get_all_neighbors(r_max)
for site_neighbors in all_neighbors:
    for neighbor in site_neighbors:
        dist = neighbor.nn_distance
        if r_min < dist < r_max:
            bin_idx = int((dist - r_min) / bin_width)
            if 0 <= bin_idx < n_bins:
                hist[bin_idx] += 1

# --- Normalize to g(r) ---
n_atoms = len(structure)
volume = structure.volume  # in Angstrom^3
rho = n_atoms / volume     # number density

# Each pair counted once from each end, so total counts = 2 * n_pairs
# Normalization: g(r) = hist / (N * rho * 4*pi*r^2 * dr)
shell_volumes = 4.0 * np.pi * r_centers**2 * bin_width
# Divide by N_atoms (each atom contributes its neighbors)
# hist already counts from all atoms, so divide by N
g_r = hist / (n_atoms * rho * shell_volumes)

# --- Plot ---
fig, ax = plt.subplots(figsize=(10, 5))
ax.plot(r_centers, g_r, "b-", linewidth=1.5)
ax.axhline(y=1.0, color="gray", linestyle="--", alpha=0.5, label="g(r) = 1 (bulk)")
ax.set_xlabel("r (Angstrom)", fontsize=12)
ax.set_ylabel("g(r)", fontsize=12)
ax.set_title(f"Pair Distribution Function: {structure.composition.reduced_formula}")
ax.set_xlim(r_min, r_max)
ax.set_ylim(0, None)
ax.legend()
plt.tight_layout()
plt.savefig("pdf_total.png", dpi=150)
plt.close()
print("Saved: pdf_total.png")

# --- Print peak positions (first few neighbor shells) ---
from scipy.signal import find_peaks
peaks, properties = find_peaks(g_r, height=1.5, distance=5)
print("\nPeak positions in g(r):")
for p in peaks:
    print(f"  r = {r_centers[p]:.3f} A, g(r) = {g_r[p]:.2f}")
```

### 2. Element-specific partial pair distribution functions

```python
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pymatgen.core import Structure
from itertools import combinations_with_replacement

structure = Structure.from_file("my_structure.cif")

# --- Parameters ---
r_max = 8.0
bin_width = 0.05
n_bins = int(r_max / bin_width)
r_edges = np.linspace(0, r_max, n_bins + 1)
r_centers = 0.5 * (r_edges[:-1] + r_edges[1:])

# --- Get unique element pairs ---
elements = sorted(set(str(s.specie) for s in structure))
print(f"Elements: {elements}")

element_indices = {}
for i, site in enumerate(structure):
    el = str(site.specie)
    if el not in element_indices:
        element_indices[el] = []
    element_indices[el].append(i)

# --- Compute partial PDFs ---
n_atoms = len(structure)
volume = structure.volume
rho = n_atoms / volume
shell_volumes = 4.0 * np.pi * r_centers**2 * bin_width

partial_pdfs = {}
all_neighbors = structure.get_all_neighbors(r_max)

for el1, el2 in combinations_with_replacement(elements, 2):
    hist = np.zeros(n_bins)
    n_el1 = len(element_indices[el1])
    n_el2 = len(element_indices[el2])

    for i in element_indices[el1]:
        for neighbor in all_neighbors[i]:
            neighbor_el = str(neighbor.specie)
            if neighbor_el == el2:
                dist = neighbor.nn_distance
                if 0 < dist < r_max:
                    bin_idx = int(dist / bin_width)
                    if 0 <= bin_idx < n_bins:
                        hist[bin_idx] += 1

    # If el1 != el2, we also need to count from el2 -> el1
    if el1 != el2:
        for i in element_indices[el2]:
            for neighbor in all_neighbors[i]:
                neighbor_el = str(neighbor.specie)
                if neighbor_el == el1:
                    dist = neighbor.nn_distance
                    if 0 < dist < r_max:
                        bin_idx = int(dist / bin_width)
                        if 0 <= bin_idx < n_bins:
                            hist[bin_idx] += 1

    # Normalize
    # For partial g_ab(r): normalize by concentration and density
    c_el1 = n_el1 / n_atoms
    c_el2 = n_el2 / n_atoms
    # Number of (el1) atoms contributing
    n_source = n_el1 if el1 == el2 else (n_el1 + n_el2)
    rho_target = (n_el2 / volume) if el1 == el2 else (n_el1 * n_el2 / (n_atoms * volume / n_atoms))

    # Simple normalization: g_ab(r) = V * hist / (N_a * N_b * 4pi*r^2*dr) for a!=b
    if el1 == el2:
        # Self-correlation: N_a*(N_a-1)/V normalization
        if n_el1 > 1:
            g_partial = hist * volume / (n_el1 * n_el1 * shell_volumes)
        else:
            g_partial = np.zeros(n_bins)
    else:
        # Cross-correlation
        if n_el1 > 0 and n_el2 > 0:
            g_partial = hist * volume / (2 * n_el1 * n_el2 * shell_volumes)
        else:
            g_partial = np.zeros(n_bins)

    pair_label = f"{el1}-{el2}"
    partial_pdfs[pair_label] = g_partial

# --- Plot ---
fig, ax = plt.subplots(figsize=(10, 6))
colors = plt.cm.Set1(np.linspace(0, 1, len(partial_pdfs)))

for (pair_label, g_partial), color in zip(partial_pdfs.items(), colors):
    ax.plot(r_centers, g_partial, label=pair_label, linewidth=1.5, color=color)

ax.axhline(y=1.0, color="gray", linestyle="--", alpha=0.5)
ax.set_xlabel("r (Angstrom)", fontsize=12)
ax.set_ylabel("g(r)", fontsize=12)
ax.set_title(f"Partial PDFs: {structure.composition.reduced_formula}")
ax.set_xlim(0, r_max)
ax.set_ylim(0, None)
ax.legend()
plt.tight_layout()
plt.savefig("pdf_partial.png", dpi=150)
plt.close()
print("Saved: pdf_partial.png")
```

### 3. Reusable PDF function

```python
import numpy as np
from pymatgen.core import Structure


def compute_pdf(structure, r_max=10.0, bin_width=0.05, element_pair=None):
    """
    Compute pair distribution function g(r) from a pymatgen Structure.

    Args:
        structure: pymatgen Structure object
        r_max: maximum distance in Angstrom
        bin_width: histogram bin width in Angstrom
        element_pair: tuple of (el1, el2) for partial PDF, or None for total

    Returns:
        r_centers: array of r values (Angstrom)
        g_r: pair distribution function
    """
    n_bins = int(r_max / bin_width)
    r_edges = np.linspace(0, r_max, n_bins + 1)
    r_centers = 0.5 * (r_edges[:-1] + r_edges[1:])
    hist = np.zeros(n_bins)

    n_atoms = len(structure)
    volume = structure.volume

    all_neighbors = structure.get_all_neighbors(r_max)

    if element_pair is None:
        # Total PDF
        for site_neighbors in all_neighbors:
            for neighbor in site_neighbors:
                dist = neighbor.nn_distance
                if 0 < dist < r_max:
                    bin_idx = int(dist / bin_width)
                    if 0 <= bin_idx < n_bins:
                        hist[bin_idx] += 1

        rho = n_atoms / volume
        shell_volumes = 4.0 * np.pi * r_centers**2 * bin_width
        g_r = hist / (n_atoms * rho * shell_volumes)

    else:
        el1, el2 = element_pair
        idx1 = [i for i, s in enumerate(structure) if str(s.specie) == el1]
        idx2 = [i for i, s in enumerate(structure) if str(s.specie) == el2]

        for i in idx1:
            for neighbor in all_neighbors[i]:
                if str(neighbor.specie) == el2:
                    dist = neighbor.nn_distance
                    if 0 < dist < r_max:
                        bin_idx = int(dist / bin_width)
                        if 0 <= bin_idx < n_bins:
                            hist[bin_idx] += 1

        n1 = len(idx1)
        n2 = len(idx2)
        rho2 = n2 / volume
        shell_volumes = 4.0 * np.pi * r_centers**2 * bin_width

        if n1 > 0 and rho2 > 0:
            g_r = hist / (n1 * rho2 * shell_volumes)
        else:
            g_r = np.zeros(n_bins)

    return r_centers, g_r
```

### 4. Compare crystalline vs amorphous structures

```python
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pymatgen.core import Structure

# Assuming compute_pdf() is defined as above

# --- Load crystalline and amorphous structures ---
crystalline = Structure.from_file("SiO2_quartz.cif")
amorphous = Structure.from_file("SiO2_glass.cif")  # e.g., from MD quench

r_cryst, g_cryst = compute_pdf(crystalline, r_max=12.0, bin_width=0.05)
r_amorph, g_amorph = compute_pdf(amorphous, r_max=12.0, bin_width=0.05)

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

ax1.plot(r_cryst, g_cryst, "b-", linewidth=1.5)
ax1.axhline(y=1.0, color="gray", linestyle="--", alpha=0.5)
ax1.set_ylabel("g(r)", fontsize=12)
ax1.set_title("Crystalline SiO2 (quartz)")
ax1.set_ylim(0, None)

ax2.plot(r_amorph, g_amorph, "r-", linewidth=1.5)
ax2.axhline(y=1.0, color="gray", linestyle="--", alpha=0.5)
ax2.set_xlabel("r (Angstrom)", fontsize=12)
ax2.set_ylabel("g(r)", fontsize=12)
ax2.set_title("Amorphous SiO2 (glass)")
ax2.set_ylim(0, None)

plt.tight_layout()
plt.savefig("pdf_cryst_vs_amorph.png", dpi=150)
plt.close()
print("Saved: pdf_cryst_vs_amorph.png")
```

### 5. Time-averaged PDF from an MD trajectory

```python
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from ase.io import read as ase_read
from ase.io.trajectory import Trajectory
from pymatgen.io.ase import AseAtomsAdaptor

# Assuming compute_pdf() is defined as above


def pdf_from_trajectory(traj_file, r_max=10.0, bin_width=0.05,
                         skip_frames=0, stride=1):
    """
    Compute time-averaged PDF from an ASE trajectory.

    Args:
        traj_file: path to ASE .traj file or any ASE-readable trajectory
        r_max: maximum r in Angstrom
        bin_width: bin width in Angstrom
        skip_frames: number of initial frames to skip (equilibration)
        stride: sample every Nth frame

    Returns:
        r_centers, g_r_avg
    """
    # Read trajectory
    try:
        traj = Trajectory(traj_file)
    except Exception:
        traj = ase_read(traj_file, index=":")

    adaptor = AseAtomsAdaptor()
    n_bins = int(r_max / bin_width)
    g_r_sum = np.zeros(n_bins)
    n_frames = 0

    frames = list(traj)[skip_frames::stride]
    print(f"Processing {len(frames)} frames...")

    for atoms in frames:
        structure = adaptor.get_structure(atoms)
        r_centers, g_r = compute_pdf(structure, r_max=r_max, bin_width=bin_width)
        g_r_sum += g_r
        n_frames += 1

    g_r_avg = g_r_sum / n_frames
    print(f"Averaged over {n_frames} frames")

    return r_centers, g_r_avg


# --- Usage ---
r, g = pdf_from_trajectory(
    "md_trajectory.traj",
    r_max=10.0,
    bin_width=0.05,
    skip_frames=100,  # skip first 100 frames (equilibration)
    stride=5,         # use every 5th frame
)

fig, ax = plt.subplots(figsize=(10, 5))
ax.plot(r, g, "b-", linewidth=1.5)
ax.axhline(y=1.0, color="gray", linestyle="--", alpha=0.5)
ax.set_xlabel("r (Angstrom)", fontsize=12)
ax.set_ylabel("g(r)", fontsize=12)
ax.set_title("Time-Averaged PDF from MD Trajectory")
ax.set_xlim(0, 10)
ax.set_ylim(0, None)
plt.tight_layout()
plt.savefig("pdf_md_averaged.png", dpi=150)
plt.close()
print("Saved: pdf_md_averaged.png")
```

### 6. PDF from MD trajectory with element-specific partials

```python
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from ase.io.trajectory import Trajectory
from pymatgen.io.ase import AseAtomsAdaptor
from itertools import combinations_with_replacement

# Assuming compute_pdf() is defined as above


def partial_pdf_from_trajectory(traj_file, r_max=10.0, bin_width=0.05,
                                 skip_frames=0, stride=1):
    """
    Compute time-averaged partial PDFs from an MD trajectory.

    Returns:
        r_centers, dict of {pair_label: g_r_avg}
    """
    try:
        traj = Trajectory(traj_file)
    except Exception:
        from ase.io import read as ase_read
        traj = ase_read(traj_file, index=":")

    adaptor = AseAtomsAdaptor()
    frames = list(traj)[skip_frames::stride]

    # Determine elements from first frame
    first_struct = adaptor.get_structure(frames[0])
    elements = sorted(set(str(s.specie) for s in first_struct))
    pairs = list(combinations_with_replacement(elements, 2))

    n_bins = int(r_max / bin_width)
    partial_sums = {f"{e1}-{e2}": np.zeros(n_bins) for e1, e2 in pairs}
    n_frames = 0

    for atoms in frames:
        structure = adaptor.get_structure(atoms)
        for e1, e2 in pairs:
            r_centers, g_partial = compute_pdf(
                structure, r_max=r_max, bin_width=bin_width,
                element_pair=(e1, e2),
            )
            partial_sums[f"{e1}-{e2}"] += g_partial
        n_frames += 1

    partial_avgs = {k: v / n_frames for k, v in partial_sums.items()}
    return r_centers, partial_avgs


# --- Usage ---
r, partials = partial_pdf_from_trajectory(
    "md_trajectory.traj",
    r_max=8.0,
    bin_width=0.05,
    skip_frames=100,
    stride=5,
)

fig, ax = plt.subplots(figsize=(10, 6))
for pair_label, g_partial in partials.items():
    ax.plot(r, g_partial, label=pair_label, linewidth=1.5)
ax.axhline(y=1.0, color="gray", linestyle="--", alpha=0.5)
ax.set_xlabel("r (Angstrom)", fontsize=12)
ax.set_ylabel("g(r)", fontsize=12)
ax.set_title("Time-Averaged Partial PDFs from MD")
ax.legend()
ax.set_xlim(0, 8)
ax.set_ylim(0, None)
plt.tight_layout()
plt.savefig("pdf_md_partials.png", dpi=150)
plt.close()
print("Saved: pdf_md_partials.png")
```

## Key Parameters

| Parameter | Description |
|-----------|-------------|
| `r_max` | Maximum distance to compute g(r). Typically 10-20 A for crystals. Must be less than half the smallest supercell dimension to avoid self-interaction artifacts. |
| `bin_width` | Histogram bin width. 0.01-0.05 A for fine resolution, 0.1 A for smoother curves. Smaller bins give sharper peaks but noisier data. |
| `element_pair` | Tuple `(el1, el2)` for partial PDF. `None` for total PDF. |
| `skip_frames` | Number of initial MD frames to discard (equilibration period). |
| `stride` | Sample every Nth frame to reduce computation. Ensure enough frames for statistics. |

## Interpreting Results

- **Sharp peaks**: Indicate well-defined interatomic distances. Crystalline materials show sharp peaks at each coordination shell.
- **Broad peaks**: Indicate disorder or a distribution of bond lengths. Amorphous materials show broad first peaks that decay toward g(r) = 1.
- **First peak position**: Nearest-neighbor distance. Compare with known bond lengths (e.g., Si-O ~ 1.61 A in silica).
- **Peak area**: Related to the coordination number. Integrate the first peak: CN = 4*pi*rho * integral(r^2 * g(r) dr) over the first shell.
- **g(r) = 1 at large r**: Indicates the structure appears homogeneous at long range. If g(r) does not approach 1, the structure may be too small or the normalization may be wrong.
- **g(r) = 0 at small r**: Expected due to atomic repulsion. If g(r) > 0 at very small r, there may be overlapping atoms (unphysical structure).
- **Partial PDFs**: Reveal element-specific bonding. E.g., in a perovskite ABO3, the A-O and B-O partial PDFs show distinct bond lengths.
- **Crystalline vs amorphous**: Crystals show sharp peaks that persist to large r. Glasses show a sharp first peak, broadened second peak, and rapid decay to g(r) = 1.

## Common Issues

| Issue | Solution |
|-------|----------|
| g(r) does not approach 1 at large r | The cell may be too small. Use a supercell. Also check that r_max < L/2 where L is the smallest cell dimension. |
| Noisy g(r) | Use smaller bin_width or smooth with a Gaussian filter. For MD, average over more frames. |
| Peaks at unphysical distances | Check for overlapping atoms in the structure. Verify the structure loaded correctly. |
| Supercell artifacts (peaks at L/2) | r_max must be less than half the smallest supercell dimension. Build a larger supercell using `structure.make_supercell([n,n,n])`. |
| Partial PDF has zero everywhere | Check element symbols match exactly (case-sensitive). Print `set(str(s.specie) for s in structure)` to verify. |
| MD trajectory too large to process | Increase `stride` to skip frames. Use `skip_frames` to discard equilibration. |
| Memory error with large structures | Reduce `r_max` or use `structure.get_neighbors(site, r_max)` site by site instead of `get_all_neighbors`. |
