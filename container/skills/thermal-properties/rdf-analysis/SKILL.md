# Radial Distribution Function (RDF) and Structural Analysis from MD

## When to Use

- Computing pair correlation functions g(r) from MD trajectories
- Analyzing partial (element-pair-specific) RDFs g_AB(r) in multi-component systems
- Determining coordination numbers by integrating the first peak of g(r)
- Extracting bond length distributions from MD or relaxed structures
- Computing bond angle distributions for local geometry analysis
- Characterizing crystalline vs amorphous vs liquid structural order
- Equivalent to VASPKIT functions 725 (RDF), 726 (partial RDF), 730 (bond length distribution), 731 (bond angle distribution)

## Method Selection

```
What structural property do you need?

Total RDF g(r)?
  --> Section A: Total pair correlation function

Partial / element-pair RDF g_AB(r)?
  --> Section B: Partial RDFs (e.g., Li-O, Zr-O separately)

Coordination number?
  --> Section C: Integrate RDF to first minimum

Bond length distribution?
  --> Section D: Histogram of specific bond lengths

Bond angle distribution?
  --> Section E: Angle distribution for A-B-C triplets

What trajectory format?
  ASE .traj  --> Use ASE Trajectory reader
  LAMMPS dump --> Use ASE read() with format="lammps-dump-text"
  VASP XDATCAR --> Use pymatgen Xdatcar parser
```

## Prerequisites

Pre-installed: `ase`, `pymatgen`, `numpy`, `scipy`, `matplotlib`.

No additional packages required. ASE's `neighborlist` module is used for efficient neighbor searches.

## Detailed Steps

### Complete RDF, Coordination Number, Bond Length, and Bond Angle Analysis

```python
#!/usr/bin/env python3
"""
Radial Distribution Function (RDF), coordination number, bond length distribution,
and bond angle distribution from MD trajectories.
Supports ASE .traj, LAMMPS dump, and VASP XDATCAR formats.
Computes total and partial (element-pair) RDFs.
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from collections import defaultdict
from itertools import combinations_with_replacement
import os

# ============================================================
# 1. CONFIGURATION
# ============================================================

# --- Trajectory source ---
TRAJ_FORMAT = "ase"          # "ase", "lammps", or "xdatcar"
TRAJ_FILE = "md.traj"        # Path to trajectory file

# --- Frame selection ---
N_EQUIL_FRAMES = 50          # Skip equilibration frames
FRAME_STEP = 1               # Use every Nth production frame (subsample for speed)

# --- RDF settings ---
R_MAX = 10.0                 # Maximum radius in Angstrom
N_BINS = 500                 # Number of histogram bins (dr = R_MAX / N_BINS)
SMOOTH_WINDOW = 5            # Savitzky-Golay smoothing window (odd integer, 0 = no smoothing)

# --- Coordination number ---
# Cutoffs for coordination number. If empty, auto-detect from first RDF minimum.
COORD_CUTOFFS = {}           # e.g., {"Li-O": 3.0, "Zr-O": 2.8}

# --- Bond angle distribution ---
ANGLE_TRIPLETS = []          # e.g., [("O", "Si", "O"), ("Si", "O", "Si")]
ANGLE_CUTOFF = 3.5           # Max bond length for angle analysis (A)
ANGLE_N_BINS = 180           # Number of bins for angle histogram

OUTPUT_DIR = "rdf_analysis"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ============================================================
# 2. LOAD TRAJECTORY
# ============================================================

def load_trajectory(traj_format, traj_file):
    """Load trajectory from various formats."""
    if traj_format == "ase":
        from ase.io import Trajectory
        traj = Trajectory(traj_file, "r")
        frames = [atoms.copy() for atoms in traj]
        traj.close()
    elif traj_format == "lammps":
        from ase.io import read
        frames = read(traj_file, index=":", format="lammps-dump-text")
    elif traj_format == "xdatcar":
        from pymatgen.io.vasp import Xdatcar
        xdatcar = Xdatcar(traj_file)
        from pymatgen.io.ase import AseAtomsAdaptor
        adaptor = AseAtomsAdaptor()
        frames = [adaptor.get_atoms(s) for s in xdatcar.structures]
    else:
        raise ValueError(f"Unknown format: {traj_format}")
    return frames


print(f"Loading trajectory: {TRAJ_FILE} (format: {TRAJ_FORMAT})")
all_frames = load_trajectory(TRAJ_FORMAT, TRAJ_FILE)
print(f"  Total frames: {len(all_frames)}")
print(f"  Atoms per frame: {len(all_frames[0])}")
print(f"  Formula: {all_frames[0].get_chemical_formula()}")

# Select production frames
frames = all_frames[N_EQUIL_FRAMES::FRAME_STEP]
n_frames = len(frames)
print(f"  Production frames used: {n_frames}")

symbols = frames[0].get_chemical_symbols()
unique_elements = sorted(set(symbols))
n_atoms = len(symbols)
print(f"  Elements: {unique_elements}")

# ============================================================
# 3. SECTION A: TOTAL RDF g(r)
# ============================================================

print("\n=== Section A: Total RDF ===")


def compute_total_rdf(frames, r_max, n_bins):
    """
    Compute the total pair radial distribution function.
    Uses ASE neighbor lists for efficiency.
    """
    from ase.neighborlist import neighbor_list

    dr = r_max / n_bins
    r_edges = np.linspace(0, r_max, n_bins + 1)
    r_centers = 0.5 * (r_edges[:-1] + r_edges[1:])
    hist = np.zeros(n_bins)

    for atoms in frames:
        n = len(atoms)
        volume = atoms.get_volume()
        rho = n / volume  # number density

        # Get all pairs within r_max
        _, _, distances = neighbor_list("ijd", atoms, cutoff=r_max)
        h, _ = np.histogram(distances, bins=r_edges)
        hist += h

    # Normalize
    # Each pair (i,j) is counted once by neighbor_list (j>i not enforced, so each pair appears twice: i->j and j->i)
    # Actually, ASE neighbor_list returns both (i,j) and (j,i), so total counts = 2 * n_pairs
    # g(r) = hist / (n_frames * n_atoms * rho * 4*pi*r^2*dr)
    # The factor of 2 in the hist accounts for both directions, matching the
    # normalization where we sum over all atoms i and all neighbors j.
    avg_volume = np.mean([f.get_volume() for f in frames])
    avg_rho = n_atoms / avg_volume

    shell_vol = 4.0 * np.pi * r_centers ** 2 * dr
    norm = n_frames * n_atoms * avg_rho * shell_vol

    g_r = hist / norm

    return r_centers, g_r


r, g_r = compute_total_rdf(frames, R_MAX, N_BINS)

# Optional smoothing
if SMOOTH_WINDOW > 2:
    from scipy.signal import savgol_filter
    g_r_smooth = savgol_filter(g_r, SMOOTH_WINDOW, 2)
else:
    g_r_smooth = g_r

# Find peaks
from scipy.signal import find_peaks
peaks, properties = find_peaks(g_r_smooth, height=1.05, distance=N_BINS // 50, prominence=0.1)

print(f"  RDF computed with dr = {R_MAX / N_BINS:.4f} A")
print(f"  Peak positions:")
for p in peaks[:8]:
    print(f"    r = {r[p]:.3f} A, g(r) = {g_r_smooth[p]:.3f}")

# Save data
np.savetxt(os.path.join(OUTPUT_DIR, "rdf_total.dat"),
           np.column_stack([r, g_r, g_r_smooth]),
           header="r(A)  g(r)  g(r)_smoothed",
           fmt="%10.5f %12.6f %12.6f")

# Plot
fig, ax = plt.subplots(figsize=(8, 5))
ax.plot(r, g_r, "b-", linewidth=0.5, alpha=0.4, label="Raw")
ax.plot(r, g_r_smooth, "b-", linewidth=1.5, label="Smoothed")
ax.axhline(y=1.0, color="gray", linestyle="--", linewidth=0.5)

# Label peaks
for p in peaks[:6]:
    ax.annotate(f"{r[p]:.2f} A",
                xy=(r[p], g_r_smooth[p]),
                xytext=(r[p] + 0.2, g_r_smooth[p] + 0.15),
                fontsize=9, color="red",
                arrowprops=dict(arrowstyle="-", color="red", lw=0.8))

ax.set_xlabel("r ($\\AA$)", fontsize=12)
ax.set_ylabel("g(r)", fontsize=12)
ax.set_title("Total Radial Distribution Function", fontsize=13)
ax.set_xlim(0, R_MAX)
ax.set_ylim(bottom=0)
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3)
fig.tight_layout()
fig.savefig(os.path.join(OUTPUT_DIR, "rdf_total.png"), dpi=150, bbox_inches="tight")
plt.close()
print(f"  Saved: {OUTPUT_DIR}/rdf_total.png")

# ============================================================
# 4. SECTION B: PARTIAL RDFs g_AB(r)
# ============================================================

print("\n=== Section B: Partial RDFs ===")


def compute_partial_rdf(frames, element_a, element_b, r_max, n_bins):
    """
    Compute the partial RDF g_AB(r) for element pair (A, B).
    Normalization accounts for the number density of B atoms.
    """
    from ase.neighborlist import neighbor_list

    dr = r_max / n_bins
    r_edges = np.linspace(0, r_max, n_bins + 1)
    r_centers = 0.5 * (r_edges[:-1] + r_edges[1:])
    hist = np.zeros(n_bins)

    for atoms in frames:
        syms = atoms.get_chemical_symbols()
        n_total = len(atoms)
        volume = atoms.get_volume()

        # Indices of A and B atoms
        idx_a = [i for i, s in enumerate(syms) if s == element_a]
        idx_b = [i for i, s in enumerate(syms) if s == element_b]

        n_a = len(idx_a)
        n_b = len(idx_b)

        if n_a == 0 or n_b == 0:
            continue

        # Number density of B atoms
        rho_b = n_b / volume

        # Get neighbor list
        i_arr, j_arr, d_arr = neighbor_list("ijd", atoms, cutoff=r_max)

        # Filter for A-B pairs
        set_a = set(idx_a)
        set_b = set(idx_b)

        for ii, jj, dd in zip(i_arr, j_arr, d_arr):
            if ii in set_a and jj in set_b:
                bin_idx = int(dd / dr)
                if 0 <= bin_idx < n_bins:
                    hist[bin_idx] += 1

    # Normalize: g_AB(r) = hist / (n_frames * n_A * rho_B * 4*pi*r^2*dr)
    avg_volume = np.mean([f.get_volume() for f in frames])
    n_a = len([s for s in symbols if s == element_a])
    n_b = len([s for s in symbols if s == element_b])
    rho_b = n_b / avg_volume

    shell_vol = 4.0 * np.pi * r_centers ** 2 * dr
    norm = n_frames * n_a * rho_b * shell_vol

    # Avoid division by zero at r=0
    norm[norm < 1e-30] = 1e-30
    g_ab = hist / norm

    return r_centers, g_ab


# Compute all element pairs
element_pairs = list(combinations_with_replacement(unique_elements, 2))
partial_rdfs = {}

for el_a, el_b in element_pairs:
    r_ab, g_ab = compute_partial_rdf(frames, el_a, el_b, R_MAX, N_BINS)
    pair_label = f"{el_a}-{el_b}"
    partial_rdfs[pair_label] = (r_ab, g_ab)

    # Smooth
    if SMOOTH_WINDOW > 2:
        from scipy.signal import savgol_filter
        g_ab_smooth = savgol_filter(g_ab, SMOOTH_WINDOW, 2)
    else:
        g_ab_smooth = g_ab

    # Find peaks
    peaks_ab, _ = find_peaks(g_ab_smooth, height=0.5, distance=N_BINS // 50, prominence=0.05)

    n_a = len([s for s in symbols if s == el_a])
    n_b = len([s for s in symbols if s == el_b])
    print(f"  {pair_label} (N_{el_a}={n_a}, N_{el_b}={n_b}):")
    for p in peaks_ab[:4]:
        print(f"    r = {r_ab[p]:.3f} A, g(r) = {g_ab_smooth[p]:.3f}")

    # Save data
    np.savetxt(os.path.join(OUTPUT_DIR, f"rdf_{el_a}_{el_b}.dat"),
               np.column_stack([r_ab, g_ab, g_ab_smooth]),
               header=f"r(A)  g_{el_a}{el_b}(r)  g_{el_a}{el_b}(r)_smoothed",
               fmt="%10.5f %12.6f %12.6f")

# Plot all partial RDFs
n_pairs = len(element_pairs)
n_cols = min(3, n_pairs)
n_rows = (n_pairs + n_cols - 1) // n_cols

fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 4 * n_rows), squeeze=False)

colors = plt.cm.Set1(np.linspace(0, 1, n_pairs))

for idx, (pair_label, (r_ab, g_ab)) in enumerate(partial_rdfs.items()):
    row = idx // n_cols
    col = idx % n_cols
    ax = axes[row, col]

    if SMOOTH_WINDOW > 2:
        g_ab_smooth = savgol_filter(g_ab, SMOOTH_WINDOW, 2)
    else:
        g_ab_smooth = g_ab

    ax.plot(r_ab, g_ab_smooth, "-", color=colors[idx], linewidth=1.5)
    ax.axhline(y=1.0, color="gray", linestyle="--", linewidth=0.5)
    ax.set_xlabel("r ($\\AA$)")
    ax.set_ylabel(f"g$_{{{pair_label}}}$(r)")
    ax.set_title(f"Partial RDF: {pair_label}")
    ax.set_xlim(0, R_MAX)
    ax.set_ylim(bottom=0)
    ax.grid(True, alpha=0.3)

    # Label first peak
    peaks_ab, _ = find_peaks(g_ab_smooth, height=0.5, distance=N_BINS // 50, prominence=0.05)
    if len(peaks_ab) > 0:
        p = peaks_ab[0]
        ax.annotate(f"{r_ab[p]:.2f}", xy=(r_ab[p], g_ab_smooth[p]),
                    fontsize=9, color="red", ha="center",
                    xytext=(r_ab[p], g_ab_smooth[p] * 1.08))

# Hide unused subplots
for idx in range(n_pairs, n_rows * n_cols):
    axes[idx // n_cols, idx % n_cols].set_visible(False)

fig.suptitle("Partial Radial Distribution Functions", fontsize=14, y=1.01)
fig.tight_layout()
fig.savefig(os.path.join(OUTPUT_DIR, "rdf_partial_all.png"), dpi=150, bbox_inches="tight")
plt.close()
print(f"  Saved: {OUTPUT_DIR}/rdf_partial_all.png")

# Combined overlay plot
fig, ax = plt.subplots(figsize=(8, 5))
for idx, (pair_label, (r_ab, g_ab)) in enumerate(partial_rdfs.items()):
    if SMOOTH_WINDOW > 2:
        g_ab_smooth = savgol_filter(g_ab, SMOOTH_WINDOW, 2)
    else:
        g_ab_smooth = g_ab
    ax.plot(r_ab, g_ab_smooth, "-", color=colors[idx], linewidth=1.5, label=pair_label)

ax.axhline(y=1.0, color="gray", linestyle="--", linewidth=0.5)
ax.set_xlabel("r ($\\AA$)", fontsize=12)
ax.set_ylabel("g(r)", fontsize=12)
ax.set_title("All Partial RDFs", fontsize=13)
ax.set_xlim(0, R_MAX)
ax.set_ylim(bottom=0)
ax.legend(fontsize=9, ncol=2)
ax.grid(True, alpha=0.3)
fig.tight_layout()
fig.savefig(os.path.join(OUTPUT_DIR, "rdf_partial_overlay.png"), dpi=150, bbox_inches="tight")
plt.close()
print(f"  Saved: {OUTPUT_DIR}/rdf_partial_overlay.png")

# ============================================================
# 5. SECTION C: COORDINATION NUMBER
# ============================================================

print("\n=== Section C: Coordination Numbers ===")


def compute_coordination_number(r, g_r, rho_b, r_cutoff=None):
    """
    Compute coordination number by integrating 4*pi*rho_B*r^2*g(r)*dr
    up to the first minimum (or specified cutoff).
    Returns coordination number and the cutoff used.
    """
    dr = r[1] - r[0]

    if r_cutoff is None:
        # Auto-detect: find first minimum after first peak
        if SMOOTH_WINDOW > 2:
            from scipy.signal import savgol_filter
            g_smooth = savgol_filter(g_r, SMOOTH_WINDOW, 2)
        else:
            g_smooth = g_r

        peaks, _ = find_peaks(g_smooth, height=1.1, distance=10)
        if len(peaks) > 0:
            first_peak = peaks[0]
            # Find minimum after first peak
            search_region = g_smooth[first_peak:]
            minima, _ = find_peaks(-search_region, distance=10)
            if len(minima) > 0:
                r_cutoff = r[first_peak + minima[0]]
            else:
                r_cutoff = r[first_peak] * 1.5
        else:
            r_cutoff = 3.0  # Fallback

    # Integrate
    mask = r <= r_cutoff
    integrand = 4.0 * np.pi * rho_b * r[mask] ** 2 * g_r[mask] * dr
    cn = np.sum(integrand)

    return cn, r_cutoff


# Running coordination number for all pairs
for pair_label, (r_ab, g_ab) in partial_rdfs.items():
    el_a, el_b = pair_label.split("-")
    n_b = len([s for s in symbols if s == el_b])
    avg_volume = np.mean([f.get_volume() for f in frames])
    rho_b = n_b / avg_volume

    # Auto or manual cutoff
    if pair_label in COORD_CUTOFFS:
        cutoff = COORD_CUTOFFS[pair_label]
    else:
        cutoff = None

    cn, r_cut_used = compute_coordination_number(r_ab, g_ab, rho_b, r_cutoff=cutoff)
    print(f"  {pair_label}: CN = {cn:.2f} (cutoff = {r_cut_used:.2f} A)")

# Plot running coordination number for each pair
fig, ax = plt.subplots(figsize=(8, 5))

for idx, (pair_label, (r_ab, g_ab)) in enumerate(partial_rdfs.items()):
    el_a, el_b = pair_label.split("-")
    n_b = len([s for s in symbols if s == el_b])
    avg_volume = np.mean([f.get_volume() for f in frames])
    rho_b = n_b / avg_volume

    dr = r_ab[1] - r_ab[0]
    running_cn = np.cumsum(4.0 * np.pi * rho_b * r_ab ** 2 * g_ab * dr)
    ax.plot(r_ab, running_cn, "-", color=colors[idx], linewidth=1.5, label=pair_label)

ax.set_xlabel("r ($\\AA$)", fontsize=12)
ax.set_ylabel("Cumulative Coordination Number", fontsize=12)
ax.set_title("Running Coordination Number", fontsize=13)
ax.set_xlim(0, R_MAX)
ax.legend(fontsize=9, ncol=2)
ax.grid(True, alpha=0.3)
fig.tight_layout()
fig.savefig(os.path.join(OUTPUT_DIR, "coordination_number.png"), dpi=150, bbox_inches="tight")
plt.close()
print(f"  Saved: {OUTPUT_DIR}/coordination_number.png")

# ============================================================
# 6. SECTION D: BOND LENGTH DISTRIBUTION
# ============================================================

print("\n=== Section D: Bond Length Distribution ===")


def compute_bond_length_distribution(frames, element_a, element_b, r_min, r_max, n_bins=200):
    """
    Compute bond length distribution (histogram of distances) for A-B pairs
    within a specified distance range.
    Unlike RDF, this is NOT normalized by the ideal gas shell volume.
    """
    from ase.neighborlist import neighbor_list

    r_edges = np.linspace(r_min, r_max, n_bins + 1)
    r_centers = 0.5 * (r_edges[:-1] + r_edges[1:])
    hist = np.zeros(n_bins)

    for atoms in frames:
        syms = atoms.get_chemical_symbols()
        idx_a = set(i for i, s in enumerate(syms) if s == element_a)
        idx_b = set(i for i, s in enumerate(syms) if s == element_b)

        i_arr, j_arr, d_arr = neighbor_list("ijd", atoms, cutoff=r_max)

        for ii, jj, dd in zip(i_arr, j_arr, d_arr):
            if dd >= r_min:
                if (ii in idx_a and jj in idx_b):
                    bin_idx = int((dd - r_min) / (r_max - r_min) * n_bins)
                    if 0 <= bin_idx < n_bins:
                        hist[bin_idx] += 1

    # Normalize to probability density
    dr = r_centers[1] - r_centers[0]
    total = np.sum(hist) * dr
    if total > 0:
        prob = hist / total
    else:
        prob = hist

    return r_centers, prob, hist


# Auto-detect bond pairs from partial RDF peaks
bond_pairs_detected = []
for pair_label, (r_ab, g_ab) in partial_rdfs.items():
    if SMOOTH_WINDOW > 2:
        from scipy.signal import savgol_filter
        g_smooth = savgol_filter(g_ab, SMOOTH_WINDOW, 2)
    else:
        g_smooth = g_ab

    peaks_ab, props = find_peaks(g_smooth, height=1.5, distance=N_BINS // 50, prominence=0.2)
    if len(peaks_ab) > 0:
        first_peak_r = r_ab[peaks_ab[0]]
        if first_peak_r < 4.0:  # Only consider short-range bonds
            el_a, el_b = pair_label.split("-")
            r_lo = max(first_peak_r - 0.8, 0.5)
            r_hi = first_peak_r + 0.8
            bond_pairs_detected.append((el_a, el_b, r_lo, r_hi, first_peak_r))

if bond_pairs_detected:
    n_bonds = len(bond_pairs_detected)
    fig, axes = plt.subplots(1, min(n_bonds, 4), figsize=(5 * min(n_bonds, 4), 4), squeeze=False)

    for idx, (el_a, el_b, r_lo, r_hi, peak_r) in enumerate(bond_pairs_detected[:4]):
        r_bl, prob_bl, _ = compute_bond_length_distribution(
            frames, el_a, el_b, r_lo, r_hi, n_bins=100
        )
        ax = axes[0, idx]
        ax.fill_between(r_bl, prob_bl, alpha=0.4, color="steelblue")
        ax.plot(r_bl, prob_bl, "b-", linewidth=1.5)
        ax.axvline(x=peak_r, color="red", linestyle="--", linewidth=1,
                   label=f"RDF peak: {peak_r:.2f} $\\AA$")
        ax.set_xlabel("Bond Length ($\\AA$)", fontsize=11)
        ax.set_ylabel("Probability Density", fontsize=11)
        ax.set_title(f"{el_a}-{el_b} Bond Length", fontsize=12)
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)

        # Mean and std of distribution
        mean_bl = np.sum(r_bl * prob_bl) / np.sum(prob_bl) if np.sum(prob_bl) > 0 else 0
        std_bl = np.sqrt(np.sum((r_bl - mean_bl) ** 2 * prob_bl) / np.sum(prob_bl)) if np.sum(prob_bl) > 0 else 0
        print(f"  {el_a}-{el_b}: mean = {mean_bl:.3f} A, std = {std_bl:.3f} A")

        np.savetxt(os.path.join(OUTPUT_DIR, f"bond_length_{el_a}_{el_b}.dat"),
                   np.column_stack([r_bl, prob_bl]),
                   header=f"r(A)  probability_density",
                   fmt="%10.5f %12.6f")

    fig.tight_layout()
    fig.savefig(os.path.join(OUTPUT_DIR, "bond_length_distributions.png"), dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {OUTPUT_DIR}/bond_length_distributions.png")
else:
    print("  No short-range bonds detected (no RDF peaks < 4 A with g(r) > 1.5)")

# ============================================================
# 7. SECTION E: BOND ANGLE DISTRIBUTION
# ============================================================

print("\n=== Section E: Bond Angle Distribution ===")


def compute_bond_angle_distribution(frames, el_a, el_center, el_c, cutoff, n_bins=180):
    """
    Compute the bond angle distribution for A-Center-C triplets.
    Considers all A and C atoms bonded to Center within cutoff.
    Angle is measured at the center atom.
    """
    from ase.neighborlist import neighbor_list

    angles_all = []

    for atoms in frames:
        syms = atoms.get_chemical_symbols()
        positions = atoms.get_positions()
        cell = atoms.get_cell()

        # Get neighbor list
        i_arr, j_arr, d_arr, D_arr = neighbor_list("ijdD", atoms, cutoff=cutoff)

        # Group neighbors by center atom
        neighbors = defaultdict(list)
        for ii, jj, dd, disp in zip(i_arr, j_arr, d_arr, D_arr):
            neighbors[ii].append((jj, disp))

        # For each center atom of type el_center
        for center_idx in range(len(syms)):
            if syms[center_idx] != el_center:
                continue

            # Find A-type and C-type neighbors
            neigh_a = []
            neigh_c = []

            for jj, disp in neighbors[center_idx]:
                if syms[jj] == el_a:
                    neigh_a.append(disp)
                if syms[jj] == el_c:
                    neigh_c.append(disp)

            # Compute angles for all A-Center-C combinations
            if el_a == el_c:
                # Same element: avoid double-counting
                for i in range(len(neigh_a)):
                    for j in range(i + 1, len(neigh_a)):
                        v1 = neigh_a[i]
                        v2 = neigh_a[j]
                        cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
                        cos_angle = np.clip(cos_angle, -1.0, 1.0)
                        angle_deg = np.degrees(np.arccos(cos_angle))
                        angles_all.append(angle_deg)
            else:
                for v1 in neigh_a:
                    for v2 in neigh_c:
                        cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
                        cos_angle = np.clip(cos_angle, -1.0, 1.0)
                        angle_deg = np.degrees(np.arccos(cos_angle))
                        angles_all.append(angle_deg)

    if len(angles_all) == 0:
        return np.linspace(0, 180, n_bins), np.zeros(n_bins), 0

    angles_all = np.array(angles_all)
    hist, edges = np.histogram(angles_all, bins=n_bins, range=(0, 180), density=True)
    centers = 0.5 * (edges[:-1] + edges[1:])

    return centers, hist, len(angles_all)


# Auto-detect angle triplets if not specified
if not ANGLE_TRIPLETS and bond_pairs_detected:
    # Generate triplets from detected bonds
    bonded_to = defaultdict(set)
    for el_a, el_b, _, _, _ in bond_pairs_detected:
        bonded_to[el_b].add(el_a)
        bonded_to[el_a].add(el_b)

    for center in unique_elements:
        bonded = sorted(bonded_to.get(center, set()))
        for pair in combinations_with_replacement(bonded, 2):
            triplet = (pair[0], center, pair[1])
            ANGLE_TRIPLETS.append(triplet)

    if ANGLE_TRIPLETS:
        print(f"  Auto-detected triplets: {ANGLE_TRIPLETS}")

if ANGLE_TRIPLETS:
    n_triplets = len(ANGLE_TRIPLETS)
    n_cols = min(3, n_triplets)
    n_rows = (n_triplets + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 4 * n_rows), squeeze=False)

    for idx, (el_a, el_center, el_c) in enumerate(ANGLE_TRIPLETS):
        angle_centers, angle_hist, n_angles = compute_bond_angle_distribution(
            frames, el_a, el_center, el_c, ANGLE_CUTOFF, ANGLE_N_BINS
        )

        row = idx // n_cols
        col = idx % n_cols
        ax = axes[row, col]

        label = f"{el_a}-{el_center}-{el_c}"
        ax.fill_between(angle_centers, angle_hist, alpha=0.4, color="coral")
        ax.plot(angle_centers, angle_hist, "r-", linewidth=1.5)
        ax.set_xlabel("Angle (degrees)", fontsize=11)
        ax.set_ylabel("Probability Density", fontsize=11)
        ax.set_title(f"{label} (n={n_angles})", fontsize=12)
        ax.set_xlim(0, 180)
        ax.grid(True, alpha=0.3)

        # Find peak angle
        if n_angles > 0:
            peak_idx = np.argmax(angle_hist)
            peak_angle = angle_centers[peak_idx]
            ax.axvline(x=peak_angle, color="darkred", linestyle="--", linewidth=1,
                       label=f"Peak: {peak_angle:.1f} deg")
            ax.legend(fontsize=9)
            print(f"  {label}: peak at {peak_angle:.1f} deg, n_angles = {n_angles}")

            # Reference angles
            # Tetrahedral: 109.47, Octahedral: 90/180, Linear: 180
            for ref_angle, ref_label in [(109.47, "Td"), (90, "Oh"), (120, "trig")]:
                if abs(peak_angle - ref_angle) < 15:
                    print(f"    Close to {ref_label} angle ({ref_angle:.1f} deg)")

        np.savetxt(os.path.join(OUTPUT_DIR, f"angle_{el_a}_{el_center}_{el_c}.dat"),
                   np.column_stack([angle_centers, angle_hist]),
                   header=f"angle(deg)  probability_density",
                   fmt="%8.2f %12.6f")

    for idx in range(n_triplets, n_rows * n_cols):
        axes[idx // n_cols, idx % n_cols].set_visible(False)

    fig.suptitle("Bond Angle Distributions", fontsize=14, y=1.01)
    fig.tight_layout()
    fig.savefig(os.path.join(OUTPUT_DIR, "bond_angle_distributions.png"), dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {OUTPUT_DIR}/bond_angle_distributions.png")
else:
    print("  No angle triplets specified or detected.")

print(f"\n=== RDF Analysis Complete ===")
print(f"All outputs in: {OUTPUT_DIR}/")
```

## Key Parameters

| Parameter | Default | Notes |
|---|---|---|
| `R_MAX` | 10.0 A | Maximum radius for RDF. Should be < half the smallest cell dimension. |
| `N_BINS` | 500 | Histogram bins. dr = R_MAX/N_BINS. Use 200-500 for smooth g(r). |
| `SMOOTH_WINDOW` | 5 | Savitzky-Golay window for smoothing. Must be odd. 0 disables smoothing. |
| `FRAME_STEP` | 1 | Use every Nth frame. Set > 1 for very long trajectories to save time. |
| `N_EQUIL_FRAMES` | 50 | Equilibration frames to skip. Check energy convergence first. |
| `ANGLE_CUTOFF` | 3.5 A | Maximum bond length for angle analysis. Should match first coordination shell. |
| `ANGLE_N_BINS` | 180 | Bins for angle histogram (1 degree resolution). |
| `COORD_CUTOFFS` | auto | Manual cutoffs for coordination number. Auto-detects from RDF first minimum. |

### Choosing R_MAX

- Must be < L/2, where L is the smallest simulation cell dimension (minimum image convention).
- For cubic cells: R_MAX < a/2 where a is the box length.
- Typical: 8-12 A for most condensed-phase systems.

## Interpreting Results

### Total RDF g(r)
- **Crystalline solid**: Sharp, well-defined peaks at discrete neighbor shell distances. g(r) does not decay to 1 at intermediate r.
- **Liquid**: Broad first peak, with g(r) oscillating toward 1 at large r. First peak position gives the most probable nearest-neighbor distance.
- **Amorphous solid**: Broad first peak (like liquid), but may retain some medium-range order (peaks at 5-10 A).
- **Gas**: g(r) ~ 1 everywhere (no spatial correlations).

### Partial RDFs g_AB(r)
- First peak of g_AB(r) gives the A-B bond length.
- If g_AB(r) has a deep first minimum (near 0), the first coordination shell is well-defined.
- Comparing partial RDFs reveals which element pairs have the strongest local ordering.

### Coordination number
- Tetrahedral: CN ~ 4 (e.g., Si-O in SiO2)
- Octahedral: CN ~ 6 (e.g., Ti-O in TiO2 rutile)
- Body-centered: CN ~ 8 (e.g., BCC metals, first shell)
- FCC/HCP: CN ~ 12 (first shell in close-packed metals)
- Non-integer CN suggests a mixture of coordination environments or thermal disorder.

### Bond angle distribution
- **Tetrahedral** (sp3): peak at ~109.5 degrees (e.g., O-Si-O in quartz)
- **Octahedral**: peaks at ~90 and ~180 degrees
- **Trigonal planar** (sp2): peak at ~120 degrees
- **Linear** (sp): peak at ~180 degrees
- Broad angle distributions indicate structural disorder or mixed coordination.

## Common Issues

| Issue | Cause | Fix |
|---|---|---|
| RDF oscillates above 1 at large r | R_MAX too close to L/2; finite-size effects | Reduce R_MAX; increase supercell size |
| RDF does not approach 1 at large r | Crystalline solid (expected) or normalization error | For crystals, g(r) oscillates indefinitely. For liquids, check volume/density normalization. |
| Partial RDF g_AB(r) is noisy | Too few A-B pairs or too few frames | Increase trajectory length; increase supercell; reduce N_BINS |
| Coordination number is fractional | Thermal fluctuations or cutoff at wrong position | Check cutoff is at the first minimum of g(r); average over many frames |
| Bond angle histogram empty | Cutoff too small; no A-B-C triplets found | Increase ANGLE_CUTOFF to include first coordination shell |
| Peaks in g(r) are asymmetric | Thermal motion (expected at finite T) | Normal physical behavior at high temperatures |
| Memory error for large trajectories | Too many frames loaded at once | Use FRAME_STEP > 1 to subsample; process in batches |
| LAMMPS dump missing atom types | Dump file does not map type IDs to elements | Use ASE with species mapping or convert with `pizza.py` first |
