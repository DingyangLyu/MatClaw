# Structural Order Parameters for Phase Transitions

## When to Use

- You need to **quantify the degree of order/disorder** during a phase transition in an MD simulation.
- You want to **distinguish solid from liquid** using Steinhardt bond-order parameters (q4, q6, w4, w6).
- You are studying **ferroelectric phase transitions** and need to track polarization as a function of temperature.
- You are investigating **octahedral tilt transitions** in perovskites (e.g., cubic to tetragonal to orthorhombic).
- You need to **detect local crystalline order** in partially melted or nucleation simulations.
- You want to compute **per-atom order parameters** to identify solid-like vs. liquid-like atoms at an interface.
- You are analyzing trajectories from melt-quench simulations (see amorphous-structure skill) and need structural metrics beyond RDF.

## Method Selection

| Transition Type | Order Parameter | Tool |
|---|---|---|
| Solid-liquid (melting, crystallization) | Steinhardt q4, q6, w4, w6 | Python (this skill) |
| Solid-liquid per-atom classification | q6 with neighbor-averaged q6_bar | Python (this skill) |
| FCC vs. BCC vs. HCP discrimination | q4, q6 scatter plot | Python (this skill) |
| Ferroelectric (paraelectric-ferroelectric) | Polarization P, Born effective charges | QE (ph.x) + Python |
| Perovskite octahedral tilting | Tilt angles, Glazer notation | Python + pymatgen |
| Antiferrodistortive (AFD) | Oxygen octahedral rotation angle | Python + pymatgen |
| Magnetic ordering | Sublattice magnetization | QE or VASP (spin-polarized) |
| Order-disorder alloy | Warren-Cowley short-range order | Python (this skill) |
| Amorphous vs. crystalline | q6 distribution, coordination number | Python (this skill) |

## Prerequisites

Pre-installed: `ase`, `numpy`, `scipy`, `matplotlib`, `pymatgen`, `spglib`.

Optional: `pyscal` (for optimized Steinhardt parameter computation): `pip install pyscal`

For ferroelectric analysis: Quantum ESPRESSO `pw.x`, `ph.x` (Born effective charges), or VASP.

For MD trajectories: ASE-readable trajectory files (`.xyz`, `.traj`, `.extxyz`, LAMMPS dump).

## Detailed Steps

### Method A: Steinhardt Bond-Order Parameters (Solid-Liquid Transitions)

Steinhardt bond-order parameters q_l are rotationally invariant measures of local structural symmetry. They are defined as:

```
q_l(i) = sqrt[ (4*pi / (2*l+1)) * sum_m |q_lm(i)|^2 ]

q_lm(i) = (1/N_b(i)) * sum_j Y_lm(r_ij)
```

where Y_lm are spherical harmonics, the sum is over N_b neighbors of atom i within a cutoff, and r_ij is the direction to neighbor j.

Key discriminators:
- **q6**: Distinguishes crystalline from liquid. q6 ~ 0.28-0.57 for crystals, ~0.25-0.35 for liquid.
- **q4**: Distinguishes FCC (q4 ~ 0.19) from BCC (q4 ~ 0.036) from HCP (q4 ~ 0.097).
- **w6**: Third-order invariant. w6 < 0 for FCC/HCP, w6 > 0 for BCC.

```python
#!/usr/bin/env python3
"""
steinhardt_order_parameters.py
Compute Steinhardt bond-order parameters (q4, q6, w4, w6) from
MD trajectories to track solid-liquid phase transitions.

Works with any ASE-readable trajectory format.

Features:
  - Global and per-atom q_l computation
  - Neighbor-averaged q_l_bar for robust solid/liquid classification
  - Crystal structure identification (FCC/BCC/HCP) from q4-q6 map
  - Time evolution of order parameters along MD trajectory
"""
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.special import sph_harm
from ase.io import read
from ase.neighborlist import neighbor_list
import os

# =============================================================================
# Configuration
# =============================================================================

TRAJECTORY_FILE = "trajectory.xyz"     # MD trajectory (multi-frame XYZ, extxyz, etc.)
CUTOFF = 3.5                           # Neighbor cutoff in Angstrom (1st shell)
                                        # Typical: 1.2-1.5 * nearest-neighbor distance
L_VALUES = [4, 6]                      # Steinhardt l-values to compute
OUTPUT_DIR = "/tmp/order_parameters"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# =============================================================================
# 1. Steinhardt parameter computation
# =============================================================================

def compute_qlm(atoms, l, cutoff):
    """
    Compute complex q_lm(i) for each atom i.

    Parameters
    ----------
    atoms : ase.Atoms
        Atomic structure with PBC.
    l : int
        Angular momentum quantum number (typically 4 or 6).
    cutoff : float
        Neighbor cutoff in Angstrom.

    Returns
    -------
    qlm : array, shape (n_atoms, 2*l+1)
        Complex q_lm values for each atom.
    n_neighbors : array, shape (n_atoms,)
        Number of neighbors for each atom.
    """
    n_atoms = len(atoms)
    qlm = np.zeros((n_atoms, 2 * l + 1), dtype=complex)
    n_neighbors = np.zeros(n_atoms, dtype=int)

    # Get neighbor list
    idx_i, idx_j, dist_vec = neighbor_list('ijD', atoms, cutoff)

    for k in range(len(idx_i)):
        i = idx_i[k]
        d = dist_vec[k]
        r = np.linalg.norm(d)
        if r < 1e-10:
            continue

        # Spherical coordinates
        theta = np.arccos(d[2] / r)
        phi = np.arctan2(d[1], d[0])

        # Spherical harmonics Y_lm(theta, phi) for m = -l to l
        for m_idx, m in enumerate(range(-l, l + 1)):
            qlm[i, m_idx] += sph_harm(m, l, phi, theta)

        n_neighbors[i] += 1

    # Normalize by number of neighbors
    mask = n_neighbors > 0
    for m_idx in range(2 * l + 1):
        qlm[mask, m_idx] /= n_neighbors[mask]

    return qlm, n_neighbors


def compute_ql(qlm, l):
    """
    Compute rotationally invariant q_l from q_lm.

    q_l = sqrt[ (4*pi / (2*l+1)) * sum_m |q_lm|^2 ]
    """
    sum_sq = np.sum(np.abs(qlm) ** 2, axis=1)
    ql = np.sqrt(4.0 * np.pi / (2 * l + 1) * sum_sq)
    return ql


def compute_wl(qlm, l):
    """
    Compute third-order invariant w_l from q_lm.

    w_l = sum_{m1+m2+m3=0} W3j(l,l,l; m1,m2,m3) * q_lm1 * q_lm2 * q_lm3
          / (sum_m |q_lm|^2)^(3/2)

    Uses Wigner 3j symbols.
    """
    from scipy.special import factorial

    def wigner3j(l1, l2, l3, m1, m2, m3):
        """Simplified Wigner 3j symbol for l1=l2=l3=l."""
        if m1 + m2 + m3 != 0:
            return 0.0
        try:
            from scipy.special import comb
            # Use the formula for equal l values
            # This is a simplified version; for general use, a proper
            # Wigner 3j library is recommended.
            j = l1
            # Numerical evaluation
            from functools import lru_cache

            @lru_cache(maxsize=None)
            def fact(n):
                return float(factorial(n, exact=True)) if n >= 0 else 0.0

            prefactor = (-1) ** (3 * j) * fact(j) ** 3 / fact(3 * j + 1)

            # Sum over t
            total = 0.0
            for t in range(max(0, -m1, m2) + j,
                           min(j - m1, j + m2, 2 * j) + 1):
                try:
                    num = fact(t) * fact(3 * j - t)
                    den = (fact(j - m1 - t + j) * fact(j + m2 - t + j) *
                           fact(t - j + m1) * fact(t - j - m2) *
                           fact(j - m3 - t + j) * fact(t - j + m3))
                    if den != 0:
                        total += (-1) ** t / den * num
                except (ValueError, OverflowError):
                    pass

            return prefactor * total
        except Exception:
            return 0.0

    n_atoms = qlm.shape[0]
    wl = np.zeros(n_atoms)

    # Precompute Wigner 3j values
    w3j_cache = {}
    for m1 in range(-l, l + 1):
        for m2 in range(-l, l + 1):
            m3 = -m1 - m2
            if abs(m3) <= l:
                w3j = wigner3j(l, l, l, m1, m2, m3)
                if abs(w3j) > 1e-15:
                    w3j_cache[(m1, m2, m3)] = w3j

    for i in range(n_atoms):
        w_sum = 0.0
        for (m1, m2, m3), w3j in w3j_cache.items():
            idx1 = m1 + l
            idx2 = m2 + l
            idx3 = m3 + l
            w_sum += w3j * qlm[i, idx1] * qlm[i, idx2] * qlm[i, idx3]

        norm = np.sum(np.abs(qlm[i]) ** 2) ** 1.5
        if norm > 1e-15:
            wl[i] = np.real(w_sum) / norm

    return wl


def compute_ql_bar(qlm, atoms, l, cutoff):
    """
    Compute neighbor-averaged Steinhardt parameter q_l_bar.
    This smooths out thermal fluctuations and provides better
    discrimination between solid and liquid atoms.

    q_lm_bar(i) = (1 / (N_b(i) + 1)) * [q_lm(i) + sum_j q_lm(j)]

    where the sum is over neighbors j of atom i.
    """
    n_atoms = len(atoms)
    qlm_bar = np.zeros_like(qlm)

    idx_i, idx_j = neighbor_list('ij', atoms, cutoff)

    for i in range(n_atoms):
        # Sum: self + neighbors
        total = qlm[i].copy()
        count = 1

        # Find neighbors of atom i
        mask = idx_i == i
        neighbors = idx_j[mask]

        for j in neighbors:
            total += qlm[j]
            count += 1

        qlm_bar[i] = total / count

    ql_bar = compute_ql(qlm_bar, l)
    return ql_bar


# =============================================================================
# 2. Crystal structure identification
# =============================================================================

# Reference values for perfect crystals (q4, q6)
CRYSTAL_REFERENCES = {
    'FCC':  {'q4': 0.1909, 'q6': 0.5745, 'w4': -0.1593, 'w6': -0.0132},
    'HCP':  {'q4': 0.0972, 'q6': 0.4848, 'w4': 0.1341, 'w6': -0.0124},
    'BCC':  {'q4': 0.0364, 'q6': 0.5107, 'w4': 0.1593, 'w6': 0.0132},
    'SC':   {'q4': 0.7637, 'q6': 0.3536, 'w4': 0.1593, 'w6': 0.0132},
    'Diamond': {'q4': 0.5092, 'q6': 0.6285, 'w4': -0.1593, 'w6': -0.0132},
    'Liquid': {'q4': 0.02, 'q6': 0.28, 'w4': 0.0, 'w6': 0.0},
}


def classify_structure(q4_mean, q6_mean):
    """Classify structure based on average q4 and q6."""
    min_dist = np.inf
    best_match = 'Unknown'

    for name, refs in CRYSTAL_REFERENCES.items():
        dist = np.sqrt((q4_mean - refs['q4'])**2 + (q6_mean - refs['q6'])**2)
        if dist < min_dist:
            min_dist = dist
            best_match = name

    return best_match, min_dist


def classify_atoms(q6_bar, threshold=0.35):
    """
    Classify atoms as solid-like or liquid-like based on q6_bar.

    Parameters
    ----------
    q6_bar : array
        Neighbor-averaged q6 per atom.
    threshold : float
        q6_bar > threshold --> solid-like.

    Returns
    -------
    solid_mask : boolean array
        True for solid-like atoms.
    """
    return q6_bar > threshold


# =============================================================================
# 3. Trajectory analysis
# =============================================================================

def analyze_trajectory(traj_file, cutoff, l_values=[4, 6], every=1):
    """
    Analyze MD trajectory frame by frame, computing order parameters.

    Parameters
    ----------
    traj_file : str
        Path to trajectory file (ASE-readable).
    cutoff : float
        Neighbor cutoff in Angstrom.
    l_values : list of int
        l values for Steinhardt parameters.
    every : int
        Analyze every N-th frame.

    Returns
    -------
    results : dict
        Time series of order parameters.
    """
    try:
        frames = read(traj_file, index=':')
    except Exception:
        frames = [read(traj_file)]

    if not isinstance(frames, list):
        frames = [frames]

    n_frames = len(frames)
    print(f"Loaded {n_frames} frames from {traj_file}")

    results = {
        'frame': [],
        'n_atoms': [],
        'temperature': [],
    }
    for l in l_values:
        results[f'q{l}_mean'] = []
        results[f'q{l}_std'] = []
        results[f'q{l}_bar_mean'] = []
        results[f'w{l}_mean'] = []
        results[f'solid_fraction_q{l}'] = []

    per_atom_data = []  # Store per-atom data for selected frames

    for i_frame in range(0, n_frames, every):
        atoms = frames[i_frame]
        n = len(atoms)

        results['frame'].append(i_frame)
        results['n_atoms'].append(n)

        # Temperature from kinetic energy if available
        try:
            T = atoms.get_temperature()
        except Exception:
            T = 0.0
        results['temperature'].append(T)

        for l in l_values:
            qlm, n_neigh = compute_qlm(atoms, l, cutoff)
            ql = compute_ql(qlm, l)
            ql_bar = compute_ql_bar(qlm, atoms, l, cutoff)
            wl = compute_wl(qlm, l)

            results[f'q{l}_mean'].append(np.mean(ql))
            results[f'q{l}_std'].append(np.std(ql))
            results[f'q{l}_bar_mean'].append(np.mean(ql_bar))
            results[f'w{l}_mean'].append(np.mean(wl))

            # Solid fraction based on q_l_bar
            threshold = 0.35 if l == 6 else 0.10
            solid_mask = ql_bar > threshold
            results[f'solid_fraction_q{l}'].append(np.sum(solid_mask) / n)

        if i_frame % max(1, n_frames // 5) == 0:
            print(f"  Frame {i_frame:6d}: T = {T:.0f} K, "
                  f"<q6> = {results['q6_mean'][-1]:.4f}, "
                  f"solid_frac = {results['solid_fraction_q6'][-1]:.3f}")

    # Convert to arrays
    for key in results:
        results[key] = np.array(results[key])

    return results


# =============================================================================
# 4. Main analysis
# =============================================================================

print("=" * 65)
print("  Steinhardt Bond-Order Parameter Analysis")
print("=" * 65)

if os.path.isfile(TRAJECTORY_FILE):
    results = analyze_trajectory(TRAJECTORY_FILE, CUTOFF, L_VALUES, every=1)
else:
    # Generate demonstration data with a model melting trajectory
    print(f"\n{TRAJECTORY_FILE} not found. Generating demonstration data.")
    print("For real analysis, provide a trajectory file.\n")

    from ase.build import bulk
    from ase.md.velocitydistribution import MaxwellBoltzmannDistribution
    from ase.md.langevin import Langevin
    from ase import units
    from mace.calculators import mace_mp

    import warnings
    warnings.filterwarnings("ignore")

    calc = mace_mp(model="small", device="cpu", default_dtype="float64")

    atoms = bulk("Cu", "fcc", a=3.615) * (3, 3, 3)
    atoms.calc = calc
    n_atoms = len(atoms)
    print(f"Demo: Cu FCC supercell, {n_atoms} atoms")

    # Save frames at different temperatures for analysis
    demo_frames = []
    temperatures_demo = [100, 300, 600, 900, 1200, 1500, 1800]

    for T in temperatures_demo:
        test_atoms = atoms.copy()
        test_atoms.calc = calc
        MaxwellBoltzmannDistribution(test_atoms, temperature_K=T)
        dyn = Langevin(test_atoms, timestep=2.0 * units.fs,
                       temperature_K=T, friction=0.01 / units.fs)
        dyn.run(200)
        demo_frames.append(test_atoms.copy())

    # Analyze each frame
    results = {
        'frame': [], 'n_atoms': [], 'temperature': [],
        'q4_mean': [], 'q4_std': [], 'q4_bar_mean': [], 'w4_mean': [],
        'q6_mean': [], 'q6_std': [], 'q6_bar_mean': [], 'w6_mean': [],
        'solid_fraction_q4': [], 'solid_fraction_q6': [],
    }

    for i, atoms_frame in enumerate(demo_frames):
        T = temperatures_demo[i]
        results['frame'].append(i)
        results['n_atoms'].append(len(atoms_frame))
        results['temperature'].append(T)

        for l in [4, 6]:
            qlm, n_neigh = compute_qlm(atoms_frame, l, CUTOFF)
            ql = compute_ql(qlm, l)
            ql_bar = compute_ql_bar(qlm, atoms_frame, l, CUTOFF)
            wl = compute_wl(qlm, l)

            results[f'q{l}_mean'].append(np.mean(ql))
            results[f'q{l}_std'].append(np.std(ql))
            results[f'q{l}_bar_mean'].append(np.mean(ql_bar))
            results[f'w{l}_mean'].append(np.mean(wl))

            threshold = 0.35 if l == 6 else 0.10
            results[f'solid_fraction_q{l}'].append(
                np.sum(ql_bar > threshold) / len(atoms_frame))

        print(f"  T = {T:6.0f} K: <q4> = {results['q4_mean'][-1]:.4f}, "
              f"<q6> = {results['q6_mean'][-1]:.4f}, "
              f"solid = {results['solid_fraction_q6'][-1]:.3f}")

    for key in results:
        results[key] = np.array(results[key])

# =============================================================================
# 5. Plotting
# =============================================================================

print("\n--- Generating plots ---")

# Plot 1: q4, q6 vs temperature (or frame)
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

x_axis = results['temperature'] if np.any(results['temperature'] > 0) else results['frame']
x_label = 'Temperature (K)' if np.any(results['temperature'] > 0) else 'Frame'

ax = axes[0, 0]
ax.plot(x_axis, results['q6_mean'], 'o-', color='steelblue', linewidth=2,
        markersize=6, label=r'$\langle q_6 \rangle$')
ax.fill_between(x_axis,
                results['q6_mean'] - results['q6_std'],
                results['q6_mean'] + results['q6_std'],
                alpha=0.2, color='steelblue')
ax.set_xlabel(x_label, fontsize=12)
ax.set_ylabel(r'$q_6$', fontsize=12)
ax.set_title(r'Bond-Order Parameter $q_6$', fontsize=13)
ax.grid(True, alpha=0.3)
ax.legend(fontsize=11)

ax = axes[0, 1]
ax.plot(x_axis, results['q4_mean'], 's-', color='darkorange', linewidth=2,
        markersize=6, label=r'$\langle q_4 \rangle$')
ax.fill_between(x_axis,
                results['q4_mean'] - results['q4_std'],
                results['q4_mean'] + results['q4_std'],
                alpha=0.2, color='darkorange')
ax.set_xlabel(x_label, fontsize=12)
ax.set_ylabel(r'$q_4$', fontsize=12)
ax.set_title(r'Bond-Order Parameter $q_4$', fontsize=13)
ax.grid(True, alpha=0.3)
ax.legend(fontsize=11)

ax = axes[1, 0]
ax.plot(x_axis, results['solid_fraction_q6'], 'D-', color='seagreen',
        linewidth=2, markersize=6)
ax.set_xlabel(x_label, fontsize=12)
ax.set_ylabel('Solid-like fraction', fontsize=12)
ax.set_title('Fraction of Solid-Like Atoms ($q_6$-based)', fontsize=13)
ax.set_ylim(-0.05, 1.05)
ax.grid(True, alpha=0.3)

# q4-q6 scatter with crystal reference points
ax = axes[1, 1]
ax.scatter(results['q4_mean'], results['q6_mean'],
           c=x_axis, cmap='coolwarm', s=80, edgecolors='black',
           linewidths=0.5, zorder=5)
# Add reference points
for name, refs in CRYSTAL_REFERENCES.items():
    ax.plot(refs['q4'], refs['q6'], '*', markersize=15, label=name,
            markeredgecolor='black', markeredgewidth=0.5)
ax.set_xlabel(r'$\langle q_4 \rangle$', fontsize=12)
ax.set_ylabel(r'$\langle q_6 \rangle$', fontsize=12)
ax.set_title('Crystal Structure Map', fontsize=13)
ax.legend(fontsize=8, ncol=2, loc='upper left')
ax.grid(True, alpha=0.3)

fig.suptitle('Steinhardt Bond-Order Parameter Analysis', fontsize=15)
fig.tight_layout()
fig.savefig(os.path.join(OUTPUT_DIR, 'steinhardt_analysis.png'),
            dpi=200, bbox_inches='tight')
plt.close()
print(f"Saved: {OUTPUT_DIR}/steinhardt_analysis.png")

# Save numerical data
data_out = np.column_stack([
    results['frame'], results['temperature'],
    results['q4_mean'], results['q6_mean'],
    results['q4_bar_mean'], results['q6_bar_mean'],
    results['w4_mean'], results['w6_mean'],
    results['solid_fraction_q6']
])
np.savetxt(
    os.path.join(OUTPUT_DIR, 'order_parameters.dat'),
    data_out,
    header="frame  T(K)  <q4>  <q6>  <q4_bar>  <q6_bar>  <w4>  <w6>  solid_frac",
    fmt="%6d %8.1f %8.4f %8.4f %8.4f %8.4f %10.6f %10.6f %8.4f"
)
print(f"Saved: {OUTPUT_DIR}/order_parameters.dat")
```

### Method B: Ferroelectric Polarization Order Parameter

For ferroelectric transitions, the order parameter is the spontaneous polarization P, which vanishes above the Curie temperature Tc.

```python
#!/usr/bin/env python3
"""
ferroelectric_order_parameter.py
Track polarization as order parameter for ferroelectric phase transitions.

Two approaches:
  1. Berry phase polarization from QE (pw.x with lberry=.true.)
  2. Point-charge model using Born effective charges from ph.x

For MD simulations, the point-charge model is more practical.
"""
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os

from ase.io import read
from pymatgen.core import Structure

OUTPUT_DIR = "/tmp/ferroelectric_op"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# =============================================================================
# 1. Born effective charge model for polarization
# =============================================================================

def compute_polarization_born(atoms, Z_star, reference_positions=None):
    """
    Compute polarization using Born effective charges.

    P = (e / V) * sum_i Z*_i . u_i

    where u_i is the displacement of atom i from its centrosymmetric
    reference position and Z*_i is the Born effective charge tensor.

    Parameters
    ----------
    atoms : ase.Atoms
        Current atomic configuration.
    Z_star : dict
        Born effective charge tensors: {element: 3x3 array}.
        For isotropic, can be {element: scalar}.
    reference_positions : array, optional
        Reference (centrosymmetric) positions. If None, uses the
        average position of symmetry-related atoms.

    Returns
    -------
    P : array, shape (3,)
        Polarization vector in C/m^2.
    """
    e_C = 1.602176634e-19  # Coulomb
    positions = atoms.get_positions()  # Angstrom
    volume = atoms.get_volume()  # Angstrom^3
    volume_m3 = volume * 1e-30
    symbols = atoms.get_chemical_symbols()

    if reference_positions is None:
        # Use current positions centroid as reference (approximation)
        reference_positions = np.mean(positions, axis=0) * np.ones_like(positions)
        # Better: user should provide centrosymmetric reference structure

    P = np.zeros(3)
    for i, sym in enumerate(symbols):
        u = positions[i] - reference_positions[i]
        # Minimum image convention for displacements
        cell = atoms.get_cell()
        frac_u = np.linalg.solve(cell.T, u)
        frac_u -= np.round(frac_u)
        u = cell.T @ frac_u

        u_m = u * 1e-10  # Angstrom to meters

        if isinstance(Z_star[sym], (int, float)):
            P += Z_star[sym] * u_m * e_C
        else:
            P += Z_star[sym] @ u_m * e_C

    P /= volume_m3  # C/m^2

    return P


def compute_polarization_trajectory(traj_file, Z_star, ref_structure_file=None,
                                      every=1):
    """
    Compute polarization along an MD trajectory.

    Returns time series of polarization components.
    """
    frames = read(traj_file, index=':')
    if not isinstance(frames, list):
        frames = [frames]

    if ref_structure_file is not None:
        ref_atoms = read(ref_structure_file)
        ref_positions = ref_atoms.get_positions()
    else:
        ref_positions = frames[0].get_positions()

    P_series = []

    for i in range(0, len(frames), every):
        P = compute_polarization_born(frames[i], Z_star, ref_positions)
        P_series.append(P)

    return np.array(P_series)  # shape: (n_frames, 3)


# =============================================================================
# 2. QE Berry phase polarization setup
# =============================================================================

def generate_berry_phase_input(prefix, nppstr=10):
    """
    Generate QE input for Berry phase polarization calculation.

    The Berry phase approach gives the exact polarization (modulo a quantum).
    It requires an NSCF calculation with specific k-point strings.

    Parameters
    ----------
    prefix : str
        QE calculation prefix.
    nppstr : int
        Number of k-points per string (along each reciprocal lattice direction).
    """
    # Berry phase input for each direction
    for idir, direction in enumerate(['x', 'y', 'z'], start=1):
        input_text = f"""&CONTROL
    calculation = 'nscf'
    prefix = '{prefix}'
    pseudo_dir = './pseudo/'
    outdir = './tmp/'
    verbosity = 'high'
    lberry = .true.
    gdir = {idir}
    nppstr = {nppstr}
/
&SYSTEM
    ! Copy from SCF input
    ! ibrav, celldm, nat, ntyp, ecutwfc, etc.
    ! Must match SCF exactly
/
&ELECTRONS
    conv_thr = 1.0d-8
/
"""
        filename = f'berry_{direction}.in'
        with open(filename, 'w') as f:
            f.write(input_text)
        print(f"Wrote {filename} (direction {idir} = {direction})")

    print("\nComplete the &SYSTEM block with parameters from your SCF input.")
    print("Run each berry_x/y/z calculation after SCF.")
    print("Parse 'P =' from the output for polarization in each direction.")
    print("\nAlternatively, parse Born effective charges from ph.x output:")
    print("  grep -A 3 'Effective charges' ph.out")


# =============================================================================
# 3. Example: BaTiO3 ferroelectric transition
# =============================================================================

print("=" * 65)
print("  Ferroelectric Order Parameter: Polarization")
print("=" * 65)

# BaTiO3 Born effective charges (from literature/DFT)
Z_star_BaTiO3 = {
    'Ba': np.diag([2.75, 2.75, 2.75]),
    'Ti': np.diag([7.25, 7.25, 7.25]),
    'O':  np.diag([-2.15, -2.15, -5.71]),   # Apical O has different Z*_zz
}

print("\nBorn effective charges (BaTiO3):")
for elem, Z in Z_star_BaTiO3.items():
    if isinstance(Z, np.ndarray):
        print(f"  {elem}: diag({Z[0,0]:.2f}, {Z[1,1]:.2f}, {Z[2,2]:.2f})")
    else:
        print(f"  {elem}: {Z:.2f}")

# If trajectory file exists, analyze it
if os.path.isfile("trajectory.xyz"):
    P_series = compute_polarization_trajectory(
        "trajectory.xyz", Z_star_BaTiO3, every=5)

    fig, ax = plt.subplots(figsize=(10, 5))
    frames = np.arange(len(P_series))
    for i, comp in enumerate(['x', 'y', 'z']):
        ax.plot(frames, P_series[:, i] * 100, linewidth=1.5,
                label=f'P_{comp}')  # Convert C/m^2 to uC/cm^2
    ax.set_xlabel('Frame', fontsize=12)
    ax.set_ylabel(r'Polarization ($\mu$C/cm$^2$)', fontsize=12)
    ax.set_title('Ferroelectric Polarization vs MD Frame', fontsize=13)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(OUTPUT_DIR, 'polarization_trajectory.png'),
                dpi=200, bbox_inches='tight')
    plt.close()
    print(f"Saved: {OUTPUT_DIR}/polarization_trajectory.png")
else:
    print("\nNo trajectory file found. For demonstration, generating a")
    print("synthetic polarization vs temperature curve.")

    # Synthetic: P(T) for a displacive ferroelectric
    T_arr = np.arange(0, 800, 10)
    Tc = 400  # Curie temperature (K)

    # Mean-field model: P(T) = P0 * (1 - T/Tc)^beta for T < Tc
    P0 = 0.26  # C/m^2 (BaTiO3 ~ 0.26 C/m^2)
    beta = 0.5  # Mean-field exponent

    P_T = np.where(T_arr < Tc,
                   P0 * (1 - T_arr / Tc) ** beta,
                   0.0)
    # Add thermal fluctuation noise
    P_T += np.random.normal(0, 0.005, len(T_arr))
    P_T = np.maximum(P_T, 0)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(T_arr, P_T * 100, 'o-', color='steelblue', markersize=3,
            linewidth=1.5)
    ax.axvline(Tc, color='red', linestyle='--', alpha=0.7,
               label=f'$T_c$ = {Tc} K')
    ax.set_xlabel('Temperature (K)', fontsize=13)
    ax.set_ylabel(r'Polarization P ($\mu$C/cm$^2$)', fontsize=13)
    ax.set_title('Ferroelectric Order Parameter P(T)', fontsize=14)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, T_arr[-1])
    ax.set_ylim(bottom=0)
    fig.tight_layout()
    fig.savefig(os.path.join(OUTPUT_DIR, 'polarization_vs_T.png'),
                dpi=200, bbox_inches='tight')
    plt.close()
    print(f"Saved: {OUTPUT_DIR}/polarization_vs_T.png")

# Generate Berry phase inputs
print("\n--- QE Berry Phase Inputs ---")
generate_berry_phase_input('batio3', nppstr=10)
```

### Method C: Octahedral Tilt Angle (Perovskite Transitions)

For perovskite ABO3 materials, octahedral tilting is the primary order parameter for many structural phase transitions.

```python
#!/usr/bin/env python3
"""
octahedral_tilt_analysis.py
Compute octahedral tilt angles in perovskite ABO3 structures.

Tracks:
  - Tilt angle around each Cartesian axis
  - Glazer tilt system classification (a+b-c-, etc.)
  - Temperature dependence from MD trajectory

Works with any perovskite structure (pymatgen Structure or ASE Atoms).
"""
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os

from pymatgen.core import Structure
from ase.io import read

OUTPUT_DIR = "/tmp/tilt_analysis"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# =============================================================================
# 1. Octahedral tilt angle computation
# =============================================================================

def find_octahedra(structure, b_site_element='Ti', x_site_element='O',
                    bond_cutoff=2.5):
    """
    Identify BO6 octahedra in a perovskite structure.

    Parameters
    ----------
    structure : pymatgen Structure
        Perovskite structure.
    b_site_element : str
        B-site cation (e.g., 'Ti', 'Zr', 'Nb').
    x_site_element : str
        Anion (typically 'O' for oxides, 'F' for fluorides).
    bond_cutoff : float
        Maximum B-X bond length in Angstrom.

    Returns
    -------
    octahedra : list of dict
        Each dict: {'center': B-site index, 'vertices': list of 6 X-site indices,
                     'center_pos': coords, 'vertex_pos': coords array}
    """
    b_sites = [i for i, site in enumerate(structure) if site.specie.symbol == b_site_element]
    x_sites = [i for i, site in enumerate(structure) if site.specie.symbol == x_site_element]

    octahedra = []

    for b_idx in b_sites:
        b_pos = structure[b_idx].coords
        distances = []

        for x_idx in x_sites:
            # Use minimum image distance
            dist = structure.get_distance(b_idx, x_idx)
            if dist < bond_cutoff:
                distances.append((x_idx, dist))

        # Sort by distance and take 6 nearest
        distances.sort(key=lambda x: x[1])
        if len(distances) >= 6:
            vertex_indices = [d[0] for d in distances[:6]]
            vertex_positions = np.array([structure[vi].coords for vi in vertex_indices])

            octahedra.append({
                'center': b_idx,
                'vertices': vertex_indices,
                'center_pos': b_pos,
                'vertex_pos': vertex_positions,
            })

    return octahedra


def compute_tilt_angles(octahedra, lattice_vectors):
    """
    Compute octahedral tilt angles around each Cartesian axis.

    The tilt angle is defined as the deviation of the B-X-B angle from 180 degrees,
    measured by the angle between opposite B-X bond vectors.

    For each octahedron, identifies three pairs of opposite vertices
    (along x, y, z pseudo-cubic axes) and computes the tilt.

    Parameters
    ----------
    octahedra : list of dict
        From find_octahedra.
    lattice_vectors : array
        Lattice vectors of the structure.

    Returns
    -------
    tilts : dict
        {'x': [angles], 'y': [angles], 'z': [angles]} in degrees.
    """
    tilts = {'x': [], 'y': [], 'z': []}

    for octa in octahedra:
        center = octa['center_pos']
        vertices = octa['vertex_pos']

        # Bond vectors from center to vertices
        bonds = vertices - center

        # Identify opposite pairs: vertices with nearly antiparallel bonds
        n_vert = len(bonds)
        paired = set()
        pairs = []

        for i in range(n_vert):
            if i in paired:
                continue
            for j in range(i + 1, n_vert):
                if j in paired:
                    continue
                # Angle between bonds[i] and bonds[j]
                cos_angle = np.dot(bonds[i], bonds[j]) / (
                    np.linalg.norm(bonds[i]) * np.linalg.norm(bonds[j]) + 1e-10)
                if cos_angle < -0.5:  # Roughly opposite (>120 degrees)
                    pairs.append((i, j))
                    paired.add(i)
                    paired.add(j)
                    break

        if len(pairs) < 3:
            continue

        # Assign pairs to pseudo-cubic axes based on bond direction
        for pair in pairs:
            i, j = pair
            avg_direction = bonds[i] - bonds[j]  # Points along the octahedral axis
            avg_direction /= (np.linalg.norm(avg_direction) + 1e-10)

            # Tilt angle: deviation of the B-X bond from the ideal axis
            # For untilted octahedron, bonds[i] and bonds[j] are antiparallel
            # Tilt = 90 - (angle between bond and axis) = arccos(|cos|) deviation
            angle_rad = np.arccos(np.clip(-np.dot(bonds[i], bonds[j]) / (
                np.linalg.norm(bonds[i]) * np.linalg.norm(bonds[j]) + 1e-10), -1, 1))
            tilt_deg = 180.0 - np.degrees(angle_rad)  # Deviation from 180

            # Determine which axis
            abs_dir = np.abs(avg_direction)
            axis_idx = np.argmax(abs_dir)
            axis_name = ['x', 'y', 'z'][axis_idx]

            tilts[axis_name].append(tilt_deg / 2.0)  # Convention: half the total tilt

    return tilts


def classify_glazer(tilts_x, tilts_y, tilts_z, tolerance=2.0):
    """
    Classify Glazer tilt system from tilt angles.

    Returns Glazer notation like 'a0a0c-', 'a+b-b-', etc.

    Parameters
    ----------
    tilts_x, tilts_y, tilts_z : float
        Average tilt angles (degrees) around each axis.
    tolerance : float
        Angles below this threshold (degrees) are considered zero tilt.
    """
    angles = [tilts_x, tilts_y, tilts_z]
    labels = []

    for angle in angles:
        if abs(angle) < tolerance:
            labels.append('0')
        else:
            # In-phase (+) vs anti-phase (-) requires checking
            # adjacent octahedra, which needs more structural info.
            # Here we just report the magnitude.
            labels.append('-')  # Default to anti-phase (most common)

    # Group equal tilts
    notation = ''
    used = [False, False, False]
    letters = 'abc'
    letter_idx = 0

    for i in range(3):
        if used[i]:
            continue
        letter = letters[letter_idx]
        letter_idx += 1
        notation += letter + labels[i]
        used[i] = True

        # Check if other axes have the same tilt magnitude
        for j in range(i + 1, 3):
            if not used[j] and abs(angles[i] - angles[j]) < tolerance:
                notation += letter + labels[j]
                used[j] = True

    return notation


# =============================================================================
# 2. Trajectory analysis for tilt angles
# =============================================================================

def analyze_tilt_trajectory(traj_file, b_element='Ti', x_element='O',
                              bond_cutoff=2.5, every=1):
    """
    Compute octahedral tilt angles from each frame of an MD trajectory.
    """
    frames = read(traj_file, index=':')
    if not isinstance(frames, list):
        frames = [frames]

    results = {'frame': [], 'temperature': [],
               'tilt_x': [], 'tilt_y': [], 'tilt_z': [],
               'tilt_total': []}

    for i in range(0, len(frames), every):
        atoms = frames[i]
        struct = Structure(
            atoms.get_cell(), atoms.get_chemical_symbols(),
            atoms.get_scaled_positions(), coords_are_cartesian=False
        )

        octahedra = find_octahedra(struct, b_element, x_element, bond_cutoff)
        if not octahedra:
            continue

        tilts = compute_tilt_angles(octahedra, struct.lattice.matrix)

        tx = np.mean(tilts['x']) if tilts['x'] else 0.0
        ty = np.mean(tilts['y']) if tilts['y'] else 0.0
        tz = np.mean(tilts['z']) if tilts['z'] else 0.0

        try:
            T = atoms.get_temperature()
        except Exception:
            T = 0.0

        results['frame'].append(i)
        results['temperature'].append(T)
        results['tilt_x'].append(tx)
        results['tilt_y'].append(ty)
        results['tilt_z'].append(tz)
        results['tilt_total'].append(np.sqrt(tx**2 + ty**2 + tz**2))

    for key in results:
        results[key] = np.array(results[key])

    return results


# =============================================================================
# 3. Warren-Cowley short-range order parameter
# =============================================================================

def warren_cowley_sro(atoms, cutoff=3.5, species_A='A', species_B='B'):
    """
    Compute Warren-Cowley short-range order parameter alpha_1 for a binary alloy.

    alpha_1 = 1 - P_AB / c_B

    where P_AB is the probability that a neighbor of species A is species B,
    and c_B is the overall concentration of species B.

    alpha_1 = 0: random (ideal solid solution)
    alpha_1 > 0: clustering (like atoms prefer like neighbors)
    alpha_1 < 0: ordering (unlike atoms prefer unlike neighbors)

    Parameters
    ----------
    atoms : ase.Atoms
        Binary alloy structure.
    cutoff : float
        First-shell cutoff distance.
    species_A, species_B : str
        Chemical symbols of the two species.

    Returns
    -------
    alpha_1 : float
        Warren-Cowley short-range order parameter.
    """
    from ase.neighborlist import neighbor_list

    symbols = atoms.get_chemical_symbols()
    n_total = len(atoms)
    n_A = sum(1 for s in symbols if s == species_A)
    n_B = sum(1 for s in symbols if s == species_B)
    c_B = n_B / n_total

    if c_B == 0 or c_B == 1:
        return 0.0

    idx_i, idx_j = neighbor_list('ij', atoms, cutoff)

    # Count AB pairs among A-centered neighbors
    n_AB = 0
    n_A_neighbors = 0

    for k in range(len(idx_i)):
        if symbols[idx_i[k]] == species_A:
            n_A_neighbors += 1
            if symbols[idx_j[k]] == species_B:
                n_AB += 1

    if n_A_neighbors == 0:
        return 0.0

    P_AB = n_AB / n_A_neighbors
    alpha_1 = 1.0 - P_AB / c_B

    return alpha_1


# =============================================================================
# 4. Main demonstration
# =============================================================================

print("=" * 65)
print("  Structural Order Parameter Analysis")
print("=" * 65)

# Example: analyze a perovskite structure
if os.path.isfile("structure.cif"):
    struct = Structure.from_file("structure.cif")
    print(f"\nStructure: {struct.composition}")

    # Determine B-site and X-site elements
    elements = [str(s) for s in struct.composition.elements]
    # Typical perovskite: A is large cation, B is small cation, X is anion
    # Auto-detect or specify manually
    b_element = 'Ti'  # Adjust for your system
    x_element = 'O'

    octahedra = find_octahedra(struct, b_element, x_element)
    print(f"Found {len(octahedra)} octahedra")

    if octahedra:
        tilts = compute_tilt_angles(octahedra, struct.lattice.matrix)
        tx = np.mean(tilts['x']) if tilts['x'] else 0.0
        ty = np.mean(tilts['y']) if tilts['y'] else 0.0
        tz = np.mean(tilts['z']) if tilts['z'] else 0.0

        print(f"\nOctahedral tilt angles:")
        print(f"  Around x: {tx:.2f} degrees")
        print(f"  Around y: {ty:.2f} degrees")
        print(f"  Around z: {tz:.2f} degrees")

        glazer = classify_glazer(tx, ty, tz)
        print(f"  Glazer notation: {glazer}")
else:
    print("\nNo structure.cif found. Using demonstration data.")

    # Demonstrate with synthetic tilt vs T data
    T_arr = np.arange(0, 1200, 20)
    T_c1 = 400   # Tetragonal to cubic transition
    T_c2 = 200   # Orthorhombic to tetragonal transition

    # Tilt angles: continuous phase transitions
    tilt_x = np.where(T_arr < T_c1, 8.0 * (1 - T_arr / T_c1) ** 0.5, 0.0)
    tilt_y = np.where(T_arr < T_c2, 5.0 * (1 - T_arr / T_c2) ** 0.5, 0.0)
    tilt_z = np.where(T_arr < T_c1, 8.0 * (1 - T_arr / T_c1) ** 0.5, 0.0)

    # Add noise
    tilt_x += np.random.normal(0, 0.3, len(T_arr))
    tilt_y += np.random.normal(0, 0.3, len(T_arr))
    tilt_z += np.random.normal(0, 0.3, len(T_arr))
    tilt_x = np.maximum(tilt_x, 0)
    tilt_y = np.maximum(tilt_y, 0)
    tilt_z = np.maximum(tilt_z, 0)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))

    ax = axes[0]
    ax.plot(T_arr, tilt_x, 'o-', color='steelblue', markersize=3, linewidth=1.5,
            label=r'$\theta_x$')
    ax.plot(T_arr, tilt_y, 's-', color='darkorange', markersize=3, linewidth=1.5,
            label=r'$\theta_y$')
    ax.plot(T_arr, tilt_z, 'D-', color='seagreen', markersize=3, linewidth=1.5,
            label=r'$\theta_z$')
    ax.axvline(T_c1, color='red', linestyle='--', alpha=0.6,
               label=f'$T_{{c1}}$ = {T_c1} K')
    ax.axvline(T_c2, color='purple', linestyle='--', alpha=0.6,
               label=f'$T_{{c2}}$ = {T_c2} K')
    ax.set_xlabel('Temperature (K)', fontsize=13)
    ax.set_ylabel('Tilt angle (degrees)', fontsize=13)
    ax.set_title('Octahedral Tilt Order Parameter', fontsize=14)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(bottom=-0.5)

    # Phase diagram annotation
    ax = axes[1]
    ax.fill_betweenx([0, 1], 0, T_c2, alpha=0.2, color='steelblue',
                      label='Orthorhombic')
    ax.fill_betweenx([0, 1], T_c2, T_c1, alpha=0.2, color='darkorange',
                      label='Tetragonal')
    ax.fill_betweenx([0, 1], T_c1, T_arr[-1], alpha=0.2, color='seagreen',
                      label='Cubic')
    tilt_total = np.sqrt(tilt_x**2 + tilt_y**2 + tilt_z**2)
    ax.plot(T_arr, tilt_total / np.max(tilt_total + 0.01), '-',
            color='black', linewidth=2, label='Total tilt (normalized)')
    ax.set_xlabel('Temperature (K)', fontsize=13)
    ax.set_ylabel('Normalized order parameter', fontsize=13)
    ax.set_title('Phase Regions from Tilt Analysis', fontsize=14)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(-0.05, 1.1)

    fig.suptitle('Perovskite Phase Transition: Octahedral Tilting', fontsize=15)
    fig.tight_layout()
    fig.savefig(os.path.join(OUTPUT_DIR, 'tilt_order_parameter.png'),
                dpi=200, bbox_inches='tight')
    plt.close()
    print(f"Saved: {OUTPUT_DIR}/tilt_order_parameter.png")

print("\nAnalysis complete.")
```

## Key Parameters

| Parameter | Typical Value | Notes |
|---|---|---|
| Neighbor cutoff (Steinhardt) | 1.2-1.5 x r_NN | First coordination shell. Material-dependent. Too large includes 2nd shell. |
| l-value (Steinhardt) | 4, 6 | l=6 best for solid/liquid. l=4 for FCC/BCC/HCP discrimination. |
| q6_bar threshold (solid/liquid) | 0.30-0.40 | System-dependent. 0.35 works for many metals. Calibrate against known solid/liquid. |
| B-X bond cutoff (octahedral) | 2.0-2.8 Angstrom | Depends on B-site cation and anion. Check nearest-neighbor distances. |
| Trajectory frame interval | Every 10-100 steps | More frequent for fast transitions, less frequent for equilibrium. |
| Warren-Cowley cutoff | 1st shell distance | Use same cutoff as nearest-neighbor RDF peak. |

### Reference q4, q6 Values for Structure Identification

| Structure | q4 | q6 | w4 | w6 |
|---|---|---|---|---|
| FCC | 0.191 | 0.575 | -0.159 | -0.013 |
| HCP | 0.097 | 0.485 | +0.134 | -0.012 |
| BCC | 0.036 | 0.511 | +0.159 | +0.013 |
| Simple cubic | 0.764 | 0.354 | +0.159 | +0.013 |
| Diamond | 0.509 | 0.629 | -0.159 | -0.013 |
| Icosahedral | 0.0 | 0.663 | 0.0 | -0.170 |
| Liquid | ~0.01-0.04 | ~0.25-0.35 | ~0 | ~0 |

## Interpreting Results

### Steinhardt Parameters for Melting

- **Global q6 drop**: A sharp decrease in mean q6 signals melting. Transition region typically 10-50 K wide depending on system size.
- **Solid fraction**: Fraction of atoms with q6_bar above threshold. Drops from ~1.0 (solid) to ~0.0 (liquid) at Tm.
- **q4-q6 scatter plot**: Solid atoms cluster near crystal reference points; liquid atoms scatter toward the origin.
- **Hysteresis**: Superheating (heating solid past Tm) and supercooling (cooling liquid below Tm) are common in simulations. Two-phase coexistence avoids this.

### Ferroelectric Polarization

- **Paraelectric phase**: P = 0 (time average). Individual snapshots may show small fluctuating P due to thermal noise.
- **Ferroelectric phase**: P > 0 (time average). Direction of P determines the domain orientation.
- **Near Tc**: Large fluctuations in P. Correlation time diverges (critical slowing down).
- **Domain structure**: In large simulations, multiple domains may form. Local P varies in space.

### Octahedral Tilts in Perovskites

- **Cubic phase (Pm-3m)**: All tilt angles = 0. No octahedral rotations.
- **Tetragonal (I4/mcm or P4/mbm)**: Tilts around one axis. Glazer: a0a0c- or a0a0c+.
- **Orthorhombic (Pnma)**: Tilts around two or three axes. Glazer: a-b+a- (most common).
- **Rhombohedral (R-3c)**: Equal tilts around all three axes. Glazer: a-a-a-.
- **Temperature sequence (typical)**: Orthorhombic -> Tetragonal -> Cubic with increasing T.

### Warren-Cowley Short-Range Order

- **alpha_1 = 0**: Perfectly random alloy (ideal solid solution).
- **alpha_1 > 0**: Clustering tendency. Like atoms prefer like neighbors.
- **alpha_1 < 0**: Ordering tendency. Unlike atoms prefer unlike neighbors.
- **alpha_1 = -c_A/(1-c_A)**: Maximum ordering (every A atom surrounded by B atoms).

## Common Issues

| Problem | Symptom | Solution |
|---|---|---|
| Wrong neighbor cutoff | Too many or too few neighbors per atom | Plot RDF first, set cutoff at the minimum between 1st and 2nd peaks. |
| q6 values all similar (no solid/liquid distinction) | Cutoff too large includes 2nd shell | Reduce cutoff to first coordination shell only. |
| q6_bar threshold not discriminating | Mixed solid-liquid classification | Plot q6_bar histogram for known solid and known liquid frames. Choose threshold at the valley between peaks. |
| Tilt angles noisy | Large fluctuations obscure transition | Average over longer MD runs. Use running mean over 10-50 frames. Increase supercell size (>80 atoms). |
| Octahedra not found | find_octahedra returns empty list | Adjust bond_cutoff. Check that b_site_element and x_site_element are correct. For distorted perovskites, increase cutoff. |
| Polarization wrong magnitude | P differs from literature by orders of magnitude | Check units (C/m^2, not uC/cm^2). Verify Born effective charges. Ensure reference positions are correct (centrosymmetric phase). |
| Born effective charges not available | Cannot compute polarization from MD | Run a QE phonon calculation (`ph.x` with `epsil = .true.`) for the centrosymmetric phase to get Z*. Or use values from literature. |
| Warren-Cowley diverges | Extreme values for dilute alloys | alpha_1 is noisy when one species concentration is very low. Use larger supercells or ensemble averaging. |
| Phase transition smeared out | Gradual change instead of sharp transition | Finite-size effect. Increase supercell. In simulations, first-order transitions are rounded by finite size; second-order transitions show gradual change inherently. |
