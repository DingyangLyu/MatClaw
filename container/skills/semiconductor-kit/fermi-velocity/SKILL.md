# Fermi Velocity

## When to Use

- You need the Fermi velocity v_F of a metal, semimetal, or Dirac material.
- You are studying graphene, topological surface states, or Weyl semimetals where the Fermi velocity characterizes the linear dispersion.
- You need v_F as input for transport models (Boltzmann equation, Drude model).
- You want to quantify how "relativistic" the low-energy quasiparticles are (v_F/c ratio).
- You need band velocities dE/dk at the Fermi level for computing conductivity.
- You want to characterize Dirac or Weyl cones by their velocity.

## Method Selection

| Method | Tool | Strengths | Limitations |
|---|---|---|---|
| QE band slope | pw.x + Python | Available in container, direct | Requires band structure along k-path |
| VASP band slope | VASP + Python | Standard VASP workflow | Requires VASP license |
| QE + Wannier90 | pw.x + wannier90 | Accurate interpolation | More complex setup |
| pymatgen band velocity | pymatgen | Automated | Needs parsed band structure |
| Finite difference | Python | Simple, general | Sensitive to k-spacing |

## Prerequisites

- A completed band structure calculation along high-symmetry k-paths (QE or VASP).
- Dense k-point sampling near the Fermi level crossing or Dirac point.
- Python packages: `numpy`, `scipy`, `matplotlib`.
- For QE: bands.x output or pw.x bands output.
- For VASP: EIGENVAL or vasprun.xml.

## Background

### Fermi Velocity Definition

The Fermi velocity is the group velocity of electrons at the Fermi level:

```
v_F = (1/hbar) * |dE/dk|_{E=E_F}
```

For a free electron gas: v_F = hbar * k_F / m_e. For real materials, v_F depends on the band structure and can be highly anisotropic.

### Dirac Materials

In Dirac materials (graphene, topological insulators), the dispersion near the Dirac point is linear:

```
E(k) = E_D +/- hbar * v_F * |k - k_D|
```

where E_D is the Dirac point energy and k_D is the Dirac point in k-space. The Fermi velocity v_F is the slope of the Dirac cone and is constant (energy-independent) near the Dirac point.

### Typical Values

| Material | v_F (m/s) | v_F/c | Notes |
|---|---|---|---|
| Free electron (Cu) | ~1.6e6 | 0.005 | Typical metal |
| Graphene | ~1.0e6 | 0.003 | Dirac cone at K point |
| Bi2Se3 surface | ~5.0e5 | 0.002 | Topological surface state |
| Weyl semimetal (TaAs) | ~3-5e5 | 0.001-0.002 | Weyl cone |
| Si (VBM) | ~1.5e5 | 0.0005 | Parabolic band |

## Detailed Steps

### Overview

```
Method A: QE band structure slope analysis
  Step 1: Band structure with dense k-path near Fermi crossing
  Step 2: Find Fermi level crossings
  Step 3: Compute dE/dk at crossings
  Step 4: Plot and report v_F

Method B: VASP band structure slope analysis
  Step 1: Parse EIGENVAL/vasprun.xml
  Step 2: Same analysis as Method A

Method C: Graphene Dirac cone (complete example)
  Full QE workflow for graphene v_F
```

---

### Method A: QE Band Structure Fermi Velocity

#### Step A1: Parse Band Structure and Compute Velocities

```python
#!/usr/bin/env python3
"""
Compute Fermi velocity from QE band structure.

Workflow:
1. Parse eigenvalues and k-points from bands calculation
2. Find bands that cross the Fermi level
3. Compute dE/dk at the crossing points
4. Convert to Fermi velocity: v_F = (1/hbar) * dE/dk
"""
import re
import os
import json
import numpy as np
from scipy.interpolate import UnivariateSpline
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


# ── Physical constants ─────────────────────────────────────────────
HBAR_SI = 1.054571817e-34       # J*s
EV_TO_J = 1.602176634e-19       # J/eV
ANGSTROM_TO_M = 1e-10
C_LIGHT = 2.99792458e8          # m/s


def parse_bands_gnu(filename):
    """
    Parse bands.x .dat.gnu output file.

    Returns
    -------
    k_distances : ndarray (nk,) in units of 2*pi/a
    energies : ndarray (nk, nbnd) in eV
    """
    with open(filename, 'r') as f:
        content = f.read()

    blocks = content.strip().split('\n\n')
    bands_data = []
    for block in blocks:
        lines = block.strip().split('\n')
        band = []
        for line in lines:
            vals = line.split()
            if len(vals) >= 2:
                band.append([float(vals[0]), float(vals[1])])
        if band:
            bands_data.append(np.array(band))

    nk = len(bands_data[0])
    nbnd = len(bands_data)
    k_distances = bands_data[0][:, 0]
    energies = np.zeros((nk, nbnd))
    for ib, band in enumerate(bands_data):
        energies[:, ib] = band[:, 1]

    return k_distances, energies


def find_fermi_crossings(k_distances, energies, fermi_energy=0.0,
                          tolerance=0.2):
    """
    Find bands that cross the Fermi level and locate the crossing points.

    Parameters
    ----------
    k_distances : ndarray (nk,)
        K-path distances.
    energies : ndarray (nk, nbnd)
        Band energies.
    fermi_energy : float
        Fermi energy in eV.
    tolerance : float
        Energy window around E_F to search for crossings (eV).

    Returns
    -------
    crossings : list of dict
        Each crossing has: band_index, k_crossing, E_crossing,
        dEdk (eV per k-unit), v_F (m/s).
    """
    nk, nbnd = energies.shape
    crossings = []

    for ib in range(nbnd):
        band = energies[:, ib]

        # Check if band crosses Fermi level
        above = band > fermi_energy
        below = band <= fermi_energy

        for i in range(nk - 1):
            if (above[i] and below[i+1]) or (below[i] and above[i+1]):
                # Linear interpolation to find crossing point
                E1, E2 = band[i], band[i+1]
                k1, k2 = k_distances[i], k_distances[i+1]

                # k at crossing
                if abs(E2 - E1) > 1e-10:
                    frac = (fermi_energy - E1) / (E2 - E1)
                    k_cross = k1 + frac * (k2 - k1)
                else:
                    k_cross = (k1 + k2) / 2

                crossings.append({
                    'band_index': ib,
                    'k_crossing': float(k_cross),
                    'k_index_left': i,
                    'k_index_right': i + 1,
                    'E_crossing': float(fermi_energy),
                })

    return crossings


def compute_fermi_velocity(k_distances, energies, crossings,
                            recip_lattice_scale_inv_m,
                            n_fit_points=5):
    """
    Compute Fermi velocity at each crossing by fitting local slope.

    Parameters
    ----------
    k_distances : ndarray (nk,)
        K-path distances (dimensionless or in 2*pi/a units).
    energies : ndarray (nk, nbnd)
        Band energies in eV.
    crossings : list of dict
        From find_fermi_crossings().
    recip_lattice_scale_inv_m : float
        Scale factor to convert k-distance units to 1/m.
        For QE bands.x output: this is 2*pi/a * 1e10 if a in Angstrom.
    n_fit_points : int
        Number of points on each side of crossing for linear fit.

    Returns
    -------
    crossings : list of dict (updated with dEdk, v_F fields)
    """
    nk = len(k_distances)

    for crossing in crossings:
        ib = crossing['band_index']
        i_left = crossing['k_index_left']

        # Select points around crossing for fitting
        i_start = max(0, i_left - n_fit_points)
        i_end = min(nk, i_left + n_fit_points + 2)
        indices = list(range(i_start, i_end))

        k_sel = k_distances[indices]  # dimensionless
        E_sel = energies[indices, ib]  # eV

        # Linear fit: E = a * k + b
        if len(k_sel) >= 2:
            coeffs = np.polyfit(k_sel, E_sel, 1)
            dEdk = coeffs[0]  # eV per k-unit
        else:
            dEdk = 0.0

        # Convert to SI:
        # dE/dk in eV / (1/m) = eV * m
        dEdk_SI = dEdk * EV_TO_J / recip_lattice_scale_inv_m

        # v_F = (1/hbar) * |dE/dk|
        v_F = abs(dEdk_SI) / HBAR_SI  # m/s

        crossing['dEdk_eV_per_kunit'] = float(dEdk)
        crossing['v_F_m_s'] = float(v_F)
        crossing['v_F_over_c'] = float(v_F / C_LIGHT)

    return crossings


def plot_fermi_velocity(k_distances, energies, crossings,
                         fermi_energy=0.0,
                         k_labels=None, k_label_positions=None,
                         output='fermi_velocity.png'):
    """Plot band structure with Fermi level crossings and v_F annotated."""
    fig, ax = plt.subplots(figsize=(10, 6))

    nk, nbnd = energies.shape

    # Plot all bands
    for ib in range(nbnd):
        ax.plot(k_distances, energies[:, ib] - fermi_energy,
                'b-', linewidth=1, alpha=0.6)

    # Fermi level
    ax.axhline(y=0, color='gray', linestyle='--', linewidth=1,
               label='$E_F$')

    # Mark crossings
    for crossing in crossings:
        k_c = crossing['k_crossing']
        v_F = crossing.get('v_F_m_s', 0)
        ax.plot(k_c, 0, 'ro', markersize=10, zorder=5)

        # Annotate with v_F
        v_F_str = f"$v_F$ = {v_F:.2e} m/s"
        ax.annotate(v_F_str, (k_c, 0),
                     textcoords="offset points", xytext=(15, 15),
                     fontsize=8, color='red',
                     arrowprops=dict(arrowstyle='->', color='red',
                                      lw=0.5),
                     bbox=dict(boxstyle='round,pad=0.3',
                               facecolor='lightyellow', alpha=0.9))

    # Labels
    if k_labels and k_label_positions:
        ax.set_xticks(k_label_positions)
        ax.set_xticklabels(k_labels, fontsize=12)
        for pos in k_label_positions:
            ax.axvline(x=pos, color='gray', linewidth=0.5, alpha=0.5)

    ax.set_xlabel('Wave vector', fontsize=13)
    ax.set_ylabel('$E - E_F$ (eV)', fontsize=13)
    ax.set_title('Band Structure with Fermi Velocity', fontsize=14)
    ax.set_xlim(k_distances[0], k_distances[-1])
    ax.set_ylim(-3, 3)
    ax.grid(True, axis='y', alpha=0.2)

    plt.tight_layout()
    plt.savefig(output, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output}")


# ── Main workflow ──────────────────────────────────────────────────

print("=" * 60)
print("Fermi Velocity Calculation")
print("=" * 60)

# Try to load bands data
bands_file = None
for f in ['bands.dat.gnu', 'si_bands.dat.gnu', 'graphene_bands.dat.gnu']:
    if os.path.exists(f):
        bands_file = f
        break

if bands_file:
    k_dist, energies = parse_bands_gnu(bands_file)
    print(f"\nLoaded: {bands_file}")
    print(f"  k-points: {len(k_dist)}, bands: {energies.shape[1]}")

    # Find crossings (assumes Fermi = 0 if bands.x adjusted it)
    crossings = find_fermi_crossings(k_dist, energies, fermi_energy=0.0)
    print(f"\nFound {len(crossings)} Fermi level crossings")

    if crossings:
        # Scale factor: need to know lattice parameter
        # For QE bands.x output, k is in units of 2*pi/a
        a_bohr = 10.2631  # Si lattice constant in Bohr
        a_m = a_bohr * 5.29177e-11
        recip_scale = 2 * np.pi / a_m  # 1/m

        crossings = compute_fermi_velocity(
            k_dist, energies, crossings,
            recip_lattice_scale_inv_m=recip_scale,
            n_fit_points=5
        )

        print("\n--- Fermi Velocity Results ---")
        for i, c in enumerate(crossings):
            print(f"  Crossing {i+1}:")
            print(f"    Band:  {c['band_index']}")
            print(f"    k:     {c['k_crossing']:.4f}")
            print(f"    v_F:   {c['v_F_m_s']:.4e} m/s")
            print(f"    v_F/c: {c['v_F_over_c']:.6f}")

        plot_fermi_velocity(k_dist, energies, crossings,
                             output='fermi_velocity.png')

        # Save
        with open('fermi_velocity_results.json', 'w') as f:
            json.dump(crossings, f, indent=2)
        print("\nResults saved: fermi_velocity_results.json")
    else:
        print("No Fermi crossings found. Material may be a semiconductor.")
        print("Fermi velocity is only defined for metals and semimetals.")
else:
    print("No bands output file found.")
    print("Run a band structure calculation first.")
```

---

### Method B: VASP Fermi Velocity

```python
#!/usr/bin/env python3
"""
Compute Fermi velocity from VASP band structure via pymatgen.
"""
import os
import numpy as np
import json

HBAR_SI = 1.054571817e-34
EV_TO_J = 1.602176634e-19
C_LIGHT = 2.99792458e8


def fermi_velocity_from_vasprun(vasprun_path='vasprun.xml'):
    """
    Extract Fermi velocity from VASP vasprun.xml.
    """
    from pymatgen.io.vasp.outputs import Vasprun
    from pymatgen.electronic_structure.core import Spin

    vrun = Vasprun(vasprun_path, parse_projected_eigen=False)
    bs = vrun.get_band_structure(line_mode=True)

    if bs.is_metal():
        print("Material is metallic -- computing Fermi velocities.")
    else:
        print(f"Material has a gap of {bs.get_band_gap()['energy']:.3f} eV.")
        print("Fermi velocity is not well-defined for semiconductors.")
        return None

    efermi = bs.efermi
    structure = bs.structure
    recip = structure.lattice.reciprocal_lattice

    results = []

    for spin in bs.bands:
        for band_idx in range(len(bs.bands[spin])):
            band_energies = bs.bands[spin][band_idx]

            # Check if this band crosses E_F
            above = band_energies > efermi
            below = band_energies <= efermi

            for i in range(len(band_energies) - 1):
                if (above[i] and below[i+1]) or (below[i] and above[i+1]):
                    # Get k-points in Cartesian
                    k1 = bs.kpoints[i].cart_coords  # 1/Angstrom
                    k2 = bs.kpoints[i+1].cart_coords
                    E1 = band_energies[i]
                    E2 = band_energies[i+1]

                    dk = np.linalg.norm(k2 - k1) * 1e10  # 1/m
                    dE = abs(E2 - E1)  # eV

                    if dk > 0:
                        dEdk_SI = dE * EV_TO_J / dk  # J*m
                        v_F = dEdk_SI / HBAR_SI  # m/s

                        results.append({
                            'band': int(band_idx),
                            'spin': str(spin),
                            'k_index': int(i),
                            'v_F_m_s': float(v_F),
                            'v_F_over_c': float(v_F / C_LIGHT),
                        })

    if results:
        print(f"\nFound {len(results)} Fermi level crossings")
        # Average Fermi velocity
        v_avg = np.mean([r['v_F_m_s'] for r in results])
        print(f"Average v_F: {v_avg:.4e} m/s")
        print(f"v_F / c: {v_avg/C_LIGHT:.6f}")

    return results


if os.path.exists('vasprun.xml'):
    results = fermi_velocity_from_vasprun('vasprun.xml')
    if results:
        with open('fermi_velocity_vasp.json', 'w') as f:
            json.dump(results, f, indent=2)
```

---

### Method C: Graphene Dirac Cone (Complete Example)

#### Step C1: Build Graphene Structure and Run SCF

```python
#!/usr/bin/env python3
"""
Complete workflow to compute the Fermi velocity of graphene.
Graphene has a Dirac cone at the K point with v_F ~ 1e6 m/s.
"""
import os
import subprocess
import numpy as np

PSEUDO_DIR = os.path.abspath("./pseudo")
OUTDIR = os.path.abspath("./tmp_graphene")
os.makedirs(PSEUDO_DIR, exist_ok=True)
os.makedirs(OUTDIR, exist_ok=True)

# ── Download C pseudopotential ─────────────────────────────────────
import urllib.request
pp_file = "C.pbe-n-kjpaw_psl.1.0.0.UPF"
pp_path = os.path.join(PSEUDO_DIR, pp_file)
if not os.path.exists(pp_path):
    url = f"https://pseudopotentials.quantum-espresso.org/upf_files/{pp_file}"
    print(f"Downloading {pp_file} ...")
    urllib.request.urlretrieve(url, pp_path)
    print("Done.")

# ── SCF calculation ────────────────────────────────────────────────
# Graphene: hexagonal cell, a = 2.46 Angstrom, vacuum = 15 Angstrom
a_ang = 2.46
c_ang = 15.0  # Vacuum spacing
a_bohr = a_ang * 1.8897259886
c_bohr = c_ang * 1.8897259886

scf_input = f"""&CONTROL
    calculation   = 'scf'
    prefix        = 'graphene'
    pseudo_dir    = '{PSEUDO_DIR}'
    outdir        = '{OUTDIR}'
    tprnfor       = .true.
    tstress       = .true.
    verbosity     = 'high'
/

&SYSTEM
    ibrav         = 4
    celldm(1)     = {a_bohr:.6f}
    celldm(3)     = {c_bohr/a_bohr:.6f}
    nat           = 2
    ntyp          = 1
    ecutwfc       = 60.0
    ecutrho       = 480.0
    occupations   = 'smearing'
    smearing      = 'mv'
    degauss       = 0.01
    nbnd          = 12
/

&ELECTRONS
    conv_thr      = 1.0d-10
    mixing_beta   = 0.7
/

ATOMIC_SPECIES
  C  12.011  {pp_file}

ATOMIC_POSITIONS {{crystal}}
  C   0.333333333   0.666666667   0.0
  C   0.666666667   0.333333333   0.0

K_POINTS {{automatic}}
  24 24 1  0 0 0
"""

with open("graphene_scf.in", "w") as f:
    f.write(scf_input)

print("Running graphene SCF ...")
result = subprocess.run(
    ["mpirun", "-np", "4", "pw.x", "-npool", "2", "-in", "graphene_scf.in"],
    capture_output=True, text=True, timeout=1800
)
with open("graphene_scf.out", "w") as f:
    f.write(result.stdout)

if "convergence has been achieved" in result.stdout:
    print("SCF converged.")
else:
    print("SCF failed or did not converge. Check graphene_scf.out")
```

#### Step C2: Dense Band Calculation Near K Point

```python
#!/usr/bin/env python3
"""
Run a dense band structure calculation near the K point for graphene.
The Dirac cone is at K = (1/3, 1/3, 0) in hexagonal reciprocal space.
"""
import os
import subprocess
import numpy as np

PSEUDO_DIR = os.path.abspath("./pseudo")
OUTDIR = os.path.abspath("./tmp_graphene")
a_ang = 2.46
a_bohr = a_ang * 1.8897259886
c_bohr = 15.0 * 1.8897259886

pp_file = "C.pbe-n-kjpaw_psl.1.0.0.UPF"

# Dense k-path through K point: Gamma-K-M with many points near K
bands_input = f"""&CONTROL
    calculation   = 'bands'
    prefix        = 'graphene'
    pseudo_dir    = '{PSEUDO_DIR}'
    outdir        = '{OUTDIR}'
    verbosity     = 'high'
/

&SYSTEM
    ibrav         = 4
    celldm(1)     = {a_bohr:.6f}
    celldm(3)     = {c_bohr/a_bohr:.6f}
    nat           = 2
    ntyp          = 1
    ecutwfc       = 60.0
    ecutrho       = 480.0
    occupations   = 'smearing'
    smearing      = 'mv'
    degauss       = 0.01
    nbnd          = 12
/

&ELECTRONS
    conv_thr      = 1.0d-10
/

ATOMIC_SPECIES
  C  12.011  {pp_file}

ATOMIC_POSITIONS {{crystal}}
  C   0.333333333   0.666666667   0.0
  C   0.666666667   0.333333333   0.0

K_POINTS {{crystal_b}}
4
  0.0000  0.0000  0.0000  40   ! Gamma
  0.3333  0.3333  0.0000  80   ! K (dense sampling)
  0.5000  0.0000  0.0000  40   ! M
  0.0000  0.0000  0.0000   0   ! Gamma
"""

with open("graphene_bands.in", "w") as f:
    f.write(bands_input)

print("Running graphene bands ...")
result = subprocess.run(
    ["mpirun", "-np", "4", "pw.x", "-npool", "4", "-in", "graphene_bands.in"],
    capture_output=True, text=True, timeout=3600
)
with open("graphene_bands.out", "w") as f:
    f.write(result.stdout)

if result.returncode == 0:
    print("Bands calculation completed.")
else:
    print("Bands calculation failed.")

# ── Run bands.x post-processing ───────────────────────────────────
bandsx_input = f"""&BANDS
    prefix  = 'graphene'
    outdir  = '{OUTDIR}'
    filband = 'graphene_bands.dat'
/
"""

with open("graphene_bandsx.in", "w") as f:
    f.write(bandsx_input)

result2 = subprocess.run(
    ["bands.x", "-in", "graphene_bandsx.in"],
    capture_output=True, text=True, timeout=120
)
with open("graphene_bandsx.out", "w") as f:
    f.write(result2.stdout)

if os.path.exists("graphene_bands.dat.gnu"):
    print("bands.x completed. Output: graphene_bands.dat.gnu")
else:
    print("bands.x may have failed. Check graphene_bandsx.out")
```

#### Step C3: Extract Graphene v_F

```python
#!/usr/bin/env python3
"""
Extract Fermi velocity from graphene Dirac cone.
The linear dispersion E = +/- hbar * v_F * |k - K| gives
v_F from the slope of the bands near the K point.
"""
import os
import json
import numpy as np
from scipy.optimize import curve_fit
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

HBAR_SI = 1.054571817e-34
EV_TO_J = 1.602176634e-19
C_LIGHT = 2.99792458e8


def parse_bands_gnu(filename):
    """Parse bands.x .gnu output."""
    with open(filename, 'r') as f:
        content = f.read()

    blocks = content.strip().split('\n\n')
    bands_data = []
    for block in blocks:
        lines = block.strip().split('\n')
        band = []
        for line in lines:
            vals = line.split()
            if len(vals) >= 2:
                band.append([float(vals[0]), float(vals[1])])
        if band:
            bands_data.append(np.array(band))

    nk = len(bands_data[0])
    nbnd = len(bands_data)
    k_distances = bands_data[0][:, 0]
    energies = np.zeros((nk, nbnd))
    for ib, band in enumerate(bands_data):
        energies[:, ib] = band[:, 1]

    return k_distances, energies


def fit_dirac_cone(k_distances, energies, fermi_energy,
                    k_dirac_distance, n_fit=15):
    """
    Fit E = E_D +/- hbar * v_F * |k - k_D| near the Dirac point.

    Parameters
    ----------
    k_distances : ndarray (nk,) in QE k-units
    energies : ndarray (nk, nbnd) in eV
    fermi_energy : float in eV
    k_dirac_distance : float
        Approximate k-distance of the Dirac point along the path.
    n_fit : int
        Number of k-points on each side of Dirac point for fitting.

    Returns
    -------
    v_F : float in m/s (from upper cone)
    v_F_lower : float in m/s (from lower cone)
    dirac_energy : float in eV
    """
    # Find Dirac point: where upper and lower Dirac bands touch
    # Near K point, look for the pair of bands closest to E_F

    nk, nbnd = energies.shape

    # Find k-index closest to Dirac point
    k_D_idx = np.argmin(np.abs(k_distances - k_dirac_distance))

    # Find bands near Fermi at K point
    E_at_K = energies[k_D_idx, :]
    # Sort by distance from Fermi
    band_distances = np.abs(E_at_K - fermi_energy)
    closest_bands = np.argsort(band_distances)[:2]
    upper_band = max(closest_bands)
    lower_band = min(closest_bands)

    E_dirac = (E_at_K[upper_band] + E_at_K[lower_band]) / 2
    print(f"  Dirac point energy: {E_dirac:.4f} eV")
    print(f"  Upper Dirac band: {upper_band}, Lower: {lower_band}")

    # Fit linear dispersion on both sides of K
    i_start = max(0, k_D_idx - n_fit)
    i_end = min(nk, k_D_idx + n_fit + 1)
    indices = list(range(i_start, i_end))

    k_sel = k_distances[indices]
    E_upper = energies[indices, upper_band]
    E_lower = energies[indices, lower_band]

    # Linear fit to upper cone: |E - E_D| vs |k - k_D|
    k_centered = k_sel - k_distances[k_D_idx]
    dE_upper = E_upper - E_dirac
    dE_lower = E_lower - E_dirac

    # Fit on positive-k side
    pos_mask = k_centered > 0.001
    if np.any(pos_mask):
        slope_upper_pos = np.polyfit(
            np.abs(k_centered[pos_mask]),
            np.abs(dE_upper[pos_mask]), 1)[0]
        slope_lower_pos = np.polyfit(
            np.abs(k_centered[pos_mask]),
            np.abs(dE_lower[pos_mask]), 1)[0]
    else:
        slope_upper_pos = 0
        slope_lower_pos = 0

    # Fit on negative-k side
    neg_mask = k_centered < -0.001
    if np.any(neg_mask):
        slope_upper_neg = np.polyfit(
            np.abs(k_centered[neg_mask]),
            np.abs(dE_upper[neg_mask]), 1)[0]
        slope_lower_neg = np.polyfit(
            np.abs(k_centered[neg_mask]),
            np.abs(dE_lower[neg_mask]), 1)[0]
    else:
        slope_upper_neg = 0
        slope_lower_neg = 0

    # Average slopes
    slope_upper = (abs(slope_upper_pos) + abs(slope_upper_neg)) / 2
    slope_lower = (abs(slope_lower_pos) + abs(slope_lower_neg)) / 2

    # Convert slope from eV / k-unit to m/s
    # Need k-unit to 1/m conversion
    # For graphene: a = 2.46 Angstrom, QE k-path is in 2*pi/a units
    a_ang = 2.46
    a_m = a_ang * 1e-10
    k_scale = 2 * np.pi / a_m  # 1/m per k-unit

    # dE/dk in SI: (eV * 1.6e-19) / (1/m)
    v_F_upper = slope_upper * EV_TO_J / (k_scale * HBAR_SI)
    v_F_lower = slope_lower * EV_TO_J / (k_scale * HBAR_SI)

    return v_F_upper, v_F_lower, E_dirac, upper_band, lower_band


# ── Main ───────────────────────────────────────────────────────────

print("=" * 60)
print("Graphene Dirac Cone Fermi Velocity")
print("=" * 60)

bands_file = "graphene_bands.dat.gnu"
if os.path.exists(bands_file):
    k_dist, energies = parse_bands_gnu(bands_file)
    print(f"\nLoaded: {bands_file}")
    print(f"  k-points: {len(k_dist)}, bands: {energies.shape[1]}")

    # Fermi energy (from SCF or set to 0 if bands.x adjusted it)
    fermi_e = 0.0

    # K point is at k-distance ~ 0.667 * max(k) for Gamma-K-M-Gamma path
    # More precisely, find it from the path definition
    k_K_approx = k_dist[-1] * 0.33  # Rough estimate

    # Better: look for where two bands touch near E_F
    nk, nbnd = energies.shape
    for ib in range(nbnd - 1):
        gap = energies[:, ib + 1] - energies[:, ib]
        min_gap_idx = np.argmin(np.abs(gap))
        if np.abs(gap[min_gap_idx]) < 0.1 and \
           np.abs(energies[min_gap_idx, ib] - fermi_e) < 1.0:
            k_K_approx = k_dist[min_gap_idx]
            print(f"\n  Dirac point detected at k = {k_K_approx:.4f}")
            print(f"  Gap at Dirac point: {gap[min_gap_idx]:.4f} eV")
            break

    v_F_upper, v_F_lower, E_D, ub, lb = fit_dirac_cone(
        k_dist, energies, fermi_e, k_K_approx, n_fit=20)

    print(f"\n{'='*60}")
    print("RESULTS")
    print(f"{'='*60}")
    print(f"  Upper cone v_F: {v_F_upper:.4e} m/s  "
          f"(v_F/c = {v_F_upper/C_LIGHT:.6f})")
    print(f"  Lower cone v_F: {v_F_lower:.4e} m/s  "
          f"(v_F/c = {v_F_lower/C_LIGHT:.6f})")
    print(f"  Average v_F:    {(v_F_upper+v_F_lower)/2:.4e} m/s")
    print(f"  Dirac energy:   {E_D:.4f} eV")
    print(f"\n  Expected: v_F ~ 1.0e6 m/s for graphene")

    # Plot
    fig, ax = plt.subplots(figsize=(8, 6))
    for ib in range(energies.shape[1]):
        color = 'red' if ib in [ub, lb] else 'blue'
        lw = 2.5 if ib in [ub, lb] else 1.0
        alpha = 1.0 if ib in [ub, lb] else 0.4
        ax.plot(k_dist, energies[:, ib] - fermi_e, color=color,
                linewidth=lw, alpha=alpha)

    ax.axhline(y=0, color='gray', linestyle='--', linewidth=0.8)
    ax.set_xlabel('Wave vector', fontsize=13)
    ax.set_ylabel('$E - E_F$ (eV)', fontsize=13)
    ax.set_title(f'Graphene Dirac Cone  '
                 f'($v_F$ = {(v_F_upper+v_F_lower)/2:.2e} m/s)',
                 fontsize=14)
    ax.set_ylim(-6, 6)
    ax.grid(True, axis='y', alpha=0.2)
    plt.tight_layout()
    plt.savefig('graphene_dirac_cone.png', dpi=200, bbox_inches='tight')
    plt.close()
    print("\nSaved: graphene_dirac_cone.png")

    # Save results
    results = {
        'v_F_upper_m_s': v_F_upper,
        'v_F_lower_m_s': v_F_lower,
        'v_F_average_m_s': (v_F_upper + v_F_lower) / 2,
        'v_F_over_c': (v_F_upper + v_F_lower) / 2 / C_LIGHT,
        'dirac_energy_eV': E_D,
    }
    with open('graphene_vF_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    print("Results saved: graphene_vF_results.json")
else:
    print(f"\n{bands_file} not found. Run the graphene bands workflow first.")
```

---

## Key Parameters

| Parameter | Typical Value | Notes |
|---|---|---|
| K-path density near E_F | 50-100 points/segment | Dense sampling needed for accurate slope |
| n_fit_points | 5-20 | Number of points for linear fit. Too many = includes non-linear region. |
| Smearing (QE) | 0.005-0.01 Ry | Small smearing for metals; helps convergence |
| K-grid (SCF) | 24x24x1 (2D), 16x16x16 (3D) | Must be converged for accurate Fermi level |
| ecutwfc | 40-80 Ry | Standard convergence requirement |
| nbnd | >= 2x occupied | Include bands around Fermi level |

## Interpreting Results

### Fermi Velocity Magnitudes

- **v_F ~ 1e6 m/s**: Graphene-like Dirac cone. Very fast electrons.
- **v_F ~ 5e5 m/s**: Typical topological surface state (Bi2Se3).
- **v_F ~ 1-2e6 m/s**: Typical metal (Cu, Au, Al).
- **v_F ~ 1e5 m/s**: Slow Fermi velocity (heavy fermion systems, flat bands).

### Dirac Materials

- Linear fit should give constant v_F over a range of |k - k_D|. If v_F varies strongly, the dispersion is not truly linear (trigonal warping, higher-order terms).
- Upper and lower cone velocities should be equal for ideal Dirac materials. Asymmetry indicates particle-hole asymmetry.
- Compare with experiment: ARPES directly measures v_F from the slope of the dispersion.

### Anisotropy

- v_F can be direction-dependent (anisotropic Fermi surface).
- Report v_F along specific crystal directions.
- For Dirac cones, check isotropy by fitting in multiple azimuthal directions around the Dirac point.

## Common Issues

| Problem | Solution |
|---|---|
| **v_F is zero or very small** | Material is a semiconductor. Fermi velocity requires band crossings at E_F. |
| **v_F varies strongly along the path** | Non-linear dispersion. Report v_F at the specific k-point of interest. |
| **Upper and lower cone v_F differ** | Particle-hole asymmetry. Physical for real materials. Report both. |
| **Cannot identify Dirac point** | Dirac point may be gapped by spin-orbit coupling (topological insulator gap). Look for the minimum gap location. |
| **Fermi level not at Dirac point** | Common in doped graphene or with substrate. v_F is still the cone slope, but carriers are at E_F, not E_Dirac. |
| **v_F depends on fitting range** | Reduce n_fit_points to stay in the linear regime. Bands become non-linear far from the cone. |
| **Too few k-points near crossing** | Increase k-path density near the Fermi level crossing (use K_POINTS {crystal_b} with many points). |
| **Negative slope for v_F** | v_F is defined as |dE/dk|/hbar, always positive. Negative slope just means downward-dispersing band. |
