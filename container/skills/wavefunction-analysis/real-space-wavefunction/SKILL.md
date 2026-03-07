# Real-Space Wavefunction Visualization

## When to Use

- Visualize the spatial distribution of a specific electronic state (band, k-point).
- Identify the character of defect levels, surface states, or band-edge states.
- Examine orbital hybridization patterns in real space.
- Generate partial charge density (|psi|^2) for selected bands.
- Corresponds to VASPKIT tasks 511--514.

## Method Selection

| Criterion | QE (pp.x) | VASP (PARCHG) | Python post-processing |
|---|---|---|---|
| Band-resolved |psi|^2 | `plot_num=7`, select kpoint+band | `LPARD=.TRUE.`, `IBAND`, `KPUSE` | Read cube/PARCHG |
| Output formats | cube, xsf | CHGCAR format | matplotlib slices |
| All-electron density | PAW reconstruction available | PAW reconstruction available | N/A |
| Automation | Python-driven | INCAR flags | Python-driven |

## Prerequisites

- Quantum ESPRESSO 7.5 (`pw.x`, `pp.x`)
- Python packages: `numpy`, `matplotlib`, `ase`
- Converged SCF (or NSCF) calculation
- Knowledge of which band index and k-point to visualize
- For VASP: WAVECAR from a calculation, INCAR with `LPARD=.TRUE.`

---

## Detailed Steps

### Method A: QE -- pp.x with plot_num=7

#### Step A1: Identify the Band and K-Point of Interest

```python
#!/usr/bin/env python3
"""
Parse QE SCF/NSCF output to identify band indices and k-points of interest.
For example, find the VBM and CBM band indices.
"""
import re
import numpy as np


def parse_eigenvalues(qe_output, n_kpoints_max=100):
    """
    Parse eigenvalues from QE pw.x output (verbosity='high').

    Returns
    -------
    kpoints : list of np.ndarray
        K-point coordinates (crystal).
    eigenvalues : list of np.ndarray
        Eigenvalues for each k-point (eV).
    occupations : list of np.ndarray
        Occupation numbers for each k-point.
    e_fermi : float
        Fermi energy (eV).
    """
    with open(qe_output, "r") as f:
        content = f.read()

    # Fermi energy
    e_fermi = 0.0
    m = re.search(r"the Fermi energy is\s+([-\d.]+)", content)
    if m:
        e_fermi = float(m.group(1))
    m = re.search(r"highest occupied, lowest unoccupied.*?:\s+([-\d.]+)\s+([-\d.]+)", content)
    if m:
        e_fermi = (float(m.group(1)) + float(m.group(2))) / 2

    # Parse k-points and eigenvalues
    kpoints = []
    eigenvalues = []
    occupations = []

    # Pattern for k-point header
    kpt_pattern = re.compile(
        r"k\s*=\s*([-\d.]+)\s+([-\d.]+)\s+([-\d.]+)\s+.*?band energies \(ev\):",
        re.DOTALL
    )

    # Split by k-point blocks
    blocks = re.split(r"\n\s*k\s*=", content)

    for block in blocks[1:]:  # skip text before first k-point
        # Parse k-point coordinates
        m = re.match(r"\s*([-\d.]+)\s+([-\d.]+)\s+([-\d.]+)", block)
        if not m:
            continue
        kpt = np.array([float(m.group(1)), float(m.group(2)), float(m.group(3))])
        kpoints.append(kpt)

        # Parse eigenvalues
        eig_section = re.search(r"band energies \(ev\):\s*\n([\s\S]*?)(?:\n\s*occupation)", block)
        if eig_section:
            eigs = [float(x) for x in eig_section.group(1).split()]
            eigenvalues.append(np.array(eigs))

        # Parse occupations
        occ_section = re.search(r"occupation numbers\s*\n([\s\S]*?)(?:\n\s*k\s*=|\Z)", block)
        if occ_section:
            occs = [float(x) for x in occ_section.group(1).split()]
            occupations.append(np.array(occs))

    return kpoints, eigenvalues, occupations, e_fermi


def find_vbm_cbm(eigenvalues, occupations, e_fermi):
    """
    Find VBM and CBM: band index and k-point index.
    """
    vbm_energy = -np.inf
    cbm_energy = np.inf
    vbm_kpt = 0
    vbm_band = 0
    cbm_kpt = 0
    cbm_band = 0

    for ik, (eigs, occs) in enumerate(zip(eigenvalues, occupations)):
        for ib, (e, occ) in enumerate(zip(eigs, occs)):
            if occ > 0.5 and e > vbm_energy:
                vbm_energy = e
                vbm_kpt = ik + 1  # 1-indexed for QE
                vbm_band = ib + 1
            if occ < 0.5 and e < cbm_energy:
                cbm_energy = e
                cbm_kpt = ik + 1
                cbm_band = ib + 1

    return {
        "VBM": {"energy": vbm_energy, "kpoint": vbm_kpt, "band": vbm_band},
        "CBM": {"energy": cbm_energy, "kpoint": cbm_kpt, "band": cbm_band},
        "gap": cbm_energy - vbm_energy,
    }


# Example usage:
import os
if os.path.exists("slab_scf.out"):
    kpts, eigs, occs, ef = parse_eigenvalues("slab_scf.out")
    info = find_vbm_cbm(eigs, occs, ef)
    print(f"Fermi energy: {ef:.4f} eV")
    print(f"VBM: band {info['VBM']['band']}, kpt {info['VBM']['kpoint']}, "
          f"E = {info['VBM']['energy']:.4f} eV")
    print(f"CBM: band {info['CBM']['band']}, kpt {info['CBM']['kpoint']}, "
          f"E = {info['CBM']['energy']:.4f} eV")
    print(f"Gap: {info['gap']:.4f} eV")
else:
    print("No output file found. Run SCF with verbosity='high' first.")
    print("For demo: VBM is typically band 4 at Gamma for Si.")
```

#### Step A2: Extract |psi|^2 with pp.x

```python
#!/usr/bin/env python3
"""
Extract |psi(r)|^2 for a specific band and k-point using pp.x.
plot_num=7 gives the squared modulus of the Kohn-Sham wavefunction.
"""
import os
import subprocess

OUTDIR = os.path.abspath("./tmp")
PREFIX = "silicon"

# Target: band 4 (VBM of Si), k-point 1 (Gamma)
KPOINT = 1   # k-point index (1-based, from the SCF/NSCF k-list)
KBAND = 4    # band index (1-based)
SPIN = 0     # 0 = total, 1 = spin-up, 2 = spin-down (for nspin=2)

# ── pp.x input for |psi|^2 ───────────────────────────────────────
pp_input = f"""&INPUTPP
    prefix   = '{PREFIX}'
    outdir   = '{OUTDIR}'
    filplot  = 'wfc_k{KPOINT}_b{KBAND}.dat'
    plot_num = 7
    kpoint(1)= {KPOINT}
    kband(1) = {KBAND}
    lsign    = .false.
/
&PLOT
    nfile       = 1
    filepp(1)   = 'wfc_k{KPOINT}_b{KBAND}.dat'
    weight(1)   = 1.0
    iflag       = 3
    output_format = 6
    fileout     = 'wfc_k{KPOINT}_b{KBAND}.cube'
/
"""

with open(f"pp_wfc_k{KPOINT}_b{KBAND}.in", "w") as f:
    f.write(pp_input)

print(f"Extracting |psi|^2 for kpoint={KPOINT}, band={KBAND}...")
result = subprocess.run(
    ["pp.x", "-in", f"pp_wfc_k{KPOINT}_b{KBAND}.in"],
    capture_output=True, text=True, timeout=300
)
with open(f"pp_wfc_k{KPOINT}_b{KBAND}.out", "w") as f:
    f.write(result.stdout)

if result.returncode == 0:
    print(f"Output: wfc_k{KPOINT}_b{KBAND}.cube")
else:
    print("ERROR in pp.x!")
    print(result.stderr[-500:] if result.stderr else "")

# ── Also produce XSF format for VESTA/XCrySDen ───────────────────
pp_xsf_input = f"""&INPUTPP
    prefix   = '{PREFIX}'
    outdir   = '{OUTDIR}'
    filplot  = 'wfc_k{KPOINT}_b{KBAND}.dat'
    plot_num = 7
    kpoint(1)= {KPOINT}
    kband(1) = {KBAND}
    lsign    = .false.
/
&PLOT
    nfile       = 1
    filepp(1)   = 'wfc_k{KPOINT}_b{KBAND}.dat'
    weight(1)   = 1.0
    iflag       = 3
    output_format = 5
    fileout     = 'wfc_k{KPOINT}_b{KBAND}.xsf'
/
"""

with open(f"pp_wfc_k{KPOINT}_b{KBAND}_xsf.in", "w") as f:
    f.write(pp_xsf_input)

result = subprocess.run(
    ["pp.x", "-in", f"pp_wfc_k{KPOINT}_b{KBAND}_xsf.in"],
    capture_output=True, text=True, timeout=300
)
if result.returncode == 0:
    print(f"Output: wfc_k{KPOINT}_b{KBAND}.xsf")
```

#### Step A3: Extract Partial Charge Density (Sum Over Bands)

```python
#!/usr/bin/env python3
"""
Extract partial charge density summed over a range of bands.
This is useful for visualizing VBM states, CBM states, or defect levels.
QE pp.x plot_num=7 with multiple bands, or use plot_num=10 for ILDOS.
"""
import os
import subprocess

OUTDIR = os.path.abspath("./tmp")
PREFIX = "silicon"

# ── Method 1: Integrated LDOS (plot_num=10) ──────────────────────
# Integrates DOS from E_min to E_max (relative to Fermi energy)
# This gives the partial charge density of all states in an energy window.

E_MIN = -2.0  # eV below Fermi level (captures VBM states)
E_MAX =  0.0  # eV (up to Fermi level)

pp_ildos_input = f"""&INPUTPP
    prefix   = '{PREFIX}'
    outdir   = '{OUTDIR}'
    filplot  = 'ildos_vbm.dat'
    plot_num = 10
    emin     = {E_MIN}
    emax     = {E_MAX}
/
&PLOT
    nfile       = 1
    filepp(1)   = 'ildos_vbm.dat'
    weight(1)   = 1.0
    iflag       = 3
    output_format = 6
    fileout     = 'ildos_vbm.cube'
/
"""

with open("pp_ildos_vbm.in", "w") as f:
    f.write(pp_ildos_input)

print("Extracting ILDOS (VBM states)...")
result = subprocess.run(
    ["pp.x", "-in", "pp_ildos_vbm.in"],
    capture_output=True, text=True, timeout=300
)
with open("pp_ildos_vbm.out", "w") as f:
    f.write(result.stdout)

if result.returncode == 0:
    print("Output: ildos_vbm.cube")

# ── Method 2: CBM states ─────────────────────────────────────────
E_MIN_CBM = 0.0
E_MAX_CBM = 2.0

pp_ildos_cbm_input = f"""&INPUTPP
    prefix   = '{PREFIX}'
    outdir   = '{OUTDIR}'
    filplot  = 'ildos_cbm.dat'
    plot_num = 10
    emin     = {E_MIN_CBM}
    emax     = {E_MAX_CBM}
/
&PLOT
    nfile       = 1
    filepp(1)   = 'ildos_cbm.dat'
    weight(1)   = 1.0
    iflag       = 3
    output_format = 6
    fileout     = 'ildos_cbm.cube'
/
"""

with open("pp_ildos_cbm.in", "w") as f:
    f.write(pp_ildos_cbm_input)

print("Extracting ILDOS (CBM states)...")
result = subprocess.run(
    ["pp.x", "-in", "pp_ildos_cbm.in"],
    capture_output=True, text=True, timeout=300
)
if result.returncode == 0:
    print("Output: ildos_cbm.cube")
```

#### Step A4: Visualize with Matplotlib

```python
#!/usr/bin/env python3
"""
Visualize |psi|^2 or partial charge density from cube files using matplotlib.
Produces 2D slice plots and isosurface-like contour plots.
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from ase.io.cube import read_cube_data


def plot_wavefunction_slice(cube_file, axis=2, slice_index=None,
                            log_scale=False, output_png="wfc_slice.png",
                            title="|psi(r)|^2"):
    """
    Plot a 2D slice of |psi|^2 from a cube file.

    Parameters
    ----------
    cube_file : str
        Path to cube file.
    axis : int
        Axis perpendicular to the slice plane (0, 1, or 2).
    slice_index : int or None
        Grid index along axis. None = midpoint.
    log_scale : bool
        Use logarithmic color scale.
    output_png : str
        Output filename.
    title : str
        Plot title.
    """
    data, atoms = read_cube_data(cube_file)
    cell = atoms.get_cell()

    if slice_index is None:
        slice_index = data.shape[axis] // 2

    # Extract 2D slice
    if axis == 0:
        sl = data[slice_index, :, :]
    elif axis == 1:
        sl = data[:, slice_index, :]
    else:
        sl = data[:, :, slice_index]

    # Axis labels
    in_plane = [i for i in range(3) if i != axis]
    extent = [0, np.linalg.norm(cell[in_plane[0]]),
              0, np.linalg.norm(cell[in_plane[1]])]
    axis_labels = ["x", "y", "z"]

    fig, ax = plt.subplots(figsize=(8, 6))

    if log_scale:
        sl_pos = np.clip(sl, 1e-10, None)  # avoid log(0)
        im = ax.contourf(
            np.linspace(extent[0], extent[1], sl.shape[0]),
            np.linspace(extent[2], extent[3], sl.shape[1]),
            sl_pos.T,
            levels=50,
            norm=LogNorm(vmin=sl_pos[sl_pos > 0].min(), vmax=sl_pos.max()),
            cmap="hot"
        )
    else:
        im = ax.contourf(
            np.linspace(extent[0], extent[1], sl.shape[0]),
            np.linspace(extent[2], extent[3], sl.shape[1]),
            sl.T,
            levels=50,
            cmap="hot"
        )

    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label(r"$|\psi|^2$ (a.u.)")

    # Overlay atom positions projected onto this plane
    positions = atoms.get_positions()
    symbols = atoms.get_chemical_symbols()
    for pos, sym in zip(positions, symbols):
        proj = [pos[in_plane[0]], pos[in_plane[1]]]
        # Only plot atoms near this slice
        slice_pos = slice_index / data.shape[axis] * np.linalg.norm(cell[axis])
        if abs(pos[axis] - slice_pos) < 1.0:
            ax.plot(proj[0], proj[1], "wo", markersize=8, markeredgecolor="black")
            ax.text(proj[0] + 0.2, proj[1] + 0.2, sym, fontsize=8, color="white",
                    fontweight="bold")

    ax.set_xlabel(f"{axis_labels[in_plane[0]]} (Ang)", fontsize=12)
    ax.set_ylabel(f"{axis_labels[in_plane[1]]} (Ang)", fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.set_aspect("equal")

    plt.tight_layout()
    plt.savefig(output_png, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"Saved: {output_png}")


def plot_wavefunction_multi_slice(cube_file, axis=2, n_slices=4,
                                   output_png="wfc_multi_slice.png",
                                   title="|psi(r)|^2"):
    """
    Plot multiple 2D slices through the wavefunction.
    """
    data, atoms = read_cube_data(cube_file)
    cell = atoms.get_cell()
    n_total = data.shape[axis]
    length = np.linalg.norm(cell[axis])

    in_plane = [i for i in range(3) if i != axis]
    extent = [0, np.linalg.norm(cell[in_plane[0]]),
              0, np.linalg.norm(cell[in_plane[1]])]

    slice_indices = np.linspace(0, n_total - 1, n_slices + 2, dtype=int)[1:-1]

    fig, axes = plt.subplots(1, n_slices, figsize=(4 * n_slices, 4))
    if n_slices == 1:
        axes = [axes]

    vmax = np.max(data) * 0.8
    vmin = 0

    for i, (ax, idx) in enumerate(zip(axes, slice_indices)):
        if axis == 0:
            sl = data[idx, :, :]
        elif axis == 1:
            sl = data[:, idx, :]
        else:
            sl = data[:, :, idx]

        z_pos = idx / n_total * length

        im = ax.contourf(
            np.linspace(extent[0], extent[1], sl.shape[0]),
            np.linspace(extent[2], extent[3], sl.shape[1]),
            sl.T, levels=30, cmap="hot", vmin=vmin, vmax=vmax
        )
        ax.set_title(f"z = {z_pos:.1f} Ang", fontsize=11)
        ax.set_aspect("equal")
        if i == 0:
            ax.set_ylabel("y (Ang)", fontsize=10)
        ax.set_xlabel("x (Ang)", fontsize=10)

    fig.suptitle(title, fontsize=14)
    plt.tight_layout()
    plt.savefig(output_png, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"Saved: {output_png}")


def plot_wavefunction_isosurface_projection(cube_file, axis=2,
                                             output_png="wfc_integrated.png",
                                             title="|psi|^2 integrated along z"):
    """
    Integrate |psi|^2 along one axis to produce a 2D projection (pseudo-isosurface view).
    """
    data, atoms = read_cube_data(cube_file)
    cell = atoms.get_cell()

    integrated = np.sum(data, axis=axis)
    in_plane = [i for i in range(3) if i != axis]
    extent = [0, np.linalg.norm(cell[in_plane[0]]),
              0, np.linalg.norm(cell[in_plane[1]])]

    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.contourf(
        np.linspace(extent[0], extent[1], integrated.shape[0]),
        np.linspace(extent[2], extent[3], integrated.shape[1]),
        integrated.T, levels=50, cmap="hot"
    )
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label(r"$\int |\psi|^2 dz$ (a.u.)")
    ax.set_xlabel("x (Ang)", fontsize=12)
    ax.set_ylabel("y (Ang)", fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.set_aspect("equal")
    plt.tight_layout()
    plt.savefig(output_png, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"Saved: {output_png}")


# ── Example usage ─────────────────────────────────────────────────
# plot_wavefunction_slice("wfc_k1_b4.cube", axis=2, output_png="vbm_slice.png",
#                         title="VBM |psi|^2 (band 4, Gamma)")
# plot_wavefunction_multi_slice("wfc_k1_b4.cube", n_slices=4,
#                               output_png="vbm_multi.png")
# plot_wavefunction_isosurface_projection("wfc_k1_b4.cube",
#                                          output_png="vbm_projection.png")
```

### Method B: VASP -- PARCHG

```python
#!/usr/bin/env python3
"""
Generate PARCHG (partial charge density) from VASP and visualize.

VASP workflow:
  1) Run a standard SCF to get WAVECAR.
  2) Set LPARD=.TRUE. in INCAR with IBAND and KPUSE to select states.
  3) Run a single-step calculation to produce PARCHG.
  4) Parse PARCHG (same format as CHGCAR) and visualize.

INCAR flags:
  LPARD  = .TRUE.     # Enable partial charge density
  IBAND  = 4          # Band index (or range: IBAND = 4 5 6)
  KPUSE  = 1          # K-point index (or range: KPUSE = 1 2 3)
  LSEPB  = .TRUE.     # Separate PARCHG files per band (optional)
  LSEPK  = .TRUE.     # Separate PARCHG files per k-point (optional)
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def parse_parchg(filename):
    """
    Parse VASP PARCHG file (same format as CHGCAR).
    Returns cell, data (3D array), and atom info.
    """
    with open(filename, "r") as f:
        lines = f.readlines()

    scale = float(lines[1].strip())
    cell = np.zeros((3, 3))
    for i in range(3):
        cell[i] = [float(x) for x in lines[2 + i].split()]
    cell *= scale

    species = lines[5].split()
    counts = [int(x) for x in lines[6].split()]
    n_atoms = sum(counts)

    coord_line = lines[7].strip()
    coord_start = 9 if coord_line[0] in ("S", "s") else 8

    atoms_frac = np.zeros((n_atoms, 3))
    for i in range(n_atoms):
        parts = lines[coord_start + i].split()
        atoms_frac[i] = [float(parts[j]) for j in range(3)]

    data_start = coord_start + n_atoms
    while data_start < len(lines) and lines[data_start].strip() == "":
        data_start += 1

    grid = [int(x) for x in lines[data_start].split()]
    nx, ny, nz = grid
    data_start += 1

    values = []
    for i in range(data_start, len(lines)):
        line = lines[i].strip()
        if line == "" and len(values) >= nx * ny * nz:
            break
        try:
            values.extend([float(x) for x in line.split()])
        except ValueError:
            break
        if len(values) >= nx * ny * nz:
            break

    data = np.array(values[:nx * ny * nz]).reshape((nx, ny, nz), order='F')

    # PARCHG stores rho * V_cell; normalize
    vol = np.abs(np.dot(cell[0], np.cross(cell[1], cell[2])))
    data = data / vol

    return cell, data, atoms_frac, species, counts


def plot_parchg_slice(parchg_file, axis=2, slice_frac=0.5,
                      output_png="parchg_slice.png"):
    """
    Plot a 2D slice of PARCHG.
    """
    cell, data, atoms_frac, species, counts = parse_parchg(parchg_file)
    nx, ny, nz = data.shape

    slice_idx = int(slice_frac * data.shape[axis])
    if axis == 0:
        sl = data[slice_idx, :, :]
    elif axis == 1:
        sl = data[:, slice_idx, :]
    else:
        sl = data[:, :, slice_idx]

    in_plane = [i for i in range(3) if i != axis]
    ext = [0, np.linalg.norm(cell[in_plane[0]]),
           0, np.linalg.norm(cell[in_plane[1]])]

    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.contourf(
        np.linspace(ext[0], ext[1], sl.shape[0]),
        np.linspace(ext[2], ext[3], sl.shape[1]),
        sl.T, levels=50, cmap="hot"
    )
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label(r"$|\psi|^2$ (e/$\mathrm{\AA}^3$)")

    # Overlay atoms near the slice
    atom_cart = atoms_frac @ cell
    for ac, af in zip(atom_cart, atoms_frac):
        if abs(af[axis] - slice_frac) < 0.05:
            ax.plot(ac[in_plane[0]], ac[in_plane[1]], "wo",
                    markersize=8, markeredgecolor="black")

    ax.set_xlabel(f"Axis {in_plane[0]} (Ang)", fontsize=12)
    ax.set_ylabel(f"Axis {in_plane[1]} (Ang)", fontsize=12)
    ax.set_title("Partial Charge Density (PARCHG)", fontsize=14)
    ax.set_aspect("equal")
    plt.tight_layout()
    plt.savefig(output_png, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"Saved: {output_png}")


# Usage:
# plot_parchg_slice("PARCHG", axis=2, slice_frac=0.5)
```

---

## Key Parameters

### QE pp.x Parameters for Wavefunction

| Parameter | Value | Notes |
|---|---|---|
| `plot_num` | 7 | |psi(r)|^2 for a specific (kpoint, band) |
| `plot_num` | 10 | Integrated LDOS in an energy window [emin, emax] |
| `kpoint(1)` | integer | K-point index (1-based from the k-list) |
| `kband(1)` | integer | Band index (1-based) |
| `lsign` | `.false.` or `.true.` | If `.true.`, plot the sign of psi (not |psi|^2). Only for Gamma-point, real wavefunctions |
| `emin`, `emax` | float (eV) | Energy window for `plot_num=10` (relative to Fermi level) |
| `output_format` | 6 (cube), 5 (xsf), 3 (xsf 2D) | Choose based on visualization tool |

### VASP INCAR Parameters for PARCHG

| Parameter | Value | Notes |
|---|---|---|
| `LPARD` | `.TRUE.` | Enable partial charge density output |
| `IBAND` | integer(s) | Band index/indices to include |
| `KPUSE` | integer(s) | K-point index/indices to include |
| `LSEPB` | `.TRUE.` | Write separate file per band |
| `LSEPK` | `.TRUE.` | Write separate file per k-point |
| `NBMOD` | -3 | Compute partial charge for all bands in [EINT(1), EINT(2)] |
| `EINT` | E_min E_max | Energy window (eV, relative to Fermi) when `NBMOD=-3` |

## Interpreting Results

- **Bonding states**: |psi|^2 concentrated between atoms indicates covalent bonding.
- **Antibonding states**: Nodal planes between atoms; density pushed away from the bond.
- **Defect states**: Localized |psi|^2 around the defect site.
- **Surface states**: |psi|^2 localized at the surface, decaying into the bulk and vacuum.
- **Band edge character**: VBM in semiconductors often has p-orbital character (elongated lobes);
  CBM may be s-like (spherical) or d-like depending on the material.
- **Symmetry**: The wavefunction should respect the crystal symmetry. Broken symmetry
  in the plot may indicate a numerical issue or a symmetry-broken state (e.g., Jahn-Teller).

## Common Issues

| Problem | Solution |
|---|---|
| pp.x crashes with "kpoint out of range" | The k-point index must match the k-list in the SCF/NSCF. Use `verbosity='high'` to print the k-list |
| Cube file appears empty/uniform | The selected band may have negligible weight at the chosen k-point. Choose a k-point where the band has significant character |
| `plot_num=7` gives all zeros | The wavefunction data may not be saved. Ensure the SCF `outdir` is intact and `prefix` matches |
| PARCHG file is empty | Ensure `LPARD=.TRUE.` and `IBAND`/`KPUSE` are set. WAVECAR must exist from a prior calculation |
| Sign of wavefunction needed | Use `lsign=.true.` in pp.x (only works for real wavefunctions at Gamma). For complex wavefunctions, this is not meaningful |
| Large cube files | Reduce ecutrho to get a coarser FFT grid, or use iflag=2 for a 2D slice directly |
