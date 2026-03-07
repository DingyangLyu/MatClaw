# Spatially-Resolved Density of States (3D Local DOS)

## When to Use

- You need the local density of states (LDOS) resolved in real space for a specific energy window.
- You want to visualize which atoms/regions contribute to states near the Fermi level.
- You are analyzing interface band alignment by examining layer-by-layer DOS profiles.
- You want to identify spatial localization of defect states, surface states, or impurity levels.
- You need plane-averaged LDOS along a slab normal for interface/surface analysis.
- You are studying charge density in a selected energy range (partial charge analysis).

## Method Selection

| Task | QE (pp.x) | VASP (LPARD) | VASPKIT (123--126) |
|---|---|---|---|
| 3D LDOS in energy window | pp.x with `plot_num=10` | `LPARD=.TRUE.` + `EINT` in INCAR | VASPKIT 123 (from PARCHG) |
| Plane-averaged LDOS | pp.x `iflag=3, output_format=0` | `macroave.x` or custom script on PARCHG | VASPKIT 124 (plane-average) |
| LDOS for specific bands | pp.x from selected bands output | `IBAND` + `LPARD=.TRUE.` | VASPKIT 125 (band-decomposed) |
| Integrated LDOS (charge in range) | pp.x `plot_num=10` + integration | `EINT` tag in INCAR | VASPKIT 126 |
| Isosurface visualization | pp.x outputs cube file; use VESTA | PARCHG file; use VESTA | VASPKIT exports cube/CHGCAR |

## Prerequisites

- A converged SCF (and optionally NSCF) calculation.
- **QE**: `pw.x`, `pp.x` executables. Converged charge density and wavefunctions in `outdir`.
- **VASP**: Converged WAVECAR. INCAR with `LPARD=.TRUE.` and `EINT` tags.
- **VASPKIT**: Installed and in PATH (for tasks 123--126).
- Python packages: `numpy`, `matplotlib`, `scipy`.
- Visualization: VESTA (for 3D isosurfaces from cube/CHGCAR files).

---

## Detailed Steps

### Method A: QE Spatially-Resolved DOS with pp.x

#### Step A1: SCF Calculation (prerequisite)

The SCF calculation must save wavefunctions (default behavior). If you already have a converged SCF from the `scf-relax` or `density-of-states` skill, skip this step.

```python
#!/usr/bin/env python3
"""
Run SCF and ensure wavefunctions are saved for pp.x.
Example: Si slab with vacuum (for surface LDOS).
"""
import os
import subprocess

PSEUDO_DIR = os.path.abspath("./pseudo")
OUTDIR = os.path.abspath("./tmp_ldos")
os.makedirs(OUTDIR, exist_ok=True)
PREFIX = "si_slab"

scf_input = f"""&CONTROL
    calculation = 'scf'
    prefix      = '{PREFIX}'
    outdir      = '{OUTDIR}'
    pseudo_dir  = '{PSEUDO_DIR}'
    wf_collect  = .true.
/

&SYSTEM
    ibrav       = 0
    nat         = 8
    ntyp        = 1
    ecutwfc     = 50.0
    ecutrho     = 400.0
    occupations = 'smearing'
    smearing    = 'cold'
    degauss     = 0.01
    nbnd        = 24
/

&ELECTRONS
    conv_thr = 1.0d-8
/

CELL_PARAMETERS (angstrom)
  3.840  0.000  0.000
  0.000  3.840  0.000
  0.000  0.000  25.000

ATOMIC_SPECIES
  Si  28.0855  Si.pbe-n-rrkjus_psl.1.0.0.UPF

ATOMIC_POSITIONS (angstrom)
  Si   0.000   0.000   8.000
  Si   1.920   1.920   8.000
  Si   0.000   1.920   9.360
  Si   1.920   0.000   9.360
  Si   0.000   0.000  10.720
  Si   1.920   1.920  10.720
  Si   0.000   1.920  12.080
  Si   1.920   0.000  12.080

K_POINTS (automatic)
  6 6 1  0 0 0
"""

with open(f"{PREFIX}_scf.in", "w") as f:
    f.write(scf_input)

print("Running SCF for LDOS calculation ...")
result = subprocess.run(
    ["mpirun", "-np", "4", "pw.x", "-in", f"{PREFIX}_scf.in"],
    capture_output=True, text=True, timeout=1200
)
with open(f"{PREFIX}_scf.out", "w") as f:
    f.write(result.stdout)

if "convergence has been achieved" in result.stdout:
    print("SCF converged. Wavefunctions saved.")
else:
    print("WARNING: SCF may not have converged!")
```

#### Step A2: Compute 3D LDOS with pp.x (plot_num=10)

```python
#!/usr/bin/env python3
"""
Use pp.x to compute the local density of states in a specified energy range.
plot_num=10 computes LDOS: sum of |psi_n(r)|^2 for states in [Emin, Emax].
Output is a 3D charge-density-like file in cube format.
"""
import os
import subprocess
import re

OUTDIR = os.path.abspath("./tmp_ldos")
PREFIX = "si_slab"

# ── Extract Fermi energy ────────────────────────────────────────────
e_fermi = 0.0
with open(f"{PREFIX}_scf.out", "r") as f:
    for line in f:
        if "the Fermi energy is" in line:
            m = re.search(r"is\s+([-\d.]+)", line)
            if m: e_fermi = float(m.group(1))
        if "highest occupied, lowest unoccupied" in line:
            m = re.search(r":\s+([-\d.]+)\s+([-\d.]+)", line)
            if m: e_fermi = (float(m.group(1)) + float(m.group(2))) / 2
        if "highest occupied" in line and "lowest" not in line:
            m = re.search(r":\s+([-\d.]+)", line)
            if m: e_fermi = float(m.group(1))

print(f"Fermi energy: {e_fermi:.4f} eV")

# ── Define energy windows for LDOS ─────────────────────────────────
# Each window produces one 3D LDOS file.
# Common choices:
#   - Near Fermi level: [E_F - 0.5, E_F + 0.5] for metallic states
#   - Valence band edge: [E_F - 2.0, E_F] for VBM states
#   - Conduction band edge: [E_F, E_F + 2.0] for CBM states
#   - Deep valence: [E_F - 10.0, E_F - 5.0]

energy_windows = [
    {"label": "valence_edge", "emin": e_fermi - 2.0, "emax": e_fermi,
     "description": "States within 2 eV below Fermi level (VBM region)"},
    {"label": "conduction_edge", "emin": e_fermi, "emax": e_fermi + 2.0,
     "description": "States within 2 eV above Fermi level (CBM region)"},
    {"label": "near_fermi", "emin": e_fermi - 0.5, "emax": e_fermi + 0.5,
     "description": "States within 0.5 eV of Fermi level"},
]

for window in energy_windows:
    label = window["label"]
    emin = window["emin"]
    emax = window["emax"]

    print(f"\nComputing LDOS: {window['description']}")
    print(f"  Energy window: [{emin:.2f}, {emax:.2f}] eV")

    pp_input = f"""&INPUTPP
    prefix  = '{PREFIX}'
    outdir  = '{OUTDIR}'
    plot_num = 10
    emin    = {emin}
    emax    = {emax}
    filplot = '{PREFIX}_ldos_{label}.dat'
/

&PLOT
    iflag         = 3
    output_format = 6
    fileout       = '{PREFIX}_ldos_{label}.cube'
/
"""
    infile = f"{PREFIX}_pp_ldos_{label}.in"
    with open(infile, "w") as f:
        f.write(pp_input)

    result = subprocess.run(
        ["pp.x", "-in", infile],
        capture_output=True, text=True, timeout=300
    )
    with open(f"{PREFIX}_pp_ldos_{label}.out", "w") as f:
        f.write(result.stdout)

    if result.returncode == 0:
        print(f"  Generated: {PREFIX}_ldos_{label}.cube")
        print(f"  Open in VESTA for 3D isosurface visualization.")
    else:
        print(f"  ERROR in pp.x for {label}!")
        print(result.stderr[-300:] if result.stderr else "")
```

#### Step A3: Plane-Averaged LDOS Profile

```python
#!/usr/bin/env python3
"""
Compute plane-averaged LDOS along the z-direction (slab normal).
Uses pp.x with iflag=3 for 3D data, then averages in xy-plane.
Alternatively, uses pp.x with iflag=3, output_format=0 to get planar average directly.
"""
import os
import subprocess
import re
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUTDIR = os.path.abspath("./tmp_ldos")
PREFIX = "si_slab"

# ── Extract Fermi energy ────────────────────────────────────────────
e_fermi = 0.0
with open(f"{PREFIX}_scf.out", "r") as f:
    for line in f:
        if "the Fermi energy is" in line:
            m = re.search(r"is\s+([-\d.]+)", line)
            if m: e_fermi = float(m.group(1))
        if "highest occupied, lowest unoccupied" in line:
            m = re.search(r":\s+([-\d.]+)\s+([-\d.]+)", line)
            if m: e_fermi = (float(m.group(1)) + float(m.group(2))) / 2

# ── Method 1: pp.x planar average (iflag=2) ─────────────────────────
# iflag=2 computes the planar average along z directly
def compute_planar_ldos(prefix, outdir, emin, emax, label):
    """Compute and return plane-averaged LDOS in the specified energy window."""

    # Step 1: compute 3D LDOS
    pp_input_3d = f"""&INPUTPP
    prefix  = '{prefix}'
    outdir  = '{outdir}'
    plot_num = 10
    emin    = {emin}
    emax    = {emax}
    filplot = '{prefix}_ldos_{label}.dat'
/

&PLOT
    iflag         = 2
    output_format = 0
    fileout       = '{prefix}_ldos_{label}_planar.dat'
    e1(1) = 1.0, e1(2) = 0.0, e1(3) = 0.0
    e2(1) = 0.0, e2(2) = 1.0, e2(3) = 0.0
    e3(1) = 0.0, e3(2) = 0.0, e3(3) = 1.0
    nx = 1, ny = 1, nz = 200
/
"""
    infile = f"{prefix}_pp_planar_{label}.in"
    with open(infile, "w") as f:
        f.write(pp_input_3d)

    result = subprocess.run(
        ["pp.x", "-in", infile],
        capture_output=True, text=True, timeout=300
    )

    if result.returncode != 0:
        print(f"  pp.x failed for {label}")
        return None, None

    # Parse the 1D planar average output
    outfile = f"{prefix}_ldos_{label}_planar.dat"
    if os.path.exists(outfile):
        data = np.loadtxt(outfile)
        z = data[:, 0]
        ldos_avg = data[:, 1]
        return z, ldos_avg
    return None, None

# ── Method 2: Parse Gaussian cube file and average ────────────────
def parse_cube_and_average(cube_file, axis=2):
    """
    Parse a Gaussian cube file, compute planar average along specified axis.
    axis: 0=x, 1=y, 2=z (default for slab normal)
    Returns: positions (Angstrom), average_values
    """
    with open(cube_file, "r") as f:
        # Skip 2 comment lines
        f.readline()
        f.readline()

        # Line 3: n_atoms, origin
        parts = f.readline().split()
        n_atoms = int(parts[0])
        origin = np.array([float(parts[1]), float(parts[2]), float(parts[3])])

        # Lines 4-6: n_voxels and voxel vectors (in Bohr)
        voxel_info = []
        for _ in range(3):
            parts = f.readline().split()
            n_vox = int(parts[0])
            vec = np.array([float(parts[1]), float(parts[2]), float(parts[3])])
            voxel_info.append((n_vox, vec))

        nx, vx = voxel_info[0]
        ny, vy = voxel_info[1]
        nz, vz = voxel_info[2]

        # Skip atom lines
        for _ in range(abs(n_atoms)):
            f.readline()

        # Read volumetric data
        values = []
        for line in f:
            values.extend([float(x) for x in line.split()])

    data = np.array(values).reshape((nx, ny, nz))

    # Compute planar average along the specified axis
    if axis == 2:
        average = np.mean(data, axis=(0, 1))
        step = np.linalg.norm(vz) * 0.529177  # Bohr -> Angstrom
        positions = np.arange(nz) * step + origin[axis] * 0.529177
    elif axis == 1:
        average = np.mean(data, axis=(0, 2))
        step = np.linalg.norm(vy) * 0.529177
        positions = np.arange(ny) * step + origin[axis] * 0.529177
    else:
        average = np.mean(data, axis=(1, 2))
        step = np.linalg.norm(vx) * 0.529177
        positions = np.arange(nx) * step + origin[axis] * 0.529177

    return positions, average

# ── Compute planar LDOS for multiple energy windows ────────────────
energy_windows = [
    {"label": "deep_valence", "emin": e_fermi - 10.0, "emax": e_fermi - 5.0},
    {"label": "valence_edge", "emin": e_fermi - 2.0, "emax": e_fermi},
    {"label": "conduction_edge", "emin": e_fermi, "emax": e_fermi + 2.0},
]

fig, axes = plt.subplots(len(energy_windows), 1,
                         figsize=(10, 3.5 * len(energy_windows)),
                         sharex=True, gridspec_kw={"hspace": 0.08})
if len(energy_windows) == 1:
    axes = [axes]

for idx, window in enumerate(energy_windows):
    label = window["label"]
    emin = window["emin"]
    emax = window["emax"]
    ax = axes[idx]

    # Try cube file method
    cube_file = f"{PREFIX}_ldos_{label}.cube"
    if os.path.exists(cube_file):
        z_pos, ldos_avg = parse_cube_and_average(cube_file, axis=2)
    else:
        # Fall back to pp.x planar average
        z_pos, ldos_avg = compute_planar_ldos(PREFIX, OUTDIR, emin, emax, label)

    if z_pos is not None and ldos_avg is not None:
        ax.plot(z_pos, ldos_avg, color="steelblue", linewidth=1.5)
        ax.fill_between(z_pos, ldos_avg, alpha=0.15, color="steelblue")
        ax.set_ylabel("Planar avg.\nLDOS", fontsize=11)
        ax.set_title(f"{label}: [{emin:.1f}, {emax:.1f}] eV", fontsize=12)
        ax.grid(alpha=0.3)
    else:
        ax.text(0.5, 0.5, f"Data not available for {label}",
                transform=ax.transAxes, ha="center", fontsize=12)

axes[-1].set_xlabel("z position (Angstrom)", fontsize=13)
axes[0].set_title("Plane-Averaged LDOS Along Slab Normal", fontsize=14)

plt.savefig("ldos_planar_average.png", dpi=200, bbox_inches="tight")
print("Saved: ldos_planar_average.png")
```

#### Step A4: Energy-Resolved LDOS Map (E vs z)

```python
#!/usr/bin/env python3
"""
Create a 2D color map of LDOS(z, E) -- the local DOS as a function of
position along the slab normal and energy.
Requires running pp.x for multiple energy windows.
"""
import os
import subprocess
import re
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUTDIR = os.path.abspath("./tmp_ldos")
PREFIX = "si_slab"

# Extract Fermi energy
e_fermi = 0.0
with open(f"{PREFIX}_scf.out", "r") as f:
    for line in f:
        if "the Fermi energy is" in line:
            m = re.search(r"is\s+([-\d.]+)", line)
            if m: e_fermi = float(m.group(1))
        if "highest occupied, lowest unoccupied" in line:
            m = re.search(r":\s+([-\d.]+)\s+([-\d.]+)", line)
            if m: e_fermi = (float(m.group(1)) + float(m.group(2))) / 2

# ── Define fine energy grid ────────────────────────────────────────
e_start = e_fermi - 8.0
e_end = e_fermi + 5.0
de = 0.5  # Energy window width
e_centers = np.arange(e_start + de / 2, e_end, de)

print(f"Computing LDOS for {len(e_centers)} energy windows ...")

# Cube file parser (reused from Step A3)
def parse_cube_planar_avg(cube_file, axis=2):
    with open(cube_file, "r") as f:
        f.readline(); f.readline()
        parts = f.readline().split()
        n_atoms = int(parts[0])
        origin = np.array([float(parts[1]), float(parts[2]), float(parts[3])])
        voxel_info = []
        for _ in range(3):
            parts = f.readline().split()
            voxel_info.append((int(parts[0]),
                               np.array([float(parts[1]), float(parts[2]), float(parts[3])])))
        for _ in range(abs(n_atoms)):
            f.readline()
        values = []
        for line in f:
            values.extend([float(x) for x in line.split()])
    nx, vx = voxel_info[0]
    ny, vy = voxel_info[1]
    nz, vz = voxel_info[2]
    data = np.array(values).reshape((nx, ny, nz))
    average = np.mean(data, axis=(0, 1))
    step = np.linalg.norm(vz) * 0.529177
    positions = np.arange(nz) * step + origin[axis] * 0.529177
    return positions, average

# ── Run pp.x for each window ──────────────────────────────────────
z_positions = None
ldos_map = []

for i, ec in enumerate(e_centers):
    emin = ec - de / 2
    emax = ec + de / 2
    label = f"emap_{i:03d}"

    pp_input = f"""&INPUTPP
    prefix  = '{PREFIX}'
    outdir  = '{OUTDIR}'
    plot_num = 10
    emin    = {emin}
    emax    = {emax}
    filplot = '{PREFIX}_ldos_{label}.dat'
/

&PLOT
    iflag         = 3
    output_format = 6
    fileout       = '{PREFIX}_ldos_{label}.cube'
/
"""
    infile = f"{PREFIX}_pp_{label}.in"
    with open(infile, "w") as f:
        f.write(pp_input)

    result = subprocess.run(
        ["pp.x", "-in", infile],
        capture_output=True, text=True, timeout=120
    )

    cube_file = f"{PREFIX}_ldos_{label}.cube"
    if os.path.exists(cube_file):
        z_pos, ldos_avg = parse_cube_planar_avg(cube_file, axis=2)
        if z_positions is None:
            z_positions = z_pos
        ldos_map.append(ldos_avg)
        print(f"  Window [{emin:.1f}, {emax:.1f}] eV done.")
    else:
        # Fill with zeros if computation failed
        if z_positions is not None:
            ldos_map.append(np.zeros_like(z_positions))
        print(f"  Window [{emin:.1f}, {emax:.1f}] eV FAILED.")

if z_positions is not None and ldos_map:
    ldos_2d = np.array(ldos_map)  # shape: (n_energies, n_z)

    # ── Plot the E vs z color map ──────────────────────────────────
    fig, ax = plt.subplots(figsize=(10, 6))

    Z, E = np.meshgrid(z_positions, e_centers - e_fermi)
    im = ax.pcolormesh(Z, E, ldos_2d, shading="auto", cmap="hot_r")
    cbar = plt.colorbar(im, ax=ax, label="LDOS (arb. units)")

    ax.axhline(0, color="cyan", linestyle="--", linewidth=1.0, label="$E_F$")
    ax.set_xlabel("z position (Angstrom)", fontsize=13)
    ax.set_ylabel("$E - E_F$ (eV)", fontsize=13)
    ax.set_title("Spatially-Resolved LDOS", fontsize=14)
    ax.legend(fontsize=11, loc="upper right")

    plt.tight_layout()
    plt.savefig("ldos_energy_position_map.png", dpi=200, bbox_inches="tight")
    print("\nSaved: ldos_energy_position_map.png")

    # Clean up intermediate cube files
    import glob
    for f in glob.glob(f"{PREFIX}_ldos_emap_*.cube"):
        os.remove(f)
    for f in glob.glob(f"{PREFIX}_ldos_emap_*.dat"):
        os.remove(f)
    for f in glob.glob(f"{PREFIX}_pp_emap_*.in"):
        os.remove(f)
    print("Cleaned up intermediate files.")
```

### Method B: VASP Spatially-Resolved DOS (LPARD / EINT)

#### Step B1: VASP INCAR Setup for Partial Charge (PARCHG)

```python
#!/usr/bin/env python3
"""
Generate VASP INCAR for partial charge density in an energy window.
VASP writes PARCHG file containing charge density for states in [EINT(1), EINT(2)].
This is the VASP equivalent of QE's pp.x with plot_num=10.
"""
import os

# ── Method 1: Energy-window decomposition ──────────────────────────
# EINT specifies energy range relative to Fermi level (in eV).
# LPARD=.TRUE. triggers partial charge output.

incar_content_energy = """\
# VASP INCAR for partial charge in energy window
# Step 1: Run a normal SCF first (LPARD=.FALSE.)
# Step 2: Copy WAVECAR, set LPARD=.TRUE. and EINT, rerun

SYSTEM = Partial charge density

# Electronic settings (must match SCF)
ENCUT  = 520
EDIFF  = 1E-6
ISMEAR = -5        # Tetrahedron method
# ISMEAR = 0       # Gaussian for metals
# SIGMA  = 0.05

# Partial charge settings
LPARD  = .TRUE.    # Enable partial charge output
EINT   = -2.0 0.0  # Energy window [E_F - 2.0, E_F] in eV relative to E_F
# EINT   = 0.0 2.0  # For conduction band edge

# NBMOD determines the mode:
# NBMOD = -3: use EINT energy window (recommended)
# NBMOD = 0: compute partial charge for all bands
# NBMOD = N>0: decompose into N energy intervals
NBMOD  = -3

LSEPB  = .FALSE.   # Do not separate by band
LSEPK  = .FALSE.   # Do not separate by k-point

# Ensure wavefunctions are available
LWAVE  = .TRUE.
ISTART = 1         # Read WAVECAR
ICHARG = 0
"""

with open("INCAR_parchg_energy", "w") as f:
    f.write(incar_content_energy)
print("Written: INCAR_parchg_energy")

# ── Method 2: Band-decomposed partial charge ───────────────────────
incar_content_band = """\
# VASP INCAR for band-decomposed partial charge
# Generates PARCHG for specific band indices

SYSTEM = Band-decomposed charge density

ENCUT  = 520
EDIFF  = 1E-6
ISMEAR = -5

LPARD  = .TRUE.
IBAND  = 10 11 12 13  # Band indices to decompose (1-based)
LSEPB  = .TRUE.       # Write separate PARCHG per band
LSEPK  = .FALSE.

LWAVE  = .TRUE.
ISTART = 1
ICHARG = 0
"""

with open("INCAR_parchg_band", "w") as f:
    f.write(incar_content_band)
print("Written: INCAR_parchg_band")

# ── Method 3: k-point-resolved partial charge ──────────────────────
incar_content_kpt = """\
# VASP INCAR for k-point-resolved partial charge
# Useful for identifying surface states at specific k-points

SYSTEM = k-resolved charge density

ENCUT  = 520
EDIFF  = 1E-6
ISMEAR = -5

LPARD  = .TRUE.
EINT   = -1.0 0.0
NBMOD  = -3
LSEPB  = .FALSE.
LSEPK  = .TRUE.   # Write separate PARCHG per k-point
KPUSE  = 1 2 3    # Only these k-point indices (optional)

LWAVE  = .TRUE.
ISTART = 1
ICHARG = 0
"""

with open("INCAR_parchg_kpt", "w") as f:
    f.write(incar_content_kpt)
print("Written: INCAR_parchg_kpt")

print("""
Workflow:
  1. Run normal SCF with LWAVE=.TRUE. to generate WAVECAR.
  2. Copy WAVECAR to the partial charge directory.
  3. Copy the appropriate INCAR_parchg_* to INCAR.
  4. Run VASP (single step: reads WAVECAR, writes PARCHG).
  5. Visualize PARCHG in VESTA, or use VASPKIT for further processing.
""")
```

#### Step B2: Parse and Visualize VASP PARCHG

```python
#!/usr/bin/env python3
"""
Parse VASP PARCHG file and compute plane-averaged LDOS along z-direction.
PARCHG has the same format as CHGCAR.
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pymatgen.io.vasp import Chgcar
from pymatgen.core import Structure

# ── Parse PARCHG using pymatgen ────────────────────────────────────
print("Reading PARCHG ...")
parchg = Chgcar.from_file("PARCHG")
structure = parchg.structure
data = parchg.data["total"]  # 3D numpy array (ngx, ngy, ngz)

# Lattice parameters
lattice = structure.lattice
c_length = lattice.c  # Length along z in Angstrom

# Grid dimensions
ngx, ngy, ngz = data.shape
print(f"Grid: {ngx} x {ngy} x {ngz}")
print(f"Cell c-parameter: {c_length:.4f} Angstrom")

# ── Compute plane-averaged charge along z ──────────────────────────
# Average over x and y for each z-plane
plane_avg = np.mean(data, axis=(0, 1))

# Normalize: VASP stores charge * volume, so divide by volume
volume = lattice.volume
plane_avg_normalized = plane_avg / volume

# z positions
z_frac = np.linspace(0, 1, ngz, endpoint=False)
z_cart = z_frac * c_length

# ── Macroscopic average (optional) ─────────────────────────────────
def macroscopic_average(z, values, period):
    """
    Compute macroscopic average by convolving with a window of width 'period'.
    Useful for removing rapid oscillations near atoms.
    """
    dz = z[1] - z[0]
    n_window = max(1, int(period / dz))
    kernel = np.ones(n_window) / n_window
    return np.convolve(values, kernel, mode="same")

# For a slab, use the interlayer spacing as averaging period
interlayer_spacing = 1.36  # Angstrom (Si interplanar distance, adjust as needed)
macro_avg = macroscopic_average(z_cart, plane_avg_normalized, interlayer_spacing)

# ── Plot ────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(10, 5))

ax.plot(z_cart, plane_avg_normalized, color="steelblue", linewidth=0.8,
        alpha=0.7, label="Microscopic")
ax.plot(z_cart, macro_avg, color="darkorange", linewidth=2.0,
        label=f"Macroscopic avg (d={interlayer_spacing:.2f} A)")

# Mark atomic positions
for site in structure:
    z_atom = site.coords[2]
    ax.axvline(z_atom, color="gray", linestyle=":", linewidth=0.5, alpha=0.5)

ax.set_xlabel("z position (Angstrom)", fontsize=13)
ax.set_ylabel("Plane-averaged LDOS (arb. units)", fontsize=13)
ax.set_title("VASP PARCHG: Plane-Averaged Partial Charge", fontsize=14)
ax.legend(fontsize=11)
ax.grid(alpha=0.3)

plt.tight_layout()
plt.savefig("parchg_planar_average.png", dpi=200, bbox_inches="tight")
print("Saved: parchg_planar_average.png")

# ── Also write planar average data to file ─────────────────────────
np.savetxt("parchg_planar_avg.dat",
           np.column_stack([z_cart, plane_avg_normalized, macro_avg]),
           header="z(Angstrom)  microscopic  macroscopic",
           fmt="%.6f")
print("Saved: parchg_planar_avg.dat")
```

#### Step B3: PARCHG Analysis for Interface Band Alignment

```python
#!/usr/bin/env python3
"""
Analyze PARCHG for interface band alignment.
Compute plane-averaged LDOS for valence and conduction energy windows,
then determine the band offset from the profiles.
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pymatgen.io.vasp import Chgcar

def parse_parchg_planar(filename):
    """Parse PARCHG and return plane-averaged LDOS along z."""
    chg = Chgcar.from_file(filename)
    data = chg.data["total"]
    lattice = chg.structure.lattice
    ngz = data.shape[2]
    plane_avg = np.mean(data, axis=(0, 1)) / lattice.volume
    z_cart = np.linspace(0, lattice.c, ngz, endpoint=False)
    return z_cart, plane_avg, chg.structure

# ── Parse multiple PARCHG files (different energy windows) ─────────
# Assumes you ran VASP with different EINT values and renamed PARCHG files
parchg_files = {
    "VBM (Material A)": "PARCHG_vbm_A",      # E_F - 2.0 to E_F for material A side
    "CBM (Material A)": "PARCHG_cbm_A",      # E_F to E_F + 2.0 for material A side
    "VBM (Material B)": "PARCHG_vbm_B",      # similarly for material B
    "CBM (Material B)": "PARCHG_cbm_B",
}

# If you only have one PARCHG per energy window for the whole interface,
# that is fine too -- just use the single file:
# z, ldos_valence, struct = parse_parchg_planar("PARCHG_valence")
# z, ldos_conduction, struct = parse_parchg_planar("PARCHG_conduction")

# ── Example: single interface calculation ──────────────────────────
# Two PARCHG files from VASP runs with different EINT settings

import os

fig, axes = plt.subplots(2, 1, figsize=(10, 7), sharex=True,
                         gridspec_kw={"hspace": 0.08})

for ax_idx, (label, filename) in enumerate([
    ("Valence edge [-2,0] eV", "PARCHG_valence"),
    ("Conduction edge [0,+2] eV", "PARCHG_conduction"),
]):
    ax = axes[ax_idx]
    if os.path.exists(filename):
        z, ldos, struct = parse_parchg_planar(filename)
        ax.plot(z, ldos, color="steelblue", linewidth=1.2)
        ax.fill_between(z, ldos, alpha=0.15, color="steelblue")
        # Mark interface position (user-defined)
        ax.axvline(z[len(z) // 2], color="red", linestyle="--",
                   linewidth=1.0, label="Interface")
    else:
        ax.text(0.5, 0.5, f"File {filename} not found",
                transform=ax.transAxes, ha="center", fontsize=12)
    ax.set_ylabel("Planar LDOS", fontsize=12)
    ax.set_title(label, fontsize=12)
    ax.legend(fontsize=10)
    ax.grid(alpha=0.3)

axes[-1].set_xlabel("z position (Angstrom)", fontsize=13)
axes[0].set_title("Interface Band Alignment from PARCHG", fontsize=14)

plt.savefig("interface_band_alignment.png", dpi=200, bbox_inches="tight")
print("Saved: interface_band_alignment.png")
```

### Method C: VASPKIT Workflows (123--126)

```bash
#!/bin/bash
# VASPKIT spatially-resolved DOS post-processing.
# Prerequisites: VASP calculation with LPARD=.TRUE. and PARCHG output.

# ── Task 123: 3D partial charge visualization ──────────────────────
# Reads PARCHG, converts to Gaussian cube format for VESTA visualization.
echo "123" | vaspkit
# Output: PARCHG.cube (Gaussian cube format)
# Open in VESTA: File -> Open -> PARCHG.cube -> set isosurface value

# ── Task 124: Plane-averaged LDOS from PARCHG ─────────────────────
echo "124" | vaspkit
# Interactive: select averaging direction (x, y, or z).
# Output: PLANAR_AVERAGE.dat with columns: position, planar_avg, macroscopic_avg
# Plot with gnuplot or python.

# ── Task 125: Band-decomposed charge density ──────────────────────
# Requires PARCHG files from LSEPB=.TRUE. calculation.
echo "125" | vaspkit
# Reads PARCHG.BAND_* files. Useful for identifying orbital character of specific bands.

# ── Task 126: Integrated partial charge ────────────────────────────
echo "126" | vaspkit
# Integrates PARCHG over user-selected spatial region.
# Useful for computing charge transfer at interfaces.
```

### Complete QE Workflow: Structure to Spatially-Resolved LDOS

```python
#!/usr/bin/env python3
"""
Complete QE workflow for spatially-resolved LDOS.
Runs: SCF -> pp.x (multiple energy windows) -> planar average -> E-vs-z map.
"""
import os
import subprocess
import re
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ══════════════════════════════════════════════════════════════════
#  Configuration
# ══════════════════════════════════════════════════════════════════
PSEUDO_DIR = os.path.abspath("./pseudo")
OUTDIR = os.path.abspath("./tmp_ldos_full")
os.makedirs(OUTDIR, exist_ok=True)
PREFIX = "slab"
ECUTWFC = 50.0
ECUTRHO = 400.0
NPROC = 4

# ══════════════════════════════════════════════════════════════════
#  Step 1: SCF (assume already done or provide input)
# ══════════════════════════════════════════════════════════════════
# [SCF input would go here -- see scf-relax skill]
# For this workflow, we assume SCF is already converged.
# The key requirement: wf_collect = .true. so pp.x can read wavefunctions.

# Extract Fermi energy
e_fermi = 0.0
scf_out = f"{PREFIX}_scf.out"
if os.path.exists(scf_out):
    with open(scf_out, "r") as f:
        for line in f:
            if "the Fermi energy is" in line:
                m = re.search(r"is\s+([-\d.]+)", line)
                if m: e_fermi = float(m.group(1))
            if "highest occupied, lowest unoccupied" in line:
                m = re.search(r":\s+([-\d.]+)\s+([-\d.]+)", line)
                if m: e_fermi = (float(m.group(1)) + float(m.group(2))) / 2
            if "highest occupied" in line and "lowest" not in line:
                m = re.search(r":\s+([-\d.]+)", line)
                if m: e_fermi = float(m.group(1))

print(f"Fermi energy: {e_fermi:.4f} eV")

# ══════════════════════════════════════════════════════════════════
#  Step 2: Compute LDOS cube files for energy grid
# ══════════════════════════════════════════════════════════════════
e_start = e_fermi - 8.0
e_end = e_fermi + 4.0
de = 0.5  # Window width in eV
e_centers = np.arange(e_start + de / 2, e_end, de)

print(f"Computing LDOS for {len(e_centers)} energy windows ...")

def run_pp_ldos(prefix, outdir, emin, emax, label):
    """Run pp.x for one energy window, return cube filename."""
    pp_input = f"""&INPUTPP
    prefix  = '{prefix}'
    outdir  = '{outdir}'
    plot_num = 10
    emin    = {emin}
    emax    = {emax}
    filplot = '{prefix}_ldos_{label}.dat'
/

&PLOT
    iflag         = 3
    output_format = 6
    fileout       = '{prefix}_ldos_{label}.cube'
/
"""
    infile = f"{prefix}_pp_{label}.in"
    with open(infile, "w") as f:
        f.write(pp_input)

    result = subprocess.run(
        ["pp.x", "-in", infile],
        capture_output=True, text=True, timeout=120
    )
    cube_file = f"{prefix}_ldos_{label}.cube"
    return cube_file if os.path.exists(cube_file) else None

def parse_cube_planar(cube_file, axis=2):
    """Parse cube file and return planar average along axis."""
    with open(cube_file, "r") as f:
        f.readline(); f.readline()
        parts = f.readline().split()
        n_atoms = int(parts[0])
        origin = [float(x) for x in parts[1:4]]
        voxel_info = []
        for _ in range(3):
            parts = f.readline().split()
            voxel_info.append((int(parts[0]), [float(x) for x in parts[1:4]]))
        for _ in range(abs(n_atoms)):
            f.readline()
        values = []
        for line in f:
            values.extend([float(x) for x in line.split()])

    dims = [v[0] for v in voxel_info]
    data = np.array(values).reshape(dims)
    avg = np.mean(data, axis=tuple(i for i in range(3) if i != axis))
    step_vec = voxel_info[axis][1]
    step = np.sqrt(sum(x**2 for x in step_vec)) * 0.529177
    positions = np.arange(dims[axis]) * step + origin[axis] * 0.529177
    return positions, avg

# Run all energy windows
z_positions = None
ldos_columns = []

for i, ec in enumerate(e_centers):
    emin = ec - de / 2
    emax = ec + de / 2
    label = f"grid_{i:03d}"

    cube = run_pp_ldos(PREFIX, OUTDIR, emin, emax, label)
    if cube:
        z_pos, pavg = parse_cube_planar(cube, axis=2)
        if z_positions is None:
            z_positions = z_pos
        ldos_columns.append(pavg)
        print(f"  [{i+1}/{len(e_centers)}] [{emin:.1f}, {emax:.1f}] eV done.")
    else:
        if z_positions is not None:
            ldos_columns.append(np.zeros_like(z_positions))
        print(f"  [{i+1}/{len(e_centers)}] [{emin:.1f}, {emax:.1f}] eV FAILED.")

# ══════════════════════════════════════════════════════════════════
#  Step 3: Plot E-vs-z LDOS map
# ══════════════════════════════════════════════════════════════════
if z_positions is not None and ldos_columns:
    ldos_2d = np.array(ldos_columns)

    fig, ax = plt.subplots(figsize=(10, 6))
    Z, E = np.meshgrid(z_positions, e_centers - e_fermi)
    im = ax.pcolormesh(Z, E, ldos_2d, shading="auto", cmap="hot_r",
                       vmin=0, vmax=np.percentile(ldos_2d[ldos_2d > 0], 95))
    plt.colorbar(im, ax=ax, label="LDOS (arb. units)")

    ax.axhline(0, color="cyan", linestyle="--", linewidth=1.0, label="$E_F$")
    ax.set_xlabel("z position (Angstrom)", fontsize=13)
    ax.set_ylabel("$E - E_F$ (eV)", fontsize=13)
    ax.set_title("Spatially-Resolved LDOS Map", fontsize=14)
    ax.legend(fontsize=11)

    plt.tight_layout()
    plt.savefig("ldos_map_complete.png", dpi=200, bbox_inches="tight")
    print("\nSaved: ldos_map_complete.png")

# ── Clean up ───────────────────────────────────────────────────────
import glob
for pattern in [f"{PREFIX}_ldos_grid_*.cube", f"{PREFIX}_ldos_grid_*.dat",
                f"{PREFIX}_pp_grid_*.in"]:
    for f in glob.glob(pattern):
        os.remove(f)
print("Cleaned up intermediate files.")
```

---

## Key Parameters

| Parameter | QE (pp.x) | VASP | Notes |
|---|---|---|---|
| Energy window min | `emin` in `&INPUTPP` (eV) | `EINT(1)` in INCAR (eV, relative to E_F) | Absolute eV in QE; relative to E_F in VASP |
| Energy window max | `emax` in `&INPUTPP` (eV) | `EINT(2)` in INCAR | Same convention as above |
| 3D output format | `output_format=6` (cube) | PARCHG (CHGCAR format) | Cube for VESTA; PARCHG readable by pymatgen |
| Planar average direction | `iflag=2`, `e3` vector | VASPKIT task 124 | Choose slab normal direction |
| Band selection | Not directly supported by pp.x `plot_num=10` | `IBAND` + `LPARD=.TRUE.` | VASP can decompose by band index |
| k-point selection | Not directly supported | `LSEPK=.TRUE.` + `KPUSE` | VASP can decompose by k-point |
| Wavefunction storage | `wf_collect=.true.` in SCF | `LWAVE=.TRUE.` | Required for post-processing |

## Interpreting Results

- **Plane-averaged LDOS profile**: Sharp peaks at atomic positions indicate localized states. Smooth profiles in vacuum indicate evanescent tails of surface states.
- **Surface states**: States that decay exponentially into the bulk but have significant weight at the surface. Visible as LDOS peaks only at surface layers.
- **Interface band alignment**: Compare LDOS on each side of the interface. The energy offset between valence/conduction features gives the band offset (Type I, II, or III).
- **Defect states**: Localized states near a point defect appear as LDOS concentrated at the defect site, especially in mid-gap energy windows.
- **Macroscopic averaging**: The raw planar average oscillates rapidly near atoms. A macroscopic average (convolving with a window equal to the interlayer spacing) reveals the electrostatic potential profile for band alignment.
- **E-vs-z map**: A 2D color plot showing LDOS as a function of position and energy. Band bending at interfaces appears as a tilt of the LDOS features.

## Common Issues

| Problem | Solution |
|---|---|
| pp.x: "could not find required data" | Ensure wavefunctions are saved (`wf_collect=.true.`). The SCF `outdir/prefix.save/` must contain `K*/evc*.dat` files. |
| PARCHG file is empty or zero | Check that `EINT` covers an energy range with actual states. Verify `LPARD=.TRUE.` and `NBMOD=-3`. Ensure WAVECAR exists. |
| Planar average is noisy | Use a larger energy window or increase k-grid density. Apply macroscopic averaging with window = interlayer spacing. |
| Cube file too large | Reduce the FFT grid by using a smaller `ecutrho`. For visualization only, coarser grids are acceptable. |
| VASP: "LPARD requires WAVECAR" | Run the SCF first with `LWAVE=.TRUE.` to generate WAVECAR. Then rerun with `LPARD=.TRUE.` and `ISTART=1`. |
| Band-decomposed PARCHG shows nothing | Verify `IBAND` indices are correct (1-based in VASP). Check that the selected bands have non-zero occupation or are within the computed eigenvalue range. |
| E-vs-z map has horizontal stripes | The energy windows are too wide. Reduce `de` to 0.2--0.3 eV for finer resolution (at the cost of more pp.x runs). |
| Charge not conserved in PARCHG | PARCHG only contains charge from states in the selected energy/band window. This is expected. It should not integrate to the total electron count. |
