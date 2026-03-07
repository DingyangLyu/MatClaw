# 2D Spin Texture Calculation

## When to Use

- You have a 2D material or surface with spin-orbit coupling (SOC) and need the spin texture around band extrema.
- You want to visualize spin-momentum locking in Rashba systems (e.g., BiTeI surface, Au(111) surface states).
- You need valley-dependent spin textures in transition metal dichalcogenides (e.g., MoS2, WSe2 at K/K' points).
- You are studying spin Hall or spin-valley coupling effects and need the k-resolved spin expectation values.
- You want to identify Rashba vs. Dresselhaus spin splitting patterns from the spin winding direction.
- You are comparing DFT spin textures with spin-resolved ARPES data.

## Method Selection

| Criterion | QE DFT | VASP DFT |
|---|---|---|
| SOC setup | `noncolin=.true.`, `lspinorb=.true.` | `LSORBIT=.TRUE.`, `LNONCOLLINEAR=.TRUE.` |
| Spin extraction | `projwfc.x` with `lsym=.false.` parses spin-projected DOS; or parse `pw.x` output directly | Parse PROCAR for spin components Sx, Sy, Sz per band per k-point |
| Pseudopotentials | Fully relativistic (`*_rel_*` or `*_FR_*`) | PAW potentials (standard, SOC handled internally) |
| 2D handling | `assume_isolated = '2D'` for Coulomb cutoff | Vacuum > 15 A along z |
| Output format | Text output parsed with Python | PROCAR file (fixed-format, machine-readable) |
| VASPKIT task | N/A | Menu 65 (tasks 651--653) |

## Prerequisites

- Quantum ESPRESSO 7.5 (`pw.x`, `projwfc.x`) with noncollinear + SOC support
- Fully relativistic pseudopotentials (FR-ONCV or PSlibrary `_rel` PPs)
- Python packages: `numpy`, `matplotlib`, `pymatgen`, `ase`
- A relaxed 2D structure with vacuum > 15 A along z
- For VASP: PROCAR file from a SOC NSCF calculation with `LORBIT=11` or `LORBIT=12`

---

## Detailed Steps

### Method A: QE DFT -- Spin Texture of Monolayer MoS2 (Valley Spin Texture)

Monolayer MoS2 has spin-valley coupling: SOC splits the valence band at K and K' with opposite spin polarization (Sz up at K, Sz down at K'). This example extracts the spin texture around the K point.

#### Step A1: SCF Calculation with SOC

```python
#!/usr/bin/env python3
"""
Step 1: SCF calculation for monolayer MoS2 with spin-orbit coupling.
MoS2 is a 2D semiconductor with strong valley spin-splitting (~150 meV at K).
"""
import os
import subprocess
import numpy as np

PSEUDO_DIR = os.path.abspath("./pseudo")
OUTDIR = os.path.abspath("./tmp_spintex_2d")
os.makedirs(OUTDIR, exist_ok=True)
PREFIX = "mos2_soc"

# MoS2 lattice parameters
a = 3.16  # Angstrom (in-plane)
c_vac = 20.0  # vacuum along z

scf_input = f"""&CONTROL
    calculation  = 'scf'
    prefix       = '{PREFIX}'
    outdir       = '{OUTDIR}'
    pseudo_dir   = '{PSEUDO_DIR}'
    tprnfor      = .true.
/
&SYSTEM
    ibrav        = 0
    nat          = 3
    ntyp         = 2
    ecutwfc      = 60.0
    ecutrho      = 480.0
    occupations  = 'smearing'
    smearing     = 'cold'
    degauss      = 0.005
    assume_isolated = '2D'
    noncolin     = .true.
    lspinorb     = .true.
/
&ELECTRONS
    conv_thr     = 1.0d-9
    mixing_beta  = 0.3
/

CELL_PARAMETERS (angstrom)
  {a:.10f}   0.0000000000   0.0000000000
  {-a/2:.10f}   {a*np.sqrt(3)/2:.10f}   0.0000000000
  0.0000000000   0.0000000000   {c_vac:.10f}

ATOMIC_SPECIES
  Mo  95.94    Mo.rel-pbe-spn-kjpaw_psl.1.0.0.UPF
  S   32.065   S.rel-pbe-n-kjpaw_psl.1.0.0.UPF

ATOMIC_POSITIONS (crystal)
  Mo  0.3333333333  0.6666666667  0.5000000000
  S   0.6666666667  0.3333333333  {0.5 + 1.56/c_vac:.10f}
  S   0.6666666667  0.3333333333  {0.5 - 1.56/c_vac:.10f}

K_POINTS (automatic)
  12 12 1  0 0 0
"""

with open(f"{PREFIX}_scf.in", "w") as f:
    f.write(scf_input)

print("Running SCF for MoS2 with SOC...")
result = subprocess.run(
    ["mpirun", "-np", "4", "pw.x", "-in", f"{PREFIX}_scf.in"],
    capture_output=True, text=True, timeout=1200
)
with open(f"{PREFIX}_scf.out", "w") as f:
    f.write(result.stdout)

if "convergence has been achieved" in result.stdout:
    print("SCF converged.")
else:
    print("WARNING: SCF may not have converged. Check output.")
    print("Try reducing mixing_beta or increasing electron_maxstep.")
```

#### Step A2: Generate Dense K-Mesh Around K Point and Run NSCF

```python
#!/usr/bin/env python3
"""
Step 2: Generate a dense circular k-mesh around the K point of MoS2.
The K point in the hexagonal BZ is at (1/3, 1/3, 0) in crystal coordinates.
We sample a disk of k-points around K to capture the spin texture.
"""
import os
import subprocess
import numpy as np

PSEUDO_DIR = os.path.abspath("./pseudo")
OUTDIR = os.path.abspath("./tmp_spintex_2d")
PREFIX = "mos2_soc"

a = 3.16
c_vac = 20.0

# K point in crystal coordinates for hexagonal lattice
K_point = np.array([1/3, 1/3, 0.0])

# Generate dense k-mesh: concentric rings around K
# Use a square grid in fractional coords, then filter to a disk
dk = 0.08  # radius in fractional coordinates (adjust for desired extent)
nk_side = 31  # grid points per side -> nk_side x nk_side grid

kpoints = []
kx_list, ky_list = [], []

for i in range(nk_side):
    for j in range(nk_side):
        dkx = -dk + 2 * dk * i / (nk_side - 1)
        dky = -dk + 2 * dk * j / (nk_side - 1)
        r = np.sqrt(dkx**2 + dky**2)
        if r <= dk:  # circular disk
            kx = K_point[0] + dkx
            ky = K_point[1] + dky
            kz = 0.0
            kpoints.append(f"  {kx:.10f}  {ky:.10f}  {kz:.10f}  1.0")
            kx_list.append(dkx)
            ky_list.append(dky)

nk_total = len(kpoints)
print(f"Generated {nk_total} k-points in disk around K = ({K_point[0]:.4f}, {K_point[1]:.4f}, 0)")

kpoints_card = f"K_POINTS (crystal)\n{nk_total}\n" + "\n".join(kpoints) + "\n"

# Save k-point offsets for later plotting
np.savetxt("kpoints_dk.dat", np.column_stack([kx_list, ky_list]),
           header="dkx  dky  (fractional coords relative to K)")

# Write NSCF input
nscf_input = f"""&CONTROL
    calculation  = 'nscf'
    prefix       = '{PREFIX}'
    outdir       = '{OUTDIR}'
    pseudo_dir   = '{PSEUDO_DIR}'
    verbosity    = 'high'
/
&SYSTEM
    ibrav        = 0
    nat          = 3
    ntyp         = 2
    ecutwfc      = 60.0
    ecutrho      = 480.0
    occupations  = 'smearing'
    smearing     = 'cold'
    degauss      = 0.005
    nbnd         = 30
    assume_isolated = '2D'
    noncolin     = .true.
    lspinorb     = .true.
    nosym        = .true.
    noinv        = .true.
/
&ELECTRONS
    conv_thr     = 1.0d-9
/

CELL_PARAMETERS (angstrom)
  {a:.10f}   0.0000000000   0.0000000000
  {-a/2:.10f}   {a*np.sqrt(3)/2:.10f}   0.0000000000
  0.0000000000   0.0000000000   {c_vac:.10f}

ATOMIC_SPECIES
  Mo  95.94    Mo.rel-pbe-spn-kjpaw_psl.1.0.0.UPF
  S   32.065   S.rel-pbe-n-kjpaw_psl.1.0.0.UPF

ATOMIC_POSITIONS (crystal)
  Mo  0.3333333333  0.6666666667  0.5000000000
  S   0.6666666667  0.3333333333  {0.5 + 1.56/c_vac:.10f}
  S   0.6666666667  0.3333333333  {0.5 - 1.56/c_vac:.10f}

{kpoints_card}
"""

with open(f"{PREFIX}_nscf_spintex.in", "w") as f:
    f.write(nscf_input)

print(f"Running NSCF on {nk_total} k-points around K...")
result = subprocess.run(
    ["mpirun", "-np", "4", "pw.x", "-in", f"{PREFIX}_nscf_spintex.in"],
    capture_output=True, text=True, timeout=3600
)
with open(f"{PREFIX}_nscf_spintex.out", "w") as f:
    f.write(result.stdout)

if result.returncode == 0:
    print("NSCF completed successfully.")
else:
    print("ERROR in NSCF!")
    print(result.stderr[-500:] if result.stderr else "")
```

#### Step A3: Run projwfc.x to Extract Spin Projections

```python
#!/usr/bin/env python3
"""
Step 3: Run projwfc.x to extract spin-projected information.
projwfc.x in noncollinear mode outputs Sx, Sy, Sz projections per state.
"""
import os
import subprocess

OUTDIR = os.path.abspath("./tmp_spintex_2d")
PREFIX = "mos2_soc"

# projwfc.x input for noncollinear + SOC calculation
projwfc_input = f"""&PROJWFC
    prefix       = '{PREFIX}'
    outdir       = '{OUTDIR}'
    filproj      = '{PREFIX}_proj'
    lsym         = .false.
    lwrite_overlaps = .false.
    filpdos      = '{PREFIX}_pdos'
/
"""

with open(f"{PREFIX}_projwfc.in", "w") as f:
    f.write(projwfc_input)

print("Running projwfc.x for spin projections...")
result = subprocess.run(
    ["mpirun", "-np", "4", "projwfc.x", "-in", f"{PREFIX}_projwfc.in"],
    capture_output=True, text=True, timeout=3600
)
with open(f"{PREFIX}_projwfc.out", "w") as f:
    f.write(result.stdout)

if result.returncode == 0:
    print("projwfc.x completed.")
    print("Output files generated:")
    print(f"  {PREFIX}_proj.projwfc_up  -- spin-up projected DOS")
    print(f"  {PREFIX}_proj.projwfc_down -- spin-down projected DOS")
    print(f"  {PREFIX}_projwfc.out -- main output with spin expectation values")
else:
    print("ERROR in projwfc.x!")
    print(result.stderr[-500:] if result.stderr else "")
```

#### Step A4: Parse Spin Expectation Values from QE Output

```python
#!/usr/bin/env python3
"""
Step 4: Parse spin expectation values <Sx>, <Sy>, <Sz> for each band at each k-point
from the pw.x NSCF output (verbosity='high') or projwfc.x output.

QE with noncolin=.true. and verbosity='high' prints spin components in the
NSCF output. We parse those directly.
"""
import re
import numpy as np
import os

PREFIX = "mos2_soc"
NSCF_OUTPUT = f"{PREFIX}_nscf_spintex.out"

def parse_qe_spin_texture(nscf_output):
    """
    Parse spin expectation values from QE NSCF output (verbosity='high').

    In noncollinear QE with verbosity='high', the output contains blocks like:

        k = ... bands (ev):
        ...
        Spin expectation values (Sx, Sy, Sz) for each band:
        ...

    For each k-point, QE prints eigenvalues and (if available) spin projections.
    When verbosity='high' and noncolin=.true., pw.x prints:
        S_x, S_y, S_z for each band after the eigenvalue block.

    Returns
    -------
    kpoints : ndarray, shape (nk, 3) -- k-point coordinates (crystal)
    eigenvalues : ndarray, shape (nk, nbnd) -- eigenvalues in eV
    sx, sy, sz : ndarray, shape (nk, nbnd) -- spin components per band
    """
    with open(nscf_output, "r") as f:
        content = f.read()

    # Parse k-points and eigenvalues
    kpoints = []
    eigenvalues = []
    sx_all, sy_all, sz_all = [], [], []

    # Pattern to find k-point blocks
    # QE prints: "k =  0.3333  0.3333  0.0000 ..."
    kpoint_pattern = re.compile(
        r"k\s*=\s*([-\d.]+)\s+([-\d.]+)\s+([-\d.]+)\s+.*?\n"
        r"\s*bands \(ev\):\s*\n"
        r"(.*?)\n\s*\n",
        re.DOTALL
    )

    matches = kpoint_pattern.findall(content)

    for match in matches:
        kx, ky, kz = float(match[0]), float(match[1]), float(match[2])
        kpoints.append([kx, ky, kz])

        # Parse eigenvalues from the band block
        eig_text = match[3]
        eigs = [float(x) for x in re.findall(r'([-\d.]+)', eig_text)]
        eigenvalues.append(eigs)

    kpoints = np.array(kpoints)
    # Pad eigenvalues to uniform length
    nbnd = max(len(e) for e in eigenvalues) if eigenvalues else 0
    eig_array = np.full((len(eigenvalues), nbnd), np.nan)
    for i, eigs in enumerate(eigenvalues):
        eig_array[i, :len(eigs)] = eigs

    # Parse spin components
    # QE noncollinear output contains lines like:
    # "  Sx=  0.0000  Sy=  0.0000  Sz=  0.4500"
    # These appear after the eigenvalue block for each k-point.
    spin_pattern = re.compile(
        r"k\s*=\s*[-\d.]+\s+[-\d.]+\s+[-\d.]+.*?"
        r"(?:Sx=\s*([-\d.]+)\s+Sy=\s*([-\d.]+)\s+Sz=\s*([-\d.]+)\s*\n)+",
        re.DOTALL
    )

    # Alternative: parse the spin from the state-by-state listing
    # In QE 7.x with noncolin + verbosity='high', each state has:
    # "     e(  1) =   -5.000 eV   Sx=  0.000  Sy=  0.000  Sz=  0.500"
    state_spin_pattern = re.compile(
        r"Sx=\s*([-\d.]+)\s+Sy=\s*([-\d.]+)\s+Sz=\s*([-\d.]+)"
    )

    # Re-parse with positional tracking
    lines = content.split("\n")
    current_k_idx = -1
    sx_k, sy_k, sz_k = [], [], []

    for line in lines:
        if re.match(r"\s*k\s*=\s*[-\d.]+\s+[-\d.]+\s+[-\d.]+", line):
            if sx_k and current_k_idx >= 0:
                sx_all.append(sx_k)
                sy_all.append(sy_k)
                sz_all.append(sz_k)
            current_k_idx += 1
            sx_k, sy_k, sz_k = [], [], []

        m = state_spin_pattern.search(line)
        if m:
            sx_k.append(float(m.group(1)))
            sy_k.append(float(m.group(2)))
            sz_k.append(float(m.group(3)))

    # Append last k-point
    if sx_k:
        sx_all.append(sx_k)
        sy_all.append(sy_k)
        sz_all.append(sz_k)

    # Convert to arrays
    if sx_all:
        nbnd_spin = max(len(s) for s in sx_all)
        sx_arr = np.full((len(sx_all), nbnd_spin), 0.0)
        sy_arr = np.full((len(sx_all), nbnd_spin), 0.0)
        sz_arr = np.full((len(sx_all), nbnd_spin), 0.0)
        for i in range(len(sx_all)):
            n = len(sx_all[i])
            sx_arr[i, :n] = sx_all[i]
            sy_arr[i, :n] = sy_all[i]
            sz_arr[i, :n] = sz_all[i]
    else:
        sx_arr = np.zeros_like(eig_array)
        sy_arr = np.zeros_like(eig_array)
        sz_arr = np.zeros_like(eig_array)
        print("WARNING: No spin expectation values found in output.")
        print("Ensure verbosity='high' and noncolin=.true. in the NSCF input.")

    return kpoints, eig_array, sx_arr, sy_arr, sz_arr


# Parse the data
kpoints, eigenvalues, sx, sy, sz = parse_qe_spin_texture(NSCF_OUTPUT)

print(f"Parsed {len(kpoints)} k-points, {eigenvalues.shape[1]} bands")
print(f"Eigenvalue range: [{np.nanmin(eigenvalues):.3f}, {np.nanmax(eigenvalues):.3f}] eV")

# Save parsed data for plotting
np.savez("spin_texture_2d_data.npz",
         kpoints=kpoints, eigenvalues=eigenvalues,
         sx=sx, sy=sy, sz=sz)
print("Data saved to spin_texture_2d_data.npz")
```

#### Step A5: Plot Spin Texture on Constant-Energy Contour (Quiver Plot)

```python
#!/usr/bin/env python3
"""
Step 5: Plot the 2D spin texture around the K point of MoS2.
Produces a quiver plot with arrows showing (Sx, Sy) and color showing Sz.
Includes both the data-driven approach (from QE) and a demonstration
with synthetic data for the Rashba and valley spin texture cases.
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib import cm

# =========================================================================
# Option 1: Load parsed QE data
# =========================================================================
def plot_from_qe_data():
    """Plot spin texture from parsed QE NSCF output."""
    data = np.load("spin_texture_2d_data.npz")
    kpoints = data["kpoints"]
    eigenvalues = data["eigenvalues"]
    sx = data["sx"]
    sy = data["sy"]
    sz = data["sz"]

    # K point for reference
    K_point = np.array([1/3, 1/3, 0.0])

    # Get Fermi energy from SCF output
    import re
    e_fermi = 0.0
    try:
        with open("mos2_soc_scf.out", "r") as f:
            for line in f:
                if "highest occupied" in line:
                    m = re.search(r":\s+([-\d.]+)", line)
                    if m:
                        e_fermi = float(m.group(1))
                if "the Fermi energy is" in line:
                    m = re.search(r"is\s+([-\d.]+)", line)
                    if m:
                        e_fermi = float(m.group(1))
    except FileNotFoundError:
        pass

    # Select band index: topmost valence band (VBM)
    # For MoS2 with 3 atoms, 13 valence electrons (Mo: 6 + S: 6 + SOC doubling)
    # With SOC, bands are spinor -- find the band closest to VBM
    vbm_band_idx = None
    for ib in range(eigenvalues.shape[1]):
        band_max = np.nanmax(eigenvalues[:, ib])
        band_min = np.nanmin(eigenvalues[:, ib])
        if band_max <= e_fermi + 0.1 and band_max > e_fermi - 0.5:
            vbm_band_idx = ib

    if vbm_band_idx is None:
        # Fallback: pick band with eigenvalue closest to Fermi at K
        k_at_K = np.argmin(np.linalg.norm(kpoints - K_point, axis=1))
        vbm_band_idx = np.argmin(np.abs(eigenvalues[k_at_K, :] - e_fermi))

    print(f"Plotting spin texture for band index {vbm_band_idx}")

    # k-point offsets relative to K (in fractional coords)
    dk = kpoints[:, :2] - K_point[:2]

    fig, ax = plt.subplots(figsize=(8, 7))

    # Color by Sz component
    sz_band = sz[:, vbm_band_idx]
    sx_band = sx[:, vbm_band_idx]
    sy_band = sy[:, vbm_band_idx]

    norm = Normalize(vmin=-0.5, vmax=0.5)
    colors = cm.coolwarm(norm(sz_band))

    # Quiver plot: arrows show (Sx, Sy), color shows Sz
    q = ax.quiver(dk[:, 0], dk[:, 1], sx_band, sy_band,
                  sz_band, cmap="coolwarm", norm=norm,
                  scale=15, width=0.004, headwidth=3, headlength=4)

    cb = plt.colorbar(cm.ScalarMappable(norm=norm, cmap="coolwarm"), ax=ax)
    cb.set_label(r"$\langle S_z \rangle$ ($\hbar/2$)", fontsize=13)

    ax.set_xlabel(r"$\Delta k_x$ (crystal)", fontsize=13)
    ax.set_ylabel(r"$\Delta k_y$ (crystal)", fontsize=13)
    ax.set_title(f"MoS$_2$ Spin Texture around K (band {vbm_band_idx})",
                 fontsize=14, fontweight="bold")
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("spin_texture_2d_mos2.png", dpi=200, bbox_inches="tight")
    print("Saved: spin_texture_2d_mos2.png")


# =========================================================================
# Option 2: Demonstration with synthetic data
# =========================================================================
def plot_synthetic_spin_textures():
    """
    Generate and plot synthetic spin textures for two common types:
    (a) Rashba-type: helical spin winding in the kx-ky plane
    (b) Valley-type (MoS2): Sz locked to valley (K vs K')
    """
    # Generate k-mesh (disk)
    nk = 31
    dk_max = 0.08
    dkx_1d = np.linspace(-dk_max, dk_max, nk)
    dky_1d = np.linspace(-dk_max, dk_max, nk)
    DKX, DKY = np.meshgrid(dkx_1d, dky_1d)
    R = np.sqrt(DKX**2 + DKY**2)
    mask = R <= dk_max
    dkx = DKX[mask]
    dky = DKY[mask]
    r = np.sqrt(dkx**2 + dky**2)
    theta = np.arctan2(dky, dkx)

    fig, axes = plt.subplots(1, 2, figsize=(16, 7))

    # --- (a) Rashba spin texture ---
    # Rashba: S is tangential, Sz = 0
    # Inner branch: clockwise, Outer branch: counterclockwise
    # For the outer branch: Sx = -sin(theta), Sy = cos(theta)
    ax = axes[0]

    sx_rashba = -np.sin(theta) * 0.5
    sy_rashba = np.cos(theta) * 0.5
    sz_rashba = np.zeros_like(r)

    # Sz is zero for ideal Rashba -- color by |S_xy| magnitude instead
    s_mag = np.sqrt(sx_rashba**2 + sy_rashba**2)

    norm = Normalize(vmin=-0.5, vmax=0.5)
    q = ax.quiver(dkx, dky, sx_rashba, sy_rashba,
                  sz_rashba, cmap="coolwarm", norm=norm,
                  scale=12, width=0.004, headwidth=3, headlength=4)

    # Draw constant-energy contour (circle)
    theta_circle = np.linspace(0, 2*np.pi, 100)
    for r_ring in [0.03, 0.06]:
        ax.plot(r_ring * np.cos(theta_circle), r_ring * np.sin(theta_circle),
                "k--", linewidth=0.8, alpha=0.4)

    cb1 = plt.colorbar(cm.ScalarMappable(norm=norm, cmap="coolwarm"), ax=ax, shrink=0.85)
    cb1.set_label(r"$\langle S_z \rangle$ ($\hbar/2$)", fontsize=12)

    ax.set_xlabel(r"$\Delta k_x$ (frac.)", fontsize=12)
    ax.set_ylabel(r"$\Delta k_y$ (frac.)", fontsize=12)
    ax.set_title("(a) Rashba Spin Texture\n(e.g., BiTeI surface)",
                 fontsize=13, fontweight="bold")
    ax.set_aspect("equal")
    ax.set_xlim(-dk_max*1.1, dk_max*1.1)
    ax.set_ylim(-dk_max*1.1, dk_max*1.1)
    ax.grid(True, alpha=0.3)
    ax.annotate("Helical spin\nwinding", xy=(0.04, 0.04), fontsize=10,
                fontstyle="italic", color="purple")

    # --- (b) Valley spin texture (MoS2 at K) ---
    # At K point: Sz is dominant, locked to valley index
    # Sz > 0 at K, Sz < 0 at K'
    # Small in-plane components from warping
    ax = axes[1]

    # Sz decays away from K but stays dominant
    sz_valley = 0.5 * np.exp(-(r / dk_max)**2 * 2)
    # Small warping-induced in-plane components (3-fold symmetry)
    sx_valley = 0.05 * np.cos(3 * theta) * (r / dk_max)
    sy_valley = 0.05 * np.sin(3 * theta) * (r / dk_max)

    norm2 = Normalize(vmin=-0.5, vmax=0.5)
    q = ax.quiver(dkx, dky, sx_valley, sy_valley,
                  sz_valley, cmap="coolwarm", norm=norm2,
                  scale=12, width=0.004, headwidth=3, headlength=4)

    for r_ring in [0.03, 0.06]:
        ax.plot(r_ring * np.cos(theta_circle), r_ring * np.sin(theta_circle),
                "k--", linewidth=0.8, alpha=0.4)

    cb2 = plt.colorbar(cm.ScalarMappable(norm=norm2, cmap="coolwarm"), ax=ax, shrink=0.85)
    cb2.set_label(r"$\langle S_z \rangle$ ($\hbar/2$)", fontsize=12)

    ax.set_xlabel(r"$\Delta k_x$ (frac.)", fontsize=12)
    ax.set_ylabel(r"$\Delta k_y$ (frac.)", fontsize=12)
    ax.set_title("(b) Valley Spin Texture at K\n(e.g., MoS$_2$ VBM)",
                 fontsize=13, fontweight="bold")
    ax.set_aspect("equal")
    ax.set_xlim(-dk_max*1.1, dk_max*1.1)
    ax.set_ylim(-dk_max*1.1, dk_max*1.1)
    ax.grid(True, alpha=0.3)
    ax.annotate(r"$S_z > 0$ at K", xy=(0.0, -0.065), fontsize=11,
                fontweight="bold", color="red", ha="center")
    ax.annotate(r"$S_z < 0$ at K$'$", xy=(0.0, -0.075), fontsize=10,
                fontstyle="italic", color="blue", ha="center")

    plt.suptitle("2D Spin Textures: Rashba vs Valley Types",
                 fontsize=15, fontweight="bold", y=1.02)
    plt.tight_layout()
    plt.savefig("spin_texture_2d_types.png", dpi=200, bbox_inches="tight")
    print("Saved: spin_texture_2d_types.png")


# =========================================================================
# Run plotting
# =========================================================================
import os
if os.path.exists("spin_texture_2d_data.npz"):
    print("Found QE data -- plotting from calculation results.")
    plot_from_qe_data()
else:
    print("No QE data found -- generating synthetic demonstration plots.")

# Always generate the type-comparison figure
plot_synthetic_spin_textures()
```

#### Complete Single-Script Workflow (QE)

```python
#!/usr/bin/env python3
"""
Complete 2D spin texture workflow for MoS2 using QE.
Runs: SCF (SOC) -> NSCF (dense disk around K) -> projwfc.x -> parse -> plot.
"""
import os
import subprocess
import re
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib import cm

# ======================================================================
# Configuration
# ======================================================================
PSEUDO_DIR = os.path.abspath("./pseudo")
OUTDIR = os.path.abspath("./tmp_spintex_2d_full")
os.makedirs(OUTDIR, exist_ok=True)
PREFIX = "mos2_soc"
NPROC = 4

a = 3.16          # MoS2 in-plane lattice constant (Angstrom)
c_vac = 20.0      # vacuum along z
dz_frac = 1.56 / c_vac  # S-Mo vertical separation in fractional coords

K_point = np.array([1/3, 1/3, 0.0])  # K point in crystal coords
dk_radius = 0.08  # disk radius in fractional coords
nk_side = 25      # k-points per side of grid

# ======================================================================
# Step 1: SCF with SOC
# ======================================================================
scf_input = f"""&CONTROL
    calculation  = 'scf'
    prefix       = '{PREFIX}'
    outdir       = '{OUTDIR}'
    pseudo_dir   = '{PSEUDO_DIR}'
/
&SYSTEM
    ibrav        = 0
    nat          = 3
    ntyp         = 2
    ecutwfc      = 60.0
    ecutrho      = 480.0
    occupations  = 'smearing'
    smearing     = 'cold'
    degauss      = 0.005
    assume_isolated = '2D'
    noncolin     = .true.
    lspinorb     = .true.
/
&ELECTRONS
    conv_thr     = 1.0d-9
    mixing_beta  = 0.3
/

CELL_PARAMETERS (angstrom)
  {a:.10f}   0.0000000000   0.0000000000
  {-a/2:.10f}   {a*np.sqrt(3)/2:.10f}   0.0000000000
  0.0000000000   0.0000000000   {c_vac:.10f}

ATOMIC_SPECIES
  Mo  95.94    Mo.rel-pbe-spn-kjpaw_psl.1.0.0.UPF
  S   32.065   S.rel-pbe-n-kjpaw_psl.1.0.0.UPF

ATOMIC_POSITIONS (crystal)
  Mo  0.3333333333  0.6666666667  0.5000000000
  S   0.6666666667  0.3333333333  {0.5 + dz_frac:.10f}
  S   0.6666666667  0.3333333333  {0.5 - dz_frac:.10f}

K_POINTS (automatic)
  12 12 1  0 0 0
"""

with open(f"{PREFIX}_scf.in", "w") as f:
    f.write(scf_input)

print("[1/4] Running SCF with SOC...")
r = subprocess.run(["mpirun", "-np", str(NPROC), "pw.x", "-in", f"{PREFIX}_scf.in"],
                   capture_output=True, text=True, timeout=1200)
with open(f"{PREFIX}_scf.out", "w") as f:
    f.write(r.stdout)
assert "convergence has been achieved" in r.stdout, "SCF did not converge!"

# Extract Fermi energy
e_fermi = 0.0
for line in r.stdout.split("\n"):
    if "the Fermi energy is" in line:
        m = re.search(r"is\s+([-\d.]+)", line)
        if m:
            e_fermi = float(m.group(1))
    if "highest occupied" in line:
        m = re.search(r":\s+([-\d.]+)", line)
        if m:
            e_fermi = float(m.group(1))
print(f"      SCF converged. E_Fermi = {e_fermi:.4f} eV")

# ======================================================================
# Step 2: Generate k-mesh disk around K and run NSCF
# ======================================================================
kpoints = []
dkx_list, dky_list = [], []
for i in range(nk_side):
    for j in range(nk_side):
        dkx = -dk_radius + 2 * dk_radius * i / (nk_side - 1)
        dky = -dk_radius + 2 * dk_radius * j / (nk_side - 1)
        if np.sqrt(dkx**2 + dky**2) <= dk_radius:
            kpoints.append(f"  {K_point[0]+dkx:.10f}  {K_point[1]+dky:.10f}  0.0  1.0")
            dkx_list.append(dkx)
            dky_list.append(dky)

nk_total = len(kpoints)
kpoints_card = f"K_POINTS (crystal)\n{nk_total}\n" + "\n".join(kpoints) + "\n"

nscf_input = f"""&CONTROL
    calculation  = 'nscf'
    prefix       = '{PREFIX}'
    outdir       = '{OUTDIR}'
    pseudo_dir   = '{PSEUDO_DIR}'
    verbosity    = 'high'
/
&SYSTEM
    ibrav        = 0
    nat          = 3
    ntyp         = 2
    ecutwfc      = 60.0
    ecutrho      = 480.0
    occupations  = 'smearing'
    smearing     = 'cold'
    degauss      = 0.005
    nbnd         = 30
    assume_isolated = '2D'
    noncolin     = .true.
    lspinorb     = .true.
    nosym        = .true.
    noinv        = .true.
/
&ELECTRONS
    conv_thr     = 1.0d-9
/

CELL_PARAMETERS (angstrom)
  {a:.10f}   0.0000000000   0.0000000000
  {-a/2:.10f}   {a*np.sqrt(3)/2:.10f}   0.0000000000
  0.0000000000   0.0000000000   {c_vac:.10f}

ATOMIC_SPECIES
  Mo  95.94    Mo.rel-pbe-spn-kjpaw_psl.1.0.0.UPF
  S   32.065   S.rel-pbe-n-kjpaw_psl.1.0.0.UPF

ATOMIC_POSITIONS (crystal)
  Mo  0.3333333333  0.6666666667  0.5000000000
  S   0.6666666667  0.3333333333  {0.5 + dz_frac:.10f}
  S   0.6666666667  0.3333333333  {0.5 - dz_frac:.10f}

{kpoints_card}
"""

with open(f"{PREFIX}_nscf.in", "w") as f:
    f.write(nscf_input)

print(f"[2/4] Running NSCF on {nk_total} k-points around K...")
r = subprocess.run(["mpirun", "-np", str(NPROC), "pw.x", "-in", f"{PREFIX}_nscf.in"],
                   capture_output=True, text=True, timeout=3600)
with open(f"{PREFIX}_nscf.out", "w") as f:
    f.write(r.stdout)
assert r.returncode == 0, "NSCF failed!"
print("      NSCF completed.")

# ======================================================================
# Step 3: Run projwfc.x
# ======================================================================
projwfc_in = f"""&PROJWFC
    prefix       = '{PREFIX}'
    outdir       = '{OUTDIR}'
    filproj      = '{PREFIX}_proj'
    lsym         = .false.
    filpdos      = '{PREFIX}_pdos'
/
"""
with open(f"{PREFIX}_projwfc.in", "w") as f:
    f.write(projwfc_in)

print("[3/4] Running projwfc.x...")
r = subprocess.run(["mpirun", "-np", str(NPROC), "projwfc.x",
                     "-in", f"{PREFIX}_projwfc.in"],
                   capture_output=True, text=True, timeout=3600)
with open(f"{PREFIX}_projwfc.out", "w") as f:
    f.write(r.stdout)
print("      projwfc.x completed.")

# ======================================================================
# Step 4: Parse spin texture and plot
# ======================================================================
print("[4/4] Parsing and plotting spin texture...")

# Parse spin components from NSCF output
with open(f"{PREFIX}_nscf.out", "r") as f:
    content = f.read()

kpoints_parsed = []
eigenvalues_all = []
sx_all, sy_all, sz_all = [], [], []

lines = content.split("\n")
current_eigs = []
current_sx, current_sy, current_sz = [], [], []
in_kblock = False

for line in lines:
    km = re.match(r"\s*k\s*=\s*([-\d.]+)\s+([-\d.]+)\s+([-\d.]+)", line)
    if km:
        if current_eigs:
            eigenvalues_all.append(current_eigs)
            sx_all.append(current_sx)
            sy_all.append(current_sy)
            sz_all.append(current_sz)
        kpoints_parsed.append([float(km.group(1)), float(km.group(2)), float(km.group(3))])
        current_eigs = []
        current_sx, current_sy, current_sz = [], [], []
        in_kblock = True

    sm = re.search(r"Sx=\s*([-\d.]+)\s+Sy=\s*([-\d.]+)\s+Sz=\s*([-\d.]+)", line)
    if sm:
        current_sx.append(float(sm.group(1)))
        current_sy.append(float(sm.group(2)))
        current_sz.append(float(sm.group(3)))

# Append last block
if current_eigs or current_sx:
    eigenvalues_all.append(current_eigs)
    sx_all.append(current_sx)
    sy_all.append(current_sy)
    sz_all.append(current_sz)

kpoints_arr = np.array(kpoints_parsed)
dkx_arr = np.array(dkx_list)
dky_arr = np.array(dky_list)

# Convert spin arrays to numpy
nbnd_spin = max(len(s) for s in sx_all) if sx_all else 1
nk = len(sx_all)
sx_arr = np.zeros((nk, nbnd_spin))
sy_arr = np.zeros((nk, nbnd_spin))
sz_arr = np.zeros((nk, nbnd_spin))
for i in range(nk):
    n = len(sx_all[i])
    sx_arr[i, :n] = sx_all[i]
    sy_arr[i, :n] = sy_all[i]
    sz_arr[i, :n] = sz_all[i]

# Find VBM band index (highest band with eigenvalue <= E_Fermi at K)
# Fallback: use the band with strongest Sz at the K point center
k_center_idx = np.argmin(np.linalg.norm(
    kpoints_arr[:, :2] - K_point[:2], axis=1))
vbm_idx = np.argmax(np.abs(sz_arr[k_center_idx, :]))

print(f"  Selected band index: {vbm_idx} (strongest Sz at K)")

# Plot
fig, ax = plt.subplots(figsize=(8, 7))

norm = Normalize(vmin=-0.5, vmax=0.5)
q = ax.quiver(dkx_arr, dky_arr,
              sx_arr[:len(dkx_arr), vbm_idx],
              sy_arr[:len(dky_arr), vbm_idx],
              sz_arr[:len(dkx_arr), vbm_idx],
              cmap="coolwarm", norm=norm,
              scale=15, width=0.004, headwidth=3)

cb = plt.colorbar(cm.ScalarMappable(norm=norm, cmap="coolwarm"), ax=ax)
cb.set_label(r"$\langle S_z \rangle$ ($\hbar/2$)", fontsize=13)

# Draw constant-energy contour rings
theta_c = np.linspace(0, 2*np.pi, 100)
for r_ring in [dk_radius/3, 2*dk_radius/3]:
    ax.plot(r_ring*np.cos(theta_c), r_ring*np.sin(theta_c),
            "k--", linewidth=0.8, alpha=0.4)

ax.plot(0, 0, "k+", markersize=12, markeredgewidth=2)
ax.annotate("K", xy=(0.002, 0.002), fontsize=12, fontweight="bold")

ax.set_xlabel(r"$\Delta k_x$ (crystal)", fontsize=13)
ax.set_ylabel(r"$\Delta k_y$ (crystal)", fontsize=13)
ax.set_title(f"MoS$_2$ Spin Texture around K (band {vbm_idx})\n"
             r"Arrows: $(S_x, S_y)$, Color: $S_z$",
             fontsize=14, fontweight="bold")
ax.set_aspect("equal")
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("spin_texture_2d_mos2_full.png", dpi=200, bbox_inches="tight")
print(f"Saved: spin_texture_2d_mos2_full.png")
```

### Method B: VASP -- Spin Texture from PROCAR Parsing

#### Step B1: VASP Input Files for SOC NSCF on Dense K-Mesh

```python
#!/usr/bin/env python3
"""
Generate VASP input files for a 2D spin texture calculation of MoS2.
Workflow: SCF (ICHARG=2) -> NSCF on dense k-mesh around K (ICHARG=11).
"""
import os
import numpy as np

WORK_DIR = os.path.abspath("vasp_spintex_2d")
os.makedirs(WORK_DIR, exist_ok=True)

# ── POSCAR: Monolayer MoS2 ──────────────────────────────────────────
a = 3.16
c_vac = 20.0
dz = 1.56  # S-Mo vertical distance (Angstrom)

poscar = f"""MoS2 monolayer
1.0
  {a:.10f}   0.0000000000   0.0000000000
  {-a/2:.10f}   {a*np.sqrt(3)/2:.10f}   0.0000000000
  0.0000000000   0.0000000000   {c_vac:.10f}
Mo S
1  2
Direct
  0.3333333333  0.6666666667  0.5000000000
  0.6666666667  0.3333333333  {0.5 + dz/c_vac:.10f}
  0.6666666667  0.3333333333  {0.5 - dz/c_vac:.10f}
"""

with open(os.path.join(WORK_DIR, "POSCAR"), "w") as f:
    f.write(poscar)

# ── INCAR for SCF (step 1) ──────────────────────────────────────────
incar_scf = """# MoS2 SOC SCF
SYSTEM   = MoS2_SOC
ENCUT    = 450
PREC     = Accurate
EDIFF    = 1E-8
ISMEAR   = 0
SIGMA    = 0.05
LREAL    = .FALSE.
# SOC settings
LSORBIT  = .TRUE.
LNONCOLLINEAR = .TRUE.
# Write CHGCAR for NSCF step
LCHARG   = .TRUE.
LWAVE    = .FALSE.
# Electronic
NELM     = 200
"""

with open(os.path.join(WORK_DIR, "INCAR_SCF"), "w") as f:
    f.write(incar_scf)

# ── INCAR for NSCF spin texture (step 2) ────────────────────────────
incar_nscf = """# MoS2 SOC NSCF for spin texture
SYSTEM   = MoS2_SOC_SPINTEX
ENCUT    = 450
PREC     = Accurate
EDIFF    = 1E-8
ISMEAR   = 0
SIGMA    = 0.05
LREAL    = .FALSE.
# SOC settings
LSORBIT  = .TRUE.
LNONCOLLINEAR = .TRUE.
# NSCF: read charge density
ICHARG   = 11
# Write PROCAR with spin projections
LORBIT   = 11
LWAVE    = .FALSE.
LCHARG   = .FALSE.
# Electronic
NELM     = 1
"""

with open(os.path.join(WORK_DIR, "INCAR_NSCF"), "w") as f:
    f.write(incar_nscf)

# ── KPOINTS for SCF ─────────────────────────────────────────────────
kpoints_scf = """Automatic mesh
0
Gamma
12 12 1
0  0  0
"""

with open(os.path.join(WORK_DIR, "KPOINTS_SCF"), "w") as f:
    f.write(kpoints_scf)

# ── KPOINTS for NSCF: dense disk around K ───────────────────────────
K_point = np.array([1/3, 1/3, 0.0])
dk_radius = 0.08
nk_side = 25

kpts = []
for i in range(nk_side):
    for j in range(nk_side):
        dkx = -dk_radius + 2 * dk_radius * i / (nk_side - 1)
        dky = -dk_radius + 2 * dk_radius * j / (nk_side - 1)
        if np.sqrt(dkx**2 + dky**2) <= dk_radius:
            kx = K_point[0] + dkx
            ky = K_point[1] + dky
            kpts.append(f"  {kx:.10f}  {ky:.10f}  0.0000000000  1.0")

nk_total = len(kpts)
kpoints_nscf = f"""Explicit k-points for spin texture
{nk_total}
Reciprocal
""" + "\n".join(kpts) + "\n"

with open(os.path.join(WORK_DIR, "KPOINTS_NSCF"), "w") as f:
    f.write(kpoints_nscf)

print(f"VASP inputs written to {WORK_DIR}/")
print(f"  SCF k-points: 12x12x1")
print(f"  NSCF k-points: {nk_total} (disk around K)")
print()
print("Run workflow:")
print("  1. cp INCAR_SCF INCAR && cp KPOINTS_SCF KPOINTS && mpirun -np N vasp_ncl")
print("  2. cp INCAR_NSCF INCAR && cp KPOINTS_NSCF KPOINTS && mpirun -np N vasp_ncl")
print("  3. Parse PROCAR for spin texture")

# Save k-point offsets for plotting
dk_offsets = []
for i in range(nk_side):
    for j in range(nk_side):
        dkx = -dk_radius + 2 * dk_radius * i / (nk_side - 1)
        dky = -dk_radius + 2 * dk_radius * j / (nk_side - 1)
        if np.sqrt(dkx**2 + dky**2) <= dk_radius:
            dk_offsets.append([dkx, dky])

np.savetxt(os.path.join(WORK_DIR, "kpoints_dk.dat"),
           np.array(dk_offsets), header="dkx  dky")
```

#### Step B2: Parse PROCAR for Spin Components

```python
#!/usr/bin/env python3
"""
Parse VASP PROCAR file to extract spin texture.

VASP PROCAR with LSORBIT=.TRUE. and LORBIT=11 contains three extra blocks
per k-point per band:
  - Sx (spin-x projection per atom and orbital)
  - Sy (spin-y projection)
  - Sz (spin-z projection)

The file format is:
  k-point N : kx ky kz  weight
  band N : energy  occ
    ion  s  py  pz  px  ... tot    <-- charge
    ...
    ion  s  py  pz  px  ... tot    <-- Sx
    ...
    ion  s  py  pz  px  ... tot    <-- Sy
    ...
    ion  s  py  pz  px  ... tot    <-- Sz
"""
import re
import numpy as np
import os


def parse_procar_spin(procar_path, nions=None):
    """
    Parse PROCAR file from a VASP SOC (noncollinear) calculation.

    Parameters
    ----------
    procar_path : str
        Path to PROCAR file
    nions : int, optional
        Number of ions. Auto-detected if None.

    Returns
    -------
    kpoints : ndarray, shape (nk, 3)
    energies : ndarray, shape (nk, nbnd)
    occupations : ndarray, shape (nk, nbnd)
    sx : ndarray, shape (nk, nbnd) -- total Sx summed over all ions
    sy : ndarray, shape (nk, nbnd) -- total Sy
    sz : ndarray, shape (nk, nbnd) -- total Sz
    """
    with open(procar_path, "r") as f:
        lines = f.readlines()

    # Parse header: "# of k-points:  N   # of bands:  N   # of ions:  N"
    header_line = lines[1].strip()
    m = re.search(
        r"k-points:\s*(\d+).*bands:\s*(\d+).*ions:\s*(\d+)",
        header_line
    )
    if not m:
        raise ValueError(f"Cannot parse PROCAR header: {header_line}")

    nk = int(m.group(1))
    nbnd = int(m.group(2))
    ni = int(m.group(3))
    if nions is None:
        nions = ni

    print(f"PROCAR: {nk} k-points, {nbnd} bands, {nions} ions")

    kpoints = np.zeros((nk, 3))
    energies = np.zeros((nk, nbnd))
    occupations = np.zeros((nk, nbnd))
    sx = np.zeros((nk, nbnd))
    sy = np.zeros((nk, nbnd))
    sz = np.zeros((nk, nbnd))

    ik = -1
    ib = -1
    block_count = 0  # counts ion-data blocks: 0=charge, 1=Sx, 2=Sy, 3=Sz

    i = 0
    while i < len(lines):
        line = lines[i].strip()

        # k-point line
        km = re.match(
            r"k-point\s+(\d+)\s*:\s*([-\d.E+]+)\s+([-\d.E+]+)\s+([-\d.E+]+)",
            line
        )
        if km:
            ik = int(km.group(1)) - 1
            kpoints[ik] = [float(km.group(2)), float(km.group(3)), float(km.group(4))]
            block_count = 0
            i += 1
            continue

        # band line
        bm = re.match(
            r"band\s+(\d+)\s*#\s*energy\s+([-\d.E+]+)\s*#\s*occ\.\s+([-\d.E+]+)",
            line
        )
        if bm:
            ib = int(bm.group(1)) - 1
            energies[ik, ib] = float(bm.group(2))
            occupations[ik, ib] = float(bm.group(3))
            block_count = 0
            i += 1
            continue

        # "tot" line (summary row after ion projections)
        if line.startswith("tot") and ik >= 0 and ib >= 0:
            parts = line.split()
            if len(parts) >= 2:
                tot_val = float(parts[-1])  # last column is the total

                if block_count == 0:
                    pass  # charge block -- skip
                elif block_count == 1:
                    sx[ik, ib] = tot_val
                elif block_count == 2:
                    sy[ik, ib] = tot_val
                elif block_count == 3:
                    sz[ik, ib] = tot_val

                block_count += 1

        i += 1

    return kpoints, energies, occupations, sx, sy, sz


# ── Parse and save ──────────────────────────────────────────────────
WORK_DIR = os.path.abspath("vasp_spintex_2d")
procar_path = os.path.join(WORK_DIR, "PROCAR")

if os.path.exists(procar_path):
    kpoints, energies, occ, sx, sy, sz = parse_procar_spin(procar_path)
    np.savez(os.path.join(WORK_DIR, "spin_texture_vasp.npz"),
             kpoints=kpoints, energies=energies,
             sx=sx, sy=sy, sz=sz)
    print(f"Spin texture data saved to {WORK_DIR}/spin_texture_vasp.npz")
else:
    print(f"PROCAR not found at {procar_path}")
    print("Run VASP NSCF calculation first (see Step B1).")
```

#### Step B3: Plot Spin Texture from VASP PROCAR

```python
#!/usr/bin/env python3
"""
Plot 2D spin texture from parsed VASP PROCAR data.
Produces quiver plot with Sx, Sy arrows colored by Sz.
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib import cm
import os

WORK_DIR = os.path.abspath("vasp_spintex_2d")
data_file = os.path.join(WORK_DIR, "spin_texture_vasp.npz")

if not os.path.exists(data_file):
    print(f"Data file not found: {data_file}")
    print("Run the PROCAR parser first (Step B2).")
    exit(1)

data = np.load(data_file)
kpoints = data["kpoints"]
energies = data["energies"]
sx = data["sx"]
sy = data["sy"]
sz = data["sz"]

# Load k-point offsets
dk_offsets = np.loadtxt(os.path.join(WORK_DIR, "kpoints_dk.dat"))
dkx = dk_offsets[:, 0]
dky = dk_offsets[:, 1]

# Read Fermi energy from OUTCAR or DOSCAR
e_fermi = 0.0
outcar_path = os.path.join(WORK_DIR, "OUTCAR")
if os.path.exists(outcar_path):
    import re
    with open(outcar_path, "r") as f:
        for line in f:
            if "E-fermi" in line:
                m = re.search(r"E-fermi\s*:\s*([-\d.]+)", line)
                if m:
                    e_fermi = float(m.group(1))

# Find VBM band: highest band below E_Fermi
nk, nbnd = energies.shape
vbm_band = 0
for ib in range(nbnd):
    if np.all(energies[:, ib] <= e_fermi + 0.1):
        vbm_band = ib

print(f"E_Fermi = {e_fermi:.4f} eV")
print(f"VBM band index: {vbm_band}")
print(f"VBM energy range: [{energies[:, vbm_band].min():.3f}, {energies[:, vbm_band].max():.3f}] eV")

# Plot
fig, ax = plt.subplots(figsize=(8, 7))

nk_plot = min(len(dkx), nk)
sx_b = sx[:nk_plot, vbm_band]
sy_b = sy[:nk_plot, vbm_band]
sz_b = sz[:nk_plot, vbm_band]

norm = Normalize(vmin=-0.5, vmax=0.5)
q = ax.quiver(dkx[:nk_plot], dky[:nk_plot], sx_b, sy_b,
              sz_b, cmap="coolwarm", norm=norm,
              scale=15, width=0.004, headwidth=3)

cb = plt.colorbar(cm.ScalarMappable(norm=norm, cmap="coolwarm"), ax=ax)
cb.set_label(r"$\langle S_z \rangle$ ($\hbar/2$)", fontsize=13)

ax.plot(0, 0, "k+", markersize=12, markeredgewidth=2)
ax.annotate("K", xy=(0.002, 0.002), fontsize=12, fontweight="bold")

ax.set_xlabel(r"$\Delta k_x$ (crystal)", fontsize=13)
ax.set_ylabel(r"$\Delta k_y$ (crystal)", fontsize=13)
ax.set_title(f"MoS$_2$ Spin Texture (VASP, band {vbm_band})\n"
             r"Arrows: $(S_x, S_y)$, Color: $S_z$",
             fontsize=14, fontweight="bold")
ax.set_aspect("equal")
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(WORK_DIR, "spin_texture_2d_vasp.png"),
            dpi=200, bbox_inches="tight")
print(f"Saved: {WORK_DIR}/spin_texture_2d_vasp.png")
```

---

## Key Parameters

| Parameter | Typical Value | Notes |
|---|---|---|
| `noncolin` / `LNONCOLLINEAR` | `.true.` / `.TRUE.` | Required for spin texture (noncollinear spinors) |
| `lspinorb` / `LSORBIT` | `.true.` / `.TRUE.` | Required for spin-orbit coupling |
| `nosym`, `noinv` (QE) | `.true.` | Disable symmetry in NSCF for arbitrary k-mesh |
| `LORBIT` (VASP) | 11 or 12 | Writes spin-projected PROCAR; 12 includes phase |
| `ICHARG` (VASP) | 11 | Read charge density for NSCF |
| Pseudopotentials (QE) | `*_rel_*` or `*_FR_*` | Must be fully relativistic |
| `ecutwfc` / `ENCUT` | 60 Ry / 450 eV | 10--20% higher than scalar-relativistic |
| `nbnd` (QE) / `NBANDS` (VASP) | 2x non-SOC value | SOC doubles bands (spinor) |
| k-mesh density | 25x25 to 41x41 disk | Denser mesh = smoother texture; 600--1300 k-points typical |
| `dk_radius` | 0.05--0.10 (frac.) | Radius of k-disk around point of interest |
| `assume_isolated` (QE) | `'2D'` | Coulomb cutoff for 2D systems |
| `degauss` | 0.005 Ry | Small smearing for semiconductors |
| `verbosity` (QE) | `'high'` | Required to print spin expectation values |

## Interpreting Results

- **Rashba spin texture**: Spins lie in-plane (Sx, Sy dominate, Sz near zero), winding helically around a high-symmetry point. The two spin-split branches wind in opposite directions (clockwise/counterclockwise). The Rashba parameter alpha_R = 2E_R/k_0 where E_R is the energy offset and k_0 is the momentum offset of the band crossing.

- **Valley spin texture (TMDs)**: At K and K' points, Sz dominates with opposite sign (spin-valley locking). Sz > 0 at K for the upper valence band, Sz < 0 at K'. In-plane components are small and show 3-fold symmetry from trigonal warping.

- **Dresselhaus spin texture**: Similar to Rashba but with different winding symmetry (radial instead of tangential), characteristic of bulk inversion asymmetry in zinc-blende crystals.

- **Spin magnitude**: Each component ranges from -1/2 to +1/2 in units of hbar. Fully polarized states have |S| = 1/2. Partial polarization indicates orbital mixing or hybridization.

- **Constant-energy contour**: For a meaningful spin texture, select bands at a fixed energy (e.g., E_F or VBM). The contour shape reveals the Fermi surface topology and spin-momentum locking pattern.

## Common Issues

| Problem | Solution |
|---|---|
| No spin components in QE output | Set `verbosity = 'high'` in `&CONTROL`. Ensure `noncolin = .true.` and `lspinorb = .true.` are set. |
| Sz is zero everywhere (Rashba system) | This is expected for ideal Rashba with in-plane spins. Plot Sx, Sy arrows instead; color by in-plane angle. |
| PROCAR has no spin blocks | Ensure `LSORBIT = .TRUE.` and `LORBIT = 11` (or 12) in INCAR. The PROCAR must have 4 blocks per band (charge, Sx, Sy, Sz). |
| Spin texture looks noisy or random | Increase k-mesh density. Ensure SCF is well converged. Check that `nosym = .true.` is set in NSCF. |
| Wrong band selected for plotting | Manually inspect eigenvalues at the center k-point. The VBM/CBM band index may differ from expectation due to SOC reordering. |
| SCF does not converge with SOC | Reduce `mixing_beta` to 0.1--0.2. Start from non-SOC converged charge density. Increase `electron_maxstep`. |
| Pseudopotential error with SOC (QE) | Must use fully relativistic PPs (`*.rel-*` or `*_FR_*`). Scalar-relativistic PPs with `lspinorb=.true.` will crash or give wrong results. |
| k-mesh does not cover enough BZ | Increase `dk_radius`. For Rashba systems with large splitting, use 0.10--0.15. For valley textures, 0.05--0.08 is usually sufficient. |
| Spin texture asymmetric when it should be symmetric | Check that the structure is correctly centered. Verify crystal orientation matches the k-point convention. |
| Memory error with large k-mesh | Reduce `nbnd`. Use k-point parallelization (`-nk` flag in QE, `KPAR` in VASP). |
