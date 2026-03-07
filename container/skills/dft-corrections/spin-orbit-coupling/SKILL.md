# Spin-Orbit Coupling (SOC)

## When to Use

Spin-orbit coupling (SOC) arises from the interaction between an electron's spin and its orbital angular momentum. It scales as ~Z^4, so it is most important for **heavy elements**. Apply SOC when:

- **Heavy-element systems** (Z > 50): Bi, Pb, Tl, Au, Pt, W, rare earths, actinides
- **Topological insulators**: Bi2Se3, Bi2Te3, Sb2Te3 -- SOC opens/closes the topological gap
- **Rashba/Dresselhaus splitting**: BiTeI, surface states of Au(111), semiconductor heterostructures
- **Magnetic anisotropy energy (MAE)**: Transition metal multilayers, permanent magnets (Nd2Fe14B)
- **Band splitting in semiconductors**: GaAs, InSb, CdTe -- valence band splitting at Gamma
- **5d transition metals**: Ir, Os, Pt oxides (e.g., Sr2IrO4) -- SOC comparable to crystal field
- **Spin Hall effect, anomalous Hall effect**: Requires SOC for spin-charge coupling
- **J_eff states**: Iridates, osmates where SOC creates j=1/2 ground states

**When NOT to use:**
- Light-element systems (Z < 20): C, N, O, Si -- SOC is negligibly small
- When only total energies/structures are needed for light systems
- When computational cost is prohibitive and SOC effects are minor

## Prerequisites

- Quantum ESPRESSO 7.5 with `pw.x`
- **Fully relativistic pseudopotentials** (`*_rel_*` or `*_FR_*`): These include the full Dirac equation treatment, not just scalar-relativistic corrections
- Where to get them:
  - **PseudoDojo** (recommended): http://www.pseudo-dojo.org/ -- select "FR" (fully relativistic) PPs. Available in ONCVPSP format.
  - **QE pseudopotential page**: https://www.quantum-espresso.org/pseudopotentials/ -- look for files with `rel` in the name (e.g., `Bi.rel-pbe-dn-kjpaw_psl.1.0.0.UPF`)
  - **SSSP library**: https://www.materialscloud.org/discover/sssp -- some entries have FR variants
- Noncollinear magnetism enabled: the SOC calculation is intrinsically noncollinear
- Denser k-point grids (SOC can create small splittings that require fine sampling)

## Detailed Steps

### Example System: Bi2Se3 (Topological Insulator)

Bi2Se3 is a prototypical 3D topological insulator. Without SOC, it is a trivial band insulator. SOC inverts the bands at the Gamma point, creating the topological surface state.

- Space group: R-3m (No. 166), rhombohedral
- 5 atoms per primitive cell (2 Bi + 3 Se)
- Experimental lattice parameters: a = 4.138 A, c = 28.636 A (hexagonal setting)

### Step 1: Structure Setup

```python
#!/usr/bin/env python3
"""Generate Bi2Se3 structure for SOC calculations."""

from pymatgen.core import Structure, Lattice
import numpy as np

# Bi2Se3: Rhombohedral structure (hexagonal setting)
# Space group R-3m (No. 166)
# 3 quintuple layers in the hexagonal cell (15 atoms)
# Primitive rhombohedral cell has 5 atoms

# Hexagonal lattice parameters
a_hex = 4.138   # Angstrom
c_hex = 28.636  # Angstrom

# Use the hexagonal setting with 15 atoms
lattice = Lattice.hexagonal(a_hex, c_hex)

# Fractional coordinates in hexagonal setting (Wyckoff positions)
# Se1: 3a (0,0,0)
# Bi:  6c (0,0,z) with z ~ 0.4008
# Se2: 6c (0,0,z) with z ~ 0.2109
species_hex = ["Se"] * 3 + ["Bi"] * 6 + ["Se"] * 6
coords_hex = [
    # Se1 (3a site)
    [0.0, 0.0, 0.0],
    [1/3, 2/3, 1/3],
    [2/3, 1/3, 2/3],
    # Bi (6c site)
    [0.0, 0.0, 0.4008],
    [1/3, 2/3, 0.7341],
    [2/3, 1/3, 0.0675],
    [0.0, 0.0, 0.5992],
    [1/3, 2/3, 0.9325],
    [2/3, 1/3, 0.2659],
    # Se2 (6c site)
    [0.0, 0.0, 0.2109],
    [1/3, 2/3, 0.5443],
    [2/3, 1/3, 0.8776],
    [0.0, 0.0, 0.7891],
    [1/3, 2/3, 0.1224],
    [2/3, 1/3, 0.4557],
]

structure_hex = Structure(lattice, species_hex, coords_hex)
print(f"Bi2Se3 (hexagonal cell):")
print(f"  a = {a_hex:.3f} A, c = {c_hex:.3f} A")
print(f"  N_atoms = {len(structure_hex)}")

# For the QE calculation, use the primitive rhombohedral cell (5 atoms) for efficiency
# Primitive vectors from hexagonal to rhombohedral:
# a_r = (2a/3, a*sqrt(3)/3, c/3), etc.

# Rhombohedral primitive cell
a_rhomb = np.sqrt(a_hex**2 / 3 + c_hex**2 / 9)
alpha_rhomb = 2 * np.arcsin(3 / (2 * np.sqrt(3 + (c_hex/a_hex)**2)))
alpha_deg = np.degrees(alpha_rhomb)

print(f"\nRhombohedral primitive cell:")
print(f"  a_rhomb = {a_rhomb:.4f} A")
print(f"  alpha = {alpha_deg:.2f} degrees")

# Primitive cell atomic positions
species_prim = ["Bi", "Bi", "Se", "Se", "Se"]
coords_prim = [
    [0.4008, 0.4008, 0.4008],
    [0.5992, 0.5992, 0.5992],
    [0.0000, 0.0000, 0.0000],
    [0.2109, 0.2109, 0.2109],
    [0.7891, 0.7891, 0.7891],
]

print(f"  N_atoms (primitive) = {len(species_prim)}")
```

### Step 2: SCF Without SOC (Scalar-Relativistic Reference)

Save as `bi2se3_no_soc.in`:

```
&CONTROL
    calculation   = 'scf'
    prefix        = 'bi2se3_nosoc'
    outdir        = './tmp_nosoc/'
    pseudo_dir    = './pseudo/'
    verbosity     = 'high'
/

&SYSTEM
    ibrav         = 5
    celldm(1)     = 18.1517
    celldm(4)     = 0.2208
    nat           = 5
    ntyp          = 2
    ecutwfc       = 50.0
    ecutrho       = 400.0
    occupations   = 'smearing'
    smearing      = 'mv'
    degauss       = 0.005
/

&ELECTRONS
    conv_thr      = 1.0d-9
    mixing_beta   = 0.3
/

ATOMIC_SPECIES
  Bi  208.98040  Bi.pbe-dn-kjpaw_psl.1.0.0.UPF
  Se   78.96000  Se.pbe-dn-kjpaw_psl.1.0.0.UPF

ATOMIC_POSITIONS {crystal}
  Bi  0.4008  0.4008  0.4008
  Bi  0.5992  0.5992  0.5992
  Se  0.0000  0.0000  0.0000
  Se  0.2109  0.2109  0.2109
  Se  0.7891  0.7891  0.7891

K_POINTS {automatic}
  8 8 8 0 0 0
```

Notes on `ibrav=5` (rhombohedral):
- `celldm(1)` = a_rhomb in bohr = 9.6007 * 1.8897 = 18.1517 (approximate; adjust to your structure)
- `celldm(4)` = cos(alpha) where alpha is the rhombohedral angle

Use **scalar-relativistic** pseudopotentials here (the standard ones).

```bash
pw.x -npool 2 < bi2se3_no_soc.in > bi2se3_no_soc.out
```

### Step 3: SCF With SOC (Noncollinear + Spin-Orbit)

Save as `bi2se3_soc.in`:

```
&CONTROL
    calculation   = 'scf'
    prefix        = 'bi2se3_soc'
    outdir        = './tmp_soc/'
    pseudo_dir    = './pseudo/'
    verbosity     = 'high'
/

&SYSTEM
    ibrav         = 5
    celldm(1)     = 18.1517
    celldm(4)     = 0.2208
    nat           = 5
    ntyp          = 2
    ecutwfc       = 60.0
    ecutrho       = 480.0
    occupations   = 'smearing'
    smearing      = 'mv'
    degauss       = 0.005
    ! --- Spin-Orbit Coupling ---
    noncolin      = .true.
    lspinorb      = .true.
/

&ELECTRONS
    conv_thr      = 1.0d-9
    mixing_beta   = 0.2
    electron_maxstep = 300
/

ATOMIC_SPECIES
  Bi  208.98040  Bi.rel-pbe-dn-kjpaw_psl.1.0.0.UPF
  Se   78.96000  Se.rel-pbe-dn-kjpaw_psl.1.0.0.UPF

ATOMIC_POSITIONS {crystal}
  Bi  0.4008  0.4008  0.4008
  Bi  0.5992  0.5992  0.5992
  Se  0.0000  0.0000  0.0000
  Se  0.2109  0.2109  0.2109
  Se  0.7891  0.7891  0.7891

K_POINTS {automatic}
  8 8 8 0 0 0
```

**Critical differences from non-SOC:**
1. `noncolin = .true.` -- Enables noncollinear magnetism (spinor wavefunctions, 2-component)
2. `lspinorb = .true.` -- Enables spin-orbit coupling
3. **Fully relativistic pseudopotentials** (`*.rel-*` or `*_FR_*` files) -- MUST use these with SOC
4. **Higher `ecutwfc`** -- Fully relativistic PPs often require 10-20% higher cutoffs
5. **Lower `mixing_beta`** -- SOC calculations can be harder to converge
6. **No `nspin`** -- When `noncolin=.true.`, do NOT set `nspin` (it is automatically 4 internally)

```bash
pw.x -npool 2 < bi2se3_soc.in > bi2se3_soc.out
```

### Step 4: Band Structure Calculation

#### 4a: NSCF on k-path (without SOC)

Save as `bi2se3_bands_nosoc.in`:

```
&CONTROL
    calculation   = 'bands'
    prefix        = 'bi2se3_nosoc'
    outdir        = './tmp_nosoc/'
    pseudo_dir    = './pseudo/'
    verbosity     = 'high'
/

&SYSTEM
    ibrav         = 5
    celldm(1)     = 18.1517
    celldm(4)     = 0.2208
    nat           = 5
    ntyp          = 2
    ecutwfc       = 50.0
    ecutrho       = 400.0
    occupations   = 'smearing'
    smearing      = 'mv'
    degauss       = 0.005
    nbnd          = 40
/

&ELECTRONS
    conv_thr      = 1.0d-9
/

ATOMIC_SPECIES
  Bi  208.98040  Bi.pbe-dn-kjpaw_psl.1.0.0.UPF
  Se   78.96000  Se.pbe-dn-kjpaw_psl.1.0.0.UPF

ATOMIC_POSITIONS {crystal}
  Bi  0.4008  0.4008  0.4008
  Bi  0.5992  0.5992  0.5992
  Se  0.0000  0.0000  0.0000
  Se  0.2109  0.2109  0.2109
  Se  0.7891  0.7891  0.7891

K_POINTS {crystal_b}
5
  0.0000  0.0000  0.0000  30  ! Gamma
  0.5000  0.0000  0.0000  30  ! L
  0.5000  0.5000  0.0000  30  ! F
  0.0000  0.0000  0.0000  30  ! Gamma
  0.5000  0.5000  0.5000   1  ! T
```

#### 4b: NSCF on k-path (with SOC)

Save as `bi2se3_bands_soc.in`:

```
&CONTROL
    calculation   = 'bands'
    prefix        = 'bi2se3_soc'
    outdir        = './tmp_soc/'
    pseudo_dir    = './pseudo/'
    verbosity     = 'high'
/

&SYSTEM
    ibrav         = 5
    celldm(1)     = 18.1517
    celldm(4)     = 0.2208
    nat           = 5
    ntyp          = 2
    ecutwfc       = 60.0
    ecutrho       = 480.0
    occupations   = 'smearing'
    smearing      = 'mv'
    degauss       = 0.005
    noncolin      = .true.
    lspinorb      = .true.
    nbnd          = 80
/

&ELECTRONS
    conv_thr      = 1.0d-9
/

ATOMIC_SPECIES
  Bi  208.98040  Bi.rel-pbe-dn-kjpaw_psl.1.0.0.UPF
  Se   78.96000  Se.rel-pbe-dn-kjpaw_psl.1.0.0.UPF

ATOMIC_POSITIONS {crystal}
  Bi  0.4008  0.4008  0.4008
  Bi  0.5992  0.5992  0.5992
  Se  0.0000  0.0000  0.0000
  Se  0.2109  0.2109  0.2109
  Se  0.7891  0.7891  0.7891

K_POINTS {crystal_b}
5
  0.0000  0.0000  0.0000  30  ! Gamma
  0.5000  0.0000  0.0000  30  ! L
  0.5000  0.5000  0.0000  30  ! Gamma via F
  0.0000  0.0000  0.0000  30  ! Gamma
  0.5000  0.5000  0.5000   1  ! T
```

**Important**: With SOC, the number of bands doubles because each k-state has 2 spinor components. Set `nbnd` accordingly (e.g., if you used 40 without SOC, use 80 with SOC).

#### 4c: Extract bands with bands.x

Save as `bi2se3_bands_pp_nosoc.in`:
```
&BANDS
    prefix  = 'bi2se3_nosoc'
    outdir  = './tmp_nosoc/'
    filband = 'bi2se3_bands_nosoc.dat'
/
```

Save as `bi2se3_bands_pp_soc.in`:
```
&BANDS
    prefix  = 'bi2se3_soc'
    outdir  = './tmp_soc/'
    filband = 'bi2se3_bands_soc.dat'
/
```

```bash
pw.x -npool 2 < bi2se3_bands_nosoc.in > bi2se3_bands_nosoc.out
pw.x -npool 2 < bi2se3_bands_soc.in > bi2se3_bands_soc.out
bands.x < bi2se3_bands_pp_nosoc.in > bi2se3_bands_pp_nosoc.out
bands.x < bi2se3_bands_pp_soc.in > bi2se3_bands_pp_soc.out
```

### Step 5: Python Post-Processing -- Band Structure Comparison

```python
#!/usr/bin/env python3
"""
Compare band structures of Bi2Se3 with and without spin-orbit coupling.
Demonstrates the band inversion that creates the topological insulator phase.
"""

import numpy as np
import matplotlib.pyplot as plt
import re
import os


def read_bands_dat(filename):
    """
    Read bands from QE bands.x output (.dat.gnu format).

    Returns
    -------
    k_points : ndarray, shape (nk,)
        k-point positions along the path
    bands : ndarray, shape (nbnd, nk)
        Band energies in eV
    """
    if not os.path.exists(filename):
        return None, None

    data = []
    current_band = []

    with open(filename, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                if current_band:
                    data.append(current_band)
                    current_band = []
                continue
            parts = line.split()
            if len(parts) == 2:
                current_band.append([float(parts[0]), float(parts[1])])

    if current_band:
        data.append(current_band)

    if not data:
        return None, None

    bands = []
    k_points = None
    for band_data in data:
        band_array = np.array(band_data)
        if k_points is None:
            k_points = band_array[:, 0]
        bands.append(band_array[:, 1])

    return k_points, np.array(bands)


def get_fermi_energy(pw_output_file):
    """Extract Fermi energy from pw.x output."""
    if not os.path.exists(pw_output_file):
        return 0.0
    with open(pw_output_file, 'r') as f:
        content = f.read()
    match = re.search(r'the Fermi energy is\s+([-\d.]+)\s+ev', content, re.IGNORECASE)
    if match:
        return float(match.group(1))
    # Try highest occupied
    match = re.search(r'highest occupied.*?:\s+([-\d.]+)', content)
    if match:
        return float(match.group(1))
    return 0.0


def plot_bands_comparison():
    """
    Plot band structures with and without SOC side by side.
    If actual QE data is not available, use schematic bands for demonstration.
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 7), sharey=True)

    # High-symmetry point labels and positions (will be set by data or manually)
    hsp_labels = ['$\\Gamma$', 'L', 'F', '$\\Gamma$', 'T']

    # Try to read actual data
    k_nosoc, bands_nosoc = read_bands_dat('bi2se3_bands_nosoc.dat.gnu')
    k_soc, bands_soc = read_bands_dat('bi2se3_bands_soc.dat.gnu')
    ef_nosoc = get_fermi_energy('bi2se3_no_soc.out')
    ef_soc = get_fermi_energy('bi2se3_soc.out')

    use_synthetic = (k_nosoc is None or k_soc is None)

    if use_synthetic:
        print("QE band data not found. Generating schematic bands for demonstration.")
        # Generate schematic bands to illustrate SOC effect
        nk = 121
        k = np.linspace(0, 4, nk)

        # High-symmetry point positions
        hsp_positions = [0, 1, 2, 3, 4]

        # --- Without SOC: trivial band gap at Gamma ---
        # Valence bands (Se 4p character)
        vb1_nosoc = -0.5 - 1.5 * np.cos(np.pi * k / 4)**2
        vb2_nosoc = -0.3 - 1.0 * np.cos(np.pi * k / 4)**2
        vb3_nosoc = -0.15 - 0.8 * np.cos(np.pi * (k - 3) / 2)**2
        # Conduction bands (Bi 6p character)
        cb1_nosoc = 0.3 + 1.2 * np.cos(np.pi * k / 4)**2
        cb2_nosoc = 0.5 + 1.0 * np.cos(np.pi * k / 4)**2
        cb3_nosoc = 0.8 + 1.5 * np.cos(np.pi * k / 4)**2

        bands_nosoc_arr = np.array([vb1_nosoc, vb2_nosoc, vb3_nosoc,
                                     cb1_nosoc, cb2_nosoc, cb3_nosoc])

        # --- With SOC: band inversion at Gamma ---
        # SOC inverts Bi-6p and Se-4p at Gamma, opening topological gap
        # Each band splits into 2 (Kramers pairs)
        vb1_soc = -0.6 - 1.5 * np.cos(np.pi * k / 4)**2
        vb1b_soc = -0.55 - 1.5 * np.cos(np.pi * k / 4)**2
        vb2_soc = -0.25 - 1.0 * np.cos(np.pi * k / 4)**2
        vb2b_soc = -0.20 - 1.0 * np.cos(np.pi * k / 4)**2

        # Band inversion: what was CB is now below VB at Gamma
        # Create the inverted gap
        gamma_idx = np.argmin(np.abs(k - 3))  # Gamma at k=3
        anticross = 0.15 * np.exp(-((k - 3)**2) / 0.1)

        vb_top_soc = -0.15 + 0.3 * (1 - np.exp(-((k-3)**2)/0.3))
        vb_topb_soc = -0.10 + 0.3 * (1 - np.exp(-((k-3)**2)/0.3))
        cb_bot_soc = 0.15 - 0.3 * (1 - np.exp(-((k-3)**2)/0.3))
        cb_botb_soc = 0.20 - 0.3 * (1 - np.exp(-((k-3)**2)/0.3))
        cb2_soc = 0.5 + 0.8 * np.cos(np.pi * k / 4)**2
        cb2b_soc = 0.55 + 0.8 * np.cos(np.pi * k / 4)**2

        bands_soc_arr = np.array([vb1_soc, vb1b_soc, vb2_soc, vb2b_soc,
                                   vb_top_soc, vb_topb_soc,
                                   cb_bot_soc, cb_botb_soc,
                                   cb2_soc, cb2b_soc])

        ef_nosoc = 0.0
        ef_soc = 0.0
    else:
        k = k_nosoc
        bands_nosoc_arr = bands_nosoc
        bands_soc_arr = bands_soc
        # Auto-detect HSP positions (evenly spaced for crystal_b)
        nk_per_seg = len(k) // (len(hsp_labels) - 1)
        hsp_positions = [i * nk_per_seg for i in range(len(hsp_labels))]
        hsp_positions[-1] = len(k) - 1
        hsp_positions = [k[i] for i in hsp_positions]

    # ---- Plot (a): Without SOC ----
    ax = axes[0]
    for i, band in enumerate(bands_nosoc_arr):
        color = 'blue' if band[len(band)//2] < ef_nosoc else 'red'
        ax.plot(k, band - ef_nosoc, color=color, linewidth=1.2)
    ax.axhline(y=0, color='green', linestyle='--', linewidth=1, alpha=0.7, label='$E_F$')
    ax.set_ylabel('Energy (eV)', fontsize=13)
    ax.set_title('(a) Without SOC\n(Scalar-Relativistic)', fontsize=14, fontweight='bold')
    ax.set_ylim(-3, 3)

    if use_synthetic:
        for pos, label in zip([0, 1, 2, 3, 4], hsp_labels):
            ax.axvline(x=pos, color='gray', linestyle='-', linewidth=0.5, alpha=0.5)
        ax.set_xticks([0, 1, 2, 3, 4])
        ax.set_xticklabels(hsp_labels, fontsize=12)
        # Annotate the trivial gap
        ax.annotate('Trivial gap\n~0.3 eV', xy=(3, 0.15), fontsize=10, color='black',
                     ha='center', fontstyle='italic',
                     bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.7))
        # Label orbital character
        ax.text(0.5, -1.5, 'Se 4p', fontsize=10, color='blue', ha='center', fontstyle='italic')
        ax.text(0.5, 1.5, 'Bi 6p', fontsize=10, color='red', ha='center', fontstyle='italic')
    else:
        if hsp_positions:
            for pos in hsp_positions:
                ax.axvline(x=pos, color='gray', linestyle='-', linewidth=0.5, alpha=0.5)
            ax.set_xticks(hsp_positions)
            ax.set_xticklabels(hsp_labels, fontsize=12)

    ax.legend(fontsize=10, loc='upper right')

    # ---- Plot (b): With SOC ----
    ax = axes[1]
    for i, band in enumerate(bands_soc_arr):
        mid_idx = len(band) // 2
        color = 'blue' if band[mid_idx] < ef_soc else 'red'
        ax.plot(k, band - ef_soc, color=color, linewidth=1.2)
    ax.axhline(y=0, color='green', linestyle='--', linewidth=1, alpha=0.7, label='$E_F$')
    ax.set_title('(b) With SOC\n(Fully Relativistic)', fontsize=14, fontweight='bold')
    ax.set_ylim(-3, 3)

    if use_synthetic:
        for pos, label in zip([0, 1, 2, 3, 4], hsp_labels):
            ax.axvline(x=pos, color='gray', linestyle='-', linewidth=0.5, alpha=0.5)
        ax.set_xticks([0, 1, 2, 3, 4])
        ax.set_xticklabels(hsp_labels, fontsize=12)
        # Annotate the band inversion
        ax.annotate('Band\ninversion!', xy=(3, 0), fontsize=11, color='purple',
                     ha='center', fontweight='bold',
                     bbox=dict(boxstyle='round,pad=0.3', facecolor='lightyellow', alpha=0.8))
        ax.annotate('', xy=(2.7, -0.2), xytext=(2.7, 0.2),
                     arrowprops=dict(arrowstyle='<->', color='purple', lw=2))
        ax.text(2.5, 0, '~0.3 eV\nSOC gap', fontsize=9, color='purple', ha='right')
        # Label spin-split bands
        ax.text(1.5, -0.8, 'Kramers\npairs', fontsize=9, color='blue',
                ha='center', fontstyle='italic')
    else:
        if hsp_positions:
            for pos in hsp_positions:
                ax.axvline(x=pos, color='gray', linestyle='-', linewidth=0.5, alpha=0.5)
            ax.set_xticks(hsp_positions)
            ax.set_xticklabels(hsp_labels, fontsize=12)

    ax.legend(fontsize=10, loc='upper right')

    plt.suptitle('Bi$_2$Se$_3$: Effect of Spin-Orbit Coupling on Band Structure',
                 fontsize=15, fontweight='bold')
    plt.tight_layout()
    plt.savefig('bi2se3_bands_soc_comparison.png', dpi=300, bbox_inches='tight')
    plt.savefig('bi2se3_bands_soc_comparison.pdf', bbox_inches='tight')
    print("Saved: bi2se3_bands_soc_comparison.png, bi2se3_bands_soc_comparison.pdf")
    plt.show()


if __name__ == '__main__':
    plot_bands_comparison()
```

### Step 6: Additional Example -- GaAs Valence Band Splitting

GaAs is a simpler system where SOC splits the valence band at Gamma into heavy-hole (HH), light-hole (LH), and split-off (SO) bands. The experimental spin-orbit splitting is Delta_SO = 0.34 eV.

Save as `gaas_soc.in`:

```
&CONTROL
    calculation   = 'scf'
    prefix        = 'gaas_soc'
    outdir        = './tmp_gaas_soc/'
    pseudo_dir    = './pseudo/'
    verbosity     = 'high'
/

&SYSTEM
    ibrav         = 2
    celldm(1)     = 10.6829
    nat           = 2
    ntyp          = 2
    ecutwfc       = 60.0
    ecutrho       = 480.0
    occupations   = 'smearing'
    smearing      = 'mv'
    degauss       = 0.005
    noncolin      = .true.
    lspinorb      = .true.
/

&ELECTRONS
    conv_thr      = 1.0d-10
    mixing_beta   = 0.3
/

ATOMIC_SPECIES
  Ga  69.723  Ga.rel-pbe-dn-kjpaw_psl.1.0.0.UPF
  As  74.922  As.rel-pbe-dn-kjpaw_psl.1.0.0.UPF

ATOMIC_POSITIONS {crystal}
  Ga  0.00  0.00  0.00
  As  0.25  0.25  0.25

K_POINTS {automatic}
  8 8 8 0 0 0
```

Note: `celldm(1) = a (bohr)` for GaAs: a = 5.653 A = 10.6829 bohr. `ibrav=2` is FCC.

### Step 7: Extracting SOC Splittings

```python
#!/usr/bin/env python3
"""
Extract and analyze spin-orbit splittings from QE band structure output.
Works with both actual QE data and demonstration data.
"""

import numpy as np
import matplotlib.pyplot as plt
import os
import re


def find_soc_splitting_at_gamma(bands_soc_file, bands_nosoc_file, ef_soc, ef_nosoc):
    """
    Compare eigenvalues at Gamma point with and without SOC to identify splittings.

    For GaAs-like systems:
    - Without SOC: 3-fold degenerate VBM at Gamma (p-like)
    - With SOC: splits into 4-fold j=3/2 (HH+LH) and 2-fold j=1/2 (SO)
    - Delta_SO = E(j=3/2) - E(j=1/2) at Gamma
    """
    # This function parses the .dat.gnu file format
    # For a specific Gamma-point analysis, you may prefer reading eigenvalues
    # directly from the pw.x output

    results = {}

    # Read eigenvalues at Gamma from pw.x output
    for label, fname, ef in [('nosoc', 'bi2se3_no_soc.out', ef_nosoc),
                              ('soc', 'bi2se3_soc.out', ef_soc)]:
        if not os.path.exists(fname):
            continue
        with open(fname, 'r') as f:
            content = f.read()

        # Find eigenvalues near Gamma
        eig_blocks = re.findall(
            r'k\s*=\s*0\.0000\s+0\.0000\s+0\.0000.*?\n(.*?)(?=\n\s*k\s*=|\n\s*the Fermi|\n\s*highest)',
            content, re.DOTALL
        )
        if eig_blocks:
            eigs = []
            for num in re.findall(r'([-\d.]+)', eig_blocks[0]):
                try:
                    eigs.append(float(num))
                except ValueError:
                    pass
            results[label] = np.array(eigs) - ef

    return results


def analyze_soc_effects():
    """Print analysis of SOC effects."""

    print("=" * 60)
    print("Spin-Orbit Coupling Analysis")
    print("=" * 60)

    # Expected results for common materials
    expected_results = {
        'Bi2Se3': {
            'gap_nosoc': '~0.3 eV (trivial)',
            'gap_soc': '~0.3 eV (inverted, topological)',
            'key_effect': 'Band inversion at Gamma; creates Z2 topological insulator',
            'delta_soc': '~1.25 eV (Bi 6p SOC splitting)',
        },
        'GaAs': {
            'gap_nosoc': '~0.5 eV (PBE underestimates; expt 1.52 eV)',
            'gap_soc': '~0.5 eV (minor change)',
            'key_effect': 'Valence band split-off at Gamma',
            'delta_soc': '~0.34 eV (expt 0.34 eV)',
        },
        'Bi (elemental)': {
            'gap_nosoc': 'Semimetal',
            'gap_soc': 'Semimetal with modified Fermi surface',
            'key_effect': 'Large band shifts, modified topology',
            'delta_soc': '~1.5 eV',
        },
        'Pb (elemental)': {
            'gap_nosoc': 'Metal',
            'gap_soc': 'Metal with modified band structure',
            'key_effect': 'Significant band structure changes',
            'delta_soc': '~0.9 eV',
        },
    }

    for material, info in expected_results.items():
        print(f"\n{material}:")
        for key, val in info.items():
            print(f"  {key}: {val}")

    print("\n" + "=" * 60)
    print("General SOC Scaling (approximate Delta_SOC):")
    print("=" * 60)

    elements = ['C', 'Si', 'Ge', 'Ga', 'As', 'Se', 'Te', 'Bi', 'Pb', 'Au', 'Pt']
    Z_values = [6, 14, 32, 31, 33, 34, 52, 83, 82, 79, 78]
    soc_values = [0.006, 0.044, 0.29, 0.11, 0.38, 0.42, 0.91, 1.25, 0.91, 0.52, 0.55]

    print(f"{'Element':<8} {'Z':>4} {'Delta_SOC (eV)':>15}")
    print("-" * 30)
    for el, z, soc in zip(elements, Z_values, soc_values):
        print(f"{el:<8} {z:>4} {soc:>15.3f}")

    # Plot SOC vs Z
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.scatter(Z_values, soc_values, s=100, c='blue', edgecolors='black', zorder=5)
    for el, z, soc in zip(elements, Z_values, soc_values):
        ax.annotate(el, (z, soc), textcoords="offset points",
                     xytext=(5, 5), fontsize=10)

    # Fit Z^4 trend
    z_fit = np.linspace(5, 85, 100)
    # Rough Z^2 scaling (actually Z^4 for hydrogen-like but screened in real atoms)
    scale = np.mean([s / (z/10)**2 for s, z in zip(soc_values, Z_values)])
    soc_fit = scale * (z_fit / 10)**2

    ax.plot(z_fit, soc_fit, 'r--', alpha=0.5, label='~$Z^2$ trend (screened)')
    ax.set_xlabel('Atomic number Z', fontsize=12)
    ax.set_ylabel('SOC splitting (eV)', fontsize=12)
    ax.set_title('Spin-Orbit Coupling Strength vs Atomic Number', fontsize=13, fontweight='bold')
    ax.legend(fontsize=11)
    ax.set_xlim(0, 90)
    ax.set_ylim(0, 1.5)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('soc_vs_Z.png', dpi=300, bbox_inches='tight')
    print("\nSaved: soc_vs_Z.png")
    plt.show()


if __name__ == '__main__':
    analyze_soc_effects()
```

## Key Parameters

| Parameter | Value | Notes |
|-----------|-------|-------|
| `noncolin` | `.true.` | Required for SOC; enables 2-component spinor wavefunctions |
| `lspinorb` | `.true.` | Activates spin-orbit coupling |
| `nspin` | **Do NOT set** | Automatically handled when `noncolin=.true.` |
| Pseudopotentials | `*.rel-*` or `*_FR_*` | MUST be fully relativistic |
| `ecutwfc` | 10-20% higher than scalar-relativistic | FR PPs need higher cutoffs; always test convergence |
| `ecutrho` | 8-10x `ecutwfc` | Standard ratio; may need increase for FR PPs |
| `nbnd` | 2x non-SOC value | Bands double due to spinor structure |
| `mixing_beta` | 0.1-0.3 | Lower than usual for convergence stability |
| `electron_maxstep` | 200-500 | May need more iterations |
| K-points | Denser than non-SOC | Fine splittings require fine k-mesh |
| `starting_magnetization` | Replaced by `angle1`, `angle2` | For noncollinear magnetic systems (see below) |

### Magnetization Direction in Noncollinear Calculations

For magnetic systems with SOC, the magnetization direction matters (magnetic anisotropy). Use:

```
&SYSTEM
    ...
    noncolin      = .true.
    lspinorb      = .true.
    ! Magnetization direction for species i:
    ! angle1(i) = polar angle theta (degrees, from z-axis)
    ! angle2(i) = azimuthal angle phi (degrees, in xy-plane)
    angle1(1)     = 0.0     ! along z
    angle2(1)     = 0.0
    ! For MAE calculations, compare E(theta=0) vs E(theta=90), etc.
/
```

### Magnetic Anisotropy Energy (MAE) Calculation

To compute MAE, run two SOC calculations with magnetization along different axes and take the energy difference:

```
MAE = E(hard axis) - E(easy axis)
```

This requires extremely tight convergence (`conv_thr = 1.0d-10` or better) and dense k-grids, because MAE is typically ~0.01-1 meV/atom.

## Interpreting Results

### What SOC Does to Bands

| Effect | Description | Example |
|--------|-------------|---------|
| Band splitting | Degenerate bands split into Kramers pairs | GaAs VBM splits into HH/LH + SO |
| Band inversion | Orbital character of bands swaps | Bi2Se3 becomes topological |
| Rashba splitting | Bands split linearly in k near high-symmetry points | BiTeI, Au(111) surface |
| Dresselhaus splitting | Cubic spin splitting in non-centrosymmetric crystals | GaAs, InSb |
| Gap modification | Band gap can increase or decrease | Bi2Se3: trivial->inverted |
| Magnetic anisotropy | Energy depends on magnetization direction | Fe/Pt multilayers |

### Identifying Topological Character

After computing bands with SOC, check for band inversion:
1. Compare orbital character of bands above/below the gap at Gamma
2. Without SOC: conduction band has cation character (Bi 6p), valence band has anion character (Se 4p)
3. With SOC: these are inverted at Gamma --> topological insulator
4. Compute the Z2 invariant (requires additional post-processing with tools like Z2Pack or WannierTools)

### Cost Comparison

| Property | Without SOC | With SOC | Ratio |
|----------|-------------|----------|-------|
| Basis set size | N | 2N (spinor) | 2x |
| Memory | M | ~4M | 4x |
| CPU time | T | ~8-16T | 8-16x |
| Symmetry operations | Full space group | Reduced (no spin-flip) | Fewer k-points in IBZ |

## Common Issues

1. **Using scalar-relativistic PPs with `lspinorb=.true.`**: This will produce WRONG results or crash. Always use fully relativistic PPs (`*.rel-*` or `*_FR_*`).

2. **Setting `nspin` with `noncolin=.true.`**: Do NOT set `nspin` when using noncollinear mode. QE handles this automatically. Setting `nspin=2` with `noncolin=.true.` will cause an error.

3. **Insufficient `nbnd`**: With SOC, bands double. If you set the same `nbnd` as without SOC, you will miss half the bands. Always double `nbnd` for SOC calculations.

4. **Convergence difficulties**: SOC calculations are harder to converge. Remedies:
   - Reduce `mixing_beta` to 0.1-0.2
   - Increase `electron_maxstep` to 300-500
   - Use `mixing_mode = 'local-TF'`
   - Start from a converged non-SOC charge density (copy `charge-density.dat` and `data-file-schema.xml` from the non-SOC calculation directory, though this requires care with consistency)

5. **Symmetry reduction**: SOC reduces symmetry operations (time-reversal symmetry is modified). This means more k-points in the irreducible BZ and thus more computational cost. For non-magnetic systems, time-reversal symmetry (Kramers theorem) is preserved, which helps.

6. **PP compatibility**: Not all PPs from the same library have FR variants. Check:
   - PseudoDojo: Most elements have FR versions
   - SSSP: Limited FR coverage
   - PSLibrary: Good coverage of `*.rel-*` files
   - Do NOT mix FR and SR PPs in the same calculation

7. **Much larger output files**: SOC calculations produce about 2x larger wavefunctions and charge densities. Ensure sufficient disk space.

8. **Wrong Fermi level in metals**: For metals with SOC, the Fermi surface can change significantly. Ensure k-point convergence is checked independently for SOC calculations.

9. **Post-processing tools**: Some QE post-processing tools (e.g., older versions of `pp.x`, `projwfc.x`) may not fully support noncollinear mode. Check the QE documentation for compatibility.

10. **Comparing with experiment**: PBE+SOC still underestimates band gaps (the gap issue is from PBE, not SOC). For quantitative comparison, consider PBE+SOC+scissors correction, or use hybrid functionals (HSE) with SOC (very expensive), or GW+SOC.
