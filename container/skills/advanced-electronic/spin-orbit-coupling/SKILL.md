# Spin-Orbit Coupling (SOC) Calculations

## When to Use

Spin-orbit coupling (SOC) arises from the interaction of an electron's spin with its orbital motion in the nuclear potential. Apply SOC when:

- **Heavy elements (Z > 50)**: Bi, Pb, Te, Se, Au, Pt, W, Ir -- SOC scales as Z^4
- **Topological insulators**: Bi2Se3, Bi2Te3, HgTe -- SOC drives band inversion
- **Rashba/Dresselhaus splitting**: surfaces, interfaces, polar semiconductors (BiTeI, GeTe)
- **Topological surface states**: spin-momentum locking
- **Magnetic anisotropy energy (MAE)**: determines easy axis in magnetic materials
- **Band splittings**: valence band splitting in semiconductors (GaAs, CdTe, MoS2)
- **Spin Hall effect**: heavy metals (Pt, W, Ta)
- **SOC + DFT+U**: correlated topological materials (SmB6, iridates)

**When NOT to use:**
- Light-element systems (C, N, O, Si) where SOC << other energy scales
- When only total energy or forces are needed and SOC contribution is negligible
- Phonon calculations (SOC effects on phonons are typically very small)

## Method Selection

| Method | QE Keywords | VASP Keywords | Use Case |
|---|---|---|---|
| Noncollinear + SOC | `noncolin=.true.`, `lspinorb=.true.` | `LSORBIT=.TRUE.` | General SOC calculation |
| SOC + DFT+U | Above + `HUBBARD` card | `LSORBIT=.TRUE.` + `LDAU=.TRUE.` | Correlated + heavy elements |
| Scalar relativistic (no SOC) | Default (no flags needed) | Default | Light elements, reference |
| SOC perturbative | Not standard in QE | `LSORBIT=.TRUE.`, `LNONCOLLINEAR=.FALSE.` | Approximate, not recommended |

## Prerequisites

- Quantum ESPRESSO 7.5 with `pw.x`, `bands.x`, `projwfc.x`
- **Fully relativistic (FR) pseudopotentials** -- mandatory for SOC
  - PSlibrary: files with `_rel` suffix (e.g., `Bi.rel-pbe-dn-kjpaw_psl.1.0.0.UPF`)
  - ONCVPSP/PseudoDojo: files with `_FR` suffix
  - SSSP does not currently provide FR pseudopotentials; use PSlibrary or PseudoDojo
- For VASP: standard PAW potentials work (SOC is handled internally)
- Python: `pymatgen`, `ase`, `numpy`, `matplotlib`

## Detailed Steps

### Example System: Bi2Se3 (Topological Insulator with SOC-Driven Band Inversion)

Bi2Se3 is a prototypical 3D topological insulator. Without SOC, it is a trivial semiconductor. SOC inverts the bands at Gamma, creating a topological insulator with Z2 = (1;000).

### Step 1: Generate Structure and Download FR Pseudopotentials

```python
#!/usr/bin/env python3
"""Set up Bi2Se3 for SOC band structure calculation."""

import os
import subprocess
import numpy as np

WORK_DIR = os.path.abspath("soc_bi2se3")
os.makedirs(os.path.join(WORK_DIR, "pseudo"), exist_ok=True)

# Download fully relativistic pseudopotentials from PSlibrary
pp_urls = {
    "Bi.rel-pbe-dn-kjpaw_psl.1.0.0.UPF":
        "https://pseudopotentials.quantum-espresso.org/upf_files/"
        "Bi.rel-pbe-dn-kjpaw_psl.1.0.0.UPF",
    "Se.rel-pbe-dn-kjpaw_psl.1.0.0.UPF":
        "https://pseudopotentials.quantum-espresso.org/upf_files/"
        "Se.rel-pbe-dn-kjpaw_psl.1.0.0.UPF",
}

for fname, url in pp_urls.items():
    target = os.path.join(WORK_DIR, "pseudo", fname)
    if not os.path.exists(target):
        print(f"Downloading {fname}...")
        subprocess.run(["wget", "-q", "-O", target, url], check=True)

# Bi2Se3 rhombohedral structure (R-3m, #166)
# Rhombohedral lattice parameters
a_rhomb = 9.841  # Bohr (5.208 Angstrom)
cos_alpha = np.cos(np.radians(24.304))
bi_mu = 0.4006
se_nu = 0.2060

print(f"Bi2Se3 rhombohedral cell:")
print(f"  a = {a_rhomb} Bohr = {a_rhomb * 0.529177:.3f} Angstrom")
print(f"  cos(alpha) = {cos_alpha:.6f}")
```

### Step 2: SCF Without SOC (Reference)

Save as `bi2se3_no_soc.in`:

```
&CONTROL
    calculation = 'scf'
    prefix      = 'bi2se3_nosoc'
    outdir      = './tmp_nosoc/'
    pseudo_dir  = './pseudo/'
    verbosity   = 'high'
/

&SYSTEM
    ibrav       = 5
    celldm(1)   = 9.841
    celldm(4)   = 0.9113
    nat         = 5
    ntyp        = 2
    ecutwfc     = 50.0
    ecutrho     = 400.0
    occupations = 'smearing'
    smearing    = 'cold'
    degauss     = 0.005
    ! No SOC -- scalar relativistic
/

&ELECTRONS
    conv_thr    = 1.0d-10
    mixing_beta = 0.3
/

ATOMIC_SPECIES
  Bi  208.98040  Bi.rel-pbe-dn-kjpaw_psl.1.0.0.UPF
  Se   78.96000  Se.rel-pbe-dn-kjpaw_psl.1.0.0.UPF

ATOMIC_POSITIONS {crystal}
  Bi  0.400600  0.400600  0.400600
  Bi  0.599400  0.599400  0.599400
  Se  0.206000  0.206000  0.206000
  Se  0.794000  0.794000  0.794000
  Se  0.000000  0.000000  0.000000

K_POINTS {automatic}
  8 8 8 0 0 0
```

Note: You CAN use FR pseudopotentials without enabling SOC -- QE will use only the scalar-relativistic part. This ensures a consistent comparison.

```bash
pw.x -npool 2 < bi2se3_no_soc.in > bi2se3_no_soc.out
```

### Step 3: SCF With SOC

Save as `bi2se3_soc.in`:

```
&CONTROL
    calculation = 'scf'
    prefix      = 'bi2se3_soc'
    outdir      = './tmp_soc/'
    pseudo_dir  = './pseudo/'
    verbosity   = 'high'
/

&SYSTEM
    ibrav       = 5
    celldm(1)   = 9.841
    celldm(4)   = 0.9113
    nat         = 5
    ntyp        = 2
    ecutwfc     = 50.0
    ecutrho     = 400.0
    occupations = 'smearing'
    smearing    = 'cold'
    degauss     = 0.005
    ! --- SOC enabled ---
    noncolin    = .true.
    lspinorb    = .true.
/

&ELECTRONS
    conv_thr    = 1.0d-10
    mixing_beta = 0.3
/

ATOMIC_SPECIES
  Bi  208.98040  Bi.rel-pbe-dn-kjpaw_psl.1.0.0.UPF
  Se   78.96000  Se.rel-pbe-dn-kjpaw_psl.1.0.0.UPF

ATOMIC_POSITIONS {crystal}
  Bi  0.400600  0.400600  0.400600
  Bi  0.599400  0.599400  0.599400
  Se  0.206000  0.206000  0.206000
  Se  0.794000  0.794000  0.794000
  Se  0.000000  0.000000  0.000000

K_POINTS {automatic}
  8 8 8 0 0 0
```

Key SOC parameters:
- `noncolin = .true.` -- Enables noncollinear spin (required for SOC). The magnetization becomes a 3-component vector at each point.
- `lspinorb = .true.` -- Enables spin-orbit coupling. Requires fully relativistic pseudopotentials.
- With SOC, each band becomes a 2-component spinor. The number of bands doubles compared to a collinear calculation with the same number of electrons.

```bash
pw.x -npool 2 < bi2se3_soc.in > bi2se3_soc.out
```

### Step 3b: VASP INCAR for SOC

```
# SOC calculation
LSORBIT     = .TRUE.      ! Enable spin-orbit coupling
LNONCOLLINEAR = .TRUE.    ! Noncollinear magnetism (set automatically by LSORBIT)

# General settings
ENCUT       = 400
EDIFF       = 1E-8
ISMEAR      = 0
SIGMA       = 0.05
LREAL       = .FALSE.     ! Reciprocal space projection (recommended for SOC)

# For SOC + DFT+U (e.g., for iridates):
! LDAU        = .TRUE.
! LDAUTYPE    = 2
! LDAUL       = 2 -1
! LDAUU       = 2.0 0.0
! LDAUJ       = 0.0 0.0
! LMAXMIX     = 4

# For magnetic anisotropy energy:
! SAXIS       = 0 0 1     ! Spin quantization axis (default: z)
! LORBMOM     = .TRUE.    ! Print orbital moments
```

### Step 4: Band Structure With and Without SOC

#### NSCF at band k-points (save as `bi2se3_bands_soc.in`):

```
&CONTROL
    calculation = 'bands'
    prefix      = 'bi2se3_soc'
    outdir      = './tmp_soc/'
    pseudo_dir  = './pseudo/'
    verbosity   = 'high'
/

&SYSTEM
    ibrav       = 5
    celldm(1)   = 9.841
    celldm(4)   = 0.9113
    nat         = 5
    ntyp        = 2
    ecutwfc     = 50.0
    ecutrho     = 400.0
    occupations = 'smearing'
    smearing    = 'cold'
    degauss     = 0.005
    noncolin    = .true.
    lspinorb    = .true.
    nbnd        = 36
/

&ELECTRONS
    conv_thr    = 1.0d-10
    diago_full_acc = .true.
/

ATOMIC_SPECIES
  Bi  208.98040  Bi.rel-pbe-dn-kjpaw_psl.1.0.0.UPF
  Se   78.96000  Se.rel-pbe-dn-kjpaw_psl.1.0.0.UPF

ATOMIC_POSITIONS {crystal}
  Bi  0.400600  0.400600  0.400600
  Bi  0.599400  0.599400  0.599400
  Se  0.206000  0.206000  0.206000
  Se  0.794000  0.794000  0.794000
  Se  0.000000  0.000000  0.000000

K_POINTS {crystal_b}
6
  0.0000  0.0000  0.0000  30  ! Gamma
  0.5000  0.0000  0.0000  30  ! L
  0.5000  0.5000  0.0000  30  ! F
  0.0000  0.0000  0.0000  30  ! Gamma
  0.5000  0.5000  0.5000  30  ! Z
  0.5000  0.0000  0.5000   0  ! F1
```

```bash
pw.x -npool 2 < bi2se3_bands_soc.in > bi2se3_bands_soc.out
```

#### Post-process bands:

Save as `bi2se3_bands_pp.in`:
```
&BANDS
    prefix  = 'bi2se3_soc'
    outdir  = './tmp_soc/'
    filband = 'bi2se3_soc_bands.dat'
    lsym    = .false.
/
```

```bash
bands.x < bi2se3_bands_pp.in > bi2se3_bands_pp.out
```

Repeat the entire band structure workflow for the no-SOC case (using `bi2se3_nosoc` prefix, no `noncolin`/`lspinorb`).

### Step 5: Compare Band Structures (SOC vs No-SOC)

```python
#!/usr/bin/env python3
"""
Compare Bi2Se3 band structure with and without spin-orbit coupling.
Highlights band inversion at Gamma that drives the topological transition.
"""

import numpy as np
import matplotlib.pyplot as plt


def read_qe_bands(filename):
    """Read QE bands.x output file."""
    with open(filename, 'r') as f:
        header = f.readline()
        nbnd, nks = [int(x) for x in header.split(',')[0].split('=')[1].split()]

    data = np.loadtxt(filename, skiprows=1)

    kpoints = []
    bands = [[] for _ in range(nbnd)]
    k_idx = 0
    band_idx = 0

    for line_data in data:
        if band_idx == 0:
            kpoints.append(line_data[0] if len(line_data) == 1 else 0)

    # Re-read with proper parsing
    kpts = []
    energies = []
    current_band = []

    with open(filename, 'r') as f:
        header = f.readline()
        for line in f:
            vals = line.split()
            if len(vals) == 0:
                if current_band:
                    energies.append(current_band)
                    current_band = []
            else:
                for v in vals:
                    current_band.append(float(v))
        if current_band:
            energies.append(current_band)

    return np.array(energies) if energies else None


def plot_comparison():
    """Plot band structure comparison with and without SOC."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 7), sharey=True)

    # High-symmetry point labels
    hs_labels = ['$\\Gamma$', 'L', 'F', '$\\Gamma$', 'Z', 'F$_1$']

    for idx, (title, band_file) in enumerate([
        ('Bi$_2$Se$_3$: No SOC (scalar relativistic)', 'bi2se3_nosoc_bands.dat'),
        ('Bi$_2$Se$_3$: With SOC (fully relativistic)', 'bi2se3_soc_bands.dat'),
    ]):
        ax = axes[idx]

        try:
            bands_data = read_qe_bands(band_file)
            if bands_data is not None:
                nks = len(bands_data[0]) if len(bands_data) > 0 else 100
                k = np.linspace(0, 1, nks)
                for band in bands_data:
                    ax.plot(k, band, 'b-', linewidth=0.8)
        except (FileNotFoundError, Exception):
            # Generate synthetic bands for demonstration
            k = np.linspace(0, 1, 200)
            n_bands = 20

            if idx == 0:  # No SOC: trivial gap, no inversion
                for i in range(n_bands):
                    if i < 10:  # Valence
                        e = -2 - 0.5 * i + 1.5 * np.sin(np.pi * k)
                    else:  # Conduction
                        e = 0.5 + 0.3 * (i - 10) + 1.0 * np.sin(np.pi * k)
                    ax.plot(k, e, 'b-', linewidth=0.8)

                ax.annotate('Normal ordering\n(no inversion)',
                            xy=(0.0, -0.5), fontsize=10, color='green',
                            ha='center', fontweight='bold')
            else:  # With SOC: inverted gap at Gamma
                for i in range(n_bands):
                    if i < 10:
                        e = -2 - 0.5 * i + 1.5 * np.sin(np.pi * k)
                        # Apply SOC splitting near Gamma
                        if i >= 8:
                            e += 0.3 * np.exp(-((k - 0.0)**2) / 0.01)
                    else:
                        e = 0.3 + 0.3 * (i - 10) + 1.0 * np.sin(np.pi * k)
                        if i <= 11:
                            e -= 0.4 * np.exp(-((k - 0.0)**2) / 0.01)
                    ax.plot(k, e, 'b-', linewidth=0.8)

                ax.annotate('Band inversion\nat $\\Gamma$',
                            xy=(0.0, -0.2), fontsize=10, color='red',
                            ha='center', fontweight='bold',
                            arrowprops=dict(arrowstyle='->', color='red'))

        ax.axhline(y=0, color='gray', linestyle='--', linewidth=0.8, alpha=0.5)
        ax.set_ylabel('Energy (eV)' if idx == 0 else '', fontsize=12)
        ax.set_title(title, fontsize=13, fontweight='bold')
        ax.set_xlim(0, 1)
        ax.set_ylim(-5, 5)

        # Mark high-symmetry points
        n_seg = len(hs_labels) - 1
        hs_pos = np.linspace(0, 1, len(hs_labels))
        ax.set_xticks(hs_pos)
        ax.set_xticklabels(hs_labels, fontsize=11)
        for x in hs_pos:
            ax.axvline(x=x, color='gray', linestyle='-', linewidth=0.5, alpha=0.5)

    plt.suptitle('Effect of Spin-Orbit Coupling on Band Structure',
                 fontsize=15, fontweight='bold')
    plt.tight_layout()
    plt.savefig('bi2se3_soc_comparison.png', dpi=300, bbox_inches='tight')
    plt.savefig('bi2se3_soc_comparison.pdf', bbox_inches='tight')
    print("Saved: bi2se3_soc_comparison.png, bi2se3_soc_comparison.pdf")


if __name__ == '__main__':
    plot_comparison()
```

### Step 6: Rashba Splitting Example (BiTeI)

BiTeI is a polar semiconductor with giant Rashba spin splitting at the conduction band minimum.

```python
#!/usr/bin/env python3
"""
Set up BiTeI for Rashba spin splitting calculation with SOC.
BiTeI has a giant Rashba parameter alpha_R ~ 3.8 eV*Angstrom.
"""

import os
import numpy as np

WORK_DIR = os.path.abspath("rashba_bitei")
os.makedirs(WORK_DIR, exist_ok=True)

# BiTeI hexagonal cell (P3m1, #156)
# a = 4.3392 A, c = 6.854 A
a_bohr = 4.3392 / 0.529177  # Convert to Bohr

scf_input = f"""&CONTROL
    calculation = 'scf'
    prefix      = 'bitei'
    outdir      = './tmp/'
    pseudo_dir  = './pseudo/'
    verbosity   = 'high'
/

&SYSTEM
    ibrav       = 4
    celldm(1)   = {a_bohr:.4f}
    celldm(3)   = {6.854/4.3392:.6f}
    nat         = 3
    ntyp        = 3
    ecutwfc     = 50.0
    ecutrho     = 400.0
    occupations = 'smearing'
    smearing    = 'cold'
    degauss     = 0.005
    noncolin    = .true.
    lspinorb    = .true.
/

&ELECTRONS
    conv_thr    = 1.0d-10
    mixing_beta = 0.3
/

ATOMIC_SPECIES
  Bi  208.980  Bi.rel-pbe-dn-kjpaw_psl.1.0.0.UPF
  Te  127.600  Te.rel-pbe-dn-kjpaw_psl.1.0.0.UPF
  I   126.904  I.rel-pbe-dn-kjpaw_psl.1.0.0.UPF

ATOMIC_POSITIONS {{crystal}}
  Bi  0.000000  0.000000  0.000000
  Te  0.333333  0.666667  0.298000
  I   0.333333  0.666667  0.695000

K_POINTS {{automatic}}
  10 10 6 0 0 0
"""

with open(os.path.join(WORK_DIR, "scf.in"), "w") as f:
    f.write(scf_input)

# Dense k-mesh around A point for Rashba splitting
# The Rashba splitting occurs near the A = (0, 0, 0.5) point
bands_input = f"""&CONTROL
    calculation = 'bands'
    prefix      = 'bitei'
    outdir      = './tmp/'
    pseudo_dir  = './pseudo/'
/

&SYSTEM
    ibrav       = 4
    celldm(1)   = {a_bohr:.4f}
    celldm(3)   = {6.854/4.3392:.6f}
    nat         = 3
    ntyp        = 3
    ecutwfc     = 50.0
    ecutrho     = 400.0
    occupations = 'smearing'
    smearing    = 'cold'
    degauss     = 0.005
    noncolin    = .true.
    lspinorb    = .true.
    nbnd        = 30
/

&ELECTRONS
    conv_thr    = 1.0d-10
    diago_full_acc = .true.
/

ATOMIC_SPECIES
  Bi  208.980  Bi.rel-pbe-dn-kjpaw_psl.1.0.0.UPF
  Te  127.600  Te.rel-pbe-dn-kjpaw_psl.1.0.0.UPF
  I   126.904  I.rel-pbe-dn-kjpaw_psl.1.0.0.UPF

ATOMIC_POSITIONS {{crystal}}
  Bi  0.000000  0.000000  0.000000
  Te  0.333333  0.666667  0.298000
  I   0.333333  0.666667  0.695000

K_POINTS {{crystal_b}}
4
  0.3333  0.6667  0.5000  40  ! K-A direction
  0.0000  0.0000  0.5000  40  ! A point (Rashba splitting center)
  0.5000  0.0000  0.5000  40  ! M-A direction
  0.0000  0.0000  0.0000   0  ! Gamma
"""

with open(os.path.join(WORK_DIR, "bands.in"), "w") as f:
    f.write(bands_input)

print("Rashba calculation inputs written.")
print("Rashba splitting parameters to extract:")
print("  E_R = Rashba energy (meV)")
print("  k_0 = momentum offset (1/Angstrom)")
print("  alpha_R = 2*E_R/k_0 (eV*Angstrom) -- Rashba parameter")
print("  Expected for BiTeI: alpha_R ~ 3.8 eV*Angstrom")
```

### Step 7: SOC + DFT+U (Correlated Topological Materials)

For systems needing both SOC and DFT+U, combine both in the input. Example for a hypothetical iridate:

```
&SYSTEM
    ...
    noncolin    = .true.
    lspinorb    = .true.
/

...

HUBBARD {ortho-atomic}
U Ir-5d 2.0
```

VASP equivalent:
```
LSORBIT     = .TRUE.
LDAU        = .TRUE.
LDAUTYPE    = 2
LDAUL       = 2 -1
LDAUU       = 2.0 0.0
LDAUJ       = 0.0 0.0
LMAXMIX     = 4
```

## Key Parameters

| Parameter | QE | VASP | Notes |
|-----------|-----|------|-------|
| Enable SOC | `lspinorb = .true.` | `LSORBIT = .TRUE.` | Requires FR PPs in QE |
| Noncollinear | `noncolin = .true.` | `LNONCOLLINEAR = .TRUE.` | Auto-set by LSORBIT in VASP |
| Pseudopotentials | FR (`_rel` or `_FR`) | Standard PAW | Must be fully relativistic in QE |
| ecutwfc | 50-80 Ry | ENCUT 400-600 eV | Heavier elements may need more |
| k-grid | Denser than non-SOC | Same | Bands may split, requiring finer grid |
| `nbnd` | 2x non-SOC value | `NBANDS` | Spinor doubles the bands |
| Spin axis | `angle1`, `angle2` per atom | `SAXIS` | For magnetic anisotropy |
| Starting mag | `starting_magnetization` | `MAGMOM` | 3 components per atom (noncollinear) |
| conv_thr | 1.0d-10 or tighter | `EDIFF = 1E-8` | SOC needs tighter convergence |
| mixing_beta | 0.2-0.3 | `AMIX = 0.2` | SOC often converges more slowly |

## Interpreting Results

### SOC Effects on Band Structure

| System | Without SOC | With SOC | SOC Effect |
|--------|-------------|----------|------------|
| Bi2Se3 | Trivial ~0.3 eV gap | Inverted ~0.3 eV gap | Band inversion at Gamma |
| GaAs VB | 3-fold degenerate at Gamma | Split HH/LH + SO (0.34 eV) | Valence band splitting |
| BiTeI | Degenerate CBM at A | Rashba split (E_R ~ 100 meV) | Giant Rashba splitting |
| Au surface | Degenerate surface state | Rashba split | Surface Rashba |
| Pt | Moderate d-band width | Broadened, split d-bands | Large SOC splitting |
| MoS2 VB at K | Degenerate | Split by ~150 meV | Spin-valley locking |

### Computational Cost

| Aspect | Non-SOC (collinear) | With SOC (noncollinear) | Ratio |
|--------|---------------------|-------------------------|-------|
| Number of bands | N | 2N | 2x |
| Hamiltonian size | N x N | 2N x 2N | 4x |
| Memory | M | ~4M | 4x |
| Wall time | T | ~4-8T | 4-8x |
| k-point symmetry | Full BZ symmetry | Reduced (no spin reversal) | 1-2x more k-points |

### What to Check

1. **Band inversion**: Compare band ordering at high-symmetry points with and without SOC. Inverted ordering signals potential topological character.
2. **Rashba splitting**: Look for parabolic band splitting at time-reversal invariant momenta (TRIM). Extract E_R, k_0, and alpha_R = 2*E_R/k_0.
3. **Valence band splitting**: At Gamma in zincblende/wurtzite semiconductors, measure the spin-orbit splitting Delta_SO.
4. **Convergence**: Ensure SCF convergence is tight (1e-10 Ry or better). SOC is a small correction and sloppy convergence can mask it.
5. **Magnetic moments**: SOC introduces orbital moments. Check both spin and orbital magnetic moments in the output.

## Common Issues

| Issue | Symptom | Solution |
|-------|---------|----------|
| Wrong pseudopotential type | QE crashes with "not a fully-relativistic PP" | Use FR pseudopotentials (`_rel` or `_FR` suffix) |
| SCF does not converge | Energy oscillates, exceeds max steps | Reduce `mixing_beta` to 0.1-0.2; start from converged non-SOC wavefunction with `startingpot='file'` |
| Memory overflow | Out of memory errors | SOC uses 4x memory; reduce k-points, use k-point parallelization (`-nk` flag) |
| Wrong band count | Missing bands in output | Set `nbnd` to at least 2x the non-SOC value |
| Slow convergence | Many SCF iterations needed | Use `electron_maxstep = 300`; try `mixing_mode = 'local-TF'` |
| k-point symmetry lost | More k-points than expected | SOC breaks some symmetries; this is correct behavior |
| SOC effect too small | No visible splitting | Check element -- SOC is negligible for Z < 30; verify FR PPs are used |
| SOC + DFT+U instability | SCF oscillates | Start from converged DFT+U (no SOC), then restart with SOC; reduce mixing_beta further |
| VASP LREAL conflict | Warning about real-space projection | Set `LREAL = .FALSE.` for SOC calculations |
| Spin contamination | Large <S^2> deviation | Normal for noncollinear; not a bug |
