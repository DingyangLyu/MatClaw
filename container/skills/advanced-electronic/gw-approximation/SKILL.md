# GW Quasiparticle Calculations

## When to Use

GW calculations provide accurate quasiparticle band gaps and band structures by going beyond DFT, treating electron-electron interactions via the screened Coulomb interaction W and the Green's function G:

- **Accurate band gaps**: DFT systematically underestimates gaps; GW corrects this
- **Comparison with photoemission**: GW quasiparticle energies directly compare to ARPES/XPS
- **Band alignment**: Accurate valence band maximum / conduction band minimum positions for heterostructures
- **Semiconductors and insulators**: Si, GaAs, ZnO, MgO, TiO2, wide-gap materials
- **Defect levels**: Accurate charge transition levels in defective crystals
- **2D materials**: Gaps in MoS2, h-BN, graphene nanoribbons

**When NOT to use:**
- Metals (no gap to correct)
- Strongly correlated systems (GW alone is insufficient; need DMFT or similar)
- Very large systems (GW scales as N^4 or worse; limited to ~100 atoms)
- Geometry optimization (GW does not provide forces efficiently)
- When DFT+U or hybrid functionals provide sufficient accuracy

## Method Selection

| Method | Description | Accuracy | Cost | Implementation |
|---|---|---|---|---|
| G0W0 (single-shot) | One-shot GW from DFT starting point | Good (depends on starting functional) | High (10-100x DFT) | Yambo, VASP, BerkeleyGW |
| GW0 | Self-consistent G, fixed W0 | Better than G0W0 | Higher | VASP (`ALGO=GW0`) |
| scGW (fully self-consistent) | Self-consistent G and W | Best (starting-point independent) | Very high | VASP (`ALGO=scGW`) |
| GW + SOC | GW with spin-orbit coupling | Accurate for heavy elements | Very high | VASP, Yambo |
| SternheimerGW | Avoids empty states | Good | Moderate | QE plugin |
| G0W0@PBE | G0W0 starting from PBE | Underestimates gaps slightly | High | Any code |
| G0W0@HSE | G0W0 starting from HSE06 | Often most accurate single-shot | High | VASP |

## Prerequisites

- **For QE workflow**: Yambo (`pip install yambo` or compile from source) or SternheimerGW plugin
- **For VASP workflow**: VASP compiled with GW support (standard in VASP 5.4+/6.x)
- Converged DFT ground state as starting point
- Sufficient empty/conduction bands (typically 8-20x occupied bands)
- Large memory and computation time (GW is expensive)
- Python: `numpy`, `matplotlib`, `pymatgen` for post-processing

## Detailed Steps

### Overview of the GW Workflow

```
1. DFT-SCF (ground state)
    |
2. DFT-NSCF (many empty bands, dense k-grid)
    |
3. GW calculation:
   a. Compute dielectric matrix / screened interaction W
   b. Compute self-energy Sigma = i*G*W
   c. Solve quasiparticle equation for corrected energies
    |
4. Extract quasiparticle band structure
```

### Example System: Silicon (Benchmark)

Silicon is the standard GW benchmark. PBE gives a band gap of ~0.6 eV (experiment: 1.17 eV). G0W0@PBE gives ~1.1-1.2 eV.

### Step 1: DFT Ground State (QE)

```python
#!/usr/bin/env python3
"""Set up Silicon for GW calculation."""

import os
import subprocess

WORK_DIR = os.path.abspath("gw_silicon")
os.makedirs(os.path.join(WORK_DIR, "pseudo"), exist_ok=True)

# Download pseudopotential
pp_url = ("https://pseudopotentials.quantum-espresso.org/upf_files/"
          "Si.pbe-n-rrkjus_psl.1.0.0.UPF")
pp_file = os.path.join(WORK_DIR, "pseudo", "Si.pbe-n-rrkjus_psl.1.0.0.UPF")
if not os.path.exists(pp_file):
    subprocess.run(["wget", "-q", "-O", pp_file, pp_url], check=True)

# Silicon diamond structure
# a = 5.43 Angstrom = 10.2631 Bohr
a_bohr = 10.2631

scf_input = f"""&CONTROL
    calculation = 'scf'
    prefix      = 'si'
    outdir      = './tmp/'
    pseudo_dir  = './pseudo/'
    verbosity   = 'high'
/

&SYSTEM
    ibrav       = 2
    celldm(1)   = {a_bohr}
    nat         = 2
    ntyp        = 1
    ecutwfc     = 60.0
    ecutrho     = 480.0
    occupations = 'smearing'
    smearing    = 'cold'
    degauss     = 0.005
/

&ELECTRONS
    conv_thr    = 1.0d-10
/

ATOMIC_SPECIES
  Si  28.0855  Si.pbe-n-rrkjus_psl.1.0.0.UPF

ATOMIC_POSITIONS {{crystal}}
  Si  0.00  0.00  0.00
  Si  0.25  0.25  0.25

K_POINTS {{automatic}}
  8 8 8 0 0 0
"""

with open(os.path.join(WORK_DIR, "scf.in"), "w") as f:
    f.write(scf_input)
print("SCF input written.")
```

```bash
cd gw_silicon
pw.x -npool 2 < scf.in > scf.out
grep "highest occupied, lowest unoccupied" scf.out
# Expected PBE gap: ~0.6 eV
```

### Step 2: NSCF with Many Empty Bands

GW requires many empty (conduction) bands to construct the dielectric matrix and self-energy. A typical rule: use 8-20x the number of occupied bands.

Save as `nscf.in`:

```
&CONTROL
    calculation = 'nscf'
    prefix      = 'si'
    outdir      = './tmp/'
    pseudo_dir  = './pseudo/'
    verbosity   = 'high'
/

&SYSTEM
    ibrav       = 2
    celldm(1)   = 10.2631
    nat         = 2
    ntyp        = 1
    ecutwfc     = 60.0
    ecutrho     = 480.0
    occupations = 'smearing'
    smearing    = 'cold'
    degauss     = 0.005
    nbnd        = 100
    ! Many empty bands for GW convergence
    ! Si has 4 occupied bands (2 atoms x 4 valence electrons / 2 spin)
    ! 100 bands = 25x occupied -- ample for convergence test
/

&ELECTRONS
    conv_thr    = 1.0d-10
    diago_full_acc = .true.
/

ATOMIC_SPECIES
  Si  28.0855  Si.pbe-n-rrkjus_psl.1.0.0.UPF

ATOMIC_POSITIONS {crystal}
  Si  0.00  0.00  0.00
  Si  0.25  0.25  0.25

K_POINTS {automatic}
  6 6 6 0 0 0
```

Note: The k-grid for GW is often smaller than for SCF due to the high computational cost. Start with 4x4x4 or 6x6x6 and test convergence.

```bash
pw.x -npool 2 < nscf.in > nscf.out
```

### Step 3a: G0W0 via Yambo (QE Interface)

Yambo is the primary GW code interfacing with QE. The workflow is:

```bash
# 1. Convert QE output to Yambo database
cd gw_silicon
p2y  # or p2y -F tmp/si.save
# This creates the SAVE/ directory with Yambo-format data

# 2. Initialize Yambo
yambo
# Creates the r_setup file with system info

# 3. Set up G0W0 calculation
yambo -d -g n -p p -V qp
# -d: Dyson equation
# -g n: GW approximation
# -p p: Plasmon-pole approximation for W (fast) or -p c for contour deformation (accurate)
# -V qp: Quasiparticle output
```

This generates a `yambo.in` input file. Key parameters to edit:

```
# yambo.in for G0W0 calculation on Silicon

GW_self_cons = "g"      # G0W0 (one-shot)

% BndsRnXp                # Bands for polarization (screening)
   1 | 100 |              # 1 to 100
%

NGsBlkXp = 5 Ry          # G-vector cutoff for dielectric matrix (convergence critical)

% GbndRnge                # Bands for GW self-energy
   1 | 100 |
%

% QPkrange                # k-points and bands for QP corrections
  1 | 19 | 1 | 10 |      # k1 to k19, bands 1-10
%

% GDamping                # Broadening for Green's function
  0.10 | eV |
%

PPAPntXp = 27.2114 eV    # Plasmon-pole energy (1 Ha = 27.2 eV for PP model)
```

```bash
# Run Yambo G0W0
yambo -F yambo.in -J gw_output
# This takes significant time (minutes to hours depending on system size)

# Extract QP corrections
ypp -e s -J gw_output
# Produces o-gw_output.qp file with quasiparticle corrections
```

#### Parse Yambo QP Output

```python
#!/usr/bin/env python3
"""Parse Yambo G0W0 quasiparticle output for Silicon."""

import numpy as np
import os


def parse_yambo_qp(filename):
    """
    Parse Yambo quasiparticle output (o-*.qp file).
    Returns dict with k-index, band, E_dft, E_qp, Z factor.
    """
    data = {'k': [], 'band': [], 'E_dft': [], 'E_gw': [],
            'E_correction': [], 'Z': []}

    with open(filename, 'r') as f:
        for line in f:
            if line.startswith('#') or line.strip() == '':
                continue
            parts = line.split()
            if len(parts) >= 5:
                data['k'].append(int(parts[0]))
                data['band'].append(int(parts[1]))
                data['E_dft'].append(float(parts[2]))
                data['E_correction'].append(float(parts[3]))
                data['E_gw'].append(float(parts[4]))
                if len(parts) >= 6:
                    data['Z'].append(float(parts[5]))

    return data


# Parse and display results
qp_file = "gw_output/o-gw_output.qp"
if os.path.exists(qp_file):
    qp = parse_yambo_qp(qp_file)

    print("=" * 70)
    print("G0W0 Quasiparticle Corrections for Silicon")
    print("=" * 70)
    print(f"{'K':>4} {'Band':>5} {'E_PBE (eV)':>12} {'E_GW (eV)':>12} "
          f"{'Corr (eV)':>10} {'Z':>6}")
    print("-" * 70)

    for i in range(len(qp['k'])):
        z_str = f"{qp['Z'][i]:.3f}" if i < len(qp['Z']) else "  ---"
        print(f"{qp['k'][i]:>4} {qp['band'][i]:>5} "
              f"{qp['E_dft'][i]:>12.4f} {qp['E_gw'][i]:>12.4f} "
              f"{qp['E_correction'][i]:>10.4f} {z_str:>6}")

    # Find VBM and CBM
    n_occ = 4  # Silicon: 4 occupied bands
    vbm_dft = max(e for b, e in zip(qp['band'], qp['E_dft']) if b <= n_occ)
    cbm_dft = min(e for b, e in zip(qp['band'], qp['E_dft']) if b > n_occ)
    vbm_gw = max(e for b, e in zip(qp['band'], qp['E_gw']) if b <= n_occ)
    cbm_gw = min(e for b, e in zip(qp['band'], qp['E_gw']) if b > n_occ)

    print(f"\nPBE band gap:  {cbm_dft - vbm_dft:.3f} eV")
    print(f"G0W0 band gap: {cbm_gw - vbm_gw:.3f} eV")
    print(f"Experiment:    1.17 eV")
else:
    print("Yambo QP file not found. Expected results for Si:")
    print("  PBE gap:   ~0.60 eV")
    print("  G0W0 gap:  ~1.12 eV")
    print("  Expt gap:  1.17 eV")
```

### Step 3b: GW via VASP

VASP has built-in GW support. The workflow is:

#### Step 3b-i: DFT Ground State (VASP)

Standard VASP SCF with accurate settings:

```
# INCAR for DFT ground state
ENCUT   = 400
EDIFF   = 1E-8
ISMEAR  = 0
SIGMA   = 0.05
LWAVE   = .TRUE.    ! Save wavefunctions (needed for GW)
LCHARG  = .TRUE.
PREC    = Accurate
ALGO    = Normal
```

#### Step 3b-ii: G0W0 Calculation (VASP)

```
# INCAR for G0W0
ALGO    = GW0       ! GW0 approximation (or EVGW0 for eigenvalue self-consistent GW0)
LWAVE   = .FALSE.
LCHARG  = .FALSE.
PREC    = Accurate
ENCUT   = 400
ISMEAR  = 0
SIGMA   = 0.05
NBANDS  = 100       ! Total bands (8-20x occupied; convergence critical!)
NOMEGA  = 100       ! Frequency grid points for W (convergence critical!)
ENCUTGW = 200       ! G-vector cutoff for response function (eV)
PRECFOCK = Normal   ! Precision for exact exchange (if hybrid starting point)

! For different GW flavors:
! ALGO = GW0      --> GW0 (self-consistent G, fixed W from DFT)
! ALGO = scGW     --> Fully self-consistent GW (most expensive, most accurate)
! ALGO = EVGW0    --> Eigenvalue-only self-consistent GW0
! ALGO = GW0R     --> GW0 in real space (faster for large cells)
! ALGO = G0W0     --> Single-shot G0W0 (least expensive)
```

#### Step 3b-iii: G0W0 Starting from HSE06 (VASP, Often Most Accurate)

```
# Step 1: HSE06 ground state
# INCAR_hse
ALGO    = Damped
LHFCALC = .TRUE.
AEXX    = 0.25
HFSCREEN = 0.2
PREC    = Accurate
ENCUT   = 400
EDIFF   = 1E-8
ISMEAR  = 0
SIGMA   = 0.05
LWAVE   = .TRUE.
NBANDS  = 100

# Step 2: G0W0 on top of HSE06
# INCAR_gw (copy WAVECAR from HSE calculation)
ALGO    = GW0
PREC    = Accurate
ENCUT   = 400
NBANDS  = 100
NOMEGA  = 100
ENCUTGW = 200
ISMEAR  = 0
SIGMA   = 0.05
LWAVE   = .FALSE.
```

### Step 3c: SternheimerGW (QE Plugin, Avoids Empty States)

SternheimerGW avoids the need for explicit empty states by solving Sternheimer equations. This is more memory-efficient but requires the SternheimerGW plugin.

```python
#!/usr/bin/env python3
"""
SternheimerGW workflow for Silicon.
Conceptual workflow -- requires SternheimerGW plugin compiled with QE.
"""

import os

WORK_DIR = os.path.abspath("gw_silicon_sgw")
os.makedirs(WORK_DIR, exist_ok=True)

# SternheimerGW input template
sgw_input = """&gw_input
  title = 'Silicon G0W0'

  ! k-points for QP corrections
  kpt_grid = 4, 4, 4
  num_band = 8        ! bands to correct

  ! Frequency integration
  num_freq = 35       ! frequency grid
  freq_type = 2       ! 2 = imaginary axis + analytic continuation

  ! Screening
  max_freq_coul = 200 ! max frequency for W (eV)

  ! Self-energy
  num_gwband = 4      ! number of bands in GW self-energy sum

  ! Convergence
  eta = 0.01          ! broadening (Ry)
  threshold = 1.0d-5  ! convergence threshold

  ! Output
  output_type = 2     ! 2 = QP energies and spectral function
/
"""

with open(os.path.join(WORK_DIR, "sgw.in"), "w") as f:
    f.write(sgw_input)

print("SternheimerGW input written.")
print("Run with: sgw.x < sgw.in > sgw.out")
print("Note: Requires SternheimerGW plugin compiled with QE.")
```

### Step 4: Convergence Testing

GW results depend critically on several convergence parameters. Always test convergence.

```python
#!/usr/bin/env python3
"""
GW convergence testing script.
Tests convergence with respect to:
  1. Number of bands (NBANDS / nbnd)
  2. G-vector cutoff (ENCUTGW / NGsBlkXp)
  3. k-point grid
  4. Frequency grid (NOMEGA)
"""

import numpy as np
import matplotlib.pyplot as plt


def plot_convergence():
    """Plot GW convergence with respect to key parameters."""

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # --- (a) Convergence with number of bands ---
    ax = axes[0, 0]
    nbands = [20, 40, 60, 80, 100, 150, 200, 300]
    # Typical Si G0W0 gap values (eV) -- from literature
    gaps_nbands = [0.95, 1.05, 1.10, 1.12, 1.13, 1.13, 1.14, 1.14]

    ax.plot(nbands, gaps_nbands, 'bo-', linewidth=2, markersize=6)
    ax.axhline(y=1.17, color='red', linestyle='--', linewidth=1.5,
               label='Expt. (1.17 eV)')
    ax.set_xlabel('Number of bands', fontsize=11)
    ax.set_ylabel('G0W0 gap (eV)', fontsize=11)
    ax.set_title('(a) Band convergence', fontsize=12, fontweight='bold')
    ax.legend(fontsize=10)
    ax.set_ylim(0.9, 1.25)

    # --- (b) Convergence with ENCUTGW / G-vector cutoff ---
    ax = axes[0, 1]
    encutgw = [50, 100, 150, 200, 250, 300, 400]
    gaps_encutgw = [1.28, 1.18, 1.14, 1.13, 1.12, 1.12, 1.12]

    ax.plot(encutgw, gaps_encutgw, 'rs-', linewidth=2, markersize=6)
    ax.axhline(y=1.17, color='red', linestyle='--', linewidth=1.5,
               label='Expt. (1.17 eV)')
    ax.set_xlabel('ENCUTGW (eV)', fontsize=11)
    ax.set_ylabel('G0W0 gap (eV)', fontsize=11)
    ax.set_title('(b) Response function cutoff', fontsize=12, fontweight='bold')
    ax.legend(fontsize=10)
    ax.set_ylim(0.9, 1.35)

    # --- (c) Convergence with k-grid ---
    ax = axes[1, 0]
    kgrids = ['2x2x2', '3x3x3', '4x4x4', '6x6x6', '8x8x8']
    gaps_kgrid = [1.35, 1.22, 1.16, 1.13, 1.12]

    ax.plot(range(len(kgrids)), gaps_kgrid, 'g^-', linewidth=2, markersize=8)
    ax.axhline(y=1.17, color='red', linestyle='--', linewidth=1.5,
               label='Expt. (1.17 eV)')
    ax.set_xticks(range(len(kgrids)))
    ax.set_xticklabels(kgrids, fontsize=10)
    ax.set_xlabel('k-grid', fontsize=11)
    ax.set_ylabel('G0W0 gap (eV)', fontsize=11)
    ax.set_title('(c) k-point convergence', fontsize=12, fontweight='bold')
    ax.legend(fontsize=10)
    ax.set_ylim(0.9, 1.45)

    # --- (d) Convergence with frequency points ---
    ax = axes[1, 1]
    nomega = [20, 40, 60, 80, 100, 150, 200]
    gaps_nomega = [1.08, 1.11, 1.12, 1.13, 1.13, 1.13, 1.13]

    ax.plot(nomega, gaps_nomega, 'md-', linewidth=2, markersize=6)
    ax.axhline(y=1.17, color='red', linestyle='--', linewidth=1.5,
               label='Expt. (1.17 eV)')
    ax.set_xlabel('NOMEGA (frequency points)', fontsize=11)
    ax.set_ylabel('G0W0 gap (eV)', fontsize=11)
    ax.set_title('(d) Frequency grid convergence', fontsize=12, fontweight='bold')
    ax.legend(fontsize=10)
    ax.set_ylim(0.9, 1.25)

    plt.suptitle('Silicon G0W0: Convergence Tests',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig('si_gw_convergence.png', dpi=300, bbox_inches='tight')
    plt.savefig('si_gw_convergence.pdf', bbox_inches='tight')
    print("Saved: si_gw_convergence.png, si_gw_convergence.pdf")
    plt.show()


plot_convergence()
```

### Step 5: Compare DFT vs GW vs Experiment

```python
#!/usr/bin/env python3
"""
Compare band gaps: PBE, G0W0@PBE, G0W0@HSE, scGW vs experiment.
Data from published benchmarks (Shishkin & Kresse, PRB 2007; van Setten et al., JCTC 2015).
"""

import numpy as np
import matplotlib.pyplot as plt

# Benchmark data: band gaps in eV
materials = ['Si', 'Ge', 'GaAs', 'GaN', 'ZnO', 'MgO', 'LiF', 'NaCl', 'C', 'SiC']

gaps = {
    'Experiment': [1.17, 0.74, 1.52, 3.50, 3.44, 7.83, 14.20, 8.50, 5.48, 2.42],
    'PBE':        [0.60, 0.00, 0.54, 1.70, 0.73, 4.76, 9.20,  5.10, 4.12, 1.35],
    'G0W0@PBE':   [1.12, 0.65, 1.30, 3.00, 2.80, 7.40, 13.50, 8.00, 5.60, 2.30],
    'G0W0@HSE':   [1.20, 0.78, 1.50, 3.40, 3.30, 7.70, 14.00, 8.40, 5.55, 2.40],
    'scGW':       [1.30, 0.90, 1.65, 3.55, 3.60, 8.10, 14.50, 8.80, 5.80, 2.55],
}

fig, ax = plt.subplots(figsize=(14, 7))

x = np.arange(len(materials))
width = 0.18

colors = {'Experiment': 'black', 'PBE': '#66c2a5', 'G0W0@PBE': '#fc8d62',
          'G0W0@HSE': '#8da0cb', 'scGW': '#e78ac3'}

for i, (method, gap_values) in enumerate(gaps.items()):
    offset = (i - 2) * width
    style = {'edgecolor': 'black', 'linewidth': 0.8}
    if method == 'Experiment':
        style['hatch'] = '///'
        style['alpha'] = 0.8
    ax.bar(x + offset, gap_values, width, label=method,
           color=colors[method], **style)

ax.set_xlabel('Material', fontsize=13)
ax.set_ylabel('Band gap (eV)', fontsize=13)
ax.set_title('Band Gap Comparison: PBE vs GW vs Experiment', fontsize=14,
             fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(materials, fontsize=11)
ax.legend(fontsize=10, ncol=3, loc='upper left')
ax.grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig('gw_benchmark_gaps.png', dpi=300, bbox_inches='tight')
plt.savefig('gw_benchmark_gaps.pdf', bbox_inches='tight')
print("Saved: gw_benchmark_gaps.png, gw_benchmark_gaps.pdf")
plt.show()
```

## Key Parameters

| Parameter | QE/Yambo | VASP | Notes |
|-----------|----------|------|-------|
| GW flavor | Yambo: `-g n` (G0W0) | `ALGO=GW0`, `scGW`, `EVGW0` | G0W0 is most common |
| Number of bands | `BndsRnXp`, `GbndRnge` | `NBANDS` | 8-20x occupied; convergence-critical |
| Response cutoff | `NGsBlkXp` (Ry) | `ENCUTGW` (eV) | 100-300 eV typical; test convergence |
| Frequency grid | Yambo: `num_freq` | `NOMEGA` | 50-100 usually sufficient |
| k-grid | Same as NSCF | KPOINTS file | 4x4x4 to 8x8x8; expensive to converge |
| Plasmon-pole | Yambo: `-p p` | `LSPECTRAL=.FALSE.` (implicit PP) | Fast but less accurate than contour deformation |
| Contour deformation | Yambo: `-p c` | `LSPECTRAL=.TRUE.` | More accurate, more expensive |
| QP bands | `QPkrange` | `NBANDSGW` | Which bands get QP corrections |
| Starting functional | PBE (default) | PBE or HSE06 | HSE starting point often gives better G0W0 |

## Interpreting Results

### Expected GW Corrections

| Material | PBE Gap (eV) | G0W0@PBE (eV) | Experiment (eV) | Correction (eV) |
|----------|-------------|----------------|-----------------|-----------------|
| Si       | 0.60        | 1.12           | 1.17            | +0.52           |
| GaAs     | 0.54        | 1.30           | 1.52            | +0.76           |
| ZnO      | 0.73        | 2.80           | 3.44            | +2.07           |
| MgO      | 4.76        | 7.40           | 7.83            | +2.64           |
| LiF      | 9.20        | 13.50          | 14.20           | +4.30           |
| MoS2     | 1.65        | 2.40           | 2.50 (optical)  | +0.75           |
| h-BN     | 4.50        | 6.10           | 6.08            | +1.60           |

### Key Observations

1. **PBE systematically underestimates gaps** -- this is the well-known "band gap problem."
2. **G0W0@PBE improves gaps significantly** but still underestimates by ~0.1-0.5 eV for wide-gap materials.
3. **G0W0@HSE is often the most accurate single-shot method** because HSE provides a better starting point.
4. **scGW can slightly overestimate gaps** but is starting-point independent.
5. **Quasiparticle renormalization factor Z** (typically 0.7-0.9) measures the fraction of spectral weight in the QP peak. Z << 1 indicates strong correlation.

### Computational Cost Scaling

| Quantity | Scaling | Typical for Si (2 atoms) | Typical for 50 atoms |
|----------|---------|--------------------------|----------------------|
| DFT-SCF | O(N^3) | Seconds | Minutes |
| NSCF (many bands) | O(N^3 * N_bands) | Minutes | Hours |
| G0W0 (dielectric) | O(N^4) | Hours | Days-weeks |
| scGW | O(N^4 * N_iter) | Days | Weeks-months |

## Common Issues

| Issue | Symptom | Solution |
|-------|---------|----------|
| Unconverged gap | Gap changes with NBANDS | Increase NBANDS systematically; plot gap vs NBANDS |
| Slow ENCUTGW convergence | Gap changes with cutoff | Use 200-300 eV; basis set extrapolation may help |
| k-point convergence | Gap oscillates with k-grid | Use at least 4x4x4; test 6x6x6, 8x8x8 |
| Plasmon-pole inaccuracy | Gap differs from contour deformation | Use contour deformation (`-p c` in Yambo, `LSPECTRAL=.TRUE.` in VASP) for accuracy |
| Starting-point dependence | G0W0@PBE and G0W0@HSE differ | Report both; G0W0@HSE often more reliable; or use scGW |
| Memory overflow | Out of memory for large NBANDS | Reduce NBANDS (check convergence); use distributed memory (MPI) |
| Negative QP weight | Z < 0 or Z > 1 | Check convergence; may indicate strong correlation (GW may not be appropriate) |
| Yambo setup errors | `p2y` fails or database incomplete | Ensure QE NSCF completed with `LWAVE=.TRUE.` equivalent; check paths |
| VASP GW fails to start | Missing WAVECAR | DFT step must save WAVECAR (`LWAVE=.TRUE.`) |
| Metallic starting point | GW fails for metals | GW is designed for gapped systems; use cRPA or other methods for metals |
