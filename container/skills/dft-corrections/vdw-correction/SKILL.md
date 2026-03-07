# Van der Waals Corrections

## When to Use

Standard GGA/LDA functionals completely miss long-range van der Waals (dispersion) interactions. Apply vdW corrections when:

- **Layered materials**: graphite, h-BN, MoS2, black phosphorus -- interlayer binding is purely dispersive
- **Molecular crystals**: organic semiconductors, pharmaceutical polymorphs, ice
- **Adsorption on surfaces**: molecules on metals/oxides/2D materials
- **Metal-organic frameworks (MOFs)**: guest-host interactions, pore stability
- **Soft matter / biomolecules**: protein-ligand, DNA base stacking
- **Weakly bound dimers**: noble gas crystals, benzene dimer

**When NOT to use:**
- Bulk metals, covalent/ionic crystals where dispersion is negligible relative to other interactions
- When accuracy in strongly bound systems is the only concern (vdW corrections add negligible contribution)

## Prerequisites

- Quantum ESPRESSO 7.5 with `pw.x`
- For DFT-D3: QE compiled with libdftd3 support (check `configure` output), or use the standalone `dftd3` program
- For vdW-DF family: no special compilation needed (built into QE)
- Pseudopotentials compatible with PBE (for DFT-D2/D3, TS-vdW) or specific functionals (for vdW-DF)
- Python with `numpy`, `matplotlib`, `pymatgen`, `ase`

## Available Methods in QE

### Quick Reference

| Method | QE Keyword | Functional Base | Cost | Accuracy | Best For |
|--------|-----------|-----------------|------|----------|----------|
| DFT-D2 (Grimme) | `vdw_corr='grimme-d2'` | Any (PBE typical) | Negligible | Fair | Quick estimates |
| DFT-D3 (Grimme) | `vdw_corr='dft-d3'` | Any (PBE typical) | Negligible | Good | General purpose |
| DFT-D3(BJ) | `vdw_corr='dft-d3', dftd3_version=4` | Any | Negligible | Good | General purpose |
| TS-vdW | `vdw_corr='ts-vdw'` | PBE | Low | Good | Molecules/surfaces |
| vdW-DF | `input_dft='vdw-df'` | revPBE exchange | Moderate | Good | Layered/molecular |
| vdW-DF2 | `input_dft='vdw-df2'` | rPW86 exchange | Moderate | Good | General |
| rev-vdW-DF2 | `input_dft='rev-vdw-df2'` | Modified exchange | Moderate | Very good | General |
| rVV10 | `input_dft='rvv10'` | rPW86+PBE corr | Moderate | Very good | Layered, general |

### Method Details

#### 1. DFT-D2 (Grimme, 2006)

Simplest pairwise correction. Adds C6/R^6 terms with fixed C6 coefficients per element.

```
vdw_corr = 'grimme-d2'
```

Pros: Very cheap, easy to use, widely available.
Cons: No environment dependence, overbinds in some cases, outdated.

#### 2. DFT-D3 (Grimme, 2010)

Improved pairwise correction with coordination-number-dependent C6 coefficients and C8/R^8 terms.

```
vdw_corr = 'dft-d3'
```

With Becke-Johnson (BJ) damping (recommended):
```
vdw_corr = 'dft-d3'
dftd3_version = 4
```

Pros: Environment-dependent, good accuracy, nearly zero cost.
Cons: Still pairwise; parameters fitted to molecular data.

#### 3. Tkatchenko-Scheffler (TS-vdW)

Uses Hirshfeld partitioning to get effective atomic volumes, rescaling free-atom C6 coefficients.

```
vdw_corr = 'ts-vdw'
```

Pros: Self-consistent density-dependent C6. Good for molecules on surfaces.
Cons: Still pairwise at long range; no many-body screening.

#### 4. vdW-DF Family (Non-Local Correlation Functionals)

These are true non-local density functionals -- not a posteriori corrections. The correlation energy is computed from the electron density via a non-local kernel.

```
input_dft = 'vdw-df'       ! Original Dion et al. (2004)
input_dft = 'vdw-df2'      ! Lee et al. (2010)
input_dft = 'rev-vdw-df2'  ! Hamada (2014) -- often best for layered
input_dft = 'vdw-df-cx'    ! Berland & Hyldgaard (2014)
```

Pros: Seamless, no empirical parameters, captures non-local correlation from density.
Cons: More expensive than DFT-D; the exchange functional matters a lot.

#### 5. rVV10 (revised Vydrov-Van Voorhis)

Non-local functional, similar to vdW-DF but with a simpler kernel. Often gives excellent results.

```
input_dft = 'rvv10'
```

Pros: Good accuracy for solids and layered materials, moderate cost.
Cons: Has one fitted parameter (b value).

## Detailed Steps

### Benchmark System: Graphite Interlayer Distance

Graphite is the canonical benchmark for vdW corrections. PBE gives essentially no interlayer binding, while the experimental values are:
- Interlayer distance: c/2 = 3.354 A (c = 6.708 A)
- Interlayer binding energy: ~52 meV/atom
- In-plane lattice parameter: a = 2.461 A

We will compare multiple vdW methods against PBE and experiment.

### Step 1: Structure Setup

```python
#!/usr/bin/env python3
"""Generate graphite structure for vdW benchmark calculations."""

from pymatgen.core import Structure, Lattice
import numpy as np

# Graphite: hexagonal cell (AB stacking), space group P6_3/mmc (No. 194)
# 4 atoms per unit cell
a = 2.461  # in-plane lattice parameter (Angstrom)
c = 6.708  # c parameter (Angstrom), so interlayer distance = c/2 = 3.354 A

# Hexagonal lattice
lattice = Lattice.hexagonal(a, c)

# Atomic positions in fractional coordinates
# Wyckoff positions: 2b (0, 0, 1/4) and 2c (1/3, 2/3, 1/4)
species = ["C", "C", "C", "C"]
coords = [
    [0.0,     0.0,     0.25],  # Layer 1, sublattice A
    [0.0,     0.0,     0.75],  # Layer 2, sublattice A (directly above)
    [1.0/3.0, 2.0/3.0, 0.25],  # Layer 1, sublattice B
    [2.0/3.0, 1.0/3.0, 0.75],  # Layer 2, sublattice B (shifted = AB stacking)
]

structure = Structure(lattice, species, coords)
print(f"Graphite structure:")
print(f"  a = {a:.3f} A, c = {c:.3f} A")
print(f"  c/a = {c/a:.4f}")
print(f"  Interlayer distance = {c/2:.3f} A")
print(f"  Number of atoms: {len(structure)}")
print(f"\n{structure}")
```

### Step 2: QE Input Files for Each Method

#### 2a. PBE Only (no vdW) -- `graphite_pbe.in`

```
&CONTROL
    calculation   = 'vc-relax'
    prefix        = 'graphite_pbe'
    outdir        = './tmp_pbe/'
    pseudo_dir    = './pseudo/'
    forc_conv_thr = 1.0d-4
    etot_conv_thr = 1.0d-6
/

&SYSTEM
    ibrav         = 4
    celldm(1)     = 4.6499
    celldm(3)     = 2.7257
    nat           = 4
    ntyp          = 1
    ecutwfc       = 60.0
    ecutrho       = 480.0
    occupations   = 'smearing'
    smearing      = 'mv'
    degauss       = 0.01
/

&ELECTRONS
    conv_thr      = 1.0d-8
    mixing_beta   = 0.4
/

&IONS
    ion_dynamics  = 'bfgs'
/

&CELL
    cell_dynamics = 'bfgs'
    press         = 0.0
    cell_dofree   = 'all'
/

ATOMIC_SPECIES
  C  12.011  C.pbe-n-kjpaw_psl.1.0.0.UPF

ATOMIC_POSITIONS {crystal}
  C  0.000000  0.000000  0.250000
  C  0.000000  0.000000  0.750000
  C  0.333333  0.666667  0.250000
  C  0.666667  0.333333  0.750000

K_POINTS {automatic}
  12 12 4 0 0 0
```

Note: `celldm(1) = a (in bohr) = 2.461 * 1.8897 = 4.6499`; `celldm(3) = c/a = 6.708/2.461 = 2.7257`.

#### 2b. DFT-D2 -- `graphite_d2.in`

Same as PBE but add to `&SYSTEM`:
```
&SYSTEM
    ibrav         = 4
    celldm(1)     = 4.6499
    celldm(3)     = 2.7257
    nat           = 4
    ntyp          = 1
    ecutwfc       = 60.0
    ecutrho       = 480.0
    occupations   = 'smearing'
    smearing      = 'mv'
    degauss       = 0.01
    vdw_corr      = 'grimme-d2'
/
```

All other sections identical to the PBE input above.

#### 2c. DFT-D3(BJ) -- `graphite_d3.in`

```
&SYSTEM
    ibrav         = 4
    celldm(1)     = 4.6499
    celldm(3)     = 2.7257
    nat           = 4
    ntyp          = 1
    ecutwfc       = 60.0
    ecutrho       = 480.0
    occupations   = 'smearing'
    smearing      = 'mv'
    degauss       = 0.01
    vdw_corr      = 'dft-d3'
    dftd3_version = 4
/
```

#### 2d. TS-vdW -- `graphite_ts.in`

```
&SYSTEM
    ibrav         = 4
    celldm(1)     = 4.6499
    celldm(3)     = 2.7257
    nat           = 4
    ntyp          = 1
    ecutwfc       = 60.0
    ecutrho       = 480.0
    occupations   = 'smearing'
    smearing      = 'mv'
    degauss       = 0.01
    vdw_corr      = 'ts-vdw'
    ts_vdw_isolated = .false.
/
```

Note: `ts_vdw_isolated = .false.` is for periodic systems (default is `.false.` but being explicit is good practice).

#### 2e. vdW-DF -- `graphite_vdwdf.in`

```
&SYSTEM
    ibrav         = 4
    celldm(1)     = 4.6499
    celldm(3)     = 2.7257
    nat           = 4
    ntyp          = 1
    ecutwfc       = 60.0
    ecutrho       = 480.0
    occupations   = 'smearing'
    smearing      = 'mv'
    degauss       = 0.01
    input_dft     = 'vdw-df'
/
```

**Important**: When using `input_dft`, QE overrides the functional from the pseudopotential. You must use pseudopotentials that are compatible (PBE PPs are generally acceptable for vdW-DF variants, though strictly the exchange is different). For publication work, generate PPs with the correct exchange functional.

#### 2f. vdW-DF2 -- `graphite_vdwdf2.in`

```
    input_dft     = 'vdw-df2'
```

#### 2g. rev-vdW-DF2 -- `graphite_revvdwdf2.in`

```
    input_dft     = 'rev-vdw-df2'
```

#### 2h. rVV10 -- `graphite_rvv10.in`

```
    input_dft     = 'rvv10'
```

#### Full example input for rVV10 (`graphite_rvv10.in`):

```
&CONTROL
    calculation   = 'vc-relax'
    prefix        = 'graphite_rvv10'
    outdir        = './tmp_rvv10/'
    pseudo_dir    = './pseudo/'
    forc_conv_thr = 1.0d-4
    etot_conv_thr = 1.0d-6
/

&SYSTEM
    ibrav         = 4
    celldm(1)     = 4.6499
    celldm(3)     = 2.7257
    nat           = 4
    ntyp          = 1
    ecutwfc       = 60.0
    ecutrho       = 480.0
    occupations   = 'smearing'
    smearing      = 'mv'
    degauss       = 0.01
    input_dft     = 'rvv10'
/

&ELECTRONS
    conv_thr      = 1.0d-8
    mixing_beta   = 0.4
/

&IONS
    ion_dynamics  = 'bfgs'
/

&CELL
    cell_dynamics = 'bfgs'
    press         = 0.0
    cell_dofree   = 'all'
/

ATOMIC_SPECIES
  C  12.011  C.pbe-n-kjpaw_psl.1.0.0.UPF

ATOMIC_POSITIONS {crystal}
  C  0.000000  0.000000  0.250000
  C  0.000000  0.000000  0.750000
  C  0.333333  0.666667  0.250000
  C  0.666667  0.333333  0.750000

K_POINTS {automatic}
  12 12 4 0 0 0
```

### Step 3: Run All Calculations

```bash
#!/bin/bash
# Run all graphite vdW benchmark calculations

METHODS=("pbe" "d2" "d3" "ts" "vdwdf" "vdwdf2" "revvdwdf2" "rvv10")
NPOOL=2

for method in "${METHODS[@]}"; do
    echo "Running graphite_${method}..."
    mkdir -p tmp_${method}
    pw.x -npool ${NPOOL} < graphite_${method}.in > graphite_${method}.out 2>&1
    echo "  Done. Check graphite_${method}.out"
done

echo "All calculations complete."
```

### Step 4: Extract Results

```python
#!/usr/bin/env python3
"""Extract structural parameters from graphite vdW benchmark calculations."""

import re
import os
import numpy as np

BOHR_TO_ANG = 0.529177

def extract_vc_relax_results(filename):
    """
    Extract final lattice parameters and energy from a vc-relax output.
    """
    results = {}
    if not os.path.exists(filename):
        return None

    with open(filename, 'r') as f:
        content = f.read()

    # Check if calculation converged
    if 'Begin final coordinates' not in content and 'Final enthalpy' not in content:
        results['converged'] = False
        # Try to get last SCF energy at least
    else:
        results['converged'] = True

    # Extract final total energy
    energies = re.findall(r'!\s+total energy\s+=\s+([-\d.]+)\s+Ry', content)
    if energies:
        results['total_energy_ry'] = float(energies[-1])
        results['total_energy_ev'] = float(energies[-1]) * 13.6057

    # Extract final cell parameters (after "Begin final coordinates" or last CELL_PARAMETERS)
    cell_blocks = re.findall(
        r'CELL_PARAMETERS\s*\{?\s*(\w+)\s*\}?\s*\n'
        r'\s*([-\d.]+)\s+([-\d.]+)\s+([-\d.]+)\s*\n'
        r'\s*([-\d.]+)\s+([-\d.]+)\s+([-\d.]+)\s*\n'
        r'\s*([-\d.]+)\s+([-\d.]+)\s+([-\d.]+)',
        content
    )
    if cell_blocks:
        last_cell = cell_blocks[-1]
        unit = last_cell[0].lower()
        v1 = np.array([float(last_cell[1]), float(last_cell[2]), float(last_cell[3])])
        v2 = np.array([float(last_cell[4]), float(last_cell[5]), float(last_cell[6])])
        v3 = np.array([float(last_cell[7]), float(last_cell[8]), float(last_cell[9])])

        if 'bohr' in unit:
            v1 *= BOHR_TO_ANG
            v2 *= BOHR_TO_ANG
            v3 *= BOHR_TO_ANG
        elif 'alat' in unit:
            # Need to find alat
            alat_match = re.search(r'lattice parameter \(alat\)\s+=\s+([\d.]+)', content)
            if alat_match:
                alat_bohr = float(alat_match.group(1))
                v1 *= alat_bohr * BOHR_TO_ANG
                v2 *= alat_bohr * BOHR_TO_ANG
                v3 *= alat_bohr * BOHR_TO_ANG

        a_param = np.linalg.norm(v1)
        c_param = np.linalg.norm(v3)
        results['a'] = a_param
        results['c'] = c_param
        results['c_over_a'] = c_param / a_param
        results['interlayer_distance'] = c_param / 2.0  # for 2-layer cell

    # Extract final pressure
    pressures = re.findall(r'P=\s*([-\d.]+)', content)
    if pressures:
        results['final_pressure_kbar'] = float(pressures[-1])

    return results


# Process all methods
methods = {
    'PBE (no vdW)': 'graphite_pbe.out',
    'DFT-D2': 'graphite_d2.out',
    'DFT-D3(BJ)': 'graphite_d3.out',
    'TS-vdW': 'graphite_ts.out',
    'vdW-DF': 'graphite_vdwdf.out',
    'vdW-DF2': 'graphite_vdwdf2.out',
    'rev-vdW-DF2': 'graphite_revvdwdf2.out',
    'rVV10': 'graphite_rvv10.out',
}

# Experimental reference values
exp = {'a': 2.461, 'c': 6.708, 'interlayer_distance': 3.354}

print("=" * 80)
print("Graphite vdW Benchmark Results")
print("=" * 80)
print(f"{'Method':<18} {'a (A)':>8} {'c (A)':>8} {'c/a':>8} {'d_inter (A)':>12} {'Conv':>6}")
print("-" * 80)
print(f"{'Experiment':<18} {exp['a']:>8.3f} {exp['c']:>8.3f} {exp['c']/exp['a']:>8.4f} "
      f"{exp['interlayer_distance']:>12.3f} {'---':>6}")
print("-" * 80)

results_dict = {}
for method, fname in methods.items():
    res = extract_vc_relax_results(fname)
    if res is None:
        print(f"{method:<18} {'(file not found)':>50}")
        # Use expected/typical values for plotting demo
        continue

    a = res.get('a', 0)
    c = res.get('c', 0)
    c_a = res.get('c_over_a', 0)
    d = res.get('interlayer_distance', 0)
    conv = 'Yes' if res.get('converged', False) else 'No'
    print(f"{method:<18} {a:>8.3f} {c:>8.3f} {c_a:>8.4f} {d:>12.3f} {conv:>6}")
    results_dict[method] = res

# If no actual results found, use typical literature values for demonstration
if not results_dict:
    print("\n(No QE output files found. Using typical literature values for plot.)")
    results_dict = {
        'PBE (no vdW)':  {'a': 2.467, 'c': 8.65, 'interlayer_distance': 4.33},
        'DFT-D2':        {'a': 2.462, 'c': 6.45, 'interlayer_distance': 3.23},
        'DFT-D3(BJ)':    {'a': 2.462, 'c': 6.68, 'interlayer_distance': 3.34},
        'TS-vdW':        {'a': 2.463, 'c': 6.72, 'interlayer_distance': 3.36},
        'vdW-DF':        {'a': 2.467, 'c': 7.04, 'interlayer_distance': 3.52},
        'vdW-DF2':       {'a': 2.466, 'c': 6.73, 'interlayer_distance': 3.37},
        'rev-vdW-DF2':   {'a': 2.463, 'c': 6.70, 'interlayer_distance': 3.35},
        'rVV10':         {'a': 2.462, 'c': 6.68, 'interlayer_distance': 3.34},
    }
```

### Step 5: Python Comparison Plot

```python
#!/usr/bin/env python3
"""
Generate publication-quality comparison plot for graphite vdW benchmark.
Compares interlayer distance and c/a ratio across different vdW methods.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

# ===== DATA =====
# Typical literature values for graphite with different vdW methods (PBE-based)
# Modify these with your actual QE results

methods = [
    'PBE\n(no vdW)',
    'DFT-D2',
    'DFT-D3\n(BJ)',
    'TS-vdW',
    'vdW-DF',
    'vdW-DF2',
    'rev-vdW\n-DF2',
    'rVV10',
]

# Interlayer distance d = c/2 (Angstrom)
d_values = [4.33, 3.23, 3.34, 3.36, 3.52, 3.37, 3.35, 3.34]

# c/a ratio
c_a_values = [3.514, 2.621, 2.714, 2.730, 2.854, 2.729, 2.720, 2.714]

# In-plane lattice parameter a (Angstrom)
a_values = [2.467, 2.462, 2.462, 2.463, 2.467, 2.466, 2.463, 2.462]

# Experimental values
d_exp = 3.354
c_a_exp = 2.7257
a_exp = 2.461

# ===== PLOT =====
fig = plt.figure(figsize=(12, 10))
gs = gridspec.GridSpec(2, 2, hspace=0.35, wspace=0.3)

colors = plt.cm.Set2(np.linspace(0, 1, len(methods)))

# --- Panel (a): Interlayer distance ---
ax1 = fig.add_subplot(gs[0, 0])
bars = ax1.bar(range(len(methods)), d_values, color=colors, edgecolor='black', linewidth=0.8)
ax1.axhline(y=d_exp, color='red', linestyle='--', linewidth=2, label=f'Expt. ({d_exp:.3f} A)')
ax1.set_xticks(range(len(methods)))
ax1.set_xticklabels(methods, fontsize=8, rotation=0)
ax1.set_ylabel('Interlayer distance d (A)', fontsize=11)
ax1.set_title('(a) Interlayer Distance', fontsize=13, fontweight='bold')
ax1.legend(fontsize=10)
ax1.set_ylim(2.8, 4.8)

# Add value labels on bars
for bar, val in zip(bars, d_values):
    ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.03,
             f'{val:.2f}', ha='center', va='bottom', fontsize=8)

# --- Panel (b): c/a ratio ---
ax2 = fig.add_subplot(gs[0, 1])
bars2 = ax2.bar(range(len(methods)), c_a_values, color=colors, edgecolor='black', linewidth=0.8)
ax2.axhline(y=c_a_exp, color='red', linestyle='--', linewidth=2, label=f'Expt. ({c_a_exp:.4f})')
ax2.set_xticks(range(len(methods)))
ax2.set_xticklabels(methods, fontsize=8, rotation=0)
ax2.set_ylabel('c/a ratio', fontsize=11)
ax2.set_title('(b) c/a Ratio', fontsize=13, fontweight='bold')
ax2.legend(fontsize=10)
ax2.set_ylim(2.3, 3.8)

for bar, val in zip(bars2, c_a_values):
    ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
             f'{val:.3f}', ha='center', va='bottom', fontsize=8)

# --- Panel (c): Percent error in interlayer distance ---
ax3 = fig.add_subplot(gs[1, 0])
errors_d = [(d - d_exp) / d_exp * 100 for d in d_values]
bar_colors_err = ['red' if e > 5 else 'green' if abs(e) < 2 else 'orange' for e in errors_d]
bars3 = ax3.bar(range(len(methods)), errors_d, color=bar_colors_err, edgecolor='black', linewidth=0.8)
ax3.axhline(y=0, color='red', linestyle='--', linewidth=1.5)
ax3.axhspan(-2, 2, alpha=0.15, color='green', label='Within 2%')
ax3.set_xticks(range(len(methods)))
ax3.set_xticklabels(methods, fontsize=8, rotation=0)
ax3.set_ylabel('Error in d (%)', fontsize=11)
ax3.set_title('(c) Percent Error in Interlayer Distance', fontsize=13, fontweight='bold')
ax3.legend(fontsize=10, loc='upper left')

for bar, val in zip(bars3, errors_d):
    y_pos = val + 0.3 if val >= 0 else val - 1.0
    ax3.text(bar.get_x() + bar.get_width()/2, y_pos,
             f'{val:.1f}%', ha='center', va='bottom', fontsize=8, fontweight='bold')

# --- Panel (d): In-plane lattice parameter ---
ax4 = fig.add_subplot(gs[1, 1])
bars4 = ax4.bar(range(len(methods)), a_values, color=colors, edgecolor='black', linewidth=0.8)
ax4.axhline(y=a_exp, color='red', linestyle='--', linewidth=2, label=f'Expt. ({a_exp:.3f} A)')
ax4.set_xticks(range(len(methods)))
ax4.set_xticklabels(methods, fontsize=8, rotation=0)
ax4.set_ylabel('a (A)', fontsize=11)
ax4.set_title('(d) In-Plane Lattice Parameter', fontsize=13, fontweight='bold')
ax4.legend(fontsize=10)
ax4.set_ylim(2.45, 2.48)

for bar, val in zip(bars4, a_values):
    ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.0005,
             f'{val:.3f}', ha='center', va='bottom', fontsize=8)

plt.suptitle('Graphite: vdW Correction Benchmark', fontsize=15, fontweight='bold', y=1.01)
plt.savefig('graphite_vdw_benchmark.png', dpi=300, bbox_inches='tight')
plt.savefig('graphite_vdw_benchmark.pdf', bbox_inches='tight')
print("Saved: graphite_vdw_benchmark.png, graphite_vdw_benchmark.pdf")
plt.show()
```

### Step 6: Binding Energy Curve (Optional Advanced)

Compute the interlayer binding energy as a function of interlayer distance to visualize the vdW potential well:

```python
#!/usr/bin/env python3
"""
Generate QE input files for graphite binding energy curve.
Vary c while keeping a fixed, compute E(c) for each vdW method.
"""

import numpy as np
import os

# Fixed in-plane lattice parameter
a_bohr = 4.6499  # 2.461 Angstrom in bohr

# Range of c/a ratios to scan
c_over_a_values = np.arange(2.2, 3.8, 0.1)  # covers d = 2.7 to 4.7 Angstrom

# Methods to compare
methods_config = {
    'pbe': {'vdw_line': ''},
    'd3': {'vdw_line': "    vdw_corr      = 'dft-d3'\n    dftd3_version = 4"},
    'rvv10': {'vdw_line': "    input_dft     = 'rvv10'"},
}

template = """&CONTROL
    calculation   = 'scf'
    prefix        = 'graphite_{method}_{idx:03d}'
    outdir        = './tmp_{method}_scan/'
    pseudo_dir    = './pseudo/'
/

&SYSTEM
    ibrav         = 4
    celldm(1)     = {a_bohr:.4f}
    celldm(3)     = {c_over_a:.6f}
    nat           = 4
    ntyp          = 1
    ecutwfc       = 60.0
    ecutrho       = 480.0
    occupations   = 'smearing'
    smearing      = 'mv'
    degauss       = 0.01
{vdw_line}
/

&ELECTRONS
    conv_thr      = 1.0d-8
    mixing_beta   = 0.4
/

ATOMIC_SPECIES
  C  12.011  C.pbe-n-kjpaw_psl.1.0.0.UPF

ATOMIC_POSITIONS {{crystal}}
  C  0.000000  0.000000  0.250000
  C  0.000000  0.000000  0.750000
  C  0.333333  0.666667  0.250000
  C  0.666667  0.333333  0.750000

K_POINTS {{automatic}}
  12 12 4 0 0 0
"""

for method, config in methods_config.items():
    os.makedirs(f'tmp_{method}_scan', exist_ok=True)
    for idx, c_a in enumerate(c_over_a_values):
        input_text = template.format(
            method=method,
            idx=idx,
            a_bohr=a_bohr,
            c_over_a=c_a,
            vdw_line=config['vdw_line'],
        )
        fname = f'graphite_{method}_scan_{idx:03d}.in'
        with open(fname, 'w') as f:
            f.write(input_text)

    # Generate run script
    with open(f'run_{method}_scan.sh', 'w') as f:
        f.write('#!/bin/bash\n')
        for idx in range(len(c_over_a_values)):
            f.write(f'pw.x -npool 2 < graphite_{method}_scan_{idx:03d}.in '
                    f'> graphite_{method}_scan_{idx:03d}.out 2>&1\n')
    os.chmod(f'run_{method}_scan.sh', 0o755)

print(f"Generated {len(c_over_a_values)} input files for each of {len(methods_config)} methods.")
print("Run: bash run_<method>_scan.sh")


# ---- Post-processing: Plot binding energy curves ----
def plot_binding_curves():
    """Plot E(d) binding energy curves for different vdW methods."""
    import re

    fig, ax = plt.subplots(figsize=(8, 6))
    colors_map = {'pbe': 'gray', 'd3': 'blue', 'rvv10': 'red'}
    labels_map = {'pbe': 'PBE (no vdW)', 'd3': 'DFT-D3(BJ)', 'rvv10': 'rVV10'}

    for method in ['pbe', 'd3', 'rvv10']:
        energies = []
        distances = []
        for idx, c_a in enumerate(c_over_a_values):
            fname = f'graphite_{method}_scan_{idx:03d}.out'
            if not os.path.exists(fname):
                continue
            with open(fname, 'r') as f:
                content = f.read()
            match = re.findall(r'!\s+total energy\s+=\s+([-\d.]+)\s+Ry', content)
            if match:
                e_ry = float(match[-1])
                d_ang = c_a * a_bohr * 0.529177 / 2.0  # interlayer distance
                energies.append(e_ry * 13605.7 / 4.0)  # meV/atom
                distances.append(d_ang)

        if energies:
            # Shift so E(max_d) = 0
            e_ref = energies[-1]
            energies = [e - e_ref for e in energies]
            ax.plot(distances, energies, 'o-', color=colors_map[method],
                    label=labels_map[method], linewidth=2, markersize=5)

    ax.axvline(x=3.354, color='green', linestyle=':', linewidth=1.5, label='Expt. d = 3.354 A')
    ax.set_xlabel('Interlayer distance (A)', fontsize=12)
    ax.set_ylabel('Binding energy (meV/atom)', fontsize=12)
    ax.set_title('Graphite: Interlayer Binding Energy Curve', fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    ax.set_xlim(2.8, 5.0)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('graphite_binding_curve.png', dpi=300, bbox_inches='tight')
    print("Saved: graphite_binding_curve.png")
    plt.show()


import matplotlib.pyplot as plt
plot_binding_curves()
```

## Key Parameters

| Parameter | Values | Notes |
|-----------|--------|-------|
| `vdw_corr` | `'grimme-d2'`, `'dft-d3'`, `'ts-vdw'` | Pairwise corrections |
| `dftd3_version` | 2 (zero-damping), 3 (zero w/ ATM), 4 (BJ damping), 6 (BJ w/ ATM) | Only with `vdw_corr='dft-d3'` |
| `input_dft` | `'vdw-df'`, `'vdw-df2'`, `'rev-vdw-df2'`, `'rvv10'` | Non-local functionals |
| `ts_vdw_isolated` | `.true.` / `.false.` | `.false.` for periodic systems |
| `london_s6` | Scaling factor for D2 | Default depends on functional |
| `london_rcut` | Cutoff radius for D2 (bohr) | Default: 200 bohr |
| K-point density | Denser along stacking direction | e.g., 12x12x4 for graphite |
| `ecutwfc` | Same as without vdW for DFT-D; may need increase for vdW-DF | Check convergence |

### Method Selection Guide

| System Type | Recommended Methods | Avoid |
|-------------|-------------------|-------|
| Layered materials (graphite, MoS2) | rVV10, rev-vdW-DF2, DFT-D3(BJ) | Plain PBE |
| Molecular crystals | DFT-D3(BJ), rVV10 | DFT-D2 (overbinds) |
| Molecule on metal surface | TS-vdW, DFT-D3(BJ) | vdW-DF (overestimates) |
| MOFs | DFT-D3(BJ), rVV10 | DFT-D2 |
| Noble gas crystals | rVV10, DFT-D3(BJ) | vdW-DF |
| Consistency with MLIP training | DFT-D3(BJ) (most common in datasets) | Mix of methods |

## Interpreting Results

### What to Compare

1. **Interlayer / intermolecular distance**: The most sensitive structural parameter. PBE without vdW will dramatically overestimate this (or find no binding at all).

2. **Binding energy**: Compute as E_bind = (E_bulk - n * E_layer) / n. Should be negative (bound). Typical values:
   - Graphite: ~52 meV/atom
   - h-BN: ~46 meV/atom
   - MoS2: ~76 meV/atom

3. **In-plane parameters**: Should not change much with vdW corrections. If they do, something may be wrong.

4. **Bulk modulus along c-axis**: vdW corrections typically increase the c-axis stiffness substantially.

### Expected Improvements with vdW

| Property | Without vdW | With vdW | Typical Improvement |
|----------|-------------|----------|---------------------|
| Interlayer distance | 20-30% too large (or unbound) | Within 1-3% of experiment | Major |
| Binding energy | ~0 (unbound) | Correct order of magnitude | Critical |
| c/a ratio | Wildly wrong | Within 1-3% | Major |
| In-plane lattice param | Already good (~0.5%) | Slightly improved | Minor |
| Elastic constants (C33) | Severely underestimated | Reasonable | Major |

## Common Issues

1. **PBE gives no interlayer binding**: This is expected -- PBE lacks dispersion. The layers will drift apart during vc-relax. You MUST use a vdW correction for layered materials.

2. **DFT-D2 overbinding**: Grimme-D2 uses fixed C6 coefficients that can overbind, especially for metallic systems. Prefer DFT-D3 or rVV10.

3. **vdW-DF overestimates distances**: The original vdW-DF with revPBE exchange tends to give interlayer distances ~5% too large. Use rev-vdW-DF2 or rVV10 instead.

4. **Mixing `input_dft` with PP functional**: When using `input_dft = 'vdw-df'`, QE overrides the functional from the pseudopotential. A warning is printed. This is usually acceptable for testing but for publication work, use pseudopotentials generated with the correct exchange functional.

5. **DFT-D3 not available**: If QE was not compiled with libdftd3, you will get an error. Options:
   - Recompile QE with `-D__DFTD3` flag and linked libdftd3
   - Use the standalone `dftd3` program to compute the correction post-hoc
   - Use DFT-D2 (always available) or vdW-DF family (always available) as alternatives

6. **Convergence with vdW-DF**: Non-local functionals can be slower to converge. Reduce `mixing_beta` to 0.2-0.3 if SCF has trouble.

7. **Memory with vdW-DF**: The non-local kernel requires additional memory. For large cells, this can be significant. DFT-D corrections have negligible memory overhead.

8. **Inconsistent results across methods**: Different vdW methods can give different results (up to 10% variation in binding energies). This is inherent to the approximations. For systematic studies, pick ONE method and use it consistently.

9. **K-point convergence**: Layered materials need careful k-point convergence, especially along the stacking direction. Use at least 4 k-points along c for graphite; test convergence.
