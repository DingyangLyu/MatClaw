# Van der Waals Dispersion Corrections

## When to Use

Standard GGA/LDA functionals completely miss long-range van der Waals (dispersion) interactions. Apply vdW corrections when:

- **Layered materials**: graphite, h-BN, MoS2, WS2, black phosphorus, MXenes -- interlayer binding is purely dispersive
- **Molecular crystals**: organic semiconductors, pharmaceutical polymorphs, ice, amino acid crystals
- **Adsorption on surfaces**: molecules on metals, oxides, or 2D materials (catalysis, sensing)
- **Metal-organic frameworks (MOFs)**: guest-host interactions, pore stability, gas adsorption
- **Weakly bound complexes**: noble gas crystals, benzene dimer, DNA base stacking
- **Intercalation compounds**: Li-graphite, Na-MoS2 (battery anodes)
- **Heterostructures**: vdW-bonded 2D material stacks (graphene/h-BN, MoS2/WS2)

**When NOT to use:**
- Bulk metals, strongly covalent or ionic crystals where dispersion is negligible
- When only in-plane properties of a 2D material are needed (interlayer interactions irrelevant)

## Method Selection

| Method | QE Keyword | VASP Keyword | Cost | Accuracy | Best For |
|---|---|---|---|---|---|
| DFT-D2 (Grimme) | `vdw_corr='grimme-d2'` | `IVDW=1` | Negligible | Fair | Quick estimates |
| DFT-D3 (zero-damp) | `vdw_corr='dft-d3'` | `IVDW=11` | Negligible | Good | General purpose |
| DFT-D3(BJ) | `vdw_corr='dft-d3', dftd3_version=4` | `IVDW=12` | Negligible | Good | General purpose (recommended) |
| TS-vdW | `vdw_corr='ts-vdw'` | `IVDW=2` | Low | Good | Molecules on surfaces |
| vdW-DF | `input_dft='vdw-df'` | `LUSE_VDW=.TRUE., AGGAC=0.0` | Moderate | Good | Layered, molecular |
| vdW-DF2 | `input_dft='vdw-df2'` | `LUSE_VDW=.TRUE.`, `GGA=ML` | Moderate | Good | General |
| rev-vdW-DF2 | `input_dft='rev-vdw-df2'` | `LUSE_VDW=.TRUE.`, `GGA=MK`, `PARAM1/2` | Moderate | Very good | Layered, general |
| optB86b-vdW | `input_dft='vdw-df-obk8'` | `GGA=OB`, `LUSE_VDW=.TRUE.`, `AGGAC=0.0` | Moderate | Very good | Layered materials |
| rVV10 | `input_dft='rvv10'` | `LUSE_VDW=.TRUE.`, `GGA=MK`, `PARAM1/2` | Moderate | Very good | General solids |

## Prerequisites

- Quantum ESPRESSO 7.5 with `pw.x`
- For DFT-D3: QE compiled with libdftd3 (check `configure` output), or standalone `dftd3` program
- For vdW-DF family: no special compilation needed (built into QE)
- Pseudopotentials compatible with PBE (for DFT-D corrections) or matching functional (for vdW-DF)
- For VASP: standard PAW potentials; `LUSE_VDW` requires `vdw_kernel.bindat` file in run directory
- Python: `numpy`, `matplotlib`, `pymatgen`, `ase`

## Detailed Steps

### Benchmark System: Graphite Interlayer Distance

Graphite is the canonical vdW benchmark. PBE gives essentially no interlayer binding, while experimental values are:
- Interlayer distance: c/2 = 3.354 A (c = 6.708 A)
- Interlayer binding energy: ~52 meV/atom
- In-plane lattice parameter: a = 2.461 A

### Step 1: Generate Structure

```python
#!/usr/bin/env python3
"""Generate graphite structure for vdW benchmark calculations."""

from pymatgen.core import Structure, Lattice
import numpy as np

# Graphite: hexagonal cell (AB stacking), P6_3/mmc (No. 194), 4 atoms
a = 2.461  # Angstrom
c = 6.708  # Angstrom, c/2 = 3.354 A interlayer distance

lattice = Lattice.hexagonal(a, c)

species = ["C", "C", "C", "C"]
coords = [
    [0.0,     0.0,     0.25],   # Layer 1, A sublattice
    [0.0,     0.0,     0.75],   # Layer 2, A sublattice (directly above)
    [1.0/3.0, 2.0/3.0, 0.25],   # Layer 1, B sublattice
    [2.0/3.0, 1.0/3.0, 0.75],   # Layer 2, B sublattice (shifted = AB stacking)
]

structure = Structure(lattice, species, coords)
print(f"Graphite structure:")
print(f"  a = {a:.3f} A, c = {c:.3f} A, c/a = {c/a:.4f}")
print(f"  Interlayer distance = {c/2:.3f} A")
print(f"  Number of atoms: {len(structure)}")

# Convert to Bohr for QE
a_bohr = a / 0.529177  # = 4.6499 Bohr
c_over_a = c / a       # = 2.7257
print(f"  a (Bohr) = {a_bohr:.4f}, c/a = {c_over_a:.4f}")
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

Note: `celldm(1) = a(Bohr) = 2.461 / 0.529177 = 4.6499`; `celldm(3) = c/a = 6.708 / 2.461 = 2.7257`.

#### 2b. DFT-D3(BJ) -- `graphite_d3.in`

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
    vdw_corr      = 'dft-d3'
    dftd3_version = 4
/
```

All other namelists identical to the PBE input above.

#### 2c. rVV10 -- `graphite_rvv10.in`

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

**Important**: When using `input_dft`, QE overrides the functional from the pseudopotential. PBE pseudopotentials are generally acceptable for vdW-DF variants, though strictly the exchange functional differs. For publication work, generate pseudopotentials with the correct exchange functional.

#### Additional QE Method Variants

For other methods, replace or add the relevant keywords in `&SYSTEM`:

```
! DFT-D2
vdw_corr = 'grimme-d2'

! TS-vdW (Tkatchenko-Scheffler)
vdw_corr = 'ts-vdw'
ts_vdw_isolated = .false.   ! .false. for periodic systems

! vdW-DF (original Dion et al. 2004)
input_dft = 'vdw-df'

! vdW-DF2 (Lee et al. 2010)
input_dft = 'vdw-df2'

! rev-vdW-DF2 (Hamada 2014)
input_dft = 'rev-vdw-df2'

! vdW-DF-cx (Berland & Hyldgaard 2014)
input_dft = 'vdw-df-cx'

! XDM (exchange-hole dipole moment)
vdw_corr = 'xdm'
```

### Step 2d: VASP INCAR for vdW Corrections

```
! ========== DFT-D3 (Grimme, BJ damping) ==========
IVDW    = 12         ! 11 = D3 zero-damp, 12 = D3(BJ)
! (No other special tags needed; works with any GGA functional)

! ========== DFT-D3 (zero damping) ==========
! IVDW    = 11

! ========== TS-vdW (Tkatchenko-Scheffler) ==========
! IVDW    = 2

! ========== optB86b-vdW (non-local vdW-DF) ==========
! GGA       = OB       ! optB86b exchange
! LUSE_VDW  = .TRUE.   ! Enable non-local vdW correlation
! AGGAC     = 0.0      ! Turn off GGA correlation (replaced by vdW-DF)
! Note: requires vdw_kernel.bindat in run directory

! ========== optPBE-vdW ==========
! GGA       = OR       ! optPBE exchange
! LUSE_VDW  = .TRUE.
! AGGAC     = 0.0

! ========== rev-vdW-DF2 ==========
! GGA       = MK       ! rPW86 exchange
! LUSE_VDW  = .TRUE.
! AGGAC     = 0.0
! PARAM1    = 0.1234   ! rev-vdW-DF2 parameters
! PARAM2    = 0.7110

! Common settings for all methods:
ENCUT   = 600         ! Higher cutoff for layered materials
EDIFF   = 1E-8
ISMEAR  = 0
SIGMA   = 0.05
PREC    = Accurate
```

VASP note: For `LUSE_VDW=.TRUE.`, copy `vdw_kernel.bindat` to the calculation directory. This file is generated once with `vdw_kernel.x` or downloaded from the VASP wiki.

### Step 3: Run All Calculations

```bash
#!/bin/bash
# Run all graphite vdW benchmark calculations (QE)

METHODS=("pbe" "d3" "rvv10" "vdwdf2" "revvdwdf2")
NPOOL=2

for method in "${METHODS[@]}"; do
    echo "Running graphite_${method}..."
    mkdir -p tmp_${method}
    pw.x -npool ${NPOOL} < graphite_${method}.in > graphite_${method}.out 2>&1
    echo "  Done. Check graphite_${method}.out"
done

echo "All calculations complete."
```

### Step 4: Extract and Compare Results

```python
#!/usr/bin/env python3
"""Extract structural parameters from graphite vdW benchmark calculations."""

import re
import os
import numpy as np

BOHR_TO_ANG = 0.529177


def extract_vc_relax_results(filename):
    """Extract final lattice parameters and energy from a vc-relax output."""
    results = {}
    if not os.path.exists(filename):
        return None

    with open(filename, 'r') as f:
        content = f.read()

    results['converged'] = ('Begin final coordinates' in content
                            or 'Final enthalpy' in content)

    # Total energy
    energies = re.findall(r'!\s+total energy\s+=\s+([-\d.]+)\s+Ry', content)
    if energies:
        results['total_energy_ry'] = float(energies[-1])
        results['total_energy_ev'] = float(energies[-1]) * 13.6057

    # Final cell parameters
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
        v1 = np.array([float(last_cell[1]), float(last_cell[2]),
                        float(last_cell[3])])
        v2 = np.array([float(last_cell[4]), float(last_cell[5]),
                        float(last_cell[6])])
        v3 = np.array([float(last_cell[7]), float(last_cell[8]),
                        float(last_cell[9])])

        if 'bohr' in unit:
            v1 *= BOHR_TO_ANG
            v2 *= BOHR_TO_ANG
            v3 *= BOHR_TO_ANG
        elif 'alat' in unit:
            alat_match = re.search(
                r'lattice parameter \(alat\)\s+=\s+([\d.]+)', content)
            if alat_match:
                alat_bohr = float(alat_match.group(1))
                v1 *= alat_bohr * BOHR_TO_ANG
                v2 *= alat_bohr * BOHR_TO_ANG
                v3 *= alat_bohr * BOHR_TO_ANG

        results['a'] = np.linalg.norm(v1)
        results['c'] = np.linalg.norm(v3)
        results['c_over_a'] = results['c'] / results['a']
        results['interlayer_distance'] = results['c'] / 2.0

    return results


# Process all methods
methods = {
    'PBE (no vdW)':  'graphite_pbe.out',
    'DFT-D3(BJ)':   'graphite_d3.out',
    'rVV10':         'graphite_rvv10.out',
    'vdW-DF2':       'graphite_vdwdf2.out',
    'rev-vdW-DF2':   'graphite_revvdwdf2.out',
}

exp = {'a': 2.461, 'c': 6.708, 'interlayer_distance': 3.354}

print("=" * 80)
print("Graphite vdW Benchmark Results")
print("=" * 80)
print(f"{'Method':<18} {'a (A)':>8} {'c (A)':>8} {'c/a':>8} "
      f"{'d_inter (A)':>12} {'Conv':>6}")
print("-" * 80)
print(f"{'Experiment':<18} {exp['a']:>8.3f} {exp['c']:>8.3f} "
      f"{exp['c']/exp['a']:>8.4f} {exp['interlayer_distance']:>12.3f} "
      f"{'---':>6}")
print("-" * 80)

results_dict = {}
for method, fname in methods.items():
    res = extract_vc_relax_results(fname)
    if res is None:
        print(f"{method:<18} {'(file not found)':>50}")
        continue
    a = res.get('a', 0)
    c = res.get('c', 0)
    c_a = res.get('c_over_a', 0)
    d = res.get('interlayer_distance', 0)
    conv = 'Yes' if res.get('converged', False) else 'No'
    print(f"{method:<18} {a:>8.3f} {c:>8.3f} {c_a:>8.4f} "
          f"{d:>12.3f} {conv:>6}")
    results_dict[method] = res

# Typical literature values if no QE output found
if not results_dict:
    print("\n(No QE output files found. Typical literature values below.)")
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
    for method, res in results_dict.items():
        d = res['interlayer_distance']
        err = (d - exp['interlayer_distance']) / exp['interlayer_distance'] * 100
        print(f"  {method:<18} d = {d:.3f} A  (error: {err:+.1f}%)")
```

### Step 5: Publication-Quality Comparison Plot

```python
#!/usr/bin/env python3
"""
Generate publication-quality comparison plot for graphite vdW benchmark.
Compares interlayer distance and binding energy across methods.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

# ===== DATA (typical literature values, update with your QE results) =====
methods = [
    'PBE\n(no vdW)', 'DFT-D2', 'DFT-D3\n(BJ)', 'TS-vdW',
    'vdW-DF', 'vdW-DF2', 'rev-vdW\n-DF2', 'rVV10',
]

d_values = [4.33, 3.23, 3.34, 3.36, 3.52, 3.37, 3.35, 3.34]  # interlayer d (A)
E_bind = [0, -65, -52, -48, -35, -50, -52, -53]  # binding energy (meV/atom)
a_values = [2.467, 2.462, 2.462, 2.463, 2.467, 2.466, 2.463, 2.462]  # in-plane a (A)

d_exp = 3.354
E_bind_exp = -52  # meV/atom (Zacharia et al. 2004)
a_exp = 2.461

fig = plt.figure(figsize=(14, 10))
gs = gridspec.GridSpec(2, 2, hspace=0.35, wspace=0.3)
colors = plt.cm.Set2(np.linspace(0, 1, len(methods)))

# --- (a) Interlayer distance ---
ax1 = fig.add_subplot(gs[0, 0])
bars = ax1.bar(range(len(methods)), d_values, color=colors,
               edgecolor='black', linewidth=0.8)
ax1.axhline(y=d_exp, color='red', linestyle='--', linewidth=2,
            label=f'Expt. ({d_exp:.3f} A)')
ax1.set_xticks(range(len(methods)))
ax1.set_xticklabels(methods, fontsize=8)
ax1.set_ylabel('Interlayer distance d (A)', fontsize=11)
ax1.set_title('(a) Interlayer Distance', fontsize=13, fontweight='bold')
ax1.legend(fontsize=10)
ax1.set_ylim(2.8, 4.8)
for bar, val in zip(bars, d_values):
    ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.03,
             f'{val:.2f}', ha='center', va='bottom', fontsize=8)

# --- (b) Binding energy ---
ax2 = fig.add_subplot(gs[0, 1])
bars2 = ax2.bar(range(len(methods)), E_bind, color=colors,
                edgecolor='black', linewidth=0.8)
ax2.axhline(y=E_bind_exp, color='red', linestyle='--', linewidth=2,
            label=f'Expt. ({E_bind_exp} meV/atom)')
ax2.set_xticks(range(len(methods)))
ax2.set_xticklabels(methods, fontsize=8)
ax2.set_ylabel('Binding energy (meV/atom)', fontsize=11)
ax2.set_title('(b) Interlayer Binding Energy', fontsize=13, fontweight='bold')
ax2.legend(fontsize=10)
for bar, val in zip(bars2, E_bind):
    y_pos = val - 3 if val < 0 else val + 1
    ax2.text(bar.get_x() + bar.get_width()/2, y_pos,
             f'{val}', ha='center', va='top', fontsize=8, fontweight='bold')

# --- (c) Percent error in interlayer distance ---
ax3 = fig.add_subplot(gs[1, 0])
errors_d = [(d - d_exp) / d_exp * 100 for d in d_values]
bar_colors_err = ['red' if e > 5 else 'green' if abs(e) < 2 else 'orange'
                  for e in errors_d]
bars3 = ax3.bar(range(len(methods)), errors_d, color=bar_colors_err,
                edgecolor='black', linewidth=0.8)
ax3.axhline(y=0, color='red', linestyle='--', linewidth=1.5)
ax3.axhspan(-2, 2, alpha=0.15, color='green', label='Within 2%')
ax3.set_xticks(range(len(methods)))
ax3.set_xticklabels(methods, fontsize=8)
ax3.set_ylabel('Error in d (%)', fontsize=11)
ax3.set_title('(c) Percent Error in Interlayer Distance', fontsize=13,
              fontweight='bold')
ax3.legend(fontsize=10, loc='upper left')
for bar, val in zip(bars3, errors_d):
    y_pos = val + 0.3 if val >= 0 else val - 1.0
    ax3.text(bar.get_x() + bar.get_width()/2, y_pos,
             f'{val:.1f}%', ha='center', va='bottom', fontsize=8,
             fontweight='bold')

# --- (d) In-plane lattice parameter ---
ax4 = fig.add_subplot(gs[1, 1])
bars4 = ax4.bar(range(len(methods)), a_values, color=colors,
                edgecolor='black', linewidth=0.8)
ax4.axhline(y=a_exp, color='red', linestyle='--', linewidth=2,
            label=f'Expt. ({a_exp:.3f} A)')
ax4.set_xticks(range(len(methods)))
ax4.set_xticklabels(methods, fontsize=8)
ax4.set_ylabel('a (A)', fontsize=11)
ax4.set_title('(d) In-Plane Lattice Parameter', fontsize=13, fontweight='bold')
ax4.legend(fontsize=10)
ax4.set_ylim(2.45, 2.48)
for bar, val in zip(bars4, a_values):
    ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.0005,
             f'{val:.3f}', ha='center', va='bottom', fontsize=8)

plt.suptitle('Graphite: vdW Correction Benchmark', fontsize=15,
             fontweight='bold', y=1.01)
plt.savefig('graphite_vdw_benchmark.png', dpi=300, bbox_inches='tight')
plt.savefig('graphite_vdw_benchmark.pdf', bbox_inches='tight')
print("Saved: graphite_vdw_benchmark.png, graphite_vdw_benchmark.pdf")
plt.show()
```

### Step 6: Binding Energy Curve

Compute interlayer binding energy as a function of distance to visualize the vdW potential well.

```python
#!/usr/bin/env python3
"""
Generate QE inputs for graphite binding energy curve.
Vary c (interlayer distance) at fixed a, compute E(d) for each vdW method.
"""

import numpy as np
import os

a_bohr = 4.6499  # Fixed in-plane parameter (Bohr)
c_over_a_values = np.arange(2.2, 3.8, 0.1)  # d = 2.7 to 4.7 Angstrom

methods_config = {
    'pbe':   {'vdw_line': ''},
    'd3':    {'vdw_line': "    vdw_corr      = 'dft-d3'\n    dftd3_version = 4"},
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
            method=method, idx=idx, a_bohr=a_bohr,
            c_over_a=c_a, vdw_line=config['vdw_line'],
        )
        fname = f'graphite_{method}_scan_{idx:03d}.in'
        with open(fname, 'w') as f:
            f.write(input_text)

print(f"Generated {len(c_over_a_values)} input files per method "
      f"for {len(methods_config)} methods.")
print("Run: for f in graphite_*_scan_*.in; do pw.x -npool 2 < $f > ${f%.in}.out; done")
```

#### Post-Processing: Plot Binding Curves

```python
#!/usr/bin/env python3
"""Plot E(d) binding energy curves for different vdW methods."""

import re
import os
import numpy as np
import matplotlib.pyplot as plt

a_bohr = 4.6499
c_over_a_values = np.arange(2.2, 3.8, 0.1)

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
            d_ang = c_a * a_bohr * 0.529177 / 2.0
            energies.append(e_ry * 13605.7 / 4.0)  # meV/atom
            distances.append(d_ang)

    if energies:
        e_ref = energies[-1]  # Reference at large distance
        energies = [e - e_ref for e in energies]
        ax.plot(distances, energies, 'o-', color=colors_map[method],
                label=labels_map[method], linewidth=2, markersize=5)

if not any(os.path.exists(f'graphite_pbe_scan_{i:03d}.out')
           for i in range(len(c_over_a_values))):
    # Demo with synthetic curves
    d = np.linspace(2.8, 5.0, 100)
    # PBE: very shallow or no minimum
    ax.plot(d, 2 * np.exp(-0.5 * (d - 3.0)) - 0.5, '-',
            color='gray', linewidth=2, label='PBE (no vdW)')
    # D3: clear minimum near 3.34 A
    ax.plot(d, -52 * (1 - np.exp(-1.5 * (d - 3.34)))**2, '-',
            color='blue', linewidth=2, label='DFT-D3(BJ)')
    # rVV10: similar minimum
    ax.plot(d, -53 * (1 - np.exp(-1.4 * (d - 3.34)))**2, '-',
            color='red', linewidth=2, label='rVV10')
    ax.set_ylim(-70, 20)

ax.axvline(x=3.354, color='green', linestyle=':', linewidth=1.5,
           label='Expt. d = 3.354 A')
ax.axhline(y=-52, color='green', linestyle=':', linewidth=1, alpha=0.5)
ax.set_xlabel('Interlayer distance (A)', fontsize=12)
ax.set_ylabel('Binding energy (meV/atom)', fontsize=12)
ax.set_title('Graphite: Interlayer Binding Energy Curve',
             fontsize=14, fontweight='bold')
ax.legend(fontsize=11)
ax.set_xlim(2.8, 5.0)
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('graphite_binding_curve.png', dpi=300, bbox_inches='tight')
print("Saved: graphite_binding_curve.png")
plt.show()
```

### Step 7: MoS2 Benchmark (Second Layered Material)

```python
#!/usr/bin/env python3
"""
MoS2 vdW benchmark: compare interlayer distance with and without vdW.
2H-MoS2: a = 3.160 A, c = 12.295 A, d_inter = c/2 = 6.148 A
"""

import os

WORK_DIR = os.path.abspath("vdw_mos2")
os.makedirs(WORK_DIR, exist_ok=True)

a_bohr = 3.160 / 0.529177  # 5.9721 Bohr
c_over_a = 12.295 / 3.160   # 3.8909

# DFT-D3(BJ) vc-relax
mos2_d3 = f"""&CONTROL
    calculation   = 'vc-relax'
    prefix        = 'mos2_d3'
    outdir        = './tmp/'
    pseudo_dir    = './pseudo/'
    forc_conv_thr = 1.0d-4
    etot_conv_thr = 1.0d-6
/

&SYSTEM
    ibrav         = 4
    celldm(1)     = {a_bohr:.4f}
    celldm(3)     = {c_over_a:.4f}
    nat           = 6
    ntyp          = 2
    ecutwfc       = 60.0
    ecutrho       = 480.0
    occupations   = 'smearing'
    smearing      = 'mv'
    degauss       = 0.01
    vdw_corr      = 'dft-d3'
    dftd3_version = 4
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
  Mo  95.94  Mo.pbe-spn-kjpaw_psl.1.0.0.UPF
  S   32.06  S.pbe-n-kjpaw_psl.1.0.0.UPF

ATOMIC_POSITIONS {{crystal}}
  Mo  0.333333  0.666667  0.250000
  Mo  0.666667  0.333333  0.750000
  S   0.333333  0.666667  0.621000
  S   0.333333  0.666667  0.879000
  S   0.666667  0.333333  0.121000
  S   0.666667  0.333333  0.379000

K_POINTS {{automatic}}
  12 12 4 0 0 0
"""

with open(os.path.join(WORK_DIR, "mos2_d3.in"), "w") as f:
    f.write(mos2_d3)

print("MoS2 vdW benchmark input written.")
print("Expected results:")
print("  PBE (no vdW): c ~ 14.5 A (d_inter ~ 7.25 A, +18% error)")
print("  DFT-D3(BJ):   c ~ 12.3 A (d_inter ~ 6.15 A, within 1%)")
print("  rVV10:         c ~ 12.2 A (d_inter ~ 6.10 A, within 1%)")
print("  Experiment:    c = 12.295 A (d_inter = 6.148 A)")
```

## Key Parameters

| Parameter | QE | VASP | Notes |
|-----------|-----|------|-------|
| D3(BJ) correction | `vdw_corr='dft-d3', dftd3_version=4` | `IVDW=12` | Recommended pairwise method |
| D3 zero-damping | `vdw_corr='dft-d3'` (default v=2) | `IVDW=11` | Less accurate than BJ |
| D2 correction | `vdw_corr='grimme-d2'` | `IVDW=1` | Outdated; use D3 instead |
| TS-vdW | `vdw_corr='ts-vdw'` | `IVDW=2` | Good for molecule-surface |
| vdW-DF | `input_dft='vdw-df'` | `LUSE_VDW=.TRUE.` | Non-local functional |
| rVV10 | `input_dft='rvv10'` | `LUSE_VDW=.TRUE.` + `GGA=MK` + params | Best non-local for solids |
| optB86b-vdW | `input_dft='vdw-df-obk8'` | `GGA=OB, LUSE_VDW=.TRUE., AGGAC=0` | Best for layered materials |
| K-points (stacking) | Denser along stacking axis | Same | e.g., 12x12x4 for graphite |
| ecutwfc | Same as without vdW for DFT-D; may need increase for vdW-DF | Same | Check convergence |
| `ts_vdw_isolated` | `.false.` for periodic | N/A | Periodic boundary correction |

### Method Selection Guide

| System Type | Recommended Methods | Avoid |
|---|---|---|
| Layered materials (graphite, MoS2, h-BN) | rVV10, optB86b-vdW, DFT-D3(BJ) | Plain PBE |
| Molecular crystals | DFT-D3(BJ), rVV10 | DFT-D2 (overbinds) |
| Molecule on metal surface | TS-vdW, DFT-D3(BJ) | vdW-DF (overestimates distance) |
| MOFs / porous materials | DFT-D3(BJ), rVV10 | DFT-D2 |
| Noble gas crystals | rVV10, DFT-D3(BJ) | vdW-DF |
| MLIP training data consistency | DFT-D3(BJ) (most common in datasets) | Mixed methods |
| High accuracy needed | optB86b-vdW, rev-vdW-DF2 | DFT-D2, plain vdW-DF |

## Interpreting Results

### What to Compare

1. **Interlayer / intermolecular distance**: The most sensitive structural parameter. PBE without vdW will dramatically overestimate this (or find no binding at all).

2. **Binding energy**: E_bind = (E_bulk - n * E_layer) / n. Should be negative (bound). Typical values:
   - Graphite: ~52 meV/atom
   - h-BN: ~46 meV/atom
   - MoS2: ~76 meV/atom (per formula unit pair)
   - Black phosphorus: ~60 meV/atom

3. **In-plane parameters**: Should NOT change much with vdW corrections. If they change significantly, something may be wrong.

4. **Bulk modulus along c-axis (C33)**: vdW corrections increase c-axis stiffness substantially.

### Expected Improvements with vdW

| Property | Without vdW | With vdW | Typical Improvement |
|----------|-------------|----------|---------------------|
| Interlayer distance | 20-30% too large (or unbound) | Within 1-3% of experiment | Major |
| Binding energy | ~0 (unbound) | Correct order of magnitude | Critical |
| c/a ratio | Wildly wrong | Within 1-3% | Major |
| In-plane lattice param | Already good (~0.5%) | Slightly improved | Minor |
| Elastic constants (C33) | Severely underestimated | Reasonable | Major |
| Phonons (out-of-plane) | Imaginary or far too soft | Correct frequency range | Major |

### Comparing Methods

| Method | Interlayer d Error | Binding E Error | Cost | Recommendation |
|--------|-------------------|-----------------|------|----------------|
| PBE (no vdW) | +20-30% | ~0 (unbound) | Baseline | Never for layered |
| DFT-D2 | -3 to -5% | Overbinds 10-30% | Negligible | Avoid |
| DFT-D3(BJ) | Within 1-2% | Within 10% | Negligible | Good default |
| TS-vdW | Within 2-3% | Within 15% | Low | Good for surfaces |
| vdW-DF | +3-5% | Underbinds 20-30% | Moderate | Avoid |
| vdW-DF2 | Within 2% | Within 10% | Moderate | Good |
| rev-vdW-DF2 | Within 1-2% | Within 5-10% | Moderate | Very good |
| optB86b-vdW | Within 1% | Within 5% | Moderate | Best for layered |
| rVV10 | Within 1-2% | Within 5-10% | Moderate | Best overall |

## Common Issues

| Issue | Symptom | Solution |
|-------|---------|----------|
| PBE gives no interlayer binding | Layers drift apart in vc-relax; c diverges | Expected -- PBE lacks dispersion. You MUST use a vdW correction for layered materials. |
| DFT-D2 overbinding | Interlayer distance 3-5% too small | Use DFT-D3(BJ) or rVV10 instead; D2 uses fixed C6 coefficients |
| vdW-DF overestimates distances | d is 3-5% too large | Use rev-vdW-DF2 or optB86b-vdW instead of original vdW-DF |
| `input_dft` PP warning | QE warns about functional mismatch | Expected when using PBE pseudopotentials with vdW-DF functionals; acceptable for testing, use correct PPs for publication |
| DFT-D3 not compiled | Error: "DFT-D3 not available" | Recompile QE with `-D__DFTD3` and linked libdftd3; or use DFT-D2 / vdW-DF family |
| Convergence with vdW-DF | Slow SCF convergence | Reduce `mixing_beta` to 0.2-0.3; non-local kernel can cause oscillations |
| Memory with vdW-DF | Large memory usage | Non-local kernel requires extra memory; DFT-D methods have negligible overhead |
| Inconsistent cross-method results | 10% variation in binding energies between methods | Inherent to approximations; pick ONE method and use consistently throughout a study |
| K-point convergence | Results depend on k-grid | Converge carefully along stacking direction; use at least 4 k-points along c |
| VASP `vdw_kernel.bindat` missing | Error when `LUSE_VDW=.TRUE.` | Generate with `vdw_kernel.x` or download from VASP wiki; place in run directory |
| Negative frequencies in phonons | Acoustic modes imaginary without vdW | PBE has no restoring force for interlayer sliding; vdW correction is required |
