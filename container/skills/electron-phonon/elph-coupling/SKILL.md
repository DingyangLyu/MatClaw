# Electron-Phonon Coupling

## When to Use

- You need the electron-phonon coupling constant **lambda** for a metallic system.
- You want to compute the **Eliashberg spectral function alpha2F(omega)** to understand which phonon modes contribute most to electron-phonon coupling.
- You want to estimate the **superconducting critical temperature Tc** using the McMillan or Allen-Dynes formula.
- You are studying conventional (phonon-mediated) superconductors.
- You need to understand how strongly electrons scatter off phonons (relevant for resistivity, mass renormalization).

This workflow applies to **metals and metallic compounds only**. The system must have a Fermi surface. For semiconductors, electron-phonon coupling requires different approaches (deformation potentials, Frohlich coupling).

## Prerequisites

- Quantum ESPRESSO 7.5 installed (`pw.x`, `ph.x`, `q2r.x`, `matdyn.x`, `lambda.x`, `alpha2f.x`)
- Appropriate pseudopotentials (SSSP Efficiency or PSlibrary recommended; ultrasoft or PAW)
- A well-converged SCF calculation for the target material
- Understanding that this is one of the most computationally demanding standard DFT workflows

## Detailed Steps

The complete workflow is:

```
SCF (pw.x) --> Phonons + el-ph coefficients (ph.x) --> lambda.x --> alpha2f.x
   |                    |                                  |            |
 coarse k     uniform q-grid on                        lambda,Tc   alpha2F(omega)
              dense k-grid (via interpolated)
```

### Overview of the Physics

The electron-phonon coupling constant is defined as:

```
lambda = 2 * integral[ alpha2F(omega) / omega ] d(omega)
```

where alpha2F(omega) is the Eliashberg spectral function, which encodes how strongly electrons at the Fermi surface couple to phonons of frequency omega. The McMillan/Allen-Dynes formula then estimates Tc:

```
Tc = (omega_log / 1.2) * exp[ -1.04*(1+lambda) / (lambda - mu_star*(1+0.62*lambda)) ]
```

where mu_star (typically 0.10-0.16) is the Coulomb pseudopotential describing screened electron-electron repulsion.

### Step 1: SCF Calculation (pw.x)

The SCF step establishes the ground-state charge density. For electron-phonon calculations, use a **coarse** k-grid here -- the dense k-grid is handled internally by `ph.x` when `electron_phonon='interpolated'`.

Create `scf.in`:

```
cat > scf.in << 'PWSCF_INPUT'
&CONTROL
    calculation   = 'scf'
    prefix        = 'mgb2'
    pseudo_dir    = './pseudo/'
    outdir        = './tmp/'
    tprnfor       = .true.
    tstress       = .true.
    verbosity     = 'high'
/
&SYSTEM
    ibrav         = 4
    celldm(1)     = 5.8261
    celldm(3)     = 1.1421
    nat           = 3
    ntyp          = 2
    ecutwfc       = 60.0
    ecutrho       = 480.0
    occupations   = 'smearing'
    smearing      = 'marzari-vanderbilt'
    degauss       = 0.02
    la2F          = .true.
/
&ELECTRONS
    conv_thr      = 1.0d-12
    mixing_beta   = 0.7
/
ATOMIC_SPECIES
  Mg  24.305   Mg.pbe-spnl-kjpaw_psl.1.0.0.UPF
  B   10.811   B.pbe-n-kjpaw_psl.1.0.0.UPF

ATOMIC_POSITIONS {crystal}
  Mg  0.000000  0.000000  0.000000
  B   0.333333  0.666667  0.500000
  B   0.666667  0.333333  0.500000

K_POINTS {automatic}
  12 12 12  0 0 0
PWSCF_INPUT
```

Key points for the SCF step:
- **`la2F = .true.`**: This flag is REQUIRED. It tells `pw.x` to produce additional output needed by `ph.x` for electron-phonon calculations.
- **`smearing = 'marzari-vanderbilt'`** (cold smearing): Best choice for metals in electron-phonon calculations. Alternatives: `'methfessel-paxton'` (order 1).
- **`degauss = 0.02`** Ry: The smearing width. Must be small enough to resolve the Fermi surface but large enough for numerical stability. Typical range: 0.005-0.05 Ry.
- **`conv_thr = 1.0d-12`**: Very tight convergence is essential because phonon calculations require highly converged charge densities.
- **k-grid 12x12x12**: This is the "coarse" grid. The dense grid for Fermi-surface integration is specified separately in `ph.x`.

Run:

```bash
mpirun -np 4 pw.x -npool 2 < scf.in > scf.out 2>&1
```

Verify convergence:

```bash
grep "convergence has been achieved" scf.out
grep "total energy" scf.out | tail -1
grep "the Fermi energy is" scf.out
```

### Step 2: Phonon Calculation with Electron-Phonon Coupling (ph.x)

This is the most computationally expensive step. `ph.x` computes phonons on a uniform q-grid and simultaneously evaluates the electron-phonon matrix elements on a dense k-grid (specified by `nk1`, `nk2`, `nk3`).

Create `ph.in`:

```
cat > ph.in << 'PH_INPUT'
Electron-phonon coupling for MgB2
&INPUTPH
    prefix        = 'mgb2'
    outdir        = './tmp/'
    fildyn        = 'mgb2.dyn'
    fildvscf      = 'dvscf'
    tr2_ph        = 1.0d-16
    ldisp         = .true.
    nq1           = 4
    nq2           = 4
    nq3           = 4
    electron_phonon = 'interpolated'
    el_ph_sigma   = 0.005
    el_ph_nsigma  = 10
    nk1           = 24
    nk2           = 24
    nk3           = 24
/
PH_INPUT
```

Key parameters explained:

- **`electron_phonon = 'interpolated'`**: Uses Fourier interpolation of the electron-phonon matrix elements. This is the standard method in QE. The alternative `'simple'` is deprecated for production.
- **`nq1, nq2, nq3 = 4, 4, 4`**: The q-point grid for phonon sampling. Start with 4x4x4, converge up to 6x6x6 or 8x8x8. Each q-point is a separate DFPT calculation.
- **`nk1, nk2, nk3 = 24, 24, 24`**: The DENSE k-point grid for Fermi surface integration of electron-phonon matrix elements. This must be significantly denser than the SCF k-grid. Converge from 24x24x24 up to 40x40x40.
- **`el_ph_sigma = 0.005`**: Gaussian broadening (Ry) for the double-delta function at the Fermi surface. This replaces the delta functions with Gaussians of this width. Typical range: 0.002-0.02 Ry.
- **`el_ph_nsigma = 10`**: Number of different sigma values to try (from `el_ph_sigma` to `el_ph_nsigma * el_ph_sigma`). This lets you check convergence with respect to broadening without re-running.
- **`tr2_ph = 1.0d-16`**: Very tight self-consistency threshold for the phonon calculation. Needed for reliable electron-phonon matrix elements.
- **`fildvscf = 'dvscf'`**: Saves the self-consistent change in potential, needed for electron-phonon matrix elements.

Run (this is the expensive step):

```bash
mpirun -np 8 ph.x -npool 2 < ph.in > ph.out 2>&1
```

This produces:
- `mgb2.dyn1`, `mgb2.dyn2`, ... (dynamical matrices at each q-point)
- `elph_dir/elph.inp_lambda.{q_index}` files containing the electron-phonon coupling data
- `a2Fq2r.{sigma_index}` files (alpha2F contributions per q-point)

Verify completion:

```bash
grep "JOB DONE" ph.out
# Check phonon frequencies at each q-point:
grep "freq" ph.out | head -30
# Verify no negative frequencies (except near Gamma for acoustic modes):
grep "omega" ph.out
```

### Step 3: Compute lambda and Tc with lambda.x

`lambda.x` reads the electron-phonon data from `ph.x` and computes the coupling constant lambda, the logarithmic average frequency omega_log, and Tc.

First, collect the `elph.inp_lambda.*` files. The input to `lambda.x` has a special format:

Create `lambda.in`:

```python
#!/usr/bin/env python3
"""
Generate input file for lambda.x from ph.x electron-phonon output.
Run this script AFTER ph.x completes.
"""
import os
import glob
import re

# Configuration
sigma_index = 1         # Which el_ph_sigma value to use (1 = smallest)
mu_star = 0.12          # Coulomb pseudopotential (0.10-0.16 typical)
prefix = 'mgb2'
elph_dir = './elph_dir'

# Find all elph.inp_lambda files
elph_files = sorted(glob.glob(os.path.join(elph_dir, f'elph.inp_lambda.*')))
nq = len(elph_files)

if nq == 0:
    # Try alternative location (some QE versions put files in outdir)
    elph_files = sorted(glob.glob(os.path.join('./tmp/', f'elph.inp_lambda.*')))
    nq = len(elph_files)
    elph_dir = './tmp/'

if nq == 0:
    raise FileNotFoundError("No elph.inp_lambda.* files found. Check ph.x output.")

print(f"Found {nq} q-points with electron-phonon data.")

# Read the first file to get number of frequencies (modes)
with open(elph_files[0], 'r') as f:
    lines = f.readlines()

# First line typically has: nq_point, number_of_modes, ef(Ry)
header = lines[0].split()

# Generate lambda.x input
# Format:
# Line 1: number of frequencies (3*nat)
# Line 2: list of elph files
# Line 3: mu_star
# ...
# The exact format depends on QE version. For QE 7.x:

with open('lambda.in', 'w') as f:
    f.write(f"10  {mu_star}  1\n")  # el_ph_nsigma, mu_star, sigma_index_to_use
    # Number of q-points
    f.write(f"{nq}\n")
    # List each elph file with its weight
    for elph_file in elph_files:
        f.write(f"{elph_file}\n")

print("Generated lambda.in")
print(f"Using mu_star = {mu_star}")
print(f"Using sigma index = {sigma_index}")
```

However, the standard approach for QE 7.x is simpler. `lambda.x` reads from standard input with this format:

```
cat > lambda.in << 'LAMBDA_INPUT'
10  0.12  1
4
elph_dir/elph.inp_lambda.1
elph_dir/elph.inp_lambda.2
elph_dir/elph.inp_lambda.3
elph_dir/elph.inp_lambda.4
LAMBDA_INPUT
```

Format explanation:
- Line 1: `el_ph_nsigma` (number of smearing values), `mu_star` (Coulomb pseudopotential), `sigma_index` (which smearing to select for final output, but all are computed)
- Line 2: number of q-points
- Lines 3+: paths to elph.inp_lambda files (one per q-point)

**Important**: The number of q-points listed here must match the number of irreducible q-points from your `nq1 x nq2 x nq3` grid, and each file must exist.

Use this Python script to auto-generate `lambda.in` robustly:

```python
#!/usr/bin/env python3
"""
generate_lambda_input.py
Automatically generate lambda.x input by finding elph files.
"""
import glob
import os
import sys

# Parameters
el_ph_nsigma = 10      # Must match ph.x input
mu_star = 0.12          # Coulomb pseudopotential
sigma_to_report = 1     # Which sigma to print Tc for (1-based)

# Locate elph files -- try several common locations
search_paths = [
    'elph_dir/elph.inp_lambda.*',
    'tmp/_ph0/elph_dir/elph.inp_lambda.*',
    'tmp/elph_dir/elph.inp_lambda.*',
]

elph_files = []
for pattern in search_paths:
    elph_files = sorted(glob.glob(pattern))
    if elph_files:
        print(f"Found elph files matching: {pattern}")
        break

if not elph_files:
    print("ERROR: No elph.inp_lambda.* files found!")
    print("Searched in:", search_paths)
    sys.exit(1)

nq = len(elph_files)
print(f"Number of irreducible q-points: {nq}")

with open('lambda.in', 'w') as f:
    f.write(f"{el_ph_nsigma}  {mu_star}  {sigma_to_report}\n")
    f.write(f"{nq}\n")
    for filepath in elph_files:
        f.write(f"{filepath}\n")

print("Wrote lambda.in")
```

Run lambda.x:

```bash
python3 generate_lambda_input.py
lambda.x < lambda.in > lambda.out 2>&1
```

Inspect output:

```bash
# lambda.out contains lambda(sigma) for each sigma value
# and Tc estimates
cat lambda.out
```

### Step 4: Eliashberg Spectral Function with alpha2f.x

After `lambda.x`, compute the full alpha2F(omega) spectral function:

```bash
# alpha2f.x uses the same a2Fq2r.* files produced by ph.x
# It reads from standard input:
cat > a2f.in << 'A2F_INPUT'
&INPUTA2F
    nfreq = 500
/
A2F_INPUT

alpha2f.x < a2f.in > a2f.out 2>&1
```

This produces a file (typically `a2F.dos.{sigma_index}`) with columns: omega (cm^-1), alpha2F(omega), cumulative lambda(omega).

### Step 5: Post-Processing and Plotting (Python)

```python
#!/usr/bin/env python3
"""
elph_postprocess.py
Parse lambda.x and alpha2f.x output, compute Tc, and plot results.
"""
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import re
import os
import sys

# =============================================================================
# Part 1: Parse lambda.x output
# =============================================================================

def parse_lambda_output(filename='lambda.out'):
    """
    Parse lambda.x output to extract lambda, omega_log, and Tc
    for each smearing value.
    """
    results = []

    with open(filename, 'r') as f:
        content = f.read()

    # Look for lines containing lambda and Tc information
    # Format varies by QE version. Common patterns:

    # Pattern: "lambda = X.XXXX   omega_log = X.XXXX (K)   Tc = X.XXXX (K)"
    pattern1 = r'lambda\s*=\s*([\d.]+)\s+omega_log\s*=\s*([\d.]+)\s*\(K\)\s+.*?Tc\s*=\s*([\d.]+)\s*\(K\)'
    matches = re.findall(pattern1, content)

    if matches:
        for m in matches:
            results.append({
                'lambda': float(m[0]),
                'omega_log_K': float(m[1]),
                'Tc_K': float(m[2])
            })
        return results

    # Alternative pattern for different QE output format
    # Parse line by line
    lines = content.split('\n')
    current = {}
    for line in lines:
        if 'lambda' in line.lower() and '=' in line:
            parts = line.split()
            for i, p in enumerate(parts):
                if p == '=' and i > 0:
                    try:
                        key = parts[i-1].lower().strip('()')
                        val = float(parts[i+1])
                        current[key] = val
                    except (ValueError, IndexError):
                        pass
        if 'Tc' in line or 'tc' in line.lower():
            tc_match = re.search(r'Tc\s*=\s*([\d.]+)', line, re.IGNORECASE)
            if tc_match:
                current['Tc_K'] = float(tc_match.group(1))

        # If we have a complete set, save it
        if 'lambda' in current and 'Tc_K' in current:
            results.append(current.copy())
            current = {}

    # Fallback: try reading entire file as structured data
    if not results:
        print(f"WARNING: Could not parse {filename} automatically.")
        print("Printing raw content for manual inspection:")
        print(content[:2000])

    return results


# =============================================================================
# Part 2: Parse alpha2F spectral function
# =============================================================================

def load_a2f(filename=None):
    """
    Load alpha2F data from alpha2f.x output file.
    Tries several common filenames.
    Returns: omega (cm^-1), a2f, cumulative_lambda
    """
    if filename is None:
        candidates = [
            'a2F.dos.1', 'a2F.dos1', 'a2F.dos.01',
            'alpha2F.dat', 'a2F.dat',
        ]
        # Also try numbered files
        for i in range(1, 20):
            candidates.append(f'a2F.dos.{i}')

        for fn in candidates:
            if os.path.isfile(fn):
                filename = fn
                print(f"Found alpha2F file: {filename}")
                break

    if filename is None or not os.path.isfile(filename):
        print("ERROR: No alpha2F data file found.")
        return None, None, None

    data = np.loadtxt(filename, comments='#')

    # Typical columns: omega(cm^-1), a2F(omega), [lambda(omega), ...]
    omega = data[:, 0]
    a2f = data[:, 1]

    # Compute cumulative lambda from alpha2F
    # lambda = 2 * integral[ alpha2F(omega) / omega ] d_omega
    # Avoid division by zero at omega=0
    domega = np.diff(omega, prepend=omega[0] - (omega[1]-omega[0]))
    integrand = np.where(omega > 1e-6, 2.0 * a2f / omega, 0.0)
    cumulative_lambda = np.cumsum(integrand * domega)

    return omega, a2f, cumulative_lambda


# =============================================================================
# Part 3: McMillan / Allen-Dynes Tc formula
# =============================================================================

def compute_tc_allen_dynes(lam, omega_log_K, mu_star=0.12):
    """
    Allen-Dynes formula for superconducting Tc.

    Parameters
    ----------
    lam : float
        Electron-phonon coupling constant lambda.
    omega_log_K : float
        Logarithmic average phonon frequency in Kelvin.
    mu_star : float
        Coulomb pseudopotential (typically 0.10-0.16).

    Returns
    -------
    Tc : float
        Superconducting critical temperature in Kelvin.
    """
    if lam <= 0 or omega_log_K <= 0:
        return 0.0

    denom = lam - mu_star * (1.0 + 0.62 * lam)
    if denom <= 0:
        return 0.0

    exponent = -1.04 * (1.0 + lam) / denom
    Tc = (omega_log_K / 1.2) * np.exp(exponent)
    return Tc


def compute_omega_log(omega_cm, a2f):
    """
    Compute logarithmic average frequency from alpha2F.

    omega_log = exp[ (2/lambda) * integral( alpha2F(w)/w * ln(w) dw ) ]

    Parameters
    ----------
    omega_cm : array
        Frequencies in cm^-1.
    a2f : array
        Eliashberg spectral function.

    Returns
    -------
    omega_log_K : float
        Logarithmic average frequency in Kelvin.
    """
    # Convert cm^-1 to meV for intermediate calculation
    # 1 cm^-1 = 0.12398 meV
    # Actually, keep in cm^-1 and convert at end

    mask = omega_cm > 1.0  # avoid log(0) and division by zero
    w = omega_cm[mask]
    a = a2f[mask]
    dw = np.gradient(w)

    # lambda
    lam = 2.0 * np.trapz(a / w, w)

    if lam <= 0:
        return 0.0

    # omega_log in cm^-1
    log_avg = (2.0 / lam) * np.trapz((a / w) * np.log(w), w)
    omega_log_cm = np.exp(log_avg)

    # Convert cm^-1 to Kelvin: 1 cm^-1 = 1.4388 K
    omega_log_K = omega_log_cm * 1.4388

    return omega_log_K


# =============================================================================
# Part 4: Plotting
# =============================================================================

def plot_a2f(omega_cm, a2f, cumulative_lambda, output='alpha2F_plot.png'):
    """
    Plot Eliashberg spectral function and cumulative lambda.
    """
    fig, ax1 = plt.subplots(figsize=(8, 5))

    ax1.fill_between(omega_cm, 0, a2f, alpha=0.3, color='steelblue')
    ax1.plot(omega_cm, a2f, color='steelblue', linewidth=1.5, label=r'$\alpha^2F(\omega)$')
    ax1.set_xlabel(r'Frequency $\omega$ (cm$^{-1}$)', fontsize=13)
    ax1.set_ylabel(r'$\alpha^2F(\omega)$', fontsize=13, color='steelblue')
    ax1.tick_params(axis='y', labelcolor='steelblue')
    ax1.set_xlim(left=0)
    ax1.set_ylim(bottom=0)

    # Cumulative lambda on right axis
    ax2 = ax1.twinx()
    ax2.plot(omega_cm, cumulative_lambda, color='crimson', linewidth=2.0,
             linestyle='--', label=r'$\lambda(\omega)$')
    ax2.set_ylabel(r'Cumulative $\lambda(\omega)$', fontsize=13, color='crimson')
    ax2.tick_params(axis='y', labelcolor='crimson')
    ax2.set_ylim(bottom=0)

    # Combined legend
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left', fontsize=11)

    total_lambda = cumulative_lambda[-1] if len(cumulative_lambda) > 0 else 0
    ax1.set_title(r'Eliashberg Spectral Function ($\lambda$ = ' + f'{total_lambda:.3f})',
                  fontsize=14)

    plt.tight_layout()
    plt.savefig(output, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"Saved plot: {output}")


def plot_lambda_convergence(sigma_values, lambda_values, output='lambda_convergence.png'):
    """
    Plot lambda vs smearing sigma to check convergence.
    """
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(sigma_values, lambda_values, 'o-', color='steelblue', linewidth=2, markersize=8)
    ax.set_xlabel(r'Gaussian broadening $\sigma$ (Ry)', fontsize=13)
    ax.set_ylabel(r'$\lambda$', fontsize=13)
    ax.set_title(r'Convergence of $\lambda$ with broadening', fontsize=14)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"Saved plot: {output}")


# =============================================================================
# Part 5: Main workflow
# =============================================================================

def main():
    print("=" * 60)
    print("Electron-Phonon Coupling Post-Processing")
    print("=" * 60)

    # --- Parse lambda.x output ---
    print("\n--- Parsing lambda.x output ---")
    if os.path.isfile('lambda.out'):
        results = parse_lambda_output('lambda.out')
        if results:
            print(f"\nFound {len(results)} smearing entries:")
            print(f"{'sigma_idx':>10} {'lambda':>10} {'omega_log(K)':>12} {'Tc(K)':>8}")
            print("-" * 45)
            for i, r in enumerate(results):
                lam = r.get('lambda', 0)
                olog = r.get('omega_log_K', r.get('omega_log', 0))
                tc = r.get('Tc_K', r.get('tc', 0))
                print(f"{i+1:>10} {lam:>10.4f} {olog:>12.2f} {tc:>8.3f}")
        else:
            print("Could not automatically parse lambda.out.")
    else:
        print("lambda.out not found. Skipping.")

    # --- Load and plot alpha2F ---
    print("\n--- Processing alpha2F spectral function ---")
    omega, a2f, cum_lambda = load_a2f()

    if omega is not None:
        # Compute derived quantities
        total_lambda = cum_lambda[-1]
        omega_log_K = compute_omega_log(omega, a2f)

        print(f"\nFrom alpha2F integration:")
        print(f"  lambda      = {total_lambda:.4f}")
        print(f"  omega_log   = {omega_log_K:.2f} K")

        # Compute Tc for several mu_star values
        print(f"\n  {'mu_star':>8}  {'Tc (K)':>8}")
        print("  " + "-" * 20)
        for mu_s in [0.10, 0.12, 0.13, 0.15]:
            Tc = compute_tc_allen_dynes(total_lambda, omega_log_K, mu_s)
            print(f"  {mu_s:>8.2f}  {Tc:>8.3f}")

        # Plot
        plot_a2f(omega, a2f, cum_lambda)
    else:
        print("No alpha2F data available.")

    print("\n" + "=" * 60)
    print("Post-processing complete.")


if __name__ == '__main__':
    main()
```

Run the post-processing:

```bash
python3 elph_postprocess.py
```

### Complete Automated Workflow Script

This script runs the entire workflow from start to finish:

```bash
#!/bin/bash
# run_elph.sh -- Complete electron-phonon coupling workflow
# Usage: bash run_elph.sh
# Adjust NP, NPOOL, and input parameters below.

set -e

NP=8          # Number of MPI processes
NPOOL=2       # Number of k-point pools (must divide NP)
PREFIX="mgb2"

echo "========================================"
echo "Electron-Phonon Coupling Workflow"
echo "========================================"

# Step 1: SCF
echo ""
echo "--- Step 1: SCF calculation ---"
mpirun -np $NP pw.x -npool $NPOOL < scf.in > scf.out 2>&1

if grep -q "convergence has been achieved" scf.out; then
    echo "SCF converged."
    grep "the Fermi energy is" scf.out
else
    echo "ERROR: SCF did not converge!"
    exit 1
fi

# Step 2: Phonons + electron-phonon
echo ""
echo "--- Step 2: Phonon + el-ph calculation ---"
mpirun -np $NP ph.x -npool $NPOOL < ph.in > ph.out 2>&1

if grep -q "JOB DONE" ph.out; then
    echo "Phonon calculation completed."
else
    echo "ERROR: Phonon calculation failed!"
    exit 1
fi

# Step 3: lambda.x
echo ""
echo "--- Step 3: Computing lambda ---"
python3 generate_lambda_input.py
lambda.x < lambda.in > lambda.out 2>&1
echo "lambda.x completed."

# Step 4: alpha2F (if alpha2f.x is available)
echo ""
echo "--- Step 4: alpha2F spectral function ---"
if command -v alpha2f.x &> /dev/null; then
    cat > a2f.in << 'EOF'
&INPUTA2F
    nfreq = 500
/
EOF
    alpha2f.x < a2f.in > a2f.out 2>&1
    echo "alpha2f.x completed."
else
    echo "alpha2f.x not found, skipping."
fi

# Step 5: Post-processing
echo ""
echo "--- Step 5: Post-processing ---"
python3 elph_postprocess.py

echo ""
echo "========================================"
echo "Workflow complete!"
echo "========================================"
```

## Key Parameters

### Convergence Parameters (in order of importance)

| Parameter | Where | Typical Range | Impact |
|---|---|---|---|
| `nk1,nk2,nk3` (in ph.in) | Dense k-grid | 24-40 per direction | Fermi surface sampling. Most critical parameter. Must converge lambda to within ~5%. |
| `nq1,nq2,nq3` (in ph.in) | q-point grid | 4-8 per direction | Phonon momentum sampling. 4x4x4 is a minimum; 6x6x6 usually sufficient. |
| `el_ph_sigma` | ph.in | 0.002-0.02 Ry | Double-delta broadening. Too small: noisy. Too large: smears out features. |
| `degauss` | scf.in | 0.005-0.05 Ry | Electronic smearing for Fermi-Dirac occupation. Independent of el_ph_sigma. |
| `ecutwfc` | scf.in | 40-80 Ry | Plane-wave cutoff. Must be converged as in any DFT calculation. |
| `mu_star` | lambda.in | 0.10-0.16 | Coulomb pseudopotential. Not a convergence parameter -- it is empirical. 0.10 for simple metals, 0.13-0.15 for transition metals. |

### Convergence Strategy

1. **Start cheap**: `nq=2x2x2`, `nk=12x12x12`, `el_ph_sigma=0.01`
2. **Increase q-grid**: `nq=4x4x4`, check lambda changes.
3. **Increase k-grid**: `nk=24x24x24`, then `32x32x32`, then `40x40x40`. Lambda should plateau.
4. **Vary el_ph_sigma**: Use `el_ph_nsigma=10` to automatically scan. Lambda should be stable over a range of sigma.
5. **Final production**: Use the smallest grids where lambda is converged to ~5%.

Typical converged values for simple systems:
- Simple metals (Al, Pb): `nq=6x6x6`, `nk=32x32x32`
- MgB2: `nq=4x4x4`, `nk=24x24x24` (minimum); `nq=6x6x6`, `nk=36x36x36` (production)
- Complex unit cells: reduce grids proportionally with cell size

## Interpreting Results

### Lambda Values

| Lambda Range | Physical Meaning | Examples |
|---|---|---|
| < 0.3 | Weak coupling, likely not superconducting or very low Tc | Noble metals (Cu, Ag, Au) |
| 0.3 - 0.5 | Moderate coupling | Al (0.43), Sn (0.60) |
| 0.5 - 1.0 | Intermediate coupling | Nb (0.82), MgB2 sigma-band (~0.8) |
| 1.0 - 1.5 | Strong coupling | Pb (1.55) |
| > 1.5 | Very strong coupling, Allen-Dynes may underestimate Tc | H3S, LaH10 |

### Alpha2F Spectral Function

- **Peaks** in alpha2F(omega) correspond to phonon modes that couple strongly to electrons.
- **Low-frequency peaks** contribute more to lambda (because of the 1/omega weighting).
- **Comparison with phonon DOS**: If alpha2F closely tracks the phonon DOS, coupling is roughly uniform across modes. If peaks are selectively enhanced, specific modes dominate.
- For MgB2: the E2g boron stretching mode (~550 cm^-1) shows a huge peak in alpha2F, explaining its high Tc.

### Tc Estimates

- McMillan/Allen-Dynes formula is reliable for **lambda < 1.5**.
- For strong coupling (lambda > 1.5), solve the full Eliashberg equations instead.
- **mu_star** introduces ~30% uncertainty in Tc. Report Tc for a range of mu_star values.
- Compare with experiment when available to calibrate mu_star.

### Sanity Checks

1. **Phonon frequencies**: All should be positive (imaginary frequencies indicate structural instability).
2. **Lambda consistency**: Lambda from lambda.x should match the integral of alpha2F.
3. **Convergence plateau**: Lambda should plateau when increasing nk and nq grids.
4. **Physical range**: Lambda > 3 is suspect for conventional superconductors (except hydrides under pressure).

## Common Issues

### 1. Negative (Imaginary) Phonon Frequencies

**Symptom**: `ph.x` reports `omega < 0` at some q-points (shown as negative values in the output).

**Causes and fixes**:
- Structure not fully relaxed: Relax with tight force thresholds (`forc_conv_thr = 1.0d-5`).
- Insufficient k-grid in SCF: Increase the SCF k-grid.
- Physical instability: The structure may genuinely be dynamically unstable at this volume/pressure.
- Insufficient ecutwfc: Increase cutoff.

### 2. Lambda Does Not Converge with k-Grid

**Symptom**: Lambda changes significantly (>10%) when increasing `nk1,nk2,nk3`.

**Fixes**:
- Keep increasing the k-grid. For systems with complex Fermi surfaces, 40x40x40 or more may be needed.
- Try different `el_ph_sigma` values. If lambda converges for some sigma but not others, choose the sigma where the plateau is broadest.
- Use `smearing = 'marzari-vanderbilt'` (cold smearing) which converges faster than Fermi-Dirac.

### 3. Lambda is Unphysically Large

**Symptom**: Lambda > 5 or fluctuating wildly.

**Causes**:
- `el_ph_sigma` too small: The Gaussian broadening is too narrow, causing numerical noise. Increase to 0.005-0.01 Ry.
- Soft phonon modes near instability: A mode with very low frequency and moderate coupling gives huge lambda (1/omega weighting). Check if the mode is physical.
- Incorrect `la2F = .true.` or missing flag: Ensure it is set in the SCF input.

### 4. ph.x Crashes or Runs Forever

**Causes**:
- Memory: Electron-phonon with dense k-grids requires significant memory. Try more MPI processes with fewer pools, or reduce `nk` temporarily.
- Disk space: `fildvscf` files can be very large. Ensure sufficient disk.
- SCF convergence in ph.x: Tighten `tr2_ph` (e.g., `1.0d-18`) or adjust `alpha_mix`.

### 5. Cannot Find elph Files

**Symptom**: `lambda.x` or the post-processing script cannot find `elph.inp_lambda.*` files.

**Fixes**:
- Check `outdir` from your SCF/phonon calculation. Files may be in `./tmp/_ph0/elph_dir/`.
- Ensure `electron_phonon = 'interpolated'` was set in `ph.in`.
- Ensure `la2F = .true.` was set in `scf.in`.

### 6. Discrepancy Between lambda.x and Manual Integration of alpha2F

**Typical cause**: Different broadening parameters or frequency grids. Small differences (< 5%) are normal due to numerical integration. Larger differences suggest an issue with the alpha2F file or integration limits.

### 7. Pseudopotential Issues

- Use the **same** pseudopotential library consistently.
- Norm-conserving pseudopotentials are compatible but often need higher ecutwfc.
- PAW/ultrasoft are more efficient. Ensure the PP supports phonon calculations (most do).
- ONCV (SG15) pseudopotentials work well for phonons and electron-phonon.

### 8. Restart After Crash

`ph.x` supports restart via `recover = .true.` in the `&INPUTPH` namelist. Add this flag if a long calculation might be interrupted:

```
&INPUTPH
    ...
    recover = .true.
/
```

Already-completed q-points will be skipped on restart.
