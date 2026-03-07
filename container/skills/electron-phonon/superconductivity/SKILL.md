# Superconducting Properties from Electron-Phonon Coupling

## When to Use

- You want to compute the **Eliashberg spectral function alpha2F(omega)** to understand which phonon modes drive superconductivity.
- You need the **electron-phonon coupling constant lambda** resolved by phonon mode and q-point.
- You want to estimate the **superconducting critical temperature Tc** via the McMillan or Allen-Dynes formula.
- You are studying **conventional (phonon-mediated) superconductors** and need publication-quality alpha2F spectra.
- You want to explore how **pressure, doping, or strain** affect Tc by repeating the workflow at modified structures.
- You need **omega_log** (logarithmic average phonon frequency) and **omega_2** (second moment) for advanced Tc corrections.

This workflow is for **metals and metallic compounds** with a Fermi surface. Conventional superconductivity is mediated by electron-phonon coupling; unconventional (e.g., cuprate, iron-based) superconductors require different methods not covered here.

## Method Selection

| Scenario | Recommended Approach | Notes |
|---|---|---|
| Simple metal (Al, Pb, Nb, Sn) | Full QE el-ph workflow below | Well-tested, reliable |
| Binary compound (MgB2, NbC, NbN) | Full QE el-ph workflow | May need denser grids for multi-sheet Fermi surfaces |
| Hydride under pressure (H3S, LaH10) | Full QE el-ph + anharmonic corrections | Allen-Dynes may underestimate Tc for lambda > 1.5; consider full Eliashberg |
| High-Tc candidate screening | Coarse grids first, refine promising candidates | Start with 2x2x2 q-grid, 12x12x12 k-grid |
| Tc under pressure/strain | Repeat workflow at each volume/strain point | Relax internal coordinates at each pressure |
| Alloy or disordered system | Virtual crystal approximation (VCA) in QE, or supercell | Supercell approach is expensive for el-ph |
| 2D material | Slab geometry with vacuum | Use `assume_isolated = '2D'` in QE 7.x |

## Prerequisites

- Quantum ESPRESSO 7.5 installed: `pw.x`, `ph.x`, `q2r.x`, `matdyn.x`, `lambda.x`, `alpha2f.x`
- Appropriate pseudopotentials (SSSP Efficiency, PSlibrary, or SG15 ONCV recommended)
- The target system must be metallic (nonzero density of states at the Fermi level)
- Significant computational resources: el-ph calculations scale with N_q x N_k^3 x N_modes

## Detailed Steps

The complete workflow is:

```
SCF (pw.x)
  |-- la2F = .true., metallic smearing
  v
Phonons + el-ph matrix elements (ph.x)
  |-- electron_phonon = 'interpolated'
  |-- nq1 x nq2 x nq3 uniform q-grid
  |-- nk1 x nk2 x nk3 dense k-grid for Fermi surface
  v
lambda.x
  |-- reads elph.inp_lambda.* files
  |-- computes lambda, omega_log, Tc for each broadening
  v
alpha2f.x
  |-- reads a2Fq2r.* files
  |-- outputs alpha2F(omega) spectral function
  v
Python post-processing
  |-- parse lambda.out, alpha2F data
  |-- compute Tc with Allen-Dynes (and strong-coupling corrections)
  |-- plot alpha2F(omega), cumulative lambda(omega)
  |-- mode-resolved lambda analysis
```

### Step 1: SCF Calculation

The SCF step establishes the ground-state charge density. The key flag `la2F = .true.` is mandatory for electron-phonon workflows.

Example: MgB2 (a well-known conventional superconductor with Tc ~ 39 K).

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

Key points:
- **`la2F = .true.`**: Mandatory. Tells `pw.x` to write data needed by `ph.x` for el-ph coupling.
- **`smearing = 'marzari-vanderbilt'`**: Cold smearing, best for metals in el-ph workflows.
- **`degauss = 0.02` Ry**: Electronic smearing width. Range 0.005-0.05 Ry.
- **`conv_thr = 1.0d-12`**: Very tight convergence for phonon calculations.
- **K-grid 12x12x12**: Coarse grid for SCF. The dense k-grid for Fermi-surface integration is in `ph.x`.

Run and verify:

```bash
mpirun -np 4 pw.x -npool 2 < scf.in > scf.out 2>&1
grep "convergence has been achieved" scf.out
grep "the Fermi energy is" scf.out
grep "total energy" scf.out | tail -1
```

### Step 2: Phonon Calculation with Electron-Phonon Matrix Elements

This is the computationally dominant step. `ph.x` computes phonons on a uniform q-grid and evaluates el-ph matrix elements on a dense k-grid using the `'interpolated'` method.

```
cat > ph_elph.in << 'PH_INPUT'
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
    recover       = .true.
/
PH_INPUT
```

Parameters explained:

- **`electron_phonon = 'interpolated'`**: Fourier-interpolates el-ph matrix elements from the coarse q-grid to arbitrary q-points. This is the standard production method in QE.
- **`nq1, nq2, nq3 = 4, 4, 4`**: q-point grid for phonon momentum sampling. Start with 4x4x4; converge to 6x6x6 or 8x8x8.
- **`nk1, nk2, nk3 = 24, 24, 24`**: Dense k-grid for double-delta Fermi surface integration. This is the most critical convergence parameter. Typical production: 24-40 per direction.
- **`el_ph_sigma = 0.005`**: Gaussian broadening (Ry) for the double-delta function. Range: 0.002-0.02 Ry.
- **`el_ph_nsigma = 10`**: Compute results for 10 different sigma values (el_ph_sigma to el_ph_nsigma * el_ph_sigma). Allows checking convergence without re-running.
- **`tr2_ph = 1.0d-16`**: Very tight DFPT self-consistency. Needed for reliable el-ph matrix elements.
- **`recover = .true.`**: Enables restart. Already-completed q-points are skipped on re-run.

Run:

```bash
mpirun -np 8 ph.x -npool 2 < ph_elph.in > ph_elph.out 2>&1
```

This produces:
- `mgb2.dyn1`, `mgb2.dyn2`, ... (dynamical matrices at each irreducible q-point)
- `elph_dir/elph.inp_lambda.*` (el-ph coupling data per q-point)
- `a2Fq2r.*` files (alpha2F contributions for Fourier interpolation)

Verify:

```bash
grep "JOB DONE" ph_elph.out
# Check phonon frequencies -- all should be positive (except Gamma acoustic ~ 0):
grep "omega(" ph_elph.out | head -30
# Count completed q-points:
ls elph_dir/elph.inp_lambda.* 2>/dev/null | wc -l
ls a2Fq2r.* 2>/dev/null | wc -l
```

### Step 3: Compute lambda, omega_log, and Tc with lambda.x

`lambda.x` reads the el-ph data and computes the coupling constant lambda, logarithmic average frequency omega_log, and Tc via the McMillan/Allen-Dynes formula.

First, auto-generate the input file:

```python
#!/usr/bin/env python3
"""
generate_lambda_input.py
Automatically generate lambda.x input by locating elph files.
"""
import glob
import os
import sys

# --- Configuration ---
el_ph_nsigma = 10      # Must match ph.x input
mu_star = 0.12          # Coulomb pseudopotential (0.10-0.16)
sigma_to_report = 1     # Which sigma to print Tc for (1-based)

# Locate elph files -- try several common QE output locations
search_paths = [
    'elph_dir/elph.inp_lambda.*',
    'tmp/_ph0/elph_dir/elph.inp_lambda.*',
    'tmp/elph_dir/elph.inp_lambda.*',
    './tmp/mgb2.phsave/elph.inp_lambda.*',
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
print(f"Using mu_star = {mu_star}, el_ph_nsigma = {el_ph_nsigma}")
```

Run lambda.x:

```bash
python3 generate_lambda_input.py
lambda.x < lambda.in > lambda.out 2>&1
cat lambda.out
```

The output contains lambda, omega_log (in K), and Tc (in K) for each sigma value. Look for lines like:

```
lambda = 0.7321  omega_log =  543.21 (K)  Tc =  25.31 (K)
```

### Step 4: Eliashberg Spectral Function with alpha2f.x

Compute the full alpha2F(omega) spectral function from the Fourier-interpolated el-ph data:

```
cat > a2f.in << 'A2F_INPUT'
&INPUTA2F
    nfreq = 500
/
A2F_INPUT

alpha2f.x < a2f.in > a2f.out 2>&1
```

This produces files `a2F.dos.{sigma_index}` with columns: omega (cm^-1), alpha2F(omega), and possibly cumulative lambda(omega) depending on QE version.

```bash
# Check output files:
ls a2F.dos.*
# Preview first file:
head -20 a2F.dos.1
```

### Step 5: Post-Processing, Tc Analysis, and Plotting

```python
#!/usr/bin/env python3
"""
superconductivity_analysis.py
Complete post-processing for superconducting properties:
  - Parse lambda.x output (lambda, omega_log, Tc vs sigma)
  - Load and analyze alpha2F(omega)
  - Compute Tc with Allen-Dynes formula (standard + strong-coupling corrections)
  - Mode-resolved lambda contributions
  - Plot alpha2F(omega), cumulative lambda(omega), lambda convergence
"""
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import re
import os
import sys

# =============================================================================
# 1. Parse lambda.x output
# =============================================================================

def parse_lambda_output(filename='lambda.out'):
    """
    Parse lambda.x output to extract lambda, omega_log, Tc for each sigma.

    Returns
    -------
    list of dict with keys: 'sigma_idx', 'lambda', 'omega_log_K', 'Tc_K'
    Also returns per-q-point mode-resolved data if available.
    """
    with open(filename, 'r') as f:
        content = f.read()
        lines = content.split('\n')

    results = []

    # Pattern 1: QE 7.x format
    # "lambda = X.XXXX  omega_log = X.XXXX (K)  Tc = X.XXXX (K)"
    pattern_tc = r'lambda\s*=\s*([\d.]+)\s+omega_log\s*=\s*([\d.]+)\s*\(K\)\s+.*?Tc\s*=\s*([\d.]+)\s*\(K\)'
    matches = re.findall(pattern_tc, content)

    if matches:
        for i, m in enumerate(matches):
            results.append({
                'sigma_idx': i + 1,
                'lambda': float(m[0]),
                'omega_log_K': float(m[1]),
                'Tc_K': float(m[2]),
            })
        return results

    # Pattern 2: line-by-line fallback
    for line in lines:
        lam_match = re.search(r'lambda\s*=\s*([\d.]+)', line)
        tc_match = re.search(r'Tc\s*=\s*([\d.]+)', line, re.IGNORECASE)
        olog_match = re.search(r'omega_log\s*=\s*([\d.]+)', line)
        if lam_match and tc_match:
            entry = {
                'sigma_idx': len(results) + 1,
                'lambda': float(lam_match.group(1)),
                'Tc_K': float(tc_match.group(1)),
            }
            if olog_match:
                entry['omega_log_K'] = float(olog_match.group(1))
            results.append(entry)

    if not results:
        print(f"WARNING: Could not parse {filename}. Raw content:")
        print(content[:3000])

    return results


def parse_mode_lambda(filename='lambda.out'):
    """
    Parse mode-resolved lambda_qnu from lambda.x output.
    Returns dict: {q_index: [lambda_mode1, lambda_mode2, ...]}
    """
    with open(filename, 'r') as f:
        lines = f.readlines()

    mode_data = {}
    current_q = None

    for line in lines:
        # Look for q-point header
        q_match = re.search(r'q\s*=\s*\(\s*([-\d.]+)\s+([-\d.]+)\s+([-\d.]+)\s*\)', line)
        if q_match:
            current_q = (float(q_match.group(1)),
                         float(q_match.group(2)),
                         float(q_match.group(3)))
            mode_data[current_q] = []

        # Look for lambda_qnu values
        # Pattern: "mode X: omega = XXX.XX cm-1  lambda = X.XXXX"
        mode_match = re.search(
            r'(?:mode|nu)\s*=?\s*(\d+).*?omega\s*=\s*([\d.]+).*?lambda\s*=\s*([\d.]+)',
            line, re.IGNORECASE
        )
        if mode_match and current_q is not None:
            mode_data[current_q].append({
                'mode': int(mode_match.group(1)),
                'omega_cm': float(mode_match.group(2)),
                'lambda_qnu': float(mode_match.group(3)),
            })

    return mode_data


# =============================================================================
# 2. Load alpha2F spectral function
# =============================================================================

def load_a2f(filename=None, sigma_index=1):
    """
    Load alpha2F(omega) data from alpha2f.x output.

    Returns
    -------
    omega_cm : array
        Frequencies in cm^-1.
    a2f : array
        Eliashberg spectral function alpha2F(omega).
    cum_lambda : array
        Cumulative lambda(omega) = 2 * integral_0^omega [alpha2F(w)/w] dw.
    """
    if filename is None:
        candidates = [f'a2F.dos.{sigma_index}', f'a2F.dos{sigma_index}',
                      f'a2F.dos.0{sigma_index}', 'a2F.dos.1', 'alpha2F.dat']
        for fn in candidates:
            if os.path.isfile(fn):
                filename = fn
                break

    if filename is None or not os.path.isfile(filename):
        print("ERROR: No alpha2F data file found.")
        print("Expected files: a2F.dos.1, a2F.dos.01, etc.")
        return None, None, None

    print(f"Loading alpha2F from: {filename}")
    data = np.loadtxt(filename, comments='#')

    omega_cm = data[:, 0]   # cm^-1
    a2f = data[:, 1]        # alpha2F(omega)

    # Compute cumulative lambda: lambda(omega) = 2 * int_0^omega [a2F(w)/w] dw
    mask = omega_cm > 1.0   # Avoid division by zero
    integrand = np.zeros_like(omega_cm)
    integrand[mask] = 2.0 * a2f[mask] / omega_cm[mask]

    # Cumulative integral via trapezoidal rule
    cum_lambda = np.zeros_like(omega_cm)
    for i in range(1, len(omega_cm)):
        dw = omega_cm[i] - omega_cm[i-1]
        cum_lambda[i] = cum_lambda[i-1] + 0.5 * (integrand[i] + integrand[i-1]) * dw

    return omega_cm, a2f, cum_lambda


# =============================================================================
# 3. Tc formulas
# =============================================================================

def tc_mcmillan(lam, omega_log_K, mu_star=0.12):
    """
    Original McMillan formula (1968).
    Valid for lambda < 1.5.

    Tc = (Theta_D / 1.45) * exp[-1.04*(1+lambda) / (lambda - mu*(1+0.62*lambda))]

    Here we use omega_log instead of Theta_D (more accurate).
    """
    if lam <= 0 or omega_log_K <= 0:
        return 0.0
    denom = lam - mu_star * (1.0 + 0.62 * lam)
    if denom <= 0:
        return 0.0
    exponent = -1.04 * (1.0 + lam) / denom
    return (omega_log_K / 1.45) * np.exp(exponent)


def tc_allen_dynes(lam, omega_log_K, mu_star=0.12):
    """
    Allen-Dynes modified McMillan formula (1975).
    More accurate than McMillan, especially for lambda ~ 1-1.5.

    Tc = (omega_log / 1.2) * exp[-1.04*(1+lambda) / (lambda - mu*(1+0.62*lambda))]
    """
    if lam <= 0 or omega_log_K <= 0:
        return 0.0
    denom = lam - mu_star * (1.0 + 0.62 * lam)
    if denom <= 0:
        return 0.0
    exponent = -1.04 * (1.0 + lam) / denom
    return (omega_log_K / 1.2) * np.exp(exponent)


def tc_allen_dynes_strong_coupling(lam, omega_log_K, omega_2_K, mu_star=0.12):
    """
    Allen-Dynes formula with strong-coupling correction factors f1 and f2.
    Valid for lambda up to ~2-3.

    Tc = f1 * f2 * (omega_log / 1.2) * exp[-1.04*(1+lambda)/(lambda - mu*(1+0.62*lambda))]

    f1 = [1 + (lambda / Lambda_1)^(3/2)]^(1/3)
    f2 = 1 + (omega_2/omega_log - 1) * lambda^2 / (lambda^2 + Lambda_2^2)

    Lambda_1 = 2.46 * (1 + 3.8*mu_star)
    Lambda_2 = 1.82 * (1 + 6.3*mu_star) * (omega_2/omega_log)

    Parameters
    ----------
    lam : float
        Electron-phonon coupling constant.
    omega_log_K : float
        Logarithmic average frequency in Kelvin.
    omega_2_K : float
        Second moment average frequency in Kelvin:
        omega_2 = sqrt{ (2/lambda) * integral[alpha2F(w)*w dw] }
    mu_star : float
        Coulomb pseudopotential.
    """
    if lam <= 0 or omega_log_K <= 0 or omega_2_K <= 0:
        return 0.0

    # Strong-coupling corrections
    Lambda_1 = 2.46 * (1.0 + 3.8 * mu_star)
    ratio = omega_2_K / omega_log_K
    Lambda_2 = 1.82 * (1.0 + 6.3 * mu_star) * ratio

    f1 = (1.0 + (lam / Lambda_1) ** 1.5) ** (1.0 / 3.0)
    f2 = 1.0 + (ratio - 1.0) * lam ** 2 / (lam ** 2 + Lambda_2 ** 2)

    Tc_AD = tc_allen_dynes(lam, omega_log_K, mu_star)
    return f1 * f2 * Tc_AD


def compute_omega_log(omega_cm, a2f):
    """
    Compute logarithmic average frequency from alpha2F.

    omega_log = exp[ (2/lambda) * integral( alpha2F(w)/w * ln(w) dw ) ]

    Returns omega_log in Kelvin.
    """
    mask = omega_cm > 1.0
    w = omega_cm[mask]
    a = a2f[mask]

    lam = 2.0 * np.trapz(a / w, w)
    if lam <= 0:
        return 0.0

    log_avg = (2.0 / lam) * np.trapz((a / w) * np.log(w), w)
    omega_log_cm = np.exp(log_avg)

    # Convert cm^-1 to Kelvin: 1 cm^-1 = 1.4388 K
    return omega_log_cm * 1.4388


def compute_omega_2(omega_cm, a2f, lam):
    """
    Compute second moment average frequency from alpha2F.

    omega_2 = sqrt{ (2/lambda) * integral( alpha2F(w) * w  dw ) }

    Returns omega_2 in Kelvin.
    """
    mask = omega_cm > 1.0
    w = omega_cm[mask]
    a = a2f[mask]

    if lam <= 0:
        return 0.0

    moment = (2.0 / lam) * np.trapz(a * w, w)
    omega_2_cm = np.sqrt(moment)

    return omega_2_cm * 1.4388


# =============================================================================
# 4. Plotting
# =============================================================================

def plot_alpha2f_and_lambda(omega_cm, a2f, cum_lambda, output='alpha2F_spectrum.png'):
    """
    Plot Eliashberg spectral function with cumulative lambda on dual y-axes.
    """
    fig, ax1 = plt.subplots(figsize=(9, 5.5))

    ax1.fill_between(omega_cm, 0, a2f, alpha=0.25, color='steelblue')
    ax1.plot(omega_cm, a2f, color='steelblue', linewidth=1.5,
             label=r'$\alpha^2F(\omega)$')
    ax1.set_xlabel(r'Frequency $\omega$ (cm$^{-1}$)', fontsize=13)
    ax1.set_ylabel(r'$\alpha^2F(\omega)$', fontsize=13, color='steelblue')
    ax1.tick_params(axis='y', labelcolor='steelblue')
    ax1.set_xlim(left=0)
    ax1.set_ylim(bottom=0)

    ax2 = ax1.twinx()
    ax2.plot(omega_cm, cum_lambda, color='crimson', linewidth=2.0,
             linestyle='--', label=r'$\lambda(\omega)$')
    ax2.set_ylabel(r'Cumulative $\lambda(\omega)$', fontsize=13, color='crimson')
    ax2.tick_params(axis='y', labelcolor='crimson')
    ax2.set_ylim(bottom=0)

    total_lambda = cum_lambda[-1] if len(cum_lambda) > 0 else 0
    ax1.set_title(r'Eliashberg Spectral Function ($\lambda_{\rm total}$ = '
                  + f'{total_lambda:.3f})', fontsize=14)

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper right', fontsize=11)

    plt.tight_layout()
    plt.savefig(output, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output}")


def plot_lambda_vs_sigma(results, output='lambda_vs_sigma.png'):
    """
    Plot lambda and Tc as a function of Gaussian broadening sigma
    to verify convergence.
    """
    if not results:
        print("No lambda results to plot.")
        return

    sigma_idx = [r['sigma_idx'] for r in results]
    lambdas = [r['lambda'] for r in results]
    tcs = [r.get('Tc_K', 0) for r in results]

    fig, ax1 = plt.subplots(figsize=(8, 5))

    ax1.plot(sigma_idx, lambdas, 'o-', color='steelblue', linewidth=2,
             markersize=7, label=r'$\lambda$')
    ax1.set_xlabel(r'Broadening index (sigma)', fontsize=13)
    ax1.set_ylabel(r'$\lambda$', fontsize=13, color='steelblue')
    ax1.tick_params(axis='y', labelcolor='steelblue')

    ax2 = ax1.twinx()
    ax2.plot(sigma_idx, tcs, 's--', color='darkorange', linewidth=2,
             markersize=7, label=r'$T_c$ (K)')
    ax2.set_ylabel(r'$T_c$ (K)', fontsize=13, color='darkorange')
    ax2.tick_params(axis='y', labelcolor='darkorange')

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='best', fontsize=11)

    ax1.set_title(r'Convergence of $\lambda$ and $T_c$ with broadening', fontsize=14)
    ax1.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output}")


def plot_a2f_decomposition(omega_cm, a2f, phonon_dos_file=None,
                            output='a2f_vs_phdos.png'):
    """
    Compare alpha2F with phonon DOS to identify which modes
    contribute most to electron-phonon coupling.
    """
    fig, ax = plt.subplots(figsize=(9, 5.5))

    # Normalize alpha2F for comparison
    a2f_max = np.max(a2f) if np.max(a2f) > 0 else 1.0
    ax.fill_between(omega_cm, 0, a2f / a2f_max, alpha=0.3, color='steelblue',
                    label=r'$\alpha^2F(\omega)$ (normalized)')
    ax.plot(omega_cm, a2f / a2f_max, color='steelblue', linewidth=1.5)

    if phonon_dos_file and os.path.isfile(phonon_dos_file):
        dos_data = np.loadtxt(phonon_dos_file, comments='#')
        dos_omega = dos_data[:, 0]
        dos_val = dos_data[:, 1]
        dos_max = np.max(dos_val) if np.max(dos_val) > 0 else 1.0
        ax.plot(dos_omega, dos_val / dos_max, color='gray', linewidth=1.5,
                linestyle='--', label='Phonon DOS (normalized)')

    ax.set_xlabel(r'Frequency $\omega$ (cm$^{-1}$)', fontsize=13)
    ax.set_ylabel('Normalized intensity', fontsize=13)
    ax.set_title(r'$\alpha^2F(\omega)$ vs Phonon DOS', fontsize=14)
    ax.legend(fontsize=11)
    ax.set_xlim(left=0)
    ax.set_ylim(bottom=0)

    plt.tight_layout()
    plt.savefig(output, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output}")


def plot_tc_vs_mu_star(lam, omega_log_K, omega_2_K=None,
                        output='Tc_vs_mustar.png'):
    """
    Plot Tc as a function of Coulomb pseudopotential mu* to show
    the sensitivity of Tc to this empirical parameter.
    """
    mu_stars = np.linspace(0.05, 0.20, 100)

    Tc_AD = [tc_allen_dynes(lam, omega_log_K, m) for m in mu_stars]
    Tc_McM = [tc_mcmillan(lam, omega_log_K, m) for m in mu_stars]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(mu_stars, Tc_AD, '-', color='steelblue', linewidth=2,
            label='Allen-Dynes')
    ax.plot(mu_stars, Tc_McM, '--', color='darkorange', linewidth=2,
            label='McMillan')

    if omega_2_K is not None and omega_2_K > 0:
        Tc_SC = [tc_allen_dynes_strong_coupling(lam, omega_log_K, omega_2_K, m)
                 for m in mu_stars]
        ax.plot(mu_stars, Tc_SC, '-.', color='seagreen', linewidth=2,
                label='Allen-Dynes + strong coupling')

    ax.set_xlabel(r'$\mu^*$', fontsize=13)
    ax.set_ylabel(r'$T_c$ (K)', fontsize=13)
    ax.set_title(f'$T_c$ vs $\\mu^*$ ($\\lambda$ = {lam:.3f}, '
                 f'$\\omega_{{\\log}}$ = {omega_log_K:.1f} K)', fontsize=13)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(bottom=0)

    # Mark common mu* values
    for ms_val, ms_label in [(0.10, 'simple metals'), (0.13, 'typical'), (0.15, 'TM')]:
        Tc_here = tc_allen_dynes(lam, omega_log_K, ms_val)
        ax.plot(ms_val, Tc_here, 'ko', markersize=6)
        ax.annotate(f'{ms_label}\n{Tc_here:.1f} K',
                    xy=(ms_val, Tc_here), xytext=(ms_val + 0.01, Tc_here + 2),
                    fontsize=8, ha='left')

    plt.tight_layout()
    plt.savefig(output, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output}")


# =============================================================================
# 5. Main workflow
# =============================================================================

def main():
    print("=" * 65)
    print("  Superconducting Properties from Electron-Phonon Coupling")
    print("=" * 65)

    # --- Parse lambda.x output ---
    print("\n--- 1. Parsing lambda.x output ---")
    if os.path.isfile('lambda.out'):
        results = parse_lambda_output('lambda.out')
        if results:
            print(f"\n{'sigma_idx':>10} {'lambda':>10} {'omega_log(K)':>14} {'Tc_AD(K)':>10}")
            print("-" * 50)
            for r in results:
                lam = r.get('lambda', 0)
                olog = r.get('omega_log_K', 0)
                tc = r.get('Tc_K', 0)
                print(f"{r['sigma_idx']:>10} {lam:>10.4f} {olog:>14.2f} {tc:>10.3f}")

            plot_lambda_vs_sigma(results)
        else:
            print("Could not parse lambda.out.")
    else:
        print("lambda.out not found. Run lambda.x first.")
        results = []

    # --- Load alpha2F ---
    print("\n--- 2. Loading Eliashberg spectral function ---")
    omega_cm, a2f_data, cum_lambda = load_a2f()

    if omega_cm is not None:
        total_lambda = cum_lambda[-1]
        omega_log_K = compute_omega_log(omega_cm, a2f_data)
        omega_2_K = compute_omega_2(omega_cm, a2f_data, total_lambda)

        print(f"\n  Integrated quantities from alpha2F:")
        print(f"    lambda        = {total_lambda:.4f}")
        print(f"    omega_log     = {omega_log_K:.2f} K")
        print(f"    omega_2       = {omega_2_K:.2f} K")
        print(f"    omega_2/omega_log = {omega_2_K/omega_log_K:.3f}" if omega_log_K > 0 else "")

        # --- Tc with multiple formulas ---
        print(f"\n  Superconducting Tc estimates:")
        print(f"  {'mu*':>6}  {'McMillan':>10}  {'Allen-Dynes':>12}  {'AD+SC corr':>12}")
        print("  " + "-" * 46)
        for mu_s in [0.10, 0.12, 0.13, 0.15, 0.16]:
            Tc_M = tc_mcmillan(total_lambda, omega_log_K, mu_s)
            Tc_AD = tc_allen_dynes(total_lambda, omega_log_K, mu_s)
            Tc_SC = tc_allen_dynes_strong_coupling(
                total_lambda, omega_log_K, omega_2_K, mu_s)
            print(f"  {mu_s:>6.2f}  {Tc_M:>10.2f} K  {Tc_AD:>10.2f} K  {Tc_SC:>10.2f} K")

        # --- Plots ---
        print("\n--- 3. Generating plots ---")
        plot_alpha2f_and_lambda(omega_cm, a2f_data, cum_lambda)

        plot_a2f_decomposition(omega_cm, a2f_data,
                                phonon_dos_file='phonon_dos.dat')

        plot_tc_vs_mu_star(total_lambda, omega_log_K, omega_2_K)

        # --- Mode-resolved analysis ---
        print("\n--- 4. Mode-resolved lambda analysis ---")
        mode_data = parse_mode_lambda('lambda.out')
        if mode_data:
            print(f"\n  Found mode-resolved data for {len(mode_data)} q-points.")
            total_from_modes = 0
            for q, modes in mode_data.items():
                q_str = f"({q[0]:.3f}, {q[1]:.3f}, {q[2]:.3f})"
                q_lambda = sum(m['lambda_qnu'] for m in modes)
                total_from_modes += q_lambda
                print(f"  q = {q_str}:  lambda_q = {q_lambda:.4f}")
                for m in modes:
                    print(f"    mode {m['mode']:2d}: omega = {m['omega_cm']:8.2f} cm-1, "
                          f"lambda_qnu = {m['lambda_qnu']:.4f}")
        else:
            print("  No mode-resolved data found in lambda.out.")

        # --- Save summary ---
        print("\n--- 5. Summary ---")
        summary = {
            'lambda': total_lambda,
            'omega_log_K': omega_log_K,
            'omega_2_K': omega_2_K,
        }
        for mu_s in [0.10, 0.12, 0.13]:
            summary[f'Tc_AD_mu{mu_s:.2f}'] = tc_allen_dynes(
                total_lambda, omega_log_K, mu_s)
            summary[f'Tc_SC_mu{mu_s:.2f}'] = tc_allen_dynes_strong_coupling(
                total_lambda, omega_log_K, omega_2_K, mu_s)

        with open('superconductivity_summary.txt', 'w') as f:
            f.write("Superconducting Properties Summary\n")
            f.write("=" * 50 + "\n\n")
            for k, v in summary.items():
                f.write(f"{k:30s} = {v:.4f}\n")

        print(f"  Summary written to superconductivity_summary.txt")

    else:
        print("  No alpha2F data. Run alpha2f.x first.")

    print("\n" + "=" * 65)
    print("  Analysis complete.")
    print("=" * 65)


if __name__ == '__main__':
    main()
```

Run the analysis:

```bash
python3 superconductivity_analysis.py
```

### Complete Automated Workflow Script

```bash
#!/bin/bash
# run_superconductivity.sh
# Full workflow: SCF -> phonons + el-ph -> lambda -> alpha2F -> analysis
set -e

NP=8
NPOOL=2
PREFIX="mgb2"

echo "========================================================"
echo "  Superconductivity Workflow: SCF -> el-ph -> Tc"
echo "========================================================"

# Step 1: SCF
echo ""
echo "--- Step 1: SCF ---"
mpirun -np $NP pw.x -npool $NPOOL < scf.in > scf.out 2>&1
if grep -q "convergence has been achieved" scf.out; then
    echo "SCF converged."
    grep "the Fermi energy is" scf.out
else
    echo "ERROR: SCF not converged!"
    exit 1
fi

# Step 2: Phonons + el-ph matrix elements
echo ""
echo "--- Step 2: Phonons + electron-phonon (most expensive step) ---"
mpirun -np $NP ph.x -npool $NPOOL < ph_elph.in > ph_elph.out 2>&1
if grep -q "JOB DONE" ph_elph.out; then
    echo "Phonon + el-ph calculation complete."
else
    echo "ERROR: ph.x did not complete. Check ph_elph.out."
    exit 1
fi

# Step 3: lambda.x
echo ""
echo "--- Step 3: lambda.x ---"
python3 generate_lambda_input.py
lambda.x < lambda.in > lambda.out 2>&1
echo "lambda.x completed."

# Step 4: alpha2f.x
echo ""
echo "--- Step 4: alpha2f.x ---"
if command -v alpha2f.x &> /dev/null; then
    cat > a2f.in << 'EOF'
&INPUTA2F
    nfreq = 500
/
EOF
    alpha2f.x < a2f.in > a2f.out 2>&1
    echo "alpha2f.x completed."
else
    echo "alpha2f.x not found. Skipping."
fi

# Step 5: Post-processing
echo ""
echo "--- Step 5: Post-processing ---"
python3 superconductivity_analysis.py

echo ""
echo "========================================================"
echo "  Workflow complete!"
echo "========================================================"
```

### Pressure-Dependent Tc Workflow

To study how Tc evolves under pressure (relevant for hydride superconductors):

```python
#!/usr/bin/env python3
"""
pressure_tc_scan.py
Scan Tc as a function of pressure by repeating the el-ph workflow
at different volumes.

This script generates input files for each pressure point.
You must run the QE calculations separately at each volume.
"""
import numpy as np
import os
import json

# Pressures to scan (GPa)
pressures_GPa = [0, 10, 20, 50, 100, 150, 200]

# Reference lattice parameters at 0 GPa
a0_bohr = 5.8261     # MgB2 example
c_over_a = 1.1421

# Bulk modulus for volume estimation (GPa)
B0 = 150.0  # Approximate bulk modulus
B0_prime = 4.0  # Pressure derivative

# Murnaghan equation of state: V(P) = V0 * [1 + B'*P/B0]^(-1/B')
V0_relative = 1.0
for P in pressures_GPa:
    V_ratio = (1.0 + B0_prime * P / B0) ** (-1.0 / B0_prime)
    scale = V_ratio ** (1.0 / 3.0)  # isotropic scaling
    a_new = a0_bohr * scale
    c_new = a_new * c_over_a  # Assuming c/a ratio is constant (approximation)

    work_dir = f"P_{P}GPa"
    os.makedirs(work_dir, exist_ok=True)

    # Write SCF input for this pressure
    scf_input = f"""&CONTROL
    calculation   = 'vc-relax'
    prefix        = 'mgb2'
    pseudo_dir    = '../pseudo/'
    outdir        = './tmp/'
    tprnfor       = .true.
    tstress       = .true.
    verbosity     = 'high'
    forc_conv_thr = 1.0d-5
/
&SYSTEM
    ibrav         = 4
    celldm(1)     = {a_new:.4f}
    celldm(3)     = {c_over_a:.4f}
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
/
&IONS
/
&CELL
    press         = {P * 10.0:.1f}
    cell_dofree   = 'all'
/
ATOMIC_SPECIES
  Mg  24.305   Mg.pbe-spnl-kjpaw_psl.1.0.0.UPF
  B   10.811   B.pbe-n-kjpaw_psl.1.0.0.UPF

ATOMIC_POSITIONS {{crystal}}
  Mg  0.000000  0.000000  0.000000
  B   0.333333  0.666667  0.500000
  B   0.666667  0.333333  0.500000

K_POINTS {{automatic}}
  12 12 12  0 0 0
"""

    with open(os.path.join(work_dir, 'scf.in'), 'w') as f:
        f.write(scf_input)

    print(f"P = {P:6.1f} GPa: a = {a_new:.4f} bohr, V/V0 = {V_ratio:.4f}, "
          f"dir = {work_dir}/")

print("\nGenerated SCF input files for each pressure.")
print("Run the full el-ph workflow in each directory, then collect Tc(P).")
```

## Key Parameters

| Parameter | Location | Typical Range | Impact |
|---|---|---|---|
| `nk1,nk2,nk3` | ph.x input | 24-40 per direction | Dense k-grid for Fermi surface. Most critical. Must converge lambda to ~5%. |
| `nq1,nq2,nq3` | ph.x input | 4-8 per direction | Phonon momentum grid. 4x4x4 minimum, 6x6x6 standard. |
| `el_ph_sigma` | ph.x input | 0.002-0.02 Ry | Gaussian broadening for double-delta. Too small: noisy. Too large: smears features. |
| `el_ph_nsigma` | ph.x input | 5-15 | Number of sigma values to scan. Use 10 for convergence check. |
| `degauss` | scf.in | 0.005-0.05 Ry | Electronic smearing. Independent of el_ph_sigma. |
| `ecutwfc` | scf.in | 40-80 Ry | Plane-wave cutoff. Must be converged for total energy. |
| `mu_star` | lambda.in + analysis | 0.10-0.16 | Coulomb pseudopotential. Empirical. 0.10 simple metals, 0.13 typical, 0.15 transition metals. |
| `nfreq` | a2f.in | 200-1000 | Number of frequency points in alpha2F. 500 usually sufficient. |
| `la2F` | scf.in | .true. | **Mandatory** for el-ph. Must be set in SCF input. |
| `tr2_ph` | ph.x input | 1.0d-14 to 1.0d-18 | DFPT convergence. Tighter is safer for el-ph matrix elements. |

### Convergence Recipe

1. **Quick test** (30 min): `nq=2x2x2`, `nk=12x12x12`, `el_ph_sigma=0.01`
2. **Coarse production** (few hours): `nq=4x4x4`, `nk=24x24x24`, `el_ph_sigma=0.005`
3. **Converged production** (1-2 days): `nq=6x6x6`, `nk=32x32x32`
4. **High accuracy** (days): `nq=8x8x8`, `nk=40x40x40`
5. At each level, check that lambda changes by <5% compared to the previous level.

## Interpreting Results

### Lambda Values and Superconductivity

| Lambda Range | Coupling Regime | Tc Behavior | Examples |
|---|---|---|---|
| < 0.3 | Weak | Tc ~ 0 or immeasurably small | Cu, Ag, Au, alkali metals |
| 0.3 - 0.5 | Moderate-weak | Tc < 5 K | Al (0.43, Tc=1.2 K), Zn (0.38) |
| 0.5 - 1.0 | Intermediate | Tc ~ 5-20 K | Nb (0.82, Tc=9.3 K), V (0.60) |
| 1.0 - 1.5 | Strong | Tc ~ 10-40 K | Pb (1.55, Tc=7.2 K), MgB2 (~0.7-1.0) |
| 1.5 - 3.0 | Very strong | Tc ~ 20-100+ K | H3S (~2.0, Tc=203 K under pressure) |
| > 3.0 | Extremely strong | Allen-Dynes unreliable; solve Eliashberg | LaH10, CaH6 |

### Alpha2F Spectral Function Analysis

- **Peaks** in alpha2F(omega) identify phonon modes that couple strongly to electrons.
- **Low-frequency peaks** contribute disproportionately to lambda due to the 1/omega weighting in lambda = 2 * integral[alpha2F(omega)/omega] d_omega.
- **Compare alpha2F with phonon DOS**: If alpha2F tracks the phonon DOS shape, coupling is roughly uniform. If specific peaks are enhanced, those modes dominate superconductivity.
- **MgB2 example**: The E2g boron stretching mode (~550 cm^-1) shows a giant peak in alpha2F, explaining the anomalously high Tc for a light-element compound.
- **Phonon softening under pressure**: Modes that soften contribute more to lambda (lower omega in denominator). This is why pressure can enhance Tc in some materials.

### Tc Formula Reliability

- **McMillan formula**: Accurate for lambda < 1.0. Systematically underestimates Tc for stronger coupling.
- **Allen-Dynes formula**: More accurate up to lambda ~ 1.5. Standard choice for most calculations.
- **Allen-Dynes + strong-coupling corrections**: Valid up to lambda ~ 2-3. Uses both omega_log and omega_2.
- **Full Eliashberg equations**: Required for lambda > 2-3. Not implemented in this skill but available in codes like EPW and Eliashberg.x.
- **mu_star sensitivity**: Tc varies by ~30-50% over the typical range mu_star = 0.10-0.15. Always report Tc for a range of mu_star values.

### Sanity Checks

1. **Positive phonon frequencies**: All phonon modes should be real. Imaginary frequencies indicate structural instability (wrong phase).
2. **Lambda consistency**: Lambda from lambda.x should agree with the integral of alpha2F to within ~5%.
3. **Lambda convergence**: Lambda should plateau when increasing nk and nq grids.
4. **Physical range**: Lambda > 3-4 is suspect for conventional superconductors (except high-pressure hydrides).
5. **Sum rule**: For each q-point, sum of lambda_qnu over modes should equal lambda_q. Sum of lambda_q * weight_q should equal total lambda.

## Common Issues

| Problem | Symptom | Solution |
|---|---|---|
| Missing `la2F = .true.` | No elph files produced by ph.x | Add `la2F = .true.` to `&SYSTEM` in SCF input. Must re-run SCF. |
| Negative phonon frequencies | `omega < 0` in ph.x output | Relax structure with tight thresholds (`forc_conv_thr = 1.0d-5`). Increase ecutwfc. Check if phase is dynamically stable. |
| Lambda not converging with k-grid | Lambda changes >10% when increasing nk | Increase nk further (40x40x40 or more). Try different el_ph_sigma. Use cold smearing. |
| Unphysically large lambda (>5) | Lambda fluctuates wildly or grows without bound | el_ph_sigma too small (increase to 0.005-0.01 Ry). Check for soft modes near instability. Verify la2F was set. |
| ph.x crashes with OOM | Out-of-memory error | Reduce nk temporarily. Use more MPI processes. Try fewer pools. |
| Cannot find elph files | lambda.x reports no input files | Check `outdir` directory. Files may be in `tmp/_ph0/elph_dir/`. Ensure `electron_phonon = 'interpolated'` was set. |
| Lambda from lambda.x disagrees with alpha2F integral | >10% discrepancy | Different broadening. Check frequency grid resolution (increase nfreq in a2f.in). Small differences (<5%) are normal. |
| ph.x runs forever at one q-point | Calculation stalls without progress | Acoustic sum rule violations at Gamma can slow convergence. Try `asr = 'crystal'` in matdyn.x for post-processing. For ph.x, adjust alpha_mix or tr2_ph. |
| Alpha2F has spikes at low frequency | Unphysical sharp features near omega = 0 | Acoustic modes near Gamma may not be properly treated. These should be small; if large, increase q-grid or use `asr = 'simple'` in alpha2f.x. |
| Tc much higher than experiment | Computed Tc >> experimental Tc | mu_star may be too low (increase toward 0.15-0.16). Lambda may be overestimated (check convergence). Anharmonic effects reduce Tc in some materials. |
| Tc much lower than experiment | Computed Tc << experimental Tc | mu_star may be too high. Check if lambda is converged. For strong coupling, McMillan/Allen-Dynes underestimates -- try strong-coupling corrections. |
