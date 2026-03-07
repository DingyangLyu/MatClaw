# Boltzmann Transport Properties (BoltzTraP2)

## When to Use

- You need thermoelectric transport coefficients: Seebeck coefficient (thermopower), electrical conductivity, electronic thermal conductivity.
- You want to compute the thermoelectric figure of merit ZT or the power factor S^2*sigma.
- You need to evaluate how transport properties vary with temperature and carrier concentration (doping level).
- You want to identify the optimal doping level for thermoelectric performance.
- You are studying semiconductors, metals, semimetals, or doped materials for thermoelectric applications.
- You need transport tensors (anisotropic Seebeck, conductivity) for layered or anisotropic materials.

This skill uses the Boltzmann transport equation (BTE) in the constant relaxation time approximation (CRTA) via BoltzTraP2. The Seebeck coefficient is obtained exactly (relaxation time cancels out). Conductivity and electronic thermal conductivity are given as sigma/tau and kappa_e/tau respectively.

## Method Selection

| Method | Tool | Strengths | Limitations |
|---|---|---|---|
| QE + BoltzTraP2 | pw.x + Python | Available in container, full workflow | Requires very dense k-grid NSCF |
| VASP + BoltzTraP2 | VASP + Python | Standard for VASP users, reads EIGENVAL | Requires VASP license (not in container) |
| QE + pymatgen BoltzTraP2 | pw.x + pymatgen | Convenient pymatgen interface | Less robust for QE data than direct BoltzTraP2 |
| VASP + pymatgen BoltzTraP2 | VASP + pymatgen | Reads vasprun.xml natively | Requires VASP license |

## Prerequisites

- Quantum ESPRESSO 7.5 (`pw.x`) on PATH -- or VASP for the VASP workflow.
- Python with `numpy`, `scipy`, `matplotlib`.
- BoltzTraP2: `pip install BoltzTraP2`.
- Appropriate pseudopotentials (SSSP or PSlibrary for QE).
- A very dense k-grid (40x40x40 or denser) for reliable transport properties.
- A relaxed crystal structure.

## Detailed Steps

### Overview

```
Method A (QE + BoltzTraP2):
  Step 0: Download pseudopotentials
  Step 1: SCF calculation (pw.x, coarse k-grid)
  Step 2: NSCF on dense uniform k-grid (pw.x, nosym=.true.)
  Step 3: BoltzTraP2 transport calculation (Python)
  Step 4: Plot and analyze results

Method B (VASP + BoltzTraP2):
  Step 1: VASP SCF with dense k-mesh (ISMEAR, NBANDS)
  Step 2: BoltzTraP2 from VASP output (Python)
  Step 3: Plot and analyze results
```

---

### Method A: QE + BoltzTraP2

#### Step A0: Download Pseudopotentials

```python
#!/usr/bin/env python3
"""Download SSSP pseudopotentials for Si (example thermoelectric material)."""
import os
import urllib.request

PSEUDO_DIR = os.path.abspath("./pseudo")
os.makedirs(PSEUDO_DIR, exist_ok=True)

pseudos = {
    "Si": "Si.pbe-n-kjpaw_psl.1.0.0.UPF",
}

BASE_URL = "https://pseudopotentials.quantum-espresso.org/upf_files/"

for element, fname in pseudos.items():
    dest = os.path.join(PSEUDO_DIR, fname)
    if os.path.exists(dest):
        print(f"  {fname} already exists, skipping.")
        continue
    url = BASE_URL + fname
    print(f"  Downloading {fname} ...")
    try:
        urllib.request.urlretrieve(url, dest)
        print(f"  OK: {fname}")
    except Exception as e:
        print(f"  WARNING: Could not download {fname}: {e}")

print(f"\nPseudopotentials in {PSEUDO_DIR}:")
for f in sorted(os.listdir(PSEUDO_DIR)):
    print(f"  {f}")
```

#### Step A1: SCF Calculation (pw.x)

```python
#!/usr/bin/env python3
"""
Run SCF for Si as prerequisite for transport calculation.
This establishes the ground-state charge density on a coarse k-grid.
"""
import os
import subprocess

PSEUDO_DIR = os.path.abspath("./pseudo")
OUTDIR = os.path.abspath("./tmp_transport")
os.makedirs(OUTDIR, exist_ok=True)

# Si diamond structure (a = 5.431 Angstrom = 10.2631 Bohr)
scf_input = f"""&CONTROL
    calculation   = 'scf'
    prefix        = 'si'
    pseudo_dir    = '{PSEUDO_DIR}'
    outdir        = '{OUTDIR}'
    tprnfor       = .true.
    tstress       = .true.
    verbosity     = 'high'
/

&SYSTEM
    ibrav         = 2
    celldm(1)     = 10.2631
    nat           = 2
    ntyp          = 1
    ecutwfc       = 60.0
    ecutrho       = 480.0
    occupations   = 'smearing'
    smearing      = 'gaussian'
    degauss       = 0.005
    nbnd          = 16
/

&ELECTRONS
    conv_thr      = 1.0d-10
    mixing_beta   = 0.7
/

ATOMIC_SPECIES
  Si  28.0855  Si.pbe-n-kjpaw_psl.1.0.0.UPF

ATOMIC_POSITIONS {{crystal}}
  Si  0.00  0.00  0.00
  Si  0.25  0.25  0.25

K_POINTS {{automatic}}
  8 8 8  0 0 0
"""

with open("scf.in", "w") as f:
    f.write(scf_input)
print("Written: scf.in")

print("Running pw.x SCF ...")
result = subprocess.run(
    ["mpirun", "-np", "4", "pw.x", "-npool", "2", "-in", "scf.in"],
    capture_output=True, text=True, timeout=1800
)
with open("scf.out", "w") as f:
    f.write(result.stdout)

if result.returncode != 0:
    print("ERROR: SCF failed!")
    print(result.stderr[-500:] if result.stderr else "No stderr")
else:
    if "convergence has been achieved" in result.stdout:
        print("SCF converged successfully.")
    else:
        print("WARNING: SCF may not have converged.")
    # Extract Fermi energy
    for line in result.stdout.split("\n"):
        if "Fermi energy" in line or "highest occupied" in line:
            print(f"  {line.strip()}")
    print("Output: scf.out")
```

**Important notes:**
- Use `occupations = 'smearing'` even for semiconductors -- it helps SCF convergence. BoltzTraP2 handles the actual Fermi-Dirac statistics internally.
- `nbnd = 16`: Include enough bands above the gap. For transport you need bands spanning the energy window of interest (typically +/- 1.5 eV around the Fermi level). Use at least 2x the number of occupied bands.
- The SCF k-grid (8x8x8) is coarse. The dense grid comes in the NSCF step.

#### Step A2: NSCF on Dense Uniform K-Grid

```python
#!/usr/bin/env python3
"""
NSCF calculation on a very dense uniform k-grid for BoltzTraP2.
This is the most critical step for transport properties.

MANDATORY: nosym = .true. -- BoltzTraP2 needs the FULL Brillouin zone,
not just the irreducible wedge.
"""
import os
import subprocess

PSEUDO_DIR = os.path.abspath("./pseudo")
OUTDIR = os.path.abspath("./tmp_transport")

# Dense k-grid: 40x40x40 for Si. Adjust for your system.
# Larger unit cells need fewer k-points per direction.
NKPT = 40

nscf_input = f"""&CONTROL
    calculation   = 'nscf'
    prefix        = 'si'
    pseudo_dir    = '{PSEUDO_DIR}'
    outdir        = '{OUTDIR}'
    verbosity     = 'high'
/

&SYSTEM
    ibrav         = 2
    celldm(1)     = 10.2631
    nat           = 2
    ntyp          = 1
    ecutwfc       = 60.0
    ecutrho       = 480.0
    occupations   = 'smearing'
    smearing      = 'gaussian'
    degauss       = 0.005
    nbnd          = 16
    nosym         = .true.
/

&ELECTRONS
    conv_thr      = 1.0d-10
/

ATOMIC_SPECIES
  Si  28.0855  Si.pbe-n-kjpaw_psl.1.0.0.UPF

ATOMIC_POSITIONS {{crystal}}
  Si  0.00  0.00  0.00
  Si  0.25  0.25  0.25

K_POINTS {{automatic}}
  {NKPT} {NKPT} {NKPT}  0 0 0
"""

with open("nscf.in", "w") as f:
    f.write(nscf_input)
print(f"Written: nscf.in (k-grid: {NKPT}x{NKPT}x{NKPT})")

print("Running pw.x NSCF (this may take 30-120 minutes for 40^3 grid) ...")
result = subprocess.run(
    ["mpirun", "-np", "4", "pw.x", "-npool", "4", "-in", "nscf.in"],
    capture_output=True, text=True, timeout=7200
)
with open("nscf.out", "w") as f:
    f.write(result.stdout)

if result.returncode != 0:
    print("ERROR: NSCF failed!")
    print(result.stderr[-500:] if result.stderr else "No stderr")
else:
    print("NSCF completed.")
    # Count k-points
    kpt_count = result.stdout.count("k =")
    print(f"  Total k-points computed: ~{kpt_count}")
    print("Output: nscf.out")
```

**Critical parameters for the NSCF step:**

| Parameter | Value | Why |
|---|---|---|
| `nosym` | `.true.` | **MANDATORY.** BoltzTraP2 needs the full BZ, not the irreducible wedge. |
| `K_POINTS` | `40 40 40` | Dense uniform grid. Transport converges slowly with k-grid. |
| `calculation` | `'nscf'` | Reads charge density from SCF; only computes eigenvalues. |
| `nbnd` | Same as SCF | Must cover the energy window of interest. |
| `prefix/outdir` | Same as SCF | Must match so NSCF reads the SCF charge density. |

#### Step A3: BoltzTraP2 Transport Calculation

```python
#!/usr/bin/env python3
"""
Complete BoltzTraP2 workflow: load QE data, compute transport coefficients,
plot Seebeck, sigma/tau, kappa_e/tau, power factor, and ZT vs T and doping.
"""
import os
import sys
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# ── Install BoltzTraP2 if needed ──────────────────────────────────────
try:
    import BoltzTraP2
    import BoltzTraP2.dft
    import BoltzTraP2.bandlib
    import BoltzTraP2.fite
except ImportError:
    print("Installing BoltzTraP2 ...")
    os.system("pip install BoltzTraP2")
    import BoltzTraP2
    import BoltzTraP2.dft
    import BoltzTraP2.bandlib
    import BoltzTraP2.fite


# =====================================================================
# Part 1: Load QE data and run BoltzTraP2 interpolation
# =====================================================================

def run_bolztrap2_from_qe(prefix='si', outdir='./tmp_transport/',
                          temp_range=(100, 1000, 19),
                          mu_window_eV=1.5, n_mu=2000):
    """
    Load QE NSCF data, perform BoltzTraP2 Fourier interpolation, and
    compute Onsager transport coefficients on a (T, mu) grid.

    Parameters
    ----------
    prefix : str
        QE calculation prefix.
    outdir : str
        QE output directory containing prefix.save/.
    temp_range : tuple
        (T_min, T_max, n_points) in Kelvin.
    mu_window_eV : float
        Chemical potential scan window: Fermi +/- this value in eV.
    n_mu : int
        Number of chemical potential points.

    Returns
    -------
    results : dict with temperatures, mu_array, fermi, sigma, seebeck,
              kappa, carrier_conc, volume_m3.
    """
    data_dir = os.path.join(outdir, f'{prefix}.save')
    if not os.path.isdir(data_dir):
        print(f"ERROR: {data_dir} not found.")
        print("Ensure the NSCF calculation completed and prefix/outdir match.")
        sys.exit(1)

    print(f"Loading QE data from: {data_dir}")
    data = BoltzTraP2.dft.DFTData(data_dir)
    print(f"  k-points: {data.kpoints.shape[0]}")
    print(f"  bands:    {data.ebands.shape[1]}")
    print(f"  Fermi:    {data.fermi:.6f} Ha")

    # ── Fourier interpolation ──────────────────────────────────────
    print("\nFourier interpolation of band energies ...")
    equivalences = BoltzTraP2.fite.equivalences(data, magmom=None)
    coeffs = BoltzTraP2.fite.fitde3D(data, equivalences)
    print(f"  Star functions: {len(equivalences)}")

    # ── Temperature and chemical potential grids ───────────────────
    T_min, T_max, n_T = temp_range
    temperatures = np.linspace(T_min, T_max, int(n_T))

    eV_to_Ha = 1.0 / 27.2114
    mu_min = data.fermi - mu_window_eV * eV_to_Ha
    mu_max = data.fermi + mu_window_eV * eV_to_Ha
    mu_array = np.linspace(mu_min, mu_max, n_mu)

    # ── Compute transport DOS ──────────────────────────────────────
    print("\nComputing transport DOS and Onsager coefficients ...")
    volume_m3 = data.atoms.get_volume() * 1e-30  # Angstrom^3 -> m^3

    LMTC = BoltzTraP2.fite.BTPDOS(
        data.ebands,
        data.mommat if hasattr(data, 'mommat') and data.mommat is not None else None,
        data.kpoints,
        temperatures,
        mu_array,
        volume_m3,
        equivalences,
        coeffs
    )
    epsilon, dos, vvdos, cdos = LMTC

    # ── Onsager transport coefficients ─────────────────────────────
    sigma, seebeck, kappa, hall = BoltzTraP2.bandlib.calc_Onsager_coefficients(
        epsilon, dos, vvdos, temperatures, mu_array, volume_m3
    )

    # ── Carrier concentration ──────────────────────────────────────
    carrier_conc = BoltzTraP2.bandlib.calc_N(
        epsilon, dos, temperatures, mu_array, volume_m3
    )

    results = {
        'temperatures': temperatures,         # shape: (n_T,)
        'mu_array': mu_array,                 # shape: (n_mu,)
        'fermi': data.fermi,                  # scalar, in Ha
        'sigma': sigma,                       # shape: (n_T, n_mu, 3, 3), 1/(Ohm*m*s)
        'seebeck': seebeck,                   # shape: (n_T, n_mu, 3, 3), V/K
        'kappa': kappa,                       # shape: (n_T, n_mu, 3, 3), W/(m*K*s)
        'carrier_conc': carrier_conc,         # shape: (n_T, n_mu), m^-3
        'hall': hall,                          # Hall coefficient
        'volume_m3': volume_m3,
    }

    print("  Transport coefficients computed successfully.")
    return results


# =====================================================================
# Part 2: Extract transport at specific doping levels
# =====================================================================

def extract_at_doping(results, target_conc_cm3, carrier_type='n'):
    """
    Extract transport properties at a specific carrier concentration.

    Parameters
    ----------
    results : dict
        Output from run_bolztrap2_from_qe.
    target_conc_cm3 : float
        Target carrier concentration in cm^-3.
    carrier_type : str
        'n' for n-type (electrons) or 'p' for p-type (holes).

    Returns
    -------
    dict with arrays: temperatures, seebeck (V/K), sigma_tau (1/(Ohm*m*s)),
    kappa_tau (W/(m*K*s)), power_factor_tau (W/(m*K^2*s)).
    """
    temperatures = results['temperatures']
    mu_array = results['mu_array']
    fermi = results['fermi']
    carrier_conc = results['carrier_conc']
    seebeck = results['seebeck']
    sigma = results['sigma']
    kappa = results['kappa']

    target_conc_m3 = target_conc_cm3 * 1e6

    extracted = {
        'temperatures': temperatures,
        'seebeck': [],
        'sigma_tau': [],
        'kappa_tau': [],
        'power_factor_tau': [],
    }

    for iT in range(len(temperatures)):
        conc = carrier_conc[iT, :]

        if carrier_type == 'n':
            mask = mu_array >= fermi
        else:
            mask = mu_array <= fermi

        conc_diff = np.abs(np.abs(conc) - target_conc_m3)
        conc_diff_masked = np.where(mask, conc_diff, np.inf)
        idx = np.argmin(conc_diff_masked)

        # Isotropic average (trace/3) of 3x3 tensor quantities
        S = np.trace(seebeck[iT, idx, :, :]) / 3.0
        sig = np.trace(sigma[iT, idx, :, :]) / 3.0
        kap = np.trace(kappa[iT, idx, :, :]) / 3.0
        PF = S**2 * sig

        extracted['seebeck'].append(S)
        extracted['sigma_tau'].append(sig)
        extracted['kappa_tau'].append(kap)
        extracted['power_factor_tau'].append(PF)

    for key in ['seebeck', 'sigma_tau', 'kappa_tau', 'power_factor_tau']:
        extracted[key] = np.array(extracted[key])

    return extracted


# =====================================================================
# Part 3: Plotting functions
# =====================================================================

def plot_seebeck_vs_temperature(results, doping_levels_cm3=None,
                                 carrier_type='n', output='seebeck_vs_T.png'):
    """Plot Seebeck coefficient vs temperature for multiple doping levels."""
    if doping_levels_cm3 is None:
        doping_levels_cm3 = [1e18, 1e19, 1e20, 1e21]

    fig, ax = plt.subplots(figsize=(8, 5))
    colors = plt.cm.viridis(np.linspace(0.2, 0.9, len(doping_levels_cm3)))

    for i, n in enumerate(doping_levels_cm3):
        ext = extract_at_doping(results, n, carrier_type)
        S_uV_K = ext['seebeck'] * 1e6  # V/K -> uV/K
        label = f'{carrier_type}-type, n = {n:.0e} cm$^{{-3}}$'
        ax.plot(ext['temperatures'], S_uV_K, '-o', color=colors[i],
                linewidth=2, markersize=4, label=label)

    ax.set_xlabel('Temperature (K)', fontsize=13)
    ax.set_ylabel(r'Seebeck coefficient ($\mu$V/K)', fontsize=13)
    ax.set_title(f'Seebeck Coefficient vs Temperature ({carrier_type}-type)',
                 fontsize=14)
    ax.legend(fontsize=9, loc='best')
    ax.grid(True, alpha=0.3)
    ax.axhline(y=0, color='k', linewidth=0.5)
    plt.tight_layout()
    plt.savefig(output, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output}")


def plot_sigma_vs_temperature(results, doping_levels_cm3=None,
                               carrier_type='n', output='sigma_vs_T.png'):
    """Plot electrical conductivity / tau vs temperature."""
    if doping_levels_cm3 is None:
        doping_levels_cm3 = [1e18, 1e19, 1e20, 1e21]

    fig, ax = plt.subplots(figsize=(8, 5))
    colors = plt.cm.plasma(np.linspace(0.2, 0.9, len(doping_levels_cm3)))

    for i, n in enumerate(doping_levels_cm3):
        ext = extract_at_doping(results, n, carrier_type)
        label = f'n = {n:.0e} cm$^{{-3}}$'
        ax.plot(ext['temperatures'], ext['sigma_tau'], '-o', color=colors[i],
                linewidth=2, markersize=4, label=label)

    ax.set_xlabel('Temperature (K)', fontsize=13)
    ax.set_ylabel(r'$\sigma/\tau$ (1/$\Omega\cdot$m$\cdot$s)', fontsize=13)
    ax.set_title(f'Electrical Conductivity / $\\tau$ ({carrier_type}-type)',
                 fontsize=14)
    ax.legend(fontsize=9, loc='best')
    ax.grid(True, alpha=0.3)
    ax.set_yscale('log')
    plt.tight_layout()
    plt.savefig(output, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output}")


def plot_kappa_vs_temperature(results, doping_levels_cm3=None,
                               carrier_type='n',
                               output='kappa_e_vs_T.png'):
    """Plot electronic thermal conductivity / tau vs temperature."""
    if doping_levels_cm3 is None:
        doping_levels_cm3 = [1e18, 1e19, 1e20, 1e21]

    fig, ax = plt.subplots(figsize=(8, 5))
    colors = plt.cm.coolwarm(np.linspace(0.2, 0.9, len(doping_levels_cm3)))

    for i, n in enumerate(doping_levels_cm3):
        ext = extract_at_doping(results, n, carrier_type)
        label = f'n = {n:.0e} cm$^{{-3}}$'
        ax.plot(ext['temperatures'], ext['kappa_tau'], '-o', color=colors[i],
                linewidth=2, markersize=4, label=label)

    ax.set_xlabel('Temperature (K)', fontsize=13)
    ax.set_ylabel(r'$\kappa_e/\tau$ (W/m$\cdot$K$\cdot$s)', fontsize=13)
    ax.set_title(f'Electronic Thermal Conductivity / $\\tau$ ({carrier_type}-type)',
                 fontsize=14)
    ax.legend(fontsize=9, loc='best')
    ax.grid(True, alpha=0.3)
    ax.set_yscale('log')
    plt.tight_layout()
    plt.savefig(output, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output}")


def plot_power_factor(results, doping_levels_cm3=None,
                      carrier_type='n', output='power_factor_vs_T.png'):
    """Plot power factor S^2*sigma/tau vs temperature."""
    if doping_levels_cm3 is None:
        doping_levels_cm3 = [1e18, 1e19, 1e20, 1e21]

    fig, ax = plt.subplots(figsize=(8, 5))
    colors = plt.cm.inferno(np.linspace(0.2, 0.85, len(doping_levels_cm3)))

    for i, n in enumerate(doping_levels_cm3):
        ext = extract_at_doping(results, n, carrier_type)
        label = f'n = {n:.0e} cm$^{{-3}}$'
        ax.plot(ext['temperatures'], ext['power_factor_tau'], '-o',
                color=colors[i], linewidth=2, markersize=4, label=label)

    ax.set_xlabel('Temperature (K)', fontsize=13)
    ax.set_ylabel(r'$S^2 \sigma / \tau$ (W/m$\cdot$K$^2\cdot$s)', fontsize=13)
    ax.set_title(f'Power Factor / $\\tau$ ({carrier_type}-type)', fontsize=14)
    ax.legend(fontsize=9, loc='best')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output}")


def plot_seebeck_vs_doping(results, temperatures_K=None,
                            carrier_type='n',
                            output='seebeck_vs_doping.png'):
    """Plot Seebeck coefficient vs carrier concentration at fixed temperatures."""
    if temperatures_K is None:
        temperatures_K = [300, 500, 700, 900]

    fig, ax = plt.subplots(figsize=(8, 5))
    all_temps = results['temperatures']
    colors = plt.cm.coolwarm(np.linspace(0.1, 0.9, len(temperatures_K)))
    doping_scan = np.logspace(17, 21, 80)

    for j, target_T in enumerate(temperatures_K):
        iT = np.argmin(np.abs(all_temps - target_T))
        actual_T = all_temps[iT]

        S_values = []
        for n in doping_scan:
            ext = extract_at_doping(results, n, carrier_type)
            S_values.append(ext['seebeck'][iT] * 1e6)

        ax.semilogx(doping_scan, S_values, '-', color=colors[j],
                     linewidth=2, label=f'T = {actual_T:.0f} K')

    ax.set_xlabel(r'Carrier concentration (cm$^{-3}$)', fontsize=13)
    ax.set_ylabel(r'Seebeck coefficient ($\mu$V/K)', fontsize=13)
    ax.set_title(f'Seebeck vs Doping ({carrier_type}-type)', fontsize=14)
    ax.legend(fontsize=10, loc='best')
    ax.grid(True, alpha=0.3)
    ax.axhline(y=0, color='k', linewidth=0.5)
    plt.tight_layout()
    plt.savefig(output, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output}")


def plot_zt_estimate(results, kappa_lattice_W_mK=1.0, tau_s=1e-14,
                     carrier_type='n', output='ZT_vs_T.png',
                     doping_levels_cm3=None):
    """
    Estimate ZT = S^2 * sigma * T / (kappa_e + kappa_lat).

    Requires external inputs:
    - tau_s: relaxation time in seconds (typical: 1e-14 s for semiconductors)
    - kappa_lattice_W_mK: lattice thermal conductivity in W/(m*K)
    """
    if doping_levels_cm3 is None:
        doping_levels_cm3 = [1e18, 1e19, 1e20, 1e21]

    fig, ax = plt.subplots(figsize=(8, 5))
    colors = plt.cm.Set2(np.linspace(0.1, 0.9, len(doping_levels_cm3)))

    for i, n in enumerate(doping_levels_cm3):
        ext = extract_at_doping(results, n, carrier_type)
        T = ext['temperatures']
        S = ext['seebeck']
        sigma_abs = ext['sigma_tau'] * tau_s       # 1/(Ohm*m)
        kappa_e = ext['kappa_tau'] * tau_s          # W/(m*K)
        kappa_total = kappa_e + kappa_lattice_W_mK
        ZT = S**2 * sigma_abs * T / kappa_total

        label = f'n = {n:.0e} cm$^{{-3}}$'
        ax.plot(T, ZT, '-o', color=colors[i], linewidth=2, markersize=4,
                label=label)

    ax.set_xlabel('Temperature (K)', fontsize=13)
    ax.set_ylabel('ZT', fontsize=13)
    ax.set_title(
        f'Thermoelectric Figure of Merit ({carrier_type}-type)\n'
        f'$\\tau$ = {tau_s:.0e} s, '
        f'$\\kappa_{{lat}}$ = {kappa_lattice_W_mK} W/(m$\\cdot$K)',
        fontsize=12)
    ax.legend(fontsize=9, loc='best')
    ax.grid(True, alpha=0.3)
    ax.set_ylim(bottom=0)
    plt.tight_layout()
    plt.savefig(output, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output}")


def plot_transport_summary(results, carrier_type='n', n_dop=1e19,
                            output='transport_summary.png'):
    """4-panel summary: Seebeck, sigma/tau, kappa_e/tau, PF/tau vs T."""
    ext = extract_at_doping(results, n_dop, carrier_type)
    T = ext['temperatures']

    fig, axes = plt.subplots(2, 2, figsize=(12, 9))

    # Seebeck
    ax = axes[0, 0]
    ax.plot(T, ext['seebeck'] * 1e6, 'b-o', linewidth=2, markersize=4)
    ax.set_ylabel(r'Seebeck ($\mu$V/K)', fontsize=12)
    ax.set_xlabel('Temperature (K)', fontsize=11)
    ax.axhline(y=0, color='k', linewidth=0.5)
    ax.grid(True, alpha=0.3)

    # sigma/tau
    ax = axes[0, 1]
    ax.plot(T, ext['sigma_tau'], 'r-o', linewidth=2, markersize=4)
    ax.set_ylabel(r'$\sigma/\tau$ (1/$\Omega\cdot$m$\cdot$s)', fontsize=12)
    ax.set_xlabel('Temperature (K)', fontsize=11)
    ax.set_yscale('log')
    ax.grid(True, alpha=0.3)

    # kappa_e/tau
    ax = axes[1, 0]
    ax.plot(T, ext['kappa_tau'], 'g-o', linewidth=2, markersize=4)
    ax.set_ylabel(r'$\kappa_e/\tau$ (W/m$\cdot$K$\cdot$s)', fontsize=12)
    ax.set_xlabel('Temperature (K)', fontsize=11)
    ax.set_yscale('log')
    ax.grid(True, alpha=0.3)

    # Power factor / tau
    ax = axes[1, 1]
    ax.plot(T, ext['power_factor_tau'], 'm-o', linewidth=2, markersize=4)
    ax.set_ylabel(r'$S^2\sigma/\tau$ (W/m$\cdot$K$^2\cdot$s)', fontsize=12)
    ax.set_xlabel('Temperature (K)', fontsize=11)
    ax.grid(True, alpha=0.3)

    fig.suptitle(
        f'Transport Summary ({carrier_type}-type, '
        f'n = {n_dop:.0e} cm$^{{-3}}$)',
        fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output}")


# =====================================================================
# Part 4: Main workflow
# =====================================================================

def main():
    print("=" * 60)
    print("Boltzmann Transport via BoltzTraP2")
    print("=" * 60)

    prefix = 'si'
    outdir = './tmp_transport/'

    # ── Run BoltzTraP2 ────────────────────────────────────────────
    results = run_bolztrap2_from_qe(
        prefix=prefix,
        outdir=outdir,
        temp_range=(100, 1000, 19),   # 100 to 1000 K, 19 points
        mu_window_eV=1.5,
        n_mu=2000,
    )

    # ── Print key results at 300 K ────────────────────────────────
    print("\n" + "=" * 60)
    print("Results at 300 K")
    print("=" * 60)
    T_idx = np.argmin(np.abs(results['temperatures'] - 300))
    actual_T = results['temperatures'][T_idx]
    print(f"Temperature: {actual_T:.0f} K\n")

    for carrier_type in ['n', 'p']:
        print(f"  {carrier_type.upper()}-type:")
        print(f"  {'n (cm^-3)':>12}  {'S (uV/K)':>10}  "
              f"{'sigma/tau':>12}  {'PF/tau':>12}")
        print("  " + "-" * 52)
        for n_dop in [1e18, 1e19, 1e20, 1e21]:
            ext = extract_at_doping(results, n_dop, carrier_type)
            S = ext['seebeck'][T_idx] * 1e6
            sig = ext['sigma_tau'][T_idx]
            PF = ext['power_factor_tau'][T_idx]
            print(f"  {n_dop:>12.2e}  {S:>10.1f}  {sig:>12.3e}  {PF:>12.3e}")
        print()

    # ── Generate plots ────────────────────────────────────────────
    print("--- Generating plots ---")
    doping_levels = [1e18, 5e18, 1e19, 5e19, 1e20]

    for ct in ['n', 'p']:
        plot_seebeck_vs_temperature(results, doping_levels, ct,
                                     f'seebeck_vs_T_{ct}type.png')
        plot_sigma_vs_temperature(results, doping_levels, ct,
                                   f'sigma_vs_T_{ct}type.png')
        plot_kappa_vs_temperature(results, doping_levels, ct,
                                   f'kappa_e_vs_T_{ct}type.png')
        plot_power_factor(results, doping_levels, ct,
                          f'power_factor_vs_T_{ct}type.png')
        plot_seebeck_vs_doping(results, [300, 500, 700], ct,
                                f'seebeck_vs_doping_{ct}type.png')
        plot_transport_summary(results, ct, 1e19,
                                f'transport_summary_{ct}type.png')

        # ZT estimate (requires external kappa_lat and tau)
        plot_zt_estimate(results, kappa_lattice_W_mK=10.0, tau_s=1e-14,
                         carrier_type=ct, output=f'ZT_vs_T_{ct}type.png',
                         doping_levels_cm3=doping_levels)

    # ── Save numerical results ────────────────────────────────────
    output_data = {
        'temperatures_K': results['temperatures'].tolist(),
        'fermi_Ha': float(results['fermi']),
        'volume_m3': float(results['volume_m3']),
    }
    for ct in ['n', 'p']:
        ct_data = {}
        for n_dop in [1e18, 1e19, 1e20, 1e21]:
            ext = extract_at_doping(results, n_dop, ct)
            ct_data[f'{n_dop:.0e}'] = {
                'seebeck_V_K': ext['seebeck'].tolist(),
                'sigma_tau': ext['sigma_tau'].tolist(),
                'kappa_tau': ext['kappa_tau'].tolist(),
                'power_factor_tau': ext['power_factor_tau'].tolist(),
            }
        output_data[f'{ct}_type'] = ct_data

    with open('transport_results.json', 'w') as f:
        json.dump(output_data, f, indent=2)
    print("\nResults saved: transport_results.json")

    print("\n" + "=" * 60)
    print("Transport calculation complete!")
    print("=" * 60)


if __name__ == '__main__':
    main()
```

---

### Method B: VASP + BoltzTraP2

#### Step B1: VASP SCF with Dense K-Mesh

Prepare the VASP INCAR for a static calculation with a dense k-mesh:

```
# INCAR for transport (dense k-mesh SCF)
SYSTEM  = Si_transport
ENCUT   = 400
EDIFF   = 1E-6
ISMEAR  = -5       # Tetrahedron method for accurate eigenvalues
SIGMA   = 0.05
LORBIT  = 11       # Write DOSCAR and PROCAR
LWAVE   = .TRUE.   # Write WAVECAR (needed if doing NSCF later)
LCHARG  = .TRUE.   # Write CHGCAR
NBANDS  = 16       # Include enough empty bands
ISYM    = -1       # Disable symmetry (needed for BoltzTraP2)
PREC    = Accurate
```

KPOINTS file for dense grid:

```
Automatic mesh
0
Gamma
40 40 40
0  0  0
```

**Important**: `ISYM = -1` disables symmetry, giving BoltzTraP2 the full BZ.

#### Step B2: BoltzTraP2 from VASP EIGENVAL

```python
#!/usr/bin/env python3
"""
Run BoltzTraP2 using VASP EIGENVAL output.
BoltzTraP2 can read VASP data via its built-in interface.
"""
import os
import sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

try:
    import BoltzTraP2
    import BoltzTraP2.dft
    import BoltzTraP2.bandlib
    import BoltzTraP2.fite
except ImportError:
    os.system("pip install BoltzTraP2")
    import BoltzTraP2
    import BoltzTraP2.dft
    import BoltzTraP2.bandlib
    import BoltzTraP2.fite


def run_bolztrap2_from_vasp(vasp_dir='./', temp_range=(100, 1000, 19),
                             mu_window_eV=1.5, n_mu=2000):
    """
    Load VASP eigenvalues and run BoltzTraP2.

    Parameters
    ----------
    vasp_dir : str
        Directory containing VASP output files (EIGENVAL, POSCAR, etc.).
    temp_range, mu_window_eV, n_mu : same as QE version.

    Returns
    -------
    results : dict (same format as QE version).
    """
    if not os.path.isfile(os.path.join(vasp_dir, 'EIGENVAL')):
        print(f"ERROR: EIGENVAL not found in {vasp_dir}")
        sys.exit(1)

    print(f"Loading VASP data from: {vasp_dir}")

    # BoltzTraP2 reads VASP data from a directory containing
    # EIGENVAL, POSCAR (or CONTCAR), and optionally DOSCAR
    data = BoltzTraP2.dft.DFTData(vasp_dir, derivatives=False)

    print(f"  k-points: {data.kpoints.shape[0]}")
    print(f"  bands:    {data.ebands.shape[1]}")
    print(f"  Fermi:    {data.fermi:.6f} Ha")

    # Fourier interpolation
    print("\nFourier interpolation ...")
    equivalences = BoltzTraP2.fite.equivalences(data, magmom=None)
    coeffs = BoltzTraP2.fite.fitde3D(data, equivalences)

    # Temperature and mu grids
    T_min, T_max, n_T = temp_range
    temperatures = np.linspace(T_min, T_max, int(n_T))

    eV_to_Ha = 1.0 / 27.2114
    mu_min = data.fermi - mu_window_eV * eV_to_Ha
    mu_max = data.fermi + mu_window_eV * eV_to_Ha
    mu_array = np.linspace(mu_min, mu_max, n_mu)

    # Transport DOS
    volume_m3 = data.atoms.get_volume() * 1e-30

    LMTC = BoltzTraP2.fite.BTPDOS(
        data.ebands,
        data.mommat if hasattr(data, 'mommat') and data.mommat is not None else None,
        data.kpoints,
        temperatures,
        mu_array,
        volume_m3,
        equivalences,
        coeffs
    )
    epsilon, dos, vvdos, cdos = LMTC

    sigma, seebeck, kappa, hall = BoltzTraP2.bandlib.calc_Onsager_coefficients(
        epsilon, dos, vvdos, temperatures, mu_array, volume_m3
    )

    carrier_conc = BoltzTraP2.bandlib.calc_N(
        epsilon, dos, temperatures, mu_array, volume_m3
    )

    results = {
        'temperatures': temperatures,
        'mu_array': mu_array,
        'fermi': data.fermi,
        'sigma': sigma,
        'seebeck': seebeck,
        'kappa': kappa,
        'carrier_conc': carrier_conc,
        'hall': hall,
        'volume_m3': volume_m3,
    }

    print("  VASP transport coefficients computed.")
    return results


# ── Usage ──────────────────────────────────────────────────────────
if __name__ == '__main__':
    results = run_bolztrap2_from_vasp(
        vasp_dir='./',
        temp_range=(100, 1000, 19),
    )
    # Use the same plotting functions from Method A (copy them or import)
    print("Done. Use plotting functions to visualize results.")
```

#### Step B3: Alternative -- pymatgen BoltzTraP2 Interface for VASP

```python
#!/usr/bin/env python3
"""
Use pymatgen's BoltzTraP2 interface with VASP vasprun.xml.
This is often the most convenient approach for VASP users.
"""
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

try:
    from pymatgen.electronic_structure.boltztrap2 import (
        VasprunBSLoader,
        BztTransportProperties,
        BztInterpolator,
    )
except ImportError:
    print("pymatgen BoltzTraP2 interface not available.")
    print("Install: pip install pymatgen BoltzTraP2")
    raise

from pymatgen.io.vasp.outputs import Vasprun


def transport_via_pymatgen(vasprun_path='vasprun.xml'):
    """
    Compute transport properties from VASP vasprun.xml via pymatgen.

    Parameters
    ----------
    vasprun_path : str
        Path to VASP vasprun.xml with dense k-mesh eigenvalues.
    """
    print(f"Loading {vasprun_path} ...")
    vrun = Vasprun(vasprun_path, parse_projected_eigen=False)

    # Load band structure
    loader = VasprunBSLoader(vrun)
    print(f"  k-points: {loader.kpoints.shape[0]}")
    print(f"  bands:    {loader.ebands.shape[1]}")

    # Interpolate
    print("Interpolating bands ...")
    interp = BztInterpolator(loader, energy_range=1.5)  # +/- 1.5 eV

    # Compute transport
    print("Computing transport properties ...")
    transport = BztTransportProperties(
        interp,
        temp_r=np.arange(100, 1001, 50),  # Temperature range
    )

    # Access results
    # transport.seebeck  -- shape (n_T, n_mu, 3, 3)
    # transport.cond     -- sigma/tau
    # transport.kappa    -- kappa_e/tau
    # transport.mu_r     -- chemical potential array

    print(f"  Temperatures: {transport.temp_r[0]} to {transport.temp_r[-1]} K")
    print(f"  Chemical potential points: {len(transport.mu_r)}")

    # Print Seebeck at 300 K, Fermi level
    T_idx = np.argmin(np.abs(transport.temp_r - 300))
    mu_idx = len(transport.mu_r) // 2  # roughly at Fermi level
    S = np.trace(transport.seebeck[T_idx, mu_idx]) / 3.0 * 1e6
    print(f"\n  Seebeck at 300 K (Fermi level): {S:.1f} uV/K")

    return transport


if __name__ == '__main__':
    transport = transport_via_pymatgen('vasprun.xml')
```

---

## Key Parameters

| Parameter | Typical Value | Notes |
|---|---|---|
| K-grid (QE NSCF) | 40x40x40 | Transport requires much denser grids than total energy. Test convergence. |
| `nosym` (QE) | `.true.` | **Mandatory** for BoltzTraP2. Full BZ needed, not irreducible wedge. |
| `ISYM` (VASP) | `-1` | VASP equivalent: disable symmetry for full BZ. |
| `nbnd` / `NBANDS` | >= 2x occupied | Must cover +/- 1.5 eV around Fermi level. |
| `occupations` (QE) | `'smearing'` | Use smearing for convergence; BoltzTraP2 handles Fermi-Dirac internally. |
| `ISMEAR` (VASP) | `-5` | Tetrahedron method for accurate eigenvalues. |
| `conv_thr` (QE) | `1.0d-10` | Tight SCF convergence for accurate eigenvalues. |
| Temperature range | 100--1000 K | Standard for thermoelectrics. Use 10--300 K for low-T studies. |
| mu window | +/- 1.5 eV | Chemical potential scan range around Fermi level. |
| tau (relaxation time) | 1e-15 to 1e-13 s | External input; needed to convert sigma/tau to sigma. |
| kappa_lat | Material-specific | Needed for ZT; obtain from phonon calculation or experiment. |

### Convergence Testing for Transport

Transport properties are more sensitive to k-grid density than total energies. Test convergence by plotting the Seebeck coefficient at T=300 K for a fixed doping level (e.g., n = 1e19 cm^-3) as a function of k-grid:

| System Type | Minimum k-grid | Recommended | Notes |
|---|---|---|---|
| Simple semiconductor (Si, GaAs) | 30x30x30 | 40x40x40 | Seebeck converges first; sigma needs denser |
| Complex unit cell (Bi2Te3) | 25x25x25 | 35x35x35 | Larger cell = fewer k/direction needed |
| Metal (Cu, Al) | 40x40x40 | 50x50x50 | Sharp Fermi surface features |
| Semimetal (Bi, graphene) | 40x40x40 | 60x60x60 | Dirac cone needs fine sampling |

## Interpreting Results

### Seebeck Coefficient (S)

- **Sign**: Negative for n-type (electron) carriers, positive for p-type (hole) carriers.
- **Magnitude**: Good thermoelectrics have |S| > 100--200 uV/K. Metals typically have |S| < 50 uV/K.
- **Temperature dependence**: For semiconductors, |S| typically increases with T at low T, then saturates or decreases at high T (bipolar conduction).
- **Doping dependence**: |S| decreases with increasing carrier concentration (more metallic). This is the fundamental tradeoff in thermoelectric optimization.
- **BoltzTraP2 accuracy**: S is exact within CRTA because the relaxation time cancels in the ratio L1/L0.

### Electrical Conductivity (sigma/tau)

- BoltzTraP2 gives sigma/tau (conductivity divided by relaxation time).
- To get absolute conductivity, multiply by tau from experiment, electron-phonon calculation, or modeling.
- **Typical tau values**: 1e-15 to 1e-14 s for semiconductors at room temperature; 1e-14 to 1e-13 s for metals.
- sigma/tau increases with carrier concentration.

### Electronic Thermal Conductivity (kappa_e/tau)

- Related to sigma by the Wiedemann-Franz law: kappa_e = L * sigma * T, where L ~ 2.44e-8 W*Ohm/K^2 (Lorenz number for metals).
- Needed for ZT computation.

### Power Factor (S^2 * sigma)

- Peaks at intermediate doping because S decreases while sigma increases with doping.
- Optimal thermoelectric doping is typically 1e19 to 1e20 cm^-3.
- BoltzTraP2 gives PF/tau; multiply by tau for absolute values.

### ZT (Figure of Merit)

```
ZT = S^2 * sigma * T / (kappa_e + kappa_lat)
```

- ZT > 1 is considered good for thermoelectric applications; ZT > 2 is excellent.
- Computing ZT requires three external inputs: tau, kappa_lat (from phonon calculation), and the temperature.
- Compare n-type and p-type to determine which carrier type gives better performance.

### Typical Values for Validation

| Material | T (K) | Carrier | n (cm^-3) | S (uV/K) | Reference |
|---|---|---|---|---|---|
| Si | 300 | n | 1e19 | -200 to -300 | Experiment |
| Bi2Te3 | 300 | n | 1e19 | -180 to -220 | Experiment |
| PbTe | 300 | p | 1e19 | +200 to +250 | Experiment |
| SnSe | 700 | p | ~1e19 | +300 to +400 | Experiment |

## Common Issues

| Problem | Solution |
|---|---|
| **BoltzTraP2 ImportError** | `pip install BoltzTraP2`. If compilation fails, try `CC=gcc pip install BoltzTraP2` or `pip install --no-binary :all: BoltzTraP2`. |
| **BoltzTraP2 cannot read QE data** | Ensure NSCF completed fully. Check `outdir/prefix.save/data-file-schema.xml` exists. Verify `nosym = .true.` was set in NSCF input. |
| **Seebeck is noisy or wrong sign** | K-grid too coarse (increase to 40x40x40+). Insufficient bands (increase `nbnd`). n-type has S < 0, p-type S > 0 by convention. |
| **Transport does not converge with k-grid** | Try 30, 35, 40, 45, 50 per direction. For anisotropic systems, use anisotropic grids (e.g., 20x20x60 for layered materials). |
| **sigma/tau vs absolute sigma** | BoltzTraP2 only gives sigma/tau. Extract tau from experiment or electron-phonon coupling calculation. For Seebeck and power factor ratios, tau cancels. |
| **NSCF too slow (40^3 with nosym)** | 64,000 k-points is expensive. Use more MPI processes and `-npool N`. Reduce ecutwfc if possible. Reduce nbnd to minimum needed. |
| **Memory errors** | Distribute with more MPI ranks. Use `disk_io = 'low'` in QE. Reduce bands or k-points. |
| **BoltzTraP2 crash during interpolation** | Usually too few k-points or symmetry issues. Verify `nosym = .true.`. Try a coarser grid first to test. |
| **Bipolar conduction at high T** | Both electrons and holes contribute, partially canceling S. This is physical (not an error) for narrow-gap semiconductors at high T. |
| **Comparing with experiment** | CRTA gives exact S but only sigma/tau. Temperature dependence of sigma may not match because tau(T) varies. For better accuracy, use EPW for tau(T). |
