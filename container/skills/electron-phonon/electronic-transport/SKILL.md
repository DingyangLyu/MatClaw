# Electronic Transport Properties

## When to Use

- You need **thermoelectric transport coefficients**: Seebeck coefficient (thermopower), electrical conductivity, electronic thermal conductivity.
- You want to compute the **thermoelectric figure of merit ZT** or the **power factor**.
- You need to evaluate how transport properties vary with **temperature** and **carrier concentration** (doping level).
- You want to identify **optimal doping** for thermoelectric performance.
- You are studying metals, semimetals, or doped semiconductors.

This skill uses the **Boltzmann transport equation (BTE) in the constant relaxation-time approximation** via BoltzTraP2. This provides the Seebeck coefficient exactly (relaxation time cancels out) and gives conductivity/thermal conductivity scaled by the relaxation time (sigma/tau, kappa_e/tau).

## Prerequisites

- Quantum ESPRESSO 7.5 (`pw.x`)
- Python with `numpy`, `scipy`, `matplotlib`
- BoltzTraP2 (`pip install BoltzTraP2`)
- Appropriate pseudopotentials
- A very dense k-grid (40x40x40 or denser) for reliable transport properties

## Detailed Steps

The workflow is:

```
SCF (pw.x)  -->  NSCF (pw.x, dense k-grid)  -->  BoltzTraP2 (Python)  -->  Plots
  coarse k          very dense k                  Seebeck, sigma/tau,
                    (40x40x40+)                   kappa_e/tau, PF, ZT
```

### Step 1: SCF Calculation (pw.x)

Standard self-consistent calculation to establish the ground-state charge density.

Create `scf.in`:

```
cat > scf.in << 'PWSCF_INPUT'
&CONTROL
    calculation   = 'scf'
    prefix        = 'si'
    pseudo_dir    = './pseudo/'
    outdir        = './tmp/'
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

ATOMIC_POSITIONS {crystal}
  Si  0.00  0.00  0.00
  Si  0.25  0.25  0.25

K_POINTS {automatic}
  8 8 8  0 0 0
PWSCF_INPUT
```

Notes:
- **`nbnd`**: Include enough empty bands above the Fermi level / valence band maximum. For transport, you need bands in the energy window of interest (typically a few eV around the Fermi level). At least 2x the number of occupied bands.
- **`occupations = 'smearing'`**: Even for semiconductors, use smearing for the SCF step (it helps convergence). BoltzTraP2 will handle the actual Fermi-Dirac statistics.

Run:

```bash
mpirun -np 4 pw.x -npool 2 < scf.in > scf.out 2>&1
grep "convergence has been achieved" scf.out
grep "the Fermi energy is" scf.out
```

### Step 2: NSCF Calculation on Dense K-Grid (pw.x)

The NSCF step computes eigenvalues on a very dense, uniform k-grid. This is the critical step for transport -- the k-grid must be dense enough to capture the band structure features that determine transport.

Create `nscf.in`:

```
cat > nscf.in << 'PWSCF_INPUT'
&CONTROL
    calculation   = 'nscf'
    prefix        = 'si'
    pseudo_dir    = './pseudo/'
    outdir        = './tmp/'
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

ATOMIC_POSITIONS {crystal}
  Si  0.00  0.00  0.00
  Si  0.25  0.25  0.25

K_POINTS {automatic}
  40 40 40  0 0 0
PWSCF_INPUT
```

**Critical parameters**:

- **K_POINTS 40 40 40**: This is the dense k-grid. Transport properties require much denser grids than total energy or DOS calculations. Convergence test: try 30x30x30, 40x40x40, 50x50x50 and check if the Seebeck coefficient changes by <5%.
- **`nosym = .true.`**: Disables symmetry reduction of the k-point set. BoltzTraP2 needs the full grid, not just the irreducible wedge. **This is mandatory** for BoltzTraP2 to work correctly.
- **`nbnd = 16`**: Same as SCF. Must include enough bands to cover the energy window of interest.
- **`calculation = 'nscf'`**: Non-self-consistent -- reads the charge density from the SCF step and just computes eigenvalues.

Run:

```bash
mpirun -np 4 pw.x -npool 4 < nscf.in > nscf.out 2>&1
grep "convergence has been achieved" nscf.out
```

Note: For `nscf` calculations, "convergence has been achieved" refers to the eigenvalue convergence at each k-point (diagonalization), not SCF convergence.

### Step 3: Install BoltzTraP2

```bash
pip install BoltzTraP2
```

BoltzTraP2 can read QE output directly via its `DFTtransport` interface or through ASE/pymatgen. The most robust approach is to use the built-in QE interface.

### Step 4: BoltzTraP2 Transport Calculation (Python)

```python
#!/usr/bin/env python3
"""
bolztrap2_transport.py
Complete BoltzTraP2 workflow: load QE data, compute transport, plot results.

Usage: python3 bolztrap2_transport.py
"""
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os
import sys
import json

# =============================================================================
# Part 1: Load QE data and run BoltzTraP2
# =============================================================================

def run_bolztrap2_from_qe(prefix='si', outdir='./tmp/',
                          temperature_range=None,
                          doping_levels=None):
    """
    Run BoltzTraP2 using QE output.

    Parameters
    ----------
    prefix : str
        QE calculation prefix.
    outdir : str
        QE output directory.
    temperature_range : list
        [T_min, T_max, n_points] in Kelvin.
    doping_levels : list
        Carrier concentrations in cm^-3 (e.g., [1e18, 1e19, 1e20]).

    Returns
    -------
    results : dict
        Dictionary containing all computed transport properties.
    """
    try:
        import BoltzTraP2
        import BoltzTraP2.dft
        import BoltzTraP2.bandlib
        import BoltzTraP2.fite
    except ImportError:
        print("BoltzTraP2 not installed. Installing...")
        os.system("pip install BoltzTraP2")
        import BoltzTraP2
        import BoltzTraP2.dft
        import BoltzTraP2.bandlib
        import BoltzTraP2.fite

    if temperature_range is None:
        temperature_range = [100, 1000, 10]  # 100 to 1000 K, 10 points

    if doping_levels is None:
        # Typical range for thermoelectric optimization
        doping_levels = np.logspace(17, 21, 50)  # 1e17 to 1e21 cm^-3

    # ---- Load QE data ----
    # BoltzTraP2 can load from a directory containing QE output
    # It looks for prefix.save/data-file-schema.xml (QE >= 6.x)
    data_dir = os.path.join(outdir, f'{prefix}.save')

    if not os.path.isdir(data_dir):
        print(f"ERROR: {data_dir} not found.")
        print("Make sure the NSCF calculation completed successfully.")
        sys.exit(1)

    print(f"Loading QE data from: {data_dir}")

    # Load the DFT data
    data = BoltzTraP2.dft.DFTData(data_dir)

    print(f"  Number of k-points: {data.kpoints.shape[0]}")
    print(f"  Number of bands: {data.ebands.shape[1]}")
    print(f"  Fermi energy: {data.fermi:.6f} Ha")

    # ---- Perform BoltzTraP2 interpolation ----
    print("\nRunning BoltzTraP2 interpolation...")

    # equivalences: group k-points by symmetry (for interpolation)
    equivalences = BoltzTraP2.fite.equivalences(data, magmom=None)

    # Fit the bands with star functions
    coeffs = BoltzTraP2.fite.fitde3D(data, equivalences)

    print("  Interpolation complete.")
    print(f"  Number of star functions: {len(equivalences)}")

    # ---- Compute transport coefficients ----
    print("\nComputing transport coefficients...")

    # Temperature array
    T_min, T_max, n_T = temperature_range
    temperatures = np.linspace(T_min, T_max, int(n_T))

    # Chemical potential array -- scan around the Fermi level
    # Energy window: typically +/- 1 eV around Fermi level
    eV_to_Ha = 1.0 / 27.2114
    mu_min = data.fermi - 1.5 * eV_to_Ha
    mu_max = data.fermi + 1.5 * eV_to_Ha
    n_mu = 2000
    mu_array = np.linspace(mu_min, mu_max, n_mu)

    # Compute the Onsager transport coefficients
    # This returns L0, L1, L2 (the three Onsager coefficients)
    # sigma/tau = L0
    # S = (1/eT) * L1/L0  (Seebeck coefficient)
    # kappa_e/tau = (1/T) * (L2 - L1^2/L0)

    LMTC = BoltzTraP2.fite.BTPDOS(
        data.ebands,
        data.mommat if hasattr(data, 'mommat') and data.mommat is not None else None,
        data.kpoints,
        temperatures,
        mu_array,
        data.atoms.get_volume() * 1e-30,  # Volume in m^3 (ASE uses Angstrom^3)
        equivalences,
        coeffs
    )

    # LMTC is a tuple: (epsilon, dos, vvdos, cdos)
    # epsilon: energy grid
    # dos: density of states
    # vvdos: velocity-velocity DOS (for transport)
    # cdos: cumulative DOS

    epsilon, dos, vvdos, cdos = LMTC

    # Now compute the actual transport coefficients
    # using BoltzTraP2.bandlib

    # sigma/tau, Seebeck, kappa_e/tau as functions of (T, mu)
    # Units: sigma/tau in 1/(Ohm*m*s), S in V/K, kappa/tau in W/(m*K*s)

    sigma, seebeck, kappa, hall = BoltzTraP2.bandlib.calc_Onsager_coefficients(
        epsilon, dos, vvdos, temperatures, mu_array,
        data.atoms.get_volume() * 1e-30
    )

    # Carrier concentration as function of (T, mu)
    # n(T, mu) = integral f(E, T, mu) * DOS(E) dE - n_electrons
    carrier_conc = BoltzTraP2.bandlib.calc_N(
        epsilon, dos, temperatures, mu_array,
        data.atoms.get_volume() * 1e-30
    )

    results = {
        'temperatures': temperatures,
        'mu_array': mu_array,
        'fermi': data.fermi,
        'sigma': sigma,           # shape: (n_T, n_mu, 3, 3)
        'seebeck': seebeck,       # shape: (n_T, n_mu, 3, 3)
        'kappa': kappa,           # shape: (n_T, n_mu, 3, 3)
        'carrier_conc': carrier_conc,  # shape: (n_T, n_mu)
        'hall': hall,             # Hall coefficient
        'volume_m3': data.atoms.get_volume() * 1e-30,
        'doping_levels': doping_levels,
    }

    print("  Transport coefficients computed.")
    return results


# =============================================================================
# Part 2: Alternative approach using pymatgen interface
# =============================================================================

def run_bolztrap2_via_pymatgen(prefix='si', outdir='./tmp/',
                               temperature_range=None):
    """
    Alternative: Use pymatgen's BoltzTraP2 interface.
    This is sometimes more convenient but requires pymatgen.
    """
    from pymatgen.io.espresso.outputs import PWOutput
    from pymatgen.electronic_structure.boltztrap2 import VasprunBSLoader, BztTransportProperties

    # Note: pymatgen's BoltzTraP2 interface was designed for VASP
    # but can work with QE via intermediate conversion.
    # The direct BoltzTraP2.dft approach (above) is more reliable for QE.

    print("For QE data, the direct BoltzTraP2.dft approach is recommended.")
    print("See run_bolztrap2_from_qe() function.")
    return None


# =============================================================================
# Part 3: Extract transport at specific doping levels
# =============================================================================

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
    dict with Seebeck, sigma/tau, kappa/tau at each temperature.
    """
    temperatures = results['temperatures']
    mu_array = results['mu_array']
    fermi = results['fermi']
    carrier_conc = results['carrier_conc']  # shape: (n_T, n_mu)
    seebeck = results['seebeck']
    sigma = results['sigma']
    kappa = results['kappa']

    # Convert target concentration to m^-3
    target_conc_m3 = target_conc_cm3 * 1e6

    extracted = {
        'temperatures': temperatures,
        'seebeck': [],        # V/K
        'sigma_tau': [],      # 1/(Ohm*m*s)
        'kappa_tau': [],      # W/(m*K*s)
        'power_factor_tau': [],  # W/(m*K^2*s)
    }

    for iT, T in enumerate(temperatures):
        conc = carrier_conc[iT, :]

        if carrier_type == 'n':
            # n-type: look for mu > fermi where carrier_conc is positive
            mask = mu_array >= fermi
        else:
            # p-type: look for mu < fermi
            mask = mu_array <= fermi

        # Find mu closest to target concentration
        conc_diff = np.abs(np.abs(conc) - target_conc_m3)
        conc_diff_masked = np.where(mask, conc_diff, np.inf)
        idx = np.argmin(conc_diff_masked)

        # Extract isotropic average (trace/3) of tensor quantities
        S = np.trace(seebeck[iT, idx, :, :]) / 3.0  # V/K
        sig = np.trace(sigma[iT, idx, :, :]) / 3.0   # 1/(Ohm*m*s)
        kap = np.trace(kappa[iT, idx, :, :]) / 3.0   # W/(m*K*s)
        PF = S**2 * sig  # Power factor / tau

        extracted['seebeck'].append(S)
        extracted['sigma_tau'].append(sig)
        extracted['kappa_tau'].append(kap)
        extracted['power_factor_tau'].append(PF)

    # Convert to numpy arrays
    for key in ['seebeck', 'sigma_tau', 'kappa_tau', 'power_factor_tau']:
        extracted[key] = np.array(extracted[key])

    return extracted


# =============================================================================
# Part 4: Plotting functions
# =============================================================================

def plot_seebeck_vs_temperature(results, doping_levels_cm3=None,
                                 carrier_type='n', output='seebeck_vs_T.png'):
    """
    Plot Seebeck coefficient vs temperature for several doping levels.
    """
    if doping_levels_cm3 is None:
        doping_levels_cm3 = [1e18, 1e19, 1e20, 1e21]

    fig, ax = plt.subplots(figsize=(8, 5))

    colors = plt.cm.viridis(np.linspace(0.2, 0.9, len(doping_levels_cm3)))

    for i, n in enumerate(doping_levels_cm3):
        ext = extract_at_doping(results, n, carrier_type)
        S_uV_K = ext['seebeck'] * 1e6  # Convert V/K to uV/K
        label = f'{carrier_type}-type, n = {n:.0e} cm$^{{-3}}$'
        ax.plot(ext['temperatures'], S_uV_K, '-o', color=colors[i],
                linewidth=2, markersize=4, label=label)

    ax.set_xlabel('Temperature (K)', fontsize=13)
    ax.set_ylabel(r'Seebeck coefficient ($\mu$V/K)', fontsize=13)
    ax.set_title(f'Seebeck Coefficient vs Temperature ({carrier_type}-type)', fontsize=14)
    ax.legend(fontsize=9, loc='best')
    ax.grid(True, alpha=0.3)
    ax.axhline(y=0, color='k', linewidth=0.5)

    plt.tight_layout()
    plt.savefig(output, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output}")


def plot_sigma_vs_temperature(results, doping_levels_cm3=None,
                               carrier_type='n', output='sigma_vs_T.png'):
    """
    Plot electrical conductivity / tau vs temperature.
    """
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
    ax.set_title(f'Electrical Conductivity / $\\tau$ ({carrier_type}-type)', fontsize=14)
    ax.legend(fontsize=9, loc='best')
    ax.grid(True, alpha=0.3)
    ax.set_yscale('log')

    plt.tight_layout()
    plt.savefig(output, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output}")


def plot_power_factor(results, doping_levels_cm3=None,
                      carrier_type='n', output='power_factor_vs_T.png'):
    """
    Plot power factor S^2*sigma/tau vs temperature.
    """
    if doping_levels_cm3 is None:
        doping_levels_cm3 = [1e18, 1e19, 1e20, 1e21]

    fig, ax = plt.subplots(figsize=(8, 5))

    colors = plt.cm.inferno(np.linspace(0.2, 0.85, len(doping_levels_cm3)))

    for i, n in enumerate(doping_levels_cm3):
        ext = extract_at_doping(results, n, carrier_type)
        label = f'n = {n:.0e} cm$^{{-3}}$'
        ax.plot(ext['temperatures'], ext['power_factor_tau'], '-o', color=colors[i],
                linewidth=2, markersize=4, label=label)

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
                            carrier_type='n', output='seebeck_vs_doping.png'):
    """
    Plot Seebeck coefficient vs carrier concentration at fixed temperatures.
    """
    if temperatures_K is None:
        temperatures_K = [300, 500, 700, 900]

    fig, ax = plt.subplots(figsize=(8, 5))

    all_temps = results['temperatures']
    colors = plt.cm.coolwarm(np.linspace(0.1, 0.9, len(temperatures_K)))

    doping_scan = np.logspace(17, 21, 100)

    for j, target_T in enumerate(temperatures_K):
        # Find closest temperature index
        iT = np.argmin(np.abs(all_temps - target_T))
        actual_T = all_temps[iT]

        S_values = []
        for n in doping_scan:
            ext = extract_at_doping(results, n, carrier_type)
            S_values.append(ext['seebeck'][iT] * 1e6)  # uV/K

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

    This requires:
    - A relaxation time tau (assumed constant, typical: 1e-14 s for semiconductors)
    - Lattice thermal conductivity kappa_lat (from phonon calculation or experiment)

    Both are external inputs -- BoltzTraP2 only gives sigma/tau and kappa_e/tau.

    Parameters
    ----------
    kappa_lattice_W_mK : float
        Lattice thermal conductivity in W/(m*K). Can be temperature-dependent
        (pass array matching temperatures).
    tau_s : float
        Relaxation time in seconds.
    """
    if doping_levels_cm3 is None:
        doping_levels_cm3 = [1e18, 1e19, 1e20, 1e21]

    fig, ax = plt.subplots(figsize=(8, 5))
    colors = plt.cm.Set2(np.linspace(0.1, 0.9, len(doping_levels_cm3)))

    for i, n in enumerate(doping_levels_cm3):
        ext = extract_at_doping(results, n, carrier_type)
        T = ext['temperatures']
        S = ext['seebeck']                    # V/K
        sigma = ext['sigma_tau'] * tau_s       # 1/(Ohm*m)
        kappa_e = ext['kappa_tau'] * tau_s     # W/(m*K)

        kappa_total = kappa_e + kappa_lattice_W_mK
        ZT = S**2 * sigma * T / kappa_total

        label = f'n = {n:.0e} cm$^{{-3}}$'
        ax.plot(T, ZT, '-o', color=colors[i], linewidth=2, markersize=4, label=label)

    ax.set_xlabel('Temperature (K)', fontsize=13)
    ax.set_ylabel('ZT', fontsize=13)
    ax.set_title(f'Thermoelectric Figure of Merit ({carrier_type}-type)\n'
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


# =============================================================================
# Part 5: Effective Mass from Band Curvature (simpler alternative)
# =============================================================================

def compute_effective_mass_from_qe(prefix='si', outdir='./tmp/'):
    """
    Compute effective mass from band curvature at band extrema.

    This is a simpler alternative to full BoltzTraP2 transport:
    - Fit a parabola to the band edge along high-symmetry directions.
    - m* = hbar^2 / (d^2E/dk^2)

    Less accurate than BoltzTraP2 (assumes parabolic bands) but requires
    no additional software and gives quick estimates.
    """
    from pymatgen.io.espresso.outputs import EspressoOutput

    # For effective mass, you need a band structure calculation
    # along high-symmetry directions (not the dense uniform grid).
    # This is separate from the BoltzTraP2 workflow.

    print("Effective mass calculation from band curvature")
    print("=" * 50)
    print()
    print("To compute effective mass, you need a band structure calculation:")
    print("  1. Run SCF (same as above)")
    print("  2. Run NSCF with K_POINTS along high-symmetry path")
    print("  3. Fit parabola near band extremum")
    print()

    # If we have a bands calculation output, we can compute m*:
    bands_file = os.path.join(outdir, f'{prefix}.save', 'data-file-schema.xml')

    if not os.path.isfile(bands_file):
        print(f"No bands data found at {bands_file}")
        print("Run a 'bands' calculation first (see band structure skill).")
        return None

    # Alternative: compute from dense grid eigenvalues
    # Find band extremum and fit parabola numerically
    print("Using dense k-grid eigenvalues to estimate effective mass...")
    print("(For accurate effective mass, a dedicated bands calculation is preferred)")

    return None


def effective_mass_from_parabolic_fit(k_points_bohr, energies_ha, band_index,
                                       k_index_extremum, direction_indices):
    """
    Fit a parabola to a band near its extremum to extract effective mass.

    Parameters
    ----------
    k_points_bohr : array, shape (nk, 3)
        k-points in units of 2*pi/a (Bohr^-1).
    energies_ha : array, shape (nk, nbnd)
        Band energies in Hartree.
    band_index : int
        Which band to fit (0-indexed).
    k_index_extremum : int
        Index of the k-point at the band extremum.
    direction_indices : list of int
        Indices of k-points along the desired direction through the extremum.

    Returns
    -------
    m_star : float
        Effective mass in units of electron mass.
    """
    hbar_eV_s = 6.582119569e-16   # eV*s
    m_e_kg = 9.1093837015e-31      # kg
    bohr_to_m = 5.29177e-11        # m
    Ha_to_eV = 27.2114             # eV

    k_sel = k_points_bohr[direction_indices]  # shape: (n, 3)
    E_sel = energies_ha[direction_indices, band_index] * Ha_to_eV  # eV

    # Compute distance along k-path from extremum (in 1/m)
    k0 = k_points_bohr[k_index_extremum]
    dk = np.linalg.norm((k_sel - k0) * (2 * np.pi / bohr_to_m), axis=1)  # 1/m

    # Fit parabola: E(k) = E0 + (hbar^2 / 2*m*) * k^2
    # --> E = a + b*k^2, where m* = hbar^2 / (2*b) in SI
    # b in units of eV*m^2

    coeffs = np.polyfit(dk, E_sel, 2)  # coeffs[0]*k^2 + coeffs[1]*k + coeffs[2]
    b = coeffs[0]  # eV * m^2

    # Convert b to SI: eV -> J
    b_SI = b * 1.602176634e-19  # J * m^2

    hbar_SI = 1.054571817e-34  # J*s
    m_star_kg = hbar_SI**2 / (2 * b_SI)
    m_star = m_star_kg / m_e_kg

    return abs(m_star)


# =============================================================================
# Part 6: Main workflow
# =============================================================================

def main():
    print("=" * 60)
    print("Electronic Transport Properties via BoltzTraP2")
    print("=" * 60)

    # Configuration
    prefix = 'si'
    outdir = './tmp/'

    # Temperature range: 100 K to 1000 K
    temperature_range = [100, 1000, 19]  # 19 points: 100, 150, 200, ..., 1000

    # Doping levels for plotting (cm^-3)
    doping_levels_plot = [1e18, 5e18, 1e19, 5e19, 1e20]

    # --- Run BoltzTraP2 ---
    print("\n--- Running BoltzTraP2 ---")
    results = run_bolztrap2_from_qe(
        prefix=prefix,
        outdir=outdir,
        temperature_range=temperature_range,
    )

    # --- Print key results at 300 K ---
    print("\n--- Results at 300 K ---")
    T_idx = np.argmin(np.abs(results['temperatures'] - 300))
    actual_T = results['temperatures'][T_idx]
    print(f"Temperature: {actual_T:.0f} K")

    for carrier_type in ['n', 'p']:
        print(f"\n  {carrier_type.upper()}-type:")
        print(f"  {'n (cm^-3)':>12} {'S (uV/K)':>10} {'sigma/tau':>12} {'PF/tau':>12}")
        print("  " + "-" * 50)
        for n_dop in [1e18, 1e19, 1e20, 1e21]:
            ext = extract_at_doping(results, n_dop, carrier_type)
            S = ext['seebeck'][T_idx] * 1e6  # uV/K
            sig = ext['sigma_tau'][T_idx]
            PF = ext['power_factor_tau'][T_idx]
            print(f"  {n_dop:>12.2e} {S:>10.1f} {sig:>12.3e} {PF:>12.3e}")

    # --- Generate plots ---
    print("\n--- Generating plots ---")

    for ct in ['n', 'p']:
        plot_seebeck_vs_temperature(results, doping_levels_plot, ct,
                                     f'seebeck_vs_T_{ct}type.png')
        plot_sigma_vs_temperature(results, doping_levels_plot, ct,
                                   f'sigma_vs_T_{ct}type.png')
        plot_power_factor(results, doping_levels_plot, ct,
                          f'power_factor_vs_T_{ct}type.png')
        plot_seebeck_vs_doping(results, [300, 500, 700], ct,
                                f'seebeck_vs_doping_{ct}type.png')

        # ZT estimate (requires kappa_lat and tau -- these are external inputs)
        # Example: kappa_lat = 10 W/(m*K) for Si at 300K, tau = 1e-14 s
        plot_zt_estimate(results, kappa_lattice_W_mK=10.0, tau_s=1e-14,
                         carrier_type=ct, output=f'ZT_vs_T_{ct}type.png',
                         doping_levels_cm3=doping_levels_plot)

    print("\n" + "=" * 60)
    print("Transport calculation complete!")
    print("=" * 60)


if __name__ == '__main__':
    main()
```

Run:

```bash
python3 bolztrap2_transport.py
```

### Complete Automated Workflow Script

```bash
#!/bin/bash
# run_transport.sh -- Complete electronic transport workflow
# Usage: bash run_transport.sh

set -e

NP=4
NPOOL_SCF=2
NPOOL_NSCF=4   # Can use more pools for NSCF since no SCF iteration

echo "========================================"
echo "Electronic Transport Workflow"
echo "========================================"

# Step 1: SCF
echo ""
echo "--- Step 1: SCF calculation ---"
mpirun -np $NP pw.x -npool $NPOOL_SCF < scf.in > scf.out 2>&1

if grep -q "convergence has been achieved" scf.out; then
    echo "SCF converged."
    grep "the Fermi energy is" scf.out
else
    echo "ERROR: SCF did not converge!"
    exit 1
fi

# Step 2: NSCF on dense k-grid
echo ""
echo "--- Step 2: NSCF on dense k-grid ---"
echo "This may take a while for 40x40x40 grid..."
mpirun -np $NP pw.x -npool $NPOOL_NSCF < nscf.in > nscf.out 2>&1

if grep -q "convergence has been achieved" nscf.out; then
    echo "NSCF completed."
else
    echo "WARNING: Check nscf.out for errors."
fi

# Step 3: Install BoltzTraP2 if needed
echo ""
echo "--- Step 3: Ensuring BoltzTraP2 is installed ---"
pip install BoltzTraP2 2>/dev/null || pip install BoltzTraP2

# Step 4: Run transport analysis
echo ""
echo "--- Step 4: BoltzTraP2 transport analysis ---"
python3 bolztrap2_transport.py

echo ""
echo "========================================"
echo "Workflow complete!"
echo "========================================"
```

## Key Parameters

### K-Grid Density (Most Critical Parameter)

| System Type | Minimum k-grid | Recommended | Notes |
|---|---|---|---|
| Simple semiconductor (Si, GaAs) | 30x30x30 | 40x40x40 | Seebeck converges first; sigma needs denser |
| Complex semiconductor (Bi2Te3) | 25x25x25 | 35x35x35 | Larger unit cell = fewer k per direction needed |
| Metal (Cu, Al) | 40x40x40 | 50x50x50 | Metals need finer sampling near Fermi surface |
| Semimetal (Bi, graphene) | 40x40x40 | 60x60x60 | Sharp features near Fermi level |

**Convergence test**: Plot the Seebeck coefficient at 300 K for a fixed doping level as a function of k-grid density. It should plateau. Increase until the change is <5%.

### Temperature Range

- Thermoelectrics: 300-1000 K (room temperature to typical operating conditions)
- Low-temperature transport: 10-300 K (may need denser energy mesh)
- Metals: 100-1000 K
- Use at least 10-20 temperature points for smooth curves

### Carrier Concentration

- Light doping: 1e16 - 1e18 cm^-3
- Moderate doping: 1e18 - 1e20 cm^-3
- Heavy doping: 1e20 - 1e21 cm^-3
- Optimal thermoelectric doping: typically 1e19 - 1e20 cm^-3

### BoltzTraP2-Specific Parameters

- **Energy window**: BoltzTraP2 interpolates bands within a window around the Fermi level. Default is usually sufficient (a few eV).
- **Number of star functions**: Controlled internally. More k-points = more star functions = better interpolation.
- **`nosym = .true.`**: MANDATORY in the NSCF input. BoltzTraP2 needs the full Brillouin zone, not just the irreducible wedge.

### Number of Bands (nbnd)

Include enough bands to cover the energy window of interest:
- At minimum: all occupied bands + same number of empty bands
- Better: compute the energy range you need (e.g., +/- 1.5 eV from Fermi level) and include all bands in that range
- Too many bands wastes computational time; too few truncates the transport window

## Interpreting Results

### Seebeck Coefficient

- **Sign**: Negative for n-type (electron) carriers, positive for p-type (hole) carriers.
- **Magnitude**: Good thermoelectrics have |S| > 100-200 uV/K. Metals typically have |S| < 50 uV/K.
- **Temperature dependence**: For semiconductors, |S| typically increases with T at low T (more carriers thermally excited), then saturates or decreases at high T (bipolar conduction).
- **Doping dependence**: |S| decreases with increasing carrier concentration (more metallic behavior). This is the fundamental tradeoff in thermoelectric optimization.

### Electrical Conductivity (sigma/tau)

- BoltzTraP2 gives sigma/tau (conductivity divided by relaxation time).
- To get absolute conductivity, you need tau from experiment, electron-phonon calculation, or modeling.
- **Typical tau values**: 1e-15 to 1e-14 s for semiconductors at room temperature; 1e-14 to 1e-13 s for metals.
- sigma/tau increases with carrier concentration (more carriers = higher conductivity).

### Power Factor (S^2 * sigma)

- The power factor S^2 * sigma peaks at an intermediate doping level because:
  - S decreases with doping (more metallic)
  - sigma increases with doping
  - The product has a maximum
- This optimal doping is typically 1e19 - 1e20 cm^-3 for good thermoelectrics.
- BoltzTraP2 gives PF/tau. Multiply by tau for absolute values.

### ZT (Figure of Merit)

```
ZT = S^2 * sigma * T / (kappa_e + kappa_lat)
```

- ZT > 1 is considered good for thermoelectric applications.
- ZT > 2 is excellent (state of the art).
- **To compute ZT, you need**:
  1. sigma and kappa_e from BoltzTraP2 (both need tau)
  2. kappa_lat from phonon calculation (separate workflow) or experiment
  3. tau from modeling or experiment
- The ratio kappa_e / sigma follows the Wiedemann-Franz law: kappa_e = L * sigma * T, where L is the Lorenz number (~2.44e-8 W*Ohm/K^2 for metals).

### N-Type vs P-Type

- Compare Seebeck coefficients and power factors for both carrier types.
- The carrier type with higher |S| at a given concentration will generally give better thermoelectric performance.
- Band structure asymmetry (different effective masses for electrons vs holes) causes one type to outperform the other.

## Common Issues

### 1. BoltzTraP2 Import Error or Crash

**Symptom**: `ImportError: No module named 'BoltzTraP2'` or crash during interpolation.

**Fixes**:
```bash
pip install --upgrade BoltzTraP2
# If compilation issues:
pip install --no-binary :all: BoltzTraP2
# Or with specific compiler:
CC=gcc pip install BoltzTraP2
```

### 2. BoltzTraP2 Cannot Read QE Data

**Symptom**: Error when loading `data-file-schema.xml`.

**Fixes**:
- Ensure the NSCF calculation completed fully.
- Check that `outdir/prefix.save/data-file-schema.xml` exists.
- QE version must be >= 6.0 (uses XML output format). QE 7.5 is fine.
- Verify `nosym = .true.` was set in the NSCF input.

### 3. Seebeck Coefficient Is Noisy or Wrong Sign

**Causes**:
- K-grid too coarse: Increase from 30x30x30 to 40x40x40 or 50x50x50.
- Insufficient bands: Add more empty bands (increase `nbnd`).
- Wrong carrier type interpretation: n-type has S < 0, p-type has S > 0 by convention.
- Bipolar conduction at high T: Both electrons and holes contribute, partially canceling S. This is physical, not an error.

### 4. Transport Properties Do Not Converge with K-Grid

**Strategy**:
- Plot the Seebeck coefficient at T=300K, n=1e19 cm^-3 for k-grids: 20, 25, 30, 35, 40, 45, 50 per direction.
- If not converged at 50x50x50, the material may have very sharp features near the Fermi level (e.g., van Hove singularities, flat bands). Consider even denser grids.
- For systems with very anisotropic unit cells, use anisotropic k-grids (e.g., 20x20x60 for layered materials).

### 5. Absolute Values of Conductivity

**Problem**: BoltzTraP2 gives sigma/tau, not sigma.

**Solutions**:
- If you have experimental resistivity at one temperature, extract tau = sigma_exp / (sigma_BoltzTraP/tau) at that temperature, then use it for other temperatures (constant relaxation time approximation).
- Compute tau from electron-phonon coupling (see the elph-coupling skill).
- For Seebeck coefficient and power factor ratios, tau cancels out and is not needed.
- The Wiedemann-Franz law relates kappa_e to sigma without needing tau explicitly.

### 6. NSCF Calculation Too Slow

A 40x40x40 grid with `nosym = .true.` produces 64,000 k-points. This can be very slow.

**Mitigations**:
- Use more MPI processes and k-point pools (`-npool N` where N divides the number of processes).
- Reduce ecutwfc if possible (must be converged for band energies, not forces).
- Use fewer bands (only the minimum needed for the energy window of interest).
- For large unit cells, the k-grid can be coarser (the BZ is smaller).

### 7. Memory Issues

**Symptom**: Out-of-memory errors during NSCF or BoltzTraP2 interpolation.

**Fixes**:
- NSCF: Increase number of MPI processes (distributes memory) or use more pools.
- BoltzTraP2: The interpolation stores all bands at all k-points. If memory is tight, reduce the number of bands or k-points.
- Use `disk_io = 'low'` in the NSCF input to reduce disk I/O (eigenvalues are kept in memory).

### 8. Comparing with Experiment

- BoltzTraP2 uses the constant relaxation time approximation. This means:
  - Seebeck coefficient S is exact within BTE (tau cancels).
  - sigma and kappa_e are only accurate up to the unknown tau.
  - The temperature dependence of sigma may not match experiment because tau(T) is not captured.
- For better accuracy, combine with electron-phonon coupling to get tau(T), or use codes like EPW or Phoebe that compute tau from first principles.
