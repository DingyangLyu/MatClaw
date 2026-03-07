# K-Points for Transport Calculations

## When to Use

- You need to generate a dense uniform k-mesh for Boltzmann transport calculations (BoltzTraP2).
- You want to perform convergence testing of transport properties (Seebeck, sigma/tau, power factor) with respect to k-grid density.
- You are setting up an NSCF calculation for transport and need to determine the optimal k-grid.
- You have an anisotropic crystal and need to determine appropriate anisotropic k-grids.
- You need to estimate computational cost before running a dense-k NSCF.

## Method Selection

| Method | Tool | When to Use |
|---|---|---|
| QE automatic k-grid | pw.x K_POINTS {automatic} | Standard uniform grid, simplest |
| QE explicit k-grid | pw.x K_POINTS {crystal} | Custom grids, anisotropic sampling |
| VASP KPOINTS | VASP KPOINTS file | Standard VASP workflow |
| pymatgen k-mesh generator | Python | Programmatic grid generation, anisotropic grids |
| Convergence test script | Python + QE | Automated convergence testing |

## Prerequisites

- Quantum ESPRESSO 7.5 (`pw.x`) on PATH -- or VASP.
- A completed SCF calculation (charge density established).
- Python with `numpy`, `pymatgen`, `matplotlib`.
- BoltzTraP2 (`pip install BoltzTraP2`) for convergence validation.

## Detailed Steps

### Overview

```
Step 1: Estimate initial k-grid based on system size and type
Step 2: Generate k-mesh (QE or VASP format)
Step 3: Run convergence test (multiple NSCF + BoltzTraP2)
Step 4: Select optimal k-grid balancing accuracy and cost
```

---

### Step 1: Estimate Initial K-Grid

```python
#!/usr/bin/env python3
"""
Estimate the required k-grid density for transport calculations.
Rule of thumb: k-point density should be at least 40-50 per reciprocal
lattice vector for simple semiconductors, scaled by the reciprocal
lattice vector length.
"""
import numpy as np
from pymatgen.core import Structure
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer

def estimate_transport_kgrid(structure, target_density=50, min_kpts=20):
    """
    Estimate k-grid for transport based on reciprocal lattice vectors.

    Parameters
    ----------
    structure : pymatgen Structure
        Input crystal structure.
    target_density : int
        Target number of k-points per reciprocal Angstrom.
        Use 40 for initial tests, 50 for production, 60+ for metals.
    min_kpts : int
        Minimum k-points per direction.

    Returns
    -------
    kgrid : tuple of (nk1, nk2, nk3)
    """
    recip_lengths = structure.lattice.reciprocal_lattice.abc  # 1/Angstrom
    kgrid = []
    for b in recip_lengths:
        nk = max(min_kpts, int(np.ceil(target_density * b)))
        # Make even for better symmetry handling
        if nk % 2 != 0:
            nk += 1
        kgrid.append(nk)
    return tuple(kgrid)


def estimate_cost(kgrid, nat, nbnd, nosym=True):
    """
    Estimate computational cost of NSCF with given k-grid.

    Parameters
    ----------
    kgrid : tuple (nk1, nk2, nk3)
    nat : int
        Number of atoms.
    nbnd : int
        Number of bands.
    nosym : bool
        If True, no symmetry reduction (full BZ).
    """
    nk_total = kgrid[0] * kgrid[1] * kgrid[2]
    if not nosym:
        # Rough estimate: symmetry reduces by ~factor of 8-48
        nk_irr = nk_total // 8
    else:
        nk_irr = nk_total

    # Rough NSCF time estimate (per k-point, single core):
    # ~0.01-0.1 s for small systems, ~1-10 s for large systems
    time_per_kpt_s = 0.05 * (nat / 2) * (nbnd / 10)
    total_time_s = nk_irr * time_per_kpt_s
    total_time_h = total_time_s / 3600

    print(f"K-grid: {kgrid[0]} x {kgrid[1]} x {kgrid[2]}")
    print(f"Total k-points: {nk_total}")
    print(f"With nosym={nosym}: {nk_irr} k-points to compute")
    print(f"Estimated NSCF time (single core): {total_time_h:.1f} hours")
    print(f"Estimated NSCF time (4 cores):     {total_time_h/4:.1f} hours")

    # Memory estimate: ~nbnd * nbnd * 16 bytes per k-point (rough)
    mem_per_kpt_MB = nbnd * nbnd * 16 / 1e6
    mem_total_GB = nk_irr * mem_per_kpt_MB / 1e3
    print(f"Estimated memory: ~{mem_total_GB:.1f} GB "
          f"(distributed across MPI ranks)")

    return nk_irr, total_time_h


# ── Example usage ──────────────────────────────────────────────────
# Si diamond structure
from pymatgen.core import Lattice
si = Structure(
    Lattice.cubic(5.431),
    ["Si", "Si"],
    [[0, 0, 0], [0.25, 0.25, 0.25]]
)

print("=== K-grid Estimation for Si ===\n")

for density in [30, 40, 50, 60]:
    kgrid = estimate_transport_kgrid(si, target_density=density)
    print(f"\n--- Target density: {density} k/recip-Angstrom ---")
    estimate_cost(kgrid, nat=2, nbnd=16, nosym=True)

# Anisotropic example: layered material with c >> a
print("\n\n=== K-grid for Layered Material (a=3.5, c=12.0 Angstrom) ===\n")
layered = Structure(
    Lattice.hexagonal(3.5, 12.0),
    ["Bi", "Bi"],
    [[0.333, 0.667, 0.25], [0.667, 0.333, 0.75]]
)
kgrid = estimate_transport_kgrid(layered, target_density=50)
print(f"Recommended: {kgrid} (anisotropic -- fewer k along short recip axis)")
estimate_cost(kgrid, nat=2, nbnd=20, nosym=True)
```

### Step 2: Generate K-Mesh Files

#### Method A: QE Automatic K-Grid

For the QE NSCF input, simply use:

```
K_POINTS {automatic}
  40 40 40  0 0 0
```

The shift (`0 0 0` vs `1 1 1`) matters:
- `0 0 0`: Gamma-centered grid (includes the Gamma point). Recommended for hexagonal and systems where Gamma is a high-symmetry point.
- `1 1 1`: Monkhorst-Pack shifted grid. Slightly more efficient for cubic systems but may miss Gamma.

For transport with BoltzTraP2, use `0 0 0` (Gamma-centered) for consistency.

#### Method B: Generate Explicit K-Points with pymatgen

```python
#!/usr/bin/env python3
"""
Generate explicit k-point lists for transport calculations.
Useful for anisotropic grids or special sampling requirements.
"""
import numpy as np
from pymatgen.core import Structure, Lattice


def generate_uniform_kgrid(nk1, nk2, nk3, shift=(0, 0, 0)):
    """
    Generate a uniform Gamma-centered k-grid in fractional coordinates.

    Parameters
    ----------
    nk1, nk2, nk3 : int
        Grid dimensions.
    shift : tuple
        Shift in units of 1/(2*nk). (0,0,0) for Gamma-centered,
        (1,1,1) for Monkhorst-Pack shifted.

    Returns
    -------
    kpoints : ndarray, shape (nk1*nk2*nk3, 3)
    weights : ndarray, shape (nk1*nk2*nk3,)
    """
    kpoints = []
    for i in range(nk1):
        for j in range(nk2):
            for k in range(nk3):
                kx = (i + shift[0] / 2.0) / nk1
                ky = (j + shift[1] / 2.0) / nk2
                kz = (k + shift[2] / 2.0) / nk3
                # Fold to [-0.5, 0.5)
                kx = kx - np.floor(kx + 0.5)
                ky = ky - np.floor(ky + 0.5)
                kz = kz - np.floor(kz + 0.5)
                kpoints.append([kx, ky, kz])

    kpoints = np.array(kpoints)
    nk_total = nk1 * nk2 * nk3
    weights = np.ones(nk_total) / nk_total
    return kpoints, weights


def write_qe_kpoints(kpoints, weights, filename='kpoints_explicit.txt'):
    """Write k-points in QE crystal format."""
    nk = len(kpoints)
    lines = [f"K_POINTS {{crystal}}", f"{nk}"]
    for i in range(nk):
        k = kpoints[i]
        w = weights[i]
        lines.append(f"  {k[0]:14.10f}  {k[1]:14.10f}  {k[2]:14.10f}  {w:.10f}")

    with open(filename, 'w') as f:
        f.write("\n".join(lines) + "\n")
    print(f"Written: {filename} ({nk} k-points)")


def write_vasp_kpoints(nk1, nk2, nk3, filename='KPOINTS', shift=(0, 0, 0)):
    """Write VASP KPOINTS file."""
    lines = [
        "Automatic mesh for transport",
        "0",
        "Gamma" if shift == (0, 0, 0) else "Monkhorst-Pack",
        f"{nk1} {nk2} {nk3}",
        f"{shift[0]} {shift[1]} {shift[2]}",
    ]
    with open(filename, 'w') as f:
        f.write("\n".join(lines) + "\n")
    print(f"Written: {filename} ({nk1}x{nk2}x{nk3})")


# ── Example: generate k-grids ─────────────────────────────────────
# Isotropic: 40x40x40 Gamma-centered
kpts, wts = generate_uniform_kgrid(40, 40, 40, shift=(0, 0, 0))
write_qe_kpoints(kpts, wts, 'kpoints_40x40x40.txt')
write_vasp_kpoints(40, 40, 40, 'KPOINTS_40x40x40')

# Anisotropic: layered material
kpts, wts = generate_uniform_kgrid(40, 40, 12, shift=(0, 0, 0))
write_qe_kpoints(kpts, wts, 'kpoints_40x40x12.txt')
write_vasp_kpoints(40, 40, 12, 'KPOINTS_40x40x12')
```

### Step 3: Convergence Test

```python
#!/usr/bin/env python3
"""
Automated convergence test for transport properties with respect to
k-grid density. Runs multiple NSCF+BoltzTraP2 calculations and plots
convergence of Seebeck coefficient.
"""
import os
import subprocess
import json
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


def run_nscf_for_kgrid(nk, prefix='si', pseudo_dir='./pseudo',
                        outdir_base='./tmp_transport'):
    """
    Run NSCF with a specific k-grid. Assumes SCF already completed
    in outdir_base.

    Parameters
    ----------
    nk : int
        K-grid size (nk x nk x nk).
    prefix : str
        QE prefix.
    pseudo_dir : str
        Pseudopotential directory.
    outdir_base : str
        QE outdir where SCF results are stored.
    """
    outdir = os.path.abspath(outdir_base)

    nscf_input = f"""&CONTROL
    calculation   = 'nscf'
    prefix        = '{prefix}'
    pseudo_dir    = '{os.path.abspath(pseudo_dir)}'
    outdir        = '{outdir}'
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
  {nk} {nk} {nk}  0 0 0
"""

    input_file = f"nscf_k{nk}.in"
    output_file = f"nscf_k{nk}.out"

    with open(input_file, "w") as f:
        f.write(nscf_input)

    print(f"  Running NSCF with {nk}x{nk}x{nk} k-grid ...")
    result = subprocess.run(
        ["mpirun", "-np", "4", "pw.x", "-npool", "4", "-in", input_file],
        capture_output=True, text=True, timeout=7200
    )
    with open(output_file, "w") as f:
        f.write(result.stdout)

    if result.returncode != 0:
        print(f"  ERROR: NSCF with k={nk} failed!")
        return False

    print(f"  NSCF k={nk} completed.")
    return True


def compute_seebeck_at_kgrid(prefix='si', outdir='./tmp_transport/'):
    """Run BoltzTraP2 and return Seebeck at 300 K, n=1e19 n-type."""
    data_dir = os.path.join(outdir, f'{prefix}.save')
    if not os.path.isdir(data_dir):
        return None

    try:
        data = BoltzTraP2.dft.DFTData(data_dir)
        equivalences = BoltzTraP2.fite.equivalences(data, magmom=None)
        coeffs = BoltzTraP2.fite.fitde3D(data, equivalences)

        temperatures = np.array([300.0])
        eV_to_Ha = 1.0 / 27.2114
        mu_array = np.linspace(
            data.fermi - 1.5 * eV_to_Ha,
            data.fermi + 1.5 * eV_to_Ha,
            1000
        )
        volume_m3 = data.atoms.get_volume() * 1e-30

        LMTC = BoltzTraP2.fite.BTPDOS(
            data.ebands, None, data.kpoints,
            temperatures, mu_array, volume_m3,
            equivalences, coeffs
        )
        epsilon, dos, vvdos, cdos = LMTC

        sigma, seebeck, kappa, hall = (
            BoltzTraP2.bandlib.calc_Onsager_coefficients(
                epsilon, dos, vvdos, temperatures, mu_array, volume_m3
            )
        )
        carrier_conc = BoltzTraP2.bandlib.calc_N(
            epsilon, dos, temperatures, mu_array, volume_m3
        )

        # Find mu for n=1e19 cm^-3, n-type
        target_n_m3 = 1e19 * 1e6
        conc = carrier_conc[0, :]
        mask = mu_array >= data.fermi
        conc_diff = np.abs(np.abs(conc) - target_n_m3)
        conc_diff_masked = np.where(mask, conc_diff, np.inf)
        idx = np.argmin(conc_diff_masked)

        S = np.trace(seebeck[0, idx, :, :]) / 3.0 * 1e6  # uV/K
        sig = np.trace(sigma[0, idx, :, :]) / 3.0
        PF = (S * 1e-6)**2 * sig

        return {'seebeck_uV_K': S, 'sigma_tau': sig, 'PF_tau': PF}
    except Exception as e:
        print(f"  BoltzTraP2 error: {e}")
        return None


def run_convergence_test(k_grids=None, prefix='si',
                          pseudo_dir='./pseudo',
                          outdir='./tmp_transport'):
    """
    Run convergence test over multiple k-grids.

    Parameters
    ----------
    k_grids : list of int
        K-grid sizes to test (e.g., [20, 25, 30, 35, 40]).
    """
    if k_grids is None:
        k_grids = [20, 25, 30, 35, 40, 45]

    results = []

    for nk in k_grids:
        print(f"\n{'='*50}")
        print(f"K-grid: {nk} x {nk} x {nk}")
        print(f"{'='*50}")

        success = run_nscf_for_kgrid(nk, prefix, pseudo_dir, outdir)
        if not success:
            results.append({'nk': nk, 'seebeck': None, 'sigma': None,
                            'PF': None})
            continue

        transport = compute_seebeck_at_kgrid(prefix, outdir)
        if transport is not None:
            results.append({
                'nk': nk,
                'seebeck': transport['seebeck_uV_K'],
                'sigma': transport['sigma_tau'],
                'PF': transport['PF_tau'],
            })
            print(f"  S = {transport['seebeck_uV_K']:.2f} uV/K")
            print(f"  sigma/tau = {transport['sigma_tau']:.3e}")
        else:
            results.append({'nk': nk, 'seebeck': None, 'sigma': None,
                            'PF': None})

    return results


def plot_convergence(results, output='kgrid_convergence.png'):
    """Plot transport property convergence with k-grid."""
    valid = [r for r in results if r['seebeck'] is not None]
    if len(valid) < 2:
        print("Not enough data points for convergence plot.")
        return

    nk_vals = [r['nk'] for r in valid]
    S_vals = [r['seebeck'] for r in valid]
    sig_vals = [r['sigma'] for r in valid]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Seebeck convergence
    ax1.plot(nk_vals, S_vals, 'bo-', linewidth=2, markersize=8)
    ax1.set_xlabel('K-grid (N x N x N)', fontsize=13)
    ax1.set_ylabel(r'Seebeck ($\mu$V/K) at 300 K, n=10$^{19}$ cm$^{-3}$',
                    fontsize=11)
    ax1.set_title('Seebeck Convergence', fontsize=14)
    ax1.grid(True, alpha=0.3)

    # Add convergence check
    if len(S_vals) >= 2:
        final = S_vals[-1]
        for i, (nk, S) in enumerate(zip(nk_vals, S_vals)):
            pct_diff = abs(S - final) / abs(final) * 100 if final != 0 else 0
            ax1.annotate(f'{pct_diff:.1f}%', (nk, S),
                         textcoords="offset points", xytext=(10, 5),
                         fontsize=8)

    # sigma/tau convergence
    ax2.plot(nk_vals, sig_vals, 'ro-', linewidth=2, markersize=8)
    ax2.set_xlabel('K-grid (N x N x N)', fontsize=13)
    ax2.set_ylabel(r'$\sigma/\tau$ (1/$\Omega\cdot$m$\cdot$s) at 300 K',
                    fontsize=11)
    ax2.set_title('Conductivity Convergence', fontsize=14)
    ax2.grid(True, alpha=0.3)
    ax2.set_yscale('log')

    plt.tight_layout()
    plt.savefig(output, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output}")

    # Print summary table
    print("\n--- Convergence Summary ---")
    print(f"{'K-grid':>10}  {'S (uV/K)':>10}  {'sigma/tau':>12}  {'PF/tau':>12}")
    print("-" * 50)
    for r in valid:
        print(f"  {r['nk']:>3}^3     {r['seebeck']:>10.2f}  "
              f"{r['sigma']:>12.3e}  {r['PF']:>12.3e}")

    # Convergence criterion
    if len(S_vals) >= 2:
        pct = abs(S_vals[-1] - S_vals[-2]) / abs(S_vals[-1]) * 100
        if pct < 5:
            print(f"\nSeebeck converged to within {pct:.1f}% "
                  f"at {nk_vals[-2]}^3 grid.")
        else:
            print(f"\nSeebeck NOT yet converged ({pct:.1f}% change). "
                  f"Try denser grids.")


# ── Main ───────────────────────────────────────────────────────────
if __name__ == '__main__':
    print("=" * 60)
    print("K-Grid Convergence Test for Transport Properties")
    print("=" * 60)

    results = run_convergence_test(
        k_grids=[20, 25, 30, 35, 40],
        prefix='si',
        pseudo_dir='./pseudo',
        outdir='./tmp_transport',
    )

    plot_convergence(results, 'kgrid_convergence.png')

    # Save raw results
    with open('kgrid_convergence.json', 'w') as f:
        json.dump(results, f, indent=2)
    print("\nResults saved: kgrid_convergence.json")
```

### Step 4: Guidelines for Selecting K-Grid

```python
#!/usr/bin/env python3
"""
Quick reference: recommended k-grids for common material classes.
This script prints guidance and generates KPOINTS files.
"""
import numpy as np


RECOMMENDATIONS = {
    'simple_semiconductor': {
        'examples': 'Si, GaAs, InP, ZnO',
        'min_kgrid': 30,
        'recommended': 40,
        'production': 50,
        'notes': 'Seebeck converges first; sigma needs denser grid.',
    },
    'complex_semiconductor': {
        'examples': 'Bi2Te3, PbTe, SnSe, skutterudites',
        'min_kgrid': 25,
        'recommended': 35,
        'production': 40,
        'notes': 'Larger unit cell = fewer k/direction needed. '
                 'Use anisotropic grid for layered systems.',
    },
    'metal': {
        'examples': 'Cu, Al, Fe, Pt',
        'min_kgrid': 40,
        'recommended': 50,
        'production': 60,
        'notes': 'Sharp Fermi surface requires fine sampling.',
    },
    'semimetal_dirac': {
        'examples': 'Bi, graphene, Weyl semimetals',
        'min_kgrid': 40,
        'recommended': 60,
        'production': 80,
        'notes': 'Dirac/Weyl cone needs very fine sampling.',
    },
    'layered_material': {
        'examples': 'MoS2, graphite, Bi2Se3',
        'min_kgrid': '30x30x10',
        'recommended': '40x40x15',
        'production': '50x50x20',
        'notes': 'Use anisotropic grid: dense in-plane, coarser out-of-plane.',
    },
}

print("=" * 70)
print("Recommended K-Grids for Transport Calculations")
print("=" * 70)

for category, info in RECOMMENDATIONS.items():
    print(f"\n--- {category.replace('_', ' ').title()} ---")
    print(f"  Examples:    {info['examples']}")
    print(f"  Minimum:     {info['min_kgrid']}")
    print(f"  Recommended: {info['recommended']}")
    print(f"  Production:  {info['production']}")
    print(f"  Notes:       {info['notes']}")

print("\n" + "=" * 70)
print("General Rules:")
print("  1. Always use nosym=.true. (QE) or ISYM=-1 (VASP)")
print("  2. Gamma-centered grids (no shift) for BoltzTraP2")
print("  3. Convergence test: plot S at 300K vs k-grid, stop when <5% change")
print("  4. sigma/tau converges slower than Seebeck -- test both")
print("  5. For anisotropic cells, scale k-grid inversely with lattice param")
print("=" * 70)
```

---

## Key Parameters

| Parameter | Typical Value | Notes |
|---|---|---|
| K-grid density | 40-60 per direction | Transport needs much denser grids than energy/DOS |
| Grid type | Gamma-centered | Use `0 0 0` shift in QE, `Gamma` in VASP |
| `nosym` (QE) | `.true.` | **Mandatory** for BoltzTraP2 |
| `ISYM` (VASP) | `-1` | **Mandatory** VASP equivalent |
| Anisotropic scaling | Proportional to 1/a | Longer lattice constant -> fewer k in that direction |
| Convergence criterion | S changes < 5% | Seebeck at fixed T and doping |

### K-Grid Density vs System Size

| Atoms in cell | Typical k-grid | Total k-points (nosym) |
|---|---|---|
| 2 (Si) | 40x40x40 | 64,000 |
| 5 (BaTiO3) | 30x30x30 | 27,000 |
| 10 (Bi2Te3) | 25x25x25 | 15,625 |
| 20+ (skutterudite) | 20x20x20 | 8,000 |

## Interpreting Results

### Convergence Behavior

- **Seebeck coefficient**: Converges relatively quickly with k-grid (often by 30x30x30 for simple systems).
- **Electrical conductivity (sigma/tau)**: Converges more slowly -- requires denser grids than Seebeck.
- **Power factor (S^2*sigma)**: Convergence intermediate between S and sigma.
- **kappa_e/tau**: Similar convergence rate to sigma/tau.

### When Is the Grid Converged?

1. Plot the Seebeck coefficient at T=300 K for a fixed doping (e.g., 1e19 cm^-3) vs. k-grid density.
2. The grid is converged when consecutive grid sizes differ by less than 5%.
3. For publication-quality results, verify that both S and sigma/tau are converged.
4. Check convergence for both n-type and p-type carriers.

### Anisotropic Grids

For layered materials (e.g., Bi2Se3, MoS2, graphite):
- Use dense sampling in the in-plane directions (large reciprocal lattice vectors).
- Use coarser sampling along the stacking direction (small reciprocal lattice vector).
- Example: 40x40x12 for a hexagonal cell with c/a ~ 3.

## Common Issues

| Problem | Solution |
|---|---|
| **NSCF too slow for dense grid** | Use more MPI processes with `-npool N`. Reduce ecutwfc if possible. Reduce nbnd. |
| **Out of memory** | Distribute with more MPI ranks. Use `disk_io = 'low'`. |
| **BoltzTraP2 fails after NSCF** | Check `nosym = .true.` in NSCF input. Verify the full BZ was sampled. |
| **Transport not converging** | Material may have sharp features (van Hove singularities, flat bands). Try much denser grids (50+). |
| **K-grid too coarse for metals** | Metals need 50x50x50 or denser. Fermi surface features require fine sampling. |
| **Anisotropic convergence** | Test each direction independently. Some directions converge faster. |
| **Cost estimate wrong** | Actual cost depends on system specifics. Use the first NSCF timing to calibrate. |
