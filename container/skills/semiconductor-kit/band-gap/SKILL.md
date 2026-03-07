# Band Gap Extraction

## When to Use

- You need the band gap value (direct or indirect) of a semiconductor or insulator.
- You want to determine the VBM (valence band maximum) and CBM (conduction band minimum) positions in the Brillouin zone.
- You need to classify whether the gap is direct or indirect.
- You want to apply a scissor correction to compensate for the well-known DFT underestimation of band gaps.
- You want to compare computed band gaps with experimental values.
- You need band gap information from band structure or density of states data.

## Method Selection

| Method | Tool | Strengths | Limitations |
|---|---|---|---|
| QE band structure | pw.x + bands.x | Direct VBM/CBM positions, gap type | Requires bands calculation along k-path |
| QE pw.x output | pw.x SCF/NSCF | Quick estimate from "highest occupied / lowest unoccupied" | Only at sampled k-points (coarse grid may miss true extrema) |
| VASP EIGENVAL | Python parser | Standard VASP workflow | Requires VASP license |
| VASP vasprun.xml | pymatgen | Rich data, automatic analysis | Requires VASP license |
| QE/VASP DOS | Python parser | Complements band structure analysis | Less precise than eigenvalue analysis |
| pymatgen BandStructure | Python | Automatic gap detection | Needs parsed band structure object |

## Prerequisites

- A completed band structure calculation (SCF + bands NSCF along k-path) for QE, or a VASP static calculation.
- Alternatively, a completed SCF or NSCF calculation (for quick gap estimate from eigenvalues at sampled k-points).
- Python packages: `pymatgen`, `numpy`, `matplotlib`.
- For scissor correction: experimental band gap value for the material.

## Background

### DFT Band Gap Problem

DFT with standard (semi)local functionals (LDA, PBE) systematically underestimates band gaps, often by 30-50%. This is not a convergence issue but a fundamental limitation of the Kohn-Sham formalism (the derivative discontinuity of the exchange-correlation functional is missing). Common corrections:

| Method | Gap accuracy | Cost |
|---|---|---|
| PBE/LDA | Underestimates by 30-50% | Baseline |
| Scissor correction | Manual shift to match experiment | Zero (post-processing) |
| PBE+U | Improved for correlated systems | Slight increase |
| HSE06 hybrid | ~10-20% of experiment | 10-100x PBE |
| GW (G0W0) | ~5-10% of experiment | 100-1000x PBE |
| GLLB-SC (QE) | ~10-15% of experiment | ~2x PBE |

### Direct vs Indirect Gap

- **Direct gap**: VBM and CBM at the same k-point. Optical transitions are allowed without phonon assistance. Important for LEDs, lasers.
- **Indirect gap**: VBM and CBM at different k-points. Optical transitions require phonon assistance. Relevant for Si, Ge, diamond.

## Detailed Steps

### Overview

```
Method A: QE band structure parsing
  Step 1: Parse QE bands output for VBM/CBM
  Step 2: Classify direct/indirect gap
  Step 3: Apply scissor correction
  Step 4: Plot annotated band structure

Method B: QE pw.x output quick analysis
  Parse "highest occupied / lowest unoccupied" from SCF output

Method C: VASP output parsing
  Parse EIGENVAL or vasprun.xml for VBM/CBM

Method D: DOS-based gap extraction
  Extract gap from DOS onset
```

---

### Method A: QE Band Structure Gap Analysis

#### Step A1: Parse QE Bands Output

```python
#!/usr/bin/env python3
"""
Extract band gap from QE band structure calculation.

Prerequisites:
  - Completed SCF + bands NSCF + bands.x post-processing
  - bands.x output file (e.g., si_bands.dat or bands.out)
  - Or use raw QE pw.x bands output

This script:
  1. Reads eigenvalues from QE output
  2. Finds VBM (highest occupied state) and CBM (lowest unoccupied state)
  3. Determines if gap is direct or indirect
  4. Outputs gap value and k-point locations
"""
import re
import os
import json
import numpy as np


def parse_bands_dat(bands_dat_file, bands_dat_gnu_file=None):
    """
    Parse bands.x output (prefix.bands.dat and prefix.bands.dat.gnu).

    The .dat.gnu file contains plottable data:
      columns: k-distance  E(k)  [one block per band, separated by blank lines]

    Parameters
    ----------
    bands_dat_file : str
        The bands.x output (.dat file with header info).
    bands_dat_gnu_file : str
        The .dat.gnu file with plottable eigenvalues. If None, auto-detect.

    Returns
    -------
    k_distances : ndarray (nk,)
    energies : ndarray (nk, nbnd)
    """
    if bands_dat_gnu_file is None:
        bands_dat_gnu_file = bands_dat_file + '.gnu'

    if not os.path.exists(bands_dat_gnu_file):
        raise FileNotFoundError(f"{bands_dat_gnu_file} not found. "
                                "Run bands.x first.")

    # Read the .gnu file: blocks separated by blank lines
    with open(bands_dat_gnu_file, 'r') as f:
        content = f.read()

    blocks = content.strip().split('\n\n')
    bands_data = []
    for block in blocks:
        lines = block.strip().split('\n')
        band = []
        for line in lines:
            vals = line.split()
            if len(vals) >= 2:
                band.append([float(vals[0]), float(vals[1])])
        if band:
            bands_data.append(np.array(band))

    nbnd = len(bands_data)
    nk = len(bands_data[0])
    k_distances = bands_data[0][:, 0]
    energies = np.zeros((nk, nbnd))
    for ib, band in enumerate(bands_data):
        energies[:, ib] = band[:, 1]

    return k_distances, energies


def parse_pw_bands_output(pw_output_file, n_electrons=None):
    """
    Parse eigenvalues from QE pw.x output (bands or nscf calculation).

    Parameters
    ----------
    pw_output_file : str
        QE pw.x output file.
    n_electrons : int or None
        Number of electrons. If None, tries to parse from output.

    Returns
    -------
    kpoints : list of ndarray (3,) -- k-point coordinates
    eigenvalues : ndarray (nk, nbnd) -- eigenvalues in eV
    fermi_energy : float -- Fermi energy in eV (if found)
    """
    with open(pw_output_file, 'r') as f:
        lines = f.readlines()

    # Parse Fermi energy
    fermi_energy = None
    for line in lines:
        if 'Fermi energy' in line:
            m = re.search(r'Fermi energy\s+is\s+([-\d.]+)', line)
            if m:
                fermi_energy = float(m.group(1))
        if 'highest occupied' in line:
            nums = re.findall(r'[-\d.]+', line)
            if len(nums) >= 1:
                fermi_energy = float(nums[0])
            if len(nums) >= 2:
                fermi_energy = (float(nums[0]) + float(nums[1])) / 2

    # Parse eigenvalues
    kpoints = []
    eigenvalues = []
    i = 0
    while i < len(lines):
        if 'k =' in lines[i] and 'bands' in lines[i]:
            # Parse k-point
            m = re.search(r'k\s*=\s*([-\d.]+)\s+([-\d.]+)\s+([-\d.]+)', lines[i])
            if m:
                kpt = np.array([float(m.group(1)), float(m.group(2)),
                                float(m.group(3))])
                kpoints.append(kpt)

                # Skip blank line(s), then read eigenvalues
                j = i + 1
                while j < len(lines) and lines[j].strip() == '':
                    j += 1

                eigs = []
                while j < len(lines) and lines[j].strip() != '':
                    nums = re.findall(r'[-\d.]+', lines[j])
                    eigs.extend([float(x) for x in nums])
                    j += 1
                    if j < len(lines) and ('k =' in lines[j] or
                                           'Writing' in lines[j] or
                                           '---' in lines[j]):
                        break

                eigenvalues.append(eigs)
        i += 1

    if not eigenvalues:
        return None, None, fermi_energy

    # Pad to uniform band count
    nbnd = max(len(e) for e in eigenvalues)
    for i in range(len(eigenvalues)):
        while len(eigenvalues[i]) < nbnd:
            eigenvalues[i].append(np.nan)

    eigenvalues = np.array(eigenvalues)
    return kpoints, eigenvalues, fermi_energy


def find_band_gap(energies, fermi_energy=None, n_occupied=None):
    """
    Find VBM, CBM, and band gap from eigenvalue array.

    Parameters
    ----------
    energies : ndarray (nk, nbnd)
        Eigenvalues at each k-point.
    fermi_energy : float or None
        Fermi energy. Used to distinguish occupied/unoccupied bands.
    n_occupied : int or None
        Number of occupied bands. If None, determined from fermi_energy.

    Returns
    -------
    dict with:
        gap_eV : float
        gap_type : 'direct' or 'indirect'
        vbm_eV : float
        cbm_eV : float
        vbm_kindex : int
        cbm_kindex : int
        direct_gap_eV : float (minimum direct gap)
        direct_gap_kindex : int
    """
    nk, nbnd = energies.shape

    if n_occupied is None and fermi_energy is not None:
        # Determine occupied bands: bands below Fermi energy at each k-point
        # Use the band index where the band crosses Fermi at mid-zone
        mid_k = nk // 2
        n_occupied = np.sum(energies[mid_k, :] < fermi_energy + 0.01)
    elif n_occupied is None:
        # Guess: half the bands are occupied
        n_occupied = nbnd // 2

    if n_occupied <= 0 or n_occupied >= nbnd:
        print(f"WARNING: n_occupied = {n_occupied}, nbnd = {nbnd}. "
              "Cannot determine gap.")
        return None

    # VBM: maximum of the highest occupied band across all k-points
    vb_index = n_occupied - 1  # 0-indexed
    cb_index = n_occupied      # 0-indexed

    vbm_values = energies[:, vb_index]
    cbm_values = energies[:, cb_index]

    vbm_kindex = np.argmax(vbm_values)
    cbm_kindex = np.argmin(cbm_values)
    vbm_eV = vbm_values[vbm_kindex]
    cbm_eV = cbm_values[cbm_kindex]

    gap_eV = cbm_eV - vbm_eV

    # Direct gap at each k-point
    direct_gaps = cbm_values - vbm_values
    direct_gap_kindex = np.argmin(direct_gaps)
    direct_gap_eV = direct_gaps[direct_gap_kindex]

    # Classify
    if vbm_kindex == cbm_kindex:
        gap_type = 'direct'
    else:
        gap_type = 'indirect'

    result = {
        'gap_eV': float(max(0, gap_eV)),
        'gap_type': gap_type,
        'vbm_eV': float(vbm_eV),
        'cbm_eV': float(cbm_eV),
        'vbm_kindex': int(vbm_kindex),
        'cbm_kindex': int(cbm_kindex),
        'vbm_band': int(vb_index),
        'cbm_band': int(cb_index),
        'direct_gap_eV': float(max(0, direct_gap_eV)),
        'direct_gap_kindex': int(direct_gap_kindex),
        'n_occupied': int(n_occupied),
    }

    return result


def apply_scissor_correction(gap_dft_eV, gap_exp_eV, energies,
                              n_occupied):
    """
    Apply scissor operator: rigidly shift conduction bands to match
    experimental gap.

    Parameters
    ----------
    gap_dft_eV : float
        DFT band gap.
    gap_exp_eV : float
        Experimental band gap.
    energies : ndarray (nk, nbnd)
        Original eigenvalues.
    n_occupied : int
        Number of occupied bands.

    Returns
    -------
    energies_corrected : ndarray (nk, nbnd)
        Scissor-corrected eigenvalues.
    scissor_eV : float
        Applied scissor shift.
    """
    scissor_eV = gap_exp_eV - gap_dft_eV
    energies_corrected = energies.copy()
    energies_corrected[:, n_occupied:] += scissor_eV
    return energies_corrected, scissor_eV


# ── Main workflow ──────────────────────────────────────────────────

print("=" * 60)
print("Band Gap Analysis from QE Band Structure")
print("=" * 60)

# ── Option 1: Parse from bands.x output ──────────────────────────
bands_gnu = "si_bands.dat.gnu"
if os.path.exists(bands_gnu):
    print(f"\nParsing bands.x output: {bands_gnu}")
    k_dist, energies = parse_bands_dat("si_bands.dat", bands_gnu)
    # For bands.x output, Fermi energy is at 0 if set_fermi_energy was used
    gap_info = find_band_gap(energies, fermi_energy=0.0)

# ── Option 2: Parse from pw.x bands output ──────────────────────
else:
    pw_out = "bands.out"
    if not os.path.exists(pw_out):
        pw_out = "si_bands.out"
    if os.path.exists(pw_out):
        print(f"\nParsing pw.x output: {pw_out}")
        kpoints, energies, fermi_e = parse_pw_bands_output(pw_out)
        if energies is not None:
            gap_info = find_band_gap(energies, fermi_energy=fermi_e)
        else:
            print("No eigenvalues found in output.")
            gap_info = None
    else:
        print("No band structure output file found.")
        print("Expected: si_bands.dat.gnu or bands.out")
        gap_info = None

if gap_info is not None:
    print(f"\n{'='*60}")
    print("BAND GAP RESULTS")
    print(f"{'='*60}")
    print(f"  Band gap:          {gap_info['gap_eV']:.4f} eV")
    print(f"  Gap type:          {gap_info['gap_type']}")
    print(f"  VBM:               {gap_info['vbm_eV']:.4f} eV "
          f"(k-index {gap_info['vbm_kindex']})")
    print(f"  CBM:               {gap_info['cbm_eV']:.4f} eV "
          f"(k-index {gap_info['cbm_kindex']})")
    print(f"  Min. direct gap:   {gap_info['direct_gap_eV']:.4f} eV "
          f"(k-index {gap_info['direct_gap_kindex']})")
    print(f"  Occupied bands:    {gap_info['n_occupied']}")

    # ── Scissor correction ────────────────────────────────────────
    # Example: Si experimental gap = 1.17 eV
    gap_exp = 1.17  # eV (Si at 0 K)
    gap_dft = gap_info['gap_eV']

    if gap_dft > 0:
        energies_corr, scissor = apply_scissor_correction(
            gap_dft, gap_exp, energies, gap_info['n_occupied'])
        print(f"\n  Scissor correction: {scissor:+.4f} eV")
        print(f"  Corrected gap:     {gap_exp:.4f} eV")

        gap_corr = find_band_gap(energies_corr,
                                  n_occupied=gap_info['n_occupied'])
        print(f"  Corrected VBM:     {gap_corr['vbm_eV']:.4f} eV")
        print(f"  Corrected CBM:     {gap_corr['cbm_eV']:.4f} eV")

    # ── Save results ──────────────────────────────────────────────
    with open('band_gap_results.json', 'w') as f:
        json.dump(gap_info, f, indent=2)
    print(f"\nResults saved: band_gap_results.json")
```

#### Step A2: Plot Annotated Band Structure with Gap

```python
#!/usr/bin/env python3
"""
Plot band structure with VBM, CBM, and band gap annotated.
"""
import os
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def plot_bandstructure_with_gap(k_distances, energies, gap_info,
                                 fermi_energy=0.0,
                                 k_labels=None, k_label_positions=None,
                                 scissor_eV=0.0,
                                 output='band_gap_annotated.png'):
    """
    Plot band structure with VBM/CBM markers and gap annotation.

    Parameters
    ----------
    k_distances : ndarray (nk,)
    energies : ndarray (nk, nbnd)
    gap_info : dict from find_band_gap()
    fermi_energy : float
    k_labels : list of str
    k_label_positions : list of float
    scissor_eV : float (scissor correction already applied to energies)
    output : str
    """
    fig, ax = plt.subplots(figsize=(8, 6))

    nk, nbnd = energies.shape
    n_occ = gap_info['n_occupied']

    # Plot valence bands (blue) and conduction bands (red)
    for ib in range(nbnd):
        color = '#2166ac' if ib < n_occ else '#d6604d'
        alpha = 0.9 if ib in [n_occ - 1, n_occ] else 0.5
        lw = 2.0 if ib in [n_occ - 1, n_occ] else 1.0
        ax.plot(k_distances, energies[:, ib] - fermi_energy,
                color=color, linewidth=lw, alpha=alpha)

    # Mark VBM and CBM
    vbm_k = gap_info['vbm_kindex']
    cbm_k = gap_info['cbm_kindex']
    vbm_e = gap_info['vbm_eV'] - fermi_energy
    cbm_e = gap_info['cbm_eV'] - fermi_energy

    ax.plot(k_distances[vbm_k], vbm_e, 'go', markersize=12, zorder=5,
            label=f'VBM = {gap_info["vbm_eV"]:.3f} eV')
    ax.plot(k_distances[cbm_k], cbm_e, 'r^', markersize=12, zorder=5,
            label=f'CBM = {gap_info["cbm_eV"]:.3f} eV')

    # Draw gap arrow
    mid_x = (k_distances[vbm_k] + k_distances[cbm_k]) / 2
    ax.annotate('', xy=(mid_x, cbm_e), xytext=(mid_x, vbm_e),
                arrowprops=dict(arrowstyle='<->', color='black', lw=1.5))

    gap_label = f'$E_g$ = {gap_info["gap_eV"]:.3f} eV ({gap_info["gap_type"]})'
    if scissor_eV != 0:
        gap_label += f'\n(scissor: {scissor_eV:+.3f} eV)'
    ax.text(mid_x + 0.02 * (k_distances[-1] - k_distances[0]),
            (vbm_e + cbm_e) / 2, gap_label, fontsize=10,
            va='center', ha='left',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

    # Fermi level
    ax.axhline(y=0, color='gray', linestyle='--', linewidth=0.8,
               label='$E_F$')

    # High-symmetry point labels
    if k_labels and k_label_positions:
        ax.set_xticks(k_label_positions)
        ax.set_xticklabels(k_labels, fontsize=12)
        for pos in k_label_positions:
            ax.axvline(x=pos, color='gray', linewidth=0.5, alpha=0.5)

    ax.set_xlabel('Wave vector', fontsize=13)
    ax.set_ylabel('Energy (eV)', fontsize=13)
    ax.set_title('Band Structure with Gap Analysis', fontsize=14)
    ax.legend(fontsize=9, loc='upper right')
    ax.set_xlim(k_distances[0], k_distances[-1])
    ax.set_ylim(-5, 5)
    ax.grid(True, axis='y', alpha=0.2)

    plt.tight_layout()
    plt.savefig(output, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output}")


# ── Example usage ──────────────────────────────────────────────────
# This would follow a bands calculation. Using placeholder data:
if os.path.exists('si_bands.dat.gnu'):
    from band_gap_analysis import parse_bands_dat, find_band_gap
    k_dist, energies = parse_bands_dat('si_bands.dat', 'si_bands.dat.gnu')
    gap_info = find_band_gap(energies, fermi_energy=0.0)
    plot_bandstructure_with_gap(
        k_dist, energies, gap_info,
        fermi_energy=0.0,
        k_labels=['L', r'$\Gamma$', 'X', 'U|K', r'$\Gamma$'],
        output='band_gap_annotated.png'
    )
else:
    print("Run the band structure calculation first, then plot.")
```

---

### Method B: Quick Gap from QE SCF Output

```python
#!/usr/bin/env python3
"""
Quick band gap estimate from QE pw.x SCF output.
Parses 'highest occupied, lowest unoccupied' line from SCF output.

This gives the gap at the sampled k-points only.
The true gap (at arbitrary k) may be slightly different.
For an accurate gap, use a full band structure calculation (Method A).
"""
import re
import os


def quick_gap_from_scf(scf_output):
    """
    Parse band gap from QE SCF output.

    QE prints:
      'highest occupied, lowest unoccupied level (ev):   X.XXXX   Y.YYYY'
    or for metals:
      'the Fermi energy is   X.XXXX ev'

    Returns
    -------
    dict or None
    """
    if not os.path.exists(scf_output):
        print(f"File not found: {scf_output}")
        return None

    with open(scf_output, 'r') as f:
        content = f.read()

    # Look for explicit gap
    m = re.search(
        r'highest occupied, lowest unoccupied level \(ev\):\s+'
        r'([-\d.]+)\s+([-\d.]+)',
        content
    )
    if m:
        vbm = float(m.group(1))
        cbm = float(m.group(2))
        gap = cbm - vbm
        return {
            'gap_eV': gap,
            'vbm_eV': vbm,
            'cbm_eV': cbm,
            'source': 'SCF k-grid (approximate)',
            'is_metal': False,
        }

    # Check for metallic system
    m2 = re.search(r'the Fermi energy is\s+([-\d.]+)', content)
    if m2:
        return {
            'gap_eV': 0.0,
            'fermi_eV': float(m2.group(1)),
            'source': 'SCF output',
            'is_metal': True,
        }

    print("Could not parse band gap from SCF output.")
    return None


# ── Example ────────────────────────────────────────────────────────
for f in ['scf.out', 'si_scf.out', 'pw.out']:
    if os.path.exists(f):
        result = quick_gap_from_scf(f)
        if result:
            print(f"From {f}:")
            if result['is_metal']:
                print(f"  Material is metallic (Fermi energy = "
                      f"{result['fermi_eV']:.4f} eV)")
            else:
                print(f"  Gap = {result['gap_eV']:.4f} eV")
                print(f"  VBM = {result['vbm_eV']:.4f} eV")
                print(f"  CBM = {result['cbm_eV']:.4f} eV")
                print(f"  ({result['source']})")
        break
```

---

### Method C: VASP Output Parsing

#### From EIGENVAL

```python
#!/usr/bin/env python3
"""
Extract band gap from VASP EIGENVAL file.
"""
import os
import numpy as np


def parse_eigenval(filename='EIGENVAL'):
    """
    Parse VASP EIGENVAL file.

    Returns
    -------
    kpoints : ndarray (nk, 3)
    eigenvalues : ndarray (nk, nbnd) for non-spin-polarized
                  or (nk, nbnd, 2) for spin-polarized
    n_electrons : int
    weights : ndarray (nk,)
    """
    with open(filename, 'r') as f:
        lines = f.readlines()

    # Line 6: NELECT, NKPT, NBANDS
    header = lines[5].split()
    n_electrons = int(float(header[0]))
    nkpt = int(header[1])
    nbands = int(header[2])

    # Determine spin-polarized or not from number of columns
    # in eigenvalue lines
    test_line = lines[7 + 1].split()  # First eigenvalue line
    is_spin_pol = len(test_line) >= 4  # col: band_idx, E_up, E_down, occ

    kpoints = []
    weights = []
    eigenvalues = []

    idx = 7  # Start of k-point data
    for ik in range(nkpt):
        # K-point line
        kpt_line = lines[idx].split()
        kpoints.append([float(kpt_line[0]), float(kpt_line[1]),
                         float(kpt_line[2])])
        weights.append(float(kpt_line[3]))
        idx += 1

        # Eigenvalue lines
        eigs = []
        for ib in range(nbands):
            vals = lines[idx].split()
            if is_spin_pol:
                eigs.append([float(vals[1]), float(vals[2])])
            else:
                eigs.append(float(vals[1]))
            idx += 1

        eigenvalues.append(eigs)
        idx += 1  # Blank line

    kpoints = np.array(kpoints)
    weights = np.array(weights)
    eigenvalues = np.array(eigenvalues)

    return kpoints, eigenvalues, n_electrons, weights


def band_gap_from_eigenval(filename='EIGENVAL'):
    """
    Compute band gap from VASP EIGENVAL.
    """
    kpoints, eigenvalues, n_electrons, weights = parse_eigenval(filename)

    # Number of occupied bands (non-spin-polarized)
    if eigenvalues.ndim == 2:
        n_occupied = int(n_electrons / 2)
    else:
        n_occupied = int(n_electrons / 2)  # Per spin channel

    nk, nbnd = eigenvalues.shape[:2]

    if eigenvalues.ndim == 2:
        # Non-spin-polarized
        vb = eigenvalues[:, n_occupied - 1]
        cb = eigenvalues[:, n_occupied]
    else:
        # Spin-polarized: use spin-up channel
        vb = eigenvalues[:, n_occupied - 1, 0]
        cb = eigenvalues[:, n_occupied, 0]

    vbm_k = np.argmax(vb)
    cbm_k = np.argmin(cb)
    vbm = vb[vbm_k]
    cbm = cb[cbm_k]
    gap = cbm - vbm

    gap_type = 'direct' if vbm_k == cbm_k else 'indirect'

    # Direct gap at each k
    direct_gaps = cb - vb
    min_direct_k = np.argmin(direct_gaps)
    min_direct_gap = direct_gaps[min_direct_k]

    result = {
        'gap_eV': float(max(0, gap)),
        'gap_type': gap_type,
        'vbm_eV': float(vbm),
        'cbm_eV': float(cbm),
        'vbm_kpoint': kpoints[vbm_k].tolist(),
        'cbm_kpoint': kpoints[cbm_k].tolist(),
        'direct_gap_eV': float(max(0, min_direct_gap)),
        'n_electrons': n_electrons,
        'n_occupied': n_occupied,
    }

    return result


# ── Example ────────────────────────────────────────────────────────
if os.path.exists('EIGENVAL'):
    result = band_gap_from_eigenval('EIGENVAL')
    print("=" * 50)
    print("VASP Band Gap from EIGENVAL")
    print("=" * 50)
    print(f"  Gap:          {result['gap_eV']:.4f} eV ({result['gap_type']})")
    print(f"  VBM:          {result['vbm_eV']:.4f} eV at k = "
          f"{result['vbm_kpoint']}")
    print(f"  CBM:          {result['cbm_eV']:.4f} eV at k = "
          f"{result['cbm_kpoint']}")
    print(f"  Direct gap:   {result['direct_gap_eV']:.4f} eV")
```

#### From vasprun.xml via pymatgen

```python
#!/usr/bin/env python3
"""
Extract band gap from VASP vasprun.xml using pymatgen.
This is the most convenient method for VASP users.
"""
import os
from pymatgen.io.vasp.outputs import Vasprun


def band_gap_from_vasprun(vasprun_path='vasprun.xml'):
    """
    Extract band gap using pymatgen's Vasprun parser.

    Returns
    -------
    dict with gap info.
    """
    vrun = Vasprun(vasprun_path, parse_projected_eigen=False)
    bs = vrun.get_band_structure()

    gap_info = bs.get_band_gap()
    # gap_info = {'direct': bool, 'energy': float, 'transition': str}

    result = {
        'gap_eV': gap_info['energy'],
        'is_direct': gap_info['direct'],
        'gap_type': 'direct' if gap_info['direct'] else 'indirect',
        'transition': gap_info['transition'],
    }

    # VBM and CBM details
    vbm = bs.get_vbm()
    cbm = bs.get_cbm()
    result['vbm_eV'] = vbm['energy']
    result['cbm_eV'] = cbm['energy']
    result['vbm_kpoint'] = vbm['kpoint'].frac_coords.tolist()
    result['cbm_kpoint'] = cbm['kpoint'].frac_coords.tolist()

    return result


if os.path.exists('vasprun.xml'):
    result = band_gap_from_vasprun('vasprun.xml')
    print("=" * 50)
    print("VASP Band Gap from vasprun.xml (pymatgen)")
    print("=" * 50)
    print(f"  Gap:        {result['gap_eV']:.4f} eV ({result['gap_type']})")
    print(f"  Transition: {result['transition']}")
    print(f"  VBM:        {result['vbm_eV']:.4f} eV at {result['vbm_kpoint']}")
    print(f"  CBM:        {result['cbm_eV']:.4f} eV at {result['cbm_kpoint']}")
```

---

### Method D: Gap from Density of States

```python
#!/usr/bin/env python3
"""
Extract band gap from DOS data.
Useful as a complement to band structure analysis.
"""
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def gap_from_dos(energies, dos_values, threshold=0.01, fermi_energy=0.0):
    """
    Extract band gap from DOS by finding the energy range where
    DOS is below threshold near the Fermi level.

    Parameters
    ----------
    energies : ndarray
        Energy values in eV.
    dos_values : ndarray
        DOS values (states/eV).
    threshold : float
        DOS threshold below which the gap is considered (noise floor).
    fermi_energy : float
        Fermi energy in eV.

    Returns
    -------
    dict with gap_eV, vbm_eV, cbm_eV.
    """
    # Shift energies relative to Fermi level
    e_shifted = energies - fermi_energy

    # Find gap region: where DOS < threshold around E_F
    in_gap = dos_values < threshold

    # Find VBM: last energy below E_F=0 where DOS drops below threshold
    vbm_idx = None
    for i in range(len(e_shifted)):
        if e_shifted[i] >= 0:
            break
        if not in_gap[i]:
            vbm_idx = i

    # Find CBM: first energy above E_F=0 where DOS rises above threshold
    cbm_idx = None
    for i in range(len(e_shifted)):
        if e_shifted[i] > 0 and not in_gap[i]:
            cbm_idx = i
            break

    if vbm_idx is not None and cbm_idx is not None:
        vbm_eV = energies[vbm_idx]
        cbm_eV = energies[cbm_idx]
        gap_eV = cbm_eV - vbm_eV
    else:
        vbm_eV = fermi_energy
        cbm_eV = fermi_energy
        gap_eV = 0.0

    return {
        'gap_eV': float(max(0, gap_eV)),
        'vbm_eV': float(vbm_eV),
        'cbm_eV': float(cbm_eV),
        'source': 'DOS analysis',
    }


def parse_qe_dos(filename='dos.dat'):
    """Parse QE dos.x output file."""
    data = np.loadtxt(filename, comments='#')
    energies = data[:, 0]  # eV
    dos = data[:, 1]       # states/eV
    return energies, dos


def parse_vasp_doscar(filename='DOSCAR', natom=None):
    """Parse VASP DOSCAR file (total DOS)."""
    with open(filename, 'r') as f:
        lines = f.readlines()

    # Line 6: EMAX, EMIN, NEDOS, EFERMI, ?
    header = lines[5].split()
    emax = float(header[0])
    emin = float(header[1])
    nedos = int(header[2])
    efermi = float(header[3])

    energies = []
    dos = []
    for i in range(6, 6 + nedos):
        vals = lines[i].split()
        energies.append(float(vals[0]))
        dos.append(float(vals[1]))

    return np.array(energies), np.array(dos), efermi


# ── Example ────────────────────────────────────────────────────────
# QE DOS
if os.path.exists('dos.dat'):
    energies, dos = parse_qe_dos('dos.dat')
    result = gap_from_dos(energies, dos, threshold=0.01, fermi_energy=0.0)
    print(f"Gap from QE DOS: {result['gap_eV']:.4f} eV")
    print(f"  VBM = {result['vbm_eV']:.4f} eV, CBM = {result['cbm_eV']:.4f} eV")

# VASP DOSCAR
if os.path.exists('DOSCAR'):
    energies, dos, efermi = parse_vasp_doscar('DOSCAR')
    result = gap_from_dos(energies, dos, threshold=0.01, fermi_energy=efermi)
    print(f"Gap from VASP DOS: {result['gap_eV']:.4f} eV")
    print(f"  VBM = {result['vbm_eV']:.4f} eV, CBM = {result['cbm_eV']:.4f} eV")
```

---

## Key Parameters

| Parameter | Typical Value | Notes |
|---|---|---|
| K-grid (SCF) | 8x8x8+ | Must be converged for accurate eigenvalues |
| K-path density | 30+ points/segment | For band structure: denser = smoother bands |
| `nbnd` / `NBANDS` | >= 2x occupied | Must include conduction bands |
| Scissor correction | Material-specific | gap_exp - gap_DFT, applied rigidly to CB |
| DOS threshold | 0.01 states/eV | For DOS-based gap extraction |
| ecutwfc (QE) | 40-80 Ry | Must be converged for gap |
| ENCUT (VASP) | 300-500 eV | Must be converged for gap |

### Experimental Band Gaps for Validation

| Material | Exp. gap (eV) | PBE gap (eV) | Underestimation |
|---|---|---|---|
| Si | 1.17 | 0.55-0.65 | ~50% |
| GaAs | 1.52 | 0.50-0.70 | ~55% |
| ZnO | 3.37 | 0.70-0.80 | ~78% |
| MgO | 7.83 | 4.50-4.70 | ~40% |
| Diamond | 5.47 | 4.10-4.20 | ~24% |
| NaCl | 8.5 | 5.00-5.20 | ~40% |
| GaN | 3.50 | 1.70-2.00 | ~48% |

## Interpreting Results

### Gap Type Classification

- **Direct gap at Gamma**: Common in III-V semiconductors (GaAs, InP, GaN). Good for optoelectronics.
- **Indirect gap**: Si (VBM at Gamma, CBM near X), Ge (VBM at Gamma, CBM at L). Poor for light emission.
- **Quasi-direct**: VBM and CBM at same k-point but close in energy to another extremum. Check for closely spaced direct and indirect gaps.

### Scissor Correction Validity

The scissor correction is appropriate when:
- The band dispersion shape is correct but uniformly shifted.
- You need approximate optical properties without hybrid DFT.
- The material has a clear gap (not a semimetal).

The scissor correction is NOT appropriate when:
- Band ordering differs between DFT and experiment (band inversion).
- The material has strong correlation effects (use DFT+U or hybrid).
- Accurate dispersion near band edges is needed (use GW).

### Zero-Gap Results

If the computed gap is zero or negative:
- The material may be a metal or semimetal.
- DFT may be failing for a narrow-gap semiconductor (common for PBE).
- Check if a Hubbard U correction or hybrid functional opens a gap.
- Spin-orbit coupling can modify the gap significantly (e.g., Bi2Se3, PbTe).

## Common Issues

| Problem | Solution |
|---|---|
| **Gap is zero but material is a semiconductor** | DFT underestimation closed the gap. Use hybrid functional (HSE06) or DFT+U. |
| **Gap is too small** | Normal for PBE/LDA. Apply scissor correction or use HSE06. |
| **VBM/CBM at wrong k-point** | Ensure dense k-path through all high-symmetry points. Check with seekpath for correct Bravais lattice. |
| **Direct gap equals indirect gap** | VBM and CBM may be at same k-point, or k-path does not pass through the true extrema. |
| **Spin-orbit effects important** | Use `lspinorb = .true.` + fully relativistic PPs in QE. Crucial for heavy elements (Pb, Bi, Te). |
| **Gap depends on k-grid** | Gap from SCF output depends on sampled k-points. Use a full band structure for accurate gap. |
| **Negative gap** | Likely a metal or semimetal. If unexpected, check DFT+U or hybrid. |
| **Parsing errors** | Check file format. QE 7.x output format may differ from older versions. |
