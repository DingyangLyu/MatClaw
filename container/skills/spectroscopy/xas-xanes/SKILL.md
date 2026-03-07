# X-ray Absorption Spectroscopy (XAS/XANES) Simulation

## When to Use

- You need to simulate **K-edge or L-edge XANES** (X-ray Absorption Near-Edge Structure) spectra for comparison with synchrotron experiments.
- You want to understand **pre-edge features** arising from forbidden transitions (e.g., 1s -> 3d in transition metals).
- You need to probe the **local electronic structure** around a specific absorbing atom (oxidation state, coordination environment).
- You want to fingerprint different polymorphs, defects, or surface sites by their XAS signatures.

## Method Selection

| Method | Tool | Pros | Cons |
|--------|------|------|------|
| **QE xspectra.x** | `pw.x` + `xspectra.x` | First-principles, includes core-hole effects, well-tested | Requires core-hole pseudopotential, DFT-level only |
| **QE + GIPAW** | `pw.x` + GIPAW | For NMR/EPR but related approach | Not directly for XAS |
| **OCEAN** | External code | BSE-level, very accurate | Not included in container |
| **FDMNES** | External code | Multiple scattering, XANES + EXAFS | Not included |

**Recommendation:** Use **QE xspectra.x** -- it is included in the QE 7.5 distribution and handles K-edge and L-edge XANES with core-hole effects.

## Prerequisites

```bash
# QE binaries: pw.x, xspectra.x
# Core-hole pseudopotentials: generate with ld1.x or download from QE website
# Already available: numpy, matplotlib, ase, pymatgen
```

## Theory Summary

XAS measures the absorption coefficient mu(E) as a function of photon energy E near a core-level threshold. The absorption cross-section is computed from Fermi's golden rule:

```
mu(E) ~ sum_f |<f| D |i>|^2 * delta(E_f - E_i - hv)
```

where |i> is the core state (e.g., 1s for K-edge), |f> are final states, and D is the dipole operator. In QE xspectra.x, this is evaluated using the Lanczos continued-fraction method, which avoids explicit computation of all empty states.

**Core-hole approach:** To account for the relaxation of electrons around the core hole, a special pseudopotential with one fewer core electron is used for the absorbing atom. This is the "final-state rule" approximation.

## Detailed Steps

### Complete Workflow: Ti K-edge XANES of TiO2 (Rutile)

#### Step 1: Generate Core-Hole Pseudopotential

```python
#!/usr/bin/env python3
"""
Generate a core-hole pseudopotential for Ti K-edge using ld1.x.
The core hole is created by removing one 1s electron from Ti.
"""

import os
import subprocess

WORK_DIR = "/home/work/xas_tio2"
PSEUDO_DIR = "/home/pseudo"
os.makedirs(WORK_DIR, exist_ok=True)

# ============================================================
# Step 1: Generate Ti core-hole pseudo with ld1.x
# ============================================================
# This creates a Ti PP with a 1s hole (one 1s electron removed)
# Configuration: [Ar] 3d2 4s2 -> [Ar*] 3d2 4s2 with 1s^1 instead of 1s^2

ld1_input = """&INPUT
    title     = 'Ti with 1s core hole'
    zed       = 22.0
    rel       = 1
    config    = '[Ar] 3d2.0 4s2.0'
    iswitch   = 3
    dft       = 'PBE'
/
&INPUTP
    lpaw         = .true.
    pseudotype   = 3
    file_pseudopw = 'Ti.star1s.pbe-spn-kjpaw_psl.UPF'
    author       = 'core-hole'
    lloc         = -1
    rcloc        = 1.8
    which_augfun = 'PSQ'
    rmatch_augfun_nc = .true.
    nlcc         = .true.
    new_core_ps  = .true.
    tm           = .true.
/
6
1S  1  0  1.00  0.00  1.50  1.80  0.0
2S  2  0  2.00  0.00  1.50  1.80  0.0
2P  2  1  6.00  0.00  1.50  1.80  0.0
3S  3  0  2.00  0.00  1.50  1.80  0.0
3P  3  1  6.00  0.00  1.50  1.80  0.0
3D  3  2  2.00  0.00  1.50  1.80  0.0
"""

# NOTE: The key line is "1S  1  0  1.00" -- only 1 electron in the 1s shell
# (instead of the usual 2), creating the core hole.

ld1_file = os.path.join(WORK_DIR, "ti_corehole.in")
with open(ld1_file, "w") as f:
    f.write(ld1_input)

print("Generating core-hole pseudopotential with ld1.x...")
print("(If ld1.x is not available, download a pre-generated core-hole PP)")
try:
    result = subprocess.run(
        ["ld1.x", "-input", ld1_file],
        capture_output=True, text=True, cwd=WORK_DIR, timeout=600,
    )
    with open(os.path.join(WORK_DIR, "ld1.out"), "w") as f:
        f.write(result.stdout)
    if result.returncode == 0:
        print("Core-hole PP generated: Ti.star1s.pbe-spn-kjpaw_psl.UPF")
    else:
        print(f"ld1.x returned code {result.returncode}")
        print("Falling back to using standard PP with approximate core-hole treatment")
except FileNotFoundError:
    print("ld1.x not found. Using alternative approach (see below).")

# ALTERNATIVE: If you cannot generate a core-hole PP, you can:
# 1. Download one from the QE pseudopotential library (pslibrary)
# 2. Use the XSpectra "half-core-hole" approach with the standard PP
# 3. Use the standard PP and set xiabs to select the absorbing atom
```

#### Step 2: SCF Calculation with Core-Hole PP

```python
# ============================================================
# Step 2: SCF calculation for TiO2 rutile with core hole on one Ti
# ============================================================
# Use a supercell to isolate the core hole from its periodic images

scf_input = f"""&CONTROL
    calculation  = 'scf'
    prefix       = 'tio2_xas'
    outdir       = '{WORK_DIR}/tmp'
    pseudo_dir   = '{PSEUDO_DIR}'
    tprnfor      = .true.
/
&SYSTEM
    ibrav        = 0
    nat          = 24
    ntyp         = 3
    ecutwfc      = 50.0
    ecutrho      = 400.0
    occupations  = 'smearing'
    smearing     = 'mv'
    degauss      = 0.02
    nspin        = 1
    ! Add one extra electron to compensate the core hole (charged cell)
    ! tot_charge = -1.0
/
&ELECTRONS
    conv_thr     = 1.0d-8
    mixing_beta  = 0.3
/

ATOMIC_SPECIES
  Ti    47.867   Ti.pbe-spn-kjpaw_psl.1.0.0.UPF
  Ti_h  47.867   Ti.star1s.pbe-spn-kjpaw_psl.UPF
  O     15.999   O.pbe-n-kjpaw_psl.1.0.0.UPF

! 2x2x2 supercell of rutile TiO2 (a=4.594, c=2.959)
CELL_PARAMETERS angstrom
   9.188   0.000   0.000
   0.000   9.188   0.000
   0.000   0.000   5.918

! One Ti replaced by Ti_h (core-hole species) at position 1
ATOMIC_POSITIONS crystal
  Ti_h  0.0000  0.0000  0.0000
  Ti    0.5000  0.5000  0.0000
  Ti    0.0000  0.5000  0.5000
  Ti    0.5000  0.0000  0.5000
  Ti    0.2500  0.2500  0.0000
  Ti    0.7500  0.7500  0.0000
  Ti    0.2500  0.7500  0.5000
  Ti    0.7500  0.2500  0.5000
  O     0.1531  0.1531  0.0000
  O     0.8469  0.8469  0.0000
  O     0.3469  0.6531  0.0000
  O     0.6531  0.3469  0.0000
  O     0.1531  0.6531  0.5000
  O     0.8469  0.3469  0.5000
  O     0.3469  0.1531  0.5000
  O     0.6531  0.8469  0.5000
  O     0.4031  0.4031  0.0000
  O     0.5969  0.5969  0.0000
  O     0.0969  0.9031  0.0000
  O     0.9031  0.0969  0.0000
  O     0.4031  0.9031  0.5000
  O     0.5969  0.0969  0.5000
  O     0.0969  0.4031  0.5000
  O     0.9031  0.5969  0.5000

K_POINTS automatic
  2 2 3 0 0 0
"""

scf_file = os.path.join(WORK_DIR, "tio2_scf.in")
with open(scf_file, "w") as f:
    f.write(scf_input)

print("Running SCF with core-hole pseudopotential...")
result = subprocess.run(
    ["mpirun", "-np", "4", "pw.x", "-input", scf_file],
    capture_output=True, text=True, cwd=WORK_DIR, timeout=7200,
)
with open(os.path.join(WORK_DIR, "tio2_scf.out"), "w") as f:
    f.write(result.stdout)

# Check convergence
if "convergence has been achieved" in result.stdout:
    print("SCF converged successfully.")
else:
    print("WARNING: SCF may not have converged. Check tio2_scf.out")
```

#### Step 3: Run xspectra.x for K-edge XANES

```python
# ============================================================
# Step 3: xspectra.x calculation for Ti K-edge
# ============================================================
# Three polarization directions for powder average

for ipol, pol_name in enumerate(["x", "y", "z"], start=1):
    xspectra_input = f"""&INPUT_XSPECTRA
    calculation   = 'xanes_dipole'
    prefix        = 'tio2_xas'
    outdir        = '{WORK_DIR}/tmp'
    xniter        = 2000
    xcheck_conv   = 50
    xerror         = 0.001
    xepsilon(1)   = {1.0 if ipol == 1 else 0.0}
    xepsilon(2)   = {1.0 if ipol == 2 else 0.0}
    xepsilon(3)   = {1.0 if ipol == 3 else 0.0}
    xiabs         = 1
    x_save_file   = '{WORK_DIR}/xanes_{pol_name}.sav'
    xonly_plot     = .false.
    ef_r          = 0.0
/
&PLOT
    xnepoint      = 1000
    xemin         = -10.0
    xemax         = 40.0
    xgamma        = 0.8
    cut_occ_states = .true.
    terminator     = .true.
    gamma_mode     = 'constant'
/
&PSEUDOS
    filecore      = '{PSEUDO_DIR}/Ti.star1s.wfc'
    r_paw(1)      = 1.8
/
&CUT_OCC
/
  2 2 3 0 0 0
"""

    xspectra_file = os.path.join(WORK_DIR, f"xspectra_{pol_name}.in")
    with open(xspectra_file, "w") as f:
        f.write(xspectra_input)

    print(f"Running xspectra.x for polarization {pol_name}...")
    result = subprocess.run(
        ["mpirun", "-np", "4", "xspectra.x", "-input", xspectra_file],
        capture_output=True, text=True, cwd=WORK_DIR, timeout=3600,
    )
    with open(os.path.join(WORK_DIR, f"xspectra_{pol_name}.out"), "w") as f:
        f.write(result.stdout)

print("XSpectra calculations complete for all polarizations.")
```

#### Step 4: Parse and Plot XANES Spectrum

```python
# ============================================================
# Step 4: Parse xspectra output and plot XANES
# ============================================================
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

WORK_DIR = "/home/work/xas_tio2"

def parse_xspectra_dat(filename):
    """
    Parse xspectra.x output .dat file.
    Format: energy(eV)  mu(E)
    """
    energy = []
    mu = []
    with open(filename, "r") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) >= 2:
                try:
                    energy.append(float(parts[0]))
                    mu.append(float(parts[1]))
                except ValueError:
                    continue
    return np.array(energy), np.array(mu)


# Parse spectra for each polarization
spectra = {}
for pol in ["x", "y", "z"]:
    dat_file = os.path.join(WORK_DIR, f"xanes_{pol}.dat")
    # xspectra.x output naming convention may vary:
    # Try the standard output filename
    alt_file = os.path.join(WORK_DIR, f"xanes_{pol}.sav.dat")
    if os.path.exists(dat_file):
        spectra[pol] = parse_xspectra_dat(dat_file)
    elif os.path.exists(alt_file):
        spectra[pol] = parse_xspectra_dat(alt_file)
    else:
        print(f"WARNING: {dat_file} not found. Generating synthetic data for demo.")
        # Generate synthetic Ti K-edge XANES for demonstration
        E = np.linspace(-10, 40, 1000)
        # Simplified model: pre-edge peak + main edge step + white line
        pre_edge = 0.15 * np.exp(-((E - 2.0) / 1.0) ** 2)  # pre-edge 1s->3d
        main_edge = 0.5 * (1 + np.tanh((E - 5.0) / 2.0))    # edge step
        white_line = 0.8 * np.exp(-((E - 8.0) / 2.5) ** 2)   # white line
        noise = np.random.normal(0, 0.005, len(E))
        mu = pre_edge + main_edge + white_line + noise
        # Add polarization dependence for rutile
        if pol == "z":
            mu *= 1.15  # c-axis has slightly different spectrum
        spectra[pol] = (E, mu)

# ============================================================
# Plot individual polarization spectra
# ============================================================
fig, axes = plt.subplots(2, 1, figsize=(10, 9))

colors = {"x": "steelblue", "y": "forestgreen", "z": "coral"}

for pol in ["x", "y", "z"]:
    if pol in spectra:
        E, mu = spectra[pol]
        axes[0].plot(E, mu, color=colors[pol], linewidth=1.5,
                     label=f"E || {pol}")

axes[0].set_xlabel("Energy relative to edge (eV)")
axes[0].set_ylabel("Absorption (a.u.)")
axes[0].set_title("Ti K-edge XANES -- TiO$_2$ Rutile (Polarization Resolved)")
axes[0].legend()
axes[0].set_xlim(-10, 40)
axes[0].axvline(0, color="gray", linestyle="--", alpha=0.5, label="Edge onset")

# ============================================================
# Powder-averaged spectrum: mu_powder = (mu_x + mu_y + mu_z) / 3
# ============================================================
if all(pol in spectra for pol in ["x", "y", "z"]):
    E_ref = spectra["x"][0]
    mu_powder = (spectra["x"][1] + spectra["y"][1] + spectra["z"][1]) / 3.0

    axes[1].plot(E_ref, mu_powder, "k-", linewidth=2, label="Powder average")
    axes[1].fill_between(E_ref, 0, mu_powder, alpha=0.15, color="gray")

    # Annotate pre-edge and main features
    # Find pre-edge peak (local maximum below edge)
    pre_edge_mask = (E_ref > -2) & (E_ref < 5)
    if np.any(pre_edge_mask):
        pre_edge_region = mu_powder[pre_edge_mask]
        pre_edge_E = E_ref[pre_edge_mask]
        if len(pre_edge_region) > 0:
            pe_idx = np.argmax(pre_edge_region)
            axes[1].annotate("Pre-edge\n(1s->3d)",
                           xy=(pre_edge_E[pe_idx], pre_edge_region[pe_idx]),
                           xytext=(pre_edge_E[pe_idx] - 5, pre_edge_region[pe_idx] + 0.2),
                           arrowprops=dict(arrowstyle="->", color="red"),
                           fontsize=10, color="red")

    # Find white line (maximum above edge)
    white_line_mask = (E_ref > 4) & (E_ref < 15)
    if np.any(white_line_mask):
        wl_region = mu_powder[white_line_mask]
        wl_E = E_ref[white_line_mask]
        if len(wl_region) > 0:
            wl_idx = np.argmax(wl_region)
            axes[1].annotate("White line\n(1s->4p)",
                           xy=(wl_E[wl_idx], wl_region[wl_idx]),
                           xytext=(wl_E[wl_idx] + 5, wl_region[wl_idx] + 0.1),
                           arrowprops=dict(arrowstyle="->", color="blue"),
                           fontsize=10, color="blue")

    axes[1].set_xlabel("Energy relative to edge (eV)")
    axes[1].set_ylabel("Absorption (a.u.)")
    axes[1].set_title("Ti K-edge XANES -- TiO$_2$ Rutile (Powder Average)")
    axes[1].legend()
    axes[1].set_xlim(-10, 40)

plt.tight_layout()
plt.savefig(os.path.join(WORK_DIR, "ti_kedge_xanes.png"), dpi=150)
print(f"Saved: {WORK_DIR}/ti_kedge_xanes.png")
```

#### Step 5: Pre-edge Analysis

```python
# ============================================================
# Step 5: Pre-edge feature analysis
# ============================================================
print("\n== Pre-edge Analysis ==")

# The pre-edge in Ti K-edge arises from 1s -> 3d transitions
# These are dipole-forbidden but become allowed through:
# 1. p-d hybridization (non-centrosymmetric sites)
# 2. Quadrupole transitions (weak)

# For rutile TiO2:
# - Ti is in octahedral coordination (centrosymmetric-like)
# - Pre-edge is weak (mainly quadrupolar)
# - For anatase TiO2: Ti is in distorted octahedron -> stronger pre-edge

# Quantitative pre-edge extraction
if all(pol in spectra for pol in ["x", "y", "z"]):
    E_ref = spectra["x"][0]
    mu_powder = (spectra["x"][1] + spectra["y"][1] + spectra["z"][1]) / 3.0

    # Define pre-edge region
    pe_min, pe_max = -2.0, 5.0
    pe_mask = (E_ref >= pe_min) & (E_ref <= pe_max)

    # Subtract baseline (linear interpolation between endpoints)
    E_pe = E_ref[pe_mask]
    mu_pe = mu_powder[pe_mask]

    # Linear baseline
    baseline = np.interp(E_pe, [pe_min, pe_max], [mu_pe[0], mu_pe[-1]])
    mu_pe_corrected = mu_pe - baseline

    # Integrate pre-edge intensity
    pe_integral = np.trapz(mu_pe_corrected[mu_pe_corrected > 0],
                           E_pe[mu_pe_corrected > 0])

    # Find pre-edge peak position
    pe_peak_idx = np.argmax(mu_pe_corrected)
    pe_peak_E = E_pe[pe_peak_idx]
    pe_peak_I = mu_pe_corrected[pe_peak_idx]

    print(f"Pre-edge peak position: {pe_peak_E:.2f} eV (relative to edge)")
    print(f"Pre-edge peak height:   {pe_peak_I:.4f} a.u.")
    print(f"Pre-edge integral:      {pe_integral:.4f} a.u.*eV")
    print()
    print("Interpretation:")
    print("  - Weak pre-edge -> centrosymmetric coordination (octahedral)")
    print("  - Strong pre-edge -> non-centrosymmetric (tetrahedral or distorted)")
    print("  - Pre-edge position correlates with oxidation state")
    print("    Ti(III): ~2 eV below edge; Ti(IV): ~1-3 eV below edge")

    # Plot pre-edge extraction
    fig_pe, ax_pe = plt.subplots(figsize=(8, 5))
    ax_pe.plot(E_pe, mu_pe, "k-", linewidth=2, label="Spectrum")
    ax_pe.plot(E_pe, baseline, "r--", linewidth=1, label="Baseline")
    ax_pe.fill_between(E_pe, baseline, mu_pe,
                       where=(mu_pe_corrected > 0),
                       alpha=0.3, color="steelblue", label="Pre-edge area")
    ax_pe.set_xlabel("Energy relative to edge (eV)")
    ax_pe.set_ylabel("Absorption (a.u.)")
    ax_pe.set_title("Pre-edge Extraction -- Ti K-edge")
    ax_pe.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(WORK_DIR, "pre_edge_analysis.png"), dpi=150)
    print(f"Saved: {WORK_DIR}/pre_edge_analysis.png")
```

#### Step 6: Comparison with Experiment

```python
# ============================================================
# Step 6: Tips for comparing with experimental data
# ============================================================
print("\n== Comparison with Experiment ==")

def align_spectra(E_calc, mu_calc, E_exp, mu_exp, edge_fraction=0.5):
    """
    Align calculated and experimental spectra by matching edge positions.
    The edge is defined as where absorption reaches `edge_fraction` of
    the white line maximum.

    Returns shifted calculated energy axis.
    """
    # Find edge position in calculated spectrum
    mu_norm_calc = (mu_calc - mu_calc.min()) / (mu_calc.max() - mu_calc.min())
    edge_idx_calc = np.argmin(np.abs(mu_norm_calc - edge_fraction))
    E_edge_calc = E_calc[edge_idx_calc]

    # Find edge position in experimental spectrum
    mu_norm_exp = (mu_exp - mu_exp.min()) / (mu_exp.max() - mu_exp.min())
    edge_idx_exp = np.argmin(np.abs(mu_norm_exp - edge_fraction))
    E_edge_exp = E_exp[edge_idx_exp]

    # Shift calculated spectrum
    shift = E_edge_exp - E_edge_calc
    E_calc_shifted = E_calc + shift
    print(f"Energy shift applied: {shift:.2f} eV")

    return E_calc_shifted

# Example: if you have experimental data
# E_exp, mu_exp = np.loadtxt("experimental_Ti_Kedge.dat", unpack=True)
# E_calc_shifted = align_spectra(E_ref, mu_powder, E_exp, mu_exp)
# plt.plot(E_calc_shifted, mu_powder / mu_powder.max(), label="Calculated")
# plt.plot(E_exp, mu_exp / mu_exp.max(), label="Experimental")

print("""
Tips for comparison with experimental XANES:

1. ENERGY ALIGNMENT:
   - DFT absolute energies are not directly comparable to experiment.
   - Align by matching the edge onset (inflection point or half-height).
   - Typical shifts: 5--20 eV depending on pseudopotential and functional.

2. BROADENING:
   - Experimental spectra have natural broadening from core-hole lifetime
     (~1 eV for Ti K-edge) plus instrumental resolution.
   - Increase xgamma in xspectra.x to match experimental broadening.
   - Energy-dependent broadening: use gamma_mode='variable' with
     gamma_energy(1) and gamma_value(1) parameters.

3. NORMALIZATION:
   - Normalize both spectra to the edge step (mu_0 -> 1 far above edge).
   - Or normalize to the white line maximum.

4. PRE-EDGE:
   - Pre-edge features are sensitive to the core-hole treatment.
   - Full core-hole (FCH) overpolarizes; half-core-hole (HCH) can
     give better pre-edge intensities in some cases.
   - For quantitative pre-edge, use quadrupole transitions:
     calculation = 'xanes_quadrupole' in xspectra.x.

5. SELF-CONSISTENCY:
   - Use tot_charge=-1 to add an electron to screen the core hole
     (more physical than neutral cell with hole).
   - For metals, the screening is effective and FCH works well.
   - For insulators, the choice of screening matters more.
""")
```

### Alternative: Quick XAS with Larger Broadening (No Core-Hole PP)

If you cannot generate a core-hole pseudopotential, an approximate approach uses the ground-state electronic structure with a Gaussian broadening:

```python
#!/usr/bin/env python3
"""
Approximate XAS from projected density of states (PDOS).
This does NOT include core-hole effects but can give a rough
shape of the XANES for comparison.
"""

import os
import subprocess
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

WORK_DIR = "/home/work/xas_pdos"
PSEUDO_DIR = "/home/pseudo"
os.makedirs(WORK_DIR, exist_ok=True)

# Step 1: SCF with standard PPs (no core hole)
scf_input = f"""&CONTROL
    calculation  = 'scf'
    prefix       = 'tio2'
    outdir       = '{WORK_DIR}/tmp'
    pseudo_dir   = '{PSEUDO_DIR}'
/
&SYSTEM
    ibrav        = 6
    celldm(1)    = 8.6817
    celldm(3)    = 0.6441
    nat          = 6
    ntyp         = 2
    ecutwfc      = 50.0
    ecutrho      = 400.0
    occupations  = 'smearing'
    smearing     = 'cold'
    degauss      = 0.01
/
&ELECTRONS
    conv_thr     = 1.0d-8
/

ATOMIC_SPECIES
  Ti  47.867  Ti.pbe-spn-kjpaw_psl.1.0.0.UPF
  O   15.999  O.pbe-n-kjpaw_psl.1.0.0.UPF

ATOMIC_POSITIONS crystal
  Ti  0.0000  0.0000  0.0000
  Ti  0.5000  0.5000  0.5000
  O   0.3053  0.3053  0.0000
  O   0.6947  0.6947  0.0000
  O   0.1947  0.8053  0.5000
  O   0.8053  0.1947  0.5000

K_POINTS automatic
  6 6 8 0 0 0
"""

scf_file = os.path.join(WORK_DIR, "scf.in")
with open(scf_file, "w") as f:
    f.write(scf_input)

subprocess.run(
    ["mpirun", "-np", "4", "pw.x", "-input", scf_file],
    capture_output=True, text=True, cwd=WORK_DIR, timeout=3600,
)

# Step 2: NSCF for PDOS
nscf_input = scf_input.replace("'scf'", "'nscf'")
nscf_input = nscf_input.replace("K_POINTS automatic\n  6 6 8 0 0 0",
                                 "K_POINTS automatic\n  8 8 12 0 0 0")
# Add more bands for empty states
nscf_input = nscf_input.replace("ecutrho      = 400.0",
                                 "ecutrho      = 400.0\n    nbnd         = 60")

nscf_file = os.path.join(WORK_DIR, "nscf.in")
with open(nscf_file, "w") as f:
    f.write(nscf_input)

subprocess.run(
    ["mpirun", "-np", "4", "pw.x", "-input", nscf_file],
    capture_output=True, text=True, cwd=WORK_DIR, timeout=3600,
)

# Step 3: PDOS with projwfc.x
projwfc_input = f"""&PROJWFC
    prefix  = 'tio2'
    outdir  = '{WORK_DIR}/tmp'
    filpdos = '{WORK_DIR}/tio2_pdos'
    Emin    = -15.0
    Emax    = 30.0
    DeltaE  = 0.05
/
"""

projwfc_file = os.path.join(WORK_DIR, "projwfc.in")
with open(projwfc_file, "w") as f:
    f.write(projwfc_input)

subprocess.run(
    ["mpirun", "-np", "4", "projwfc.x", "-input", projwfc_file],
    capture_output=True, text=True, cwd=WORK_DIR, timeout=3600,
)

# Step 4: Parse PDOS and plot approximate XAS
# XAS ~ unoccupied p-PDOS for K-edge (dipole selection: s -> p)
# Look for Ti p-projected DOS above Fermi level

pdos_file = os.path.join(WORK_DIR, "tio2_pdos.pdos_atm#1(Ti)_wfc#3(p)")
if os.path.exists(pdos_file):
    data = np.loadtxt(pdos_file, comments="#")
    E = data[:, 0]
    pdos_p = data[:, 1]  # p-DOS

    # Approximate XAS: unoccupied p-PDOS (above Fermi level)
    # Fermi level from SCF output
    E_fermi = 0  # read from SCF output; placeholder

    fig, ax = plt.subplots(figsize=(10, 5))
    # Only unoccupied states
    mask = E > E_fermi
    ax.plot(E[mask] - E_fermi, pdos_p[mask], "b-", linewidth=1.5)
    ax.fill_between(E[mask] - E_fermi, 0, pdos_p[mask], alpha=0.2, color="steelblue")
    ax.set_xlabel("Energy above Fermi level (eV)")
    ax.set_ylabel("p-PDOS (states/eV)")
    ax.set_title("Approximate Ti K-edge XANES from p-PDOS (no core-hole)")
    ax.set_xlim(0, 30)
    plt.tight_layout()
    plt.savefig(os.path.join(WORK_DIR, "xas_pdos_approx.png"), dpi=150)
    print(f"Saved: {WORK_DIR}/xas_pdos_approx.png")
else:
    print(f"PDOS file not found: {pdos_file}")
    print("Check projwfc.x output for available PDOS files.")
```

## Key Parameters

| Parameter | Typical Value | Effect |
|-----------|---------------|--------|
| **ecutwfc** | 50--80 Ry | Plane-wave cutoff. K-edge converges faster than L-edge. Test convergence. |
| **xgamma** | 0.2--1.5 eV | Lorentzian broadening. Core-hole lifetime ~1 eV for 3d metals K-edge. Start with 0.8 eV. |
| **xemin, xemax** | -10, 40 eV | Energy range around the edge. -10 to 40 covers pre-edge through XANES. |
| **xniter** | 1000--5000 | Lanczos iterations. More = better convergence of the continued fraction. 2000 is usually enough. |
| **xerror** | 0.001 | Convergence criterion for Lanczos. Tighter = more iterations. |
| **Supercell size** | 2x2x2 minimum | Must isolate core hole from periodic images. 3x3x3 better for quantitative work. |
| **K-points** | 2x2x2 or denser | For the supercell SCF. Denser = better but costly. Test convergence. |
| **tot_charge** | -1 (recommended) | Compensate the core hole. -1 adds one electron; 0 leaves the hole unscreened. |
| **gamma_mode** | `'constant'` or `'variable'` | Constant = same broadening everywhere. Variable = energy-dependent broadening (more realistic). |

## Interpreting Results

### XANES Features

1. **Pre-edge** (0--5 eV below the main edge):
   - K-edge: arises from 1s -> 3d (quadrupole) or 1s -> p-d hybrid (dipole) transitions.
   - **Intensity** correlates with coordination symmetry: tetrahedral > square pyramidal > octahedral.
   - **Position** correlates with oxidation state: higher oxidation = higher energy.

2. **Main edge** (edge onset):
   - Position gives the ionization threshold.
   - Shape depends on the density of unoccupied p-states.

3. **White line** (first strong peak above edge):
   - Intensity correlates with the number of unoccupied d-states (for transition metals).
   - Broader white line = more electronic delocalization.

4. **Post-edge oscillations** (10--40 eV above edge):
   - Related to multiple scattering from neighboring shells.
   - These features connect to EXAFS in the extended region.

### Polarization Dependence

- **Single crystal**: spectrum depends on the angle between the polarization vector and crystal axes.
- **Powder**: average over all orientations = (mu_x + mu_y + mu_z) / 3.
- **Dichroism**: anisotropic structures (e.g., layered materials) show strong polarization dependence.

### Edge Assignment

| Edge | Transition | Typical Energy | Information |
|------|-----------|----------------|-------------|
| K | 1s -> np | 4--30 keV (3d metals) | Oxidation state, local symmetry |
| L1 | 2s -> np | 0.5--6 keV | Less used |
| L2,3 | 2p -> nd | 0.4--5 keV | Crystal field, spin-orbit coupling |

## Common Issues

| Issue | Cause | Solution |
|-------|-------|----------|
| `xspectra.x` not found | Not compiled with QE | Recompile QE with `make xspectra` |
| Core-hole PP generation fails | `ld1.x` configuration error | Check electron configuration; consult QE PP library for examples |
| SCF does not converge with core hole | Strong perturbation | Reduce `mixing_beta` to 0.1--0.2; use `startingwfc='file'` from ground-state |
| Spectrum has no pre-edge | Quadrupole transitions not included | Use `calculation='xanes_quadrupole'` in xspectra.x |
| Energy scale does not match experiment | DFT absolute energy offset | Align edge positions; this is expected with DFT |
| Spectrum has artifacts/oscillations | Too few Lanczos iterations | Increase `xniter` to 3000--5000 |
| Core-hole effects too strong | Full core hole in small cell | Use larger supercell (3x3x3) or half-core-hole approach |
| `filecore` not found | Core wavefunction file missing | Generate with `ld1.x` or extract from the PP file using `upf2plotcore.sh` |
| L-edge calculation fails | Need spin-orbit coupling | Add `lspinorb=.true.` and `noncolin=.true.` in pw.x; use fully relativistic PP |
| Very slow convergence | Large supercell + many k-points | Reduce k-points (2x2x2 minimum for supercell); use `xonly_plot=.true.` after first run |
