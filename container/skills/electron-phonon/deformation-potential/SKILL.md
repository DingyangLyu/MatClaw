# Deformation Potential Calculation

## When to Use

- You need the **deformation potential constant** (acoustic or optical) for a semiconductor or insulator.
- You are modeling **carrier scattering rates** and **mobility** limited by phonon scattering.
- You want to understand how the **band edge (CBM or VBM) shifts under strain**.
- You are computing **electron-phonon coupling in semiconductors** where the Fermi-surface formalism does not apply.
- You need the **absolute deformation potential** of a band edge for transport or device modeling.
- You are studying **piezoelectric** or **piezoresistive** response at the electronic level.

Deformation potentials quantify how electronic energies shift when the lattice is deformed. Unlike the metallic el-ph coupling (lambda, alpha2F), deformation potentials apply to **semiconductors and insulators** where there is a band gap.

Two types:
- **Acoustic deformation potential (D_ac)**: Band edge shift per unit strain. Relevant for acoustic phonon scattering of carriers.
- **Optical deformation potential (D_opt or D_0)**: Band edge shift per unit displacement of the sublattice (optical phonon modes). Relevant for optical phonon scattering, especially in polar semiconductors.

## Method Selection

| Scenario | Tool | Method |
|---|---|---|
| Acoustic deformation potential, any system | QE (`pw.x`) | Apply hydrostatic/uniaxial strain, track band edge shift, fit slope |
| Acoustic deformation potential, any system | VASP | Same strain approach with VASP SCF |
| Optical deformation potential (non-polar) | QE (`pw.x`) | Frozen-phonon approach: displace atoms along optical mode, track band edge |
| Optical deformation potential (polar, Frohlich) | QE (`ph.x`) | Born effective charges + dielectric constant for Frohlich coupling |
| High-throughput screening | VASP + pymatgen | Automated strain + band edge tracking with Materials Project workflows |
| Alloy or surface system | QE or VASP | Supercell approach, may need unfolding |

## Prerequisites

**For QE workflow:**
- Quantum ESPRESSO 7.5: `pw.x`, optionally `ph.x` for optical modes
- Pseudopotentials (SSSP, PSlibrary, or SG15)
- Python with `numpy`, `scipy`, `matplotlib`, `pymatgen`

**For VASP workflow (future):**
- VASP 6.x with PAW potentials
- `pymatgen`, `vaspkit` (optional)
- Access to external VASP execution environment

## Detailed Steps

### Method A: Acoustic Deformation Potential via Strain (QE)

The acoustic deformation potential is defined as:

```
D_ac = dE_edge / d(epsilon)
```

where E_edge is the band edge energy (CBM or VBM) and epsilon is the strain. For hydrostatic strain, D_ac relates to the volume deformation potential:

```
D_V = V * dE_edge / dV = (1/3) * dE_edge / d(epsilon_hydro)
```

The workflow applies a series of strains, runs SCF + band edge determination at each strain, and fits the linear slope.

#### Step 1: Relax the structure at zero strain

```
cat > relax.in << 'PWSCF_INPUT'
&CONTROL
    calculation   = 'vc-relax'
    prefix        = 'si'
    pseudo_dir    = './pseudo/'
    outdir        = './tmp/'
    tprnfor       = .true.
    tstress       = .true.
    forc_conv_thr = 1.0d-5
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
    degauss       = 0.001
    nbnd          = 12
/
&ELECTRONS
    conv_thr      = 1.0d-10
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
  Si  28.0855  Si.pbe-n-kjpaw_psl.1.0.0.UPF

ATOMIC_POSITIONS {crystal}
  Si  0.00  0.00  0.00
  Si  0.25  0.25  0.25

K_POINTS {automatic}
  12 12 12  0 0 0
PWSCF_INPUT
```

```bash
mpirun -np 4 pw.x -npool 2 < relax.in > relax.out 2>&1
grep "convergence has been achieved" relax.out
grep "Final enthalpy" relax.out
```

#### Step 2: Generate strained structures and compute band edges

```python
#!/usr/bin/env python3
"""
deformation_potential_acoustic.py
Compute acoustic deformation potential by applying hydrostatic and uniaxial
strains, running SCF + NSCF at each strain, and fitting band edge vs strain.

Complete workflow:
  1. Read relaxed structure
  2. Apply strains (hydrostatic and/or uniaxial)
  3. Run SCF at each strain point
  4. Run NSCF on dense k-grid to get accurate band edges
  5. Extract VBM, CBM
  6. Fit D_ac = dE/d(epsilon)
"""
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os
import subprocess
import sys

from pymatgen.core import Structure
from pymatgen.io.espresso.inputs import PWInput

# =============================================================================
# Configuration
# =============================================================================

STRUCTURE_FILE = "relaxed_structure.cif"  # Relaxed structure
PSEUDO_DIR = "./pseudo/"
PREFIX = "si"
NP = 4           # MPI processes
NPOOL = 2        # k-point pools

# Strain range: -2% to +2% in 0.5% steps (9 points including 0)
STRAINS = np.arange(-0.02, 0.021, 0.005)

# Strain type: 'hydrostatic' (isotropic) or 'uniaxial_z' (along c-axis)
STRAIN_TYPE = 'hydrostatic'

# QE parameters
ECUTWFC = 60.0
ECUTRHO = 480.0
K_GRID_SCF = (12, 12, 12)
K_GRID_NSCF = (20, 20, 20)   # Denser for accurate band edges
NBND = 12

WORK_DIR = "/tmp/deformation_potential"
os.makedirs(WORK_DIR, exist_ok=True)

# =============================================================================
# 1. Read relaxed structure
# =============================================================================

struct = Structure.from_file(STRUCTURE_FILE)
print(f"Relaxed structure: {struct.composition}")
print(f"Lattice parameters: a={struct.lattice.a:.4f}, "
      f"b={struct.lattice.b:.4f}, c={struct.lattice.c:.4f} Angstrom")

# =============================================================================
# 2. Apply strains and generate input files
# =============================================================================

def apply_strain(structure, epsilon, strain_type='hydrostatic'):
    """
    Apply strain to a structure.

    Parameters
    ----------
    structure : pymatgen Structure
    epsilon : float
        Strain magnitude (e.g., 0.01 for 1% strain)
    strain_type : str
        'hydrostatic': isotropic scaling
        'uniaxial_z': strain along c-axis only
        'uniaxial_x': strain along a-axis only
        'biaxial_xy': equal strain in a and b, free along c

    Returns
    -------
    strained_structure : pymatgen Structure
    """
    s = structure.copy()

    if strain_type == 'hydrostatic':
        # Isotropic scaling: all lattice vectors scaled by (1 + epsilon)
        s.apply_strain(epsilon)
    elif strain_type == 'uniaxial_z':
        # Only strain along c-axis
        strain_matrix = np.eye(3)
        strain_matrix[2, 2] = 1.0 + epsilon
        s.apply_strain(strain_matrix - np.eye(3))
    elif strain_type == 'uniaxial_x':
        strain_matrix = np.eye(3)
        strain_matrix[0, 0] = 1.0 + epsilon
        s.apply_strain(strain_matrix - np.eye(3))
    elif strain_type == 'biaxial_xy':
        strain_matrix = np.eye(3)
        strain_matrix[0, 0] = 1.0 + epsilon
        strain_matrix[1, 1] = 1.0 + epsilon
        s.apply_strain(strain_matrix - np.eye(3))
    else:
        raise ValueError(f"Unknown strain_type: {strain_type}")

    return s


def write_qe_scf_input(structure, filename, prefix, pseudo_dir,
                         ecutwfc, ecutrho, k_grid):
    """Write QE SCF input file from pymatgen Structure."""
    # Get species list
    species = [site.specie.symbol for site in structure]
    unique_species = list(dict.fromkeys(species))

    # Build input manually for control
    lines = []
    lines.append("&CONTROL")
    lines.append(f"    calculation   = 'scf'")
    lines.append(f"    prefix        = '{prefix}'")
    lines.append(f"    pseudo_dir    = '{pseudo_dir}'")
    lines.append(f"    outdir        = './tmp/'")
    lines.append(f"    verbosity     = 'high'")
    lines.append("/")
    lines.append("&SYSTEM")
    lines.append(f"    ibrav         = 0")
    lines.append(f"    nat           = {len(structure)}")
    lines.append(f"    ntyp          = {len(unique_species)}")
    lines.append(f"    ecutwfc       = {ecutwfc}")
    lines.append(f"    ecutrho       = {ecutrho}")
    lines.append(f"    occupations   = 'smearing'")
    lines.append(f"    smearing      = 'gaussian'")
    lines.append(f"    degauss       = 0.001")
    lines.append(f"    nbnd          = {NBND}")
    lines.append("/")
    lines.append("&ELECTRONS")
    lines.append(f"    conv_thr      = 1.0d-10")
    lines.append("/")
    lines.append("")

    # Cell parameters in Angstrom
    lines.append("CELL_PARAMETERS {angstrom}")
    for vec in structure.lattice.matrix:
        lines.append(f"  {vec[0]:16.10f} {vec[1]:16.10f} {vec[2]:16.10f}")
    lines.append("")

    # Atomic species
    # Map species to pseudopotential files
    pp_map = {
        'Si': ('28.0855', 'Si.pbe-n-kjpaw_psl.1.0.0.UPF'),
        'Ge': ('72.630', 'Ge.pbe-dn-kjpaw_psl.1.0.0.UPF'),
        'C':  ('12.011', 'C.pbe-n-kjpaw_psl.1.0.0.UPF'),
        'Ga': ('69.723', 'Ga.pbe-dn-kjpaw_psl.1.0.0.UPF'),
        'As': ('74.922', 'As.pbe-n-kjpaw_psl.1.0.0.UPF'),
        'N':  ('14.007', 'N.pbe-n-kjpaw_psl.1.0.0.UPF'),
    }
    lines.append("ATOMIC_SPECIES")
    for sp in unique_species:
        mass, pp = pp_map.get(sp, ('1.0', f'{sp}.UPF'))
        lines.append(f"  {sp:4s} {mass}  {pp}")
    lines.append("")

    lines.append("ATOMIC_POSITIONS {crystal}")
    for site in structure:
        fc = site.frac_coords
        lines.append(f"  {site.specie.symbol:4s} {fc[0]:14.10f} {fc[1]:14.10f} {fc[2]:14.10f}")
    lines.append("")

    lines.append("K_POINTS {automatic}")
    lines.append(f"  {k_grid[0]} {k_grid[1]} {k_grid[2]}  0 0 0")

    with open(filename, 'w') as f:
        f.write('\n'.join(lines))


def run_qe(input_file, output_file, np_procs, npool):
    """Run QE pw.x calculation."""
    cmd = f"mpirun -np {np_procs} pw.x -npool {npool} < {input_file} > {output_file} 2>&1"
    result = subprocess.run(cmd, shell=True, cwd=os.path.dirname(input_file))
    return result.returncode


def extract_band_edges(output_file):
    """
    Extract VBM and CBM from QE output.
    For semiconductors with 'smearing', look for the highest occupied
    and lowest unoccupied eigenvalue.

    Returns (VBM_eV, CBM_eV, E_gap_eV) or (None, None, None) if parsing fails.
    """
    with open(output_file, 'r') as f:
        content = f.read()

    # Look for "highest occupied, lowest unoccupied level"
    import re
    match = re.search(
        r'highest occupied, lowest unoccupied level.*?:\s+([-\d.]+)\s+([-\d.]+)',
        content
    )
    if match:
        vbm = float(match.group(1))
        cbm = float(match.group(2))
        return vbm, cbm, cbm - vbm

    # Alternative: look for Fermi energy (metallic case)
    match_fermi = re.search(r'the Fermi energy is\s+([-\d.]+)', content)
    if match_fermi:
        ef = float(match_fermi.group(1))
        return ef, ef, 0.0

    return None, None, None


# =============================================================================
# 3. Run calculations at each strain
# =============================================================================

print(f"\n{'='*60}")
print(f"Acoustic Deformation Potential Calculation")
print(f"Strain type: {STRAIN_TYPE}")
print(f"Strain range: {STRAINS[0]*100:.1f}% to {STRAINS[-1]*100:.1f}%")
print(f"{'='*60}\n")

results = []

for i, eps in enumerate(STRAINS):
    print(f"\n--- Strain {eps*100:+.1f}% ({i+1}/{len(STRAINS)}) ---")

    strain_dir = os.path.join(WORK_DIR, f"strain_{eps:+.4f}")
    os.makedirs(strain_dir, exist_ok=True)
    os.makedirs(os.path.join(strain_dir, 'tmp'), exist_ok=True)

    # Apply strain
    strained = apply_strain(struct, eps, STRAIN_TYPE)
    volume = strained.volume

    # Write SCF input
    scf_in = os.path.join(strain_dir, 'scf.in')
    scf_out = os.path.join(strain_dir, 'scf.out')
    write_qe_scf_input(strained, scf_in, PREFIX, os.path.abspath(PSEUDO_DIR),
                         ECUTWFC, ECUTRHO, K_GRID_SCF)

    # Run SCF
    print(f"  Running SCF (V = {volume:.3f} A^3)...")
    ret = run_qe(scf_in, scf_out, NP, NPOOL)

    if ret != 0:
        print(f"  WARNING: SCF returned non-zero exit code {ret}")

    # Extract band edges
    vbm, cbm, gap = extract_band_edges(scf_out)

    if vbm is not None:
        print(f"  VBM = {vbm:.4f} eV, CBM = {cbm:.4f} eV, gap = {gap:.4f} eV")
        results.append({
            'strain': eps,
            'volume': volume,
            'vbm': vbm,
            'cbm': cbm,
            'gap': gap,
        })
    else:
        print(f"  WARNING: Could not extract band edges from {scf_out}")

# =============================================================================
# 4. Fit deformation potentials
# =============================================================================

if len(results) < 3:
    print("\nERROR: Not enough data points for fitting. Need at least 3.")
    sys.exit(1)

strains_arr = np.array([r['strain'] for r in results])
vbm_arr = np.array([r['vbm'] for r in results])
cbm_arr = np.array([r['cbm'] for r in results])
gap_arr = np.array([r['gap'] for r in results])
vol_arr = np.array([r['volume'] for r in results])

# Linear fit: E = E0 + D_ac * epsilon
# For hydrostatic strain, D_ac is the hydrostatic deformation potential

# VBM deformation potential
coeffs_vbm = np.polyfit(strains_arr, vbm_arr, 1)
D_ac_vbm = coeffs_vbm[0]  # eV per unit strain

# CBM deformation potential
coeffs_cbm = np.polyfit(strains_arr, cbm_arr, 1)
D_ac_cbm = coeffs_cbm[0]

# Gap deformation potential
coeffs_gap = np.polyfit(strains_arr, gap_arr, 1)
D_ac_gap = coeffs_gap[0]

# Volume deformation potential: D_V = (V/3) * dE/dV for hydrostatic
# Or equivalently, D_V = (1/3) * dE/d(epsilon_hydro) for small hydrostatic strain
if STRAIN_TYPE == 'hydrostatic':
    D_V_vbm = D_ac_vbm / 3.0
    D_V_cbm = D_ac_cbm / 3.0
    D_V_gap = D_ac_gap / 3.0
    print(f"\n--- Volume Deformation Potentials (hydrostatic) ---")
    print(f"  D_V(VBM) = {D_V_vbm:.3f} eV")
    print(f"  D_V(CBM) = {D_V_cbm:.3f} eV")
    print(f"  D_V(gap) = {D_V_gap:.3f} eV")

print(f"\n--- Deformation Potentials ({STRAIN_TYPE}) ---")
print(f"  D_ac(VBM) = {D_ac_vbm:.3f} eV")
print(f"  D_ac(CBM) = {D_ac_cbm:.3f} eV")
print(f"  D_ac(gap) = {D_ac_gap:.3f} eV")

# Fit quality
vbm_fit = np.polyval(coeffs_vbm, strains_arr)
cbm_fit = np.polyval(coeffs_cbm, strains_arr)
r2_vbm = 1.0 - np.sum((vbm_arr - vbm_fit)**2) / np.sum((vbm_arr - np.mean(vbm_arr))**2)
r2_cbm = 1.0 - np.sum((cbm_arr - cbm_fit)**2) / np.sum((cbm_arr - np.mean(cbm_arr))**2)
print(f"  R^2(VBM fit) = {r2_vbm:.6f}")
print(f"  R^2(CBM fit) = {r2_cbm:.6f}")

# =============================================================================
# 5. Save and plot
# =============================================================================

# Save numerical data
data_out = np.column_stack([strains_arr, vol_arr, vbm_arr, cbm_arr, gap_arr])
np.savetxt(
    os.path.join(WORK_DIR, 'deformation_potential_data.dat'),
    data_out,
    header="strain  volume(A^3)  VBM(eV)  CBM(eV)  gap(eV)",
    fmt="%10.5f %12.4f %12.6f %12.6f %12.6f"
)

# Plot
fig, axes = plt.subplots(1, 3, figsize=(16, 5))

# VBM vs strain
ax = axes[0]
ax.plot(strains_arr * 100, vbm_arr, 'o', color='steelblue', markersize=7)
strain_fine = np.linspace(strains_arr[0], strains_arr[-1], 100) * 100
ax.plot(strain_fine, np.polyval(coeffs_vbm, strain_fine / 100),
        '-', color='steelblue', linewidth=1.5)
ax.set_xlabel('Strain (%)', fontsize=12)
ax.set_ylabel('VBM (eV)', fontsize=12)
ax.set_title(f'VBM: D_ac = {D_ac_vbm:.2f} eV', fontsize=13)
ax.grid(True, alpha=0.3)

# CBM vs strain
ax = axes[1]
ax.plot(strains_arr * 100, cbm_arr, 's', color='darkorange', markersize=7)
ax.plot(strain_fine, np.polyval(coeffs_cbm, strain_fine / 100),
        '-', color='darkorange', linewidth=1.5)
ax.set_xlabel('Strain (%)', fontsize=12)
ax.set_ylabel('CBM (eV)', fontsize=12)
ax.set_title(f'CBM: D_ac = {D_ac_cbm:.2f} eV', fontsize=13)
ax.grid(True, alpha=0.3)

# Gap vs strain
ax = axes[2]
ax.plot(strains_arr * 100, gap_arr, 'D', color='seagreen', markersize=7)
ax.plot(strain_fine, np.polyval(coeffs_gap, strain_fine / 100),
        '-', color='seagreen', linewidth=1.5)
ax.set_xlabel('Strain (%)', fontsize=12)
ax.set_ylabel('Band gap (eV)', fontsize=12)
ax.set_title(f'Gap: D_ac = {D_ac_gap:.2f} eV', fontsize=13)
ax.grid(True, alpha=0.3)

fig.suptitle(f'Acoustic Deformation Potential ({STRAIN_TYPE} strain)', fontsize=14)
fig.tight_layout()
fig.savefig(os.path.join(WORK_DIR, 'deformation_potential_acoustic.png'),
            dpi=200, bbox_inches='tight')
plt.close()
print(f"\nSaved: {WORK_DIR}/deformation_potential_acoustic.png")

# Write summary
with open(os.path.join(WORK_DIR, 'deformation_potential_summary.txt'), 'w') as f:
    f.write("Acoustic Deformation Potential Summary\n")
    f.write("=" * 50 + "\n\n")
    f.write(f"Strain type: {STRAIN_TYPE}\n")
    f.write(f"Material: {struct.composition}\n\n")
    f.write(f"D_ac(VBM) = {D_ac_vbm:.4f} eV  (R^2 = {r2_vbm:.6f})\n")
    f.write(f"D_ac(CBM) = {D_ac_cbm:.4f} eV  (R^2 = {r2_cbm:.6f})\n")
    f.write(f"D_ac(gap) = {D_ac_gap:.4f} eV\n")
    if STRAIN_TYPE == 'hydrostatic':
        f.write(f"\nVolume deformation potentials:\n")
        f.write(f"D_V(VBM) = {D_V_vbm:.4f} eV\n")
        f.write(f"D_V(CBM) = {D_V_cbm:.4f} eV\n")
        f.write(f"D_V(gap) = {D_V_gap:.4f} eV\n")

print(f"Saved: {WORK_DIR}/deformation_potential_summary.txt")
```

Run:

```bash
python3 deformation_potential_acoustic.py
```

### Method B: Optical Deformation Potential (Frozen Phonon)

The optical deformation potential couples carriers to optical phonon modes. For a zone-center optical mode, the deformation potential is:

```
D_opt = dE_edge / d(u)
```

where u is the atomic displacement amplitude along the optical phonon eigenvector.

```python
#!/usr/bin/env python3
"""
deformation_potential_optical.py
Compute optical deformation potential using the frozen-phonon method.

Workflow:
  1. Compute zone-center phonon eigenvectors (from ph.x or manually)
  2. Displace atoms along the optical mode eigenvector by various amplitudes
  3. Run SCF at each displacement
  4. Track band edge shift
  5. Fit D_opt = dE_edge / d(u)
"""
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os
import subprocess

from pymatgen.core import Structure

# =============================================================================
# Configuration
# =============================================================================

STRUCTURE_FILE = "relaxed_structure.cif"
PSEUDO_DIR = "./pseudo/"
PREFIX = "si"
NP = 4
NPOOL = 2
ECUTWFC = 60.0
ECUTRHO = 480.0

# Displacement amplitudes (Angstrom)
# Use small displacements for linear response regime
AMPLITUDES = np.array([-0.06, -0.04, -0.02, 0.0, 0.02, 0.04, 0.06])

# Optical mode eigenvector (must be determined from phonon calculation)
# For diamond-structure Si, the Gamma-point optical mode is the two sublattices
# moving in opposite directions.
# Eigenvector: atom 0 moves +z, atom 1 moves -z (normalized)
# Format: list of (dx, dy, dz) per atom
OPTICAL_EIGENVECTOR = [
    [0.0, 0.0, +1.0],   # Atom 0: Si at (0,0,0)
    [0.0, 0.0, -1.0],   # Atom 1: Si at (1/4,1/4,1/4)
]
# Normalize
eigvec = np.array(OPTICAL_EIGENVECTOR)
norm = np.sqrt(np.sum(eigvec**2) / len(eigvec))
eigvec_normalized = eigvec / norm

# Alternatively, read eigenvector from ph.x output (dynmat.x output)
# This example uses a manually specified eigenvector for simplicity.

WORK_DIR = "/tmp/optical_deformation_potential"
os.makedirs(WORK_DIR, exist_ok=True)

# =============================================================================
# Helper functions
# =============================================================================

def write_scf_input(structure, filename, prefix, pseudo_dir,
                     ecutwfc, ecutrho, k_grid=(12,12,12)):
    """Write QE SCF input from pymatgen Structure."""
    species = [site.specie.symbol for site in structure]
    unique_species = list(dict.fromkeys(species))

    pp_map = {
        'Si': ('28.0855', 'Si.pbe-n-kjpaw_psl.1.0.0.UPF'),
        'Ge': ('72.630', 'Ge.pbe-dn-kjpaw_psl.1.0.0.UPF'),
        'C':  ('12.011', 'C.pbe-n-kjpaw_psl.1.0.0.UPF'),
        'Ga': ('69.723', 'Ga.pbe-dn-kjpaw_psl.1.0.0.UPF'),
        'As': ('74.922', 'As.pbe-n-kjpaw_psl.1.0.0.UPF'),
        'N':  ('14.007', 'N.pbe-n-kjpaw_psl.1.0.0.UPF'),
    }

    lines = []
    lines.append("&CONTROL")
    lines.append(f"    calculation   = 'scf'")
    lines.append(f"    prefix        = '{prefix}'")
    lines.append(f"    pseudo_dir    = '{pseudo_dir}'")
    lines.append(f"    outdir        = './tmp/'")
    lines.append(f"    verbosity     = 'high'")
    lines.append("/")
    lines.append("&SYSTEM")
    lines.append(f"    ibrav         = 0")
    lines.append(f"    nat           = {len(structure)}")
    lines.append(f"    ntyp          = {len(unique_species)}")
    lines.append(f"    ecutwfc       = {ecutwfc}")
    lines.append(f"    ecutrho       = {ecutrho}")
    lines.append(f"    occupations   = 'smearing'")
    lines.append(f"    smearing      = 'gaussian'")
    lines.append(f"    degauss       = 0.001")
    lines.append(f"    nbnd          = 12")
    lines.append("/")
    lines.append("&ELECTRONS")
    lines.append(f"    conv_thr      = 1.0d-10")
    lines.append("/")
    lines.append("")
    lines.append("CELL_PARAMETERS {angstrom}")
    for vec in structure.lattice.matrix:
        lines.append(f"  {vec[0]:16.10f} {vec[1]:16.10f} {vec[2]:16.10f}")
    lines.append("")
    lines.append("ATOMIC_SPECIES")
    for sp in unique_species:
        mass, pp = pp_map.get(sp, ('1.0', f'{sp}.UPF'))
        lines.append(f"  {sp:4s} {mass}  {pp}")
    lines.append("")
    lines.append("ATOMIC_POSITIONS {angstrom}")
    for site in structure:
        c = site.coords
        lines.append(f"  {site.specie.symbol:4s} {c[0]:14.10f} {c[1]:14.10f} {c[2]:14.10f}")
    lines.append("")
    lines.append("K_POINTS {automatic}")
    lines.append(f"  {k_grid[0]} {k_grid[1]} {k_grid[2]}  0 0 0")

    with open(filename, 'w') as f:
        f.write('\n'.join(lines))


def extract_band_edges(output_file):
    """Extract VBM and CBM from QE output."""
    import re
    with open(output_file, 'r') as f:
        content = f.read()
    match = re.search(
        r'highest occupied, lowest unoccupied level.*?:\s+([-\d.]+)\s+([-\d.]+)',
        content)
    if match:
        return float(match.group(1)), float(match.group(2))
    match_fermi = re.search(r'the Fermi energy is\s+([-\d.]+)', content)
    if match_fermi:
        ef = float(match_fermi.group(1))
        return ef, ef
    return None, None


def run_qe(input_file, output_file, np_procs, npool):
    """Run QE pw.x."""
    cmd = f"mpirun -np {np_procs} pw.x -npool {npool} < {input_file} > {output_file} 2>&1"
    subprocess.run(cmd, shell=True, cwd=os.path.dirname(input_file))

# =============================================================================
# Main workflow
# =============================================================================

struct = Structure.from_file(STRUCTURE_FILE)
n_atoms = len(struct)

print(f"Structure: {struct.composition}, {n_atoms} atoms")
print(f"Optical mode eigenvector (normalized):")
for i, ev in enumerate(eigvec_normalized):
    print(f"  Atom {i}: ({ev[0]:.4f}, {ev[1]:.4f}, {ev[2]:.4f})")

results = []

for amp in AMPLITUDES:
    print(f"\n--- Displacement amplitude: {amp:+.4f} Angstrom ---")

    disp_dir = os.path.join(WORK_DIR, f"disp_{amp:+.4f}")
    os.makedirs(disp_dir, exist_ok=True)
    os.makedirs(os.path.join(disp_dir, 'tmp'), exist_ok=True)

    # Create displaced structure
    displaced = struct.copy()
    cart_coords = displaced.cart_coords.copy()
    for i_atom in range(n_atoms):
        cart_coords[i_atom] += amp * eigvec_normalized[i_atom]

    # Create new structure with displaced positions
    displaced = Structure(
        displaced.lattice, displaced.species, cart_coords,
        coords_are_cartesian=True
    )

    # Write and run SCF
    scf_in = os.path.join(disp_dir, 'scf.in')
    scf_out = os.path.join(disp_dir, 'scf.out')
    write_scf_input(displaced, scf_in, PREFIX, os.path.abspath(PSEUDO_DIR),
                     ECUTWFC, ECUTRHO)
    run_qe(scf_in, scf_out, NP, NPOOL)

    vbm, cbm = extract_band_edges(scf_out)
    if vbm is not None:
        print(f"  VBM = {vbm:.4f} eV, CBM = {cbm:.4f} eV")
        results.append({'amplitude': amp, 'vbm': vbm, 'cbm': cbm, 'gap': cbm - vbm})
    else:
        print(f"  WARNING: Could not extract band edges.")

# Fit optical deformation potential
amps = np.array([r['amplitude'] for r in results])
vbms = np.array([r['vbm'] for r in results])
cbms = np.array([r['cbm'] for r in results])
gaps = np.array([r['gap'] for r in results])

coeffs_vbm = np.polyfit(amps, vbms, 1)
coeffs_cbm = np.polyfit(amps, cbms, 1)
coeffs_gap = np.polyfit(amps, gaps, 1)

D_opt_vbm = coeffs_vbm[0]  # eV / Angstrom
D_opt_cbm = coeffs_cbm[0]
D_opt_gap = coeffs_gap[0]

print(f"\n--- Optical Deformation Potentials ---")
print(f"  D_opt(VBM) = {D_opt_vbm:.3f} eV/Angstrom")
print(f"  D_opt(CBM) = {D_opt_cbm:.3f} eV/Angstrom")
print(f"  D_opt(gap) = {D_opt_gap:.3f} eV/Angstrom")

# Plot
fig, ax = plt.subplots(figsize=(8, 5.5))
ax.plot(amps, vbms, 'o-', color='steelblue', markersize=7, linewidth=1.5,
        label=f'VBM (D = {D_opt_vbm:.2f} eV/A)')
ax.plot(amps, cbms, 's-', color='darkorange', markersize=7, linewidth=1.5,
        label=f'CBM (D = {D_opt_cbm:.2f} eV/A)')
ax.set_xlabel('Displacement amplitude (Angstrom)', fontsize=13)
ax.set_ylabel('Energy (eV)', fontsize=13)
ax.set_title('Optical Deformation Potential (Frozen Phonon)', fontsize=14)
ax.legend(fontsize=11)
ax.grid(True, alpha=0.3)
fig.tight_layout()
fig.savefig(os.path.join(WORK_DIR, 'deformation_potential_optical.png'),
            dpi=200, bbox_inches='tight')
plt.close()
print(f"\nSaved: {WORK_DIR}/deformation_potential_optical.png")
```

### Method C: VASP Workflow (Future)

The same strain-based approach applies to VASP. The key steps are:

```python
#!/usr/bin/env python3
"""
deformation_potential_vasp.py
Compute acoustic deformation potential using VASP.

This script generates VASP input files (POSCAR, INCAR, KPOINTS) for
each strain point. VASP execution must be done externally.

Workflow:
  1. Read relaxed CONTCAR/POSCAR
  2. Apply hydrostatic/uniaxial strains
  3. Generate VASP inputs at each strain
  4. [Run VASP externally]
  5. Parse vasprun.xml for band edges
  6. Fit deformation potential
"""
import numpy as np
import os

from pymatgen.core import Structure
from pymatgen.io.vasp import Poscar, Incar, Kpoints

# --- Configuration ---
STRUCTURE_FILE = "CONTCAR"   # Relaxed VASP structure
STRAINS = np.arange(-0.02, 0.021, 0.005)
STRAIN_TYPE = 'hydrostatic'

INCAR_SETTINGS = {
    'PREC': 'Accurate',
    'ENCUT': 520,
    'EDIFF': 1e-7,
    'ISMEAR': 0,         # Gaussian for semiconductors
    'SIGMA': 0.05,
    'LWAVE': False,
    'LCHARG': False,
    'LORBIT': 11,        # Projected DOS for band character
    'NEDOS': 2001,
}

KPOINTS_GRID = (12, 12, 12)

WORK_DIR = "/tmp/deformation_potential_vasp"
os.makedirs(WORK_DIR, exist_ok=True)

# --- Read structure ---
struct = Structure.from_file(STRUCTURE_FILE)
print(f"Structure: {struct.composition}")

# --- Generate inputs for each strain ---
for eps in STRAINS:
    strain_dir = os.path.join(WORK_DIR, f"strain_{eps:+.4f}")
    os.makedirs(strain_dir, exist_ok=True)

    strained = struct.copy()
    if STRAIN_TYPE == 'hydrostatic':
        strained.apply_strain(eps)
    elif STRAIN_TYPE == 'uniaxial_z':
        strain_matrix = np.eye(3)
        strain_matrix[2, 2] = 1.0 + eps
        strained.apply_strain(strain_matrix - np.eye(3))

    # Write POSCAR
    Poscar(strained).write_file(os.path.join(strain_dir, 'POSCAR'))

    # Write INCAR
    Incar(INCAR_SETTINGS).write_file(os.path.join(strain_dir, 'INCAR'))

    # Write KPOINTS
    Kpoints.automatic_density_by_vol(strained, 1000).write_file(
        os.path.join(strain_dir, 'KPOINTS'))

    print(f"Strain {eps*100:+.1f}%: wrote inputs to {strain_dir}/")

print(f"\nGenerated {len(STRAINS)} strain directories.")
print("Copy POTCAR to each directory and run VASP.")
print("After VASP completes, parse vasprun.xml for band edges:")
print()
print("  from pymatgen.io.vasp import Vasprun")
print("  vr = Vasprun('vasprun.xml')")
print("  bs = vr.get_band_structure()")
print("  vbm = bs.get_vbm()['energy']")
print("  cbm = bs.get_cbm()['energy']")
```

After running VASP at each strain point, parse and fit:

```python
#!/usr/bin/env python3
"""
parse_vasp_deformation_potential.py
Parse VASP results at each strain point and fit deformation potential.
Run this after VASP completes at all strain points.
"""
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os
import glob

from pymatgen.io.vasp import Vasprun

WORK_DIR = "/tmp/deformation_potential_vasp"

# Find all strain directories
strain_dirs = sorted(glob.glob(os.path.join(WORK_DIR, "strain_*")))

results = []
for sd in strain_dirs:
    # Extract strain from directory name
    eps_str = os.path.basename(sd).replace("strain_", "")
    eps = float(eps_str)

    vasprun_file = os.path.join(sd, 'vasprun.xml')
    if not os.path.isfile(vasprun_file):
        print(f"WARNING: No vasprun.xml in {sd}")
        continue

    try:
        vr = Vasprun(vasprun_file, parse_dos=False, parse_potcar_file=False)
        bs = vr.get_band_structure()
        vbm = bs.get_vbm()['energy']
        cbm = bs.get_cbm()['energy']
        gap = bs.get_band_gap()['energy']
        results.append({'strain': eps, 'vbm': vbm, 'cbm': cbm, 'gap': gap})
        print(f"Strain {eps*100:+.1f}%: VBM={vbm:.4f}, CBM={cbm:.4f}, gap={gap:.4f} eV")
    except Exception as e:
        print(f"ERROR parsing {sd}: {e}")

# Fit
strains = np.array([r['strain'] for r in results])
vbms = np.array([r['vbm'] for r in results])
cbms = np.array([r['cbm'] for r in results])

c_vbm = np.polyfit(strains, vbms, 1)
c_cbm = np.polyfit(strains, cbms, 1)

print(f"\nD_ac(VBM) = {c_vbm[0]:.3f} eV")
print(f"D_ac(CBM) = {c_cbm[0]:.3f} eV")
print(f"D_ac(gap) = {c_cbm[0] - c_vbm[0]:.3f} eV")
```

### Method D: Frohlich Coupling Constant (Polar Optical Phonons)

For polar semiconductors, the dominant carrier-optical-phonon coupling is the Frohlich interaction, parameterized by the Frohlich coupling constant alpha_F:

```python
#!/usr/bin/env python3
"""
frohlich_coupling.py
Compute Frohlich electron-phonon coupling constant for polar semiconductors.

alpha_F = (e^2 / hbar) * sqrt(m* / (2*hbar*omega_LO))
          * (1/eps_inf - 1/eps_0)

Requires:
  - Effective mass m* (from band curvature or BoltzTraP2)
  - LO phonon frequency omega_LO (from ph.x)
  - Dielectric constants eps_inf (electronic) and eps_0 (static)
    (from ph.x or epsilon.x)
"""
import numpy as np

# Physical constants
e_C = 1.602176634e-19       # Coulomb
hbar_Js = 1.054571817e-34   # J*s
m_e_kg = 9.1093837015e-31   # kg
eps_0_SI = 8.8541878128e-12  # F/m
eV_to_J = 1.602176634e-19

def frohlich_alpha(m_star_me, omega_LO_meV, eps_inf, eps_0):
    """
    Compute Frohlich coupling constant alpha_F.

    Parameters
    ----------
    m_star_me : float
        Effective mass in units of electron mass.
    omega_LO_meV : float
        LO phonon frequency in meV.
    eps_inf : float
        High-frequency (electronic/optical) dielectric constant.
    eps_0 : float
        Static dielectric constant.

    Returns
    -------
    alpha_F : float
        Dimensionless Frohlich coupling constant.
    """
    m_star = m_star_me * m_e_kg
    omega_LO = omega_LO_meV * 1e-3 * eV_to_J / hbar_Js  # rad/s

    # alpha_F = (e^2 / (4*pi*eps_0*hbar)) * sqrt(m / (2*hbar*omega_LO))
    #           * (1/eps_inf - 1/eps_0)
    prefactor = e_C**2 / (4 * np.pi * eps_0_SI * hbar_Js)
    mass_factor = np.sqrt(m_star / (2.0 * hbar_Js * omega_LO))
    dielectric_factor = (1.0 / eps_inf) - (1.0 / eps_0)

    alpha_F = prefactor * mass_factor * dielectric_factor
    return alpha_F


def polaron_energy(alpha_F, omega_LO_meV):
    """
    Compute polaron binding energy (Feynman approximation).
    E_polaron = -alpha_F * hbar * omega_LO  (weak coupling)
    """
    return -alpha_F * omega_LO_meV  # meV


def polaron_mass_enhancement(alpha_F):
    """
    Polaron mass enhancement (weak coupling).
    m_polaron / m_band = 1 + alpha_F/6
    """
    return 1.0 + alpha_F / 6.0


# === Example: GaAs ===
print("Frohlich Coupling Constant Calculation")
print("=" * 50)

# GaAs parameters
m_star = 0.067       # electron effective mass in m_e
omega_LO = 36.2      # meV (LO phonon at Gamma)
eps_inf = 10.89       # high-frequency dielectric constant
eps_0 = 12.90         # static dielectric constant

alpha = frohlich_alpha(m_star, omega_LO, eps_inf, eps_0)
E_pol = polaron_energy(alpha, omega_LO)
m_enh = polaron_mass_enhancement(alpha)

print(f"\nMaterial: GaAs")
print(f"  m*/m_e          = {m_star}")
print(f"  omega_LO        = {omega_LO} meV")
print(f"  eps_inf          = {eps_inf}")
print(f"  eps_0            = {eps_0}")
print(f"\nResults:")
print(f"  alpha_F          = {alpha:.4f}")
print(f"  Polaron energy   = {E_pol:.2f} meV")
print(f"  Mass enhancement = {m_enh:.4f}")
print(f"  m_polaron/m_band = {m_enh:.4f}")

# Coupling regime
if alpha < 1:
    print(f"\n  Regime: WEAK coupling (alpha < 1)")
elif alpha < 6:
    print(f"\n  Regime: INTERMEDIATE coupling (1 < alpha < 6)")
else:
    print(f"\n  Regime: STRONG coupling (alpha > 6)")
```

To obtain the required parameters (eps_inf, eps_0, omega_LO) from QE:

```
cat > epsilon.in << 'PH_INPUT'
Dielectric properties
&INPUTPH
    prefix   = 'gaas'
    outdir   = './tmp/'
    tr2_ph   = 1.0d-14
    epsil    = .true.
    fildyn   = 'gaas.dyn'
/
PH_INPUT

mpirun -np 4 ph.x < epsilon.in > epsilon.out 2>&1
grep "Dielectric constant" epsilon.out
grep "omega(" epsilon.out
```

## Key Parameters

| Parameter | Typical Value | Notes |
|---|---|---|
| Strain range | -2% to +2% | Must stay in linear regime. Extend to +/-3% if R^2 drops. |
| Strain increment | 0.5% (coarse), 0.25% (fine) | 9-17 points total. More points improve fit quality. |
| ecutwfc | 40-80 Ry | Must be converged for band gap. |
| k-grid for SCF | 12x12x12 | Standard convergence. |
| k-grid for NSCF (band edges) | 20x20x20 | Denser grid for accurate VBM/CBM. |
| conv_thr | 1.0d-10 | Tight convergence for meV-level band edge shifts. |
| Frozen phonon displacement | 0.02-0.06 Angstrom | Must be small enough for harmonic regime. |
| Pseudopotential consistency | Same PP for all strain points | Changing PP between points invalidates comparison. |

### Convergence Checklist

1. **ecutwfc**: Band gap should be converged to <10 meV. Test at 40, 60, 80 Ry.
2. **k-grid**: Band edge energies should be converged to <5 meV. Test 8x8x8, 12x12x12, 16x16x16.
3. **Strain range**: Fit should be linear (R^2 > 0.999). If R^2 drops, reduce strain range.
4. **Number of strain points**: At least 5 points (preferably 9+). More points improve fit robustness.
5. **Band edge identification**: Verify that the same k-point gives VBM/CBM across all strains. If the band extremum moves in k-space, use denser k-grid or track the specific k-point.

## Interpreting Results

### Acoustic Deformation Potentials

| Material | D_ac(VBM) (eV) | D_ac(CBM) (eV) | Notes |
|---|---|---|---|
| Si | ~1.8 (hydrostatic) | ~9.0 (hydrostatic) | CBM at Delta minimum, indirect gap |
| GaAs | ~-8.0 | ~-7.5 | Direct gap at Gamma |
| GaN | ~-6.0 | ~-9.0 | Wurtzite, direct gap |
| Diamond | ~-1.0 | ~-8.5 | Indirect gap |

- **Sign convention**: Positive D means the band edge shifts up (higher energy) under tensile strain.
- **Magnitude**: Larger |D_ac| means stronger electron-acoustic-phonon scattering and lower carrier mobility.
- **Hydrostatic vs. uniaxial**: Hydrostatic deformation potential is isotropic. Uniaxial gives direction-dependent coupling and is relevant for shear-mode scattering.
- **For mobility**: Acoustic-phonon-limited mobility goes as mu ~ 1/D_ac^2 * T^(-3/2) (non-degenerate semiconductors).

### Optical Deformation Potentials

- **Units**: eV/Angstrom (energy shift per displacement amplitude) or eV/cm (after appropriate conversion).
- **Comparison with Frohlich**: In non-polar materials (Si, Ge), optical deformation potential scattering dominates. In polar materials (GaAs, GaN), Frohlich coupling typically dominates for long-wavelength optical phonons.
- **Intervalley scattering**: For indirect-gap semiconductors, optical phonons at zone-boundary q-points scatter carriers between valleys. This requires zone-boundary optical deformation potentials, which need supercell or DFPT calculations.

### Frohlich Coupling Constant

| Material | alpha_F | Regime |
|---|---|---|
| InSb | 0.02 | Weak |
| GaAs | 0.07 | Weak |
| CdTe | 0.35 | Weak |
| GaN | 0.48 | Weak-intermediate |
| ZnO | 0.85 | Intermediate |
| SrTiO3 | 3.8 | Intermediate-strong |
| KCl | 5.6 | Strong |

- alpha_F < 1: Weak coupling, perturbation theory valid.
- 1 < alpha_F < 6: Intermediate coupling, Feynman path integral approach needed.
- alpha_F > 6: Strong coupling, small polaron formation.

## Common Issues

| Problem | Symptom | Solution |
|---|---|---|
| Non-linear E(strain) | R^2 < 0.99, or visible curvature in plot | Reduce strain range to +/-1%. Or add quadratic term to fit (but report linear coefficient). |
| Band crossing under strain | VBM or CBM character changes between strain points | Use orbital-projected band structure to track specific band. May need to identify band by symmetry label. |
| VBM/CBM at different k-points under strain | Apparent discontinuity in E(strain) | Use denser k-grid (NSCF with 20x20x20+). The band extremum may move in k-space. Track the envelope (maximum over all k-points). |
| Absolute vs. relative band edge | Computed energies are relative to internal reference | For absolute deformation potentials, align to vacuum level (slab calculation) or average electrostatic potential. For relative (gap deformation potential), alignment cancels. |
| Spin-orbit coupling effects | Band edges split under strain in heavy-element systems | Include SOC (`lspinorb = .true.`, `noncolin = .true.` in QE) for systems with heavy atoms (Bi, Pb, Sb, Te). |
| Frozen phonon amplitude too large | Non-linear response, overestimated D_opt | Reduce displacement to 0.01-0.02 Angstrom. Check that E(u) is symmetric and linear. |
| Wrong phonon eigenvector | D_opt value disagrees with literature | Compute eigenvectors from `ph.x` + `dynmat.x` rather than guessing. The normalized eigenvector matters for the absolute value of D_opt. |
| Pseudopotential inconsistency | Unexpected band edge shifts | Use the exact same pseudopotentials for all strain points. Ghost states can appear at certain strains with some PPs. |
