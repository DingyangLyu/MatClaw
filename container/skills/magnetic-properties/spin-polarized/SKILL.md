# Spin-Polarized DFT Calculations

## When to Use

- Determine whether a material is magnetic (nonzero net magnetization).
- Compute magnetic moments per atom and total magnetic moment.
- Obtain spin-resolved density of states (majority/minority spin channels).
- Visualize spin density (difference between spin-up and spin-down charge densities).
- Prerequisite step before magnetic ordering comparisons or magnetic anisotropy calculations.

## Prerequisites

- Quantum ESPRESSO 7.5 with `pw.x`, `pp.x`, `dos.x`, `projwfc.x`
- Scalar-relativistic pseudopotentials (e.g., SSSP Efficiency or PseudoDojo)
- Python: `pymatgen`, `ase`, `numpy`, `matplotlib`
- Basic knowledge of the target material (lattice parameters, atomic species, expected magnetic behavior)

## Detailed Steps

### Step 1: Generate QE Input for a Spin-Polarized SCF Calculation

The example below uses BCC Fe (a classic ferromagnet). The key parameters for spin-polarized
calculations are `nspin=2` and `starting_magnetization` for each atomic species.

#### Complete QE Input File: `fe_scf.in`

```
&CONTROL
    calculation   = 'scf'
    prefix        = 'fe'
    outdir        = './tmp'
    pseudo_dir    = './pseudo'
    tprnfor       = .true.
    tstress       = .true.
    verbosity     = 'high'
/
&SYSTEM
    ibrav         = 3
    celldm(1)     = 5.42
    nat           = 1
    ntyp          = 1
    ecutwfc       = 60.0
    ecutrho       = 480.0
    nspin         = 2
    starting_magnetization(1) = 0.6
    occupations   = 'smearing'
    smearing      = 'mv'
    degauss       = 0.02
/
&ELECTRONS
    conv_thr      = 1.0d-8
    mixing_beta   = 0.3
    mixing_mode   = 'plain'
    electron_maxstep = 200
/
ATOMIC_SPECIES
    Fe  55.845  Fe.pbe-spn-kjpaw_psl.0.2.1.UPF
ATOMIC_POSITIONS crystal
    Fe  0.0  0.0  0.0
K_POINTS automatic
    12 12 12  0 0 0
```

**Key parameters explained:**

- `nspin = 2`: Enables collinear spin-polarized (LSDA) calculation. Spin-up and spin-down
  channels are treated separately.
- `starting_magnetization(1) = 0.6`: Initial guess for the magnetization of species 1 (Fe).
  Range is -1.0 (fully spin-down) to +1.0 (fully spin-up). A value of 0.6 is a reasonable
  starting guess for Fe. The SCF will converge to the self-consistent value.
- `occupations = 'smearing'` and `smearing = 'mv'`: Marzari-Vanderbilt cold smearing is
  recommended for metals. Gaussian smearing also works.
- `degauss = 0.02`: Smearing width in Ry. For metals, values of 0.01-0.03 Ry are typical.
- `ecutwfc = 60.0`: Plane-wave cutoff. Must be converged for your pseudopotential.
- `mixing_beta = 0.3`: Reduced mixing for magnetic systems (default 0.7 can cause
  oscillations). Values of 0.2-0.4 are safer for magnetic metals.

#### LSDA vs GGA+Spin

- **LSDA** (`nspin=2` with `input_dft='PZ'` or LDA pseudopotentials): The local spin density
  approximation. Older but still useful for some systems.
- **GGA+Spin** (`nspin=2` with PBE pseudopotentials, the default shown above): The more
  common modern approach. PBE-GGA with spin polarization gives good magnetic moments for
  most 3d transition metals.
- For strongly correlated systems (e.g., NiO, MnO), consider adding Hubbard U corrections
  (`lda_plus_u = .true.`, `Hubbard_U(1) = 5.0`).

### Step 2: Run the SCF Calculation

```bash
pw.x -npool 1 < fe_scf.in > fe_scf.out
```

### Step 3: Python Script to Generate QE Input and Parse Magnetic Moments

```python
#!/usr/bin/env python3
"""
generate_and_run_spinpol.py

Generate a spin-polarized QE input for BCC Fe, run the SCF calculation,
and parse the resulting magnetic moments from the output.
"""

import os
import re
import subprocess
import numpy as np
from pymatgen.core import Structure, Lattice

# ============================================================
# 1. Build structure with pymatgen
# ============================================================
a = 2.87  # BCC Fe lattice constant in Angstrom
lattice = Lattice.cubic(a)
structure = Structure(lattice, ["Fe", "Fe"], [[0.0, 0.0, 0.0], [0.5, 0.5, 0.5]])
print(f"Structure: {structure}")
print(f"Number of atoms: {len(structure)}")

# ============================================================
# 2. Write QE input file
# ============================================================
PSEUDO_DIR = "./pseudo"
OUTDIR = "./tmp"
PREFIX = "fe_bcc"

os.makedirs(PSEUDO_DIR, exist_ok=True)
os.makedirs(OUTDIR, exist_ok=True)

# Map species to pseudopotential files and starting magnetization
pseudo_map = {
    "Fe": "Fe.pbe-spn-kjpaw_psl.0.2.1.UPF",
}
mag_map = {
    "Fe": 0.6,  # starting magnetization for Fe (ferromagnetic guess)
}

def write_qe_scf_input(structure, filename, ecutwfc=60.0, ecutrho=480.0,
                        kgrid=(12, 12, 12), degauss=0.02):
    """Write a spin-polarized QE SCF input file."""
    species = list(set([str(s) for s in structure.species]))
    species.sort()

    # Build ntyp-indexed starting_magnetization lines
    mag_lines = ""
    for i, sp in enumerate(species, start=1):
        mag_lines += f"    starting_magnetization({i}) = {mag_map[sp]}\n"

    # Cell parameters in Angstrom
    cell = structure.lattice.matrix

    input_text = f"""&CONTROL
    calculation   = 'scf'
    prefix        = '{PREFIX}'
    outdir        = '{OUTDIR}'
    pseudo_dir    = '{PSEUDO_DIR}'
    tprnfor       = .true.
    tstress       = .true.
    verbosity     = 'high'
/
&SYSTEM
    ibrav         = 0
    nat           = {len(structure)}
    ntyp          = {len(species)}
    ecutwfc       = {ecutwfc}
    ecutrho       = {ecutrho}
    nspin         = 2
{mag_lines.rstrip()}
    occupations   = 'smearing'
    smearing      = 'mv'
    degauss       = {degauss}
/
&ELECTRONS
    conv_thr      = 1.0d-8
    mixing_beta   = 0.3
    electron_maxstep = 200
/
ATOMIC_SPECIES
"""
    for sp in species:
        from pymatgen.core import Element
        mass = Element(sp).atomic_mass
        input_text += f"    {sp}  {mass:.3f}  {pseudo_map[sp]}\n"

    input_text += f"""
CELL_PARAMETERS angstrom
    {cell[0][0]:.10f}  {cell[0][1]:.10f}  {cell[0][2]:.10f}
    {cell[1][0]:.10f}  {cell[1][1]:.10f}  {cell[1][2]:.10f}
    {cell[2][0]:.10f}  {cell[2][1]:.10f}  {cell[2][2]:.10f}

ATOMIC_POSITIONS crystal
"""
    for site in structure:
        sp = str(site.specie)
        fc = site.frac_coords
        input_text += f"    {sp}  {fc[0]:.10f}  {fc[1]:.10f}  {fc[2]:.10f}\n"

    input_text += f"""
K_POINTS automatic
    {kgrid[0]} {kgrid[1]} {kgrid[2]}  0 0 0
"""

    with open(filename, "w") as f:
        f.write(input_text)
    print(f"Written QE input: {filename}")

write_qe_scf_input(structure, "fe_scf.in")

# ============================================================
# 3. Run QE (uncomment to execute)
# ============================================================
# subprocess.run(["pw.x", "-npool", "1"], stdin=open("fe_scf.in"),
#                stdout=open("fe_scf.out", "w"), stderr=subprocess.STDOUT)

# ============================================================
# 4. Parse magnetic moments from QE output
# ============================================================
def parse_magnetic_moments(filename):
    """
    Parse per-atom and total magnetic moments from pw.x output.

    Returns:
        atom_moments: list of (atom_index, species, moment_uB)
        total_moment: float (total magnetization in Bohr magnetons)
        absolute_moment: float (absolute magnetization in Bohr magnetons)
    """
    atom_moments = []
    total_moment = None
    absolute_moment = None

    with open(filename, "r") as f:
        lines = f.readlines()

    # Parse total and absolute magnetization
    for line in lines:
        if "total magnetization" in line and "Bohr mag/cell" in line:
            total_moment = float(line.split("=")[1].split("Bohr")[0].strip())
        if "absolute magnetization" in line and "Bohr mag/cell" in line:
            absolute_moment = float(line.split("=")[1].split("Bohr")[0].strip())

    # Parse per-atom Lowdin charges (which include magnetic moments)
    # Look for the Lowdin section in verbose output
    in_lowdin = False
    for i, line in enumerate(lines):
        if "Magnetic moment per site" in line:
            # QE 7.x prints per-site magnetic moments directly
            # Format: atom:    1    charge:   14.0000    magn:    2.2000    ...
            for j in range(i + 1, len(lines)):
                mline = lines[j].strip()
                if mline.startswith("atom:"):
                    parts = mline.split()
                    atom_idx = int(parts[1])
                    magn_idx = parts.index("magn:") + 1
                    magn = float(parts[magn_idx])
                    atom_moments.append((atom_idx, magn))
                elif mline == "" or not mline.startswith("atom:"):
                    break

    return atom_moments, total_moment, absolute_moment


def parse_magnetic_moments_robust(filename):
    """
    Robust parser that handles multiple QE output formats.
    Falls back to regex-based parsing if structured parsing fails.
    """
    atom_moments = []
    total_moment = None
    absolute_moment = None

    with open(filename, "r") as f:
        content = f.read()

    # Total magnetization
    m = re.search(r"total magnetization\s*=\s*([\d\.\-]+)\s*Bohr", content)
    if m:
        total_moment = float(m.group(1))

    # Absolute magnetization
    m = re.search(r"absolute magnetization\s*=\s*([\d\.\-]+)\s*Bohr", content)
    if m:
        absolute_moment = float(m.group(1))

    # Per-atom magnetic moments (QE 7.x format)
    pattern = r"atom:\s+(\d+)\s+charge:\s+[\d\.]+\s+magn:\s+([\d\.\-]+)"
    for m in re.finditer(pattern, content):
        atom_idx = int(m.group(1))
        magn = float(m.group(2))
        atom_moments.append((atom_idx, magn))

    return atom_moments, total_moment, absolute_moment


# Example usage (with a pre-existing output file):
output_file = "fe_scf.out"
if os.path.exists(output_file):
    atom_moments, total_mag, abs_mag = parse_magnetic_moments_robust(output_file)
    print(f"\nTotal magnetization:    {total_mag:.4f} Bohr mag/cell")
    print(f"Absolute magnetization: {abs_mag:.4f} Bohr mag/cell")
    print("\nPer-atom magnetic moments:")
    for idx, magn in atom_moments:
        print(f"  Atom {idx}: {magn:.4f} uB")
else:
    print(f"\nOutput file '{output_file}' not found. Run pw.x first.")
```

### Step 4: Spin-Polarized DOS Calculation

After the SCF converges, perform an NSCF calculation on a denser k-grid, then run `dos.x`.

#### NSCF Input: `fe_nscf.in`

```
&CONTROL
    calculation   = 'nscf'
    prefix        = 'fe'
    outdir        = './tmp'
    pseudo_dir    = './pseudo'
    verbosity     = 'high'
/
&SYSTEM
    ibrav         = 3
    celldm(1)     = 5.42
    nat           = 1
    ntyp          = 1
    ecutwfc       = 60.0
    ecutrho       = 480.0
    nspin         = 2
    starting_magnetization(1) = 0.6
    occupations   = 'tetrahedra_opt'
    nbnd          = 20
/
&ELECTRONS
    conv_thr      = 1.0d-8
/
ATOMIC_SPECIES
    Fe  55.845  Fe.pbe-spn-kjpaw_psl.0.2.1.UPF
ATOMIC_POSITIONS crystal
    Fe  0.0  0.0  0.0
K_POINTS automatic
    24 24 24  0 0 0
```

**Note:** For the DOS (NSCF) step, use `occupations = 'tetrahedra_opt'` (the optimized
tetrahedron method) for accurate DOS integration on a uniform k-grid. The denser 24x24x24
k-grid gives smoother DOS curves.

#### DOS Input: `fe_dos.in`

```
&DOS
    prefix  = 'fe'
    outdir  = './tmp'
    fildos  = 'fe_dos.dat'
    Emin    = -15.0
    Emax    =  20.0
    DeltaE  = 0.05
/
```

#### Run Commands

```bash
pw.x  < fe_nscf.in > fe_nscf.out
dos.x < fe_dos.in  > fe_dos.out
```

### Step 5: Python Post-Processing -- Plot Spin-Polarized DOS

```python
#!/usr/bin/env python3
"""
plot_spin_dos.py

Parse and plot spin-polarized DOS from QE dos.x output.
Majority spin (up) is plotted upward, minority spin (down) is plotted downward.
"""

import numpy as np
import matplotlib.pyplot as plt
import re
import os


def parse_qe_dos(filename):
    """
    Parse QE dos.x output file.

    For nspin=2, the file has columns:
        E(eV)  dos_up(E)  dos_down(E)  idos_up(E)  idos_down(E)

    For nspin=1:
        E(eV)  dos(E)  idos(E)

    Returns:
        energy: np.array of energies (eV, relative to Fermi level)
        dos_up: np.array of majority-spin DOS (states/eV)
        dos_down: np.array of minority-spin DOS (states/eV) or None
        e_fermi: float, Fermi energy (eV)
    """
    # Read the Fermi energy from the header
    e_fermi = 0.0
    header_lines = 0
    with open(filename, "r") as f:
        for line in f:
            if line.strip().startswith("#"):
                header_lines += 1
                m = re.search(r"EFermi\s*=\s*([\d\.\-]+)\s*eV", line)
                if m:
                    e_fermi = float(m.group(1))
            else:
                break

    data = np.loadtxt(filename, comments="#")

    if data.shape[1] >= 5:
        # Spin-polarized: E, dos_up, dos_down, idos_up, idos_down
        energy = data[:, 0] - e_fermi
        dos_up = data[:, 1]
        dos_down = data[:, 2]
    elif data.shape[1] >= 3:
        # Non-spin-polarized: E, dos, idos
        energy = data[:, 0] - e_fermi
        dos_up = data[:, 1]
        dos_down = None
    else:
        raise ValueError(f"Unexpected number of columns: {data.shape[1]}")

    return energy, dos_up, dos_down, e_fermi


def plot_spin_dos(energy, dos_up, dos_down, e_fermi, output_file="spin_dos.png",
                  emin=-10, emax=5, title="Spin-Polarized DOS"):
    """
    Plot spin-polarized DOS with majority spin up and minority spin down.
    """
    fig, ax = plt.subplots(figsize=(8, 5))

    # Mask to energy window
    mask = (energy >= emin) & (energy <= emax)

    # Plot majority spin (upward, positive)
    ax.fill_between(energy[mask], 0, dos_up[mask],
                    color="steelblue", alpha=0.5, label="Spin Up")
    ax.plot(energy[mask], dos_up[mask], color="steelblue", linewidth=0.8)

    if dos_down is not None:
        # Plot minority spin (downward, negative)
        ax.fill_between(energy[mask], 0, -dos_down[mask],
                        color="tomato", alpha=0.5, label="Spin Down")
        ax.plot(energy[mask], -dos_down[mask], color="tomato", linewidth=0.8)

    # Fermi level
    ax.axvline(x=0, color="black", linestyle="--", linewidth=0.8, label="$E_F$")
    ax.axhline(y=0, color="gray", linestyle="-", linewidth=0.5)

    ax.set_xlabel("$E - E_F$ (eV)", fontsize=13)
    ax.set_ylabel("DOS (states/eV)", fontsize=13)
    ax.set_title(title, fontsize=14)
    ax.legend(fontsize=11, loc="upper right")
    ax.set_xlim(emin, emax)

    plt.tight_layout()
    plt.savefig(output_file, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"Saved spin-polarized DOS plot to: {output_file}")


# ============================================================
# Main
# ============================================================
dos_file = "fe_dos.dat"
if os.path.exists(dos_file):
    energy, dos_up, dos_down, e_fermi = parse_qe_dos(dos_file)
    print(f"Fermi energy: {e_fermi:.4f} eV")
    print(f"Energy range: {energy.min():.2f} to {energy.max():.2f} eV")
    plot_spin_dos(energy, dos_up, dos_down, e_fermi,
                  output_file="fe_spin_dos.png", emin=-10, emax=5,
                  title="BCC Fe Spin-Polarized DOS")
else:
    print(f"DOS file '{dos_file}' not found. Run dos.x first.")
    print("Generating example plot with synthetic data for demonstration...")

    # Synthetic demonstration data
    energy = np.linspace(-10, 5, 500)
    # Fake Fe-like DOS: broader d-band with exchange splitting
    dos_up = 2.0 * np.exp(-0.5 * ((energy + 1.5) / 1.5) ** 2) + \
             0.5 * np.exp(-0.5 * ((energy + 4.0) / 2.0) ** 2)
    dos_down = 1.5 * np.exp(-0.5 * ((energy - 0.5) / 1.5) ** 2) + \
               0.5 * np.exp(-0.5 * ((energy + 3.0) / 2.0) ** 2)
    plot_spin_dos(energy, dos_up, dos_down, e_fermi=0.0,
                  output_file="fe_spin_dos_demo.png", emin=-10, emax=5,
                  title="BCC Fe Spin-Polarized DOS (demo)")
```

### Step 6: Spin Density Visualization with pp.x

The spin density is the difference between spin-up and spin-down charge densities:
`rho_spin(r) = rho_up(r) - rho_down(r)`.

In QE's `pp.x`, this is `plot_num = 6`.

#### pp.x Input: `fe_spin_density.in`

```
&INPUTPP
    prefix  = 'fe'
    outdir  = './tmp'
    filplot = 'fe_spin_density.dat'
    plot_num = 6
/
&PLOT
    nfile         = 1
    filepp(1)     = 'fe_spin_density.dat'
    weight(1)     = 1.0
    iflag         = 3
    output_format = 6
    fileout       = 'fe_spin_density.cube'
/
```

**Parameters:**
- `plot_num = 6`: Spin polarization (rho_up - rho_down).
- `output_format = 6`: Gaussian cube file format (viewable in VESTA, XCrySDen, etc.).
- `iflag = 3`: 3D plot.

```bash
pp.x < fe_spin_density.in > fe_spin_density.out
```

### Step 7: NiO Example (Antiferromagnetic Insulator with Hubbard U)

NiO is a classic antiferromagnetic insulator. A spin-polarized calculation with Hubbard U
correction is needed to open the gap.

#### QE Input: `nio_scf.in`

```
&CONTROL
    calculation   = 'scf'
    prefix        = 'nio'
    outdir        = './tmp'
    pseudo_dir    = './pseudo'
    verbosity     = 'high'
/
&SYSTEM
    ibrav         = 0
    nat           = 4
    ntyp          = 3
    ecutwfc       = 60.0
    ecutrho       = 480.0
    nspin         = 2
    starting_magnetization(1) =  0.8
    starting_magnetization(2) = -0.8
    starting_magnetization(3) =  0.0
    occupations   = 'smearing'
    smearing      = 'gauss'
    degauss       = 0.005
    lda_plus_u    = .true.
    Hubbard_U(1)  = 6.5
    Hubbard_U(2)  = 6.5
/
&ELECTRONS
    conv_thr        = 1.0d-8
    mixing_beta     = 0.2
    electron_maxstep = 300
/
ATOMIC_SPECIES
    Ni1  58.693  Ni.pbe-spn-kjpaw_psl.1.0.0.UPF
    Ni2  58.693  Ni.pbe-spn-kjpaw_psl.1.0.0.UPF
    O    15.999  O.pbe-n-kjpaw_psl.1.0.0.UPF

CELL_PARAMETERS angstrom
    2.954  2.954  0.000
    2.954  0.000  2.954
    0.000  2.954  2.954

ATOMIC_POSITIONS crystal
    Ni1  0.0  0.0  0.0
    Ni2  0.5  0.5  0.5
    O    0.25 0.25 0.25
    O    0.75 0.75 0.75

K_POINTS automatic
    6 6 6  0 0 0
```

**Note:** To set up antiferromagnetic NiO, we define two distinct Ni types (`Ni1`, `Ni2`)
with opposite `starting_magnetization`. Both use the same pseudopotential file but are
treated as different species by QE. This forces the AFM-II ordering (alternating (111) planes).

## Key Parameters

| Parameter | Typical Values | Notes |
|---|---|---|
| `nspin` | `2` | Enables collinear spin polarization. |
| `starting_magnetization` | -1.0 to 1.0 | Initial guess per species. Positive = spin-up, negative = spin-down. Magnitude: 0.5-0.8 for 3d metals. |
| `ecutwfc` | 40-80 Ry | Must be converged; check pseudopotential recommendations. |
| `degauss` | 0.01-0.03 Ry | Smearing width. Smaller for semiconductors, larger for metals. |
| `smearing` | `'mv'` or `'gauss'` | Marzari-Vanderbilt for metals; Gaussian also works. |
| `mixing_beta` | 0.2-0.4 | Reduced mixing prevents oscillations in magnetic systems. |
| `conv_thr` | 1.0d-8 | Standard for total energy. Use 1.0d-10 for forces or phonons. |
| `lda_plus_u` | `.true.` | Enable Hubbard U correction (for strongly correlated systems). |
| `Hubbard_U(i)` | 3-8 eV | Species-dependent. Common values: NiO ~6.5 eV, FeO ~4.3 eV, MnO ~4.0 eV. |

## Interpreting Results

### Magnetic Moments

- **BCC Fe**: Expected magnetic moment ~ 2.2 uB/atom. QE (PBE) typically gives 2.2-2.3 uB.
- **NiO**: Expected Ni moment ~ 1.7-1.9 uB (with Hubbard U). Without U, the moment is too
  small and the band gap is underestimated.
- The **total magnetization** (in Bohr magnetons per cell) should match expectations:
  - FM Fe (1 atom/cell): ~2.2 uB
  - AFM NiO: ~0.0 uB (opposite Ni moments cancel)
- The **absolute magnetization** is the integral of |m(r)| and is always positive. It
  quantifies total spin polarization regardless of ordering.

### Spin-Polarized DOS

- For a ferromagnet (Fe), the majority and minority spin channels are shifted relative to
  each other (exchange splitting). The Fermi level cuts through both channels.
- For an antiferromagnet (NiO with U), the spin-up and spin-down DOS should be mirror images
  if plotted per atom type. The total DOS may look symmetric.
- A large exchange splitting (difference in peak positions between spin channels) indicates
  strong magnetism.

### Spin Density

- Positive (red) regions in the spin density plot indicate spin-up excess.
- Negative (blue) regions indicate spin-down excess.
- For BCC Fe, the spin density is concentrated around Fe atoms with roughly spherical shape
  (d-electron contribution).

## Common Issues

1. **SCF does not converge**: Reduce `mixing_beta` to 0.1-0.2. Try `mixing_mode = 'local-TF'`.
   Increase `electron_maxstep` to 300+.

2. **Wrong magnetic state**: The SCF may converge to a non-magnetic or wrong magnetic state if
   `starting_magnetization` is too small. Use values of 0.5-0.8 for 3d transition metals.
   For AFM, make sure to use opposite signs for different sublattices.

3. **Metallic NiO**: NiO without Hubbard U gives a metallic (or small-gap) ground state with
   PBE. Add `lda_plus_u = .true.` and appropriate `Hubbard_U` values.

4. **Smearing artifacts**: If the DOS shows unphysical features near E_F, reduce `degauss`.
   For the NSCF/DOS step, use `occupations = 'tetrahedra_opt'` (no smearing needed).

5. **k-point convergence**: Magnetic energies can be sensitive to k-grid density. Converge
   the total magnetization with respect to k-points. For BCC Fe, 12x12x12 is usually
   sufficient for SCF; use 24x24x24+ for DOS.

6. **Pseudopotential mismatch**: Ensure your pseudopotential supports spin polarization
   (scalar-relativistic PAW or ultrasoft). Check that `ecutwfc` matches the pseudopotential
   recommendation.

7. **Multiple Ni types for AFM**: When defining `Ni1` and `Ni2` as separate species, both
   must point to the same pseudopotential file. QE treats them as distinct types only for
   the starting magnetization. The Hubbard U must be specified for each type index separately.
