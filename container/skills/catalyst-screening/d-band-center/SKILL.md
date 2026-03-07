# d-Band Center Analysis for Catalyst Design

## When to Use

- Screening transition metal catalysts for adsorption activity
- Predicting trends in adsorption energies across transition metals
- Understanding electronic structure origins of catalytic activity
- Ranking candidate catalyst surfaces without computing full adsorption energies
- Comparing alloy surfaces, strained surfaces, or supported catalysts to pure metals

## Prerequisites

- Quantum ESPRESSO 7.5 (`pw.x`, `projwfc.x`)
- Python packages: `ase`, `pymatgen`, `numpy`, `scipy`, `matplotlib`
- MACE-torch (for structure optimization / quick screening)
- Pseudopotentials for transition metals (PBE, SSSP or PSlibrary)

## Theory

The **Norskov d-band model** establishes that the adsorption energy of simple adsorbates
on transition metal surfaces correlates with the d-band center (first moment of the
d-projected density of states relative to the Fermi level):

```
epsilon_d = integral[ E * rho_d(E) dE ] / integral[ rho_d(E) dE ]
```

where rho_d(E) is the d-orbital projected DOS of the surface atoms.

**Key relationships:**
- Higher (less negative) epsilon_d -> stronger adsorption (more reactive)
- Lower (more negative) epsilon_d -> weaker adsorption (more noble)
- Ordering: Pt > Pd > Ni > Cu > Ag > Au (for d-band center position)

**Additional descriptors:**
- d-band width: W_d = sqrt( integral[ (E - epsilon_d)^2 * rho_d(E) dE ] / integral[ rho_d(E) dE ] )
- d-band filling: f_d = integral_{-inf}^{E_F} rho_d(E) dE / integral rho_d(E) dE
- Upper d-band edge: epsilon_d + W_d (sometimes better descriptor)

## Detailed Steps

### Step 1: Build Transition Metal Surface Slabs

```python
#!/usr/bin/env python3
"""
Generate surface slabs for multiple transition metals using ASE.
Computes d-band center from QE PDOS for each.
"""
import os
import numpy as np
from ase.build import fcc111, bcc110, hcp0001, bulk
from ase.io import write
from ase.constraints import FixAtoms

WORK_DIR = os.path.abspath("dband_screening")
os.makedirs(WORK_DIR, exist_ok=True)

# ------------------------------------------------------------------
# Metal database: structure type, lattice constant (Angstrom), PP name
# ------------------------------------------------------------------
METALS = {
    "Cu": {"structure": "fcc", "a": 3.615, "pp": "Cu.pbe-dn-kjpaw_psl.1.0.0.UPF",
            "mass": 63.546, "valence_e": 11, "d_electrons": 10},
    "Ag": {"structure": "fcc", "a": 4.086, "pp": "Ag.pbe-dn-kjpaw_psl.1.0.0.UPF",
            "mass": 107.868, "valence_e": 11, "d_electrons": 10},
    "Au": {"structure": "fcc", "a": 4.078, "pp": "Au.pbe-dn-kjpaw_psl.1.0.0.UPF",
            "mass": 196.967, "valence_e": 11, "d_electrons": 10},
    "Pd": {"structure": "fcc", "a": 3.890, "pp": "Pd.pbe-dn-kjpaw_psl.1.0.0.UPF",
            "mass": 106.42, "valence_e": 10, "d_electrons": 9},
    "Pt": {"structure": "fcc", "a": 3.924, "pp": "Pt.pbe-dn-kjpaw_psl.1.0.0.UPF",
            "mass": 195.084, "valence_e": 10, "d_electrons": 9},
    "Ni": {"structure": "fcc", "a": 3.524, "pp": "Ni.pbe-dn-kjpaw_psl.1.0.0.UPF",
            "mass": 58.693, "valence_e": 10, "d_electrons": 9},
}


def build_slab(metal, nlayers=4, vacuum=15.0):
    """
    Build a (111) surface slab for fcc metals.

    Parameters
    ----------
    metal : str
        Metal symbol
    nlayers : int
        Number of atomic layers
    vacuum : float
        Vacuum thickness in Angstrom

    Returns
    -------
    slab : ASE Atoms object
    """
    info = METALS[metal]
    a = info["a"]

    if info["structure"] == "fcc":
        slab = fcc111(metal, size=(2, 2, nlayers), a=a, vacuum=vacuum,
                      periodic=True)
    elif info["structure"] == "bcc":
        slab = bcc110(metal, size=(2, 2, nlayers), a=a, vacuum=vacuum,
                      periodic=True)
    elif info["structure"] == "hcp":
        slab = hcp0001(metal, size=(2, 2, nlayers), a=a, vacuum=vacuum,
                       periodic=True)
    else:
        raise ValueError(f"Unknown structure: {info['structure']}")

    # Fix bottom two layers
    positions = slab.get_positions()
    z_coords = positions[:, 2]
    z_sorted = np.sort(np.unique(np.round(z_coords, 2)))
    z_fix_threshold = z_sorted[1] + 0.1  # fix bottom 2 layers
    fix_mask = z_coords < z_fix_threshold
    slab.set_constraint(FixAtoms(mask=fix_mask))

    return slab


# Build all slabs
for metal in METALS:
    slab = build_slab(metal)
    slab_dir = os.path.join(WORK_DIR, metal)
    os.makedirs(slab_dir, exist_ok=True)
    write(os.path.join(slab_dir, f"{metal}_slab.xyz"), slab)
    print(f"{metal}: {len(slab)} atoms, cell = {slab.cell.lengths()}")
```

### Step 2: Generate QE Input Files (SCF + NSCF + projwfc)

```python
#!/usr/bin/env python3
"""
Generate QE input files for SCF, NSCF, and projwfc.x for d-band center calculation.
"""
import os
import numpy as np
from ase.io import read

WORK_DIR = os.path.abspath("dband_screening")
PSEUDO_DIR = os.environ.get("PSEUDO_DIR", "./pseudo")

METALS = {
    "Cu": {"pp": "Cu.pbe-dn-kjpaw_psl.1.0.0.UPF", "mass": 63.546,
            "ecutwfc": 50, "ecutrho": 400},
    "Ag": {"pp": "Ag.pbe-dn-kjpaw_psl.1.0.0.UPF", "mass": 107.868,
            "ecutwfc": 50, "ecutrho": 400},
    "Au": {"pp": "Au.pbe-dn-kjpaw_psl.1.0.0.UPF", "mass": 196.967,
            "ecutwfc": 55, "ecutrho": 440},
    "Pd": {"pp": "Pd.pbe-dn-kjpaw_psl.1.0.0.UPF", "mass": 106.42,
            "ecutwfc": 50, "ecutrho": 400},
    "Pt": {"pp": "Pt.pbe-dn-kjpaw_psl.1.0.0.UPF", "mass": 195.084,
            "ecutwfc": 55, "ecutrho": 440},
    "Ni": {"pp": "Ni.pbe-dn-kjpaw_psl.1.0.0.UPF", "mass": 58.693,
            "ecutwfc": 50, "ecutrho": 400},
}


def atoms_to_qe_positions(atoms):
    """Convert ASE atoms to QE ATOMIC_POSITIONS block (angstrom)."""
    lines = "ATOMIC_POSITIONS {angstrom}\n"
    for atom in atoms:
        x, y, z = atom.position
        # Mark fixed atoms with 0 0 0 (if_pos flags)
        if hasattr(atoms, 'constraints') and atoms.constraints:
            from ase.constraints import FixAtoms
            for c in atoms.constraints:
                if isinstance(c, FixAtoms):
                    if atom.index in c.index:
                        lines += f"  {atom.symbol}  {x:16.10f}  {y:16.10f}  {z:16.10f}  0 0 0\n"
                        break
            else:
                lines += f"  {atom.symbol}  {x:16.10f}  {y:16.10f}  {z:16.10f}\n"
        else:
            lines += f"  {atom.symbol}  {x:16.10f}  {y:16.10f}  {z:16.10f}\n"
    return lines


def atoms_to_qe_cell(atoms):
    """Convert ASE cell to QE CELL_PARAMETERS block."""
    lines = "CELL_PARAMETERS {angstrom}\n"
    cell = atoms.get_cell()
    for i in range(3):
        lines += f"  {cell[i,0]:16.10f}  {cell[i,1]:16.10f}  {cell[i,2]:16.10f}\n"
    return lines


def generate_qe_inputs(metal, slab, work_dir):
    """Generate SCF, NSCF, and projwfc inputs for a metal slab."""
    info = METALS[metal]
    metal_dir = os.path.join(work_dir, metal)
    os.makedirs(metal_dir, exist_ok=True)

    nat = len(slab)
    cell_block = atoms_to_qe_cell(slab)
    pos_block = atoms_to_qe_positions(slab)

    # Determine nbnd (at least 20% more than valence electrons / 2)
    nelectrons = nat * info.get("ecutwfc", 50)  # rough estimate
    # Better: use valence electrons from PP
    # For these metals, typically 10-11 valence electrons
    valence = {"Cu": 11, "Ag": 11, "Au": 11, "Pd": 10, "Pt": 10, "Ni": 10}
    n_val_e = nat * valence.get(metal, 10)
    nbnd = int(n_val_e * 0.6) + 20  # some extra empty bands

    # ---- SCF ----
    scf_input = f"""&CONTROL
  calculation = 'scf'
  prefix = '{metal}_slab'
  outdir = './tmp'
  pseudo_dir = '{PSEUDO_DIR}'
  tprnfor = .true.
  tstress = .true.
/
&SYSTEM
  ibrav = 0
  nat = {nat}
  ntyp = 1
  ecutwfc = {info['ecutwfc']:.1f}
  ecutrho = {info['ecutrho']:.1f}
  occupations = 'smearing'
  smearing = 'mv'
  degauss = 0.02
/
&ELECTRONS
  conv_thr = 1.0d-8
  mixing_beta = 0.3
  electron_maxstep = 200
/
ATOMIC_SPECIES
  {metal}  {info['mass']}  {info['pp']}
{cell_block}
{pos_block}
K_POINTS {{automatic}}
  4 4 1 0 0 0
"""

    with open(os.path.join(metal_dir, "scf.in"), "w") as f:
        f.write(scf_input)

    # ---- NSCF (denser k-grid for DOS) ----
    nscf_input = f"""&CONTROL
  calculation = 'nscf'
  prefix = '{metal}_slab'
  outdir = './tmp'
  pseudo_dir = '{PSEUDO_DIR}'
  verbosity = 'high'
/
&SYSTEM
  ibrav = 0
  nat = {nat}
  ntyp = 1
  ecutwfc = {info['ecutwfc']:.1f}
  ecutrho = {info['ecutrho']:.1f}
  occupations = 'tetrahedra'
  nbnd = {nbnd}
/
&ELECTRONS
  conv_thr = 1.0d-8
  diago_full_acc = .true.
/
ATOMIC_SPECIES
  {metal}  {info['mass']}  {info['pp']}
{cell_block}
{pos_block}
K_POINTS {{automatic}}
  8 8 1 0 0 0
"""

    with open(os.path.join(metal_dir, "nscf.in"), "w") as f:
        f.write(nscf_input)

    # ---- projwfc.x (projected DOS) ----
    projwfc_input = f"""&PROJWFC
  outdir = './tmp'
  prefix = '{metal}_slab'
  ngauss = 0
  degauss = 0.02
  Emin = -15.0
  Emax = 5.0
  DeltaE = 0.01
  filpdos = '{metal}_pdos'
/
"""

    with open(os.path.join(metal_dir, "projwfc.in"), "w") as f:
        f.write(projwfc_input)

    print(f"{metal}: SCF, NSCF, projwfc inputs written to {metal_dir}")
    return metal_dir


# Generate inputs for all metals
from ase.io import read as ase_read
from ase.build import fcc111

for metal in METALS:
    slab = fcc111(metal, size=(2, 2, 4),
                  a={"Cu": 3.615, "Ag": 4.086, "Au": 4.078,
                     "Pd": 3.890, "Pt": 3.924, "Ni": 3.524}[metal],
                  vacuum=15.0, periodic=True)
    # Fix bottom 2 layers
    positions = slab.get_positions()
    z_coords = positions[:, 2]
    z_sorted = np.sort(np.unique(np.round(z_coords, 2)))
    z_fix = z_sorted[1] + 0.1
    from ase.constraints import FixAtoms
    slab.set_constraint(FixAtoms(mask=z_coords < z_fix))

    generate_qe_inputs(metal, slab, WORK_DIR)
```

```bash
# Run for each metal (example for Cu)
cd dband_screening/Cu
mpirun -np 4 pw.x -in scf.in > scf.out 2>&1
mpirun -np 4 pw.x -in nscf.in > nscf.out 2>&1
mpirun -np 1 projwfc.x -in projwfc.in > projwfc.out 2>&1
```

### Step 3: Parse PDOS and Compute d-Band Center

```python
#!/usr/bin/env python3
"""
Parse projwfc.x output and compute d-band center, width, and filling.
Complete analysis for multiple transition metals.
"""
import os
import glob
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import trapezoid

WORK_DIR = os.path.abspath("dband_screening")

# d-orbital angular momentum quantum numbers
# l=2 corresponds to d orbitals
# In QE projwfc output, columns are labeled by (atom, n, l, m)
# d orbitals: l=2, m = -2, -1, 0, 1, 2


def parse_pdos_file(filepath):
    """
    Parse a projwfc.x PDOS file.
    File format: E(eV)  ldos(E)  pdos(E)
    First line is a header comment.

    Returns
    -------
    energy : 1D array (eV)
    pdos : 1D array (states/eV)
    """
    data = np.loadtxt(filepath, comments="#")
    energy = data[:, 0]
    pdos = data[:, 1]  # LDOS column
    return energy, pdos


def find_pdos_files(metal_dir, metal, l_quantum=2):
    """
    Find PDOS files for d-orbitals (l=2) of surface atoms.

    QE projwfc output naming: {filpdos}.pdos_atm#{iatom}({symbol})_wfc#{iwfc}({label})
    For d-orbitals: label contains 'd'

    Parameters
    ----------
    metal_dir : str
        Directory containing PDOS files
    metal : str
        Metal symbol
    l_quantum : int
        Angular momentum quantum number (2 for d)

    Returns
    -------
    files : list of str
        Paths to PDOS files for d-orbitals
    """
    pattern = os.path.join(metal_dir, f"{metal}_pdos.pdos_atm*({metal})*_wfc*")
    all_files = sorted(glob.glob(pattern))

    # Filter for d-orbital files
    d_files = [f for f in all_files if "(d)" in f.lower() or "wfc#" in f]

    # More robust: check the header of each file for l=2
    d_files_verified = []
    for f in all_files:
        with open(f, "r") as fh:
            header = fh.readline()
        # Header typically contains: "# ... (l= 2, ...)"
        if "l= 2" in header or "l=2" in header:
            d_files_verified.append(f)

    return d_files_verified if d_files_verified else d_files


def compute_dband_center(energy, pdos, e_fermi=0.0):
    """
    Compute d-band center and related descriptors.

    Parameters
    ----------
    energy : 1D array (eV), relative to E_Fermi
    pdos : 1D array (states/eV)
    e_fermi : float, Fermi energy (if energy is absolute)

    Returns
    -------
    dict with:
        center : float, d-band center (eV relative to E_F)
        width : float, d-band width (eV)
        filling : float, d-band filling fraction
        upper_edge : float, upper d-band edge (eV)
        skewness : float, d-band skewness
    """
    E = energy - e_fermi

    # Filter to relevant energy range (where DOS is significant)
    mask = pdos > 1e-6 * np.max(pdos)
    if np.sum(mask) < 10:
        mask = np.ones_like(E, dtype=bool)

    # Total integral (normalization)
    norm = trapezoid(pdos, E)
    if norm < 1e-10:
        return {"center": 0.0, "width": 0.0, "filling": 0.0,
                "upper_edge": 0.0, "skewness": 0.0}

    # 1st moment: d-band center
    center = trapezoid(E * pdos, E) / norm

    # 2nd moment: d-band width
    variance = trapezoid((E - center)**2 * pdos, E) / norm
    width = np.sqrt(variance)

    # 3rd moment: skewness
    skewness = trapezoid((E - center)**3 * pdos, E) / (norm * width**3) if width > 0 else 0.0

    # d-band filling (fraction below E_F)
    mask_occ = E <= 0.0
    filling = trapezoid(pdos[mask_occ], E[mask_occ]) / norm if np.any(mask_occ) else 0.0

    # Upper edge (center + width)
    upper_edge = center + width

    return {
        "center": center,
        "width": width,
        "filling": filling,
        "upper_edge": upper_edge,
        "skewness": skewness,
    }


def get_fermi_energy(metal_dir):
    """Extract Fermi energy from QE SCF or NSCF output."""
    for fname in ["nscf.out", "scf.out"]:
        filepath = os.path.join(metal_dir, fname)
        if not os.path.exists(filepath):
            continue
        with open(filepath, "r") as f:
            for line in f:
                if "Fermi energy" in line or "highest occupied" in line:
                    parts = line.split()
                    for i, p in enumerate(parts):
                        if p == "is" or p == "level":
                            try:
                                return float(parts[i+1])
                            except (IndexError, ValueError):
                                pass
                        try:
                            val = float(p)
                            if -50 < val < 50:  # reasonable Fermi energy
                                return val
                        except ValueError:
                            continue
    return 0.0


def analyze_metal(metal, work_dir, surface_atom_indices=None):
    """
    Full d-band center analysis for one metal.

    Parameters
    ----------
    metal : str
    work_dir : str
    surface_atom_indices : list of int or None
        Indices of surface atoms (0-based). If None, auto-detect top layer.

    Returns
    -------
    dict with d-band descriptors
    """
    metal_dir = os.path.join(work_dir, metal)

    # Get Fermi energy
    e_fermi = get_fermi_energy(metal_dir)
    print(f"  {metal}: E_Fermi = {e_fermi:.3f} eV")

    # Find d-orbital PDOS files
    d_files = find_pdos_files(metal_dir, metal)
    if not d_files:
        print(f"  WARNING: No d-orbital PDOS files found for {metal}")
        return None

    # Sum d-PDOS over surface atoms
    # If surface_atom_indices not specified, use all atoms
    total_energy = None
    total_pdos = None

    for f in d_files:
        # Extract atom index from filename
        # Format: prefix.pdos_atm#1(Cu)_wfc#2(d)
        import re
        match = re.search(r"atm#(\d+)", f)
        if match:
            atom_idx = int(match.group(1))
            # If filtering by surface atoms
            if surface_atom_indices is not None:
                if atom_idx not in surface_atom_indices:
                    continue

        energy, pdos = parse_pdos_file(f)
        if total_energy is None:
            total_energy = energy
            total_pdos = pdos.copy()
        else:
            total_pdos += pdos

    if total_energy is None:
        print(f"  WARNING: Could not parse PDOS for {metal}")
        return None

    # Compute d-band descriptors
    result = compute_dband_center(total_energy, total_pdos, e_fermi)
    print(f"  d-band center: {result['center']:.3f} eV")
    print(f"  d-band width:  {result['width']:.3f} eV")
    print(f"  d-band filling: {result['filling']:.3f}")

    result["energy"] = total_energy - e_fermi
    result["pdos"] = total_pdos
    result["metal"] = metal

    return result


# ------------------------------------------------------------------
# Run analysis for all metals
# ------------------------------------------------------------------
results = {}
metals_list = ["Cu", "Ag", "Au", "Pd", "Pt", "Ni"]

print("="*60)
print("d-Band Center Analysis")
print("="*60)

for metal in metals_list:
    metal_dir = os.path.join(WORK_DIR, metal)
    if not os.path.exists(metal_dir):
        print(f"\n{metal}: Directory not found, skipping")
        continue

    print(f"\nAnalyzing {metal}...")
    # Use top-layer atom indices (atoms 13-16 for 4-layer 2x2 slab)
    # In practice, identify surface atoms by z-coordinate
    result = analyze_metal(metal, WORK_DIR, surface_atom_indices=None)
    if result:
        results[metal] = result

# ------------------------------------------------------------------
# If no DFT results available, use literature values for demonstration
# ------------------------------------------------------------------
if not results:
    print("\nNo DFT results found. Using literature d-band centers for demonstration.")
    # Literature values (PBE, (111) surface, d-band center in eV relative to E_F)
    literature_dband = {
        "Cu": -2.67, "Ag": -4.30, "Au": -3.56,
        "Pd": -1.83, "Pt": -2.25, "Ni": -1.29,
    }
    # Experimental adsorption energies of CO (eV, more negative = stronger)
    literature_E_CO = {
        "Cu": -0.50, "Ag": -0.10, "Au": -0.25,
        "Pd": -1.50, "Pt": -1.30, "Ni": -1.30,
    }
    for metal in metals_list:
        results[metal] = {
            "metal": metal,
            "center": literature_dband[metal],
            "E_CO": literature_E_CO[metal],
        }
```

### Step 4: Plot Results

```python
#!/usr/bin/env python3
"""
Create d-band center plots: (1) PDOS comparison, (2) d-band center bar chart,
(3) correlation with adsorption energy.
"""
import os
import numpy as np
import matplotlib.pyplot as plt

WORK_DIR = os.path.abspath("dband_screening")

# ------------------------------------------------------------------
# Use literature values for complete demonstration
# ------------------------------------------------------------------
metals_list = ["Ni", "Pd", "Pt", "Cu", "Au", "Ag"]

# d-band centers from DFT (PBE, 111 surface)
dband_centers = {
    "Cu": -2.67, "Ag": -4.30, "Au": -3.56,
    "Pd": -1.83, "Pt": -2.25, "Ni": -1.29,
}

# Experimental/DFT adsorption energies of CO on (111) surfaces (eV)
E_CO = {
    "Cu": -0.50, "Ag": -0.10, "Au": -0.25,
    "Pd": -1.50, "Pt": -1.30, "Ni": -1.30,
}

# Experimental/DFT adsorption energies of O on (111) surfaces (eV)
E_O = {
    "Cu": -1.20, "Ag": -0.50, "Au": -0.40,
    "Pd": -1.00, "Pt": -0.90, "Ni": -1.90,
}

# ------------------------------------------------------------------
# Plot 1: d-band center bar chart
# ------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(8, 5))

x_pos = range(len(metals_list))
centers = [dband_centers[m] for m in metals_list]
colors = plt.cm.RdYlBu(np.linspace(0.2, 0.8, len(metals_list)))

bars = ax.bar(x_pos, centers, color=colors, edgecolor="black", linewidth=1.2)
ax.set_xticks(x_pos)
ax.set_xticklabels(metals_list, fontsize=13)
ax.set_ylabel(r"$\varepsilon_d$ (eV)", fontsize=13)
ax.set_title("d-Band Center of (111) Transition Metal Surfaces", fontsize=14)
ax.axhline(y=0, color="gray", linewidth=0.5, linestyle="--")

# Add value labels on bars
for bar, val in zip(bars, centers):
    ax.text(bar.get_x() + bar.get_width()/2, val - 0.15,
            f"{val:.2f}", ha="center", va="top", fontsize=11, fontweight="bold")

plt.tight_layout()
plt.savefig(os.path.join(WORK_DIR, "dband_centers.png"), dpi=150)
plt.close()
print(f"d-band center plot saved to {os.path.join(WORK_DIR, 'dband_centers.png')}")

# ------------------------------------------------------------------
# Plot 2: d-band center vs CO adsorption energy
# ------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(7, 6))

x_vals = [dband_centers[m] for m in metals_list]
y_vals = [E_CO[m] for m in metals_list]

ax.scatter(x_vals, y_vals, s=120, c="steelblue", edgecolor="black",
           zorder=5, linewidth=1.2)

# Label each point
for m, x, y in zip(metals_list, x_vals, y_vals):
    ax.annotate(m, (x, y), textcoords="offset points", xytext=(8, 8),
                fontsize=12, fontweight="bold")

# Linear fit
coeffs = np.polyfit(x_vals, y_vals, 1)
x_fit = np.linspace(min(x_vals) - 0.3, max(x_vals) + 0.3, 100)
y_fit = np.polyval(coeffs, x_fit)
ax.plot(x_fit, y_fit, "r--", linewidth=1.5, alpha=0.7,
        label=f"Linear fit: slope={coeffs[0]:.2f}")

# R-squared
y_pred = np.polyval(coeffs, x_vals)
ss_res = np.sum((np.array(y_vals) - y_pred)**2)
ss_tot = np.sum((np.array(y_vals) - np.mean(y_vals))**2)
r_squared = 1 - ss_res / ss_tot

ax.set_xlabel(r"$\varepsilon_d$ (eV)", fontsize=13)
ax.set_ylabel(r"$E_{ads}$(CO) (eV)", fontsize=13)
ax.set_title(f"d-Band Center vs CO Adsorption Energy ($R^2$={r_squared:.3f})",
             fontsize=13)
ax.legend(fontsize=11)

plt.tight_layout()
plt.savefig(os.path.join(WORK_DIR, "dband_vs_CO.png"), dpi=150)
plt.close()
print(f"Correlation plot saved to {os.path.join(WORK_DIR, 'dband_vs_CO.png')}")

# ------------------------------------------------------------------
# Plot 3: d-band center vs O adsorption energy
# ------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(7, 6))

x_vals = [dband_centers[m] for m in metals_list]
y_vals_O = [E_O[m] for m in metals_list]

ax.scatter(x_vals, y_vals_O, s=120, c="darkorange", edgecolor="black",
           zorder=5, linewidth=1.2)

for m, x, y in zip(metals_list, x_vals, y_vals_O):
    ax.annotate(m, (x, y), textcoords="offset points", xytext=(8, 8),
                fontsize=12, fontweight="bold")

coeffs_O = np.polyfit(x_vals, y_vals_O, 1)
y_fit_O = np.polyval(coeffs_O, x_fit)
ax.plot(x_fit, y_fit_O, "r--", linewidth=1.5, alpha=0.7)

ax.set_xlabel(r"$\varepsilon_d$ (eV)", fontsize=13)
ax.set_ylabel(r"$E_{ads}$(O) (eV)", fontsize=13)
ax.set_title("d-Band Center vs O Adsorption Energy", fontsize=13)

plt.tight_layout()
plt.savefig(os.path.join(WORK_DIR, "dband_vs_O.png"), dpi=150)
plt.close()
print(f"O correlation plot saved to {os.path.join(WORK_DIR, 'dband_vs_O.png')}")

# ------------------------------------------------------------------
# Summary table
# ------------------------------------------------------------------
print("\n" + "="*70)
print(f"{'Metal':>6s} | {'eps_d (eV)':>10s} | {'E_CO (eV)':>10s} | "
      f"{'E_O (eV)':>10s}")
print("-"*70)
for m in metals_list:
    print(f"{m:>6s} | {dband_centers[m]:>10.2f} | {E_CO[m]:>10.2f} | "
          f"{E_O[m]:>10.2f}")
print("="*70)
```

### Optional: Quick Slab Optimization with MACE

```python
#!/usr/bin/env python3
"""
Use MACE for rapid surface slab optimization before QE PDOS calculation.
MACE does not provide PDOS directly -- use it only for geometry optimization.
"""
import os
import numpy as np
from ase.build import fcc111, add_adsorbate
from ase.optimize import BFGS
from ase.constraints import FixAtoms
from ase.io import write

# Import MACE calculator
try:
    from mace.calculators import mace_mp
    calc = mace_mp(model="medium", device="cpu", default_dtype="float64")
    MACE_AVAILABLE = True
    print("MACE calculator loaded successfully")
except ImportError:
    MACE_AVAILABLE = False
    print("MACE not available; skipping MACE optimization")


def optimize_slab_mace(metal, a, nlayers=4, vacuum=15.0, fmax=0.02):
    """
    Build and optimize a (111) slab with MACE.

    Parameters
    ----------
    metal : str
    a : float, lattice constant in Angstrom
    nlayers : int
    vacuum : float
    fmax : float, force convergence criterion (eV/Ang)

    Returns
    -------
    slab : optimized ASE Atoms
    """
    if not MACE_AVAILABLE:
        print(f"MACE not available; returning unrelaxed slab for {metal}")
        slab = fcc111(metal, size=(2, 2, nlayers), a=a, vacuum=vacuum, periodic=True)
        return slab

    slab = fcc111(metal, size=(2, 2, nlayers), a=a, vacuum=vacuum, periodic=True)

    # Fix bottom 2 layers
    positions = slab.get_positions()
    z_coords = positions[:, 2]
    z_sorted = np.sort(np.unique(np.round(z_coords, 2)))
    z_fix = z_sorted[1] + 0.1
    slab.set_constraint(FixAtoms(mask=z_coords < z_fix))

    slab.calc = calc
    opt = BFGS(slab, logfile=f"{metal}_mace_opt.log")
    opt.run(fmax=fmax, steps=200)

    energy = slab.get_potential_energy()
    print(f"{metal}: MACE energy = {energy:.4f} eV, converged in {opt.nsteps} steps")

    return slab


if MACE_AVAILABLE:
    lattice_constants = {"Cu": 3.615, "Ag": 4.086, "Au": 4.078,
                         "Pd": 3.890, "Pt": 3.924, "Ni": 3.524}

    for metal, a in lattice_constants.items():
        print(f"\nOptimizing {metal} slab with MACE...")
        slab = optimize_slab_mace(metal, a)
        outdir = os.path.join("dband_screening", metal)
        os.makedirs(outdir, exist_ok=True)
        write(os.path.join(outdir, f"{metal}_slab_relaxed.xyz"), slab)
        print(f"  Saved to {os.path.join(outdir, f'{metal}_slab_relaxed.xyz')}")
```

## Key Parameters

| Parameter | Typical Value | Notes |
|-----------|--------------|-------|
| Slab thickness | 4-6 layers | Converge d-band center vs layers |
| Vacuum | 15+ Angstrom | Prevent slab-slab interaction |
| k-grid (SCF) | 4x4x1 to 6x6x1 | Slab calculations; Gamma-centered |
| k-grid (NSCF/DOS) | 8x8x1 to 12x12x1 | Denser for smooth PDOS |
| `ecutwfc` | 40-80 Ry | Depends on pseudopotential |
| `degauss` for projwfc | 0.01-0.03 Ry | Broadening for PDOS; smaller = sharper peaks |
| `Emin`, `Emax` | -15 to +5 eV | Energy range for PDOS |
| `DeltaE` | 0.01-0.05 eV | Energy resolution |
| Fixed layers | Bottom 2 | Mimic bulk behavior |
| Smearing | Marzari-Vanderbilt (`mv`) | Good for metals |

## Interpreting Results

**d-band center ordering (typical, 111 surfaces):**
Ni (-1.3) > Pd (-1.8) > Pt (-2.3) > Cu (-2.7) > Au (-3.6) > Ag (-4.3) eV

**Correlation with adsorption:**
- More negative epsilon_d -> weaker binding (noble metals: Au, Ag)
- Less negative epsilon_d -> stronger binding (reactive metals: Ni, Pd)
- The d-band model works best for comparing similar adsorbates across different metals

**Limitations:**
- Correlation breaks down for very early transition metals (d-band nearly empty)
- Not reliable for comparing different adsorbates on the same metal
- Alloy surfaces may show site-dependent d-band centers
- Strain and ligand effects both shift the d-band center

**What to check:**
- PDOS should show a clear d-band with significant weight
- d-band center should be below E_F for late transition metals
- Total d-electron count should match expectations (e.g., ~10 for Cu, ~9 for Ni)

## Common Issues

1. **PDOS is noisy or has sharp spikes:**
   - Increase k-grid density for NSCF
   - Increase `degauss` in projwfc (but not too much -- smears real features)
   - Use tetrahedron method (`occupations = 'tetrahedra'`) for NSCF

2. **d-band center does not match literature:**
   - Check that surface atoms (not bulk-like atoms) are selected
   - Verify Fermi energy is correctly parsed
   - Ensure pseudopotential valence electron count is correct
   - Check for magnetic solutions (Ni, Co, Fe are magnetic)

3. **projwfc.x takes very long:**
   - Reduce number of bands (`nbnd`)
   - projwfc scales as O(nbnd * nk * natom * nproj)
   - Run in parallel if possible (limited parallelization in projwfc)

4. **Cannot identify d-orbital PDOS files:**
   - QE naming: `prefix.pdos_atm#N(Symbol)_wfc#M(d)` for d-orbitals
   - Parse the file header for `l= 2` to confirm d character
   - If using ultrasoft PPs, d-orbitals may have different wfc indices

5. **Slab not converged:**
   - Increase number of layers (test 4, 5, 6 layers)
   - Check interlayer spacing convergence after relaxation
   - Surface energy should converge with slab thickness
