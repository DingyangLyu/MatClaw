# Magnetic Ordering Comparison

## When to Use

- Determine whether a material's ground state is ferromagnetic (FM), antiferromagnetic (AFM),
  ferrimagnetic, or non-magnetic (NM).
- Compare total energies of different spin configurations to identify the most stable ordering.
- Estimate exchange coupling constants (J) from energy differences between magnetic orderings.
- Screen magnetic materials for specific properties (e.g., high Curie temperature candidates).

## Prerequisites

- Quantum ESPRESSO 7.5 with `pw.x`
- Python: `pymatgen`, `ase`, `numpy`, `matplotlib`
- Appropriate pseudopotentials (SSSP Efficiency or PseudoDojo, scalar-relativistic)
- Completed spin-polarized calculations skill (understanding of `nspin=2`,
  `starting_magnetization`)

## Detailed Steps

### Overview of Approach

1. Start with the primitive cell of the material.
2. For FM: set all magnetic atoms with the same sign of `starting_magnetization`.
3. For AFM: create a supercell (if needed) and assign opposite `starting_magnetization`
   to different magnetic sublattices. Define separate species types for spin-up and spin-down
   atoms (e.g., `Fe1`, `Fe2`).
4. For NM: set `nspin=1` (no spin polarization).
5. Run SCF for all configurations and compare total energies per formula unit.
6. The lowest energy configuration is the ground state magnetic order.

### Step 1: Setting Up Different Magnetic Orderings in QE

#### Ferromagnetic (FM)

All magnetic atoms have the same `starting_magnetization` sign:

```
    ntyp = 1
    starting_magnetization(1) = 0.6    ! Fe: all spin-up
```

#### Antiferromagnetic (AFM)

Different magnetic sublattices have opposite `starting_magnetization`. You must define
separate species types for atoms with different spin orientations:

```
    ntyp = 2
    starting_magnetization(1) =  0.6   ! Fe1: spin-up sublattice
    starting_magnetization(2) = -0.6   ! Fe2: spin-down sublattice
```

Both `Fe1` and `Fe2` use the **same pseudopotential file** in the `ATOMIC_SPECIES` block.

#### Non-Magnetic (NM)

Simply omit spin polarization:

```
    nspin = 1
    ! No starting_magnetization needed
```

### Step 2: Supercell Approach for AFM Configurations

Many AFM orderings require supercells. For example:

- **BCC Fe**: The primitive cell has 1 atom. A-type AFM (alternating ferromagnetic planes)
  requires at least a 2-atom cell. The conventional BCC cell (2 atoms) can represent
  simple AFM ordering.
- **FCC Ni**: The primitive cell has 1 atom. AF-I ordering needs a 2-atom cell;
  AF-II ordering (alternating (111) planes) needs a 4-atom cell.
- **MnO, NiO** (rocksalt): AF-II ordering (the ground state) requires the rhombohedral
  or conventional 4-atom cell.
- **Perovskites (e.g., LaMnO3)**: A-type, C-type, and G-type AFM need different supercells
  (often 2x or sqrt(2)x expansions).

### Step 3: Complete Python Workflow

The following script automates the entire workflow: generate FM/AFM/NM structures,
write QE inputs, run calculations, parse energies, compute J, and plot results.

```python
#!/usr/bin/env python3
"""
magnetic_ordering_comparison.py

Complete workflow to compare FM, AFM, and NM configurations for BCC Fe.
Generates QE inputs, runs calculations, parses results, estimates J,
and creates an energy comparison bar chart.

For BCC Fe, we use the conventional cell (2 atoms) to allow FM and AFM orderings.
FM: both atoms spin-up.  AFM: one atom spin-up, one spin-down.  NM: nspin=1.
"""

import os
import re
import subprocess
import numpy as np
import matplotlib.pyplot as plt
from pymatgen.core import Structure, Lattice, Element

# ============================================================
# Configuration
# ============================================================
PSEUDO_DIR = os.path.abspath("./pseudo")
OUTDIR_BASE = os.path.abspath("./tmp_magorder")
ECUTWFC = 60.0
ECUTRHO = 480.0
KGRID = (12, 12, 12)
DEGAUSS = 0.02
PSEUDO_FILE = "Fe.pbe-spn-kjpaw_psl.0.2.1.UPF"

os.makedirs(PSEUDO_DIR, exist_ok=True)

# BCC Fe conventional cell (2 atoms)
a_fe = 2.87  # Angstrom
lattice = Lattice.cubic(a_fe)
structure_bcc = Structure(
    lattice,
    ["Fe", "Fe"],
    [[0.0, 0.0, 0.0], [0.5, 0.5, 0.5]]
)

# ============================================================
# Define magnetic configurations
# ============================================================
configs = {
    "FM": {
        "nspin": 2,
        "species_map": {"Fe": ["Fe"]},        # all same species
        "site_species": ["Fe", "Fe"],
        "starting_mag": {"Fe": 0.6},
        "ntyp": 1,
    },
    "AFM": {
        "nspin": 2,
        "species_map": {"Fe": ["Fe1", "Fe2"]}, # two sublattices
        "site_species": ["Fe1", "Fe2"],         # atom 0 = up, atom 1 = down
        "starting_mag": {"Fe1": 0.6, "Fe2": -0.6},
        "ntyp": 2,
    },
    "NM": {
        "nspin": 1,
        "species_map": {"Fe": ["Fe"]},
        "site_species": ["Fe", "Fe"],
        "starting_mag": {},
        "ntyp": 1,
    },
}


# ============================================================
# Function: Write QE input
# ============================================================
def write_qe_input(structure, config_name, config, filename):
    """Write a QE SCF input file for a given magnetic configuration."""
    outdir = os.path.join(OUTDIR_BASE, config_name)
    os.makedirs(outdir, exist_ok=True)

    prefix = f"fe_{config_name.lower()}"
    nspin = config["nspin"]
    site_species = config["site_species"]
    starting_mag = config["starting_mag"]
    ntyp = config["ntyp"]

    # Unique species list (preserving order)
    seen = set()
    species_list = []
    for sp in site_species:
        if sp not in seen:
            species_list.append(sp)
            seen.add(sp)

    cell = structure.lattice.matrix

    lines = []
    lines.append("&CONTROL")
    lines.append(f"    calculation   = 'scf'")
    lines.append(f"    prefix        = '{prefix}'")
    lines.append(f"    outdir        = '{outdir}'")
    lines.append(f"    pseudo_dir    = '{PSEUDO_DIR}'")
    lines.append(f"    tprnfor       = .true.")
    lines.append(f"    tstress       = .true.")
    lines.append(f"    verbosity     = 'high'")
    lines.append("/")
    lines.append("&SYSTEM")
    lines.append(f"    ibrav         = 0")
    lines.append(f"    nat           = {len(structure)}")
    lines.append(f"    ntyp          = {ntyp}")
    lines.append(f"    ecutwfc       = {ECUTWFC}")
    lines.append(f"    ecutrho       = {ECUTRHO}")
    lines.append(f"    nspin         = {nspin}")

    if nspin == 2:
        for i, sp in enumerate(species_list, start=1):
            lines.append(f"    starting_magnetization({i}) = {starting_mag[sp]}")

    lines.append(f"    occupations   = 'smearing'")
    lines.append(f"    smearing      = 'mv'")
    lines.append(f"    degauss       = {DEGAUSS}")
    lines.append("/")
    lines.append("&ELECTRONS")
    lines.append(f"    conv_thr        = 1.0d-8")
    lines.append(f"    mixing_beta     = 0.3")
    lines.append(f"    electron_maxstep = 200")
    lines.append("/")

    # ATOMIC_SPECIES
    lines.append("ATOMIC_SPECIES")
    for sp in species_list:
        mass = Element("Fe").atomic_mass  # all species are Fe
        lines.append(f"    {sp:<4s}  {mass:.3f}  {PSEUDO_FILE}")

    # CELL_PARAMETERS
    lines.append("")
    lines.append("CELL_PARAMETERS angstrom")
    for v in cell:
        lines.append(f"    {v[0]:14.10f}  {v[1]:14.10f}  {v[2]:14.10f}")

    # ATOMIC_POSITIONS
    lines.append("")
    lines.append("ATOMIC_POSITIONS crystal")
    for i, site in enumerate(structure):
        fc = site.frac_coords
        sp = site_species[i]
        lines.append(f"    {sp:<4s}  {fc[0]:14.10f}  {fc[1]:14.10f}  {fc[2]:14.10f}")

    # K_POINTS
    lines.append("")
    lines.append("K_POINTS automatic")
    lines.append(f"    {KGRID[0]} {KGRID[1]} {KGRID[2]}  0 0 0")
    lines.append("")

    with open(filename, "w") as f:
        f.write("\n".join(lines))

    print(f"Written: {filename}")
    return filename


# ============================================================
# Function: Parse total energy from QE output
# ============================================================
def parse_total_energy(filename):
    """
    Parse the final total energy from pw.x output.
    Returns energy in Ry. Returns None if not found.
    """
    energy = None
    with open(filename, "r") as f:
        for line in f:
            if "!" in line and "total energy" in line:
                # Format: !    total energy              =    -254.12345678 Ry
                m = re.search(r"=\s*([\d\.\-]+)\s*Ry", line)
                if m:
                    energy = float(m.group(1))
    return energy


def parse_magnetization(filename):
    """Parse total and absolute magnetization from pw.x output."""
    total_mag = None
    abs_mag = None
    with open(filename, "r") as f:
        for line in f:
            if "total magnetization" in line and "Bohr" in line:
                m = re.search(r"=\s*([\d\.\-]+)\s*Bohr", line)
                if m:
                    total_mag = float(m.group(1))
            if "absolute magnetization" in line and "Bohr" in line:
                m = re.search(r"=\s*([\d\.\-]+)\s*Bohr", line)
                if m:
                    abs_mag = float(m.group(1))
    return total_mag, abs_mag


# ============================================================
# Generate all input files
# ============================================================
input_files = {}
output_files = {}

for config_name, config in configs.items():
    infile = f"fe_{config_name.lower()}_scf.in"
    outfile = f"fe_{config_name.lower()}_scf.out"
    write_qe_input(structure_bcc, config_name, config, infile)
    input_files[config_name] = infile
    output_files[config_name] = outfile


# ============================================================
# Run all calculations (uncomment to execute)
# ============================================================
def run_all_calculations():
    """Run all QE SCF calculations sequentially."""
    for config_name in configs:
        infile = input_files[config_name]
        outfile = output_files[config_name]
        print(f"\nRunning {config_name} calculation...")
        with open(infile, "r") as fin, open(outfile, "w") as fout:
            result = subprocess.run(
                ["pw.x", "-npool", "1"],
                stdin=fin, stdout=fout, stderr=subprocess.STDOUT
            )
        if result.returncode != 0:
            print(f"  WARNING: {config_name} calculation may have failed (rc={result.returncode})")
        else:
            print(f"  {config_name} calculation completed.")

# Uncomment the next line to actually run the calculations:
# run_all_calculations()


# ============================================================
# Parse results and compare
# ============================================================
def analyze_results():
    """Parse all output files and compare energies."""
    results = {}
    for config_name in configs:
        outfile = output_files[config_name]
        if not os.path.exists(outfile):
            print(f"Output file not found: {outfile}")
            continue

        energy = parse_total_energy(outfile)
        total_mag, abs_mag = parse_magnetization(outfile)
        results[config_name] = {
            "energy_ry": energy,
            "total_mag": total_mag,
            "abs_mag": abs_mag,
        }
        if energy is not None:
            print(f"{config_name}: E = {energy:.8f} Ry, "
                  f"M_tot = {total_mag}, M_abs = {abs_mag}")

    if len(results) < 2:
        print("Need at least 2 completed calculations to compare.")
        return results

    # Find ground state
    valid = {k: v for k, v in results.items() if v["energy_ry"] is not None}
    if valid:
        ground_state = min(valid, key=lambda k: valid[k]["energy_ry"])
        print(f"\nGround state: {ground_state}")

        # Energy differences relative to ground state (in meV/atom)
        e_gs = valid[ground_state]["energy_ry"]
        n_atoms = len(structure_bcc)
        ry_to_meV = 13605.693  # 1 Ry = 13605.693 meV

        print(f"\nEnergy differences (meV/atom) relative to {ground_state}:")
        for k, v in valid.items():
            de = (v["energy_ry"] - e_gs) * ry_to_meV / n_atoms
            print(f"  {k}: {de:+.2f} meV/atom")

    return results


# Uncomment to analyze results after calculations are complete:
# results = analyze_results()


# ============================================================
# Exchange Coupling Constant J Estimation
# ============================================================
def estimate_exchange_J(e_fm_ry, e_afm_ry, n_atoms, z_nn, S):
    """
    Estimate the nearest-neighbor exchange coupling constant J from
    the energy difference between FM and AFM orderings.

    Using the Heisenberg model: H = -J * sum_{<ij>} S_i . S_j

    For a system with z nearest neighbors per magnetic atom:
        E_FM  = E_0 - N * z * J * S^2
        E_AFM = E_0 + N * z * J * S^2  (for simple AFM with half NN flipped)

    So: E_AFM - E_FM = 2 * N * z * J * S^2
    => J = (E_AFM - E_FM) / (2 * N * z * S^2)

    Note: This is a simplified formula. The exact relation depends on
    the crystal structure and the specific AFM ordering.

    For BCC with AFM ordering where each atom has 8 NN all of opposite spin:
        E_FM  = E_0 - N * (z/2) * J * S^2
        E_AFM = E_0 + N * (z/2) * J * S^2
        => J = (E_AFM - E_FM) / (N * z * S^2)

    Parameters:
        e_fm_ry:  FM total energy in Ry
        e_afm_ry: AFM total energy in Ry
        n_atoms:  number of magnetic atoms in the cell
        z_nn:     number of nearest neighbors
        S:        spin quantum number (e.g., for Fe: ~1.1 from moment ~2.2 uB)

    Returns:
        J in meV
    """
    ry_to_meV = 13605.693
    de = (e_afm_ry - e_fm_ry) * ry_to_meV  # meV per cell
    J = de / (n_atoms * z_nn * S ** 2)
    return J


# Example (using typical BCC Fe values):
# E_FM  ~ -5098.12345 Ry (hypothetical)
# E_AFM ~ -5098.08000 Ry (hypothetical, higher energy)
# BCC: z = 8 nearest neighbors, S ~ 1.1 (moment ~ 2.2 uB)
# J = (E_AFM - E_FM) * 13605.693 / (2 * 8 * 1.1^2)
# For Fe, J should be positive (FM ground state), ~15-20 meV

print("\n--- Example J estimation ---")
e_fm_example = -5098.12345
e_afm_example = -5098.08000
J_example = estimate_exchange_J(e_fm_example, e_afm_example, n_atoms=2, z_nn=8, S=1.1)
print(f"E_FM  = {e_fm_example:.5f} Ry")
print(f"E_AFM = {e_afm_example:.5f} Ry")
print(f"J = {J_example:.2f} meV")
print(f"J > 0 => FM ground state" if J_example > 0 else f"J < 0 => AFM ground state")


# ============================================================
# Visualization: Energy comparison bar chart
# ============================================================
def plot_energy_comparison(results, n_atoms, output_file="magnetic_ordering_comparison.png"):
    """
    Create a bar chart comparing energies of different magnetic configurations.

    Parameters:
        results: dict with keys=config_names, values=dict with 'energy_ry'
        n_atoms: number of atoms per cell (for per-atom normalization)
        output_file: output filename for the plot
    """
    valid = {k: v for k, v in results.items() if v["energy_ry"] is not None}
    if len(valid) < 2:
        print("Need at least 2 results to plot.")
        return

    # Find ground state energy
    e_min = min(v["energy_ry"] for v in valid.values())
    ry_to_meV = 13605.693

    config_names = list(valid.keys())
    energies_meV = [(valid[k]["energy_ry"] - e_min) * ry_to_meV / n_atoms
                    for k in config_names]

    # Color coding
    colors = []
    for name in config_names:
        if name == "FM":
            colors.append("steelblue")
        elif name == "AFM":
            colors.append("tomato")
        elif name == "NM":
            colors.append("gray")
        else:
            colors.append("seagreen")

    fig, ax = plt.subplots(figsize=(6, 5))
    bars = ax.bar(config_names, energies_meV, color=colors, edgecolor="black",
                  linewidth=1.2, width=0.5)

    # Annotate bars with values
    for bar, val in zip(bars, energies_meV):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                f"{val:.1f}", ha="center", va="bottom", fontsize=12, fontweight="bold")

    ax.set_ylabel("$\\Delta E$ (meV/atom)", fontsize=13)
    ax.set_title("Magnetic Ordering Comparison", fontsize=14)
    ax.set_ylim(bottom=0)

    # Mark ground state
    gs_idx = energies_meV.index(min(energies_meV))
    ax.annotate("Ground State", xy=(gs_idx, 0), xytext=(gs_idx, max(energies_meV) * 0.5),
                fontsize=11, ha="center", color="green",
                arrowprops=dict(arrowstyle="->", color="green", lw=1.5))

    plt.tight_layout()
    plt.savefig(output_file, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"Saved: {output_file}")


# Demo plot with example data
demo_results = {
    "FM":  {"energy_ry": -5098.12345},
    "AFM": {"energy_ry": -5098.08000},
    "NM":  {"energy_ry": -5098.00000},
}
plot_energy_comparison(demo_results, n_atoms=2,
                       output_file="magnetic_ordering_comparison_demo.png")


# ============================================================
# Extended Example: MnO with Multiple AFM Orderings
# ============================================================
def generate_mno_configs():
    """
    Generate QE input files for MnO in FM, AFM-I, AFM-II, and NM configurations.

    MnO has rocksalt structure.
    - FM:     All Mn spin-up
    - AFM-I:  Alternating (001) planes of Mn spin-up / spin-down
    - AFM-II: Alternating (111) planes (ground state for MnO)
    - NM:     No spin polarization

    We use the conventional cubic cell (4 Mn + 4 O = 8 atoms).
    """
    a_mno = 4.445  # Angstrom, experimental lattice constant

    # Conventional rocksalt cell: 4 formula units
    lattice = Lattice.cubic(a_mno)

    # Mn positions (FCC sublattice): (0,0,0), (0.5,0.5,0), (0.5,0,0.5), (0,0.5,0.5)
    # O positions:  (0.5,0,0), (0,0.5,0), (0,0,0.5), (0.5,0.5,0.5)

    mn_pos = [[0, 0, 0], [0.5, 0.5, 0], [0.5, 0, 0.5], [0, 0.5, 0.5]]
    o_pos = [[0.5, 0, 0], [0, 0.5, 0], [0, 0, 0.5], [0.5, 0.5, 0.5]]

    species = ["Mn"] * 4 + ["O"] * 4
    positions = mn_pos + o_pos

    structure = Structure(lattice, species, positions)

    # AFM-II ordering for rocksalt MnO:
    # The (111) planes alternate. In the conventional cell:
    # Mn at (0,0,0):       sum of indices = 0 (even) -> spin up
    # Mn at (0.5,0.5,0):   sum = 1 (odd)  -> spin down
    # Mn at (0.5,0,0.5):   sum = 1 (odd)  -> spin down
    # Mn at (0,0.5,0.5):   sum = 1 (odd)  -> spin down
    # Wait -- that gives 1 up and 3 down. For proper AFM-II, we need the
    # rhombohedral cell or a different assignment.

    # Actually, AFM-II for rocksalt uses the (111) plane criterion:
    # plane index = round(x + y + z) for positions in fractional coords
    # (0,0,0) -> 0, (0.5,0.5,0) -> 1, (0.5,0,0.5) -> 1, (0,0.5,0.5) -> 1
    # This gives 3 atoms in one sublattice and 1 in another (not balanced).
    #
    # For proper AFM-II, use the PRIMITIVE rocksalt cell with 2 formula units
    # (rhombohedral setting) or use the 4-atom rhombohedral cell.
    # Alternatively, use a tetragonal supercell.
    #
    # For simplicity, we demonstrate AFM-I (alternating 001 planes):
    # (0,0,0) -> z=0.0 plane -> spin up
    # (0.5,0.5,0) -> z=0.0 plane -> spin up
    # (0.5,0,0.5) -> z=0.5 plane -> spin down
    # (0,0.5,0.5) -> z=0.5 plane -> spin down

    afm1_site_species = ["Mn1", "Mn1", "Mn2", "Mn2", "O", "O", "O", "O"]

    # FM: all Mn the same
    fm_site_species = ["Mn"] * 4 + ["O"] * 4

    configs = {
        "MnO_FM": {
            "nspin": 2,
            "site_species": fm_site_species,
            "species_list": ["Mn", "O"],
            "starting_mag": {"Mn": 0.8, "O": 0.0},
            "ntyp": 2,
            "pseudo": {"Mn": "Mn.pbe-spn-kjpaw_psl.0.3.1.UPF",
                       "O": "O.pbe-n-kjpaw_psl.1.0.0.UPF"},
            "hubbard": {"Mn": 3.9},
        },
        "MnO_AFM1": {
            "nspin": 2,
            "site_species": afm1_site_species,
            "species_list": ["Mn1", "Mn2", "O"],
            "starting_mag": {"Mn1": 0.8, "Mn2": -0.8, "O": 0.0},
            "ntyp": 3,
            "pseudo": {"Mn1": "Mn.pbe-spn-kjpaw_psl.0.3.1.UPF",
                       "Mn2": "Mn.pbe-spn-kjpaw_psl.0.3.1.UPF",
                       "O": "O.pbe-n-kjpaw_psl.1.0.0.UPF"},
            "hubbard": {"Mn1": 3.9, "Mn2": 3.9},
        },
        "MnO_NM": {
            "nspin": 1,
            "site_species": fm_site_species,
            "species_list": ["Mn", "O"],
            "starting_mag": {},
            "ntyp": 2,
            "pseudo": {"Mn": "Mn.pbe-spn-kjpaw_psl.0.3.1.UPF",
                       "O": "O.pbe-n-kjpaw_psl.1.0.0.UPF"},
            "hubbard": {},
        },
    }

    for config_name, cfg in configs.items():
        lines = []
        outdir = os.path.join(OUTDIR_BASE, config_name)
        os.makedirs(outdir, exist_ok=True)
        prefix = config_name.lower()

        lines.append("&CONTROL")
        lines.append(f"    calculation   = 'scf'")
        lines.append(f"    prefix        = '{prefix}'")
        lines.append(f"    outdir        = '{outdir}'")
        lines.append(f"    pseudo_dir    = '{PSEUDO_DIR}'")
        lines.append(f"    verbosity     = 'high'")
        lines.append("/")
        lines.append("&SYSTEM")
        lines.append(f"    ibrav         = 0")
        lines.append(f"    nat           = {len(structure)}")
        lines.append(f"    ntyp          = {cfg['ntyp']}")
        lines.append(f"    ecutwfc       = 60.0")
        lines.append(f"    ecutrho       = 480.0")
        lines.append(f"    nspin         = {cfg['nspin']}")

        if cfg["nspin"] == 2:
            for i, sp in enumerate(cfg["species_list"], start=1):
                if sp in cfg["starting_mag"]:
                    lines.append(f"    starting_magnetization({i}) = {cfg['starting_mag'][sp]}")

        if cfg.get("hubbard"):
            lines.append(f"    lda_plus_u    = .true.")
            for i, sp in enumerate(cfg["species_list"], start=1):
                if sp in cfg["hubbard"]:
                    lines.append(f"    Hubbard_U({i}) = {cfg['hubbard'][sp]}")

        lines.append(f"    occupations   = 'smearing'")
        lines.append(f"    smearing      = 'gauss'")
        lines.append(f"    degauss       = 0.005")
        lines.append("/")
        lines.append("&ELECTRONS")
        lines.append(f"    conv_thr        = 1.0d-8")
        lines.append(f"    mixing_beta     = 0.2")
        lines.append(f"    electron_maxstep = 300")
        lines.append("/")

        lines.append("ATOMIC_SPECIES")
        for sp in cfg["species_list"]:
            base_elem = re.sub(r"\d+", "", sp)
            mass = Element(base_elem).atomic_mass
            lines.append(f"    {sp:<4s}  {mass:.3f}  {cfg['pseudo'][sp]}")

        lines.append("")
        lines.append("CELL_PARAMETERS angstrom")
        cell = structure.lattice.matrix
        for v in cell:
            lines.append(f"    {v[0]:14.10f}  {v[1]:14.10f}  {v[2]:14.10f}")

        lines.append("")
        lines.append("ATOMIC_POSITIONS crystal")
        for idx, site in enumerate(structure):
            fc = site.frac_coords
            sp = cfg["site_species"][idx]
            lines.append(f"    {sp:<4s}  {fc[0]:14.10f}  {fc[1]:14.10f}  {fc[2]:14.10f}")

        lines.append("")
        lines.append("K_POINTS automatic")
        lines.append("    6 6 6  0 0 0")
        lines.append("")

        filename = f"{config_name.lower()}_scf.in"
        with open(filename, "w") as f:
            f.write("\n".join(lines))
        print(f"Written: {filename}")


generate_mno_configs()
```

### Step 4: Run Calculations

```bash
# Run sequentially
for config in FM AFM NM; do
    echo "Running $config..."
    pw.x < fe_${config,,}_scf.in > fe_${config,,}_scf.out
done

# Or for MnO:
for config in mno_fm mno_afm1 mno_nm; do
    echo "Running $config..."
    pw.x < ${config}_scf.in > ${config}_scf.out
done
```

### Step 5: Parse and Compare Results

```python
#!/usr/bin/env python3
"""
parse_and_compare.py

Parse QE outputs from multiple magnetic ordering calculations,
compare energies, estimate J, and generate plots.
"""

import re
import os
import numpy as np
import matplotlib.pyplot as plt


def parse_qe_output(filename):
    """Parse total energy, magnetization, and convergence info from pw.x output."""
    result = {
        "energy_ry": None,
        "total_mag": None,
        "abs_mag": None,
        "converged": False,
        "n_iterations": None,
    }

    if not os.path.exists(filename):
        return result

    with open(filename, "r") as f:
        content = f.read()

    # Total energy
    matches = re.findall(r"!\s+total energy\s+=\s+([\d\.\-]+)\s*Ry", content)
    if matches:
        result["energy_ry"] = float(matches[-1])

    # Magnetization
    mag_matches = re.findall(
        r"total magnetization\s+=\s+([\d\.\-]+)\s*Bohr", content
    )
    if mag_matches:
        result["total_mag"] = float(mag_matches[-1])

    abs_mag_matches = re.findall(
        r"absolute magnetization\s+=\s+([\d\.\-]+)\s*Bohr", content
    )
    if abs_mag_matches:
        result["abs_mag"] = float(abs_mag_matches[-1])

    # Convergence
    if "convergence has been achieved" in content:
        result["converged"] = True
        m = re.search(r"convergence has been achieved in\s+(\d+)\s+iterations", content)
        if m:
            result["n_iterations"] = int(m.group(1))

    return result


def compare_orderings(output_files, n_atoms_per_fu, n_fu):
    """
    Compare magnetic orderings from parsed QE outputs.

    Parameters:
        output_files: dict {config_name: filename}
        n_atoms_per_fu: atoms per formula unit (for normalization)
        n_fu: number of formula units in cell
    """
    ry_to_meV = 13605.693
    ry_to_eV = 13.605693

    results = {}
    for name, fname in output_files.items():
        results[name] = parse_qe_output(fname)

    print("=" * 70)
    print(f"{'Config':<12s} {'E (Ry)':<18s} {'M_tot (uB)':<12s} "
          f"{'M_abs (uB)':<12s} {'Conv':<6s}")
    print("-" * 70)

    for name, r in results.items():
        e_str = f"{r['energy_ry']:.8f}" if r["energy_ry"] is not None else "N/A"
        mt_str = f"{r['total_mag']:.4f}" if r["total_mag"] is not None else "N/A"
        ma_str = f"{r['abs_mag']:.4f}" if r["abs_mag"] is not None else "N/A"
        cv_str = "Yes" if r["converged"] else "No"
        print(f"{name:<12s} {e_str:<18s} {mt_str:<12s} {ma_str:<12s} {cv_str:<6s}")

    # Energy comparison
    valid = {k: v for k, v in results.items()
             if v["energy_ry"] is not None and v["converged"]}

    if len(valid) >= 2:
        e_min = min(v["energy_ry"] for v in valid.values())
        gs = min(valid, key=lambda k: valid[k]["energy_ry"])

        print(f"\nGround state: {gs}")
        print(f"\nEnergy differences relative to {gs}:")

        for name in valid:
            de_meV = (valid[name]["energy_ry"] - e_min) * ry_to_meV / (n_atoms_per_fu * n_fu)
            de_eV = (valid[name]["energy_ry"] - e_min) * ry_to_eV / (n_atoms_per_fu * n_fu)
            print(f"  {name}: {de_meV:+.2f} meV/atom ({de_eV:+.6f} eV/atom)")

    return results


# Example usage:
output_files = {
    "FM":  "fe_fm_scf.out",
    "AFM": "fe_afm_scf.out",
    "NM":  "fe_nm_scf.out",
}

# Check which outputs exist
existing = {k: v for k, v in output_files.items() if os.path.exists(v)}
if existing:
    compare_orderings(existing, n_atoms_per_fu=1, n_fu=2)
else:
    print("No output files found. Run QE calculations first.")
    print("Showing demonstration with example data.\n")

    # Demo comparison
    demo_data = {
        "FM":  {"energy_ry": -254.12345, "total_mag": 4.40, "abs_mag": 4.45, "converged": True},
        "AFM": {"energy_ry": -254.08000, "total_mag": 0.00, "abs_mag": 4.30, "converged": True},
        "NM":  {"energy_ry": -254.00000, "total_mag": None, "abs_mag": None, "converged": True},
    }

    ry_to_meV = 13605.693
    e_min = min(v["energy_ry"] for v in demo_data.values())
    n_atoms = 2

    print(f"{'Config':<8s} {'dE (meV/atom)':<16s} {'M_tot (uB/cell)':<18s}")
    print("-" * 45)
    for name, d in demo_data.items():
        de = (d["energy_ry"] - e_min) * ry_to_meV / n_atoms
        mt = d["total_mag"] if d["total_mag"] is not None else 0.0
        print(f"{name:<8s} {de:+12.2f}     {mt:>8.2f}")
```

## Key Parameters

| Parameter | Typical Values | Notes |
|---|---|---|
| `nspin` | `1` (NM) or `2` (spin-polarized) | Must be 2 for FM and AFM calculations. |
| `starting_magnetization` | -1.0 to 1.0 per species | Opposite signs for AFM sublattices. Use 0.5-0.8 for 3d metals. |
| `ntyp` | Increases for AFM | Need separate species types for spin-up/spin-down sublattices. |
| `ecutwfc` | 40-80 Ry | Must be consistent across all configurations for fair comparison. |
| `mixing_beta` | 0.2-0.3 | Lower values for AFM (harder convergence). |
| `conv_thr` | 1.0d-8 to 1.0d-10 | Tight convergence needed for meV-level energy differences. |
| `lda_plus_u` | `.true.` for TM oxides | Essential for correct magnetic ground state in strongly correlated systems. |

## Interpreting Results

### Energy Ordering

- **E(FM) < E(AFM) < E(NM)**: Material is ferromagnetic.
- **E(AFM) < E(FM) < E(NM)**: Material is antiferromagnetic.
- **E(NM) < E(FM) and E(NM) < E(AFM)**: Material is non-magnetic (or very weakly magnetic).
- Energy differences are typically 10-500 meV/atom for strongly magnetic materials,
  but can be <1 meV/atom for weak magnetism.

### Exchange Coupling J

- **J > 0**: Ferromagnetic coupling (FM is lower energy).
- **J < 0**: Antiferromagnetic coupling (AFM is lower energy).
- Typical values: J ~ 10-50 meV for 3d transition metals and their oxides.
- The Heisenberg model is only a rough approximation; itinerant magnets (like Fe, Co, Ni)
  are not perfectly described by localized spin models.

### Magnetic Moments

- FM: Total magnetization should be nonzero and equal to sum of atomic moments.
- AFM: Total magnetization should be ~0 (opposite moments cancel). Absolute magnetization
  is nonzero and indicates the magnitude of local moments.
- NM: No magnetization.

### Sanity Checks

- The absolute magnetization in the AFM state should be similar to the FM state
  (the local moments are similar, just ordered differently).
- If the AFM calculation converges to FM (or vice versa), the initial magnetic guess
  was too weak, or the system strongly prefers one ordering. Increase
  `starting_magnetization` or try `mixing_mode = 'local-TF'`.

## Common Issues

1. **AFM converges to FM**: The initial `starting_magnetization` is too weak, or the
   energy landscape favors FM so strongly that the SCF cannot maintain the AFM ordering.
   Solutions: increase `starting_magnetization` to 0.8-1.0; reduce `mixing_beta` to 0.1;
   try `mixing_mode = 'local-TF'`.

2. **Wrong supercell for AFM**: Make sure the supercell is compatible with the desired
   AFM ordering. For example, AF-II in rocksalt structures requires specific cell shapes.
   Use pymatgen's `MagOrderingTransformation` or manual construction.

3. **Inconsistent parameters**: All configurations must use the same `ecutwfc`, `ecutrho`,
   `pseudo_dir`, pseudopotentials, and k-point density for fair energy comparison.
   If supercells differ in size, scale the k-grid inversely with cell size.

4. **Multiple species with same pseudopotential**: When defining `Fe1` and `Fe2` for AFM,
   both must reference the same `.UPF` file in `ATOMIC_SPECIES`. QE uses the species
   label only for bookkeeping; the physics comes from the pseudopotential.

5. **Structural relaxation**: For accurate energy comparisons, relax atomic positions
   (and optionally cell shape) for each magnetic configuration separately
   (`calculation = 'relax'` or `'vc-relax'`). Different magnetic orderings can have
   different equilibrium structures (magnetostructural coupling).

6. **k-point convergence**: Energy differences between magnetic orderings can be sensitive
   to k-grid density. Converge the energy difference (not just absolute energies) with
   respect to k-points. A good rule of thumb is k * a ~ 30-50 Angstrom (where a is the
   lattice parameter and k is the number of k-points along that direction).
