# Phase Diagram Construction

## When to Use

- You need to build a **temperature-composition** or **temperature-pressure** phase diagram from first principles.
- You want to determine **phase boundaries** between competing polymorphs (e.g., BCC vs. FCC vs. HCP).
- You need to compute **free energies** of competing phases including vibrational (phonon) entropy.
- You want to use the **quasi-harmonic approximation (QHA)** to account for thermal expansion and finite-temperature phase stability.
- You need to construct a **convex hull** from DFT energies and/or Materials Project data to assess thermodynamic stability.
- You are studying **pressure-induced phase transitions** and need P-V equations of state.

## Method Selection

| Scenario | Recommended Approach | Tools |
|---|---|---|
| T=0 K stability (convex hull) | DFT total energies of competing phases | QE or VASP + pymatgen |
| Stability with Materials Project data | Query MP for existing formation energies | mp-api + pymatgen |
| T-dependent phase boundaries (harmonic) | Phonon free energy F(T) = E_DFT + F_vib(T) | QE (ph.x) or ASE+MACE + phonopy |
| T-P phase diagram with thermal expansion | Quasi-harmonic approximation (QHA) | QE (ph.x at multiple volumes) + phonopy-qha |
| Binary alloy phase diagram | Cluster expansion + Monte Carlo | ATAT or icet (not covered in detail) |
| Melting curve P(T) | Two-phase coexistence MD at multiple pressures | ASE+MACE or LAMMPS (see mpmorph-melting skill) |
| Magnetic phase transitions | DFT + magnetic configurations | QE or VASP with spin-polarized calculations |

## Prerequisites

- Quantum ESPRESSO 7.5: `pw.x`, `ph.x`, `q2r.x`, `matdyn.x`
- Python packages: `numpy`, `scipy`, `matplotlib`, `pymatgen`, `mp-api`, `ase`, `phonopy` (install via `pip install phonopy`)
- MACE-MP-0 for quick phonon calculations via finite displacement: `mace-torch`
- Materials Project API key (for convex hull queries): set `MP_API_KEY` environment variable

## Detailed Steps

### Method A: T=0 K Convex Hull from Materials Project

The simplest phase diagram is the T=0 K convex hull, showing which compositions are thermodynamically stable.

```python
#!/usr/bin/env python3
"""
convex_hull_mp.py
Construct a convex hull phase diagram from Materials Project data.
Identifies stable phases and their formation energies.
"""
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os

from mp_api.client import MPRester
from pymatgen.analysis.phase_diagram import PhaseDiagram, PDPlotter
from pymatgen.core import Composition

# =============================================================================
# Configuration
# =============================================================================

# Chemical system to study (e.g., binary: "Ti-O", ternary: "Li-Fe-O")
CHEMICAL_SYSTEM = "Ti-O"

# Materials Project API key
API_KEY = os.environ.get("MP_API_KEY", None)
if API_KEY is None:
    print("WARNING: MP_API_KEY not set. Set it with:")
    print("  export MP_API_KEY='your_api_key_here'")

WORK_DIR = "/tmp/phase_diagram"
os.makedirs(WORK_DIR, exist_ok=True)

# =============================================================================
# 1. Query Materials Project
# =============================================================================

print(f"Querying Materials Project for system: {CHEMICAL_SYSTEM}")

with MPRester(API_KEY) as mpr:
    # Get all entries in the chemical system
    entries = mpr.get_entries_in_chemsys(
        CHEMICAL_SYSTEM.split("-"),
        additional_criteria={"is_stable": False}  # Get all, not just stable
    )

print(f"Found {len(entries)} entries.")

# =============================================================================
# 2. Construct Phase Diagram
# =============================================================================

pd = PhaseDiagram(entries)

# Print stable phases
print(f"\nStable phases on the convex hull:")
print(f"{'Composition':>20} {'Formation E (eV/atom)':>22} {'Space Group':>15}")
print("-" * 60)

for entry in pd.stable_entries:
    comp = entry.composition.reduced_formula
    e_form = pd.get_form_energy_per_atom(entry)
    # Try to get structure info
    try:
        sg = entry.structure.get_space_group_info()[0] if hasattr(entry, 'structure') else "N/A"
    except Exception:
        sg = "N/A"
    print(f"{comp:>20} {e_form:>22.4f} {sg:>15}")

# Print decomposition for unstable phases (top 10 by energy above hull)
print(f"\nUnstable phases (top 10 by distance from hull):")
print(f"{'Composition':>20} {'E above hull (eV/atom)':>22}")
print("-" * 45)

unstable = []
for entry in entries:
    if entry not in pd.stable_entries:
        e_above_hull = pd.get_e_above_hull(entry)
        unstable.append((entry, e_above_hull))

unstable.sort(key=lambda x: x[1])
for entry, e_ah in unstable[:10]:
    comp = entry.composition.reduced_formula
    print(f"{comp:>20} {e_ah:>22.4f}")

# =============================================================================
# 3. Plot Phase Diagram
# =============================================================================

elements = CHEMICAL_SYSTEM.split("-")

if len(elements) == 2:
    # Binary phase diagram
    plotter = PDPlotter(pd, show_unstable=0.1)
    fig = plotter.get_plot()
    fig.savefig(os.path.join(WORK_DIR, 'convex_hull.png'),
                dpi=200, bbox_inches='tight')
    plt.close()
    print(f"\nSaved: {WORK_DIR}/convex_hull.png")

elif len(elements) == 3:
    # Ternary phase diagram
    plotter = PDPlotter(pd, show_unstable=0.05)
    fig = plotter.get_plot()
    fig.savefig(os.path.join(WORK_DIR, 'ternary_phase_diagram.png'),
                dpi=200, bbox_inches='tight')
    plt.close()
    print(f"\nSaved: {WORK_DIR}/ternary_phase_diagram.png")
else:
    print("Phase diagram plotting supports 2-3 component systems visually.")

# =============================================================================
# 4. Check stability of a specific composition
# =============================================================================

test_comp = Composition("TiO2")
decomp, e_above_hull = pd.get_decomp_and_e_above_hull(
    pd.get_form_energy(pd.all_entries[0])  # placeholder
)

# Better approach: find the entry for TiO2
for entry in entries:
    if entry.composition.reduced_formula == "TiO2":
        e_ah = pd.get_e_above_hull(entry)
        decomp_products = pd.get_decomposition(entry.composition)
        print(f"\nTiO2 entry: E above hull = {e_ah:.4f} eV/atom")
        if e_ah < 1e-6:
            print("  --> STABLE (on the convex hull)")
        else:
            print(f"  --> UNSTABLE, decomposes to: {decomp_products}")
        break
```

### Method B: Finite-Temperature Phase Boundaries via Phonon Free Energy

For temperature-dependent phase stability, compute the Helmholtz free energy:

```
F(T) = E_DFT + F_vib(T)
F_vib(T) = sum_q,nu [ (1/2)*hbar*omega + kT * ln(1 - exp(-hbar*omega/kT)) ]
```

The phase boundary is where F_phase1(T) = F_phase2(T).

#### Step 1: Compute phonon dispersions for each competing phase

For each phase, run the phonon workflow:

```bash
# Phase 1: e.g., BCC Ti
mkdir -p phase_bcc && cd phase_bcc
# ... set up scf.in for BCC Ti ...
mpirun -np 4 pw.x -npool 2 < scf.in > scf.out 2>&1
mpirun -np 4 ph.x -npool 2 < ph.in > ph.out 2>&1
q2r.x < q2r.in > q2r.out 2>&1
matdyn.x < matdyn_dos.in > matdyn_dos.out 2>&1  # phonon DOS

cd ..

# Phase 2: e.g., HCP Ti
mkdir -p phase_hcp && cd phase_hcp
# ... same workflow ...
cd ..
```

#### Step 2: Compute free energy and find phase boundary

```python
#!/usr/bin/env python3
"""
phase_boundary_phonon.py
Compute T-dependent free energies of competing phases from DFT+phonon
and determine the phase boundary temperature.

Workflow:
  1. Read DFT total energies for each phase
  2. Read phonon DOS for each phase
  3. Compute vibrational free energy F_vib(T) from phonon DOS
  4. Total free energy: F(T) = E_DFT + F_vib(T)
  5. Phase boundary: F_phase1(T) = F_phase2(T)
"""
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.optimize import brentq
import os

# Physical constants
kB_eV = 8.617333e-5     # eV/K
hbar_eV = 6.582119e-16  # eV*s
cm_to_eV = 1.23984e-4   # 1 cm^-1 = 1.23984e-4 eV

WORK_DIR = "/tmp/phase_boundary"
os.makedirs(WORK_DIR, exist_ok=True)

# =============================================================================
# Configuration: phases to compare
# =============================================================================

# Each phase needs:
#   - DFT total energy per atom (eV/atom)
#   - Phonon DOS file (columns: omega[cm^-1], g(omega))
#   - Number of atoms per formula unit

phases = {
    'BCC': {
        'E_DFT': -7.8923,                    # eV/atom (example for Ti)
        'phonon_dos_file': 'phase_bcc/phonon_dos.dat',
        'nat_per_fu': 1,                      # atoms per formula unit
        'label': 'BCC Ti',
    },
    'HCP': {
        'E_DFT': -7.9102,                    # eV/atom
        'phonon_dos_file': 'phase_hcp/phonon_dos.dat',
        'nat_per_fu': 2,
        'label': 'HCP Ti',
    },
}

T_range = np.arange(0, 2001, 10)  # 0 to 2000 K

# =============================================================================
# Functions
# =============================================================================

def load_phonon_dos(filename):
    """
    Load phonon DOS from file.
    Expected format: columns of omega(cm^-1) and g(omega).
    Returns omega in eV and normalized g(omega).
    """
    data = np.loadtxt(filename, comments='#')
    omega_cm = data[:, 0]
    g_omega = data[:, 1]

    # Keep only positive frequencies
    mask = omega_cm > 0.1
    omega_cm = omega_cm[mask]
    g_omega = g_omega[mask]

    # Convert cm^-1 to eV
    omega_eV = omega_cm * cm_to_eV

    # Normalize: integral g(omega) d_omega = 3*N_atoms (number of phonon modes)
    # The normalization is handled by using per-atom quantities consistently.

    return omega_eV, g_omega


def vibrational_free_energy(omega_eV, g_omega, T, nat_per_fu=1):
    """
    Compute vibrational free energy per atom at temperature T.

    F_vib(T) = integral g(omega) * [hbar*omega/2 + kT*ln(1-exp(-hbar*omega/kT))] d_omega
               / N_atoms_per_fu

    Parameters
    ----------
    omega_eV : array
        Phonon frequencies in eV.
    g_omega : array
        Phonon DOS g(omega).
    T : float
        Temperature in Kelvin.
    nat_per_fu : int
        Number of atoms per formula unit.

    Returns
    -------
    F_vib : float
        Vibrational free energy per atom in eV.
    """
    d_omega = omega_eV[1] - omega_eV[0]

    # Zero-point energy contribution
    zpe = np.sum(g_omega * omega_eV / 2.0) * d_omega

    if T < 1e-6:
        return zpe

    beta = 1.0 / (kB_eV * T)

    # Thermal contribution
    x = omega_eV * beta
    # Avoid overflow for large x
    x_safe = np.clip(x, 0, 500)
    thermal = np.sum(g_omega * kB_eV * T * np.log(1.0 - np.exp(-x_safe))) * d_omega

    F_vib = zpe + thermal
    return F_vib


def vibrational_entropy(omega_eV, g_omega, T):
    """Compute vibrational entropy per atom."""
    if T < 1e-6:
        return 0.0

    d_omega = omega_eV[1] - omega_eV[0]
    beta = 1.0 / (kB_eV * T)
    x = omega_eV * beta
    x_safe = np.clip(x, 0, 500)

    bose = 1.0 / (np.exp(x_safe) - 1.0 + 1e-30)
    s_integrand = g_omega * (x_safe * bose - np.log(1.0 - np.exp(-x_safe) + 1e-30))
    S_vib = kB_eV * np.sum(s_integrand) * d_omega

    return S_vib


def vibrational_energy(omega_eV, g_omega, T):
    """Compute vibrational internal energy per atom."""
    d_omega = omega_eV[1] - omega_eV[0]

    zpe = np.sum(g_omega * omega_eV / 2.0) * d_omega

    if T < 1e-6:
        return zpe

    beta = 1.0 / (kB_eV * T)
    x = omega_eV * beta
    x_safe = np.clip(x, 0, 500)
    bose = 1.0 / (np.exp(x_safe) - 1.0 + 1e-30)

    E_thermal = np.sum(g_omega * omega_eV * bose) * d_omega
    return zpe + E_thermal

# =============================================================================
# Compute free energies
# =============================================================================

print("=" * 65)
print("  Finite-Temperature Phase Boundary from Phonon Free Energy")
print("=" * 65)

free_energies = {}
entropies = {}

for phase_name, phase_info in phases.items():
    print(f"\n--- Phase: {phase_info['label']} ---")
    print(f"  E_DFT = {phase_info['E_DFT']:.4f} eV/atom")

    dos_file = phase_info['phonon_dos_file']
    if not os.path.isfile(dos_file):
        print(f"  WARNING: {dos_file} not found. Using synthetic phonon DOS.")
        # Generate a simple Debye DOS for demonstration
        omega_max = 0.04  # eV (~320 cm^-1, typical for metals)
        omega = np.linspace(0.001, omega_max, 500)
        g = 3.0 * omega**2 / omega_max**3  # Debye DOS
    else:
        omega, g = load_phonon_dos(dos_file)
        print(f"  Loaded phonon DOS: {len(omega)} points, "
              f"max omega = {omega[-1]*1000:.1f} meV")

    F_T = []
    S_T = []
    for T in T_range:
        F_vib = vibrational_free_energy(omega, g, T, phase_info['nat_per_fu'])
        F_total = phase_info['E_DFT'] + F_vib
        F_T.append(F_total)
        S_T.append(vibrational_entropy(omega, g, T))

    free_energies[phase_name] = np.array(F_T)
    entropies[phase_name] = np.array(S_T)

    # Print at selected temperatures
    for T_sample in [0, 300, 500, 1000, 1500, 2000]:
        idx = np.argmin(np.abs(T_range - T_sample))
        if idx < len(F_T):
            print(f"  T = {T_range[idx]:6.0f} K: F = {F_T[idx]:.4f} eV/atom, "
                  f"S = {S_T[idx]*1000:.3f} meV/K/atom")

# =============================================================================
# Find phase boundary (crossing point)
# =============================================================================

phase_names = list(phases.keys())
if len(phase_names) >= 2:
    p1, p2 = phase_names[0], phase_names[1]
    delta_F = free_energies[p1] - free_energies[p2]

    print(f"\n--- Phase Boundary: {phases[p1]['label']} vs {phases[p2]['label']} ---")

    # Find sign changes in delta_F
    sign_changes = np.where(np.diff(np.sign(delta_F)))[0]

    transition_temps = []
    for idx in sign_changes:
        if idx + 1 < len(T_range):
            # Linear interpolation for crossing temperature
            T1, T2 = T_range[idx], T_range[idx + 1]
            dF1, dF2 = delta_F[idx], delta_F[idx + 1]
            T_cross = T1 - dF1 * (T2 - T1) / (dF2 - dF1)
            transition_temps.append(T_cross)
            print(f"  Phase transition at T = {T_cross:.0f} K")

    if not transition_temps:
        # Report which phase is more stable throughout
        if delta_F[0] < 0:
            print(f"  {phases[p1]['label']} more stable throughout 0-{T_range[-1]} K range")
        else:
            print(f"  {phases[p2]['label']} more stable throughout 0-{T_range[-1]} K range")
        print(f"  Delta F at 300 K = {delta_F[np.argmin(np.abs(T_range-300))]:.4f} eV/atom")

# =============================================================================
# Plot
# =============================================================================

fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))

# Free energy vs T
ax = axes[0]
colors = ['steelblue', 'darkorange', 'seagreen', 'crimson']
for i, (name, F) in enumerate(free_energies.items()):
    ax.plot(T_range, F, '-', color=colors[i % len(colors)], linewidth=2,
            label=phases[name]['label'])
for Tt in transition_temps:
    ax.axvline(Tt, color='gray', linestyle='--', alpha=0.7)
ax.set_xlabel('Temperature (K)', fontsize=13)
ax.set_ylabel('Free energy F(T) (eV/atom)', fontsize=13)
ax.set_title('Free Energy vs Temperature', fontsize=14)
ax.legend(fontsize=11)
ax.grid(True, alpha=0.3)

# Delta F vs T
ax = axes[1]
if len(phase_names) >= 2:
    ax.plot(T_range, (free_energies[phase_names[0]] - free_energies[phase_names[1]]) * 1000,
            '-', color='steelblue', linewidth=2)
    ax.axhline(0, color='black', linewidth=0.5)
    for Tt in transition_temps:
        ax.axvline(Tt, color='red', linestyle='--', alpha=0.7,
                   label=f'T_trans = {Tt:.0f} K')
    ax.set_xlabel('Temperature (K)', fontsize=13)
    ax.set_ylabel(f'$\\Delta F$ ({phases[phase_names[0]]["label"]} - '
                  f'{phases[phase_names[1]]["label"]}) (meV/atom)', fontsize=13)
    ax.set_title('Free Energy Difference', fontsize=14)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)

# Entropy vs T
ax = axes[2]
for i, (name, S) in enumerate(entropies.items()):
    ax.plot(T_range, np.array(S) * 1000, '-', color=colors[i % len(colors)],
            linewidth=2, label=phases[name]['label'])
ax.set_xlabel('Temperature (K)', fontsize=13)
ax.set_ylabel('Vibrational entropy (meV/K/atom)', fontsize=13)
ax.set_title('Vibrational Entropy', fontsize=14)
ax.legend(fontsize=11)
ax.grid(True, alpha=0.3)

fig.suptitle('Phonon-Based Phase Stability Analysis', fontsize=15)
fig.tight_layout()
fig.savefig(os.path.join(WORK_DIR, 'phase_boundary.png'),
            dpi=200, bbox_inches='tight')
plt.close()
print(f"\nSaved: {WORK_DIR}/phase_boundary.png")
```

### Method C: Quasi-Harmonic Approximation (QHA) for T-P Phase Diagram

QHA accounts for thermal expansion by computing phonon frequencies at multiple volumes and constructing F(V,T).

#### Step 1: Compute phonon frequencies at multiple volumes

```python
#!/usr/bin/env python3
"""
qha_setup.py
Generate QE input files for phonon calculations at multiple volumes.
This is the setup step for the quasi-harmonic approximation.

For each volume:
  1. Relax internal coordinates at fixed cell
  2. Compute phonon DOS
"""
import numpy as np
import os

from pymatgen.core import Structure

STRUCTURE_FILE = "relaxed_structure.cif"
PSEUDO_DIR = "./pseudo/"
PREFIX = "mgo"

# Volume scaling factors (relative to equilibrium)
V_SCALES = np.arange(0.94, 1.071, 0.01)  # 0.94 to 1.07, step 0.01

WORK_DIR = "/tmp/qha"
os.makedirs(WORK_DIR, exist_ok=True)

struct = Structure.from_file(STRUCTURE_FILE)
print(f"Reference structure: {struct.composition}")
print(f"Reference volume: {struct.volume:.3f} A^3")
print(f"Number of volume points: {len(V_SCALES)}")

for v_scale in V_SCALES:
    vol_dir = os.path.join(WORK_DIR, f"vol_{v_scale:.3f}")
    os.makedirs(vol_dir, exist_ok=True)

    # Scale structure
    scaled = struct.copy()
    linear_scale = v_scale ** (1.0 / 3.0)
    scaled.scale_lattice(struct.volume * v_scale)

    # Write QE SCF input (relax internal coordinates only)
    scf_in = os.path.join(vol_dir, 'scf.in')
    with open(scf_in, 'w') as f:
        f.write("&CONTROL\n")
        f.write("    calculation = 'relax'\n")
        f.write(f"    prefix = '{PREFIX}'\n")
        f.write(f"    pseudo_dir = '{os.path.abspath(PSEUDO_DIR)}'\n")
        f.write("    outdir = './tmp/'\n")
        f.write("    tprnfor = .true.\n")
        f.write("    tstress = .true.\n")
        f.write("/\n")
        f.write("&SYSTEM\n")
        f.write(f"    ibrav = 0\n")
        f.write(f"    nat = {len(scaled)}\n")
        f.write(f"    ntyp = {len(set(s.symbol for s in scaled.species))}\n")
        f.write("    ecutwfc = 80.0\n")
        f.write("    ecutrho = 640.0\n")
        f.write("/\n")
        f.write("&ELECTRONS\n")
        f.write("    conv_thr = 1.0d-10\n")
        f.write("/\n")
        f.write("&IONS\n")
        f.write("/\n\n")

        f.write("CELL_PARAMETERS {angstrom}\n")
        for vec in scaled.lattice.matrix:
            f.write(f"  {vec[0]:16.10f} {vec[1]:16.10f} {vec[2]:16.10f}\n")
        f.write("\n")

        unique_species = list(dict.fromkeys(s.symbol for s in scaled.species))
        f.write("ATOMIC_SPECIES\n")
        for sp in unique_species:
            f.write(f"  {sp}  1.0  {sp}.UPF\n")
        f.write("\n")

        f.write("ATOMIC_POSITIONS {crystal}\n")
        for site in scaled:
            fc = site.frac_coords
            f.write(f"  {site.specie.symbol} {fc[0]:.10f} {fc[1]:.10f} {fc[2]:.10f}\n")
        f.write("\n")

        f.write("K_POINTS {automatic}\n")
        f.write("  8 8 8  0 0 0\n")

    # Write phonon input
    ph_in = os.path.join(vol_dir, 'ph.in')
    with open(ph_in, 'w') as f:
        f.write(f"Phonons for V/V0 = {v_scale:.3f}\n")
        f.write("&INPUTPH\n")
        f.write(f"    prefix = '{PREFIX}'\n")
        f.write("    outdir = './tmp/'\n")
        f.write("    fildyn = 'dyn'\n")
        f.write("    tr2_ph = 1.0d-14\n")
        f.write("    ldisp = .true.\n")
        f.write("    nq1 = 4, nq2 = 4, nq3 = 4\n")
        f.write("/\n")

    vol = scaled.volume
    print(f"V/V0 = {v_scale:.3f}: V = {vol:.3f} A^3, dir = {vol_dir}")

print(f"\nGenerated {len(V_SCALES)} directories.")
print("Run SCF -> ph.x -> q2r.x -> matdyn.x (phonon DOS) in each directory.")
```

#### Step 2: Compute QHA free energy and P-T phase diagram

```python
#!/usr/bin/env python3
"""
qha_analysis.py
Quasi-harmonic approximation analysis:
  1. Collect E(V) and phonon DOS at each volume
  2. Compute F(V,T) = E(V) + F_vib(V,T)
  3. Minimize F(V,T) w.r.t. V to get V(T) -- thermal expansion
  4. Compute G(P,T) = F(V,T) + PV for pressure-dependent stability
  5. Construct T-P phase boundary between phases
"""
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.optimize import minimize_scalar
from scipy.interpolate import interp1d
import os
import glob

# Physical constants
kB_eV = 8.617333e-5      # eV/K
eV_per_A3_to_GPa = 160.2176634  # 1 eV/A^3 = 160.2 GPa
cm_to_eV = 1.23984e-4

WORK_DIR = "/tmp/qha"
os.makedirs(WORK_DIR, exist_ok=True)

# =============================================================================
# 1. Collect E(V) and phonon DOS from calculations
# =============================================================================

def collect_qha_data(base_dir, vol_pattern="vol_*"):
    """
    Collect DFT energies and phonon DOS from volume directories.

    Returns
    -------
    volumes : array
        Volumes in A^3.
    energies : array
        DFT total energies in eV.
    phonon_dos_list : list of (omega_eV, g_omega) tuples.
    """
    vol_dirs = sorted(glob.glob(os.path.join(base_dir, vol_pattern)))

    volumes = []
    energies = []
    phonon_dos_list = []

    for vd in vol_dirs:
        # Parse volume from directory name
        v_scale = float(os.path.basename(vd).replace("vol_", ""))

        # Read total energy from SCF output
        scf_out = os.path.join(vd, 'scf.out')
        if not os.path.isfile(scf_out):
            continue

        E = None
        with open(scf_out, 'r') as f:
            for line in f:
                if '!' in line and 'total energy' in line:
                    E = float(line.split('=')[1].split('Ry')[0]) * 13.6057  # Ry to eV
        if E is None:
            continue

        # Read volume from output
        V = None
        with open(scf_out, 'r') as f:
            for line in f:
                if 'unit-cell volume' in line:
                    V = float(line.split('=')[1].split('(')[0])  # a.u.^3
                    V *= 0.14818471  # a.u.^3 to A^3

        if V is None:
            continue

        # Read phonon DOS
        dos_file = os.path.join(vd, 'phonon_dos.dat')
        if os.path.isfile(dos_file):
            data = np.loadtxt(dos_file, comments='#')
            omega_cm = data[:, 0]
            g = data[:, 1]
            mask = omega_cm > 0.1
            omega_eV = omega_cm[mask] * cm_to_eV
            g = g[mask]
        else:
            # Use Debye model as fallback
            omega_eV = np.linspace(0.001, 0.05, 200)
            g = omega_eV**2

        volumes.append(V)
        energies.append(E)
        phonon_dos_list.append((omega_eV, g))

    return np.array(volumes), np.array(energies), phonon_dos_list


def f_vib(omega_eV, g_omega, T):
    """Vibrational free energy from phonon DOS at temperature T."""
    dw = omega_eV[1] - omega_eV[0]
    zpe = np.sum(g_omega * omega_eV / 2.0) * dw
    if T < 1e-6:
        return zpe
    x = omega_eV / (kB_eV * T)
    x_safe = np.clip(x, 0, 500)
    thermal = np.sum(g_omega * kB_eV * T * np.log(1.0 - np.exp(-x_safe))) * dw
    return zpe + thermal


# =============================================================================
# 2. Compute F(V,T) and find equilibrium volume V(T)
# =============================================================================

def compute_free_energy_surface(volumes, energies, phonon_dos_list, T_range):
    """
    Compute F(V,T) at each (V,T) point.

    Returns
    -------
    F_VT : array, shape (n_V, n_T)
    V_eq : array, shape (n_T,) -- equilibrium volume at each T
    F_eq : array, shape (n_T,) -- minimum F at each T
    """
    n_V = len(volumes)
    n_T = len(T_range)
    F_VT = np.zeros((n_V, n_T))

    for iv in range(n_V):
        omega_eV, g = phonon_dos_list[iv]
        for it, T in enumerate(T_range):
            F_VT[iv, it] = energies[iv] + f_vib(omega_eV, g, T)

    # Find equilibrium volume at each T by fitting polynomial
    V_eq = np.zeros(n_T)
    F_eq = np.zeros(n_T)

    for it in range(n_T):
        # Fit 3rd-order polynomial to F(V) at this T
        coeffs = np.polyfit(volumes, F_VT[:, it], 3)
        # Find minimum
        dcoeffs = np.polyder(coeffs)
        roots = np.roots(dcoeffs)
        real_roots = roots[np.isreal(roots)].real
        valid = real_roots[(real_roots >= volumes.min()) &
                           (real_roots <= volumes.max())]
        if len(valid) > 0:
            F_at_roots = np.polyval(coeffs, valid)
            V_eq[it] = valid[np.argmin(F_at_roots)]
            F_eq[it] = np.min(F_at_roots)
        else:
            V_eq[it] = volumes[np.argmin(F_VT[:, it])]
            F_eq[it] = np.min(F_VT[:, it])

    return F_VT, V_eq, F_eq


def compute_gibbs_energy(volumes, F_VT, T_range, pressures_GPa):
    """
    Compute Gibbs free energy G(P,T) = min_V [F(V,T) + PV].

    Returns
    -------
    G_PT : array, shape (n_P, n_T)
    """
    n_T = len(T_range)
    n_P = len(pressures_GPa)
    G_PT = np.zeros((n_P, n_T))

    for ip, P in enumerate(pressures_GPa):
        P_eV_A3 = P / eV_per_A3_to_GPa
        for it in range(n_T):
            enthalpy = F_VT[:, it] + P_eV_A3 * volumes
            G_PT[ip, it] = np.min(enthalpy)

    return G_PT


# =============================================================================
# 3. Main workflow
# =============================================================================

print("=" * 65)
print("  Quasi-Harmonic Approximation Phase Diagram")
print("=" * 65)

T_range = np.arange(0, 2001, 25)
pressures = np.arange(0, 101, 5)  # 0 to 100 GPa

# For two competing phases, run the analysis on each phase directory
# Here we demonstrate with one phase and use synthetic data for the second

# Phase 1
print("\n--- Collecting data for Phase 1 ---")
volumes_1, energies_1, dos_list_1 = collect_qha_data(WORK_DIR, "vol_*")

if len(volumes_1) < 3:
    print("  Not enough volume points found. Using synthetic data for demonstration.")
    # Synthetic Murnaghan EOS data
    V0 = 19.0  # A^3/atom
    E0 = -8.50  # eV/atom
    B0 = 150.0  # GPa
    Bp = 4.0

    volumes_1 = np.linspace(0.92 * V0, 1.08 * V0, 14)
    B0_eV = B0 / eV_per_A3_to_GPa
    energies_1 = E0 + (B0_eV * volumes_1 / Bp) * (
        ((V0/volumes_1)**Bp) / (Bp - 1) + 1
    ) - B0_eV * V0 / (Bp - 1)

    dos_list_1 = []
    for V in volumes_1:
        omega_max = 0.035 * (V0 / V) ** (2.0/3.0)  # Gruneisen-scaled
        omega = np.linspace(0.001, omega_max, 300)
        g = 3.0 * omega**2 / omega_max**3
        dos_list_1.append((omega, g))

# Phase 2 (synthetic, slightly different)
V0_2 = 18.5
E0_2 = -8.48  # Slightly less stable at T=0
B0_2 = 170.0

volumes_2 = np.linspace(0.92 * V0_2, 1.08 * V0_2, 14)
B0_eV_2 = B0_2 / eV_per_A3_to_GPa
energies_2 = E0_2 + (B0_eV_2 * volumes_2 / Bp) * (
    ((V0_2/volumes_2)**Bp) / (Bp - 1) + 1
) - B0_eV_2 * V0_2 / (Bp - 1)

dos_list_2 = []
for V in volumes_2:
    omega_max = 0.040 * (V0_2 / V) ** (2.0/3.0)
    omega = np.linspace(0.001, omega_max, 300)
    g = 3.0 * omega**2 / omega_max**3
    dos_list_2.append((omega, g))

# Compute free energy surfaces
print("\nComputing F(V,T) surfaces...")
F_VT_1, V_eq_1, F_eq_1 = compute_free_energy_surface(
    volumes_1, energies_1, dos_list_1, T_range)
F_VT_2, V_eq_2, F_eq_2 = compute_free_energy_surface(
    volumes_2, energies_2, dos_list_2, T_range)

# Compute Gibbs free energies
print("Computing G(P,T)...")
G_PT_1 = compute_gibbs_energy(volumes_1, F_VT_1, T_range, pressures)
G_PT_2 = compute_gibbs_energy(volumes_2, F_VT_2, T_range, pressures)

# Find phase boundary: G_1(P,T) = G_2(P,T)
delta_G = G_PT_1 - G_PT_2  # Shape: (n_P, n_T)

# Extract phase boundary
boundary_P = []
boundary_T = []

for ip in range(len(pressures)):
    dG = delta_G[ip, :]
    sign_changes = np.where(np.diff(np.sign(dG)))[0]
    for idx in sign_changes:
        T1, T2 = T_range[idx], T_range[idx + 1]
        dG1, dG2 = dG[idx], dG[idx + 1]
        T_cross = T1 - dG1 * (T2 - T1) / (dG2 - dG1)
        boundary_P.append(pressures[ip])
        boundary_T.append(T_cross)

if boundary_P:
    print(f"\nPhase boundary found ({len(boundary_P)} points)")
    for P, T in zip(boundary_P, boundary_T):
        print(f"  P = {P:6.1f} GPa, T = {T:6.0f} K")

# =============================================================================
# 4. Plot results
# =============================================================================

fig, axes = plt.subplots(2, 2, figsize=(14, 12))

# Thermal expansion V(T)
ax = axes[0, 0]
ax.plot(T_range, V_eq_1, '-', color='steelblue', linewidth=2, label='Phase 1')
ax.plot(T_range, V_eq_2, '--', color='darkorange', linewidth=2, label='Phase 2')
ax.set_xlabel('Temperature (K)', fontsize=12)
ax.set_ylabel('Equilibrium volume (A^3/atom)', fontsize=12)
ax.set_title('Thermal Expansion (QHA)', fontsize=13)
ax.legend(fontsize=11)
ax.grid(True, alpha=0.3)

# F(T) at P=0
ax = axes[0, 1]
ax.plot(T_range, F_eq_1, '-', color='steelblue', linewidth=2, label='Phase 1')
ax.plot(T_range, F_eq_2, '--', color='darkorange', linewidth=2, label='Phase 2')
ax.set_xlabel('Temperature (K)', fontsize=12)
ax.set_ylabel('Helmholtz free energy (eV/atom)', fontsize=12)
ax.set_title('F(T) at P = 0 GPa', fontsize=13)
ax.legend(fontsize=11)
ax.grid(True, alpha=0.3)

# G(T) at selected pressures
ax = axes[1, 0]
P_samples = [0, 20, 50, 100]
for P_val in P_samples:
    ip = np.argmin(np.abs(pressures - P_val))
    dG = (G_PT_1[ip, :] - G_PT_2[ip, :]) * 1000  # meV
    ax.plot(T_range, dG, linewidth=1.5, label=f'P = {pressures[ip]} GPa')
ax.axhline(0, color='black', linewidth=0.5)
ax.set_xlabel('Temperature (K)', fontsize=12)
ax.set_ylabel('$\\Delta G$ (Phase 1 - Phase 2) (meV/atom)', fontsize=12)
ax.set_title('Gibbs Free Energy Difference', fontsize=13)
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3)

# T-P phase diagram
ax = axes[1, 1]
if boundary_P:
    ax.plot(boundary_T, boundary_P, 'o-', color='crimson', linewidth=2,
            markersize=5)
    # Fill regions
    T_bound = np.array(boundary_T)
    P_bound = np.array(boundary_P)
    # Phase labels
    ax.text(np.mean(T_bound) * 0.3, np.max(P_bound) * 0.7,
            'Phase 1', fontsize=14, fontweight='bold', color='steelblue')
    ax.text(np.mean(T_bound) * 1.3, np.max(P_bound) * 0.3,
            'Phase 2', fontsize=14, fontweight='bold', color='darkorange')
ax.set_xlabel('Temperature (K)', fontsize=12)
ax.set_ylabel('Pressure (GPa)', fontsize=12)
ax.set_title('T-P Phase Diagram', fontsize=13)
ax.grid(True, alpha=0.3)
ax.set_xlim(left=0)
ax.set_ylim(bottom=0)

fig.suptitle('Quasi-Harmonic Phase Diagram Analysis', fontsize=15)
fig.tight_layout()
fig.savefig(os.path.join(WORK_DIR, 'qha_phase_diagram.png'),
            dpi=200, bbox_inches='tight')
plt.close()
print(f"\nSaved: {WORK_DIR}/qha_phase_diagram.png")

# Save phase boundary data
if boundary_P:
    np.savetxt(
        os.path.join(WORK_DIR, 'phase_boundary_TP.dat'),
        np.column_stack([boundary_T, boundary_P]),
        header="T(K)  P(GPa)",
        fmt="%8.1f %8.2f"
    )
    print(f"Saved: {WORK_DIR}/phase_boundary_TP.dat")
```

### Method D: Quick Phonon Free Energy via ASE + MACE

For rapid screening without DFPT, compute phonon frequencies using ASE finite displacements with MACE:

```python
#!/usr/bin/env python3
"""
quick_phonon_free_energy.py
Compute phonon free energy using ASE + MACE finite displacements.
Much faster than QE DFPT, suitable for screening.
"""
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os

from ase.io import read
from ase.optimize import LBFGS
from ase.constraints import ExpCellFilter
from ase.phonons import Phonons
from mace.calculators import mace_mp

import warnings
warnings.filterwarnings("ignore")

STRUCTURE_FILE = "structure.cif"
MACE_MODEL = "medium"
DEVICE = "cpu"
SUPERCELL = (2, 2, 2)
DELTA = 0.01  # Displacement in Angstrom

WORK_DIR = "/tmp/quick_phonon_fe"
os.makedirs(WORK_DIR, exist_ok=True)

calc = mace_mp(model=MACE_MODEL, device=DEVICE, default_dtype="float64")

# Relax
atoms = read(STRUCTURE_FILE)
atoms.calc = calc
ecf = ExpCellFilter(atoms)
opt = LBFGS(ecf, logfile=os.path.join(WORK_DIR, "relax.log"))
opt.run(fmax=0.01, steps=200)
print(f"Relaxed: {atoms.get_chemical_formula()}, E = {atoms.get_potential_energy():.4f} eV")
E_DFT = atoms.get_potential_energy() / len(atoms)

# Phonon calculation
ph = Phonons(atoms, calc, supercell=SUPERCELL, delta=DELTA,
             name=os.path.join(WORK_DIR, 'phonon'))
ph.run()
ph.read(acoustic=True)

# Get phonon DOS
omega_points, dos = ph.dos(kpts=(20, 20, 20), npts=500, delta=3e-4)
# omega_points in eV, dos in states/eV

# Compute free energy
kB = 8.617333e-5  # eV/K
temperatures = np.arange(0, 2001, 25)
F_total = []
S_vib = []

for T in temperatures:
    if T < 1e-6:
        # Zero-point energy only
        mask = omega_points > 1e-6
        zpe = np.trapz(dos[mask] * omega_points[mask] / 2.0, omega_points[mask])
        F_total.append(E_DFT + zpe / len(atoms))
        S_vib.append(0.0)
    else:
        mask = omega_points > 1e-6
        w = omega_points[mask]
        g = dos[mask]
        x = w / (kB * T)
        x_safe = np.clip(x, 0, 500)

        zpe = np.trapz(g * w / 2.0, w)
        thermal = np.trapz(g * kB * T * np.log(1 - np.exp(-x_safe)), w)
        F_vib = (zpe + thermal) / len(atoms)
        F_total.append(E_DFT + F_vib)

        # Entropy
        bose = 1.0 / (np.exp(x_safe) - 1.0 + 1e-30)
        s_int = np.trapz(g * (x_safe * bose - np.log(1 - np.exp(-x_safe) + 1e-30)), w)
        S_vib.append(kB * s_int / len(atoms))

F_total = np.array(F_total)
S_vib = np.array(S_vib)

# Save and plot
np.savetxt(os.path.join(WORK_DIR, 'free_energy_vs_T.dat'),
           np.column_stack([temperatures, F_total, S_vib]),
           header="T(K)  F(eV/atom)  S(eV/K/atom)", fmt="%8.1f %14.6f %14.8f")

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

ax1.plot(temperatures, F_total, '-', color='steelblue', linewidth=2)
ax1.set_xlabel('Temperature (K)', fontsize=12)
ax1.set_ylabel('Free energy (eV/atom)', fontsize=12)
ax1.set_title(f'Free Energy: {atoms.get_chemical_formula()}', fontsize=13)
ax1.grid(True, alpha=0.3)

ax2.plot(omega_points * 1000, dos, '-', color='steelblue', linewidth=1)
ax2.set_xlabel('Frequency (meV)', fontsize=12)
ax2.set_ylabel('Phonon DOS', fontsize=12)
ax2.set_title('Phonon DOS (ASE + MACE)', fontsize=13)
ax2.set_xlim(left=0)
ax2.grid(True, alpha=0.3)

fig.tight_layout()
fig.savefig(os.path.join(WORK_DIR, 'quick_phonon_fe.png'), dpi=200, bbox_inches='tight')
plt.close()
print(f"\nSaved: {WORK_DIR}/quick_phonon_fe.png")
print(f"Data: {WORK_DIR}/free_energy_vs_T.dat")
```

## Key Parameters

| Parameter | Typical Value | Notes |
|---|---|---|
| Number of volume points (QHA) | 10-15 | Cover V/V0 = 0.94-1.06 for moderate pressures |
| Volume range | +/-6% of equilibrium | Extend to +/-10% for high-pressure studies |
| q-grid for phonon DOS | 4x4x4 (DFPT), 20x20x20 (matdyn.x) | matdyn.x q-grid is for DOS interpolation, not DFPT |
| Temperature range | 0-2000 K (metals), 0-1000 K (semiconductors) | Stay below melting point; QHA breaks down near Tm |
| Temperature step | 10-50 K | Finer near expected transition |
| Pressure range | 0-100 GPa (most applications) | Material-dependent |
| ecutwfc | 40-80 Ry | Must be the same across all phases being compared |
| k-grid | Converged for total energy to < 1 meV/atom | Critical for energy differences between phases |
| Pseudopotentials | Same set for all calculations | Mixing PP sets invalidates energy comparisons |

## Interpreting Results

### Convex Hull

- Phases on the hull are thermodynamically stable at T=0 K.
- Phases above the hull are metastable. Distance from hull (in eV/atom) indicates the thermodynamic driving force for decomposition.
- Energy above hull < 25 meV/atom: potentially synthesizable (metastable but kinetically accessible).
- Energy above hull > 100 meV/atom: unlikely to be observed experimentally.

### Phase Boundaries

- Phase boundary is the T (or P) where free energies of two phases cross.
- Entropy drives transitions: the higher-entropy phase is favored at high T.
- Volume drives pressure transitions: the smaller-volume phase is favored at high P.
- Clausius-Clapeyron slope: dP/dT = Delta_S / Delta_V. Positive slope means the high-T phase has larger volume.

### QHA Validity

- QHA is valid when anharmonic effects are small (T << T_melting, no strongly anharmonic modes).
- Rule of thumb: QHA is reliable up to about 2/3 of the melting temperature.
- Near phase transitions or melting, anharmonic corrections or MD simulations are needed.
- Soft modes (imaginary frequencies) break QHA. If a phase has imaginary phonons, it is dynamically unstable at that volume.

### Common Pitfalls in Phase Diagram Construction

- **Different reference states**: Always use the same pseudopotentials, ecutwfc, and k-grid convergence level for all phases.
- **Missing competing phases**: Check Materials Project for all known phases in the system. Missing a stable phase gives incorrect hull.
- **Configurational entropy**: For alloys and solid solutions, vibrational entropy alone is insufficient. Configurational entropy (from cluster expansion + Monte Carlo) is needed.
- **Magnetic entropy**: For magnetic materials near the Curie temperature, magnetic disorder entropy contributes to phase stability.

## Common Issues

| Problem | Symptom | Solution |
|---|---|---|
| Imaginary phonons at some volumes | Negative frequencies in phonon DOS | That phase is dynamically unstable at that volume. Exclude those volumes or use a different phase. |
| QHA thermal expansion diverges | V(T) shoots up at high T | QHA breaks down near melting. Truncate analysis below ~2/3 Tm. |
| Energy differences too small | Phase boundary uncertain | Tighten convergence: ecutwfc, k-grid, conv_thr. Differences < 1 meV/atom require very careful convergence. |
| MP API returns no entries | Empty results from MPRester | Check chemical system syntax ("Ti-O" not "TiO"). Verify API key. Check MP server status. |
| Phonon DOS normalization | Free energies inconsistent between phases | Ensure phonon DOS integrates to 3*N_atoms. Use consistent frequency grid and broadening. |
| Phase boundary not found | No crossing in F(T) | Phases may not cross in the studied T range. Extend range. Or phases differ by too much (>50 meV) and transition occurs only at very high T. |
| Wrong phase predicted stable | T=0 hull disagrees with experiment | Check DFT functional (PBE vs. PBEsol vs. SCAN). Some phases require better treatment of correlation (DFT+U for oxides, van der Waals for layered materials). |
| Pressure calculation incorrect | Phase boundary shifted | Verify volume-energy data fits well to EOS. Use enough volume points. Check unit conversions (eV/A^3 to GPa). |
