# Gruneisen Parameters and Quasi-Harmonic Approximation (QHA)

## When to Use

- Computing thermal expansion coefficient as a function of temperature
- Obtaining temperature-dependent bulk modulus
- Calculating mode Gruneisen parameters (how phonon frequencies shift with volume)
- Going beyond the harmonic approximation without full anharmonic (AIMD) calculations
- Estimating thermal equation of state P(V,T)

## Method Selection

```
Need thermal expansion or T-dependent bulk modulus?
  YES --> QHA (this skill)
    Is MACE accurate for your system?
      YES --> Method A: ASE + MACE + phonopy (fast, minutes to hours)
      NO  --> Method A with QE forces, or full QE DFPT at each volume

Need mode-resolved anharmonicity info?
  YES --> Gruneisen parameters (included in this skill)

Need intrinsic thermal conductivity (3-phonon scattering)?
  --> Use phono3py (separate workflow, not covered here)
```

The QHA assumes phonons are harmonic at each volume but allows the equilibrium volume to change with temperature. This captures most of the thermal expansion in solids below the Debye temperature. It fails near melting or for strongly anharmonic systems (e.g., PbTe, SnSe).

## Prerequisites

```bash
pip install phonopy seekpath
```

Pre-installed: `ase`, `mace-torch`, `pymatgen`, `numpy`, `scipy`, `matplotlib`.

## Detailed Steps

### Method A: ASE + MACE + phonopy (QHA)

The workflow:
1. Relax the structure at ground state volume V0
2. Generate a set of strained volumes: V0 * (1 + epsilon) for epsilon in [-0.02, ..., +0.02]
3. At each volume, run a full phonon calculation (displacements + forces + force constants)
4. Collect free energies F(V,T) from phonon thermodynamics at each volume
5. For each temperature, fit E(V) + F_phonon(V,T) to an equation of state to find V(T)
6. Extract thermal expansion, T-dependent bulk modulus, and Gruneisen parameters

```python
#!/usr/bin/env python3
"""
Quasi-Harmonic Approximation (QHA) using ASE + MACE + phonopy.
Computes thermal expansion, T-dependent bulk modulus, and Gruneisen parameters.
Complete runnable script.
"""

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from copy import deepcopy

from ase.io import read, write
from ase.optimize import LBFGS
from ase.constraints import ExpCellFilter

from mace.calculators import mace_mp

from pymatgen.core.structure import Structure
from pymatgen.io.ase import AseAtomsAdaptor
from pymatgen.io.phonopy import get_phonopy_structure

import phonopy
from phonopy import PhonopyQHA

import warnings
warnings.filterwarnings("ignore")

# ============================================================
# 1. CONFIGURATION
# ============================================================

STRUCTURE_FILE = "structure.cif"       # Input structure
MACE_MODEL = "medium"                  # "small", "medium", "large"
DEVICE = "cpu"                         # "cpu" or "cuda"

# Volume strain range: V0 * (1 + strain)
# Typical: -2% to +2% in 5-7 points (minimum 5 for good EOS fit)
STRAINS = np.array([-0.02, -0.01, -0.005, 0.0, 0.005, 0.01, 0.02])

# Phonon settings (same as phonon/ skill)
MIN_LENGTH = 15.0                      # Supercell min length (A) - can use smaller for QHA screening
DISPLACEMENT = 0.01                    # Displacement distance (A)
SYMPREC = 1e-5                         # Symmetry precision
FMAX = 1e-4                            # Relaxation force convergence (eV/A)
MESH = [15, 15, 15]                    # q-mesh for phonon DOS

# Temperature range for QHA
T_MIN = 0
T_MAX = 800
T_STEP = 5

WORK_DIR = "/tmp/qha_calc"
os.makedirs(WORK_DIR, exist_ok=True)

# ============================================================
# 2. SET UP CALCULATOR AND RELAX GROUND STATE
# ============================================================

calc = mace_mp(model=MACE_MODEL, device=DEVICE, default_dtype="float64")

atoms_orig = read(STRUCTURE_FILE)
atoms_orig.calc = calc

print("=== Ground State Relaxation ===")
print(f"  Formula: {atoms_orig.get_chemical_formula()}")

ecf = ExpCellFilter(atoms_orig, hydrostatic_strain=False)
opt = LBFGS(ecf, logfile=os.path.join(WORK_DIR, "relax.log"))
opt.run(fmax=FMAX, steps=500)

V0 = atoms_orig.get_volume()
E0 = atoms_orig.get_potential_energy()
print(f"  Equilibrium volume: {V0:.4f} A^3")
print(f"  Equilibrium energy: {E0:.6f} eV")
write(os.path.join(WORK_DIR, "relaxed.cif"), atoms_orig)

# ============================================================
# 3. DETERMINE SUPERCELL MATRIX
# ============================================================

def get_supercell_matrix(atoms, min_length=15.0):
    cell_lengths = atoms.cell.lengths()
    multiples = np.ceil(min_length / cell_lengths).astype(int)
    multiples = np.maximum(multiples, 1)
    return np.diag(multiples)

supercell_matrix = get_supercell_matrix(atoms_orig, min_length=MIN_LENGTH)
print(f"  Supercell matrix: diag({np.diag(supercell_matrix)})")

# ============================================================
# 4. PHONON CALCULATIONS AT EACH VOLUME
# ============================================================

print(f"\n=== Phonon Calculations at {len(STRAINS)} Volumes ===")

adaptor = AseAtomsAdaptor()
volumes = []
electronic_energies = []
free_energies_all = []      # List of F(T) arrays, one per volume
entropy_all = []
cv_all = []
temperatures = None

for i_strain, strain in enumerate(STRAINS):
    print(f"\n--- Volume point {i_strain + 1}/{len(STRAINS)}: strain = {strain:+.3f} ---")

    # Apply isotropic strain to the relaxed structure
    atoms_strained = atoms_orig.copy()
    atoms_strained.calc = calc

    # Scale cell isotropically: V' = V0 * (1 + strain)
    # Linear scale factor: s = (1 + strain)^(1/3)
    scale = (1.0 + strain) ** (1.0 / 3.0)
    atoms_strained.set_cell(atoms_orig.cell * scale, scale_atoms=True)

    V = atoms_strained.get_volume()
    E = atoms_strained.get_potential_energy()
    volumes.append(V)
    electronic_energies.append(E)
    print(f"  V = {V:.4f} A^3 ({V/V0*100:.1f}% of V0), E = {E:.6f} eV")

    # Convert to phonopy structure
    pmg_struct = adaptor.get_structure(atoms_strained)
    phonopy_struct = get_phonopy_structure(pmg_struct)

    # Set up phonopy
    phonon = phonopy.Phonopy(
        phonopy_struct,
        supercell_matrix=supercell_matrix.tolist(),
        symprec=SYMPREC,
    )
    phonon.generate_displacements(distance=DISPLACEMENT)
    supercells = phonon.supercells_with_displacements
    n_disp = len(supercells)
    print(f"  Displacements: {n_disp}")

    # Compute forces
    forces_list = []
    for j, sc in enumerate(supercells):
        sc_atoms = adaptor.get_atoms(
            Structure(
                lattice=sc.cell,
                species=sc.symbols,
                coords=sc.scaled_positions,
            )
        )
        sc_atoms.calc = calc
        forces = sc_atoms.get_forces()
        forces_list.append(forces)

    phonon.forces = forces_list
    phonon.produce_force_constants()

    # Check for imaginary modes
    phonon.run_mesh(MESH, with_eigenvectors=False, is_gamma_center=True)
    mesh_dict = phonon.get_mesh_dict()
    min_freq = mesh_dict["frequencies"].min()
    if min_freq < -0.5:
        print(f"  WARNING: Imaginary modes detected (min freq: {min_freq:.3f} THz)")
        print(f"  QHA results may be unreliable for this volume point.")

    # Get thermal properties
    phonon.run_thermal_properties(t_min=T_MIN, t_max=T_MAX, t_step=T_STEP)
    tp = phonon.get_thermal_properties_dict()

    if temperatures is None:
        temperatures = tp["temperatures"]

    # phonopy returns free_energy in kJ/mol, entropy in J/K/mol, Cv in J/K/mol
    free_energies_all.append(tp["free_energy"])
    entropy_all.append(tp["entropy"])
    cv_all.append(tp["heat_capacity"])

    # Save individual phonon calc
    phonon.save(os.path.join(WORK_DIR, f"phonopy_strain_{i_strain:02d}.yaml"))

volumes = np.array(volumes)
electronic_energies = np.array(electronic_energies)
free_energies_all = np.array(free_energies_all)     # (n_volumes, n_temps)
entropy_all = np.array(entropy_all)
cv_all = np.array(cv_all)

# Number of formula units in the cell
n_atoms = len(atoms_orig)

print(f"\n=== All phonon calculations complete ===")
print(f"  Volumes: {volumes}")
print(f"  Energies: {electronic_energies}")
print(f"  Temperature points: {len(temperatures)}")

# ============================================================
# 5. RUN QHA USING PHONOPY's PhonopyQHA
# ============================================================

print("\n=== Quasi-Harmonic Approximation ===")

# PhonopyQHA expects:
# - volumes in A^3
# - electronic_energies in eV
# - free_energies (phonon Helmholtz) in kJ/mol per unit cell
# - cv in J/K/mol
# - entropy in J/K/mol
# - temperatures in K

# Convert free energies from kJ/mol to eV:
# 1 kJ/mol = 1000 J / (6.022e23) = 1.6605e-21 J = 0.010364 eV
kj_mol_to_ev = 0.010364

# PhonopyQHA wants free energies in eV (matching electronic_energies units)
fe_ev = free_energies_all * kj_mol_to_ev  # (n_volumes, n_temps)

# Similarly entropy J/K/mol -> eV/K
j_kmol_to_ev_k = kj_mol_to_ev / 1000.0
entropy_ev = entropy_all * j_kmol_to_ev_k

# Cv: J/K/mol -> eV/K
cv_ev = cv_all * j_kmol_to_ev_k

try:
    qha = PhonopyQHA(
        volumes=volumes,
        electronic_energies=electronic_energies,
        temperatures=temperatures,
        free_energy=fe_ev,
        cv=cv_ev,
        entropy=entropy_ev,
        eos="vinet",  # Vinet EOS; alternatives: "birch_murnaghan", "murnaghan"
        t_max=T_MAX,
    )

    print("  QHA fit successful!")

    # ============================================================
    # 6. EXTRACT QHA RESULTS
    # ============================================================

    # Thermal expansion
    qha_temps = qha.temperatures
    thermal_expansion = qha.thermal_expansion  # 1/K

    # Volume vs T
    volume_temperature = qha.volume_temperature  # A^3

    # Bulk modulus vs T
    bulk_modulus_temperature = qha.bulk_modulus_temperature  # GPa

    # Helmholtz free energy vs T
    helmholtz_volume = qha.helmholtz_volume  # eV at each (V, T)

    # ============================================================
    # 7. COMPUTE GRUNEISEN PARAMETER
    # ============================================================

    # Thermodynamic Gruneisen parameter: gamma = V * alpha * B_T / Cv
    # where alpha = thermal expansion, B_T = isothermal bulk modulus, Cv = heat capacity

    # Get Cv at equilibrium volume for each T (interpolate from QHA)
    # Use the middle volume point's Cv as approximation
    mid_idx = len(STRAINS) // 2
    cv_equilibrium = cv_all[mid_idx]  # J/K/mol

    # Convert Cv from J/K/mol to eV/K (per unit cell)
    cv_ev_cell = cv_equilibrium * j_kmol_to_ev_k

    # Compute Gruneisen parameter where we have all quantities
    n_T = min(len(qha_temps), len(thermal_expansion), len(bulk_modulus_temperature), len(cv_ev_cell))
    gruneisen = np.zeros(n_T)

    for i in range(n_T):
        if cv_ev_cell[i] > 1e-15 and i < len(volume_temperature):
            # gamma = V * alpha * B / Cv
            # V in A^3, alpha in 1/K, B in GPa, Cv in eV/K
            # Need consistent units: V*alpha*B should give energy/K units matching Cv
            # V(A^3) * B(GPa) = V * B * 1e-21 * 1e9 = V * B * 1e-12 J = V * B * 6.242e-3 eV (with V in A^3, B in GPa)
            # Actually: 1 A^3 * 1 GPa = 1e-30 m^3 * 1e9 Pa = 1e-21 J = 6.242e-3 eV
            V_i = volume_temperature[i] if i < len(volume_temperature) else volumes[mid_idx]
            alpha_i = thermal_expansion[i] if not np.isnan(thermal_expansion[i]) else 0
            B_i = bulk_modulus_temperature[i] if i < len(bulk_modulus_temperature) else 0
            cv_i = cv_ev_cell[i]

            if abs(cv_i) > 1e-15 and not np.isnan(alpha_i) and not np.isnan(B_i):
                # Convert V*B from A^3*GPa to eV: multiply by 6.242e-3
                gruneisen[i] = V_i * alpha_i * B_i * 6.242e-3 / cv_i
            else:
                gruneisen[i] = np.nan
        else:
            gruneisen[i] = np.nan

    # ============================================================
    # 8. MODE GRUNEISEN PARAMETERS
    # ============================================================

    print("\n=== Mode Gruneisen Parameters ===")

    # Mode Gruneisen: gamma_i = -V/omega_i * (d omega_i / d V)
    # Compute from finite differences of phonon frequencies at different volumes

    # Get frequencies at each volume (at Gamma point for simplicity)
    # We already have the phonon objects; recompute at Gamma
    mode_freqs = []  # (n_volumes, n_modes)

    for i_strain in range(len(STRAINS)):
        phonon_file = os.path.join(WORK_DIR, f"phonopy_strain_{i_strain:02d}.yaml")
        ph = phonopy.load(phonon_file)
        # Get frequencies at Gamma
        ph.run_qpoints([[0, 0, 0]])
        qp = ph.get_qpoints_dict()
        freqs = qp["frequencies"][0]  # First q-point = Gamma
        mode_freqs.append(freqs)

    mode_freqs = np.array(mode_freqs)  # (n_volumes, n_modes)
    n_modes = mode_freqs.shape[1]

    # Compute d(omega)/d(V) by finite differences, then gamma = -V/omega * domega/dV
    # Use central differences where possible
    mode_gruneisen_gamma = np.zeros(n_modes)
    V_ref = volumes[len(STRAINS) // 2]
    freq_ref = mode_freqs[len(STRAINS) // 2]

    for m in range(n_modes):
        if freq_ref[m] > 0.1:  # Skip acoustic modes near zero
            # Fit omega(V) linearly to get slope
            valid = mode_freqs[:, m] > 0.01  # Only positive frequencies
            if np.sum(valid) >= 3:
                coeffs = np.polyfit(volumes[valid], mode_freqs[valid, m], 1)
                domega_dV = coeffs[0]
                mode_gruneisen_gamma[m] = -V_ref / freq_ref[m] * domega_dV
            else:
                mode_gruneisen_gamma[m] = np.nan
        else:
            mode_gruneisen_gamma[m] = np.nan

    valid_modes = ~np.isnan(mode_gruneisen_gamma) & (freq_ref > 0.1)
    if np.any(valid_modes):
        mean_gruneisen = np.nanmean(mode_gruneisen_gamma[valid_modes])
        print(f"  Mean mode Gruneisen parameter (Gamma point): {mean_gruneisen:.3f}")
        print(f"  Mode frequencies and Gruneisen parameters:")
        print(f"    {'Mode':>6s}  {'Freq (THz)':>11s}  {'Gruneisen':>10s}")
        for m in range(n_modes):
            if valid_modes[m]:
                print(f"    {m:6d}  {freq_ref[m]:11.4f}  {mode_gruneisen_gamma[m]:10.4f}")

    # ============================================================
    # 9. PLOTTING
    # ============================================================

    print("\n=== Generating Plots ===")

    # --- Plot 1: E(V) curve ---
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(volumes, electronic_energies, "ko-", label="DFT (MACE)")
    ax.set_xlabel("Volume (A$^3$)")
    ax.set_ylabel("Energy (eV)")
    ax.set_title("Energy-Volume Curve")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(WORK_DIR, "e_v_curve.png"), dpi=150)
    plt.close()
    print("  Saved: e_v_curve.png")

    # --- Plot 2: Volume vs T (thermal expansion) ---
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    valid_idx = np.isfinite(volume_temperature) & (qha_temps > 0)
    if np.any(valid_idx):
        ax1.plot(qha_temps[valid_idx], volume_temperature[valid_idx], "b-")
        ax1.set_xlabel("Temperature (K)")
        ax1.set_ylabel("Volume (A$^3$)")
        ax1.set_title("Volume vs Temperature")
        ax1.grid(True, alpha=0.3)

    valid_alpha = np.isfinite(thermal_expansion) & (qha_temps > 10)
    if np.any(valid_alpha):
        ax2.plot(qha_temps[valid_alpha], thermal_expansion[valid_alpha] * 1e6, "r-")
        ax2.set_xlabel("Temperature (K)")
        ax2.set_ylabel("Thermal Expansion (10$^{-6}$ K$^{-1}$)")
        ax2.set_title("Linear Thermal Expansion Coefficient")
        ax2.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(os.path.join(WORK_DIR, "thermal_expansion.png"), dpi=150)
    plt.close()
    print("  Saved: thermal_expansion.png")

    # --- Plot 3: Bulk modulus vs T ---
    fig, ax = plt.subplots(figsize=(6, 4))
    valid_B = np.isfinite(bulk_modulus_temperature) & (qha_temps > 0)
    if np.any(valid_B):
        ax.plot(qha_temps[valid_B], bulk_modulus_temperature[valid_B], "g-")
    ax.set_xlabel("Temperature (K)")
    ax.set_ylabel("Bulk Modulus (GPa)")
    ax.set_title("Isothermal Bulk Modulus vs Temperature")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(WORK_DIR, "bulk_modulus_T.png"), dpi=150)
    plt.close()
    print("  Saved: bulk_modulus_T.png")

    # --- Plot 4: Gruneisen parameter vs T ---
    fig, ax = plt.subplots(figsize=(6, 4))
    valid_g = np.isfinite(gruneisen) & (qha_temps[:n_T] > 10)
    if np.any(valid_g):
        ax.plot(qha_temps[:n_T][valid_g], gruneisen[valid_g], "m-")
    ax.set_xlabel("Temperature (K)")
    ax.set_ylabel("Gruneisen Parameter")
    ax.set_title("Thermodynamic Gruneisen Parameter vs Temperature")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(WORK_DIR, "gruneisen_T.png"), dpi=150)
    plt.close()
    print("  Saved: gruneisen_T.png")

    # --- Plot 5: Mode Gruneisen parameters ---
    fig, ax = plt.subplots(figsize=(6, 4))
    if np.any(valid_modes):
        ax.scatter(freq_ref[valid_modes], mode_gruneisen_gamma[valid_modes],
                   c="navy", s=30, alpha=0.7, edgecolors="k", linewidths=0.5)
        ax.axhline(y=0, color="gray", linestyle="--", linewidth=0.5)
        if np.any(valid_modes):
            ax.axhline(y=mean_gruneisen, color="red", linestyle=":",
                       label=f"Mean = {mean_gruneisen:.3f}")
            ax.legend()
    ax.set_xlabel("Frequency (THz)")
    ax.set_ylabel("Mode Gruneisen Parameter")
    ax.set_title("Mode Gruneisen Parameters (at Gamma)")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(WORK_DIR, "mode_gruneisen.png"), dpi=150)
    plt.close()
    print("  Saved: mode_gruneisen.png")

    # --- Plot 6: Free energy vs volume at select temperatures ---
    fig, ax = plt.subplots(figsize=(6, 4))
    temp_indices = [0]
    for target_T in [100, 300, 500, 800]:
        idx = np.argmin(np.abs(temperatures - target_T))
        if idx not in temp_indices:
            temp_indices.append(idx)

    for t_idx in temp_indices:
        T_val = temperatures[t_idx]
        # Total free energy = electronic + phonon vibrational
        F_total = electronic_energies + fe_ev[:, t_idx]
        ax.plot(volumes, F_total, "o-", markersize=4, label=f"T = {T_val:.0f} K")

    ax.set_xlabel("Volume (A$^3$)")
    ax.set_ylabel("Free Energy (eV)")
    ax.set_title("F(V) at Different Temperatures")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(WORK_DIR, "free_energy_volume.png"), dpi=150)
    plt.close()
    print("  Saved: free_energy_volume.png")

    # ============================================================
    # 10. SUMMARY
    # ============================================================

    print("\n=== QHA Results Summary ===")
    print(f"  Ground state volume: {V0:.4f} A^3")
    print(f"  Number of volume points: {len(STRAINS)}")
    print(f"  Strain range: [{STRAINS[0]:+.3f}, {STRAINS[-1]:+.3f}]")

    # Print select values
    for target_T in [100, 300, 500]:
        idx = np.argmin(np.abs(qha_temps - target_T))
        if idx < len(volume_temperature) and idx < len(thermal_expansion) and idx < len(bulk_modulus_temperature):
            V_T = volume_temperature[idx]
            alpha_T = thermal_expansion[idx]
            B_T = bulk_modulus_temperature[idx]
            g_T = gruneisen[idx] if idx < n_T else np.nan
            print(f"\n  At T = {qha_temps[idx]:.0f} K:")
            print(f"    Volume:    {V_T:.4f} A^3 ({(V_T/V0 - 1)*100:+.3f}%)")
            if np.isfinite(alpha_T):
                print(f"    Alpha:     {alpha_T*1e6:.2f} x 10^-6 K^-1")
            if np.isfinite(B_T):
                print(f"    B_T:       {B_T:.2f} GPa")
            if np.isfinite(g_T):
                print(f"    Gruneisen: {g_T:.3f}")

except Exception as e:
    print(f"\nERROR in QHA fitting: {e}")
    print("This often happens when:")
    print("  - Not enough volume points (need >= 5)")
    print("  - Volume range too narrow or too wide")
    print("  - Imaginary phonon modes at some volumes")
    print("  - Non-convex E(V) curve (check e_v_curve.png)")
    print("\nTry adjusting STRAINS or MIN_LENGTH and rerun.")

    # Still plot E(V) for diagnostics
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(volumes, electronic_energies, "ko-")
    ax.set_xlabel("Volume (A$^3$)")
    ax.set_ylabel("Energy (eV)")
    ax.set_title("E(V) - Check for convexity")
    fig.tight_layout()
    fig.savefig(os.path.join(WORK_DIR, "e_v_curve.png"), dpi=150)
    plt.close()

print("\nDone. All outputs in:", WORK_DIR)
```

## Key Parameters

| Parameter | Default | Notes |
|---|---|---|
| `STRAINS` | [-0.02, ..., +0.02] | Isotropic volume strains around equilibrium. 5-7 points minimum. Symmetric around 0. |
| Volume range | +/- 2% | Too narrow: poor EOS fit. Too wide: may hit instabilities or phase transitions. +/- 1-2% is standard. |
| Number of volume points | 5-7 | Minimum 5 for Birch-Murnaghan / Vinet fit. 7 gives more robust results. |
| `MIN_LENGTH` | 15-20 A | Supercell for phonons. 15 A acceptable for QHA screening; 20 A for publication. |
| `DISPLACEMENT` | 0.01 A | Same as for standard phonon calculations. |
| `MESH` | [15,15,15] | q-mesh for phonon DOS at each volume. 15x15x15 is sufficient; 20x20x20 for higher accuracy. |
| `T_MAX` | 800 K | QHA validity limit. Should be well below melting. For high-T, consider MD instead. |
| `eos` | "vinet" | Equation of state for fitting. Options: "vinet", "birch_murnaghan", "murnaghan". Vinet is most general. |

### Choosing the strain range

- **Metals**: +/- 2% works well. They are generally stable over this range.
- **Semiconductors**: +/- 1.5 to 2% is fine.
- **Soft materials / molecular crystals**: use +/- 1% (larger strains may cause phase transitions).
- **If imaginary modes appear**: reduce the strain range, or accept that QHA breaks down at that volume.

## Interpreting Results

### Thermal expansion coefficient (alpha)
- **Typical metals**: 10-30 x 10^-6 K^-1 (e.g., Cu ~ 17, Al ~ 23)
- **Ceramics/oxides**: 5-15 x 10^-6 K^-1
- **Diamond/SiC**: 1-5 x 10^-6 K^-1
- **Negative thermal expansion**: rare, but occurs in ZrW2O8, ScF3 at certain T. QHA can capture this if phonon modes soften with expansion.
- Alpha increases with T and typically plateaus at high T.

### Bulk modulus B(T)
- Decreases with increasing T (materials soften as they expand).
- Typical decrease: 5-20% from 0 K to 1000 K for metals.
- If B(T) shows non-monotonic behavior, check the QHA fit quality.

### Gruneisen parameter (gamma)
- **Typical solids**: gamma = 1-3.
  - gamma ~ 1: weak anharmonicity (e.g., diamond ~ 1.0).
  - gamma ~ 2-3: moderate anharmonicity (most metals, e.g., Cu ~ 1.96, Al ~ 2.17).
  - gamma > 3: strong anharmonicity, QHA may be unreliable.
- **Negative Gruneisen modes**: some modes soften with compression. Common for transverse acoustic modes in open structures.
- The thermodynamic Gruneisen parameter is a Cv-weighted average of mode Gruneisen parameters.

### Free energy F(V,T)
- The minimum of F(V) at each T gives the equilibrium volume V(T).
- If the minimum shifts significantly with T, the material has large thermal expansion.
- If F(V) becomes non-convex, a phase transition may occur.

## Common Issues

| Issue | Cause | Fix |
|---|---|---|
| QHA fit fails | Too few volume points or non-convex E(V) | Use at least 5 points; check E(V) curve is smooth and parabolic |
| Imaginary modes at compressed volumes | Structure destabilized by compression | Reduce negative strain magnitude; exclude problematic points |
| Imaginary modes at expanded volumes | Structure approaching mechanical instability | Reduce positive strain; may indicate proximity to phase transition |
| Thermal expansion looks wrong | MACE not accurate for this chemistry, or strain range too narrow | Compare E(V) curve shape with DFT; try QE for critical volumes |
| Gruneisen parameter unreasonable (> 5 or < -2) | Poor finite difference of frequencies | Use more volume points; ensure frequencies are well-converged |
| B(T) non-monotonic | Poor EOS fit at some temperatures | Check F(V,T) curves visually; try different EOS ("birch_murnaghan") |
| Results differ from experiment | QHA neglects explicit anharmonicity (3+ phonon processes) | Expected limitation; QHA best below ~0.5*T_Debye. Use MD for higher T. |
| Very slow computation | Large supercell + many volumes | Reduce MIN_LENGTH to 15 A for screening; reduce number of strain points to 5 |
