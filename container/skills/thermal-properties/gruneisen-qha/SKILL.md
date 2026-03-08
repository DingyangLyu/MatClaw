# Gruneisen Parameters and Quasi-Harmonic Approximation (QHA)

## When to Use

- Computing thermal expansion coefficient alpha(T) as a function of temperature
- Obtaining temperature-dependent bulk modulus B(T)
- Calculating mode Gruneisen parameters (how phonon frequencies shift with volume)
- Going beyond the harmonic approximation without full anharmonic (AIMD) calculations
- Estimating thermal equation of state P(V,T)
- Obtaining Cp(T) vs Cv(T) -- the QHA correction that relates them
- Computing Helmholtz and Gibbs free energy surfaces F(V,T) and G(P,T)
- Understanding intrinsic anharmonicity strength for a material

## Method Selection

```
Need thermal expansion or T-dependent bulk modulus?
  YES --> QHA (this skill)
    Is MACE accurate for your system?
      YES --> Method A: ASE + MACE + phonopy (fast, minutes to hours)
      NO  --> Method C: QHA with QE forces at each volume (DFT accuracy)
    Want a quick Debye-model estimate first?
      YES --> Method B: Debye Model Comparison (seconds, E-V data only)
    Need to verify convergence?
      YES --> Method D: Automated QHA Convergence (systematic checks)

Need mode-resolved anharmonicity info?
  YES --> Gruneisen parameters (included in this skill)

Need intrinsic thermal conductivity (3-phonon scattering)?
  --> Use phono3py (separate workflow, not covered here)
```

The QHA assumes phonons are harmonic at each volume but allows the equilibrium volume to change with temperature. This captures most of the thermal expansion in solids below the Debye temperature. It fails near melting or for strongly anharmonic systems (e.g., PbTe, SnSe).

## Physical Background

### The QHA Workflow

The central idea of QHA is to compute the total Helmholtz free energy as:

```
F(V,T) = E_static(V) + F_phonon(V,T)
```

where `E_static(V)` is the T=0 electronic energy at volume V, and `F_phonon(V,T)` is the phonon contribution from the partition function. At each temperature T, we minimize F(V,T) with respect to V to find the equilibrium volume V(T). All temperature-dependent properties flow from this.

The integrated workflow (inspired by pyiron_atomistics QuasiHarmonicJob):

1. **Relax** the structure at ground state volume V0
2. **Volume scan**: generate N strained volumes V0*(1+epsilon) for epsilon in [-vol_range, +vol_range]
3. **Phonon at each volume**: run a full phonon calculation (displacements, forces, force constants) at each volume
4. **Collect F(V,T)**: extract free energies, entropy, Cv from phonon thermodynamics at each volume and temperature
5. **Minimize F(V,T) vs V**: for each temperature, fit F_total(V) = E_static(V) + F_phonon(V,T) to an equation of state to find V(T)
6. **Extract properties**: thermal expansion alpha(T), B(T), Gruneisen gamma(T), Cp(T), Gibbs free energy G(T)

### Key Derived Quantities

- **Thermal expansion coefficient**: alpha(T) = (1/V) * dV/dT
- **Isothermal bulk modulus**: B_T(T) = -V * d^2F/dV^2 at V=V(T)
- **Thermodynamic Gruneisen parameter**: gamma(T) = V * alpha * B_T / Cv
- **Heat capacity at constant pressure**: Cp(T) = Cv(T) * (1 + alpha * gamma * T) -- this is the QHA correction
- **Gibbs free energy**: G(T,P) = F(V(T),T) + P*V(T)

## Prerequisites

```bash
pip install phonopy seekpath
```

Pre-installed: `ase`, `mace-torch`, `pymatgen`, `numpy`, `scipy`, `matplotlib`.

## Detailed Steps

### Method A: ASE + MACE + phonopy (QHA) -- Complete Standalone Script

```python
#!/usr/bin/env python3
"""
Quasi-Harmonic Approximation (QHA) using ASE + MACE + phonopy.

Integrated workflow: volume scan -> phonon at each volume -> F(V,T) -> minimize for V(T).

Computes:
  - Thermal expansion coefficient alpha(T)
  - Temperature-dependent bulk modulus B(T)
  - Thermodynamic and mode Gruneisen parameters
  - Cp(T) vs Cv(T) via the QHA correction
  - Helmholtz free energy surface F(V,T) and Gibbs free energy G(T)

No pyiron dependency -- fully standalone.
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

# --- Volume scan parameters ---
# Number of volume points and range (inspired by pyiron QuasiHarmonicJob defaults)
# pyiron default: num_points=11, vol_range=0.1 (i.e., +/-10%)
# For most solids +/-2% with 7 points is sufficient and faster.
# For soft materials or near phase transitions, use wider range with more points.
NUM_POINTS = 11                        # Number of volume points (odd for symmetric, minimum 5)
VOL_RANGE = 0.05                       # Fractional volume range: V0*(1-VOL_RANGE) to V0*(1+VOL_RANGE)
STRAINS = np.linspace(-VOL_RANGE, VOL_RANGE, NUM_POINTS)

# --- Phonon settings (same as phonon/ skill) ---
MIN_LENGTH = 15.0                      # Supercell min length (A) -- 15 A for screening, 20 A for publication
DISPLACEMENT = 0.01                    # Displacement distance (A)
SYMPREC = 1e-5                         # Symmetry precision
FMAX = 1e-4                            # Relaxation force convergence (eV/A)
MESH = [15, 15, 15]                    # q-mesh for phonon DOS at each volume

# --- Temperature range for QHA ---
T_MIN = 0
T_MAX = 800
T_STEP = 5

# --- EOS and fitting ---
EOS_TYPE = "vinet"                     # "vinet", "birch_murnaghan", "murnaghan"
POLYNOMIAL_DEGREE = 3                  # Degree of polynomial for F(V) fitting (pyiron default: 3)

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
n_atoms = len(atoms_orig)

print(f"  Equilibrium volume: {V0:.4f} A^3")
print(f"  Equilibrium energy: {E0:.6f} eV")
print(f"  Atoms in unit cell: {n_atoms}")
write(os.path.join(WORK_DIR, "relaxed.cif"), atoms_orig)

# ============================================================
# 3. DETERMINE SUPERCELL MATRIX
# ============================================================

def get_supercell_matrix(atoms, min_length=15.0):
    """Determine supercell matrix so each lattice vector >= min_length.
    Mirrors pyiron PhonopyJob._phonopy_supercell_matrix() logic:
    supercell_range = ceil(interaction_range / cell_length)."""
    cell_lengths = atoms.cell.lengths()
    multiples = np.ceil(min_length / cell_lengths).astype(int)
    multiples = np.maximum(multiples, 1)
    return np.diag(multiples)

supercell_matrix = get_supercell_matrix(atoms_orig, min_length=MIN_LENGTH)
print(f"  Supercell matrix: diag({np.diag(supercell_matrix)})")

# ============================================================
# 4. PHONON CALCULATIONS AT EACH VOLUME
#    (pyiron QuasiHarmonicJob runs PhonopyJob at each Murnaghan volume)
# ============================================================

print(f"\n=== Phonon Calculations at {len(STRAINS)} Volumes ===")
print(f"  Strain range: [{STRAINS[0]:+.4f}, {STRAINS[-1]:+.4f}]")
print(f"  Volume range: [{V0*(1+STRAINS[0]):.2f}, {V0*(1+STRAINS[-1]):.2f}] A^3")

adaptor = AseAtomsAdaptor()
volumes = []
electronic_energies = []
free_energies_all = []      # List of F(T) arrays, one per volume (kJ/mol)
entropy_all = []            # (J/K/mol)
cv_all = []                 # (J/K/mol)
temperatures = None

for i_strain, strain in enumerate(STRAINS):
    print(f"\n--- Volume point {i_strain + 1}/{len(STRAINS)}: strain = {strain:+.4f} ---")

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

    # Compute forces for all displaced supercells
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
    # phonopy returns: free_energy in kJ/mol, entropy in J/K/mol, Cv in J/K/mol
    phonon.run_thermal_properties(t_min=T_MIN, t_max=T_MAX, t_step=T_STEP)
    tp = phonon.get_thermal_properties_dict()

    if temperatures is None:
        temperatures = tp["temperatures"]

    free_energies_all.append(tp["free_energy"])
    entropy_all.append(tp["entropy"])
    cv_all.append(tp["heat_capacity"])

    # Save individual phonon calc for later reloading (mode Gruneisen, etc.)
    phonon.save(os.path.join(WORK_DIR, f"phonopy_strain_{i_strain:02d}.yaml"))

volumes = np.array(volumes)
electronic_energies = np.array(electronic_energies)
free_energies_all = np.array(free_energies_all)     # (n_volumes, n_temps)
entropy_all = np.array(entropy_all)
cv_all = np.array(cv_all)

# Sort by volume (pyiron QuasiHarmonicJob.collect_output does this)
arg_sort = np.argsort(volumes)
volumes = volumes[arg_sort]
electronic_energies = electronic_energies[arg_sort]
free_energies_all = free_energies_all[arg_sort]
entropy_all = entropy_all[arg_sort]
cv_all = cv_all[arg_sort]

print(f"\n=== All phonon calculations complete ===")
print(f"  Volumes: {volumes}")
print(f"  Temperature points: {len(temperatures)}")

# ============================================================
# 5. RUN QHA USING PHONOPY'S PhonopyQHA
# ============================================================

print("\n=== Quasi-Harmonic Approximation ===")

# Unit conversions:
# 1 kJ/mol = 0.010364 eV (per formula unit)
kj_mol_to_ev = 0.010364
j_kmol_to_ev_k = kj_mol_to_ev / 1000.0

# PhonopyQHA wants free energies in eV (matching electronic_energies units)
fe_ev = free_energies_all * kj_mol_to_ev       # (n_volumes, n_temps)
entropy_ev = entropy_all * j_kmol_to_ev_k       # eV/K
cv_ev = cv_all * j_kmol_to_ev_k                 # eV/K

try:
    qha = PhonopyQHA(
        volumes=volumes,
        electronic_energies=electronic_energies,
        temperatures=temperatures,
        free_energy=fe_ev,
        cv=cv_ev,
        entropy=entropy_ev,
        eos=EOS_TYPE,
        t_max=T_MAX,
    )
    print("  QHA fit successful!")

    # ============================================================
    # 6. EXTRACT QHA RESULTS
    # ============================================================

    qha_temps = qha.temperatures                          # K
    thermal_expansion = qha.thermal_expansion              # 1/K (volumetric)
    volume_temperature = qha.volume_temperature            # A^3
    bulk_modulus_temperature = qha.bulk_modulus_temperature # GPa
    helmholtz_volume = qha.helmholtz_volume                # eV at each (V, T)
    gibbs_temperature = qha.gibbs_temperature              # eV

    # ============================================================
    # 7. POLYNOMIAL FIT OF F(V,T) -- alternative to PhonopyQHA
    #    (mirrors pyiron QuasiHarmonicJob.optimise_volume logic)
    # ============================================================

    # pyiron fits F_total(V) = E_static(V) + F_phonon(V,T) with np.polyfit
    # at each temperature, then finds the minimum analytically.
    # This is useful for extracting V(T), F(T), S(T), Cv(T) directly.

    print("\n=== Polynomial F(V,T) Minimization (pyiron-style) ===")

    v0_poly = []
    f_min_poly = []
    for i_t in range(len(temperatures)):
        F_total = electronic_energies + fe_ev[:, i_t]
        coeffs = np.polyfit(volumes, F_total, POLYNOMIAL_DEGREE)
        poly = np.poly1d(coeffs)
        # Find minimum: roots of derivative where second derivative > 0
        deriv = poly.deriv()
        roots = deriv.r
        real_roots = roots[roots.imag == 0].real
        # Keep roots within volume range with positive second derivative
        second_deriv = poly.deriv(2)
        valid = (second_deriv(real_roots) > 0) & \
                (real_roots > volumes.min() * 0.95) & \
                (real_roots < volumes.max() * 1.05)
        if np.any(valid):
            v_opt = real_roots[valid][0]
            v0_poly.append(v_opt)
            f_min_poly.append(poly(v_opt))
        else:
            v0_poly.append(np.nan)
            f_min_poly.append(np.nan)

    v0_poly = np.array(v0_poly)
    f_min_poly = np.array(f_min_poly)

    # ============================================================
    # 8. COMPUTE THERMODYNAMIC GRUNEISEN PARAMETER
    # ============================================================

    # gamma(T) = V(T) * alpha(T) * B_T(T) / Cv(T)
    # Using QHA-derived quantities

    # Cv at equilibrium volume: interpolate from the volume scan
    # For each T, Cv(V,T) is known at discrete volumes; pick the value closest to V(T)
    cv_equilibrium = np.zeros(len(qha_temps))
    for i_t in range(len(qha_temps)):
        if i_t < len(temperatures) and not np.isnan(volume_temperature[i_t]):
            # Find nearest volume index
            idx_v = np.argmin(np.abs(volumes - volume_temperature[i_t]))
            if i_t < cv_all.shape[1]:
                cv_equilibrium[i_t] = cv_all[idx_v, i_t]
            else:
                cv_equilibrium[i_t] = np.nan
        else:
            cv_equilibrium[i_t] = np.nan

    # Unit conversion: 1 A^3 * 1 GPa = 1e-30 m^3 * 1e9 Pa = 1e-21 J = 6.2415e-3 eV
    A3_GPa_to_eV = 6.2415e-3

    n_T = min(len(qha_temps), len(thermal_expansion),
              len(bulk_modulus_temperature), len(volume_temperature))
    gruneisen = np.full(n_T, np.nan)

    for i in range(n_T):
        cv_i = cv_equilibrium[i] * j_kmol_to_ev_k  # eV/K
        if (cv_i > 1e-15 and np.isfinite(thermal_expansion[i]) and
                np.isfinite(bulk_modulus_temperature[i]) and
                np.isfinite(volume_temperature[i])):
            gruneisen[i] = (volume_temperature[i] * thermal_expansion[i] *
                            bulk_modulus_temperature[i] * A3_GPa_to_eV / cv_i)

    # ============================================================
    # 9. Cp vs Cv -- THE QHA CORRECTION
    # ============================================================

    # Cp = Cv * (1 + alpha * gamma * T)
    # Equivalently: Cp - Cv = alpha^2 * B_T * V * T  (Nernst-Lindemann)

    print("\n=== Cp vs Cv (QHA Correction) ===")

    cp_qha = np.full(n_T, np.nan)
    cv_at_equil = np.full(n_T, np.nan)

    for i in range(n_T):
        T_i = qha_temps[i]
        cv_i_jkmol = cv_equilibrium[i]  # J/K/mol

        if (np.isfinite(cv_i_jkmol) and cv_i_jkmol > 0 and
                np.isfinite(thermal_expansion[i]) and
                np.isfinite(bulk_modulus_temperature[i]) and
                np.isfinite(volume_temperature[i]) and T_i > 0):

            cv_at_equil[i] = cv_i_jkmol  # J/K/mol

            # Cp - Cv = alpha^2 * B_T * V * T
            # alpha in 1/K, B_T in GPa, V in A^3, T in K
            # Result in eV, convert to J/K/mol:
            # 1 eV = 96.485 kJ/mol, so eV/K -> J/(K*mol) = eV/K * 96485
            alpha_i = thermal_expansion[i]
            B_i = bulk_modulus_temperature[i]
            V_i = volume_temperature[i]

            delta_cp = alpha_i**2 * B_i * V_i * T_i * A3_GPa_to_eV  # eV
            delta_cp_jkmol = delta_cp / j_kmol_to_ev_k  # J/K/mol

            cp_qha[i] = cv_i_jkmol + delta_cp_jkmol
        elif T_i == 0 and np.isfinite(cv_i_jkmol):
            cv_at_equil[i] = cv_i_jkmol
            cp_qha[i] = cv_i_jkmol  # At T=0, Cp = Cv

    # ============================================================
    # 10. MODE GRUNEISEN PARAMETERS
    # ============================================================

    print("\n=== Mode Gruneisen Parameters ===")

    # Mode Gruneisen: gamma_i = -V/omega_i * (d omega_i / d V)
    # Computed from finite differences of phonon frequencies at different volumes

    mode_freqs = []  # (n_volumes, n_modes) at Gamma point

    for i_strain in range(len(STRAINS)):
        phonon_file = os.path.join(WORK_DIR, f"phonopy_strain_{arg_sort[i_strain]:02d}.yaml")
        ph = phonopy.load(phonon_file)
        ph.run_qpoints([[0, 0, 0]])
        qp = ph.get_qpoints_dict()
        freqs = qp["frequencies"][0]  # Gamma point
        mode_freqs.append(freqs)

    mode_freqs = np.array(mode_freqs)  # (n_volumes, n_modes)
    n_modes = mode_freqs.shape[1]

    # Reference values at equilibrium volume
    mid_idx = len(STRAINS) // 2
    V_ref = volumes[mid_idx]
    freq_ref = mode_freqs[mid_idx]

    # gamma_i = -V/omega * domega/dV via linear fit
    mode_gruneisen_gamma = np.full(n_modes, np.nan)

    for m in range(n_modes):
        if freq_ref[m] > 0.1:  # Skip acoustic modes near zero
            valid = mode_freqs[:, m] > 0.01
            if np.sum(valid) >= 3:
                coeffs = np.polyfit(volumes[valid], mode_freqs[valid, m], 1)
                domega_dV = coeffs[0]
                mode_gruneisen_gamma[m] = -V_ref / freq_ref[m] * domega_dV

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
    # 11. HELMHOLTZ AND GIBBS FREE ENERGY SURFACES
    # ============================================================

    print("\n=== Free Energy Surfaces ===")

    # Helmholtz F(V,T) is already computed: F_total = E_static + F_phonon
    # Build the full surface for output
    F_surface = np.zeros((len(volumes), len(temperatures)))
    for i_v in range(len(volumes)):
        for i_t in range(len(temperatures)):
            F_surface[i_v, i_t] = electronic_energies[i_v] + fe_ev[i_v, i_t]

    # Gibbs free energy at P=0 is just F(V_eq(T), T)
    # gibbs_temperature from PhonopyQHA already has this

    # Save numerical data
    np.savez(
        os.path.join(WORK_DIR, "qha_results.npz"),
        temperatures=temperatures,
        qha_temps=qha_temps[:n_T],
        volumes=volumes,
        electronic_energies=electronic_energies,
        free_energy_phonon_ev=fe_ev,
        F_surface=F_surface,
        volume_temperature=volume_temperature,
        thermal_expansion=thermal_expansion,
        bulk_modulus_temperature=bulk_modulus_temperature,
        gruneisen=gruneisen[:n_T],
        cv_at_equil=cv_at_equil[:n_T],
        cp_qha=cp_qha[:n_T],
        gibbs_temperature=gibbs_temperature,
        mode_freqs=mode_freqs,
        mode_gruneisen=mode_gruneisen_gamma,
        freq_ref=freq_ref,
    )
    print(f"  Saved numerical data: {WORK_DIR}/qha_results.npz")

    # ============================================================
    # 12. PUBLICATION-QUALITY PLOTS
    # ============================================================

    print("\n=== Generating Plots ===")

    # --- Plot 1: E(V) curve ---
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(volumes, electronic_energies, "ko-", label="E(V)")
    ax.set_xlabel(r"Volume ($\mathrm{\AA}^3$)")
    ax.set_ylabel("Energy (eV)")
    ax.set_title("Static Energy vs Volume")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(WORK_DIR, "e_v_curve.png"), dpi=150)
    plt.close()
    print("  Saved: e_v_curve.png")

    # --- Plot 2: F(V) at multiple temperatures ---
    fig, ax = plt.subplots(figsize=(7, 5))
    cmap = plt.cm.coolwarm
    temp_targets = [0, 100, 200, 300, 400, 500, 600, 800]
    temp_indices = []
    for target_T in temp_targets:
        idx = np.argmin(np.abs(temperatures - target_T))
        if idx not in temp_indices and target_T <= T_MAX:
            temp_indices.append(idx)

    for i, t_idx in enumerate(temp_indices):
        T_val = temperatures[t_idx]
        F_total = electronic_energies + fe_ev[:, t_idx]
        color = cmap(i / max(len(temp_indices) - 1, 1))
        ax.plot(volumes, F_total, "o-", markersize=3, color=color,
                label=f"T = {T_val:.0f} K")
        # Mark minimum
        i_min = np.argmin(F_total)
        ax.plot(volumes[i_min], F_total[i_min], "v", color=color, markersize=6)

    ax.set_xlabel(r"Volume ($\mathrm{\AA}^3$)")
    ax.set_ylabel("Helmholtz Free Energy F (eV)")
    ax.set_title(r"$F(V,T) = E_{\mathrm{static}}(V) + F_{\mathrm{phonon}}(V,T)$")
    ax.legend(fontsize=7, ncol=2)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(WORK_DIR, "free_energy_volume.png"), dpi=150)
    plt.close()
    print("  Saved: free_energy_volume.png")

    # --- Plot 3: V(T) and alpha(T) ---
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    valid_idx = np.isfinite(volume_temperature) & (qha_temps > 0)
    if np.any(valid_idx):
        ax1.plot(qha_temps[valid_idx], volume_temperature[valid_idx], "b-", linewidth=1.5)
        ax1.set_xlabel("Temperature (K)")
        ax1.set_ylabel(r"Volume ($\mathrm{\AA}^3$)")
        ax1.set_title("Equilibrium Volume V(T)")
        ax1.grid(True, alpha=0.3)

    valid_alpha = np.isfinite(thermal_expansion) & (qha_temps > 10)
    if np.any(valid_alpha):
        ax2.plot(qha_temps[valid_alpha], thermal_expansion[valid_alpha] * 1e6, "r-",
                 linewidth=1.5)
        ax2.set_xlabel("Temperature (K)")
        ax2.set_ylabel(r"$\alpha_V$ ($10^{-6}$ K$^{-1}$)")
        ax2.set_title("Volumetric Thermal Expansion Coefficient")
        ax2.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(os.path.join(WORK_DIR, "thermal_expansion.png"), dpi=150)
    plt.close()
    print("  Saved: thermal_expansion.png")

    # --- Plot 4: B(T) ---
    fig, ax = plt.subplots(figsize=(6, 4))
    valid_B = np.isfinite(bulk_modulus_temperature) & (qha_temps > 0)
    if np.any(valid_B):
        ax.plot(qha_temps[valid_B], bulk_modulus_temperature[valid_B], "g-", linewidth=1.5)
    ax.set_xlabel("Temperature (K)")
    ax.set_ylabel("Bulk Modulus (GPa)")
    ax.set_title("Isothermal Bulk Modulus B(T)")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(WORK_DIR, "bulk_modulus_T.png"), dpi=150)
    plt.close()
    print("  Saved: bulk_modulus_T.png")

    # --- Plot 5: Gruneisen parameter vs T ---
    fig, ax = plt.subplots(figsize=(6, 4))
    valid_g = np.isfinite(gruneisen[:n_T]) & (qha_temps[:n_T] > 10)
    if np.any(valid_g):
        ax.plot(qha_temps[:n_T][valid_g], gruneisen[:n_T][valid_g], "m-", linewidth=1.5)
    ax.set_xlabel("Temperature (K)")
    ax.set_ylabel(r"Gr\"uneisen Parameter $\gamma$")
    ax.set_title("Thermodynamic Gruneisen Parameter")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(WORK_DIR, "gruneisen_T.png"), dpi=150)
    plt.close()
    print("  Saved: gruneisen_T.png")

    # --- Plot 6: Cp(T) vs Cv(T) ---
    fig, ax = plt.subplots(figsize=(6, 4))
    valid_cp = np.isfinite(cp_qha[:n_T]) & np.isfinite(cv_at_equil[:n_T]) & (qha_temps[:n_T] > 0)
    if np.any(valid_cp):
        ax.plot(qha_temps[:n_T][valid_cp], cv_at_equil[:n_T][valid_cp],
                "b-", linewidth=1.5, label=r"$C_v$ (harmonic)")
        ax.plot(qha_temps[:n_T][valid_cp], cp_qha[:n_T][valid_cp],
                "r--", linewidth=1.5, label=r"$C_p$ (QHA)")
    # Dulong-Petit limit: 3*N*k_B in J/K/mol = 3*N*R where N = atoms/formula unit
    # R = 8.314 J/(K*mol)
    dp_limit = 3 * n_atoms * 8.314  # J/K/mol
    ax.axhline(y=dp_limit, color="gray", linestyle=":", alpha=0.5,
               label=f"Dulong-Petit ({dp_limit:.1f} J/K/mol)")
    ax.set_xlabel("Temperature (K)")
    ax.set_ylabel("Heat Capacity (J/K/mol)")
    ax.set_title(r"$C_p(T)$ vs $C_v(T)$ -- QHA Correction")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(WORK_DIR, "cp_vs_cv.png"), dpi=150)
    plt.close()
    print("  Saved: cp_vs_cv.png")

    # --- Plot 7: Mode Gruneisen parameters ---
    fig, ax = plt.subplots(figsize=(6, 4))
    if np.any(valid_modes):
        ax.scatter(freq_ref[valid_modes], mode_gruneisen_gamma[valid_modes],
                   c="navy", s=30, alpha=0.7, edgecolors="k", linewidths=0.5)
        ax.axhline(y=0, color="gray", linestyle="--", linewidth=0.5)
        ax.axhline(y=mean_gruneisen, color="red", linestyle=":",
                   label=f"Mean = {mean_gruneisen:.3f}")
        ax.legend()
    ax.set_xlabel("Frequency (THz)")
    ax.set_ylabel(r"Mode Gr\"uneisen Parameter $\gamma_i$")
    ax.set_title("Mode Gruneisen Parameters (at Gamma)")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(WORK_DIR, "mode_gruneisen.png"), dpi=150)
    plt.close()
    print("  Saved: mode_gruneisen.png")

    # --- Plot 8: Gibbs free energy G(T) ---
    fig, ax = plt.subplots(figsize=(6, 4))
    valid_gibbs = np.isfinite(gibbs_temperature) & (qha_temps > 0)
    if np.any(valid_gibbs):
        ax.plot(qha_temps[valid_gibbs], gibbs_temperature[valid_gibbs], "k-", linewidth=1.5)
    ax.set_xlabel("Temperature (K)")
    ax.set_ylabel("Gibbs Free Energy G (eV)")
    ax.set_title("Gibbs Free Energy G(T) at P = 0")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(WORK_DIR, "gibbs_free_energy.png"), dpi=150)
    plt.close()
    print("  Saved: gibbs_free_energy.png")

    # ============================================================
    # 13. SUMMARY
    # ============================================================

    print("\n" + "=" * 60)
    print("  QHA RESULTS SUMMARY")
    print("=" * 60)
    print(f"  Material: {atoms_orig.get_chemical_formula()}")
    print(f"  Ground state volume V0: {V0:.4f} A^3")
    print(f"  Volume points: {NUM_POINTS} over [{STRAINS[0]:+.4f}, {STRAINS[-1]:+.4f}]")
    print(f"  Supercell: {np.diag(supercell_matrix)}, min_length = {MIN_LENGTH} A")
    print(f"  EOS type: {EOS_TYPE}")

    for target_T in [100, 300, 500, 800]:
        if target_T > T_MAX:
            continue
        idx = np.argmin(np.abs(qha_temps - target_T))
        if (idx < n_T and idx < len(volume_temperature) and
                idx < len(thermal_expansion) and idx < len(bulk_modulus_temperature)):
            V_T = volume_temperature[idx]
            alpha_T = thermal_expansion[idx]
            B_T = bulk_modulus_temperature[idx]
            g_T = gruneisen[idx] if idx < n_T else np.nan
            cv_T = cv_at_equil[idx] if idx < n_T else np.nan
            cp_T = cp_qha[idx] if idx < n_T else np.nan

            print(f"\n  At T = {qha_temps[idx]:.0f} K:")
            if np.isfinite(V_T):
                print(f"    Volume:       {V_T:.4f} A^3 ({(V_T/V0 - 1)*100:+.3f}%)")
            if np.isfinite(alpha_T):
                print(f"    alpha_V:      {alpha_T*1e6:.2f} x 10^-6 K^-1")
            if np.isfinite(B_T):
                print(f"    B_T:          {B_T:.2f} GPa")
            if np.isfinite(g_T):
                print(f"    Gruneisen:    {g_T:.3f}")
            if np.isfinite(cv_T):
                print(f"    Cv:           {cv_T:.2f} J/K/mol")
            if np.isfinite(cp_T):
                print(f"    Cp:           {cp_T:.2f} J/K/mol")
            if np.isfinite(cp_T) and np.isfinite(cv_T) and cv_T > 0:
                print(f"    Cp/Cv:        {cp_T/cv_T:.4f}")

except Exception as e:
    print(f"\nERROR in QHA fitting: {e}")
    print("This often happens when:")
    print("  - Not enough volume points (need >= 5)")
    print("  - Volume range too narrow or too wide")
    print("  - Imaginary phonon modes at some volumes")
    print("  - Non-convex E(V) curve (check e_v_curve.png)")
    print("\nTry adjusting NUM_POINTS, VOL_RANGE, or MIN_LENGTH and rerun.")

    # Still plot E(V) for diagnostics
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(volumes, electronic_energies, "ko-")
    ax.set_xlabel(r"Volume ($\mathrm{\AA}^3$)")
    ax.set_ylabel("Energy (eV)")
    ax.set_title("E(V) -- Check for Convexity")
    fig.tight_layout()
    fig.savefig(os.path.join(WORK_DIR, "e_v_curve.png"), dpi=150)
    plt.close()

print("\nDone. All outputs in:", WORK_DIR)
```

### Method B: Debye Model Comparison

The Debye-Gruneisen model provides a fast estimate of QHA thermal properties using only the E-V curve (no explicit phonon calculations). It estimates the Debye temperature from bulk modulus and atomic mass, then computes thermodynamic properties analytically. This is useful for:
- Quick screening before running full phonon QHA
- Validating phonopy QHA results against a simpler model
- Understanding which features come from the phonon DOS shape vs. simple Debye behavior

This script takes the same E-V data from Method A and compares Debye model predictions with full phonopy QHA results. The Debye model implementation follows the approach used in pyiron's `DebyeModel` class within the `QuasiHarmonicApproximation` framework.

```python
#!/usr/bin/env python3
"""
Debye-Gruneisen model for QHA thermal properties.

Takes E-V data (from Method A or any source), fits an EOS to extract V0, B0, B0',
then uses the Debye-Gruneisen model to compute thermal expansion, heat capacity,
and Gruneisen parameter. Optionally compares with full phonopy QHA results.

Inspired by pyiron_atomistics DebyeModel class.
No pyiron dependency -- fully standalone.
"""

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.optimize import brentq, minimize_scalar
from scipy.integrate import quad

import warnings
warnings.filterwarnings("ignore")

# ============================================================
# 1. CONFIGURATION
# ============================================================

# Option 1: Load E-V data from Method A output
# QHA_RESULTS_FILE = "/tmp/qha_calc/qha_results.npz"

# Option 2: Provide E-V data directly (volumes in A^3, energies in eV)
# If QHA_RESULTS_FILE is set and exists, it takes priority.
QHA_RESULTS_FILE = None  # Set to path of qha_results.npz from Method A, or None

# If no QHA results file, provide data manually:
# Example: Al (4-atom FCC cell)
VOLUMES_MANUAL = np.array([63.5, 64.0, 64.5, 65.0, 65.5, 66.0, 66.5, 67.0, 67.5])  # A^3
ENERGIES_MANUAL = np.array([-14.92, -14.95, -14.97, -14.98, -14.985, -14.98, -14.97, -14.95, -14.92])  # eV
N_ATOMS = 4  # Number of atoms in the unit cell
AVERAGE_MASS_AMU = 26.98  # Average atomic mass in AMU (Al = 26.98)

# Temperature range
T_MIN = 1
T_MAX = 800
T_STEP = 5
TEMPERATURES = np.arange(T_MIN, T_MAX + T_STEP, T_STEP, dtype=float)

WORK_DIR = "/tmp/debye_qha"
os.makedirs(WORK_DIR, exist_ok=True)

# ============================================================
# 2. LOAD OR SET E-V DATA
# ============================================================

have_phonopy_qha = False

if QHA_RESULTS_FILE is not None and os.path.exists(QHA_RESULTS_FILE):
    print(f"Loading QHA results from: {QHA_RESULTS_FILE}")
    data = np.load(QHA_RESULTS_FILE, allow_pickle=True)
    volumes = data["volumes"]
    electronic_energies = data["electronic_energies"]

    # Try to load phonopy QHA results for comparison
    try:
        qha_temps = data["qha_temps"]
        qha_alpha = data["thermal_expansion"]
        qha_bulk = data["bulk_modulus_temperature"]
        qha_gruneisen = data["gruneisen"]
        qha_cv = data["cv_at_equil"]
        have_phonopy_qha = True
        print("  Loaded phonopy QHA data for comparison.")
    except KeyError:
        print("  No phonopy QHA comparison data found.")

    # Infer N_ATOMS and average mass from structure if possible
    # (Fall back to manual values if not available)
    print(f"  {len(volumes)} volume points loaded.")
else:
    print("Using manually specified E-V data.")
    volumes = VOLUMES_MANUAL.copy()
    electronic_energies = ENERGIES_MANUAL.copy()

# Sort by volume
sort_idx = np.argsort(volumes)
volumes = volumes[sort_idx]
electronic_energies = electronic_energies[sort_idx]

# ============================================================
# 3. FIT EQUATION OF STATE (BIRCH-MURNAGHAN 3RD ORDER)
# ============================================================

print("\n=== Fitting Birch-Murnaghan EOS ===")

def birch_murnaghan_energy(V, E0, V0, B0, B0p):
    """3rd-order Birch-Murnaghan equation of state: E(V)."""
    eta = (V0 / V) ** (2.0 / 3.0)
    E = E0 + (9.0 * V0 * B0 / 16.0) * (
        (eta - 1.0) ** 3 * B0p +
        (eta - 1.0) ** 2 * (6.0 - 4.0 * eta)
    )
    return E

def birch_murnaghan_pressure(V, V0, B0, B0p):
    """Pressure from 3rd-order Birch-Murnaghan EOS, in GPa if B0 in GPa."""
    eta = (V0 / V) ** (2.0 / 3.0)
    P = (3.0 * B0 / 2.0) * (eta ** (7.0 / 2.0) - eta ** (5.0 / 2.0)) * (
        1.0 + 0.75 * (B0p - 4.0) * (eta - 1.0)
    )
    return P

from scipy.optimize import curve_fit

def bm_fit_func(V, E0, V0, B0_eV_A3, B0p):
    """Wrapper for curve_fit. B0 in eV/A^3 internally."""
    return birch_murnaghan_energy(V, E0, V0, B0_eV_A3, B0p)

# Initial guesses
V0_guess = volumes[np.argmin(electronic_energies)]
E0_guess = electronic_energies.min()
# Rough B0 from finite difference: B = -V * d^2E/dV^2
dV = volumes[1] - volumes[0]
d2E_dV2 = np.gradient(np.gradient(electronic_energies, dV), dV)
mid = len(volumes) // 2
B0_guess_eV_A3 = max(abs(volumes[mid] * d2E_dV2[mid]), 0.01)  # eV/A^3
B0p_guess = 4.0  # Universal default

p0 = [E0_guess, V0_guess, B0_guess_eV_A3, B0p_guess]

try:
    popt, pcov = curve_fit(bm_fit_func, volumes, electronic_energies, p0=p0,
                           maxfev=10000)
    E0_fit, V0_fit, B0_eV_A3, B0p_fit = popt

    # Convert B0 from eV/A^3 to GPa: 1 eV/A^3 = 160.2177 GPa
    eV_A3_to_GPa = 160.2177
    B0_GPa = B0_eV_A3 * eV_A3_to_GPa

    print(f"  E0 = {E0_fit:.6f} eV")
    print(f"  V0 = {V0_fit:.4f} A^3")
    print(f"  B0 = {B0_GPa:.2f} GPa ({B0_eV_A3:.6f} eV/A^3)")
    print(f"  B0' = {B0p_fit:.2f}")

except Exception as e:
    print(f"  EOS fit failed: {e}")
    print("  Using parabolic fallback.")
    coeffs = np.polyfit(volumes, electronic_energies, 2)
    V0_fit = -coeffs[1] / (2 * coeffs[0])
    E0_fit = np.polyval(coeffs, V0_fit)
    B0_eV_A3 = 2 * coeffs[0] * V0_fit
    B0_GPa = B0_eV_A3 * 160.2177
    B0p_fit = 4.0

# ============================================================
# 4. DEBYE-GRUNEISEN MODEL
#    (Following pyiron DebyeModel approach)
# ============================================================

print("\n=== Debye-Gruneisen Model ===")

# Physical constants
kB = 8.617333262e-5       # eV/K (Boltzmann constant)
kB_J = 1.380649e-23       # J/K
hbar = 6.582119569e-16    # eV*s (reduced Planck)
hbar_J = 1.054571817e-34  # J*s
NA = 6.02214076e23        # Avogadro
AMU_kg = 1.66053906660e-27  # kg per AMU

def estimate_debye_temperature(V0, B0_GPa, mass_amu, n_atoms):
    """
    Estimate the Debye temperature from bulk modulus, volume, and atomic mass.

    Uses the Debye-Gruneisen relation (as in pyiron DebyeModel):
      Theta_D = (hbar/kB) * (6*pi^2 * n/V)^(1/3) * v_mean

    where v_mean is estimated from B and rho:
      v_mean ~ (B/rho)^(1/2) * correction_factor

    A simpler empirical formula (Anderson):
      Theta_D = (hbar/kB) * [3*n_atoms*NA / (4*pi*V_molar)]^(1/3) * sqrt(B0 / (M_avg * rho))
    """
    # Volume per atom in m^3
    V_per_atom = (V0 / n_atoms) * 1e-30  # A^3 to m^3

    # Mass per atom in kg
    M_avg = mass_amu * AMU_kg

    # Bulk modulus in Pa
    B0_Pa = B0_GPa * 1e9

    # Mean sound velocity estimate: v_s ~ sqrt(B / rho)
    # where rho = M_avg / V_per_atom
    rho = M_avg / V_per_atom
    v_s = np.sqrt(B0_Pa / rho)

    # Debye temperature: Theta_D = (hbar/kB) * (6*pi^2 / V_per_atom)^(1/3) * v_s
    # Factor of (6*pi^2)^(1/3) ~ 3.898
    theta_D = (hbar_J / kB_J) * (6.0 * np.pi**2 / V_per_atom) ** (1.0 / 3.0) * v_s

    # Apply empirical correction factor ~0.617 (accounts for shear vs. longitudinal averaging)
    # This factor is calibrated to reproduce experimental Debye temperatures
    # for simple metals; pyiron uses a similar empirical prefactor.
    theta_D *= 0.617

    return theta_D

def debye_function_3(x):
    """
    Debye function D_3(x) = (3/x^3) * integral_0^x t^3/(e^t - 1) dt.

    Used to compute Debye model thermodynamic properties.
    For x -> 0: D_3 -> 1
    For x -> inf: D_3 -> (pi^4/5) / x^3 ~ 19.4818 / x^3
    """
    if x < 1e-10:
        return 1.0

    def integrand(t):
        if t < 1e-10:
            return t * t  # Limit: t^3/(e^t-1) -> t^2
        return t**3 / (np.exp(t) - 1.0)

    result, _ = quad(integrand, 0, x, limit=100)
    return 3.0 / (x**3) * result

debye_function_3_vec = np.vectorize(debye_function_3)

def debye_internal_energy(T, theta_D, n_atoms):
    """
    Internal energy from Debye model (vibrational part), in eV.
    U_vib = 3*n_atoms*kB*T * D_3(Theta_D/T) + (9/8)*n_atoms*kB*Theta_D (ZPE)
    """
    if T < 1e-10:
        # Zero-point energy only
        return 9.0 / 8.0 * n_atoms * kB * theta_D
    x = theta_D / T
    D3 = debye_function_3(x)
    zpe = 9.0 / 8.0 * n_atoms * kB * theta_D
    return 3.0 * n_atoms * kB * T * D3 + zpe

def debye_free_energy(T, theta_D, n_atoms):
    """
    Vibrational Helmholtz free energy from Debye model, in eV.
    F_vib = n_atoms * kB * T * [3*ln(1 - exp(-Theta_D/T)) - D_3(Theta_D/T)] + ZPE

    More precisely:
    F_vib = 9/8 * n * kB * Theta_D + 3*n*kB*T * [ln(1 - exp(-x)) - D_3(x)/3]
    where x = Theta_D / T
    """
    if T < 1e-10:
        # ZPE only
        return 9.0 / 8.0 * n_atoms * kB * theta_D

    x = theta_D / T
    D3 = debye_function_3(x)

    zpe = 9.0 / 8.0 * n_atoms * kB * theta_D

    # For large x, exp(-x) may underflow, which is fine (ln(1-0) = 0)
    if x > 500:
        ln_term = 0.0
    else:
        ln_term = np.log(1.0 - np.exp(-x))

    F_vib = zpe + n_atoms * kB * T * (3.0 * ln_term - D3)
    return F_vib

def debye_entropy(T, theta_D, n_atoms):
    """Entropy from Debye model, in eV/K."""
    if T < 1e-10:
        return 0.0
    x = theta_D / T
    D3 = debye_function_3(x)
    if x > 500:
        ln_term = 0.0
    else:
        ln_term = np.log(1.0 - np.exp(-x))
    S = n_atoms * kB * (4.0 * D3 - 3.0 * ln_term)
    return S

def debye_cv(T, theta_D, n_atoms):
    """
    Isochoric heat capacity from Debye model, in eV/K.
    Cv = 3*n_atoms*kB * [4*D_3(x) - 3*x/(e^x - 1)]
    where x = Theta_D / T.
    """
    if T < 1e-10:
        return 0.0
    x = theta_D / T
    D3 = debye_function_3(x)
    if x > 500:
        ex_term = 0.0
    else:
        ex_term = x / (np.exp(x) - 1.0)
    Cv = 3.0 * n_atoms * kB * (4.0 * D3 - 3.0 * ex_term)
    return Cv

# Gruneisen parameter from EOS: gamma = -d(ln Theta_D)/d(ln V)
# In the Debye-Gruneisen model: gamma = (1+B0')/2 - 1/3 (Slater)
# or gamma = (1+B0')/2 - 5/6 (Dugdale-MacDonald)
# or gamma = (1+B0')/2 - 1/2 (free-volume, Vashchenko-Zubarev)
# We use the Dugdale-MacDonald form as in pyiron:
gamma_0 = (1.0 + B0p_fit) / 2.0 - 5.0 / 6.0
print(f"  Gruneisen parameter gamma_0 (Dugdale-MacDonald): {gamma_0:.3f}")

# Debye temperature at V0
theta_D_0 = estimate_debye_temperature(V0_fit, B0_GPa, AVERAGE_MASS_AMU, N_ATOMS)
print(f"  Debye temperature Theta_D(V0): {theta_D_0:.1f} K")

def theta_debye_of_V(V):
    """
    Volume-dependent Debye temperature via Gruneisen relation:
    Theta_D(V) = Theta_D(V0) * (V0/V)^gamma_0
    """
    return theta_D_0 * (V0_fit / V) ** gamma_0

def E_static(V):
    """Static energy from EOS fit."""
    return birch_murnaghan_energy(V, E0_fit, V0_fit, B0_eV_A3, B0p_fit)

# ============================================================
# 5. MINIMIZE F(V,T) = E_static(V) + F_vib_Debye(V,T)
# ============================================================

print("\n=== Debye QHA: Minimizing F(V,T) ===")

V_min_search = volumes.min() * 0.95
V_max_search = volumes.max() * 1.05

debye_V_of_T = np.zeros(len(TEMPERATURES))
debye_F_of_T = np.zeros(len(TEMPERATURES))
debye_alpha = np.zeros(len(TEMPERATURES))
debye_cv_arr = np.zeros(len(TEMPERATURES))
debye_gruneisen_arr = np.zeros(len(TEMPERATURES))
debye_bulk_mod = np.zeros(len(TEMPERATURES))

for i, T in enumerate(TEMPERATURES):
    # Total free energy
    def F_total(V):
        theta = theta_debye_of_V(V)
        return E_static(V) + debye_free_energy(T, theta, N_ATOMS)

    result = minimize_scalar(F_total, bounds=(V_min_search, V_max_search), method="bounded")
    V_eq = result.x
    debye_V_of_T[i] = V_eq
    debye_F_of_T[i] = result.fun

    # Cv at equilibrium volume
    theta_eq = theta_debye_of_V(V_eq)
    debye_cv_arr[i] = debye_cv(T, theta_eq, N_ATOMS)

    # Gruneisen parameter (constant in simple model, but recompute for generality)
    debye_gruneisen_arr[i] = gamma_0

    # Bulk modulus at this volume from EOS
    # B(V) = -V * dP/dV, approximate numerically
    dV_num = V_eq * 1e-5
    P_plus = birch_murnaghan_pressure(V_eq + dV_num, V0_fit, B0_eV_A3 * 160.2177, B0p_fit)
    P_minus = birch_murnaghan_pressure(V_eq - dV_num, V0_fit, B0_eV_A3 * 160.2177, B0p_fit)
    dP_dV = (P_plus - P_minus) / (2 * dV_num)
    debye_bulk_mod[i] = -V_eq * dP_dV

# Thermal expansion: alpha = (1/V) * dV/dT (numerical derivative)
dT = TEMPERATURES[1] - TEMPERATURES[0]
debye_alpha[1:-1] = (debye_V_of_T[2:] - debye_V_of_T[:-2]) / (2 * dT * debye_V_of_T[1:-1])
debye_alpha[0] = debye_alpha[1]
debye_alpha[-1] = debye_alpha[-2]

# Convert Cv from eV/K to J/K/mol
debye_cv_jkmol = debye_cv_arr / kB * kB_J * NA  # eV/K -> J/K/mol

print(f"  V(0K)   = {debye_V_of_T[0]:.4f} A^3")
print(f"  V(300K) = {debye_V_of_T[np.argmin(np.abs(TEMPERATURES - 300))]:.4f} A^3")
idx_300 = np.argmin(np.abs(TEMPERATURES - 300))
print(f"  alpha(300K) = {debye_alpha[idx_300]*1e6:.2f} x 10^-6 K^-1")
print(f"  Cv(300K) = {debye_cv_jkmol[idx_300]:.2f} J/K/mol")
print(f"  B(300K) = {debye_bulk_mod[idx_300]:.2f} GPa")
print(f"  gamma   = {gamma_0:.3f} (constant in Debye model)")

# ============================================================
# 6. COMPARISON PLOTS
# ============================================================

print("\n=== Generating Comparison Plots ===")

# --- Plot 1: E-V curve with EOS fit ---
fig, ax = plt.subplots(figsize=(6, 4))
ax.plot(volumes, electronic_energies, "ko", markersize=6, label="Computed E(V)")
V_fine = np.linspace(volumes.min() * 0.98, volumes.max() * 1.02, 200)
E_fit_fine = birch_murnaghan_energy(V_fine, E0_fit, V0_fit, B0_eV_A3, B0p_fit)
ax.plot(V_fine, E_fit_fine, "r-", linewidth=1.5, label="BM3 fit")
ax.axvline(V0_fit, color="gray", linestyle=":", alpha=0.5, label=f"V0 = {V0_fit:.2f}")
ax.set_xlabel(r"Volume ($\mathrm{\AA}^3$)")
ax.set_ylabel("Energy (eV)")
ax.set_title("E(V) with Birch-Murnaghan Fit")
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)
fig.tight_layout()
fig.savefig(os.path.join(WORK_DIR, "debye_ev_fit.png"), dpi=150)
plt.close()
print("  Saved: debye_ev_fit.png")

# --- Plot 2: Thermal expansion comparison ---
fig, ax = plt.subplots(figsize=(7, 5))
mask = TEMPERATURES > 10
ax.plot(TEMPERATURES[mask], debye_alpha[mask] * 1e6, "b-", linewidth=2,
        label="Debye-Gruneisen model")
if have_phonopy_qha:
    mask_qha = (qha_temps > 10) & np.isfinite(qha_alpha)
    ax.plot(qha_temps[mask_qha], qha_alpha[mask_qha] * 1e6, "r--", linewidth=2,
            label="phonopy QHA (full phonons)")
ax.set_xlabel("Temperature (K)")
ax.set_ylabel(r"$\alpha_V$ ($10^{-6}$ K$^{-1}$)")
ax.set_title("Thermal Expansion: Debye Model vs QHA")
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3)
fig.tight_layout()
fig.savefig(os.path.join(WORK_DIR, "debye_vs_qha_alpha.png"), dpi=150)
plt.close()
print("  Saved: debye_vs_qha_alpha.png")

# --- Plot 3: Heat capacity comparison ---
fig, ax = plt.subplots(figsize=(7, 5))
ax.plot(TEMPERATURES[mask], debye_cv_jkmol[mask], "b-", linewidth=2,
        label=r"$C_v$ Debye model")
if have_phonopy_qha:
    mask_cv = (qha_temps > 0) & np.isfinite(qha_cv)
    ax.plot(qha_temps[mask_cv], qha_cv[mask_cv], "r--", linewidth=2,
            label=r"$C_v$ phonopy QHA")
# Dulong-Petit limit
dp = 3 * N_ATOMS * 8.314  # J/K/mol
ax.axhline(y=dp, color="gray", linestyle=":", alpha=0.5,
           label=f"Dulong-Petit ({dp:.1f} J/K/mol)")
ax.set_xlabel("Temperature (K)")
ax.set_ylabel("Heat Capacity (J/K/mol)")
ax.set_title("Heat Capacity: Debye Model vs QHA")
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)
fig.tight_layout()
fig.savefig(os.path.join(WORK_DIR, "debye_vs_qha_cv.png"), dpi=150)
plt.close()
print("  Saved: debye_vs_qha_cv.png")

# --- Plot 4: Gruneisen parameter comparison ---
fig, ax = plt.subplots(figsize=(7, 5))
ax.plot(TEMPERATURES[mask], debye_gruneisen_arr[mask], "b-", linewidth=2,
        label=f"Debye model (constant = {gamma_0:.3f})")
if have_phonopy_qha:
    mask_g = (qha_temps > 10) & np.isfinite(qha_gruneisen)
    ax.plot(qha_temps[mask_g], qha_gruneisen[mask_g], "r--", linewidth=2,
            label="phonopy QHA (thermodynamic)")
ax.set_xlabel("Temperature (K)")
ax.set_ylabel(r"Gr\"uneisen Parameter $\gamma$")
ax.set_title("Gruneisen Parameter: Debye Model vs QHA")
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3)
fig.tight_layout()
fig.savefig(os.path.join(WORK_DIR, "debye_vs_qha_gruneisen.png"), dpi=150)
plt.close()
print("  Saved: debye_vs_qha_gruneisen.png")

# --- Plot 5: Combined 2x2 summary ---
fig, axes = plt.subplots(2, 2, figsize=(12, 10))

# (a) E-V
ax = axes[0, 0]
ax.plot(volumes, electronic_energies, "ko", markersize=5)
ax.plot(V_fine, E_fit_fine, "r-", linewidth=1)
ax.set_xlabel(r"Volume ($\mathrm{\AA}^3$)")
ax.set_ylabel("Energy (eV)")
ax.set_title(f"(a) E(V): B0={B0_GPa:.1f} GPa, B0'={B0p_fit:.2f}")
ax.grid(True, alpha=0.3)

# (b) Thermal expansion
ax = axes[0, 1]
ax.plot(TEMPERATURES[mask], debye_alpha[mask] * 1e6, "b-", linewidth=1.5, label="Debye")
if have_phonopy_qha:
    ax.plot(qha_temps[mask_qha], qha_alpha[mask_qha] * 1e6, "r--", linewidth=1.5, label="QHA")
ax.set_xlabel("Temperature (K)")
ax.set_ylabel(r"$\alpha_V$ ($10^{-6}$ K$^{-1}$)")
ax.set_title(r"(b) Thermal Expansion $\alpha(T)$")
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)

# (c) Cv
ax = axes[1, 0]
ax.plot(TEMPERATURES[mask], debye_cv_jkmol[mask], "b-", linewidth=1.5, label="Debye")
if have_phonopy_qha:
    ax.plot(qha_temps[mask_cv], qha_cv[mask_cv], "r--", linewidth=1.5, label="QHA")
ax.axhline(y=dp, color="gray", linestyle=":", alpha=0.5)
ax.set_xlabel("Temperature (K)")
ax.set_ylabel("Cv (J/K/mol)")
ax.set_title("(c) Heat Capacity Cv(T)")
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)

# (d) Gruneisen
ax = axes[1, 1]
ax.plot(TEMPERATURES[mask], debye_gruneisen_arr[mask], "b-", linewidth=1.5,
        label=f"Debye ({gamma_0:.2f})")
if have_phonopy_qha:
    ax.plot(qha_temps[mask_g], qha_gruneisen[mask_g], "r--", linewidth=1.5, label="QHA")
ax.set_xlabel("Temperature (K)")
ax.set_ylabel(r"$\gamma$")
ax.set_title(r"(d) Gr\"uneisen Parameter $\gamma(T)$")
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)

fig.suptitle("Debye-Gruneisen Model vs Full Phonopy QHA", fontsize=14, y=1.01)
fig.tight_layout()
fig.savefig(os.path.join(WORK_DIR, "debye_vs_qha_summary.png"), dpi=150, bbox_inches="tight")
plt.close()
print("  Saved: debye_vs_qha_summary.png")

# ============================================================
# 7. SUMMARY
# ============================================================

print("\n" + "=" * 60)
print("  DEBYE-GRUNEISEN MODEL SUMMARY")
print("=" * 60)
print(f"  EOS: Birch-Murnaghan 3rd order")
print(f"  V0 = {V0_fit:.4f} A^3")
print(f"  B0 = {B0_GPa:.2f} GPa")
print(f"  B0' = {B0p_fit:.2f}")
print(f"  Debye temperature = {theta_D_0:.1f} K")
print(f"  Gruneisen parameter = {gamma_0:.3f} (Dugdale-MacDonald)")

for target_T in [100, 300, 500, 800]:
    if target_T > T_MAX:
        continue
    idx = np.argmin(np.abs(TEMPERATURES - target_T))
    print(f"\n  At T = {TEMPERATURES[idx]:.0f} K:")
    print(f"    V   = {debye_V_of_T[idx]:.4f} A^3")
    print(f"    alpha = {debye_alpha[idx]*1e6:.2f} x 10^-6 K^-1")
    print(f"    Cv  = {debye_cv_jkmol[idx]:.2f} J/K/mol")
    print(f"    B   = {debye_bulk_mod[idx]:.2f} GPa")

if have_phonopy_qha:
    print("\n  Comparison with phonopy QHA at 300 K:")
    idx_d = np.argmin(np.abs(TEMPERATURES - 300))
    idx_q = np.argmin(np.abs(qha_temps - 300))
    if np.isfinite(qha_alpha[idx_q]):
        print(f"    alpha: Debye={debye_alpha[idx_d]*1e6:.2f}, QHA={qha_alpha[idx_q]*1e6:.2f} x 10^-6 K^-1")
    if np.isfinite(qha_cv[idx_q]):
        print(f"    Cv:    Debye={debye_cv_jkmol[idx_d]:.2f}, QHA={qha_cv[idx_q]:.2f} J/K/mol")
    if np.isfinite(qha_gruneisen[idx_q]):
        print(f"    gamma: Debye={gamma_0:.3f}, QHA={qha_gruneisen[idx_q]:.3f}")

print(f"\nDone. All outputs in: {WORK_DIR}")
```

### Method C: QHA with QE Forces (DFT Accuracy)

When MACE is not accurate enough for your system, replace the force engine with Quantum ESPRESSO DFT while keeping the same QHA workflow. The structure is identical to Method A but each displaced supercell force calculation is done with `pw.x` instead of MACE.

```python
#!/usr/bin/env python3
"""
QHA workflow using phonopy + Quantum ESPRESSO forces.
Generates all QE input files for the volume scan, then post-processes.

Usage:
  1. Run this script to generate all QE inputs
  2. Execute QE calculations (pw.x for each displacement at each volume)
  3. Run the post-processing script to collect forces and run QHA

This script handles step 1. Steps 2-3 are shown below.
"""

import os
import numpy as np
import subprocess

from ase.io import read, write
from ase.optimize import LBFGS
from ase.constraints import ExpCellFilter

from mace.calculators import mace_mp

from pymatgen.core.structure import Structure
from pymatgen.core.periodic_table import Element
from pymatgen.io.ase import AseAtomsAdaptor
from pymatgen.io.phonopy import get_phonopy_structure

import phonopy

import warnings
warnings.filterwarnings("ignore")

# ============================================================
# CONFIGURATION
# ============================================================

STRUCTURE_FILE = "structure.cif"
NUM_POINTS = 7
VOL_RANGE = 0.03
STRAINS = np.linspace(-VOL_RANGE, VOL_RANGE, NUM_POINTS)

MIN_LENGTH = 15.0
DISPLACEMENT = 0.01
SYMPREC = 1e-5
FMAX = 1e-4

ECUTWFC = 60.0
ECUTRHO = 480.0
K_DENSITY = 0.03   # k-point density in 1/A (for automatic k-mesh)

WORK_DIR = "/tmp/qha_qe"
PSEUDO_DIR = os.path.join(WORK_DIR, "pseudo")
os.makedirs(PSEUDO_DIR, exist_ok=True)

T_MIN, T_MAX, T_STEP = 0, 800, 5

# Pseudopotential map (extend as needed)
PSEUDO_MAP = {
    "H": "H.pbe-kjpaw_psl.1.0.0.UPF",
    "Li": "Li.pbe-s-kjpaw_psl.1.0.0.UPF",
    "C": "C.pbe-n-kjpaw_psl.1.0.0.UPF",
    "N": "N.pbe-n-kjpaw_psl.1.0.0.UPF",
    "O": "O.pbe-n-kjpaw_psl.1.0.0.UPF",
    "Na": "Na.pbe-spn-kjpaw_psl.1.0.0.UPF",
    "Al": "Al.pbe-n-kjpaw_psl.1.0.0.UPF",
    "Si": "Si.pbe-n-kjpaw_psl.1.0.0.UPF",
    "Fe": "Fe.pbe-spn-kjpaw_psl.1.0.0.UPF",
    "Cu": "Cu.pbe-dn-kjpaw_psl.1.0.0.UPF",
    "Ti": "Ti.pbe-spn-kjpaw_psl.1.0.0.UPF",
    "Ni": "Ni.pbe-n-kjpaw_psl.1.0.0.UPF",
    "Ba": "Ba.pbe-spn-kjpaw_psl.1.0.0.UPF",
    "Zn": "Zn.pbe-dn-kjpaw_psl.1.0.0.UPF",
}

# ============================================================
# RELAX WITH MACE (for initial structure), THEN GENERATE QE INPUTS
# ============================================================

calc = mace_mp(model="medium", device="cpu", default_dtype="float64")
atoms_orig = read(STRUCTURE_FILE)
atoms_orig.calc = calc
ecf = ExpCellFilter(atoms_orig, hydrostatic_strain=False)
opt = LBFGS(ecf, logfile=os.path.join(WORK_DIR, "relax.log"))
opt.run(fmax=FMAX, steps=500)

V0 = atoms_orig.get_volume()
print(f"Relaxed V0 = {V0:.4f} A^3")

adaptor = AseAtomsAdaptor()

# Download pseudopotentials
elements = set(atoms_orig.get_chemical_symbols())
base_url = "https://pseudopotentials.quantum-espresso.org/upf_files"
for el in elements:
    if el not in PSEUDO_MAP:
        raise ValueError(f"No pseudopotential for {el}. Add to PSEUDO_MAP.")
    pp_file = PSEUDO_MAP[el]
    pp_path = os.path.join(PSEUDO_DIR, pp_file)
    if not os.path.exists(pp_path):
        subprocess.run(["wget", "-q", f"{base_url}/{pp_file}", "-O", pp_path], check=True)

def get_supercell_matrix(atoms, min_length):
    cell_lengths = atoms.cell.lengths()
    multiples = np.ceil(min_length / cell_lengths).astype(int)
    return np.diag(np.maximum(multiples, 1))

supercell_matrix = get_supercell_matrix(atoms_orig, MIN_LENGTH)

def write_qe_input(atoms, calc_dir, prefix="scf"):
    """Write pw.x input for a given ASE atoms object."""
    os.makedirs(calc_dir, exist_ok=True)
    structure = adaptor.get_structure(atoms)

    # Auto k-mesh from density
    k_mesh = [max(1, int(np.ceil(2 * np.pi / (K_DENSITY * l))))
              for l in structure.lattice.lengths]

    els = sorted(set(str(sp) for sp in structure.species))
    inp = f"""&CONTROL
    calculation = 'scf'
    prefix = '{prefix}'
    outdir = './tmp'
    pseudo_dir = '{PSEUDO_DIR}'
    tprnfor = .true.
    tstress = .false.
/
&SYSTEM
    ibrav = 0
    nat = {len(structure)}
    ntyp = {len(els)}
    ecutwfc = {ECUTWFC}
    ecutrho = {ECUTRHO}
    occupations = 'smearing'
    smearing = 'cold'
    degauss = 0.01
/
&ELECTRONS
    conv_thr = 1.0d-8
    mixing_beta = 0.7
/

ATOMIC_SPECIES
"""
    for el in els:
        mass = Element(el).atomic_mass
        inp += f"  {el:4s} {mass:10.4f}  {PSEUDO_MAP[el]}\n"

    inp += "\nCELL_PARAMETERS angstrom\n"
    for vec in structure.lattice.matrix:
        inp += f"  {vec[0]:16.10f} {vec[1]:16.10f} {vec[2]:16.10f}\n"

    inp += "\nATOMIC_POSITIONS crystal\n"
    for site in structure:
        el = str(site.specie)
        inp += f"  {el:4s} {site.frac_coords[0]:16.10f} {site.frac_coords[1]:16.10f} {site.frac_coords[2]:16.10f}\n"

    inp += f"\nK_POINTS automatic\n  {k_mesh[0]} {k_mesh[1]} {k_mesh[2]}  0 0 0\n"

    with open(os.path.join(calc_dir, "scf.in"), "w") as f:
        f.write(inp)

# Generate all displaced supercells at each volume
run_script_lines = ["#!/bin/bash", f"NPROCS=$(nproc)", ""]

for i_strain, strain in enumerate(STRAINS):
    strain_dir = os.path.join(WORK_DIR, f"strain_{i_strain:02d}")
    os.makedirs(strain_dir, exist_ok=True)

    atoms_strained = atoms_orig.copy()
    scale = (1.0 + strain) ** (1.0 / 3.0)
    atoms_strained.set_cell(atoms_orig.cell * scale, scale_atoms=True)

    pmg_struct = adaptor.get_structure(atoms_strained)
    phonopy_struct = get_phonopy_structure(pmg_struct)

    ph = phonopy.Phonopy(phonopy_struct, supercell_matrix=supercell_matrix.tolist(), symprec=SYMPREC)
    ph.generate_displacements(distance=DISPLACEMENT)
    ph.save(os.path.join(strain_dir, "phonopy_disp.yaml"))

    supercells = ph.supercells_with_displacements
    for j, sc in enumerate(supercells):
        disp_dir = os.path.join(strain_dir, f"disp_{j:04d}")
        sc_atoms = adaptor.get_atoms(
            Structure(lattice=sc.cell, species=sc.symbols, coords=sc.scaled_positions)
        )
        write_qe_input(sc_atoms, disp_dir, prefix=f"disp_{j:04d}")
        run_script_lines.append(
            f"cd {disp_dir} && mpirun --allow-run-as-root -np $NPROCS pw.x < scf.in > scf.out 2>&1 && cd -"
        )

    # Also write the undisplaced supercell for the static energy
    sc_eq = ph.supercell
    eq_dir = os.path.join(strain_dir, "equilibrium")
    sc_eq_atoms = adaptor.get_atoms(
        Structure(lattice=sc_eq.cell, species=sc_eq.symbols, coords=sc_eq.scaled_positions)
    )
    write_qe_input(sc_eq_atoms, eq_dir, prefix="equil")
    run_script_lines.append(
        f"cd {eq_dir} && mpirun --allow-run-as-root -np $NPROCS pw.x < scf.in > scf.out 2>&1 && cd -"
    )

    print(f"  Strain {i_strain}: {len(supercells)} displacements generated in {strain_dir}")

# Write run script
run_script_path = os.path.join(WORK_DIR, "run_all_qe.sh")
with open(run_script_path, "w") as f:
    f.write("\n".join(run_script_lines) + "\n")
os.chmod(run_script_path, 0o755)

print(f"\nAll QE inputs generated in: {WORK_DIR}")
print(f"Run script: bash {run_script_path}")
print(f"After QE finishes, run the post-processing script below.")
```

#### Post-processing: Collect QE forces and run QHA

```python
#!/usr/bin/env python3
"""
Post-process QE forces and run QHA analysis.
Run this after all pw.x calculations from the generation script have completed.
"""

import os
import re
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import phonopy
from phonopy import PhonopyQHA

# Must match the generation script
WORK_DIR = "/tmp/qha_qe"
NUM_POINTS = 7
VOL_RANGE = 0.03
STRAINS = np.linspace(-VOL_RANGE, VOL_RANGE, NUM_POINTS)
T_MIN, T_MAX, T_STEP = 0, 800, 5
MESH = [15, 15, 15]
EOS_TYPE = "vinet"

Ry_to_eV = 13.605693123
Bohr_to_Ang = 0.529177249

def parse_qe_forces(scf_out_path):
    """Parse forces from a QE scf.out file. Returns forces in eV/A."""
    forces = []
    with open(scf_out_path) as f:
        lines = f.readlines()

    in_forces = False
    for line in lines:
        if "Forces acting on atoms" in line:
            forces = []
            in_forces = True
            continue
        if in_forces:
            if "force =" in line:
                parts = line.split("force =")[1].split()
                fx, fy, fz = float(parts[0]), float(parts[1]), float(parts[2])
                forces.append([fx, fy, fz])
            elif len(forces) > 0 and "force =" not in line:
                in_forces = False

    # QE forces are in Ry/Bohr; convert to eV/Ang
    forces = np.array(forces) * Ry_to_eV / Bohr_to_Ang
    return forces

def parse_qe_energy(scf_out_path):
    """Parse total energy from QE scf.out. Returns energy in eV."""
    energy = None
    with open(scf_out_path) as f:
        for line in f:
            if "!" in line and "total energy" in line:
                energy = float(line.split("=")[1].split("Ry")[0]) * Ry_to_eV
    return energy

volumes = []
electronic_energies = []
free_energies_all = []
entropy_all = []
cv_all = []
temperatures = None

for i_strain, strain in enumerate(STRAINS):
    strain_dir = os.path.join(WORK_DIR, f"strain_{i_strain:02d}")

    # Load phonopy object
    ph = phonopy.load(os.path.join(strain_dir, "phonopy_disp.yaml"))
    n_disp = len(ph.supercells_with_displacements)

    # Parse forces
    forces_list = []
    for j in range(n_disp):
        scf_out = os.path.join(strain_dir, f"disp_{j:04d}", "scf.out")
        forces = parse_qe_forces(scf_out)
        forces_list.append(forces)

    ph.forces = forces_list
    ph.produce_force_constants()

    # Get volume and energy from equilibrium supercell
    eq_out = os.path.join(strain_dir, "equilibrium", "scf.out")
    E_eq = parse_qe_energy(eq_out)
    V_eq = np.abs(np.linalg.det(ph.supercell.cell))

    # Scale energy and volume to primitive cell
    n_sc = np.prod(np.diagonal(ph.supercell_matrix)).astype(int)
    volumes.append(V_eq / n_sc)
    electronic_energies.append(E_eq / n_sc)

    # Thermal properties
    ph.run_mesh(MESH, with_eigenvectors=False, is_gamma_center=True)
    ph.run_thermal_properties(t_min=T_MIN, t_max=T_MAX, t_step=T_STEP)
    tp = ph.get_thermal_properties_dict()

    if temperatures is None:
        temperatures = tp["temperatures"]

    free_energies_all.append(tp["free_energy"])
    entropy_all.append(tp["entropy"])
    cv_all.append(tp["heat_capacity"])

    print(f"  Strain {i_strain}: V = {volumes[-1]:.4f} A^3, E = {electronic_energies[-1]:.6f} eV")

# Sort and convert
volumes = np.array(volumes)
electronic_energies = np.array(electronic_energies)
free_energies_all = np.array(free_energies_all)
entropy_all = np.array(entropy_all)
cv_all = np.array(cv_all)

arg_sort = np.argsort(volumes)
volumes = volumes[arg_sort]
electronic_energies = electronic_energies[arg_sort]
free_energies_all = free_energies_all[arg_sort]
entropy_all = entropy_all[arg_sort]
cv_all = cv_all[arg_sort]

# Run QHA (same as Method A from here)
kj_mol_to_ev = 0.010364
j_kmol_to_ev_k = kj_mol_to_ev / 1000.0

fe_ev = free_energies_all * kj_mol_to_ev
entropy_ev = entropy_all * j_kmol_to_ev_k
cv_ev = cv_all * j_kmol_to_ev_k

qha = PhonopyQHA(
    volumes=volumes,
    electronic_energies=electronic_energies,
    temperatures=temperatures,
    free_energy=fe_ev,
    cv=cv_ev,
    entropy=entropy_ev,
    eos=EOS_TYPE,
    t_max=T_MAX,
)

print("\nQHA fit successful!")
print(f"  V(300K) = {qha.volume_temperature[np.argmin(np.abs(qha.temperatures - 300))]:.4f} A^3")
print(f"  B(300K) = {qha.bulk_modulus_temperature[np.argmin(np.abs(qha.temperatures - 300))]:.2f} GPa")

# Plotting follows identically to Method A (see above)
# ...
```

### Method D: Automated QHA Convergence

Systematic convergence testing for QHA calculations. This script automatically runs QHA with varying numbers of volume points and supercell sizes, then checks convergence of thermal expansion and bulk modulus. Inspired by pyiron's automated convergence checking in the `QuasiHarmonicApproximation` workflow.

```python
#!/usr/bin/env python3
"""
Automated QHA convergence testing.

Checks convergence with respect to:
  1. Number of volume points (5, 7, 9, 11)
  2. Supercell size (MIN_LENGTH = 10, 15, 20 A)

For each setting, runs a full QHA calculation and extracts key properties
at reference temperatures. Generates convergence plots showing when results
are converged.

Inspired by pyiron_atomistics QuasiHarmonicApproximation convergence checks.
No pyiron dependency -- fully standalone.
"""

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from copy import deepcopy
import json
import time

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

# Convergence test parameters
VOLUME_POINT_TESTS = [5, 7, 9, 11]    # Number of volume points to test
VOL_RANGE = 0.04                       # Fractional volume range (+/-4%)
SUPERCELL_SIZE_TESTS = [10.0, 15.0, 20.0]  # MIN_LENGTH values to test (A)

# Fixed parameters for each sub-test
DISPLACEMENT = 0.01                    # Displacement distance (A)
SYMPREC = 1e-5                         # Symmetry precision
FMAX = 1e-4                            # Relaxation convergence (eV/A)
MESH = [15, 15, 15]                    # q-mesh for phonon DOS

# Temperature settings
T_MIN = 0
T_MAX = 800
T_STEP = 5

# Reference temperatures for convergence checks
T_REF = [100, 300, 500]

EOS_TYPE = "vinet"

WORK_DIR = "/tmp/qha_convergence"
os.makedirs(WORK_DIR, exist_ok=True)

# ============================================================
# 2. HELPER FUNCTIONS
# ============================================================

calc = mace_mp(model=MACE_MODEL, device=DEVICE, default_dtype="float64")
adaptor = AseAtomsAdaptor()

# Unit conversions
kj_mol_to_ev = 0.010364
j_kmol_to_ev_k = kj_mol_to_ev / 1000.0

def get_supercell_matrix(atoms, min_length):
    """Determine supercell matrix so each lattice vector >= min_length."""
    cell_lengths = atoms.cell.lengths()
    multiples = np.ceil(min_length / cell_lengths).astype(int)
    return np.diag(np.maximum(multiples, 1))

def run_qha_single(atoms_relaxed, strains, min_length, label=""):
    """
    Run a complete QHA calculation for given strains and supercell size.

    Parameters
    ----------
    atoms_relaxed : ASE Atoms
        Relaxed structure (with calculator attached).
    strains : array
        Volume strains relative to V0.
    min_length : float
        Minimum supercell length for phonon calculations.
    label : str
        Label for this run.

    Returns
    -------
    dict with keys:
        'volumes', 'energies', 'temperatures',
        'thermal_expansion', 'bulk_modulus', 'volume_temperature',
        'gruneisen', 'success', 'error', 'n_disp', 'wall_time'
    """
    t_start = time.time()
    result = {
        'label': label,
        'n_vol_points': len(strains),
        'min_length': min_length,
        'success': False,
        'error': None,
        'n_disp': 0,
        'wall_time': 0.0,
    }

    V0 = atoms_relaxed.get_volume()
    supercell_matrix = get_supercell_matrix(atoms_relaxed, min_length)

    volumes = []
    electronic_energies = []
    free_energies_all = []
    entropy_all = []
    cv_all = []
    temperatures = None
    total_disp = 0

    for i_strain, strain in enumerate(strains):
        atoms_strained = atoms_relaxed.copy()
        atoms_strained.calc = calc

        scale = (1.0 + strain) ** (1.0 / 3.0)
        atoms_strained.set_cell(atoms_relaxed.cell * scale, scale_atoms=True)

        V = atoms_strained.get_volume()
        E = atoms_strained.get_potential_energy()
        volumes.append(V)
        electronic_energies.append(E)

        pmg_struct = adaptor.get_structure(atoms_strained)
        phonopy_struct = get_phonopy_structure(pmg_struct)

        phonon = phonopy.Phonopy(
            phonopy_struct,
            supercell_matrix=supercell_matrix.tolist(),
            symprec=SYMPREC,
        )
        phonon.generate_displacements(distance=DISPLACEMENT)
        supercells = phonon.supercells_with_displacements
        total_disp += len(supercells)

        forces_list = []
        for sc in supercells:
            sc_atoms = adaptor.get_atoms(
                Structure(lattice=sc.cell, species=sc.symbols, coords=sc.scaled_positions)
            )
            sc_atoms.calc = calc
            forces_list.append(sc_atoms.get_forces())

        phonon.forces = forces_list
        phonon.produce_force_constants()

        phonon.run_thermal_properties(t_min=T_MIN, t_max=T_MAX, t_step=T_STEP)
        tp = phonon.get_thermal_properties_dict()

        if temperatures is None:
            temperatures = tp["temperatures"]

        free_energies_all.append(tp["free_energy"])
        entropy_all.append(tp["entropy"])
        cv_all.append(tp["heat_capacity"])

    volumes = np.array(volumes)
    electronic_energies = np.array(electronic_energies)
    free_energies_all = np.array(free_energies_all)
    entropy_all = np.array(entropy_all)
    cv_all = np.array(cv_all)

    # Sort by volume
    arg_sort = np.argsort(volumes)
    volumes = volumes[arg_sort]
    electronic_energies = electronic_energies[arg_sort]
    free_energies_all = free_energies_all[arg_sort]
    entropy_all = entropy_all[arg_sort]
    cv_all = cv_all[arg_sort]

    result['n_disp'] = total_disp
    result['volumes'] = volumes
    result['electronic_energies'] = electronic_energies

    # Run QHA
    fe_ev = free_energies_all * kj_mol_to_ev
    entropy_ev = entropy_all * j_kmol_to_ev_k
    cv_ev = cv_all * j_kmol_to_ev_k

    try:
        qha = PhonopyQHA(
            volumes=volumes,
            electronic_energies=electronic_energies,
            temperatures=temperatures,
            free_energy=fe_ev,
            cv=cv_ev,
            entropy=entropy_ev,
            eos=EOS_TYPE,
            t_max=T_MAX,
        )

        result['temperatures'] = qha.temperatures
        result['thermal_expansion'] = qha.thermal_expansion
        result['bulk_modulus'] = qha.bulk_modulus_temperature
        result['volume_temperature'] = qha.volume_temperature
        result['success'] = True

        # Extract values at reference temperatures
        ref_data = {}
        for T_r in T_REF:
            idx = np.argmin(np.abs(qha.temperatures - T_r))
            ref_data[T_r] = {
                'alpha': float(qha.thermal_expansion[idx]) if np.isfinite(qha.thermal_expansion[idx]) else None,
                'B': float(qha.bulk_modulus_temperature[idx]) if np.isfinite(qha.bulk_modulus_temperature[idx]) else None,
                'V': float(qha.volume_temperature[idx]) if np.isfinite(qha.volume_temperature[idx]) else None,
            }
        result['ref_data'] = ref_data

    except Exception as e:
        result['error'] = str(e)
        result['ref_data'] = {}

    result['wall_time'] = time.time() - t_start
    return result

# ============================================================
# 3. RELAX GROUND STATE
# ============================================================

print("=== Ground State Relaxation ===")
atoms_orig = read(STRUCTURE_FILE)
atoms_orig.calc = calc

ecf = ExpCellFilter(atoms_orig, hydrostatic_strain=False)
opt = LBFGS(ecf, logfile=os.path.join(WORK_DIR, "relax.log"))
opt.run(fmax=FMAX, steps=500)

V0 = atoms_orig.get_volume()
print(f"  Formula: {atoms_orig.get_chemical_formula()}")
print(f"  V0 = {V0:.4f} A^3")
write(os.path.join(WORK_DIR, "relaxed.cif"), atoms_orig)

# ============================================================
# 4. CONVERGENCE TEST 1: NUMBER OF VOLUME POINTS
# ============================================================

print("\n" + "=" * 60)
print("  CONVERGENCE TEST 1: Number of Volume Points")
print("=" * 60)

FIXED_MIN_LENGTH = 15.0  # Hold supercell size constant

vol_point_results = []
for n_pts in VOLUME_POINT_TESTS:
    strains = np.linspace(-VOL_RANGE, VOL_RANGE, n_pts)
    label = f"npts={n_pts}"
    print(f"\n--- Running QHA with {n_pts} volume points ---")
    print(f"    Strains: {strains}")

    res = run_qha_single(atoms_orig, strains, FIXED_MIN_LENGTH, label=label)
    vol_point_results.append(res)

    if res['success']:
        print(f"    Success! Wall time: {res['wall_time']:.1f} s, "
              f"Total displacements: {res['n_disp']}")
        for T_r in T_REF:
            rd = res['ref_data'].get(T_r, {})
            alpha_str = f"{rd['alpha']*1e6:.3f}" if rd.get('alpha') else "N/A"
            B_str = f"{rd['B']:.2f}" if rd.get('B') else "N/A"
            print(f"    T={T_r}K: alpha={alpha_str} x10^-6/K, B={B_str} GPa")
    else:
        print(f"    FAILED: {res['error']}")

# ============================================================
# 5. CONVERGENCE TEST 2: SUPERCELL SIZE
# ============================================================

print("\n" + "=" * 60)
print("  CONVERGENCE TEST 2: Supercell Size (MIN_LENGTH)")
print("=" * 60)

FIXED_N_POINTS = 7
strains_fixed = np.linspace(-VOL_RANGE, VOL_RANGE, FIXED_N_POINTS)

supercell_results = []
for min_len in SUPERCELL_SIZE_TESTS:
    label = f"sc={min_len:.0f}A"
    print(f"\n--- Running QHA with MIN_LENGTH = {min_len} A ---")
    sc_mat = get_supercell_matrix(atoms_orig, min_len)
    print(f"    Supercell matrix: diag({np.diag(sc_mat)})")

    res = run_qha_single(atoms_orig, strains_fixed, min_len, label=label)
    supercell_results.append(res)

    if res['success']:
        print(f"    Success! Wall time: {res['wall_time']:.1f} s, "
              f"Total displacements: {res['n_disp']}")
        for T_r in T_REF:
            rd = res['ref_data'].get(T_r, {})
            alpha_str = f"{rd['alpha']*1e6:.3f}" if rd.get('alpha') else "N/A"
            B_str = f"{rd['B']:.2f}" if rd.get('B') else "N/A"
            print(f"    T={T_r}K: alpha={alpha_str} x10^-6/K, B={B_str} GPa")
    else:
        print(f"    FAILED: {res['error']}")

# ============================================================
# 6. CONVERGENCE ANALYSIS
# ============================================================

print("\n" + "=" * 60)
print("  CONVERGENCE ANALYSIS")
print("=" * 60)

def assess_convergence(results, param_values, param_name):
    """
    Assess convergence by comparing successive results.
    Returns dict with convergence metrics.
    """
    metrics = {'converged': False, 'best_idx': -1}

    for T_r in T_REF:
        alphas = []
        Bs = []
        for res in results:
            if res['success']:
                rd = res['ref_data'].get(T_r, {})
                alphas.append(rd.get('alpha'))
                Bs.append(rd.get('B'))
            else:
                alphas.append(None)
                Bs.append(None)

        # Check relative change between successive values
        alpha_changes = []
        B_changes = []
        for i in range(1, len(alphas)):
            if alphas[i] is not None and alphas[i-1] is not None and alphas[i-1] != 0:
                alpha_changes.append(abs(alphas[i] - alphas[i-1]) / abs(alphas[i-1]))
            else:
                alpha_changes.append(None)
            if Bs[i] is not None and Bs[i-1] is not None and Bs[i-1] != 0:
                B_changes.append(abs(Bs[i] - Bs[i-1]) / abs(Bs[i-1]))
            else:
                B_changes.append(None)

        metrics[f'alpha_at_{T_r}K'] = alphas
        metrics[f'B_at_{T_r}K'] = Bs
        metrics[f'alpha_change_{T_r}K'] = alpha_changes
        metrics[f'B_change_{T_r}K'] = B_changes

    # Convergence criterion: relative change < 1% for all T_REF
    last_changes_ok = True
    for T_r in T_REF:
        ac = metrics.get(f'alpha_change_{T_r}K', [])
        bc = metrics.get(f'B_change_{T_r}K', [])
        if ac and ac[-1] is not None and ac[-1] > 0.01:
            last_changes_ok = False
        if bc and bc[-1] is not None and bc[-1] > 0.01:
            last_changes_ok = False

    metrics['converged'] = last_changes_ok
    return metrics

vol_conv = assess_convergence(vol_point_results, VOLUME_POINT_TESTS, "n_vol_points")
sc_conv = assess_convergence(supercell_results, SUPERCELL_SIZE_TESTS, "min_length")

print(f"\n  Volume points convergence: {'CONVERGED' if vol_conv['converged'] else 'NOT CONVERGED'}")
for T_r in T_REF:
    changes = vol_conv.get(f'alpha_change_{T_r}K', [])
    formatted = [f"{c*100:.2f}%" if c is not None else "N/A" for c in changes]
    print(f"    alpha relative change at {T_r}K: {formatted}")

print(f"\n  Supercell size convergence: {'CONVERGED' if sc_conv['converged'] else 'NOT CONVERGED'}")
for T_r in T_REF:
    changes = sc_conv.get(f'alpha_change_{T_r}K', [])
    formatted = [f"{c*100:.2f}%" if c is not None else "N/A" for c in changes]
    print(f"    alpha relative change at {T_r}K: {formatted}")

# ============================================================
# 7. CONVERGENCE PLOTS
# ============================================================

print("\n=== Generating Convergence Plots ===")

# --- Plot 1: Volume points convergence ---
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

for i_T, T_r in enumerate(T_REF):
    ax = axes[i_T]

    # Thermal expansion
    alphas = vol_conv[f'alpha_at_{T_r}K']
    valid_npts = []
    valid_alphas = []
    for j, a in enumerate(alphas):
        if a is not None:
            valid_npts.append(VOLUME_POINT_TESTS[j])
            valid_alphas.append(a * 1e6)

    if valid_alphas:
        ax.plot(valid_npts, valid_alphas, "bo-", markersize=8, linewidth=2)
        # Shade +/-1% band around final value
        final_val = valid_alphas[-1]
        ax.axhspan(final_val * 0.99, final_val * 1.01, alpha=0.2, color="green",
                   label=r"$\pm$1% of final")
        ax.axhline(final_val, color="green", linestyle="--", alpha=0.5)

    ax.set_xlabel("Number of Volume Points")
    ax.set_ylabel(r"$\alpha_V$ ($10^{-6}$ K$^{-1}$)")
    ax.set_title(f"T = {T_r} K")
    ax.set_xticks(VOLUME_POINT_TESTS)
    ax.grid(True, alpha=0.3)
    if i_T == 0:
        ax.legend(fontsize=9)

fig.suptitle(r"Convergence of $\alpha(T)$ with Number of Volume Points", fontsize=13)
fig.tight_layout()
fig.savefig(os.path.join(WORK_DIR, "convergence_vol_points_alpha.png"), dpi=150)
plt.close()
print("  Saved: convergence_vol_points_alpha.png")

# --- Plot 2: Volume points convergence -- bulk modulus ---
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

for i_T, T_r in enumerate(T_REF):
    ax = axes[i_T]
    Bs = vol_conv[f'B_at_{T_r}K']
    valid_npts = []
    valid_Bs = []
    for j, b in enumerate(Bs):
        if b is not None:
            valid_npts.append(VOLUME_POINT_TESTS[j])
            valid_Bs.append(b)

    if valid_Bs:
        ax.plot(valid_npts, valid_Bs, "rs-", markersize=8, linewidth=2)
        final_val = valid_Bs[-1]
        ax.axhspan(final_val * 0.99, final_val * 1.01, alpha=0.2, color="green",
                   label=r"$\pm$1% of final")
        ax.axhline(final_val, color="green", linestyle="--", alpha=0.5)

    ax.set_xlabel("Number of Volume Points")
    ax.set_ylabel("Bulk Modulus (GPa)")
    ax.set_title(f"T = {T_r} K")
    ax.set_xticks(VOLUME_POINT_TESTS)
    ax.grid(True, alpha=0.3)
    if i_T == 0:
        ax.legend(fontsize=9)

fig.suptitle("Convergence of B(T) with Number of Volume Points", fontsize=13)
fig.tight_layout()
fig.savefig(os.path.join(WORK_DIR, "convergence_vol_points_B.png"), dpi=150)
plt.close()
print("  Saved: convergence_vol_points_B.png")

# --- Plot 3: Supercell size convergence ---
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

for i_T, T_r in enumerate(T_REF):
    ax = axes[i_T]
    alphas = sc_conv[f'alpha_at_{T_r}K']
    valid_sc = []
    valid_alphas = []
    for j, a in enumerate(alphas):
        if a is not None:
            valid_sc.append(SUPERCELL_SIZE_TESTS[j])
            valid_alphas.append(a * 1e6)

    if valid_alphas:
        ax.plot(valid_sc, valid_alphas, "go-", markersize=8, linewidth=2)
        final_val = valid_alphas[-1]
        ax.axhspan(final_val * 0.99, final_val * 1.01, alpha=0.2, color="green",
                   label=r"$\pm$1% of final")
        ax.axhline(final_val, color="green", linestyle="--", alpha=0.5)

    ax.set_xlabel(r"MIN\_LENGTH ($\mathrm{\AA}$)")
    ax.set_ylabel(r"$\alpha_V$ ($10^{-6}$ K$^{-1}$)")
    ax.set_title(f"T = {T_r} K")
    ax.grid(True, alpha=0.3)
    if i_T == 0:
        ax.legend(fontsize=9)

fig.suptitle(r"Convergence of $\alpha(T)$ with Supercell Size", fontsize=13)
fig.tight_layout()
fig.savefig(os.path.join(WORK_DIR, "convergence_supercell_alpha.png"), dpi=150)
plt.close()
print("  Saved: convergence_supercell_alpha.png")

# --- Plot 4: Supercell size convergence -- bulk modulus ---
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

for i_T, T_r in enumerate(T_REF):
    ax = axes[i_T]
    Bs = sc_conv[f'B_at_{T_r}K']
    valid_sc = []
    valid_Bs = []
    for j, b in enumerate(Bs):
        if b is not None:
            valid_sc.append(SUPERCELL_SIZE_TESTS[j])
            valid_Bs.append(b)

    if valid_Bs:
        ax.plot(valid_sc, valid_Bs, "ms-", markersize=8, linewidth=2)
        final_val = valid_Bs[-1]
        ax.axhspan(final_val * 0.99, final_val * 1.01, alpha=0.2, color="green",
                   label=r"$\pm$1% of final")
        ax.axhline(final_val, color="green", linestyle="--", alpha=0.5)

    ax.set_xlabel(r"MIN\_LENGTH ($\mathrm{\AA}$)")
    ax.set_ylabel("Bulk Modulus (GPa)")
    ax.set_title(f"T = {T_r} K")
    ax.grid(True, alpha=0.3)
    if i_T == 0:
        ax.legend(fontsize=9)

fig.suptitle("Convergence of B(T) with Supercell Size", fontsize=13)
fig.tight_layout()
fig.savefig(os.path.join(WORK_DIR, "convergence_supercell_B.png"), dpi=150)
plt.close()
print("  Saved: convergence_supercell_B.png")

# --- Plot 5: Full alpha(T) curves at different volume point counts ---
fig, ax = plt.subplots(figsize=(8, 5))
colors = plt.cm.viridis(np.linspace(0, 0.8, len(vol_point_results)))

for i, res in enumerate(vol_point_results):
    if res['success']:
        temps = res['temperatures']
        alpha = res['thermal_expansion']
        mask = (temps > 10) & np.isfinite(alpha)
        if np.any(mask):
            ax.plot(temps[mask], alpha[mask] * 1e6, color=colors[i], linewidth=1.5,
                    label=f"{res['n_vol_points']} vol. points")

ax.set_xlabel("Temperature (K)")
ax.set_ylabel(r"$\alpha_V$ ($10^{-6}$ K$^{-1}$)")
ax.set_title("Thermal Expansion at Different Volume Point Counts")
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)
fig.tight_layout()
fig.savefig(os.path.join(WORK_DIR, "convergence_alpha_curves_npts.png"), dpi=150)
plt.close()
print("  Saved: convergence_alpha_curves_npts.png")

# --- Plot 6: Full alpha(T) curves at different supercell sizes ---
fig, ax = plt.subplots(figsize=(8, 5))
colors = plt.cm.plasma(np.linspace(0, 0.8, len(supercell_results)))

for i, res in enumerate(supercell_results):
    if res['success']:
        temps = res['temperatures']
        alpha = res['thermal_expansion']
        mask = (temps > 10) & np.isfinite(alpha)
        if np.any(mask):
            ax.plot(temps[mask], alpha[mask] * 1e6, color=colors[i], linewidth=1.5,
                    label=f"MIN_LENGTH={SUPERCELL_SIZE_TESTS[i]:.0f} A")

ax.set_xlabel("Temperature (K)")
ax.set_ylabel(r"$\alpha_V$ ($10^{-6}$ K$^{-1}$)")
ax.set_title("Thermal Expansion at Different Supercell Sizes")
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)
fig.tight_layout()
fig.savefig(os.path.join(WORK_DIR, "convergence_alpha_curves_sc.png"), dpi=150)
plt.close()
print("  Saved: convergence_alpha_curves_sc.png")

# --- Plot 7: Cost vs accuracy tradeoff ---
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

# Volume points: cost vs accuracy
costs_npts = [res['n_disp'] for res in vol_point_results]
times_npts = [res['wall_time'] for res in vol_point_results]
alpha_300_npts = []
for res in vol_point_results:
    if res['success'] and res['ref_data'].get(300, {}).get('alpha') is not None:
        alpha_300_npts.append(res['ref_data'][300]['alpha'] * 1e6)
    else:
        alpha_300_npts.append(np.nan)

ax1.bar(range(len(VOLUME_POINT_TESTS)), times_npts, color="steelblue", alpha=0.7)
ax1.set_xticks(range(len(VOLUME_POINT_TESTS)))
ax1.set_xticklabels([str(n) for n in VOLUME_POINT_TESTS])
ax1.set_xlabel("Number of Volume Points")
ax1.set_ylabel("Wall Time (s)", color="steelblue")
ax1.tick_params(axis="y", labelcolor="steelblue")

ax1b = ax1.twinx()
ax1b.plot(range(len(VOLUME_POINT_TESTS)), alpha_300_npts, "ro-", markersize=8)
ax1b.set_ylabel(r"$\alpha_V$ at 300K ($10^{-6}$/K)", color="red")
ax1b.tick_params(axis="y", labelcolor="red")
ax1.set_title("Cost vs Accuracy: Volume Points")

# Supercell size: cost vs accuracy
costs_sc = [res['n_disp'] for res in supercell_results]
times_sc = [res['wall_time'] for res in supercell_results]
alpha_300_sc = []
for res in supercell_results:
    if res['success'] and res['ref_data'].get(300, {}).get('alpha') is not None:
        alpha_300_sc.append(res['ref_data'][300]['alpha'] * 1e6)
    else:
        alpha_300_sc.append(np.nan)

ax2.bar(range(len(SUPERCELL_SIZE_TESTS)), times_sc, color="steelblue", alpha=0.7)
ax2.set_xticks(range(len(SUPERCELL_SIZE_TESTS)))
ax2.set_xticklabels([f"{s:.0f}" for s in SUPERCELL_SIZE_TESTS])
ax2.set_xlabel(r"MIN\_LENGTH ($\mathrm{\AA}$)")
ax2.set_ylabel("Wall Time (s)", color="steelblue")
ax2.tick_params(axis="y", labelcolor="steelblue")

ax2b = ax2.twinx()
ax2b.plot(range(len(SUPERCELL_SIZE_TESTS)), alpha_300_sc, "ro-", markersize=8)
ax2b.set_ylabel(r"$\alpha_V$ at 300K ($10^{-6}$/K)", color="red")
ax2b.tick_params(axis="y", labelcolor="red")
ax2.set_title("Cost vs Accuracy: Supercell Size")

fig.suptitle("Computational Cost vs Accuracy Tradeoff", fontsize=13)
fig.tight_layout()
fig.savefig(os.path.join(WORK_DIR, "convergence_cost_accuracy.png"), dpi=150)
plt.close()
print("  Saved: convergence_cost_accuracy.png")

# ============================================================
# 8. SAVE CONVERGENCE DATA
# ============================================================

convergence_data = {
    'volume_point_tests': VOLUME_POINT_TESTS,
    'supercell_size_tests': SUPERCELL_SIZE_TESTS,
    'vol_range': VOL_RANGE,
    'vol_convergence': vol_conv,
    'sc_convergence': sc_conv,
}

# Save summary to JSON-compatible format
summary = {
    'vol_points_converged': vol_conv['converged'],
    'supercell_converged': sc_conv['converged'],
    'results_vol_points': [],
    'results_supercell': [],
}

for res in vol_point_results:
    summary['results_vol_points'].append({
        'n_points': res['n_vol_points'],
        'success': res['success'],
        'wall_time': res['wall_time'],
        'n_disp': res['n_disp'],
        'ref_data': res.get('ref_data', {}),
    })

for res in supercell_results:
    summary['results_supercell'].append({
        'min_length': res['min_length'],
        'success': res['success'],
        'wall_time': res['wall_time'],
        'n_disp': res['n_disp'],
        'ref_data': res.get('ref_data', {}),
    })

with open(os.path.join(WORK_DIR, "convergence_summary.json"), "w") as f:
    json.dump(summary, f, indent=2, default=str)
print(f"\n  Saved: {WORK_DIR}/convergence_summary.json")

# ============================================================
# 9. FINAL SUMMARY
# ============================================================

print("\n" + "=" * 60)
print("  CONVERGENCE TEST SUMMARY")
print("=" * 60)

print(f"\n  Material: {atoms_orig.get_chemical_formula()}")
print(f"  V0 = {V0:.4f} A^3")

print(f"\n  Volume points test (MIN_LENGTH={FIXED_MIN_LENGTH} A):")
print(f"    Tested: {VOLUME_POINT_TESTS}")
print(f"    Converged: {vol_conv['converged']}")
for T_r in T_REF:
    alphas = vol_conv[f'alpha_at_{T_r}K']
    vals = [f"{a*1e6:.3f}" if a is not None else "N/A" for a in alphas]
    print(f"    alpha({T_r}K): {vals} x 10^-6/K")

print(f"\n  Supercell size test ({FIXED_N_POINTS} volume points):")
print(f"    Tested: {SUPERCELL_SIZE_TESTS} A")
print(f"    Converged: {sc_conv['converged']}")
for T_r in T_REF:
    alphas = sc_conv[f'alpha_at_{T_r}K']
    vals = [f"{a*1e6:.3f}" if a is not None else "N/A" for a in alphas]
    print(f"    alpha({T_r}K): {vals} x 10^-6/K")

total_time = sum(r['wall_time'] for r in vol_point_results + supercell_results)
print(f"\n  Total wall time: {total_time:.0f} s ({total_time/60:.1f} min)")

# Recommendation
if vol_conv['converged'] and sc_conv['converged']:
    # Find minimum converged settings
    min_npts = VOLUME_POINT_TESTS[0]
    for i in range(1, len(VOLUME_POINT_TESTS)):
        all_small = True
        for T_r in T_REF:
            ch = vol_conv[f'alpha_change_{T_r}K']
            if i-1 < len(ch) and ch[i-1] is not None and ch[i-1] < 0.01:
                continue
            else:
                all_small = False
        if all_small:
            min_npts = VOLUME_POINT_TESTS[i]
            break

    min_sc = SUPERCELL_SIZE_TESTS[0]
    for i in range(1, len(SUPERCELL_SIZE_TESTS)):
        all_small = True
        for T_r in T_REF:
            ch = sc_conv[f'alpha_change_{T_r}K']
            if i-1 < len(ch) and ch[i-1] is not None and ch[i-1] < 0.01:
                continue
            else:
                all_small = False
        if all_small:
            min_sc = SUPERCELL_SIZE_TESTS[i]
            break

    print(f"\n  RECOMMENDATION: Use NUM_POINTS={min_npts}, MIN_LENGTH={min_sc} A")
    print(f"  These are the minimum converged settings (<1% change).")
else:
    print("\n  RECOMMENDATION: Results are not fully converged.")
    print("  Consider testing more volume points or larger supercells.")

print(f"\nDone. All outputs in: {WORK_DIR}")
```

## Key Parameters

| Parameter | Default | Notes |
|---|---|---|
| `NUM_POINTS` | 11 | Number of volume points. pyiron default is 11. Minimum 5 for a reliable EOS fit. 7 is a good compromise between cost and accuracy. |
| `VOL_RANGE` | 0.05 | Fractional volume range (+/-5%). pyiron default is 0.10 (+/-10%). For most solids +/-2% to +/-5% is sufficient. |
| `STRAINS` | linspace(-0.05, 0.05, 11) | Isotropic volume strains. Symmetric around 0. |
| `MIN_LENGTH` | 15-20 A | Supercell size for phonons. 15 A for QHA screening; 20 A for publication quality. |
| `DISPLACEMENT` | 0.01 A | Finite displacement for phonon force constants. Standard value. |
| `MESH` | [15,15,15] | q-mesh for phonon DOS at each volume. 15x15x15 is sufficient; 20x20x20 for higher accuracy. |
| `T_MAX` | 800 K | Maximum temperature. QHA validity limit: should be well below melting point. |
| `EOS_TYPE` | "vinet" | Equation of state. Options: "vinet", "birch_murnaghan", "murnaghan". Vinet is most general and recommended. |
| `POLYNOMIAL_DEGREE` | 3 | Degree of polynomial for F(V) fitting (pyiron-style analysis). 3 is standard. |
| `AVERAGE_MASS_AMU` | (material-dependent) | Average atomic mass in AMU for Debye model (Method B). E.g., Al=26.98, Cu=63.55, Si=28.09. |
| `gamma_0` (Debye) | Dugdale-MacDonald | Gruneisen parameter estimate from EOS: (1+B0')/2 - 5/6. Alternatives: Slater (-1/3), free-volume (-1/2). |
| `theta_D_0` (Debye) | Estimated from B0, V0 | Debye temperature. If experimental value is known, override the estimate for better accuracy. |
| `ECUTWFC` (QE) | 60 Ry | Plane-wave kinetic energy cutoff for QE calculations (Method C). Material-dependent; check convergence. |
| `ECUTRHO` (QE) | 480 Ry | Charge density cutoff for QE (Method C). Typically 6-10x ECUTWFC for PAW/ultrasoft. |
| `K_DENSITY` (QE) | 0.03 1/A | k-point density for automatic k-mesh generation in QE (Method C). Smaller = denser mesh = more accurate. |
| `tprnfor` (QE) | `.true.` | Must be `.true.` in QE input to print forces. Required for phonopy force extraction. |
| `VOLUME_POINT_TESTS` | [5,7,9,11] | Volume point counts tested in convergence study (Method D). |
| `SUPERCELL_SIZE_TESTS` | [10,15,20] A | Supercell sizes tested in convergence study (Method D). |
| `T_REF` | [100,300,500] K | Reference temperatures for convergence checks (Method D). |

### Convergence: How Many Volumes?

| Points | Quality | Use Case |
|---|---|---|
| 5 | Minimum viable | Quick screening, qualitative trends |
| 7 | Good | Most production calculations |
| 11 | Excellent | Publication quality, pyiron default |
| 15+ | Overkill | Only needed for unusual E(V) shapes or near phase transitions |

The critical test: plot F(V) at several temperatures. If the minimum is well-defined and the fit looks smooth, you have enough points.

### Convergence: Supercell Size for Phonons

| MIN_LENGTH | Quality | Use Case |
|---|---|---|
| 10 A | Rough | Very quick screening; only for metals with short-range interactions |
| 15 A | Good | QHA screening; adequate for most purposes |
| 20 A | Publication | Standard for publishable phonon calculations |
| 25+ A | Conservative | Needed for materials with long-range force constants (e.g., polar materials, molecular crystals) |

Remember: QHA cost scales as (NUM_POINTS) * (n_displacements per volume) * (cost per force evaluation). Increasing supercell size dramatically increases n_displacements.

### Choosing the Strain Range

- **Metals**: +/-2% to +/-5% works well. Generally stable over this range.
- **Semiconductors**: +/-2% to +/-3%. Covalent bonds are stiff.
- **Soft materials / molecular crystals**: use +/-1% to +/-2% (larger strains may trigger phase transitions).
- **Near a phase boundary**: use +/-1% with many points (11+). The E(V) curve may have shoulders.
- **If imaginary modes appear at extreme volumes**: reduce the strain range rather than discarding those points.

## Interpreting Results

### Thermal Expansion Coefficient (alpha)

- **Typical metals**: 10-30 x 10^-6 K^-1 (e.g., Cu ~ 17, Al ~ 23, Fe ~ 12)
- **Ceramics/oxides**: 5-15 x 10^-6 K^-1
- **Diamond/SiC**: 1-5 x 10^-6 K^-1
- **Negative thermal expansion**: rare; occurs in ZrW2O8, ScF3 at certain T. QHA can capture this if phonon modes soften with expansion.
- alpha increases with T and typically plateaus at high T.
- The script computes volumetric alpha. Linear alpha = volumetric alpha / 3 for cubic systems.

### Bulk Modulus B(T)

- Decreases with increasing T (materials soften as they expand).
- Typical decrease: 5-20% from 0 K to 1000 K for metals.
- If B(T) shows non-monotonic behavior, check the QHA fit quality.

### Gruneisen Parameter (gamma)

- **Typical solids**: gamma = 1-3.
  - gamma ~ 1: weak anharmonicity (diamond ~ 1.0)
  - gamma ~ 2-3: moderate anharmonicity (Cu ~ 1.96, Al ~ 2.17)
  - gamma > 3: strong anharmonicity, QHA may be unreliable
- **Negative Gruneisen modes**: some modes soften with compression. Common for transverse acoustic modes in open structures.
- The thermodynamic Gruneisen parameter is a Cv-weighted average of mode Gruneisen parameters.

### Cp(T) vs Cv(T)

- At T = 0: Cp = Cv (third law of thermodynamics).
- At room temperature: Cp/Cv is typically 1.01 to 1.05 for hard solids, up to 1.1 for soft metals.
- The difference Cp - Cv = alpha^2 * B_T * V * T grows with temperature.
- Experimental heat capacity is Cp (measured at constant pressure). Phonon calculations give Cv. The QHA correction bridges this gap.
- Both Cp and Cv approach the Dulong-Petit limit 3NR at high T; Cp approaches it from above.

### Free Energy F(V,T)

- The minimum of F(V) at each T gives the equilibrium volume V(T).
- If the minimum shifts significantly with T, the material has large thermal expansion.
- If F(V) becomes non-convex or develops a double minimum, a phase transition may occur -- QHA cannot describe this.
- The Gibbs free energy G(T) = F(V_eq(T),T) at P=0 should decrease monotonically with T.

### Debye Model Validity (Method B)

- The Debye model assumes a single Debye cutoff frequency and a parabolic phonon DOS. It works best for simple monatomic solids (metals, diamond).
- For compounds with large mass contrast (e.g., BaTiO3, PbTe), the actual phonon DOS has optical branches that deviate strongly from the Debye form. The Debye model will underestimate Cv at intermediate T.
- If the Debye model thermal expansion agrees with full QHA within 10-20%, the material has a relatively simple phonon spectrum and the Debye model is a useful approximation.
- If they disagree by > 50%, the material has complex phonon behavior (van Hove singularities, optical branches, soft modes) and the full QHA is essential.
- The Debye model Gruneisen parameter is constant (from the EOS), while the thermodynamic Gruneisen parameter from QHA varies with T due to mode-dependent Gruneisen parameters.
- The Debye temperature estimate from B0 and V0 is approximate (empirical prefactor ~0.617). If experimental Debye temperature is known, override the estimate for better results.

### QHA vs Debye Comparison

- **Good agreement** (both give similar alpha, Cv): The material has a Debye-like phonon DOS. The Debye model can be used for quick estimates and extrapolation.
- **QHA gives higher alpha**: Common for materials with low-frequency optical modes that contribute strongly to thermal expansion. These modes are not captured by the Debye model.
- **QHA gives lower alpha**: Can happen for materials with flat (dispersionless) optical bands that contribute to Cv but not to thermal expansion.
- **Cv agrees but alpha differs**: The Gruneisen parameter is mode-dependent. The Debye model uses a single constant gamma, while QHA captures mode-resolved gamma_i.

### Convergence Assessment (Method D)

- **Volume points**: 7 points typically suffice for smooth E(V) curves. If the relative change in alpha(300K) is < 1% when going from 7 to 9 points, the result is converged.
- **Supercell size**: For most materials, MIN_LENGTH=15 A gives results within 5% of the converged value. 20 A is needed for polar materials with long-range force constants.
- **Convergence metric**: Compare relative change in alpha(T) and B(T) at multiple reference temperatures. If all are < 1%, the calculation is converged.
- **Cost scaling**: Total cost ~ NUM_POINTS * n_displacements * (supercell_size)^3. Doubling the supercell length increases cost ~8x per volume point.
- **Recommend**: Run Method D first on a new material to determine the minimum converged settings before running a full production calculation with Method A.

## Limitations

QHA is an approximation. It works well in a defined regime and fails outside it:

| Limitation | Description | Alternative |
|---|---|---|
| **Breaks down near melting** | QHA assumes harmonic phonons at each volume. Near T_melt, explicit anharmonicity (phonon-phonon scattering, diffusion) dominates. | Use molecular dynamics (MD) for T > ~0.5 * T_melt |
| **Strongly anharmonic systems** | Materials like PbTe, SnSe, CsPbBr3, and many perovskites have giant anharmonic phonon renormalization that QHA misses. | Use TDEP, SSCHA, or temperature-dependent effective potential (TDEP) methods |
| **Phase transitions** | QHA cannot describe first-order structural transitions or order-disorder transitions. | Run separate QHA for each phase; compare G(T) curves |
| **Soft modes** | If a phonon mode goes imaginary within the volume range, QHA fails at those volumes. | Restrict volume range; consider anharmonic methods |
| **Quantum nuclear effects** | Important for light atoms (H, Li) at low T. QHA uses classical thermal occupation modified by quantum statistics, but path-integral effects are missing. | Path integral molecular dynamics (PIMD) |
| **Magnetic transitions** | QHA ignores magnetic entropy and magnetic ordering transitions. | Couple with magnetic free energy models |
| **Electronic entropy** | Not included in the standard QHA script above. Important for metals at very high T. | Add Sommerfeld electronic free energy term: F_el = -1/2 * gamma_el * T^2 |

**Rule of thumb**: QHA is reliable up to approximately 2/3 of the Debye temperature or 1/2 of the melting point, whichever is lower.

## Common Issues

| Issue | Cause | Fix |
|---|---|---|
| QHA fit fails | Too few volume points or non-convex E(V) | Use at least 5 points; check E(V) curve is smooth and parabolic |
| Imaginary modes at compressed volumes | Structure destabilized by compression | Reduce negative strain magnitude; exclude problematic points |
| Imaginary modes at expanded volumes | Structure approaching mechanical instability | Reduce positive strain; may indicate proximity to phase transition |
| Thermal expansion looks wrong | MACE not accurate for this chemistry, or strain range too narrow | Compare E(V) curve shape with DFT; try Method C (QE) for critical volumes |
| Gruneisen parameter unreasonable (> 5 or < -2) | Poor finite difference of frequencies | Use more volume points; ensure frequencies are well-converged |
| B(T) non-monotonic | Poor EOS fit at some temperatures | Check F(V,T) curves visually; try different EOS ("birch_murnaghan") |
| Cp < Cv | Numerical artifact from noisy alpha or B_T | Smooth alpha(T) before computing Cp; increase volume points |
| Results differ from experiment | QHA neglects explicit anharmonicity | Expected limitation; QHA best below ~0.5*T_Debye. Use MD for higher T |
| Very slow computation | Large supercell + many volumes | Reduce MIN_LENGTH to 15 A for screening; reduce NUM_POINTS to 5-7 |
| PhonopyQHA crashes with "singular matrix" | Volume range too narrow for EOS fit | Increase VOL_RANGE to +/-5% or switch to polynomial fit (pyiron-style) |
| Debye model Theta_D is unrealistic | Bad EOS fit or wrong average mass | Check BM3 fit quality; verify AVERAGE_MASS_AMU matches your structure |
| Debye model alpha much larger than QHA | Low-lying optical modes not captured by Debye | Expected for complex structures; trust the full QHA result |
| Debye model Cv saturates too early | Debye temperature underestimated | Override theta_D_0 with experimental value if available |
| QE forces parsing fails | QE output format mismatch or incomplete calculation | Check scf.out for convergence; verify `tprnfor = .true.` in input |
| QE SCF not converging | Poor k-mesh, ecutwfc too low, or bad pseudopotential | Increase ecutwfc; try `mixing_beta = 0.3`; check pseudopotential compatibility |
| Convergence test shows oscillating alpha | EOS fitting instability with few points | Use "vinet" EOS which is more robust; or switch to polynomial fit |
| Convergence not reached at largest supercell | Long-range force constants (polar material) | Add Born effective charges and dielectric constant to phonopy (NAC correction) |
| Method D takes too long | Too many convergence test combinations | Reduce SUPERCELL_SIZE_TESTS to [10, 15] for initial screening; test 20 A separately |
| E(V) curve asymmetric | Anharmonic potential energy surface | Use more points on the compressed side; consider asymmetric strain range |

## Output Files

| File | Description |
|---|---|
| `qha_results.npz` | All numerical data (loadable with `np.load`) |
| `e_v_curve.png` | Static E(V) curve |
| `free_energy_volume.png` | F(V,T) at multiple temperatures -- the central QHA plot |
| `thermal_expansion.png` | V(T) and alpha(T) |
| `bulk_modulus_T.png` | Isothermal bulk modulus B(T) |
| `gruneisen_T.png` | Thermodynamic Gruneisen parameter gamma(T) |
| `cp_vs_cv.png` | Cp(T) vs Cv(T) with Dulong-Petit limit |
| `mode_gruneisen.png` | Mode Gruneisen parameters gamma_i vs frequency |
| `gibbs_free_energy.png` | Gibbs free energy G(T) at P=0 |
| `phonopy_strain_XX.yaml` | Individual phonon calculations (reloadable) |
| `debye_ev_fit.png` | E(V) with Birch-Murnaghan fit (Method B) |
| `debye_vs_qha_summary.png` | 2x2 comparison: Debye model vs phonopy QHA (Method B) |
| `debye_vs_qha_alpha.png` | Thermal expansion comparison (Method B) |
| `debye_vs_qha_cv.png` | Heat capacity comparison (Method B) |
| `debye_vs_qha_gruneisen.png` | Gruneisen parameter comparison (Method B) |
| `convergence_vol_points_alpha.png` | alpha convergence vs number of volume points (Method D) |
| `convergence_vol_points_B.png` | B(T) convergence vs number of volume points (Method D) |
| `convergence_supercell_alpha.png` | alpha convergence vs supercell size (Method D) |
| `convergence_supercell_B.png` | B(T) convergence vs supercell size (Method D) |
| `convergence_alpha_curves_npts.png` | Full alpha(T) curves at different volume point counts (Method D) |
| `convergence_alpha_curves_sc.png` | Full alpha(T) curves at different supercell sizes (Method D) |
| `convergence_cost_accuracy.png` | Cost vs accuracy tradeoff plot (Method D) |
| `convergence_summary.json` | Machine-readable convergence results (Method D) |
