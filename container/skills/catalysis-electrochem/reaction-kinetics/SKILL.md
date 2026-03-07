# Reaction Kinetics and Transition State Theory

## When to Use

- Compute reaction rate constants from activation barriers using transition state theory (TST)
- Correct zero-point energy for transition states with imaginary frequencies (VASPKIT 507)
- Calculate half-lives for reactions or decay processes (VASPKIT 509)
- Apply the Eyring equation for temperature-dependent rate constants
- Estimate diffusion coefficients from migration barriers
- Build microkinetic models from DFT-computed barriers and energies

## Method Selection

| Task | Method | Notes |
|---|---|---|
| Rate constant from barrier | Eyring / TST (Method A) | Uses DFT barrier + vibrational frequencies |
| Imaginary frequency correction | Filter imaginary mode (Method B) | Remove imaginary mode from TS ZPE sum |
| Half-life calculation | First-order kinetics (Method C) | t_1/2 = ln(2) / k |
| Tunneling correction | Wigner / Eckart (Method A) | Important for light atoms (H transfer) |
| Arrhenius parameters | Linear fit of ln(k) vs 1/T (Method A) | Extract A and Ea from computed rates |

```
Have a transition state with an imaginary frequency?
  --> Method B: Correct ZPE by excluding the imaginary mode

Need rate constant at a given temperature?
  --> Method A: Eyring equation with TST

Need half-life or reaction timescale?
  --> Method C: First-order kinetics from rate constant

Need to account for quantum tunneling?
  --> Method A: Apply Wigner or Eckart correction
```

## Prerequisites

- numpy, scipy (numerical computation)
- matplotlib (plotting)
- Activation barrier (from NEB or transition state search)
- Vibrational frequencies at the transition state and reactant (from frequency calculations)
- ASE thermochemistry module (optional, for partition functions)

## Detailed Steps

### Method A: Transition State Theory Rate Constants

```python
#!/usr/bin/env python3
"""
Compute reaction rate constants using transition state theory (TST).

The Eyring equation:
  k = (kB * T / h) * exp(-dG_barrier / (kB * T))

With vibrational partition functions (harmonic TST):
  k = (kB * T / h) * (q_TS / q_reactant) * exp(-dE_barrier / (kB * T))

where q_TS excludes the imaginary frequency mode.

Includes Wigner tunneling correction for light-atom transfers.
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import json

# ============================================================
# Physical constants
# ============================================================
kB = 8.617333262e-5       # Boltzmann constant (eV/K)
kB_J = 1.380649e-23       # Boltzmann constant (J/K)
h_eV = 4.135667696e-15    # Planck constant (eV*s)
h_J = 6.62607015e-34      # Planck constant (J*s)
hbar_eV = 6.582119569e-16 # reduced Planck constant (eV*s)

# ============================================================
# 1. Input: Barrier and frequencies
# ============================================================
# Replace these with your computed values

# Electronic energy barrier (eV) from NEB or TS search
dE_barrier = 0.75  # forward barrier in eV

# Vibrational frequencies at the REACTANT (cm^-1)
# All should be real and positive for a true minimum
freq_reactant_cm = [3650.0, 1595.0, 580.0, 490.0, 420.0, 350.0]

# Vibrational frequencies at the TRANSITION STATE (cm^-1)
# One should be imaginary (listed as negative by convention)
# The imaginary mode corresponds to the reaction coordinate
freq_ts_cm = [-450.0, 3200.0, 1200.0, 520.0, 400.0, 300.0]

# Temperature range
T_values = np.arange(200, 1001, 50)  # 200-1000 K


# ============================================================
# 2. Helper functions
# ============================================================
def cm_to_eV(freq_cm):
    """Convert frequency from cm^-1 to eV."""
    return abs(freq_cm) / 8065.54


def vibrational_partition_function(frequencies_cm, T):
    """
    Compute harmonic vibrational partition function.

    q_vib = product over modes: 1 / (2 * sinh(hv / 2kT))

    Parameters
    ----------
    frequencies_cm : list
        Vibrational frequencies in cm^-1 (positive, real only).
    T : float
        Temperature in K.

    Returns
    -------
    q_vib : float
        Vibrational partition function.
    zpe : float
        Zero-point energy in eV.
    """
    q_vib = 1.0
    zpe = 0.0

    for freq in frequencies_cm:
        if freq <= 0:
            continue  # skip imaginary or zero frequencies
        hv = cm_to_eV(freq)  # in eV
        zpe += 0.5 * hv
        x = hv / (2 * kB * T)
        if x > 500:
            # Avoid overflow: q ~ exp(-hv/2kT)
            q_vib *= np.exp(-x)
        else:
            q_vib *= 1.0 / (2.0 * np.sinh(x))

    return q_vib, zpe


def tst_rate_constant(dE_barrier, freq_reactant_cm, freq_ts_cm, T,
                      tunneling="none"):
    """
    Compute TST rate constant.

    Parameters
    ----------
    dE_barrier : float
        Electronic energy barrier (eV).
    freq_reactant_cm : list
        Reactant vibrational frequencies (cm^-1), all positive.
    freq_ts_cm : list
        TS vibrational frequencies (cm^-1), one may be negative (imaginary).
    T : float
        Temperature (K).
    tunneling : str
        Tunneling correction: "none", "wigner", or "eckart".

    Returns
    -------
    k : float
        Rate constant (s^-1 for unimolecular, or s^-1 if normalized).
    """
    # Separate real and imaginary TS frequencies
    real_ts_freqs = [f for f in freq_ts_cm if f > 0]
    imag_freq = [f for f in freq_ts_cm if f < 0]

    # Partition functions
    q_react, zpe_react = vibrational_partition_function(freq_reactant_cm, T)
    q_ts, zpe_ts = vibrational_partition_function(real_ts_freqs, T)

    # ZPE-corrected barrier
    dE_zpe = dE_barrier + zpe_ts - zpe_react

    # TST prefactor: kB*T/h
    prefactor = kB * T / h_eV  # in s^-1

    # Rate constant
    k = prefactor * (q_ts / q_react) * np.exp(-dE_zpe / (kB * T))

    # Tunneling correction
    if tunneling == "wigner" and imag_freq:
        # Wigner tunneling correction (first-order)
        # kappa = 1 + (1/24) * (h*|nu_imag| / kB*T)^2
        nu_imag = abs(imag_freq[0])
        hv_imag = cm_to_eV(nu_imag)
        kappa = 1.0 + (1.0 / 24.0) * (hv_imag / (kB * T))**2
        k *= kappa
    elif tunneling == "eckart" and imag_freq:
        # Simplified Eckart tunneling (Bell's formula)
        nu_imag = abs(imag_freq[0])
        hv_imag = cm_to_eV(nu_imag)
        alpha = 2 * np.pi * dE_zpe / hv_imag
        if alpha > 0:
            kappa = alpha / np.sin(alpha) if abs(alpha) < np.pi else 2.0
        else:
            kappa = 1.0
        k *= max(kappa, 1.0)

    return k, dE_zpe


# ============================================================
# 3. Compute rate constants vs temperature
# ============================================================
print(f"{'='*60}")
print(f"TRANSITION STATE THEORY RATE CONSTANTS")
print(f"{'='*60}")
print(f"Electronic barrier: {dE_barrier:.4f} eV")

# Compute ZPE-corrected barrier
_, zpe_r = vibrational_partition_function(freq_reactant_cm, 300)
_, zpe_ts = vibrational_partition_function([f for f in freq_ts_cm if f > 0], 300)
dE_zpe_corrected = dE_barrier + zpe_ts - zpe_r
print(f"ZPE (reactant):     {zpe_r:.4f} eV")
print(f"ZPE (TS):           {zpe_ts:.4f} eV")
print(f"ZPE-corrected barrier: {dE_zpe_corrected:.4f} eV")

if any(f < 0 for f in freq_ts_cm):
    imag = [f for f in freq_ts_cm if f < 0]
    print(f"Imaginary frequency: {imag[0]:.1f} cm^-1 (reaction coordinate)")

print(f"\n{'T (K)':<8} {'k_TST (s^-1)':<15} {'k_Wigner (s^-1)':<17} {'Half-life':<15}")
print("-" * 60)

k_tst_values = []
k_wigner_values = []

for T in T_values:
    k_tst, _ = tst_rate_constant(dE_barrier, freq_reactant_cm, freq_ts_cm, T,
                                  tunneling="none")
    k_wig, _ = tst_rate_constant(dE_barrier, freq_reactant_cm, freq_ts_cm, T,
                                  tunneling="wigner")

    k_tst_values.append(k_tst)
    k_wigner_values.append(k_wig)

    # Half-life (for first-order kinetics)
    if k_tst > 0:
        half_life = np.log(2) / k_tst
        if half_life < 1e-9:
            hl_str = f"{half_life*1e12:.2f} ps"
        elif half_life < 1e-6:
            hl_str = f"{half_life*1e9:.2f} ns"
        elif half_life < 1e-3:
            hl_str = f"{half_life*1e6:.2f} us"
        elif half_life < 1:
            hl_str = f"{half_life*1e3:.2f} ms"
        elif half_life < 3600:
            hl_str = f"{half_life:.2f} s"
        elif half_life < 86400:
            hl_str = f"{half_life/3600:.2f} hr"
        elif half_life < 3.156e7:
            hl_str = f"{half_life/86400:.1f} days"
        else:
            hl_str = f"{half_life/3.156e7:.2e} yr"
    else:
        hl_str = "inf"

    print(f"{T:<8.0f} {k_tst:<15.4e} {k_wig:<17.4e} {hl_str:<15}")

# ============================================================
# 4. Arrhenius plot
# ============================================================
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Left: Arrhenius plot (ln(k) vs 1/T)
ax1 = axes[0]
inv_T = 1000.0 / T_values  # 1000/T for convenient axis

ax1.semilogy(inv_T, k_tst_values, "o-", color="steelblue", markersize=5,
             label="TST (no tunneling)")
ax1.semilogy(inv_T, k_wigner_values, "s--", color="crimson", markersize=5,
             label="TST + Wigner tunneling")

ax1.set_xlabel("1000/T (K$^{-1}$)", fontsize=12)
ax1.set_ylabel("Rate constant k (s$^{-1}$)", fontsize=12)
ax1.set_title("Arrhenius Plot", fontsize=14)
ax1.legend(fontsize=10)
ax1.grid(True, alpha=0.3)

# Linear fit for Arrhenius parameters: ln(k) = ln(A) - Ea/(kB*T)
log_k = np.log(np.array(k_tst_values))
inv_T_raw = 1.0 / T_values
# Fit: slope = -Ea/kB, intercept = ln(A)
coeffs = np.polyfit(inv_T_raw, log_k, 1)
Ea_fit = -coeffs[0] * kB  # in eV
A_fit = np.exp(coeffs[1])  # in s^-1
print(f"\nArrhenius fit: Ea = {Ea_fit:.4f} eV, A = {A_fit:.4e} s^-1")

# Right: Rate constant vs temperature
ax2 = axes[1]
ax2.semilogy(T_values, k_tst_values, "o-", color="steelblue", markersize=5,
             label="TST")
ax2.semilogy(T_values, k_wigner_values, "s--", color="crimson", markersize=5,
             label="TST + Wigner")

ax2.set_xlabel("Temperature (K)", fontsize=12)
ax2.set_ylabel("Rate constant k (s$^{-1}$)", fontsize=12)
ax2.set_title("Rate Constant vs Temperature", fontsize=14)
ax2.legend(fontsize=10)
ax2.grid(True, alpha=0.3)

fig.tight_layout()
fig.savefig("reaction_kinetics.png", dpi=150)
print("\nSaved reaction_kinetics.png")

# ============================================================
# 5. Save results
# ============================================================
results = {
    "dE_barrier_eV": dE_barrier,
    "dE_zpe_corrected_eV": dE_zpe_corrected,
    "imaginary_freq_cm": [f for f in freq_ts_cm if f < 0],
    "arrhenius_Ea_eV": float(Ea_fit),
    "arrhenius_A_s-1": float(A_fit),
    "rate_constants": [
        {"T_K": float(T), "k_tst": float(k), "k_wigner": float(kw)}
        for T, k, kw in zip(T_values, k_tst_values, k_wigner_values)
    ],
}

with open("kinetics_results.json", "w") as f:
    json.dump(results, f, indent=2)
print("Saved kinetics_results.json")
```

### Method B: Imaginary Frequency Correction for Transition States

```python
#!/usr/bin/env python3
"""
Correct the ZPE and thermal properties of a transition state
by properly handling the imaginary frequency.

At a true transition state (first-order saddle point):
  - Exactly ONE imaginary frequency (the reaction coordinate mode)
  - All other frequencies are real and positive

The imaginary mode is EXCLUDED from:
  - ZPE sum
  - Vibrational partition function
  - Thermal energy and entropy

This corresponds to VASPKIT task 507 (imaginary frequency correction).
"""

import numpy as np
import json

# ============================================================
# 1. Input: Transition state frequencies
# ============================================================
# Frequencies from a TS frequency calculation (cm^-1)
# Negative values represent imaginary frequencies
frequencies_cm = [
    -487.2,     # imaginary: reaction coordinate
    3215.4,     # real mode 1
    1532.1,     # real mode 2
    986.7,      # real mode 3
    754.3,      # real mode 4
    521.0,      # real mode 5
    483.6,      # real mode 6
    312.4,      # real mode 7
    245.1,      # real mode 8
    187.3,      # real mode 9
    95.2,       # real mode 10 (low-frequency, check if physical)
    42.1,       # real mode 11 (very low, possibly frustrated translation)
]

# Physical constants
kB = 8.617333262e-5  # eV/K
h_eV = 4.135667696e-15  # eV*s

T = 298.15  # Temperature in K


def cm_to_eV(freq_cm):
    """Convert wavenumber to energy in eV."""
    return abs(freq_cm) / 8065.54


# ============================================================
# 2. Classify frequencies
# ============================================================
imaginary_freqs = [f for f in frequencies_cm if f < 0]
real_freqs = [f for f in frequencies_cm if f > 0]
near_zero = [f for f in real_freqs if abs(f) < 50]
physical_freqs = [f for f in real_freqs if abs(f) >= 50]

print("TRANSITION STATE FREQUENCY ANALYSIS")
print("=" * 50)
print(f"\nTotal modes: {len(frequencies_cm)}")
print(f"Imaginary modes: {len(imaginary_freqs)}")
for f in imaginary_freqs:
    print(f"  {f:.1f} cm^-1  ({cm_to_eV(f)*1000:.2f} meV)")

print(f"\nReal positive modes: {len(real_freqs)}")
print(f"  Physical (> 50 cm^-1): {len(physical_freqs)}")
print(f"  Near-zero (< 50 cm^-1): {len(near_zero)}")

# Validate: should have exactly 1 imaginary frequency for a true TS
if len(imaginary_freqs) == 0:
    print("\nWARNING: No imaginary frequency found!")
    print("  This is NOT a transition state. It may be a minimum.")
    print("  Re-run the saddle point search.")
elif len(imaginary_freqs) == 1:
    print(f"\nValid transition state: 1 imaginary frequency")
    print(f"  Reaction coordinate mode: {imaginary_freqs[0]:.1f} cm^-1")
elif len(imaginary_freqs) > 1:
    print(f"\nWARNING: {len(imaginary_freqs)} imaginary frequencies found!")
    print("  This is a higher-order saddle point, not a true TS.")
    print("  Distort along the second imaginary mode and re-optimize.")

# ============================================================
# 3. Compute corrected ZPE (exclude imaginary mode)
# ============================================================
# Standard ZPE: sum of 0.5 * hv for all modes
# TS ZPE: exclude the imaginary mode

zpe_all = sum(0.5 * cm_to_eV(f) for f in real_freqs)
zpe_physical = sum(0.5 * cm_to_eV(f) for f in physical_freqs)

print(f"\nZPE (all real modes):       {zpe_all:.4f} eV")
print(f"ZPE (physical modes only):  {zpe_physical:.4f} eV")

if near_zero:
    print(f"\nNote: {len(near_zero)} near-zero modes excluded from 'physical' ZPE:")
    for f in near_zero:
        print(f"  {f:.1f} cm^-1 ({cm_to_eV(f)*1000:.2f} meV)")
    print("  These may be frustrated translations/rotations of adsorbates.")

# ============================================================
# 4. Compute corrected vibrational partition function
# ============================================================
def vib_partition_function_harmonic(freq_cm_list, T):
    """Harmonic vibrational partition function, excluding imaginary modes."""
    q = 1.0
    for f in freq_cm_list:
        if f <= 0:
            continue
        hv = cm_to_eV(f)
        x = hv / (2 * kB * T)
        if x > 500:
            q *= np.exp(-x)
        else:
            q *= 1.0 / (2.0 * np.sinh(x))
    return q


q_ts_all = vib_partition_function_harmonic(real_freqs, T)
q_ts_physical = vib_partition_function_harmonic(physical_freqs, T)

print(f"\nVibrational partition function at {T:.2f} K:")
print(f"  q_vib (all real modes):      {q_ts_all:.6e}")
print(f"  q_vib (physical modes):      {q_ts_physical:.6e}")

# ============================================================
# 5. Compute thermal energy and entropy (corrected)
# ============================================================
def thermal_properties_harmonic(freq_cm_list, T):
    """
    Compute thermal internal energy and entropy in the harmonic approximation.
    Excludes imaginary modes.
    """
    u_vib = 0.0  # internal energy (includes ZPE)
    s_vib = 0.0  # entropy

    for f in freq_cm_list:
        if f <= 0:
            continue
        hv = cm_to_eV(f)
        x = hv / (kB * T)

        # Internal energy contribution: hv * [0.5 + 1/(exp(x)-1)]
        if x < 500:
            n_bose = 1.0 / (np.exp(x) - 1.0)
        else:
            n_bose = 0.0
        u_vib += hv * (0.5 + n_bose)

        # Entropy contribution: kB * [x/(exp(x)-1) - ln(1-exp(-x))]
        if x < 500:
            s_vib += kB * (x * n_bose - np.log(1.0 - np.exp(-x)))

    return u_vib, s_vib


u_vib, s_vib = thermal_properties_harmonic(physical_freqs, T)
f_vib = u_vib - T * s_vib  # Helmholtz free energy

print(f"\nThermal properties at {T:.2f} K (corrected, physical modes only):")
print(f"  U_vib = {u_vib:.4f} eV (includes ZPE)")
print(f"  S_vib = {s_vib:.6f} eV/K")
print(f"  T*S   = {T * s_vib:.4f} eV")
print(f"  F_vib = {f_vib:.4f} eV")

# ============================================================
# 6. Save corrected results
# ============================================================
results = {
    "frequencies_cm-1": frequencies_cm,
    "n_imaginary": len(imaginary_freqs),
    "imaginary_freqs_cm-1": imaginary_freqs,
    "n_real_modes": len(real_freqs),
    "n_physical_modes": len(physical_freqs),
    "zpe_corrected_eV": float(zpe_physical),
    "zpe_uncorrected_eV": float(zpe_all),
    "U_vib_eV": float(u_vib),
    "S_vib_eV_per_K": float(s_vib),
    "F_vib_eV": float(f_vib),
    "T_K": T,
    "is_valid_ts": len(imaginary_freqs) == 1,
}

with open("ts_corrected.json", "w") as f:
    json.dump(results, f, indent=2)
print("\nSaved ts_corrected.json")
```

### Method C: Half-Life Calculation

```python
#!/usr/bin/env python3
"""
Calculate half-life and reaction timescales from rate constants.

For a first-order reaction: A -> products
  Rate: d[A]/dt = -k * [A]
  Solution: [A](t) = [A]_0 * exp(-k*t)
  Half-life: t_1/2 = ln(2) / k

For temperature-dependent analysis, compute k(T) from TST
and derive half-lives at various temperatures.

Corresponds to VASPKIT task 509.
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import json

# ============================================================
# Physical constants
# ============================================================
kB = 8.617333262e-5       # eV/K
h_eV = 4.135667696e-15    # eV*s

# ============================================================
# 1. Input parameters
# ============================================================
# Option 1: Provide activation barrier directly
Ea = 0.85       # activation barrier (eV), ZPE-corrected
A = 1.0e13      # pre-exponential factor (s^-1), typical for surface reactions

# Option 2: Compute from TST (use Method A results)
# k = A * exp(-Ea / (kB * T))

temperatures = np.arange(200, 1001, 25)


# ============================================================
# 2. Compute rate constants and half-lives
# ============================================================
def compute_half_life(k):
    """
    Compute half-life from rate constant (first-order kinetics).

    Parameters
    ----------
    k : float
        Rate constant (s^-1).

    Returns
    -------
    t_half : float
        Half-life in seconds.
    """
    if k > 0:
        return np.log(2) / k
    return np.inf


def format_time(seconds):
    """Format time in human-readable units."""
    if seconds == np.inf:
        return "infinity"
    elif seconds < 1e-12:
        return f"{seconds*1e15:.2f} fs"
    elif seconds < 1e-9:
        return f"{seconds*1e12:.2f} ps"
    elif seconds < 1e-6:
        return f"{seconds*1e9:.2f} ns"
    elif seconds < 1e-3:
        return f"{seconds*1e6:.2f} us"
    elif seconds < 1:
        return f"{seconds*1e3:.2f} ms"
    elif seconds < 60:
        return f"{seconds:.2f} s"
    elif seconds < 3600:
        return f"{seconds/60:.2f} min"
    elif seconds < 86400:
        return f"{seconds/3600:.2f} hr"
    elif seconds < 3.156e7:
        return f"{seconds/86400:.1f} days"
    elif seconds < 3.156e10:
        return f"{seconds/3.156e7:.2f} yr"
    else:
        return f"{seconds/3.156e7:.2e} yr"


print(f"HALF-LIFE CALCULATION")
print(f"{'='*60}")
print(f"Activation barrier (Ea): {Ea:.4f} eV")
print(f"Pre-exponential (A):     {A:.4e} s^-1")
print(f"\n{'T (K)':<8} {'k (s^-1)':<14} {'t_1/2':>15}  {'Reaction?':<10}")
print("-" * 55)

results = []
for T in temperatures:
    k = A * np.exp(-Ea / (kB * T))
    t_half = compute_half_life(k)

    # Practical assessment
    if t_half < 1e-6:
        assessment = "instant"
    elif t_half < 1:
        assessment = "fast"
    elif t_half < 3600:
        assessment = "moderate"
    elif t_half < 86400:
        assessment = "slow"
    elif t_half < 3.156e7:
        assessment = "very slow"
    else:
        assessment = "negligible"

    print(f"{T:<8.0f} {k:<14.4e} {format_time(t_half):>15}  {assessment:<10}")

    results.append({
        "T_K": float(T),
        "k_s-1": float(k),
        "t_half_s": float(t_half),
        "assessment": assessment,
    })

# ============================================================
# 3. Find characteristic temperatures
# ============================================================
# Temperature at which half-life = 1 second
# k = ln(2), A * exp(-Ea/(kB*T)) = ln(2)
# T = Ea / (kB * ln(A/ln(2)))
T_1s = Ea / (kB * np.log(A / np.log(2)))

# Temperature at which half-life = 1 hour
T_1hr = Ea / (kB * np.log(A * 3600 / np.log(2)))

# Temperature at which half-life = 1 year
T_1yr = Ea / (kB * np.log(A * 3.156e7 / np.log(2)))

print(f"\nCharacteristic temperatures:")
print(f"  t_1/2 = 1 s:    T = {T_1s:.0f} K")
print(f"  t_1/2 = 1 hr:   T = {T_1hr:.0f} K")
print(f"  t_1/2 = 1 yr:   T = {T_1yr:.0f} K")

# ============================================================
# 4. Concentration decay profile at a given temperature
# ============================================================
T_plot = 500.0  # K
k_plot = A * np.exp(-Ea / (kB * T_plot))
t_half_plot = compute_half_life(k_plot)

# Time axis: 0 to 5 half-lives
t_max = 5 * t_half_plot
t = np.linspace(0, t_max, 500)
concentration = np.exp(-k_plot * t)  # normalized: [A]/[A]_0

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Left: Concentration decay
ax1 = axes[0]
ax1.plot(t / t_half_plot, concentration, "b-", linewidth=2)
ax1.axhline(0.5, color="red", linestyle="--", linewidth=1, alpha=0.7,
            label=f"$t_{{1/2}}$ = {format_time(t_half_plot)}")
ax1.axvline(1.0, color="red", linestyle="--", linewidth=1, alpha=0.7)
ax1.set_xlabel("$t / t_{1/2}$", fontsize=13)
ax1.set_ylabel("$[A] / [A]_0$", fontsize=13)
ax1.set_title(f"First-Order Decay at T = {T_plot:.0f} K", fontsize=14)
ax1.set_xlim(0, 5)
ax1.set_ylim(0, 1)
ax1.legend(fontsize=11)
ax1.grid(alpha=0.3)

# Right: Half-life vs temperature
ax2 = axes[1]
k_arr = A * np.exp(-Ea / (kB * temperatures))
t_half_arr = np.log(2) / k_arr

ax2.semilogy(temperatures, t_half_arr, "o-", color="steelblue", markersize=4)

# Reference lines
for ref_time, ref_label, ref_color in [
    (1.0, "1 s", "green"),
    (3600, "1 hr", "orange"),
    (86400, "1 day", "red"),
    (3.156e7, "1 yr", "darkred"),
]:
    ax2.axhline(ref_time, color=ref_color, linestyle="--", linewidth=1,
                alpha=0.7, label=ref_label)

ax2.set_xlabel("Temperature (K)", fontsize=13)
ax2.set_ylabel("Half-life (s)", fontsize=13)
ax2.set_title(f"Half-Life vs Temperature (Ea = {Ea:.2f} eV)", fontsize=14)
ax2.legend(fontsize=9, loc="upper right")
ax2.grid(True, alpha=0.3)

fig.tight_layout()
fig.savefig("half_life_analysis.png", dpi=150)
print("\nSaved half_life_analysis.png")

# ============================================================
# 5. Multiple barriers comparison
# ============================================================
barriers = [0.5, 0.75, 1.0, 1.25, 1.5]  # eV
T_compare = 300.0  # room temperature

print(f"\nBarrier comparison at T = {T_compare:.0f} K:")
print(f"{'Ea (eV)':<10} {'k (s^-1)':<14} {'t_1/2':<15}")
print("-" * 40)
for Ea_i in barriers:
    k_i = A * np.exp(-Ea_i / (kB * T_compare))
    t_i = compute_half_life(k_i)
    print(f"{Ea_i:<10.2f} {k_i:<14.4e} {format_time(t_i):<15}")

# Save all results
output = {
    "Ea_eV": Ea,
    "A_s-1": A,
    "T_1s_K": float(T_1s),
    "T_1hr_K": float(T_1hr),
    "T_1yr_K": float(T_1yr),
    "rate_constants": results,
}

with open("half_life_results.json", "w") as f:
    json.dump(output, f, indent=2)
print("\nSaved half_life_results.json")
```

## Key Parameters

| Parameter | Recommended Value | Notes |
|---|---|---|
| Activation barrier | From NEB or TS search | ZPE-corrected: Ea + ZPE_TS - ZPE_reactant |
| Pre-exponential factor A | 10^12 - 10^13 s^-1 | Typical for surface reactions; from partition functions |
| Temperature | Application-dependent | Room T (298 K), catalysis (500-800 K), etc. |
| Imaginary frequency | Exactly 1 for true TS | > 100 cm^-1 magnitude; weak imaginary may indicate incomplete optimization |
| Tunneling correction | Wigner for light atoms | Important for H-transfer reactions below 500 K |
| Low-frequency cutoff | 50 cm^-1 | Modes below this may be numerical artifacts or frustrated translations |

## Interpreting Results

1. **Rate constant magnitude**: k > 10^6 s^-1 at operating temperature means fast reaction. k < 1 s^-1 means the reaction is kinetically hindered.
2. **Half-life**: At 300 K, barriers > 1.0 eV give half-lives > years (negligible reaction). Barriers < 0.5 eV give sub-microsecond half-lives.
3. **Wigner correction**: Typically 1.0-2.0 at room temperature. Becomes significant (> 2) only for very light atoms and low temperatures.
4. **Arrhenius parameters**: The fitted Ea should be close to the ZPE-corrected barrier. The fitted A should be ~10^12-10^13 for simple surface reactions.
5. **Multiple imaginary frequencies**: Indicates a higher-order saddle point, not a true TS. Distort along the extra imaginary modes and re-optimize.
6. **Comparison to experiment**: TST typically overestimates barriers by 0.0-0.2 eV compared to experiment due to anharmonic effects and recrossing.

## Common Issues

| Issue | Solution |
|---|---|
| Zero or multiple imaginary frequencies at TS | Re-optimize the transition state. Use a tighter force convergence (fmax < 0.01 eV/A). |
| Rate constant seems too fast/slow | Check that the barrier is ZPE-corrected. Verify the pre-exponential factor. |
| Wigner correction is very large (> 10) | The barrier is very thin and tunneling dominates. Use Eckart or semiclassical methods instead. |
| Arrhenius plot is curved | Non-Arrhenius behavior at low T (tunneling) or high T (anharmonicity). Fit only the relevant T range. |
| Very low imaginary frequency (< 50 cm^-1) | The TS may be a very flat saddle point. Check if this is the correct TS for the reaction. |
| Half-life is negative or NaN | Rate constant is zero or negative. Check for numerical issues in the barrier or temperature. |
