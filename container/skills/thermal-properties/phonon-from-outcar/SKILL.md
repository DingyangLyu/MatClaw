# Phonon Extraction and Post-Processing from VASP OUTCAR / QE Output

## When to Use

- Extracting phonon frequencies and eigenvectors from a VASP OUTCAR produced by IBRION=5,6,7,8 (finite differences or DFPT)
- Post-processing QE dynmat.x or matdyn.x output files for phonon band structures and DOS
- Computing element-projected (atom-projected) phonon band structure and DOS
- Sorting phonon bands by eigenvector continuity to avoid band-crossing artifacts in plots
- Parsing Born effective charges and dielectric tensors for LO-TO splitting
- Equivalent to VASPKIT functions 952 (phonon from OUTCAR), 953 (projected phonon dispersion), 954 (band sorting), 955 (LO-TO correction)

## Method Selection

```
What phonon data source do you have?

VASP OUTCAR with IBRION=5,6,7,8?
  --> Method A: Parse OUTCAR for phonon frequencies and eigenvectors

VASP with phonopy (SPOSCAR + force sets)?
  --> Use phonon/ skill (phonopy workflow is more standard)

QE ph.x / dynmat.x / matdyn.x output?
  --> Method B: Parse QE phonon output

Need element-projected phonon dispersion?
  --> Section C: Weight bands by atom participation

Need band sorting (avoid crossing artifacts)?
  --> Section D: Sort bands by eigenvector overlap
```

## Prerequisites

Pre-installed: `ase`, `pymatgen`, `numpy`, `scipy`, `matplotlib`.

For VASP: OUTCAR from a phonon calculation (IBRION=5/6/7/8).

For QE: Output from `ph.x`, `q2r.x`, `matdyn.x`, or `dynmat.x`.

Optional: `phonopy` (for comparison and additional analysis).

## Detailed Steps

### Method A: Parse Phonon Data from VASP OUTCAR

```python
#!/usr/bin/env python3
"""
Extract phonon frequencies, eigenvectors, and Born effective charges
from VASP OUTCAR (IBRION=5,6,7,8).
Compute projected phonon DOS and band structure.
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import re
import os

# ============================================================
# 1. CONFIGURATION
# ============================================================

OUTCAR_FILE = "OUTCAR"           # VASP OUTCAR from phonon calculation
POSCAR_FILE = "POSCAR"           # VASP POSCAR for structure info
OUTPUT_DIR = "phonon_from_outcar"

FREQ_UNIT = "THz"               # "THz", "cm-1", or "meV"
# Conversion: 1 THz = 33.3564 cm^-1 = 4.13567 meV

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ============================================================
# 2. PARSE OUTCAR FOR PHONON FREQUENCIES AND EIGENVECTORS
# ============================================================

def parse_outcar_phonons(outcar_file):
    """
    Parse VASP OUTCAR for phonon frequencies and eigenvectors.
    Works with IBRION=5 (finite differences) and IBRION=7/8 (DFPT).
    Returns:
        frequencies: array of shape (n_modes,) in THz
        eigenvectors: array of shape (n_modes, n_atoms, 3) - mass-weighted displacement patterns
        is_imaginary: boolean array indicating imaginary modes
    """
    frequencies = []
    eigenvectors = []
    is_imaginary = []
    n_atoms = None

    with open(outcar_file, "r") as f:
        lines = f.readlines()

    # Find number of atoms
    for line in lines:
        if "NIONS" in line:
            match = re.search(r"NIONS\s*=\s*(\d+)", line)
            if match:
                n_atoms = int(match.group(1))
                break

    if n_atoms is None:
        raise ValueError("Could not find NIONS in OUTCAR")

    print(f"  Number of ions: {n_atoms}")
    n_modes = 3 * n_atoms

    # Find phonon frequency block
    # VASP prints: "Eigenvectors and eigenvalues of the dynamical matrix"
    # followed by frequency lines and eigenvector blocks

    i = 0
    while i < len(lines):
        if "Eigenvectors and eigenvalues of the dynamical matrix" in lines[i]:
            i += 1
            # Skip separator lines
            while i < len(lines) and lines[i].strip() == "":
                i += 1
            if i < len(lines) and "---" in lines[i]:
                i += 1

            # Now parse each mode
            mode_count = 0
            while mode_count < n_modes and i < len(lines):
                # Skip blank lines
                while i < len(lines) and lines[i].strip() == "":
                    i += 1

                if i >= len(lines):
                    break

                line = lines[i].strip()

                # Frequency line format:
                # "   1 f  =    1.234567 THz    2.345678 2PiTHz ... meV"
                # or "   1 f/i=    1.234567 THz   ..." for imaginary
                freq_match = re.match(
                    r"\s*\d+\s+f(/i)?\s*=\s*([\d.]+)\s+THz", line
                )
                if freq_match:
                    imag = freq_match.group(1) is not None
                    freq_thz = float(freq_match.group(2))
                    if imag:
                        freq_thz = -freq_thz  # Convention: negative for imaginary
                    frequencies.append(freq_thz)
                    is_imaginary.append(imag)
                    i += 1

                    # Skip "X  Y  Z  dx  dy  dz" header
                    while i < len(lines) and ("dx" in lines[i] or "X" in lines[i] or lines[i].strip() == ""):
                        i += 1

                    # Read eigenvector components for this mode
                    eigvec = []
                    for atom_idx in range(n_atoms):
                        if i < len(lines):
                            parts = lines[i].split()
                            if len(parts) >= 6:
                                # Format: X Y Z dx dy dz
                                dx, dy, dz = float(parts[3]), float(parts[4]), float(parts[5])
                                eigvec.append([dx, dy, dz])
                            i += 1

                    if len(eigvec) == n_atoms:
                        eigenvectors.append(eigvec)
                    else:
                        print(f"  WARNING: Incomplete eigenvector for mode {mode_count + 1}")
                        eigenvectors.append([[0, 0, 0]] * n_atoms)

                    mode_count += 1
                else:
                    i += 1

            break  # Found the phonon block, done
        i += 1

    frequencies = np.array(frequencies)
    eigenvectors = np.array(eigenvectors)
    is_imaginary = np.array(is_imaginary)

    return frequencies, eigenvectors, is_imaginary


def parse_outcar_born_charges(outcar_file):
    """
    Parse Born effective charge tensors from OUTCAR.
    Returns: array of shape (n_atoms, 3, 3)
    """
    born_charges = []

    with open(outcar_file, "r") as f:
        lines = f.readlines()

    i = 0
    while i < len(lines):
        if "BORN EFFECTIVE CHARGES" in lines[i]:
            i += 1
            # Skip separator
            while i < len(lines) and ("---" in lines[i] or lines[i].strip() == ""):
                i += 1

            # Read each atom's Born charge tensor
            while i < len(lines):
                if "ion" in lines[i]:
                    i += 1
                    tensor = []
                    for row in range(3):
                        if i < len(lines):
                            parts = lines[i].split()
                            if len(parts) >= 4:
                                tensor.append([float(parts[1]), float(parts[2]), float(parts[3])])
                            i += 1
                    if len(tensor) == 3:
                        born_charges.append(tensor)
                elif "---" in lines[i] or lines[i].strip() == "":
                    if born_charges:
                        break
                    i += 1
                else:
                    i += 1
            break
        i += 1

    return np.array(born_charges) if born_charges else None


def parse_outcar_dielectric(outcar_file):
    """
    Parse the macroscopic dielectric tensor from OUTCAR.
    Returns: array of shape (3, 3)
    """
    with open(outcar_file, "r") as f:
        lines = f.readlines()

    for i, line in enumerate(lines):
        if "MACROSCOPIC STATIC DIELECTRIC TENSOR" in line and "including" not in line:
            # Next 3 lines after separator contain the tensor
            j = i + 1
            while j < len(lines) and "---" in lines[j]:
                j += 1
            tensor = []
            for row in range(3):
                if j < len(lines):
                    parts = lines[j].split()
                    if len(parts) >= 3:
                        tensor.append([float(x) for x in parts[:3]])
                    j += 1
            if len(tensor) == 3:
                return np.array(tensor)
    return None


print(f"=== Parsing VASP OUTCAR: {OUTCAR_FILE} ===")

frequencies, eigenvectors, is_imaginary = parse_outcar_phonons(OUTCAR_FILE)
print(f"  Found {len(frequencies)} phonon modes")

if len(frequencies) == 0:
    print("ERROR: No phonon frequencies found in OUTCAR.")
    print("  Ensure the calculation used IBRION=5, 6, 7, or 8.")
    raise SystemExit(1)

# Load structure for element info
from pymatgen.io.vasp import Poscar
try:
    poscar = Poscar.from_file(POSCAR_FILE)
    structure = poscar.structure
    symbols = [str(sp) for sp in structure.species]
    masses = [sp.atomic_mass for sp in structure.species]
    print(f"  Structure: {structure.composition.reduced_formula}")
except Exception:
    from ase.io import read as ase_read
    atoms = ase_read(POSCAR_FILE, format="vasp")
    symbols = atoms.get_chemical_symbols()
    masses = atoms.get_masses().tolist()
    print(f"  Structure: {atoms.get_chemical_formula()}")

unique_elements = sorted(set(symbols))
n_atoms = len(symbols)
n_modes = len(frequencies)

# Convert frequencies
if FREQ_UNIT == "cm-1":
    freq_display = frequencies * 33.3564
    freq_label = "Frequency (cm$^{-1}$)"
elif FREQ_UNIT == "meV":
    freq_display = frequencies * 4.13567
    freq_label = "Frequency (meV)"
else:
    freq_display = frequencies
    freq_label = "Frequency (THz)"

# Print frequencies
print(f"\n  Phonon frequencies ({FREQ_UNIT}):")
for i, (f, imag) in enumerate(zip(freq_display, is_imaginary)):
    marker = " (imaginary)" if imag else ""
    print(f"    Mode {i+1:3d}: {f:10.4f} {FREQ_UNIT}{marker}")

# Save frequencies
np.savetxt(os.path.join(OUTPUT_DIR, "frequencies.dat"),
           np.column_stack([np.arange(1, n_modes + 1), frequencies, freq_display, is_imaginary.astype(int)]),
           header=f"mode  freq_THz  freq_{FREQ_UNIT}  is_imaginary",
           fmt="%4d %12.6f %12.6f %2d")

# ============================================================
# 3. SECTION C: ELEMENT-PROJECTED PHONON ANALYSIS
# ============================================================

print("\n=== Element-Projected Analysis ===")


def compute_atom_participation(eigenvectors, symbols, masses):
    """
    Compute the participation ratio of each element in each mode.
    Weight by mass-weighted displacement squared: w_i = m_i * |e_i|^2
    Returns: dict of {element: array of shape (n_modes,)} with participation ratios (0 to 1).
    """
    n_modes = eigenvectors.shape[0]
    n_atoms = eigenvectors.shape[1]
    masses_arr = np.array(masses)

    unique_elements = sorted(set(symbols))
    participation = {el: np.zeros(n_modes) for el in unique_elements}

    for mode in range(n_modes):
        # Mass-weighted displacement squared for each atom
        disp_sq = np.sum(eigenvectors[mode] ** 2, axis=1)  # |e_i|^2
        weighted = masses_arr * disp_sq  # m_i * |e_i|^2
        total_weight = np.sum(weighted)

        if total_weight > 1e-30:
            for el in unique_elements:
                indices = [j for j, s in enumerate(symbols) if s == el]
                participation[el][mode] = np.sum(weighted[indices]) / total_weight

    return participation


if eigenvectors.shape[0] > 0 and eigenvectors.shape[1] == n_atoms:
    participation = compute_atom_participation(eigenvectors, symbols, masses)

    # Print participation for each mode
    print(f"\n  {'Mode':>6s} {'Freq':>10s}", end="")
    for el in unique_elements:
        print(f" {el:>8s}", end="")
    print()

    for i in range(min(n_modes, 30)):  # Print first 30 modes
        print(f"  {i+1:6d} {freq_display[i]:10.4f}", end="")
        for el in unique_elements:
            print(f" {participation[el][i]:8.3f}", end="")
        print()

    if n_modes > 30:
        print(f"  ... ({n_modes - 30} more modes)")

    # --- Plot: Projected phonon frequencies ---
    fig, ax = plt.subplots(figsize=(10, 6))

    mode_indices = np.arange(1, n_modes + 1)
    bottom = np.zeros(n_modes)
    colors = plt.cm.Set2(np.linspace(0, 1, len(unique_elements)))

    for idx, el in enumerate(unique_elements):
        heights = participation[el] * np.abs(freq_display)
        ax.bar(mode_indices, heights, bottom=bottom, width=0.8,
               color=colors[idx], alpha=0.7, label=el)
        bottom += heights

    ax.set_xlabel("Mode Index", fontsize=12)
    ax.set_ylabel(f"Weighted {freq_label}", fontsize=12)
    ax.set_title("Element-Projected Phonon Frequencies", fontsize=13)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(os.path.join(OUTPUT_DIR, "projected_phonon_freq.png"), dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\n  Saved: {OUTPUT_DIR}/projected_phonon_freq.png")

    # --- Plot: Frequency spectrum colored by dominant element ---
    fig, ax = plt.subplots(figsize=(8, 5))

    for i in range(n_modes):
        # Find dominant element
        max_el = max(unique_elements, key=lambda el: participation[el][i])
        el_idx = unique_elements.index(max_el)
        color = colors[el_idx]
        ax.plot(i + 1, freq_display[i], "o", color=color, markersize=6, alpha=0.7)

    # Legend
    for idx, el in enumerate(unique_elements):
        ax.plot([], [], "o", color=colors[idx], markersize=8, label=el)

    ax.axhline(y=0, color="gray", linestyle="--", linewidth=0.5)
    ax.set_xlabel("Mode Index", fontsize=12)
    ax.set_ylabel(freq_label, fontsize=12)
    ax.set_title("Phonon Frequencies (Colored by Dominant Element)", fontsize=13)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(OUTPUT_DIR, "freq_by_element.png"), dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {OUTPUT_DIR}/freq_by_element.png")

    # --- Plot: Projected DOS (Gaussian broadening) ---
    sigma = 0.3 if FREQ_UNIT == "THz" else 10.0 if FREQ_UNIT == "cm-1" else 1.0  # Broadening width
    freq_grid = np.linspace(min(freq_display) - 5 * sigma, max(freq_display) + 5 * sigma, 1000)

    fig, ax = plt.subplots(figsize=(8, 5))

    for idx, el in enumerate(unique_elements):
        dos_el = np.zeros_like(freq_grid)
        for i in range(n_modes):
            weight = participation[el][i]
            dos_el += weight * np.exp(-0.5 * ((freq_grid - freq_display[i]) / sigma) ** 2)
        dos_el /= (sigma * np.sqrt(2 * np.pi))
        ax.fill_between(freq_grid, dos_el, alpha=0.4, color=colors[idx], label=el)
        ax.plot(freq_grid, dos_el, "-", color=colors[idx], linewidth=1.2)

    ax.axvline(x=0, color="gray", linestyle="--", linewidth=0.5)
    ax.set_xlabel(freq_label, fontsize=12)
    ax.set_ylabel("Projected DOS (arb. units)", fontsize=12)
    ax.set_title("Element-Projected Phonon DOS (from Gamma-point)", fontsize=13)
    ax.legend(fontsize=10)
    ax.set_xlim(freq_grid[0], freq_grid[-1])
    ax.set_ylim(bottom=0)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(OUTPUT_DIR, "projected_dos.png"), dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {OUTPUT_DIR}/projected_dos.png")

    # Save participation data
    header = "mode freq_THz " + " ".join(f"weight_{el}" for el in unique_elements)
    data_cols = [np.arange(1, n_modes + 1), frequencies]
    for el in unique_elements:
        data_cols.append(participation[el])
    np.savetxt(os.path.join(OUTPUT_DIR, "atom_participation.dat"),
               np.column_stack(data_cols), header=header,
               fmt="%4d " + " %12.6f" * (1 + len(unique_elements)))

else:
    print("  WARNING: Eigenvectors not available or incomplete. Skipping projected analysis.")

# ============================================================
# 4. BORN EFFECTIVE CHARGES AND DIELECTRIC TENSOR
# ============================================================

print("\n=== Born Effective Charges and Dielectric Tensor ===")

born_charges = parse_outcar_born_charges(OUTCAR_FILE)
dielectric = parse_outcar_dielectric(OUTCAR_FILE)

if born_charges is not None:
    print(f"  Born effective charges for {len(born_charges)} atoms:")
    for i, (sym, Z) in enumerate(zip(symbols, born_charges)):
        Z_diag = np.diag(Z)
        print(f"    {sym:>4s} {i+1:3d}: Zxx={Z_diag[0]:7.3f}, Zyy={Z_diag[1]:7.3f}, Zzz={Z_diag[2]:7.3f}")

    np.save(os.path.join(OUTPUT_DIR, "born_charges.npy"), born_charges)
    print(f"  Saved: {OUTPUT_DIR}/born_charges.npy")
else:
    print("  No Born effective charges found (requires LEPSILON=.TRUE. or IBRION=7/8).")

if dielectric is not None:
    print(f"\n  Macroscopic dielectric tensor:")
    for row in dielectric:
        print(f"    [{row[0]:8.4f} {row[1]:8.4f} {row[2]:8.4f}]")
    np.save(os.path.join(OUTPUT_DIR, "dielectric_tensor.npy"), dielectric)
else:
    print("  No dielectric tensor found.")

# ============================================================
# 5. SECTION D: BAND SORTING BY EIGENVECTOR CONTINUITY
# ============================================================

print("\n=== Section D: Band Sorting (Eigenvector Continuity) ===")
print("  (Applicable when phonon bands are computed along a q-path)")
print("  This section demonstrates the algorithm for sorting bands to avoid crossing artifacts.")


def sort_bands_by_eigenvector_overlap(frequencies_qpath, eigenvectors_qpath):
    """
    Sort phonon bands along a q-path by eigenvector continuity.
    At each q-point, assign bands to maintain maximum overlap with the previous q-point's eigenvectors.

    Parameters:
        frequencies_qpath: (n_qpoints, n_bands) array of frequencies
        eigenvectors_qpath: (n_qpoints, n_bands, n_atoms, 3) array of eigenvectors

    Returns:
        sorted_frequencies: (n_qpoints, n_bands) with bands reordered for continuity
        sort_indices: (n_qpoints, n_bands) the permutation applied at each q-point
    """
    from scipy.optimize import linear_sum_assignment

    n_qpoints, n_bands = frequencies_qpath.shape
    sorted_freq = np.copy(frequencies_qpath)
    sort_indices = np.zeros((n_qpoints, n_bands), dtype=int)
    sort_indices[0] = np.arange(n_bands)

    for q in range(1, n_qpoints):
        # Compute overlap matrix between eigenvectors at q-1 and q
        # overlap[i,j] = |<e_i(q-1) | e_j(q)>|^2
        ev_prev = eigenvectors_qpath[q - 1][sort_indices[q - 1]].reshape(n_bands, -1)
        ev_curr = eigenvectors_qpath[q].reshape(n_bands, -1)

        overlap = np.abs(ev_prev @ ev_curr.T) ** 2

        # Use Hungarian algorithm to find optimal assignment
        # We want to maximize overlap, so minimize -overlap
        row_ind, col_ind = linear_sum_assignment(-overlap)

        sort_indices[q] = col_ind
        sorted_freq[q] = frequencies_qpath[q, col_ind]

    return sorted_freq, sort_indices


# Demonstrate with dummy data (replace with actual q-path data if available)
print("  Band sorting algorithm is available. Use it when you have phonon data along a q-path.")
print("  Example usage:")
print("    sorted_freq, indices = sort_bands_by_eigenvector_overlap(freq_qpath, eigvec_qpath)")
print("  This resolves band-crossing artifacts common in interpolated phonon dispersions.")

# ============================================================
# 6. SUMMARY
# ============================================================

n_imaginary = np.sum(is_imaginary)
n_acoustic = np.sum(np.abs(frequencies) < 0.5)  # modes near zero frequency

print(f"\n=== Summary ===")
print(f"  Total modes: {n_modes}")
print(f"  Acoustic modes (|f| < 0.5 THz): {n_acoustic}")
print(f"  Optical modes: {n_modes - n_acoustic}")
print(f"  Imaginary modes: {n_imaginary}")
if n_imaginary > 0:
    print(f"    Most negative frequency: {min(frequencies):.4f} THz")
    print(f"    Structure may be dynamically UNSTABLE.")
else:
    print(f"    Structure is dynamically STABLE.")
print(f"  Frequency range: {min(frequencies):.4f} to {max(frequencies):.4f} THz")
print(f"  All outputs in: {OUTPUT_DIR}/")
```

### Method B: Parse QE Phonon Output

```python
#!/usr/bin/env python3
"""
Parse and post-process Quantum ESPRESSO phonon output.
Handles dynmat.x (Gamma-point) and matdyn.x (interpolated dispersion) output.
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import os
import re

# ============================================================
# CONFIGURATION
# ============================================================

# Choose data source:
# Option 1: dynmat.x output (Gamma-point only)
DYNMAT_OUTPUT = ""              # e.g., "dynmat.out"

# Option 2: matdyn.x band structure
MATDYN_FREQ_FILE = ""           # e.g., "phonon_bands.freq"
MATDYN_MODES_FILE = ""          # e.g., "phonon_bands.modes" (for eigenvectors)

# Option 3: matdyn.x DOS
MATDYN_DOS_FILE = ""            # e.g., "phonon.dos"

# Option 4: phonon dispersion from ph.x dynamical matrices
FILDYN = ""                     # e.g., "phonon.dyn" (base name)

FREQ_UNIT = "THz"               # "THz", "cm-1", or "meV"
OUTPUT_DIR = "qe_phonon_postproc"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ============================================================
# PARSE DYNMAT.X OUTPUT (Gamma-point)
# ============================================================

def parse_dynmat_output(filename):
    """
    Parse dynmat.x output for Gamma-point frequencies and IR/Raman intensities.
    Returns dict with frequencies, IR intensities, Raman intensities.
    """
    frequencies = []
    ir_intensity = []
    raman_intensity = []

    with open(filename, "r") as f:
        lines = f.readlines()

    for line in lines:
        # Format: mode  freq(cm-1)  freq(THz)  IR  Raman
        match = re.match(r"\s*(\d+)\s+([\d.-]+)\s+([\d.-]+)\s+([\d.eE+-]+)\s+([\d.eE+-]+)", line)
        if match:
            freq_cm1 = float(match.group(2))
            freq_thz = float(match.group(3))
            ir_int = float(match.group(4))
            raman_int = float(match.group(5))
            frequencies.append(freq_thz)
            ir_intensity.append(ir_int)
            raman_intensity.append(raman_int)

    return {
        "frequencies_THz": np.array(frequencies),
        "frequencies_cm1": np.array(frequencies) * 33.3564,
        "IR_intensity": np.array(ir_intensity),
        "Raman_intensity": np.array(raman_intensity),
    }


if DYNMAT_OUTPUT and os.path.exists(DYNMAT_OUTPUT):
    print(f"=== Parsing dynmat.x output: {DYNMAT_OUTPUT} ===")
    dynmat_data = parse_dynmat_output(DYNMAT_OUTPUT)

    print(f"  Found {len(dynmat_data['frequencies_THz'])} modes")
    print(f"\n  {'Mode':>6s} {'Freq(THz)':>10s} {'Freq(cm-1)':>12s} {'IR':>10s} {'Raman':>10s}")
    for i, (f_thz, f_cm, ir, ra) in enumerate(zip(
        dynmat_data["frequencies_THz"], dynmat_data["frequencies_cm1"],
        dynmat_data["IR_intensity"], dynmat_data["Raman_intensity"]
    )):
        print(f"  {i+1:6d} {f_thz:10.4f} {f_cm:12.4f} {ir:10.4f} {ra:10.4f}")

    # Plot IR and Raman spectra
    sigma = 5.0  # Broadening in cm^-1
    freq_grid = np.linspace(0, max(dynmat_data["frequencies_cm1"]) * 1.1, 1000)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 8), sharex=True)

    ir_spectrum = np.zeros_like(freq_grid)
    raman_spectrum = np.zeros_like(freq_grid)
    for f, ir_int, ra_int in zip(
        dynmat_data["frequencies_cm1"], dynmat_data["IR_intensity"], dynmat_data["Raman_intensity"]
    ):
        if f > 0:
            gauss = np.exp(-0.5 * ((freq_grid - f) / sigma) ** 2)
            ir_spectrum += ir_int * gauss
            raman_spectrum += ra_int * gauss

    ax1.fill_between(freq_grid, ir_spectrum, alpha=0.4, color="steelblue")
    ax1.plot(freq_grid, ir_spectrum, "b-", linewidth=1.2)
    ax1.set_ylabel("IR Intensity", fontsize=12)
    ax1.set_title("IR Spectrum (from dynmat.x)", fontsize=13)
    ax1.grid(True, alpha=0.3)

    ax2.fill_between(freq_grid, raman_spectrum, alpha=0.4, color="coral")
    ax2.plot(freq_grid, raman_spectrum, "r-", linewidth=1.2)
    ax2.set_xlabel("Frequency (cm$^{-1}$)", fontsize=12)
    ax2.set_ylabel("Raman Intensity", fontsize=12)
    ax2.set_title("Raman Spectrum (from dynmat.x)", fontsize=13)
    ax2.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(os.path.join(OUTPUT_DIR, "ir_raman_spectra.png"), dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\n  Saved: {OUTPUT_DIR}/ir_raman_spectra.png")

# ============================================================
# PARSE MATDYN.X BAND STRUCTURE
# ============================================================

def parse_matdyn_bands(freq_file):
    """
    Parse matdyn.x frequency output for phonon band structure.
    Returns q-points and frequencies.
    """
    qpoints = []
    frequencies = []
    current_freqs = []

    with open(freq_file, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if "nbnd" in line and "nks" in line:
                # Header: nbnd=  XX, nks=  YY
                match = re.search(r"nbnd=\s*(\d+).*nks=\s*(\d+)", line)
                if match:
                    n_bands = int(match.group(1))
                    n_qpoints = int(match.group(2))
                continue

            parts = line.split()

            # Try to detect q-point line (starts with 2-3 floats as coordinates)
            # Q-point lines typically have exactly 3 or 4 numbers
            if len(parts) <= 4:
                try:
                    vals = [float(x) for x in parts]
                    if len(vals) >= 3 and all(abs(v) <= 2.0 for v in vals[:3]):
                        if current_freqs:
                            frequencies.append(current_freqs)
                        current_freqs = []
                        qpoints.append(vals[:3])
                        continue
                except ValueError:
                    pass

            # Frequency line
            try:
                freqs = [float(x) for x in parts]
                current_freqs.extend(freqs)
            except ValueError:
                pass

    if current_freqs:
        frequencies.append(current_freqs)

    return np.array(qpoints), np.array(frequencies)


if MATDYN_FREQ_FILE and os.path.exists(MATDYN_FREQ_FILE):
    print(f"\n=== Parsing matdyn.x bands: {MATDYN_FREQ_FILE} ===")
    qpoints, freq_bands = parse_matdyn_bands(MATDYN_FREQ_FILE)
    print(f"  Q-points: {len(qpoints)}, Bands: {freq_bands.shape[1] if len(freq_bands) > 0 else 0}")

    # freq_bands are in cm^-1 from matdyn.x
    if FREQ_UNIT == "THz":
        freq_plot = freq_bands * 0.02998  # cm^-1 to THz
    elif FREQ_UNIT == "meV":
        freq_plot = freq_bands * 0.12398  # cm^-1 to meV
    else:
        freq_plot = freq_bands

    # Compute q-path distance
    q_dist = np.zeros(len(qpoints))
    for i in range(1, len(qpoints)):
        q_dist[i] = q_dist[i - 1] + np.linalg.norm(qpoints[i] - qpoints[i - 1])

    # Plot
    fig, ax = plt.subplots(figsize=(8, 5))
    for band in range(freq_plot.shape[1]):
        ax.plot(q_dist, freq_plot[:, band], "b-", linewidth=0.8)

    ax.axhline(y=0, color="gray", linestyle="--", linewidth=0.5)
    ax.set_xlabel("q-path", fontsize=12)
    ax.set_ylabel(f"Frequency ({FREQ_UNIT})", fontsize=12)
    ax.set_title("QE Phonon Band Structure (matdyn.x)", fontsize=13)
    ax.set_xlim(q_dist[0], q_dist[-1])
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(OUTPUT_DIR, "qe_phonon_bands.png"), dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {OUTPUT_DIR}/qe_phonon_bands.png")

# ============================================================
# PARSE MATDYN.X DOS
# ============================================================

if MATDYN_DOS_FILE and os.path.exists(MATDYN_DOS_FILE):
    print(f"\n=== Parsing matdyn.x DOS: {MATDYN_DOS_FILE} ===")
    dos_data = np.loadtxt(MATDYN_DOS_FILE, comments="#")
    freq_dos = dos_data[:, 0]  # cm^-1
    dos_vals = dos_data[:, 1]

    if FREQ_UNIT == "THz":
        freq_dos_plot = freq_dos * 0.02998
    elif FREQ_UNIT == "meV":
        freq_dos_plot = freq_dos * 0.12398
    else:
        freq_dos_plot = freq_dos

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.fill_between(freq_dos_plot, dos_vals, alpha=0.4, color="steelblue")
    ax.plot(freq_dos_plot, dos_vals, "b-", linewidth=1.2)
    ax.set_xlabel(f"Frequency ({FREQ_UNIT})", fontsize=12)
    ax.set_ylabel("Phonon DOS", fontsize=12)
    ax.set_title("QE Phonon DOS (matdyn.x)", fontsize=13)
    ax.set_xlim(left=min(freq_dos_plot) - 1)
    ax.set_ylim(bottom=0)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(OUTPUT_DIR, "qe_phonon_dos.png"), dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {OUTPUT_DIR}/qe_phonon_dos.png")

print(f"\n=== Post-Processing Complete ===")
print(f"All outputs in: {OUTPUT_DIR}/")
```

## Key Parameters

| Parameter | Default | Notes |
|---|---|---|
| `FREQ_UNIT` | "THz" | Frequency display unit. 1 THz = 33.3564 cm^-1 = 4.13567 meV. |
| VASP `IBRION` | 5 or 7 | 5 = finite differences (forces), 6 = finite differences (stresses), 7 = DFPT, 8 = DFPT with IR/Raman. |
| VASP `NWRITE` | 3 | Set NWRITE=3 in INCAR to ensure eigenvectors are written to OUTCAR. |
| VASP `LEPSILON` | .TRUE. | Needed for Born effective charges and dielectric tensor (LO-TO splitting). |
| `sigma` (broadening) | 0.3 THz / 5 cm^-1 | Gaussian broadening for projected DOS plots. Adjust based on frequency range. |
| Band sorting | auto | Uses Hungarian algorithm for optimal eigenvector overlap matching between adjacent q-points. |

## Interpreting Results

### Frequencies from OUTCAR
- VASP reports 3N modes at the Gamma point (for IBRION=5/6/7/8 without QPOINTS).
- First 3 modes should be acoustic (near zero frequency). Non-zero acoustic modes indicate residual forces or numerical noise.
- Imaginary frequencies (labeled f/i in OUTCAR) indicate dynamical instability.

### Element-projected phonon DOS
- Heavy atoms dominate low-frequency acoustic modes.
- Light atoms (H, O, N) dominate high-frequency optical modes.
- In perovskites: B-site cation modes at intermediate frequencies, A-site at low frequencies.
- The projection at Gamma only is approximate. For full projected phonon DOS, use phonopy with `--pdos` flag.

### Band sorting
- Without sorting, phonon bands can appear to cross at points where they should anti-cross.
- Sorting by eigenvector overlap resolves this by tracking mode character along the q-path.
- The Hungarian algorithm finds the global optimal permutation (not greedy).

### Born effective charges
- Diagonal elements give the effective ionic charge for each direction.
- For ideal ionic compounds: Z* ~ formal charge.
- Large anomalous Born charges (Z* >> formal charge) indicate strong covalent-ionic hybridization and are hallmarks of ferroelectric instabilities.

### LO-TO splitting
- In polar materials, longitudinal optical (LO) and transverse optical (TO) modes split at the Gamma point.
- The splitting is proportional to Born charges and inversely proportional to dielectric constant.
- Only relevant for non-centrosymmetric or ionic materials.

## Common Issues

| Issue | Cause | Fix |
|---|---|---|
| No phonon data in OUTCAR | Wrong IBRION or calculation not converged | Ensure IBRION=5,6,7,8 and EDIFF is tight (1e-8 or smaller) |
| Eigenvectors missing | NWRITE < 3 in INCAR | Set NWRITE=3 to print eigenvectors |
| Only 3N modes, no q-dependence | VASP finite differences only compute Gamma | Use phonopy for full dispersion, or set QPOINTS in INCAR for IBRION=7/8 |
| Acoustic modes not at zero | Residual forces, ASR not applied | Apply acoustic sum rule in post-processing; tighten EDIFF and EDIFFG |
| Band crossings in plot | Bands not sorted by character | Apply band sorting (Section D) using eigenvector overlap |
| QE frequencies in wrong units | matdyn.x outputs in cm^-1 by default | Convert: 1 cm^-1 = 0.02998 THz = 0.12398 meV |
| Born charges not found | LEPSILON=.FALSE. or GGA+U | Set LEPSILON=.TRUE.; Born charges may not work with +U |
| Very large Born charges (> 10) | Strong hybridization or metallic-like screening | Physical for some ferroelectrics; check convergence with k-points and cutoff |
| matdyn.x parsing fails | Non-standard output format | Check QE version; parse format may differ between QE 6.x and 7.x |
