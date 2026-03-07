# XRD Pattern Simulation

## When to Use

- Simulate powder X-ray diffraction patterns for a known crystal structure.
- Compare predicted patterns of two phases (e.g., before/after a phase transition, polymorph identification).
- Identify which reflections (hkl indices) correspond to observed peaks.
- Estimate peak broadening from crystallite size using the Scherrer equation.
- Choose optimal wavelength (Cu K-alpha, Mo K-alpha, synchrotron) for a measurement.

## Prerequisites

- `pymatgen` (XRDCalculator, Structure)
- `matplotlib` (plotting)
- `numpy` (numerical operations)
- A crystal structure file (CIF, POSCAR) or structure built programmatically.

## Detailed Steps

### 1. Basic XRD pattern simulation

```python
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from pymatgen.core import Structure
from pymatgen.analysis.diffraction.xrd import XRDCalculator

# --- Load structure ---
# From a CIF file:
structure = Structure.from_file("my_structure.cif")
# Or from a POSCAR:
# structure = Structure.from_file("POSCAR")

# --- Simulate XRD with Cu K-alpha radiation ---
xrd_calc = XRDCalculator(wavelength="CuKa")  # 1.5406 Angstrom
pattern = xrd_calc.get_pattern(structure, two_theta_range=(10, 90))

# --- Plot ---
fig, ax = plt.subplots(figsize=(12, 5))
ax.stem(
    pattern.x, pattern.y,
    linefmt="b-", markerfmt=" ", basefmt="b-",
    use_line_collection=True,
)

# Label the strongest peaks with hkl indices
for i, (x, y, hkls) in enumerate(zip(pattern.x, pattern.y, pattern.hkls)):
    if y > 15:  # only label peaks above 15% relative intensity
        # hkls is a list of dicts; take the first hkl
        hkl = hkls[0]["hkl"]
        label = f"({hkl[0]}{hkl[1]}{hkl[2]})"
        ax.annotate(
            label, (x, y), textcoords="offset points",
            xytext=(0, 8), fontsize=7, ha="center", rotation=45,
        )

ax.set_xlabel(r"2$\theta$ (degrees)", fontsize=12)
ax.set_ylabel("Relative Intensity (%)", fontsize=12)
ax.set_title(f"XRD Pattern: {structure.composition.reduced_formula} (Cu K$\\alpha$)")
ax.set_xlim(10, 90)
ax.set_ylim(0, 110)
plt.tight_layout()
plt.savefig("xrd_pattern.png", dpi=150)
plt.close()
print("Saved: xrd_pattern.png")

# --- Print peak table ---
print(f"\n{'2theta':>8s} {'d-spacing':>10s} {'Intensity':>10s} {'hkl':>12s}")
print("-" * 44)
for x, y, d, hkls in zip(pattern.x, pattern.y, pattern.d_hkls, pattern.hkls):
    if y > 5:
        hkl = hkls[0]["hkl"]
        hkl_str = f"({hkl[0]} {hkl[1]} {hkl[2]})"
        print(f"{x:>8.3f} {d:>10.4f} {y:>10.1f} {hkl_str:>12s}")
```

### 2. Build a structure programmatically and compute XRD

```python
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pymatgen.core import Structure, Lattice
from pymatgen.analysis.diffraction.xrd import XRDCalculator

# --- Build NaCl (rock salt) structure ---
lattice = Lattice.cubic(5.64)  # a = 5.64 Angstrom
structure = Structure(
    lattice,
    ["Na", "Na", "Na", "Na", "Cl", "Cl", "Cl", "Cl"],
    [
        [0.0, 0.0, 0.0], [0.5, 0.5, 0.0], [0.5, 0.0, 0.5], [0.0, 0.5, 0.5],
        [0.5, 0.5, 0.5], [0.0, 0.0, 0.5], [0.0, 0.5, 0.0], [0.5, 0.0, 0.0],
    ],
)

# --- Compute XRD ---
xrd_calc = XRDCalculator(wavelength="CuKa")
pattern = xrd_calc.get_pattern(structure, two_theta_range=(10, 100))

fig, ax = plt.subplots(figsize=(12, 5))
ax.stem(pattern.x, pattern.y, linefmt="b-", markerfmt=" ", basefmt="b-",
        use_line_collection=True)

for x, y, hkls in zip(pattern.x, pattern.y, pattern.hkls):
    if y > 10:
        hkl = hkls[0]["hkl"]
        ax.annotate(f"({hkl[0]}{hkl[1]}{hkl[2]})", (x, y),
                     textcoords="offset points", xytext=(0, 8),
                     fontsize=7, ha="center", rotation=45)

ax.set_xlabel(r"2$\theta$ (degrees)")
ax.set_ylabel("Relative Intensity (%)")
ax.set_title("NaCl Powder XRD (Cu K$\\alpha$)")
ax.set_xlim(10, 100)
plt.tight_layout()
plt.savefig("xrd_NaCl.png", dpi=150)
plt.close()
print("Saved: xrd_NaCl.png")
```

### 3. Compare XRD patterns across different wavelengths

```python
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pymatgen.core import Structure
from pymatgen.analysis.diffraction.xrd import XRDCalculator

structure = Structure.from_file("my_structure.cif")

# --- Available built-in wavelengths ---
# "CuKa" = 1.5406 A (most common lab source)
# "MoKa" = 0.7107 A (used for absorbing samples, smaller 2theta range)
# "AgKa" = 0.5594 A
# "CoKa" = 1.7890 A (good for Fe-containing samples to avoid fluorescence)
# Custom wavelength: pass a float in Angstrom

wavelengths = {
    "Cu K$\\alpha$ (1.5406 A)":   "CuKa",
    "Mo K$\\alpha$ (0.7107 A)":   "MoKa",
    "Co K$\\alpha$ (1.7890 A)":   "CoKa",
    "Synchrotron (0.4 A)":        0.4,
}

fig, axes = plt.subplots(len(wavelengths), 1, figsize=(12, 3 * len(wavelengths)),
                          sharex=False)

for ax, (label, wl) in zip(axes, wavelengths.items()):
    xrd_calc = XRDCalculator(wavelength=wl)
    pattern = xrd_calc.get_pattern(structure, two_theta_range=(5, 90))

    ax.stem(pattern.x, pattern.y, linefmt="b-", markerfmt=" ", basefmt="b-",
            use_line_collection=True)
    ax.set_ylabel("I (%)")
    ax.set_title(label)
    ax.set_xlim(5, 90)
    ax.set_ylim(0, 110)

axes[-1].set_xlabel(r"2$\theta$ (degrees)")
plt.suptitle(f"XRD Patterns: {structure.composition.reduced_formula}", fontsize=14, y=1.01)
plt.tight_layout()
plt.savefig("xrd_wavelength_comparison.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved: xrd_wavelength_comparison.png")
```

### 4. Compare two phases (e.g., before/after phase transition)

```python
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pymatgen.core import Structure, Lattice
from pymatgen.analysis.diffraction.xrd import XRDCalculator

# --- Example: BCC Fe vs FCC Fe ---
bcc_fe = Structure(
    Lattice.cubic(2.87),
    ["Fe"], [[0, 0, 0]],
)

fcc_fe = Structure(
    Lattice.cubic(3.59),
    ["Fe", "Fe", "Fe", "Fe"],
    [[0, 0, 0], [0.5, 0.5, 0], [0.5, 0, 0.5], [0, 0.5, 0.5]],
)

xrd_calc = XRDCalculator(wavelength="CuKa")
pattern_bcc = xrd_calc.get_pattern(bcc_fe, two_theta_range=(20, 120))
pattern_fcc = xrd_calc.get_pattern(fcc_fe, two_theta_range=(20, 120))

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 7), sharex=True)

# BCC
ax1.stem(pattern_bcc.x, pattern_bcc.y, linefmt="b-", markerfmt=" ", basefmt="b-",
         use_line_collection=True)
for x, y, hkls in zip(pattern_bcc.x, pattern_bcc.y, pattern_bcc.hkls):
    if y > 10:
        hkl = hkls[0]["hkl"]
        ax1.annotate(f"({hkl[0]}{hkl[1]}{hkl[2]})", (x, y),
                      textcoords="offset points", xytext=(0, 8),
                      fontsize=8, ha="center")
ax1.set_ylabel("Intensity (%)")
ax1.set_title("BCC Fe (alpha-Fe)", fontsize=12)
ax1.set_ylim(0, 120)

# FCC
ax2.stem(pattern_fcc.x, pattern_fcc.y, linefmt="r-", markerfmt=" ", basefmt="r-",
         use_line_collection=True)
for x, y, hkls in zip(pattern_fcc.x, pattern_fcc.y, pattern_fcc.hkls):
    if y > 10:
        hkl = hkls[0]["hkl"]
        ax2.annotate(f"({hkl[0]}{hkl[1]}{hkl[2]})", (x, y),
                      textcoords="offset points", xytext=(0, 8),
                      fontsize=8, ha="center")
ax2.set_ylabel("Intensity (%)")
ax2.set_xlabel(r"2$\theta$ (degrees)")
ax2.set_title("FCC Fe (gamma-Fe)", fontsize=12)
ax2.set_ylim(0, 120)

plt.tight_layout()
plt.savefig("xrd_bcc_vs_fcc_Fe.png", dpi=150)
plt.close()
print("Saved: xrd_bcc_vs_fcc_Fe.png")
```

### 5. Simulate peak broadening with the Scherrer equation

The Scherrer equation relates peak width to crystallite size:

    FWHM (radians) = K * lambda / (D * cos(theta))

where K ~ 0.9, lambda is the wavelength, D is the crystallite diameter, and theta is the Bragg angle.

```python
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from pymatgen.core import Structure
from pymatgen.analysis.diffraction.xrd import XRDCalculator

structure = Structure.from_file("my_structure.cif")

xrd_calc = XRDCalculator(wavelength="CuKa")
pattern = xrd_calc.get_pattern(structure, two_theta_range=(10, 90))

wavelength = 1.5406  # Cu Ka in Angstrom
K = 0.9              # Scherrer constant


def apply_scherrer_broadening(two_theta_peaks, intensities, crystallite_size_nm,
                               wavelength_A, two_theta_range=(10, 90),
                               n_points=5000):
    """
    Apply Scherrer broadening to a stick pattern.

    Args:
        two_theta_peaks: array of peak positions in degrees
        intensities: array of peak intensities (relative, 0-100)
        crystallite_size_nm: crystallite diameter in nm
        wavelength_A: X-ray wavelength in Angstrom
        two_theta_range: tuple of (min, max) 2theta
        n_points: number of points in the output pattern

    Returns:
        two_theta_array, broadened_intensity
    """
    D = crystallite_size_nm * 10  # convert nm to Angstrom
    two_theta_array = np.linspace(two_theta_range[0], two_theta_range[1], n_points)
    broadened = np.zeros_like(two_theta_array)

    for peak_pos, intensity in zip(two_theta_peaks, intensities):
        theta_rad = np.radians(peak_pos / 2)
        # Scherrer FWHM in radians
        fwhm_rad = K * wavelength_A / (D * np.cos(theta_rad))
        # Convert FWHM to degrees
        fwhm_deg = np.degrees(fwhm_rad)
        # Gaussian sigma from FWHM
        sigma = fwhm_deg / (2 * np.sqrt(2 * np.log(2)))
        # Add Gaussian peak
        broadened += intensity * np.exp(
            -0.5 * ((two_theta_array - peak_pos) / sigma) ** 2
        )

    return two_theta_array, broadened


# --- Compare different crystallite sizes ---
sizes_nm = [5, 10, 30, 100]  # nanometers

fig, ax = plt.subplots(figsize=(12, 6))

for size in sizes_nm:
    tt, intensity = apply_scherrer_broadening(
        pattern.x, pattern.y, size, wavelength
    )
    # Normalize
    intensity = intensity / intensity.max() * 100
    ax.plot(tt, intensity, label=f"D = {size} nm", linewidth=1.2)

# Add stick pattern for reference
ax.stem(pattern.x, pattern.y, linefmt="k-", markerfmt=" ", basefmt=" ",
        use_line_collection=True, label="Ideal (infinite)")

ax.set_xlabel(r"2$\theta$ (degrees)", fontsize=12)
ax.set_ylabel("Relative Intensity (%)", fontsize=12)
ax.set_title(
    f"Effect of Crystallite Size on XRD: "
    f"{structure.composition.reduced_formula}",
    fontsize=13,
)
ax.legend()
ax.set_xlim(10, 90)
ax.set_ylim(0, 115)
plt.tight_layout()
plt.savefig("xrd_scherrer_broadening.png", dpi=150)
plt.close()
print("Saved: xrd_scherrer_broadening.png")
```

### 6. Fetch structure from Materials Project and compute XRD

```python
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mp_api.client import MPRester
from pymatgen.analysis.diffraction.xrd import XRDCalculator

api_key = os.environ.get("MP_API_KEY")

with MPRester(api_key) as mpr:
    # Fetch a specific material (e.g., TiO2 rutile)
    docs = mpr.materials.summary.search(
        formula="TiO2",
        fields=["material_id", "structure", "symmetry"],
    )
    # Pick the first result
    structure = docs[0].structure
    mat_id = docs[0].material_id
    spacegroup = docs[0].symmetry.symbol

print(f"Material: {mat_id}, Space group: {spacegroup}")

xrd_calc = XRDCalculator(wavelength="CuKa")
pattern = xrd_calc.get_pattern(structure, two_theta_range=(10, 90))

fig, ax = plt.subplots(figsize=(12, 5))
ax.stem(pattern.x, pattern.y, linefmt="b-", markerfmt=" ", basefmt="b-",
        use_line_collection=True)

for x, y, hkls in zip(pattern.x, pattern.y, pattern.hkls):
    if y > 15:
        hkl = hkls[0]["hkl"]
        ax.annotate(f"({hkl[0]}{hkl[1]}{hkl[2]})", (x, y),
                     textcoords="offset points", xytext=(0, 8),
                     fontsize=7, ha="center", rotation=45)

ax.set_xlabel(r"2$\theta$ (degrees)")
ax.set_ylabel("Relative Intensity (%)")
ax.set_title(f"XRD: TiO2 ({mat_id}, {spacegroup}) - Cu K$\\alpha$")
ax.set_xlim(10, 90)
plt.tight_layout()
plt.savefig("xrd_TiO2_from_MP.png", dpi=150)
plt.close()
print("Saved: xrd_TiO2_from_MP.png")
```

## Key Parameters

| Parameter | Description |
|-----------|-------------|
| `wavelength` | X-ray wavelength. Built-in: `"CuKa"` (1.5406 A), `"MoKa"` (0.7107 A), `"CoKa"` (1.7890 A), `"AgKa"` (0.5594 A), `"CrKa"` (2.2910 A), `"FeKa"` (1.9373 A). Or pass a float in Angstrom for synchrotron. |
| `two_theta_range` | Tuple `(min, max)` in degrees. Default is `(0, 90)`. Extend for high-angle peaks; narrow for focused analysis. |
| `debye_waller_factors` | Dictionary of element -> B factor (A^2) for thermal displacement. Reduces peak intensity at high angles. Example: `{"Si": 0.5, "O": 0.7}`. |
| Scherrer constant K | Typically 0.89-0.94 depending on crystallite shape. Use 0.9 as default. |
| Crystallite size D | Affects peak width. Below ~100 nm, broadening becomes visible. Below ~5 nm, peaks overlap significantly. |

## Interpreting Results

- **Peak positions** (2-theta): Determined by the lattice parameters and symmetry. Shifts indicate changes in unit cell dimensions (strain, composition change, thermal expansion).
- **Peak intensities**: Determined by atomic positions, site occupancies, and atomic scattering factors. Intensity ratios identify the structure type.
- **Systematic absences**: Missing reflections indicate symmetry elements (glide planes, screw axes). These determine the space group.
- **Peak broadening**: Broad peaks indicate small crystallite size (Scherrer broadening) or microstrain. Instrumental broadening also contributes.
- **Comparing two patterns**: Matching peak positions confirms the same crystal structure. New peaks indicate a phase transformation or secondary phase. Peak shifts indicate compositional or strain changes.

## Common Issues

| Issue | Solution |
|-------|----------|
| No peaks appear | Check that `two_theta_range` is appropriate for the wavelength. Shorter wavelengths compress the pattern to lower angles. |
| Peaks at wrong positions | Verify the lattice parameters in your structure. Check units (pymatgen expects Angstrom). |
| `hkls` attribute empty or confusing | `pattern.hkls` is a list of lists of dicts. Each peak may have multiple overlapping hkl reflections. Access via `pattern.hkls[i][0]["hkl"]`. |
| Structure from relaxation has wrong symmetry | MACE relaxation may break symmetry slightly. Use `SpacegroupAnalyzer` to symmetrize before computing XRD. |
| Too many peaks labeled | Increase the intensity threshold in the annotation loop (e.g., `y > 20`). |
| `use_line_collection` deprecation warning | This parameter may be removed in future matplotlib versions. Remove it if you get a warning; it only suppresses a performance warning in older versions. |
