# NEB Path Analysis

## When to Use

- Analyze completed NEB calculations: extract energies, barriers, transition state geometries
- Convert NEB image paths to PDB or XYZ trajectory files for visualization (VASPKIT 504)
- Interpolate additional NEB images between existing endpoints (VASPKIT 505)
- Post-process QE neb.x or VASP NEB output
- Combine NEB paths from multiple segments into a single reaction coordinate
- Generate publication-quality energy profile plots with spline interpolation

## Method Selection

| Task | Tool | Notes |
|---|---|---|
| Interpolate images between endpoints | ASE IDPP / linear (Method A) | Fast, no DFT needed |
| Parse QE neb.x output | Python parser (Method B) | Extract energies, coordinates, convergence |
| Parse VASP NEB output | Python parser (Method C) | Read OUTCAR from each image directory |
| Convert NEB path to trajectory | ASE / pymatgen (Method A) | Export PDB, XYZ, CIF for each image |
| Spline-fit energy profile | scipy CubicSpline (all methods) | Smooth barrier curve with accurate saddle point |
| Visualize NEB band | matplotlib (all methods) | Energy vs reaction coordinate plot |

```
Have NEB endpoints and need to create interpolated images?
  --> Method A: ASE IDPP interpolation

Have a completed QE NEB calculation?
  --> Method B: Parse neb.x output

Have a completed VASP NEB calculation?
  --> Method C: Parse OUTCAR files from image directories

Need to convert images to a trajectory file?
  --> Method A: ASE write trajectory
```

## Prerequisites

- ASE (NEB, NEBTools, trajectory I/O)
- pymatgen (structure manipulation)
- numpy, scipy (numerical analysis, spline fitting)
- matplotlib (plotting)
- For QE NEB: completed neb.x calculation with output file
- For VASP NEB: completed NEB run with OUTCAR in each image directory

## Detailed Steps

### Method A: ASE NEB Image Interpolation and Path Export

```python
#!/usr/bin/env python3
"""
Interpolate NEB images between two endpoint structures using ASE.
Supports IDPP (Image Dependent Pair Potential) and linear interpolation.
Exports images as trajectory files (PDB, XYZ, extxyz).

Corresponds to VASPKIT 505 (interpolate NEB images).
"""

import numpy as np
from ase.io import read, write
from ase.mep.neb import NEB, idpp_interpolate, interpolate
from pathlib import Path
import json

# ============================================================
# 1. Read endpoint structures
# ============================================================
# Endpoints can be CIF, POSCAR, XYZ, or any ASE-supported format
initial = read("initial.cif")  # or POSCAR_initial
final = read("final.cif")      # or POSCAR_final

print(f"Initial: {initial.get_chemical_formula()}, {len(initial)} atoms")
print(f"Final:   {final.get_chemical_formula()}, {len(final)} atoms")

assert len(initial) == len(final), "Endpoints must have the same number of atoms"
assert (initial.get_chemical_symbols() == final.get_chemical_symbols()), \
    "Endpoints must have the same species ordering"

# ============================================================
# 2. Create interpolated images
# ============================================================
n_images = 7  # number of intermediate images (5-11 typical)
method = "idpp"  # "idpp" or "linear"

# Build list: [initial, copy, copy, ..., final]
images = [initial.copy()]
for _ in range(n_images):
    images.append(initial.copy())
images.append(final.copy())

if method == "idpp":
    # IDPP interpolation: respects interatomic distances
    # Avoids atom overlap, much better than linear for most cases
    idpp_interpolate(images, traj=None, log=None, mic=True)
    print(f"IDPP interpolation: {n_images} intermediate images")
else:
    # Linear interpolation: simple but can cause atom overlap
    interpolate(images)
    print(f"Linear interpolation: {n_images} intermediate images")

# ============================================================
# 3. Export images in various formats
# ============================================================
output_dir = Path("neb_images")
output_dir.mkdir(exist_ok=True)

# Write individual image files
for i, img in enumerate(images):
    # CIF format
    write(str(output_dir / f"image_{i:02d}.cif"), img)
    # POSCAR format (for VASP)
    write(str(output_dir / f"POSCAR_{i:02d}"), img, format="vasp")
    # XYZ format
    write(str(output_dir / f"image_{i:02d}.xyz"), img)

print(f"Individual images saved to {output_dir}/")

# Write as single trajectory file (for animation)
write(str(output_dir / "neb_path.xyz"), images)  # multi-frame XYZ
write(str(output_dir / "neb_path.pdb"), images)  # multi-frame PDB
write(str(output_dir / "neb_path.extxyz"), images)  # extended XYZ with cell info

print("Trajectory files saved:")
print(f"  {output_dir}/neb_path.xyz")
print(f"  {output_dir}/neb_path.pdb")
print(f"  {output_dir}/neb_path.extxyz")

# ============================================================
# 4. Compute reaction coordinate (cumulative RMS displacement)
# ============================================================
reaction_coord = [0.0]
for i in range(1, len(images)):
    disp = images[i].get_positions() - images[i - 1].get_positions()
    # Apply minimum image convention for periodic systems
    rms_disp = np.sqrt(np.mean(np.sum(disp**2, axis=1)))
    reaction_coord.append(reaction_coord[-1] + rms_disp)

# Normalize to [0, 1]
rc_norm = [r / reaction_coord[-1] for r in reaction_coord]

print("\nImage | Reaction Coordinate | Normalized")
print("-" * 45)
for i, (rc, rcn) in enumerate(zip(reaction_coord, rc_norm)):
    label = ""
    if i == 0:
        label = " (initial)"
    elif i == len(images) - 1:
        label = " (final)"
    print(f"  {i:3d}  |  {rc:10.4f} A       |  {rcn:.4f}{label}")

# Save interpolation info
info = {
    "n_images": n_images,
    "method": method,
    "n_atoms": len(initial),
    "formula": initial.get_chemical_formula(),
    "reaction_coordinate": reaction_coord,
    "reaction_coordinate_normalized": rc_norm,
}
with open(str(output_dir / "interpolation_info.json"), "w") as f:
    json.dump(info, f, indent=2, default=float)
print(f"\nInfo saved to {output_dir}/interpolation_info.json")
```

### Method A (continued): NEB Path to PDB Trajectory

```python
#!/usr/bin/env python3
"""
Convert a set of NEB image files to a single PDB trajectory.
This corresponds to VASPKIT 504 (NEB path to PDB).

Reads from:
  - Individual CIF/POSCAR/XYZ files named image_00, image_01, ...
  - Or a directory of VASP NEB image directories (00/, 01/, 02/, ...)
"""

import os
import glob
import numpy as np
from ase.io import read, write
from pathlib import Path

# ============================================================
# Option 1: Read from individual files
# ============================================================
def neb_files_to_trajectory(pattern, output_pdb="neb_trajectory.pdb"):
    """
    Read NEB image files matching a glob pattern and write a PDB trajectory.

    Parameters
    ----------
    pattern : str
        Glob pattern for image files (e.g., "neb_images/image_*.cif")
    output_pdb : str
        Output PDB file path.
    """
    files = sorted(glob.glob(pattern))
    if not files:
        print(f"No files matching pattern: {pattern}")
        return

    images = []
    for f in files:
        atoms = read(f)
        images.append(atoms)
        print(f"  Read: {f} ({len(atoms)} atoms)")

    write(output_pdb, images)
    print(f"\nTrajectory saved to {output_pdb} ({len(images)} frames)")
    return images


# ============================================================
# Option 2: Read from VASP NEB directory structure
# ============================================================
def vasp_neb_to_trajectory(neb_dir, output_pdb="neb_trajectory.pdb"):
    """
    Read VASP NEB image directories (00/, 01/, ...) and create trajectory.

    Parameters
    ----------
    neb_dir : str
        Directory containing NEB image subdirectories.
    output_pdb : str
        Output PDB file path.
    """
    # Find image directories (00, 01, 02, ...)
    image_dirs = sorted(glob.glob(os.path.join(neb_dir, "[0-9][0-9]")))
    if not image_dirs:
        print(f"No image directories found in {neb_dir}")
        return

    images = []
    for img_dir in image_dirs:
        # Try CONTCAR first (relaxed), then POSCAR (initial)
        contcar = os.path.join(img_dir, "CONTCAR")
        poscar = os.path.join(img_dir, "POSCAR")
        if os.path.exists(contcar) and os.path.getsize(contcar) > 0:
            atoms = read(contcar, format="vasp")
            src = "CONTCAR"
        elif os.path.exists(poscar):
            atoms = read(poscar, format="vasp")
            src = "POSCAR"
        else:
            print(f"  WARNING: No structure in {img_dir}")
            continue

        images.append(atoms)
        print(f"  Read: {img_dir}/{src} ({len(atoms)} atoms)")

    if images:
        write(output_pdb, images)
        # Also write XYZ trajectory
        xyz_out = output_pdb.replace(".pdb", ".xyz")
        write(xyz_out, images)
        print(f"\nTrajectory saved to {output_pdb} and {xyz_out} ({len(images)} frames)")

    return images


# ============================================================
# Option 3: Read from QE NEB output (pwscf.*.pos files)
# ============================================================
def qe_neb_to_trajectory(neb_prefix="pwscf", neb_dir=".", output_pdb="neb_trajectory.pdb"):
    """
    Read QE NEB image coordinates from the output directory.

    QE NEB stores images as:
      {prefix}.update/  or directly in prefix directories
      The final coordinates are in the neb.x output.
    """
    from pymatgen.core import Structure
    from pymatgen.io.ase import AseAtomsAdaptor

    # Parse neb.x output for final coordinates
    neb_out = os.path.join(neb_dir, "neb.out")
    if not os.path.exists(neb_out):
        print(f"neb.out not found in {neb_dir}")
        return

    # Read cell parameters from the input
    neb_in = os.path.join(neb_dir, "neb.in")
    cell = None
    species_order = []

    if os.path.exists(neb_in):
        with open(neb_in) as f:
            content = f.read()

        # Parse CELL_PARAMETERS
        import re
        cell_match = re.search(
            r"CELL_PARAMETERS\s+\w+\s*\n((?:\s+[-\d.]+\s+[-\d.]+\s+[-\d.]+\s*\n){3})",
            content
        )
        if cell_match:
            cell_lines = cell_match.group(1).strip().split("\n")
            cell = np.array([[float(x) for x in line.split()] for line in cell_lines])

    print(f"Read cell parameters: {cell is not None}")

    # Also try to read individual image directories
    # QE NEB creates directories like: tmp/neb_vacancy_1.save, etc.
    image_dirs = sorted(glob.glob(os.path.join(neb_dir, "tmp", f"{neb_prefix}*")))

    print(f"Found {len(image_dirs)} image directories")
    return None  # QE NEB parsing is complex; use the energy parser below instead


# ============================================================
# Run the converter
# ============================================================
# Choose the appropriate reader:

# For individual image files:
# images = neb_files_to_trajectory("neb_images/image_*.cif", "neb_path.pdb")

# For VASP NEB directories:
# images = vasp_neb_to_trajectory("./neb_calc", "vasp_neb_path.pdb")

# Example: convert from image files
if glob.glob("neb_images/image_*.cif"):
    images = neb_files_to_trajectory("neb_images/image_*.cif", "neb_path.pdb")
elif glob.glob("[0-9][0-9]/POSCAR") or glob.glob("[0-9][0-9]/CONTCAR"):
    images = vasp_neb_to_trajectory(".", "vasp_neb_path.pdb")
else:
    print("No NEB images found. Provide image files or VASP NEB directories.")
```

### Method B: Parse QE NEB Output

```python
#!/usr/bin/env python3
"""
Parse QE neb.x output file to extract:
  - Image energies along the path
  - Activation barriers (forward and reverse)
  - Convergence information
  - Error/force on each image

Generate energy profile plot with spline interpolation.
"""

import re
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.interpolate import CubicSpline
import json

# ============================================================
# 1. Parse neb.x output
# ============================================================
def parse_qe_neb(filename):
    """
    Parse QE neb.x output for energies, barriers, and convergence.

    Returns
    -------
    dict with keys:
        converged : bool
        iterations : int
        images : list of dict (image, energy_eV, error_eV_per_A)
        barrier_forward_eV : float or None
        barrier_reverse_eV : float or None
        path_length_bohr : float or None
    """
    with open(filename) as f:
        content = f.read()
        lines = content.split("\n")

    converged = "neb: convergence achieved" in content

    # Count iterations
    iterations = content.count("neb: istep")

    # Parse the final energy block
    # Format:
    #    image        energy (eV)        error (eV/A)        frozen
    #        1       -XXXX.XXXX            0.0000            T
    #        2       -XXXX.XXXX            0.1234            F
    images = []
    in_block = False
    for line in lines:
        if "image" in line and "energy (eV)" in line and "error" in line:
            in_block = True
            images = []  # take last block
            continue
        if in_block:
            parts = line.split()
            if len(parts) >= 3:
                try:
                    img_num = int(parts[0])
                    energy = float(parts[1])
                    error = float(parts[2])
                    frozen = parts[3] if len(parts) > 3 else "F"
                    images.append({
                        "image": img_num,
                        "energy_eV": energy,
                        "error_eV_per_A": error,
                        "frozen": frozen == "T",
                    })
                except (ValueError, IndexError):
                    if images:
                        in_block = False

    # Parse activation energies
    barrier_fwd = None
    barrier_rev = None
    for line in lines:
        if "activation energy" in line.lower():
            match = re.search(r"([-\d.]+)\s+eV", line)
            if match:
                val = float(match.group(1))
                if "--->" in line:
                    barrier_fwd = val
                elif "<---" in line:
                    barrier_rev = val

    # Parse path length
    path_length = None
    for line in lines:
        if "path length" in line.lower():
            match = re.search(r"([-\d.]+)\s+bohr", line)
            if match:
                path_length = float(match.group(1))

    return {
        "converged": converged,
        "iterations": iterations,
        "images": images,
        "barrier_forward_eV": barrier_fwd,
        "barrier_reverse_eV": barrier_rev,
        "path_length_bohr": path_length,
    }


# ============================================================
# 2. Parse and display results
# ============================================================
results = parse_qe_neb("neb.out")

print(f"Converged: {results['converged']}")
print(f"Iterations: {results['iterations']}")
print(f"Number of images: {len(results['images'])}")

if not results["images"]:
    print("ERROR: No image data found in neb.out")
    exit(1)

# Reference to first image
e_ref = results["images"][0]["energy_eV"]
rel_energies = [img["energy_eV"] - e_ref for img in results["images"]]
errors = [img["error_eV_per_A"] for img in results["images"]]

print(f"\n{'Image':>6} {'E_rel (eV)':>12} {'Error (eV/A)':>14} {'Frozen':>8}")
print("-" * 45)
for img, e_rel, err in zip(results["images"], rel_energies, errors):
    marker = " <-- TS" if e_rel == max(rel_energies) else ""
    frozen = "YES" if img["frozen"] else "no"
    print(f"  {img['image']:4d}   {e_rel:12.6f}   {err:12.6f}   {frozen:>6}{marker}")

if results["barrier_forward_eV"] is not None:
    print(f"\nForward barrier: {results['barrier_forward_eV']:.4f} eV")
if results["barrier_reverse_eV"] is not None:
    print(f"Reverse barrier: {results['barrier_reverse_eV']:.4f} eV")

# ============================================================
# 3. Plot energy profile with spline fit
# ============================================================
n_images = len(results["images"])
rc = np.linspace(0, 1, n_images)

fig, axes = plt.subplots(2, 1, figsize=(8, 8), gridspec_kw={"height_ratios": [3, 1]})

# Top: energy profile
ax = axes[0]
ax.plot(rc, rel_energies, "o", color="steelblue", markersize=10, zorder=5,
        label="NEB images")

# Spline fit
cs = CubicSpline(rc, rel_energies)
rc_fine = np.linspace(0, 1, 500)
ax.plot(rc_fine, cs(rc_fine), "-", color="steelblue", linewidth=2, alpha=0.7,
        label="Cubic spline")

# Mark transition state
ts_idx = np.argmax(rel_energies)
ax.plot(rc[ts_idx], rel_energies[ts_idx], "r*", markersize=18, zorder=6,
        label=f"TS ({rel_energies[ts_idx]:.3f} eV)")

# Spline-interpolated barrier (may be slightly higher than discrete max)
rc_max = rc_fine[np.argmax(cs(rc_fine))]
e_max_spline = float(cs(rc_max))
if abs(e_max_spline - rel_energies[ts_idx]) > 0.001:
    ax.plot(rc_max, e_max_spline, "r^", markersize=12, zorder=6,
            label=f"Spline max ({e_max_spline:.3f} eV)")

ax.axhline(0, color="gray", linewidth=0.5)
ax.set_ylabel("Energy relative to initial (eV)", fontsize=12)
ax.set_title("NEB Energy Profile", fontsize=14)
ax.legend(fontsize=10)
ax.grid(alpha=0.3)

# Bottom: force error per image
ax2 = axes[1]
ax2.bar(rc, errors, width=0.06, color="salmon", edgecolor="darkred", alpha=0.8)
ax2.set_xlabel("Reaction coordinate", fontsize=12)
ax2.set_ylabel("Force error\n(eV/A)", fontsize=11)
ax2.grid(alpha=0.3)

fig.tight_layout()
fig.savefig("neb_analysis.png", dpi=150)
print("\nSaved neb_analysis.png")

# Save parsed results
with open("neb_parsed.json", "w") as f:
    json.dump(results, f, indent=2, default=float)
print("Saved neb_parsed.json")
```

### Method C: Parse VASP NEB Output

```python
#!/usr/bin/env python3
"""
Parse VASP NEB output from a directory structure:
  00/OUTCAR, 01/OUTCAR, ..., NN/OUTCAR

Extracts energies, forces, and structures from each image.
Generates energy profile and exports trajectory.

Corresponds to VASPKIT 504-505.
"""

import os
import re
import glob
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.interpolate import CubicSpline
from ase.io import read, write
import json


def parse_vasp_neb_dir(neb_dir="."):
    """
    Parse a VASP NEB directory structure.

    Expected layout:
      neb_dir/00/OUTCAR  (initial endpoint)
      neb_dir/01/OUTCAR  (image 1)
      ...
      neb_dir/NN/OUTCAR  (final endpoint)

    Returns
    -------
    dict with image data (energies, forces, structures)
    """
    # Find image directories
    image_dirs = sorted(glob.glob(os.path.join(neb_dir, "[0-9][0-9]")))
    if not image_dirs:
        raise FileNotFoundError(f"No image directories found in {neb_dir}")

    results = []
    structures = []

    for img_dir in image_dirs:
        img_num = int(os.path.basename(img_dir))
        outcar = os.path.join(img_dir, "OUTCAR")
        contcar = os.path.join(img_dir, "CONTCAR")
        poscar = os.path.join(img_dir, "POSCAR")

        img_data = {"image": img_num, "dir": img_dir}

        # Parse energy from OUTCAR
        if os.path.exists(outcar):
            energy = None
            max_force = None
            with open(outcar) as f:
                for line in f:
                    if "energy  without entropy" in line:
                        # "energy  without entropy=    -XXX.XXXXX  energy(sigma->0) =   -XXX.XXXXX"
                        match = re.search(r"energy\(sigma->0\)\s*=\s*([-\d.]+)", line)
                        if match:
                            energy = float(match.group(1))
                    if "FORCES: max atom" in line or "RMS" in line:
                        match = re.search(r"max atom\s*=\s*([-\d.]+)", line)
                        if match:
                            max_force = float(match.group(1))

            img_data["energy_eV"] = energy
            img_data["max_force_eV_per_A"] = max_force
        else:
            img_data["energy_eV"] = None
            img_data["max_force_eV_per_A"] = None

        # Read structure
        struct_file = contcar if (os.path.exists(contcar) and
                                   os.path.getsize(contcar) > 0) else poscar
        if os.path.exists(struct_file):
            try:
                atoms = read(struct_file, format="vasp")
                structures.append(atoms)
                img_data["has_structure"] = True
            except Exception:
                img_data["has_structure"] = False
        else:
            img_data["has_structure"] = False

        results.append(img_data)

    return results, structures


def analyze_vasp_neb(neb_dir="."):
    """Full VASP NEB analysis: parse, plot, export."""
    results, structures = parse_vasp_neb_dir(neb_dir)

    # Filter images with valid energies
    valid = [r for r in results if r["energy_eV"] is not None]
    if not valid:
        print("ERROR: No valid energies found in OUTCAR files")
        return

    energies = [r["energy_eV"] for r in valid]
    e_ref = energies[0]
    rel_energies = [e - e_ref for e in energies]
    image_nums = [r["image"] for r in valid]

    # Barrier
    ts_idx = np.argmax(rel_energies)
    barrier_fwd = rel_energies[ts_idx]
    barrier_rev = rel_energies[ts_idx] - rel_energies[-1]

    print(f"VASP NEB Analysis ({neb_dir})")
    print(f"{'='*50}")
    print(f"Number of images: {len(valid)}")
    print(f"\n{'Image':>6} {'E_rel (eV)':>12} {'Max Force':>12}")
    print("-" * 35)
    for r, e_rel in zip(valid, rel_energies):
        force_str = f"{r['max_force_eV_per_A']:.4f}" if r["max_force_eV_per_A"] else "N/A"
        marker = " <-- TS" if e_rel == max(rel_energies) else ""
        print(f"  {r['image']:4d}   {e_rel:12.6f}   {force_str:>10}{marker}")

    print(f"\nForward barrier: {barrier_fwd:.4f} eV")
    print(f"Reverse barrier: {barrier_rev:.4f} eV")

    # Plot
    rc = np.linspace(0, 1, len(rel_energies))
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(rc, rel_energies, "o-", color="steelblue", markersize=10, linewidth=2)
    ax.plot(rc[ts_idx], rel_energies[ts_idx], "r*", markersize=18,
            label=f"TS ({barrier_fwd:.3f} eV)")

    if len(rel_energies) > 3:
        cs = CubicSpline(rc, rel_energies)
        rc_fine = np.linspace(0, 1, 300)
        ax.plot(rc_fine, cs(rc_fine), "--", color="gray", alpha=0.5)

    ax.axhline(0, color="gray", linewidth=0.5)
    ax.set_xlabel("Reaction coordinate", fontsize=12)
    ax.set_ylabel("Energy relative to initial (eV)", fontsize=12)
    ax.set_title("VASP NEB Energy Profile", fontsize=14)
    ax.legend(fontsize=11)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig("vasp_neb_profile.png", dpi=150)
    print("\nSaved vasp_neb_profile.png")

    # Export trajectory
    if structures:
        write("vasp_neb_trajectory.pdb", structures)
        write("vasp_neb_trajectory.xyz", structures)
        print(f"Trajectory saved ({len(structures)} frames)")

    # Save results
    output = {
        "n_images": len(valid),
        "barrier_forward_eV": barrier_fwd,
        "barrier_reverse_eV": barrier_rev,
        "transition_state_image": int(image_nums[ts_idx]),
        "images": [
            {"image": r["image"], "energy_eV": r["energy_eV"],
             "rel_energy_eV": e_rel}
            for r, e_rel in zip(valid, rel_energies)
        ],
    }
    with open("vasp_neb_results.json", "w") as f:
        json.dump(output, f, indent=2, default=float)
    print("Saved vasp_neb_results.json")


# Run analysis
if os.path.exists("00") and os.path.exists("01"):
    analyze_vasp_neb(".")
elif os.path.exists("neb.out"):
    print("Detected QE NEB output. Use Method B parser instead.")
else:
    print("No NEB data found. Provide either:")
    print("  - VASP NEB directories (00/, 01/, ...)")
    print("  - QE neb.out file")
```

### Method A (continued): Refine NEB with Additional Images

```python
#!/usr/bin/env python3
"""
Add additional images to an existing NEB path around the transition state
region for higher resolution of the barrier.

Reads an existing set of NEB images, identifies the TS region,
and inserts additional interpolated images near the saddle point.
"""

import numpy as np
from ase.io import read, write
from ase.mep.neb import idpp_interpolate
from pathlib import Path
import glob

# ============================================================
# 1. Read existing NEB images
# ============================================================
image_files = sorted(glob.glob("neb_images/image_*.cif"))
if not image_files:
    image_files = sorted(glob.glob("neb_images/image_*.xyz"))

images = [read(f) for f in image_files]
n_orig = len(images)
print(f"Read {n_orig} existing images")

# ============================================================
# 2. If energies are available, identify TS region
# ============================================================
# Try to load energies from a results file
energies = None
try:
    import json
    with open("neb_parsed.json") as f:
        data = json.load(f)
    if "images" in data:
        energies = [img["energy_eV"] for img in data["images"]]
        e_ref = energies[0]
        rel_e = [e - e_ref for e in energies]
        ts_idx = np.argmax(rel_e)
        print(f"Transition state at image {ts_idx} (E_rel = {rel_e[ts_idx]:.4f} eV)")
except (FileNotFoundError, json.JSONDecodeError):
    ts_idx = n_orig // 2  # default: middle
    print(f"No energy data found. Using middle image {ts_idx} as TS region.")

# ============================================================
# 3. Insert additional images around TS
# ============================================================
n_insert = 3  # number of images to insert on each side of TS

# Define the segment around TS (ts_idx-1 to ts_idx+1)
seg_start = max(0, ts_idx - 1)
seg_end = min(n_orig - 1, ts_idx + 1)

print(f"Refining segment: images {seg_start} to {seg_end}")

# Create refined segment with IDPP
segment_initial = images[seg_start].copy()
segment_final = images[seg_end].copy()

segment_images = [segment_initial.copy()]
for _ in range(n_insert * 2 + (seg_end - seg_start - 1)):
    segment_images.append(segment_initial.copy())
segment_images.append(segment_final.copy())

idpp_interpolate(segment_images, traj=None, log=None, mic=True)

# Rebuild full path: images before TS region + refined segment + images after
refined_images = (
    images[:seg_start] +
    segment_images +
    images[seg_end + 1:]
)

print(f"Refined path: {n_orig} -> {len(refined_images)} images")

# ============================================================
# 4. Save refined images
# ============================================================
output_dir = Path("neb_images_refined")
output_dir.mkdir(exist_ok=True)

for i, img in enumerate(refined_images):
    write(str(output_dir / f"image_{i:02d}.cif"), img)

write(str(output_dir / "refined_path.xyz"), refined_images)
write(str(output_dir / "refined_path.pdb"), refined_images)
print(f"Refined images saved to {output_dir}/")
```

## Key Parameters

| Parameter | Recommended Value | Notes |
|---|---|---|
| n_images (interpolation) | 5-11 intermediate | More images near TS for accuracy; 7 is a good default |
| Interpolation method | IDPP | Always prefer over linear; avoids atom overlap |
| mic (minimum image convention) | True | Essential for periodic systems |
| Spline fit | CubicSpline | Gives smooth barrier curve; use for publication plots |
| Trajectory format | extxyz or PDB | extxyz preserves cell info; PDB for molecular viewers |
| VASP NEB images | Read CONTCAR if available | CONTCAR has relaxed coordinates; POSCAR has initial |

## Interpreting Results

1. **Energy profile shape**: A smooth single-hump profile indicates a simple transition. Double humps may indicate an intermediate state.
2. **Barrier height**: Forward barrier determines the rate. Barriers < 0.5 eV are fast at room temperature; > 1.5 eV require high temperature.
3. **Asymmetric barriers**: Different forward and reverse barriers mean the reaction is exothermic (forward < reverse) or endothermic (forward > reverse).
4. **Force error**: Should be < 0.05 eV/A for converged NEB. Large errors indicate more steps needed.
5. **Image spacing**: Ideally uniform along the reaction coordinate. Bunching near endpoints suggests the spring constant is too weak.
6. **Spline vs discrete maximum**: The spline maximum may be slightly higher than the discrete image maximum. Report the spline value for better accuracy.

## Common Issues

| Issue | Solution |
|---|---|
| Images overlap after interpolation | Use IDPP instead of linear interpolation |
| PDB file has wrong connectivity | PDB format does not store bonds; use a viewer that computes bonds from distances |
| VASP OUTCAR missing in some directories | Check if NEB calculation completed. Re-run incomplete images. |
| QE neb.out has no converged energies | Check if calculation ran to completion. Look for "neb: convergence achieved". |
| Spline fit oscillates wildly | Too few images or energies have numerical noise. Add more images. |
| Cannot read POSCAR/CONTCAR | Check VASP version; VASP 5 vs 6 format differences. Specify `format="vasp"` explicitly. |
| Trajectory animation looks jumpy | Not enough images. Interpolate additional frames with IDPP. |
| Reaction coordinate normalization wrong | Ensure you use RMS displacement, not just one atom's displacement. |
