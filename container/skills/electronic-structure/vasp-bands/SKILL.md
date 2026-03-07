# VASP Band Structure Workflow

## When to Use

- You need to compute electronic band structure using VASP.
- You want projected (fat) band plots showing orbital character along the k-path.
- You need element-projected band structure with color-coded atomic contributions.
- You want to compare VASP band structure results with QE calculations.
- You need to identify orbital character of specific bands at high-symmetry points.
- You are analyzing spin-orbit coupling effects on the band structure (VASP LSORBIT).

## Method Selection

| Task | VASP + pymatgen | VASP + VASPKIT (211--216) | QE (see band-structure skill) |
|---|---|---|---|
| Generate KPOINTS (line-mode) | `pymatgen.symmetry.bandstructure.HighSymmKpath` | VASPKIT task 302 | seekpath |
| Plain band structure | Parse EIGENVAL with pymatgen | VASPKIT 211 (REFORMATTED_BAND.dat) | bands.x + plot |
| Projected bands (fat bands) | Parse PROCAR with pymatgen | VASPKIT 213 (projected band) | Not directly supported |
| Element-projected bands | `BandStructure.get_projections_on_elements` | VASPKIT 214 | Not directly supported |
| Orbital-projected bands | Parse PROCAR per orbital | VASPKIT 215 (select orbitals) | Not directly supported |
| Sum of projected bands | Custom script on PROCAR | VASPKIT 216 (sum selected) | Not directly supported |
| Spin-textured bands | SOC-aware PROCAR parsing | VASPKIT 212 (spin-polarized) | QE bands.x with SOC |

## Prerequisites

- A relaxed crystal structure (POSCAR from the `scf-relax` skill).
- **VASP**: Installed and accessible. POTCAR files available.
- **pymatgen**: For generating inputs and parsing outputs. Requires `PMG_VASP_PSP_DIR` environment variable pointing to POTCAR library.
- **VASPKIT**: Installed and in PATH (for tasks 211--216, 302).
- Python packages: `pymatgen`, `numpy`, `matplotlib`, `seekpath`.

---

## Detailed Steps

### Step 1: Generate VASP Inputs for Band Structure

```python
#!/usr/bin/env python3
"""
Generate VASP input files (INCAR, KPOINTS, POSCAR, POTCAR) for
band structure calculation using pymatgen.

Band structure workflow:
  1. SCF on uniform k-grid -> converged CHGCAR
  2. Non-SCF on line-mode k-path (reads CHGCAR) -> EIGENVAL, PROCAR
"""
import os
import numpy as np
from pymatgen.core import Structure
from pymatgen.io.vasp.sets import MPStaticSet, MPNonSCFSet
from pymatgen.symmetry.bandstructure import HighSymmKpath
from pymatgen.io.vasp.inputs import Kpoints, Incar, Poscar

# ── Load structure ─────────────────────────────────────────────────
struct = Structure.from_file("POSCAR")  # or "relaxed.cif"
print(f"Structure: {struct.formula}")
print(f"Space group: {struct.get_space_group_info()}")

# ══════════════════════════════════════════════════════════════════
#  Step 1a: Generate SCF inputs
# ══════════════════════════════════════════════════════════════════
scf_dir = "01_scf"
os.makedirs(scf_dir, exist_ok=True)

scf_set = MPStaticSet(struct, user_incar_settings={
    "ENCUT": 520,
    "EDIFF": 1e-6,
    "ISMEAR": -5,         # Tetrahedron for semiconductors
    # "ISMEAR": 1,         # MP for metals
    # "SIGMA": 0.2,
    "LWAVE": False,
    "LCHARG": True,       # Save CHGCAR for non-SCF step
    "LORBIT": 11,         # For projected bands in the non-SCF step
    "NEDOS": 3001,
    "PREC": "Accurate",
})
scf_set.write_input(scf_dir)
print(f"\nSCF inputs written to {scf_dir}/")

# ══════════════════════════════════════════════════════════════════
#  Step 1b: Generate Non-SCF (bands) inputs
# ══════════════════════════════════════════════════════════════════
bands_dir = "02_bands"
os.makedirs(bands_dir, exist_ok=True)

# Method 1: pymatgen MPNonSCFSet (automatic k-path)
bands_set = MPNonSCFSet.from_prev_calc(
    scf_dir,
    mode="line",
    user_incar_settings={
        "ENCUT": 520,
        "EDIFF": 1e-6,
        "ISMEAR": 0,        # Gaussian smearing for bands
        "SIGMA": 0.05,
        "ICHARG": 11,       # Read CHGCAR (non-SCF)
        "LORBIT": 12,       # l+m decomposed with phase factors (needed for fat bands)
        "LWAVE": False,
        "LCHARG": False,
        "PREC": "Accurate",
        "NBANDS": 40,       # Include empty bands
    },
    kpoints_line_density=40,  # Points per segment
)
bands_set.write_input(bands_dir)
print(f"Bands inputs written to {bands_dir}/")

# ── Print the k-path for reference ────────────────────────────────
kpath = HighSymmKpath(struct)
print("\nHigh-symmetry k-path:")
for segment in kpath.kpath["path"]:
    labels = " -> ".join(segment)
    print(f"  {labels}")
print("\nHigh-symmetry points:")
for label, coords in sorted(kpath.kpath["kpoints"].items()):
    print(f"  {label}: ({coords[0]:.4f}, {coords[1]:.4f}, {coords[2]:.4f})")

print(f"""
Workflow:
  1. cd {scf_dir} && run VASP (mpirun -np N vasp_std)
  2. cp {scf_dir}/CHGCAR {bands_dir}/
  3. cd {bands_dir} && run VASP
  4. Parse EIGENVAL and PROCAR for band structure plots.
""")
```

### Step 2: Generate KPOINTS Line-Mode Manually

```python
#!/usr/bin/env python3
"""
Generate VASP KPOINTS file in line-mode manually using seekpath.
Gives more control over the k-path than pymatgen's automatic generation.
"""
import numpy as np
import seekpath
from pymatgen.core import Structure
from ase.io import read as ase_read

# ── Load structure ─────────────────────────────────────────────────
struct = Structure.from_file("POSCAR")
# Convert to seekpath format
cell = struct.lattice.matrix
positions = struct.frac_coords
numbers = [s.Z for s in struct.species]

path_data = seekpath.get_path((cell, positions, numbers))

# ── Build KPOINTS file ────────────────────────────────────────────
kpoint_coords = path_data["point_coords"]
kpath_segments = path_data["path"]

npoints_per_segment = 40  # Points between each pair of high-symmetry points

kpoints_lines = ["Line-mode KPOINTS generated by seekpath"]
kpoints_lines.append(f"{npoints_per_segment}")
kpoints_lines.append("Reciprocal")

for start_label, end_label in kpath_segments:
    sc = kpoint_coords[start_label]
    ec = kpoint_coords[end_label]
    kpoints_lines.append(
        f"  {sc[0]:.10f}  {sc[1]:.10f}  {sc[2]:.10f}   ! {start_label}"
    )
    kpoints_lines.append(
        f"  {ec[0]:.10f}  {ec[1]:.10f}  {ec[2]:.10f}   ! {end_label}"
    )
    kpoints_lines.append("")  # Blank line separates segments

kpoints_content = "\n".join(kpoints_lines)

with open("KPOINTS_bands", "w") as f:
    f.write(kpoints_content)

print("Written: KPOINTS_bands")
print("\nK-path segments:")
for s, e in kpath_segments:
    sc = kpoint_coords[s]
    ec = kpoint_coords[e]
    print(f"  {s} ({sc[0]:.3f},{sc[1]:.3f},{sc[2]:.3f}) -> "
          f"{e} ({ec[0]:.3f},{ec[1]:.3f},{ec[2]:.3f})")

# ── Also generate KPOINTS via VASPKIT (alternative) ─────────────────
print("""
Alternative: use VASPKIT to generate KPOINTS:
  echo "302" | vaspkit
This interactively generates a line-mode KPOINTS file based on the
crystal symmetry detected from POSCAR.
""")
```

### Step 3: Parse EIGENVAL for Plain Band Structure

```python
#!/usr/bin/env python3
"""
Parse VASP EIGENVAL file and plot band structure.
Works without PROCAR (plain bands, no orbital projection).
"""
import numpy as np
import re
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

def parse_eigenval(filename="EIGENVAL"):
    """
    Parse VASP EIGENVAL file.
    Returns:
        n_kpoints, n_bands, kpoints (Nx3), eigenvalues (N_kpt x N_bands),
        occupations (N_kpt x N_bands)
    """
    with open(filename, "r") as f:
        lines = f.readlines()

    # Line 6: n_electrons, n_kpoints, n_bands
    parts = lines[5].split()
    n_electrons = int(parts[0])
    n_kpoints = int(parts[1])
    n_bands = int(parts[2])

    kpoints = []
    eigenvalues = np.zeros((n_kpoints, n_bands))
    occupations = np.zeros((n_kpoints, n_bands))

    line_idx = 7  # Skip header lines
    for ik in range(n_kpoints):
        # K-point line: kx, ky, kz, weight
        kpt_parts = lines[line_idx].split()
        kpoints.append([float(kpt_parts[i]) for i in range(3)])
        line_idx += 1

        # Band eigenvalues
        for ib in range(n_bands):
            band_parts = lines[line_idx].split()
            eigenvalues[ik, ib] = float(band_parts[1])
            if len(band_parts) > 2:
                occupations[ik, ib] = float(band_parts[2])
            line_idx += 1

        line_idx += 1  # Blank line between k-points

    kpoints = np.array(kpoints)
    return n_kpoints, n_bands, kpoints, eigenvalues, occupations


def compute_k_distances(kpoints, reciprocal_lattice):
    """
    Compute cumulative k-path distances in reciprocal space.
    kpoints: Nx3 array of fractional k-coordinates.
    reciprocal_lattice: 3x3 matrix of reciprocal lattice vectors.
    """
    k_cart = kpoints @ reciprocal_lattice  # Convert to Cartesian
    distances = [0.0]
    for i in range(1, len(k_cart)):
        dk = np.linalg.norm(k_cart[i] - k_cart[i - 1])
        # Detect discontinuity (jump between path segments)
        if dk > 0.5:  # Threshold for discontinuity
            distances.append(distances[-1])
        else:
            distances.append(distances[-1] + dk)
    return np.array(distances)


def find_high_sym_points(k_distances, threshold=1e-6):
    """Find positions of high-symmetry points (where consecutive distances are equal)."""
    positions = [k_distances[0]]
    for i in range(1, len(k_distances)):
        if abs(k_distances[i] - k_distances[i - 1]) < threshold:
            if k_distances[i] not in positions:
                positions.append(k_distances[i])
    positions.append(k_distances[-1])
    return positions


# ── Parse and plot ─────────────────────────────────────────────────
from pymatgen.core import Structure
struct = Structure.from_file("POSCAR")
recip = struct.lattice.reciprocal_lattice.matrix

n_kpts, n_bands, kpoints, eigenvalues, occupations = parse_eigenval("EIGENVAL")
k_dist = compute_k_distances(kpoints, recip)

# Determine Fermi energy from OUTCAR or DOSCAR
e_fermi = 0.0
try:
    with open("OUTCAR", "r") as f:
        for line in f:
            if "E-fermi" in line:
                m = re.search(r"E-fermi\s*:\s*([-\d.]+)", line)
                if m:
                    e_fermi = float(m.group(1))
except FileNotFoundError:
    # Try DOSCAR
    try:
        with open("DOSCAR", "r") as f:
            for _ in range(5):
                f.readline()
            parts = f.readline().split()
            e_fermi = float(parts[3])
    except FileNotFoundError:
        print("WARNING: Could not find OUTCAR or DOSCAR. Using E_F = 0.")

print(f"Fermi energy: {e_fermi:.4f} eV")
print(f"K-points: {n_kpts}, Bands: {n_bands}")

# ── Plot ───────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(8, 6))

for ib in range(n_bands):
    ax.plot(k_dist, eigenvalues[:, ib] - e_fermi,
            color="steelblue", linewidth=1.2)

ax.axhline(0, color="red", linestyle="--", linewidth=0.8, label="$E_F$")

# High-symmetry lines
sym_positions = find_high_sym_points(k_dist)
for xpos in sym_positions:
    ax.axvline(xpos, color="black", linewidth=0.5, alpha=0.5)

ax.set_xlim(k_dist[0], k_dist[-1])
ax.set_ylim(-8, 6)
ax.set_ylabel("$E - E_F$ (eV)", fontsize=14)
ax.set_title("Band Structure (VASP)", fontsize=15)
ax.legend(fontsize=11)
ax.grid(axis="y", alpha=0.3)

plt.tight_layout()
plt.savefig("vasp_bands_plain.png", dpi=200, bbox_inches="tight")
print("Saved: vasp_bands_plain.png")

# ── Band gap analysis ─────────────────────────────────────────────
vbm = np.max(eigenvalues[eigenvalues <= e_fermi])
cbm = np.min(eigenvalues[eigenvalues > e_fermi])
gap = cbm - vbm

vbm_idx = np.unravel_index(np.argmax(np.where(eigenvalues <= e_fermi, eigenvalues, -np.inf)),
                           eigenvalues.shape)
cbm_idx = np.unravel_index(np.argmin(np.where(eigenvalues > e_fermi, eigenvalues, np.inf)),
                           eigenvalues.shape)

print(f"\nBand gap: {gap:.4f} eV")
print(f"VBM at k-index {vbm_idx[0]}, band {vbm_idx[1]}: {vbm:.4f} eV")
print(f"CBM at k-index {cbm_idx[0]}, band {cbm_idx[1]}: {cbm:.4f} eV")
print(f"Type: {'Direct' if vbm_idx[0] == cbm_idx[0] else 'Indirect'}")
```

### Step 4: Parse PROCAR for Fat Band (Orbital-Projected) Plot

```python
#!/usr/bin/env python3
"""
Parse VASP PROCAR file for orbital-projected (fat) band structure.
PROCAR contains the projection of each Kohn-Sham state onto local orbitals.
Requires LORBIT=11 or 12 in INCAR.
"""
import numpy as np
import re
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

def parse_procar(filename="PROCAR"):
    """
    Parse VASP PROCAR file.
    Returns:
        n_kpts, n_bands, n_ions, n_orbitals,
        kpoints (n_kpts x 3),
        eigenvalues (n_kpts x n_bands),
        projections (n_kpts x n_bands x n_ions x n_orbitals),
        orbital_names (list of str)
    """
    with open(filename, "r") as f:
        text = f.read()

    # Parse header
    header = re.search(r"# of k-points:\s+(\d+)\s+# of bands:\s+(\d+)\s+# of ions:\s+(\d+)",
                       text)
    if not header:
        raise ValueError("Could not parse PROCAR header")

    n_kpts = int(header.group(1))
    n_bands = int(header.group(2))
    n_ions = int(header.group(3))

    # Find orbital names from first projection block
    orb_match = re.search(r"ion\s+((?:s|p[xyz]?|d[a-z0-9\-]*|f[a-z0-9\-]*|\s+tot)+)", text)
    if orb_match:
        orbital_names = orb_match.group(1).split()
        # Remove 'tot' if present
        if orbital_names[-1] == "tot":
            orbital_names = orbital_names[:-1]
    else:
        # Default for LORBIT=11
        orbital_names = ["s", "py", "pz", "px", "dxy", "dyz", "dz2", "dxz", "dx2-y2"]

    n_orbitals = len(orbital_names)

    kpoints = np.zeros((n_kpts, 3))
    eigenvalues = np.zeros((n_kpts, n_bands))
    projections = np.zeros((n_kpts, n_bands, n_ions, n_orbitals))

    # Parse k-point by k-point
    kpt_blocks = re.split(r"k-point\s+(\d+)\s*:", text)[1:]
    # kpt_blocks alternates: kpt_index, block_content

    for ik in range(n_kpts):
        kpt_idx_str = kpt_blocks[2 * ik]
        block = kpt_blocks[2 * ik + 1]

        # Parse k-point coordinates
        kpt_match = re.search(r"([-\d.]+)\s+([-\d.]+)\s+([-\d.]+)", block[:200])
        if kpt_match:
            kpoints[ik] = [float(kpt_match.group(j)) for j in range(1, 4)]

        # Parse bands within this k-point
        band_blocks = re.split(r"band\s+(\d+)\s*#", block)[1:]

        for ib in range(min(n_bands, len(band_blocks) // 2)):
            band_content = band_blocks[2 * ib + 1]

            # Parse eigenvalue
            e_match = re.search(r"energy\s+([-\d.]+)", band_content)
            if e_match:
                eigenvalues[ik, ib] = float(e_match.group(1))

            # Parse projections for each ion
            # Find lines with ion index and orbital values
            proj_lines = re.findall(
                r"^\s*(\d+)\s+([\d.]+(?:\s+[\d.]+)*)",
                band_content, re.MULTILINE
            )

            for ion_str, values_str in proj_lines:
                ion_idx = int(ion_str) - 1  # 0-based
                if ion_idx < n_ions:
                    values = [float(x) for x in values_str.split()]
                    for io in range(min(n_orbitals, len(values))):
                        projections[ik, ib, ion_idx, io] = values[io]

    return n_kpts, n_bands, n_ions, n_orbitals, kpoints, eigenvalues, projections, orbital_names


# ── Parse PROCAR ───────────────────────────────────────────────────
print("Parsing PROCAR ...")
n_kpts, n_bands, n_ions, n_orbs, kpoints, eigenvalues, projections, orb_names = \
    parse_procar("PROCAR")
print(f"K-points: {n_kpts}, Bands: {n_bands}, Ions: {n_ions}, Orbitals: {n_orbs}")
print(f"Orbital names: {orb_names}")

# ── Load structure for element info and reciprocal lattice ─────────
from pymatgen.core import Structure
struct = Structure.from_file("POSCAR")
recip = struct.lattice.reciprocal_lattice.matrix
elements = [str(s.specie) for s in struct]

# Compute k-distances
k_cart = kpoints @ recip
k_dist = [0.0]
for i in range(1, len(k_cart)):
    dk = np.linalg.norm(k_cart[i] - k_cart[i - 1])
    if dk > 0.5:
        k_dist.append(k_dist[-1])
    else:
        k_dist.append(k_dist[-1] + dk)
k_dist = np.array(k_dist)

# Fermi energy
e_fermi = 0.0
try:
    with open("OUTCAR", "r") as f:
        for line in f:
            if "E-fermi" in line:
                m = re.search(r"E-fermi\s*:\s*([-\d.]+)", line)
                if m: e_fermi = float(m.group(1))
except FileNotFoundError:
    pass
print(f"Fermi energy: {e_fermi:.4f} eV")

# ══════════════════════════════════════════════════════════════════
#  FAT BAND PLOT 1: Orbital-Projected (s, p, d character)
# ══════════════════════════════════════════════════════════════════
print("\nPlotting orbital-projected fat bands ...")

# Group orbitals: s, p (sum of px+py+pz), d (sum of dxy+dyz+dz2+dxz+dx2-y2)
orb_groups = {}
for io, name in enumerate(orb_names):
    if name == "s":
        orb_groups.setdefault("s", []).append(io)
    elif name.startswith("p"):
        orb_groups.setdefault("p", []).append(io)
    elif name.startswith("d"):
        orb_groups.setdefault("d", []).append(io)
    elif name.startswith("f"):
        orb_groups.setdefault("f", []).append(io)

# Sum projections over all ions for each orbital group
orbital_weights = {}
for group_name, indices in orb_groups.items():
    # Sum over all ions and all orbitals in the group
    w = np.zeros((n_kpts, n_bands))
    for io in indices:
        w += np.sum(projections[:, :, :, io], axis=2)  # sum over ions
    orbital_weights[group_name] = w

orbital_colors = {"s": "#e74c3c", "p": "#3498db", "d": "#2ecc71", "f": "#9b59b6"}

fig, ax = plt.subplots(figsize=(8, 6))

# Plot plain bands as thin lines
for ib in range(n_bands):
    ax.plot(k_dist, eigenvalues[:, ib] - e_fermi,
            color="gray", linewidth=0.5, alpha=0.3)

# Overlay fat bands as colored scatter points
for group_name in ["s", "p", "d", "f"]:
    if group_name not in orbital_weights:
        continue
    w = orbital_weights[group_name]
    color = orbital_colors[group_name]
    for ib in range(n_bands):
        weights = w[:, ib]
        # Scale marker size by projection weight
        sizes = weights * 50  # Adjust multiplier for visibility
        mask = sizes > 0.5  # Only plot non-negligible weights
        if np.any(mask):
            ax.scatter(k_dist[mask], eigenvalues[mask, ib] - e_fermi,
                       s=sizes[mask], c=color, alpha=0.6,
                       label=group_name if ib == 0 else None,
                       edgecolors="none")

ax.axhline(0, color="red", linestyle="--", linewidth=0.8, label="$E_F$")
ax.set_xlim(k_dist[0], k_dist[-1])
ax.set_ylim(-8, 6)
ax.set_ylabel("$E - E_F$ (eV)", fontsize=14)
ax.set_title("Orbital-Projected Band Structure (Fat Bands)", fontsize=14)

# Remove duplicate labels
handles, labels = ax.get_legend_handles_labels()
by_label = dict(zip(labels, handles))
ax.legend(by_label.values(), by_label.keys(), fontsize=11, loc="upper right")
ax.grid(axis="y", alpha=0.3)

plt.tight_layout()
plt.savefig("vasp_bands_fat_orbital.png", dpi=200, bbox_inches="tight")
print("Saved: vasp_bands_fat_orbital.png")

# ══════════════════════════════════════════════════════════════════
#  FAT BAND PLOT 2: Element-Projected
# ══════════════════════════════════════════════════════════════════
print("\nPlotting element-projected fat bands ...")

# Group atoms by element
unique_elements = list(dict.fromkeys(elements))
element_indices = {}
for i, el in enumerate(elements):
    element_indices.setdefault(el, []).append(i)

# Compute element weights (sum all orbitals for atoms of each element)
element_weights = {}
for el, atom_indices in element_indices.items():
    w = np.zeros((n_kpts, n_bands))
    for ia in atom_indices:
        w += np.sum(projections[:, :, ia, :], axis=2)  # sum over orbitals
    element_weights[el] = w

fig, ax = plt.subplots(figsize=(8, 6))
elem_colors = plt.cm.Set1(np.linspace(0, 0.8, len(unique_elements)))

for ib in range(n_bands):
    ax.plot(k_dist, eigenvalues[:, ib] - e_fermi,
            color="gray", linewidth=0.5, alpha=0.3)

for idx, el in enumerate(unique_elements):
    w = element_weights[el]
    color = elem_colors[idx]
    for ib in range(n_bands):
        sizes = w[:, ib] * 50
        mask = sizes > 0.5
        if np.any(mask):
            ax.scatter(k_dist[mask], eigenvalues[mask, ib] - e_fermi,
                       s=sizes[mask], c=[color], alpha=0.6,
                       label=el if ib == 0 else None,
                       edgecolors="none")

ax.axhline(0, color="red", linestyle="--", linewidth=0.8, label="$E_F$")
ax.set_xlim(k_dist[0], k_dist[-1])
ax.set_ylim(-8, 6)
ax.set_ylabel("$E - E_F$ (eV)", fontsize=14)
ax.set_title("Element-Projected Band Structure", fontsize=14)

handles, labels = ax.get_legend_handles_labels()
by_label = dict(zip(labels, handles))
ax.legend(by_label.values(), by_label.keys(), fontsize=11, loc="upper right")
ax.grid(axis="y", alpha=0.3)

plt.tight_layout()
plt.savefig("vasp_bands_fat_element.png", dpi=200, bbox_inches="tight")
print("Saved: vasp_bands_fat_element.png")

# ══════════════════════════════════════════════════════════════════
#  FAT BAND PLOT 3: RGB Color-Coded Bands (3-orbital mixing)
# ══════════════════════════════════════════════════════════════════
print("\nPlotting RGB color-coded fat bands (s=R, p=G, d=B) ...")

fig, ax = plt.subplots(figsize=(8, 6))

# Normalize weights so they sum to 1 at each (k, band) point
for ib in range(n_bands):
    for ik in range(n_kpts):
        ws = orbital_weights.get("s", np.zeros((n_kpts, n_bands)))[ik, ib]
        wp = orbital_weights.get("p", np.zeros((n_kpts, n_bands)))[ik, ib]
        wd = orbital_weights.get("d", np.zeros((n_kpts, n_bands)))[ik, ib]
        total = ws + wp + wd
        if total > 1e-6:
            r = ws / total
            g = wp / total
            b = wd / total
        else:
            r, g, b = 0.5, 0.5, 0.5

        ax.plot(k_dist[ik:ik+2] if ik < n_kpts - 1 else [k_dist[ik]],
                [eigenvalues[ik, ib] - e_fermi],
                "o", markersize=1.5, color=(r, g, b), alpha=0.8)

ax.axhline(0, color="black", linestyle="--", linewidth=0.8)
ax.set_xlim(k_dist[0], k_dist[-1])
ax.set_ylim(-8, 6)
ax.set_ylabel("$E - E_F$ (eV)", fontsize=14)
ax.set_title("Band Structure (R=s, G=p, B=d)", fontsize=14)
ax.grid(axis="y", alpha=0.3)

# Custom legend
from matplotlib.patches import Patch
legend_elements = [
    Patch(facecolor="red", label="s"),
    Patch(facecolor="green", label="p"),
    Patch(facecolor="blue", label="d"),
]
ax.legend(handles=legend_elements, fontsize=11, loc="upper right")

plt.tight_layout()
plt.savefig("vasp_bands_rgb.png", dpi=200, bbox_inches="tight")
print("Saved: vasp_bands_rgb.png")
```

### Step 5: Parse VASP Bands with pymatgen (Recommended)

```python
#!/usr/bin/env python3
"""
Parse VASP band structure using pymatgen's Vasprun parser.
This is the recommended approach for most use cases -- it handles
all the k-path reconstruction and symmetry labels automatically.
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pymatgen.io.vasp import Vasprun, BSVasprun
from pymatgen.electronic_structure.plotter import BSPlotter, BSPlotterProjected
from pymatgen.electronic_structure.core import Spin, OrbitalType

# ── Method 1: Quick plot with BSPlotter ────────────────────────────
print("Parsing vasprun.xml for band structure ...")
vrun = BSVasprun("vasprun.xml", parse_projected_eigen=True)
bs = vrun.get_band_structure(line_mode=True)

print(f"Band gap: {bs.get_band_gap()['energy']:.4f} eV")
print(f"Is metal: {bs.is_metal()}")
if not bs.is_metal():
    direct = bs.get_band_gap()["direct"]
    print(f"Direct gap: {direct}")

# Plain band structure plot
plotter = BSPlotter(bs)
fig = plotter.get_plot(ylim=(-8, 6), zero_to_efermi=True)
plt.savefig("vasp_bands_pymatgen.png", dpi=200, bbox_inches="tight")
print("Saved: vasp_bands_pymatgen.png")
plt.close()

# ── Method 2: Projected band structure plot ────────────────────────
print("\nPlotting projected band structure ...")
proj_plotter = BSPlotterProjected(bs)

# Element-projected (one color per element)
fig = proj_plotter.get_elt_projected_plots(zero_to_efermi=True, ylim=(-8, 6))
plt.savefig("vasp_bands_elt_projected.png", dpi=200, bbox_inches="tight")
print("Saved: vasp_bands_elt_projected.png")
plt.close()

# Element + orbital projected (color shows element, size shows orbital)
fig = proj_plotter.get_elt_projected_plots_color(zero_to_efermi=True, ylim=(-8, 6))
plt.savefig("vasp_bands_elt_color.png", dpi=200, bbox_inches="tight")
print("Saved: vasp_bands_elt_color.png")
plt.close()

print("\nDone. All band structure plots generated.")
```

### Step 6: VASPKIT Band Structure Workflows (211--216)

```bash
#!/bin/bash
# VASPKIT band structure post-processing.
# Run these commands in the directory containing VASP output files
# (EIGENVAL, PROCAR, OUTCAR, POSCAR, KPOINTS).

# ── Task 211: Standard band structure ──────────────────────────────
echo "211" | vaspkit
# Output:
#   REFORMATTED_BAND.dat        -- band data for plotting
#   BAND_GAP                    -- band gap info (if semiconductor)
#   KPATH.in                    -- k-path used
# The .dat file has columns: k-distance, E1, E2, ..., EN (already shifted by E_F)

# ── Task 212: Spin-polarized band structure ────────────────────────
# For ISPIN=2 calculations
echo "212" | vaspkit
# Output: REFORMATTED_BAND_UP.dat, REFORMATTED_BAND_DW.dat

# ── Task 213: Projected band structure ─────────────────────────────
echo "213" | vaspkit
# Interactive: select atoms and orbitals for projection.
# Output: PBAND_*.dat files with projection weights

# ── Task 214: Element-projected band structure ─────────────────────
echo "214" | vaspkit
# Output: PBAND_{element}.dat for each element
# Columns: k-dist, energy, weight_s, weight_p, weight_d, weight_f, weight_total

# ── Task 215: Orbital-projected band structure ─────────────────────
echo "215" | vaspkit
# Interactive: select specific orbitals (e.g., dxy, dxz)
# Output: PBAND with selected orbital weights

# ── Task 216: Sum of projected bands for selected atoms ────────────
echo "216" | vaspkit
# Interactive: select atom indices and orbitals to sum
# Useful for projecting onto a subset of atoms (e.g., surface layer)
```

#### Plot VASPKIT Output Files

```python
#!/usr/bin/env python3
"""
Plot VASPKIT band structure output files.
Works with REFORMATTED_BAND.dat and PBAND_*.dat files.
"""
import os
import re
import glob
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ── Parse VASPKIT REFORMATTED_BAND.dat ─────────────────────────────
def parse_reformatted_band(filename="REFORMATTED_BAND.dat"):
    """
    Parse VASPKIT REFORMATTED_BAND.dat.
    Returns: k_distances (1D), eigenvalues (n_kpts x n_bands)
    High-symmetry points are marked by comment lines.
    """
    k_dist = []
    bands = []
    sym_points = []

    with open(filename, "r") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                # Check for high-symmetry point label
                if line.startswith("#"):
                    m = re.search(r"([\d.]+)\s+!\s+(\S+)", line)
                    if m:
                        sym_points.append((float(m.group(1)), m.group(2)))
                continue
            parts = line.split()
            k_dist.append(float(parts[0]))
            bands.append([float(x) for x in parts[1:]])

    k_dist = np.array(k_dist)
    eigenvalues = np.array(bands)
    return k_dist, eigenvalues, sym_points


# ── Parse VASPKIT PBAND files (element-projected) ──────────────────
def parse_pband(filename):
    """
    Parse VASPKIT PBAND_Element.dat.
    Returns: k_dist, energy, weights dict (s, p, d, f, total)
    """
    data = np.loadtxt(filename, comments="#")
    result = {
        "k_dist": data[:, 0],
        "energy": data[:, 1],
    }
    # Columns vary: typically k_dist, energy, s, p, d, [f], total
    if data.shape[1] == 6:
        result["s"] = data[:, 2]
        result["p"] = data[:, 3]
        result["d"] = data[:, 4]
        result["total"] = data[:, 5]
    elif data.shape[1] == 7:
        result["s"] = data[:, 2]
        result["p"] = data[:, 3]
        result["d"] = data[:, 4]
        result["f"] = data[:, 5]
        result["total"] = data[:, 6]
    return result


# ── Plot 1: Plain band structure from REFORMATTED_BAND.dat ─────────
if os.path.exists("REFORMATTED_BAND.dat"):
    k_dist, eigenvalues, sym_points = parse_reformatted_band()
    print(f"Bands: {eigenvalues.shape[1]}, K-points: {len(k_dist)}")

    fig, ax = plt.subplots(figsize=(8, 6))
    for ib in range(eigenvalues.shape[1]):
        ax.plot(k_dist, eigenvalues[:, ib], color="steelblue", linewidth=1.2)

    ax.axhline(0, color="red", linestyle="--", linewidth=0.8, label="$E_F$")

    # High-symmetry labels
    if sym_points:
        positions = [sp[0] for sp in sym_points]
        labels = [sp[1].replace("GAMMA", r"$\Gamma$").replace("G", r"$\Gamma$")
                  for sp in sym_points]
        for xpos in positions:
            ax.axvline(xpos, color="black", linewidth=0.5, alpha=0.5)
        ax.set_xticks(positions)
        ax.set_xticklabels(labels, fontsize=13)

    ax.set_xlim(k_dist[0], k_dist[-1])
    ax.set_ylim(-8, 6)
    ax.set_ylabel("$E - E_F$ (eV)", fontsize=14)
    ax.set_title("Band Structure (VASPKIT)", fontsize=15)
    ax.legend(fontsize=11)
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    plt.savefig("vaspkit_bands.png", dpi=200, bbox_inches="tight")
    print("Saved: vaspkit_bands.png")

# ── Plot 2: Element-projected bands from PBAND files ───────────────
pband_files = sorted(glob.glob("PBAND_*.dat"))
if pband_files:
    print(f"\nFound {len(pband_files)} PBAND files:")
    for f in pband_files:
        print(f"  {f}")

    orbital_colors = {"s": "#e74c3c", "p": "#3498db", "d": "#2ecc71", "f": "#9b59b6"}

    for pband_file in pband_files:
        element = os.path.basename(pband_file).replace("PBAND_", "").replace(".dat", "")
        data = parse_pband(pband_file)

        fig, ax = plt.subplots(figsize=(8, 6))

        # Get unique k-distances to determine band structure
        # PBAND files list all bands concatenated
        n_kpts_unique = len(set(data["k_dist"]))

        for orb in ["s", "p", "d", "f"]:
            if orb not in data:
                continue
            color = orbital_colors[orb]
            sizes = data[orb] * 30
            mask = sizes > 0.3
            if np.any(mask):
                ax.scatter(data["k_dist"][mask], data["energy"][mask],
                           s=sizes[mask], c=color, alpha=0.5,
                           label=f"{element} ({orb})", edgecolors="none")

        ax.axhline(0, color="red", linestyle="--", linewidth=0.8, label="$E_F$")
        ax.set_xlim(min(data["k_dist"]), max(data["k_dist"]))
        ax.set_ylim(-8, 6)
        ax.set_ylabel("$E - E_F$ (eV)", fontsize=14)
        ax.set_title(f"{element}-Projected Band Structure (VASPKIT)", fontsize=14)
        ax.legend(fontsize=10)
        ax.grid(axis="y", alpha=0.3)

        plt.tight_layout()
        outname = f"vaspkit_pband_{element}.png"
        plt.savefig(outname, dpi=200, bbox_inches="tight")
        print(f"Saved: {outname}")
        plt.close()

print("\nDone.")
```

### Step 7: Compare VASP and QE Band Structures

```python
#!/usr/bin/env python3
"""
Compare band structures from VASP and QE on the same plot.
Requires: VASP vasprun.xml (or EIGENVAL) and QE bands.dat.gnu.
Both calculations must use the same structure and k-path.
"""
import numpy as np
import re
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ══════════════════════════════════════════════════════════════════
#  Parse QE bands
# ══════════════════════════════════════════════════════════════════
def parse_qe_bands_gnu(gnu_file):
    """Parse QE bands.dat.gnu file."""
    with open(gnu_file, "r") as f:
        text = f.read()
    blocks = text.strip().split("\n\n")
    bands = []
    k_dist = None
    for block in blocks:
        lines = block.strip().split("\n")
        kvs, evs = [], []
        for line in lines:
            parts = line.split()
            if len(parts) >= 2:
                kvs.append(float(parts[0]))
                evs.append(float(parts[1]))
        if k_dist is None:
            k_dist = np.array(kvs)
        bands.append(np.array(evs))
    return k_dist, np.column_stack(bands)

def get_qe_fermi(scf_output):
    """Extract Fermi energy from QE SCF output."""
    e_fermi = 0.0
    with open(scf_output, "r") as f:
        for line in f:
            if "the Fermi energy is" in line:
                m = re.search(r"is\s+([-\d.]+)", line)
                if m: e_fermi = float(m.group(1))
            if "highest occupied, lowest unoccupied" in line:
                m = re.search(r":\s+([-\d.]+)\s+([-\d.]+)", line)
                if m: e_fermi = (float(m.group(1)) + float(m.group(2))) / 2
            if "highest occupied" in line and "lowest" not in line:
                m = re.search(r":\s+([-\d.]+)", line)
                if m: e_fermi = float(m.group(1))
    return e_fermi

# ══════════════════════════════════════════════════════════════════
#  Parse VASP bands (using pymatgen)
# ══════════════════════════════════════════════════════════════════
def get_vasp_bands(vasprun_file="vasprun.xml"):
    """Parse VASP band structure from vasprun.xml."""
    from pymatgen.io.vasp import BSVasprun
    from pymatgen.electronic_structure.core import Spin
    vrun = BSVasprun(vasprun_file, parse_projected_eigen=False)
    bs = vrun.get_band_structure(line_mode=True)

    # Extract k-distances and eigenvalues
    branches = bs.branches
    k_dist = []
    eigenvalues = []

    for branch in branches:
        start_idx = branch["start_index"]
        end_idx = branch["end_index"]
        for ik in range(start_idx, end_idx + 1):
            k_dist.append(bs.distance[ik])
            eigs = []
            for ib in range(bs.nb_bands):
                eigs.append(bs.bands[Spin.up][ib][ik])
            eigenvalues.append(eigs)

    return np.array(k_dist), np.array(eigenvalues), bs.efermi

# ══════════════════════════════════════════════════════════════════
#  Plot comparison
# ══════════════════════════════════════════════════════════════════
import os

fig, ax = plt.subplots(figsize=(8, 6))

# QE bands
qe_gnu = "qe_bands/si_bands.dat.gnu"       # Adjust path
qe_scf = "qe_bands/si_scf.out"             # Adjust path
if os.path.exists(qe_gnu):
    k_qe, eig_qe = parse_qe_bands_gnu(qe_gnu)
    e_fermi_qe = get_qe_fermi(qe_scf) if os.path.exists(qe_scf) else 0.0

    # Normalize k-distance to [0, 1] for comparison
    k_qe_norm = (k_qe - k_qe[0]) / (k_qe[-1] - k_qe[0])

    for ib in range(eig_qe.shape[1]):
        label = "QE (PBE)" if ib == 0 else None
        ax.plot(k_qe_norm, eig_qe[:, ib] - e_fermi_qe,
                color="steelblue", linewidth=1.5, label=label)

# VASP bands
vasp_xml = "vasp_bands/vasprun.xml"         # Adjust path
if os.path.exists(vasp_xml):
    k_vasp, eig_vasp, e_fermi_vasp = get_vasp_bands(vasp_xml)

    k_vasp_norm = (k_vasp - k_vasp[0]) / (k_vasp[-1] - k_vasp[0])

    for ib in range(eig_vasp.shape[1]):
        label = "VASP (PBE)" if ib == 0 else None
        ax.plot(k_vasp_norm, eig_vasp[:, ib] - e_fermi_vasp,
                color="darkorange", linewidth=1.5, linestyle="--",
                alpha=0.8, label=label)

ax.axhline(0, color="red", linestyle=":", linewidth=0.8, label="$E_F$")
ax.set_xlim(0, 1)
ax.set_ylim(-8, 6)
ax.set_xlabel("Normalized k-distance", fontsize=13)
ax.set_ylabel("$E - E_F$ (eV)", fontsize=14)
ax.set_title("Band Structure Comparison: QE vs VASP", fontsize=14)
ax.legend(fontsize=11, loc="upper right")
ax.grid(axis="y", alpha=0.3)

plt.tight_layout()
plt.savefig("bands_comparison_qe_vasp.png", dpi=200, bbox_inches="tight")
print("Saved: bands_comparison_qe_vasp.png")
```

---

## Key Parameters

| Parameter | VASP INCAR | Notes |
|---|---|---|
| `ICHARG` | 11 | Read CHGCAR for non-SCF band calculation (essential) |
| `LORBIT` | 11 or 12 | 11: lm-decomposed PDOS/PROCAR. 12: with phase factors (for fat bands). |
| `NBANDS` | 1.5--2x occupied | Must include empty bands for conduction structure |
| `ISMEAR` | 0 (Gaussian) | Use Gaussian smearing for bands (not tetrahedron) |
| `SIGMA` | 0.05 | Small smearing for accurate eigenvalues |
| `ENCUT` | Same as SCF | Must match the SCF calculation |
| `ISYM` | -1 (recommended) | Turn off symmetry for proper projection weights |
| `LSORBIT` | .TRUE. (if needed) | Enable spin-orbit coupling |
| KPOINTS density | 30--50 per segment | More points = smoother bands |

## Interpreting Results

- **Plain bands**: Identify metal (bands cross E_F), semiconductor (gap at E_F), or insulator (large gap).
- **Fat bands**: The marker size at each (k, E) point represents the orbital/element weight. Large markers mean that orbital/element dominates that state.
- **Element-projected bands**: In multi-component systems, identifies which element contributes to which bands. Useful for understanding bonding and charge transfer.
- **RGB bands**: Red=s, Green=p, Blue=d gives an intuitive visual of orbital character evolution along the k-path.
- **Band inversion**: In topological materials, watch for bands with swapped orbital character at certain k-points (e.g., s-p inversion at Gamma).
- **PBE limitations**: PBE underestimates band gaps by 30--50%. For accurate gaps, use HSE06 (`LHFCALC=.TRUE.`) or GW.
- **QE vs VASP comparison**: Bands should agree within ~0.05 eV if same structure, k-path, and functional are used. Differences arise from pseudopotential choice (PAW vs US vs NC) and implementation details.

## Common Issues

| Problem | Solution |
|---|---|
| VASP bands calculation fails to start | Ensure CHGCAR from SCF is in the same directory. Set `ICHARG=11`. |
| PROCAR is empty or missing | Set `LORBIT=11` or `12` in INCAR. Rerun the bands calculation. |
| K-path labels wrong | Use seekpath or `pymatgen.symmetry.bandstructure.HighSymmKpath` for correct path. Verify crystal system. |
| Fat band markers too small/large | Adjust the scaling factor (e.g., `sizes = weights * 100`). Normalize weights if needed. |
| Band structure has gaps/jumps | The line-mode KPOINTS has discontinuities between path segments. This is normal -- blank lines in KPOINTS separate segments. |
| `BSVasprun` parsing error | Ensure vasprun.xml is complete (VASP finished normally). Try `BSVasprun(file, parse_projected_eigen=False)` if projection parsing fails. |
| VASP and QE bands don't align | Normalize k-distances to [0,1]. Ensure both use the same structure (lattice parameters, atomic positions). Check that both use the same k-path convention (seekpath standardizes the cell). |
| Spin-orbit bands have wrong degeneracies | Ensure `LSORBIT=.TRUE.` is set. SOC doubles the number of bands. Use `NBANDS` = 2x non-SOC value. |
| VASPKIT task 214 produces empty files | Ensure PROCAR file exists and is non-empty. Check that LORBIT was set correctly in the VASP calculation. |
