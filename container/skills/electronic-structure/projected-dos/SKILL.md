# Projected Density of States (PDOS)

## When to Use

- You need atom-resolved, element-resolved, or orbital-resolved density of states.
- You want to understand which orbitals (s, p, d, f) contribute to specific energy regions.
- You are analyzing crystal field splitting, hybridization, or bonding character.
- You need local DOS (LDOS) for a specific atom or group of atoms in a heterostructure or defect system.
- You want k-resolved PDOS to connect band character with density of states.
- You want band-resolved PDOS to identify orbital contributions to individual bands.
- You need the inverse participation ratio (IPR) to quantify state localization.

## Method Selection

| Task | QE (projwfc.x) | VASP + VASPKIT | VASP + pymatgen |
|---|---|---|---|
| PDOS by atom/orbital | projwfc.x parses projections from NSCF | VASPKIT 113--115 from PROCAR/vasprun.xml | `pymatgen.electronic_structure.dos.CompleteDos` |
| Element-projected DOS | Sum projwfc.x outputs by element | VASPKIT 112 | `complete_dos.get_element_dos()` |
| LDOS for selected atoms | Sum projwfc.x outputs for chosen atom indices | VASPKIT 116 (select atoms interactively) | `complete_dos.get_site_dos()` |
| k-resolved PDOS | Not directly available; use fat bands instead | VASPKIT 119 | Parse PROCAR manually |
| Band-resolved PDOS | Not directly available | VASPKIT 120 | Parse PROCAR per band |
| Inverse participation ratio | Custom post-processing of projwfc.x output | VASPKIT 110 | Custom script from PROCAR |
| Spin-polarized PDOS | projwfc.x handles collinear spin automatically | VASPKIT 113 (ISPIN=2) | `complete_dos.get_element_spd_dos()` |

## Prerequisites

- A converged SCF calculation (use the `scf-relax` skill).
- A completed NSCF calculation on a dense k-grid (see `density-of-states` skill for the NSCF step).
- **QE**: `pw.x`, `projwfc.x` executables. Pseudopotentials with projection information.
- **VASP**: Converged calculation with `LORBIT=11` (or `LORBIT=12` for m-resolved). `ISYM=-1` recommended for PDOS. Output files: `vasprun.xml`, `DOSCAR`, `PROCAR`.
- **VASPKIT**: Installed and in PATH (for VASP post-processing tasks 110--120).
- Python packages: `numpy`, `matplotlib`, `pymatgen`.

---

## Detailed Steps

### Method A: QE PDOS with projwfc.x

This extends the basic DOS workflow. Assumes SCF and NSCF are already done (see `density-of-states` skill). The key addition is detailed parsing and plotting of atom-by-atom, orbital-by-orbital PDOS.

#### Step A1: Run projwfc.x for Full Projections

```python
#!/usr/bin/env python3
"""
Run projwfc.x to compute atom- and orbital-projected DOS.
Prerequisite: SCF + NSCF already completed with matching prefix/outdir.
"""
import os
import subprocess

OUTDIR = os.path.abspath("./tmp_dos")
PREFIX = "srtio3"  # Example: SrTiO3

# projwfc.x input
# kresolveddos=.true. writes k-resolved PDOS (large files)
projwfc_input = f"""&PROJWFC
    prefix   = '{PREFIX}'
    outdir   = '{OUTDIR}'
    filpdos  = '{PREFIX}_pdos'
    Emin     = -15.0
    Emax     = 10.0
    DeltaE   = 0.01
    ngauss   = 0
    degauss  = 0.005
/
"""

with open(f"{PREFIX}_projwfc.in", "w") as f:
    f.write(projwfc_input)

print("Running projwfc.x ...")
result = subprocess.run(
    ["mpirun", "-np", "4", "projwfc.x", "-in", f"{PREFIX}_projwfc.in"],
    capture_output=True, text=True, timeout=600
)
with open(f"{PREFIX}_projwfc.out", "w") as f:
    f.write(result.stdout)

if result.returncode == 0:
    print("projwfc.x completed successfully.")
    pdos_files = sorted([f for f in os.listdir(".") if f.startswith(f"{PREFIX}_pdos")])
    print(f"Generated {len(pdos_files)} PDOS files:")
    for pf in pdos_files[:20]:
        print(f"  {pf}")
    if len(pdos_files) > 20:
        print(f"  ... and {len(pdos_files) - 20} more")
else:
    print("ERROR in projwfc.x!")
    print(result.stderr[-500:] if result.stderr else "")
```

#### Step A2: Parse and Plot Stacked Orbital PDOS

```python
#!/usr/bin/env python3
"""
Parse projwfc.x output files and create stacked orbital PDOS plots.
Handles multi-element systems (e.g., SrTiO3, Fe2O3).
"""
import os
import re
import glob
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

PREFIX = "srtio3"

# ── Parse Fermi energy from NSCF output ────────────────────────────
def get_fermi_energy(nscf_output):
    e_fermi = 0.0
    with open(nscf_output, "r") as f:
        for line in f:
            if "the Fermi energy is" in line:
                m = re.search(r"is\s+([-\d.]+)", line)
                if m:
                    e_fermi = float(m.group(1))
            if "highest occupied, lowest unoccupied" in line:
                m = re.search(r":\s+([-\d.]+)\s+([-\d.]+)", line)
                if m:
                    e_fermi = (float(m.group(1)) + float(m.group(2))) / 2
            if "highest occupied" in line and "lowest" not in line:
                m = re.search(r":\s+([-\d.]+)", line)
                if m:
                    e_fermi = float(m.group(1))
    return e_fermi

e_fermi = get_fermi_energy(f"{PREFIX}_nscf.out")
print(f"Fermi energy: {e_fermi:.4f} eV")

# ── Parse all PDOS files ───────────────────────────────────────────
def parse_all_pdos(prefix):
    """
    Parse all projwfc.x PDOS output files.
    Returns dict: {(atom_idx, element, wfc_idx, orbital): (energy, pdos)}
    Also returns total DOS from pdos_tot file.
    """
    pdos_dict = {}
    pattern = f"{prefix}_pdos.pdos_atm#*"
    files = glob.glob(pattern)

    for fpath in sorted(files):
        fname = os.path.basename(fpath)
        # Filename format: prefix_pdos.pdos_atm#1(Sr)_wfc#1(s)
        m = re.search(r"atm#(\d+)\((\w+)\)_wfc#(\d+)\((\w+)\)", fname)
        if m:
            atom_idx = int(m.group(1))
            element = m.group(2)
            wfc_idx = int(m.group(3))
            orbital = m.group(4)

            data = np.loadtxt(fpath, comments="#")
            energy = data[:, 0]
            # Column 1: total LDOS for this projection
            # Columns 2+: m-resolved (m=-l,...,+l) if available
            ldos = data[:, 1]

            pdos_dict[(atom_idx, element, wfc_idx, orbital)] = {
                "energy": energy,
                "ldos": ldos,
                "m_resolved": data[:, 2:] if data.shape[1] > 2 else None,
            }

    # Total DOS
    tot_file = f"{prefix}_pdos.pdos_tot"
    total_dos = None
    if os.path.exists(tot_file):
        data = np.loadtxt(tot_file, comments="#")
        total_dos = {"energy": data[:, 0], "dos": data[:, 1]}

    return pdos_dict, total_dos

pdos_dict, total_dos = parse_all_pdos(PREFIX)
print(f"Parsed {len(pdos_dict)} PDOS projections")

# ── Group by element and orbital ───────────────────────────────────
def group_pdos_by_element_orbital(pdos_dict):
    """Sum PDOS over all atoms of the same element, per orbital."""
    grouped = {}
    for (atom_idx, element, wfc_idx, orbital), data in pdos_dict.items():
        key = (element, orbital)
        if key not in grouped:
            grouped[key] = {"energy": data["energy"], "pdos": np.zeros_like(data["ldos"])}
        grouped[key]["pdos"] += data["ldos"]
    return grouped

def group_pdos_by_element(pdos_dict):
    """Sum all orbitals for each element."""
    grouped = {}
    for (atom_idx, element, wfc_idx, orbital), data in pdos_dict.items():
        if element not in grouped:
            grouped[element] = {"energy": data["energy"], "pdos": np.zeros_like(data["ldos"])}
        grouped[element]["pdos"] += data["ldos"]
    return grouped

def group_pdos_by_atom(pdos_dict):
    """Sum all orbitals for each individual atom."""
    grouped = {}
    for (atom_idx, element, wfc_idx, orbital), data in pdos_dict.items():
        key = (atom_idx, element)
        if key not in grouped:
            grouped[key] = {"energy": data["energy"], "pdos": np.zeros_like(data["ldos"])}
        grouped[key]["pdos"] += data["ldos"]
    return grouped

elem_orb = group_pdos_by_element_orbital(pdos_dict)
elem_total = group_pdos_by_element(pdos_dict)
atom_total = group_pdos_by_atom(pdos_dict)

# ── Orbital color scheme ──────────────────────────────────────────
orbital_colors = {
    "s": "#e74c3c",
    "p": "#3498db",
    "d": "#2ecc71",
    "f": "#9b59b6",
}

# ══════════════════════════════════════════════════════════════════
#  PLOT 1: Stacked Orbital PDOS per Element
# ══════════════════════════════════════════════════════════════════
elements = sorted(set(el for (el, orb) in elem_orb.keys()))
n_elements = len(elements)

fig, axes = plt.subplots(n_elements + 1, 1, figsize=(8, 3.5 * (n_elements + 1)),
                         sharex=True, gridspec_kw={"hspace": 0.05})
if n_elements + 1 == 1:
    axes = [axes]

# Top panel: Total DOS
ax = axes[0]
if total_dos is not None:
    ax.plot(total_dos["energy"] - e_fermi, total_dos["dos"],
            color="black", linewidth=1.2, label="Total DOS")
    ax.fill_between(total_dos["energy"] - e_fermi, total_dos["dos"],
                    alpha=0.1, color="gray")
ax.axvline(0, color="red", linestyle="--", linewidth=0.8)
ax.set_ylabel("DOS\n(states/eV)", fontsize=11)
ax.set_title("Projected Density of States", fontsize=14)
ax.legend(fontsize=10, loc="upper right")
ax.set_ylim(0, None)
ax.grid(alpha=0.3)

# Element panels: stacked orbital contributions
for i, element in enumerate(elements):
    ax = axes[i + 1]

    # Collect orbitals for this element
    orbitals_for_element = {}
    for (el, orb), data in sorted(elem_orb.items()):
        if el == element:
            orbitals_for_element[orb] = data

    # Plot as stacked area
    energy = None
    cumulative = None
    for orb in ["s", "p", "d", "f"]:
        if orb not in orbitals_for_element:
            continue
        data = orbitals_for_element[orb]
        energy = data["energy"] - e_fermi
        pdos = data["pdos"]
        color = orbital_colors.get(orb, "gray")

        if cumulative is None:
            cumulative = np.zeros_like(pdos)

        ax.fill_between(energy, cumulative, cumulative + pdos,
                        alpha=0.4, color=color, label=f"{element} ({orb})")
        ax.plot(energy, cumulative + pdos, color=color, linewidth=0.8)
        cumulative = cumulative + pdos

    ax.axvline(0, color="red", linestyle="--", linewidth=0.8)
    ax.set_ylabel(f"{element}\nPDOS", fontsize=11)
    ax.legend(fontsize=9, ncol=4, loc="upper right")
    ax.set_ylim(0, None)
    ax.grid(alpha=0.3)

axes[-1].set_xlabel("$E - E_F$ (eV)", fontsize=13)
axes[-1].set_xlim(-12, 8)

plt.savefig("pdos_stacked_orbital.png", dpi=200, bbox_inches="tight")
print("Saved: pdos_stacked_orbital.png")

# ══════════════════════════════════════════════════════════════════
#  PLOT 2: LDOS for Selected Atoms
# ══════════════════════════════════════════════════════════════════
print("\nPlotting LDOS for individual atoms...")

fig, ax = plt.subplots(figsize=(8, 5))
colors_cycle = plt.cm.tab10(np.linspace(0, 1, min(len(atom_total), 10)))

for i, ((atom_idx, element), data) in enumerate(sorted(atom_total.items())):
    ax.plot(data["energy"] - e_fermi, data["pdos"],
            linewidth=1.2, label=f"Atom {atom_idx} ({element})",
            color=colors_cycle[i % len(colors_cycle)])

ax.axvline(0, color="red", linestyle="--", linewidth=0.8)
ax.set_xlabel("$E - E_F$ (eV)", fontsize=13)
ax.set_ylabel("LDOS (states/eV)", fontsize=13)
ax.set_title("Local DOS by Atom", fontsize=14)
ax.set_xlim(-12, 8)
ax.set_ylim(0, None)
ax.legend(fontsize=9, ncol=2, loc="upper right")
ax.grid(alpha=0.3)

plt.tight_layout()
plt.savefig("pdos_ldos_atoms.png", dpi=200, bbox_inches="tight")
print("Saved: pdos_ldos_atoms.png")

# ══════════════════════════════════════════════════════════════════
#  PLOT 3: LDOS for User-Specified Atom Group (e.g., surface layer)
# ══════════════════════════════════════════════════════════════════
print("\nPlotting LDOS for selected atom group...")

# Define atom indices for the group (1-based, matching QE numbering)
SELECTED_ATOMS = [1, 2, 3]  # Modify for your system

fig, ax = plt.subplots(figsize=(8, 5))
group_pdos = None
group_energy = None

for (atom_idx, element, wfc_idx, orbital), data in pdos_dict.items():
    if atom_idx in SELECTED_ATOMS:
        if group_energy is None:
            group_energy = data["energy"]
            group_pdos = np.zeros_like(data["ldos"])
        group_pdos += data["ldos"]

if group_pdos is not None:
    ax.plot(group_energy - e_fermi, group_pdos,
            color="steelblue", linewidth=1.5,
            label=f"LDOS (atoms {SELECTED_ATOMS})")
    ax.fill_between(group_energy - e_fermi, group_pdos, alpha=0.15, color="steelblue")

    # Also plot total for comparison
    if total_dos is not None:
        ax.plot(total_dos["energy"] - e_fermi, total_dos["dos"],
                color="gray", linewidth=0.8, alpha=0.5, label="Total DOS")

ax.axvline(0, color="red", linestyle="--", linewidth=0.8)
ax.set_xlabel("$E - E_F$ (eV)", fontsize=13)
ax.set_ylabel("LDOS (states/eV)", fontsize=13)
ax.set_title("Local DOS for Selected Atoms", fontsize=14)
ax.set_xlim(-12, 8)
ax.set_ylim(0, None)
ax.legend(fontsize=11)
ax.grid(alpha=0.3)

plt.tight_layout()
plt.savefig("pdos_ldos_group.png", dpi=200, bbox_inches="tight")
print("Saved: pdos_ldos_group.png")
```

#### Step A3: Inverse Participation Ratio from projwfc.x

```python
#!/usr/bin/env python3
"""
Compute inverse participation ratio (IPR) from projwfc.x lowdin charges.
IPR measures the spatial localization of electronic states.
IPR ~ 1/N for fully delocalized states, IPR ~ 1 for fully localized states.

The IPR at energy E is defined as:
  IPR(E) = sum_i |<i|psi>|^4 / (sum_i |<i|psi>|^2)^2

where i runs over atoms. We approximate this from the PDOS:
  IPR(E) ~ sum_i [PDOS_i(E)]^2 / [sum_i PDOS_i(E)]^2
"""
import os
import re
import glob
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

PREFIX = "srtio3"

# Parse Fermi energy
e_fermi = 0.0
with open(f"{PREFIX}_nscf.out", "r") as f:
    for line in f:
        if "the Fermi energy is" in line:
            m = re.search(r"is\s+([-\d.]+)", line)
            if m:
                e_fermi = float(m.group(1))
        if "highest occupied, lowest unoccupied" in line:
            m = re.search(r":\s+([-\d.]+)\s+([-\d.]+)", line)
            if m:
                e_fermi = (float(m.group(1)) + float(m.group(2))) / 2

# Parse PDOS per atom (sum all orbitals per atom)
atom_pdos = {}
files = sorted(glob.glob(f"{PREFIX}_pdos.pdos_atm#*"))
for fpath in files:
    fname = os.path.basename(fpath)
    m = re.search(r"atm#(\d+)\((\w+)\)_wfc#(\d+)\((\w+)\)", fname)
    if m:
        atom_idx = int(m.group(1))
        data = np.loadtxt(fpath, comments="#")
        energy = data[:, 0]
        pdos = data[:, 1]
        if atom_idx not in atom_pdos:
            atom_pdos[atom_idx] = np.zeros_like(pdos)
        atom_pdos[atom_idx] += pdos

if not atom_pdos:
    print("No PDOS files found. Run projwfc.x first.")
    exit(1)

# Compute IPR
n_atoms = len(atom_pdos)
atom_indices = sorted(atom_pdos.keys())
pdos_matrix = np.array([atom_pdos[idx] for idx in atom_indices])  # shape: (n_atoms, n_energy)

# IPR(E) = sum_i [PDOS_i(E)]^2 / [sum_i PDOS_i(E)]^2
numerator = np.sum(pdos_matrix**2, axis=0)
denominator = np.sum(pdos_matrix, axis=0)**2
# Avoid division by zero
mask = denominator > 1e-20
ipr = np.zeros_like(energy)
ipr[mask] = numerator[mask] / denominator[mask]

# Plot
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 7), sharex=True,
                                gridspec_kw={"hspace": 0.05})

# Top: total DOS
total_pdos = np.sum(pdos_matrix, axis=0)
ax1.plot(energy - e_fermi, total_pdos, color="steelblue", linewidth=1.2)
ax1.fill_between(energy - e_fermi, total_pdos, alpha=0.15, color="steelblue")
ax1.axvline(0, color="red", linestyle="--", linewidth=0.8)
ax1.set_ylabel("DOS (states/eV)", fontsize=12)
ax1.set_title("DOS and Inverse Participation Ratio", fontsize=14)
ax1.set_ylim(0, None)
ax1.grid(alpha=0.3)

# Bottom: IPR
ax2.plot(energy - e_fermi, ipr, color="darkorange", linewidth=1.0)
ax2.axhline(1.0 / n_atoms, color="gray", linestyle=":", linewidth=0.8,
            label=f"Fully delocalized (1/{n_atoms})")
ax2.axvline(0, color="red", linestyle="--", linewidth=0.8)
ax2.set_xlabel("$E - E_F$ (eV)", fontsize=13)
ax2.set_ylabel("IPR", fontsize=12)
ax2.set_xlim(-12, 8)
ax2.set_ylim(0, min(1.0, np.max(ipr[mask]) * 1.2))
ax2.legend(fontsize=10)
ax2.grid(alpha=0.3)

plt.savefig("pdos_ipr.png", dpi=200, bbox_inches="tight")
print("Saved: pdos_ipr.png")
print(f"\nIPR ranges from {1.0/n_atoms:.4f} (fully delocalized) to 1.0 (fully localized)")
print(f"Number of atoms: {n_atoms}")
```

### Method B: VASP PDOS from vasprun.xml with pymatgen

#### Step B1: VASP Input Preparation for PDOS

```python
#!/usr/bin/env python3
"""
Generate VASP input files for a PDOS calculation.
Two-step process: (1) SCF with standard k-grid, (2) NSCF with dense k-grid and LORBIT=11.
"""
from pymatgen.core import Structure
from pymatgen.io.vasp.sets import MPStaticSet
import os
import json

# ── Load structure ─────────────────────────────────────────────────
# From a CIF, POSCAR, or Materials Project
struct = Structure.from_file("POSCAR")  # or "relaxed.cif"
print(f"Structure: {struct.formula}, {struct.num_sites} atoms")

# ── Step 1: SCF calculation ────────────────────────────────────────
scf_dir = "01_scf"
os.makedirs(scf_dir, exist_ok=True)

scf_set = MPStaticSet(struct, user_incar_settings={
    "ENCUT": 520,
    "EDIFF": 1e-6,
    "ISMEAR": -5,       # Tetrahedron with Blochl corrections (insulators)
    # "ISMEAR": 1,       # Methfessel-Paxton (metals)
    # "SIGMA": 0.2,      # Smearing width for metals
    "LWAVE": True,       # Write WAVECAR for NSCF restart
    "LCHARG": True,      # Write CHGCAR for NSCF restart
    "LORBIT": 11,        # Write PDOS in DOSCAR and PROCAR
    "NEDOS": 3001,       # Number of DOS grid points
    "EMIN": -15,         # DOS energy range min
    "EMAX": 10,          # DOS energy range max
})
scf_set.write_input(scf_dir)
print(f"SCF inputs written to {scf_dir}/")

# ── Step 2: NSCF on denser k-grid for smooth DOS ──────────────────
nscf_dir = "02_nscf_dos"
os.makedirs(nscf_dir, exist_ok=True)

nscf_set = MPStaticSet(struct, user_incar_settings={
    "ENCUT": 520,
    "EDIFF": 1e-6,
    "ISMEAR": -5,
    "LWAVE": False,
    "LCHARG": False,
    "LORBIT": 11,        # Atom+orbital projected DOS
    "NEDOS": 3001,
    "EMIN": -15,
    "EMAX": 10,
    "ICHARG": 11,        # Read CHGCAR from SCF (non-self-consistent)
    "ISYM": -1,          # Turn off symmetry for proper PDOS weights
}, user_kpoints_settings={"reciprocal_density": 1000})  # Denser k-mesh
nscf_set.write_input(nscf_dir)
print(f"NSCF inputs written to {nscf_dir}/")
print("NOTE: Copy CHGCAR from SCF directory to NSCF directory before running.")

# ── LORBIT options reference ───────────────────────────────────────
lorbit_info = """
LORBIT options:
  10 : PDOS decomposed by atom and l-quantum number (s, p, d, f)
  11 : PDOS decomposed by atom and lm-quantum number (s, px, py, pz, dxy, ...)
  12 : Same as 11 but with phase factors (for orbital-projected band structure)
  13 : Like 11, requires PAW pseudopotentials with proper projectors
  14 : Like 12 with phase factors and PAW
"""
print(lorbit_info)
```

#### Step B2: Parse VASP PDOS with pymatgen

```python
#!/usr/bin/env python3
"""
Parse VASP PDOS from vasprun.xml using pymatgen.
Produces element-projected, orbital-projected, and site-projected DOS.
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pymatgen.io.vasp import Vasprun
from pymatgen.electronic_structure.dos import CompleteDos
from pymatgen.electronic_structure.core import Spin, OrbitalType

# ── Parse vasprun.xml ──────────────────────────────────────────────
print("Parsing vasprun.xml ...")
vrun = Vasprun("vasprun.xml", parse_dos=True, parse_projected_eigen=True)
complete_dos = vrun.complete_dos
e_fermi = vrun.efermi
print(f"Fermi energy: {e_fermi:.4f} eV")
print(f"Band gap: {complete_dos.get_gap():.4f} eV")

# ── Helper to extract arrays from Dos objects ──────────────────────
def dos_to_arrays(dos_obj, e_fermi=0.0):
    """Convert a pymatgen Dos object to (energy, dos_up, dos_down) arrays."""
    energies = np.array(dos_obj.energies) - e_fermi
    densities_up = np.array(dos_obj.densities[Spin.up])
    densities_down = None
    if Spin.down in dos_obj.densities:
        densities_down = np.array(dos_obj.densities[Spin.down])
    return energies, densities_up, densities_down

# ══════════════════════════════════════════════════════════════════
#  PLOT 1: Element-Projected DOS
# ══════════════════════════════════════════════════════════════════
print("\nPlotting element-projected DOS...")
element_dos = complete_dos.get_element_dos()

fig, ax = plt.subplots(figsize=(8, 5))
colors_elem = plt.cm.Set1(np.linspace(0, 0.8, len(element_dos)))

for i, (element, dos_obj) in enumerate(sorted(element_dos.items(), key=lambda x: str(x[0]))):
    energies, dos_up, dos_down = dos_to_arrays(dos_obj, e_fermi)
    ax.plot(energies, dos_up, linewidth=1.5, label=str(element), color=colors_elem[i])
    ax.fill_between(energies, dos_up, alpha=0.1, color=colors_elem[i])
    if dos_down is not None:
        ax.plot(energies, -dos_down, linewidth=1.5, color=colors_elem[i])
        ax.fill_between(energies, -dos_down, alpha=0.1, color=colors_elem[i])

# Total DOS
total_e, total_up, total_down = dos_to_arrays(complete_dos, e_fermi)
ax.plot(total_e, total_up, color="black", linewidth=0.8, alpha=0.5, label="Total")

ax.axvline(0, color="red", linestyle="--", linewidth=0.8, label="$E_F$")
ax.set_xlabel("$E - E_F$ (eV)", fontsize=13)
ax.set_ylabel("DOS (states/eV)", fontsize=13)
ax.set_title("Element-Projected DOS (VASP)", fontsize=14)
ax.set_xlim(-12, 8)
ax.legend(fontsize=11)
ax.grid(alpha=0.3)

plt.tight_layout()
plt.savefig("vasp_pdos_element.png", dpi=200, bbox_inches="tight")
print("Saved: vasp_pdos_element.png")

# ══════════════════════════════════════════════════════════════════
#  PLOT 2: Orbital-Projected DOS per Element (spd decomposition)
# ══════════════════════════════════════════════════════════════════
print("\nPlotting element+orbital PDOS (spd decomposition)...")
element_spd = complete_dos.get_element_spd_dos()

orbital_colors = {
    OrbitalType.s: "#e74c3c",
    OrbitalType.p: "#3498db",
    OrbitalType.d: "#2ecc71",
    OrbitalType.f: "#9b59b6",
}
orbital_names = {OrbitalType.s: "s", OrbitalType.p: "p",
                 OrbitalType.d: "d", OrbitalType.f: "f"}

elements = sorted(element_spd.keys(), key=str)
fig, axes = plt.subplots(len(elements), 1, figsize=(8, 3.5 * len(elements)),
                         sharex=True, gridspec_kw={"hspace": 0.05})
if len(elements) == 1:
    axes = [axes]

for idx, element in enumerate(elements):
    ax = axes[idx]
    spd_dos = element_spd[element]

    for orb_type, dos_obj in sorted(spd_dos.items(), key=lambda x: x[0].value):
        energies, dos_up, dos_down = dos_to_arrays(dos_obj, e_fermi)
        color = orbital_colors.get(orb_type, "gray")
        label = f"{element} ({orbital_names.get(orb_type, '?')})"
        ax.plot(energies, dos_up, linewidth=1.5, label=label, color=color)
        ax.fill_between(energies, dos_up, alpha=0.15, color=color)
        if dos_down is not None:
            ax.plot(energies, -dos_down, linewidth=1.5, color=color)

    ax.axvline(0, color="red", linestyle="--", linewidth=0.8)
    ax.set_ylabel(f"{element}\nPDOS", fontsize=11)
    ax.legend(fontsize=9, ncol=4, loc="upper right")
    ax.grid(alpha=0.3)

axes[-1].set_xlabel("$E - E_F$ (eV)", fontsize=13)
axes[-1].set_xlim(-12, 8)
axes[0].set_title("Orbital-Projected DOS by Element (VASP)", fontsize=14)

plt.savefig("vasp_pdos_spd.png", dpi=200, bbox_inches="tight")
print("Saved: vasp_pdos_spd.png")

# ══════════════════════════════════════════════════════════════════
#  PLOT 3: Site-Projected DOS (LDOS for specific atoms)
# ══════════════════════════════════════════════════════════════════
print("\nPlotting site-projected DOS ...")

structure = vrun.final_structure
# Select atoms by index (0-based in pymatgen)
selected_sites = [0, 1, 2]  # Modify for your system

fig, ax = plt.subplots(figsize=(8, 5))
colors_site = plt.cm.tab10(np.linspace(0, 1, len(selected_sites)))

for i, site_idx in enumerate(selected_sites):
    site = structure[site_idx]
    site_dos = complete_dos.get_site_dos(site)
    energies, dos_up, dos_down = dos_to_arrays(site_dos, e_fermi)
    label = f"Site {site_idx}: {site.specie} ({site.frac_coords[0]:.2f}, {site.frac_coords[1]:.2f}, {site.frac_coords[2]:.2f})"
    ax.plot(energies, dos_up, linewidth=1.2, label=label, color=colors_site[i])

ax.axvline(0, color="red", linestyle="--", linewidth=0.8, label="$E_F$")
ax.set_xlabel("$E - E_F$ (eV)", fontsize=13)
ax.set_ylabel("LDOS (states/eV)", fontsize=13)
ax.set_title("Site-Projected DOS (LDOS)", fontsize=14)
ax.set_xlim(-12, 8)
ax.set_ylim(0, None)
ax.legend(fontsize=9)
ax.grid(alpha=0.3)

plt.tight_layout()
plt.savefig("vasp_pdos_site.png", dpi=200, bbox_inches="tight")
print("Saved: vasp_pdos_site.png")
```

#### Step B3: Parse VASP DOSCAR Directly (Without pymatgen)

```python
#!/usr/bin/env python3
"""
Parse VASP DOSCAR file directly for PDOS.
Useful when vasprun.xml is not available or too large.
DOSCAR format: header (6 lines), then total DOS block, then PDOS blocks per atom.
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

def parse_doscar(filename="DOSCAR", n_atoms=None):
    """
    Parse VASP DOSCAR.
    Returns:
        total_dos: dict with 'energy', 'dos', 'integrated_dos'
        pdos_list: list of dicts, one per atom. Each has 'energy' and orbital keys.
    """
    with open(filename, "r") as f:
        lines = f.readlines()

    # Line 1: number of atoms
    if n_atoms is None:
        n_atoms = int(lines[0].split()[0])

    # Line 6: NEDOS, Efermi, etc.
    parts = lines[5].split()
    e_max = float(parts[0])
    e_min = float(parts[1])
    nedos = int(parts[2])
    e_fermi = float(parts[3])

    # Total DOS: lines 6+1 to 6+NEDOS
    total_data = []
    for i in range(6, 6 + nedos):
        total_data.append([float(x) for x in lines[i].split()])
    total_data = np.array(total_data)

    total_dos = {
        "energy": total_data[:, 0],
        "dos_up": total_data[:, 1],
        "e_fermi": e_fermi,
    }
    if total_data.shape[1] >= 5:
        # Spin-polarized: E, DOS_up, DOS_down, INT_up, INT_down
        total_dos["dos_down"] = total_data[:, 2]
        total_dos["int_dos_up"] = total_data[:, 3]
        total_dos["int_dos_down"] = total_data[:, 4]
    else:
        # Non-spin-polarized: E, DOS, INT_DOS
        total_dos["int_dos"] = total_data[:, 2]

    # PDOS per atom
    pdos_list = []
    offset = 6 + nedos
    for atom_idx in range(n_atoms):
        # Skip the header line for this atom
        offset += 1
        atom_data = []
        for i in range(offset, offset + nedos):
            atom_data.append([float(x) for x in lines[i].split()])
        atom_data = np.array(atom_data)
        offset += nedos

        pdos = {"energy": atom_data[:, 0]}

        # LORBIT=11 non-spin-polarized: E, s, py, pz, px, dxy, dyz, dz2, dxz, dx2-y2
        # LORBIT=11 spin-polarized: E, s_up, s_down, py_up, py_down, ...
        ncols = atom_data.shape[1] - 1  # exclude energy column

        if ncols == 9:
            # Non-spin-polarized, lm-decomposed (LORBIT=11)
            pdos["s"] = atom_data[:, 1]
            pdos["py"] = atom_data[:, 2]
            pdos["pz"] = atom_data[:, 3]
            pdos["px"] = atom_data[:, 4]
            pdos["p"] = atom_data[:, 2] + atom_data[:, 3] + atom_data[:, 4]
            pdos["dxy"] = atom_data[:, 5]
            pdos["dyz"] = atom_data[:, 6]
            pdos["dz2"] = atom_data[:, 7]
            pdos["dxz"] = atom_data[:, 8]
            pdos["dx2-y2"] = atom_data[:, 9]
            pdos["d"] = atom_data[:, 5] + atom_data[:, 6] + atom_data[:, 7] + \
                        atom_data[:, 8] + atom_data[:, 9]
        elif ncols == 3:
            # Non-spin-polarized, l-decomposed (LORBIT=10)
            pdos["s"] = atom_data[:, 1]
            pdos["p"] = atom_data[:, 2]
            pdos["d"] = atom_data[:, 3]
        elif ncols == 16:
            # f-orbitals included (LORBIT=11, f-element)
            pdos["s"] = atom_data[:, 1]
            pdos["p"] = atom_data[:, 2] + atom_data[:, 3] + atom_data[:, 4]
            pdos["d"] = atom_data[:, 5] + atom_data[:, 6] + atom_data[:, 7] + \
                        atom_data[:, 8] + atom_data[:, 9]
            pdos["f"] = np.sum(atom_data[:, 10:17], axis=1)

        pdos_list.append(pdos)

    return total_dos, pdos_list


# ── Usage example ──────────────────────────────────────────────────
total_dos, pdos_list = parse_doscar("DOSCAR")
e_fermi = total_dos["e_fermi"]
print(f"Fermi energy: {e_fermi:.4f} eV")
print(f"Number of atoms: {len(pdos_list)}")

# Plot total + PDOS for first few atoms
fig, ax = plt.subplots(figsize=(8, 5))
energy = total_dos["energy"] - e_fermi
ax.plot(energy, total_dos["dos_up"], color="black", linewidth=1.0, alpha=0.5, label="Total")

orbital_colors = {"s": "#e74c3c", "p": "#3498db", "d": "#2ecc71", "f": "#9b59b6"}
for i, pdos in enumerate(pdos_list[:5]):
    for orb in ["s", "p", "d", "f"]:
        if orb in pdos:
            ax.plot(pdos["energy"] - e_fermi, pdos[orb],
                    linewidth=1.0, alpha=0.7,
                    color=orbital_colors[orb],
                    label=f"Atom {i+1} ({orb})" if i == 0 else None)

ax.axvline(0, color="red", linestyle="--", linewidth=0.8)
ax.set_xlabel("$E - E_F$ (eV)", fontsize=13)
ax.set_ylabel("DOS (states/eV)", fontsize=13)
ax.set_title("VASP DOSCAR PDOS", fontsize=14)
ax.set_xlim(-12, 8)
ax.set_ylim(0, None)
ax.legend(fontsize=9, ncol=2)
ax.grid(alpha=0.3)

plt.tight_layout()
plt.savefig("vasp_doscar_pdos.png", dpi=200, bbox_inches="tight")
print("Saved: vasp_doscar_pdos.png")
```

### Method C: VASPKIT Workflows (112--120)

#### Step C1: VASPKIT PDOS Post-Processing

```bash
#!/bin/bash
# VASPKIT post-processing for PDOS
# Prerequisites: completed VASP calculation with LORBIT=11, ISYM=-1
# VASPKIT reads vasprun.xml and DOSCAR automatically from current directory.

# ── Task 112: Element-projected DOS ────────────────────────────────
echo "112" | vaspkit
# Generates: EDOS_*, one file per element with columns: Energy, s, p, d, f, total

# ── Task 113: Element-projected DOS (spin-resolved) ────────────────
# For ISPIN=2 calculations:
echo "113" | vaspkit
# Generates: EDOS_UP_*, EDOS_DOWN_* for each element

# ── Task 114: Atom-projected DOS ───────────────────────────────────
echo "114" | vaspkit
# Generates: PDOS_ATM#_*, one file per atom

# ── Task 115: Orbital-projected DOS for selected atoms ─────────────
echo "115" | vaspkit
# Interactive: select atoms and orbitals. Generates PDOS files.

# ── Task 116: Local DOS for selected atoms ─────────────────────────
echo "116" | vaspkit
# Interactive: select atom indices. Generates LDOS_* files.
# Useful for surface/interface atoms.

# ── Task 119: k-resolved projected DOS ─────────────────────────────
echo "119" | vaspkit
# Generates k-resolved PDOS for band-character analysis.
# Requires PROCAR from a bands calculation.

# ── Task 120: Band-resolved projected DOS ──────────────────────────
echo "120" | vaspkit
# Generates PDOS resolved per band index.
# Useful to identify orbital character of specific bands.
```

#### Step C2: VASPKIT IPR (Task 110)

```bash
#!/bin/bash
# VASPKIT Task 110: Inverse Participation Ratio
# Quantifies localization of electronic states.
# Reads PROCAR from a calculation with LORBIT=11.

echo "110" | vaspkit
# Generates: IPR.dat with columns: Energy, IPR
# IPR close to 1/N_atoms indicates delocalized states.
# IPR close to 1 indicates localized states.
```

#### Step C3: Plot VASPKIT Output Files

```python
#!/usr/bin/env python3
"""
Plot VASPKIT PDOS output files (EDOS_*, PDOS_ATM#_*).
VASPKIT output format: header lines starting with #, then columns of data.
"""
import os
import glob
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ── Parse VASPKIT EDOS files (element-projected) ───────────────────
def parse_vaspkit_edos(filename):
    """Parse VASPKIT EDOS_Element file."""
    data = np.loadtxt(filename, comments="#")
    # Columns: Energy, s, p, d, [f], total
    result = {"energy": data[:, 0]}
    if data.shape[1] == 5:
        result["s"] = data[:, 1]
        result["p"] = data[:, 2]
        result["d"] = data[:, 3]
        result["total"] = data[:, 4]
    elif data.shape[1] == 6:
        result["s"] = data[:, 1]
        result["p"] = data[:, 2]
        result["d"] = data[:, 3]
        result["f"] = data[:, 4]
        result["total"] = data[:, 5]
    return result

# Find all EDOS files
edos_files = sorted(glob.glob("EDOS_*"))
if not edos_files:
    print("No EDOS_* files found. Run VASPKIT task 112 first.")
    exit(1)

orbital_colors = {"s": "#e74c3c", "p": "#3498db", "d": "#2ecc71", "f": "#9b59b6"}

fig, axes = plt.subplots(len(edos_files), 1, figsize=(8, 3.5 * len(edos_files)),
                         sharex=True, gridspec_kw={"hspace": 0.05})
if len(edos_files) == 1:
    axes = [axes]

for i, fpath in enumerate(edos_files):
    ax = axes[i]
    element = os.path.basename(fpath).replace("EDOS_", "")
    data = parse_vaspkit_edos(fpath)

    cumulative = np.zeros_like(data["energy"])
    for orb in ["s", "p", "d", "f"]:
        if orb in data:
            ax.fill_between(data["energy"], cumulative, cumulative + data[orb],
                            alpha=0.4, color=orbital_colors[orb],
                            label=f"{element} ({orb})")
            cumulative += data[orb]

    ax.axvline(0, color="red", linestyle="--", linewidth=0.8)
    ax.set_ylabel(f"{element}\nPDOS", fontsize=11)
    ax.legend(fontsize=9, ncol=4, loc="upper right")
    ax.set_ylim(0, None)
    ax.grid(alpha=0.3)

axes[-1].set_xlabel("$E - E_F$ (eV)", fontsize=13)
axes[-1].set_xlim(-12, 8)
axes[0].set_title("Element-Projected DOS (VASPKIT)", fontsize=14)

plt.savefig("vaspkit_edos.png", dpi=200, bbox_inches="tight")
print("Saved: vaspkit_edos.png")
```

### Complete Workflow: QE PDOS from Structure to Plot

```python
#!/usr/bin/env python3
"""
Complete QE PDOS workflow: SCF -> NSCF -> projwfc.x -> stacked PDOS plot.
Example: SrTiO3 perovskite.
"""
import os
import subprocess
import re
import glob
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pymatgen.core import Structure, Lattice

# ══════════════════════════════════════════════════════════════════
#  Configuration
# ══════════════════════════════════════════════════════════════════
PSEUDO_DIR = os.path.abspath("./pseudo")
OUTDIR = os.path.abspath("./tmp_pdos")
os.makedirs(OUTDIR, exist_ok=True)
PREFIX = "srtio3"
ECUTWFC = 60.0
ECUTRHO = 480.0
NPROC = 4
NBND = 40

# ── Build SrTiO3 structure ─────────────────────────────────────────
a = 3.905  # Angstrom
struct = Structure(
    lattice=Lattice.cubic(a),
    species=["Sr", "Ti", "O", "O", "O"],
    coords=[
        [0.0, 0.0, 0.0],
        [0.5, 0.5, 0.5],
        [0.5, 0.5, 0.0],
        [0.5, 0.0, 0.5],
        [0.0, 0.5, 0.5],
    ],
)

# Build QE input cards
from ase.data import atomic_masses, atomic_numbers

unique_elements = list(dict.fromkeys([str(s) for s in struct.species]))
species_lines = []
for el in unique_elements:
    z = atomic_numbers[el]
    mass = atomic_masses[z]
    species_lines.append(f"  {el}  {mass:.4f}  {el}.upf")
species_card = "\n".join(species_lines)

pos_lines = []
for site in struct:
    fc = site.frac_coords
    pos_lines.append(f"  {site.specie}  {fc[0]:.10f}  {fc[1]:.10f}  {fc[2]:.10f}")
pos_card = "\n".join(pos_lines)

cell_lines = []
for row in struct.lattice.matrix:
    cell_lines.append(f"  {row[0]:.10f}  {row[1]:.10f}  {row[2]:.10f}")
cell_card = "\n".join(cell_lines)

nat = struct.num_sites
ntyp = len(unique_elements)

# ══════════════════════════════════════════════════════════════════
#  Step 1: SCF
# ══════════════════════════════════════════════════════════════════
scf_input = f"""&CONTROL
    calculation = 'scf'
    prefix      = '{PREFIX}'
    outdir      = '{OUTDIR}'
    pseudo_dir  = '{PSEUDO_DIR}'
/
&SYSTEM
    ibrav       = 0
    nat         = {nat}
    ntyp        = {ntyp}
    ecutwfc     = {ECUTWFC}
    ecutrho     = {ECUTRHO}
    occupations = 'smearing'
    smearing    = 'cold'
    degauss     = 0.005
/
&ELECTRONS
    conv_thr = 1.0d-8
/
CELL_PARAMETERS (angstrom)
{cell_card}

ATOMIC_SPECIES
{species_card}

ATOMIC_POSITIONS (crystal)
{pos_card}

K_POINTS (automatic)
  8 8 8  0 0 0
"""
with open(f"{PREFIX}_scf.in", "w") as f:
    f.write(scf_input)

print("[1/4] Running SCF ...")
r = subprocess.run(["mpirun", "-np", str(NPROC), "pw.x", "-in", f"{PREFIX}_scf.in"],
                    capture_output=True, text=True, timeout=1200)
with open(f"{PREFIX}_scf.out", "w") as f:
    f.write(r.stdout)
assert "convergence has been achieved" in r.stdout, "SCF did not converge!"
print("      SCF converged.")

# ══════════════════════════════════════════════════════════════════
#  Step 2: NSCF (dense k-grid)
# ══════════════════════════════════════════════════════════════════
nscf_input = f"""&CONTROL
    calculation = 'nscf'
    prefix      = '{PREFIX}'
    outdir      = '{OUTDIR}'
    pseudo_dir  = '{PSEUDO_DIR}'
    verbosity   = 'high'
/
&SYSTEM
    ibrav       = 0
    nat         = {nat}
    ntyp        = {ntyp}
    ecutwfc     = {ECUTWFC}
    ecutrho     = {ECUTRHO}
    occupations = 'tetrahedra'
    nbnd        = {NBND}
/
&ELECTRONS
    conv_thr = 1.0d-8
/
CELL_PARAMETERS (angstrom)
{cell_card}

ATOMIC_SPECIES
{species_card}

ATOMIC_POSITIONS (crystal)
{pos_card}

K_POINTS (automatic)
  16 16 16  0 0 0
"""
with open(f"{PREFIX}_nscf.in", "w") as f:
    f.write(nscf_input)

print("[2/4] Running NSCF ...")
r = subprocess.run(["mpirun", "-np", str(NPROC), "pw.x", "-in", f"{PREFIX}_nscf.in"],
                    capture_output=True, text=True, timeout=1800)
with open(f"{PREFIX}_nscf.out", "w") as f:
    f.write(r.stdout)
assert r.returncode == 0, "NSCF failed!"
print("      NSCF completed.")

# Extract Fermi energy
e_fermi = 0.0
for line in r.stdout.split("\n"):
    if "the Fermi energy is" in line:
        m = re.search(r"is\s+([-\d.]+)", line)
        if m: e_fermi = float(m.group(1))
    if "highest occupied, lowest unoccupied" in line:
        m = re.search(r":\s+([-\d.]+)\s+([-\d.]+)", line)
        if m: e_fermi = (float(m.group(1)) + float(m.group(2))) / 2
    if "highest occupied" in line and "lowest" not in line:
        m = re.search(r":\s+([-\d.]+)", line)
        if m: e_fermi = float(m.group(1))
print(f"      Fermi energy: {e_fermi:.4f} eV")

# ══════════════════════════════════════════════════════════════════
#  Step 3: projwfc.x
# ══════════════════════════════════════════════════════════════════
projwfc_input = f"""&PROJWFC
    prefix   = '{PREFIX}'
    outdir   = '{OUTDIR}'
    filpdos  = '{PREFIX}_pdos'
    Emin     = {e_fermi - 15.0}
    Emax     = {e_fermi + 10.0}
    DeltaE   = 0.01
    ngauss   = 0
    degauss  = 0.005
/
"""
with open(f"{PREFIX}_projwfc.in", "w") as f:
    f.write(projwfc_input)

print("[3/4] Running projwfc.x ...")
r = subprocess.run(["mpirun", "-np", str(NPROC), "projwfc.x", "-in", f"{PREFIX}_projwfc.in"],
                    capture_output=True, text=True, timeout=600)
with open(f"{PREFIX}_projwfc.out", "w") as f:
    f.write(r.stdout)
assert r.returncode == 0, "projwfc.x failed!"
print("      projwfc.x completed.")

# ══════════════════════════════════════════════════════════════════
#  Step 4: Parse and Plot
# ══════════════════════════════════════════════════════════════════
print("[4/4] Plotting stacked PDOS ...")

# Parse total DOS
tot_file = f"{PREFIX}_pdos.pdos_tot"
tot_data = np.loadtxt(tot_file, comments="#")
energy_tot = tot_data[:, 0]
dos_tot = tot_data[:, 1]

# Parse PDOS files and group by element+orbital
pdos_files = sorted(glob.glob(f"{PREFIX}_pdos.pdos_atm#*"))
elem_orb = {}
for fpath in pdos_files:
    fname = os.path.basename(fpath)
    m = re.search(r"atm#(\d+)\((\w+)\)_wfc#(\d+)\((\w+)\)", fname)
    if m:
        element = m.group(2)
        orbital = m.group(4)
        data = np.loadtxt(fpath, comments="#")
        key = (element, orbital)
        if key not in elem_orb:
            elem_orb[key] = np.zeros_like(data[:, 1])
        elem_orb[key] += data[:, 1]

elements = sorted(set(el for (el, orb) in elem_orb.keys()))
orbital_colors = {"s": "#e74c3c", "p": "#3498db", "d": "#2ecc71", "f": "#9b59b6"}

fig, axes = plt.subplots(len(elements) + 1, 1,
                         figsize=(8, 3.0 * (len(elements) + 1)),
                         sharex=True, gridspec_kw={"hspace": 0.05})

# Total DOS
axes[0].plot(energy_tot - e_fermi, dos_tot, color="black", linewidth=1.2, label="Total")
axes[0].fill_between(energy_tot - e_fermi, dos_tot, alpha=0.1, color="gray")
axes[0].axvline(0, color="red", linestyle="--", linewidth=0.8)
axes[0].set_ylabel("DOS", fontsize=11)
axes[0].set_title("Projected DOS (QE)", fontsize=14)
axes[0].legend(fontsize=10)
axes[0].set_ylim(0, None)
axes[0].grid(alpha=0.3)

# Stacked per element
for i, element in enumerate(elements):
    ax = axes[i + 1]
    cumulative = np.zeros(len(energy_tot))
    for orb in ["s", "p", "d", "f"]:
        key = (element, orb)
        if key not in elem_orb:
            continue
        pdos = elem_orb[key]
        color = orbital_colors.get(orb, "gray")
        ax.fill_between(energy_tot - e_fermi, cumulative, cumulative + pdos,
                        alpha=0.4, color=color, label=f"{element} ({orb})")
        ax.plot(energy_tot - e_fermi, cumulative + pdos, color=color, linewidth=0.8)
        cumulative += pdos

    ax.axvline(0, color="red", linestyle="--", linewidth=0.8)
    ax.set_ylabel(f"{element}\nPDOS", fontsize=11)
    ax.legend(fontsize=9, ncol=4, loc="upper right")
    ax.set_ylim(0, None)
    ax.grid(alpha=0.3)

axes[-1].set_xlabel("$E - E_F$ (eV)", fontsize=13)
axes[-1].set_xlim(-12, 8)

plt.savefig("pdos_stacked_complete.png", dpi=200, bbox_inches="tight")
print("Saved: pdos_stacked_complete.png")
print("\nDone. All PDOS plots generated.")
```

---

## Key Parameters

| Parameter | QE | VASP | Notes |
|---|---|---|---|
| Orbital decomposition | projwfc.x `filpdos` output files | `LORBIT=11` in INCAR | Both give atom+lm-resolved PDOS |
| Dense k-grid | NSCF `K_POINTS (automatic) 16 16 16` | Increase KPOINTS density | Smoother DOS requires denser k-grid |
| Broadening | `degauss` in projwfc.x input (Ry) | `SIGMA` in INCAR (eV) + `NEDOS` | QE broadening in Ry; VASP in eV |
| Energy range | `Emin`, `Emax` in projwfc.x | `EMIN`, `EMAX` in INCAR | Set relative to expected Fermi level |
| Energy resolution | `DeltaE` in projwfc.x (eV) | `NEDOS` in INCAR | More points = finer resolution |
| m-resolved PDOS | Default in projwfc.x output | `LORBIT=11` (or 12 with phase) | Needed for px/py/pz, dxy/dyz/... |
| Symmetry | QE uses symmetry by default | `ISYM=-1` recommended for PDOS | Turn off symmetry for proper PDOS weights |
| Tetrahedron method | `occupations='tetrahedra'` in NSCF | `ISMEAR=-5` | Best for semiconductors/insulators |

## Interpreting Results

- **Orbital character near Fermi level**: The dominant orbital contributions at E_F determine transport and chemical reactivity. In transition metal oxides, look for d-states at E_F (metallic) or a d-d gap (Mott insulator).
- **Crystal field splitting**: For octahedral coordination (e.g., TiO6 in SrTiO3), the d-orbitals split into t2g (dxy, dyz, dxz) and eg (dz2, dx2-y2). The splitting magnitude is visible as two d-state peaks separated in energy.
- **Hybridization**: When peaks from different elements (e.g., Ti-d and O-p) overlap at the same energy, it indicates orbital hybridization and covalent bonding.
- **IPR values**: IPR close to 1/N_atoms indicates fully delocalized band states. IPR approaching 1 indicates localized defect or impurity states.
- **Stacked PDOS**: The sum of all orbital contributions should approximately reproduce the total DOS. Discrepancies indicate interstitial charge not captured by atomic projections.
- **LDOS for specific sites**: Comparing LDOS of surface vs. bulk atoms reveals surface states. In interfaces, layer-resolved LDOS shows band alignment and charge transfer.

## Common Issues

| Problem | Solution |
|---|---|
| PDOS sum does not match total DOS | Some charge is in the interstitial region not assigned to any atom. This is normal for plane-wave DFT. The sum of PDOS is a lower bound. |
| Missing d or f orbital projections | The pseudopotential may not include d/f projectors. Use a PP with the required angular momentum channels. Check the UPF header. |
| Negative PDOS values | Can occur with tetrahedron method at sharp features. Apply a small Gaussian broadening or use a denser k-grid. |
| VASPKIT hangs or asks for input | VASPKIT tasks 115/116 are interactive and require atom selection. Use `echo "115\n1-5" | vaspkit` to provide input non-interactively, or write a small expect-style script. |
| Spin-polarized PDOS asymmetry | Expected for magnetic materials. Plot spin-up as positive, spin-down as negative. The difference gives the local magnetic moment. |
| k-resolved PDOS too large | VASPKIT task 119 generates one file per k-point. Only use for specific analyses; the output can be very large for dense k-grids. |
| projwfc.x out of memory | Reduce `nbnd` or the k-grid density. Large systems with many projections can exhaust memory. Run on more MPI ranks. |
| VASP PDOS depends on LORBIT choice | `LORBIT=10` gives only s/p/d/f totals. `LORBIT=11` gives lm-resolved (px, py, pz, etc.). Use 11 for detailed analysis. |
