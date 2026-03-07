# Inverse Participation Ratio (IPR) Analysis

## When to Use

- Quantify the spatial localization of electronic wave functions
- Identify localized defect states, impurity levels, or trap states in the band structure
- Study Anderson localization in disordered systems (alloys, amorphous materials)
- Distinguish extended Bloch-like states from localized states in complex materials
- Characterize the nature of band-edge states in alloys or heterostructures
- Analyze wave function spreading across sites in doped semiconductors
- Detect mobility edges separating localized and extended states
- Corresponds to VASPKIT task 110

## Method Selection

| Criterion | QE DFT (Method A) | VASP (Method B) |
|---|---|---|
| Orbital projections | projwfc.x (PDOS output) | PROCAR (LORBIT = 11 or 12) |
| IPR computation | Post-process projwfc.x output | Post-process PROCAR or use VASPKIT 110 |
| Supercell support | Yes, any size | Yes, any size |
| Spin-polarized | Yes (projwfc.x handles spin) | Yes (ISPIN = 2) |
| SOC support | Yes (noncolin) | Yes (LSORBIT) |
| Best for | Open-source workflows | VASPKIT 110 automated analysis |

```
Need IPR for a disordered supercell?
  --> Method A (QE) or Method B (VASP): both require DFT

Need automated IPR analysis?
  --> Method B (VASP) with VASPKIT 110

Need to analyze Anderson localization?
  --> Use a large supercell with disorder, compute IPR for all bands

What does IPR tell you?
  --> IPR ~ 1/N  ==> fully delocalized (Bloch state)
  --> IPR ~ 1    ==> fully localized on one site
  --> Intermediate values indicate partial localization

Can MACE help?
  --> Only for structure generation/relaxation before DFT
  --> IPR requires electronic wave functions (not available from MLIPs)
```

## Prerequisites

- Quantum ESPRESSO (pw.x, projwfc.x) -- Method A
- VASP (with LORBIT = 11 or 12 for PROCAR) -- Method B
- VASPKIT (task 110) -- optional, for automated VASP post-processing
- numpy, matplotlib
- A converged SCF/NSCF calculation with sufficient bands

## Detailed Steps

### Method A: QE DFT -- IPR from Projected Wave Functions

#### Complete Workflow

```python
#!/usr/bin/env python3
"""
Compute the Inverse Participation Ratio (IPR) for each electronic state
using Quantum ESPRESSO.

Workflow:
  1. Build or load a structure (optionally disordered)
  2. Run SCF calculation
  3. Run NSCF calculation with many bands on a dense k-grid
  4. Run projwfc.x to get site-projected wave function weights
  5. Compute IPR = sum_i |c_i|^4 / (sum_i |c_i|^2)^2 for each band/k-point
  6. Plot IPR vs energy to identify localized and delocalized states

Example: Si supercell with a vacancy (to create localized defect states).

Definition:
  IPR_n(k) = sum_i |<phi_i | psi_nk>|^4 / (sum_i |<phi_i | psi_nk>|^2)^2

  where phi_i are atomic orbitals on site i, psi_nk is the Bloch state.

  - IPR -> 1/N for a state equally spread over N sites (fully delocalized)
  - IPR -> 1   for a state localized on a single site
"""

import os
import subprocess
import re
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from ase.build import bulk
from ase.io import write as ase_write
from pymatgen.core import Structure
from pymatgen.io.ase import AseAtomsAdaptor
import json

# ============================================================
# 1. Build a structure with a defect (Si vacancy supercell)
# ============================================================
def build_si_vacancy_supercell(supercell_size=(2, 2, 2)):
    """
    Build a Si supercell with one vacancy.

    Parameters
    ----------
    supercell_size : tuple
        Supercell dimensions.

    Returns
    -------
    struct : pymatgen.Structure
        Supercell with a vacancy.
    """
    atoms = bulk("Si", "diamond", a=5.431)
    atoms = atoms.repeat(supercell_size)
    n_atoms = len(atoms)
    print(f"Perfect supercell: {n_atoms} atoms")

    # Remove one atom to create a vacancy
    del atoms[0]
    print(f"Vacancy supercell: {len(atoms)} atoms")

    # Optional: relax with MACE
    from mace.calculators import mace_mp
    from ase.optimize import LBFGS
    from ase.constraints import FixSymmetry

    calc = mace_mp(model="medium", dispersion=False, default_dtype="float64")
    atoms.calc = calc
    opt = LBFGS(atoms, logfile="relax_vacancy.log")
    opt.run(fmax=0.01, steps=200)
    print(f"Relaxation done. Final energy: {atoms.get_potential_energy():.4f} eV")

    adaptor = AseAtomsAdaptor()
    struct = adaptor.get_structure(atoms)
    return struct


# Build the structure
struct = build_si_vacancy_supercell(supercell_size=(2, 2, 2))
print(f"Structure: {struct.formula}, {len(struct)} atoms")

# ============================================================
# 2. Generate QE input files
# ============================================================
PSEUDO_DIR = os.path.abspath("./pseudo")
OUTDIR = os.path.abspath("./tmp_ipr")
os.makedirs(PSEUDO_DIR, exist_ok=True)
os.makedirs(OUTDIR, exist_ok=True)
PREFIX = "si_vac"

n_atoms = len(struct)
cell = struct.lattice.matrix
positions = struct.frac_coords
symbols = [str(s.specie) for s in struct]
elements = sorted(set(symbols))

pseudos = {
    "Si": "Si.pbe-n-rrkjus_psl.1.0.0.UPF",
}

# Download pseudopotentials
for el, pp in pseudos.items():
    pp_path = os.path.join(PSEUDO_DIR, pp)
    if not os.path.exists(pp_path):
        subprocess.run([
            "wget", "-q",
            f"https://pseudopotentials.quantum-espresso.org/upf_files/{pp}",
            "-O", pp_path
        ], check=True)

from pymatgen.core.periodic_table import Element

# Number of bands: include many empty bands to capture high-energy states
n_electrons = sum(Element(el).Z for el in symbols)  # crude estimate
n_bands = max(int(n_atoms * 2.5), n_electrons // 2 + 40)

# --- SCF input ---
scf_input = f"""&CONTROL
    calculation = 'scf'
    prefix      = '{PREFIX}'
    outdir      = '{OUTDIR}'
    pseudo_dir  = '{PSEUDO_DIR}'
    tprnfor     = .true.
/
&SYSTEM
    ibrav       = 0
    nat         = {n_atoms}
    ntyp        = {len(elements)}
    ecutwfc     = 40.0
    ecutrho     = 320.0
    occupations = 'smearing'
    smearing    = 'cold'
    degauss     = 0.01
/
&ELECTRONS
    conv_thr = 1.0d-8
    mixing_beta = 0.4
/

ATOMIC_SPECIES
"""

for el in elements:
    mass = Element(el).atomic_mass
    scf_input += f"  {el}  {mass:.4f}  {pseudos[el]}\n"

scf_input += "\nCELL_PARAMETERS angstrom\n"
for vec in cell:
    scf_input += f"  {vec[0]:.10f}  {vec[1]:.10f}  {vec[2]:.10f}\n"

scf_input += "\nATOMIC_POSITIONS crystal\n"
for sym, fc in zip(symbols, positions):
    scf_input += f"  {sym}  {fc[0]:.10f}  {fc[1]:.10f}  {fc[2]:.10f}\n"

# K-points: Gamma-only for supercell (adjust for smaller cells)
scf_input += "\nK_POINTS automatic\n  2 2 2  0 0 0\n"

with open(f"{PREFIX}_scf.in", "w") as f:
    f.write(scf_input)

# --- NSCF input (more bands, denser k-mesh) ---
nscf_input = scf_input.replace("'scf'", "'nscf'")
nscf_input = nscf_input.replace(
    "degauss     = 0.01",
    f"degauss     = 0.01\n    nbnd        = {n_bands}"
)
nscf_input = nscf_input.replace("2 2 2  0 0 0", "3 3 3  0 0 0")

with open(f"{PREFIX}_nscf.in", "w") as f:
    f.write(nscf_input)

# --- projwfc.x input ---
projwfc_input = f"""&PROJWFC
    prefix  = '{PREFIX}'
    outdir  = '{OUTDIR}'
    filpdos = '{PREFIX}_pdos'
    filproj = '{PREFIX}_proj'
    Emin    = -20.0
    Emax    = 20.0
    DeltaE  = 0.01
    lwrite_overlaps = .false.
/
"""

with open(f"{PREFIX}_projwfc.in", "w") as f:
    f.write(projwfc_input)

print("Generated QE input files:")
print(f"  {PREFIX}_scf.in")
print(f"  {PREFIX}_nscf.in")
print(f"  {PREFIX}_projwfc.in")
print(f"  Number of bands: {n_bands}")
```

#### Step 2: Run QE Calculations

```bash
#!/bin/bash
# Complete IPR calculation workflow

PREFIX="si_vac"
NPROC=$(nproc)

echo "=== Step 1: SCF ==="
mpirun --allow-run-as-root -np $NPROC pw.x -in ${PREFIX}_scf.in > ${PREFIX}_scf.out 2>&1
grep "convergence has been achieved" ${PREFIX}_scf.out && echo "SCF converged" || echo "SCF FAILED"

echo "=== Step 2: NSCF ==="
mpirun --allow-run-as-root -np $NPROC pw.x -in ${PREFIX}_nscf.in > ${PREFIX}_nscf.out 2>&1
echo "NSCF done"

echo "=== Step 3: projwfc.x ==="
mpirun --allow-run-as-root -np $NPROC projwfc.x -in ${PREFIX}_projwfc.in > ${PREFIX}_projwfc.out 2>&1
echo "projwfc.x done"
```

#### Step 3: Compute IPR from projwfc.x Output

```python
#!/usr/bin/env python3
"""
Parse projwfc.x output to compute the Inverse Participation Ratio (IPR)
for each electronic state.

projwfc.x writes a file (PREFIX_proj.projwfc_up or similar) containing:
  - For each k-point and band: the projection |<phi_i | psi_nk>|^2
    onto each atomic orbital phi_i

We compute:
  IPR_n(k) = sum_i w_i^2 / (sum_i w_i)^2

where w_i = |<phi_i | psi_nk>|^2 summed over orbitals on site i.

For site-resolved IPR, we group projections by atom.
"""

import os
import re
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import json

PREFIX = "si_vac"

# ============================================================
# 1. Parse Fermi energy from NSCF output
# ============================================================
def get_fermi_energy(nscf_output):
    """Extract Fermi energy from QE output."""
    with open(nscf_output) as f:
        content = f.read()

    # Semiconductor: highest occupied, lowest unoccupied
    m = re.search(
        r"highest occupied, lowest unoccupied level \(ev\):\s+([-\d.]+)\s+([-\d.]+)",
        content, re.IGNORECASE
    )
    if m:
        return (float(m.group(1)) + float(m.group(2))) / 2.0

    # Metal: Fermi energy
    m = re.search(r"the Fermi energy is\s+([-\d.]+)\s+ev", content, re.IGNORECASE)
    if m:
        return float(m.group(1))

    # Insulator: highest occupied
    m = re.search(r"highest occupied level \(ev\):\s+([-\d.]+)", content, re.IGNORECASE)
    if m:
        return float(m.group(1))

    return 0.0


e_fermi = get_fermi_energy(f"{PREFIX}_nscf.out")
print(f"Fermi energy: {e_fermi:.4f} eV")

# ============================================================
# 2. Parse projwfc.x output to get projections
# ============================================================
def parse_projwfc_output(projwfc_output):
    """
    Parse projwfc.x stdout to extract projection weights per band per k-point.

    Returns
    -------
    data : list of dict
        Each entry: {
            'k_index': int,
            'k_point': (float, float, float),
            'band_index': int,
            'energy_eV': float,
            'projections': dict mapping atom_index -> total_weight
        }
    n_atoms : int
        Number of atoms in the system.
    """
    with open(projwfc_output) as f:
        lines = f.readlines()

    data = []
    n_atoms = 0
    i = 0

    # First pass: determine number of atoms from the state listing
    atom_indices_seen = set()
    while i < len(lines):
        line = lines[i]
        # Match "state #   N: atom  M ..." lines
        m = re.match(r"\s*state\s+#\s+\d+:\s+atom\s+(\d+)", line)
        if m:
            atom_indices_seen.add(int(m.group(1)))
        i += 1
    n_atoms = max(atom_indices_seen) if atom_indices_seen else 0
    print(f"Number of atoms detected: {n_atoms}")

    # Second pass: parse k-points, bands, and projections
    i = 0
    current_k_idx = -1
    current_k_point = None

    while i < len(lines):
        line = lines[i]

        # Detect k-point line
        m = re.match(r"\s*k\s*=\s*([-\d.]+)\s+([-\d.]+)\s+([-\d.]+)", line)
        if m:
            current_k_idx += 1
            current_k_point = (float(m.group(1)), float(m.group(2)), float(m.group(3)))
            i += 1
            continue

        # Detect band line: "==== e(   N) =    E.EEEE eV ===="
        m = re.match(r"\s*====\s+e\(\s*(\d+)\)\s*=\s*([-\d.]+)\s+eV\s*====", line)
        if m:
            band_idx = int(m.group(1))
            energy = float(m.group(2))
            i += 1

            # Read projection lines until next band or k-point
            projections = {}  # atom_index -> weight
            while i < len(lines):
                pline = lines[i]

                # Check for next band or k-point
                if re.match(r"\s*====\s+e\(", pline) or re.match(r"\s*k\s*=", pline):
                    break

                # Match projection: "psi = W*[#N] ..." or "|psi|^2 = X"
                # projwfc.x format: "    0.1234*[#  1]+    0.0567*[#  2]+ ..."
                proj_matches = re.findall(r"([\d.]+)\*\[\#\s*(\d+)\]", pline)
                for weight_str, state_str in proj_matches:
                    weight = float(weight_str)
                    state_idx = int(state_str)
                    # We need to map state index to atom index
                    # This requires the state->atom mapping from above
                    if state_idx not in projections:
                        projections[state_idx] = 0.0
                    projections[state_idx] += weight

                i += 1

            data.append({
                "k_index": current_k_idx,
                "k_point": current_k_point,
                "band_index": band_idx,
                "energy_eV": energy,
                "projections_by_state": projections,
            })
            continue

        i += 1

    return data, n_atoms


def parse_state_atom_mapping(projwfc_output):
    """
    Parse the atomic orbital state listing to map state# -> atom#.

    projwfc.x prints lines like:
      state #   1: atom   1 (Si ), wfc  1 (l=0 m= 1)
      state #   2: atom   1 (Si ), wfc  2 (l=1 m= 1)
      ...
    """
    mapping = {}
    with open(projwfc_output) as f:
        for line in f:
            m = re.match(r"\s*state\s+#\s+(\d+):\s+atom\s+(\d+)", line)
            if m:
                state_idx = int(m.group(1))
                atom_idx = int(m.group(2))
                mapping[state_idx] = atom_idx
    return mapping


# Parse state->atom mapping
state_to_atom = parse_state_atom_mapping(f"{PREFIX}_projwfc.out")
print(f"Number of orbital states: {len(state_to_atom)}")

# Parse projections
proj_data, n_atoms = parse_projwfc_output(f"{PREFIX}_projwfc.out")
print(f"Parsed {len(proj_data)} (k-point, band) entries")

# ============================================================
# 3. Compute IPR for each state
# ============================================================
def compute_ipr(proj_entry, state_to_atom, n_atoms):
    """
    Compute the site-resolved IPR for a single (k, n) state.

    IPR = sum_i w_i^2 / (sum_i w_i)^2

    where w_i is the total projection weight on atom i.

    Parameters
    ----------
    proj_entry : dict
        Contains 'projections_by_state': {state_idx: weight}.
    state_to_atom : dict
        Maps state index -> atom index.
    n_atoms : int
        Total number of atoms.

    Returns
    -------
    ipr : float
        Inverse participation ratio.
    """
    # Accumulate weights by atom
    atom_weights = np.zeros(n_atoms)
    for state_idx, weight in proj_entry["projections_by_state"].items():
        if state_idx in state_to_atom:
            atom_idx = state_to_atom[state_idx] - 1  # 1-indexed to 0-indexed
            if 0 <= atom_idx < n_atoms:
                atom_weights[atom_idx] += weight

    sum_w = np.sum(atom_weights)
    sum_w2 = np.sum(atom_weights ** 2)

    if sum_w < 1e-10:
        return 0.0  # No projection data

    ipr = sum_w2 / (sum_w ** 2)
    return ipr


# Compute IPR for all states
energies = []
iprs = []
k_indices = []
band_indices = []

for entry in proj_data:
    ipr = compute_ipr(entry, state_to_atom, n_atoms)
    energies.append(entry["energy_eV"] - e_fermi)
    iprs.append(ipr)
    k_indices.append(entry["k_index"])
    band_indices.append(entry["band_index"])

energies = np.array(energies)
iprs = np.array(iprs)

print(f"\nIPR statistics:")
print(f"  Min IPR: {np.min(iprs):.6f} (most delocalized)")
print(f"  Max IPR: {np.max(iprs):.6f} (most localized)")
print(f"  Mean IPR: {np.mean(iprs):.6f}")
print(f"  1/N (fully delocalized): {1.0 / n_atoms:.6f}")

# ============================================================
# 4. Identify localized states
# ============================================================
# A state is considered localized if IPR >> 1/N
ipr_threshold = 5.0 / n_atoms  # 5x the delocalized limit
localized_mask = iprs > ipr_threshold

n_localized = np.sum(localized_mask)
print(f"\nLocalized states (IPR > {ipr_threshold:.4f}):")
print(f"  Count: {n_localized} out of {len(iprs)}")

if n_localized > 0:
    loc_energies = energies[localized_mask]
    loc_iprs = iprs[localized_mask]
    print(f"  Energy range: {np.min(loc_energies):.4f} to {np.max(loc_energies):.4f} eV")
    print(f"  IPR range: {np.min(loc_iprs):.6f} to {np.max(loc_iprs):.6f}")

    # Sort by IPR (most localized first)
    sort_idx = np.argsort(loc_iprs)[::-1]
    print(f"\n  Top 10 most localized states:")
    print(f"  {'Energy (eV)':<15} {'IPR':<12} {'k-index':<10} {'band':<10}")
    print(f"  {'-'*47}")
    for idx in sort_idx[:10]:
        full_idx = np.where(localized_mask)[0][idx]
        print(f"  {energies[full_idx]:<15.4f} {iprs[full_idx]:<12.6f} "
              f"{k_indices[full_idx]:<10} {band_indices[full_idx]:<10}")

# ============================================================
# 5. Plot IPR vs Energy
# ============================================================
fig, axes = plt.subplots(2, 1, figsize=(12, 10), gridspec_kw={"height_ratios": [3, 1]})

# -- Top panel: IPR vs Energy scatter --
ax1 = axes[0]
scatter = ax1.scatter(energies, iprs, c=iprs, cmap="hot_r", s=15, alpha=0.7,
                       edgecolors="none", norm=matplotlib.colors.LogNorm(
                           vmin=max(1e-4, np.min(iprs[iprs > 0])),
                           vmax=np.max(iprs)))
cbar = fig.colorbar(scatter, ax=ax1, label="IPR", shrink=0.8)

# Mark delocalized limit
ax1.axhline(1.0 / n_atoms, color="blue", linestyle="--", linewidth=1.5,
            alpha=0.7, label=f"1/N = {1.0 / n_atoms:.4f} (delocalized)")
ax1.axhline(ipr_threshold, color="red", linestyle="--", linewidth=1.5,
            alpha=0.7, label=f"Threshold = {ipr_threshold:.4f}")
ax1.axvline(0, color="green", linestyle="-", linewidth=1, alpha=0.5,
            label="$E_F$")

ax1.set_xlabel("$E - E_F$ (eV)", fontsize=13)
ax1.set_ylabel("IPR", fontsize=13)
ax1.set_title("Inverse Participation Ratio vs Energy", fontsize=14)
ax1.set_yscale("log")
ax1.legend(fontsize=10, loc="upper right")
ax1.grid(alpha=0.3)

# Annotate localized states near the gap
if n_localized > 0:
    gap_mask = localized_mask & (np.abs(energies) < 3.0)
    if np.any(gap_mask):
        gap_energies = energies[gap_mask]
        gap_iprs = iprs[gap_mask]
        ax1.scatter(gap_energies, gap_iprs, c="red", s=60, marker="*",
                    zorder=5, label="Localized near gap")
        ax1.legend(fontsize=10, loc="upper right")

# -- Bottom panel: DOS-like histogram colored by average IPR --
ax2 = axes[1]
e_bins = np.linspace(np.min(energies) - 0.5, np.max(energies) + 0.5, 100)
counts, edges = np.histogram(energies, bins=e_bins)
ipr_avg = np.zeros(len(counts))

for b in range(len(counts)):
    mask = (energies >= edges[b]) & (energies < edges[b + 1])
    if np.any(mask):
        ipr_avg[b] = np.mean(iprs[mask])

bin_centers = (edges[:-1] + edges[1:]) / 2
colors = plt.cm.hot_r(matplotlib.colors.LogNorm(
    vmin=max(1e-4, np.min(ipr_avg[ipr_avg > 0])),
    vmax=np.max(ipr_avg))(ipr_avg))

ax2.bar(bin_centers, counts, width=(edges[1] - edges[0]), color=colors,
        edgecolor="none", alpha=0.8)
ax2.axvline(0, color="green", linestyle="-", linewidth=1, alpha=0.5)
ax2.set_xlabel("$E - E_F$ (eV)", fontsize=13)
ax2.set_ylabel("States", fontsize=13)
ax2.set_title("Density of States (colored by average IPR)", fontsize=12)
ax2.grid(alpha=0.3)

fig.tight_layout()
fig.savefig("ipr_analysis.png", dpi=200, bbox_inches="tight")
print("\nSaved ipr_analysis.png")

# ============================================================
# 6. Plot IPR histogram
# ============================================================
fig2, ax = plt.subplots(figsize=(8, 5))
ipr_nonzero = iprs[iprs > 0]
ax.hist(np.log10(ipr_nonzero), bins=50, color="steelblue", edgecolor="black",
        alpha=0.7)
ax.axvline(np.log10(1.0 / n_atoms), color="blue", linestyle="--", linewidth=2,
           label=f"log10(1/N) = {np.log10(1.0 / n_atoms):.2f}")
ax.axvline(np.log10(ipr_threshold), color="red", linestyle="--", linewidth=2,
           label=f"Threshold = {np.log10(ipr_threshold):.2f}")
ax.set_xlabel("log$_{10}$(IPR)", fontsize=13)
ax.set_ylabel("Count", fontsize=13)
ax.set_title("Distribution of IPR Values", fontsize=14)
ax.legend(fontsize=11)
ax.grid(alpha=0.3)
fig2.tight_layout()
fig2.savefig("ipr_histogram.png", dpi=200, bbox_inches="tight")
print("Saved ipr_histogram.png")

# ============================================================
# 7. Save results
# ============================================================
results = {
    "system": f"{PREFIX}",
    "n_atoms": n_atoms,
    "e_fermi_eV": e_fermi,
    "n_states_analyzed": len(iprs),
    "ipr_min": float(np.min(iprs)),
    "ipr_max": float(np.max(iprs)),
    "ipr_mean": float(np.mean(iprs)),
    "ipr_delocalized_limit": 1.0 / n_atoms,
    "localization_threshold": float(ipr_threshold),
    "n_localized_states": int(n_localized),
}

if n_localized > 0:
    results["localized_energy_range_eV"] = [
        float(np.min(energies[localized_mask])),
        float(np.max(energies[localized_mask])),
    ]

with open("ipr_results.json", "w") as f:
    json.dump(results, f, indent=2)
print("Saved ipr_results.json")

# Save raw data
np.savez(
    "ipr_data.npz",
    energies=energies,
    iprs=iprs,
    k_indices=np.array(k_indices),
    band_indices=np.array(band_indices),
)
print("Saved ipr_data.npz")
```

#### Alternative: Direct IPR from QE XML Output

```python
#!/usr/bin/env python3
"""
Compute IPR directly from QE's data-file-schema.xml without projwfc.x.

This approach reads the KS eigenvalues and occupation numbers from the
QE XML output. It does NOT compute site-resolved IPR (which needs
projections) but can identify states with anomalous occupation as
potentially localized.

For proper site-resolved IPR, use the projwfc.x workflow above.
For a quick check of eigenvalue distribution, this is sufficient.
"""

import os
import xml.etree.ElementTree as ET
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

PREFIX = "si_vac"
OUTDIR = os.path.abspath("./tmp_ipr")
XML_FILE = os.path.join(OUTDIR, f"{PREFIX}.save", "data-file-schema.xml")

# Parse XML
tree = ET.parse(XML_FILE)
root = tree.getroot()

# Namespace
ns = {"qes": "http://www.quantum-espresso.org/ns/qes/qes-1.0"}

# Get Fermi energy
fermi_node = root.find(".//qes:fermi_energy", ns)
if fermi_node is None:
    fermi_node = root.find(".//qes:two_fermi_energies", ns)
if fermi_node is not None:
    e_fermi_ha = float(fermi_node.text.split()[0])
    e_fermi = e_fermi_ha * 27.211386  # Ha -> eV
else:
    e_fermi = 0.0

print(f"Fermi energy: {e_fermi:.4f} eV")

# Get eigenvalues for all k-points
ks_energies = root.findall(".//qes:ks_energies", ns)
all_energies = []
all_occupations = []

for ks in ks_energies:
    eig_node = ks.find("qes:eigenvalues", ns)
    occ_node = ks.find("qes:occupations", ns)
    if eig_node is not None:
        eigs = np.array([float(x) for x in eig_node.text.split()]) * 27.211386
        all_energies.append(eigs)
    if occ_node is not None:
        occs = np.array([float(x) for x in occ_node.text.split()])
        all_occupations.append(occs)

all_energies = np.array(all_energies)  # shape: (n_kpts, n_bands)
all_occupations = np.array(all_occupations)

print(f"K-points: {all_energies.shape[0]}, Bands: {all_energies.shape[1]}")

# Flatten for plotting
energies_flat = (all_energies - e_fermi).flatten()
occs_flat = all_occupations.flatten()

# Plot eigenvalue spectrum
fig, ax = plt.subplots(figsize=(10, 5))
ax.hist(energies_flat, bins=200, color="steelblue", edgecolor="none", alpha=0.7)
ax.axvline(0, color="red", linestyle="--", linewidth=1.5, label="$E_F$")
ax.set_xlabel("$E - E_F$ (eV)", fontsize=13)
ax.set_ylabel("States", fontsize=13)
ax.set_title("Eigenvalue Spectrum (for IPR, use projwfc.x workflow)", fontsize=13)
ax.legend(fontsize=11)
ax.grid(alpha=0.3)
fig.tight_layout()
fig.savefig("eigenvalue_spectrum.png", dpi=200, bbox_inches="tight")
print("Saved eigenvalue_spectrum.png")
```

### Method B: VASP -- IPR from PROCAR

#### Step 1: VASP Calculation with PROCAR Output

```python
#!/usr/bin/env python3
"""
Generate VASP input files for an IPR analysis.

VASP writes PROCAR when LORBIT = 11 or 12, which contains the
projected weights |<Y_lm | psi_nk>|^2 for each atom, orbital, band,
and k-point.

Example: Si supercell with a vacancy.
"""

import os
import numpy as np
from pymatgen.core import Structure, Lattice
from pymatgen.io.vasp import Poscar, Incar, Kpoints
from ase.build import bulk
from pymatgen.io.ase import AseAtomsAdaptor

# ============================================================
# Build structure (same as Method A)
# ============================================================
atoms = bulk("Si", "diamond", a=5.431)
atoms = atoms.repeat((2, 2, 2))
del atoms[0]  # vacancy

adaptor = AseAtomsAdaptor()
struct = adaptor.get_structure(atoms)
n_atoms = len(struct)
print(f"Structure: {struct.formula}, {n_atoms} atoms")

WORKDIR = os.path.abspath("./vasp_ipr")
os.makedirs(WORKDIR, exist_ok=True)

# Estimate number of bands
n_electrons = n_atoms * 4  # Si has 4 valence electrons per atom
n_bands = max(int(n_atoms * 2.5), n_electrons // 2 + 40)

# ============================================================
# INCAR
# ============================================================
incar_dict = {
    "SYSTEM": "Si_vacancy_IPR",
    "ISTART": 0,
    "ICHARG": 2,
    "ENCUT": 400,
    "EDIFF": 1e-6,
    "ISMEAR": 0,
    "SIGMA": 0.05,
    "NBANDS": n_bands,
    "LORBIT": 11,       # Write PROCAR with site+orbital projections
    "LWAVE": False,
    "LCHARG": False,
    "PREC": "Accurate",
    "NPAR": 4,
}
incar = Incar(incar_dict)
incar.write_file(os.path.join(WORKDIR, "INCAR"))

# ============================================================
# KPOINTS
# ============================================================
kpoints = Kpoints.gamma_automatic(kpts=(2, 2, 2), shift=(0, 0, 0))
kpoints.write_file(os.path.join(WORKDIR, "KPOINTS"))

# ============================================================
# POSCAR
# ============================================================
poscar = Poscar(struct)
poscar.write_file(os.path.join(WORKDIR, "POSCAR"))

print(f"VASP inputs written to {WORKDIR}/")
print(f"Number of bands: {n_bands}")
print("Note: POTCAR must be generated separately.")
print("Note: LORBIT = 11 produces PROCAR with site-projected weights.")
```

#### Step 2: Run VASP

```bash
#!/bin/bash
# Run VASP for IPR analysis
cd ./vasp_ipr
mpirun -np $(nproc) vasp_std > vasp.log 2>&1
echo "VASP completed. Check PROCAR for projection data."
```

#### Step 3: Parse PROCAR and Compute IPR

```python
#!/usr/bin/env python3
"""
Parse VASP PROCAR to compute the Inverse Participation Ratio (IPR)
for each electronic state.

PROCAR format (LORBIT = 11):
  For each k-point and band:
    ion     s      py     pz     px    dxy    dyz    dz2    dxz  x2-y2   tot
      1  0.000  0.000  0.000  0.000  0.000  0.000  0.000  0.000  0.000  0.XXX
      2  ...
    tot  ...

We compute:
  w_i = tot_i  (total projection on atom i)
  IPR = sum_i w_i^2 / (sum_i w_i)^2
"""

import os
import re
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import json

WORKDIR = os.path.abspath("./vasp_ipr")
PROCAR = os.path.join(WORKDIR, "PROCAR")

# ============================================================
# 1. Parse Fermi energy from OUTCAR
# ============================================================
def get_fermi_from_outcar(outcar_file):
    """Extract Fermi energy from VASP OUTCAR."""
    with open(outcar_file) as f:
        for line in f:
            if "E-fermi" in line:
                m = re.search(r"E-fermi\s*:\s*([-\d.]+)", line)
                if m:
                    return float(m.group(1))
    return 0.0


e_fermi = get_fermi_from_outcar(os.path.join(WORKDIR, "OUTCAR"))
print(f"Fermi energy: {e_fermi:.4f} eV")

# ============================================================
# 2. Parse PROCAR
# ============================================================
def parse_procar(procar_file):
    """
    Parse VASP PROCAR file to extract projection weights per atom
    for each (k-point, band).

    Returns
    -------
    results : list of dict
        Each entry: {
            'k_index': int,
            'band_index': int,
            'energy_eV': float,
            'occupation': float,
            'atom_weights': np.array of shape (n_atoms,)
        }
    n_atoms : int
    n_kpoints : int
    n_bands : int
    """
    with open(procar_file) as f:
        lines = f.readlines()

    # First line: header
    # Second line: "# of k-points:   N         # of bands:   M         # of ions:   P"
    m = re.search(
        r"# of k-points:\s*(\d+)\s+# of bands:\s*(\d+)\s+# of ions:\s*(\d+)",
        lines[1]
    )
    n_kpoints = int(m.group(1))
    n_bands = int(m.group(2))
    n_atoms = int(m.group(3))

    print(f"PROCAR: {n_kpoints} k-points, {n_bands} bands, {n_atoms} atoms")

    results = []
    i = 2  # Start after header

    k_idx = -1
    while i < len(lines):
        line = lines[i].strip()

        # k-point line: " k-point    1 :    0.00000000 0.00000000 0.00000000     weight = 0.03703704"
        m = re.match(r"k-point\s+(\d+)", line)
        if m:
            k_idx = int(m.group(1)) - 1
            i += 2  # Skip blank line
            continue

        # Band line: "band    1 # energy   -5.12345678 # occ.  1.00000000"
        m = re.match(r"band\s+(\d+)\s+#\s+energy\s+([-\d.]+)\s+#\s+occ\.\s+([-\d.]+)", line)
        if m:
            band_idx = int(m.group(1))
            energy = float(m.group(2))
            occupation = float(m.group(3))
            i += 2  # Skip blank line and header line

            # Read atom projection lines
            atom_weights = np.zeros(n_atoms)
            for a in range(n_atoms):
                if i < len(lines):
                    parts = lines[i].strip().split()
                    if len(parts) >= 2:
                        # Last column is "tot" (total projection on this atom)
                        atom_weights[a] = float(parts[-1])
                    i += 1

            # Skip "tot" line
            if i < len(lines) and "tot" in lines[i]:
                i += 1

            results.append({
                "k_index": k_idx,
                "band_index": band_idx,
                "energy_eV": energy,
                "occupation": occupation,
                "atom_weights": atom_weights,
            })
            continue

        i += 1

    return results, n_atoms, n_kpoints, n_bands


procar_data, n_atoms, n_kpoints, n_bands = parse_procar(PROCAR)
print(f"Parsed {len(procar_data)} (k-point, band) entries")

# ============================================================
# 3. Compute IPR for each state
# ============================================================
energies = []
iprs = []
k_indices = []
band_indices = []

for entry in procar_data:
    w = entry["atom_weights"]
    sum_w = np.sum(w)
    sum_w2 = np.sum(w ** 2)

    if sum_w > 1e-10:
        ipr = sum_w2 / (sum_w ** 2)
    else:
        ipr = 0.0

    energies.append(entry["energy_eV"] - e_fermi)
    iprs.append(ipr)
    k_indices.append(entry["k_index"])
    band_indices.append(entry["band_index"])

energies = np.array(energies)
iprs = np.array(iprs)

print(f"\nIPR statistics:")
print(f"  Min IPR: {np.min(iprs):.6f}")
print(f"  Max IPR: {np.max(iprs):.6f}")
print(f"  Mean IPR: {np.mean(iprs):.6f}")
print(f"  1/N (delocalized limit): {1.0 / n_atoms:.6f}")

# ============================================================
# 4. Identify localized states
# ============================================================
ipr_threshold = 5.0 / n_atoms
localized_mask = iprs > ipr_threshold
n_localized = np.sum(localized_mask)

print(f"\nLocalized states (IPR > {ipr_threshold:.4f}): {n_localized}")
if n_localized > 0:
    loc_e = energies[localized_mask]
    loc_ipr = iprs[localized_mask]
    sort_idx = np.argsort(loc_ipr)[::-1]
    print(f"\n  Top 10 most localized:")
    print(f"  {'Energy (eV)':<15} {'IPR':<12}")
    print(f"  {'-'*27}")
    for idx in sort_idx[:10]:
        full_idx = np.where(localized_mask)[0][idx]
        print(f"  {energies[full_idx]:<15.4f} {iprs[full_idx]:<12.6f}")

# ============================================================
# 5. Plot IPR vs Energy
# ============================================================
fig, axes = plt.subplots(2, 1, figsize=(12, 10),
                          gridspec_kw={"height_ratios": [3, 1]})

ax1 = axes[0]
scatter = ax1.scatter(energies, iprs, c=iprs, cmap="hot_r", s=15, alpha=0.7,
                       edgecolors="none",
                       norm=matplotlib.colors.LogNorm(
                           vmin=max(1e-4, np.min(iprs[iprs > 0])),
                           vmax=np.max(iprs)))
fig.colorbar(scatter, ax=ax1, label="IPR", shrink=0.8)

ax1.axhline(1.0 / n_atoms, color="blue", linestyle="--", linewidth=1.5,
            label=f"1/N = {1.0 / n_atoms:.4f}")
ax1.axhline(ipr_threshold, color="red", linestyle="--", linewidth=1.5,
            label=f"Threshold = {ipr_threshold:.4f}")
ax1.axvline(0, color="green", linestyle="-", linewidth=1, alpha=0.5,
            label="$E_F$")

if n_localized > 0:
    gap_mask = localized_mask & (np.abs(energies) < 3.0)
    if np.any(gap_mask):
        ax1.scatter(energies[gap_mask], iprs[gap_mask], c="red", s=60,
                    marker="*", zorder=5, label="Localized near gap")

ax1.set_xlabel("$E - E_F$ (eV)", fontsize=13)
ax1.set_ylabel("IPR", fontsize=13)
ax1.set_title("IPR Analysis (VASP PROCAR)", fontsize=14)
ax1.set_yscale("log")
ax1.legend(fontsize=10)
ax1.grid(alpha=0.3)

# Bottom panel: DOS histogram
ax2 = axes[1]
ax2.hist(energies, bins=100, color="steelblue", edgecolor="none", alpha=0.7)
ax2.axvline(0, color="green", linestyle="-", linewidth=1, alpha=0.5)
ax2.set_xlabel("$E - E_F$ (eV)", fontsize=13)
ax2.set_ylabel("States", fontsize=13)
ax2.set_title("Density of States", fontsize=12)
ax2.grid(alpha=0.3)

fig.tight_layout()
fig.savefig("ipr_vasp.png", dpi=200, bbox_inches="tight")
print("\nSaved ipr_vasp.png")

# ============================================================
# 6. Save results
# ============================================================
results = {
    "method": "VASP_PROCAR",
    "n_atoms": n_atoms,
    "n_kpoints": n_kpoints,
    "n_bands": n_bands,
    "e_fermi_eV": e_fermi,
    "ipr_min": float(np.min(iprs)),
    "ipr_max": float(np.max(iprs)),
    "ipr_mean": float(np.mean(iprs)),
    "delocalized_limit": 1.0 / n_atoms,
    "threshold": float(ipr_threshold),
    "n_localized": int(n_localized),
}

with open("ipr_vasp_results.json", "w") as f:
    json.dump(results, f, indent=2)
print("Saved ipr_vasp_results.json")

np.savez(
    "ipr_vasp_data.npz",
    energies=energies,
    iprs=iprs,
    k_indices=np.array(k_indices),
    band_indices=np.array(band_indices),
)
print("Saved ipr_vasp_data.npz")
```

#### VASPKIT 110 Automated Workflow

```
VASPKIT 110 provides automated IPR analysis from VASP output:

1. Run a VASP calculation with LORBIT = 11 (or 12) to produce PROCAR.

2. Run VASPKIT:
     vaspkit -task 110

3. VASPKIT reads PROCAR and EIGENVAL to:
   - Compute IPR for each band at each k-point
   - Identify localized states above a threshold
   - Generate IPR vs energy plots

4. Output files:
   - IPR.dat -- IPR values for all (k, band) states
   - IPR.png -- scatter plot of IPR vs energy

Note: VASPKIT 110 uses the same definition:
  IPR = sum_i |c_i|^4 / (sum_i |c_i|^2)^2
where c_i are the site-projected weights from PROCAR.
```

## Key Parameters

| Parameter | Recommended Value | Notes |
|---|---|---|
| `nbnd` / `NBANDS` | 2-3x occupied bands | Must include many empty bands to probe localization above the gap |
| K-points | Gamma-only or 2x2x2 for supercells | Supercell folds the BZ; fewer k-points needed |
| Supercell size | >= 2x2x2 for point defects | Larger supercells give more meaningful IPR (1/N is smaller) |
| `LORBIT` (VASP) | 11 | Writes PROCAR with site-projected orbital weights |
| projwfc.x (QE) | filproj set | Writes projection data needed for IPR |
| `ecutwfc` / `ENCUT` | Standard for the system | Same as any SCF; no special requirements for IPR |
| IPR threshold | 5/N to 10/N | States with IPR above this are considered localized |
| `occupations` (QE) | `smearing` | Use smearing for metallic/small-gap systems |
| `ISMEAR` (VASP) | 0 (Gaussian) | Appropriate for semiconductors with defects |

## Interpreting Results

1. **IPR = 1/N**: The state is perfectly delocalized over all N atoms. This is the lower bound for a system of N sites. Bulk Bloch states typically have IPR close to 1/N.
2. **IPR = 1**: The state is completely localized on a single site. In practice, values above 0.1 indicate strong localization.
3. **IPR near the band gap**: Localized states with high IPR near E_F often correspond to defect levels (vacancy states, impurity levels, dangling bonds).
4. **IPR in the valence/conduction band**: If bulk band states show elevated IPR, it may indicate resonant defect states hybridized with the band, or an insufficiently large supercell.
5. **Anderson localization**: In disordered systems (e.g., random alloys), a mobility edge may separate localized tail states from extended band states. The IPR vs energy plot reveals this transition.
6. **Band-edge states in alloys**: In semiconductor alloys (e.g., InGaN), composition fluctuations can localize band-edge states, visible as elevated IPR near CBM/VBM.
7. **Projection completeness**: The sum of all projections for a given state may be less than 1.0 due to incomplete basis or interstitial charge. If sum(w_i) << 1, IPR values may be unreliable.
8. **K-point dependence**: For supercells, most physics is captured at Gamma. For primitive cells, IPR varies with k; plot IPR as a function of both k and energy for a full picture.

## Common Issues

| Issue | Solution |
|---|---|
| All IPR values are near 1/N | No localized states present, or defect level is resonant with the band continuum |
| IPR is 0 for some states | Zero projection weights -- check that projwfc.x / PROCAR completed successfully |
| projwfc.x crashes or hangs | Reduce Emin/Emax range; ensure NSCF completed and tmp dir is intact |
| PROCAR is empty or truncated | Ensure LORBIT = 11 in INCAR; check that VASP completed normally |
| Sum of projections << 1 | Pseudopotential does not project all electrons; use PAW with LORBIT = 12 for better coverage |
| Supercell too small -- 1/N too large | Use at least 3x3x3 or larger supercells so 1/N << 1 and localization is distinguishable |
| Too few bands -- localized states missed | Increase `nbnd`/`NBANDS` to include states well above CBM |
| IPR plot looks noisy | Increase k-mesh or use k-point averaging; check for unconverged SCF |
| Spin-polarized system | Parse spin-up and spin-down projections separately from PROCAR |
| projwfc.x output format changed | Check QE version; parsing regex may need updating for different versions |
