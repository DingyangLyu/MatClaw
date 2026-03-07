# Work Function Calculation

## When to Use

- Calculate the work function of a surface (energy required to remove an electron to vacuum).
- Compare work functions of different surface facets or terminations.
- Assess the effect of adsorbates, defects, or surface reconstructions on the work function.
- Determine the vacuum level and Fermi energy from a slab calculation.
- Corresponds to VASPKIT tasks 420--421.

## Method Selection

| Criterion | QE (pp.x + average.x) | VASP (LOCPOT parsing) | pymatgen/ASE |
|---|---|---|---|
| Availability | Full workflow in container | Future: LOCPOT file required | Post-processing only |
| Accuracy | DFT-level | DFT-level | Reads DFT output |
| SOC support | Yes (noncollinear) | Yes | N/A |
| Automation | Python-driven | Python-driven | Python-driven |

## Prerequisites

- Quantum ESPRESSO 7.5 (`pw.x`, `pp.x`, `average.x`)
- Python packages: `numpy`, `matplotlib`, `ase`, `pymatgen`
- A slab model with sufficient vacuum (15--20 Angstrom) and converged surface
- Pseudopotential files for all elements
- For VASP: LOCPOT file from a slab calculation

---

## Detailed Steps

### Method A: QE -- Full Work Function Workflow

The work function is defined as:

    phi = E_vac - E_F

where E_vac is the electrostatic potential in the vacuum region and E_F is the Fermi energy.

#### Step A1: Build a Slab Model

```python
#!/usr/bin/env python3
"""
Build a surface slab for work function calculation using pymatgen.
Example: Al(111) slab with vacuum.
"""
import numpy as np
from pymatgen.core import Structure, Lattice
from pymatgen.core.surface import SlabGenerator
from pymatgen.io.ase import AseAtomsAdaptor
from ase.io import write

# Build bulk Al (FCC)
a_al = 4.05  # Angstrom
bulk_al = Structure(
    Lattice.cubic(a_al),
    ["Al"] * 4,
    [[0, 0, 0], [0.5, 0.5, 0], [0.5, 0, 0.5], [0, 0.5, 0.5]],
)

# Generate (111) slab: 5 layers, 20 Angstrom vacuum
slabgen = SlabGenerator(
    bulk_al,
    miller_index=(1, 1, 1),
    min_slab_size=10.0,   # Angstrom (slab thickness)
    min_vacuum_size=20.0, # Angstrom (vacuum thickness)
    center_slab=True,
    in_unit_planes=False,
)

slabs = slabgen.get_slabs(symmetrize=False)
slab = slabs[0]  # Take the first slab

print(f"Slab formula: {slab.composition.reduced_formula}")
print(f"Number of atoms: {len(slab)}")
print(f"Cell height (c): {slab.lattice.c:.2f} Angstrom")

# Convert to ASE and save
atoms = AseAtomsAdaptor.get_atoms(slab)
write("al111_slab.cif", atoms)
write("al111_slab.xsf", atoms)
print("Saved: al111_slab.cif, al111_slab.xsf")

# Print fractional z-coordinates to verify vacuum region
z_frac = slab.frac_coords[:, 2]
print(f"Fractional z range of atoms: {z_frac.min():.4f} -- {z_frac.max():.4f}")
print(f"Vacuum region: {z_frac.max():.4f} -- {1.0 + z_frac.min():.4f} (fractional)")
```

#### Step A2: SCF Calculation for the Slab

```python
#!/usr/bin/env python3
"""
Run SCF calculation for the slab to obtain charge density and Fermi energy.
"""
import os
import subprocess
import re
import numpy as np
from pymatgen.core import Structure, Element

PSEUDO_DIR = os.path.abspath("./pseudo")
OUTDIR = os.path.abspath("./tmp_wf")
os.makedirs(OUTDIR, exist_ok=True)
os.makedirs(PSEUDO_DIR, exist_ok=True)

# Load slab structure
slab = Structure.from_file("al111_slab.cif")
cell = slab.lattice.matrix
species_set = sorted(set(str(s) for s in slab.species))

# Pseudopotential map (adjust filenames to match your PP library)
pseudo_map = {
    "Al": "Al.pbe-n-kjpaw_psl.1.0.0.UPF",
}

# Build input
nat = len(slab)
ntyp = len(species_set)

species_card = ""
for sp in species_set:
    mass = Element(sp).atomic_mass
    species_card += f"  {sp}  {mass:.4f}  {pseudo_map[sp]}\n"

pos_card = ""
for site in slab:
    sp = str(site.specie)
    fc = site.frac_coords
    pos_card += f"  {sp}  {fc[0]:.10f}  {fc[1]:.10f}  {fc[2]:.10f}\n"

cell_card = ""
for row in cell:
    cell_card += f"  {row[0]:.10f}  {row[1]:.10f}  {row[2]:.10f}\n"

scf_input = f"""&CONTROL
    calculation = 'scf'
    prefix      = 'slab'
    outdir      = '{OUTDIR}'
    pseudo_dir  = '{PSEUDO_DIR}'
    tprnfor     = .true.
    tstress     = .true.
/
&SYSTEM
    ibrav       = 0
    nat         = {nat}
    ntyp        = {ntyp}
    ecutwfc     = 40.0
    ecutrho     = 320.0
    occupations = 'smearing'
    smearing    = 'mv'
    degauss     = 0.02
/
&ELECTRONS
    conv_thr = 1.0d-8
    mixing_beta = 0.5
/

CELL_PARAMETERS angstrom
{cell_card}
ATOMIC_SPECIES
{species_card}
ATOMIC_POSITIONS crystal
{pos_card}
K_POINTS automatic
  12 12 1  0 0 0
"""

with open("slab_scf.in", "w") as f:
    f.write(scf_input)

print("Running SCF for slab...")
result = subprocess.run(
    ["mpirun", "-np", "4", "pw.x", "-in", "slab_scf.in"],
    capture_output=True, text=True, timeout=1200
)
with open("slab_scf.out", "w") as f:
    f.write(result.stdout)

# Parse Fermi energy
e_fermi = None
for line in result.stdout.split("\n"):
    if "the Fermi energy is" in line:
        m = re.search(r"is\s+([-\d.]+)\s+ev", line, re.IGNORECASE)
        if m:
            e_fermi = float(m.group(1))
    if "highest occupied, lowest unoccupied" in line:
        m = re.search(r":\s+([-\d.]+)\s+([-\d.]+)", line)
        if m:
            e_fermi = (float(m.group(1)) + float(m.group(2))) / 2

if "convergence has been achieved" in result.stdout:
    print(f"SCF converged. Fermi energy: {e_fermi} eV")
else:
    print("WARNING: SCF may not have converged. Check slab_scf.out")
```

#### Step A3: Extract Electrostatic Potential with pp.x

```python
#!/usr/bin/env python3
"""
Extract the electrostatic potential using pp.x.
plot_num=11 gives the V_bare + V_Hartree (electrostatic potential without xc).
For work function, plot_num=0 (total potential) or plot_num=11 are commonly used.
We use plot_num=11 for the bare ionic + Hartree potential.
"""
import os
import subprocess

OUTDIR = os.path.abspath("./tmp_wf")

# pp.x input: extract electrostatic potential to a 3D file
pp_input = f"""&INPUTPP
    prefix  = 'slab'
    outdir  = '{OUTDIR}'
    filplot = 'potential.dat'
    plot_num = 11
/
&PLOT
    nfile       = 1
    filepp(1)   = 'potential.dat'
    weight(1)   = 1.0
    iflag       = 3
    output_format = 6
    fileout     = 'potential.cube'
/
"""

with open("pp_potential.in", "w") as f:
    f.write(pp_input)

print("Running pp.x to extract electrostatic potential...")
result = subprocess.run(
    ["pp.x", "-in", "pp_potential.in"],
    capture_output=True, text=True, timeout=300
)
with open("pp_potential.out", "w") as f:
    f.write(result.stdout)

if result.returncode == 0:
    print("pp.x completed. Output: potential.cube")
else:
    print("ERROR in pp.x!")
    print(result.stderr[-500:] if result.stderr else "")
```

#### Step A4: Planar Average with average.x

```python
#!/usr/bin/env python3
"""
Use QE average.x to compute the planar average of the electrostatic potential
along the z-direction (slab normal).

average.x reads the intermediate file from pp.x (plot_num output before the
PLOT section) and produces a 1D planar-averaged profile.
"""
import os
import subprocess

# First, re-run pp.x to get the raw data file (without PLOT section)
OUTDIR = os.path.abspath("./tmp_wf")

pp_raw_input = f"""&INPUTPP
    prefix  = 'slab'
    outdir  = '{OUTDIR}'
    filplot = 'potential_raw.dat'
    plot_num = 11
/
"""

with open("pp_potential_raw.in", "w") as f:
    f.write(pp_raw_input)

print("Running pp.x to extract raw potential data...")
result = subprocess.run(
    ["pp.x", "-in", "pp_potential_raw.in"],
    capture_output=True, text=True, timeout=300
)
with open("pp_potential_raw.out", "w") as f:
    f.write(result.stdout)

# average.x input
# npt: number of points in the cell along the averaging direction
# For slab calculations, the averaging direction is 3 (z-axis)
# idir: direction of averaging (1=x, 2=y, 3=z)
# awin: window for macroscopic averaging (in units of the cell parameter along idir)
#        Set to 0 for just planar averaging (no macroscopic averaging)

avg_input = """1
potential_raw.dat
1.0
200
3
0.0
"""
# Format: nfile, filename, weight, npt, idir, awin

with open("average.in", "w") as f:
    f.write(avg_input)

print("Running average.x for planar averaging...")
result = subprocess.run(
    ["average.x"],
    input=avg_input,
    capture_output=True, text=True, timeout=120
)
with open("average.out", "w") as f:
    f.write(result.stdout)

if result.returncode == 0:
    print("average.x completed. Output: avg.dat")
else:
    print("ERROR in average.x!")
    print(result.stderr[-500:] if result.stderr else "")
```

#### Step A5: Plot and Extract Work Function

```python
#!/usr/bin/env python3
"""
Parse the planar-averaged potential from average.x output,
identify vacuum level and Fermi energy, and compute the work function.
"""
import numpy as np
import re
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ── Parse Fermi energy from SCF output ────────────────────────────
def get_fermi_energy(scf_output):
    e_fermi = 0.0
    with open(scf_output, "r") as f:
        for line in f:
            if "the Fermi energy is" in line:
                m = re.search(r"is\s+([-\d.]+)\s+ev", line, re.IGNORECASE)
                if m:
                    e_fermi = float(m.group(1))
            if "highest occupied, lowest unoccupied" in line:
                m = re.search(r":\s+([-\d.]+)\s+([-\d.]+)", line)
                if m:
                    e_fermi = (float(m.group(1)) + float(m.group(2))) / 2
    return e_fermi

# ── Parse planar-averaged potential from average.x output ─────────
def parse_average_output(avg_output_file):
    """
    Parse average.x output. The output contains lines with:
      z(bohr)   V_planar(Ry)   V_macro(Ry)
    The planar average and optionally macroscopic average.
    """
    z_bohr = []
    v_planar_ry = []
    v_macro_ry = []

    with open(avg_output_file, "r") as f:
        in_data = False
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            # Data lines have 2 or 3 numeric columns
            try:
                if len(parts) >= 2:
                    z_val = float(parts[0])
                    v_val = float(parts[1])
                    z_bohr.append(z_val)
                    v_planar_ry.append(v_val)
                    if len(parts) >= 3:
                        v_macro_ry.append(float(parts[2]))
            except ValueError:
                continue

    BOHR_TO_ANG = 0.529177
    RY_TO_EV = 13.6057

    z_ang = np.array(z_bohr) * BOHR_TO_ANG
    v_planar_ev = np.array(v_planar_ry) * RY_TO_EV
    v_macro_ev = np.array(v_macro_ry) * RY_TO_EV if v_macro_ry else None

    return z_ang, v_planar_ev, v_macro_ev

# ── Alternative: parse from cube file directly ────────────────────
def planar_average_from_cube(cube_file, axis=2):
    """
    Compute planar average of a cube file along the specified axis.
    Returns z (Angstrom) and V_avg (values in cube file units).
    """
    from ase.io.cube import read_cube_data

    data, atoms = read_cube_data(cube_file)
    cell = atoms.get_cell()
    length = np.linalg.norm(cell[axis])

    # Average over the two in-plane axes
    avg_axes = tuple(i for i in range(3) if i != axis)
    planar_avg = np.mean(data, axis=avg_axes)

    n_pts = planar_avg.shape[0]
    z = np.linspace(0, length, n_pts, endpoint=False)

    # QE cube files store potential in Ry; convert to eV
    RY_TO_EV = 13.6057
    planar_avg_ev = planar_avg * RY_TO_EV

    return z, planar_avg_ev

# ── Compute work function ─────────────────────────────────────────
e_fermi = get_fermi_energy("slab_scf.out")
print(f"Fermi energy: {e_fermi:.4f} eV")

# Try average.x output first, fall back to cube file
try:
    z, v_planar, v_macro = parse_average_output("average.out")
    print(f"Parsed average.x output: {len(z)} points")
except Exception:
    print("Falling back to cube file planar average...")
    z, v_planar = planar_average_from_cube("potential.cube", axis=2)
    v_macro = None

# Identify vacuum region: find the plateau in the potential
# The vacuum level is the average potential in the flat vacuum region
# For a centered slab, vacuum is at the edges of the z-range
n = len(z)
# Take 10% of points from each edge as vacuum region
n_vac = max(n // 10, 5)
v_left = np.mean(v_planar[:n_vac])
v_right = np.mean(v_planar[-n_vac:])
v_vacuum = (v_left + v_right) / 2.0

print(f"Vacuum level (left):  {v_left:.4f} eV")
print(f"Vacuum level (right): {v_right:.4f} eV")
print(f"Vacuum level (avg):   {v_vacuum:.4f} eV")

work_function = v_vacuum - e_fermi
print(f"\n=== Work Function ===")
print(f"phi = E_vac - E_F = {v_vacuum:.4f} - {e_fermi:.4f} = {work_function:.4f} eV")

# Check for asymmetric slabs
asymmetry = abs(v_left - v_right)
if asymmetry > 0.1:
    print(f"\nWARNING: Potential is asymmetric (|V_left - V_right| = {asymmetry:.4f} eV)")
    print("This may indicate a polar surface or insufficient vacuum.")
    print(f"Work function (left face):  {v_left - e_fermi:.4f} eV")
    print(f"Work function (right face): {v_right - e_fermi:.4f} eV")

# ── Plot ──────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(10, 5))

ax.plot(z, v_planar, "b-", linewidth=1.2, label="Planar average")
if v_macro is not None and len(v_macro) == len(z):
    ax.plot(z, v_macro, "r-", linewidth=1.5, label="Macroscopic average")

ax.axhline(e_fermi, color="green", linestyle="--", linewidth=1.0,
           label=f"$E_F$ = {e_fermi:.2f} eV")
ax.axhline(v_vacuum, color="orange", linestyle="--", linewidth=1.0,
           label=f"$E_{{vac}}$ = {v_vacuum:.2f} eV")

# Annotate work function
z_mid = z[len(z)//10]
ax.annotate("", xy=(z_mid, e_fermi), xytext=(z_mid, v_vacuum),
            arrowprops=dict(arrowstyle="<->", color="black", lw=1.5))
ax.text(z_mid + 0.5, (e_fermi + v_vacuum) / 2,
        f"$\\phi$ = {work_function:.2f} eV",
        fontsize=12, va="center")

ax.set_xlabel(r"z ($\mathrm{\AA}$)", fontsize=13)
ax.set_ylabel("Electrostatic potential (eV)", fontsize=13)
ax.set_title("Work Function from Planar-Averaged Potential", fontsize=14)
ax.legend(fontsize=11)
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("work_function.png", dpi=200, bbox_inches="tight")
print("\nSaved: work_function.png")
```

#### Complete Single-Script Workflow (QE)

```python
#!/usr/bin/env python3
"""
Complete work function workflow: build slab -> SCF -> pp.x -> average -> plot.
Example: Al(111) surface.
"""
import os
import subprocess
import re
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pymatgen.core import Structure, Lattice, Element
from pymatgen.core.surface import SlabGenerator

# ── Configuration ──────────────────────────────────────────────────
PSEUDO_DIR = os.path.abspath("./pseudo")
OUTDIR = os.path.abspath("./tmp_wf_full")
os.makedirs(OUTDIR, exist_ok=True)
os.makedirs(PSEUDO_DIR, exist_ok=True)

PREFIX = "al111"
ECUTWFC = 40.0
ECUTRHO = 320.0
NPROC = 4

pseudo_map = {"Al": "Al.pbe-n-kjpaw_psl.1.0.0.UPF"}

# ── Step 1: Build slab ────────────────────────────────────────────
a_al = 4.05
bulk_al = Structure(
    Lattice.cubic(a_al),
    ["Al"] * 4,
    [[0, 0, 0], [0.5, 0.5, 0], [0.5, 0, 0.5], [0, 0.5, 0.5]],
)

slabgen = SlabGenerator(
    bulk_al, miller_index=(1, 1, 1),
    min_slab_size=10.0, min_vacuum_size=20.0,
    center_slab=True, in_unit_planes=False,
)
slab = slabgen.get_slabs(symmetrize=False)[0]
cell = slab.lattice.matrix
species_set = sorted(set(str(s) for s in slab.species))

print(f"Slab: {len(slab)} atoms, c = {slab.lattice.c:.2f} Ang")

# ── Step 2: SCF ───────────────────────────────────────────────────
species_card = ""
for sp in species_set:
    mass = Element(sp).atomic_mass
    species_card += f"  {sp}  {mass:.4f}  {pseudo_map[sp]}\n"

pos_card = ""
for site in slab:
    sp = str(site.specie)
    fc = site.frac_coords
    pos_card += f"  {sp}  {fc[0]:.10f}  {fc[1]:.10f}  {fc[2]:.10f}\n"

cell_card = ""
for row in cell:
    cell_card += f"  {row[0]:.10f}  {row[1]:.10f}  {row[2]:.10f}\n"

scf_input = f"""&CONTROL
    calculation = 'scf'
    prefix      = '{PREFIX}'
    outdir      = '{OUTDIR}'
    pseudo_dir  = '{PSEUDO_DIR}'
/
&SYSTEM
    ibrav       = 0
    nat         = {len(slab)}
    ntyp        = {len(species_set)}
    ecutwfc     = {ECUTWFC}
    ecutrho     = {ECUTRHO}
    occupations = 'smearing'
    smearing    = 'mv'
    degauss     = 0.02
/
&ELECTRONS
    conv_thr = 1.0d-8
/

CELL_PARAMETERS angstrom
{cell_card}
ATOMIC_SPECIES
{species_card}
ATOMIC_POSITIONS crystal
{pos_card}
K_POINTS automatic
  12 12 1  0 0 0
"""

with open(f"{PREFIX}_scf.in", "w") as f:
    f.write(scf_input)

print("[1/4] Running SCF...")
r = subprocess.run(["mpirun", "-np", str(NPROC), "pw.x", "-in", f"{PREFIX}_scf.in"],
                    capture_output=True, text=True, timeout=1200)
with open(f"{PREFIX}_scf.out", "w") as f:
    f.write(r.stdout)
assert "convergence has been achieved" in r.stdout, "SCF did not converge!"

e_fermi = 0.0
for line in r.stdout.split("\n"):
    if "the Fermi energy is" in line:
        m = re.search(r"is\s+([-\d.]+)", line)
        if m: e_fermi = float(m.group(1))
print(f"      E_F = {e_fermi:.4f} eV")

# ── Step 3: pp.x ──────────────────────────────────────────────────
pp_input = f"""&INPUTPP
    prefix  = '{PREFIX}'
    outdir  = '{OUTDIR}'
    filplot = '{PREFIX}_potential.dat'
    plot_num = 11
/
&PLOT
    nfile       = 1
    filepp(1)   = '{PREFIX}_potential.dat'
    weight(1)   = 1.0
    iflag       = 3
    output_format = 6
    fileout     = '{PREFIX}_potential.cube'
/
"""

with open(f"{PREFIX}_pp.in", "w") as f:
    f.write(pp_input)

print("[2/4] Running pp.x...")
r = subprocess.run(["pp.x", "-in", f"{PREFIX}_pp.in"],
                    capture_output=True, text=True, timeout=300)
with open(f"{PREFIX}_pp.out", "w") as f:
    f.write(r.stdout)

# ── Step 4: Planar average from cube file ─────────────────────────
print("[3/4] Computing planar average...")
from ase.io.cube import read_cube_data

data, atoms = read_cube_data(f"{PREFIX}_potential.cube")
cell_ase = atoms.get_cell()
c_length = np.linalg.norm(cell_ase[2])

# Planar average along z (axis=2)
planar_avg = np.mean(data, axis=(0, 1))
n_pts = len(planar_avg)
z = np.linspace(0, c_length, n_pts, endpoint=False)

RY_TO_EV = 13.6057
v_ev = planar_avg * RY_TO_EV

# ── Step 5: Compute work function and plot ─────────────────────────
print("[4/4] Computing work function and plotting...")

n_vac = max(n_pts // 10, 5)
v_left = np.mean(v_ev[:n_vac])
v_right = np.mean(v_ev[-n_vac:])
v_vacuum = (v_left + v_right) / 2.0
work_function = v_vacuum - e_fermi

print(f"\n=== Results ===")
print(f"Fermi energy:  {e_fermi:.4f} eV")
print(f"Vacuum level:  {v_vacuum:.4f} eV")
print(f"Work function: {work_function:.4f} eV")

fig, ax = plt.subplots(figsize=(10, 5))
ax.plot(z, v_ev, "b-", linewidth=1.2, label="Planar average $V(z)$")
ax.axhline(e_fermi, color="green", linestyle="--", linewidth=1.0,
           label=f"$E_F$ = {e_fermi:.2f} eV")
ax.axhline(v_vacuum, color="orange", linestyle="--", linewidth=1.0,
           label=f"$E_{{vac}}$ = {v_vacuum:.2f} eV")

z_arrow = z[n_pts // 10]
ax.annotate("", xy=(z_arrow, e_fermi), xytext=(z_arrow, v_vacuum),
            arrowprops=dict(arrowstyle="<->", color="black", lw=1.5))
ax.text(z_arrow + 0.5, (e_fermi + v_vacuum) / 2,
        f"$\\phi$ = {work_function:.2f} eV", fontsize=12, va="center")

ax.set_xlabel(r"z ($\mathrm{\AA}$)", fontsize=13)
ax.set_ylabel("Electrostatic potential (eV)", fontsize=13)
ax.set_title(f"Work Function: Al(111)", fontsize=14)
ax.legend(fontsize=11)
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("work_function.png", dpi=200, bbox_inches="tight")
print("Saved: work_function.png")
```

### Method B: VASP -- LOCPOT Parsing

```python
#!/usr/bin/env python3
"""
Calculate work function from VASP LOCPOT file.
LOCPOT contains the local potential on the FFT grid.
Planar-average along z gives V(z), from which E_vac is extracted.

Requires: LOCPOT, OUTCAR (for Fermi energy) from a slab calculation.
"""
import numpy as np
import re
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ── Parse LOCPOT ──────────────────────────────────────────────────
def parse_locpot(locpot_file):
    """
    Parse VASP LOCPOT file.
    Returns: cell (3x3 Angstrom), atom_positions, potential_3d (nx, ny, nz) in eV.
    """
    with open(locpot_file, "r") as f:
        lines = f.readlines()

    scale = float(lines[1].strip())
    cell = np.zeros((3, 3))
    for i in range(3):
        cell[i] = [float(x) for x in lines[2 + i].split()]
    cell *= scale

    # Species and counts
    species_line = lines[5].split()
    counts_line = lines[6].split()
    n_atoms = sum(int(c) for c in counts_line)

    # Skip atomic positions (line 7 is coordinate type, then n_atoms lines)
    data_start = 8 + n_atoms
    # Blank line before grid
    while data_start < len(lines) and lines[data_start].strip() == "":
        data_start += 1

    # Grid dimensions
    grid_dims = [int(x) for x in lines[data_start].split()]
    nx, ny, nz = grid_dims
    data_start += 1

    # Read potential values
    values = []
    for i in range(data_start, len(lines)):
        line = lines[i].strip()
        if line == "":
            if len(values) >= nx * ny * nz:
                break
            continue
        values.extend([float(x) for x in line.split()])
        if len(values) >= nx * ny * nz:
            break

    potential = np.array(values[:nx * ny * nz]).reshape((nx, ny, nz), order='F')

    return cell, potential

def get_fermi_from_outcar(outcar_file):
    """Extract Fermi energy from VASP OUTCAR."""
    e_fermi = 0.0
    with open(outcar_file, "r") as f:
        for line in f:
            if "E-fermi" in line:
                m = re.search(r"E-fermi\s*:\s*([-\d.]+)", line)
                if m:
                    e_fermi = float(m.group(1))
    return e_fermi

def work_function_from_locpot(locpot_file, outcar_file, axis=2, output_png="work_function_vasp.png"):
    """
    Compute work function from VASP LOCPOT.
    """
    cell, potential = parse_locpot(locpot_file)
    e_fermi = get_fermi_from_outcar(outcar_file)
    print(f"Fermi energy: {e_fermi:.4f} eV")

    # Planar average along the specified axis
    avg_axes = tuple(i for i in range(3) if i != axis)
    v_planar = np.mean(potential, axis=avg_axes)

    c_length = np.linalg.norm(cell[axis])
    z = np.linspace(0, c_length, len(v_planar), endpoint=False)

    # Vacuum level: average potential at edges
    n_vac = max(len(v_planar) // 10, 5)
    v_left = np.mean(v_planar[:n_vac])
    v_right = np.mean(v_planar[-n_vac:])
    v_vacuum = (v_left + v_right) / 2.0

    wf = v_vacuum - e_fermi
    print(f"Vacuum level: {v_vacuum:.4f} eV")
    print(f"Work function: {wf:.4f} eV")

    # Plot
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(z, v_planar, "b-", linewidth=1.2, label="Planar average $V(z)$")
    ax.axhline(e_fermi, color="green", linestyle="--", label=f"$E_F$ = {e_fermi:.2f} eV")
    ax.axhline(v_vacuum, color="orange", linestyle="--", label=f"$E_{{vac}}$ = {v_vacuum:.2f} eV")

    z_arr = z[len(z) // 10]
    ax.annotate("", xy=(z_arr, e_fermi), xytext=(z_arr, v_vacuum),
                arrowprops=dict(arrowstyle="<->", color="black", lw=1.5))
    ax.text(z_arr + 0.5, (e_fermi + v_vacuum) / 2,
            f"$\\phi$ = {wf:.2f} eV", fontsize=12, va="center")

    ax.set_xlabel(r"z ($\mathrm{\AA}$)", fontsize=13)
    ax.set_ylabel("Electrostatic potential (eV)", fontsize=13)
    ax.set_title("Work Function (VASP LOCPOT)", fontsize=14)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_png, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"Saved: {output_png}")

    return wf

# Usage:
# wf = work_function_from_locpot("LOCPOT", "OUTCAR")
```

---

## Key Parameters

| Parameter | Typical Value | Notes |
|---|---|---|
| Vacuum thickness | 15--20 Ang | Must be large enough for the potential to plateau in vacuum |
| `ecutwfc` | 40--60 Ry | Must be converged for the surface |
| K-points | dense in-plane, 1 along z | e.g., `12 12 1` for a slab |
| `plot_num` | 11 | V_bare + V_Hartree (electrostatic potential). Use 1 for total V including V_xc |
| `degauss` | 0.01--0.02 Ry | Smearing for metallic slabs |
| Slab thickness | 5--7 layers minimum | Must be converged with respect to number of layers |
| `center_slab` | True | Center the slab in the cell for symmetric vacuum |

## Interpreting Results

- **Typical work functions**: Al(111) ~ 4.2 eV, Cu(111) ~ 4.9 eV, Au(111) ~ 5.3 eV, W(110) ~ 5.2 eV.
- **PBE accuracy**: PBE work functions are typically within 0.1--0.3 eV of experiment.
- **Symmetric slabs**: If the slab has identical top and bottom surfaces, the potential should plateau to the same value on both sides (V_left ~ V_right). Large asymmetry indicates insufficient vacuum or a polar surface.
- **Asymmetric slabs**: For slabs with different top/bottom terminations, each face has a different work function. Report both.
- **Dipole correction**: For polar slabs, use `tefield=.true.` and `dipfield=.true.` in QE (or `LDIPOL=.TRUE.` in VASP) to correct for the artificial electric field from periodic images.

## Common Issues

| Problem | Solution |
|---|---|
| Potential does not plateau in vacuum | Increase vacuum thickness to 20+ Angstrom |
| Large asymmetry between left/right vacuum levels | Enable dipole correction (`tefield=.true.`, `dipfield=.true.`) for polar surfaces |
| Work function converges slowly with slab thickness | Use at least 7 layers; test convergence by adding 2 more layers |
| pp.x gives "wrong plot_num" error | `plot_num=11` requires a completed SCF. Ensure `prefix` and `outdir` match |
| average.x gives unexpected output | Check that the input format matches: nfile, filename, weight, npt, idir, awin |
| K-point convergence issues | Converge in-plane k-grid; keep kz=1 for the slab supercell direction |
| Fermi energy parsing fails | Use `verbosity='high'` in the SCF input to print the Fermi energy explicitly |
