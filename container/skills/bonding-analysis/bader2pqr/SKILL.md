# Bader Charge to PQR Format Conversion

## When to Use

- Visualize atomic charges as color-mapped 3D spheres in VMD or PyMOL
- Analyze electrostatic potential distributions using charge-colored atomic models
- Generate publication-quality charge transfer visualizations with per-atom coloring
- Export Bader or Lowdin charges in PQR format readable by molecular visualization tools
- Corresponds to VASPKIT task 508 (Bader2PQR)

## Method Selection

| Criterion | From Bader - VASP/QE (Method A/B) | From Mulliken/Lowdin - QE (Method C) |
|---|---|---|
| Input | ACF.dat from bader program | projwfc.x output |
| Accuracy | High (real-space partitioning) | Moderate (basis-dependent) |
| External tools | bader binary, pp.x | projwfc.x only |
| Best for | Charge transfer, oxidation states | Quick qualitative analysis |
| PQR output | ATOM x y z charge radius | ATOM x y z charge radius |

```
Have VASP CHGCAR and want charge visualization?
  --> Method B: Run bader on CHGCAR, convert ACF.dat to PQR

Have QE charge density and want accurate Bader charges?
  --> Method A: Run pp.x for cube file, bader analysis, convert to PQR

Need a quick qualitative charge picture without external tools?
  --> Method C: Parse projwfc.x Lowdin charges, convert to PQR
```

## Prerequisites

- Python: `numpy`, `matplotlib`, `ase`
- For Bader (Methods A/B): Henkelman group `bader` binary
- For QE (Methods A/C): Quantum ESPRESSO `pw.x`, `pp.x`, `projwfc.x`
- For VASP (Method B): CHGCAR / AECCAR files from VASP run
- VMD or PyMOL for visualization of PQR files

## Detailed Steps

### Method A: From QE Bader Analysis to PQR

```python
#!/usr/bin/env python3
"""
Complete workflow: run QE SCF, extract charge density, run Bader analysis,
parse ACF.dat, convert to PQR format, and generate VMD visualization script.
Covers VASPKIT 508 functionality: Bader charges to PQR format.
"""
import numpy as np
import subprocess
import os
import json
from ase.io import read
from ase.data import covalent_radii, atomic_numbers
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ============================================================
# 1. Configuration
# ============================================================
PREFIX = "system"
PSEUDO_DIR = os.path.abspath("./pseudo")
WORK_DIR = os.path.abspath("./bader2pqr")
os.makedirs(WORK_DIR, exist_ok=True)
os.makedirs(os.path.join(WORK_DIR, "tmp"), exist_ok=True)
NPROC = os.cpu_count() or 4

VALENCE_ELECTRONS = {
    "Ti": 12, "O": 6, "Cu": 11, "C": 4, "N": 5, "H": 1,
    "Fe": 16, "Ni": 10, "Pt": 10, "Si": 4, "Zn": 12, "Co": 9,
}

# ============================================================
# 2. SCF calculation (example: TiO2 rutile)
# ============================================================
scf_input = f"""&CONTROL
    calculation='scf', prefix='{PREFIX}', outdir='./tmp',
    pseudo_dir='{PSEUDO_DIR}', tprnfor=.true.
/
&SYSTEM
    ibrav=6, celldm(1)=8.6806, celldm(3)=0.6441,
    nat=6, ntyp=2, ecutwfc=60.0, ecutrho=600.0
/
&ELECTRONS
    conv_thr=1.0d-8, mixing_beta=0.4
/
ATOMIC_SPECIES
  Ti 47.867 Ti.pbe-spn-rrkjus_psl.1.0.0.UPF
  O  15.999 O.pbe-n-rrkjus_psl.1.0.0.UPF
ATOMIC_POSITIONS {{crystal}}
  Ti 0.0000 0.0000 0.0000
  Ti 0.5000 0.5000 0.5000
  O  0.3053 0.3053 0.0000
  O  0.6947 0.6947 0.0000
  O  0.1947 0.8053 0.5000
  O  0.8053 0.1947 0.5000
K_POINTS {{automatic}}
  6 6 8 0 0 0
"""
with open(os.path.join(WORK_DIR, "scf.in"), "w") as f:
    f.write(scf_input)

print("Running SCF...")
r = subprocess.run(["mpirun","--allow-run-as-root","-np",str(NPROC),
    "pw.x","-in","scf.in"], capture_output=True, text=True, timeout=1800, cwd=WORK_DIR)
with open(os.path.join(WORK_DIR,"scf.out"),"w") as f: f.write(r.stdout)

# ============================================================
# 3. Extract charge density as cube file
# ============================================================
pp_valence = f"""&INPUTPP
    prefix='{PREFIX}', outdir='./tmp', filplot='charge_val.dat', plot_num=0
/
&PLOT
    nfile=1, filepp(1)='charge_val.dat', weight(1)=1.0,
    iflag=3, output_format=6, fileout='charge_valence.cube'
/
"""
pp_ae = f"""&INPUTPP
    prefix='{PREFIX}', outdir='./tmp', filplot='charge_ae.dat', plot_num=21
/
&PLOT
    nfile=1, filepp(1)='charge_ae.dat', weight(1)=1.0,
    iflag=3, output_format=6, fileout='charge_ae.cube'
/
"""
for name, content in [("pp_val.in", pp_valence), ("pp_ae.in", pp_ae)]:
    with open(os.path.join(WORK_DIR, name), "w") as f: f.write(content)
    subprocess.run(["pp.x","-in",name], capture_output=True, text=True,
                   timeout=300, cwd=WORK_DIR)

# ============================================================
# 4. Run Bader analysis
# ============================================================
bader_cmd = ["bader", "charge_valence.cube"]
if os.path.exists(os.path.join(WORK_DIR, "charge_ae.cube")):
    bader_cmd += ["-ref", "charge_ae.cube"]
subprocess.run(bader_cmd, capture_output=True, text=True, timeout=300, cwd=WORK_DIR)

# ============================================================
# 5. Parse ACF.dat
# ============================================================
def parse_acf(acf_file):
    """Parse ACF.dat from Henkelman Bader program."""
    atoms = []
    with open(acf_file) as f:
        for line in f:
            line = line.strip()
            if line.startswith(("#","-")) or not line or line.startswith(("VACUUM","NUMBER")):
                continue
            parts = line.split()
            if len(parts) >= 7:
                try:
                    atoms.append({"index": int(parts[0]), "x": float(parts[1]),
                        "y": float(parts[2]), "z": float(parts[3]),
                        "charge": float(parts[4]), "volume": float(parts[6])})
                except ValueError: continue
    return atoms

atom_symbols = ["Ti", "Ti", "O", "O", "O", "O"]
acf_file = os.path.join(WORK_DIR, "ACF.dat")
if os.path.exists(acf_file):
    bader_data = parse_acf(acf_file)
else:
    print("WARNING: ACF.dat not found. Using example data.")
    bader_data = [
        {"index":1,"x":0.0,"y":0.0,"z":0.0,"charge":9.78,"volume":45.3},
        {"index":2,"x":2.3,"y":2.3,"z":1.48,"charge":9.78,"volume":45.3},
        {"index":3,"x":1.4,"y":1.4,"z":0.0,"charge":7.11,"volume":18.2},
        {"index":4,"x":3.2,"y":3.2,"z":0.0,"charge":7.11,"volume":18.2},
        {"index":5,"x":0.9,"y":3.7,"z":1.48,"charge":7.11,"volume":18.2},
        {"index":6,"x":3.7,"y":0.9,"z":1.48,"charge":7.11,"volume":18.2},
    ]

# Compute charge transfers
charge_transfers = []
print(f"\n{'#':<4} {'Sym':<4} {'Bader(e)':<10} {'Val(e)':<8} {'dQ(e)':<10} {'Oxid':<6}")
print("-" * 44)
for i, ad in enumerate(bader_data):
    sym = atom_symbols[i]
    val_e = VALENCE_ELECTRONS.get(sym, 0)
    dq = val_e - ad["charge"]
    charge_transfers.append(dq)
    sign = "+" if dq > 0 else "-" if dq < 0 else ""
    print(f" {i+1:<3} {sym:<4} {ad['charge']:<10.4f} {val_e:<8} {dq:<+10.4f} {sign}{abs(round(dq))}")

# ============================================================
# 6. Write PQR file
# ============================================================
def write_pqr(filename, symbols, positions, charges, radii=None, remark="Bader charges"):
    """Write PQR format: ATOM serial name resName chain resSeq x y z charge radius"""
    if radii is None:
        radii = [covalent_radii[atomic_numbers.get(s, 1)] for s in symbols]
    lines = [f"REMARK  {remark}", f"REMARK  N_atoms = {len(symbols)}"]
    for i, (sym, pos, q, r) in enumerate(zip(symbols, positions, charges, radii)):
        s = i + 1
        lines.append(f"ATOM  {s:>5d} {sym+str(s):<4s} MOL A   1    "
                      f"{pos[0]:>8.3f}{pos[1]:>8.3f}{pos[2]:>8.3f}{q:>8.4f}{r:>7.4f}")
    lines.append("END")
    with open(filename, "w") as f: f.write("\n".join(lines) + "\n")
    print(f"Saved PQR: {filename}")

positions = np.array([[ad["x"], ad["y"], ad["z"]] for ad in bader_data])
pqr_file = os.path.join(WORK_DIR, "bader_charges.pqr")
write_pqr(pqr_file, atom_symbols, positions, charge_transfers)

# ============================================================
# 7. Generate VMD visualization script
# ============================================================
def write_vmd_script(vmd_file, pqr_file, charges):
    """VMD Tcl script: load PQR, color by charge, Blue-White-Red scale."""
    q_abs = max(abs(min(charges)), abs(max(charges)), 0.01)
    lines = [
        "# VMD script for Bader charge visualization",
        "display projection Orthographic", "display depthcue off",
        "axes location Off", "color Display Background white", "",
        f'mol new "{os.path.basename(pqr_file)}" type pqr waitfor all',
        "mol delrep 0 top", "",
        "mol representation CPK 1.0 0.3 30 30", "mol color Beta",
        "mol selection all", "mol addrep top", "",
        "color scale method BWR",
        f"color scale min {-q_abs:.4f}", f"color scale max {q_abs:.4f}", "",
        "set sel [atomselect top all]",
        "set charges {" + " ".join(f"{q:.6f}" for q in charges) + "}",
        "$sel set beta $charges", "",
        "display resetview", "scale by 1.5", "",
        f'puts "Charges loaded. Blue=anion({-q_abs:.2f}) White=0 Red=cation({q_abs:.2f})"',
    ]
    with open(vmd_file, "w") as f: f.write("\n".join(lines) + "\n")
    print(f"Saved VMD script: {vmd_file}")

vmd_file = os.path.join(WORK_DIR, "visualize_charges.tcl")
write_vmd_script(vmd_file, pqr_file, charge_transfers)

# ============================================================
# 8. Quick matplotlib charge map
# ============================================================
fig, ax = plt.subplots(figsize=(8, 7))
q_vals = np.array(charge_transfers)
q_abs_max = max(abs(q_vals.min()), abs(q_vals.max()), 0.01)
sc = ax.scatter(positions[:,0], positions[:,1], c=q_vals, cmap="RdBu_r",
    vmin=-q_abs_max, vmax=q_abs_max, s=800, edgecolors="black", linewidths=1.0)
for i, (x,y,sym,q) in enumerate(zip(positions[:,0], positions[:,1], atom_symbols, q_vals)):
    ax.annotate(f"{sym}\n{q:+.2f}e", (x,y), ha="center", va="center", fontsize=9)
plt.colorbar(sc, ax=ax, shrink=0.8, label="Charge transfer (e)")
ax.set_xlabel("x (A)"); ax.set_ylabel("y (A)")
ax.set_title("Bader Charge Transfer Map"); ax.set_aspect("equal"); ax.grid(alpha=0.3)
fig.tight_layout(); fig.savefig(os.path.join(WORK_DIR,"bader_charge_map.png"), dpi=150)
plt.close(); print("Saved bader_charge_map.png")

with open(os.path.join(WORK_DIR,"bader2pqr_results.json"),"w") as f:
    json.dump({"atom_symbols": atom_symbols, "charge_transfers": [float(q) for q in charge_transfers],
               "pqr_file": pqr_file, "vmd_script": vmd_file}, f, indent=2)
print("Saved bader2pqr_results.json")
```

### Method B: From VASP Bader Analysis to PQR (VASPKIT 508)

```python
#!/usr/bin/env python3
"""
Convert VASP Bader charge analysis to PQR format for VMD visualization.
Replicates VASPKIT task 508.

Prerequisites:
  1. VASP run with LAECHG=.TRUE. -> produces AECCAR0, AECCAR2
  2. chgsum.pl AECCAR0 AECCAR2 -> CHGCAR_sum
  3. bader CHGCAR -ref CHGCAR_sum -> ACF.dat
"""
import numpy as np
import os
import json
from ase.io import read
from ase.data import covalent_radii, atomic_numbers

POSCAR_FILE = "CONTCAR"
ACF_FILE = "ACF.dat"

VALENCE_ELECTRONS = {
    "Ti": 12, "O": 6, "Cu": 11, "C": 4, "N": 5, "H": 1,
    "Fe": 16, "Ni": 10, "Pt": 10, "Si": 4, "Zn": 12, "Al": 3,
}

# 1. Read structure
try:
    atoms = read(POSCAR_FILE)
except FileNotFoundError:
    print(f"WARNING: {POSCAR_FILE} not found. Using example TiO2.")
    from ase.build import bulk
    atoms = bulk("TiO2", crystalstructure="rutile", a=4.594, c=2.959)

atom_symbols = atoms.get_chemical_symbols()
positions = atoms.get_positions()
n_atoms = len(atoms)
print(f"Structure: {atoms.get_chemical_formula()}, {n_atoms} atoms")

# 2. Parse ACF.dat
def parse_acf(acf_file):
    data = []
    with open(acf_file) as f:
        for line in f:
            line = line.strip()
            if line.startswith(("#","-")) or not line or line.startswith(("VACUUM","NUMBER")):
                continue
            parts = line.split()
            if len(parts) >= 7:
                try:
                    data.append({"index": int(parts[0]), "charge": float(parts[4]),
                                 "volume": float(parts[6])})
                except ValueError: continue
    return data

try:
    bader_data = parse_acf(ACF_FILE)
except FileNotFoundError:
    print(f"WARNING: {ACF_FILE} not found. Using example charges.")
    bader_data = []
    for i, sym in enumerate(atom_symbols):
        val = VALENCE_ELECTRONS.get(sym, 4)
        bq = val - (2.3 if sym == "Ti" else -1.15 if sym == "O" else 0)
        bader_data.append({"index": i+1, "charge": bq, "volume": 20.0})

# 3. Compute charge transfers
charge_transfers = []
print(f"\n{'#':<4} {'Sym':<4} {'Bader(e)':<10} {'dQ(e)':<10}")
print("-" * 30)
for i, ad in enumerate(bader_data):
    sym = atom_symbols[i] if i < len(atom_symbols) else "?"
    val = VALENCE_ELECTRONS.get(sym, 0)
    dq = val - ad["charge"]
    charge_transfers.append(dq)
    print(f" {i+1:<3} {sym:<4} {ad['charge']:<10.4f} {dq:<+10.4f}")

# 4. Write PQR
def write_pqr(fname, syms, pos, charges, radii=None):
    if radii is None:
        radii = [covalent_radii[atomic_numbers.get(s,1)] for s in syms]
    lines = ["REMARK  PQR from VASP Bader (VASPKIT 508)"]
    for i, (sym, p, q, r) in enumerate(zip(syms, pos, charges, radii)):
        s = i + 1
        lines.append(f"ATOM  {s:>5d} {sym+str(s):<4s} MOL A   1    "
                      f"{p[0]:>8.3f}{p[1]:>8.3f}{p[2]:>8.3f}{q:>8.4f}{r:>7.4f}")
    lines.append("END")
    with open(fname, "w") as f: f.write("\n".join(lines)+"\n")
    print(f"Saved PQR: {fname}")

pqr_path = "bader_vasp.pqr"
write_pqr(pqr_path, atom_symbols, positions, charge_transfers)

# 5. VMD script
q_abs = max(abs(min(charge_transfers)), abs(max(charge_transfers)), 0.01)
vmd_lines = [
    "# VMD script for VASP Bader charge visualization",
    "display projection Orthographic", "display depthcue off",
    "axes location Off", "color Display Background white", "",
    f'mol new "{pqr_path}" type pqr waitfor all',
    "mol delrep 0 top", "",
    "mol representation CPK 1.2 0.3 30 30", "mol color Beta",
    "mol selection all", "mol addrep top", "",
    "color scale method BWR",
    f"color scale min {-q_abs:.4f}", f"color scale max {q_abs:.4f}", "",
    "set sel [atomselect top all]",
    "set charges {" + " ".join(f"{q:.6f}" for q in charge_transfers) + "}",
    "$sel set beta $charges", "",
    "pbc box -color black -width 2",
    "display resetview", "scale by 1.5",
]
with open("visualize_bader_vasp.tcl", "w") as f: f.write("\n".join(vmd_lines)+"\n")
print("Saved visualize_bader_vasp.tcl")

# 6. PyMOL script
pymol_lines = [
    f"# PyMOL script for Bader charge visualization",
    f"load {pqr_path}, bader", "",
    "hide everything, bader", "show spheres, bader",
    "set sphere_scale, 0.4, bader", "",
    "spectrum b, blue_white_red, bader",
    "bg_color white", "",
]
for i, (sym, q) in enumerate(zip(atom_symbols, charge_transfers)):
    pymol_lines.append(f'label bader and id {i+1}, "%s %+.2f" % (name, b)')
pymol_lines += ["", "set label_size, 14", "set label_color, black", "zoom bader"]
with open("visualize_bader_vasp.pml", "w") as f: f.write("\n".join(pymol_lines)+"\n")
print("Saved visualize_bader_vasp.pml")

with open("bader2pqr_vasp.json","w") as f:
    json.dump({"atom_symbols": atom_symbols,
               "charge_transfers": [float(q) for q in charge_transfers],
               "pqr_file": pqr_path}, f, indent=2)
print("Saved bader2pqr_vasp.json")
```

### Method C: From QE Lowdin/Mulliken Charges to PQR

```python
#!/usr/bin/env python3
"""
Parse Lowdin charges from QE projwfc.x output and convert to PQR format.
No external Bader program needed -- uses QE's built-in orbital projection.
Less accurate than Bader but useful for quick qualitative visualization.
"""
import numpy as np
import re
import os
import json
import subprocess
from ase.io import read
from ase.data import covalent_radii, atomic_numbers
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

PREFIX = "system"
WORK_DIR = os.path.abspath("./lowdin2pqr")
os.makedirs(WORK_DIR, exist_ok=True)
os.makedirs(os.path.join(WORK_DIR, "tmp"), exist_ok=True)
NPROC = os.cpu_count() or 4

VALENCE_ELECTRONS = {
    "Ti": 12, "O": 6, "Cu": 11, "C": 4, "N": 5, "H": 1,
    "Fe": 16, "Ni": 10, "Pt": 10, "Si": 4,
}

# 1. Run projwfc.x (assumes SCF already completed)
projwfc_input = f"""&PROJWFC
    prefix='{PREFIX}', outdir='./tmp', filpdos='pdos',
    Emin=-30.0, Emax=20.0, DeltaE=0.1, ngauss=0, degauss=0.01
/
"""
with open(os.path.join(WORK_DIR,"projwfc.in"),"w") as f: f.write(projwfc_input)
r = subprocess.run(["mpirun","--allow-run-as-root","-np",str(NPROC),
    "projwfc.x","-in","projwfc.in"], capture_output=True, text=True,
    timeout=600, cwd=WORK_DIR)
with open(os.path.join(WORK_DIR,"projwfc.out"),"w") as f: f.write(r.stdout)

# 2. Parse Lowdin charges
def parse_lowdin(projwfc_out):
    """Parse 'Atom # N: total charge = X, s = Y, p = Z, ...' lines."""
    with open(projwfc_out) as f: content = f.read()
    pattern = r'Atom\s*#\s*(\d+):\s*total charge\s*=\s*([\d.]+)((?:,\s*\w+\s*=\s*[\d.]+)*)'
    matches = re.findall(pattern, content)
    atoms = []
    for m in matches:
        orbitals = {}
        if m[2]:
            for orb, val in re.findall(r'(\w+)\s*=\s*([\d.]+)', m[2]):
                orbitals[orb] = float(val)
        atoms.append({"atom_index": int(m[0]), "total": float(m[1]), **orbitals})
    spilling = None
    sm = re.search(r'Spilling\s+Parameter\s*:\s*([\d.]+)', content)
    if sm: spilling = float(sm.group(1))
    return atoms, spilling

projwfc_out = os.path.join(WORK_DIR, "projwfc.out")
if os.path.exists(projwfc_out):
    lowdin_data, spilling = parse_lowdin(projwfc_out)
else:
    print("WARNING: projwfc.out not found. Using example data.")
    lowdin_data = [
        {"atom_index":1,"total":10.15,"s":2.15,"p":5.80,"d":2.20},
        {"atom_index":2,"total":10.15,"s":2.15,"p":5.80,"d":2.20},
        {"atom_index":3,"total":6.92,"s":1.82,"p":5.10},
        {"atom_index":4,"total":6.92,"s":1.82,"p":5.10},
        {"atom_index":5,"total":6.92,"s":1.82,"p":5.10},
        {"atom_index":6,"total":6.92,"s":1.82,"p":5.10},
    ]
    spilling = 0.005

atom_symbols = ["Ti", "Ti", "O", "O", "O", "O"]
if spilling is not None:
    print(f"Spilling: {spilling:.4f} ({'OK' if spilling < 0.01 else 'WARNING: >1%'})")

# 3. Charge transfers
charge_transfers = []
print(f"\n{'#':<4} {'Sym':<4} {'Lowdin(e)':<11} {'dQ(e)':<10}")
print("-" * 32)
for i, ld in enumerate(lowdin_data):
    sym = atom_symbols[i]
    dq = VALENCE_ELECTRONS.get(sym, 0) - ld["total"]
    charge_transfers.append(dq)
    print(f" {i+1:<3} {sym:<4} {ld['total']:<11.4f} {dq:<+10.4f}")

# 4. Get positions
try:
    struct = read(os.path.join(WORK_DIR, "scf.in"), format="espresso-in")
    positions = struct.get_positions()
except Exception:
    positions = np.array([[0,0,0],[2.3,2.3,1.48],[1.4,1.4,0],
                          [3.2,3.2,0],[0.9,3.7,1.48],[3.7,0.9,1.48]])

# 5. Write PQR
pqr_file = os.path.join(WORK_DIR, "lowdin_charges.pqr")
lines = ["REMARK  PQR from QE Lowdin charges"]
for i, (sym, pos, q) in enumerate(zip(atom_symbols, positions, charge_transfers)):
    r = covalent_radii[atomic_numbers.get(sym, 1)]
    s = i + 1
    lines.append(f"ATOM  {s:>5d} {sym+str(s):<4s} MOL A   1    "
                  f"{pos[0]:>8.3f}{pos[1]:>8.3f}{pos[2]:>8.3f}{q:>8.4f}{r:>7.4f}")
lines.append("END")
with open(pqr_file, "w") as f: f.write("\n".join(lines)+"\n")
print(f"\nSaved PQR: {pqr_file}")

# 6. VMD script
q_abs = max(abs(min(charge_transfers)), abs(max(charge_transfers)), 0.01)
vmd_lines = [
    "# VMD script for Lowdin charge visualization",
    "display projection Orthographic", "display depthcue off",
    "axes location Off", "color Display Background white", "",
    f'mol new "{os.path.basename(pqr_file)}" type pqr waitfor all',
    "mol delrep 0 top", "mol representation CPK 1.0 0.3 30 30",
    "mol color Beta", "mol selection all", "mol addrep top", "",
    "color scale method BWR",
    f"color scale min {-q_abs:.4f}", f"color scale max {q_abs:.4f}", "",
    "set sel [atomselect top all]",
    "set charges {" + " ".join(f"{q:.6f}" for q in charge_transfers) + "}",
    "$sel set beta $charges", "",
    "display resetview", "scale by 1.5",
]
vmd_file = os.path.join(WORK_DIR, "visualize_lowdin.tcl")
with open(vmd_file, "w") as f: f.write("\n".join(vmd_lines)+"\n")
print(f"Saved VMD script: {vmd_file}")

# 7. Comparison plot
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
labels = [f"{atom_symbols[i]}{i+1}" for i in range(len(charge_transfers))]
colors = ["#d62728" if q>0.1 else "#2ca02c" if q<-0.1 else "#7f7f7f" for q in charge_transfers]
bars = ax1.bar(range(len(charge_transfers)), charge_transfers, color=colors, edgecolor="black")
ax1.set_xticks(range(len(labels))); ax1.set_xticklabels(labels, rotation=45, ha="right")
ax1.set_ylabel("Charge transfer (e)"); ax1.set_title("Lowdin Charge Transfer")
ax1.axhline(0, color="black", linewidth=0.5); ax1.grid(axis="y", alpha=0.3)
for bar, val in zip(bars, charge_transfers):
    ax1.text(bar.get_x()+bar.get_width()/2, bar.get_height(),
             f"{val:+.2f}", ha="center", va="bottom" if val>=0 else "top", fontsize=9)

# Orbital breakdown
orb_types = [k for k in lowdin_data[0] if k not in ("atom_index","total")]
orb_colors = {"s":"#1f77b4","p":"#ff7f0e","d":"#2ca02c","f":"#d62728"}
x = np.arange(len(lowdin_data)); bottom = np.zeros(len(lowdin_data))
for orb in orb_types:
    vals = [ld.get(orb,0) for ld in lowdin_data]
    ax2.bar(x, vals, bottom=bottom, label=orb, color=orb_colors.get(orb,"#8c564b"),
            edgecolor="black", linewidth=0.3)
    bottom += np.array(vals)
ax2.set_xticks(x); ax2.set_xticklabels(labels, rotation=45, ha="right")
ax2.set_ylabel("Lowdin charge (e)"); ax2.set_title("Orbital Breakdown")
ax2.legend(title="Orbital"); ax2.grid(axis="y", alpha=0.3)

fig.tight_layout()
fig.savefig(os.path.join(WORK_DIR,"lowdin_analysis.png"), dpi=150); plt.close()
print("Saved lowdin_analysis.png")

with open(os.path.join(WORK_DIR,"lowdin2pqr_results.json"),"w") as f:
    json.dump({"atom_symbols": atom_symbols,
               "charge_transfers": [float(q) for q in charge_transfers],
               "orbital_data": lowdin_data, "spilling": spilling}, f, indent=2)
print("Saved lowdin2pqr_results.json")
```

## Key Parameters

| Parameter | Recommended Value | Notes |
|---|---|---|
| PQR charge field | Charge transfer (val_e - bader_charge) | Positive = cation, Negative = anion |
| PQR radius field | Covalent radius (ASE data) | Or van der Waals for space-filling |
| VMD color scale | BWR (Blue-White-Red) | Blue = anion, White = neutral, Red = cation |
| Bader reference density | AECCAR0+AECCAR2 (VASP), plot_num=21 (QE PAW) | All-electron for accurate partitioning |
| ecutrho for cube file | >= 600 Ry | Fine grid needed for Bader basins |
| Spilling (Lowdin) | < 0.01 (1%) | Above 1%: projection incomplete |
| Valence electrons | From pseudopotential header | MUST match your PP |

## Interpreting Results

1. **PQR format**: Each ATOM line has coordinates (Angstrom), charge, and radius. VMD and PyMOL read this natively.
2. **Sign convention**: Positive charge transfer = lost electrons (cation). Negative = gained electrons (anion).
3. **VMD coloring**: BWR scale maps blue to anions, white to neutral, red to cations.
4. **Bader vs Lowdin**: Bader is more physical (real-space zero-flux surfaces). Lowdin depends on basis and underestimates transfer. Both work for qualitative visualization.
5. **Typical values**: Ionic compounds (TiO2): Ti ~+2.3 e, O ~-1.1 e. Metals/alloys: < 0.5 e per atom.
6. **VMD tips**: Adjust color range via "Graphics > Colors > Color Scale". Use "Extensions > Visualization > Color Scale Bar" for legends.

## Common Issues

| Issue | Solution |
|---|---|
| PQR not loading in VMD | Set file type to "pqr": `mol new file.pqr type pqr` |
| Colors all look the same | Charge range too wide/narrow. Adjust `color scale min/max`. |
| Wrong atom count in ACF.dat | Bader splits atoms at periodic boundaries. Use `bader -vac off`. |
| Charges do not sum to zero | Small residual (<0.05 e) normal. Larger: check vacuum charge. |
| Lowdin spilling > 1% | Bader is more reliable in this case. |
| AECCAR files missing | Set `LAECHG=.TRUE.` in INCAR and re-run VASP. |
| Cube file too large | Reduce ecutrho (minimum ~400 Ry for Bader). |
| bader not found | Install from theory.cm.utexas.edu/henkelman/code/bader or `pip install pybader`. |
