# Band Edge Alignment for 2D Materials

## When to Use

- Determine the absolute positions of the valence band maximum (VBM) and conduction band minimum (CBM) relative to the vacuum level
- Calculate the work function of a 2D material or surface
- Compute ionization potential (IP) and electron affinity (EA)
- Assess photocatalytic potential for water splitting (band edges relative to redox potentials)
- Design heterojunctions by aligning band edges of two materials
- Corresponds to VASPKIT task 927

## Method Selection

| Criterion | QE DFT (Method A) | VASP (Method B) | MACE (N/A) |
|---|---|---|---|
| Work function | From planar-averaged potential | From LOCPOT via VASPKIT 927 | Not available |
| Band edges | Vacuum level - E_VBM, Vacuum level - E_CBM | Same via VASPKIT | Not available |
| Why MACE cannot | Force field has no electronic states or electrostatic potential | -- | -- |

```
Need absolute band edge positions?
  --> Method A (QE) or Method B (VASP): DFT required

Need work function of a surface?
  --> Same workflow: phi = V_vacuum - E_Fermi

Need to compare two materials for heterojunction?
  --> Compute band edges for each material separately, then align

Can MACE help?
  --> Only for structure relaxation before the DFT calculation
```

## Prerequisites

- Quantum ESPRESSO (pw.x, pp.x, average.x) -- Method A
- A relaxed 2D/slab structure with sufficient vacuum (>= 20 A)
- numpy, matplotlib
- pymatgen (structure manipulation)

## Detailed Steps

### Method A: QE DFT -- Work Function and Band Edges

#### Complete Workflow

```python
#!/usr/bin/env python3
"""
Compute work function, ionization potential, and electron affinity
for a 2D material using QE.

Workflow:
  1. Relax 2D structure with MACE
  2. Run QE SCF
  3. Extract band edges (E_VBM, E_CBM) from NSCF
  4. Extract vacuum level from planar-averaged potential (pp.x + average.x)
  5. Compute: phi = V_vacuum - E_Fermi
              IP  = V_vacuum - E_VBM
              EA  = V_vacuum - E_CBM

Example: Monolayer MoS2.
"""

import os
import subprocess
import re
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pymatgen.core import Structure, Lattice
import json

# ============================================================
# 1. Prepare structure
# ============================================================
# MoS2 monolayer with 20 A vacuum
mos2 = Structure(
    Lattice.hexagonal(3.16, 25.0),
    ["Mo", "S", "S"],
    [[1/3, 2/3, 0.50], [1/3, 2/3, 0.562], [1/3, 2/3, 0.438]],
)

# Optional: relax with MACE first
from ase.optimize import LBFGS
from ase.constraints import FixAtoms
from pymatgen.io.ase import AseAtomsAdaptor
from mace.calculators import mace_mp

calc = mace_mp(model="medium", dispersion=True, default_dtype="float64")
adaptor = AseAtomsAdaptor()
atoms = adaptor.get_atoms(mos2)
atoms.calc = calc

# Fix c-axis (only relax in-plane)
from ase.filters import FrechetCellFilter
ecf = FrechetCellFilter(atoms, mask=[True, True, False, False, False, True])
opt = LBFGS(ecf, logfile="relax_mos2.log")
opt.run(fmax=0.005, steps=200)

mos2_relaxed = adaptor.get_structure(atoms)
print(f"Relaxed MoS2: a = {mos2_relaxed.lattice.a:.4f} A")

# ============================================================
# 2. Generate QE inputs (SCF, NSCF, pp.x)
# ============================================================
PSEUDO_DIR = os.path.abspath("./pseudo")
OUTDIR = os.path.abspath("./tmp_bandedge")
os.makedirs(PSEUDO_DIR, exist_ok=True)
os.makedirs(OUTDIR, exist_ok=True)
PREFIX = "mos2"

struct = mos2_relaxed
n_atoms = len(struct)
cell = struct.lattice.matrix
positions = struct.frac_coords
symbols = [str(s.specie) for s in struct]
elements = sorted(set(symbols))

pseudos = {
    "Mo": "Mo.pbe-spn-kjpaw_psl.1.0.0.UPF",
    "S": "S.pbe-n-kjpaw_psl.1.0.0.UPF",
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

# --- SCF input ---
from pymatgen.core.periodic_table import Element

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
    ecutwfc     = 50.0
    ecutrho     = 400.0
    occupations = 'smearing'
    smearing    = 'cold'
    degauss     = 0.005
    assume_isolated = '2D'
/
&ELECTRONS
    conv_thr = 1.0d-8
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

scf_input += "\nK_POINTS automatic\n  12 12 1  0 0 0\n"

with open(f"{PREFIX}_scf.in", "w") as f:
    f.write(scf_input)

# --- NSCF input (for band edges) ---
nscf_input = scf_input.replace("'scf'", "'nscf'")
nscf_input = nscf_input.replace("12 12 1  0 0 0", "18 18 1  0 0 0")
nscf_input = nscf_input.replace(
    "degauss     = 0.005",
    "degauss     = 0.005\n    nbnd        = 30"
)

with open(f"{PREFIX}_nscf.in", "w") as f:
    f.write(nscf_input)

# --- pp.x input (electrostatic potential) ---
pp_input = f"""&INPUTPP
    prefix   = '{PREFIX}'
    outdir   = '{OUTDIR}'
    filplot  = '{PREFIX}_vlocal.dat'
    plot_num = 11
/
&PLOT
    iflag         = 3
    output_format = 0
    fileout       = '{PREFIX}_v_avg.dat'
    e1(1) = 0.0, e1(2) = 0.0, e1(3) = 1.0
    x0(1) = 0.0, x0(2) = 0.0, x0(3) = 0.0
    nx = 1000
/
"""

with open(f"{PREFIX}_pp.in", "w") as f:
    f.write(pp_input)

# --- average.x input (planar averaging) ---
avg_input = f"""1
{PREFIX}_vlocal.dat
1.0
1000
3
1.000
"""

with open(f"{PREFIX}_average.in", "w") as f:
    f.write(avg_input)

print("Generated QE input files:")
print(f"  {PREFIX}_scf.in")
print(f"  {PREFIX}_nscf.in")
print(f"  {PREFIX}_pp.in")
print(f"  {PREFIX}_average.in")
```

#### Step 2: Run QE Calculations

```bash
#!/bin/bash
# Complete band edge calculation workflow

PREFIX="mos2"
NPROC=$(nproc)

echo "=== Step 1: SCF ==="
mpirun --allow-run-as-root -np $NPROC pw.x -in ${PREFIX}_scf.in > ${PREFIX}_scf.out 2>&1
grep "convergence has been achieved" ${PREFIX}_scf.out && echo "SCF converged" || echo "SCF FAILED"

echo "=== Step 2: NSCF ==="
mpirun --allow-run-as-root -np $NPROC pw.x -in ${PREFIX}_nscf.in > ${PREFIX}_nscf.out 2>&1
echo "NSCF done"

echo "=== Step 3: pp.x (electrostatic potential) ==="
pp.x -in ${PREFIX}_pp.in > ${PREFIX}_pp.out 2>&1
echo "pp.x done"

echo "=== Step 4: average.x (planar averaging) ==="
average.x < ${PREFIX}_average.in > ${PREFIX}_average.out 2>&1
echo "average.x done"
```

#### Step 3: Extract Band Edges and Vacuum Level

```python
#!/usr/bin/env python3
"""
Parse QE outputs to extract:
  - Fermi energy / band edges (VBM, CBM)
  - Vacuum level from planar-averaged potential
  - Work function, ionization potential, electron affinity
"""

import re
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import json

PREFIX = "mos2"

# ============================================================
# 1. Extract Fermi energy and band edges from NSCF output
# ============================================================
def extract_band_edges(nscf_output):
    """
    Extract VBM, CBM, and Fermi energy from QE NSCF output.

    Returns
    -------
    dict with e_fermi, e_vbm, e_cbm, band_gap
    """
    e_fermi = None
    e_vbm = None
    e_cbm = None

    with open(nscf_output) as f:
        content = f.read()

    # Try "highest occupied, lowest unoccupied" format (semiconductors)
    m = re.search(
        r"highest occupied, lowest unoccupied level \(ev\):\s+([-\d.]+)\s+([-\d.]+)",
        content, re.IGNORECASE
    )
    if m:
        e_vbm = float(m.group(1))
        e_cbm = float(m.group(2))
        e_fermi = (e_vbm + e_cbm) / 2  # mid-gap

    # Try "the Fermi energy is" format (metals)
    m2 = re.search(r"the Fermi energy is\s+([-\d.]+)\s+ev", content, re.IGNORECASE)
    if m2:
        e_fermi = float(m2.group(1))
        if e_vbm is None:
            e_vbm = e_fermi
            e_cbm = e_fermi  # metal: no gap

    # Try "highest occupied level" format (insulators, no empty bands)
    m3 = re.search(
        r"highest occupied level \(ev\):\s+([-\d.]+)",
        content, re.IGNORECASE
    )
    if m3 and e_vbm is None:
        e_vbm = float(m3.group(1))
        e_fermi = e_vbm

    result = {
        "e_fermi_eV": e_fermi,
        "e_vbm_eV": e_vbm,
        "e_cbm_eV": e_cbm,
    }

    if e_vbm is not None and e_cbm is not None:
        result["band_gap_eV"] = e_cbm - e_vbm
    else:
        result["band_gap_eV"] = None

    return result


# ============================================================
# 2. Extract vacuum level from planar-averaged potential
# ============================================================
def extract_vacuum_level(avg_output_file, potential_data_file=None):
    """
    Extract the vacuum level from the planar-averaged electrostatic potential.

    The vacuum level is the average value of the potential in the vacuum
    region (where it should be flat/constant).

    Parameters
    ----------
    avg_output_file : str
        Output from average.x.
    potential_data_file : str, optional
        Direct potential data file (z, V(z) columns).

    Returns
    -------
    v_vacuum : float
        Vacuum level in eV.
    z_data : array
        z-coordinates (Angstrom).
    v_data : array
        Potential values (eV).
    """
    # Try to read from average.x output
    z_data = []
    v_data = []

    with open(avg_output_file) as f:
        lines = f.readlines()

    # average.x output format: columns of z (Bohr), V(z) (Ry)
    for line in lines:
        parts = line.strip().split()
        if len(parts) >= 2:
            try:
                z = float(parts[0])
                v = float(parts[1])
                z_data.append(z)
                v_data.append(v)
            except ValueError:
                continue

    if not z_data:
        print("WARNING: Could not parse potential data")
        return None, None, None

    z_data = np.array(z_data)
    v_data = np.array(v_data)

    # Convert units: Bohr -> Angstrom, Ry -> eV
    z_ang = z_data * 0.529177
    v_ev = v_data * 13.605693123

    # Find vacuum region: where potential is roughly constant
    # Take the average of the top 10% and bottom 10% of z-values
    z_max = np.max(z_ang)
    high_mask = z_ang > 0.8 * z_max
    low_mask = z_ang < 0.1 * z_max

    v_vacuum_top = np.mean(v_ev[high_mask]) if np.any(high_mask) else None
    v_vacuum_bot = np.mean(v_ev[low_mask]) if np.any(low_mask) else None

    # Use the more stable (flatter) region
    if v_vacuum_top is not None and v_vacuum_bot is not None:
        std_top = np.std(v_ev[high_mask])
        std_bot = np.std(v_ev[low_mask])
        if std_top < std_bot:
            v_vacuum = v_vacuum_top
        else:
            v_vacuum = v_vacuum_bot
        # Or average both sides for centered slab
        v_vacuum = (v_vacuum_top + v_vacuum_bot) / 2
    elif v_vacuum_top is not None:
        v_vacuum = v_vacuum_top
    else:
        v_vacuum = v_vacuum_bot

    return v_vacuum, z_ang, v_ev


# ============================================================
# 3. Compute and display results
# ============================================================
band_data = extract_band_edges(f"{PREFIX}_nscf.out")
print("Band edges from NSCF:")
for key, val in band_data.items():
    if val is not None:
        print(f"  {key}: {val:.4f}")

v_vacuum, z, v = extract_vacuum_level(f"{PREFIX}_average.out")
if v_vacuum is not None:
    print(f"\nVacuum level: {v_vacuum:.4f} eV")

    # Compute absolute positions
    e_fermi = band_data["e_fermi_eV"]
    e_vbm = band_data["e_vbm_eV"]
    e_cbm = band_data["e_cbm_eV"]

    # Work function: phi = V_vacuum - E_Fermi
    work_function = v_vacuum - e_fermi if e_fermi is not None else None

    # Ionization potential: IP = V_vacuum - E_VBM
    ionization_potential = v_vacuum - e_vbm if e_vbm is not None else None

    # Electron affinity: EA = V_vacuum - E_CBM
    electron_affinity = v_vacuum - e_cbm if e_cbm is not None else None

    print(f"\n{'='*50}")
    print(f"BAND EDGE ALIGNMENT RESULTS")
    print(f"{'='*50}")
    if work_function is not None:
        print(f"Work function (phi):           {work_function:.4f} eV")
    if ionization_potential is not None:
        print(f"Ionization potential (IP):      {ionization_potential:.4f} eV")
    if electron_affinity is not None:
        print(f"Electron affinity (EA):         {electron_affinity:.4f} eV")
    if band_data["band_gap_eV"] is not None:
        print(f"Band gap:                       {band_data['band_gap_eV']:.4f} eV")
    print(f"IP - EA = band gap check:       "
          f"{(ionization_potential - electron_affinity):.4f} eV")

    # ============================================================
    # 4. Check for water splitting
    # ============================================================
    # Water redox potentials vs vacuum (at pH=0):
    # H+/H2:  -4.44 eV (reduction, CBM must be above this)
    # O2/H2O: -5.67 eV (oxidation, VBM must be below this)
    H_redox = -4.44  # eV vs vacuum
    O_redox = -5.67  # eV vs vacuum

    # Band edges relative to vacuum (negative values)
    vbm_abs = -ionization_potential if ionization_potential else None
    cbm_abs = -electron_affinity if electron_affinity else None

    if vbm_abs is not None and cbm_abs is not None:
        print(f"\nWater Splitting Assessment (pH=0):")
        print(f"  VBM (abs):  {vbm_abs:.4f} eV vs vacuum")
        print(f"  CBM (abs):  {cbm_abs:.4f} eV vs vacuum")
        print(f"  H+/H2:     {H_redox:.4f} eV vs vacuum")
        print(f"  O2/H2O:    {O_redox:.4f} eV vs vacuum")

        can_reduce = cbm_abs > H_redox  # CBM above H+/H2
        can_oxidize = vbm_abs < O_redox  # VBM below O2/H2O

        print(f"  Can reduce H+?   {'YES' if can_reduce else 'NO'} "
              f"(CBM {'>' if can_reduce else '<'} H+/H2)")
        print(f"  Can oxidize H2O? {'YES' if can_oxidize else 'NO'} "
              f"(VBM {'<' if can_oxidize else '>'} O2/H2O)")

        if can_reduce and can_oxidize:
            print(f"  ==> Suitable for overall water splitting!")
        else:
            print(f"  ==> NOT suitable for overall water splitting at pH=0")

    # ============================================================
    # 5. Plot potential profile and band alignment
    # ============================================================
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Left: Planar-averaged potential
    ax1 = axes[0]
    ax1.plot(z, v, "b-", linewidth=1.5)
    if e_fermi is not None:
        ax1.axhline(e_fermi, color="red", linestyle="--", linewidth=1.5,
                     label=f"$E_F$ = {e_fermi:.2f} eV")
    if v_vacuum is not None:
        ax1.axhline(v_vacuum, color="green", linestyle="-.", linewidth=1.5,
                     label=f"$V_{{vac}}$ = {v_vacuum:.2f} eV")
    ax1.set_xlabel("z (A)", fontsize=12)
    ax1.set_ylabel("Potential (eV)", fontsize=12)
    ax1.set_title("Planar-Averaged Electrostatic Potential", fontsize=13)
    ax1.legend(fontsize=10)
    ax1.grid(alpha=0.3)

    # Right: Band alignment diagram
    ax2 = axes[1]

    if vbm_abs is not None and cbm_abs is not None:
        # Draw band edges as horizontal bars
        bar_width = 0.4
        bar_x = 0.5

        # VBM
        ax2.barh(0, bar_width, left=bar_x - bar_width/2,
                 height=0.05, color="blue", alpha=0.7)
        ax2.text(bar_x + bar_width/2 + 0.1, vbm_abs,
                 f"VBM = {vbm_abs:.2f} eV", va="center", fontsize=10)

        # CBM
        ax2.barh(0, bar_width, left=bar_x - bar_width/2,
                 height=0.05, color="red", alpha=0.7)
        ax2.text(bar_x + bar_width/2 + 0.1, cbm_abs,
                 f"CBM = {cbm_abs:.2f} eV", va="center", fontsize=10)

        # Band gap shading
        ax2.fill_between([bar_x - bar_width/2, bar_x + bar_width/2],
                         vbm_abs, cbm_abs, alpha=0.2, color="gray")

        # Water redox levels
        ax2.axhline(H_redox, color="blue", linestyle="--", linewidth=1.5,
                     label=f"H$^+$/H$_2$ = {H_redox:.2f} eV")
        ax2.axhline(O_redox, color="red", linestyle="--", linewidth=1.5,
                     label=f"O$_2$/H$_2$O = {O_redox:.2f} eV")

        # Vacuum level
        ax2.axhline(0, color="black", linestyle="-", linewidth=1,
                     label="Vacuum level")

        ax2.set_ylabel("Energy vs vacuum (eV)", fontsize=12)
        ax2.set_title("Band Edge Alignment", fontsize=13)
        ax2.set_xlim(0, 2)
        ax2.set_ylim(min(vbm_abs, O_redox) - 1, 1)
        ax2.legend(fontsize=9, loc="upper right")
        ax2.set_xticks([bar_x])
        ax2.set_xticklabels(["MoS2"])
        ax2.grid(axis="y", alpha=0.3)

    fig.tight_layout()
    fig.savefig("band_edge_alignment.png", dpi=150)
    print("\nSaved band_edge_alignment.png")

    # Save results
    results = {
        "system": "MoS2 monolayer",
        "e_fermi_eV": e_fermi,
        "e_vbm_eV": e_vbm,
        "e_cbm_eV": e_cbm,
        "band_gap_eV": band_data["band_gap_eV"],
        "v_vacuum_eV": float(v_vacuum),
        "work_function_eV": float(work_function) if work_function else None,
        "ionization_potential_eV": float(ionization_potential) if ionization_potential else None,
        "electron_affinity_eV": float(electron_affinity) if electron_affinity else None,
    }

    with open("band_edge_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print("Saved band_edge_results.json")
else:
    print("ERROR: Could not extract vacuum level. Check pp.x and average.x output.")
```

### Method B: VASP Workflow (Future)

```
VASP band edge alignment workflow (requires VASP license):

1. Relax 2D structure:
   ISIF = 4 (relax ions + in-plane cell, fix c)
   EDIFF = 1E-6, EDIFFG = -0.01

2. SCF calculation:
   LCHARG = .TRUE., LVHAR = .TRUE. (write LOCPOT for potential)
   ISMEAR = 0, SIGMA = 0.05
   KPOINTS: dense in-plane (e.g., 12x12x1)

3. Post-process with VASPKIT 927:
   - Reads LOCPOT to get planar-averaged potential
   - Computes vacuum level, work function, IP, EA
   - Generates PLANAR_AVERAGE.dat and BAND_ALIGNMENT.dat

4. Alternative: manual extraction
   - Use p4vasp or VASPKIT to get V(z) from LOCPOT
   - V_vacuum = average of V(z) in the flat vacuum region
   - phi = V_vacuum - E_Fermi (from OUTCAR/EIGENVAL)
   - IP = V_vacuum - E_VBM
   - EA = V_vacuum - E_CBM

Note: VASP's LVHAR = .TRUE. writes the ionic + Hartree potential to LOCPOT.
This is the electrostatic potential needed for vacuum alignment.
```

## Key Parameters

| Parameter | Recommended Value | Notes |
|---|---|---|
| Vacuum thickness | >= 20 A | Potential must be flat in vacuum region |
| K-points (in-plane) | 12x12x1 or denser | Dense k-grid for accurate band edges |
| K-points (perpendicular) | 1 | Only 1 k-point along vacuum direction |
| `assume_isolated = '2D'` | Recommended | 2D Coulomb cutoff in QE; reduces required vacuum |
| `plot_num = 11` (pp.x) | Electrostatic potential | Use 11 for V_Hartree + V_ion; not 1 (total V with xc) |
| nbnd (NSCF) | Include empty bands | Need CBM for electron affinity |
| ecutwfc | 50-80 Ry | Converge for your pseudopotentials |

## Interpreting Results

1. **Work function (phi)**: Typically 4-5 eV for most metals, 4-7 eV for 2D semiconductors. MoS2 monolayer: ~5.1 eV.
2. **Ionization potential (IP)**: Energy needed to remove an electron from the VBM to vacuum. Larger IP = harder to oxidize.
3. **Electron affinity (EA)**: Energy gained by adding an electron to the CBM from vacuum. Larger EA = easier to reduce.
4. **Band gap**: IP - EA = band gap (consistency check).
5. **Water splitting criteria** (pH=0):
   - CBM must be above H+/H2 level (-4.44 eV vs vacuum): can reduce protons
   - VBM must be below O2/H2O level (-5.67 eV vs vacuum): can oxidize water
   - pH correction: levels shift by 0.059 * pH eV
6. **PBE band gap underestimation**: PBE underestimates band gaps by ~30-50%. HSE06 or GW corrections are needed for accurate band edges.

## Common Issues

| Issue | Solution |
|---|---|
| Potential is not flat in vacuum | Increase vacuum thickness. Use dipole correction for asymmetric slabs. |
| VBM/CBM not reported in QE output | Add nbnd to include empty bands. Use `verbosity = 'high'` in NSCF. |
| average.x gives wrong result | Check input: nfile=1, file is the correct potential file, direction=3 (z). |
| Work function is negative | Sanity check: V_vacuum should be higher than E_Fermi. Check potential sign convention. |
| Band edges shift with vacuum size | Vacuum is too thin. Increase to >= 25 A or use 2D Coulomb cutoff. |
| pp.x crash with `plot_num = 11` | Try `plot_num = 0` (total potential) or check if the SCF completed successfully. |
| PBE band gap too small | Expected. Use HSE06 (@QE: `input_dft = 'HSE'`) for more accurate gaps. |
| Different IP/EA for two sides of slab | Slab is asymmetric. Use dipole correction or a symmetric slab. |
