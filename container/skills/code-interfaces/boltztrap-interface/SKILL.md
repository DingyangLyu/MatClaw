# BoltzTraP2 Interface for Transport Properties

## When to Use

- You need thermoelectric transport coefficients: Seebeck coefficient, electrical conductivity, electronic thermal conductivity.
- You want to compute the power factor or estimate the thermoelectric figure of merit ZT.
- You need transport properties as a function of temperature and carrier concentration (doping).
- You have a completed QE or VASP band structure calculation and want to post-process it for transport.
- Equivalent to VASPKIT function 781 (BoltzTraP interface).

## Method Selection

| Source | Method | Notes |
|---|---|---|
| QE NSCF output | Method A: BoltzTraP2 native QE reader | Reads `prefix.save/data-file-schema.xml` directly. Most reliable for QE. |
| QE NSCF output | Method B: pymatgen intermediate | Convert QE bands to pymatgen `BandStructure`, then feed to BoltzTraP2. More steps but flexible. |
| VASP NSCF output | Method C: BoltzTraP2 via pymatgen/vasprun.xml | Use pymatgen's `BoltzTraP2` interface with `vasprun.xml`. Input generation only (VASP not in container). |
| MACE/MLIP + QE | Method D: MLIP screening + QE transport | Use MLIP for structure relaxation, then QE for electronic structure, then BoltzTraP2. |

## Prerequisites

```bash
# Install BoltzTraP2 (not pre-installed)
pip install BoltzTraP2
```

- Pre-installed: `pymatgen`, `ase`, `numpy`, `scipy`, `matplotlib`.
- QE binaries: `pw.x` in `/opt/qe/bin/`.
- A **very dense** k-grid NSCF calculation (40x40x40 or denser) with `nosym = .true.`.
- For VASP: `vasprun.xml` from a dense NSCF run (input preparation only).

---

## Detailed Steps

### Method A: QE to BoltzTraP2 (Native Interface)

Complete workflow: SCF, dense NSCF, BoltzTraP2 interpolation, transport calculation, plotting.

#### Step 1: Prepare and Run QE SCF + Dense NSCF

```python
#!/usr/bin/env python3
"""
Generate QE SCF and dense NSCF input files for BoltzTraP2 transport calculation.
"""
import os
import numpy as np
from pymatgen.core.structure import Structure, Lattice

WORK_DIR = "/tmp/transport"
PSEUDO_DIR = os.path.join(WORK_DIR, "pseudo")
os.makedirs(PSEUDO_DIR, exist_ok=True)

# ============================================================
# CONFIGURATION -- Edit for your system
# ============================================================

# Example: Silicon (diamond)
structure = Structure(
    lattice=Lattice.from_parameters(5.431, 5.431, 5.431, 90, 90, 90),
    species=["Si"] * 8,
    coords=[
        [0.00, 0.00, 0.00], [0.50, 0.50, 0.00],
        [0.50, 0.00, 0.50], [0.00, 0.50, 0.50],
        [0.25, 0.25, 0.25], [0.75, 0.75, 0.25],
        [0.75, 0.25, 0.75], [0.25, 0.75, 0.75],
    ],
)

PREFIX = "transport"
ECUTWFC = 60.0
ECUTRHO = 480.0
K_SCF = (8, 8, 8)          # coarse grid for SCF
K_NSCF = (40, 40, 40)      # dense grid for BoltzTraP2
NBND = 20                   # number of bands (>= 2x occupied)

elements = list(set(str(sp) for sp in structure.species))
pseudo_map = {el: f"{el}.pbe-n-kjpaw_psl.1.0.0.UPF" for el in elements}

# ============================================================
# HELPER: Write QE input file
# ============================================================

def write_qe_input(filename, calc_type, kgrid, nosym=False):
    """Write a QE pw.x input file."""
    lines = []
    lines.append("&CONTROL")
    lines.append(f"    calculation = '{calc_type}'")
    lines.append(f"    prefix      = '{PREFIX}'")
    lines.append(f"    outdir      = './tmp'")
    lines.append(f"    pseudo_dir  = '{PSEUDO_DIR}'")
    lines.append(f"    tprnfor     = .true.")
    lines.append(f"    tstress     = .true.")
    lines.append(f"    verbosity   = 'high'")
    lines.append("/\n")

    lines.append("&SYSTEM")
    lines.append(f"    ibrav       = 0")
    lines.append(f"    nat         = {len(structure)}")
    lines.append(f"    ntyp        = {len(elements)}")
    lines.append(f"    ecutwfc     = {ECUTWFC}")
    lines.append(f"    ecutrho     = {ECUTRHO}")
    lines.append(f"    occupations = 'smearing'")
    lines.append(f"    smearing    = 'gaussian'")
    lines.append(f"    degauss     = 0.005")
    lines.append(f"    nbnd        = {NBND}")
    if nosym:
        lines.append(f"    nosym       = .true.")
    lines.append("/\n")

    lines.append("&ELECTRONS")
    lines.append(f"    conv_thr    = 1.0d-10")
    lines.append(f"    mixing_beta = 0.7")
    lines.append("/\n")

    lines.append("ATOMIC_SPECIES")
    for el in sorted(elements):
        from pymatgen.core.periodic_table import Element
        mass = Element(el).atomic_mass
        lines.append(f"  {el:4s} {float(mass):10.4f}  {pseudo_map[el]}")
    lines.append("")

    lines.append("CELL_PARAMETERS angstrom")
    for row in structure.lattice.matrix:
        lines.append(f"  {row[0]:16.10f} {row[1]:16.10f} {row[2]:16.10f}")
    lines.append("")

    lines.append("ATOMIC_POSITIONS crystal")
    for site in structure:
        fc = site.frac_coords
        lines.append(f"  {str(site.specie):4s} {fc[0]:16.10f} {fc[1]:16.10f} {fc[2]:16.10f}")
    lines.append("")

    lines.append("K_POINTS automatic")
    lines.append(f"  {kgrid[0]} {kgrid[1]} {kgrid[2]}  0 0 0")

    filepath = os.path.join(WORK_DIR, filename)
    with open(filepath, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"Written: {filepath}")


# ============================================================
# GENERATE INPUT FILES
# ============================================================

write_qe_input("scf.in", "scf", K_SCF, nosym=False)
write_qe_input("nscf.in", "nscf", K_NSCF, nosym=True)

# Print run commands
nprocs = os.cpu_count() or 4
print(f"\n=== Run these commands ===")
print(f"cd {WORK_DIR}")
print(f"# Step 1: SCF")
print(f"mpirun --allow-run-as-root -np {nprocs} pw.x -npool 2 < scf.in > scf.out 2>&1")
print(f"# Step 2: NSCF on dense grid (nosym=.true.)")
print(f"mpirun --allow-run-as-root -np {nprocs} pw.x -npool {nprocs} < nscf.in > nscf.out 2>&1")
```

#### Step 2: Run BoltzTraP2 and Compute Transport

```python
#!/usr/bin/env python3
"""
Run BoltzTraP2 on QE NSCF output and compute transport properties.
Produces Seebeck, conductivity, thermal conductivity vs T and doping.
"""
import os
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ============================================================
# CONFIGURATION
# ============================================================

WORK_DIR = "/tmp/transport"
PREFIX = "transport"
OUTDIR = os.path.join(WORK_DIR, "tmp")

# Temperature range
T_MIN, T_MAX, N_T = 100, 1000, 19  # 100 to 1000 K

# Doping levels for plots (cm^-3)
DOPING_LEVELS = [1e18, 5e18, 1e19, 5e19, 1e20]

# Relaxation time (for absolute sigma and kappa_e)
TAU = 1e-14  # seconds (typical for semiconductors)

# Lattice thermal conductivity (for ZT estimate)
KAPPA_LAT = 10.0  # W/(m*K) -- from phonon calc or experiment

# ============================================================
# INSTALL BOLTZTRAP2 IF NEEDED
# ============================================================

try:
    import BoltzTraP2
    import BoltzTraP2.dft
    import BoltzTraP2.bandlib
    import BoltzTraP2.fite
except ImportError:
    print("Installing BoltzTraP2...")
    os.system("pip install BoltzTraP2")
    import BoltzTraP2
    import BoltzTraP2.dft
    import BoltzTraP2.bandlib
    import BoltzTraP2.fite

# ============================================================
# LOAD QE DATA
# ============================================================

data_dir = os.path.join(OUTDIR, f"{PREFIX}.save")
if not os.path.isdir(data_dir):
    print(f"ERROR: {data_dir} not found.")
    print("Make sure the NSCF calculation completed successfully.")
    sys.exit(1)

print(f"Loading QE data from: {data_dir}")
data = BoltzTraP2.dft.DFTData(data_dir)

print(f"  K-points: {data.kpoints.shape[0]}")
print(f"  Bands: {data.ebands.shape[1]}")
print(f"  Fermi energy: {data.fermi:.6f} Ha")
print(f"  Volume: {data.atoms.get_volume():.4f} A^3")

# ============================================================
# BOLTZTRAP2 INTERPOLATION
# ============================================================

print("\nRunning BoltzTraP2 interpolation...")
equivalences = BoltzTraP2.fite.equivalences(data, magmom=None)
coeffs = BoltzTraP2.fite.fitde3D(data, equivalences)
print(f"  Star functions: {len(equivalences)}")

# ============================================================
# COMPUTE TRANSPORT COEFFICIENTS
# ============================================================

print("\nComputing transport coefficients...")

temperatures = np.linspace(T_MIN, T_MAX, N_T)

# Chemical potential scan window (+/- 1.5 eV around Fermi level)
eV_to_Ha = 1.0 / 27.2114
mu_min = data.fermi - 1.5 * eV_to_Ha
mu_max = data.fermi + 1.5 * eV_to_Ha
n_mu = 2000
mu_array = np.linspace(mu_min, mu_max, n_mu)

# Compute DOS and transport DOS
epsilon, dos, vvdos, cdos = BoltzTraP2.fite.BTPDOS(
    data.ebands, None, data.kpoints, temperatures, mu_array,
    data.atoms.get_volume() * 1e-30,
    equivalences, coeffs
)

# Compute Onsager coefficients
sigma, seebeck, kappa, hall = BoltzTraP2.bandlib.calc_Onsager_coefficients(
    epsilon, dos, vvdos, temperatures, mu_array,
    data.atoms.get_volume() * 1e-30
)

# Carrier concentration
carrier_conc = BoltzTraP2.bandlib.calc_N(
    epsilon, dos, temperatures, mu_array,
    data.atoms.get_volume() * 1e-30
)

print("  Transport coefficients computed.")
print(f"  sigma shape: {sigma.shape}")       # (n_T, n_mu, 3, 3)
print(f"  seebeck shape: {seebeck.shape}")   # (n_T, n_mu, 3, 3)

# ============================================================
# EXTRACT AT SPECIFIC DOPING LEVELS
# ============================================================

def extract_at_doping(target_n_cm3, carrier_type="n"):
    """Extract isotropic transport at a target carrier concentration."""
    target_n_m3 = target_n_cm3 * 1e6
    result = {"T": temperatures, "S": [], "sigma_tau": [], "kappa_tau": [], "PF_tau": []}

    for iT in range(len(temperatures)):
        conc = carrier_conc[iT, :]
        if carrier_type == "n":
            mask = mu_array >= data.fermi
        else:
            mask = mu_array <= data.fermi

        diff = np.abs(np.abs(conc) - target_n_m3)
        diff_masked = np.where(mask, diff, np.inf)
        idx = np.argmin(diff_masked)

        S = np.trace(seebeck[iT, idx]) / 3.0
        sig = np.trace(sigma[iT, idx]) / 3.0
        kap = np.trace(kappa[iT, idx]) / 3.0
        PF = S**2 * sig

        result["S"].append(S)
        result["sigma_tau"].append(sig)
        result["kappa_tau"].append(kap)
        result["PF_tau"].append(PF)

    for key in ["S", "sigma_tau", "kappa_tau", "PF_tau"]:
        result[key] = np.array(result[key])
    return result

# ============================================================
# PRINT RESULTS TABLE
# ============================================================

print("\n" + "=" * 80)
print("Transport Properties at 300 K")
print("=" * 80)

iT_300 = np.argmin(np.abs(temperatures - 300))
T_actual = temperatures[iT_300]
print(f"Temperature: {T_actual:.0f} K\n")

for ct in ["n", "p"]:
    print(f"  {ct.upper()}-type doping:")
    print(f"  {'n (cm^-3)':>12} {'S (uV/K)':>10} {'sigma/tau':>14} {'PF/tau':>14}")
    print("  " + "-" * 54)
    for n_dop in DOPING_LEVELS:
        ext = extract_at_doping(n_dop, ct)
        S_uV = ext["S"][iT_300] * 1e6
        sig = ext["sigma_tau"][iT_300]
        PF = ext["PF_tau"][iT_300]
        print(f"  {n_dop:>12.1e} {S_uV:>10.1f} {sig:>14.3e} {PF:>14.3e}")
    print()

# ============================================================
# PLOTTING
# ============================================================

os.chdir(WORK_DIR)

for ct in ["n", "p"]:
    # --- Seebeck vs Temperature ---
    fig, ax = plt.subplots(figsize=(8, 5))
    colors = plt.cm.viridis(np.linspace(0.15, 0.85, len(DOPING_LEVELS)))

    for i, n in enumerate(DOPING_LEVELS):
        ext = extract_at_doping(n, ct)
        S_uV = ext["S"] * 1e6
        ax.plot(ext["T"], S_uV, "-o", color=colors[i], linewidth=2,
                markersize=3, label=f"n = {n:.0e} cm$^{{-3}}$")

    ax.set_xlabel("Temperature (K)", fontsize=13)
    ax.set_ylabel(r"Seebeck coefficient ($\mu$V/K)", fontsize=13)
    ax.set_title(f"Seebeck Coefficient ({ct.upper()}-type)", fontsize=14)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.axhline(y=0, color="k", linewidth=0.5)
    plt.tight_layout()
    plt.savefig(f"seebeck_vs_T_{ct}type.png", dpi=150, bbox_inches="tight")
    plt.close()

    # --- Conductivity vs Temperature ---
    fig, ax = plt.subplots(figsize=(8, 5))
    for i, n in enumerate(DOPING_LEVELS):
        ext = extract_at_doping(n, ct)
        ax.plot(ext["T"], ext["sigma_tau"], "-o", color=colors[i], linewidth=2,
                markersize=3, label=f"n = {n:.0e} cm$^{{-3}}$")

    ax.set_xlabel("Temperature (K)", fontsize=13)
    ax.set_ylabel(r"$\sigma/\tau$ (1/$\Omega\cdot$m$\cdot$s)", fontsize=13)
    ax.set_title(f"Electrical Conductivity / tau ({ct.upper()}-type)", fontsize=14)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.set_yscale("log")
    plt.tight_layout()
    plt.savefig(f"sigma_vs_T_{ct}type.png", dpi=150, bbox_inches="tight")
    plt.close()

    # --- Power Factor vs Temperature ---
    fig, ax = plt.subplots(figsize=(8, 5))
    for i, n in enumerate(DOPING_LEVELS):
        ext = extract_at_doping(n, ct)
        ax.plot(ext["T"], ext["PF_tau"], "-o", color=colors[i], linewidth=2,
                markersize=3, label=f"n = {n:.0e} cm$^{{-3}}$")

    ax.set_xlabel("Temperature (K)", fontsize=13)
    ax.set_ylabel(r"$S^2 \sigma / \tau$ (W/m$\cdot$K$^2\cdot$s)", fontsize=13)
    ax.set_title(f"Power Factor / tau ({ct.upper()}-type)", fontsize=14)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"power_factor_vs_T_{ct}type.png", dpi=150, bbox_inches="tight")
    plt.close()

    # --- ZT Estimate ---
    fig, ax = plt.subplots(figsize=(8, 5))
    for i, n in enumerate(DOPING_LEVELS):
        ext = extract_at_doping(n, ct)
        T = ext["T"]
        sig_abs = ext["sigma_tau"] * TAU
        kappa_e = ext["kappa_tau"] * TAU
        S = ext["S"]
        kappa_total = kappa_e + KAPPA_LAT
        ZT = S**2 * sig_abs * T / kappa_total

        ax.plot(T, ZT, "-o", color=colors[i], linewidth=2,
                markersize=3, label=f"n = {n:.0e} cm$^{{-3}}$")

    ax.set_xlabel("Temperature (K)", fontsize=13)
    ax.set_ylabel("ZT", fontsize=13)
    ax.set_title(f"ZT Estimate ({ct.upper()}-type)\n"
                 f"tau = {TAU:.0e} s, kappa_lat = {KAPPA_LAT} W/(m*K)", fontsize=12)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(bottom=0)
    plt.tight_layout()
    plt.savefig(f"ZT_vs_T_{ct}type.png", dpi=150, bbox_inches="tight")
    plt.close()

# --- Seebeck vs Doping at fixed T ---
fig, ax = plt.subplots(figsize=(8, 5))
T_targets = [300, 500, 700, 900]
colors_T = plt.cm.coolwarm(np.linspace(0.1, 0.9, len(T_targets)))
doping_scan = np.logspace(17, 21, 60)

for j, T_target in enumerate(T_targets):
    iT = np.argmin(np.abs(temperatures - T_target))
    S_vals = []
    for n in doping_scan:
        ext = extract_at_doping(n, "n")
        S_vals.append(ext["S"][iT] * 1e6)
    ax.semilogx(doping_scan, S_vals, "-", color=colors_T[j], linewidth=2,
                label=f"T = {temperatures[iT]:.0f} K")

ax.set_xlabel(r"Carrier concentration (cm$^{-3}$)", fontsize=13)
ax.set_ylabel(r"Seebeck coefficient ($\mu$V/K)", fontsize=13)
ax.set_title("Seebeck vs Doping (n-type)", fontsize=14)
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("seebeck_vs_doping.png", dpi=150, bbox_inches="tight")
plt.close()

print(f"\nPlots saved in {WORK_DIR}/")
print("  seebeck_vs_T_ntype.png, seebeck_vs_T_ptype.png")
print("  sigma_vs_T_ntype.png, sigma_vs_T_ptype.png")
print("  power_factor_vs_T_ntype.png, power_factor_vs_T_ptype.png")
print("  ZT_vs_T_ntype.png, ZT_vs_T_ptype.png")
print("  seebeck_vs_doping.png")
```

### Method C: VASP to BoltzTraP2 (Input Preparation)

Generate VASP input files for a dense NSCF calculation suitable for BoltzTraP2. After running VASP externally, parse results with BoltzTraP2.

```python
#!/usr/bin/env python3
"""
Generate VASP input files for BoltzTraP2 transport calculation.
Equivalent to VASPKIT function 781 setup.

After running VASP externally, use BoltzTraP2 to process the results.
"""
import os
from pymatgen.core.structure import Structure
from pymatgen.io.vasp import Poscar, Incar, Kpoints

STRUCTURE_FILE = "POSCAR"
OUTPUT_DIR = "./vasp_transport"
os.makedirs(OUTPUT_DIR, exist_ok=True)

structure = Structure.from_file(STRUCTURE_FILE)

# ============================================================
# Step 1: SCF INCAR
# ============================================================

scf_dir = os.path.join(OUTPUT_DIR, "scf")
os.makedirs(scf_dir, exist_ok=True)

scf_incar = Incar({
    "SYSTEM": f"{structure.composition.reduced_formula} SCF for transport",
    "ENCUT": 520,
    "PREC": "Accurate",
    "EDIFF": 1e-8,
    "NELM": 200,
    "ISMEAR": 0,
    "SIGMA": 0.05,
    "LWAVE": True,     # need WAVECAR for NSCF
    "LCHARG": True,    # need CHGCAR for NSCF
    "ALGO": "Normal",
    "NSW": 0,
    "IBRION": -1,
    "LORBIT": 11,
    "NCORE": 4,
})
scf_incar.write_file(os.path.join(scf_dir, "INCAR"))

scf_kpoints = Kpoints.automatic_density(structure, kppa=2000, force_gamma=True)
scf_kpoints.write_file(os.path.join(scf_dir, "KPOINTS"))

Poscar(structure).write_file(os.path.join(scf_dir, "POSCAR"))
print(f"SCF inputs written to {scf_dir}/")

# ============================================================
# Step 2: Dense NSCF INCAR for BoltzTraP2
# ============================================================

nscf_dir = os.path.join(OUTPUT_DIR, "nscf")
os.makedirs(nscf_dir, exist_ok=True)

nscf_incar = Incar({
    "SYSTEM": f"{structure.composition.reduced_formula} NSCF dense for BoltzTraP2",
    "ENCUT": 520,
    "PREC": "Accurate",
    "EDIFF": 1e-8,
    "NELM": 200,
    "ISMEAR": 0,
    "SIGMA": 0.05,
    "ICHARG": 11,       # read CHGCAR
    "LWAVE": False,
    "LCHARG": False,
    "ALGO": "Normal",
    "NSW": 0,
    "IBRION": -1,
    "LORBIT": 11,
    "NBANDS": max(2 * len(structure), 20),  # enough empty bands
    "ISYM": -1,          # no symmetry -- BoltzTraP2 needs full BZ
    "NCORE": 4,
})
nscf_incar.write_file(os.path.join(nscf_dir, "INCAR"))

# Dense k-grid: 40x40x40 for small cells, scale down for large cells
recip_abc = structure.lattice.reciprocal_lattice.abc
k_dense = tuple(max(20, int(round(40 * r / max(recip_abc)))) for r in recip_abc)
nscf_kpoints = Kpoints.gamma_automatic(k_dense)
nscf_kpoints.write_file(os.path.join(nscf_dir, "KPOINTS"))

Poscar(structure).write_file(os.path.join(nscf_dir, "POSCAR"))
print(f"NSCF inputs written to {nscf_dir}/")
print(f"  K-grid: {k_dense[0]}x{k_dense[1]}x{k_dense[2]}")
print(f"  NBANDS: {nscf_incar['NBANDS']}")
print(f"  ISYM: -1 (no symmetry -- required for BoltzTraP2)")

# ============================================================
# Step 3: BoltzTraP2 post-processing script (after VASP run)
# ============================================================

bt2_script = '''#!/usr/bin/env python3
"""
Run BoltzTraP2 on VASP vasprun.xml.
Execute after VASP NSCF calculation completes.
"""
import numpy as np
import BoltzTraP2
import BoltzTraP2.dft
import BoltzTraP2.bandlib
import BoltzTraP2.fite

# Load VASP data from vasprun.xml
data = BoltzTraP2.dft.DFTData("./nscf")  # directory containing vasprun.xml

print(f"K-points: {data.kpoints.shape[0]}")
print(f"Bands: {data.ebands.shape[1]}")
print(f"Fermi: {data.fermi:.6f} Ha")

# Interpolation
equivalences = BoltzTraP2.fite.equivalences(data, magmom=None)
coeffs = BoltzTraP2.fite.fitde3D(data, equivalences)

# Transport
temperatures = np.linspace(100, 1000, 19)
eV_to_Ha = 1.0 / 27.2114
mu_array = np.linspace(data.fermi - 1.5*eV_to_Ha, data.fermi + 1.5*eV_to_Ha, 2000)

epsilon, dos, vvdos, cdos = BoltzTraP2.fite.BTPDOS(
    data.ebands, None, data.kpoints, temperatures, mu_array,
    data.atoms.get_volume() * 1e-30, equivalences, coeffs
)

sigma, seebeck, kappa, hall = BoltzTraP2.bandlib.calc_Onsager_coefficients(
    epsilon, dos, vvdos, temperatures, mu_array,
    data.atoms.get_volume() * 1e-30
)

print("Transport coefficients computed. Shape:", sigma.shape)
'''

with open(os.path.join(OUTPUT_DIR, "run_boltztrap2.py"), "w") as f:
    f.write(bt2_script)
print(f"BoltzTraP2 post-processing script: {OUTPUT_DIR}/run_boltztrap2.py")
```

### Complete Shell Workflow (QE)

```bash
#!/bin/bash
# transport_workflow.sh -- Complete QE + BoltzTraP2 transport workflow
set -e

WORK_DIR=/tmp/transport
cd $WORK_DIR
NP=$(nproc)

echo "=== Electronic Transport Workflow ==="

# Step 1: Download pseudopotentials
echo "--- Downloading pseudopotentials ---"
python3 -c "
import urllib.request, os
os.makedirs('$WORK_DIR/pseudo', exist_ok=True)
base = 'https://pseudopotentials.quantum-espresso.org/upf_files/'
pp = 'Si.pbe-n-kjpaw_psl.1.0.0.UPF'
if not os.path.exists(f'$WORK_DIR/pseudo/{pp}'):
    urllib.request.urlretrieve(base + pp, f'$WORK_DIR/pseudo/{pp}')
    print(f'Downloaded {pp}')
else:
    print(f'{pp} exists')
"

# Step 2: SCF
echo "--- SCF ---"
mpirun --allow-run-as-root -np $NP pw.x -npool 2 < scf.in > scf.out 2>&1
grep "convergence has been achieved" scf.out && echo "SCF converged."

# Step 3: Dense NSCF
echo "--- NSCF (dense k-grid) ---"
mpirun --allow-run-as-root -np $NP pw.x -npool $NP < nscf.in > nscf.out 2>&1
echo "NSCF completed."

# Step 4: Install BoltzTraP2
pip install BoltzTraP2 2>/dev/null

# Step 5: Run transport analysis
echo "--- BoltzTraP2 transport ---"
python3 bolztrap2_transport.py

echo "=== Workflow Complete ==="
```

## Key Parameters

| Parameter | Value | Notes |
|---|---|---|
| K-grid (NSCF) | 40x40x40 minimum | Most critical parameter. Seebeck converges at ~30x30x30; sigma needs denser. |
| `nosym` (QE) / `ISYM=-1` (VASP) | MANDATORY | BoltzTraP2 needs the full BZ, not symmetry-reduced. |
| `nbnd` / `NBANDS` | >= 2x occupied | Include enough empty bands above the Fermi level. |
| Temperature range | 100-1000 K | Use 10-20 points for smooth curves. |
| Doping range | 1e17 - 1e21 cm^-3 | Optimal thermoelectric: typically 1e19-1e20 cm^-3. |
| Chemical potential window | +/- 1.5 eV around E_F | Covers the transport-relevant energy range. |
| `conv_thr` / `EDIFF` | 1e-10 Ry / 1e-8 eV | Tight convergence for accurate band energies. |
| Relaxation time (tau) | 1e-15 to 1e-13 s | External input: from experiment, e-ph coupling, or modeling. |
| Lattice thermal conductivity | System-dependent | From phonon calculation or experiment. Needed for ZT. |

## Interpreting Results

### Seebeck coefficient (S)
- **Sign**: Negative for n-type, positive for p-type.
- **Magnitude**: Good thermoelectrics have |S| > 100-200 uV/K. Metals: |S| < 50 uV/K.
- **BoltzTraP2 gives the exact S** (relaxation time cancels in the ratio).
- S decreases with increasing carrier concentration (tradeoff with conductivity).

### Electrical conductivity (sigma/tau)
- BoltzTraP2 gives sigma/tau. Multiply by tau for absolute conductivity.
- For Seebeck and power factor ratios, tau cancels out.
- Typical tau: 1e-14 s (semiconductors at 300 K), 1e-13 s (metals).

### Power factor (S^2 * sigma/tau)
- Peaks at intermediate doping (~1e19-1e20 cm^-3).
- Optimal doping balances high S (low doping) vs high sigma (high doping).

### ZT estimate
- ZT = S^2 * sigma * T / (kappa_e + kappa_lat).
- Requires tau and kappa_lat as external inputs.
- ZT > 1 is good; ZT > 2 is excellent.

## Common Issues

| Issue | Cause | Fix |
|---|---|---|
| `ImportError: No module named 'BoltzTraP2'` | Not installed | `pip install BoltzTraP2` |
| BoltzTraP2 cannot read QE data | Missing `data-file-schema.xml` | Ensure NSCF completed. Check `outdir/prefix.save/`. |
| Seebeck is noisy or wrong sign | K-grid too coarse | Increase to 40x40x40 or 50x50x50. |
| Results change dramatically with k-grid | Not converged | Run convergence test: 30, 35, 40, 45, 50 per direction. |
| BoltzTraP2 gives zero conductivity | `nosym = .true.` not set | MANDATORY for BoltzTraP2. It needs the full BZ. |
| NSCF too slow (40^3 = 64000 k-points) | Large computation | Use more MPI pools (`-npool N`). Reduce ecutwfc if converged. |
| Cannot compute absolute sigma | tau unknown | Use experimental resistivity to extract tau at one T, or compute from e-ph coupling. |
| ZT estimate seems unreasonable | Wrong tau or kappa_lat | Cross-check tau with Wiedemann-Franz law: kappa_e/sigma should be ~L*T where L=2.44e-8 W*Ohm/K^2. |
