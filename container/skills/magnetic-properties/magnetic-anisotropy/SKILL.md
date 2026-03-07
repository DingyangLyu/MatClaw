# Magnetic Anisotropy Energy (MAE)

## When to Use

- Calculate the magnetic anisotropy energy (MAE): the energy difference between magnetization
  along easy and hard crystallographic axes.
- Determine the easy axis of magnetization for a material.
- Study spin-orbit coupling effects on magnetic properties.
- Investigate materials for permanent magnet or spintronic applications where MAE is a key
  figure of merit.
- Plot the angular dependence of MAE and extract anisotropy constants K1, K2.
- Corresponds to VASPKIT task 621.

## Method Selection

| Criterion | QE (noncollinear + SOC) | VASP (LSORBIT + SAXIS) |
|---|---|---|
| SOC calculation | `noncolin=.true.`, `lspinorb=.true.` | `LSORBIT=.TRUE.`, `LNONCOLLINEAR=.TRUE.` |
| Spin direction control | `angle1`, `angle2` per species | `SAXIS` unit vector |
| Constraint | `constrained_magnetization` | `I_CONSTRAINED_M`, `LAMBDA` |
| Pseudopotentials | Fully relativistic (`_rel`) | PAW with SOC |
| Force theorem | NSCF with SOC on collinear charge | `ICHARG=11` |

## Prerequisites

- Quantum ESPRESSO 7.5 with `pw.x` (compiled with noncollinear + SOC support)
- **Fully relativistic pseudopotentials** (files with `_rel` or `rel` in the name, e.g.,
  `Fe.rel-pbe-spn-kjpaw_psl.1.0.0.UPF`). Standard scalar-relativistic pseudopotentials
  will NOT work with `lspinorb=.true.`.
- Python: `pymatgen`, `numpy`, `scipy`, `matplotlib`
- Dense k-point grids and very tight SCF convergence (MAE values are typically in the
  micro-eV to milli-eV per atom range)
- For VASP: PAW potentials, tight EDIFF (1E-8 or better)

---

## Detailed Steps

### Background

Magnetic anisotropy energy (MAE) is the energy cost of rotating the magnetization from the
easy axis to a hard axis. It arises primarily from spin-orbit coupling (SOC).

**MAE = E(hard axis) - E(easy axis)**

- For cubic systems (e.g., BCC Fe): MAE ~ 1 ueV/atom (very small).
- For hexagonal systems (e.g., HCP Co): MAE ~ 50-65 ueV/atom.
- For rare-earth compounds (e.g., Nd2Fe14B): MAE ~ 1-10 meV/atom.

In QE noncollinear calculations, the initial magnetization direction for each species is
specified using spherical coordinates:

| Direction | angle1 | angle2 | Description |
|---|---|---|---|
| [001] (z) | 0 | 0 | Along c-axis |
| [100] (x) | 90 | 0 | Along a-axis |
| [010] (y) | 90 | 90 | Along b-axis |
| [110] | 90 | 45 | In-plane diagonal |
| [111] | 54.7356 | 45 | Body diagonal |

### Method A: QE -- Noncollinear + SOC

#### Step A1: QE Inputs for Different Magnetization Directions

```python
#!/usr/bin/env python3
"""
Generate QE inputs for MAE calculation with different magnetization directions.
Example: HCP Co (a good test case with measurable MAE ~ 65 ueV/atom).
"""
import os
import subprocess
import re
import numpy as np
from pymatgen.core import Structure, Lattice, Element

PSEUDO_DIR = os.path.abspath("./pseudo")
OUTDIR = os.path.abspath("./tmp_mae")
ECUTWFC = 80.0
ECUTRHO = 640.0
KGRID = (20, 20, 12)
DEGAUSS = 0.005
CONV_THR = "1.0d-10"
MIXING_BETA = 0.2

os.makedirs(PSEUDO_DIR, exist_ok=True)
os.makedirs(OUTDIR, exist_ok=True)

# HCP Co structure
a_co = 2.5071
c_co = 4.0695
lattice = Lattice.hexagonal(a_co, c_co)
structure = Structure(
    lattice,
    ["Co", "Co"],
    [[1/3, 2/3, 1/4], [2/3, 1/3, 3/4]]
)

PSEUDO_FILE = "Co.rel-pbe-spn-kjpaw_psl.0.3.1.UPF"

# Magnetization directions: (label, angle1, angle2, description)
mag_directions = [
    ("001", 0.0, 0.0, "[001] c-axis (expected easy axis)"),
    ("100", 90.0, 0.0, "[100] a-axis (expected hard axis)"),
    ("110", 90.0, 30.0, "[110] in-plane"),
    ("111", 54.7356, 45.0, "[111] body diagonal"),
]


def write_soc_input(structure, label, angle1, angle2, description=""):
    """Write a noncollinear + SOC QE input file."""
    prefix = f"co_soc_{label}"
    outdir_calc = os.path.join(OUTDIR, label)
    os.makedirs(outdir_calc, exist_ok=True)

    cell = structure.lattice.matrix
    species = sorted(set(str(s) for s in structure.species))

    lines = []
    lines.append(f"! MAE calculation: magnetization along {description}")
    lines.append("&CONTROL")
    lines.append(f"    calculation   = 'scf'")
    lines.append(f"    prefix        = '{prefix}'")
    lines.append(f"    outdir        = '{outdir_calc}'")
    lines.append(f"    pseudo_dir    = '{PSEUDO_DIR}'")
    lines.append(f"    tprnfor       = .true.")
    lines.append(f"    verbosity     = 'high'")
    lines.append("/")
    lines.append("&SYSTEM")
    lines.append(f"    ibrav         = 0")
    lines.append(f"    nat           = {len(structure)}")
    lines.append(f"    ntyp          = {len(species)}")
    lines.append(f"    ecutwfc       = {ECUTWFC}")
    lines.append(f"    ecutrho       = {ECUTRHO}")
    lines.append(f"    noncolin      = .true.")
    lines.append(f"    lspinorb      = .true.")

    for i, sp in enumerate(species, start=1):
        lines.append(f"    starting_magnetization({i}) = 0.6")
        lines.append(f"    angle1({i})     = {angle1:.4f}")
        lines.append(f"    angle2({i})     = {angle2:.4f}")

    lines.append(f"    occupations   = 'smearing'")
    lines.append(f"    smearing      = 'mv'")
    lines.append(f"    degauss       = {DEGAUSS}")
    lines.append(f"    nosym         = .true.")
    lines.append("/")
    lines.append("&ELECTRONS")
    lines.append(f"    conv_thr        = {CONV_THR}")
    lines.append(f"    mixing_beta     = {MIXING_BETA}")
    lines.append(f"    electron_maxstep = 300")
    lines.append(f"    diagonalization = 'david'")
    lines.append("/")
    lines.append("ATOMIC_SPECIES")
    for sp in species:
        mass = Element(sp).atomic_mass
        lines.append(f"    {sp}  {mass:.3f}  {PSEUDO_FILE}")

    lines.append("")
    lines.append("CELL_PARAMETERS angstrom")
    for v in cell:
        lines.append(f"    {v[0]:14.10f}  {v[1]:14.10f}  {v[2]:14.10f}")

    lines.append("")
    lines.append("ATOMIC_POSITIONS crystal")
    for site in structure:
        sp = str(site.specie)
        fc = site.frac_coords
        lines.append(f"    {sp}  {fc[0]:14.10f}  {fc[1]:14.10f}  {fc[2]:14.10f}")

    lines.append("")
    lines.append("K_POINTS automatic")
    lines.append(f"    {KGRID[0]} {KGRID[1]} {KGRID[2]}  0 0 0")
    lines.append("")

    filename = f"co_soc_{label}.in"
    with open(filename, "w") as f:
        f.write("\n".join(lines))
    print(f"Written: {filename}  ({description})")
    return filename


# Generate all input files
input_files = {}
for label, a1, a2, desc in mag_directions:
    fname = write_soc_input(structure, label, a1, a2, desc)
    input_files[label] = fname
```

#### Step A2: Run Calculations and Parse Results

```python
#!/usr/bin/env python3
"""
Run SOC calculations for all magnetization directions and parse MAE.
"""
import os
import re
import subprocess
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

mag_directions = [
    ("001", 0.0, 0.0),
    ("100", 90.0, 0.0),
    ("110", 90.0, 30.0),
    ("111", 54.7356, 45.0),
]

PREFIX = "co_soc"


def run_all_soc():
    """Run all SOC calculations."""
    for label, a1, a2 in mag_directions:
        infile = f"{PREFIX}_{label}.in"
        outfile = f"{PREFIX}_{label}.out"
        if not os.path.exists(infile):
            continue
        print(f"Running [{label}]...")
        r = subprocess.run(
            ["mpirun", "-np", "4", "pw.x", "-in", infile],
            capture_output=True, text=True, timeout=3600
        )
        with open(outfile, "w") as f:
            f.write(r.stdout)
        if "convergence has been achieved" in r.stdout:
            m = re.search(r"!\s+total energy\s+=\s+([-\d.]+)\s*Ry", r.stdout)
            if m:
                print(f"  E[{label}] = {m.group(1)} Ry")
        else:
            print(f"  WARNING: [{label}] did not converge!")


def parse_soc_output(filename):
    """Parse total energy and magnetization from noncollinear QE output."""
    result = {
        "energy_ry": None,
        "mag_x": None, "mag_y": None, "mag_z": None,
        "mag_total": None,
        "converged": False,
    }

    if not os.path.exists(filename):
        return result

    with open(filename, "r") as f:
        content = f.read()

    matches = re.findall(r"!\s+total energy\s+=\s+([\d\.\-]+)\s*Ry", content)
    if matches:
        result["energy_ry"] = float(matches[-1])

    mag_pattern = (
        r"total magnetization\s+=\s+"
        r"([\d\.\-]+)\s+([\d\.\-]+)\s+([\d\.\-]+)\s+Bohr"
    )
    mag_matches = re.findall(mag_pattern, content)
    if mag_matches:
        mx, my, mz = [float(x) for x in mag_matches[-1]]
        result["mag_x"] = mx
        result["mag_y"] = my
        result["mag_z"] = mz
        result["mag_total"] = np.sqrt(mx**2 + my**2 + mz**2)

    if "convergence has been achieved" in content:
        result["converged"] = True

    return result


def compute_and_plot_mae(results, n_atoms, reference="001"):
    """Compute MAE and create plots."""
    ry_to_ueV = 13605693.0  # 1 Ry = 13605693 ueV

    ref = results.get(reference)
    if ref is None or ref["energy_ry"] is None:
        print(f"No data for reference direction [{reference}]")
        return

    e_ref = ref["energy_ry"]
    mae_dict = {}

    print(f"\n{'Direction':<12s} {'E (Ry)':<22s} {'dE (ueV/atom)':<16s} {'|M| (uB)':<10s}")
    print("-" * 65)

    for label, r in sorted(results.items()):
        if r["energy_ry"] is None:
            continue
        de = (r["energy_ry"] - e_ref) * ry_to_ueV / n_atoms
        mae_dict[label] = de
        mag = r["mag_total"] if r["mag_total"] else 0.0
        print(f"[{label:<5s}]     {r['energy_ry']:<22.12f} {de:>+14.2f}   {mag:>8.3f}")

    if len(mae_dict) < 2:
        return mae_dict

    easy = min(mae_dict, key=lambda k: mae_dict[k])
    hard = max(mae_dict, key=lambda k: mae_dict[k])
    mae_total = mae_dict[hard] - mae_dict[easy]
    print(f"\nEasy axis: [{easy}], Hard axis: [{hard}]")
    print(f"MAE = {mae_total:.2f} ueV/atom ({mae_total/1000:.4f} meV/atom)")

    # ── Bar chart ─────────────────────────────────────────────────
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    labels = sorted(mae_dict.keys())
    values = [mae_dict[l] for l in labels]
    colors = ["steelblue" if v == min(values) else
              "tomato" if v == max(values) else "gray" for v in values]

    bars = ax1.bar([f"[{l}]" for l in labels], values, color=colors,
                   edgecolor="black", linewidth=1.2)
    for bar, val in zip(bars, values):
        ax1.text(bar.get_x() + bar.get_width()/2, val + 2,
                 f"{val:.1f}", ha="center", va="bottom", fontsize=10)

    ax1.set_ylabel(r"$\Delta E$ ($\mu$eV/atom)", fontsize=13)
    ax1.set_xlabel("Magnetization direction", fontsize=13)
    ax1.set_title(f"MAE relative to [{reference}]", fontsize=14)
    ax1.axhline(0, color="black", linewidth=0.5)

    # ── Angular fit ───────────────────────────────────────────────
    angle_map = {"001": 0.0, "111": 54.7356, "100": 90.0, "110": 90.0}
    thetas = []
    energies = []
    for l in labels:
        if l in angle_map:
            thetas.append(angle_map[l] * np.pi / 180)
            energies.append(mae_dict[l])

    thetas = np.array(thetas)
    energies = np.array(energies)

    # Fit E = K1 sin^2(theta) + K2 sin^4(theta)
    A = np.column_stack([np.sin(thetas)**2, np.sin(thetas)**4])
    from numpy.linalg import lstsq
    coeffs, _, _, _ = lstsq(A, energies, rcond=None)
    K1, K2 = coeffs

    theta_fine = np.linspace(0, np.pi/2, 200)
    e_fit = K1 * np.sin(theta_fine)**2 + K2 * np.sin(theta_fine)**4

    ax2.plot(theta_fine * 180 / np.pi, e_fit, "b-", linewidth=2,
             label=f"Fit: $K_1$={K1:.1f}, $K_2$={K2:.1f}")
    ax2.scatter(thetas * 180 / np.pi, energies, c="red", s=80, zorder=5,
                edgecolors="black", label="DFT")
    ax2.set_xlabel(r"$\theta$ from [001] (degrees)", fontsize=13)
    ax2.set_ylabel(r"$\Delta E$ ($\mu$eV/atom)", fontsize=13)
    ax2.set_title("Angular Dependence of MAE", fontsize=14)
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("mae_comparison.png", dpi=200, bbox_inches="tight")
    plt.close()
    print(f"\nSaved: mae_comparison.png")
    return mae_dict


# ── Parse actual or demo data ─────────────────────────────────────
results = {}
for label, a1, a2 in mag_directions:
    outfile = f"{PREFIX}_{label}.out"
    results[label] = parse_soc_output(outfile)

has_data = any(r["energy_ry"] is not None for r in results.values())

if has_data:
    compute_and_plot_mae(results, n_atoms=2, reference="001")
else:
    print("No output files found. Using demo data for HCP Co.\n")
    demo = {
        "001": {"energy_ry": -678.12345678900, "mag_x": 0.00, "mag_y": 0.00,
                "mag_z": 3.28, "mag_total": 3.28, "converged": True},
        "100": {"energy_ry": -678.12345590000, "mag_x": 3.28, "mag_y": 0.00,
                "mag_z": 0.00, "mag_total": 3.28, "converged": True},
        "110": {"energy_ry": -678.12345592000, "mag_x": 2.32, "mag_y": 2.32,
                "mag_z": 0.00, "mag_total": 3.28, "converged": True},
        "111": {"energy_ry": -678.12345640000, "mag_x": 1.89, "mag_y": 1.09,
                "mag_z": 1.89, "mag_total": 3.28, "converged": True},
    }
    compute_and_plot_mae(demo, n_atoms=2, reference="001")
```

### Method B: VASP -- LSORBIT + SAXIS

```python
#!/usr/bin/env python3
"""
MAE calculation using VASP with LSORBIT and SAXIS.

Workflow:
  1) Collinear spin-polarized SCF (ISPIN=2) to get CHGCAR.
  2) For each magnetization direction, SOC calculation with SAXIS.
  3) Compare total energies.
"""
import numpy as np
import re
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def generate_vasp_incar_soc(saxis, label, encut=500, ediff=1e-8, icharg=11):
    """
    Generate INCAR for a VASP SOC calculation.

    Parameters
    ----------
    saxis : tuple of 3 floats
        Magnetization direction (unit vector).
    label : str
        Direction label for comments.
    encut : float
        Plane-wave cutoff (eV).
    ediff : float
        SCF convergence criterion.
    icharg : int
        11 = non-SCF (force theorem), 1 = self-consistent from CHGCAR.
    """
    sx, sy, sz = saxis
    incar = f"""SYSTEM  = MAE_SOC_{label}
PREC    = Accurate
ENCUT   = {encut}
EDIFF   = {ediff}
ISMEAR  = 1
SIGMA   = 0.1
ISPIN   = 2
LSORBIT = .TRUE.
LNONCOLLINEAR = .TRUE.
SAXIS   = {sx:.6f} {sy:.6f} {sz:.6f}
ICHARG  = {icharg}
LMAXMIX = 6
GGA_COMPAT = .FALSE.
LORBIT  = 11
LWAVE   = .FALSE.
LCHARG  = .FALSE.
"""
    return incar


def parse_vasp_energy(outcar_file):
    """Parse final total energy from VASP OUTCAR."""
    energy = None
    with open(outcar_file, "r") as f:
        for line in f:
            if "energy  without entropy" in line:
                m = re.search(r"=\s*([-\d.]+)", line)
                if m:
                    energy = float(m.group(1))
    return energy


def compute_mae_vasp(directions_dict, n_atoms=1):
    """
    Compute MAE from VASP OUTCAR files.

    Parameters
    ----------
    directions_dict : dict
        {label: outcar_path}
    n_atoms : int
        Atoms per cell.
    """
    results = {}
    for label, path in directions_dict.items():
        if os.path.exists(path):
            e = parse_vasp_energy(path)
            if e is not None:
                results[label] = e

    if len(results) < 2:
        print("Need at least 2 converged calculations.")
        return

    e_min = min(results.values())
    ref = min(results, key=lambda k: results[k])

    print(f"Reference: [{ref}]")
    for label in sorted(results):
        de_ueV = (results[label] - e_min) * 1e6 / n_atoms  # eV -> ueV
        print(f"  [{label}]: dE = {de_ueV:+.2f} ueV/atom")


# SAXIS directions (unit vectors)
vasp_directions = {
    "001": (0, 0, 1),
    "100": (1, 0, 0),
    "110": (1/np.sqrt(2), 1/np.sqrt(2), 0),
    "111": (1/np.sqrt(3), 1/np.sqrt(3), 1/np.sqrt(3)),
}

for label, saxis in vasp_directions.items():
    incar = generate_vasp_incar_soc(saxis, label)
    with open(f"INCAR_soc_{label}", "w") as f:
        f.write(incar)
    print(f"Written: INCAR_soc_{label}  SAXIS = {saxis}")

# Usage with existing OUTCARs:
# compute_mae_vasp({"001": "soc_001/OUTCAR", "100": "soc_100/OUTCAR"}, n_atoms=2)
```

### K-Point Convergence Test

```python
#!/usr/bin/env python3
"""
Plot MAE convergence with k-point density.
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def plot_kpoint_convergence(kgrids, mae_values_ueV, output_file="mae_kconv.png"):
    """
    Plot MAE as a function of k-grid density.
    """
    nk_total = [k[0] * k[1] * k[2] for k in kgrids]

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(nk_total, mae_values_ueV, "o-", color="steelblue", markersize=8,
            linewidth=2, markeredgecolor="black")
    ax.set_xlabel(r"Total k-points ($N_{k1} \times N_{k2} \times N_{k3}$)", fontsize=13)
    ax.set_ylabel(r"MAE ($\mu$eV/atom)", fontsize=13)
    ax.set_title("MAE Convergence with K-Point Density", fontsize=14)
    ax.axhline(y=mae_values_ueV[-1], color="gray", linestyle="--", linewidth=0.8,
               label=f"Converged: {mae_values_ueV[-1]:.1f} ueV/atom")
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_file, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"Saved: {output_file}")


# Demo convergence data (HCP Co)
demo_kgrids = [(8,8,5), (12,12,7), (16,16,10), (20,20,12), (24,24,14), (28,28,16)]
demo_mae = [120.5, 85.3, 68.2, 62.1, 60.8, 60.5]
plot_kpoint_convergence(demo_kgrids, demo_mae)
```

---

## Key Parameters

| Parameter | Typical Values | Notes |
|---|---|---|
| `noncolin` | `.true.` | Required for SOC calculations |
| `lspinorb` | `.true.` | Enables spin-orbit coupling |
| `angle1(i)`, `angle2(i)` | degrees | Polar/azimuthal angles for species i |
| `nosym` | `.true.` | Ensures identical k-grids for all directions. Critical. |
| `conv_thr` | 1.0d-10 to 1.0d-12 | Very tight; MAE is in ueV range |
| `ecutwfc` | 70--100 Ry | Higher than standard for SOC accuracy |
| `degauss` | 0.003--0.01 Ry | Small smearing to minimize artifacts |
| `mixing_beta` | 0.1--0.2 | Low mixing for stable SOC convergence |
| K-grid | 20x20x20+ | MAE converges slowly; must be tested |
| `SAXIS` (VASP) | unit vector | Magnetization direction |
| `ICHARG` (VASP) | 11 (force theorem) or 1 (SCF) | 11 is faster but less accurate |
| Pseudopotential | Fully relativistic (`*rel*`) | Scalar-relativistic PPs cannot do SOC |

## Interpreting Results

### Typical MAE Values

| Material | MAE (ueV/atom) | Easy Axis | Notes |
|---|---|---|---|
| BCC Fe | ~1.4 | [100] | Very small; extremely hard to converge |
| FCC Ni | ~2.7 | [111] | Very small |
| HCP Co | ~50--65 | [001] | Moderate; good test case |
| FePt L10 | ~1000--2000 | [001] | Large; permanent magnets |
| CrI3 (2D) | ~500--800 | out-of-plane | Large due to reduced symmetry |

### Sign and Anisotropy Constants

- **MAE > 0** (E_hard > E_easy): preferred (easy) magnetization direction exists.
- **Uniaxial systems**: E(theta) = K1 sin^2(theta) + K2 sin^4(theta).
  K1 > 0 means easy axis is the c-axis. K1 < 0 means easy plane.
- **Cubic systems**: described by K1 and K2 with E = K1(alpha1^2 alpha2^2 + ...)
  where alpha_i are direction cosines.

### Convergence Checklist

MAE must be converged with respect to:
1. K-point grid (most critical)
2. Plane-wave cutoff (ecutwfc)
3. SCF convergence threshold
4. Smearing width (degauss)

A well-converged MAE should change by less than ~10% when doubling the k-grid.

## Common Issues

| Problem | Solution |
|---|---|
| Wrong pseudopotential (crash or garbage) | Must use fully relativistic PPs (`*rel*`). Scalar-relativistic will not work with `lspinorb=.true.` |
| Different k-grids for different directions | Set `nosym=.true.` in QE. In VASP, use `ISYM=-1` |
| MAE not converged with k-points | Increase grid systematically. For BCC Fe, need 30x30x30+ |
| SCF oscillates or does not converge | Reduce `mixing_beta` to 0.1--0.15. Increase `electron_maxstep`. Try `mixing_mode='local-TF'` |
| Magnetization rotates during SCF | Use `constrained_magnetization='total direction'` in QE or `I_CONSTRAINED_M` in VASP |
| Smearing artifacts comparable to MAE | Reduce `degauss` to 0.003 Ry or less. Use Methfessel-Paxton or cold smearing |
| Cost too high (4--10x collinear) | SOC doubles basis size + nosym increases k-points. Use force theorem (NSCF with SOC) for faster estimates |
| BCC Fe gives wrong easy axis sign | Fe MAE is ~1 ueV/atom and extremely sensitive to all parameters. Not a good first test case; use HCP Co instead |
