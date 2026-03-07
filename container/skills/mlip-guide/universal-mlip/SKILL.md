# Universal Machine Learning Interatomic Potentials

## When to Use

- Rapid geometry relaxation and energy evaluation (orders of magnitude faster than DFT)
- High-throughput screening of many structures
- Molecular dynamics simulations at near-DFT accuracy
- Phonon calculations via finite displacements
- Elastic constants and equations of state
- Any task that needs energy/forces/stress but not electronic properties (band structure, DOS, charge density)

## Method Selection

```
Need electronic properties?
  YES --> Cannot use MLIPs. Use Quantum ESPRESSO DFT.
  NO  --> Continue below.

Need fastest possible screening (thousands of structures)?
  YES --> MACE-MP-0 medium or CHGNet (both fast, good accuracy)

Need best accuracy for structures/energies?
  YES --> MACE-MP-0 large (most accurate universal MLIP)

Need charge-aware predictions (ionic systems, batteries)?
  YES --> CHGNet (predicts magnetic moments and charges)

Need to use with LAMMPS for large-scale MD?
  YES --> SevenNet or MACE (both have LAMMPS interfaces)

Unsure?
  --> Start with MACE-MP-0 medium. Validate against DFT for your specific system.
```

## Prerequisites

- **MACE-MP-0**: Pre-installed (`mace-torch` package). No additional installation needed.
- **CHGNet**: `pip install chgnet`
- **M3GNet/MatGL**: `pip install matgl dgl`
- **SevenNet**: `pip install sevenn`
- All MLIPs use ASE as the interface layer. ASE and pymatgen are pre-installed.

## Detailed Steps

---

### 1. MACE-MP-0 (Pre-installed)

MACE-MP-0 is a universal potential trained on the Materials Project database (~150k structures, 89 elements). It provides energy, forces, and stress predictions.

#### Basic usage: energy, forces, stress

```python
from ase.build import bulk
from mace.calculators import mace_mp

# Load the model -- "medium" is the default, "large" is more accurate
calc = mace_mp(model="medium", device="cpu", default_dtype="float64")

# Create a structure
atoms = bulk("Si", "diamond", a=5.43)
atoms.calc = calc

# Get properties
energy = atoms.get_potential_energy()
forces = atoms.get_forces()
stress = atoms.get_stress()  # Voigt notation: xx, yy, zz, yz, xz, xy

print(f"Energy:  {energy:.6f} eV")
print(f"Forces (eV/A):\n{forces}")
print(f"Stress (eV/A^3): {stress}")
print(f"Energy per atom: {energy / len(atoms):.6f} eV/atom")
```

#### Model sizes

```python
# Small -- fastest, least accurate
calc_small = mace_mp(model="small", device="cpu")

# Medium -- good balance (recommended default)
calc_medium = mace_mp(model="medium", device="cpu")

# Large -- most accurate, slowest
calc_large = mace_mp(model="large", device="cpu")
```

On first use, the model weights are downloaded automatically from the MACE repository and cached locally. Subsequent calls load from cache.

#### Structure relaxation

```python
from ase.build import bulk
from ase.optimize import BFGS, FIRE
from ase.constraints import ExpCellFilter
from mace.calculators import mace_mp

calc = mace_mp(model="medium", device="cpu", default_dtype="float64")

# Create initial structure (e.g., slightly distorted)
atoms = bulk("Si", "diamond", a=5.50)  # intentionally wrong lattice constant
atoms.calc = calc

# Relax atomic positions only
opt = BFGS(atoms, logfile="/tmp/relax_positions.log")
opt.run(fmax=0.01)  # eV/Angstrom

print(f"Relaxed energy: {atoms.get_potential_energy():.6f} eV")
print(f"Max force: {atoms.get_forces().max():.6f} eV/A")

# Relax both positions and cell (variable-cell relaxation)
atoms2 = bulk("Si", "diamond", a=5.50)
atoms2.calc = calc
ecf = ExpCellFilter(atoms2)
opt2 = BFGS(ecf, logfile="/tmp/relax_cell.log")
opt2.run(fmax=0.01)

print(f"\nVariable-cell relaxed energy: {atoms2.get_potential_energy():.6f} eV")
print(f"Relaxed lattice constant: {atoms2.cell.cellpar()[0]:.4f} A")
```

#### Molecular dynamics

```python
from ase.build import bulk
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution
from ase.md.langevin import Langevin
from ase import units
from mace.calculators import mace_mp

calc = mace_mp(model="medium", device="cpu", default_dtype="float64")

# Build a supercell for MD
atoms = bulk("Cu", "fcc", a=3.615) * (3, 3, 3)  # 108 atoms
atoms.calc = calc

# Initialize velocities at 300 K
MaxwellBoltzmannDistribution(atoms, temperature_K=300)

# Langevin thermostat (NVT)
dyn = Langevin(
    atoms,
    timestep=1.0 * units.fs,
    temperature_K=300,
    friction=0.01 / units.fs,
    logfile="/tmp/md.log",
    loginterval=10,
)

# Run 1000 steps (1 ps)
print("Running MD...")
dyn.run(1000)
print(f"Final temperature: {atoms.get_temperature():.1f} K")
print(f"Final energy: {atoms.get_potential_energy():.4f} eV")
```

#### Read structures from files

```python
from ase.io import read
from mace.calculators import mace_mp

calc = mace_mp(model="medium", device="cpu", default_dtype="float64")

# Read from CIF
atoms = read("structure.cif")
atoms.calc = calc
print(f"Energy: {atoms.get_potential_energy():.6f} eV")

# Read from POSCAR
atoms = read("POSCAR", format="vasp")
atoms.calc = calc

# Read from Materials Project via pymatgen, convert to ASE
from pymatgen.core import Structure
from pymatgen.io.ase import AseAtomsAdaptor

struct = Structure.from_file("structure.cif")
atoms = AseAtomsAdaptor.get_atoms(struct)
atoms.calc = calc
energy = atoms.get_potential_energy()
```

---

### 2. CHGNet (pip install chgnet)

CHGNet is a charge-informed graph neural network potential. It predicts energies, forces, stresses, and additionally magnetic moments and atomic charges.

```bash
pip install chgnet
```

#### Basic usage

```python
from chgnet.model.dynamics import CHGNetCalculator
from ase.build import bulk

# Load pre-trained CHGNet model
calc = CHGNetCalculator(use_device="cpu")

atoms = bulk("Si", "diamond", a=5.43)
atoms.calc = calc

energy = atoms.get_potential_energy()
forces = atoms.get_forces()
stress = atoms.get_stress()

print(f"CHGNet Energy: {energy:.6f} eV")
print(f"CHGNet Forces:\n{forces}")
print(f"CHGNet Stress: {stress}")
```

#### Relaxation with CHGNet

```python
from chgnet.model.dynamics import CHGNetCalculator
from ase.build import bulk
from ase.optimize import BFGS
from ase.constraints import ExpCellFilter

calc = CHGNetCalculator(use_device="cpu")
atoms = bulk("LiFePO4", crystalstructure="olivine", a=10.33, b=6.01, c=4.69)

# If building from scratch is complex, use pymatgen:
from pymatgen.core import Structure, Lattice
from pymatgen.io.ase import AseAtomsAdaptor

# Example: build NaCl
struct = Structure.from_spacegroup(
    "Fm-3m",
    Lattice.cubic(5.64),
    ["Na", "Cl"],
    [[0, 0, 0], [0.5, 0.5, 0.5]]
)
atoms = AseAtomsAdaptor.get_atoms(struct)
atoms.calc = calc

ecf = ExpCellFilter(atoms)
opt = BFGS(ecf, logfile="/tmp/chgnet_relax.log")
opt.run(fmax=0.05)

print(f"Relaxed energy: {atoms.get_potential_energy():.4f} eV")
print(f"Relaxed cell: {atoms.cell.cellpar()}")
```

#### Charge and magnetic moment predictions

CHGNet uniquely predicts site-wise magnetic moments:

```python
from chgnet.model import CHGNet
from pymatgen.core import Structure, Lattice

# Load model directly for charge/magmom predictions
model = CHGNet.load()

# Build an iron structure
struct = Structure.from_spacegroup(
    "Im-3m",
    Lattice.cubic(2.87),
    ["Fe"],
    [[0, 0, 0]]
)

prediction = model.predict_structure(struct)
print(f"Energy: {prediction['e']:.4f} eV/atom")
print(f"Forces: {prediction['f']}")
print(f"Stress: {prediction['s']}")
print(f"Magnetic moments: {prediction['m']}")
```

---

### 3. M3GNet / MatGL (pip install matgl)

M3GNet is a universal potential from the Materials Virtual Lab. MatGL is its PyTorch-based implementation.

```bash
pip install matgl dgl
```

#### Basic usage with ASE

```python
import matgl
from matgl.ext.ase import M3GNetCalculator
from ase.build import bulk

# Load pre-trained M3GNet universal potential
pot = matgl.load_model("M3GNet-MP-2021.2.8-DIRECT-PES")
calc = M3GNetCalculator(potential=pot)

atoms = bulk("Si", "diamond", a=5.43)
atoms.calc = calc

energy = atoms.get_potential_energy()
forces = atoms.get_forces()
stress = atoms.get_stress()

print(f"M3GNet Energy: {energy:.6f} eV")
print(f"M3GNet Forces:\n{forces}")
```

#### Relaxation with M3GNet

```python
import matgl
from matgl.ext.ase import M3GNetCalculator, Relaxer
from pymatgen.core import Structure, Lattice

# Use the built-in Relaxer (wraps ASE optimization)
pot = matgl.load_model("M3GNet-MP-2021.2.8-DIRECT-PES")
relaxer = Relaxer(potential=pot)

struct = Structure.from_spacegroup(
    "Fm-3m",
    Lattice.cubic(5.50),
    ["Si"],
    [[0, 0, 0]]
)

result = relaxer.relax(struct, fmax=0.01)
relaxed = result["final_structure"]
print(f"Relaxed lattice constant: {relaxed.lattice.a:.4f} A")
print(f"Final energy: {result['trajectory'].energies[-1]:.4f} eV")
```

---

### 4. SevenNet (pip install sevenn)

SevenNet-0 is a universal MLIP based on the NequIP architecture, trained on the MPtrj dataset.

```bash
pip install sevenn
```

#### Basic usage with ASE

```python
from sevenn.sevennet_calculator import SevenNetCalculator
from ase.build import bulk

# Load the pre-trained SevenNet-0 model
# "7net-0" is the standard model identifier
calc = SevenNetCalculator("7net-0", device="cpu")

atoms = bulk("Si", "diamond", a=5.43)
atoms.calc = calc

energy = atoms.get_potential_energy()
forces = atoms.get_forces()
stress = atoms.get_stress()

print(f"SevenNet Energy: {energy:.6f} eV")
print(f"SevenNet Forces:\n{forces}")
```

#### Relaxation with SevenNet

```python
from sevenn.sevennet_calculator import SevenNetCalculator
from ase.build import bulk
from ase.optimize import BFGS
from ase.constraints import ExpCellFilter

calc = SevenNetCalculator("7net-0", device="cpu")

atoms = bulk("Si", "diamond", a=5.50)
atoms.calc = calc

ecf = ExpCellFilter(atoms)
opt = BFGS(ecf, logfile="/tmp/sevenn_relax.log")
opt.run(fmax=0.01)

print(f"SevenNet relaxed energy: {atoms.get_potential_energy():.4f} eV")
print(f"SevenNet lattice constant: {atoms.cell.cellpar()[0]:.4f} A")
```

---

### 5. Comparison: when to use which MLIP

| Feature | MACE-MP-0 | CHGNet | M3GNet | SevenNet-0 |
|---|---|---|---|---|
| **Pre-installed** | Yes | No (pip) | No (pip) | No (pip) |
| **Training data** | MPtrj (~150k) | MPtrj (~150k) | MP-2021 (~187k) | MPtrj (~190k) |
| **Elements** | 89 | 89 | 89 | 89 |
| **Architecture** | Equivariant message passing | GNN + charge | GNN (M3GNet) | NequIP (equivariant) |
| **Model sizes** | small/medium/large | single | single | single |
| **Predicts charges** | No | Yes (magmom) | No | No |
| **Speed (relative)** | Fast (medium) | Fast | Fast | Medium |
| **Accuracy (general)** | Best (large) | Very good | Good | Very good |
| **LAMMPS interface** | Yes (pair_style mace) | No | No | Yes (pair_style sevenn) |
| **Best for** | General use, MD | Ionic/magnetic | Quick screening | Equivariant accuracy |

#### Recommended use cases

| Task | Recommended MLIP | Why |
|---|---|---|
| Quick screening (100+ structures) | MACE-MP-0 medium | Fast, pre-installed, good accuracy |
| Accurate relaxation | MACE-MP-0 large | Best energy/force accuracy |
| Molecular dynamics | MACE-MP-0 medium/large | Good forces, stable dynamics |
| Phonon band structure | MACE-MP-0 large | Needs accurate force constants |
| Elastic constants | Any | All give reasonable elastic tensors |
| Battery/ionic materials | CHGNet | Charge-aware, good for Li-ion systems |
| Large-scale LAMMPS MD | SevenNet or MACE | Native LAMMPS interfaces |
| Initial structure screening | M3GNet or CHGNet | Fast, decent accuracy |

#### NOT recommended for

- **Band gaps / electronic structure**: MLIPs do not model electrons. Use DFT.
- **Magnetic properties**: Only CHGNet predicts magnetic moments (and only approximately). Use DFT for precise magnetic ordering.
- **Charged systems / surfaces with adsorbates**: MLIPs trained on bulk periodic crystals. Results for surfaces, molecules, or charged states may be unreliable.
- **High-pressure phases**: Training data is mostly near-equilibrium. Extrapolation to extreme pressures is risky.
- **Rare/radioactive elements**: Training data may have very few examples. Validate carefully.

---

### 6. Benchmark: lattice constant and bulk modulus of Si with all MLIPs

This complete script computes the equilibrium lattice constant and bulk modulus of diamond Si using all available universal MLIPs, and compares results.

```python
#!/usr/bin/env python3
"""
Benchmark: compute Si lattice constant and bulk modulus with all available
universal MLIPs. Compare against experimental values.

Experimental reference:
  a0 = 5.431 A
  B0 = 99.2 GPa (Birch-Murnaghan fit)
"""
import numpy as np
from ase.build import bulk
from ase.optimize import BFGS
from ase.constraints import ExpCellFilter
from ase.eos import EquationOfState
from ase.units import kJ
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")


def get_calculators():
    """Load all available MLIP calculators. Returns dict of name -> calculator."""
    calcs = {}

    # 1. MACE-MP-0 (pre-installed)
    try:
        from mace.calculators import mace_mp
        calcs["MACE-MP-0 (medium)"] = mace_mp(model="medium", device="cpu", default_dtype="float64")
        print("  Loaded MACE-MP-0 medium")
    except Exception as e:
        print(f"  MACE-MP-0 medium not available: {e}")

    try:
        from mace.calculators import mace_mp
        calcs["MACE-MP-0 (large)"] = mace_mp(model="large", device="cpu", default_dtype="float64")
        print("  Loaded MACE-MP-0 large")
    except Exception as e:
        print(f"  MACE-MP-0 large not available: {e}")

    # 2. CHGNet
    try:
        from chgnet.model.dynamics import CHGNetCalculator
        calcs["CHGNet"] = CHGNetCalculator(use_device="cpu")
        print("  Loaded CHGNet")
    except ImportError:
        print("  CHGNet not installed (pip install chgnet)")
    except Exception as e:
        print(f"  CHGNet error: {e}")

    # 3. M3GNet / MatGL
    try:
        import matgl
        from matgl.ext.ase import M3GNetCalculator
        pot = matgl.load_model("M3GNet-MP-2021.2.8-DIRECT-PES")
        calcs["M3GNet"] = M3GNetCalculator(potential=pot)
        print("  Loaded M3GNet")
    except ImportError:
        print("  MatGL not installed (pip install matgl dgl)")
    except Exception as e:
        print(f"  M3GNet error: {e}")

    # 4. SevenNet
    try:
        from sevenn.sevennet_calculator import SevenNetCalculator
        calcs["SevenNet-0"] = SevenNetCalculator("7net-0", device="cpu")
        print("  Loaded SevenNet-0")
    except ImportError:
        print("  SevenNet not installed (pip install sevenn)")
    except Exception as e:
        print(f"  SevenNet error: {e}")

    return calcs


def compute_lattice_constant(calc, element="Si", crystal="diamond", a_guess=5.43):
    """Relax the cell to find equilibrium lattice constant."""
    atoms = bulk(element, crystal, a=a_guess)
    atoms.calc = calc
    ecf = ExpCellFilter(atoms)
    opt = BFGS(ecf, logfile=None)
    opt.run(fmax=0.001, steps=200)
    a_eq = atoms.cell.cellpar()[0]
    e_eq = atoms.get_potential_energy() / len(atoms)
    return a_eq, e_eq


def compute_bulk_modulus(calc, element="Si", crystal="diamond", a_eq=5.43,
                         strain_range=0.06, n_points=11):
    """Compute bulk modulus via E-V curve and Birch-Murnaghan EOS fit."""
    volumes = []
    energies = []

    strains = np.linspace(1 - strain_range, 1 + strain_range, n_points)

    for s in strains:
        atoms = bulk(element, crystal, a=a_eq * s)
        atoms.calc = calc
        e = atoms.get_potential_energy()
        v = atoms.get_volume()
        volumes.append(v)
        energies.append(e)

    volumes = np.array(volumes)
    energies = np.array(energies)

    # Fit Birch-Murnaghan EOS
    eos = EquationOfState(volumes, energies, eos="birchmurnaghan")
    try:
        v0, e0, B = eos.fit()
        B_GPa = B / kJ * 1.0e24  # eV/A^3 to GPa
        return B_GPa, volumes, energies, eos
    except Exception as e:
        print(f"    EOS fit failed: {e}")
        return None, volumes, energies, None


def main():
    print("=" * 70)
    print("MLIP Benchmark: Si lattice constant and bulk modulus")
    print("=" * 70)

    # Experimental reference
    a_exp = 5.431  # Angstrom
    B_exp = 99.2   # GPa

    print("\nLoading calculators...")
    calcs = get_calculators()

    if not calcs:
        print("No MLIPs available. Install at least one.")
        return

    results = {}

    for name, calc in calcs.items():
        print(f"\n--- {name} ---")

        # Step 1: Find equilibrium lattice constant
        try:
            a_eq, e_eq = compute_lattice_constant(calc)
            print(f"  Lattice constant: {a_eq:.4f} A (exp: {a_exp:.3f} A, "
                  f"error: {abs(a_eq - a_exp)/a_exp*100:.2f}%)")
            print(f"  Energy/atom: {e_eq:.4f} eV")
        except Exception as e:
            print(f"  Relaxation failed: {e}")
            continue

        # Step 2: Compute bulk modulus
        try:
            B_GPa, vols, energies, eos = compute_bulk_modulus(calc, a_eq=a_eq)
            if B_GPa is not None:
                print(f"  Bulk modulus: {B_GPa:.1f} GPa (exp: {B_exp:.1f} GPa, "
                      f"error: {abs(B_GPa - B_exp)/B_exp*100:.1f}%)")
                results[name] = {
                    "a_eq": a_eq,
                    "e_eq": e_eq,
                    "B_GPa": B_GPa,
                    "volumes": vols,
                    "energies": energies,
                    "eos": eos,
                }
            else:
                results[name] = {"a_eq": a_eq, "e_eq": e_eq, "B_GPa": None}
        except Exception as e:
            print(f"  Bulk modulus calculation failed: {e}")
            results[name] = {"a_eq": a_eq, "e_eq": e_eq, "B_GPa": None}

    # Print summary table
    print("\n" + "=" * 70)
    print(f"{'Model':<25} {'a (A)':>10} {'Error (%)':>10} {'B (GPa)':>10} {'Error (%)':>10}")
    print("-" * 70)
    print(f"{'Experiment':<25} {a_exp:>10.3f} {'--':>10} {B_exp:>10.1f} {'--':>10}")
    for name, r in results.items():
        a_err = abs(r["a_eq"] - a_exp) / a_exp * 100
        if r.get("B_GPa"):
            b_err = abs(r["B_GPa"] - B_exp) / B_exp * 100
            print(f"{name:<25} {r['a_eq']:>10.4f} {a_err:>9.2f}% {r['B_GPa']:>10.1f} {b_err:>9.1f}%")
        else:
            print(f"{name:<25} {r['a_eq']:>10.4f} {a_err:>9.2f}% {'N/A':>10} {'N/A':>10}")
    print("=" * 70)

    # Plot E-V curves
    fig, ax = plt.subplots(figsize=(10, 7))
    colors = ["#1565C0", "#2E7D32", "#E65100", "#6A1B9A", "#C62828"]

    for i, (name, r) in enumerate(results.items()):
        if "volumes" not in r:
            continue
        color = colors[i % len(colors)]
        vols = r["volumes"]
        energies = r["energies"]
        # Normalize energies to minimum
        energies_shifted = energies - energies.min()

        ax.plot(vols, energies_shifted, "o", color=color, markersize=6)

        # Plot EOS fit if available
        if r.get("eos"):
            try:
                v_fit = np.linspace(vols.min(), vols.max(), 100)
                # Use the eos object to plot
                ax.plot([], [], "-", color=color, linewidth=2, label=name)
                # Manual BM fit curve
                v0, e0, B = r["eos"].fit()
                from ase.eos import birchmurnaghan
                e_fit = birchmurnaghan(v_fit, e0, v0, B, 4.0)
                e_fit_shifted = e_fit - energies.min()
                ax.plot(v_fit, e_fit_shifted, "-", color=color, linewidth=2)
            except Exception:
                ax.plot(vols, energies_shifted, "-", color=color, linewidth=1,
                        label=name)
        else:
            ax.plot(vols, energies_shifted, "-", color=color, linewidth=1,
                    label=name)

    ax.set_xlabel("Volume ($\\AA^3$)", fontsize=14)
    ax.set_ylabel("Energy - E$_{min}$ (eV)", fontsize=14)
    ax.set_title("E-V curves for diamond Si: MLIP comparison", fontsize=16)
    ax.legend(fontsize=12)
    ax.grid(True, alpha=0.3)
    ax.tick_params(labelsize=12)
    plt.tight_layout()

    plot_file = "/tmp/mlip_benchmark_si.png"
    plt.savefig(plot_file, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\nE-V plot saved to {plot_file}")


if __name__ == "__main__":
    main()
```

---

### 7. Validating MLIP results against DFT

Always validate MLIP predictions for novel or unusual systems. Run DFT on a few representative structures and compare.

```python
#!/usr/bin/env python3
"""
Validate MLIP predictions against Quantum ESPRESSO DFT for a set of structures.
"""
import numpy as np
from ase.build import bulk
from ase.io import write
from mace.calculators import mace_mp
import subprocess
import os


def run_qe_scf(atoms, work_dir, pseudo_dir, ecutwfc=60, ecutrho=480, kpoints=(6,6,6)):
    """Run a QE SCF calculation and return the total energy."""
    os.makedirs(work_dir, exist_ok=True)

    # Generate QE input using ASE
    from ase.io.espresso import write_espresso_in

    input_data = {
        "control": {
            "calculation": "scf",
            "outdir": os.path.join(work_dir, "tmp"),
            "pseudo_dir": pseudo_dir,
            "tprnfor": True,
            "tstress": True,
        },
        "system": {
            "ecutwfc": ecutwfc,
            "ecutrho": ecutrho,
            "occupations": "smearing",
            "smearing": "cold",
            "degauss": 0.01,
        },
        "electrons": {
            "conv_thr": 1.0e-8,
        },
    }

    pseudopotentials = {}
    for symbol in set(atoms.get_chemical_symbols()):
        pseudopotentials[symbol] = f"{symbol}.UPF"

    input_file = os.path.join(work_dir, "scf.in")
    with open(input_file, "w") as f:
        write_espresso_in(
            f, atoms,
            input_data=input_data,
            pseudopotentials=pseudopotentials,
            kpts=kpoints,
        )

    # Run QE
    output_file = os.path.join(work_dir, "scf.out")
    with open(output_file, "w") as fout:
        result = subprocess.run(
            ["mpirun", "--allow-run-as-root", "-np", "4",
             "pw.x", "-in", input_file],
            stdout=fout, stderr=subprocess.STDOUT,
            timeout=3600
        )

    # Parse energy from output
    energy = None
    with open(output_file) as f:
        for line in f:
            if "!" in line and "total energy" in line:
                energy = float(line.split("=")[1].split("Ry")[0].strip()) * 13.6057
    return energy


def compare_mlip_dft():
    """Compare MLIP energies vs DFT for strained Si."""
    calc = mace_mp(model="medium", device="cpu", default_dtype="float64")

    a0 = 5.431
    strains = [0.96, 0.98, 1.00, 1.02, 1.04]

    mlip_energies = []
    dft_energies = []

    for s in strains:
        atoms = bulk("Si", "diamond", a=a0 * s)

        # MLIP energy
        atoms.calc = calc
        e_mlip = atoms.get_potential_energy() / len(atoms)
        mlip_energies.append(e_mlip)

        # DFT energy (uncomment to run -- requires pseudopotentials)
        # e_dft = run_qe_scf(atoms, f"/tmp/qe_si_{s:.2f}", "/tmp/pseudo")
        # dft_energies.append(e_dft / len(atoms))
        # For demonstration, use placeholder
        dft_energies.append(None)

        print(f"Strain {s:.2f}: MLIP = {e_mlip:.4f} eV/atom")

    # If DFT results are available, compute error metrics
    if all(e is not None for e in dft_energies):
        mlip_arr = np.array(mlip_energies)
        dft_arr = np.array(dft_energies)
        mae = np.mean(np.abs(mlip_arr - dft_arr))
        rmse = np.sqrt(np.mean((mlip_arr - dft_arr)**2))
        print(f"\nMAE:  {mae*1000:.1f} meV/atom")
        print(f"RMSE: {rmse*1000:.1f} meV/atom")
        print(f"Max error: {np.max(np.abs(mlip_arr - dft_arr))*1000:.1f} meV/atom")


if __name__ == "__main__":
    compare_mlip_dft()
```

## Key Parameters

### MACE-MP-0

| Parameter | Options | Notes |
|---|---|---|
| `model` | `"small"`, `"medium"`, `"large"` | `"medium"` is the recommended default |
| `device` | `"cpu"`, `"cuda"` | Use `"cpu"` in this container (no GPU) |
| `default_dtype` | `"float32"`, `"float64"` | Use `"float64"` for relaxation and EOS fitting |
| `dispersion` | `True` / `False` | Enable D3 dispersion correction (useful for molecular crystals, layered materials) |

### CHGNet

| Parameter | Options | Notes |
|---|---|---|
| `use_device` | `"cpu"`, `"cuda"` | Use `"cpu"` in this container |
| Model is loaded automatically | -- | Downloads on first use (~50 MB) |

### M3GNet

| Parameter | Options | Notes |
|---|---|---|
| Model name | `"M3GNet-MP-2021.2.8-DIRECT-PES"` | Standard universal potential |
| `stress_weight` | float | Weight for stress in multi-task loss (default set by model) |

### SevenNet

| Parameter | Options | Notes |
|---|---|---|
| Model name | `"7net-0"` | Standard SevenNet-0 model |
| `device` | `"cpu"`, `"cuda"` | Use `"cpu"` |

### ASE relaxation parameters

| Parameter | Description | Typical Value |
|---|---|---|
| `fmax` | Force convergence threshold (eV/A) | 0.01 (tight), 0.05 (loose) |
| `steps` | Maximum optimization steps | 200-500 |
| Optimizer | BFGS, FIRE, LBFGS | BFGS (default), FIRE (for difficult cases) |
| `ExpCellFilter` | Enables variable-cell relaxation | Always use for cell optimization |

## Interpreting Results

### Energy
- Reported in eV (total) or eV/atom
- Absolute values differ between MLIPs (different reference states)
- Compare **relative** energies (formation energy differences, E-V curves) rather than absolute values
- Typical MLIP accuracy: ~10-30 meV/atom vs. DFT for well-represented systems

### Forces
- Units: eV/Angstrom
- Used for relaxation convergence (fmax criterion)
- Typical MLIP accuracy: ~50-100 meV/A vs. DFT

### Stress
- ASE returns Voigt notation: [xx, yy, zz, yz, xz, xy] in eV/A^3
- Convert to GPa: multiply by 160.2177 (1 eV/A^3 = 160.2177 GPa)
- Used for pressure and elastic constant calculations

### Lattice constant
- Compare to experimental values
- MLIPs trained on PBE-DFT data inherit PBE's systematic overbinding (~1-2%)
- GGA (PBE) typically overestimates lattice constants by 1-3%

### Bulk modulus
- Sensitive to the curvature of the E-V curve
- Typical accuracy: 5-20% vs. experiment
- Use at least 7-11 points in the E-V curve for reliable fitting

## Common Issues

### Extrapolation to unseen chemistry

**Problem**: MLIPs are interpolators. If your system is far from the training data (unusual oxidation states, extreme pressures, exotic compositions), predictions may be unreliable or the model may produce unphysical results.

**Solution**: Always validate against DFT for a few representative configurations. Check if the elements and structural motifs are well-represented in the training set (Materials Project).

### Rare elements

**Problem**: Training data for elements like Tc, Pm, Ac, or most actinides is extremely limited. Predictions for these elements will be unreliable.

**Solution**: Check the Materials Project for how many structures containing your element exist. If fewer than ~50 training structures, treat MLIP results with extreme caution.

### Numerical precision

**Problem**: `float32` precision can cause noisy forces, making relaxation fail to converge or giving jagged E-V curves.

**Solution**: Use `default_dtype="float64"` for MACE. For CHGNet/M3GNet, forces are typically stable in float32, but switch to float64 if you observe convergence issues.

### Model download failures

**Problem**: On first use, models are downloaded from the internet. In air-gapped environments, this fails.

**Solution**: Pre-download models before going offline:
```python
# Pre-cache MACE model
from mace.calculators import mace_mp
calc = mace_mp(model="medium", device="cpu")
# Model is now cached in ~/.cache/mace/
```

### Memory issues with large systems

**Problem**: MLIPs scale with the number of atoms. Systems with >10,000 atoms may exhaust RAM.

**Solution**:
- Use the `"small"` model variant (MACE) for large systems
- Reduce the neighbor list cutoff if the model supports it
- For very large systems, use LAMMPS with the MACE or SevenNet pair style

### Comparing energies across different MLIPs

**Problem**: Different MLIPs use different reference energies. You cannot directly compare absolute energies from MACE vs. CHGNet.

**Solution**: Compare relative energies only. For example, compute formation energies or energy differences between polymorphs using each MLIP separately, then compare those relative values.

### Relaxation oscillation or divergence

**Problem**: Structure relaxation oscillates or energy increases.

**Solution**:
- Switch optimizer: try `FIRE` instead of `BFGS`
- Reduce the initial step size: `BFGS(atoms, maxstep=0.05)`
- Check for unphysical initial geometry (overlapping atoms, extremely distorted cells)
- Use a looser `fmax` first (0.1), then tighten (0.01)

```python
from ase.optimize import FIRE
opt = FIRE(atoms, logfile="/tmp/fire_relax.log", maxstep=0.1)
opt.run(fmax=0.05, steps=500)
```
