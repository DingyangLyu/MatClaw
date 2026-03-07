# Surface Reaction Pathway Analysis

## When to Use

- Build free energy diagrams for catalytic reaction mechanisms on surfaces
- Compute reaction energetics with ZPE and thermal corrections for intermediates
- Calculate overpotentials for electrocatalytic reactions (OER, HER, ORR, CO2RR)
- Construct volcano plots to compare catalytic activity across materials
- Identify scaling relations between adsorption energies of reaction intermediates
- Screen catalyst surfaces for optimal activity using thermodynamic descriptors

## Method Selection

| Criterion | MACE Screening (Method A) | QE DFT (Method B) | VASP (Method C) |
|---|---|---|---|
| Speed | Minutes per intermediate | Hours per intermediate | Hours per intermediate |
| Accuracy | Good for trends and ranking | Publication quality | Publication quality |
| Free energy corrections | ASE thermochemistry | ASE thermochemistry from QE frequencies | VASPKIT 501/502 |
| Volcano plots | Fast multi-surface screening | Validate key points | Validate key points |
| Best for | Catalyst screening, scaling relations | Final publication energetics | VASP-ecosystem workflows |
| Limitations | ML potential accuracy varies | Expensive for many surfaces | Requires VASP license |

**Recommended workflow**: Screen many surfaces with MACE (Method A) to identify trends and build volcano plots. Refine the most promising candidates with QE DFT (Method B) or VASP (Method C) including ZPE and thermal corrections for publication-quality free energy diagrams.

## Prerequisites

- pymatgen (SlabGenerator, AdsorbateSiteFinder, Molecule)
- ASE + mace-torch (Method A)
- ASE thermochemistry (`ase.thermochemistry.HarmonicThermo`, `ase.thermochemistry.IdealGasThermo`)
- ASE vibrations (`ase.vibrations.Vibrations`)
- Quantum ESPRESSO pw.x (Method B)
- SSSP pseudopotentials for QE
- numpy, scipy, matplotlib for analysis

## Detailed Steps

### Free Energy Diagram Framework

The Gibbs free energy of each intermediate is:

```
G = E_DFT + ZPE + integral(Cp dT) - T*S
```

For adsorbates on surfaces (harmonic approximation):
```
G_ads = E_DFT + ZPE_ads - T*S_vib
```

For gas-phase molecules (ideal gas approximation):
```
G_gas = E_DFT + ZPE + H_trans + H_rot + H_vib - T*(S_trans + S_rot + S_vib)
```

For electrochemical reactions (computational hydrogen electrode, CHE):
```
G(H+ + e-) = 0.5 * G(H2) - eU
```
where U is the electrode potential vs. SHE.

### Method A: MACE -- Complete Electrocatalysis Workflow

#### Hydrogen Evolution Reaction (HER) Free Energy Diagram

```python
#!/usr/bin/env python3
"""
HER free energy diagram on multiple metal surfaces.
Uses MACE for rapid screening + ASE thermochemistry for thermal corrections.

HER mechanism (Volmer-Heyrovsky or Volmer-Tafel):
  H+ + e- + * -> H*            (Volmer step)
  H* + H+ + e- -> H2 + *       (Heyrovsky step)

Descriptor: dG_H* (hydrogen adsorption free energy)
  dG_H* = E(slab+H) - E(slab) - 0.5*E(H2) + dZPE - T*dS
  dZPE ~ 0.04 eV (empirical for H on metals)
  T*dS ~ -0.20 eV at 300 K (loss of gas-phase H2 entropy)
  Net thermal correction: dG_H* ~ dE_H + 0.24 eV

Optimal catalyst: dG_H* = 0 (top of volcano)
"""

import numpy as np
import warnings
warnings.filterwarnings("ignore")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from pymatgen.core import Structure, Lattice, Molecule
from pymatgen.core.surface import SlabGenerator
from pymatgen.analysis.adsorption import AdsorbateSiteFinder
from pymatgen.io.ase import AseAtomsAdaptor
from ase.optimize import BFGS
from ase.filters import FrechetCellFilter
from ase.constraints import FixAtoms
from ase.vibrations import Vibrations
from ase.thermochemistry import HarmonicThermo, IdealGasThermo
from ase.build import molecule as ase_molecule
from mace.calculators import mace_mp
import json

adaptor = AseAtomsAdaptor()

def get_calc():
    return mace_mp(model="medium", dispersion=False, default_dtype="float64")

# ============================================================
# 1. Compute gas-phase H2 reference with full thermochemistry
# ============================================================
print("=== Gas-phase H2 reference ===")
h2 = ase_molecule("H2")
h2.center(vacuum=10.0)
h2.calc = get_calc()
BFGS(h2, logfile=None).run(fmax=0.001, steps=100)
e_h2 = h2.get_potential_energy()

# Vibrational frequency of H2
vib_h2 = Vibrations(h2, name="vib_h2", delta=0.01)
vib_h2.run()
freq_h2 = vib_h2.get_energies()

# Keep only the single stretch mode (3N-5 = 1 mode for linear diatomic)
real_pos = sorted([f.real for f in freq_h2 if np.isreal(f) and f.real > 0.01],
                  reverse=True)
vib_energy_h2 = real_pos[:1]

thermo_h2 = IdealGasThermo(
    vib_energies=vib_energy_h2,
    potentialenergy=e_h2,
    atoms=h2,
    geometry="linear",
    symmetrynumber=2,
    spin=0,
)

T = 298.15
P = 101325.0
G_h2 = thermo_h2.get_gibbs_energy(T, P, verbose=False)
ZPE_h2 = sum(vib_energy_h2) / 2
print(f"E_H2 = {e_h2:.6f} eV")
print(f"ZPE_H2 = {ZPE_h2:.4f} eV")
print(f"G_H2(298K) = {G_h2:.4f} eV")

vib_h2.clean()

# ============================================================
# 2. Screen HER on multiple metal surfaces
# ============================================================
# Define metal surfaces to screen
metals = {
    "Pt": {"spacegroup": "Fm-3m", "a": 3.924, "species": ["Pt"]},
    "Pd": {"spacegroup": "Fm-3m", "a": 3.890, "species": ["Pd"]},
    "Cu": {"spacegroup": "Fm-3m", "a": 3.615, "species": ["Cu"]},
    "Au": {"spacegroup": "Fm-3m", "a": 4.078, "species": ["Au"]},
    "Ag": {"spacegroup": "Fm-3m", "a": 4.086, "species": ["Ag"]},
    "Ni": {"spacegroup": "Fm-3m", "a": 3.524, "species": ["Ni"]},
}

MILLER = (1, 1, 1)
H_adsorbate = Molecule(["H"], [[0, 0, 0]])

her_results = {}

for metal_name, metal_info in metals.items():
    print(f"\n=== {metal_name}(111) ===")

    # Build and relax bulk
    bulk_struct = Structure.from_spacegroup(
        metal_info["spacegroup"],
        lattice=Lattice.cubic(metal_info["a"]),
        species=metal_info["species"],
        coords=[[0.0, 0.0, 0.0]],
    )
    atoms_bulk = adaptor.get_atoms(bulk_struct)
    atoms_bulk.calc = get_calc()
    ecf = FrechetCellFilter(atoms_bulk)
    BFGS(ecf, logfile=None).run(fmax=0.005, steps=200)
    bulk_relaxed = adaptor.get_structure(atoms_bulk)

    # Generate (111) slab (3x3 supercell for low H coverage)
    slabgen = SlabGenerator(
        bulk_relaxed, MILLER,
        min_slab_size=10.0,
        min_vacuum_size=18.0,
        center_slab=True,
        primitive=True,
    )
    slabs = slabgen.get_slabs(symmetrize=False)
    if not slabs:
        print(f"  No slab generated for {metal_name}")
        continue

    slab = slabs[0]
    slab.make_supercell([3, 3, 1])
    n_slab = len(slab)

    # Relax clean slab
    atoms_slab = adaptor.get_atoms(slab)
    atoms_slab.calc = get_calc()
    z = atoms_slab.get_positions()[:, 2]
    fix = z < np.median(z)
    atoms_slab.set_constraint(FixAtoms(mask=fix))
    BFGS(atoms_slab, logfile=None).run(fmax=0.01, steps=300)
    e_slab = atoms_slab.get_potential_energy()
    slab_relaxed = adaptor.get_structure(atoms_slab)

    # Add H at preferred site (fcc hollow for most (111) metals)
    asf = AdsorbateSiteFinder(slab_relaxed)
    adslabs = asf.generate_adsorption_structures(
        H_adsorbate,
        repeat=[1, 1, 1],
        find_args={"distance": 1.5},
    )

    if not adslabs:
        print(f"  No H adslab for {metal_name}")
        continue

    adslab = adslabs[0]
    atoms_ads = adaptor.get_atoms(adslab)
    atoms_ads.calc = get_calc()
    z_a = atoms_ads.get_positions()[:, 2]
    fix_a = np.zeros(len(atoms_ads), dtype=bool)
    fix_a[:n_slab] = z_a[:n_slab] < np.median(z_a[:n_slab])
    atoms_ads.set_constraint(FixAtoms(mask=fix_a))
    BFGS(atoms_ads, logfile=None).run(fmax=0.01, steps=300)
    e_adslab = atoms_ads.get_potential_energy()

    # Adsorption energy
    dE_H = e_adslab - e_slab - 0.5 * e_h2

    # Compute ZPE for adsorbed H (vibrate only H atom)
    h_idx = [n_slab]  # H is the last atom
    fix_slab_all = list(range(n_slab))
    atoms_ads.set_constraint(FixAtoms(indices=fix_slab_all))

    vib_h = Vibrations(atoms_ads, indices=h_idx, name=f"vib_H_{metal_name}",
                       delta=0.01)
    vib_h.run()
    freq_h = vib_h.get_energies()
    real_freqs = [f.real for f in freq_h if np.isreal(f) and f.real > 0.005]

    if real_freqs:
        thermo_h = HarmonicThermo(vib_energies=real_freqs)
        zpe_h = thermo_h.get_ZPE_correction()
        ts_h = T * thermo_h.get_entropy(T, verbose=False)
    else:
        zpe_h = 0.0
        ts_h = 0.0

    # dG_H* = dE_H + (ZPE_H* - 0.5*ZPE_H2) - T*(S_H* - 0.5*S_H2)
    # Approximate: use computed values
    dZPE = zpe_h - 0.5 * ZPE_h2
    # For gas H2 entropy at 298 K: S_H2 ~ 0.00135 eV/K -> T*S = 0.403 eV
    TS_h2_gas = 0.403  # approximate T*S for H2 at 298K, 1 atm
    dTS = ts_h - 0.5 * TS_h2_gas

    dG_H = dE_H + dZPE - dTS

    her_results[metal_name] = {
        "dE_H_eV": float(dE_H),
        "ZPE_H_ads_eV": float(zpe_h),
        "dZPE_eV": float(dZPE),
        "dG_H_eV": float(dG_H),
    }

    print(f"  dE_H = {dE_H:.4f} eV")
    print(f"  ZPE_H* = {zpe_h:.4f} eV, dZPE = {dZPE:.4f} eV")
    print(f"  dG_H* = {dG_H:.4f} eV")

    vib_h.clean()

# ============================================================
# 3. Summary and free energy diagrams
# ============================================================
print("\n" + "=" * 60)
print("HER SCREENING SUMMARY")
print("=" * 60)
print(f"{'Metal':<8} {'dE_H (eV)':<12} {'dG_H* (eV)':<12} {'Activity'}")
print("-" * 45)
for metal, data in sorted(her_results.items(), key=lambda x: abs(x[1]["dG_H_eV"])):
    activity = "GOOD" if abs(data["dG_H_eV"]) < 0.2 else (
        "moderate" if abs(data["dG_H_eV"]) < 0.5 else "poor")
    print(f"{metal:<8} {data['dE_H_eV']:<12.4f} {data['dG_H_eV']:<12.4f} {activity}")

# ============================================================
# 4. Plot free energy diagram for each metal
# ============================================================
fig, ax = plt.subplots(figsize=(10, 6))

colors = plt.cm.Set1(np.linspace(0, 0.8, len(her_results)))
step_labels = [r"H$^+$ + e$^-$ + *", r"H*", r"$\frac{1}{2}$H$_2$ + *"]

for (metal, data), color in zip(
    sorted(her_results.items(), key=lambda x: x[1]["dG_H_eV"]),
    colors
):
    dG = data["dG_H_eV"]
    # Free energy levels: initial (0), H* (dG_H), product (0)
    levels = [0.0, dG, 0.0]
    x_positions = [0, 1, 2]
    width = 0.3

    for i, (x, g) in enumerate(zip(x_positions, levels)):
        ax.plot([x - width, x + width], [g, g],
                color=color, linewidth=2.5)
        if i < len(levels) - 1:
            ax.plot([x + width, x + 1 - width],
                    [levels[i], levels[i + 1]],
                    color=color, linewidth=1, linestyle="--", alpha=0.4)

    ax.plot([], [], color=color, linewidth=2.5,
            label=f"{metal} (dG = {dG:.2f} eV)")

ax.set_xticks([0, 1, 2])
ax.set_xticklabels(step_labels, fontsize=12)
ax.set_ylabel("Free energy (eV)", fontsize=13)
ax.set_title("HER Free Energy Diagram on (111) Metal Surfaces", fontsize=14)
ax.axhline(y=0, color="gray", linewidth=0.5)
ax.legend(fontsize=9, loc="upper right")
ax.grid(axis="y", alpha=0.3)
fig.tight_layout()
fig.savefig("her_free_energy_diagram.png", dpi=150)
print("\nPlot saved to her_free_energy_diagram.png")

with open("her_screening_results.json", "w") as f:
    json.dump(her_results, f, indent=2)
print("Results saved to her_screening_results.json")
```

#### Oxygen Evolution Reaction (OER) Free Energy Diagram

```python
#!/usr/bin/env python3
"""
OER free energy diagram using the computational hydrogen electrode (CHE).
Uses MACE for energetics + ASE thermochemistry for corrections.

OER 4-electron mechanism (associative pathway):
  * + H2O -> OH* + H+ + e-             (Step 1)
  OH* -> O* + H+ + e-                   (Step 2)
  O* + H2O -> OOH* + H+ + e-            (Step 3)
  OOH* -> * + O2 + H+ + e-              (Step 4)

Overall: 2H2O -> O2 + 4H+ + 4e-   (dG_rxn = 4 * 1.23 = 4.92 eV)

Key descriptors: dG_O*, dG_OH*
Overpotential: eta = max(dGi) / e - 1.23 V
"""

import numpy as np
import warnings
warnings.filterwarnings("ignore")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from pymatgen.core import Structure, Lattice, Molecule
from pymatgen.core.surface import SlabGenerator
from pymatgen.analysis.adsorption import AdsorbateSiteFinder
from pymatgen.io.ase import AseAtomsAdaptor
from ase.optimize import BFGS
from ase.filters import FrechetCellFilter
from ase.constraints import FixAtoms
from ase.build import molecule as ase_molecule
from mace.calculators import mace_mp
import json

adaptor = AseAtomsAdaptor()

def get_calc():
    return mace_mp(model="medium", dispersion=False, default_dtype="float64")

# ============================================================
# 1. Gas-phase references
# ============================================================
print("=== Gas-phase references ===")

# H2O
h2o = ase_molecule("H2O")
h2o.center(vacuum=10.0)
h2o.calc = get_calc()
BFGS(h2o, logfile=None).run(fmax=0.001, steps=100)
e_h2o = h2o.get_potential_energy()

# H2
h2 = ase_molecule("H2")
h2.center(vacuum=10.0)
h2.calc = get_calc()
BFGS(h2, logfile=None).run(fmax=0.001, steps=100)
e_h2 = h2.get_potential_energy()

print(f"E_H2O = {e_h2o:.6f} eV")
print(f"E_H2  = {e_h2:.6f} eV")

# Standard ZPE and thermal corrections (empirical values widely used)
# These can be computed explicitly using ASE thermochemistry as shown
# in the thermal-corrections skill
dG_corr = {
    "OH":  0.35,   # ZPE + thermal correction for OH*
    "O":   0.05,   # ZPE + thermal correction for O*
    "OOH": 0.40,   # ZPE + thermal correction for OOH*
    "H2O": 0.56,   # G - E for gas-phase H2O at 298K, 1 atm
    "H2":  0.27,   # G - E for gas-phase H2 at 298K, 1 atm
}

# G_H2O (gas) and G_H2 (gas) including thermal corrections
G_h2o = e_h2o + dG_corr["H2O"]
G_h2 = e_h2 + dG_corr["H2"]

# Standard free energy of water formation: G_H2O(l) ~ G_H2O(g) - 0.0
# (At 298K, liquid water free energy correction is small; we use gas value)

# ============================================================
# 2. Surface calculations: example IrO2(110)
# ============================================================
print("\n=== IrO2(110) OER ===")

# Build IrO2 rutile structure
iro2 = Structure.from_spacegroup(
    "P4_2/mnm",
    lattice=Lattice.tetragonal(4.505, 3.159),
    species=["Ir", "O"],
    coords=[[0.0, 0.0, 0.0], [0.3056, 0.3056, 0.0]],
)

# Relax bulk
atoms_iro2 = adaptor.get_atoms(iro2)
atoms_iro2.calc = get_calc()
ecf = FrechetCellFilter(atoms_iro2)
BFGS(ecf, logfile=None).run(fmax=0.005, steps=300)
iro2_relaxed = adaptor.get_structure(atoms_iro2)

# Generate (110) slab
slabgen = SlabGenerator(
    iro2_relaxed, (1, 1, 0),
    min_slab_size=10.0,
    min_vacuum_size=18.0,
    center_slab=True,
    primitive=False,
)
slabs = slabgen.get_slabs(symmetrize=False)
if slabs:
    slab = slabs[0]
    slab.make_supercell([2, 1, 1])
    n_slab = len(slab)
    print(f"Slab: {slab.formula}, {n_slab} atoms")

    # Relax clean slab
    atoms_slab = adaptor.get_atoms(slab)
    atoms_slab.calc = get_calc()
    z = atoms_slab.get_positions()[:, 2]
    fix = z < np.median(z)
    atoms_slab.set_constraint(FixAtoms(mask=fix))
    BFGS(atoms_slab, logfile=None).run(fmax=0.01, steps=500)
    e_slab = atoms_slab.get_potential_energy()
    slab_relaxed = adaptor.get_structure(atoms_slab)

    # Define adsorbates for each OER intermediate
    adsorbates = {
        "OH":  Molecule(["O", "H"], [[0, 0, 0], [0, 0, 0.97]]),
        "O":   Molecule(["O"], [[0, 0, 0]]),
        "OOH": Molecule(["O", "O", "H"], [[0, 0, 0], [0, 0, 1.21], [0, 0.94, 1.70]]),
    }

    e_intermediates = {}
    asf = AdsorbateSiteFinder(slab_relaxed)

    for name, mol in adsorbates.items():
        print(f"\n  --- {name}* ---")
        adslabs = asf.generate_adsorption_structures(
            mol, repeat=[1, 1, 1],
            find_args={"distance": 1.8},
        )
        if not adslabs:
            print(f"    No adslab generated for {name}")
            continue

        adslab = adslabs[0]
        atoms_a = adaptor.get_atoms(adslab)
        atoms_a.calc = get_calc()
        z_a = atoms_a.get_positions()[:, 2]
        fix_a = np.zeros(len(atoms_a), dtype=bool)
        fix_a[:n_slab] = z_a[:n_slab] < np.median(z_a[:n_slab])
        atoms_a.set_constraint(FixAtoms(mask=fix_a))
        BFGS(atoms_a, logfile=None).run(fmax=0.01, steps=500)
        e_a = atoms_a.get_potential_energy()
        e_intermediates[name] = e_a
        print(f"    E({name}*) = {e_a:.6f} eV")

    # ============================================================
    # 3. Compute adsorption free energies
    # ============================================================
    # dG_OH* = G(OH*) - G(*) - G(H2O) + 0.5*G(H2)
    # dG_O*  = G(O*)  - G(*) - G(H2O) + G(H2)
    # dG_OOH* = G(OOH*) - G(*) - 2*G(H2O) + 1.5*G(H2)

    if all(k in e_intermediates for k in ["OH", "O", "OOH"]):
        dG_OH = (e_intermediates["OH"] + dG_corr["OH"]) - e_slab - G_h2o + 0.5 * G_h2
        dG_O = (e_intermediates["O"] + dG_corr["O"]) - e_slab - G_h2o + G_h2
        dG_OOH = (e_intermediates["OOH"] + dG_corr["OOH"]) - e_slab - 2 * G_h2o + 1.5 * G_h2

        print(f"\n  Adsorption free energies:")
        print(f"    dG_OH*  = {dG_OH:.4f} eV")
        print(f"    dG_O*   = {dG_O:.4f} eV")
        print(f"    dG_OOH* = {dG_OOH:.4f} eV")

        # ============================================================
        # 4. Compute OER step free energies and overpotential
        # ============================================================
        # Step free energies at U = 0 V vs SHE:
        dG1 = dG_OH                          # * -> OH*
        dG2 = dG_O - dG_OH                   # OH* -> O*
        dG3 = dG_OOH - dG_O                  # O* -> OOH*
        dG4 = 4.92 - dG_OOH                  # OOH* -> O2 + *

        steps = [dG1, dG2, dG3, dG4]
        step_names = [
            r"$*$ $\to$ OH$*$",
            r"OH$*$ $\to$ O$*$",
            r"O$*$ $\to$ OOH$*$",
            r"OOH$*$ $\to$ O$_2$ + $*$",
        ]

        print(f"\n  OER Step Free Energies (U = 0 V):")
        for i, (dG, name) in enumerate(zip(steps, step_names)):
            print(f"    Step {i+1}: {dG:.4f} eV")

        # Potential-determining step and overpotential
        max_step = max(steps)
        pds_idx = steps.index(max_step)
        eta_oer = max_step / 1.0 - 1.23  # overpotential in V
        # More precisely: eta = max(dGi) - 1.23 at equilibrium potential

        print(f"\n  Potential-determining step: Step {pds_idx + 1}")
        print(f"  max(dGi) = {max_step:.4f} eV")
        print(f"  Overpotential eta_OER = {max(0, eta_oer):.4f} V")

        # ============================================================
        # 5. Plot OER free energy diagram at different potentials
        # ============================================================
        fig, ax = plt.subplots(figsize=(10, 6))

        potentials = [0.0, 1.23, 1.23 + max(0, eta_oer)]
        colors = ["#2ecc71", "#3498db", "#e74c3c"]
        labels_U = [f"U = {u:.2f} V" for u in potentials]

        state_labels = [
            r"$*$ + 2H$_2$O",
            r"OH$*$ + H$_2$O",
            r"O$*$ + H$_2$O",
            r"OOH$*$",
            r"$*$ + O$_2$",
        ]

        for U, color, label in zip(potentials, colors, labels_U):
            levels = [0.0]
            for dG in steps:
                levels.append(levels[-1] + dG - U)

            x_pos = np.arange(len(levels))
            width = 0.30

            for i, (x, g) in enumerate(zip(x_pos, levels)):
                ax.plot([x - width, x + width], [g, g],
                        color=color, linewidth=2.5)
                if i < len(levels) - 1:
                    ax.plot([x + width, x + 1 - width],
                            [levels[i], levels[i + 1]],
                            color=color, linewidth=1, linestyle="--", alpha=0.4)

            ax.plot([], [], color=color, linewidth=2.5, label=label)

        ax.set_xticks(x_pos)
        ax.set_xticklabels(state_labels, fontsize=10)
        ax.set_ylabel("Free energy (eV)", fontsize=13)
        ax.set_title("OER Free Energy Diagram on IrO2(110)", fontsize=14)
        ax.axhline(y=0, color="gray", linewidth=0.5)
        ax.legend(fontsize=10)
        ax.grid(axis="y", alpha=0.3)
        fig.tight_layout()
        fig.savefig("oer_free_energy_diagram.png", dpi=150)
        print("\nPlot saved to oer_free_energy_diagram.png")

        oer_results = {
            "surface": "IrO2(110)",
            "dG_OH_eV": float(dG_OH),
            "dG_O_eV": float(dG_O),
            "dG_OOH_eV": float(dG_OOH),
            "step_dG_eV": [float(s) for s in steps],
            "potential_determining_step": pds_idx + 1,
            "overpotential_V": float(max(0, eta_oer)),
        }
        with open("oer_results.json", "w") as f:
            json.dump(oer_results, f, indent=2)
        print("Results saved to oer_results.json")
```

#### Volcano Plot Construction

```python
#!/usr/bin/env python3
"""
Construct a volcano plot for OER activity.
Uses the scaling relation: dG_OOH* = dG_OH* + 3.2 (approximately).
Plots overpotential vs. dG_OH* (or dG_O* - dG_OH*) for multiple surfaces.
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import json

# ============================================================
# 1. Scaling relations and volcano framework
# ============================================================
# Universal scaling relation for OER:
#   dG_OOH* ~ dG_OH* + 3.2 (+/- 0.2 eV)
#
# This means we can parameterize the entire OER free energy diagram
# using only dG_OH* and dG_O* as independent variables.
#
# OER step free energies:
#   dG1 = dG_OH*
#   dG2 = dG_O* - dG_OH*
#   dG3 = dG_OOH* - dG_O* ~ (dG_OH* + 3.2) - dG_O*
#   dG4 = 4.92 - dG_OOH* ~ 4.92 - dG_OH* - 3.2 = 1.72 - dG_OH*
#
# Overpotential: eta = max(dG1, dG2, dG3, dG4) - 1.23

SCALING_OFFSET = 3.2  # dG_OOH* = dG_OH* + SCALING_OFFSET

def compute_oer_overpotential(dG_OH, dG_O):
    """
    Compute OER overpotential from adsorption free energies.

    Parameters
    ----------
    dG_OH : float
        Free energy of OH* adsorption (eV).
    dG_O : float
        Free energy of O* adsorption (eV).

    Returns
    -------
    eta : float
        Overpotential (V).
    pds : int
        Potential-determining step (1-4).
    """
    dG_OOH = dG_OH + SCALING_OFFSET

    dG1 = dG_OH
    dG2 = dG_O - dG_OH
    dG3 = dG_OOH - dG_O
    dG4 = 4.92 - dG_OOH

    steps = [dG1, dG2, dG3, dG4]
    max_step = max(steps)
    pds = steps.index(max_step) + 1
    eta = max(0, max_step - 1.23)

    return eta, pds

# ============================================================
# 2. Generate volcano curve (1D: parameterize by dG_OH*)
# ============================================================
# Assume dG_O* - dG_OH* ~ 1.5 eV (typical for oxides)
# This is a simplification; real surfaces have variable dG_O - dG_OH

dG_OH_range = np.linspace(-0.5, 3.5, 200)
delta_OH_O = 1.5  # typical dG_O - dG_OH for oxide surfaces

eta_values = []
for dG_OH in dG_OH_range:
    dG_O = dG_OH + delta_OH_O
    eta, _ = compute_oer_overpotential(dG_OH, dG_O)
    eta_values.append(eta)

# ============================================================
# 3. Plot volcano with example data points
# ============================================================
# Example computed dG values for various oxide surfaces
# Replace these with your actual computed values
surface_data = {
    "IrO2(110)":  {"dG_OH": 1.60, "dG_O": 3.20, "color": "red"},
    "RuO2(110)":  {"dG_OH": 1.30, "dG_O": 2.90, "color": "blue"},
    "TiO2(110)":  {"dG_OH": 2.10, "dG_O": 4.00, "color": "green"},
    "MnO2(110)":  {"dG_OH": 1.70, "dG_O": 3.10, "color": "orange"},
    "Co3O4(311)": {"dG_OH": 1.50, "dG_O": 3.00, "color": "purple"},
}

fig, ax = plt.subplots(figsize=(10, 7))

# Plot volcano curve
ax.plot(dG_OH_range, eta_values, "k-", linewidth=2, label="Volcano (scaling relation)")

# Plot individual surfaces
for name, data in surface_data.items():
    eta_pt, pds = compute_oer_overpotential(data["dG_OH"], data["dG_O"])
    ax.plot(data["dG_OH"], eta_pt, "o", color=data["color"],
            markersize=12, markeredgecolor="black", linewidth=1.5,
            label=f"{name} (eta={eta_pt:.2f} V)", zorder=5)

# Mark the ideal point (zero overpotential)
ideal_dG_OH = 1.23  # at the volcano apex
ax.axvline(x=ideal_dG_OH, color="gray", linestyle=":", alpha=0.5,
           label=r"Ideal $\Delta G_{OH*}$")

# Annotations
ax.annotate("Bind too weakly\n(Step 1 limiting)",
            xy=(2.8, 1.5), fontsize=10, ha="center", color="gray")
ax.annotate("Bind too strongly\n(Step 4 limiting)",
            xy=(0.3, 1.5), fontsize=10, ha="center", color="gray")

ax.set_xlabel(r"$\Delta G_{OH*}$ (eV)", fontsize=14)
ax.set_ylabel("Overpotential (V)", fontsize=14)
ax.set_title("OER Volcano Plot", fontsize=15)
ax.set_ylim(-0.1, 3.0)
ax.invert_yaxis()
ax.legend(fontsize=9, loc="lower right")
ax.grid(True, alpha=0.3)
fig.tight_layout()
fig.savefig("oer_volcano_plot.png", dpi=150)
print("Volcano plot saved to oer_volcano_plot.png")

# ============================================================
# 4. 2D volcano (heatmap of eta vs dG_OH and dG_O)
# ============================================================
dG_OH_2d = np.linspace(0.0, 3.5, 100)
dG_O_2d = np.linspace(1.0, 5.0, 100)
OH_grid, O_grid = np.meshgrid(dG_OH_2d, dG_O_2d)
eta_grid = np.zeros_like(OH_grid)

for i in range(len(dG_O_2d)):
    for j in range(len(dG_OH_2d)):
        eta_grid[i, j], _ = compute_oer_overpotential(OH_grid[i, j], O_grid[i, j])

fig2, ax2 = plt.subplots(figsize=(9, 7))
c = ax2.contourf(OH_grid, O_grid, eta_grid, levels=20, cmap="RdYlGn_r")
plt.colorbar(c, ax=ax2, label="Overpotential (V)")

# Plot surfaces on the 2D map
for name, data in surface_data.items():
    ax2.plot(data["dG_OH"], data["dG_O"], "o", color="white",
             markersize=10, markeredgecolor="black", linewidth=1.5, zorder=5)
    ax2.annotate(name, (data["dG_OH"], data["dG_O"]),
                 textcoords="offset points", xytext=(8, 5),
                 fontsize=9, fontweight="bold")

# Scaling relation line
dG_OH_line = np.linspace(0, 3.5, 50)
ax2.plot(dG_OH_line, dG_OH_line + SCALING_OFFSET,
         "k--", linewidth=1.5, alpha=0.5, label="dG_O = dG_OH + 3.2")

ax2.set_xlabel(r"$\Delta G_{OH*}$ (eV)", fontsize=14)
ax2.set_ylabel(r"$\Delta G_{O*}$ (eV)", fontsize=14)
ax2.set_title("OER 2D Volcano Plot", fontsize=15)
ax2.legend(fontsize=10)
fig2.tight_layout()
fig2.savefig("oer_volcano_2d.png", dpi=150)
print("2D volcano plot saved to oer_volcano_2d.png")
```

#### Scaling Relations Analysis

```python
#!/usr/bin/env python3
"""
Compute and validate scaling relations between adsorption free energies
of OER/ORR intermediates.

Key scaling relations:
  dG_OOH* = dG_OH* + C    (C ~ 3.2 eV for oxides)
  dG_O*   = 2 * dG_OH* + C' (approximate, less universal)

These arise because OOH* and OH* bind through similar O-surface bonds.
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats
import json

# ============================================================
# 1. Example data: computed dG values for multiple surfaces
# ============================================================
# Replace with your actual computed values from MACE or DFT
data = {
    "IrO2(110)":   {"dG_OH": 1.60, "dG_O": 3.20, "dG_OOH": 4.80},
    "RuO2(110)":   {"dG_OH": 1.30, "dG_O": 2.90, "dG_OOH": 4.50},
    "TiO2(110)":   {"dG_OH": 2.10, "dG_O": 4.00, "dG_OOH": 5.30},
    "MnO2(110)":   {"dG_OH": 1.70, "dG_O": 3.10, "dG_OOH": 4.90},
    "Co3O4(311)":  {"dG_OH": 1.50, "dG_O": 3.00, "dG_OOH": 4.70},
    "NiOOH(001)":  {"dG_OH": 1.80, "dG_O": 3.50, "dG_OOH": 5.00},
    "Fe2O3(001)":  {"dG_OH": 1.90, "dG_O": 3.70, "dG_OOH": 5.10},
    "PtO2(110)":   {"dG_OH": 1.40, "dG_O": 3.05, "dG_OOH": 4.60},
}

# ============================================================
# 2. Fit scaling relations
# ============================================================
dG_OH_arr = np.array([d["dG_OH"] for d in data.values()])
dG_O_arr = np.array([d["dG_O"] for d in data.values()])
dG_OOH_arr = np.array([d["dG_OOH"] for d in data.values()])

# Linear fit: dG_OOH vs dG_OH
slope_ooh, intercept_ooh, r_ooh, p_ooh, se_ooh = stats.linregress(dG_OH_arr, dG_OOH_arr)
print(f"Scaling: dG_OOH* = {slope_ooh:.2f} * dG_OH* + {intercept_ooh:.2f}")
print(f"  R^2 = {r_ooh**2:.4f}, MAE = {np.mean(np.abs(dG_OOH_arr - (slope_ooh * dG_OH_arr + intercept_ooh))):.3f} eV")

# Linear fit: dG_O vs dG_OH
slope_o, intercept_o, r_o, p_o, se_o = stats.linregress(dG_OH_arr, dG_O_arr)
print(f"\nScaling: dG_O* = {slope_o:.2f} * dG_OH* + {intercept_o:.2f}")
print(f"  R^2 = {r_o**2:.4f}, MAE = {np.mean(np.abs(dG_O_arr - (slope_o * dG_OH_arr + intercept_o))):.3f} eV")

# ============================================================
# 3. Plot scaling relations
# ============================================================
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

# (a) dG_OOH vs dG_OH
x_fit = np.linspace(min(dG_OH_arr) - 0.2, max(dG_OH_arr) + 0.2, 50)
ax1.plot(x_fit, slope_ooh * x_fit + intercept_ooh, "r--", linewidth=1.5,
         label=f"Fit: slope={slope_ooh:.2f}, R$^2$={r_ooh**2:.3f}")
ax1.plot(x_fit, x_fit + 3.2, "k:", linewidth=1, alpha=0.5,
         label="Universal: dG_OOH = dG_OH + 3.2")

for name, d in data.items():
    ax1.plot(d["dG_OH"], d["dG_OOH"], "o", markersize=8, markeredgecolor="black")
    ax1.annotate(name.replace("(", "\n("), (d["dG_OH"], d["dG_OOH"]),
                 textcoords="offset points", xytext=(5, 5), fontsize=7)

ax1.set_xlabel(r"$\Delta G_{OH*}$ (eV)", fontsize=13)
ax1.set_ylabel(r"$\Delta G_{OOH*}$ (eV)", fontsize=13)
ax1.set_title("Scaling: OOH* vs OH*", fontsize=14)
ax1.legend(fontsize=9)
ax1.grid(True, alpha=0.3)

# (b) dG_O vs dG_OH
ax2.plot(x_fit, slope_o * x_fit + intercept_o, "b--", linewidth=1.5,
         label=f"Fit: slope={slope_o:.2f}, R$^2$={r_o**2:.3f}")

for name, d in data.items():
    ax2.plot(d["dG_OH"], d["dG_O"], "s", markersize=8, markeredgecolor="black")
    ax2.annotate(name.replace("(", "\n("), (d["dG_OH"], d["dG_O"]),
                 textcoords="offset points", xytext=(5, 5), fontsize=7)

ax2.set_xlabel(r"$\Delta G_{OH*}$ (eV)", fontsize=13)
ax2.set_ylabel(r"$\Delta G_{O*}$ (eV)", fontsize=13)
ax2.set_title("Scaling: O* vs OH*", fontsize=14)
ax2.legend(fontsize=9)
ax2.grid(True, alpha=0.3)

fig.tight_layout()
fig.savefig("scaling_relations.png", dpi=150)
print("\nScaling relations plot saved to scaling_relations.png")

# Save
scaling_results = {
    "OOH_vs_OH": {
        "slope": float(slope_ooh),
        "intercept": float(intercept_ooh),
        "R2": float(r_ooh**2),
    },
    "O_vs_OH": {
        "slope": float(slope_o),
        "intercept": float(intercept_o),
        "R2": float(r_o**2),
    },
    "surface_data": data,
}
with open("scaling_relations.json", "w") as f:
    json.dump(scaling_results, f, indent=2)
print("Results saved to scaling_relations.json")
```

### Method B: QE DFT -- Publication-Quality Free Energy Diagram

#### Step 1: Generate QE Inputs for All Intermediates

```python
#!/usr/bin/env python3
"""
Generate QE input files for all OER intermediates on a surface.
Produces inputs for: clean slab, OH*, O*, OOH*, gas-phase H2O and H2.
"""

import numpy as np
from pymatgen.core import Structure, Lattice, Molecule
from pymatgen.core.surface import SlabGenerator
from pymatgen.analysis.adsorption import AdsorbateSiteFinder
from pymatgen.io.pwscf import PWInput
from pathlib import Path

PSEUDO_DIR = "./pseudo"
ECUTWFC = 60.0
ECUTRHO = 480.0
Path(PSEUDO_DIR).mkdir(exist_ok=True)

# Pseudopotentials -- adjust filenames to match your SSSP library
pseudos = {
    "Ir": "Ir.pbe-n-kjpaw_psl.1.0.0.UPF",
    "O":  "O.pbe-n-kjpaw_psl.1.0.0.UPF",
    "H":  "H.pbe-kjpaw_psl.1.0.0.UPF",
}

system_params = {
    "ecutwfc": ECUTWFC,
    "ecutrho": ECUTRHO,
    "occupations": "smearing",
    "smearing": "cold",
    "degauss": 0.02,
}
electron_params = {"conv_thr": 1.0e-8, "mixing_beta": 0.3}

# Build IrO2(110) slab
iro2 = Structure.from_spacegroup(
    "P4_2/mnm",
    lattice=Lattice.tetragonal(4.505, 3.159),
    species=["Ir", "O"],
    coords=[[0.0, 0.0, 0.0], [0.3056, 0.3056, 0.0]],
)

slabgen = SlabGenerator(
    iro2, (1, 1, 0),
    min_slab_size=10.0,
    min_vacuum_size=18.0,
    center_slab=True,
    primitive=False,
)
slabs = slabgen.get_slabs(symmetrize=False)
slab = slabs[0]
slab.make_supercell([2, 1, 1])
kpts = (4, 4, 1)

# 1. Clean slab
control_relax = {
    "calculation": "relax",
    "restart_mode": "from_scratch",
    "pseudo_dir": PSEUDO_DIR,
    "outdir": "./tmp_slab",
    "prefix": "slab",
    "tprnfor": True,
    "forc_conv_thr": 1.0e-4,
}

pw = PWInput(slab, pseudo=pseudos, control=control_relax,
             system=system_params, electrons=electron_params,
             kpoints_grid=kpts)
pw.write_file("pw_slab_clean.in")
print(f"Written pw_slab_clean.in ({len(slab)} atoms)")

# 2. Gas-phase molecules
for mol_name, mol_struct in [
    ("H2O", Structure(Lattice.cubic(15.0), ["O", "H", "H"],
                      [[0.5, 0.5, 0.5],
                       [0.5 + 0.757/15, 0.5 + 0.586/15, 0.5],
                       [0.5 - 0.757/15, 0.5 + 0.586/15, 0.5]])),
    ("H2", Structure(Lattice.cubic(15.0), ["H", "H"],
                     [[0.5, 0.5, 0.5], [0.5, 0.5, 0.5 + 0.74/15]])),
]:
    pw_mol = PWInput(
        mol_struct, pseudo=pseudos,
        control=control_relax | {"outdir": f"./tmp_{mol_name}", "prefix": mol_name},
        system=system_params | {"degauss": 0.005, "smearing": "gaussian"},
        electrons=electron_params | {"mixing_beta": 0.4},
        kpoints_grid=(1, 1, 1),
    )
    pw_mol.write_file(f"pw_{mol_name}.in")
    print(f"Written pw_{mol_name}.in")

# 3. Adsorbate intermediates
asf = AdsorbateSiteFinder(slab)
adsorbates = {
    "OH":  Molecule(["O", "H"], [[0, 0, 0], [0, 0, 0.97]]),
    "O":   Molecule(["O"], [[0, 0, 0]]),
    "OOH": Molecule(["O", "O", "H"], [[0, 0, 0], [0, 0, 1.21], [0, 0.94, 1.70]]),
}

for ads_name, ads_mol in adsorbates.items():
    adslabs = asf.generate_adsorption_structures(
        ads_mol, repeat=[1, 1, 1],
        find_args={"distance": 1.8},
    )
    if adslabs:
        adslab = adslabs[0]
        pw_ads = PWInput(
            adslab, pseudo=pseudos,
            control=control_relax | {
                "outdir": f"./tmp_{ads_name}",
                "prefix": ads_name,
            },
            system=system_params,
            electrons=electron_params,
            kpoints_grid=kpts,
        )
        pw_ads.write_file(f"pw_{ads_name}_slab.in")
        print(f"Written pw_{ads_name}_slab.in ({len(adslab)} atoms)")

print("\nIMPORTANT: Add selective dynamics to fix bottom slab layers.")
print("After relaxation, run frequency calculations for ZPE corrections.")
print("See catalysis-electrochem/thermal-corrections/ for the frequency workflow.")
```

#### Step 2: Run QE and Compute Free Energies

```bash
#!/bin/bash
# Run QE calculations for all OER intermediates

NPROC=4

# Download pseudopotentials
mkdir -p pseudo
cd pseudo
wget -nc "https://pseudopotentials.quantum-espresso.org/upf_files/Ir.pbe-n-kjpaw_psl.1.0.0.UPF" 2>/dev/null
wget -nc "https://pseudopotentials.quantum-espresso.org/upf_files/O.pbe-n-kjpaw_psl.1.0.0.UPF" 2>/dev/null
wget -nc "https://pseudopotentials.quantum-espresso.org/upf_files/H.pbe-kjpaw_psl.1.0.0.UPF" 2>/dev/null
cd ..

# Run all calculations
for CALC in slab_clean H2O H2 OH_slab O_slab OOH_slab; do
    mkdir -p tmp_$(echo $CALC | sed 's/_slab//')
    echo "=== Running $CALC ==="
    mpirun -np $NPROC pw.x -in pw_${CALC}.in > pw_${CALC}.out 2>&1
    echo "$CALC: $(grep '!' pw_${CALC}.out | tail -1)"
done

echo ""
echo "After relaxation, run frequency calculations for each intermediate"
echo "to obtain ZPE and thermal corrections."
echo "See catalysis-electrochem/thermal-corrections/ skill for details."
```

#### Step 3: Parse Results and Build Free Energy Diagram

```python
#!/usr/bin/env python3
"""
Parse QE outputs for OER intermediates and build publication-quality
free energy diagram with ZPE and thermal corrections.
"""

import re
import numpy as np
import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def parse_qe_energy(filename):
    """Extract final total energy in eV."""
    energy = None
    with open(filename) as f:
        for line in f:
            if "!" in line and "total energy" in line:
                match = re.search(r"=\s+([-\d.]+)\s+Ry", line)
                if match:
                    energy = float(match.group(1)) * 13.605693123
    return energy


# Parse all energies
e_slab = parse_qe_energy("pw_slab_clean.out")
e_h2o = parse_qe_energy("pw_H2O.out")
e_h2 = parse_qe_energy("pw_H2.out")
e_OH = parse_qe_energy("pw_OH_slab.out")
e_O = parse_qe_energy("pw_O_slab.out")
e_OOH = parse_qe_energy("pw_OOH_slab.out")

print("=== DFT Energies ===")
print(f"E_slab  = {e_slab:.6f} eV")
print(f"E_H2O   = {e_h2o:.6f} eV")
print(f"E_H2    = {e_h2:.6f} eV")
print(f"E_OH*   = {e_OH:.6f} eV")
print(f"E_O*    = {e_O:.6f} eV")
print(f"E_OOH*  = {e_OOH:.6f} eV")

# Thermal corrections (from frequency calculations or empirical values)
# Replace these with your computed values from ASE HarmonicThermo/IdealGasThermo
# See catalysis-electrochem/thermal-corrections/ skill
dZPE_TS = {
    "OH":  0.35,   # dZPE - TdS for OH* (typical)
    "O":   0.05,   # dZPE - TdS for O*
    "OOH": 0.40,   # dZPE - TdS for OOH*
}
dG_corr_h2o = 0.56  # G - E for gas H2O at 298 K
dG_corr_h2 = 0.27   # G - E for gas H2 at 298 K

# Gas-phase free energies
G_h2o = e_h2o + dG_corr_h2o
G_h2 = e_h2 + dG_corr_h2

# Adsorption free energies
dG_OH = (e_OH + dZPE_TS["OH"]) - e_slab - G_h2o + 0.5 * G_h2
dG_O = (e_O + dZPE_TS["O"]) - e_slab - G_h2o + G_h2
dG_OOH = (e_OOH + dZPE_TS["OOH"]) - e_slab - 2 * G_h2o + 1.5 * G_h2

print(f"\n=== Adsorption Free Energies ===")
print(f"dG_OH*  = {dG_OH:.4f} eV")
print(f"dG_O*   = {dG_O:.4f} eV")
print(f"dG_OOH* = {dG_OOH:.4f} eV")

# OER step free energies
dG1 = dG_OH
dG2 = dG_O - dG_OH
dG3 = dG_OOH - dG_O
dG4 = 4.92 - dG_OOH

steps = [dG1, dG2, dG3, dG4]
pds_idx = np.argmax(steps)
eta = max(0, max(steps) - 1.23)

print(f"\n=== OER Step Free Energies ===")
for i, dg in enumerate(steps):
    marker = " <-- PDS" if i == pds_idx else ""
    print(f"  Step {i+1}: dG = {dg:.4f} eV{marker}")
print(f"\nOverpotential: eta = {eta:.4f} V")

# Plot
fig, ax = plt.subplots(figsize=(10, 6))

state_labels = [
    r"$*$ + 2H$_2$O",
    r"OH$*$ + H$_2$O",
    r"O$*$ + H$_2$O",
    r"OOH$*$",
    r"$*$ + O$_2$",
]

for U, color, label in [(0.0, "#2ecc71", "U = 0 V"),
                          (1.23, "#3498db", "U = 1.23 V (equil.)"),
                          (1.23 + eta, "#e74c3c", f"U = {1.23+eta:.2f} V (onset)")]:
    levels = [0.0]
    for dg in steps:
        levels.append(levels[-1] + dg - U)

    x = np.arange(len(levels))
    w = 0.30
    for i, (xi, g) in enumerate(zip(x, levels)):
        ax.plot([xi - w, xi + w], [g, g], color=color, linewidth=2.5)
        if i < len(levels) - 1:
            ax.plot([xi + w, xi + 1 - w], [levels[i], levels[i + 1]],
                    color=color, linewidth=1, linestyle="--", alpha=0.4)
    ax.plot([], [], color=color, linewidth=2.5, label=label)

ax.set_xticks(np.arange(len(state_labels)))
ax.set_xticklabels(state_labels, fontsize=10)
ax.set_ylabel("Free energy (eV)", fontsize=13)
ax.set_title("OER Free Energy Diagram (QE DFT)", fontsize=14)
ax.axhline(y=0, color="gray", linewidth=0.5)
ax.legend(fontsize=10)
ax.grid(axis="y", alpha=0.3)
fig.tight_layout()
fig.savefig("oer_free_energy_dft.png", dpi=150)
print("\nPlot saved to oer_free_energy_dft.png")

results = {
    "dG_OH_eV": float(dG_OH),
    "dG_O_eV": float(dG_O),
    "dG_OOH_eV": float(dG_OOH),
    "steps_dG_eV": [float(s) for s in steps],
    "pds": int(pds_idx + 1),
    "overpotential_V": float(eta),
}
with open("oer_dft_results.json", "w") as f:
    json.dump(results, f, indent=2)
print("Results saved to oer_dft_results.json")
```

### Method C: VASP Workflow (Future)

```
VASP OER/HER/ORR workflow (requires VASP license):

1. Slab + intermediate relaxation:
   IBRION = 2, ISIF = 2, NSW = 200
   PREC = Accurate, ENCUT = 400
   Selective dynamics to fix bottom layers
   KPOINTS: Nx Ny 1

2. Frequency calculations for ZPE:
   IBRION = 5, NSW = 1, NFREE = 2, POTIM = 0.015
   Selective dynamics: freeze slab, vibrate only adsorbate
   Post-process with VASPKIT 501 for adsorbate ZPE/thermal corrections

3. Gas-phase molecule frequencies:
   Same IBRION = 5 settings in large box
   KPOINTS: Gamma only
   Post-process with VASPKIT 502 for ideal gas thermodynamics

4. Dispersion corrections:
   IVDW = 11  (DFT-D3 BJ)
   Apply to ALL calculations consistently (slab, molecule, adslab)

5. Free energy diagram:
   G = E_VASP + ZPE + thermal - T*S (from VASPKIT outputs)
   Apply CHE: G(H+ + e-) = 0.5 * G(H2) - eU

6. d-band center (VASPKIT 503):
   Compute projected DOS on surface metal atoms
   d-band center = integral(E * PDOS_d dE) / integral(PDOS_d dE)
   Correlates with adsorption strength (Hammer-Norskov model)

7. Overpotential:
   eta = max(dG_step) - 1.23 V (for OER)
   eta = max(dG_step) (for ORR at U = 1.23 V)
```

## Key Parameters

| Parameter | Recommended Value | Notes |
|---|---|---|
| Slab thickness | 4-6 layers (8-14 A) | Same as adsorption energy calculations |
| Vacuum | 15-20 A | Prevent periodic interactions |
| Surface supercell | >= 2x2 (metals), >= 1x1 (oxides) | Low coverage limit for single adsorbate |
| Temperature | 298.15 K | Standard conditions; adjust for operating T |
| Pressure | 101325 Pa | Standard for gas-phase references |
| MACE fmax | 0.01 eV/A | Tight relaxation before frequency calculation |
| Vibration delta | 0.01 A | Finite displacement for frequency calculation |
| dG_OOH scaling | dG_OH + 3.2 eV | Approximate; validate with explicit calculation |
| Equilibrium potential (OER) | 1.23 V vs SHE | Standard water oxidation potential |
| Equilibrium potential (HER) | 0 V vs SHE | Standard hydrogen evolution potential |
| Equilibrium potential (ORR) | 1.23 V vs SHE | Same as OER (reverse reaction) |
| ZPE threshold | > 50 cm^-1 | Ignore near-zero frequencies (numerical noise) |
| Symmetry number | Molecule-dependent | H2: 2, H2O: 2, O2: 2 |

## Interpreting Results

1. **Overpotential**: The additional potential beyond equilibrium needed to make all reaction steps thermodynamically favorable. Lower eta = better catalyst. State-of-the-art OER catalysts: eta ~ 0.2-0.4 V. HER: eta < 0.1 V for Pt-group metals.
2. **Potential-determining step (PDS)**: The elementary step with the largest free energy change at the equilibrium potential. This limits the overall reaction rate.
3. **Volcano peak**: The optimal dG_H* for HER is 0 eV (thermoneutral). For OER, the ideal catalyst has all steps equal to 1.23 eV, but scaling relations prevent this (minimum theoretical overpotential ~ 0.37 V for OER).
4. **Scaling relations**: dG_OOH* = dG_OH* + 3.2 eV is robust for many oxide surfaces. Breaking this scaling (e.g., through surface modification, strain, or bifunctional sites) is the key to lowering overpotential.
5. **Thermal corrections magnitude**: ZPE ~ 0.3-0.5 eV for OH*, 0.05-0.1 eV for O*, 0.4-0.5 eV for OOH*. The entropy loss when gas-phase molecules adsorb is significant (T*S ~ 0.2-0.6 eV at 298 K).
6. **MACE vs. DFT**: MACE provides good qualitative trends for volcano ranking but may shift absolute dG values by 0.1-0.5 eV. Validate the top candidates with DFT.
7. **Solvation corrections**: In electrochemistry, explicit or implicit solvation can shift adsorption energies by 0.1-0.3 eV. For preliminary screening, gas-phase calculations are acceptable; for publication, consider solvation models.

## Common Issues

| Issue | Solution |
|---|---|
| Adsorbate desorbs or dissociates | Start with shorter distance; try different orientation; use gentler optimizer settings |
| OOH* dissociates into O* + OH* | Use constrained optimization; try different initial geometry; this may indicate O* + OH* is more stable |
| Negative overpotential | Check signs of dG values; verify reference energies; ensure consistent methods |
| Scaling relation does not hold | Expected for non-oxide surfaces or modified surfaces; compute all intermediates explicitly |
| ZPE frequencies have imaginary modes | Re-relax structure with tighter fmax (< 0.005 eV/A) before frequency calculation |
| Large discrepancy between MACE and DFT | Expected for complex oxide surfaces; use DFT for final values |
| Volcano plot shows no good catalysts | The activity may be limited by the scaling relation; look for materials that break the scaling |
| Free energy diagram steps do not sum to 4.92 eV | Verify all reference energies and thermal corrections; the total must equal 2*1.23 + 2*1.23 = 4.92 eV for OER |
| Coverage effects change the picture | At high coverage, lateral interactions modify dG; use explicit coverage models for operating conditions |
| Solvation shifts the volcano | Add implicit solvation (e.g., VASPsol, ENVIRON for QE) or explicit water molecules for key intermediates |
