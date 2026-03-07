# Configuration Coordinate Diagram

## When to Use

- Evaluate non-radiative recombination rates at defects (Shockley-Read-Hall recombination)
- Assess whether a defect acts as a luminescence quencher or an efficient recombination center
- Compute the classical energy barrier for carrier capture at a defect
- Determine the zero-phonon line (ZPL), Franck-Condon relaxation energies, and Huang-Rhys factor
- Compare non-radiative vs. radiative transition rates for defect engineering (e.g., quantum emitters, LED efficiency droop)
- Predict photoluminescence Stokes shift and lineshape broadening

## Method Selection

```
Need a configuration coordinate diagram?

  What level of accuracy?

  Quick screening / trends across many defects?
    --> ASE + MACE (Method A): minutes, no electronic structure, neutral approximation
        Good for: structural relaxation comparison, approximate DeltaQ, qualitative CCD shape
        Limitation: no true charged-state energetics (MACE has no electrons)

  Publication-quality CCD with correct charge states?
    --> QE DFT (Method B): hours to days, full electronic structure
        Required for: accurate ZPL, barriers, Huang-Rhys factors, non-radiative rates
        Handles: charged supercells with tot_charge, Freysoldt corrections

  Already have relaxed structures from another code?
    --> Use either method for single-shot energies on interpolated geometries
```

## Prerequisites

- pymatgen (structure manipulation, defect generation)
- ASE + mace-torch (Method A)
- Quantum ESPRESSO pw.x (Method B)
- SSSP pseudopotentials for QE
- numpy, scipy, matplotlib for analysis and plotting
- Optional: `pip install pymatgen-analysis-defects` for advanced defect workflows
- Two charge states of a defect must be defined (e.g., V_O^0 and V_O^{2+})

## Detailed Steps

### Method A: ASE + MACE (Fast Screening)

#### Complete Workflow: Configuration Coordinate Diagram

```python
#!/usr/bin/env python3
"""
Configuration Coordinate Diagram using ASE + MACE.

Example system: Oxygen vacancy in MgO, comparing q=0 and q=+2 charge states.

NOTE: MACE is a classical potential with no electronic degrees of freedom.
It cannot truly represent different charge states. This script approximates
the two "charge states" by using the same potential for both, which means
the two parabolas will share the same PES and the CCD is only useful for:
  - Validating the workflow / interpolation machinery
  - Estimating DeltaQ (mass-weighted displacement) between two relaxed geometries
  - Structural screening before running full DFT

For physically meaningful CCDs with real charge-state energetics, use Method B (QE DFT).

Workflow:
  1. Build defect supercell
  2. Relax the defect in two configurations (mimicking two charge states)
  3. Compute DeltaQ (mass-weighted displacement)
  4. Interpolate/extrapolate structures along the configuration coordinate
  5. Compute single-shot energies at each distortion for both "states"
  6. Plot the CCD and extract key quantities
"""

import numpy as np
from pymatgen.core import Structure, Element
from pymatgen.io.ase import AseAtomsAdaptor
from ase.optimize import BFGS
from ase.constraints import FixAtoms
from mace.calculators import mace_mp
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from pathlib import Path
import json

# ============================================================
# 1. Configuration
# ============================================================
SUPERCELL_SIZE = 3        # NxNxN supercell
FMAX = 0.005              # eV/A convergence for relaxation
# Fractional distortions along Q: 0 = state1 equilibrium, 1 = state2 equilibrium
# Negative values extrapolate beyond state1; >1 extrapolates beyond state2
DISTORTION_FRACTIONS = [-0.2, -0.15, -0.1, -0.05, 0.0, 0.05, 0.1, 0.15,
                         0.2, 0.4, 0.6, 0.8, 0.85, 0.9, 0.95, 1.0,
                         1.05, 1.1, 1.15, 1.2]

# ============================================================
# 2. Build the host structure and defect supercell
# ============================================================
primitive = Structure.from_spacegroup(
    "Fm-3m",
    lattice=[[4.212, 0, 0], [0, 4.212, 0], [0, 0, 4.212]],
    species=["Mg", "O"],
    coords=[[0.0, 0.0, 0.0], [0.5, 0.5, 0.5]],
)

bulk_sc = primitive.copy()
bulk_sc.make_supercell([SUPERCELL_SIZE] * 3)
print(f"Bulk supercell: {bulk_sc.formula}, {len(bulk_sc)} atoms")

# Create oxygen vacancy
defect_sc = bulk_sc.copy()
o_indices = [i for i, s in enumerate(defect_sc) if s.specie == Element("O")]
vac_idx = o_indices[len(o_indices) // 2]  # pick a central O atom
print(f"Removing O at site {vac_idx}")
defect_sc.remove_sites([vac_idx])

# ============================================================
# 3. Set up MACE calculator
# ============================================================
calc = mace_mp(model="medium", dispersion=False, default_dtype="float64")
adaptor = AseAtomsAdaptor()

# ============================================================
# 4. Relax "charge state 1" (e.g., V_O^0, neutral vacancy)
# ============================================================
print("\n=== Relaxing State 1 (mimicking q=0) ===")
atoms_state1 = adaptor.get_atoms(defect_sc)
atoms_state1.calc = calc
opt1 = BFGS(atoms_state1, logfile="relax_state1.log")
opt1.run(fmax=FMAX, steps=500)
e_state1_eq = atoms_state1.get_potential_energy()
pos_state1 = atoms_state1.get_positions().copy()  # Q1 equilibrium positions
print(f"State 1 equilibrium energy: {e_state1_eq:.6f} eV")

# ============================================================
# 5. Relax "charge state 2" (e.g., V_O^{2+}, charged vacancy)
#    With MACE we simulate a different local minimum by perturbing
#    the neighbors of the vacancy and re-relaxing.
#    In a real DFT calculation, the different charge state produces
#    a genuinely different PES and equilibrium geometry.
# ============================================================
print("\n=== Relaxing State 2 (mimicking q=+2, perturbed start) ===")
atoms_state2 = adaptor.get_atoms(defect_sc)

# Perturb nearest neighbors of the vacancy site inward (simulate lattice
# contraction around the charged vacancy, as V_O^{2+} typically relaxes
# inward in MgO)
vac_cart = bulk_sc[vac_idx].coords
for i, pos in enumerate(atoms_state2.get_positions()):
    dist = np.linalg.norm(pos - vac_cart)
    if 1.0 < dist < 3.5:  # nearest neighbors (~2.1 A in MgO)
        direction = (pos - vac_cart) / dist
        # Push neighbors inward by 0.15 A (mimics charged-state relaxation)
        atoms_state2.positions[i] -= 0.15 * direction

atoms_state2.calc = calc
opt2 = BFGS(atoms_state2, logfile="relax_state2.log")
opt2.run(fmax=FMAX, steps=500)
e_state2_eq = atoms_state2.get_potential_energy()
pos_state2 = atoms_state2.get_positions().copy()  # Q2 equilibrium positions
print(f"State 2 equilibrium energy: {e_state2_eq:.6f} eV")

# ============================================================
# 6. Compute DeltaQ (mass-weighted displacement)
# ============================================================
masses = atoms_state1.get_masses()  # atomic masses in amu
# DeltaQ = sqrt( sum_i m_i * |R1_i - R2_i|^2 )  [amu^{1/2} * Angstrom]
displacement = pos_state1 - pos_state2
delta_Q = np.sqrt(np.sum(masses[:, None] * displacement**2))
print(f"\nDelta Q = {delta_Q:.6f} amu^(1/2) * A")

# Per-atom displacements (unweighted, for visualization)
per_atom_disp = np.linalg.norm(displacement, axis=1)
print(f"Max atomic displacement: {np.max(per_atom_disp):.4f} A")
print(f"Mean atomic displacement: {np.mean(per_atom_disp):.4f} A")

# ============================================================
# 7. Generate interpolated/extrapolated structures
# ============================================================
print("\n=== Computing energies along configuration coordinate ===")
print(f"{'Fraction':>10s}  {'Q (amu^1/2 A)':>15s}  {'E_state1 (eV)':>15s}  {'E_state2 (eV)':>15s}")
print("-" * 62)

results = []
for frac in DISTORTION_FRACTIONS:
    # Interpolate positions: R(frac) = R1 + frac * (R2 - R1)
    # frac=0 -> state1 equilibrium, frac=1 -> state2 equilibrium
    pos_interp = pos_state1 + frac * (pos_state2 - pos_state1)
    q_val = frac * delta_Q  # configuration coordinate value

    # Single-shot energy in "state 1" PES
    atoms_eval = atoms_state1.copy()
    atoms_eval.calc = calc
    atoms_eval.set_positions(pos_interp)
    e1 = atoms_eval.get_potential_energy()

    # Single-shot energy in "state 2" PES
    # (With MACE, both states use the same calculator, so e1 == e2.
    #  With DFT, these would differ due to different electronic configurations.)
    atoms_eval2 = atoms_state2.copy()
    atoms_eval2.calc = calc
    atoms_eval2.set_positions(pos_interp)
    e2 = atoms_eval2.get_potential_energy()

    results.append({
        "fraction": frac,
        "Q": q_val,
        "E_state1": e1,
        "E_state2": e2,
    })
    print(f"{frac:10.3f}  {q_val:15.6f}  {e1:15.6f}  {e2:15.6f}")

# ============================================================
# 8. Fit parabolas and extract key quantities
# ============================================================
Q_arr = np.array([r["Q"] for r in results])
E1_arr = np.array([r["E_state1"] for r in results])
E2_arr = np.array([r["E_state2"] for r in results])

# Shift energies relative to state1 minimum
E1_arr_shifted = E1_arr - e_state1_eq
E2_arr_shifted = E2_arr - e_state1_eq

def parabola(x, a, x0, c):
    """Quadratic: E = a*(x - x0)^2 + c"""
    return a * (x - x0)**2 + c

# Fit parabola to state 1 energies (near its minimum at Q ~ 0)
mask1 = np.abs(Q_arr) < 0.6 * delta_Q  # fit around the minimum
if np.sum(mask1) >= 3:
    popt1, _ = curve_fit(parabola, Q_arr[mask1], E1_arr_shifted[mask1],
                         p0=[1.0, 0.0, 0.0])
else:
    popt1, _ = curve_fit(parabola, Q_arr, E1_arr_shifted, p0=[1.0, 0.0, 0.0])

# Fit parabola to state 2 energies (near its minimum at Q ~ delta_Q)
mask2 = np.abs(Q_arr - delta_Q) < 0.6 * delta_Q
if np.sum(mask2) >= 3:
    popt2, _ = curve_fit(parabola, Q_arr[mask2], E2_arr_shifted[mask2],
                         p0=[1.0, delta_Q, E2_arr_shifted[np.argmin(np.abs(Q_arr - delta_Q))]])
else:
    popt2, _ = curve_fit(parabola, Q_arr, E2_arr_shifted,
                         p0=[1.0, delta_Q, 0.0])

print(f"\n=== Parabola Fits ===")
print(f"State 1: E = {popt1[0]:.4f} * (Q - {popt1[1]:.4f})^2 + {popt1[2]:.4f}")
print(f"State 2: E = {popt2[0]:.4f} * (Q - {popt2[1]:.4f})^2 + {popt2[2]:.4f}")

# Extract key quantities
E1_min = popt1[2]              # minimum of parabola 1
E2_min = popt2[2]              # minimum of parabola 2
Q1_min = popt1[1]              # Q at minimum of state 1
Q2_min = popt2[1]              # Q at minimum of state 2

# Zero-phonon line: energy difference between minima
ZPL = abs(E2_min - E1_min)

# Franck-Condon relaxation energies (vertical transitions)
# d_fc1 = E_state1(Q2) - E_state1(Q1): relaxation energy after excitation from state2->state1
d_fc1 = parabola(Q2_min, *popt1) - E1_min
# d_fc2 = E_state2(Q1) - E_state2(Q2): relaxation energy after excitation from state1->state2
d_fc2 = parabola(Q1_min, *popt2) - E2_min

# Classical barrier: crossing point of the two parabolas
# Solve: a1*(Q - Q1)^2 + c1 = a2*(Q - Q2)^2 + c2
a1, q1, c1 = popt1
a2, q2, c2 = popt2
# (a1 - a2)*Q^2 - 2*(a1*q1 - a2*q2)*Q + (a1*q1^2 - a2*q2^2 + c1 - c2) = 0
A = a1 - a2
B = -2 * (a1 * q1 - a2 * q2)
C = a1 * q1**2 - a2 * q2**2 + c1 - c2

if abs(A) > 1e-10:
    discriminant = B**2 - 4 * A * C
    if discriminant >= 0:
        Q_cross_1 = (-B + np.sqrt(discriminant)) / (2 * A)
        Q_cross_2 = (-B - np.sqrt(discriminant)) / (2 * A)
        # Pick the crossing point between the two minima
        Q_cross_candidates = [Q_cross_1, Q_cross_2]
        Q_cross = min(Q_cross_candidates, key=lambda q: abs(q - (Q1_min + Q2_min) / 2))
        E_cross = parabola(Q_cross, *popt1)
        barrier_1 = E_cross - E1_min  # barrier from state 1
        barrier_2 = E_cross - E2_min  # barrier from state 2
    else:
        Q_cross = np.nan
        E_cross = np.nan
        barrier_1 = np.nan
        barrier_2 = np.nan
else:
    # Parabolas have same curvature: single crossing at midpoint
    Q_cross = (C) / (-B) if abs(B) > 1e-10 else np.nan
    E_cross = parabola(Q_cross, *popt1) if not np.isnan(Q_cross) else np.nan
    barrier_1 = E_cross - E1_min if not np.isnan(E_cross) else np.nan
    barrier_2 = E_cross - E2_min if not np.isnan(E_cross) else np.nan

# Huang-Rhys factor: S = d_fc / (hbar * omega)
# For a single effective mode: S = d_fc / (hbar * omega),
# and hbar*omega can be estimated from the curvature: omega = sqrt(2*a/M_eff)
# Approximate: S ~ DeltaQ^2 * omega / (2 * hbar) or simply S = d_fc / (hbar * omega)
# Simpler definition used in literature: S = d_fc / E_phonon
# We report d_fc directly; the user can divide by their chosen phonon energy.

print(f"\n=== Configuration Coordinate Diagram Results ===")
print(f"Delta Q               = {delta_Q:.6f} amu^(1/2) * A")
print(f"Zero-Phonon Line (ZPL)= {ZPL:.4f} eV")
print(f"Franck-Condon shift 1 (d_fc1) = {d_fc1:.4f} eV  (relaxation in state 1 at Q2)")
print(f"Franck-Condon shift 2 (d_fc2) = {d_fc2:.4f} eV  (relaxation in state 2 at Q1)")
print(f"Classical barrier (from state 1) = {barrier_1:.4f} eV")
print(f"Classical barrier (from state 2) = {barrier_2:.4f} eV")
print(f"Crossing point Q = {Q_cross:.4f} amu^(1/2) * A, E = {E_cross:.4f} eV")

# ============================================================
# 9. Plot the CCD
# ============================================================
fig, ax = plt.subplots(figsize=(8, 6))

Q_fine = np.linspace(Q_arr.min() - 0.1 * delta_Q, Q_arr.max() + 0.1 * delta_Q, 500)
E1_fit = parabola(Q_fine, *popt1)
E2_fit = parabola(Q_fine, *popt2)

# Plot fitted parabolas
ax.plot(Q_fine, E1_fit, "-", color="steelblue", linewidth=2, label="State 1 (q=0) fit")
ax.plot(Q_fine, E2_fit, "-", color="firebrick", linewidth=2, label="State 2 (q=+2) fit")

# Plot computed data points
ax.plot(Q_arr, E1_arr_shifted, "o", color="steelblue", markersize=5, alpha=0.7)
ax.plot(Q_arr, E2_arr_shifted, "s", color="firebrick", markersize=5, alpha=0.7)

# Mark minima
ax.plot(Q1_min, E1_min, "*", color="steelblue", markersize=15, zorder=5)
ax.plot(Q2_min, E2_min, "*", color="firebrick", markersize=15, zorder=5)

# Mark ZPL (vertical line between minima)
Q_mid = (Q1_min + Q2_min) / 2
ax.annotate("", xy=(Q1_min, E2_min), xytext=(Q1_min, E1_min),
            arrowprops=dict(arrowstyle="<->", color="green", lw=1.5))
ax.text(Q1_min - 0.05 * delta_Q, (E1_min + E2_min) / 2,
        f"ZPL = {ZPL:.3f} eV", fontsize=9, color="green",
        ha="right", va="center")

# Mark crossing point
if not np.isnan(Q_cross):
    ax.plot(Q_cross, E_cross, "D", color="black", markersize=8, zorder=5,
            label=f"Crossing (barrier = {barrier_1:.3f} eV)")

# Mark Franck-Condon shifts
# Vertical transition at Q2: E_state1(Q2) - E1_min
E1_at_Q2 = parabola(Q2_min, *popt1)
ax.annotate("", xy=(Q2_min, E1_at_Q2), xytext=(Q2_min, E2_min),
            arrowprops=dict(arrowstyle="<->", color="orange", lw=1.5))
ax.text(Q2_min + 0.03 * delta_Q, (E1_at_Q2 + E2_min) / 2,
        f"d$_{{FC,1}}$ = {d_fc1:.3f}", fontsize=8, color="orange", va="center")

ax.set_xlabel(r"Configuration Coordinate $Q$ (amu$^{1/2}$ $\AA$)", fontsize=12)
ax.set_ylabel("Energy (eV, relative to State 1 min)", fontsize=12)
ax.set_title("Configuration Coordinate Diagram", fontsize=13)
ax.legend(fontsize=9, loc="upper center")
ax.grid(True, alpha=0.3)
fig.tight_layout()
fig.savefig("ccd_mace.png", dpi=150)
print(f"\nCCD plot saved to ccd_mace.png")

# ============================================================
# 10. Save all results to JSON
# ============================================================
output = {
    "system": "MgO",
    "defect": "V_O",
    "charge_states": ["q=0 (state 1)", "q=+2 (state 2)"],
    "method": "MACE (screening only -- same PES for both states)",
    "supercell": f"{SUPERCELL_SIZE}x{SUPERCELL_SIZE}x{SUPERCELL_SIZE}",
    "n_atoms": len(atoms_state1),
    "delta_Q_amu_half_A": float(delta_Q),
    "max_atomic_displacement_A": float(np.max(per_atom_disp)),
    "ZPL_eV": float(ZPL),
    "franck_condon_shift_1_eV": float(d_fc1),
    "franck_condon_shift_2_eV": float(d_fc2),
    "classical_barrier_from_state1_eV": float(barrier_1),
    "classical_barrier_from_state2_eV": float(barrier_2),
    "crossing_Q": float(Q_cross),
    "crossing_E_eV": float(E_cross),
    "parabola_fit_state1": {"a": float(popt1[0]), "Q0": float(popt1[1]), "E0": float(popt1[2])},
    "parabola_fit_state2": {"a": float(popt2[0]), "Q0": float(popt2[1]), "E0": float(popt2[2])},
    "distortion_data": results,
}

with open("ccd_results.json", "w") as f:
    json.dump(output, f, indent=2, default=str)
print("Results saved to ccd_results.json")
```

### Method B: QE DFT (Publication Quality)

#### Complete Workflow: CCD with True Charge States

The DFT workflow has four phases:
1. Generate and relax defect supercells in both charge states (pw.x relax)
2. Compute DeltaQ and generate interpolated structures
3. Run single-shot SCF on each interpolated structure in both charge states
4. Parse results, fit parabolas, plot the CCD

##### Phase 1: Generate QE Inputs for Relaxation

```python
#!/usr/bin/env python3
"""
Phase 1: Generate QE relax inputs for V_O in MgO in two charge states (q=0, q=+2).
Produces: pw_relax_q0.in, pw_relax_q2.in
"""

import numpy as np
from pymatgen.core import Structure, Element
from pymatgen.io.pwscf import PWInput
from pathlib import Path

# ============================================================
# Configuration
# ============================================================
PSEUDO_DIR = "./pseudo"
ECUTWFC = 60.0     # Ry
ECUTRHO = 480.0    # Ry
SC_SIZE = 3        # 3x3x3 supercell
KPTS = (2, 2, 2)   # k-point grid for supercell

pseudos = {
    "Mg": "Mg.pbe-spnl-kjpaw_psl.1.0.0.UPF",
    "O":  "O.pbe-n-kjpaw_psl.1.0.0.UPF",
}

Path(PSEUDO_DIR).mkdir(exist_ok=True)

# ============================================================
# Build defect supercell
# ============================================================
primitive = Structure.from_spacegroup(
    "Fm-3m",
    lattice=[[4.212, 0, 0], [0, 4.212, 0], [0, 0, 4.212]],
    species=["Mg", "O"],
    coords=[[0.0, 0.0, 0.0], [0.5, 0.5, 0.5]],
)
bulk_sc = primitive.copy()
bulk_sc.make_supercell([SC_SIZE] * 3)

defect_sc = bulk_sc.copy()
o_indices = [i for i, s in enumerate(defect_sc) if s.specie == Element("O")]
vac_idx = o_indices[len(o_indices) // 2]
print(f"Removing O at site {vac_idx}, supercell has {len(defect_sc)} atoms")
defect_sc.remove_sites([vac_idx])
print(f"Defect supercell: {defect_sc.formula}, {len(defect_sc)} atoms")

# ============================================================
# Common QE parameters
# ============================================================
control_common = {
    "calculation": "relax",
    "restart_mode": "from_scratch",
    "pseudo_dir": PSEUDO_DIR,
    "outdir": "./tmp",
    "tprnfor": True,
    "tstress": False,
    "etot_conv_thr": 1.0e-6,
    "forc_conv_thr": 1.0e-4,
    "nstep": 200,
}

electrons_common = {
    "conv_thr": 1.0e-8,
    "mixing_beta": 0.3,
    "electron_maxstep": 200,
}

# ============================================================
# Charge state q=0 (neutral vacancy)
# ============================================================
pw_q0 = PWInput(
    defect_sc,
    pseudo=pseudos,
    control=control_common | {"prefix": "vo_q0"},
    system={
        "ecutwfc": ECUTWFC,
        "ecutrho": ECUTRHO,
        "occupations": "smearing",
        "smearing": "cold",
        "degauss": 0.01,
        "tot_charge": 0.0,
        "nspin": 1,
    },
    electrons=electrons_common,
    kpoints_grid=KPTS,
)
pw_q0.write_file("pw_relax_q0.in")
print("Written pw_relax_q0.in (neutral V_O)")

# ============================================================
# Charge state q=+2 (doubly positive vacancy, 2 electrons removed)
# ============================================================
pw_q2 = PWInput(
    defect_sc,
    pseudo=pseudos,
    control=control_common | {"prefix": "vo_q2"},
    system={
        "ecutwfc": ECUTWFC,
        "ecutrho": ECUTRHO,
        "occupations": "smearing",
        "smearing": "cold",
        "degauss": 0.01,
        "tot_charge": 2.0,     # +2 charge: 2 electrons removed
        "nspin": 1,
    },
    electrons=electrons_common | {"mixing_beta": 0.2},
    kpoints_grid=KPTS,
)
pw_q2.write_file("pw_relax_q2.in")
print("Written pw_relax_q2.in (V_O^{2+})")

# Also write the bulk supercell for reference energy
pw_bulk = PWInput(
    bulk_sc,
    pseudo=pseudos,
    control=control_common | {"prefix": "bulk", "calculation": "scf"},
    system={
        "ecutwfc": ECUTWFC,
        "ecutrho": ECUTRHO,
        "occupations": "smearing",
        "smearing": "cold",
        "degauss": 0.01,
        "nspin": 1,
    },
    electrons=electrons_common,
    kpoints_grid=KPTS,
)
pw_bulk.write_file("pw_bulk_scf.in")
print(f"Written pw_bulk_scf.in ({len(bulk_sc)} atoms)")
```

##### Phase 1b: Run Relaxations

```bash
#!/bin/bash
# Run relaxation calculations for both charge states
# Adjust NPROC to your system

NPROC=16

# Download pseudopotentials
mkdir -p pseudo
cd pseudo
wget -nc "https://pseudopotentials.quantum-espresso.org/upf_files/Mg.pbe-spnl-kjpaw_psl.1.0.0.UPF" 2>/dev/null
wget -nc "https://pseudopotentials.quantum-espresso.org/upf_files/O.pbe-n-kjpaw_psl.1.0.0.UPF" 2>/dev/null
cd ..

mkdir -p tmp

# Bulk SCF (needed for reference energy and VBM)
echo "=== Running bulk SCF ==="
mpirun -np $NPROC pw.x -in pw_bulk_scf.in > pw_bulk_scf.out 2>&1
echo "Bulk: $(grep '!' pw_bulk_scf.out | tail -1)"

# Relax charge state q=0
echo "=== Relaxing V_O (q=0) ==="
mpirun -np $NPROC pw.x -in pw_relax_q0.in > pw_relax_q0.out 2>&1
echo "q=0: $(grep '!' pw_relax_q0.out | tail -1)"

# Relax charge state q=+2
echo "=== Relaxing V_O (q=+2) ==="
mpirun -np $NPROC pw.x -in pw_relax_q2.in > pw_relax_q2.out 2>&1
echo "q=+2: $(grep '!' pw_relax_q2.out | tail -1)"

echo "=== All relaxations complete ==="
```

##### Phase 2: Compute DeltaQ and Generate Interpolated SCF Inputs

```python
#!/usr/bin/env python3
"""
Phase 2: Parse relaxed structures from QE output, compute DeltaQ,
generate interpolated structures, and write single-shot SCF inputs.

Reads:  pw_relax_q0.out, pw_relax_q2.out
Writes: scf_q0_frac_XXX.in, scf_q2_frac_XXX.in for each distortion fraction
"""

import re
import numpy as np
from pymatgen.core import Structure, Lattice
from pymatgen.io.pwscf import PWInput
from pathlib import Path
import json

# ============================================================
# Configuration
# ============================================================
PSEUDO_DIR = "./pseudo"
ECUTWFC = 60.0
ECUTRHO = 480.0
KPTS = (2, 2, 2)

pseudos = {
    "Mg": "Mg.pbe-spnl-kjpaw_psl.1.0.0.UPF",
    "O":  "O.pbe-n-kjpaw_psl.1.0.0.UPF",
}

# Distortion fractions: 0 = q0 equilibrium, 1 = q2 equilibrium
DISTORTION_FRACTIONS = [-0.2, -0.15, -0.1, -0.05, 0.0,
                         0.05, 0.1, 0.15, 0.2,
                         0.4, 0.6, 0.8,
                         0.85, 0.9, 0.95, 1.0,
                         1.05, 1.1, 1.15, 1.2]

# ============================================================
# Parse relaxed structures from QE output
# ============================================================
def parse_qe_relaxed_structure(qe_output_file):
    """
    Extract the final relaxed structure from a QE relax output file.
    Returns a pymatgen Structure.
    """
    bohr_to_ang = 0.529177249

    with open(qe_output_file) as f:
        lines = f.readlines()

    # Find the last set of CELL_PARAMETERS and ATOMIC_POSITIONS
    cell_start = None
    pos_start = None
    for i, line in enumerate(lines):
        if "CELL_PARAMETERS" in line:
            cell_start = i
        if "ATOMIC_POSITIONS" in line:
            pos_start = i

    # If no CELL_PARAMETERS found in output (cell not optimized), get from input echo
    if cell_start is None:
        # Parse the lattice from the initial input echo
        for i, line in enumerate(lines):
            if "celldm(1)" in line:
                celldm1 = float(line.split("=")[1].split()[0]) * bohr_to_ang
            if "crystal axes: (cart. coord. in units of alat)" in line:
                a1 = [float(x) for x in lines[i+1].split()[3:6]]
                a2 = [float(x) for x in lines[i+2].split()[3:6]]
                a3 = [float(x) for x in lines[i+3].split()[3:6]]
                lattice = Lattice(np.array([a1, a2, a3]) * celldm1)
                break
    else:
        # Parse CELL_PARAMETERS (angstrom or bohr)
        header = lines[cell_start].strip()
        scale = 1.0
        if "bohr" in header.lower():
            scale = bohr_to_ang
        a1 = [float(x) * scale for x in lines[cell_start + 1].split()]
        a2 = [float(x) * scale for x in lines[cell_start + 2].split()]
        a3 = [float(x) * scale for x in lines[cell_start + 3].split()]
        lattice = Lattice([a1, a2, a3])

    # Parse ATOMIC_POSITIONS
    if pos_start is None:
        raise ValueError(f"No ATOMIC_POSITIONS found in {qe_output_file}")

    header = lines[pos_start].strip()
    coord_type = "crystal" if "crystal" in header.lower() else "angstrom"

    species = []
    coords = []
    for line in lines[pos_start + 1:]:
        parts = line.split()
        if len(parts) < 4:
            break
        try:
            species.append(parts[0])
            coords.append([float(parts[1]), float(parts[2]), float(parts[3])])
        except (ValueError, IndexError):
            break

    coords = np.array(coords)
    if coord_type == "crystal":
        struct = Structure(lattice, species, coords)
    else:
        struct = Structure(lattice, species, coords, coords_are_cartesian=True)

    return struct


# Parse both relaxed structures
struct_q0 = parse_qe_relaxed_structure("pw_relax_q0.out")
struct_q2 = parse_qe_relaxed_structure("pw_relax_q2.out")

print(f"Relaxed q=0 structure: {struct_q0.formula}, {len(struct_q0)} atoms")
print(f"Relaxed q=+2 structure: {struct_q2.formula}, {len(struct_q2)} atoms")

# Verify structures are compatible (same species ordering)
assert len(struct_q0) == len(struct_q2), "Structures must have the same number of atoms"
for i in range(len(struct_q0)):
    assert str(struct_q0[i].specie) == str(struct_q2[i].specie), \
        f"Species mismatch at site {i}: {struct_q0[i].specie} vs {struct_q2[i].specie}"

# ============================================================
# Compute DeltaQ
# ============================================================
pos_q0 = struct_q0.cart_coords
pos_q2 = struct_q2.cart_coords

# Atomic masses (amu)
from pymatgen.core import Element as Elem
masses = np.array([Elem(str(s.specie)).atomic_mass for s in struct_q0])

displacement = pos_q0 - pos_q2
delta_Q = np.sqrt(np.sum(masses[:, None] * displacement**2))

per_atom_disp = np.linalg.norm(displacement, axis=1)
print(f"\nDelta Q = {delta_Q:.6f} amu^(1/2) * A")
print(f"Max atomic displacement: {np.max(per_atom_disp):.4f} A")

# Save DeltaQ info
with open("delta_Q.json", "w") as f:
    json.dump({
        "delta_Q_amu_half_A": float(delta_Q),
        "max_displacement_A": float(np.max(per_atom_disp)),
        "mean_displacement_A": float(np.mean(per_atom_disp)),
    }, f, indent=2)

# ============================================================
# Generate interpolated structures and write SCF inputs
# ============================================================
Path("scf_inputs").mkdir(exist_ok=True)

control_scf = {
    "calculation": "scf",
    "restart_mode": "from_scratch",
    "pseudo_dir": PSEUDO_DIR,
    "outdir": "./tmp",
    "tprnfor": True,
    "tstress": False,
}

electrons_scf = {
    "conv_thr": 1.0e-8,
    "mixing_beta": 0.3,
    "electron_maxstep": 200,
}

input_files = {"q0": [], "q2": []}

for frac in DISTORTION_FRACTIONS:
    # Interpolate Cartesian coordinates
    pos_interp = pos_q0 + frac * (pos_q2 - pos_q0)

    # Create interpolated structure (use q0 lattice; cell should be similar)
    species_list = [str(s.specie) for s in struct_q0]
    struct_interp = Structure(
        struct_q0.lattice, species_list, pos_interp, coords_are_cartesian=True
    )

    frac_label = f"{frac:+.3f}".replace("+", "p").replace("-", "m").replace(".", "")
    q_val = frac * delta_Q

    # SCF in charge state q=0 at this geometry
    fname_q0 = f"scf_inputs/scf_q0_frac_{frac_label}.in"
    pw_q0 = PWInput(
        struct_interp,
        pseudo=pseudos,
        control=control_scf | {"prefix": f"ccd_q0_{frac_label}"},
        system={
            "ecutwfc": ECUTWFC,
            "ecutrho": ECUTRHO,
            "occupations": "smearing",
            "smearing": "cold",
            "degauss": 0.01,
            "tot_charge": 0.0,
            "nspin": 1,
        },
        electrons=electrons_scf,
        kpoints_grid=KPTS,
    )
    pw_q0.write_file(fname_q0)
    input_files["q0"].append(fname_q0)

    # SCF in charge state q=+2 at this geometry
    fname_q2 = f"scf_inputs/scf_q2_frac_{frac_label}.in"
    pw_q2 = PWInput(
        struct_interp,
        pseudo=pseudos,
        control=control_scf | {"prefix": f"ccd_q2_{frac_label}"},
        system={
            "ecutwfc": ECUTWFC,
            "ecutrho": ECUTRHO,
            "occupations": "smearing",
            "smearing": "cold",
            "degauss": 0.01,
            "tot_charge": 2.0,
            "nspin": 1,
        },
        electrons=electrons_scf | {"mixing_beta": 0.2},
        kpoints_grid=KPTS,
    )
    pw_q2.write_file(fname_q2)
    input_files["q2"].append(fname_q2)

    print(f"Fraction {frac:+6.3f}  Q = {q_val:8.4f}  -> {fname_q0}, {fname_q2}")

with open("scf_input_list.json", "w") as f:
    json.dump({"fractions": DISTORTION_FRACTIONS, "files": input_files}, f, indent=2)
print(f"\nGenerated {len(DISTORTION_FRACTIONS)} x 2 = {2*len(DISTORTION_FRACTIONS)} SCF inputs")
print("File list saved to scf_input_list.json")
```

##### Phase 2b: Run All Single-Shot SCF Calculations

```bash
#!/bin/bash
# Run all single-shot SCF calculations for the CCD.
# These are independent and can be parallelized or submitted as array jobs.

NPROC=16
INPUT_DIR="scf_inputs"
OUTPUT_DIR="scf_outputs"
mkdir -p "$OUTPUT_DIR" tmp

echo "=== Running CCD single-shot SCFs ==="

for infile in "$INPUT_DIR"/scf_*.in; do
    base=$(basename "$infile" .in)
    outfile="$OUTPUT_DIR/${base}.out"

    if [ -f "$outfile" ] && grep -q "convergence has been achieved" "$outfile"; then
        echo "SKIP (already done): $base"
        continue
    fi

    echo "Running: $base ..."
    mpirun -np $NPROC pw.x -in "$infile" > "$outfile" 2>&1

    if grep -q "convergence has been achieved" "$outfile"; then
        energy=$(grep '!' "$outfile" | tail -1 | awk '{print $5}')
        echo "  Done: E = $energy Ry"
    else
        echo "  WARNING: SCF did not converge for $base"
    fi
done

echo "=== All CCD SCFs complete ==="
```

##### Phase 3: Parse Results, Fit, and Plot

```python
#!/usr/bin/env python3
"""
Phase 3: Parse single-shot SCF energies, fit parabolas, and plot the CCD.

Reads:  scf_outputs/scf_q0_frac_*.out, scf_outputs/scf_q2_frac_*.out
        delta_Q.json, scf_input_list.json
Writes: ccd_dft.png, ccd_dft_results.json
"""

import re
import json
import numpy as np
from scipy.optimize import curve_fit
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ============================================================
# 1. Load metadata
# ============================================================
with open("delta_Q.json") as f:
    dq_data = json.load(f)
delta_Q = dq_data["delta_Q_amu_half_A"]

with open("scf_input_list.json") as f:
    scf_meta = json.load(f)
fractions = scf_meta["fractions"]

Ry_to_eV = 13.605693123

# ============================================================
# 2. Parse energies from QE output
# ============================================================
def parse_qe_energy(filename):
    """Extract final total energy (eV) from QE output."""
    energy = None
    with open(filename) as f:
        for line in f:
            if "!" in line and "total energy" in line:
                match = re.search(r"=\s+([-\d.]+)\s+Ry", line)
                if match:
                    energy = float(match.group(1)) * Ry_to_eV
    return energy

E_q0_list = []
E_q2_list = []
Q_list = []
valid_fractions = []

for i, frac in enumerate(fractions):
    frac_label = f"{frac:+.3f}".replace("+", "p").replace("-", "m").replace(".", "")
    q_val = frac * delta_Q

    fname_q0 = f"scf_outputs/scf_q0_frac_{frac_label}.out"
    fname_q2 = f"scf_outputs/scf_q2_frac_{frac_label}.out"

    e_q0 = parse_qe_energy(fname_q0)
    e_q2 = parse_qe_energy(fname_q2)

    if e_q0 is not None and e_q2 is not None:
        E_q0_list.append(e_q0)
        E_q2_list.append(e_q2)
        Q_list.append(q_val)
        valid_fractions.append(frac)
        print(f"frac={frac:+6.3f}  Q={q_val:8.4f}  E(q=0)={e_q0:.6f}  E(q=+2)={e_q2:.6f} eV")
    else:
        print(f"frac={frac:+6.3f}  MISSING or unconverged")

Q_arr = np.array(Q_list)
E_q0_arr = np.array(E_q0_list)
E_q2_arr = np.array(E_q2_list)

# ============================================================
# 3. Shift energies: use the minimum of E_q0 as reference
# ============================================================
E_q0_min = np.min(E_q0_arr)
E_q0_shifted = E_q0_arr - E_q0_min
E_q2_shifted = E_q2_arr - E_q0_min

# ============================================================
# 4. Fit parabolas
# ============================================================
def parabola(x, a, x0, c):
    return a * (x - x0)**2 + c

# Fit state q=0 (minimum near Q=0)
mask_q0 = np.array(valid_fractions)
mask_near0 = np.abs(mask_q0) < 0.5
if np.sum(mask_near0) >= 3:
    popt_q0, _ = curve_fit(parabola, Q_arr[mask_near0], E_q0_shifted[mask_near0],
                           p0=[1.0, 0.0, 0.0])
else:
    popt_q0, _ = curve_fit(parabola, Q_arr, E_q0_shifted, p0=[1.0, 0.0, 0.0])

# Fit state q=+2 (minimum near Q=delta_Q)
mask_near1 = np.abs(mask_q0 - 1.0) < 0.5
if np.sum(mask_near1) >= 3:
    popt_q2, _ = curve_fit(parabola, Q_arr[mask_near1], E_q2_shifted[mask_near1],
                           p0=[1.0, delta_Q, np.min(E_q2_shifted)])
else:
    popt_q2, _ = curve_fit(parabola, Q_arr, E_q2_shifted,
                           p0=[1.0, delta_Q, np.min(E_q2_shifted)])

print(f"\n=== Parabola Fits ===")
print(f"q=0:  E = {popt_q0[0]:.4f}*(Q - {popt_q0[1]:.4f})^2 + {popt_q0[2]:.4f}")
print(f"q=+2: E = {popt_q2[0]:.4f}*(Q - {popt_q2[1]:.4f})^2 + {popt_q2[2]:.4f}")

# ============================================================
# 5. Extract CCD quantities
# ============================================================
Q0_min = popt_q0[1]
Q2_min = popt_q2[1]
E0_min = popt_q0[2]
E2_min = popt_q2[2]

# Zero-phonon line
ZPL = abs(E2_min - E0_min)

# Franck-Condon shifts (relaxation energies)
# d_fc_q0 = E_q0(Q2_min) - E_q0(Q0_min) : vertical absorption/emission in q=0 PES
d_fc_q0 = parabola(Q2_min, *popt_q0) - E0_min
# d_fc_q2 = E_q2(Q0_min) - E_q2(Q2_min) : vertical absorption/emission in q=+2 PES
d_fc_q2 = parabola(Q0_min, *popt_q2) - E2_min

# Stokes shift (total relaxation energy for an optical cycle)
stokes_shift = d_fc_q0 + d_fc_q2

# Classical crossing point (barrier)
a1, q1, c1 = popt_q0
a2, q2, c2 = popt_q2
A = a1 - a2
B = -2 * (a1 * q1 - a2 * q2)
C = a1 * q1**2 - a2 * q2**2 + c1 - c2

barrier_q0 = np.nan
barrier_q2 = np.nan
Q_cross = np.nan
E_cross = np.nan

if abs(A) > 1e-10:
    disc = B**2 - 4 * A * C
    if disc >= 0:
        Qc1 = (-B + np.sqrt(disc)) / (2 * A)
        Qc2 = (-B - np.sqrt(disc)) / (2 * A)
        Q_cross = min([Qc1, Qc2], key=lambda q: abs(q - (Q0_min + Q2_min) / 2))
        E_cross = parabola(Q_cross, *popt_q0)
        barrier_q0 = E_cross - E0_min
        barrier_q2 = E_cross - E2_min
else:
    if abs(B) > 1e-10:
        Q_cross = -C / B
        E_cross = parabola(Q_cross, *popt_q0)
        barrier_q0 = E_cross - E0_min
        barrier_q2 = E_cross - E2_min

# Effective phonon frequency from parabola curvature: omega = sqrt(2*a / M_eff)
# where a is in eV / (amu * A^2) and M_eff is the effective mass
# The curvature 'a' already has mass built into Q, so:
# E = a * Q^2 => omega = sqrt(2*a) * hbar (if Q is in amu^{1/2}*A)
# hbar = 6.582e-16 eV*s, 1 amu = 1.661e-27 kg, 1 A = 1e-10 m
# omega (meV) = sqrt(2*a) * hbar_eVs * sqrt(eV_to_J / (amu_to_kg * A_to_m^2))
# Simpler: hbar*omega (eV) = hbar * sqrt(2*a * eV/(amu*A^2))
hbar_eVs = 6.582119569e-16  # eV * s
amu_to_kg = 1.66053906660e-27
A_to_m = 1e-10
eV_to_J = 1.602176634e-19

omega_q0 = np.sqrt(2 * abs(popt_q0[0]) * eV_to_J / (amu_to_kg * A_to_m**2))
hbar_omega_q0 = hbar_eVs * omega_q0  # in eV

omega_q2 = np.sqrt(2 * abs(popt_q2[0]) * eV_to_J / (amu_to_kg * A_to_m**2))
hbar_omega_q2 = hbar_eVs * omega_q2

# Huang-Rhys factor: S = d_fc / (hbar * omega)
S_q0 = d_fc_q0 / hbar_omega_q0 if hbar_omega_q0 > 0 else np.nan
S_q2 = d_fc_q2 / hbar_omega_q2 if hbar_omega_q2 > 0 else np.nan

print(f"\n=== Configuration Coordinate Diagram Results ===")
print(f"Delta Q                    = {delta_Q:.6f} amu^(1/2) * A")
print(f"Zero-Phonon Line (ZPL)     = {ZPL:.4f} eV")
print(f"Franck-Condon shift (q=0)  = {d_fc_q0:.4f} eV")
print(f"Franck-Condon shift (q=+2) = {d_fc_q2:.4f} eV")
print(f"Stokes shift               = {stokes_shift:.4f} eV")
print(f"Classical barrier (from q=0)  = {barrier_q0:.4f} eV")
print(f"Classical barrier (from q=+2) = {barrier_q2:.4f} eV")
print(f"hbar*omega (q=0 parabola)  = {hbar_omega_q0*1000:.2f} meV")
print(f"hbar*omega (q=+2 parabola) = {hbar_omega_q2*1000:.2f} meV")
print(f"Huang-Rhys factor S (q=0)  = {S_q0:.2f}")
print(f"Huang-Rhys factor S (q=+2) = {S_q2:.2f}")

# ============================================================
# 6. Plot the CCD
# ============================================================
fig, ax = plt.subplots(figsize=(8, 6))

Q_fine = np.linspace(Q_arr.min() - 0.15 * delta_Q,
                     Q_arr.max() + 0.15 * delta_Q, 500)
E_q0_fit = parabola(Q_fine, *popt_q0)
E_q2_fit = parabola(Q_fine, *popt_q2)

# Fitted parabolas
ax.plot(Q_fine, E_q0_fit, "-", color="steelblue", linewidth=2.5,
        label=r"$V_O^{0}$ (q = 0)")
ax.plot(Q_fine, E_q2_fit, "-", color="firebrick", linewidth=2.5,
        label=r"$V_O^{2+}$ (q = +2)")

# DFT data points
ax.plot(Q_arr, E_q0_shifted, "o", color="steelblue", markersize=6, alpha=0.8)
ax.plot(Q_arr, E_q2_shifted, "s", color="firebrick", markersize=6, alpha=0.8)

# Mark equilibrium minima
ax.plot(Q0_min, E0_min, "*", color="steelblue", markersize=16, zorder=5)
ax.plot(Q2_min, E2_min, "*", color="firebrick", markersize=16, zorder=5)

# ZPL annotation (vertical double arrow between minima at Q0)
ax.annotate("", xy=(Q0_min, E2_min + d_fc_q2), xytext=(Q0_min, E0_min),
            arrowprops=dict(arrowstyle="<->", color="green", lw=2))
ax.text(Q0_min - 0.08 * delta_Q, (E0_min + E2_min + d_fc_q2) / 2,
        f"ZPL\n{ZPL:.3f} eV", fontsize=9, color="green",
        ha="right", va="center", fontweight="bold")

# Crossing point and barrier
if not np.isnan(Q_cross):
    ax.plot(Q_cross, E_cross, "D", color="black", markersize=10, zorder=5)
    ax.annotate(f"Barrier\n{barrier_q0:.3f} eV",
                xy=(Q_cross, E_cross), xytext=(Q_cross + 0.1 * delta_Q, E_cross + 0.1),
                fontsize=9, ha="left",
                arrowprops=dict(arrowstyle="->", color="black"))

# Franck-Condon shift arrows
E_q0_at_Q2 = parabola(Q2_min, *popt_q0)
ax.annotate("", xy=(Q2_min, E_q0_at_Q2), xytext=(Q2_min, E2_min),
            arrowprops=dict(arrowstyle="<->", color="orange", lw=1.5))
ax.text(Q2_min + 0.04 * delta_Q, (E_q0_at_Q2 + E2_min) / 2,
        f"$d_{{FC}}$ = {d_fc_q0:.3f} eV", fontsize=8, color="orange", va="center")

E_q2_at_Q0 = parabola(Q0_min, *popt_q2)
ax.annotate("", xy=(Q0_min, E_q2_at_Q0), xytext=(Q0_min, E0_min),
            arrowprops=dict(arrowstyle="<->", color="purple", lw=1.5))
ax.text(Q0_min - 0.04 * delta_Q, (E_q2_at_Q0 + E0_min) / 2,
        f"$d_{{FC}}$ = {d_fc_q2:.3f} eV", fontsize=8, color="purple",
        ha="right", va="center")

ax.set_xlabel(r"Configuration Coordinate $Q$ (amu$^{1/2}$ $\AA$)", fontsize=13)
ax.set_ylabel("Energy (eV)", fontsize=13)
ax.set_title("Configuration Coordinate Diagram (QE DFT)", fontsize=14)
ax.legend(fontsize=10, loc="upper center")
ax.grid(True, alpha=0.3)
fig.tight_layout()
fig.savefig("ccd_dft.png", dpi=200)
print(f"\nCCD plot saved to ccd_dft.png")

# ============================================================
# 7. Save all results
# ============================================================
output = {
    "system": "MgO",
    "defect": "V_O",
    "charge_states": ["q=0", "q=+2"],
    "method": "QE DFT (PBE)",
    "ecutwfc_Ry": ECUTWFC,
    "kpoints": list(KPTS),
    "delta_Q_amu_half_A": float(delta_Q),
    "ZPL_eV": float(ZPL),
    "franck_condon_shift_q0_eV": float(d_fc_q0),
    "franck_condon_shift_q2_eV": float(d_fc_q2),
    "stokes_shift_eV": float(stokes_shift),
    "classical_barrier_from_q0_eV": float(barrier_q0),
    "classical_barrier_from_q2_eV": float(barrier_q2),
    "crossing_Q": float(Q_cross),
    "crossing_E_eV": float(E_cross),
    "hbar_omega_q0_meV": float(hbar_omega_q0 * 1000),
    "hbar_omega_q2_meV": float(hbar_omega_q2 * 1000),
    "huang_rhys_factor_q0": float(S_q0),
    "huang_rhys_factor_q2": float(S_q2),
    "parabola_fit_q0": {"a": float(popt_q0[0]), "Q0": float(popt_q0[1]), "E0": float(popt_q0[2])},
    "parabola_fit_q2": {"a": float(popt_q2[0]), "Q0": float(popt_q2[1]), "E0": float(popt_q2[2])},
    "note": "Energies for charged state include compensating jellium. Apply Freysoldt correction for quantitative results.",
    "distortion_data": [
        {"fraction": float(valid_fractions[i]),
         "Q": float(Q_arr[i]),
         "E_q0_eV": float(E_q0_arr[i]),
         "E_q2_eV": float(E_q2_arr[i])}
        for i in range(len(Q_arr))
    ],
}

with open("ccd_dft_results.json", "w") as f:
    json.dump(output, f, indent=2, default=str)
print("Results saved to ccd_dft_results.json")
```

## Key Parameters

| Parameter | Recommended Value | Notes |
|---|---|---|
| Supercell size | >= 3x3x3 or min ~10 A between defect images | Must be large enough for finite-size corrections to be small |
| Distortion fractions | -0.2 to 1.2, denser near minima (0, 1) | Denser sampling near equilibria improves parabola fit; include extrapolation points to constrain curvature |
| MACE fmax | 0.005 eV/A | Tight relaxation ensures accurate equilibrium geometries |
| QE ecutwfc | 50-80 Ry | Follow SSSP recommendations for the elements |
| QE k-points | Gamma-only or 2x2x2 for large supercells | Converge separately; must be identical across all SCF runs |
| QE conv_thr | 1e-8 Ry | Tight SCF convergence needed for small energy differences along CCD |
| tot_charge | float (e.g., 2.0 for q=+2) | QE convention: positive = electrons removed. Jellium background is automatic |
| mixing_beta | 0.2-0.3 for charged defects | Lower values improve convergence for charged supercells |
| Number of distortion points | 15-20 total | Minimum ~8 points per parabola; more points near crossing region for accurate barrier |

## Interpreting Results

1. **DeltaQ magnitude**: Typical values for point defects are 1-10 amu^{1/2} A. Larger DeltaQ means larger structural relaxation between charge states, which generally favors non-radiative recombination.

2. **Zero-phonon line (ZPL)**: The energy difference between the parabola minima. This is the minimum photon energy for the optical transition. Compare with experimental PL peak energy.

3. **Franck-Condon shifts (d_FC)**: The relaxation energy after a vertical transition. Large d_FC values indicate strong electron-phonon coupling. The absorption peak is at ZPL + d_FC,absorption and the emission peak is at ZPL - d_FC,emission.

4. **Stokes shift**: d_FC,1 + d_FC,2 gives the total Stokes shift, which is the energy difference between absorption and emission peaks.

5. **Classical barrier**: The energy at the crossing point of the two parabolas relative to the lower minimum. This is the activation energy for non-radiative carrier capture. Barriers below ~0.3 eV indicate efficient non-radiative recombination at room temperature.

6. **Huang-Rhys factor (S)**: S = d_FC / (hbar * omega_eff). Large S (>> 1) means strong coupling, broad luminescence bands, and efficient non-radiative decay. S < 5 corresponds to a narrow zero-phonon line with phonon sidebands.

7. **MACE limitations**: Since MACE has no electronic degrees of freedom, both "charge states" share the same potential energy surface. The MACE CCD is only meaningful for estimating structural quantities (DeltaQ, displacement patterns). All energetic quantities (ZPL, barriers, Huang-Rhys) require DFT.

8. **Finite-size corrections**: For charged defects in DFT, the energies include spurious interactions with periodic images and the compensating jellium. Apply Freysoldt or Kumagai corrections to the energy difference between charge states. The correction affects ZPL but not the individual parabola shapes.

9. **Non-radiative rate estimation**: The classical Arrhenius rate is R ~ nu * exp(-E_barrier / kT). For a quantum treatment using the static coupling approximation, use the Huang-Rhys factor and multi-phonon transition theory (e.g., the formalism in Alkauskas et al., Phys. Rev. B 90, 075202 (2014)).

## Common Issues

| Issue | Solution |
|---|---|
| SCF does not converge for charged defect at distorted geometry | Reduce `mixing_beta` to 0.1-0.2; increase `electron_maxstep`; try `mixing_mode = 'local-TF'` |
| Parabola fit is poor (large residuals) | Add more distortion points near the minimum; exclude outlier points far from equilibrium where anharmonicity is large |
| Two parabolas do not cross (no classical barrier) | This means the transition is thermally inaccessible via the classical path; check that the correct charge states are used; consider that the crossing may occur at very high energy |
| DeltaQ is unexpectedly small (~0) | Both charge states may have very similar relaxed structures; verify that the relaxation converged properly and that the initial structures had the correct charge |
| DeltaQ is unexpectedly large (> 15 amu^{1/2} A) | Check for atoms that "jumped" to a different site during relaxation; the atom ordering may differ between the two outputs; re-map atoms if needed |
| Different number of electrons warning in QE | This is expected when comparing q=0 and q=+2 at the same geometry; the single-shot SCF uses the correct electron count for each charge state |
| Negative Huang-Rhys factor | Indicates a fitting error; check that the parabola curvature (a) is positive and that d_FC > 0 |
| Energies along CCD are not smooth | One or more SCF calculations did not converge; rerun unconverged points with tighter `conv_thr` or smaller `mixing_beta` |
| MACE gives identical energies for both "charge states" | Expected behavior -- MACE has no charge-state dependence. Use DFT for distinct charge-state energetics |
| Formation energy vs. Fermi level needed | Combine CCD results with the defect formation energy workflow (see `point-defect/` skill) to place the CCD on an absolute energy scale |
