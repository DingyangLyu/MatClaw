# High-Throughput Materials Screening Workflow

## When to Use

- You need to discover new materials with specific target properties (stability, band gap, stiffness, low density, etc.).
- You want to explore a chemical space systematically (e.g., all ternary oxides of 3d transition metals with a rare earth).
- You need a ranked list of candidate materials from hundreds of possibilities, filtered by multiple criteria.
- You want to combine database mining with computational validation before committing to expensive DFT runs.

## Method Selection

| Approach | Speed | Accuracy | Use When |
|---|---|---|---|
| MP query only | Seconds | Database-level (PBE DFT) | You need known data, no new calculations |
| MP query + MACE screening | Minutes--hours | ~10-30 meV/atom for energies, ~10-20% for moduli | First pass over large candidate pools |
| MP query + MACE + QE validation | Hours--days | Publication quality for top candidates | Final validation of short-listed materials |
| Full DFT screening (QE/VASP on all) | Days--weeks | Systematic DFT | Small candidate pools (<20 structures) |

## Prerequisites

- `MP_API_KEY` environment variable set (obtain from https://next-gen.materialsproject.org/api).
- Python packages: `mp-api`, `pymatgen`, `ase`, `mace-torch`, `numpy`, `pandas`, `matplotlib`, `scipy`.
- For DFT validation: Quantum ESPRESSO 7.5 (`pw.x`) with SSSP pseudopotentials.
- Sufficient disk space for structure files and calculation outputs.

## Detailed Steps

### Full Screening Pipeline

The script below performs the complete workflow: define search criteria, query Materials Project, MACE-relax all candidates, filter by multiple properties, rank with a composite score, and generate a report with plots.

```python
#!/usr/bin/env python3
"""
High-throughput materials screening workflow.

Pipeline:
  1. Query Materials Project for candidate materials matching chemical/structural criteria.
  2. MACE-relax each structure and compute energy per atom.
  3. Filter by stability (e_above_hull), band gap range, and size constraints.
  4. Compute additional properties for survivors: bulk modulus, elastic anisotropy.
  5. Rank candidates using a weighted composite score.
  6. Export results to CSV/JSON with full property tables.
  7. Generate visualization: distributions, correlation plots, Pareto fronts.
  8. Write QE input files for the top N candidates for DFT validation.
"""

import os
import json
import warnings
import traceback
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from typing import Optional

from ase import Atoms
from ase.optimize import LBFGS
from ase.constraints import ExpCellFilter
from ase.io import write as ase_write

from pymatgen.core import Structure, Element
from pymatgen.io.ase import AseAtomsAdaptor
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer

from mp_api.client import MPRester

warnings.filterwarnings("ignore")

# ================================================================== #
#  CONFIGURATION -- Adjust these for your screening campaign
# ================================================================== #
MP_API_KEY = os.environ.get("MP_API_KEY")
assert MP_API_KEY, "Set MP_API_KEY environment variable first."

# --- Search criteria ---
# Option A: search by chemical system
CHEMSYS_LIST = [
    "Ti-O", "Zr-O", "Hf-O",       # Group 4 oxides
    "V-O", "Nb-O", "Ta-O",         # Group 5 oxides
    "Cr-O", "Mo-O", "W-O",         # Group 6 oxides
]
# Option B: search by required elements (uncomment to use instead)
# REQUIRED_ELEMENTS = ["O"]
# CRYSTAL_SYSTEM = "Cubic"

# --- Filters ---
MAX_ATOMS = 48                      # skip large unit cells for speed
EHULL_CUTOFF = 0.050                # eV/atom -- thermodynamic stability
BANDGAP_RANGE = (0.0, None)         # (min, max) eV; None = no upper bound
MIN_BULK_MODULUS = None             # GPa; None = no filter
MAX_CANDIDATES = 500                # safety limit on total candidates

# --- MACE settings ---
MACE_MODEL = "medium"               # "small", "medium", "large"
FMAX_RELAX = 0.02                   # eV/Ang for full relaxation
FMAX_BULK = 1e-3                    # eV/Ang for bulk modulus E(V) points

# --- Ranking weights (higher = more important) ---
WEIGHT_STABILITY = 0.4              # lower e_above_hull is better
WEIGHT_BULK_MOD = 0.3               # higher bulk modulus is better
WEIGHT_BANDGAP = 0.3                # closer to target gap is better
BANDGAP_TARGET = 2.0                # eV -- ideal band gap for ranking

# --- Output ---
OUTPUT_DIR = "screening_results"
TOP_N_FOR_DFT = 5                   # number of top candidates for QE input

# ================================================================== #

os.makedirs(OUTPUT_DIR, exist_ok=True)
adaptor = AseAtomsAdaptor()

# ================================================================== #
#  STEP 1: Query Materials Project
# ================================================================== #
def query_materials_project():
    """Fetch candidate materials from MP matching the search criteria."""
    candidates = []

    with MPRester(MP_API_KEY) as mpr:
        for chemsys in CHEMSYS_LIST:
            docs = mpr.materials.summary.search(
                chemsys=chemsys,
                energy_above_hull=(0, EHULL_CUTOFF),
                fields=[
                    "material_id", "formula_pretty", "structure",
                    "energy_above_hull", "formation_energy_per_atom",
                    "band_gap", "nsites", "symmetry", "density",
                    "is_stable",
                ],
            )
            for doc in docs:
                if doc.nsites > MAX_ATOMS:
                    continue
                if doc.structure is None:
                    continue

                # Band gap filter
                bg = doc.band_gap
                if BANDGAP_RANGE[0] is not None and bg < BANDGAP_RANGE[0]:
                    continue
                if BANDGAP_RANGE[1] is not None and bg > BANDGAP_RANGE[1]:
                    continue

                candidates.append({
                    "mp_id": str(doc.material_id),
                    "formula": doc.formula_pretty,
                    "structure": doc.structure,
                    "e_above_hull_mp": doc.energy_above_hull,
                    "form_energy_mp": doc.formation_energy_per_atom,
                    "band_gap_mp": bg,
                    "nsites": doc.nsites,
                    "spacegroup": (doc.symmetry.symbol
                                   if doc.symmetry else "unknown"),
                    "crystal_system": (doc.symmetry.crystal_system
                                       if doc.symmetry else "unknown"),
                    "density_mp": doc.density,
                    "is_stable_mp": doc.is_stable,
                })

                if len(candidates) >= MAX_CANDIDATES:
                    break
            if len(candidates) >= MAX_CANDIDATES:
                break

    print(f"Step 1: Fetched {len(candidates)} candidates from Materials Project.")
    return candidates


# ================================================================== #
#  STEP 2: MACE relaxation
# ================================================================== #
def mace_relax(candidate: dict) -> dict:
    """Relax a single structure with MACE and return energy per atom."""
    from mace.calculators import mace_mp
    calc = mace_mp(model=MACE_MODEL, default_dtype="float64")

    atoms = adaptor.get_atoms(candidate["structure"])
    atoms.calc = calc

    try:
        ecf = ExpCellFilter(atoms, hydrostatic_strain=False)
        opt = LBFGS(ecf, logfile=None)
        opt.run(fmax=FMAX_RELAX, steps=300)

        energy_per_atom = atoms.get_potential_energy() / len(atoms)
        volume_per_atom = atoms.get_volume() / len(atoms)
        max_force = np.max(np.abs(atoms.get_forces()))
        converged = opt.converged()

        # Store relaxed structure back
        relaxed_structure = adaptor.get_structure(atoms)

        return {
            **candidate,
            "mace_energy_per_atom": energy_per_atom,
            "mace_volume_per_atom": volume_per_atom,
            "mace_max_force": max_force,
            "mace_converged": converged,
            "relaxed_structure": relaxed_structure,
            "relaxed_atoms": atoms,
        }
    except Exception as e:
        print(f"  MACE relax failed for {candidate['mp_id']}: {e}")
        return {
            **candidate,
            "mace_energy_per_atom": None,
            "mace_converged": False,
        }


def run_mace_relaxations(candidates):
    """Relax all candidates sequentially with MACE."""
    print(f"Step 2: MACE relaxation of {len(candidates)} structures ...")
    results = []
    for i, cand in enumerate(candidates):
        result = mace_relax(cand)
        status = "OK" if result.get("mace_converged") else "FAIL"
        if result.get("mace_energy_per_atom") is not None:
            print(f"  [{i+1}/{len(candidates)}] {cand['mp_id']} "
                  f"{cand['formula']:>12s}  E={result['mace_energy_per_atom']:.4f} "
                  f"eV/atom  {status}")
        results.append(result)

    ok = [r for r in results if r.get("mace_energy_per_atom") is not None]
    print(f"  {len(ok)}/{len(candidates)} relaxations succeeded.")
    return ok


# ================================================================== #
#  STEP 3: Filter by stability
# ================================================================== #
def apply_filters(results):
    """Apply thermodynamic and property filters."""
    filtered = []
    for r in results:
        # Stability filter (from MP data)
        if r["e_above_hull_mp"] > EHULL_CUTOFF:
            continue
        # Convergence filter
        if not r.get("mace_converged", False):
            continue
        filtered.append(r)

    print(f"Step 3: After stability/convergence filter: "
          f"{len(filtered)} candidates remain.")
    return filtered


# ================================================================== #
#  STEP 4: Compute bulk modulus for survivors
# ================================================================== #
def compute_bulk_modulus(result: dict) -> dict:
    """
    Estimate bulk modulus via Birch-Murnaghan fit.
    Apply isotropic strain, collect E(V), fit parabola.
    """
    from mace.calculators import mace_mp
    calc = mace_mp(model=MACE_MODEL, default_dtype="float64")

    atoms0 = result["relaxed_atoms"].copy()
    atoms0.calc = calc
    v0 = atoms0.get_volume()
    cell0 = atoms0.get_cell().copy()

    # Sample 7 volumes around equilibrium (+/- 3% isotropic strain)
    strains = np.linspace(-0.03, 0.03, 7)
    volumes, energies = [], []

    for s in strains:
        atoms_s = atoms0.copy()
        atoms_s.calc = calc
        atoms_s.set_cell(cell0 * (1 + s), scale_atoms=True)
        try:
            volumes.append(atoms_s.get_volume())
            energies.append(atoms_s.get_potential_energy())
        except Exception:
            continue

    if len(volumes) < 5:
        result["bulk_modulus_gpa"] = None
        return result

    volumes = np.array(volumes)
    energies = np.array(energies)

    # Fit E(V) = a*V^2 + b*V + c; B = V0 * d2E/dV2 = V0 * 2a
    coeffs = np.polyfit(volumes, energies, 2)
    bulk_mod = v0 * 2 * coeffs[0] * 160.2176634  # eV/Ang^3 -> GPa

    result["bulk_modulus_gpa"] = max(bulk_mod, 0.0)
    return result


def compute_properties(filtered):
    """Compute bulk modulus for all filtered candidates."""
    print(f"Step 4: Computing bulk modulus for {len(filtered)} candidates ...")
    results = []
    for i, r in enumerate(filtered):
        r = compute_bulk_modulus(r)
        bm = r.get("bulk_modulus_gpa")
        bm_str = f"{bm:.1f} GPa" if bm is not None else "N/A"
        print(f"  [{i+1}/{len(filtered)}] {r['mp_id']} {r['formula']:>12s}  "
              f"B={bm_str}")
        results.append(r)

    # Optional: filter by minimum bulk modulus
    if MIN_BULK_MODULUS is not None:
        results = [r for r in results
                   if r.get("bulk_modulus_gpa") is not None
                   and r["bulk_modulus_gpa"] >= MIN_BULK_MODULUS]
        print(f"  After B >= {MIN_BULK_MODULUS} GPa filter: {len(results)} remain.")

    return results


# ================================================================== #
#  STEP 5: Rank candidates with composite score
# ================================================================== #
def rank_candidates(results):
    """
    Build a DataFrame, compute a composite ranking score, sort, and export.
    Score = weighted sum of normalized properties.
    """
    rows = []
    for r in results:
        rows.append({
            "mp_id": r["mp_id"],
            "formula": r["formula"],
            "nsites": r["nsites"],
            "spacegroup": r["spacegroup"],
            "crystal_system": r["crystal_system"],
            "e_above_hull_mp": r["e_above_hull_mp"],
            "form_energy_mp": r["form_energy_mp"],
            "band_gap_mp": r["band_gap_mp"],
            "density_mp": r["density_mp"],
            "mace_energy_per_atom": r["mace_energy_per_atom"],
            "mace_volume_per_atom": r.get("mace_volume_per_atom"),
            "bulk_modulus_gpa": r.get("bulk_modulus_gpa"),
            "is_stable_mp": r.get("is_stable_mp"),
        })

    df = pd.DataFrame(rows)

    # --- Compute composite score ---
    # Normalize each property to [0, 1] range
    # Stability: lower e_above_hull -> higher score
    ehull_max = df["e_above_hull_mp"].max()
    if ehull_max > 0:
        df["score_stability"] = 1.0 - df["e_above_hull_mp"] / ehull_max
    else:
        df["score_stability"] = 1.0

    # Bulk modulus: higher -> higher score
    bm = df["bulk_modulus_gpa"].fillna(0)
    bm_max = bm.max()
    if bm_max > 0:
        df["score_bulk_mod"] = bm / bm_max
    else:
        df["score_bulk_mod"] = 0.0

    # Band gap: closer to target -> higher score
    bg_dev = (df["band_gap_mp"] - BANDGAP_TARGET).abs()
    bg_dev_max = bg_dev.max()
    if bg_dev_max > 0:
        df["score_bandgap"] = 1.0 - bg_dev / bg_dev_max
    else:
        df["score_bandgap"] = 1.0

    # Weighted composite
    df["composite_score"] = (
        WEIGHT_STABILITY * df["score_stability"]
        + WEIGHT_BULK_MOD * df["score_bulk_mod"]
        + WEIGHT_BANDGAP * df["score_bandgap"]
    )

    df.sort_values("composite_score", ascending=False, inplace=True)
    df.reset_index(drop=True, inplace=True)

    # Export
    csv_path = os.path.join(OUTPUT_DIR, "screening_ranked.csv")
    json_path = os.path.join(OUTPUT_DIR, "screening_ranked.json")
    df.to_csv(csv_path, index=False)
    df.to_json(json_path, orient="records", indent=2)

    print(f"\nStep 5: Ranked {len(df)} candidates. Top 10:")
    display_cols = ["mp_id", "formula", "e_above_hull_mp", "band_gap_mp",
                    "bulk_modulus_gpa", "composite_score"]
    print(df[display_cols].head(10).to_string(index=False))
    print(f"\nSaved {csv_path} and {json_path}")
    return df


# ================================================================== #
#  STEP 6: Visualization
# ================================================================== #
def generate_plots(df):
    """Generate distribution and correlation plots."""

    # --- Property distributions ---
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    axes[0, 0].hist(df["e_above_hull_mp"], bins=25, edgecolor="black", alpha=0.8)
    axes[0, 0].set_xlabel("Energy above hull (eV/atom)")
    axes[0, 0].set_ylabel("Count")
    axes[0, 0].set_title("Stability distribution")

    axes[0, 1].hist(df["band_gap_mp"], bins=25, edgecolor="black",
                    color="C1", alpha=0.8)
    axes[0, 1].set_xlabel("Band gap (eV)")
    axes[0, 1].set_ylabel("Count")
    axes[0, 1].set_title("Band gap distribution")

    bm = df["bulk_modulus_gpa"].dropna()
    axes[1, 0].hist(bm, bins=25, edgecolor="black", color="C2", alpha=0.8)
    axes[1, 0].set_xlabel("Bulk modulus (GPa)")
    axes[1, 0].set_ylabel("Count")
    axes[1, 0].set_title("Bulk modulus distribution")

    axes[1, 1].hist(df["composite_score"], bins=25, edgecolor="black",
                    color="C3", alpha=0.8)
    axes[1, 1].set_xlabel("Composite score")
    axes[1, 1].set_ylabel("Count")
    axes[1, 1].set_title("Ranking score distribution")

    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, "distributions.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved {path}")

    # --- Pareto front: stability vs stiffness ---
    fig, ax = plt.subplots(figsize=(9, 7))
    x = df["e_above_hull_mp"].values
    y = df["bulk_modulus_gpa"].values
    mask = ~np.isnan(y)
    x_p, y_p = x[mask], y[mask]
    formulas = df["formula"].values[mask]

    # Identify Pareto front (minimize x, maximize y)
    pareto = np.zeros(len(x_p), dtype=bool)
    sorted_idx = np.argsort(x_p)
    max_y = -np.inf
    for i in sorted_idx:
        if y_p[i] > max_y:
            pareto[i] = True
            max_y = y_p[i]

    ax.scatter(x_p[~pareto], y_p[~pareto], c="gray", alpha=0.4, s=30,
               label="Dominated")
    ax.scatter(x_p[pareto], y_p[pareto], c="red", s=80, zorder=5,
               edgecolors="black", label="Pareto front")

    # Connect Pareto points
    pidx = np.where(pareto)[0]
    pidx = pidx[np.argsort(x_p[pidx])]
    ax.plot(x_p[pidx], y_p[pidx], "r--", alpha=0.6)

    for i in pidx:
        ax.annotate(formulas[i], (x_p[i], y_p[i]),
                    textcoords="offset points", xytext=(5, 5), fontsize=7)

    ax.set_xlabel("Energy above hull (eV/atom)")
    ax.set_ylabel("Bulk modulus (GPa)")
    ax.set_title("Pareto front: stability vs. stiffness")
    ax.legend()
    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, "pareto_front.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved {path}")

    # --- Correlation matrix ---
    numeric_cols = ["e_above_hull_mp", "band_gap_mp", "bulk_modulus_gpa",
                    "density_mp", "mace_energy_per_atom"]
    corr = df[numeric_cols].corr()

    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(corr.values, cmap="RdBu_r", vmin=-1, vmax=1)
    ax.set_xticks(range(len(numeric_cols)))
    ax.set_yticks(range(len(numeric_cols)))
    labels = ["E_hull", "Band gap", "Bulk mod.", "Density", "MACE E"]
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_yticklabels(labels)
    for i in range(len(numeric_cols)):
        for j in range(len(numeric_cols)):
            ax.text(j, i, f"{corr.values[i, j]:.2f}",
                    ha="center", va="center", fontsize=10)
    plt.colorbar(im, shrink=0.8)
    ax.set_title("Property correlations")
    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, "correlations.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved {path}")


# ================================================================== #
#  STEP 7: Generate QE input files for top candidates
# ================================================================== #
def write_qe_inputs_for_top_n(results, df, n=TOP_N_FOR_DFT):
    """
    Write Quantum ESPRESSO pw.x input files for the top N ranked candidates.
    Uses pymatgen's PWInput for robust input generation.
    """
    from pymatgen.io.pwscf import PWInput

    qe_dir = os.path.join(OUTPUT_DIR, "qe_validation")
    os.makedirs(qe_dir, exist_ok=True)

    top_ids = df["mp_id"].head(n).tolist()
    result_map = {r["mp_id"]: r for r in results}

    print(f"\nStep 7: Writing QE inputs for top {n} candidates ...")

    for rank, mp_id in enumerate(top_ids):
        r = result_map.get(mp_id)
        if r is None or "relaxed_structure" not in r:
            print(f"  Skipping {mp_id}: no relaxed structure available.")
            continue

        structure = r["relaxed_structure"]
        formula = r["formula"]
        calc_dir = os.path.join(qe_dir, f"rank{rank+1:02d}_{mp_id}")
        os.makedirs(calc_dir, exist_ok=True)

        # Build pseudopotential map (SSSP convention)
        pseudo_map = {}
        for el in structure.composition.elements:
            sym = el.symbol
            pseudo_map[sym] = f"{sym}.pbe-n-rrkjus_psl.1.0.0.UPF"

        # Determine k-point grid from lattice parameters
        abc = structure.lattice.abc
        kpts = tuple(max(1, int(round(40 / a))) for a in abc)

        pw_input = PWInput(
            structure,
            pseudo=pseudo_map,
            control={
                "calculation": "vc-relax",
                "prefix": mp_id.replace("-", "_"),
                "outdir": "./tmp",
                "pseudo_dir": "./pseudo",
                "tprnfor": True,
                "tstress": True,
                "forc_conv_thr": 1.0e-4,
                "etot_conv_thr": 1.0e-6,
            },
            system={
                "ecutwfc": 60.0,
                "ecutrho": 480.0,
                "occupations": "smearing",
                "smearing": "mv",
                "degauss": 0.02,
            },
            electrons={
                "conv_thr": 1.0e-8,
                "mixing_beta": 0.4,
            },
            ions={"ion_dynamics": "bfgs"},
            cell={"cell_dynamics": "bfgs", "press_conv_thr": 0.1},
            kpoints_grid=kpts,
        )

        input_path = os.path.join(calc_dir, "relax.in")
        pw_input.write_file(input_path)

        # Save the structure as CIF for reference
        structure.to(os.path.join(calc_dir, f"{formula}.cif"))

        print(f"  Rank {rank+1}: {mp_id} ({formula}) -> {input_path}  "
              f"k-grid={kpts}")

    # Generate a batch run script
    run_script = "#!/bin/bash\n"
    run_script += "# Run QE vc-relax for all top candidates\n\n"
    for rank, mp_id in enumerate(top_ids):
        d = f"rank{rank+1:02d}_{mp_id}"
        run_script += f"echo 'Running {d} ...'\n"
        run_script += f"cd {qe_dir}/{d}\n"
        run_script += f"mpirun -np 4 pw.x -in relax.in > relax.out 2>&1\n"
        run_script += f"cd -\n\n"

    script_path = os.path.join(qe_dir, "run_all.sh")
    with open(script_path, "w") as f:
        f.write(run_script)
    os.chmod(script_path, 0o755)
    print(f"  Batch script: {script_path}")


# ================================================================== #
#  MAIN
# ================================================================== #
if __name__ == "__main__":
    # 1. Query
    candidates = query_materials_project()

    # 2. MACE relax
    relaxed = run_mace_relaxations(candidates)

    # 3. Filter
    filtered = apply_filters(relaxed)

    # 4. Compute properties
    with_props = compute_properties(filtered)

    # 5. Rank
    df = rank_candidates(with_props)

    # 6. Plots
    generate_plots(df)

    # 7. QE inputs for top candidates
    write_qe_inputs_for_top_n(with_props, df)

    print(f"\nScreening complete. {len(df)} candidates ranked.")
    print(f"Results in {OUTPUT_DIR}/")
```

### Post-DFT Comparison Script

After running the QE validation calculations, compare DFT results with MACE predictions.

```python
#!/usr/bin/env python3
"""
Compare QE DFT results with MACE screening predictions.
Run after the QE validation calculations complete.
"""

import os
import re
import glob
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUTPUT_DIR = "screening_results"
QE_DIR = os.path.join(OUTPUT_DIR, "qe_validation")


def parse_qe_output(pwo_file):
    """Extract total energy and stress from QE output."""
    energy_ry = None
    n_atoms = 0
    converged = False

    with open(pwo_file) as f:
        for line in f:
            if "!" in line and "total energy" in line:
                energy_ry = float(line.split("=")[1].split("Ry")[0].strip())
            if "number of atoms/cell" in line:
                n_atoms = int(line.split("=")[1].strip())
            if "convergence has been achieved" in line.lower():
                converged = True

    if energy_ry is not None:
        energy_ev = energy_ry * 13.605693123  # Ry -> eV
        energy_per_atom = energy_ev / n_atoms if n_atoms > 0 else None
    else:
        energy_ev = None
        energy_per_atom = None

    return {
        "qe_energy_total_eV": energy_ev,
        "qe_energy_per_atom": energy_per_atom,
        "qe_n_atoms": n_atoms,
        "qe_converged": converged,
    }


# Collect QE results
qe_rows = []
for calc_dir in sorted(glob.glob(os.path.join(QE_DIR, "rank*"))):
    dirname = os.path.basename(calc_dir)
    # Extract mp_id from directory name: rank01_mp-12345
    parts = dirname.split("_", 1)
    mp_id = parts[1] if len(parts) > 1 else dirname

    pwo_file = os.path.join(calc_dir, "relax.out")
    if not os.path.exists(pwo_file):
        print(f"  WARNING: {pwo_file} not found, skipping.")
        continue

    qe_data = parse_qe_output(pwo_file)
    qe_data["mp_id"] = mp_id
    qe_data["rank"] = dirname.split("_")[0]
    qe_rows.append(qe_data)

df_qe = pd.DataFrame(qe_rows)

# Merge with MACE screening results
df_mace = pd.read_csv(os.path.join(OUTPUT_DIR, "screening_ranked.csv"))
df_merged = df_mace.merge(df_qe, on="mp_id", how="inner")

print("MACE vs QE comparison:")
print(df_merged[["mp_id", "formula", "mace_energy_per_atom",
                  "qe_energy_per_atom", "bulk_modulus_gpa",
                  "composite_score"]].to_string(index=False))

# --- Parity plot ---
if len(df_merged) >= 2:
    fig, ax = plt.subplots(figsize=(7, 7))
    x = df_merged["mace_energy_per_atom"].values
    y = df_merged["qe_energy_per_atom"].values

    ax.scatter(x, y, s=60, edgecolors="black", zorder=5)
    for _, row in df_merged.iterrows():
        ax.annotate(row["formula"], (row["mace_energy_per_atom"],
                    row["qe_energy_per_atom"]),
                    textcoords="offset points", xytext=(5, 5), fontsize=8)

    # Perfect agreement line
    lim_min = min(x.min(), y.min()) - 0.1
    lim_max = max(x.max(), y.max()) + 0.1
    ax.plot([lim_min, lim_max], [lim_min, lim_max], "k--", alpha=0.5,
            label="Perfect agreement")

    ax.set_xlabel("MACE energy per atom (eV)")
    ax.set_ylabel("QE DFT energy per atom (eV)")
    ax.set_title("MACE vs QE: Energy Parity")
    ax.legend()
    ax.set_aspect("equal")
    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, "mace_vs_qe_parity.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved {path}")

df_merged.to_csv(os.path.join(OUTPUT_DIR, "mace_qe_comparison.csv"),
                  index=False)
print(f"Saved {OUTPUT_DIR}/mace_qe_comparison.csv")
```

## Key Parameters

| Parameter | Typical Value | Notes |
|---|---|---|
| `CHEMSYS_LIST` | List of `"A-B"` strings | Chemical systems to search. Use `"-".join(sorted(elements))` format. |
| `EHULL_CUTOFF` | 0.025--0.100 eV/atom | 0.025 = only hull-stable + marginal; 0.100 = include moderately metastable |
| `MAX_ATOMS` | 24--48 | Larger cells slow down MACE; keep small for rapid screening |
| `MACE_MODEL` | `"medium"` | `"small"` is fastest, `"large"` is most accurate; `"medium"` is the best trade-off |
| `FMAX_RELAX` | 0.01--0.05 eV/Ang | Tighter = more accurate but slower; 0.02 is a good screening default |
| `BANDGAP_RANGE` | `(1.0, 3.0)` | Filter by MP band gap (PBE, so underestimated by ~30-50%) |
| `WEIGHT_*` | Sum to 1.0 | Adjust relative importance of stability, stiffness, band gap in ranking |
| `TOP_N_FOR_DFT` | 3--10 | Number of top candidates to validate with QE |
| `ecutwfc` (QE) | 60--80 Ry | Must be converged for the pseudopotentials used |
| `k-grid` (QE) | Density ~40/a per direction | Automatic from lattice parameters; increase for metals |

## Interpreting Results

- **Composite score**: A value between 0 and 1 combining normalized stability, bulk modulus, and band gap proximity to target. Higher is better. The weighting is adjustable; report the weights used.
- **Pareto front**: Points on the red line in the Pareto plot represent the best trade-off between two objectives (e.g., stability vs. stiffness). No other candidate is simultaneously better in both properties.
- **MACE vs. MP energies**: MACE-MP-0 total energies are on a different reference scale than MP DFT energies. Compare trends (rankings) rather than absolute values.
- **MACE bulk modulus**: The parabolic E(V) fit gives a rough estimate (within ~20% of full elastic tensor calculation). Use for ranking, not for reporting precise values.
- **Band gaps from MP**: PBE band gaps systematically underestimate experimental values by 30--50%. A material reported as metallic (gap=0) in PBE may have a small gap experimentally.
- **QE validation**: The parity plot should show MACE and QE energies tracking each other, though with an offset. If a candidate's ranking changes drastically after DFT, investigate the structure for distortions or phase changes during relaxation.

## Common Issues

| Problem | Cause | Solution |
|---|---|---|
| `MP_API_KEY` not set | Environment variable missing | `export MP_API_KEY="your_key_here"` |
| MP query returns too many results | Broad search criteria | Tighten `EHULL_CUTOFF`, add `MAX_ATOMS` limit, restrict `CHEMSYS_LIST` |
| MACE relaxation diverges | Unusual chemistry or very strained structure | Skip with try/except; reduce `FMAX_RELAX`; try `MACE_MODEL="large"` |
| Negative bulk modulus from fit | Anharmonic E(V) curve or too few points | Increase number of strain points; narrow strain range to +/- 2% |
| All candidates filtered out | Filters too strict | Relax `EHULL_CUTOFF`, widen `BANDGAP_RANGE`, remove `MIN_BULK_MODULUS` |
| QE pseudopotential not found | SSSP filenames vary by library version | Verify filenames in pseudo directory; adjust `pseudo_map` dict |
| Ranking dominated by one property | Unbalanced weights | Adjust `WEIGHT_*` values; ensure properties have comparable ranges |
| Memory error with many candidates | Too many structures loaded simultaneously | Process in batches; reduce `MAX_CANDIDATES` |
