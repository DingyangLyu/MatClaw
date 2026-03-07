# High-Throughput Batch Screening with MACE and Materials Project

## When to Use

- You need to screen tens to hundreds of candidate materials for a target property (stability, band gap proxy, elastic stiffness, etc.).
- You want a fast first pass with a machine-learning potential (MACE-MP-0) before committing to expensive DFT.
- You are exploring a chemical space (e.g., all binary oxides of 3d transition metals) and need to rank candidates.

## Prerequisites

- `MP_API_KEY` environment variable set.
- Python packages: `mp-api`, `pymatgen`, `ase`, `mace-torch`, `numpy`, `pandas`, `matplotlib`.
- For final validation: Quantum ESPRESSO 7.5 (`pw.x`).

## Detailed Steps

### Step 1 -- Complete High-Throughput Screening Workflow

The script below performs the full pipeline: fetch from Materials Project, MACE-relax, filter, rank, and export.

```python
#!/usr/bin/env python3
"""
High-throughput batch screening workflow.

Pipeline:
  1. Fetch binary oxide candidates from Materials Project
  2. MACE-relax each structure and compute formation energy
  3. Filter by thermodynamic stability (e_above_hull < 0.1 eV/atom)
  4. Compute a mechanical proxy (bulk modulus via stress-strain) for survivors
  5. Rank and export results to CSV / JSON
  6. Generate plots (property distributions, Pareto front)
"""

import os
import json
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from multiprocessing import Pool, cpu_count

# -- ASE / MACE imports --
from ase import Atoms
from ase.optimize import LBFGS
from ase.io import write as ase_write
from ase.units import GPa

# -- pymatgen / MP imports --
from pymatgen.core import Structure
from pymatgen.io.ase import AseAtomsAdaptor

from mp_api.client import MPRester

warnings.filterwarnings("ignore")

# ------------------------------------------------------------------ #
#  Configuration
# ------------------------------------------------------------------ #
MP_API_KEY = os.environ.get("MP_API_KEY", None)
assert MP_API_KEY, "Set the MP_API_KEY environment variable first."

# Chemical system: 3d transition-metal binary oxides
ELEMENTS_WITH_O = [
    ("Ti", "O"), ("V", "O"), ("Cr", "O"), ("Mn", "O"),
    ("Fe", "O"), ("Co", "O"), ("Ni", "O"), ("Cu", "O"), ("Zn", "O"),
]
EHULL_CUTOFF = 0.1        # eV/atom -- stability filter
MAX_ATOMS = 48            # skip very large cells for speed
MACE_MODEL = "medium"     # "small", "medium", or "large"
FMAX = 0.05               # eV/Ang -- optimizer convergence
N_WORKERS = min(4, cpu_count())  # parallel MACE workers

# ------------------------------------------------------------------ #
#  Helper: pymatgen Structure -> ASE Atoms
# ------------------------------------------------------------------ #
adaptor = AseAtomsAdaptor()


def pmg_to_ase(structure: Structure) -> Atoms:
    """Convert pymatgen Structure to ASE Atoms."""
    return adaptor.get_atoms(structure)


# ------------------------------------------------------------------ #
#  Step 1 -- Fetch candidates from Materials Project
# ------------------------------------------------------------------ #
def fetch_candidates(chemsys_list):
    """
    Query MP for all binary oxides in the given chemical systems.
    Returns a list of dicts with mp_id, formula, structure, e_above_hull.
    """
    candidates = []
    with MPRester(MP_API_KEY) as mpr:
        for chemsys in chemsys_list:
            chemsys_str = "-".join(chemsys)
            docs = mpr.materials.summary.search(
                chemsys=chemsys_str,
                num_elements=2,          # binary only
                fields=[
                    "material_id", "formula_pretty", "structure",
                    "energy_above_hull", "formation_energy_per_atom",
                    "nsites",
                ],
            )
            for doc in docs:
                if doc.nsites > MAX_ATOMS:
                    continue
                candidates.append({
                    "mp_id": str(doc.material_id),
                    "formula": doc.formula_pretty,
                    "structure": doc.structure,
                    "e_above_hull_mp": doc.energy_above_hull,
                    "form_energy_mp": doc.formation_energy_per_atom,
                    "nsites": doc.nsites,
                })
    print(f"Fetched {len(candidates)} candidates from Materials Project.")
    return candidates


# ------------------------------------------------------------------ #
#  Step 2 -- MACE relax + formation-energy proxy
# ------------------------------------------------------------------ #
def mace_relax_single(candidate: dict) -> dict:
    """
    Relax one structure with MACE-MP-0 and return energy per atom.
    Runs in a worker process -- each worker loads its own calculator.
    """
    from mace.calculators import mace_mp  # import inside worker

    calc = mace_mp(model=MACE_MODEL, default_dtype="float64")
    atoms = pmg_to_ase(candidate["structure"])
    atoms.calc = calc

    try:
        opt = LBFGS(atoms, logfile=None)
        opt.run(fmax=FMAX, steps=200)
        energy_per_atom = atoms.get_potential_energy() / len(atoms)
        converged = opt.converged()
    except Exception as e:
        print(f"  MACE relax failed for {candidate['mp_id']}: {e}")
        return {**candidate, "mace_energy_per_atom": None, "mace_converged": False}

    result = {
        **candidate,
        "mace_energy_per_atom": energy_per_atom,
        "mace_converged": converged,
    }
    # Drop the pymatgen Structure (not serialisable for multiprocessing return)
    result["structure_ase"] = atoms
    return result


def run_mace_screening(candidates, n_workers=N_WORKERS):
    """Relax all candidates with MACE using multiprocessing."""
    print(f"Running MACE relaxation on {len(candidates)} structures "
          f"({n_workers} workers) ...")
    # Note: for MACE with GPU, serial may be faster (GPU contention).
    # With CPU-only, multiprocessing gives a nice speedup.
    if n_workers > 1:
        with Pool(n_workers) as pool:
            results = pool.map(mace_relax_single, candidates)
    else:
        results = [mace_relax_single(c) for c in candidates]

    ok = [r for r in results if r["mace_energy_per_atom"] is not None]
    print(f"  {len(ok)} / {len(candidates)} relaxations succeeded.")
    return ok


# ------------------------------------------------------------------ #
#  Step 3 -- Filter by stability
# ------------------------------------------------------------------ #
def filter_by_stability(results, cutoff=EHULL_CUTOFF):
    """Keep candidates whose MP e_above_hull is below the cutoff."""
    filtered = [r for r in results if r["e_above_hull_mp"] <= cutoff]
    print(f"After stability filter (e_above_hull < {cutoff} eV): "
          f"{len(filtered)} candidates remain.")
    return filtered


# ------------------------------------------------------------------ #
#  Step 4 -- Compute bulk modulus proxy via Birch-Murnaghan fit
# ------------------------------------------------------------------ #
def compute_bulk_modulus_proxy(result: dict) -> dict:
    """
    Estimate bulk modulus by computing energy at several volumes
    around the relaxed structure and fitting a parabola.

    B = V * d2E/dV2  (Birch-Murnaghan leading term)
    """
    from mace.calculators import mace_mp

    calc = mace_mp(model=MACE_MODEL, default_dtype="float64")
    atoms0 = result["structure_ase"].copy()
    atoms0.calc = calc
    v0 = atoms0.get_volume()
    e0 = atoms0.get_potential_energy()
    n = len(atoms0)

    strains = np.linspace(-0.03, 0.03, 7)  # +/- 3 % isotropic strain
    volumes, energies = [], []
    for s in strains:
        atoms_s = atoms0.copy()
        atoms_s.calc = calc
        cell = atoms0.get_cell()
        atoms_s.set_cell(cell * (1 + s), scale_atoms=True)
        volumes.append(atoms_s.get_volume())
        energies.append(atoms_s.get_potential_energy())

    volumes = np.array(volumes)
    energies = np.array(energies)

    # Fit parabola E(V) = a*V^2 + b*V + c
    coeffs = np.polyfit(volumes, energies, 2)
    a = coeffs[0]
    # B = V0 * d2E/dV2 = V0 * 2a   (convert eV/Ang^3 -> GPa)
    bulk_mod = v0 * 2 * a * 160.2176634  # eV/Ang^3 -> GPa

    result["bulk_modulus_gpa"] = max(bulk_mod, 0.0)  # clamp negative fits
    return result


def compute_bulk_moduli(filtered, n_workers=N_WORKERS):
    """Compute bulk modulus proxy for all filtered candidates."""
    print(f"Computing bulk modulus for {len(filtered)} candidates ...")
    if n_workers > 1:
        with Pool(n_workers) as pool:
            results = pool.map(compute_bulk_modulus_proxy, filtered)
    else:
        results = [compute_bulk_modulus_proxy(r) for r in filtered]
    return results


# ------------------------------------------------------------------ #
#  Step 5 -- Rank and export
# ------------------------------------------------------------------ #
def rank_and_export(results, out_csv="screening_results.csv",
                    out_json="screening_results.json"):
    """Build a DataFrame, rank by bulk modulus, and export."""
    rows = []
    for r in results:
        rows.append({
            "mp_id": r["mp_id"],
            "formula": r["formula"],
            "nsites": r["nsites"],
            "e_above_hull_mp": r["e_above_hull_mp"],
            "form_energy_mp": r["form_energy_mp"],
            "mace_energy_per_atom": r["mace_energy_per_atom"],
            "bulk_modulus_gpa": r.get("bulk_modulus_gpa", None),
        })

    df = pd.DataFrame(rows)
    df.sort_values("bulk_modulus_gpa", ascending=False, inplace=True)
    df.reset_index(drop=True, inplace=True)

    df.to_csv(out_csv, index=False)
    df.to_json(out_json, orient="records", indent=2)

    print(f"\nTop 10 candidates by bulk modulus:")
    print(df.head(10).to_string(index=False))
    print(f"\nResults saved to {out_csv} and {out_json}")
    return df


# ------------------------------------------------------------------ #
#  Step 6 -- Plots
# ------------------------------------------------------------------ #
def plot_distributions(df, prefix="screening"):
    """Property distribution histograms."""
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))

    axes[0].hist(df["e_above_hull_mp"], bins=20, edgecolor="black")
    axes[0].set_xlabel("E above hull (eV/atom)")
    axes[0].set_ylabel("Count")
    axes[0].set_title("Stability distribution")

    axes[1].hist(df["form_energy_mp"], bins=20, edgecolor="black", color="C1")
    axes[1].set_xlabel("Formation energy (eV/atom)")
    axes[1].set_ylabel("Count")
    axes[1].set_title("Formation energy distribution")

    bm = df["bulk_modulus_gpa"].dropna()
    axes[2].hist(bm, bins=20, edgecolor="black", color="C2")
    axes[2].set_xlabel("Bulk modulus (GPa)")
    axes[2].set_ylabel("Count")
    axes[2].set_title("Bulk modulus distribution")

    plt.tight_layout()
    plt.savefig(f"{prefix}_distributions.png", dpi=150)
    plt.close()
    print(f"Saved {prefix}_distributions.png")


def plot_pareto_front(df, prefix="screening"):
    """
    Pareto front for multi-objective screening:
      x-axis: formation energy (more negative = more stable, we want low)
      y-axis: bulk modulus (higher = stiffer, we want high)
    """
    x = df["form_energy_mp"].values
    y = df["bulk_modulus_gpa"].values
    mask = ~np.isnan(y)
    x, y = x[mask], y[mask]
    formulas = df["formula"].values[mask]

    # Identify Pareto-optimal points (minimize x, maximize y)
    pareto_mask = np.zeros(len(x), dtype=bool)
    sorted_idx = np.argsort(x)  # sort by formation energy ascending
    max_y = -np.inf
    for i in sorted_idx:
        if y[i] > max_y:
            pareto_mask[i] = True
            max_y = y[i]

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(x[~pareto_mask], y[~pareto_mask],
               c="gray", alpha=0.5, label="Dominated")
    ax.scatter(x[pareto_mask], y[pareto_mask],
               c="red", s=80, zorder=5, label="Pareto front")

    # Sort Pareto points for line
    pidx = np.where(pareto_mask)[0]
    pidx_sorted = pidx[np.argsort(x[pidx])]
    ax.plot(x[pidx_sorted], y[pidx_sorted], "r--", alpha=0.7)

    # Label Pareto points
    for i in pidx:
        ax.annotate(formulas[i], (x[i], y[i]),
                     textcoords="offset points", xytext=(5, 5), fontsize=7)

    ax.set_xlabel("Formation energy (eV/atom)")
    ax.set_ylabel("Bulk modulus (GPa)")
    ax.set_title("Pareto front: stability vs. stiffness")
    ax.legend()
    plt.tight_layout()
    plt.savefig(f"{prefix}_pareto.png", dpi=150)
    plt.close()
    print(f"Saved {prefix}_pareto.png")


# ------------------------------------------------------------------ #
#  Step 7 -- Detailed QE calculation for top N candidates
# ------------------------------------------------------------------ #
def write_qe_input(atoms, prefix, pseudo_dir="/opt/pseudo",
                   ecutwfc=60, ecutrho=480, kpts=(4, 4, 4)):
    """
    Generate a Quantum ESPRESSO pw.x input file for a single-point
    or relaxation calculation.
    """
    from ase.io.espresso import write_espresso_in

    input_data = {
        "control": {
            "calculation": "scf",
            "prefix": prefix,
            "outdir": f"./tmp_{prefix}",
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
            "conv_thr": 1.0e-6,
            "mixing_beta": 0.4,
        },
    }

    # Map element symbols to pseudopotential files
    pseudopotentials = {}
    for sym in set(atoms.get_chemical_symbols()):
        pseudopotentials[sym] = f"{sym}.pbe-n-kjpaw_psl.1.0.0.UPF"

    fname = f"{prefix}.pwi"
    write_espresso_in(
        open(fname, "w"),
        atoms,
        input_data=input_data,
        pseudopotentials=pseudopotentials,
        kpts=kpts,
    )
    print(f"  Wrote QE input: {fname}")
    return fname


def prepare_qe_for_top_n(results, n=3):
    """Write QE input files for the top N candidates."""
    print(f"\nPreparing QE inputs for top {n} candidates ...")
    for i, r in enumerate(results[:n]):
        prefix = r["mp_id"].replace("-", "_")
        atoms = r["structure_ase"]
        write_qe_input(atoms, prefix)


# ------------------------------------------------------------------ #
#  Main
# ------------------------------------------------------------------ #
if __name__ == "__main__":
    # 1. Fetch
    candidates = fetch_candidates(ELEMENTS_WITH_O)

    # 2. MACE relax
    relaxed = run_mace_screening(candidates, n_workers=N_WORKERS)

    # 3. Filter
    stable = filter_by_stability(relaxed, cutoff=EHULL_CUTOFF)

    # 4. Bulk modulus
    with_props = compute_bulk_moduli(stable, n_workers=N_WORKERS)

    # 5. Rank and export
    df = rank_and_export(with_props)

    # 6. Plots
    plot_distributions(df)
    plot_pareto_front(df)

    # 7. QE for top candidates
    prepare_qe_for_top_n(with_props, n=3)

    print("\nDone.")
```

### Step 2 -- Running QE Validation on Top Candidates

After the screening script generates `.pwi` input files, run them:

```bash
# Run QE for each top candidate
for f in *.pwi; do
    prefix="${f%.pwi}"
    echo "Running QE for $prefix ..."
    mpirun -np 4 pw.x -in "$f" > "${prefix}.pwo" 2>&1
done
```

Parse the QE output to extract total energies and compare with MACE predictions:

```python
#!/usr/bin/env python3
"""Parse QE outputs and compare with MACE energies."""

import re
import glob
import pandas as pd

def parse_qe_energy(pwo_file):
    """Extract total energy from pw.x output."""
    energy = None
    with open(pwo_file) as f:
        for line in f:
            if "!" in line and "total energy" in line:
                energy = float(line.split("=")[1].split("Ry")[0].strip())
                energy *= 13.605693123   # Ry -> eV
    return energy

rows = []
for pwo in sorted(glob.glob("*.pwo")):
    mp_id = pwo.replace(".pwo", "").replace("_", "-")
    e_qe = parse_qe_energy(pwo)
    rows.append({"mp_id": mp_id, "qe_total_energy_eV": e_qe})

df_qe = pd.DataFrame(rows)
df_mace = pd.read_csv("screening_results.csv")
df_merged = df_mace.merge(df_qe, on="mp_id", how="left")
print(df_merged[["mp_id", "formula", "mace_energy_per_atom",
                  "qe_total_energy_eV", "bulk_modulus_gpa"]])
```

## Key Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `ELEMENTS_WITH_O` | 3d TM oxides | List of (M, O) tuples defining the chemical space |
| `EHULL_CUTOFF` | 0.1 eV/atom | Maximum energy above convex hull to keep |
| `MAX_ATOMS` | 48 | Skip cells with more atoms (speed) |
| `MACE_MODEL` | `"medium"` | MACE-MP-0 model size (`"small"`, `"medium"`, `"large"`) |
| `FMAX` | 0.05 eV/A | Force convergence for MACE relaxation |
| `N_WORKERS` | 4 | Number of parallel workers for MACE |
| `ecutwfc` | 60 Ry | QE plane-wave cutoff |
| `kpts` | (4,4,4) | QE k-point grid |

## Interpreting Results

- **Pareto front** -- Points on the red line represent the best trade-off between stability (low formation energy) and stiffness (high bulk modulus). Candidates above and to the left are optimal.
- **MACE vs. MP energies** -- MACE-MP-0 energies correlate well with DFT but have an MAE of roughly 10-30 meV/atom. Use them for ranking, not absolute values.
- **Bulk modulus proxy** -- The parabolic E(V) fit gives a reasonable first estimate but is less accurate than a full elastic tensor calculation. Treat values as relative rankings.
- **QE validation** -- Always run DFT on your top 3-5 candidates to confirm MACE predictions before drawing conclusions.

## Common Issues

| Issue | Solution |
|-------|----------|
| `MP_API_KEY` not set | `export MP_API_KEY="your_key_here"` before running |
| MACE relaxation diverges | Increase `steps` or reduce `FMAX`; skip structures with very high initial forces |
| Out of memory with multiprocessing | Reduce `N_WORKERS` or set `N_WORKERS=1` for serial execution |
| GPU contention with multiprocessing | Use `N_WORKERS=1` when MACE runs on GPU; multiprocessing is for CPU mode |
| Too many candidates fetched | Tighten `MAX_ATOMS`, reduce the element list, or add `e_above_hull` pre-filter in the MP query |
| QE pseudopotential not found | Verify filenames match your pseudopotential library; adjust the `pseudopotentials` dict |
| Negative bulk modulus | The parabolic fit failed (too few points or anharmonic region); increase the number of strain points or narrow the strain range |
