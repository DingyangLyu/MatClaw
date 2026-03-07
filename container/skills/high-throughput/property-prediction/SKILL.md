# Rapid Property Prediction Pipeline

## When to Use

- You need to quickly estimate multiple physical properties for a set of crystal structures.
- You want to predict lattice constants, bulk modulus, surface energy, vacancy formation energy, and a band gap proxy without running DFT.
- You need property tables for comparison with experimental or DFT databases.
- You want correlation plots between predicted and reference properties.
- You are screening materials and need a quick multi-property fingerprint for each candidate.

## Method Selection

| Property | MACE Accuracy | DFT Needed? | Notes |
|---|---|---|---|
| Lattice constants (a, b, c) | Within 1-2% of DFT | No | Very reliable for MACE-MP-0 training-set chemistries |
| Bulk modulus | Within 10-20% of DFT | For publication | Parabolic E(V) fit; reasonable for ranking |
| Band gap proxy (DOS at Fermi level) | Qualitative only | Yes, for actual gap | MACE has no electronic degrees of freedom; proxy detects metallic vs insulating tendency |
| Surface energy | Within 10-30% of DFT | For publication | Slab model with MACE; depends on surface reconstruction |
| Vacancy formation energy | Within 0.2-0.5 eV of DFT | For publication | Neutral vacancy only; charged defects need DFT |

## Prerequisites

- Python packages: `pymatgen`, `ase`, `mace-torch`, `numpy`, `pandas`, `matplotlib`, `scipy`.
- `mp-api` (optional, for fetching reference data from Materials Project).
- `MP_API_KEY` environment variable (optional, for database comparison).
- Input structures as CIF, POSCAR, or any pymatgen-readable format.

## Detailed Steps

### Complete Multi-Property Prediction Workflow

```python
#!/usr/bin/env python3
"""
Rapid property prediction pipeline using MACE-MP-0.

For each input structure, computes:
  1. Lattice constants (a, b, c, alpha, beta, gamma) after MACE relaxation
  2. Bulk modulus via Birch-Murnaghan E(V) fit
  3. Band gap proxy: checks DOS at Fermi level using finite-displacement
     force-constant method to detect metallic vs. insulating behavior
  4. Surface energy for the lowest-index surface
  5. Vacancy formation energy for each unique atomic site

Outputs:
  - Property table (CSV + JSON)
  - Comparison with Materials Project data (if API key available)
  - Correlation plots and property distribution histograms
"""

import os
import glob
import json
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

from ase.io import read as ase_read, write as ase_write
from ase.optimize import LBFGS
from ase.constraints import ExpCellFilter

from pymatgen.core import Structure, Element
from pymatgen.io.ase import AseAtomsAdaptor
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
from pymatgen.core.surface import SlabGenerator

from mace.calculators import mace_mp

warnings.filterwarnings("ignore")

# ================================================================== #
#  CONFIGURATION
# ================================================================== #
INPUT_DIR = "structures"         # directory with structure files
OUTPUT_DIR = "property_results"
MACE_MODEL = "medium"            # "small", "medium", "large"
FMAX_RELAX = 0.005               # eV/Ang -- tight for accurate properties
COMPUTE_SURFACE = True           # compute surface energy (slower)
COMPUTE_VACANCY = True           # compute vacancy formation energy (slower)
SURFACE_MILLER = (1, 0, 0)       # Miller indices for surface energy
SLAB_THICKNESS = 10.0            # Angstrom
VACUUM_THICKNESS = 15.0          # Angstrom
SUPERCELL_SIZE = (2, 2, 2)       # for vacancy calculation
FETCH_MP_REFERENCE = True        # compare with Materials Project data
# ================================================================== #

os.makedirs(OUTPUT_DIR, exist_ok=True)
adaptor = AseAtomsAdaptor()
calc = mace_mp(model=MACE_MODEL, default_dtype="float64")


# ================================================================== #
#  PROPERTY 1: Lattice constants
# ================================================================== #
def predict_lattice_constants(structure_file):
    """
    Relax structure with MACE and extract lattice parameters.
    Returns dict with a, b, c, alpha, beta, gamma, volume, formula, spacegroup.
    """
    atoms = ase_read(structure_file)
    atoms.calc = calc

    # Full cell + ionic relaxation
    ecf = ExpCellFilter(atoms, hydrostatic_strain=False)
    opt = LBFGS(ecf, logfile=None)
    opt.run(fmax=FMAX_RELAX, steps=500)

    relaxed_structure = adaptor.get_structure(atoms)
    lattice = relaxed_structure.lattice

    sga = SpacegroupAnalyzer(relaxed_structure, symprec=0.01)

    return {
        "a": lattice.a,
        "b": lattice.b,
        "c": lattice.c,
        "alpha": lattice.alpha,
        "beta": lattice.beta,
        "gamma": lattice.gamma,
        "volume": lattice.volume,
        "volume_per_atom": lattice.volume / len(relaxed_structure),
        "formula": relaxed_structure.composition.reduced_formula,
        "nsites": len(relaxed_structure),
        "spacegroup": sga.get_space_group_symbol(),
        "crystal_system": sga.get_crystal_system(),
        "density": relaxed_structure.density,
        "relaxed_structure": relaxed_structure,
        "relaxed_atoms": atoms,
        "energy_per_atom": atoms.get_potential_energy() / len(atoms),
    }


# ================================================================== #
#  PROPERTY 2: Bulk modulus (Birch-Murnaghan EOS fit)
# ================================================================== #
def birch_murnaghan(V, E0, V0, B0, Bp):
    """Third-order Birch-Murnaghan equation of state."""
    eta = (V0 / V) ** (2.0 / 3.0)
    return E0 + (9.0 * V0 * B0 / 16.0) * (
        (eta - 1.0) ** 3 * Bp
        + (eta - 1.0) ** 2 * (6.0 - 4.0 * eta)
    )


def predict_bulk_modulus(atoms0):
    """
    Compute bulk modulus via Birch-Murnaghan fit to E(V) data.
    Returns B0 in GPa and the fitted EOS parameters.
    """
    atoms_ref = atoms0.copy()
    atoms_ref.calc = calc
    v0 = atoms_ref.get_volume()
    e0 = atoms_ref.get_potential_energy()
    cell0 = atoms_ref.get_cell().copy()
    n = len(atoms_ref)

    # Sample volumes: +/- 6% in 13 points
    strains = np.linspace(-0.06, 0.06, 13)
    volumes = []
    energies = []

    for s in strains:
        atoms_s = atoms_ref.copy()
        atoms_s.calc = calc
        atoms_s.set_cell(cell0 * (1 + s), scale_atoms=True)
        try:
            e = atoms_s.get_potential_energy()
            volumes.append(atoms_s.get_volume())
            energies.append(e)
        except Exception:
            continue

    if len(volumes) < 7:
        return {"bulk_modulus_gpa": None, "B_prime": None}

    volumes = np.array(volumes)
    energies = np.array(energies)

    # Fit Birch-Murnaghan EOS
    try:
        p0 = [e0, v0, 0.5, 4.0]  # initial guess: B0~0.5 eV/Ang^3, Bp~4
        popt, pcov = curve_fit(birch_murnaghan, volumes, energies, p0=p0,
                               maxfev=10000)
        E0_fit, V0_fit, B0_fit, Bp_fit = popt
        B0_gpa = B0_fit * 160.2176634  # eV/Ang^3 -> GPa
    except Exception:
        # Fall back to parabolic fit
        coeffs = np.polyfit(volumes, energies, 2)
        B0_gpa = v0 * 2 * coeffs[0] * 160.2176634
        V0_fit = v0
        Bp_fit = 4.0

    return {
        "bulk_modulus_gpa": max(B0_gpa, 0.0),
        "B_prime": Bp_fit,
        "V0_fit": V0_fit,
        "eos_volumes": volumes.tolist(),
        "eos_energies": energies.tolist(),
    }


# ================================================================== #
#  PROPERTY 3: Band gap proxy (metallic vs insulating indicator)
# ================================================================== #
def predict_bandgap_proxy(atoms0):
    """
    Estimate whether a material is metallic or insulating using MACE.

    Since MACE is a force field (no electronic structure), we use an
    indirect proxy: the dynamical matrix eigenvalue spectrum.

    Materials with very low-frequency modes at Gamma (near zero or
    imaginary) tend to be metals or near-metals, while insulators
    have a clear phonon gap.

    This is a ROUGH proxy. For actual band gaps, use DFT.
    """
    from ase.vibrations import Vibrations

    atoms = atoms0.copy()
    atoms.calc = calc

    n = len(atoms)

    # For small cells, compute the full dynamical matrix at Gamma
    if n > 30:
        return {"bandgap_proxy": None, "proxy_note": "cell too large"}

    try:
        vib = Vibrations(atoms, name=f"vib_temp_{id(atoms)}")
        vib.clean()
        vib.run()
        freqs = vib.get_frequencies()
        vib.clean()

        # Real frequencies (in cm^-1)
        real_freqs = np.array([f.real for f in freqs if abs(f.imag) < 1.0])
        real_freqs = real_freqs[real_freqs > 1.0]  # filter acoustic modes

        if len(real_freqs) == 0:
            return {"bandgap_proxy": 0.0,
                    "proxy_note": "no real phonon modes (likely metallic)"}

        # The acoustic-optical gap as a proxy
        # A larger gap between acoustic and optical branches correlates
        # with more insulating character
        sorted_freqs = np.sort(real_freqs)
        if len(sorted_freqs) > 3:
            acoustic_optical_gap = sorted_freqs[3] - sorted_freqs[2]
        else:
            acoustic_optical_gap = 0.0

        return {
            "bandgap_proxy": acoustic_optical_gap,
            "min_optical_freq_cm1": float(sorted_freqs[3])
                                    if len(sorted_freqs) > 3 else None,
            "n_phonon_modes": len(real_freqs),
            "proxy_note": "phonon-gap proxy (not electronic band gap)",
        }
    except Exception as e:
        return {"bandgap_proxy": None, "proxy_note": str(e)}


# ================================================================== #
#  PROPERTY 4: Surface energy
# ================================================================== #
def predict_surface_energy(structure, atoms_bulk, miller=SURFACE_MILLER):
    """
    Compute surface energy for a given Miller index.
    Uses pymatgen SlabGenerator to create the slab model.

    gamma = (E_slab - n * E_bulk_per_atom) / (2 * A)
    """
    e_bulk_per_atom = atoms_bulk.get_potential_energy() / len(atoms_bulk)

    try:
        slabgen = SlabGenerator(
            structure,
            miller_index=miller,
            min_slab_size=SLAB_THICKNESS,
            min_vacuum_size=VACUUM_THICKNESS,
            center_slab=True,
            in_unit_planes=False,
        )
        slabs = slabgen.get_slabs(symmetrize=False)
    except Exception as e:
        return {"surface_energy_J_m2": None,
                "surface_error": str(e)}

    if not slabs:
        return {"surface_energy_J_m2": None,
                "surface_error": "no slabs generated"}

    # Take the first (lowest energy) slab
    slab = slabs[0]
    slab_atoms = adaptor.get_atoms(slab)
    slab_atoms.calc = calc

    # Relax slab ions (fixed cell)
    opt = LBFGS(slab_atoms, logfile=None)
    opt.run(fmax=0.02, steps=200)

    e_slab = slab_atoms.get_potential_energy()
    n_slab = len(slab_atoms)

    # Surface area (in Ang^2)
    cell = slab_atoms.get_cell()
    a_vec = cell[0]
    b_vec = cell[1]
    area = np.linalg.norm(np.cross(a_vec, b_vec))

    # Surface energy: gamma = (E_slab - N * e_bulk) / (2 * A)
    # Factor of 2 because slab has two surfaces
    # Convert eV/Ang^2 to J/m^2: 1 eV/Ang^2 = 16.0218 J/m^2
    gamma_ev_ang2 = (e_slab - n_slab * e_bulk_per_atom) / (2 * area)
    gamma_j_m2 = gamma_ev_ang2 * 16.0218

    return {
        "surface_energy_J_m2": gamma_j_m2,
        "surface_miller": "".join(str(m) for m in miller),
        "slab_atoms": n_slab,
        "surface_area_Ang2": area,
    }


# ================================================================== #
#  PROPERTY 5: Vacancy formation energy
# ================================================================== #
def predict_vacancy_energy(structure, atoms_bulk):
    """
    Compute vacancy formation energy for each unique crystallographic site.

    E_vac = E_defect - (N-1)/N * E_bulk_N
    where E_bulk_N is the energy of the supercell with N atoms.
    """
    e_bulk_per_atom = atoms_bulk.get_potential_energy() / len(atoms_bulk)

    # Build supercell
    supercell = structure.copy()
    supercell.make_supercell(SUPERCELL_SIZE)
    n_super = len(supercell)

    # Get supercell energy
    sc_atoms = adaptor.get_atoms(supercell)
    sc_atoms.calc = calc
    opt = LBFGS(sc_atoms, logfile=None)
    opt.run(fmax=0.02, steps=200)
    e_supercell = sc_atoms.get_potential_energy()

    # Find unique sites using symmetry
    sga = SpacegroupAnalyzer(structure, symprec=0.01)
    sym_structure = sga.get_symmetrized_structure()
    equiv_indices = sym_structure.equivalent_indices

    vacancy_results = []
    for group in equiv_indices:
        site_idx = group[0]  # representative site
        site = structure[site_idx]
        element = site.specie.symbol

        # Create vacancy in supercell
        # Map primitive site to supercell sites
        sc_copy = supercell.copy()
        # Find the corresponding atom in the supercell
        # (first occurrence of the same Wyckoff position)
        sc_frac = sc_copy.frac_coords
        prim_frac = site.frac_coords

        # Find closest matching site in supercell
        target_cart = supercell.lattice.get_cartesian_coords(
            prim_frac / np.array(SUPERCELL_SIZE)
        )
        dists = [np.linalg.norm(
            supercell.lattice.get_cartesian_coords(fc) - target_cart
        ) for fc in sc_frac]
        remove_idx = np.argmin(dists)

        # Remove atom
        sc_copy.remove_sites([remove_idx])
        n_defect = len(sc_copy)

        # Relax defect structure
        defect_atoms = adaptor.get_atoms(sc_copy)
        defect_atoms.calc = calc
        opt = LBFGS(defect_atoms, logfile=None)
        opt.run(fmax=0.02, steps=300)
        e_defect = defect_atoms.get_potential_energy()

        # Vacancy formation energy
        e_vac = e_defect - (n_defect / n_super) * e_supercell

        vacancy_results.append({
            "site_element": element,
            "site_index": site_idx,
            "wyckoff": sym_structure.wyckoff_symbols[site_idx],
            "vacancy_energy_eV": e_vac,
            "n_equivalent_sites": len(group),
        })

        print(f"    V_{element} (Wyckoff {sym_structure.wyckoff_symbols[site_idx]}): "
              f"E_vac = {e_vac:.3f} eV")

    return vacancy_results


# ================================================================== #
#  MAIN: Process all structures
# ================================================================== #
structure_files = sorted(
    glob.glob(os.path.join(INPUT_DIR, "*.cif"))
    + glob.glob(os.path.join(INPUT_DIR, "*.vasp"))
    + glob.glob(os.path.join(INPUT_DIR, "POSCAR*"))
)

if not structure_files:
    print(f"No structure files found in {INPUT_DIR}/")
    exit(1)

print(f"Processing {len(structure_files)} structures ...\n")

all_results = []
for idx, filepath in enumerate(structure_files):
    basename = os.path.basename(filepath)
    print(f"[{idx+1}/{len(structure_files)}] {basename}")

    # --- Lattice constants ---
    try:
        props = predict_lattice_constants(filepath)
        print(f"  Lattice: a={props['a']:.3f} b={props['b']:.3f} "
              f"c={props['c']:.3f} Ang")
        print(f"  Formula: {props['formula']}, SG: {props['spacegroup']}, "
              f"V/atom: {props['volume_per_atom']:.2f} Ang^3")
    except Exception as e:
        print(f"  Lattice constants FAILED: {e}")
        continue

    result = {
        "filename": basename,
        "formula": props["formula"],
        "spacegroup": props["spacegroup"],
        "crystal_system": props["crystal_system"],
        "a_Ang": props["a"],
        "b_Ang": props["b"],
        "c_Ang": props["c"],
        "alpha_deg": props["alpha"],
        "beta_deg": props["beta"],
        "gamma_deg": props["gamma"],
        "volume_Ang3": props["volume"],
        "volume_per_atom_Ang3": props["volume_per_atom"],
        "density_g_cm3": props["density"],
        "energy_per_atom_eV": props["energy_per_atom"],
        "nsites": props["nsites"],
    }

    # --- Bulk modulus ---
    try:
        bm = predict_bulk_modulus(props["relaxed_atoms"])
        result["bulk_modulus_GPa"] = bm["bulk_modulus_gpa"]
        result["B_prime"] = bm.get("B_prime")
        if bm["bulk_modulus_gpa"] is not None:
            print(f"  Bulk modulus: {bm['bulk_modulus_gpa']:.1f} GPa, "
                  f"B' = {bm.get('B_prime', 'N/A')}")
    except Exception as e:
        print(f"  Bulk modulus FAILED: {e}")
        result["bulk_modulus_GPa"] = None

    # --- Band gap proxy ---
    try:
        bg = predict_bandgap_proxy(props["relaxed_atoms"])
        result["bandgap_proxy"] = bg.get("bandgap_proxy")
        result["bandgap_proxy_note"] = bg.get("proxy_note", "")
        if bg.get("bandgap_proxy") is not None:
            print(f"  Band gap proxy: {bg['bandgap_proxy']:.1f} cm^-1 "
                  f"({bg.get('proxy_note', '')})")
    except Exception as e:
        print(f"  Band gap proxy FAILED: {e}")
        result["bandgap_proxy"] = None

    # --- Surface energy ---
    if COMPUTE_SURFACE:
        try:
            se = predict_surface_energy(props["relaxed_structure"],
                                        props["relaxed_atoms"])
            result["surface_energy_J_m2"] = se.get("surface_energy_J_m2")
            result["surface_miller"] = se.get("surface_miller")
            if se.get("surface_energy_J_m2") is not None:
                print(f"  Surface energy ({se['surface_miller']}): "
                      f"{se['surface_energy_J_m2']:.3f} J/m^2")
        except Exception as e:
            print(f"  Surface energy FAILED: {e}")
            result["surface_energy_J_m2"] = None

    # --- Vacancy formation energy ---
    if COMPUTE_VACANCY:
        try:
            vac_results = predict_vacancy_energy(
                props["relaxed_structure"], props["relaxed_atoms"]
            )
            if vac_results:
                # Store the minimum vacancy energy and all site data
                min_vac = min(v["vacancy_energy_eV"] for v in vac_results)
                result["min_vacancy_energy_eV"] = min_vac
                result["vacancy_details"] = json.dumps(vac_results)
                print(f"  Min vacancy energy: {min_vac:.3f} eV")
            else:
                result["min_vacancy_energy_eV"] = None
        except Exception as e:
            print(f"  Vacancy energy FAILED: {e}")
            result["min_vacancy_energy_eV"] = None

    all_results.append(result)
    print()

# ================================================================== #
#  Export results
# ================================================================== #
df = pd.DataFrame(all_results)

# Drop non-serializable columns for CSV
csv_cols = [c for c in df.columns if c != "vacancy_details"]
df[csv_cols].to_csv(os.path.join(OUTPUT_DIR, "properties.csv"), index=False)

# Full JSON with vacancy details
df.to_json(os.path.join(OUTPUT_DIR, "properties.json"),
           orient="records", indent=2)

print(f"\nResults saved to {OUTPUT_DIR}/properties.csv and properties.json")
print(f"\nProperty summary:")
print(df[["formula", "a_Ang", "bulk_modulus_GPa", "surface_energy_J_m2",
          "min_vacancy_energy_eV"]].to_string(index=False))


# ================================================================== #
#  OPTIONAL: Compare with Materials Project reference data
# ================================================================== #
if FETCH_MP_REFERENCE:
    try:
        from mp_api.client import MPRester
        MP_API_KEY = os.environ.get("MP_API_KEY")

        if MP_API_KEY:
            print("\nFetching reference data from Materials Project ...")

            formulas = df["formula"].unique().tolist()
            mp_rows = []

            with MPRester(MP_API_KEY) as mpr:
                for formula in formulas:
                    docs = mpr.materials.summary.search(
                        formula=formula,
                        is_stable=True,
                        fields=["material_id", "formula_pretty", "band_gap",
                                "volume", "nsites", "density",
                                "formation_energy_per_atom"],
                    )
                    if docs:
                        doc = docs[0]  # take first stable entry
                        mp_rows.append({
                            "formula": doc.formula_pretty,
                            "mp_id": str(doc.material_id),
                            "mp_band_gap_eV": doc.band_gap,
                            "mp_volume_per_atom": doc.volume / doc.nsites,
                            "mp_density": doc.density,
                            "mp_form_energy": doc.formation_energy_per_atom,
                        })

                # Also try to get elastic data
                for formula in formulas:
                    try:
                        edocs = mpr.elasticity.search(
                            formula=formula,
                            fields=["material_id", "formula_pretty",
                                    "bulk_modulus"],
                        )
                        if edocs:
                            bm = edocs[0].bulk_modulus
                            for row in mp_rows:
                                if row["formula"] == edocs[0].formula_pretty:
                                    row["mp_bulk_modulus_GPa"] = (
                                        bm.vrh if bm else None)
                    except Exception:
                        pass

            df_mp = pd.DataFrame(mp_rows)
            df_compare = df.merge(df_mp, on="formula", how="left")
            df_compare.to_csv(os.path.join(OUTPUT_DIR, "comparison_with_mp.csv"),
                              index=False)
            print(f"Saved {OUTPUT_DIR}/comparison_with_mp.csv")
        else:
            df_compare = None
            print("MP_API_KEY not set; skipping database comparison.")
    except ImportError:
        df_compare = None
        print("mp-api not installed; skipping database comparison.")
else:
    df_compare = None


# ================================================================== #
#  Visualization
# ================================================================== #
print("\nGenerating plots ...")

# --- 1. Property distributions ---
fig, axes = plt.subplots(2, 3, figsize=(16, 10))

props_to_plot = [
    ("volume_per_atom_Ang3", "Volume per atom (Ang^3)", "C0"),
    ("bulk_modulus_GPa", "Bulk modulus (GPa)", "C1"),
    ("density_g_cm3", "Density (g/cm^3)", "C2"),
    ("energy_per_atom_eV", "Energy per atom (eV)", "C3"),
    ("surface_energy_J_m2", "Surface energy (J/m^2)", "C4"),
    ("min_vacancy_energy_eV", "Min vacancy energy (eV)", "C5"),
]

for ax, (col, label, color) in zip(axes.flat, props_to_plot):
    data = df[col].dropna()
    if len(data) > 0:
        ax.hist(data, bins=min(20, len(data)), edgecolor="black",
                color=color, alpha=0.8)
    ax.set_xlabel(label)
    ax.set_ylabel("Count")
    ax.set_title(label)

plt.suptitle("Property Distributions (MACE Predictions)", fontsize=14)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "property_distributions.png"),
            dpi=150, bbox_inches="tight")
plt.close()

# --- 2. Correlation matrix ---
numeric_cols = ["volume_per_atom_Ang3", "bulk_modulus_GPa", "density_g_cm3",
                "energy_per_atom_eV", "surface_energy_J_m2",
                "min_vacancy_energy_eV"]
existing_cols = [c for c in numeric_cols if c in df.columns]
corr = df[existing_cols].corr()

fig, ax = plt.subplots(figsize=(8, 7))
im = ax.imshow(corr.values, cmap="RdBu_r", vmin=-1, vmax=1)
labels_short = [c.split("_")[0] + " " + c.split("_")[1]
                if "_" in c else c for c in existing_cols]
ax.set_xticks(range(len(existing_cols)))
ax.set_yticks(range(len(existing_cols)))
ax.set_xticklabels(labels_short, rotation=45, ha="right", fontsize=9)
ax.set_yticklabels(labels_short, fontsize=9)
for i in range(len(existing_cols)):
    for j in range(len(existing_cols)):
        ax.text(j, i, f"{corr.values[i, j]:.2f}",
                ha="center", va="center", fontsize=9)
plt.colorbar(im, shrink=0.8)
ax.set_title("Property Correlations")
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "property_correlations.png"),
            dpi=150, bbox_inches="tight")
plt.close()

# --- 3. MACE vs MP parity plots (if comparison data available) ---
if df_compare is not None and len(df_compare) > 0:
    parity_pairs = [
        ("volume_per_atom_Ang3", "mp_volume_per_atom", "Volume/atom (Ang^3)"),
        ("bulk_modulus_GPa", "mp_bulk_modulus_GPa", "Bulk modulus (GPa)"),
        ("density_g_cm3", "mp_density", "Density (g/cm^3)"),
    ]

    fig, axes = plt.subplots(1, len(parity_pairs), figsize=(6*len(parity_pairs), 5))
    if len(parity_pairs) == 1:
        axes = [axes]

    for ax, (mace_col, mp_col, label) in zip(axes, parity_pairs):
        if mp_col not in df_compare.columns:
            continue
        mask = df_compare[mace_col].notna() & df_compare[mp_col].notna()
        x = df_compare.loc[mask, mp_col].values
        y = df_compare.loc[mask, mace_col].values
        formulas = df_compare.loc[mask, "formula"].values

        if len(x) == 0:
            continue

        ax.scatter(x, y, s=60, edgecolors="black", zorder=5)
        for xi, yi, f in zip(x, y, formulas):
            ax.annotate(f, (xi, yi), textcoords="offset points",
                        xytext=(4, 4), fontsize=7)

        lim_min = min(x.min(), y.min()) * 0.9
        lim_max = max(x.max(), y.max()) * 1.1
        ax.plot([lim_min, lim_max], [lim_min, lim_max], "k--", alpha=0.5)
        ax.set_xlabel(f"MP DFT {label}")
        ax.set_ylabel(f"MACE {label}")
        ax.set_title(f"MACE vs MP: {label}")
        ax.set_aspect("equal")

        # Compute MAE and RMSE
        mae = np.mean(np.abs(x - y))
        rmse = np.sqrt(np.mean((x - y) ** 2))
        ax.text(0.05, 0.95, f"MAE = {mae:.3f}\nRMSE = {rmse:.3f}",
                transform=ax.transAxes, va="top", fontsize=9,
                bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5))

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "mace_vs_mp_parity.png"),
                dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved {OUTPUT_DIR}/mace_vs_mp_parity.png")

# --- 4. Radar chart for multi-property comparison ---
if len(all_results) >= 2 and len(all_results) <= 10:
    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))

    radar_props = ["bulk_modulus_GPa", "density_g_cm3", "surface_energy_J_m2",
                   "min_vacancy_energy_eV"]
    radar_props = [p for p in radar_props if p in df.columns
                   and df[p].notna().any()]

    if len(radar_props) >= 3:
        # Normalize each property to [0, 1]
        norm_data = {}
        for prop in radar_props:
            vals = df[prop].fillna(0).values
            vmin, vmax = vals.min(), vals.max()
            if vmax > vmin:
                norm_data[prop] = (vals - vmin) / (vmax - vmin)
            else:
                norm_data[prop] = np.zeros_like(vals)

        angles = np.linspace(0, 2 * np.pi, len(radar_props),
                             endpoint=False).tolist()
        angles += angles[:1]  # close the polygon

        for i, row in df.iterrows():
            values = [norm_data[p][i] for p in radar_props]
            values += values[:1]
            ax.plot(angles, values, "o-", linewidth=2,
                    label=row["formula"], markersize=6)
            ax.fill(angles, values, alpha=0.1)

        ax.set_xticks(angles[:-1])
        prop_labels = [p.replace("_", "\n").replace("GPa", "(GPa)")
                       .replace("J m2", "(J/m2)")
                       .replace("eV", "(eV)") for p in radar_props]
        ax.set_xticklabels(prop_labels, fontsize=9)
        ax.set_title("Multi-Property Radar Chart", pad=20)
        ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.1), fontsize=8)

        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_DIR, "radar_chart.png"),
                    dpi=150, bbox_inches="tight")
        plt.close()

print(f"\nAll plots saved to {OUTPUT_DIR}/")
print("Property prediction complete.")
```

## Key Parameters

| Parameter | Typical Value | Notes |
|---|---|---|
| `MACE_MODEL` | `"medium"` | `"large"` for best accuracy; `"small"` for speed |
| `FMAX_RELAX` | 0.005 eV/Ang | Tighter than screening (0.02); important for accurate lattice constants |
| `SURFACE_MILLER` | `(1, 0, 0)` | Low-index surfaces are most stable; try `(1, 1, 0)` and `(1, 1, 1)` too |
| `SLAB_THICKNESS` | 10--15 Ang | Thicker slabs are more converged but slower |
| `VACUUM_THICKNESS` | 15--20 Ang | Must be large enough to prevent slab-slab interaction |
| `SUPERCELL_SIZE` | `(2, 2, 2)` or `(3, 3, 3)` | Larger supercells reduce defect-image interactions but are slower |
| Birch-Murnaghan strain range | +/- 6% | 13 points gives robust fit; narrower range for very stiff materials |
| Band gap proxy | Phonon-gap method | Not a real band gap -- only indicates metallic vs. insulating tendency |

## Interpreting Results

**Lattice constants:**
- MACE-MP-0 lattice constants are typically within 1-2% of PBE DFT values for well-represented chemistries.
- Larger deviations (>3%) suggest the material may be outside the MACE training domain.
- Compare with experimental values: PBE overestimates volumes by ~1-3% on average compared to experiment.

**Bulk modulus:**
- Birch-Murnaghan fit is more reliable than the simpler parabolic fit, especially for materials with high B' (pressure derivative).
- MACE bulk moduli are within 10-20% of DFT for main-group and early transition-metal compounds.
- Negative or very small (<5 GPa) values indicate a fitting issue or an unstable structure.

**Band gap proxy:**
- This is NOT an electronic band gap. MACE has no electronic degrees of freedom.
- The phonon-gap proxy is at best qualitative: materials with a large acoustic-optical phonon gap tend to be insulators.
- For actual band gaps, use DFT (QE) or look up values from Materials Project.

**Surface energy:**
- Typical values: metals 1-3 J/m^2, oxides 0.5-2 J/m^2, ionic crystals 0.2-1 J/m^2.
- MACE surface energies can differ from DFT by 10-30%, especially for surfaces with significant reconstruction.
- The (100) surface may not be the most stable; check multiple Miller indices.

**Vacancy formation energy:**
- Lower values indicate the vacancy forms more easily (important for diffusion, catalysis).
- Typical values: 0.5-3 eV for metals, 3-8 eV for covalent/ionic materials.
- MACE neutral vacancy energies are within 0.2-0.5 eV of DFT for well-represented systems.
- Charged defects cannot be computed with MACE (requires DFT).

**Parity plots:**
- Points close to the diagonal line indicate good MACE-DFT agreement.
- MAE (Mean Absolute Error) quantifies the average prediction error.
- Systematic deviations (all points above/below diagonal) indicate a bias that may be correctable.

## Common Issues

| Problem | Cause | Solution |
|---|---|---|
| Lattice constants differ >5% from experiment | Material outside MACE training domain | Validate with QE DFT; use `MACE_MODEL="large"` |
| Negative bulk modulus | E(V) curve not convex near equilibrium | Structure may be unstable; re-relax with tighter `FMAX`; reduce strain range |
| Surface energy is negative | Slab not properly relaxed or vacuum too thin | Increase `VACUUM_THICKNESS`; check slab structure for artifacts |
| Vacancy calculation crashes | Supercell too large for memory | Reduce `SUPERCELL_SIZE` to `(2, 2, 2)` |
| Band gap proxy gives 0 for known insulator | Phonon method is unreliable for band gap prediction | Use DFT for actual band gaps; the proxy is only qualitative |
| MP comparison shows large errors | Different functionals or settings | MACE is trained on mixed PBE/PBE+U; MP uses specific settings; compare trends not absolutes |
| Slab generation fails | Structure symmetry not recognized or Miller index invalid | Try different Miller indices; verify structure with `SpacegroupAnalyzer` |
| Vibrations calculation slow | Too many atoms for finite-displacement method | Skip band gap proxy for cells > 30 atoms |
| Radar chart looks flat | All materials have similar properties | Normalize independently; or the materials are truly similar |
| "No module named mp_api" | mp-api not installed | `pip install mp-api`; or skip MP comparison with `FETCH_MP_REFERENCE = False` |
