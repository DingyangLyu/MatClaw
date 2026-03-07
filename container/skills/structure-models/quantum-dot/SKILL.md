# Quantum Dot Builder

## When to Use

- You need to build a quantum dot (nanoparticle) model by carving a sphere from bulk crystal.
- You are studying 0D quantum confinement effects in semiconductor nanocrystals.
- You need catalytic nanoparticle models for surface reactivity studies.
- You want to construct core-shell quantum dot heterostructures.
- You need Wulff-shape nanoparticles that respect facet surface energies.
- You need passivated QD models with realistic surface termination.

## Method Selection

| Criterion | ASE + pymatgen | VASP (VASPKIT 806) |
|---|---|---|
| Spherical QD from any bulk | Full control, any radius | Interactive via VASPKIT menu |
| Wulff-shape nanoparticles | `WulffShape` from pymatgen | Not available |
| Surface passivation with H | Scriptable (detect dangling bonds) | Manual post-processing |
| Core-shell QDs | Build core, add shell layer | Not available |
| Pre-relaxation screening | MACE (fast, no DFT license) | Requires VASP license |
| Production DFT on QD | Generate POSCAR for VASP | Native VASP workflow |

## Prerequisites

- ASE, pymatgen, numpy, matplotlib, scipy (pre-installed)
- MACE (pre-installed, for fast relaxation)
- Optional: VASP + VASPKIT for production DFT calculations

---

## Detailed Steps

### Method A: ASE + pymatgen (Spherical and Wulff-Shape QDs)

```python
#!/usr/bin/env python3
"""
Build quantum dot (nanoparticle) models from bulk crystals.
Covers spherical QDs, Wulff-shape QDs, passivation, core-shell,
MACE relaxation, and surface-to-volume analysis.
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from collections import Counter
from pathlib import Path

from pymatgen.core import Structure, Lattice, Element
from pymatgen.analysis.wulff import WulffShape
from pymatgen.io.vasp import Poscar
from pymatgen.io.ase import AseAtomsAdaptor
from ase.io import write as ase_write
from ase.data import covalent_radii
from scipy.spatial import ConvexHull

adaptor = AseAtomsAdaptor()

# ============================================================
# 1. Load bulk structures
# ============================================================
print("=== Step 1: Load Bulk Structures ===")

bulk_cdse = Structure.from_spacegroup(
    "F-43m", lattice=Lattice.cubic(6.077),
    species=["Cd", "Se"],
    coords=[[0.0, 0.0, 0.0], [0.25, 0.25, 0.25]],
)

bulk_si = Structure.from_spacegroup(
    "Fd-3m", lattice=Lattice.cubic(5.431),
    species=["Si"], coords=[[0.0, 0.0, 0.0]],
)

for name, s in [("CdSe", bulk_cdse), ("Si", bulk_si)]:
    print(f"  {name}: {s.formula}, a = {s.lattice.a:.3f} A")

# ============================================================
# 2. Build supercell large enough to carve from
# ============================================================
def required_supercell_size(lattice_param, radius):
    """Min supercell repetition to contain a sphere of given radius."""
    return max(int(np.ceil((2 * radius + 2.0) / lattice_param)), 2)

# ============================================================
# 3. Carve spherical QD
# ============================================================
print("\n=== Step 3: Carve Spherical QD ===")

def carve_spherical_qd(structure, radius, vacuum=15.0):
    """Carve a spherical QD from the center of a supercell."""
    cart = structure.cart_coords
    center = np.mean(cart, axis=0)
    distances = np.linalg.norm(cart - center, axis=1)
    keep_mask = distances <= radius
    kept_species = [structure.species[i] for i in range(len(structure))
                    if keep_mask[i]]
    kept_coords = cart[keep_mask]
    if len(kept_species) == 0:
        raise ValueError(f"No atoms within radius {radius} A.")
    kept_coords -= np.mean(kept_coords, axis=0)
    box_size = 2 * radius + 2 * vacuum
    kept_coords += box_size / 2
    return Structure(Lattice(np.eye(3) * box_size), kept_species,
                     kept_coords, coords_are_cartesian=True)

# Build QDs at several radii
for r in [5.0, 8.0, 12.0]:
    n = required_supercell_size(bulk_cdse.lattice.a, r)
    sc = bulk_cdse.copy()
    sc.make_supercell([n, n, n])
    qd = carve_spherical_qd(sc, radius=r)
    n_cd = sum(1 for s in qd.species if s.symbol == "Cd")
    n_se = sum(1 for s in qd.species if s.symbol == "Se")
    print(f"  CdSe r={r:.0f} A: {len(qd)} atoms (Cd={n_cd}, Se={n_se})")
    qd.to(f"CdSe_QD_r{r:.0f}.cif")

n_si = required_supercell_size(bulk_si.lattice.a, 12.0)
sc_si = bulk_si.copy()
sc_si.make_supercell([n_si, n_si, n_si])
qd_si = carve_spherical_qd(sc_si, radius=12.0)
print(f"  Si r=12 A: {len(qd_si)} atoms")
qd_si.to("Si_QD_r12.cif")
Poscar(qd_si).write_file("POSCAR_Si_QD_r12")

# ============================================================
# 4. Fix stoichiometry for compound QDs
# ============================================================
print("\n=== Step 4: Stoichiometry Fix ===")

def fix_stoichiometry(structure, target_ratio=None):
    """Remove outermost excess atoms to restore target cation/anion ratio."""
    counts = Counter(str(s) for s in structure.species)
    elements = sorted(counts.keys())
    print(f"    Before: {dict(counts)}")
    if len(elements) != 2:
        return structure
    if target_ratio is None:
        target_ratio = {elements[0]: 1, elements[1]: 1}
    e1, e2 = elements
    target_n = min(counts[e1] // target_ratio[e1],
                   counts[e2] // target_ratio[e2])
    cart = structure.cart_coords
    center = np.mean(cart, axis=0)
    dists = np.linalg.norm(cart - center, axis=1)
    sites_to_remove = []
    for elem, ratio in [(e1, target_ratio[e1]), (e2, target_ratio[e2])]:
        excess = counts[elem] - target_n * ratio
        if excess <= 0:
            continue
        idx = [i for i, s in enumerate(structure.species) if str(s) == elem]
        order = np.argsort(dists[idx])[::-1]
        sites_to_remove.extend(idx[order[j]] for j in range(excess))
    fixed = structure.copy()
    fixed.remove_sites(sorted(sites_to_remove, reverse=True))
    print(f"    After:  {dict(Counter(str(s) for s in fixed.species))}")
    return fixed

n = required_supercell_size(bulk_cdse.lattice.a, 10.0)
sc = bulk_cdse.copy()
sc.make_supercell([n, n, n])
qd_fixed = fix_stoichiometry(carve_spherical_qd(sc, 10.0))
qd_fixed.to("CdSe_QD_r10_fixed.cif")

# ============================================================
# 5. Wulff-shape quantum dot
# ============================================================
print("\n=== Step 5: Wulff-Shape QD ===")

def carve_wulff_qd(structure, surface_energies, total_atoms_target,
                   vacuum=15.0):
    """Carve a Wulff-shape QD using surface energy anisotropy."""
    lattice = structure.lattice
    millers = list(surface_energies.keys())
    energies = [surface_energies[m] for m in millers]
    wulff = WulffShape(lattice, millers, energies)
    print(f"    Anisotropy: {wulff.anisotropy:.3f}")

    cart = structure.cart_coords
    center = np.mean(cart, axis=0)
    recip = lattice.reciprocal_lattice_crystallographic

    # Build facet normals with cubic symmetry equivalents
    all_normals, all_dists = [], []
    min_e = min(energies)
    for miller, energy in zip(millers, energies):
        normal = sum(miller[i] * np.array(recip.matrix[i]) for i in range(3))
        normal /= np.linalg.norm(normal)
        dist = energy / min_e
        perms = set()
        for signs in [(s0,s1,s2) for s0 in [1,-1] for s1 in [1,-1]
                      for s2 in [1,-1]]:
            for p in [(0,1,2),(0,2,1),(1,0,2),(1,2,0),(2,0,1),(2,1,0)]:
                v = np.array([signs[i]*normal[p[i]] for i in range(3)])
                v /= np.linalg.norm(v)
                key = tuple(np.round(v, 8))
                if key not in perms:
                    perms.add(key)
                    all_normals.append(v)
                    all_dists.append(dist)

    # Binary search for scale to match target atom count
    lo, hi = 1.0, 50.0
    for _ in range(30):
        scale = (lo + hi) / 2
        inside = np.ones(len(cart), dtype=bool)
        for n_vec, d in zip(all_normals, all_dists):
            inside &= ((cart - center) @ n_vec <= scale * d)
        if np.sum(inside) < total_atoms_target:
            lo = scale
        else:
            hi = scale
    inside = np.ones(len(cart), dtype=bool)
    for n_vec, d in zip(all_normals, all_dists):
        inside &= ((cart - center) @ n_vec <= scale * d)

    kept = cart[inside]
    kept -= np.mean(kept, axis=0)
    box = np.max(np.abs(kept)) * 2 + 2 * vacuum
    kept += box / 2
    species = [structure.species[i] for i in range(len(structure)) if inside[i]]
    qd = Structure(Lattice(np.eye(3)*box), species, kept,
                   coords_are_cartesian=True)
    print(f"    Wulff QD: {len(qd)} atoms (target: {total_atoms_target})")
    return qd

n_w = required_supercell_size(bulk_si.lattice.a, 15.0)
sc_w = bulk_si.copy()
sc_w.make_supercell([n_w, n_w, n_w])
wulff_qd = carve_wulff_qd(sc_w,
    {(1,0,0): 2.13, (1,1,0): 1.70, (1,1,1): 1.36}, 200)
wulff_qd.to("Si_QD_wulff.cif")

# ============================================================
# 6. Passivate dangling bonds with hydrogen
# ============================================================
print("\n=== Step 6: Passivation with H ===")

def passivate_with_hydrogen(structure, cutoff_scale=1.3, h_dist=1.5):
    """Add H atoms to under-coordinated surface atoms."""
    cart = structure.cart_coords
    species = [str(s) for s in structure.species]
    n = len(structure)
    cov = {s: covalent_radii[Element(s).Z] for s in set(species)}

    neighbors = [[] for _ in range(n)]
    for i in range(n):
        for j in range(i+1, n):
            if np.linalg.norm(cart[i]-cart[j]) <= \
               (cov[species[i]]+cov[species[j]])*cutoff_scale:
                neighbors[i].append(j)
                neighbors[j].append(i)

    coord = [len(nb) for nb in neighbors]
    bulk_coord = {}
    for s in set(species):
        vals = [coord[i] for i in range(n) if species[i] == s]
        bulk_coord[s] = max(vals) if vals else 4

    center = np.mean(cart, axis=0)
    h_pos = []
    for i in range(n):
        missing = bulk_coord[species[i]] - coord[i]
        if missing <= 0:
            continue
        outward = cart[i] - center
        outward /= (np.linalg.norm(outward) + 1e-10)
        if neighbors[i]:
            dirs = [(cart[j]-cart[i])/np.linalg.norm(cart[j]-cart[i])
                    for j in neighbors[i]]
            avg = np.mean(dirs, axis=0)
            if np.linalg.norm(avg) > 0.1:
                outward = -avg / np.linalg.norm(avg)
        h_pos.append(cart[i] + outward * h_dist)

    print(f"    Adding {len(h_pos)} H atoms")
    all_sp = list(structure.species) + [Element("H")] * len(h_pos)
    all_co = np.vstack([cart, h_pos]) if h_pos else cart
    return Structure(structure.lattice, all_sp, all_co,
                     coords_are_cartesian=True)

qd_small = carve_spherical_qd(sc_si, radius=8.0)
qd_pass = passivate_with_hydrogen(qd_small)
print(f"    {len(qd_small)} -> {len(qd_pass)} atoms")
qd_pass.to("Si_QD_r8_passivated.cif")

# ============================================================
# 7. Vacuum box adjustment
# ============================================================
def adjust_vacuum(structure, vacuum=15.0):
    """Re-center QD and resize vacuum box."""
    cart = structure.cart_coords
    mins, maxs = cart.min(axis=0), cart.max(axis=0)
    box = maxs - mins + 2 * vacuum
    coords = cart - mins + vacuum
    return Structure(Lattice(np.diag(box)), structure.species, coords,
                     coords_are_cartesian=True)

qd_adj = adjust_vacuum(qd_pass, vacuum=20.0)
print(f"\n  Box: {qd_adj.lattice.a:.1f} x {qd_adj.lattice.b:.1f} x "
      f"{qd_adj.lattice.c:.1f} A")
qd_adj.to("Si_QD_r8_vac20.cif")

# ============================================================
# 8. Relax with MACE
# ============================================================
print("\n=== Step 8: MACE Relaxation ===")
import warnings; warnings.filterwarnings("ignore")
from mace.calculators import mace_mp
from ase.optimize import BFGS

qd_relax = carve_spherical_qd(sc_si, radius=6.0)
atoms = adaptor.get_atoms(qd_relax)
atoms.calc = mace_mp(model="medium", dispersion=False,
                     default_dtype="float64")
e0 = atoms.get_potential_energy()
opt = BFGS(atoms, logfile="qd_relax.log", trajectory="qd_relax.traj")
opt.run(fmax=0.05, steps=200)
e1 = atoms.get_potential_energy()
print(f"  E: {e0:.4f} -> {e1:.4f} eV ({len(atoms)} atoms)")
ase_write("Si_QD_r6_relaxed.cif", atoms)

# ============================================================
# 9. Surface-to-volume ratio analysis
# ============================================================
print("\n=== Step 9: Surface Analysis ===")

def analyze_qd(structure, cutoff_scale=1.3):
    """Compute surface/volume ratio and sphericity."""
    cart = structure.cart_coords
    species = [str(s) for s in structure.species]
    n = len(structure)
    cov = {s: covalent_radii[Element(s).Z] for s in set(species)}
    coord = np.zeros(n, dtype=int)
    for i in range(n):
        for j in range(i+1, n):
            if np.linalg.norm(cart[i]-cart[j]) <= \
               (cov[species[i]]+cov[species[j]])*cutoff_scale:
                coord[i] += 1; coord[j] += 1
    n_surf = np.sum(coord < max(coord)) if n > 0 else 0
    hull = ConvexHull(cart) if n >= 4 else None
    vol = hull.volume if hull else 0
    area = hull.area if hull else 0
    return {"n_atoms": n, "n_surface": n_surf,
            "surface_fraction": n_surf/n if n else 0,
            "sv_ratio": area/vol if vol else 0,
            "sphericity": (np.pi**(1/3)*(6*vol)**(2/3)/area)
                          if area else 0}

print(f"{'r':>5} {'N':>6} {'Surf%':>7} {'S/V':>8} {'Spher':>7}")
for r in [5.0, 8.0, 10.0, 12.0]:
    nr = required_supercell_size(bulk_si.lattice.a, r)
    sc = bulk_si.copy(); sc.make_supercell([nr,nr,nr])
    g = analyze_qd(carve_spherical_qd(sc, r))
    print(f"{r:5.0f} {g['n_atoms']:6d} {g['surface_fraction']*100:6.1f}% "
          f"{g['sv_ratio']:8.4f} {g['sphericity']:7.4f}")

# ============================================================
# 10. Core-shell quantum dot
# ============================================================
print("\n=== Step 10: Core-Shell QD ===")

def build_core_shell_qd(core_bulk, shell_bulk, core_r, shell_t, vac=15.0):
    """Build core-shell QD: core from one material, shell from another."""
    outer_r = core_r + shell_t
    # Core
    nc = required_supercell_size(core_bulk.lattice.a, core_r)
    sc_c = core_bulk.copy(); sc_c.make_supercell([nc,nc,nc])
    cc = sc_c.cart_coords; ctr = np.mean(cc, axis=0)
    mask_c = np.linalg.norm(cc - ctr, axis=1) <= core_r
    core_sp = [sc_c.species[i] for i in range(len(sc_c)) if mask_c[i]]
    core_co = cc[mask_c]
    # Shell
    ns = required_supercell_size(shell_bulk.lattice.a, outer_r)
    sc_s = shell_bulk.copy(); sc_s.make_supercell([ns,ns,ns])
    cs = sc_s.cart_coords; ctr_s = np.mean(cs, axis=0)
    ds = np.linalg.norm(cs - ctr_s, axis=1)
    mask_s = (ds > core_r) & (ds <= outer_r)
    shell_sp = [sc_s.species[i] for i in range(len(sc_s)) if mask_s[i]]
    shell_co = cs[mask_s]
    shell_co -= np.mean(shell_co, axis=0) if len(shell_co) else ctr_s
    shell_co += np.mean(core_co, axis=0) if len(core_co) else ctr
    # Combine
    all_co = np.vstack([core_co, shell_co])
    all_co -= np.mean(all_co, axis=0)
    box = 2*outer_r + 2*vac
    all_co += box/2
    return Structure(Lattice(np.eye(3)*box), core_sp+shell_sp,
                     all_co, coords_are_cartesian=True)

bulk_zns = Structure.from_spacegroup(
    "F-43m", lattice=Lattice.cubic(5.409),
    species=["Zn","S"], coords=[[0,0,0],[0.25,0.25,0.25]])
cs_qd = build_core_shell_qd(bulk_cdse, bulk_zns, 8.0, 4.0)
cs_counts = Counter(str(s) for s in cs_qd.species)
print(f"  CdSe/ZnS: {len(cs_qd)} atoms, {dict(cs_counts)}")
cs_qd.to("CdSe_ZnS_core_shell.cif")

# ============================================================
# 11. Visualization
# ============================================================
print("\n=== Step 11: Visualization ===")
fig, axes = plt.subplots(1, 3, figsize=(18, 6))

# Spherical QD cross-section
qd_v = carve_spherical_qd(sc_si, 10.0)
c = qd_v.cart_coords; ctr = np.mean(c, axis=0)
sl = np.abs(c[:,2]-ctr[2]) < 2.0
axes[0].scatter(c[sl,0], c[sl,1], s=20, c="steelblue")
axes[0].add_patch(plt.Circle((ctr[0],ctr[1]), 10, fill=False,
                              ls="--", color="red"))
axes[0].set_aspect("equal"); axes[0].set_title("Si QD r=10 A (z-slice)")
axes[0].set_xlabel("x (A)"); axes[0].set_ylabel("y (A)")

# Wulff QD cross-section
cw = wulff_qd.cart_coords; ctr_w = np.mean(cw, axis=0)
sl_w = np.abs(cw[:,2]-ctr_w[2]) < 2.0
axes[1].scatter(cw[sl_w,0], cw[sl_w,1], s=20, c="darkorange")
axes[1].set_aspect("equal"); axes[1].set_title("Si Wulff QD (z-slice)")
axes[1].set_xlabel("x (A)"); axes[1].set_ylabel("y (A)")

# Surface fraction vs radius
radii = np.linspace(4, 20, 15)
sf = []
for r in radii:
    nr = required_supercell_size(bulk_si.lattice.a, r)
    sc = bulk_si.copy(); sc.make_supercell([nr,nr,nr])
    sf.append(analyze_qd(carve_spherical_qd(sc, r))["surface_fraction"]*100)
axes[2].plot(radii, sf, "o-", color="teal")
axes[2].set_xlabel("QD Radius (A)")
axes[2].set_ylabel("Surface Fraction (%)")
axes[2].set_title("Size-Dependent Surface Fraction")

for ax in axes: ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("quantum_dot_analysis.png", dpi=150)
print("  Saved: quantum_dot_analysis.png")

print("\n=== All QD files written ===")
for f in sorted(Path(".").glob("*QD*")) + sorted(Path(".").glob("*core*")):
    print(f"  {f}")
```

### Method B: VASP (VASPKIT Task 806)

```python
#!/usr/bin/env python3
"""
Generate VASP input files for quantum dot (cluster) calculations.
Equivalent to VASPKIT Task 806: Build Quantum-Dot by Specified Radius.
"""

import numpy as np
from pymatgen.core import Structure, Lattice
from pymatgen.io.vasp import Poscar, Incar, Kpoints

# ============================================================
# 1. Build QD POSCAR (equivalent to VASPKIT 806)
# ============================================================
print("=== VASP QD Setup (VASPKIT 806 equivalent) ===")

bulk = Structure.from_spacegroup(
    "Fd-3m", lattice=Lattice.cubic(5.431),
    species=["Si"], coords=[[0.0, 0.0, 0.0]])

qd_radius = 10.0  # Angstrom
vacuum = 15.0

n_rep = max(int(np.ceil((2*qd_radius + 2.0) / bulk.lattice.a)), 2)
sc = bulk.copy()
sc.make_supercell([n_rep, n_rep, n_rep])

cart = sc.cart_coords
center = np.mean(cart, axis=0)
keep = np.linalg.norm(cart - center, axis=1) <= qd_radius
sites_to_remove = [i for i, k in enumerate(keep) if not k]
qd = sc.copy()
qd.remove_sites(sites_to_remove)

box = 2*qd_radius + 2*vacuum
coords = qd.cart_coords - np.mean(qd.cart_coords, axis=0) + box/2
qd_vasp = Structure(Lattice(np.eye(3)*box), qd.species, coords,
                    coords_are_cartesian=True)
Poscar(qd_vasp).write_file("POSCAR_QD")
print(f"  POSCAR_QD: {len(qd_vasp)} atoms, box = {box:.1f} A")

# ============================================================
# 2. INCAR for cluster calculation
# ============================================================
incar = Incar({
    "SYSTEM": f"Si QD r={qd_radius} A",
    "ENCUT": 400, "PREC": "Accurate",
    "EDIFF": 1e-5, "NELM": 200,
    "IBRION": 2, "NSW": 200, "EDIFFG": -0.02,
    "ISIF": 2,          # Relax ions only
    "ISMEAR": 0, "SIGMA": 0.05,
    "LDIPOL": True, "IDIPOL": 4, "LMONO": True,
    "ALGO": "Normal", "LREAL": "Auto", "NCORE": 4,
    "LWAVE": False, "LCHARG": False, "LORBIT": 11,
})
incar.write_file("INCAR_QD")
print("  INCAR_QD: cluster settings (ISIF=2, IDIPOL=4)")

# ============================================================
# 3. KPOINTS: Gamma-only for cluster
# ============================================================
Kpoints.gamma_automatic(kpts=(1, 1, 1)).write_file("KPOINTS_QD")
print("  KPOINTS_QD: Gamma-only (1x1x1)")

# ============================================================
# 4. VASPKIT 806 interactive workflow reference
# ============================================================
print("\n=== VASPKIT 806 Workflow ===")
print("  1. Prepare bulk POSCAR")
print("  2. Run vaspkit -> 8 (Structure Model) -> 806")
print("  3. Enter QD radius in Angstrom")
print("  4. VASPKIT generates POSCAR_QD with vacuum box")
print("  Key: ISIF=2, Gamma-only, IDIPOL=4, LREAL=Auto, NELM>=200")
```

---

## Key Parameters

| Parameter | Typical Value | Notes |
|---|---|---|
| QD radius | 5--30 A | 5 A ~ 50-100 atoms, 15 A ~ 1000+ atoms |
| Vacuum padding | 15--20 A | 20 A for charged QDs |
| Bond cutoff scale | 1.2--1.4 | scale * sum(covalent radii) for neighbor detection |
| H passivation length | 1.0--1.5 A | Si-H ~ 1.48 A, Cd-H ~ 1.70 A |
| MACE fmax | 0.01--0.05 eV/A | 0.05 for screening, 0.01 for production |
| Surface energy (Wulff) | 0.5--3.0 J/m^2 | From DFT slab calculations |
| Core-shell thickness | 2--10 A | 1--3 monolayers typical |
| VASP ENCUT | 400--520 eV | Same as bulk; convergence test recommended |
| VASP ISIF | 2 | Relax ions only for cluster |
| KPOINTS | 1x1x1 Gamma | No dispersion in 0D system |

## Interpreting Results

| Observable | What to Check | Typical Behavior |
|---|---|---|
| Band gap vs size | HOMO-LUMO gap vs bulk gap | Gap increases as QD shrinks (quantum confinement) |
| Surface fraction | N_surface / N_total | Increases sharply below r ~ 10 A |
| Relaxation energy | E_relaxed - E_unrelaxed | Surface reconstruction lowers energy |
| Stoichiometry | Cation/anion ratio | May deviate from bulk after carving; fix before DFT |
| Sphericity | 1.0 = perfect sphere | Wulff shapes: 0.8--0.95 |
| Core-shell interface | Atomic displacements | Lattice mismatch causes strain |
| VASP convergence | NELM steps | Clusters converge slower; use NELM >= 200 |

## Common Issues

| Problem | Solution |
|---|---|
| Broken stoichiometry after carving | Use `fix_stoichiometry()` to remove excess surface atoms. |
| Too few atoms for desired radius | Increase supercell. Use `required_supercell_size()`. |
| Dangling bonds create mid-gap states | Passivate with H via `passivate_with_hydrogen()`. |
| MACE gives unrealistic QD relaxation | QD surfaces outside training data. Validate with DFT. |
| VASP SCF fails to converge | NELM=200+, SIGMA=0.02, try ALGO=All or ALGO=Damped. |
| Periodic images interact | Vacuum >= 20 A. Enable IDIPOL=4, LDIPOL=.TRUE. |
| Wulff atom count mismatch | Adjust `total_atoms_target`; binary search converges to approximate count. |
| Core-shell overlap at interface | Remove atoms with d < 1.5 A across core/shell boundary. |
| Charged QD in VASP | Set NELECT explicitly. Use LMONO=.TRUE. |
| QD too large for DFT | MACE relaxation + single-point DFT, or linear-scaling codes. |
