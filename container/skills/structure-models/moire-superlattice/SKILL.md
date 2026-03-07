# Moire Superlattice Builder

## When to Use

- You need to build a twisted bilayer graphene (TBG) model at a specified twist angle.
- You are studying TMD (transition metal dichalcogenide) heterostructures with moire patterns.
- You want to find flat bands in magic-angle twisted bilayer systems.
- You need commensurate moire superlattice cells for periodic DFT calculations.
- You are constructing twisted homobilayer or heterobilayer structures.
- You need to relate twist angle to supercell size (number of atoms).

## Method Selection

| Criterion | pymatgen + ASE | VASP (VASPKIT 825) |
|---|---|---|
| Hexagonal moire from twist angle | Full control via (m,n) notation | Interactive menu |
| Commensurate cell search | Scriptable, arbitrary angles | Built-in for hexagonal lattices |
| Heterobilayer (different materials) | Flexible stacking | Not specialized |
| Interlayer distance control | Precise scripting | Manual POSCAR editing |
| Atom count vs angle analysis | Built-in with script | Manual |
| Pre-relaxation screening | MACE (fast) | Requires VASP license |
| Production DFT with vdW | Generate POSCAR for VASP | Native VASP workflow |

## Prerequisites

- ASE, pymatgen, numpy, matplotlib (pre-installed)
- MACE (pre-installed, for fast relaxation)
- scipy (pre-installed, for spatial analysis)
- Optional: VASP + VASPKIT for production DFT calculations

---

## Detailed Steps

### Method A: pymatgen + ASE (Commensurate Moire Superlattice)

```python
#!/usr/bin/env python3
"""
Build commensurate moire superlattices from twist angles.

Uses (m, n) notation: cos(theta) = (m^2+4mn+n^2) / (2*(m^2+mn+n^2))
Superlattice vectors: L1 = m*a1 + n*a2, L2 = -n*a1 + (m+n)*a2
Atoms per layer = atoms_per_cell * (m^2 + mn + n^2)

Covers: commensurate angle search, TBG construction, TMD heterobilayer,
magic angle (1.1 deg), atom count analysis, MACE relaxation.
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

from pymatgen.core import Structure, Lattice
from pymatgen.io.vasp import Poscar
from pymatgen.io.ase import AseAtomsAdaptor
from ase.io import write as ase_write

adaptor = AseAtomsAdaptor()

# ============================================================
# 1. Commensurate moire angles from (m, n) notation
# ============================================================
print("=== Step 1: Commensurate Moire Angles ===")

def moire_angle(m, n):
    """Twist angle (degrees) for commensurate indices (m, n)."""
    cos_t = (m**2 + 4*m*n + n**2) / (2*(m**2 + m*n + n**2))
    return np.degrees(np.arccos(np.clip(cos_t, -1.0, 1.0)))

def moire_natoms(m, n, atoms_per_cell=2):
    """Atoms per layer in the commensurate moire cell."""
    return atoms_per_cell * (m**2 + m*n + n**2)

# Table of commensurate angles
pairs = []
for m in range(1, 35):
    for n in range(0, m):
        if np.gcd(m, n) != 1:
            continue
        theta = moire_angle(m, n)
        if 0.5 < theta < 30.0:
            pairs.append((m, n, theta, moire_natoms(m, n)))
pairs.sort(key=lambda x: x[2])

print(f"{'(m,n)':>8} {'Angle':>10} {'Atoms/layer':>12} {'Total':>8}")
for m, n, theta, nl in pairs[:15]:
    print(f"  ({m},{n})  {theta:9.4f}  {nl:11d}  {2*nl:7d}")
print(f"  ... {len(pairs)} commensurate angles in 0.5-30 deg")

# ============================================================
# 2. Build twisted bilayer graphene
# ============================================================
print("\n=== Step 2: Twisted Bilayer Graphene ===")

def build_twisted_bilayer(m, n, a_cc=1.42, d_layer=3.35, stacking="AB"):
    """
    Build commensurate TBG. Bottom layer unrotated, top rotated by theta.
    Returns (Structure, angle_deg).
    """
    a = a_cc * np.sqrt(3)
    theta = moire_angle(m, n)
    theta_rad = np.radians(theta)

    a1 = np.array([a, 0, 0])
    a2 = np.array([a*np.cos(np.radians(60)), a*np.sin(np.radians(60)), 0])

    L1 = m*a1 + n*a2
    L2 = -n*a1 + (m+n)*a2
    L3 = np.array([0, 0, d_layer + 20.0])
    print(f"    Moire period: {np.linalg.norm(L1):.2f} A")

    b1 = (1/3)*a1 + (2/3)*a2
    b2 = (2/3)*a1 + (1/3)*a2

    R = np.array([[np.cos(theta_rad), -np.sin(theta_rad), 0],
                  [np.sin(theta_rad),  np.cos(theta_rad), 0],
                  [0, 0, 1]])

    a1r, a2r = R @ a1, R @ a2
    shift = (a1 + a2)/3 if stacking == "AB" else np.zeros(3)
    b1r, b2r = R @ (b1 + shift), R @ (b2 + shift)

    inv_L = np.linalg.inv(np.array([L1[:2], L2[:2]]).T)
    L1t, L2t = R @ L1, R @ L2
    inv_Lt = np.linalg.inv(np.array([L1t[:2], L2t[:2]]).T)

    max_pq = m + n + 2
    bottom, top = [], []

    for p in range(-max_pq, max_pq+1):
        for q in range(-max_pq, max_pq+1):
            # Bottom layer
            for basis in [b1, b2]:
                pos = p*a1 + q*a2 + basis
                f = inv_L @ pos[:2]
                if -1e-6 <= f[0] < 1-1e-6 and -1e-6 <= f[1] < 1-1e-6:
                    pos[2] = 10.0
                    bottom.append(pos.copy())
            # Top layer
            for basis in [b1r, b2r]:
                pos = p*a1r + q*a2r + basis
                f = inv_Lt @ pos[:2]
                if -1e-6 <= f[0] < 1-1e-6 and -1e-6 <= f[1] < 1-1e-6:
                    pos[2] = 10.0 + d_layer
                    top.append(pos.copy())

    all_pos = np.array(bottom + top)
    all_sp = ["C"] * len(all_pos)
    print(f"    Bottom: {len(bottom)}, Top: {len(top)}, "
          f"Total: {len(all_pos)}")

    return Structure(Lattice(np.array([L1, L2, L3])),
                     all_sp, all_pos, coords_are_cartesian=True), theta

# Build TBG at several angles
for mv, nv in [(3, 2), (5, 4), (7, 6)]:
    theta = moire_angle(mv, nv)
    print(f"\n  ({mv},{nv}), theta = {theta:.4f} deg:")
    tbg, _ = build_twisted_bilayer(mv, nv)
    tbg.to(f"TBG_{mv}_{nv}_theta{theta:.2f}.cif")
    Poscar(tbg).write_file(f"POSCAR_TBG_{mv}_{nv}")

# ============================================================
# 3. Magic angle TBG (1.1 deg)
# ============================================================
print("\n=== Step 3: Magic Angle TBG ===")

def find_closest_commensurate(target, max_mn=50, apc=2):
    """Find (m,n) giving commensurate angle closest to target."""
    best, best_d = None, float("inf")
    for ms in range(1, max_mn+1):
        for ns in range(0, ms):
            if np.gcd(ms, ns) != 1:
                continue
            ang = moire_angle(ms, ns)
            if abs(ang - target) < best_d:
                best_d = abs(ang - target)
                best = (ms, ns, ang, moire_natoms(ms, ns, apc))
    return best

magic = find_closest_commensurate(1.1)
if magic:
    mm, nm, tm, nlm = magic
    print(f"  Closest to 1.1 deg: ({mm},{nm}), "
          f"angle={tm:.4f}, atoms={2*nlm}")
    if 2*nlm <= 15000:
        tbg_m, _ = build_twisted_bilayer(mm, nm)
        tbg_m.to("TBG_magic_angle.cif")
        Poscar(tbg_m).write_file("POSCAR_TBG_magic")
    else:
        print(f"  Too large ({2*nlm} atoms). Use tight-binding or MACE.")

# ============================================================
# 4. TMD heterobilayer moire (MoS2/WS2)
# ============================================================
print("\n=== Step 4: TMD Heterobilayer Moire ===")

def build_twisted_tmd(m, n, l1_params, l2_params, d_inter=6.15):
    """Build twisted TMD heterobilayer. Returns (Structure, angle)."""
    theta = moire_angle(m, n)
    theta_rad = np.radians(theta)
    a_avg = (l1_params["a"] + l2_params["a"]) / 2

    a1 = np.array([a_avg, 0, 0])
    a2 = np.array([a_avg*np.cos(np.radians(60)),
                    a_avg*np.sin(np.radians(60)), 0])

    L1 = m*a1 + n*a2
    L2 = -n*a1 + (m+n)*a2
    L3 = np.array([0, 0, d_inter + 25.0])
    print(f"    Moire period: {np.linalg.norm(L1):.2f} A")

    R = np.array([[np.cos(theta_rad), -np.sin(theta_rad), 0],
                  [np.sin(theta_rad),  np.cos(theta_rad), 0],
                  [0, 0, 1]])

    b_metal = (1/3)*a1 + (2/3)*a2
    a1r, a2r = R @ a1, R @ a2
    b_metal_r = R @ b_metal

    inv_L = np.linalg.inv(np.array([L1[:2], L2[:2]]).T)
    L1t, L2t = R @ L1, R @ L2
    inv_Lt = np.linalg.inv(np.array([L1t[:2], L2t[:2]]).T)

    max_pq = m + n + 2
    all_atoms, all_sp = [], []
    z1, z2 = 10.0, 10.0 + d_inter

    for p in range(-max_pq, max_pq+1):
        for q in range(-max_pq, max_pq+1):
            # Bottom layer (unrotated)
            pos = p*a1 + q*a2 + b_metal
            f = inv_L @ pos[:2]
            if -1e-6 <= f[0] < 1-1e-6 and -1e-6 <= f[1] < 1-1e-6:
                for z_off, sp in [(0, l1_params["metal"]),
                                   (l1_params["d"], l1_params["X"]),
                                   (-l1_params["d"], l1_params["X"])]:
                    all_atoms.append([pos[0], pos[1], z1+z_off])
                    all_sp.append(sp)

            # Top layer (rotated)
            pos2 = p*a1r + q*a2r + b_metal_r
            f2 = inv_Lt @ pos2[:2]
            if -1e-6 <= f2[0] < 1-1e-6 and -1e-6 <= f2[1] < 1-1e-6:
                for z_off, sp in [(0, l2_params["metal"]),
                                   (l2_params["d"], l2_params["X"]),
                                   (-l2_params["d"], l2_params["X"])]:
                    all_atoms.append([pos2[0], pos2[1], z2+z_off])
                    all_sp.append(sp)

    print(f"    Total atoms: {len(all_atoms)}")
    return Structure(Lattice(np.array([L1, L2, L3])),
                     all_sp, np.array(all_atoms),
                     coords_are_cartesian=True), theta

mos2 = {"metal": "Mo", "X": "S", "a": 3.16, "d": 1.58}
ws2  = {"metal": "W",  "X": "S", "a": 3.155, "d": 1.57}

for mv, nv in [(3, 2), (5, 4)]:
    theta = moire_angle(mv, nv)
    print(f"\n  MoS2/WS2 ({mv},{nv}), theta={theta:.4f} deg:")
    tmd, _ = build_twisted_tmd(mv, nv, mos2, ws2)
    tmd.to(f"MoS2_WS2_moire_{mv}_{nv}.cif")

# ============================================================
# 5. Approximate commensurate cells for arbitrary angles
# ============================================================
print("\n=== Step 5: Arbitrary Angle Search ===")

def find_approximate(target, tol=0.5, max_mn=40, max_atoms=5000, apc=2):
    """Find all (m,n) within tolerance of target angle, sorted by size."""
    results = []
    for ms in range(1, max_mn+1):
        for ns in range(0, ms):
            if np.gcd(ms, ns) != 1:
                continue
            ang = moire_angle(ms, ns)
            if abs(ang - target) <= tol:
                nl = moire_natoms(ms, ns, apc)
                if nl <= max_atoms:
                    results.append((ms, ns, ang, nl))
    results.sort(key=lambda x: x[3])
    return results

matches = find_approximate(5.0, tol=0.5)
print(f"  Near 5 deg (+/- 0.5): {len(matches)} cells found")
for m, n, t, nl in matches[:8]:
    print(f"    ({m},{n}) {t:8.4f} deg, {2*nl:6d} atoms")

# ============================================================
# 6. Atom count vs twist angle plot
# ============================================================
print("\n=== Step 6: Atom Count vs Angle ===")

fig, axes = plt.subplots(1, 2, figsize=(14, 6))

angles, natoms = [], []
for mv in range(1, 45):
    for nv in range(0, mv):
        if np.gcd(mv, nv) != 1:
            continue
        t = moire_angle(mv, nv)
        if 0.5 < t < 30:
            angles.append(t)
            natoms.append(2 * moire_natoms(mv, nv))

axes[0].scatter(angles, natoms, s=8, c="steelblue", alpha=0.6)
axes[0].set_xlabel("Twist Angle (deg)")
axes[0].set_ylabel("Total Atoms (bilayer)")
axes[0].set_yscale("log")
axes[0].axvline(1.1, color="red", ls="--", label="Magic angle")
axes[0].legend()
axes[0].set_title("Atoms vs Twist Angle")

a_gr = 1.42 * np.sqrt(3)
t_range = np.linspace(0.5, 30, 200)
lam = a_gr / (2 * np.sin(np.radians(t_range/2)))
axes[1].plot(t_range, lam, color="teal", lw=2)
axes[1].axvline(1.1, color="red", ls="--", label="Magic angle")
axes[1].set_xlabel("Twist Angle (deg)")
axes[1].set_ylabel("Moire Wavelength (A)")
axes[1].set_title("Moire Wavelength vs Angle")
axes[1].legend()

for ax in axes: ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("moire_angle_analysis.png", dpi=150)
print("  Saved: moire_angle_analysis.png")

# ============================================================
# 7. MACE relaxation of TBG
# ============================================================
print("\n=== Step 7: MACE Relaxation ===")

import warnings; warnings.filterwarnings("ignore")
from mace.calculators import mace_mp
from ase.optimize import BFGS
from ase.constraints import FixedLine

tbg_r, _ = build_twisted_bilayer(3, 2)
atoms = adaptor.get_atoms(tbg_r)
atoms.calc = mace_mp(model="medium", dispersion=True,
                     default_dtype="float64")

e0 = atoms.get_potential_energy()
# Constrain in-plane, allow z relaxation for interlayer adjustment
atoms.set_constraint([FixedLine(i, [0,0,1]) for i in range(len(atoms))])
opt = BFGS(atoms, logfile="tbg_relax.log", trajectory="tbg_relax.traj")
opt.run(fmax=0.02, steps=200)
e1 = atoms.get_potential_energy()

n_half = len(atoms) // 2
z = atoms.positions[:, 2]
d_layer = abs(np.mean(z[n_half:]) - np.mean(z[:n_half]))
print(f"  E: {e0:.4f} -> {e1:.4f} eV ({len(atoms)} atoms)")
print(f"  Interlayer distance: {d_layer:.3f} A")
ase_write("TBG_relaxed.cif", atoms)

# ============================================================
# 8. Moire pattern visualization
# ============================================================
print("\n=== Step 8: Visualization ===")
from scipy.spatial import cKDTree

fig, axes = plt.subplots(1, 2, figsize=(14, 7))
tbg_v, tv = build_twisted_bilayer(4, 3)
c = tbg_v.cart_coords
nh = len(tbg_v) // 2

axes[0].scatter(c[:nh,0], c[:nh,1], s=5, c="navy", alpha=0.6,
                label="Bottom")
axes[0].scatter(c[nh:,0], c[nh:,1], s=5, c="crimson", alpha=0.6,
                label="Top")
axes[0].set_aspect("equal")
axes[0].set_title(f"TBG (4,3), theta={tv:.2f} deg")
axes[0].legend(markerscale=3)

tree = cKDTree(c[nh:,:2])
_, idx = tree.query(c[:nh,:2])
offsets = np.linalg.norm(c[nh:,:2][idx] - c[:nh,:2], axis=1)
sc = axes[1].scatter(c[:nh,0], c[:nh,1], s=5, c=offsets, cmap="viridis")
plt.colorbar(sc, ax=axes[1], label="Interlayer offset (A)")
axes[1].set_aspect("equal")
axes[1].set_title("Local Stacking Registry")

for ax in axes:
    ax.set_xlabel("x (A)"); ax.set_ylabel("y (A)")
    ax.grid(True, alpha=0.2)
plt.tight_layout()
plt.savefig("moire_pattern.png", dpi=150)
print("  Saved: moire_pattern.png")

print("\n=== All files written ===")
for f in sorted(Path(".").glob("*moire*")) + sorted(Path(".").glob("TBG*")) \
       + sorted(Path(".").glob("POSCAR_TBG*")):
    print(f"  {f}")
```

### Method B: VASP (VASPKIT Task 825)

```python
#!/usr/bin/env python3
"""
VASP input files for moire superlattice DFT.
Equivalent to VASPKIT Task 825: Build Hexagonal Moire Superlattices.
"""

import numpy as np
from pymatgen.core import Structure, Lattice
from pymatgen.io.vasp import Poscar, Incar, Kpoints

# ============================================================
# 1. Build moire POSCAR
# ============================================================
print("=== VASP Moire Setup (VASPKIT 825 equivalent) ===")

def moire_angle_calc(m, n):
    return np.degrees(np.arccos(np.clip(
        (m**2+4*m*n+n**2) / (2*(m**2+m*n+n**2)), -1, 1)))

def build_tbg_poscar(m, n, a_cc=1.42, d_layer=3.35):
    """Build TBG Structure for VASP."""
    a = a_cc * np.sqrt(3)
    tr = np.radians(moire_angle_calc(m, n))
    a1 = np.array([a, 0, 0])
    a2 = np.array([a*np.cos(np.radians(60)), a*np.sin(np.radians(60)), 0])
    L1, L2 = m*a1+n*a2, -n*a1+(m+n)*a2
    L3 = np.array([0, 0, d_layer+20.0])
    b1, b2 = (1/3)*a1+(2/3)*a2, (2/3)*a1+(1/3)*a2
    R = np.array([[np.cos(tr),-np.sin(tr),0],
                  [np.sin(tr), np.cos(tr),0], [0,0,1]])
    shift = (a1+a2)/3
    b1r, b2r = R@(b1+shift), R@(b2+shift)
    a1r, a2r = R@a1, R@a2
    inv_L = np.linalg.inv(np.array([L1[:2],L2[:2]]).T)
    inv_Lt = np.linalg.inv(np.array([(R@L1)[:2],(R@L2)[:2]]).T)
    atoms, sp = [], []
    mx = m+n+2
    for p in range(-mx, mx+1):
        for q in range(-mx, mx+1):
            for b in [b1,b2]:
                pos = p*a1+q*a2+b
                f = inv_L @ pos[:2]
                if -1e-6<=f[0]<1-1e-6 and -1e-6<=f[1]<1-1e-6:
                    atoms.append([pos[0],pos[1],10.0]); sp.append("C")
            for b in [b1r,b2r]:
                pos = p*a1r+q*a2r+b
                f = inv_Lt @ pos[:2]
                if -1e-6<=f[0]<1-1e-6 and -1e-6<=f[1]<1-1e-6:
                    atoms.append([pos[0],pos[1],10.0+d_layer]); sp.append("C")
    return Structure(Lattice(np.array([L1,L2,L3])), sp,
                     np.array(atoms), coords_are_cartesian=True)

m_v, n_v = 5, 4
theta_v = moire_angle_calc(m_v, n_v)
tbg = build_tbg_poscar(m_v, n_v)
Poscar(tbg).write_file("POSCAR_moire")
print(f"  ({m_v},{n_v}), angle={theta_v:.4f} deg, {len(tbg)} atoms")

# ============================================================
# 2. INCAR with vdW corrections
# ============================================================
# DFT-D3-BJ
Incar({
    "SYSTEM": f"TBG ({m_v},{n_v}) D3",
    "ENCUT": 500, "PREC": "Accurate", "EDIFF": 1e-6, "NELM": 200,
    "GGA": "PE", "IVDW": 12,
    "IBRION": 2, "NSW": 100, "EDIFFG": -0.01, "ISIF": 2,
    "ISMEAR": 0, "SIGMA": 0.05, "ISPIN": 1,
    "ALGO": "Normal", "LREAL": "Auto", "NCORE": 4,
    "LWAVE": False, "LCHARG": True, "LORBIT": 11,
}).write_file("INCAR_moire_D3")

# optB88-vdW
Incar({
    "SYSTEM": f"TBG ({m_v},{n_v}) vdW-DF",
    "ENCUT": 500, "PREC": "Accurate", "EDIFF": 1e-6, "NELM": 200,
    "GGA": "BO", "PARAM1": 0.1833333333, "PARAM2": 0.22,
    "LUSE_VDW": True, "AGGAC": 0.0, "LASPH": True,
    "IBRION": 2, "NSW": 100, "EDIFFG": -0.01, "ISIF": 2,
    "ISMEAR": 0, "SIGMA": 0.05, "ISPIN": 1,
    "ALGO": "Normal", "LREAL": "Auto", "NCORE": 4,
    "LWAVE": False, "LCHARG": True, "LORBIT": 11,
}).write_file("INCAR_moire_vdW")
print("  INCAR_moire_D3 (PBE+D3-BJ), INCAR_moire_vdW (optB88-vdW)")

# ============================================================
# 3. KPOINTS
# ============================================================
L = np.linalg.norm(tbg.lattice.matrix[0])
k = max(1, int(np.ceil(30.0 / L)))
Kpoints.gamma_automatic(kpts=(k, k, 1)).write_file("KPOINTS_moire")
print(f"  KPOINTS_moire: {k}x{k}x1 (L={L:.1f} A)")

# ============================================================
# 4. VASPKIT 825 reference
# ============================================================
print("\n=== VASPKIT 825 Workflow ===")
print("  1. Prepare monolayer POSCAR")
print("  2. vaspkit -> 8 -> 825")
print("  3. Enter (m,n) or twist angle")
print("  4. Generates moire POSCAR")
print("  Key: vdW ESSENTIAL (IVDW=12 or LUSE_VDW), ISIF=2,")
print("       sparse k-mesh, LREAL=Auto, vacuum >= 20 A")
```

---

## Key Parameters

| Parameter | Typical Value | Notes |
|---|---|---|
| Twist angle | 0.5--30 deg | Only discrete commensurate angles give periodic cells |
| (m, n) indices | m > n >= 0, gcd=1 | Determines angle and cell size |
| C-C bond length | 1.42 A | Standard sp2 carbon |
| Interlayer distance | 3.35 A (graphene), 6.1--6.5 A (TMD) | Relaxes with vdW |
| Vacuum | 20--25 A | Larger than bulk 2D due to corrugation |
| vdW correction | DFT-D3-BJ or optB88-vdW | Essential for interlayer binding |
| k-mesh | 1x1x1 to 3x3x1 | k*L ~ 30-40 A |
| Magic angle | 1.08--1.12 deg | ~11,000 atoms for commensurate cell |
| TMD mismatch | 0--5% | MoS2/WS2 ~ 0.2%; use average a |

## Interpreting Results

| Observable | What to Check | Typical Behavior |
|---|---|---|
| Flat bands | Bandwidth near E_F | < 10 meV at magic angle |
| Interlayer binding | E_bilayer - 2*E_mono | ~20 meV/atom (graphene, vdW-DF) |
| Stacking domains | AA vs AB/BA regions | AA has larger interlayer distance |
| Corrugation | z-variation across cell | 0.1--0.3 A (TBG), larger for TMDs |
| DOS peaks | Van Hove singularities | Peaks near E_F = flat bands |
| Moire wavelength | a / (2 sin(theta/2)) | ~12.8 nm at magic angle |

## Common Issues

| Problem | Solution |
|---|---|
| Commensurate cell too large | Use larger twist angle or approximate (m,n). Magic angle TBG ~11k atoms: use MACE or tight-binding. |
| Layer atom counts differ | Increase search grid (max_pq). Boundary atoms may be missed. |
| Layers fly apart (no binding) | Always use vdW: IVDW=12 or LUSE_VDW=.TRUE. |
| SCF convergence issues | ALGO=All, SIGMA=0.02, NELM=300. Start from pre-converged CHGCAR. |
| Heterobilayer lattice mismatch | Use average a. Strain ~0.1% for MoS2/WS2 is acceptable. |
| k-mesh too dense | Gamma-only or 2x2x1 max for L > 30 A cells. |
| Wrong band structure | Verify interlayer distance after relaxation. Compare to literature. |
| MACE interlayer distance off | MACE-MP may not capture vdW well. Use `dispersion=True`. Compare to DFT-D3. |
| Cell not hexagonal | Verify |L1|=|L2| and 60 deg angle. Guaranteed by (m,n) construction. |
