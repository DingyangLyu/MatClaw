# Bond Length and Bond Angle Distribution Analysis

## When to Use

- Analyzing MD trajectories for interatomic distance statistics and local geometry
- Detecting bond breaking and forming events during reactive simulations
- Quantifying structural disorder in amorphous materials, glasses, and melts
- Characterizing local coordination environments in liquids and supercooled liquids
- Comparing simulated bond length/angle distributions with experimental X-ray or neutron data
- Monitoring structural evolution across phase transitions (crystalline to amorphous, melting)

## Method Selection

| Criterion | ASE + MACE | QE AIMD | VASP AIMD |
|---|---|---|---|
| Accuracy | ML potential (near-DFT) | Full DFT | Full DFT |
| Speed | 1000x faster than DFT | Slow (plane-wave DFT) | Slow (plane-wave DFT) |
| System size | 100-10000+ atoms | 50-300 atoms | 50-300 atoms |
| Trajectory length | 100 ps - 10 ns | 5-50 ps | 5-50 ps |
| Reactive chemistry | Model-dependent | Yes (Born-Oppenheimer) | Yes (Born-Oppenheimer) |
| Best for | Large-scale statistics, long trajectories | Benchmark accuracy, small cells | Benchmark accuracy, VASPKIT 730/731 |
| Trajectory format | ASE `.traj` | `.pos` / `.cel` files | `XDATCAR` |

## Prerequisites

Pre-installed: `ase`, `pymatgen`, `numpy`, `scipy`, `matplotlib`, `mace-torch`.

For trajectory generation: MACE foundation models are available locally. QE and VASP require separate installations. Analysis works on any trajectory format: ASE `.traj`, LAMMPS dump, VASP `XDATCAR`, or QE `.pos`/`.cel`.

## Detailed Steps

### Method A: ASE + MACE MD followed by Bond Analysis

```python
#!/usr/bin/env python3
"""
Bond length and bond angle distribution from MACE MD trajectories.
Runs NVT MD, then computes: (a) bond length distributions per element pair
with KDE smoothing, (b) bond angle distributions for A-B-C triplets,
(c) time evolution of mean bond lengths.
Equivalent to VASPKIT tasks 730 and 731.
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from collections import defaultdict
from itertools import combinations_with_replacement
import os, warnings
warnings.filterwarnings("ignore")

# ============================================================
# 1. CONFIGURATION
# ============================================================
STRUCTURE_FILE = "POSCAR"
SUPERCELL = (2, 2, 2)
MACE_MODEL = "medium"             # "small", "medium", "large", or path to .model
TEMPERATURE = 300.0               # K
TIMESTEP = 1.0                    # fs
N_EQUIL_STEPS = 500
N_PROD_STEPS = 5000
TRAJ_INTERVAL = 10
FRICTION = 0.01                   # Langevin friction (1/fs)

BOND_CUTOFFS = {}                 # e.g. {"Si-O": (1.2, 2.2)}; empty = auto
AUTO_CUTOFF_FACTOR = 1.3
BL_N_BINS = 200
KDE_BANDWIDTH = 0.02              # Angstrom; 0 = no KDE

ANGLE_TRIPLETS = []               # e.g. [("O","Si","O")]; empty = auto
ANGLE_BOND_CUTOFF = 2.5           # A
ANGLE_N_BINS = 180
FRAME_STEP = 1

OUTPUT_DIR = "bond_analysis"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ============================================================
# 2. RUN MACE NVT MD
# ============================================================
from ase.io import read
from ase.md.langevin import Langevin
from ase.io.trajectory import Trajectory
from ase import units
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution
from ase.neighborlist import neighbor_list

print("=" * 60)
print("  Bond Length & Angle Distribution — MACE NVT MD")
print("=" * 60)

atoms = read(STRUCTURE_FILE)
atoms = atoms.repeat(SUPERCELL)
n_atoms = len(atoms)
print(f"Structure: {atoms.get_chemical_formula()}, {n_atoms} atoms")
print(f"Cell: {atoms.cell.cellpar()[:3].round(3)} A")

from mace.calculators import mace_mp
atoms.calc = mace_mp(model=MACE_MODEL, default_dtype="float64")

MaxwellBoltzmannDistribution(atoms, temperature_K=TEMPERATURE)
dyn = Langevin(atoms, TIMESTEP * units.fs, temperature_K=TEMPERATURE, friction=FRICTION)

traj_file = os.path.join(OUTPUT_DIR, "md.traj")
traj_writer = Trajectory(traj_file, "w", atoms)
log_data = []

def log_step():
    t = dyn.get_number_of_steps() * TIMESTEP / 1000
    epot = atoms.get_potential_energy() / n_atoms
    ekin = atoms.get_kinetic_energy() / n_atoms
    temp = 2 * atoms.get_kinetic_energy() / (3 * n_atoms * units.kB)
    log_data.append([t, epot, ekin, epot + ekin, temp])

dyn.attach(log_step, interval=TRAJ_INTERVAL)

print(f"Equilibration: {N_EQUIL_STEPS} steps")
dyn.run(N_EQUIL_STEPS)
log_data.clear()

print(f"Production: {N_PROD_STEPS} steps")
dyn.attach(lambda: traj_writer.write(atoms), interval=TRAJ_INTERVAL)
dyn.run(N_PROD_STEPS)
traj_writer.close()

log_arr = np.array(log_data)
np.savetxt(os.path.join(OUTPUT_DIR, "md_energy.dat"), log_arr,
           header="time(ps) Epot Ekin Etot T(K)", fmt="%10.4f %14.6f %14.6f %14.6f %10.2f")

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 6), sharex=True)
ax1.plot(log_arr[:,0], log_arr[:,3], "b-", lw=0.8); ax1.set_ylabel("Etot (eV/atom)"); ax1.grid(True, alpha=0.3)
ax2.plot(log_arr[:,0], log_arr[:,4], "r-", lw=0.8); ax2.axhline(TEMPERATURE, color="gray", ls="--", lw=0.5)
ax2.set_xlabel("Time (ps)"); ax2.set_ylabel("T (K)"); ax2.grid(True, alpha=0.3)
fig.tight_layout(); fig.savefig(os.path.join(OUTPUT_DIR, "md_energy_temp.png"), dpi=150); plt.close()

# ============================================================
# 3. LOAD FRAMES
# ============================================================
traj = Trajectory(traj_file, "r")
frames = [traj[i].copy() for i in range(0, len(traj), FRAME_STEP)]
traj.close()
n_frames = len(frames)
symbols = frames[0].get_chemical_symbols()
unique_elements = sorted(set(symbols))
print(f"Frames: {n_frames}, Elements: {unique_elements}")

# ============================================================
# 4. BOND LENGTH DISTRIBUTION (VASPKIT 730)
# ============================================================
print("\n--- Bond Length Distribution (VASPKIT 730) ---")

def compute_bond_lengths(frames, el_a, el_b, r_min, r_max):
    all_dists = []
    for atoms in frames:
        syms = atoms.get_chemical_symbols()
        idx_a = set(i for i,s in enumerate(syms) if s == el_a)
        idx_b = set(i for i,s in enumerate(syms) if s == el_b)
        i_arr, j_arr, d_arr = neighbor_list("ijd", atoms, cutoff=r_max)
        for ii, jj, dd in zip(i_arr, j_arr, d_arr):
            if dd < r_min: continue
            if el_a == el_b:
                if ii in idx_a and jj in idx_b and ii < jj: all_dists.append(dd)
            else:
                if ii in idx_a and jj in idx_b: all_dists.append(dd)
    return np.array(all_dists)

# Auto-detect cutoffs
if not BOND_CUTOFFS:
    ref = frames[0]; syms_ref = ref.get_chemical_symbols()
    i0, j0, d0 = neighbor_list("ijd", ref, cutoff=5.0)
    for el_a, el_b in combinations_with_replacement(unique_elements, 2):
        sa = set(i for i,s in enumerate(syms_ref) if s == el_a)
        sb = set(i for i,s in enumerate(syms_ref) if s == el_b)
        dists = [dd for ii,jj,dd in zip(i0,j0,d0) if (ii in sa and jj in sb) or (ii in sb and jj in sa)]
        if dists:
            d_min = min(dists)
            h,e = np.histogram(dists, bins=100, range=(d_min*0.8, 5.0))
            pk = np.argmax(h); srch = h[pk:]; mi = np.argmin(srch)
            r_cut = 0.5*(e[pk+mi]+e[pk+mi+1])
            BOND_CUTOFFS[f"{el_a}-{el_b}"] = (round(max(d_min*0.85,0.5),2), round(min(r_cut*1.1,5.0),2))
    print(f"  Auto cutoffs: {BOND_CUTOFFS}")

bond_results = {}
for pair_key, (r_lo, r_hi) in BOND_CUTOFFS.items():
    el_a, el_b = pair_key.split("-")
    raw = compute_bond_lengths(frames, el_a, el_b, r_lo, r_hi)
    if len(raw) == 0:
        print(f"  {pair_key}: no bonds in ({r_lo},{r_hi}) A"); continue
    hist, edges = np.histogram(raw, bins=BL_N_BINS, range=(r_lo,r_hi), density=True)
    rc = 0.5*(edges[:-1]+edges[1:])
    kde = np.zeros_like(rc)
    if KDE_BANDWIDTH > 0 and len(raw) > 10:
        from scipy.stats import gaussian_kde as gkde
        kde = gkde(raw, bw_method=KDE_BANDWIDTH)(rc)
    bond_results[pair_key] = {"r":rc, "hist":hist, "kde":kde, "raw":raw, "r_lo":r_lo, "r_hi":r_hi}
    print(f"  {pair_key}: mean={np.mean(raw):.4f} std={np.std(raw):.4f} A, bonds/frame~{len(raw)//n_frames}")
    np.savetxt(os.path.join(OUTPUT_DIR, f"bond_length_{el_a}_{el_b}.dat"),
               np.column_stack([rc,hist,kde]), header="r(A) histogram KDE", fmt="%10.5f %12.6f %12.6f")

# Stacked histogram plot
if bond_results:
    colors = plt.cm.tab10(np.linspace(0,1,10))
    n_p = len(bond_results); nc = min(3,n_p); nr = (n_p+nc-1)//nc
    fig, axes = plt.subplots(nr, nc, figsize=(5.5*nc, 4*nr), squeeze=False)
    for idx,(pk,d) in enumerate(bond_results.items()):
        ax = axes[idx//nc][idx%nc]
        ax.bar(d["r"], d["hist"], width=(d["r_hi"]-d["r_lo"])/BL_N_BINS, alpha=0.45, color=colors[idx%10], label="Hist")
        if KDE_BANDWIDTH > 0: ax.plot(d["r"], d["kde"], "k-", lw=1.8, label="KDE")
        m = np.mean(d["raw"]); ax.axvline(m, color="red", ls="--", lw=1.2, label=f"Mean: {m:.3f}")
        ax.set_xlabel("Bond Length ($\\AA$)"); ax.set_ylabel("P(r)"); ax.set_title(pk)
        ax.legend(fontsize=8); ax.grid(True, alpha=0.3)
    for idx in range(n_p, nr*nc): axes[idx//nc][idx%nc].set_visible(False)
    fig.suptitle("Bond Length Distributions", fontsize=14, y=1.01)
    fig.tight_layout(); fig.savefig(os.path.join(OUTPUT_DIR, "bond_length_distributions.png"), dpi=150, bbox_inches="tight"); plt.close()

    fig, ax = plt.subplots(figsize=(8,5))
    for idx,(pk,d) in enumerate(bond_results.items()):
        curve = d["kde"] if KDE_BANDWIDTH > 0 else d["hist"]
        ax.plot(d["r"], curve, "-", color=colors[idx%10], lw=1.8, label=pk)
    ax.set_xlabel("Bond Length ($\\AA$)"); ax.set_ylabel("P(r)"); ax.set_title("All Pairs Overlay")
    ax.legend(); ax.grid(True, alpha=0.3)
    fig.tight_layout(); fig.savefig(os.path.join(OUTPUT_DIR, "bond_length_overlay.png"), dpi=150); plt.close()

# ============================================================
# 5. BOND ANGLE DISTRIBUTION (VASPKIT 731)
# ============================================================
print("\n--- Bond Angle Distribution (VASPKIT 731) ---")

def compute_angles(frames, el_a, el_cen, el_c, cutoff, n_bins=180):
    all_ang = []
    for atoms in frames:
        syms = atoms.get_chemical_symbols()
        i_arr, j_arr, d_arr, D_arr = neighbor_list("ijdD", atoms, cutoff=cutoff)
        nbrs = defaultdict(list)
        for ii,jj,dd,disp in zip(i_arr,j_arr,d_arr,D_arr): nbrs[ii].append((jj,disp,syms[jj]))
        for ci in range(len(syms)):
            if syms[ci] != el_cen: continue
            va = [disp for _,disp,sj in nbrs[ci] if sj == el_a]
            vc = [disp for _,disp,sj in nbrs[ci] if sj == el_c]
            pairs = ([(va[i],va[j]) for i in range(len(va)) for j in range(i+1,len(va))]
                     if el_a == el_c else [(v1,v2) for v1 in va for v2 in vc])
            for v1,v2 in pairs:
                n1,n2 = np.linalg.norm(v1), np.linalg.norm(v2)
                if n1 < 1e-10 or n2 < 1e-10: continue
                ct = np.clip(np.dot(v1,v2)/(n1*n2), -1, 1)
                all_ang.append(np.degrees(np.arccos(ct)))
    all_ang = np.array(all_ang) if all_ang else np.array([])
    if len(all_ang) == 0: return np.linspace(0,180,n_bins), np.zeros(n_bins), all_ang
    hist, edges = np.histogram(all_ang, bins=n_bins, range=(0,180), density=True)
    return 0.5*(edges[:-1]+edges[1:]), hist, all_ang

if not ANGLE_TRIPLETS:
    bto = defaultdict(set)
    for pk in BOND_CUTOFFS:
        a,b = pk.split("-"); bto[a].add(b); bto[b].add(a)
    for cen in unique_elements:
        for pair in combinations_with_replacement(sorted(bto.get(cen,set())), 2):
            ANGLE_TRIPLETS.append((pair[0], cen, pair[1]))
    print(f"  Auto triplets: {[f'{a}-{b}-{c}' for a,b,c in ANGLE_TRIPLETS]}")

angle_results = {}
ref_geom = {109.47:"tetrahedral(sp3)", 90.0:"octahedral", 120.0:"trigonal(sp2)", 180.0:"linear(sp)"}
for el_a, el_cen, el_c in ANGLE_TRIPLETS:
    lab = f"{el_a}-{el_cen}-{el_c}"
    theta, prob, raw = compute_angles(frames, el_a, el_cen, el_c, ANGLE_BOND_CUTOFF, ANGLE_N_BINS)
    angle_results[lab] = {"theta":theta, "prob":prob, "raw":raw}
    if len(raw) > 0:
        pk = theta[np.argmax(prob)]
        print(f"  {lab}: peak={pk:.1f} deg, mean={np.mean(raw):.1f}, std={np.std(raw):.1f}, n={len(raw)}")
        for ref,geom in ref_geom.items():
            if abs(pk - ref) < 10: print(f"    --> {geom} ({ref} deg)")
    else:
        print(f"  {lab}: no angles (increase ANGLE_BOND_CUTOFF)")
    np.savetxt(os.path.join(OUTPUT_DIR, f"bond_angle_{el_a}_{el_cen}_{el_c}.dat"),
               np.column_stack([theta,prob]), header="angle(deg) prob_density", fmt="%8.2f %12.6f")

if angle_results:
    nt = len(angle_results); nc = min(3,nt); nr = (nt+nc-1)//nc
    fig, axes = plt.subplots(nr, nc, figsize=(5.5*nc, 4*nr), squeeze=False)
    for idx,(lab,d) in enumerate(angle_results.items()):
        ax = axes[idx//nc][idx%nc]
        ax.fill_between(d["theta"], d["prob"], alpha=0.4, color="coral")
        ax.plot(d["theta"], d["prob"], "r-", lw=1.5)
        if len(d["raw"]) > 0:
            pv = d["theta"][np.argmax(d["prob"])]
            ax.axvline(pv, color="darkred", ls="--", lw=1, label=f"Peak: {pv:.1f}")
            for ref,ml in [(109.47,"Td"),(90.0,"Oh"),(120.0,"sp2")]:
                if abs(pv-ref) < 20: ax.axvline(ref, color="gray", ls=":", lw=0.8, label=f"{ml}({ref:.0f})")
            ax.legend(fontsize=8)
        ax.set_xlabel("Angle (deg)"); ax.set_ylabel("P(theta)"); ax.set_title(lab); ax.set_xlim(0,180); ax.grid(True, alpha=0.3)
    for idx in range(nt, nr*nc): axes[idx//nc][idx%nc].set_visible(False)
    fig.suptitle("Bond Angle Distributions", fontsize=14, y=1.01)
    fig.tight_layout(); fig.savefig(os.path.join(OUTPUT_DIR, "bond_angle_distributions.png"), dpi=150, bbox_inches="tight"); plt.close()

# ============================================================
# 6. TIME EVOLUTION OF MEAN BOND LENGTH
# ============================================================
print("\n--- Mean Bond Length vs Time ---")

time_ps = np.arange(n_frames) * TRAJ_INTERVAL * TIMESTEP / 1000.0
fig, axes = plt.subplots(len(BOND_CUTOFFS), 1, figsize=(8, 3.5*len(BOND_CUTOFFS)), squeeze=False)

for idx, (pk, (r_lo, r_hi)) in enumerate(BOND_CUTOFFS.items()):
    el_a, el_b = pk.split("-")
    means, stds = [], []
    for atoms in frames:
        syms = atoms.get_chemical_symbols()
        ia = set(i for i,s in enumerate(syms) if s == el_a)
        ib = set(i for i,s in enumerate(syms) if s == el_b)
        i_arr, j_arr, d_arr = neighbor_list("ijd", atoms, cutoff=r_hi)
        ds = [dd for ii,jj,dd in zip(i_arr,j_arr,d_arr) if dd >= r_lo and
              ((el_a==el_b and ii in ia and jj in ib and ii<jj) or (el_a!=el_b and ii in ia and jj in ib))]
        means.append(np.mean(ds) if ds else np.nan)
        stds.append(np.std(ds) if ds else np.nan)
    means, stds = np.array(means), np.array(stds)

    ax = axes[idx][0]; v = ~np.isnan(means)
    ax.plot(time_ps[v], means[v], "b-", lw=0.8, alpha=0.6, label="Per-frame")
    w = max(5, n_frames//10)
    if np.sum(v) > w:
        ra = np.convolve(means[v], np.ones(w)/w, mode="valid")
        ta = time_ps[v][w//2:w//2+len(ra)]
        ax.plot(ta, ra, "r-", lw=2, label=f"Running avg (w={w})")
    om = np.nanmean(means)
    ax.axhline(om, color="gray", ls="--", lw=1, label=f"Overall: {om:.4f}")
    ax.fill_between(time_ps[v], means[v]-stds[v], means[v]+stds[v], alpha=0.15, color="blue")
    ax.set_xlabel("Time (ps)"); ax.set_ylabel(f"{pk} ($\\AA$)"); ax.set_title(f"{pk} Evolution")
    ax.legend(fontsize=9); ax.grid(True, alpha=0.3)
    np.savetxt(os.path.join(OUTPUT_DIR, f"bond_evolution_{el_a}_{el_b}.dat"),
               np.column_stack([time_ps,means,stds]), header="time(ps) mean(A) std(A)", fmt="%10.4f %12.6f %12.6f")
    print(f"  {pk}: overall mean = {om:.4f} A")

fig.tight_layout(); fig.savefig(os.path.join(OUTPUT_DIR, "bond_length_evolution.png"), dpi=150, bbox_inches="tight"); plt.close()

print(f"\nDone. Output: {OUTPUT_DIR}/")
for f in sorted(os.listdir(OUTPUT_DIR)): print(f"  {f}  ({os.path.getsize(os.path.join(OUTPUT_DIR,f))/1024:.1f} kB)")
```

### Method B: QE AIMD Trajectory

```python
#!/usr/bin/env python3
"""
Bond length/angle distribution from QE AIMD .pos/.cel files.
Parses cp.x or pw.x output, converts to ASE Atoms, reuses the same
neighbor-list analysis as Method A.
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from collections import defaultdict
from itertools import combinations_with_replacement
from ase import Atoms
from ase.neighborlist import neighbor_list
import os

QE_POS_FILE = "prefix.pos"        # atomic positions (Bohr)
QE_CEL_FILE = "prefix.cel"        # cell vectors (Bohr); optional for constant-cell
ELEMENT_ORDER = ["Si", "O"]
ATOMS_PER_ELEMENT = [8, 16]
N_EQUIL_FRAMES = 100; FRAME_STEP = 1
BL_PAIRS = {"Si-O": (1.2, 2.2), "O-O": (2.0, 3.2)}
ANGLE_TRIPLETS = [("O", "Si", "O"), ("Si", "O", "Si")]
ANGLE_CUTOFF = 2.2; BL_N_BINS = 200; ANGLE_N_BINS = 180
OUTPUT_DIR = "bond_analysis_qe"
os.makedirs(OUTPUT_DIR, exist_ok=True)

def parse_qe_pos(pos_file, elements, counts):
    bohr2ang = 0.529177249
    n = sum(counts)
    syms = [el for el,c in zip(elements,counts) for _ in range(c)]
    frames = []
    with open(pos_file) as f: lines = f.readlines()
    i = 0
    while i < len(lines):
        line = lines[i].strip()
        if not line: i += 1; continue
        try: int(line.split()[0])
        except ValueError: i += 1; continue
        pos = []
        for j in range(1, n+1):
            if i+j >= len(lines): break
            c = lines[i+j].split()
            if len(c) >= 3: pos.append([float(x)*bohr2ang for x in c[:3]])
        if len(pos) == n: frames.append(np.array(pos))
        i += n + 1
    return frames, syms

def parse_qe_cel(cel_file):
    bohr2ang = 0.529177249; cells = []
    with open(cel_file) as f: lines = f.readlines()
    i = 0
    while i < len(lines):
        line = lines[i].strip()
        if not line: i += 1; continue
        try: int(line.split()[0])
        except ValueError: i += 1; continue
        cell = []
        for j in range(1,4):
            if i+j < len(lines): cell.append([float(x)*bohr2ang for x in lines[i+j].split()[:3]])
        if len(cell) == 3: cells.append(np.array(cell))
        i += 4
    return cells

all_pos, symbols = parse_qe_pos(QE_POS_FILE, ELEMENT_ORDER, ATOMS_PER_ELEMENT)
all_cells = parse_qe_cel(QE_CEL_FILE) if os.path.exists(QE_CEL_FILE) else None

frames = []
for idx in range(N_EQUIL_FRAMES, len(all_pos), FRAME_STEP):
    cell = all_cells[idx] if all_cells and idx < len(all_cells) else (all_cells[0] if all_cells else np.eye(3)*20)
    frames.append(Atoms(symbols=symbols, positions=all_pos[idx], cell=cell, pbc=True))
n_frames = len(frames)
print(f"QE frames: {n_frames}, atoms: {len(symbols)}")

# Bond length distributions
for pair_key, (r_lo, r_hi) in BL_PAIRS.items():
    el_a, el_b = pair_key.split("-"); all_d = []
    for atoms in frames:
        syms = atoms.get_chemical_symbols()
        sa = set(i for i,s in enumerate(syms) if s == el_a)
        sb = set(i for i,s in enumerate(syms) if s == el_b)
        ia, ja, da = neighbor_list("ijd", atoms, cutoff=r_hi)
        for ii,jj,dd in zip(ia,ja,da):
            if dd < r_lo: continue
            if el_a == el_b:
                if ii in sa and jj in sb and ii < jj: all_d.append(dd)
            else:
                if ii in sa and jj in sb: all_d.append(dd)
    if all_d:
        all_d = np.array(all_d)
        hist,edges = np.histogram(all_d, bins=BL_N_BINS, range=(r_lo,r_hi), density=True)
        rc = 0.5*(edges[:-1]+edges[1:])
        print(f"  {pair_key}: mean={np.mean(all_d):.4f} std={np.std(all_d):.4f} A")
        np.savetxt(os.path.join(OUTPUT_DIR, f"bond_length_{el_a}_{el_b}.dat"),
                   np.column_stack([rc,hist]), header="r(A) prob_density", fmt="%10.5f %12.6f")

# Bond angle distributions
for el_a, el_cen, el_c in ANGLE_TRIPLETS:
    lab = f"{el_a}-{el_cen}-{el_c}"; all_ang = []
    for atoms in frames:
        syms = atoms.get_chemical_symbols()
        ia,ja,da,Da = neighbor_list("ijdD", atoms, cutoff=ANGLE_CUTOFF)
        nbrs = defaultdict(list)
        for ii,jj,dd,disp in zip(ia,ja,da,Da): nbrs[ii].append((jj,disp,syms[jj]))
        for ci in range(len(syms)):
            if syms[ci] != el_cen: continue
            va = [d for _,d,s in nbrs[ci] if s == el_a]
            vc = [d for _,d,s in nbrs[ci] if s == el_c]
            prs = ([(va[i],va[j]) for i in range(len(va)) for j in range(i+1,len(va))]
                   if el_a == el_c else [(v1,v2) for v1 in va for v2 in vc])
            for v1,v2 in prs:
                n1,n2 = np.linalg.norm(v1),np.linalg.norm(v2)
                if n1<1e-10 or n2<1e-10: continue
                all_ang.append(np.degrees(np.arccos(np.clip(np.dot(v1,v2)/(n1*n2),-1,1))))
    if all_ang:
        all_ang = np.array(all_ang)
        hist,edges = np.histogram(all_ang, bins=ANGLE_N_BINS, range=(0,180), density=True)
        cen = 0.5*(edges[:-1]+edges[1:])
        print(f"  {lab}: peak={cen[np.argmax(hist)]:.1f} deg, n={len(all_ang)}")
        np.savetxt(os.path.join(OUTPUT_DIR, f"bond_angle_{el_a}_{el_cen}_{el_c}.dat"),
                   np.column_stack([cen,hist]), header="angle(deg) prob_density", fmt="%8.2f %12.6f")

print(f"Done. Output: {OUTPUT_DIR}/")
```

### Method C: VASP AIMD (XDATCAR)

```python
#!/usr/bin/env python3
"""
Bond length/angle distribution from VASP XDATCAR.
Equivalent to VASPKIT tasks 730 (bond length) and 731 (bond angle).
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from collections import defaultdict
from itertools import combinations_with_replacement
from pymatgen.io.vasp import Xdatcar
from pymatgen.io.ase import AseAtomsAdaptor
from ase.neighborlist import neighbor_list
import os

XDATCAR_FILE = "XDATCAR"
N_EQUIL_FRAMES = 200; FRAME_STEP = 1
BL_PAIRS = {}                     # empty = auto-detect
ANGLE_TRIPLETS = []               # empty = auto-detect
ANGLE_CUTOFF = 2.5; BL_N_BINS = 200; ANGLE_N_BINS = 180
OUTPUT_DIR = "bond_analysis_vasp"
os.makedirs(OUTPUT_DIR, exist_ok=True)

print("Parsing XDATCAR...")
xdatcar = Xdatcar(XDATCAR_FILE)
adaptor = AseAtomsAdaptor()
frames = [adaptor.get_atoms(xdatcar.structures[i])
          for i in range(N_EQUIL_FRAMES, len(xdatcar.structures), FRAME_STEP)]
n_frames = len(frames)
symbols = frames[0].get_chemical_symbols()
unique_elements = sorted(set(symbols))
print(f"Frames: {n_frames}, Formula: {frames[0].get_chemical_formula()}")

# Auto-detect bond pairs
if not BL_PAIRS:
    ref = frames[0]; sr = ref.get_chemical_symbols()
    i0,j0,d0 = neighbor_list("ijd", ref, cutoff=5.0)
    for ea,eb in combinations_with_replacement(unique_elements, 2):
        sa = set(i for i,s in enumerate(sr) if s == ea)
        sb = set(i for i,s in enumerate(sr) if s == eb)
        ds = [dd for ii,jj,dd in zip(i0,j0,d0) if (ii in sa and jj in sb) or (ii in sb and jj in sa)]
        if ds:
            dm = min(ds)
            BL_PAIRS[f"{ea}-{eb}"] = (round(dm*0.85,2), round(dm*1.4,2))
    print(f"  Auto pairs: {BL_PAIRS}")

# Bond length distributions
print("\n--- Bond Lengths (VASPKIT 730) ---")
for pk,(r_lo,r_hi) in BL_PAIRS.items():
    ea,eb = pk.split("-"); all_d = []
    for atoms in frames:
        sy = atoms.get_chemical_symbols()
        sa = set(i for i,s in enumerate(sy) if s == ea)
        sb = set(i for i,s in enumerate(sy) if s == eb)
        ia,ja,da = neighbor_list("ijd", atoms, cutoff=r_hi)
        for ii,jj,dd in zip(ia,ja,da):
            if dd < r_lo: continue
            if ea == eb:
                if ii in sa and jj in sb and ii < jj: all_d.append(dd)
            else:
                if ii in sa and jj in sb: all_d.append(dd)
    if all_d:
        all_d = np.array(all_d)
        hist,edges = np.histogram(all_d, bins=BL_N_BINS, range=(r_lo,r_hi), density=True)
        rc = 0.5*(edges[:-1]+edges[1:])
        pk_r = rc[np.argmax(hist)]
        print(f"  {pk}: peak={pk_r:.4f}, mean={np.mean(all_d):.4f}, std={np.std(all_d):.4f} A")
        np.savetxt(os.path.join(OUTPUT_DIR, f"bond_length_{ea}_{eb}.dat"),
                   np.column_stack([rc,hist]), header="r(A) prob_density", fmt="%10.5f %12.6f")

# Bond angle distributions
print("\n--- Bond Angles (VASPKIT 731) ---")
if not ANGLE_TRIPLETS:
    bto = defaultdict(set)
    for pk in BL_PAIRS: a,b = pk.split("-"); bto[a].add(b); bto[b].add(a)
    for cen in unique_elements:
        for pair in combinations_with_replacement(sorted(bto.get(cen,set())), 2):
            ANGLE_TRIPLETS.append((pair[0], cen, pair[1]))

for ea,ec,ecc in ANGLE_TRIPLETS:
    lab = f"{ea}-{ec}-{ecc}"; all_a = []
    for atoms in frames:
        sy = atoms.get_chemical_symbols()
        ia,ja,da,Da = neighbor_list("ijdD", atoms, cutoff=ANGLE_CUTOFF)
        nb = defaultdict(list)
        for ii,jj,dd,dp in zip(ia,ja,da,Da): nb[ii].append((jj,dp,sy[jj]))
        for ci in range(len(sy)):
            if sy[ci] != ec: continue
            va = [d for _,d,s in nb[ci] if s == ea]
            vc = [d for _,d,s in nb[ci] if s == ecc]
            prs = ([(va[i],va[j]) for i in range(len(va)) for j in range(i+1,len(va))]
                   if ea == ecc else [(v1,v2) for v1 in va for v2 in vc])
            for v1,v2 in prs:
                n1,n2 = np.linalg.norm(v1),np.linalg.norm(v2)
                if n1<1e-10 or n2<1e-10: continue
                all_a.append(np.degrees(np.arccos(np.clip(np.dot(v1,v2)/(n1*n2),-1,1))))
    if all_a:
        all_a = np.array(all_a)
        hist,edges = np.histogram(all_a, bins=ANGLE_N_BINS, range=(0,180), density=True)
        cen = 0.5*(edges[:-1]+edges[1:])
        print(f"  {lab}: peak={cen[np.argmax(hist)]:.1f} deg, mean={np.mean(all_a):.1f}, n={len(all_a)}")
        np.savetxt(os.path.join(OUTPUT_DIR, f"bond_angle_{ea}_{ec}_{ecc}.dat"),
                   np.column_stack([cen,hist]), header="angle(deg) prob_density", fmt="%8.2f %12.6f")

print(f"\nDone. Output: {OUTPUT_DIR}/")
```

## Key Parameters

| Parameter | Default | Notes |
|---|---|---|
| `BOND_CUTOFFS` / `BL_PAIRS` | auto-detect | Distance range (r_min, r_max) per element pair. Auto-detection uses first neighbor shell. |
| `BL_N_BINS` | 200 | Histogram bins for bond length. 100-300 typical. Higher = finer but noisier. |
| `KDE_BANDWIDTH` | 0.02 A | Gaussian KDE bandwidth. 0 disables. Smaller = sharper peaks. |
| `ANGLE_BOND_CUTOFF` | 2.5 A | Max distance defining "bonded" for angle calc. Must cover first coordination shell. |
| `ANGLE_TRIPLETS` | auto-detect | (A, Center, C) tuples. Auto generates all plausible from bonded pairs. |
| `ANGLE_N_BINS` | 180 | Angle histogram bins. 180 gives 1-degree resolution. |
| `N_EQUIL_FRAMES` | 50-200 | Equilibration frames to discard. Check energy convergence first. |
| `FRAME_STEP` | 1 | Subsample factor. Use 2-5 for very long trajectories. |
| `AUTO_CUTOFF_FACTOR` | 1.3 | Multiplier on nearest-neighbor distance for auto upper cutoff. |

### Choosing ANGLE_BOND_CUTOFF

- Must encompass first coordination shell: check bond length distribution or RDF first.
- Too small: misses bonds, produces empty angle histograms.
- Too large: includes second-shell neighbors, spurious angles appear.
- Typical: 1.8-2.2 A for Si-O, 2.5-3.0 A for metal-O, 3.0-3.5 A for metal-metal.

## Interpreting Results

### Bond length distribution peaks
- **Sharp symmetric peak**: well-defined bond, crystalline or strongly-bonded system.
- **Broad/asymmetric peak**: thermal disorder, anharmonicity, or mixed environments.
- **Bimodal distribution**: two distinct bond lengths (Jahn-Teller, mixed oxidation states, coexisting phases).
- **Peak shift with T**: thermal expansion; compare distributions at different temperatures.

### Bond angle distribution peaks
- **Tetrahedral** (sp3): ~109.5 deg (O-Si-O in SiO2, O-C-O in diamond).
- **Octahedral**: ~90 and ~180 deg (O-Ti-O in rutile).
- **Trigonal planar** (sp2): ~120 deg (O-B-O in B2O3).
- **Linear** (sp): ~180 deg (O-C-O in CO2).
- **Broad distribution**: structural disorder, liquid-like environment.

### Comparison with crystalline reference
- Compute for both relaxed crystal (0 K) and MD trajectory at finite T.
- Broader peaks at higher T are expected from thermal vibrations.
- Peak position shift indicates anharmonic effects on equilibrium bond length.
- Loss of well-defined peaks signals amorphization or melting.

### Time evolution of mean bond length
- **Flat**: equilibrium reached, stable bonds.
- **Monotonic drift**: not equilibrated, or ongoing transformation.
- **Sudden jump**: bond breaking/forming, phase transition.
- **Oscillation**: coupled to pressure/volume fluctuations (NPT).

## Common Issues

| Issue | Cause | Fix |
|---|---|---|
| Empty bond length histogram | Cutoff range misses actual bonds | Auto-detect, or check RDF to find correct range |
| Bond angle histogram empty | `ANGLE_BOND_CUTOFF` too small | Increase to cover first coordination shell |
| Spurious angles near 0 or 180 deg | Second-shell atoms included | Reduce `ANGLE_BOND_CUTOFF` to first shell only |
| Noisy distributions | Too few frames or atoms | Longer trajectory, larger supercell, or fewer bins |
| KDE artifacts at boundaries | Kernel boundary effects | Use histogram instead, or widen range |
| Double-counting same-element pairs | Both (i,j) and (j,i) counted | Code uses `ii < jj` filter for A=B pairs |
| Memory error on large XDATCAR | All frames loaded at once | Use `FRAME_STEP > 1` or split XDATCAR |
| QE .pos parsed incorrectly | Header format differs between cp.x/pw.x | Inspect file; adjust parser column indices |
| Bond lengths differ from RDF peaks | Different normalization | Expected: bond hist is raw P(r); RDF divides by 4*pi*r^2*rho |
| XDATCAR missing element symbols | Old VASP4 format | Add element line to header or specify in pymatgen |
