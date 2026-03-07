# Topological Invariant Calculation

## When to Use

- **Classifying topological order**: Determine whether a material is a trivial insulator, topological insulator (TI), or topological semimetal
- **Z2 invariant**: Time-reversal invariant systems (Bi2Se3, Bi2Te3, Sb2Te3, HgTe)
- **Chern number**: Time-reversal broken systems (quantum anomalous Hall, magnetic TI)
- **Screening candidate materials**: Predict topological surface states before ARPES experiments
- **Band inversion analysis**: Confirm SOC-driven band inversion at TRIM points
- **Material design**: Engineer topological phase transitions via strain, alloying, or heterostructuring

**Classification of topological phases:**

| Z2 Invariant (nu0; nu1 nu2 nu3) | Classification | Surface States | Examples |
|---|---|---|---|
| (1; 000) | Strong TI | On ALL surfaces | Bi2Se3, Bi2Te3 |
| (0; nu_i=1) | Weak TI | On specific surfaces only | Bi14Rh3I9, stacked QSH |
| (0; 000) | Trivial insulator | None | NaCl, Si |
| Chern number C != 0 | Quantum anomalous Hall | Chiral edge states | Cr-doped (Bi,Sb)2Te3 |
| Weyl/Dirac points | Topological semimetal | Fermi arcs | TaAs, Cd3As2, Na3Bi |

## Method Selection

| Method | Applicability | Complexity | Accuracy |
|---|---|---|---|
| Wilson loop / HWCC (z2pack) | General -- any crystal | Moderate | High (gauge-invariant) |
| Parity at TRIM (Fu-Kane) | Inversion-symmetric only | Simple | Exact (if parities correct) |
| Berry phase (1D) | 2D systems, edge states | Moderate | High |
| Wannier charge centers | General | Moderate | High |
| Band inversion counting | Quick screening | Simple | Qualitative |

## Prerequisites

- Quantum ESPRESSO 7.5 (`pw.x`, `pw2wannier90.x`)
- Wannier90 (bundled with QE or standalone)
- Python packages: `z2pack` (`pip install z2pack`), `numpy`, `matplotlib`
- Fully relativistic pseudopotentials with SOC support (FR-ONCV or PSlibrary `_rel`)
- Crystal structure of the candidate material
- Converged SOC band structure showing a bulk gap

## Detailed Steps

### Method 1: Wilson Loop via z2pack (General Method)

The Wilson loop method tracks hybrid Wannier charge centers (HWCC) across the Brillouin zone. If the HWCC wind an odd number of times across a reference line, Z2 = 1. This method is gauge-invariant and works for any crystal (no symmetry requirement).

#### Step 1: SCF with SOC (Bi2Se3 Example)

```python
#!/usr/bin/env python3
"""
Complete Z2 invariant workflow for Bi2Se3 using QE + z2pack.
Bi2Se3 is a known strong topological insulator with Z2 = (1;000).
"""
import os
import numpy as np

WORK_DIR = os.path.abspath("z2_bi2se3")
os.makedirs(WORK_DIR, exist_ok=True)

# Bi2Se3 rhombohedral structure (R-3m, #166)
a_rhomb = 9.841  # Bohr
cos_alpha = np.cos(np.radians(24.304))
bi_mu = 0.4006
se_nu = 0.2060

PP_BI = "Bi.rel-pbe-dn-kjpaw_psl.1.0.0.UPF"
PP_SE = "Se.rel-pbe-dn-kjpaw_psl.1.0.0.UPF"
PSEUDO_DIR = os.environ.get("PSEUDO_DIR", "./pseudo")

scf_input = f"""&CONTROL
  calculation = 'scf'
  prefix = 'bi2se3'
  outdir = './tmp'
  pseudo_dir = '{PSEUDO_DIR}'
  verbosity = 'high'
/
&SYSTEM
  ibrav = 5
  celldm(1) = {a_rhomb}
  celldm(4) = {cos_alpha:.6f}
  nat = 5
  ntyp = 2
  ecutwfc = 50.0
  ecutrho = 400.0
  occupations = 'smearing'
  smearing = 'cold'
  degauss = 0.005
  noncolin = .true.
  lspinorb = .true.
/
&ELECTRONS
  conv_thr = 1.0d-10
  mixing_beta = 0.3
/
ATOMIC_SPECIES
  Bi  208.98040  {PP_BI}
  Se   78.96000  {PP_SE}
ATOMIC_POSITIONS {{crystal}}
  Bi  {bi_mu:.6f}  {bi_mu:.6f}  {bi_mu:.6f}
  Bi  {1-bi_mu:.6f}  {1-bi_mu:.6f}  {1-bi_mu:.6f}
  Se  {se_nu:.6f}  {se_nu:.6f}  {se_nu:.6f}
  Se  {1-se_nu:.6f}  {1-se_nu:.6f}  {1-se_nu:.6f}
  Se  0.000000  0.000000  0.000000
K_POINTS {{automatic}}
  8 8 8 0 0 0
"""

with open(os.path.join(WORK_DIR, "scf.in"), "w") as f:
    f.write(scf_input)
print("SCF input written.")
```

```bash
cd z2_bi2se3
mpirun -np 4 pw.x -in scf.in > scf.out 2>&1
grep "convergence has been achieved" scf.out
```

#### Step 2: Set Up z2pack with QE

```python
#!/usr/bin/env python3
"""
z2pack workflow: compute Z2 invariant for Bi2Se3.
z2pack drives QE NSCF + Wannier90 iteratively to compute Wilson loops.
"""
import os
import numpy as np
import z2pack
import matplotlib.pyplot as plt

WORK_DIR = os.path.abspath("z2_bi2se3")

PP_BI = "Bi.rel-pbe-dn-kjpaw_psl.1.0.0.UPF"
PP_SE = "Se.rel-pbe-dn-kjpaw_psl.1.0.0.UPF"
PSEUDO_DIR = os.environ.get("PSEUDO_DIR", "./pseudo")

a_rhomb = 9.841
cos_alpha = np.cos(np.radians(24.304))
bi_mu = 0.4006
se_nu = 0.2060

# --- NSCF template (z2pack replaces K_POINTS) ---
nscf_input = f"""&CONTROL
  calculation = 'nscf'
  prefix = 'bi2se3'
  outdir = './tmp'
  pseudo_dir = '{PSEUDO_DIR}'
  verbosity = 'high'
/
&SYSTEM
  ibrav = 5
  celldm(1) = {a_rhomb}
  celldm(4) = {cos_alpha:.6f}
  nat = 5
  ntyp = 2
  ecutwfc = 50.0
  ecutrho = 400.0
  occupations = 'smearing'
  smearing = 'cold'
  degauss = 0.005
  noncolin = .true.
  lspinorb = .true.
  nosym = .true.
  nbnd = 36
/
&ELECTRONS
  conv_thr = 1.0d-10
  mixing_beta = 0.3
  diago_full_acc = .true.
/
ATOMIC_SPECIES
  Bi  208.98040  {PP_BI}
  Se   78.96000  {PP_SE}
ATOMIC_POSITIONS {{crystal}}
  Bi  {bi_mu:.6f}  {bi_mu:.6f}  {bi_mu:.6f}
  Bi  {1-bi_mu:.6f}  {1-bi_mu:.6f}  {1-bi_mu:.6f}
  Se  {se_nu:.6f}  {se_nu:.6f}  {se_nu:.6f}
  Se  {1-se_nu:.6f}  {1-se_nu:.6f}  {1-se_nu:.6f}
  Se  0.000000  0.000000  0.000000
"""

with open(os.path.join(WORK_DIR, "nscf.in"), "w") as f:
    f.write(nscf_input)

# --- Wannier90 input ---
num_wann = 14   # Must be even for spinor
num_bands = 36

wannier_input = f"""num_wann = {num_wann}
num_bands = {num_bands}

exclude_bands = 1-8

begin projections
Bi: p
Se: p
end projections

write_mmn = .true.
write_amn = .true.
write_eig = .true.

spinors = .true.
"""

with open(os.path.join(WORK_DIR, "bi2se3.win"), "w") as f:
    f.write(wannier_input)

# --- pw2wannier90 input ---
pw2wan_input = """&inputpp
  outdir = './tmp'
  prefix = 'bi2se3'
  seedname = 'bi2se3'
  spin_component = 'none'
  write_mmn = .true.
  write_amn = .true.
  write_unk = .false.
/
"""

with open(os.path.join(WORK_DIR, "pw2wan.in"), "w") as f:
    f.write(pw2wan_input)

# --- Build z2pack system ---
qe_system = z2pack.fp.System(
    input_files=[
        os.path.join(WORK_DIR, "nscf.in"),
        os.path.join(WORK_DIR, "bi2se3.win"),
        os.path.join(WORK_DIR, "pw2wan.in"),
    ],
    kpt_fct=z2pack.fp.kpoint.qe_explicit,
    kpt_path="nscf.in",
    command=(
        "cd {work_dir} && "
        "mpirun -np 4 pw.x -in nscf.in > nscf.out 2>&1 && "
        "wannier90.x -pp bi2se3 > wan_pp.out 2>&1 && "
        "mpirun -np 4 pw2wannier90.x -in pw2wan.in > pw2wan.out 2>&1"
    ),
    executable="",
    mmn_path="bi2se3.mmn",
    build_folder=os.path.join(WORK_DIR, "z2pack_build"),
)

# --- Compute Z2 on six TRI planes ---
# For 3D Z2 = (nu0; nu1 nu2 nu3):
# TRI planes: k_i = 0 and k_i = 0.5 for i = 1,2,3
surfaces = {
    "k3_0":   lambda s, t: [s, t, 0.0],
    "k3_0.5": lambda s, t: [s, t, 0.5],
    "k2_0":   lambda s, t: [s, 0.0, t],
    "k2_0.5": lambda s, t: [s, 0.5, t],
    "k1_0":   lambda s, t: [0.0, s, t],
    "k1_0.5": lambda s, t: [0.5, s, t],
}

results = {}
for name, surface in surfaces.items():
    print(f"\n{'='*60}")
    print(f"Computing Wilson loop on surface: {name}")
    print(f"{'='*60}")

    result = z2pack.surface.run(
        system=qe_system,
        surface=surface,
        num_lines=11,
        min_neighbour_dist=0.01,
        iterator=range(8, 27, 2),
        pos_tol=0.01,
        gap_tol=0.3,
        move_tol=0.3,
        save_file=os.path.join(WORK_DIR, f"result_{name}.json"),
    )
    results[name] = result
    z2_val = z2pack.invariant.z2(result)
    print(f"Z2 invariant on {name}: {z2_val}")

# --- Compute full 3D Z2 invariant ---
z2_k1_0  = z2pack.invariant.z2(results["k1_0"])
z2_k1_05 = z2pack.invariant.z2(results["k1_0.5"])
z2_k2_0  = z2pack.invariant.z2(results["k2_0"])
z2_k3_0  = z2pack.invariant.z2(results["k3_0"])

nu0 = (z2_k1_0 + z2_k1_05) % 2  # Strong invariant
nu1 = z2_k1_0
nu2 = z2_k2_0
nu3 = z2_k3_0

print(f"\n{'='*60}")
print(f"Z2 Topological Invariant: ({nu0}; {nu1}{nu2}{nu3})")
print(f"{'='*60}")
if nu0 == 1:
    print("Result: STRONG topological insulator")
elif nu1 or nu2 or nu3:
    print("Result: WEAK topological insulator")
else:
    print("Result: Trivial insulator")

# --- Plot Wilson loop ---
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
for ax, (name, result) in zip(axes, [("k3_0", results["k3_0"]),
                                       ("k3_0.5", results["k3_0.5"])]):
    z2pack.plot.wcc(result, axis=ax)
    z2_val = z2pack.invariant.z2(result)
    ax.set_title(f"{name} plane, Z2 = {z2_val}")
    ax.set_xlabel("$k_1$")
    ax.set_ylabel("WCC ($\\bar{x}$)")

fig.suptitle("Bi$_2$Se$_3$ Wilson Loop (Hybrid Wannier Centers)")
plt.tight_layout()
plt.savefig(os.path.join(WORK_DIR, "wilson_loop_bi2se3.png"), dpi=150)
plt.close()
print(f"\nWilson loop plot saved.")
```

### Method 2: Parity Analysis at TRIM Points (Fu-Kane Formula)

For inversion-symmetric crystals, Z2 can be determined from parity eigenvalues of occupied bands at TRIM points. This is simpler than the Wilson loop method but only works when inversion symmetry is present.

**Theory**: (-1)^nu0 = product over all TRIM_i of delta_i, where delta_i = product over occupied Kramers pairs m of xi_{2m}(TRIM_i), and xi_{2m} is the parity eigenvalue.

```python
#!/usr/bin/env python3
"""
Z2 from parity eigenvalues at TRIM points (Fu-Kane formula).
For inversion-symmetric crystals only.
"""
import os
import re
import numpy as np

WORK_DIR = os.path.abspath("z2_bi2se3_parity")
os.makedirs(WORK_DIR, exist_ok=True)

# 8 TRIM points for rhombohedral BZ (crystal coordinates)
TRIM_POINTS = {
    "Gamma": [0.0, 0.0, 0.0],
    "Z":     [0.5, 0.5, 0.5],
    "L1":    [0.5, 0.0, 0.0],
    "L2":    [0.0, 0.5, 0.0],
    "L3":    [0.0, 0.0, 0.5],
    "F1":    [0.5, 0.5, 0.0],
    "F2":    [0.0, 0.5, 0.5],
    "F3":    [0.5, 0.0, 0.5],
}


def parse_parity_from_qe(output_file, n_occupied_pairs):
    """
    Parse parity eigenvalues from QE output at TRIM points.

    QE prints symmetry representations at high-symmetry points when
    verbosity='high'. Even representations (g subscript) have parity +1,
    odd representations (u subscript) have parity -1.
    """
    parities = {}

    if not os.path.exists(output_file):
        return parities

    with open(output_file, 'r') as f:
        content = f.read()

    # Parse symmetry labels from QE output
    # Look for blocks like:
    #   k = 0.0000 0.0000 0.0000
    #   Band symmetry, D_3d ...
    #   e(  1) = ... eV  2  --> A_1g (even parity)
    #   e(  2) = ... eV  2  --> A_1u (odd parity)
    pattern = re.compile(
        r'k\s*=\s*([\d.\-]+)\s+([\d.\-]+)\s+([\d.\-]+).*?'
        r'Band symmetry.*?\n((?:\s+e\(.*\n)*)',
        re.DOTALL,
    )

    for match in pattern.findall(content):
        kx, ky, kz = float(match[0]), float(match[1]), float(match[2])
        k_vec = np.array([kx, ky, kz])

        trim_label = None
        for label, trim_k in TRIM_POINTS.items():
            if np.allclose(k_vec, trim_k, atol=1e-4):
                trim_label = label
                break
        if trim_label is None:
            continue

        parity_list = []
        for line in match[3].strip().split("\n"):
            if "g" in line.lower() and "u" not in line.lower():
                parity_list.append(+1)
            elif "u" in line.lower():
                parity_list.append(-1)

        parities[trim_label] = parity_list[:n_occupied_pairs]

    return parities


def fu_kane_z2(parities):
    """
    Compute Z2 from parity eigenvalues using the Fu-Kane formula.

    Returns (nu0, nu1, nu2, nu3).
    """
    deltas = {}
    for label, parity_list in parities.items():
        delta = 1
        for p in parity_list:
            delta *= p
        deltas[label] = delta
        print(f"  TRIM {label:6s}: delta = {delta:+d}  "
              f"(parities: {[f'{p:+d}' for p in parity_list]})")

    # Strong Z2: product over ALL 8 TRIM points
    product_all = 1
    for delta in deltas.values():
        product_all *= delta
    nu0 = 0 if product_all == 1 else 1

    # Weak Z2 indices: product over 4 TRIM with k_i = 0.5
    product_nu1 = (deltas.get("L1", 1) * deltas.get("F1", 1)
                   * deltas.get("F3", 1) * deltas.get("Z", 1))
    nu1 = 0 if product_nu1 == 1 else 1

    product_nu2 = (deltas.get("L2", 1) * deltas.get("F1", 1)
                   * deltas.get("F2", 1) * deltas.get("Z", 1))
    nu2 = 0 if product_nu2 == 1 else 1

    product_nu3 = (deltas.get("L3", 1) * deltas.get("F2", 1)
                   * deltas.get("F3", 1) * deltas.get("Z", 1))
    nu3 = 0 if product_nu3 == 1 else 1

    return (nu0, nu1, nu2, nu3)


# --- Known parity results for Bi2Se3 (validation) ---
print("=" * 60)
print("Fu-Kane Parity Analysis for Bi2Se3")
print("=" * 60)

# At Gamma, band inversion flips one parity eigenvalue
known_parities = {
    "Gamma": [+1, +1, +1, -1, +1, -1, +1],  # Inverted pair
    "Z":     [+1, +1, +1, +1, +1, -1, +1],
    "L1":    [+1, +1, +1, +1, +1, -1, +1],
    "L2":    [+1, +1, +1, +1, +1, -1, +1],
    "L3":    [+1, +1, +1, +1, +1, -1, +1],
    "F1":    [+1, +1, +1, +1, +1, -1, +1],
    "F2":    [+1, +1, +1, +1, +1, -1, +1],
    "F3":    [+1, +1, +1, +1, +1, -1, +1],
}

z2 = fu_kane_z2(known_parities)
print(f"\nZ2 = ({z2[0]}; {z2[1]}{z2[2]}{z2[3]})")
if z2[0] == 1:
    print("Strong topological insulator confirmed!")


# --- NSCF input targeting TRIM points only ---
def write_trim_nscf(work_dir, trim_points):
    """Write QE NSCF input for TRIM-point parity analysis."""
    nk = len(trim_points)
    kpt_block = f"K_POINTS {{crystal}}\n{nk}\n"
    for label, kpt in trim_points.items():
        kpt_block += f"  {kpt[0]:.8f} {kpt[1]:.8f} {kpt[2]:.8f}  1.0  ! {label}\n"

    cos_alpha = np.cos(np.radians(24.304))

    nscf_in = f"""&CONTROL
  calculation = 'nscf'
  prefix = 'bi2se3'
  outdir = './tmp'
  pseudo_dir = './pseudo'
  verbosity = 'high'
/
&SYSTEM
  ibrav = 5
  celldm(1) = 9.841
  celldm(4) = {cos_alpha:.6f}
  nat = 5
  ntyp = 2
  ecutwfc = 50.0
  ecutrho = 400.0
  occupations = 'smearing'
  smearing = 'cold'
  degauss = 0.005
  noncolin = .true.
  lspinorb = .true.
  nbnd = 36
/
&ELECTRONS
  conv_thr = 1.0d-10
  diago_full_acc = .true.
/
ATOMIC_SPECIES
  Bi  208.98040  Bi.rel-pbe-dn-kjpaw_psl.1.0.0.UPF
  Se   78.96000  Se.rel-pbe-dn-kjpaw_psl.1.0.0.UPF
ATOMIC_POSITIONS {{crystal}}
  Bi  0.400600  0.400600  0.400600
  Bi  0.599400  0.599400  0.599400
  Se  0.206000  0.206000  0.206000
  Se  0.794000  0.794000  0.794000
  Se  0.000000  0.000000  0.000000
{kpt_block}"""

    filepath = os.path.join(work_dir, "nscf_trim.in")
    with open(filepath, "w") as f:
        f.write(nscf_in)
    print(f"NSCF input for TRIM points written to {filepath}")


write_trim_nscf(WORK_DIR, TRIM_POINTS)
```

### Method 3: Tight-Binding Validation with z2pack (No DFT Needed)

Use this to validate the z2pack workflow before expensive DFT calculations. The BHZ model describes the HgTe/CdTe quantum well that hosts the quantum spin Hall effect.

```python
#!/usr/bin/env python3
"""
Quick Z2 test using z2pack with the BHZ tight-binding model.
No DFT required -- validates the z2pack workflow.
BHZ model: M < 0 -> topological (inverted), M > 0 -> trivial.
"""
import numpy as np
import z2pack
import matplotlib.pyplot as plt


def bhz_hamiltonian(k, A=0.3645, B=-0.686, C=0.0, D=-0.512, M=-0.01):
    """
    BHZ model Hamiltonian for HgTe/CdTe quantum well.
    4x4 matrix in basis |E1+>, |H1+>, |E1->, |H1->.
    Reference: Bernevig, Hughes, Zhang, Science 314, 1757 (2006).
    """
    kx, ky = k[0], k[1]
    k_sq = kx**2 + ky**2

    epsilon_k = C - D * k_sq
    d_z = M - B * k_sq
    d_plus = A * (kx + 1j * ky)
    d_minus = A * (kx - 1j * ky)

    H = np.zeros((4, 4), dtype=complex)
    # Upper block (spin up)
    H[0, 0] = epsilon_k + d_z
    H[0, 1] = d_plus
    H[1, 0] = d_minus
    H[1, 1] = epsilon_k - d_z
    # Lower block (spin down, TR partner)
    H[2, 2] = epsilon_k + d_z
    H[2, 3] = -d_minus
    H[3, 2] = -d_plus
    H[3, 3] = epsilon_k - d_z

    return H


# Topological phase (M < 0, inverted)
print("=" * 60)
print("BHZ Model - Topological Phase (M < 0)")
print("=" * 60)

system_topo = z2pack.hm.System(
    lambda k: bhz_hamiltonian([k[0] * 2 * np.pi, k[1] * 2 * np.pi], M=-0.01),
    dim=2, bands=2,
)

result_topo = z2pack.surface.run(
    system=system_topo,
    surface=lambda s, t: [s, t],
    num_lines=51,
    pos_tol=None, gap_tol=None, move_tol=None,
)
z2_topo = z2pack.invariant.z2(result_topo)
print(f"Z2 = {z2_topo}")

# Trivial phase (M > 0, normal)
print(f"\n{'='*60}")
print("BHZ Model - Trivial Phase (M > 0)")
print("=" * 60)

system_trivial = z2pack.hm.System(
    lambda k: bhz_hamiltonian([k[0] * 2 * np.pi, k[1] * 2 * np.pi], M=+0.01),
    dim=2, bands=2,
)

result_trivial = z2pack.surface.run(
    system=system_trivial,
    surface=lambda s, t: [s, t],
    num_lines=51,
    pos_tol=None, gap_tol=None, move_tol=None,
)
z2_trivial = z2pack.invariant.z2(result_trivial)
print(f"Z2 = {z2_trivial}")

# Plot Wilson loops
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

z2pack.plot.wcc(result_topo, axis=axes[0])
axes[0].set_title(f"Topological (M<0), Z2={z2_topo}")
axes[0].set_xlabel("$k_y / (2\\pi)$")
axes[0].set_ylabel("WCC ($\\bar{x}$)")

z2pack.plot.wcc(result_trivial, axis=axes[1])
axes[1].set_title(f"Trivial (M>0), Z2={z2_trivial}")
axes[1].set_xlabel("$k_y / (2\\pi)$")
axes[1].set_ylabel("WCC ($\\bar{x}$)")

plt.tight_layout()
plt.savefig("bhz_wilson_loops.png", dpi=150)
plt.close()
print(f"\nExpected: Topological Z2=1, Trivial Z2=0")
print(f"Got:      Topological Z2={z2_topo}, Trivial Z2={z2_trivial}")
```

### Method 4: Berry Phase Approach (2D Systems)

For 2D systems, the Z2 invariant can be computed from the Berry phase on time-reversal invariant lines in the BZ.

```python
#!/usr/bin/env python3
"""
Berry phase Z2 for 2D systems.
Compute the change in time-reversal polarization between k_y=0 and k_y=pi.
If the Berry phase difference is pi, Z2 = 1.
"""
import numpy as np


def berry_phase_on_line(eigenstates_func, ky, nk=100):
    """
    Compute Berry phase along k_x for a fixed k_y.

    Parameters
    ----------
    eigenstates_func : callable
        Returns occupied eigenstates as columns of a matrix for given (kx, ky).
    ky : float
        Fixed k_y value.
    nk : int
        Number of k-points along k_x.

    Returns
    -------
    float : Berry phase (mod 2*pi).
    """
    kx_values = np.linspace(0, 2 * np.pi, nk, endpoint=False)
    dk = kx_values[1] - kx_values[0]

    # Compute overlap product
    # phi = -Im(log(det(product of overlaps)))
    total_overlap = None

    for i in range(nk):
        kx1 = kx_values[i]
        kx2 = kx_values[(i + 1) % nk]

        psi1 = eigenstates_func(kx1, ky)  # (n_basis, n_occ)
        psi2 = eigenstates_func(kx2, ky)

        overlap = psi1.conj().T @ psi2  # (n_occ, n_occ)

        if total_overlap is None:
            total_overlap = overlap
        else:
            total_overlap = total_overlap @ overlap

    phase = -np.imag(np.log(np.linalg.det(total_overlap)))
    return phase % (2 * np.pi)


def z2_from_berry_phase(eigenstates_func, nk=100):
    """
    Compute 2D Z2 invariant from Berry phases at TRIM lines.

    Z2 = [theta(pi) - theta(0)] / pi  (mod 2)

    where theta(ky) is the Berry phase along k_x at fixed k_y.
    """
    theta_0 = berry_phase_on_line(eigenstates_func, ky=0.0, nk=nk)
    theta_pi = berry_phase_on_line(eigenstates_func, ky=np.pi, nk=nk)

    # Z2 = difference in Berry phase / pi (mod 2)
    delta_theta = theta_pi - theta_0
    z2 = int(np.round(delta_theta / np.pi)) % 2

    print(f"Berry phase at ky=0:  {theta_0:.4f} rad = {theta_0/np.pi:.4f} * pi")
    print(f"Berry phase at ky=pi: {theta_pi:.4f} rad = {theta_pi/np.pi:.4f} * pi")
    print(f"Delta theta:          {delta_theta:.4f} rad = {delta_theta/np.pi:.4f} * pi")
    print(f"Z2 = {z2}")

    return z2


# Example with BHZ model
def bhz_eigenstates(kx, ky, M=-0.01, A=0.3645, B=-0.686):
    """Return occupied eigenstates of BHZ model at (kx, ky)."""
    k_sq = kx**2 + ky**2
    d_z = M - B * k_sq
    d_plus = A * (kx + 1j * ky)
    d_minus = A * (kx - 1j * ky)

    H = np.zeros((4, 4), dtype=complex)
    H[0, 0] = d_z;     H[0, 1] = d_plus
    H[1, 0] = d_minus;  H[1, 1] = -d_z
    H[2, 2] = d_z;     H[2, 3] = -d_minus
    H[3, 2] = -d_plus;  H[3, 3] = -d_z

    eigvals, eigvecs = np.linalg.eigh(H)
    # Return the two occupied states (lowest eigenvalues)
    return eigvecs[:, :2]


print("2D Z2 from Berry phase (BHZ model, M<0):")
z2_result = z2_from_berry_phase(
    lambda kx, ky: bhz_eigenstates(kx, ky, M=-0.01), nk=200)
```

## Key Parameters

| Parameter | Typical Value | Notes |
|-----------|--------------|-------|
| `ecutwfc` | 50-80 Ry | Higher for heavier elements; test convergence |
| `ecutrho` | 400-640 Ry | 8x ecutwfc for PAW/USPP |
| k-grid (SCF) | 8x8x8 | Must be well-converged for reliable topology |
| `noncolin` | `.true.` | Required for SOC |
| `lspinorb` | `.true.` | Required for SOC |
| `nosym` | `.true.` | Required in NSCF for z2pack/Wannier90 |
| `num_wann` | System-dependent | Must be even (spinor); cover occupied manifold |
| z2pack `num_lines` | 11-51 | Odd values; more lines = better convergence |
| z2pack `pos_tol` | 0.01 | Tolerance for WCC positions |
| z2pack `gap_tol` | 0.3 | Gap detection tolerance in WCC spectrum |
| Pseudopotentials | FR (fully relativistic) | `_rel` (PSlibrary) or `_FR` (ONCVPSP) |
| `conv_thr` | 1.0d-10 | Tight convergence needed for topological properties |

## Interpreting Results

### Z2 Classification

**Z2 = (1; 000):** Strong topological insulator. Robust metallic surface states on ALL surfaces. Protected by time-reversal symmetry. Cannot be removed by disorder (only by closing the bulk gap). Examples: Bi2Se3, Bi2Te3.

**Z2 = (0; nu_i=1):** Weak topological insulator. Surface states exist on certain surfaces only. Can be localized by disorder. Equivalent to stacked 2D TI layers. Example: Bi14Rh3I9.

**Z2 = (0; 000):** Trivial insulator. No topologically protected surface states.

### Wilson Loop Interpretation

- Plot WCC (Wannier charge centers) vs. k along one direction
- Draw any horizontal reference line at arbitrary y-value
- If reference line crosses WCC lines an **ODD** number of times: Z2 = 1 (topological)
- If reference line crosses an **EVEN** number of times: Z2 = 0 (trivial)
- The WCC lines show a characteristic "partner switching" pattern for Z2 = 1

### Parity Method Checklist

1. Confirm crystal has inversion symmetry
2. Identify the 8 TRIM points for your lattice type
3. Compute parity eigenvalues for all occupied Kramers pairs at each TRIM
4. delta_i = product of parities at TRIM_i
5. (-1)^nu0 = product of all delta_i
6. If delta_i = -1 at an odd number of TRIM points, you have Z2 = 1

## Common Issues

| Issue | Symptom | Solution |
|-------|---------|----------|
| SCF does not converge with SOC | Energy oscillates | Reduce `mixing_beta` to 0.1-0.2; start from non-SOC solution |
| z2pack convergence | WCC positions unstable | Increase `num_lines` (21, 31, 51); decrease tolerances |
| Wrong Z2 value | Known TI gives Z2=0 | Verify FR pseudopotentials; check `noncolin` AND `lspinorb`; verify gap is correct |
| Parity method gives wrong result | Only valid for centrosymmetric crystals | Switch to Wilson loop method if no inversion symmetry |
| FR pseudopotential errors | QE crashes at startup | Use `_rel` PPs from PSlibrary; all PPs must be FR when `lspinorb=.true.` |
| Gap closes on integration surface | z2pack warns about gap closure | Choose a different surface or verify material actually has a gap |
| Wannier90 disentanglement fails | Poor spread convergence | Adjust frozen/disentanglement windows; increase `num_iter` |
| Memory issues | Out of memory | SOC doubles bands; use k-point parallelization (`-nk`) |
| Nearly degenerate bands at TRIM | Ambiguous parity assignment | Use Wilson loop method instead; degeneracy needs careful tracking |
| Topological semimetal misidentified | Gap closes at some k-points | Not a TI -- material may be a Weyl or Dirac semimetal; compute Chern number instead |
