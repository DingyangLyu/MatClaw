# Angular-Dependent Mechanical Properties

## When to Use

- You have a computed 6x6 elastic stiffness tensor (C_ij in Voigt notation) and want to visualize directional dependence of mechanical properties.
- You need 3D polar plots of direction-dependent Young's modulus E(theta,phi), shear modulus G(theta,phi,chi), Poisson's ratio nu(theta,phi,chi), or linear compressibility beta(theta,phi).
- You want to assess elastic anisotropy quantitatively (universal elastic anisotropy index, Zener ratio, etc.).
- You need Vickers hardness estimates from elastic constants (Chen model, Tian model).
- You want Cauchy pressure analysis to assess ductile vs. brittle character.
- You are working with 2D materials and need in-plane angular mechanical properties from C_ij^2D.
- You want publication-quality directional mechanical property plots (inspired by VASPKIT task 204 / ELATE / MechElastic).

## Method Selection

| Task | This Skill | elastic-constants/ | equation-of-state/ |
|---|---|---|---|
| Compute C_ij from scratch | No -- use elastic-constants first | Yes | No |
| Directional E, G, nu, beta plots | Yes | No | No |
| Hardness estimation from C_ij | Yes | No | No |
| Anisotropy indices (A_U, A_Z) | Yes | Zener only (cubic) | No |
| 2D mechanical properties | Yes | No | No |
| Bulk/shear modulus (VRH) | Derived from C_ij | Yes | B0 from EOS |
| Cauchy pressure / ductility | Yes | Pugh ratio only | No |

## Prerequisites

- A 6x6 elastic stiffness tensor C_ij in GPa (Voigt notation). Obtain from:
  - The `elastic-constants/` skill (MACE or QE workflow, saved as `elastic_results.json`).
  - Literature values.
  - Materials Project API (`mp-api`, returns `ElasticTensor` objects).
- Python packages: `numpy`, `scipy`, `matplotlib` (pre-installed).
- Optional: `pymatgen` (for `ElasticTensor` class utilities).

---

## Detailed Steps

### Step 1: Angular-Dependent Properties for 3D Crystals

Given C_ij (6x6), compute the compliance tensor S_ij = C_ij^{-1}, then evaluate direction-dependent properties on a spherical mesh (theta, phi). The formulas follow the Nye/Hearmon framework used by ELATE and VASPKIT 204.

```python
#!/usr/bin/env python3
"""
Angular-dependent mechanical properties from the elastic tensor.

Given C_ij (6x6 Voigt), computes and plots:
  - Young's modulus E(theta, phi)
  - Linear compressibility beta(theta, phi)
  - Shear modulus G(theta, phi, chi)   [min/max over chi]
  - Poisson's ratio nu(theta, phi, chi) [min/max over chi]

Also computes:
  - Universal elastic anisotropy index A_U
  - Zener anisotropy ratio (cubic systems)
  - Vickers hardness (Chen, Tian models)
  - Cauchy pressure analysis

References:
  - Nye, "Physical Properties of Crystals" (1957)
  - Ranganathan & Ostoja-Starzewski, PRL 101, 055504 (2008)  [A_U]
  - Chen et al., Intermetallics 19, 1275 (2011)  [Hardness]
  - Tian et al., Int. J. Refract. Met. Hard Mater. 33, 93 (2012) [Hardness]
  - Mouhat & Coudert, Phys. Rev. B 90, 224104 (2014)  [Stability]
  - Gaillac, Pullumbi & Coudert, J. Phys.: Condens. Matter 28, 275201 (2016) [ELATE]
"""

import json
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm

# ─── CONFIGURATION ───────────────────────────────────────────────────────────
# Option 1: Load C_ij from elastic_results.json (produced by elastic-constants skill)
ELASTIC_JSON = "elastic_results/elastic_results.json"  # or None to use manual input

# Option 2: Enter C_ij manually (6x6, GPa, Voigt notation)
# Example: silicon (cubic, C11=166, C12=64, C44=80 GPa)
C_IJ_MANUAL = np.array([
    [166.0,  64.0,  64.0,   0.0,   0.0,   0.0],
    [ 64.0, 166.0,  64.0,   0.0,   0.0,   0.0],
    [ 64.0,  64.0, 166.0,   0.0,   0.0,   0.0],
    [  0.0,   0.0,   0.0,  80.0,   0.0,   0.0],
    [  0.0,   0.0,   0.0,   0.0,  80.0,   0.0],
    [  0.0,   0.0,   0.0,   0.0,   0.0,  80.0],
])

FORMULA = "Si"              # Label for plots
N_THETA = 90                # Angular resolution (theta: 0 to pi)
N_PHI = 180                 # Angular resolution (phi: 0 to 2*pi)
N_CHI = 72                  # Angular resolution for chi sweep (shear/Poisson)
OUTPUT_DIR = "angular_mechanics_results"
# ─────────────────────────────────────────────────────────────────────────────

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ─── Load elastic tensor ─────────────────────────────────────────────────────
if ELASTIC_JSON and os.path.exists(ELASTIC_JSON):
    with open(ELASTIC_JSON) as f:
        data = json.load(f)
    C = np.array(data["elastic_tensor_GPa"])
    FORMULA = data.get("formula", FORMULA)
    print(f"Loaded C_ij from {ELASTIC_JSON} ({FORMULA})")
else:
    C = C_IJ_MANUAL.copy()
    print(f"Using manual C_ij for {FORMULA}")

# Symmetrize C_ij
C = 0.5 * (C + C.T)
print(f"\nElastic tensor C_ij (GPa):")
for i in range(6):
    print("  " + "  ".join(f"{C[i,j]:8.2f}" for j in range(6)))

# Check positive definiteness (Born stability)
eigenvalues = np.linalg.eigvalsh(C)
print(f"\nEigenvalues of C_ij: {', '.join(f'{ev:.2f}' for ev in eigenvalues)}")
if any(ev <= 0 for ev in eigenvalues):
    print("WARNING: C_ij is NOT positive definite. Structure is mechanically unstable.")
    print("Angular properties may be meaningless for unstable phases.")

# Compliance tensor S = C^{-1}
S = np.linalg.inv(C)
print(f"\nCompliance tensor S_ij (1/GPa):")
for i in range(6):
    print("  " + "  ".join(f"{S[i,j]:10.6f}" for j in range(6)))


# ─── Voigt-Reuss-Hill moduli (needed for anisotropy indices) ─────────────────
def voigt_reuss_hill(C, S):
    """Compute VRH polycrystalline moduli from C_ij and S_ij."""
    # Voigt averages
    K_V = ((C[0,0] + C[1,1] + C[2,2]) + 2*(C[0,1] + C[0,2] + C[1,2])) / 9.0
    G_V = ((C[0,0] + C[1,1] + C[2,2]) - (C[0,1] + C[0,2] + C[1,2])
           + 3*(C[3,3] + C[4,4] + C[5,5])) / 15.0

    # Reuss averages
    K_R = 1.0 / ((S[0,0] + S[1,1] + S[2,2]) + 2*(S[0,1] + S[0,2] + S[1,2]))
    G_R = 15.0 / (4*(S[0,0] + S[1,1] + S[2,2]) - 4*(S[0,1] + S[0,2] + S[1,2])
                  + 3*(S[3,3] + S[4,4] + S[5,5]))

    # Hill (VRH) averages
    K_H = 0.5 * (K_V + K_R)
    G_H = 0.5 * (G_V + G_R)

    E_H = 9*K_H*G_H / (3*K_H + G_H) if (3*K_H + G_H) > 0 else 0.0
    nu_H = (3*K_H - 2*G_H) / (6*K_H + 2*G_H) if (6*K_H + 2*G_H) > 0 else 0.0

    return {
        "K_Voigt": K_V, "K_Reuss": K_R, "K_VRH": K_H,
        "G_Voigt": G_V, "G_Reuss": G_R, "G_VRH": G_H,
        "E_VRH": E_H, "nu_VRH": nu_H,
    }


moduli = voigt_reuss_hill(C, S)
K_V, K_R, K_H = moduli["K_Voigt"], moduli["K_Reuss"], moduli["K_VRH"]
G_V, G_R, G_H = moduli["G_Voigt"], moduli["G_Reuss"], moduli["G_VRH"]

print(f"\nPolycrystalline Moduli (GPa):")
print(f"  K: Voigt={K_V:.2f}, Reuss={K_R:.2f}, VRH={K_H:.2f}")
print(f"  G: Voigt={G_V:.2f}, Reuss={G_R:.2f}, VRH={G_H:.2f}")
print(f"  E (VRH) = {moduli['E_VRH']:.2f} GPa")
print(f"  nu (VRH) = {moduli['nu_VRH']:.4f}")


# ─── Anisotropy indices ──────────────────────────────────────────────────────
# Universal elastic anisotropy index (Ranganathan & Ostoja-Starzewski, 2008)
# A_U = 5 * G_V/G_R + K_V/K_R - 6 >= 0   (0 = isotropic)
A_U = 5.0 * G_V / G_R + K_V / K_R - 6.0 if G_R > 0 and K_R > 0 else float('inf')
print(f"\nAnisotropy Indices:")
print(f"  Universal elastic anisotropy A_U = {A_U:.4f}  (0 = isotropic)")

# Log-Euclidean anisotropy index
A_L = np.sqrt(
    np.log(K_V/K_R)**2 + 5*np.log(G_V/G_R)**2
) if K_R > 0 and G_R > 0 else float('inf')
print(f"  Log-Euclidean anisotropy A_L = {A_L:.4f}")

# Zener ratio (cubic systems only)
# A_Z = 2*C44 / (C11 - C12)
c11, c12, c44 = C[0,0], C[0,1], C[3,3]
if abs(c11 - c12) > 1e-6:
    A_Z = 2.0 * c44 / (c11 - c12)
    print(f"  Zener ratio A_Z = {A_Z:.4f}  (1.0 = isotropic, cubic only)")


# ─── Cauchy pressure analysis ────────────────────────────────────────────────
# Cauchy pressures: C12-C44, C13-C55, C23-C66 for cubic/general crystals
# Positive Cauchy pressure -> metallic/ionic bonding, ductile
# Negative Cauchy pressure -> covalent/directional bonding, brittle
# (Pettifor, Mater. Sci. Technol. 8, 345, 1992)

cauchy_12_44 = C[0,1] - C[3,3]
cauchy_13_55 = C[0,2] - C[4,4]
cauchy_23_66 = C[1,2] - C[5,5]

print(f"\nCauchy Pressure Analysis:")
print(f"  C12 - C44 = {cauchy_12_44:.2f} GPa  "
      f"({'ductile/ionic' if cauchy_12_44 > 0 else 'brittle/covalent'})")
print(f"  C13 - C55 = {cauchy_13_55:.2f} GPa  "
      f"({'ductile/ionic' if cauchy_13_55 > 0 else 'brittle/covalent'})")
print(f"  C23 - C66 = {cauchy_23_66:.2f} GPa  "
      f"({'ductile/ionic' if cauchy_23_66 > 0 else 'brittle/covalent'})")

# Pugh's ratio
pugh = K_H / G_H if G_H > 0 else float('inf')
print(f"\n  Pugh's ratio K/G = {pugh:.3f}  "
      f"({'ductile (>1.75)' if pugh > 1.75 else 'brittle (<1.75)'})")

# Frantsevich ratio (Poisson)
nu_H = moduli["nu_VRH"]
print(f"  Poisson's ratio nu = {nu_H:.4f}  "
      f"({'ductile (>0.26)' if nu_H > 0.26 else 'brittle (<0.26)'})")


# ─── Vickers hardness estimation ─────────────────────────────────────────────
print(f"\nVickers Hardness Estimates:")

# Chen et al., Intermetallics 19, 1275 (2011):
#   H_V = 2 * (k^2 * G)^0.585 - 3    where k = G/K
k_ratio = G_H / K_H if K_H > 0 else 0
H_chen = 2.0 * (k_ratio**2 * G_H)**0.585 - 3.0
print(f"  Chen model:  H_V = {H_chen:.2f} GPa")
print(f"    Formula: H_V = 2*(k^2*G)^0.585 - 3,  k = G/K = {k_ratio:.4f}")

# Tian et al., Int. J. Refract. Met. Hard Mater. 33, 93 (2012):
#   H_V = 0.92 * (K/G)^1.137 * G^0.708
if K_H > 0 and G_H > 0:
    H_tian = 0.92 * (K_H/G_H)**1.137 * G_H**0.708
    print(f"  Tian model:  H_V = {H_tian:.2f} GPa")
    print(f"    Formula: H_V = 0.92*(K/G)^1.137 * G^0.708")
else:
    H_tian = 0.0
    print(f"  Tian model:  H_V = N/A (K or G <= 0)")

# Classification
avg_H = 0.5 * (H_chen + H_tian)
if avg_H > 40:
    hardness_class = "superhard"
elif avg_H > 20:
    hardness_class = "hard"
elif avg_H > 10:
    hardness_class = "medium"
else:
    hardness_class = "soft"
print(f"  Average: {avg_H:.2f} GPa ({hardness_class})")


# ─── Direction-dependent property functions ──────────────────────────────────

def direction_cosines(theta, phi):
    """Unit direction vector from spherical angles."""
    l1 = np.sin(theta) * np.cos(phi)
    l2 = np.sin(theta) * np.sin(phi)
    l3 = np.cos(theta)
    return l1, l2, l3


def youngs_modulus(l1, l2, l3, S):
    """
    Direction-dependent Young's modulus E(n).
    E^{-1} = sum_{ijkl} S_{ijkl} * n_i * n_j * n_k * n_l
    In Voigt: E^{-1} = S11*l1^4 + S22*l2^4 + S33*l3^4
              + (2*S12 + S66)*l1^2*l2^2
              + (2*S13 + S55)*l1^2*l3^2
              + (2*S23 + S44)*l2^2*l3^2
              + 2*(S14 + S56)*l1^2*l2*l3
              + 2*(S25 + S46)*l1*l2^2*l3
              + 2*(S36 + S45)*l1*l2*l3^2
              + 2*S15*l1^3*l3 + 2*S16*l1^3*l2
              + 2*S24*l2^3*l3 + 2*S26*l1*l2^3
              + 2*S34*l2*l3^3 + 2*S35*l1*l3^3
    """
    # Build the full 4th rank compliance tensor approach via direct summation
    # Map Voigt indices to pairs
    voigt_map = {0: (0,0), 1: (1,1), 2: (2,2), 3: (1,2), 4: (0,2), 5: (0,1)}
    n = np.array([l1, l2, l3])

    # Convert Voigt S to full S_ijkl
    inv_E = 0.0
    for p in range(6):
        for q in range(6):
            i, j = voigt_map[p]
            k, l = voigt_map[q]
            # Voigt factor: 1 for p<3, 2 for p>=3 (engineering shear)
            # S_ijkl = S_pq * factor_p * factor_q
            # where factor_p = 1 if p<3 else 0.5 (for compliance)
            # But for the formula E^{-1} = S_{ijkl} n_i n_j n_k n_l
            # with proper symmetrization:
            fp = 1.0 if p < 3 else 0.5
            fq = 1.0 if q < 3 else 0.5
            # Multiplicity: how many distinct (i,j) pairs map to Voigt index p
            mp = 1 if p < 3 else 2
            mq = 1 if q < 3 else 2
            inv_E += fp * fq * mp * mq * S[p, q] * n[i]*n[j]*n[k]*n[l]

    return 1.0 / inv_E if abs(inv_E) > 1e-30 else 0.0


def linear_compressibility(l1, l2, l3, S):
    """
    Direction-dependent linear compressibility beta(n).
    beta = sum_{ij} S_{iijj_contracted} * n_i * n_j
         = (S11+S12+S13)*l1^2 + (S12+S22+S23)*l2^2 + (S13+S23+S33)*l3^2
           + (S14+S24+S34)*2*l2*l3 + (S15+S25+S35)*2*l1*l3 + (S16+S26+S36)*2*l1*l2
    Units: 1/GPa (= TPa^{-1} * 1e-3)
    """
    n = np.array([l1, l2, l3])
    beta = 0.0
    for i in range(3):
        for j in range(3):
            # Sum S_{iikk} for k=0,1,2 -> S_{Voigt(ii), Voigt(jj)} = S[i,j] (for i,j<3)
            s_sum = S[i, 0] + S[i, 1] + S[i, 2]
            beta += s_sum * n[i] * n[j]  # This double counts, simplify below

    # Correct formula:
    # beta(n) = sum_i sum_j S_{ijkk->contracted} n_i n_j
    # = sum_i n_i^2 * (S[i,0]+S[i,1]+S[i,2])  +  cross terms for shear
    beta = 0.0
    for i in range(3):
        s_row_sum = S[i, 0] + S[i, 1] + S[i, 2]
        beta += s_row_sum * n[i]**2

    # Off-diagonal (shear) contributions
    # Voigt index 3 -> (1,2), 4 -> (0,2), 5 -> (0,1)
    s_yz_sum = S[3, 0] + S[3, 1] + S[3, 2]  # S_{4k} in 1-indexed
    s_xz_sum = S[4, 0] + S[4, 1] + S[4, 2]
    s_xy_sum = S[5, 0] + S[5, 1] + S[5, 2]
    beta += 2 * s_yz_sum * n[1] * n[2]
    beta += 2 * s_xz_sum * n[0] * n[2]
    beta += 2 * s_xy_sum * n[0] * n[1]

    return beta


def shear_modulus_minmax(l1, l2, l3, S, n_chi=72):
    """
    Shear modulus G(n, m) depends on both the loading direction n and the
    shear plane normal m (perpendicular to n). Sweep chi to find min/max G.

    G^{-1}(n,m) = sum_{ijkl} S_{ijkl} * (n_i*m_j*n_k*m_l + ...)
    In practice: 1/G = 4 * S_{ijkl} * n_i * m_j * n_k * m_l

    Returns (G_min, G_max).
    """
    n = np.array([l1, l2, l3])

    # Find two vectors perpendicular to n
    if abs(n[0]) < 0.9:
        ref = np.array([1.0, 0.0, 0.0])
    else:
        ref = np.array([0.0, 1.0, 0.0])
    e1 = np.cross(n, ref)
    e1 /= np.linalg.norm(e1)
    e2 = np.cross(n, e1)
    e2 /= np.linalg.norm(e2)

    voigt_map = {0: (0,0), 1: (1,1), 2: (2,2), 3: (1,2), 4: (0,2), 5: (0,1)}

    G_values = []
    for chi in np.linspace(0, np.pi, n_chi, endpoint=False):
        m = np.cos(chi) * e1 + np.sin(chi) * e2

        # 1/G = sum over full S_{ijkl} * n_i * m_j * n_k * m_l (factor 4 in Voigt)
        inv_G = 0.0
        for p in range(6):
            for q in range(6):
                i, j = voigt_map[p]
                k, l = voigt_map[q]
                fp = 1.0 if p < 3 else 0.5
                fq = 1.0 if q < 3 else 0.5
                mp = 1 if p < 3 else 2
                mq = 1 if q < 3 else 2
                # The shear compliance: S_{ijkl} n_i m_j n_k m_l
                # With symmetry: average over (ijkl), (jikl), (ijlk), (jilk)
                term = fp * fq * mp * mq * S[p, q]
                term *= (n[i]*m[j]*n[k]*m[l])
                inv_G += term

        # The factor of 4 comes from the symmetrization of the shear formula
        inv_G *= 4.0

        if abs(inv_G) > 1e-30:
            G_values.append(1.0 / inv_G)

    if G_values:
        return min(G_values), max(G_values)
    return 0.0, 0.0


def poisson_ratio_minmax(l1, l2, l3, S, n_chi=72):
    """
    Poisson's ratio nu(n, m) = -S_{ijkl}*n_i*n_j*m_k*m_l / (S_{pqrs}*n_p*n_q*n_r*n_s)
    Sweep chi to find min/max nu.

    Returns (nu_min, nu_max).
    """
    n = np.array([l1, l2, l3])

    # Find perpendicular vectors
    if abs(n[0]) < 0.9:
        ref = np.array([1.0, 0.0, 0.0])
    else:
        ref = np.array([0.0, 1.0, 0.0])
    e1 = np.cross(n, ref)
    e1 /= np.linalg.norm(e1)
    e2 = np.cross(n, e1)
    e2 /= np.linalg.norm(e2)

    voigt_map = {0: (0,0), 1: (1,1), 2: (2,2), 3: (1,2), 4: (0,2), 5: (0,1)}

    # Denominator: S_{ijkl} n_i n_j n_k n_l = 1/E(n) -- precompute
    inv_E = 0.0
    for p in range(6):
        for q in range(6):
            i, j = voigt_map[p]
            k, l = voigt_map[q]
            fp = 1.0 if p < 3 else 0.5
            fq = 1.0 if q < 3 else 0.5
            mp_mult = 1 if p < 3 else 2
            mq_mult = 1 if q < 3 else 2
            inv_E += fp * fq * mp_mult * mq_mult * S[p, q] * n[i]*n[j]*n[k]*n[l]

    if abs(inv_E) < 1e-30:
        return 0.0, 0.0

    nu_values = []
    for chi in np.linspace(0, np.pi, n_chi, endpoint=False):
        m = np.cos(chi) * e1 + np.sin(chi) * e2

        # Numerator: S_{ijkl} n_i n_j m_k m_l
        numerator = 0.0
        for p in range(6):
            for q in range(6):
                i, j = voigt_map[p]
                k, l = voigt_map[q]
                fp = 1.0 if p < 3 else 0.5
                fq = 1.0 if q < 3 else 0.5
                mp_mult = 1 if p < 3 else 2
                mq_mult = 1 if q < 3 else 2
                numerator += fp * fq * mp_mult * mq_mult * S[p, q] * n[i]*n[j]*m[k]*m[l]

        nu = -numerator / inv_E
        nu_values.append(nu)

    if nu_values:
        return min(nu_values), max(nu_values)
    return 0.0, 0.0


# ─── Compute on angular mesh ─────────────────────────────────────────────────
theta_arr = np.linspace(0, np.pi, N_THETA)
phi_arr = np.linspace(0, 2*np.pi, N_PHI)
THETA, PHI = np.meshgrid(theta_arr, phi_arr, indexing="ij")

E_map = np.zeros_like(THETA)
beta_map = np.zeros_like(THETA)
G_min_map = np.zeros_like(THETA)
G_max_map = np.zeros_like(THETA)
nu_min_map = np.zeros_like(THETA)
nu_max_map = np.zeros_like(THETA)

print(f"\nComputing angular properties on {N_THETA}x{N_PHI} mesh...")
for it in range(N_THETA):
    for ip in range(N_PHI):
        l1, l2, l3 = direction_cosines(THETA[it, ip], PHI[it, ip])

        E_map[it, ip] = youngs_modulus(l1, l2, l3, S)
        beta_map[it, ip] = linear_compressibility(l1, l2, l3, S)
        G_min_map[it, ip], G_max_map[it, ip] = shear_modulus_minmax(l1, l2, l3, S, N_CHI)
        nu_min_map[it, ip], nu_max_map[it, ip] = poisson_ratio_minmax(l1, l2, l3, S, N_CHI)

    if (it + 1) % 30 == 0:
        print(f"  {it+1}/{N_THETA} theta values done")

print(f"\nProperty Ranges:")
print(f"  Young's modulus:       E_min = {E_map.min():.2f}, E_max = {E_map.max():.2f} GPa, "
      f"ratio = {E_map.max()/E_map.min():.3f}")
print(f"  Linear compressibility: beta_min = {beta_map.min()*1e3:.4f}, "
      f"beta_max = {beta_map.max()*1e3:.4f} TPa^-1")
print(f"  Shear modulus (min):   G_min = {G_min_map.min():.2f}, G_max_of_min = {G_min_map.max():.2f} GPa")
print(f"  Shear modulus (max):   G_min_of_max = {G_max_map.min():.2f}, G_max = {G_max_map.max():.2f} GPa")
print(f"  Poisson's ratio (min): nu_min = {nu_min_map.min():.4f}")
print(f"  Poisson's ratio (max): nu_max = {nu_max_map.max():.4f}")

# Check for auxetic behavior (negative Poisson's ratio)
if nu_min_map.min() < 0:
    print(f"  ** AUXETIC directions detected (nu < 0) **")


# ─── 3D Polar Plots ──────────────────────────────────────────────────────────
def plot_3d_polar(THETA, PHI, R, title, filename, cmap="coolwarm", unit="GPa"):
    """Create a 3D polar surface plot where r = |R(theta, phi)|."""
    R_abs = np.abs(R)
    X = R_abs * np.sin(THETA) * np.cos(PHI)
    Y = R_abs * np.sin(THETA) * np.sin(PHI)
    Z = R_abs * np.cos(THETA)

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")

    norm = plt.Normalize(R.min(), R.max())
    colors = cm.get_cmap(cmap)(norm(R))

    ax.plot_surface(X, Y, Z, facecolors=colors, alpha=0.85,
                    rstride=1, cstride=1, linewidth=0, antialiased=True)

    # Add colorbar
    mappable = cm.ScalarMappable(norm=norm, cmap=cmap)
    mappable.set_array(R)
    cbar = fig.colorbar(mappable, ax=ax, shrink=0.6, pad=0.1)
    cbar.set_label(f"{title} ({unit})", fontsize=11)

    # Labels
    max_r = R_abs.max()
    ax.set_xlim(-max_r, max_r)
    ax.set_ylim(-max_r, max_r)
    ax.set_zlim(-max_r, max_r)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.set_title(f"{title}: {FORMULA}\nMin={R.min():.2f}, Max={R.max():.2f} {unit}",
                 fontsize=13)

    plt.tight_layout()
    plt.savefig(filename, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved {filename}")


print(f"\nGenerating 3D polar plots...")

plot_3d_polar(THETA, PHI, E_map,
              "Young's Modulus E",
              os.path.join(OUTPUT_DIR, "youngs_modulus_3d.png"),
              cmap="coolwarm", unit="GPa")

plot_3d_polar(THETA, PHI, beta_map * 1e3,
              "Linear Compressibility",
              os.path.join(OUTPUT_DIR, "linear_compressibility_3d.png"),
              cmap="viridis", unit="TPa$^{-1}$")

plot_3d_polar(THETA, PHI, G_min_map,
              "Shear Modulus G (min over chi)",
              os.path.join(OUTPUT_DIR, "shear_modulus_min_3d.png"),
              cmap="plasma", unit="GPa")

plot_3d_polar(THETA, PHI, G_max_map,
              "Shear Modulus G (max over chi)",
              os.path.join(OUTPUT_DIR, "shear_modulus_max_3d.png"),
              cmap="plasma", unit="GPa")

plot_3d_polar(THETA, PHI, nu_max_map,
              "Poisson's Ratio (max over chi)",
              os.path.join(OUTPUT_DIR, "poisson_ratio_max_3d.png"),
              cmap="RdYlBu_r", unit="")

plot_3d_polar(THETA, PHI, nu_min_map,
              "Poisson's Ratio (min over chi)",
              os.path.join(OUTPUT_DIR, "poisson_ratio_min_3d.png"),
              cmap="RdYlBu_r", unit="")


# ─── 2D cross-section plots ──────────────────────────────────────────────────
fig, axes = plt.subplots(2, 2, figsize=(12, 12), subplot_kw={"projection": "polar"})

# E in the xy-plane (theta=pi/2)
ax = axes[0, 0]
phi_1d = phi_arr
idx_theta_90 = N_THETA // 2  # theta = pi/2
E_xy = E_map[idx_theta_90, :]
ax.plot(phi_1d, E_xy, "b-", linewidth=2)
ax.set_title("Young's Modulus E (xy-plane)", pad=20)

# E in the xz-plane (phi=0)
ax = axes[0, 1]
E_xz = E_map[:, 0]
ax.plot(theta_arr, E_xz, "r-", linewidth=2)
ax.set_title("Young's Modulus E (xz-plane)", pad=20)

# G_max in the xy-plane
ax = axes[1, 0]
G_max_xy = G_max_map[idx_theta_90, :]
G_min_xy = G_min_map[idx_theta_90, :]
ax.plot(phi_1d, G_max_xy, "b-", linewidth=2, label="G_max")
ax.plot(phi_1d, G_min_xy, "r--", linewidth=2, label="G_min")
ax.set_title("Shear Modulus G (xy-plane)", pad=20)
ax.legend(loc="upper right", fontsize=9)

# nu_max / nu_min in the xy-plane
ax = axes[1, 1]
nu_max_xy = nu_max_map[idx_theta_90, :]
nu_min_xy = nu_min_map[idx_theta_90, :]
ax.plot(phi_1d, nu_max_xy, "b-", linewidth=2, label="nu_max")
ax.plot(phi_1d, nu_min_xy, "r--", linewidth=2, label="nu_min")
ax.set_title("Poisson's Ratio (xy-plane)", pad=20)
ax.legend(loc="upper right", fontsize=9)

fig.suptitle(f"Angular Mechanical Properties: {FORMULA}", fontsize=14, y=1.02)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "angular_cross_sections.png"),
            dpi=150, bbox_inches="tight")
plt.close()
print(f"  Saved {OUTPUT_DIR}/angular_cross_sections.png")


# ─── Save results ────────────────────────────────────────────────────────────
results = {
    "formula": FORMULA,
    "C_ij_GPa": C.tolist(),
    "S_ij_invGPa": S.tolist(),
    "eigenvalues_C": eigenvalues.tolist(),
    "moduli_GPa": moduli,
    "anisotropy": {
        "A_U_universal": A_U,
        "A_L_log_euclidean": A_L,
        "A_Z_Zener": 2.0*c44/(c11-c12) if abs(c11-c12) > 1e-6 else None,
        "E_max_GPa": float(E_map.max()),
        "E_min_GPa": float(E_map.min()),
        "E_ratio": float(E_map.max() / E_map.min()) if E_map.min() > 0 else None,
        "G_max_GPa": float(G_max_map.max()),
        "G_min_GPa": float(G_min_map.min()),
        "nu_max": float(nu_max_map.max()),
        "nu_min": float(nu_min_map.min()),
        "is_auxetic": bool(nu_min_map.min() < 0),
    },
    "cauchy_pressure_GPa": {
        "C12_minus_C44": float(cauchy_12_44),
        "C13_minus_C55": float(cauchy_13_55),
        "C23_minus_C66": float(cauchy_23_66),
    },
    "hardness_GPa": {
        "Chen_model": float(H_chen),
        "Tian_model": float(H_tian),
        "average": float(avg_H),
        "classification": hardness_class,
    },
    "ductility": {
        "Pugh_ratio_K_over_G": float(pugh),
        "is_ductile_Pugh": bool(pugh > 1.75),
        "Poisson_ratio": float(nu_H),
        "is_ductile_Frantsevich": bool(nu_H > 0.26),
    },
}

with open(os.path.join(OUTPUT_DIR, "angular_mechanics_results.json"), "w") as f:
    json.dump(results, f, indent=2)
print(f"\nResults saved to {OUTPUT_DIR}/angular_mechanics_results.json")
```

### Step 2: 2D Materials -- In-Plane Angular Properties from C_ij^{2D}

For 2D materials, the elastic tensor is reduced to the in-plane components. The relevant stiffness matrix is 3x3 in Voigt notation: C_11, C_22, C_12, C_66 (with possible C_16, C_26 for low-symmetry cases). Properties are computed as functions of the in-plane angle theta.

```python
#!/usr/bin/env python3
"""
2D in-plane angular mechanical properties from the 2D elastic tensor.

For 2D materials (monolayers, thin films), the elastic stiffness is given by
a 3x3 matrix in Voigt notation:
    [C11  C12  C16]
    [C12  C22  C26]
    [C16  C26  C66]

Units: N/m (force per unit length) for 2D, or GPa*nm if normalized by thickness.

Computes:
  - In-plane Young's modulus E_2D(theta)
  - In-plane Poisson's ratio nu_2D(theta)
  - In-plane shear modulus G_2D(theta)
  - Polar plots

References:
  - Andrew, Mapasha & Ukpong, Phys. Rev. B 85, 125428 (2012)
  - Cadelano et al., Phys. Rev. B 82, 235414 (2010)
"""

import json
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ─── CONFIGURATION ───────────────────────────────────────────────────────────
# 2D elastic stiffness matrix (3x3 Voigt, N/m or GPa*nm)
# Example: graphene (C11=C22=358.1 N/m, C12=60.4 N/m, C66=148.9 N/m)
C2D = np.array([
    [358.1,  60.4,   0.0],
    [ 60.4, 358.1,   0.0],
    [  0.0,   0.0, 148.9],
])
FORMULA = "graphene"
UNITS = "N/m"          # "N/m" for 2D, "GPa" if normalized by layer thickness
N_THETA = 360          # Angular resolution
OUTPUT_DIR = "angular_mechanics_2d_results"
# ─────────────────────────────────────────────────────────────────────────────

os.makedirs(OUTPUT_DIR, exist_ok=True)

C2D = 0.5 * (C2D + C2D.T)
S2D = np.linalg.inv(C2D)

print(f"2D elastic tensor C_ij^2D ({UNITS}):")
for i in range(3):
    print("  " + "  ".join(f"{C2D[i,j]:10.3f}" for j in range(3)))

print(f"\n2D compliance tensor S_ij^2D (1/{UNITS}):")
for i in range(3):
    print("  " + "  ".join(f"{S2D[i,j]:12.6f}" for j in range(3)))

# Eigenvalue check
eigs = np.linalg.eigvalsh(C2D)
print(f"\nEigenvalues of C^2D: {', '.join(f'{e:.3f}' for e in eigs)}")
print(f"Mechanically stable: {'YES' if all(e > 0 for e in eigs) else 'NO'}")

# 2D moduli (isotropic averages)
E_2D_avg = (C2D[0,0]**2 - C2D[0,1]**2) / C2D[0,0] if C2D[0,0] != 0 else 0
nu_2D_avg = C2D[0,1] / C2D[0,0] if C2D[0,0] != 0 else 0
G_2D_avg = C2D[2,2]
print(f"\nIsotropic-limit 2D moduli:")
print(f"  E_2D = {E_2D_avg:.3f} {UNITS}")
print(f"  nu_2D = {nu_2D_avg:.4f}")
print(f"  G_2D = {G_2D_avg:.3f} {UNITS}")

# ─── Angular-dependent 2D properties ─────────────────────────────────────────
theta_arr = np.linspace(0, 2*np.pi, N_THETA)

E_arr = np.zeros(N_THETA)
nu_arr = np.zeros(N_THETA)
G_arr = np.zeros(N_THETA)

for it, theta in enumerate(theta_arr):
    c = np.cos(theta)
    s = np.sin(theta)
    c2 = c**2
    s2 = s**2
    cs = c * s

    # E_2D(theta) = 1 / (S11*c^4 + S22*s^4 + (2*S12 + S66)*c^2*s^2
    #               + 2*S16*c^3*s + 2*S26*c*s^3)
    inv_E = (S2D[0,0]*c2**2 + S2D[1,1]*s2**2
             + (2*S2D[0,1] + S2D[2,2])*c2*s2
             + 2*S2D[0,2]*c*c2*s + 2*S2D[1,2]*c*s*s2)

    if abs(inv_E) > 1e-30:
        E_arr[it] = 1.0 / inv_E
    else:
        E_arr[it] = 0.0

    # nu_2D(theta) = -(S12*c^4 + S12*s^4 - (2*S11+2*S22-4*S12-S66)*c^2*s^2 + ...) / inv_E
    # Simplified for orthorhombic+ (S16=S26=0):
    numerator = (S2D[0,1]*(c2**2 + s2**2)
                 + (S2D[0,0] + S2D[1,1] - S2D[2,2])*c2*s2
                 + S2D[0,2]*c*c2*s + S2D[1,2]*c*s*s2)  # cross terms if present
    if abs(inv_E) > 1e-30:
        nu_arr[it] = -numerator / inv_E
    else:
        nu_arr[it] = 0.0

    # G_2D(theta) = 1 / (4*(S11+S22-2*S12)*c^2*s^2 + S66*(c^2-s^2)^2
    #               + ... shear cross terms)
    inv_G = (4*(S2D[0,0] + S2D[1,1] - 2*S2D[0,1])*c2*s2
             + S2D[2,2]*(c2 - s2)**2
             - 4*S2D[0,2]*c*s*(c2 - s2) + 4*S2D[1,2]*c*s*(c2 - s2))
    # Note: sign convention for S16/S26 cross terms
    if abs(inv_G) > 1e-30:
        G_arr[it] = 1.0 / inv_G
    else:
        G_arr[it] = 0.0

print(f"\n2D Angular Property Ranges:")
print(f"  E_2D:  min={E_arr.min():.3f}, max={E_arr.max():.3f} {UNITS}, "
      f"ratio={E_arr.max()/E_arr.min():.3f}" if E_arr.min() > 0 else "")
print(f"  nu_2D: min={nu_arr.min():.4f}, max={nu_arr.max():.4f}")
print(f"  G_2D:  min={G_arr.min():.3f}, max={G_arr.max():.3f} {UNITS}")

# ─── Polar plots ─────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(18, 6), subplot_kw={"projection": "polar"})

ax = axes[0]
ax.plot(theta_arr, E_arr, "b-", linewidth=2)
ax.set_title(f"Young's Modulus E ({UNITS})", pad=20, fontsize=12)
ax.set_rticks([])

ax = axes[1]
ax.plot(theta_arr, np.abs(nu_arr), "r-", linewidth=2)
ax.set_title("Poisson's Ratio |nu|", pad=20, fontsize=12)
ax.set_rticks([])

ax = axes[2]
ax.plot(theta_arr, G_arr, "g-", linewidth=2)
ax.set_title(f"Shear Modulus G ({UNITS})", pad=20, fontsize=12)
ax.set_rticks([])

fig.suptitle(f"2D Angular Mechanical Properties: {FORMULA}", fontsize=14, y=1.02)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "angular_2d_polar.png"), dpi=150, bbox_inches="tight")
plt.close()
print(f"\nSaved {OUTPUT_DIR}/angular_2d_polar.png")

# Cartesian plot for clarity
fig, axes = plt.subplots(1, 3, figsize=(15, 4))

axes[0].plot(np.degrees(theta_arr), E_arr, "b-", linewidth=2)
axes[0].set_xlabel("Angle (degrees)")
axes[0].set_ylabel(f"E ({UNITS})")
axes[0].set_title("Young's Modulus")
axes[0].grid(True, alpha=0.3)

axes[1].plot(np.degrees(theta_arr), nu_arr, "r-", linewidth=2)
axes[1].set_xlabel("Angle (degrees)")
axes[1].set_ylabel("nu")
axes[1].set_title("Poisson's Ratio")
axes[1].axhline(0, color="gray", ls=":", alpha=0.5)
axes[1].grid(True, alpha=0.3)

axes[2].plot(np.degrees(theta_arr), G_arr, "g-", linewidth=2)
axes[2].set_xlabel("Angle (degrees)")
axes[2].set_ylabel(f"G ({UNITS})")
axes[2].set_title("Shear Modulus")
axes[2].grid(True, alpha=0.3)

fig.suptitle(f"2D Angular Properties: {FORMULA}", fontsize=13)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "angular_2d_cartesian.png"), dpi=150, bbox_inches="tight")
plt.close()
print(f"  Saved {OUTPUT_DIR}/angular_2d_cartesian.png")

# Save results
results_2d = {
    "formula": FORMULA,
    "C2D": C2D.tolist(),
    "S2D": S2D.tolist(),
    "units": UNITS,
    "E_2D_avg": float(E_2D_avg),
    "nu_2D_avg": float(nu_2D_avg),
    "G_2D_avg": float(G_2D_avg),
    "E_min": float(E_arr.min()),
    "E_max": float(E_arr.max()),
    "nu_min": float(nu_arr.min()),
    "nu_max": float(nu_arr.max()),
    "G_min": float(G_arr.min()),
    "G_max": float(G_arr.max()),
    "is_auxetic_2D": bool(nu_arr.min() < 0),
}
with open(os.path.join(OUTPUT_DIR, "angular_2d_results.json"), "w") as f:
    json.dump(results_2d, f, indent=2)
print(f"Results saved to {OUTPUT_DIR}/angular_2d_results.json")
```

## Key Parameters

| Parameter | Typical Value | Notes |
|---|---|---|
| C_ij (6x6) | From elastic-constants skill or literature | Must be in GPa, Voigt notation, symmetric. Positive-definite for stable phases. |
| Angular resolution (N_THETA, N_PHI) | 90 x 180 for 3D, 360 for 2D | Finer mesh gives smoother plots but slower computation. 45x90 is minimum for publication. |
| N_CHI (shear/Poisson sweep) | 72 | Number of chi angles to sweep for min/max G and nu. 36 is fast, 72 is converged. |
| Hardness model | Chen + Tian | Both are empirical. Chen model tends to be more accurate for hard materials; Tian model for metals. Neither is reliable for layered/van der Waals materials. |
| 2D elastic tensor | 3x3 Voigt (N/m) | From 2D elastic constants calculation. Convert from GPa*Angstrom by dividing by 10 to get N/m. |

## Interpreting Results

**Young's modulus E(theta,phi):**
- Shape of the 3D surface reveals elastic anisotropy. A perfect sphere means isotropic.
- Elongated lobes indicate stiff directions (e.g., along covalent bonds or close-packed planes).
- For cubic crystals: E is maximum along <111> if A_Z > 1, along <100> if A_Z < 1.
- E_max/E_min ratio: 1.0 = isotropic. Typical: 1.0--1.5 for metals, 1.5--5+ for layered materials.

**Shear modulus G(theta,phi,chi):**
- Depends on both the shear direction n and the shear plane orientation chi.
- G_min and G_max surfaces show the extremes over chi for each direction.
- Large G_max/G_min spread indicates strong in-plane anisotropy of shear resistance.

**Poisson's ratio nu(theta,phi,chi):**
- Can be negative (auxetic) in certain directions even for non-auxetic materials on average.
- nu < 0 regions appear in many cubic metals along specific crystal directions.
- nu > 0.5 is physically allowed for anisotropic single crystals (only forbidden for isotropic materials).

**Linear compressibility beta(theta,phi):**
- Negative linear compressibility (NLC) in some directions means the material expands along that direction under hydrostatic pressure.
- NLC is rare but known in some framework structures (e.g., silver oxalate, certain MOFs).

**Anisotropy indices:**
- A_U = 0 for isotropic, A_U > 0 for anisotropic. A_U > 1 is strongly anisotropic.
- Zener ratio A_Z: only meaningful for cubic symmetry. A_Z = 1 is isotropic. Al: A_Z ~ 1.2, Cu: A_Z ~ 3.2, W: A_Z ~ 1.0.

**Hardness estimates:**
- These are semi-empirical. Reliable within a factor of ~2 for hard ceramics, less reliable for metals or layered materials.
- Superhard: > 40 GPa (diamond, c-BN). Hard: 20--40 GPa. Medium: 10--20 GPa. Soft: < 10 GPa.

**Cauchy pressure:**
- C_12 - C_44 > 0: metallic/ionic bonding dominant, ductile tendency.
- C_12 - C_44 < 0: directional/covalent bonding dominant, brittle tendency.
- Combined with Pugh's ratio (K/G) and Frantsevich criterion (nu): consistent indicators strengthen confidence.

**2D materials:**
- E_2D in N/m: divide by layer thickness (in nm) to convert to GPa for comparison with bulk.
- Graphene: E_2D ~ 340 N/m, nu ~ 0.16. MoS2: E_2D ~ 120 N/m, nu ~ 0.25.
- Isotropic 2D materials (hexagonal symmetry) show circular polar plots.
- Anisotropic 2D materials (e.g., black phosphorus) show strongly elongated polar plots.

## Common Issues

| Problem | Cause | Solution |
|---|---|---|
| 3D surface has spikes or holes | Angular resolution too coarse, or C_ij near singular | Increase N_THETA and N_PHI. Check that C_ij is positive definite. |
| Negative Young's modulus in some directions | C_ij is not positive definite (unstable phase) | Verify the elastic tensor. Negative E means the structure is mechanically unstable along those directions. |
| Extremely large nu (>10) or G ratios | Near-zero denominator in compliance-based formula | Often occurs near directions where E -> 0 or in nearly unstable systems. Check C_ij stability. |
| Hardness estimates are negative | Very low shear modulus relative to bulk modulus | Chen model can give negative H_V for very ductile metals (K/G >> 1). Hardness models are not applicable for soft metals. |
| 2D polar plot is not closed | Angle range does not cover full 360 degrees | Ensure theta goes from 0 to 2*pi for 2D plots. |
| Computation is slow | Full 4th-rank tensor contraction at every grid point | Reduce N_THETA/N_PHI to 45/90. Reduce N_CHI to 36. For production plots, use vectorized numpy operations (advanced). |
| Cauchy pressure analysis disagrees with Pugh's ratio | Different bonding character in different crystallographic planes | This is physically meaningful. Report both indicators and note that anisotropic materials can have mixed ductile/brittle character. |
| VASPKIT 204 gives different plots | Different angular conventions or compliance tensor indexing | Ensure consistent Voigt ordering: 1=xx, 2=yy, 3=zz, 4=yz, 5=xz, 6=xy. VASPKIT uses the same convention. |
