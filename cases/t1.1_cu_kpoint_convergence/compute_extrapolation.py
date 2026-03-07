import numpy as np
Ry_to_meV = 13605.698
a_bohr = 3.615 / 0.529177
vol = (a_bohr**3) / 4
# Data: k-grid string, total energy
data = [
    ("4x4x4", -213.08027744),
    ("6x6x6", -213.08760178),
    ("8x8x8", -213.08978619),
    ("10x10x10", -213.09110877),
    ("12x12x12", -213.09191603),
    ("14x14x14", -213.09224092),
    ("16x16x16", -213.09223379),
    ("18x18x18", -213.09223824),
]
# Compute N_k
N = []
E = []
for kstr, e in data:
    k = int(kstr.split('x')[0])
    N.append(k**3)
    E.append(e)
N = np.array(N)
E = np.array(E)
# Linear fit E = E_inf + B/N
invN = 1.0 / N
coeff = np.polyfit(invN, E, 1)
E_inf = coeff[1]
B = coeff[0]
print(f"Linear fit: E_inf = {E_inf:.8f} Ry, B = {B:.6f} Ry")
# Compute deviations
for i, (kstr, e) in enumerate(data):
    dE = (e - E_inf) * Ry_to_meV
    print(f"{kstr}: ΔE = {dE:.3f} meV/atom")
# Required N for 1 meV
target_dE_Ry = 1.0 / Ry_to_meV
N_target = B / target_dE_Ry
k_target = int(np.ceil(N_target ** (1/3)))
print(f"\nRequired N_k for 1 meV/atom: {N_target:.0f}")
print(f"Cubic mesh: {k_target}x{k_target}x{k_target}")
print(f"Density: {N_target/vol:.3f} k-points/Bohr^3")
print(f"k-point spacing: {2*np.pi/(a_bohr * k_target):.6f} Bohr^-1")
# Also compute using power law fit (E - E_inf) = A / N^alpha
# Use last point as E_inf estimate
E_inf_est = E[-1]
dE = E - E_inf_est
valid = dE > 0
if np.sum(valid) >= 3:
    logN = np.log(N[valid])
    logdE = np.log(dE[valid])
    coeff2 = np.polyfit(logN, logdE, 1)
    alpha = -coeff2[0]
    A = np.exp(coeff2[1])
    print(f"\nPower law fit: alpha = {alpha:.4f}, A = {A:.6f} Ry")
    N_target2 = (A / target_dE_Ry) ** (1/alpha)
    k_target2 = int(np.ceil(N_target2 ** (1/3)))
    print(f"Required N_k: {N_target2:.0f}")
    print(f"Cubic mesh: {k_target2}x{k_target2}x{k_target2}")
    print(f"Density: {N_target2/vol:.3f} k-points/Bohr^3")