#!/usr/bin/env python3
import os, re, numpy as np

Ry_to_meV = 13605.698  # 1 Ry = 13.605698 eV, 1 eV = 1000 meV

# Lattice constant a = 3.615 Angstrom = 6.831 Bohr
a_bohr = 3.615 / 0.529177
vol = (a_bohr**3) / 4  # volume of primitive cell in Bohr^3
print(f"Lattice constant a = {3.615} Angstrom = {a_bohr:.6f} Bohr")
print(f"Volume of primitive cell = {vol:.6f} Bohr^3")

# Gather all k-point directories
k_dirs = []
for d in os.listdir('.'):
    if d.startswith('k_') and os.path.isdir(d):
        k_dirs.append(d)
# sort by grid size
def grid_size(dirname):
    # k_4x4x4 -> 4
    return int(dirname.split('_')[1].split('x')[0])
k_dirs.sort(key=grid_size)

energies = {}
for d in k_dirs:
    outfile = os.path.join(d, "pw.out")
    if not os.path.exists(outfile):
        print(f"Warning: {outfile} not found")
        continue
    with open(outfile, 'r') as f:
        content = f.read()
    lines = content.split('\n')
    for line in lines:
        if '!' in line and 'total energy' in line:
            match = re.search(r'total energy\s*=\s*([-\d\.]+)\s*Ry', line)
            if match:
                energy_ry = float(match.group(1))
                energies[d] = energy_ry
                break
    if d not in energies:
        print(f"Could not extract energy from {outfile}")

# Print table
print("\n# k-point convergence for FCC Cu")
print("# ecutwfc = 35 Ry, PAW pseudopotential")
print("# k-grid   Total Energy (Ry)   ΔE (meV/atom) relative to finest")
print("#" + "-" * 60)
finest = k_dirs[-1]
E_ref = energies[finest]
for d in k_dirs:
    E = energies[d]
    delta_meV = (E - E_ref) * Ry_to_meV
    kstr = d.split('_')[1]
    print(f"{kstr:8}    {E:12.8f}   {delta_meV:12.3f}")

# Compute differences between successive meshes
print("\n# Successive differences (meV/atom):")
for i in range(len(k_dirs)-1):
    d1 = k_dirs[i]
    d2 = k_dirs[i+1]
    E1 = energies[d1]
    E2 = energies[d2]
    delta = (E2 - E1) * Ry_to_meV
    k1 = d1.split('_')[1]
    k2 = d2.split('_')[1]
    print(f"{k1} -> {k2}: {delta:.3f} meV/atom")

# Compute k-point density
print("\n# K-point density analysis")
print("# k-grid   N_k   density (k-points/Bohr^3)   ΔE (meV/atom)")
for d in k_dirs:
    kstr = d.split('_')[1]
    k1, k2, k3 = map(int, kstr.split('x'))
    nk = k1 * k2 * k3
    density = nk / vol
    E = energies[d]
    delta = (E - E_ref) * Ry_to_meV
    print(f"{kstr:8} {nk:6}   {density:12.6f}          {delta:12.3f}")

# Determine if any mesh achieves 1 meV/atom convergence
print("\n# Convergence to 1 meV/atom:")
converged = False
for d in k_dirs:
    kstr = d.split('_')[1]
    delta = (energies[d] - E_ref) * Ry_to_meV
    if abs(delta) <= 1.0:
        converged = True
        k1, k2, k3 = map(int, kstr.split('x'))
        nk = k1 * k2 * k3
        density = nk / vol
        print(f"Mesh {kstr} converged within 1 meV/atom (ΔE = {delta:.3f} meV/atom)")
        print(f"  Required density = {density:.6f} k-points/Bohr^3")
        break
if not converged:
    print("No mesh in the tested range converged to 1 meV/atom.")
    print("Extrapolating using power law fit...")

# Fit energy vs. 1/N_k (or 1/density) to estimate required density
# Assume E(N) = E_inf + A / N^alpha
# Use log(E - E_inf) = log(A) - alpha * log(N)
# We'll use the finest mesh as E_inf (may have error). Use two-parameter fit.
# Let's try using all data points.
N_vals = []
E_vals = []
for d in k_dirs:
    kstr = d.split('_')[1]
    k1, k2, k3 = map(int, kstr.split('x'))
    nk = k1 * k2 * k3
    N_vals.append(nk)
    E_vals.append(energies[d])
N = np.array(N_vals)
E = np.array(E_vals)
# Use E_inf approximated by finest mesh
E_inf = E[-1]
dE = E - E_inf
# Only use points where dE > 0 (should be positive since energy decreases with N)
# Use logarithmic fit: log(dE) = log(A) - alpha * log(N)
valid = dE > 0
if np.sum(valid) >= 3:
    logN = np.log(N[valid])
    logdE = np.log(dE[valid])
    # linear regression
    coeff = np.polyfit(logN, logdE, 1)
    alpha = -coeff[0]
    logA = coeff[1]
    A = np.exp(logA)
    print(f"\n# Power law fit: E(N) = E_inf + A / N^alpha")
    print(f"  alpha = {alpha:.4f}")
    print(f"  A = {A:.6f} Ry")
    # Predict required N for dE = 1 meV/atom = 1 / Ry_to_meV Ry
    target_dE_Ry = 1.0 / Ry_to_meV
    # Solve A / N^alpha = target_dE_Ry => N = (A / target_dE_Ry)^(1/alpha)
    N_target = (A / target_dE_Ry) ** (1/alpha)
    # Corresponding k-mesh (cubic) roughly (N_target)**(1/3)
    k_target = int(np.ceil(N_target ** (1/3)))
    # Adjust to have same k in all directions
    k_target = max(k_target, 1)
    density_target = N_target / vol
    print(f"  Predicted N_k required for 1 meV/atom: {N_target:.0f} k-points")
    print(f"  Approximate cubic mesh: {k_target}x{k_target}x{k_target}")
    print(f"  Required density: {density_target:.6f} k-points/Bohr^3")
    # Also compute k-point spacing Δk = 2π/(a*N_kpoints_per_direction)
    # For cubic mesh, k-point spacing along each reciprocal lattice vector:
    # Δk = 2π/(a * k_target)
    delta_k = 2 * np.pi / (a_bohr * k_target)
    print(f"  Approximate k-point spacing: {delta_k:.6f} Bohr^-1")
else:
    print("Not enough data for power law fit.")

# Alternative: fit energy vs. 1/N
# E = E_inf + B/N
# Use linear regression of E vs 1/N
print("\n# Linear fit: E = E_inf + B/N")
# Use all points
invN = 1.0 / N
coeff2 = np.polyfit(invN, E, 1)
E_inf_lin = coeff2[1]
B = coeff2[0]
print(f"  E_inf = {E_inf_lin:.8f} Ry")
print(f"  B = {B:.6f} Ry")
# Predict N for target dE
target_dE_Ry = 1.0 / Ry_to_meV
N_target_lin = B / target_dE_Ry
k_target_lin = int(np.ceil(N_target_lin ** (1/3)))
density_target_lin = N_target_lin / vol
print(f"  Predicted N_k (linear): {N_target_lin:.0f} k-points")
print(f"  Approximate cubic mesh: {k_target_lin}x{k_target_lin}x{k_target_lin}")
print(f"  Required density: {density_target_lin:.6f} k-points/Bohr^3")
delta_k_lin = 2 * np.pi / (a_bohr * k_target_lin)
print(f"  Approximate k-point spacing: {delta_k_lin:.6f} Bohr^-1")