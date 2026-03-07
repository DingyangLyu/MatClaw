#!/usr/bin/env python3
import os, re

# Conversion factor
Ry_to_meV = 13605.698  # 1 Ry = 13.605698 eV, 1 eV = 1000 meV

# Directories
k_dirs = ["k_4x4x4", "k_6x6x6", "k_8x8x8", "k_10x10x10"]
base_path = "/workspace/group/cu_kpoint_convergence"

energies = {}
for d in k_dirs:
    outfile = os.path.join(base_path, d, "pw.out")
    if not os.path.exists(outfile):
        print(f"Warning: {outfile} not found")
        continue
    with open(outfile, 'r') as f:
        content = f.read()
    # Find line with '!' and 'total energy'
    lines = content.split('\n')
    for line in lines:
        if '!' in line and 'total energy' in line:
            # Extract number using regex
            match = re.search(r'total energy\s*=\s*([-\d\.]+)\s*Ry', line)
            if match:
                energy_ry = float(match.group(1))
                energies[d] = energy_ry
                break
    if d not in energies:
        print(f"Could not extract energy from {outfile}")

# Print table
print("# k-point convergence for FCC Cu")
print("# ecutwfc = 35 Ry, PAW pseudopotential")
print("# k-grid   Total Energy (Ry)   ΔE (meV/atom) relative to finest")
print("#" + "-" * 60)
# Sort by grid size (parse k)
def grid_size(dirname):
    # k_4x4x4 -> 4
    return int(dirname.split('_')[1].split('x')[0])

sorted_dirs = sorted(k_dirs, key=grid_size)
# Use finest as reference
finest = sorted_dirs[-1]
E_ref = energies[finest]
for d in sorted_dirs:
    E = energies[d]
    delta_meV = (E - E_ref) * Ry_to_meV
    kstr = d.split('_')[1]
    print(f"{kstr:8}    {E:12.8f}   {delta_meV:12.3f}")

# Compute differences between successive meshes
print("\n# Successive differences (meV/atom):")
for i in range(len(sorted_dirs)-1):
    d1 = sorted_dirs[i]
    d2 = sorted_dirs[i+1]
    E1 = energies[d1]
    E2 = energies[d2]
    delta = (E2 - E1) * Ry_to_meV
    k1 = d1.split('_')[1]
    k2 = d2.split('_')[1]
    print(f"{k1} -> {k2}: {delta:.3f} meV/atom")

# Determine k-point density for 1 meV/atom convergence
# Density = number of k-points per reciprocal volume
# For cubic lattice, k-point density = (k1 * k2 * k3) / volume
# Volume of primitive cell = a^3/4 for FCC? Actually volume of primitive cell = a^3/4
# Let's compute lattice constant a = 3.615 Angstrom = 6.831 Bohr
a_bohr = 3.615 / 0.529177
vol = (a_bohr**3) / 4  # volume of primitive cell in Bohr^3
print(f"\n# Lattice constant a = {3.615} Angstrom = {a_bohr:.3f} Bohr")
print(f"# Volume of primitive cell = {vol:.3f} Bohr^3")
for d in sorted_dirs:
    kstr = d.split('_')[1]
    k1, k2, k3 = map(int, kstr.split('x'))
    nk = k1 * k2 * k3
    density = nk / vol
    E = energies[d]
    delta = (E - E_ref) * Ry_to_meV
    print(f"{kstr}: {nk} k-points, density = {density:.3f} k-points/Bohr^3, ΔE = {delta:.3f} meV/atom")
    if abs(delta) <= 1.0:
        print(f"   --> Converged to within 1 meV/atom with density {density:.3f} k-points/Bohr^3")
        break