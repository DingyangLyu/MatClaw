#!/usr/bin/env python3
"""
Create FCC aluminum crystal for LAMMPS simulation.
Lattice constant: 4.041 Å (Zhou potential)
"""

import numpy as np

# Parameters
a = 4.041  # lattice constant (Å)
nx, ny, nz = 10, 10, 50  # number of unit cells in each direction
# FCC basis vectors (fractional coordinates)
basis = np.array([
    [0.0, 0.0, 0.0],
    [0.5, 0.5, 0.0],
    [0.5, 0.0, 0.5],
    [0.0, 0.5, 0.5]
])

# Compute box dimensions
lx = nx * a
ly = ny * a
lz = nz * a
print(f"Box dimensions: {lx:.3f} × {ly:.3f} × {lz:.3f} Å")

# Generate atom positions
positions = []
for i in range(nx):
    for j in range(ny):
        for k in range(nz):
            # Unit cell origin
            origin = np.array([i, j, k]) * a
            for b in basis:
                pos = origin + b * a
                positions.append(pos)

natoms = len(positions)
print(f"Total atoms: {natoms}")

# Write LAMMPS data file
with open('al_fcc.data', 'w') as f:
    f.write(f"FCC aluminum crystal for melting point simulation\n")
    f.write(f"\n")
    f.write(f"{natoms} atoms\n")
    f.write(f"1 atom types\n")
    f.write(f"\n")
    f.write(f"0.0 {lx:.8f} xlo xhi\n")
    f.write(f"0.0 {ly:.8f} ylo yhi\n")
    f.write(f"0.0 {lz:.8f} zlo zhi\n")
    f.write(f"\n")
    f.write(f"Masses\n")
    f.write(f"\n")
    f.write(f"1 26.9815  # Al atomic mass (g/mol)\n")
    f.write(f"\n")
    f.write(f"Atoms # atomic\n")
    f.write(f"\n")
    for i, pos in enumerate(positions, start=1):
        f.write(f"{i} 1 {pos[0]:.8f} {pos[1]:.8f} {pos[2]:.8f}\n")

print("Data file written to al_fcc.data")

# Also create a smaller box for liquid equilibration (for generating liquid configuration)
nx_small, ny_small, nz_small = 5, 5, 5
lx_small = nx_small * a
ly_small = ny_small * a
lz_small = nz_small * a

positions_small = []
for i in range(nx_small):
    for j in range(ny_small):
        for k in range(nz_small):
            origin = np.array([i, j, k]) * a
            for b in basis:
                pos = origin + b * a
                positions_small.append(pos)

natoms_small = len(positions_small)
with open('al_small.data', 'w') as f:
    f.write(f"Small FCC aluminum crystal\n")
    f.write(f"\n")
    f.write(f"{natoms_small} atoms\n")
    f.write(f"1 atom types\n")
    f.write(f"\n")
    f.write(f"0.0 {lx_small:.8f} xlo xhi\n")
    f.write(f"0.0 {ly_small:.8f} ylo yhi\n")
    f.write(f"0.0 {lz_small:.8f} zlo zhi\n")
    f.write(f"\n")
    f.write(f"Masses\n")
    f.write(f"\n")
    f.write(f"1 26.9815\n")
    f.write(f"\n")
    f.write(f"Atoms # atomic\n")
    f.write(f"\n")
    for i, pos in enumerate(positions_small, start=1):
        f.write(f"{i} 1 {pos[0]:.8f} {pos[1]:.8f} {pos[2]:.8f}\n")

print(f"Small data file written to al_small.data ({natoms_small} atoms)")