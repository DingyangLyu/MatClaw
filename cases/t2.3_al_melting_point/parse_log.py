#!/usr/bin/env python3
import sys

# Parse heat_test.log for thermo lines
data = []
with open('heat_test.log', 'r') as f:
    for line in f:
        if line.strip() and line[0].isdigit():
            parts = line.split()
            if len(parts) >= 7:
                try:
                    step = int(parts[0])
                    temp = float(parts[1])
                    pe = float(parts[2])
                    density = float(parts[6])
                    data.append((step, temp, pe, density))
                except:
                    pass

print("# Step Temp(K) PotEng(eV) Density(g/cm3)")
for d in data:
    print(d[0], d[1], d[2], d[3])

# Compute potential energy per atom (total PE / atoms)
# The PotEng column is total potential energy for all atoms?
# In thermo_style custom step temp pe ke etotal press density vol
# pe is total potential energy. Need to divide by number of atoms.
# Number of atoms is 864.
print("\n# Per atom:")
for d in data:
    pe_per_atom = d[2] / 864
    print(d[0], d[1], pe_per_atom, d[3])