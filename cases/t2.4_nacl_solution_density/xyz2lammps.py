#!/usr/bin/env python3
import sys

# Parameters
box_size = 31.6  # angstrom
box_lo = -box_size/2
box_hi = box_size/2

# Atom type mapping
type_map = {'O': 1, 'H': 2, 'Na': 3, 'Cl': 4}
charge_map = {'O': -0.8476, 'H': 0.4238, 'Na': 1.0, 'Cl': -1.0}
mass_map = {'O': 15.9994, 'H': 1.008, 'Na': 22.9898, 'Cl': 35.453}

# Read XYZ file
with open('solution.xyz', 'r') as f:
    lines = f.readlines()
num_atoms = int(lines[0].strip())
print(f"Number of atoms: {num_atoms}")
# Skip comment line
atoms = []
for i in range(2, 2 + num_atoms):
    parts = lines[i].split()
    symbol = parts[0]
    x, y, z = map(float, parts[1:4])
    atoms.append((symbol, x, y, z))

# Determine molecule assignment
# Assume order: 1000 water molecules (O H H) then 18 Na then 18 Cl
nwater = 1000
nna = 18
ncl = 18
# Validate
assert len(atoms) == nwater*3 + nna + ncl

# Assign molecule IDs
molecule_id = 1
atom_data = []  # list of (mol, type, charge, x, y, z, symbol)
bond_data = []   # list of (type, atom1, atom2)
angle_data = []  # list of (type, atom1, atom2, atom3)

atom_index = 1
for w in range(nwater):
    # O atom
    sym, x, y, z = atoms[3*w]
    assert sym == 'O'
    atom_data.append((molecule_id, type_map[sym], charge_map[sym], x, y, z, sym))
    o_idx = atom_index
    atom_index += 1
    # H1
    sym, x, y, z = atoms[3*w + 1]
    assert sym == 'H'
    atom_data.append((molecule_id, type_map[sym], charge_map[sym], x, y, z, sym))
    h1_idx = atom_index
    atom_index += 1
    # H2
    sym, x, y, z = atoms[3*w + 2]
    assert sym == 'H'
    atom_data.append((molecule_id, type_map[sym], charge_map[sym], x, y, z, sym))
    h2_idx = atom_index
    atom_index += 1
    # Bonds
    bond_data.append((1, o_idx, h1_idx))
    bond_data.append((1, o_idx, h2_idx))
    # Angle
    angle_data.append((1, h1_idx, o_idx, h2_idx))
    molecule_id += 1

# Na ions
for i in range(nna):
    sym, x, y, z = atoms[3*nwater + i]
    assert sym == 'Na'
    atom_data.append((molecule_id, type_map[sym], charge_map[sym], x, y, z, sym))
    atom_index += 1
    molecule_id += 1

# Cl ions
for i in range(ncl):
    sym, x, y, z = atoms[3*nwater + nna + i]
    assert sym == 'Cl'
    atom_data.append((molecule_id, type_map[sym], charge_map[sym], x, y, z, sym))
    atom_index += 1
    molecule_id += 1

# Write LAMMPS data file
with open('data.nacl', 'w') as f:
    f.write('# LAMMPS data file for NaCl solution\n\n')
    f.write(f'{len(atom_data)} atoms\n')
    f.write(f'{len(bond_data)} bonds\n')
    f.write(f'{len(angle_data)} angles\n')
    f.write('0 dihedrals\n')
    f.write('0 impropers\n\n')
    f.write(f'{len(type_map)} atom types\n')
    f.write('1 bond types\n')
    f.write('1 angle types\n\n')
    f.write(f'{box_lo} {box_hi} xlo xhi\n')
    f.write(f'{box_lo} {box_hi} ylo yhi\n')
    f.write(f'{box_lo} {box_hi} zlo zhi\n\n')
    f.write('Masses\n\n')
    for sym, tid in type_map.items():
        f.write(f'{tid} {mass_map[sym]}\n')
    f.write('\nAtoms\n\n')
    for i, (mol, typ, charge, x, y, z, sym) in enumerate(atom_data, 1):
        f.write(f'{i} {mol} {typ} {charge} {x} {y} {z}\n')
    f.write('\nBonds\n\n')
    for i, (typ, a1, a2) in enumerate(bond_data, 1):
        f.write(f'{i} {typ} {a1} {a2}\n')
    f.write('\nAngles\n\n')
    for i, (typ, a1, a2, a3) in enumerate(angle_data, 1):
        f.write(f'{i} {typ} {a1} {a2} {a3}\n')

print(f"Written data.nacl with {len(atom_data)} atoms, {len(bond_data)} bonds, {len(angle_data)} angles")