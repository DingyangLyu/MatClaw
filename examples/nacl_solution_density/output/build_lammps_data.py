import math

# Box parameters
L = 31.0  # Angstrom (will be adjusted by NPT)
N_water = 1000
N_na = 18
N_cl = 18
N_atoms = N_water*3 + N_na + N_cl  # 3036

# Atom types: 1=OW, 2=HW, 3=Na, 4=Cl
# SPC/E: O-H bond = 1.0 Å, H-O-H = 109.47°
# Joung-Cheatham (SPC/E) params:
#   Na+: q=+1, eps=0.3526418 kcal/mol, sigma=2.159538 Å
#   Cl-: q=-1, eps=0.0127850 kcal/mol, sigma=4.830500 Å
# SPC/E: O: q=-0.8476, eps=0.15535 kcal/mol, sigma=3.1660 Å
#         H: q=+0.4238, mass=1.008

masses = {1: 15.9994, 2: 1.00794, 3: 22.98977, 4: 35.45300}
charges = {1: -0.8476, 2: 0.4238, 3: 1.0, 4: -1.0}

# Read packmol output
with open('nacl_solution.xyz') as f:
    lines = f.readlines()

# Lines: [count, comment, atom1, atom2, ...]
atom_lines = lines[2:]  # skip header

atoms = []
bonds = []
angles = []

atom_id = 0
bond_id = 0
angle_id = 0
mol_id = 0

# Process water molecules (first 3000 lines = 1000 waters * 3 atoms)
for i in range(N_water):
    mol_id += 1
    base = i * 3
    o_line = atom_lines[base].split()
    h1_line = atom_lines[base+1].split()
    h2_line = atom_lines[base+2].split()
    
    atom_id += 1; oid = atom_id
    atoms.append((atom_id, mol_id, 1, charges[1], float(o_line[1]), float(o_line[2]), float(o_line[3])))
    atom_id += 1; h1id = atom_id
    atoms.append((atom_id, mol_id, 2, charges[2], float(h1_line[1]), float(h1_line[2]), float(h1_line[3])))
    atom_id += 1; h2id = atom_id
    atoms.append((atom_id, mol_id, 2, charges[2], float(h2_line[1]), float(h2_line[2]), float(h2_line[3])))
    
    bond_id += 1
    bonds.append((bond_id, 1, oid, h1id))
    bond_id += 1
    bonds.append((bond_id, 1, oid, h2id))
    
    angle_id += 1
    angles.append((angle_id, 1, h1id, oid, h2id))

# Na ions
for i in range(N_na):
    mol_id += 1
    atom_id += 1
    line = atom_lines[N_water*3 + i].split()
    atoms.append((atom_id, mol_id, 3, charges[3], float(line[1]), float(line[2]), float(line[3])))

# Cl ions
for i in range(N_cl):
    mol_id += 1
    atom_id += 1
    line = atom_lines[N_water*3 + N_na + i].split()
    atoms.append((atom_id, mol_id, 4, charges[4], float(line[1]), float(line[2]), float(line[3])))

# Write LAMMPS data file
with open('nacl.data', 'w') as f:
    f.write("NaCl aqueous solution (SPC/E + Joung-Cheatham)\n\n")
    f.write(f"{len(atoms)} atoms\n")
    f.write(f"{len(bonds)} bonds\n")
    f.write(f"{len(angles)} angles\n\n")
    f.write("4 atom types\n")
    f.write("1 bond types\n")
    f.write("1 angle types\n\n")
    f.write(f"0.0 {L} xlo xhi\n")
    f.write(f"0.0 {L} ylo yhi\n")
    f.write(f"0.0 {L} zlo zhi\n\n")
    f.write("Masses\n\n")
    for t, m in masses.items():
        f.write(f"  {t} {m}\n")
    f.write("\nAtoms  # full\n\n")
    for a in atoms:
        f.write(f"  {a[0]} {a[1]} {a[2]} {a[3]:.6f} {a[4]:.6f} {a[5]:.6f} {a[6]:.6f}\n")
    f.write("\nBonds\n\n")
    for b in bonds:
        f.write(f"  {b[0]} {b[1]} {b[2]} {b[3]}\n")
    f.write("\nAngles\n\n")
    for a in angles:
        f.write(f"  {a[0]} {a[1]} {a[2]} {a[3]} {a[4]}\n")

print(f"Written nacl.data: {len(atoms)} atoms, {len(bonds)} bonds, {len(angles)} angles")
print(f"Mol count: {mol_id} molecules")
