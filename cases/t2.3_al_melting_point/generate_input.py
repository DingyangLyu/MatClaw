#!/usr/bin/env python3
import sys

# Parameters
latparam = 4.041  # FCC lattice constant for Al (Angstrom)
nx, ny, nz = 6, 6, 6  # unit cells
potential_path = "/usr/share/lammps/potentials/Al_zhou.eam.alloy"

# Generate temperature list from 300K to 1500K step 25K
temps = list(range(300, 1525, 25))
# Add extra points around melting point ~933K
extra = [900, 925, 950, 975, 1000]
for t in extra:
    if t not in temps:
        temps.append(t)
temps.sort()

# Write LAMMPS input
with open("melting_loop.in", "w") as f:
    f.write("""# LAMMPS input script for aluminum melting point calculation
# Using Zhou EAM potential
# Loop over temperatures

# Units
units metal

# Atom style
atom_style atomic

# Create box and lattice
lattice fcc {latparam}
region box block 0 {nx} 0 {ny} 0 {nz}
create_box 1 box
create_atoms 1 box

# Define potential
pair_style eam/alloy
pair_coeff * * {potential_path} Al

# Velocity initialization
velocity all create 300.0 12345 rot yes dist gaussian

# Neighbor settings
neighbor 2.0 bin
neigh_modify delay 0 every 1 check yes

# Thermodynamic output
thermo 100
thermo_style custom step temp pe ke etotal press density vol

# Minimization
minimize 1e-10 1e-10 1000 1000

# Equilibration at 300K and 1 atm (NPT)
reset_timestep 0
timestep 0.001
fix 1 all npt temp 300 300 0.1 iso 0 0 1.0
run 5000
unfix 1

# Define temperature list as index variable
variable T index """.format(latparam=latparam, nx=nx, ny=ny, nz=nz, potential_path=potential_path))
    # Write temperature list
    for temp in temps:
        f.write(f"{temp} ")
    f.write("\n")

    f.write("""
# Setup compute for potential energy per atom
compute peatom all pe/atom
compute pe all reduce sum c_peatom
variable peatom equal c_pe/atoms
variable density equal density

# Output file
variable myTemp equal ${T}
fix 2 all print 100 "${myTemp} ${peatom} ${density}" file thermo.out screen no

# Loop over temperatures
label loop
variable Tcur equal ${T}
print "Running temperature ${Tcur}"
fix 3 all npt temp ${Tcur} ${Tcur} 0.1 iso 0 0 1.0
run 2000
unfix 3
next T
jump SELF loop

# After loop
write_data final.data
write_dump all atom final.dump
print "All temperatures done"
""")

print(f"Generated input with {len(temps)} temperature points.")
print("Temperatures:", temps)