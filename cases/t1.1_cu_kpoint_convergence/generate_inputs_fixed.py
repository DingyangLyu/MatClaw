#!/usr/bin/env python3
"""
Generate Quantum ESPRESSO input files for k-point convergence study of FCC Cu.
Using ibrav=2 (FCC) for simplicity.
"""

import os

# Lattice constant for FCC Cu in Bohr (3.615 Angstrom = 6.831 Bohr)
a_bohr = 3.615 / 0.529177

def generate_input(kgrid, outdir="."):
    """Generate pw.x input file for given k-point grid."""
    k1, k2, k3 = kgrid
    input_text = f"""&control
    calculation = 'scf'
    restart_mode = 'from_scratch'
    prefix = 'cu_fcc'
    pseudo_dir = './'
    outdir = './tmp'
    tprnfor = .false.
    tstress = .false.
    verbosity = 'high'
/
&system
    ibrav = 2
    celldm(1) = {a_bohr:.6f}
    nat = 1
    ntyp = 1
    ecutwfc = 35.0
    ecutrho = 280.0
    occupations = 'smearing'
    smearing = 'mv'
    degauss = 0.01
/
&electrons
    conv_thr = 1.0d-8
    mixing_beta = 0.7
    electron_maxstep = 100
/
&ions
/
&cell
/
ATOMIC_SPECIES
 Cu 63.546 Cu.pbe-dn-kjpaw_psl.1.0.0.UPF
ATOMIC_POSITIONS crystal
 Cu 0.0 0.0 0.0
K_POINTS automatic
 {k1} {k2} {k3} 0 0 0
"""
    os.makedirs(outdir, exist_ok=True)
    with open(os.path.join(outdir, "pw.in"), "w") as f:
        f.write(input_text)
    print(f"Generated input for k-grid {k1}x{k2}x{k3} in {outdir}")

def main():
    # List of k-point grids to test
    kgrids = [(4,4,4), (6,6,6), (8,8,8), (10,10,10)]

    for kgrid in kgrids:
        kstr = f"{kgrid[0]}x{kgrid[1]}x{kgrid[2]}"
        outdir = f"k_{kstr}"
        generate_input(kgrid, outdir)

if __name__ == "__main__":
    main()