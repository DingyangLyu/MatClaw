#!/usr/bin/env python3
import sys
import re

def parse_eigenvalues(filename):
    """Return list of lists of eigenvalues per k-point."""
    with open(filename, 'r') as f:
        lines = f.readlines()
    evals_per_k = []
    i = 0
    while i < len(lines):
        if 'bands (ev):' in lines[i]:
            evals = []
            # next line(s) contain eigenvalues
            j = i + 1
            while j < len(lines) and lines[j].strip() and not lines[j].startswith('          k ='):
                # parse numbers
                parts = lines[j].split()
                for p in parts:
                    try:
                        evals.append(float(p))
                    except ValueError:
                        pass
                j += 1
            if evals:
                evals_per_k.append(evals)
            i = j
        else:
            i += 1
    return evals_per_k

def main():
    if len(sys.argv) < 2:
        print("Usage: python parse_gap.py <bands_output_file>")
        sys.exit(1)
    bands_file = sys.argv[1]
    evals_list = parse_eigenvalues(bands_file)
    if not evals_list:
        print("No eigenvalues found.")
        sys.exit(1)

    print(f"Found {len(evals_list)} k-points.")
    # Each k-point should have same number of bands
    nbands = len(evals_list[0])
    print(f"Number of bands per k-point: {nbands}")
    # Total valence electrons = 24 (Ni 18 + O 6) per formula unit (2 atoms)
    # Each band holds 2 electrons (non-spin-polarized)
    noccupied = 12  # 24/2
    print(f"Assuming {noccupied} occupied bands.")

    # Compute VBM and CBM across all k-points
    vbm = -float('inf')
    cbm = float('inf')
    for evals in evals_list:
        # eigenvalues are sorted ascending? Looks like they are in increasing order.
        # The first noccupied bands are occupied.
        vbm_k = max(evals[:noccupied])
        cbm_k = min(evals[noccupied:])
        if vbm_k > vbm:
            vbm = vbm_k
        if cbm_k < cbm:
            cbm = cbm_k

    gap = cbm - vbm
    print(f"Valence band maximum (across k-points): {vbm:.6f} eV")
    print(f"Conduction band minimum (across k-points): {cbm:.6f} eV")
    print(f"Band gap: {gap:.6f} eV")
    if gap <= 0:
        print("Material appears metallic (negative or zero gap).")
    else:
        print("Material is insulating.")

    # Also compute Fermi level from SCF output (optional)
    # Parse Fermi energy from SCF output if provided as second argument
    if len(sys.argv) >= 3:
        scf_file = sys.argv[2]
        with open(scf_file, 'r') as f:
            content = f.read()
        match = re.search(r"the Fermi energy is\s+([\d\.]+)\s+ev", content, re.IGNORECASE)
        if match:
            fermi = float(match.group(1))
            print(f"Fermi energy from SCF: {fermi:.6f} eV")
            print(f"VBM relative to Fermi: {vbm - fermi:.6f} eV")
            print(f"CBM relative to Fermi: {cbm - fermi:.6f} eV")

if __name__ == '__main__':
    main()