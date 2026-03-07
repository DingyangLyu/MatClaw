#!/usr/bin/env python3
import sys
import re

def parse_bands_output(filename):
    """Parse QE bands output file and return eigenvalues per k-point."""
    with open(filename, 'r') as f:
        lines = f.readlines()

    # Find lines with k-point coordinates
    kpoints = []
    eigenvalues = []  # list of lists
    i = 0
    while i < len(lines):
        line = lines[i]
        # Match k-point line like "k(    1) = (   0.0000000   0.0000000   0.0000000), wk =   0.0000000"
        if re.match(r'\s*k\(\s*\d+\)\s*=', line):
            # Extract k-point index (not needed)
            # Look ahead for "bands (ev):" or "bands (eV):"
            j = i + 1
            while j < len(lines) and 'bands' not in lines[j].lower():
                j += 1
            if j < len(lines):
                # Found bands header
                # Next lines contain eigenvalues
                evals = []
                j += 1
                while j < len(lines) and (lines[j].strip() and not lines[j].startswith(' ' * 10) and not re.match(r'\s*k\(\s*\d+\)\s*=', lines[j])):
                    # Parse numbers
                    parts = lines[j].split()
                    for p in parts:
                        try:
                            evals.append(float(p))
                        except ValueError:
                            pass
                    j += 1
                if evals:
                    eigenvalues.append(evals)
                    kpoints.append(line.strip())
                i = j
                continue
        i += 1

    # Alternative parsing: look for lines with many numbers after "k ="
    # Actually eigenvalues are printed after a blank line?
    # Let's try a simpler approach: find all floating numbers in lines after "End of band structure calculation"
    # But we'll implement a more robust method later.
    # For now, return empty.
    return kpoints, eigenvalues

def parse_bands_output_simple(filename):
    """Simpler parser: look for lines with 'k(' and then subsequent lines with numbers."""
    with open(filename, 'r') as f:
        content = f.read()
    # Split by k-point blocks
    blocks = re.split(r'\s*k\(\s*\d+\)\s*=.*\n', content)
    # First block is before first k-point, ignore
    blocks = blocks[1:]
    eigenvalues = []
    for blk in blocks:
        # Extract numbers that look like eigenvalues (maybe preceded by spaces)
        # Use regex to find floating point numbers in lines that are not part of other text
        # Heuristic: find lines that contain only numbers and spaces
        lines = blk.split('\n')
        for line in lines:
            if re.match(r'^\s*[-\d\.\s]+$', line):
                nums = [float(x) for x in line.split()]
                if nums:
                    eigenvalues.append(nums)
                    break
    return eigenvalues

def parse_bands_output_verbose(filename):
    """Parse by scanning for 'bands (ev)' or 'eigenvalues'."""
    with open(filename, 'r') as f:
        lines = f.readlines()
    eigenvalues = []
    i = 0
    while i < len(lines):
        if 'bands' in lines[i].lower() and 'ev' in lines[i].lower():
            # Next lines contain eigenvalues until blank line or next k-point
            evals = []
            j = i + 1
            while j < len(lines) and lines[j].strip() and not lines[j].startswith('   k('):
                parts = lines[j].split()
                for p in parts:
                    try:
                        evals.append(float(p))
                    except ValueError:
                        pass
                j += 1
            if evals:
                eigenvalues.append(evals)
            i = j
        else:
            i += 1
    return eigenvalues

def main():
    if len(sys.argv) < 2:
        print("Usage: python extract_gap.py <bands_output_file>")
        sys.exit(1)
    bands_file = sys.argv[1]
    # Try parsing
    evals_list = parse_bands_output_verbose(bands_file)
    if not evals_list:
        # fallback
        evals_list = parse_bands_output_simple(bands_file)

    if not evals_list:
        print("Could not parse eigenvalues from bands output.")
        sys.exit(1)

    # Determine number of occupied bands: total valence electrons = 24 (Ni 18 + O 6)
    # With spin degeneracy 2, occupied bands = 24 / 2 = 12
    noccupied = 12
    # Flatten across k-points? We need per k-point.
    vbm = -float('inf')
    cbm = float('inf')
    for evals in evals_list:
        # Ensure we have enough bands
        if len(evals) >= noccupied:
            # Occupied bands are the first noccupied (assuming sorted)
            # Actually eigenvalues are sorted ascending
            vbm_k = max(evals[:noccupied])
            cbm_k = min(evals[noccupied:])
            vbm = max(vbm, vbm_k)
            cbm = min(cbm, cbm_k)

    gap = cbm - vbm
    print(f"Valence band maximum (across k-points): {vbm:.6f} eV")
    print(f"Conduction band minimum (across k-points): {cbm:.6f} eV")
    print(f"Band gap: {gap:.6f} eV")
    if gap <= 0:
        print("Material appears metallic (negative or zero gap).")
    else:
        print("Material is insulating.")

if __name__ == '__main__':
    main()