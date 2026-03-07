#!/usr/bin/env python3
import sys

def parse_log(filename):
    densities = []
    with open(filename, 'r') as f:
        for line in f:
            if line.strip().startswith('Loop time'):
                break
            parts = line.split()
            # Thermo output lines start with step number (integer)
            if len(parts) >= 4 and parts[0].isdigit():
                step = int(parts[0])
                # Find density column; we know thermo style includes density after press
                # Assuming format: step temp press density ...
                # We'll locate index of density by header? Hard.
                # Instead we know density is the 4th column (index 3)
                try:
                    dens = float(parts[3])
                    densities.append((step, dens))
                except (ValueError, IndexError):
                    pass
    return densities

def main():
    logfile = 'log2.nacl'
    densities = parse_log(logfile)
    if not densities:
        print("No density data found")
        return
    # Assume equilibration steps up to 10000
    prod_densities = [d for step, d in densities if step >= 10000]
    if not prod_densities:
        prod_densities = [d for step, d in densities]
    avg = sum(prod_densities) / len(prod_densities)
    std = (sum((d - avg)**2 for d in prod_densities) / len(prod_densities))**0.5
    print(f"Number of density samples: {len(prod_densities)}")
    print(f"Average density: {avg:.6f} g/cm^3")
    print(f"Standard deviation: {std:.6f} g/cm^3")
    print(f"Range: {min(prod_densities):.6f} - {max(prod_densities):.6f}")

if __name__ == '__main__':
    main()