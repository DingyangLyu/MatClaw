#!/usr/bin/env python3
import re
import sys

def main():
    with open('simulation.out', 'r') as f:
        content = f.read()
    # Find lines that look like thermo output: step number then floats
    # pattern: start of line, spaces, digits, spaces, float, spaces, float, spaces, float (density)
    # We'll capture step and density (fourth column)
    lines = content.split('\n')
    densities = []
    for line in lines:
        if re.match(r'^\s*\d+\s+[-+]?\d*\.?\d+', line):
            parts = line.split()
            if len(parts) >= 4:
                try:
                    step = int(parts[0])
                    dens = float(parts[3])
                    densities.append((step, dens))
                except ValueError:
                    pass
    # Filter steps >= 10000 (production)
    prod = [(s,d) for s,d in densities if s >= 10000]
    if not prod:
        prod = densities
    steps = [s for s,_ in prod]
    vals = [d for _,d in prod]
    print(f"Found {len(vals)} density samples")
    if vals:
        avg = sum(vals)/len(vals)
        std = (sum((d-avg)**2 for d in vals)/len(vals))**0.5
        print(f"Average density: {avg:.6f} g/cm^3")
        print(f"Std dev: {std:.6f}")
        print(f"Min: {min(vals):.6f}, Max: {max(vals):.6f}")
        # Print last 10 values
        print("\nLast 10 density values:")
        for s,d in prod[-10:]:
            print(f"  {s}: {d:.6f}")

if __name__ == '__main__':
    main()