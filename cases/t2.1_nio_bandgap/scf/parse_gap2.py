#!/usr/bin/env python3
import sys
import re

def main():
    with open('nio.bands.out', 'r') as f:
        text = f.read()
    # find all matches of "bands (ev):" followed by numbers
    # Use regex to capture the block after each occurrence
    pattern = r'bands \(ev\):\s*\n(.*?)(?=\n\s*\n|\n\s*k = |\Z)'
    blocks = re.findall(pattern, text, re.DOTALL)
    if not blocks:
        print("No blocks found")
        return
    evals_all = []
    for blk in blocks:
        numbers = []
        for line in blk.split('\n'):
            line = line.strip()
            if line:
                numbers.extend([float(x) for x in line.split()])
        evals_all.append(numbers)
    print(f"Found {len(evals_all)} k-points")
    nbands = len(evals_all[0])
    print(f"Bands per k-point: {nbands}")
    noccupied = 12
    vbm = -1e9
    cbm = 1e9
    for evals in evals_all:
        vbm_k = max(evals[:noccupied])
        cbm_k = min(evals[noccupied:])
        if vbm_k > vbm:
            vbm = vbm_k
        if cbm_k < cbm:
            cbm = cbm_k
    gap = cbm - vbm
    print(f"VBM: {vbm:.6f} eV")
    print(f"CBM: {cbm:.6f} eV")
    print(f"Band gap: {gap:.6f} eV")
    if gap <= 0:
        print("Metallic")
    else:
        print("Insulating")
    # Fermi energy from SCF
    with open('nio.scf.out', 'r') as f:
        scf_text = f.read()
    match = re.search(r"the Fermi energy is\s+([\d\.]+)\s+ev", scf_text, re.IGNORECASE)
    if match:
        fermi = float(match.group(1))
        print(f"Fermi energy: {fermi:.6f} eV")
        print(f"VBM - Fermi: {vbm - fermi:.6f} eV")
        print(f"CBM - Fermi: {cbm - fermi:.6f} eV")

if __name__ == '__main__':
    main()