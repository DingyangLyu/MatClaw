---
name: wavefunction-analysis
description: Wavefunction Analysis (2 sub-skills: real-space-wavefunction, wavefunction-parity)
---

# Wavefunction Analysis

## Overview

This skill group covers real-space visualization and symmetry analysis of Kohn-Sham
wavefunctions from first-principles DFT calculations. These workflows are essential for
understanding orbital character, bonding topology, and topological classification of
electronic states.

All workflows correspond to VASPKIT menu 51 (tasks 511--516).

## Sub-Skills

| Sub-Skill | Directory | Description |
|---|---|---|
| Real-Space Wavefunction | `real-space-wavefunction/` | Plot wavefunctions (|psi|^2) in real space for selected bands and k-points. QE: pp.x with plot_num=7. VASP: PARCHG. Output in cube, xsf formats. Matplotlib slice visualization. |
| Wavefunction Parity | `wavefunction-parity/` | Determine parity eigenvalues of wavefunctions at TRIM points for Z2 topological invariant classification (Fu-Kane method). |

## When to Use

- **real-space-wavefunction/** -- You need to visualize the spatial distribution of specific
  electronic states (e.g., defect levels, band-edge states, surface states).
- **wavefunction-parity/** -- You need parity eigenvalues at time-reversal invariant momentum
  (TRIM) points for topological classification via the Fu-Kane formula.

## General Prerequisites

- Quantum ESPRESSO 7.5 (`pw.x`, `pp.x`, `projwfc.x`)
- Python packages: `pymatgen`, `ase`, `numpy`, `matplotlib`
- Converged SCF calculation with the target material
- For parity analysis: system must have inversion symmetry; SOC requires fully relativistic pseudopotentials
