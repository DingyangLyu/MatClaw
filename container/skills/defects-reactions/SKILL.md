# Defects and Reactions

## Overview

This skill group covers calculations involving crystallographic defects and chemical reaction pathways in solid-state materials. Two main approaches are available:

1. **MACE (via ASE)** -- Fast ML-potential-based calculations. Good for rapid screening of defect formation energies, NEB barriers, and adsorption energies. Seconds to minutes per calculation on typical supercells.
2. **Quantum ESPRESSO (QE)** -- Full DFT. Required for publication-quality energetics, charged defect calculations, electronic structure at defect sites, and accurate chemical bonding at surfaces.

Both approaches follow workflows inspired by atomate2's defect, NEB, and adsorption flows: create the defect/surface structure, relax, compute relevant energies, and post-process thermodynamic quantities.

## Sub-Skills

| Sub-Skill | Directory | Description |
|---|---|---|
| Point Defects | `point-defect/` | Vacancy and interstitial creation, supercell convergence, formation energy with chemical potential references, finite-size corrections for charged defects |
| NEB Transition States | `neb-transition-state/` | Nudged Elastic Band calculations for migration barriers and transition states using ASE+MACE or QE neb.x |
| Surface Adsorption | `surface-adsorption/` | Slab generation, adsorption site identification, adsorption energy calculations, work function |
| Configuration Coordinate Diagram | `configuration-coordinate/` | CCD for non-radiative transitions: DeltaQ, ZPL, Franck-Condon shifts, Huang-Rhys factor, classical barrier for carrier capture |

## Method Decision Guide

```
What do you need?

Defect formation energy (neutral)?
  Quick screening --> ASE + MACE (point-defect/)
  Publication quality --> QE DFT (point-defect/)

Charged defect formation energy / transition levels?
  --> QE DFT required (point-defect/, need electrostatic corrections)

Migration barrier / reaction pathway?
  Quick estimate --> ASE + MACE NEB (neb-transition-state/)
  Accurate barrier --> QE NEB (neb-transition-state/)

Adsorption energy / surface chemistry?
  Quick screening --> ASE + MACE (surface-adsorption/)
  Publication quality --> QE DFT with slab model (surface-adsorption/)

Non-radiative recombination / luminescence quenching / carrier capture?
  Structural screening (DeltaQ) --> ASE + MACE (configuration-coordinate/)
  Publication quality (ZPL, barrier, Huang-Rhys) --> QE DFT (configuration-coordinate/)
```

## Common Prerequisites

- **Structure**: Start from a CIF, POSCAR, or Materials Project query. Use pymatgen for structure manipulation (supercells, defect creation, slab generation).
- **Pseudopotentials**: QE calculations need pseudopotential files (SSSP library recommended).
- **Python packages**: pymatgen, ASE, mace-torch, numpy, scipy, matplotlib are pre-installed. Install extras with `pip install pymatgen-analysis-defects pymatgen-diffusion` as needed.
- **Supercell sizes**: Defect and NEB calculations require supercells large enough to minimize periodic image interactions. Typical minimum: 3x3x3 for cubic, or at least 10 A between periodic images.
