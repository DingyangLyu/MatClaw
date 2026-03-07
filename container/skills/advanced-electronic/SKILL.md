---
name: advanced-electronic
description: Advanced Electronic Structure Methods (5 sub-skills: gw-approximation, hubbard-u, spin-orbit-coupling, topological-invariants, van-der-waals)
---

# Advanced Electronic Structure Methods

Beyond-standard-DFT methods for accurate electronic properties in materials where standard GGA/LDA fails qualitatively. These methods address strong correlation, spin-orbit physics, topological classification, quasiparticle energies, and long-range dispersion interactions.

## Sub-Skills

| Sub-Skill | Directory | Use Case | Key QE / VASP Keywords |
|---|---|---|---|
| DFT+U (Hubbard U) | `hubbard-u/` | Strongly correlated d/f-electron systems, Mott insulators, transition metal oxides | `lda_plus_u`, `Hubbard_U(i)`, `HUBBARD` card, `hp.x`; VASP: `LDAU`, `LDAUU` |
| Spin-Orbit Coupling | `spin-orbit-coupling/` | Heavy elements, topological insulators, Rashba splitting, magnetic anisotropy, band inversion | `noncolin`, `lspinorb`, FR pseudopotentials; VASP: `LSORBIT`, `LNONCOLLINEAR` |
| Topological Invariants | `topological-invariants/` | Z2 classification, Wilson loops, Berry phase, Wannier charge centers, topological insulator screening | `z2pack`, `pw2wannier90.x`, Wannier90; parity eigenvalues at TRIM |
| GW Approximation | `gw-approximation/` | Accurate quasiparticle band gaps, band alignment, comparison with photoemission | Yambo / SternheimerGW interface for QE; VASP: `ALGO=GW0`, `ALGO=scGW` |
| Van der Waals Corrections | `van-der-waals/` | Layered materials, molecular crystals, adsorption, MOFs | `vdw_corr='dft-d3'`, `input_dft='rvv10'`; VASP: `IVDW`, `LUSE_VDW` |

## Decision Guide

```
Does your system have open d or f shells (TM oxides, lanthanides, actinides)?
  YES --> DFT+U (hubbard-u/)
  NO  --> Standard DFT may suffice for ground-state energetics

Does your system contain heavy elements (Z > 50) or involve spin-dependent phenomena?
  YES --> Spin-Orbit Coupling (spin-orbit-coupling/)
  NO  --> Scalar-relativistic pseudopotentials are sufficient

Do you need to classify topological order (Z2, Chern number, surface states)?
  YES --> Topological Invariants (topological-invariants/)
  NO  --> Standard band structure is sufficient

Do you need accurate band gaps (comparison with photoemission/optical data)?
  YES --> GW Approximation (gw-approximation/)
  NO  --> DFT gaps with scissors correction may suffice

Are van der Waals interactions important (layered, molecular, adsorption)?
  YES --> Van der Waals Corrections (van-der-waals/)
  NO  --> Standard DFT is fine
```

**These methods are NOT mutually exclusive.** Common combinations include:
- DFT+U + SOC for correlated topological materials (e.g., iridates, SmB6)
- SOC + topological invariants for TI classification (Bi2Se3, Bi2Te3)
- DFT+U + vdW for correlated layered materials (e.g., NiPS3, CrI3)
- GW + SOC for accurate gaps in heavy-element semiconductors

## Method Selection

| Material Class | Recommended Method(s) | Example Systems |
|---|---|---|
| Transition metal oxides | DFT+U | NiO, Fe2O3, LiFePO4, SrTiO3 |
| f-electron compounds | DFT+U (large U) | CeO2, UO2, SmB6 |
| Topological insulators | SOC + topological invariants | Bi2Se3, Bi2Te3, HgTe |
| Rashba systems | SOC | BiTeI, GeTe, oxide interfaces |
| Semiconductors (accurate gaps) | GW | Si, GaAs, ZnO, MgO |
| Layered materials | vdW corrections | graphite, MoS2, h-BN, black P |
| Molecular crystals | vdW corrections | organic semiconductors, ice |
| Correlated topological | DFT+U + SOC + Z2 | SmB6, pyrochlore iridates |
| 2D magnets | DFT+U + vdW + SOC | CrI3, Fe3GeTe2 |

## General Prerequisites

- Quantum ESPRESSO 7.5 installed (`pw.x`, `pp.x`, `hp.x`, `projwfc.x`, `bands.x`)
- For GW: Yambo or SternheimerGW (QE interface) or VASP with GW support
- Fully relativistic pseudopotentials for SOC calculations (PSlibrary `_rel` or ONCVPSP `_FR`)
- Wannier90 for topological invariant workflows
- Python packages: `pymatgen`, `ASE`, `numpy`, `scipy`, `matplotlib`, `z2pack`
