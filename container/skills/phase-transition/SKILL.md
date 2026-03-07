# Phase Transitions

Melting point determination, amorphous structure generation, and solid-solid phase transition analysis using molecular dynamics and energy-based methods.

## Sub-Skills

| Sub-Skill | Directory | Description |
|---|---|---|
| MPMorph Melting | `mpmorph-melting/` | Melting point determination via heating curves, Lindemann criterion, two-phase coexistence, and liquid structure analysis (ASE+MACE or LAMMPS) |
| Amorphous Structure | `amorphous-structure/` | Amorphous/glassy structure generation via melt-quench MD; structural analysis (RDF, coordination, bond angles, structure factor); glass transition temperature |

## Method Decision Guide

```
What phase transition property do you need?

Melting point / melting temperature?
  --> mpmorph-melting/  (heating curve or two-phase coexistence)

Liquid structure (RDF, diffusion, viscosity)?
  --> mpmorph-melting/  (high-temperature MD above Tm)

Amorphous / glassy structure generation?
  --> amorphous-structure/  (melt-quench protocol)

Glass transition temperature (Tg)?
  --> amorphous-structure/  (volume vs T during quench)

Structural analysis of disordered phase (RDF, coordination, bond angles)?
  --> amorphous-structure/  (post-processing tools)

Quick screening vs. publication accuracy?
  Quick --> ASE + MACE (both sub-skills support this)
  Publication --> LAMMPS with validated potential (mpmorph-melting/) or larger MACE supercells
```
