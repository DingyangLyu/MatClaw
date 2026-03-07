# Materials Compute Skills Reference

MatClaw's container agent ships with a comprehensive library of computational materials science skills. Each skill is a `SKILL.md` file containing method selection guides, complete runnable Python/Bash scripts, key parameter tables, result interpretation guidance, and common issue troubleshooting.

**Total: 44 skill groups / 169 sub-skills / 213 SKILL.md files**

## Computation Environment

The container includes the following pre-installed tools:

| Engine | Description |
|--------|-------------|
| Quantum ESPRESSO 7.5 | Plane-wave DFT: `pw.x`, `ph.x`, `pp.x`, `bands.x`, `dos.x`, `projwfc.x`, etc. |
| LAMMPS | Classical molecular dynamics with OpenKIM potential support |
| RASPA3 | Grand Canonical Monte Carlo for gas adsorption in porous materials |
| MACE | Universal machine learning interatomic potential (via ASE) |
| pymatgen | Crystal structure manipulation, phase diagrams, electronic structure analysis |
| ASE | Atomic Simulation Environment: calculators, optimization, MD |
| phonopy | Phonon dispersion, thermal properties, Gruneisen parameters |
| BoltzTraP2 | Boltzmann transport equation solver |
| spglib | Space group and symmetry analysis |
| mp-api | Materials Project database access |
| Miniconda | Flexible package management for additional tools |

## Skill Organization

Skills are organized hierarchically: each **skill group** has a top-level `SKILL.md` with an overview, method decision guide, and sub-skill table. Each **sub-skill** has its own `SKILL.md` with complete, self-contained computation workflows.

### SKILL.md Format

Every sub-skill follows a consistent format:

1. **When to Use** — problem types this skill addresses
2. **Method Selection** — decision tree choosing between MACE (fast screening) and QE/VASP DFT (publication quality)
3. **Prerequisites** — structure, pseudopotentials, packages
4. **Detailed Steps** — complete runnable Python/Bash scripts (typically 200–500 lines)
5. **Key Parameters** — table with parameter name, typical value, and description
6. **Interpreting Results** — what the output numbers mean physically
7. **Common Issues** — troubleshooting table with problem, cause, and fix

---

## Complete Skill Inventory

### 1. Electronic Structure (8 sub-skills)

Core DFT calculations: self-consistent field, structural relaxation, band structure, and density of states.

| Sub-Skill | Description |
|-----------|-------------|
| `scf-relax` | Ground-state energy and ionic relaxation with QE `pw.x` or ASE+MACE |
| `band-structure` | Electronic band structure along high-symmetry k-paths |
| `density-of-states` | Total density of states (DOS) |
| `projected-dos` | Atom- and orbital-projected DOS (PDOS) for bonding character analysis |
| `spatially-resolved-dos` | Local DOS resolved in real space for interface/surface studies |
| `inverse-participation-ratio` | Wavefunction localization metric for disordered/defective systems |
| `convergence-testing` | Systematic ENCUT and k-point convergence studies |
| `vasp-bands` | VASP-specific band structure workflow |

### 2. Band Structure — Advanced (3 sub-skills)

Beyond-standard band structure techniques for challenging electronic structure problems.

| Sub-Skill | Description |
|-----------|-------------|
| `hybrid-dft-bands` | HSE06/PBE0 hybrid functional bands for accurate band gaps |
| `band-unfolding` | Recover effective primitive-cell bands from supercell calculations (alloys, defects) |
| `3d-band-structure` | Full Brillouin zone dispersion visualization for 2D/surface systems |

### 3. Advanced Electronic (5 sub-skills)

Beyond-GGA methods for strong correlation, spin-orbit, topology, and quasiparticle physics.

| Sub-Skill | Description |
|-----------|-------------|
| `gw-approximation` | GW quasiparticle band structure for accurate band gaps and optical spectra |
| `hubbard-u` | DFT+U for localized d/f-electron systems (transition metal oxides, rare earths) |
| `spin-orbit-coupling` | Fully relativistic SOC for heavy elements, topological materials, Rashba splitting |
| `topological-invariants` | Z2, mirror Chern numbers, and symmetry indicators from DFT wavefunctions |
| `van-der-waals` | DFT-D3, vdW-DF for layered materials, molecular crystals, adsorption |

### 4. Mechanical Properties (5 sub-skills)

Elastic constants, bulk/shear/Young's modulus, and equation of state.

| Sub-Skill | Description |
|-----------|-------------|
| `elastic-constants` | Full elastic tensor C_ij via stress-strain or energy-strain methods |
| `energy-strain-method` | Elastic constants from polynomial fitting of energy vs. strain curves |
| `stress-strain-method` | Elastic constants from stress response to applied strains |
| `equation-of-state` | E(V) fitting (Birch-Murnaghan, Vinet) for bulk modulus and equilibrium volume |
| `angular-mechanics` | Directional Young's modulus and Poisson ratio visualization |

### 5. Thermal Properties (11 sub-skills)

Phonons, molecular dynamics, thermodynamic functions, and thermal transport.

| Sub-Skill | Description |
|-----------|-------------|
| `phonon` | Phonon dispersion and DOS via finite displacement (phonopy + QE/MACE) |
| `phonon-from-outcar` | Extract phonon frequencies from VASP OUTCAR (Gamma-point) |
| `gruneisen-qha` | Quasi-harmonic approximation: Gruneisen parameters, thermal expansion, C_p(T), F(T) |
| `anharmonicity` | Anharmonicity quantification via σ^A score (one-shot rattled supercells) |
| `molecular-dynamics` | NVT/NPT molecular dynamics with ASE+MACE or LAMMPS |
| `md-trajectory-tools` | Trajectory I/O, frame extraction, format conversion |
| `rdf-analysis` | Radial distribution function g(r) from MD trajectories |
| `msd-diffusion` | Mean square displacement and diffusion coefficient extraction |
| `vacf-vdos` | Velocity autocorrelation function and vibrational DOS from MD |
| `bond-distribution` | Bond length/angle distributions from MD trajectories |
| `thermal-conductivity` | Lattice thermal conductivity via Green-Kubo or NEMD |

### 6. Thermal Conductivity (1 sub-skill)

| Sub-Skill | Description |
|-----------|-------------|
| `lattice-thermal-conductivity` | Phonon Boltzmann transport for lattice κ (ShengBTE / phono3py) |

### 7. Defects and Reactions (12 sub-skills)

Point defects, migration barriers, transition states, surface chemistry, and non-radiative recombination.

| Sub-Skill | Description |
|-----------|-------------|
| `vacancy-formation` | Vacancy formation energy with supercell convergence |
| `interstitial-defect` | Interstitial formation energy (split, tetrahedral, octahedral configurations) |
| `substitution-defect` | Substitutional impurity formation energy and site preference |
| `point-defect` | General point defect workflow with chemical potential references |
| `defect-thermodynamics` | Defect concentration vs. temperature/Fermi level (formation energy diagrams) |
| `neb-transition-state` | Nudged elastic band for migration barriers (ASE+MACE or QE `neb.x`) |
| `migration-barrier` | Ion migration pathway and activation energy |
| `reaction-pathway` | Multi-step reaction energy profiles |
| `surface-adsorption` | Slab models, adsorption site identification, adsorption energies |
| `adsorption-energy` | Molecular adsorption energy on surfaces |
| `surface-energy` | Surface formation energy for different Miller indices |
| `configuration-coordinate` | Configuration coordinate diagram: ΔQ, ZPL, Huang-Rhys factor, carrier capture barrier |

### 8. Ferroelectric (5 sub-skills)

Berry phase polarization, Born effective charges, and ferroelectric switching.

| Sub-Skill | Description |
|-----------|-------------|
| `polarization` | Spontaneous polarization via Berry phase (modern theory of polarization) |
| `born-effective-charge` | Born effective charge tensors Z* for polar instability analysis |
| `dielectric-tensor` | Static and high-frequency dielectric tensors (DFPT) |
| `ferroelectric-switching` | Polarization switching path, double-well energy profile, coercive field estimate |
| `piezoelectric` | Piezoelectric stress/strain tensors e_ij / d_ij |

### 9. Piezoelectric (1 sub-skill)

| Sub-Skill | Description |
|-----------|-------------|
| `piezoelectric-tensor` | Full piezoelectric tensor via DFPT or finite-field method |

### 10. Optical Properties (6 sub-skills)

Frequency-dependent dielectric function and derived optical spectra.

| Sub-Skill | Description |
|-----------|-------------|
| `dielectric-function` | ε₁(ω) + iε₂(ω) from interband transitions |
| `absorption-spectrum` | Optical absorption coefficient α(ω) |
| `optical-conductivity` | Optical conductivity σ(ω) |
| `joint-dos` | Joint density of states for optical transition analysis |
| `transition-dipole` | Transition dipole matrix elements for selection rule analysis |
| `slme` | Spectroscopic limited maximum efficiency for solar cell absorbers |

### 11. Magnetic Properties (3 sub-skills)

Magnetic ground state, exchange interactions, and magnetocrystalline anisotropy.

| Sub-Skill | Description |
|-----------|-------------|
| `spin-polarized` | Spin-polarized SCF for magnetic moment and spin density |
| `magnetic-ordering` | Enumerate FM/AFM configurations and determine ground state ordering |
| `magnetic-anisotropy` | Magnetocrystalline anisotropy energy from SOC calculations |

### 12. Topological (2 sub-skills)

Topological classification of electronic band structures.

| Sub-Skill | Description |
|-----------|-------------|
| `z2-invariant` | Z₂ topological invariant from parity eigenvalues or Wilson loop |
| `berry-curvature` | Berry curvature and anomalous Hall conductivity |

### 13. Electron-Phonon (4 sub-skills)

Electron-phonon coupling and its consequences for transport and superconductivity.

| Sub-Skill | Description |
|-----------|-------------|
| `elph-coupling` | Electron-phonon coupling matrix elements and Eliashberg function α²F(ω) |
| `deformation-potential` | Acoustic and optical deformation potentials for carrier scattering rates |
| `electronic-transport` | Electron mobility from electron-phonon scattering (EPA method) |
| `superconductivity` | Superconducting T_c from Allen-Dynes / McMillan formula |

### 14. Transport Properties (2 sub-skills)

Semiclassical electronic transport via Boltzmann theory.

| Sub-Skill | Description |
|-----------|-------------|
| `boltzmann-transport` | Seebeck coefficient, electrical conductivity σ/τ, power factor (BoltzTraP2) |
| `kpoints-transport` | Dense k-mesh generation for accurate transport integrals |

### 15. Bonding Analysis (10 sub-skills)

Chemical bonding characterization: charge partitioning, electron localization, and orbital-resolved bonding.

| Sub-Skill | Description |
|-----------|-------------|
| `bader-charge` | Bader (QTAIM) charge analysis for ionic/covalent character |
| `bader2pqr` | Convert Bader charges to PQR format for visualization |
| `charge-density` | 3D charge density ρ(r) plotting and isosurface generation |
| `charge-density-difference` | Δρ = ρ(AB) - ρ(A) - ρ(B) for bonding/antibonding visualization |
| `charge-format-conversion` | Convert between charge density file formats (CUBE, XSF, CHGCAR) |
| `elf-analysis` | Electron localization function for bond type identification |
| `lobster-cohp` | Crystal Orbital Hamilton Population for orbital-resolved bond strength |
| `orbital-projection` | Fat bands: band structure with orbital character coloring |
| `planar-charge` | Planar-averaged charge density along a direction |
| `stm-simulation` | Simulated STM images (Tersoff-Hamann approximation) |

### 16. Catalysis & Electrochemistry (5 sub-skills)

Thermodynamic corrections, reaction kinetics, and descriptor-based activity analysis.

| Sub-Skill | Description |
|-----------|-------------|
| `thermal-corrections` | Zero-point energy, entropy, and Gibbs free energy corrections to DFT energies |
| `reaction-kinetics` | Transition state theory rate constants from NEB barriers |
| `neb-analysis` | Post-processing of NEB results: barrier, reaction coordinate, MEP plots |
| `band-center` | d-band center and p-band center descriptors for catalytic activity |
| `imaginary-freq-correction` | Handle imaginary frequencies in transition state vibrational analysis |

### 17. Catalyst Screening (3 sub-skills)

High-throughput catalyst discovery using scaling relations and activity descriptors.

| Sub-Skill | Description |
|-----------|-------------|
| `d-band-center` | d-band center as descriptor for adsorption strength (Norskov model) |
| `overpotential` | OER/ORR/HER overpotential from adsorption free energies |
| `scaling-relations` | Linear scaling relations between adsorbate binding energies |

### 18. Battery & Electrode (2 sub-skills)

Intercalation voltage profiles and ion transport for battery electrode design.

| Sub-Skill | Description |
|-----------|-------------|
| `intercalation-voltage` | Average intercalation voltage from DFT total energies |
| `ion-diffusion` | Ion migration barriers and diffusion pathways in electrode materials |

### 19. Surface Energy (2 sub-skills)

Surface thermodynamics and equilibrium crystal shape prediction.

| Sub-Skill | Description |
|-----------|-------------|
| `surface-energy-calc` | Surface energy γ(hkl) for different Miller indices and terminations |
| `wulff-construction` | Equilibrium crystal shape from surface energy anisotropy |

### 20. Phase Diagram (2 sub-skills)

Thermodynamic and electrochemical stability analysis.

| Sub-Skill | Description |
|-----------|-------------|
| `convex-hull` | Convex hull construction for phase stability (formation energy above hull) |
| `pourbaix-diagram` | Electrochemical phase diagram as function of pH and potential |

### 21. Phase Transition (4 sub-skills)

Melting, amorphization, and solid-solid phase transitions.

| Sub-Skill | Description |
|-----------|-------------|
| `mpmorph-melting` | Melting point determination via coexistence MD (MPMorph protocol) |
| `amorphous-structure` | Melt-quench amorphous structure generation |
| `order-parameter` | Order parameters for phase transition characterization |
| `phase-diagram` | Temperature-pressure phase boundaries from free energy comparison |

### 22. Fermi Surface (3 sub-skills)

Fermi surface visualization for metals and semimetals.

| Sub-Skill | Description |
|-----------|-------------|
| `3d-fermi-surface` | 3D Fermi surface rendering from dense k-mesh eigenvalues |
| `2d-fermi-surface` | 2D Fermi surface cross-sections |
| `projected-fermi-surface` | Orbital-resolved Fermi surface coloring |

### 23. Spin Texture (2 sub-skills)

Spin-resolved electronic structure for Rashba/Dresselhaus and topological surface states.

| Sub-Skill | Description |
|-----------|-------------|
| `2d-spin-texture` | 2D spin texture ⟨S⟩(k) at constant energy |
| `3d-spin-texture` | 3D spin-polarized band structure |

### 24. Potential Analysis (3 sub-skills)

Electrostatic potential processing for work function and band alignment.

| Sub-Skill | Description |
|-----------|-------------|
| `work-function` | Work function from slab calculation (vacuum level - Fermi level) |
| `planar-average` | Planar-averaged electrostatic potential V(z) |
| `macroscopic-average` | Macroscopic average of potential for band offset determination |

### 25. Semiconductor Kit (4 sub-skills)

Essential semiconductor parameters from DFT band structures.

| Sub-Skill | Description |
|-----------|-------------|
| `band-gap` | Direct and indirect band gap extraction with VBM/CBM locations |
| `effective-mass` | Carrier effective masses m* from band curvature fitting |
| `angular-effective-mass` | Direction-dependent effective mass visualization |
| `fermi-velocity` | Fermi velocity for Dirac/Weyl semimetals and topological surface states |

### 26. Spectroscopy (2 sub-skills)

Vibrational and core-level spectra connecting simulation to experiment.

| Sub-Skill | Description |
|-----------|-------------|
| `raman-ir` | Raman and infrared spectra from DFPT (phonon mode intensities) |
| `xas-xanes` | X-ray absorption spectra (XAS/XANES) from core-level DFT |

### 27. Wannier Functions (1 sub-skill)

Maximally localized Wannier functions for tight-binding models and interpolation.

| Sub-Skill | Description |
|-----------|-------------|
| `wannier90-workflow` | MLWF construction: DFT → pw2wannier90 → Wannier90 → interpolated bands |

### 28. Wavefunction Analysis (2 sub-skills)

Real-space wavefunction visualization and symmetry classification.

| Sub-Skill | Description |
|-----------|-------------|
| `real-space-wavefunction` | Plot |ψ_nk(r)|² for selected bands/k-points |
| `wavefunction-parity` | Parity eigenvalue analysis at TRIM points for topological classification |

### 29. 2D Materials (4 sub-skills)

Tools specific to layered and 2D material systems.

| Sub-Skill | Description |
|-----------|-------------|
| `layer-manipulation` | Layer separation, extraction, and stacking for van der Waals heterostructures |
| `vacuum-resize` | Vacuum layer thickness optimization for slab/2D calculations |
| `stacking-energy` | Interlayer binding energy and sliding energy landscapes |
| `band-edges` | Band edge alignment for 2D semiconductor heterostructure design |

### 30. Alloy & Disorder (2 sub-skills)

Chemical disorder modeling in alloys and solid solutions.

| Sub-Skill | Description |
|-----------|-------------|
| `sqs-generation` | Special quasi-random structures for random alloy approximation |
| `cluster-expansion` | Cluster expansion fitting for alloy energetics and phase diagrams |

### 31. Interface (2 sub-skills)

Grain boundary and heterostructure interface modeling.

| Sub-Skill | Description |
|-----------|-------------|
| `grain-boundary` | Grain boundary structure construction and energy calculation |
| `heterostructure` | Lattice-matched heterostructure interface builder |

### 32. Structure Models (8 sub-skills)

Advanced structural model construction for simulation.

| Sub-Skill | Description |
|-----------|-------------|
| `supercell-builder` | Supercell generation with minimum image distance control |
| `surface-builder` | Slab model construction for arbitrary Miller indices |
| `defect-builder` | Point defect insertion (vacancy, interstitial, substitution) in supercells |
| `alloy-builder` | Random/ordered alloy structure generation |
| `heterostructure` | Lattice-matched interface construction |
| `moire-superlattice` | Moire pattern superlattice for twisted bilayer systems |
| `nanowire-nanotube` | 1D nanowire and nanotube model generation |
| `quantum-dot` | 0D quantum dot / nanoparticle cluster construction |

### 33. Structure Tools (7 sub-skills)

Foundational structure manipulation, symmetry, and format handling.

| Sub-Skill | Description |
|-----------|-------------|
| `format-conversion` | Convert between CIF, POSCAR, XSF, XYZ, QE input, etc. |
| `input-generation` | Generate complete QE/VASP input files from a crystal structure |
| `symmetry-analysis` | Space group detection, Wyckoff positions, symmetry operations |
| `structure-editing` | Modify atomic positions, lattice parameters, species |
| `structure-matching` | Compare structures accounting for symmetry (StructureMatcher) |
| `pdf-analysis` | Pair distribution function from structure or MD trajectory |
| `xrd-pattern` | Simulated X-ray diffraction pattern |

### 34. K-Path Utilities (5 sub-skills)

High-symmetry k-point path generation for band structure and phonon dispersion.

| Sub-Skill | Description |
|-----------|-------------|
| `bulk-kpath` | 3D bulk k-path via SeeKpath (Bravais lattice conventions) |
| `2d-kpath` | 2D k-path for layered/surface systems |
| `1d-kpath` | 1D k-path for chain/wire systems |
| `phonopy-kpath` | k-path formatted for phonopy band structure |
| `cp2k-kpath` | k-path formatted for CP2K input |

### 35. Code Interfaces (5 sub-skills)

Bridges between different computational codes and post-processing tools.

| Sub-Skill | Description |
|-----------|-------------|
| `vasp-qe-converter` | Convert between VASP and QE input/output formats |
| `phonopy-interface` | Interface QE/VASP force calculations with phonopy |
| `boltztrap-interface` | Prepare DFT data for BoltzTraP2 transport calculations |
| `wannier90-interface` | Set up QE → Wannier90 workflow |
| `ifc-analysis` | Interatomic force constant extraction and analysis |

### 36. DFT Corrections (3 sub-skills)

Method corrections for improved DFT accuracy.

| Sub-Skill | Description |
|-----------|-------------|
| `hubbard-u` | DFT+U setup for transition metal and rare earth systems |
| `spin-orbit-coupling` | SOC inclusion for heavy elements and topological materials |
| `vdw-correction` | van der Waals dispersion corrections (DFT-D3, vdW-DF) |

### 37. Monte Carlo (5 sub-skills)

GCMC simulations for gas adsorption in porous materials using RASPA3.

| Sub-Skill | Description |
|-----------|-------------|
| `gcmc-simulation` | Grand canonical MC for gas uptake at given T, P |
| `gas-adsorption` | Single-component gas adsorption isotherms |
| `gas-separation` | Multi-component mixture adsorption selectivity |
| `adsorption-isotherm` | Full adsorption/desorption isotherm generation |
| `pore-analysis` | Pore size distribution, surface area, pore volume characterization |

### 38. Materials Databases (2 sub-skills)

Query external databases for structures and properties.

| Sub-Skill | Description |
|-----------|-------------|
| `materials-project` | Materials Project API queries for structures, energies, band gaps, phase diagrams |
| `2d-semiconductors` | Computational 2D materials database access |

### 39. High-Throughput (7 sub-skills)

Automated screening and batch computation workflows.

| Sub-Skill | Description |
|-----------|-------------|
| `batch-calculations` | Run calculations over multiple structures in batch |
| `batch-screening` | Screen material candidates by computed properties |
| `materials-filtering` | Filter structures by composition, symmetry, property criteria |
| `screening-workflow` | End-to-end screening pipeline design |
| `phase-stability` | Automated thermodynamic stability assessment |
| `property-prediction` | Multi-property prediction workflow |
| `matpes-dual-static` | PBE → r²SCAN dual-functional static calculation (Materials Project protocol) |

### 40. MLIP Guide (4 sub-skills)

Machine learning interatomic potential usage and validation.

| Sub-Skill | Description |
|-----------|-------------|
| `universal-mlip` | Using MACE-MP-0, CHGNet, M3GNet for rapid structure screening |
| `mace-advanced` | Advanced MACE usage: fine-tuning, multi-head models, uncertainty |
| `mlip-validation` | Validate MLIP predictions against DFT reference data |
| `torchsim-batch` | TorchSim GPU-accelerated batch MD for thousands of structures |

### 41. Biomolecular MD (1 sub-skill)

Biomolecular simulations for proteins and solvated systems.

| Sub-Skill | Description |
|-----------|-------------|
| `openmm-simulation` | OpenMM MD: protein-ligand, solvation, AMBER/CHARMM force fields, free energy |

### 42. Molecular QChem (1 sub-skill)

Quantum chemistry for non-periodic molecular systems.

| Sub-Skill | Description |
|-----------|-------------|
| `gaussian-qchem-workflow` | PySCF/Psi4 molecular calculations: optimization, frequencies, thermochemistry, excited states |

### 43. Materials Compute (root skill)

Environment reference documenting all available computation engines, Python packages, and common workflow patterns. Not a calculation skill itself — serves as the foundation for all other skills.

### 44. Agent Browser (utility skill)

Browser automation tool for web-based tasks (data retrieval, literature search). Not a materials calculation skill.

---

## Coverage Summary

The skill library covers the following computational materials science domains:

| Domain | Skill Groups | Sub-Skills |
|--------|-------------|------------|
| Electronic structure & bands | electronic-structure, band-advanced, advanced-electronic, semiconductor-kit, fermi-surface, wavefunction-analysis | 28 |
| Mechanical properties | mechanical-properties | 5 |
| Thermal & phonon properties | thermal-properties, thermoconductivity | 12 |
| Defects & reactions | defects-reactions | 12 |
| Optical properties | optical-properties | 6 |
| Magnetic properties | magnetic-properties, spin-texture | 5 |
| Topological properties | topological | 2 |
| Ferroelectric & piezoelectric | ferroelectric, piezoelectric | 6 |
| Transport properties | transport-properties, electron-phonon | 6 |
| Catalysis & electrochemistry | catalysis-electrochem, catalyst-screening | 8 |
| Battery & electrode | battery-electrode | 2 |
| Surface & interface | surface-energy, interface, potential-analysis | 7 |
| Phase stability & transitions | phase-diagram, phase-transition | 6 |
| Bonding & charge analysis | bonding-analysis | 10 |
| Spectroscopy | spectroscopy | 2 |
| Wannier functions | wannier-functions | 1 |
| 2D materials | 2d-materials | 4 |
| Alloy & disorder | alloy-disorder | 2 |
| Structure modeling & tools | structure-models, structure-tools, kpath-utilities | 20 |
| Code interfaces | code-interfaces, dft-corrections | 8 |
| Monte Carlo | monte-carlo | 5 |
| Databases | materials-databases | 2 |
| High-throughput screening | high-throughput | 7 |
| Machine learning potentials | mlip-guide | 4 |
| Biomolecular simulation | biomolecular-md | 1 |
| Molecular quantum chemistry | molecular-qchem | 1 |

## Method Philosophy

Each skill supports a **dual-method approach** where applicable:

1. **MACE (via ASE)** — Fast ML-potential screening. Seconds to minutes per calculation. Use for rapid exploration, trend identification, and large-scale screening.
2. **Quantum ESPRESSO / VASP DFT** — Full first-principles accuracy. Minutes to hours. Use for publication-quality results, properties requiring explicit electronic structure (optical, magnetic, topological), and validation of MACE screening.

The decision tree in each skill's "Method Selection" section guides the agent to choose the appropriate level of theory for the task at hand.

## Source Verification

This skill library has been verified against:

- **atomate2** (materialsproject/atomate2) — all Maker workflow classes are covered
- **aiida-quantumespresso** (aiidateam/aiida-quantumespresso) — all QE workflow capabilities are covered
- **aiida-vasp** (aiida-vasp/aiida-vasp) — all VASP workchain capabilities are covered
- **VASPKIT** menu structure — most post-processing tasks (menus 01–92) have corresponding skills
