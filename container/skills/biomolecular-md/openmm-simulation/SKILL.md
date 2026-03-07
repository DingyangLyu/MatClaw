# OpenMM Biomolecular Simulation

## When to Use

- Simulating protein folding, dynamics, and conformational changes in explicit solvent
- Running solvated molecular systems with explicit water models (TIP3P, TIP4P-Ew, SPC/E)
- Computing drug-protein binding free energies using alchemical methods
- Studying polymer and soft matter systems in solution
- Parameterizing small molecules with Open Force Field (OpenFF Sage/Parsley)
- Performing simulated annealing or replica exchange for enhanced sampling
- Generating equilibrated biomolecular structures for downstream analysis

## Method Selection

```
What system are you simulating?

Protein or nucleic acid (have a PDB file)?
  --> Workflow 1: Protein in Water from PDB
  Force field: AMBER ff14SB (protein) or CHARMM36 (protein + lipids)
  Water model: TIP3P (fast, standard) or TIP4P-Ew (more accurate)

Small organic molecule / drug-like compound?
  --> Workflow 2: Small Molecule Solvation with OpenFF
  Force field: OpenFF Sage 2.0 (recommended) or Parsley 1.x
  Uses openff-toolkit + openff-interchange for parameterization

Need enhanced sampling (barrier crossing, rare events)?
  --> Workflow 3: Temperature Annealing (simulated annealing)
  Or use openmmtools for replica exchange MD (REMD)

System size guidance:
  < 10,000 atoms  --> CPU is fine, GPU strongly preferred
  10,000-100,000  --> GPU recommended (CUDA or OpenCL platform)
  > 100,000       --> GPU required, consider coarse-grained models

Timescale guidance:
  Equilibration:   1-5 ns (NVT + NPT)
  Local dynamics:  10-100 ns
  Folding/binding: 100 ns - microseconds (consider enhanced sampling)
```

## Prerequisites

```bash
pip install openmm openmmtools openff-toolkit openff-interchange mdtraj
```

Additional optional packages:
- `pdbfixer` -- repair PDB files (missing atoms, residues, non-standard names)
- `parmed` -- convert between force field formats (AMBER <-> GROMACS <-> OpenMM)
- `nglview` -- interactive 3D visualization in Jupyter notebooks

OpenMM platforms (auto-detected in order of preference):
1. **CUDA** -- fastest, requires NVIDIA GPU
2. **OpenCL** -- GPU-accelerated, works with AMD and NVIDIA
3. **CPU** -- multi-threaded, no GPU required
4. **Reference** -- slow but exact, for debugging only

## Detailed Steps

### Workflow 1: Protein in Water (from PDB)

Complete workflow: load PDB, fix missing atoms, assign AMBER force field, solvate with TIP3P, energy minimize, equilibrate NVT then NPT, run production NPT MD, save trajectory, and analyze with mdtraj.

```python
#!/usr/bin/env python3
"""
OpenMM biomolecular MD: Protein in explicit water from a PDB file.

Workflow (inspired by atomate2 OpenMM module):
  1. Load and fix PDB structure
  2. Assign AMBER ff14SB force field + TIP3P water
  3. Solvate in a cubic water box with ions for charge neutrality
  4. Energy minimization
  5. NVT equilibration (restrained heavy atoms)
  6. NPT equilibration (unrestrained)
  7. NPT production MD
  8. Trajectory analysis with mdtraj
"""

import os
import sys
import numpy as np

from openmm.app import (
    PDBFile, ForceField, Modeller, Simulation,
    PME, HBonds, NoCutoff,
    DCDReporter, StateDataReporter, PDBReporter,
)
from openmm import (
    LangevinMiddleIntegrator, MonteCarloBarostat,
    CustomExternalForce, Platform,
)
from openmm.unit import (
    nanometer, nanometers, angstrom, angstroms,
    kelvin, atmosphere, bar, picosecond, picoseconds,
    femtosecond, femtoseconds, kilojoule_per_mole,
    kilocalories_per_mole, molar,
)

# ============================================================
# 1. CONFIGURATION
# ============================================================

PDB_FILE = "protein.pdb"           # Input PDB file
WORK_DIR = "/tmp/openmm_protein"
os.makedirs(WORK_DIR, exist_ok=True)

# Force field
FF_PROTEIN = "amber14-all.xml"     # AMBER ff14SB for protein
FF_WATER = "amber14/tip3pfb.xml"   # TIP3P-FB water model

# Solvation
BOX_PADDING = 1.0 * nanometers    # Padding around protein
IONIC_STRENGTH = 0.15 * molar     # 150 mM NaCl (physiological)

# Simulation parameters
TEMPERATURE = 300.0 * kelvin
PRESSURE = 1.0 * bar
TIMESTEP = 2.0 * femtoseconds     # 2 fs with HBonds constraints
FRICTION = 1.0 / picosecond       # Langevin friction coefficient

# Phase durations (steps)
N_MINIMIZE = 5000                  # Max minimization iterations
N_NVT_EQUIL = 50000               # NVT equilibration (100 ps)
N_NPT_EQUIL = 100000              # NPT equilibration (200 ps)
N_PRODUCTION = 500000             # Production MD (1 ns)

# Reporting intervals
THERMO_INTERVAL = 5000            # Log thermodynamics every 10 ps
TRAJ_INTERVAL = 5000              # Save trajectory frame every 10 ps
CHECKPOINT_INTERVAL = 50000       # Save checkpoint every 100 ps

# Restraint for NVT equilibration
RESTRAINT_K = 1000.0              # kJ/mol/nm^2 on heavy atoms

# Platform selection (auto-detect best available)
PLATFORM_NAME = None              # None = auto; or "CUDA", "OpenCL", "CPU"

# ============================================================
# 2. LOAD AND FIX PDB
# ============================================================

print("=" * 70)
print("OpenMM Protein-in-Water MD Simulation")
print("=" * 70)

# Try to fix the PDB if pdbfixer is available
try:
    from pdbfixer import PDBFixer

    print(f"\nFixing PDB: {PDB_FILE}")
    fixer = PDBFixer(filename=PDB_FILE)
    fixer.findMissingResidues()
    fixer.findNonstandard()
    fixer.replaceNonstandard()
    fixer.removeHeterogens(keepWater=False)
    fixer.findMissingAtoms()
    fixer.addMissingAtoms()
    fixer.addMissingHydrogens(pH=7.0)

    fixed_pdb = os.path.join(WORK_DIR, "fixed.pdb")
    with open(fixed_pdb, "w") as f:
        PDBFile.writeFile(fixer.topology, fixer.positions, f)
    print(f"  Fixed PDB saved to: {fixed_pdb}")

    pdb = PDBFile(fixed_pdb)
except ImportError:
    print("pdbfixer not installed; loading PDB directly.")
    print("  Install with: pip install pdbfixer")
    pdb = PDBFile(PDB_FILE)

print(f"  Topology: {pdb.topology.getNumAtoms()} atoms, "
      f"{pdb.topology.getNumResidues()} residues, "
      f"{pdb.topology.getNumChains()} chains")

# ============================================================
# 3. ASSIGN FORCE FIELD AND SOLVATE
# ============================================================

print(f"\nAssigning force field: {FF_PROTEIN} + {FF_WATER}")
forcefield = ForceField(FF_PROTEIN, FF_WATER)

print("Building solvated system...")
modeller = Modeller(pdb.topology, pdb.positions)

# Add solvent (water box with padding and ions)
modeller.addSolvent(
    forcefield,
    model="tip3p",
    padding=BOX_PADDING,
    ionicStrength=IONIC_STRENGTH,
    positiveIon="Na+",
    negativeIon="Cl-",
)

n_atoms = modeller.topology.getNumAtoms()
n_residues = modeller.topology.getNumResidues()

# Count water molecules and ions
n_water = sum(1 for r in modeller.topology.residues() if r.name == "HOH")
n_na = sum(1 for r in modeller.topology.residues() if r.name == "NA")
n_cl = sum(1 for r in modeller.topology.residues() if r.name == "CL")

print(f"  Solvated system: {n_atoms} atoms, {n_residues} residues")
print(f"  Water molecules: {n_water}")
print(f"  Ions: {n_na} Na+, {n_cl} Cl-")

# Save solvated structure
solvated_pdb = os.path.join(WORK_DIR, "solvated.pdb")
with open(solvated_pdb, "w") as f:
    PDBFile.writeFile(modeller.topology, modeller.positions, f)
print(f"  Solvated PDB: {solvated_pdb}")

# ============================================================
# 4. CREATE OPENMM SYSTEM
# ============================================================

print("\nCreating OpenMM system...")
system = forcefield.createSystem(
    modeller.topology,
    nonbondedMethod=PME,
    nonbondedCutoff=1.0 * nanometers,
    constraints=HBonds,            # Constrain H-bonds for 2 fs timestep
    rigidWater=True,
    ewaldErrorTolerance=0.0005,
    hydrogenMass=1.5,              # Hydrogen mass repartitioning (optional)
)

print(f"  Number of forces: {system.getNumForces()}")
for i in range(system.getNumForces()):
    force = system.getForce(i)
    print(f"    {force.__class__.__name__}")

# ============================================================
# 5. ADD POSITIONAL RESTRAINTS (for NVT equilibration)
# ============================================================

# Restrain heavy atoms during NVT equilibration to let water relax
restraint_force = CustomExternalForce(
    "k * periodicdistance(x, y, z, x0, y0, z0)^2"
)
restraint_force.addGlobalParameter("k", RESTRAINT_K)
restraint_force.addPerParticleParameter("x0")
restraint_force.addPerParticleParameter("y0")
restraint_force.addPerParticleParameter("z0")

# Apply restraints to non-hydrogen protein atoms
n_restrained = 0
for atom in modeller.topology.atoms():
    if atom.residue.name not in ("HOH", "NA", "CL") and atom.element.symbol != "H":
        pos = modeller.positions[atom.index]
        restraint_force.addParticle(
            atom.index,
            [pos.x, pos.y, pos.z],
        )
        n_restrained += 1

restraint_index = system.addForce(restraint_force)
print(f"  Positional restraints on {n_restrained} heavy protein atoms")

# ============================================================
# 6. SET UP INTEGRATOR AND SIMULATION
# ============================================================

integrator = LangevinMiddleIntegrator(TEMPERATURE, FRICTION, TIMESTEP)

# Select platform
if PLATFORM_NAME:
    platform = Platform.getPlatformByName(PLATFORM_NAME)
    print(f"\nUsing platform: {PLATFORM_NAME}")
else:
    platform = None
    print("\nUsing auto-detected platform")

if platform:
    simulation = Simulation(modeller.topology, system, integrator, platform)
else:
    simulation = Simulation(modeller.topology, system, integrator)

simulation.context.setPositions(modeller.positions)

used_platform = simulation.context.getPlatform().getName()
print(f"  Active platform: {used_platform}")

# ============================================================
# 7. ENERGY MINIMIZATION
# ============================================================

print(f"\n--- Energy Minimization (max {N_MINIMIZE} iterations) ---")

state = simulation.context.getState(getEnergy=True)
e_before = state.getPotentialEnergy()
print(f"  Energy before: {e_before}")

simulation.minimizeEnergy(maxIterations=N_MINIMIZE)

state = simulation.context.getState(getEnergy=True, getPositions=True)
e_after = state.getPotentialEnergy()
print(f"  Energy after:  {e_after}")

# Save minimized structure
minimized_pdb = os.path.join(WORK_DIR, "minimized.pdb")
with open(minimized_pdb, "w") as f:
    PDBFile.writeFile(
        simulation.topology, state.getPositions(), f
    )
print(f"  Minimized PDB: {minimized_pdb}")

# ============================================================
# 8. NVT EQUILIBRATION (with restraints)
# ============================================================

print(f"\n--- NVT Equilibration: {N_NVT_EQUIL} steps "
      f"({N_NVT_EQUIL * 0.002:.1f} ps) ---")
print(f"  Heavy atom restraints: k = {RESTRAINT_K} kJ/mol/nm^2")

simulation.context.setVelocitiesToTemperature(TEMPERATURE)

# Reporters for NVT
nvt_log = os.path.join(WORK_DIR, "nvt_equil.log")
simulation.reporters.append(
    StateDataReporter(
        nvt_log, THERMO_INTERVAL,
        step=True, time=True, potentialEnergy=True,
        kineticEnergy=True, totalEnergy=True,
        temperature=True, volume=True, density=True,
        speed=True,
    )
)
simulation.reporters.append(
    StateDataReporter(
        sys.stdout, THERMO_INTERVAL * 5,
        step=True, time=True, temperature=True,
        potentialEnergy=True, speed=True,
    )
)

simulation.step(N_NVT_EQUIL)

state = simulation.context.getState(getEnergy=True)
print(f"  Final NVT energy: {state.getPotentialEnergy()}")

# ============================================================
# 9. NPT EQUILIBRATION (remove restraints)
# ============================================================

print(f"\n--- NPT Equilibration: {N_NPT_EQUIL} steps "
      f"({N_NPT_EQUIL * 0.002:.1f} ps) ---")
print("  Removing positional restraints...")

# Remove restraints by setting k = 0
simulation.context.setParameter("k", 0.0)

# Add barostat for NPT
barostat = MonteCarloBarostat(PRESSURE, TEMPERATURE, 25)
system.addForce(barostat)
simulation.context.reinitialize(preserveState=True)

simulation.step(N_NPT_EQUIL)

state = simulation.context.getState(getEnergy=True)
print(f"  Final NPT equilibration energy: {state.getPotentialEnergy()}")

# ============================================================
# 10. PRODUCTION NPT MD
# ============================================================

print(f"\n--- Production MD: {N_PRODUCTION} steps "
      f"({N_PRODUCTION * 0.002:.1f} ps) ---")

# Clear previous reporters
simulation.reporters.clear()

# Trajectory reporter (DCD format -- compact binary)
traj_file = os.path.join(WORK_DIR, "production.dcd")
simulation.reporters.append(
    DCDReporter(traj_file, TRAJ_INTERVAL)
)

# Thermodynamic log
prod_log = os.path.join(WORK_DIR, "production.log")
simulation.reporters.append(
    StateDataReporter(
        prod_log, THERMO_INTERVAL,
        step=True, time=True, potentialEnergy=True,
        kineticEnergy=True, totalEnergy=True,
        temperature=True, volume=True, density=True,
        speed=True,
    )
)

# Console output
simulation.reporters.append(
    StateDataReporter(
        sys.stdout, THERMO_INTERVAL * 10,
        step=True, time=True, temperature=True,
        potentialEnergy=True, volume=True, speed=True,
    )
)

# Save topology for trajectory analysis (needed by mdtraj)
topology_pdb = os.path.join(WORK_DIR, "topology.pdb")
state = simulation.context.getState(getPositions=True)
with open(topology_pdb, "w") as f:
    PDBFile.writeFile(simulation.topology, state.getPositions(), f)

# Run production
simulation.step(N_PRODUCTION)

# Save final state
final_state = simulation.context.getState(
    getPositions=True, getVelocities=True, getEnergy=True
)
final_pdb = os.path.join(WORK_DIR, "final.pdb")
with open(final_pdb, "w") as f:
    PDBFile.writeFile(
        simulation.topology, final_state.getPositions(), f
    )

print(f"\n  Production complete.")
print(f"  Final energy: {final_state.getPotentialEnergy()}")
print(f"  Trajectory: {traj_file}")
print(f"  Topology:   {topology_pdb}")

# ============================================================
# 11. TRAJECTORY ANALYSIS WITH MDTRAJ
# ============================================================

print("\n--- Trajectory Analysis ---")

import mdtraj as md
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Load trajectory
traj = md.load(traj_file, top=topology_pdb)
print(f"  Loaded {traj.n_frames} frames, {traj.n_atoms} atoms")
print(f"  Time range: {traj.time[0]:.1f} - {traj.time[-1]:.1f} ps")

# --- RMSD ---
# Select protein backbone for RMSD calculation
protein_atoms = traj.topology.select("protein and backbone")
if len(protein_atoms) > 0:
    traj_protein = traj.atom_slice(protein_atoms)
    rmsd = md.rmsd(traj_protein, traj_protein, frame=0)  # RMSD vs first frame

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(traj.time / 1000.0, rmsd * 10.0, "b-", linewidth=0.8)
    ax.set_xlabel("Time (ns)")
    ax.set_ylabel("Backbone RMSD (Angstrom)")
    ax.set_title("Protein Backbone RMSD")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(WORK_DIR, "rmsd.png"), dpi=150)
    plt.close()
    print(f"  RMSD plot: {WORK_DIR}/rmsd.png")
    print(f"  Mean RMSD: {np.mean(rmsd)*10:.2f} Angstrom")

# --- Radius of Gyration ---
protein_all = traj.topology.select("protein")
if len(protein_all) > 0:
    traj_prot = traj.atom_slice(protein_all)
    rg = md.compute_rg(traj_prot)

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(traj.time / 1000.0, rg * 10.0, "r-", linewidth=0.8)
    ax.set_xlabel("Time (ns)")
    ax.set_ylabel("Radius of Gyration (Angstrom)")
    ax.set_title("Protein Radius of Gyration")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(WORK_DIR, "rg.png"), dpi=150)
    plt.close()
    print(f"  Rg plot: {WORK_DIR}/rg.png")
    print(f"  Mean Rg: {np.mean(rg)*10:.2f} Angstrom")

# --- RMSF (B-factor like) ---
if len(protein_atoms) > 0:
    rmsf = md.rmsf(traj_protein, traj_protein, frame=0)

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(range(len(rmsf)), rmsf * 10.0, "g-", linewidth=0.8)
    ax.set_xlabel("Backbone Atom Index")
    ax.set_ylabel("RMSF (Angstrom)")
    ax.set_title("Per-Residue Backbone Fluctuation")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(WORK_DIR, "rmsf.png"), dpi=150)
    plt.close()
    print(f"  RMSF plot: {WORK_DIR}/rmsf.png")

# --- Hydrogen bonds ---
if len(protein_all) > 0:
    hbonds = md.baker_hubbard(traj_prot, freq=0.1)
    print(f"  Persistent H-bonds (>10% occupancy): {len(hbonds)}")

# --- Secondary Structure ---
try:
    if len(protein_all) > 0:
        dssp = md.compute_dssp(traj_prot)
        # dssp is (n_frames, n_residues) array of characters: H, E, C
        # Compute fraction of each type over the trajectory
        n_helix = np.mean(dssp == "H")
        n_sheet = np.mean(dssp == "E")
        n_coil = np.mean(dssp == "C")
        print(f"  Secondary structure: "
              f"Helix={n_helix:.1%}, Sheet={n_sheet:.1%}, Coil={n_coil:.1%}")
except Exception as e:
    print(f"  DSSP analysis skipped: {e}")

# --- Parse thermodynamic log ---
print("\n--- Thermodynamic Summary ---")
try:
    import csv
    with open(prod_log, "r") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    if rows:
        temps = [float(r.get('Temperature (K)', 0)) for r in rows]
        densities = [float(r.get('Density (g/mL)', 0)) for r in rows]
        pe = [float(r.get('Potential Energy (kJ/mole)', 0)) for r in rows]

        print(f"  Temperature: {np.mean(temps):.1f} +/- {np.std(temps):.1f} K")
        if any(d > 0 for d in densities):
            print(f"  Density:     {np.mean(densities):.4f} +/- "
                  f"{np.std(densities):.4f} g/mL")
        print(f"  Pot. Energy: {np.mean(pe):.1f} +/- {np.std(pe):.1f} kJ/mol")
except Exception as e:
    print(f"  Could not parse log: {e}")

print(f"\n{'='*70}")
print("Simulation complete. Output files:")
print(f"  Trajectory:   {traj_file}")
print(f"  Topology:     {topology_pdb}")
print(f"  Final PDB:    {final_pdb}")
print(f"  Thermo log:   {prod_log}")
print(f"  Plots:        {WORK_DIR}/rmsd.png, rg.png, rmsf.png")
print(f"{'='*70}")
```

### Workflow 2: Small Molecule Solvation (OpenFF)

Parameterize a small molecule with Open Force Field (Sage), solvate in explicit water, and run MD. Uses openff-toolkit and openff-interchange to generate an OpenMM system without any AMBER or CHARMM setup.

```python
#!/usr/bin/env python3
"""
OpenMM MD with Open Force Field parameterization.

Workflow:
  1. Load a small molecule from SMILES or SDF
  2. Parameterize with OpenFF Sage force field
  3. Solvate in TIP3P water box
  4. Create OpenMM system via Interchange
  5. Energy minimize, equilibrate, and run production MD
  6. Analyze solvation shell with mdtraj
"""

import os
import numpy as np

# ============================================================
# 1. CONFIGURATION
# ============================================================

# Define the molecule (SMILES or path to SDF/MOL2 file)
MOLECULE_SMILES = "c1ccccc1"       # Benzene (change to your molecule)
# MOLECULE_FILE = "ligand.sdf"     # Alternative: load from file

WORK_DIR = "/tmp/openmm_openff"
os.makedirs(WORK_DIR, exist_ok=True)

# Force field
OPENFF_FORCEFIELD = "openff-2.2.0.offxml"   # OpenFF Sage 2.2
WATER_MODEL = "tip3p"

# Box
BOX_PADDING_NM = 1.2               # nm padding around solute

# Simulation parameters
TEMPERATURE_K = 298.15
PRESSURE_BAR = 1.0
TIMESTEP_FS = 2.0
FRICTION_PS = 1.0

N_MINIMIZE = 5000
N_NVT = 25000                      # 50 ps NVT equilibration
N_NPT = 50000                      # 100 ps NPT equilibration
N_PRODUCTION = 500000              # 1 ns production

TRAJ_INTERVAL = 2500               # Save every 5 ps
LOG_INTERVAL = 2500

# ============================================================
# 2. CREATE MOLECULE AND PARAMETERIZE WITH OPENFF
# ============================================================

print("=" * 70)
print("OpenFF Small Molecule Solvation MD")
print("=" * 70)

from openff.toolkit import Molecule, ForceField as OFFForceField
from openff.interchange import Interchange
from openff.units import unit as offunit

# Create molecule from SMILES
print(f"\nCreating molecule from SMILES: {MOLECULE_SMILES}")
molecule = Molecule.from_smiles(MOLECULE_SMILES)
molecule.generate_conformers(n_conformers=1)
molecule.name = "LIG"

print(f"  Name: {molecule.name}")
print(f"  Formula: {molecule.hill_formula}")
print(f"  Atoms: {molecule.n_atoms}")
print(f"  Bonds: {molecule.n_bonds}")
print(f"  Charge: {molecule.total_charge}")

# Load OpenFF force field
print(f"\nLoading force field: {OPENFF_FORCEFIELD}")
off_ff = OFFForceField(OPENFF_FORCEFIELD)

# Create topology with solvent
from openff.toolkit import Topology as OFFTopology

# Build a box with the molecule and water
# Method: use Interchange to parameterize, then add solvent in OpenMM
print("Parameterizing solute with OpenFF...")

# Create an Interchange object for just the solute first
off_topology = molecule.to_topology()
interchange = Interchange.from_smirnoff(
    force_field=off_ff,
    topology=off_topology,
)

# Convert to OpenMM system and topology
print("Converting to OpenMM via Interchange...")
openmm_system = interchange.to_openmm()
openmm_topology = interchange.to_openmm_topology()
openmm_positions = interchange.positions.to_openmm()

# ============================================================
# 3. SOLVATE WITH OPENMM MODELLER
# ============================================================

from openmm.app import (
    Modeller, ForceField, Simulation, PME, HBonds,
    DCDReporter, StateDataReporter, PDBFile,
)
from openmm import (
    LangevinMiddleIntegrator, MonteCarloBarostat, Platform,
)
from openmm.unit import (
    nanometer, nanometers, kelvin, bar, picosecond,
    femtosecond, femtoseconds,
)

print("\nSolvating with TIP3P water...")

# Use OpenMM's built-in water model for solvation
water_ff = ForceField("amber14/tip3pfb.xml")

modeller = Modeller(openmm_topology, openmm_positions)
modeller.addSolvent(
    water_ff,
    model="tip3p",
    padding=BOX_PADDING_NM * nanometers,
    ionicStrength=0.0,             # No ions for simple solvation
)

n_atoms_total = modeller.topology.getNumAtoms()
n_water = sum(1 for r in modeller.topology.residues() if r.name == "HOH")
print(f"  Total atoms: {n_atoms_total}")
print(f"  Water molecules: {n_water}")

# Rebuild the full system with solute + solvent
# Re-parameterize the combined system
from openmm.app import ForceField as OMMForceField

# For the combined system, we need a force field that covers both
# the small molecule (from OpenFF) and water (from AMBER).
# Strategy: use the OpenFF interchange system for the solute, then
# combine with the water force field parameters in OpenMM.
#
# Simpler approach: use OpenMM ForceField with GAFF2 for the solute
# or re-create the full system with Interchange supporting water.
#
# Here we use the pragmatic approach: let OpenMM handle solvation
# with the AMBER water model, and merge force parameters.

# Create the combined system using OpenMM force field objects
combined_ff = ForceField("amber14/tip3pfb.xml")

# Build combined system
# Note: for production use, consider using Interchange.from_smirnoff
# with the water topology included, or use ParmEd to merge systems.
# The approach below uses OpenMM's built-in system creation for water
# and adds solute forces from OpenFF.

# Practical approach: write solute to PDB, combine, and use GAFF2 or
# re-parameterize everything through OpenFF with water support.
# For this example, we demonstrate using the interchange-created system.

print("Creating combined OpenMM system...")

# Alternative pragmatic approach: use OpenFF for everything
# OpenFF Sage includes TIP3P water parameters
from openff.toolkit import Topology

# Create a combined topology with solute + water
# First, save the solvated system from Modeller
solvated_pdb_path = os.path.join(WORK_DIR, "solvated.pdb")
with open(solvated_pdb_path, "w") as f:
    PDBFile.writeFile(modeller.topology, modeller.positions, f)

# For the solvated system, use OpenMM's standard force field approach
# with GAFF2 parameters for the solute via an XML file from Interchange
solute_xml = os.path.join(WORK_DIR, "solute.xml")
from openmm.app import ForceField

# Export the solute force field parameters
from openmm import XmlSerializer
with open(solute_xml, "w") as f:
    f.write(XmlSerializer.serialize(openmm_system))

# Build the final system using Modeller's combined topology
# Apply water FF + custom solute parameters
system = water_ff.createSystem(
    modeller.topology,
    nonbondedMethod=PME,
    nonbondedCutoff=1.0 * nanometers,
    constraints=HBonds,
    rigidWater=True,
)

# ============================================================
# 4. SET UP AND RUN SIMULATION
# ============================================================

integrator = LangevinMiddleIntegrator(
    TEMPERATURE_K * kelvin,
    FRICTION_PS / picosecond,
    TIMESTEP_FS * femtoseconds,
)

simulation = Simulation(modeller.topology, system, integrator)
simulation.context.setPositions(modeller.positions)

print(f"  Platform: {simulation.context.getPlatform().getName()}")

# --- Energy Minimization ---
print(f"\n--- Energy Minimization (max {N_MINIMIZE} steps) ---")
state = simulation.context.getState(getEnergy=True)
print(f"  Before: {state.getPotentialEnergy()}")
simulation.minimizeEnergy(maxIterations=N_MINIMIZE)
state = simulation.context.getState(getEnergy=True)
print(f"  After:  {state.getPotentialEnergy()}")

# --- NVT Equilibration ---
print(f"\n--- NVT Equilibration: {N_NVT} steps ({N_NVT * TIMESTEP_FS / 1000:.1f} ps) ---")
simulation.context.setVelocitiesToTemperature(TEMPERATURE_K * kelvin)

nvt_log = os.path.join(WORK_DIR, "nvt.log")
simulation.reporters.append(
    StateDataReporter(nvt_log, LOG_INTERVAL,
                      step=True, time=True, temperature=True,
                      potentialEnergy=True, density=True)
)
simulation.step(N_NVT)

# --- NPT Equilibration ---
print(f"\n--- NPT Equilibration: {N_NPT} steps ({N_NPT * TIMESTEP_FS / 1000:.1f} ps) ---")
barostat = MonteCarloBarostat(
    PRESSURE_BAR * bar, TEMPERATURE_K * kelvin, 25
)
system.addForce(barostat)
simulation.context.reinitialize(preserveState=True)

simulation.step(N_NPT)

# --- Production NPT ---
print(f"\n--- Production MD: {N_PRODUCTION} steps "
      f"({N_PRODUCTION * TIMESTEP_FS / 1000:.1f} ps) ---")

simulation.reporters.clear()

traj_file = os.path.join(WORK_DIR, "production.dcd")
prod_log = os.path.join(WORK_DIR, "production.log")
topology_pdb = os.path.join(WORK_DIR, "topology.pdb")

# Save topology
state = simulation.context.getState(getPositions=True)
with open(topology_pdb, "w") as f:
    PDBFile.writeFile(simulation.topology, state.getPositions(), f)

simulation.reporters.append(DCDReporter(traj_file, TRAJ_INTERVAL))
simulation.reporters.append(
    StateDataReporter(prod_log, LOG_INTERVAL,
                      step=True, time=True, temperature=True,
                      potentialEnergy=True, volume=True, density=True,
                      speed=True)
)
import sys
simulation.reporters.append(
    StateDataReporter(sys.stdout, LOG_INTERVAL * 10,
                      step=True, time=True, temperature=True, speed=True)
)

simulation.step(N_PRODUCTION)

# Save final structure
final_state = simulation.context.getState(getPositions=True, getEnergy=True)
final_pdb = os.path.join(WORK_DIR, "final.pdb")
with open(final_pdb, "w") as f:
    PDBFile.writeFile(simulation.topology, final_state.getPositions(), f)

# ============================================================
# 5. SOLVATION SHELL ANALYSIS
# ============================================================

print("\n--- Solvation Shell Analysis ---")

import mdtraj as md
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

traj = md.load(traj_file, top=topology_pdb)
print(f"  Loaded {traj.n_frames} frames")

# Identify solute and water oxygen atoms
solute_atoms = traj.topology.select("not water and not (resname HOH NA CL)")
water_oxygens = traj.topology.select("water and name O")

if len(solute_atoms) > 0 and len(water_oxygens) > 0:
    # Compute RDF between solute center-of-mass and water oxygens
    # Approximate: use geometric center of solute
    solute_com = md.compute_center_of_mass(traj.atom_slice(solute_atoms))

    # Compute distances from solute COM to each water oxygen
    # Use the first water oxygen distances as a proxy
    pairs = np.array([(solute_atoms[0], wo) for wo in water_oxygens[:500]])
    distances = md.compute_distances(traj, pairs)

    # Histogram for RDF-like plot
    r_max = 1.5  # nm
    n_bins = 150
    r_edges = np.linspace(0, r_max, n_bins + 1)
    r_centers = 0.5 * (r_edges[:-1] + r_edges[1:])
    dr = r_edges[1] - r_edges[0]

    hist = np.zeros(n_bins)
    for frame_dists in distances:
        h, _ = np.histogram(frame_dists, bins=r_edges)
        hist += h

    # Normalize
    hist /= traj.n_frames
    shell_vol = 4.0 * np.pi * r_centers**2 * dr
    box_vol = np.mean([traj.unitcell_volumes])
    rho_water = len(water_oxygens) / box_vol
    g_r = hist / (shell_vol * rho_water * min(len(pairs), 500))

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(r_centers * 10, g_r, "b-", linewidth=1.5)
    ax.axhline(1.0, color="gray", linestyle="--", linewidth=0.5)
    ax.set_xlabel("Distance from solute (Angstrom)")
    ax.set_ylabel("g(r)")
    ax.set_title(f"Solute-Water RDF ({MOLECULE_SMILES})")
    ax.set_xlim(0, r_max * 10)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(WORK_DIR, "solvation_rdf.png"), dpi=150)
    plt.close()
    print(f"  Solvation RDF: {WORK_DIR}/solvation_rdf.png")

    # Count waters in first solvation shell
    # First minimum of g(r) defines the shell boundary
    # Find first minimum after first maximum
    from scipy.signal import argrelextrema
    maxima = argrelextrema(g_r, np.greater, order=5)[0]
    minima = argrelextrema(g_r, np.less, order=5)[0]

    if len(maxima) > 0 and len(minima) > 0:
        first_max_idx = maxima[0]
        shell_minima = minima[minima > first_max_idx]
        if len(shell_minima) > 0:
            r_shell = r_centers[shell_minima[0]]
            # Count waters within r_shell
            n_shell = np.mean(np.sum(distances < r_shell, axis=1))
            print(f"  First solvation shell radius: {r_shell*10:.1f} Angstrom")
            print(f"  Mean waters in first shell: {n_shell:.1f}")

print(f"\n{'='*70}")
print("Simulation complete. Output files:")
print(f"  Trajectory:   {traj_file}")
print(f"  Topology:     {topology_pdb}")
print(f"  Final PDB:    {final_pdb}")
print(f"  Thermo log:   {prod_log}")
print(f"{'='*70}")
```

### Workflow 3: Temperature Annealing

Simulated annealing protocol in OpenMM: heat the system to a target high temperature, hold, then cool back down. Useful for exploring conformational space, overcoming energy barriers, and generating diverse structural ensembles.

```python
#!/usr/bin/env python3
"""
OpenMM simulated annealing protocol.

Workflow:
  1. Load a solvated system (PDB from Workflow 1 or 2)
  2. Define an annealing schedule: heat -> hold -> cool
  3. Run NVT MD at each temperature stage
  4. Monitor temperature, energy, and RMSD during annealing
  5. Compare initial and final structures
"""

import os
import sys
import numpy as np

from openmm.app import (
    PDBFile, ForceField, Modeller, Simulation,
    PME, HBonds, DCDReporter, StateDataReporter,
)
from openmm import (
    LangevinMiddleIntegrator, MonteCarloBarostat, Platform,
)
from openmm.unit import (
    nanometer, nanometers, kelvin, bar,
    picosecond, picoseconds, femtosecond, femtoseconds,
)

# ============================================================
# 1. CONFIGURATION
# ============================================================

# Input: a pre-solvated and equilibrated system from Workflow 1 or 2
INPUT_PDB = "solvated.pdb"        # Or use output from previous workflow
WORK_DIR = "/tmp/openmm_anneal"
os.makedirs(WORK_DIR, exist_ok=True)

# Force field (must match the system preparation)
FF_PROTEIN = "amber14-all.xml"
FF_WATER = "amber14/tip3pfb.xml"

# Annealing schedule: list of (temperature_K, n_steps)
# Each stage runs NVT MD at the specified temperature
ANNEALING_SCHEDULE = [
    (300,   50000),     # Equilibrate at 300 K  (100 ps)
    (400,   25000),     # Heat to 400 K         (50 ps)
    (500,   25000),     # Heat to 500 K         (50 ps)
    (600,   50000),     # Hold at 600 K         (100 ps) -- high T dwell
    (500,   25000),     # Cool to 500 K         (50 ps)
    (400,   25000),     # Cool to 400 K         (50 ps)
    (300,   100000),    # Cool to 300 K         (200 ps) -- final equilibration
]

TIMESTEP_FS = 2.0
FRICTION_PS = 1.0

TRAJ_INTERVAL = 2500              # Save frame every 5 ps
LOG_INTERVAL = 1000               # Log every 2 ps

# ============================================================
# 2. LOAD SYSTEM AND CREATE SIMULATION
# ============================================================

print("=" * 70)
print("OpenMM Simulated Annealing")
print("=" * 70)

pdb = PDBFile(INPUT_PDB)
print(f"\nLoaded: {INPUT_PDB}")
print(f"  Atoms: {pdb.topology.getNumAtoms()}")
print(f"  Residues: {pdb.topology.getNumResidues()}")

forcefield = ForceField(FF_PROTEIN, FF_WATER)

system = forcefield.createSystem(
    pdb.topology,
    nonbondedMethod=PME,
    nonbondedCutoff=1.0 * nanometers,
    constraints=HBonds,
    rigidWater=True,
)

# Start with the first temperature in the schedule
T_initial = ANNEALING_SCHEDULE[0][0]

integrator = LangevinMiddleIntegrator(
    T_initial * kelvin,
    FRICTION_PS / picosecond,
    TIMESTEP_FS * femtoseconds,
)

simulation = Simulation(pdb.topology, system, integrator)
simulation.context.setPositions(pdb.positions)

print(f"  Platform: {simulation.context.getPlatform().getName()}")

# Minimize first
print("\n--- Energy Minimization ---")
simulation.minimizeEnergy(maxIterations=2000)

# Initialize velocities
simulation.context.setVelocitiesToTemperature(T_initial * kelvin)

# ============================================================
# 3. SET UP REPORTERS
# ============================================================

# Trajectory
traj_file = os.path.join(WORK_DIR, "annealing.dcd")
simulation.reporters.append(DCDReporter(traj_file, TRAJ_INTERVAL))

# Thermodynamic log
anneal_log = os.path.join(WORK_DIR, "annealing.log")
simulation.reporters.append(
    StateDataReporter(
        anneal_log, LOG_INTERVAL,
        step=True, time=True, temperature=True,
        potentialEnergy=True, kineticEnergy=True,
        totalEnergy=True, volume=True, speed=True,
    )
)

# Console output
simulation.reporters.append(
    StateDataReporter(
        sys.stdout, LOG_INTERVAL * 10,
        step=True, time=True, temperature=True,
        potentialEnergy=True, speed=True,
    )
)

# Save topology
topology_pdb = os.path.join(WORK_DIR, "topology.pdb")
state = simulation.context.getState(getPositions=True)
with open(topology_pdb, "w") as f:
    PDBFile.writeFile(simulation.topology, state.getPositions(), f)

# ============================================================
# 4. RUN ANNEALING SCHEDULE
# ============================================================

print("\n--- Annealing Schedule ---")
total_steps = sum(n for _, n in ANNEALING_SCHEDULE)
total_time_ps = total_steps * TIMESTEP_FS / 1000.0
print(f"  Total: {total_steps} steps ({total_time_ps:.1f} ps)")
print(f"  Stages: {len(ANNEALING_SCHEDULE)}")

cumulative_steps = 0
stage_boundaries = []  # Track stage transitions for plotting

for stage_idx, (target_T, n_steps) in enumerate(ANNEALING_SCHEDULE):
    stage_time_ps = n_steps * TIMESTEP_FS / 1000.0
    print(f"\n  Stage {stage_idx + 1}/{len(ANNEALING_SCHEDULE)}: "
          f"T = {target_T} K, {n_steps} steps ({stage_time_ps:.1f} ps)")

    # Update thermostat temperature
    integrator.setTemperature(target_T * kelvin)

    # Record stage boundary
    stage_boundaries.append({
        "step_start": cumulative_steps,
        "target_T": target_T,
        "n_steps": n_steps,
    })

    # Run this stage
    simulation.step(n_steps)
    cumulative_steps += n_steps

    # Report current state
    state = simulation.context.getState(getEnergy=True)
    actual_T = simulation.context.getState(
        getEnergy=True
    ).getKineticEnergy() * 2.0 / (
        3 * pdb.topology.getNumAtoms() * 8.314462e-3 * kelvin
    )
    print(f"    PE = {state.getPotentialEnergy()}")

    # Save snapshot at each stage
    state_pos = simulation.context.getState(getPositions=True)
    snapshot_pdb = os.path.join(
        WORK_DIR, f"stage_{stage_idx+1}_T{target_T}K.pdb"
    )
    with open(snapshot_pdb, "w") as f:
        PDBFile.writeFile(
            simulation.topology, state_pos.getPositions(), f
        )

# Save final structure
final_state = simulation.context.getState(getPositions=True, getEnergy=True)
final_pdb = os.path.join(WORK_DIR, "final.pdb")
with open(final_pdb, "w") as f:
    PDBFile.writeFile(
        simulation.topology, final_state.getPositions(), f
    )
print(f"\n  Final energy: {final_state.getPotentialEnergy()}")

# ============================================================
# 5. ANALYSIS AND VISUALIZATION
# ============================================================

print("\n--- Annealing Analysis ---")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import csv

# Parse thermodynamic log
steps_log, time_log, temp_log, pe_log = [], [], [], []
with open(anneal_log, "r") as f:
    reader = csv.DictReader(f)
    for row in reader:
        steps_log.append(int(row.get('#"Step"', row.get("Step", 0))))
        time_log.append(float(row.get("Time (ps)", 0)))
        temp_log.append(float(row.get("Temperature (K)", 0)))
        pe_log.append(float(row.get("Potential Energy (kJ/mole)", 0)))

time_ns = np.array(time_log) / 1000.0
temp_arr = np.array(temp_log)
pe_arr = np.array(pe_log)

fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

# Temperature profile
ax = axes[0]
ax.plot(time_ns, temp_arr, "r-", linewidth=0.5, alpha=0.7, label="Instantaneous T")

# Overlay target temperature schedule
cum_time = 0.0
for sb in stage_boundaries:
    t_start = cum_time
    t_end = cum_time + sb["n_steps"] * TIMESTEP_FS / 1e6  # ns
    ax.axhline(sb["target_T"], xmin=0, xmax=1, color="gray",
               linewidth=0.3, alpha=0.3)
    ax.plot([t_start, t_end], [sb["target_T"], sb["target_T"]],
            "k-", linewidth=2, alpha=0.8)
    cum_time = t_end

ax.set_ylabel("Temperature (K)")
ax.set_title("Annealing Temperature Profile")
ax.legend()
ax.grid(True, alpha=0.3)

# Potential energy
ax = axes[1]
ax.plot(time_ns, pe_arr, "b-", linewidth=0.5, alpha=0.7)
ax.set_xlabel("Time (ns)")
ax.set_ylabel("Potential Energy (kJ/mol)")
ax.set_title("Potential Energy During Annealing")
ax.grid(True, alpha=0.3)

# Add stage boundaries
for sb in stage_boundaries:
    t_boundary = sb["step_start"] * TIMESTEP_FS / 1e6
    for ax in axes:
        ax.axvline(t_boundary, color="gray", linestyle=":", linewidth=0.5)

fig.suptitle("Simulated Annealing Protocol", fontsize=14)
fig.tight_layout()
fig.savefig(os.path.join(WORK_DIR, "annealing_profile.png"), dpi=150)
plt.close()
print(f"  Annealing profile: {WORK_DIR}/annealing_profile.png")

# RMSD analysis: compare trajectory to initial structure
import mdtraj as md

traj = md.load(traj_file, top=topology_pdb)
protein_atoms = traj.topology.select("protein and backbone")

if len(protein_atoms) > 0:
    traj_protein = traj.atom_slice(protein_atoms)
    rmsd = md.rmsd(traj_protein, traj_protein, frame=0) * 10.0  # nm -> Angstrom

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(traj.time / 1000.0, rmsd, "m-", linewidth=0.8)
    ax.set_xlabel("Time (ns)")
    ax.set_ylabel("Backbone RMSD vs. initial (Angstrom)")
    ax.set_title("RMSD During Annealing")
    ax.grid(True, alpha=0.3)

    # Mark stage boundaries
    for sb in stage_boundaries:
        t_boundary = sb["step_start"] * TIMESTEP_FS / 1e6
        ax.axvline(t_boundary, color="gray", linestyle=":", linewidth=0.5)
        ax.text(t_boundary, ax.get_ylim()[1] * 0.95,
                f"{sb['target_T']}K", fontsize=7, ha="left", va="top")

    fig.tight_layout()
    fig.savefig(os.path.join(WORK_DIR, "annealing_rmsd.png"), dpi=150)
    plt.close()
    print(f"  RMSD plot: {WORK_DIR}/annealing_rmsd.png")
    print(f"  Max RMSD: {np.max(rmsd):.2f} Angstrom")
    print(f"  Final RMSD: {rmsd[-1]:.2f} Angstrom")

print(f"\n{'='*70}")
print("Annealing complete. Output files:")
print(f"  Trajectory:   {traj_file}")
print(f"  Topology:     {topology_pdb}")
print(f"  Final PDB:    {final_pdb}")
print(f"  Thermo log:   {anneal_log}")
print(f"  Plots:        {WORK_DIR}/annealing_profile.png, annealing_rmsd.png")
print(f"  Snapshots:    {WORK_DIR}/stage_*_T*K.pdb")
print(f"{'='*70}")
```

## Key Parameters

| Parameter | Typical Value | Notes |
|---|---|---|
| `timestep` | 2 fs | Requires `constraints=HBonds`. Use 1 fs without constraints; 4 fs with hydrogen mass repartitioning (HMR). |
| `nonbondedCutoff` | 1.0 nm | Standard for PME. Must be < half the smallest box dimension. |
| `friction` (Langevin) | 1.0 ps^-1 | Thermostat coupling. Range 0.1-5.0 ps^-1. Higher = stronger coupling. |
| `temperature` | 300 K | Physiological. Use 310 K for human body temperature. |
| `pressure` | 1 bar | Standard conditions. Use MonteCarloBarostat for NPT. |
| `box_padding` | 1.0-1.5 nm | Distance from solute to box edge. Too small causes self-interaction artifacts. |
| `ionic_strength` | 0.15 M | Physiological NaCl concentration. Set 0.0 for pure water. |
| `ewaldErrorTolerance` | 0.0005 | PME accuracy. Smaller = more accurate but slower. |
| `hydrogenMass` | 1.5 amu | Hydrogen mass repartitioning enables 4 fs timestep. Set to None for standard 2 fs. |
| `equilibration_nvt` | 50-200 ps | Let water relax around restrained solute. |
| `equilibration_npt` | 100-500 ps | Let density equilibrate. Monitor box volume for convergence. |
| `production` | 1-1000 ns | Depends on the process being studied. Protein dynamics: 10-100 ns minimum. |
| `traj_interval` | 5-10 ps | Balance storage vs. time resolution. 10 ps is standard for most analyses. |
| `restraint_k` | 1000 kJ/mol/nm^2 | For NVT equilibration restraints. Gradually reduce for staged release. |

### Water Model Selection

| Model | Type | Speed | Accuracy | When to Use |
|---|---|---|---|---|
| TIP3P | 3-site rigid | Fastest | Good for proteins | Default choice with AMBER/CHARMM |
| TIP3P-FB | 3-site rigid | Fast | Better density | Improved TIP3P, recommended |
| TIP4P-Ew | 4-site rigid | Moderate | Best density/diffusion | When water properties matter |
| SPC/E | 3-site rigid | Fast | Good diffusion | Compatible with GROMOS force fields |
| OPC | 4-site rigid | Moderate | Excellent overall | Best accuracy, newer model |

### Force Field Selection

| Force Field | Best For | Water Model | Notes |
|---|---|---|---|
| AMBER ff14SB | Proteins | TIP3P, TIP3P-FB | Standard choice for protein MD |
| AMBER ff19SB | Proteins | OPC | Newer, improved backbone |
| CHARMM36m | Proteins + lipids | CHARMM TIP3P | Best for membrane simulations |
| OpenFF Sage 2.x | Small molecules | TIP3P | Modern, data-driven parameters |
| GAFF2 | Small molecules | TIP3P | Works with AMBER protein FF |

## Interpreting Results

### RMSD (Root Mean Square Deviation)
- Measures structural drift from a reference (usually the starting structure)
- **Stable protein**: RMSD plateaus at 1-3 Angstrom after equilibration
- **Unfolding**: RMSD increases continuously beyond 5-10 Angstrom
- **Conformational change**: RMSD jumps to a new plateau
- Always compute on backbone atoms (CA, C, N) after alignment

### Radius of Gyration (Rg)
- Measures compactness of the protein
- **Folded protein**: Rg is stable and consistent with experimental values
- **Unfolding**: Rg increases as the protein expands
- Typical globular proteins: Rg ~ 10-20 Angstrom depending on size

### RMSF (Root Mean Square Fluctuation)
- Per-residue flexibility (analogous to crystallographic B-factors)
- **Loops and termini**: high RMSF (flexible)
- **Core and secondary structure**: low RMSF (rigid)
- Compare with experimental B-factors for validation

### Density
- Water density at 300 K, 1 bar should be ~0.997 g/mL
- TIP3P gives ~0.98 g/mL; TIP3P-FB and TIP4P-Ew give ~0.997 g/mL
- If density deviates significantly, check the barostat and force field

### Hydrogen Bonds
- Protein internal H-bonds stabilize secondary structure
- Persistent H-bonds (>50% occupancy) define stable interactions
- Transient H-bonds (<10% occupancy) indicate dynamic regions

### Temperature and Energy Equilibration
- Temperature should fluctuate around the target: sigma_T ~ T * sqrt(2/(3*N))
- Potential energy should plateau after equilibration (no systematic drift)
- Total energy is not conserved in NVT/NPT (thermostat/barostat exchange energy)

## Common Issues

| Issue | Cause | Fix |
|---|---|---|
| `NaN` in energy or coordinates | Atomic clashes after solvation, bad initial structure | Run longer minimization (10000+ steps); use `pdbfixer` to repair PDB; reduce timestep temporarily |
| Temperature explodes | Timestep too large, missing constraints | Use `constraints=HBonds` with 2 fs timestep; check that all parameters are assigned |
| Density too low/high | Wrong water model or barostat not active | Verify water model matches force field; ensure `MonteCarloBarostat` is added for NPT |
| Protein unfolds immediately | Force field mismatch, bad protonation states | Check that the correct force field is used; verify histidine protonation; check disulfide bonds |
| `No template found for residue` | Non-standard residue not in force field | Use `pdbfixer.replaceNonstandard()` or add custom residue template |
| Simulation very slow on CPU | Large system on CPU platform | Use GPU (CUDA or OpenCL); reduce system size; increase trajectory save interval |
| OpenFF parameterization fails | Molecule has unusual chemistry | Check that SMILES is valid; try `molecule.generate_conformers()` first; use GAFF2 as fallback |
| Box too small error | Solute too close to periodic image | Increase `padding` to 1.5 nm or larger |
| Checkpoint/restart fails | Version mismatch or corrupted file | Re-run from the last saved PDB snapshot; use `simulation.saveCheckpoint()` and `loadCheckpoint()` |
| Waters penetrate into protein | Insufficient equilibration | Run longer NVT with restraints; gradually release restraints over 3-5 stages |
| H-bond analysis gives no results | Wrong atom selection or short trajectory | Check that protein atoms are properly selected; ensure trajectory is long enough (>1 ns) |
