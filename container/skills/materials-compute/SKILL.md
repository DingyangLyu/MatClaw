# Materials Computation Environment

This container includes a full materials science computation environment. Use these tools for atomistic simulation tasks.

## Available Computation Engines

### Quantum ESPRESSO 7.5 (DFT)
- Binary: `pw.x` (also `ph.x`, `pp.x`, `bands.x`, `dos.x`, `projwfc.x`, etc.)
- Location: `/opt/qe/bin/`
- Use for: electronic structure, band gaps, density of states, phonons, elastic constants
- Run with MPI: `mpirun --allow-run-as-root -np N pw.x < input.in > output.out`
- Pseudopotentials download URL:
  ```
  https://pseudopotentials.quantum-espresso.org/upf_files/<FILENAME>.UPF
  ```
  Common pseudopotentials (PAW PBE from pslibrary):
  - Si: `Si.pbe-n-kjpaw_psl.1.0.0.UPF`
  - Cu: `Cu.pbe-dn-kjpaw_psl.1.0.0.UPF`
  - Al: `Al.pbe-n-kjpaw_psl.1.0.0.UPF`
  - Ni: `Ni.pbe-n-kjpaw_psl.1.0.0.UPF`
  - O:  `O.pbe-n-kjpaw_psl.1.0.0.UPF`
  - C:  `C.pbe-n-kjpaw_psl.1.0.0.UPF`
  - H:  `H.pbe-kjpaw_psl.1.0.0.UPF`
  - N:  `N.pbe-n-kjpaw_psl.1.0.0.UPF`
  - Fe: `Fe.pbe-spn-kjpaw_psl.1.0.0.UPF`
  - Ti: `Ti.pbe-spn-kjpaw_psl.1.0.0.UPF`
  - Zn: `Zn.pbe-dn-kjpaw_psl.1.0.0.UPF`
  - Ba: `Ba.pbe-spn-kjpaw_psl.1.0.0.UPF`
  - Li: `Li.pbe-s-kjpaw_psl.1.0.0.UPF`
  - Na: `Na.pbe-spn-kjpaw_psl.1.0.0.UPF`
  - K:  `K.pbe-spn-kjpaw_psl.1.0.0.UPF`

  Example download:
  ```bash
  wget -q https://pseudopotentials.quantum-espresso.org/upf_files/Si.pbe-n-kjpaw_psl.1.0.0.UPF -O Si.UPF
  ```
  Alternative sources: [SSSP](https://www.materialscloud.org/discover/sssp) or [PseudoDojo](http://www.pseudo-dojo.org/)

### LAMMPS (Molecular Dynamics)
- Binary: `lmp`
- Use for: MD simulations, thermal properties, diffusion, mechanical properties
- Supports OpenKIM potentials (pre-installed)
- Run: `lmp -in input.lammps`
- EAM potentials: use `pair_style eam/alloy` with potentials from OpenKIM or download from NIST
- For water: use SPC/E or TIP3P model with `pair_style lj/cut/coul/long`

### RASPA3 (Monte Carlo)
- Binary: `raspa3`
- Use for: gas adsorption in porous materials (MOFs, zeolites), adsorption isotherms, Henry constants
- Run: `cd /path/to/simulation && raspa3`
- Input format: JSON files (`simulation.json`, `force_field.json`, molecule JSON files)
- Official examples: `/usr/share/raspa3/examples/` (basic MC, adsorption, breakthrough, etc.)
- IMPORTANT: Always copy an official example as starting point and modify it:
  ```bash
  cp -r /usr/share/raspa3/examples/basic/1_mc_methane_in_box /tmp/my_sim
  cd /tmp/my_sim && raspa3
  ```

## Python Materials Science Stack

### Pre-installed in base conda environment:
- **pymatgen**: Crystal structure manipulation, phase diagrams, electronic structure analysis
- **ASE (Atomic Simulation Environment)**: Atoms objects, calculators, optimization, MD
- **mp-api**: Materials Project API access (needs API key)
- **MACE-torch**: Universal machine learning interatomic potential
- **spglib**: Space group analysis
- **torch**: PyTorch (CPU version)
- **numpy / scipy / matplotlib**: Scientific computing and visualization

### Conda/pip available:
The agent can install additional packages as needed:
```bash
# Create isolated environment for specific tasks
conda create -n myenv python=3.11 -y
conda activate myenv

# Install additional ML potentials
pip install chgnet sevenn

# Install workflow managers
pip install fireworks jobflow atomate2
```

## Common Workflows

### DFT Calculation (QE)
1. Prepare structure (use pymatgen to read CIF/POSCAR and generate QE input)
2. Download pseudopotentials
3. Run SCF calculation: `pw.x < scf.in > scf.out`
4. Post-process: bands, DOS, charge density, etc.

### MD Simulation (LAMMPS or ASE+MACE)
1. Prepare structure and force field
2. Set up LAMMPS input or ASE calculator
3. Run simulation
4. Analyze trajectory: RDF, MSD, thermal conductivity, etc.

### MLIP Calculation (ASE + MACE)
```python
from ase.io import read
from mace.calculators import mace_mp
calc = mace_mp(model="medium", device="cpu")
atoms = read("structure.cif")
atoms.calc = calc
energy = atoms.get_potential_energy()
forces = atoms.get_forces()
```

### Monte Carlo (RASPA3)
1. Prepare framework structure (CIF)
2. Configure simulation input (guest molecules, temperature, pressure)
3. Run RASPA3
4. Analyze adsorption isotherm
