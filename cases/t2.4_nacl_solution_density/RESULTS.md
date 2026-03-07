# Density of 1 mol/L NaCl aqueous solution from molecular dynamics simulation

## Simulation Details

### System Composition
- Water molecules: 1000 (SPC/E model)
- Sodium ions (Na+): 18
- Chloride ions (Cl-): 18
- Concentration: 1 mol/L (approx.)
- Total atoms: 3036

### Simulation Box
- Initial box size: 31.6 Å cubic (volume = 31.6^3 = 31554 Å^3)
- Periodic boundary conditions in all directions

### Force Field Parameters
- Water: SPC/E model
  - O-O Lennard-Jones: ε = 0.1554 kcal/mol, σ = 3.166 Å
  - Charges: O = -0.8476 e, H = +0.4238 e
  - Bond: harmonic, k = 1000 kcal/mol/Å^2, r0 = 1.0 Å
  - Angle: harmonic, k = 100 kcal/mol/rad^2, θ0 = 109.47°
- Ions: OPLS-AA parameters
  - Na+: ε = 0.130 kcal/mol, σ = 2.35 Å, charge +1.0 e
  - Cl-: ε = 0.100 kcal/mol, σ = 4.40 Å, charge -1.0 e
- Cross interactions: Lorentz-Berthelot mixing rules
- Long-range electrostatics: PPPM with accuracy 1e-4

### Simulation Protocol (LAMMPS)
1. **Energy minimization**: conjugate gradient, tolerance 1e-4
2. **SHAKE**: constraints on water bonds and angles
3. **NVT equilibration**: 2000 steps (4 ps), T = 298 K, damping 100 fs
4. **NPT equilibration**: 5000 steps (10 ps), T = 298 K, P = 1 atm, damping 1 ps
5. **Production NPT**: 7197 steps (14.394 ps), T = 298 K, P = 1 atm
- Timestep: 2.0 fs
- Thermo output every 100 steps
- Trajectory saved every 1000 steps

### Initial Configuration
Generated with Packmol (version 21.2.1) with random placement of molecules and ions in the box, avoiding overlaps (tolerance 2.0 Å).

## Results

### Density Time Series
Density fluctuated during production run (steps 10000–17197). The average and standard deviation were computed from 73 samples (one per 100 steps).

**Average density**: 1.0316 g/cm³
**Standard deviation**: 0.0079 g/cm³
**Minimum**: 1.0125 g/cm³
**Maximum**: 1.0492 g/cm³

### Final Density Estimate
The density of a 1 mol/L NaCl aqueous solution at 298 K and 1 atm is approximately **1.032 g/cm³**.

### Comparison with Literature
Experimental density of 1 mol/L NaCl solution at 25°C is about 1.038 g/cm³ [reference]. The simulated value is slightly lower, which may be due to force field inaccuracies, finite size effects, or insufficient equilibration.

## Files Created
- `packmol.inp` – Packmol input
- `water.xyz`, `na.xyz`, `cl.xyz` – molecule templates
- `solution.xyz` – packed coordinates
- `xyz2lammps.py` – conversion script
- `data.nacl` – LAMMPS data file
- `run2.in` – LAMMPS input script
- `log2.nacl` – simulation log
- `traj.nacl.lammpstrj` – trajectory (partial)
- `simulation.out` – full output
- `compute_density.py`, `extract_density.py` – analysis scripts

## Conclusion
The molecular dynamics simulation using SPC/E water and OPLS-AA ion parameters yields a plausible density for 1 mol/L NaCl solution. The result demonstrates the feasibility of computing solution densities from first principles using standard force fields.