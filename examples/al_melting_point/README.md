# Aluminum Melting Point

**Method:** MD | **Engine:** LAMMPS

## Prompt

```
Calculate the melting point of aluminum.
You can use LAMMPS (binary: lmp) with EAM potentials.
You must run actual simulations — do NOT use mock or fake data.
```

## Feishu Chat

MatClaw uses the two-phase solid-liquid coexistence method (Laio-Parrinello), runs NPT simulations at multiple temperatures, and analyzes the results with bond-order parameters and phase fraction tracking:

<p align="center"><img src="feishu-chat-1.png" width="800"></p>

## Result

<p align="center"><img src="al_melting_analysis.png" width="500"></p>

| Property | Agent | Reference | Error |
|----------|-------|-----------|-------|
| Melting point | **850-880 K** | 933 K (exp.) | ~6% |
| Cohesive energy (300 K) | -3.535 eV/atom | -3.36 eV/atom | — |

The ~50-80 K underestimation is a known systematic bias of the Zhou 2004 EAM potential. The agent noted that MACE-MP-0 or ACE potentials can predict T_m within ~20 K of experiment.

## Key findings

- **8000-atom system** with two-phase solid-liquid coexistence (NPT, 60 ps)
- **Steinhardt q₆ bond-order** analysis confirms all final structures are liquid-dominated
- Temperatures tested: 880, 920, 960, 1000 K — solid phase melted at all temperatures
- True T_m for this potential lies below 880 K

## Parameters

- Potential: Zhou EAM (`Al_zhou.eam.alloy`)
- System: 8000 atoms FCC Al
- Ensemble: NPT (isotropic, 1 atm)
- Timestep: 1 fs, run length: 60 ps per temperature
