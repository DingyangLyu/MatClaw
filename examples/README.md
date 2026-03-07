# Examples

Benchmark tasks adapted from [QUASAR](https://github.com/fengxuyy/QUASAR). All tasks were executed autonomously by MatClaw — the agent wrote scripts, ran simulations, and reported results without human intervention.

## Basic Tasks

| Example | Method | Engine | Reference | Agent Result |
|---------|--------|--------|-----------|-------------|
| [Cu k-point convergence](cu_kpoint_convergence/) | DFT | Quantum ESPRESSO | < 1 meV/atom | Converged at 12x12x12 (4.3 meV/atom) |
| [Water density](water_density/) | MD | LAMMPS | 0.997 g/cm³ | 0.985 ± 0.010 g/cm³ |
| [IRMOF-1 void fraction](irmof1_void_fraction/) | MC | RASPA3 | 0.7988 | 0.8025 ± 0.0005 |

## Workflow Orchestration

| Example | Method | Engine | Reference | Agent Result |
|---------|--------|--------|-----------|-------------|
| [NiO band gap](nio_bandgap/) | DFT | Quantum ESPRESSO | 4.0 eV (exp.) | Metallic (PBE); needs DFT+U |
| [CO₂ in UiO-66](co2_uio66_adsorption/) | MC | RASPA3 | 5.98 mmol/g @10bar | 5.477 ± 0.019 mmol/g |
| [Al melting point](al_melting_point/) | MD | LAMMPS | 933 K | ~920-950 K (EAM estimate) |
| [NaCl solution density](nacl_solution_density/) | MD | LAMMPS | 1.038 g/cm³ | 1.032 ± 0.008 g/cm³ |

## How to run

Send the prompt to MatClaw via any configured channel (Feishu, Gmail, etc.), or run directly:

```bash
echo '{
  "prompt": "<paste prompt from example README>",
  "groupFolder": "test",
  "chatJid": "test@g.us",
  "isMain": false,
  "secrets": {
    "ANTHROPIC_API_KEY": "your-api-key",
    "ANTHROPIC_BASE_URL": "https://api.anthropic.com"
  }
}' | docker run -i -v ./workspace:/workspace/group matclaw-agent:latest
```
