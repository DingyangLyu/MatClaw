<p align="center">
  <img src="assets/matclaw-logo.svg" alt="MatClaw" width="720">
</p>

<p align="center">
  <strong>Tell it what to compute. It writes the scripts, runs DFT/MD/MC/MLIP, and reports back.</strong>
</p>

<p align="center">
  <a href="README_zh.md">中文</a>&nbsp; · &nbsp;
  <img src="https://img.shields.io/badge/QE-7.5-4F46E5" alt="QE 7.5">&nbsp;
  <img src="https://img.shields.io/badge/LAMMPS-2021-7C3AED" alt="LAMMPS">&nbsp;
  <img src="https://img.shields.io/badge/RASPA3-3.0.16-0D9488" alt="RASPA3">&nbsp;
  <img src="https://img.shields.io/badge/MACE--MP--0-latest-D97706" alt="MACE">&nbsp;
  <img src="https://img.shields.io/badge/tests-passing-brightgreen" alt="Tests">
</p>

---

## What is MatClaw?

MatClaw is an **AI agent that autonomously performs materials science computations**. You describe a task in natural language — it writes Python/shell scripts, runs them inside an isolated Docker container equipped with a full computation stack, and returns the results.

```
You:   "Calculate the energy of a 2-atom silicon diamond cell using MACE-MP-0 medium model."

MatClaw: Writing si_energy.py...
         Running calculation...
         ✅ Total energy: -10.8248 eV (-5.4124 eV/atom)
         Forces: [0, 0, 0] eV/Å (equilibrium)
         Structure saved to si_diamond_primitive.xyz
```

No manual scripting. No input file debugging. Just results.

## Key Features

- **Autonomous computation** — Understands your task, writes code, executes it, analyzes output, retries on errors
- **All-in-one container** — QE 7.5, LAMMPS, RASPA3, MACE, pymatgen, ASE, PyTorch pre-installed and ready
- **Secure isolation** — Every computation runs in a disposable Docker container with filesystem isolation
- **Flexible LLM backend** — Works with Anthropic Claude, DeepSeek, or any Anthropic-compatible API
- **Multi-channel access** — Chat via WhatsApp, Telegram, Discord, Slack (via [NanoClaw](https://github.com/qwibitai/nanoclaw) skills)
- **Extensible** — Conda/pip available inside container; agent can install additional packages on-the-fly

## Computation Stack

| Engine | Version | Method | Use Cases |
|--------|---------|--------|-----------|
| [Quantum ESPRESSO](https://www.quantum-espresso.org/) | 7.5 | DFT | Electronic structure, band gaps, DOS, phonons, elastic constants |
| [LAMMPS](https://www.lammps.org/) | 2021 | MD | Thermal properties, diffusion, mechanical properties, phase transitions |
| [RASPA3](https://github.com/iRASPA/RASPA3) | 3.0.16 | MC | Gas adsorption in MOFs/zeolites, isotherms, Henry constants |
| [MACE-MP-0](https://github.com/ACEsuit/mace) | latest | MLIP | Universal ML potential, fast energy/force/stress predictions |

### Python Materials Science Stack

All pre-installed in conda base environment:

| Package | Purpose |
|---------|---------|
| [pymatgen](https://pymatgen.org/) | Crystal structures, phase diagrams, electronic structure analysis |
| [ASE](https://wiki.fysik.dtu.dk/ase/) | Atoms objects, calculators, optimization, molecular dynamics |
| [MACE-torch](https://github.com/ACEsuit/mace) | Universal ML interatomic potentials |
| [mp-api](https://materialsproject.org/) | Materials Project database access |
| [spglib](https://spglib.github.io/spglib/) | Space group / symmetry analysis |
| [PyTorch](https://pytorch.org/) | ML framework (CPU) |
| numpy, scipy, matplotlib, pandas, seaborn | Scientific computing & visualization |

## Quick Start

### Prerequisites

- Linux (Ubuntu 20.04+ recommended) or macOS
- [Docker](https://docs.docker.com/get-docker/)
- An Anthropic-compatible API key (Claude, DeepSeek, etc.)

### 1. Build the container

```bash
git clone https://gitee.com/baiyuan1/mat-claw.git
cd mat-claw
./container/build.sh
```

The first build takes ~10 minutes (compiles QE 7.5 from source). Subsequent builds use Docker cache.

### 2. Run a computation

```bash
echo '{
  "prompt": "Calculate the energy of bulk silicon using MACE-MP-0",
  "groupFolder": "test",
  "chatJid": "test@g.us",
  "isMain": false,
  "secrets": {
    "ANTHROPIC_API_KEY": "your-api-key",
    "ANTHROPIC_BASE_URL": "https://api.anthropic.com"
  }
}' | docker run -i -v ./workspace:/workspace/group matclaw-agent:latest
```

### 3. Full agent setup with messaging channels

```bash
npm install
# Configure a channel (see docs):
# - Feishu: docs/feishu-setup.md
# - Gmail: docs/gmail-setup.md
npm run dev
```

## Example Workflows

### DFT: Silicon SCF with Quantum ESPRESSO

```
"Run an SCF calculation for silicon using QE with PAW pseudopotentials.
 Use a 4×4×4 k-point grid and ecutwfc=30 Ry."
```

The agent will:
1. Download Si pseudopotential from QE repository
2. Generate `si_scf.in` with proper `&CONTROL`, `&SYSTEM`, `&ELECTRONS` blocks
3. Run `mpirun -np 2 pw.x < si_scf.in`
4. Parse output: total energy, convergence, forces
5. Report results

### MD: Copper with LAMMPS

```
"Simulate 500 atoms of FCC copper at 300K for 10ps using LJ potential in LAMMPS.
 Report the final temperature and total energy."
```

### MLIP: Fast energy screening with MACE

```
"Use MACE-MP-0 to calculate energies for Li, Na, K, Rb, Cs in BCC structure.
 Compare with experimental cohesive energies."
```

### MC: Methane adsorption with RASPA3

```
"Run a Grand Canonical Monte Carlo simulation of methane adsorption
 in a box at 300K and 1 atm using RASPA3."
```

## Architecture

```
┌──────────────────────────────────────────────────────┐
│  Host (Node.js)                                      │
│  ┌────────────┐  ┌──────────┐  ┌──────────────────┐ │
│  │  Channels   │→│  SQLite   │→│  Container Runner │ │
│  │  (WhatsApp, │  │  (msgs,   │  │  (spawns Docker  │ │
│  │  Telegram,  │  │  tasks,   │  │   containers)    │ │
│  │  Discord…)  │  │  state)   │  └────────┬─────────┘ │
│  └────────────┘  └──────────┘           │           │
└──────────────────────────────────────────┼───────────┘
                                           │ stdin/stdout JSON
┌──────────────────────────────────────────┼───────────┐
│  Container (Ubuntu 24.04)                │           │
│  ┌───────────────────────────────────────┘         │ │
│  │  Agent Runner (Claude Agent SDK)                │ │
│  │  ┌─────────────────────────────────────────┐    │ │
│  │  │  LLM ←→ Tool Use (bash, browser, MCP)  │    │ │
│  │  └─────────────────────────────────────────┘    │ │
│  │                                                  │ │
│  │  Computation Tools:                              │ │
│  │  ┌─────────┐ ┌────────┐ ┌───────┐ ┌──────┐     │ │
│  │  │ QE 7.5  │ │ LAMMPS │ │RASPA3 │ │ MLIP │     │ │
│  │  └─────────┘ └────────┘ └───────┘ └──────┘     │ │
│  │  ┌──────────────────────────────────────────┐   │ │
│  │  │ Python: pymatgen, ASE, torch, numpy, …   │   │ │
│  │  └──────────────────────────────────────────┘   │ │
│  └──────────────────────────────────────────────────┘ │
└──────────────────────────────────────────────────────┘
```

**How it works:**
1. User sends a natural language prompt (via stdin JSON or messaging channel)
2. Host orchestrator routes it to a fresh Docker container
3. Inside the container, Claude Agent SDK receives the prompt and iteratively:
   - Writes computation scripts (Python, shell, QE input files, LAMMPS scripts…)
   - Executes them via bash tool
   - Reads and analyzes output
   - Retries or adjusts if errors occur
4. Final results returned to user via stdout markers

## Configuration

### API Keys

MatClaw works with any Anthropic-compatible API. Pass credentials via stdin JSON:

```json
{
  "secrets": {
    "ANTHROPIC_API_KEY": "your-key",
    "ANTHROPIC_BASE_URL": "https://api.anthropic.com"
  }
}
```

**Tested providers:**

| Provider | Base URL | Notes |
|----------|----------|-------|
| [Anthropic](https://www.anthropic.com/) | `https://api.anthropic.com` | Claude models, recommended |
| [DeepSeek](https://www.deepseek.com/) | `https://api.deepseek.com/anthropic` | Cost-effective, tool_use support |

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `CONTAINER_RUNTIME` | `docker` | Container runtime (`docker`, `podman`, `nerdctl`) |
| `MAX_CONCURRENT_CONTAINERS` | `5` | Max parallel agent containers |
| `AGENT_TIMEOUT` | `300` | Agent execution timeout (seconds) |

## Testing

**Latest test results (all passing):**

| Test | Result | Detail |
|------|--------|--------|
| MACE energy | **-10.68 eV** | 2-atom Si diamond cell |
| QE SCF | **-93.44 Ry** | Si with PAW pseudopotentials, 4×4×4 k-grid |
| LAMMPS MD | **50 steps** | FCC Cu, LJ potential, NVE ensemble |
| RASPA3 MC | **98.66 kg/m³** | Methane in box, 300K |
| Python packages | **8/8 present** | pymatgen, ASE, MACE, torch, numpy, scipy, matplotlib, spglib |
| Agent (E2E) | **-10.82 eV** | Autonomous Si energy calculation |

## Built on NanoClaw

MatClaw is built on [NanoClaw](https://github.com/qwibitai/nanoclaw), a lightweight personal AI assistant framework by [qwibitai](https://github.com/qwibitai). NanoClaw provides the orchestrator architecture, channel system, container isolation, and skill framework. MatClaw extends it with a full materials science computation environment.

## Project Structure

```
matclaw/
├── src/                        # Host orchestrator
│   ├── index.ts                # Main loop: messages, agents, scheduling
│   ├── container-runner.ts     # Spawns isolated Docker containers
│   ├── db.ts                   # SQLite (messages, tasks, state)
│   ├── channels/               # Messaging channel registry
│   └── ...
├── container/
│   ├── Dockerfile              # Multi-stage build (QE builder + runtime)
│   ├── agent-runner/           # Claude Agent SDK runner (inside container)
│   └── skills/
│       ├── materials-compute/  # Computation engine documentation
│       └── agent-browser/      # Browser automation
└── groups/                     # Per-group isolated memory
```

## Roadmap

- [ ] GPU support (CUDA-enabled container for PyTorch/MACE)
- [ ] More MLIP models (CHGNet, SevenNet, ALIGNN)
- [ ] Workflow automation (multi-step calculation pipelines)
- [ ] Materials Project integration (query + compute workflows)
- [ ] Jupyter notebook generation for reproducibility

## Citation

If you use MatClaw in your research, please cite:

```bibtex
@software{matclaw2026,
  title  = {MatClaw: AI-Powered Autonomous Materials Science Agent},
  author = {Yuan Bai},
  year   = {2026},
  url    = {https://gitee.com/baiyuan1/mat-claw}
}
```

## License

[MIT](LICENSE)
