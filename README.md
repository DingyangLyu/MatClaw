<p align="center">
  <img src="assets/matclaw-logo.svg" alt="MatClaw" width="720">
</p>

<p align="center">
  <strong>Describe a materials problem in plain language. MatClaw writes the code, runs the simulation, and delivers results.</strong>
</p>

<p align="center">
  <a href="README_draft_zh.md">中文</a>&nbsp; · &nbsp;
  <a href="LICENSE"><img src="https://img.shields.io/badge/license-MIT-blue.svg" alt="MIT License"></a>&nbsp;
  <img src="https://img.shields.io/badge/Docker-24.04-2496ED?logo=docker&logoColor=white" alt="Docker">&nbsp;
  <img src="https://img.shields.io/badge/Node.js-20+-339933?logo=nodedotjs&logoColor=white" alt="Node.js 20+">&nbsp;
  <img src="https://img.shields.io/badge/Python-3.11-3776AB?logo=python&logoColor=white" alt="Python 3.11">
</p>

<p align="center">
  <img src="https://img.shields.io/badge/QE-7.5-4F46E5" alt="QE 7.5">&nbsp;
  <img src="https://img.shields.io/badge/LAMMPS-2021-7C3AED" alt="LAMMPS">&nbsp;
  <img src="https://img.shields.io/badge/RASPA3-3.0.16-0D9488" alt="RASPA3">
</p>

---

## Contents

- [What is MatClaw?](#what-is-matclaw)
- [Basic Usage](#basic-usage)
- [Examples](#examples)
- [Computation Stack](#computation-stack)
- [Quick Start](#quick-start)
- [Architecture](#architecture)
- [Configuration](#configuration)
- [Documentation](#documentation)
- [Citation](#citation)

## What is MatClaw?

MatClaw is an **AI agent that autonomously performs materials science computations**. You describe a task in natural language — it writes Python/shell scripts, runs them inside an isolated Docker container equipped with a full computation stack, and returns the results.

<p align="center"><img src="examples/water_density/feishu-chat-1.png" width="800"></p>

<p align="center"><em>Send a task via Feishu (or any channel). Get back scripts, simulations, plots, and analysis — no manual coding required.</em></p>

**Key Features:**

- **Autonomous computation** — Understands your task, writes code, executes it, analyzes output, retries on errors
- **All-in-one container** — QE 7.5, LAMMPS, RASPA3, MACE, pymatgen, ASE, PyTorch pre-installed and ready
- **Secure isolation** — Every computation runs in a disposable Docker container with filesystem isolation
- **Flexible LLM backend** — Works with Anthropic Claude, DeepSeek, or any Anthropic-compatible API
- **Multi-channel access** — Chat via Feishu, Gmail, WhatsApp, Telegram, Discord, Slack (via [NanoClaw](https://github.com/qwibitai/nanoclaw) skill system)
- **Extensible** — Conda/pip available inside container; agent can install additional packages on-the-fly

## Basic Usage

Send a natural language task via any connected channel (Feishu, WhatsApp, Gmail, etc.) or directly via Docker:

```bash
echo '{
  "prompt": "Calculate the band gap of silicon using DFT (Quantum ESPRESSO)",
  "groupFolder": "test",
  "chatJid": "test@g.us",
  "isMain": false,
  "secrets": {
    "ANTHROPIC_API_KEY": "your-api-key",
    "ANTHROPIC_BASE_URL": "https://api.anthropic.com"
  }
}' | docker run -i -v ./workspace:/workspace/group matclaw-agent:latest
```

The agent will autonomously:

1. Parse your request and plan the computation
2. Write input files and scripts
3. Run the simulation (DFT, MD, MC, or MLIP)
4. Analyze results, generate plots, and report back
5. Retry with adjustments if errors occur

Results — including figures, data files, and structured summaries — are returned directly.

## Examples

Benchmark tasks adapted from [QUASAR](https://github.com/fengxuyy/QUASAR). All executed autonomously — the agent writes scripts, runs simulations, and reports results. See [`examples/`](examples/) for full details.

### Basic Tasks

#### Cu K-point Convergence (DFT / Quantum ESPRESSO)
> Converge bulk Cu energy to < 1 meV/atom. Agent runs 8 QE calculations and plots the convergence curve.

<p align="center"><img src="examples/cu_kpoint_convergence/feishu-chat-1.png" width="700"></p>

---

#### Water Density (MD / LAMMPS)
> Calculate the density of water at 298 K, 1 bar. Agent builds a SPC/E water box, runs NPT, and reports with diagnostic plots.

<p align="center"><img src="examples/water_density/feishu-chat-1.png" width="700"></p>

---

#### IRMOF-1 Void Fraction (MC / RASPA3)
> Compute helium-accessible void fraction for IRMOF-1 at 298 K. Agent configures RASPA3 Widom insertion MC.

<p align="center"><img src="examples/irmof1_void_fraction/feishu-chat-1.png" width="700"></p>

---

### Workflow Orchestration

#### NiO Band Gap (DFT+U / Quantum ESPRESSO)
> Calculate the electronic band gap of NiO. Agent recognizes a strongly correlated system and **autonomously applies DFT+U**.

<p align="center"><img src="examples/nio_bandgap/feishu-chat-1.png" width="700"></p>

> [!TIP]
> The agent was only asked to "calculate the band gap of NiO" — it independently decided DFT+U was necessary for a correlated oxide, chose appropriate U values, and ran the full SCF → NSCF → DOS workflow.

---

#### CO₂ Adsorption in UiO-66 (MC / RASPA3)
> Calculate CO₂ adsorption isotherm at 4 pressure points. Agent runs GCMC simulations and generates the isotherm plot.

<p align="center"><img src="examples/co2_uio66_adsorption/feishu-chat-1.png" width="700"></p>

---

#### Al Melting Point (MD / LAMMPS)
> Determine the melting point of aluminum via two-phase coexistence. Agent builds an 8000-atom system and analyzes with bond-order parameters.

<p align="center"><img src="examples/al_melting_point/feishu-chat-1.png" width="700"></p>

---

#### NaCl Solution Density (MD / LAMMPS + packmol)
> Calculate density of 1 mol/L NaCl solution. Agent **autonomously installs packmol** (not pre-installed), builds the system, and runs MD.

<p align="center"><img src="examples/nacl_solution_density/feishu-chat-1.png" width="700"></p>

> [!TIP]
> packmol was not pre-installed in the container. The agent detected the missing dependency, downloaded and compiled it from source (retrying 3 times with different approaches), then proceeded with the simulation.

---

### Results Summary

| Example | Method | Engine | Reference | Agent Result |
|---------|--------|--------|-----------|-------------|
| [Cu k-point convergence](examples/cu_kpoint_convergence/) | DFT | QE | < 1 meV/atom | Converged at 12×12×12 |
| [Water density](examples/water_density/) | MD | LAMMPS | 0.997 g/cm³ | 0.985 g/cm³ |
| [IRMOF-1 void fraction](examples/irmof1_void_fraction/) | MC | RASPA3 | 0.7988 | 0.8025 |
| [NiO band gap](examples/nio_bandgap/) | DFT+U | QE | 4.0 eV | 2.11 eV |
| [CO₂ in UiO-66](examples/co2_uio66_adsorption/) | MC | RASPA3 | 5.98 mmol/g | 5.48 mmol/g |
| [Al melting point](examples/al_melting_point/) | MD | LAMMPS | 933 K | ~850–880 K |
| [NaCl solution density](examples/nacl_solution_density/) | MD | LAMMPS | 1.038 g/cm³ | 1.033 g/cm³ |


## Computation Stack

| Engine | Version | Method | Use Cases |
|--------|---------|--------|-----------|
| [Quantum ESPRESSO](https://www.quantum-espresso.org/) | 7.5 | DFT | Electronic structure, band gaps, DOS, phonons, elastic constants |
| [LAMMPS](https://www.lammps.org/) | 2021 | MD | Thermal properties, diffusion, mechanical properties, phase transitions |
| [RASPA3](https://github.com/iRASPA/RASPA3) | 3.0.16 | MC | Gas adsorption in MOFs/zeolites, isotherms, Henry constants |
| [MACE-MP-0](https://github.com/ACEsuit/mace) | latest | MLIP | Universal ML potential, fast energy/force/stress predictions |

<details>
<summary><strong>Python Materials Science Stack</strong> (all pre-installed)</summary>

| Package | Purpose |
|---------|---------|
| [pymatgen](https://pymatgen.org/) | Crystal structures, phase diagrams, electronic structure analysis |
| [ASE](https://wiki.fysik.dtu.dk/ase/) | Atoms objects, calculators, optimization, molecular dynamics |
| [MACE-torch](https://github.com/ACEsuit/mace) | Universal ML interatomic potentials |
| [mp-api](https://materialsproject.org/) | Materials Project database access |
| [spglib](https://spglib.github.io/spglib/) | Space group / symmetry analysis |
| [PyTorch](https://pytorch.org/) | ML framework (CPU) |
| numpy, scipy, matplotlib, pandas, seaborn | Scientific computing & visualization |

</details>

## Quick Start

### Prerequisites

- Linux (Ubuntu 24.04+ recommended) or macOS
- [Docker](https://docs.docker.com/get-docker/)
- An Anthropic-compatible API key (Claude, DeepSeek, etc.)

### 1. Build the container

```bash
git clone https://gitee.com/baiyuan1/mat-claw.git
cd mat-claw
./container/build.sh
```

> [!NOTE]
> The first build takes ~10 minutes (compiles QE 7.5 from source). Subsequent builds use Docker cache.

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
npm run dev
```

Configure at least one messaging channel to chat with your agent:

- **[Feishu (飞书)](docs/feishu-setup.md)** — WebSocket connection, no public URL needed. Recommended for China users.
- **[Gmail](docs/gmail-setup.md)** — Send computation tasks via email, receive results back.
- **WhatsApp** — Add via `/add-whatsapp` skill, QR code authentication.
- **Telegram** — Add via `/add-telegram` skill, Bot API.
- **Discord / Slack** — Add via `/add-discord` or `/add-slack` skill.

Channels are added via [NanoClaw skills](https://github.com/qwibitai/nanoclaw) — run the corresponding `/add-*` command inside `claude` CLI.

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

For full architecture details, see [docs/SPEC.md](docs/SPEC.md). For the security model and container isolation design, see [docs/SECURITY.md](docs/SECURITY.md).

<details>
<summary><strong>Configuration</strong></summary>

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

</details>

## Acknowledgments

MatClaw is built on [NanoClaw](https://github.com/qwibitai/nanoclaw) and relies on the following open-source projects:

- [Quantum ESPRESSO](https://www.quantum-espresso.org/) — DFT calculations
- [LAMMPS](https://www.lammps.org/) — Molecular dynamics
- [RASPA3](https://github.com/iRASPA/RASPA3) — Monte Carlo simulations
- [MACE](https://github.com/ACEsuit/mace) — Machine learning interatomic potentials
- [pymatgen](https://pymatgen.org/) — Python materials analysis
- [ASE](https://wiki.fysik.dtu.dk/ase/) — Atomic simulation environment
- [Claude Agent SDK](https://github.com/anthropics/claude-agent-sdk) — AI agent framework
- [QUASAR](https://github.com/fengxuyy/QUASAR) — Benchmark test cases referenced from this project

<details>
<summary><strong>Project Structure</strong></summary>

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

</details>

## Documentation

| Document | Description |
|----------|-------------|
| [Architecture Spec](docs/SPEC.md) | Full system architecture and design |
| [Security Model](docs/SECURITY.md) | Container isolation and trust model |
| [Requirements](docs/REQUIREMENTS.md) | Original requirements and design decisions |
| [Debug Checklist](docs/DEBUG_CHECKLIST.md) | Troubleshooting known issues |
| [Feishu Setup](docs/feishu-setup.md) | Feishu channel configuration guide |
| [Gmail Setup](docs/gmail-setup.md) | Gmail channel configuration guide |
| [SDK Deep Dive](docs/SDK_DEEP_DIVE.md) | Claude Agent SDK internals |
| [Creating Skills](docs/creating-skills.md) | How to create a new skill (template included) |
| [Skills Architecture](docs/nanorepo-architecture.md) | How the skill system works (internals) |
| [Apple Container Networking](docs/APPLE-CONTAINER-NETWORKING.md) | macOS container network setup |

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
