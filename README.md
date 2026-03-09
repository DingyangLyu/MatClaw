<p align="center">
  <img src="assets/matclaw-logo.svg" alt="MatClaw" width="720">
</p>

<p align="center">
  <strong>Describe a materials problem in plain language. MatClaw writes the code, runs the simulation, and delivers results.</strong>
</p>

<p align="center">
  <a href="README_zh.md"><img src="https://img.shields.io/badge/中文-README-blue" alt="中文"></a>&nbsp;
  <a href="LICENSE"><img src="https://img.shields.io/badge/license-Apache%202.0-blue.svg" alt="Apache 2.0"></a>&nbsp;
  <a href="https://github.com/bjzgcai"><img src="https://img.shields.io/badge/Linked%20to-bjzgcai%20Org-blue?logo=github" alt="bjzgcai Org"></a>&nbsp;
  <img src="https://img.shields.io/badge/Docker-24.04-2496ED?logo=docker&logoColor=white" alt="Docker">&nbsp;
  <img src="https://img.shields.io/badge/Node.js-20+-339933?logo=nodedotjs&logoColor=white" alt="Node.js 20+">&nbsp;
  <img src="https://img.shields.io/badge/Python-3.11-3776AB?logo=python&logoColor=white" alt="Python 3.11">
</p>

<p align="center">
  <img src="https://img.shields.io/badge/QE-7.5-4F46E5" alt="QE 7.5">&nbsp;
  <img src="https://img.shields.io/badge/LAMMPS-2021-7C3AED" alt="LAMMPS">&nbsp;
  <img src="https://img.shields.io/badge/RASPA3-3.0.16-0D9488" alt="RASPA3">&nbsp;
  <img src="https://img.shields.io/badge/VASP-external-F59E0B" alt="VASP (external)">&nbsp;
  <img src="https://img.shields.io/badge/CUDA-12.8-76B900?logo=nvidia&logoColor=white" alt="CUDA 12.8">&nbsp;
  <img src="https://img.shields.io/badge/Skills-221-E11D48" alt="221 Skills">
</p>

---

## Contents

- [What is MatClaw?](#what-is-matclaw)
- [Chat Commands](#chat-commands)
- [Built-in Computation Skills](#built-in-computation-skills)
- [Basic Usage](#basic-usage)
- [Examples](#examples)
- [Computation Stack](#computation-stack)
- [Quick Start](#quick-start)
- [Architecture](#architecture)
- [Configuration](#configuration)
- [Documentation](#documentation)
- [Contributing](#contributing)
- [Citation](#citation)

## What is MatClaw?

MatClaw is an **AI agent that autonomously performs materials science computations**. You describe a task in natural language — it writes Python/shell scripts, runs them inside an isolated Docker container equipped with a full computation stack, and returns the results.

<p align="center"><img src="examples/water_density/feishu-chat-1.png" width="800"></p>

<p align="center"><em>Send a task via Feishu (or any channel). Get back scripts, simulations, plots, and analysis — no manual coding required.</em></p>

**Key Features:**

- **Autonomous computation** — Understands your task, writes code, executes it, analyzes output, retries on errors
- **221 built-in computation skills** — 44 skill groups covering the full spectrum of computational materials science: electronic structure, phonons, mechanical properties, defects, optical/magnetic/topological properties, catalysis, batteries, phase diagrams, transport, spectroscopy, Monte Carlo, molecular dynamics, and more. Each skill contains complete runnable scripts, parameter guides, and method selection decision trees. See [Materials Compute Skills Reference](docs/materials-compute-skills.md) for the full inventory.
- **VASP support** — Connect your own VASP installation via SSH (HPC cluster) or local mount. The agent generates inputs, submits jobs, and parses results automatically. See [VASP Integration](docs/vasp-integration.md).
- **GPU acceleration** — Optional CUDA-enabled container (`./container/build.sh --cuda`) for GPU-accelerated MACE, CHGNet, SevenNet, and MatGL. Auto-detects GPU at runtime with graceful CPU fallback.
- **Multiple MLIP models** — MACE-MP-0 (pre-installed), CHGNet, SevenNet, MatGL — all pre-installed and ready for rapid screening and molecular dynamics
- **All-in-one container** — QE 7.5, LAMMPS, RASPA3, MACE, pymatgen, ASE, PyTorch pre-installed and ready
- **Secure isolation** — Every computation runs in a disposable Docker container with filesystem isolation
- **Flexible LLM backend** — Works with Anthropic Claude, DeepSeek, or any Anthropic-compatible API
- **Multi-channel access** — Chat via Feishu, DingTalk, Gmail, WhatsApp, Telegram, Discord, Slack
- **Chat commands** — `/watch`, `/status`, `/stop`, `/sessions`, `/new`, `/resume`, `/compact` — manage sessions, monitor progress, and control the agent directly from chat
- **Real-time dashboard** — Web UI at `localhost:3210` with live agent activity, parsed transcripts, and container logs
- **Extensible** — Conda/pip available inside container; agent can install additional packages on-the-fly

## Chat Commands

Control the agent directly from any messaging channel — no terminal or dashboard required.

| Command | Description |
|---------|-------------|
| `/watch` | See what the agent is doing right now (recent tool calls, reads, bash commands) |
| `/status` | Agent status — running/idle, current session, container name, queued tasks |
| `/stop` | Force stop a running agent |
| `/sessions` | List all conversation sessions (ID, timestamp, size, active marker) |
| `/new` | Start a fresh conversation with no prior memory |
| `/resume [id]` | Restore previous session, or switch to any session by ID prefix |
| `/compact [focus]` | Compress agent memory. Optionally specify what to keep (e.g. `/compact keep only VASP config`) |
| `/help` | Show all available commands |

**Session management** — Every conversation is a resumable session. Use `/new` to start clean, `/sessions` to browse history, and `/resume` to jump back to any previous context. The agent picks up exactly where it left off.

**Real-time monitoring** — Send `/watch` at any time to see the agent's recent activity without waiting for it to finish. For a full graphical view, open the built-in dashboard at `http://localhost:3210`.

## Built-in Computation Skills

MatClaw ships with **221 SKILL.md files across 44 skill groups**, covering the full spectrum of computational materials science. Each skill contains complete runnable scripts, parameter guides, method selection decision trees, and troubleshooting tables — enabling the agent to autonomously execute any mainstream materials computation workflow.

**44 groups / 177 sub-skills / 221 SKILL.md files**

| # | Skill Group | Sub-Skills | Contents |
|---|-------------|:----------:|----------|
| 1 | **2d-materials** | 4 | band-edges, layer-manipulation, stacking-energy, vacuum-resize |
| 2 | **advanced-electronic** | 5 | gw-approximation, hubbard-u, spin-orbit-coupling, topological-invariants, van-der-waals |
| 3 | **agent-browser** | 0 | *(browser automation, non-computation)* |
| 4 | **alloy-disorder** | 2 | cluster-expansion, sqs-generation |
| 5 | **band-advanced** | 3 | 3d-band-structure, band-unfolding, hybrid-dft-bands |
| 6 | **battery-electrode** | 2 | intercalation-voltage, ion-diffusion |
| 7 | **biomolecular-md** | 1 | openmm-simulation |
| 8 | **bonding-analysis** | 10 | bader2pqr, bader-charge, charge-density, charge-density-difference, charge-format-conversion, elf-analysis, lobster-cohp, orbital-projection, planar-charge, stm-simulation |
| 9 | **catalysis-electrochem** | 6 | band-center, imaginary-freq-correction, implicit-solvation, neb-analysis, reaction-kinetics, thermal-corrections |
| 10 | **catalyst-screening** | 3 | d-band-center, overpotential, scaling-relations |
| 11 | **code-interfaces** | 5 | boltztrap-interface, ifc-analysis, phonopy-interface, vasp-qe-converter, wannier90-interface |
| 12 | **defects-reactions** | 13 | activation-relaxation-technique, adsorption-energy, configuration-coordinate, defect-thermodynamics, interstitial-defect, migration-barrier, neb-transition-state, point-defect, reaction-pathway, substitution-defect, surface-adsorption, surface-energy, vacancy-formation |
| 13 | **dft-corrections** | 3 | hubbard-u, spin-orbit-coupling, vdw-correction |
| 14 | **electronic-structure** | 8 | band-structure, convergence-testing, density-of-states, inverse-participation-ratio, projected-dos, scf-relax, spatially-resolved-dos, vasp-bands |
| 15 | **electron-phonon** | 4 | deformation-potential, electronic-transport, elph-coupling, superconductivity |
| 16 | **fermi-surface** | 3 | 2d-fermi-surface, 3d-fermi-surface, projected-fermi-surface |
| 17 | **ferroelectric** | 5 | born-effective-charge, dielectric-tensor, ferroelectric-switching, piezoelectric, polarization |
| 18 | **high-throughput** | 8 | batch-calculations, batch-screening, convergence-automation, materials-filtering, matpes-dual-static, phase-stability, property-prediction, screening-workflow |
| 19 | **interface** | 2 | grain-boundary, heterostructure |
| 20 | **kpath-utilities** | 5 | 1d-kpath, 2d-kpath, bulk-kpath, cp2k-kpath, phonopy-kpath |
| 21 | **magnetic-properties** | 3 | magnetic-anisotropy, magnetic-ordering, spin-polarized |
| 22 | **materials-compute** | 0 | *(root skill: QE/LAMMPS/MACE environment reference)* |
| 23 | **materials-databases** | 2 | 2d-semiconductors, materials-project |
| 24 | **mechanical-properties** | 5 | angular-mechanics, elastic-constants, energy-strain-method, equation-of-state, stress-strain-method |
| 25 | **mlip-guide** | 4 | mace-advanced, mlip-validation, torchsim-batch, universal-mlip |
| 26 | **molecular-qchem** | 1 | gaussian-qchem-workflow |
| 27 | **monte-carlo** | 5 | adsorption-isotherm, gas-adsorption, gas-separation, gcmc-simulation, pore-analysis |
| 28 | **optical-properties** | 6 | absorption-spectrum, dielectric-function, joint-dos, optical-conductivity, slme, transition-dipole |
| 29 | **phase-diagram** | 2 | convex-hull, pourbaix-diagram |
| 30 | **phase-transition** | 6 | amorphous-structure, melting-point-coexistence, metadynamics, mpmorph-melting, order-parameter, phase-diagram |
| 31 | **piezoelectric** | 1 | piezoelectric-tensor |
| 32 | **potential-analysis** | 3 | macroscopic-average, planar-average, work-function |
| 33 | **semiconductor-kit** | 4 | angular-effective-mass, band-gap, effective-mass, fermi-velocity |
| 34 | **spectroscopy** | 2 | raman-ir, xas-xanes |
| 35 | **spin-texture** | 2 | 2d-spin-texture, 3d-spin-texture |
| 36 | **structure-models** | 8 | alloy-builder, defect-builder, heterostructure, moire-superlattice, nanowire-nanotube, quantum-dot, supercell-builder, surface-builder |
| 37 | **structure-tools** | 8 | advanced-optimization, format-conversion, input-generation, pdf-analysis, structure-editing, structure-matching, symmetry-analysis, xrd-pattern |
| 38 | **surface-energy** | 2 | surface-energy-calc, wulff-construction |
| 39 | **thermal-properties** | 13 | anharmonicity, bond-distribution, free-energy-calculation, gruneisen-qha, md-trajectory-tools, molecular-dynamics, msd-diffusion, phonon, phonon-from-outcar, quasi-harmonic-debye, rdf-analysis, thermal-conductivity, vacf-vdos |
| 40 | **thermoconductivity** | 1 | lattice-thermal-conductivity |
| 41 | **topological** | 2 | berry-curvature, z2-invariant |
| 42 | **transport-properties** | 2 | boltzmann-transport, kpoints-transport |
| 43 | **wannier-functions** | 1 | wannier90-workflow |
| 44 | **wavefunction-analysis** | 2 | real-space-wavefunction, wavefunction-parity |

> **Coverage**: electronic structure, mechanics, thermodynamics, phonons, defects, optics, magnetism, topology, catalysis, batteries, phase diagrams, ferroelectric/piezoelectric, transport, surfaces, interfaces, 2D materials, alloys, Monte Carlo, molecular dynamics, machine learning potentials, biomolecular simulation, quantum chemistry, and more. Verified against [atomate2](https://github.com/materialsproject/atomate2), [aiida-quantumespresso](https://github.com/aiidateam/aiida-quantumespresso), and [aiida-vasp](https://github.com/aiida-vasp/aiida-vasp) — all workflow capabilities are covered. See [Materials Compute Skills Reference](docs/materials-compute-skills.md) for detailed descriptions of each skill.

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
| [VASP](https://www.vasp.at/) | 5.x / 6.x | DFT | Full-featured DFT via external connection ([setup guide](docs/vasp-integration.md)) |
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
git clone https://github.com/DingyangLyu/MatClaw.git
cd MatClaw
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
- **[DingTalk (钉钉)](docs/dingtalk-setup.md)** — Stream Mode (WebSocket), no public URL needed. Auto-registers groups on first message.
- **[Gmail](docs/gmail-setup.md)** — Send computation tasks via email, receive results back.
- **WhatsApp** — Add via `/add-whatsapp` skill, QR code authentication.
- **Telegram** — Add via `/add-telegram` skill, Bot API.
- **Discord / Slack** — Add via `/add-discord` or `/add-slack` skill.

Channels are added via the skill system — run the corresponding `/add-*` command inside `claude` CLI.

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

## Contributing

We welcome contributions — especially new computation skills! Adding a skill is as simple as writing a single `SKILL.md` file with runnable scripts and parameter guides. No need to touch core code.

```
container/skills/<group>/<your-skill>/SKILL.md
```

See [CONTRIBUTING.md](CONTRIBUTING.md) for the full guide, including the SKILL.md template, testing instructions, and PR workflow.

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

The built-in computation skills were designed and verified against the following workflow frameworks:

- [atomate2](https://github.com/materialsproject/atomate2) — Materials Project's computational workflows (VASP, QE, force fields, and more)
- [pyiron_atomistics](https://github.com/pyiron/pyiron_atomistics) — Integrated materials science workflow platform (Murnaghan EOS, QHA, SQS, Debye model, ART, metadynamics, and more)
- [VASPKIT](https://vaspkit.com/) — VASP pre-/post-processing toolkit
- [AiiDA](https://github.com/aiidateam/aiida-core) — Automated Interactive Infrastructure and Database for Computational Science
- [aiida-quantumespresso](https://github.com/aiidateam/aiida-quantumespresso) — AiiDA plugin for Quantum ESPRESSO workflows
- [aiida-vasp](https://github.com/aiida-vasp/aiida-vasp) — AiiDA plugin for VASP workflows

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
│   └── skills/                 # 221 computation skills (44 groups)
│       ├── materials-compute/  # Root: computation engine docs
│       ├── electronic-structure/  # Band structure, DOS, SCF
│       ├── thermal-properties/    # Phonons, QHA, MD, thermal transport
│       ├── mechanical-properties/ # Elastic constants, EOS
│       ├── defects-reactions/     # Defects, NEB, adsorption, CCD
│       ├── optical-properties/    # Dielectric function, absorption
│       ├── magnetic-properties/   # Spin ordering, MAE
│       ├── topological/           # Z2, Berry curvature
│       ├── catalysis-electrochem/ # Reaction kinetics, d-band
│       ├── battery-electrode/     # Intercalation voltage, ion diffusion
│       ├── monte-carlo/           # GCMC, gas adsorption (RASPA3)
│       ├── ...                    # 34 more groups
│       └── agent-browser/         # Browser automation
└── groups/                     # Per-group isolated memory
```

</details>

## Documentation

| Document | Description |
|----------|-------------|
| [Architecture Spec](docs/SPEC.md) | Full system architecture and design |
| [Security Model](docs/SECURITY.md) | Container isolation and trust model |
| [Requirements](docs/REQUIREMENTS.md) | Original requirements and design decisions |
| [Feishu Setup](docs/feishu-setup.md) | Feishu channel configuration guide |
| [DingTalk Setup](docs/dingtalk-setup.md) | DingTalk channel configuration guide |
| [Gmail Setup](docs/gmail-setup.md) | Gmail channel configuration guide |
| [SDK Deep Dive](docs/SDK_DEEP_DIVE.md) | Claude Agent SDK internals |
| [Materials Compute Skills](docs/materials-compute-skills.md) | Full inventory of 221 built-in computation skills |
| [VASP Integration](docs/vasp-integration.md) | Connect your VASP installation (SSH or local) |
| [Creating Skills](docs/creating-skills.md) | How to create a new skill (template included) |
| [Skills Architecture](docs/nanorepo-architecture.md) | How the skill system works (internals) |

## Roadmap

- [x] GPU support (CUDA 12.8 container for PyTorch/MACE — `./container/build.sh --cuda`)
- [x] More MLIP models (CHGNet, SevenNet, MatGL pre-installed)
- [x] Materials Project integration (set `MP_API_KEY` in `.env` — query structures, phase diagrams, properties)
- [ ] Workflow automation (multi-step calculation pipelines)
- [ ] Jupyter notebook generation for reproducibility

## Citation

If you use MatClaw in your research, please cite:

```bibtex
@software{matclaw2026,
  title  = {MatClaw: AI-Powered Autonomous Materials Science Agent},
  author = {Dingyang Lyu and Baole Wei and Hongwei Du and Yongheng Li and Feng Yu},
  email  = {s-ldy25@bza.edu.cn, weibaole@zgci.ac.cn, duhongwei@zgci.ac.cn},
  year   = {2026},
  url    = {https://github.com/DingyangLyu/MatClaw}
}
```

## License

[Apache License 2.0](LICENSE)
