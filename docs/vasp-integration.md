# VASP Integration

MatClaw supports [VASP](https://www.vasp.at/) (Vienna Ab initio Simulation Package) as an external computation engine. Since VASP is proprietary software, it is not bundled in the container — instead, you connect your existing VASP installation, and the agent executes VASP calculations seamlessly alongside the built-in open-source engines.

## How It Works

```
User: "Calculate the band structure of GaAs using VASP with HSE06"
                          │
                          ▼
              Agent (inside container)
              ├─ Reads relevant skill guide
              ├─ Generates INCAR, POSCAR, KPOINTS, POTCAR
              ├─ Runs: vasp-remote run
              │   ├─ SSH mode: uploads files → submits job → polls → fetches results
              │   └─ Local mode: mpirun vasp_std directly
              └─ Parses vasprun.xml with pymatgen → returns band structure plot
```

The `vasp-remote` tool bridges the container and your VASP installation. It handles file transfer, job submission, status polling, and result retrieval — the agent just calls `vasp-remote run` and gets back the output files.

## Two Connection Modes

### SSH to HPC Cluster (Recommended)

Best for: users with access to a university/institute HPC cluster where VASP is installed.

```
Container ──SSH──▶ Login Node ──SLURM/PBS──▶ Compute Nodes
                                                  │
                                              vasp_std
                                                  │
                              ◀──SCP── vasprun.xml, OUTCAR, ...
```

The agent uploads VASP input files via SCP, submits a job through SLURM or PBS, waits for completion, and downloads the results. This is the most common setup — virtually all VASP users have cluster access.

**Supported schedulers:** SLURM (`sbatch`/`squeue`) and PBS (`qsub`/`qstat`).

### Local VASP Binary

Best for: users with VASP installed on the same machine running MatClaw.

The host's VASP binary directory is volume-mounted into the container. The agent calls `mpirun vasp_std` directly.

> **Note:** This mode requires that the container's MPI and library versions are compatible with the VASP binary compiled on the host. SSH mode avoids this issue entirely.

## Setup

Run the `/add-vasp` skill in Claude CLI:

```bash
claude
# Then type: /add-vasp
```

The skill will interactively guide you through:

1. Choosing connection mode (SSH or local)
2. Entering cluster/binary details
3. Testing the connection
4. Rebuilding the container

### SSH Mode Configuration

You'll need:

| Field | Example |
|-------|---------|
| SSH host | `login.hpc.example.edu` |
| SSH user | `jdoe` |
| SSH key | `~/.ssh/id_rsa` |
| VASP binary path (on cluster) | `/opt/vasp/6.4.3/bin/vasp_std` |
| Scratch directory (on cluster) | `/scratch/jdoe/vasp-jobs` |
| Scheduler | `slurm` or `pbs` |
| Default cores | `16` |

### Local Mode Configuration

You'll need:

| Field | Example |
|-------|---------|
| VASP binary path | `/opt/vasp/bin/vasp_std` |
| POTCAR directory | `/opt/vasp/potpaw_PBE` |
| Number of cores | `4` |

## Usage

Once configured, VASP is available automatically. The agent chooses VASP when appropriate, or you can request it explicitly:

```
"Calculate the elastic constants of BaTiO3 using VASP"
"Run an HSE band structure for MoS2 with VASP"
"Relax the structure of Fe3O4 with VASP, use DFT+U"
```

The `vasp-remote` command is also available directly:

```bash
# Run VASP (blocking — waits for results)
vasp-remote run

# Run with custom settings
vasp-remote run --nprocs 32 --queue express --walltime 04:00:00

# Submit without waiting (SSH mode)
vasp-remote submit
# → Returns job_id

# Check status
vasp-remote status <job_id>

# Fetch results
vasp-remote fetch <job_id>

# Show configuration
vasp-remote config
```

## VASP Is Optional

**You do not need VASP to use MatClaw.** All 213 computation skills provide up to three methods:

| Method | Engine | Speed | Accuracy | Pre-installed |
|--------|--------|-------|----------|:-------------:|
| **A** | ASE + MACE | Seconds | Screening | Yes |
| **B** | Quantum ESPRESSO | Minutes–Hours | Publication | Yes |
| **C** | VASP | Minutes–Hours | Publication | No (optional) |

Without VASP, the agent automatically uses Method A or B. VASP adds another high-accuracy option — particularly useful when you need VASP-specific features (hybrid functionals with VASP optimizations, specific pseudopotentials, compatibility with existing VASP workflows).

## Troubleshooting

| Problem | Solution |
|---------|----------|
| `vasp-remote: Config not found` | Run `/add-vasp` to configure |
| SSH connection refused | Check VPN/firewall, verify with `ssh -v user@host` |
| Permission denied (publickey) | Check key permissions: `chmod 600 ~/.ssh/id_rsa` |
| Job stuck in PENDING | Try a different queue: `vasp-remote run --queue express` |
| `vasp_std: command not found` | Verify VASP path: `ssh user@host "ls /path/to/vasp_std"` |
| Library errors (local mode) | Switch to SSH mode to avoid container/host library mismatch |
