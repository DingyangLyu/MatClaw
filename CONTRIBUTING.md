# Contributing

We welcome contributions! Whether it's a new computation skill, a bug fix, or a documentation improvement.

## Ways to Contribute

| Type | Difficulty | Description |
|------|:----------:|-------------|
| Add a computation skill | Easy | Write a SKILL.md for a new calculation workflow |
| Improve existing skills | Easy | Fix bugs, add methods, improve parameters in existing skills |
| Add examples | Easy | Benchmark tasks with screenshots and results |
| Report issues | Easy | Bugs, wrong results, missing capabilities |
| Add a channel skill | Medium | Integrate a new messaging platform |
| Core improvements | Hard | Agent runner, container, orchestrator |

## Source Code Changes

**Accepted:** Bug fixes, security fixes, simplifications, reducing code.

**Not accepted:** Features, capabilities, enhancements. These should be skills.

---

## Contributing Computation Skills

This is the most impactful way to contribute. Computation skills live in `container/skills/` — each is a Markdown file that teaches the agent how to perform a specific materials science calculation.

### Directory Structure

```
container/skills/
├── <skill-group>/                # e.g., thermal-properties
│   ├── SKILL.md                  # Group overview + sub-skill table + decision guide
│   └── <sub-skill>/              # e.g., phonon
│       └── SKILL.md              # Complete calculation workflow
```

### Step 1: Choose Where It Belongs

Check the [existing 44 skill groups](docs/materials-compute-skills.md):

- **Fits an existing group?** → Add `container/skills/<group>/<your-skill>/SKILL.md`
- **New domain?** → Create `container/skills/<new-group>/SKILL.md` (group overview) + sub-skill directory

### Step 2: Write the SKILL.md

Every computation skill follows this structure:

```markdown
# Skill Title

## When to Use

- Scenario 1 where this skill applies
- Scenario 2

## Method Selection

    Need X property?
      Quick screening --> ASE + MACE (Method A)
      Publication quality --> QE DFT (Method B)

## Prerequisites

- Structure source, required packages, pseudopotentials, etc.

## Detailed Steps

### Method A: ASE + MACE

    ```python
    # Complete, self-contained, runnable Python script
    from ase.io import read
    from mace.calculators import mace_mp
    # ... 200-500 lines, produces results + plots
    ```

### Method B: QE DFT

    Complete workflow with input generation and post-processing.

## Key Parameters

| Parameter | Typical Value | Description |
|-----------|---------------|-------------|

## Interpreting Results

What the numbers mean physically. Typical ranges and anomalies.

## Common Issues

| Problem | Cause | Fix |
|---------|-------|-----|
```

### Key Principles

- **Scripts must be complete and runnable** — The agent runs your script directly. Include all imports, I/O, and plotting. No pseudocode.
- **Always generate plots** — matplotlib visualizations. The agent sends them back to the user.
- **Dual-method approach** — Fast MACE/ASE path (seconds) + DFT path (publication quality), where applicable.
- **Physical interpretation** — Don't just output numbers; explain what they mean.
- **Real parameter values** — Realistic defaults, not placeholders.

### Step 3: Update the Group SKILL.md

If adding to an existing group, update `container/skills/<group>/SKILL.md`:
- Add a row to the Sub-Skills table
- Add an entry to the Method Decision Guide

### Step 4: Test

```bash
echo '{
  "prompt": "Your test prompt that triggers the skill",
  "groupFolder": "test",
  "chatJid": "test@g.us",
  "isMain": false,
  "secrets": {
    "ANTHROPIC_API_KEY": "your-key",
    "ANTHROPIC_BASE_URL": "https://api.anthropic.com"
  }
}' | docker run -i -v ./workspace:/workspace/group matclaw-agent:latest
```

### Step 5: Submit a PR

```bash
git checkout -b add-skill-<name>
git add container/skills/<group>/<your-skill>/SKILL.md
git commit -m "Add <name> computation skill"
```

PR description should include:
- What computation this skill performs
- Which engine(s) it uses (MACE, QE, LAMMPS, etc.)
- A test result showing it works

---

## Contributing Channel Skills

Channel skills (`.claude/skills/add-*/`) teach Claude Code how to integrate a messaging platform. A PR that contributes a channel skill should not modify source files — it should contain the **instructions** Claude follows to add the feature. See `/add-telegram` for a good example.

---

## Contributing Examples

Add to `examples/`:

```
examples/<task-name>/
├── feishu-chat-1.png    # Screenshot of agent conversation
└── README.md            # Task description, reference value, agent result
```

Update the results table in `README.md` / `README_zh.md`.
