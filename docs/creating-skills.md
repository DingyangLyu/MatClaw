# Creating Skills

MatClaw has two distinct types of skills:

1. **Container Computation Skills** — Markdown guides (`SKILL.md`) that teach the agent how to perform materials science calculations inside containers. These are the `container/skills/` files.
2. **Platform Skills** — Code packages that modify the MatClaw codebase itself (add channels, integrations, container tools). These use the `apply-skill.ts` engine.

Most contributors will be adding **container computation skills**. If you're adding a new materials computation workflow, start with [Part 1](#part-1-container-computation-skills).

---

## Part 1: Container Computation Skills

### Overview

Container computation skills are self-contained Markdown files that guide the AI agent through materials science calculation workflows. Each skill lives at:

```
container/skills/<group>/<sub-skill>/SKILL.md
```

The agent reads these files at runtime and follows the instructions to execute calculations autonomously. No code compilation or deployment is needed — just write the Markdown and the agent can use it immediately.

### Directory Structure

```
container/skills/
├── materials-compute/           # Root skill: environment reference and master index
│   └── SKILL.md
├── <group>/                     # Skill group (e.g., thermal-properties, defects-reactions)
│   ├── SKILL.md                 # Group overview: sub-skill table + method decision guide
│   ├── <sub-skill-1>/
│   │   └── SKILL.md             # Full computation workflow
│   └── <sub-skill-2>/
│       └── SKILL.md
```

### SKILL.md Format (7 Sections)

Every sub-skill SKILL.md follows this exact structure:

```markdown
# Skill Title

## When to Use

- Bullet list of problem types this skill addresses
- Be specific: "Computing phonon band structures" not "Phonon stuff"

## Method Selection

Decision tree or table helping the agent choose the right method.
Typical structure:

| Method | Tool | Pros | Cons |
|--------|------|------|------|
| **Method A** (recommended) | ASE + MACE | Fast, seconds | Accuracy depends on MACE |
| **Method B** | QE DFT | Publication quality | Hours per calculation |

Or as a decision tree in a code block:

\```
Need property X?
  Is speed important?
    YES --> Method A (ASE + MACE)
    NO  --> Method B (QE DFT)
\```

## Prerequisites

\```bash
pip install package-name
\```

List what's pre-installed vs what needs `pip install`.

## Detailed Steps

### Method A: ASE + MACE (Recommended)

Brief description of the workflow steps.

\```python
#!/usr/bin/env python3
"""
Complete, standalone, runnable Python script.
"""

import matplotlib
matplotlib.use("Agg")  # REQUIRED: no display in container
import matplotlib.pyplot as plt

# ... complete workflow ...

plt.savefig("result.png", dpi=150)
print("Saved: result.png")
\```

### Method B: QE DFT

Similar structure with QE-specific workflow.

## Key Parameters

| Parameter | Typical Value | Effect |
|-----------|---------------|--------|
| `param_name` | 60 Ry | Description of what it controls |

## Interpreting Results

Guidance on what the output numbers mean physically.
Include typical ranges, comparison with literature, red flags.

## Common Issues

| Issue | Cause | Solution |
|-------|-------|----------|
| Error message | Why it happens | How to fix it |
```

### Code Quality Requirements

**Scripts must be:**

1. **Complete and standalone** — Copy-paste into a Python file and run. No missing imports, no undefined variables, no `...` placeholders.
2. **Self-contained** — Do not depend on other skill files. Each script is independent.
3. **Headless** — Use `matplotlib.use("Agg")` before any pyplot import. The container has no display.
4. **MACE-compatible** — For MLIP methods, use:
   ```python
   from mace.calculators import mace_mp
   calc = mace_mp(model="medium", default_dtype="float64")
   ```
5. **No pyiron/atomate2/AiiDA dependency** — Port concepts from these frameworks into standalone scripts. The container does not have workflow managers installed.
6. **Well-commented** — Use step headers (`# Step 1: ...`) and print statements for progress tracking.
7. **Output files** — Save results as `.png` plots, `.json` data, and `.cif`/`.vasp` structures. Print saved filenames.

**Typical skill size:**

| Component | Lines |
|-----------|-------|
| Small skill (1 method) | 300–600 |
| Medium skill (2 methods) | 600–1,200 |
| Large skill (3+ methods with analysis) | 1,200–2,500 |

### Step-by-Step: Adding a New Computation Skill

#### 1. Identify the group

Check if an existing skill group fits. Current groups:

```
thermal-properties/    mechanical-properties/    electronic-structure/
phase-transition/      defects-reactions/        catalysis-electrochem/
alloy-disorder/        structure-tools/          high-throughput/
optical-properties/    magnetic-properties/      ...
```

If no group fits, create a new group directory with its own group-level `SKILL.md`.

#### 2. Create the sub-skill directory and write SKILL.md

```bash
mkdir -p container/skills/<group>/<sub-skill>/
# Write SKILL.md following the 7-section format above
```

#### 3. Update the group-level SKILL.md

Edit `container/skills/<group>/SKILL.md`:

- Update the `description:` field in the YAML front matter (increment sub-skill count, add name to list)
- Add a row to the Sub-Skills table
- Add an entry to the Method Decision Guide

Example diff:
```diff
- description: Thermal Properties (13 sub-skills: ...)
+ description: Thermal Properties (14 sub-skills: ..., my-new-skill)

  | Sub-Skill | Directory | Description |
+ | My New Skill | `my-new-skill/` | One-line description |
```

#### 4. Update the materials-compute skill index

Edit `container/skills/materials-compute/SKILL.md`:

- Add the new sub-skill to the appropriate table row in the "Skill Reference Index" section

#### 5. Update README.md and README_zh.md

In both files, update:

- The header counts: `**N SKILL.md files across 44 skill groups**` and `**44 groups / M sub-skills / N SKILL.md files**`
- The skill group row in the table: increment sub-skill count and add the new name to the contents list

#### 6. Update docs/materials-compute-skills.md

- Update the header count: `**Total: 44 skill groups / M sub-skills / N SKILL.md files**`
- Add a row to the appropriate section's sub-skill table
- Update the Coverage Summary table if sub-skill counts changed

#### 7. Checklist before submitting

```
□ SKILL.md follows the 7-section format
□ All Python scripts are complete, standalone, and runnable
□ matplotlib.use("Agg") is called before any pyplot import
□ MACE calculator uses mace_mp(model="medium", default_dtype="float64")
□ No external workflow manager dependencies (pyiron, atomate2, AiiDA)
□ Group-level SKILL.md updated (description, table, decision guide)
□ container/skills/materials-compute/SKILL.md index updated
□ README.md counts and table updated
□ README_zh.md counts and table updated (Chinese)
□ docs/materials-compute-skills.md counts, table, and coverage updated
```

### Example: Adding a "phonon-lifetime" skill

```bash
# 1. Create directory
mkdir -p container/skills/thermal-properties/phonon-lifetime/

# 2. Write SKILL.md (following the 7-section format)
#    - When to Use: computing phonon lifetimes and linewidths
#    - Method A: phono3py + MACE
#    - Method B: QE + phono3py
#    - Key Parameters: supercell size, q-mesh, temperature
#    - Interpreting Results: typical lifetimes, comparison with experiment
#    - Common Issues: memory, convergence

# 3. Update thermal-properties/SKILL.md
#    description: ... (14 sub-skills: ..., phonon-lifetime)
#    Add row to Sub-Skills table
#    Add entry to Method Decision Guide

# 4. Update materials-compute/SKILL.md Skill Reference Index

# 5. Update README.md and README_zh.md
#    221 → 222 SKILL.md, 177 → 178 sub-skills
#    thermal-properties row: 13 → 14, add phonon-lifetime

# 6. Update docs/materials-compute-skills.md
#    Total line, thermal-properties table, Coverage Summary
```

---

## Part 2: Platform Skills

Platform skills modify the MatClaw codebase itself via git three-way merges. They can add channels, integrations, container tools, or change infrastructure.

For the full architecture details, see [Skills Architecture](nanorepo-architecture.md).

### Overview

A platform skill is a directory under `.claude/skills/<skill-name>/` containing a `SKILL.md` that tells Claude Code how to apply and configure the feature. Users run `/<skill-name>` in the `claude` CLI to apply it.

```
.claude/skills/<skill-name>/
  SKILL.md                              # Instructions for Claude Code
  manifest.yaml                         # (optional) Metadata, deps, structured ops
  add/path/to/new-file.ts               # New files to add
  modify/path/to/existing-file.ts       # Full modified file for 3-way merge
  modify/path/to/existing-file.ts.intent.md  # Intent for conflict resolution
  tests/                                # Integration tests
```

### SKILL.md Template

```markdown
---
name: my-skill-name
description: One-line description. This appears in the Claude Code skill list.
---

# Skill Title

Brief description of what this skill adds and why.

## Phase 1: Pre-flight

### Check if already applied

Read `.matclaw/state.yaml`. If `my-skill-name` is in `applied_skills`, skip to Phase 3 (Configure). The code changes are already in place.

### Ask the user

Use `AskUserQuestion` to collect any required information:

AskUserQuestion: Do you have a <service> API key, or do you need to create one?

## Phase 2: Apply Code Changes

Run the skills engine to apply this skill's code package.

### Initialize skills system (if needed)

If `.matclaw/` directory doesn't exist yet:

\```bash
npx tsx scripts/apply-skill.ts --init
\```

### Apply the skill

\```bash
npx tsx scripts/apply-skill.ts .claude/skills/my-skill-name
\```

This deterministically:
- Adds `src/path/to/new-file.ts` (describe what it does)
- Three-way merges changes into `src/path/to/existing-file.ts` (describe what changed)
- Installs npm dependencies (list them)
- Updates `.env.example` with new variables
- Records the application in `.matclaw/state.yaml`

If the apply reports merge conflicts, read the intent files:
- `modify/src/path/to/existing-file.ts.intent.md`

### Validate code changes

\```bash
npm test
npm run build
\```

All tests must pass and build must be clean before proceeding.

## Phase 3: Configure

### Set up credentials

Add to `.env`:

\```bash
MY_API_KEY=<their-key>
\```

Sync to container environment:

\```bash
mkdir -p data/env && cp .env data/env/env
\```

### Build and restart

\```bash
npm run build
systemctl --user restart matclaw  # Linux
# macOS: launchctl kickstart -k gui/$(id -u)/com.matclaw
\```

## Phase 4: Verify

### Test the integration

Tell the user how to test (send a message, trigger an action, etc.).

### Check logs if needed

\```bash
tail -f logs/matclaw.log | grep -i my-skill
\```

## Troubleshooting

### Common issue 1

Cause and fix.

### Common issue 2

Cause and fix.

## Removal

Steps to cleanly uninstall:
1. Delete added files
2. Remove imports from barrel files
3. Remove env vars from `.env`
4. Uninstall npm packages
5. Rebuild and restart
```

### Key Concepts

#### New files vs. modified files

- **New files** (`add/` directory): Files the skill creates from scratch. Copied directly, no merging needed.
- **Modified files** (`modify/` directory): Files that already exist in the codebase. The skill carries the **full modified version** (core + skill changes). Applied via `git merge-file` three-way merge against the shared base in `.matclaw/base/`.

#### Intent files

Each modified file should have a companion `.intent.md` with structured headings:

```markdown
## What this skill adds
- Brief description of changes

## Key sections
- Which code sections were added/modified

## Invariants
- Things that must remain true after merging

## Must-keep sections
- Existing code that must not be removed
```

These guide Claude Code during conflict resolution when `git merge-file` can't auto-merge.

#### Manifest (optional)

`manifest.yaml` declares metadata, dependencies, and structured operations:

```yaml
name: my-skill-name
version: 1.0.0
core_compatibility: ">=1.0.0"

files:
  add:
    - src/path/to/new-file.ts
  modify:
    - src/path/to/existing-file.ts

structured:
  npm_dependencies:
    some-package: "^1.0.0"
  env_additions:
    - MY_API_KEY

relationships:
  depends: []              # Skills that must be applied first
  conflicts: []            # Skills that can't coexist
  tested_with: []          # Verified compatible skills

post_apply:
  - npm run build

test_command: npm test
```

#### Phases

All platform skills follow the same 4-phase structure:

| Phase | Purpose |
|-------|---------|
| **1. Pre-flight** | Check if already applied, collect user input |
| **2. Apply** | Run skills engine, merge code, validate build |
| **3. Configure** | Set env vars, credentials, service config |
| **4. Verify** | Test the integration end-to-end |

### Existing Platform Skills

| Skill | Type | Description |
|-------|------|-------------|
| `/add-whatsapp` | Channel | WhatsApp via QR code auth |
| `/add-telegram` | Channel | Telegram Bot API |
| `/add-discord` | Channel | Discord bot |
| `/add-slack` | Channel | Slack Socket Mode |
| `/add-gmail` | Integration | Gmail read/send via OAuth |
| `/add-ollama-tool` | Integration | Local LLM via Ollama MCP server |
| `/add-voice-transcription` | Container tool | OpenAI Whisper voice transcription |
| `/use-local-whisper` | Container tool | Local whisper.cpp transcription |
| `/add-telegram-swarm` | Channel extension | Agent swarm bots in Telegram groups |
| `/x-integration` | Integration | X (Twitter) post/like/reply |
| `/add-parallel` | Integration | Parallel AI integration |
| `/convert-to-apple-container` | Infrastructure | Switch Docker to Apple Container |

### Tips

- **One skill, one happy path.** Implement the reasonable default for 80% of users. Customization happens after applying.
- **Skills layer via `depends`.** Extension skills build on base skills (e.g., `add-telegram-swarm` depends on `add-telegram`).
- **Always include tests.** Tests run after every operation — apply, update, uninstall, replay.
- **Keep SKILL.md conversational.** It's read by Claude Code, which follows the instructions step-by-step and interacts with the user along the way.
- **Include a Troubleshooting section.** Common issues save users from running `/debug`.
- **Include Removal steps.** Users should be able to cleanly uninstall.
