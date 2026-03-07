# Creating Skills

MatClaw uses [NanoClaw's skill system](https://github.com/qwibitai/nanoclaw) to extend functionality. Skills are not plugins — they modify the actual codebase via git three-way merges, so they can change anything: add channels, integrations, container tools, or replace internals entirely.

For the full architecture details, see [Skills Architecture](nanorepo-architecture.md).

## Quick Overview

A skill is a directory under `.claude/skills/<skill-name>/` containing a `SKILL.md` that tells Claude Code how to apply and configure the feature. Users run `/<skill-name>` in the `claude` CLI to apply it.

```
.claude/skills/<skill-name>/
  SKILL.md                              # Instructions for Claude Code
  manifest.yaml                         # (optional) Metadata, deps, structured ops
  add/path/to/new-file.ts               # New files to add
  modify/path/to/existing-file.ts       # Full modified file for 3-way merge
  modify/path/to/existing-file.ts.intent.md  # Intent for conflict resolution
  tests/                                # Integration tests
```

## SKILL.md Template

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

## Key Concepts

### New files vs. modified files

- **New files** (`add/` directory): Files the skill creates from scratch. Copied directly, no merging needed.
- **Modified files** (`modify/` directory): Files that already exist in the codebase. The skill carries the **full modified version** (core + skill changes). Applied via `git merge-file` three-way merge against the shared base in `.matclaw/base/`.

### Intent files

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

### Manifest (optional)

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

### Phases

All skills follow the same 4-phase structure:

| Phase | Purpose |
|-------|---------|
| **1. Pre-flight** | Check if already applied, collect user input |
| **2. Apply** | Run skills engine, merge code, validate build |
| **3. Configure** | Set env vars, credentials, service config |
| **4. Verify** | Test the integration end-to-end |

### Skill types

| Type | Example | What it does |
|------|---------|-------------|
| Channel | `add-telegram`, `add-slack` | Adds a messaging channel to the orchestrator |
| Integration | `add-gmail`, `add-ollama-tool` | Adds external service connectivity |
| Container tool | `add-voice-transcription` | Adds tools available inside agent containers |
| Infrastructure | `convert-to-apple-container` | Changes runtime or deployment |

## Existing Skills

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

## Tips

- **One skill, one happy path.** Implement the reasonable default for 80% of users. Customization happens after applying.
- **Skills layer via `depends`.** Extension skills build on base skills (e.g., `add-telegram-swarm` depends on `add-telegram`).
- **Always include tests.** Tests run after every operation — apply, update, uninstall, replay.
- **Keep SKILL.md conversational.** It's read by Claude Code, which follows the instructions step-by-step and interacts with the user along the way.
- **Include a Troubleshooting section.** Common issues save users from running `/debug`.
- **Include Removal steps.** Users should be able to cleanly uninstall.
