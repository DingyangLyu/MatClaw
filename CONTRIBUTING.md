# Contributing

We welcome contributions! Whether it's a new computation skill, a bug fix, or a documentation improvement.

## Ways to Contribute

| Type | Difficulty | Description |
|------|:----------:|-------------|
| Add a computation skill | Easy | Write a SKILL.md for a new calculation workflow |
| Improve existing skills | Easy | Fix bugs, add methods, improve parameters in existing skills |
| Add examples | Easy | Benchmark tasks with screenshots and results |
| Submit benchmark test cases | Easy | Provide real-world tasks from your domain expertise to test MatClaw |
| Report issues | Easy | Bugs, wrong results, missing capabilities |
| Add a channel skill | Medium | Integrate a new messaging platform |
| Core improvements | Hard | Agent runner, container, orchestrator |

## Review Process

All contributions are welcome. Skill contributions are merged quickly once tests pass. Core source code changes go through a more thorough review.

---

## Contributing Computation Skills

This is the most impactful way to contribute. Computation skills live in `container/skills/` — each is a Markdown file that teaches the agent how to perform a specific materials science calculation.

For detailed format specifications, see [Creating Skills](docs/creating-skills.md).

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

Every computation skill MUST follow the **7-section format**:

```markdown
# Skill Title

## When to Use

- Scenario 1 where this skill applies
- Scenario 2 (be specific: "Computing phonon band structures" not "Phonon stuff")

## Method Selection

Decision tree or comparison table:

| Method | Tool | Pros | Cons |
|--------|------|------|------|
| **Method A** (recommended) | ASE + MACE | Fast, seconds | Accuracy depends on MACE |
| **Method B** | QE DFT | Publication quality | Hours per calculation |

Or as a code-block decision tree:

    ```
    Need property X?
      Quick screening --> Method A (ASE + MACE)
      Publication quality --> Method B (QE DFT)
    ```

## Prerequisites

    ```bash
    pip install package-name
    ```

List what's pre-installed vs. what needs `pip install`.
Pre-installed: ase, pymatgen, mace-torch, numpy, scipy, matplotlib, spglib, torch.

## Detailed Steps

### Method A: ASE + MACE (Recommended)

Brief description of the workflow, then a complete script:

    ```python
    #!/usr/bin/env python3
    """
    One-line description of what this script does.
    Complete, standalone, runnable.
    """

    import matplotlib
    matplotlib.use("Agg")  # REQUIRED: no display in container
    import matplotlib.pyplot as plt

    from ase.io import read
    from mace.calculators import mace_mp

    # ── Step 1: Setup ──
    calc = mace_mp(model="medium", default_dtype="float64")
    atoms = read("structure.cif")
    atoms.calc = calc

    # ... complete workflow with progress prints ...

    # ── Step N: Save results ──
    plt.savefig("result.png", dpi=150)
    print("Saved: result.png")
    ```

### Method B: QE DFT

Complete workflow with QE input generation and post-processing scripts.

## Key Parameters

| Parameter | Typical Value | Effect |
|-----------|---------------|--------|
| `ecutwfc` | 60 Ry | Plane-wave cutoff energy |

## Interpreting Results

What the output numbers mean physically.
Include typical ranges for common materials, comparison with literature.
Flag anomalous values (e.g., "B0 < 10 GPa is likely wrong for metals").

## Common Issues

| Issue | Cause | Solution |
|-------|-------|----------|
| Error message | Why it happens | How to fix it |
```

### Code Quality Rules

These are **hard requirements** — PRs that violate them will be rejected:

| Rule | Rationale |
|------|-----------|
| Scripts must be **complete and standalone** | The agent copies the script and runs it directly. No `...` placeholders, no missing imports. |
| `matplotlib.use("Agg")` before any pyplot import | Container has no display server. Scripts crash without this. |
| MACE calculator: `mace_mp(model="medium", default_dtype="float64")` | Standard API. `float64` avoids numerical noise. |
| **No pyiron / atomate2 / AiiDA / FireWorks dependency** | Container doesn't have workflow managers. Port concepts into standalone scripts. |
| All plots saved to files with `print("Saved: filename.png")` | Agent needs to know what files were generated. |
| JSON output for structured results | Machine-readable results enable follow-up analysis. |
| Print progress at each major step | Agent logs these; user can see them via `/watch`. |
| Use `#!/usr/bin/env python3` shebang | Makes scripts executable. |
| Include realistic default parameters | Not placeholders. Use values that work for typical systems (e.g., Si, Cu, Al). |

**Typical skill size:**

| Complexity | Methods | Lines | Python Scripts |
|-----------|---------|-------|----------------|
| Small | 1 method | 300–600 | 1–2 |
| Medium | 2 methods | 600–1,200 | 2–4 |
| Large | 3+ methods + analysis | 1,200–2,500 | 4–9 |

### Step 3: Update All Documentation

This is critical — a skill that isn't indexed is invisible to the agent. You must update **5 files**:

#### 3a. Group-level SKILL.md

Edit `container/skills/<group>/SKILL.md`:

```diff
  ---
- description: Group Name (N sub-skills: existing-1, existing-2)
+ description: Group Name (N+1 sub-skills: existing-1, existing-2, your-new-skill)
  ---

  | Sub-Skill | Directory | Description |
+ | Your New Skill | `your-new-skill/` | One-line description |

  # In the Method Decision Guide:
+ Need <your capability>?
+   --> your-new-skill/  (brief note)
```

#### 3b. Materials-compute skill index

Edit `container/skills/materials-compute/SKILL.md`:

- Update the description line (skill count)
- Add the new sub-skill name to the appropriate group row in the Skill Reference Index table

#### 3c. README.md

- Update header: `**N SKILL.md files across 44 skill groups**`
- Update counts: `**44 groups / M sub-skills / N SKILL.md files**`
- Update the skill group row: increment count, add sub-skill name

#### 3d. README_zh.md

Same changes as README.md, in Chinese.

#### 3e. docs/materials-compute-skills.md

- Update header count
- Add a row to the appropriate section's sub-skill table with description
- Update the Coverage Summary table if sub-skill counts changed

#### Quick count formula

```
New sub-skill count = old count + 1
New SKILL.md count = old count + 1  (each sub-skill has exactly one SKILL.md)
```

### Step 4: Test

#### Local test (recommended)

Run the agent with a prompt that triggers your skill:

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

#### Script validation (minimum)

At minimum, ensure your Python scripts parse correctly:

```bash
python3 -c "import ast; ast.parse(open('container/skills/<group>/<skill>/SKILL.md').read().split('\`\`\`python')[1].split('\`\`\`')[0])"
```

### Step 5: Submit a PR

```bash
git checkout -b add-skill-<name>
git add container/skills/<group>/<your-skill>/SKILL.md
git add container/skills/<group>/SKILL.md              # updated group overview
git add container/skills/materials-compute/SKILL.md     # updated index
git add README.md README_zh.md                          # updated counts
git add docs/materials-compute-skills.md                # updated inventory
git commit -m "Add <name> computation skill"
```

**PR description template:**

```markdown
## New Computation Skill: <name>

**Group:** `<skill-group>`
**Methods:** MACE (Method A), QE DFT (Method B)
**Lines:** ~N lines, M Python scripts

### What it computes
- Brief description

### Test result
- Screenshot or output showing the skill works
- Example material tested on

### Checklist
- [ ] SKILL.md follows the 7-section format
- [ ] All scripts are complete, standalone, runnable
- [ ] matplotlib.use("Agg") used in all scripts
- [ ] No workflow manager dependencies
- [ ] Group SKILL.md updated
- [ ] materials-compute/SKILL.md index updated
- [ ] README.md and README_zh.md updated
- [ ] docs/materials-compute-skills.md updated
```

### Enhancing Existing Skills

To improve an existing skill, follow the same process but focus on:

1. **Adding new methods** — e.g., adding a LAMMPS method to an existing MACE + QE skill
2. **Expanding parameter tables** — More entries, more edge cases
3. **Adding Common Issues** — Real problems you've encountered
4. **Improving scripts** — Better defaults, more robust error handling, more plots

When enhancing, keep all existing content intact. Add new sections after the existing ones. Update the documentation files only if sub-skill counts change (they don't change for enhancements to existing skills).

---

## Contributing Channel / Platform Skills

Channel skills (`.claude/skills/add-*/`) teach Claude Code how to integrate a messaging platform. They modify the MatClaw codebase via the skills engine.

See [Creating Skills — Part 2: Platform Skills](docs/creating-skills.md#part-2-platform-skills) for the full guide, including the SKILL.md template, manifest format, and merge mechanics.

A PR that contributes a channel skill should not modify source files directly — it should contain the **instructions** Claude follows to add the feature. See `/add-telegram` for a good example.

---

## Contributing Examples

Add to `examples/`:

```
examples/<task-name>/
├── feishu-chat-1.png    # Screenshot of agent conversation
└── README.md            # Task description, reference value, agent result
```

Update the results table in `README.md` / `README_zh.md`.

---

## Contributing Benchmark Test Cases

**For domain experts in computational or experimental materials science.** You don't need to write code — just provide real-world tasks and reference answers so we can evaluate how well MatClaw performs and where it needs improvement.

This is one of the most valuable contributions. MatClaw's skills are only as good as the benchmarks that validate them. Tasks from real research problems surface failure modes that synthetic tests miss.

### What Makes a Good Test Case

A good benchmark test case has:

1. **A clear natural language task** — the prompt you'd send to MatClaw, as you would describe it to a knowledgeable colleague
2. **A reference answer** — the expected result with source (literature, your own calculation, experiment)
3. **Difficulty and domain tags** — so we can track coverage across subfields
4. **Evaluation criteria** — how to judge if the result is correct (exact match, within N%, qualitative agreement)

### Difficulty Levels

| Level | Description | Example |
|-------|-------------|---------|
| **Basic** | Single calculation, textbook system | Band gap of Si, bulk modulus of Cu |
| **Intermediate** | Multi-step workflow, parameter decisions | Phonon dispersion of a ternary oxide, adsorption isotherm with charge equilibration |
| **Advanced** | Requires method selection, domain knowledge | DFT+U for correlated oxide, NEB for migration barrier with correct endpoint relaxation |
| **Expert** | Multi-tool pipeline, non-obvious pitfalls | Phase diagram from high-throughput screening, thermal conductivity requiring converged phonon BTE |

### How to Submit

#### Option A: Open a GitHub Issue (easiest)

Use the **Benchmark Test Case** issue template, or structure your issue like this:

```markdown
## Benchmark: [Short Title]

**Domain:** [e.g., electronic structure, catalysis, thermodynamics, MD, MC, experimental comparison]
**Difficulty:** [Basic / Intermediate / Advanced / Expert]
**Computation engine:** [QE / LAMMPS / RASPA3 / MACE / any]

### Task (natural language prompt)

> Calculate the formation energy of a single oxygen vacancy in rutile TiO2 using DFT+U (U=4.2 eV on Ti-3d).

### Reference Answer

- **Value:** 4.2 ± 0.3 eV (depending on supercell size and U value)
- **Source:** Morgan & Watson, PRB 80, 233102 (2009) / your own VASP calculation / experiment
- **Method used for reference:** VASP PAW-PBE, 3×3×4 supercell, 450 eV cutoff, Γ-centered 2×2×2 k-mesh

### Evaluation Criteria

- [ ] Correctly identifies TiO2 as requiring DFT+U
- [ ] Chooses reasonable U value for Ti-3d (3.5–5.0 eV)
- [ ] Uses adequate supercell (at least 2×2×2)
- [ ] Formation energy within 0.5 eV of reference
- [ ] Correctly applies finite-size correction or notes the limitation

### Known Pitfalls

- Using bare PBE (no +U) gives ~1 eV, far too low
- 1×1×1 unit cell has severe finite-size errors
- Oxygen chemical potential choice matters: O2 molecule vs. bulk oxide reference

### Notes (optional)

Any additional context: why this is a meaningful test, what you'd expect a competent researcher to check, common mistakes students make, etc.
```

#### Option B: Submit a PR with structured YAML

Add a file to `benchmarks/`:

```
benchmarks/
├── electronic-structure/
│   ├── si-band-gap.yaml
│   └── tio2-vacancy-formation.yaml
├── thermodynamics/
│   └── al-melting-point.yaml
└── README.md
```

Each YAML file:

```yaml
title: Oxygen vacancy formation energy in rutile TiO2
domain: defects
difficulty: advanced
engine: QE  # or LAMMPS, RASPA3, MACE, any

prompt: >
  Calculate the formation energy of a single oxygen vacancy
  in rutile TiO2 using DFT+U.

reference:
  value: "4.2 ± 0.3 eV"
  source: "Morgan & Watson, PRB 80, 233102 (2009)"
  method: "VASP PAW-PBE+U (U=4.2 eV), 3×3×4 supercell"

evaluation:
  - "Applies DFT+U with reasonable U value (3.5–5.0 eV)"
  - "Uses supercell ≥ 2×2×2"
  - "Formation energy within 0.5 eV of reference"

pitfalls:
  - "Bare PBE gives ~1 eV (too low)"
  - "1×1×1 cell has severe finite-size errors"

tags: [oxide, defect, DFT+U, correlated-electron]
```

### What Happens After You Submit

1. We run MatClaw on your test case and record the result
2. If it passes — great, it becomes part of our benchmark suite
3. If it fails — we file an internal issue to improve the relevant skill
4. Either way, your contribution is credited in the benchmark results

### Ideas for Test Cases We Need

We especially welcome test cases in these areas:

| Area | Why |
|------|-----|
| **Experimental comparison** | Tasks where the reference is a measured value, not another calculation |
| **Method selection** | Tasks where choosing the wrong method gives a plausible but wrong answer |
| **Multi-step workflows** | Tasks requiring 3+ sequential calculations (e.g., relax → bands → DOS → effective mass) |
| **Edge cases** | Unusual crystal symmetries, magnetic ordering, SOC effects, heavy elements |
| **Cross-engine** | Tasks that can be done with both DFT and MLIP, to benchmark accuracy vs. speed tradeoffs |
| **Negative tests** | Tasks where the correct answer is "this method can't reliably compute this" |

You don't need to run MatClaw yourself — just describe the task and the expected answer. We handle the testing.

---

## Code Style

- TypeScript for the orchestrator (`src/`)
- Python for computation scripts in skills
- No linting enforcement currently — just follow existing patterns
- Commit messages: imperative mood, concise ("Add phonon-lifetime skill", not "Added new skill for phonon lifetime calculations")

## Questions?

Open an issue or start a discussion. We're happy to help you plan a skill before you write it.
