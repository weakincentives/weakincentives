# Skills Specification

> Agent Skills: lightweight, open format for extending AI agent capabilities.

This document describes the [Agent Skills specification](https://agentskills.io)
and WINK's implementation.

**Implementation:** `src/weakincentives/skills/`

## What are Skills?

A skill is a directory containing `SKILL.md` with metadata and instructions.
Can also bundle scripts, templates, and references.

```
my-skill/
├── SKILL.md          # Required
├── scripts/          # Optional
├── references/       # Optional
└── assets/           # Optional
```

### Progressive Disclosure

Skills follow the same progressive disclosure rules as tools:

1. Skills attached to enabled sections are collected during rendering
1. Skills attached to sections with `SUMMARY` visibility are **not** collected
1. When a section is expanded (via `open_sections`), its skills become available

This allows skills to be conditionally activated based on section visibility,
matching the behavior of tools.

## Agent Skills Specification

### SKILL.md Format

YAML frontmatter + Markdown instructions:

```markdown
---
name: pdf-processing
description: Extract text and tables from PDF files.
---

# PDF Processing
...
```

### Required Fields

| Field | Constraints |
|-------|-------------|
| `name` | 1-64 chars, lowercase/numbers/hyphens, no `--`, no start/end `-` |
| `description` | 1-1024 chars |

### Optional Fields

| Field | Constraints | Description |
|-------|-------------|-------------|
| `license` | String | License reference |
| `compatibility` | Max 500 chars | Environment requirements |
| `metadata` | Dict[str, str] | Arbitrary key-value |
| `allowed-tools` | String | Pre-approved tools (experimental) |

## WINK Implementation

### Section-Level Attachment

Skills are attached to prompt sections via the `skills` parameter, following
the same pattern as tools:

```python
from pathlib import Path
from weakincentives.prompt import MarkdownSection
from weakincentives.skills import SkillMount

section = MarkdownSection(
    title="Code Review",
    key="code-review",
    template="Review the code for issues.",
    skills=(
        SkillMount(Path("./skills/code-review")),
        SkillMount(Path("./skills/security-audit")),
    ),
)
```

Skills are collected from enabled sections during prompt rendering and made
available to the adapter for mounting.

### Architecture

**Core Library (`weakincentives.skills`):**

- Types: `Skill`, `SkillMount`
- Validation: `validate_skill()`, `validate_skill_name()`
- Errors: `SkillError`, `SkillValidationError`, `SkillNotFoundError`, `SkillMountError`
- Constants: `MAX_SKILL_FILE_BYTES` (1 MiB), `MAX_SKILL_TOTAL_BYTES` (10 MiB)

**Prompt Integration (`weakincentives.prompt`):**

- `Section.skills`: Tuple of `SkillMount` attached to the section
- `RenderedPrompt.skills`: Collected skills from enabled sections

**Adapter (`src/weakincentives/adapters/claude_agent_sdk`):**

- Reads skills from `RenderedPrompt.skills`
- `EphemeralHome._mount_skills()`: Copies to `$HOME/.claude/skills/`

### Data Model

| Type | Fields |
|------|--------|
| `Skill` | `name`, `source`, `content` |
| `SkillMount` | `source`, `name` (optional) |

### Name Resolution

1. Explicit `mount.name` if provided
1. Directory name if source is directory
1. File stem (without `.md`) if source is file

### Destination Layout

| Source Type | Destination |
|-------------|-------------|
| Directory | `{skills_dir}/{name}/` (recursive) |
| File | `{skills_dir}/{name}/SKILL.md` (wrapped) |

### Validation Errors

| Error | Condition |
|-------|-----------|
| `SkillNotFoundError` | Source doesn't exist |
| `SkillValidationError` | Missing SKILL.md, invalid frontmatter, constraint violation |
| `SkillMountError` | Invalid name format, duplicate names, I/O error, size exceeded |

## Usage Examples

### Attaching Skills to Sections

```python
from pathlib import Path
from weakincentives.prompt import MarkdownSection, PromptTemplate
from weakincentives.skills import SkillMount

# Section with skills
review_section = MarkdownSection(
    title="Code Review",
    key="code-review",
    template="Review the following code for issues.",
    skills=(
        SkillMount(Path("./skills/code-review")),
        SkillMount(Path("./skills/testing")),
    ),
)

# Create prompt with the section
template = PromptTemplate(
    ns="demo",
    key="reviewer",
    sections=[review_section],
)
```

### Conditional Skills via Section Visibility

Skills follow section visibility rules. Attach skills to sections with
`visibility=SUMMARY` to defer skill loading until the section is expanded:

```python
from weakincentives.prompt import MarkdownSection, SectionVisibility
from weakincentives.skills import SkillMount

# Skills only loaded when section is expanded
advanced_section = MarkdownSection(
    title="Advanced Analysis",
    key="advanced",
    template="Perform deep code analysis.",
    summary="Advanced analysis capabilities available on request.",
    visibility=SectionVisibility.SUMMARY,
    skills=(
        SkillMount(Path("./skills/deep-analysis")),
        SkillMount(Path("./skills/performance-profiling")),
    ),
)
```

### Auto-Discovery from Directory

```python
from pathlib import Path
from weakincentives.skills import SkillMount

SKILLS_ROOT = Path("./skills")

skill_mounts = tuple(
    SkillMount(source=skill_dir)
    for skill_dir in SKILLS_ROOT.iterdir()
    if skill_dir.is_dir() and (skill_dir / "SKILL.md").exists()
)

section = MarkdownSection(
    title="Tasks",
    key="tasks",
    template="Complete the assigned tasks.",
    skills=skill_mounts,
)
```

### Name Override

```python
SkillMount(
    source=Path("./internal/review-v2"),
    name="code-review",  # Override derived name
)
```

## Security Considerations

- **Path traversal**: Names sanitized; `/`, `\`, `..` raise error
- **Symlinks**: Disabled by default
- **Size limits**: 10 MiB per skill

## Limitations

- **No runtime updates**: Skills are collected at render time
- **No dependencies**: Compose skills manually via section structure
- **No templating**: Use prompt composition for dynamic content

## Related Specifications

- `specs/PROMPTS.md` - Prompt system, section attachment
- `specs/CLAUDE_AGENT_SDK.md` - Adapter that mounts skills
- `specs/WORKSPACE.md` - Workspace and mount patterns
