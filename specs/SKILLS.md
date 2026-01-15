# Skills Specification

> Agent Skills: lightweight, open format for extending AI agent capabilities.

This document describes the [Agent Skills specification](https://agentskills.io)
and WINK's implementation for Claude Code.

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

1. **Discovery**: Load name/description only
2. **Activation**: Read full instructions when task matches
3. **Execution**: Load referenced files as needed

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

### Architecture

**Core Library (`weakincentives.skills`):**
- Types: `Skill`, `SkillMount`, `SkillConfig`
- Validation: `validate_skill()`, `validate_skill_name()`
- Errors: `SkillError`, `SkillValidationError`, `SkillNotFoundError`, `SkillMountError`
- Constants: `MAX_SKILL_FILE_BYTES` (1 MiB), `MAX_SKILL_TOTAL_BYTES` (10 MiB)

**Adapter (`adapters/claude_agent_sdk`):**
- `IsolationConfig.skills`: Integration point
- `EphemeralHome._mount_skills()`: Copies to `.claude/skills/`

### Data Model

| Type | Fields |
|------|--------|
| `Skill` | `name`, `source`, `content` |
| `SkillMount` | `source`, `name` (optional), `enabled` |
| `SkillConfig` | `skills`, `validate_on_mount` |

### Name Resolution

1. Explicit `mount.name` if provided
2. Directory name if source is directory
3. File stem (without `.md`) if source is file

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

### Basic Mounting

```python
adapter = ClaudeAgentSDKAdapter(
    client_config=ClaudeAgentSDKClientConfig(
        isolation=IsolationConfig(
            skills=SkillConfig(
                skills=(
                    SkillMount(Path("./skills/code-review")),
                    SkillMount(Path("./skills/testing")),
                )
            ),
        ),
    ),
)
```

### Auto-Discovery

```python
skill_mounts = tuple(
    SkillMount(source=skill_dir)
    for skill_dir in SKILLS_ROOT.iterdir()
    if skill_dir.is_dir() and (skill_dir / "SKILL.md").exists()
)
skills = SkillConfig(skills=skill_mounts)
```

### Conditional Skills

```python
SkillMount(source=Path("./skills/experimental"), enabled=include_experimental)
```

### Disable Validation

```python
SkillConfig(skills=(...), validate_on_mount=False)
```

## Security Considerations

- **Path traversal**: Names sanitized; `/`, `\`, `..` raise error
- **Symlinks**: Disabled by default
- **Size limits**: 10 MiB per skill

## Limitations

- **No runtime updates**: Requires recreating adapter
- **No dependencies**: Manual composition via SkillConfig
- **No templating**: Use prompt composition for dynamic content

## Related Specifications

- `specs/CLAUDE_AGENT_SDK.md` - Parent adapter specification
- `specs/WORKSPACE.md` - Workspace and mount patterns
- `specs/PROMPTS.md` - Prompt composition alternative
