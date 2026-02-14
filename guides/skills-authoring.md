# Skills Authoring

*Canonical spec: [specs/SKILLS.md](../specs/SKILLS.md)*

Skills are a lightweight way to extend Claude Code's capabilities. A skill is a
directory (or single file) containing instructions and optional resources that
the agent loads when relevant to a task.

This guide covers how to create skills and mount them with WINK.

## When to Use Skills

Skills are useful when you want to:

- **Add domain expertise**: Code review checklists, style guides, debugging
  procedures
- **Bundle instructions with resources**: Reference files, scripts, templates
- **Share capabilities**: Package reusable agent behaviors

Skills follow the [Agent Skills specification](https://agentskills.io), an open
format supported by Claude Code and other tools.

## Skill Structure

A skill is either a directory containing `SKILL.md` or a standalone markdown
file:

```
my-skill/
├── SKILL.md          # Required: metadata + instructions
├── scripts/          # Optional: executable scripts
├── references/       # Optional: reference files
├── examples/         # Optional: example files
└── assets/           # Optional: images, templates
```

Or as a single file:

```
my-skill.md           # Standalone skill (name derived from filename)
```

## The SKILL.md Format

Every skill needs a `SKILL.md` file with YAML frontmatter and markdown
instructions. The frontmatter is delimited by `---` lines:

```yaml
---
name: code-review
description: Perform thorough code reviews checking for security vulnerabilities, error handling, test coverage, and performance issues.
---
```

After the frontmatter, include markdown instructions:

```markdown
# Code Review Skill

You are a thorough code reviewer. When reviewing code:

## Review Checklist

- Check for security vulnerabilities (injection, XSS, auth bypass)
- Verify error handling covers edge cases
- Ensure tests cover new functionality

## Output Format

Structure your review as:

1. **Summary**: One-paragraph overview
2. **Issues**: Problems found (severity: high/medium/low)
3. **Suggestions**: Non-blocking improvements
```

### Required Fields

| Field | Constraints |
|-------|-------------|
| `name` | 1-64 chars, lowercase letters/numbers/hyphens only, no `--`, no leading/trailing `-` |
| `description` | 1-1024 chars, human-readable purpose |

### Optional Fields

| Field | Constraints | Description |
|-------|-------------|-------------|
| `license` | String | License reference (e.g., "MIT", "Apache-2.0") |
| `compatibility` | Max 500 chars | Environment requirements |
| `metadata` | Dict[str, str] | Arbitrary key-value pairs |
| `allowed-tools` | String | Pre-approved tools (experimental) |

### Name Validation

Skill names must match the pattern `^[a-z0-9]+(-[a-z0-9]+)*$`:

```
valid:   code-review, python-style, my-tool-v2
invalid: Code-Review (uppercase), code--review (consecutive hyphens)
invalid: -code-review (leading hyphen), code_review (underscores)
```

For directory skills, the `name` field must match the directory name.

## Writing Effective Instructions

The markdown body of `SKILL.md` is injected into the agent's context when the
skill is activated. Write instructions as if you're briefing a colleague:

**Be specific about scope.** State what languages, frameworks, or contexts the
skill applies to.

**Use checklists for systematic tasks.** List items the agent should verify or
complete.

**Provide concrete examples.** Show good and bad patterns with brief
explanations. One example often beats three paragraphs.

**Keep instructions focused.** A 200-line skill that tries to cover everything
is less useful than three focused 50-line skills.

## Adding Resources

Skills can bundle additional files that the agent can reference:

```
testing-skill/
├── SKILL.md
├── scripts/
│   └── run-tests.sh
├── references/
│   └── test-patterns.md
├── examples/
│   └── sample-test.py
└── assets/
    └── workflow.png
```

Reference these in your instructions:

```
See `references/test-patterns.md` for the canonical test structure.
Use `scripts/run-tests.sh` to execute the test suite.
```

## Size Limits

Skills have size constraints to prevent bloated context:

- **Per-file limit**: 1 MiB
- **Per-skill limit**: 10 MiB total

Keep skills lean. If you're hitting limits, consider splitting into multiple
skills or moving large assets elsewhere.

## Mounting Skills with WINK

Skills are attached at the section level using the `skills` parameter on
`Section` (or any section subclass like `MarkdownSection`):

```python nocheck
from pathlib import Path
from weakincentives.prompt import MarkdownSection
from weakincentives.skills import SkillMount

section = MarkdownSection(
    title="Instructions",
    key="instructions",
    template="You are a code reviewer.",
    skills=(
        SkillMount(Path("./skills/code-review")),
        SkillMount(Path("./skills/python-style")),
    ),
)
```

When the section is enabled, its skills are included in the agent's context.
This co-locates skills with the instructions that use them, following the same
pattern as tools.

### Auto-Discovery

Mount all skills in a directory:

```python nocheck
from pathlib import Path
from weakincentives.skills import SkillMount

SKILLS_ROOT = Path("./skills")

skill_mounts = tuple(
    SkillMount(source=skill_dir)
    for skill_dir in SKILLS_ROOT.iterdir()
    if skill_dir.is_dir() and (skill_dir / "SKILL.md").exists()
)

section = MarkdownSection(
    title="Instructions",
    key="instructions",
    template="...",
    skills=skill_mounts,
)
```

### Conditional Skills

Enable or disable skills at runtime:

```python nocheck
SkillMount(
    source=Path("./skills/experimental"),
    enabled=config.include_experimental,
)
```

Disabled skills are not copied to the agent's skill directory.

### Name Override

Override the skill name when mounting. This is primarily useful for **file skills**
where there's no directory name constraint:

```python nocheck
SkillMount(
    source=Path("./skills/code-reviewer.md"),
    name="code-review",  # Mount as "code-review" instead of "code-reviewer"
)
```

For directory skills, the `name` field in SKILL.md must match the source
directory name to pass validation.

## Complete Example

Here's a complete example of a Python testing skill.

**Directory structure:**

```
skills/pytest-patterns/
├── SKILL.md
└── references/
    └── fixtures.md
```

**SKILL.md frontmatter:**

```yaml
---
name: pytest-patterns
description: Write effective pytest tests with proper fixtures, parametrization, and assertions.
---
```

**SKILL.md body** (markdown instructions for test patterns, fixture usage,
parametrization with `@pytest.mark.parametrize`, and assertion best practices).

**references/fixtures.md** contains common fixture patterns like database
sessions and temporary files that the skill instructions can reference.

**Mount the skill on a section:**

```python nocheck
from pathlib import Path
from weakincentives.prompt import MarkdownSection
from weakincentives.skills import SkillMount

section = MarkdownSection(
    title="Testing",
    key="testing",
    template="Write tests following our patterns.",
    skills=(
        SkillMount(Path("./skills/pytest-patterns")),
    ),
)
```

See `demo-skills/` in the repository for complete working examples:

- `code-review/` - Code review checklist
- `python-style/` - Python style guidelines
- `ascii-art/` - ASCII art generation

## Troubleshooting

### SkillNotFoundError

The source path doesn't exist:

```
SkillNotFoundError: Skill source not found: ./skills/missing
```

Check that the path is correct and the directory/file exists.

### SkillValidationError

The skill structure is invalid:

```
SkillValidationError: Missing required field 'name' in frontmatter
```

Ensure `SKILL.md` has valid YAML frontmatter with `name` and `description`.

### SkillMountError

Mounting failed due to name conflicts or I/O errors:

```
SkillMountError: Duplicate skill name: code-review
```

Each skill must have a unique name. Use the `name` parameter on `SkillMount` to
override if needed.

## What's Next

- [Claude Agent SDK](claude-agent-sdk.md): Full adapter configuration
- [Progressive Disclosure](progressive-disclosure.md): Manage context size
