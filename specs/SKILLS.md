# Skills Specification

> **Claude Agent SDK**: `claude-agent-sdk>=0.1.15`

## Purpose

Skills package domain-specific expertise—workflows, context, and best
practices—into reusable units that extend Claude's capabilities. This
specification defines:

- `Skill` - A section type wrapping procedural knowledge with optional tools
- `SkillsSection` - A container that manages skill installation and disclosure

Skills without custom tools are automatically installed as native Claude skills
in the isolated home directory. Skills with custom tools remain in the prompt
and participate in progressive disclosure.

## Guiding Principles

- **Native Integration**: Skills without tools become first-class Claude skills,
  benefiting from Claude's progressive loading (metadata → body → resources).
- **Tool Retention**: Skills with custom tools stay in the prompt where tool
  handlers can be invoked via MCP bridging.
- **Summary by Default**: All skills render with `SUMMARY` visibility initially,
  reducing token usage until the model requests expansion.
- **Hermetic Setup**: Native skill installation requires isolation to write to
  the ephemeral `~/.claude/skills/` directory.

## Core Components

### Skill

`Skill` is a section type that packages procedural knowledge with optional
tooling. It mirrors `MarkdownSection` but defaults to `SUMMARY` visibility and
carries metadata for native installation.

```python
from dataclasses import dataclass
from weakincentives.prompt import SectionVisibility

@dataclass(slots=True, frozen=True)
class SkillParams:
    """Default params for skills (can be specialized)."""
    pass


class Skill[ParamsT: SupportsDataclass = SkillParams](Section[ParamsT]):
    """Section type for domain-specific expertise."""

    def __init__(
        self,
        *,
        name: str,                      # 1-64 chars, lowercase/hyphens/numbers
        description: str,               # 1-1024 chars, plain text
        title: str,                     # Display heading
        template: str,                  # Body content (string.Template syntax)
        key: str,                       # Section identifier
        default_params: ParamsT | None = None,
        children: Sequence[Section] | None = None,
        enabled: Callable[[ParamsT], bool] | None = None,
        tools: Sequence[Tool] | None = None,
        accepts_overrides: bool = True,
        summary: str | None = None,     # Summary template (auto-generated if None)
        visibility: VisibilitySelector = SectionVisibility.SUMMARY,  # Default SUMMARY
    ) -> None: ...

    @property
    def has_tools(self) -> bool:
        """True if this skill has custom tool handlers."""
        return len(self.tools) > 0

    def to_skill_md(self, params: ParamsT) -> str:
        """Render as SKILL.md content with YAML frontmatter."""
        ...
```

**Field Constraints:**

- `name`: Maximum 64 characters, pattern `^[a-z0-9][a-z0-9-]{0,63}$`
- `description`: Maximum 1024 characters, non-empty, no XML tags
- `title`: Required, non-empty
- `template`: Required, uses `string.Template` syntax for substitution

**Default Summary Generation:**

When `summary` is `None`, the skill auto-generates a summary from the
description:

```
[Skill: {name}] {description}
```

### SkillsSection

`SkillsSection` is a container that holds multiple `Skill` instances and
orchestrates their installation or disclosure.

```python
class SkillsSection(Section[SkillsSectionParams]):
    """Container for skill management and installation."""

    def __init__(
        self,
        *,
        title: str = "Skills",
        key: str = "skills",
        skills: Sequence[Skill] = (),
        enabled: Callable[[SkillsSectionParams], bool] | None = None,
        accepts_overrides: bool = True,
    ) -> None: ...

    @property
    def native_skills(self) -> tuple[Skill, ...]:
        """Skills without custom tools (installed natively)."""
        return tuple(s for s in self.skills if not s.has_tools)

    @property
    def prompt_skills(self) -> tuple[Skill, ...]:
        """Skills with custom tools (kept in prompt)."""
        return tuple(s for s in self.skills if s.has_tools)

    def install_native_skills(
        self,
        *,
        skills_dir: Path,
        params_lookup: Mapping[type, SupportsDataclass],
    ) -> tuple[Path, ...]:
        """Write SKILL.md files to the skills directory."""
        ...
```

## Skill File Format

Native skills are written as SKILL.md files with YAML frontmatter:

```yaml
---
name: code-review
description: Provides code review workflows and best practices for Python projects.
---

# Code Review Workflow

Follow these steps when reviewing code:

1. Check for security vulnerabilities
2. Verify test coverage
3. Review naming conventions
4. Assess performance implications

## Security Checklist

- [ ] No hardcoded secrets
- [ ] Input validation present
- [ ] SQL injection prevented
...
```

The `to_skill_md()` method generates this format by:

1. Rendering the skill template with bound parameters
2. Prepending YAML frontmatter with `name` and `description`

## Installation Lifecycle

```mermaid
flowchart TB
    subgraph Render["Prompt Rendering"]
        Skills["SkillsSection"]
        Native["Native Skills<br/>(no tools)"]
        Prompt["Prompt Skills<br/>(with tools)"]
    end

    subgraph Install["Native Installation"]
        Check["IsolationConfig?"]
        Dir["Create ~/.claude/skills/"]
        Write["Write SKILL.md files"]
    end

    subgraph Disclose["Progressive Disclosure"]
        Summary["Render with SUMMARY"]
        Open["open_sections tool"]
        Full["Expand to FULL"]
    end

    Skills --> Native
    Skills --> Prompt
    Native --> Check
    Check -->|Yes| Dir --> Write
    Check -->|No| Summary
    Prompt --> Summary
    Summary --> Open --> Full
```

### Installation Steps (Claude Agent SDK)

When `ClaudeAgentSDKAdapter` evaluates a prompt containing `SkillsSection`:

1. **Detect Skills**: Locate `SkillsSection` in rendered sections
2. **Partition**: Split into `native_skills` and `prompt_skills`
3. **Check Isolation**: Verify `IsolationConfig` is set (hermetic requirement)
4. **Create Directory**: Ensure `{ephemeral_home}/.claude/skills/` exists
5. **Install Native Skills**:
   - For each skill without tools, call `skill.to_skill_md(params)`
   - Write to `{skills_dir}/{skill.name}/SKILL.md`
6. **Exclude from Prompt**: Native skills are not rendered into prompt text
7. **Render Prompt Skills**: Skills with tools render with their default
   `SUMMARY` visibility

### Without Isolation (Non-Hermetic)

When `IsolationConfig` is not set:

- All skills render into the prompt with `SUMMARY` visibility
- Native skill installation is skipped (no filesystem access)
- Warning logged: "Skills installation requires hermetic isolation"

## Progressive Disclosure

Skills with tools participate in the standard progressive disclosure system:

1. **Initial Render**: Skill renders with `SUMMARY` visibility
2. **Summary Suffix**: Appended automatically with `open_sections` instruction
3. **Model Request**: Model calls `open_sections` with skill key
4. **Expansion**: `VisibilityExpansionRequired` updates session state
5. **Re-render**: Skill renders with `FULL` visibility, exposing tools

### Summary Suffix for Skills

Skills append a skill-aware suffix:

```
---
[This skill is summarized. Call `open_sections` with key "skills.code-review"
to view full workflow and access tools: lint_code, run_tests.]
```

## Configuration

### SkillsSectionParams

```python
@dataclass(slots=True, frozen=True)
class SkillsSectionParams:
    """Parameters for SkillsSection rendering."""
    enabled_skills: frozenset[str] | None = None  # Filter by name, None = all
```

When `enabled_skills` is set, only skills with matching names are processed.

### Adapter Integration

The adapter extends `IsolationConfig` with skills support:

```python
@dataclass(slots=True, frozen=True)
class IsolationConfig:
    # ... existing fields ...
    install_skills: bool = True  # Enable native skill installation
```

When `install_skills=False`, skills are treated as prompt-only regardless of
whether they have tools.

## Events

### SkillsInstalled

Published after native skills are written to disk:

```python
@FrozenDataclass()
class SkillsInstalled:
    prompt_ns: str
    prompt_key: str
    skills: tuple[str, ...]           # Installed skill names
    skills_dir: str                    # Installation path
    session_id: UUID | None
    created_at: datetime
    event_id: UUID = field(default_factory=uuid4)
```

### SkillExpanded

Published when a skill expands from `SUMMARY` to `FULL`:

```python
@FrozenDataclass()
class SkillExpanded:
    skill_name: str
    skill_key: str                     # Dot-notation path
    prompt_ns: str
    prompt_key: str
    session_id: UUID | None
    created_at: datetime
    event_id: UUID = field(default_factory=uuid4)
```

## User Stories

### Story 1: Native skill installation (no tools)

As a developer, I want to provide domain expertise that Claude loads
progressively via its native skill system.

```python
from weakincentives.adapters.claude_agent_sdk import (
    ClaudeAgentSDKAdapter,
    ClaudeAgentSDKClientConfig,
    IsolationConfig,
)
from weakincentives.prompt import Prompt, PromptTemplate, MarkdownSection
from weakincentives.prompt.skills import Skill, SkillsSection
from weakincentives.runtime import InProcessEventBus, Session


code_review_skill = Skill(
    name="code-review",
    description="Code review workflows and checklists for Python projects.",
    title="Code Review",
    key="code-review",
    template="""
    # Code Review Workflow

    1. **Security**: Check for hardcoded secrets, SQL injection, XSS
    2. **Tests**: Verify coverage meets 80% threshold
    3. **Style**: Run ruff, check docstrings
    4. **Performance**: Profile hot paths

    ## Severity Levels

    - **Critical**: Security vulnerabilities, data loss risks
    - **Major**: Missing tests, breaking changes
    - **Minor**: Style issues, documentation gaps
    """,
)

refactoring_skill = Skill(
    name="refactoring",
    description="Safe refactoring patterns and techniques.",
    title="Refactoring",
    key="refactoring",
    template="""
    # Refactoring Patterns

    ## Extract Method
    When a code fragment can be grouped together...

    ## Rename Symbol
    Use IDE support to safely rename across codebase...
    """,
)

session = Session(bus=InProcessEventBus())

template = PromptTemplate[None](
    ns="review",
    key="python-review",
    sections=[
        MarkdownSection(
            title="Task",
            key="task",
            template="Review the Python code in the workspace.",
        ),
        SkillsSection(
            skills=[code_review_skill, refactoring_skill],
        ),
    ],
)

adapter = ClaudeAgentSDKAdapter(
    client_config=ClaudeAgentSDKClientConfig(
        isolation=IsolationConfig(),  # Enables hermetic mode
    ),
)

# Skills are installed to ~/.claude/skills/{name}/SKILL.md
# Claude loads them progressively via native skill system
response = adapter.evaluate(Prompt(template), session=session)
```

### Story 2: Skill with custom tools (prompt-based)

As a platform team, I want a skill that provides both guidance and actionable
tools, keeping tools accessible via MCP.

```python
from dataclasses import dataclass
from weakincentives.prompt import Tool, ToolContext, ToolResult
from weakincentives.prompt.skills import Skill, SkillsSection


@dataclass(frozen=True)
class LintParams:
    paths: tuple[str, ...]


@dataclass(frozen=True)
class LintResult:
    issues: int
    fixed: int

    def render(self) -> str:
        return f"Found {self.issues} issues, fixed {self.fixed}"


def lint_handler(params: LintParams, *, context: ToolContext) -> ToolResult[LintResult]:
    # Run linter on paths
    return ToolResult(message="Linting complete", value=LintResult(issues=5, fixed=3))


lint_tool = Tool[LintParams, LintResult](
    name="lint_code",
    description="Run ruff linter on specified paths",
    handler=lint_handler,
)

linting_skill = Skill(
    name="linting",
    description="Code linting workflows with automated fixes.",
    title="Linting",
    key="linting",
    template="""
    # Linting Workflow

    Use the `lint_code` tool to check Python files:

    1. Run on changed files first
    2. Review unfixable issues manually
    3. Re-run to verify fixes

    ## Configuration

    Ruff settings in pyproject.toml control rules.
    """,
    tools=[lint_tool],  # Has tools → stays in prompt
)

# This skill renders with SUMMARY visibility initially.
# When expanded, the lint_code tool becomes available via MCP.
```

### Story 3: Mixed skills (native + prompt)

As an architect, I want some skills installed natively for low-overhead context,
while others with tools remain in the prompt.

```python
skills_section = SkillsSection(
    skills=[
        # Native: no tools, installed to filesystem
        Skill(
            name="architecture",
            description="System architecture principles.",
            title="Architecture",
            key="architecture",
            template="...",
        ),
        # Prompt: has tools, subject to disclosure
        Skill(
            name="testing",
            description="Testing workflows with execution tools.",
            title="Testing",
            key="testing",
            template="...",
            tools=[run_tests_tool, coverage_tool],
        ),
    ],
)

# architecture → installed to ~/.claude/skills/architecture/SKILL.md
# testing → rendered in prompt with SUMMARY, tools bridged via MCP
```

### Story 4: Conditional skill enablement

As an operator, I want to enable only specific skills based on runtime context.

```python
@dataclass(slots=True, frozen=True)
class ReviewParams:
    language: str


skills_section = SkillsSection(
    skills=[
        Skill(
            name="python-review",
            description="Python-specific review guidance.",
            title="Python Review",
            key="python-review",
            template="...",
            enabled=lambda p: p.language == "python",
        ),
        Skill(
            name="typescript-review",
            description="TypeScript-specific review guidance.",
            title="TypeScript Review",
            key="typescript-review",
            template="...",
            enabled=lambda p: p.language == "typescript",
        ),
    ],
)

# Bind params to control which skills are active
prompt = Prompt(template).bind(ReviewParams(language="python"))
# Only python-review skill is enabled
```

## Validation Rules

### Skill Validation

- `name` must match `^[a-z0-9][a-z0-9-]{0,63}$`
- `description` must be 1-1024 characters, no XML tags
- `title` must be non-empty
- `template` must be non-empty
- Duplicate skill names within a `SkillsSection` raise `PromptValidationError`

### Installation Validation

- Without `IsolationConfig`, skill installation logs a warning and skips
- Skills directory must be writable (or isolation setup fails)
- Invalid skill names fail at construction, not installation

## Error Handling

### Exception Types

- `PromptValidationError` - Invalid skill name, description, or duplicates
- `SkillInstallationError` - Filesystem write failures during installation
- `VisibilityExpansionRequired` - Model requests skill expansion (handled by
  adapter)

```python
class SkillInstallationError(WinkError, RuntimeError):
    """Failed to install skill to filesystem."""
    skill_name: str
    skills_dir: Path
    cause: Exception
```

## Filesystem Layout

Native skills are written to the ephemeral home's skill directory:

```
{ephemeral_home}/
└── .claude/
    ├── settings.json          # Sandbox/network config
    └── skills/
        ├── code-review/
        │   └── SKILL.md
        └── refactoring/
            └── SKILL.md
```

Each skill gets its own directory to support future resource bundling (scripts,
reference docs, templates).

## Operational Notes

- Native skills reduce prompt size but add filesystem I/O during setup
- Skills with tools always consume prompt tokens (summary or full)
- The `open_sections` tool is injected when any skill has `SUMMARY` visibility
- Skill expansion is logged via `SkillExpanded` events for observability
- Use `enabled_skills` filter to reduce installed skills in constrained
  environments

## Limitations

- **Native skills are read-only**: Claude cannot modify installed skills
- **No dynamic skill discovery**: Skills must be declared in the prompt template
- **Single-level resources**: Skill directories support only SKILL.md (no nested
  resources yet)
- **Isolation required**: Native installation needs hermetic setup; otherwise
  all skills go to prompt
- **No skill dependencies**: Skills cannot declare dependencies on other skills
