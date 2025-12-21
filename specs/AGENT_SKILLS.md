# Agent Skills Specification

> **Status**: Proposal
> **MVP Adapter**: Claude Agent SDK

## Purpose

`AgentSkillsSection` integrates the open [Agent Skills](https://agentskills.io)
standard into weakincentives prompts. Skills are lightweight packages that
extend agent capabilities with procedural knowledge, scripts, and resources.

The MVP targets the Claude Agent SDK adapter, where skills leverage native
filesystem access (Read, Bash) for progressive disclosure and script execution.

## Concepts

### What Are Agent Skills?

Skills are folders containing a `SKILL.md` file with YAML frontmatter and
markdown instructions:

```
my-skill/
├── SKILL.md           # Required: frontmatter + instructions
├── scripts/           # Optional: executable scripts
├── references/        # Optional: detailed documentation
└── assets/            # Optional: templates, data files
```

The `SKILL.md` format:

```markdown
---
name: pdf-processing
description: |
  Extract text and tables from PDFs, fill forms, merge documents.
  Use for any task involving PDF manipulation.
license: MIT
compatibility: Requires poppler-utils for extraction
allowed-tools: Bash(pdftotext:*) Bash(pdfunite:*) Read
---

## Instructions

1. Use `pdftotext` for text extraction
2. Use `pdfunite` to merge multiple PDFs
...
```

### Progressive Disclosure

Skills use a three-tier loading strategy to minimize context overhead:

1. **Metadata** (~50-100 tokens per skill): Name and description loaded at
   prompt render time, injected as XML
2. **Instructions** (<5000 tokens): Full `SKILL.md` loaded on-demand when the
   agent activates the skill
3. **Resources** (as needed): Scripts and references loaded via filesystem
   access during execution

## Integration Design

### AgentSkillsSection

A new `MarkdownSection` subclass that discovers skills from a directory and
renders availability metadata:

```python
from weakincentives.adapters.claude_agent_sdk import AgentSkillsSection
from weakincentives.prompt import MarkdownSection, Prompt, PromptTemplate
from weakincentives.runtime import Session

session = Session(bus=InProcessEventBus())

skills_section = AgentSkillsSection(
    session=session,
    skills_dir="/path/to/skills",           # Root directory to scan
    allowed_skills=("pdf-*", "code-*"),     # Optional glob filter
    denied_skills=("deprecated-*",),        # Optional deny patterns
)

template = PromptTemplate[TaskResult](
    ns="demo",
    key="task-with-skills",
    sections=[
        MarkdownSection(title="Task", key="task", template="..."),
        skills_section,
    ],
)
```

### Skill Discovery

At construction time, `AgentSkillsSection`:

1. Scans `skills_dir` for subdirectories containing `SKILL.md`
2. Parses YAML frontmatter to extract metadata (name, description)
3. Validates skill names against allow/deny patterns
4. Stores skill metadata and paths for prompt injection

### Prompt Injection

The section renders skill availability as XML (the format Claude models expect):

```xml
<available_skills>
  <skill>
    <name>pdf-processing</name>
    <description>Extract text and tables from PDFs...</description>
    <location>skills/pdf-processing</location>
  </skill>
  <skill>
    <name>code-review</name>
    <description>Performs structured code review...</description>
    <location>skills/code-review</location>
  </skill>
</available_skills>
```

The section also includes activation instructions:

```markdown
## Agent Skills

When a task matches an available skill, activate it by reading the full
instructions:

1. Read the skill's SKILL.md file to get detailed instructions
2. Follow the instructions step-by-step
3. Use any scripts or references in the skill directory as needed
```

### Claude Agent SDK Integration

For the Claude Agent SDK adapter, skills integrate naturally:

1. **Workspace mounting**: Skills directory is mounted into the workspace via
   `ClaudeAgentWorkspaceSection` or included in the cwd
2. **Native tool access**: Claude Code's Read, Bash tools enable:
   - Reading full `SKILL.md` instructions on activation
   - Executing scripts from `scripts/` directory
   - Accessing reference materials from `references/`
3. **No MCP bridging needed**: Skills leverage filesystem access, not custom
   tool handlers

```python
from weakincentives.adapters.claude_agent_sdk import (
    AgentSkillsSection,
    ClaudeAgentSDKAdapter,
    ClaudeAgentSDKClientConfig,
    ClaudeAgentWorkspaceSection,
    HostMount,
)

session = Session(bus=InProcessEventBus())

workspace = ClaudeAgentWorkspaceSection(
    session=session,
    mounts=(
        HostMount(host_path="/path/to/repo", mount_path="repo"),
    ),
    allowed_host_roots=("/path/to",),
)

skills = AgentSkillsSection(
    session=session,
    skills_dir="/path/to/skills",
)

template = PromptTemplate[Review](
    ns="review",
    key="with-skills",
    sections=[
        MarkdownSection(title="Task", key="task", template="Review the code..."),
        workspace,
        skills,
    ],
)

adapter = ClaudeAgentSDKAdapter(
    client_config=ClaudeAgentSDKClientConfig(
        cwd=str(workspace.temp_dir),
    ),
)

response = adapter.evaluate(Prompt(template), session=session)
```

## Data Model

### SkillMetadata

Parsed from `SKILL.md` frontmatter:

```python
@dataclass(slots=True, frozen=True)
class SkillMetadata:
    """Skill metadata extracted from SKILL.md frontmatter."""

    name: str                              # 1-64 chars, lowercase alphanumeric + hyphens
    description: str                       # 1-1024 chars
    path: Path                             # Absolute path to skill directory
    license: str | None = None
    compatibility: str | None = None       # Environment requirements
    allowed_tools: tuple[str, ...] = ()    # Tool allowlist (experimental)
    metadata: Mapping[str, str] = field(default_factory=dict)
```

### AgentSkillsSectionParams

```python
@dataclass(slots=True, frozen=True)
class AgentSkillsSectionParams:
    """Parameters for AgentSkillsSection."""

    pass  # Empty placeholder for now
```

### AgentSkillsSection

```python
class AgentSkillsSection(MarkdownSection[AgentSkillsSectionParams]):
    """Section exposing Agent Skills to the prompt."""

    def __init__(
        self,
        *,
        session: Session,
        skills_dir: os.PathLike[str] | str,
        allowed_skills: Sequence[str] = (),     # Glob patterns to include
        denied_skills: Sequence[str] = (),      # Glob patterns to exclude
        mount_path: str = "skills",             # Path within workspace
        accepts_overrides: bool = False,
    ) -> None:
        ...

    @property
    def skills(self) -> tuple[SkillMetadata, ...]:
        """Return discovered skills."""
        ...

    @property
    def session(self) -> Session:
        """Return the session associated with this section."""
        ...

    def clone(self, **kwargs: object) -> AgentSkillsSection:
        """Clone with a new session."""
        ...
```

## Rendering

### Metadata Rendering

The section renders skill availability as XML followed by activation
instructions:

```python
def render_body(
    self,
    params: SupportsDataclass | None,
    *,
    visibility: SectionVisibility | None = None,
    path: tuple[str, ...] = (),
    session: SessionProtocol | None = None,
) -> str:
    lines = ["<available_skills>"]
    for skill in self._skills:
        lines.extend([
            "  <skill>",
            f"    <name>{skill.name}</name>",
            f"    <description>{skill.description}</description>",
            f"    <location>{self._mount_path}/{skill.name}</location>",
            "  </skill>",
        ])
    lines.append("</available_skills>")

    lines.extend([
        "",
        "When a task matches an available skill:",
        "1. Read the skill's SKILL.md file for full instructions",
        "2. Follow the instructions step-by-step",
        "3. Use scripts or references in the skill directory as needed",
    ])

    return "\n".join(lines)
```

### Full vs Summary Visibility

- **FULL**: Renders complete skill list with descriptions
- **SUMMARY**: Renders only skill names as a compact list

## Validation

### Skill Name Validation

Names must follow the Agent Skills specification:

- 1-64 characters
- Lowercase alphanumeric and hyphens only
- Cannot start/end with hyphen
- No consecutive hyphens
- Must match parent directory name

```python
_SKILL_NAME_PATTERN = re.compile(r"^[a-z0-9]([a-z0-9-]*[a-z0-9])?$")

def _validate_skill_name(name: str, dir_name: str) -> None:
    if not 1 <= len(name) <= 64:
        raise SkillValidationError(f"Skill name must be 1-64 chars: {name}")
    if not _SKILL_NAME_PATTERN.match(name):
        raise SkillValidationError(f"Invalid skill name format: {name}")
    if "--" in name:
        raise SkillValidationError(f"Consecutive hyphens not allowed: {name}")
    if name != dir_name:
        raise SkillValidationError(
            f"Skill name '{name}' must match directory name '{dir_name}'"
        )
```

### YAML Frontmatter Parsing

```python
def _parse_skill_frontmatter(content: str) -> dict[str, Any]:
    """Extract YAML frontmatter from SKILL.md content."""
    if not content.startswith("---"):
        raise SkillValidationError("SKILL.md must start with YAML frontmatter")

    end_marker = content.find("\n---", 3)
    if end_marker == -1:
        raise SkillValidationError("SKILL.md frontmatter not terminated")

    frontmatter = content[4:end_marker]
    return yaml.safe_load(frontmatter)
```

## Security Considerations

### Script Execution

Skills may include executable scripts. Security controls:

1. **Sandboxing**: When using `IsolationConfig`, scripts run within the
   configured sandbox with restricted filesystem and network access
2. **Allowlisting**: The `allowed-tools` frontmatter field (experimental)
   declares which tools a skill needs; future versions may enforce this
3. **User confirmation**: For sensitive operations, rely on Claude Code's
   permission system (when not bypassed)

### Path Traversal

The section validates that:

- All skill paths resolve within the configured `skills_dir`
- Mount paths don't escape the workspace root
- Relative paths in skill content are resolved safely

### Untrusted Skills

For skills from untrusted sources:

1. Use `NetworkPolicy.no_network()` to prevent data exfiltration
2. Mount skills read-only when possible
3. Review `allowed-tools` declarations before enabling

## Events

### SkillActivated (Future)

For telemetry, a future version may emit events when skills are activated:

```python
@dataclass(frozen=True)
class SkillActivated:
    """Emitted when an agent reads a skill's full instructions."""

    skill_name: str
    skill_path: str
    timestamp: datetime
```

This requires integration with the adapter's tool hooks to detect when the
agent reads a `SKILL.md` file.

## User Stories

### Story 1: Code review with specialized skills

```python
from dataclasses import dataclass
from weakincentives.adapters.claude_agent_sdk import (
    AgentSkillsSection,
    ClaudeAgentSDKAdapter,
    ClaudeAgentWorkspaceSection,
    HostMount,
)
from weakincentives.prompt import MarkdownSection, Prompt, PromptTemplate
from weakincentives.runtime import InProcessEventBus, Session


@dataclass(frozen=True)
class Review:
    summary: str
    findings: list[str]


session = Session(bus=InProcessEventBus())

workspace = ClaudeAgentWorkspaceSection(
    session=session,
    mounts=(HostMount(host_path="./repo", mount_path="repo"),),
    allowed_host_roots=(".",),
)

skills = AgentSkillsSection(
    session=session,
    skills_dir="./skills",
    allowed_skills=("security-*", "python-*"),
)

template = PromptTemplate[Review](
    ns="review",
    key="security",
    sections=[
        MarkdownSection(
            title="Task",
            key="task",
            template="Review repo/ for security issues. Use available skills.",
        ),
        workspace,
        skills,
    ],
)

try:
    adapter = ClaudeAgentSDKAdapter()
    response = adapter.evaluate(Prompt(template), session=session)
    print(response.output)
finally:
    workspace.cleanup()
```

### Story 2: Skills from multiple directories

```python
# Combine organization skills with project-specific skills
org_skills = AgentSkillsSection(
    session=session,
    skills_dir="/shared/org-skills",
    mount_path="skills/org",
)

project_skills = AgentSkillsSection(
    session=session,
    skills_dir="./project-skills",
    mount_path="skills/project",
)

template = PromptTemplate[Result](
    ns="multi",
    key="skills",
    sections=[
        MarkdownSection(title="Task", key="task", template="..."),
        org_skills,
        project_skills,
    ],
)
```

### Story 3: Filtered skills by capability

```python
# Only include data-processing skills for a data pipeline task
skills = AgentSkillsSection(
    session=session,
    skills_dir="./skills",
    allowed_skills=("csv-*", "json-*", "parquet-*"),
    denied_skills=("*-deprecated",),
)
```

## Implementation Plan

### Phase 1: MVP (Claude Agent SDK)

1. Implement `SkillMetadata` dataclass and YAML parsing
2. Implement `AgentSkillsSection` with directory scanning
3. Add skill name validation per specification
4. Render XML skill availability block
5. Integration tests with Claude Agent SDK adapter

### Phase 2: Enhanced Discovery

1. Recursive skill discovery (nested directories)
2. Skill versioning support via metadata
3. Skill compatibility checking (Python version, OS, packages)

### Phase 3: Other Adapters

1. OpenAI adapter: Inject skill instructions directly (no filesystem)
2. LiteLLM adapter: Same approach as OpenAI
3. Custom skill activation tool for adapters without filesystem access

### Phase 4: Telemetry

1. `SkillActivated` event emission
2. Skill usage metrics in `BudgetTracker`
3. LangSmith integration for skill tracing

## File Organization

```
src/weakincentives/
├── adapters/
│   └── claude_agent_sdk/
│       ├── skills.py              # AgentSkillsSection implementation
│       └── __init__.py            # Export AgentSkillsSection
└── contrib/
    └── skills/                    # Future: adapter-agnostic utilities
        ├── metadata.py            # SkillMetadata, parsing
        └── validation.py          # Name/structure validation
```

## Dependencies

- `pyyaml` for YAML frontmatter parsing (already in dependencies)
- No additional dependencies required for MVP

## Compatibility

The Agent Skills format is supported by:

- Claude Code
- Cursor
- OpenCode
- GitHub Copilot
- VS Code (various extensions)
- OpenAI Codex
- Goose, Amp, Letta

Skills authored for weakincentives will work with these tools and vice versa.
