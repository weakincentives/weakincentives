# Skill Mounting Specification

## Purpose

This specification defines how WINK mounts skills into the hermetic workspace
for Claude Code discovery. Skills are markdown files or directories that
Claude Code loads from `.claude/skills/` to provide specialized behavior,
knowledge, or workflows. By mounting skills at isolation setup time, WINK
enables declarative skill composition without modifying prompts or requiring
Claude Code internals.

## Guiding Principles

- **Passthrough over abstraction**: Mount skills for native Claude Code
  discovery rather than re-implementing skill rendering in WINK.
- **Hermetic by default**: Skills are copied into the ephemeral home's
  `.claude/skills/` directory, isolated from the host's configuration.
- **Declarative composition**: Skills are specified as configuration, not
  imperative setup code.
- **Fail-fast validation**: Invalid skill paths or malformed skill files
  surface errors before Claude Code spawns.

## Data Model

### SkillMount

```python
@FrozenDataclass()
class SkillMount:
    """Mount a skill into the hermetic environment.

    Attributes:
        source: Path to a skill file (SKILL.md) or skill directory on the
            host filesystem. Relative paths are resolved against the current
            working directory.
        name: Optional skill name override. If None, derived from the source
            path (directory name or filename without extension).
        enabled: Whether the skill is active. Disabled skills are not copied.
            Defaults to True.
    """

    source: Path
    name: str | None = None
    enabled: bool = True
```

### SkillConfig

```python
@FrozenDataclass()
class SkillConfig:
    """Skills to install in the hermetic environment.

    Attributes:
        skills: Tuple of skill mounts to copy into the workspace.
        validate_on_mount: If True, validate skill structure before copying.
            Validation checks for required SKILL.md file in directories.
            Defaults to True.
    """

    skills: tuple[SkillMount, ...] = ()
    validate_on_mount: bool = True
```

### Integration with IsolationConfig

```python
@FrozenDataclass()
class IsolationConfig:
    """Configuration for hermetic SDK isolation.

    Attributes:
        network_policy: Network access constraints.
        sandbox: Sandbox configuration.
        env: Additional environment variables.
        api_key: Anthropic API key.
        include_host_env: Inherit non-sensitive host env vars.
        skills: Skills to mount in the hermetic environment. Skills are
            copied to {ephemeral_home}/.claude/skills/ before spawning
            Claude Code.
    """

    network_policy: NetworkPolicy | None = None
    sandbox: SandboxConfig | None = None
    env: Mapping[str, str] | None = None
    api_key: str | None = None
    include_host_env: bool = False
    skills: SkillConfig | None = None
```

## Skill Discovery Behavior

Claude Code discovers skills in the following order (first match wins):

1. `.claude/skills/` in the current working directory
2. `~/.claude/skills/` in the user's home directory

When `IsolationConfig.skills` is set, WINK:

1. Creates `{ephemeral_home}/.claude/skills/` during `EphemeralHome` setup
2. Copies each enabled `SkillMount` into the skills directory
3. Claude Code discovers them via path #2 (redirected `HOME`)

```mermaid
sequenceDiagram
    participant W as WINK Adapter
    participant E as EphemeralHome
    participant S as Skills Directory
    participant C as Claude Code

    W->>E: Create ephemeral home
    E->>S: mkdir .claude/skills/
    loop Each SkillMount
        E->>E: Validate skill (if enabled)
        E->>S: Copy skill files
    end
    W->>C: Spawn with HOME=ephemeral
    C->>S: Discover skills natively
```

## Skill Validation

When `SkillConfig.validate_on_mount` is True (default), each skill mount is
validated before copying:

### Directory Skills

A skill directory must contain:

- `SKILL.md` file at the root (required)
- Optional subdirectories and supporting files

```
my-skill/
├── SKILL.md        # Required: skill definition
├── examples/       # Optional: example files
└── templates/      # Optional: templates
```

### File Skills

A single-file skill must:

- Have `.md` extension
- Contain valid markdown content
- File size must be ≤ 1 MiB

### Validation Errors

```python
class SkillValidationError(WinkError):
    """Raised when skill validation fails."""
    pass

class SkillNotFoundError(WinkError):
    """Raised when a skill source path does not exist."""
    pass

class SkillMountError(WinkError):
    """Raised when skill mounting fails."""
    pass
```

Validation errors include:

| Error | Condition |
| ----------------------- | ----------------------------------------------- |
| `SkillNotFoundError` | Source path does not exist |
| `SkillValidationError` | Directory missing SKILL.md |
| `SkillValidationError` | File not a markdown file |
| `SkillValidationError` | File exceeds size limit |
| `SkillMountError` | Duplicate skill names in config |
| `SkillMountError` | I/O error during copy |

## Copying Behavior

### Name Resolution

The skill name determines its destination directory:

```python
def resolve_skill_name(mount: SkillMount) -> str:
    """Resolve the effective skill name from a mount."""
    if mount.name is not None:
        return mount.name
    if mount.source.is_dir():
        return mount.source.name
    # File: strip .md extension
    return mount.source.stem
```

### Destination Layout

Skills are copied to `{ephemeral_home}/.claude/skills/{skill_name}/`:

| Source Type | Destination |
| ----------- | ------------------------------------------------- |
| Directory | `{skills_dir}/{name}/` (recursive copy) |
| File | `{skills_dir}/{name}/SKILL.md` (wrapped in dir) |

Example:

```python
# Source: ./skills/code-review/ (directory with SKILL.md)
# Destination: ~/.claude/skills/code-review/

# Source: ./my-skill.md (single file)
# Destination: ~/.claude/skills/my-skill/SKILL.md
```

### Copy Options

```python
def _copy_skill(
    source: Path,
    dest_dir: Path,
    *,
    follow_symlinks: bool = False,
    max_total_bytes: int = 10 * 1024 * 1024,  # 10 MiB per skill
) -> int:
    """Copy a skill to the destination directory.

    Returns:
        Total bytes copied.

    Raises:
        SkillMountError: If copy fails or exceeds byte limit.
    """
```

## Usage Examples

### Basic Skill Mounting

```python
from pathlib import Path
from weakincentives.adapters.claude_agent_sdk import (
    ClaudeAgentSDKAdapter,
    ClaudeAgentSDKClientConfig,
    IsolationConfig,
    NetworkPolicy,
    SkillConfig,
    SkillMount,
)

adapter = ClaudeAgentSDKAdapter(
    client_config=ClaudeAgentSDKClientConfig(
        permission_mode="bypassPermissions",
        isolation=IsolationConfig(
            network_policy=NetworkPolicy.no_network(),
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

### Custom Skill Names

```python
skills = SkillConfig(
    skills=(
        # Use custom name instead of directory name
        SkillMount(
            source=Path("./internal/review-v2"),
            name="code-review",
        ),
        # Single-file skill with explicit name
        SkillMount(
            source=Path("./prompts/test-helper.md"),
            name="testing",
        ),
    )
)
```

### Conditional Skills

```python
def get_skills(include_experimental: bool) -> SkillConfig:
    """Build skill config based on feature flags."""
    mounts = [
        SkillMount(Path("./skills/core")),
        SkillMount(
            source=Path("./skills/experimental"),
            enabled=include_experimental,
        ),
    ]
    return SkillConfig(skills=tuple(mounts))
```

### Disable Validation for Development

```python
# Skip validation during rapid iteration
skills = SkillConfig(
    skills=(SkillMount(Path("./wip-skill")),),
    validate_on_mount=False,
)
```

## EphemeralHome Integration

The `EphemeralHome` class handles skill mounting during setup:

```python
class EphemeralHome:
    def __init__(
        self,
        isolation: IsolationConfig,
        *,
        workspace_path: str | None = None,
        temp_dir_prefix: str = "claude-agent-",
    ) -> None:
        # ... existing initialization ...
        self._generate_settings()
        self._mount_skills()  # New step

    def _mount_skills(self) -> None:
        """Mount configured skills into the ephemeral home."""
        skills_config = self._isolation.skills
        if skills_config is None:
            return

        skills_dir = self._claude_dir / "skills"
        skills_dir.mkdir(parents=True, exist_ok=True)

        seen_names: set[str] = set()
        for mount in skills_config.skills:
            if not mount.enabled:
                continue

            name = resolve_skill_name(mount)
            if name in seen_names:
                raise SkillMountError(f"Duplicate skill name: {name}")
            seen_names.add(name)

            source = Path(mount.source).resolve()
            if not source.exists():
                raise SkillNotFoundError(f"Skill not found: {mount.source}")

            if skills_config.validate_on_mount:
                _validate_skill(source)

            dest = skills_dir / name
            _copy_skill(source, dest)

    @property
    def skills_dir(self) -> Path:
        """Path to the skills directory within ephemeral home."""
        return self._claude_dir / "skills"
```

## Testing Considerations

### Unit Tests

- Skill name resolution for directories and files
- Validation of directory skills (SKILL.md present/missing)
- Validation of file skills (extension, size)
- Duplicate name detection
- Disabled skill filtering
- Copy behavior for directories and files

### Integration Tests

- End-to-end skill discovery by Claude Code
- Skills from hermetic home override CWD skills
- Multiple skills compose correctly
- Skill with subdirectories copies recursively

### Test Fixtures

```python
@pytest.fixture
def skill_directory(tmp_path: Path) -> Path:
    """Create a valid skill directory."""
    skill_dir = tmp_path / "test-skill"
    skill_dir.mkdir()
    (skill_dir / "SKILL.md").write_text("# Test Skill\n\nDoes testing.")
    return skill_dir

@pytest.fixture
def skill_file(tmp_path: Path) -> Path:
    """Create a valid skill file."""
    skill_file = tmp_path / "test-skill.md"
    skill_file.write_text("# Test Skill\n\nDoes testing.")
    return skill_file
```

## Limitations

- **No runtime skill updates**: Skills are copied at `EphemeralHome` creation.
  Changes to source skills require recreating the adapter.
- **No skill dependencies**: Skills cannot declare dependencies on other skills.
  Compose manually via `SkillConfig`.
- **No skill templating**: Skills are copied verbatim. Use prompt composition
  for dynamic content.
- **Size limits enforced**: Individual skills capped at 10 MiB, total skill
  directory at 50 MiB to prevent workspace bloat.

## Security Considerations

- **Path traversal**: Skill names are sanitized to prevent directory traversal.
  Names containing `/`, `\`, or `..` raise `SkillMountError`.
- **Symlink following**: Disabled by default. Enable with caution as symlinks
  can escape intended boundaries.
- **Executable files**: Skill directories may contain scripts. The sandbox
  config controls whether these can execute.

## Future Extensions

- **Skill registries**: Load skills from remote registries (PyPI, npm, custom).
- **Skill versioning**: Pin skill versions for reproducible agent behavior.
- **Skill composition**: Define skill dependencies and load order.
- **Runtime skill injection**: Add skills to running sessions via events.

## Related Specifications

- [CLAUDE_AGENT_SDK.md](CLAUDE_AGENT_SDK.md): Parent adapter specification
- [WORKSPACE.md](WORKSPACE.md): Workspace and mount patterns
- [PROMPTS.md](PROMPTS.md): Prompt composition (alternative to skills)
