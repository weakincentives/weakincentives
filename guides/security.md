# Security Best Practices

Agent systems have unique security challenges. They handle untrusted input from
two sources: users and models. They perform privileged operations: file access,
command execution, API calls. And they do this autonomously, without human
review of each action.

This guide covers security practices specific to building agents with WINK. For
general Python security, see OWASP and your organization's policies.

## Threat Model

Before diving into mitigations, understand what you're defending against:

**Untrusted user input:**

- Users can provide malicious prompts designed to manipulate the model
- Input may contain injection attacks targeting tool handlers
- File paths, URLs, and identifiers may be crafted to escape sandboxes

**Untrusted model output:**

- Models can hallucinate malicious tool calls
- Prompt injection in retrieved content can hijack model behavior
- Output may contain unexpected formats that break assumptions

**Supply chain risks:**

- Dependencies may contain vulnerabilities or malicious code
- Model providers may have compromised infrastructure
- Serialized data may contain malicious payloads

## Tool Handler Security

Tool handlers are the primary attack surface. They bridge model output (which
you don't control) to system operations (which have real consequences).

### Never Trust Model Output

The model decides what parameters to pass to your tools. Treat these parameters
as untrusted input:

```python nocheck
# BAD: Shell injection vulnerability
def run_command(params: RunParams, *, context: ToolContext) -> ToolResult[RunResult]:
    result = subprocess.run(f"grep {params.pattern} {params.file}", shell=True)
    return ToolResult.ok(RunResult(output=result.stdout))

# GOOD: Use argument lists, never shell=True with untrusted input
def run_command(params: RunParams, *, context: ToolContext) -> ToolResult[RunResult]:
    result = subprocess.run(
        ["grep", "--", params.pattern, params.file],
        capture_output=True,
        text=True,
    )
    return ToolResult.ok(RunResult(output=result.stdout))
```

### Validate File Paths

Path traversal attacks (`../../../etc/passwd`) are common. Always validate
paths against an allowed root:

```python nocheck
from pathlib import Path

def validate_path(user_path: str, allowed_root: Path) -> Path:
    """Resolve path and verify it's within allowed_root."""
    resolved = (allowed_root / user_path).resolve()
    if not resolved.is_relative_to(allowed_root.resolve()):
        raise ValueError(f"Path escapes allowed root: {user_path}")
    return resolved
```

WINK's `Filesystem` implementations do this automatically. Use them instead of
raw `pathlib` when possible.

### Avoid Dangerous Operations

Some operations should never be performed on untrusted input:

```python nocheck
# NEVER do these with model-provided input:
eval(model_output)           # Arbitrary code execution
exec(model_output)           # Arbitrary code execution
pickle.loads(model_output)   # Arbitrary code execution via pickle
yaml.load(model_output)      # Use yaml.safe_load instead
importlib.import_module(x)   # Importing arbitrary modules
getattr(obj, model_output)   # Accessing arbitrary attributes
```

If you need expression evaluation, use `AstevalSection` which restricts what
code can do.

### Return Errors, Don't Raise

Tool handlers should return `ToolResult.error()` for expected failures. This
gives the model useful feedback. Reserve exceptions for unexpected failures:

```python nocheck
def read_file(params: ReadParams, *, context: ToolContext) -> ToolResult[ReadResult]:
    try:
        content = context.filesystem.read(params.path)
        return ToolResult.ok(ReadResult(content=content))
    except FileNotFoundError:
        # Expected failure: return error
        return ToolResult.error(f"File not found: {params.path}")
    except PermissionError:
        # Expected failure: return error
        return ToolResult.error(f"Permission denied: {params.path}")
    # Unexpected failures (IOError, etc.) propagate as exceptions
```

## Sandboxing

WINK provides two sandboxing mechanisms. Use them.

### VFS (Virtual Filesystem)

VFS creates an in-memory copy of files. Writes modify the copy, not the host:

```python nocheck
from weakincentives.contrib.tools import VfsToolsSection, VfsConfig, HostMount

vfs = VfsToolsSection(
    session=session,
    config=VfsConfig(
        mounts=(
            HostMount(
                host_path="src",
                include_glob=("*.py",),
                exclude_glob=("__pycache__/*", "*.pyc"),
            ),
        ),
        # Restrict what host paths can be mounted
        allowed_host_roots=("src", "tests"),
    ),
)
```

**Security properties:**

- Writes never affect the host filesystem
- Only explicitly mounted paths are accessible
- `include_glob` and `exclude_glob` filter what files are copied

**Limitations:**

- Large files consume memory
- No process isolation (can't run untrusted executables)

### Podman (Container Isolation)

For stronger isolation, use Podman containers:

```python nocheck
from weakincentives.contrib.tools import PodmanSandboxSection, PodmanSandboxConfig

sandbox = PodmanSandboxSection(
    session=session,
    config=PodmanSandboxConfig(
        image="python:3.12-slim",
        work_dir="/workspace",
        # Resource limits
        memory_limit="512m",
        cpu_limit=1.0,
        # Network isolation
        network="none",
    ),
)
```

**Security properties:**

- Process isolation via Linux namespaces
- Resource limits prevent DoS
- Network isolation prevents data exfiltration
- Filesystem changes are contained

**When to use Podman:**

- Running tests or linters (untrusted code execution)
- Processing untrusted file formats
- Any operation where VFS isolation isn't sufficient

## Input Validation with Contracts

Use design-by-contract decorators to validate inputs at API boundaries:

```python nocheck
from weakincentives.dbc import require

@require(lambda path: not path.startswith("/"), "Path must be relative")
@require(lambda path: ".." not in path, "Path must not contain ..")
def process_file(path: str) -> None:
    ...
```

Contracts fail fast with clear messages. This is better than silently accepting
bad input and failing later with confusing errors.

## Secret Management

Never hardcode secrets. Never log secrets. Never include secrets in prompts.

```python nocheck
# BAD: Secret in code
api_key = "sk-abc123..."

# GOOD: Secret from environment
import os
api_key = os.environ.get("API_KEY")
if not api_key:
    raise RuntimeError("API_KEY environment variable required")

# GOOD: Secret from secret manager
from your_secret_manager import get_secret
api_key = get_secret("api-key")
```

**For tool handlers:**

- Inject secrets via resources, not tool parameters
- Never let the model see or request secrets
- Audit what gets logged (secrets in tool results leak to logs)

```python nocheck
# Inject API client with embedded credentials via resources
http_client = AuthenticatedClient(api_key=os.environ["API_KEY"])
prompt = Prompt(template).bind(params, resources={HTTPClient: http_client})
```

## Dependency Security

WINK includes security scanning in `make check`:

```bash
make bandit      # Static analysis for Python security issues
make pip-audit   # Check dependencies for known vulnerabilities
make deptry      # Find unused/missing dependencies
```

**Best practices:**

- Pin dependencies to specific versions in production
- Regularly update dependencies and re-run security scans
- Review new dependencies before adding them
- Use `pip-audit` in CI to catch vulnerable dependencies

## Serialization Security

WINK uses its own `serde` module for serialization. This is safer than pickle
but still requires care:

```python nocheck
from weakincentives.serde import parse, dump

# Safe: parse validates against the expected type
data = parse(MyDataclass, untrusted_json)

# Safe: dump produces JSON, not executable code
json_str = dump(my_object)
```

**Never use:**

- `pickle` with untrusted data (arbitrary code execution)
- `yaml.load` with untrusted data (use `yaml.safe_load`)
- Custom deserializers that call `eval` or `exec`

## Session State Security

Session state persists across tool calls. Be careful what you store:

- Don't store secrets in session state
- Don't store raw user input without validation
- Consider what happens if session state is logged or debugged

```python nocheck
# BAD: Storing sensitive data in session
session[UserCredentials].append(Credentials(password=user_password))

# GOOD: Store references, not secrets
session[UserContext].append(UserContext(user_id=user.id))
```

## Prompt Injection Defense

Prompt injection occurs when untrusted content manipulates model behavior. This
is hard to prevent completely, but you can reduce risk:

**Separate instructions from data:**

```python nocheck
# BAD: User input mixed with instructions
prompt = f"Summarize this: {user_input}"

# BETTER: Clear separation
prompt = """
<instructions>
Summarize the content in the <user_content> section.
</instructions>

<user_content>
{user_input}
</user_content>
"""
```

**Use tool policies to enforce invariants:**

```python nocheck
from weakincentives.prompt import SequentialDependencyPolicy

# Require review before deploy, regardless of what the model says
policy = SequentialDependencyPolicy(
    dependencies={"deploy": frozenset({"review", "test"})}
)
```

**Limit tool capabilities:**

- Don't give agents tools they don't need
- Use read-only tools when writes aren't required
- Scope filesystem access to specific directories

## Logging and Observability

Logs are essential for security forensics, but they're also a leak vector.

**Do log:**

- Tool invocations (name, sanitized params)
- Authorization decisions
- Errors and exceptions

**Don't log:**

- Secrets, tokens, passwords
- Full user input (may contain PII)
- Full model output (may contain sensitive retrieved content)

```python nocheck
import logging

logger = logging.getLogger(__name__)

def my_handler(params: MyParams, *, context: ToolContext) -> ToolResult[MyResult]:
    # Log action, not sensitive details
    logger.info("Processing request", extra={"action": params.action})
    ...
```

## Security Checklist

Before deploying an agent:

- [ ] All tool handlers validate input (paths, formats, ranges)
- [ ] No shell=True with untrusted input
- [ ] No eval/exec/pickle on untrusted data
- [ ] Secrets injected via environment/resources, not hardcoded
- [ ] Filesystem access sandboxed (VFS or Podman)
- [ ] Network access restricted where possible
- [ ] Dependencies scanned for vulnerabilities
- [ ] Logging doesn't leak secrets or PII
- [ ] Tool policies enforce critical invariants
- [ ] `make check` passes (includes security scans)

## Next Steps

- [Code Quality](code-quality.md): Security scanning tools
- [Workspace Tools](workspace-tools.md): VFS and Podman sandboxing
- [Tools](tools.md): Tool handler patterns and policies
