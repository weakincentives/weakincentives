# Podman Builtin Section Specification

## Introduction

This document describes the builtin section that provisions a Podman-backed shell
environment for large language model (LLM) sessions. The section exposes a
cohesive set of tools that adapters can present as a single "generic shell"
affordance while the runtime maintains strong lifecycle guarantees. The feature
relies on the [`podman`](https://github.com/containers/podman-py) Python
library and ships behind an optional extra so core installations remain light.

## Goals

- **First-class container shell** – provision an isolated Podman container that
  LLMs can target through a stable shell command interface.
- **Adapter-friendly abstraction** – expose a single logical shell tool to
  adapters even though multiple lower-level operations exist under the hood.
- **Deterministic lifecycle** – ensure containers start lazily, stream command
  output predictably, and shut down promptly after use.
- **Composable prompts** – model the capability as a builtin `Section` so
  prompts opt in declaratively and inherit tool registration behaviour.
- **Optional dependency** – depend on the `podman` Python client only when the
  `podman` extra is requested.

## Non-Goals

- Replace existing Docker or local-shell tooling.
- Manage long-running containers beyond the lifespan of a prompt session.
- Provide arbitrary Podman management (e.g., image building, volume snapshots).
- Implement adapter UI elements; adapters only receive structured tool
  metadata and invocation results.

## Extra Definition

Add a new optional dependency set to `pyproject.toml`:

```toml
[project.optional-dependencies]
podman = ["podman>=5.0.0"]
```

Runtime guards should raise a descriptive `MissingDependencyError` when the
extra is absent and the section is enabled. `make test` already installs all
extras, so coverage remains intact.

## Section Overview

Introduce `PodmanShellSection`, a builtin section located in
`src/weakincentives/prompt/builtins/podman.py`. The section subclasses the base
`Section` type and registers a trio of tools that collectively model a
container-backed shell session. The section accepts configuration via keyword
arguments:

- `image: str` (default: `"docker.io/library/python:3.12"`) – container image
  reference.
- `name_prefix: str` (default: `"wai-shell"`) – prefix for generated container
  names.
- `workdir: str | None` (default: `None`) – working directory for command
  execution.
- `environment: Mapping[str, str] | None` – container environment overrides.
- `timeout: datetime.timedelta | None` – optional hard limit for container
  uptime; exceeding the limit triggers automatic cleanup.

The section renders a short markdown snippet describing the capability and
registers tools during initialization. Prompt authors can enable the section by
including it in the `Prompt` tree without needing additional boilerplate.

## Tooling Contract

Adapters receive a single high-level tool named `podman_shell` that delegates to
three internal operations:

1. **`podman_shell.start`** – ensure a container exists and is running.
2. **`podman_shell.exec`** – execute a shell command inside the container.
3. **`podman_shell.stop`** – tear down the container and release resources.

The section exports only the `podman_shell` tool; the runtime wires `params` and
`result` unions so adapters invoke it with a `"action"` discriminator. This
keeps the adapter surface area minimal while preserving fine-grained lifecycle
control.

### Parameter Dataclasses

```python
@dataclass
class PodmanShellParams:
    action: Literal["start", "exec", "stop"]
    command: str | None = None
    timeout_seconds: int | None = None
```

- `action` controls which operation is executed.
- `command` is required when `action == "exec"`; it contains the raw shell
  command to run.
- `timeout_seconds` optionally overrides the execution timeout for the request
  (defaults to a per-section constant for `exec`).

Validation occurs inside the handler using existing prompt validation helpers.

### Result Dataclasses

```python
@dataclass
class PodmanShellResult:
    action: Literal["start", "exec", "stop"]
    container_name: str
    stdout: str | None
    stderr: str | None
    exit_code: int | None
```

- `stdout`, `stderr`, and `exit_code` are populated for `exec` actions.
- `stdout` and `stderr` remain `None` for lifecycle operations.
- Errors (e.g., missing container, non-zero exit codes) set `success=False` in
  the `ToolResult` and place a human-friendly message in `stderr`.

### Tool Handler Flow

`PodmanShellSection` owns a private `PodmanShellController` helper that wraps the
`podman` client, tracks the container ID, and serializes access. The handler
pipeline:

1. **Dependency Loading** – import `podman` lazily inside the section factory to
   avoid import errors when the extra is missing.
2. **Client Bootstrap** – instantiate `podman.PodmanClient` using configuration
   (socket path or service URI resolved from environment variables such as
   `PODMAN_URL`).
3. **Start Action** –
   - Pull the configured image if absent.
   - Create and start a container with deterministic naming and resource limits.
   - Stream minimal logs back to the LLM through the tool response.
4. **Exec Action** –
   - Ensure the container is running; start it if necessary.
   - Use `container.exec()` with `tty=False` and `stream=True`.
   - Accumulate output (respecting `timeout_seconds`) and return aggregated
     stdout/stderr with the exit code.
5. **Stop Action** – stop and remove the container, ignoring "not found" errors.

Handlers capture exceptions, log via the runtime logger, and convert failures to
`ToolResult(success=False, ...)` payloads.

## Adapter Integration

Adapters treat `podman_shell` as a generic shell interface. The prompt runtime
registers the tool with a concise description, e.g. "Execute shell commands in
an isolated Podman container". Adapters map provider-specific function-call
schemas to the dataclasses described above. No adapter changes are required
beyond wiring the tool into existing generic-shell UX flows.

The reducer pipeline should mark the tool as mutually exclusive with other
container-backed shells (e.g., Docker) to avoid double provisioning. A future
capability negotiation step can surface the active provider to the LLM, but the
section itself simply registers tooling.

## Lifecycle and Cleanup

- Containers receive a deterministic name derived from `name_prefix` and the
  prompt session ID to simplify cleanup.
- A background `atexit` or prompt teardown hook ensures `stop` is called even if
  the LLM never does so explicitly.
- The controller tracks container state in memory; concurrent tool calls acquire
  an `asyncio.Lock` or threading lock (matching the runtime's execution model)
  to guard against overlapping exec requests.
- Timeouts trigger forced removal via `container.remove(force=True)`.

## Security Considerations

- The default image should be minimal and non-privileged; avoid mounting host
  directories unless explicitly configured.
- Environment variables exposed to the container are whitelisted via the
  `environment` parameter—no implicit host environment leakage.
- Disable Podman features that require elevated privileges (e.g., `--privileged`
  mode) unless a prompt explicitly enables them.
- Cap resource usage through CPU and memory limits supplied at container
  creation; defaults mirror existing Docker-based shell tooling.
- All command output is truncated to a configurable byte budget before returning
  to the LLM to prevent context overflow.

## Testing Strategy

- **Unit tests** – mock the `podman` client to assert lifecycle behaviour,
  timeout handling, and error translation. Place these under
  `tests/prompt/builtins/test_podman_shell.py`.
- **Integration tests** – behind an opt-in marker (e.g., `@pytest.mark.podman`),
  spin up a real container when the host has Podman available.
- **Documentation checks** – ensure the section appears in `PROMPTS.md` and the
  tooling overview once implemented.

## Rollout Plan

1. Land the section, tests, and documentation guarded by the new extra.
2. Update adapters to recognize `podman_shell` in their generic-shell registries.
3. Expose configuration knobs (image, environment) through user-facing configs.
4. Monitor integration tests in CI and gather feedback on container startup
   latencies before enabling by default.

## Open Questions

- Should the section support volume mounts for persistence between tool calls?
- Do we need metrics on container lifecycle events to aid debugging?
- How should the runtime reconcile simultaneous requests for Docker and Podman
  shells within the same prompt?

