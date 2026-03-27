# Sandboxing Specification

> **Status:** Implemented (Claude Agent SDK, Codex App Server); Partial (Gemini ACP); N/A (OpenCode ACP)
> **Related packages:**
>
> - `src/weakincentives/adapters/claude_agent_sdk/isolation.py`
> - `src/weakincentives/adapters/codex_app_server/config.py`
> - `src/weakincentives/adapters/gemini_acp/config.py`
> - `src/weakincentives/adapters/acp/config.py`

## Purpose

This specification documents sandboxing options, default settings, and
enforcement mechanisms across all WINK adapters. Sandboxing restricts what
tools and agent subprocesses can do — limiting filesystem access, network
connectivity, and command execution to reduce blast radius during autonomous
agent execution.

Each adapter exposes sandboxing through different mechanisms depending on the
underlying runtime. WINK does not impose a single sandbox abstraction; instead,
each adapter maps to its runtime's native sandboxing capabilities.

## Adapter Summary

| Adapter | Sandbox Mechanism | Enforcement Layer | WINK Control |
|---------|-------------------|-------------------|--------------|
| Claude Agent SDK | bubblewrap (Linux) / seatbelt (macOS) | OS kernel | Full — `SandboxConfig` + `NetworkPolicy` |
| Codex App Server | Codex thread-level sandbox | Codex runtime | Partial — `SandboxMode` string |
| Gemini ACP | seatbelt (macOS) / Docker (Linux) | OS kernel | Config only — `--sandbox` CLI flag |
| OpenCode ACP | None (agent-controlled) | N/A | None |

## Claude Agent SDK

**Package:** `src/weakincentives/adapters/claude_agent_sdk/isolation.py`

The Claude Agent SDK adapter provides the most granular sandboxing control
through two complementary configuration objects: `SandboxConfig` for filesystem
and command restrictions, and `NetworkPolicy` for network access control.
Both are enforced at the OS level via bubblewrap (Linux) or seatbelt (macOS).

### SandboxConfig

Controls filesystem access, command execution, and sandbox behavior.

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `enabled` | `bool` | `True` | Enable OS-level sandboxing |
| `writable_paths` | `tuple[str, ...]` | `()` | Additional paths the agent can write to beyond the workspace. Relative paths resolved against workspace root |
| `readable_paths` | `tuple[str, ...]` | `()` | Additional paths the agent can read beyond the workspace |
| `excluded_commands` | `tuple[str, ...]` | `()` | Commands that bypass the sandbox (e.g., `"docker"`). Each exclusion is a potential security hole |
| `allow_unsandboxed_commands` | `bool` | `False` | Allow specific commands to run outside the sandbox. Requires `excluded_commands` to be set |
| `bash_auto_allow` | `bool` | `True` | Auto-approve Bash commands in sandbox mode. Only safe when `NetworkPolicy` blocks external access |
| `enable_weaker_nested_sandbox` | `bool` | `False` | Use a weaker sandbox inside unprivileged Docker containers where full bubblewrap is unavailable. Better than `enabled=False` but substantially weaker than the full sandbox. Linux only |
| `ignore_file_violations` | `tuple[str, ...]` | `()` | File paths for which sandbox violations are silently ignored |
| `ignore_network_violations` | `tuple[str, ...]` | `()` | Network hosts for which sandbox violations are silently ignored |

**Default behavior:** Sandbox is enabled. The agent can read and write within
the workspace directory. No additional paths are writable or readable. All
commands run inside the sandbox.

### NetworkPolicy

Controls which network resources tools can access. Enforced at the OS level.
The Claude API connection and the MCP bridge for WINK tools are **not**
affected by this policy.

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `allowed_domains` | `tuple[str, ...]` | `()` | Domains tools can access. Empty = no network. `("*",)` = unrestricted (not recommended) |
| `allow_unix_sockets` | `tuple[str, ...]` | `()` | Unix socket paths accessible within the sandbox (macOS only; Linux uses seccomp) |
| `allow_all_unix_sockets` | `bool` | `False` | Allow access to all Unix sockets |
| `allow_local_binding` | `bool` | `False` | Allow binding to localhost ports (macOS only) |
| `http_proxy_port` | `int \| None` | `None` | HTTP proxy port for custom proxy |
| `socks_proxy_port` | `int \| None` | `None` | SOCKS5 proxy port for custom proxy |

**Factory methods:**

| Factory | Description |
|---------|-------------|
| `NetworkPolicy.no_network()` | Block all tool network access |
| `NetworkPolicy.with_domains(*domains)` | Allow specific domains only |

**Default behavior:** No network access for tools (`allowed_domains=()`).

### IsolationConfig

`IsolationConfig` wraps `SandboxConfig` and `NetworkPolicy` together with
authentication configuration. When provided to the adapter, it creates an
ephemeral HOME directory with generated settings, preventing interaction
with the host's `~/.claude` configuration.

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `network_policy` | `NetworkPolicy \| None` | `None` | Tool network restrictions |
| `sandbox` | `SandboxConfig \| None` | `None` | Sandbox configuration |
| `env` | `Mapping[str, str] \| None` | `None` | Additional environment variables |
| `api_key` | `str \| None` | `None` | Explicit API key (disables Bedrock) |
| `aws_config_path` | `Path \| str \| None` | `None` | AWS config path for Docker |
| `include_host_env` | `bool` | `False` | Copy non-sensitive host env vars |

**Factory methods** (recommended for explicit intent and fail-fast validation):

| Factory | Description |
|---------|-------------|
| `IsolationConfig.inherit_host_auth()` | Inherit auth from host, fail if none configured |
| `IsolationConfig.with_api_key(key)` | Use explicit Anthropic API key |
| `IsolationConfig.for_anthropic_api()` | Require `ANTHROPIC_API_KEY` from env |
| `IsolationConfig.for_bedrock()` | Require AWS Bedrock, fail if not configured |

**Default behavior:** When `isolation` is `None` on the client config, the
adapter creates a default `IsolationConfig()` automatically — it does **not**
run against the host environment. The default `IsolationConfig()` creates an
ephemeral HOME, enables the sandbox (`SandboxConfig(enabled=True)`), and
blocks all tool network access (`NetworkPolicy.no_network()`). To opt out of
isolation entirely, callers must bypass `_setup_ephemeral_home` at the adapter
level; there is no `IsolationConfig` setting that disables isolation.

### Isolation Lifecycle

1. Factory method called (e.g., `IsolationConfig.inherit_host_auth()`)
1. Authentication validated — fail fast if not configured
1. Ephemeral HOME created — temporary directory with `$HOME/.claude/settings.json`
1. Minimal settings generated to avoid host interference
1. SDK invoked with `env` (HOME override) and `setting_sources=["user"]`
1. Ephemeral HOME deleted after evaluation (even on error)

### Platform Support

| Platform | Sandbox Method | Network Enforcement |
|----------|---------------|---------------------|
| Linux | bubblewrap (`bwrap`) | seccomp filters |
| macOS | seatbelt (`sandbox-exec`) | seatbelt profiles |
| Windows | Not enforced | Not enforced (HOME redirection only) |

### Usage Examples

```python
from weakincentives.adapters.claude_agent_sdk import (
    ClaudeAgentSDKAdapter,
    ClaudeAgentSDKClientConfig,
    IsolationConfig,
    NetworkPolicy,
    SandboxConfig,
)

# Fully locked down: no network, sandbox enabled
adapter = ClaudeAgentSDKAdapter(
    client_config=ClaudeAgentSDKClientConfig(
        isolation=IsolationConfig(
            network_policy=NetworkPolicy.no_network(),
            sandbox=SandboxConfig(enabled=True),
        ),
    ),
)

# Domain allowlist
adapter = ClaudeAgentSDKAdapter(
    client_config=ClaudeAgentSDKClientConfig(
        isolation=IsolationConfig(
            network_policy=NetworkPolicy.with_domains("docs.python.org", "pypi.org"),
            sandbox=SandboxConfig(enabled=True),
        ),
    ),
)

# Docker: weaker nested sandbox + extra writable paths
adapter = ClaudeAgentSDKAdapter(
    client_config=ClaudeAgentSDKClientConfig(
        isolation=IsolationConfig(
            sandbox=SandboxConfig(
                enabled=True,
                enable_weaker_nested_sandbox=True,
                writable_paths=("/tmp/build-output",),
            ),
        ),
    ),
)
```

## Codex App Server

**Package:** `src/weakincentives/adapters/codex_app_server/config.py`

The Codex App Server adapter exposes sandbox control via a `SandboxMode`
string on `thread/start`. Codex enforces the sandbox internally at the
runtime level.

### SandboxMode

```python
SandboxMode = Literal["read-only", "workspace-write", "danger-full-access"]
```

| Mode | Filesystem Writes | Description |
|------|-------------------|-------------|
| `"read-only"` | Blocked outside workspace | No file writes outside the workspace directory |
| `"workspace-write"` | Allowed under `cwd` | Write allowed within the working directory |
| `"danger-full-access"` | Unrestricted | Full filesystem access — no restrictions |

**Default behavior:** `sandbox_mode=None` — no sandbox mode is sent on
`thread/start`, deferring to Codex's own default behavior.

### ApprovalPolicy

Approval policy complements sandboxing by controlling command and file change
approvals:

```python
ApprovalPolicy = Literal["never", "untrusted", "on-failure", "on-request"]
```

| Policy | Behavior |
|--------|----------|
| `"never"` | Auto-accept all approvals (default for non-interactive WINK) |
| `"untrusted"` | Require approval for non-trusted commands |
| `"on-failure"` | Require approval after command failure |
| `"on-request"` | Require approval on every action |

**Default behavior:** `approval_policy="never"` — auto-accept since WINK
executes without a human in the loop.

### Configuration

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `sandbox_mode` | `SandboxMode \| None` | `None` | Sandbox mode for `thread/start` |
| `approval_policy` | `ApprovalPolicy` | `"never"` | How to handle command/file approvals |

### Comparison with Claude Agent SDK

| Aspect | Claude Agent SDK | Codex App Server |
|--------|-----------------|------------------|
| Granularity | Per-path (`writable_paths`, `readable_paths`) | Three preset modes |
| Network control | Per-domain allowlist | Not available |
| Enforcement | OS kernel (bubblewrap/seatbelt) | Codex runtime |
| Command exclusions | `excluded_commands` tuple | Not available |
| Nested container support | `enable_weaker_nested_sandbox` | Not available |

### Usage Example

```python
from weakincentives.adapters.codex_app_server import (
    CodexAppServerAdapter,
    CodexAppServerClientConfig,
)

adapter = CodexAppServerAdapter(
    client_config=CodexAppServerClientConfig(
        cwd="/absolute/path/to/workspace",
        sandbox_mode="workspace-write",
        approval_policy="never",
    ),
)
```

## Gemini ACP

**Package:** `src/weakincentives/adapters/gemini_acp/config.py`

Gemini CLI supports OS-level sandboxing via the `--sandbox` CLI flag. On
macOS, this uses seatbelt profiles; on Linux, it uses Docker/Podman containers.

### Known Limitation: ACP Incompatibility

> **`--sandbox` and `--experimental-acp` are incompatible** in Gemini v0.29.5.
> The sandbox re-launches Gemini via `sandbox-exec` (macOS) or a container
> (Linux), which breaks ACP's stdio pipe protocol. All sandbox profiles cause
> the ACP handshake to time out. The config fields are retained for
> documentation and forward-compatibility but **cannot be used with the ACP
> adapter** until the upstream issue is resolved.

### Configuration

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `sandbox` | `bool` | `False` | Enable OS-level sandboxing (`--sandbox` flag) |
| `sandbox_profile` | `str \| None` | `None` | Seatbelt profile name (macOS only), set via `SEATBELT_PROFILE` env var |
| `approval_mode` | `str \| None` | `None` | CLI `--approval-mode` flag: `"default"`, `"auto_edit"`, `"yolo"`, `"plan"` |

**Default behavior:** Sandbox is disabled (`sandbox=False`). The
`approval_mode` flag controls agent-level restrictions as an alternative to
OS-level sandboxing.

### Seatbelt Profiles (macOS)

When `--sandbox` is active on macOS, Gemini uses seatbelt profiles to restrict
file system and network access. Six built-in profiles ship with Gemini CLI:

| Profile | Default Policy | File Reads | File Writes | Network Out |
|---------|---------------|------------|-------------|-------------|
| `permissive-open` | `(allow default)` | Unrestricted | Workspace + tmp + cache | Unrestricted |
| `permissive-proxied` | `(allow default)` | Unrestricted | Workspace + tmp + cache | Proxy only (localhost:8877) |
| `permissive-closed` | `(allow default)` | Unrestricted | Workspace + tmp + cache | Blocked |
| `restrictive-open` | `(deny default)` | `(allow file-read*)` | Workspace + tmp + cache | Unrestricted |
| `restrictive-proxied` | `(deny default)` | `(allow file-read*)` | Workspace + tmp + cache | Proxy only |
| `restrictive-closed` | `(deny default)` | `(allow file-read*)` | Workspace + tmp + cache | Blocked |

The default profile is `permissive-open`.

**Permissive** profiles start with `(allow default)` and selectively deny
writes. All system calls are allowed.

**Restrictive** profiles start with `(deny default)` and explicitly allowlist
file reads, process execution, and specific system calls.

All profiles restrict writes to: `TARGET_DIR` (subprocess `cwd`), `TMP_DIR`,
`CACHE_DIR`, `~/.gemini`, `~/.npm`, `~/.cache`, `~/.gitconfig`, plus up to 5
additional directories via `--include-directories`.

> **Note:** The `-closed` and `-proxied` profiles block or restrict outbound
> network access, which also blocks Gemini API access, making them unsuitable
> for any Gemini usage without a local proxy.

### Approval Modes

Since OS-level sandboxing is currently incompatible with ACP mode, the
`--approval-mode` flag is the primary mechanism for restricting agent behavior:

| Mode | Description |
|------|-------------|
| `default` | Prompt for approval on actions |
| `auto_edit` | Auto-approve edit tools only |
| `yolo` | Auto-approve all tools |
| `plan` | Read-only mode (no edits) |

For non-interactive WINK execution, `yolo` is the recommended default.

### Platform Support

| Platform | Sandbox Method | Notes |
|----------|---------------|-------|
| macOS | seatbelt (`sandbox-exec`) | Six built-in profiles |
| Linux | Docker/Podman container | Container isolation |

## OpenCode ACP

**Package:** `src/weakincentives/adapters/opencode_acp/config.py`

OpenCode does not expose sandboxing configuration at the ACP protocol level.
All sandboxing decisions are made internally by the OpenCode agent binary.
WINK has no control over OpenCode's sandbox behavior.

### Configuration

The only sandbox-adjacent controls are permission and capability flags:

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `permission_mode` | `Literal["auto", "deny", "prompt"]` | `"auto"` | Response to permission requests |
| `allow_file_reads` | `bool` | `False` | Advertise `readTextFile` capability |
| `allow_file_writes` | `bool` | `False` | Advertise `writeTextFile` capability |

**Default behavior:** Auto-approve all permission requests. File read/write
capabilities are not advertised (OpenCode uses its own file access mechanisms).

### Comparison with Other Adapters

| Aspect | OpenCode ACP |
|--------|-------------|
| OS-level sandbox | Not available |
| Network control | Not available |
| Filesystem restrictions | Capability advertisement only (`allow_file_reads`, `allow_file_writes`) |
| Agent-level restrictions | `permission_mode` controls request handling |

## Generic ACP Base

**Package:** `src/weakincentives/adapters/acp/config.py`

The generic `ACPAdapter` base class provides no sandbox-specific configuration.
Sandboxing is delegated entirely to the agent binary (OpenCode, Gemini, etc.)
or exposed through subclass-specific config fields.

The generic config controls only capability advertisement and permission
handling:

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `permission_mode` | `Literal["auto", "deny", "prompt"]` | `"auto"` | Response to permission requests |
| `allow_file_reads` | `bool` | `False` | Advertise `readTextFile` capability |
| `allow_file_writes` | `bool` | `False` | Advertise `writeTextFile` capability |

> **Note:** `permission_mode="prompt"` is treated as `"deny"` since the
> adapter cannot block for interactive prompting.

## Default Settings Comparison

| Setting | Claude Agent SDK | Codex App Server | Gemini ACP | OpenCode ACP |
|---------|-----------------|------------------|------------|--------------|
| Sandbox enabled | `True` | `None` (not sent) | `False` | N/A |
| Network access | Blocked (`()`) | Not controlled | Not controlled | Not controlled |
| Filesystem writes | Workspace only | Not controlled | Not controlled | Not controlled |
| Command approval | N/A (sandbox auto-allows) | `"never"` (auto-accept) | `None` (agent default) | `"auto"` (auto-approve) |
| Nested container support | `False` (opt-in) | N/A | N/A | N/A |

## Security Considerations

### Defense in Depth

The adapters offer different layers of protection:

1. **OS-level sandboxing** (Claude Agent SDK, Gemini): Kernel-enforced
   restrictions that the agent process cannot bypass. Strongest guarantee.

1. **Runtime-level sandboxing** (Codex App Server): Enforced by the agent
   runtime. The agent process itself enforces restrictions — weaker than
   OS-level but still effective for well-behaved agents.

1. **Capability advertisement** (ACP adapters): Tells the agent what it can
   and cannot do. The agent is trusted to respect these boundaries — no
   enforcement mechanism.

1. **Permission/approval gating** (all adapters): Controls whether specific
   actions require approval. In non-interactive WINK execution, approvals are
   typically auto-accepted.

### Recommendations

- **Untrusted code review:** Use Claude Agent SDK with
  `NetworkPolicy.no_network()` and `SandboxConfig(enabled=True)`. This
  provides kernel-enforced isolation from both network and filesystem.

- **Trusted workspace editing:** Use Codex App Server with
  `sandbox_mode="workspace-write"` or Claude Agent SDK with
  `SandboxConfig(writable_paths=(...))`.

- **Docker/container environments:** Use Claude Agent SDK with
  `enable_weaker_nested_sandbox=True`. The container itself provides the
  outer isolation boundary; the weaker sandbox adds defense in depth.

- **Gemini in ACP mode:** Rely on `--approval-mode` for agent-level
  restrictions until the sandbox + ACP incompatibility is resolved upstream.

## Related Specifications

- `specs/CLAUDE_AGENT_SDK.md` — Claude Agent SDK adapter (full isolation docs)
- `specs/CODEX_APP_SERVER.md` — Codex App Server adapter (sandbox modes)
- `specs/GEMINI_ACP_ADAPTER.md` — Gemini ACP adapter (seatbelt profiles)
- `specs/ACP_ADAPTER.md` — Generic ACP adapter (base class)
- `specs/OPENCODE_ADAPTER.md` — OpenCode ACP adapter
- `specs/ADAPTERS.md` — Provider adapter protocol
- `specs/TOOLS.md` — Tool registration and policies
- `specs/GUARDRAILS.md` — Tool policies and feedback providers
