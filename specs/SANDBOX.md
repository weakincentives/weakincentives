# Sandbox Specification

## Purpose

One environment, one truth — and effects need a command channel. The
`Sandbox` aggregate names the environment an agent acts on: it vends the two
effect facets (`Filesystem`, `Shell`) rooted at one directory, owns the
egress/credential control plane, and has exactly one lifecycle owner —
whoever opens a sandbox closes it; facets never self-close. A
`WorkspaceConfig` declares intent as a serde value; a `SandboxProvider`
materializes it. Local first: remote sandboxes (M4) differ only in
transport.

**Implementation:** `src/weakincentives/sandbox/`

## Guiding Principles

- **One environment, one owner**: facets share a root and a lifecycle;
  `close()` is idempotent and is the only teardown
- **Exec is an argv surface**: `Shell.run` executes a vector — never
  `/bin/sh`; no globbing, quoting, or variable expansion
- **Intent is data**: `WorkspaceConfig` is a frozen serde value carrying
  credential *names* only — never secret material
- **Default-deny egress**: an empty `EgressPolicy` allows nothing; rules
  are explicit allow entries
- **Control plane, not tool surface**: `configure_egress` /
  `configure_credentials` are driven by the harness and policies, never
  exposed to the model via `ToolContext` — an agent cannot widen its own
  egress or read a bound secret
- **Fail-closed open**: any error while materializing (mounts, setup
  commands) removes the partially-built environment before propagating

## Architecture Overview

```
┌───────────────────────────────────────────────────────────────┐
│                     SandboxProvider.open(config)               │
│   WorkspaceConfig: mounts · allowed_host_roots · read_only ·     │
│                  egress · env · setup                          │
├───────────────────────────────────────────────────────────────┤
│                          Sandbox                               │
│  id · root · filesystem · shell · egress · credential_names    │
│  configure_egress · configure_credentials                      │
│  snapshot/restore (delegates to filesystem backend) · close    │
├──────────────────────────────┬────────────────────────────────┤
│   Filesystem(HostBackend)    │   LocalShell (argv subprocess)  │
│   (see FILESYSTEM.md)        │   output caps · default timeout │
└──────────────────────────────┴────────────────────────────────┘
```

`LocalSandbox` composes `Filesystem(HostBackend)` + `LocalShell` over one
temp directory; `sandbox.root == filesystem.root == shell` working
directory.

## Shell Facet

`Shell` at `src/weakincentives/sandbox/_shell.py`:

```python
def run(argv, *, cwd=None, env=None, stdin=None, timeout_s=None) -> CommandResult
```

| Contract | Behavior |
|----------|----------|
| Execution | `argv` vector executed directly; no shell interpretation |
| `cwd` | Relative to the sandbox root; absolute paths raise `ValueError`, escapes raise `PermissionError`, missing dirs raise `FileNotFoundError` |
| `env` | Layered over the shell's base environment for one invocation |
| `stdin` | Bytes piped to standard input; `None` closes it |
| `timeout_s` | `None` applies the default (60 s); non-positive raises `ValueError` |
| Output caps | stdout/stderr each capped (default 1 MiB); `truncated=True` when cut |
| Launch failures | Shell-conventional exit codes, never exceptions: 127 missing executable, 126 not executable, 124 timeout (`timed_out=True`, partial output preserved) |

`CommandResult` carries `exit_code`, `stdout`/`stderr` (bytes),
`truncated`, `duration_s`, `timed_out`, and an `ok` convenience property.

`LocalShell` env hygiene mirrors the snapshot plumbing's `git_env`: the
base environment is the host environment minus `GIT_*` variables, captured
at construction, with constructor `env` overrides layered on top.

## Sandbox Aggregate

`Sandbox` protocol at `src/weakincentives/sandbox/_sandbox.py`:

| Member | Contract |
|--------|----------|
| `id` | Stable instance identifier |
| `root` | Environment root; equals `filesystem.root` |
| `filesystem` / `shell` | Effect facets; raise `SandboxClosedError` after close |
| `egress` | `EgressPolicy` currently in effect; readable after close |
| `credential_names` | Names of bound credentials — never the material |
| `configure_egress(policy)` | Replaces the policy live (proxy reconfiguration when one exists) |
| `configure_credentials(bindings)` | Replaces the full binding set; duplicate names raise `ValueError` |
| `snapshot(tag=...)` / `restore(ref)` | Delegate to the filesystem backend (see FILESYSTEM.md) |
| `close()` | Idempotent teardown; `LocalSandbox` removes its root and snapshot storage |

After `close()`, facet access and control-plane calls raise
`SandboxClosedError`; identity (`id`, `root`) and policy introspection stay
readable.

## Intent and Provider

`WorkspaceConfig` at `src/weakincentives/sandbox/_config.py` — a frozen
serde value:

| Field | Meaning |
|-------|---------|
| `mounts` | `HostMount` entries copied in at open time (`_mounts.py`) |
| `allowed_host_roots` | Security boundary mount sources must live under |
| `read_only` | Filesystem facet rejects writes (the shell is OS-level and unaffected) |
| `egress` | `EgressPolicy` seeding the sandbox (default deny) |
| `env` | Variables layered over the shell's hygienic base environment |
| `setup` | Commands run in order after mounts; `shlex`-split into argv, no shell; non-zero exit fails the open with `SandboxSetupError` |

`SandboxProvider.open(config) -> Sandbox` is the factory seam.
`LocalSandboxProvider` materializes mounts with allowed-root validation,
symlink rejection and escape checks, byte budgets, and mount-target
confinement, then roots the facets at the new directory and hands
ownership to the returned sandbox.

## Egress & Credential Control Plane

`EgressPolicy` is default-deny. Each `EgressRule(host_glob, ports, protocol, credential)` is an allow entry; `host_glob` and `protocol` match
case-insensitively, empty `ports` means any port. `rule_for`/`allows`
answer queries; the first matching rule wins.

`CredentialInjection(credential, header_template)` names a credential and
how a proxy injects it (the `{secret}` placeholder is replaced at
injection time, inside the proxy only). Configs and serialized state carry
**names only**.

`CredentialBinding(name, secret)` supplies material at runtime through
`configure_credentials`. Invariants, enforced by tests:

- Secret material never appears in `WorkspaceConfig`, serialized state,
  `repr`, or logs (`CredentialBinding` excludes the secret from `repr`)
- Secret material is never written into the sandbox environment or
  filesystem
- Bindings are cleared on `close()`

**Local semantics:** there is no local proxy. `configure_egress` records
the policy the sandbox reports; `configure_credentials` holds bindings in
process memory — a documented no-op beyond bookkeeping. Enforcing,
live-reconfigurable proxies arrive with the remote sandbox (M4); the wire
ops are out of scope here.

## Lifecycle

```
provider.open(config)
  ├── materialize mounts → fresh temp dir   (fail: dir removed, error raised)
  ├── Filesystem(HostBackend(root, read_only))
  ├── LocalShell(root, env=config.env)
  ├── run config.setup in order             (fail: dir removed, SandboxSetupError)
  └── LocalSandbox(root, fs, shell, egress=config.egress)

sandbox.close()        # idempotent: rm root, rm snapshot storage, clear bindings
```

The sandbox is the execution context (`refactor/M3.md`): prompt templates
declare intent via `PromptTemplate.create(sandbox=...)`, adapters run the
harness with `cwd = sandbox.root`, `ToolContext.sandbox` exposes the
facets to handlers and policies, and tool transactions snapshot/restore
the (session, sandbox) pair atomically.

**Lease semantics.** A `Sandbox` handle is a *lease* on an environment:
`close()` releases the holder's claim — for locally provisioned sandboxes
that destroys the directory; future providers may pool environments or
attach to harness-provided ones, where release means detach. The lease is
held by an `AgentRuntime` (see `specs/ADAPTERS.md`): the bound
(adapter, prompt, sandbox) triple, paired once inside
`ProviderAdapter.runtime(prompt)` so mismatched pairings are
unrepresentable. `AgentLoop` holds one runtime per request, spanning
visibility-expansion retries and debug-bundle capture.

**Environment vs. workspace.** Today the sandbox conflates two roles that
local execution makes indistinguishable: the *workspace* (root, filesystem
facet, snapshots — cheap, per-run) and the *execution substrate* (the
place where the harness process runs — expensive, reusable, owner of
shell transport and egress). The harness currently runs on the host with
`cwd = sandbox.root`; when the environment becomes a container or SSH box
(`refactor/M4.md`), the substrate moves behind the provider and the lease
abstraction is the seam that keeps `evaluate` unchanged.

## Testing

Tests in `tests/sandbox/` mirror the package: `test_shell.py` (argv
semantics, cwd/env/stdin/timeout/caps, launch-failure exit codes),
`test_sandbox.py` (facets, snapshot/restore, control plane, close
idempotency, secret-material invariants), `test_config.py` (validation,
default-deny, serde round-trip), `test_provider.py` (mount parity, setup
commands, fail-closed open), `test_mounts.py` (the mount machinery).

## Related Specifications

- `specs/FILESYSTEM.md` - The filesystem facet and backend protocol
- `specs/WORKSPACE.md` - Workspace preview rendered from the opened sandbox
- `specs/MODULE_BOUNDARIES.md` - `sandbox` is a core-layer package
- `refactor/M2.md` - Milestone introducing this package
- `refactor/M3.md` - Sandbox as the execution context
- `refactor/M4.md` - Remote sandbox and the proxy sidecar (next)
