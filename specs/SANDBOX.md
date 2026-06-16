# Sandbox Specification

## Purpose

One environment, one truth — and effects need a command channel. The
`Sandbox` aggregate names the environment an agent acts on: it vends the two
effect facets (`Filesystem`, `Shell`) rooted at one directory, owns the
egress/credential control plane, and has exactly one lifecycle owner —
whoever opens a sandbox closes it; facets never self-close. A
`WorkspaceConfig` declares intent as a serde value; a `SandboxProvider`
materializes it. Local and remote differ only in transport: a
`SandboxTransport` carries the same primitives over one connection, and
the remote facets are thin clients of it (M4).

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
how the egress sidecar injects it (the `{secret}` placeholder is replaced
at injection time, inside the sidecar only). Configs and serialized state
carry **names only**.

`CredentialBinding(name, secret)` supplies material at runtime through
`configure_credentials`. Invariants, enforced by tests:

- Secret material never appears in `WorkspaceConfig`, serialized state,
  `repr`, or logs (`CredentialBinding` excludes the secret from `repr`)
- Secret material is never written into the sandbox environment or
  filesystem
- Bindings are cleared on `close()`

**Where enforcement lives.** The egress chokepoint is a **sidecar process
the environment owns** — WINK configures it over the control plane but
never runs it. A local environment has no sidecar, so `configure_egress`
records the policy the sandbox reports and `configure_credentials` holds
bindings in process memory (bookkeeping only). A remote sandbox carries the
same calls over its transport to the sidecar; `RemoteSandbox` retains
credential *names* only, and secret material lives only in the sidecar.

## Remote Sandbox

Local and remote differ only in transport (`refactor/M4.md`). This
milestone lands the *interface* and its *local-host* implementations; the
remote transports (SSH, container) implement the same protocols later.
The pieces, all in `src/weakincentives/sandbox/`:

**`SandboxTransport`** (`_transport.py`) — the remote environment's narrow
waist: one method per filesystem primitive (`stat`, `list`, `read_range`,
`write`, `glob`, `grep`, `delete`, `mkdir`, `rename`, `snapshot`,
`restore`), one exec surface (`exec`, concrete timeout), the egress-sidecar
control plane (`configure_egress`, `configure_credentials`), a `root`
property, and idempotent `close()`. `glob`/`grep` are transport methods so
they execute **server-side** — never a client-side `rglob` over RPC.

**Fault contract** — transport methods raise `TransportFault(code, message)` with a portable `TransportFaultCode`. `exception_for_fault`
maps faults to the facet protocols' exception contract (`not-found` →
`FileNotFoundError`, `is-a-directory`/`not-a-directory`/`exists`/
`permission`/`invalid`/`io`/`snapshot` → the matching standard exception,
`connectivity` → `RuntimeError`: the transport is broken, not the
operation); `fault_for_exception` is the server-side inverse. The
round-trip preserves type and message.

**Remote facets** (`_remote.py`) — `RemoteBackend(transport)` implements
`FilesystemBackend`; `RemoteShell(transport)` implements `Shell`
(argument-shape validation client-side; `cwd` existence/escape checks are
the environment's and come back as faults); `RemoteSandbox` composes them
over one transport and `close()` tears the transport down. After close,
facet access raises `SandboxClosedError`; identity and policy stay
readable.

**`LoopbackTransport`** (`_loopback.py`) — the transport with no wire:
drives a `HostBackend` + `LocalShell` in-process, converting native
exceptions to faults so clients exercise the same fault path a wire
transport produces. The full `FilesystemValidationSuite` and Shell
contract suite run over it (no sockets). After `close()` every method
raises a `connectivity` fault; `owns_root=True` removes the root on close.

**Egress sidecar (control plane).** Enforcement is **not** a WINK-run
object: the egress chokepoint is a sidecar the *environment* owns, and the
transport's `configure_egress`/`configure_credentials` carry policy and
credential material to it. An enforcing sidecar applies the policy to real
traffic and injects bound credentials into allowed requests — the agent
makes credential-less requests and can *use* a secret it can never *read*.
A local environment (host, loopback) has no sidecar, so those calls record
policy and credential names for observability and enforce nothing; secret
material is held in process memory only, never logged, and dropped on
`close()`.

**`RemoteSandboxProvider(connect)`** (`_provider.py`) — materializes a
`WorkspaceConfig` through a fresh transport per open: mounts stage locally
under the same allowed-root/symlink/byte-budget guards as the local
provider and upload through the transport's own primitives, the egress
policy seeds the enforcement point, setup commands run through the remote
shell, and any failure after connect closes the transport (fail-closed).
The staging directory is always removed. Paired with `LoopbackTransport`
it is the local-host provider; remote transports slot in unchanged.

Still to land from M4: the remote transport implementations (the SSH
transport — ACP harness on a box over one SSH connection — and the
container reference topology with the enforcing egress/credential proxy
sidecar) and the optional funnel mode.

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
held by an `Runtime` (see `specs/ADAPTERS.md`): the bound
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

Tests in `tests/sandbox/` mirror the package: `test_shell.py` (the
generic Shell contract from `tests/helpers/shell.py` over `LocalShell`,
plus host env hygiene), `test_sandbox.py` (facets, snapshot/restore,
control plane, close idempotency, secret-material invariants),
`test_config.py` (validation, default-deny, serde round-trip),
`test_provider.py` (mount parity, setup commands, fail-closed open, the
remote provider over loopback), `test_mounts.py` (the mount machinery),
`test_transport.py` (fault↔exception mapping, both directions and the
round-trip law), `test_loopback.py` (loopback transport semantics),
and `test_remote.py` (the full `FilesystemValidationSuite` and Shell
contract suites over `Filesystem(RemoteBackend)`/`RemoteShell` + loopback,
fault translation against a failing transport stub, `RemoteSandbox`
lifecycle, including the local egress-bookkeeping semantics).

## Related Specifications

- `specs/FILESYSTEM.md` - The filesystem facet and backend protocol
- `specs/WORKSPACE.md` - Workspace preview rendered from the opened sandbox
- `specs/MODULE_BOUNDARIES.md` - `sandbox` is a core-layer package
- `refactor/M2.md` - Milestone introducing this package
- `refactor/M3.md` - Sandbox as the execution context
- `refactor/M4.md` - Remote sandbox and the proxy sidecar (next)
