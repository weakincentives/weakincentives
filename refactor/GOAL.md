# GOAL — Sandboxed Execution as the Core Model

> North-star design for the sandbox refactor. The milestones (`M1.md` … `M8.md`)
> are the path here. **Backwards compatibility is a non-goal**: we improve
> abstractions, delete dead weight, and accept partial rewrites. Each milestone
> still lands green (`make check`).

______________________________________________________________________

## Vision

There is exactly one place where an agent's tool calls take effect: an
**environment** with a filesystem and a command surface. Today that place is
implicit (a temp dir), and it is modelled twice — once as WINK's `Filesystem`
resource and once as the harness's native `cwd` — so the two drift apart the
moment execution goes remote.

The end state makes the environment a single first-class concept, the
**`Sandbox`**, that is local or remote behind one interface and is the target for
*every* tool call. Getting there lets us simplify several abstractions that exist
only because the environment was never named.

______________________________________________________________________

## First Principles

1. **The environment is fundamental execution context** — like the session, the
   clock, or the deadline. It is not an optional injected resource.
1. **Definition declares intent; the harness provides the environment.** A prompt
   says *what* environment it needs (`SandboxSpec`); the adapter *materializes*
   it (`SandboxProvider`).
1. **One environment, one truth.** WINK-bridged tools and harness-native tools
   target the same `Sandbox`. Divergence must be structurally impossible.
1. **Narrow capabilities, ergonomic facades.** A backend implements the *smallest*
   primitive set; convenience (text, pagination, streaming) is composed once on
   top, never reimplemented per backend.
1. **Transactions are over (session, sandbox).** A tool call snapshots both and
   restores both on failure. Nothing else needs bespoke rollback.
1. **Local and remote differ only in transport.** The same facets run over direct
   syscalls or over an RPC channel.

______________________________________________________________________

## End-State Architecture

```
SandboxSpec ──► SandboxProvider.open() ──► Sandbox ──┬── filesystem: Filesystem  (facade over a backend)
 (intent,         (Local / Remote)        (aggregate │                              └─ FilesystemBackend
  in the prompt)                           root)     └── shell: Shell
                                              │ remote facets run over
                                              ▼
                                        SandboxTransport   (WebSocket/JSON-RPC, podman exec, …)
```

| Concept | Kind | Responsibility |
|---------|------|----------------|
| `SandboxSpec` | frozen value (in the prompt) | Desired environment: mounts, network (ingress/egress), env, setup, read-only |
| `SandboxProvider` | factory (held by the adapter) | `open(spec) -> Sandbox`; pools/manages connections |
| `Sandbox` | aggregate root | Vends `filesystem` + `shell`; `snapshot`/`restore`; `close`. The unit of isolation and rollback |
| `Filesystem` | **facade** (one concrete class) | Ergonomic file API (text, bytes, pagination, streaming) over any backend |
| `FilesystemBackend` | narrow protocol | The ~10 primitives a storage backend implements |
| `Shell` | facet | `run(argv, …) -> CommandResult` inside the sandbox |
| `SandboxTransport` | infrastructure | RPC for a remote sandbox's facets |

### Target module layout

```
src/weakincentives/sandbox/          # NEW foundation package
  _spec.py        SandboxSpec, HostMount, NetworkPolicy
  _shell.py       Shell, CommandResult, LocalShell
  _sandbox.py     Sandbox, LocalSandbox
  _provider.py    SandboxProvider, LocalSandboxProvider
  _transport.py   SandboxTransport
  _remote.py      RemoteSandbox, RemoteShell

src/weakincentives/filesystem/       # SIMPLIFIED
  _backend.py     FilesystemBackend (narrow protocol)
  _facade.py      Filesystem (one concrete facade)
  _streams.py     ByteReader/ByteWriter/TextReader — one impl over a backend
  _host.py        HostBackend
  _memory.py      MemoryBackend   (moved from contrib)
  _remote.py      RemoteBackend   (over SandboxTransport)
  _types.py, _path.py  (kept)
```

### Key shapes (sketch)

```python
class FilesystemBackend(Protocol):           # the narrow waist
    @property
    def root(self) -> str: ...
    @property
    def read_only(self) -> bool: ...
    def stat(self, path: str) -> FileStat: ...
    def list(self, path: str) -> Sequence[FileEntry]: ...
    def glob(self, pattern: str, *, path: str) -> Sequence[GlobMatch]: ...
    def grep(self, pattern: str, *, path: str, glob: str | None,
             max_matches: int | None) -> Sequence[GrepMatch]: ...
    def read_range(self, path: str, *, offset: int, length: int | None) -> bytes: ...
    def write(self, path: str, data: bytes, *, mode: WriteMode) -> int: ...
    def delete(self, path: str, *, recursive: bool) -> None: ...
    def mkdir(self, path: str, *, parents: bool, exist_ok: bool) -> None: ...
    def snapshot(self, *, tag: str | None) -> SnapshotRef: ...
    def restore(self, ref: SnapshotRef) -> None: ...

class Sandbox(Protocol):
    @property
    def id(self) -> str: ...
    @property
    def root(self) -> str: ...
    @property
    def filesystem(self) -> Filesystem: ...
    @property
    def shell(self) -> Shell: ...
    def snapshot(self, *, tag: str | None = None) -> SnapshotRef: ...   # delegates to fs backend
    def restore(self, ref: SnapshotRef) -> None: ...
    def close(self) -> None: ...
```

______________________________________________________________________

## What We Delete or Simplify

| Today | End state | Why |
|-------|-----------|-----|
| Fat `Filesystem` `Protocol` reimplemented per backend (`_host.py`, `_stream_host.py`, `_stream_memory.py`, `_stream_text.py`, `contrib/.../filesystem_memory.py` — **~2,500 lines**) | Narrow `FilesystemBackend` + one `Filesystem` facade + small backends | One place owns streaming/text/pagination; backends shrink to primitives |
| `WorkspaceSection` god-object: intent + temp-dir creation + FS holder + markdown render + refcount/clone | `SandboxSpec` (intent) + provider (materialization) + auto preview (render); section dissolved | Five responsibilities → separate owners |
| `Filesystem` bound as a DI resource; `context.filesystem` is a resource lookup; per-tool snapshot iterates "SINGLETON Snapshotable resources" | `Sandbox` is first-class `ToolContext` state; transactions snapshot (session, sandbox) | The environment is context, not an optional dependency |
| Each adapter resolves `cwd`/workspace its own way (`_resolve_cwd`, Codex `cwd`, ephemeral home) | Adapter base opens a `Sandbox` from the spec; `cwd = sandbox.root` | One materialization path; remote becomes a provider swap |
| Two filesystems (WINK resource vs harness `cwd`) that silently diverge remotely | One `Sandbox` both target | Correctness |
| `SnapshotableFilesystem` as a separate protocol | `snapshot`/`restore` on the backend; `Sandbox` is the rollback unit | Snapshot is intrinsic to a backend, not a bolt-on |

______________________________________________________________________

## Lifecycles

- **`SandboxSpec`** — immutable value; lives with the prompt; no teardown.
- **`Sandbox`** — `Provisioning → Ready → Closed`. Opened by the adapter via its
  provider; **closed by whoever opened it** (`open … try … finally: close`). The
  sole `Closeable`; facets never self-close.
- **`SandboxProvider`** — adapter-held; may pool connections; closed with the
  adapter.
- **Facets (`Filesystem`, `Shell`)** — valid only while their `Sandbox` is Ready.

______________________________________________________________________

## Ground Rules

- **No backwards-compatibility shims.** Delete replaced code outright (consistent
  with the repo's alpha-stability stance).
- **Partial rewrites are fine** when they yield a simpler whole.
- **Every milestone lands green** (`make check`, 100% coverage). Bold within a
  milestone; never leave the tree broken between them.

______________________________________________________________________

## Milestone Map

| # | Milestone | Outcome |
|---|-----------|---------|
| [M1](M1.md) | Filesystem: narrow backend + one facade | Delete per-backend streaming/text dup; backends become primitives |
| [M2](M2.md) | Shell facet | A first-class command surface |
| [M3](M3.md) | Sandbox aggregate + provider | Materialization moves out of the section into a factory |
| [M4](M4.md) | Sandbox as execution context | `ToolContext.sandbox`; transactions over (session, sandbox); drop FS-as-resource |
| [M5](M5.md) | Dissolve `WorkspaceSection`; unify adapters | Prompt declares a spec; one `cwd` path; per-adapter logic deleted |
| [M6](M6.md) | Remote facets over a transport | `RemoteBackend`/`RemoteShell`/`RemoteSandbox`; loopback-tested |
| [M7](M7.md) | Remote provider — Codex in a container | The reference topology, end-to-end |
| [M8](M8.md) | (Optional) Unify the tool surface | Every tool call funnels through the sandbox; transactional totality |

______________________________________________________________________

## Beyond M8 — Continuous Review

The sandbox arc (M1–M8) is the seed, not the whole ambition. [REVIEW.md](REVIEW.md)
is a repeatable workflow that audits the entire library against a quality rubric
and emits further milestones (`M9.md`+) to level up code quality, cut technical
debt, and keep this best-in-class. Its living output — a prioritized,
evidence-based candidate list — is [BACKLOG.md](BACKLOG.md).
