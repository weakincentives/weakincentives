Declarative safety constraints and production hardening!

Weak Incentives (WINK) v0.19.0 is now available—focused on making agent tooling safer and more observable.

**Highlights:**

* **Tool Policies** for declarative constraints on tool invocation sequences. `SequentialDependencyPolicy` enforces unconditional ordering (lint → build → test → deploy). `ReadBeforeWritePolicy` prevents file overwrites without reading first—enabled by default in VFS and Podman sections.

* **Exhaustiveness checking** for union types via `assert_never` sentinels. When you add a variant to `SliceOp` or other unions, pyright catches missing handlers at type-check time—no runtime surprises.

* **Skills as a core library concept.** Skills promoted from the Claude Agent SDK adapter to `weakincentives.skills`, following the Agent Skills specification. Mount-time validation catches misconfigured skills early with clear error messages.

* **Binary filesystem support.** `read_bytes()` and `write_bytes()` enable proper handling of images, archives, and exact-copy operations without encoding overhead.

* **Comprehensive prompt overrides.** Override task examples, individual steps, and tool examples—not just section bodies. Fine-grained control for prompt optimization pipelines.

v0.18.0 (released two days ago) added **LoopGroup** for Kubernetes-ready deployments with health probes and watchdog monitoring, **session evaluators** for testing agent behavior patterns, and **TLA+ spec embedding** for co-locating formal verification with implementation.
