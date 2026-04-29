# weakincentives guides

Short, focused walkthroughs of the most common patterns. Each guide is
runnable as-is against the spine and stdlib-only extras (no provider
SDK required). See `specs/` for design documents.

| Guide | What you'll build |
| --- | --- |
| `quickstart.md` | A typed prompt with one tool, driven through the agent loop with the noop adapter. |
| `evaluation.md` | A dataset of cases scored by built-in evaluators, run from the CLI. |
| `transactions.md` | A failing tool whose filesystem mutations roll back automatically. |
| `progressive-disclosure.md` | A summarised section the model expands on demand. |

The package philosophy is in `specs/POLICIES_OVER_WORKFLOWS.md`; the
layered architecture is in `specs/ARCHITECTURE.md`; the spine itself is
documented in `specs/SPINE.md`.
