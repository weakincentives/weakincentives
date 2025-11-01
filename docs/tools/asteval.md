# ASTEVAL Evaluation Tool

The `AstevalSection` integrates the [`asteval`](https://github.com/lmfit/asteval)
interpreter into prompts so agents can execute short Python snippets without
leaving the weakincentives runtime. The section surfaces the
`evaluate_python` tool, which combines code execution with optional reads and
writes against the session-scoped virtual filesystem.

## Highlights

- **Deterministic sandbox** – The interpreter runs with `use_numpy=False`, a
  minimal symbol table, and a five second timeout guard. Stdout/stderr streams
  are captured and truncated to 4,096 characters to keep transcripts tidy.
- **VFS bridge** – Parameters accept `reads` and `writes` lists composed of
  `EvalFileRead`/`EvalFileWrite` dataclasses. Reads inject file contents into the
  runtime, while writes resolve templated content against the final globals and
  stage updates through the `VirtualFileSystem` reducers shared with the VFS
  tool suite.
- **Helper functions** – Evaluation code can call `read_text(path)` to fetch the
  latest VFS contents or `write_text(path, content, mode)` to queue additional
  writes. Helper writes obey the same ASCII and length constraints as native VFS
  operations.
- **Traceable output** – Results include the `repr` of the final expression (when
  available), captured stdout/stderr, the stringified globals snapshot, and the
  resolved read/write descriptors. This mirrors the expectations described in
  `specs/ASTEVAL.md` so adapters can persist full telemetry.

## Usage

Instantiate the section alongside the VFS tools when constructing a prompt:

```python
from weakincentives.prompt import Prompt
from weakincentives.session import Session
from weakincentives.tools import AstevalSection, VfsToolsSection

session = Session()
prompt = Prompt(
    ns="agents/background",
    key="workspace",
    name="workspace-tools",
    sections=[
        VfsToolsSection(session=session),
        AstevalSection(session=session),
    ],
)
```

The spec in `specs/ASTEVAL.md` covers parameter semantics, timeout behaviour,
and the reducer integration in more detail.
