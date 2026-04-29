# Progressive Disclosure

A long prompt wastes tokens when the model only needs part of it.
Progressive disclosure lets you ship sections in `SUMMARY` mode (a
short stub) and have the model expand them on demand.

## Authoring a summarised section

```python
from weakincentives.core import (
    MarkdownSection,
    Prompt,
    SectionVisibility,
)

prompt = Prompt(
    ns="demo",
    key="docs",
    sections=(
        MarkdownSection[None](
            title="Reference",
            key="reference",
            template="### Full reference\n\n... long content ...",
            summary="Reference is hidden. Call `open_sections(keys=['reference'])` to expand.",
            visibility=SectionVisibility.SUMMARY,
        ),
    ),
)

# When rendered, only the summary appears:
assert "Reference is hidden" in prompt.render().text
assert "Full reference" not in prompt.render().text
```

## Wiring the disclosure tools

`weakincentives.disclosure` ships two tools and a session slice:

- `open_sections_tool` — the model calls it with a list of keys to
  expand.
- `read_section_tool` — read-only fetch of a single section's full
  body.
- `SectionExpansions` — the slice that records which keys are
  currently expanded.

```python
from weakincentives.core import Session
from weakincentives.disclosure import (
    SectionExpansions,
    apply_visibility_overrides,
    open_sections_tool,
    read_section_tool,
)

session = Session()
session.install(SectionExpansions, initial=SectionExpansions)
```

Add the two tools to the section that documents the disclosure
contract (or any always-visible section), then on every iteration
rebuild the prompt for the model with:

```python
visible_prompt = apply_visibility_overrides(prompt, session)
rendered = visible_prompt.render()
```

`apply_visibility_overrides` walks the entire section tree (including
nested children) and rebuilds expanded sections with
`visibility=FULL`. The original prompt is unchanged; the rebuild only
materialises when the model actually expands something.

## Why it composes

The slice lives on the session, so:

- The expansion state survives across iterations of the agent loop
  without any extra plumbing.
- It's captured in debug bundles via `core.capture(session)`.
- It rolls back automatically inside a `tool_transaction` if the call
  that triggered the expansion later fails.

## Picking what to summarise

A useful default: anything the model rarely needs (long reference
sections, exhaustive examples). The summary should explicitly tell
the model how to expand the section, e.g.:

```
This section is summarised. Call open_sections(keys=["reference"])
to view the full body.
```

Models follow that instruction reliably; the result is a compact base
prompt with on-demand depth.
