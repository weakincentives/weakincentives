# Prompt Versioning & Persistence Spec

## Purpose

External optimization services need a stable way to identify prompts and to override their content without editing source files. This document defines the metadata that the prompt library must expose so callers can detect when code changes a prompt, decide whether an override is still valid, and swap in alternative text at render time.

## Terminology

- **Namespace (`ns`)** – Required string that groups related prompts (e.g., `webapp/agents`). Namespaces partition override storage and lookup.
- **Prompt key** – Required machine identifier for every `Prompt` within an `ns`. Distinct from the optional human-readable `name`.
- **Section key** – Required machine identifier for every `Section`. Keys compose into ordered tuples (`SectionPath`) that uniquely identify a location inside a prompt.
- **Hash-aware section** – A section type (currently `MarkdownSection` and subclasses such as `ResponseFormatSection`) whose original body template participates in content hashing.
- **Content hash** – Deterministic digest (SHA-256) of a hash-aware section’s original body template as written in source control. Runtime overrides never change the descriptor hash.
- **Lookup key** – The quadruple `(ns, prompt_key, section_path, expected_hash)` used to decide whether an override still applies.
- **Tag** – Free-form label (e.g., `latest`, `stable`, `experiment-a`) that callers use to request a specific family of overrides.

## Data Model

1. Every `Prompt` constructor must accept `key: str`.
1. Every `Prompt` constructor must accept `ns: str` (non-empty).
1. Every `Section` constructor must accept `key: str`. Keys remain stable even for sections that do not participate in hashing.
1. `SectionPath` is the tuple of section keys from root to the target section. The root section’s key appears first.
1. A prompt descriptor enumerates only hash-aware sections. Each entry includes:
   - `path: SectionPath`
   - `content_hash: str` (SHA-256 of the original body template)
1. Hash-aware sections must expose their original body template string so the hashing utility can operate without rendering parameters.

## Hashing Rules

1. Compute hashes solely from the original body template text. Ignore defaults, enable predicates, tools, children, and runtime params.
1. The lookup key `(ns, prompt_key, section_path, expected_hash)` is authoritative. An override is valid only when its stored `expected_hash` equals the descriptor value for the matching path.

## Descriptor Contract

Expose `PromptDescriptor` with:

```python
@dataclass(slots=True)
class SectionDescriptor:
    path: tuple[str, ...]
    content_hash: str

@dataclass(slots=True)
class PromptDescriptor:
    ns: str
    key: str
    sections: list[SectionDescriptor]  # ordered depth-first
```

- `PromptDescriptor.from_prompt(prompt: Prompt) -> PromptDescriptor` walks the section tree, gathers all hash-aware sections, and computes hashes.
- The descriptor omits sections that lack hashes but callers can still traverse the prompt tree if needed via other APIs.

## Override Contract

Introduce:

```python
@dataclass(slots=True)
class PromptOverride:
    ns: str
    prompt_key: str
    tag: str
    overrides: dict[tuple[str, ...], str]  # SectionPath -> replacement body template
```

- Each dict entry represents the new body template to use when the lookup key matches.
- The replacement template may hash to any value after application; descriptors continue to publish the in-code hash.

Define a `PromptVersionStore` protocol:

```python
class PromptVersionStore(Protocol):
    def resolve(
        self,
        description: PromptDescriptor,
        tag: str = "latest",
    ) -> PromptOverride | None: ...
```

Responsibilities:

1. Receive the descriptor and optional tag.
1. Examine available overrides keyed by `(ns, prompt_key, section_path, expected_hash)`.
1. Return a `PromptOverride` containing only the overrides whose expected hash matches the descriptor. Return `None` when nothing applies.
1. Tags are advisory; stores decide which strings they honor.

## Rendering API

Augment `Prompt` with:

```python
class Prompt:
    def render_with_overrides(
        self,
        *params: SupportsDataclass,
        version_store: PromptVersionStore,
        tag: str = "latest",
    ) -> RenderedPrompt:
        ...
```

Behavior:

1. Produce the `PromptDescriptor` for `self`.
1. Ask the store for overrides using the descriptor and tag.
1. For each section path in the override, confirm `expected_hash == descriptor` before substituting the new body template.
1. Render the prompt using the modified bodies, falling back to defaults for sections without overrides or mismatched hashes.
1. Return the same `RenderedPrompt` structure used by `Prompt.render`.

A convenience `render` method MAY delegate to `render_with_overrides` when no store is provided.

## Operational Flow

1. **Bootstrap** – Enumerate descriptors for all prompts and publish them to the optimization service. The service stores overrides keyed by `(ns, prompt_key, section_path, expected_hash)` and tagged as desired.
1. **Runtime** – Call `prompt.render_with_overrides(..., version_store=store, tag=...)`. Overrides whose lookup key matches replace the in-code bodies; the rest are ignored.
1. **Author workflow** – Developers edit prompt bodies in source control. Any change produces a new hash, invalidating stale overrides automatically because the lookup key no longer matches.

## Usage Example

```python
from dataclasses import dataclass

from weakincentives.prompt import Prompt, MarkdownSection
from weakincentives.prompt.versioning import (
    PromptDescriptor,
    PromptOverride,
    PromptVersionStore,
)


@dataclass
class GreetingParams:
    audience: str


prompt = Prompt(
    ns="demo",
    key="welcome_prompt",
    sections=[
        MarkdownSection[GreetingParams](
            key="system",
            title="System",
            template="You are a concise assistant. Greet ${audience} politely.",
        ),
        MarkdownSection[GreetingParams](
            key="closing",
            title="Closing",
            template="Say goodbye to ${audience}.",
        ),
    ],
)

descriptor = PromptDescriptor.from_prompt(prompt)


class MemoryStore(PromptVersionStore):
    def __init__(self) -> None:
        self._data: dict[
            tuple[str, str, tuple[str, ...], str],
            dict[str, str],
        ] = {}

    def register(
        self,
        *,
        ns: str,
        prompt_key: str,
        section_path: tuple[str, ...],
        expected_hash: str,
        tag: str,
        body: str,
    ) -> None:
        lookup = (ns, prompt_key, section_path, expected_hash)
        self._data.setdefault(lookup, {})[tag] = body

    def resolve(
        self,
        descriptor: PromptDescriptor,
        tag: str = "latest",
    ) -> PromptOverride | None:
        overrides: dict[tuple[str, ...], str] = {}
        for section in descriptor.sections:
            lookup = (
                descriptor.ns,
                descriptor.key,
                section.path,
                section.content_hash,
            )
            tagged = self._data.get(lookup)
            if tagged is None:
                continue
            body = tagged.get(tag)
            if body is not None:
                overrides[section.path] = body
        if not overrides:
            return None
        return PromptOverride(description.ns, description.key, tag, overrides)


store = MemoryStore()
store.register(
    ns=descriptor.ns,
    prompt_key=descriptor.key,
    section_path=descriptor.sections[0].path,
    expected_hash=descriptor.sections[0].content_hash,
    tag="stable",
    template="You are an enthusiastic assistant. Welcome ${audience} with energy.",
)

rendered = prompt.render_with_overrides(
    GreetingParams(audience="Operators"),
    version_store=store,
    tag="stable",
)
```

The store only returns overrides when `(prompt_key, section_path, expected_hash)` matches the descriptor. The descriptor itself continues to report the original in-code hash even after an override is applied.

## Non-Goals

- Introducing new templating features or changing section rendering semantics.
- Persisting analytics or override history; external systems control storage and retention.
