# Prompt Versioning & Persistence Spec

## Purpose

External systems need stable identifiers and reproducible digests for every `Prompt` and content-bearing section so optimized variants can be stored, recalled, and diffed against in-code defaults. This spec extends the existing prompt model with minimal metadata to make each artifact addressable and hashable without disrupting current rendering behavior. All `Section` instances carry stable keys; only `TextSection` (and concrete subclasses such as `ResponseFormatSection`) participate in automatic content hashing. Custom section types must opt into the hashing contract explicitly if they need versioning semantics.

## Core Concepts

- **Key**: Mandatory machine identifier distinct from display titles. Keys appear on both `Prompt` and `Section` instances and compose into section paths.
- **Content Hash**: Deterministic digest (e.g., SHA-256) derived solely from the original section body template (prior to rendering). Serves as the version token.
- **Descriptor**: Lightweight data object that captures keys and section content hashes for export to external systems.
- **Tag**: Human-readable label such as `stable` or `latest` that selects which optimized artifact a resolver should return without changing the hash contract.

## Addressability

1. Require a `key: str` when constructing a `Prompt`. Keep `name` as the optional human label.
1. Add an optional `key: str | None` to `Section`; default to a slug derived from the title when not provided.
1. Represent section identity with a `SectionPath` = tuple of section keys from root to leaf. Paths stay stable even if titles change. Sections that do not participate in hashing still expose keys so downstream systems can reference them even without hashes.
1. Expose helper utilities that enumerate every section path alongside the prompt key. Include content hashes only for sections that opt into hashing.

## Hashing

1. Compute a deterministic content hash (e.g., SHA-256) for every hash-aware section using only the original body template string defined in code (ignore defaults, predicates, child keys, and runtime parameter values). Custom section classes that want to participate must supply the same hashing hooks.
1. Derive the `Prompt` content hash from its key plus the ordered set of participating section body hashes so the prompt hash changes whenever any descendant hash-aware section's in-code body template changes. Override bodies returned at runtime never influence descriptor hashes.
1. Emit hashes through the descriptor API so external systems can treat `(key, content_hash)` as the cache key and detect drift at any level.
1. Provide helpers that surface these hashes alongside the section tree to simplify change detection for callers.

## External Overrides

1. Define a `PromptVersionStore` protocol with `resolve(description, tag="latest") -> PromptOverride | None`. Implementations return optimized prompt text plus section body overrides (keyed by section path) and the original body hash each override expects. Each section override is the raw body template that should replace the in-code body string.
1. Update `Prompt.render` to query the store before rendering. When the store provides an override whose expected source hash matches the in-code section hash, substitute the persisted body template before rendering; otherwise fall back to code-defined defaults. The replacement template may hash to a different value once applied.
1. Require resolvers to understand at least the `latest` (default) and `stable` tags so callers can opt into slower-moving artifacts without bypassing hash validation. Additional tags remain implementation-defined.
1. Ship a default `PromptVersionStore` implementation backed by the local filesystem so projects can persist overrides without extra infrastructure.
1. Allow partial overrides: unspecified sections still render from the in-code tree, leveraging existing parameter resolution. Sections that do not participate in hashing never receive overrides. Overrides only apply when the expected source hash matches; the replacement template is free to produce a new hash for downstream consumers.

## Operational Flow

1. **Bootstrap**: On startup or during a build step, enumerate descriptors for all prompts and publish them to the external optimization service.
1. **Runtime**: Each render call consults the `PromptVersionStore`. If an override is unavailable or stale, render using defaults and optionally record the computed hash so the store can detect changes later.
1. **Author Workflow**: Developers do not manage manual version numbers. Hashes automatically detect drift whenever code changes a prompt or section.

## Non-Goals

- Introducing new templating features or changing section rendering semantics.
- Persisting runtime analytics; external systems own their storage format and retention policies.
