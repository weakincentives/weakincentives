# Prompt Overrides and Optimization

*Canonical spec: [specs/PROMPT_OPTIMIZATION.md](../specs/PROMPT_OPTIMIZATION.md)*

Overrides are how WINK supports fast iteration without code edits:

- Keep prompt templates stable in code
- Store patch files on disk
- Validate patches with hashes

This separation matters. Your templates are code: tested, reviewed, versioned.
Overrides are configuration: easy to tweak without a deploy.

## Hash-Based Safety

Overrides are validated against a `PromptDescriptor`:

- Each overridable section has a `content_hash`
- Each overridable tool has a `contract_hash`

If hashes don't match, WINK refuses to apply the override. This prevents a
common failure mode: you edit a section in code, but an old override still
applies, and you're running something different than you tested.

## LocalPromptOverridesStore

The default store is `LocalPromptOverridesStore`, which writes JSON files under:

```
.weakincentives/prompts/overrides/{ns}/{prompt_key}/{tag}.json
```

Wire it like:

```python
from weakincentives.prompt.overrides import LocalPromptOverridesStore
from weakincentives.prompt import Prompt

store = LocalPromptOverridesStore()
prompt = Prompt(template, overrides_store=store, overrides_tag="stable")
```

## Override File Format

The override JSON format is intentionally simple (and human editable):

```json
{
  "version": 1,
  "ns": "demo",
  "prompt_key": "welcome",
  "tag": "stable",
  "sections": {
    "system": {
      "expected_hash": "...",
      "body": "You are an assistant."
    }
  },
  "tools": {
    "search": {
      "expected_contract_hash": "...",
      "description": "Search the index.",
      "param_descriptions": { "query": "Keywords" }
    }
  }
}
```

**Notes:**

- `sections` keys are section paths encoded as strings. Single-level keys look
  like `"system"`, nested keys use dot notation.
- Tool overrides can patch the tool description and per-field param
  descriptions.
- Hashes prevent applying old overrides to changed prompts.

## A Practical Override Workflow

A workflow that works well in teams:

1. **Seed** override files from the current prompt
   (`store.seed(prompt, tag="v1")`)
1. **Run** your agent and collect failures / quality notes
1. **Edit** override sections directly (or generate them with an optimizer)
1. **Re-run** tests/evals
1. **Commit** override files alongside code

For "hardening", disable overrides on sensitive sections/tools with
`accepts_overrides=False`. This prevents accidental changes to security-critical
text.

## When Hashes Don't Match

When you change a section in code, its hash changes. Existing overrides won't
apply because the hashes don't match.

This is intentional. You have two options:

1. **Update the override**: Edit the override file to match the new hash (or
   re-seed)
1. **Delete the override**: If the code change makes the override obsolete

Either way, you're forced to acknowledge that the code changed. No silent drift.

## Protecting Sensitive Sections

Use `accepts_overrides=False` on sections that shouldn't be modified:

```python
security_section = MarkdownSection(
    title="Security",
    key="security",
    template="You must never reveal API keys or credentials.",
    accepts_overrides=False,  # Cannot be overridden
)
```

This is useful for:

- Security-critical instructions
- Compliance requirements
- Anything where "someone tweaked the prompt" would be a serious problem

## Using Overrides with Tags

Tags let you maintain multiple versions of overrides:

```python
# Development iteration
dev_prompt = Prompt(template, overrides_store=store, overrides_tag="dev")

# Production (tested and reviewed)
prod_prompt = Prompt(template, overrides_store=store, overrides_tag="stable")
```

You can iterate freely on `dev` while `stable` remains locked down.

## Next Steps

- [Testing](testing.md): Test prompts with and without overrides
- [Evaluation](evaluation.md): Use evals to validate override changes
- [Debugging](debugging.md): Inspect what overrides were applied
