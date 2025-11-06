# wink Terminal Agent Specification

## Purpose

The `wink` terminal agent is a conversational assistant built on top of
`weakincentives`. Instead of exposing a collection of discrete CLI commands, the
agent guides the user through prompt override workflows directly from the
terminal. It specializes in helping practitioners understand, inspect, and
modify prompt overrides without juggling file paths or manual JSON editing.

## Core Responsibilities

- Provide a natural-language interface for exploring prompt overrides stored in
  the local overrides store.
- Streamline the process of reviewing and updating override content while
  preserving provenance information surfaced by `weakincentives`.
- Maintain session context so follow-up questions and refinements remain
  anchored to the active override the user is working on.

## Model & Transport

- Default model: `gpt-5` served through the `openai` adapter provided by
  `weakincentives`.
- The agent should surface an explicit message whenever a different model or
  transport is requested so the user can confirm the change.

## Tooling Interface

The terminal agent exposes a minimal tool surface tailored to prompt override
management.

### Retrieve Override Tool

- Input: namespace (`ns`), prompt key (`prompt`), and tag (`tag`, default
  `latest`).
- Behavior: resolves the override using the configured
  `PromptOverridesStore`, returning structured metadata and the JSON body.
- Usage: invoked automatically when the user asks to inspect an override or to
  confirm the current working context before proposing edits.

### Update Override Tool

- Input: namespace, prompt key, tag (default `latest`), and the replacement JSON
  payload.
- Behavior: validates the payload against descriptor expectations when
  available, then calls `PromptOverridesStore.upsert` and returns the resolved
  file path.
- Usage: activated when the user accepts the agent’s proposed modifications or
  supplies their own override content.

## Interaction Flow

1. **Session kickoff** – the agent introduces itself, states that it defaults to
   `gpt-5` via the OpenAI adapter, and explains how it can help with prompt
   overrides.
1. **Intent capture** – it asks clarifying questions to gather the relevant
   namespace, prompt key, and tag before taking any action.
1. **Inspection loop** – when viewing an override, the agent surfaces a concise
   summary plus the full JSON body, highlighting notable fields or differences
   compared to descriptor defaults when available.
1. **Editing loop** – the agent drafts proposed changes, confirms the diff with
   the user, and only writes updates after receiving explicit approval.
1. **Session recap** – upon completion, it summarizes the changes made and the
   file locations touched so the user can continue work outside the chat.

## User Experience Guidelines

- Prefer succinct, actionable prompts that keep the user oriented within the
  current override context.
- Always confirm destructive actions, even when the user’s instruction is
  direct, unless the user explicitly opts into automation for the current
  session.
- Emit validation or storage errors verbosely so users can correct malformed
  payloads without rerunning commands.
- Encourage version control hygiene by reminding the user to commit changes
  after successfully updating overrides.

## Extensibility Notes

- Additional tools (for example, descriptor discovery or diff generation) can be
  layered on later, but the default experience should remain focused on
  retrieval and update workflows.
- Keep the agent shell-agnostic: all messaging and prompts should work in any
  terminal environment without relying on escape sequences or color output.
- Document any future capability expansions alongside updated usage examples so
  downstream agents remain aligned with the conversational interface.
