# Feedback

**Feedback** is soft guidance: advisory text injected into the agent's
working memory during execution. Where a policy *gates* a tool call,
feedback *nudges* the agent without blocking. The agent reads it,
considers it, and decides what to do.

This is the second tier of WINK's three-tier control model.

______________________________________________________________________

## What feedback does

A feedback provider observes the trajectory of an agent's execution and
emits a short message at the right moment. Examples:

- "You have 2 minutes remaining before the deadline. Prioritize critical
  remaining work."
- "You have invoked 12 tools without producing any output. Consider
  summarizing what you have found."
- "A file matching `report.draft.md` was just created. Remember that the
  required output is `report.md`, not the draft."

The feedback is injected into the agent's context immediately after a
tool call completes. The agent sees it on its next reasoning step and
incorporates it (or not — feedback is non-binding).

______________________________________________________________________

## Why advisory, not blocking

Some constraints are not invariants. "Don't waste time" is not a yes/no
gate; it is a slope. "Watch out for drift" is not a precondition; it is a
nudge. Trying to encode these as policies forces a binary decision the
constraint doesn't have.

Feedback fits the gradient. The provider observes accumulated state — how
many calls have happened, how much time has elapsed, what files exist —
and produces a message scaled to the situation. As the situation changes,
the message changes. Nothing is denied; nothing is required.

The model handles the trade-off. If the feedback says "you're running out
of time, consider wrapping up" and the agent decides the remaining work is
critical, it can press on. If the feedback says "you're churning, try a
different approach" and the agent agrees, it pivots. The author of the
feedback provider does not have to predict every situation.

______________________________________________________________________

## Triggers

A provider runs only when its trigger conditions are met. Triggers are:

- **Every N tool calls.** Useful for periodic check-ins.
- **Every N seconds.** Useful for time-based reminders (deadlines, lease
  renewals).
- **On file created.** Useful for one-shot guidance tied to specific
  outputs appearing on disk.

Conditions are independent — providers can specify several, and any one
firing triggers a run. Each provider tracks its own trigger state, so
multiple providers cadence independently.

The trigger model is what makes feedback proportional. A provider that
reminds the agent of a deadline does not need to fire on every tool call;
it fires every 30 seconds. A provider that warns about a draft filename
fires once when the file appears and not again.

______________________________________________________________________

## All matching providers run

When several providers' triggers fire on the same tool result, *all* of
them produce feedback. The framework collects every message and combines
them into a single block delivered to the agent.

This is deliberate. First-match-wins would make ordering significant, hide
guidance, and create surprising omissions. Combining all matched feedback
ensures no guidance is silently dropped, even when several conditions hit
simultaneously.

The combined block is wrapped in a structured envelope (one message per
provider) so the agent can parse and respond to each independently. From
the agent's perspective, it is reading a small set of nested advisories,
not a single concatenated paragraph.

______________________________________________________________________

## Where feedback is delivered

Feedback is delivered *in-band*, immediately after the tool call that
triggered it. It rides on the same response the model is about to read —
it does not wait for the next render of the prompt.

The implication: feedback can correct course on the very next reasoning
step. There is no outer workflow loop required. The agent reads the tool
result, reads the feedback alongside it, and proceeds.

Different harnesses deliver this through different channels — a hook
that adds context to a tool result, a server-side append, an MCP
post-call hook — but the *semantic* is uniform: feedback arrives with the
tool result and informs the next turn.

______________________________________________________________________

## Stored if delivered

Every piece of feedback that ships to the agent is recorded in session
state. This serves several purposes:

- **Debug bundles** can show every nudge the agent received.
- **Trigger cadence** uses past delivery to compute "calls since last
  feedback from this provider."
- **Provider state** (e.g., "the file-created trigger has fired") is
  recoverable across restarts because it lives in the session.
- **Tests** can assert on the feedback that was issued.

This is the same principle as everywhere else in WINK: if a thing
happened, it is in the session.

______________________________________________________________________

## Feedback and policies are different tools

It is tempting to encode every constraint as a policy or every constraint
as feedback. They serve different purposes:

| Need | Use a policy | Use feedback |
|---|---|---|
| Hard invariant (read-before-write) | Yes | No |
| Termination criterion (must produce report.md) | Use a completion checker | No |
| Soft drift correction ("you're running out of time") | No | Yes |
| Periodic check-in ("every N calls") | No | Yes |
| Hint on a specific event ("file created, did you mean…") | No | Yes |

When in doubt: if the answer is "the call must not happen," it is a
policy. If the answer is "the agent should consider this," it is
feedback. If the answer is "the agent must not stop until X," it is a
completion checker.

______________________________________________________________________

## Anti-patterns

- **Feedback that should be a policy.** "Don't write to that file" is a
  hard rule, not a nudge. Use a policy.
- **Feedback that should be in the prompt.** Static instructions belong
  in section text. Feedback exists for *runtime-conditional* guidance.
- **Trigger conditions that fire on every call.** That is not a trigger;
  it is a static instruction. Move it into the prompt.
- **Verbose, multi-paragraph feedback.** The agent has a limited
  attention budget. Feedback should be tight, specific, and actionable.

______________________________________________________________________

## Pointers

- [POLICIES](06-POLICIES.md) — the gating tier above feedback.
- [COMPLETION-CHECKING](08-COMPLETION-CHECKING.md) — the verification
  tier below feedback.
- [STATE](05-STATE.md) — where feedback history and trigger state are
  stored.
- [PRINCIPLES](PRINCIPLES.md) §1 — why feedback travels with the prompt
  definition rather than the harness.
