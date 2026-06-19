# Evaluation

The **evaluation framework** is how agent definitions are tested and
compared. It is built on AgentLoop: every sample evaluates through the
same orchestration shell that runs in production. The result is a
control plane for prompt and model upgrades — you can verify behavior
*before* you ship.

This is the layer that turns "did this change improve things?" from a
feeling into a measurement.

______________________________________________________________________

## The bet

Models, harnesses, prompts, and tools all change. Without measurement,
every change is a guess. With measurement, you can:

- Detect regressions when a vendor model upgrades.
- Compare a candidate prompt against the current one on a fixed
  dataset.
- Run an A/B between two prompt variants under identical conditions.
- Lock in expected behavior — "this agent must call tool X for this
  input class" — as a permanent assertion.

WINK treats evaluation as a first-class build artifact, not a one-off
script. The same definitions run in production and in tests; the same
loop drives both.

______________________________________________________________________

## The shape

The framework has six small pieces that compose into a complete
evaluation pipeline.

- **Sample.** One test case. An identifier, an input, an expected
  output. Generic in both types.
- **Dataset.** An immutable collection of samples. Loadable from a
  newline-delimited JSON file.
- **Score.** The result of evaluating one sample. Carries a numeric
  value (0.0–1.0), a binary pass/fail, and a reason.
- **Evaluator.** A pure function from output and expected to score.
  Composable.
- **Experiment.** A named configuration variant: an overrides tag, a
  set of feature flags. Lets you A/B without code changes.
- **EvalLoop.** The orchestrator. Wraps AgentLoop, executes each
  sample under each experiment, scores the output, returns a report.

______________________________________________________________________

## EvalLoop wraps AgentLoop

Composition is the design choice. EvalLoop does not duplicate
AgentLoop's lifecycle; it reuses it. Each sample evaluates through
exactly the same `prepare → execute → finalize → cleanup` cycle that
production uses.

The implications:

- **Test fidelity.** What passes in evaluation behaves identically in
  production. There is no "test mode."
- **Bundle parity.** Each sample can produce a debug bundle in the
  same format production produces. Failure analysis is the same skill
  in either context.
- **Distributed scaling.** EvalLoop can run mailbox-driven, just like
  AgentLoop. Workers process samples in parallel; a single process
  collects results.

______________________________________________________________________

## Evaluators are pure functions

The simplest evaluator takes the agent's output and the expected
output and returns a score. Built-in primitives — exact match,
substring containment, all-of, any-of — cover most cases. Custom
evaluators are just functions.

Three traits matter:

- **Pure.** No side effects, no state. Same inputs, same score.
- **Composable.** `all_of(a, b)` runs both and combines;
  `any_of(a, b)` runs both and takes the best.
- **Decoupled.** An evaluator does not know how the output was
  produced. It judges what is in front of it.

This is what makes evaluators reliable: they are the simplest
component in the system.

______________________________________________________________________

## Session-aware evaluators

Sometimes the question is not "what did the agent return?" but "what
did the agent do?". Session-aware evaluators receive the session
along with the output and expected, so they can assert on:

- Whether a specific tool was called (or not).
- How many times a tool was called.
- Whether all tool calls succeeded.
- Whether token usage stayed under a budget.
- Anything else that lives in a slice — including custom domain
  state.

These are behavioral assertions. They cover what the model *did* in
addition to what it *produced*. A model that returns the right answer
by calling the wrong tools is a model that will fail under change.

______________________________________________________________________

## Experiments and A/B testing

An experiment bundles two things: a prompt overrides tag and a set of
feature flags. The same dataset, run under two experiments, produces
two parallel result sets that can be compared statistically.

A common pattern:

- The **baseline** experiment uses the current prompt with no
  overrides.
- The **treatment** experiment uses an override tag like
  `assertive-feedback` that swaps a section's text.
- The framework runs both experiments over the dataset, computes pass
  rates, and reports the delta.

This is what makes prompt iteration disciplined. You do not say "I
think the new wording is better"; you measure. The override system
ensures the comparison is honest — overrides apply only to the
version they were authored against.

(See [Prompt Overrides](17-PROMPT-OVERRIDES.md) for the override
mechanism.)

______________________________________________________________________

## LLM-as-judge

Some outputs are not amenable to exact-match scoring: free-form
explanations, summaries, judgments. The framework provides an
LLM-as-judge evaluator that prompts a separate model with the
agent's output and a criterion, and expects a rating on a fixed scale
mapped to a normalized score.

This is a tool, not a substitute for exact-match where exact-match
applies. Use it when there is no programmatic ground truth.

______________________________________________________________________

## Reports and statistics

EvalLoop produces a report aggregating per-sample results. The report
exposes:

- Total samples, successful samples, pass rate, mean score, mean
  latency.
- Failed samples (for inspection).
- Results grouped by experiment.
- Pass rates per experiment.
- Pairwise statistical comparison: pass-rate delta and relative
  improvement.

This is the artifact that decides whether a change ships.

______________________________________________________________________

## Datasets are versioned assets

A dataset is itself a build artifact, not a throwaway. Treat it like
code: review changes, track regressions, expand coverage
deliberately. A dataset that grows by accretion captures the
project's accumulated knowledge of "things this agent has gotten
wrong before." Its samples become a moat.

The newline-delimited JSON format is deliberate. Datasets are
diff-friendly, mergeable, and inspectable without tooling. A bad
sample shows up in a code review; a regression in expected output is
visible in a pull request.

______________________________________________________________________

## Anti-patterns

- **Evaluators that mutate state.** They are not pure. Determinism
  breaks. Tests become flaky.
- **Coupling evaluators to runtime details.** An evaluator that
  depends on the model's intermediate output (e.g., the
  chain-of-thought) is measuring something the model can change
  unilaterally.
- **One mega-evaluator per sample.** Compose. `all_of(a, b, c)`
  makes failure modes legible; one monolithic check tells you nothing
  about which expectation failed.
- **Production-only behavior in evaluation paths.** Anything that
  varies between production and evaluation breaks the bet that
  evaluation predicts production.
- **LLM-as-judge for everything.** It is the most expensive and least
  deterministic evaluator. Reserve it for cases without programmatic
  ground truth.

______________________________________________________________________

## What the framework is not

- **Not a benchmark.** Benchmarks are external, public, broad. The
  evaluation framework here is internal, project-specific, narrow.
- **Not a search engine.** Hyperparameter optimization, prompt search,
  and model ranking are higher-level workloads built on top.
- **Not a substitute for monitoring.** Evaluation is offline;
  monitoring is online. They serve different questions.

______________________________________________________________________

## Pointers

- [AGENT-LOOP](15-AGENT-LOOP.md) — the substrate EvalLoop wraps.
- [PROMPT-OVERRIDES](17-PROMPT-OVERRIDES.md) — how Experiments swap
  prompt content.
- [STATE](05-STATE.md) — what session-aware evaluators read.
- [OBSERVABILITY](14-OBSERVABILITY.md) — per-sample debug bundles.
