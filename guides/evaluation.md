# Evaluation

`weakincentives.evals` runs a `Dataset` of `EvalCase` values against a
prompt, scores each with an `Evaluator`, and produces an `EvalReport`.
The evaluator framework composes with the rest of the stack — fresh
session and transcript per case so checkers see only that case's
events.

## Built-in evaluators

| Evaluator | Passes when |
| --- | --- |
| `ExactMatch` | `result.final_response.text == case.expected` |
| `Contains` | `case.expected` (str) is a substring of the response |
| `RegexMatch` | `case.expected` (regex) matches the response |
| `ToolCalled` | a particular tool was invoked at least once |
| `AllToolsSucceeded` | every tool call in the run succeeded |
| `LlmJudge` | a custom `JudgeProtocol` impl returns `passed=True` |

## Defining a dataset

```python
# my_evals.py
from weakincentives.adapters import NoopAdapter, ScriptedResponse
from weakincentives.core import MarkdownSection, Prompt
from weakincentives.evals import Contains, Dataset, EvalCase

prompt = Prompt(
    ns="demo",
    key="greet",
    sections=(
        MarkdownSection[None](
            title="System", key="system", template="Be brief and helpful."
        ),
    ),
)

dataset = Dataset(
    cases=(
        EvalCase(name="greet", expected="hello"),
        EvalCase(name="missed", expected="missing"),
    )
)

evaluator = Contains()


def adapter_for(case):
    """Return an adapter scripted for the case under test."""
    text = "hello world" if case.name == "greet" else "nothing here"
    return NoopAdapter(
        responses=(ScriptedResponse(text=text, finish_reason="stop"),)
    )
```

## Running it

From Python:

```python
from weakincentives.evals import run_evaluation

report = run_evaluation(
    dataset=dataset,
    prompt_factory=lambda case: prompt,
    adapter_for=adapter_for,
    evaluator=evaluator,
)
print(f"{report.passed}/{report.total} passed")
```

From the CLI (the `wink` script registered by the package):

```sh
wink eval \
  --prompt my_evals:prompt \
  --dataset my_evals:dataset \
  --evaluator my_evals:evaluator \
  --adapter my_evals:adapter_for
```

The exit code is `0` when every case passes and `1` otherwise, so
`wink eval` slots into CI gates.

## Composing checkers

Build richer evaluators by combining the built-ins inside a custom
`Evaluator`:

```python
from weakincentives.evals import (
    AllToolsSucceeded,
    Contains,
    EvalScore,
    Evaluator,
    ToolCalled,
)


class MustGreetAndCallTool:
    def score(self, case, result, transcript):
        for child in (Contains(), ToolCalled(tool_name="greet"), AllToolsSucceeded()):
            score = child.score(case, result, transcript)
            if not score.passed:
                return score
        return EvalScore(passed=True, score=1.0, detail="all checks passed")


assert isinstance(MustGreetAndCallTool(), Evaluator)  # protocol satisfied
```
