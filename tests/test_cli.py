# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Tests for ``weakincentives.cli``."""

from __future__ import annotations

import io
from dataclasses import dataclass
from pathlib import Path

import pytest

from weakincentives.cli import main
from weakincentives.core import Session, TypeRegistry
from weakincentives.debug import from_session
from weakincentives.transcript import Transcript, TranscriptEntry


@dataclass(frozen=True)
class Plan:
    steps: tuple[str, ...] = ()


def _bundle_path(tmp_path: Path) -> Path:
    session = Session()
    session[Plan].seed(Plan(steps=("a", "b")))
    transcript = Transcript()
    transcript.append(
        TranscriptEntry(
            timestamp=__import__("datetime").datetime.fromisoformat(
                "2025-01-01T00:00:00+00:00"
            ),
            kind="kind",
            payload={"k": "v"},
        )
    )
    bundle = from_session(session, transcript, metadata={"label": "demo"})
    registry = TypeRegistry()
    registry.register(Plan)
    path = tmp_path / "bundle.json"
    path.write_text(bundle.to_json(registry=registry), encoding="utf-8")
    return path


def test_debug_prints_bundle_summary(tmp_path: Path) -> None:
    path = _bundle_path(tmp_path)
    buffer = io.StringIO()
    code = main(["debug", str(path)], stdout=buffer)
    output = buffer.getvalue()
    assert code == 0
    assert "schema_version: 1" in output
    assert "metadata:" in output
    assert "label = 'demo'" in output
    assert "entries: 1" in output
    assert "[2025-01-01T00:00:00+00:00] kind" in output
    assert "slices: 1" in output
    assert "Plan" in output


def test_debug_reports_missing_path(tmp_path: Path) -> None:
    buffer = io.StringIO()
    code = main(["debug", str(tmp_path / "nope.json")], stdout=buffer)
    assert code == 1
    assert "not found" in buffer.getvalue()


def test_debug_reports_invalid_json(tmp_path: Path) -> None:
    path = tmp_path / "bad.json"
    path.write_text("{", encoding="utf-8")
    buffer = io.StringIO()
    code = main(["debug", str(path)], stdout=buffer)
    assert code == 1
    assert "invalid JSON" in buffer.getvalue()


def test_debug_reports_non_object_payload(tmp_path: Path) -> None:
    path = tmp_path / "list.json"
    path.write_text("[]", encoding="utf-8")
    buffer = io.StringIO()
    code = main(["debug", str(path)], stdout=buffer)
    assert code == 1
    assert "must be an object" in buffer.getvalue()


def test_debug_skips_metadata_section_when_empty(tmp_path: Path) -> None:
    bundle = from_session(Session(), Transcript())
    registry = TypeRegistry()
    path = tmp_path / "bundle.json"
    path.write_text(bundle.to_json(registry=registry), encoding="utf-8")
    buffer = io.StringIO()
    main(["debug", str(path)], stdout=buffer)
    output = buffer.getvalue()
    assert "metadata:" not in output
    assert "entries: 0" in output


def test_main_requires_subcommand() -> None:
    buffer = io.StringIO()
    with pytest.raises(SystemExit):
        main([], stdout=buffer)


# ---------------------------------------------------------------------------
# describe
# ---------------------------------------------------------------------------


def test_describe_rejects_missing_colon() -> None:
    buffer = io.StringIO()
    code = main(["describe", "no_colon"], stdout=buffer)
    assert code == 1
    assert "package.module:attr" in buffer.getvalue()


def test_describe_reports_unimportable_module() -> None:
    buffer = io.StringIO()
    code = main(["describe", "nope_no_such:obj"], stdout=buffer)
    assert code == 1
    assert "cannot import" in buffer.getvalue()


def test_describe_reports_missing_attribute() -> None:
    buffer = io.StringIO()
    code = main(["describe", "weakincentives.core:DoesNotExist"], stdout=buffer)
    assert code == 1
    assert "no attribute" in buffer.getvalue()


def test_describe_rejects_non_prompt_attribute() -> None:
    buffer = io.StringIO()
    code = main(["describe", "weakincentives.core:Session"], stdout=buffer)
    assert code == 1
    assert "not a" in buffer.getvalue()


def test_describe_prints_prompt_summary() -> None:
    """Use a fixture module on disk."""
    import sys
    from pathlib import Path
    from textwrap import dedent

    fixture_root = Path("/tmp/_wink_describe_fixture")
    fixture_root.mkdir(parents=True, exist_ok=True)
    pkg = fixture_root / "wink_demo"
    pkg.mkdir(exist_ok=True)
    _ = (pkg / "__init__.py").write_text("", encoding="utf-8")
    _ = (pkg / "demo.py").write_text(
        dedent(
            """
            from dataclasses import dataclass

            from weakincentives.core import (
                MarkdownSection,
                Prompt,
                Tool,
                ToolContext,
                ToolResult,
            )


            def _noop(params, context):
                del params, context
                return ToolResult.ok(None)


            tool = Tool(name="echo", description="echo back", handler=_noop)
            prompt = Prompt(
                ns="demo",
                key="hello",
                sections=(
                    MarkdownSection[None](
                        title="System",
                        key="system",
                        template="Be helpful.",
                        tools=(tool,),
                    ),
                ),
            )
            """
        ),
        encoding="utf-8",
    )
    sys.path.insert(0, str(fixture_root))
    try:
        if "wink_demo" in sys.modules:
            del sys.modules["wink_demo"]
        if "wink_demo.demo" in sys.modules:
            del sys.modules["wink_demo.demo"]
        buffer = io.StringIO()
        code = main(["describe", "wink_demo.demo:prompt"], stdout=buffer)
        output = buffer.getvalue()
        assert code == 0
        assert "prompt: demo/hello" in output
        assert "System" in output
        assert "echo: echo back" in output
    finally:
        sys.path.remove(str(fixture_root))


# ---------------------------------------------------------------------------
# eval
# ---------------------------------------------------------------------------


def _write_eval_fixture() -> str:
    import sys
    from pathlib import Path
    from textwrap import dedent

    fixture_root = Path("/tmp/_wink_eval_fixture")
    fixture_root.mkdir(parents=True, exist_ok=True)
    pkg = fixture_root / "wink_eval_demo"
    pkg.mkdir(exist_ok=True)
    _ = (pkg / "__init__.py").write_text("", encoding="utf-8")
    _ = (pkg / "demo.py").write_text(
        dedent(
            """
            from weakincentives.adapters import NoopAdapter, ScriptedResponse
            from weakincentives.core import MarkdownSection, Prompt
            from weakincentives.evals import (
                Contains,
                Dataset,
                EvalCase,
            )


            prompt = Prompt(
                ns="demo",
                key="eval",
                sections=(
                    MarkdownSection[None](
                        title="System", key="system", template="Be brief."
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
                if case.name == "greet":
                    text = "hello world"
                else:
                    text = "nothing here"
                return NoopAdapter(
                    responses=(ScriptedResponse(text=text, finish_reason="stop"),)
                )
            """
        ),
        encoding="utf-8",
    )
    if str(fixture_root) not in sys.path:
        sys.path.insert(0, str(fixture_root))
    if "wink_eval_demo" in sys.modules:
        del sys.modules["wink_eval_demo"]
    if "wink_eval_demo.demo" in sys.modules:
        del sys.modules["wink_eval_demo.demo"]
    return "wink_eval_demo.demo"


def test_eval_runs_full_dataset_and_summarises_results() -> None:
    module = _write_eval_fixture()
    buffer = io.StringIO()
    code = main(
        [
            "eval",
            "--prompt",
            f"{module}:prompt",
            "--dataset",
            f"{module}:dataset",
            "--evaluator",
            f"{module}:evaluator",
            "--adapter",
            f"{module}:adapter_for",
        ],
        stdout=buffer,
    )
    output = buffer.getvalue()
    assert code == 1  # one case fails
    assert "passed: 1/2" in output
    assert "PASS greet" in output
    assert "FAIL missed" in output


def test_eval_rejects_invalid_target() -> None:
    buffer = io.StringIO()
    code = main(
        [
            "eval",
            "--prompt",
            "no_colon",
            "--dataset",
            "x:y",
            "--evaluator",
            "x:y",
            "--adapter",
            "x:y",
        ],
        stdout=buffer,
    )
    assert code == 1
    assert "package.module:attr" in buffer.getvalue()


def test_eval_rejects_unimportable_module() -> None:
    buffer = io.StringIO()
    code = main(
        [
            "eval",
            "--prompt",
            "doesnotexist_xyz:prompt",
            "--dataset",
            "doesnotexist_xyz:dataset",
            "--evaluator",
            "doesnotexist_xyz:evaluator",
            "--adapter",
            "doesnotexist_xyz:adapter_for",
        ],
        stdout=buffer,
    )
    assert code == 1
    assert "cannot import" in buffer.getvalue()


def test_eval_rejects_missing_attribute() -> None:
    module = _write_eval_fixture()
    buffer = io.StringIO()
    code = main(
        [
            "eval",
            "--prompt",
            f"{module}:nope_no_such",
            "--dataset",
            f"{module}:dataset",
            "--evaluator",
            f"{module}:evaluator",
            "--adapter",
            f"{module}:adapter_for",
        ],
        stdout=buffer,
    )
    assert code == 1
    assert "no attribute" in buffer.getvalue()


def test_eval_rejects_wrong_typed_targets() -> None:
    module = _write_eval_fixture()
    buffer = io.StringIO()
    code = main(
        [
            "eval",
            "--prompt",
            f"{module}:dataset",  # wrong type
            "--dataset",
            f"{module}:dataset",
            "--evaluator",
            f"{module}:evaluator",
            "--adapter",
            f"{module}:adapter_for",
        ],
        stdout=buffer,
    )
    assert code == 1
    assert "is not a Prompt" in buffer.getvalue()


def test_eval_rejects_non_dataset_dataset() -> None:
    module = _write_eval_fixture()
    buffer = io.StringIO()
    code = main(
        [
            "eval",
            "--prompt",
            f"{module}:prompt",
            "--dataset",
            f"{module}:prompt",  # wrong type
            "--evaluator",
            f"{module}:evaluator",
            "--adapter",
            f"{module}:adapter_for",
        ],
        stdout=buffer,
    )
    assert code == 1
    assert "is not a Dataset" in buffer.getvalue()


def test_eval_rejects_non_evaluator_evaluator() -> None:
    module = _write_eval_fixture()
    buffer = io.StringIO()
    code = main(
        [
            "eval",
            "--prompt",
            f"{module}:prompt",
            "--dataset",
            f"{module}:dataset",
            "--evaluator",
            f"{module}:prompt",  # wrong type
            "--adapter",
            f"{module}:adapter_for",
        ],
        stdout=buffer,
    )
    assert code == 1
    assert "is not an Evaluator" in buffer.getvalue()


def test_eval_rejects_non_callable_adapter_factory() -> None:
    module = _write_eval_fixture()
    buffer = io.StringIO()
    code = main(
        [
            "eval",
            "--prompt",
            f"{module}:prompt",
            "--dataset",
            f"{module}:dataset",
            "--evaluator",
            f"{module}:evaluator",
            "--adapter",
            f"{module}:dataset",  # wrong type
        ],
        stdout=buffer,
    )
    assert code == 1
    assert "is not callable" in buffer.getvalue()
