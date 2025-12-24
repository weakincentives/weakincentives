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

"""Tests for the wink docs CLI subcommand."""

from __future__ import annotations

import pytest

from weakincentives.cli import wink


def test_docs_reference_outputs_llms(capsys: pytest.CaptureFixture[str]) -> None:
    """--reference prints llms.md content."""
    exit_code = wink.main(["docs", "--reference"])

    assert exit_code == 0
    captured = capsys.readouterr()
    assert "WINK" in captured.out or "weakincentives" in captured.out.lower()


def test_docs_guide_outputs_guide(capsys: pytest.CaptureFixture[str]) -> None:
    """--guide prints WINK_GUIDE.md content."""
    exit_code = wink.main(["docs", "--guide"])

    assert exit_code == 0
    captured = capsys.readouterr()
    # Guide is substantial
    assert len(captured.out) > 1000


def test_docs_specs_outputs_all_specs(capsys: pytest.CaptureFixture[str]) -> None:
    """--specs prints all spec files with headers."""
    exit_code = wink.main(["docs", "--specs"])

    assert exit_code == 0
    captured = capsys.readouterr()
    assert "<!-- specs/ADAPTERS.md -->" in captured.out
    assert "<!-- specs/TOOLS.md -->" in captured.out


def test_docs_specs_sorted_alphabetically(capsys: pytest.CaptureFixture[str]) -> None:
    """Specs are printed in alphabetical order."""
    exit_code = wink.main(["docs", "--specs"])

    assert exit_code == 0
    captured = capsys.readouterr()
    adapters_pos = captured.out.find("<!-- specs/ADAPTERS.md -->")
    tools_pos = captured.out.find("<!-- specs/TOOLS.md -->")
    workspace_pos = captured.out.find("<!-- specs/WORKSPACE.md -->")
    assert adapters_pos < tools_pos < workspace_pos


def test_docs_no_flags_returns_error(capsys: pytest.CaptureFixture[str]) -> None:
    """No flags prints usage and returns error."""
    exit_code = wink.main(["docs"])

    assert exit_code == 1
    captured = capsys.readouterr()
    assert "--reference" in captured.out
    assert "--guide" in captured.out
    assert "--specs" in captured.out


def test_docs_multiple_flags_combines_output(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Multiple flags combine output with separators."""
    exit_code = wink.main(["docs", "--reference", "--guide"])

    assert exit_code == 0
    captured = capsys.readouterr()
    assert "\n---\n" in captured.out


def test_docs_all_flags_outputs_in_order(capsys: pytest.CaptureFixture[str]) -> None:
    """Output is printed in order: reference, guide, specs."""
    exit_code = wink.main(["docs", "--reference", "--guide", "--specs"])

    assert exit_code == 0
    captured = capsys.readouterr()

    # Find positions of key content
    ref_pos = captured.out.find("WINK")
    specs_pos = captured.out.find("<!-- specs/")

    assert ref_pos < specs_pos


def test_docs_missing_file_returns_error(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """FileNotFoundError returns exit code 2."""

    def raise_not_found(name: str) -> str:
        msg = f"No such file: {name}"
        raise FileNotFoundError(msg)

    monkeypatch.setattr(wink, "_read_doc", raise_not_found)

    exit_code = wink.main(["docs", "--reference"])

    assert exit_code == 2
    captured = capsys.readouterr()
    assert "Documentation not found" in captured.err
    assert "packaging error" in captured.err


def test_read_doc_reads_bundled_file() -> None:
    """_read_doc reads from weakincentives.docs package."""
    content = wink._read_doc("llms.md")

    assert isinstance(content, str)
    assert len(content) > 100
    assert "WINK" in content or "weakincentives" in content.lower()


def test_read_specs_concatenates_with_headers() -> None:
    """_read_specs concatenates all specs with headers."""
    content = wink._read_specs()

    assert isinstance(content, str)
    assert "<!-- specs/ADAPTERS.md -->" in content
    assert "<!-- specs/SESSIONS.md -->" in content
    # Verify separation between specs
    assert "\n\n" in content
