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

# =============================================================================
# List subcommand tests
# =============================================================================


def test_docs_list_all(capsys: pytest.CaptureFixture[str]) -> None:
    """wink docs list shows both specs and guides."""
    exit_code = wink.main(["docs", "list"])

    assert exit_code == 0
    captured = capsys.readouterr()
    assert "SPECS" in captured.out
    assert "GUIDES" in captured.out
    assert "ADAPTERS" in captured.out
    assert "quickstart" in captured.out


def test_docs_list_specs(capsys: pytest.CaptureFixture[str]) -> None:
    """wink docs list specs shows only specs."""
    exit_code = wink.main(["docs", "list", "specs"])

    assert exit_code == 0
    captured = capsys.readouterr()
    assert "SPECS" in captured.out
    assert "GUIDES" not in captured.out
    assert "ADAPTERS" in captured.out


def test_docs_list_guides(capsys: pytest.CaptureFixture[str]) -> None:
    """wink docs list guides shows only guides."""
    exit_code = wink.main(["docs", "list", "guides"])

    assert exit_code == 0
    captured = capsys.readouterr()
    assert "GUIDES" in captured.out
    assert "SPECS" not in captured.out
    assert "quickstart" in captured.out


def test_docs_list_shows_descriptions(capsys: pytest.CaptureFixture[str]) -> None:
    """List output includes document descriptions."""
    exit_code = wink.main(["docs", "list", "specs"])

    assert exit_code == 0
    captured = capsys.readouterr()
    # ADAPTERS should show its description
    assert "Provider integrations" in captured.out


def test_docs_list_shows_document_count(capsys: pytest.CaptureFixture[str]) -> None:
    """List output includes document count."""
    exit_code = wink.main(["docs", "list", "specs"])

    assert exit_code == 0
    captured = capsys.readouterr()
    assert "documents)" in captured.out


# =============================================================================
# Search subcommand tests
# =============================================================================


def test_docs_search_finds_matches(capsys: pytest.CaptureFixture[str]) -> None:
    """wink docs search finds matching content."""
    exit_code = wink.main(["docs", "search", "Session"])

    assert exit_code == 0
    captured = capsys.readouterr()
    assert "Found" in captured.out
    assert "matches" in captured.out


def test_docs_search_no_matches(capsys: pytest.CaptureFixture[str]) -> None:
    """wink docs search handles no matches gracefully."""
    exit_code = wink.main(["docs", "search", "xyznonexistentpatternxyz"])

    assert exit_code == 0
    captured = capsys.readouterr()
    assert "No matches found" in captured.out


def test_docs_search_specs_only(capsys: pytest.CaptureFixture[str]) -> None:
    """--specs flag limits search to specs."""
    exit_code = wink.main(["docs", "search", "Session", "--specs"])

    assert exit_code == 0
    captured = capsys.readouterr()
    # All matches should be from specs
    assert "specs/" in captured.out
    # Should not include guides in results
    lines = captured.out.split("\n")
    for line in lines:
        if line.startswith("guides/"):
            pytest.fail("Search with --specs should not return guide results")


def test_docs_search_guides_only(capsys: pytest.CaptureFixture[str]) -> None:
    """--guides flag limits search to guides."""
    exit_code = wink.main(["docs", "search", "quickstart", "--guides"])

    assert exit_code == 0
    captured = capsys.readouterr()
    # All matches should be from guides
    assert "guides/" in captured.out


def test_docs_search_with_context(capsys: pytest.CaptureFixture[str]) -> None:
    """--context controls number of context lines."""
    exit_code = wink.main(
        ["docs", "search", "Session", "--context", "1", "--max-results", "1"]
    )

    assert exit_code == 0
    captured = capsys.readouterr()
    # Should have limited context
    assert "Found" in captured.out


def test_docs_search_max_results(capsys: pytest.CaptureFixture[str]) -> None:
    """--max-results limits number of results."""
    exit_code = wink.main(["docs", "search", "the", "--max-results", "3"])

    assert exit_code == 0
    captured = capsys.readouterr()
    assert "Found 3 matches" in captured.out


def test_docs_search_regex(capsys: pytest.CaptureFixture[str]) -> None:
    """--regex enables regex pattern matching."""
    exit_code = wink.main(
        ["docs", "search", r"Session\s+", "--regex", "--max-results", "5"]
    )

    assert exit_code == 0
    captured = capsys.readouterr()
    assert "Found" in captured.out


def test_docs_search_invalid_regex(capsys: pytest.CaptureFixture[str]) -> None:
    """Invalid regex returns error."""
    exit_code = wink.main(["docs", "search", "[invalid", "--regex"])

    assert exit_code == 1
    captured = capsys.readouterr()
    assert "Invalid regex" in captured.err


# =============================================================================
# TOC subcommand tests
# =============================================================================


def test_docs_toc_spec(capsys: pytest.CaptureFixture[str]) -> None:
    """wink docs toc spec NAME shows spec headings."""
    exit_code = wink.main(["docs", "toc", "spec", "SESSIONS"])

    assert exit_code == 0
    captured = capsys.readouterr()
    assert "Table of Contents" in captured.out
    assert "specs/SESSIONS.md" in captured.out
    assert "#" in captured.out  # Has markdown headings


def test_docs_toc_guide(capsys: pytest.CaptureFixture[str]) -> None:
    """wink docs toc guide NAME shows guide headings."""
    exit_code = wink.main(["docs", "toc", "guide", "quickstart"])

    assert exit_code == 0
    captured = capsys.readouterr()
    assert "Table of Contents" in captured.out
    assert "guides/quickstart.md" in captured.out


def test_docs_toc_case_insensitive(capsys: pytest.CaptureFixture[str]) -> None:
    """wink docs toc is case-insensitive."""
    exit_code = wink.main(["docs", "toc", "spec", "sessions"])

    assert exit_code == 0
    captured = capsys.readouterr()
    assert "specs/SESSIONS.md" in captured.out


def test_docs_toc_not_found(capsys: pytest.CaptureFixture[str]) -> None:
    """wink docs toc with invalid name returns error."""
    exit_code = wink.main(["docs", "toc", "spec", "NONEXISTENT"])

    assert exit_code == 2
    captured = capsys.readouterr()
    assert "not found" in captured.err


# =============================================================================
# Read subcommand tests
# =============================================================================


def test_docs_read_reference(capsys: pytest.CaptureFixture[str]) -> None:
    """wink docs read reference outputs llms.md."""
    exit_code = wink.main(["docs", "read", "reference"])

    assert exit_code == 0
    captured = capsys.readouterr()
    assert "WINK" in captured.out or "weakincentives" in captured.out.lower()


def test_docs_read_changelog(capsys: pytest.CaptureFixture[str]) -> None:
    """wink docs read changelog outputs CHANGELOG.md."""
    exit_code = wink.main(["docs", "read", "changelog"])

    assert exit_code == 0
    captured = capsys.readouterr()
    assert "# Changelog" in captured.out


def test_docs_read_example(capsys: pytest.CaptureFixture[str]) -> None:
    """wink docs read example outputs formatted example."""
    exit_code = wink.main(["docs", "read", "example"])

    assert exit_code == 0
    captured = capsys.readouterr()
    assert "# Minimal WINK Example" in captured.out
    assert "```python" in captured.out
    assert "PromptTemplate" in captured.out


def test_docs_read_spec(capsys: pytest.CaptureFixture[str]) -> None:
    """wink docs read spec NAME outputs single spec."""
    exit_code = wink.main(["docs", "read", "spec", "ADAPTERS"])

    assert exit_code == 0
    captured = capsys.readouterr()
    assert "<!-- specs/ADAPTERS.md -->" in captured.out


def test_docs_read_spec_with_md_extension(capsys: pytest.CaptureFixture[str]) -> None:
    """wink docs read spec NAME.md works."""
    exit_code = wink.main(["docs", "read", "spec", "ADAPTERS.md"])

    assert exit_code == 0
    captured = capsys.readouterr()
    assert "<!-- specs/ADAPTERS.md -->" in captured.out


def test_docs_read_spec_case_insensitive(capsys: pytest.CaptureFixture[str]) -> None:
    """wink docs read spec is case-insensitive."""
    exit_code = wink.main(["docs", "read", "spec", "adapters"])

    assert exit_code == 0
    captured = capsys.readouterr()
    # Should resolve to the correctly-cased filename
    assert "<!-- specs/ADAPTERS.md -->" in captured.out


def test_docs_read_spec_not_found(capsys: pytest.CaptureFixture[str]) -> None:
    """wink docs read spec with invalid name returns error."""
    exit_code = wink.main(["docs", "read", "spec", "NONEXISTENT"])

    assert exit_code == 2
    captured = capsys.readouterr()
    assert "not found" in captured.err
    assert "Available specs:" in captured.err


def test_docs_read_spec_missing_name(capsys: pytest.CaptureFixture[str]) -> None:
    """wink docs read spec without name returns error."""
    exit_code = wink.main(["docs", "read", "spec"])

    assert exit_code == 1
    captured = capsys.readouterr()
    assert "spec name required" in captured.err


def test_docs_read_guide(capsys: pytest.CaptureFixture[str]) -> None:
    """wink docs read guide NAME outputs single guide."""
    exit_code = wink.main(["docs", "read", "guide", "quickstart"])

    assert exit_code == 0
    captured = capsys.readouterr()
    assert "<!-- guides/quickstart.md -->" in captured.out


def test_docs_read_guide_case_insensitive(capsys: pytest.CaptureFixture[str]) -> None:
    """wink docs read guide is case-insensitive."""
    exit_code = wink.main(["docs", "read", "guide", "Quickstart"])

    assert exit_code == 0
    captured = capsys.readouterr()
    # Should resolve to the correctly-cased filename
    assert "<!-- guides/quickstart.md -->" in captured.out


def test_docs_read_guide_not_found(capsys: pytest.CaptureFixture[str]) -> None:
    """wink docs read guide with invalid name returns error."""
    exit_code = wink.main(["docs", "read", "guide", "NONEXISTENT"])

    assert exit_code == 2
    captured = capsys.readouterr()
    assert "not found" in captured.err
    assert "Available guides:" in captured.err


def test_docs_read_guide_missing_name(capsys: pytest.CaptureFixture[str]) -> None:
    """wink docs read guide without name returns error."""
    exit_code = wink.main(["docs", "read", "guide"])

    assert exit_code == 1
    captured = capsys.readouterr()
    assert "guide name required" in captured.err


# =============================================================================
# No subcommand / help tests
# =============================================================================


def test_docs_no_subcommand_shows_usage(capsys: pytest.CaptureFixture[str]) -> None:
    """wink docs without subcommand shows usage."""
    exit_code = wink.main(["docs"])

    assert exit_code == 1
    captured = capsys.readouterr()
    assert "list" in captured.out
    assert "search" in captured.out
    assert "toc" in captured.out
    assert "read" in captured.out


# =============================================================================
# Internal function tests
# =============================================================================


def test_read_doc_reads_bundled_file() -> None:
    """_read_doc reads from weakincentives.docs package."""
    content = wink._read_doc("llms.md")

    assert isinstance(content, str)
    assert len(content) > 100


def test_read_spec_reads_single_spec() -> None:
    """_read_spec reads a single spec file with header."""
    content = wink._read_spec("ADAPTERS")

    assert isinstance(content, str)
    assert content.startswith("<!-- specs/ADAPTERS.md -->")


def test_read_guide_reads_single_guide() -> None:
    """_read_guide reads a single guide file with header."""
    content = wink._read_guide("quickstart")

    assert isinstance(content, str)
    assert content.startswith("<!-- guides/quickstart.md -->")


def test_list_specs_returns_sorted_names() -> None:
    """_list_specs returns sorted spec names."""
    specs = wink._list_specs()

    assert isinstance(specs, list)
    assert len(specs) > 0
    assert specs == sorted(specs)
    assert "ADAPTERS" in specs


def test_list_guides_returns_sorted_names() -> None:
    """_list_guides returns sorted guide names."""
    guides = wink._list_guides()

    assert isinstance(guides, list)
    assert len(guides) > 0
    assert guides == sorted(guides)
    assert "quickstart" in guides


def test_extract_headings_extracts_markdown_headings() -> None:
    """_extract_headings extracts all markdown headings."""
    content = "# Title\n\nSome text\n\n## Section\n\nMore text\n\n### Subsection\n"
    headings = wink._extract_headings(content)

    assert headings == ["# Title", "## Section", "### Subsection"]


def test_search_docs_returns_matches_with_context() -> None:
    """_search_docs returns matches with context lines."""
    opts = wink.SearchOptions(max_results=5)
    results = wink._search_docs("Session", opts)

    assert isinstance(results, list)
    assert len(results) > 0
    # Each result is (path, line_num, context_lines)
    path, line_num, context = results[0]
    assert isinstance(path, str)
    assert isinstance(line_num, int)
    assert isinstance(context, list)


def test_search_docs_respects_max_results() -> None:
    """_search_docs respects max_results limit."""
    opts = wink.SearchOptions(max_results=3)
    results = wink._search_docs("the", opts)

    assert len(results) <= 3


def test_search_docs_invalid_regex_raises() -> None:
    """_search_docs raises ValueError for invalid regex."""
    opts = wink.SearchOptions(use_regex=True)
    with pytest.raises(ValueError, match="Invalid regex"):
        wink._search_docs("[invalid", opts)


def test_format_doc_list_formats_correctly() -> None:
    """_format_doc_list produces expected format."""
    names = ["FOO", "BARBAZ"]
    descriptions = {"FOO": "Foo description", "BARBAZ": "Bar baz desc"}
    output = wink._format_doc_list(names, descriptions, "TEST")

    assert "TEST (2 documents)" in output
    assert "FOO" in output
    assert "BARBAZ" in output
    assert "Foo description" in output


def test_normalize_doc_name_case_insensitive() -> None:
    """_normalize_doc_name finds names case-insensitively."""
    available = ["ADAPTERS", "SESSIONS", "quickstart"]

    assert wink._normalize_doc_name("ADAPTERS", available) == "ADAPTERS"
    assert wink._normalize_doc_name("adapters", available) == "ADAPTERS"
    assert wink._normalize_doc_name("Adapters", available) == "ADAPTERS"
    assert wink._normalize_doc_name("quickstart", available) == "quickstart"
    assert wink._normalize_doc_name("QUICKSTART", available) == "quickstart"
    assert wink._normalize_doc_name("adapters.md", available) == "ADAPTERS"
    assert wink._normalize_doc_name("NONEXISTENT", available) is None


# =============================================================================
# Metadata synchronization tests
# =============================================================================


def test_spec_metadata_covers_all_spec_files() -> None:
    """SPEC_DESCRIPTIONS must have an entry for every spec file."""
    from pathlib import Path

    from weakincentives.cli.docs_metadata import SPEC_DESCRIPTIONS

    specs_dir = Path(__file__).parent.parent.parent / "specs"
    actual_specs = {p.stem for p in specs_dir.glob("*.md")}
    metadata_specs = set(SPEC_DESCRIPTIONS.keys())

    missing = actual_specs - metadata_specs
    extra = metadata_specs - actual_specs

    assert not missing, f"Specs missing from SPEC_DESCRIPTIONS: {sorted(missing)}"
    assert not extra, (
        f"Extra specs in SPEC_DESCRIPTIONS (files removed?): {sorted(extra)}"
    )


def test_guide_metadata_covers_all_guide_files() -> None:
    """GUIDE_DESCRIPTIONS must have an entry for every guide file."""
    from pathlib import Path

    from weakincentives.cli.docs_metadata import GUIDE_DESCRIPTIONS

    guides_dir = Path(__file__).parent.parent.parent / "guides"
    actual_guides = {p.stem for p in guides_dir.glob("*.md")}
    metadata_guides = set(GUIDE_DESCRIPTIONS.keys())

    missing = actual_guides - metadata_guides
    extra = metadata_guides - actual_guides

    assert not missing, f"Guides missing from GUIDE_DESCRIPTIONS: {sorted(missing)}"
    assert not extra, (
        f"Extra guides in GUIDE_DESCRIPTIONS (files removed?): {sorted(extra)}"
    )
