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

"""Tests for markdown utilities."""

from __future__ import annotations

from pathlib import Path

from weakincentives.verify._markdown import (
    extract_code_blocks,
    extract_file_path,
    extract_links,
    is_python_code_block,
    is_shell_output,
)


class TestExtractCodeBlocks:
    """Tests for extract_code_blocks function."""

    def test_simple_python_block(self, tmp_path: Path) -> None:
        """Extract a simple Python code block."""
        md_file = tmp_path / "test.md"
        md_file.write_text("""\
# Test

```python
print("hello")
```
""")
        blocks = extract_code_blocks(md_file)
        assert len(blocks) == 1
        assert blocks[0].language == "python"
        assert 'print("hello")' in blocks[0].code

    def test_multiple_blocks(self, tmp_path: Path) -> None:
        """Extract multiple code blocks."""
        md_file = tmp_path / "test.md"
        md_file.write_text("""\
# Test

```python
code1()
```

Some text.

```python
code2()
```
""")
        blocks = extract_code_blocks(md_file)
        assert len(blocks) == 2
        assert "code1" in blocks[0].code
        assert "code2" in blocks[1].code

    def test_skip_marked_blocks(self, tmp_path: Path) -> None:
        """Skip blocks with skip markers."""
        md_file = tmp_path / "test.md"
        md_file.write_text("""\
```python nocheck
skipped()
```

```python
included()
```
""")
        blocks = extract_code_blocks(md_file)
        assert len(blocks) == 1
        assert "included" in blocks[0].code

    def test_filter_by_language(self, tmp_path: Path) -> None:
        """Filter blocks by language."""
        md_file = tmp_path / "test.md"
        md_file.write_text("""\
```python
python_code()
```

```javascript
js_code()
```

```bash
bash_code
```
""")
        # Default filters to python/py
        blocks = extract_code_blocks(md_file, languages=frozenset({"python", "py"}))
        assert len(blocks) == 1
        assert "python_code" in blocks[0].code

    def test_line_numbers(self, tmp_path: Path) -> None:
        """Track correct line numbers."""
        md_file = tmp_path / "test.md"
        md_file.write_text("""\
Line 1
Line 2
Line 3
```python
code()
```
""")
        blocks = extract_code_blocks(md_file)
        assert len(blocks) == 1
        assert blocks[0].start_line == 5  # Line after fence


class TestExtractLinks:
    """Tests for extract_links function."""

    def test_simple_link(self, tmp_path: Path) -> None:
        """Extract a simple link."""
        md_file = tmp_path / "test.md"
        md_file.write_text("Check out [Google](https://google.com).")

        links = extract_links(md_file)
        assert len(links) == 1
        assert links[0].text == "Google"
        assert links[0].target == "https://google.com"
        assert links[0].is_local is False

    def test_local_link(self, tmp_path: Path) -> None:
        """Extract a local link."""
        md_file = tmp_path / "test.md"
        md_file.write_text("See [docs](./docs/README.md).")

        links = extract_links(md_file)
        assert len(links) == 1
        assert links[0].is_local is True

    def test_anchor_link(self, tmp_path: Path) -> None:
        """Extract anchor links."""
        md_file = tmp_path / "test.md"
        md_file.write_text("Jump to [section](#section-name).")

        links = extract_links(md_file)
        assert len(links) == 1
        assert links[0].is_local is False  # Pure anchors aren't local file links

    def test_skip_links_in_code(self, tmp_path: Path) -> None:
        """Skip links inside code blocks."""
        md_file = tmp_path / "test.md"
        md_file.write_text("""\
```python
# [not a link](not/a/path.md)
```

[real link](real.md)
""")
        links = extract_links(md_file)
        assert len(links) == 1
        assert links[0].text == "real link"

    def test_skip_links_in_inline_code(self, tmp_path: Path) -> None:
        """Skip links inside inline code."""
        md_file = tmp_path / "test.md"
        md_file.write_text("Use `[link](path)` syntax. [real](real.md)")

        links = extract_links(md_file)
        assert len(links) == 1
        assert links[0].text == "real"


class TestExtractFilePath:
    """Tests for extract_file_path function."""

    def test_simple_path(self) -> None:
        """Extract simple file path."""
        result = extract_file_path("docs/README.md")
        assert result == "docs/README.md"

    def test_path_with_anchor(self) -> None:
        """Extract path from link with anchor."""
        result = extract_file_path("docs/README.md#section")
        assert result == "docs/README.md"

    def test_anchor_only(self) -> None:
        """Handle anchor-only links."""
        result = extract_file_path("#section")
        assert result == ""


class TestIsPythonCodeBlock:
    """Tests for is_python_code_block function."""

    def test_python_language(self, tmp_path: Path) -> None:
        """Detect Python code blocks."""
        md_file = tmp_path / "test.md"
        md_file.write_text("```python\ncode()\n```")

        blocks = extract_code_blocks(md_file, languages=None)
        assert len(blocks) == 1
        assert is_python_code_block(blocks[0]) is True

    def test_py_language(self, tmp_path: Path) -> None:
        """Detect py code blocks."""
        md_file = tmp_path / "test.md"
        md_file.write_text("```py\ncode()\n```")

        blocks = extract_code_blocks(md_file, languages=None)
        assert len(blocks) == 1
        assert is_python_code_block(blocks[0]) is True

    def test_other_language(self, tmp_path: Path) -> None:
        """Detect non-Python code blocks."""
        md_file = tmp_path / "test.md"
        md_file.write_text("```javascript\ncode()\n```")

        blocks = extract_code_blocks(md_file, languages=None)
        assert len(blocks) == 1
        assert is_python_code_block(blocks[0]) is False


class TestIsShellOutput:
    """Tests for is_shell_output function."""

    def test_dollar_prefix(self) -> None:
        """Detect shell commands with $ prefix."""
        assert is_shell_output("$ echo hello") is True

    def test_chevron_prefix(self) -> None:
        """Detect shell with > prefix."""
        assert is_shell_output("> command") is True

    def test_python_repl(self) -> None:
        """Detect Python REPL output."""
        assert is_shell_output(">>> print('hello')") is True
        assert is_shell_output("... continuation") is True

    def test_normal_code(self) -> None:
        """Normal code is not shell output."""
        assert is_shell_output("print('hello')") is False
        assert is_shell_output("def function():") is False
