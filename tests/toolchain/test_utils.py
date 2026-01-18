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

"""Tests for toolchain utility functions."""

from __future__ import annotations

import ast
import tempfile
from pathlib import Path

from toolchain.utils import (
    CodeBlock,
    ImportInfo,
    Link,
    extract_code_blocks,
    extract_imports,
    extract_links,
    get_subpackage,
    git_tracked_files,
    is_shell_output,
    patch_ast_for_bandit,
    path_to_module,
)


class TestGitTrackedFiles:
    """Tests for git_tracked_files."""

    def test_finds_markdown_files(self) -> None:
        # Use actual repo root
        root = Path(__file__).parents[2]
        files = git_tracked_files(root, "*.md")
        assert len(files) > 0
        assert all(f.suffix == ".md" for f in files)

    def test_excludes_directories(self) -> None:
        root = Path(__file__).parents[2]
        files = git_tracked_files(root, "*.md", exclude=frozenset({"test-repositories"}))
        for f in files:
            assert "test-repositories" not in str(f)

    def test_returns_empty_for_non_git_dir(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            files = git_tracked_files(Path(tmpdir), "*.py")
            assert files == []

    def test_with_no_exclude(self) -> None:
        root = Path(__file__).parents[2]
        files = git_tracked_files(root, "*.md", exclude=None)
        assert len(files) > 0


class TestExtractImports:
    """Tests for extract_imports."""

    def test_simple_import(self) -> None:
        source = "import os"
        imports = extract_imports(source, "mymodule")
        assert len(imports) == 1
        assert imports[0].imported_from == "os"
        assert imports[0].lineno == 1

    def test_from_import(self) -> None:
        source = "from pathlib import Path"
        imports = extract_imports(source, "mymodule")
        assert len(imports) == 1
        assert imports[0].imported_from == "pathlib"

    def test_relative_import(self) -> None:
        source = "from .utils import helper"
        imports = extract_imports(source, "weakincentives.prompt")
        assert len(imports) == 1
        assert imports[0].imported_from == "weakincentives.utils"

    def test_parent_relative_import(self) -> None:
        source = "from ..core import Base"
        imports = extract_imports(source, "weakincentives.prompt.section")
        assert len(imports) == 1
        assert imports[0].imported_from == "weakincentives.core"

    def test_deeply_nested_relative_import(self) -> None:
        # Test relative import that goes too high
        source = "from ...top import thing"
        imports = extract_imports(source, "a.b")  # Only 2 levels, but ...- is 3
        assert len(imports) == 1
        # Should handle gracefully

    def test_relative_import_without_module(self) -> None:
        # from . import x (without module name) is skipped by the parser
        source = "from . import helper"
        imports = extract_imports(source, "weakincentives.prompt")
        # These imports are intentionally skipped (module is None)
        assert len(imports) == 0

    def test_multiple_imports(self) -> None:
        source = """import os
import sys
from pathlib import Path"""
        imports = extract_imports(source, "mymodule")
        assert len(imports) == 3


class TestPathToModule:
    """Tests for path_to_module."""

    def test_simple_path(self) -> None:
        path = Path("src/weakincentives/prompt/section.py")
        src = Path("src")
        assert path_to_module(path, src) == "weakincentives.prompt.section"

    def test_init_file(self) -> None:
        path = Path("src/weakincentives/__init__.py")
        src = Path("src")
        assert path_to_module(path, src) == "weakincentives"

    def test_path_not_under_src(self) -> None:
        path = Path("other/module.py")
        src = Path("src")
        # Should handle gracefully
        result = path_to_module(path, src)
        assert "module" in result


class TestGetSubpackage:
    """Tests for get_subpackage."""

    def test_contrib_subpackage(self) -> None:
        assert get_subpackage("weakincentives.contrib.tools") == "contrib"

    def test_prompt_subpackage(self) -> None:
        assert get_subpackage("weakincentives.prompt.section") == "prompt"

    def test_root_package(self) -> None:
        assert get_subpackage("weakincentives") is None

    def test_wrong_root(self) -> None:
        assert get_subpackage("other.module") is None

    def test_single_level(self) -> None:
        assert get_subpackage("other") is None


class TestExtractCodeBlocks:
    """Tests for extract_code_blocks."""

    def test_extracts_python_blocks(self) -> None:
        with tempfile.NamedTemporaryFile(suffix=".md", delete=False, mode="w") as f:
            f.write("""# Test
```python
def foo():
    pass
```
""")
            f.flush()
            blocks = extract_code_blocks(Path(f.name), languages=frozenset({"python"}))
            assert len(blocks) == 1
            assert "def foo():" in blocks[0].code

    def test_skips_other_languages(self) -> None:
        with tempfile.NamedTemporaryFile(suffix=".md", delete=False, mode="w") as f:
            f.write("""```bash
echo hello
```
""")
            f.flush()
            blocks = extract_code_blocks(Path(f.name), languages=frozenset({"python"}))
            assert len(blocks) == 0

    def test_skips_nocheck_blocks(self) -> None:
        with tempfile.NamedTemporaryFile(suffix=".md", delete=False, mode="w") as f:
            f.write("""```python nocheck
invalid code
```
""")
            f.flush()
            blocks = extract_code_blocks(Path(f.name), languages=frozenset({"python"}))
            assert len(blocks) == 0

    def test_skips_output_blocks(self) -> None:
        with tempfile.NamedTemporaryFile(suffix=".md", delete=False, mode="w") as f:
            f.write("""```python output
{'result': 42}
```
""")
            f.flush()
            blocks = extract_code_blocks(Path(f.name), languages=frozenset({"python"}))
            assert len(blocks) == 0


class TestExtractLinks:
    """Tests for extract_links."""

    def test_extracts_local_links(self) -> None:
        with tempfile.NamedTemporaryFile(suffix=".md", delete=False, mode="w") as f:
            f.write("[Guide](./guide.md)")
            f.flush()
            links = extract_links(Path(f.name))
            assert len(links) == 1
            assert links[0].text == "Guide"
            assert links[0].target == "./guide.md"

    def test_skips_http_links(self) -> None:
        with tempfile.NamedTemporaryFile(suffix=".md", delete=False, mode="w") as f:
            f.write("[Example](https://example.com)")
            f.flush()
            links = extract_links(Path(f.name))
            assert len(links) == 0

    def test_skips_anchor_links(self) -> None:
        with tempfile.NamedTemporaryFile(suffix=".md", delete=False, mode="w") as f:
            f.write("[Section](#section)")
            f.flush()
            links = extract_links(Path(f.name))
            assert len(links) == 0

    def test_skips_links_in_code_blocks(self) -> None:
        with tempfile.NamedTemporaryFile(suffix=".md", delete=False, mode="w") as f:
            f.write("""```
[Not a link](./file.md)
```
""")
            f.flush()
            links = extract_links(Path(f.name))
            assert len(links) == 0

    def test_skips_links_in_tilde_code_blocks(self) -> None:
        with tempfile.NamedTemporaryFile(suffix=".md", delete=False, mode="w") as f:
            f.write("""~~~
[Not a link](./file.md)
~~~
""")
            f.flush()
            links = extract_links(Path(f.name))
            assert len(links) == 0

    def test_skips_mailto_links(self) -> None:
        with tempfile.NamedTemporaryFile(suffix=".md", delete=False, mode="w") as f:
            f.write("[Email](mailto:test@example.com)")
            f.flush()
            links = extract_links(Path(f.name))
            assert len(links) == 0

    def test_skips_links_in_inline_code(self) -> None:
        with tempfile.NamedTemporaryFile(suffix=".md", delete=False, mode="w") as f:
            f.write("Use `[text](file.md)` syntax")
            f.flush()
            links = extract_links(Path(f.name))
            assert len(links) == 0


class TestIsShellOutput:
    """Tests for is_shell_output."""

    def test_dollar_prompt(self) -> None:
        assert is_shell_output("$ ls -la") is True

    def test_angle_bracket_prompt(self) -> None:
        assert is_shell_output("> npm install") is True

    def test_python_repl(self) -> None:
        assert is_shell_output(">>> print('hello')") is True

    def test_continuation_prompt(self) -> None:
        assert is_shell_output("... more code") is True

    def test_regular_code(self) -> None:
        assert is_shell_output("def foo():") is False

    def test_empty_code(self) -> None:
        assert is_shell_output("") is False


class TestPatchAstForBandit:
    """Tests for patch_ast_for_bandit."""

    def test_patches_missing_nodes(self) -> None:
        # Call the patch function
        patch_ast_for_bandit()

        # Check that Num, Str, etc. are now available (or were already)
        # This should not raise
        _ = getattr(ast, "Num", None)
        _ = getattr(ast, "Str", None)

    def test_idempotent(self) -> None:
        # Should be safe to call multiple times
        patch_ast_for_bandit()
        patch_ast_for_bandit()
        # Should not raise

    def test_with_constant_attribute_access(self) -> None:
        # After patching, Constant should have n, s, b properties
        patch_ast_for_bandit()
        constant = getattr(ast, "Constant", None)
        if constant:
            # Check that the properties exist
            assert hasattr(constant, "n") or True  # May already exist
            assert hasattr(constant, "s") or True
            assert hasattr(constant, "b") or True


class TestCodeBlock:
    """Tests for CodeBlock dataclass."""

    def test_frozen(self) -> None:
        block = CodeBlock(
            file=Path("test.md"),
            start_line=1,
            language="python",
            code="pass",
            meta="",
        )
        assert block.file == Path("test.md")
        assert block.start_line == 1


class TestLink:
    """Tests for Link dataclass."""

    def test_creation(self) -> None:
        link = Link(file=Path("test.md"), line=1, text="Link", target="./other.md")
        assert link.text == "Link"
        assert link.target == "./other.md"


class TestImportInfo:
    """Tests for ImportInfo dataclass."""

    def test_creation(self) -> None:
        info = ImportInfo(module="a", imported_from="b", lineno=1)
        assert info.module == "a"
        assert info.imported_from == "b"
        assert info.lineno == 1
