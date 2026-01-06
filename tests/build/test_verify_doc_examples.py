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

"""Tests for build/verify_doc_examples.py."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

# Add build directory to path so we can import the module
_BUILD_DIR = Path(__file__).resolve().parent.parent.parent / "build"
sys.path.insert(0, str(_BUILD_DIR))

from verify_doc_examples import (  # noqa: E402
    API_DEFAULTS_SIGNATURE,
    API_RETURN_SIGNATURE,
    FENCE_PATTERN,
    CodeBlock,
    Diagnostic,
    _is_api_reference_block,
    _is_doc_noise,
    extract_python_blocks,
    find_undefined_names,
    map_diagnostic_to_source,
    preprocess_code,
    synthesize_module,
)


class TestFencePattern:
    """Tests for the FENCE_PATTERN regex."""

    def test_matches_python_block(self) -> None:
        """Matches basic python code block."""
        content = "```python\nprint('hello')\n```"
        match = FENCE_PATTERN.search(content)
        assert match is not None
        assert match.group("lang") == "python"
        assert match.group("code") == "print('hello')"

    def test_matches_py_shorthand(self) -> None:
        """Matches 'py' language marker."""
        content = "```py\nx = 1\n```"
        match = FENCE_PATTERN.search(content)
        assert match is not None
        assert match.group("lang") == "py"

    def test_captures_metadata(self) -> None:
        """Captures metadata after language marker."""
        content = "```python nocheck\ncode\n```"
        match = FENCE_PATTERN.search(content)
        assert match is not None
        assert match.group("meta") == " nocheck"

    def test_handles_indented_blocks(self) -> None:
        """Matches indented code blocks (e.g., in lists)."""
        content = "    ```python\n    x = 1\n    ```"
        match = FENCE_PATTERN.search(content)
        assert match is not None
        assert match.group("indent") == "    "
        assert match.group("code") == "    x = 1"

    def test_multiline_code(self) -> None:
        """Matches multi-line code blocks."""
        content = "```python\nline1\nline2\nline3\n```"
        match = FENCE_PATTERN.search(content)
        assert match is not None
        assert match.group("code") == "line1\nline2\nline3"

    def test_does_not_match_other_languages(self) -> None:
        """Does not match non-Python blocks."""
        content = "```javascript\nconsole.log('hi')\n```"
        match = FENCE_PATTERN.search(content)
        assert match is None

    def test_does_not_match_bare_fence(self) -> None:
        """Does not match fence without language marker."""
        content = "```\ncode\n```"
        match = FENCE_PATTERN.search(content)
        assert match is None


class TestApiSignaturePatterns:
    """Tests for API signature regex patterns."""

    def test_return_signature_matches(self) -> None:
        """API_RETURN_SIGNATURE matches func() -> Type."""
        assert API_RETURN_SIGNATURE.match("func(a, b) -> ReturnType")
        assert API_RETURN_SIGNATURE.match("Class[T](param) -> None")

    def test_return_signature_no_match_bare_call(self) -> None:
        """API_RETURN_SIGNATURE does not match bare calls."""
        assert not API_RETURN_SIGNATURE.match("print('hello')")
        assert not API_RETURN_SIGNATURE.match("MyClass()")

    def test_defaults_signature_matches(self) -> None:
        """API_DEFAULTS_SIGNATURE matches func(arg=None)."""
        assert API_DEFAULTS_SIGNATURE.match("Class(param=None)")
        assert API_DEFAULTS_SIGNATURE.match("func(a=None, b=...)")
        assert API_DEFAULTS_SIGNATURE.match("Template[T](ns, key, name=None)")

    def test_defaults_signature_no_match_assignment(self) -> None:
        """API_DEFAULTS_SIGNATURE does not match assignments."""
        assert not API_DEFAULTS_SIGNATURE.match("x = func()")
        assert not API_DEFAULTS_SIGNATURE.match("result = Class(param)")


class TestIsApiReferenceBlock:
    """Tests for _is_api_reference_block function."""

    def test_detects_function_signatures(self) -> None:
        """Detects blocks with function signatures (must have -> annotation)."""
        code = "func(arg1, arg2) -> ReturnType\nother_func() -> None"
        assert _is_api_reference_block(code) is True

    def test_detects_default_value_signatures(self) -> None:
        """Detects blocks with default value documentation."""
        code = "Class(param=None, other=...)\nFunc(a=None)"
        assert _is_api_reference_block(code) is True

    def test_does_not_detect_bare_calls(self) -> None:
        """Does not detect bare function calls as signatures."""
        code = "MyClass(param1, param2)\nOtherClass()"
        assert _is_api_reference_block(code) is False

    def test_detects_method_chains(self) -> None:
        """Detects method chain documentation."""
        code = ".method1() / .method2()\n.attr"
        assert _is_api_reference_block(code) is True

    def test_allows_actual_code(self) -> None:
        """Does not flag actual executable code."""
        code = "x = MyClass()\nresult = x.method()"
        assert _is_api_reference_block(code) is False

    def test_allows_imports(self) -> None:
        """Does not flag import statements."""
        code = "from module import func\nimport other"
        assert _is_api_reference_block(code) is False

    def test_ignores_comments(self) -> None:
        """Ignores comment lines in classification."""
        code = "# This is a comment\nfunc(args) -> Type"
        assert _is_api_reference_block(code) is True

    def test_empty_block(self) -> None:
        """Handles empty blocks."""
        assert _is_api_reference_block("") is False
        assert _is_api_reference_block("   \n   ") is False

    def test_mixed_content_below_threshold(self) -> None:
        """Mixed content below 70% signature threshold is not API ref."""
        code = "x = 1\ny = 2\nfunc() -> Type"  # 1/3 = 33% signatures
        assert _is_api_reference_block(code) is False


class TestExtractPythonBlocks:
    """Tests for extract_python_blocks function."""

    @pytest.fixture
    def tmp_md_file(self, tmp_path: Path) -> Path:
        """Create a temporary markdown file."""
        return tmp_path / "test.md"

    def test_extracts_single_block(self, tmp_md_file: Path) -> None:
        """Extracts a single Python block."""
        tmp_md_file.write_text("# Header\n\n```python\nprint('hello')\n```\n")
        blocks = extract_python_blocks(tmp_md_file)
        assert len(blocks) == 1
        assert blocks[0].code == "print('hello')"
        assert blocks[0].file == tmp_md_file

    def test_extracts_multiple_blocks(self, tmp_md_file: Path) -> None:
        """Extracts multiple Python blocks."""
        tmp_md_file.write_text("```python\nblock1\n```\n\n```python\nblock2\n```\n")
        blocks = extract_python_blocks(tmp_md_file)
        assert len(blocks) == 2
        assert blocks[0].code == "block1"
        assert blocks[1].code == "block2"

    def test_skips_nocheck_marker(self, tmp_md_file: Path) -> None:
        """Skips blocks with nocheck marker."""
        tmp_md_file.write_text(
            "```python nocheck\nskipped\n```\n\n```python\nkept\n```\n"
        )
        blocks = extract_python_blocks(tmp_md_file)
        assert len(blocks) == 1
        assert blocks[0].code == "kept"

    def test_skips_shell_output(self, tmp_md_file: Path) -> None:
        """Skips blocks starting with shell prompt."""
        tmp_md_file.write_text(
            "```python\n$ pip install pkg\n```\n\n```python\nkept\n```\n"
        )
        blocks = extract_python_blocks(tmp_md_file)
        assert len(blocks) == 1
        assert blocks[0].code == "kept"

    def test_skips_api_reference_blocks(self, tmp_md_file: Path) -> None:
        """Skips API reference documentation blocks."""
        tmp_md_file.write_text(
            "```python\nfunc(a, b) -> Type\nother() -> None\n```\n\n"
            "```python\nx = 1\n```\n"
        )
        blocks = extract_python_blocks(tmp_md_file)
        assert len(blocks) == 1
        assert blocks[0].code == "x = 1"

    def test_correct_line_numbers(self, tmp_md_file: Path) -> None:
        """Calculates correct line numbers."""
        tmp_md_file.write_text("line1\nline2\n```python\ncode\n```\n")
        blocks = extract_python_blocks(tmp_md_file)
        assert len(blocks) == 1
        # Line 3 is ```python, line 4 is code
        assert blocks[0].start_line == 4


class TestFindUndefinedNames:
    """Tests for find_undefined_names function."""

    def test_detects_undefined_variable(self) -> None:
        """Detects undefined variable usage."""
        code = "x = undefined_var"
        undefined = find_undefined_names(code)
        assert "undefined_var" in undefined

    def test_ignores_defined_variable(self) -> None:
        """Ignores variables that are defined."""
        code = "x = 1\ny = x"
        undefined = find_undefined_names(code)
        assert "x" not in undefined
        assert "y" not in undefined

    def test_ignores_builtins(self) -> None:
        """Ignores builtin names."""
        code = "x = len([1, 2, 3])"
        undefined = find_undefined_names(code)
        assert "len" not in undefined

    def test_handles_imports(self) -> None:
        """Handles import statements."""
        code = "import os\nx = os.path"
        undefined = find_undefined_names(code)
        assert "os" not in undefined

    def test_handles_from_imports(self) -> None:
        """Handles from imports."""
        code = "from pathlib import Path\np = Path('.')"
        undefined = find_undefined_names(code)
        assert "Path" not in undefined

    def test_handles_function_args(self) -> None:
        """Handles function arguments."""
        code = "def f(x, y):\n    return x + y"
        undefined = find_undefined_names(code)
        assert "x" not in undefined
        assert "y" not in undefined

    def test_handles_for_loop_target(self) -> None:
        """Handles for loop targets."""
        code = "for item in items:\n    print(item)"
        undefined = find_undefined_names(code)
        assert "item" not in undefined
        assert "items" in undefined

    def test_handles_with_statement(self) -> None:
        """Handles with statement targets."""
        code = "with open('f') as f:\n    x = f.read()"
        undefined = find_undefined_names(code)
        assert "f" not in undefined

    def test_handles_class_definition(self) -> None:
        """Handles class definitions."""
        code = "class Foo:\n    pass\nx = Foo()"
        undefined = find_undefined_names(code)
        assert "Foo" not in undefined

    def test_handles_syntax_error(self) -> None:
        """Returns empty set for invalid syntax."""
        code = "def f(:\n    pass"
        undefined = find_undefined_names(code)
        assert undefined == set()


class TestPreprocessCode:
    """Tests for preprocess_code function."""

    def test_replaces_standalone_ellipsis(self) -> None:
        """Replaces standalone ... with pass."""
        code = "def f():\n    ..."
        result = preprocess_code(code)
        assert "pass" in result
        assert "..." not in result

    def test_preserves_indentation(self) -> None:
        """Preserves indentation when replacing ellipsis."""
        code = "class C:\n    def m(self):\n        ..."
        result = preprocess_code(code)
        lines = result.split("\n")
        assert lines[2] == "        pass"

    def test_preserves_ellipsis_in_expression(self) -> None:
        """Preserves ellipsis when part of expression."""
        code = "x = ..."
        result = preprocess_code(code)
        assert "x = ..." in result

    def test_preserves_normal_code(self) -> None:
        """Preserves normal code unchanged."""
        code = "x = 1\ny = 2"
        result = preprocess_code(code)
        assert result == code


class TestSynthesizeModule:
    """Tests for synthesize_module function."""

    def test_includes_common_stubs(self) -> None:
        """Includes common stubs at top of module."""
        blocks = [
            CodeBlock(
                file=Path("test.md"),
                start_line=1,
                end_line=2,
                code="x = 1",
                meta="",
            )
        ]
        result = synthesize_module(blocks)
        assert "from __future__ import annotations" in result
        assert "from typing import Any" in result

    def test_adds_source_comments(self) -> None:
        """Adds source comments before each block."""
        blocks = [
            CodeBlock(
                file=Path("test.md"),
                start_line=10,
                end_line=12,
                code="x = 1",
                meta="",
            )
        ]
        result = synthesize_module(blocks)
        assert "# Source: test.md:10" in result

    def test_adds_undefined_name_stubs(self) -> None:
        """Adds stubs for undefined names."""
        blocks = [
            CodeBlock(
                file=Path("test.md"),
                start_line=1,
                end_line=2,
                code="x = my_custom_var",
                meta="",
            )
        ]
        result = synthesize_module(blocks)
        assert "my_custom_var: Any = None" in result

    def test_preprocesses_code(self) -> None:
        """Preprocesses code blocks (ellipsis -> pass)."""
        blocks = [
            CodeBlock(
                file=Path("test.md"),
                start_line=1,
                end_line=2,
                code="def f():\n    ...",
                meta="",
            )
        ]
        result = synthesize_module(blocks)
        assert "pass" in result


class TestIsDocNoise:
    """Tests for _is_doc_noise function."""

    def test_filters_variable_not_allowed(self) -> None:
        """Filters 'Variable not allowed in type expression'."""
        assert _is_doc_noise("Variable not allowed in type expression") is True

    def test_filters_obscured_declaration(self) -> None:
        """Filters obscured declaration messages."""
        msg = '"Foo" is obscured by a declaration of the same name'
        assert _is_doc_noise(msg) is True

    def test_filters_ellipsis_type(self) -> None:
        """Filters EllipsisType messages."""
        assert _is_doc_noise("EllipsisType cannot be assigned") is True

    def test_filters_unresolved(self) -> None:
        """Filters unresolved import messages."""
        assert _is_doc_noise("Import 'foo' could not be resolved") is True

    def test_filters_indentation(self) -> None:
        """Filters indentation error messages."""
        assert _is_doc_noise("Unexpected indentation") is True
        assert _is_doc_noise("Unindent not expected") is True

    def test_filters_return_value(self) -> None:
        """Filters return value messages."""
        assert _is_doc_noise("must return value on all code paths") is True

    def test_filters_unused_expression(self) -> None:
        """Filters unused expression warnings."""
        assert _is_doc_noise("Expression value is unused") is True

    def test_filters_stub_type_issues(self) -> None:
        """Filters Any type issues from stubs."""
        assert _is_doc_noise('"x" is not defined but Any is') is True

    def test_filters_doc_examples_assignability(self) -> None:
        """Filters _doc_examples type assignability."""
        msg = 'Type is not assignable to declared type "_doc_examples.Foo"'
        assert _is_doc_noise(msg) is True

    def test_allows_real_errors(self) -> None:
        """Allows real error messages through."""
        assert _is_doc_noise("Cannot access attribute 'foo'") is False
        assert _is_doc_noise("Unknown import symbol") is False


class TestMapDiagnosticToSource:
    """Tests for map_diagnostic_to_source function."""

    def test_maps_diagnostic_to_source(self) -> None:
        """Maps diagnostic line to source file."""
        blocks = [
            CodeBlock(
                file=Path("test.md"),
                start_line=10,
                end_line=12,
                code="line1\nline2",
                meta="",
            )
        ]
        module_content = "stubs\n# Source: test.md:10\nline1\nline2"
        diag = Diagnostic(
            file=Path("_doc_examples.py"),
            line=3,  # line1
            column=0,
            severity="error",
            message="test",
        )
        result = map_diagnostic_to_source(diag, blocks, module_content)
        assert result is not None
        assert result[0] == Path("test.md")
        assert result[1] == 10  # First line of block

    def test_handles_offset_within_block(self) -> None:
        """Handles diagnostic on later line within block."""
        blocks = [
            CodeBlock(
                file=Path("test.md"),
                start_line=10,
                end_line=13,
                code="line1\nline2\nline3",
                meta="",
            )
        ]
        module_content = "# Source: test.md:10\nline1\nline2\nline3"
        diag = Diagnostic(
            file=Path("_doc_examples.py"),
            line=3,  # line2
            column=0,
            severity="error",
            message="test",
        )
        result = map_diagnostic_to_source(diag, blocks, module_content)
        assert result is not None
        assert result[1] == 11  # Second line of block

    def test_returns_none_for_stub_area(self) -> None:
        """Returns None for diagnostics in stub area."""
        blocks = [
            CodeBlock(
                file=Path("test.md"),
                start_line=10,
                end_line=12,
                code="code",
                meta="",
            )
        ]
        module_content = "stubs\nmore_stubs\n# Source: test.md:10\ncode"
        diag = Diagnostic(
            file=Path("_doc_examples.py"),
            line=1,  # In stubs area
            column=0,
            severity="error",
            message="test",
        )
        result = map_diagnostic_to_source(diag, blocks, module_content)
        assert result is None

    def test_returns_none_for_invalid_line(self) -> None:
        """Returns None for out-of-bounds line."""
        blocks: list[CodeBlock] = []
        module_content = "line1"
        diag = Diagnostic(
            file=Path("_doc_examples.py"),
            line=100,
            column=0,
            severity="error",
            message="test",
        )
        result = map_diagnostic_to_source(diag, blocks, module_content)
        assert result is None


class TestCodeBlockDataclass:
    """Tests for CodeBlock dataclass."""

    def test_frozen(self) -> None:
        """CodeBlock is immutable."""
        block = CodeBlock(
            file=Path("test.md"),
            start_line=1,
            end_line=2,
            code="x = 1",
            meta="",
        )
        with pytest.raises(AttributeError):
            block.code = "y = 2"  # type: ignore[misc]

    def test_equality(self) -> None:
        """CodeBlock equality works."""
        b1 = CodeBlock(Path("t.md"), 1, 2, "x", "")
        b2 = CodeBlock(Path("t.md"), 1, 2, "x", "")
        assert b1 == b2


class TestDiagnosticDataclass:
    """Tests for Diagnostic dataclass."""

    def test_frozen(self) -> None:
        """Diagnostic is immutable."""
        diag = Diagnostic(Path("f.py"), 1, 0, "error", "msg")
        with pytest.raises(AttributeError):
            diag.message = "new"  # type: ignore[misc]
