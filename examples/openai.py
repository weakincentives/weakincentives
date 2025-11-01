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

"""OpenAI-powered example that behaves as a basic code review agent."""

from __future__ import annotations

import os
import sys
from pathlib import Path

from weakincentives.adapters import OpenAIAdapter

_SCRIPT_DIR = Path(__file__).resolve().parent


def _remove_script_dir_from_path() -> None:
    script_dir = _SCRIPT_DIR
    script_dir_parent = script_dir.parent
    resolved_script_dir = script_dir.resolve()
    for index in range(len(sys.path) - 1, -1, -1):
        entry = sys.path[index]
        if not entry:
            try:
                cwd = Path.cwd().resolve()
            except (OSError, RuntimeError):
                continue
            if cwd == resolved_script_dir:
                sys.path[index] = str(script_dir_parent)
            continue
        try:
            entry_path = Path(entry).resolve()
        except (OSError, RuntimeError):
            continue
        if entry_path == resolved_script_dir:
            del sys.path[index]


if __package__ is None:  # pragma: no cover - script execution path
    _PROJECT_ROOT = _SCRIPT_DIR.parent
    _PROJECT_ROOT_STR = str(_PROJECT_ROOT)
    _INSERTED = False
    if _PROJECT_ROOT_STR not in sys.path:
        sys.path.insert(0, _PROJECT_ROOT_STR)
        _INSERTED = True
    try:
        from examples.common import CodeReviewSession  # type: ignore[import]
    finally:
        if _INSERTED:
            sys.path.pop(0)
else:
    from .common import CodeReviewSession


_remove_script_dir_from_path()


def main() -> None:
    if "OPENAI_API_KEY" not in os.environ:
        raise SystemExit("Set OPENAI_API_KEY before running this example.")

    model = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
    session = CodeReviewSession(OpenAIAdapter(model=model))
    print("Type 'exit' or 'quit' to stop the conversation.")
    print("Type 'history' to display the recorded tool call summary.")
    while True:
        try:
            prompt = input("Review prompt: ").strip()
        except EOFError:  # pragma: no cover - interactive convenience
            break
        if not prompt:
            print("Goodbye.")
            break
        if prompt.lower() in {"exit", "quit"}:
            break
        if prompt.lower() == "history":
            print(session.render_tool_history())
            continue
        answer = session.evaluate(prompt)
        print(f"Agent: {answer}")


if __name__ == "__main__":
    main()
