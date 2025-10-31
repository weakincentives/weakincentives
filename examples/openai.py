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

from weakincentives.adapters import OpenAIAdapter

from .common import CodeReviewSession


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
