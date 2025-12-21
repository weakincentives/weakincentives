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

"""Minimal example: structured output from a prompt.

This is the simplest possible WINK example. It demonstrates:
- Defining a structured output type with a dataclass
- Creating a prompt template with a single section
- Evaluating the prompt against an LLM adapter
- Getting typed, validated output

Run with: uv run python examples/progressive/01_minimal_prompt.py
Requires: OPENAI_API_KEY environment variable
"""

from __future__ import annotations

import os
from dataclasses import dataclass

from weakincentives import MarkdownSection, Prompt
from weakincentives.adapters.openai import OpenAIAdapter
from weakincentives.prompt import PromptTemplate
from weakincentives.runtime import Session


# Step 1: Define your structured output as a frozen dataclass
@dataclass(slots=True, frozen=True)
class PRDescription:
    """Structured output for a pull request description."""

    title: str
    summary: str
    changes: list[str]
    breaking: bool


# Step 2: Define parameters for your prompt template
@dataclass(slots=True, frozen=True)
class PRParams:
    """Parameters passed to the prompt at runtime."""

    diff: str


# Step 3: Create a prompt template with sections
#
# The type parameter [PRDescription] tells WINK to validate the LLM's
# response against the PRDescription schema and return a typed result.
template = PromptTemplate[PRDescription](
    ns="examples/progressive",
    key="pr-description",
    name="pr_description_generator",
    sections=(
        MarkdownSection[PRParams](
            title="Instructions",
            template="""
You are a helpful assistant that generates pull request descriptions.

Given the following diff, generate a concise PR description:

```diff
${diff}
```

Respond with a JSON object containing:
- title: A short, descriptive title (max 72 chars)
- summary: One paragraph explaining the change
- changes: A list of specific changes made
- breaking: Whether this is a breaking change
            """,
            key="instructions",
        ),
    ),
)


def main() -> None:
    """Generate a PR description from a sample diff."""
    if "OPENAI_API_KEY" not in os.environ:
        raise SystemExit("Set OPENAI_API_KEY to run this example.")

    # Sample diff to describe
    sample_diff = """\
--- a/utils.py
+++ b/utils.py
@@ -10,6 +10,15 @@ def parse_config(path: str) -> dict:
     return json.load(f)


+def validate_config(config: dict) -> bool:
+    \"\"\"Validate configuration has required fields.\"\"\"
+    required = {"name", "version", "author"}
+    missing = required - set(config.keys())
+    if missing:
+        raise ValueError(f"Missing required fields: {missing}")
+    return True
+
+
 def format_output(data: dict) -> str:
     return json.dumps(data, indent=2)
"""

    # Create adapter and session
    model = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
    adapter = OpenAIAdapter(model=model)
    session = Session()

    # Bind parameters and evaluate
    prompt = Prompt(template).bind(PRParams(diff=sample_diff))
    response = adapter.evaluate(prompt, session=session)

    # Access the typed output
    if response.output is not None:
        pr = response.output
        print(f"Title: {pr.title}")
        print(f"\nSummary: {pr.summary}")
        print("\nChanges:")
        for change in pr.changes:
            print(f"  - {change}")
        print(f"\nBreaking change: {pr.breaking}")
    else:
        print("No structured output received")
        if response.text:
            print(f"Raw response: {response.text}")


if __name__ == "__main__":
    main()
