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

"""Example with tools: give the LLM callable capabilities.

Building on 01_minimal_prompt.py, this example adds tools that the LLM
can invoke during evaluation. It demonstrates:
- Defining tool parameters and results as dataclasses
- Creating tool handlers with the (params, *, context) signature
- Attaching tools to sections
- The LLM calling tools and receiving results

Run with: uv run python examples/progressive/02_with_tools.py
Requires: OPENAI_API_KEY environment variable
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field

from weakincentives import MarkdownSection, Prompt, Tool, ToolContext, ToolResult
from weakincentives.adapters.openai import OpenAIAdapter
from weakincentives.prompt import PromptTemplate
from weakincentives.runtime import Session

# --- Structured Output ---


@dataclass(slots=True, frozen=True)
class AnalysisResult:
    """The final analysis from the assistant."""

    summary: str
    statistics: dict[str, int]
    recommendation: str


# --- Tool Definitions ---


@dataclass(slots=True, frozen=True)
class WordCountParams:
    """Parameters for the word_count tool."""

    text: str = field(metadata={"description": "The text to count words in."})


@dataclass(slots=True, frozen=True)
class WordCountResult:
    """Result from the word_count tool."""

    word_count: int
    char_count: int
    line_count: int


def word_count_handler(
    params: WordCountParams, *, context: ToolContext
) -> ToolResult[WordCountResult]:
    """Count words, characters, and lines in text."""
    del context  # Unused in this simple handler
    text = params.text
    result = WordCountResult(
        word_count=len(text.split()),
        char_count=len(text),
        line_count=text.count("\n") + 1 if text else 0,
    )
    return ToolResult(
        message=f"Counted {result.word_count} words, {result.char_count} characters, {result.line_count} lines.",
        value=result,
        success=True,
    )


word_count_tool: Tool[WordCountParams, WordCountResult] = Tool(
    name="word_count",
    description="Count words, characters, and lines in a piece of text.",
    handler=word_count_handler,
)


@dataclass(slots=True, frozen=True)
class SentimentParams:
    """Parameters for the analyze_sentiment tool."""

    text: str = field(metadata={"description": "The text to analyze."})


@dataclass(slots=True, frozen=True)
class SentimentResult:
    """Result from the analyze_sentiment tool."""

    sentiment: str  # "positive", "negative", or "neutral"
    confidence: float


def sentiment_handler(
    params: SentimentParams, *, context: ToolContext
) -> ToolResult[SentimentResult]:
    """Simple keyword-based sentiment analysis."""
    del context
    text = params.text.lower()

    positive_words = {"good", "great", "excellent", "happy", "love", "wonderful"}
    negative_words = {"bad", "terrible", "awful", "sad", "hate", "horrible"}

    words = set(text.split())
    pos_count = len(words & positive_words)
    neg_count = len(words & negative_words)
    total = pos_count + neg_count

    if total == 0:
        sentiment, confidence = "neutral", 0.5
    elif pos_count > neg_count:
        sentiment, confidence = "positive", pos_count / total
    else:
        sentiment, confidence = "negative", neg_count / total

    result = SentimentResult(sentiment=sentiment, confidence=round(confidence, 2))
    return ToolResult(
        message=f"Sentiment: {result.sentiment} (confidence: {result.confidence})",
        value=result,
        success=True,
    )


sentiment_tool: Tool[SentimentParams, SentimentResult] = Tool(
    name="analyze_sentiment",
    description="Analyze the sentiment of text (positive, negative, or neutral).",
    handler=sentiment_handler,
)


# --- Prompt Parameters ---


@dataclass(slots=True, frozen=True)
class AnalysisParams:
    """Parameters for the analysis prompt."""

    content: str


# --- Prompt Template ---

# Tools are attached to sections via the `tools` parameter.
# The LLM sees the tool schemas and can call them during evaluation.

template = PromptTemplate[AnalysisResult](
    ns="examples/progressive",
    key="text-analysis",
    name="text_analyzer",
    sections=(
        MarkdownSection[AnalysisParams](
            title="Instructions",
            template="""
You are a text analysis assistant. Use the available tools to analyze
the following content:

---
${content}
---

Use the tools to:
1. Count the words, characters, and lines
2. Analyze the overall sentiment

Then provide your analysis as JSON with:
- summary: A brief description of the content
- statistics: Dictionary with word_count, char_count, line_count, sentiment
- recommendation: What the author could improve
            """,
            key="instructions",
            tools=(word_count_tool, sentiment_tool),
        ),
    ),
)


def main() -> None:
    """Analyze a piece of text using tools."""
    if "OPENAI_API_KEY" not in os.environ:
        raise SystemExit("Set OPENAI_API_KEY to run this example.")

    sample_text = """\
I had a wonderful experience at the restaurant yesterday.
The food was excellent and the service was great.
I would love to go back again soon.
The only issue was the parking, which was terrible.
"""

    model = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
    adapter = OpenAIAdapter(model=model)
    session = Session()

    prompt = Prompt(template).bind(AnalysisParams(content=sample_text))
    response = adapter.evaluate(prompt, session=session)

    if response.output is not None:
        result = response.output
        print(f"Summary: {result.summary}")
        print("\nStatistics:")
        for key, value in result.statistics.items():
            print(f"  {key}: {value}")
        print(f"\nRecommendation: {result.recommendation}")
    else:
        print("No structured output received")
        if response.text:
            print(f"Raw response: {response.text}")


if __name__ == "__main__":
    main()
