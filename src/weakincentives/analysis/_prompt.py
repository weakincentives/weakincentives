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

"""Prebuilt analysis prompt template.

Provides a default prompt for the AnalysisLoop that guides the analysis
agent through structured investigation of debug bundles. Sections can
be overridden via AnalysisPromptOverrides for different analysis styles.
"""

from __future__ import annotations

from dataclasses import dataclass

from ..prompt import MarkdownSection, Prompt, PromptTemplate
from ._types import AnalysisBundle, AnalysisPromptOverrides, AnalysisRequest

_DEFAULT_METHODOLOGY = """\
You are an expert agent debugger. Your task is to analyze debug bundles
from agent executions and produce actionable insights.

**Approach:**
1. Start by understanding the objective and the bundles provided
2. Use `wink query` to examine bundle contents systematically
3. Look for patterns across multiple executions
4. Identify root causes, not just symptoms
5. Quantify findings with data from queries

**Analysis patterns:**
- Compare successful vs failed executions
- Look for common tool call patterns
- Check for budget/deadline exhaustion
- Examine error patterns and frequencies
- Identify prompt-related issues vs tool-related issues
"""

_DEFAULT_OUTPUT_FORMAT = """\
Produce a markdown report with this structure:

```markdown
# Analysis Report

**Objective:** [research question]
**Bundles Analyzed:** [count]

## Executive Summary
[2-3 sentences summarizing key findings]

## Findings
### Finding 1: [title]
**Severity:** Critical | Warning | Info
**Frequency:** X/Y samples
[description with evidence references]
**Evidence:** See `evidence/...`

## Recommendations
1. **[High]** [action]
   - Rationale: ...
   - Expected impact: ...
```

**Severity levels:**
- **Critical**: Blocks correct behavior, affects >50% of samples
- **Warning**: Degrades quality, affects 10-50% of samples
- **Info**: Minor observation, affects <10% of samples
"""

_DEFAULT_EVIDENCE_GATHERING = """\
Use `wink query` to examine bundle contents via SQL. Save query results
to files and reference them in the report.

**Common queries:**
- Failed samples: `SELECT sample_id, score FROM eval WHERE passed = false`
- Tool usage: `SELECT tool_name, COUNT(*) FROM tool_invocations GROUP BY tool_name`
- Error patterns: `SELECT * FROM logs WHERE level = 'ERROR'`
- Token usage: `SELECT prompt_tokens, completion_tokens FROM token_usage`

**Best practices:**
- Run broad queries first to understand the data
- Narrow down to specific patterns based on initial findings
- Save query results as evidence for the report
- Cross-reference findings across multiple queries
"""


@dataclass(frozen=True, slots=True)
class _AnalysisParams:
    """Parameters for the analysis prompt template."""

    objective: str
    bundles: str
    source: str


@dataclass(frozen=True, slots=True)
class _StaticParams:
    """Empty params for static (no-substitution) sections."""


def build_analysis_template(
    overrides: AnalysisPromptOverrides | None = None,
) -> PromptTemplate[AnalysisBundle]:
    """Build the analysis prompt template with optional overrides.

    Args:
        overrides: Optional section content overrides.

    Returns:
        A PromptTemplate configured for analysis tasks.
    """
    eff = overrides or AnalysisPromptOverrides()

    methodology = (
        eff.methodology if eff.methodology is not None else _DEFAULT_METHODOLOGY
    )
    output_format = (
        eff.output_format if eff.output_format is not None else _DEFAULT_OUTPUT_FORMAT
    )
    evidence = (
        eff.evidence_gathering
        if eff.evidence_gathering is not None
        else _DEFAULT_EVIDENCE_GATHERING
    )

    return PromptTemplate[AnalysisBundle](
        ns="weakincentives.analysis",
        key="analysis-agent",
        sections=[
            MarkdownSection[_AnalysisParams](
                title="Objective",
                key="objective",
                template="Analyze the following debug bundles.\n\n**Objective:** $objective\n\n**Bundles:** $bundles\n\n**Source:** $source",
            ),
            MarkdownSection[_StaticParams](
                title="Methodology",
                key="methodology",
                template=methodology,
            ),
            MarkdownSection[_StaticParams](
                title="Output Format",
                key="output-format",
                template=output_format,
            ),
            MarkdownSection[_StaticParams](
                title="Evidence Gathering",
                key="evidence-gathering",
                template=evidence,
            ),
        ],
    )


def build_analysis_prompt(
    request: AnalysisRequest,
    overrides: AnalysisPromptOverrides | None = None,
) -> Prompt[AnalysisBundle]:
    """Build a bound analysis prompt for a specific request.

    Args:
        request: The analysis request with objective and bundles.
        overrides: Optional section content overrides.

    Returns:
        A Prompt instance ready for evaluation.
    """
    template = build_analysis_template(overrides)
    bundles_str = ", ".join(str(b) for b in request.bundles)
    return Prompt(template).bind(
        _AnalysisParams(
            objective=request.objective,
            bundles=bundles_str,
            source=request.source,
        )
    )


__all__ = [
    "build_analysis_prompt",
    "build_analysis_template",
]
