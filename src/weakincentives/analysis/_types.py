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

"""Data types for the analysis loop.

Defines the core data structures used by the analysis subsystem:
completion notifications, analysis requests, analysis bundles,
and configuration objects.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Literal
from uuid import UUID, uuid4

from ..clock import SYSTEM_CLOCK


@dataclass(frozen=True, slots=True)
class CompletionNotification:
    """Notification emitted when an AgentLoop or EvalLoop execution finishes.

    Loops publish these to a shared notifications mailbox so downstream
    consumers (e.g. AnalysisForwarder) can react to completions.
    """

    source: Literal["agent_loop", "eval_loop"]
    """Which loop produced this notification."""

    bundle_path: Path
    """Path to the debug bundle for this execution."""

    request_id: UUID
    """Correlates with the original request."""

    success: bool
    """Whether the execution completed without error."""

    passed: bool | None = None
    """EvalLoop only: whether the evaluation passed."""

    score: float | None = None
    """EvalLoop only: the evaluation score."""

    completed_at: datetime = field(default_factory=lambda: datetime.now(UTC))
    """When the execution completed."""


@dataclass(frozen=True, slots=True)
class EvalContext:
    """Optional context from an evaluation run.

    Provides metadata about the evaluation that produced the bundles
    being analyzed, giving the analysis agent additional context.
    """

    experiment_name: str | None = None
    """Name of the experiment the bundles came from."""

    pass_rate: float | None = None
    """Overall pass rate of the evaluation run."""

    sample_count: int | None = None
    """Total number of samples evaluated."""


@dataclass(frozen=True, slots=True)
class AnalysisRequest:
    """Request to analyze one or more debug bundles.

    Sent to the AnalysisLoop's mailbox. The analysis agent uses
    ``wink query`` to interrogate the bundles and produces a report.
    """

    objective: str
    """The research question or analysis goal."""

    bundles: tuple[Path, ...]
    """Paths to debug bundles to analyze."""

    source: Literal["agent_loop", "eval_loop", "manual"] = "manual"
    """Where this request originated."""

    eval_context: EvalContext | None = None
    """Optional evaluation context for additional metadata."""

    request_id: UUID = field(default_factory=uuid4)
    """Unique identifier for this analysis request."""

    created_at: datetime = field(default_factory=SYSTEM_CLOCK.utcnow)
    """When this request was created."""


@dataclass(frozen=True, slots=True)
class AnalysisBundle:
    """Structured output from an analysis run.

    Contains the markdown report produced by the analysis agent. The
    physical archive (zip with evidence, queries, source bundles) is
    created by AnalysisLoop.finalize() in the configured output directory.
    """

    report: str
    """Markdown report following the standard analysis format."""


@dataclass(frozen=True, slots=True)
class AnalysisBudget:
    """Rate limiting budget for the AnalysisForwarder.

    Controls how many analysis requests can be forwarded within a
    given time window to prevent runaway analysis costs.
    """

    max_requests: int = 100
    """Maximum analysis requests per reset interval."""

    reset_interval: timedelta = field(default_factory=lambda: timedelta(hours=1))
    """Time window after which the request counter resets."""


@dataclass(frozen=True, slots=True)
class AnalysisForwarderConfig:
    """Configuration for the AnalysisForwarder.

    Controls which completion notifications are forwarded for analysis
    and how to construct the analysis requests.
    """

    objective: str
    """The research question passed to each analysis request."""

    sample_rate: float = 0.1
    """Fraction of successful notifications to forward (0.0 to 1.0)."""

    always_forward_failures: bool = True
    """If True, all failed notifications are forwarded regardless of sample_rate."""

    budget: AnalysisBudget = field(default_factory=AnalysisBudget)
    """Rate limiting configuration."""


@dataclass(frozen=True, slots=True)
class AnalysisPromptOverrides:
    """Override individual sections of the prebuilt analysis prompt.

    Each field replaces the corresponding section content when provided.
    """

    methodology: str | None = None
    """How to approach the analysis."""

    output_format: str | None = None
    """Report structure and formatting requirements."""

    evidence_gathering: str | None = None
    """How to use wink query for evidence collection."""


@dataclass(frozen=True, slots=True)
class AnalysisLoopConfig:
    """Configuration for the AnalysisLoop.

    Controls output directory, source bundle inclusion, and prompt
    customization.
    """

    output_dir: Path = field(default_factory=lambda: Path("./analysis-bundles/"))
    """Directory where analysis bundles are written."""

    include_source_bundles: bool = True
    """Whether to include original debug bundles in the analysis archive."""

    max_source_bundle_size: int = 50_000_000
    """Maximum size in bytes for included source bundles (50MB default)."""

    overrides: AnalysisPromptOverrides | None = None
    """Optional prompt section overrides."""


__all__ = [
    "AnalysisBudget",
    "AnalysisBundle",
    "AnalysisForwarderConfig",
    "AnalysisLoopConfig",
    "AnalysisPromptOverrides",
    "AnalysisRequest",
    "CompletionNotification",
    "EvalContext",
]
