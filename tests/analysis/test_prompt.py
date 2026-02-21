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

"""Tests for the analysis prompt template."""

from __future__ import annotations

from pathlib import Path

from weakincentives.analysis import (
    AnalysisPromptOverrides,
    AnalysisRequest,
    build_analysis_prompt,
    build_analysis_template,
)


class TestBuildAnalysisTemplate:
    """Tests for build_analysis_template."""

    def test_default_template(self) -> None:
        """Default template has all expected sections."""
        template = build_analysis_template()
        assert template.ns == "weakincentives.analysis"
        assert template.key == "analysis-agent"
        assert len(template.sections) == 4

    def test_template_section_keys(self) -> None:
        """Template sections have the expected keys."""
        template = build_analysis_template()
        keys = [s.section.key for s in template.sections]
        assert "objective" in keys
        assert "methodology" in keys
        assert "output-format" in keys
        assert "evidence-gathering" in keys

    def test_overrides_replace_content(self) -> None:
        """Overrides replace the default section content."""
        overrides = AnalysisPromptOverrides(
            methodology="Custom methodology.",
            output_format="Custom output format.",
            evidence_gathering="Custom evidence.",
        )
        template = build_analysis_template(overrides)
        # The template should build without error
        assert len(template.sections) == 4


class TestBuildAnalysisPrompt:
    """Tests for build_analysis_prompt."""

    def test_builds_from_request(self) -> None:
        """Prompt is built and bound from an AnalysisRequest."""
        request = AnalysisRequest(
            objective="Find root cause",
            bundles=(Path("/tmp/b1.zip"), Path("/tmp/b2.zip")),
            source="eval_loop",
        )
        prompt = build_analysis_prompt(request)
        rendered = prompt.render()
        assert "Find root cause" in rendered.text
        assert "/tmp/b1.zip" in rendered.text
        assert "/tmp/b2.zip" in rendered.text
        assert "eval_loop" in rendered.text

    def test_builds_with_overrides(self) -> None:
        """Prompt respects overrides when building."""
        request = AnalysisRequest(
            objective="test",
            bundles=(Path("/tmp/b.zip"),),
        )
        overrides = AnalysisPromptOverrides(
            methodology="Focus on security patterns.",
        )
        prompt = build_analysis_prompt(request, overrides=overrides)
        rendered = prompt.render()
        assert "security patterns" in rendered.text

    def test_single_bundle(self) -> None:
        """Works with a single bundle."""
        request = AnalysisRequest(
            objective="analyze single",
            bundles=(Path("/tmp/single.zip"),),
        )
        prompt = build_analysis_prompt(request)
        rendered = prompt.render()
        assert "single.zip" in rendered.text
