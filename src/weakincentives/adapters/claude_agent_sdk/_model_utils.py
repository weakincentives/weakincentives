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

"""Model ID constants and conversion utilities for Claude Agent SDK adapter.

Maps between Anthropic API model names and AWS Bedrock model IDs.
"""

from __future__ import annotations

import logging

_logger = logging.getLogger(__name__)

# Default model for both Anthropic API and Bedrock
DEFAULT_MODEL = "claude-opus-4-6"
DEFAULT_BEDROCK_MODEL = "us.anthropic.claude-opus-4-6-v1"

# Model name mappings between Anthropic API and Bedrock
_ANTHROPIC_TO_BEDROCK: dict[str, str] = {
    "claude-opus-4-6": "us.anthropic.claude-opus-4-6-v1",
    "claude-opus-4-5-20251101": "us.anthropic.claude-opus-4-5-20251101-v1:0",
    "claude-sonnet-4-5-20250929": "us.anthropic.claude-sonnet-4-5-20250929-v1:0",
    "claude-sonnet-4-20250514": "us.anthropic.claude-sonnet-4-20250514-v1:0",
    "claude-3-5-sonnet-20241022": "us.anthropic.claude-3-5-sonnet-20241022-v2:0",
}

_BEDROCK_TO_ANTHROPIC: dict[str, str] = {v: k for k, v in _ANTHROPIC_TO_BEDROCK.items()}


def get_supported_bedrock_models() -> dict[str, str]:
    """Get the mapping of supported Anthropic models to Bedrock model IDs.

    Returns:
        Dictionary mapping Anthropic model names to Bedrock model IDs.
        Example: {"claude-opus-4-5-20251101": "us.anthropic.claude-opus-4-5-20251101-v1:0"}
    """
    return dict(_ANTHROPIC_TO_BEDROCK)


def to_bedrock_model_id(anthropic_model: str) -> str:
    """Convert an Anthropic model name to Bedrock model ID.

    Args:
        anthropic_model: Anthropic model name (e.g., "claude-opus-4-5-20251101")

    Returns:
        Bedrock model ID with cross-region inference prefix.
        Returns the input unchanged if already a Bedrock ID or not in mapping.
    """
    # Already a Bedrock model ID
    if anthropic_model.startswith(("us.", "anthropic.")):
        return anthropic_model

    # Look up in mapping
    result = _ANTHROPIC_TO_BEDROCK.get(anthropic_model, anthropic_model)
    if result == anthropic_model:
        _logger.debug(
            "isolation.to_bedrock_model_id.unmapped",
            extra={"model": anthropic_model, "returned_unchanged": True},
        )
    return result


def to_anthropic_model_name(bedrock_model_id: str) -> str:
    """Convert a Bedrock model ID to Anthropic model name.

    Args:
        bedrock_model_id: Bedrock model ID (e.g., "us.anthropic.claude-opus-4-5-20251101-v1:0")

    Returns:
        Anthropic model name.
        Returns the input unchanged if not a Bedrock ID or not in mapping.
    """
    # Not a Bedrock model ID
    if not bedrock_model_id.startswith(("us.", "anthropic.")):
        return bedrock_model_id

    # Look up in mapping
    return _BEDROCK_TO_ANTHROPIC.get(bedrock_model_id, bedrock_model_id)
