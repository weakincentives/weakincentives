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

"""Core override protocol and data models.

This module provides the minimal interface for prompt overrides:

- :class:`PromptOverridesStore`: Protocol for override persistence
- :class:`PromptOverride`: Override payload with sections and tools
- :class:`PromptDescriptor`: Hash-based prompt metadata for versioning
- Hashing utilities: :func:`hash_text`, :func:`hash_json`

For the full filesystem-backed implementation with validation and inspection,
see :mod:`weakincentives.contrib.overrides`.
"""

from __future__ import annotations

# Import runtime.logging early to avoid circular import issues.
# This mirrors the original import order where local_store.py was imported here.
from ...runtime.logging import (  # noqa: F401
    get_logger as _get_logger,  # pyright: ignore[reportUnusedImport]
)
from .versioning import (
    HexDigest,
    PromptDescriptor,
    PromptLike,
    PromptOverride,
    PromptOverridesError,
    PromptOverridesStore,
    SectionDescriptor,
    SectionLike,
    SectionOverride,
    TaskExampleDescriptor,
    TaskExampleOverride,
    TaskStepOverride,
    ToolContractProtocol,
    ToolDescriptor,
    ToolExampleOverride,
    ToolOverride,
    descriptor_for_prompt,
    ensure_hex_digest,
    hash_json,
    hash_text,
)

__all__ = [
    "HexDigest",
    "PromptDescriptor",
    "PromptLike",
    "PromptOverride",
    "PromptOverridesError",
    "PromptOverridesStore",
    "SectionDescriptor",
    "SectionLike",
    "SectionOverride",
    "TaskExampleDescriptor",
    "TaskExampleOverride",
    "TaskStepOverride",
    "ToolContractProtocol",
    "ToolDescriptor",
    "ToolExampleOverride",
    "ToolOverride",
    "descriptor_for_prompt",
    "ensure_hex_digest",
    "hash_json",
    "hash_text",
]
