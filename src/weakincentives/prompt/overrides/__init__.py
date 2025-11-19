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

"""Prompt override infrastructure used by :mod:`weakincentives.prompt`."""

from __future__ import annotations

from .inspection import (
    OverrideFileMetadata,
    iter_override_files,
    resolve_overrides_root,
)
from .local_store import LocalPromptOverridesStore
from .validation import filter_override_for_descriptor
from .versioning import (
    ChapterDescriptor,
    HexDigest,
    PromptDescriptor,
    PromptLike,
    PromptOverride,
    PromptOverridesError,
    PromptOverridesStore,
    SectionDescriptor,
    SectionOverride,
    ToolDescriptor,
    ToolOverride,
    ensure_hex_digest,
    hash_json,
    hash_text,
)

__all__ = [
    "ChapterDescriptor",
    "HexDigest",
    "LocalPromptOverridesStore",
    "OverrideFileMetadata",
    "PromptDescriptor",
    "PromptLike",
    "PromptOverride",
    "PromptOverridesError",
    "PromptOverridesStore",
    "SectionDescriptor",
    "SectionOverride",
    "ToolDescriptor",
    "ToolOverride",
    "ensure_hex_digest",
    "filter_override_for_descriptor",
    "hash_json",
    "hash_text",
    "iter_override_files",
    "resolve_overrides_root",
]
