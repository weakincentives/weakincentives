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

"""Full-featured prompt overrides implementation.

This package provides a filesystem-backed implementation of the
:class:`~weakincentives.prompt.PromptOverridesStore` protocol, including
versioning, validation, and inspection utilities.

Core protocol and data models remain in :mod:`weakincentives.prompt.overrides`.
This contrib package adds:

- ``LocalPromptOverridesStore``: Filesystem-backed store implementation
- Validation utilities for hash-based override verification
- Inspection helpers for browsing override files

Example usage::

    from weakincentives.contrib.overrides import LocalPromptOverridesStore

    store = LocalPromptOverridesStore()
    override = store.seed(prompt, tag="stable")
"""

from __future__ import annotations

from .inspection import (
    OverrideFileMetadata,
    iter_override_files,
    resolve_overrides_root,
)
from .local_store import LocalPromptOverridesStore
from .validation import filter_override_for_descriptor

__all__ = [
    "LocalPromptOverridesStore",
    "OverrideFileMetadata",
    "filter_override_for_descriptor",
    "iter_override_files",
    "resolve_overrides_root",
]
