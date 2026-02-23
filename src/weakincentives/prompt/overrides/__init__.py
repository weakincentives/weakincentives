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

"""Runtime customization of prompts without code changes.

This subpackage provides infrastructure for overriding prompt sections, tool
descriptions, and task examples at runtime. Overrides are validated against
content hashes to detect staleness when the underlying prompt changes.

Core Concepts
-------------

**PromptDescriptor**
    A stable metadata object describing a prompt's structure. Contains:

    - Namespace and key identifying the prompt
    - Section descriptors with content hashes
    - Tool descriptors with contract hashes
    - Task example descriptors with content hashes

    Descriptors are computed from prompts and used to validate overrides.

**PromptOverride**
    A collection of section, tool, and task example overrides for a prompt.
    Each override is tagged (e.g., "latest", "experiment-1") for versioning.

**SectionOverride**
    Replacement content for a section. Includes:

    - Path tuple identifying the section (e.g., ``("workspace", "tools")``)
    - Expected hash of the original content (staleness detection)
    - Replacement body text
    - Optional summary text for SUMMARY visibility
    - Optional default visibility override

**ToolOverride**
    Description patches for a tool. Includes:

    - Tool name
    - Expected contract hash (validates schema hasn't changed)
    - Optional replacement description
    - Optional parameter description patches

**TaskExampleOverride**
    Modifications to task examples within TaskExamplesSection. Supports
    modify, remove, and append actions.

Hash-Based Staleness Detection
------------------------------

Overrides include expected hashes that must match the current prompt content.
When the underlying prompt changes, mismatched hashes cause the override to
be skipped (soft failure) or rejected (hard failure during write).

This prevents applying outdated overrides that may no longer be semantically
valid after prompt changes.

Hash computation:

- **Sections**: SHA-256 of the body template text
- **Tools**: SHA-256 of description + params schema + result schema
- **Task examples**: SHA-256 of objective + steps + outcome

Override Store Protocol
-----------------------

The ``PromptOverridesStore`` protocol defines the interface for override
persistence:

- ``resolve(descriptor, tag)``: Load and validate overrides
- ``upsert(descriptor, override)``: Save overrides with validation
- ``delete(ns, prompt_key, tag)``: Remove override file
- ``store(descriptor, override)``: Store a single override atomically
- ``seed(prompt, tag)``: Initialize override file with current content

LocalPromptOverridesStore
-------------------------

File-based implementation storing overrides as JSON files:

- Default location: ``.weakincentives/prompts/overrides/``
- File naming: ``{ns}/{prompt_key}/{tag}.json``
- Atomic writes via temp file + rename
- File locking for concurrent access safety

Basic Usage
-----------

Creating an override store::

    from weakincentives.prompt.overrides import (
        LocalPromptOverridesStore,
        PromptDescriptor,
        SectionOverride,
    )

    store = LocalPromptOverridesStore(root_path="/path/to/project")

Seeding overrides from a prompt::

    from weakincentives.prompt import Prompt, PromptTemplate

    template = PromptTemplate(ns="my-app", key="main", sections=[...])
    prompt = Prompt(template)

    # Creates override file with current section/tool content
    override = store.seed(prompt, tag="latest")

Modifying a section override::

    descriptor = PromptDescriptor.from_prompt(prompt)

    # Find the section's current hash
    section_desc = next(
        s for s in descriptor.sections
        if s.path == ("instructions",)
    )

    # Create section override with hash for validation
    section_override = SectionOverride(
        path=("instructions",),
        expected_hash=section_desc.content_hash,
        body="Updated instructions content...",
    )

    # Store atomically (read-modify-write with lock)
    store.store(descriptor, section_override, tag="latest")

Applying overrides at render time::

    prompt = Prompt(template, overrides_store=store, overrides_tag="latest")
    prompt.bind(params)

    with prompt.resource_scope():
        rendered = prompt.render()  # Applies matching overrides

Override File Format
--------------------

Override files are JSON with this structure::

    {
        "version": 2,
        "ns": "my-app",
        "prompt_key": "main",
        "tag": "latest",
        "sections": {
            "instructions": {
                "path": ["instructions"],
                "expected_hash": "abc123...",
                "body": "Override content...",
                "summary": "Optional summary...",
                "visibility": "full"
            }
        },
        "tools": {
            "my_tool": {
                "expected_contract_hash": "def456...",
                "description": "Updated description",
                "param_descriptions": {
                    "path": "File path to process"
                }
            }
        },
        "task_example_overrides": [...]
    }

Helper Functions
----------------

**hash_text(value)**
    Compute SHA-256 hex digest of UTF-8 encoded text.

**hash_json(value)**
    Compute SHA-256 of canonical JSON representation.

**ensure_hex_digest(value, field_name)**
    Validate and normalize a hex digest string.

**descriptor_for_prompt(prompt)**
    Get or compute a PromptDescriptor for a prompt.

**filter_override_for_descriptor(descriptor, override)**
    Filter out stale or unknown overrides, returning valid ones.

Module Structure
----------------

- ``versioning``: Core types (Descriptor, Override, HexDigest, hash functions)
- ``local_store``: LocalPromptOverridesStore implementation
- ``validation``: Override loading, serialization, and validation
- ``inspection``: Override file discovery utilities
- ``_fs``: Filesystem operations (locking, atomic writes)

Error Handling
--------------

**PromptOverridesError**
    Raised when override operations fail:

    - Invalid hash format
    - Hash mismatch (strict mode)
    - Unknown section/tool path
    - Invalid override file format
    - File I/O errors

Stale overrides are handled gracefully:

- During resolve: Stale overrides are filtered out with debug logging
- During upsert/store: Stale overrides raise PromptOverridesError

Concurrency Safety
------------------

LocalPromptOverridesStore uses file locking for concurrent access:

- Exclusive locks during read-modify-write operations
- Atomic writes via temp file + rename
- Lock scope covers entire store() operation to prevent TOCTOU races
"""

from __future__ import annotations

from .inspection import (
    OverrideFileMetadata,
    iter_override_files,
    resolve_overrides_root,
)
from .local_store import LocalPromptOverridesStore
from .validation import filter_override_for_descriptor
from .versioning import (
    HexDigest,
    PromptDescriptor,
    PromptLike,
    PromptOverride,
    PromptOverridesError,
    PromptOverridesStore,
    SectionDescriptor,
    SectionOverride,
    TaskExampleDescriptor,
    TaskExampleOverride,
    TaskStepOverride,
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
    "LocalPromptOverridesStore",
    "OverrideFileMetadata",
    "PromptDescriptor",
    "PromptLike",
    "PromptOverride",
    "PromptOverridesError",
    "PromptOverridesStore",
    "SectionDescriptor",
    "SectionOverride",
    "TaskExampleDescriptor",
    "TaskExampleOverride",
    "TaskStepOverride",
    "ToolDescriptor",
    "ToolExampleOverride",
    "ToolOverride",
    "descriptor_for_prompt",
    "ensure_hex_digest",
    "filter_override_for_descriptor",
    "hash_json",
    "hash_text",
    "iter_override_files",
    "resolve_overrides_root",
]
