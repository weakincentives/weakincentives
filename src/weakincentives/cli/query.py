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

"""SQL query interface for debug bundles.

Enables querying debug bundle contents via SQL by loading bundle artifacts
into a cached SQLite database.
"""

from __future__ import annotations

from pathlib import Path

from ..debug.bundle import BundleValidationError, DebugBundle
from .query_format import format_as_json, format_as_table
from .query_parsers import (
    apply_tool_result_details,
    apply_transcript_content_fallbacks,
    extract_slices_from_snapshot,
    extract_tool_call_from_entry,
    extract_tool_use_from_content,
    extract_transcript_details,
    extract_transcript_message_details,
    extract_transcript_parsed_obj,
    extract_transcript_row,
    is_tool_event,
    safe_json_dumps,
    stringify_transcript_content,
    stringify_transcript_mapping,
    stringify_transcript_tool_use,
)
from .query_schema import (
    SCHEMA_VERSION,
    ColumnInfo,
    QueryDatabase,
    QueryError,
    SchemaHints,
    SchemaOutput,
    TableInfo,
    create_dynamic_slice_table,
    flatten_json,
    get_table_description,
    infer_sqlite_type,
    is_cache_valid,
    iter_bundle_files,
    json_to_sql_value,
    normalize_slice_type,
    process_session_line,
    resolve_bundle_path,
)

_apply_tool_result_details = apply_tool_result_details
_apply_transcript_content_fallbacks = apply_transcript_content_fallbacks
_create_dynamic_slice_table = create_dynamic_slice_table
_extract_slices_from_snapshot = extract_slices_from_snapshot
_extract_tool_call_from_entry = extract_tool_call_from_entry
_extract_tool_use_from_content = extract_tool_use_from_content
_extract_transcript_details = extract_transcript_details
_extract_transcript_message_details = extract_transcript_message_details
_extract_transcript_parsed_obj = extract_transcript_parsed_obj
_extract_transcript_row = extract_transcript_row
_flatten_json = flatten_json
_get_table_description = get_table_description
_infer_sqlite_type = infer_sqlite_type
_is_tool_event = is_tool_event
_json_to_sql_value = json_to_sql_value
_SCHEMA_VERSION = SCHEMA_VERSION
_normalize_slice_type = normalize_slice_type
_is_cache_valid = is_cache_valid
_process_session_line = process_session_line
_safe_json_dumps = safe_json_dumps
_stringify_transcript_content = stringify_transcript_content
_stringify_transcript_mapping = stringify_transcript_mapping
_stringify_transcript_tool_use = stringify_transcript_tool_use


def open_query_database(bundle_path: Path) -> QueryDatabase:
    """Open or create a query database for a bundle.

    Implements caching: creates bundle.zip.sqlite alongside the bundle,
    reusing if valid (cache mtime >= bundle mtime).

    Args:
        bundle_path: Path to the debug bundle zip file or directory.
            If a directory, uses the most recently modified .zip file.

    Returns:
        QueryDatabase instance ready for queries.

    Raises:
        QueryError: If bundle cannot be loaded.
    """
    resolved_path = resolve_bundle_path(bundle_path)
    cache_path = resolved_path.with_suffix(resolved_path.suffix + ".sqlite")

    try:
        bundle = DebugBundle.load(resolved_path)
    except BundleValidationError as e:
        raise QueryError(f"Failed to load bundle: {e}") from e

    db = QueryDatabase(bundle, cache_path)

    if not is_cache_valid(resolved_path, cache_path):
        # Remove stale cache if exists
        if cache_path.exists():
            cache_path.unlink()
        db.build()
    else:
        # Cache is valid, mark as built for read-only mode
        db.mark_built()

    return db


def export_jsonl(bundle: DebugBundle, source: str) -> str | None:
    """Export raw JSONL content from bundle.

    Args:
        bundle: Debug bundle to export from.
        source: Either "logs" for logs/app.jsonl or "session" for session/after.jsonl.

    Returns:
        Raw JSONL content, or None if not present.
    """
    if source == "logs":
        return bundle.logs
    if source == "session":
        return bundle.session_after
    return None


__all__ = [  # noqa: RUF022
    "_apply_tool_result_details",
    "_apply_transcript_content_fallbacks",
    "_create_dynamic_slice_table",
    "_extract_slices_from_snapshot",
    "_extract_tool_call_from_entry",
    "_extract_tool_use_from_content",
    "_extract_transcript_details",
    "_extract_transcript_message_details",
    "_extract_transcript_parsed_obj",
    "_extract_transcript_row",
    "_flatten_json",
    "_get_table_description",
    "_infer_sqlite_type",
    "_is_cache_valid",
    "_is_tool_event",
    "_json_to_sql_value",
    "_normalize_slice_type",
    "_process_session_line",
    "_safe_json_dumps",
    "_SCHEMA_VERSION",
    "_stringify_transcript_content",
    "_stringify_transcript_mapping",
    "_stringify_transcript_tool_use",
    "ColumnInfo",
    "export_jsonl",
    "format_as_json",
    "format_as_table",
    "iter_bundle_files",
    "open_query_database",
    "QueryDatabase",
    "QueryError",
    "resolve_bundle_path",
    "SchemaHints",
    "SchemaOutput",
    "TableInfo",
]
