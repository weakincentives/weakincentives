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
from . import (
    query_format as _query_format,
    query_parsers as _query_parsers,
    query_schema as _query_schema,
)

ColumnInfo = _query_schema.ColumnInfo
QueryDatabase = _query_schema.QueryDatabase
QueryError = _query_schema.QueryError
SchemaHints = _query_schema.SchemaHints
SchemaOutput = _query_schema.SchemaOutput
TableInfo = _query_schema.TableInfo

format_as_json = _query_format.format_as_json
format_as_table = _query_format.format_as_table

_apply_tool_result_details = _query_parsers.apply_tool_result_details
_apply_transcript_content_fallbacks = _query_parsers.apply_transcript_content_fallbacks
_create_dynamic_slice_table = _query_schema.create_dynamic_slice_table
_extract_slices_from_snapshot = _query_parsers.extract_slices_from_snapshot
_extract_tool_call_from_entry = _query_parsers.extract_tool_call_from_entry
_extract_tool_use_from_content = _query_parsers.extract_tool_use_from_content
_extract_transcript_details = _query_parsers.extract_transcript_details
_extract_transcript_message_details = _query_parsers.extract_transcript_message_details
_extract_transcript_parsed_obj = _query_parsers.extract_transcript_parsed_obj
_extract_transcript_row = _query_parsers.extract_transcript_row
_flatten_json = _query_schema.flatten_json
_get_table_description = _query_schema.get_table_description
_infer_sqlite_type = _query_schema.infer_sqlite_type
_is_tool_event = _query_parsers.is_tool_event
is_cache_valid = _query_schema.is_cache_valid
_json_to_sql_value = _query_schema.json_to_sql_value
_SCHEMA_VERSION = _query_schema.SCHEMA_VERSION
_normalize_slice_type = _query_schema.normalize_slice_type
_is_cache_valid = _query_schema.is_cache_valid
_process_session_line = _query_schema.process_session_line
_safe_json_dumps = _query_parsers.safe_json_dumps
_stringify_transcript_content = _query_parsers.stringify_transcript_content
_stringify_transcript_mapping = _query_parsers.stringify_transcript_mapping
_stringify_transcript_tool_use = _query_parsers.stringify_transcript_tool_use

iter_bundle_files = _query_schema.iter_bundle_files
resolve_bundle_path = _query_schema.resolve_bundle_path


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
    "ColumnInfo",
    "QueryDatabase",
    "QueryError",
    "SchemaHints",
    "SchemaOutput",
    "TableInfo",
    "_SCHEMA_VERSION",
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
    "_stringify_transcript_content",
    "_stringify_transcript_mapping",
    "_stringify_transcript_tool_use",
    "export_jsonl",
    "format_as_json",
    "format_as_table",
    "iter_bundle_files",
    "open_query_database",
    "resolve_bundle_path",
]
