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

"""Low-level utilities for the query module.

Contains JSON flattening, SQLite type inference, table descriptions,
and cache validation helpers.
"""

from __future__ import annotations

import json
import sqlite3
from collections.abc import Mapping
from pathlib import Path
from typing import cast

from ..types import JSONValue

__all__ = [
    "_MAX_COLUMN_WIDTH",
    "_SCHEMA_VERSION",
    "_flatten_json",
    "_get_table_description",
    "_infer_sqlite_type",
    "_is_cache_valid",
    "_json_to_sql_value",
    "_normalize_slice_type",
    "_safe_json_dumps",
]

# Maximum column width for ASCII table output
_MAX_COLUMN_WIDTH = 50

# Schema version for cache invalidation - increment when schema changes
_SCHEMA_VERSION = (
    9  # v9: fix Codex agentMessage contaminating tool metrics, bridged tool events
)


def _normalize_slice_type(type_name: str) -> str:
    """Normalize a slice type name to a valid table name.

    Example: 'myapp.state:AgentPlan' -> 'slice_agentplan'
    """
    # Extract class name after colon if present
    if ":" in type_name:
        type_name = type_name.split(":")[-1]
    # Extract class name after last dot
    if "." in type_name:
        type_name = type_name.rsplit(".", 1)[-1]
    # Lowercase and prefix
    return f"slice_{type_name.lower()}"


def _flatten_json(
    obj: JSONValue, prefix: str = "", sep: str = "_"
) -> dict[str, JSONValue]:
    """Flatten nested JSON object into flat key-value pairs."""
    result: dict[str, JSONValue] = {}

    if isinstance(obj, Mapping):
        # Cast to proper type for iteration
        mapping = cast("Mapping[str, JSONValue]", obj)
        for key, value in mapping.items():
            new_key = f"{prefix}{sep}{key}" if prefix else str(key)
            if isinstance(value, Mapping):
                nested = cast(JSONValue, value)
                result.update(_flatten_json(nested, new_key, sep))
            elif isinstance(value, list):
                # Store lists as JSON strings
                result[new_key] = json.dumps(value)
            else:
                result[new_key] = value
    elif isinstance(obj, list):
        result[prefix] = json.dumps(obj)
    else:
        result[prefix] = obj

    return result


def _infer_sqlite_type(value: object) -> str:
    """Infer SQLite type from Python value."""
    if value is None:
        return "TEXT"
    if isinstance(value, bool):
        return "INTEGER"
    if isinstance(value, int):
        return "INTEGER"
    if isinstance(value, float):
        return "REAL"
    return "TEXT"


def _json_to_sql_value(value: JSONValue) -> object:
    """Convert JSON value to SQL-compatible value."""
    if value is None:
        return None
    if isinstance(value, bool):
        return 1 if value else 0
    if isinstance(value, int | float | str):
        return value
    # Lists and dicts become JSON strings
    return json.dumps(value)


def _safe_json_dumps(value: object) -> str:
    """Serialize value to JSON, falling back to str on failure."""
    try:
        return json.dumps(value, ensure_ascii=False)
    except (TypeError, ValueError):
        return str(value)


def _get_table_description(table_name: str, *, is_view: bool = False) -> str:
    """Get description for a table or view by name."""
    descriptions = {
        "manifest": "Bundle metadata",
        "logs": "Log entries (seq extracted from context.sequence_number when present)",
        "transcript": "Transcript entries extracted from logs",
        "tool_calls": "Tool invocations",
        "errors": "Aggregated errors",
        "session_slices": "Session state items",
        "files": "Workspace files",
        "config": "Flattened configuration",
        "metrics": "Token usage and timing",
        "run_context": "Execution IDs",
        "prompt_overrides": "Visibility overrides",
        "eval": "Eval metadata",
        # Environment tables
        "environment": "Flattened environment data (key-value)",
        "env_system": "System/OS info (architecture, CPU, memory)",
        "env_python": "Python runtime (version, venv, executable)",
        "env_git": "Git repository state (commit, branch, remotes)",
        "env_container": "Container runtime info (Docker/K8s)",
        "env_vars": "Filtered environment variables",
        # Views
        "tool_timeline": "View: Tool calls ordered by timestamp",
        "native_tool_calls": "View: Native tool calls from transcripts or legacy logs",
        "transcript_entries": "View: Transcript entries (alias of transcript table)",
        "transcript_flow": "View: Conversation flow with message previews",
        "transcript_tools": "View: Tool usage analysis with paired calls and results",
        "transcript_thinking": "View: Thinking blocks with preview and length",
        "transcript_agents": "View: Agent hierarchy and activity metrics",
        "error_summary": "View: Errors with truncated traceback",
    }
    if table_name.startswith("slice_"):
        return f"Session slice: {table_name[6:]}"
    desc = descriptions.get(table_name, "")
    if is_view and not desc.startswith("View:"):
        desc = f"View: {desc}" if desc else "View"
    return desc


def _is_cache_valid(bundle_path: Path, cache_path: Path) -> bool:
    """Check if cache is still valid based on mtime and schema version."""
    if not cache_path.exists():
        return False
    if cache_path.stat().st_mtime < bundle_path.stat().st_mtime:
        return False
    # Check schema version
    try:
        conn = sqlite3.connect(f"file:{cache_path}?mode=ro", uri=True)
        try:
            cursor = conn.execute("SELECT version FROM _schema_version LIMIT 1")
            row = cursor.fetchone()
            if row is None or row[0] != _SCHEMA_VERSION:
                return False
        except sqlite3.OperationalError:
            # Table doesn't exist (old cache) or other error
            return False
        finally:
            conn.close()
    except sqlite3.Error:
        return False
    return True
