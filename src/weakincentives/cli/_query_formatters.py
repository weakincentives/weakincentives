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

"""Output formatting for query results (table, JSON, JSONL export)."""

# pyright: reportPrivateUsage=false

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from typing import Any

from ..debug import DebugBundle
from ._query_helpers import _MAX_COLUMN_WIDTH

__all__ = [
    "export_jsonl",
    "format_as_json",
    "format_as_table",
]


def format_as_table(rows: Sequence[Mapping[str, Any]], *, truncate: bool = True) -> str:
    """Format query results as ASCII table."""
    if not rows:
        return "(no results)"
    max_width = _MAX_COLUMN_WIDTH if truncate else None
    columns = list(rows[0].keys())
    widths: dict[str, int] = {}
    for col in columns:
        col_max = len(col)
        for row in rows:
            val = str(row.get(col, ""))
            if max_width is not None and len(val) > max_width:
                val = val[: max_width - 3] + "..."
            col_max = max(col_max, len(val))
        widths[col] = col_max if max_width is None else min(col_max, max_width)
    header = " | ".join(col.ljust(widths[col]) for col in columns)
    separator = "-+-".join("-" * widths[col] for col in columns)
    lines: list[str] = [header, separator]
    for row in rows:
        cells: list[str] = []
        for col in columns:
            val = str(row.get(col, ""))
            if max_width is not None and len(val) > max_width:
                val = val[: max_width - 3] + "..."
            cells.append(val.ljust(widths[col]))
        lines.append(" | ".join(cells))
    return "\n".join(lines)


def format_as_json(rows: Sequence[Mapping[str, Any]]) -> str:
    """Format query results as JSON."""
    result = [dict(row) for row in rows]
    return json.dumps(result, indent=2)


def export_jsonl(bundle: DebugBundle, source: str) -> str | None:
    """Export raw JSONL content from bundle."""
    if source == "logs":
        return bundle.logs
    if source == "session":
        return bundle.session_after
    return None
