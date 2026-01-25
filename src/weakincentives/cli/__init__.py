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

"""Command line interfaces for the weakincentives package.

This package provides the ``wink`` CLI executable with three main commands
for working with WINK agents and debug bundles.

Commands
--------

debug
    Start a web-based debug server for inspecting debug bundle zip files.
    The server provides a UI for exploring session state, logs, tool calls,
    and other bundle contents via a FastAPI application backed by SQLite.

    Usage::

        wink debug /path/to/bundle.zip
        wink debug /path/to/bundles/  # Uses most recent .zip
        wink debug bundle.zip --host 0.0.0.0 --port 9000
        wink debug bundle.zip --no-open-browser

docs
    Access bundled WINK documentation from the command line. Supports
    listing available documents, searching content, viewing table of
    contents, and reading full documents.

    Usage::

        wink docs list              # List all specs and guides
        wink docs list specs        # List only specs
        wink docs search "session"  # Search for pattern
        wink docs search "reducer" --specs --context 3
        wink docs toc spec SESSIONS # Show table of contents
        wink docs read reference    # Read llms.md reference
        wink docs read spec TOOLS   # Read a specific spec
        wink docs read guide quickstart  # Read a guide

query
    Query debug bundles via SQL. Loads bundle contents into a cached
    SQLite database for flexible exploration. Supports JSON and ASCII
    table output formats.

    Usage::

        wink query bundle.zip --schema  # Show database schema
        wink query bundle.zip "SELECT * FROM logs LIMIT 10"
        wink query bundle.zip "SELECT * FROM tool_calls" --table
        wink query bundle.zip "SELECT * FROM errors" --no-truncate
        wink query bundle.zip --export-jsonl logs  # Raw log export

Global Options
--------------

--log-level LEVEL
    Override log level (CRITICAL, ERROR, WARNING, INFO, DEBUG, NOTSET).

--json-logs / --no-json-logs
    Enable or disable structured JSON log output (default: enabled).

Modules
-------

wink
    Main CLI entry point implementing argument parsing and command dispatch.
    Contains the ``main()`` function that serves as the package entry point.

debug_app
    FastAPI application for the debug bundle web UI. Provides endpoints for
    browsing bundle metadata, session slices, logs, tool calls, metrics, and
    configuration. Includes ``BundleStore`` for managing loaded bundles and
    ``build_debug_app()`` for constructing the FastAPI application.

query
    SQL query interface for debug bundles. Implements ``QueryDatabase`` for
    building and querying SQLite databases from bundle contents. Includes
    schema introspection, result formatting (JSON/ASCII table), and JSONL
    export capabilities.

docs_metadata
    Document descriptions for the ``wink docs list`` command. Contains
    ``SPEC_DESCRIPTIONS`` and ``GUIDE_DESCRIPTIONS`` dictionaries mapping
    document names to short summaries.

Examples
--------

Inspect a debug bundle with the web UI::

    wink debug ./debug-bundles/session-123.zip

Query tool call failures::

    wink query bundle.zip "SELECT * FROM tool_calls WHERE success = 0" --table

Search documentation for reducer patterns::

    wink docs search "reducer" --specs --max-results 5

Export raw session data for external processing::

    wink query bundle.zip --export-jsonl session > session.jsonl
"""
# pyright: reportImportCycles=false

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from . import wink

__all__ = ["wink"]


def __dir__() -> list[str]:
    return sorted({*globals().keys(), *(__all__)})


def __getattr__(name: str) -> object:
    if name == "wink":
        import importlib

        module = importlib.import_module(".wink", __name__)
        globals()["wink"] = module
        return module
    raise AttributeError(name)
