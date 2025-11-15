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

"""Command line entry points for the ``wink`` executable."""

from __future__ import annotations

import argparse
from collections.abc import Sequence

from ..runtime.logging import (
    StructuredLogPayload,
    configure_logging,
    get_logger,
)


def main(argv: Sequence[str] | None = None) -> int:
    """Run the wink CLI."""

    parser = _build_parser()
    args = parser.parse_args(list(argv) if argv is not None else None)

    configure_logging(level=args.log_level, json_mode=args.json_logs)
    logger = get_logger(__name__)

    logger.info(
        "wink CLI placeholder executed.",
        payload=StructuredLogPayload(event="wink.placeholder", context={}),
    )

    return 0


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="wink",
        description="Placeholder command line interface for future wink features.",
    )
    _ = parser.add_argument(
        "--log-level",
        choices=("CRITICAL", "ERROR", "WARNING", "INFO", "DEBUG", "NOTSET"),
        default=None,
        help="Override the log level emitted by the CLI.",
    )
    _ = parser.add_argument(
        "--json-logs",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Emit structured JSON logs (disable with --no-json-logs).",
    )

    return parser
