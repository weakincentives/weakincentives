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

"""Tests for transcript API endpoints: entries, facets, filtering, markdown."""

from __future__ import annotations

import json
from pathlib import Path

from fastapi.testclient import TestClient

from tests.cli.conftest import ExampleSlice
from weakincentives.cli import debug_app
from weakincentives.debug import BundleWriter
from weakincentives.debug.bundle import BundleConfig
from weakincentives.runtime.session import Session


def test_api_transcript_endpoint(tmp_path: Path) -> None:
    """Test the transcript API endpoint."""
    session = Session()
    session.dispatch(ExampleSlice("test"))

    with BundleWriter(tmp_path, config=BundleConfig()) as writer:
        writer.write_session_after(session)
        writer.write_request_input({})
        writer.write_request_output({})
        with writer.capture_logs():
            import logging

            transcript_logger = logging.getLogger("test.transcript")
            transcript_logger.setLevel(logging.DEBUG)
            transcript_logger.debug(
                "Transcript entry",
                extra={
                    "event": "transcript.entry",
                    "context": {
                        "prompt_name": "test",
                        "source": "main",
                        "entry_type": "user_message",
                        "sequence_number": 1,
                        "raw": json.dumps(
                            {
                                "type": "user",
                                "message": {"role": "user", "content": "Hi"},
                            }
                        ),
                        "detail": {
                            "type": "user",
                            "message": {"role": "user", "content": "Hi"},
                        },
                    },
                },
            )

    assert writer.path is not None
    logger = debug_app.get_logger("test.transcript.api")
    store = debug_app.BundleStore(writer.path, logger=logger)
    app = debug_app.build_debug_app(store, logger=logger)
    client = TestClient(app)

    meta_response = client.get("/api/meta")
    assert meta_response.json()["has_transcript"] is True

    transcript_response = client.get("/api/transcript")
    assert transcript_response.status_code == 200
    transcript = transcript_response.json()
    assert transcript["total"] >= 1
    assert transcript["entries"][0]["entry_type"] == "user_message"

    facets_response = client.get("/api/transcript/facets")
    assert facets_response.status_code == 200
    facets = facets_response.json()
    assert "sources" in facets


def test_api_transcript_markdown_rendering(tmp_path: Path) -> None:
    """Test that transcript entries with markdown content include rendered HTML."""
    session = Session()
    session.dispatch(ExampleSlice("test"))

    md_content = "# Heading\n\nSome **bold** text and a [link](http://example.com)."

    with BundleWriter(tmp_path, config=BundleConfig()) as writer:
        writer.write_session_after(session)
        writer.write_request_input({})
        writer.write_request_output({})
        with writer.capture_logs():
            import logging

            transcript_logger = logging.getLogger("test.transcript.md")
            transcript_logger.setLevel(logging.DEBUG)
            transcript_logger.debug(
                "Transcript entry",
                extra={
                    "event": "transcript.entry",
                    "context": {
                        "prompt_name": "test",
                        "source": "main",
                        "entry_type": "assistant_message",
                        "sequence_number": 1,
                        "raw": json.dumps(
                            {
                                "type": "assistant",
                                "message": {
                                    "role": "assistant",
                                    "content": md_content,
                                },
                            }
                        ),
                        "detail": {
                            "type": "assistant",
                            "message": {
                                "role": "assistant",
                                "content": md_content,
                            },
                        },
                    },
                },
            )

    assert writer.path is not None
    logger = debug_app.get_logger("test.transcript.markdown")
    store = debug_app.BundleStore(writer.path, logger=logger)
    app = debug_app.build_debug_app(store, logger=logger)
    client = TestClient(app)

    response = client.get("/api/transcript")
    assert response.status_code == 200
    data = response.json()
    entry = data["entries"][0]
    assert entry["content"] == md_content
    assert "content_html" in entry
    assert "<h1>" in entry["content_html"]
    assert "<strong>bold</strong>" in entry["content_html"]


def test_api_transcript_no_markdown_for_short_content(tmp_path: Path) -> None:
    """Test that short or plain text content does not get content_html."""
    session = Session()
    session.dispatch(ExampleSlice("test"))

    with BundleWriter(tmp_path, config=BundleConfig()) as writer:
        writer.write_session_after(session)
        writer.write_request_input({})
        writer.write_request_output({})
        with writer.capture_logs():
            import logging

            transcript_logger = logging.getLogger("test.transcript.plain")
            transcript_logger.setLevel(logging.DEBUG)
            transcript_logger.debug(
                "Transcript entry",
                extra={
                    "event": "transcript.entry",
                    "context": {
                        "prompt_name": "test",
                        "source": "main",
                        "entry_type": "user_message",
                        "sequence_number": 1,
                        "raw": json.dumps(
                            {
                                "type": "user",
                                "message": {"role": "user", "content": "Hi"},
                            }
                        ),
                        "detail": {
                            "type": "user",
                            "message": {"role": "user", "content": "Hi"},
                        },
                    },
                },
            )

    assert writer.path is not None
    logger = debug_app.get_logger("test.transcript.plain")
    store = debug_app.BundleStore(writer.path, logger=logger)
    app = debug_app.build_debug_app(store, logger=logger)
    client = TestClient(app)

    response = client.get("/api/transcript")
    assert response.status_code == 200
    data = response.json()
    entry = data["entries"][0]
    assert entry["content"] == "Hi"
    assert "content_html" not in entry


def test_transcript_endpoints_and_filters(tmp_path: Path) -> None:
    """Test transcript endpoints return entries and support filtering."""
    import logging

    session = Session()
    session.dispatch(ExampleSlice("test"))

    with BundleWriter(tmp_path, config=BundleConfig()) as writer:
        writer.write_session_after(session)
        writer.write_request_input({})
        writer.write_request_output({})
        with writer.capture_logs():
            transcript_logger = logging.getLogger("test.transcript")
            transcript_logger.setLevel(logging.DEBUG)

            transcript_logger.debug(
                "Transcript entry: user",
                extra={
                    "event": "transcript.entry",
                    "context": {
                        "prompt_name": "test-prompt",
                        "source": "main",
                        "entry_type": "user_message",
                        "sequence_number": 1,
                        "raw": json.dumps(
                            {
                                "type": "user",
                                "message": {"role": "user", "content": "Hello"},
                            }
                        ),
                        "detail": {
                            "type": "user",
                            "message": {"role": "user", "content": "Hello"},
                        },
                    },
                },
            )

            transcript_logger.debug(
                "Transcript entry: assistant",
                extra={
                    "event": "transcript.entry",
                    "context": {
                        "prompt_name": "test-prompt",
                        "source": "main",
                        "entry_type": "assistant_message",
                        "sequence_number": 2,
                        "raw": None,
                        "detail": {
                            "type": "assistant",
                            "message": {"role": "assistant", "content": "Hi"},
                        },
                    },
                },
            )

    assert writer.path is not None
    test_logger = debug_app.get_logger("test.transcript.api")
    store = debug_app.BundleStore(writer.path, logger=test_logger)
    app = debug_app.build_debug_app(store, logger=test_logger)
    client = TestClient(app)

    result = client.get("/api/transcript").json()
    assert result["total"] == 2
    assert len(result["entries"]) == 2
    assert result["entries"][0]["transcript_source"] == "main"

    facets = client.get("/api/transcript/facets").json()
    assert any(item["name"] == "main" for item in facets["sources"])
    assert any(item["name"] == "user_message" for item in facets["entry_types"])
    assert any(item["name"] == "assistant_message" for item in facets["entry_types"])

    # Search (server-side)
    searched = client.get("/api/transcript", params={"search": "Hello"}).json()
    assert searched["total"] == 1
    assert searched["entries"][0]["entry_type"] == "user_message"

    # Include filters
    filtered = client.get(
        "/api/transcript",
        params={"source": "main", "entry_type": "assistant_message"},
    ).json()
    assert filtered["total"] == 1
    assert filtered["entries"][0]["entry_type"] == "assistant_message"

    # Exclude filters
    excluded = client.get(
        "/api/transcript", params={"exclude_entry_type": "assistant_message"}
    ).json()
    assert excluded["total"] == 1
    assert excluded["entries"][0]["entry_type"] == "user_message"

    # Pagination branches
    limited = client.get("/api/transcript", params={"limit": 1}).json()
    assert limited["total"] == 2
    assert len(limited["entries"]) == 1

    offset_only = client.get("/api/transcript", params={"offset": 1}).json()
    assert offset_only["total"] == 2
    assert len(offset_only["entries"]) == 1
