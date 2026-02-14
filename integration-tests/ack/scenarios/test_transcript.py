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

"""Tier 2 ACK scenarios for transcript envelope and lifecycle validation."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import cast

import pytest

from weakincentives.adapters.core import ProviderAdapter
from weakincentives.debug import BundleConfig, BundleWriter, DebugBundle
from weakincentives.prompt import Prompt
from weakincentives.runtime.session import Session
from weakincentives.runtime.transcript import CANONICAL_ENTRY_TYPES

from ..adapters import AdapterFixture
from . import (
    GreetingParams,
    TransformRequest,
    build_greeting_prompt,
    build_tool_prompt,
    build_uppercase_tool,
    make_adapter_ns,
)
from ._transcript_helpers import (
    assert_envelope_complete,
    assert_sequence_monotonic,
    assert_tool_use_before_result,
    collect_transcript_entries,
)

pytestmark = pytest.mark.ack_capability("transcript")


def _enable_transcript_logging(caplog: pytest.LogCaptureFixture) -> None:
    caplog.clear()
    caplog.set_level(logging.DEBUG, logger="weakincentives.runtime.transcript")


def test_transcript_contains_user_message(
    adapter: ProviderAdapter[object],
    session: Session,
    adapter_fixture: AdapterFixture,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Transcript contains at least one user_message entry."""
    _enable_transcript_logging(caplog)
    prompt = Prompt(
        build_greeting_prompt(make_adapter_ns(adapter_fixture.adapter_name))
    ).bind(GreetingParams(audience="ack transcript user message"))

    _ = adapter.evaluate(prompt, session=session)

    entries = collect_transcript_entries(caplog)
    user_entries = [
        entry for entry in entries if entry.get("entry_type") == "user_message"
    ]
    assert user_entries


def test_transcript_contains_assistant_message(
    adapter: ProviderAdapter[object],
    session: Session,
    adapter_fixture: AdapterFixture,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Transcript contains at least one assistant_message entry."""
    _enable_transcript_logging(caplog)
    prompt = Prompt(
        build_greeting_prompt(make_adapter_ns(adapter_fixture.adapter_name))
    ).bind(GreetingParams(audience="ack transcript assistant message"))

    _ = adapter.evaluate(prompt, session=session)

    entries = collect_transcript_entries(caplog)
    assistant_entries = [
        entry for entry in entries if entry.get("entry_type") == "assistant_message"
    ]
    assert assistant_entries


@pytest.mark.ack_capability("tool_invocation")
def test_transcript_tool_use_before_tool_result(
    adapter: ProviderAdapter[object],
    session: Session,
    adapter_fixture: AdapterFixture,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """tool_use entries appear before tool_result entries."""
    _enable_transcript_logging(caplog)

    tool = build_uppercase_tool()
    prompt = Prompt(
        build_tool_prompt(make_adapter_ns(adapter_fixture.adapter_name), tool)
    ).bind(TransformRequest(text="hello"))

    _ = adapter.evaluate(prompt, session=session)

    entries = collect_transcript_entries(caplog)
    assert_tool_use_before_result(entries, "uppercase_text")


def test_transcript_envelope_completeness(
    adapter: ProviderAdapter[object],
    session: Session,
    adapter_fixture: AdapterFixture,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Each transcript entry contains the required canonical envelope."""
    _enable_transcript_logging(caplog)

    prompt = Prompt(
        build_greeting_prompt(make_adapter_ns(adapter_fixture.adapter_name))
    ).bind(GreetingParams(audience="ack transcript envelope"))

    _ = adapter.evaluate(prompt, session=session)

    entries = collect_transcript_entries(caplog)
    assert entries
    for entry in entries:
        assert_envelope_complete(entry)


def test_transcript_sequence_monotonicity(
    adapter: ProviderAdapter[object],
    session: Session,
    adapter_fixture: AdapterFixture,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Sequence numbers are strictly increasing per transcript source."""
    _enable_transcript_logging(caplog)

    prompt = Prompt(
        build_greeting_prompt(make_adapter_ns(adapter_fixture.adapter_name))
    ).bind(GreetingParams(audience="ack transcript sequence"))

    _ = adapter.evaluate(prompt, session=session)

    entries = collect_transcript_entries(caplog)
    assert entries

    sources = sorted({cast(str, entry["source"]) for entry in entries})
    for source in sources:
        assert_sequence_monotonic(entries, source)


def test_transcript_canonical_types_only(
    adapter: ProviderAdapter[object],
    session: Session,
    adapter_fixture: AdapterFixture,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """All entries use canonical transcript entry types."""
    _enable_transcript_logging(caplog)

    prompt = Prompt(
        build_greeting_prompt(make_adapter_ns(adapter_fixture.adapter_name))
    ).bind(GreetingParams(audience="ack transcript canonical types"))

    _ = adapter.evaluate(prompt, session=session)

    entries = collect_transcript_entries(caplog)
    assert entries
    assert all(entry.get("entry_type") in CANONICAL_ENTRY_TYPES for entry in entries)


def test_transcript_adapter_label(
    adapter: ProviderAdapter[object],
    session: Session,
    adapter_fixture: AdapterFixture,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Every transcript entry is attributed to the current adapter."""
    _enable_transcript_logging(caplog)

    prompt = Prompt(
        build_greeting_prompt(make_adapter_ns(adapter_fixture.adapter_name))
    ).bind(GreetingParams(audience="ack transcript adapter label"))

    _ = adapter.evaluate(prompt, session=session)

    entries = collect_transcript_entries(caplog)
    assert entries
    assert all(
        entry.get("adapter") == adapter_fixture.adapter_name for entry in entries
    )


def test_transcript_start_stop_events(
    adapter: ProviderAdapter[object],
    session: Session,
    adapter_fixture: AdapterFixture,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Adapters emit transcript.start and transcript.stop around entries."""
    _enable_transcript_logging(caplog)

    prompt = Prompt(
        build_greeting_prompt(make_adapter_ns(adapter_fixture.adapter_name))
    ).bind(GreetingParams(audience="ack transcript lifecycle"))

    _ = adapter.evaluate(prompt, session=session)

    transcript_records = [
        record
        for record in caplog.records
        if record.name == "weakincentives.runtime.transcript"
    ]
    events = [getattr(record, "event", "") for record in transcript_records]

    assert "transcript.start" in events
    assert "transcript.entry" in events
    assert "transcript.stop" in events

    first_start = events.index("transcript.start")
    first_entry = events.index("transcript.entry")
    last_stop = len(events) - 1 - events[::-1].index("transcript.stop")
    assert first_start < first_entry < last_stop

    stop_record = next(
        record
        for record in transcript_records
        if getattr(record, "event", "") == "transcript.stop"
    )
    stop_context = getattr(stop_record, "context", {})
    assert isinstance(stop_context, dict)
    assert "total_entries" in stop_context
    assert "entries_by_source" in stop_context
    assert "entries_by_type" in stop_context


def test_transcript_in_debug_bundle(
    adapter: ProviderAdapter[object],
    session: Session,
    adapter_fixture: AdapterFixture,
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Debug bundles contain transcript.jsonl entries in order."""
    _enable_transcript_logging(caplog)

    prompt = Prompt(
        build_greeting_prompt(make_adapter_ns(adapter_fixture.adapter_name))
    ).bind(GreetingParams(audience="ack transcript bundle"))

    bundle_dir = tmp_path / "bundles"
    bundle_dir.mkdir()

    with BundleWriter(
        target=bundle_dir,
        config=BundleConfig(target=bundle_dir),
        trigger="ack_test",
    ) as writer:
        writer.set_prompt_info(
            ns=prompt.ns,
            key=prompt.key,
            adapter=adapter_fixture.adapter_name,
        )
        with writer.capture_logs():
            _ = adapter.evaluate(prompt, session=session)
        writer.write_session_after(session)

    bundle = DebugBundle.load(writer.path)
    bundle_files = bundle.list_files()
    assert "transcript.jsonl" in bundle_files

    caplog_entries = collect_transcript_entries(caplog)
    assert caplog_entries

    bundle_records = bundle.transcript
    bundle_entries: list[dict[str, object]] = []
    for record in bundle_records:
        context = record.get("context")
        if isinstance(context, dict):
            bundle_entries.append(cast("dict[str, object]", context))

    assert bundle_entries

    caplog_projection = [
        (
            cast(str, entry["source"]),
            cast(int, entry["sequence_number"]),
            cast(str, entry["entry_type"]),
        )
        for entry in caplog_entries
    ]
    bundle_projection = [
        (
            cast(str, entry["source"]),
            cast(int, entry["sequence_number"]),
            cast(str, entry["entry_type"]),
        )
        for entry in bundle_entries
    ]

    assert bundle_projection == caplog_projection
