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

"""Property-based tests for dataclass serde."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Annotated

import pytest
from hypothesis import given, strategies as st

from weakincentives.dbc import dbc_enabled
from weakincentives.serde import dump, parse


@dataclass
class NestedNote:
    text: Annotated[str, {"min_length": 1, "strip": True}]
    tags: list[Annotated[str, {"regex": r"^tag-[a-z]+$"}]]


@dataclass
class NestedRecord:
    code: Annotated[str, {"regex": r"^REC-\d{3}$"}]
    note: NestedNote
    metrics: dict[str, Annotated[int | None, {"ge": 0, "le": 5}]]
    optional_notes: list[NestedNote | None]
    rating: Annotated[int | None, {"ge": 1, "le": 10}] = None


_tag_texts = st.text(
    alphabet=st.sampled_from(tuple("abcdefghijklmnopqrstuvwxyz")),
    min_size=1,
    max_size=8,
)


def note_strategy() -> st.SearchStrategy[NestedNote]:
    tag_strategy = st.from_regex(r"^tag-[a-z]+$", fullmatch=True)
    return st.builds(
        NestedNote,
        text=_tag_texts,
        tags=st.lists(tag_strategy, min_size=1, max_size=3),
    )


def record_strategy() -> st.SearchStrategy[NestedRecord]:
    return st.builds(
        NestedRecord,
        code=st.from_regex(r"^REC-\d{3}$", fullmatch=True),
        note=note_strategy(),
        metrics=st.dictionaries(
            st.sampled_from(["alpha", "beta", "gamma"]),
            st.one_of(st.integers(min_value=0, max_value=5), st.none()),
            min_size=1,
            max_size=3,
        ),
        optional_notes=st.lists(st.one_of(note_strategy(), st.none()), max_size=2),
        rating=st.one_of(st.integers(min_value=1, max_value=10), st.none()),
    )


@given(record_strategy())
def test_round_trip_with_nested_collections(record: NestedRecord) -> None:
    with dbc_enabled(False):
        payload = dump(record)
        restored = parse(NestedRecord, payload)
    assert restored == record


def test_nested_constraint_failures_include_paths() -> None:
    invalid_note = {
        "code": "REC-123",
        "note": {"text": " ", "tags": ["tag-good"]},
        "metrics": {"alpha": 1},
        "optional_notes": [],
    }
    with pytest.raises(ValueError) as note_exc:
        parse(NestedRecord, invalid_note)
    assert "note.text" in str(note_exc.value)

    invalid_metric = {
        "code": "REC-123",
        "note": {"text": "tagged", "tags": ["tag-good"]},
        "metrics": {"alpha": -1},
        "optional_notes": [],
    }
    with pytest.raises(ValueError) as metric_exc:
        parse(NestedRecord, invalid_metric)
    assert "metrics[alpha]" in str(metric_exc.value)

    invalid_rating = {
        "code": "REC-123",
        "note": {"text": "tagged", "tags": ["tag-good"]},
        "metrics": {"alpha": 1},
        "optional_notes": [],
        "rating": 42,
    }
    with pytest.raises(ValueError) as rating_exc:
        parse(NestedRecord, invalid_rating)
    assert "rating" in str(rating_exc.value)
