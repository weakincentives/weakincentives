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

"""Tests for serde validators, converters, and hooks."""

from __future__ import annotations

import pytest

from tests.serde._fixtures import (
    AnnotationPrecedence,
    BadConvert,
    BadValidator,
    CompiledRegexCodeModel,
    HookModel,
    PostValidated,
    SingleValidatorModel,
    TransformModel,
    User,
    ValueErrorConvertModel,
    camel,
    user_payload,
)
from weakincentives.serde import clone, parse

pytestmark = pytest.mark.core


def test_parse_metadata_constraints_and_hooks() -> None:
    model = parse(HookModel, {"code": "  id-007  ", "amount": "8"})
    assert model.code == "ID-007-DONE"
    assert model.amount == 4


def test_parse_metadata_validator_errors() -> None:
    with pytest.raises(ValueError) as exc:
        parse(HookModel, {"code": "ID-100", "amount": "3"})
    assert "amount: must be even" in str(exc.value)

    with pytest.raises(TypeError) as exc2:
        parse(HookModel, {"code": "ID-100", "amount": "-4"})
    assert "amount: must be positive" in str(exc2.value)


@pytest.mark.parametrize(
    "model_cls,payload,expected_message",
    [
        (BadConvert, {"value": "3"}, "converter raised"),
        (BadValidator, {"value": "x"}, "validator raised"),
        (SingleValidatorModel, {"token": "oops"}, "token: bad prefix"),
    ],
)
def test_parse_converter_and_validator_exception_wrapping(
    model_cls: type[object], payload: dict[str, object], expected_message: str
) -> None:
    with pytest.raises(Exception) as exc:
        parse(model_cls, payload)
    assert expected_message in str(exc.value)


def test_parse_annotation_precedence_and_transform() -> None:
    model = parse(AnnotationPrecedence, {"code": " abc "})
    assert model.code == "ABC"

    transformed = parse(TransformModel, {"token": " data "})
    assert transformed.token == "data!"

    single = parse(SingleValidatorModel, {"token": "okay"})
    assert single.token == "okay"


def test_parse_model_validator_runs() -> None:
    with pytest.raises(ValueError) as exc:
        parse(
            User,
            user_payload(AGE="12"),
            aliases={"user_id": "USER"},
            alias_generator=camel,
            case_insensitive=True,
        )
    assert str(exc.value) == "age must be >= 13"


def test_post_validate_hook_runs() -> None:
    instance = parse(PostValidated, {"value": 1})
    assert instance.value == 1
    with pytest.raises(ValueError):
        parse(PostValidated, {"value": -1})
    with pytest.raises(ValueError):
        clone(instance, value=-5)


def test_compiled_regex_and_converter_errors() -> None:
    assert parse(CompiledRegexCodeModel, {"code": "ABC"}).code == "ABC"
    with pytest.raises(ValueError):
        parse(CompiledRegexCodeModel, {"code": "abc"})

    with pytest.raises(ValueError) as exc:
        parse(ValueErrorConvertModel, {"value": "1"})
    assert "value: boom" in str(exc.value)
