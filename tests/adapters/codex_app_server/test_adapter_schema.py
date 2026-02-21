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

"""Tests for schema, adapter init, and structured output parsing."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from weakincentives.adapters.codex_app_server._response import (
    parse_structured_output_or_raise,
)
from weakincentives.adapters.codex_app_server._schema import (
    bridged_tools_to_dynamic_specs,
    openai_strict_schema,
)
from weakincentives.adapters.codex_app_server.adapter import (
    CODEX_APP_SERVER_ADAPTER_NAME,
    CodexAppServerAdapter,
)
from weakincentives.adapters.codex_app_server.config import (
    CodexAppServerClientConfig,
    CodexAppServerModelConfig,
)
from weakincentives.adapters.core import PromptEvaluationError

from .conftest import (
    make_mock_client,
    make_session,
    make_simple_prompt,
    messages_iterator,
)


class TestAdapterName:
    def test_name_constant(self) -> None:
        assert CODEX_APP_SERVER_ADAPTER_NAME == "codex_app_server"


class TestBridgedToolsToDynamicSpecs:
    def test_empty(self) -> None:
        assert bridged_tools_to_dynamic_specs(()) == []

    def test_converts_tools(self) -> None:
        tool = MagicMock()
        tool.name = "my_tool"
        tool.description = "Does something"
        tool.input_schema = {"type": "object", "properties": {}}
        specs = bridged_tools_to_dynamic_specs((tool,))
        assert len(specs) == 1
        assert specs[0]["name"] == "my_tool"
        assert specs[0]["description"] == "Does something"
        assert specs[0]["inputSchema"] == {"type": "object", "properties": {}}


class TestOpenaiStrictSchema:
    def test_sets_additional_properties_false(self) -> None:
        s = {
            "type": "object",
            "properties": {"a": {"type": "string"}},
            "additionalProperties": True,
        }
        result = openai_strict_schema(s)
        assert result["additionalProperties"] is False

    def test_all_properties_required(self) -> None:
        s = {
            "type": "object",
            "properties": {"a": {"type": "string"}, "b": {"type": "integer"}},
            "required": ["a"],
        }
        result = openai_strict_schema(s)
        assert sorted(result["required"]) == ["a", "b"]

    def test_nested_objects(self) -> None:
        s = {
            "type": "object",
            "properties": {
                "inner": {
                    "type": "object",
                    "properties": {"x": {"type": "integer"}},
                    "additionalProperties": True,
                }
            },
            "additionalProperties": True,
        }
        result = openai_strict_schema(s)
        assert result["additionalProperties"] is False
        assert result["properties"]["inner"]["additionalProperties"] is False
        assert result["properties"]["inner"]["required"] == ["x"]

    def test_non_object_unchanged(self) -> None:
        s = {"type": "array", "items": {"type": "string"}}
        result = openai_strict_schema(s)
        assert result["type"] == "array"
        assert result["items"] == {"type": "string"}

    def test_object_without_properties(self) -> None:
        s = {"type": "object", "additionalProperties": True}
        result = openai_strict_schema(s)
        assert result["additionalProperties"] is False
        assert "required" not in result

    def test_preserves_other_fields(self) -> None:
        s = {"type": "object", "title": "Foo", "properties": {"a": {"type": "string"}}}
        result = openai_strict_schema(s)
        assert result["title"] == "Foo"
        assert result["required"] == ["a"]

    def test_array_items_strictified(self) -> None:
        s = {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {"x": {"type": "integer"}},
                "additionalProperties": True,
            },
        }
        result = openai_strict_schema(s)
        assert result["items"]["additionalProperties"] is False
        assert result["items"]["required"] == ["x"]

    def test_anyof_strictified(self) -> None:
        s = {
            "anyOf": [
                {
                    "type": "object",
                    "properties": {"a": {"type": "string"}},
                    "additionalProperties": True,
                },
                {"type": "string"},
            ]
        }
        result = openai_strict_schema(s)
        assert result["anyOf"][0]["additionalProperties"] is False
        assert result["anyOf"][0]["required"] == ["a"]
        assert result["anyOf"][1] == {"type": "string"}

    def test_oneof_strictified(self) -> None:
        s = {
            "oneOf": [
                {
                    "type": "object",
                    "properties": {"b": {"type": "integer"}},
                    "additionalProperties": True,
                },
            ]
        }
        result = openai_strict_schema(s)
        assert result["oneOf"][0]["additionalProperties"] is False

    def test_allof_strictified(self) -> None:
        s = {
            "allOf": [
                {
                    "type": "object",
                    "properties": {"c": {"type": "boolean"}},
                    "additionalProperties": True,
                },
            ]
        }
        result = openai_strict_schema(s)
        assert result["allOf"][0]["additionalProperties"] is False

    def test_defs_strictified(self) -> None:
        s = {
            "type": "object",
            "properties": {"ref": {"$ref": "#/$defs/Inner"}},
            "$defs": {
                "Inner": {
                    "type": "object",
                    "properties": {"val": {"type": "string"}},
                    "additionalProperties": True,
                }
            },
        }
        result = openai_strict_schema(s)
        assert result["$defs"]["Inner"]["additionalProperties"] is False
        assert result["$defs"]["Inner"]["required"] == ["val"]

    def test_definitions_strictified(self) -> None:
        s = {
            "definitions": {
                "Foo": {
                    "type": "object",
                    "properties": {"z": {"type": "number"}},
                    "additionalProperties": True,
                }
            }
        }
        result = openai_strict_schema(s)
        assert result["definitions"]["Foo"]["additionalProperties"] is False

    def test_deeply_nested_array_of_objects(self) -> None:
        s = {
            "type": "object",
            "properties": {
                "items": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "nested": {
                                "type": "array",
                                "items": {
                                    "type": "object",
                                    "properties": {"v": {"type": "integer"}},
                                    "additionalProperties": True,
                                },
                            }
                        },
                        "additionalProperties": True,
                    },
                }
            },
            "additionalProperties": True,
        }
        result = openai_strict_schema(s)
        # Top-level object
        assert result["additionalProperties"] is False
        # Array items object
        items_obj = result["properties"]["items"]["items"]
        assert items_obj["additionalProperties"] is False
        assert items_obj["required"] == ["nested"]
        # Deeply nested array items object
        deep_obj = items_obj["properties"]["nested"]["items"]
        assert deep_obj["additionalProperties"] is False
        assert deep_obj["required"] == ["v"]


class TestAdapterInit:
    def test_defaults(self) -> None:
        adapter = CodexAppServerAdapter()
        assert adapter._model_config.model == "gpt-5.3-codex"
        assert adapter._client_config.codex_bin == "codex"

    def test_custom_config(self) -> None:
        model_cfg = CodexAppServerModelConfig(model="o3", effort="high")
        client_cfg = CodexAppServerClientConfig(codex_bin="/usr/bin/codex")
        adapter = CodexAppServerAdapter(
            model_config=model_cfg, client_config=client_cfg
        )
        assert adapter._model_config.model == "o3"
        assert adapter._client_config.codex_bin == "/usr/bin/codex"

    def test_adapter_name_property(self) -> None:
        adapter = CodexAppServerAdapter()
        assert adapter.adapter_name == "codex_app_server"


class TestParseStructuredOutput:
    def test_valid_json(self) -> None:
        from weakincentives.prompt.rendering import RenderedPrompt as RP
        from weakincentives.prompt.structured_output import StructuredOutputConfig

        @dataclass(slots=True, frozen=True)
        class Result:
            answer: int

        rendered = RP(
            text="",
            structured_output=StructuredOutputConfig(
                dataclass_type=Result,
                container="object",
                allow_extra_keys=False,
            ),
        )

        result = parse_structured_output_or_raise('{"answer": 42}', rendered, "test")
        assert result is not None
        assert result.answer == 42

    def test_invalid_json_raises(self) -> None:
        from weakincentives.prompt.rendering import RenderedPrompt as RP
        from weakincentives.prompt.structured_output import StructuredOutputConfig

        @dataclass(slots=True, frozen=True)
        class Dummy:
            x: int

        rendered = RP(
            text="",
            structured_output=StructuredOutputConfig(
                dataclass_type=Dummy,
                container="object",
                allow_extra_keys=False,
            ),
        )

        with pytest.raises(PromptEvaluationError, match="parse structured"):
            parse_structured_output_or_raise("not json", rendered, "test")

    def test_array_container_parsed(self) -> None:
        from weakincentives.prompt.rendering import RenderedPrompt as RP
        from weakincentives.prompt.structured_output import StructuredOutputConfig

        @dataclass(slots=True, frozen=True)
        class Item:
            value: int

        rendered = RP(
            text="",
            structured_output=StructuredOutputConfig(
                dataclass_type=Item,
                container="array",
                allow_extra_keys=False,
            ),
        )

        # Array wrapper format: {"items": [...]}
        text = '{"items": [{"value": 1}, {"value": 2}]}'
        result = parse_structured_output_or_raise(text, rendered, "test")
        assert isinstance(result, list)
        assert len(result) == 2
        assert result[0].value == 1
        assert result[1].value == 2


class TestArraySchemaWrapping:
    def test_array_container_wraps_schema(self) -> None:
        """When container='array', the output schema wraps element in items."""
        from weakincentives.prompt.rendering import RenderedPrompt as RP
        from weakincentives.prompt.structured_output import StructuredOutputConfig

        @dataclass(slots=True, frozen=True)
        class Item:
            value: int

        adapter = CodexAppServerAdapter(
            client_config=CodexAppServerClientConfig(cwd="/tmp/test"),
        )
        session, _ = make_session()
        prompt = make_simple_prompt()

        messages = [
            {
                "method": "item/completed",
                "params": {
                    "item": {
                        "type": "agentMessage",
                        "text": '{"items": [{"value": 1}]}',
                    }
                },
            },
            {
                "method": "turn/completed",
                "params": {"turn": {"status": "completed"}},
            },
        ]

        original_render = prompt.render

        def patched_render(**kwargs: Any) -> RP[Any]:
            rendered = original_render(**kwargs)
            return RP(
                text=rendered.text,
                structured_output=StructuredOutputConfig(
                    dataclass_type=Item,
                    container="array",
                    allow_extra_keys=False,
                ),
                _tools=rendered.tools,
            )

        with (
            patch(
                "weakincentives.adapters.codex_app_server.adapter.CodexAppServerClient"
            ) as MockClient,
            patch.object(prompt, "render", side_effect=patched_render),
        ):
            mock_client = make_mock_client()
            mock_client.send_request.side_effect = [
                {"capabilities": {}},
                {"thread": {"id": "t-1"}},
                {"turn": {"id": 1}},
            ]
            mock_client.read_messages.return_value = messages_iterator(messages)
            MockClient.return_value = mock_client

            result = adapter.evaluate(prompt, session=session)

        assert result.output is not None
        assert isinstance(result.output, list)
        assert result.output[0].value == 1

        # Verify outputSchema was wrapped with items array
        turn_call = mock_client.send_request.call_args_list[2]
        output_schema = turn_call[0][1]["outputSchema"]
        assert output_schema["type"] == "object"
        assert "items" in output_schema["properties"]
        assert output_schema["properties"]["items"]["type"] == "array"
