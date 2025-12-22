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

"""Tests for idempotency ledger and related components."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest

from weakincentives.dataclasses import FrozenDataclass
from weakincentives.prompt.tool import Tool, ToolContext, ToolResult
from weakincentives.runtime.idempotency import (
    EffectLedger,
    IdempotencyConfig,
    IdempotencyStrategy,
    ToolEffect,
    compute_idempotency_key,
    compute_params_hash,
)

# --- Test Data ---


@FrozenDataclass()
class SampleParams:
    """Sample tool parameters for testing."""

    file_path: str
    content: str = ""


@FrozenDataclass()
class SampleResult:
    """Sample tool result for testing."""

    message: str
    lines_written: int = 0

    def render(self) -> str:
        return f"{self.message} ({self.lines_written} lines)"


# --- IdempotencyStrategy Tests ---


class TestIdempotencyStrategy:
    def test_all_strategies_exist(self) -> None:
        assert IdempotencyStrategy.AUTO.value == "auto"
        assert IdempotencyStrategy.PARAMS.value == "params"
        assert IdempotencyStrategy.CUSTOM.value == "custom"
        assert IdempotencyStrategy.NONE.value == "none"


# --- IdempotencyConfig Tests ---


class TestIdempotencyConfig:
    def test_default_config(self) -> None:
        config = IdempotencyConfig()
        assert config.strategy == IdempotencyStrategy.AUTO
        assert config.param_keys == ()
        assert config.key_fn is None
        assert config.ttl == timedelta(hours=24)
        assert config.scope == "session"

    def test_string_strategy_normalized(self) -> None:
        config = IdempotencyConfig(strategy="auto")
        assert config.strategy == IdempotencyStrategy.AUTO

    def test_params_strategy_requires_param_keys(self) -> None:
        with pytest.raises(ValueError, match="requires param_keys"):
            IdempotencyConfig(strategy="params")

    def test_params_strategy_with_keys(self) -> None:
        config = IdempotencyConfig(strategy="params", param_keys=("file_path",))
        assert config.strategy == IdempotencyStrategy.PARAMS
        assert config.param_keys == ("file_path",)

    def test_custom_strategy_requires_key_fn(self) -> None:
        with pytest.raises(ValueError, match="requires key_fn"):
            IdempotencyConfig(strategy="custom")

    def test_custom_strategy_with_key_fn(self) -> None:
        def key_fn(p: SampleParams | None) -> str:
            return f"custom:{p.file_path}" if p else "custom:none"

        config = IdempotencyConfig(strategy="custom", key_fn=key_fn)
        assert config.strategy == IdempotencyStrategy.CUSTOM
        assert config.key_fn is key_fn

    def test_none_strategy(self) -> None:
        config = IdempotencyConfig(strategy="none")
        assert config.strategy == IdempotencyStrategy.NONE

    def test_custom_ttl(self) -> None:
        config = IdempotencyConfig(ttl=timedelta(minutes=30))
        assert config.ttl == timedelta(minutes=30)

    def test_no_ttl(self) -> None:
        config = IdempotencyConfig(ttl=None)
        assert config.ttl is None

    def test_custom_scope(self) -> None:
        config = IdempotencyConfig(scope="global")
        assert config.scope == "global"


# --- compute_params_hash Tests ---


class TestComputeParamsHash:
    def test_none_params(self) -> None:
        hash_val = compute_params_hash(None)
        assert isinstance(hash_val, str)
        assert len(hash_val) == 16  # Truncated length

    def test_dataclass_params(self) -> None:
        params = SampleParams(file_path="/test.txt", content="hello")
        hash_val = compute_params_hash(params)
        assert isinstance(hash_val, str)
        assert len(hash_val) == 16

    def test_same_params_same_hash(self) -> None:
        params1 = SampleParams(file_path="/test.txt", content="hello")
        params2 = SampleParams(file_path="/test.txt", content="hello")
        assert compute_params_hash(params1) == compute_params_hash(params2)

    def test_different_params_different_hash(self) -> None:
        params1 = SampleParams(file_path="/test.txt", content="hello")
        params2 = SampleParams(file_path="/test.txt", content="world")
        assert compute_params_hash(params1) != compute_params_hash(params2)

    def test_non_dataclass_uses_repr(self) -> None:
        # Non-dataclass object falls back to repr
        params = "not_a_dataclass"  # type: ignore[arg-type]
        hash_val = compute_params_hash(params)
        assert isinstance(hash_val, str)
        assert len(hash_val) == 16

    def test_unserializable_dataclass_uses_repr(self) -> None:
        # Dataclass with unserializable field falls back to repr
        from dataclasses import dataclass

        @dataclass
        class UnserializableParams:
            # A lambda can't be serialized
            callback: object

        params = UnserializableParams(callback=lambda x: x)
        hash_val = compute_params_hash(params)  # type: ignore[arg-type]
        assert isinstance(hash_val, str)
        assert len(hash_val) == 16


# --- compute_idempotency_key Tests ---


class TestComputeIdempotencyKey:
    def test_auto_strategy(self) -> None:
        config = IdempotencyConfig(strategy="auto")
        params = SampleParams(file_path="/test.txt")
        key = compute_idempotency_key(
            tool_name="write_file", params=params, config=config
        )
        assert key is not None
        assert key.startswith("session:write_file:")
        assert len(key.split(":")) == 3

    def test_auto_strategy_with_none_params(self) -> None:
        config = IdempotencyConfig(strategy="auto")
        key = compute_idempotency_key(tool_name="read_all", params=None, config=config)
        assert key is not None
        assert key.startswith("session:read_all:")

    def test_params_strategy(self) -> None:
        config = IdempotencyConfig(strategy="params", param_keys=("file_path",))
        params = SampleParams(file_path="/test.txt", content="hello")
        key = compute_idempotency_key(
            tool_name="write_file", params=params, config=config
        )
        assert key is not None
        assert key.startswith("session:write_file:")

    def test_params_strategy_ignores_other_fields(self) -> None:
        config = IdempotencyConfig(strategy="params", param_keys=("file_path",))
        params1 = SampleParams(file_path="/test.txt", content="hello")
        params2 = SampleParams(file_path="/test.txt", content="world")
        key1 = compute_idempotency_key(
            tool_name="write_file", params=params1, config=config
        )
        key2 = compute_idempotency_key(
            tool_name="write_file", params=params2, config=config
        )
        # Same key because only file_path is considered
        assert key1 == key2

    def test_params_strategy_with_none_params(self) -> None:
        config = IdempotencyConfig(strategy="params", param_keys=("file_path",))
        key = compute_idempotency_key(
            tool_name="write_file", params=None, config=config
        )
        assert key is not None
        assert key.startswith("session:write_file:")

    def test_params_strategy_with_non_dataclass(self) -> None:
        config = IdempotencyConfig(strategy="params", param_keys=("file_path",))
        key = compute_idempotency_key(
            tool_name="write_file",
            params="not_a_dataclass",  # type: ignore[arg-type]
            config=config,
        )
        assert key is not None
        assert key.startswith("session:write_file:")

    def test_custom_strategy(self) -> None:
        def key_fn(p: SampleParams | None) -> str:
            return f"order:{p.file_path}" if p else "order:none"

        config = IdempotencyConfig(strategy="custom", key_fn=key_fn)
        params = SampleParams(file_path="/test.txt")
        key = compute_idempotency_key(
            tool_name="write_file", params=params, config=config
        )
        assert key == "session:order:/test.txt"

    def test_none_strategy_returns_none(self) -> None:
        config = IdempotencyConfig(strategy="none")
        params = SampleParams(file_path="/test.txt")
        key = compute_idempotency_key(
            tool_name="write_file", params=params, config=config
        )
        assert key is None

    def test_custom_scope(self) -> None:
        config = IdempotencyConfig(strategy="auto", scope="global")
        params = SampleParams(file_path="/test.txt")
        key = compute_idempotency_key(
            tool_name="write_file", params=params, config=config
        )
        assert key is not None
        assert key.startswith("global:write_file:")

    def test_empty_scope(self) -> None:
        config = IdempotencyConfig(strategy="auto", scope="")
        params = SampleParams(file_path="/test.txt")
        key = compute_idempotency_key(
            tool_name="write_file", params=params, config=config
        )
        assert key is not None
        assert key.startswith("write_file:")


# --- ToolEffect Tests ---


class TestToolEffect:
    def test_create_tool_effect(self) -> None:
        now = datetime.now(UTC)
        effect = ToolEffect(
            idempotency_key="session:test:abc123",
            tool_name="test",
            params_hash="abc123",
            result_message="Success",
            result_value={"status": "ok"},
            result_success=True,
            created_at=now,
        )
        assert effect.idempotency_key == "session:test:abc123"
        assert effect.tool_name == "test"
        assert effect.result_message == "Success"
        assert effect.result_success is True
        assert effect.expires_at is None
        assert effect.effect_id is not None

    def test_effect_not_expired_without_expiry(self) -> None:
        now = datetime.now(UTC)
        effect = ToolEffect(
            idempotency_key="key",
            tool_name="test",
            params_hash="abc",
            result_message="ok",
            result_value=None,
            result_success=True,
            created_at=now,
            expires_at=None,
        )
        assert effect.is_expired() is False

    def test_effect_not_expired_before_expiry(self) -> None:
        now = datetime.now(UTC)
        effect = ToolEffect(
            idempotency_key="key",
            tool_name="test",
            params_hash="abc",
            result_message="ok",
            result_value=None,
            result_success=True,
            created_at=now,
            expires_at=now + timedelta(hours=1),
        )
        assert effect.is_expired(now) is False

    def test_effect_expired_after_expiry(self) -> None:
        now = datetime.now(UTC)
        past = now - timedelta(hours=1)
        effect = ToolEffect(
            idempotency_key="key",
            tool_name="test",
            params_hash="abc",
            result_message="ok",
            result_value=None,
            result_success=True,
            created_at=past,
            expires_at=past + timedelta(minutes=30),
        )
        assert effect.is_expired(now) is True


# --- EffectLedger Tests ---


class TestEffectLedger:
    def test_empty_ledger(self) -> None:
        ledger = EffectLedger()
        assert len(ledger) == 0
        assert ledger.lookup("nonexistent") is None

    def test_record_and_lookup(self) -> None:
        ledger = EffectLedger()
        result = ToolResult(message="Success", value=None, success=True)

        effect = ledger.record(
            idempotency_key="session:test:abc",
            tool_name="test",
            params=None,
            result=result,
        )

        assert len(ledger) == 1
        assert "session:test:abc" in ledger

        looked_up = ledger.lookup("session:test:abc")
        assert looked_up is not None
        assert looked_up.effect_id == effect.effect_id
        assert looked_up.result_message == "Success"
        assert looked_up.result_success is True

    def test_record_with_dataclass_params(self) -> None:
        ledger = EffectLedger()
        params = SampleParams(file_path="/test.txt", content="hello")
        result = ToolResult(
            message="Written", value=SampleResult("ok", 5), success=True
        )

        effect = ledger.record(
            idempotency_key="session:write:xyz",
            tool_name="write",
            params=params,
            result=result,
        )

        assert effect.params_hash == compute_params_hash(params)
        assert effect.result_value is not None

    def test_record_with_dict_value(self) -> None:
        ledger = EffectLedger()
        result = ToolResult(
            message="ok",
            value={"status": "success", "count": 42},  # type: ignore[arg-type]
            success=True,
        )

        effect = ledger.record(
            idempotency_key="key1",
            tool_name="test",
            params=None,
            result=result,
        )

        assert effect.result_value == {"status": "success", "count": 42}

    def test_record_with_list_value(self) -> None:
        ledger = EffectLedger()
        result = ToolResult(
            message="ok",
            value=["item1", "item2", "item3"],  # type: ignore[arg-type]
            success=True,
        )

        effect = ledger.record(
            idempotency_key="key2",
            tool_name="test",
            params=None,
            result=result,
        )

        assert effect.result_value == ["item1", "item2", "item3"]

    def test_record_with_unserializable_dataclass_value(self) -> None:
        from dataclasses import dataclass
        from unittest.mock import patch

        @dataclass
        class MockResult:
            data: str

        ledger = EffectLedger()
        result = ToolResult(
            message="ok",
            value=MockResult(data="test"),  # type: ignore[arg-type]
            success=True,
        )

        # Mock dump to raise TypeError
        with patch(
            "weakincentives.runtime.idempotency.dump",
            side_effect=TypeError("Cannot serialize"),
        ):
            effect = ledger.record(
                idempotency_key="key3",
                tool_name="test",
                params=None,
                result=result,
            )

        # Unserializable dataclass results in None
        assert effect.result_value is None

    def test_record_with_other_value_type(self) -> None:
        # Test value that is not None, dataclass, dict, or list
        ledger = EffectLedger()
        result = ToolResult(
            message="ok",
            value="just a string",  # type: ignore[arg-type]
            success=True,
        )

        effect = ledger.record(
            idempotency_key="key4",
            tool_name="test",
            params=None,
            result=result,
        )

        # Non-dict/list/dataclass values result in None
        assert effect.result_value is None

    def test_record_with_ttl(self) -> None:
        ledger = EffectLedger()
        now = datetime.now(UTC)
        result = ToolResult(message="ok", value=None, success=True)

        effect = ledger.record(
            idempotency_key="key",
            tool_name="test",
            params=None,
            result=result,
            ttl=timedelta(hours=1),
            now=now,
        )

        assert effect.expires_at == now + timedelta(hours=1)

    def test_lookup_expired_removes_entry(self) -> None:
        ledger = EffectLedger()
        past = datetime.now(UTC) - timedelta(hours=2)
        result = ToolResult(message="ok", value=None, success=True)

        ledger.record(
            idempotency_key="key",
            tool_name="test",
            params=None,
            result=result,
            ttl=timedelta(hours=1),
            now=past,
        )

        assert len(ledger) == 1
        looked_up = ledger.lookup("key")  # Uses current time
        assert looked_up is None
        assert len(ledger) == 0  # Entry was removed

    def test_invalidate(self) -> None:
        ledger = EffectLedger()
        result = ToolResult(message="ok", value=None, success=True)
        ledger.record(
            idempotency_key="key1",
            tool_name="test",
            params=None,
            result=result,
        )

        assert ledger.invalidate("key1") is True
        assert ledger.invalidate("key1") is False
        assert len(ledger) == 0

    def test_invalidate_by_tool(self) -> None:
        ledger = EffectLedger()
        result = ToolResult(message="ok", value=None, success=True)

        ledger.record(
            idempotency_key="key1", tool_name="tool_a", params=None, result=result
        )
        ledger.record(
            idempotency_key="key2", tool_name="tool_a", params=None, result=result
        )
        ledger.record(
            idempotency_key="key3", tool_name="tool_b", params=None, result=result
        )

        assert len(ledger) == 3
        removed = ledger.invalidate_by_tool("tool_a")
        assert removed == 2
        assert len(ledger) == 1
        assert "key3" in ledger

    def test_clear(self) -> None:
        ledger = EffectLedger()
        result = ToolResult(message="ok", value=None, success=True)

        ledger.record(
            idempotency_key="key1", tool_name="test", params=None, result=result
        )
        ledger.record(
            idempotency_key="key2", tool_name="test", params=None, result=result
        )

        assert len(ledger) == 2
        removed = ledger.clear()
        assert removed == 2
        assert len(ledger) == 0

    def test_prune_expired(self) -> None:
        ledger = EffectLedger()
        now = datetime.now(UTC)
        past = now - timedelta(hours=2)
        result = ToolResult(message="ok", value=None, success=True)

        # Add expired entry
        ledger.record(
            idempotency_key="expired",
            tool_name="test",
            params=None,
            result=result,
            ttl=timedelta(hours=1),
            now=past,
        )

        # Add valid entry
        ledger.record(
            idempotency_key="valid",
            tool_name="test",
            params=None,
            result=result,
            ttl=timedelta(hours=1),
            now=now,
        )

        assert len(ledger) == 2
        removed = ledger.prune_expired(now)
        assert removed == 1
        assert len(ledger) == 1
        assert "valid" in ledger
        assert "expired" not in ledger

    def test_contains(self) -> None:
        ledger = EffectLedger()
        result = ToolResult(message="ok", value=None, success=True)

        ledger.record(
            idempotency_key="key1", tool_name="test", params=None, result=result
        )

        assert "key1" in ledger
        assert "key2" not in ledger


# --- Tool with Idempotency Integration Tests ---


class TestToolWithIdempotency:
    def test_tool_without_idempotency(self) -> None:
        def handler(
            params: SampleParams, *, context: ToolContext
        ) -> ToolResult[SampleResult]:
            return ToolResult(message="ok", value=SampleResult("done", 1))

        tool = Tool[SampleParams, SampleResult](
            name="test_tool",
            description="A test tool",
            handler=handler,
        )

        assert tool.idempotency is None

    def test_tool_with_auto_idempotency(self) -> None:
        def handler(
            params: SampleParams, *, context: ToolContext
        ) -> ToolResult[SampleResult]:
            return ToolResult(message="ok", value=SampleResult("done", 1))

        tool = Tool[SampleParams, SampleResult](
            name="test_tool",
            description="A test tool",
            handler=handler,
            idempotency=IdempotencyConfig(strategy="auto"),
        )

        assert tool.idempotency is not None
        assert tool.idempotency.strategy == IdempotencyStrategy.AUTO

    def test_tool_with_params_idempotency(self) -> None:
        def handler(
            params: SampleParams, *, context: ToolContext
        ) -> ToolResult[SampleResult]:
            return ToolResult(message="ok", value=SampleResult("done", 1))

        tool = Tool[SampleParams, SampleResult](
            name="test_tool",
            description="A test tool",
            handler=handler,
            idempotency=IdempotencyConfig(strategy="params", param_keys=("file_path",)),
        )

        assert tool.idempotency is not None
        assert tool.idempotency.strategy == IdempotencyStrategy.PARAMS
        assert tool.idempotency.param_keys == ("file_path",)

    def test_tool_with_custom_idempotency(self) -> None:
        def handler(
            params: SampleParams, *, context: ToolContext
        ) -> ToolResult[SampleResult]:
            return ToolResult(message="ok", value=SampleResult("done", 1))

        def key_fn(p: SampleParams | None) -> str:
            return f"file:{p.file_path}" if p else "file:none"

        tool = Tool[SampleParams, SampleResult](
            name="test_tool",
            description="A test tool",
            handler=handler,
            idempotency=IdempotencyConfig(strategy="custom", key_fn=key_fn),
        )

        assert tool.idempotency is not None
        assert tool.idempotency.strategy == IdempotencyStrategy.CUSTOM
        assert tool.idempotency.key_fn is key_fn
