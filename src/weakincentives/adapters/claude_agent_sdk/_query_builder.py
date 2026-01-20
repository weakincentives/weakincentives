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

"""Builder pattern for SDK query options.

This module provides a fluent builder for constructing SDK query options,
extracting the options-building logic from _run_sdk_query() for better
testability and separation of concerns.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Self

from ...dataclasses import FrozenDataclass

if TYPE_CHECKING:
    from .config import ClaudeAgentSDKClientConfig, ClaudeAgentSDKModelConfig
    from .isolation import EphemeralHome

__all__ = [
    "SdkQueryBuilder",
    "SdkQueryOptions",
]


@FrozenDataclass()
class SdkQueryOptions:
    """Immutable SDK query configuration.

    Contains all options needed to execute an SDK query. This is the
    result of building options with SdkQueryBuilder.

    Attributes:
        model: Model identifier for the SDK query.
        cwd: Working directory for SDK operations.
        permission_mode: Tool permission handling mode.
        max_turns: Maximum number of conversation turns.
        max_budget_usd: Maximum budget in USD for the session.
        betas: Beta features to enable.
        output_format: Structured output format specification.
        allowed_tools: Tuple of allowed tool names.
        disallowed_tools: Tuple of disallowed tool names.
        env: Environment variables for the SDK subprocess.
        setting_sources: Sources for SDK settings.
        max_thinking_tokens: Maximum tokens for extended thinking.
        mcp_servers: MCP server configurations.
        stderr_handler: Handler function for stderr output.
        hooks: Hook configuration dict for SDK.
    """

    model: str
    cwd: str | None = None
    permission_mode: str | None = None
    max_turns: int | None = None
    max_budget_usd: float | None = None
    betas: tuple[str, ...] = ()
    output_format: dict[str, Any] | None = None
    allowed_tools: tuple[str, ...] | None = None
    disallowed_tools: tuple[str, ...] = ()
    env: Mapping[str, str] = field(default_factory=dict)
    setting_sources: tuple[str, ...] = ()
    max_thinking_tokens: int | None = None
    mcp_servers: Mapping[str, Any] = field(default_factory=dict)
    stderr_handler: Callable[[str], None] | None = None
    hooks: dict[str, list[Any]] = field(default_factory=dict)

    def to_kwargs(self) -> dict[str, Any]:
        """Convert options to SDK kwargs dict.

        Returns:
            Dictionary suitable for passing to ClaudeAgentOptions.
        """
        kwargs: dict[str, Any] = {"model": self.model}
        self._add_basic_options(kwargs)
        self._add_tool_options(kwargs)
        self._add_advanced_options(kwargs)
        return kwargs

    def _add_basic_options(self, kwargs: dict[str, Any]) -> None:
        """Add basic configuration options to kwargs."""
        if self.cwd:
            kwargs["cwd"] = self.cwd
        if self.permission_mode:
            kwargs["permission_mode"] = self.permission_mode
        if self.max_turns:
            kwargs["max_turns"] = self.max_turns
        if self.max_budget_usd is not None:
            kwargs["max_budget_usd"] = self.max_budget_usd
        if self.betas:
            kwargs["betas"] = list(self.betas)
        if self.output_format:
            kwargs["output_format"] = self.output_format

    def _add_tool_options(self, kwargs: dict[str, Any]) -> None:
        """Add tool-related options to kwargs."""
        if self.allowed_tools is not None:
            kwargs["allowed_tools"] = list(self.allowed_tools)
        if self.disallowed_tools:
            kwargs["disallowed_tools"] = list(self.disallowed_tools)
        if self.mcp_servers:
            kwargs["mcp_servers"] = dict(self.mcp_servers)

    def _add_advanced_options(self, kwargs: dict[str, Any]) -> None:
        """Add advanced configuration options to kwargs."""
        if self.env:
            kwargs["env"] = dict(self.env)
        if self.setting_sources:
            kwargs["setting_sources"] = list(self.setting_sources)
        if self.max_thinking_tokens is not None:
            kwargs["max_thinking_tokens"] = self.max_thinking_tokens
        if self.stderr_handler is not None:
            kwargs["stderr"] = self.stderr_handler
        if self.hooks:
            kwargs["hooks"] = self.hooks


@dataclass(slots=True)
class SdkQueryBuilder:
    """Builder for SDK query options with validation.

    Provides a fluent API for constructing SDK query options from
    various configuration sources. Each with_* method returns self
    for method chaining.

    Example:
        >>> builder = SdkQueryBuilder(model="claude-sonnet-4-5-20250929")
        >>> options = (
        ...     builder
        ...     .with_client_config(client_config)
        ...     .with_model_config(model_config)
        ...     .with_ephemeral_home(ephemeral_home)
        ...     .with_hooks(hooks_dict)
        ...     .build()
        ... )
    """

    _model: str
    _cwd: str | None = None
    _permission_mode: str | None = None
    _max_turns: int | None = None
    _max_budget_usd: float | None = None
    _betas: tuple[str, ...] = ()
    _output_format: dict[str, Any] | None = None
    _allowed_tools: tuple[str, ...] | None = None
    _disallowed_tools: tuple[str, ...] = ()
    _env: dict[str, str] = field(default_factory=dict)
    _setting_sources: tuple[str, ...] = ()
    _max_thinking_tokens: int | None = None
    _mcp_servers: dict[str, Any] = field(default_factory=dict)
    _stderr_handler: Callable[[str], None] | None = None
    _hooks: dict[str, list[Any]] = field(default_factory=dict)

    def with_cwd(self, cwd: str | None) -> Self:
        """Set the working directory.

        Args:
            cwd: Working directory path, or None for default.

        Returns:
            Self for method chaining.
        """
        self._cwd = cwd
        return self

    def with_client_config(self, config: ClaudeAgentSDKClientConfig) -> Self:
        """Apply client configuration options.

        Extracts relevant options from ClaudeAgentSDKClientConfig and
        applies them to the builder state.

        Args:
            config: Client configuration to apply.

        Returns:
            Self for method chaining.
        """
        if config.permission_mode:
            self._permission_mode = config.permission_mode

        if config.max_turns:
            self._max_turns = config.max_turns

        if config.max_budget_usd is not None:
            self._max_budget_usd = config.max_budget_usd

        if config.betas:
            self._betas = config.betas

        return self

    def with_model_config(self, config: ClaudeAgentSDKModelConfig) -> Self:
        """Apply model configuration.

        Extracts relevant options from ClaudeAgentSDKModelConfig and
        applies them to the builder state.

        Args:
            config: Model configuration to apply.

        Returns:
            Self for method chaining.
        """
        if config.max_thinking_tokens is not None:
            self._max_thinking_tokens = config.max_thinking_tokens

        return self

    def with_ephemeral_home(self, ephemeral_home: EphemeralHome) -> Self:
        """Apply isolation configuration from ephemeral home.

        Sets environment variables and setting sources from the
        ephemeral home configuration.

        Args:
            ephemeral_home: Ephemeral home with isolation config.

        Returns:
            Self for method chaining.
        """
        self._env = dict(ephemeral_home.get_env())
        self._setting_sources = tuple(ephemeral_home.get_setting_sources())
        return self

    def with_output_format(self, output_format: dict[str, Any] | None) -> Self:
        """Set the structured output format.

        Args:
            output_format: Output format specification, or None.

        Returns:
            Self for method chaining.
        """
        self._output_format = output_format
        return self

    def with_tool_constraints(
        self,
        *,
        allowed_tools: tuple[str, ...] | None = None,
        disallowed_tools: tuple[str, ...] = (),
    ) -> Self:
        """Set tool access constraints.

        Args:
            allowed_tools: Tuple of allowed tool names, or None for all.
            disallowed_tools: Tuple of disallowed tool names.

        Returns:
            Self for method chaining.
        """
        self._allowed_tools = allowed_tools
        self._disallowed_tools = disallowed_tools
        return self

    def with_mcp_server(self, name: str, config: object) -> Self:
        """Register an MCP server configuration.

        Args:
            name: Server name identifier.
            config: MCP server configuration.

        Returns:
            Self for method chaining.
        """
        self._mcp_servers[name] = config
        return self

    def with_stderr_handler(self, handler: Callable[[str], None] | None) -> Self:
        """Set the stderr handler.

        Args:
            handler: Function to handle stderr lines, or None.

        Returns:
            Self for method chaining.
        """
        self._stderr_handler = handler
        return self

    def with_hooks(self, hooks: dict[str, list[Any]]) -> Self:
        """Set the complete hooks configuration.

        Args:
            hooks: Hook configuration dict for SDK.

        Returns:
            Self for method chaining.
        """
        self._hooks = hooks
        return self

    def build(self) -> SdkQueryOptions:
        """Build immutable options from current state.

        Returns:
            SdkQueryOptions with all configured values.
        """
        return SdkQueryOptions(
            model=self._model,
            cwd=self._cwd,
            permission_mode=self._permission_mode,
            max_turns=self._max_turns,
            max_budget_usd=self._max_budget_usd,
            betas=self._betas,
            output_format=self._output_format,
            allowed_tools=self._allowed_tools,
            disallowed_tools=self._disallowed_tools,
            env=self._env,
            setting_sources=self._setting_sources,
            max_thinking_tokens=self._max_thinking_tokens,
            mcp_servers=self._mcp_servers,
            stderr_handler=self._stderr_handler,
            hooks=self._hooks,
        )
