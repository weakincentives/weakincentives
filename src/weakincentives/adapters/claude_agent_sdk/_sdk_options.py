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

"""SDK options and hooks configuration for Claude Agent SDK adapter.

Functions for building ClaudeAgentOptions kwargs and hooks configuration
dictionaries from adapter configuration state.
"""

from __future__ import annotations

import inspect
from collections.abc import Callable
from typing import Any

from ...runtime.logging import StructuredLogger, get_logger
from ._bridge import create_mcp_server
from ._ephemeral_home import EphemeralHome
from ._hooks import (
    HookContext,
    create_post_tool_use_hook,
    create_pre_compact_hook,
    create_pre_tool_use_hook,
    create_stop_hook,
    create_subagent_stop_hook,
    create_task_completion_stop_hook,
    create_user_prompt_submit_hook,
)
from ._transcript_collector import TranscriptCollector
from .config import ClaudeAgentSDKClientConfig, ClaudeAgentSDKModelConfig

logger: StructuredLogger = get_logger(
    __name__, context={"component": "claude_agent_sdk"}
)


def create_stderr_handler(
    stderr_buffer: list[str],
) -> Callable[[str], None]:
    """Create a stderr handler that buffers output for debug logging.

    Even when stderr is suppressed from display, we capture it for
    debugging purposes when errors occur.
    """

    def stderr_handler(line: str) -> None:
        # Always buffer stderr for debug logging on errors
        stderr_buffer.append(line)
        # Log individual stderr lines at DEBUG level
        logger.debug(
            "claude_agent_sdk.sdk_query.stderr",
            event="sdk_query.stderr",
            context={"line": line.rstrip()},
        )

    return stderr_handler


def supported_option_names(
    options_type: type[Any],
) -> set[str] | None:
    """Return supported option names for ClaudeAgentOptions.

    Returns None when the options type accepts arbitrary keyword arguments.
    """
    dataclass_fields = getattr(options_type, "__dataclass_fields__", None)
    if isinstance(dataclass_fields, dict):
        return set(dataclass_fields)

    try:
        signature = inspect.signature(options_type)
    except (TypeError, ValueError):
        return None

    if any(
        param.kind == inspect.Parameter.VAR_KEYWORD
        for param in signature.parameters.values()
    ):
        return None

    return {
        name
        for name, param in signature.parameters.items()
        if param.kind
        in {
            inspect.Parameter.POSITIONAL_OR_KEYWORD,
            inspect.Parameter.KEYWORD_ONLY,
        }
    }


def filter_unsupported_options(
    options_kwargs: dict[str, Any],
    *,
    options_type: type[Any],
) -> dict[str, Any]:
    """Drop SDK option kwargs unsupported by the installed SDK version."""
    names = supported_option_names(options_type)
    if names is None:
        return options_kwargs

    unsupported = sorted(key for key in options_kwargs if key not in names)
    if not unsupported:
        return options_kwargs

    for key in unsupported:
        options_kwargs.pop(key, None)

    logger.info(
        "claude_agent_sdk.sdk_query.options_filtered",
        event="sdk_query.options_filtered",
        context={
            "unsupported_option_names": unsupported,
        },
    )
    return options_kwargs


def add_client_config_options(  # noqa: PLR0913
    options_kwargs: dict[str, Any],
    effective_cwd: str | None,
    output_format: dict[str, Any] | None,
    *,
    client_config: ClaudeAgentSDKClientConfig,
    model_config: ClaudeAgentSDKModelConfig,
    allowed_tools: tuple[str, ...] | None,
    disallowed_tools: tuple[str, ...],
) -> None:
    """Add client config options to the options dictionary."""
    # Add non-None optional values using dict comprehension
    optional_values = {
        "cwd": effective_cwd,
        "permission_mode": client_config.permission_mode,
        "max_turns": client_config.max_turns,
        "output_format": output_format,
        "reasoning": model_config.reasoning,
    }
    options_kwargs.update({k: v for k, v in optional_values.items() if v is not None})

    # Handle special cases that need explicit None checks or transformations
    if client_config.max_budget_usd is not None:
        options_kwargs["max_budget_usd"] = client_config.max_budget_usd
    if client_config.betas:
        options_kwargs["betas"] = list(client_config.betas)
    if allowed_tools is not None:
        options_kwargs["allowed_tools"] = list(allowed_tools)
    if disallowed_tools:
        options_kwargs["disallowed_tools"] = list(disallowed_tools)


def build_sdk_options_kwargs(  # noqa: PLR0913
    *,
    model: str,
    output_format: dict[str, Any] | None,
    bridged_tools: tuple[Any, ...],
    ephemeral_home: EphemeralHome,
    effective_cwd: str | None,
    client_config: ClaudeAgentSDKClientConfig,
    model_config: ClaudeAgentSDKModelConfig,
    allowed_tools: tuple[str, ...] | None,
    disallowed_tools: tuple[str, ...],
    stderr_buffer: list[str],
) -> dict[str, Any]:
    """Build the SDK options dictionary for ClaudeAgentOptions."""
    options_kwargs: dict[str, Any] = {"model": model}

    # Add client config options
    add_client_config_options(
        options_kwargs,
        effective_cwd,
        output_format,
        client_config=client_config,
        model_config=model_config,
        allowed_tools=allowed_tools,
        disallowed_tools=disallowed_tools,
    )

    # Apply isolation configuration from ephemeral home
    env_vars = ephemeral_home.get_env()
    options_kwargs["env"] = env_vars
    options_kwargs["setting_sources"] = ephemeral_home.get_setting_sources()

    # Pass typed sandbox settings via ClaudeAgentOptions.sandbox so the SDK
    # receives them directly (in addition to the settings.json path).
    from ._sandbox_conversion import to_sdk_sandbox_settings

    isolation = ephemeral_home.isolation
    options_kwargs["sandbox"] = to_sdk_sandbox_settings(
        isolation.sandbox,
        isolation.network_policy,
    )

    logger.debug(
        "claude_agent_sdk.sdk_query.env_configured",
        event="sdk_query.env_configured",
        context={
            "home_override": env_vars.get("HOME"),
            "has_api_key": "ANTHROPIC_API_KEY" in env_vars,
            "env_var_count": len(env_vars),
            "env_keys": [k for k in env_vars if "KEY" not in k.upper()],
        },
    )

    # Register custom tools via MCP server if any are provided
    if bridged_tools:
        mcp_server_config = create_mcp_server(bridged_tools)
        options_kwargs["mcp_servers"] = {"wink": mcp_server_config}
        logger.debug(
            "claude_agent_sdk.sdk_query.mcp_server_configured",
            event="sdk_query.mcp_server_configured",
            context={"mcp_server_name": "wink"},
        )

    options_kwargs["stderr"] = create_stderr_handler(stderr_buffer)
    return options_kwargs


def build_hooks_config(
    *,
    hook_context: HookContext,
    client_config: ClaudeAgentSDKClientConfig,
    collector: TranscriptCollector | None,
) -> dict[str, list[Any]]:
    """Build the hooks configuration for the SDK client."""
    from claude_agent_sdk.types import HookMatcher

    from ._task_completion import resolve_checker

    checker = resolve_checker(prompt=hook_context.prompt)
    pre_hook = create_pre_tool_use_hook(hook_context)
    post_hook = create_post_tool_use_hook(
        hook_context,
        stop_on_structured_output=client_config.stop_on_structured_output,
        task_completion_checker=checker,
    )

    # Use task completion stop hook if checker is configured
    if checker is not None:  # pragma: no cover - tested via hook tests
        stop_hook_fn = create_task_completion_stop_hook(
            hook_context,
            checker=checker,
        )
    else:
        stop_hook_fn = create_stop_hook(hook_context)

    prompt_hook = create_user_prompt_submit_hook(hook_context)
    subagent_stop_hook = create_subagent_stop_hook(hook_context)
    pre_compact_hook = create_pre_compact_hook(hook_context)

    # Get collector hooks (empty if collector is disabled)
    collector_hooks = collector.hooks_config() if collector else {}

    hook_types = [
        "PreToolUse",
        "PostToolUse",
        "Stop",
        "UserPromptSubmit",
        "SubagentStop",
        "PreCompact",
    ]

    logger.debug(
        "claude_agent_sdk.sdk_query.hooks_registered",
        event="sdk_query.hooks_registered",
        context={"hook_types": hook_types},
    )

    return {
        "PreToolUse": [
            HookMatcher(matcher=None, hooks=[pre_hook]),
            *collector_hooks.get("PreToolUse", []),
        ],
        "PostToolUse": [
            HookMatcher(matcher=None, hooks=[post_hook]),
            *collector_hooks.get("PostToolUse", []),
        ],
        "Stop": [
            HookMatcher(matcher=None, hooks=[stop_hook_fn]),
            *collector_hooks.get("Stop", []),
        ],
        "UserPromptSubmit": [
            HookMatcher(matcher=None, hooks=[prompt_hook]),
            *collector_hooks.get("UserPromptSubmit", []),
        ],
        "SubagentStop": [
            HookMatcher(matcher=None, hooks=[subagent_stop_hook]),
            *collector_hooks.get("SubagentStop", []),
        ],
        "PreCompact": [
            HookMatcher(matcher=None, hooks=[pre_compact_hook]),
            *collector_hooks.get("PreCompact", []),
        ],
    }
