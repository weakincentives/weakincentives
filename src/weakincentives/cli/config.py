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

"""Configuration helpers for the :mod:`weakincentives.cli.wink` entry points."""

from __future__ import annotations

import os
import tomllib
from collections.abc import Iterable, Mapping, MutableMapping
from dataclasses import dataclass, field
from pathlib import Path
from types import MappingProxyType
from typing import Any, cast

import yaml

DEFAULT_CONFIG_PATH = Path("~/.config/wink/config.toml")

ENV_WORKSPACE_ROOT = "WINK_WORKSPACE_ROOT"
ENV_OVERRIDES_DIR = "WINK_OVERRIDES_DIR"
ENV_ENVIRONMENT = "WINK_ENV"
ENV_PROMPT_REGISTRY = "WINK_PROMPT_REGISTRY"
ENV_LISTEN_HOST = "WINK_LISTEN_HOST"
ENV_LISTEN_PORT = "WINK_LISTEN_PORT"
ENV_AUTH_TOKENS = "WINK_AUTH_TOKENS"

__all__ = ["DEFAULT_CONFIG_PATH", "ConfigError", "MCPServerConfig", "load_config"]


@dataclass(frozen=True, slots=True)
class MCPServerConfig:
    """Resolved configuration for the wink MCP server."""

    workspace_root: Path
    overrides_dir: Path
    environment: str | None = None
    prompt_registry_modules: tuple[str, ...] = field(default_factory=tuple)
    auth_tokens: Mapping[str, str] = field(default_factory=dict[str, str])
    listen_host: str = "127.0.0.1"
    listen_port: int = 0

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "auth_tokens", MappingProxyType(dict(self.auth_tokens))
        )


class ConfigError(ValueError):
    """Raised when the wink MCP configuration is invalid."""


def load_config(
    path: Path | Mapping[str, Any] | None,
    cli_overrides: object | None = None,
    *,
    env: Mapping[str, str] | None = None,
) -> MCPServerConfig:
    """Load and validate the wink MCP configuration.

    Parameters
    ----------
    path:
        Path to the configuration file. ``None`` falls back to
        ``~/.config/wink/config.toml``. Tests may pass an in-memory mapping to
        skip filesystem I/O.
    cli_overrides:
        Overrides provided by CLI processing. Keys mirror ``MCPServerConfig``'s
        field names.
    env:
        Optional environment mapping. Defaults to :data:`os.environ`.

    Returns
    -------
    MCPServerConfig
        The resolved configuration object.
    """

    env_map = dict(os.environ if env is None else env)

    if isinstance(path, Mapping):
        config_data: dict[str, object] = dict(path)
        config_path: Path | None = None
    else:
        config_path = path if path is not None else DEFAULT_CONFIG_PATH.expanduser()
        config_data = _load_config_file(config_path)

    config = _normalise_config(config_data)
    config = _apply_environment_overrides(config=config, env=env_map)
    config = _apply_cli_overrides(config=config, overrides=cli_overrides)

    return _build_config(config=config, config_path=config_path)


def _load_config_file(path: Path) -> dict[str, object]:
    if not path.exists():
        if path == DEFAULT_CONFIG_PATH.expanduser():
            return {}
        msg = f"Configuration file not found: {path}"
        raise FileNotFoundError(msg)

    suffix = path.suffix.lower()
    data: object
    if suffix == ".toml" or not suffix:
        with path.open("rb") as handle:
            data = tomllib.load(handle)
    elif suffix in {".yaml", ".yml"}:
        with path.open("r", encoding="utf-8") as handle:
            data = yaml.safe_load(handle) or {}
    else:
        msg = f"Unsupported configuration format: {path.suffix}"
        raise ConfigError(msg)

    if not isinstance(data, MutableMapping):  # pragma: no cover - defensive
        msg = "Configuration file must contain a mapping at the root."
        raise ConfigError(msg)

    mapping = cast(MutableMapping[object, object], data)
    typed_data: dict[str, object] = {}
    for key, value in mapping.items():
        if not isinstance(key, str):
            msg = f"Configuration keys must be strings (got {key!r})."
            raise ConfigError(msg)
        typed_data[key] = value
    return typed_data


def _normalise_config(raw: Mapping[str, object]) -> dict[str, object]:
    config: dict[str, object] = {
        "workspace_root": raw.get("workspace_root"),
        "overrides_dir": raw.get("overrides_dir"),
        "environment": raw.get("environment") or raw.get("env"),
        "prompt_registry_modules": raw.get("prompt_registry_modules")
        or raw.get("prompt_registry"),
        "auth_tokens": dict[str, str](),
        "listen_host": raw.get("listen_host"),
        "listen_port": raw.get("listen_port"),
    }

    workspace_section_obj = raw.get("workspace")
    if config["workspace_root"] is None and isinstance(workspace_section_obj, Mapping):
        workspace_section = cast(Mapping[str, object], workspace_section_obj)
        config["workspace_root"] = workspace_section.get("root")

    overrides_section_obj = raw.get("overrides")
    if config["overrides_dir"] is None and isinstance(overrides_section_obj, Mapping):
        overrides_section = cast(Mapping[str, object], overrides_section_obj)
        overrides_dir = overrides_section.get("dir") or overrides_section.get("path")
        if overrides_dir is not None:
            config["overrides_dir"] = overrides_dir

    listen_section_obj = raw.get("listen")
    if isinstance(listen_section_obj, Mapping):
        listen_section = cast(Mapping[str, object], listen_section_obj)
        host_value = listen_section.get("host")
        port_value = listen_section.get("port")
        if host_value is not None:
            config["listen_host"] = host_value
        if port_value is not None:
            config["listen_port"] = port_value

    tokens_obj = raw.get("auth_tokens")
    if isinstance(tokens_obj, Mapping):
        tokens_mapping = cast(Mapping[str, object], tokens_obj)
        config["auth_tokens"] = dict(tokens_mapping)
    elif tokens_obj is not None:
        config["auth_tokens"] = tokens_obj
    else:
        auth_section_obj = raw.get("auth")
        if isinstance(auth_section_obj, Mapping):
            auth_section = cast(Mapping[str, object], auth_section_obj)
            nested_tokens = auth_section.get("tokens")
            if isinstance(nested_tokens, Mapping):
                tokens_mapping = cast(Mapping[str, object], nested_tokens)
                config["auth_tokens"] = dict(tokens_mapping)
            elif nested_tokens is not None:
                config["auth_tokens"] = nested_tokens

    return config


def _apply_environment_overrides(
    *, config: dict[str, object], env: Mapping[str, str]
) -> dict[str, object]:
    if ENV_WORKSPACE_ROOT in env:
        config["workspace_root"] = env[ENV_WORKSPACE_ROOT]
    if ENV_OVERRIDES_DIR in env:
        config["overrides_dir"] = env[ENV_OVERRIDES_DIR]
    if ENV_ENVIRONMENT in env:
        config["environment"] = env[ENV_ENVIRONMENT]
    if ENV_PROMPT_REGISTRY in env:
        config["prompt_registry_modules"] = _split_modules(env[ENV_PROMPT_REGISTRY])
    if ENV_LISTEN_HOST in env:
        config["listen_host"] = env[ENV_LISTEN_HOST]
    if ENV_LISTEN_PORT in env:
        try:
            config["listen_port"] = int(env[ENV_LISTEN_PORT])
        except ValueError as exc:  # pragma: no cover - defensive guard
            msg = f"Invalid port in {ENV_LISTEN_PORT}: {env[ENV_LISTEN_PORT]!r}"
            raise ConfigError(msg) from exc
    if ENV_AUTH_TOKENS in env:
        config["auth_tokens"] = _parse_mapping_string(env[ENV_AUTH_TOKENS])

    return config


def _apply_cli_overrides(
    *, config: dict[str, object], overrides: object | None
) -> dict[str, object]:
    if overrides is None:
        return config

    materialised: dict[str, object]
    if isinstance(overrides, Mapping):
        materialised = dict(cast(Mapping[str, object], overrides))
    elif hasattr(overrides, "__dict__"):
        materialised = {key: getattr(overrides, key) for key in vars(overrides)}
    else:  # pragma: no cover - defensive guard
        msg = "CLI overrides must be a mapping or support attribute access."
        raise TypeError(msg)

    for key, value in materialised.items():
        if value is None:
            continue
        config[key] = value

    return config


def _build_config(
    *, config: Mapping[str, object], config_path: Path | None
) -> MCPServerConfig:
    workspace_root = _coerce_path(config.get("workspace_root"), "workspace_root")
    if workspace_root is None:
        location = (
            str(config_path) if config_path is not None else str(DEFAULT_CONFIG_PATH)
        )
        msg = f"`workspace_root` must be configured (source: {location})."
        raise ConfigError(msg)

    overrides_dir = _coerce_path(config.get("overrides_dir"), "overrides_dir")
    if overrides_dir is None:
        overrides_dir = workspace_root / ".wink" / "overrides"

    environment = _coerce_optional_str(config.get("environment"))
    prompt_modules = _coerce_modules(config.get("prompt_registry_modules"))
    auth_tokens = _coerce_auth_tokens(config.get("auth_tokens"))
    listen_host = _coerce_optional_str(config.get("listen_host")) or "127.0.0.1"
    listen_port = _coerce_port(config.get("listen_port"))

    return MCPServerConfig(
        workspace_root=workspace_root,
        overrides_dir=overrides_dir,
        environment=environment,
        prompt_registry_modules=prompt_modules,
        auth_tokens=auth_tokens,
        listen_host=listen_host,
        listen_port=listen_port,
    )


def _coerce_path(value: object, field_name: str) -> Path | None:
    if value is None:
        return None
    if isinstance(value, Path):
        return value.expanduser()
    if isinstance(value, str):
        return Path(value).expanduser()
    msg = f"{field_name} must be a path-like value."
    raise ConfigError(msg)


def _coerce_optional_str(value: object) -> str | None:
    if value is None:
        return None
    if isinstance(value, str):
        return value
    msg = "Value must be a string."
    raise ConfigError(msg)


def _coerce_modules(value: object) -> tuple[str, ...]:
    if value is None:
        return ()
    if isinstance(value, str):
        return tuple(part for part in _split_modules(value))
    if isinstance(value, Iterable):
        iterable_value = cast(Iterable[object], value)
        result: list[str] = []
        for item in iterable_value:
            if not isinstance(item, str):
                msg = "Prompt registry modules must be strings."
                raise ConfigError(msg)
            module = item.strip()
            if module:
                result.append(module)
        return tuple(result)
    msg = "Prompt registry modules must be a sequence of strings."
    raise ConfigError(msg)


def _split_modules(value: str) -> tuple[str, ...]:
    parts = [part.strip() for part in value.split(",")]
    return tuple(part for part in parts if part)


def _coerce_auth_tokens(value: object) -> Mapping[str, str]:
    if value is None:
        return MappingProxyType({})
    if isinstance(value, Mapping):
        mapping_value = cast(Mapping[object, object], value)
        tokens: dict[str, str] = {}
        for key, token in mapping_value.items():
            if not isinstance(key, str) or not isinstance(token, str):
                msg = "Auth tokens must map strings to strings."
                raise ConfigError(msg)
            tokens[key] = token
        return MappingProxyType(tokens)
    msg = "Auth tokens must be provided as a mapping of strings."
    raise ConfigError(msg)


def _parse_mapping_string(value: str) -> dict[str, str]:
    tokens: dict[str, str] = {}
    for fragment in value.split(","):
        fragment = fragment.strip()
        if not fragment:
            continue
        key, _, token = fragment.partition("=")
        key = key.strip()
        token = token.strip()
        if not key or not token:
            msg = f"Invalid auth token entry: {fragment!r}"
            raise ConfigError(msg)
        tokens[key] = token
    return tokens


def _coerce_port(value: object) -> int:
    if value is None:
        return 0
    if isinstance(value, int):
        port = value
    elif isinstance(value, str):
        try:
            port = int(value)
        except ValueError as exc:
            msg = f"Port must be an integer: {value!r}"
            raise ConfigError(msg) from exc
    else:
        msg = "Port must be an integer."
        raise ConfigError(msg)

    if not 0 <= port <= 65535:
        msg = f"Port must be between 0 and 65535 (inclusive): {port}"
        raise ConfigError(msg)
    return port
