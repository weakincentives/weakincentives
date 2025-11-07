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

"""Tests for :mod:`weakincentives.cli.config`."""

from __future__ import annotations

from pathlib import Path
from textwrap import dedent
from types import SimpleNamespace

import pytest
from pytest import MonkeyPatch

from weakincentives.cli.config import ConfigError, _coerce_auth_tokens, load_config


def test_load_config_from_mapping(tmp_path: Path) -> None:
    config_dict = {
        "workspace_root": str(tmp_path / "workspace"),
        "overrides_dir": str(tmp_path / "overrides"),
        "prompt_registry_modules": ["a.registry", "b.registry"],
        "environment": "dev",
        "auth": {"tokens": {"client": "token"}},
        "listen": {"host": "0.0.0.0", "port": 8765},
    }

    config = load_config(config_dict, env={})

    assert config.workspace_root == Path(tmp_path / "workspace")
    assert config.overrides_dir == Path(tmp_path / "overrides")
    assert config.environment == "dev"
    assert config.prompt_registry_modules == ("a.registry", "b.registry")
    assert dict(config.auth_tokens) == {"client": "token"}
    assert config.listen_host == "0.0.0.0"
    assert config.listen_port == 8765


def test_load_config_supports_nested_sections(tmp_path: Path) -> None:
    config_dict = {
        "workspace": {"root": str(tmp_path / "workspace")},
        "overrides": {"path": str(tmp_path / "overrides")},
        "prompt_registry": ["nested.module"],
        "auth": {"tokens": {"client": "token"}},
        "listen": {"host": "0.0.0.0", "port": 4321},
    }

    config = load_config(config_dict, env={})

    assert config.workspace_root == Path(tmp_path / "workspace")
    assert config.overrides_dir == Path(tmp_path / "overrides")
    assert config.prompt_registry_modules == ("nested.module",)
    assert dict(config.auth_tokens) == {"client": "token"}
    assert config.listen_host == "0.0.0.0"
    assert config.listen_port == 4321


def test_load_config_applies_environment_overrides(tmp_path: Path) -> None:
    env_workspace = tmp_path / "env-workspace"
    env_overrides = tmp_path / "env-overrides"
    env = {
        "WINK_WORKSPACE_ROOT": str(env_workspace),
        "WINK_OVERRIDES_DIR": str(env_overrides),
        "WINK_ENV": "production",
        "WINK_PROMPT_REGISTRY": "alpha.registry,beta.registry",
        "WINK_LISTEN_HOST": "::",
        "WINK_LISTEN_PORT": "9000",
        "WINK_AUTH_TOKENS": "client=secret,other=token",
    }

    config = load_config({}, env=env)

    assert config.workspace_root == env_workspace
    assert config.overrides_dir == env_overrides
    assert config.environment == "production"
    assert config.prompt_registry_modules == ("alpha.registry", "beta.registry")
    assert dict(config.auth_tokens) == {"client": "secret", "other": "token"}
    assert config.listen_host == "::"
    assert config.listen_port == 9000


def test_cli_overrides_take_precedence(tmp_path: Path) -> None:
    base_config = {
        "workspace_root": str(tmp_path / "workspace"),
        "overrides_dir": str(tmp_path / "overrides"),
        "prompt_registry_modules": ["base.module"],
        "listen": {"port": 1111},
    }
    env = {"WINK_LISTEN_PORT": "2222"}
    cli_overrides = {
        "listen_port": 3333,
        "prompt_registry_modules": ["cli.module"],
        "auth_tokens": {"client": "cli"},
    }

    config = load_config(base_config, cli_overrides=cli_overrides, env=env)

    assert config.listen_port == 3333
    assert config.prompt_registry_modules == ("cli.module",)
    assert dict(config.auth_tokens) == {"client": "cli"}


def test_cli_overrides_accepts_attribute_namespace(tmp_path: Path) -> None:
    base_config = {
        "workspace_root": str(tmp_path / "workspace"),
        "overrides_dir": str(tmp_path / "overrides"),
    }

    config = load_config(
        base_config,
        cli_overrides=SimpleNamespace(listen_port=4444),
        env={},
    )

    assert config.listen_port == 4444


def test_missing_workspace_root_raises() -> None:
    with pytest.raises(ConfigError) as excinfo:
        load_config({}, env={})

    assert "workspace_root" in str(excinfo.value)


def test_load_config_uses_default_path(
    monkeypatch: MonkeyPatch, tmp_path: Path
) -> None:
    home = tmp_path / "home"
    config_path = home / ".config" / "wink" / "config.toml"
    config_path.parent.mkdir(parents=True)
    config_path.write_text(
        dedent(
            """
            workspace_root = "/workspace"
            overrides_dir = "/overrides"
            listen_port = 6543
            """
        ).strip()
    )

    monkeypatch.setenv("HOME", str(home))

    config = load_config(None, env={})

    assert config.workspace_root == Path("/workspace")
    assert config.overrides_dir == Path("/overrides")
    assert config.listen_port == 6543


def test_load_config_handles_missing_default_with_env(tmp_path: Path) -> None:
    workspace_root = tmp_path / "workspace"
    overrides_dir = tmp_path / "overrides"
    env = {
        "WINK_WORKSPACE_ROOT": str(workspace_root),
        "WINK_OVERRIDES_DIR": str(overrides_dir),
    }

    config = load_config(None, env=env)

    assert config.workspace_root == workspace_root
    assert config.overrides_dir == overrides_dir


def test_load_config_supports_yaml(tmp_path: Path) -> None:
    config_path = tmp_path / "wink.yaml"
    config_path.write_text(
        dedent(
            """
            workspace_root: "/workspace"
            overrides_dir: "/overrides"
            listen:
              host: 0.0.0.0
              port: 4321
            """
        ).strip()
    )

    config = load_config(config_path, env={})

    assert config.listen_host == "0.0.0.0"
    assert config.listen_port == 4321


def test_load_config_missing_file_raises(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        load_config(tmp_path / "missing.toml", env={})


def test_load_config_rejects_invalid_path_type() -> None:
    with pytest.raises(ConfigError):
        load_config({"workspace_root": 123}, env={})


def test_load_config_rejects_invalid_environment_type(tmp_path: Path) -> None:
    config_dict = {
        "workspace_root": str(tmp_path / "workspace"),
        "overrides_dir": str(tmp_path / "overrides"),
        "environment": 123,
    }

    with pytest.raises(ConfigError):
        load_config(config_dict, env={})


def test_load_config_rejects_invalid_modules(tmp_path: Path) -> None:
    config_dict = {
        "workspace_root": str(tmp_path / "workspace"),
        "overrides_dir": str(tmp_path / "overrides"),
        "prompt_registry_modules": [123],
    }

    with pytest.raises(ConfigError):
        load_config(config_dict, env={})


def test_load_config_rejects_invalid_auth_tokens(tmp_path: Path) -> None:
    config_dict = {
        "workspace_root": str(tmp_path / "workspace"),
        "overrides_dir": str(tmp_path / "overrides"),
        "auth_tokens": {"client": 123},
    }

    with pytest.raises(ConfigError):
        load_config(config_dict, env={})


def test_load_config_rejects_invalid_port(tmp_path: Path) -> None:
    config_dict = {
        "workspace_root": str(tmp_path / "workspace"),
        "overrides_dir": str(tmp_path / "overrides"),
        "listen": {"port": "abc"},
    }

    with pytest.raises(ConfigError):
        load_config(config_dict, env={})


def test_load_config_rejects_out_of_range_port(tmp_path: Path) -> None:
    config_dict = {
        "workspace_root": str(tmp_path / "workspace"),
        "overrides_dir": str(tmp_path / "overrides"),
        "listen": {"port": 99999},
    }

    with pytest.raises(ConfigError):
        load_config(config_dict, env={})


def test_load_config_uses_default_override_directory(tmp_path: Path) -> None:
    workspace_root = tmp_path / "workspace"
    config_dict = {"workspace_root": workspace_root}

    config = load_config(config_dict, env={})

    assert config.overrides_dir == workspace_root / ".wink" / "overrides"


def test_load_config_accepts_path_objects(tmp_path: Path) -> None:
    workspace_root = tmp_path / "workspace"
    overrides_dir = tmp_path / "overrides"

    config = load_config(
        {"workspace_root": workspace_root, "overrides_dir": overrides_dir},
        env={},
    )

    assert config.workspace_root == workspace_root
    assert config.overrides_dir == overrides_dir


def test_load_config_rejects_non_iterable_modules(tmp_path: Path) -> None:
    config_dict = {
        "workspace_root": str(tmp_path / "workspace"),
        "overrides_dir": str(tmp_path / "overrides"),
        "prompt_registry_modules": 123,
    }

    with pytest.raises(ConfigError):
        load_config(config_dict, env={})


def test_load_config_rejects_invalid_auth_token_env() -> None:
    env = {
        "WINK_WORKSPACE_ROOT": "/workspace",
        "WINK_OVERRIDES_DIR": "/overrides",
        "WINK_AUTH_TOKENS": "invalid",
    }

    with pytest.raises(ConfigError):
        load_config({}, env=env)


def test_load_config_rejects_non_numeric_port(tmp_path: Path) -> None:
    config_dict = {
        "workspace_root": str(tmp_path / "workspace"),
        "overrides_dir": str(tmp_path / "overrides"),
        "listen": {"port": 3.14},
    }

    with pytest.raises(ConfigError):
        load_config(config_dict, env={})


def test_load_config_rejects_unsupported_format(tmp_path: Path) -> None:
    config_path = tmp_path / "config.ini"
    config_path.write_text("workspace_root=/workspace")

    with pytest.raises(ConfigError):
        load_config(config_path, env={})


def test_load_config_rejects_non_string_keys(tmp_path: Path) -> None:
    config_path = tmp_path / "config.yaml"
    config_path.write_text("1: value\n")

    with pytest.raises(ConfigError):
        load_config(config_path, env={})


def test_cli_overrides_updates_values(tmp_path: Path) -> None:
    workspace_root = tmp_path / "workspace"
    cli_overrides = {"overrides_dir": tmp_path / "custom"}

    config = load_config(
        {"workspace_root": workspace_root}, cli_overrides=cli_overrides, env={}
    )

    assert config.overrides_dir == tmp_path / "custom"


def test_load_config_accepts_string_modules(tmp_path: Path) -> None:
    config_dict = {
        "workspace_root": str(tmp_path / "workspace"),
        "overrides_dir": str(tmp_path / "overrides"),
        "prompt_registry_modules": "module.path",
    }

    config = load_config(config_dict, env={})

    assert config.prompt_registry_modules == ("module.path",)


def test_load_config_defaults_empty_auth_tokens(tmp_path: Path) -> None:
    config_dict = {
        "workspace_root": str(tmp_path / "workspace"),
        "overrides_dir": str(tmp_path / "overrides"),
    }

    config = load_config(config_dict, env={})

    assert dict(config.auth_tokens) == {}


def test_load_config_rejects_non_mapping_auth_tokens(tmp_path: Path) -> None:
    config_dict = {
        "workspace_root": str(tmp_path / "workspace"),
        "overrides_dir": str(tmp_path / "overrides"),
        "auth_tokens": ["token"],
    }

    with pytest.raises(ConfigError):
        load_config(config_dict, env={})


def test_load_config_rejects_non_string_auth_token_keys(tmp_path: Path) -> None:
    config_dict = {
        "workspace_root": str(tmp_path / "workspace"),
        "overrides_dir": str(tmp_path / "overrides"),
        "auth_tokens": {1: "token"},
    }

    with pytest.raises(ConfigError):
        load_config(config_dict, env={})


def test_parse_mapping_string_ignores_empty_fragments() -> None:
    env = {
        "WINK_WORKSPACE_ROOT": "/workspace",
        "WINK_OVERRIDES_DIR": "/overrides",
        "WINK_AUTH_TOKENS": "foo=bar,,baz=qux",
    }

    config = load_config({}, env=env)

    assert dict(config.auth_tokens) == {"foo": "bar", "baz": "qux"}


def test_load_config_rejects_nested_non_mapping_auth_tokens(tmp_path: Path) -> None:
    config_dict = {
        "workspace_root": str(tmp_path / "workspace"),
        "overrides_dir": str(tmp_path / "overrides"),
        "auth": {"tokens": ["token"]},
    }

    with pytest.raises(ConfigError):
        load_config(config_dict, env={})


def test_cli_overrides_skip_none_values(tmp_path: Path) -> None:
    workspace_root = tmp_path / "workspace"
    cli_overrides = {"environment": None, "listen_port": 5000}

    config = load_config(
        {"workspace_root": workspace_root}, cli_overrides=cli_overrides, env={}
    )

    assert config.listen_port == 5000
    assert config.environment is None


def test_coerce_auth_tokens_none_returns_empty_mapping() -> None:
    tokens = _coerce_auth_tokens(None)

    assert dict(tokens) == {}
