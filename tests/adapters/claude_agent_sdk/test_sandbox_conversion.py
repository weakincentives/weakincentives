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

"""Tests for WINK → SDK sandbox config conversion.

Verifies that the conversion functions in ``_sandbox_conversion`` produce
valid SDK TypedDicts and settings.json sections that are consistent with
each other and with the source WINK configs.
"""

from __future__ import annotations

import os

from claude_agent_sdk import (
    SandboxIgnoreViolations,
    SandboxNetworkConfig,
    SandboxSettings,
)

from weakincentives.adapters.claude_agent_sdk._sandbox_conversion import (
    to_sdk_ignore_violations,
    to_sdk_network_config,
    to_sdk_sandbox_settings,
    to_settings_json_sandbox,
)
from weakincentives.adapters.claude_agent_sdk.isolation import (
    NetworkPolicy,
    SandboxConfig,
)


class TestToSdkNetworkConfig:
    def test_defaults_produce_empty_dict(self) -> None:
        config = to_sdk_network_config(NetworkPolicy())
        assert config == {}

    def test_allow_unix_sockets(self) -> None:
        config = to_sdk_network_config(
            NetworkPolicy(allow_unix_sockets=("/tmp/agent.sock",))
        )
        expected: SandboxNetworkConfig = {
            "allowUnixSockets": ["/tmp/agent.sock"],
        }
        assert config == expected

    def test_all_fields(self) -> None:
        config = to_sdk_network_config(
            NetworkPolicy(
                allow_unix_sockets=("/a",),
                allow_all_unix_sockets=True,
                allow_local_binding=True,
                http_proxy_port=8080,
                socks_proxy_port=1080,
            )
        )
        expected: SandboxNetworkConfig = {
            "allowUnixSockets": ["/a"],
            "allowAllUnixSockets": True,
            "allowLocalBinding": True,
            "httpProxyPort": 8080,
            "socksProxyPort": 1080,
        }
        assert config == expected

    def test_only_proxy_ports(self) -> None:
        config = to_sdk_network_config(NetworkPolicy(http_proxy_port=3128))
        assert config == {"httpProxyPort": 3128}


class TestToSdkIgnoreViolations:
    def test_returns_none_when_empty(self) -> None:
        assert to_sdk_ignore_violations(SandboxConfig()) is None

    def test_file_violations(self) -> None:
        result = to_sdk_ignore_violations(
            SandboxConfig(ignore_file_violations=("/tmp/noisy",))
        )
        expected: SandboxIgnoreViolations = {"file": ["/tmp/noisy"]}
        assert result == expected

    def test_network_violations(self) -> None:
        result = to_sdk_ignore_violations(
            SandboxConfig(ignore_network_violations=("localhost",))
        )
        expected: SandboxIgnoreViolations = {"network": ["localhost"]}
        assert result == expected

    def test_both_violations(self) -> None:
        result = to_sdk_ignore_violations(
            SandboxConfig(
                ignore_file_violations=("/tmp/noisy",),
                ignore_network_violations=("localhost",),
            )
        )
        expected: SandboxIgnoreViolations = {
            "file": ["/tmp/noisy"],
            "network": ["localhost"],
        }
        assert result == expected


class TestToSdkSandboxSettings:
    def test_defaults(self) -> None:
        settings = to_sdk_sandbox_settings(None, None)
        expected: SandboxSettings = {
            "enabled": True,
            "autoAllowBashIfSandboxed": True,
            "allowUnsandboxedCommands": False,
        }
        assert settings == expected

    def test_weaker_nested_sandbox(self) -> None:
        settings = to_sdk_sandbox_settings(
            SandboxConfig(enable_weaker_nested_sandbox=True), None
        )
        assert settings["enableWeakerNestedSandbox"] is True

    def test_weaker_nested_sandbox_omitted_when_false(self) -> None:
        settings = to_sdk_sandbox_settings(SandboxConfig(), None)
        assert "enableWeakerNestedSandbox" not in settings

    def test_excluded_commands(self) -> None:
        settings = to_sdk_sandbox_settings(
            SandboxConfig(excluded_commands=("docker", "git")), None
        )
        assert settings["excludedCommands"] == ["docker", "git"]

    def test_ignore_violations_included(self) -> None:
        settings = to_sdk_sandbox_settings(
            SandboxConfig(ignore_file_violations=("/tmp/noisy",)), None
        )
        assert settings["ignoreViolations"] == {"file": ["/tmp/noisy"]}

    def test_ignore_violations_omitted_when_empty(self) -> None:
        settings = to_sdk_sandbox_settings(SandboxConfig(), None)
        assert "ignoreViolations" not in settings

    def test_network_config_included(self) -> None:
        settings = to_sdk_sandbox_settings(
            None,
            NetworkPolicy(allow_local_binding=True),
        )
        assert settings["network"] == {"allowLocalBinding": True}

    def test_network_config_omitted_when_default(self) -> None:
        settings = to_sdk_sandbox_settings(SandboxConfig(), NetworkPolicy())
        assert "network" not in settings

    def test_full_config(self) -> None:
        settings = to_sdk_sandbox_settings(
            SandboxConfig(
                enabled=True,
                enable_weaker_nested_sandbox=True,
                excluded_commands=("docker",),
                bash_auto_allow=False,
                allow_unsandboxed_commands=True,
                ignore_file_violations=("/var/log",),
            ),
            NetworkPolicy(
                allow_unix_sockets=("/tmp/ssh.sock",),
                http_proxy_port=8080,
            ),
        )
        assert settings == {
            "enabled": True,
            "autoAllowBashIfSandboxed": False,
            "allowUnsandboxedCommands": True,
            "excludedCommands": ["docker"],
            "enableWeakerNestedSandbox": True,
            "ignoreViolations": {"file": ["/var/log"]},
            "network": {
                "allowUnixSockets": ["/tmp/ssh.sock"],
                "httpProxyPort": 8080,
            },
        }


class TestToSettingsJsonSandbox:
    def test_includes_sdk_fields(self) -> None:
        """settings.json should include all fields from the SDK TypedDict."""
        section = to_settings_json_sandbox(
            SandboxConfig(enable_weaker_nested_sandbox=True),
            NetworkPolicy(allow_local_binding=True),
        )
        assert section["enableWeakerNestedSandbox"] is True
        assert section["network"]["allowLocalBinding"] is True

    def test_includes_writable_paths(self) -> None:
        """settings.json has writablePaths (not in SDK TypedDict)."""
        section = to_settings_json_sandbox(
            SandboxConfig(writable_paths=("/tmp/output",)),
            None,
        )
        writable = section["writablePaths"]
        assert "/tmp/output" in writable

    def test_auto_adds_claude_temp_when_enabled(self) -> None:
        """Claude Code temp dir auto-added when sandbox is enabled."""
        section = to_settings_json_sandbox(SandboxConfig(enabled=True), None)
        claude_temp = f"/tmp/claude-{os.getuid()}"
        assert claude_temp in section["writablePaths"]

    def test_no_claude_temp_when_disabled(self) -> None:
        """No Claude Code temp dir when sandbox is disabled."""
        section = to_settings_json_sandbox(SandboxConfig(enabled=False), None)
        assert "writablePaths" not in section

    def test_no_duplicate_claude_temp(self) -> None:
        """Claude temp dir not duplicated if already in writable_paths."""
        claude_temp = f"/tmp/claude-{os.getuid()}"
        section = to_settings_json_sandbox(
            SandboxConfig(writable_paths=(claude_temp,)),
            None,
        )
        assert section["writablePaths"].count(claude_temp) == 1

    def test_includes_readable_paths(self) -> None:
        """settings.json has readablePaths (not in SDK TypedDict)."""
        section = to_settings_json_sandbox(
            SandboxConfig(readable_paths=("/data/readonly",)),
            None,
        )
        assert section["readablePaths"] == ["/data/readonly"]

    def test_includes_allowed_domains(self) -> None:
        """settings.json has network.allowedDomains (not in SDK TypedDict)."""
        section = to_settings_json_sandbox(
            None,
            NetworkPolicy(allowed_domains=("github.com",)),
        )
        assert section["network"]["allowedDomains"] == ["github.com"]

    def test_allowed_domains_empty_by_default(self) -> None:
        """allowedDomains should be empty list by default (no network)."""
        section = to_settings_json_sandbox(None, None)
        assert section["network"]["allowedDomains"] == []

    def test_network_merges_sdk_and_json_only_fields(self) -> None:
        """Network section should contain both SDK and JSON-only fields."""
        section = to_settings_json_sandbox(
            None,
            NetworkPolicy(
                allowed_domains=("github.com",),
                allow_local_binding=True,
                http_proxy_port=8080,
            ),
        )
        network = section["network"]
        # SDK fields
        assert network["allowLocalBinding"] is True
        assert network["httpProxyPort"] == 8080
        # JSON-only field
        assert network["allowedDomains"] == ["github.com"]
