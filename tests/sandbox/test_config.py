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

"""Tests for sandbox intent types: config, egress policy, injections."""

from __future__ import annotations

import json
from dataclasses import FrozenInstanceError

import pytest

from weakincentives import serde
from weakincentives.sandbox import (
    CredentialInjection,
    EgressPolicy,
    EgressRule,
    HostMount,
    SandboxConfig,
)


class TestCredentialInjection:
    def test_defaults(self) -> None:
        injection = CredentialInjection(credential="api-token")
        assert injection.credential == "api-token"
        assert injection.header_template == "Authorization: Bearer {secret}"

    def test_custom_template(self) -> None:
        injection = CredentialInjection(
            credential="api-token", header_template="X-Api-Key: {secret}"
        )
        assert injection.header_template == "X-Api-Key: {secret}"

    def test_empty_credential_rejected(self) -> None:
        with pytest.raises(ValueError, match="credential name"):
            CredentialInjection(credential="")

    def test_template_without_placeholder_rejected(self) -> None:
        with pytest.raises(ValueError, match=r"\{secret\} placeholder"):
            CredentialInjection(credential="api-token", header_template="X-Api-Key: y")


class TestEgressRule:
    def test_defaults_allow_any_port_https(self) -> None:
        rule = EgressRule(host_glob="pypi.org")
        assert rule.matches("pypi.org", port=443, protocol="https")
        assert rule.matches("pypi.org", port=8443, protocol="https")
        assert not rule.matches("pypi.org", port=443, protocol="http")

    def test_host_glob_matching(self) -> None:
        rule = EgressRule(host_glob="*.pypi.org")
        assert rule.matches("files.pypi.org", port=443, protocol="https")
        assert not rule.matches("pypi.org", port=443, protocol="https")
        assert not rule.matches("evil.com", port=443, protocol="https")

    def test_host_matching_case_insensitive(self) -> None:
        rule = EgressRule(host_glob="PyPI.org")
        assert rule.matches("pypi.ORG", port=443, protocol="https")

    def test_protocol_matching_case_insensitive(self) -> None:
        rule = EgressRule(host_glob="pypi.org", protocol="HTTPS")
        assert rule.matches("pypi.org", port=443, protocol="https")

    def test_port_restriction(self) -> None:
        rule = EgressRule(host_glob="db.internal", ports=(5432, 5433), protocol="tcp")
        assert rule.matches("db.internal", port=5432, protocol="tcp")
        assert not rule.matches("db.internal", port=5434, protocol="tcp")

    def test_empty_host_glob_rejected(self) -> None:
        with pytest.raises(ValueError, match="host_glob"):
            EgressRule(host_glob="")

    def test_empty_protocol_rejected(self) -> None:
        with pytest.raises(ValueError, match="protocol"):
            EgressRule(host_glob="pypi.org", protocol="")

    @pytest.mark.parametrize("port", [0, -1, 65536])
    def test_out_of_range_port_rejected(self, port: int) -> None:
        with pytest.raises(ValueError, match="port must be in"):
            EgressRule(host_glob="pypi.org", ports=(port,))


class TestEgressPolicy:
    def test_default_denies_everything(self) -> None:
        policy = EgressPolicy()
        assert not policy.allows("pypi.org")
        assert policy.rule_for("pypi.org") is None

    def test_deny_all_equals_default(self) -> None:
        assert EgressPolicy.deny_all() == EgressPolicy()

    def test_allows_matching_rule(self) -> None:
        policy = EgressPolicy(allow=(EgressRule(host_glob="pypi.org"),))
        assert policy.allows("pypi.org")
        assert not policy.allows("evil.com")

    def test_rule_for_returns_first_match(self) -> None:
        first = EgressRule(host_glob="*.example.com")
        second = EgressRule(
            host_glob="api.example.com",
            credential=CredentialInjection(credential="api-token"),
        )
        policy = EgressPolicy(allow=(first, second))
        assert policy.rule_for("api.example.com") is first

    def test_rule_for_carries_credential(self) -> None:
        injection = CredentialInjection(credential="api-token")
        policy = EgressPolicy(
            allow=(EgressRule(host_glob="api.example.com", credential=injection),)
        )
        rule = policy.rule_for("api.example.com")
        assert rule is not None
        assert rule.credential == injection

    def test_allows_respects_port_and_protocol(self) -> None:
        policy = EgressPolicy(
            allow=(EgressRule(host_glob="db.internal", ports=(5432,), protocol="tcp"),)
        )
        assert policy.allows("db.internal", port=5432, protocol="tcp")
        assert not policy.allows("db.internal", port=80, protocol="tcp")
        assert not policy.allows("db.internal", port=5432, protocol="udp")


class TestSandboxConfig:
    def test_defaults(self) -> None:
        config = SandboxConfig()
        assert config.mounts == ()
        assert config.allowed_host_roots == ()
        assert config.read_only is False
        assert config.egress == EgressPolicy.deny_all()
        assert config.env is None
        assert config.setup == ()

    def test_frozen(self) -> None:
        config = SandboxConfig()
        with pytest.raises(FrozenInstanceError):
            config.read_only = True  # type: ignore[misc]

    def test_serde_round_trip(self) -> None:
        config = SandboxConfig(
            mounts=(HostMount(host_path="/src", include_glob=("*.py",)),),
            allowed_host_roots=("/src",),
            read_only=True,
            egress=EgressPolicy(
                allow=(
                    EgressRule(
                        host_glob="*.pypi.org",
                        ports=(443,),
                        credential=CredentialInjection(credential="pypi-token"),
                    ),
                )
            ),
            env={"PIP_INDEX_URL": "https://pypi.org/simple"},
            setup=("python -m venv .venv",),
        )
        data = serde.dump(config)
        restored = serde.parse(SandboxConfig, data)
        assert restored == config

    def test_serde_value_carries_credential_names_only(self) -> None:
        config = SandboxConfig(
            egress=EgressPolicy(
                allow=(
                    EgressRule(
                        host_glob="api.example.com",
                        credential=CredentialInjection(credential="api-token"),
                    ),
                )
            ),
        )
        dumped = json.dumps(serde.dump(config))
        assert "api-token" in dumped
        assert "{secret}" in dumped  # the template placeholder, never material
