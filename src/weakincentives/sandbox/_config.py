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

"""Sandbox intent: what environment to materialize, declared as data.

:class:`SandboxConfig` is a serde value describing the environment an
agent should run in — mounts, write posture, environment variables, setup
commands, and the egress policy. It carries credential **names only**;
secret material is bound at runtime through the sandbox control plane
(:meth:`~weakincentives.sandbox.Sandbox.configure_credentials`) and never
appears in configuration, serialized state, or logs.

:class:`EgressPolicy` is **default-deny**: a policy with no rules allows
nothing. Each :class:`EgressRule` allows matching outbound traffic and may
name a credential the egress proxy injects into allowed requests.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import field
from fnmatch import fnmatchcase

from ..dataclasses import FrozenDataclass
from ._mounts import HostMount

__all__ = [
    "CredentialInjection",
    "EgressPolicy",
    "EgressRule",
    "SandboxConfig",
]

_MAX_PORT = 65535


@FrozenDataclass()
class CredentialInjection:
    """How the egress proxy injects a named credential into a request.

    Carries the credential *name* and an injection shape — never the
    secret. ``{secret}`` in ``header_template`` is replaced by the bound
    material at injection time, inside the proxy only.
    """

    credential: str
    header_template: str = "Authorization: Bearer {secret}"

    def __post_init__(self) -> None:
        if not self.credential:
            msg = "credential name must not be empty"
            raise ValueError(msg)
        if "{secret}" not in self.header_template:
            msg = "header_template must contain a {secret} placeholder"
            raise ValueError(msg)


@FrozenDataclass()
class EgressRule:
    """Allow entry for outbound traffic matching host, ports, and protocol.

    Attributes:
        host_glob: Case-insensitive glob matched against the destination
            host (e.g. ``"*.pypi.org"``).
        ports: Allowed destination ports; empty allows any port.
        protocol: Protocol the rule applies to (e.g. ``"https"``),
            compared case-insensitively.
        credential: Optional injection the proxy applies to requests this
            rule allows.
    """

    host_glob: str
    ports: tuple[int, ...] = ()
    protocol: str = "https"
    credential: CredentialInjection | None = None

    def __post_init__(self) -> None:
        if not self.host_glob:
            msg = "host_glob must not be empty"
            raise ValueError(msg)
        if not self.protocol:
            msg = "protocol must not be empty"
            raise ValueError(msg)
        for port in self.ports:
            if not 1 <= port <= _MAX_PORT:
                msg = f"port must be in 1..{_MAX_PORT}, got {port}"
                raise ValueError(msg)

    def matches(self, host: str, *, port: int, protocol: str) -> bool:
        """True when this rule allows ``host:port`` over ``protocol``."""
        if not fnmatchcase(host.lower(), self.host_glob.lower()):
            return False
        if self.ports and port not in self.ports:
            return False
        return protocol.lower() == self.protocol.lower()


@FrozenDataclass()
class EgressPolicy:
    """Default-deny outbound policy: only ``allow`` rules pass traffic."""

    allow: tuple[EgressRule, ...] = ()

    @classmethod
    def deny_all(cls) -> EgressPolicy:
        """Policy that denies all outbound traffic (the default)."""
        return cls()

    def rule_for(
        self, host: str, *, port: int = 443, protocol: str = "https"
    ) -> EgressRule | None:
        """First rule allowing ``host:port`` over ``protocol``, if any."""
        for rule in self.allow:
            if rule.matches(host, port=port, protocol=protocol):
                return rule
        return None

    def allows(self, host: str, *, port: int = 443, protocol: str = "https") -> bool:
        """True when some rule allows ``host:port`` over ``protocol``."""
        return self.rule_for(host, port=port, protocol=protocol) is not None


@FrozenDataclass()
class SandboxConfig:
    """Declarative intent for a sandbox; a provider materializes it.

    A serde value safe to store and transmit: egress rules reference
    credentials by name only.

    Attributes:
        mounts: Host content copied into the sandbox at open time.
        allowed_host_roots: Security boundary mount sources must live under
            (empty = any host path).
        read_only: When True the sandbox's *filesystem facet* rejects
            writes. The shell facet is OS-level and unaffected.
        egress: Outbound network policy seeding the sandbox (default deny).
        env: Environment variables layered over the shell's hygienic base
            environment.
        setup: Commands run inside the sandbox after mounts materialize,
            in order. Each entry is ``shlex``-split into an argv vector and
            executed directly — no shell interpretation. A non-zero exit
            fails the open.
    """

    mounts: tuple[HostMount, ...] = ()
    allowed_host_roots: tuple[str, ...] = ()
    read_only: bool = False
    egress: EgressPolicy = field(default_factory=EgressPolicy)
    env: Mapping[str, str] | None = None
    setup: tuple[str, ...] = ()
