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

"""Claude Agent SDK integration scenarios for isolation and network policies."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pytest
from claude_agent_sdk_fixtures import (
    _PROMPT_NS,
    EchoParams,
    EmptyParams,
    FileReadTestResult,
    GreetingParams,
    ReadFileParams,
    ReviewAnalysis,
    ReviewParams,
    TransformRequest,
    _assert_prompt_usage,
    _build_greeting_prompt,
    _build_structured_prompt,
    _build_tool_prompt,
    _build_uppercase_tool,
    _make_adapter,
    _make_config,
    _make_session_with_usage_tracking,
    pytestmark as claude_agent_sdk_pytestmark,
)

from weakincentives.adapters.claude_agent_sdk import (
    IsolationConfig,
    NetworkPolicy,
    SandboxConfig,
)
from weakincentives.prompt import MarkdownSection, Prompt, PromptTemplate

pytest.importorskip("claude_agent_sdk")

pytestmark = claude_agent_sdk_pytestmark
# =============================================================================
# Isolation Integration Tests
# =============================================================================
#
# These tests validate that the IsolationConfig properly isolates SDK execution
# from the host's ~/.claude configuration, preventing interference with the
# user's personal Claude Code installation.
# =============================================================================


def test_claude_agent_sdk_adapter_with_isolation_returns_text(tmp_path: Path) -> None:
    """Verify adapter works correctly with IsolationConfig enabled.

    This test validates that:
    1. The adapter creates an ephemeral home directory
    2. SDK execution uses the ephemeral home instead of ~/.claude
    3. The adapter still returns valid responses
    4. Cleanup happens after execution
    """
    isolation = IsolationConfig(
        network_policy=NetworkPolicy.no_network(),
        sandbox=SandboxConfig(enabled=True),
    )

    config = _make_config(
        tmp_path,
        isolation=isolation,
    )

    adapter = _make_adapter(
        tmp_path,
        client_config=config,
    )

    prompt_template = _build_greeting_prompt()
    params = GreetingParams(audience="isolation tests")
    prompt = Prompt(prompt_template).bind(params)

    session = _make_session_with_usage_tracking()
    response = adapter.evaluate(prompt, session=session)

    assert response.prompt_name == "greeting"
    assert response.text is not None
    assert response.text.strip()
    _assert_prompt_usage(session)


def test_claude_agent_sdk_adapter_isolation_with_custom_tools(tmp_path: Path) -> None:
    """Verify custom tools work correctly in isolated mode.

    This test validates that MCP-bridged tools function correctly when
    the adapter is configured with isolation. The ephemeral home should
    not affect tool bridging functionality.
    """
    isolation = IsolationConfig(
        network_policy=NetworkPolicy.no_network(),
        sandbox=SandboxConfig(enabled=True),
    )

    config = _make_config(
        tmp_path,
        isolation=isolation,
    )

    tool = _build_uppercase_tool()
    prompt_template = _build_tool_prompt(tool)
    params = TransformRequest(text="isolation mode")

    adapter = _make_adapter(
        tmp_path,
        client_config=config,
    )

    prompt = Prompt(prompt_template).bind(params)

    session = _make_session_with_usage_tracking()
    response = adapter.evaluate(prompt, session=session)

    assert response.prompt_name == "uppercase_workflow"
    assert response.text is not None
    # The uppercase text should appear in the response
    assert "ISOLATION MODE" in response.text
    _assert_prompt_usage(session)


def test_claude_agent_sdk_adapter_isolation_with_structured_output(
    tmp_path: Path,
) -> None:
    """Verify structured output works in isolated mode.

    This test validates that structured output parsing functions correctly
    when the adapter is configured with isolation.
    """
    isolation = IsolationConfig(
        network_policy=NetworkPolicy.no_network(),
    )

    config = _make_config(
        tmp_path,
        isolation=isolation,
    )

    adapter = _make_adapter(
        tmp_path,
        client_config=config,
    )

    prompt_template = _build_structured_prompt()
    sample = ReviewParams(
        text="The isolated execution mode provides security benefits. Users report peace of mind.",
    )

    prompt = Prompt(prompt_template).bind(sample)

    session = _make_session_with_usage_tracking()
    response = adapter.evaluate(prompt, session=session)

    assert response.prompt_name == "structured_review"
    assert response.output is not None
    assert isinstance(response.output, ReviewAnalysis)
    assert response.output.summary
    assert response.output.sentiment
    _assert_prompt_usage(session)


def test_claude_agent_sdk_adapter_isolation_does_not_modify_host_claude_dir(
    tmp_path: Path,
) -> None:
    """Verify isolation prevents modification of host ~/.claude directory.

    This test validates that running with IsolationConfig does not create,
    modify, or read from the real ~/.claude directory. This is critical for
    ensuring the user's personal configuration is not affected.
    """
    import tempfile

    # Create a fake ~/.claude directory to monitor
    with tempfile.TemporaryDirectory() as fake_home:
        fake_claude_dir = Path(fake_home) / ".claude"
        fake_claude_dir.mkdir()

        # Create a marker file to verify it's not modified
        marker_file = fake_claude_dir / "marker.txt"
        marker_file.write_text("original content")
        original_mtime = marker_file.stat().st_mtime

        # Configure isolation
        isolation = IsolationConfig(
            network_policy=NetworkPolicy.no_network(),
            sandbox=SandboxConfig(enabled=True),
            # Note: We don't use include_host_env because that would
            # potentially inherit HOME from the test environment
        )

        config = _make_config(
            tmp_path,
            isolation=isolation,
            cwd=fake_home,  # Use the temp directory as cwd
        )

        adapter = _make_adapter(
            tmp_path,
            client_config=config,
        )

        prompt_template = _build_greeting_prompt()
        params = GreetingParams(audience="host protection test")
        prompt = Prompt(prompt_template).bind(params)

        session = _make_session_with_usage_tracking()
        response = adapter.evaluate(prompt, session=session)

        # Verify execution completed
        assert response.text is not None

        # Verify marker file was not modified
        assert marker_file.exists(), "Marker file should still exist"
        assert marker_file.read_text() == "original content"
        assert marker_file.stat().st_mtime == original_mtime

        # Verify no new files were created in fake_claude_dir
        # (besides our marker file)
        files_in_claude_dir = list(fake_claude_dir.iterdir())
        assert len(files_in_claude_dir) == 1
        assert files_in_claude_dir[0].name == "marker.txt"


def test_claude_agent_sdk_adapter_isolation_network_policy_no_network(
    tmp_path: Path,
) -> None:
    """Verify NetworkPolicy.no_network() still allows Claude API access.

    This test validates that even with no_network() (which blocks all tool
    network access), the Claude Code CLI can still reach the Anthropic API.
    The network policy only affects tools running in the sandbox, not the
    CLI itself.
    """
    isolation = IsolationConfig(
        network_policy=NetworkPolicy.no_network(),
        sandbox=SandboxConfig(enabled=True),
    )

    config = _make_config(
        tmp_path,
        isolation=isolation,
    )

    adapter = _make_adapter(
        tmp_path,
        client_config=config,
    )

    prompt_template = _build_greeting_prompt()
    params = GreetingParams(audience="network policy test")
    prompt = Prompt(prompt_template).bind(params)

    session = _make_session_with_usage_tracking()

    # This should succeed because api.anthropic.com is allowed
    response = adapter.evaluate(prompt, session=session)

    assert response.text is not None
    assert response.text.strip()
    _assert_prompt_usage(session)


def test_claude_agent_sdk_adapter_isolation_with_custom_env(tmp_path: Path) -> None:
    """Verify custom environment variables are passed to SDK subprocess.

    This test validates that the env parameter in IsolationConfig correctly
    passes environment variables to the SDK subprocess.
    """
    isolation = IsolationConfig(
        network_policy=NetworkPolicy.no_network(),
        env={
            "WINK_TEST_VAR": "isolation_test_value",
        },
    )

    config = _make_config(
        tmp_path,
        isolation=isolation,
    )

    adapter = _make_adapter(
        tmp_path,
        client_config=config,
    )

    prompt_template = _build_greeting_prompt()
    params = GreetingParams(audience="custom env test")
    prompt = Prompt(prompt_template).bind(params)

    session = _make_session_with_usage_tracking()
    response = adapter.evaluate(prompt, session=session)

    assert response.text is not None
    _assert_prompt_usage(session)


def test_claude_agent_sdk_adapter_isolation_cleanup_on_success(tmp_path: Path) -> None:
    """Verify ephemeral home is cleaned up after successful execution.

    This test validates that the ephemeral home directory created for
    isolation is properly cleaned up after the adapter.evaluate() returns,
    even on successful execution.
    """
    # Count temp directories with our prefix before
    temp_dir = Path("/tmp")
    temp_dirs_before = set(temp_dir.glob("claude-agent-*"))

    isolation = IsolationConfig(
        network_policy=NetworkPolicy.no_network(),
    )

    config = _make_config(
        tmp_path,
        isolation=isolation,
    )

    adapter = _make_adapter(
        tmp_path,
        client_config=config,
    )

    prompt_template = _build_greeting_prompt()
    params = GreetingParams(audience="cleanup test")
    prompt = Prompt(prompt_template).bind(params)

    session = _make_session_with_usage_tracking()
    response = adapter.evaluate(prompt, session=session)

    assert response.text is not None

    # Count temp directories after - should be same as before
    # (cleanup should have removed the ephemeral home)
    temp_dirs_after = set(temp_dir.glob("claude-agent-*"))

    # Any new directories should have been cleaned up
    new_dirs = temp_dirs_after - temp_dirs_before
    assert len(new_dirs) == 0, (
        f"Expected ephemeral home to be cleaned up. Found orphaned dirs: {new_dirs}"
    )


def test_claude_agent_sdk_adapter_isolation_creates_files_in_ephemeral_home(
    tmp_path: Path,
) -> None:
    """Verify files are created in the ephemeral home directory during execution.

    This test validates that the isolation mechanism actually uses the ephemeral
    home directory by capturing the directory during execution and verifying
    that expected files (like .claude/settings.json) are created there.

    This complements the negative test (host not modified) with positive proof
    that the ephemeral home is being used.
    """
    import json
    from unittest.mock import patch

    from weakincentives.adapters.claude_agent_sdk.isolation import EphemeralHome

    # Capture state: {home_path: (env_home, settings_dict or None)}
    captured: dict[Path, tuple[str, dict[str, object] | None]] = {}
    original_cleanup = EphemeralHome.cleanup

    def capturing_cleanup(self: EphemeralHome) -> None:
        """Capture ephemeral home state before cleanup."""
        home_path = Path(self._temp_dir)
        if home_path not in captured:
            settings_path = home_path / ".claude" / "settings.json"
            env_home = self.get_env().get("HOME", "")
            settings = (
                json.loads(settings_path.read_text())
                if settings_path.exists()
                else None
            )
            captured[home_path] = (env_home, settings)
        original_cleanup(self)

    isolation = IsolationConfig(
        network_policy=NetworkPolicy.no_network(),
        sandbox=SandboxConfig(enabled=True),
    )
    config = _make_config(tmp_path, isolation=isolation)
    adapter = _make_adapter(tmp_path, client_config=config)
    prompt = Prompt(_build_greeting_prompt()).bind(
        GreetingParams(audience="ephemeral home verification")
    )
    session = _make_session_with_usage_tracking()

    with patch.object(EphemeralHome, "cleanup", capturing_cleanup):
        response = adapter.evaluate(prompt, session=session)

    assert response.text is not None
    _assert_prompt_usage(session)

    # Verify exactly one ephemeral home was created
    assert len(captured) == 1, f"Expected one ephemeral home, got {len(captured)}"
    ephemeral_home, (env_home, settings) = next(iter(captured.items()))

    # Verify HOME env var pointed to ephemeral directory
    assert env_home == str(ephemeral_home), f"HOME mismatch: {env_home}"

    # Verify settings.json was created with expected structure
    assert settings is not None, "Expected settings.json in ephemeral .claude directory"
    assert "sandbox" in settings, "Expected sandbox configuration in settings"
    sandbox = settings["sandbox"]

    # Print captured settings for verification
    print("\nCaptured settings.json sandbox configuration:")
    print(f"  enabled: {sandbox.get('enabled')}")
    print(f"  autoAllowBashIfSandboxed: {sandbox.get('autoAllowBashIfSandboxed')}")
    print(f"  allowUnsandboxedCommands: {sandbox.get('allowUnsandboxedCommands')}")

    assert isinstance(sandbox, dict) and sandbox.get("enabled") is True
    assert sandbox.get("autoAllowBashIfSandboxed") is True
    # Critical: allowUnsandboxedCommands must be explicitly False (bug 70a64c78)
    assert sandbox.get("allowUnsandboxedCommands") is False

    # Verify network policy was applied (no_network() means empty allowed domains)
    assert "network" in sandbox, "Expected network configuration in sandbox"
    network = sandbox["network"]
    assert isinstance(network, dict)
    # no_network() creates empty allowed domains - CLI reaches API outside sandbox
    assert network.get("allowedDomains") == []

    # Verify cleanup occurred
    assert not ephemeral_home.exists(), (
        f"Ephemeral home not cleaned up: {ephemeral_home}"
    )


@dataclass(slots=True)
class NetworkTestParams:
    """Parameters for network test prompt."""

    url: str


@dataclass(slots=True, frozen=True)
class NetworkTestResult:
    """Result of network connectivity test."""

    reachable: bool
    http_status: int | None
    error_message: str | None


def _build_network_test_prompt() -> PromptTemplate[NetworkTestResult]:
    """Build a prompt that tests network connectivity from within the sandbox."""
    return PromptTemplate[NetworkTestResult](
        ns=_PROMPT_NS,
        key="network-test",
        name="network_test",
        sections=(
            MarkdownSection[NetworkTestParams](
                title="Task",
                key="task",
                template=(
                    "Use the Bash tool to test network connectivity to ${url}. "
                    "Run: curl -s -o /dev/null -w '%{http_code}' --connect-timeout 5 ${url} 2>&1 "
                    "If successful, return reachable=true with the HTTP status code. "
                    "If it fails (connection refused, timeout, etc.), return reachable=false "
                    "with the error message. Set http_status to null if no response was received."
                ),
            ),
        ),
    )


def test_claude_agent_sdk_adapter_network_policy_allows_listed_domain(
    tmp_path: Path,
) -> None:
    """Verify network policy allows access to explicitly listed domains.

    This test validates that when a domain IS in allowed_domains, tools
    running in the sandbox can successfully reach it.
    """
    config = _make_config(
        tmp_path,
        isolation=IsolationConfig(
            network_policy=NetworkPolicy(
                allowed_domains=("api.anthropic.com", "example.com"),
            ),
            sandbox=SandboxConfig(enabled=True),
        ),
    )
    adapter = _make_adapter(
        tmp_path,
        client_config=config,
        allowed_tools=("Bash",),
    )

    prompt = Prompt(_build_network_test_prompt()).bind(
        NetworkTestParams(url="https://example.com")
    )
    session = _make_session_with_usage_tracking()

    response = adapter.evaluate(prompt, session=session)

    assert response.output is not None, (
        f"Expected structured output, got: {response.text}"
    )
    result = response.output
    print(
        f"Allowed domain test: reachable={result.reachable}, status={result.http_status}, error={result.error_message}"
    )
    # example.com is in allowed_domains, so it should be reachable with HTTP 200
    http_ok = 200
    assert result.reachable is True, (
        f"Expected example.com to be reachable: {result.error_message}"
    )
    assert result.http_status == http_ok


def test_claude_agent_sdk_adapter_network_policy_blocks_unlisted_domain(
    tmp_path: Path,
) -> None:
    """Verify network policy blocks access to domains not in the allowed list.

    This test validates that when a domain is NOT in allowed_domains, tools
    running in the sandbox cannot reach it.

    See: https://github.com/anthropic-experimental/sandbox-runtime
    """
    config = _make_config(
        tmp_path,
        isolation=IsolationConfig(
            network_policy=NetworkPolicy.no_network(),  # Block all tool network
            sandbox=SandboxConfig(enabled=True),
        ),
    )
    adapter = _make_adapter(
        tmp_path,
        client_config=config,
        allowed_tools=("Bash",),
    )

    prompt = Prompt(_build_network_test_prompt()).bind(
        NetworkTestParams(url="https://example.com")
    )
    session = _make_session_with_usage_tracking()

    response = adapter.evaluate(prompt, session=session)

    assert response.output is not None, (
        f"Expected structured output, got: {response.text}"
    )
    result = response.output
    print(
        f"Blocked domain test: reachable={result.reachable}, status={result.http_status}, error={result.error_message}"
    )

    # example.com should be blocked since it's not in allowed_domains
    assert result.reachable is False, (
        f"Expected example.com to be blocked, but got status {result.http_status}"
    )


def test_claude_agent_sdk_adapter_no_network_blocks_all_tool_access(
    tmp_path: Path,
) -> None:
    """Verify no_network() blocks all network access for tools.

    This test validates that NetworkPolicy.no_network() blocks tools from
    accessing any external network resources. The Claude Code CLI can still
    reach the API (it runs outside the tool sandbox), but tools like Bash
    cannot make network requests.
    """
    config = _make_config(
        tmp_path,
        isolation=IsolationConfig(
            network_policy=NetworkPolicy.no_network(),  # Block all tool network
            sandbox=SandboxConfig(enabled=True),
        ),
    )
    adapter = _make_adapter(
        tmp_path,
        client_config=config,
        allowed_tools=("Bash",),
    )

    prompt = Prompt(_build_network_test_prompt()).bind(
        NetworkTestParams(url="https://example.com")
    )
    session = _make_session_with_usage_tracking()

    response = adapter.evaluate(prompt, session=session)

    assert response.output is not None, (
        f"Expected structured output, got: {response.text}"
    )
    result = response.output
    print(
        f"No network test: reachable={result.reachable}, status={result.http_status}, error={result.error_message}"
    )

    # All network should be blocked for tools
    assert result.reachable is False, (
        f"Expected all network blocked, but got status {result.http_status}"
    )


# =============================================================================
# Additional Isolation Configuration Tests
# =============================================================================
#
# These tests cover additional IsolationConfig options to ensure each
# configuration option behaves as documented.
# =============================================================================


@dataclass(slots=True, frozen=True)
class EnvTestResult:
    """Result of environment variable test."""

    found_vars: list[str]
    path_value: str | None
    custom_var_value: str | None


def _build_env_test_prompt() -> PromptTemplate[EnvTestResult]:
    """Build a prompt that tests environment variable inheritance."""
    return PromptTemplate[EnvTestResult](
        ns=_PROMPT_NS,
        key="env-test",
        name="env_test",
        sections=(
            MarkdownSection[EmptyParams](
                title="Task",
                key="task",
                template=(
                    "Use the Bash tool to inspect environment variables. "
                    "Run: env | head -20 "
                    "Return found_vars with a list of variable names you see. "
                    "Return path_value with the value of PATH if present (or null). "
                    "Return custom_var_value with the value of WINK_CUSTOM_TEST_VAR "
                    "if present (or null)."
                ),
            ),
        ),
    )


def test_claude_agent_sdk_adapter_include_host_env_false(tmp_path: Path) -> None:
    """Verify include_host_env=False limits environment passed to SDK.

    This test validates that when include_host_env=False (the default),
    we only pass HOME, ANTHROPIC_API_KEY, and custom env vars to the SDK.

    NOTE: The actual subprocess environment may still include variables
    from the Claude Code CLI process itself. This test validates that
    custom env vars are correctly passed through our isolation layer.
    """
    config = _make_config(
        tmp_path,
        isolation=IsolationConfig(
            network_policy=NetworkPolicy.no_network(),
            include_host_env=False,  # Explicitly false
            env={"WINK_CUSTOM_TEST_VAR": "custom_value"},
        ),
    )
    adapter = _make_adapter(
        tmp_path,
        client_config=config,
        allowed_tools=("Bash",),
    )

    prompt = Prompt(_build_env_test_prompt()).bind(EmptyParams())
    session = _make_session_with_usage_tracking()

    response = adapter.evaluate(prompt, session=session)

    assert response.output is not None, (
        f"Expected structured output, got: {response.text}"
    )
    result = response.output
    print(
        f"Env test (include_host_env=False): found_vars={result.found_vars}, "
        f"path={result.path_value}, custom={result.custom_var_value}"
    )

    # The key validation: our custom env var is present
    assert result.custom_var_value == "custom_value", (
        f"Expected custom var to be 'custom_value', got {result.custom_var_value}"
    )


def test_claude_agent_sdk_adapter_include_host_env_true(tmp_path: Path) -> None:
    """Verify include_host_env=True inherits non-sensitive host environment.

    This test validates that when include_host_env=True, safe host
    environment variables like PATH are inherited, but sensitive ones
    (AWS_*, OPENAI_*, etc.) are filtered out.
    """
    import os as test_os

    config = _make_config(
        tmp_path,
        isolation=IsolationConfig(
            network_policy=NetworkPolicy.no_network(),
            include_host_env=True,  # Inherit host env
        ),
    )
    adapter = _make_adapter(
        tmp_path,
        client_config=config,
        allowed_tools=("Bash",),
    )

    prompt = Prompt(_build_env_test_prompt()).bind(EmptyParams())
    session = _make_session_with_usage_tracking()

    response = adapter.evaluate(prompt, session=session)

    assert response.output is not None, (
        f"Expected structured output, got: {response.text}"
    )
    result = response.output
    print(
        f"Env test (include_host_env=True): found_vars={result.found_vars}, "
        f"path={result.path_value}"
    )

    # PATH should be inherited from host
    host_path = test_os.environ.get("PATH")
    if host_path:
        assert result.path_value is not None, "Expected PATH to be inherited from host"

    # Sensitive variables should NOT be in found_vars.
    # Note: Some vars like AWS_REGION may still appear if they're set in the
    # user's shell profile (~/.bashrc, ~/.zshrc). The SDK spawns a bash subprocess
    # which sources these profiles, bypassing our env filtering. This is expected
    # behavior - we filter at the SDK options level, but can't control shell profiles.
    # We verify that the vars we EXPLICITLY filter (like AWS credentials) are not
    # passed through our isolation layer by checking vars NOT commonly in profiles.
    sensitive_vars_not_in_profiles = [
        "AWS_SECRET_ACCESS_KEY",
        "AWS_SESSION_TOKEN",
        "OPENAI_API_KEY",
        "GOOGLE_API_KEY",
        "AZURE_API_KEY",
    ]
    for var in result.found_vars:
        assert var not in sensitive_vars_not_in_profiles, (
            f"Sensitive credential {var} should be filtered by isolation"
        )


@dataclass(slots=True, frozen=True)
class FileWriteTestResult:
    """Result of file write test."""

    write_succeeded: bool
    file_path: str | None
    error_message: str | None


def _build_file_write_test_prompt() -> PromptTemplate[FileWriteTestResult]:
    """Build a prompt that tests file writing to a specific path."""
    return PromptTemplate[FileWriteTestResult](
        ns=_PROMPT_NS,
        key="file-write-test",
        name="file_write_test",
        sections=(
            MarkdownSection[ReadFileParams](
                title="Task",
                key="task",
                template=(
                    "Use the Bash tool to try writing a file to ${file_path}. "
                    "Run: echo 'test content' > ${file_path} 2>&1 && echo SUCCESS || echo FAILED "
                    "Return write_succeeded=true if the write succeeded, false otherwise. "
                    "Return the file_path you tried to write to. "
                    "Return any error message if the write failed."
                ),
            ),
        ),
    )


def test_claude_agent_sdk_adapter_sandbox_writable_paths(tmp_path: Path) -> None:
    """Verify writable_paths allows writing to extra directories.

    This test validates that SandboxConfig.writable_paths correctly
    allows writing to directories outside the default workspace.
    """
    import tempfile

    # Create a temp directory to use as a writable path
    with tempfile.TemporaryDirectory(prefix="wink-writable-") as writable_dir:
        test_file = f"{writable_dir}/test-writable.txt"

        config = _make_config(
            tmp_path,
            isolation=IsolationConfig(
                network_policy=NetworkPolicy.no_network(),
                sandbox=SandboxConfig(
                    enabled=True,
                    writable_paths=(writable_dir,),
                ),
            ),
        )
        adapter = _make_adapter(
            tmp_path,
            client_config=config,
            allowed_tools=("Bash",),
        )

        prompt = Prompt(_build_file_write_test_prompt()).bind(
            ReadFileParams(file_path=test_file)
        )
        session = _make_session_with_usage_tracking()

        response = adapter.evaluate(prompt, session=session)

        assert response.output is not None, (
            f"Expected structured output, got: {response.text}"
        )
        result = response.output
        print(
            f"Writable paths test: succeeded={result.write_succeeded}, "
            f"path={result.file_path}, error={result.error_message}"
        )

        # With writable_paths configured, write should succeed
        assert result.write_succeeded is True, (
            f"Expected write to succeed with writable_paths: {result.error_message}"
        )

        # Verify file actually exists
        assert Path(test_file).exists(), "Expected file to be created"


def _print_bundle_logs(bundle: object) -> None:
    """Print relevant logs from a debug bundle."""
    import json

    print("\nCaptured Python logs from debug bundle:")
    logs_content = bundle.logs  # type: ignore[attr-defined]
    if not logs_content:
        print("  (SDK runs as subprocess - Python logs not captured)")
        return

    for line in logs_content.strip().split("\n"):
        if not line:
            continue
        try:
            log_entry = json.loads(line)
            level = log_entry.get("level", "?")
            event = log_entry.get("event", "")
            message = log_entry.get("message", "")[:80]
            # Filter to show only relevant logs
            keywords = ["sandbox", "permission", "error", "fail"]
            if any(kw in (event + message).lower() for kw in keywords):
                print(f"  [{level}] {event}: {message}")
        except json.JSONDecodeError:
            print(f"  [RAW] {line[:100]}")


def test_claude_agent_sdk_adapter_sandbox_blocks_write_outside_writable_paths(
    tmp_path: Path,
) -> None:
    """Verify sandbox BLOCKS writes to paths NOT in writable_paths.

    This is the NEGATIVE counterpart to test_sandbox_writable_paths.
    It validates that the sandbox actually enforces write restrictions.
    """
    import tempfile

    from weakincentives.debug import BundleConfig, BundleWriter, DebugBundle

    with tempfile.TemporaryDirectory(prefix="wink-allowed-") as allowed_dir:
        blocked_file = f"/tmp/wink-sandbox-escape-test-{tmp_path.name}.txt"
        bundle_dir = tmp_path / "bundles"
        bundle_dir.mkdir()

        config = _make_config(
            tmp_path,
            isolation=IsolationConfig(
                network_policy=NetworkPolicy.no_network(),
                sandbox=SandboxConfig(
                    enabled=True,
                    writable_paths=(allowed_dir,),
                ),
            ),
        )
        adapter = _make_adapter(tmp_path, client_config=config, allowed_tools=("Bash",))
        prompt = Prompt(_build_file_write_test_prompt()).bind(
            ReadFileParams(file_path=blocked_file)
        )
        session = _make_session_with_usage_tracking()

        try:
            bundle_config = BundleConfig(target=bundle_dir)
            with BundleWriter(
                target=bundle_dir, config=bundle_config, trigger="sandbox_test"
            ) as writer:
                writer.set_prompt_info(
                    ns=prompt.ns, key=prompt.key, adapter="claude_agent_sdk"
                )
                with writer.capture_logs():
                    response = adapter.evaluate(prompt, session=session)
                writer.write_session_after(session)

            bundle = DebugBundle.load(writer.path)
            assert response.output is not None, (
                f"Expected structured output, got: {response.text}"
            )
            result = response.output

            # Verify sandbox blocked the write
            assert result.write_succeeded is False, (
                "Expected write to FAIL outside writable_paths, but it succeeded. "
                "This indicates sandbox is not enforcing write restrictions!"
            )
            assert not Path(blocked_file).exists(), (
                f"File {blocked_file} should NOT exist - sandbox escape detected!"
            )

            # Verify error message indicates sandbox enforcement
            error_msg = (result.error_message or "").lower()
            sandbox_indicators = [
                "permission denied",
                "read-only",
                "operation not permitted",
            ]
            assert any(ind in error_msg for ind in sandbox_indicators), (
                f"Expected sandbox-related error message, got: {result.error_message}"
            )

            # Print enforcement signals
            print(
                f"\nBlocked write: succeeded={result.write_succeeded}, "
                f"error={result.error_message}"
            )
            _print_bundle_logs(bundle)

        finally:
            if Path(blocked_file).exists():
                Path(blocked_file).unlink()


def _build_file_read_test_prompt() -> PromptTemplate[FileReadTestResult]:
    """Build a prompt that tests file reading from a specific path."""
    return PromptTemplate[FileReadTestResult](
        ns=_PROMPT_NS,
        key="file-read-test",
        name="file_read_test",
        sections=(
            MarkdownSection[ReadFileParams](
                title="Task",
                key="task",
                template=(
                    "Use the Read tool to read the file at ${file_path}. "
                    "Return read_succeeded=true if you can read it, false otherwise. "
                    "Return the content if readable, or the error message if not."
                ),
            ),
        ),
    )


def test_claude_agent_sdk_adapter_sandbox_readable_paths(tmp_path: Path) -> None:
    """Verify readable_paths allows reading extra directories.

    This test validates that SandboxConfig.readable_paths correctly
    allows reading from directories outside the default workspace.
    """
    import tempfile

    # Create a temp directory with a file to read
    with tempfile.TemporaryDirectory(prefix="wink-readable-") as readable_dir:
        test_file = f"{readable_dir}/test-readable.txt"
        Path(test_file).write_text("readable test content")

        config = _make_config(
            tmp_path,
            isolation=IsolationConfig(
                network_policy=NetworkPolicy.no_network(),
                sandbox=SandboxConfig(
                    enabled=True,
                    readable_paths=(readable_dir,),
                ),
            ),
        )
        adapter = _make_adapter(
            tmp_path,
            client_config=config,
            allowed_tools=("Read",),
        )

        prompt = Prompt(_build_file_read_test_prompt()).bind(
            ReadFileParams(file_path=test_file)
        )
        session = _make_session_with_usage_tracking()

        response = adapter.evaluate(prompt, session=session)

        assert response.output is not None, (
            f"Expected structured output, got: {response.text}"
        )
        result = response.output
        print(
            f"Readable paths test: succeeded={result.read_succeeded}, "
            f"content={result.content}, error={result.error_message}"
        )

        # With readable_paths configured, read should succeed
        assert result.read_succeeded is True, (
            f"Expected read to succeed with readable_paths: {result.error_message}"
        )
        assert result.content is not None
        assert "readable test content" in result.content


def test_claude_agent_sdk_adapter_sandbox_disabled(tmp_path: Path) -> None:
    """Verify sandbox can be disabled with enabled=False.

    This test validates that SandboxConfig(enabled=False) disables
    OS-level sandboxing, allowing full filesystem access.
    """
    config = _make_config(
        tmp_path,
        isolation=IsolationConfig(
            network_policy=NetworkPolicy.no_network(),
            sandbox=SandboxConfig(enabled=False),
        ),
    )
    adapter = _make_adapter(
        tmp_path,
        client_config=config,
        allowed_tools=("Read",),
    )

    # Try to read a system file that would normally be outside sandbox
    prompt = Prompt(_build_file_read_test_prompt()).bind(
        ReadFileParams(file_path="/etc/hosts")
    )
    session = _make_session_with_usage_tracking()

    response = adapter.evaluate(prompt, session=session)

    assert response.output is not None, (
        f"Expected structured output, got: {response.text}"
    )
    result = response.output
    print(
        f"Sandbox disabled test: succeeded={result.read_succeeded}, "
        f"error={result.error_message}"
    )

    # With sandbox disabled, /etc/hosts should be readable
    assert result.read_succeeded is True, (
        f"Expected /etc/hosts to be readable with sandbox disabled: {result.error_message}"
    )
    assert result.content is not None
    assert "localhost" in result.content.lower() or "127.0.0.1" in result.content


@dataclass(slots=True, frozen=True)
class CommandTestResult:
    """Result of command execution test."""

    command_succeeded: bool
    output: str | None
    error_message: str | None


def _build_command_test_prompt() -> PromptTemplate[CommandTestResult]:
    """Build a prompt that tests running a specific command."""
    return PromptTemplate[CommandTestResult](
        ns=_PROMPT_NS,
        key="command-test",
        name="command_test",
        sections=(
            MarkdownSection[EchoParams](
                title="Task",
                key="task",
                template=(
                    "Use the Bash tool to run the following command: ${text} "
                    "Return command_succeeded=true if it ran successfully. "
                    "Return the output if successful, or error_message if it failed."
                ),
            ),
        ),
    )


def test_claude_agent_sdk_adapter_sandbox_excluded_commands(tmp_path: Path) -> None:
    """Verify excluded_commands allows specific commands to bypass sandbox.

    This test validates that SandboxConfig.excluded_commands correctly
    allows specific commands to run outside the sandbox.
    """
    config = _make_config(
        tmp_path,
        isolation=IsolationConfig(
            network_policy=NetworkPolicy.no_network(),
            sandbox=SandboxConfig(
                enabled=True,
                excluded_commands=("ls",),
                allow_unsandboxed_commands=True,
            ),
        ),
    )
    adapter = _make_adapter(
        tmp_path,
        client_config=config,
        allowed_tools=("Bash",),
    )

    # Run ls which is in excluded_commands
    prompt = Prompt(_build_command_test_prompt()).bind(EchoParams(text="ls /"))
    session = _make_session_with_usage_tracking()

    response = adapter.evaluate(prompt, session=session)

    assert response.output is not None, (
        f"Expected structured output, got: {response.text}"
    )
    result = response.output
    print(
        f"Excluded commands test: succeeded={result.command_succeeded}, "
        f"output={result.output}, error={result.error_message}"
    )

    # Command in excluded_commands should succeed
    assert result.command_succeeded is True, (
        f"Expected excluded command to succeed: {result.error_message}"
    )
