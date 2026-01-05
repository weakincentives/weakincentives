#!/usr/bin/env python3
"""Comprehensive auth debugging script for Claude Agent SDK integration tests.

Run with: uv run python auth_debug_script.py
"""

from __future__ import annotations

import asyncio
import json
import os
import shutil
import signal
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

TIMEOUT_SECONDS = 30


def section(title: str) -> None:
    print(f"\n{'=' * 60}")
    print(f" {title}")
    print("=" * 60)


def subsection(title: str) -> None:
    print(f"\n--- {title} ---")


# =============================================================================
# 1. Environment Variable Analysis
# =============================================================================


def check_environment() -> dict[str, str | None]:
    section("1. ENVIRONMENT VARIABLES")

    sensitive_prefixes = (
        "ANTHROPIC_",
        "CLAUDE_",
        "AWS_",
        "GOOGLE_",
        "AZURE_",
        "OPENAI_",
        "HOME",
        "XDG_",
    )

    results: dict[str, str | None] = {}

    subsection("Relevant environment variables")
    for key, value in sorted(os.environ.items()):
        if any(key.startswith(p) for p in sensitive_prefixes):
            # Mask sensitive values
            if "KEY" in key or "SECRET" in key or "TOKEN" in key or "PASSWORD" in key:
                masked = value[:8] + "..." + value[-4:] if len(value) > 12 else "***"
                print(f"  {key}={masked}")
            else:
                print(f"  {key}={value}")
            results[key] = value

    subsection("Critical checks")
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if api_key:
        print(f"  ANTHROPIC_API_KEY: SET (length={len(api_key)}, prefix={api_key[:7]}...)")
        if api_key.startswith("sk-ant-"):
            print("    -> Looks like a valid Anthropic API key format")
        else:
            print("    -> WARNING: Does not start with 'sk-ant-', may be invalid")
    else:
        print("  ANTHROPIC_API_KEY: NOT SET")

    # Check for AWS credentials that might interfere
    aws_vars = [k for k in os.environ if k.startswith("AWS_")]
    if aws_vars:
        print(f"  AWS credentials detected: {aws_vars}")
        print("    -> These may indicate Bedrock configuration")

    return results


# =============================================================================
# 2. Claude Configuration Analysis
# =============================================================================


def check_claude_config() -> dict[str, Any]:
    section("2. CLAUDE CONFIGURATION (~/.claude)")

    home = Path.home()
    claude_dir = home / ".claude"
    results: dict[str, Any] = {"exists": claude_dir.exists()}

    if not claude_dir.exists():
        print(f"  {claude_dir} does not exist")
        return results

    print(f"  {claude_dir} exists")

    subsection("Directory contents")
    try:
        for item in sorted(claude_dir.iterdir()):
            stat = item.stat()
            print(f"    {item.name} ({stat.st_size} bytes)")
            results[item.name] = {"size": stat.st_size}
    except Exception as e:
        print(f"    Error listing: {e}")

    subsection("settings.json analysis")
    settings_path = claude_dir / "settings.json"
    if settings_path.exists():
        try:
            settings = json.loads(settings_path.read_text())
            results["settings"] = settings
            print(f"    Content: {json.dumps(settings, indent=4)}")

            # Check for provider configuration
            if "provider" in settings:
                print(f"    -> Provider configured: {settings['provider']}")
            if "apiProvider" in settings:
                print(f"    -> API Provider: {settings['apiProvider']}")
            if "bedrock" in str(settings).lower():
                print("    -> WARNING: Bedrock configuration detected!")
            if "model" in settings:
                print(f"    -> Model: {settings['model']}")

        except Exception as e:
            print(f"    Error reading settings.json: {e}")
    else:
        print("    settings.json does not exist")

    subsection("credentials.json analysis")
    creds_path = claude_dir / "credentials.json"
    if creds_path.exists():
        try:
            creds = json.loads(creds_path.read_text())
            # Don't print actual credentials, just structure
            print(f"    Keys present: {list(creds.keys())}")
            results["credentials_keys"] = list(creds.keys())
        except Exception as e:
            print(f"    Error reading: {e}")
    else:
        print("    credentials.json does not exist")

    return results


# =============================================================================
# 3. Claude Agent SDK Import Check
# =============================================================================


def check_sdk_import() -> dict[str, Any]:
    section("3. CLAUDE AGENT SDK IMPORT")

    results: dict[str, Any] = {}

    try:
        import claude_agent_sdk

        results["installed"] = True
        print(f"  Module: {claude_agent_sdk}")
        print(f"  File: {claude_agent_sdk.__file__}")

        if hasattr(claude_agent_sdk, "__version__"):
            print(f"  Version: {claude_agent_sdk.__version__}")
            results["version"] = claude_agent_sdk.__version__

        # Check available exports
        exports = [x for x in dir(claude_agent_sdk) if not x.startswith("_")]
        print(f"  Exports: {exports}")
        results["exports"] = exports

    except ImportError as e:
        results["installed"] = False
        print(f"  NOT INSTALLED: {e}")

    return results


# =============================================================================
# 4. Isolation Environment Check
# =============================================================================


def check_isolation_env() -> dict[str, Any]:
    section("4. ISOLATION ENVIRONMENT SIMULATION")

    results: dict[str, Any] = {}

    # Simulate what EphemeralHome.get_env() does
    sensitive_prefixes = (
        "HOME",
        "CLAUDE_",
        "ANTHROPIC_",
        "AWS_",
        "GOOGLE_",
        "AZURE_",
        "OPENAI_",
    )

    # Create ephemeral home
    temp_dir = tempfile.mkdtemp(prefix="claude-debug-")
    claude_dir = Path(temp_dir) / ".claude"
    claude_dir.mkdir(parents=True)

    # Generate minimal settings
    settings = {
        "sandbox": {
            "enabled": True,
            "autoAllowBashIfSandboxed": True,
        }
    }
    (claude_dir / "settings.json").write_text(json.dumps(settings, indent=2))

    subsection("Ephemeral home created")
    print(f"  Path: {temp_dir}")
    print(f"  Settings: {settings}")

    # Build isolated env
    isolated_env: dict[str, str] = {}

    # Don't inherit sensitive vars
    for k, v in os.environ.items():
        if not any(k.startswith(p) for p in sensitive_prefixes):
            isolated_env[k] = v

    # Override HOME
    isolated_env["HOME"] = temp_dir

    # Set API key
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if api_key:
        isolated_env["ANTHROPIC_API_KEY"] = api_key

    subsection("Isolated environment")
    print(f"  HOME: {isolated_env.get('HOME')}")
    print(f"  ANTHROPIC_API_KEY: {'SET' if 'ANTHROPIC_API_KEY' in isolated_env else 'NOT SET'}")

    # Check what's NOT in isolated env
    excluded = [k for k in os.environ if k not in isolated_env]
    print(f"  Excluded vars: {excluded}")

    results["temp_dir"] = temp_dir
    results["isolated_env"] = {k: v for k, v in isolated_env.items() if k in ["HOME", "PATH"]}
    results["excluded"] = excluded

    # Cleanup
    shutil.rmtree(temp_dir, ignore_errors=True)

    return results


# =============================================================================
# 5. Claude CLI Check
# =============================================================================


def check_claude_cli() -> dict[str, Any]:
    section("5. CLAUDE CLI CHECK")

    results: dict[str, Any] = {}

    # Find claude binary
    claude_path = shutil.which("claude")
    if claude_path:
        print(f"  Claude CLI found: {claude_path}")
        results["path"] = claude_path

        # Check version
        try:
            result = subprocess.run(
                [claude_path, "--version"],
                capture_output=True,
                text=True,
                timeout=10,
            )
            print(f"  Version output: {result.stdout.strip()}")
            if result.stderr:
                print(f"  Stderr: {result.stderr.strip()}")
            results["version"] = result.stdout.strip()
        except subprocess.TimeoutExpired:
            print("  WARNING: --version timed out!")
            results["version_timeout"] = True
        except Exception as e:
            print(f"  Error: {e}")
    else:
        print("  Claude CLI not found in PATH")
        results["path"] = None

    return results


# =============================================================================
# 6. Minimal SDK Query Test
# =============================================================================


async def test_minimal_sdk_query() -> dict[str, Any]:
    section("6. MINIMAL SDK QUERY TEST")

    results: dict[str, Any] = {}

    try:
        import claude_agent_sdk
        from claude_agent_sdk.types import ClaudeAgentOptions
    except ImportError as e:
        print(f"  Cannot import SDK: {e}")
        results["error"] = str(e)
        return results

    # Create ephemeral home
    temp_dir = tempfile.mkdtemp(prefix="claude-debug-query-")
    claude_dir = Path(temp_dir) / ".claude"
    claude_dir.mkdir(parents=True)
    work_dir = Path(temp_dir) / "workspace"
    work_dir.mkdir(parents=True)

    settings = {
        "sandbox": {
            "enabled": True,
            "autoAllowBashIfSandboxed": True,
            "network": {"allowedDomains": []},
        }
    }
    (claude_dir / "settings.json").write_text(json.dumps(settings, indent=2))

    # Build isolated env
    sensitive_prefixes = ("HOME", "CLAUDE_", "ANTHROPIC_", "AWS_", "GOOGLE_", "AZURE_", "OPENAI_")
    isolated_env: dict[str, str] = {
        k: v for k, v in os.environ.items() if not any(k.startswith(p) for p in sensitive_prefixes)
    }
    isolated_env["HOME"] = temp_dir
    if os.environ.get("ANTHROPIC_API_KEY"):
        isolated_env["ANTHROPIC_API_KEY"] = os.environ["ANTHROPIC_API_KEY"]

    print(f"  Ephemeral HOME: {temp_dir}")
    print(f"  Work dir: {work_dir}")
    print(f"  API key in env: {'YES' if 'ANTHROPIC_API_KEY' in isolated_env else 'NO'}")

    subsection("SDK Options")
    options_kwargs: dict[str, Any] = {
        "model": "claude-sonnet-4-20250514",
        "cwd": str(work_dir),
        "permission_mode": "bypassPermissions",
        "env": isolated_env,
        "setting_sources": ["user"],  # Load from ephemeral HOME
        "max_turns": 1,
    }
    print(f"  Options: {json.dumps({k: v for k, v in options_kwargs.items() if k != 'env'}, indent=4)}")

    options = ClaudeAgentOptions(**options_kwargs)

    subsection("Attempting SDK query (timeout: {TIMEOUT_SECONDS}s)")
    print("  Prompt: 'Say hello in exactly 3 words'")

    async def stream_prompt():
        yield {
            "type": "user",
            "message": {"role": "user", "content": "Say hello in exactly 3 words"},
            "parent_tool_use_id": None,
            "session_id": "debug-test",
        }

    start_time = time.time()
    messages_received = 0
    last_message_type = None

    try:

        async def run_with_timeout():
            nonlocal messages_received, last_message_type
            async for message in claude_agent_sdk.query(prompt=stream_prompt(), options=options):
                messages_received += 1
                last_message_type = type(message).__name__
                elapsed = time.time() - start_time
                print(f"    [{elapsed:.1f}s] Message {messages_received}: {last_message_type}")

                # Print some details
                if hasattr(message, "result"):
                    print(f"           Result: {message.result[:100] if message.result else None}...")
                if hasattr(message, "usage"):
                    print(f"           Usage: {message.usage}")

        await asyncio.wait_for(run_with_timeout(), timeout=TIMEOUT_SECONDS)

        elapsed = time.time() - start_time
        print(f"\n  SUCCESS: Received {messages_received} messages in {elapsed:.1f}s")
        results["success"] = True
        results["messages"] = messages_received
        results["elapsed"] = elapsed

    except asyncio.TimeoutError:
        elapsed = time.time() - start_time
        print(f"\n  TIMEOUT after {elapsed:.1f}s")
        print(f"  Messages received before timeout: {messages_received}")
        print(f"  Last message type: {last_message_type}")
        results["timeout"] = True
        results["messages_before_timeout"] = messages_received
        results["last_message_type"] = last_message_type

    except Exception as e:
        elapsed = time.time() - start_time
        print(f"\n  ERROR after {elapsed:.1f}s: {type(e).__name__}: {e}")
        results["error"] = str(e)
        results["error_type"] = type(e).__name__

    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)

    return results


# =============================================================================
# 7. Direct API Test (bypass SDK)
# =============================================================================


def test_direct_api() -> dict[str, Any]:
    section("7. DIRECT ANTHROPIC API TEST (bypass SDK)")

    results: dict[str, Any] = {}

    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        print("  ANTHROPIC_API_KEY not set, skipping")
        results["skipped"] = True
        return results

    try:
        import anthropic

        print(f"  Using anthropic library: {anthropic.__version__}")
    except ImportError:
        print("  anthropic library not installed, using requests")
        import requests

        subsection("Testing with requests")
        try:
            response = requests.post(
                "https://api.anthropic.com/v1/messages",
                headers={
                    "x-api-key": api_key,
                    "anthropic-version": "2023-06-01",
                    "content-type": "application/json",
                },
                json={
                    "model": "claude-sonnet-4-20250514",
                    "max_tokens": 50,
                    "messages": [{"role": "user", "content": "Say hello"}],
                },
                timeout=30,
            )
            print(f"  Status: {response.status_code}")
            if response.status_code == 200:
                data = response.json()
                print(f"  Response: {data.get('content', [{}])[0].get('text', '')[:100]}")
                results["success"] = True
            else:
                print(f"  Error: {response.text[:200]}")
                results["error"] = response.text[:200]
        except Exception as e:
            print(f"  Error: {e}")
            results["error"] = str(e)
        return results

    subsection("Testing with anthropic library")
    try:
        client = anthropic.Anthropic(api_key=api_key)
        response = client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=50,
            messages=[{"role": "user", "content": "Say hello"}],
        )
        print(f"  Response: {response.content[0].text[:100]}")
        results["success"] = True
    except Exception as e:
        print(f"  Error: {type(e).__name__}: {e}")
        results["error"] = str(e)
        results["error_type"] = type(e).__name__

    return results


# =============================================================================
# 8. Process Tree Check
# =============================================================================


def check_claude_processes() -> dict[str, Any]:
    section("8. CLAUDE PROCESSES CHECK")

    results: dict[str, Any] = {}

    try:
        # Check for any claude-related processes
        result = subprocess.run(
            ["ps", "aux"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        lines = [line for line in result.stdout.split("\n") if "claude" in line.lower()]
        if lines:
            print("  Claude-related processes found:")
            for line in lines:
                print(f"    {line[:120]}")
            results["processes"] = lines
        else:
            print("  No claude-related processes found")
            results["processes"] = []
    except Exception as e:
        print(f"  Error checking processes: {e}")

    return results


# =============================================================================
# 9. Network Connectivity Check
# =============================================================================


def check_network() -> dict[str, Any]:
    section("9. NETWORK CONNECTIVITY")

    results: dict[str, Any] = {}

    endpoints = [
        ("api.anthropic.com", 443),
        ("claude.ai", 443),
    ]

    import socket

    for host, port in endpoints:
        subsection(f"{host}:{port}")
        try:
            sock = socket.create_connection((host, port), timeout=10)
            sock.close()
            print(f"  Connected successfully")
            results[host] = "ok"
        except Exception as e:
            print(f"  Failed: {e}")
            results[host] = str(e)

    return results


# =============================================================================
# Main
# =============================================================================


def main() -> None:
    print("=" * 60)
    print(" CLAUDE AGENT SDK AUTH DEBUG SCRIPT")
    print(" " + time.strftime("%Y-%m-%d %H:%M:%S"))
    print("=" * 60)

    all_results: dict[str, Any] = {}

    # Run all checks
    all_results["environment"] = check_environment()
    all_results["claude_config"] = check_claude_config()
    all_results["sdk_import"] = check_sdk_import()
    all_results["isolation_env"] = check_isolation_env()
    all_results["claude_cli"] = check_claude_cli()
    all_results["network"] = check_network()
    all_results["direct_api"] = test_direct_api()
    all_results["processes"] = check_claude_processes()

    # Run async SDK test
    all_results["sdk_query"] = asyncio.run(test_minimal_sdk_query())

    # Summary
    section("SUMMARY")

    if all_results.get("direct_api", {}).get("success"):
        print("  Direct API test: PASSED")
    else:
        print("  Direct API test: FAILED")

    if all_results.get("sdk_query", {}).get("success"):
        print("  SDK query test: PASSED")
    elif all_results.get("sdk_query", {}).get("timeout"):
        print("  SDK query test: TIMEOUT")
        print("    -> This indicates the SDK is hanging during execution")
        print("    -> Likely causes:")
        print("       1. SDK is trying to use a different auth provider")
        print("       2. SDK subprocess is waiting for interactive input")
        print("       3. Network/firewall issue")
    else:
        print(f"  SDK query test: ERROR - {all_results.get('sdk_query', {}).get('error')}")

    # Write full results to file
    results_file = Path("auth_debug_results.json")
    results_file.write_text(json.dumps(all_results, indent=2, default=str))
    print(f"\n  Full results written to: {results_file}")


if __name__ == "__main__":
    main()
