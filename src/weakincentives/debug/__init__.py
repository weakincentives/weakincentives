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

"""Debugging utilities for capturing, inspecting, and reproducing execution state.

This package provides tools for creating debug bundles - self-contained zip archives
that capture everything needed to understand, reproduce, and debug an AgentLoop
execution. Debug bundles unify session state, logs, filesystem snapshots,
configuration, metrics, and environment information into a single portable artifact.

Overview
--------

The debug package offers three main capabilities:

1. **Bundle Creation** - Capture execution state during AgentLoop runs
2. **Bundle Inspection** - Load and analyze existing debug bundles
3. **Environment Capture** - Collect reproducibility information (system, Python,
   git state, packages, etc.)


Exports
-------

Bundle Classes
~~~~~~~~~~~~~~

``BundleConfig``
    Configuration for automatic bundle creation in AgentLoop. Set ``target`` to
    a directory path to enable bundling. Supports limits for file sizes and
    total capture size, plus compression options.

``BundleWriter``
    Context manager for streaming bundle creation. Use this for programmatic
    bundle creation outside of AgentLoop, or when you need fine-grained control
    over what gets captured.

``DebugBundle``
    Load and inspect existing debug bundles. Provides properties to access
    all captured artifacts (request input/output, session state, logs,
    configuration, metrics, environment, etc.).

``BundleManifest``
    Metadata structure embedded in every bundle. Contains format version,
    bundle ID, timestamps, request status, file list, and integrity checksums.

``BundleError``
    Base exception for bundle operations.

``BundleValidationError``
    Raised when bundle validation fails (missing files, checksum mismatch, etc.).

Environment Classes
~~~~~~~~~~~~~~~~~~~

``capture_environment``
    Function to capture the complete execution environment. Returns an
    ``EnvironmentCapture`` containing system, Python, packages, env vars,
    git state, and container info.

``EnvironmentCapture``
    Complete environment snapshot for reproducibility. Contains all the
    information needed to recreate the execution context.

``SystemInfo``
    System/OS information: OS name, release, kernel version, architecture,
    CPU count, total memory, and hostname.

``PythonInfo``
    Python runtime details: version, implementation (CPython, PyPy, etc.),
    executable path, virtual environment status.

``GitInfo``
    Git repository state: commit SHA, branch, dirty status, remotes, and tags.
    Credentials in remote URLs are automatically redacted.

``ContainerInfo``
    Container runtime information: runtime type (docker, containerd, kubernetes),
    container ID, image name, and cgroup path.

``CommandInfo``
    Command invocation details: argv, working directory, entrypoint, and
    Python executable.


Debug Bundle Format
-------------------

A debug bundle is a zip archive with the following structure::

    debug_bundle/
        manifest.json           # Bundle metadata and integrity checksums
        README.txt              # Human-readable navigation guide

        request/
            input.json          # AgentLoop request
            output.json         # AgentLoop response

        session/
            before.jsonl        # Session state before execution
            after.jsonl         # Session state after execution

        logs/
            app.jsonl           # Log records captured during execution

        environment/            # Reproducibility envelope
            system.json         # OS, kernel, arch, CPU, memory
            python.json         # Python version, executable, venv info
            packages.txt        # Installed packages (pip freeze)
            env_vars.json       # Environment variables (filtered/redacted)
            git.json            # Repo root, commit, branch, remotes
            git.diff            # Uncommitted changes (if any)
            command.txt         # argv, working dir, entrypoint
            container.json      # Container runtime info (if applicable)

        config.json             # AgentLoop and adapter configuration
        run_context.json        # Execution context (IDs, tracing)
        metrics.json            # Token usage, timing, budget state

        # Optional files:
        prompt_overrides.json   # Visibility overrides
        error.json              # Error details (if execution failed)
        eval.json               # Eval metadata (EvalLoop only)
        filesystem/             # Workspace snapshot (if captured)


Usage Examples
--------------

AgentLoop Integration
~~~~~~~~~~~~~~~~~~~~~

Enable automatic debug bundle creation for all AgentLoop executions::

    from weakincentives.debug import BundleConfig
    from weakincentives.runtime import AgentLoop, AgentLoopConfig

    config = AgentLoopConfig(
        debug_bundle=BundleConfig(
            target="./debug_bundles/",
            max_file_size=10_000_000,   # Skip files > 10MB
            max_total_size=52_428_800,  # Cap total capture at 50MB
            compression="deflate",       # ZIP compression method
        ),
    )
    loop = MyLoop(adapter=adapter, requests=requests, config=config)
    # Bundles are created automatically for each request

Manual Bundle Creation
~~~~~~~~~~~~~~~~~~~~~~

Create debug bundles programmatically with fine-grained control::

    from uuid import uuid4
    from weakincentives.debug import BundleWriter, capture_environment

    run_id = uuid4()
    with BundleWriter(target="./debug/", bundle_id=run_id) as writer:
        # Capture environment early
        writer.write_environment()

        # Capture session state before execution
        writer.write_session_before(session)
        writer.write_request_input(request)
        writer.write_config(config)

        # Capture logs during execution
        with writer.capture_logs():
            response = adapter.evaluate(prompt, session=session)

        # Capture results
        writer.write_session_after(session)
        writer.write_request_output(response)
        writer.write_metrics(metrics)

    # Bundle finalized on exit: README generated, checksums computed, zip created
    print(f"Bundle created at: {writer.path}")

Inspecting Existing Bundles
~~~~~~~~~~~~~~~~~~~~~~~~~~~

Load and analyze debug bundles for debugging::

    from weakincentives.debug import DebugBundle

    # Load a bundle
    bundle = DebugBundle.load("./debug_bundles/abc123_20240115_103000.zip")

    # Check manifest metadata
    print(f"Bundle ID: {bundle.manifest.bundle_id}")
    print(f"Created: {bundle.manifest.created_at}")
    print(f"Status: {bundle.manifest.request.status}")

    # Access captured artifacts
    print(f"Request: {bundle.request_input}")
    print(f"Response: {bundle.request_output}")

    # Check session state
    if bundle.session_after:
        print(f"Session after: {bundle.session_after}")

    # Review logs
    if bundle.logs:
        for line in bundle.logs.splitlines():
            print(line)

    # Check environment
    if bundle.environment:
        env = bundle.environment
        print(f"OS: {env['system']}")
        print(f"Python: {env['python']}")
        print(f"Git commit: {env['git']}")

    # Verify integrity
    if bundle.verify_integrity():
        print("Bundle integrity verified")
    else:
        print("WARNING: Bundle may be corrupted")

    # Extract for detailed analysis
    extracted = bundle.extract("./extracted/")
    print(f"Extracted to: {extracted}")

    # List all files in bundle
    for filename in bundle.list_files():
        print(f"  {filename}")

Capturing Environment Separately
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Capture environment information independently of bundle creation::

    from weakincentives.debug import capture_environment

    # Full capture (may take a few seconds due to package listing)
    env = capture_environment()
    print(f"OS: {env.system.os_name} {env.system.os_release}")
    print(f"Python: {env.python.version}")
    print(f"Virtual env: {env.python.is_virtualenv}")

    if env.git:
        print(f"Git commit: {env.git.commit_sha}")
        print(f"Branch: {env.git.branch}")
        print(f"Dirty: {env.git.is_dirty}")

    if env.container:
        print(f"Container runtime: {env.container.runtime}")
        print(f"Container ID: {env.container.container_id}")

    # Fast capture (skip slow operations)
    env_fast = capture_environment(
        include_packages=False,  # Skip pip freeze
        include_git_diff=False,  # Skip git diff
    )


Notes
-----

- Bundles are created atomically: either the full bundle is written or nothing
  is written. This prevents partial/corrupted bundles.

- Sensitive information is automatically redacted:
    - Environment variables matching patterns like ``*SECRET*``, ``*TOKEN*``,
      ``*PASSWORD*``, ``*API_KEY*`` are redacted
    - Git remote URLs with credentials are redacted
    - Sensitive files (``*.pem``, ``*.key``, ``.env``, etc.) are excluded from
      untracked file capture

- The ``filesystem/`` directory in bundles contains a snapshot of the workspace.
  Large files are automatically skipped based on ``BundleConfig`` limits.

- Log capture uses a custom handler that converts log records to JSONL format,
  preserving structured logging context.

- Bundle integrity is verified using SHA-256 checksums stored in the manifest.
"""

from __future__ import annotations

from .bundle import (
    BundleConfig,
    BundleError,
    BundleManifest,
    BundleValidationError,
    BundleWriter,
    DebugBundle,
)
from .environment import (
    CommandInfo,
    ContainerInfo,
    EnvironmentCapture,
    GitInfo,
    PythonInfo,
    SystemInfo,
    capture_environment,
)

__all__ = [
    "BundleConfig",
    "BundleError",
    "BundleManifest",
    "BundleValidationError",
    "BundleWriter",
    "CommandInfo",
    "ContainerInfo",
    "DebugBundle",
    "EnvironmentCapture",
    "GitInfo",
    "PythonInfo",
    "SystemInfo",
    "capture_environment",
]
