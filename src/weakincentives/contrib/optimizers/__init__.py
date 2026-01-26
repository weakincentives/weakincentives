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

"""Prompt optimizers for workspace analysis and digest generation.

This package provides specialized optimizers that analyze workspaces and
generate task-agnostic summaries for efficient prompt caching. These
optimizers work with the contributed tool suites (VFS, asteval, planning)
to explore codebases and extract relevant information.

Public Exports
--------------

WorkspaceDigestOptimizer
    Generates workspace digests by composing an internal LLM prompt that
    explores the mounted workspace. The resulting digest captures:

    - Build and test commands
    - Dependency managers and package configurations
    - Project structure and key entry points
    - Environment capabilities and versions
    - Important conventions and watchouts

    The digest is task-agnostic and can be cached for reuse across multiple
    prompt evaluations, reducing redundant workspace exploration.

    Persistence scopes:
        - ``SESSION``: Digest stored in session state (default)
        - ``GLOBAL``: Digest stored in the override store for cross-session reuse

Architecture
------------

The optimizer follows a composable design:

1. **Requirement detection**: Finds ``WorkspaceDigestSection`` and
   ``WorkspaceSection`` in the target prompt
2. **Tool assembly**: Clones planning, VFS, and asteval sections for
   internal exploration
3. **LLM exploration**: Runs an optimization prompt that uses tools to
   analyze README, docs, configs, and source structure
4. **Result extraction**: Parses structured output containing summary
   and full digest
5. **Persistence**: Stores result in session or global override store

The optimizer shares filesystem state between cloned sections to ensure
asteval can read files from the VFS workspace.

Configuration
-------------

Use ``OptimizerConfig`` to control optimizer behavior::

    from weakincentives.optimizers.base import OptimizerConfig

    config = OptimizerConfig(
        accepts_overrides=True,  # Allow prompt override integration
    )
    optimizer = WorkspaceDigestOptimizer(context, config=config)

Result Types
------------

The optimizer returns ``WorkspaceDigestResult`` containing:

- ``response``: The full prompt response from the optimization run
- ``digest``: The extracted digest content
- ``scope``: Where the digest was persisted (SESSION or GLOBAL)
- ``section_key``: Key of the target digest section

Example Usage
-------------

Basic session-scoped optimization::

    from weakincentives.contrib.optimizers import WorkspaceDigestOptimizer
    from weakincentives.optimizers.context import OptimizationContext
    from weakincentives.optimizers.results import PersistenceScope

    # Create optimization context with adapter and deadline
    context = OptimizationContext(
        adapter=my_adapter,
        deadline=datetime.now(UTC) + timedelta(minutes=5),
    )

    # Create optimizer (SESSION scope by default)
    optimizer = WorkspaceDigestOptimizer(context)

    # Run optimization - digest is stored in session
    result = optimizer.optimize(prompt, session=session)

    print(f"Generated digest ({len(result.digest)} chars)")
    print(f"Stored in: {result.scope}")

Global persistence for cross-session reuse::

    # Configure optimizer for global persistence
    optimizer = WorkspaceDigestOptimizer(
        context,
        store_scope=PersistenceScope.GLOBAL,
    )

    # Prompt must have overrides_store configured
    prompt = Prompt(
        template,
        overrides_store=my_override_store,
        overrides_tag="v1",
    )

    # Digest is stored in override store, survives session restarts
    result = optimizer.optimize(prompt, session=session)

Integration with prompt templates::

    from weakincentives.contrib.tools import (
        WorkspaceDigestSection,
        VfsToolsSection,
    )

    # Template must include both workspace and digest sections
    template = PromptTemplate(
        ns="myapp",
        key="coding-assistant",
        sections=(
            VfsToolsSection(session=session, mounts=mounts),
            WorkspaceDigestSection(session=session),
            # ... other sections
        ),
    )

    # Optimizer finds WorkspaceDigestSection and populates it
    prompt = Prompt(template)
    optimizer.optimize(prompt, session=session)

    # Future renders of this prompt use the cached digest

See Also
--------

- ``weakincentives.optimizers.base``: Base optimizer classes
- ``weakincentives.contrib.tools.digests``: Digest section and helpers
- ``weakincentives.prompt.overrides``: Override store for global persistence
"""

from __future__ import annotations

from .workspace_digest import WorkspaceDigestOptimizer

__all__ = ["WorkspaceDigestOptimizer"]
